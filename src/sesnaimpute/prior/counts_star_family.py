"""The per-source counts and normalisers for STAR, AGB, PAHC and GAL
(SPEC_PRIORS.md section 0.2's ``N_C(s)``/``lambda~_C``; section 1.3's
per-source selection recipe; section 2.1's ``N_STAR``; section 3's
``N_AGB``; section 4's ``N_PAHC``; section 5.1's ``N_GAL``).

Per source this module answers, for each of the four classes, "how many
catalogued objects of this class per deg**2 sit at this source's own
position, column and depth" (``N_C``) and "what fraction of the class's
intrinsic mass survives selection here" (``Z_C``, the normaliser the
fitter's shape divides by, spec section 0.2). YSO and H2S are built
elsewhere; this module writes only the four named here.

Owner ruling 2026-09-06: ``N_C`` and ``Z_C`` are two different
computations, not one identity read twice. ``N_C`` (STAR, AGB, PAHC) is
``direct_family_counts``: the population sum spec section 2.1/4 write --
every one of the tile's simulated stars, its own class weight, its own
extinction under a true column drawn from the kernel mixture (an 8-point
Gauss-Hermite quadrature per component), its own detection read off the
source's exact selection array. ``Z_C`` stays ``family_counts``'s dot
product of the smoothed, gridded shape against the selection -- the
fitter's own pointwise density needs a normaliser built from the SAME
tabulation it reads, not from the population. The two differ by exactly
the shape's smoothing (measured on NGC 7129: STAR +2.9%, PAHC +21.5%,
AGB negligible where selection barely bites) -- a kernel-density
bandwidth chosen for the shape has no business deciding the count.

STAR, AGB, PAHC share one construction for ``Z_C``. Each class's per-tile shape
(``prior.star_shapes.ClassShape``) already blends its own two bracketing
shape nodes (and, for PAHC, its own two bracketing limit grids) in the
source's own column and 8 micron limit -- ``ClassShape.density(a,
log10_b, tile_id, a_col[, f_lim8])`` evaluates the shape at an arbitrary
query point per source. The class's own exact per-source selection
(``prior.star_selection``'s ``EPS_STAR``/``EPS_PAHC``/``EPS_AGB``, one
small array per source on the shared ladder ``X_LADDER`` by the class's
own brightness grid) is read directly, per source: no depth grouping,
no common-mode shift, no shared model across sources.

The selected fraction

    Z_C(s) = Sum_{x,b in shape's own grid} shape(a = A_s . x, log10 B | s) . eps_s(x, b)

is evaluated on the shape's OWN tabulation grid (its ``xi_centers`` /
``b_centers``, the resolution its density estimate actually carries):
because the grid point's physical extinction is built as ``A_s . x``
BY CONSTRUCTION, the ladder coordinate at that grid point is exactly the
grid's own ``x``, independent of the source -- so the bilinear weights
that interpolate the source's small ``eps_s`` array onto the shape's
grid are the SAME two matrices (one per axis) for every source, computed
once per class and reused. Per batch of sources (``sesnaimpute.batches.
batches``) this collapses to one small matrix product for the eps
interpolation and one vectorised call to ``ClassShape.density`` over the
batch's own flattened ``(source, x, b)`` grid.

GAL (spec section 5.1) needs no shape: a galaxy's ``a`` is the true
column, spread around the source's own adopted column ``A_s`` by the
same kernel mixture (``prior.kernel.Kernel``) every other class reads.
Its selection is evaluated on the fly, per batch of sources, at every
``X_LADDER`` point (``prior.gal.source_selection_from_cdf`` against the
survey-wide colour-CDF tables -- there is no per-source galaxy product,
owner 2026-09-06), the survey counts law integrated out first
(``numpy.trapz`` over ``LOG10_S_GRID``, normalised to one), then the
ladder points combined by the kernel's own exact probability mass in
each ladder cell (``Kernel.cdf``, no bicubic, no Monte Carlo).

Writes, per region, ``bms/table/counts-star-family_table_source.hdf5``:
root attr ``GRANULE = "source"``; ``TILE_ID``, ``NODE_LO``, ``NODE_W``
(``column_grid.bracket`` on ``A_COL_K`` against the shared column-kernel
grid, kept for the fitter's own kernel read), ``F_LIM8_MJY``, and the
four counts and four normalisers, catalogue row order, float32.
"""

import os

import h5py
import numba
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid, pahc_curve, selection, star_population, star_shapes
from sesnaimpute.prior import star_selection as star_selection_module
from sesnaimpute.prior import gal as gal_module
from sesnaimpute import progress

FAMILY_CLASSES = ("star", "agb", "pahc")

#: The per-batch working-array budget for the (source, shape-x, shape-b)
#: grid expansion, the module's own largest intermediate.
BATCH_BUDGET_BYTES = 512 << 20

#: The fixed 8-point quadrature the direct count (below) uses for the
#: expectation over the true column `T` inside one mixture component
#: (owner ruling 2026-09-06: "a small fixed quadrature per component, 8
#: points"). Physicists' Gauss-Hermite nodes/weights, converted once to a
#: standard-normal `z` and a weight that sums to one: `E[f(Z)] ~= Sum_q
#: GH_W[q] . f(GH_Z[q])` for `Z ~ Normal(0, 1)`.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(8)
GH_Z = (np.sqrt(2.0) * _GH_NODES).astype(np.float64)
GH_W = (_GH_WEIGHTS / np.sqrt(np.pi)).astype(np.float64)


# ---------------------------------------------------------------------------
# per-source conditioning (brief item 1)
# ---------------------------------------------------------------------------

def _source_tile_ids(config, region):
    """Every catalogued source's own STAR-anchor tile index, by the same
    nearest-pixel join `prior.star_shapes.tile_map_classes` uses: the
    tiles product's `HPX_PIX_512 -> TILE_ID` table, joined through the
    source's own `hpx_pix_512` (`granules.access.region_slice`).
    """
    tiles_path = config_module.product_path(config, "bms", "anchors", "tiles", "hpx512", region=region)
    if not os.path.exists(tiles_path):
        raise FileNotFoundError(
            "prior.counts_star_family: tiles product missing for region %r at %s -- "
            "run the 'prior.anchor_tiles' RUNBOOK line first" % (region, tiles_path))
    with h5py.File(tiles_path, "r") as f:
        pix512 = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        tile_of_pix = np.asarray(f["TILE_ID"][:], dtype=np.int64)
    rs = access.region_slice(config, region)
    src_pix512 = np.asarray(rs["hpx_pix_512"], dtype=np.int64)
    order = np.argsort(pix512)
    pix_sorted = pix512[order]
    loc = np.searchsorted(pix_sorted, src_pix512)
    capped = np.minimum(loc, pix_sorted.size - 1) if pix_sorted.size else loc
    return tile_of_pix[order][capped]


def _read_zp_sigma_k(path, n):
    """`ZP_SIGMA_K` (mag) per source -- the Herschel field zero point's own
    uncertainty, 0 for a Planck-arm source (owner, 2026-09-06) -- if the
    adopted column product has it, else zeros: a region not yet rebuilt
    with the per-field offset runs exactly as before."""
    with h5py.File(path, "r") as f:
        if "ZP_SIGMA_K" in f:
            return np.asarray(f["ZP_SIGMA_K"][:], dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def source_conditioning(config, region):
    """Every catalogued source's own conditioning scalars: `TILE_ID`, the
    shared column-kernel grid's own bracket on `A_COL_K` (`NODE_LO`/
    `NODE_W`, kept for the fitter's kernel read), and the source's own
    8 micron 50%-completeness limit `F_LIM8_MJY` (PAHC's shape reads it
    directly; the GAL literature check reports against it)."""
    f_lim_8band = limits_module.limits(config, region)
    f_lim8 = f_lim_8band[:, star_population.IDX_I4]

    adopted_path = config_module.product_path(config, "sky/derived", "adopted",
                                               "column", "source", region=region)
    a_col = access.per_source(config, region, adopted_path, ["A_COL_K"])["A_COL_K"]
    a_col = np.asarray(a_col, dtype=np.float64)

    a_nodes_full = column_grid.nodes(config)
    node_lo, node_w = column_grid.bracket(a_col, a_nodes_full)

    tile_id = _source_tile_ids(config, region)

    adopted = access.per_source(config, region, adopted_path, ["A_COL_SIG_K", "A_COL_PROVENANCE"])
    sigma_col = np.asarray(adopted["A_COL_SIG_K"], dtype=np.float64)
    provenance = np.asarray(adopted["A_COL_PROVENANCE"])
    map_class = np.where(provenance == star_shapes._PLANCK_PROVENANCE_CODE, "planck", "herschel")
    zp_sigma_k = _read_zp_sigma_k(adopted_path, a_col.size)

    return dict(n_source=f_lim_8band.shape[0], a_col=a_col, f_lim8=f_lim8,
               node_lo=node_lo, node_w=node_w, tile_id=tile_id,
               sigma_col=sigma_col, map_class=map_class, zp_sigma_k=zp_sigma_k)


# ---------------------------------------------------------------------------
# reading a class's exact per-source selection (prior.star_selection's
# and prior.gal's products)
# ---------------------------------------------------------------------------

_STAR_SEL_KEY = {"star": "EPS_STAR", "pahc": "EPS_PAHC", "agb": "EPS_AGB"}
_STAR_SEL_B_KEY = {"star": "LOG10_B_GRID_STAR", "pahc": "LOG10_B_GRID_PAHC", "agb": "LOG10_B_GRID_AGB"}


def read_family_selection(config, region, cls):
    """`(eps, x_ladder, b_grid)`: the class's exact per-source selection
    off `prior.star_selection`'s product -- `eps` is `(n_source, n_x,
    n_b)`, row-aligned with the curated catalogue."""
    path = config_module.product_path(config, "bms", "star", "selection", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.counts_star_family: no star-selection product for region %r at %s -- "
            "run the 'prior.star_selection' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        eps = f[_STAR_SEL_KEY[cls]][:].astype(np.float32)
        x_ladder = f["X_LADDER"][:].astype(np.float64)
        b_grid = f[_STAR_SEL_B_KEY[cls]][:].astype(np.float64)
    return eps, x_ladder, b_grid


#: `gal_counts`'s own per-batch working-set budget (owner, 2026-09-06: no
#: per-source galaxy product -- the exact selection is read from the
#: survey-wide colour-CDF tables on the fly, per batch, instead).
GAL_BATCH_BUDGET_BYTES = 512 << 20


# ---------------------------------------------------------------------------
# the fixed bilinear-interpolation matrices: the shape's own grid onto
# the class's own selection ladder/brightness grid -- the same for every
# source (module docstring)
# ---------------------------------------------------------------------------

def _lin_interp_matrix(grid, query):
    """`(n_query, n_grid)`: the dense linear-interpolation weights taking
    values on `grid` (monotonic increasing) to `query`, clamped at either
    edge -- two nonzero entries per row."""
    grid = np.asarray(grid, dtype=np.float64)
    query = np.clip(np.asarray(query, dtype=np.float64), grid[0], grid[-1])
    idx = np.clip(np.searchsorted(grid, query) - 1, 0, grid.size - 2)
    span = grid[idx + 1] - grid[idx]
    t = np.where(span > 0.0, (query - grid[idx]) / np.where(span > 0.0, span, 1.0), 0.0)
    w = np.zeros((query.size, grid.size), dtype=np.float64)
    rows = np.arange(query.size)
    w[rows, idx] += 1.0 - t
    w[rows, idx + 1] += t
    return w


def selection_on_shape_grid(shape, x_ladder, b_grid):
    """`Wb`: the fixed bilinear-interpolation matrix carrying a source's
    own small selection array's `log10 B` axis onto the shape's own
    `b_centers` tabulation axis (module docstring), `(n_b_shape,
    n_b_class)` -- the same matrix for every source, since only the
    class's own selection brightness grid and the shape's own grid are
    involved, neither of which varies by source. The `x` axis carries
    the per-source kernel shift instead (`_shift_and_interp_eps`), so it
    has no fixed matrix."""
    return _lin_interp_matrix(b_grid, shape.b_centers)


def _shift_and_interp_eps(eps_batch, x_ladder, wb, xi_centers, mu_shift):
    """`(nb, n_x_shape, n_b_shape)`: a batch's own small selection array
    read onto the shape's grid with the class shape's per-source, per-
    mixture-component shift applied to the SELECTION's query point
    instead of the density's (the fix, item 1) -- `Sum_x shape(x - mu) .
    eps(x)` and `Sum_x shape(x) . eps(x - mu)` are the same integral
    (a plain relabelling of which factor carries the shift), and only the
    second needs no per-source density read: `x` (linear) is queried at
    `10**(xi_centers - mu_shift)` per source, clamped at the ladder's own
    ends, and interpolated by the SAME two-point linear rule the fixed
    build-time matrix used, now evaluated per source since the query
    point now depends on the source; `log10 B` uses the fixed `wb`
    unchanged (no shift on that axis)."""
    query_log_x = xi_centers[np.newaxis, :] - mu_shift[:, np.newaxis]     # (nb, n_x_shape)
    query_x = np.clip(10.0 ** query_log_x, x_ladder[0], x_ladder[-1])
    idx = np.clip(np.searchsorted(x_ladder, query_x) - 1, 0, x_ladder.size - 2)
    span = x_ladder[idx + 1] - x_ladder[idx]
    t = np.where(span > 0.0, (query_x - x_ladder[idx]) / np.where(span > 0.0, span, 1.0), 0.0)
    rows = np.arange(eps_batch.shape[0])[:, np.newaxis]
    lo = eps_batch[rows, idx, :]                                        # (nb, n_x_shape, n_b_class)
    hi = eps_batch[rows, idx + 1, :]
    eps_x = (1.0 - t)[..., np.newaxis] * lo + t[..., np.newaxis] * hi
    return np.einsum("bj,nxj->nxb", wb, eps_x)                          # (nb, n_x_shape, n_b_shape)


# ---------------------------------------------------------------------------
# STAR / AGB / PAHC: Z_C = E[eps] by grid quadrature on the shape's own
# tabulation grid, batched over sources -- no bicubic read: the stored
# tile array read at its own two bracketing width nodes (PAHC: also its
# own two bracketing limit grids), doubled under the mixture kernel and
# combined by its own weight (brief item 1)
# ---------------------------------------------------------------------------

def family_counts(config, region, cls, cond):
    """`(n_c, z_c, shape)`: the count and normaliser for one family class
    at every source. `Z_C(s)` is the dot product, on the shape's own
    `(xi_centers, b_centers)` grid, of the shape's own stored tile array
    (read exactly at its two bracketing width-ladder nodes and linearly
    blended -- no bicubic) with the source's own small selection array
    (interpolated onto the grid, its query point carrying the per-source
    kernel shift instead of the density, brief item 1). The kernel is
    now a two-component mixture (`Kernel.mixture`): each component has
    its own shift and width, so the whole thing is built twice -- once
    per component, each with its own width-node blend and its own
    shifted selection read -- and combined by the mixture's own weight
    `w`. PAHC additionally blends the two bracketing 8-micron limit
    grids, on top of the width blend, for each component."""
    shape = star_shapes.read(config, region, cls)
    eps, x_ladder, b_grid = read_family_selection(config, region, cls)
    wb = selection_on_shape_grid(shape, x_ladder, b_grid)

    n_x, n_b = shape.xi_centers.size, shape.b_centers.size
    n_source = cond["n_source"]
    a_col, tile_id = cond["a_col"], cond["tile_id"]
    sigma_col, map_class = cond["sigma_col"], cond["map_class"]
    zp_sigma_k = cond["zp_sigma_k"]
    amp = family_amplitude(config, region, cls, cond)

    kern = shape.kern
    w_mix, mu_mix, sigma_mix = kern.mixture(a_col, sigma_col, map_class, zp_sigma_k=zp_sigma_k)  # (n,), (n,2), (n,2)
    log_shape_nodes = np.log(shape.shape_nodes)
    is_pahc = cls == "pahc"
    if is_pahc:
        m_lo, t_limit = column_grid.bracket(np.log10(cond["f_lim8"]), shape.limit_log)
        m_hi = np.minimum(m_lo + 1, shape.limit_log.size - 1)

    row_bytes = n_x * n_b * 8 * 8

    def _one_batch(start, stop):
        eps_batch = eps[start:stop].astype(np.float64)   # (nb, n_x_ladder, n_b_class)
        tile_b = tile_id[start:stop]
        comp_sum = np.zeros(stop - start, dtype=np.float64)

        for k in range(2):
            mu_k = mu_mix[start:stop, k]
            sigma_k = sigma_mix[start:stop, k]
            i_lo, t_w = column_grid.bracket(np.log(sigma_k), log_shape_nodes)
            i_hi = np.minimum(i_lo + 1, shape.shape_nodes.size - 1)

            if is_pahc:
                mlo_b, mhi_b, tlim_b = m_lo[start:stop], m_hi[start:stop], t_limit[start:stop]
                d_lolo = shape.density_table[tile_b, i_lo, mlo_b]
                d_lohi = shape.density_table[tile_b, i_lo, mhi_b]
                d_hilo = shape.density_table[tile_b, i_hi, mlo_b]
                d_hihi = shape.density_table[tile_b, i_hi, mhi_b]
                d_lo = (1.0 - tlim_b)[:, None, None] * d_lolo + tlim_b[:, None, None] * d_lohi
                d_hi = (1.0 - tlim_b)[:, None, None] * d_hilo + tlim_b[:, None, None] * d_hihi
            else:
                d_lo = shape.density_table[tile_b, i_lo]
                d_hi = shape.density_table[tile_b, i_hi]
            dens_k = (1.0 - t_w)[:, None, None] * d_lo + t_w[:, None, None] * d_hi

            eps_grid_k = _shift_and_interp_eps(eps_batch, x_ladder, wb, shape.xi_centers, mu_k)
            weight_k = w_mix[start:stop] if k == 0 else (1.0 - w_mix[start:stop])
            comp_sum += weight_k * (dens_k * eps_grid_k).sum(axis=(1, 2))
        return start, stop, comp_sum

    spans = list(batches_module.batches(n_source, row_bytes, budget_bytes=BATCH_BUDGET_BYTES))
    z_c = np.zeros(n_source, dtype=np.float64)
    for start, stop, comp_sum in Parallel(n_jobs=config.n_jobs, prefer="threads")(
            delayed(_one_batch)(start, stop) for start, stop in spans):
        z_c[start:stop] = comp_sum

    n_c = amp * z_c
    return n_c, z_c, shape


def family_amplitude(config, region, cls, cond):
    """`N_C = amplitude . E[eps]`: STAR/AGB's tile-level `Sigma W_STAR` /
    `Sigma W_AGB` over `OMEGA_SIM_DEG2`; PAHC's tile-level `Sigma W .
    P_PAHC[:, j]`, interpolated in `log10` limit between the population
    product's own eight completeness-limit grid points to the source's
    own `F_LIM8_MJY`."""
    pop_path = config_module.product_path(config, "bms", "star", "population", "tile", region=region)
    with h5py.File(pop_path, "r") as f:
        omega_sim_deg2 = float(f.attrs["OMEGA_SIM_DEG2"])
    pop = star_shapes.read_population(config, region)
    tile_id = cond["tile_id"]

    if cls == "star":
        tile_tot = np.array([t["w_star"].sum() for t in pop["tiles"]], dtype=np.float64)
        return tile_tot[tile_id] / omega_sim_deg2
    if cls == "agb":
        tile_tot = np.array([t["w_agb"].sum() for t in pop["tiles"]], dtype=np.float64)
        return tile_tot[tile_id] / omega_sim_deg2

    limit_grid = pop["limit8_grid_mjy"]
    tile_tot = np.array(
        [[(t["w"] * t["p_pahc"][:, j]).sum() for j in range(limit_grid.size)] for t in pop["tiles"]],
        dtype=np.float64)
    m_lo, t_limit = column_grid.bracket(np.log10(cond["f_lim8"]), np.log10(limit_grid))
    m_hi = np.minimum(m_lo + 1, limit_grid.size - 1)
    lo_val = tile_tot[tile_id, m_lo]
    hi_val = tile_tot[tile_id, m_hi]
    return ((1.0 - t_limit) * lo_val + t_limit * hi_val) / omega_sim_deg2


# ---------------------------------------------------------------------------
# STAR / AGB / PAHC: the DIRECT count, coordinator ruling 2026-09-06 -- NO
# selection table. The count is the population sum SPEC_PRIORS.md section
# 2.1/4 write with the two-of-eight test itself run inside the compiled
# loop, on the star's own 8-band flux dimmed at its own drawn extinction,
# against the SOURCE's own eight limits directly (`selection.epsilon`'s
# arithmetic, `catalog.limits`, the hybrid law's per-band kappa) -- not a
# bilinear read of a pre-tabulated `eps_s(x, B)` array (the earlier form,
# still how `family_counts`/`Z_C` reads the shape's own normaliser below;
# the fitter still reads the stored selection table pointwise, only the
# count no longer does). Per source: every one of the tile's simulated
# stars, at its own placement `u_i` and its own 8-band flux, carries its
# own class weight (PAHC: `W_i . P(q_i(s))` at THIS source's own 8 micron
# limit); its extinction is `a_i(T) = T . u_i` for a true column `T` drawn
# from the source's own two-component kernel mixture (`Kernel.mixture`),
# and its detection is `sum(log10_flux_i - 0.4 . a_i(T) . kappa(a_i(T)) >=
# log10_lim_s) >= 2`, `kappa` the hybrid law's own smooth blend between
# the diffuse and dense curves (`selection.law_dense_weight`/`kappa_ak`,
# hoisted to two plain (8,) vectors and blended inside the loop). The
# expectation over `T` within one mixture component has a known
# distribution (`log10(T/A_s) ~ Normal(mu_k, sigma_k)`) and is taken by
# the fixed 8-point Gauss-Hermite quadrature `GH_Z`/`GH_W` above (found,
# by testing 8-point/32-point Gauss-Hermite and an exact 48-cell mass form
# against the literal 200k-draw reference, to already be converged -- the
# residual against that reference lived in the selection TABLE this form
# removes, not in the column quadrature); the two components are combined
# by the mixture's own weight. This IS the reference's own arithmetic, run
# inside the compiled loop instead of by Monte Carlo sampling.
# ---------------------------------------------------------------------------

#: The hybrid law's ramp domain (`selection.LAW_RAMP_LO`/`LAW_RAMP_HI`),
#: inlined as plain floats so the compiled loop below needs no call back
#: into `selection` per member.
_RAMP_LO = float(selection.LAW_RAMP_LO)
_RAMP_HI = float(selection.LAW_RAMP_HI)
_LN_RAMP_SPAN = float(np.log(_RAMP_HI / _RAMP_LO))
_MIN_BANDS = int(selection.MIN_BANDS)
_N_BANDS = int(selection.N_BANDS)


def _tile_star_index(config, region, n_tile):
    """`STAR_INDEX` per tile (n_star,) int64, straight off `prior.
    star_population`'s own product -- the row into the region's field-star
    register (`prior.field_stars`) each tile member is, not carried by
    `star_shapes.read_population` since the shape-density path never
    needed a member's own flux, only its already-reduced brightness unit."""
    path = config_module.product_path(config, "bms", "star", "population", "tile", region=region)
    with h5py.File(path, "r") as f:
        return [np.asarray(f["tile_%d/STAR_INDEX" % t][:], dtype=np.int64) for t in range(n_tile)]


def _region_log10_flux_and_l(config, region):
    """`(log10_flux, log_l)`: the region's own TRILEGAL field-star
    register (`bms/trilegal/field-stars/region`), `log10_flux` `(n, 8)`
    with a non-positive flux mapped to a large negative sentinel (never
    finite, never clears a limit -- `pass_curves`' own "a band whose
    log10_flux is not finite never clears" rule, made a finite sentinel
    here since the compiled loop below runs under `fastmath`, which does
    not guarantee IEEE NaN/inf comparisons)."""
    path = config_module.product_path(config, "bms", "trilegal", "field-stars", "region", region=region)
    with h5py.File(path, "r") as f:
        flux = f["FNU_MJY"][:].astype(np.float64)
        log_l = f["LOG_L"][:].astype(np.float64)
    log10_flux = np.where(flux > 0.0, np.log10(np.where(flux > 0.0, flux, 1.0)), -1.0e30)
    return log10_flux, log_l


@numba.njit(cache=True, fastmath=True, inline="always")
def _passes_two_of_eight(a, log10_flux_i, log10_lim_s, kd, kw, ramp_lo, ln_ramp_span, min_bands, n_bands):
    """Whether one star, dimmed by extinction `a`, clears the source's own
    detection cut (`selection.epsilon`'s arithmetic, module docstring):
    the hybrid law's ramp weight at `a` (`selection.law_dense_weight`),
    the per-band kappa it gives (`selection.kappa_hybrid`), and the
    two-of-eight count against `log10_lim_s`."""
    if a <= ramp_lo:
        wramp = 0.0
    else:
        x = np.log(a / ramp_lo) / ln_ramp_span
        if x >= 1.0:
            wramp = 1.0
        else:
            wramp = x * x * (3.0 - 2.0 * x)
    n_clear = 0
    for b in range(n_bands):
        kappa_b = (1.0 - wramp) * kd[b] + wramp * kw[b]
        if log10_flux_i[b] - 0.4 * a * kappa_b >= log10_lim_s[b]:
            n_clear += 1
    return n_clear >= min_bands


@numba.njit(cache=True, parallel=True, fastmath=True)
def _direct_sum_literal_fixed(weight, u, log10_flux, log10_lim, a_col, kd, kw,
                               w_mix, mu_mix, sigma_mix, gh_z, gh_w,
                               ramp_lo, ln_ramp_span, min_bands, n_bands):
    """`(n_source,)`: `Sum_i weight_i . E_T[epsilon(a_i(T), B_i)]`, the
    two-of-eight test run directly (module docstring) -- `weight`/`u`/
    `log10_flux` the tile's own fixed per-star arrays (STAR, AGB -- the
    same for every source of the tile), `log10_lim` `(n_source, 8)` each
    source's own limits, `a_col` `(n_source,)` each source's own adopted
    column: the quadrature draws `mult = T/A_s` (the kernel mixture is in
    that ratio), so the star's own REAL extinction is `a = u_i . T = u_i
    . mult . A_s` -- the `. a_col[s]` matters, since `epsilon`'s dimming
    `0.4 . a . kappa(a)` needs the true magnitude, not the scaled ratio
    the old table's `x`-axis used."""
    n_source = log10_lim.shape[0]
    n_member = u.shape[0]
    n_q = gh_z.shape[0]
    out = np.zeros(n_source, dtype=np.float64)
    for s in numba.prange(n_source):
        lim_s = log10_lim[s]
        a_s = a_col[s]
        total = 0.0
        for k in range(2):
            wk = w_mix[s] if k == 0 else (1.0 - w_mix[s])
            if wk <= 0.0:
                continue
            mu_k = mu_mix[s, k]
            sigma_k = sigma_mix[s, k]
            for q in range(n_q):
                mult = a_s * 10.0 ** (mu_k + sigma_k * gh_z[q])
                comp = 0.0
                for i in range(n_member):
                    wi = weight[i]
                    if wi == 0.0:
                        continue
                    a = u[i] * mult
                    if _passes_two_of_eight(a, log10_flux[i], lim_s, kd, kw,
                                             ramp_lo, ln_ramp_span, min_bands, n_bands):
                        comp += wi
                total += wk * gh_w[q] * comp
        out[s] = total
    return out


@numba.njit(cache=True, parallel=True, fastmath=True)
def _direct_sum_literal_pahc(weight, u, log10_flux, log10_lim, a_col, kd, kw,
                              w_mix, mu_mix, sigma_mix, gh_z, gh_w,
                              ramp_lo, ln_ramp_span, min_bands, n_bands):
    """As `_direct_sum_literal_fixed`, but `weight` is `(n_source,
    n_member)` -- PAHC's own `W_i . P(q_i(s))`, a different number per
    source since `q` depends on the source's own 8 micron limit."""
    n_source = log10_lim.shape[0]
    n_member = u.shape[0]
    n_q = gh_z.shape[0]
    out = np.zeros(n_source, dtype=np.float64)
    for s in numba.prange(n_source):
        lim_s = log10_lim[s]
        weight_s = weight[s]
        a_s = a_col[s]
        total = 0.0
        for k in range(2):
            wk = w_mix[s] if k == 0 else (1.0 - w_mix[s])
            if wk <= 0.0:
                continue
            mu_k = mu_mix[s, k]
            sigma_k = sigma_mix[s, k]
            for q in range(n_q):
                mult = a_s * 10.0 ** (mu_k + sigma_k * gh_z[q])
                comp = 0.0
                for i in range(n_member):
                    wi = weight_s[i]
                    if wi == 0.0:
                        continue
                    a = u[i] * mult
                    if _passes_two_of_eight(a, log10_flux[i], lim_s, kd, kw,
                                             ramp_lo, ln_ramp_span, min_bands, n_bands):
                        comp += wi
                total += wk * gh_w[q] * comp
        out[s] = total
    return out


@numba.njit(cache=True, parallel=True, fastmath=True)
def _pahc_weight_matrix(log10_flim8, log10_q0, w_raw, x0, dx, curve_y):
    """`(n_source, n_member)`: PAHC's own per-source member weight,
    `w_raw_i . P(q_i(s))`, `q_i(s) = F_lim,8(s) . 10**LOG10_Q0_i` (module
    docstring). `P(q)` is `prior.pahc_curve.read`'s curve read off its
    OWN evenly-spaced `log10 q` grid (`x0`, the spacing `dx`, and
    `curve_y`, the plateau already folded into its own end values so a
    plain clamp reproduces the curve's `fill_value` exactly) by direct
    index arithmetic, no search. One fused, parallel, compiled pass over
    the whole `(n_source, n_member)` grid -- profiled: building this
    weight matrix as a chain of separate numpy ops (bracket, two gathers,
    two blends, then the `w_raw` multiply) was the batch's real cost, not
    the direct sum's own compiled loop, each numpy pass paying for a full
    memory round trip over an array this size."""
    n_source = log10_flim8.shape[0]
    n_member = log10_q0.shape[0]
    n_bin = curve_y.shape[0]
    out = np.empty((n_source, n_member), dtype=np.float64)
    for s in numba.prange(n_source):
        flim = log10_flim8[s]
        for i in range(n_member):
            pos = (flim + log10_q0[i] - x0) / dx
            if pos < 0.0:
                pos = 0.0
            elif pos > n_bin - 1:
                pos = n_bin - 1.0
            idx = int(pos)
            if idx > n_bin - 2:
                idx = n_bin - 2
            t = pos - idx
            p = curve_y[idx] * (1.0 - t) + curve_y[idx + 1] * t
            out[s, i] = p * w_raw[i]
    return out


def _tile_log10_q0(config, region, n_tile):
    """`LOG10_Q0` per tile (n_star,), straight off `prior.star_population`'s
    own product -- the star's own 8 micron contrast at unit limit
    (`pahc_contamination_weight`), not carried by `star_shapes.
    read_population` since the shape-density path never needed it as a
    per-star number (only the region-wide limit-grid `P_PAHC`, module
    docstring)."""
    path = config_module.product_path(config, "bms", "star", "population", "tile", region=region)
    with h5py.File(path, "r") as f:
        return [np.asarray(f["tile_%d/LOG10_Q0" % t][:], dtype=np.float64) for t in range(n_tile)]


def direct_family_counts(config, region, cls, cond, pop, shape):
    """`N_C(s)` (n_source,), objects per deg**2 (module docstring): the
    population sum SPEC_PRIORS.md section 2.1 (STAR/AGB) and section 4
    (PAHC) write, the two-of-eight test itself run inside the compiled
    loop against the source's own eight limits -- no selection table, one
    tile at a time (every source of a tile shares that tile's own
    simulated population), PAHC's own per-source weight matrix batched
    through `sesnaimpute.batches` so no tile holds more than
    `BATCH_BUDGET_BYTES` of it at once."""
    n_source = cond["n_source"]
    tile_id = cond["tile_id"]
    a_col, sigma_col, map_class = cond["a_col"], cond["sigma_col"], cond["map_class"]
    zp_sigma_k = cond["zp_sigma_k"]
    kern = shape.kern
    w_mix, mu_mix, sigma_mix = kern.mixture(a_col, sigma_col, map_class, zp_sigma_k=zp_sigma_k)

    log10_lim = np.log10(limits_module.limits(config, region)).astype(np.float64)
    kd = selection.kappa_ak(config, selection.LAW_DIFFUSE).astype(np.float64)
    kw = selection.kappa_ak(config, selection.LAW_DENSE).astype(np.float64)

    n_c = np.zeros(n_source, dtype=np.float64)
    n_tile = len(pop["tiles"])
    star_index_by_tile = _tile_star_index(config, region, n_tile)
    log10_flux_region, log_l_region = _region_log10_flux_and_l(config, region)

    if cls == "pahc":
        log10_q0_by_tile = _tile_log10_q0(config, region, n_tile)
        curve = pahc_curve.read(config)
        log10_flim8 = np.log10(cond["f_lim8"])

    ramp_args = (_RAMP_LO, _LN_RAMP_SPAN, _MIN_BANDS, _N_BANDS)

    for t in np.unique(tile_id):
        src_idx = np.flatnonzero(tile_id == t)
        tile = pop["tiles"][int(t)]
        star_index = star_index_by_tile[int(t)]

        if cls == "star":
            w, u = tile["w_star"], tile["u"]
            log10_flux = log10_flux_region[star_index]
            n_c[src_idx] = _direct_sum_literal_fixed(
                w, u, log10_flux, log10_lim[src_idx], a_col[src_idx], kd, kw,
                w_mix[src_idx], mu_mix[src_idx], sigma_mix[src_idx], GH_Z, GH_W, *ramp_args)
        elif cls == "agb":
            ev = tile["is_evolved"]
            w_ev, f_c = tile["w_agb"][ev], pop["f_c"]
            u = np.concatenate([tile["u"][ev], tile["u"][ev]])
            w = np.concatenate([w_ev * (1.0 - f_c), w_ev * f_c])
            log_l_ev = log_l_region[star_index[ev]]
            flux_o, flux_c = star_selection_module.agb_matched_flux(
                config, log_l_ev, tile["log10_b_agb_o"][ev], tile["log10_b_agb_c"][ev])
            log10_flux = np.concatenate([
                np.where(flux_o > 0.0, np.log10(np.where(flux_o > 0.0, flux_o, 1.0)), -1.0e30),
                np.where(flux_c > 0.0, np.log10(np.where(flux_c > 0.0, flux_c, 1.0)), -1.0e30)], axis=0)
            n_c[src_idx] = _direct_sum_literal_fixed(
                w, u, log10_flux, log10_lim[src_idx], a_col[src_idx], kd, kw,
                w_mix[src_idx], mu_mix[src_idx], sigma_mix[src_idx], GH_Z, GH_W, *ramp_args)
        else:
            u, w_raw = tile["u"], tile["w"]
            log10_flux = log10_flux_region[star_index]
            log10_q0 = log10_q0_by_tile[int(t)]
            row_bytes = w_raw.size * 8
            for bstart, bstop in batches_module.batches(src_idx.size, row_bytes, budget_bytes=BATCH_BUDGET_BYTES):
                s_idx = src_idx[bstart:bstop]
                # one vectorised, fused lookup of P(q) for the whole
                # (n_batch, n_pop) grid of log10 q = log10 F_lim(s) +
                # LOG10_Q0(member), built once outside the compiled loop.
                weight = _pahc_weight_matrix(
                    log10_flim8[s_idx], log10_q0, w_raw,
                    float(curve.x[0]), float(curve.x[1] - curve.x[0]), curve.y)
                n_c[s_idx] = _direct_sum_literal_pahc(
                    weight, u, log10_flux, log10_lim[s_idx], a_col[s_idx], kd, kw,
                    w_mix[s_idx], mu_mix[s_idx], sigma_mix[s_idx], GH_Z, GH_W, *ramp_args)

    pop_path = config_module.product_path(config, "bms", "star", "population", "tile", region=region)
    with h5py.File(pop_path, "r") as f:
        omega_sim_deg2 = float(f.attrs["OMEGA_SIM_DEG2"])
    return n_c / omega_sim_deg2


# ---------------------------------------------------------------------------
# GAL: N_GAL and Z_GAL by quadrature over BOTH flux and extinction -- a
# galaxy's own true column is spread by the kernel around A_s, exactly
# as the callable reads it (`Kernel.pdf(a | A_s) . eps_s(a, S)`)
# ---------------------------------------------------------------------------

def gal_counts(config, region, cond):
    """`(n_gal, z_gal, fazio_params)`: `Z_GAL(s) = Integral da p_T(a |
    A_s) . Integral dlog10 S p(log10 S) . eps_s(a, log10 S)`, `N_GAL =
    amplitude . Z_GAL` with `amplitude = Integral phi(S) dS` (the same
    amplitude-times-normalised-integral factoring `family_counts` uses
    for STAR/AGB/PAHC). A galaxy carries the true column, not `A_s`
    exactly, so the extinction factor is the kernel's own mixture
    density in `a` (`Kernel`'s two-component log-normal mixture), not a
    single point at `x = a / A_s = 1`: the source's own selection curve
    is read at every ladder point (`selection.X_LADDER`), the flux
    integral done first at each of those (the counts law normalised to
    one), then the result is weighted by the exact PROBABILITY MASS the
    mixture places in each ladder cell -- the cell edges are the
    midpoints between consecutive ladder nodes (the first cell running
    down to `a = 0`, the last running up to `a = Infinity`), read off the
    kernel's own closed-form CDF (`Kernel.cdf`, the two-Gaussian mixture
    in `log10 T`), so no bicubic and no Monte Carlo, only the ladder's
    own resolution.

    There is no per-source galaxy selection product (owner, 2026-09-06):
    `eps` is evaluated on the fly, per batch of sources, by
    `prior.gal.source_selection_from_cdf` against the survey-wide
    colour-CDF tables (`prior.gal.read_cdf_tables`, loaded once), the
    working set kept under `GAL_BATCH_BUDGET_BYTES` per batch."""
    from sesnaimpute.prior.kernel import Kernel

    x_ladder = selection.X_LADDER
    counts_path = config_module.product_path(config, "bms", "gal", "counts", "survey")
    with h5py.File(counts_path, "r") as f:
        log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
        phi_s = f["PHI_S"][:].astype(np.float64)
        fazio_params = dict(log10_a=float(f["LOG10_A"][()]), log10_s_break=float(f["LOG10_S_BREAK"][()]),
                            alpha_faint=float(f["ALPHA_FAINT"][()]), alpha_bright=float(f["ALPHA_BRIGHT"][()]),
                            smoothness=float(f["SMOOTHNESS"][()]))
    cdf = gal_module.read_cdf_tables(counts_path)

    s_lin = 10.0 ** log10_s_grid
    ln10 = float(np.log(10.0))
    w1 = phi_s * s_lin * ln10          # Integral phi(S) eps dS = Integral w1(S) eps dlogS
    amplitude = float(np.trapz(w1, log10_s_grid))

    log10_lim = np.log10(limits_module.limits(config, region))     # (n_source, 8)
    a_col, sigma_col, map_class = cond["a_col"], cond["sigma_col"], cond["map_class"]
    zp_sigma_k = cond["zp_sigma_k"]
    n_source = a_col.size
    n_x, n_s = x_ladder.size, log10_s_grid.size

    kern = Kernel.read(config)
    mids = 0.5 * (x_ladder[:-1] + x_ladder[1:])                    # (n_x - 1,)

    row_bytes = n_x * (log10_lim.shape[1] * 8 + 8) + n_x * n_s * 8 * 3
    z_gal = np.zeros(n_source, dtype=np.float64)
    for start, stop in batches_module.batches(n_source, row_bytes, budget_bytes=GAL_BATCH_BUDGET_BYTES):
        lim_b = np.ascontiguousarray(log10_lim[start:stop])
        a_b = a_col[start:stop]
        a_query_b = np.ascontiguousarray(x_ladder[np.newaxis, :] * a_b[:, np.newaxis])
        w_dense_b = selection.law_dense_weight(a_query_b)
        kappa_b = np.ascontiguousarray(selection.kappa_hybrid(config, w_dense_b))
        eps_b = gal_module.source_selection_from_cdf(lim_b, a_query_b, kappa_b, log10_s_grid, cdf)

        eps_s_integral_b = np.trapz(eps_b * w1[np.newaxis, np.newaxis, :], log10_s_grid, axis=2) / amplitude

        t_edges = mids[np.newaxis, :] * a_b[:, np.newaxis]         # (n_batch, n_x - 1)
        cdf_edges = kern.cdf(t_edges, a_b, sigma_col[start:stop], map_class[start:stop],
                              zp_sigma_k=zp_sigma_k[start:stop])
        w_x = np.empty((stop - start, n_x), dtype=np.float64)
        w_x[:, 0] = cdf_edges[:, 0]
        w_x[:, 1:-1] = np.diff(cdf_edges, axis=1)
        w_x[:, -1] = 1.0 - cdf_edges[:, -1]

        z_gal[start:stop] = np.sum(w_x * eps_s_integral_b, axis=1)

    n_gal = amplitude * z_gal
    return n_gal, z_gal, fazio_params


# ---------------------------------------------------------------------------
# algebraic acceptance (rule 11): the grid-quadrature Z_C this module
# actually writes, at one representative source, against a direct
# weighted average over that source's tile's own raw stars, each read
# through the SAME source's own selection array (the identity
# `Integral shape . eps = Z`, spec section 0.2/1.3)
# ---------------------------------------------------------------------------

def _bilinear_at_points(table, grid_x, grid_b, x_q, b_q):
    """`(n,)`: `table` (n_x, n_b) bilinearly evaluated at points `(x_q,
    b_q)` (linear `x`, `log10 B`), clamped at either edge."""
    x_q = np.clip(np.asarray(x_q, dtype=np.float64), grid_x[0], grid_x[-1])
    b_q = np.clip(np.asarray(b_q, dtype=np.float64), grid_b[0], grid_b[-1])
    ix = np.clip(np.searchsorted(grid_x, x_q) - 1, 0, grid_x.size - 2)
    ib = np.clip(np.searchsorted(grid_b, b_q) - 1, 0, grid_b.size - 2)
    tx = (x_q - grid_x[ix]) / (grid_x[ix + 1] - grid_x[ix])
    tb = (b_q - grid_b[ib]) / (grid_b[ib + 1] - grid_b[ib])
    v00, v10 = table[ix, ib], table[ix + 1, ib]
    v01, v11 = table[ix, ib + 1], table[ix + 1, ib + 1]
    return ((1 - tx) * (1 - tb) * v00 + tx * (1 - tb) * v10
           + (1 - tx) * tb * v01 + tx * tb * v11)


def algebraic_check(config, region, cls, cond, pop, z_c, src_idx=0):
    """`(direct, approx, dev)`: source `src_idx`'s own `Z_C` as this
    module wrote it (`approx`) against the DIRECT weighted average, over
    that source's own tile's raw star population, of that SAME source's
    own selection array read at each star's own `(x_i = u_i, log10
    B_i)` (a galaxy-free identity check on the family classes' own
    grid quadrature, module docstring)."""
    eps, x_ladder, b_grid = read_family_selection(config, region, cls)
    tile0 = pop["tiles"][int(cond["tile_id"][src_idx])]
    if cls == "star":
        w, u, logb = tile0["w_star"], tile0["u"], tile0["log10_b"]
    elif cls == "agb":
        ev = tile0["is_evolved"]
        w_ev = tile0["w_agb"][ev]
        f_c = pop["f_c"]
        u = np.concatenate([tile0["u"][ev], tile0["u"][ev]])
        logb = np.concatenate([tile0["log10_b_agb_o"][ev], tile0["log10_b_agb_c"][ev]])
        w = np.concatenate([w_ev * (1.0 - f_c), w_ev * f_c])
    else:
        m_lo, _ = column_grid.bracket(np.log10(cond["f_lim8"][src_idx:src_idx + 1]),
                                      np.log10(pop["limit8_grid_mjy"]))
        w = tile0["w"] * tile0["p_pahc"][:, int(m_lo[0])]
        u, logb = tile0["u"], tile0["log10_b_pahc"]

    if w.sum() <= 0.0:
        return float("nan"), float("nan"), float("nan")

    table = eps[src_idx].astype(np.float64)
    eps_i = _bilinear_at_points(table, x_ladder, b_grid, u, logb)
    direct = float(np.sum(w * eps_i) / np.sum(w))
    approx = float(z_c[src_idx])
    return direct, approx, abs(direct - approx)


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

_OUTPUT_KEYS = ("N_STAR", "N_AGB", "N_PAHC", "N_GAL", "Z_STAR", "Z_AGB", "Z_PAHC", "Z_GAL")


def _write(config, region, cond, out):
    path = config_module.product_path(config, "bms", "table", "counts-star-family", "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.create_dataset("TILE_ID", data=cond["tile_id"].astype(np.int32))
        f.create_dataset("NODE_LO", data=cond["node_lo"].astype(np.int32))
        f.create_dataset("NODE_W", data=cond["node_w"].astype(np.float32))
        f.create_dataset("F_LIM8_MJY", data=cond["f_lim8"].astype(np.float32))
        for key in _OUTPUT_KEYS:
            f.create_dataset(key, data=np.asarray(out[key], dtype=np.float32))
    return path


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _pct(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    return tuple(float(v) for v in np.percentile(x, [16.0, 50.0, 84.0]))


def _mosaic_area_deg2(config, region):
    """The region's own mosaic area, deg**2, from the I2-band coverage
    fraction summed over its nside-512 pixels (`sky.derived.coverage`) --
    a sanity SCALE for the literature check, not a fitted quantity."""
    import healpy as hp
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        bands = [b.decode() if isinstance(b, bytes) else b for b in f["BANDS"][:]]
        frac = f["FRAC"][:, bands.index("I2")]
    pix_area_deg2 = hp.nside2pixarea(512, degrees=True)
    return float(np.sum(frac) * pix_area_deg2)


def report(config, region, cond, out, wall_s, tabulated_medians=None):
    lines = []
    n_source = cond["n_source"]
    lines.append("counts_star_family: %s: %d sources, wall=%.1fs" % (region, n_source, wall_s))
    for key in _OUTPUT_KEYS:
        lo, med, hi = _pct(out[key])
        lines.append("counts_star_family: %s: %s median=%.5g [16%%=%.5g, 84%%=%.5g]"
                     % (region, key, med, lo, hi))
    if tabulated_medians:
        for cls, before in tabulated_medians.items():
            after = float(np.median(out["N_%s" % cls.upper()]))
            lines.append(
                "counts_star_family: %s: N_%s before(tabulated shape)=%.5g after(direct "
                "population sum)=%.5g ratio(after/before)=%.4f"
                % (region, cls.upper(), before, after, after / before if before else float("nan")))

    z_all = np.concatenate([out["Z_STAR"], out["Z_AGB"], out["Z_PAHC"], out["Z_GAL"]])
    z_all = z_all[np.isfinite(z_all)]
    lines.append("counts_star_family: %s: Z range [%.4g, %.4g] (acceptance: 0 <= Z <= 1)"
                 % (region, float(z_all.min()) if z_all.size else float("nan"),
                    float(z_all.max()) if z_all.size else float("nan")))

    total_family = out["N_STAR"] + out["N_AGB"] + out["N_PAHC"]
    med_family = float(np.median(total_family[np.isfinite(total_family)]))
    try:
        area = _mosaic_area_deg2(config, region)
        catalogued_density = n_source / area
        lines.append("counts_star_family: %s: literature check (report only): "
                     "median N_STAR+N_AGB+N_PAHC=%.4g deg^-2 vs catalogued source density=%.4g "
                     "deg^-2 over %.4g deg^2 mosaic (most catalogued sources are field stars, "
                     "so this is a sanity scale, not an identity)"
                     % (region, med_family, catalogued_density, area))
    except Exception as exc:
        lines.append("counts_star_family: %s: mosaic-area literature check skipped (%s)" % (region, exc))

    return lines


def report_gal_check(config, region, cond, n_gal, fazio_model_params):
    from sesnaimpute import definitions
    from sesnaimpute.catalog import limits as limits_mod
    from sesnaimpute.prior.gal import BrokenPowerLaw
    fit = BrokenPowerLaw([fazio_model_params["log10_a"], fazio_model_params["log10_s_break"],
                          fazio_model_params["alpha_faint"], fazio_model_params["alpha_bright"],
                          fazio_model_params["smoothness"]])
    idx_i2 = [b.key for b in definitions.BANDS].index("I2")
    f_lim_8band = limits_mod.limits(config, region)
    med_flim_i2 = float(np.median(np.log10(f_lim_8band[:, idx_i2])))
    fazio_cum = float(fit.cumulative(10.0 ** med_flim_i2))
    med_n_gal = float(np.median(n_gal[np.isfinite(n_gal)]))
    return ["counts_star_family: %s: literature check (report only): median N_GAL=%.4g deg^-2 "
           "vs Fazio+2004 cumulative N(>median 4.5um 50%% limit=%.4g mJy)=%.4g deg^-2 "
           "(N_GAL is post-selection, Fazio's is the intrinsic sky count -- N_GAL <= Fazio expected)"
           % (region, med_n_gal, 10.0 ** med_flim_i2, fazio_cum)]


def report_algebraic_checks(region, checks):
    lines = []
    for cls, (direct, approx, dev) in checks.items():
        lines.append("counts_star_family: %s: N/amplitude vs Z (differ by the density's "
                     "smoothing; not a bar) %s: direct=%.5f approx=%.5f dev=%.4f (source 0)"
                     % (region, cls, direct, approx, dev))
    return lines


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _build_one(config, region):
    import time
    with progress.Stage("prior.counts_star_family", region) as st:
        cond = source_conditioning(config, region)

        out = {}
        checks = {}
        tabulated_medians = {}
        pop = star_shapes.read_population(config, region)
        n_batches = len(FAMILY_CLASSES) + 1
        for i_batch, cls in enumerate(FAMILY_CLASSES, start=1):
            # `family_counts` still supplies Z_C (the fitter's smoothed-shape
            # normaliser, unchanged) and, as `n_c_tabulated`, the amplitude-
            # times-tabulated-shape number the blessing page's "before" column
            # reports; the shipped N_C is the population sum (owner ruling
            # 2026-09-06, `direct_family_counts`).
            n_c_tabulated, z_c, shape = family_counts(config, region, cls, cond)
            n_c = direct_family_counts(config, region, cls, cond, pop, shape)
            out["N_%s" % cls.upper()] = n_c
            out["Z_%s" % cls.upper()] = z_c
            tabulated_medians[cls] = float(np.median(n_c_tabulated[np.isfinite(n_c_tabulated)]))
            checks[cls] = algebraic_check(config, region, cls, cond, pop, z_c, src_idx=0)
            st.tick(i_batch, n_batches, "family batches")

        n_gal, z_gal, fazio_params = gal_counts(config, region, cond)
        out["N_GAL"] = n_gal
        out["Z_GAL"] = z_gal
        st.tick(n_batches, n_batches, "family batches")

        path = _write(config, region, cond, out)
        wall_s = time.time() - st.t0

        n_source = cond["a_col"].size
        st.done(path, n_source=n_source,
                N_STAR=float(np.median(out["N_STAR"][np.isfinite(out["N_STAR"])])))

    for line in report(config, region, cond, out, wall_s, tabulated_medians=tabulated_medians):
        print(line, flush=True)
    for line in report_gal_check(config, region, cond, n_gal, fazio_params):
        print(line, flush=True)
    for line in report_algebraic_checks(region, checks):
        print(line, flush=True)
    return path


def build(config, regions=None):
    """Writes the per-source STAR/AGB/PAHC/GAL counts and normalisers for
    `regions` (default: all thirty), one product per region."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        _build_one(config, region)


if __name__ == "__main__":
    run(build)
