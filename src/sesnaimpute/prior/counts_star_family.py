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

STAR, AGB, PAHC share one construction. Each class's per-tile shape
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

is evaluated on the shape's OWN tabulation grid (its ``x_centers`` /
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

GAL (spec section 5.1) needs no shape and no grid: a galaxy's ``a`` IS
the column exactly, so its own ladder coordinate is always ``x = 1``,
one exact point on ``X_LADDER``. Its per-source selection (``prior.
gal``'s ``EPS``, on ``X_LADDER`` by ``LOG10_S_GRID``) is read at that one
ladder point and combined with the survey counts law by direct
quadrature (``numpy.trapz``) over ``LOG10_S_GRID``.

Writes, per region, ``bms/table/counts-star-family_table_source.hdf5``:
root attr ``GRANULE = "source"``; ``TILE_ID``, ``NODE_LO``, ``NODE_W``
(``column_grid.bracket`` on ``A_COL_K`` against the shared column-kernel
grid, kept for the fitter's own kernel read), ``F_LIM8_MJY``, and the
four counts and four normalisers, catalogue row order, float32.
"""

import os

import h5py
import numpy as np

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid, selection, star_population, star_shapes

FAMILY_CLASSES = ("star", "agb", "pahc")

#: The per-batch working-array budget for the (source, shape-x, shape-b)
#: grid expansion, the module's own largest intermediate.
BATCH_BUDGET_BYTES = 512 << 20


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

    return dict(n_source=f_lim_8band.shape[0], a_col=a_col, f_lim8=f_lim8,
               node_lo=node_lo, node_w=node_w, tile_id=tile_id)


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


def read_gal_selection(config, region):
    """`(eps, x_ladder, log10_s_grid)`: GAL's exact per-source selection
    off `prior.gal`'s product -- `eps` is `(n_source, n_x, n_s)`."""
    path = config_module.product_path(config, "bms", "gal", "selection", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.counts_star_family: no GAL selection product for region %r at %s -- "
            "run the 'prior.gal' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        eps = f["EPS"][:].astype(np.float64)
        x_ladder = f["X_LADDER"][:].astype(np.float64)
        log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
    return eps, x_ladder, log10_s_grid


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
    """`(Wx, Wb)`: the fixed bilinear-interpolation matrices carrying a
    source's own small `(n_x, n_b)` selection array onto the shape's
    own `(x_centers, b_centers)` tabulation grid (module docstring) --
    `Wx` is `(n_x_shape, n_x_ladder)` in LINEAR `x` (the shape grid's
    `log10 x` exponentiated, since the ladder is linear), `Wb` is
    `(n_b_shape, n_b_class)` in `log10 B` directly."""
    wx = _lin_interp_matrix(x_ladder, 10.0 ** shape.x_centers)
    wb = _lin_interp_matrix(b_grid, shape.b_centers)
    return wx, wb


# ---------------------------------------------------------------------------
# STAR / AGB / PAHC: Z_C = E[eps] by grid quadrature on the shape's own
# tabulation grid, batched over sources
# ---------------------------------------------------------------------------

def family_counts(config, region, cls, cond):
    """`(n_c, z_c, shape)`: the count and normaliser for one family class
    at every source, evaluated on the shape's own `(x_centers, b_centers)`
    grid (module docstring): the source's own small selection array is
    interpolated onto that grid (`Wx`, `Wb`, fixed across sources) and
    multiplied against `ClassShape.density` at the source's own column
    and tile (which does its own node -- and, for PAHC, limit -- blend
    internally), summed."""
    shape = star_shapes.read(config, region, cls)
    eps, x_ladder, b_grid = read_family_selection(config, region, cls)
    wx, wb = selection_on_shape_grid(shape, x_ladder, b_grid)

    n_x, n_b = shape.x_centers.size, shape.b_centers.size
    a_x = 10.0 ** shape.x_centers            # (n_x,), the shape grid's own linear x
    logb_flat = np.tile(shape.b_centers, n_x)  # (n_x*n_b,)

    n_source = cond["n_source"]
    a_col, tile_id, f_lim8 = cond["a_col"], cond["tile_id"], cond["f_lim8"]
    amp = family_amplitude(config, region, cls, cond)

    z_c = np.empty(n_source, dtype=np.float64)
    row_bytes = n_x * n_b * 8 * 3
    for start, stop in batches_module.batches(n_source, row_bytes, budget_bytes=BATCH_BUDGET_BYTES):
        nb = stop - start
        eps_batch = eps[start:stop].astype(np.float64)              # (nb, n_x_ladder, n_b_class)
        eps_interp = np.einsum("xi,nij,bj->nxb", wx, eps_batch, wb)  # (nb, n_x, n_b)

        a_full = (a_col[start:stop, None] * a_x[None, :]).repeat(n_b, axis=1).reshape(-1)
        logb_full = np.tile(logb_flat, nb)
        tile_full = np.repeat(tile_id[start:stop], n_x * n_b)
        acol_full = np.repeat(a_col[start:stop], n_x * n_b)
        flim_full = np.repeat(f_lim8[start:stop], n_x * n_b) if cls == "pahc" else None

        density_flat = shape.density(a_full, logb_full, tile_full, acol_full, flim_full)
        density_grid = density_flat.reshape(nb, n_x, n_b)
        z_c[start:stop] = (density_grid * eps_interp).sum(axis=(1, 2))

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
# GAL: N_GAL and Z_GAL by direct quadrature at the one ladder point
# x = 1 (a galaxy carries the whole column, module docstring)
# ---------------------------------------------------------------------------

def gal_counts(config, region, cond):
    """`(n_gal, z_gal, fazio_params)`: `N_GAL = Integral phi(S) eps(S | A_s)
    dS`, `Z_GAL` the S-weighted average of `eps`, both `numpy.trapz` over
    `LOG10_S_GRID`, `eps` read at the one ladder point `x = 1` (module
    docstring)."""
    eps, x_ladder, log10_s_grid = read_gal_selection(config, region)
    ladder_ix1 = int(np.argmin(np.abs(x_ladder - 1.0)))
    if abs(x_ladder[ladder_ix1] - 1.0) > 1.0e-9:
        raise ValueError("prior.counts_star_family: GAL selection's X_LADDER carries no exact x=1 point")
    eps_curve = eps[:, ladder_ix1, :]                              # (n_source, n_s)

    counts_path = config_module.product_path(config, "bms", "gal", "counts", "survey")
    with h5py.File(counts_path, "r") as f:
        law_log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
        phi_s = f["PHI_S"][:].astype(np.float64)
        fazio_params = dict(log10_a=float(f["LOG10_A"][()]), log10_s_break=float(f["LOG10_S_BREAK"][()]),
                            alpha_faint=float(f["ALPHA_FAINT"][()]), alpha_bright=float(f["ALPHA_BRIGHT"][()]),
                            smoothness=float(f["SMOOTHNESS"][()]))
    if not np.allclose(law_log10_s_grid, log10_s_grid):
        raise ValueError("prior.counts_star_family: GAL's per-source selection and the survey counts "
                         "law disagree on LOG10_S_GRID for region %r" % region)

    s_lin = 10.0 ** log10_s_grid
    ln10 = float(np.log(10.0))
    w1 = phi_s * s_lin * ln10          # N_GAL: Integral phi(S) eps dS = Integral w1(S) eps dlogS
    w2 = w1 * s_lin                    # Z_GAL: S-weighted average of eps

    n_gal = np.trapz(eps_curve * w1[None, :], log10_s_grid, axis=1)
    num = np.trapz(eps_curve * w2[None, :], log10_s_grid, axis=1)
    den = float(np.trapz(w2, log10_s_grid))
    z_gal = num / den
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


def report(config, region, cond, out, wall_s):
    lines = []
    n_source = cond["n_source"]
    lines.append("counts_star_family: %s: %d sources, wall=%.1fs" % (region, n_source, wall_s))
    for key in _OUTPUT_KEYS:
        lo, med, hi = _pct(out[key])
        lines.append("counts_star_family: %s: %s median=%.5g [16%%=%.5g, 84%%=%.5g]"
                     % (region, key, med, lo, hi))

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
        lines.append("counts_star_family: %s: algebraic check %s: direct=%.5f approx=%.5f "
                     "dev=%.4f (bar 0.02, source 0)"
                     % (region, cls, direct, approx, dev))
    return lines


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _build_one(config, region):
    import time
    t0 = time.time()
    cond = source_conditioning(config, region)

    out = {}
    checks = {}
    pop = star_shapes.read_population(config, region)
    for cls in FAMILY_CLASSES:
        n_c, z_c, shape = family_counts(config, region, cls, cond)
        out["N_%s" % cls.upper()] = n_c
        out["Z_%s" % cls.upper()] = z_c
        checks[cls] = algebraic_check(config, region, cls, cond, pop, z_c, src_idx=0)

    n_gal, z_gal, fazio_params = gal_counts(config, region, cond)
    out["N_GAL"] = n_gal
    out["Z_GAL"] = z_gal

    path = _write(config, region, cond, out)
    wall_s = time.time() - t0

    for line in report(config, region, cond, out, wall_s):
        print(line, flush=True)
    for line in report_gal_check(config, region, cond, n_gal, fazio_params):
        print(line, flush=True)
    for line in report_algebraic_checks(region, checks):
        print(line, flush=True)
    print("counts_star_family: %s -> %s" % (region, path), flush=True)
    return path


def build(config, regions=None):
    """Writes the per-source STAR/AGB/PAHC/GAL counts and normalisers for
    `regions` (default: all thirty), one product per region."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        _build_one(config, region)


if __name__ == "__main__":
    run(build)
