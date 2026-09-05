"""Per-tile STAR/AGB/PAHC shapes (SPEC_PRIORS.md section 2.2's "The
shape", section 3's AGB blend, section 4's PAHC shape; `IMPLEMENTATION.md`
section 3, the STAR/AGB and PAHC storage rows).

The population is `prior.star_population`'s per-tile point cloud: every
retained field star carries `U` (the scaled extinction `x = a/A` -- at
column `A` the star sits at `a = A*U`, so `x = U` exactly, spec section
2.2's "Per tile") and a class weight (`W_STAR`, `W_AGB`, or the whole `W`
for PAHC). This module deposits that cloud ONCE per tile into a
`(x, log10 B[, log10 q0])` histogram (step 1), then reuses it at every
shape node by a single linear remap along `x` (step 2): the column
kernel's own `r = T/A` distribution at that node rescales `x`, and since
the population's density is piecewise constant on the tabulated bins the
remap is exact -- a cumulative-distribution lookup along `x`, applied to
every `log10 B` (and `log10 q0`) column at once: `IMPLEMENTATION.md`
section 3's `(n_x, n_x)` matrix product `M`, built once per (shape node,
map class) from the identity basis's own piecewise-linear CDF
(`convolution_matrix`) and applied by a real matrix product
(`apply_convolution`) to every tile sharing that node and map class --
this reuse, not any per-node cost, is where the fidelity-driven
candidate scan over the shared column grid dominates the build's own
wall time (`build`'s own timed report).

Shape nodes (a coarse subset of the shared column grid, `prior.column_
grid.nodes`) and the tabulation grid's own resolution are both chosen by
the 0.002 relative-L1 fidelity bar (`IMPLEMENTATION.md` section 2's
`EPS_GRID`, reused here as section 3's shape bar), measured on the
region's first tile (spec section 3's own convention for the AGB/PAHC
grid choice, extended here to STAR): the grid is chosen first, by testing
whether halving `x` or `log10 B` resolution on the RAW (pre-kernel)
histogram still reconstructs the finer histogram via the evaluator's own
bilinear read to within the bar; nodes are chosen second, on that grid,
by testing whether a candidate node's own convolved shape is reproduced,
to within the bar, by linear interpolation in `log A` between its
immediate neighbours in the full column-grid candidate list.

AGB blends the O-rich and C-rich shapes (spec section 3) BEFORE the
kernel convolution and the fidelity tests see it, so what the fidelity
bar measures is the object that is finally stored. PAHC carries a third,
uncoveloved axis, `log10 q0` (the star's own 8um contrast at a unit
limit, `prior.star_population.LOG10_Q0`); its resolution is fixed at 24
bins (spec section 4, `IMPLEMENTATION.md` section 3's own PAHC row),
never halved, since the product's own `LOG10_Q0_EDGES` is a fixed-length
dataset; the read-time collapse over this axis (`ClassShape.density`'s
`f_lim8`) is the only place a source's own completeness limit enters.

Analytic tails beyond the tabulated box (one-sided at the far `x` edge,
two-sided in `log10 B`) are declared from the convolved shape's own edge
behaviour (a matched-slope exponential, the `declare_analytic_tails`
construction), and the interior array is rescaled so interior mass plus
declared tail mass sums to exactly one; PAHC's tails are declared on the
`(x, log10 B)` marginal (summed over `log10 q0`), a simplification stated
in the build's own report.

Writes, per region and class, `bms/<class>/shape_<class>_tile__<Region>.
hdf5` for `class` in `star`, `agb`, `pahc`.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid
from sesnaimpute.prior import kernel as kernel_module
from sesnaimpute.prior import pahc_curve

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

CLASSES = ("star", "agb", "pahc")

#: The fidelity bar (`IMPLEMENTATION.md` section 2's `EPS_GRID`, reused by
#: section 3 for the shape nodes and the tabulation grid): relative L1
#: mass a coarser representation is allowed to misplace.
EPS_SHAPE = 0.002

#: The mass fraction of the kernel's own `r = T/A` distribution excluded
#: from `x_max` (module docstring): tighter than `EPS_SHAPE` because the
#: excluded mass sits at the far end of `x`, where it pulls the interior
#: array's own first moment by more than its bare mass fraction -- set so
#: that the `x`-marginal's mean at a shape node holds `E[r] * mean u`
#: (`IMPLEMENTATION.md` section 3's own algebraic identity) within 1e-3
#: relative, tighter than `EPS_SHAPE` alone gets it.
X_MAX_EPS = 2.0e-4

#: The population's own 0.1-99.9th percentile range sets `log10 B` (and,
#: for PAHC, `log10 q0`), fixed per region (`IMPLEMENTATION.md` section 3).
RANGE_PERCENTILE = (0.1, 99.9)

#: Starting tabulation resolution before the halving search
#: (`IMPLEMENTATION.md` section 3): "start at 256 x 128".
N_X_START = 256
N_B_START = 128

#: PAHC's own `log10 q0` axis is fixed at 24 bins, never halved (spec
#: section 4, `IMPLEMENTATION.md` section 3): the product's own
#: `LOG10_Q0_EDGES` is a 25-edge dataset by construction.
N_Q0_BINS = 24

#: The column kernel's own per-node `T` quadrature (`prior.kernel.Kernel.
#: nodes`) is called at its own default resolution (`n=None`): a coarser
#: `n` under-resolves the quadrature at some low-column nodes, where the
#: kernel's own captured-mass check then refuses, so this module never
#: overrides it.

#: A depth-group / map-class code the kernel understands
#: (`prior.kernel.STATED_BEAM_ARCSEC`): a tile reads whichever of the two
#: its own sources are majority-provenance in (`A_COL_PROVENANCE`, spec
#: section 1.1: 0 Herschel, 1 Planck).
_PLANCK_PROVENANCE_CODE = 1


# ---------------------------------------------------------------------------
# reads: the population product, the tile-to-map-class table
# ---------------------------------------------------------------------------

def read_population(config, region):
    """The region's per-tile field-star point cloud (`prior.star_
    population`'s own product): `F_C`, `LIMIT8_GRID_MJY`, and, per tile,
    `U` (=`x`, module docstring), the class weights, and the three
    brightness units plus `LOG10_Q0`."""
    path = config_module.product_path(config, "bms", "star", "population", "tile", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.star_shapes: no field-star population for region %r at %s -- "
            "run the 'prior.star_population' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        f_c = float(f.attrs["F_C"])
        limit8_grid_mjy = f["LIMIT8_GRID_MJY"][:].astype(np.float64)
        tile_names = sorted((k for k in f.keys() if k.startswith("tile_")),
                             key=lambda s: int(s.split("_")[1]))
        tiles = []
        for name in tile_names:
            g = f[name]
            tiles.append(dict(
                a_tile=float(g.attrs["A_TILE_K"]),
                u=g["U"][:].astype(np.float64),
                w=g["W"][:].astype(np.float64),
                w_star=g["W_STAR"][:].astype(np.float64),
                w_agb=g["W_AGB"][:].astype(np.float64),
                is_evolved=g["IS_EVOLVED"][:].astype(bool),
                log10_b=g["LOG10_B"][:].astype(np.float64),
                log10_b_pahc=g["LOG10_B_PAHC"][:].astype(np.float64),
                log10_b_agb_c=g["LOG10_B_AGB_C"][:].astype(np.float64),
                log10_b_agb_o=g["LOG10_B_AGB_O"][:].astype(np.float64),
                log10_q0=g["LOG10_Q0"][:].astype(np.float64),
            ))
    return dict(f_c=f_c, limit8_grid_mjy=limit8_grid_mjy, tiles=tiles, n_tile=len(tiles))


def tile_map_classes(config, region, n_tile):
    """One kernel map class per tile (spec section 1.2's `map_class`,
    `herschel`/`planck`): the majority `A_COL_PROVENANCE` of the real
    catalogue sources the tile's own `hpx512` pixels contain
    (`prior.anchor_tiles`'s `TILE_ID` map), the same tile-membership join
    `prior.star_population._region_source_geometry` uses."""
    tiles_path = config_module.product_path(config, "bms", "anchors", "tiles", "hpx512", region=region)
    if not os.path.exists(tiles_path):
        raise FileNotFoundError(
            "prior.star_shapes: tiles product missing for region %r at %s -- "
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
    tile_of_source = tile_of_pix[order][capped]

    adopted_path = config_module.product_path(config, "sky/derived", "adopted",
                                               "column", "source", region=region)
    prov = access.per_source(config, region, adopted_path,
                             ["A_COL_PROVENANCE"])["A_COL_PROVENANCE"]
    prov = np.asarray(prov)

    map_classes = []
    for t in range(n_tile):
        sel = tile_of_source == t
        frac_planck = float((prov[sel] == _PLANCK_PROVENANCE_CODE).mean()) if sel.any() else 0.0
        map_classes.append("planck" if frac_planck > 0.5 else "herschel")
    return map_classes


# ---------------------------------------------------------------------------
# CIC deposit (rule 8: vectorised, no per-star loop)
# ---------------------------------------------------------------------------

def _cic_hist(values_per_axis, weight, centers_per_axis):
    """The CIC-deposited histogram on an N-D grid of bin CENTRES: every
    point's weight is split, on each axis, between its two bracketing
    bins (`column_grid.bracket`'s own linear weight), the `2**N` corner
    weights combined and scattered with one `bincount` (rule 8, "np.add.at
    or bincount on flattened indices")."""
    n_axes = len(centers_per_axis)
    brackets = [column_grid.bracket(values_per_axis[k], centers_per_axis[k])
                for k in range(n_axes)]
    shape = tuple(c.size for c in centers_per_axis)
    total = int(np.prod(shape))
    strides = np.cumprod((1,) + shape[::-1][:-1])[::-1]

    flat_idx = []
    flat_w = []
    for corner in range(2 ** n_axes):
        idx = np.zeros_like(weight, dtype=np.int64)
        w_corner = weight.copy()
        for k in range(n_axes):
            i_k, t_k = brackets[k]
            take_hi = bool((corner >> k) & 1)
            idx = idx + strides[k] * (i_k + (1 if take_hi else 0))
            w_corner = w_corner * (t_k if take_hi else (1.0 - t_k))
        flat_idx.append(idx)
        flat_w.append(w_corner)
    flat = np.bincount(np.concatenate(flat_idx), weights=np.concatenate(flat_w),
                       minlength=total)
    return flat.reshape(shape)


# ---------------------------------------------------------------------------
# the column-kernel remap along x, exact for piecewise-constant bins
# ---------------------------------------------------------------------------

def _interp_cdf_at(cdf, x_edges, query):
    """Linear interpolation of a piecewise-linear CDF (`cdf`, `(n_x+1,
    n_col)`, matching a piecewise-constant density on `x_edges`) at
    `query` points, one bracket per query shared by every column (`query`
    does not depend on the column). Beyond the tabulated range the CDF
    holds its endpoint value (0 below, the column's own total above): the
    missing mass becomes the declared analytic tail, not a modelling
    choice here."""
    idx = np.clip(np.searchsorted(x_edges, query) - 1, 0, x_edges.size - 2)
    x0, x1 = x_edges[idx], x_edges[idx + 1]
    span = np.where(x1 > x0, x1 - x0, 1.0)
    t = np.clip((query - x0) / span, 0.0, 1.0)
    lo, hi = cdf[idx, :], cdf[idx + 1, :]
    val = lo + t[:, None] * (hi - lo)
    val = np.where((query < x_edges[0])[:, None], 0.0, val)
    val = np.where((query > x_edges[-1])[:, None], cdf[-1, :][None, :], val)
    return val


def convolution_matrix(x_edges, r, w):
    """`M`, `(n_x, n_x)` (module docstring, `IMPLEMENTATION.md` section
    3): the linear map `h_A(x) = sum_q w_q * h(x/r_q) / r_q`, built once
    from the identity basis (`_interp_cdf_at` applied to the identity's
    own piecewise-linear CDF is exactly this map on every basis bin at
    once) so it can be reused as a dense matrix product against every
    `log10 B` (and `log10 q0`) column of every tile sharing this node and
    map class. Loops over the kernel's own quadrature points one at a
    time: broadcasting them into one `(n_quad, n_x+1, n_x)` array costs
    more to materialise than `n_quad` small ones do to loop over."""
    n_x = x_edges.size - 1
    identity_cdf = np.vstack([np.zeros((1, n_x)), np.cumsum(np.eye(n_x), axis=0)])
    acc = np.zeros_like(identity_cdf)
    for r_q, w_q in zip(r, w):
        acc = acc + w_q * _interp_cdf_at(identity_cdf, x_edges, x_edges / r_q)
    return np.diff(acc, axis=0)


def apply_convolution(matrix, raw):
    """`M.dot(raw)` (module docstring): `raw` is `(n_x, n_b)` for
    STAR/AGB or `(n_x, n_b, n_q0)` for PAHC, flattened to `(n_x, *)` so
    one matrix product convolves every `log10 B`/`log10 q0` column at
    once (rule 8; `IMPLEMENTATION.md` section 3, "if any step is per-star
    per-node, it is wrong")."""
    shape = raw.shape
    return matrix.dot(raw.reshape(shape[0], -1)).reshape(shape)


# ---------------------------------------------------------------------------
# the class's own raw histogram, per tile (step 1)
# ---------------------------------------------------------------------------

def class_raw_hist(pop, tile_idx, cls, x_centers, b_centers, q0_centers=None):
    """The class's one CIC-deposited, unit-mass `(x, log10 B[, log10
    q0])` histogram for one tile (spec sections 2.2/3/4, step 1). AGB
    normalises the O-rich and C-rich shapes SEPARATELY on the evolved
    subset before blending by `f_C` (spec section 3), so the blended
    object -- not either chemistry alone -- is what the kernel convolves
    and the fidelity tests measure."""
    tile = pop["tiles"][tile_idx]
    if cls == "star":
        h = _cic_hist([tile["u"], tile["log10_b"]], tile["w_star"], [x_centers, b_centers])
        total = float(tile["w_star"].sum())
        return h / total if total > 0.0 else h
    if cls == "pahc":
        # `q0_centers=None` (the grid/node fidelity tests, which only
        # look at the (x, log10 B) marginal): the CIC deposit's own q0
        # corner weights sum to 1 over that axis, so omitting it and
        # depositing straight into (x, log10 B) is the exact marginal,
        # not an approximation of it.
        if q0_centers is None:
            h = _cic_hist([tile["u"], tile["log10_b_pahc"]], tile["w"], [x_centers, b_centers])
        else:
            h = _cic_hist([tile["u"], tile["log10_b_pahc"], tile["log10_q0"]], tile["w"],
                         [x_centers, b_centers, q0_centers])
        total = float(tile["w"].sum())
        return h / total if total > 0.0 else h
    if cls == "agb":
        ev = tile["is_evolved"]
        w_ev = tile["w_agb"][ev]
        h_o = _cic_hist([tile["u"][ev], tile["log10_b_agb_o"][ev]], w_ev, [x_centers, b_centers])
        h_c = _cic_hist([tile["u"][ev], tile["log10_b_agb_c"][ev]], w_ev, [x_centers, b_centers])
        t_o, t_c = float(h_o.sum()), float(h_c.sum())
        h_o = h_o / t_o if t_o > 0.0 else h_o
        h_c = h_c / t_c if t_c > 0.0 else h_c
        return (1.0 - pop["f_c"]) * h_o + pop["f_c"] * h_c
    raise ValueError("prior.star_shapes: unknown class %r" % (cls,))


def class_b_range(pop, cls):
    """`(b_lo, b_hi)`, the population's own 0.1-99.9th percentile range
    (`RANGE_PERCENTILE`), measured on the first tile: `LOG10_B`/`LOG10_
    B_AGB_*`/`LOG10_B_PAHC` are per-star, tile-independent quantities
    (`prior.star_population`), so any tile's own copy is the whole
    region's population."""
    tile = pop["tiles"][0]
    if cls == "star":
        vals = tile["log10_b"]
    elif cls == "pahc":
        vals = tile["log10_b_pahc"]
    else:
        ev = tile["is_evolved"]
        vals = np.concatenate([tile["log10_b_agb_o"][ev], tile["log10_b_agb_c"][ev]])
    lo, hi = np.percentile(vals, RANGE_PERCENTILE)
    return float(lo), float(hi)


def q0_range(pop):
    """`(q0_lo, q0_hi)`: PAHC's `log10 q0` axis range, pooled over every
    tile (`LOG10_Q0` is tile-specific: it depends on the star's own tile
    extinction, unlike `LOG10_B`)."""
    vals = np.concatenate([t["log10_q0"] for t in pop["tiles"]])
    lo, hi = np.percentile(vals, RANGE_PERCENTILE)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# the fidelity bar: grid resolution, then shape nodes (step 3)
# ---------------------------------------------------------------------------

def _rel_l1(reference, candidate):
    denom = float(np.sum(np.abs(reference)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(np.abs(candidate - reference)) / denom)


def _bilinear_reconstruct(density, centers0, centers1, query0, query1):
    """The coarse-grid evaluator's own bilinear read (`ClassShape._eval_
    interior`'s construction, inlined for the fidelity test): `density`
    on `(centers0, centers1)`, sampled at every `(query0, query1)` pair."""
    i0, t0 = column_grid.bracket(query0, centers0)
    i1, t1 = column_grid.bracket(query1, centers1)
    v00 = density[np.ix_(i0, i1)]
    v01 = density[np.ix_(i0, i1 + 1)]
    v10 = density[np.ix_(i0 + 1, i1)]
    v11 = density[np.ix_(i0 + 1, i1 + 1)]
    return ((1 - t0)[:, None] * (1 - t1)[None, :] * v00
            + (1 - t0)[:, None] * t1[None, :] * v01
            + t0[:, None] * (1 - t1)[None, :] * v10
            + t0[:, None] * t1[None, :] * v11)


def _halve_axis_ok(pop, tile_idx, cls, x_edges, b_edges, axis, eps):
    """One halving trial (`IMPLEMENTATION.md` section 3, "halve an axis
    while the coarser grid's mass moves by less than 0.002 relative L1
    against the finer one"): the coarser RAW histogram, read back by the
    evaluator's own bilinear interpolation at the finer grid's bin
    centres, must reproduce the finer RAW histogram (both CIC-deposited
    straight from the point cloud, never resampled from each other)."""
    edges = [x_edges, b_edges]
    if edges[axis].size - 1 < 4 or (edges[axis].size - 1) % 2 != 0:
        return False, None
    coarse_edges = list(edges)
    coarse_edges[axis] = edges[axis][::2]

    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    coarse_centers = [0.5 * (e[:-1] + e[1:]) for e in coarse_edges]

    fine_hist = class_raw_hist(pop, tile_idx, cls, centers[0], centers[1])
    coarse_hist = class_raw_hist(pop, tile_idx, cls, coarse_centers[0], coarse_centers[1])

    dx_fine, db_fine = np.diff(edges[0]), np.diff(edges[1])
    dx_coarse, db_coarse = np.diff(coarse_edges[0]), np.diff(coarse_edges[1])
    coarse_density = coarse_hist / dx_coarse[:, None] / db_coarse[None, :]
    approx_density = _bilinear_reconstruct(coarse_density, coarse_centers[0], coarse_centers[1],
                                           centers[0], centers[1])
    approx_mass = approx_density * dx_fine[:, None] * db_fine[None, :]
    ok = _rel_l1(fine_hist, approx_mass) < eps
    return ok, (coarse_edges[axis] if ok else None)


def choose_grid(pop, tile_idx, cls, x_max, b_lo, b_hi, eps=EPS_SHAPE):
    """`(x_edges, b_edges)` (`IMPLEMENTATION.md` section 3): start at
    `N_X_START` x `N_B_START`, halve `x` while the bar holds, then halve
    `log10 B` at the resulting `x` resolution -- one axis at a time, on
    the region's first tile."""
    x_edges = np.linspace(0.0, x_max, N_X_START + 1)
    b_edges = np.linspace(b_lo, b_hi, N_B_START + 1)
    while True:
        ok, coarser = _halve_axis_ok(pop, tile_idx, cls, x_edges, b_edges, axis=0, eps=eps)
        if not ok:
            break
        x_edges = coarser
    while True:
        ok, coarser = _halve_axis_ok(pop, tile_idx, cls, x_edges, b_edges, axis=1, eps=eps)
        if not ok:
            break
        b_edges = coarser
    return x_edges, b_edges


def select_shape_nodes(candidate_a, node_shapes, eps=EPS_SHAPE):
    """The kept subset of `candidate_a` (`IMPLEMENTATION.md` section 3):
    drop a node when linear interpolation, in `log A`, between its
    surviving neighbours reproduces its own convolved shape to within
    `eps` relative L1; endpoints are always kept. One left-to-right pass
    drops every candidate it can, chaining a dropped node's own left
    anchor forward so a run of droppable nodes is tested against the
    same surviving neighbour; repeated to a fixed point (a node spared
    only because of a neighbour dropped later in the same pass can still
    fall on the next one) -- O(n) per pass rather than restarting the
    scan after every single removal."""
    n = len(candidate_a)
    keep = np.ones(n, dtype=bool)
    log_a = np.log(candidate_a)
    changed = True
    while changed:
        changed = False
        kept_idx = np.flatnonzero(keep)
        if kept_idx.size <= 2:
            break
        anchor_pos = 0
        for pos in range(1, kept_idx.size - 1):
            i, lo, hi = kept_idx[pos], kept_idx[anchor_pos], kept_idx[pos + 1]
            span = log_a[hi] - log_a[lo]
            t = 0.0 if span <= 0.0 else (log_a[i] - log_a[lo]) / span
            interp = (1.0 - t) * node_shapes[lo] + t * node_shapes[hi]
            if _rel_l1(node_shapes[i], interp) < eps:
                keep[i] = False
                changed = True
            else:
                anchor_pos = pos
    return np.flatnonzero(keep)


# ---------------------------------------------------------------------------
# analytic tails beyond the tabulated box (step 2)
# ---------------------------------------------------------------------------

def _edge_tail(marginal, grid, side):
    """One edge's matched-slope exponential tail, `(slope, mass)` --
    `(0.0, 0.0)` where matching would require an upward slope (never
    extrapolate upward). `side="hi"` matches the grid's last two points
    (slope must be negative); `side="lo"` matches the first two (slope
    must be positive)."""
    if side == "hi":
        v0, v1, g0, g1, edge_val = marginal[-2], marginal[-1], grid[-2], grid[-1], marginal[-1]
    else:
        v0, v1, g0, g1, edge_val = marginal[0], marginal[1], grid[0], grid[1], marginal[0]
    if not (v0 > 0.0 and v1 > 0.0 and edge_val > 0.0 and g1 > g0):
        return 0.0, 0.0
    slope = float((np.log(v1) - np.log(v0)) / (g1 - g0))
    if side == "hi":
        if not slope < 0.0:
            return 0.0, 0.0
        mass = edge_val / (-slope)
    else:
        if not slope > 0.0:
            return 0.0, 0.0
        mass = edge_val / slope
    return slope, float(mass)


def finalise_node_shape(conv, x_centers, b_centers, is_3d):
    """The stored per-(tile, node) product (step 2): declares the
    analytic tail from the convolved array's own edge marginals (`(x,
    log10 B)`, summed over `log10 q0` for PAHC -- the stated
    simplification, module docstring), then rescales the interior array
    so interior mass plus declared tail mass sums to exactly one
    (algebraic acceptance: `DENSITY.sum() == 1 - MASS_OUTSIDE`)."""
    marg2d = conv.sum(axis=2) if is_3d else conv
    x_marg = marg2d.sum(axis=1)
    b_marg = marg2d.sum(axis=0)
    x_slope, x_mass = _edge_tail(x_marg, x_centers, "hi")
    b_lo_slope, b_lo_mass = _edge_tail(b_marg, b_centers, "lo")
    b_hi_slope, b_hi_mass = _edge_tail(b_marg, b_centers, "hi")
    tail_mass = x_mass + b_lo_mass + b_hi_mass
    total = float(conv.sum()) + tail_mass
    density = (conv / total) if total > 0.0 else conv
    mass_outside = tail_mass / total if total > 0.0 else 0.0
    return (density.astype(np.float32), x_slope, b_lo_slope, b_hi_slope, mass_outside)


# ---------------------------------------------------------------------------
# one region and class, end to end
# ---------------------------------------------------------------------------

def _tile_shapes(pop, t, cls, kept_nodes, matrices_by_map_class, map_class,
                 x_centers, b_centers, q0_centers, is_3d):
    """One tile's own densities at every kept shape node (rule 8: the
    per-tile cost is one raw histogram plus one matrix product per node,
    never per star)."""
    raw = class_raw_hist(pop, t, cls, x_centers, b_centers, q0_centers)
    matrices = matrices_by_map_class[map_class]
    density = np.empty((len(kept_nodes),) + raw.shape, dtype=np.float32)
    tail_x, tail_b_lo, tail_b_hi, mass_outside = (np.empty(len(kept_nodes)) for _ in range(4))
    for j, m in enumerate(matrices):
        conv = apply_convolution(m, raw)
        d, sx, slo, shi, mo = finalise_node_shape(conv, x_centers, b_centers, is_3d)
        density[j], tail_x[j], tail_b_lo[j], tail_b_hi[j], mass_outside[j] = d, sx, slo, shi, mo
    return density, tail_x, tail_b_lo, tail_b_hi, mass_outside


def _kernel_relative_widths(kern, cache, a, map_class):
    """`(r, w)` at column `a` and `map_class`, cached by `(map_class, a)`:
    every one of the up to three places this module needs the kernel's
    own `r = T/A` distribution at a candidate node reads the same call
    once (rule 9, "never touch ... inside a per-item loop" -- the kernel
    quadrature is the expensive per-item cost here)."""
    key = (map_class, float(a))
    cached = cache.get(key)
    if cached is None:
        t_q, w_q = kern.nodes(float(a), map_class)
        cached = (t_q / a, w_q)
        cache[key] = cached
    return cached


def build_region_class(config, region, cls, kern, candidate_a, kernel_cache):
    """One region and class, end to end (module docstring): reads the
    population, chooses the grid then the shape nodes on the first tile,
    builds the node's convolution matrix once per (node, map class), and
    applies it to every tile in parallel (rule 8, 10a). `kernel_cache` is
    shared across every region and class this build call touches (rule
    9): the kernel's own `r = T/A` distribution at a given column and map
    class does not depend on the region or the class, only on the shared
    column-grid candidates and the two map classes, so it is the single
    most expensive per-item computation this module repeats needlessly if
    not shared."""
    pop = read_population(config, region)
    n_tile = pop["n_tile"]
    map_classes = tile_map_classes(config, region, n_tile)
    is_3d = cls == "pahc"

    b_lo, b_hi = class_b_range(pop, cls)
    q0_centers = q0_edges = None
    if is_3d:
        q0_lo, q0_hi = q0_range(pop)
        q0_edges = np.linspace(q0_lo, q0_hi, N_Q0_BINS + 1)
        q0_centers = 0.5 * (q0_edges[:-1] + q0_edges[1:])

    # x_max (module docstring): the largest kernel ratio r=T/A "in use",
    # over every candidate node and every map class this region's tiles
    # carry, times the population's own observed extent in x=u (spec
    # section 1.4: u <= 1 everywhere). The kernel's own quadrature grid
    # (`prior.kernel.Kernel.nodes`) spans a wide multiplicative range in
    # `T` to resolve its tails numerically, at negligible weight near the
    # edge, so "in use" reads the weighted (1-EPS_SHAPE) quantile of `r`,
    # not the raw grid extent -- the same fidelity bar as everywhere
    # else, with the excluded mass falling to the declared analytic tail.
    raw_x_max = max(float(t["u"].max()) for t in pop["tiles"])
    distinct_map_classes = sorted(set(map_classes))
    r_max = 0.0
    for mc in distinct_map_classes:
        for a in candidate_a:
            r, w_q = _kernel_relative_widths(kern, kernel_cache, a, mc)
            order = np.argsort(r)
            cum = np.cumsum(w_q[order]) / np.sum(w_q)
            idx = min(int(np.searchsorted(cum, 1.0 - X_MAX_EPS)), r.size - 1)
            r_max = max(r_max, float(r[order][idx]))
    x_max = r_max * raw_x_max

    x_edges, b_edges = choose_grid(pop, 0, cls, x_max, b_lo, b_hi)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])

    # node selection (`IMPLEMENTATION.md` section 3): every candidate's
    # own convolved shape, on the first tile's own map class, at the
    # chosen grid.
    map_class0 = map_classes[0]
    raw0 = class_raw_hist(pop, 0, cls, x_centers, b_centers, q0_centers)
    node_shapes = []
    for a in candidate_a:
        r, w_q = _kernel_relative_widths(kern, kernel_cache, a, map_class0)
        m = convolution_matrix(x_edges, r, w_q)
        node_shapes.append(apply_convolution(m, raw0))
    kept = select_shape_nodes(candidate_a, node_shapes, EPS_SHAPE)
    shape_nodes = candidate_a[kept]

    # one convolution matrix per (kept node, distinct map class) --
    # reused across every tile that shares a map class (module docstring).
    matrices_by_map_class = {}
    for mc in distinct_map_classes:
        mats = []
        for a in shape_nodes:
            r, w_q = _kernel_relative_widths(kern, kernel_cache, a, mc)
            mats.append(convolution_matrix(x_edges, r, w_q))
        matrices_by_map_class[mc] = mats

    results = Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(_tile_shapes)(pop, t, cls, shape_nodes, matrices_by_map_class,
                              map_classes[t], x_centers, b_centers, q0_centers, is_3d)
        for t in range(n_tile))

    density = np.stack([r[0] for r in results], axis=0)
    tail_x = np.stack([r[1] for r in results], axis=0)
    tail_b_lo = np.stack([r[2] for r in results], axis=0)
    tail_b_hi = np.stack([r[3] for r in results], axis=0)
    mass_outside = np.stack([r[4] for r in results], axis=0)

    return dict(
        region=region, cls=cls, shape_nodes=shape_nodes, x_edges=x_edges, b_edges=b_edges,
        q0_edges=q0_edges, density=density, tail_x=tail_x, tail_b_lo=tail_b_lo,
        tail_b_hi=tail_b_hi, mass_outside=mass_outside, map_classes=map_classes,
        n_tile=n_tile, n_candidate=len(candidate_a), f_c=pop["f_c"], pop=pop,
    )


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def write_region_class(config, region, cls, result):
    path = config_module.product_path(config, "bms", cls, "shape", "tile", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "tile"
        if cls == "agb":
            f.attrs["F_C"] = result["f_c"]
        f.create_dataset("SHAPE_NODES", data=result["shape_nodes"].astype(np.float64))
        f.create_dataset("X_EDGES", data=result["x_edges"].astype(np.float64))
        f.create_dataset("LOG10_B_EDGES", data=result["b_edges"].astype(np.float64))
        if result["q0_edges"] is not None:
            f.create_dataset("LOG10_Q0_EDGES", data=result["q0_edges"].astype(np.float64))
        f.create_dataset("TILE_ID", data=np.arange(result["n_tile"], dtype=np.int64))
        f.create_dataset("MAP_CLASS", data=np.array(result["map_classes"], dtype="S8"))
        f.create_dataset("DENSITY", data=result["density"], compression="gzip", compression_opts=4)
        f.create_dataset("TAIL_X_SCALE", data=result["tail_x"].astype(np.float64))
        f.create_dataset("TAIL_B_LO_SCALE", data=result["tail_b_lo"].astype(np.float64))
        f.create_dataset("TAIL_B_HI_SCALE", data=result["tail_b_hi"].astype(np.float64))
        f.create_dataset("MASS_OUTSIDE", data=result["mass_outside"].astype(np.float64))
    return path


# ---------------------------------------------------------------------------
# read: the evaluator every prior-table build calls
# ---------------------------------------------------------------------------

class ClassShape(object):
    """One class's per-tile shape, evaluated per source (`IMPLEMENTATION.md`
    section 3's evaluation column): the tile's own density, the two
    bracketing shape nodes blended in `log A`, bilinear inside the grid,
    the analytic tail outside, `a < 0` mapped to zero. Vectorised over
    sources (rule 8)."""

    def __init__(self, cls, shape_nodes, x_edges, b_edges, density, tail_x, tail_b_lo,
                 tail_b_hi, mass_outside, q0_edges=None, pahc_curve_fn=None):
        self.cls = cls
        self.shape_nodes = np.asarray(shape_nodes, dtype=np.float64)
        self.x_edges = np.asarray(x_edges, dtype=np.float64)
        self.b_edges = np.asarray(b_edges, dtype=np.float64)
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.b_centers = 0.5 * (self.b_edges[:-1] + self.b_edges[1:])
        self.density_table = np.asarray(density, dtype=np.float64)
        self.tail_x = np.asarray(tail_x, dtype=np.float64)
        self.tail_b_lo = np.asarray(tail_b_lo, dtype=np.float64)
        self.tail_b_hi = np.asarray(tail_b_hi, dtype=np.float64)
        self.mass_outside = np.asarray(mass_outside, dtype=np.float64)
        self.q0_edges = None if q0_edges is None else np.asarray(q0_edges, dtype=np.float64)
        if self.q0_edges is not None:
            self.q0_centers = 0.5 * (self.q0_edges[:-1] + self.q0_edges[1:])
        self._curve = pahc_curve_fn

    def _collapse_q0(self, tile_idx, node_idx, f_lim8):
        """PAHC only (spec section 4): the `log10 q0` axis weighted by
        the measured curve `P(q0 * f_lim8)` at this source's own limit,
        the weights renormalised to a mean (no limit grid, module
        docstring). `q0_centers` is already `log10 q0` (the histogram's
        own axis), so `log10(q0 * f_lim8) = q0_centers + log10(f_lim8)`."""
        w = self._curve(self.q0_centers + np.log10(f_lim8))
        w = np.clip(w, 0.0, None)
        total = w.sum()
        w = w / total if total > 0.0 else np.full_like(w, 1.0 / w.size)
        return np.tensordot(self.density_table[tile_idx, node_idx], w, axes=([2], [0]))

    def _slab(self, tile_idx, node_idx, f_lim8):
        if self.q0_edges is None:
            return self.density_table[tile_idx, node_idx]
        return self._collapse_q0(tile_idx, node_idx, f_lim8)

    def _eval_interior(self, tile_ids, node_idx, x, logb, f_lim8):
        ix, tx = column_grid.bracket(x, self.x_centers)
        ib, tb = column_grid.bracket(logb, self.b_centers)
        n = x.shape[0]
        out = np.empty(n, dtype=np.float64)
        # grouped by (tile, node): a query slab is built once per group,
        # not per source (rule 8/9).
        keys = tile_ids.astype(np.int64) * (self.shape_nodes.size + 1) + node_idx
        for key in np.unique(keys):
            sel = keys == key
            t_i = int(tile_ids[sel][0])
            n_i = int(node_idx[sel][0])
            lim = float(f_lim8[sel][0]) if f_lim8 is not None else None
            slab = self._slab(t_i, n_i, lim)
            ix_s, tx_s, ib_s, tb_s = ix[sel], tx[sel], ib[sel], tb[sel]
            v00, v01 = slab[ix_s, ib_s], slab[ix_s, ib_s + 1]
            v10, v11 = slab[ix_s + 1, ib_s], slab[ix_s + 1, ib_s + 1]
            out[sel] = ((1 - tx_s) * (1 - tb_s) * v00 + (1 - tx_s) * tb_s * v01
                       + tx_s * (1 - tb_s) * v10 + tx_s * tb_s * v11)
        return out

    def _eval_node(self, tile_ids, node_idx, x, logb, f_lim8):
        """One shape node's value, blending the interior bilinear read
        with the declared analytic tail beyond `x_edges[-1]` or beyond
        either `log10 B` edge (module docstring's `finalise_node_shape`)."""
        interior = self._eval_interior(tile_ids, node_idx, x, logb, f_lim8)
        x_hi = x > self.x_edges[-1]
        b_lo = logb < self.b_edges[0]
        b_hi = logb > self.b_edges[-1]
        if not (x_hi.any() or b_lo.any() or b_hi.any()):
            return interior
        edge_x = np.clip(x, None, self.x_edges[-1])
        edge_b = np.clip(logb, self.b_edges[0], self.b_edges[-1])
        edge_val = self._eval_interior(tile_ids, node_idx, edge_x, edge_b, f_lim8)
        tx = self.tail_x[tile_ids, node_idx]
        tblo = self.tail_b_lo[tile_ids, node_idx]
        tbhi = self.tail_b_hi[tile_ids, node_idx]
        out = interior.copy()
        out = np.where(x_hi, edge_val * np.exp(tx * (x - self.x_edges[-1])), out)
        out = np.where(b_lo & ~x_hi, edge_val * np.exp(tblo * (logb - self.b_edges[0])), out)
        out = np.where(b_hi & ~x_hi, edge_val * np.exp(tbhi * (logb - self.b_edges[-1])), out)
        return out

    def density(self, a, log10_b, tile_ids, a_col, f_lim8=None):
        """`density(a, log10_b, tile_ids, a_col[, f_lim8])`
        (`IMPLEMENTATION.md` section 3): `x = a/a_col` per source, the two
        bracketing shape nodes blended in `log A`, `a < 0` mapped to zero."""
        a = np.asarray(a, dtype=np.float64)
        log10_b = np.asarray(log10_b, dtype=np.float64)
        a_col = np.asarray(a_col, dtype=np.float64)
        tile_ids = np.asarray(tile_ids, dtype=np.int64)
        f_lim8_arr = None if f_lim8 is None else np.broadcast_to(
            np.asarray(f_lim8, dtype=np.float64), a.shape)
        ok = a >= 0.0
        x = np.zeros_like(a)
        x[ok] = a[ok] / a_col[ok]

        log_nodes = np.log(self.shape_nodes)
        i_lo, t = column_grid.bracket(np.log(a_col), log_nodes)
        i_hi = np.minimum(i_lo + 1, self.shape_nodes.size - 1)

        val_lo = self._eval_node(tile_ids, i_lo, x, log10_b, f_lim8_arr)
        val_hi = self._eval_node(tile_ids, i_hi, x, log10_b, f_lim8_arr)
        out = (1.0 - t) * val_lo + t * val_hi
        out[~ok] = 0.0
        return out



def read(config, region, cls):
    """The stored per-tile shape as a `ClassShape` (`IMPLEMENTATION.md`
    section 3's evaluation column)."""
    path = config_module.product_path(config, "bms", cls, "shape", "tile", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.star_shapes: no %r shape for region %r at %s -- "
            "run the 'prior.star_shapes' RUNBOOK line first" % (cls, region, path))
    with h5py.File(path, "r") as f:
        shape_nodes = f["SHAPE_NODES"][:]
        x_edges = f["X_EDGES"][:]
        b_edges = f["LOG10_B_EDGES"][:]
        q0_edges = f["LOG10_Q0_EDGES"][:] if "LOG10_Q0_EDGES" in f else None
        density = f["DENSITY"][:]
        tail_x = f["TAIL_X_SCALE"][:]
        tail_b_lo = f["TAIL_B_LO_SCALE"][:]
        tail_b_hi = f["TAIL_B_HI_SCALE"][:]
        mass_outside = f["MASS_OUTSIDE"][:]
    curve_fn = pahc_curve.read(config) if cls == "pahc" else None
    return ClassShape(cls, shape_nodes, x_edges, b_edges, density, tail_x, tail_b_lo,
                      tail_b_hi, mass_outside, q0_edges=q0_edges, pahc_curve_fn=curve_fn)


# ---------------------------------------------------------------------------
# report (rule 10, 11, 13) -- the algebraic acceptance the brief names
# ---------------------------------------------------------------------------

def _report(result):
    density, mass_outside = result["density"], result["mass_outside"]
    sum_interior = density.reshape(density.shape[0], density.shape[1], -1).sum(axis=2)
    sum_identity_dev = float(np.max(np.abs(sum_interior + mass_outside - 1.0)))

    return dict(
        n_node=result["shape_nodes"].size, n_candidate=result["n_candidate"],
        n_x=result["x_edges"].size - 1, n_b=result["b_edges"].size - 1,
        median_mass_outside=float(np.median(mass_outside)),
        sum_identity_dev=sum_identity_dev,
    )


def _mean_moment_check(config, region, cls, result, kern):
    """Acceptance (brief): the `x`-marginal's mean at a shape node equals
    `E[r] * (sum w u / sum w)`, the kernel's own first moment times the
    population's mean `u`, within 1e-3 relative -- measured on the first
    tile, at the node nearest the tile's own column."""
    pop = result["pop"]
    tile0 = pop["tiles"][0]
    map_class0 = result["map_classes"][0]
    node_idx = int(np.argmin(np.abs(result["shape_nodes"] - tile0["a_tile"])))
    a_node = float(result["shape_nodes"][node_idx])
    t_q, w_q = kern.nodes(a_node, map_class0)
    e_r = float(np.sum(w_q * (t_q / a_node)))

    if cls == "star":
        w = tile0["w_star"]
        mean_u = float(np.sum(w * tile0["u"]) / np.sum(w))
    elif cls == "agb":
        ev = tile0["is_evolved"]
        w = tile0["w_agb"][ev]
        mean_u = float(np.sum(w * tile0["u"][ev]) / np.sum(w)) if w.sum() > 0 else float("nan")
    else:
        w = tile0["w"]
        mean_u = float(np.sum(w * tile0["u"]) / np.sum(w))
    expected = e_r * mean_u

    x_centers = 0.5 * (result["x_edges"][:-1] + result["x_edges"][1:])
    density_2d = (result["density"][0, node_idx].sum(axis=-1) if cls == "pahc"
                 else result["density"][0, node_idx])
    x_marg = density_2d.sum(axis=1)
    measured = float(np.sum(x_marg * x_centers) / np.sum(x_marg)) if x_marg.sum() > 0 else float("nan")
    rel_dev = abs(measured - expected) / expected if expected != 0 else float("nan")
    return expected, measured, rel_dev


def _grid_centre_exact_check(result):
    """Acceptance (brief): the evaluator at a grid centre, with a source
    whose `a_col` equals a shape node, reproduces the stored value
    exactly."""
    shape_nodes, x_edges, b_edges = result["shape_nodes"], result["x_edges"], result["b_edges"]
    density = result["density"]
    q0_edges = result["q0_edges"]
    shape = ClassShape(result["cls"], shape_nodes, x_edges, b_edges, density,
                       result["tail_x"], result["tail_b_lo"], result["tail_b_hi"],
                       result["mass_outside"], q0_edges=q0_edges,
                       pahc_curve_fn=(lambda lq: np.ones_like(lq)) if q0_edges is not None else None)
    node_idx = 0
    a_node = float(shape_nodes[node_idx])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    a_query = a_node * x_centers
    tile_ids = np.zeros(a_query.size, dtype=np.int64)
    a_col = np.full(a_query.size, a_node)
    f_lim8 = np.full(a_query.size, 1.0) if q0_edges is not None else None
    got = shape.density(a_query, np.full(a_query.size, b_centers[0]), tile_ids, a_col,
                          f_lim8=f_lim8)
    if q0_edges is not None:
        expected = density[0, node_idx, :, 0, :].mean(axis=-1)
    else:
        expected = density[0, node_idx, :, 0]
    return float(np.max(np.abs(got - expected)))


def _pahc_collapse_check(config, region, result):
    """Acceptance (brief, PAHC only): the read-time `log10 q0` collapse
    at the region's median 8um limit reproduces, within 1e-3 relative L1,
    a direct histogram of the (raw, unconvolved) population weighted by
    `P_PAHC[:, PAHC_LIMIT_MEDIAN_INDEX]` -- the population product's own
    median-limit weights (`prior.star_population`). Measured on the first
    tile, isolating the collapse-and-renormalise mechanism from the
    kernel convolution."""
    from sesnaimpute.prior import star_population

    pop = result["pop"]
    tile0 = pop["tiles"][0]
    path = config_module.product_path(config, "bms", "star", "population", "tile", region=region)
    with h5py.File(path, "r") as f:
        p_pahc_median = f["tile_0"]["P_PAHC"][:, star_population.PAHC_LIMIT_MEDIAN_INDEX].astype(np.float64)
    f_lim8_median = float(pop["limit8_grid_mjy"][star_population.PAHC_LIMIT_MEDIAN_INDEX])

    x_edges, b_edges, q0_edges = result["x_edges"], result["b_edges"], result["q0_edges"]
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    q0_centers = 0.5 * (q0_edges[:-1] + q0_edges[1:])

    direct = _cic_hist([tile0["u"], tile0["log10_b_pahc"]], tile0["w"] * p_pahc_median,
                       [x_centers, b_centers])
    direct = direct / direct.sum() if direct.sum() > 0.0 else direct

    raw3 = class_raw_hist(pop, 0, "pahc", x_centers, b_centers, q0_centers)
    curve = pahc_curve.read(config)
    w_q0 = np.clip(curve(q0_centers + np.log10(f_lim8_median)), 0.0, None)
    w_q0 = w_q0 / w_q0.sum() if w_q0.sum() > 0.0 else np.full_like(w_q0, 1.0 / w_q0.size)
    collapsed = np.tensordot(raw3, w_q0, axes=([2], [0]))
    collapsed = collapsed / collapsed.sum() if collapsed.sum() > 0.0 else collapsed

    return _rel_l1(direct, collapsed)


def build(config, regions=None):
    """Writes the per-tile STAR/AGB/PAHC shape products for `regions`
    (default: all thirty), one file per region and class (module
    docstring). The column kernel, the shared column-grid node
    candidates, and the per-(column, map class) kernel cache are
    survey-wide and built once, not per region or per class (rule 9)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    kern = kernel_module.load(config)
    candidate_a = column_grid.nodes(config)
    kernel_cache = {}

    for region in region_names:
        for cls in CLASSES:
            result = build_region_class(config, region, cls, kern, candidate_a, kernel_cache)
            path = write_region_class(config, region, cls, result)
            rep = _report(result)
            expected, measured, rel_dev = _mean_moment_check(config, region, cls, result, kern)
            exact_dev = _grid_centre_exact_check(result)
            pahc_line = ""
            if cls == "pahc":
                pahc_rel_l1 = _pahc_collapse_check(config, region, result)
                pahc_line = " pahc_collapse_rel_l1=%.2e" % pahc_rel_l1
            print(
                "star_shapes: %s/%s: candidates=%d nodes_kept=%d grid=%dx%d "
                "median_mass_outside=%.4e sum_identity_max_dev=%.2e "
                "mean_moment(expected=%.4f measured=%.4f rel_dev=%.2e) "
                "grid_centre_exact_max_dev=%.2e%s -> %s"
                % (region, cls, rep["n_candidate"], rep["n_node"], rep["n_x"], rep["n_b"],
                   rep["median_mass_outside"], rep["sum_identity_dev"],
                   expected, measured, rel_dev, exact_dev, pahc_line, path))


if __name__ == "__main__":
    run(build)
