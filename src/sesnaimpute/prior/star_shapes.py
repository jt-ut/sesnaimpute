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

**The tabulation grid comes from the catalogue's own resolution, not from
the deposited histogram's shot noise** (`IMPLEMENTATION.md` section 3's
amendment, 2026-09-05). `prior.posterior_width` measures, for every
source, the 1-sigma width of its own `(a, log10 B)` posterior from its
detected bands' flux errors alone -- the same curvature the fitter's
nuisance quadrature forms at fit time (`10_POSTERIOR.md` section 1) --
and this module reads back its survey medians, `sigma_a` (K magnitudes)
and `sigma_logB`. The deposited histogram is smoothed to that scale
before any node sees it: a Gaussian of width `sigma_logB` on the `log10
B` axis (one operator per class, independent of node, applied once per
tile before the per-node step below), and, on `x`, a Gaussian of width
`sigma_a / A` AT THE SHAPE NODE'S OWN COLUMN `A` -- folded into the same
per-node matrix as the kernel's relative-width convolution
(`combined_x_matrix`), since both are linear operators along `x` and a
single matrix product carries both. The `x`-axis tabulation (`x_max`,
hence `x_edges`) does not depend on class -- STAR, AGB and PAHC share one
field-star population's `u` column and one kernel -- so the per-node
combined operator is built ONCE per region, at every column-grid
candidate, and shared by every class's node selection and tile
evaluation (`shared_x_matrices`); only the `log10 B` (and PAHC's `log10
q0`) axis is class-specific.

The grid cell on each axis is set to HALF that axis's smoothing width --
`sigma_logB / 2` on `log10 B`; on `x`, `0.5 * sigma_a / A_max` where
`A_max` is the shared column grid's own ceiling (`AK_CAP`, the largest
column any node in use can sit at): `x` cells are one grid shared by
every node, so the finest requirement -- the smallest fractional width,
at the largest column -- has to set it, or a low-column node's coarser
natural scale would under-resolve a high-column node sharing the same
grid. Bin counts are rounded UP to the next power of two on each axis
and reported alongside the un-rounded count (`choose_x_grid`, `choose_
b_grid`).

Shape nodes are a coarse subset of the shared column grid
(`prior.column_grid.nodes`), chosen on the SMOOTHED density by the 0.002
relative-L1 fidelity bar (`IMPLEMENTATION.md` section 2's `EPS_GRID`,
reused here as section 3's shape bar): a candidate node is dropped when
linear interpolation, in `log A`, between its surviving neighbours
reproduces its own smoothed, convolved shape to within the bar
(`select_shape_nodes`). Candidates are evaluated lazily, one at a time,
and only currently-surviving shapes are cached -- a dropped candidate's
shape is discarded immediately -- so the full column-grid candidate list
(order 183) is never resident as shapes all at once, one class at a
time (owner, 2026-09-05).

AGB blends the O-rich and C-rich shapes (spec section 3) BEFORE the
smoothing and the fidelity tests see it, so what the fidelity bar
measures is the object that is finally stored. PAHC carries a third,
unconvolved axis, `log10 q0` (the star's own 8um contrast at a unit
limit, `prior.star_population.LOG10_Q0`); its resolution is fixed at 48
bins (spec section 4, `IMPLEMENTATION.md` section 3's own PAHC row --
the measured contamination curve's own notch needs them), never halved,
since the product's own `LOG10_Q0_EDGES` is a fixed-length dataset; the
read-time collapse over this axis (`ClassShape.density`'s `f_lim8`) is
the only place a source's own completeness limit enters.

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
from scipy.special import ndtr

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid
from sesnaimpute.prior import kernel as kernel_module
from sesnaimpute.prior import pahc_curve
from sesnaimpute.prior import posterior_width

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

#: PAHC's own `log10 q0` axis is fixed at 48 bins, never halved (spec
#: section 4, `IMPLEMENTATION.md` section 3: the measured contamination
#: curve's own notch needs them): the product's own `LOG10_Q0_EDGES` is
#: a 49-edge dataset by construction.
N_Q0_BINS = 48

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
# the catalogue's own smoothing (module docstring's amendment)
# ---------------------------------------------------------------------------

def _rel_l1(reference, candidate):
    denom = float(np.sum(np.abs(reference)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(np.abs(candidate - reference)) / denom)


def _next_power_of_two(n):
    """The smallest power of two `>= n` -- never coarser than the
    resolution the data set (module docstring: bin counts round UP)."""
    return int(2 ** np.ceil(np.log2(max(1.0, float(n)))))


def gaussian_smoothing_matrix(edges, sigma):
    """The bin-integrated Gaussian smoothing operator on `edges`' own
    bins (module docstring): mass placed at a bin's own centre spreads
    into every output bin by the Gaussian's cumulative distribution
    across that bin's edges, `M[i, j] = Phi((edges[i+1] - centre_j) /
    sigma) - Phi((edges[i] - centre_j) / sigma)`. Mass the truncated
    tabulation box cannot hold is not renormalised back in here -- it
    becomes part of the declared analytic tail (`finalise_node_shape`),
    the same convention `convolution_matrix` uses. `sigma <= 0` (no
    catalogue width to smooth by) is the identity."""
    if not (sigma > 0.0) or not np.isfinite(sigma):
        return np.eye(edges.size - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    z = (edges[:, None] - centers[None, :]) / sigma
    return np.diff(ndtr(z), axis=0)


def apply_b_smoothing(m_b, raw):
    """`m_b` applied along the `log10 B` axis (axis 1) of `raw`, `(n_x,
    n_b)` for STAR/AGB or `(n_x, n_b, n_q0)` for PAHC's node-selection
    marginal (module docstring): `log10 B` smoothing does not depend on
    the shape node, so it is applied once per tile, before the per-node
    `x` convolution (`combined_x_matrix`) sees the histogram."""
    moved = np.moveaxis(raw, 1, 0)
    shape = moved.shape
    out = m_b.dot(moved.reshape(shape[0], -1)).reshape((m_b.shape[0],) + shape[1:])
    return np.moveaxis(out, 0, 1)


def combined_x_matrix(x_edges, r, w, sigma_a_median, node_a):
    """One shape node's combined `x`-axis operator (module docstring):
    the column kernel's own relative-width convolution (`convolution_
    matrix`) followed by the Gaussian smoothing at this node's own
    fractional width `sigma_a_median / node_a` -- two linear operators
    on the same axis, folded into one matrix product so evaluation
    costs exactly what the kernel convolution alone used to."""
    kernel_matrix = convolution_matrix(x_edges, r, w)
    smoothing_matrix = gaussian_smoothing_matrix(x_edges, sigma_a_median / node_a)
    return smoothing_matrix.dot(kernel_matrix)


def choose_x_grid(sigma_a_median, a_max_candidate, x_max):
    """`(x_edges, report)` (module docstring): cell = half the smallest
    fractional catalogue width any node in use can have, `sigma_a_median
    / a_max_candidate`; bin count rounded up to the next power of two.
    Class-independent (the `x`-axis tabulation does not depend on
    class), so this is computed once per region and shared."""
    x_cell = 0.5 * sigma_a_median / a_max_candidate
    n_x_raw = x_max / x_cell if x_cell > 0.0 else 1.0
    n_x = _next_power_of_two(n_x_raw)
    x_edges = np.linspace(0.0, x_max, n_x + 1)
    return x_edges, dict(n_x=n_x, n_x_raw=float(n_x_raw), x_cell=float(x_cell))


def choose_b_grid(sigma_logb_median, b_lo, b_hi):
    """`(b_edges, report)` (module docstring): cell = half the survey-
    median `sigma_logB`; bin count rounded up to the next power of two.
    One class's own `log10 B` range, so this is chosen per class."""
    b_cell = sigma_logb_median / 2.0
    n_b_raw = (b_hi - b_lo) / b_cell if b_cell > 0.0 else 1.0
    n_b = _next_power_of_two(n_b_raw)
    b_edges = np.linspace(b_lo, b_hi, n_b + 1)
    return b_edges, dict(n_b=n_b, n_b_raw=float(n_b_raw), b_cell=float(b_cell))


def select_shape_nodes(candidate_a, shape_fn, eps=EPS_SHAPE):
    """The kept subset of `candidate_a` (`IMPLEMENTATION.md` section 3),
    on the SMOOTHED density `shape_fn(i)` returns for candidate `i`:
    drop a node when linear interpolation, in `log A`, between its
    surviving neighbours reproduces its own shape to within `eps`
    relative L1; endpoints are always kept. One left-to-right pass drops
    every candidate it can, chaining a dropped node's own left anchor
    forward so a run of droppable nodes is tested against the same
    surviving neighbour; repeated to a fixed point (a node spared only
    because of a neighbour dropped later in the same pass can still fall
    on the next one). `shape_fn` is called lazily and its results cached
    only for the currently-surviving scan window -- a dropped candidate's
    shape is evicted immediately -- so candidates never all sit resident
    at once (module docstring)."""
    n = len(candidate_a)
    keep = np.ones(n, dtype=bool)
    log_a = np.log(candidate_a)
    cache = {}

    def shape(i):
        if i not in cache:
            cache[i] = shape_fn(i)
        return cache[i]

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
            interp = (1.0 - t) * shape(lo) + t * shape(hi)
            if _rel_l1(shape(i), interp) < eps:
                keep[i] = False
                cache.pop(i, None)
                changed = True
            else:
                cache.pop(lo, None)  # the old anchor is never read again this pass
                anchor_pos = pos
        # end of pass: only nodes still kept can be reused by the next
        # pass's scan, so nothing else needs to stay cached.
        for stale in [k for k in cache if not keep[k]]:
            cache.pop(stale, None)
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
                 x_centers, b_centers, q0_centers, is_3d, m_b):
    """One tile's own densities at every kept shape node (rule 8: the
    per-tile cost is one raw histogram, one `log10 B` smoothing, plus
    one combined `x`-matrix product per node, never per star). `m_b`
    smooths the deposit on `log10 B` before the per-node `x` matrices
    (module docstring) -- computed once here, not once per node."""
    raw = class_raw_hist(pop, t, cls, x_centers, b_centers, q0_centers)
    raw = apply_b_smoothing(m_b, raw)
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


def build_region_shared(config, region, kern, candidate_a, kernel_cache):
    """Everything a region's three classes share (module docstring):
    the population read, the tile map classes, and the `x`-axis
    tabulation -- STAR, AGB and PAHC deposit the same field-star `u`
    column and read the same kernel, so `x_max` (hence `x_edges`, hence
    every per-node `x`-operator) does not depend on class. Computed
    once per region."""
    pop = read_population(config, region)
    n_tile = pop["n_tile"]
    map_classes = tile_map_classes(config, region, n_tile)
    distinct_map_classes = sorted(set(map_classes))

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
    r_max = 0.0
    for mc in distinct_map_classes:
        for a in candidate_a:
            r, w_q = _kernel_relative_widths(kern, kernel_cache, a, mc)
            order = np.argsort(r)
            cum = np.cumsum(w_q[order]) / np.sum(w_q)
            idx = min(int(np.searchsorted(cum, 1.0 - X_MAX_EPS)), r.size - 1)
            r_max = max(r_max, float(r[order][idx]))
    x_max = r_max * raw_x_max

    return dict(pop=pop, n_tile=n_tile, map_classes=map_classes,
               distinct_map_classes=distinct_map_classes, x_max=x_max,
               a_max_candidate=float(candidate_a.max()))


def shared_x_matrices(x_edges, candidate_a, kern, kernel_cache, distinct_map_classes,
                      sigma_a_median):
    """`{(map_class, candidate_index): matrix}` (module docstring): the
    kernel convolution folded with the node's own `x`-smoothing
    (`combined_x_matrix`), built once per (map class, column-grid
    candidate) and shared by every class's node selection and tile
    evaluation ("share the per-node x matrices across the three
    classes"). Cheap to hold in full: the grid is now sized to the
    catalogue's own resolution, not 256 bins, so 183 candidates times
    two map classes of small `(n_x, n_x)` matrices is a minor cost next
    to the per-tile density arrays these matrices multiply."""
    mats = {}
    for mc in distinct_map_classes:
        for i, a in enumerate(candidate_a):
            r, w_q = _kernel_relative_widths(kern, kernel_cache, a, mc)
            mats[(mc, i)] = combined_x_matrix(x_edges, r, w_q, sigma_a_median, float(a))
    return mats


def build_region_class(config, region, cls, shared, x_matrices, candidate_a, x_edges, x_centers,
                       sigma_logb_median):
    """One region and class, end to end (module docstring): the class's
    own `log10 B` range and smoothing, node selection on the smoothed
    density (streamed, `select_shape_nodes`), and the per-tile
    evaluation in parallel (rule 8, 10a), reusing `shared`'s population
    and `x_matrices`' per-node operators rather than rebuilding either."""
    pop, n_tile = shared["pop"], shared["n_tile"]
    map_classes, distinct_map_classes = shared["map_classes"], shared["distinct_map_classes"]
    is_3d = cls == "pahc"

    b_lo, b_hi = class_b_range(pop, cls)
    q0_centers = q0_edges = None
    if is_3d:
        q0_lo, q0_hi = q0_range(pop)
        q0_edges = np.linspace(q0_lo, q0_hi, N_Q0_BINS + 1)
        q0_centers = 0.5 * (q0_edges[:-1] + q0_edges[1:])

    b_edges, b_grid_report = choose_b_grid(sigma_logb_median, b_lo, b_hi)
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    m_b = gaussian_smoothing_matrix(b_edges, sigma_logb_median)

    # node selection (`IMPLEMENTATION.md` section 3): the first tile's
    # own smoothed density at every candidate, evaluated lazily
    # (`select_shape_nodes`) so candidates never all sit resident at once.
    map_class0 = map_classes[0]
    raw0 = class_raw_hist(pop, 0, cls, x_centers, b_centers, q0_centers=None)
    raw0 = apply_b_smoothing(m_b, raw0)
    shape_fn = lambda idx: apply_convolution(x_matrices[(map_class0, idx)], raw0)
    kept = select_shape_nodes(candidate_a, shape_fn, EPS_SHAPE)
    shape_nodes = candidate_a[kept]

    # the kept nodes' already-built combined operators, one list per
    # map class -- no matrix is rebuilt here (module docstring, "share
    # the per-node x matrices across the three classes").
    matrices_by_map_class = {mc: [x_matrices[(mc, int(idx))] for idx in kept]
                             for mc in distinct_map_classes}

    results = Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(_tile_shapes)(pop, t, cls, shape_nodes, matrices_by_map_class,
                              map_classes[t], x_centers, b_centers, q0_centers, is_3d, m_b)
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
        b_grid_report=b_grid_report,
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
    candidates, the survey posterior width, and the per-(column, map
    class) kernel cache are survey-wide and built once, not per region
    or per class (rule 9); the `x`-axis tabulation and its per-node
    operators are class-independent and built once per region
    (`build_region_shared`, `shared_x_matrices`)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    kern = kernel_module.load(config)
    candidate_a = column_grid.nodes(config)
    kernel_cache = {}
    sigma_a_median, sigma_logb_median = posterior_width.read(config)
    print("star_shapes: catalogue posterior width sigma_a_median=%.4g A_K "
         "sigma_logb_median=%.4g (prior.posterior_width)" % (sigma_a_median, sigma_logb_median),
         flush=True)

    for region in region_names:
        shared = build_region_shared(config, region, kern, candidate_a, kernel_cache)
        x_edges, x_grid_report = choose_x_grid(sigma_a_median, shared["a_max_candidate"],
                                               shared["x_max"])
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        x_matrices = shared_x_matrices(x_edges, candidate_a, kern, kernel_cache,
                                       shared["distinct_map_classes"], sigma_a_median)

        for cls in CLASSES:
            result = build_region_class(config, region, cls, shared, x_matrices, candidate_a,
                                        x_edges, x_centers, sigma_logb_median)
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
                "(x_cell=%.4g raw_n_x=%.1f b_cell=%.4g raw_n_b=%.1f) "
                "median_mass_outside=%.4e sum_identity_max_dev=%.2e "
                "mean_moment(expected=%.4f measured=%.4f rel_dev=%.2e) "
                "grid_centre_exact_max_dev=%.2e%s -> %s"
                % (region, cls, rep["n_candidate"], rep["n_node"], rep["n_x"], rep["n_b"],
                   x_grid_report["x_cell"], x_grid_report["n_x_raw"],
                   result["b_grid_report"]["b_cell"], result["b_grid_report"]["n_b_raw"],
                   rep["median_mass_outside"], rep["sum_identity_dev"],
                   expected, measured, rel_dev, exact_dev, pahc_line, path))


if __name__ == "__main__":
    run(build)
