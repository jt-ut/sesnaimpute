"""Per-tile STAR/AGB/PAHC shapes (SPEC_PRIORS.md section 2.2's "The
shape", section 3's AGB blend, section 4's PAHC shape; `IMPLEMENTATION.md`
section 3, the STAR/AGB and PAHC storage rows).

The population is `prior.star_population`'s per-tile point cloud: every
retained field star carries `U` (the scaled extinction `x = a/A` -- at
column `A` the star sits at `a = A*U`, so `x = U` exactly) and a class
weight (`W_STAR`, `W_AGB`, or the whole `W` for PAHC). The class deposits
its own weighted `(log10 U, log10 B)` point cloud per tile onto a fixed
64 x 64 grid (`N_CELLS`), smoothed on `log10 B` at the tile's own
Silverman bandwidth (Silverman 1986, `h = sigma * n**(-1/6)`,
`silverman_bandwidth`).

Shape nodes are a fixed ladder of KERNEL WIDTHS (`SHAPE_WIDTH_LADDER`,
log-spaced dex values), not columns: node `k` is the same tile's
`log10 U` histogram smoothed along `log10 x` by `sqrt(h_x**2 + w_k**2)`,
NO shift -- the per-source shift and the per-source total width (the
sub-beam width composed in quadrature with the source's own measurement
uncertainty and, for Herschel, the field zero point, `Kernel.params`) are
applied entirely at READ time, since both depend on the source's own
column and measurement uncertainty, not on a fixed column ladder. An
analytic exponential tail declared from each node's own convolved edge
behaviour (`_edge_tail`/`finalise_node_shape`) carries the mass the fixed
grid cannot hold.

The grid itself spans the class's region-wide 0.1-99.9th percentile range
on `log10 u` and `log10 B` (`class_u_range`/`class_b_range`), widened by
three times the largest smoothing width seen on that axis (the ladder's
own top entry combined with the largest per-tile Silverman `h_x`,
`_class_grid`). The bicubic-reconstruction and width-interpolation
residuals against the shape fidelity bar (`EPS_SHAPE`) are measured and
reported once per region and class, never searched:
`_bicubic_residual_report` rebuilds tile 0's density at the ladder's
largest width on a 128 x 128 grid and compares a bicubic reconstruction
of it from the 64 x 64 build; `_width_interp_residual_report` compares,
at the geometric midpoint of each adjacent width pair, the two-width
blend against a direct build at that width.

At read time (`ClassShape.density`), a source's own `(w, mu, sigma) =
Kernel.mixture(a_col, sigma_col, map_class)` gives the mixture weight and
each of the two components' own shift and total width; each component's
`sigma` is bracketed in the width ladder separately (log-linear blend,
clamped at the ends) and its own bicubic read taken at `(log10(a / a_col)
- mu, log10 B)` -- the shift moves the query, not the table -- and the
two component reads combined `w * D_1 + (1 - w) * D_2`.

AGB blends the O-rich and C-rich shapes (spec section 3) before the
smoothing sees it; its own bandwidth is measured on the same blend
(`class_x_b_samples`). PAHC's grid and shape nodes are the same as
STAR's, on the whole population's unreduced weight (`class_raw_hist`'s
"pahc" branch); the STORED density at each node is built separately for
every one of the population product's eight 8 micron completeness
limits, weighted by that limit's own contamination probability
(`pahc_raw_hist_for_limit`). At read time a source's own 8 micron limit
brackets two stored limit grids and blends linearly between them in
`log10` limit, the same bracket-and-blend rule the shape nodes use in
`sigma_s` (`ClassShape.density`).

Writes, per region and class, `bms/<class>/shape_<class>_tile__<Region>.
hdf5` for `class` in `star`, `agb`, `pahc`; PAHC's own file carries the
extra `LIMIT8_GRID_MJY` grid and an extra `DENSITY`/tail axis over it.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RectBivariateSpline
from scipy.special import ndtr

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid
from sesnaimpute.prior.kernel import Kernel

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

CLASSES = ("star", "agb", "pahc")

#: The build grid, fixed per axis, per region and class: bicubic
#: reconstruction error scales as the fourth power of the cell size, so
#: halving the cell (64 -> 128) takes the ~6% reconstruction error
#: measured at the widths sources actually use to well under 1%.
N_CELLS = 128

#: The finer grid the bicubic-reconstruction residual is measured
#: against -- kept twice `N_CELLS` so the check still reconstructs a
#: strictly finer grid from the build grid.
N_CELLS_CHECK = 256

#: The fixed ladder of kernel WIDTHS (dex) shape nodes are built at,
#: replacing a column ladder: the sub-beam width alone runs 0.05-0.26 dex
#: over the column grid (measured), and the per-source measurement term
#: (`Kernel.params`'s own quadrature addition) can add several tenths of
#: a dex more at the grid floor, so the ladder is widened to ~1.5 dex to
#: cover it without extrapolating.
SHAPE_WIDTH_LADDER = np.geomspace(0.02, 1.5, 16)

#: The shape fidelity bar the two reported residuals are measured
#: against: relative L1 mass a coarser representation is allowed to
#: misplace (STAR 10% from the anchor weights, GAL 0.1 dex, YSO 0.36 dex,
#: an order of magnitude below any of them).
EPS_SHAPE = 0.02

#: Silverman's rule for the bandwidth of a 2-D kernel density estimate,
#: one bandwidth per axis, `h = sigma * n**(-1/6)` (Silverman 1986).
SILVERMAN_EXPONENT = -1.0 / 6.0

#: `scipy.interpolate.RectBivariateSpline`'s degree per axis.
BICUBIC_DEGREE = 3

#: The population's own 0.1-99.9th percentile range sets both `log10 u`
#: and `log10 B`, region-wide, before the smoothing-width padding.
RANGE_PERCENTILE = (0.1, 99.9)

#: The grid is widened past the raw percentile range by this many times
#: the largest smoothing width seen on that axis.
WIDEN_SIGMAS = 3.0

#: A depth-group / map-class code the kernel understands
#: (`prior.kernel.STATED_BEAM_ARCSEC`): a tile reads whichever of the two
#: its own sources are majority-provenance in (`A_COL_PROVENANCE`, spec
#: section 1.1: 0 Herschel, 1 Planck).
_PLANCK_PROVENANCE_CODE = 1

#: The floor `u` (and `x`) values are clipped to before taking `log10`
#: (a star at exactly zero scaled extinction has no `log10 x`; it falls
#: below the tabulated box and reads as the declared low-`x` tail).
_LOG_FLOOR = 1.0e-300


# ---------------------------------------------------------------------------
# reads: the population product, the tile-to-map-class table
# ---------------------------------------------------------------------------

def read_population(config, region):
    """The region's per-tile field-star point cloud (`prior.star_
    population`'s own product): `F_C`, `LIMIT8_GRID_MJY`, and, per tile,
    `U` (=`x`, module docstring) and its `log10`, the class weights, the
    three brightness units, and PAHC's own per-star, per-limit
    contamination probability `P_PAHC`."""
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
            u = g["U"][:].astype(np.float64)
            tiles.append(dict(
                a_tile=float(g.attrs["A_TILE_K"]),
                u=u,
                log10_u=np.log10(np.clip(u, _LOG_FLOOR, None)),
                w=g["W"][:].astype(np.float64),
                w_star=g["W_STAR"][:].astype(np.float64),
                w_agb=g["W_AGB"][:].astype(np.float64),
                is_evolved=g["IS_EVOLVED"][:].astype(bool),
                log10_b=g["LOG10_B"][:].astype(np.float64),
                log10_b_pahc=g["LOG10_B_PAHC"][:].astype(np.float64),
                log10_b_agb_c=g["LOG10_B_AGB_C"][:].astype(np.float64),
                log10_b_agb_o=g["LOG10_B_AGB_O"][:].astype(np.float64),
                p_pahc=g["P_PAHC"][:].astype(np.float64),
            ))
    return dict(f_c=f_c, limit8_grid_mjy=limit8_grid_mjy, tiles=tiles, n_tile=len(tiles))


def tile_map_classes(config, region, n_tile):
    """One kernel map class per tile (spec section 1.2's `map_class`,
    `herschel`/`planck`): the majority `A_COL_PROVENANCE` of the real
    catalogue sources the tile's own `hpx512` pixels contain
    (`prior.anchor_tiles`'s `TILE_ID` map), the same tile-membership join
    `prior.star_population._region_source_geometry` uses. Stored per
    tile for downstream readers; the build itself no longer depends on
    it (the per-node smoothing carries no shift or map-class term)."""
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
# CIC deposit (vectorised, no per-star loop)
# ---------------------------------------------------------------------------

def _cic_hist(values_per_axis, weight, centers_per_axis):
    """The CIC-deposited histogram on an N-D grid of bin CENTRES: every
    point's weight is split, on each axis, between its two bracketing
    bins (`column_grid.bracket`'s own linear weight), the `2**N` corner
    weights combined and scattered with one `bincount` on flattened
    indices."""
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
# the class's own raw histogram, per tile -- on log10 x
# ---------------------------------------------------------------------------

def class_raw_hist(pop, tile_idx, cls, log_x_centers, b_centers):
    """The class's one CIC-deposited, unit-mass `(log10 x, log10 B)`
    histogram for one tile (spec sections 2.2/3/4). AGB normalises the
    O-rich and C-rich shapes SEPARATELY on the evolved subset before
    blending by `f_C` (spec section 3), so the blended object -- not
    either chemistry alone -- is what the smoothing sees. PAHC deposits
    the WHOLE population's weight, unreduced (spec section 4): this is
    the grid-CHOICE density only -- the density actually stored per
    limit is `pahc_raw_hist_for_limit`'s own, weighted by that limit's
    measured contamination probability."""
    tile = pop["tiles"][tile_idx]
    if cls == "star":
        h = _cic_hist([tile["log10_u"], tile["log10_b"]], tile["w_star"],
                     [log_x_centers, b_centers])
        total = float(tile["w_star"].sum())
        return h / total if total > 0.0 else h
    if cls == "pahc":
        h = _cic_hist([tile["log10_u"], tile["log10_b_pahc"]], tile["w"],
                     [log_x_centers, b_centers])
        total = float(tile["w"].sum())
        return h / total if total > 0.0 else h
    if cls == "agb":
        ev = tile["is_evolved"]
        w_ev = tile["w_agb"][ev]
        h_o = _cic_hist([tile["log10_u"][ev], tile["log10_b_agb_o"][ev]], w_ev,
                       [log_x_centers, b_centers])
        h_c = _cic_hist([tile["log10_u"][ev], tile["log10_b_agb_c"][ev]], w_ev,
                       [log_x_centers, b_centers])
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


def class_u_range(pop, cls):
    """`(u_lo, u_hi)`: the population's own 0.1-99.9th percentile range of
    `log10 u`, over every tile (unlike `LOG10_B`, `log10 u` varies by
    sightline), on the class's own weighted subset."""
    parts = []
    for tile in pop["tiles"]:
        if cls == "star":
            sel = tile["w_star"] > 0.0
        elif cls == "pahc":
            sel = tile["w"] > 0.0
        else:
            sel = tile["is_evolved"] & (tile["w_agb"] > 0.0)
        if sel.any():
            parts.append(tile["log10_u"][sel])
    vals = np.concatenate(parts) if parts else np.zeros(1)
    lo, hi = np.percentile(vals, RANGE_PERCENTILE)
    return float(lo), float(hi)


def class_x_b_samples(pop, tile, cls):
    """`(log_u, x_weight, log_b, b_weight)`: the per-star `log10 u` (this
    class's `log10 x` before any node smoothing) and `log10 B` samples
    and weights the Silverman bandwidth measures for one tile and class
    -- the same population and weights `class_raw_hist` deposits, so the
    bandwidth describes exactly what is smoothed. AGB pools its O-rich
    and C-rich `log10 B` columns weighted by the spec's own `(1 - f_C)` /
    `f_C` chemistry mixture (section 3), matching the blended shape
    `class_raw_hist` builds."""
    if cls == "star":
        w = tile["w_star"]
        return tile["log10_u"], w, tile["log10_b"], w
    if cls == "pahc":
        w = tile["w"]
        return tile["log10_u"], w, tile["log10_b_pahc"], w
    if cls == "agb":
        ev = tile["is_evolved"]
        w_ev = tile["w_agb"][ev]
        f_c = pop["f_c"]
        b_vals = np.concatenate([tile["log10_b_agb_o"][ev], tile["log10_b_agb_c"][ev]])
        b_w = np.concatenate([w_ev * (1.0 - f_c), w_ev * f_c])
        return tile["log10_u"][ev], w_ev, b_vals, b_w
    raise ValueError("prior.star_shapes: unknown class %r" % (cls,))


def _weighted_mean_std(values, weight):
    """The weighted mean and standard deviation of `values` under
    `weight` -- a plain moment identity, not a fitted or tuned number."""
    total = float(weight.sum())
    if total <= 0.0:
        return 0.0, 0.0
    mean = float(np.sum(weight * values) / total)
    var = float(np.sum(weight * (values - mean) ** 2) / total)
    return mean, float(np.sqrt(max(var, 0.0)))


def _effective_n(weight):
    """`(Sum w)**2 / Sum w**2`, the effective sample size of weighted
    data -- what a Silverman bandwidth scales by, not the raw star
    count."""
    total = float(weight.sum())
    sq = float(np.sum(weight ** 2))
    return (total * total / sq) if sq > 0.0 else 0.0


def silverman_bandwidth(values, weight):
    """Silverman's rule for one axis of a 2-D kernel density estimate,
    `h = sigma * n**(-1/6)` (`SILVERMAN_EXPONENT`): `sigma` the weighted
    standard deviation of `values` under `weight`, `n` the weighted
    effective sample size (`_effective_n`). `0.0` where there is no
    weight to measure a width from (an empty tile)."""
    n_eff = _effective_n(weight)
    if n_eff <= 0.0:
        return 0.0
    _, sigma = _weighted_mean_std(values, weight)
    return float(sigma * n_eff ** SILVERMAN_EXPONENT)


def tile_class_bandwidths(pop, cls):
    """Per tile: `dict(h_x=..., h_b=..., n_eff=...)` -- Silverman's rule
    applied to the tile's own `log10 u` and `log10 B` samples at the
    class's own weight (`class_x_b_samples`). `n_eff` is measured on the
    `log10 x` axis's own weight (STAR/PAHC: the one weight column; AGB:
    the evolved weight before the chemistry split, the tile's own
    effective star count, not the doubled pooled-`B` sample's)."""
    out = []
    for tile in pop["tiles"]:
        log_u, w_x, log_b, w_b = class_x_b_samples(pop, tile, cls)
        out.append(dict(h_x=silverman_bandwidth(log_u, w_x),
                        h_b=silverman_bandwidth(log_b, w_b),
                        n_eff=_effective_n(w_x)))
    return out


def pahc_raw_hist_for_limit(tile, limit_idx, log_x_centers, b_centers):
    """PAHC's one CIC-deposited, unit-mass `(log10 x, log10 B)` histogram
    for one tile at one of the population product's eight 8 micron
    completeness-limit grid values: the same whole-population `(log10 u,
    LOG10_B_PAHC)` point cloud `class_raw_hist`'s "pahc" branch deposits,
    but weighted by this limit's own measured nebular-contamination
    probability, `W * P_PAHC[:, limit_idx]` (spec section 4) -- exact per
    limit, no conditional approximation of the per-star weight."""
    w = tile["w"] * tile["p_pahc"][:, limit_idx]
    h = _cic_hist([tile["log10_u"], tile["log10_b_pahc"]], w, [log_x_centers, b_centers])
    total = float(w.sum())
    return h / total if total > 0.0 else h


# ---------------------------------------------------------------------------
# the fixed grid: region-wide percentile range, widened by the smoothing
# ---------------------------------------------------------------------------

def _rel_l1(reference, candidate):
    denom = float(np.sum(np.abs(reference)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(np.abs(candidate - reference)) / denom)


def _class_grid(pop, cls, tile_bw, width_ladder):
    """`(x_edges, b_edges)`: the fixed `N_CELLS`-cell grid per region and
    class (`IMPLEMENTATION.md` section 3) -- the region-wide 0.1-99.9th
    percentile range on `log10 u` and `log10 B` (`class_u_range`,
    `class_b_range`), each widened by `WIDEN_SIGMAS` times the largest
    smoothing width seen on that axis: the ladder's own top entry
    combined with the largest per-tile Silverman `h_x` for `log10 x`,
    the largest per-tile `h_b` alone for `log10 B`."""
    u_lo, u_hi = class_u_range(pop, cls)
    b_lo, b_hi = class_b_range(pop, cls)
    h_x_max = max(bw["h_x"] for bw in tile_bw)
    h_b_max = max(bw["h_b"] for bw in tile_bw)
    width_x_max = float(np.sqrt(h_x_max ** 2 + float(width_ladder.max()) ** 2))
    x_pad = WIDEN_SIGMAS * width_x_max
    b_pad = WIDEN_SIGMAS * h_b_max
    x_edges = np.linspace(u_lo - x_pad, u_hi + x_pad, N_CELLS + 1)
    b_edges = np.linspace(b_lo - b_pad, b_hi + b_pad, N_CELLS + 1)
    return x_edges, b_edges


# ---------------------------------------------------------------------------
# the two smoothing operators: log10 B (tile-only) and log10 x (per width
# node, no shift), each one dense (N_CELLS, N_CELLS) matrix.
# ---------------------------------------------------------------------------

def gaussian_smoothing_matrix(edges, sigma):
    """The bin-integrated Gaussian smoothing operator on `edges`' own
    bins: mass placed at a bin's own centre spreads into every output
    bin by the Gaussian's cumulative distribution across that bin's
    edges. Mass the tabulation box cannot hold becomes part of the
    declared analytic tail (`finalise_node_shape`). `sigma <= 0` is the
    identity."""
    if not (sigma > 0.0) or not np.isfinite(sigma):
        return np.eye(edges.size - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    z = (edges[:, None] - centers[None, :]) / sigma
    return np.diff(ndtr(z), axis=0)


def apply_b_smoothing(m_b, raw):
    """`m_b` applied along the `log10 B` axis (axis 1) of `raw`, `(n_x,
    n_b)`: `log10 B` smoothing does not depend on the shape node, so it
    is applied once per tile, before the per-node `log10 x` operator
    sees the histogram."""
    moved = np.moveaxis(raw, 1, 0)
    shape = moved.shape
    out = m_b.dot(moved.reshape(shape[0], -1)).reshape((m_b.shape[0],) + shape[1:])
    return np.moveaxis(out, 0, 1)


def _bicubic_reconstruct(coarse, coarse_x_centers, coarse_b_centers,
                         fine_x_centers, fine_b_centers):
    """The coarse `(log10 x, log10 B)` density read back at every fine
    grid centre by a BICUBIC spline (`scipy.interpolate.RectBivariateSpline`,
    `kx = ky = BICUBIC_DEGREE`) -- the same interpolation `ClassShape`
    evaluates a source at (`_eval_interior`), so the residual report
    measures exactly what a consumer would see."""
    spline = RectBivariateSpline(coarse_x_centers, coarse_b_centers, coarse,
                                 kx=BICUBIC_DEGREE, ky=BICUBIC_DEGREE)
    return spline(fine_x_centers, fine_b_centers)


# ---------------------------------------------------------------------------
# analytic tails beyond the tabulated box, two-sided on log10 x as well as
# log10 B
# ---------------------------------------------------------------------------

#: The floor on `_edge_tail`'s decay LENGTH, as a fraction of the axis's
#: own tabulated range: a tail that has not decayed within a quarter of
#: the box's own extent is not "the mass the fixed grid cannot hold"
#: (module docstring) any more, it is most of the distribution, so the
#: magnitude is floored to decay within this fraction instead.
_EDGE_TAIL_MAX_DECAY_FRACTION = 0.25

#: How many of the outermost cells `_edge_tail` fits its magnitude from
#: -- more than the bare two, so one noisy adjacent pair cannot set it.
_EDGE_TAIL_FIT_CELLS = 4


def _edge_tail(marginal, grid, cell, side):
    """One edge's exponential tail, `(slope, mass)`. The SIGN is fixed by
    `side` alone -- `"hi"` always decays negative (outward, rightward),
    `"lo"` always positive -- never inferred from the data and never
    allowed to flip, so a noisy or non-monotonic edge cannot declare a
    tail that GROWS away from the grid. The MAGNITUDE is the median
    log-slope over the outermost `_EDGE_TAIL_FIT_CELLS` cells (robust to
    one noisy adjacent pair), floored so the tail decays within
    `_EDGE_TAIL_MAX_DECAY_FRACTION` of the axis's own tabulated range --
    a bound on the tail's own mass, not a fit. Where the edge cell's own
    value is zero (or negative -- CIC/smoothing noise), there is nothing
    to anchor a tail to and it is exactly zero. `marginal` has already
    summed the OTHER axis, so its value at a centre is a BIN MASS over
    one cell of width `cell` on THIS axis, not a density -- dividing by
    `cell` is the same bin-mass-to-density conversion a pointwise
    `ClassShape.density` read applies, and it is what the mass integral
    below needs to be a probability rather than a bin count. The
    declared mass integrates the model from the TABULATED EDGE -- half a
    cell beyond the outermost centre -- to infinity, not from the centre
    itself, so it does not double the mass that bin already carries;
    `ClassShape._eval_node` reads the same centre value and the same
    slope and reaches the same edge amplitude by the same algebra, so
    declaration and evaluation are one model, not two."""
    n_fit = min(_EDGE_TAIL_FIT_CELLS, marginal.size)
    if side == "hi":
        vs, gs = marginal[-n_fit:], grid[-n_fit:]
        v_edge = vs[-1]
    else:
        vs, gs = marginal[:n_fit], grid[:n_fit]
        v_edge = vs[0]
    if not (v_edge > 0.0):
        return 0.0, 0.0

    valid = vs > 0.0
    if valid.sum() >= 2:
        raw_slopes = np.diff(np.log(vs[valid])) / np.diff(gs[valid])
        magnitude = float(np.median(np.abs(raw_slopes)))
    else:
        magnitude = 0.0
    grid_range = float(grid[-1] - grid[0])
    if grid_range > 0.0:
        magnitude = max(magnitude, 1.0 / (_EDGE_TAIL_MAX_DECAY_FRACTION * grid_range))
    slope = -magnitude if side == "hi" else magnitude

    half_cell = 0.5 * cell
    if side == "hi":
        density_at_edge = (v_edge / cell) * np.exp(slope * half_cell)
        mass = density_at_edge / (-slope)
    else:
        density_at_edge = (v_edge / cell) * np.exp(-slope * half_cell)
        mass = density_at_edge / slope
    return slope, float(mass)


def finalise_node_shape(conv, x_centers, b_centers):
    """The stored per-(tile, node[, limit]) product: declares the
    analytic tail from the convolved array's own `(log10 x, log10 B)`
    edge marginals, two-sided on `log10 x` as well as `log10 B`, then
    rescales the interior array so interior mass plus declared tail mass
    sums to exactly one (algebraic acceptance: `DENSITY.sum() == 1 -
    MASS_OUTSIDE`)."""
    x_cell = float(x_centers[1] - x_centers[0])
    b_cell = float(b_centers[1] - b_centers[0])
    x_marg = conv.sum(axis=1)
    b_marg = conv.sum(axis=0)
    x_lo_slope, x_lo_mass = _edge_tail(x_marg, x_centers, x_cell, "lo")
    x_hi_slope, x_hi_mass = _edge_tail(x_marg, x_centers, x_cell, "hi")
    b_lo_slope, b_lo_mass = _edge_tail(b_marg, b_centers, b_cell, "lo")
    b_hi_slope, b_hi_mass = _edge_tail(b_marg, b_centers, b_cell, "hi")
    tail_mass = x_lo_mass + x_hi_mass + b_lo_mass + b_hi_mass
    total = float(conv.sum()) + tail_mass
    density = (conv / total) if total > 0.0 else conv
    mass_outside = tail_mass / total if total > 0.0 else 0.0
    return (density.astype(np.float32), x_lo_slope, x_hi_slope,
           b_lo_slope, b_hi_slope, mass_outside)


# ---------------------------------------------------------------------------
# one region and class, end to end
# ---------------------------------------------------------------------------

def _tile_class_densities(density_ds, t, pop, cls, width_ladder, x_edges, x_centers, b_centers,
                          b_edges, h_x, h_b):
    """One tile's own finalised density at every fixed width node (PAHC:
    every width and limit), written straight into the pre-sized HDF5
    dataset one slice at a time. No shift or map-class dependence: every
    node is the tile's own histogram smoothed by `sqrt(h_x**2 + w_k**2)`
    along `log10 x`."""
    m_b = gaussian_smoothing_matrix(b_edges, h_b)
    n_node = width_ladder.size

    if cls == "pahc":
        tile = pop["tiles"][t]
        n_limit = pop["limit8_grid_mjy"].size
        raws = [apply_b_smoothing(m_b, pahc_raw_hist_for_limit(tile, k, x_centers, b_centers))
               for k in range(n_limit)]
        tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi, mass_outside = (
            np.empty((n_node, n_limit)) for _ in range(5))
        for j, w_k in enumerate(width_ladder):
            sigma = float(np.sqrt(h_x ** 2 + w_k ** 2))
            mx = gaussian_smoothing_matrix(x_edges, sigma)
            for k in range(n_limit):
                conv = mx.dot(raws[k])
                d, sxlo, sxhi, slo, shi, mo = finalise_node_shape(conv, x_centers, b_centers)
                density_ds[t, j, k] = d
                tail_x_lo[j, k], tail_x_hi[j, k], tail_b_lo[j, k], tail_b_hi[j, k], \
                    mass_outside[j, k] = sxlo, sxhi, slo, shi, mo
        return tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi, mass_outside

    raw = apply_b_smoothing(m_b, class_raw_hist(pop, t, cls, x_centers, b_centers))
    tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi, mass_outside = (
        np.empty(n_node) for _ in range(5))
    for j, w_k in enumerate(width_ladder):
        sigma = float(np.sqrt(h_x ** 2 + w_k ** 2))
        mx = gaussian_smoothing_matrix(x_edges, sigma)
        conv = mx.dot(raw)
        d, sxlo, sxhi, slo, shi, mo = finalise_node_shape(conv, x_centers, b_centers)
        density_ds[t, j] = d
        tail_x_lo[j], tail_x_hi[j], tail_b_lo[j], tail_b_hi[j], mass_outside[j] = (
            sxlo, sxhi, slo, shi, mo)
    return tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi, mass_outside


def _write_region_class_streaming(config, region, cls, pop, n_tile, map_classes, width_ladder,
                                  x_edges, b_edges, tile_bw, n_jobs):
    """The `DENSITY` dataset streamed one batch of tiles at a time
    (`config.n_jobs` tiles' own density arrays resident together, never
    all of `n_tile`); the tail scales and `MASS_OUTSIDE` are assembled as
    ordinary in-memory arrays, negligible next to `DENSITY`."""
    n_node = width_ladder.size
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    grid_shape = (x_centers.size, b_centers.size)
    path = config_module.product_path(config, "bms", cls, "shape", "tile", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    is_pahc = cls == "pahc"
    n_limit = int(pop["limit8_grid_mjy"].size) if is_pahc else None
    tail_shape = (n_tile, n_node, n_limit) if is_pahc else (n_tile, n_node)

    tail_x_lo = np.empty(tail_shape)
    tail_x_hi = np.empty(tail_shape)
    tail_b_lo = np.empty(tail_shape)
    tail_b_hi = np.empty(tail_shape)
    mass_outside = np.empty(tail_shape)

    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "tile"
        if cls == "agb":
            f.attrs["F_C"] = pop["f_c"]
        f.create_dataset("SHAPE_NODES", data=width_ladder.astype(np.float64))
        f.create_dataset("LOG10_X_EDGES", data=x_edges.astype(np.float64))
        f.create_dataset("LOG10_B_EDGES", data=b_edges.astype(np.float64))
        f.create_dataset("TILE_ID", data=np.arange(n_tile, dtype=np.int64))
        f.create_dataset("MAP_CLASS", data=np.array(map_classes, dtype="S8"))
        if is_pahc:
            f.create_dataset("LIMIT8_GRID_MJY", data=pop["limit8_grid_mjy"].astype(np.float64))
            density_shape = (n_tile, n_node, n_limit) + grid_shape
            chunks = (1, 1, 1) + grid_shape
        else:
            density_shape = (n_tile, n_node) + grid_shape
            chunks = (1, 1) + grid_shape
        density_ds = f.create_dataset(
            "DENSITY", shape=density_shape, dtype=np.float32,
            chunks=chunks, shuffle=True, compression="gzip", compression_opts=1)

        for batch_start in range(0, n_tile, n_jobs):
            batch = range(batch_start, min(batch_start + n_jobs, n_tile))
            batch_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_tile_class_densities)(
                    density_ds, t, pop, cls, width_ladder, x_edges, x_centers, b_centers,
                    b_edges, tile_bw[t]["h_x"], tile_bw[t]["h_b"])
                for t in batch)
            for t, (sxlo, sxhi, slo, shi, mo) in zip(batch, batch_results):
                tail_x_lo[t], tail_x_hi[t], tail_b_lo[t], tail_b_hi[t], mass_outside[t] = (
                    sxlo, sxhi, slo, shi, mo)
            del batch_results

        f.create_dataset("TAIL_X_LO_SCALE", data=tail_x_lo)
        f.create_dataset("TAIL_X_HI_SCALE", data=tail_x_hi)
        f.create_dataset("TAIL_B_LO_SCALE", data=tail_b_lo)
        f.create_dataset("TAIL_B_HI_SCALE", data=tail_b_hi)
        f.create_dataset("MASS_OUTSIDE", data=mass_outside)

    return path, (tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi), mass_outside


def _build_width_density(pop, cls, tile_idx, h_x, h_b, w_k, x_edges, b_edges, pahc_limit_idx=0):
    """One (tile, width)'s convolved, un-finalised `(log10 x, log10 B)`
    array, built directly at `w_k` on the given grid -- what
    `_tile_class_densities` builds, exposed for the residual reports
    below at an arbitrary width and resolution."""
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    if cls == "pahc":
        raw = pahc_raw_hist_for_limit(pop["tiles"][tile_idx], pahc_limit_idx, x_centers, b_centers)
    else:
        raw = class_raw_hist(pop, tile_idx, cls, x_centers, b_centers)
    m_b = gaussian_smoothing_matrix(b_edges, h_b)
    raw = apply_b_smoothing(m_b, raw)
    sigma = float(np.sqrt(h_x ** 2 + w_k ** 2))
    mx = gaussian_smoothing_matrix(x_edges, sigma)
    return mx.dot(raw)


def _bicubic_residual_report(pop, cls, tile0_bw, widths, x_edges, b_edges):
    """Tile 0's density at each of `widths`, built directly on both the
    region's own `N_CELLS` grid and an `N_CELLS_CHECK` grid spanning the
    same range; the relative L1 error of a bicubic reconstruction of the
    fine grid from the coarse one, one number per width
    (`IMPLEMENTATION.md` section 3, report only, never searched)."""
    x_edges_fine = np.linspace(x_edges[0], x_edges[-1], N_CELLS_CHECK + 1)
    b_edges_fine = np.linspace(b_edges[0], b_edges[-1], N_CELLS_CHECK + 1)
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_c = 0.5 * (b_edges[:-1] + b_edges[1:])
    x_c_fine = 0.5 * (x_edges_fine[:-1] + x_edges_fine[1:])
    b_c_fine = 0.5 * (b_edges_fine[:-1] + b_edges_fine[1:])
    group = N_CELLS_CHECK // N_CELLS
    out = []
    for w in widths:
        coarse = _build_width_density(pop, cls, 0, tile0_bw["h_x"], tile0_bw["h_b"], w,
                                      x_edges, b_edges)
        fine = _build_width_density(pop, cls, 0, tile0_bw["h_x"], tile0_bw["h_b"], w,
                                    x_edges_fine, b_edges_fine)
        recon = _bicubic_reconstruct(coarse / (group * group), x_c, b_c, x_c_fine, b_c_fine)
        out.append(_rel_l1(fine, recon))
    return out


def _width_interp_residual_report(pop, cls, tile0_bw, width_ladder, x_edges, b_edges, density0,
                                  widths):
    """The relative L1 between the width blend (`ClassShape.density`'s
    own bracket-and-blend rule) and a direct build, at each of `widths`
    (bracketed in the ladder the same way a read would), tile 0
    (`IMPLEMENTATION.md` section 3, report only, never searched)."""
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_c = 0.5 * (b_edges[:-1] + b_edges[1:])
    log_widths = np.log(width_ladder)
    out = []
    for w in widths:
        i_lo_arr, t_arr = column_grid.bracket(np.array([np.log(w)]), log_widths)
        i_lo, t = int(i_lo_arr[0]), float(t_arr[0])
        i_hi = min(i_lo + 1, width_ladder.size - 1)
        direct = _build_width_density(pop, cls, 0, tile0_bw["h_x"], tile0_bw["h_b"], w,
                                      x_edges, b_edges)
        direct_density = finalise_node_shape(direct, x_c, b_c)[0].astype(np.float64)
        if cls == "pahc":
            blended = ((1.0 - t) * density0[i_lo, 0].astype(np.float64)
                      + t * density0[i_hi, 0].astype(np.float64))
        else:
            blended = ((1.0 - t) * density0[i_lo].astype(np.float64)
                      + t * density0[i_hi].astype(np.float64))
        out.append(_rel_l1(direct_density, blended))
    return out


def median_kernel_width(config, region):
    """The median, over the region's own catalogued sources, of `Kernel.
    params`'s single-Gaussian-equivalent width at each source's own
    column, uncertainty and arm -- the middle of the three widths (the
    ladder floor, this, and the ladder top) the two residual reports
    above are measured at, since the ladder floor and top rarely bracket
    the widths sources actually read."""
    from sesnaimpute.prior import table as table_module
    from sesnaimpute.prior.kernel import Kernel
    src = table_module.read(config, region)
    kern = Kernel.read(config)
    map_class = np.where(src["A_COL_PROVENANCE"] == _PLANCK_PROVENANCE_CODE, "planck", "herschel")
    _, sigma = kern.params(src["A_COL_K"], src["A_COL_SIG_K"], map_class)
    return float(np.median(sigma))


def build_region_class(config, region, cls, shared, width_ladder):
    """One region and class, end to end: the class's own grid
    (`_class_grid`) and per-tile Silverman bandwidths, the per-tile
    evaluation at every fixed width node in parallel
    (`_write_region_class_streaming`), and the two residual reports on
    tile 0."""
    pop, n_tile = shared["pop"], shared["n_tile"]
    map_classes = shared["map_classes"]

    tile_bw = tile_class_bandwidths(pop, cls)
    x_edges, b_edges = _class_grid(pop, cls, tile_bw, width_ladder)

    node_idx_moment = int(width_ladder.size // 2)
    node_idx_exact = 0

    limit_lo_idx = limit_hi_idx = f_lim8_median = None
    if cls == "pahc":
        # IMPLEMENTATION.md section 3: a source's own 8um limit brackets
        # two of the eight stored limit grids; the region's own median
        # real-source limit exercises that bracket for the report's own
        # interpolation check.
        from sesnaimpute.catalog import limits as limits_module
        from sesnaimpute.prior import star_population
        f_lim_i4 = limits_module.limits(config, region)[:, star_population.IDX_I4]
        f_lim8_median = float(np.median(f_lim_i4))
        limit_grid = pop["limit8_grid_mjy"]
        lo_arr, _ = column_grid.bracket(
            np.array([np.log10(f_lim8_median)]), np.log10(limit_grid))
        limit_lo_idx = int(lo_arr[0])
        limit_hi_idx = min(limit_lo_idx + 1, limit_grid.size - 1)

    path, tails, mass_outside = _write_region_class_streaming(
        config, region, cls, pop, n_tile, map_classes, width_ladder, x_edges, b_edges,
        tile_bw, config.n_jobs)

    with h5py.File(path, "r") as f:
        density0 = f["DENSITY"][0]

    report_widths = [float(width_ladder[0]), shared["median_kernel_width"], float(width_ladder[-1])]
    bicubic_rel_l1 = _bicubic_residual_report(pop, cls, tile_bw[0], report_widths, x_edges, b_edges)
    width_interp_rel_l1 = _width_interp_residual_report(
        pop, cls, tile_bw[0], width_ladder, x_edges, b_edges, density0, report_widths)

    return dict(
        region=region, cls=cls, shape_nodes=width_ladder, x_edges=x_edges, b_edges=b_edges,
        density0=density0, node_idx_moment=node_idx_moment, node_idx_exact=node_idx_exact,
        tail_x_lo=tails[0], tail_x_hi=tails[1], tail_b_lo=tails[2], tail_b_hi=tails[3],
        mass_outside=mass_outside, map_classes=map_classes, n_tile=n_tile,
        f_c=pop["f_c"], pop=pop, path=path, h_x0=tile_bw[0]["h_x"],
        pahc_limit_lo_idx=limit_lo_idx, pahc_limit_hi_idx=limit_hi_idx,
        pahc_limit_median_mjy=f_lim8_median,
        report_widths=report_widths,
        bicubic_rel_l1=bicubic_rel_l1, width_interp_rel_l1=width_interp_rel_l1,
    )


# ---------------------------------------------------------------------------
# read: the evaluator every prior-table build calls
# ---------------------------------------------------------------------------

class _FixedKernel(object):
    """A stand-in `Kernel` for the acceptance checks below: fixes `mu=0`
    and `sigma` to a chosen ladder width regardless of `a_col`, so a
    query lands exactly on a ladder entry -- independent of the real
    column kernel, which the checks below are not testing. Both mixture
    components are given the same `(mu, sigma)`, so `mixture`'s weight
    does not matter -- the two-component read reduces to the same single
    read `params` describes."""

    def __init__(self, sigma_val):
        self.sigma_val = float(sigma_val)

    def params(self, a_col, sigma_col, map_class):
        a_col = np.asarray(a_col, dtype=np.float64)
        return np.zeros_like(a_col), np.full_like(a_col, self.sigma_val)

    def mixture(self, a_col, sigma_col, map_class):
        a_col = np.asarray(a_col, dtype=np.float64)
        w = np.full(a_col.shape, 0.5)
        mu = np.zeros(a_col.shape + (2,))
        sigma = np.full(a_col.shape + (2,), self.sigma_val)
        return w, mu, sigma


class ClassShape(object):
    """One class's per-tile shape, evaluated per source (`IMPLEMENTATION.md`
    section 3's evaluation column): the tile's own density, read once per
    component of the source's own two-component kernel mixture `(w, mu,
    sigma) = Kernel.mixture(a_col, sigma_col, map_class)` and combined `w
    * D_1 + (1 - w) * D_2`. Each component's own `sigma` is bracketed in
    the width ladder and blended, BICUBIC in `(log10(a/a_col) - mu, log10
    B)` inside the grid, the analytic tail outside on either axis, `a < 0`
    (or `a == 0`, which has no `log10 x`) mapped to the low-`x` tail or
    zero. PAHC additionally carries a limit grid (`limit_grid_mjy`): a
    source's own 8 micron limit brackets two of the eight stored limit
    grids and blends between them linearly in `log10` limit -- the four
    (width, limit) corners are evaluated and bilinearly combined, per
    mixture component. Vectorised over sources."""

    def __init__(self, cls, shape_nodes, x_edges, b_edges, density, tail_x_lo, tail_x_hi,
                 tail_b_lo, tail_b_hi, mass_outside, kern, limit_grid_mjy=None):
        self.cls = cls
        self.shape_nodes = np.asarray(shape_nodes, dtype=np.float64)  # the width ladder
        self.x_edges = np.asarray(x_edges, dtype=np.float64)  # log10 x
        self.b_edges = np.asarray(b_edges, dtype=np.float64)  # log10 B
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.b_centers = 0.5 * (self.b_edges[:-1] + self.b_edges[1:])
        self.density_table = np.asarray(density, dtype=np.float64)
        self.tail_x_lo = np.asarray(tail_x_lo, dtype=np.float64)
        self.tail_x_hi = np.asarray(tail_x_hi, dtype=np.float64)
        self.tail_b_lo = np.asarray(tail_b_lo, dtype=np.float64)
        self.tail_b_hi = np.asarray(tail_b_hi, dtype=np.float64)
        self.mass_outside = np.asarray(mass_outside, dtype=np.float64)
        self.kern = kern
        # PAHC only: log10 of the eight stored 8um completeness-limit
        # grid values, sorted ascending.
        self.limit_log = (None if limit_grid_mjy is None else
                          np.log10(np.asarray(limit_grid_mjy, dtype=np.float64)))
        # running count of sources whose sigma_s fell outside the width
        # ladder and had to be clamped at either end.
        self.n_clamp_lo = 0
        self.n_clamp_hi = 0

    def _slab(self, tile_idx, node_idx, limit_idx):
        """The stored `(log10 x, log10 B)` array at one (tile, width[,
        limit]) -- `limit_idx` is `None` for STAR/AGB, an integer index
        into the limit grid for PAHC."""
        if limit_idx is None:
            return self.density_table[tile_idx, node_idx]
        return self.density_table[tile_idx, node_idx, limit_idx]

    def _eval_interior(self, tile_ids, node_idx, log_x, logb, limit_idx):
        n = log_x.shape[0]
        out = np.empty(n, dtype=np.float64)
        # grouped by (tile, width[, limit]): a query slab is built, and
        # its bicubic spline fitted, once per group, not per source.
        if limit_idx is None:
            keys = tile_ids.astype(np.int64) * (self.shape_nodes.size + 1) + node_idx
        else:
            keys = ((tile_ids.astype(np.int64) * (self.shape_nodes.size + 1) + node_idx)
                    * (self.limit_log.size + 1) + limit_idx)
        for key in np.unique(keys):
            sel = keys == key
            t_i = int(tile_ids[sel][0])
            n_i = int(node_idx[sel][0])
            l_i = int(limit_idx[sel][0]) if limit_idx is not None else None
            slab = self._slab(t_i, n_i, l_i)
            spline = RectBivariateSpline(self.x_centers, self.b_centers, slab,
                                         kx=BICUBIC_DEGREE, ky=BICUBIC_DEGREE)
            out[sel] = spline.ev(log_x[sel], logb[sel])
        # a density cannot be negative; a cubic spline can ring slightly
        # below zero near a sharp edge (the non-negativity identity, not
        # a tuned threshold).
        return np.clip(out, 0.0, None)

    def _eval_node(self, tile_ids, node_idx, log_x, logb, limit_idx):
        """One width node's (PAHC: one width-and-limit-grid's) value,
        blending the interior bicubic read with the declared analytic
        tail beyond either `log10 x` edge or either `log10 B` edge
        (`finalise_node_shape`). The tail amplitude is read at the
        tabulated grid's own outermost bin CENTRE -- never extrapolated
        past the bicubic fit's own support -- and the exponent is
        referenced from that same centre, matching `_edge_tail`'s own
        declaration exactly, so the mass this evaluator delivers when
        integrated is the mass `finalise_node_shape` declared as `MASS_
        OUTSIDE`."""
        interior = self._eval_interior(tile_ids, node_idx, log_x, logb, limit_idx)
        x_lo = log_x < self.x_edges[0]
        x_hi = log_x > self.x_edges[-1]
        b_lo = logb < self.b_edges[0]
        b_hi = logb > self.b_edges[-1]
        if not (x_lo.any() or x_hi.any() or b_lo.any() or b_hi.any()):
            return interior
        # a corner (both axes out of range) reads the tail along the axis
        # that triggers first in the mutually exclusive chain below (x
        # over b, matching `finalise_node_shape`'s own two independent,
        # one-axis-at-a-time tail declarations) -- the OTHER axis's own
        # query value, clipped into the box, is what that axis's
        # amplitude is read at.
        centre_x = np.where(x_lo, self.x_centers[0], np.where(x_hi, self.x_centers[-1], log_x))
        centre_b = np.where(b_lo, self.b_centers[0], np.where(b_hi, self.b_centers[-1], logb))
        clip_b = np.clip(logb, self.b_edges[0], self.b_edges[-1])
        clip_x = np.clip(log_x, self.x_edges[0], self.x_edges[-1])
        amp_x = self._eval_interior(tile_ids, node_idx, centre_x, clip_b, limit_idx)
        amp_b = self._eval_interior(tile_ids, node_idx, clip_x, centre_b, limit_idx)
        if limit_idx is None:
            idx = (tile_ids, node_idx)
        else:
            idx = (tile_ids, node_idx, limit_idx)
        txlo, txhi = self.tail_x_lo[idx], self.tail_x_hi[idx]
        tblo, tbhi = self.tail_b_lo[idx], self.tail_b_hi[idx]
        out = interior.copy()
        out = np.where(x_lo, amp_x * np.exp(txlo * (log_x - self.x_centers[0])), out)
        out = np.where(x_hi & ~x_lo, amp_x * np.exp(txhi * (log_x - self.x_centers[-1])), out)
        out = np.where(b_lo & ~x_lo & ~x_hi,
                      amp_b * np.exp(tblo * (logb - self.b_centers[0])), out)
        out = np.where(b_hi & ~x_lo & ~x_hi,
                      amp_b * np.exp(tbhi * (logb - self.b_centers[-1])), out)
        return out

    def _component_density(self, log_x_raw, log10_b, tile_ids, mu_c, sigma_c, f_lim8):
        """One mixture component's bracket-and-blend read: the component's
        own shift `mu_c` moves the query (`log_x = log_x_raw - mu_c`),
        the component's own total width `sigma_c` is bracketed in the
        width ladder (log-linear blend, clamped at the ends -- clamps
        counted in `n_clamp_lo`/`n_clamp_hi`) and the bicubic read taken
        at `(log_x, log10 B)` (PAHC: crossed with the two bracketing
        limit grids, `f_lim8` this source's own 8 micron limit, blended
        in `log10` limit). Shared by both components of the kernel
        mixture in `density`."""
        log_x = log_x_raw - mu_c

        log_widths = np.log(self.shape_nodes)
        i_lo, t_w = column_grid.bracket(np.log(sigma_c), log_widths)
        self.n_clamp_lo += int(np.sum(sigma_c < self.shape_nodes[0]))
        self.n_clamp_hi += int(np.sum(sigma_c > self.shape_nodes[-1]))
        i_hi = np.minimum(i_lo + 1, self.shape_nodes.size - 1)

        if self.limit_log is None:
            val_lo = self._eval_node(tile_ids, i_lo, log_x, log10_b, None)
            val_hi = self._eval_node(tile_ids, i_hi, log_x, log10_b, None)
        else:
            log_f = np.broadcast_to(
                np.log10(np.asarray(f_lim8, dtype=np.float64)), log_x.shape)
            m_lo, t_limit = column_grid.bracket(log_f, self.limit_log)
            m_hi = np.minimum(m_lo + 1, self.limit_log.size - 1)
            v_lo_lo = self._eval_node(tile_ids, i_lo, log_x, log10_b, m_lo)
            v_lo_hi = self._eval_node(tile_ids, i_lo, log_x, log10_b, m_hi)
            v_hi_lo = self._eval_node(tile_ids, i_hi, log_x, log10_b, m_lo)
            v_hi_hi = self._eval_node(tile_ids, i_hi, log_x, log10_b, m_hi)
            val_lo = (1.0 - t_limit) * v_lo_lo + t_limit * v_lo_hi
            val_hi = (1.0 - t_limit) * v_hi_lo + t_limit * v_hi_hi
        return (1.0 - t_w) * val_lo + t_w * val_hi

    def density(self, a, log10_b, tile_ids, a_col, sigma_col, map_class, f_lim8=None):
        """`density(a, log10_b, tile_ids, a_col, sigma_col, map_class[,
        f_lim8])`: per source `(w, mu, sigma) = Kernel.mixture(a_col,
        sigma_col, map_class)`, `mu`/`sigma` shape `(n, 2)` -- each
        component's own sub-beam width composed in quadrature with the
        source's own measurement uncertainty and, for Herschel, the field
        zero point. Each component is read separately
        (`_component_density`: its own shift and width-ladder bracket)
        and the two combined `w * D_1 + (1 - w) * D_2`. `a < 0` mapped to
        zero (`a == 0`, having no `log10 x`, reads as the declared
        low-`x` tail's own limit)."""
        a = np.asarray(a, dtype=np.float64)
        log10_b = np.asarray(log10_b, dtype=np.float64)
        a_col = np.asarray(a_col, dtype=np.float64)
        tile_ids = np.asarray(tile_ids, dtype=np.int64)
        ok = a >= 0.0
        x_lin = np.zeros_like(a)
        x_lin[ok] = a[ok] / a_col[ok]
        # a sentinel far below any tabulated x_edges[0] -- exp() of the
        # (positive) low-side tail slope times this offset underflows to
        # 0.0 harmlessly, mapping a == 0 to "below x_min".
        log_x_raw = np.where(x_lin > 0.0, np.log10(np.clip(x_lin, _LOG_FLOOR, None)),
                             self.x_edges[0] - 1.0e3)

        w, mu, sigma = self.kern.mixture(a_col, sigma_col, map_class)
        val_1 = self._component_density(log_x_raw, log10_b, tile_ids, mu[:, 0], sigma[:, 0], f_lim8)
        val_2 = self._component_density(log_x_raw, log10_b, tile_ids, mu[:, 1], sigma[:, 1], f_lim8)
        out = w * val_1 + (1.0 - w) * val_2
        out[~ok] = 0.0
        return out


def read(config, region, cls):
    """The stored per-tile shape as a `ClassShape`, with the survey-wide
    column kernel attached for the shift/width `Kernel.params` needs at
    read time."""
    path = config_module.product_path(config, "bms", cls, "shape", "tile", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.star_shapes: no %r shape for region %r at %s -- "
            "run the 'prior.star_shapes' RUNBOOK line first" % (cls, region, path))
    with h5py.File(path, "r") as f:
        shape_nodes = f["SHAPE_NODES"][:]
        x_edges = f["LOG10_X_EDGES"][:]
        b_edges = f["LOG10_B_EDGES"][:]
        density = f["DENSITY"][:]
        tail_x_lo = f["TAIL_X_LO_SCALE"][:]
        tail_x_hi = f["TAIL_X_HI_SCALE"][:]
        tail_b_lo = f["TAIL_B_LO_SCALE"][:]
        tail_b_hi = f["TAIL_B_HI_SCALE"][:]
        mass_outside = f["MASS_OUTSIDE"][:]
        limit_grid_mjy = f["LIMIT8_GRID_MJY"][:] if "LIMIT8_GRID_MJY" in f else None
    kern = Kernel.read(config)
    return ClassShape(cls, shape_nodes, x_edges, b_edges, density, tail_x_lo, tail_x_hi,
                      tail_b_lo, tail_b_hi, mass_outside, kern, limit_grid_mjy=limit_grid_mjy)


# ---------------------------------------------------------------------------
# checks (report only)
# ---------------------------------------------------------------------------

def _mean_moment_check(config, region, cls, result):
    """Acceptance: node `k`'s std GROWTH over the raw (unsmoothed-on-x,
    b-smoothed-only) input equals `sqrt(h_x**2 + w_k**2)` exactly (no
    shift is baked into a node any more, so the mean should not move at
    all) -- measured on the first tile, at a representative middle
    ladder entry (PAHC: at the stored limit grid bracketing the region's
    own median source limit)."""
    pop = result["pop"]
    tile0 = pop["tiles"][0]
    node_idx = result["node_idx_moment"]
    w_k = float(result["shape_nodes"][node_idx])
    h_x0 = result["h_x0"]

    if cls == "star":
        w, log_u = tile0["w_star"], tile0["log10_u"]
        density = result["density0"][node_idx]
    elif cls == "agb":
        ev = tile0["is_evolved"]
        w, log_u = tile0["w_agb"][ev], tile0["log10_u"][ev]
        density = result["density0"][node_idx]
    else:
        limit_idx = result["pahc_limit_lo_idx"]
        w = tile0["w"] * tile0["p_pahc"][:, limit_idx]
        log_u = tile0["log10_u"]
        density = result["density0"][node_idx, limit_idx]

    raw_mean, raw_std = _weighted_mean_std(log_u, w)
    x_centers = 0.5 * (result["x_edges"][:-1] + result["x_edges"][1:])
    x_marg = density.astype(np.float64).sum(axis=1)
    node_mean, node_std = _weighted_mean_std(x_centers, x_marg)

    mean_dev = abs(node_mean - raw_mean)
    expected_var_growth = h_x0 ** 2 + w_k ** 2
    measured_var_growth = node_std ** 2 - raw_std ** 2
    return w_k, expected_var_growth, measured_var_growth, mean_dev


def _grid_centre_exact_check(result):
    """Acceptance: the evaluator at a grid centre, with a source whose
    `sigma_s` equals a ladder width exactly (`_FixedKernel` forces `mu=0`
    and that exact `sigma`), reproduces the stored value exactly (PAHC
    additionally: at a source's own 8um limit equal to one of the eight
    stored limit grids). A second, duplicated dummy width (PAHC: also a
    duplicated dummy limit) gives `column_grid.bracket` a valid span; the
    query sits exactly on the first of each pair, so the duplicate's
    interpolation weight is always zero."""
    shape_nodes, x_edges, b_edges = result["shape_nodes"], result["x_edges"], result["b_edges"]
    node_idx = result["node_idx_exact"]
    is_pahc = result["cls"] == "pahc"
    w_k = float(shape_nodes[node_idx])
    shape_nodes_pair = np.array([w_k, w_k * (1.0 + 1.0e-9)])

    if is_pahc:
        limit_idx = result["pahc_limit_lo_idx"]
        d = result["density0"][node_idx, limit_idx]
        limit_val = float(result["pop"]["limit8_grid_mjy"][limit_idx])
        limit_grid_pair = np.array([limit_val, limit_val * (1.0 + 1.0e-9)])
        density_pair = np.empty((1, 2, 2) + d.shape, dtype=np.float32)
        density_pair[0, :, :] = d
        tx_lo = np.full((1, 2, 2), result["tail_x_lo"][0, node_idx, limit_idx])
        tx_hi = np.full((1, 2, 2), result["tail_x_hi"][0, node_idx, limit_idx])
        tb_lo = np.full((1, 2, 2), result["tail_b_lo"][0, node_idx, limit_idx])
        tb_hi = np.full((1, 2, 2), result["tail_b_hi"][0, node_idx, limit_idx])
        mo = np.full((1, 2, 2), result["mass_outside"][0, node_idx, limit_idx])
    else:
        d = result["density0"][node_idx]
        limit_grid_pair = None
        density_pair = np.stack([d, d])[np.newaxis]  # (1 tile, 2 width, n_x, n_b)
        tx_lo = np.full((1, 2), result["tail_x_lo"][0, node_idx])
        tx_hi = np.full((1, 2), result["tail_x_hi"][0, node_idx])
        tb_lo = np.full((1, 2), result["tail_b_lo"][0, node_idx])
        tb_hi = np.full((1, 2), result["tail_b_hi"][0, node_idx])
        mo = np.full((1, 2), result["mass_outside"][0, node_idx])

    shape = ClassShape(result["cls"], shape_nodes_pair, x_edges, b_edges, density_pair,
                       tx_lo, tx_hi, tb_lo, tb_hi, mo, _FixedKernel(w_k),
                       limit_grid_mjy=limit_grid_pair)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    a_query = w_k * (10.0 ** x_centers)
    tile_ids = np.zeros(a_query.size, dtype=np.int64)
    a_col = np.full(a_query.size, w_k)
    sigma_col = np.zeros(a_query.size)
    f_lim8 = np.full(a_query.size, limit_val) if is_pahc else None
    got = shape.density(a_query, np.full(a_query.size, b_centers[0]), tile_ids, a_col,
                          sigma_col, "planck", f_lim8=f_lim8)
    expected = d[:, 0].astype(np.float64)
    return float(np.max(np.abs(got - expected)))


def _pahc_between_check(config, region, result):
    """Acceptance (PAHC only): a read at the region's own median source
    8um limit lies between the two limit grids it brackets, cell by cell
    -- convexity of the linear blend in `log10` limit, at a width held
    exactly on a tabulated ladder entry (`_FixedKernel`) so only the
    limit axis is exercised."""
    node_idx = result["node_idx_exact"]
    i_lo, i_hi = result["pahc_limit_lo_idx"], result["pahc_limit_hi_idx"]
    f_lim8_median = result["pahc_limit_median_mjy"]
    limit_grid = result["pop"]["limit8_grid_mjy"]
    d_lo = result["density0"][node_idx, i_lo]
    d_hi = result["density0"][node_idx, i_hi]

    shape_nodes, x_edges, b_edges = result["shape_nodes"], result["x_edges"], result["b_edges"]
    w_k = float(shape_nodes[node_idx])
    shape_nodes_pair = np.array([w_k, w_k * (1.0 + 1.0e-9)])
    limit_grid_pair = np.array([float(limit_grid[i_lo]), float(limit_grid[i_hi])])
    density_pair = np.empty((1, 2, 2) + d_lo.shape, dtype=np.float32)
    density_pair[0, :, 0] = d_lo
    density_pair[0, :, 1] = d_hi
    tx_lo = np.stack([np.full((1, 2), result["tail_x_lo"][0, node_idx, i_lo]),
                      np.full((1, 2), result["tail_x_lo"][0, node_idx, i_hi])], axis=-1)
    tx_hi = np.stack([np.full((1, 2), result["tail_x_hi"][0, node_idx, i_lo]),
                      np.full((1, 2), result["tail_x_hi"][0, node_idx, i_hi])], axis=-1)
    tb_lo = np.stack([np.full((1, 2), result["tail_b_lo"][0, node_idx, i_lo]),
                      np.full((1, 2), result["tail_b_lo"][0, node_idx, i_hi])], axis=-1)
    tb_hi = np.stack([np.full((1, 2), result["tail_b_hi"][0, node_idx, i_lo]),
                      np.full((1, 2), result["tail_b_hi"][0, node_idx, i_hi])], axis=-1)
    mo = np.stack([np.full((1, 2), result["mass_outside"][0, node_idx, i_lo]),
                   np.full((1, 2), result["mass_outside"][0, node_idx, i_hi])], axis=-1)

    shape = ClassShape("pahc", shape_nodes_pair, x_edges, b_edges, density_pair,
                       tx_lo, tx_hi, tb_lo, tb_hi, mo, _FixedKernel(w_k),
                       limit_grid_mjy=limit_grid_pair)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    a_query = w_k * (10.0 ** x_centers)
    tile_ids = np.zeros(a_query.size, dtype=np.int64)
    a_col = np.full(a_query.size, w_k)
    sigma_col = np.zeros(a_query.size)
    f_lim8 = np.full(a_query.size, f_lim8_median)
    got = shape.density(a_query, np.full(a_query.size, b_centers[0]), tile_ids, a_col,
                          sigma_col, "planck", f_lim8=f_lim8)
    lo_vals = d_lo[:, 0].astype(np.float64)
    hi_vals = d_hi[:, 0].astype(np.float64)
    cell_lo = np.minimum(lo_vals, hi_vals)
    cell_hi = np.maximum(lo_vals, hi_vals)
    violation = np.maximum(cell_lo - got, got - cell_hi)
    t_limit = float(np.clip(
        (np.log10(f_lim8_median) - np.log10(limit_grid[i_lo]))
        / (np.log10(limit_grid[i_hi]) - np.log10(limit_grid[i_lo])), 0.0, 1.0))
    return dict(f_lim8_median=f_lim8_median, limit_lo_mjy=float(limit_grid[i_lo]),
               limit_hi_mjy=float(limit_grid[i_hi]), t_limit=t_limit,
               max_violation=float(np.max(violation)))


def build(config, regions=None):
    """Writes the per-tile STAR/AGB/PAHC shape products for `regions`
    (default: all thirty), one file per region and class. The width
    ladder is fixed and survey-wide; the build itself needs no kernel --
    the shift and total width are read-time-only (`ClassShape.density`)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    width_ladder = SHAPE_WIDTH_LADDER

    for region in region_names:
        pop = read_population(config, region)
        n_tile = pop["n_tile"]
        map_classes = tile_map_classes(config, region, n_tile)
        shared = dict(pop=pop, n_tile=n_tile, map_classes=map_classes,
                      median_kernel_width=median_kernel_width(config, region))

        for cls in CLASSES:
            result = build_region_class(config, region, cls, shared, width_ladder)
            path = result["path"]
            w_k, expected_growth, measured_growth, mean_dev = _mean_moment_check(
                config, region, cls, result)
            exact_dev = _grid_centre_exact_check(result)
            pahc_line = ""
            if cls == "pahc":
                between = _pahc_between_check(config, region, result)
                pahc_line = (
                    " pahc_median_between(f_lim8_median=%.4g mJy, bracket=[%.4g,%.4g] mJy, "
                    "t=%.3f, max_violation=%.2e)"
                    % (between["f_lim8_median"], between["limit_lo_mjy"],
                       between["limit_hi_mjy"], between["t_limit"], between["max_violation"]))
            n_x = result["x_edges"].size - 1
            n_b = result["b_edges"].size - 1
            width_labels = ("floor", "median_kernel", "top")
            residual_line = " ".join(
                "bicubic_rel_l1[%s w=%.4f]=%.4f width_interp_rel_l1[%s w=%.4f]=%.4f"
                % (label, w, bic, label, w, wid)
                for label, w, bic, wid in zip(
                    width_labels, result["report_widths"],
                    result["bicubic_rel_l1"], result["width_interp_rel_l1"]))
            print(
                "star_shapes: %s/%s: grid=%dx%d widths=%d "
                "median_mass_outside=%.4e std_growth(w=%.4f expected_var=%.5f "
                "measured_var=%.5f mean_dev=%.2e dex) "
                "grid_centre_exact_max_dev=%.2e %s (bar=%.2f)%s -> %s"
                % (region, cls, n_x, n_b, width_ladder.size,
                   float(np.median(result["mass_outside"])), w_k, expected_growth,
                   measured_growth, mean_dev, exact_dev,
                   residual_line, EPS_SHAPE,
                   pahc_line, path), flush=True)


if __name__ == "__main__":
    run(build)
