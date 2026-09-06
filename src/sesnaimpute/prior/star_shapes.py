"""Per-tile STAR/AGB/PAHC shapes (SPEC_PRIORS.md section 2.2's "The
shape", section 3's AGB blend, section 4's PAHC shape; `IMPLEMENTATION.md`
section 3, the STAR/AGB and PAHC storage rows, amended 2026-09-05 for the
log-spaced `x` axis).

The population is `prior.star_population`'s per-tile point cloud: every
retained field star carries `U` (the scaled extinction `x = a/A` -- at
column `A` the star sits at `a = A*U`, so `x = U` exactly) and a class
weight (`W_STAR`, `W_AGB`, or the whole `W` for PAHC). At a shape node of
column `A_node`, the true column is `a = T*U` for `T` a draw from the
column kernel's own `p(T | A_node)` (`prior.kernel.Kernel.nodes`), so the
node's own `x = a/A_node = (T/A_node)*U = r*U`, `r = T/A_node`. In
`log10`: `log x = log r + log U` -- a SUM of two independent quantities,
so the node's kernel is a plain SHIFT-INVARIANT CONVOLUTION along `log10
x` of the population's own `log10 U` histogram with the kernel's `log10
r` distribution. This is why the axis is `log10 x`, not `x`: no `(n_x,
n_x)` matrix is built anywhere in this module: every node's operator is
one small 1-D array (`node_log_x_kernel`), applied to the whole
`(log10 x, log10 B)` histogram by one FFT convolution along axis 0
(`apply_log_x_kernel`), batched over every `log10 B` column at once.

**The smoothing width comes from the population, not the fitter's own
resolution** (`IMPLEMENTATION.md` section 3, amended 2026-09-05). Per
tile and class, Silverman's rule for a 2-D kernel density estimate
(Silverman 1986, `SILVERMAN_EXPONENT`) sets one bandwidth per axis,
`h = sigma * n**(-1/6)`, with `sigma` the tile's own weighted standard
deviation on that axis under the class's own weight (`log10 u` -- this
class's `log10 x` at the reference node `r = T/A = 1` -- for `h_x`,
`log10 B` for `h_b`) and `n` the tile's own effective star count,
`(Sum w)**2 / Sum w**2` (`silverman_bandwidth`, `tile_class_bandwidths`).
Two smoothings are applied to the deposited histogram before any node's
shape is stored:

- on `log10 B`, a Gaussian of the tile's own `h_b` (one operator per
  tile and class, class-independent of node, applied once per tile
  before the per-node step below: `apply_b_smoothing`);
- on `log10 x`, at each node, a Gaussian whose width is the tile's own
  `h_x` and the node's own kernel-quadrature width added in QUADRATURE,
  `sqrt(h_x**2 + kernel_width**2)` -- `kernel_width` is the node's own
  kernel's weighted standard deviation of `log10 r`, the floor below
  which the node's own kernel quadrature cannot be trusted to resolve
  structure, tile-independent (`node_point_kernel`); the tile's own
  Gaussian is folded onto it into ONE small 1-D array per (tile, node)
  (`node_log_x_kernel`) before the FFT convolution runs.

**The grid cell comes from a fidelity test against the fine smoothed
density, not a formula** (`IMPLEMENTATION.md` section 3's coarsening
amendment). The fine working cell is the smallest Silverman bandwidth
seen anywhere in the region and class, on either axis, divided by
`FINE_CELL_BANDWIDTH_DIVISOR` (5): fine enough that the bandwidth itself
is resolved by several cells, reported at build time. A fine working
grid at that cell on both `log10 x` and `log10 B` is smoothed and
convolved once at a reference shape node (the column-grid candidate with
the largest column -- the node needing the finest resolution,
`build_region_shared`'s `a_max_candidate`) and tile 0's own bandwidths,
and the grid cell on each axis is then DOUBLED alternately while the
BICUBIC reconstruction (`scipy.interpolate.RectBivariateSpline`,
`kx = ky = BICUBIC_DEGREE = 3`) of that reference density from the
coarser grid stays within `EPS_SHAPE` (0.02) relative L1
(`choose_grid_by_bar`); each candidate coarser grid is built directly at
its own resolution (deposit, smooth, convolve), not by downsampling the
fine array, since a density already smoothed to the population's own
resolution gives the same coarse-grained answer either way. This grid is
class-specific (the class's own `log10 B` range and bandwidths shape the
joint 2-D test), reported at build time. `ClassShape` reads a source's
`(log10 x, log10 B)` the same bicubic way inside the tabulated box.

Shape nodes are then a coarse subset of the shared column grid
(`prior.column_grid.nodes`), chosen on the CHOSEN grid's own smoothed
density by the same fidelity bar (`select_shape_nodes`): a candidate
node is dropped when linear interpolation, in `log A`, between its
surviving neighbours reproduces its own smoothed, convolved shape within
the bar. Candidates are evaluated lazily and only currently-surviving
shapes stay cached, so the full column-grid candidate list (order 183)
is never resident as shapes all at once.

AGB blends the O-rich and C-rich shapes (spec section 3) BEFORE the
smoothing and the fidelity tests see it; its own bandwidth is measured
on the same blend (`class_x_b_samples`). PAHC's grid and shape nodes are
chosen exactly as STAR's own, on the whole population's unreduced weight
(`class_raw_hist`'s "pahc" branch, unchanged) -- the fidelity tests never
see the limit weighting. The STORED density at each node is then built
separately for every entry of the population product's own eight 8
micron completeness limits (`LIMIT8_GRID_MJY`): the same `(log10 u,
LOG10_B_PAHC)` point cloud, deposited, smoothed and convolved exactly
like STAR, but weighted by that limit's own measured nebular-
contamination probability, `W * P_PAHC[:, j]` (spec section 4,
`pahc_raw_hist_for_limit`) -- exact per limit, no conditional
approximation. At read time, a source's own 8 micron limit brackets two
of the eight stored limit grids and blends linearly between them in
`log10` limit, the same bracket-and-blend rule the shape nodes use in
`log A` (`ClassShape.density`).

Analytic tails beyond the tabulated box (two-sided on `log10 x` as well
as `log10 B`, since the axis no longer starts at zero) are declared per
stored (tile, node[, limit]) slab from its own convolved edge behaviour
(a matched-slope exponential, `_edge_tail`), and the interior array is
rescaled so interior mass plus declared tail mass sums to exactly one.

Writes, per region and class, `bms/<class>/shape_<class>_tile__<Region>.
hdf5` for `class` in `star`, `agb`, `pahc`; PAHC's own file carries the
extra `LIMIT8_GRID_MJY` grid and an extra `DENSITY`/tail axis over it.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.fft import irfft, next_fast_len, rfft
from scipy.interpolate import RectBivariateSpline
from scipy.special import ndtr

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid
from sesnaimpute.prior import kernel as kernel_module

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

CLASSES = ("star", "agb", "pahc")

#: The SHAPE fidelity bar (`IMPLEMENTATION.md` section 3, amended
#: 2026-09-05), reused for both the grid-coarsening test and the shape-
#: node selection: relative L1 mass a coarser representation is allowed
#: to misplace, set by the priors' own uncertainty bands (STAR 10 % from
#: the anchor weights, GAL 0.1 dex, YSO 0.36 dex) an order of magnitude
#: below any of them -- fifty times looser than the column grid's own
#: `EPS_GRID` (0.002, `IMPLEMENTATION.md` section 2), which governs where
#: a source's column sits, not a shape.
EPS_SHAPE = 0.02

#: The mass fraction of the kernel's own `r = T/A` distribution excluded
#: from the tabulated `log10 x` range (module docstring): tighter than
#: `EPS_SHAPE` because the excluded mass sits at the far end of `x`,
#: where it pulls the interior array's own first moment by more than its
#: bare mass fraction -- set so the `log x`-marginal's mean at a shape
#: node holds `E[log10 r] + mean(log10 u)` (brief's algebraic identity)
#: within 1e-3 dex, tighter than `EPS_SHAPE` alone gets it.
X_MAX_EPS = 2.0e-4

#: Silverman's rule for the bandwidth of a 2-D kernel density estimate,
#: one bandwidth per axis, `h = sigma * n**(-1/6)` (Silverman 1986,
#: "Density Estimation for Statistics and Data Analysis", section 4.5):
#: `sigma` the tile-and-class's own weighted standard deviation on that
#: axis, `n` the tile's own effective star count `(Sum w)**2 / Sum w**2`
#: (module docstring, brief item 1).
SILVERMAN_EXPONENT = -1.0 / 6.0

#: The fine working cell before coarsening is the smallest Silverman
#: bandwidth seen anywhere in the region and class, on either axis,
#: divided by this factor (brief item 1: "h_min / 5") -- fine enough that
#: the bandwidth itself is resolved by several cells.
FINE_CELL_BANDWIDTH_DIVISOR = 5.0

#: Bicubic grid choice and read (brief item 2):
#: `scipy.interpolate.RectBivariateSpline`'s degree per axis.
BICUBIC_DEGREE = 3

#: `ln(10)`: converts a fractional (linear) width to a `log10` width by
#: the small-perturbation identity `d(log10 x) = dx / (x ln 10)`
#: (module docstring).
_LN10 = float(np.log(10.0))

#: The population's own 0.1-99.9th percentile range sets `log10 B`,
#: fixed per region (`IMPLEMENTATION.md` section 3).
RANGE_PERCENTILE = (0.1, 99.9)

#: The population's own tail percentile trimmed from the `log10 x` range
#: on the `u` side (brief item 1: "the 0.01% point of u"), mirroring
#: `X_MAX_EPS`'s trim on the kernel's own `r` side.
U_TAIL_PERCENTILE = 0.01

#: A depth-group / map-class code the kernel understands
#: (`prior.kernel.STATED_BEAM_ARCSEC`): a tile reads whichever of the two
#: its own sources are majority-provenance in (`A_COL_PROVENANCE`, spec
#: section 1.1: 0 Herschel, 1 Planck).
_PLANCK_PROVENANCE_CODE = 1

#: The Gaussian smoothing kernel array's truncation radius, in units of
#: its own sigma (`prior.kernel`'s own `_KERNEL_TRUNCATE_SIGMA`, the same
#: convention for a discretised Gaussian).
_GAUSSIAN_TRUNCATE_SIGMA = 6.0

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
# the class's own raw histogram, per tile (step 1) -- on log10 x
# ---------------------------------------------------------------------------

def class_raw_hist(pop, tile_idx, cls, log_x_centers, b_centers):
    """The class's one CIC-deposited, unit-mass `(log10 x, log10 B)`
    histogram for one tile (spec sections 2.2/3/4, step 1; module
    docstring). AGB normalises the O-rich and C-rich shapes SEPARATELY on
    the evolved subset before blending by `f_C` (spec section 3), so the
    blended object -- not either chemistry alone -- is what the kernel
    convolves and the fidelity tests measure. PAHC deposits the WHOLE
    population's weight, unreduced (spec section 4): this is the grid-
    and node-CHOICE density only -- the density actually stored per
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


def class_x_b_samples(pop, tile, cls):
    """`(log_u, x_weight, log_b, b_weight)`: the per-star `log10 u` (this
    class's `log10 x` at the reference node, module docstring) and
    `log10 B` samples and weights the Silverman bandwidth measures for
    one tile and class (brief item 1) -- the same population and weights
    `class_raw_hist` deposits, so the bandwidth describes exactly what is
    smoothed. AGB pools its O-rich and C-rich `log10 B` columns weighted
    by the spec's own `(1 - f_C)` / `f_C` chemistry mixture (section 3),
    matching the blended shape `class_raw_hist` builds."""
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
    data (brief item 1) -- what a Silverman bandwidth scales by, not the
    raw star count."""
    total = float(weight.sum())
    sq = float(np.sum(weight ** 2))
    return (total * total / sq) if sq > 0.0 else 0.0


def silverman_bandwidth(values, weight):
    """Silverman's rule for one axis of a 2-D kernel density estimate,
    `h = sigma * n**(-1/6)` (`SILVERMAN_EXPONENT`, module docstring):
    `sigma` the weighted standard deviation of `values` under `weight`,
    `n` the weighted effective sample size (`_effective_n`). `0.0` where
    there is no weight to measure a width from (an empty tile)."""
    n_eff = _effective_n(weight)
    if n_eff <= 0.0:
        return 0.0
    _, sigma = _weighted_mean_std(values, weight)
    return float(sigma * n_eff ** SILVERMAN_EXPONENT)


def tile_class_bandwidths(pop, cls):
    """Per tile: `dict(h_x=..., h_b=..., n_eff=...)` (brief item 1) --
    Silverman's rule applied to the tile's own `log10 u` and `log10 B`
    samples at the class's own weight (`class_x_b_samples`). `n_eff` is
    measured on the `log10 x` axis's own weight (STAR/PAHC: the one
    weight column; AGB: the evolved weight before the chemistry split,
    the tile's own effective star count, not the doubled pooled-`B`
    sample's)."""
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
    completeness-limit grid values (`IMPLEMENTATION.md` section 3,
    amended 2026-09-05): the same whole-population `(log10 u, LOG10_B_
    PAHC)` point cloud `class_raw_hist`'s "pahc" branch deposits, but
    weighted by this limit's own measured nebular-contamination
    probability, `W * P_PAHC[:, limit_idx]` (spec section 4) -- exact per
    limit, no conditional approximation of the per-star weight."""
    w = tile["w"] * tile["p_pahc"][:, limit_idx]
    h = _cic_hist([tile["log10_u"], tile["log10_b_pahc"]], w, [log_x_centers, b_centers])
    total = float(w.sum())
    return h / total if total > 0.0 else h


# ---------------------------------------------------------------------------
# the log10-x range, and the fine/coarse grid edges
# ---------------------------------------------------------------------------

def _rel_l1(reference, candidate):
    denom = float(np.sum(np.abs(reference)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(np.abs(candidate - reference)) / denom)


def _kernel_relative_widths(kern, cache, a, map_class):
    """`(r, w)` at column `a` and `map_class`, cached by `(map_class, a)`:
    every one of the several places this module needs the kernel's own
    `r = T/A` distribution at a candidate node reads the same call once
    (rule 9: the kernel quadrature is the expensive per-item cost here)."""
    key = (map_class, float(a))
    cached = cache.get(key)
    if cached is None:
        t_q, w_q = kern.nodes(float(a), map_class)
        cached = (t_q / a, w_q)
        cache[key] = cached
    return cached


def log_x_range(pop, kern, kernel_cache, candidate_a, distinct_map_classes):
    """`(log_x_min, log_x_max, report)` (brief item 1): `log10 x_min` is
    `log10(u_lo) + log10(r_lo)`, `log10 x_max` is `log10(u_hi) +
    log10(r_hi)`, where `u_lo`/`u_hi` are the population's own
    `U_TAIL_PERCENTILE`/`(100-U_TAIL_PERCENTILE)` points over every
    tile's `u`, and `r_lo`/`r_hi` are the smallest/largest `r = T/A` "in
    use" over every candidate shape-node column and map class this
    region's tiles carry -- the `X_MAX_EPS`-trimmed extremes of the
    kernel's own quadrature weight, the same fidelity bar as everywhere
    else, with the excluded mass falling to the declared analytic tail.
    Class-independent (every class deposits the same field-star `log10
    u` column and reads the same kernel), so computed once per region."""
    all_u = np.concatenate([t["u"] for t in pop["tiles"]])
    all_u = all_u[all_u > 0.0]
    u_lo, u_hi = np.percentile(all_u, [U_TAIL_PERCENTILE, 100.0 - U_TAIL_PERCENTILE])

    r_min, r_max = np.inf, 0.0
    for mc in distinct_map_classes:
        for a in candidate_a:
            r, w_q = _kernel_relative_widths(kern, kernel_cache, a, mc)
            order = np.argsort(r)
            r_sorted, w_sorted = r[order], w_q[order]
            cum = np.cumsum(w_sorted) / np.sum(w_sorted)
            idx_hi = min(int(np.searchsorted(cum, 1.0 - X_MAX_EPS)), r_sorted.size - 1)
            idx_lo = int(np.searchsorted(cum, X_MAX_EPS))
            r_max = max(r_max, float(r_sorted[idx_hi]))
            r_min = min(r_min, float(r_sorted[idx_lo]))

    x_min = float(u_lo) * r_min
    x_max = float(u_hi) * r_max
    report = dict(u_lo=float(u_lo), u_hi=float(u_hi), r_min=r_min, r_max=r_max,
                 x_min=x_min, x_max=x_max)
    return float(np.log10(x_min)), float(np.log10(x_max)), report


def _log_edges(log_lo, log_hi, cell):
    """Uniform bin edges on a `log10` axis from `log_lo` to `log_hi` at
    cell width `cell` or finer, bin count rounded UP to the next power of
    two: the fine grid `choose_grid_by_bar` starts from must be evenly
    halvable many times over, or the coarsening search would freeze on
    its very first step whenever the raw `ceil(range / cell)` count is
    odd (as it almost always is) -- this is the ONLY grid built by this
    function, at the fine cell, so rounding up here never coarsens
    anything the brief asked to be fine."""
    n_raw = max(1, int(np.ceil((log_hi - log_lo) / cell)))
    n = int(2 ** np.ceil(np.log2(n_raw)))
    return np.linspace(log_lo, log_hi, n + 1)


# ---------------------------------------------------------------------------
# the node's combined log10-x kernel: point masses + Gaussian smoothing,
# one small 1-D array, applied by FFT convolution (no (n_x, n_x) matrix)
# ---------------------------------------------------------------------------

def _offset_kernel_array(log_r, w, cell):
    """The kernel's own `log10(T/A)` distribution deposited by linear
    (CIC) interpolation onto the grid's own cell width, as a discrete
    point-mass array indexed by integer bin OFFSET from zero shift:
    entry `k` is the mass landing `offset_lo + k` cells from zero.
    Returns `(array, offset_lo)`."""
    frac = log_r / cell
    lo = np.floor(frac).astype(np.int64)
    t = frac - lo
    offset_lo = int(lo.min())
    offset_hi = int(lo.max()) + 1
    n = offset_hi - offset_lo + 1
    arr = np.zeros(n)
    idx_lo = lo - offset_lo
    np.add.at(arr, idx_lo, w * (1.0 - t))
    np.add.at(arr, idx_lo + 1, w * t)
    total = arr.sum()
    return (arr / total if total > 0.0 else arr), offset_lo


def _gaussian_kernel_array(sigma_dex, cell, truncate=_GAUSSIAN_TRUNCATE_SIGMA):
    """The bin-integrated discretised Gaussian of width `sigma_dex`
    (dex) on a grid of cell width `cell`, truncated at `truncate` sigma
    (module docstring; `prior.kernel`'s own truncation convention).
    `sigma_dex <= 0` is the identity (a single unit-mass bin at offset
    zero)."""
    if not (sigma_dex > 0.0) or not np.isfinite(sigma_dex):
        return np.array([1.0]), 0
    half = max(1, int(np.ceil(truncate * sigma_dex / cell)))
    edges = (np.arange(-half, half + 2) - 0.5) * cell
    z = edges / sigma_dex
    arr = np.diff(ndtr(z))
    total = arr.sum()
    return (arr / total if total > 0.0 else arr), -half


def _combine_kernels(arr_a, off_a, arr_b, off_b):
    """The two small 1-D kernels folded into one by convolution (module
    docstring: the kernel's own point masses and the node's Gaussian
    smoothing are both linear operators on `log10 x`, so one combined
    array carries both, evaluated once per node, not once per tile)."""
    combined = np.convolve(arr_a, arr_b, mode="full")
    total = combined.sum()
    return (combined / total if total > 0.0 else combined), off_a + off_b


def _kernel_relative_width_dex(log_r, w):
    """The node's own kernel's weighted standard deviation of `log10 r`
    (module docstring): the floor below which the node's own kernel
    quadrature cannot resolve finer structure."""
    mean = np.sum(w * log_r) / np.sum(w)
    var = np.sum(w * (log_r - mean) ** 2) / np.sum(w)
    return float(np.sqrt(max(var, 0.0)))


def node_point_kernel(r, w, cell):
    """`(point_arr, point_off, kernel_width_dex)`: the TILE-INDEPENDENT
    half of a shape node's `log10 x` operator at grid cell width `cell`
    -- the node's own kernel `log10 r = log10(T/A)` deposited as point
    masses (`_offset_kernel_array`) and their weighted standard deviation
    (`_kernel_relative_width_dex`, module docstring), both functions of
    `(a_node, map_class)` only. Cheap to compute once per candidate node
    and reused by every tile (brief item 1: only the Gaussian half below
    depends on the tile, through its own Silverman bandwidth)."""
    log_r = np.log10(r)
    point_arr, point_off = _offset_kernel_array(log_r, w, cell)
    kernel_width_dex = _kernel_relative_width_dex(log_r, w)
    return point_arr, point_off, kernel_width_dex


def node_log_x_kernel(point_arr, point_off, kernel_width_dex, cell, sigma_x_dex):
    """`(kernel_arr, offset_lo, sigma_dex)`: one shape node's combined
    `log10 x` operator FOR ONE TILE (module docstring, brief item 1) --
    the node's own kernel point masses (`node_point_kernel`) convolved
    with a Gaussian whose width is the node's own kernel resolution and
    the tile's own Silverman bandwidth `sigma_x_dex` added in QUADRATURE,
    `sqrt(kernel_width_dex**2 + sigma_x_dex**2)` (Silverman 1986 gives the
    population's own smoothing width; the node's kernel quadrature sets a
    floor below which structure cannot be resolved)."""
    sigma_dex = float(np.sqrt(kernel_width_dex ** 2 + sigma_x_dex ** 2))
    gauss_arr, gauss_off = _gaussian_kernel_array(sigma_dex, cell)
    kernel_arr, offset_lo = _combine_kernels(point_arr, point_off, gauss_arr, gauss_off)
    return kernel_arr, offset_lo, sigma_dex


def apply_log_x_kernel_fft(raw_fft, nfft, n_x, kernel_arr, offset_lo):
    """`raw_fft` (the `rfft` of a `(n_x, n_b)` histogram along axis 0, at
    a FIXED `nfft` shared by every node this tile evaluates), convolved
    with one node's combined kernel by multiplying in the frequency
    domain and inverting (module docstring: shift-invariant because the
    axis is `log10 x`), batched over every `log10 B` column at once.
    `nfft` MUST be the same value across
    every call sharing one `raw_fft` -- `scipy.fft`'s own transform-size
    cache holds a working buffer close to the FULL padded array size per
    DISTINCT length it sees, so calling this with a size that changes
    from node to node (each node's kernel is a different length) grows
    that cache without bound; a fixed, pre-sized `nfft` (the region's
    largest kept kernel, `_tile_shapes`) makes every call after the
    first a cache HIT (owner, 2026-09-05, measured: unbounded growth to
    5+ GB over 49 varying-length calls on PAHC's own array size, flat
    at under 1 GB with one fixed length). Cropped back to the original
    `n_x` window; mass that convolves past either edge is not
    renormalised back in here -- it is what the declared analytic tail
    (`finalise_node_shape`) represents."""
    kernel_fft = rfft(kernel_arr.astype(np.float32, copy=False), n=nfft)
    kshape = (kernel_fft.size,) + (1,) * (raw_fft.ndim - 1)
    full = irfft(raw_fft * kernel_fft.reshape(kshape), n=nfft, axis=0).astype(
        np.float32, copy=False)
    start = -offset_lo
    out = np.zeros((n_x,) + raw_fft.shape[1:], dtype=np.float32)
    src_lo = max(start, 0)
    src_hi = min(start + n_x, full.shape[0])
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    if src_hi > src_lo:
        out[dst_lo:dst_hi] = full[src_lo:src_hi]
    return out


def apply_log_x_kernel(raw, kernel_arr, offset_lo):
    """`apply_log_x_kernel_fft` for a ONE-OFF call (the grid-coarsening
    search, `choose_grid_by_bar`): computes its own `raw_fft` and `nfft`
    fresh, since each trial resolution has a different `raw` and a
    different kernel length anyway. Not for a loop over many nodes
    sharing one `raw` -- `_tile_shapes` and node selection precompute
    `raw_fft` once at a shared `nfft` instead, for exactly the reason
    `apply_log_x_kernel_fft`'s docstring states."""
    n_x = raw.shape[0]
    nfft = next_fast_len(n_x + kernel_arr.size - 1)
    raw_fft = rfft(raw.astype(np.float32, copy=False), n=nfft, axis=0)
    return apply_log_x_kernel_fft(raw_fft, nfft, n_x, kernel_arr, offset_lo)


# ---------------------------------------------------------------------------
# the catalogue's own smoothing on log10 B (unchanged from the linear-x
# module: one operator per class, applied once per tile, before the
# per-node log10-x kernel)
# ---------------------------------------------------------------------------

def gaussian_smoothing_matrix(edges, sigma):
    """The bin-integrated Gaussian smoothing operator on `edges`' own
    bins: mass placed at a bin's own centre spreads into every output
    bin by the Gaussian's cumulative distribution across that bin's
    edges. Mass the truncated tabulation box cannot hold becomes part of
    the declared analytic tail (`finalise_node_shape`). `sigma <= 0` is
    the identity."""
    if not (sigma > 0.0) or not np.isfinite(sigma):
        return np.eye(edges.size - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    z = (edges[:, None] - centers[None, :]) / sigma
    return np.diff(ndtr(z), axis=0)


def apply_b_smoothing(m_b, raw):
    """`m_b` applied along the `log10 B` axis (axis 1) of `raw`, `(n_x,
    n_b)`: `log10 B` smoothing does not depend on the shape node, so it
    is applied once per tile, before the per-node `log10 x` kernel sees
    the histogram."""
    moved = np.moveaxis(raw, 1, 0)
    shape = moved.shape
    out = m_b.dot(moved.reshape(shape[0], -1)).reshape((m_b.shape[0],) + shape[1:])
    return np.moveaxis(out, 0, 1)


# ---------------------------------------------------------------------------
# grid coarsening: halve the cell alternately while bicubic
# reconstruction of the fine smoothed density stays under the bar
# ---------------------------------------------------------------------------

def _bicubic_reconstruct(coarse, coarse_x_centers, coarse_b_centers,
                         fine_x_centers, fine_b_centers):
    """The coarse `(log10 x, log10 B)` density read back at every fine
    grid centre by a BICUBIC spline (`scipy.interpolate.RectBivariateSpline`,
    `kx = ky = BICUBIC_DEGREE`, brief item 2) -- the same interpolation
    `ClassShape` evaluates a source at (`_eval_interior`), so the
    coarsening test measures exactly what a consumer would see."""
    spline = RectBivariateSpline(coarse_x_centers, coarse_b_centers, coarse,
                                 kx=BICUBIC_DEGREE, ky=BICUBIC_DEGREE)
    return spline(fine_x_centers, fine_b_centers)


def choose_grid_by_bar(build_density_fn, log_x_min, log_x_max, b_lo, b_hi, fine_cell,
                       eps=EPS_SHAPE):
    """`(x_edges, b_edges, report)` (brief item 2): the coarsest grid
    reached by DOUBLING the `log10 x` and `log10 B` cell alternately from
    the fine working cell (the region and class's own Silverman-bandwidth
    floor, `FINE_CELL_BANDWIDTH_DIVISOR`), whose BICUBIC reconstruction of
    the reference node's fine smoothed density (`build_density_fn(x_edges,
    b_edges)`, rebuilt directly at each trial resolution) stays within
    `eps` relative L1 of the fine density. Each axis freezes on its own
    first failure, once it cannot be halved evenly, or once halving would
    leave fewer than `BICUBIC_DEGREE + 1` cells (the spline's own minimum
    knot count); the search stops when both are frozen."""
    fine_x_edges = _log_edges(log_x_min, log_x_max, fine_cell)
    fine_b_edges = _log_edges(b_lo, b_hi, fine_cell)
    fine_density = build_density_fn(fine_x_edges, fine_b_edges)

    x_edges, b_edges = fine_x_edges, fine_b_edges
    fine_x_centers = 0.5 * (fine_x_edges[:-1] + fine_x_edges[1:])
    fine_b_centers = 0.5 * (fine_b_edges[:-1] + fine_b_edges[1:])

    x_frozen = b_frozen = False
    turn = "x"
    while not (x_frozen and b_frozen):
        axis, turn = turn, ("b" if turn == "x" else "x")
        if axis == "x":
            if x_frozen:
                continue
            n_bins = x_edges.size - 1
            if n_bins < 2 * (BICUBIC_DEGREE + 1) or n_bins % 2 != 0:
                x_frozen = True
                continue
            trial_x_edges, trial_b_edges = x_edges[::2], b_edges
        else:
            if b_frozen:
                continue
            n_bins = b_edges.size - 1
            if n_bins < 2 * (BICUBIC_DEGREE + 1) or n_bins % 2 != 0:
                b_frozen = True
                continue
            trial_x_edges, trial_b_edges = x_edges, b_edges[::2]

        trial_density = build_density_fn(trial_x_edges, trial_b_edges)
        trial_x_centers = 0.5 * (trial_x_edges[:-1] + trial_x_edges[1:])
        trial_b_centers = 0.5 * (trial_b_edges[:-1] + trial_b_edges[1:])
        # a coarser bin holds the SUM of the fine bins it merges, so it
        # must be divided by that group size before the bicubic spline
        # reads it back at fine-bin resolution -- comparing raw bin MASS
        # across two different bin widths is not a reconstruction test at
        # all (it is off by the group size, which is exactly what
        # "coarser" means).
        group_x = (fine_x_edges.size - 1) // (trial_x_edges.size - 1)
        group_b = (fine_b_edges.size - 1) // (trial_b_edges.size - 1)
        recon = _bicubic_reconstruct(trial_density / (group_x * group_b), trial_x_centers,
                                     trial_b_centers, fine_x_centers, fine_b_centers)
        err = _rel_l1(fine_density, recon)
        if err < eps:
            x_edges, b_edges = trial_x_edges, trial_b_edges
        elif axis == "x":
            x_frozen = True
        else:
            b_frozen = True

    report = dict(n_x=x_edges.size - 1, n_b=b_edges.size - 1,
                 x_cell_dex=float(np.mean(np.diff(x_edges))),
                 b_cell_dex=float(np.mean(np.diff(b_edges))),
                 n_x_fine=fine_x_edges.size - 1, n_b_fine=fine_b_edges.size - 1)
    return x_edges, b_edges, report


def select_shape_nodes(candidate_a, shape_fn, eps=EPS_SHAPE):
    """The kept subset of `candidate_a` (`IMPLEMENTATION.md` section 3),
    on the SMOOTHED, grid-resolution density `shape_fn(i)` returns for
    candidate `i`: drop a node when linear interpolation, in `log A`,
    between its surviving neighbours reproduces its own shape to within
    `eps` relative L1; endpoints are always kept. One left-to-right pass
    drops every candidate it can, chaining a dropped node's own left
    anchor forward; repeated to a fixed point. `shape_fn` is called
    lazily and its results cached only for the currently-surviving scan
    window -- a dropped candidate's shape is evicted immediately -- so
    candidates never all sit resident at once."""
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
        for stale in [k for k in cache if not keep[k]]:
            cache.pop(stale, None)
    return np.flatnonzero(keep)


# ---------------------------------------------------------------------------
# analytic tails beyond the tabulated box (step 2), two-sided on log10 x
# ---------------------------------------------------------------------------

def _edge_tail(marginal, grid, cell, side):
    """One edge's matched-slope exponential tail, `(slope, mass)` --
    `(0.0, 0.0)` where matching would require an upward slope (never
    extrapolate upward). `side="hi"` matches the grid's last two bin
    CENTRES (slope must be negative); `side="lo"` matches the first two
    (slope must be positive). `marginal` has already summed the OTHER
    axis (`finalise_node_shape`'s own x/b marginals), so its value at a
    centre is a BIN MASS over one cell of width `cell` on THIS axis, not
    a density -- dividing by `cell` is the same bin-mass-to-density
    conversion `prior.callable`'s own module docstring applies to a
    pointwise `ClassShape.density` read (`10_POSTERIOR.md` section 3's
    normalisation), and it is what the mass integral below needs to be a
    probability rather than a bin count (omitting it declared a tail
    mass smaller than the true one by a factor of `cell`, while
    `ClassShape._eval_node`'s own tail, read at the true edge with no
    such factor, carried far more mass than that declaration -- brief
    item 1's "large deviations"). The declared mass integrates the model
    from the TABULATED EDGE -- half a cell beyond the outermost centre,
    where the tabulated array's own last bin already stops -- to
    infinity, not from the centre itself, so it does not double the mass
    that bin already carries; `ClassShape._eval_node` reads the same
    centre value and the same slope and reaches the same edge amplitude
    by the same algebra (`exp(slope . half_cell) . exp(slope . (query -
    edge)) == exp(slope . (query - centre))`), so declaration and
    evaluation are one model, not two."""
    if side == "hi":
        v0, v1, g0, g1 = marginal[-2], marginal[-1], grid[-2], grid[-1]
    else:
        v0, v1, g0, g1 = marginal[0], marginal[1], grid[0], grid[1]
    if not (v0 > 0.0 and v1 > 0.0 and g1 > g0):
        return 0.0, 0.0
    slope = float((np.log(v1) - np.log(v0)) / (g1 - g0))
    half_cell = 0.5 * cell
    if side == "hi":
        if not slope < 0.0:
            return 0.0, 0.0
        density_at_edge = (v1 / cell) * np.exp(slope * half_cell)
        mass = density_at_edge / (-slope)
    else:
        if not slope > 0.0:
            return 0.0, 0.0
        density_at_edge = (v0 / cell) * np.exp(-slope * half_cell)
        mass = density_at_edge / slope
    return slope, float(mass)


def finalise_node_shape(conv, x_centers, b_centers):
    """The stored per-(tile, node[, limit]) product (step 2): declares the
    analytic tail from the convolved array's own `(log10 x, log10 B)`
    edge marginals (always 2-D, one call per stored slab -- PAHC calls it
    once per limit, `_tile_shapes`), two-sided on `log10 x` as well as
    `log10 B` since the axis no longer starts at zero, then rescales the
    interior array so interior mass plus declared tail mass sums to
    exactly one (algebraic acceptance: `DENSITY.sum() == 1 -
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

def _tile_shapes(density_ds, t, pop, cls, kept_nodes, point_kernels_by_map_class, map_class,
                 x_centers, b_centers, x_cell, h_x, h_b, b_edges, nfft, checked_node_indices=(),
                 checked_pahc_keys=None):
    """One tile's own densities at every kept shape node, WRITTEN ONE
    NODE (PAHC: one node and limit) AT A TIME into `density_ds[t, j]`
    (CODING_RULES.md 10a): holding every kept node's finalised array for
    one tile before returning it (as an earlier version of this function
    did) multiplies the working set by `n_node`, which is exactly what
    drove PAHC's third axis past the memory ceiling before it was retired
    (owner, 2026-09-05). Writing straight into the pre-sized HDF5 dataset
    (safe from multiple `prefer="threads"` workers: h5py serialises the
    underlying HDF5 calls with its own global lock) means only ONE node's
    (PAHC: one node-and-limit's) convolved array is ever resident per
    tile, on top of the raw histogram(s) and their own `rfft` (rule 8:
    the per-tile cost is one raw histogram, one `log10 B` smoothing at
    the tile's own Silverman `h_b`, ONE forward FFT -- PAHC: one PER
    LIMIT, `IMPLEMENTATION.md` section 3's own "eight times" cost --
    plus one small per-node kernel build and FFT/inverse along `log10 x`
    at the tile's own `h_x` combined with the node's own kernel width,
    never per star -- `nfft`, fixed for every node this tile evaluates,
    is what keeps that per-node FFT cheap and bounded,
    `apply_log_x_kernel_fft`'s own docstring). `checked_node_indices`
    (PAHC: `checked_pahc_keys`, `(node, limit)` pairs) -- tile 0 only, the
    report's own acceptance checks -- keeps a COPY of just those few
    arrays, not a second `n_node`- (or `n_node * n_limit`-) sized
    accumulator: an earlier version of this streaming rewrite kept every
    node's array "for tile 0" to feed the checks and reintroduced the
    exact blow-up this function exists to avoid (owner, 2026-09-05)."""
    m_b = gaussian_smoothing_matrix(b_edges, h_b)
    n_x = x_centers.size
    point_kernels = point_kernels_by_map_class[map_class]
    n_node = len(kept_nodes)
    sum_identity_dev = 0.0
    checked_density = {}

    if cls == "pahc":
        tile = pop["tiles"][t]
        n_limit = pop["limit8_grid_mjy"].size
        raw_ffts = []
        for k in range(n_limit):
            raw_k = pahc_raw_hist_for_limit(tile, k, x_centers, b_centers)
            raw_k = apply_b_smoothing(m_b, raw_k).astype(np.float32, copy=False)
            raw_ffts.append(rfft(raw_k, n=nfft, axis=0))
        tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi, mass_outside = (
            np.empty((n_node, n_limit)) for _ in range(5))
        for j, (point_arr, point_off, kernel_width_dex) in enumerate(point_kernels):
            kernel_arr, offset_lo, _ = node_log_x_kernel(point_arr, point_off, kernel_width_dex,
                                                          x_cell, h_x)
            for k in range(n_limit):
                conv = apply_log_x_kernel_fft(raw_ffts[k], nfft, n_x, kernel_arr, offset_lo)
                d, sxlo, sxhi, slo, shi, mo = finalise_node_shape(conv, x_centers, b_centers)
                density_ds[t, j, k] = d
                if t == 0 and checked_pahc_keys and (j, k) in checked_pahc_keys:
                    checked_density[(j, k)] = d.copy()
                tail_x_lo[j, k], tail_x_hi[j, k], tail_b_lo[j, k], tail_b_hi[j, k], \
                    mass_outside[j, k] = sxlo, sxhi, slo, shi, mo
                dev = abs(float(d.astype(np.float64).sum()) + mo - 1.0)
                sum_identity_dev = max(sum_identity_dev, dev)
        return (tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi, mass_outside,
               sum_identity_dev, checked_density)

    raw = class_raw_hist(pop, t, cls, x_centers, b_centers)
    raw = apply_b_smoothing(m_b, raw).astype(np.float32, copy=False)
    raw_fft = rfft(raw, n=nfft, axis=0)
    tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi, mass_outside = (
        np.empty(n_node) for _ in range(5))
    for j, (point_arr, point_off, kernel_width_dex) in enumerate(point_kernels):
        kernel_arr, offset_lo, _ = node_log_x_kernel(point_arr, point_off, kernel_width_dex,
                                                      x_cell, h_x)
        conv = apply_log_x_kernel_fft(raw_fft, nfft, n_x, kernel_arr, offset_lo)
        d, sxlo, sxhi, slo, shi, mo = finalise_node_shape(conv, x_centers, b_centers)
        density_ds[t, j] = d
        if t == 0 and j in checked_node_indices:
            checked_density[j] = d.copy()
        tail_x_lo[j], tail_x_hi[j], tail_b_lo[j], tail_b_hi[j], mass_outside[j] = (
            sxlo, sxhi, slo, shi, mo)
        dev = abs(float(d.astype(np.float64).sum()) + mo - 1.0)
        sum_identity_dev = max(sum_identity_dev, dev)
    return (tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi, mass_outside,
           sum_identity_dev, checked_density)


def build_region_shared(config, region, kern, candidate_a, kernel_cache):
    """Everything a region's three classes share (module docstring): the
    population read, the tile map classes, and the `log10 x` range --
    STAR, AGB and PAHC deposit the same field-star `log10 u` column and
    read the same kernel, so `log_x_min`/`log_x_max` do not depend on
    class. Computed once per region."""
    pop = read_population(config, region)
    n_tile = pop["n_tile"]
    map_classes = tile_map_classes(config, region, n_tile)
    distinct_map_classes = sorted(set(map_classes))

    log_x_min, log_x_max, x_range_report = log_x_range(
        pop, kern, kernel_cache, candidate_a, distinct_map_classes)

    return dict(pop=pop, n_tile=n_tile, map_classes=map_classes,
               distinct_map_classes=distinct_map_classes,
               log_x_min=log_x_min, log_x_max=log_x_max,
               x_range_report=x_range_report,
               a_max_candidate=float(candidate_a.max()))


def build_region_class(config, region, cls, shared, candidate_a, kern, kernel_cache):
    """One region and class, end to end (module docstring): the class's
    own `log10 B` range and per-tile Silverman bandwidths, the grid-
    coarsening fidelity test at the reference (largest-column) candidate
    node and tile 0's own bandwidths (`choose_grid_by_bar`), node
    selection on the chosen grid's own smoothed density (streamed,
    `select_shape_nodes`), and the per-tile evaluation in parallel (rule
    8, 10a)."""
    pop, n_tile = shared["pop"], shared["n_tile"]
    map_classes, distinct_map_classes = shared["map_classes"], shared["distinct_map_classes"]

    b_lo, b_hi = class_b_range(pop, cls)

    # brief item 1: per-tile, per-class Silverman bandwidths on log10 u
    # (this class's own `log10 x` at the reference node) and log10 B, and
    # the region/class's own fine working cell, the smallest positive
    # bandwidth anywhere divided by FINE_CELL_BANDWIDTH_DIVISOR.
    tile_bw = tile_class_bandwidths(pop, cls)
    positive = np.array([v for bw in tile_bw for v in (bw["h_x"], bw["h_b"]) if v > 0.0])
    if positive.size == 0:
        raise ValueError("prior.star_shapes: no tile in region %r, class %r has any "
                         "weight to measure a Silverman bandwidth from" % (region, cls))
    fine_cell = float(positive.min()) / FINE_CELL_BANDWIDTH_DIVISOR
    h_x_max = max(bw["h_x"] for bw in tile_bw)

    # the grid-coarsening reference: the candidate needing the finest
    # resolution (the largest column, module docstring), tile 0's own
    # map class and bandwidths.
    map_class0 = map_classes[0]
    a_ref = shared["a_max_candidate"]
    r_ref, w_ref = _kernel_relative_widths(kern, kernel_cache, a_ref, map_class0)
    h_x0, h_b0 = tile_bw[0]["h_x"], tile_bw[0]["h_b"]

    def build_density_fn(x_edges, b_edges):
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
        raw = class_raw_hist(pop, 0, cls, x_centers, b_centers)
        m_b_trial = gaussian_smoothing_matrix(b_edges, h_b0)
        raw = apply_b_smoothing(m_b_trial, raw)
        cell = float(np.mean(np.diff(x_edges)))
        point_arr, point_off, kw = node_point_kernel(r_ref, w_ref, cell)
        kernel_arr, offset_lo, _ = node_log_x_kernel(point_arr, point_off, kw, cell, h_x0)
        return apply_log_x_kernel(raw, kernel_arr, offset_lo)

    x_edges, b_edges, grid_report = choose_grid_by_bar(
        build_density_fn, shared["log_x_min"], shared["log_x_max"], b_lo, b_hi, fine_cell)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    x_cell = float(np.mean(np.diff(x_edges)))

    # every candidate's TILE-INDEPENDENT half of the log10-x kernel at
    # the chosen grid cell, one small array each -- cheap to hold in full
    # (rule 9: this is not the per-tile density array the streaming rule
    # protects); the tile's own Gaussian half is folded on per tile
    # (`_tile_shapes`, `node_log_x_kernel`).
    point_cache_by_mc = {
        mc: {int(i): node_point_kernel(*_kernel_relative_widths(kern, kernel_cache, a, mc), x_cell)
            for i, a in enumerate(candidate_a)}
        for mc in distinct_map_classes}

    # a FIXED nfft, shared by node selection and every tile's evaluation
    # below, sized from the WORST-CASE (largest) Silverman `h_x` any tile
    # carries for this class: every candidate's kernel is a different
    # length regardless (`a` sets the node's own kernel width, the tile
    # sets its own Gaussian), and re-deriving the FFT size per call is
    # exactly what drives `scipy.fft`'s transform-size cache unbounded
    # (`apply_log_x_kernel_fft`'s docstring) -- one size, sized to the
    # longest kernel any (tile, candidate) pair at this grid can have,
    # makes every call after the first a cache hit.
    max_kernel_len = 0
    for entries in point_cache_by_mc.values():
        for point_arr, _, kw in entries.values():
            sigma_worst = float(np.sqrt(kw ** 2 + h_x_max ** 2))
            gauss_arr, _ = _gaussian_kernel_array(sigma_worst, x_cell)
            max_kernel_len = max(max_kernel_len, point_arr.size + gauss_arr.size - 1)
    nfft = next_fast_len(x_centers.size + max_kernel_len - 1)

    # node selection (`IMPLEMENTATION.md` section 3): tile 0's own
    # smoothed density at every candidate, on the CHOSEN grid (brief item
    # 2), evaluated lazily (`select_shape_nodes`) so candidates never all
    # sit resident at once.
    m_b0 = gaussian_smoothing_matrix(b_edges, h_b0)
    raw0 = class_raw_hist(pop, 0, cls, x_centers, b_centers)
    raw0 = apply_b_smoothing(m_b0, raw0).astype(np.float32, copy=False)
    raw0_fft = rfft(raw0, n=nfft, axis=0)

    def node_kernel0(idx):
        point_arr, point_off, kw = point_cache_by_mc[map_class0][idx]
        return node_log_x_kernel(point_arr, point_off, kw, x_cell, h_x0)[:2]

    shape_fn = lambda idx: apply_log_x_kernel_fft(
        raw0_fft, nfft, x_centers.size, *node_kernel0(idx))
    kept = select_shape_nodes(candidate_a, shape_fn, EPS_SHAPE)
    shape_nodes = candidate_a[kept]

    point_kernels_by_map_class = {mc: [point_cache_by_mc[mc][int(idx)] for idx in kept]
                                  for mc in distinct_map_classes}

    # the report's acceptance checks (`_mean_moment_check`, `_grid_
    # centre_exact_check`, PAHC's `_pahc_between_check`) each read a
    # handful of (tile 0, node[, limit]) arrays -- `checked_node_indices`
    # (PAHC: `checked_pahc_keys`) tells `_tile_shapes` to keep a copy of
    # only those, not every kept node's array for tile 0 (module
    # docstring).
    node_idx_moment = int(np.argmin(np.abs(shape_nodes - pop["tiles"][0]["a_tile"])))
    node_idx_exact = 0
    checked_node_indices = frozenset((node_idx_moment, node_idx_exact))

    checked_pahc_keys = None
    limit_lo_idx = limit_hi_idx = None
    f_lim8_median = None
    if cls == "pahc":
        # `IMPLEMENTATION.md` section 3: a source's own 8um limit brackets
        # two of the eight stored limit grids; the region's own median
        # real-source limit (`catalog.limits.limits`, the same per-source
        # array the population product's own grid is percentiled from)
        # exercises that bracket for the report's own interpolation check.
        from sesnaimpute.catalog import limits as limits_module
        from sesnaimpute.prior import star_population
        f_lim_i4 = limits_module.limits(config, region)[:, star_population.IDX_I4]
        f_lim8_median = float(np.median(f_lim_i4))
        limit_grid = pop["limit8_grid_mjy"]
        lo_arr, _ = column_grid.bracket(
            np.array([np.log10(f_lim8_median)]), np.log10(limit_grid))
        limit_lo_idx = int(lo_arr[0])
        limit_hi_idx = min(limit_lo_idx + 1, limit_grid.size - 1)
        checked_pahc_keys = frozenset(((node_idx_moment, limit_lo_idx),
                                       (node_idx_exact, limit_lo_idx),
                                       (node_idx_exact, limit_hi_idx)))

    path, density_checks, sum_identity_dev, tails, mass_outside = (
        _write_region_class_streaming(
            config, region, cls, pop, n_tile, map_classes, shape_nodes, x_edges, b_edges,
            point_kernels_by_map_class, x_centers, b_centers, tile_bw, config.n_jobs,
            checked_node_indices, nfft, checked_pahc_keys))

    result = dict(
        region=region, cls=cls, shape_nodes=shape_nodes, x_edges=x_edges, b_edges=b_edges,
        density_checks=density_checks, node_idx_moment=node_idx_moment,
        node_idx_exact=node_idx_exact, tail_x_lo=tails[0], tail_x_hi=tails[1],
        tail_b_lo=tails[2], tail_b_hi=tails[3], mass_outside=mass_outside,
        map_classes=map_classes, n_tile=n_tile, n_candidate=len(candidate_a),
        f_c=pop["f_c"], pop=pop, grid_report=grid_report, sum_identity_dev=sum_identity_dev,
        path=path, fine_cell=fine_cell,
        h_x_tiles=np.array([bw["h_x"] for bw in tile_bw]),
        h_b_tiles=np.array([bw["h_b"] for bw in tile_bw]),
        pahc_limit_lo_idx=limit_lo_idx, pahc_limit_hi_idx=limit_hi_idx,
        pahc_limit_median_mjy=f_lim8_median,
    )
    return result


# ---------------------------------------------------------------------------
# write, streamed one batch of tiles at a time (CODING_RULES.md 10a): the
# HDF5 file's DENSITY dataset -- the one array whose full-survey size
# matters -- is pre-sized and filled tile by tile; only `config.n_jobs`
# tiles' own density arrays are ever resident together, never all of
# `n_tile`. The tail scales and `MASS_OUTSIDE` are `(n_tile, n_node)`
# (PAHC: `(n_tile, n_node, n_limit)`) scalars per class -- negligible next
# to `DENSITY` -- and are still assembled as ordinary in-memory arrays.
# ---------------------------------------------------------------------------

def _write_region_class_streaming(config, region, cls, pop, n_tile, map_classes, shape_nodes,
                                  x_edges, b_edges, point_kernels_by_map_class,
                                  x_centers, b_centers, tile_bw, n_jobs,
                                  checked_node_indices, nfft, checked_pahc_keys=None):
    n_node = shape_nodes.size
    grid_shape = (x_centers.size, b_centers.size)
    x_cell = float(np.mean(np.diff(x_edges)))
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
    density_checks = {}
    sum_identity_dev = 0.0

    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "tile"
        if cls == "agb":
            f.attrs["F_C"] = pop["f_c"]
        f.create_dataset("SHAPE_NODES", data=shape_nodes.astype(np.float64))
        f.create_dataset("LOG10_X_EDGES", data=x_edges.astype(np.float64))
        f.create_dataset("LOG10_B_EDGES", data=b_edges.astype(np.float64))
        f.create_dataset("TILE_ID", data=np.arange(n_tile, dtype=np.int64))
        f.create_dataset("MAP_CLASS", data=np.array(map_classes, dtype="S8"))
        # one HDF5 chunk per (tile, node[, limit]) slice: `_tile_shapes`
        # writes one such slice at a time, and a chunk any bigger (e.g. a
        # whole tile) would force HDF5 to hold the WHOLE chunk's worth in
        # memory while it fills in one slice at a time -- reintroducing
        # the same blow-up one level down, inside the HDF5 library.
        # gzip level 1 with the byte-shuffle filter (rule 8, timed): level
        # 4 on order-50 MB (tile, node) chunks measured minutes per region
        # on real data, dominated by compression, not I/O; level 1 plus
        # shuffle (which reorders a float array's bytes so gzip sees long
        # runs) is markedly faster on this mostly-smooth, many-zero data
        # for a comparable ratio.
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
                delayed(_tile_shapes)(density_ds, t, pop, cls, shape_nodes,
                                      point_kernels_by_map_class, map_classes[t],
                                      x_centers, b_centers, x_cell,
                                      tile_bw[t]["h_x"], tile_bw[t]["h_b"], b_edges,
                                      nfft, checked_node_indices, checked_pahc_keys)
                for t in batch)
            for t, (sxlo, sxhi, slo, shi, mo, dev, checked) in zip(batch, batch_results):
                tail_x_lo[t], tail_x_hi[t], tail_b_lo[t], tail_b_hi[t], mass_outside[t] = (
                    sxlo, sxhi, slo, shi, mo)
                sum_identity_dev = max(sum_identity_dev, dev)
                if checked:
                    density_checks.update(checked)
            del batch_results

        f.create_dataset("TAIL_X_LO_SCALE", data=tail_x_lo)
        f.create_dataset("TAIL_X_HI_SCALE", data=tail_x_hi)
        f.create_dataset("TAIL_B_LO_SCALE", data=tail_b_lo)
        f.create_dataset("TAIL_B_HI_SCALE", data=tail_b_hi)
        f.create_dataset("MASS_OUTSIDE", data=mass_outside)

    return (path, density_checks, sum_identity_dev,
           (tail_x_lo, tail_x_hi, tail_b_lo, tail_b_hi), mass_outside)


# ---------------------------------------------------------------------------
# read: the evaluator every prior-table build calls
# ---------------------------------------------------------------------------

class ClassShape(object):
    """One class's per-tile shape, evaluated per source (`IMPLEMENTATION.md`
    section 3's evaluation column): the tile's own density, the two
    bracketing shape nodes blended in `log A`, BICUBIC in `(log10 x,
    log10 B)` inside the grid (brief item 2), the analytic tail outside
    on either axis, `a < 0` (or `a == 0`, which has no `log10 x`) mapped
    to the low-`x` tail or zero. PAHC additionally carries a limit grid
    (`limit_grid_mjy`, `IMPLEMENTATION.md` section 3, amended 2026-09-05):
    a source's own 8 micron limit brackets two of the eight stored limit
    grids and blends between them linearly in `log10` limit, the same
    bracket-and-blend rule the shape nodes use in `log A` -- the four
    (node, limit) corners are evaluated and bilinearly combined.
    Vectorised over sources (rule 8)."""

    def __init__(self, cls, shape_nodes, x_edges, b_edges, density, tail_x_lo, tail_x_hi,
                 tail_b_lo, tail_b_hi, mass_outside, limit_grid_mjy=None):
        self.cls = cls
        self.shape_nodes = np.asarray(shape_nodes, dtype=np.float64)
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
        # PAHC only: log10 of the eight stored 8um completeness-limit
        # grid values (`IMPLEMENTATION.md` section 3), sorted ascending.
        self.limit_log = (None if limit_grid_mjy is None else
                          np.log10(np.asarray(limit_grid_mjy, dtype=np.float64)))

    def _slab(self, tile_idx, node_idx, limit_idx):
        """The stored `(log10 x, log10 B)` array at one (tile, node[,
        limit]) -- `limit_idx` is `None` for STAR/AGB, an integer index
        into the limit grid for PAHC."""
        if limit_idx is None:
            return self.density_table[tile_idx, node_idx]
        return self.density_table[tile_idx, node_idx, limit_idx]

    def _eval_interior(self, tile_ids, node_idx, log_x, logb, limit_idx):
        n = log_x.shape[0]
        out = np.empty(n, dtype=np.float64)
        # grouped by (tile, node[, limit]): a query slab is built, and
        # its bicubic spline fitted, once per group, not per source
        # (rule 8/9).
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
        """One shape node's (PAHC: one shape-node-and-limit-grid's) value,
        blending the interior bicubic read with the declared analytic
        tail beyond either `log10 x` edge or either `log10 B` edge
        (`finalise_node_shape`). The tail amplitude is read at the
        tabulated grid's own outermost bin CENTRE -- never extrapolated
        past the bicubic fit's own support -- and the exponent is
        referenced from that same centre, matching `_edge_tail`'s own
        declaration exactly (that function's own docstring algebra), so
        the mass this evaluator delivers when integrated is the mass
        `finalise_node_shape` declared as `MASS_OUTSIDE`, not a
        different number reached by extrapolating a spline past its own
        edge with an unrelated amplitude and reference point (brief item
        1)."""
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

    def density(self, a, log10_b, tile_ids, a_col, f_lim8=None):
        """`density(a, log10_b, tile_ids, a_col[, f_lim8])`
        (`IMPLEMENTATION.md` section 3): `x = a/a_col` per source,
        `log10 x`, the two bracketing shape nodes blended in `log A`
        (PAHC: crossed with the two bracketing limit grids, `f_lim8`
        this source's own 8 micron limit, blended in `log10` limit),
        `a < 0` mapped to zero (`a == 0`, having no `log10 x`, reads as
        the declared low-`x` tail's own limit)."""
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
        log_x = np.where(x_lin > 0.0, np.log10(np.clip(x_lin, _LOG_FLOOR, None)),
                         self.x_edges[0] - 1.0e3)

        log_nodes = np.log(self.shape_nodes)
        i_lo, t_node = column_grid.bracket(np.log(a_col), log_nodes)
        i_hi = np.minimum(i_lo + 1, self.shape_nodes.size - 1)

        if self.limit_log is None:
            val_lo = self._eval_node(tile_ids, i_lo, log_x, log10_b, None)
            val_hi = self._eval_node(tile_ids, i_hi, log_x, log10_b, None)
        else:
            log_f = np.broadcast_to(
                np.log10(np.asarray(f_lim8, dtype=np.float64)), a.shape)
            m_lo, t_limit = column_grid.bracket(log_f, self.limit_log)
            m_hi = np.minimum(m_lo + 1, self.limit_log.size - 1)
            v_lo_lo = self._eval_node(tile_ids, i_lo, log_x, log10_b, m_lo)
            v_lo_hi = self._eval_node(tile_ids, i_lo, log_x, log10_b, m_hi)
            v_hi_lo = self._eval_node(tile_ids, i_hi, log_x, log10_b, m_lo)
            v_hi_hi = self._eval_node(tile_ids, i_hi, log_x, log10_b, m_hi)
            val_lo = (1.0 - t_limit) * v_lo_lo + t_limit * v_lo_hi
            val_hi = (1.0 - t_limit) * v_hi_lo + t_limit * v_hi_hi
        out = (1.0 - t_node) * val_lo + t_node * val_hi
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
        x_edges = f["LOG10_X_EDGES"][:]
        b_edges = f["LOG10_B_EDGES"][:]
        density = f["DENSITY"][:]
        tail_x_lo = f["TAIL_X_LO_SCALE"][:]
        tail_x_hi = f["TAIL_X_HI_SCALE"][:]
        tail_b_lo = f["TAIL_B_LO_SCALE"][:]
        tail_b_hi = f["TAIL_B_HI_SCALE"][:]
        mass_outside = f["MASS_OUTSIDE"][:]
        limit_grid_mjy = f["LIMIT8_GRID_MJY"][:] if "LIMIT8_GRID_MJY" in f else None
    return ClassShape(cls, shape_nodes, x_edges, b_edges, density, tail_x_lo, tail_x_hi,
                      tail_b_lo, tail_b_hi, mass_outside, limit_grid_mjy=limit_grid_mjy)


# ---------------------------------------------------------------------------
# report (rule 10, 11, 13) -- the algebraic acceptance the brief names
# ---------------------------------------------------------------------------

def _report(result):
    """`DENSITY.sum() == 1 - MASS_OUTSIDE` (rule 11's algebraic
    acceptance): `sum_identity_dev` is tracked incrementally while
    `DENSITY` streams to disk (`_write_region_class_streaming`), one
    tile at a time, since the full array is never resident here to
    reduce over."""
    return dict(
        n_node=result["shape_nodes"].size, n_candidate=result["n_candidate"],
        n_x=result["x_edges"].size - 1, n_b=result["b_edges"].size - 1,
        median_mass_outside=float(np.median(result["mass_outside"])),
        sum_identity_dev=result["sum_identity_dev"],
    )


def _mean_moment_check(config, region, cls, result, kern):
    """Acceptance (brief): the `log10 x`-marginal's mean at a shape node
    equals `E[log10 r] + mean(log10 u)` (module docstring's algebraic
    identity, `log x = log r + log u`) -- measured on the first tile, at
    the node nearest the tile's own column (PAHC: at the stored limit
    grid bracketing the region's own median source limit, so the
    checked slab is one this module actually keeps)."""
    pop = result["pop"]
    tile0 = pop["tiles"][0]
    map_class0 = result["map_classes"][0]
    node_idx = result["node_idx_moment"]
    a_node = float(result["shape_nodes"][node_idx])
    t_q, w_q = kern.nodes(a_node, map_class0)
    log_r = np.log10(t_q / a_node)
    e_log_r = float(np.sum(w_q * log_r) / np.sum(w_q))

    if cls == "star":
        w, log_u = tile0["w_star"], tile0["log10_u"]
        key = node_idx
    elif cls == "agb":
        ev = tile0["is_evolved"]
        w, log_u = tile0["w_agb"][ev], tile0["log10_u"][ev]
        key = node_idx
    else:
        limit_idx = result["pahc_limit_lo_idx"]
        w = tile0["w"] * tile0["p_pahc"][:, limit_idx]
        log_u = tile0["log10_u"]
        key = (node_idx, limit_idx)
    mean_log_u = float(np.sum(w * log_u) / np.sum(w)) if w.sum() > 0 else float("nan")
    expected = e_log_r + mean_log_u

    x_centers = 0.5 * (result["x_edges"][:-1] + result["x_edges"][1:])
    d = result["density_checks"][key]
    x_marg = d.sum(axis=1)
    measured = float(np.sum(x_marg * x_centers) / np.sum(x_marg)) if x_marg.sum() > 0 else float("nan")
    dev = abs(measured - expected)
    return expected, measured, dev


def _grid_centre_exact_check(result):
    """Acceptance (brief): the evaluator at a grid centre, with a source
    whose `a_col` equals a shape node, reproduces the stored value
    exactly (PAHC additionally: at a source's own 8um limit equal to one
    of the eight stored limit grids). Built from a single cached
    `density_checks` array for tile 0 (module docstring) -- not the whole
    node (or node x limit) axis, which `_write_region_class_streaming`
    never keeps resident. A second, duplicated dummy node (PAHC: also a
    duplicated dummy limit) gives `column_grid.bracket` a valid span; the
    query sits exactly on the first of each pair, so the duplicate's
    interpolation weight is always zero."""
    shape_nodes, x_edges, b_edges = result["shape_nodes"], result["x_edges"], result["b_edges"]
    node_idx = result["node_idx_exact"]
    is_pahc = result["cls"] == "pahc"
    a_node = float(shape_nodes[node_idx])
    shape_nodes_pair = np.array([a_node, a_node * (1.0 + 1.0e-9)])

    if is_pahc:
        limit_idx = result["pahc_limit_lo_idx"]
        d = result["density_checks"][(node_idx, limit_idx)]
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
        d = result["density_checks"][node_idx]
        limit_grid_pair = None
        density_pair = np.stack([d, d])[np.newaxis]  # (1 tile, 2 node, n_x, n_b)
        tx_lo = np.full((1, 2), result["tail_x_lo"][0, node_idx])
        tx_hi = np.full((1, 2), result["tail_x_hi"][0, node_idx])
        tb_lo = np.full((1, 2), result["tail_b_lo"][0, node_idx])
        tb_hi = np.full((1, 2), result["tail_b_hi"][0, node_idx])
        mo = np.full((1, 2), result["mass_outside"][0, node_idx])

    shape = ClassShape(result["cls"], shape_nodes_pair, x_edges, b_edges, density_pair,
                       tx_lo, tx_hi, tb_lo, tb_hi, mo, limit_grid_mjy=limit_grid_pair)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    a_query = a_node * (10.0 ** x_centers)
    tile_ids = np.zeros(a_query.size, dtype=np.int64)
    a_col = np.full(a_query.size, a_node)
    f_lim8 = np.full(a_query.size, limit_val) if is_pahc else None
    got = shape.density(a_query, np.full(a_query.size, b_centers[0]), tile_ids, a_col,
                          f_lim8=f_lim8)
    expected = d[:, 0]
    return float(np.max(np.abs(got - expected)))


def _pahc_between_check(config, region, result):
    """Acceptance (brief, PAHC only): a read at the region's own median
    source 8um limit lies between the two limit grids it brackets,
    cell by cell -- convexity of the linear blend in `log10` limit
    (`ClassShape.density`), at a column held exactly on a tabulated shape
    node (a duplicated dummy pair, `_grid_centre_exact_check`'s own
    trick) so only the limit axis is exercised. Built from the two REAL
    limit grids `build_region_class` already cached for tile 0 at
    `node_idx_exact` (`checked_pahc_keys`)."""
    node_idx = result["node_idx_exact"]
    i_lo, i_hi = result["pahc_limit_lo_idx"], result["pahc_limit_hi_idx"]
    f_lim8_median = result["pahc_limit_median_mjy"]
    limit_grid = result["pop"]["limit8_grid_mjy"]
    d_lo = result["density_checks"][(node_idx, i_lo)]
    d_hi = result["density_checks"][(node_idx, i_hi)]

    shape_nodes, x_edges, b_edges = result["shape_nodes"], result["x_edges"], result["b_edges"]
    a_node = float(shape_nodes[node_idx])
    shape_nodes_pair = np.array([a_node, a_node * (1.0 + 1.0e-9)])
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
                       tx_lo, tx_hi, tb_lo, tb_hi, mo, limit_grid_mjy=limit_grid_pair)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    a_query = a_node * (10.0 ** x_centers)
    tile_ids = np.zeros(a_query.size, dtype=np.int64)
    a_col = np.full(a_query.size, a_node)
    f_lim8 = np.full(a_query.size, f_lim8_median)
    got = shape.density(a_query, np.full(a_query.size, b_centers[0]), tile_ids, a_col,
                          f_lim8=f_lim8)
    lo_vals, hi_vals = d_lo[:, 0], d_hi[:, 0]
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
    (default: all thirty), one file per region and class (module
    docstring). The column kernel and the shared column-grid node
    candidates are survey-wide and built once, not per region or per
    class (rule 9); the `log10 x` range and the per-tile Silverman
    bandwidths are class-dependent and built once per region and class
    (`build_region_shared`, `tile_class_bandwidths`)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    kern = kernel_module.load(config)
    candidate_a = column_grid.nodes(config)
    kernel_cache = {}

    for region in region_names:
        shared = build_region_shared(config, region, kern, candidate_a, kernel_cache)
        xr = shared["x_range_report"]
        print("star_shapes: %s: log10 x range [%.4g, %.4g] (x_min=%.4g x_max=%.4g, "
             "u=[%.4g,%.4g] r=[%.4g,%.4g])"
             % (region, shared["log_x_min"], shared["log_x_max"], xr["x_min"], xr["x_max"],
                xr["u_lo"], xr["u_hi"], xr["r_min"], xr["r_max"]), flush=True)

        for cls in CLASSES:
            result = build_region_class(config, region, cls, shared, candidate_a, kern,
                                        kernel_cache)
            path = result["path"]
            rep = _report(result)
            expected, measured, dev = _mean_moment_check(config, region, cls, result, kern)
            exact_dev = _grid_centre_exact_check(result)
            pahc_line = ""
            if cls == "pahc":
                # grid_centre_exact_max_dev above is measured AT a real
                # limit-grid value (result["pahc_limit_lo_idx"]) for PAHC,
                # so it already IS the "reproduces one of the eight grid
                # values exactly" check; only the interpolation check is
                # new here.
                between = _pahc_between_check(config, region, result)
                pahc_line = (
                    " (grid_centre_exact measured at limit=%.4g mJy, grid index %d) "
                    "pahc_median_between(f_lim8_median=%.4g mJy, bracket=[%.4g,%.4g] mJy, "
                    "t=%.3f, max_violation=%.2e)"
                    % (result["pop"]["limit8_grid_mjy"][result["pahc_limit_lo_idx"]],
                       result["pahc_limit_lo_idx"], between["f_lim8_median"],
                       between["limit_lo_mjy"], between["limit_hi_mjy"], between["t_limit"],
                       between["max_violation"]))
            gr = result["grid_report"]
            print(
                "star_shapes: %s/%s: bandwidth h_x(median)=%.4g h_b(median)=%.4g dex "
                "(per tile, Silverman 1986) fine_cell=%.4g dex "
                "candidates=%d nodes_kept=%d grid=%dx%d "
                "(x_cell=%.4g dex, from %d fine; b_cell=%.4g dex, from %d fine) "
                "median_mass_outside=%.4e sum_identity_max_dev=%.2e "
                "mean_moment(expected=%.4f measured=%.4f dev=%.2e dex) "
                "grid_centre_exact_max_dev=%.2e%s -> %s"
                % (region, cls, float(np.median(result["h_x_tiles"])),
                   float(np.median(result["h_b_tiles"])), result["fine_cell"],
                   rep["n_candidate"], rep["n_node"], rep["n_x"], rep["n_b"],
                   gr["x_cell_dex"], gr["n_x_fine"], gr["b_cell_dex"], gr["n_b_fine"],
                   rep["median_mass_outside"], rep["sum_identity_dev"],
                   expected, measured, dev, exact_dev, pahc_line, path))


if __name__ == "__main__":
    run(build)
