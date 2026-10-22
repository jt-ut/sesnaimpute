"""P6, the prior atlas (SPEC_BMSTP_DRAFT.md sec. 8; IMPLEMENTATION_BMSTP_DRAFT.md
sec. 1.2 P6, sec. 3 row 1.9).

Per admitted nside-512 pixel, a deterministic weighted quadrature over each
class's own population, dimmed at the pixel's own column through the blended
law (`population.selection.kappa_hybrid`), weighted by each member's
probability of being catalogued (sec. 8, sec. 6.2): per band the
completeness `C_i(f)` at the pixel's own marginalised 50% limit
(`catalog.depth_grid`'s `F_LIM_50_PIX_MJY`) and its own marginalised width
(`W_DEX_PIX`), and `P(>=2 of 8)` a mixture of the eight bands' independent
non-detection probabilities and a fully comonotone model, by `constants.
LAMBDA` (C20 brief, briefs/round4_2026-10-20/C20_detection_correlation.md:
bands observed at the same position share the local background and
confusion, so independence alone under-predicted SESNA's own measured
catalogued fraction) -- the same completeness model `fittp.likelihood`
prices, never a step at the 50% limit. The selection appears here and nowhere
else in the atlas (sec. 1.2): the prior itself is unthinned.

This build writes STAR/AGB/PAHC (sec. 5.1-5.3, partitioning the field population:
a star is a STAR or a PAHC member of the population, never both, weighted
`W_STAR*(1-P_PAHC)`/`W_STAR*P_PAHC`), GAL (sec. 5.4: SWIRE's four IRAC fluxes
per galaxy, S from the counts law's own node, colours from a galaxy measured at
that node, at `x=1`), YSO (sec. 5.5: a weighted quadrature over the population --
`N_YSO_NODES` quantile nodes of the register's own census weight
(`template_weights.yso_population_weight`'s Dunham et al. 2015 census
density over `log10 f_ref,4.5,theta` divided by the library's density of
templates in the same quantity, times inclination uniform in cos i and the
evolutionary-class census, sec 1.4, owner's ruling 2026-09-09) crossed with the
region's own shift-kernel cells (`bmstp.sample_cloud.shift_kernel`) and the
sightline's own `p(x)` cells on the cloud interval (`bmstp.sample_cloud.
sample_x`'s binned return) -- each template's own eight `F_REF` scaled by
`10^delta` at the shift cell's own centre) and H2S (sec. 5.6: a weighted
quadrature over the region's 2.12 um lognormal cells crossed with each IRAC
band's own `N_RATIO_NODES`-quantile colour-ratio cells and YSO's own depth
cells, the bands carried by the measured knot line-to-band ratios). AGB's
members are the star-family sampler's own JOINT measure over evolved star
and shell template (`bmstp.sample_star.sample_agb`, FINDINGS.md S10,
coordinator's joint-measure ruling: every evolved star crossed with every
template of its drawn chemistry, Riebel+2012's optical-depth distribution
as that template's own share of the mass), read here directly, no second
crossing -- each member's own template's eight `F_REF` scaled so its own
4.5 um flux equals that member's OWN joint-measure brightness (`log10_f45`),
never a chemistry-pooled anchor. All six classes enter the total-count check.

Every class's catalogued density is a weighted quadrature over its own whole
population (sec. 8): alongside it this build writes `N_CAT_CELL_<C>` (128, 110,
`grid.LOG10_XI_EDGES` by `grid.LOG10_F45_EDGES`) for all six classes, each
class's own expected number of catalogued objects per parameter cell, summing
over cells to `RATIO_<C> * N_source`; and `N_CELL_<C>` (same axes, sec. 8's
intrinsic population), the SAME members quadrature as `N_CAT_CELL_<C>` with
every member's own probability of being catalogued set to one (no flux cut,
no dimming enters that one factor).

`N_CAT_CELL_<C>`'s own `log10 F_4.5` bin position is the member's DIMMED flux
at the pixel compared -- `catalogued_probability`'s own `flux0 * 10**(-0.4 a
kappa_band)`, the same dimming the catalogued probability itself already
evaluates there -- so the cell a member's expected catalogued count lands in
is the cell the sky would actually catalogue it in (C29, briefs/
round4_2026-10-20/C29_atlas_check_dimmed_axis.md): the per-bin check this
builds for (`atlas.shapes`'s own page, and `c18_diag/reproduce.py`'s own
construction outside this package) compares predicted counts against
observed sky counts, themselves necessarily binned by their OWN measured
(dimmed) flux, never the undimmed flux no sky measurement ever sees.
`N_CELL_<C>`'s own bin position is UNCHANGED, still the member's UNDIMMED
flux: it carries no selection and no dimming at all (every member's
probability of being catalogued is one), so there is no pixel-dependent
flux to bin it by, and the GRID this density lives on (`grid.LOG10_F45_EDGES`
itself, the axis every other stored shape in this package shares) is not
touched by C29 -- only this ONE class's own check histogram moved. The two
axes now differing means `N_CAT_CELL_<C> <= N_CELL_<C>` cell BY CELL no
longer holds by construction (a cell N_CAT_CELL_<C> now reaches by a
member's dimmed flux can differ from the cell N_CELL_<C> holds that SAME
member's weight in, undimmed) -- only their TOTALS keep the old relation
(`N_CAT_CELL_<C>.sum() <= N_CELL_<C>.sum()`, RATIO_<C> is axis-free). Never
read by `bmstp`/`fittp` (`atlas.shapes`'s own module docstring): no consumer
outside this check and the external diagnostic scripts it matches reads
either array.

Every module on this atlas's worker path (this module, `knot_field`,
`sample_gal`, `sample_star`, `sample_cloud`, `grid`, `density`,
`template_weights`, `sky.derived.profile`, `catalog.completeness`,
`population.yso`, `population.selection`, `sky.derived.herschel_column`, and
whatever they import) carries every import of a module holding a compiled
extension (`astropy.*`, `scipy.*`, `healpy`, `h5py`, `numba`) at module top
level, never inside a function: a loky worker imports a task's module, and
everything that module imports at top level, only when it unpickles that
worker's first task -- so a native extension deferred to a function runs its
first `dlopen` at whatever moment that function is later called, which on
this path is after the tile phase has already started numba's `workqueue`
thread pool in that worker, and dyld's loader is not safe to enter again
once numba's threads are live.
"""

import os

import h5py
import healpy as hp
import numpy as np
from scipy.special import ndtri

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import pieces
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog import completeness as completeness_module
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.population import h2s as h2s_module
from sesnaimpute.population import knot_rate
from sesnaimpute.population import pahc_curve
from sesnaimpute.population import pahn as pahn_module

_STEM = "prior_atlas_hpx512"
from sesnaimpute.population import selection as selection_module
from sesnaimpute.population import yso as yso_module
from sesnaimpute.population import yso_law as yso_law_module
from sesnaimpute.bmstp import density as density_module
from sesnaimpute.bmstp import grid
from sesnaimpute.bmstp import knot_field
from sesnaimpute.bmstp import sample_cloud
from sesnaimpute.bmstp import sample_gal
from sesnaimpute.bmstp import sample_star
from sesnaimpute.bmstp import template_weights
from sesnaimpute.fittp import prior_reader
from sesnaimpute.sky.derived import column as column_module

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)
IDX_I4 = BAND_KEYS.index("I4")
IDX_I2 = BAND_KEYS.index("I2")
CLASSES = ("STAR", "AGB", "PAHC", "PAHN", "GAL", "YSO", "H2S")

#: The bright-end count check's two multiples of the pixel's own 4.5 um
#: 50% limit (SPEC_BMSTP_DRAFT.md sec. 9 "the bright-end count ratio"):
#: brighter than these, completeness is 1 on both the catalog and the
#: model side and the counts lie inside the range Gaia and 2MASS
#: calibrate, so the ratio there separates a wrong overall level from a
#: wrong faint extrapolation.
BRIGHT_MULT_3, BRIGHT_MULT_10 = 3.0, 10.0

#: Two of eight bands clear -- the survey's own catalogue rule (sec. 1.2),
#: the same constant `population.selection.MIN_BANDS` sets.
MIN_BANDS_CLEAR = selection_module.MIN_BANDS

#: GAL member colour-collapse cell (`_gal_members`, rule 9): the width, in
#: dex, within which two "galz" templates' three colour log-ratios
#: (F_I1/F_I2, F_I3/F_I2, F_I4/F_I2) are treated as the same member. Set
#: far below the completeness roll-off width (`W_DEX` ~= 0.1-0.3 dex,
#: `catalog.depth_grid`), so the collapse moves `catalogued_probability`'s
#: result by much less than the roll-off itself resolves.
GAL_COLOUR_CELL_DEX = 0.05

#: STAR/PAHC member colour-collapse cell (`_build_one_tile`, `_collapse_star_members`,
#: rule 9): the width, in dex, within which two field stars' seven log flux
#: ratios to I2 (J, H, Ks, I1, I3, I4, M1 -- every band a member enters
#: `catalogued_probability` through, besides `u` and `F_4.5` itself) are
#: treated as the same member. Set far below the completeness roll-off
#: width (`W_DEX` ~= 0.1-0.3 dex, `catalog.depth_grid`), so the collapse
#: moves `catalogued_fraction`'s result by much less than the roll-off
#: itself resolves.
STAR_COLOUR_CELL_DEX = 0.1

#: YSO's template quadrature (sec. 5.5 "Marks"): equally spaced quantile
#: nodes of the register's own census weight. 500 is the smallest of
#: {250, 500, 1000, 2000} whose per-pixel `frac` at the region's
#: median-column sightline is within the 0.005 bar of the fine
#: (4000-node, full shift/depth-cell) reference -- 250 misses the bar
#: (0.0056), 500 clears it (0.0004).
N_YSO_NODES = 500

#: YSO's distance-shift quadrature (sec. 5.5 "Marks"): equally spaced
#: quantile nodes of the region's own shift kernel, each node the
#: kernel's own cell centre at that quantile, weight
#: `1/N_YSO_SHIFT_NODES` -- every kernel cell with nonzero mass makes the
#: member set too large for a per-sightline build.
N_YSO_SHIFT_NODES = 5

#: YSO and H2S's shared depth quadrature (sec. 5.5/5.6 "Marks"): equally
#: spaced quantile nodes of the sightline's own `p(x)`, each node the
#: cell centre at that quantile, weight `1/N_DEPTH_NODES` -- every `p(x)`
#: cell with nonzero mass makes the member set too large for a
#: per-sightline build.
N_DEPTH_NODES = 8

#: H2S's colour-ratio quadrature (sec. 5.6 "Marks"): equally spaced
#: quantile nodes of each IRAC band's own Giannini table, one band
#: independent of the others so the joint is their product (5**4 = 625
#: nodes). 3 nodes/band (81 joint) is the dominant source of H2S's own
#: convergence miss (isolated at a converged sigma count: N_RATIO_NODES=3
#: alone gives 0.0102 relative to the fine reference at the region's
#: median-column sightline, N_RATIO_NODES=5 gives 0.0012).
N_RATIO_NODES = 5

#: H2S's surface-brightness quadrature (sec. 5.6 "Marks"): equally spaced
#: quantile nodes of the region's own lognormal itself (`scipy.special.
#: ndtri`'s exact inverse CDF). 20 clears the 0.005 bar at the region's
#: median-column sightline against the fine reference once N_RATIO_NODES
#: is 5 (0.0033; the sigma count is not the dominant source -- see
#: N_RATIO_NODES's own comment).
N_SIGMA_NODES = 20

#: one definition, shared with `bmstp.density` (it feeds area-weighted
#: density totals in both, so a divergence would silently move a number).
_HPX512_PIXEL_DEG2 = density_module._HPX512_PIXEL_DEG2


def _depth_grid(config, region):
    """The admitted pixel axis and its marginalised 50% limit, read from
    Rob Gutermuth's own measured completeness curve (COMPLETENESS_2026-
    10-20.md "The reader") through the one reader object, `catalog.
    limits.completeness`: `(pix, f_lim_50_pix_mjy)`, `f_lim_50_pix_mjy`
    each pixel's own `fifty_percent_pixel` (the curve's own
    completeness=0.5 crossing), used only for the bright-end threshold
    tests (sec. 9's 3x/10x check, `_above_fraction`'s own 4.5 um limit) --
    the detection-probability model itself (`catalogued_probability`) no
    longer reduces the curve to a limit and a width at all: it reads the
    pixel's own `PIXEL_CURVE_<band>` directly, at the row this function's
    caller gets from `np.arange(pix.size)` (`pix` IS `reader.
    hpx_pix_512`'s own order, so a pixel's position in `pix` is exactly
    its row into `PIXEL_CURVE_<band>`; `_completeness_band_params` below
    is the one place that table is read, once per worker). This replaces
    `catalog.depth_grid`'s retired region-wide counts fit
    (`F_LIM_50_PIX_MJY`/`W_DEX_PIX`) outright. A band with no on-map
    pixel anywhere in this admitted pixel is unsurveyed there and carries
    `+inf` for `f_lim` (never `NaN`, the convention `catalog.depth_grid`
    used): the curve's own crossing search returns `NaN` for exactly that
    case (never reaches 0.5 because every plane is 0, module docstring of
    `catalog.limits.Completeness._crossing_mag`), substituted here; the
    SAME all-zero row read through `PIXEL_CURVE_<band>` directly already
    gives a detection probability of exactly 0 at any flux (`catalog.
    completeness.completeness_many`'s own zero-curve case), so no second
    substitution is needed on that path."""
    reader = limits_module.completeness(config, region)
    pix = reader.hpx_pix_512
    rows = np.arange(pix.shape[0])
    f_lim = np.empty((pix.shape[0], N_BANDS), dtype=np.float64)
    for j, band in enumerate(BAND_KEYS):
        flux50 = reader.fifty_percent_pixel(band, rows)
        f_lim[:, j] = np.where(np.isfinite(flux50), flux50, np.inf)
    return pix, f_lim


def _coverage(config, region, pix):
    """The catalogue's own surveyed fraction at each admitted pixel (sec.
    3.3's "coverage", sec. 8's catalogue definition): `sky/derived/
    spitzer/coverage_spitzer_hpx512`'s own per-band `FRAC` (I1 I2 I3 I4
    M1, that product's own column order), the fraction of the pixel on a
    covered mosaic pixel. The four IRAC bands' `max` is taken as the
    any-IRAC union fraction -- a LOWER BOUND on the true union (the
    product stores only each band's own fraction, not the union mask
    itself), exact where the four IRAC footprints coincide, which is
    nearly everywhere since 3.6/5.8 um share one camera array and 4.5/8.0
    um the other. The Spitzer product covers a superset of the admitted
    pixels (every pixel a mosaic touches), so a pixel absent from it
    (never touched by a mosaic) reads FRAC = 0; joins are by HPX_PIX, not
    by row."""
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "bmstp.atlas: no Spitzer mosaic coverage for %s at %s -- run the "
            "'sesnaimpute.sky.derived.coverage' RUNBOOKtp.sh line first" % (region, path))
    with h5py.File(path, "r") as f:
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)
    union = frac[:, :4].max(axis=1) if frac.size else np.empty(0, dtype=np.float64)
    order = np.argsort(cov_pix)
    loc = np.searchsorted(cov_pix[order], pix)
    loc = np.minimum(loc, cov_pix.size - 1)
    hit = order[loc]
    found = cov_pix[hit] == pix if cov_pix.size else np.zeros(pix.size, dtype=bool)
    out = np.zeros(pix.size, dtype=np.float64)
    out[found] = union[hit[found]]
    return out


def _pixel_column(config, region, pix):
    """The pixel's own extinction column and arm, from the sightline it is
    a child of (SPEC_BMSTP_DRAFT.md sec. 8: "placed at the pixel (its
    column, its tile or sightline, its arm)"). Nested HEALPix, confirmed
    from `granules.build`'s own `HPX_PIX_256 = HPX_PIX_512 // 4`: every
    admitted nside-512 pixel's parent nside-256 sightline is `pix // 4`.
    `sky/derived/adopted/extinction_adopted_sightline.hdf5` (survey-wide,
    no region argument) carries `A_K`/`PROVENANCE` at that granule for
    every source-bearing sightline, so every admitted pixel resolves --
    no NaN. This is the column a member's own light passes through, so
    it dims every class here; the young-star law and the knot density
    instead read `_pixel_gas_column`'s gas column below, the quantity
    Pokhrel's law was measured on (SPEC_BMSTP_DRAFT.md sec. 0 "column",
    sec. 5.5, the same read `bmstp.density` uses) -- the two differ by
    `F_EXTINCTION >= 1` (`sky/derived/column.py`), so using this one for
    the law would inflate it by `F_EXTINCTION**2`."""
    parent256 = pix // 4
    path = config_module.product_path(config, "sky/derived", "adopted", "extinction", "sightline")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "bmstp.atlas: no extinction sightline column at %s -- run the "
            "'sesnaimpute.sky.derived.column' RUNBOOKtp.sh line first" % path)
    with h5py.File(path, "r") as f:
        mine = np.asarray(f["REGION_CODE"][:]) == column_module._region_codes(config, [region])[region]
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)[mine]
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)[mine]
        prov = np.asarray(f["PROVENANCE"][:])[mine]
    order = np.argsort(sl_pix)
    loc = np.searchsorted(sl_pix[order], parent256)
    loc = np.minimum(loc, sl_pix.size - 1)
    hit = order[loc]
    found = sl_pix[hit] == parent256
    if not np.all(found):
        raise ValueError("bmstp.atlas: %d admitted pixel(s) have no sightline column in %s"
                          % (int(np.sum(~found)), path))
    return a_k[hit], prov[hit]


def _pixel_gas_column(config, region, pix):
    """The pixel's own GAS column, from the sightline it is a child of --
    the same nesting `_pixel_column` uses (`pix // 4` into the nside-256
    parent). `sky/derived/adopted/column_adopted_sightline.hdf5`
    (survey-wide, no region argument) carries `A_K` at that granule, the
    quantity the young-star law and the knot density are evaluated on
    (SPEC_BMSTP_DRAFT.md sec. 5.5): starlight suffers the extinction
    column `_pixel_column` returns, but Pokhrel's star-gas law was
    measured against the gas column, not against extinction inflated by
    `F_EXTINCTION`."""
    parent256 = pix // 4
    path = config_module.product_path(config, "sky/derived", "adopted", "column", "sightline")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "bmstp.atlas: no gas sightline column at %s -- run the "
            "'sesnaimpute.sky.derived.column' RUNBOOKtp.sh line first" % path)
    with h5py.File(path, "r") as f:
        mine = np.asarray(f["REGION_CODE"][:]) == column_module._region_codes(config, [region])[region]
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)[mine]
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)[mine]
    order = np.argsort(sl_pix)
    loc = np.searchsorted(sl_pix[order], parent256)
    loc = np.minimum(loc, sl_pix.size - 1)
    hit = order[loc]
    found = sl_pix[hit] == parent256
    if not np.all(found):
        raise ValueError("bmstp.atlas: %d admitted pixel(s) have no gas sightline column in %s"
                          % (int(np.sum(~found)), path))
    return a_k[hit]


def _pixel_tile(config, region, pix):
    """`(tile, n_filled)`: `TILE_ID` at each admitted pixel
    (`population/anchors/tiles/hpx512__R.hdf5`, the star-family tile
    definition, sec. 8: "a pixel outside every tile takes its
    sightline's tile"). A pixel absent from the tile map takes its
    nside-256 parent's tile, from any tiled sibling under that parent
    (every admitted pixel is a child of a source-bearing nside-256
    pixel, whose tiled children resolve it, sec. 8); the siblings are
    asserted to agree, so no ambiguity and no NaN density remains."""
    path = config_module.product_path(config, "population", "anchors", "tiles", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        t_pix_id = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        t_tile_id = np.asarray(f["TILE_ID"][:], dtype=np.int64)
    order = np.argsort(t_pix_id)
    loc = np.searchsorted(t_pix_id[order], pix)
    loc = np.minimum(loc, t_pix_id.size - 1)
    hit = order[loc]
    found = t_pix_id[hit] == pix
    tile = np.full(pix.size, -1, dtype=np.int64)
    tile[found] = t_tile_id[hit[found]]

    missing = ~found
    n_filled = int(np.count_nonzero(missing))
    if n_filled:
        parent_of_tiled = t_pix_id // 4
        order_p = np.argsort(parent_of_tiled, kind="stable")
        parent_sorted = parent_of_tiled[order_p]
        tile_sorted = t_tile_id[order_p]
        parent_missing = pix[missing] // 4
        starts = np.searchsorted(parent_sorted, parent_missing, side="left")
        ends = np.searchsorted(parent_sorted, parent_missing, side="right")
        if np.any(starts == ends):
            raise ValueError("bmstp.atlas: %d tile-less pixel(s) with no tiled "
                              "nside-256 sibling" % int(np.sum(starts == ends)))
        filled = np.empty(n_filled, dtype=np.int64)
        for k in range(n_filled):
            sibs = tile_sorted[starts[k]:ends[k]]
            if np.unique(sibs).size != 1:
                raise ValueError("bmstp.atlas: tile-less pixel's nside-256 "
                                  "siblings disagree on tile")
            filled[k] = sibs[0]
        tile[missing] = filled
    return tile, n_filled


#: rule 10b's 512 MB batch budget, for the (n_pixel_batch, n_mem, 8) arrays
#: `catalogued_fraction` holds via `catalogued_probability`: `kappa`,
#: `dimming`, `flux`, `log10_f`, `z`, `p`, `one_minus_p` and the per-band
#: `np.delete` term in the closed-form loop are each that shape, and each
#: expression above allocates its own buffer rather than reusing one --
#: `_N_TEMP_ARRAYS` counts that many same-shape buffers live at once,
#: generously, so the true peak (measured below) sits under the target
#: with margin. `_build_one_tile` runs this inside `config.n_jobs` joblib
#: workers at once (STAR/AGB/PAHC, one tile per worker), each held to
#: this 512 MB working set on its own (rule 10b); the total across all
#: workers is therefore `n_jobs * 512 MB`, not 512 MB in aggregate --
#: sizing `n_jobs` to the machine's memory is `root.cfg`'s job (rule
#: 10a), not this batch size's.
_PIXEL_BATCH_BUDGET_BYTES = 512 * 1024 * 1024
_N_TEMP_ARRAYS = 12

#: `catalogued_probability`'s own inner member sub-block (F4 2026-10-09
#: "change B"): a few thousand members keeps one sub-block's temporaries
#: (`n_pix_batch * _MEMBER_SUBBLOCK * N_BANDS * 8 bytes`, several of
#: them at once) within an L2-sized working set rather than the whole
#: pixel batch's own member axis, without changing `_pixel_batch_size`
#: or `catalogued_fraction`'s own chunking (rule 10b's F4 risk 2).
_MEMBER_SUBBLOCK = 3000


#: Per-process cache for `_completeness_band_params`, keyed `(data_root,
#: region)` (`pieces.py`'s own per-process-cache convention, `bmstp.
#: sample_star`'s pattern): `catalogued_probability` reads this for every
#: tile or sightline a worker handles and would otherwise reread the
#: region's whole `PIXEL_CURVE_<band>` table (`catalog.limits.
#: Completeness.pixel_fraction_params`'s own `curves[:]`) on every one of
#: them.
_COMPLETENESS_PARAMS_CACHE = {}


def _completeness_band_params(config, region):
    """Every Spitzer band's own `(curve, g0, dm, n, a, c, scale)`
    (`catalog.limits.Completeness.pixel_fraction_params`), read ONCE per
    `(data root, region)` and cached for this worker process's lifetime."""
    key = (config.data_root, region)
    if key not in _COMPLETENESS_PARAMS_CACHE:
        _COMPLETENESS_PARAMS_CACHE.clear()  # one region's tables per worker at a time
        reader = limits_module.completeness(config, region)
        _COMPLETENESS_PARAMS_CACHE[key] = {
            band: reader.pixel_fraction_params(band) for band in BAND_KEYS}
    return _COMPLETENESS_PARAMS_CACHE[key]


def _pixel_batch_size(n_mem):
    """Pixels per batch so `n_pixel_batch * n_mem * N_BANDS * 8 bytes
    (float64) * _N_TEMP_ARRAYS` stays under `_PIXEL_BATCH_BUDGET_BYTES`
    per worker, independent of how many pixels the caller (a tile, a
    sightline, or GAL's whole region) holds -- CODING_RULES_BMSTP.md rule
    10b, one worker's own 512 MB working set."""
    row_bytes = n_mem * N_BANDS * 8 * _N_TEMP_ARRAYS
    return max(1, _PIXEL_BATCH_BUDGET_BYTES // row_bytes)


def catalogued_probability(a_col, u, flux0, pixel_row, band_params, config):
    """`(p_cat, p_cat_i2, flux_i2)` for ONE pixel batch: the closed-form
    probability that a member is catalogued (SPEC_BMSTP_DRAFT.md sec. 8,
    sec. 6.2) -- `catalogued_fraction` forms the weighted means over these
    three arrays, batch by batch.

    `a_col` (n_pix,) the pixel's own column, `u` (n_mem,) the member's own
    placement fraction, `flux0` (n_mem, 8) undimmed member fluxes (zero
    where the member carries no flux in a band), `pixel_row` (n_pix,)
    each pixel's own row into `PIXEL_CURVE_<band>` (`bmstp.atlas._depth_
    grid`'s own `np.arange(pix.size)`, sliced the same way `a_col` is),
    `band_params` the region's own `(curve, g0, dm, n, a, c, scale)` per
    band (`_completeness_band_params`, read once per worker, never per
    pixel batch or member sub-block). `a = a_col * u` (sec. 8's "its own
    extinction"), dimmed through the blended law (`population.selection.
    kappa_hybrid`). The two-of-eight step test is replaced by the
    member's probability of being catalogued: per band `p_i = C_i(f_i)`,
    Rob Gutermuth's own MEASURED completeness curve at this pixel
    (`catalog.completeness.completeness_many`, the design's one kernel,
    COMPLETENESS_2026-10-20.md "The reader") interpolated at the member's
    own dimmed flux, capped at the owner's `C_CAP` (the measurement's own
    ~1/200 resolution floor, `catalog.completeness.C_CAP`) -- never a
    Gaussian roll-off fit to it -- zero where the member carries no flux
    in that band (GAL's 2MASS/24um, H2S's J/H/24um).

    `p_cat` (n_pix, n_mem) (C20 brief, briefs/round4_2026-10-20/
    C20_detection_correlation.md): `(1 - LAMBDA) * P_indep + LAMBDA *
    p_(2)`, a mixture of SESNA's independent band-detection model,
    `P_indep = P(>=2 of 8) = 1 - Prod_i(1-p_i) - Sum_i p_i Prod_{j!=i}
    (1-p_j)` (bands independent given the fluxes -- today's closed form,
    LAMBDA=0), and a fully comonotone model, `p_(2)` the SECOND-largest
    of the eight `p_b` (one latent uniform draw per member, shared by
    every band: at least two bands clear iff the draw is below the
    second-largest `p_b`). `constants.LAMBDA` is the one survey-wide
    share (`population.detection_correlation`'s own measurement,
    against UKIDSS stars matched to SESNA in North America Nebula and
    Perseus): bands observed at the same position share the local
    background and confusion, so independence is not the whole story.

    `p_cat_i2` (n_pix, n_mem) (sec. 9 "the bright-end count ratio"): the
    SAME mixture for each member's own probability of being catalogued
    WITH I2 (4.5 um) ITSELF MEASURED -- independent, `p_I2 * (1 -
    prod_{i!=I2}(1-p_i))` -- the exact condition the catalogue side
    applies to a catalogued source (`ORIGIN_FNU[:, I2] == 1`,
    `_observed_bright_counts`); comonotone, `min(p_I2, max_{i!=I2} p_i)`
    (I2 and at least one other band both clear iff the one draw is below
    both p_I2 and the brightest of the other seven).

    `flux_i2` (n_pix, n_mem): the dimmed I2 flux, for the bright tests
    (compared there to 3x/10x the pixel's own I2 50% limit).

    A pixel off-map in a band (no on-map 30'' pixel inside it, module
    docstring of `catalog.completeness.build`) carries an all-zero
    `PIXEL_CURVE_<band>` row there, which `completeness_many` evaluates
    to exactly 0 at any flux: the member is never detected in that band,
    with no NaN at any step, the same outcome the retired `f_lim = +inf`
    convention gave. `has_flux`'s own `np.where` keeps a band with no
    flux at all out of this path already, so the two guards do not
    interact.

    No batching inside: the caller (`catalogued_fraction`, rule 10b) passes
    one pixel batch and its own slice of `a_col`/`pixel_row`; `u` and
    `flux0` are the caller's whole (unsliced) member draw."""
    assert MIN_BANDS_CLEAR == 2, "the closed form below is `P(>=2 of 8)` only"
    n_pix, n_mem = a_col.size, u.size
    p_cat = np.empty((n_pix, n_mem), dtype=np.float64)
    p_cat_i2 = np.empty((n_pix, n_mem), dtype=np.float64)
    flux_i2 = np.empty((n_pix, n_mem), dtype=np.float64)
    # member sub-blocks of `_MEMBER_SUBBLOCK` (F4 2026-10-09 "change B"):
    # every array below is the SAME elementwise-per-(pixel,member)
    # computation on a slice of the member axis, so splitting it changes
    # no value -- `catalogued_fraction`'s own pixel batches and member
    # chunks still see exactly the SAME `p_cat`/`p_cat_i2`/`flux_i2` this
    # function always returned. The working set per sub-block
    # (`n_pix * _MEMBER_SUBBLOCK * 8` elements) fits comfortably in L2
    # cache instead of the full pixel-batch's own `_N_TEMP_ARRAYS *
    # n_mem` DRAM-bound footprint.
    for mstart in range(0, n_mem, _MEMBER_SUBBLOCK):
        mstop = min(mstart + _MEMBER_SUBBLOCK, n_mem)
        u_s = u[mstart:mstop]
        flux0_s = flux0[mstart:mstop]
        # `a`/`w_ramp`/`kappa`/`dimming` below depend on the member only
        # through its own `u`, never through `flux0` -- a member sharing
        # another member's own `u` gets the EXACT SAME `dimming` (GAL's
        # whole population has `u = 1` for every member, sec. 5.4; AGB
        # repeats one star's own `u` once per its own shell template,
        # sec. 5.2), so it is computed once per DISTINCT `u` IN THIS
        # SUB-BLOCK and gathered back to the sub-block's own member axis
        # (F4 2026-10-09 "extinction per distinct (column, u)"):
        # `a_col[p] * u_unique[inverse[m]] == a_col[p] * u[m]` exactly
        # (the same two floats multiplied in the same order), so this is
        # a bit-identical reorganisation of the same arithmetic, not an
        # approximation.
        u_unique, inverse = np.unique(u_s, return_inverse=True)
        a_unique = a_col[:, None] * u_unique[None, :]  # (n_pix, n_unique)
        w_ramp_unique = selection_module.law_dense_weight(a_unique)
        kappa_unique = selection_module.kappa_hybrid(config, w_ramp_unique)  # (n_pix, n_unique, 8)
        dimming_unique = 0.4 * a_unique[:, :, None] * kappa_unique  # (n_pix, n_unique, 8)
        dimming = dimming_unique[:, inverse, :]  # (n_pix, n_sub, 8): gathered to the sub-block
        flux = flux0_s[None, :, :] * 10.0 ** (-dimming)  # (n_pix, n_sub, 8)
        has_flux = flux0_s[None, :, :] > 0.0  # (1, n_sub, 8), broadcasts
        # per band, the MEASURED completeness curve at this pixel's own
        # row (`catalog.completeness.completeness_many`, the design's one
        # kernel), never a Gaussian fit to it: one vectorised call per
        # band over the sub-block's own (pixel, member) flux pairs,
        # capped at the owner's `C_CAP` (`catalogued_probability`'s own
        # docstring).
        p = np.empty_like(flux)
        rows_2d = np.broadcast_to(pixel_row[:, None], flux.shape[:2])
        for bi, band in enumerate(BAND_KEYS):
            curve_b, g0_b, dm_b, n_b, a_b, c_b, scale_b = band_params[band]
            flux_b = flux[:, :, bi]
            has_b = has_flux[:, :, bi]
            log10_f_b = np.log10(np.where(has_b, flux_b, 1.0))
            c_vals = completeness_module.completeness_many(
                curve_b, np.ascontiguousarray(rows_2d).ravel(), g0_b, dm_b, n_b,
                np.ascontiguousarray(log10_f_b).ravel(), a_b, c_b, scale_b
            ).reshape(flux_b.shape)
            p[:, :, bi] = np.where(has_b, np.minimum(c_vals, completeness_module.C_CAP), 0.0)
        one_minus_p = 1.0 - p  # (n_pix, n_sub, 8)
        # the leave-one-out products (F4 2026-10-09 "change A", the single
        # largest kernel cost: 9 `np.delete` copies of this whole (n_pix,
        # n_sub, 7) array plus 9 strided `np.prod` reductions). `prefix[k]`
        # is the LEFT FOLD of bands `0..k-1` on the band-major views
        # `one_minus_p[:, :, k]` (no copy): `prefix[N_BANDS]` is therefore
        # the same left-to-right fold of all 8 bands `np.prod(one_minus_p,
        # axis=2)` performs along its own short contiguous last axis, and
        # leaving band `i` out continues that SAME fold from `prefix[i]`
        # through bands `i+1..N_BANDS-1` one band at a time -- the exact
        # continuation of the one left fold `np.prod(np.delete(...))` used
        # to perform on the remaining 7 bands in their original order,
        # never a separately-folded suffix (a suffix product folds
        # right-to-left, a different association that can move the last
        # bit, F4 risk 1/3).
        prefix = [np.ones(one_minus_p.shape[:2], dtype=one_minus_p.dtype)]
        for k in range(N_BANDS):
            prefix.append(prefix[-1] * one_minus_p[:, :, k])
        prod_all = prefix[N_BANDS]
        loo = []
        for i in range(N_BANDS):
            term = prefix[i]
            for k in range(i + 1, N_BANDS):
                term = term * one_minus_p[:, :, k]
            loo.append(term)
        sum_term = np.zeros_like(prod_all)
        for i in range(N_BANDS):
            sum_term += p[:, :, i] * loo[i]
        p_indep = 1.0 - prod_all - sum_term
        # C20 brief: p_(2), the second-largest of the eight p_b (the
        # comonotone model's own P(>=2 of 8)); `np.partition` finds it
        # without fully sorting the short band axis. At LAMBDA=0.0 exactly,
        # `(1.0 - 0.0) * p_indep + 0.0 * p2 == p_indep` bit for bit (`1.0 *
        # x == x`, `0.0 * finite == 0.0`, `x + 0.0 == x` in IEEE754), so
        # this reproduces today's p_cat bit-identically (the brief's own
        # identity, proved on NGC 7129).
        p2 = np.partition(p, -2, axis=2)[:, :, -2]
        p_cat[:, mstart:mstop] = (1.0 - constants.LAMBDA) * p_indep + constants.LAMBDA * p2

        # the bright-end check (sec. 9): a member counts as bright only if
        # it would be CATALOGUED WITH I2 ITSELF MEASURED (its own I2
        # cleared, `p[:,:,IDX_I2]`, AND at least one of the other seven
        # also clears, `1 - prod_{i!=I2}(1-p_i)`); `flux_i2` lets the
        # caller apply the 3x/10x test against this pixel's own I2 50%
        # limit. `loo[IDX_I2]` is already exactly that leave-I2-out
        # product, computed once above. The SAME LAMBDA mixture as
        # `p_cat` above, the comonotone term now `min(p_I2, max_{i!=I2}
        # p_i)` (C20 brief): `np.delete` here mirrors the ONE already paid
        # in `_collapse_star_members`'s own kind of per-band exclusion,
        # not a new cost class on this already O(8)-per-band path.
        prob_other_detect = 1.0 - loo[IDX_I2]
        p_i2_indep = p[:, :, IDX_I2] * prob_other_detect
        max_other = np.max(np.delete(p, IDX_I2, axis=2), axis=2)
        p_i2_comon = np.minimum(p[:, :, IDX_I2], max_other)
        p_cat_i2[:, mstart:mstop] = (1.0 - constants.LAMBDA) * p_i2_indep + constants.LAMBDA * p_i2_comon
        flux_i2[:, mstart:mstop] = flux[:, :, IDX_I2]  # (n_pix, n_sub)
    return p_cat, p_cat_i2, flux_i2


def _xi_bin_index(u):
    """`(idx, valid)`, a member's own xi-bin index on `grid.LOG10_XI_EDGES`
    for its placement fraction `u` (C29, briefs/round4_2026-10-20/
    C29_atlas_check_dimmed_axis.md): the SAME `nextafter` edge nudge
    `grid.bin` applies to its own `log10 xi` samples, so a member sitting
    exactly at THE EDGE xi = 1 (GAL's `u = 1`, sec. 5.4) lands in the
    SAME cell `grid.bin` would put it in -- this index is pixel-
    independent (xi = a/T = u exactly, `catalogued_probability`'s own
    docstring), computed once per member here rather than once per pixel
    batch. `valid` is False only for a `u` whose nudged `log10 xi` falls
    outside the array's own `[-3.0, +1.0)` extent -- never true for any
    member this build draws (`x = u <= 1` is the whole prior's own
    support, `bmstp.grid`'s own module docstring), kept so a future caller
    cannot silently mis-bin one that does; the returned `idx` is clipped
    into range regardless, so the caller must still gate on `valid`.

    `np.searchsorted` against the edges array directly, NOT a floor-
    divide from `LOG10_XI_EDGES[0]` (a first implementation's own bug,
    caught by this function's own test against `u = 1` exactly): at
    `u = 1`, `log10 xi` nudges to a value so close to `0.0` that `0.0 -
    LOG10_XI_EDGES[0]` (`3.0`) swallows the nudge in double precision
    (`3.0 - 5e-324 == 3.0` exactly) and a floor-divide would silently
    place the member in the WRONG cell -- the edge ξ = 1's own cell,
    never the one below it the module docstring's edge convention
    requires. `searchsorted` instead COMPARES the nudged value against
    each edge directly (no subtraction, no cancellation), the same
    comparison `np.histogram2d` performs internally, so this index
    always agrees with `grid.bin`'s own binning bit for bit."""
    with np.errstate(divide="ignore"):
        log10_xi = np.log10(u)
    log10_xi = np.nextafter(log10_xi, -np.inf)
    n_x = grid.LOG10_XI_EDGES.size - 1
    idx = np.searchsorted(grid.LOG10_XI_EDGES, log10_xi, side="right") - 1
    valid = np.isfinite(log10_xi) & (idx >= 0) & (idx < n_x)
    return np.clip(idx, 0, n_x - 1), valid


def _accumulate_dimmed_hist(hist, x_idx_c, valid_x_c, flux_i2, p_cat, w_c):
    """Add one pixel-batch/member-chunk's own contribution to `hist`
    (n_x, n_b), in place -- C29's dimmed-axis check, computed inside the
    SAME member batches `catalogued_fraction`'s caller already runs, no
    second pass over members. For every (pixel, member) pair in this
    chunk, `weight_pix[pix] * p_cat[pix, m] * w_c[m]` lands at the
    member's own xi-bin (`x_idx_c[m]`, pixel-independent, `_xi_bin_
    index`) and THIS PIXEL's own dimmed I2 flux bin -- `flux_i2`, the
    SAME dimmed flux `catalogued_probability` already formed for the
    bright-end check (sec. 9), never a second dimming computation. A
    non-finite or non-positive dimmed flux (a member with no I2 flux at
    all, or one whose bin falls outside the array's own `log10 F_4.5`
    extent) drops that pair's weight silently, the same convention
    `grid.bin`'s own `np.histogram2d` applies to a raw mark outside its
    array; `weight_pix` is the caller's own argument to `catalogued_
    fraction`, already multiplied in by the caller (`p_cat` alone here,
    not `weight_pix @ p_cat` -- the caller passes this function `p_cat`
    broadcast by `weight_pix` already, see the call site). The y-bin
    index uses `np.searchsorted` against `grid.LOG10_F45_EDGES` directly,
    the same `np.histogram2d` convention `_xi_bin_index`'s own docstring
    explains (no edge nudge on this axis: `grid.bin` never nudges its own
    `log10_f45` samples either, so none is needed here)."""
    n_b = grid.LOG10_F45_EDGES.size - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_f45 = np.log10(flux_i2)
    y_idx = np.searchsorted(grid.LOG10_F45_EDGES, log10_f45, side="right") - 1
    valid = np.isfinite(log10_f45) & (y_idx >= 0) & (y_idx < n_b) & valid_x_c[None, :]
    if not np.any(valid):
        return
    x_bcast = np.broadcast_to(x_idx_c[None, :], flux_i2.shape)
    weight = p_cat * w_c[None, :]
    packed = x_bcast[valid] * n_b + y_idx[valid]
    counts = np.bincount(packed, weights=weight[valid], minlength=hist.size)
    hist += counts.reshape(hist.shape)


def catalogued_fraction(a_col, u, flux0, w, f_lim, pixel_row, band_params, config, weight_pix, tick=None):
    """`(frac, frac_bright3, frac_bright10, s_member, dimmed_hist)`: the weighted,
    deterministic quadrature over a class's WHOLE population (SPEC_BMSTP_DRAFT.md
    sec. 8) -- exact up to the population's own binning, for every class (GAL
    first, sec. 5.4). `w` (n_mem,) is each member's own population weight, on any
    positive scale; every mean below normalises by `w.sum()`. `catalogued_
    probability` supplies the per-pixel, per-member catalogued probability
    (`p_cat`) and the I2-measured probability and dimmed I2 flux for the bright
    tests (sec. 9); this function forms the weighted sums over the class's whole
    population.

    `frac` (n_pix,) = `(p_cat @ w) / w.sum()`; `frac_bright3`/`frac_bright10` (sec. 9
    "the bright-end count ratio") the same weighted mean restricted to members whose
    dimmed I2 flux exceeds 3x/10x the pixel's own I2 50% limit -- the exact condition
    the catalogue side applies to a catalogued source. `s_member` (n_mem,) =
    `sum_pix weight_pix[pix] * p_cat[pix, m]`, the pixel-area-weighted catalogued
    probability of that member over the whole region, exact per member (a region
    total built from `density * (w / w.sum()) @ s_member` needs no further error
    term: the quadrature is exact, not a draw).

    `dimmed_hist` (n_x, n_b), RAW (unsmoothed, unnormalised) weighted counts (C29):
    every (pixel, member) pair's own `weight_pix[pix] * p_cat[pix, m] * w[m]` binned
    at the member's own xi cell and THIS PIXEL's own DIMMED I2 flux cell -- the
    per-bin predicted-catalogued-count CHECK's own histogram, which (unlike
    `s_member`) cannot collapse to one point per member, since dimming (and so the
    bin a member's own catalogued probability lands in) varies pixel to pixel. The
    caller still forms its own `n_cat_cell` from this the same way it forms one
    from `s_member` today -- `_safe_cell_bin_counts` in place of `_safe_cell_bin` --
    scaling by whatever per-class, per-member-independent factor (`density /
    w.sum()`, or just `1 / w.sum()` where density is already folded into
    `weight_pix`) it already applies to `s_member`-based `c_m`. No second pass over
    members: `_accumulate_dimmed_hist` reads the SAME `p_cat`/`flux_i2` this
    function's own pixel-batch/member-chunk loop already computed for `frac`/
    `s_member`.

    Processed in pixel batches of `_pixel_batch_size` (rule 10b): each batch is the
    same elementwise-per-pixel computation on a slice of `a_col`/`f_lim`/`pixel_row`,
    so splitting the pixel axis changes no result. Within one pixel batch the member
    axis is ALSO chunked, in
    `n_mem_chunk = _PIXEL_BATCH_BUDGET_BYTES // (N_BANDS * 8 * _N_TEMP_ARRAYS *
    n_pix_batch)` members at a time (AGB's cross-product population, sec. 5.2, can
    reach ~1e6 members per tile: a `(1, 1e6, 8)` working set at one pixel per batch
    would still exceed the 512 MB budget even at the smallest pixel batch, so the
    member axis is chunked too, sized to the batch's own pixel count). Each chunk's
    weighted sum accumulates into `frac`/`frac_bright3`/`frac_bright10` (a sum over
    disjoint member chunks is exact, no result changes for a member set that fit
    before) and fills its own slice of `s_member` directly; `dimmed_hist` accumulates
    the same way, in place, across every batch and chunk."""
    n_mem = u.size
    n_pix = a_col.size
    w_sum = float(w.sum())
    frac = np.empty(n_pix, dtype=np.float64)
    frac_bright3 = np.empty(n_pix, dtype=np.float64)
    frac_bright10 = np.empty(n_pix, dtype=np.float64)
    s_member = np.zeros(n_mem, dtype=np.float64)
    n_x, n_b = grid.LOG10_XI_EDGES.size - 1, grid.LOG10_F45_EDGES.size - 1
    dimmed_hist = np.zeros((n_x, n_b), dtype=np.float64)
    x_idx, valid_x = _xi_bin_index(u)
    batch = _pixel_batch_size(n_mem)
    n_batches = (n_pix + batch - 1) // batch
    for b, start in enumerate(range(0, n_pix, batch)):
        stop = min(start + batch, n_pix)
        n_pix_batch = stop - start
        f_lim_b = f_lim[start:stop]
        pixel_row_b = pixel_row[start:stop]
        a_col_b = a_col[start:stop]
        i2_lim_b = f_lim_b[:, IDX_I2:IDX_I2 + 1]  # (n_pix_batch, 1)

        weighted_sum = np.zeros(n_pix_batch, dtype=np.float64)
        weighted_bright3 = np.zeros(n_pix_batch, dtype=np.float64)
        weighted_bright10 = np.zeros(n_pix_batch, dtype=np.float64)
        n_mem_chunk = max(1, _PIXEL_BATCH_BUDGET_BYTES
                           // (N_BANDS * 8 * _N_TEMP_ARRAYS * n_pix_batch))
        for mstart in range(0, n_mem, n_mem_chunk):
            mstop = min(mstart + n_mem_chunk, n_mem)
            u_c = u[mstart:mstop]
            flux0_c = flux0[mstart:mstop]
            w_c = w[mstart:mstop]
            p_cat, catalogued_i2_measured, flux_i2 = catalogued_probability(
                a_col_b, u_c, flux0_c, pixel_row_b, band_params, config)
            weighted_sum += p_cat @ w_c

            bright3 = flux_i2 > BRIGHT_MULT_3 * i2_lim_b
            bright10 = flux_i2 > BRIGHT_MULT_10 * i2_lim_b
            weighted_bright3 += (catalogued_i2_measured * bright3) @ w_c
            weighted_bright10 += (catalogued_i2_measured * bright10) @ w_c

            s_member[mstart:mstop] += weight_pix[start:stop] @ p_cat
            _accumulate_dimmed_hist(
                dimmed_hist, x_idx[mstart:mstop], valid_x[mstart:mstop], flux_i2,
                weight_pix[start:stop, None] * p_cat, w_c)
        frac[start:stop] = weighted_sum / w_sum
        frac_bright3[start:stop] = weighted_bright3 / w_sum
        frac_bright10[start:stop] = weighted_bright10 / w_sum
        if tick is not None:
            tick(b + 1, n_batches)
    return frac, frac_bright3, frac_bright10, s_member, dimmed_hist


def catalogued_fraction_pair(a_col, u, flux0, w_a, w_b, f_lim, pixel_row, band_params, config, weight_pix):
    """`(frac_a, frac_bright3_a, frac_bright10_a, frac_b, frac_bright3_b,
    frac_bright10_b, s_member, dimmed_hist_a, dimmed_hist_b)`:
    `catalogued_fraction`'s own quadrature, run ONCE for two classes that
    share one member set (STAR/PAHC, `_build_one_tile`, F4 2026-10-09
    'change C') instead of once each.
    `u`/`flux0` are the two classes' shared collapsed cells (their
    weights' own UNION support); `w_a`/`w_b` are each class's own weight
    over that SAME union, zero wherever a cell belongs to the other
    class only. `catalogued_probability` depends on `a_col`/`u`/`flux0`/
    `pixel_row`/`band_params` only, never on the weight, so it is evaluated
    once per pixel batch/member chunk and both classes' weighted sums
    are formed from that ONE evaluation (a zero weight contributes
    exactly 0.0 to a dot product, `x + w*0 == x`). `s_member` -- `sum_pix
    weight_pix[pix] * p_cat[pix, m]` -- does not depend on either
    weight at all, so it is the SAME single array for both classes (not
    one each): a cell both classes share gets one `s_member` value,
    read by both classes' own `c_m = density * (w / w.sum()) *
    s_member` downstream. Chunked exactly as `catalogued_fraction`, over
    the union's own member count -- bit-identical to summing `w_a`/`w_b`
    separately is NOT claimed (the union's member count sets different
    pixel-batch/member-chunk boundaries than each class's own former,
    smaller compacted count would), only mathematically exact; measured
    against the un-shared baseline (brief's 1e-13 bar).

    `dimmed_hist_a`/`dimmed_hist_b` (n_x, n_b) are C29's own raw
    dimmed-axis check histogram (`catalogued_fraction`'s own docstring),
    one per class since they differ in their member WEIGHT (`w_a`/`w_b`)
    even though they share one `p_cat`/`flux_i2` evaluation -- the same
    `_accumulate_dimmed_hist` call, run twice per chunk (once per class's
    own weight), no second `catalogued_probability` evaluation."""
    n_mem = u.size
    n_pix = a_col.size
    w_a_sum = float(w_a.sum())
    w_b_sum = float(w_b.sum())
    # a class with no member at all in this union (`w_a`/`w_b` all
    # zero -- e.g. a tile with no PAHC members) would divide 0/0 below;
    # `catalogued_fraction`'s own caller instead special-cased
    # `w.size == 0` to an all-zero `frac`. Dividing by 1.0 here gives the
    # SAME zero (the numerator is already exactly 0.0 when the weight is
    # all zero), with no branch needed.
    w_a_sum_safe = w_a_sum if w_a_sum > 0 else 1.0
    w_b_sum_safe = w_b_sum if w_b_sum > 0 else 1.0
    frac_a = np.empty(n_pix, dtype=np.float64)
    frac_bright3_a = np.empty(n_pix, dtype=np.float64)
    frac_bright10_a = np.empty(n_pix, dtype=np.float64)
    frac_b = np.empty(n_pix, dtype=np.float64)
    frac_bright3_b = np.empty(n_pix, dtype=np.float64)
    frac_bright10_b = np.empty(n_pix, dtype=np.float64)
    s_member = np.zeros(n_mem, dtype=np.float64)
    n_x, n_b = grid.LOG10_XI_EDGES.size - 1, grid.LOG10_F45_EDGES.size - 1
    dimmed_hist_a = np.zeros((n_x, n_b), dtype=np.float64)
    dimmed_hist_b = np.zeros((n_x, n_b), dtype=np.float64)
    x_idx, valid_x = _xi_bin_index(u)
    batch = _pixel_batch_size(n_mem)
    for start in range(0, n_pix, batch):
        stop = min(start + batch, n_pix)
        n_pix_batch = stop - start
        f_lim_b = f_lim[start:stop]
        pixel_row_b = pixel_row[start:stop]
        a_col_b = a_col[start:stop]
        i2_lim_b = f_lim_b[:, IDX_I2:IDX_I2 + 1]  # (n_pix_batch, 1)

        weighted_sum_a = np.zeros(n_pix_batch, dtype=np.float64)
        weighted_bright3_a = np.zeros(n_pix_batch, dtype=np.float64)
        weighted_bright10_a = np.zeros(n_pix_batch, dtype=np.float64)
        weighted_sum_b = np.zeros(n_pix_batch, dtype=np.float64)
        weighted_bright3_b = np.zeros(n_pix_batch, dtype=np.float64)
        weighted_bright10_b = np.zeros(n_pix_batch, dtype=np.float64)
        n_mem_chunk = max(1, _PIXEL_BATCH_BUDGET_BYTES
                           // (N_BANDS * 8 * _N_TEMP_ARRAYS * n_pix_batch))
        for mstart in range(0, n_mem, n_mem_chunk):
            mstop = min(mstart + n_mem_chunk, n_mem)
            u_c = u[mstart:mstop]
            flux0_c = flux0[mstart:mstop]
            wa_c = w_a[mstart:mstop]
            wb_c = w_b[mstart:mstop]
            p_cat, catalogued_i2_measured, flux_i2 = catalogued_probability(
                a_col_b, u_c, flux0_c, pixel_row_b, band_params, config)
            weighted_sum_a += p_cat @ wa_c
            weighted_sum_b += p_cat @ wb_c

            bright3 = flux_i2 > BRIGHT_MULT_3 * i2_lim_b
            bright10 = flux_i2 > BRIGHT_MULT_10 * i2_lim_b
            cat_bright3 = catalogued_i2_measured * bright3
            cat_bright10 = catalogued_i2_measured * bright10
            weighted_bright3_a += cat_bright3 @ wa_c
            weighted_bright10_a += cat_bright10 @ wa_c
            weighted_bright3_b += cat_bright3 @ wb_c
            weighted_bright10_b += cat_bright10 @ wb_c

            s_member[mstart:mstop] += weight_pix[start:stop] @ p_cat
            weighted_pix_p_cat = weight_pix[start:stop, None] * p_cat
            x_idx_c, valid_x_c = x_idx[mstart:mstop], valid_x[mstart:mstop]
            _accumulate_dimmed_hist(dimmed_hist_a, x_idx_c, valid_x_c, flux_i2, weighted_pix_p_cat, wa_c)
            _accumulate_dimmed_hist(dimmed_hist_b, x_idx_c, valid_x_c, flux_i2, weighted_pix_p_cat, wb_c)
        frac_a[start:stop] = weighted_sum_a / w_a_sum_safe
        frac_bright3_a[start:stop] = weighted_bright3_a / w_a_sum_safe
        frac_bright10_a[start:stop] = weighted_bright10_a / w_a_sum_safe
        frac_b[start:stop] = weighted_sum_b / w_b_sum_safe
        frac_bright3_b[start:stop] = weighted_bright3_b / w_b_sum_safe
        frac_bright10_b[start:stop] = weighted_bright10_b / w_b_sum_safe
    return (frac_a, frac_bright3_a, frac_bright10_a,
            frac_b, frac_bright3_b, frac_bright10_b, s_member,
            dimmed_hist_a, dimmed_hist_b)


def _pahc_weight(limit8_grid, p_pahc, x):
    """`P_PAHC` interpolated to the flux limit `x` (a scalar mJy, the
    tile's own mean I4 limit -- an approximation of sec. 4's per-pixel
    contrast, disclosed rather than refit per pixel, since `P_PAHC`
    is tabulated on `LIMIT8_GRID_MJY`'s eight region-wide quantile nodes,
    not per pixel): vectorised over stars via `searchsorted`, no
    per-star loop. `frac` is clamped to `[0, 1]` (no extrapolation past
    the grid's own end nodes): `x` is now the tile's own pixels' marginalised
    `F_LIM_50_PIX_MJY` mean (`catalog.depth_grid`, sec. 3.3), which a single
    poorly-covered pixel can push past the region-wide grid's brightest
    node, where `P_PAHC` itself is not defined."""
    j = np.clip(np.searchsorted(limit8_grid, x), 1, limit8_grid.size - 1)
    lo, hi = j - 1, j
    frac = np.clip((x - limit8_grid[lo]) / (limit8_grid[hi] - limit8_grid[lo]), 0.0, 1.0)
    return p_pahc[:, lo] + frac * (p_pahc[:, hi] - p_pahc[:, lo])


def _collapse_star_members(u, flux0, w_star, w_star_only, w_pahc_only, a_star_k):
    """Collapse the tile's field-star members into cells before
    `catalogued_fraction` (`_build_one_tile`, rule 9): a member enters
    `catalogued_probability` only through its placement fraction `u`, its
    4.5 um flux `F_4.5 = flux0[:, IDX_I2]`, and its seven flux ratios to
    I2 (every other band). Two stars that agree in those nine quantities
    to within a cell width far below the completeness roll-off
    (`W_DEX` ~= 0.1-0.3 dex, `catalog.depth_grid`) are one member:
    `log10 u` and `log10 F_4.5` at the cell grid's OWN bin widths
    (`grid.LOG10_XI_EDGES`' 1/32 dex, `grid.D_LOG10_F45`'s 0.1 dex -- the
    collapse can never blur two stars the cell grid would itself
    resolve), each of the seven log ratios at `STAR_COLOUR_CELL_DEX`.

    The nine per-star cell indices (one floor-divide array expression
    each, rule 8) are grouped with one `np.unique(..., axis=0,
    return_inverse=True)` call -- a vector generalisation of
    `_gal_members`'s single packed 64-bit integer key: nine independent
    dex-scale axes do not fit one 64-bit key with a safe margin against
    every realistic star (a silent overflow would merge unrelated cells,
    rule 6's "no silent fallbacks"), so the row of nine indices is the
    key itself, still one vectorised grouping with no Python loop over
    stars.

    Returns the cell's `u`/`flux0`/`a_star_k` (the ORIGINAL per-star
    weight `w_star`-weighted mean within the cell -- each star's own
    population weight, before the STAR/PAHC split) and the cell's summed
    STAR-only and PAHC-only weights (the SAME cell grouping serves both
    classes, sec. 5.1/5.3, since a star's placement/colour does not
    depend on which class it falls in), plus the member counts
    before/after for the build's own report. `a_star_k` (C26 owner
    ruling 2026-10-21, briefs/round4_2026-10-20/C26_profile_adopted.md):
    each member's own cell-mean of `population_star_tile`'s own
    `A_STAR_K` -- the star's own UNCLIPPED local extinction at its own
    drawn pixel, collapsed at the SAME `log10 u` cell width as `u`
    itself (far below the completeness roll-off), so the quadrature
    reads each member's OWN column directly (`catalogued_probability`
    called with `a_col` fixed at 1 and `u` replaced by this array, the
    caller's docstring) instead of recombining a DIFFERENT pixel's own
    column with this member's clipped placement fraction."""
    n_before = int(u.size)
    tiny = np.finfo(np.float64).tiny
    log10_u = np.log10(np.maximum(u, tiny))
    log10_f45 = np.log10(np.maximum(flux0[:, IDX_I2], tiny))
    x_width = grid.LOG10_XI_EDGES[1] - grid.LOG10_XI_EDGES[0]
    cell_u = np.floor(log10_u / x_width).astype(np.int64)
    cell_f45 = np.floor(log10_f45 / grid.D_LOG10_F45).astype(np.int64)
    ratio_bands = [k for k in range(N_BANDS) if k != IDX_I2]
    ratio_cells = [
        np.floor((np.log10(np.maximum(flux0[:, k], tiny)) - log10_f45) / STAR_COLOUR_CELL_DEX).astype(np.int64)
        for k in ratio_bands]
    key = np.column_stack([cell_u, cell_f45] + ratio_cells)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    n_after = int(inverse.max()) + 1 if inverse.size else 0

    w_cell = np.bincount(inverse, weights=w_star, minlength=n_after)
    w_cell_star_only = np.bincount(inverse, weights=w_star_only, minlength=n_after)
    w_cell_pahc_only = np.bincount(inverse, weights=w_pahc_only, minlength=n_after)
    safe_w = np.where(w_cell > 0, w_cell, 1.0)
    u_cell = np.bincount(inverse, weights=w_star * u, minlength=n_after) / safe_w
    a_star_cell = np.bincount(inverse, weights=w_star * a_star_k, minlength=n_after) / safe_w
    flux0_cell = np.empty((n_after, N_BANDS), dtype=np.float64)
    for k in range(N_BANDS):
        flux0_cell[:, k] = np.bincount(inverse, weights=w_star * flux0[:, k], minlength=n_after) / safe_w
    return u_cell, flux0_cell, a_star_cell, w_cell_star_only, w_cell_pahc_only, n_before, n_after


def _build_one_tile(config, region, tile_id, pix_in_tile, a_col_in_tile, f_lim_in_tile, pixel_row_in_tile,
                     agb_pool, coverage_in_tile):
    """One tile's `{cls: (frac, density, frac_bright3, frac_bright10,
    n_cat_cell_tile, n_cell_tile)}` for STAR/AGB/PAHC, over its own admitted pixels,
    from the star-family population's own retained sample (`population/
    star/population_star_tile__R.hdf5`'s `tile_<id>` group): each class's
    catalogued fraction is the WEIGHTED SUM over every one of its own
    members (`catalogued_fraction`, sec. 8), no draw. STAR's and PAHC's
    members collapse into cells first (`_collapse_star_members`, rule 9:
    `log10 u` and `log10 F_4.5` at the cell grid's own bin widths, each of
    the seven flux ratios to I2 at `STAR_COLOUR_CELL_DEX` -- all far below
    the completeness roll-off, so the collapse moves `catalogued_fraction`
    by much less than the roll-off itself resolves). AGB's members are
    `sample_star.sample_agb`'s own JOINT measure of evolved star and
    shell template (FINDINGS.md S10, coordinator's joint-measure ruling;
    sec. 5.2: the member's own `F_4.5` is the shell flux, not TRILEGAL's
    photosphere), read directly -- no second crossing against a template
    pool -- each member's own template's eight `F_REF` rescaled so its
    own 4.5 um reference flux equals that member's OWN joint-measure
    `F_4.5`. `pixel_row_in_tile` (n_pix_in_tile,)
    is this tile's own slice of `bmstp.atlas._depth_grid`'s row index
    into `PIXEL_CURVE_<band>` (`catalogued_probability`'s own docstring):
    opened once here (`_completeness_band_params`, this worker's
    per-process cache) rather than reduced to a region-wide width, since
    the measured curve now feeds `catalogued_probability` directly.
    `n_cat_cell_tile` (128, 110) is this tile's own contribution to the
    region's expected catalogued count per parameter cell (module
    docstring's `N_CAT_CELL_GAL`, the same construction here): the class's
    members at their own `x`, binned at EACH PIXEL's own DIMMED I2 flux
    (C29, module docstring) rather than one undimmed point per member --
    `catalogued_fraction_pair`'s own `dimmed_hist`, a per-(pixel, member)
    weighted accumulation inside its existing member batches, rescaled
    here by this call's one per-member-independent factor `density /
    w.sum()` (`_safe_cell_bin_counts`, in place of `_safe_cell_bin`).
    `n_cell_tile` (128, 110) is the SAME members quadrature with the
    tile's own pixel-area total (`weight_pix.sum()`) standing in for
    `s_member` -- every member's own probability of being catalogued set
    to one, no flux cut, no dimming, STILL binned at the member's own
    UNDIMMED `log10 F_4.5` (module docstring): `c_m_all = density * (w /
    w.sum()) * weight_pix.sum()`, the same `_safe_cell_bin(u, log10
    F_4.5, c_m_all)` as before C29. The two axes now differing, `n_cat_
    cell_tile <= n_cell_tile` cell by cell no longer holds by
    construction (module docstring) -- only `n_cat_cell_tile.sum() <=
    n_cell_tile.sum()` does.

    The returned dict also carries `"_FIELD_STAR_DENSITY"`, the tile's
    own UNSPLIT field-star level `w_star.sum() / omega_t` (`density.
    density_star_raw`'s own quantity, before the STAR/PAHC retention
    split): sec. 5.3's `A_PAHC(s) = A_STAR(s)`, the field-star intensity
    both classes' intrinsic path (`build_region`'s `INTENSITY_STAR`/
    `INTENSITY_PAHC`) shares, the split itself carried only by the
    template weight factor `_above_fraction` applies (sec. 5.1's
    `1 - P(q)`/`P(q)`), never by `INTENSITY_<C>` again. `STAR`'s and
    `PAHC`'s own `density` (this function's per-class split density,
    `w_star_only.sum()`/`w_pahc_only.sum()` over `omega_t`) still feeds
    `N_CAT_<C>` and the members path exactly as before -- untouched."""
    band_params = _completeness_band_params(config, region)
    star_path = config_module.product_path(
        config, "population", "star", "population", "tile", region=region)
    field_path = config_module.product_path(
        config, "population", "trilegal", "field-stars", "region", region=region)
    with h5py.File(star_path, "r") as f:
        limit8_grid = np.asarray(f["LIMIT8_GRID_MJY"][()], dtype=np.float64)
        grp = f[f"tile_{tile_id}"]
        # this tile's sample is drawn from ONE TRILEGAL pointing (owner
        # ruling 2026-09-06); its own OMEGA_POINTING_DEG2, not the
        # region's whole-simulation OMEGA_SIM_DEG2, is the solid angle it
        # covers.
        omega_t = float(grp.attrs["OMEGA_POINTING_DEG2"])
        star_index = np.asarray(grp["STAR_INDEX"][()], dtype=np.int64)
        u = np.asarray(grp["XI"][()], dtype=np.float64)
        a_star_k = np.asarray(grp["A_STAR_K"][()], dtype=np.float64)
        w_star = np.asarray(grp["W_STAR"][()], dtype=np.float64)
        p_pahc_grid = np.asarray(grp["P_PAHC"][()], dtype=np.float64)  # (n_star, 8)
        evolved_for_agb = np.asarray(grp["IS_EVOLVED"][()], dtype=bool)
    with h5py.File(field_path, "r") as f:
        flux0_all = np.asarray(f["FNU_MJY"][star_index], dtype=np.float64)  # (n_star, 8)

    tile_i4_limit = float(np.mean(f_lim_in_tile[:, IDX_I4])) if pix_in_tile.size else float(limit8_grid[len(limit8_grid) // 2])
    p_pahc = _pahc_weight(limit8_grid, p_pahc_grid, tile_i4_limit)

    # STAR and PAHC partition the field population (spec sec. 5.1, 5.3;
    # coordinator ruling): a star is EITHER a STAR member or a PAHC member,
    # weighted `W_STAR*(1-P_PAHC)` / `W_STAR*P_PAHC`, so `N_CAT_STAR +
    # N_CAT_PAHC` never exceeds the field-star count.
    w_star_only = w_star * (1.0 - p_pahc)
    w_pahc_only = w_star * p_pahc

    # each pixel's own share of the tile's admitted area (sec. 8): the
    # coefficient `catalogued_fraction`'s `s_member` and the cell grid's
    # own catalogued count per member are built from.
    weight_pix = coverage_in_tile * _HPX512_PIXEL_DEG2
    n_x, n_b = grid.LOG10_XI_EDGES.size - 1, grid.LOG10_F45_EDGES.size - 1

    # STAR's and PAHC's members collapse into cells BEFORE `catalogued_fraction`
    # (`_collapse_star_members`, rule 9): the SAME collapsed cells serve both
    # classes, since a star's placement/colour does not depend on which class
    # it falls in, only its STAR-only/PAHC-only weight does.
    u_c, flux0_c, a_star_c, w_star_c, w_pahc_c, n_members_before, n_members_after = (
        _collapse_star_members(u, flux0_all, w_star, w_star_only, w_pahc_only, a_star_k))

    weight_pix_total = float(weight_pix.sum())
    density_star = float(w_star_only.sum()) / omega_t  # objects deg^-2, sec. 5.1/5.2's Omega_pointing
    density_pahc = float(w_pahc_only.sum()) / omega_t
    out = {}
    # STAR and PAHC share the SAME collapsed cells (`u_c`/`flux0_c`
    # above): `catalogued_probability` depends on the cell, never on
    # which class's weight is being summed, so the two classes'
    # quadratures run as ONE call over the union of their own nonzero
    # cells (`catalogued_fraction_pair`, F4 2026-10-09 "change C")
    # instead of `_build_one_tile`'s former two separate
    # `catalogued_fraction` calls on the identical members.
    m_union = (w_star_c > 0) | (w_pahc_c > 0)
    if not np.any(m_union):
        for cls, density in (("STAR", density_star), ("PAHC", density_pahc)):
            out[cls] = (np.zeros(pix_in_tile.size), density, np.zeros(pix_in_tile.size),
                        np.zeros(pix_in_tile.size), np.zeros((n_x, n_b)), np.zeros((n_x, n_b)))
    else:
        u_u, flux0_u, a_star_u = u_c[m_union], flux0_c[m_union], a_star_c[m_union]
        w_star_u, w_pahc_u = w_star_c[m_union], w_pahc_c[m_union]
        log10_f45_u = np.log10(flux0_u[:, IDX_I2])
        # C26 (owner ruling 2026-10-21): each member's own A_STAR_K is
        # already its own absolute local extinction -- `a_col` fixed at 1
        # so `catalogued_probability`'s `a = a_col * u` becomes `a_star_u`
        # unchanged, broadcast identically to every pixel in the tile,
        # instead of recombined with THIS tile's own per-pixel A_COL_K
        # (the retired pattern: beyond the Edenhofer cube's edge that
        # recombination would silently drop the far-field reddening XI's
        # own clip discards). `u_u` (XI, clipped) still only sets the
        # cell's grid position (`_safe_cell_bin` below), never the dimming.
        (frac_star, fb3_star, fb10_star,
         frac_pahc, fb3_pahc, fb10_pahc, s_member,
         dimmed_hist_star, dimmed_hist_pahc) = catalogued_fraction_pair(
            np.ones_like(a_col_in_tile), a_star_u, flux0_u, w_star_u, w_pahc_u, f_lim_in_tile,
            pixel_row_in_tile, band_params, config, weight_pix)
        for cls, density, w_u, frac, frac_bright3, frac_bright10, dimmed_hist in (
                ("STAR", density_star, w_star_u, frac_star, fb3_star, fb10_star, dimmed_hist_star),
                ("PAHC", density_pahc, w_pahc_u, frac_pahc, fb3_pahc, fb10_pahc, dimmed_hist_pahc)):
            w_sum = float(w_u.sum())
            if w_sum == 0:
                n_cat_cell_tile = np.zeros((n_x, n_b))
                n_cell_tile = np.zeros((n_x, n_b))
            else:
                # C29: the CHECK's own histogram (N_CAT_CELL_<C>) bins each
                # member's catalogued probability at its DIMMED I2 flux
                # (`dimmed_hist`, `catalogued_fraction_pair`'s own per-
                # (pixel, member) accumulation) -- N_CELL_<C> below is
                # UNCHANGED, still the member's own UNDIMMED flux
                # (`log10_f45_u`), sec. 8's intrinsic population, never
                # thinned by dimming or a flux cut at all.
                n_cat_cell_tile = _safe_cell_bin_counts(dimmed_hist, density / w_sum)
                c_m_all = density * (w_u / w_sum) * weight_pix_total
                n_cell_tile = _safe_cell_bin(u_u, log10_f45_u, c_m_all)
            out[cls] = (frac, density, frac_bright3, frac_bright10, n_cat_cell_tile, n_cell_tile)

    # AGB (sec. 5.2, FINDINGS.md S10, coordinator's joint-measure ruling):
    # `sample_star.sample_agb` already places one member per (evolved
    # star, own-chemistry template) pair -- the joint measure's own
    # sample, mass `w_a[row]` already resolved -- so there is no second
    # crossing against a template pool left to build here. Each row's
    # `template_index_a` names its own register row directly;
    # `agb_star_distances` repeats this tile's evolved stars' A_STAR_K
    # the SAME way it repeats their distance, so it lines up with
    # `x_a`/`f45_a`/`template_index_a` row for row (C26 owner ruling
    # 2026-10-21: each member's own absolute extinction, not a second
    # mark derived from a_col_in_tile).
    x_a, f45_a, w_a, template_index_a = sample_star.sample_agb(config, region, tile_id)
    density_agb = float(w_a.sum()) / omega_t
    a_star_evolved = a_star_k[evolved_for_agb]
    names_a, chem_a, _log10_r_a = sample_star.agb_log10_ratio_per_template(config.inputs["sed_models"])
    n_o_a, n_c_a = int(np.sum(chem_a == "O")), int(np.sum(chem_a == "C"))
    a_star_rows = np.concatenate([np.repeat(a_star_evolved, n_o_a), np.repeat(a_star_evolved, n_c_a)])

    keep = w_a > 0
    if not np.any(keep):
        frac_agb = np.zeros(pix_in_tile.size)
        frac_agb_bright3 = np.zeros(pix_in_tile.size)
        frac_agb_bright10 = np.zeros(pix_in_tile.size)
        n_cat_cell_agb = np.zeros((n_x, n_b))
        n_cell_agb = np.zeros((n_x, n_b))
    else:
        u_agb = x_a[keep]
        a_star_agb = a_star_rows[keep]
        w_agb_member = w_a[keep]
        f45_agb_member = f45_a[keep]  # already log10 F_4.5 (mJy)
        tmpl_agb = template_index_a[keep]
        f_ref_i2 = np.maximum(agb_pool["f_ref"]["I2"][tmpl_agb], agb_pool["floor_linear"][tmpl_agb])
        scale = (10.0 ** f45_agb_member) / f_ref_i2  # rescales the WHOLE shell SED
        flux0_agb = np.empty((tmpl_agb.size, N_BANDS), dtype=np.float64)
        for k, key in enumerate(BAND_KEYS):
            f_band = np.maximum(agb_pool["f_ref"][key][tmpl_agb], agb_pool["floor_linear"][tmpl_agb])
            flux0_agb[:, k] = f_band * scale
        # C26: each AGB member's own shell carries its OWN star's A_STAR_K
        # (absolute), not a_col_in_tile's per-pixel column times XI.
        frac_agb, frac_agb_bright3, frac_agb_bright10, s_member_agb, dimmed_hist_agb = catalogued_fraction(
            np.ones_like(a_col_in_tile), a_star_agb, flux0_agb, w_agb_member, f_lim_in_tile,
            pixel_row_in_tile, band_params, config, weight_pix)
        w_sum_agb = float(w_agb_member.sum())
        # C29: N_CAT_CELL_AGB bins at the dimmed I2 flux; N_CELL_AGB is
        # unchanged (undimmed, no selection).
        n_cat_cell_agb = _safe_cell_bin_counts(dimmed_hist_agb, density_agb / w_sum_agb)
        c_m_agb_all = density_agb * (w_agb_member / w_sum_agb) * weight_pix_total
        n_cell_agb = _safe_cell_bin(u_agb, f45_agb_member, c_m_agb_all)
    out["AGB"] = (frac_agb, density_agb, frac_agb_bright3, frac_agb_bright10, n_cat_cell_agb, n_cell_agb)
    # sec. 5.3's shared field-star intensity (`build_one_tile`'s own
    # docstring paragraph): the UNSPLIT field-star level, before the
    # STAR/PAHC retention split above -- `density.density_star_raw`'s
    # own quantity, `w_star.sum()` (not `w_star_only`/`w_pahc_only`)
    # over this tile's own `omega_t`.
    out["_FIELD_STAR_DENSITY"] = float(w_star.sum()) / omega_t
    return out


def _agb_shell_pool(config):
    """The AGB register's own eight `F_REF` and `FLOOR_LINEAR` (sec.
    3.5's FREFRAW convention), in the register's OWN row order -- the
    SAME order `sample_star.sample_agb`'s own `template_index` indexes
    into (both read the register directly, sec 1.4). `build_one_tile`'s
    AGB block (FINDINGS.md S10, coordinator's joint-measure ruling) reads
    `sample_agb`'s joint-measure rows directly and indexes this dict by
    their own `template_index`, row for row -- there is no chemistry
    pool or population weight kept here any more: `sample_agb`'s own
    `w` already carries the fully-resolved per-(star, template) mass,
    so there is no second crossing against a template pool left to do.
    Survey-wide, independent of region -- the caller computes this once
    and reuses it, rather than this function caching on an unhashable
    `Config` (its `inputs` mapping)."""
    reg = template_weights._read_register(config, "agb")
    return dict(f_ref=reg["f_ref"], floor_linear=reg["floor_linear"])


def _yso_register(config):
    """The YSO register's own SURVEY-WIDE `population` weight
    (`template_weights.yso_population_weight`, sec. 5.5 "Template
    weights", sec. 1.4, owner's ruling 2026-09-09) and eight `F_REF`, in
    the register's own row order -- called directly rather than
    re-derived, the SAME construction `build_yso` uses. Survey-wide,
    independent of region, computed once for the whole build and cached
    by the caller (`build`) -- the diskless Class III members' own
    region-specific share is NOT folded in here (owner's ruling
    2026-10-20 correction: carried instead by `_yso_region_weight`,
    applied inside `_yso_region_nodes` once per region, the one place
    this dict's own `weight` is actually consumed)."""
    reg = template_weights._read_register(config, "yso")
    f_ref, floor_linear = reg["f_ref"], reg["floor_linear"]
    names_w, weight = template_weights.yso_population_weight(config)
    if names_w.size != reg["names"].size or not np.all(names_w == reg["names"]):
        raise ValueError("bmstp.atlas: yso_population_weight's row order "
                          "disagrees with the yso register")
    return dict(weight=weight, f_ref=f_ref, floor_linear=floor_linear)


def _yso_region_weight(config, region, reg):
    """`reg`'s own survey-wide `weight` (`_yso_register`, cached,
    computed once for the whole build), scaled by this region's own
    `DISK_FRACTION` plus its own diskless Class III members' share
    added on top (`template_weights.yso_diskless_weight`, the photospheric
    brightness density, not the census's) -- owner's ruling 2026-10-20:
    "the atlas needs no change" (this unit's own earlier report) was
    wrong; the atlas's own Monte Carlo YSO member draw (`_yso_region_
    nodes`) must carry the SAME diskless component `template_weights.
    build_yso`'s P5 conditional table does, region by region, or the
    atlas's own `N_CAT_CELL_YSO` total scales up (through `DENSITY_YSO`'s
    corrected level) while every one of those extra members is still
    drawn from the OLD, census-only brightness distribution -- measured
    and reported, C11 brief round 2. `Sigma weight_region = DISK_FRACTION
    + CLASS_SHARE_DISKLESS = 1` exactly, same identity `build_yso` keeps.
    Bit-identical to `reg["weight"]` alone at the identity case
    (DISK_FRACTION = 1): `yso_diskless_weight` returns all zeros without
    reading the region's own age."""
    disk_frac = yso_law_module.disk_fraction(config, region)
    _names_d, w_diskless = template_weights.yso_diskless_weight(config, region)
    return disk_frac * reg["weight"] + w_diskless


def _quantile_nodes_from_weight(values, weight, n):
    """`n` equally spaced quantile nodes `(i+0.5)/n` of `values` weighted
    by `weight` (any positive scale, need not be pre-sorted): `values`
    sorted, the weight's cumulative sum normalised, `np.searchsorted` at
    each quantile -- the same construction the template register's own
    quantile nodes use, generalised to an arbitrary (values, weight)
    pair (`shift_quantile_nodes`, `_depth_nodes`)."""
    order = np.argsort(values)
    v_sorted, w_sorted = values[order], weight[order]
    csum = np.cumsum(w_sorted) / w_sorted.sum()
    q = (np.arange(n) + 0.5) / n
    idx = np.clip(np.searchsorted(csum, q), 0, v_sorted.size - 1)
    return v_sorted[idx]


def _yso_region_nodes(reg, config, region, d_front, d_back):
    """`(flux0_ts, w_ts)`, region-wide, built once (sec. 5.5 "Marks"):
    the product of two independent quadratures, each a set of nodes with
    weights summing to one. Templates: `N_YSO_NODES` equally spaced
    quantiles of the register's own census weight (`reg`, `_yso_register`'s
    own survey-wide return -- computed once for the whole build by the
    caller, `build`, and passed in here rather than re-derived per
    region), each carrying weight `1/N_YSO_NODES`, its own eight `F_REF`
    floored at the register's own `FLOOR_LINEAR`. Distance shift:
    `N_YSO_SHIFT_NODES` equally spaced quantiles of the region's own
    shift kernel (`sample_cloud.shift_kernel`), each a node at its own
    cell centre on `grid.LOG10_F45_EDGES` (`_quantile_nodes_from_weight`),
    weight `1/N_YSO_SHIFT_NODES` -- every kernel cell with nonzero mass
    makes the member set too large for a per-sightline build. The product
    set (`np.repeat`/`np.tile`, no Python loop over nodes): `flux0_ts[i,:] =
    flux0_template * 10**delta`, `w_ts = w_template * w_shift`.

    `weight_t` is this REGION's own weight (`_yso_region_weight`, owner's
    ruling 2026-10-20): `reg["weight"]` (the cached, survey-wide census
    weight) times `DISK_FRACTION` plus this region's own diskless Class
    III share added on top, exactly what `template_weights.build_yso`
    does for the P5 table -- so the templates these quantile nodes draw
    from, and therefore `flux0_ts`'s own brightness distribution, carry
    the SAME two-component mixture `GRID_YSO` does. Bit-identical to the
    census-only draw at the identity case (`DISK_FRACTION = 1`)."""
    f_ref, floor_linear = reg["f_ref"], reg["floor_linear"]
    weight_t = _yso_region_weight(config, region, reg)
    csum = np.cumsum(weight_t) / weight_t.sum()
    q = (np.arange(N_YSO_NODES) + 0.5) / N_YSO_NODES
    idx_t = np.clip(np.searchsorted(csum, q), 0, weight_t.size - 1)
    flux0_t = np.empty((N_YSO_NODES, N_BANDS), dtype=np.float64)
    for k, key in enumerate(BAND_KEYS):
        flux0_t[:, k] = np.maximum(f_ref[key][idx_t], floor_linear[idx_t])
    w_t = np.full(N_YSO_NODES, 1.0 / N_YSO_NODES)

    kernel, _mo_k = sample_cloud.shift_kernel(config, region, d_front, d_back)
    mask_s = kernel > 0
    all_centers = grid.LOG10_F45_EDGES[:-1] + 0.5 * grid.D_LOG10_F45
    shift_centers = _quantile_nodes_from_weight(all_centers[mask_s], kernel[mask_s], N_YSO_SHIFT_NODES)
    scale_s = 10.0 ** shift_centers
    n_s = N_YSO_SHIFT_NODES
    w_s = np.full(n_s, 1.0 / n_s)

    flux0_ts = np.repeat(flux0_t, n_s, axis=0) * np.tile(scale_s, N_YSO_NODES)[:, None]
    w_ts = np.repeat(w_t, n_s) * np.tile(w_s, N_YSO_NODES)
    return flux0_ts, w_ts


def _h2s_sigma_nodes(logsig_mean, logsig_std):
    """`(centers, weight)`, `N_SIGMA_NODES` equally spaced quantiles
    `(i+0.5)/N_SIGMA_NODES` of the region's own lognormal itself (sec.
    5.6 "Marks"): `scipy.special.ndtri`'s exact inverse standard-normal
    CDF at each quantile, scaled by `logsig_std` and shifted by
    `logsig_mean` -- each node weight `1/N_SIGMA_NODES`."""
    q = (np.arange(N_SIGMA_NODES) + 0.5) / N_SIGMA_NODES
    centers = logsig_mean + logsig_std * ndtri(q)
    return centers, np.full(N_SIGMA_NODES, 1.0 / N_SIGMA_NODES)


def _h2s_region_nodes(logsig_mean, logsig_std, giannini_ratios):
    """`(flux0_br, w_br)`, region-wide, built once (sec. 5.6 "Marks"):
    the product of the brightness lognormal's own cells
    (`_h2s_sigma_nodes`) and the four IRAC bands' own colour-ratio
    quadrature. Each band's ratio is drawn from its own Giannini table
    independently today, so the joint is a product measure: the sorted
    table sampled at `N_RATIO_NODES` equally spaced quantiles
    `(i+0.5)/N_RATIO_NODES`, weight `1/N_RATIO_NODES` each, the four
    bands' nodes crossed (`N_RATIO_NODES**4` joint nodes, weight the
    product). `flux0_br` from `knot_ks_log10_flux` and the ratios exactly
    as before (J, H, M1 zero)."""
    sigma_centers, w_sigma = _h2s_sigma_nodes(logsig_mean, logsig_std)
    n_sig = sigma_centers.size

    band_nodes = {}
    for band in h2s_module.IRAC_RATIO_BAND_KEYS:
        table = np.sort(np.asarray(giannini_ratios[band], dtype=np.float64))
        n_tab = table.size
        qidx = np.clip(np.floor((np.arange(N_RATIO_NODES) + 0.5) / N_RATIO_NODES * n_tab).astype(np.int64),
                        0, n_tab - 1)
        band_nodes[band] = table[qidx]

    mesh = np.meshgrid(*[band_nodes[b] for b in h2s_module.IRAC_RATIO_BAND_KEYS], indexing="ij")
    ratio_joint = {b: mesh[i].ravel() for i, b in enumerate(h2s_module.IRAC_RATIO_BAND_KEYS)}
    n_ratio = ratio_joint[h2s_module.IRAC_RATIO_BAND_KEYS[0]].size
    w_ratio = np.full(n_ratio, 1.0 / n_ratio)

    sigma_full = np.repeat(sigma_centers, n_ratio)
    w_br = np.repeat(w_sigma, n_ratio) * np.tile(w_ratio, n_sig)
    log10_f_ks = h2s_module.knot_ks_log10_flux(sigma_full)
    flux0_br = np.zeros((sigma_full.size, N_BANDS), dtype=np.float64)
    flux0_br[:, BAND_KEYS.index("Ks")] = 10.0 ** log10_f_ks
    for band in h2s_module.IRAC_RATIO_BAND_KEYS:
        ratio_full_b = np.tile(ratio_joint[band], n_sig)
        flux0_br[:, BAND_KEYS.index(band)] = 10.0 ** (log10_f_ks + ratio_full_b)
    return flux0_br, w_br


def _depth_nodes(loaded_profile, sl_row, d_front, d_back):
    """`(xi_centers, weight)`: `N_DEPTH_NODES` equally spaced quantiles of
    the sightline's own `p(x)` (`sample_cloud.sample_x`'s binned return
    on `grid.LOG10_XI_EDGES`), each node at its own cell centre
    (`_quantile_nodes_from_weight`), weight `1/N_DEPTH_NODES` -- every
    `p(x)` cell with nonzero mass makes the member set too large for a
    per-sightline build; shared by YSO and H2S (sec. 5.5/5.6, the same
    `p(x)`, no second depth quadrature)."""
    p_x, _mo_x, _removed_frac = sample_cloud.sample_x(loaded_profile, sl_row, d_front, d_back)
    mask = p_x > 0
    width = grid.LOG10_XI_EDGES[1] - grid.LOG10_XI_EDGES[0]
    centers = grid.LOG10_XI_EDGES[:-1] + 0.5 * width
    log10_xi_nodes = _quantile_nodes_from_weight(centers[mask], p_x[mask], N_DEPTH_NODES)
    xi_centers = 10.0 ** log10_xi_nodes
    return xi_centers, np.full(N_DEPTH_NODES, 1.0 / N_DEPTH_NODES)


def _combine_with_depth(flux0_pre, w_pre, xi_centers, w_depth):
    """The product set of a region-wide (template x shift, or brightness
    x ratio) node table with one sightline's own depth nodes:
    `(u, flux0, w)`, built with `np.tile`/`np.repeat`, no Python loop over
    nodes."""
    n_pre = flux0_pre.shape[0]
    n_d = xi_centers.size
    flux0_full = np.tile(flux0_pre, (n_d, 1))
    w_full = np.tile(w_pre, n_d) * np.repeat(w_depth, n_pre)
    u_full = np.repeat(xi_centers, n_pre)
    return u_full, flux0_full, w_full


def _safe_cell_bin(u, log10_f45, c_m):
    """`grid.bin(u, log10_f45, c_m) * c_m.sum()`, guarded against a
    sightline with zero total catalogued weight (`c_m.sum() == 0`, e.g.
    a sightline with essentially no catalogued members at all): `grid.
    bin`'s own `H = H / total_weight` step is `0/0` there, which would
    poison the region-wide accumulator (`n_cat_cell["YSO"] += ...`) with
    NaN for every sightline summed after it -- returns an all-zero
    (128, 110) cell grid instead, the same convention `_build_one_tile`'s
    own empty-population branch already uses."""
    total = float(c_m.sum())
    if total <= 0.0:
        return np.zeros((grid.LOG10_XI_EDGES.size - 1, grid.LOG10_F45_EDGES.size - 1), dtype=np.float64)
    H, _out = grid.bin(u, log10_f45, c_m)
    return H * total


def _safe_cell_bin_counts(raw_hist, scale):
    """`grid.bin_counts(raw_hist, raw_hist.sum()) * raw_hist.sum() * scale`
    (C29, briefs/round4_2026-10-20/C29_atlas_check_dimmed_axis.md):
    `_safe_cell_bin`'s own pattern, for a caller that has already
    accumulated its own (n_x, n_b) weighted histogram (`catalogued_
    fraction`'s own `dimmed_hist`, one raw weighted count per (pixel,
    member) pair -- no single per-member point, since the dimmed flux a
    pair lands at varies pixel to pixel) instead of one point per
    member. `scale` is the ONE per-class, per-member-independent factor
    (`density / w.sum()`, or `1 / w.sum()` where density is already
    folded into `weight_pix` before `catalogued_fraction` ever saw it)
    `_safe_cell_bin`'s own caller already multiplies into its `c_m`
    before binning -- deferred here since it is a single scalar for the
    whole array (linear), never per-member, so applying it after
    smoothing changes nothing `grid.bin_counts` itself would see.
    Guarded against `raw_hist.sum() == 0` the same way `_safe_cell_bin`
    guards `c_m.sum() == 0` (an empty tile/sightline/region slice)."""
    total = float(raw_hist.sum())
    if total <= 0.0:
        return np.zeros_like(raw_hist)
    H, _out = grid.bin_counts(raw_hist, total)
    return H * total * scale


def _build_one_sightline(config, region, sl_row, a_col_in_sl, a_col_gas_in_sl, arm_in_sl, f_lim_in_sl,
                          loaded_profile, flux0_ts, w_ts, cloud_frac_sl, d_front, d_back,
                          flux0_br, w_br, pixel_row_in_sl,
                          weight_pix, eta_r, l_of_pix_in_sl):
    """One sightline's YSO and H2S quadratures over the population, no
    draw, shared by every admitted pixel it parents: `(frac_yso,
    density_yso, frac_yso_bright3, frac_yso_bright10, n_cat_cell_yso,
    n_cell_yso, frac_h2s, frac_h2s_bright3, frac_h2s_bright10,
    n_cat_cell_h2s, n_cell_h2s, n_members_yso, n_members_h2s)` (sec. 9's
    bright-end check, same members).

    YSO (sec. 5.5): `flux0_ts`/`w_ts` are the region's own template x
    shift product (`_yso_region_nodes`, computed once by the caller),
    IDENTICAL at every sightline; only the depth factor (`_depth_nodes`,
    this sightline's own `p(x)`) is formed here, crossed with the shared
    table (`_combine_with_depth`). Density `population.yso.law_count` on
    the CLOUD'S own share of the GAS column, `a_col_gas_in_sl *
    cloud_frac_sl` (`bmstp.density._cloud_column_fraction`, W26;
    `a_col_in_sl`, the extinction column, is kept for the members' own
    dimming below, never for the law).

    H2S (sec. 5.6): `flux0_br`/`w_br` are the region's own brightness x
    colour-ratio product (`_h2s_region_nodes`, computed once by the
    caller), crossed with YSO's own depth nodes (the same `p(x)`, no
    second depth quadrature). H2S's own density is `l_of_pix_in_sl *
    eta_r * eps_ext` (sec. 5.6 "Sky density"), folded directly into the
    `weight_pix` `catalogued_fraction` receives, so its own `s_member`
    already carries the density factor.

    `pixel_row_in_sl` (n_pix_in_sl,) is this sightline's own slice of
    `bmstp.atlas._depth_grid`'s row index into `PIXEL_CURVE_<band>`
    (`catalogued_probability`'s own docstring): the measured curve feeds
    `catalogued_fraction` directly, never a region-wide width. `weight_pix`
    (n_pix_in_sl,) is this sightline's own pixels' coverage times pixel area.

    Cell grids (W67c's own construction): `c_m = (w_m/w.sum()) *
    s_member[m]`, `s_member` `catalogued_fraction`'s own pixel-area-
    weighted catalogued probability per member -- for YSO the `weight_pix`
    argument passed to `catalogued_fraction` already carries
    `density_yso` (per pixel, since the young-star law varies pixel to
    pixel even within one sightline), and for H2S it already carries
    `l_of_pix_in_sl * eta_r * eps_ext`, so `s_member` in both cases is
    already density-weighted and no second multiply is needed;
    `grid.bin(u, log10 F_4.5, c_m) * c_m.sum()` with `log10 F_4.5` the
    member's own undimmed `log10 flux0[:, IDX_I2]`. `n_cell_yso`/
    `n_cell_h2s` are the SAME members quadrature with every member's own
    probability of being catalogued set to one: `s_member`'s pixel sum
    (`sum_pix weight_pix[pix] * p_cat[pix, m]`) is the same for every
    member once `p_cat` is one, so it collapses to the sightline's own
    density-weighted pixel-area total (`(density_yso * weight_pix).sum()`
    for YSO, `weight_h2s.sum()` for H2S) -- `c_m_all = (w / w.sum()) *
    that total`, binned the same way. No `rng`, no `seed`:
    every factor here is a deterministic quadrature, so this call induces
    no correlation between sightlines and no error term of its own."""
    band_params = _completeness_band_params(config, region)
    a_cloud_in_sl = a_col_gas_in_sl * cloud_frac_sl
    density_yso = yso_module.law_count(config, region, a_cloud_in_sl, arm_in_sl)

    xi_centers, w_depth = _depth_nodes(loaded_profile, sl_row, d_front, d_back)

    u_yso, flux0_yso_full, w_yso_full = _combine_with_depth(flux0_ts, w_ts, xi_centers, w_depth)
    frac_yso, frac_yso_bright3, frac_yso_bright10, s_member_yso, dimmed_hist_yso = catalogued_fraction(
        a_col_in_sl, u_yso, flux0_yso_full, w_yso_full, f_lim_in_sl, pixel_row_in_sl, band_params, config,
        weight_pix=density_yso * weight_pix)
    # C29: N_CAT_CELL_YSO bins at the dimmed I2 flux (`dimmed_hist_yso`,
    # density already folded into `weight_pix` above, so `scale` here is
    # just `1 / w.sum()`); N_CELL_YSO is unchanged (undimmed, no
    # selection).
    n_cat_cell_yso = _safe_cell_bin_counts(dimmed_hist_yso, 1.0 / w_yso_full.sum())
    weight_pix_yso_total = float(np.sum(density_yso * weight_pix))
    c_m_yso_all = (w_yso_full / w_yso_full.sum()) * weight_pix_yso_total
    n_cell_yso = _safe_cell_bin(u_yso, np.log10(flux0_yso_full[:, IDX_I2]), c_m_yso_all)

    u_h2s, flux0_h2s_full, w_h2s_full = _combine_with_depth(flux0_br, w_br, xi_centers, w_depth)
    weight_h2s = l_of_pix_in_sl * eta_r * density_module.EPS_EXT * weight_pix
    frac_h2s, frac_h2s_bright3, frac_h2s_bright10, s_member_h2s, dimmed_hist_h2s = catalogued_fraction(
        a_col_in_sl, u_h2s, flux0_h2s_full, w_h2s_full, f_lim_in_sl, pixel_row_in_sl, band_params, config,
        weight_pix=weight_h2s)
    n_cat_cell_h2s = _safe_cell_bin_counts(dimmed_hist_h2s, 1.0 / w_h2s_full.sum())
    weight_pix_h2s_total = float(weight_h2s.sum())
    c_m_h2s_all = (w_h2s_full / w_h2s_full.sum()) * weight_pix_h2s_total
    n_cell_h2s = _safe_cell_bin(u_h2s, np.log10(flux0_h2s_full[:, IDX_I2]), c_m_h2s_all)

    return (frac_yso, density_yso, frac_yso_bright3, frac_yso_bright10, n_cat_cell_yso, n_cell_yso,
            frac_h2s, frac_h2s_bright3, frac_h2s_bright10, n_cat_cell_h2s, n_cell_h2s,
            int(u_yso.size), int(u_h2s.size))


def _herschel_convolved_law(law_map, law_wcs, pix, arm, density_h2s_law_pooled_pix):
    """H2S's sky density (sec. 5.6) is the SAME pooled-coefficient
    quadratic law (`knot_field.law_count_at_kappa` at `kappa_pooled`,
    never `population.yso.law_count`'s region-fitted value -- WP-POP-5's
    finding) convolved by the knot-driver kernel on the Herschel arm,
    the region's convolved-law map (`knot_field.convolved_law`, itself
    built at `kappa_pooled`, computed ONCE per region by the caller,
    sec. 5.6 "a map operation, once per region") sampled at each pixel's
    own mean position (`mean_over_area`); a Planck pixel, or a Herschel
    pixel the map does not reach, keeps the point law
    `density_h2s_law_pooled_pix` already carries. Two callers read this
    from the SAME deterministic `density_h2s_law_pooled_pix`: `build_
    region`'s own H2S error weight, before the sightline quadrature
    runs, and its H2S mean density, after -- one function, one convolved
    map, so the two agree exactly, never a pre-convolution weight beside
    a convolved mean."""
    l_of_pix = density_h2s_law_pooled_pix.copy()
    herschel_pix = arm == yso_module.PROVENANCE_HERSCHEL
    if law_map is not None and herschel_pix.any():
        gl_pix, gb_pix = hp.pix2ang(512, pix[herschel_pix], nest=True, lonlat=True)
        width_deg = float(np.sqrt(_HPX512_PIXEL_DEG2))
        l_convolved = knot_field.mean_over_area(law_map, law_wcs, gl_pix, gb_pix,
                                                  width_deg, frame="galactic")
        finite = np.isfinite(l_convolved)
        idx = np.flatnonzero(herschel_pix)
        l_of_pix[idx[finite]] = l_convolved[finite]
    return l_of_pix


def _gal_members(config):
    """`(flux0, u, w, log10_s, density, n_members_before, n_members_after)`, GAL's
    members, DETERMINISTIC (sec. 5.4): the weighted sum over every member of the
    population, no draw -- exact up to the population's own binning and the colour
    collapse below. One member per distinct `(counts-law node, galz template)` pair
    actually realised in the SWIRE sample: at each of the counts law's tabulated
    `log10 S` nodes (`bmstp.sample_gal.sample`'s `phi(S).S` weight `w_law`, the same
    law `bmstp.shapes.build_gal` bins), the SWIRE galaxies AT THAT NODE under the
    SAME selection `template_weights.build_galz` applies to its own colour
    population -- `isfinite(COLOUR_I1I2) & isfinite(SIGMA_COLOUR_I1I2) &
    SIGMA_COLOUR_I1I2 > 0` (`sky/derived/swire/galaxies_swire_survey.hdf5`'s own
    `NODE` axis; a node with none borrows its nearest node that has one) -- each
    mapped to its nearest "galz" register template in `log10 F_REF,I1 - log10
    F_REF,I2` (the SAME register and colour axis `build_galz` weights the fitter's
    GAL templates on), vectorised by `searchsorted` over every selected galaxy at
    once, no per-galaxy loop (rule 8).

    The distinct `(node, template)` pairs collapse from a `(n_node, n_template)`
    bincount of the packed key `node * n_template + template` (rule 8: a bincount on
    a packed key, not a loop over nodes or galaxies), gathered per counts-law node
    through its own borrowed source node: `weight_grid[k, t] = (w_law[k] /
    w_law.sum()) * (galaxies at node k's source mapping to template t) / (galaxies
    at node k's source)`, so `weight_grid[k, :].sum() == w_law[k] / w_law.sum()` and
    `w.sum() == 1` over every member.

    A member enters `catalogued_probability` only through `u = 1` and its four
    IRAC fluxes, `S_k` times the template's own three colour ratios `F_I1/F_I2,
    F_I3/F_I2, F_I4/F_I2` (the same floor-protected ratio `f_ref[band] /
    max(f_ref['I2'], floor_linear)` the flux scaling below always used): two
    templates whose three log-ratios agree to within `GAL_COLOUR_CELL_DEX` (a cell
    far narrower than the completeness roll-off, `W_DEX` ~= 0.1-0.3 dex) are the
    same member (rule 9). So, per counts-law node, each template's three log-ratios
    are binned at that cell width (a property of the template alone, the same for
    every node) and the `(node, template)` pairs' weights are summed within each
    occupied `(node, cell)` -- one `np.unique` on a packed integer key for the
    template-only cell id, a second on `node * n_cell + cell` for the pair grouping
    (rule 8: no loop over nodes, templates or pairs) -- with the cell represented by
    its weight-weighted mean ratios. `n_members_before`/`n_members_after` are the
    `(node, template)` and `(node, cell)` counts, for the build's own report.

    The collapsed members: `flux0`'s `F_REF,I2` scaled so it equals the node's own
    `S_k`, `F_REF,I1/I3/I4` from the cell's mean ratio times `S_k`, `u = 1` (sec.
    5.4's "whole column"), and `log10_s = log10 S_k` per member, for the cell grid's
    brightness axis (`F_4.5 = S`, sec. 5.4). J, H, Ks, M1 are unmeasured for a
    galaxy and held at zero flux, so the two-of-eight test runs on the four IRAC
    bands only (disclosed). `density` is `sample_gal.density(config)` (sec. 5.4 "Sky
    density"), unchanged: the population's own sky density is one survey number,
    independent of how its members are enumerated."""
    _, log10_s_grid, w_law = sample_gal.sample(config)
    n_node = log10_s_grid.size

    gal_path = config_module.product_path(config, "sky/derived", "swire", "galaxies", "survey")
    with h5py.File(gal_path, "r") as f:
        node = np.asarray(f["NODE"][:], dtype=np.int64)
        c12 = np.asarray(f["COLOUR_I1I2"][:], dtype=np.float64)
        sigma_c12 = np.asarray(f["SIGMA_COLOUR_I1I2"][:], dtype=np.float64)
    # the SAME selection `template_weights.build_galz` applies to the
    # colour population it weights the fitter's GAL templates on.
    finite = (node >= 0) & np.isfinite(c12) & np.isfinite(sigma_c12) & (sigma_c12 > 0)
    node, c12 = node[finite], c12[finite]

    counts = np.bincount(node, minlength=n_node)
    node_ids = np.arange(n_node)
    has = counts > 0
    nearest = node_ids.copy()
    if not has.all():
        have_idx = node_ids[has]
        nearest[~has] = have_idx[np.argmin(np.abs(node_ids[~has, None] - have_idx[None, :]), axis=1)]
    # `counts[nearest]` must be positive everywhere by the borrowing above
    # (`nearest` only ever points at a node with `has[node] == True`); if it
    # is ever zero regardless, fail loud naming the empty node (rule 5b, rule 6's
    # "no silent fallbacks") rather than dividing by a zero count below.
    if np.any(counts[nearest] == 0):
        bad = int(nearest[counts[nearest] == 0][0])
        raise ValueError("bmstp.atlas: colour node %d has no galaxies to draw from" % bad)

    # the "galz" register template nearest EACH SELECTED GALAXY'S OWN colour
    # (`template_weights.build_galz`'s SAME `colour_theta`), vectorised by
    # `searchsorted` over the whole selection at once, no per-galaxy loop.
    reg = template_weights._read_register(config, "galz")
    f_ref, floor_linear = reg["f_ref"], reg["floor_linear"]
    colour_theta = np.log10(f_ref["I1"]) - np.log10(f_ref["I2"])
    n_tmpl = colour_theta.size
    reg_order = np.argsort(colour_theta)
    sorted_colour = colour_theta[reg_order]
    j = np.clip(np.searchsorted(sorted_colour, c12), 1, sorted_colour.size - 1)
    lo, hi = j - 1, j
    pick_hi = np.abs(sorted_colour[hi] - c12) < np.abs(c12 - sorted_colour[lo])
    template = reg_order[np.where(pick_hi, hi, lo)]

    # the (raw node, template) counts, a bincount on a packed key -- no
    # per-galaxy or per-node loop (rule 8).
    packed = node * n_tmpl + template
    counts_pair = np.bincount(packed, minlength=n_node * n_tmpl).reshape(n_node, n_tmpl)

    # each counts-law node `k` borrows `nearest[k]`'s own galaxy/template
    # counts (a fancy-index gather, not a loop); the distinct `(k, template)`
    # members are that gathered grid's non-zero cells (`np.nonzero`, again no
    # per-node loop).
    gathered = counts_pair[nearest]  # (n_node, n_tmpl)
    denom = counts[nearest].astype(np.float64)  # (n_node,)
    weight_grid = (w_law / w_law.sum())[:, None] * gathered / denom[:, None]
    k_idx, t_idx = np.nonzero(weight_grid)
    w_pair = weight_grid[k_idx, t_idx]
    n_members_before = int(k_idx.size)

    # the template's own three colour log-ratios (node-independent), the
    # same floor-protected denominator the flux scaling below always used
    # (`np.maximum(f_ref['I2'], floor_linear)`, rather than a raw divide
    # that a zero-flux template could blow up).
    denom_i2 = np.maximum(f_ref["I2"], floor_linear)
    log10_denom_i2 = np.log10(denom_i2)
    ratio_i1_tmpl = np.log10(f_ref["I1"]) - log10_denom_i2
    ratio_i3_tmpl = np.log10(f_ref["I3"]) - log10_denom_i2
    ratio_i4_tmpl = np.log10(f_ref["I4"]) - log10_denom_i2

    # bin each template's three log-ratios at `GAL_COLOUR_CELL_DEX` (rule
    # 9: the cell size and why it is below the roll-off width, module
    # docstring) into one packed integer key -- a template property alone,
    # the same cell id at every node -- then compact it with `np.unique`
    # (rule 8, no loop over templates).
    cell1 = np.floor(ratio_i1_tmpl / GAL_COLOUR_CELL_DEX).astype(np.int64)
    cell3 = np.floor(ratio_i3_tmpl / GAL_COLOUR_CELL_DEX).astype(np.int64)
    cell4 = np.floor(ratio_i4_tmpl / GAL_COLOUR_CELL_DEX).astype(np.int64)
    cell_offset = 1 << 19  # far past any realistic colour-ratio cell index
    template_key = (((cell1 + cell_offset).astype(np.int64) << 40)
                    | ((cell3 + cell_offset).astype(np.int64) << 20)
                    | (cell4 + cell_offset).astype(np.int64))
    _, template_cell = np.unique(template_key, return_inverse=True)
    n_cell = int(template_cell.max()) + 1 if template_cell.size else 0

    # sum the pair weights within each occupied (node, cell) -- one more
    # `np.unique` on a packed key, no loop over nodes or pairs (rule 8) --
    # and take the weight-weighted mean of the three ratios per cell.
    pair_key = k_idx * n_cell + template_cell[t_idx]
    group_key, group_id = np.unique(pair_key, return_inverse=True)
    n_group = group_key.size
    w = np.bincount(group_id, weights=w_pair, minlength=n_group)
    mean_r1 = np.bincount(group_id, weights=w_pair * ratio_i1_tmpl[t_idx], minlength=n_group) / w
    mean_r3 = np.bincount(group_id, weights=w_pair * ratio_i3_tmpl[t_idx], minlength=n_group) / w
    mean_r4 = np.bincount(group_id, weights=w_pair * ratio_i4_tmpl[t_idx], minlength=n_group) / w
    k_group = group_key // n_cell
    n_members_after = int(n_group)

    s_member = 10.0 ** log10_s_grid[k_group]
    flux0 = np.zeros((n_group, N_BANDS), dtype=np.float64)
    i1, i2, i3, i4 = (BAND_KEYS.index(k) for k in ("I1", "I2", "I3", "I4"))
    flux0[:, i1] = s_member * 10.0 ** mean_r1
    flux0[:, i2] = s_member
    flux0[:, i3] = s_member * 10.0 ** mean_r3
    flux0[:, i4] = s_member * 10.0 ** mean_r4
    u = np.ones(n_group, dtype=np.float64)
    log10_s = log10_s_grid[k_group]
    # `A_GAL`, sec. 5.4 "Sky density": the density the quadrature stands
    # for is `sample_gal.density` (the `ln 10` integral), NOT the shape
    # weight `w_law.sum()` the node weights above use.
    return (flux0, u, w, log10_s, sample_gal.density(config),
            n_members_before, n_members_after)


def _gal_pixel_chunks(config, n_pix):
    """The admitted-pixel axis split into `config.n_jobs` contiguous chunks
    (`np.array_split`): GAL's own pieces in `build_region`'s one flat piece
    list (owner's ruling: one pool, one flat list across tiles, GAL,
    sightlines and the tail, parent reduces in a fixed order). Kept at
    exactly `config.n_jobs` chunks: `catalogued_fraction`'s own `s_member`
    sums across chunks exactly wherever the chunk boundaries are these
    same ones (F4 risk 2), so a different chunk count would move the
    region's GAL numbers at their last bit."""
    n_jobs = max(1, int(config.n_jobs))
    return [c for c in np.array_split(np.arange(n_pix), n_jobs) if c.size]


def _gal_chunk_result(config, region, gal_members, idx, a_col, f_lim, pixel_row, weight_pix_all):
    """One GAL pixel chunk's own `catalogued_fraction` call -- the piece
    `build_region`'s flat list submits to the region's one pool. Reached
    only inside a worker, never the parent: `catalogued_probability`
    reaches `catalog.completeness`'s one `@njit(parallel=True)` kernel, and
    numba's thread pool is not fork-safe (a process that has itself
    started that pool poisons every child a later `os.fork()` creates
    from it, and joblib respawns idle-expired workers by forking the
    parent between phases); `pieces.run` never runs a piece in the
    parent process except its own `n_jobs <= 1` fallback, so this call
    never reaches the kernel outside a worker under the region's normal
    `n_jobs >= 2`."""
    flux0, u, w, _log10_s, _density, _n_before, _n_after = gal_members
    band_params = _completeness_band_params(config, region)
    return catalogued_fraction(
        a_col[idx], u, flux0, w, f_lim[idx], pixel_row[idx], band_params, config,
        weight_pix=weight_pix_all[idx])


def _gal_reduce(gal_members, chunk_results, weight_pix_all):
    """`(frac, frac_bright3, frac_bright10, n_cat_cell, n_cell, density,
    n_members_before, n_members_after)` for GAL's one region-wide
    DETERMINISTIC quadrature (sec. 8), reduced from `_gal_chunk_result`'s
    own per-chunk pieces IN CHUNK ORDER (`build_region`'s flat piece list
    keeps `pieces.run`'s own results in the order its pieces were
    submitted, so `chunk_results` here is in the same order
    `_gal_pixel_chunks` built them -- this reduction is exactly what the
    former `_gal_catalogued_fraction` did with its own `Parallel` call's
    results). `frac`/`frac_bright3`/`frac_bright10` concatenate in pixel
    order (contiguous chunks keep the axis ordered); `s_member` values are
    SUMMED (`catalogued_fraction`'s own docstring: `s_member` is a sum
    over pixels, so splitting the pixel axis and adding the pieces back is
    exact). The cell grid below is formed once from that summed
    `s_member`, exactly as when one call held every pixel.

    `n_cat_cell` (128, 110): the region's expected number of catalogued
    GAL objects per parameter cell (module docstring), the same one-cell
    smoothing and wall fold every stored shape carries (`grid.bin`), so
    the region grid and the shapes share one convention. Each member's own
    expected catalogued count in the region is `c_m = density * (w_m /
    w.sum()) * s_member[m]` (the population's own weight share times its
    pixel-area-weighted catalogued probability over the region,
    `catalogued_fraction`'s `s_member`); `_safe_cell_bin(u, log10_s, c_m)`
    guards the `c_m.sum() == 0` case the same way as every other class's
    own cell grid (`_safe_cell_bin`'s own docstring). The mass `grid.bin`'s
    own edges drop is not stored (expected ~= 0 for GAL, sec. 5.4's
    population sitting well inside the grid). `n_cell` (128, 110) is the
    SAME quadrature with every member's own probability of being
    catalogued set to one: `s_member`'s pixel sum collapses to the
    region's own coverage-weighted pixel-area total (`weight_pix_all.
    sum()`, the same for every member), `c_m_all = density * (w / w.sum())
    * weight_pix_all.sum()`, `_safe_cell_bin(u, log10_s, c_m_all)`.

    `gal_members` is `_gal_members`'s own survey-wide return, computed
    once by `build` and passed in here rather than re-derived per region.
    `n_members_before`/`n_members_after` (`_gal_members`'s own return,
    passed through unchanged) are the region's `(node, template)` and
    `(node, colour cell)` member counts, for the build's own report."""
    flux0, u, w, log10_s, density, n_members_before, n_members_after = gal_members
    frac = np.concatenate([r[0] for r in chunk_results])
    frac_bright3 = np.concatenate([r[1] for r in chunk_results])
    frac_bright10 = np.concatenate([r[2] for r in chunk_results])
    s_member = sum(r[3] for r in chunk_results)
    dimmed_hist = sum(r[4] for r in chunk_results)

    w_sum = float(w.sum())
    # C29: N_CAT_CELL_GAL bins at the dimmed I2 flux (`dimmed_hist`,
    # summed across chunks exactly as `s_member` is); N_CELL_GAL is
    # unchanged (undimmed, no selection).
    n_cat_cell = _safe_cell_bin_counts(dimmed_hist, density / w_sum)
    c_m_all = density * (w / w_sum) * float(weight_pix_all.sum())
    n_cell = _safe_cell_bin(u, log10_s, c_m_all)
    return (frac, frac_bright3, frac_bright10, n_cat_cell, n_cell, density,
            n_members_before, n_members_after)


# ---------------------------------------------------------------------------
# PAHN: region-wide members (like GAL's own, not tile/sightline-grouped --
# C14_pahn_population.md, owner's 2026-10-20 ruling "H2S is the pattern")
# ---------------------------------------------------------------------------

def _pahn_depth_nodes():
    """`(xi_centers, weight)`: `N_DEPTH_NODES` equally spaced quantiles of
    PAHN's own `p(x)` (`sample_cloud.pahn_front_marginal`, the SAME
    front-surface marginal `bmstp.shapes.build_cloud` places `GRID_PAHN`'s
    own x-axis with) -- unlike YSO/H2S's own `_depth_nodes`, this is NOT a
    function of the sightline: PAHN's own placement does not depend on
    one, so this is computed once per region, not once per sightline."""
    p_x, _mo_x = sample_cloud.pahn_front_marginal()
    mask = p_x > 0
    width = grid.LOG10_XI_EDGES[1] - grid.LOG10_XI_EDGES[0]
    centers = grid.LOG10_XI_EDGES[:-1] + 0.5 * width
    log10_xi_nodes = _quantile_nodes_from_weight(centers[mask], p_x[mask], N_DEPTH_NODES)
    xi_centers = 10.0 ** log10_xi_nodes
    return xi_centers, np.full(N_DEPTH_NODES, 1.0 / N_DEPTH_NODES)


def _pahn_region_nodes(config, region, reader_pahn, d_r_pc):
    """`(flux0_pahn, u_pahn, w_pahn)`: PAHN's own region-wide deterministic
    quadrature (the brief: "the member draw from the PAHN shape and
    weights") -- the brightness PLACEMENT is `bmstp.template_weights.
    build_pahn`'s own `n_model` nodes (`reader_pahn.c_theta`, `bmstp.
    template_weights`'s product already read through `prior_reader.load`,
    uniform weight `1/n_model` each -- the SAME nodes WEIGHTS wrote, never
    recomputed here), PLACED AT THIS REGION's own distance (`delta = -2
    log10(d_r_pc / 1 kpc)`, the SAME shift every other class's `C_THETA`
    is placed by): `log10 F_I2 = C_THETA + delta` undimmed I2 flux at the
    region. The SHAPE (I1, I3, I4, M1, J, H, Ks, all relative to I2) is
    the register's own 33 templates' 8-band reference fluxes (C17
    2026-10-20, the same register `template_weights._read_register`
    reads for every other class), row i's own template matched to
    `C_THETA`'s i-th placement node by row position (both are the
    register's own row order: `build_pahn` reads `n_model` and
    `MODEL_NAME` straight off it) -- an arbitrary but uniform pairing,
    harmless to the member set's own marginal distribution, since every
    node already carries the same weight `1/n_model` regardless of which
    register row supplies its shape. This recovers I1/I3/M1 as the
    template's own PAH fluxes (the 3.3 um feature, the 6.2/7.7 complex,
    the 24 um aperture growth) instead of the retired all-zero stand-in.
    Crossed with PAHN's own depth nodes (`_pahn_depth_nodes`, region-wide,
    not per-sightline) via `_combine_with_depth`, the SAME generic
    combiner YSO/H2S use."""
    reg = template_weights._read_register(config, "pahn")
    n_model = reader_pahn.c_theta.size
    if reg["names"].size != n_model:
        raise ValueError(
            "bmstp.atlas: PAHN register has %d templates, WEIGHTS' own C_THETA "
            "has %d -- template_weights.build_pahn must place exactly the "
            "register's own rows" % (reg["names"].size, n_model))
    w_pre = np.full(n_model, 1.0 / n_model, dtype=np.float64)
    delta = -2.0 * np.log10(d_r_pc / 1000.0)
    flux0_i2 = 10.0 ** (reader_pahn.c_theta + delta)
    f_i2_reg = np.maximum(reg["f_ref"]["I2"], reg["floor_linear"])
    flux0_pre = np.empty((n_model, N_BANDS), dtype=np.float64)
    for j, bkey in enumerate(BAND_KEYS):
        band_ratio = np.maximum(reg["f_ref"][bkey], reg["floor_linear"]) / f_i2_reg
        flux0_pre[:, j] = flux0_i2 * band_ratio

    xi_centers, w_depth = _pahn_depth_nodes()
    return _combine_with_depth(flux0_pre, w_pre, xi_centers, w_depth)


def _pahn_chunk_result(config, region, pahn_members, density_pahn_pix, idx, a_col, f_lim, pixel_row,
                        weight_pix_all):
    """One PAHN pixel chunk's own `catalogued_fraction` call -- the GAL
    pattern (`_gal_chunk_result`), but with `weight_pix` already carrying
    this chunk's own INTRINSIC density (`density_pahn_pix[idx]`), the
    SAME construction `_build_one_sightline` uses for H2S's own
    density-weighted `s_member` (PAHN's own density varies pixel to
    pixel, unlike GAL's one survey-wide scalar, so it cannot be folded
    in only at the reduction step the way `_gal_reduce` does)."""
    u_pahn, flux0_pahn, w_pahn = pahn_members
    band_params = _completeness_band_params(config, region)
    weight_pix = density_pahn_pix[idx] * weight_pix_all[idx]
    return catalogued_fraction(
        a_col[idx], u_pahn, flux0_pahn, w_pahn, f_lim[idx], pixel_row[idx], band_params, config,
        weight_pix=weight_pix)


def _pahn_reduce(pahn_members, chunk_results, density_pahn_pix, weight_pix_all):
    """`(frac, frac_bright3, frac_bright10, n_cat_cell, n_cell)` for
    PAHN's own region-wide deterministic quadrature, reduced from
    `_pahn_chunk_result`'s own per-chunk pieces in chunk order (the SAME
    `_gal_reduce` pattern) -- but `s_member` already carries PAHN's own
    per-pixel density (`_pahn_chunk_result`'s own `weight_pix`), so `c_m`
    needs no further density multiply here, unlike `_gal_reduce`'s own
    `density * (w/w_sum) * s_member` (GAL's one scalar density is never
    folded into `weight_pix`): `c_m = (w/w_sum) * s_member` directly, the
    SAME form `_build_one_sightline` uses for H2S's own `c_m_h2s`.
    `n_cell` is the SAME quadrature with every member's own probability
    of being catalogued set to one: `s_member`'s pixel sum collapses to
    the region's own density-weighted pixel-area total, `(density_pahn_
    pix * weight_pix_all).sum()`, the SAME construction `_build_one_
    sightline`'s own `weight_pix_yso_total`/`weight_pix_h2s_total` use."""
    u_pahn, flux0_pahn, w_pahn = pahn_members
    frac = np.concatenate([r[0] for r in chunk_results])
    frac_bright3 = np.concatenate([r[1] for r in chunk_results])
    frac_bright10 = np.concatenate([r[2] for r in chunk_results])
    s_member = sum(r[3] for r in chunk_results)
    dimmed_hist = sum(r[4] for r in chunk_results)

    w_sum = float(w_pahn.sum())
    log10_f_i2 = np.log10(flux0_pahn[:, IDX_I2])
    # C29: N_CAT_CELL_PAHN bins at the dimmed I2 flux (`dimmed_hist`,
    # already density-weighted exactly as `s_member` is, `_pahn_chunk_
    # result`'s own `weight_pix`, so `scale` is just `1/w.sum()`
    # unlike `_gal_reduce`'s own `density/w.sum()`); N_CELL_PAHN is
    # unchanged (undimmed, no selection).
    n_cat_cell = _safe_cell_bin_counts(dimmed_hist, 1.0 / w_sum)
    total_density_area = float(np.sum(density_pahn_pix * weight_pix_all))
    c_m_all = (w_pahn / w_sum) * total_density_area
    n_cell = _safe_cell_bin(u_pahn, log10_f_i2, c_m_all)
    return frac, frac_bright3, frac_bright10, n_cat_cell, n_cell


def _observed_bright_counts(config, region, pix, f_lim):
    """`(n_obs, n_obs_bright3, n_obs_bright10)`, int32 (n_pix,): the
    catalogue's own counterpart to the bright-end check (SPEC_BMSTP_DRAFT.md
    sec. 9 "the bright-end count ratio"). Every SESNA source's own
    nside-512 pixel (`catalog.curated`'s `GAL_L_DEG`/`GAL_B_DEG`, nested,
    `healpy.ang2pix`) is binned against the admitted pixel axis `pix`;
    `n_obs` counts every source there, `n_obs_bright3`/`n_obs_bright10`
    only those with a measured (`ORIGIN_FNU[:, I2] == 1`) I2 flux above
    3x/10x that pixel's own `F_LIM_50_PIX_MJY[:, I2]` -- the same
    threshold `catalogued_fraction` applies to the model's own members."""
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        gl = np.asarray(f["GAL_L_DEG"][:], dtype=np.float64)
        gb = np.asarray(f["GAL_B_DEG"][:], dtype=np.float64)
        flux_i2 = np.asarray(f["FNU_MJY"][:, IDX_I2], dtype=np.float64)
        measured_i2 = np.asarray(f["ORIGIN_FNU"][:, IDX_I2]) == 1
    src_pix = hp.ang2pix(512, gl, gb, nest=True, lonlat=True)

    order = np.argsort(pix)
    pix_sorted = pix[order]
    loc = np.minimum(np.searchsorted(pix_sorted, src_pix), pix_sorted.size - 1)
    found = pix_sorted[loc] == src_pix
    idx_pix = order[loc[found]]  # index into the caller's own `pix` order

    n_obs = np.bincount(idx_pix, minlength=pix.size).astype(np.int32)

    lim_i2_at_src = f_lim[idx_pix, IDX_I2]
    flux_i2_v = flux_i2[found]
    measured_v = measured_i2[found]
    bright3 = measured_v & (flux_i2_v > BRIGHT_MULT_3 * lim_i2_at_src)
    bright10 = measured_v & (flux_i2_v > BRIGHT_MULT_10 * lim_i2_at_src)
    n_obs_bright3 = np.bincount(idx_pix[bright3], minlength=pix.size).astype(np.int32)
    n_obs_bright10 = np.bincount(idx_pix[bright10], minlength=pix.size).astype(np.int32)
    return n_obs, n_obs_bright3, n_obs_bright10


#: rule 10b's 512 MB batch budget for `_above_fraction`'s own
#: (n_pixel_batch, n_x, n_b) working set: the gathered grain shape, the
#: extinction column outer product, the blended law's I2 entry, the
#: brightness threshold and the boolean "above the limit" array are each
#: that shape (float64), so this counts that many same-shape buffers,
#: generously.
_ABOVE_BATCH_BUDGET_BYTES = 512 * 1024 * 1024
_N_ABOVE_TEMP_ARRAYS = 6

#: STAR/PAHC's own template-weight factor (`_factor_marginal`) has an
#: (n_pixel_batch, n_model, n_b) working set instead -- the register's
#: model count can run to several hundred, far larger than the shape
#: grid's own n_x (128), so it is its own bound (`arg`, `p_val`, `term`
#: and the `pi_theta_f * term` product).
_N_FACTOR_TEMP_ARRAYS = 4


def _above_batch_size(n_x, n_b):
    """Pixels per batch so `n_pixel_batch * n_x * n_b * 8 bytes *
    _N_ABOVE_TEMP_ARRAYS` stays under `_ABOVE_BATCH_BUDGET_BYTES`
    (CODING_RULES_BMSTP.md rule 10b)."""
    row_bytes = n_x * n_b * 8 * _N_ABOVE_TEMP_ARRAYS
    return max(1, _ABOVE_BATCH_BUDGET_BYTES // row_bytes)


def _above_factor_batch_size(n_model, n_b):
    """Pixels per batch for STAR/PAHC's own `_factor_marginal` working
    set, the same 512 MB budget (rule 10b) applied to its `(n_pixel_batch,
    n_model, n_b)` shape rather than the shape grid's `(n_x, n_b)`."""
    row_bytes = n_model * n_b * 8 * _N_FACTOR_TEMP_ARRAYS
    return max(1, _ABOVE_BATCH_BUDGET_BYTES // row_bytes)


def _above_shared(config, reader, a_col, f0):
    """`(above, b_centers)`, shared by every class's own `_above_fraction`
    call below (F4 2026-10-09 "change C", "cheapest and safest"): `above`
    (n_pix, n_x, n_b) bool, `F_obs(cell) = F_j * 10^(-0.4 kappa_4.5(a) a) >
    F_0`, `a = x_i * a_col[pixel]` (the pixel's own extinction column,
    `_pixel_column`), `kappa_4.5` the I2 entry of the blended law at that
    extinction (`population.selection.kappa_hybrid`/`law_dense_weight`),
    `F_0 = f0[pixel]` the pixel's own 4.5 um 50% completeness limit --
    depends only on `a_col`/`f0`, NEVER on the class's own grain or shape
    grid, so every one of the six classes shares the SAME `above` (every
    class's own stored `xi_edges`/`b_edges` are the SAME `grid.
    LOG10_XI_EDGES`/`grid.LOG10_F45_EDGES` written to every one of the
    six classes' own files; `reader` is whichever class's own reader the
    caller already opened first). Computed once for the WHOLE region's
    pixel axis, unbatched: `(n_pix, n_x, n_b)` bool is a few tens of MB
    even on the largest region (nowhere near the rule 10b 512 MB budget,
    which exists for the per-class `(n_pixel_batch, n_model, n_b)`
    `_factor_marginal` working set below, not for this), and every step
    here is elementwise with no reduction across the pixel axis, so no
    batch boundary -- one call here, or the six per-class batched calls
    this replaces -- can move a bit."""
    xi_centers = 0.5 * (reader.xi_edges[:-1] + reader.xi_edges[1:])  # log10 ξ
    b_centers = 0.5 * (reader.b_edges[:-1] + reader.b_edges[1:])  # log10 F_4.5
    x_lin = 10.0 ** xi_centers
    f_j = 10.0 ** b_centers  # mJy
    a = x_lin[None, :] * a_col[:, None]  # (n_pix, n_x)
    w_ramp = selection_module.law_dense_weight(a)
    kappa45 = selection_module.kappa_hybrid(config, w_ramp)[..., IDX_I2]  # (n_pix, n_x)
    f_min = f0[:, None] * 10.0 ** (0.4 * kappa45 * a)  # (n_pix, n_x): F_j needed to clear F_0
    above = f_j[None, None, :] > f_min[:, :, None]  # (n_pix, n_x, n_b)
    return above, b_centers


def _above_fraction(reader, cls, grain_of_pix, above, d_pahc, curve):
    """`out`. Per admitted pixel, the deterministic sum
    `Sum_cells h_C(cell; grain) * f_C(F_j; pixel) * 1[F_obs(cell) > F_0]`
    that `N_ABOVE_C(pixel) = A_C(pixel) *` this sum multiplies
    (SPEC_BMSTP_DRAFT.md sec. 8). `h_C` is the class's own RAW stored grain
    shape (`reader.grid_all[grain]`, unit mass on the support -- no
    per-source column-kernel blur: this is a pixel-level sum, not a
    source's own read). `f_C` is 1 for GAL/YSO/AGB/H2S and, for STAR/PAHC,
    the type-times-contamination marginal at the pixel's own 8 um limit
    (`bmstp.template_weights._factor_marginal`, moved out of
    `atlas.shapes` so the atlas and the shapes page share one
    definition). `above` (n_pix, n_x, n_b) is `_above_shared`'s own
    `1[F_obs(cell) > F_0]`, this call's own slice of the region's shared
    mask -- identical whichever class computed it, so it is computed once
    by the caller and passed in here, never recomputed per class.
    Batched over the pixel axis: each batch holds one `(n_pixel_batch,
    n_x, n_b)` working set (`_above_batch_size`) and, for STAR/PAHC, one
    `(n_pixel_batch, n_model, n_b)` working set for `_factor_marginal`
    (`_above_factor_batch_size`) -- evaluated for the BATCH's own pixels
    only, never the whole admitted-pixel axis at once (a per-source-style
    call over every pixel at once ran 8.6 GB, Orion A PAHC, to 27 GB,
    Cygnus X, since the model axis can run to several hundred
    templates)."""
    b_centers = 0.5 * (reader.b_edges[:-1] + reader.b_edges[1:])  # log10 F_4.5
    n_x, n_b = above.shape[1], above.shape[2]
    n_pix = above.shape[0]
    out = np.empty(n_pix, dtype=np.float64)

    is_factor_cls = cls in ("STAR", "PAHC")
    batch = _above_batch_size(n_x, n_b)
    if is_factor_cls:
        batch = min(batch, _above_factor_batch_size(reader.c_theta.size, n_b))
    for start in range(0, n_pix, batch):
        stop = min(start + batch, n_pix)
        h_b = reader.grid_all[grain_of_pix[start:stop]].astype(np.float64)  # (n_p, n_x, n_b)
        above_b = above[start:stop]  # (n_p, n_x, n_b): the region-shared mask's own slice
        if is_factor_cls:
            # STAR/PAHC's own template-weight factor at this batch's own
            # pixels' 8 um limits only, `d_pahc` reshaped to
            # `(n_p, 1, 1)` so `_factor_marginal`'s broadcast adds the
            # pixel axis in front of its own `(n_model, n_b)` term.
            f_c_batch = template_weights._factor_marginal(
                b_centers, reader, cls, d_pahc[start:stop, None, None], curve)  # (n_p, n_b)
            term_cell = h_b * f_c_batch[:, None, :]  # no flux cut, no dimming
            term = term_cell * above_b
        else:
            term_cell = h_b
            term = term_cell * above_b
        out[start:stop] = term.sum(axis=(1, 2))
    return out


def build_region(config, region, agb_pool, gal_members, giannini_ratios, yso_reg, fitted_rates, pooled_eta):
    """Writes `bmstp/atlas/prior_atlas_hpx512__R.hdf5` for one region: the
    admitted pixel axis (`catalog.depth_grid`), its column and coverage,
    and every class's `N_CAT_*`/`SHARE_*` from its own deterministic
    quadrature above (module docstring).

    `INTENSITY_<C>` (deg^-2, one value per admitted pixel) is the class's
    own intrinsic sky density at that pixel -- the quantity the
    catalogued-density sum (`N_CAT_<C> = INTENSITY_<C> * accepted
    fraction`) starts from before the pixel's own completeness is applied
    (SPEC_BMSTP_DRAFT.md sec. 8): the young-star law (Herschel-convolved
    where it reaches) and its H2S-scaled density for YSO/H2S, the
    star-family tile's own evolved-star total per its own
    `OMEGA_POINTING_DEG2` for AGB, and the counts law's single
    survey-wide density for GAL -- each an `INTENSITY_<C> * accepted
    fraction` relation exactly. STAR and PAHC are the one exception
    (sec. 5.3: `A_PAHC(s) = A_STAR(s)`): `INTENSITY_STAR` and
    `INTENSITY_PAHC` are both the star-family tile's own UNSPLIT
    field-star total per its own `OMEGA_POINTING_DEG2`, identical for
    the two classes pixel by pixel, with the STAR/PAHC split carried
    only by the template weight factor (`_above_fraction`'s own
    `f_C`, sec. 5.1's `1 - P(q)`/`P(q)`) that scales `N_ABOVE_<C>` and
    `N_CELL_<C>` below -- `N_CAT_STAR`/`N_CAT_PAHC` keep their OWN split
    density from the members path above (`_build_one_tile`'s per-class
    `density`) and are therefore NOT `INTENSITY_<C> * accepted fraction`.
    `INTENSITY_<C>` carries no selection and no error attribute of its
    own, and no page draws its own ratio: `N_ABOVE_<C>` (deg^-2)
    is `INTENSITY_<C>` times the deterministic (no draw) fraction of the
    class's own stored grain shape brighter, once dimmed by the pixel's
    own extinction column, than the pixel's own 4.5 um 50% completeness
    limit (`_above_fraction`'s own docstring paragraph); the atlas's
    intrinsic page draws `P(C | pixel, above the limit) = N_ABOVE_C /
    sum(N_ABOVE)`, never `INTENSITY_C`'s own ratio.

    `agb_pool`, `gal_members`, `giannini_ratios`, `yso_reg`,
    `fitted_rates` and `pooled_eta` are every survey-wide input this
    build needs (F1, F5 2026-10-09 diagnosis): computed once by `build`
    and passed down here unchanged, never re-derived per region --
    `_agb_shell_pool`, `_gal_members`, `h2s_module._load_giannini_ratios`,
    `_yso_register`, `knot_rate.fitted` and `knot_rate.pooled`'s own
    returns, in that order."""
    with progress.Stage("bmstp.atlas", region) as st:
        # the detection probability (sec. 6.2, sec. 3.3's "the depth
        # grid") reads Rob Gutermuth's own measured completeness curve
        # directly, at each admitted pixel's own row into
        # `PIXEL_CURVE_<band>` (`catalogued_probability`'s own
        # docstring) -- the SAME curve `fittp.sweep` prices, never a
        # second region-band fit.
        pix, f_lim = _depth_grid(config, region)
        pixel_row = np.arange(pix.size, dtype=np.int64)
        n_pix = pix.size
        coverage = _coverage(config, region, pix)
        a_col, arm = _pixel_column(config, region, pix)
        a_col_gas = _pixel_gas_column(config, region, pix)
        tile_of_pix, n_tile_filled = _pixel_tile(config, region, pix)

        n_cat = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in CLASSES}
        # sec. 8: each class's own intrinsic
        # sky density at the pixel, `build_region`'s own docstring
        # paragraph above.
        intensity = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in CLASSES}
        # sec. 9's bright-end check: the same per-class densities as
        # `n_cat`, restricted to the members whose dimmed I2 flux clears
        # 3x/10x the pixel's own I2 50% limit.
        n_cat_bright3 = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in CLASSES}
        n_cat_bright10 = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in CLASSES}

        star_path = config_module.product_path(
            config, "population", "star", "population", "tile", region=region)
        with h5py.File(star_path, "r") as f:
            tile_ids_present = sorted(int(k.split("_")[1]) for k in f.keys()
                                       if k.startswith("tile_"))

        # every admitted pixel now carries a tile (`_pixel_tile` fills the
        # tile-less ones from their nside-256 parent, sec. 8); a tile
        # absent from this star product would otherwise leave that
        # pixel's STAR/AGB/PAHC as a silent NaN that turns every
        # SHARE_*/RATIO_* NaN for the whole region (sec. 8's total-count
        # check), so it is checked here instead.
        usable = tile_of_pix >= 0
        missing_tile = usable & ~np.isin(tile_of_pix, np.asarray(tile_ids_present, dtype=np.int64))
        if np.any(missing_tile):
            bad = int(np.flatnonzero(missing_tile)[0])
            raise ValueError("bmstp.atlas: pixel %d's tile %d is absent from the star "
                              "population product %s" % (int(pix[bad]), int(tile_of_pix[bad]), star_path))
        tiles_here = sorted(set(int(t) for t in tile_of_pix[usable]))

        # worker count is `root.cfg`'s own `[run] n_jobs` (CODING_RULES_BMSTP.md
        # rule 10a): the owner sets it to what the machine's memory allows.
        n_jobs = int(config.n_jobs)

        # every class is a deterministic quadrature (sec. 8): the region's
        # own catalogued-count-per-cell grid, accumulated tile by tile
        # (STAR/AGB/PAHC), sightline by sightline (YSO/H2S) and once for
        # GAL, below. `n_cell` is the SAME accumulation, every member's
        # own probability of being catalogued set to one (module
        # docstring's `N_CELL_<C>`).
        n_cat_cell = {c: np.zeros((grid.LOG10_XI_EDGES.size - 1, grid.LOG10_F45_EDGES.size - 1))
                      for c in ("STAR", "PAHC", "AGB", "YSO", "H2S")}
        n_cell = {c: np.zeros((grid.LOG10_XI_EDGES.size - 1, grid.LOG10_F45_EDGES.size - 1))
                  for c in ("STAR", "PAHC", "AGB", "YSO", "H2S")}

        # ONE pool, one flat list of pieces across tiles, GAL, sightlines
        # and the tail (owner's ruling 2026-10-09): every piece below is
        # computed by `_dispatch`, below, on the region's one pool
        # (`pieces.run`); the parent reduces each group's own results,
        # AFTER the one dispatch call returns, in the SAME fixed order
        # (tiles_here order, GAL's own chunk order, sls_here order, the
        # tail's class/pixel-slice order) the three former `Parallel`
        # calls and the N_ABOVE-only `pieces.run` call used -- so this is
        # a dispatch-mechanism change only, not a computation change:
        # every piece below runs the exact function call its own former
        # phase made. GAL's own pixel chunks (`_gal_pixel_chunks`) are
        # built here instead of inside a GAL-only call, so they can share
        # the one flat list.
        gal_idx_chunks = _gal_pixel_chunks(config, n_pix)
        weight_pix_all = coverage * _HPX512_PIXEL_DEG2

        # YSO/H2S, sec. 5.5-5.6: grouped by the pixel's own nside-256
        # sightline (YSO's grain), one deterministic quadrature per
        # sightline shared by every admitted pixel it parents. This setup
        # (region-wide, no tile/GAL/sightline RESULT it depends on) used
        # to sit between the GAL call and the sightline `Parallel` call;
        # it is hoisted here, before the one dispatch, so every piece
        # below -- tiles, GAL, sightlines and the tail alike -- can be
        # submitted together.
        reg = regions_module.REGIONS_BY_NAME[region]
        d_r_pc = float(reg.d_r_pc)
        loaded_profile = sample_cloud._region_profile(config, region)
        # the cloud's own share of the column, per sightline
        # (`bmstp.density._cloud_column_fraction`, W26): the intrinsic
        # YSO/H2S density below is the law applied to `a_cloud`, not the
        # sightline's whole adopted column.
        cloud_frac_by_sl, d_front = density_module._cloud_column_fraction(config, region)
        _d_front, d_back = sample_cloud.cloud_interval_pc(config, region)
        # YSO's members: the region-wide template x shift-kernel
        # quadrature (sec. 5.5 "Template weights"), built once, shared by
        # every sightline of the region -- only the depth factor differs
        # per sightline (`_build_one_sightline`).
        flux0_ts, w_ts = _yso_region_nodes(yso_reg, config, region, d_front, d_back)
        # H2S's brightness lognormal (sec. 5.6 "Marks") is P3's own
        # attribute (`bmstp.shapes.build_cloud`, sec. 4.1's shape grids):
        # no population/h2s region product to read, that stage no longer
        # runs (RUNBOOKtp.sh).
        p3_path = config_module.product_path(
            config, "bmstp", "shape", "cloud", "sightline", region=region)
        with h5py.File(p3_path, "r") as f:
            logsig_mean = float(f.attrs["LOGSIG_MEAN"])
            logsig_std = float(f.attrs["LOGSIG_STD"])
            p3_sl_axis = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
            on_grid_yso_p3 = np.asarray(f["ON_GRID_YSO"][:], dtype=np.float64)
        # H2S's own brightness x colour-ratio quadrature (sec. 5.6
        # "Marks"), region-wide, built once, crossed with YSO's own depth
        # nodes per sightline (`_build_one_sightline`).
        flux0_br, w_br = _h2s_region_nodes(logsig_mean, logsig_std, giannini_ratios)

        sl_axis = loaded_profile["hpx_pix_256"]
        order_sl = np.argsort(sl_axis)
        sl_parent = pix // 4
        loc_sl = np.minimum(np.searchsorted(sl_axis[order_sl], sl_parent), sl_axis.size - 1)
        hit_sl = order_sl[loc_sl]
        has_sl = sl_axis[hit_sl] == sl_parent
        if not np.all(has_sl):
            bad = int(np.flatnonzero(~has_sl)[0])
            raise ValueError("bmstp.atlas: pixel %d's nside-256 parent %d is absent from the "
                              "cloud profile's sightline axis" % (int(pix[bad]), int(sl_parent[bad])))
        sl_row_of_pix = hit_sl
        sls_here = sorted(set(int(r) for r in sl_row_of_pix))

        # every class's density is the retained density everywhere (sec.
        # 4.1): P1 (`bmstp.density`) multiplies its own YSO law by
        # `ON_GRID_YSO` (P3's per-sightline retention on the common
        # depth axis); the atlas's own YSO density below is that SAME
        # law with no retention (it needs the untouched intrinsic to
        # weight H2S, sec. 5.6), so `CATALOGABLE_FRACTION_YSO`'s
        # denominator applies `ON_GRID_YSO` here instead, to be
        # normalised against the same retained density P1 reports as
        # `DENSITY_YSO`.
        loc_p3 = np.searchsorted(p3_sl_axis, sl_axis)
        loc_p3 = np.minimum(loc_p3, p3_sl_axis.size - 1)
        found_p3 = p3_sl_axis[loc_p3] == sl_axis
        if not np.all(found_p3[sl_row_of_pix]):
            bad = int(np.flatnonzero(~found_p3[sl_row_of_pix])[0])
            raise ValueError("bmstp.atlas: sightline pixel %d is absent from P3's ON_GRID_YSO "
                              "axis %s" % (int(sl_axis[sl_row_of_pix[bad]]), p3_path))
        on_grid_yso_by_sl_row = on_grid_yso_p3[loc_p3]

        # H2S, sec. 5.6 "Sky density": `A_H2S = L . eta_r . eps_ext`. `L` is
        # a SEPARATE evaluation of the SAME quadratic law, at the
        # SURVEY-POOLED coefficient `KAPPA_POOLED`, never this region's
        # own fitted `KAPPA_USED` (`population.yso.law_count` reads for
        # YSO's own density, `density_yso` inside `_build_one_sightline`,
        # untouched) -- `population.knot_rate.eta_for_region`'s own
        # calibration denominator is formed at that pooled level, so the
        # law it multiplies must be too, on both arms, or the ratio does
        # not cancel (WP-POP-5's finding; `knot_field.kappa_pooled`/
        # `law_count_at_kappa`, the SAME functions `bmstp.density` reads
        # for its own H2S path). `L` is entirely deterministic (no random
        # draw), so it is built here, before the sightline quadrature,
        # from the same `a_col_gas * cloud_frac` every sightline itself
        # computes for its own YSO density, at the pooled coefficient
        # instead -- EXCEPT on the Herschel arm, where it is the region's
        # convolved law map (`bmstp.knot_field.convolved_law`, already
        # built at `kappa_pooled`) averaged over the pixel's own area
        # (item 3: the mean of L over the pixel, not a single nearest
        # sample -- a pixel is much larger than the map's own downsampled
        # grid); a Planck pixel, or a Herschel pixel the convolved map
        # does not reach, keeps the pooled-coefficient point law.
        # `_herschel_convolved_law` is the one function both this weight
        # and H2S's mean density below read, so they never disagree. The
        # map itself is convolved ONCE here (sec. 5.6 "a map operation,
        # once per region") and passed to both calls.
        # the region's knot rate, formed at build against the fitted law
        # (`population.knot_rate`, the same call `bmstp.density` makes):
        # `knot_rate.eta_for_region`'s own branch, inlined here against
        # `fitted_rates`/`pooled_eta` (`build`'s own `knot_rate.fitted`/
        # `knot_rate.pooled` call, computed once for the whole survey --
        # F1 2026-10-09: 27 of 30 regions share the one pooled value,
        # which `eta_for_region` otherwise recomputes, from scratch,
        # every region).
        if region in knot_rate.KNOT_COMPLETENESS:
            eta_r, _eta_band_dex = fitted_rates[region], 0.0
        else:
            eta_r, _eta_band_dex = pooled_eta
        law_map, law_wcs, knot_meta = knot_field.convolved_law(config, region)
        density_h2s_law_pooled_pix = knot_field.law_count_at_kappa(
            config, region, a_col_gas * cloud_frac_by_sl[sl_row_of_pix], arm,
            knot_field.kappa_pooled(config))
        l_of_pix_for_weight = _herschel_convolved_law(law_map, law_wcs, pix, arm, density_h2s_law_pooled_pix)

        # the tail's own per-pixel inputs (F5 2026-10-09 "change A"; same
        # computation as before, only hoisted above the one dispatch so
        # the tail's pieces can join the same flat list as tiles/GAL/
        # sightlines): each pixel's `N_ABOVE` depends only on that
        # pixel's own slice of `a_col`/`f0_pix`/`d_pahc_pix`/grain
        # (`_above_fraction`'s own per-pixel reduction, `term.sum(axis=
        # (1,2))`), never on a tile, GAL or sightline RESULT, so these
        # pieces need nothing the dispatch below produces.
        f0_pix = f_lim[:, IDX_I2]
        d_pahc_pix = -np.log10(f_lim[:, IDX_I4])
        # `fittp.prior_reader.load`'s `GRID_YSO`/`GRID_H2S` are P3's own
        # array, in P3's own `HPX_PIX_256` row order -- NOT the cloud
        # profile's own order `sl_row_of_pix` indexes (the two sightline
        # axes can differ, which is exactly why `loc_p3` above resolves
        # `on_grid_yso_by_sl_row` the same way): `loc_p3[sl_row_of_pix]`
        # is the pixel's row in P3's own order, the correct grain index
        # into `reader.grid_all` for YSO/H2S.
        p3_row_of_pix = loc_p3[sl_row_of_pix]
        _gal_grain = np.zeros(n_pix, dtype=np.int64)
        grain_of_pix_by_cls = {
            "STAR": tile_of_pix, "AGB": tile_of_pix, "PAHC": tile_of_pix,  # P2's TILE_ID is dense 0..n_tile-1
            "YSO": p3_row_of_pix, "H2S": p3_row_of_pix, "PAHN": p3_row_of_pix,  # PAHN is on P3 too
            "GAL": _gal_grain,  # one survey-wide grain (`prior_reader.load`'s own convention)
        }
        # STAR and PAHC alone carry the tail's own cost (F5: PAHC's own
        # 57,603-model factor is over 98% of the element work; STAR's own
        # 151-model factor is the same shape, far smaller): only they are
        # cut into many small pixel pieces; GAL/AGB/YSO/H2S (no
        # template-weight factor, ~1% of the tail together) are each one
        # piece -- more pieces than that buys nothing for them (measured
        # on NGC 7129: piece dispatch overhead, not file-open or compute,
        # dominates a class with ~1 ms/pixel of real work), but putting
        # even their one piece each in the SAME flat list (rather than a
        # direct parent-process call) is what lets every piece in the
        # region share the one pool. `reader`/`curve` are opened once per
        # class, not once per piece (the same NGC 7129 measurement:
        # re-opening `prior_reader.load`'s several HDF5 files per piece
        # cost far more than one pixel's own work) and closed over by
        # every piece; `reader`'s own arrays are memmaps (`prior_reader.
        # mmap_dataset`) and `curve` is one small `interp1d`, so the
        # closure joblib pickles once per task batch carries file paths
        # and small arrays, not a full read. `min_items=8` keeps a
        # region's piece count from outrunning what 8 pixels of this
        # cost is worth dispatching.
        readers_by_cls = {cls: prior_reader.load(config, region, cls) for cls in CLASSES}
        curve = pahc_curve.read(config, region)

        # PAHN (C14_pahn_population.md): INTENSITY_PAHN is the measured
        # curve alone at each admitted pixel's own I4 50% limit (`f_lim`,
        # already this pixel's own row into PIXEL_CURVE_I4) -- no eta, no
        # eps_ext, no on-grid retention here (sec. 4.1: the atlas's own
        # intrinsic page is pre-retention everywhere, the SAME convention
        # YSO/H2S's own `density_yso`/`density_h2s_law_pooled_pix` use).
        # Its own region-wide member quadrature (`_pahn_region_nodes`)
        # reads WEIGHTS' own placement templates (`readers_by_cls["PAHN"]
        # .c_theta`) straight off disk -- "the member draw from the PAHN
        # shape and weights" (brief) -- placed at this region's own
        # distance, with the PAHN register's own 8-band SHAPE (C17
        # 2026-10-20), not a single global ratio.
        density_pahn_pix = pahn_module.read(config, region)(np.log10(f_lim[:, IDX_I4]))
        intensity["PAHN"] = density_pahn_pix
        pahn_members = _pahn_region_nodes(config, region, readers_by_cls["PAHN"], d_r_pc)

        # the N_ABOVE tail's own "brighter than the pixel's own limit"
        # mask (`_above_shared`'s docstring): depends only on `a_col`/
        # `f0_pix`, identical for every one of the six classes, so it is
        # computed once here (any one class's own reader supplies the
        # SAME shape-grid edges) instead of once per class inside
        # `_above_fraction`.
        above_pix, _b_centers_shared = _above_shared(config, readers_by_cls["STAR"], a_col, f0_pix)

        def _dispatch(piece):
            kind = piece[0]
            if kind == "TILE":
                tile_id = piece[1]
                m = usable & (tile_of_pix == tile_id)
                return piece, (m, _build_one_tile(
                    config, region, tile_id, pix[m], a_col[m], f_lim[m], pixel_row[m], agb_pool,
                    coverage[m]))
            if kind == "GAL":
                idx = gal_idx_chunks[piece[1]]
                return piece, _gal_chunk_result(
                    config, region, gal_members, idx, a_col, f_lim, pixel_row, weight_pix_all)
            if kind == "PAHN":
                idx = gal_idx_chunks[piece[1]]
                return piece, _pahn_chunk_result(
                    config, region, pahn_members, density_pahn_pix, idx, a_col, f_lim, pixel_row,
                    weight_pix_all)
            if kind == "SL":
                sl_row = piece[1]
                m = sl_row_of_pix == sl_row
                out = _build_one_sightline(
                    config, region, sl_row, a_col[m], a_col_gas[m], arm[m], f_lim[m],
                    loaded_profile, flux0_ts, w_ts, float(cloud_frac_by_sl[sl_row]), d_front, d_back,
                    flux0_br, w_br, pixel_row[m],
                    coverage[m] * _HPX512_PIXEL_DEG2, eta_r, l_of_pix_for_weight[m])
                return piece, (m,) + out
            if kind == "TAIL":
                cls, start, stop = piece[1], piece[2], piece[3]
                frac_above = _above_fraction(
                    readers_by_cls[cls], cls, grain_of_pix_by_cls[cls][start:stop],
                    above_pix[start:stop], d_pahc_pix[start:stop], curve)
                return piece, frac_above
            cls = piece[1]  # "TAIL_WHOLE"
            frac_above = _above_fraction(
                readers_by_cls[cls], cls, grain_of_pix_by_cls[cls],
                above_pix, d_pahc_pix, curve)
            return piece, frac_above

        tail_split_pieces = [("TAIL", cls, start, stop) for cls in ("STAR", "PAHC")
                             for start, stop in pieces.slices(n_pix, n_jobs, min_items=8)]
        flat_pieces = (
            [("TILE", t) for t in tiles_here]
            + [("GAL", i) for i in range(len(gal_idx_chunks))]
            + [("PAHN", i) for i in range(len(gal_idx_chunks))]
            + [("SL", r) for r in sls_here]
            + tail_split_pieces
            + [("TAIL_WHOLE", cls) for cls in ("AGB", "GAL", "YSO", "H2S", "PAHN")])
        # ---- reduce each piece as it arrives, in the SAME fixed order
        # the former collect-then-reduce dict walk used (tiles_here order,
        # GAL's own chunk order, sls_here order, the tail's class/pixel-
        # slice order): `pieces.iterate` yields results in piece order, so
        # processing one result per iteration and dropping it is the same
        # reduction, only without `flat_out` holding every one of the
        # region's pieces (tiles, GAL chunks, sightlines and the tail)
        # live at once (F4 2026-10-09: +29% peak on Perseus from exactly
        # that collection). A later piece may depend on an earlier one's
        # reduction (the tail's `intensity[cls]`, H2S's `density_h2s`
        # from the last sightline) only because it is later in this same
        # fixed order, never because of anything outside it.
        gal_chunk_results = [None] * len(gal_idx_chunks)
        n_gal_seen = 0
        pahn_chunk_results = [None] * len(gal_idx_chunks)
        n_pahn_seen = 0
        frac_h2s_pix = np.full(n_pix, np.nan, dtype=np.float64)
        frac_h2s_bright3_pix = np.full(n_pix, np.nan, dtype=np.float64)
        frac_h2s_bright10_pix = np.full(n_pix, np.nan, dtype=np.float64)
        density_yso_pix = np.full(n_pix, np.nan, dtype=np.float64)
        n_members_yso_list, n_members_h2s_list = [], []
        n_sl_seen = 0
        n_above = {cls: np.empty(n_pix, dtype=np.float64) for cls in CLASSES}

        for piece, data in pieces.iterate(_dispatch, flat_pieces, n_jobs, stage=st, unit="region pieces"):
            kind = piece[0]
            if kind == "TILE":
                m, out = data
                # sec. 5.3: `A_PAHC(s) = A_STAR(s)`, the field-star level
                # shared by both classes -- the STAR/PAHC split lives only
                # in `_above_fraction`'s own weight factor (sec. 5.1's
                # `1 - P(q)`/`P(q)`), never in `INTENSITY_<C>` again.
                field_star_density = out["_FIELD_STAR_DENSITY"]
                for cls in ("STAR", "AGB", "PAHC"):
                    frac, density, frac_bright3, frac_bright10, n_cat_cell_tile, n_cell_tile = out[cls]
                    n_cat[cls][m] = density * frac
                    intensity[cls][m] = field_star_density if cls in ("STAR", "PAHC") else density
                    n_cat_bright3[cls][m] = density * frac_bright3
                    n_cat_bright10[cls][m] = density * frac_bright10
                    n_cat_cell[cls] += n_cat_cell_tile
                    n_cell[cls] += n_cell_tile
            elif kind == "GAL":
                gal_chunk_results[piece[1]] = data
                n_gal_seen += 1
                if n_gal_seen == len(gal_idx_chunks):
                    (frac_gal, frac_gal_bright3, frac_gal_bright10, n_cat_cell_gal, n_cell_gal,
                     density_gal, n_gal_members_before, n_gal_members_after) = _gal_reduce(
                        gal_members, gal_chunk_results, weight_pix_all)
                    n_cat["GAL"] = density_gal * frac_gal
                    # sec. 5.4's own density is one survey-wide constant,
                    # so GAL's intensity is that same scalar broadcast to
                    # every admitted pixel.
                    intensity["GAL"] = np.full(n_pix, density_gal, dtype=np.float64)
                    n_cat_bright3["GAL"] = density_gal * frac_gal_bright3
                    n_cat_bright10["GAL"] = density_gal * frac_gal_bright10
                    n_cat_cell["GAL"] = n_cat_cell_gal
                    n_cell["GAL"] = n_cell_gal
            elif kind == "PAHN":
                pahn_chunk_results[piece[1]] = data
                n_pahn_seen += 1
                if n_pahn_seen == len(gal_idx_chunks):
                    (frac_pahn, frac_pahn_bright3, frac_pahn_bright10,
                     n_cat_cell_pahn, n_cell_pahn) = _pahn_reduce(
                        pahn_members, pahn_chunk_results, density_pahn_pix, weight_pix_all)
                    n_cat["PAHN"] = density_pahn_pix * frac_pahn
                    n_cat_bright3["PAHN"] = density_pahn_pix * frac_pahn_bright3
                    n_cat_bright10["PAHN"] = density_pahn_pix * frac_pahn_bright10
                    n_cat_cell["PAHN"] = n_cat_cell_pahn
                    n_cell["PAHN"] = n_cell_pahn
            elif kind == "SL":
                sl_row = piece[1]
                (m, f_y, d_y, fb3_y, fb10_y, ncc_y, ncl_y,
                 f_h, fb3_h, fb10_h, ncc_h, ncl_h, n_mem_y, n_mem_h) = data
                n_cat["YSO"][m] = d_y * f_y
                intensity["YSO"][m] = d_y
                frac_h2s_pix[m] = f_h
                density_yso_pix[m] = d_y
                n_cat_bright3["YSO"][m] = d_y * fb3_y
                n_cat_bright10["YSO"][m] = d_y * fb10_y
                frac_h2s_bright3_pix[m] = fb3_h
                frac_h2s_bright10_pix[m] = fb10_h
                n_cat_cell["YSO"] += ncc_y
                n_cell["YSO"] += ncl_y
                n_cat_cell["H2S"] += ncc_h
                n_cell["H2S"] += ncl_h
                n_members_yso_list.append(n_mem_y)
                n_members_h2s_list.append(n_mem_h)
                n_sl_seen += 1
                if n_sl_seen == len(sls_here):
                    print(f"bmstp.atlas {region}: YSO members/sightline "
                          f"median={int(np.median(n_members_yso_list))} "
                          f"max={int(np.max(n_members_yso_list))}; H2S members/sightline "
                          f"median={int(np.median(n_members_h2s_list))} max={int(np.max(n_members_h2s_list))}")
                    # this atlas's own INTRINSIC YSO density (before
                    # retention), multiplied by P3's own `ON_GRID_YSO` at
                    # the pixel's sightline: the SAME retained density P1
                    # reports as `DENSITY_YSO`.
                    density_yso_retained_pix = density_yso_pix * on_grid_yso_by_sl_row[sl_row_of_pix]
                    # H2S's mean density reads the SAME pooled-coefficient
                    # baseline the weight above does
                    # (`density_h2s_law_pooled_pix`), never
                    # `density_yso_pix` (YSO's own region-coefficient
                    # density, kept for YSO's own
                    # `density_yso_retained_pix`/`intensity["YSO"]` above
                    # and untouched) -- the same basis mismatch WP-POP-5
                    # found in `bmstp.density` existed here too, on the
                    # SAME two lines.
                    l_of_pix = _herschel_convolved_law(law_map, law_wcs, pix, arm, density_h2s_law_pooled_pix)
                    density_h2s_before = density_h2s_law_pooled_pix * eta_r * density_module.EPS_EXT
                    density_h2s = l_of_pix * eta_r * density_module.EPS_EXT
                    n_cat["H2S"] = density_h2s * frac_h2s_pix
                    intensity["H2S"] = density_h2s
                    n_cat_h2s_before = density_h2s_before * frac_h2s_pix
                    n_cat_bright3["H2S"] = density_h2s * frac_h2s_bright3_pix
                    n_cat_bright10["H2S"] = density_h2s * frac_h2s_bright10_pix
            elif kind == "TAIL":
                # `N_ABOVE_C(pixel) = A_C(pixel) * _above_fraction(...)`
                # (`_above_fraction`'s own docstring paragraph), read
                # straight off each class's own stored grain shape
                # (`fittp.prior_reader.load`) rather than the catalogued-
                # fraction members above -- the atlas's intrinsic page
                # draws this, not `INTENSITY_C`'s own ratio
                # (SPEC_BMSTP_DRAFT.md sec. 8). `intensity[cls]` is
                # already filled: every TILE/GAL/SL piece for this class
                # precedes every TAIL piece in `flat_pieces`.
                cls, start, stop = piece[1], piece[2], piece[3]
                n_above[cls][start:stop] = intensity[cls][start:stop] * data
            else:  # "TAIL_WHOLE"
                cls = piece[1]
                n_above[cls] = intensity[cls] * data

        built = CLASSES
        # every admitted pixel now has every class's `N_CAT` (finding 3
        # above), so a plain sum replaces the `nansum` that used to treat
        # a tile-less pixel's family densities as zero.
        built_total = np.sum([n_cat[c] for c in built], axis=0)
        share = {c: n_cat[c] / built_total for c in built}

        n_source = access.region_slice(config, region)["n_sources"]
        area_deg2 = n_pix * _HPX512_PIXEL_DEG2
        # sec. 8's total-count check compares against sources actually
        # catalogued, which only covered sky can catalogue: each pixel's
        # predicted count is weighted by its own `COVERAGE` before the
        # totals are formed (the surveyed area is `Sigma COVERAGE * area`,
        # not the admitted grid's own `area_deg2`).
        surveyed_area_deg2 = float(np.sum(coverage) * _HPX512_PIXEL_DEG2)
        total_predicted_built = float(np.sum(built_total * coverage) * _HPX512_PIXEL_DEG2)
        ratio = {c: float(np.sum(n_cat[c] * coverage) * _HPX512_PIXEL_DEG2) / n_source
                 if n_source else float("nan") for c in built}
        ratio_built = total_predicted_built / n_source if n_source else float("nan")

        # the YSO population's catalogable fraction (spec sec 5.5, sec
        # 8): the region's own detection-weighted accepted share of the
        # RETAINED young-star density, `Sigma(n_cat[YSO] *
        # coverage) / Sigma(density_yso_retained_pix * coverage)` -- the
        # same coverage weighting the total-count check above applies,
        # normalised against the same retained density P1 reports as
        # `DENSITY_YSO` (`ON_GRID_YSO` applied, sec. 4.1), not the raw
        # law with no retention, and never the library's eight-band
        # density (sec 1.4).
        density_yso_total = float(np.sum(density_yso_retained_pix * coverage) * _HPX512_PIXEL_DEG2)
        n_cat_yso_total = float(np.sum(n_cat["YSO"] * coverage) * _HPX512_PIXEL_DEG2)
        catalogable_fraction_yso = (n_cat_yso_total / density_yso_total
                                     if density_yso_total > 0 else float("nan"))
        print(f"bmstp.atlas {region}: YSO catalogable fraction={catalogable_fraction_yso:.4f} "
              f"(n_cat={n_cat_yso_total:.6g} / intrinsic={density_yso_total:.6g})")

        # sec. 9's bright-end check: the catalogue's own bright
        # counterpart per admitted pixel, and the same coverage/area
        # weighting the total-count check applies above, denominator
        # swapped for the observed bright total.
        n_obs, n_obs_bright3, n_obs_bright10 = _observed_bright_counts(config, region, pix, f_lim)
        sum_n_obs_bright3 = float(np.sum(n_obs_bright3))
        sum_n_obs_bright10 = float(np.sum(n_obs_bright10))
        ratio_bright3 = {c: float(np.sum(n_cat_bright3[c] * coverage) * _HPX512_PIXEL_DEG2) / sum_n_obs_bright3
                          if sum_n_obs_bright3 else float("nan") for c in built}
        ratio_bright10 = {c: float(np.sum(n_cat_bright10[c] * coverage) * _HPX512_PIXEL_DEG2) / sum_n_obs_bright10
                           if sum_n_obs_bright10 else float("nan") for c in built}
        ratio_bright3_total = sum(ratio_bright3.values())
        ratio_bright10_total = sum(ratio_bright10.values())

        path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)

        print(f"bmstp.atlas: {region} ratio_all={ratio_built:.4g} ratio_bright3={ratio_bright3_total:.4g} "
              f"ratio_bright10={ratio_bright10_total:.4g} "
              f"(STAR {ratio_bright3['STAR']:.4g}, GAL {ratio_bright3['GAL']:.4g}) "
              f"n_obs_bright3={sum_n_obs_bright3:.6g} n_obs_bright10={sum_n_obs_bright10:.6g}")

        # sec. 5.6's brief report: RATIO_H2S before (the pre-W5k law at the
        # pixel's own column, no kernel) and after (this unit's convolved
        # law on the Herschel arm) -- the same total-count ratio the other
        # classes get, sec. 8.
        ratio_h2s_before = (float(np.sum(n_cat_h2s_before * coverage) * _HPX512_PIXEL_DEG2) / n_source
                             if n_source else float("nan"))
        print(f"bmstp.atlas {region}: RATIO_H2S before={ratio_h2s_before:.6g} "
              f"after={ratio['H2S']:.6g} (sec. 5.6's convolution vs. the law at the pixel's own column)")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "hpx512"
            f.attrs["TOTAL_PREDICTED"] = total_predicted_built
            f.attrs["SURVEYED_AREA_DEG2"] = surveyed_area_deg2
            f.attrs["TOTAL_OBSERVED"] = float(n_source)
            for c in built:
                f.attrs[f"RATIO_{c}"] = ratio[c]
            f.attrs["RATIO_BUILT"] = ratio_built
            # sec. 9's bright-end check: the same catalogued/observed
            # ratio, above 3x/10x the pixel's own I2 50% limit, where
            # completeness is 1 on both sides. Read by fittp.figure_
            # count_check as the two TOTAL numbers only (grep, step 5
            # 2026-10-19): the per-class RATIO_BRIGHT3_<C> attr, DENSITY_GAL,
            # ADMITTED_AREA_DEG2 and CATALOGABLE_FRACTION_YSO are read by
            # nothing and no longer written; the build still computes and
            # prints catalogable_fraction_yso and density_gal above.
            f.attrs["RATIO_BRIGHT3"] = ratio_bright3_total
            f.attrs["RATIO_BRIGHT10"] = ratio_bright10_total
            for name, data in (
                ("HPX_PIX_512", pix),
                ("A_COL_K", a_col.astype(np.float32)),
                ("COVERAGE", coverage.astype(np.float32)),
            ):
                build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])
            # F_LIM_50_PIX_MJY is read by nothing here: every reader goes
            # straight to catalog.depth_grid's own copy (atlas/shapes.py,
            # atlas/protostars.py); no longer written.
            for c in CLASSES:
                name = f"N_CAT_{c}"
                build_module.write_dataset(f, name, n_cat[c].astype(np.float32), *REGISTRY[(_STEM, name)])
            for c in CLASSES:
                name = f"N_ABOVE_{c}"
                build_module.write_dataset(f, name, n_above[c].astype(np.float32), *REGISTRY[(_STEM, name)])
            # INTENSITY_YSO is atlas/protostars.py's own read (grep, step 5
            # 2026-10-19); INTENSITY_STAR/AGB/PAHC/GAL/H2S and every
            # N_CAT_BRIGHT3_<C>/N_CAT_BRIGHT10_<C> (the bright-end check's
            # per-pixel tables; only their coverage-weighted sums survive,
            # as the two RATIO_BRIGHT3/RATIO_BRIGHT10 attrs above) are read
            # by nothing and no longer written.
            build_module.write_dataset(f, "INTENSITY_YSO", intensity["YSO"].astype(np.float32),
                                        *REGISTRY[(_STEM, "INTENSITY_YSO")])
            # the region's expected number of catalogued objects per
            # parameter cell (module docstring), summing to `RATIO_<C> *
            # N_source`; `(128, 110)` on `grid.LOG10_XI_EDGES` x
            # `grid.LOG10_F45_EDGES`, the shapes' own axes.
            for c, data in n_cat_cell.items():
                name = f"N_CAT_CELL_{c}"
                build_module.write_dataset(f, name, data.astype(np.float32), *REGISTRY[(_STEM, name)])
            # the region's UNTHINNED intrinsic population per parameter
            # cell (module docstring, sec. 8): the same members quadrature
            # as `N_CAT_CELL_<C>` with every member's own probability of
            # being catalogued set to one, accumulated tile by tile,
            # sightline by sightline and once for GAL exactly like
            # `N_CAT_CELL_<C>` above, same axes.
            for c in CLASSES:
                name = f"N_CELL_{c}"
                build_module.write_dataset(f, name, n_cell[c].astype(np.float32), *REGISTRY[(_STEM, name)])
            for c in built:
                name = f"SHARE_{c}"
                build_module.write_dataset(f, name, share[c].astype(np.float32), *REGISTRY[(_STEM, name)])
            # N_OBS/N_OBS_BRIGHT3/N_OBS_BRIGHT10 are read by nothing here
            # (grep, step 5 2026-10-19): the build still computes them
            # in-process, for RATIO_BRIGHT3/RATIO_BRIGHT10 above, and no
            # longer writes them.

        st.done(path, n_pix=n_pix, n_tile=len(tiles_here), n_sightline=len(sls_here),
                n_tile_filled=n_tile_filled,
                area_deg2=area_deg2, surveyed_area_deg2=surveyed_area_deg2,
                d_front_pc=d_front, d_back_pc=d_back,
                total_predicted_built=total_predicted_built, total_observed=n_source,
                ratio_star=ratio["STAR"], ratio_agb=ratio["AGB"], ratio_pahc=ratio["PAHC"],
                ratio_gal=ratio["GAL"], ratio_yso=ratio["YSO"], ratio_h2s=ratio["H2S"],
                ratio_built=ratio_built, density_gal_deg2=density_gal,
                n_gal_members_before=n_gal_members_before, n_gal_members_after=n_gal_members_after,
                ratio_bright3=ratio_bright3_total, ratio_bright10=ratio_bright10_total,
                n_obs_bright3=int(sum_n_obs_bright3), n_obs_bright10=int(sum_n_obs_bright10))
    return path


def build(config, regions=None):
    """`build(config, regions=None)`: per region, `build_region` (rule
    5c's per-region product, one file per region).

    Every survey-wide input `build_region` needs is computed exactly
    ONCE here, for the whole build, and passed down unchanged (F1, F5
    2026-10-09 diagnosis: `build_region` used to re-derive each of these
    from scratch at the start of every one of the 30 regions -- about
    460 s of the measured 4,200 s run, almost all of it
    `knot_rate.fitted`'s own three HGBS area-integral calls, repeated
    for every region without its own knot survey): `_agb_shell_pool`
    (sec. 5.2's shell-template pool by chemistry), `_gal_members` (sec.
    5.4's GAL population, sky density included), `h2s_module.
    _load_giannini_ratios` (sec. 5.6's colour-ratio tables), `_yso_register`
    (sec. 5.5's census-weighted template register), and `knot_rate.
    fitted`/`knot_rate.pooled` (sec. 5.6's knot rate: each knot-surveyed
    field's own fitted rate, and the geometric-mean rate every other
    region reads, `knot_rate.eta_for_region`'s own two branches -- see
    `build_region`'s inlined copy of that branch). None of these depends
    on the region loop below; every one is a deterministic read with no
    `rng`, so computing it once and reusing it for 30 regions changes no
    number, only how many times the same arithmetic runs."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    agb_pool = _agb_shell_pool(config)
    gal_members = _gal_members(config)
    giannini_ratios = h2s_module._load_giannini_ratios(config)
    yso_reg = _yso_register(config)
    fitted_rates = knot_rate.fitted(config)
    pooled_eta = knot_rate.pooled(config, fitted_rates)
    for region in region_names:
        build_region(config, region, agb_pool, gal_members, giannini_ratios,
                     yso_reg, fitted_rates, pooled_eta)


if __name__ == "__main__":
    run(build)
