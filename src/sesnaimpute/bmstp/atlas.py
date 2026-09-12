"""P6, the prior atlas (SPEC_BMSTP_DRAFT.md sec. 8; IMPLEMENTATION_BMSTP_DRAFT.md
sec. 1.2 P6, sec. 3 row 1.9).

Per admitted nside-512 pixel, a deterministic weighted quadrature over each
class's own population, dimmed at the pixel's own column through the blended
law (`population.selection.kappa_hybrid`), weighted by each member's
probability of being catalogued (sec. 8, sec. 6.2): per band the
completeness `C_i(f)` at the pixel's own marginalised 50% limit
(`catalog.depth_grid`'s `F_LIM_50_PIX_MJY`) and its own marginalised width
(`W_DEX_PIX`), and `P(>=2 of 8)` from the eight bands' independent
non-detection probabilities -- the same completeness model `fittp.likelihood`
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
members are the star-family sampler's own evolved stars (`bmstp.sample_star.
sample_agb`), each carrying one shell template of its own drawn chemistry
(Riebel+2012's optical-depth distribution, `bmstp.template_weights.build_agb`'s
`tau` factor construction) whose eight `F_REF` are scaled so its own 4.5 um
flux equals the star's `F_4.5`. All six classes enter the total-count check.

Every class's catalogued density is a weighted quadrature over its own whole
population (sec. 8): alongside it this build writes `N_CAT_CELL_<C>` (128, 110,
`grid.LOG10_X_EDGES` by `grid.LOG10_F45_EDGES`) for all six classes, each
class's own expected number of catalogued objects per parameter cell, summing
over cells to `RATIO_<C> * N_source`; and `N_CELL_<C>` (same axes, sec. 8's
intrinsic population), the class's UNTHINNED population per cell -- no flux
cut, no dimming -- `Sum_pix coverage * pixel area * A_C(pixel) * h_C(cell;
pixel) * f_C(cell; pixel)`, the same per-pixel grain shape and weight factor
the intrinsic view's `_above_fraction` reads, gathered over every cell rather
than collapsed above one flux limit.

Every module on this atlas's worker path (this module, `knot_field`,
`sample_gal`, `sample_star`, `sample_cloud`, `grid`, `density`,
`template_weights`, `sky.derived.profile`, `fittp.likelihood`,
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
from joblib import Parallel, delayed
from scipy.special import ndtri

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.population import h2s as h2s_module
from sesnaimpute.population import pahc_curve
from sesnaimpute.population import selection as selection_module
from sesnaimpute.population import yso as yso_module
from sesnaimpute.bmstp import density as density_module
from sesnaimpute.bmstp import grid
from sesnaimpute.bmstp import knot_field
from sesnaimpute.bmstp import sample_cloud
from sesnaimpute.bmstp import sample_gal
from sesnaimpute.bmstp import sample_star
from sesnaimpute.bmstp import template_weights
from sesnaimpute.fittp import likelihood as likelihood_module
from sesnaimpute.fittp import prior_reader

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)
IDX_I4 = BAND_KEYS.index("I4")
IDX_I2 = BAND_KEYS.index("I2")
CLASSES = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")

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

_HPX512_PIXEL_DEG2 = 41252.96 / (12 * 512 ** 2)


def _depth_grid(config, region):
    """The admitted pixel axis and its marginalised 50% limit and width
    (`catalog.depth_grid`, sec. 3.3, the fixed point of the per-pixel
    `log10 DCOMP90` vs `log10 f` fit, not the brightness-biased median):
    `(pix, f_lim_50_pix_mjy, w_dex_pix)`. A band the region's own counts
    fit cannot solve is unsurveyed there and carries `+inf`/`0` (never
    `NaN`, `catalog.depth_grid`'s own rule); a `NaN` here means that rule
    was not kept upstream, and every reader downstream would silently
    turn it into a NaN region total (module docstring), so it raises here
    instead, exactly as `_pixel_column` raises on a missing sightline."""
    path = config_module.product_path(config, "catalog", "sesna", "depth-grid", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        f_lim = np.asarray(f["F_LIM_50_PIX_MJY"][:], dtype=np.float64)
        w_dex_pix = np.asarray(f["W_DEX_PIX"][:], dtype=np.float64)
    bad = np.isnan(f_lim) | np.isnan(w_dex_pix)
    if bad.any():
        p_idx, b_idx = np.unravel_index(int(np.argmax(bad)), bad.shape)
        raise ValueError("bmstp.atlas: pixel %d's %s limit or width is NaN in %s -- an "
                          "unsurveyed band must be +inf/0, never NaN (catalog.depth_grid)"
                          % (int(pix[p_idx]), BAND_KEYS[b_idx], path))
    return pix, f_lim, w_dex_pix


def _coverage(config, region, pix):
    """The catalogue's own IRAC footprint at each admitted pixel (sec.
    3.3's "coverage", sec. 8's catalogue definition): `catalog.coverage`'s
    `FRAC`, the fraction of the pixel's 16 nside-2048 children holding a
    catalogued source with a measured IRAC flux -- replaces
    `sky.derived.coverage`'s Spitzer field-mask union, whose native masks
    are missing mosaics outright for Pipe and Auriga-California
    (`studies/count_discrepancy.md`, cause 1). `catalog.coverage` writes
    one row per admitted pixel, the same axis as `pix` (`catalog.
    depth_grid`'s own), so this is a direct read, not a join."""
    path = config_module.product_path(config, "catalog", "sesna", "coverage", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "bmstp.atlas: no catalogue coverage for %s at %s -- run the "
            "'catalog.coverage' RUNBOOKtp.sh line first" % (region, path))
    with h5py.File(path, "r") as f:
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)
    order = np.argsort(cov_pix)
    loc = np.searchsorted(cov_pix[order], pix)
    loc = np.minimum(loc, cov_pix.size - 1)
    hit = order[loc]
    found = cov_pix[hit] == pix
    out = np.zeros(pix.size, dtype=np.float64)
    out[found] = frac[hit[found]]
    return out


def _pixel_column(config, pix):
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
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)
        prov = np.asarray(f["PROVENANCE"][:])
    order = np.argsort(sl_pix)
    loc = np.searchsorted(sl_pix[order], parent256)
    loc = np.minimum(loc, sl_pix.size - 1)
    hit = order[loc]
    found = sl_pix[hit] == parent256
    if not np.all(found):
        raise ValueError("bmstp.atlas: %d admitted pixel(s) have no sightline column in %s"
                          % (int(np.sum(~found)), path))
    return a_k[hit], prov[hit]


def _pixel_gas_column(config, pix):
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
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)
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


def _pixel_batch_size(n_mem):
    """Pixels per batch so `n_pixel_batch * n_mem * N_BANDS * 8 bytes
    (float64) * _N_TEMP_ARRAYS` stays under `_PIXEL_BATCH_BUDGET_BYTES`
    per worker, independent of how many pixels the caller (a tile, a
    sightline, or GAL's whole region) holds -- CODING_RULES_BMSTP.md rule
    10b, one worker's own 512 MB working set."""
    row_bytes = n_mem * N_BANDS * 8 * _N_TEMP_ARRAYS
    return max(1, _PIXEL_BATCH_BUDGET_BYTES // row_bytes)


def catalogued_probability(a_col, u, flux0, f_lim, width_dex, config):
    """`(p_cat, p_cat_i2, flux_i2)` for ONE pixel batch: the closed-form
    probability that a member is catalogued (SPEC_BMSTP_DRAFT.md sec. 8,
    sec. 6.2) -- `catalogued_fraction` forms the weighted means over these
    three arrays, batch by batch.

    `a_col` (n_pix,) the pixel's own column, `u` (n_mem,) the member's own
    placement fraction, `flux0` (n_mem, 8) undimmed member fluxes (zero
    where the member carries no flux in a band), `f_lim` (n_pix, 8) the
    pixel's own marginalised limits, `width_dex` (n_pix, 8)
    `catalog.depth_grid`'s `W_DEX_PIX` (sec. 3.3): one value per band for
    the region, broadcast across the pixel axis passed in here -- unlike
    `f_lim`, it does not vary pixel to pixel. `a = a_col * u` (sec. 8's
    "its own extinction"), dimmed through the blended law
    (`population.selection.kappa_hybrid`). The two-of-eight step test is
    replaced by the member's probability of being catalogued: per band
    `p_i = C_i(f_i) = 1 - exp(ln[1-C_i])`, `ln[1-C_i]` from
    `fittp.likelihood`'s own numerically stable erf kernel at
    `z_i = (log10 f_i - log10 F_lim,50,i) / (sqrt(2) w_{r,i})` -- the same
    completeness `fittp` prices non-detection with -- zero where the
    member carries no flux in that band (GAL's 2MASS/24um, H2S's
    J/H/24um).

    `p_cat` (n_pix, n_mem): `P(>=2 of 8) = 1 - Prod_i(1-p_i)
    - Sum_i p_i Prod_{j!=i}(1-p_j)` (bands independent given the fluxes).

    `p_cat_i2` (n_pix, n_mem) (sec. 9 "the bright-end count ratio"): each
    member's own probability of being catalogued WITH I2 (4.5 um) ITSELF
    MEASURED, `p_I2 * (1 - prod_{i!=I2}(1-p_i))` -- the exact condition
    the catalogue side applies to a catalogued source (`ORIGIN_FNU[:, I2]
    == 1`, `_observed_bright_counts`).

    `flux_i2` (n_pix, n_mem): the dimmed I2 flux, for the bright tests
    (compared there to 3x/10x the pixel's own I2 50% limit).

    An unsurveyed band's `f_lim` is `+inf` and its `width_dex` is 0
    (`catalog.depth_grid`'s rule): `log10(f_lim) = +inf`, so `z`'s
    numerator is `-inf` for every finite member flux regardless of the
    zero width in its denominator (`-inf / 0 = -inf`, never `0 * inf` --
    the numerator is never zero because a member's flux is never
    infinite), and `_ln_one_minus_c(-inf) = 0` (`erfc(+inf) = 0`
    exactly), so `p_i = 0`: the member is never detected in that band,
    with no NaN at any step. `has_flux`'s own `np.where` keeps a
    band with no flux at all out of this path already, so the two guards
    do not interact.

    No batching inside: the caller (`catalogued_fraction`, rule 10b) passes
    one pixel batch and its own slice of
    `a_col`/`f_lim`/`width_dex`; `u` and `flux0` are the caller's whole
    (unsliced) member draw."""
    assert MIN_BANDS_CLEAR == 2, "the closed form below is `P(>=2 of 8)` only"
    a = a_col[:, None] * u[None, :]  # (n_pix, n_mem)
    w_ramp = selection_module.law_dense_weight(a)  # (n_pix, n_mem)
    kappa = selection_module.kappa_hybrid(config, w_ramp)  # (n_pix, n_mem, 8)
    dimming = 0.4 * a[:, :, None] * kappa  # (n_pix, n_mem, 8)
    flux = flux0[None, :, :] * 10.0 ** (-dimming)  # (n_pix, n_mem, 8)
    has_flux = flux0[None, :, :] > 0.0  # (1, n_mem, 8), broadcasts
    log10_f = np.log10(np.where(has_flux, flux, 1.0))
    z = ((log10_f - np.log10(f_lim)[:, None, :])
         / (likelihood_module._SQRT2 * width_dex[:, None, :]))
    p = np.where(has_flux, 1.0 - np.exp(likelihood_module._ln_one_minus_c(z)), 0.0)
    one_minus_p = 1.0 - p  # (n_pix, n_mem, 8)
    prod_all = np.prod(one_minus_p, axis=2)
    sum_term = np.zeros_like(prod_all)
    for i in range(N_BANDS):
        sum_term += p[:, :, i] * np.prod(np.delete(one_minus_p, i, axis=2), axis=2)
    p_cat = 1.0 - prod_all - sum_term

    # the bright-end check (sec. 9): a member counts as bright only if it
    # would be CATALOGUED WITH I2 ITSELF MEASURED (its own I2 cleared,
    # `p[:,:,IDX_I2]`, AND at least one of the other seven also clears,
    # `1 - prod_{i!=I2}(1-p_i)`); flux_i2 lets the caller apply the 3x/10x
    # test against this pixel's own I2 50% limit.
    prob_other_detect = 1.0 - np.prod(np.delete(one_minus_p, IDX_I2, axis=2), axis=2)
    p_cat_i2 = p[:, :, IDX_I2] * prob_other_detect
    flux_i2 = flux[:, :, IDX_I2]  # (n_pix, n_mem)
    return p_cat, p_cat_i2, flux_i2


def catalogued_fraction(a_col, u, flux0, w, f_lim, width_dex, config, weight_pix, tick=None):
    """`(frac, frac_bright3, frac_bright10, s_member)`: the weighted, deterministic
    quadrature over a class's WHOLE population (SPEC_BMSTP_DRAFT.md sec. 8) -- exact
    up to the population's own binning, for every class (GAL first, sec. 5.4). `w`
    (n_mem,) is each member's own population weight, on any positive scale; every
    mean below normalises by `w.sum()`. `catalogued_probability` supplies the
    per-pixel, per-member catalogued probability (`p_cat`) and the I2-measured
    probability and dimmed I2 flux for the bright tests (sec. 9); this function forms
    the weighted sums over the class's whole population.

    `frac` (n_pix,) = `(p_cat @ w) / w.sum()`; `frac_bright3`/`frac_bright10` (sec. 9
    "the bright-end count ratio") the same weighted mean restricted to members whose
    dimmed I2 flux exceeds 3x/10x the pixel's own I2 50% limit -- the exact condition
    the catalogue side applies to a catalogued source. `s_member` (n_mem,) =
    `sum_pix weight_pix[pix] * p_cat[pix, m]`, the pixel-area-weighted catalogued
    probability of that member over the whole region, exact per member (a region
    total built from `density * (w / w.sum()) @ s_member` needs no further error
    term: the quadrature is exact, not a draw).

    Processed in pixel batches of `_pixel_batch_size` (rule 10b): each batch is the
    same elementwise-per-pixel computation on a slice of `a_col`/`f_lim`/`width_dex`,
    so splitting the pixel axis changes no result. Within one pixel batch the member
    axis is ALSO chunked, in
    `n_mem_chunk = _PIXEL_BATCH_BUDGET_BYTES // (N_BANDS * 8 * _N_TEMP_ARRAYS *
    n_pix_batch)` members at a time (AGB's cross-product population, sec. 5.2, can
    reach ~1e6 members per tile: a `(1, 1e6, 8)` working set at one pixel per batch
    would still exceed the 512 MB budget even at the smallest pixel batch, so the
    member axis is chunked too, sized to the batch's own pixel count). Each chunk's
    weighted sum accumulates into `frac`/`frac_bright3`/`frac_bright10` (a sum over
    disjoint member chunks is exact, no result changes for a member set that fit
    before) and fills its own slice of `s_member` directly."""
    n_mem = u.size
    n_pix = a_col.size
    w_sum = float(w.sum())
    frac = np.empty(n_pix, dtype=np.float64)
    frac_bright3 = np.empty(n_pix, dtype=np.float64)
    frac_bright10 = np.empty(n_pix, dtype=np.float64)
    s_member = np.zeros(n_mem, dtype=np.float64)
    batch = _pixel_batch_size(n_mem)
    n_batches = (n_pix + batch - 1) // batch
    for b, start in enumerate(range(0, n_pix, batch)):
        stop = min(start + batch, n_pix)
        n_pix_batch = stop - start
        f_lim_b = f_lim[start:stop]
        width_dex_b = width_dex[start:stop]
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
                a_col_b, u_c, flux0_c, f_lim_b, width_dex_b, config)
            weighted_sum += p_cat @ w_c

            bright3 = flux_i2 > BRIGHT_MULT_3 * i2_lim_b
            bright10 = flux_i2 > BRIGHT_MULT_10 * i2_lim_b
            weighted_bright3 += (catalogued_i2_measured * bright3) @ w_c
            weighted_bright10 += (catalogued_i2_measured * bright10) @ w_c

            s_member[mstart:mstop] += weight_pix[start:stop] @ p_cat
        frac[start:stop] = weighted_sum / w_sum
        frac_bright3[start:stop] = weighted_bright3 / w_sum
        frac_bright10[start:stop] = weighted_bright10 / w_sum
        if tick is not None:
            tick(b + 1, n_batches)
    return frac, frac_bright3, frac_bright10, s_member


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


def _collapse_star_members(u, flux0, w_star, w_star_only, w_pahc_only):
    """Collapse the tile's field-star members into cells before
    `catalogued_fraction` (`_build_one_tile`, rule 9): a member enters
    `catalogued_probability` only through its placement fraction `u`, its
    4.5 um flux `F_4.5 = flux0[:, IDX_I2]`, and its seven flux ratios to
    I2 (every other band). Two stars that agree in those nine quantities
    to within a cell width far below the completeness roll-off
    (`W_DEX` ~= 0.1-0.3 dex, `catalog.depth_grid`) are one member:
    `log10 u` and `log10 F_4.5` at the cell grid's OWN bin widths
    (`grid.LOG10_X_EDGES`' 1/32 dex, `grid.D_LOG10_F45`'s 0.1 dex -- the
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

    Returns the cell's `u`/`flux0` (the ORIGINAL per-star weight
    `w_star`-weighted mean within the cell -- each star's own population
    weight, before the STAR/PAHC split) and the cell's summed STAR-only
    and PAHC-only weights (the SAME cell grouping serves both classes,
    sec. 5.1/5.3, since a star's placement/colour does not depend on
    which class it falls in), plus the member counts before/after for
    the build's own report."""
    n_before = int(u.size)
    tiny = np.finfo(np.float64).tiny
    log10_u = np.log10(np.maximum(u, tiny))
    log10_f45 = np.log10(np.maximum(flux0[:, IDX_I2], tiny))
    x_width = grid.LOG10_X_EDGES[1] - grid.LOG10_X_EDGES[0]
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
    flux0_cell = np.empty((n_after, N_BANDS), dtype=np.float64)
    for k in range(N_BANDS):
        flux0_cell[:, k] = np.bincount(inverse, weights=w_star * flux0[:, k], minlength=n_after) / safe_w
    return u_cell, flux0_cell, w_cell_star_only, w_cell_pahc_only, n_before, n_after


def _build_one_tile(config, region, tile_id, pix_in_tile, a_col_in_tile, f_lim_in_tile, width_dex,
                     agb_pool, coverage_in_tile):
    """One tile's `{cls: (frac, density, frac_bright3, frac_bright10,
    n_cat_cell_tile)}` for STAR/AGB/PAHC, over its own admitted pixels,
    from the star-family population's own retained sample (`population/
    star/population_star_tile__R.hdf5`'s `tile_<id>` group): each class's
    catalogued fraction is the WEIGHTED SUM over every one of its own
    members (`catalogued_fraction`, sec. 8), no draw. STAR's and PAHC's
    members collapse into cells first (`_collapse_star_members`, rule 9:
    `log10 u` and `log10 F_4.5` at the cell grid's own bin widths, each of
    the seven flux ratios to I2 at `STAR_COLOUR_CELL_DEX` -- all far below
    the completeness roll-off, so the collapse moves `catalogued_fraction`
    by much less than the roll-off itself resolves). AGB's members are
    the cross product of `sample_star.sample_agb`'s own evolved stars
    with every shell template of THEIR OWN chemistry (sec. 5.2: the
    star's own `F_4.5` is the shell flux, not TRILEGAL's photosphere),
    each template's eight `F_REF` rescaled so its own 4.5 um reference
    flux equals the star's `F_4.5`. `width_dex` is `catalog.depth_grid`'s
    `W_DEX_PIX` (sec. 3.3): one value per band for the region, broadcast
    to this tile's own pixels (n_pix_in_tile, 8), not a per-pixel fit.
    `n_cat_cell_tile` (128, 110) is this tile's own contribution to the
    region's expected catalogued count per parameter cell (module
    docstring's `N_CAT_CELL_GAL`, the same construction here): `grid.bin`
    on the class's own members at their own `x`/`log10 F_4.5`, weighted
    by each member's own expected catalogued count (`density * (w /
    w.sum()) * s_member`, `s_member` `catalogued_fraction`'s exact
    pixel-area-weighted catalogued probability per member)."""
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
        u = np.asarray(grp["U"][()], dtype=np.float64)
        w_star = np.asarray(grp["W_STAR"][()], dtype=np.float64)
        p_pahc_grid = np.asarray(grp["P_PAHC"][()], dtype=np.float64)  # (n_star, 8)
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
    n_x, n_b = grid.LOG10_X_EDGES.size - 1, grid.LOG10_F45_EDGES.size - 1

    # STAR's and PAHC's members collapse into cells BEFORE `catalogued_fraction`
    # (`_collapse_star_members`, rule 9): the SAME collapsed cells serve both
    # classes, since a star's placement/colour does not depend on which class
    # it falls in, only its STAR-only/PAHC-only weight does.
    u_c, flux0_c, w_star_c, w_pahc_c, n_members_before, n_members_after = (
        _collapse_star_members(u, flux0_all, w_star, w_star_only, w_pahc_only))

    out = {}
    for cls, weight_full, weight_c in (("STAR", w_star_only, w_star_c), ("PAHC", w_pahc_only, w_pahc_c)):
        m = weight_c > 0
        w = weight_c[m]
        density = float(weight_full.sum()) / omega_t  # objects deg^-2, sec. 5.1/5.2's Omega_pointing
        if w.size == 0:
            frac = np.zeros(pix_in_tile.size)
            frac_bright3 = np.zeros(pix_in_tile.size)
            frac_bright10 = np.zeros(pix_in_tile.size)
            n_cat_cell_tile = np.zeros((n_x, n_b))
        else:
            frac, frac_bright3, frac_bright10, s_member = catalogued_fraction(
                a_col_in_tile, u_c[m], flux0_c[m], w, f_lim_in_tile, width_dex, config, weight_pix)
            c_m = density * (w / w.sum()) * s_member
            n_cat_cell_tile = _safe_cell_bin(u_c[m], np.log10(flux0_c[m, IDX_I2]), c_m)
        out[cls] = (frac, density, frac_bright3, frac_bright10, n_cat_cell_tile)

    # AGB (sec. 5.2): every evolved star with positive weight
    # (`sample_star.sample_agb`'s `w_a`; `x_a`'s first half O-rich,
    # second half C-rich) crossed with every shell template of ITS OWN
    # chemistry (`agb_pool[label]`), the cross product built with
    # `np.repeat`/`np.tile` (rule 8: no Python loop over stars). A
    # member's own weight is `w_a[star] * pool["weight"][template] /
    # pool["weight"].sum()` (the within-chemistry template distribution,
    # sec. 5.2's template-weights paragraph); summed over one star's own
    # templates this recovers `w_a[star]` exactly, so `density_agb` still
    # reads the star population's own `w_a.sum()`.
    x_a, f45_a, w_a = sample_star.sample_agb(config, region, tile_id)
    n_evolved = x_a.size // 2
    is_c = np.arange(x_a.size) >= n_evolved
    density_agb = float(w_a.sum()) / omega_t
    u_parts, flux0_parts, w_parts, f45_parts = [], [], [], []
    for label, chem_mask in (("O", ~is_c), ("C", is_c)):
        star_idx = np.flatnonzero(chem_mask & (w_a > 0))
        if star_idx.size == 0:
            continue
        pool = agb_pool[label]
        n_tmpl = pool["weight"].size
        pool_weight_sum = float(pool["weight"].sum())
        star_rep = np.repeat(star_idx, n_tmpl)
        tmpl_rep = np.tile(np.arange(n_tmpl), star_idx.size)

        u_m = x_a[star_rep]
        f45_m = f45_a[star_rep]  # already log10 F_4.5 (mJy)
        f_ref_i2 = np.maximum(pool["f_ref"]["I2"][tmpl_rep], pool["floor_linear"][tmpl_rep])
        scale = (10.0 ** f45_m) / f_ref_i2  # rescales the WHOLE shell SED
        flux0_m = np.empty((star_rep.size, N_BANDS), dtype=np.float64)
        for k, key in enumerate(BAND_KEYS):
            f_band = np.maximum(pool["f_ref"][key][tmpl_rep], pool["floor_linear"][tmpl_rep])
            flux0_m[:, k] = f_band * scale
        w_m = w_a[star_rep] * pool["weight"][tmpl_rep] / pool_weight_sum

        u_parts.append(u_m)
        flux0_parts.append(flux0_m)
        w_parts.append(w_m)
        f45_parts.append(f45_m)

    if not u_parts:
        frac_agb = np.zeros(pix_in_tile.size)
        frac_agb_bright3 = np.zeros(pix_in_tile.size)
        frac_agb_bright10 = np.zeros(pix_in_tile.size)
        n_cat_cell_agb = np.zeros((n_x, n_b))
    else:
        u_agb = np.concatenate(u_parts)
        flux0_agb = np.concatenate(flux0_parts, axis=0)
        w_agb_member = np.concatenate(w_parts)
        f45_agb_member = np.concatenate(f45_parts)
        frac_agb, frac_agb_bright3, frac_agb_bright10, s_member_agb = catalogued_fraction(
            a_col_in_tile, u_agb, flux0_agb, w_agb_member, f_lim_in_tile, width_dex, config, weight_pix)
        w_sum_agb = float(w_agb_member.sum())
        c_m_agb = density_agb * (w_agb_member / w_sum_agb) * s_member_agb
        n_cat_cell_agb = _safe_cell_bin(u_agb, f45_agb_member, c_m_agb)
    out["AGB"] = (frac_agb, density_agb, frac_agb_bright3, frac_agb_bright10, n_cat_cell_agb)
    return out


def _agb_shell_pool(config):
    """`{"O": {...}, "C": {...}}`, the AGB library's own shell-template
    pool by chemistry (sec. 5.2, sec 1.4, owner's ruling 2026-09-09):
    `template_weights.agb_tau_ratio`'s `p_chem(tau) / n_chem(tau)`
    WITHOUT the carbon-fraction admixture (`sample_star.sample_agb`
    already resolves which chemistry a given draw is, so only the
    WITHIN-chemistry template distribution is needed here) -- called
    directly rather than re-derived, the one place the Riebel fit and the
    density-of-templates-in-tau construction live -- plus that
    chemistry's own templates' eight `F_REF` and `FLOOR_LINEAR` (sec.
    3.5's FREFRAW convention). Survey-wide, independent of region -- the
    caller computes this once and reuses it, rather than this function
    caching on an unhashable `Config` (its `inputs` mapping)."""
    reg = template_weights._read_register(config, "agb")
    f_ref, floor_linear = reg["f_ref"], reg["floor_linear"]
    names_r, chem, _log10_tau, ratio = template_weights.agb_tau_ratio(config)
    if names_r.size != reg["names"].size or not np.all(names_r == reg["names"]):
        raise ValueError("bmstp.atlas: agb_tau_ratio's row order disagrees "
                          "with the agb register")

    pools = {}
    for label in ("O", "C"):
        sel = chem == label
        pools[label] = dict(
            f_ref={k: v[sel] for k, v in f_ref.items()},
            floor_linear=floor_linear[sel],
            weight=ratio[sel])
    return pools


def _yso_register(config):
    """The YSO register's own `population` weight
    (`template_weights.yso_population_weight`, sec. 5.5 "Template
    weights", sec. 1.4, owner's ruling 2026-09-09) and eight `F_REF`, in
    the register's own row order -- called directly rather than
    re-derived, the SAME construction `build_yso` uses. Survey-wide,
    independent of region; the caller computes this once and reuses it."""
    reg = template_weights._read_register(config, "yso")
    f_ref, floor_linear = reg["f_ref"], reg["floor_linear"]
    names_w, weight = template_weights.yso_population_weight(config)
    if names_w.size != reg["names"].size or not np.all(names_w == reg["names"]):
        raise ValueError("bmstp.atlas: yso_population_weight's row order "
                          "disagrees with the yso register")
    return dict(weight=weight, f_ref=f_ref, floor_linear=floor_linear)


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


def _yso_region_nodes(config, region, d_front, d_back):
    """`(flux0_ts, w_ts)`, region-wide, built once (sec. 5.5 "Marks"):
    the product of two independent quadratures, each a set of nodes with
    weights summing to one. Templates: `N_YSO_NODES` equally spaced
    quantiles of the register's own census weight (`_yso_register`'s
    `weight`), each carrying weight `1/N_YSO_NODES`, its own eight
    `F_REF` floored at the register's own `FLOOR_LINEAR`. Distance
    shift: `N_YSO_SHIFT_NODES` equally spaced quantiles of the region's
    own shift kernel (`sample_cloud.shift_kernel`), each a node at its
    own cell centre on `grid.LOG10_F45_EDGES` (`_quantile_nodes_from_weight`),
    weight `1/N_YSO_SHIFT_NODES` -- every kernel cell with nonzero mass
    makes the member set too large for a per-sightline build. The product
    set (`np.repeat`/`np.tile`, no Python loop over nodes): `flux0_ts[i,:] =
    flux0_template * 10**delta`, `w_ts = w_template * w_shift`."""
    reg = _yso_register(config)
    weight_t, f_ref, floor_linear = reg["weight"], reg["f_ref"], reg["floor_linear"]
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
    """`(x_centers, weight)`: `N_DEPTH_NODES` equally spaced quantiles of
    the sightline's own `p(x)` (`sample_cloud.sample_x`'s binned return
    on `grid.LOG10_X_EDGES`), each node at its own cell centre
    (`_quantile_nodes_from_weight`), weight `1/N_DEPTH_NODES` -- every
    `p(x)` cell with nonzero mass makes the member set too large for a
    per-sightline build; shared by YSO and H2S (sec. 5.5/5.6, the same
    `p(x)`, no second depth quadrature)."""
    p_x, _mo_x, _removed_frac = sample_cloud.sample_x(loaded_profile, sl_row, d_front, d_back)
    mask = p_x > 0
    width = grid.LOG10_X_EDGES[1] - grid.LOG10_X_EDGES[0]
    centers = grid.LOG10_X_EDGES[:-1] + 0.5 * width
    log10x_nodes = _quantile_nodes_from_weight(centers[mask], p_x[mask], N_DEPTH_NODES)
    x_centers = 10.0 ** log10x_nodes
    return x_centers, np.full(N_DEPTH_NODES, 1.0 / N_DEPTH_NODES)


def _combine_with_depth(flux0_pre, w_pre, x_centers, w_depth):
    """The product set of a region-wide (template x shift, or brightness
    x ratio) node table with one sightline's own depth nodes:
    `(u, flux0, w)`, built with `np.tile`/`np.repeat`, no Python loop over
    nodes."""
    n_pre = flux0_pre.shape[0]
    n_d = x_centers.size
    flux0_full = np.tile(flux0_pre, (n_d, 1))
    w_full = np.tile(w_pre, n_d) * np.repeat(w_depth, n_pre)
    u_full = np.repeat(x_centers, n_pre)
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
        return np.zeros((grid.LOG10_X_EDGES.size - 1, grid.LOG10_F45_EDGES.size - 1), dtype=np.float64)
    H, _out = grid.bin(u, log10_f45, c_m)
    return H * total


def _build_one_sightline(config, region, sl_row, a_col_in_sl, a_col_gas_in_sl, arm_in_sl, f_lim_in_sl,
                          loaded_profile, flux0_ts, w_ts, cloud_frac_sl, d_front, d_back,
                          flux0_br, w_br, width_dex,
                          weight_pix, eta_r, l_of_pix_in_sl):
    """One sightline's YSO and H2S quadratures over the population, no
    draw, shared by every admitted pixel it parents: `(frac_yso,
    density_yso, frac_yso_bright3, frac_yso_bright10, n_cat_cell_yso,
    frac_h2s, frac_h2s_bright3, frac_h2s_bright10, n_cat_cell_h2s,
    n_members_yso, n_members_h2s)` (sec. 9's bright-end check, same
    members).

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

    `width_dex` is `catalog.depth_grid`'s `W_DEX_PIX` (sec. 3.3): one
    value per band for the region, broadcast to this sightline's own
    pixels. `weight_pix` (n_pix_in_sl,) is this sightline's own pixels'
    coverage times pixel area.

    Cell grids (W67c's own construction): `c_m = (w_m/w.sum()) *
    s_member[m]`, `s_member` `catalogued_fraction`'s own pixel-area-
    weighted catalogued probability per member -- for YSO the `weight_pix`
    argument passed to `catalogued_fraction` already carries
    `density_yso` (per pixel, since the young-star law varies pixel to
    pixel even within one sightline), and for H2S it already carries
    `l_of_pix_in_sl * eta_r * eps_ext`, so `s_member` in both cases is
    already density-weighted and no second multiply is needed;
    `grid.bin(u, log10 F_4.5, c_m) * c_m.sum()` with `log10 F_4.5` the
    member's own undimmed `log10 flux0[:, IDX_I2]`. No `rng`, no `seed`:
    every factor here is a deterministic quadrature, so this call induces
    no correlation between sightlines and no error term of its own."""
    a_cloud_in_sl = a_col_gas_in_sl * cloud_frac_sl
    density_yso = yso_module.law_count(config, region, a_cloud_in_sl, arm_in_sl)

    x_centers, w_depth = _depth_nodes(loaded_profile, sl_row, d_front, d_back)

    u_yso, flux0_yso_full, w_yso_full = _combine_with_depth(flux0_ts, w_ts, x_centers, w_depth)
    frac_yso, frac_yso_bright3, frac_yso_bright10, s_member_yso = catalogued_fraction(
        a_col_in_sl, u_yso, flux0_yso_full, w_yso_full, f_lim_in_sl, width_dex, config,
        weight_pix=density_yso * weight_pix)
    c_m_yso = (w_yso_full / w_yso_full.sum()) * s_member_yso
    n_cat_cell_yso = _safe_cell_bin(u_yso, np.log10(flux0_yso_full[:, IDX_I2]), c_m_yso)

    u_h2s, flux0_h2s_full, w_h2s_full = _combine_with_depth(flux0_br, w_br, x_centers, w_depth)
    weight_h2s = l_of_pix_in_sl * eta_r * density_module.EPS_EXT * weight_pix
    frac_h2s, frac_h2s_bright3, frac_h2s_bright10, s_member_h2s = catalogued_fraction(
        a_col_in_sl, u_h2s, flux0_h2s_full, w_h2s_full, f_lim_in_sl, width_dex, config,
        weight_pix=weight_h2s)
    c_m_h2s = (w_h2s_full / w_h2s_full.sum()) * s_member_h2s
    n_cat_cell_h2s = _safe_cell_bin(u_h2s, np.log10(flux0_h2s_full[:, IDX_I2]), c_m_h2s)

    return (frac_yso, density_yso, frac_yso_bright3, frac_yso_bright10, n_cat_cell_yso,
            frac_h2s, frac_h2s_bright3, frac_h2s_bright10, n_cat_cell_h2s,
            int(u_yso.size), int(u_h2s.size))


def _herschel_convolved_law(law_map, law_wcs, pix, arm, density_yso_pix):
    """H2S's sky density (sec. 5.6) is the young-star law convolved by the
    knot-driver kernel on the Herschel arm, the region's convolved-law map
    (`knot_field.convolved_law`, computed ONCE per region by the caller,
    sec. 5.6 "a map operation, once per region") sampled at each pixel's
    own mean position (`mean_over_area`); a Planck pixel, or a Herschel
    pixel the map does not reach, keeps the point law `density_yso_pix`
    already carries. Two callers read this from the SAME deterministic
    `density_yso_pix` (`population.yso.law_count`, no randomness):
    `build_region`'s own H2S error weight, before the sightline quadrature
    runs, and its H2S mean density, after -- one function, one convolved
    map, so the two agree exactly, never a pre-convolution weight beside
    a convolved mean."""
    l_of_pix = density_yso_pix.copy()
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


def _gal_catalogued_fraction(config, a_col, f_lim, width_dex, coverage, tick):
    """`(frac, frac_bright3, frac_bright10, n_cat_cell, density, n_members_before,
    n_members_after)` for GAL's one region-wide DETERMINISTIC quadrature (sec. 8).
    `catalogued_probability` reaches `fittp.likelihood`'s one `@njit(parallel=True)`
    kernel (`_ln_one_minus_c_kernel`, via `_ln_one_minus_c`) -- the atlas's only
    numba call -- and numba's thread pool is not fork-safe: a process that has
    itself started that pool poisons every child a later `os.fork()` creates from
    it -- and joblib respawns idle-expired workers by forking the parent between
    phases. This function is called directly by `build_region`'s own parent frame
    (no wrapping `Parallel` there), but the admitted pixels are split into
    `config.n_jobs` contiguous chunks (`np.array_split`, rule 8's "spread the
    largest iterator") and each chunk's `catalogued_fraction` call -- the one that
    reaches the numba kernel -- is dispatched through `Parallel`, so the kernel
    still only ever runs inside a worker and the parent never starts numba's pool.
    The chunks' `frac`/`frac_bright3`/`frac_bright10` are concatenated in pixel
    order (contiguous chunks keep the axis ordered) and their `s_member` values are
    SUMMED (`catalogued_fraction`'s own docstring: `s_member` is a sum over pixels,
    so splitting the pixel axis and adding the pieces back is exact). The cell grid
    below is formed once from that summed `s_member`, exactly as when one call held
    every pixel.

    `n_cat_cell` (128, 110): the region's expected number of catalogued GAL objects
    per parameter cell (module docstring), the same one-cell smoothing and wall
    fold every stored shape carries (`grid.bin`), so the region grid and the shapes
    share one convention. Each member's own expected catalogued count in the region
    is `c_m = density * (w_m / w.sum()) * s_member[m]` (the population's own weight
    share times its pixel-area-weighted catalogued probability over the region,
    `catalogued_fraction`'s `s_member`); `_safe_cell_bin(u, log10_s, c_m)` guards
    the `c_m.sum() == 0` case the same way as every other class's own cell grid
    (`_safe_cell_bin`'s own docstring). The mass `grid.bin`'s own edges drop is not
    stored (expected ~= 0 for GAL, sec. 5.4's population sitting well inside the
    grid).

    `n_members_before`/`n_members_after` (`_gal_members`'s own return, passed
    through unchanged) are the region's `(node, template)` and `(node, colour
    cell)` member counts, for the build's own report."""
    flux0, u, w, log10_s, density, n_members_before, n_members_after = _gal_members(config)
    n_pix = a_col.size
    n_jobs = max(1, int(config.n_jobs))
    idx_chunks = [c for c in np.array_split(np.arange(n_pix), n_jobs) if c.size]
    n_chunks = len(idx_chunks)
    weight_pix_all = coverage * _HPX512_PIXEL_DEG2

    def _one_chunk(i, idx):
        result = catalogued_fraction(
            a_col[idx], u, flux0, w, f_lim[idx], width_dex[idx], config,
            weight_pix=weight_pix_all[idx])
        if tick is not None:
            tick(i + 1, n_chunks)
        return result

    results = Parallel(n_jobs=n_jobs)(
        delayed(_one_chunk)(i, idx) for i, idx in enumerate(idx_chunks))
    frac = np.concatenate([r[0] for r in results])
    frac_bright3 = np.concatenate([r[1] for r in results])
    frac_bright10 = np.concatenate([r[2] for r in results])
    s_member = sum(r[3] for r in results)

    w_sum = float(w.sum())
    c_m = density * (w / w_sum) * s_member
    n_cat_cell = _safe_cell_bin(u, log10_s, c_m)
    return (frac, frac_bright3, frac_bright10, n_cat_cell, density,
            n_members_before, n_members_after)


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


def _above_fraction(config, reader, cls, grain_of_pix, a_col, f0, d_pahc, curve, cell_weight):
    """`(out, n_cell)`. Per admitted pixel, the deterministic sum
    `Sum_cells h_C(cell; grain) * f_C(F_j; pixel) * 1[F_obs(cell) > F_0]`
    that `N_ABOVE_C(pixel) = A_C(pixel) *` this sum multiplies
    (SPEC_BMSTP_DRAFT.md sec. 8). `h_C` is the class's own RAW stored grain
    shape (`reader.grid_all[grain]`, unit mass on the support -- no
    per-source column-kernel blur: this is a pixel-level sum, not a
    source's own read). `f_C` is 1 for GAL/YSO/AGB/H2S and, for STAR/PAHC,
    the type-times-contamination marginal at the pixel's own 8 um limit
    (`bmstp.template_weights._factor_marginal`, moved out of
    `atlas.shapes` so the atlas and the shapes page share one
    definition). `F_obs(cell) = F_j * 10^(-0.4 kappa_4.5(a) a)`, `a = x_i
    * a_col[pixel]` (the pixel's own extinction column, `_pixel_column`;
    GAL's grid already carries its whole-column delta at `x = 1`, so no
    class-specific case is needed here), `kappa_4.5` the I2 entry of the
    blended law at that extinction (`population.selection.kappa_hybrid`/
    `law_dense_weight`); `F_0 = f0[pixel]`, the pixel's own 4.5 um 50%
    completeness limit. Batched over the pixel axis: each batch holds one
    `(n_pixel_batch, n_x, n_b)` working set (`_above_batch_size`) and, for
    STAR/PAHC, one `(n_pixel_batch, n_model, n_b)` working set for
    `_factor_marginal` (`_above_factor_batch_size`) -- evaluated for the
    BATCH's own pixels only, never the whole admitted-pixel axis at once
    (a per-source-style call over every pixel at once ran 8.6 GB, Orion A
    PAHC, to 27 GB, Cygnus X, since the model axis can run to several
    hundred templates).

    `n_cell` (n_x, n_b): the region's UNTHINNED population per cell
    (sec. 8's intrinsic population, `N_CELL_<C>`) -- `Sum_pix cell_weight[pix]
    * h_C(cell; pixel) * f_C(cell; pixel)`, no flux cut and no dimming (the
    `above`/`a`/`kappa45` terms above never enter it): the same `h_C`/`f_C`
    this function already gathers for `out`, summed over every cell of the
    grid rather than collapsed above one flux limit. `cell_weight[pix]` is
    the caller's own `coverage * pixel area * INTENSITY_C(pixel)`, so
    `n_cell.sum() == Sum_pix cell_weight[pix] * out[pix]` if `out` were
    computed with `f0 = -inf` (no cut) -- one extra weighted sum per pixel
    batch, no new loop over pixels or cells."""
    x_centers = 0.5 * (reader.x_edges[:-1] + reader.x_edges[1:])  # log10 x
    b_centers = 0.5 * (reader.b_edges[:-1] + reader.b_edges[1:])  # log10 F_4.5
    x_lin = 10.0 ** x_centers
    f_j = 10.0 ** b_centers  # mJy
    n_x, n_b = x_centers.size, b_centers.size
    n_pix = a_col.size
    out = np.empty(n_pix, dtype=np.float64)
    n_cell = np.zeros((n_x, n_b), dtype=np.float64)

    is_factor_cls = cls in ("STAR", "PAHC")
    batch = _above_batch_size(n_x, n_b)
    if is_factor_cls:
        batch = min(batch, _above_factor_batch_size(reader.c_theta.size, n_b))
    for start in range(0, n_pix, batch):
        stop = min(start + batch, n_pix)
        h_b = reader.grid_all[grain_of_pix[start:stop]].astype(np.float64)  # (n_p, n_x, n_b)
        a = x_lin[None, :] * a_col[start:stop, None]  # (n_p, n_x)
        w_ramp = selection_module.law_dense_weight(a)
        kappa45 = selection_module.kappa_hybrid(config, w_ramp)[..., IDX_I2]  # (n_p, n_x)
        f_min = f0[start:stop, None] * 10.0 ** (0.4 * kappa45 * a)  # (n_p, n_x): F_j needed to clear F_0
        above = f_j[None, None, :] > f_min[:, :, None]  # (n_p, n_x, n_b)
        if is_factor_cls:
            # STAR/PAHC's own template-weight factor at this batch's own
            # pixels' 8 um limits only, `d_pahc` reshaped to
            # `(n_p, 1, 1)` so `_factor_marginal`'s broadcast adds the
            # pixel axis in front of its own `(n_model, n_b)` term.
            f_c_batch = template_weights._factor_marginal(
                b_centers, reader, cls, d_pahc[start:stop, None, None], curve)  # (n_p, n_b)
            term_cell = h_b * f_c_batch[:, None, :]  # no flux cut, no dimming
            term = term_cell * above
        else:
            term_cell = h_b
            term = term_cell * above
        out[start:stop] = term.sum(axis=(1, 2))
        n_cell += np.tensordot(cell_weight[start:stop], term_cell, axes=(0, 0))
    return out, n_cell


def build_region(config, region):
    """Writes `bmstp/atlas/prior_atlas_hpx512__R.hdf5` for one region: the
    admitted pixel axis (`catalog.depth_grid`), its column and coverage,
    and every class's `N_CAT_*`/`SHARE_*` from its own deterministic
    quadrature above (module docstring).

    `INTENSITY_<C>` (deg^-2, one value per admitted pixel) is the class's
    own intrinsic sky density at that pixel -- the quantity the
    catalogued-density sum (`N_CAT_<C> = INTENSITY_<C> * accepted
    fraction`) starts from before the pixel's own completeness is applied
    (SPEC_BMSTP_DRAFT.md sec. 8): the star-
    family tile's own field-star/evolved-star total per its own
    `OMEGA_POINTING_DEG2` for STAR/AGB/PAHC, the young-star law
    (Herschel-convolved where it reaches) and its H2S-scaled density for
    YSO/H2S, and the counts law's single survey-wide density for GAL. It
    carries no selection and no error attribute of its own, and no page
    draws its own ratio: `N_ABOVE_<C>` (deg^-2)
    is `INTENSITY_<C>` times the deterministic (no draw) fraction of the
    class's own stored grain shape brighter, once dimmed by the pixel's
    own extinction column, than the pixel's own 4.5 um 50% completeness
    limit (`_above_fraction`'s own docstring paragraph); the atlas's
    intrinsic page draws `P(C | pixel, above the limit) = N_ABOVE_C /
    sum(N_ABOVE)`, never `INTENSITY_C`'s own ratio."""
    with progress.Stage("bmstp.atlas", region) as st:
        # the detection probability (sec. 6.2, sec. 3.3's "the depth
        # grid") reads the pixel's own marginalised limit
        # (`F_LIM_50_PIX_MJY`, which does vary pixel to pixel) and the
        # region's own marginalised roll-off width (`W_DEX_PIX`: one
        # value per band for the whole region, stored at every admitted
        # pixel) -- a second, different region-band fit from the one
        # `fittp.sweep` uses.
        pix, f_lim, width_dex = _depth_grid(config, region)
        n_pix = pix.size
        coverage = _coverage(config, region, pix)
        a_col, arm = _pixel_column(config, pix)
        a_col_gas = _pixel_gas_column(config, pix)
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

        # AGB's own shell-template pool (sec. 5.2): survey-wide, computed
        # once here rather than per tile.
        agb_pool = _agb_shell_pool(config)

        # every class is a deterministic quadrature (sec. 8): the region's
        # own catalogued-count-per-cell grid, accumulated tile by tile
        # (STAR/AGB/PAHC), sightline by sightline (YSO/H2S) and once for
        # GAL, below.
        n_cat_cell = {c: np.zeros((grid.LOG10_X_EDGES.size - 1, grid.LOG10_F45_EDGES.size - 1))
                      for c in ("STAR", "PAHC", "AGB", "YSO", "H2S")}

        def _one(tile_id):
            m = usable & (tile_of_pix == tile_id)
            return tile_id, m, _build_one_tile(
                config, region, tile_id, pix[m], a_col[m], f_lim[m], width_dex[m], agb_pool,
                coverage[m])

        results = Parallel(n_jobs=n_jobs)(delayed(_one)(t) for t in tiles_here)
        for i, (tile_id, m, out) in enumerate(results):
            for cls in ("STAR", "AGB", "PAHC"):
                frac, density, frac_bright3, frac_bright10, n_cat_cell_tile = out[cls]
                n_cat[cls][m] = density * frac
                intensity[cls][m] = density
                n_cat_bright3[cls][m] = density * frac_bright3
                n_cat_bright10[cls][m] = density * frac_bright10
                n_cat_cell[cls] += n_cat_cell_tile
            st.tick(i + 1, len(tiles_here), "tiles")

        # GAL, sec. 5.4: one region-wide DETERMINISTIC quadrature (no draw,
        # no seed -- GAL has no tile and no population sample), evaluated
        # at every admitted pixel's own column and limits with the shared
        # `catalogued_fraction`. Called directly (`_gal_catalogued_fraction`'s
        # own docstring): it splits the pixel axis into `config.n_jobs`
        # chunks and dispatches each chunk's `catalogued_fraction` call
        # through `Parallel` itself, so `build_region`'s parent process
        # still never runs a numba kernel, and no child a later fork
        # spawns can inherit a live numba thread pool.
        (frac_gal, frac_gal_bright3, frac_gal_bright10, n_cat_cell_gal, density_gal,
         n_gal_members_before, n_gal_members_after) = _gal_catalogued_fraction(
            config, a_col, f_lim, width_dex, coverage,
            lambda done, total: st.tick(done, total, "GAL pixel chunks"))
        n_cat["GAL"] = density_gal * frac_gal
        # sec. 5.4's own density is one survey-wide constant, so GAL's
        # intensity is that same scalar broadcast to every admitted pixel.
        intensity["GAL"] = np.full(n_pix, density_gal, dtype=np.float64)
        n_cat_bright3["GAL"] = density_gal * frac_gal_bright3
        n_cat_bright10["GAL"] = density_gal * frac_gal_bright10
        # the region's expected number of catalogued objects per parameter
        # cell (module docstring): STAR/PAHC/AGB already accumulated
        # theirs, tile by tile, above.
        n_cat_cell["GAL"] = n_cat_cell_gal

        # YSO/H2S, sec. 5.5-5.6: grouped by the pixel's own nside-256
        # sightline (YSO's grain), one deterministic quadrature per
        # sightline shared by every admitted pixel it parents.
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
        flux0_ts, w_ts = _yso_region_nodes(config, region, d_front, d_back)
        giannini_ratios = h2s_module._load_giannini_ratios(config)
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
        # the young-star law at the pixel's own column EXCEPT on the
        # Herschel arm, where it is the region's convolved law map
        # (`bmstp.knot_field.convolved_law`) averaged over the pixel's
        # own area (item 3: the mean of L over the pixel, not a single
        # nearest sample -- a pixel is much larger than the map's own
        # downsampled grid); a Planck pixel, or a Herschel pixel the
        # convolved map does not reach, keeps the law at its own column.
        # `L` is entirely deterministic (`population.yso.law_count`, no
        # random draw), so it is built here, before the sightline
        # quadrature, from the same `a_col_gas * cloud_frac` every
        # sightline itself computes (`_build_one_sightline`'s own
        # `density_yso`) --
        # `_herschel_convolved_law` is the one function both this weight
        # and H2S's mean density below read, so they never disagree. The
        # map itself is convolved ONCE here (sec. 5.6 "a map operation,
        # once per region") and passed to both calls.
        eta_r = density_module.ETA.get(region, density_module.ETA_ELSEWHERE)
        law_map, law_wcs, knot_meta = knot_field.convolved_law(config, region)
        density_yso_pix_law = yso_module.law_count(
            config, region, a_col_gas * cloud_frac_by_sl[sl_row_of_pix], arm)
        l_of_pix_for_weight = _herschel_convolved_law(law_map, law_wcs, pix, arm, density_yso_pix_law)

        def _one_sl(sl_row):
            m = sl_row_of_pix == sl_row
            (f_y, d_y, fb3_y, fb10_y, ncc_y,
             f_h, fb3_h, fb10_h, ncc_h, n_mem_y, n_mem_h) = _build_one_sightline(
                config, region, sl_row, a_col[m], a_col_gas[m], arm[m], f_lim[m],
                loaded_profile, flux0_ts, w_ts, float(cloud_frac_by_sl[sl_row]), d_front, d_back,
                flux0_br, w_br, width_dex[m],
                coverage[m] * _HPX512_PIXEL_DEG2, eta_r, l_of_pix_for_weight[m])
            return m, f_y, d_y, fb3_y, fb10_y, ncc_y, f_h, fb3_h, fb10_h, ncc_h, n_mem_y, n_mem_h

        frac_h2s_pix = np.full(n_pix, np.nan, dtype=np.float64)
        frac_h2s_bright3_pix = np.full(n_pix, np.nan, dtype=np.float64)
        frac_h2s_bright10_pix = np.full(n_pix, np.nan, dtype=np.float64)
        density_yso_pix = np.full(n_pix, np.nan, dtype=np.float64)
        results_sl = Parallel(n_jobs=n_jobs)(delayed(_one_sl)(r) for r in sls_here)
        # every sightline's own quadrature is deterministic (no `rng`):
        # YSO's shared region-wide template x shift table is a fixed
        # number, not a random sample, so it induces no covariance
        # between sightlines and no error term of its own.
        n_members_yso_list, n_members_h2s_list = [], []
        for i, (m, f_y, d_y, fb3_y, fb10_y, ncc_y, f_h, fb3_h, fb10_h, ncc_h, n_mem_y, n_mem_h) in enumerate(results_sl):
            n_cat["YSO"][m] = d_y * f_y
            intensity["YSO"][m] = d_y
            frac_h2s_pix[m] = f_h
            density_yso_pix[m] = d_y
            n_cat_bright3["YSO"][m] = d_y * fb3_y
            n_cat_bright10["YSO"][m] = d_y * fb10_y
            frac_h2s_bright3_pix[m] = fb3_h
            frac_h2s_bright10_pix[m] = fb10_h
            n_cat_cell["YSO"] += ncc_y
            n_cat_cell["H2S"] += ncc_h
            n_members_yso_list.append(n_mem_y)
            n_members_h2s_list.append(n_mem_h)
            st.tick(i + 1, len(sls_here), "sightlines")
        print(f"bmstp.atlas {region}: YSO members/sightline median={int(np.median(n_members_yso_list))} "
              f"max={int(np.max(n_members_yso_list))}; H2S members/sightline "
              f"median={int(np.median(n_members_h2s_list))} max={int(np.max(n_members_h2s_list))}")

        # this atlas's own INTRINSIC YSO density (before retention),
        # multiplied by P3's own `ON_GRID_YSO` at the pixel's sightline:
        # the SAME retained density P1 reports as `DENSITY_YSO`.
        density_yso_retained_pix = density_yso_pix * on_grid_yso_by_sl_row[sl_row_of_pix]

        l_of_pix = _herschel_convolved_law(law_map, law_wcs, pix, arm, density_yso_pix)
        density_h2s_before = density_yso_pix * eta_r * density_module.EPS_EXT
        density_h2s = l_of_pix * eta_r * density_module.EPS_EXT
        n_cat["H2S"] = density_h2s * frac_h2s_pix
        intensity["H2S"] = density_h2s
        n_cat_h2s_before = density_h2s_before * frac_h2s_pix
        n_cat_bright3["H2S"] = density_h2s * frac_h2s_bright3_pix
        n_cat_bright10["H2S"] = density_h2s * frac_h2s_bright10_pix

        # The intrinsic view above a fixed flux, deterministic, no draw.
        # `N_ABOVE_C(pixel) = A_C(pixel) * _above_fraction(...)`
        # (`_above_fraction`'s own docstring paragraph), read straight off
        # each class's own stored grain shape (`fittp.prior_reader.load`)
        # rather than the catalogued-fraction members above -- the
        # atlas's intrinsic page draws this, not `INTENSITY_C`'s own
        # ratio (SPEC_BMSTP_DRAFT.md sec. 8).
        f0_pix = f_lim[:, IDX_I2]
        d_pahc_pix = -np.log10(f_lim[:, IDX_I4])
        curve = pahc_curve.read(config)
        # `fittp.prior_reader.load`'s `GRID_YSO`/`GRID_H2S` are P3's own
        # array, in P3's own `HPX_PIX_256` row order -- NOT the cloud
        # profile's own order `sl_row_of_pix` indexes (the two sightline
        # axes can differ, which is exactly why `loc_p3` above resolves
        # `on_grid_yso_by_sl_row` the same way): `loc_p3[sl_row_of_pix]`
        # is the pixel's row in P3's own order, the correct grain index
        # into `reader.grid_all` for YSO/H2S.
        p3_row_of_pix = loc_p3[sl_row_of_pix]
        n_above = {}
        # sec. 8's intrinsic population per cell (`N_CELL_<C>`, module
        # docstring): `coverage * pixel area * A_C(pixel)`, the same
        # per-pixel coefficient `N_ABOVE_C` itself scales `frac_above` by
        # below, gathered into cells instead of collapsed above one flux
        # limit (`_above_fraction`'s own docstring paragraph).
        n_cell = {}
        for cls in CLASSES:
            reader = prior_reader.load(config, region, cls)
            if cls in ("STAR", "AGB", "PAHC"):
                grain_of_pix = tile_of_pix  # P2's TILE_ID is dense 0..n_tile-1, tile id == array position
            elif cls in ("YSO", "H2S"):
                grain_of_pix = p3_row_of_pix
            else:  # GAL: one survey-wide grain (`prior_reader.load`'s own convention)
                grain_of_pix = np.zeros(n_pix, dtype=np.int64)
            cell_weight = coverage * _HPX512_PIXEL_DEG2 * intensity[cls]
            frac_above, n_cell[cls] = _above_fraction(
                config, reader, cls, grain_of_pix, a_col, f0_pix, d_pahc_pix, curve, cell_weight)
            n_above[cls] = intensity[cls] * frac_above

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

        # the fixed-seed reproducibility check the run asks for: the
        # region's pre-existing `RATIO_BUILT` header, read before this
        # build overwrites it.
        path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
        old_ratio_built = None
        if os.path.exists(path):
            with h5py.File(path, "r") as fold:
                old_ratio_built = float(fold.attrs["RATIO_BUILT"]) if "RATIO_BUILT" in fold.attrs else None
        print(f"bmstp.atlas {region}: RATIO_BUILT reproduction old={old_ratio_built} new={ratio_built:.6g}")

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
            f.attrs["DENSITY_GAL"] = density_gal
            f.attrs["TOTAL_PREDICTED"] = total_predicted_built
            f.attrs["SURVEYED_AREA_DEG2"] = surveyed_area_deg2
            f.attrs["ADMITTED_AREA_DEG2"] = area_deg2
            f.attrs["TOTAL_OBSERVED"] = float(n_source)
            for c in built:
                f.attrs[f"RATIO_{c}"] = ratio[c]
            f.attrs["RATIO_BUILT"] = ratio_built
            # the YSO population's catalogable fraction (sec 5.5, sec 8):
            # accepted / intrinsic density, never the survey's observed
            # source count.
            f.attrs["CATALOGABLE_FRACTION_YSO"] = catalogable_fraction_yso
            # sec. 9's bright-end check: the same catalogued/observed
            # ratio, above 3x/10x the pixel's own I2 50% limit, where
            # completeness is 1 on both sides.
            f.attrs["RATIO_BRIGHT3"] = ratio_bright3_total
            f.attrs["RATIO_BRIGHT10"] = ratio_bright10_total
            for c in built:
                f.attrs[f"RATIO_BRIGHT3_{c}"] = ratio_bright3[c]
            f.create_dataset("HPX_PIX_512", data=pix)
            f.create_dataset("A_COL_K", data=a_col.astype(np.float32))
            f.create_dataset("COVERAGE", data=coverage.astype(np.float32))
            f.create_dataset("F_LIM_50_PIX_MJY", data=f_lim.astype(np.float32))
            for c in CLASSES:
                f.create_dataset(f"N_CAT_{c}", data=n_cat[c].astype(np.float32))
                f.create_dataset(f"INTENSITY_{c}", data=intensity[c].astype(np.float32))
                f.create_dataset(f"N_ABOVE_{c}", data=n_above[c].astype(np.float32))
                f.create_dataset(f"N_CAT_BRIGHT3_{c}", data=n_cat_bright3[c].astype(np.float32))
                f.create_dataset(f"N_CAT_BRIGHT10_{c}", data=n_cat_bright10[c].astype(np.float32))
            # the region's expected number of catalogued objects per
            # parameter cell (module docstring), summing to `RATIO_<C> *
            # N_source`; `(128, 110)` on `grid.LOG10_X_EDGES` x
            # `grid.LOG10_F45_EDGES`, the shapes' own axes.
            f.create_dataset("N_CAT_CELL_GAL", data=n_cat_cell["GAL"].astype(np.float32))
            f.create_dataset("N_CAT_CELL_STAR", data=n_cat_cell["STAR"].astype(np.float32))
            f.create_dataset("N_CAT_CELL_PAHC", data=n_cat_cell["PAHC"].astype(np.float32))
            f.create_dataset("N_CAT_CELL_AGB", data=n_cat_cell["AGB"].astype(np.float32))
            f.create_dataset("N_CAT_CELL_YSO", data=n_cat_cell["YSO"].astype(np.float32))
            f.create_dataset("N_CAT_CELL_H2S", data=n_cat_cell["H2S"].astype(np.float32))
            # the region's UNTHINNED intrinsic population per parameter
            # cell (module docstring, sec. 8): no flux cut, no dimming --
            # `_above_fraction`'s own `n_cell` return, same axes.
            for c in CLASSES:
                f.create_dataset(f"N_CELL_{c}", data=n_cell[c].astype(np.float32))
            for c in built:
                f.create_dataset(f"SHARE_{c}", data=share[c].astype(np.float32))
            f.create_dataset("N_OBS", data=n_obs.astype(np.int32))
            f.create_dataset("N_OBS_BRIGHT3", data=n_obs_bright3.astype(np.int32))
            f.create_dataset("N_OBS_BRIGHT10", data=n_obs_bright10.astype(np.int32))

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
    5c's per-region product, one file per region)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region(config, region)


if __name__ == "__main__":
    run(build)
