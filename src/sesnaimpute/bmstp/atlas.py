"""P6, the prior atlas (SPEC_BMSTP_DRAFT.md sec. 8; IMPLEMENTATION_BMSTP_DRAFT.md
sec. 1.2 P6, sec. 3 row 1.9).

Per admitted nside-512 pixel, a fixed-seed Monte Carlo sample of each class's
population, dimmed at the pixel's own column through the blended law
(`population.selection.kappa_hybrid`), weighted by each member's probability of
being catalogued (sec. 8, sec. 6.2): per band the completeness `C_i(f)` at the
pixel's own marginalised 50% limit (`catalog.depth_grid`'s `F_LIM_50_PIX_MJY`)
and its own marginalised width (`W_DEX_PIX`), and `P(>=2 of 8)` from the eight bands' independent
non-detection probabilities -- the same completeness model `fittp.likelihood`
prices, never a step at the 50% limit. The selection appears here and nowhere
else in the atlas (sec. 1.2): the prior itself is unthinned.

This build writes STAR/AGB/PAHC (sec. 5.1-5.3, partitioning the field population:
a star is a STAR or a PAHC member of the Monte Carlo, never both, weighted
`W_STAR*(1-P_PAHC)`/`W_STAR*P_PAHC`), GAL (sec. 5.4: SWIRE's four IRAC fluxes
per galaxy, S from the counts law's own node, colours from a galaxy measured at
that node, at `x=1`), YSO (sec. 5.5: the region's own fixed-seed draw of
`N_MC` YSO library templates by the population weight --
`template_weights.yso_population_weight`'s Dunham et al. 2015 census
density over `log10 f_ref,4.5,theta` divided by the library's density of
templates in the same quantity, times inclination uniform in cos i and the
evolutionary-class census (sec 1.4, owner's ruling 2026-09-09), the same
construction `build_yso` uses -- each template's own eight `F_REF` scaled
by `10^delta`, `delta` drawn per member from the region's own shift kernel
(`bmstp.sample_cloud.shift_kernel`), placed along the
sightline's own `p(x)` on the cloud interval, `bmstp.sample_cloud.sample_x`'s
binned return) and H2S (sec. 5.6: the region's 2.12 um lognormal carried into
the bands by the measured knot line-to-band ratios, at YSO's own `x`). AGB's
members are the star-family sampler's own evolved stars (`bmstp.sample_star.
sample_agb`), each carrying one shell template of its own drawn chemistry
(Riebel+2012's optical-depth distribution, `bmstp.template_weights.build_agb`'s
`tau` factor construction) whose eight `F_REF` are scaled so its own 4.5 um
flux equals the star's `F_4.5`. All six classes enter the total-count check.
"""

import os

import h5py
import healpy as hp
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.population import h2s as h2s_module
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

#: Monte Carlo draw size per pixel and class -- SPEC_BMSTP_DRAFT.md sec. 8:
#: "A sample of 1e4 per class holds the Monte Carlo error under 1%."
N_MC = 10_000

#: A fixed seed for the per-tile resample (reproducible, not a science
#: constant): one `RandomState` per tile, seeded by this value plus the
#: tile id, so two runs draw the same members.
MC_SEED = 137

#: Disjoint offsets for the four independent draw streams that share
#: `MC_SEED` (sec. 8's "a fixed-seed sample"): a tile id, a sightline row,
#: the region-wide YSO template pool, and GAL. Each offset is far larger
#: than any real tile id or nside-256 sightline row (nside 256 has under
#: 800,000 pixels total), so no stream's range can reach into another's.
_SEED_OFFSET_TILE = 0
_SEED_OFFSET_SIGHTLINE = 100_000_000
_SEED_OFFSET_YSO_POOL = 200_000_000
_SEED_OFFSET_GAL = 300_000_000

#: Two of eight bands clear -- the survey's own catalogue rule (sec. 1.2),
#: the same constant `population.selection.MIN_BANDS` sets.
MIN_BANDS_CLEAR = selection_module.MIN_BANDS

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


def _draw_members(rng, weight, n_mc):
    """`(idx, total)`: `n_mc` indices into the tile's retained field-star
    arrays, drawn with replacement in proportion to `weight` -- the fixed-
    seed Monte Carlo sample of sec. 8. Each draw stands for `total / n_mc`
    objects, `total = sum(weight)`, so the accepted fraction over the draw
    times `total / OMEGA_POINTING_DEG2` (the tile's own pointing) recovers
    the class's own catalogued density. `None` where the tile carries no
    weight at all (e.g. no evolved star for AGB)."""
    total = float(np.sum(weight, dtype=np.float64))
    if total <= 0.0 or weight.size == 0:
        return None, 0.0
    p = weight / total
    idx = rng.choice(weight.size, size=n_mc, replace=True, p=p)
    return idx, total


#: rule 10b's 512 MB batch budget, for the (n_pixel_batch, N_MC, 8) arrays
#: `_accepted_fraction` holds: `kappa`, `dimming`, `flux`, `log10_f`, `z`,
#: `p`, `one_minus_p` and the per-band `np.delete` term in the closed-form
#: loop are each that shape, and each expression above allocates its own
#: buffer rather than reusing one -- `_N_TEMP_ARRAYS` counts that many
#: same-shape buffers live at once, generously, so the true peak (measured
#: below) sits under the target with margin. `_build_one_tile` runs this
#: inside `config.n_jobs` joblib workers at once (STAR/AGB/PAHC, one tile
#: per worker), each held to this 512 MB working set on its own (rule
#: 10b); the total across all workers is therefore `n_jobs * 512 MB`, not
#: 512 MB in aggregate -- sizing `n_jobs` to the machine's memory is
#: `root.cfg`'s job (rule 10a), not this batch size's.
_PIXEL_BATCH_BUDGET_BYTES = 512 * 1024 * 1024
_N_TEMP_ARRAYS = 12


def _pixel_batch_size(n_mc):
    """Pixels per batch so `n_pixel_batch * N_MC * N_BANDS * 8 bytes
    (float64) * _N_TEMP_ARRAYS` stays under `_PIXEL_BATCH_BUDGET_BYTES`
    per worker, independent of how many pixels the caller (a tile, a
    sightline, or GAL's whole region) holds -- CODING_RULES_BMSTP.md rule
    10b, one worker's own 512 MB working set."""
    row_bytes = n_mc * N_BANDS * 8 * _N_TEMP_ARRAYS
    return max(1, _PIXEL_BATCH_BUDGET_BYTES // row_bytes)


def _accepted_fraction(a_col, u, flux0, f_lim, width_dex, config, tick=None, weight_pix=None):
    """`(frac, mc_error, frac_bright3, frac_bright10, block_total, block_total_se)`
    per pixel: `flux0` (n_mc, 8) undimmed, `u` (n_mc,)
    the member's own placement fraction, `a_col` (n_pix,) the pixel's own
    column, `f_lim` (n_pix, 8) the pixel's own marginalised limits,
    `width_dex` (n_pix, 8) `catalog.depth_grid`'s `W_DEX_PIX` (sec. 3.3):
    one value per band for the region, broadcast across the pixel axis
    passed in here -- unlike `f_lim`, it does not vary pixel to pixel.
    `a = a_col * u`
    (sec. 8's "its own extinction"), dimmed through the blended law
    (`population.selection.kappa_hybrid`). The two-of-eight step test is
    replaced by the member's probability of being catalogued
    (SPEC_BMSTP_DRAFT.md sec. 8, sec. 6.2): per band `p_i = C_i(f_i) =
    1 - exp(ln[1-C_i])`, `ln[1-C_i]` from `fittp.likelihood`'s own
    numerically stable erf kernel at `z_i = (log10 f_i - log10
    F_lim,50,i) / (sqrt(2) w_{r,i})` -- the same completeness `fittp`
    prices non-detection with -- zero where the member carries no flux in
    that band (GAL's 2MASS/24um, H2S's J/H/24um); `P(>=2 of 8) =
    1 - Prod_i(1-p_i) - Sum_i p_i Prod_{j!=i}(1-p_j)` (bands independent
    given the fluxes). `mc_error` is the Monte Carlo standard error of
    THIS PIXEL'S OWN mean catalogued probability at `N_MC` draws (the
    sample standard deviation of `accepted_prob` over members). The same
    `N_MC` members are the caller's one draw for the whole tile,
    sightline or region, so `mc_error` at different pixels of one call is
    not independent -- it does not shrink by summing pixels, and does not
    describe the error on a REGION TOTAL; `weight_pix`/`block_total`/
    `block_total_se` below are what a region total's own error needs.

    `frac_bright3`/`frac_bright10` (sec. 9 "the bright-end count ratio"):
    each member's own two-of-eight catalogued probability `accepted_prob`
    (the SAME quantity `frac` averages), restricted to draws whose dimmed
    I2 (4.5 um) flux exceeds 3x/10x the pixel's own
    `F_LIM_50_PIX_MJY[:, IDX_I2]` -- a member counts as bright only if it
    would be catalogued at all AND is bright on its own dimmed I2, the
    exact condition the catalogue side applies (a catalogued source,
    `ORIGIN_FNU[:, I2] == 1`, bright above the same multiple,
    `_observed_bright_counts`): one bright test, stated once, so the two
    sides of the ratio are the same population. I2 clearance alone would
    admit members that never clear a second band and so could never be
    catalogued, biasing the ratio high.

    `weight_pix` (n_pix,), optional: this pixel's own coefficient (e.g.
    density x coverage x pixel area) by which its `frac` is summed into
    some larger total. When given, `block_total`/`block_total_se` are the
    mean and Monte Carlo standard error of `sum_pix weight_pix *
    accepted_prob`, i.e. of the weighted total THIS CALL'S SHARED DRAW
    OF MEMBERS stands for -- the correct error for a total built from
    this one draw, since it is computed from the N_MC replicate totals
    before they are averaged down to `frac`, rather than from the
    per-pixel `mc_error` values (`None`, `None` when `weight_pix` is not
    given).

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

    Processed in pixel batches of `_pixel_batch_size` (rule 10b): each
    batch is the same elementwise-per-pixel computation on a slice of
    `a_col`/`f_lim`/`width_dex`, so splitting the pixel axis changes no result -- the
    member draws (`u`, `flux0`) are unsliced and shared by every batch.
    Each batch's rows are written straight into the preallocated
    `frac`/`mc_error` outputs; `tick(done, total)` is called once per
    batch when given (only GAL's region-wide call passes one -- a tile or
    a sightline's own pixel count is already small)."""
    assert MIN_BANDS_CLEAR == 2, "the closed form below is `P(>=2 of 8)` only"
    n_mc = u.size
    n_pix = a_col.size
    frac = np.empty(n_pix, dtype=np.float64)
    mc_error = np.empty(n_pix, dtype=np.float64)
    frac_bright3 = np.empty(n_pix, dtype=np.float64)
    frac_bright10 = np.empty(n_pix, dtype=np.float64)
    block_sum = np.zeros(n_mc, dtype=np.float64) if weight_pix is not None else None
    batch = _pixel_batch_size(n_mc)
    n_batches = (n_pix + batch - 1) // batch
    for b, start in enumerate(range(0, n_pix, batch)):
        stop = min(start + batch, n_pix)
        a_b = a_col[start:stop]
        f_lim_b = f_lim[start:stop]
        width_dex_b = width_dex[start:stop]
        a = a_b[:, None] * u[None, :]  # (n_pix_batch, n_mc)
        w_ramp = selection_module.law_dense_weight(a)  # (n_pix_batch, n_mc)
        kappa = selection_module.kappa_hybrid(config, w_ramp)  # (n_pix_batch, n_mc, 8)
        dimming = 0.4 * a[:, :, None] * kappa  # (n_pix_batch, n_mc, 8)
        flux = flux0[None, :, :] * 10.0 ** (-dimming)  # (n_pix_batch, n_mc, 8)
        has_flux = flux0[None, :, :] > 0.0  # (1, n_mc, 8), broadcasts
        log10_f = np.log10(np.where(has_flux, flux, 1.0))
        z = ((log10_f - np.log10(f_lim_b)[:, None, :])
             / (likelihood_module._SQRT2 * width_dex_b[:, None, :]))
        p = np.where(has_flux, 1.0 - np.exp(likelihood_module._ln_one_minus_c(z)), 0.0)
        one_minus_p = 1.0 - p  # (n_pix_batch, n_mc, 8)
        prod_all = np.prod(one_minus_p, axis=2)
        sum_term = np.zeros_like(prod_all)
        for i in range(N_BANDS):
            sum_term += p[:, :, i] * np.prod(np.delete(one_minus_p, i, axis=2), axis=2)
        accepted_prob = 1.0 - prod_all - sum_term
        frac[start:stop] = accepted_prob.mean(axis=1)
        mc_error[start:stop] = accepted_prob.std(axis=1) / np.sqrt(n_mc)
        if block_sum is not None:
            block_sum += weight_pix[start:stop] @ accepted_prob

        # the bright-end check (sec. 9): a member counts as bright only if
        # it would be CATALOGUED (`accepted_prob`, the two-of-eight
        # probability already computed above) AND its own dimmed I2 flux
        # clears 3x/10x this pixel's own I2 50% limit -- brighter than
        # that, completeness is 1 on both sides, so this isolates the
        # level from the faint extrapolation. This is the catalogue
        # side's exact condition (a catalogued, i.e. two-of-eight-accepted,
        # source with `ORIGIN_FNU[:, I2] == 1` and bright I2 flux), not
        # I2's own clearance alone.
        i2_lim_b = f_lim_b[:, IDX_I2:IDX_I2 + 1]  # (n_pix_batch, 1)
        flux_i2 = flux[:, :, IDX_I2]  # (n_pix_batch, n_mc)
        bright3 = flux_i2 > BRIGHT_MULT_3 * i2_lim_b
        bright10 = flux_i2 > BRIGHT_MULT_10 * i2_lim_b
        frac_bright3[start:stop] = (accepted_prob * bright3).mean(axis=1)
        frac_bright10[start:stop] = (accepted_prob * bright10).mean(axis=1)

        if tick is not None:
            tick(b + 1, n_batches)
    if block_sum is not None:
        block_total = float(block_sum.mean())
        block_total_se = float(block_sum.std() / np.sqrt(n_mc))
    else:
        block_total, block_total_se = None, None
    return frac, mc_error, frac_bright3, frac_bright10, block_total, block_total_se


def _pahc_weight(limit8_grid, p_pahc, x):
    """`P_PAHC` interpolated to the flux limit `x` (a scalar mJy, the
    tile's own mean I4 limit -- an approximation of sec. 4's per-pixel
    contrast, disclosed rather than resampled per pixel, since `P_PAHC`
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


def _build_one_tile(config, region, tile_id, pix_in_tile, a_col_in_tile, f_lim_in_tile, width_dex,
                     agb_pool, coverage_in_tile):
    """One tile's `{cls: (frac, mc_error, density, frac_bright3,
    frac_bright10, total_se)}` for STAR/AGB/PAHC, over its own
    admitted pixels, from the star-family population's own retained
    sample (`population/star/population_star_tile__R.hdf5`'s `tile_<id>`
    group): STAR/PAHC draw from the SAME TRILEGAL flux table (sec. 8's
    members list), each with its own weight column and its own Monte
    Carlo resample, so the two densities carry independent binomial noise
    rather than the same draw reweighted after the fact. AGB draws from
    `sample_star.sample_agb`'s own evolved-star sample instead (sec. 5.2:
    the star's own `F_4.5` is the shell flux, not TRILEGAL's photosphere)
    -- each drawn star's own shell chemistry (already resolved by
    `sample_agb`'s carbon-share split, `x`'s first/second half) picks one
    AGB library template from `agb_pool`'s own tau-factor weight within
    that chemistry, whose eight `F_REF` are rescaled so its own 4.5 um
    reference flux equals the star's `F_4.5`. `width_dex` is `catalog.
    depth_grid`'s `W_DEX_PIX` (sec. 3.3): one value per band for the
    region, broadcast to this tile's own pixels (n_pix_in_tile, 8), not a
    per-pixel fit. `total_se` is the Monte Carlo standard error of this
    class's own contribution to the region total (`density` times this
    ONE shared tile draw's weighted total, `coverage_in_tile * pixel
    area`), the error a region total actually carries -- every pixel of
    the tile shares this one draw, so it is not the sum of the per-pixel
    `mc_error` values above."""
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
    # coordinator ruling): a star is EITHER a STAR member or a PAHC member
    # of the Monte Carlo, weighted `W_STAR*(1-P_PAHC)` / `W_STAR*P_PAHC`,
    # so `N_CAT_STAR + N_CAT_PAHC` never exceeds the field-star count.
    w_star_only = w_star * (1.0 - p_pahc)
    w_pahc_only = w_star * p_pahc

    # this tile's one shared draw's own weight for the region-total error
    # (module docstring's `total_se`): density multiplies afterward,
    # since it is a fixed number, not itself drawn.
    weight_pix = coverage_in_tile * _HPX512_PIXEL_DEG2

    rng = np.random.RandomState(MC_SEED + _SEED_OFFSET_TILE + tile_id)
    out = {}
    for cls, weight in (("STAR", w_star_only), ("PAHC", w_pahc_only)):
        idx, total = _draw_members(rng, weight, N_MC)
        density = total / omega_t  # objects deg^-2, sec. 5.1/5.2's Omega_pointing
        if idx is None:
            frac = np.zeros(pix_in_tile.size)
            mc_error = np.zeros(pix_in_tile.size)
            frac_bright3 = np.zeros(pix_in_tile.size)
            frac_bright10 = np.zeros(pix_in_tile.size)
            total_se = 0.0
        else:
            frac, mc_error, frac_bright3, frac_bright10, _block_total, block_total_se = _accepted_fraction(
                a_col_in_tile, u[idx], flux0_all[idx], f_lim_in_tile, width_dex, config,
                weight_pix=weight_pix)
            total_se = density * block_total_se
        out[cls] = (frac, mc_error, density, frac_bright3, frac_bright10, total_se)

    # AGB (sec. 5.2): the SAME sampler `bmstp.shapes` bins its own shape
    # from -- `x_a` is `concatenate([u, u])` over the tile's evolved
    # stars, `f45_a` each star's own shell `log10 F_4.5` in its assigned
    # chemistry, `w_a` the carbon-share-split weight -- so drawing from
    # `w_a` reproduces the O/C admixture exactly as the shape's own draw
    # does; the first half of the concatenation is O-rich, the second C-rich.
    x_a, f45_a, w_a = sample_star.sample_agb(config, region, tile_id)
    idx_agb, total_agb = _draw_members(rng, w_a, N_MC)
    density_agb = total_agb / omega_t
    if idx_agb is None:
        frac_agb = np.zeros(pix_in_tile.size)
        mc_agb = np.zeros(pix_in_tile.size)
        frac_agb_bright3 = np.zeros(pix_in_tile.size)
        frac_agb_bright10 = np.zeros(pix_in_tile.size)
        total_se_agb = 0.0
    else:
        n_evolved = x_a.size // 2
        is_c = idx_agb >= n_evolved
        u_agb = x_a[idx_agb]
        f45_target = 10.0 ** f45_a[idx_agb]  # the star's own shell F_4.5 (mJy)
        flux0_agb = np.empty((idx_agb.size, N_BANDS), dtype=np.float64)
        for label, mask in (("O", ~is_c), ("C", is_c)):
            n_sel = int(np.count_nonzero(mask))
            if n_sel == 0:
                continue
            pool = agb_pool[label]
            shell = rng.choice(pool["weight"].size, size=n_sel, replace=True,
                                p=pool["weight"] / pool["weight"].sum())
            f_ref_i2 = np.maximum(pool["f_ref"]["I2"][shell], pool["floor_linear"][shell])
            scale = f45_target[mask] / f_ref_i2  # rescales the WHOLE shell SED
            for k, key in enumerate(BAND_KEYS):
                f_band = np.maximum(pool["f_ref"][key][shell], pool["floor_linear"][shell])
                flux0_agb[mask, k] = f_band * scale
        frac_agb, mc_agb, frac_agb_bright3, frac_agb_bright10, _block_agb, block_se_agb = _accepted_fraction(
            a_col_in_tile, u_agb, flux0_agb, f_lim_in_tile, width_dex, config,
            weight_pix=weight_pix)
        total_se_agb = density_agb * block_se_agb
    out["AGB"] = (frac_agb, mc_agb, density_agb, frac_agb_bright3, frac_agb_bright10, total_se_agb)
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


def _yso_template_pool(config, region, d_front, d_back, n_mc, seed):
    """`(n_mc, 8)` mJy: `n_mc` YSO library templates drawn by the
    population weight (sec. 5.5 "Template weights"), one fixed-seed draw
    per region (shared by every sightline), each template's own eight
    `F_REF` (floored at the register's own `FLOOR_LINEAR`) scaled by
    `10^delta`, `delta` drawn independently per member from the region's
    own shift kernel `K` (`bmstp.sample_cloud.shift_kernel`:
    a `rng.choice` over `K`'s own cells, jittered uniformly within the
    cell) -- the SAME kernel `template_weights.build_yso`'s conditional
    table reads, so a member's distance placement and the cloud's own
    depth are drawn together, never a single fixed `d_r` scale."""
    reg = _yso_register(config)
    weight, f_ref, floor_linear = reg["weight"], reg["f_ref"], reg["floor_linear"]
    rng = np.random.RandomState(seed)
    idx = rng.choice(weight.size, size=n_mc, replace=True, p=weight / weight.sum())
    kernel, _mo_k = sample_cloud.shift_kernel(config, region, d_front, d_back)
    cell = rng.choice(kernel.size, size=n_mc, replace=True, p=kernel / kernel.sum())
    delta = grid.LOG10_F45_EDGES[cell] + rng.random(n_mc) * grid.D_LOG10_F45
    scale = 10.0 ** delta
    flux0 = np.empty((n_mc, N_BANDS), dtype=np.float64)
    for k, key in enumerate(BAND_KEYS):
        flux0[:, k] = np.maximum(f_ref[key][idx], floor_linear[idx]) * scale
    return flux0


def _draw_x(rng, p_x, n):
    """`n` `x` draws from a sightline's own `p(x)` (sec. 5.5 "Marks",
    `sample_cloud.sample_x`'s binned return on `grid.LOG10_X_EDGES`): a
    grid cell drawn by its own probability, then a uniform position
    within that cell's own `log10 x` width -- the shape's own one-cell
    resolution, no finer information is on offer."""
    p = p_x / p_x.sum()
    cell = rng.choice(p_x.size, size=n, replace=True, p=p)
    lo = grid.LOG10_X_EDGES[cell]
    hi = grid.LOG10_X_EDGES[cell + 1]
    log10_u = lo + rng.random(n) * (hi - lo)
    return 10.0 ** log10_u


def _build_one_sightline(config, region, sl_row, a_col_in_sl, a_col_gas_in_sl, arm_in_sl, f_lim_in_sl,
                          loaded_profile, flux0_yso, cloud_frac_sl, d_front, d_back,
                          logsig_mean, logsig_std, giannini_ratios, width_dex, seed,
                          weight_pix, eta_r, l_of_pix_in_sl):
    """One sightline's YSO and H2S Monte Carlo draws, shared by every
    admitted pixel it parents: `(frac_yso, mc_yso, density_yso, frac_h2s,
    mc_h2s, frac_yso_bright3, frac_yso_bright10, frac_h2s_bright3,
    frac_h2s_bright10, total_se_yso, total_se_h2s)` (sec. 9's bright-end
    check, same draws). YSO (sec.
    5.5): `flux0_yso` is the region's own fixed-seed draw of library
    templates by the population weight, IDENTICAL at every sightline
    (`_yso_template_pool`, computed once by the caller); only the
    placement `x` is drawn here, per sightline, from this sightline's own
    `p(x)` on the cloud interval (`bmstp.sample_cloud.sample_x`'s binned
    return, `_draw_x`); density `population.yso.law_count` on the CLOUD'S
    own share of the GAS column, `a_col_gas_in_sl * cloud_frac_sl`
    (`bmstp.density._cloud_column_fraction`, W26; `a_col_in_sl`, the
    extinction column, is kept for the members' own dimming below,
    never for the law) -- the count check compares intrinsic members
    through the pixel's own completeness below, so no on-grid factor
    enters here (a member below the grid's retention edge is simply
    never accepted, sec. 8).
    H2S (sec. 5.6): 2.12 um surface brightness from the region's own
    `LOGSIG_MEAN`/`LOGSIG_STD` lognormal, carried into Ks
    (`population.h2s.knot_ks_log10_flux`) and the four IRAC bands (a
    Giannini colour-ratio vector drawn per member; J, H, M1 unmeasured,
    zero flux, disclosed), at YSO's own `x` (the same `p(x)`, an
    independent draw); H2S's own density is `density_yso * eta_r *
    eps_ext` (sec. 5.6 "Sky density", the same young-star law density
    scaled by the region's knot rate and extraction fraction), computed
    by the caller from this same `density_yso`, not here.
    `width_dex` is `catalog.depth_grid`'s `W_DEX_PIX` (sec. 3.3): one
    value per band for the region, broadcast to this sightline's own
    pixels (n_pix_in_sl, 8), not a per-pixel fit.

    `weight_pix` (n_pix_in_sl,) is this sightline's own pixels' coverage
    times pixel area, the coefficient the caller sums a class's `density
    * frac` by to build the region total; `total_se_yso`/`total_se_h2s`
    are the resulting Monte Carlo standard error this ONE sightline draw
    contributes to that total (`_accepted_fraction`'s `block_total_se`,
    weighted by `density_yso` for YSO). H2S's own weight is
    `l_of_pix_in_sl * eta_r * eps_ext`: `l_of_pix_in_sl` is the SAME young-star
    law `n_cat["H2S"]` is built from (sec. 5.6) -- the region's Herschel-arm
    convolution where it reaches, `density_yso` (the point law) elsewhere --
    not the point law `density_yso` itself; the standard error scales with
    the mean it is a fraction of, so the weight the caller convolves must
    be the one the density line uses, never the pre-convolution law.

    `flux0_yso` is one region-wide fixed-seed draw, IDENTICAL at every
    sightline: the YSO Monte Carlo error it contributes to a region total is
    correlated across every sightline that shares it, not independent of the
    other sightlines' own draws (`build_region`'s own docstring paragraph
    states how the caller sums this)."""
    rng = np.random.RandomState(seed)

    p_x, _mo_x, _removed_frac = sample_cloud.sample_x(loaded_profile, sl_row, d_front, d_back)
    a_cloud_in_sl = a_col_gas_in_sl * cloud_frac_sl
    density_yso = yso_module.law_count(config, region, a_cloud_in_sl, arm_in_sl)

    u_yso = _draw_x(rng, p_x, N_MC)
    frac_yso, mc_yso, frac_yso_bright3, frac_yso_bright10, _blk_yso, total_se_yso = _accepted_fraction(
        a_col_in_sl, u_yso, flux0_yso, f_lim_in_sl, width_dex, config,
        weight_pix=density_yso * weight_pix)

    log10_sigma = rng.normal(logsig_mean, logsig_std, size=N_MC)
    log10_f_ks = h2s_module.knot_ks_log10_flux(log10_sigma)
    flux0_h2s = np.zeros((N_MC, N_BANDS), dtype=np.float64)
    flux0_h2s[:, BAND_KEYS.index("Ks")] = 10.0 ** log10_f_ks
    for band in h2s_module.IRAC_RATIO_BAND_KEYS:
        table = giannini_ratios[band]
        ratio_draw = table[rng.randint(0, table.size, size=N_MC)]
        flux0_h2s[:, BAND_KEYS.index(band)] = 10.0 ** (log10_f_ks + ratio_draw)
    u_h2s = _draw_x(rng, p_x, N_MC)
    weight_h2s = l_of_pix_in_sl * eta_r * density_module.EPS_EXT * weight_pix
    frac_h2s, mc_h2s, frac_h2s_bright3, frac_h2s_bright10, _blk_h2s, total_se_h2s = _accepted_fraction(
        a_col_in_sl, u_h2s, flux0_h2s, f_lim_in_sl, width_dex, config, weight_pix=weight_h2s)

    return (frac_yso, mc_yso, density_yso, frac_h2s, mc_h2s,
            frac_yso_bright3, frac_yso_bright10, frac_h2s_bright3, frac_h2s_bright10,
            total_se_yso, total_se_h2s)


def _herschel_convolved_law(config, region, pix, arm, density_yso_pix):
    """H2S's sky density (sec. 5.6) is the young-star law convolved by the
    knot-driver kernel on the Herschel arm, the region's convolved-law map
    sampled at each pixel's own mean position (`knot_field.convolved_law`,
    `mean_over_area`); a Planck pixel, or a Herschel pixel the map does not
    reach, keeps the point law `density_yso_pix` already carries. Two
    callers read this from the SAME deterministic `density_yso_pix`
    (`population.yso.law_count`, no randomness): `build_region`'s own H2S
    error weight, before the sightline Monte Carlo runs, and its H2S mean
    density, after -- one function so the two agree exactly, never a
    pre-convolution weight beside a convolved mean."""
    law_map, law_wcs, knot_meta = knot_field.convolved_law(config, region)
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


def _gal_members(config, rng, n_mc):
    """`(flux0, u)`, GAL's Monte Carlo sample (sec. 5.4): `S` drawn from
    the counts law's own tabulated `log10 S` node
    (`bmstp.sample_gal.sample`'s `phi(S).S` weight, the same law
    `bmstp.shapes.build_gal` bins). The atlas's member and the fitter's GAL
    template are ONE population by construction (read audit R13 item 3):
    the SWIRE galaxy this draw's colour comes from is selected exactly as
    `template_weights.build_galz` selects its own colour population --
    `isfinite(COLOUR_I1I2) & isfinite(SIGMA_COLOUR_I1I2) & SIGMA_COLOUR_I1I2
    > 0` (`sky/derived/swire/galaxies_swire_survey.hdf5`'s own `NODE` axis;
    a node with no such galaxy borrows its nearest node that has one),
    never SWIRE's own I3/I4-detected subset (7.9% of the sample, 0.19 dex
    bluer in I1-I2 at the nodes that carry most of the draws -- the
    discrepancy this replaces). The member's eight-band SED is then the
    "galz" register's own template nearest that drawn `COLOUR_I1I2` in
    `log10 F_REF,I1 - log10 F_REF,I2` (the SAME register and colour axis
    `build_galz` weights the fitter's GAL templates on), scaled so its own
    `F_REF,I2` equals the drawn `S` (already I2's own flux) -- the template
    the fitter would evaluate for that colour, never a flux built from
    SWIRE's own I2I3/I2I4 ratios. J, H, Ks, M1 are unmeasured for a galaxy
    and held at zero flux, so the two-of-eight test runs on the four IRAC
    bands only (disclosed). `x = 1`: sec. 5.4's "whole column"."""
    x_law, log10_s_grid, w_law = sample_gal.sample(config)
    node_draw = rng.choice(log10_s_grid.size, size=n_mc, replace=True, p=w_law / w_law.sum())
    s_draw = 10.0 ** log10_s_grid[node_draw]

    gal_path = config_module.product_path(config, "sky/derived", "swire", "galaxies", "survey")
    with h5py.File(gal_path, "r") as f:
        node = np.asarray(f["NODE"][:], dtype=np.int64)
        c12 = np.asarray(f["COLOUR_I1I2"][:], dtype=np.float64)
        sigma_c12 = np.asarray(f["SIGMA_COLOUR_I1I2"][:], dtype=np.float64)
    # the SAME selection `template_weights.build_galz` applies to the
    # colour population it weights the fitter's GAL templates on.
    finite = (node >= 0) & np.isfinite(c12) & np.isfinite(sigma_c12) & (sigma_c12 > 0)
    node, c12 = node[finite], c12[finite]

    order = np.argsort(node, kind="stable")
    counts = np.bincount(node[order], minlength=log10_s_grid.size)
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    node_ids = np.arange(log10_s_grid.size)
    has = counts > 0
    nearest = node_ids.copy()
    if not has.all():
        have_idx = node_ids[has]
        nearest[~has] = have_idx[np.argmin(np.abs(node_ids[~has, None] - have_idx[None, :]), axis=1)]
    src_node = nearest[node_draw]
    within = np.minimum((rng.random(n_mc) * counts[src_node]).astype(np.int64), counts[src_node] - 1)
    gal_row = order[starts[src_node] + within]
    colour_draw = c12[gal_row]

    # the "galz" register template nearest this draw's own colour
    # (`template_weights.build_galz`'s SAME `colour_theta`), vectorised by
    # `searchsorted` on the sorted register axis, no per-member loop.
    reg = template_weights._read_register(config, "galz")
    f_ref, floor_linear = reg["f_ref"], reg["floor_linear"]
    colour_theta = np.log10(f_ref["I1"]) - np.log10(f_ref["I2"])
    reg_order = np.argsort(colour_theta)
    sorted_colour = colour_theta[reg_order]
    j = np.clip(np.searchsorted(sorted_colour, colour_draw), 1, sorted_colour.size - 1)
    lo, hi = j - 1, j
    pick_hi = np.abs(sorted_colour[hi] - colour_draw) < np.abs(colour_draw - sorted_colour[lo])
    tmpl = reg_order[np.where(pick_hi, hi, lo)]

    flux = np.zeros((n_mc, N_BANDS), dtype=np.float64)
    i1, i2, i3, i4 = (BAND_KEYS.index(k) for k in ("I1", "I2", "I3", "I4"))
    scale = s_draw / np.maximum(f_ref["I2"][tmpl], floor_linear[tmpl])
    flux[:, i1] = f_ref["I1"][tmpl] * scale
    flux[:, i2] = s_draw
    flux[:, i3] = f_ref["I3"][tmpl] * scale
    flux[:, i4] = f_ref["I4"][tmpl] * scale
    u = np.ones(n_mc, dtype=np.float64)
    # `A_GAL`, sec. 5.4 "Sky density": the density the Monte Carlo total
    # stands for is `sample_gal.density` (the `ln 10` integral), NOT the
    # shape weight `w_law.sum()` the node-draw probabilities above use.
    return flux, u, sample_gal.density(config)


def _observed_bright_counts(config, region, pix, f_lim):
    """`(n_obs, n_obs_bright3, n_obs_bright10)`, int32 (n_pix,): the
    catalogue's own counterpart to the bright-end check (SPEC_BMSTP_DRAFT.md
    sec. 9 "the bright-end count ratio"). Every SESNA source's own
    nside-512 pixel (`catalog.curated`'s `GAL_L_DEG`/`GAL_B_DEG`, nested,
    `healpy.ang2pix`) is binned against the admitted pixel axis `pix`;
    `n_obs` counts every source there, `n_obs_bright3`/`n_obs_bright10`
    only those with a measured (`ORIGIN_FNU[:, I2] == 1`) I2 flux above
    3x/10x that pixel's own `F_LIM_50_PIX_MJY[:, I2]` -- the same
    threshold `_accepted_fraction` applies to the model's own draws."""
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


def build_region(config, region):
    """Writes `bmstp/atlas/prior_atlas_hpx512__R.hdf5` for one region: the
    admitted pixel axis (`catalog.depth_grid`), its column and coverage,
    and every class's `N_CAT_*`/`SHARE_*` from its own Monte Carlo
    selection above (module docstring)."""
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
        mc_err = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in CLASSES}
        # sec. 9's bright-end check: the same per-class densities as
        # `n_cat`, restricted to the Monte Carlo members whose dimmed I2
        # flux clears 3x/10x the pixel's own I2 50% limit.
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

        # the region total's own Monte Carlo error (sec. 8): each block's
        # draw is shared within its own tile/sightline/region, not per
        # pixel (`_accepted_fraction`'s own docstring). STAR/AGB/PAHC draw
        # independently PER TILE (each tile resamples its own star-family
        # population fresh) and GAL is one single region-wide block, so
        # both add in quadrature, `total_var += se**2`, as they always
        # have. YSO's `flux0_yso` and H2S's own density weight are instead
        # ONE region-wide, deterministic construction shared by every
        # sightline (`_yso_template_pool`, `_herschel_convolved_law`), so a
        # sightline's own fluctuation about that shared construction is
        # correlated, not independent, with every other sightline's -- the
        # YSO block's per-sightline standard errors are summed LINEARLY
        # over all sightlines, and H2S's likewise, each block's linear sum
        # squared once into `total_var` after the sightline loop, never
        # added sightline by sightline in quadrature.
        total_var = 0.0

        def _one(tile_id):
            m = usable & (tile_of_pix == tile_id)
            return tile_id, m, _build_one_tile(
                config, region, tile_id, pix[m], a_col[m], f_lim[m], width_dex[m], agb_pool,
                coverage[m])

        results = Parallel(n_jobs=n_jobs)(delayed(_one)(t) for t in tiles_here)
        for i, (tile_id, m, out) in enumerate(results):
            for cls in ("STAR", "AGB", "PAHC"):
                frac, mc_error, density, frac_bright3, frac_bright10, total_se = out[cls]
                n_cat[cls][m] = density * frac
                mc_err[cls][m] = mc_error
                n_cat_bright3[cls][m] = density * frac_bright3
                n_cat_bright10[cls][m] = density * frac_bright10
                total_var += total_se ** 2
            st.tick(i + 1, len(tiles_here), "tiles")

        # GAL, sec. 5.4: one region-wide Monte Carlo sample (fixed seed,
        # not per tile -- GAL has no tile), evaluated at every admitted
        # pixel's own column and limits with the shared `_accepted_fraction`.
        gal_rng = np.random.RandomState(MC_SEED + _SEED_OFFSET_GAL)
        gal_flux, gal_u, density_gal = _gal_members(config, gal_rng, N_MC)
        frac_gal, mc_gal, frac_gal_bright3, frac_gal_bright10, _blk_gal, se_gal = _accepted_fraction(
            a_col, gal_u, gal_flux, f_lim, width_dex, config,
            tick=lambda done, total: st.tick(done, total, "GAL pixel batches"),
            weight_pix=coverage * _HPX512_PIXEL_DEG2)
        n_cat["GAL"] = density_gal * frac_gal
        mc_err["GAL"] = mc_gal
        n_cat_bright3["GAL"] = density_gal * frac_gal_bright3
        n_cat_bright10["GAL"] = density_gal * frac_gal_bright10
        total_var += (density_gal * se_gal) ** 2

        # YSO/H2S, sec. 5.5-5.6: grouped by the pixel's own nside-256
        # sightline (YSO's grain), one Monte Carlo draw per sightline
        # shared by every admitted pixel it parents.
        reg = regions_module.REGIONS_BY_NAME[region]
        d_r_pc = float(reg.d_r_pc)
        loaded_profile = sample_cloud._region_profile(config, region)
        # the cloud's own share of the column, per sightline
        # (`bmstp.density._cloud_column_fraction`, W26): the intrinsic
        # YSO/H2S density below is the law applied to `a_cloud`, not the
        # sightline's whole adopted column.
        cloud_frac_by_sl, d_front = density_module._cloud_column_fraction(config, region)
        _d_front, d_back = sample_cloud.cloud_interval_pc(config, region)
        # YSO's members: one fixed-seed draw of N_MC library templates by
        # the population weight and the region's own shift kernel (sec.
        # 5.5 "Template weights"), shared by every sightline
        # of the region.
        flux0_yso = _yso_template_pool(config, region, d_front, d_back, N_MC,
                                        MC_SEED + _SEED_OFFSET_YSO_POOL)
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
        # random draw), so it is built here, before the sightline Monte
        # Carlo, from the same `a_col_gas * cloud_frac` every sightline
        # itself computes (`_build_one_sightline`'s own `density_yso`) --
        # `_herschel_convolved_law` is the one function both this weight
        # and H2S's mean density below read, so they never disagree.
        eta_r = density_module.ETA.get(region, density_module.ETA_ELSEWHERE)
        density_yso_pix_law = yso_module.law_count(
            config, region, a_col_gas * cloud_frac_by_sl[sl_row_of_pix], arm)
        l_of_pix_for_weight = _herschel_convolved_law(config, region, pix, arm, density_yso_pix_law)

        def _one_sl(sl_row):
            m = sl_row_of_pix == sl_row
            (f_y, e_y, d_y, f_h, e_h,
             fb3_y, fb10_y, fb3_h, fb10_h, se_y, se_h) = _build_one_sightline(
                config, region, sl_row, a_col[m], a_col_gas[m], arm[m], f_lim[m],
                loaded_profile, flux0_yso, float(cloud_frac_by_sl[sl_row]), d_front, d_back,
                logsig_mean, logsig_std, giannini_ratios, width_dex[m],
                MC_SEED + _SEED_OFFSET_SIGHTLINE + sl_row,
                coverage[m] * _HPX512_PIXEL_DEG2, eta_r, l_of_pix_for_weight[m])
            return m, f_y, e_y, d_y, f_h, e_h, fb3_y, fb10_y, fb3_h, fb10_h, se_y, se_h

        frac_h2s_pix = np.full(n_pix, np.nan, dtype=np.float64)
        frac_h2s_bright3_pix = np.full(n_pix, np.nan, dtype=np.float64)
        frac_h2s_bright10_pix = np.full(n_pix, np.nan, dtype=np.float64)
        density_yso_pix = np.full(n_pix, np.nan, dtype=np.float64)
        results_sl = Parallel(n_jobs=n_jobs)(delayed(_one_sl)(r) for r in sls_here)
        # rule (1): YSO's `flux0_yso` and H2S's own weight (above) are each
        # ONE region-wide, deterministic construction shared by every
        # sightline, so the Monte Carlo noise about that shared
        # construction is correlated across sightlines, not independent --
        # each block's per-sightline standard errors are summed LINEARLY
        # here and squared once, below, into the region total's variance
        # (never sightline-by-sightline in quadrature, which is only
        # correct for independent draws, e.g. the tile-drawn star families).
        se_yso_sum = 0.0
        se_h2s_sum = 0.0
        for i, (m, f_y, e_y, d_y, f_h, e_h, fb3_y, fb10_y, fb3_h, fb10_h, se_y, se_h) in enumerate(results_sl):
            n_cat["YSO"][m] = d_y * f_y
            mc_err["YSO"][m] = e_y
            mc_err["H2S"][m] = e_h
            frac_h2s_pix[m] = f_h
            density_yso_pix[m] = d_y
            n_cat_bright3["YSO"][m] = d_y * fb3_y
            n_cat_bright10["YSO"][m] = d_y * fb10_y
            frac_h2s_bright3_pix[m] = fb3_h
            frac_h2s_bright10_pix[m] = fb10_h
            se_yso_sum += se_y
            se_h2s_sum += se_h
            st.tick(i + 1, len(sls_here), "sightlines")
        total_var += se_yso_sum ** 2 + se_h2s_sum ** 2

        # this atlas's own INTRINSIC YSO density (before retention),
        # multiplied by P3's own `ON_GRID_YSO` at the pixel's sightline:
        # the SAME retained density P1 reports as `DENSITY_YSO`.
        density_yso_retained_pix = density_yso_pix * on_grid_yso_by_sl_row[sl_row_of_pix]

        l_of_pix = _herschel_convolved_law(config, region, pix, arm, density_yso_pix)
        density_h2s_before = density_yso_pix * eta_r * density_module.EPS_EXT
        density_h2s = l_of_pix * eta_r * density_module.EPS_EXT
        n_cat["H2S"] = density_h2s * frac_h2s_pix
        n_cat_h2s_before = density_h2s_before * frac_h2s_pix
        n_cat_bright3["H2S"] = density_h2s * frac_h2s_bright3_pix
        n_cat_bright10["H2S"] = density_h2s * frac_h2s_bright10_pix

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
        # 8): the region's own MC-and-detection-weighted accepted share
        # of the RETAINED young-star density, `Sigma(n_cat[YSO] *
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

        # the region total's own Monte Carlo error (sec. 8): every class's
        # draw is shared by a whole tile, sightline, or the region (GAL),
        # not drawn per pixel, so the correct error on `TOTAL_PREDICTED`/
        # `RATIO_BUILT` is the sum of each of those independent draws'
        # own contribution's variance (`total_var`, accumulated above as
        # each block finishes), never a sum over pixels.
        total_predicted_se = float(np.sqrt(total_var))
        ratio_built_se = total_predicted_se / n_source if n_source else float("nan")
        print(f"bmstp.atlas {region}: TOTAL_PREDICTED={total_predicted_built:.6g} "
              f"+/- {total_predicted_se:.6g} (RATIO_BUILT {ratio_built:.6g} +/- {ratio_built_se:.6g}), "
              f"the shared-draw Monte Carlo error on the region total")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "hpx512"
            f.attrs["N_MC"] = N_MC
            f.attrs["DENSITY_GAL"] = density_gal
            f.attrs["TOTAL_PREDICTED"] = total_predicted_built
            f.attrs["SURVEYED_AREA_DEG2"] = surveyed_area_deg2
            f.attrs["ADMITTED_AREA_DEG2"] = area_deg2
            f.attrs["TOTAL_OBSERVED"] = float(n_source)
            for c in built:
                f.attrs[f"RATIO_{c}"] = ratio[c]
            f.attrs["RATIO_BUILT"] = ratio_built
            # sec. 8's total-count check: the shared-draw Monte Carlo
            # error on the two numbers above, summed as independent
            # per-tile/per-sightline/region-wide block variances, not
            # read off any per-pixel `mc_error` (those do not describe a
            # region total, module docstring).
            f.attrs["TOTAL_PREDICTED_MC_ERROR"] = total_predicted_se
            f.attrs["RATIO_BUILT_MC_ERROR"] = ratio_built_se
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
                f.create_dataset(f"N_CAT_BRIGHT3_{c}", data=n_cat_bright3[c].astype(np.float32))
                f.create_dataset(f"N_CAT_BRIGHT10_{c}", data=n_cat_bright10[c].astype(np.float32))
            for c in built:
                f.create_dataset(f"SHARE_{c}", data=share[c].astype(np.float32))
            f.create_dataset("N_OBS", data=n_obs.astype(np.int32))
            f.create_dataset("N_OBS_BRIGHT3", data=n_obs_bright3.astype(np.int32))
            f.create_dataset("N_OBS_BRIGHT10", data=n_obs_bright10.astype(np.int32))

        def _max_mc_err(c):
            # a single pixel's own standard error (this pixel's share of
            # its tile/sightline/region's one shared draw), not a region
            # total's error -- `total_predicted_se`/`ratio_built_se`
            # above are that.
            frac_c = n_cat[c] / (built_total + 1e-300)
            return float(np.nanmax(mc_err[c][frac_c > 0.1])) if np.any(frac_c > 0.1) else 0.0

        max_mc_err = max((_max_mc_err(c) for c in built), default=0.0) if n_pix else 0.0
        st.done(path, n_pix=n_pix, n_tile=len(tiles_here), n_sightline=len(sls_here),
                n_tile_filled=n_tile_filled,
                area_deg2=area_deg2, surveyed_area_deg2=surveyed_area_deg2,
                d_front_pc=d_front, d_back_pc=d_back,
                total_predicted_built=total_predicted_built, total_observed=n_source,
                ratio_star=ratio["STAR"], ratio_agb=ratio["AGB"], ratio_pahc=ratio["PAHC"],
                ratio_gal=ratio["GAL"], ratio_yso=ratio["YSO"], ratio_h2s=ratio["H2S"],
                ratio_built=ratio_built, density_gal_deg2=density_gal,
                mc_error_max_where_frac_gt_0p1=max_mc_err,
                total_predicted_mc_error=total_predicted_se, ratio_built_mc_error=ratio_built_se,
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
