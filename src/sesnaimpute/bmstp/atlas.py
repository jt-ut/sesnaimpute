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
that node, at `x=1`), YSO (sec. 5.5: Chabrier 2003 IMF masses through the BHAC15
1 Myr isochrone -- MIST's own basic isochrone above BHAC15's top mass carries no
band magnitudes, so it supplies only `(L, Teff)` for a bare Rayleigh-Jeans
extrapolation anchored at Ks, disclosed -- placed along the sightline's own
`p(u)`, sec. 5.5 "Population"/"Marks") and H2S (sec. 5.6: the region's 2.12 um
lognormal carried into the bands by the measured knot line-to-band ratios, at
YSO's own `x`). All six classes enter the total-count check.
"""

import os

import h5py
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
from sesnaimpute.population.yso_mass import AGE_1MYR_GYR, _read_mist_1myr_track
from sesnaimpute.bmstp import density as density_module
from sesnaimpute.bmstp import sample_cloud
from sesnaimpute.bmstp import sample_gal
from sesnaimpute.fittp import likelihood as likelihood_module

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)
IDX_I4 = BAND_KEYS.index("I4")
CLASSES = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")

#: Monte Carlo draw size per pixel and class -- SPEC_BMSTP_DRAFT.md sec. 8:
#: "A sample of 1e4 per class holds the Monte Carlo error under 1%."
N_MC = 10_000

#: A fixed seed for the per-tile resample (reproducible, not a science
#: constant): one `RandomState` per tile, seeded by this value plus the
#: tile id, so two runs draw the same members.
MC_SEED = 137

#: Two of eight bands clear -- the survey's own catalogue rule (sec. 1.2),
#: the same constant `population.selection.MIN_BANDS` sets.
MIN_BANDS_CLEAR = selection_module.MIN_BANDS

_HPX512_PIXEL_DEG2 = 41252.96 / (12 * 512 ** 2)


def _depth_grid(config, region):
    """The admitted pixel axis and its marginalised 50% limit and width
    (`catalog.depth_grid`, sec. 3.3, the fixed point of the per-pixel
    `log10 DCOMP90` vs `log10 f` fit, not the brightness-biased median):
    `(pix, f_lim_50_pix_mjy, w_dex_pix)`."""
    path = config_module.product_path(config, "catalog", "sesna", "depth-grid", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        f_lim = np.asarray(f["F_LIM_50_PIX_MJY"][:], dtype=np.float64)
        w_dex_pix = np.asarray(f["W_DEX_PIX"][:], dtype=np.float64)
    return pix, f_lim, w_dex_pix


def _coverage(config, region, pix):
    """The IRAC coverage fraction at each admitted pixel (sec. 3.3's
    "coverage", sec. 8's catalogue definition): the union of the four IRAC
    bands' fraction from `sky.derived.coverage`'s five-band `FRAC` at
    `hpx512` granule (MIPS-24 excluded; it covers far more sky than IRAC
    and the catalogue is IRAC-defined), 0 where a pixel is absent from
    that product (never observed in any Spitzer band). `FRAC` stores each
    band's fraction separately rather than the union itself, so the
    per-band max is the union's lower bound (0.1072 of 0.1076 deg^2 at
    NGC 7129, 5.111 of 5.113 deg^2 at Perseus)."""
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)[:, :4].max(axis=1)
    order = np.argsort(cov_pix)
    loc = np.searchsorted(cov_pix[order], pix)
    loc = np.minimum(loc, cov_pix.size - 1)
    hit = order[loc]
    found = cov_pix[hit] == pix
    out = np.zeros(pix.size, dtype=np.float64)
    out[found] = frac[hit[found]]
    return out


def _pixel_column(config, pix):
    """The pixel's own column and arm, from the sightline it is a child of
    (SPEC_BMSTP_DRAFT.md sec. 8: "placed at the pixel (its column, its
    tile or sightline, its arm)"). Nested HEALPix, confirmed from
    `granules.build`'s own `HPX_PIX_256 = HPX_PIX_512 // 4`: every
    admitted nside-512 pixel's parent nside-256 sightline is `pix // 4`.
    `sky/derived/adopted/column_adopted_sightline.hdf5` (survey-wide, no
    region argument) carries `A_K`/`PROVENANCE` at that granule for every
    source-bearing sightline, so every admitted pixel resolves -- no NaN."""
    parent256 = pix // 4
    path = config_module.product_path(config, "sky/derived", "adopted", "column", "sightline")
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


def _accepted_fraction(a_col, u, flux0, f_lim, width_dex, config, tick=None):
    """`(frac, mc_error)` per pixel: `flux0` (n_mc, 8) undimmed, `u` (n_mc,)
    the member's own placement fraction, `a_col` (n_pix,) the pixel's own
    column, `f_lim` (n_pix, 8) the pixel's own marginalised limits,
    `width_dex` (n_pix, 8) the pixel's own marginalised roll-off width
    (`catalog.depth_grid`'s `W_DEX_PIX`, sec. 3.3). `a = a_col * u`
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
    given the fluxes). `mc_error` is the Monte Carlo standard error of the
    mean catalogued probability at `N_MC` draws (the sample standard
    deviation of `accepted_prob` over members, since each draw is now a
    probability rather than a 0/1 outcome, unlike the retired step test's
    Bernoulli `frac*(1-frac)` form).

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
        if tick is not None:
            tick(b + 1, n_batches)
    return frac, mc_error


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


def _build_one_tile(config, region, tile_id, pix_in_tile, a_col_in_tile, f_lim_in_tile, width_dex):
    """One tile's `(frac_star, frac_agb, frac_pahc, mc_star, mc_agb,
    mc_pahc, density_star, density_agb, density_pahc)`, over its own
    admitted pixels, from the star-family population's own retained
    sample (`population/star/population_star_tile__R.hdf5`'s `tile_<id>`
    group): STAR/AGB/PAHC all draw from the SAME TRILEGAL flux table
    (sec. 8's members list), each with its own weight column and its own
    Monte Carlo resample, so the three densities carry independent
    binomial noise rather than the same draw reweighted after the fact.
    `width_dex` is this tile's own pixels' `W_DEX_PIX` (n_pix_in_tile, 8),
    not one region-band constant (`catalog.depth_grid`, sec. 3.3)."""
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
        w_agb = np.asarray(grp["W_AGB"][()], dtype=np.float64)
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

    rng = np.random.RandomState(MC_SEED + tile_id)
    out = {}
    for cls, weight in (("STAR", w_star_only), ("AGB", w_agb), ("PAHC", w_pahc_only)):
        idx, total = _draw_members(rng, weight, N_MC)
        density = total / omega_t  # objects deg^-2, sec. 5.1/5.2's Omega_pointing
        if idx is None:
            frac = np.zeros(pix_in_tile.size)
            mc_error = np.zeros(pix_in_tile.size)
        else:
            frac, mc_error = _accepted_fraction(
                a_col_in_tile, u[idx], flux0_all[idx], f_lim_in_tile, width_dex, config)
        out[cls] = (frac, mc_error, density)
    return out


_ZP_MJY = {b.key: b.vega_zero_point_jy * 1000.0 for b in definitions.BANDS}
_ZP_MJY_ARR = np.array([_ZP_MJY[k] for k in BAND_KEYS])

#: Chabrier 2003 system IMF (SPEC_BMSTP_DRAFT.md sec. 10 "IMF", sec. 5.5
#: "Population"): lognormal in log10 mass below 1 Msun, a power law above,
#: continuous at the join; sampled between 0.1 and 150 Msun.
IMF_M_C_MSUN = 0.2
IMF_SIGMA_DEX = 0.55
IMF_SLOPE_HIGH = 1.35
IMF_M_LO_MSUN = 0.1
IMF_M_HI_MSUN = 150.0
_IMF_LOG10M_GRID = np.linspace(np.log10(IMF_M_LO_MSUN), np.log10(IMF_M_HI_MSUN), 4001)


def _chabrier_pdf_unnorm(log10m):
    """The Chabrier 2003 system IMF's `dN/d(log10 M)`, unnormalised
    (sec. 5.5 "Population"): a lognormal below 1 Msun (`M_c`, `sigma`),
    a power law of index `-IMF_SLOPE_HIGH` above, matched to the
    lognormal's own value at 1 Msun so the two pieces join continuously."""
    below = np.exp(-(log10m - np.log10(IMF_M_C_MSUN)) ** 2 / (2.0 * IMF_SIGMA_DEX ** 2))
    join = np.exp(-np.log10(IMF_M_C_MSUN) ** 2 / (2.0 * IMF_SIGMA_DEX ** 2))
    above = join * 10.0 ** (-IMF_SLOPE_HIGH * log10m)
    return np.where(log10m <= 0.0, below, above)


_IMF_PDF = _chabrier_pdf_unnorm(_IMF_LOG10M_GRID)
_IMF_CDF = np.concatenate(([0.0], np.cumsum(
    0.5 * (_IMF_PDF[1:] + _IMF_PDF[:-1]) * np.diff(_IMF_LOG10M_GRID))))
_IMF_CDF /= _IMF_CDF[-1]


def _draw_chabrier_mass(rng, n):
    """`n` stellar masses (Msun) drawn from the Chabrier 2003 system IMF
    by inverse-CDF interpolation on a fixed log10-mass grid (sec. 5.5
    "Population": "their masses from the Chabrier 2003 system IMF")."""
    log10m = np.interp(rng.random(n), _IMF_CDF, _IMF_LOG10M_GRID)
    return 10.0 ** log10m


def _bhac15_1myr_block(path, n_expected_min_cols):
    """Every column of the BHAC15 1 Myr age block of `path` (either
    filter file), mass-sorted (sec. 3.5, sec. 10 "isochrone"): the
    author's own `! t (Gyr) =` age headers bracket each block, `!`-comment
    lines and blank lines skipped."""
    rows = []
    age = None
    with open(path) as f:
        for line in f:
            if "t (Gyr)" in line:
                age = float(line.split("=")[1])
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            if age is not None and abs(age - AGE_1MYR_GYR) < 1.0e-6:
                vals = [float(x) for x in stripped.split()]
                if len(vals) >= n_expected_min_cols:
                    rows.append(vals)
    if not rows:
        raise ValueError(f"bmstp.atlas: no {AGE_1MYR_GYR} Gyr block found in {path}")
    return np.array(sorted(rows, key=lambda r: r[0]), dtype=np.float64)


def _isochrone_table(config):
    """`(mass, abs_mag)`, the YSO member SED table (SPEC_BMSTP_DRAFT.md
    sec. 5.5 "Population", sec. 10 "isochrone"): BHAC15's 1 Myr photosphere
    up to its own top mass (2MASS `Mj/Mh/Mk` from `BHAC15_iso.2mass`,
    IRAC1-4/MIPS24=M1 from `BHAC15_iso.SPITZER`, the same mass grid in both
    files), extended above that top by a bare Rayleigh-Jeans law anchored
    at Ks: MIST v1.2's basic isochrone carries `(L, Teff)` but no band
    magnitudes there (disclosed), so `Mk(M) = Mk_top - 2.5 log10([L/Teff^3]
    (M) / [L/Teff^3](M_top))` (Stefan-Boltzmann's `R^2 ~ L/Teff^4` folded
    into the Rayleigh-Jeans `F_nu ~ T R^2`), and every other band's colour
    against Ks is the wavelength-only Rayleigh-Jeans law `F_nu ~ nu^2`."""
    twomass_path = f"{config.data_root}/sky/download/baraffe2015_bhac15/BHAC15_iso.2mass"
    spitzer_path = f"{config.data_root}/sky/download/baraffe2015_bhac15/BHAC15_iso.SPITZER"
    tm = _bhac15_1myr_block(twomass_path, 9)   # M Teff L g R Li Mj Mh Mk
    sp = _bhac15_1myr_block(spitzer_path, 13)  # M Teff L g R Li I1 I2 I3 I4 IRSb IRSr MIPS24 ...
    if tm.shape[0] != sp.shape[0] or not np.allclose(tm[:, 0], sp[:, 0]):
        raise ValueError("bmstp.atlas: BHAC15 2MASS/Spitzer 1 Myr mass grids disagree")

    mass_lo = tm[:, 0]
    abs_mag_lo = np.empty((mass_lo.size, N_BANDS), dtype=np.float64)
    abs_mag_lo[:, BAND_KEYS.index("J")] = tm[:, 6]
    abs_mag_lo[:, BAND_KEYS.index("H")] = tm[:, 7]
    abs_mag_lo[:, BAND_KEYS.index("Ks")] = tm[:, 8]
    abs_mag_lo[:, BAND_KEYS.index("I1")] = sp[:, 6]
    abs_mag_lo[:, BAND_KEYS.index("I2")] = sp[:, 7]
    abs_mag_lo[:, BAND_KEYS.index("I3")] = sp[:, 8]
    abs_mag_lo[:, BAND_KEYS.index("I4")] = sp[:, 9]
    abs_mag_lo[:, BAND_KEYS.index("M1")] = sp[:, 12]

    m_top = float(mass_lo[-1])
    log10l_top = float(tm[-1, 2])
    teff_top = float(tm[-1, 1])
    mk_top = float(abs_mag_lo[-1, BAND_KEYS.index("Ks")])

    mist_mass, mist_log_l, mist_log_teff = _read_mist_1myr_track(config)
    hi = mist_mass > m_top
    mass_hi = mist_mass[hi]
    if mass_hi.size:
        d_log_lt3 = ((mist_log_l[hi] - 3.0 * mist_log_teff[hi])
                     - (log10l_top - 3.0 * np.log10(teff_top)))
        mk_hi = mk_top - 2.5 * d_log_lt3
        abs_mag_hi = np.empty((mass_hi.size, N_BANDS), dtype=np.float64)
        ks_wvl = definitions.BANDS_BY_KEY["Ks"].wvl_um
        ks_zp = definitions.BANDS_BY_KEY["Ks"].vega_zero_point_jy
        for k, band in enumerate(definitions.BANDS):
            wvl_ratio = band.wvl_um / ks_wvl
            zp_ratio = band.vega_zero_point_jy / ks_zp
            abs_mag_hi[:, k] = mk_hi + 5.0 * np.log10(wvl_ratio) + 2.5 * np.log10(zp_ratio)
        mass = np.concatenate([mass_lo, mass_hi])
        abs_mag = np.concatenate([abs_mag_lo, abs_mag_hi], axis=0)
    else:
        mass, abs_mag = mass_lo, abs_mag_lo
    return mass, abs_mag, m_top


def _yso_flux0(mass, mass_grid, abs_mag_grid, d_r_pc):
    """`(n, 8)` apparent mJy flux at the region distance (sec. 5.5):
    each band's isochrone absolute magnitude (`np.interp` on the mass
    grid, no per-star loop), scaled by the inverse-square law from the
    isochrone's own 10 pc to `d_r_pc`."""
    abs_mag = np.empty((mass.size, N_BANDS), dtype=np.float64)
    for k in range(N_BANDS):
        abs_mag[:, k] = np.interp(mass, mass_grid, abs_mag_grid[:, k])
    return _ZP_MJY_ARR[None, :] * 10.0 ** (-0.4 * abs_mag) * (10.0 / d_r_pc) ** 2


def _build_one_sightline(config, region, sl_row, a_col_in_sl, arm_in_sl, f_lim_in_sl,
                          loaded_profile, mass_grid, abs_mag_grid, d_r_pc,
                          logsig_mean, logsig_std, giannini_ratios, width_dex, seed):
    """One sightline's YSO and H2S Monte Carlo draws, shared by every
    admitted pixel it parents: `(frac_yso, mc_yso, density_yso, frac_h2s,
    mc_h2s)`. YSO (sec. 5.5): masses from the Chabrier IMF through the
    isochrone, placed along the sightline's own `p(u)`
    (`bmstp.sample_cloud.sample_yso`'s `(x, log10_b, w)` nodes, resampled
    by their own weight into `N_MC` member placements); density
    `population.yso.law_count`'s `kappa_arm * A_pixel^2 * (d_r*pi/180)^2`.
    H2S (sec. 5.6): 2.12 um surface brightness from the region's own
    `LOGSIG_MEAN`/`LOGSIG_STD` lognormal, carried into Ks
    (`population.h2s.knot_ks_log10_flux`) and the four IRAC bands (a
    Giannini colour-ratio vector drawn per member; J, H, M1 unmeasured,
    zero flux, disclosed), at YSO's own `x`; H2S's own density is
    `density_yso * eta_r * eps_ext` (sec. 5.6 "Sky density", the same
    young-star law density scaled by the region's knot rate and
    extraction fraction), computed by the caller from this same
    `density_yso`, not here.
    `width_dex` is this sightline's own pixels' `W_DEX_PIX` (n_pix_in_sl, 8),
    not one region-band constant."""
    rng = np.random.RandomState(seed)

    mass = _draw_chabrier_mass(rng, N_MC)
    flux0_yso = _yso_flux0(mass, mass_grid, abs_mag_grid, d_r_pc)
    x_nodes, _log10b_nodes, w_nodes = sample_cloud.sample_yso(loaded_profile, sl_row)
    p_nodes = w_nodes / w_nodes.sum()
    u_yso = x_nodes[rng.choice(x_nodes.size, size=N_MC, replace=True, p=p_nodes)]
    frac_yso, mc_yso = _accepted_fraction(a_col_in_sl, u_yso, flux0_yso, f_lim_in_sl, width_dex, config)
    density_yso = yso_module.law_count(config, region, a_col_in_sl, arm_in_sl)

    log10_sigma = rng.normal(logsig_mean, logsig_std, size=N_MC)
    log10_f_ks = h2s_module.knot_ks_log10_flux(log10_sigma)
    flux0_h2s = np.zeros((N_MC, N_BANDS), dtype=np.float64)
    flux0_h2s[:, BAND_KEYS.index("Ks")] = 10.0 ** log10_f_ks
    for band in h2s_module.IRAC_RATIO_BAND_KEYS:
        table = giannini_ratios[band]
        ratio_draw = table[rng.randint(0, table.size, size=N_MC)]
        flux0_h2s[:, BAND_KEYS.index(band)] = 10.0 ** (log10_f_ks + ratio_draw)
    u_h2s = x_nodes[rng.choice(x_nodes.size, size=N_MC, replace=True, p=p_nodes)]
    frac_h2s, mc_h2s = _accepted_fraction(a_col_in_sl, u_h2s, flux0_h2s, f_lim_in_sl, width_dex, config)

    return frac_yso, mc_yso, density_yso, frac_h2s, mc_h2s


def _gal_members(config, rng, n_mc):
    """`(flux0, u)`, GAL's Monte Carlo sample (sec. 5.4): `S` drawn from
    the counts law's own tabulated `log10 S` node
    (`bmstp.sample_gal.sample`'s `phi(S).S` weight, the same law
    `bmstp.shapes.build_gal` bins), the three IRAC colours from a galaxy
    measured at that node (`sky/derived/swire/galaxies_swire_survey.hdf5`'s
    finite-colour subset, its own `NODE` axis; a node with no measured
    galaxy borrows its nearest node that has one) -- "draw S from the law
    and colours from the node's galaxies" (coordinator ruling). The SWIRE
    colours are dex flux ratios, `COLOUR_AB = log10 F_A - log10 F_B`
    (`sky/derived/swire/galaxies_swire_survey.hdf5`'s own `DEFINITION`
    attr), not Vega magnitudes -- I1/I3/I4 come from `S` (already I2's
    own flux) by `F_A = S . 10**c` for `COLOUR_I1I2` and `F_B = S .
    10**(-c)` for `COLOUR_I2I3`/`COLOUR_I2I4`, no zero point and no
    `-0.4` factor. J, H, Ks, M1 are unmeasured for a galaxy and held at
    zero flux, so the two-of-eight test runs on the four IRAC bands only
    (disclosed). `x = 1`: sec. 5.4's "whole column"."""
    x_law, log10_s_grid, w_law = sample_gal.sample(config)
    node_draw = rng.choice(log10_s_grid.size, size=n_mc, replace=True, p=w_law / w_law.sum())
    s_draw = 10.0 ** log10_s_grid[node_draw]

    gal_path = config_module.product_path(config, "sky/derived", "swire", "galaxies", "survey")
    with h5py.File(gal_path, "r") as f:
        node = np.asarray(f["NODE"][:], dtype=np.int64)
        c12 = np.asarray(f["COLOUR_I1I2"][:], dtype=np.float64)
        c23 = np.asarray(f["COLOUR_I2I3"][:], dtype=np.float64)
        c24 = np.asarray(f["COLOUR_I2I4"][:], dtype=np.float64)
    finite = (node >= 0) & np.isfinite(c12) & np.isfinite(c23) & np.isfinite(c24)
    node, c12, c23, c24 = node[finite], c12[finite], c23[finite], c24[finite]

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

    flux = np.zeros((n_mc, N_BANDS), dtype=np.float64)
    i1, i2, i3, i4 = (BAND_KEYS.index(k) for k in ("I1", "I2", "I3", "I4"))
    flux[:, i2] = s_draw
    flux[:, i1] = s_draw * 10.0 ** c12[gal_row]
    flux[:, i3] = s_draw * 10.0 ** (-c23[gal_row])
    flux[:, i4] = s_draw * 10.0 ** (-c24[gal_row])
    u = np.ones(n_mc, dtype=np.float64)
    # `A_GAL`, sec. 5.4 "Sky density": the density the Monte Carlo total
    # stands for is `sample_gal.density` (the `ln 10` integral), NOT the
    # shape weight `w_law.sum()` the node-draw probabilities above use.
    return flux, u, sample_gal.density(config)


def build_region(config, region):
    """Writes `bmstp/atlas/prior_atlas_hpx512__R.hdf5` for one region: the
    admitted pixel axis (`catalog.depth_grid`), its column and coverage,
    and every class's `N_CAT_*`/`SHARE_*` from its own Monte Carlo
    selection above (module docstring)."""
    with progress.Stage("bmstp.atlas", region) as st:
        # the pixel's own marginalised limit and roll-off width the
        # detection probability reads (sec. 6.2, sec. 3.3's "the depth
        # grid"), not the region-band constants `fittp.sweep` uses.
        pix, f_lim, width_dex = _depth_grid(config, region)
        n_pix = pix.size
        coverage = _coverage(config, region, pix)
        a_col, arm = _pixel_column(config, pix)
        tile_of_pix, n_tile_filled = _pixel_tile(config, region, pix)

        n_cat = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in CLASSES}
        mc_err = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in CLASSES}

        star_path = config_module.product_path(
            config, "population", "star", "population", "tile", region=region)
        with h5py.File(star_path, "r") as f:
            tile_ids_present = sorted(int(k.split("_")[1]) for k in f.keys()
                                       if k.startswith("tile_"))

        # every admitted pixel now carries a tile (`_pixel_tile` fills the
        # tile-less ones from their nside-256 parent, sec. 8); `usable`
        # stays as the guard against a tile absent from this star product.
        usable = tile_of_pix >= 0
        tiles_here = sorted(set(int(t) for t in tile_of_pix[usable]) & set(tile_ids_present))

        # worker count is `root.cfg`'s own `[run] n_jobs` (CODING_RULES_BMSTP.md
        # rule 10a): the owner sets it to what the machine's memory allows.
        n_jobs = int(config.n_jobs)

        def _one(tile_id):
            m = usable & (tile_of_pix == tile_id)
            return tile_id, m, _build_one_tile(
                config, region, tile_id, pix[m], a_col[m], f_lim[m], width_dex[m])

        results = Parallel(n_jobs=n_jobs)(delayed(_one)(t) for t in tiles_here)
        for i, (tile_id, m, out) in enumerate(results):
            for cls in ("STAR", "AGB", "PAHC"):
                frac, mc_error, density = out[cls]
                n_cat[cls][m] = density * frac
                mc_err[cls][m] = mc_error
            st.tick(i + 1, len(tiles_here), "tiles")

        # GAL, sec. 5.4: one region-wide Monte Carlo sample (fixed seed,
        # not per tile -- GAL has no tile), evaluated at every admitted
        # pixel's own column and limits with the shared `_accepted_fraction`.
        gal_rng = np.random.RandomState(MC_SEED)
        gal_flux, gal_u, density_gal = _gal_members(config, gal_rng, N_MC)
        frac_gal, mc_gal = _accepted_fraction(
            a_col, gal_u, gal_flux, f_lim, width_dex, config,
            tick=lambda done, total: st.tick(done, total, "GAL pixel batches"))
        n_cat["GAL"] = density_gal * frac_gal
        mc_err["GAL"] = mc_gal

        # YSO/H2S, sec. 5.5-5.6: grouped by the pixel's own nside-256
        # sightline (YSO's grain), one Monte Carlo draw per sightline
        # shared by every admitted pixel it parents.
        reg = regions_module.REGIONS_BY_NAME[region]
        d_r_pc = float(reg.d_r_pc)
        mass_grid, abs_mag_grid, m_top = _isochrone_table(config)
        loaded_profile = sample_cloud._region_profile(config, region)
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

        sl_axis = loaded_profile["hpx_pix_256"]
        order_sl = np.argsort(sl_axis)
        sl_parent = pix // 4
        loc_sl = np.minimum(np.searchsorted(sl_axis[order_sl], sl_parent), sl_axis.size - 1)
        hit_sl = order_sl[loc_sl]
        has_sl = sl_axis[hit_sl] == sl_parent
        sl_row_of_pix = np.where(has_sl, hit_sl, -1)
        sls_here = sorted(set(int(r) for r in sl_row_of_pix[has_sl]))

        def _one_sl(sl_row):
            m = sl_row_of_pix == sl_row
            f_y, e_y, d_y, f_h, e_h = _build_one_sightline(
                config, region, sl_row, a_col[m], arm[m], f_lim[m],
                loaded_profile, mass_grid, abs_mag_grid, d_r_pc,
                logsig_mean, logsig_std, giannini_ratios, width_dex[m],
                MC_SEED + 10_000 + sl_row)
            return m, f_y, e_y, d_y, f_h, e_h

        frac_h2s_pix = np.full(n_pix, np.nan, dtype=np.float64)
        density_yso_pix = np.full(n_pix, np.nan, dtype=np.float64)
        results_sl = Parallel(n_jobs=n_jobs)(delayed(_one_sl)(r) for r in sls_here)
        for i, (m, f_y, e_y, d_y, f_h, e_h) in enumerate(results_sl):
            n_cat["YSO"][m] = d_y * f_y
            mc_err["YSO"][m] = e_y
            mc_err["H2S"][m] = e_h
            frac_h2s_pix[m] = f_h
            density_yso_pix[m] = d_y
            st.tick(i + 1, len(sls_here), "sightlines")

        # H2S, sec. 5.6 "Sky density": `A_H2S = A_YSO . eta_r . eps_ext`
        # at the pixel's own column -- the same young-star law density
        # `density_yso_pix` above, no spatial kernel.
        eta_r = density_module.ETA.get(region, density_module.ETA_ELSEWHERE)
        density_h2s = density_yso_pix * eta_r * density_module.EPS_EXT
        n_cat["H2S"] = density_h2s * frac_h2s_pix

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

        path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
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
            f.create_dataset("HPX_PIX_512", data=pix)
            f.create_dataset("A_COL_K", data=a_col.astype(np.float32))
            f.create_dataset("COVERAGE", data=coverage.astype(np.float32))
            f.create_dataset("F_LIM_50_PIX_MJY", data=f_lim.astype(np.float32))
            for c in CLASSES:
                f.create_dataset(f"N_CAT_{c}", data=n_cat[c].astype(np.float32))
            for c in built:
                f.create_dataset(f"SHARE_{c}", data=share[c].astype(np.float32))

        def _max_mc_err(c):
            frac_c = n_cat[c] / (built_total + 1e-300)
            return float(np.nanmax(mc_err[c][frac_c > 0.1])) if np.any(frac_c > 0.1) else 0.0

        max_mc_err = max((_max_mc_err(c) for c in built), default=0.0) if n_pix else 0.0
        st.done(path, n_pix=n_pix, n_tile=len(tiles_here), n_sightline=len(sls_here),
                n_tile_filled=n_tile_filled,
                area_deg2=area_deg2, surveyed_area_deg2=surveyed_area_deg2,
                isochrone_top_msun=m_top,
                total_predicted_built=total_predicted_built, total_observed=n_source,
                ratio_star=ratio["STAR"], ratio_agb=ratio["AGB"], ratio_pahc=ratio["PAHC"],
                ratio_gal=ratio["GAL"], ratio_yso=ratio["YSO"], ratio_h2s=ratio["H2S"],
                ratio_built=ratio_built, density_gal_deg2=density_gal,
                mc_error_max_where_frac_gt_0p1=max_mc_err)
    return path


def build(config, regions=None):
    """`build(config, regions=None)`: per region, `build_region` (rule
    5c's per-region product, one file per region)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region(config, region)


if __name__ == "__main__":
    run(build)
