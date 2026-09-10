"""The depth grid: the atlas's per-pixel completeness limit and width from
the catalogue's own counts (SPEC_BMSTP_DRAFT.md section 3.3, "the depth
grid"), the object the prior atlas reads at a pixel's position instead of a
per-source map value (SPEC_PRIORS.md section 1.3). One row per ADMITTED
pixel (IMPLEMENTATION_BMSTP_DRAFT.md P6), not merely per occupied one.

Per region, admission is the granule map's own rule (`granules/build.py`):
every nside-512 child of one of the region's source-bearing nside-256
pixels. For an OCCUPIED admitted pixel (carrying at least one catalogued
source), `F_LIM_50_MED_MJY` is the median, over the pixel's own sources, of
each band's 50%-completeness limit -- `catalog.limits.limits`, the one
limits reader every consumer uses, evaluated once for the whole region and
grouped by pixel with a pandas groupby (vectorised: no loop over pixels).
An admitted pixel with no sources of its own (a mosaic-support sibling of an
occupied nside-256 pixel) takes the row of its nearest occupied pixel by
angular distance between pixel centres (a vectorised argmax of the
unit-vector dot product against every occupied pixel, no loop over pixels),
with its own `N_SOURCES = 0` so a consumer can tell a filled row from a
measured one. This is the pixel's median source flux and is kept only as
the diagnostic the study (`studies/star_galaxy_level.md`) compares the fix
against: it biases bright by 0.14-0.64 dex, because within a pixel the
per-source map value `DCOMP90` rises with the source's own flux, so the
median tracks the pixel's median source flux rather than the survey's own
turnover.

The counts fit, per region and Spitzer band, is the one limit every reader
now shares (SPEC_BMSTP_DRAFT.md sec. 1.2, 3.3, 6.2): on the sources of
"complete low-column" admitted pixels -- catalogue coverage above 0.9
(`catalog.coverage`, the same product and read the atlas's own admission
uses) and adopted column below the 25th percentile of
that well-covered set's own column (`sky/derived/adopted/column`), the
selection `studies/star_galaxy_level.md` section 1 uses; a region with
fewer than `MIN_LOW_COLUMN_SOURCES` such sources uses every one of its
sources instead, flagged in `LOW_COLUMN_FALLBACK` -- the catalogue's own
detected counts (`ORIGIN_FNU == 1`) are histogrammed in 0.1-dex bins of
absolute flux and fit with `catalog.depths`' own power-law-times-erf model,
reused rather than re-derived: `y = -log10(f)` plays the role of that
module's map-relative magnitude `m`, so its `_bin_histogram`/`_fit_free`
give back a region's bright-end slope `ALPHA_REGION`, 50%-turnover flux
`F_50_REGION_MJY` and roll-off width `W_REGION_DEX` directly in dex, with no
unit conversion.

A band's own counts are a valid 50%-turnover fit only where that band's
own recovery is what cuts them off. Detection is per band (Gutermuth
2009): a source too faint in one band leaves the SESNA catalogue by the
two-band rule before its flux in any OTHER band is ever histogrammed, so a
band whose faint end is set by a DIFFERENT band's requirement reports that
other band's cutoff as its own 50% limit -- and does so as a CLEAN
roll-off, since a histogram already cut off by another band's requirement
is itself well fit by the same power-law-times-erf form the fit assumes
(the cut-off projected through the catalogue's own colour). The fitted
model's own expected-vs-observed count below its 50% point therefore
cannot detect this: the fit reproduces whatever shape it was handed. A
REPORT-ONLY diagnostic instead reads the recovery curve itself
(`_recovery_curve`): for band `b`, the band `r` most often measured
alongside it supplies a median colour on the doubly-detected sources,
which predicts `b`'s flux for every source with `r` measured; the
fraction of those predicted fluxes that actually carry `b`'s own
detection, binned in 0.1 dex, is `b`'s recovery curve as the catalogue
demonstrates it. A genuine roll-off gives 0.5 at its own fitted `F_50` by
definition; a band whose recovery curve is still at or above
`RECOVERY_FLOOR_FRACTION` (90%) there is measured for nearly all of `r`'s
own catalogued sources at the flux its own fit calls "50% complete",
flagged `LIMIT_KIND = "bound"` (`"fit"` otherwise) -- but the limit
itself is NOT substituted (coordinator's ruling 2026-09-10): for a pair
the two-band rule makes mutually required at the faint end (3.6 and 4.5
um for field stars), a catalogued source carries both by construction,
so the measured fraction of EITHER band among catalogued sources is near
1 whatever its true recovery -- the estimator is a tautology for that
pair and cannot bound either band's recovery. The pair's inclusion is
measured only jointly, and the counts fit of each band is that joint
inclusion projected onto the band; `F_50_REGION_MJY` stays the counts
fit for every band, truncated or not, and `LIMIT_KIND`/the recovery
fraction at `F_50` are carried in the per-source product as a diagnostic
for a later, joint treatment of a coupled pair.

Per source, `F_LIM_50_MJY` is `F_50_REGION_MJY` (the counts fit, unmodified by Part B)
shifted by the offset of the source's own `log10 DCOMP90` from the region
selection's median (every one of the region's sources, not detections
only -- `DCOMP90` is a map value every source carries): `F_LIM_50_MJY =
F_50_REGION_MJY * 10**(log10 DCOMP90 - median_sel log10 DCOMP90)` -- the
recovery map supplies only the spatial pattern, normalised on the same
selection the fit used. This is written as the per-source product
`catalog/sesna/limits_sesna_source__<Region>.hdf5`
(`F_LIM_50_MJY` (n, 8), `W_DEX` (8,), `F_50_REGION_MJY`, `ALPHA_REGION`,
`DCOMP90_REF_LOG10` (8,), `LIMIT_KIND` (8,)), the one product
`catalog.limits.limits` reads. Per admitted pixel, `F_LIM_50_PIX_MJY` is
the pixel median of these per-source limits (a pandas groupby, no loop
over pixels); `F_LIM_50_MED_MJY`, once a separate brightness-biased
median of the OLD per-source rule, is now the identical array -- the two
readers' 0.14-0.64 dex drift (`studies/star_galaxy_level.md`) is closed
by construction. `W_DEX_PIX` is `W_DEX` broadcast to every pixel. A pixel
with no sources of its own takes its nearest occupied pixel's row, the
same fill both columns use. The three 2MASS bands carry no per-source map
and keep the region's constant `F50_2MASS_MJY`/`WIDTH_DEX`
(`catalog.depths`), unchanged, for both products.
"""

import os

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import depths as depths_module
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access as access_module

NSIDE_512 = 512

# Ten pixels for the fixed-seed identity check the brief asks for.
CHECK_SEED = 20260907
N_CHECK_PIXELS = 10

# Coverage floor for a "well-covered" admitted pixel (studies/
# star_galaxy_level.md section 1: "COVERAGE > 0.9, the any-IRAC union").
COVERAGE_FLOOR = 0.9

# The low-column quartile of the well-covered set (same section).
COLUMN_PERCENTILE = 25.0

# Below this many low-column-pixel sources, the region-band counts fit
# falls back to every one of the region's own sources (flagged in the
# product's LOW_COLUMN_FALLBACK attribute).
MIN_LOW_COLUMN_SOURCES = 500

# The recovery curve's own fraction at the band's fitted F_50: a genuine
# roll-off gives 0.5 there by definition, so a curve still at or above
# this fraction there has not thinned on its own account at all -- the
# band is flagged as coupled to its limiting band's requirement, report-
# only (module docstring, Part B).
RECOVERY_FLOOR_FRACTION = 0.9


def _admitted_pixels_512(config, region):
    """Every nside-512 child of the region's own source-bearing nside-256
    pixels -- the granule map's admission rule (`granules/build.py`): a
    pixel/region association, not a per-source column, so read directly
    rather than through `granules.access.per_source`.
    """
    path = config_module.product_path(config, "granules", "sesna", "granule-map", "source")
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["region/REGION"][:]]
        code = f["region/REGION_CODE"][names.index(region)]
        g = f["association/region_healpix256"]
        pix256 = np.asarray(g["HPX_PIX_256"][:], dtype=np.int64)
        region_code = np.asarray(g["REGION_CODE"][:])
        n_source_rows = np.asarray(g["N_SOURCE_ROWS"][:])
    source_pix256 = pix256[(region_code == code) & (n_source_rows > 0)]
    children = (source_pix256[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).reshape(-1)
    return np.unique(children)


def _pixel_medians(hpx_pix_512, source_limits):
    """One region's per-pixel source count and median 50%-completeness
    limit, over every band, grouped by occupied pixel -- a single pandas
    groupby, no loop over pixels.
    """
    band_keys = [b.key for b in definitions.BANDS]
    df = pd.DataFrame(source_limits, columns=band_keys)
    df["HPX_PIX_512"] = hpx_pix_512
    grouped = df.groupby("HPX_PIX_512", sort=True)
    n_sources = grouped.size()
    medians = grouped[band_keys].median()
    return (
        medians.index.to_numpy(dtype=np.int64),
        n_sources.to_numpy(dtype=np.int32),
        medians.to_numpy(dtype=np.float32),
    )


def _raw_source_arrays(config, region):
    """The curated per-source `FNU_MJY`/`DCOMP90_MJY`/`ORIGIN_FNU` (every
    band, the curated file's own order) that the region-band counts fit
    and the pixel shift need, and `catalog.limits.limits` does not expose
    (it returns the already rescaled 50%-completeness limit, not the raw
    map value or flux).
    """
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        fnu = f["FNU_MJY"][:]
        dcomp90 = f["DCOMP90_MJY"][:]
        origin = f["ORIGIN_FNU"][:]
        bands = [b.decode() if isinstance(b, bytes) else b for b in f.attrs["BANDS"]]
    return fnu, dcomp90, origin, bands


def _depth_fit_params(config, region):
    """The region's fitted `WIDTH_DEX` (8 bands) and `F50_2MASS_MJY` (3
    bands) from `catalog.depths` -- the same product `catalog.limits.limits`
    reads, and the 2MASS bands' own constant limit and width.
    """
    path = config_module.product_path(config, "catalog", "sesna", "depths", "region")
    with h5py.File(path, "r") as f:
        regions = [r.decode() if isinstance(r, bytes) else r for r in f["REGION"][:]]
        ridx = regions.index(region)
        width_dex = f["WIDTH_DEX"][ridx, :]
        f50_2mass = f["F50_2MASS_MJY"][ridx, :]
    return width_dex, f50_2mass


def _pixel_coverage(config, region, pix):
    """Each admitted pixel's catalogue coverage fraction: `catalog.
    coverage`'s `FRAC`, the fraction of the pixel's nside-2048 children
    holding a catalogued source with a measured IRAC flux -- the same
    product and read `bmstp.atlas._coverage` performs, not `sky.derived.
    coverage`'s Spitzer field-mask union, whose native masks are missing
    mosaics outright for Pipe and Auriga-California (`catalog.coverage`'s
    own module docstring), 0 where a pixel is absent (never observed).
    """
    path = config_module.product_path(config, "catalog", "sesna", "coverage", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.depth_grid: no catalogue coverage for %s at %s -- run the "
            "'catalog.coverage' RUNBOOKtp.sh line first" % (region, path))
    with h5py.File(path, "r") as f:
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)
    order = np.argsort(cov_pix)
    loc = np.minimum(np.searchsorted(cov_pix[order], pix), max(cov_pix.size - 1, 0))
    hit = order[loc]
    found = cov_pix[hit] == pix if cov_pix.size else np.zeros(pix.size, dtype=bool)
    out = np.zeros(pix.size, dtype=np.float64)
    out[found] = frac[hit[found]]
    return out


def _pixel_column(config, pix):
    """Each admitted pixel's own extinction column, from the nside-256
    sightline it is a child of (survey-wide product, no region argument;
    `granules/build.py`'s own `HPX_PIX_256 = HPX_PIX_512 // 4`) -- the same
    join `bmstp.atlas._pixel_column` performs. The depth grid's low-column
    selection is a starlight (IRAC) selection, so it reads the extinction
    column, not the gas column the young-star law was measured on.
    """
    parent256 = pix // 4
    path = config_module.product_path(config, "sky/derived", "adopted", "extinction", "sightline")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.depth_grid: no extinction sightline column at %s -- run the "
            "'sesnaimpute.sky.derived.column' RUNBOOKtp.sh line first" % path)
    with h5py.File(path, "r") as f:
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)
    order = np.argsort(sl_pix)
    loc = np.minimum(np.searchsorted(sl_pix[order], parent256), sl_pix.size - 1)
    hit = order[loc]
    found = sl_pix[hit] == parent256
    if not np.all(found):
        raise ValueError("catalog.depth_grid: %d admitted pixel(s) have no sightline column in %s"
                          % (int(np.sum(~found)), path))
    return a_k[hit]


def _low_column_selection(config, region, hpx_pix_512, admitted):
    """The region's own "complete low-column" source mask
    (`studies/star_galaxy_level.md` section 1, the region counts fit's
    input): IRAC-union coverage above `COVERAGE_FLOOR` at the source's own
    admitted pixel, and that pixel's own adopted column below the
    `COLUMN_PERCENTILE` of the well-covered set's own column. Falls back to
    every one of the region's sources, flagged, if fewer than
    `MIN_LOW_COLUMN_SOURCES` sources pass. Returns `(mask, n_selected,
    fallback)`.
    """
    pix = np.sort(admitted)
    cov = _pixel_coverage(config, region, pix)
    col = _pixel_column(config, pix)
    well_covered = cov > COVERAGE_FLOOR
    if well_covered.any():
        threshold = float(np.percentile(col[well_covered], COLUMN_PERCENTILE))
        low_column_pix = pix[well_covered & (col < threshold)]
    else:
        low_column_pix = np.empty(0, dtype=np.int64)
    mask = np.isin(hpx_pix_512, low_column_pix)
    n_selected = int(mask.sum())
    if n_selected < MIN_LOW_COLUMN_SOURCES:
        return np.ones(hpx_pix_512.shape[0], dtype=bool), n_selected, True
    return mask, n_selected, False


def _region_band_counts_fit(fnu, origin, curated_bands, selected_mask):
    """Per Spitzer band, the region's own fitted bright-end slope
    `alpha`, 50%-turnover flux `f50` (mJy) and roll-off width `w` (dex)
    from the selected sources' own detected counts (module docstring,
    spec section 3.3): `catalog.depths`' Poisson-profiled
    power-law-times-erf estimator (`_bin_histogram`, `_fit_free`) reused
    directly on `y = -log10(f)` in place of that module's map-relative
    magnitude, in 0.1-dex bins.
    """
    n = len(limits_module.IRAC_MIPS_KEYS)
    alpha = np.full(n, np.nan)
    f50 = np.full(n, np.nan)
    w = np.full(n, np.nan)
    for jk, key in enumerate(limits_module.IRAC_MIPS_KEYS):
        cb = curated_bands.index(key)
        detected = selected_mask & (origin[:, cb] == 1)
        flux = fnu[detected, cb]
        if flux.size == 0:
            continue
        y = -np.log10(flux)
        centers, counts = depths_module._bin_histogram(y)
        if np.count_nonzero(counts > 0) < depths_module.MIN_POPULATED_BINS:
            print("catalog.depth_grid: %s region counts histogram too sparse to fit" % key)
            continue
        res = depths_module._fit_free(centers, counts)
        if not res.success:
            print("catalog.depth_grid: %s region counts fit did not converge" % key)
            continue
        alpha_hat, y50_hat, w_hat = res.x
        alpha[jk] = float(alpha_hat)
        f50[jk] = float(10.0 ** (-y50_hat))
        w[jk] = float(w_hat)
    return alpha, f50, w


def _source_limits(dcomp90, curated_bands, selected_mask, f50_region_mjy, f50_2mass):
    """Every catalogued source's own `F_LIM_50_MJY` (8 bands, module
    docstring, Part A): the region's fitted 50% flux (the counts fit,
    unmodified by Part B) shifted by the source's own `log10 DCOMP90` offset from the region
    selection's own median, for the five Spitzer bands -- the recovery
    map's spatial pattern only, normalised on the same selection the
    counts fit used. The three 2MASS bands carry no per-source map and
    take the region's constant `F50_2MASS_MJY` alike for every source.
    Returns `(f_lim_50_mjy (n, 8), dcomp90_ref_log10 (8,), NaN for 2MASS)`.
    """
    band_keys = [b.key for b in definitions.BANDS]
    n = dcomp90.shape[0]
    f_lim = np.empty((n, len(band_keys)), dtype=np.float64)
    ref_log10 = np.full(len(band_keys), np.nan)
    for jk, key in enumerate(limits_module.IRAC_MIPS_KEYS):
        cb = curated_bands.index(key)
        j = band_keys.index(key)
        log_dcomp = np.log10(dcomp90[:, cb])
        median_sel = float(np.median(log_dcomp[selected_mask]))
        ref_log10[j] = median_sel
        f_lim[:, j] = f50_region_mjy[jk] * 10.0 ** (log_dcomp - median_sel)
    for tk, key in enumerate(limits_module.TWOMASS_KEYS):
        j = band_keys.index(key)
        f_lim[:, j] = f50_2mass[tk]
    return f_lim, ref_log10


def _limiting_band(origin, curated_bands, selected_mask, band_key):
    """The other band most often measured alongside `band_key` within the
    region's own low-column selection (module docstring, Part B): the
    pairing the catalogue affords for a truncated band's recovery curve.
    """
    band_keys = [b.key for b in definitions.BANDS]
    cb = curated_bands.index(band_key)
    best_key, best_n = None, -1
    for key in band_keys:
        if key == band_key:
            continue
        cr = curated_bands.index(key)
        n_both = int(np.count_nonzero(selected_mask & (origin[:, cb] == 1) & (origin[:, cr] == 1)))
        if n_both > best_n:
            best_n, best_key = n_both, key
    return best_key


def _recovery_curve(fnu, origin, curated_bands, selected_mask, band_key, ref_key):
    """The catalogue's own demonstrated recovery curve of `band_key`
    (module docstring, Part B): the median colour `log10(F_b/F_r)` on
    sources with both `band_key` and the limiting band `ref_key` measured
    predicts `F_b` for every source with `ref_key` measured; the fraction
    of those predictions that carry `band_key`'s own detection, binned in
    0.1 dex of the predicted flux, is `band_key`'s recovery curve -- the
    diagnostic a truncated band's OWN counts fit cannot supply, since a
    histogram cut off by another band's requirement is itself well fit by
    a roll-off (the cut-off projected through the catalogue's own
    colour). Returns `(centers_y, frac, n_total)` sorted bright to faint
    (`y = -log10 F`), or `(None, None, None)` if `band_key` and `ref_key`
    are never doubly detected.
    """
    cb = curated_bands.index(band_key)
    cr = curated_bands.index(ref_key)
    both = selected_mask & (origin[:, cb] == 1) & (origin[:, cr] == 1)
    if not np.any(both):
        return None, None, None
    colour = float(np.median(np.log10(fnu[both, cb]) - np.log10(fnu[both, cr])))
    has_r = selected_mask & (origin[:, cr] == 1)
    f_b_pred = fnu[has_r, cr] * 10.0 ** colour
    b_measured = origin[has_r, cb] == 1

    y = -np.log10(f_b_pred)
    centers, _ = depths_module._bin_histogram(y)
    edges = (centers[0] - 0.5 * depths_module.MAG_BIN
             + depths_module.MAG_BIN * np.arange(centers.size + 1))
    idx = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, centers.size - 1)
    n_total = np.bincount(idx, minlength=centers.size).astype(float)
    n_meas = np.bincount(idx[b_measured], minlength=centers.size).astype(float)
    occ = n_total > 0
    c, frac, nt = centers[occ], n_meas[occ] / n_total[occ], n_total[occ]
    order = np.argsort(c)
    return c[order], frac[order], nt[order]


def _recovery_fraction_at(c, frac, y_query):
    """The recovery curve's own fraction at `y_query` (`y = -log10 F`,
    linear interpolation, module docstring, Part B's diagnostic): a
    genuine roll-off gives 0.5 at its own fitted 50% point by definition,
    so this is what the diagnostic reads at `F_50,b`. Clipped to the
    curve's own bright/faint endpoints; `NaN` if there is no curve.
    """
    if c is None or c.size == 0:
        return np.nan
    return float(np.interp(y_query, c, frac))


def _truncation_test(fnu, origin, curated_bands, selected_mask, f50):
    """Per Spitzer band, the catalogue's own recovery curve (`_recovery_
    curve`) evaluated at the band's own fitted `F_50` (module docstring,
    Part B, REPORT-ONLY): a genuine roll-off gives 0.5 there by
    definition, since the fit's `F_50` IS the flux at which the band's
    own detections thin to half; a band still measured for
    `RECOVERY_FLOOR_FRACTION` (90%) or more of the catalogued sources at
    that same flux is not thinning on its own account there at all --
    its faint end is coupled to the limiting band `r`'s requirement, not
    to its own recovery. (The expected-vs-observed-count-below-F50 form
    of this test cannot see the effect: a histogram already cut off by
    another band is itself well fit by a roll-off, so the fit's own
    model reproduces the cut-off it was handed.) Returns `(truncated
    (5,) bool, recovery_at_f50 (5,), limiting_band (5,) str)`.
    """
    n = len(limits_module.IRAC_MIPS_KEYS)
    truncated = np.zeros(n, dtype=bool)
    recovery_at_f50 = np.full(n, np.nan)
    limiting_band = [None] * n
    for jk, key in enumerate(limits_module.IRAC_MIPS_KEYS):
        if not np.isfinite(f50[jk]):
            continue
        r_key = _limiting_band(origin, curated_bands, selected_mask, key)
        c, frac, _nt = _recovery_curve(fnu, origin, curated_bands, selected_mask, key, r_key)
        limiting_band[jk] = r_key
        rec = _recovery_fraction_at(c, frac, -np.log10(f50[jk]))
        recovery_at_f50[jk] = rec
        truncated[jk] = np.isfinite(rec) and rec >= RECOVERY_FLOOR_FRACTION
    return truncated, recovery_at_f50, limiting_band


def _fill_unoccupied(admitted, occupied, occupied_n_sources, occupied_medians):
    """Every admitted pixel's row: an occupied pixel keeps its own
    measured count and median; an admitted pixel with no sources takes
    the row of its nearest occupied pixel by angular distance between
    pixel centres -- a vectorised argmax of the unit-vector dot product
    (monotonic with angular distance) against every occupied pixel, no
    loop over pixels -- with its own `N_SOURCES = 0`.
    Returns `(pix, n_sources, filled_rows, n_filled)`, `pix` sorted.
    """
    pix = np.sort(admitted)
    is_occupied = np.isin(pix, occupied)

    occ_order = np.argsort(occupied)
    occ_row = occ_order[np.searchsorted(occupied[occ_order], pix[is_occupied])]

    n_sources = np.zeros(pix.size, dtype=np.int32)
    filled_rows = np.empty((pix.size, occupied_medians.shape[1]), dtype=occupied_medians.dtype)
    n_sources[is_occupied] = occupied_n_sources[occ_row]
    filled_rows[is_occupied] = occupied_medians[occ_row]

    filled_mask = ~is_occupied
    if filled_mask.any():
        occ_vec = np.asarray(hp.pix2vec(NSIDE_512, occupied, nest=True)).T  # (n_occ, 3)
        filled_vec = np.asarray(hp.pix2vec(NSIDE_512, pix[filled_mask], nest=True)).T
        nearest = np.argmax(filled_vec @ occ_vec.T, axis=1)  # no loop over pixels
        filled_rows[filled_mask] = occupied_medians[nearest]

    return pix, n_sources, filled_rows, int(filled_mask.sum())


def _check_identity(hpx_pix_512, source_limits, occupied, occupied_medians,
                     n_admitted, n_filled, region):
    """Prints, for ten occupied pixels chosen by a fixed seed, a direct
    numpy median over the per-source limits against the stored row --
    the brief's own identity check, run from the build itself -- plus
    the region's admitted/occupied/filled pixel counts.
    """
    rng = np.random.default_rng(CHECK_SEED)
    n_pick = min(N_CHECK_PIXELS, occupied.size)
    picked = rng.choice(occupied.size, size=n_pick, replace=False)
    print("depth_grid identity, %s, %d pixels (admitted=%d occupied=%d filled=%d):"
          % (region, n_pick, n_admitted, occupied.size, n_filled))
    for i in picked:
        p = occupied[i]
        direct = np.median(source_limits[hpx_pix_512 == p, :], axis=0).astype(np.float32)
        stored = occupied_medians[i]
        print("  pixel %d: direct median == stored row -> %s" %
              (p, bool(np.allclose(direct, stored, equal_nan=True))))


def build(config, regions=None):
    """Builds the per-region depth-grid product and the per-source limits
    product it is built from (default: every region in `regions.REGIONS`).
    """
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]

    band_keys = [b.key for b in definitions.BANDS]
    with progress_module.Stage("catalog.depth_grid") as st:
        n_done = 0
        out_paths = []
        n_admitted_total = n_occupied_total = n_filled_total = 0
        for region in regions:
            rs = access_module.region_slice(config, region)
            hpx_pix_512 = rs["hpx_pix_512"]
            admitted = _admitted_pixels_512(config, region)

            # the marginalised limit (module docstring, Part A): the
            # region-band counts fit on the complete low-column pixels'
            # own detected sources, in absolute flux.
            fnu, dcomp90, origin, curated_bands = _raw_source_arrays(config, region)
            width_dex, f50_2mass = _depth_fit_params(config, region)
            selected_mask, n_low_column, fallback = _low_column_selection(
                config, region, hpx_pix_512, admitted
            )
            alpha_region, f50_region_mjy, w_region_dex = _region_band_counts_fit(
                fnu, origin, curated_bands, selected_mask
            )

            # the recovery curve's own fraction at each band's fitted
            # F_50 is a REPORT-ONLY diagnostic and flag (module
            # docstring, Part B): for a pair the two-band rule makes
            # mutually required at the faint end, a catalogued source
            # carries both by construction, so this estimator is a
            # tautology for that pair and cannot bound either band's
            # recovery -- the limit itself stays at the counts fit.
            truncated, recovery_at_f50, limiting_band = _truncation_test(
                fnu, origin, curated_bands, selected_mask, f50_region_mjy
            )
            limit_kind = ["fit"] * len(limits_module.IRAC_MIPS_KEYS)
            for jk, key in enumerate(limits_module.IRAC_MIPS_KEYS):
                if truncated[jk]:
                    limit_kind[jk] = "bound"
                print("catalog.depth_grid: %s %s in %s (recovery at fitted F_50=%.3f, limiting band %s)"
                      % (key, "coupled" if truncated[jk] else "not coupled", region,
                         recovery_at_f50[jk], limiting_band[jk]))

            # every catalogued source's own limit (module docstring, Part
            # A) -- the one array `catalog.limits.limits` reads.
            f_lim_50_mjy, dcomp90_ref_log10 = _source_limits(
                dcomp90, curated_bands, selected_mask, f50_region_mjy, f50_2mass
            )

            alpha_full = np.full(len(band_keys), np.nan)
            f50_full = np.full(len(band_keys), np.nan)
            w_full = np.full(len(band_keys), np.nan)          # Spitzer-only, diagnostic
            w_dex_full = np.full(len(band_keys), np.nan)      # effective width, all 8 bands
            limit_kind_full = np.array(["fit"] * len(band_keys), dtype="S8")
            for jk, key in enumerate(limits_module.IRAC_MIPS_KEYS):
                j = band_keys.index(key)
                alpha_full[j] = alpha_region[jk]
                f50_full[j] = f50_region_mjy[jk]
                w_full[j] = w_region_dex[jk]
                w_dex_full[j] = w_region_dex[jk]
                limit_kind_full[j] = limit_kind[jk].encode()
            for key in limits_module.TWOMASS_KEYS:
                j = band_keys.index(key)
                w_dex_full[j] = width_dex[j]

            source_out_path = config_module.product_path(
                config, "catalog", "sesna", "limits", "source", region=region
            )
            os.makedirs(os.path.dirname(source_out_path), exist_ok=True)
            with h5py.File(source_out_path, "w") as f:
                f.attrs["GRANULE"] = "source"
                f.create_dataset("F_LIM_50_MJY", data=f_lim_50_mjy.astype(np.float32))
                f.create_dataset("W_DEX", data=w_dex_full)
                f.create_dataset("F_50_REGION_MJY", data=f50_full)
                f.create_dataset("ALPHA_REGION", data=alpha_full)
                f.create_dataset("DCOMP90_REF_LOG10", data=dcomp90_ref_log10)
                f.create_dataset("LIMIT_KIND", data=limit_kind_full)

            # the pixel grid (module docstring, Part A): F_LIM_50_PIX_MJY
            # and F_LIM_50_MED_MJY are now the same pixel median of the
            # corrected per-source limits, closing the two readers' drift.
            occupied, occupied_n_sources, occupied_medians = _pixel_medians(
                hpx_pix_512, f_lim_50_mjy.astype(np.float32)
            )
            pix, n_sources, f_lim_50_pix_mjy, n_filled = _fill_unoccupied(
                admitted, occupied, occupied_n_sources, occupied_medians
            )
            _check_identity(hpx_pix_512, f_lim_50_mjy.astype(np.float32), occupied, occupied_medians,
                             admitted.size, n_filled, region)
            w_dex_pix = np.broadcast_to(w_dex_full[None, :], (pix.size, len(band_keys))).astype(np.float32)

            print("depth_grid fit, %s: low_column_sources=%d fallback=%s"
                  % (region, n_low_column, fallback))
            print("  ALPHA_REGION=%s" % np.round(alpha_full, 3).tolist())
            print("  F_50_REGION_MJY=%s (Part B does not substitute; LIMIT_KIND is report-only)"
                  % np.round(f50_region_mjy, 4).tolist())
            print("  W_REGION_DEX=%s LIMIT_KIND=%s" % (np.round(w_full, 3).tolist(), limit_kind))

            out_path = config_module.product_path(
                config, "catalog", "sesna", "depth-grid", "hpx512", region=region
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with h5py.File(out_path, "w") as f:
                f.attrs["GRANULE"] = "hpx512"
                f.attrs["ALPHA_REGION"] = alpha_full
                f.attrs["F_50_REGION_MJY"] = f50_full
                f.attrs["W_REGION_DEX"] = w_full
                f.attrs["LOW_COLUMN_N_SOURCES"] = n_low_column
                f.attrs["LOW_COLUMN_FALLBACK"] = fallback
                f.attrs["LIMIT_KIND"] = limit_kind_full
                f.create_dataset("HPX_PIX_512", data=pix)
                f.create_dataset("N_SOURCES", data=n_sources)
                f.create_dataset("F_LIM_50_MED_MJY", data=f_lim_50_pix_mjy.astype(np.float32))
                f.create_dataset("F_LIM_50_PIX_MJY", data=f_lim_50_pix_mjy.astype(np.float32))
                f.create_dataset("W_DEX_PIX", data=w_dex_pix)

            out_paths.append(out_path)
            n_admitted_total += int(admitted.size)
            n_occupied_total += int(occupied.size)
            n_filled_total += int(n_filled)
            n_done += 1
            st.tick(n_done, len(regions), "regions")

        st.done(out_paths[-1] if out_paths else None, regions=len(regions),
                admitted=n_admitted_total, occupied=n_occupied_total, filled=n_filled_total)


if __name__ == "__main__":
    run(build)
