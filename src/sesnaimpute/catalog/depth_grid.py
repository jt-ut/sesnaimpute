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

The atlas instead reads two more columns, fitted the marginalised way (spec
section 3.3): per region and Spitzer band, on the sources of "complete
low-column" admitted pixels -- IRAC-union coverage above 0.9
(`sky.derived.coverage`) and adopted column below the 25th percentile of
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
unit conversion. Per admitted pixel, `F_LIM_50_PIX_MJY` is the region's
fitted 50% flux shifted by the positional offset of the pixel's own
sources' median `log10 DCOMP90` from the region selection's median (every
one of the pixel's own sources, not detections only -- `DCOMP90` is a map
value every source carries): `F_LIM_50_PIX_MJY = F_50_REGION_MJY *
10**(median_pix - median_region)`. `W_DEX_PIX` is the region's own fitted
width, broadcast to every pixel: no per-source scatter term. A pixel with
no sources of its own takes its nearest occupied pixel's shift, the same
fill `F_LIM_50_MED_MJY` uses. The three 2MASS bands carry no per-source map
and keep the region's constant `F50_2MASS_MJY`/`WIDTH_DEX`
(`catalog.depths`), unchanged.
"""

import os

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import depths as depths_module
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access as access_module

NSIDE_512 = 512

# Row-batch memory budget for the per-source limits array before its
# grouping by pixel (CODING_RULES.md rule 10b); at 8 bands of float32
# plus one int64 pixel id this admits far more than any region's source
# count in one batch.
ROW_BATCH_BUDGET_BYTES = 512 << 20

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
    """Each admitted pixel's IRAC-union coverage fraction
    (`sky.derived.coverage`'s five-band `FRAC` at `hpx512`, MIPS-24
    excluded): the per-band max as the union's lower bound, the same read
    `bmstp.atlas._coverage` performs (not imported here: `catalog` sits
    below `bmstp`), 0 where a pixel is absent (never observed).
    """
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)[:, :4].max(axis=1)
    order = np.argsort(cov_pix)
    loc = np.minimum(np.searchsorted(cov_pix[order], pix), max(cov_pix.size - 1, 0))
    hit = order[loc]
    found = cov_pix[hit] == pix if cov_pix.size else np.zeros(pix.size, dtype=bool)
    out = np.zeros(pix.size, dtype=np.float64)
    out[found] = frac[hit[found]]
    return out


def _pixel_column(config, pix):
    """Each admitted pixel's own adopted column, from the nside-256
    sightline it is a child of (survey-wide product, no region argument;
    `granules/build.py`'s own `HPX_PIX_256 = HPX_PIX_512 // 4`) -- the same
    join `bmstp.atlas._pixel_column` performs.
    """
    parent256 = pix // 4
    path = config_module.product_path(config, "sky/derived", "adopted", "column", "sightline")
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


def _pixel_dcomp_shift(admitted, hpx_pix_512, dcomp90, curated_bands, selected_mask, f50_region_mjy):
    """Every admitted pixel's `F_LIM_50_PIX_MJY` (5 Spitzer bands, module
    docstring): the region's fitted 50% flux shifted by the positional
    offset of the pixel's own sources' median `log10 DCOMP90` from the
    region selection's own median, both over every source (not detections
    only -- `DCOMP90` is a map value every source carries). The empty-pixel
    fill is `_fill_unoccupied`'s nearest-occupied rule.
    """
    spitzer_cols = [curated_bands.index(k) for k in limits_module.IRAC_MIPS_KEYS]
    log_dcomp = np.log10(dcomp90[:, spitzer_cols])
    median_region = np.median(log_dcomp[selected_mask], axis=0)

    df = pd.DataFrame(log_dcomp, columns=list(limits_module.IRAC_MIPS_KEYS))
    df["HPX_PIX_512"] = hpx_pix_512
    grouped = df.groupby("HPX_PIX_512", sort=True)
    occ_n = grouped.size().to_numpy(dtype=np.int32)
    occ_med = grouped[list(limits_module.IRAC_MIPS_KEYS)].median()
    occupied = occ_med.index.to_numpy(dtype=np.int64)
    occupied_medians = occ_med.to_numpy(dtype=np.float64)

    pix, _, filled_medians, n_filled = _fill_unoccupied(admitted, occupied, occ_n, occupied_medians)
    shift = filled_medians - median_region[None, :]
    f_lim = f50_region_mjy[None, :] * 10.0 ** shift
    return pix, f_lim, shift, n_filled


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
    """Builds the per-region depth-grid product for the given regions
    (default: every region in `regions.REGIONS`).
    """
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]

    with progress_module.Stage("catalog.depth_grid") as st:
        n_done = 0
        out_paths = []
        n_admitted_total = n_occupied_total = n_filled_total = 0
        for region in regions:
            rs = access_module.region_slice(config, region)
            source_limits = limits_module.limits(config, region)

            hpx_pix_512_parts, limits_parts = [], []
            row_bytes = 8 + 8 * 4
            for start, stop in batches_module.batches(
                rs["n_sources"], row_bytes, budget_bytes=ROW_BATCH_BUDGET_BYTES
            ):
                hpx_pix_512_parts.append(rs["hpx_pix_512"][start:stop])
                limits_parts.append(source_limits[start:stop])
            hpx_pix_512 = np.concatenate(hpx_pix_512_parts)
            all_limits = np.concatenate(limits_parts)

            occupied, occupied_n_sources, occupied_medians = _pixel_medians(hpx_pix_512, all_limits)
            admitted = _admitted_pixels_512(config, region)
            pix, n_sources, f_lim_50_med_mjy, n_filled = _fill_unoccupied(
                admitted, occupied, occupied_n_sources, occupied_medians
            )
            _check_identity(hpx_pix_512, all_limits, occupied, occupied_medians,
                             admitted.size, n_filled, region)

            # the marginalised limit and width (module docstring, spec
            # section 3.3): the region-band counts fit on the complete
            # low-column pixels' own sources, never on `limits()`'s
            # already-rescaled `F_lim,50`.
            fnu, dcomp90, origin, curated_bands = _raw_source_arrays(config, region)
            width_dex, f50_2mass = _depth_fit_params(config, region)
            selected_mask, n_low_column, fallback = _low_column_selection(
                config, region, hpx_pix_512, admitted
            )
            alpha_region, f50_region_mjy, w_region_dex = _region_band_counts_fit(
                fnu, origin, curated_bands, selected_mask
            )
            _, f_lim_spitzer, shift, n_filled_shift = _pixel_dcomp_shift(
                admitted, hpx_pix_512, dcomp90, curated_bands, selected_mask, f50_region_mjy
            )

            band_keys = [b.key for b in definitions.BANDS]
            f_lim_50_pix_mjy = np.empty((pix.size, len(band_keys)), dtype=np.float64)
            w_dex_pix = np.empty((pix.size, len(band_keys)), dtype=np.float64)
            for jk, key in enumerate(limits_module.IRAC_MIPS_KEYS):
                j = band_keys.index(key)
                f_lim_50_pix_mjy[:, j] = f_lim_spitzer[:, jk]
                w_dex_pix[:, j] = w_region_dex[jk]
            for tk, key in enumerate(limits_module.TWOMASS_KEYS):
                j = band_keys.index(key)
                f_lim_50_pix_mjy[:, j] = f50_2mass[tk]
                w_dex_pix[:, j] = width_dex[j]

            alpha_full = np.full(len(band_keys), np.nan)
            f50_full = np.full(len(band_keys), np.nan)
            w_full = np.full(len(band_keys), np.nan)
            for jk, key in enumerate(limits_module.IRAC_MIPS_KEYS):
                j = band_keys.index(key)
                alpha_full[j] = alpha_region[jk]
                f50_full[j] = f50_region_mjy[jk]
                w_full[j] = w_region_dex[jk]

            spitzer_cols = [band_keys.index(k) for k in limits_module.IRAC_MIPS_KEYS]
            shift_med = float(np.median(np.log10(
                f_lim_50_pix_mjy[:, spitzer_cols] / f_lim_50_med_mjy[:, spitzer_cols]
            )))
            print("depth_grid fit, %s: low_column_sources=%d fallback=%s"
                  % (region, n_low_column, fallback))
            print("  ALPHA_REGION=%s" % np.round(alpha_full, 3).tolist())
            print("  F_50_REGION_MJY=%s" % np.round(f50_full, 4).tolist())
            print("  W_REGION_DEX=%s" % np.round(w_full, 3).tolist())
            print("  median log10(F_LIM_50_PIX/F_LIM_50_MED), Spitzer bands=%.3f" % shift_med)

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
                f.create_dataset("HPX_PIX_512", data=pix)
                f.create_dataset("N_SOURCES", data=n_sources)
                f.create_dataset("F_LIM_50_MED_MJY", data=f_lim_50_med_mjy)
                f.create_dataset("F_LIM_50_PIX_MJY", data=f_lim_50_pix_mjy.astype(np.float32))
                f.create_dataset("W_DEX_PIX", data=w_dex_pix.astype(np.float32))

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
