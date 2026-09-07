"""The depth grid: median 50%-completeness limit per nside-512 pixel
(SPEC_BMSTP_DRAFT.md section 3.3), the object the prior atlas reads at a
pixel's median limits instead of a per-source one (SPEC_PRIORS.md
section 1.3, "the atlas evaluates the same code at a pixel's median
limits"). One row per ADMITTED pixel (IMPLEMENTATION_BMSTP_DRAFT.md P6),
not merely per occupied one.

Per region, admission is the granule map's own rule (`granules/build.py`):
every nside-512 child of one of the region's source-bearing nside-256
pixels. For an OCCUPIED admitted pixel (carrying at least one catalogued
source), `F_LIM_50_MED_MJY` is the median, over the pixel's own sources,
of each band's 50%-completeness limit -- `catalog.limits.limits`, the one
limits reader every consumer uses, evaluated once for the whole region
and grouped by pixel with a pandas groupby (vectorised: no loop over
pixels). An admitted pixel with no sources of its own (a mosaic-support
sibling of an occupied nside-256 pixel) takes the row of its nearest
occupied pixel by angular distance between pixel centres (a vectorised
argmax of the unit-vector dot product against every occupied pixel, no
loop over pixels), with its own `N_SOURCES = 0` so a consumer can tell a
filled row from a measured one.

The pixel's median source flux biases `F_LIM_50_MED_MJY` bright by 0.2-0.4
dex for the five Spitzer bands (`DCOMP90` rises with a source's own flux
within a pixel, `studies/total_count_level.md` section 5(i)), so the
atlas instead reads two more columns computed the marginalised way (spec
section 3.3, "the depth grid"). Per region and Spitzer band, on the
band's detections (`ORIGIN_FNU == 1`), `log10 DCOMP90 = a_pix + b log10 f
+ eps` is fit once: both logs are pixel-demeaned (a pandas groupby mean,
no loop over pixels) and `b` is the least-squares slope of the demeaned
pairs, `sigma_eps` the scatter left over. Per pixel, over every one of
its own sources (not detections only -- a non-detection's substituted
flux equals its own `DCOMP90`, so it still enters), `a_pix =
median(log10 DCOMP90 - b log10 f)`; `F_LIM_50_PIX_MJY = 10**((a_pix -
Delta) / (1 - b))` is the fixed point of the fitted relation, and
`W_DEX_PIX = sqrt(WIDTH_DEX**2 + sigma_eps**2) / (1 - b)` is that same
source-level width carried to a fixed limit -- one width per region-band,
broadcast to every pixel (only `F_LIM_50_PIX_MJY` varies pixel to pixel,
through `a_pix`). The three 2MASS bands carry no per-source proxy: their
`F_LIM_50_PIX_MJY`/`W_DEX_PIX` are the region's constant
`F50_2MASS_MJY`/`WIDTH_DEX`. A pixel with none of the band's own
detections takes its nearest detected pixel's row, the same fill as
`F_LIM_50_MED_MJY` uses for an unoccupied pixel. `F_LIM_50_MED_MJY`
stays, unread by the fitter, as the diagnostic the study compares the
fix against.
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
    band, the curated file's own order) that the pixel regression needs
    and `catalog.limits.limits` does not expose (it returns the already
    rescaled 50%-completeness limit, not the raw map value or flux).
    """
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        fnu = f["FNU_MJY"][:]
        dcomp90 = f["DCOMP90_MJY"][:]
        origin = f["ORIGIN_FNU"][:]
        bands = [b.decode() if isinstance(b, bytes) else b for b in f.attrs["BANDS"]]
    return fnu, dcomp90, origin, bands


def _depth_fit_params(config, region):
    """The region's fitted `DELTA_DEX` (5 Spitzer bands), `WIDTH_DEX` (8
    bands) and `F50_2MASS_MJY` (3 bands) from `catalog.depths` -- the same
    product `catalog.limits.limits` reads.
    """
    path = config_module.product_path(config, "catalog", "sesna", "depths", "region")
    with h5py.File(path, "r") as f:
        regions = [r.decode() if isinstance(r, bytes) else r for r in f["REGION"][:]]
        ridx = regions.index(region)
        delta_dex = f["DELTA_DEX"][ridx, :]
        width_dex = f["WIDTH_DEX"][ridx, :]
        f50_2mass = f["F50_2MASS_MJY"][ridx, :]
    return delta_dex, width_dex, f50_2mass


def _band_slope(pix_detected, log_dcomp_detected, log_f_detected):
    """One Spitzer band's region-wide slope `b` and residual scatter
    `sigma_eps` of `log10 DCOMP90` against `log10 f` (studies/
    total_count_level.md section 5(i)): a free intercept per pixel is
    absorbed by demeaning both logs within their own pixel (a pandas
    groupby mean, no loop over pixels); `b` is the least-squares slope of
    the demeaned pairs, `sigma_eps` the scatter left over.
    """
    df = pd.DataFrame({"pix": pix_detected, "logd": log_dcomp_detected, "logf": log_f_detected})
    means = df.groupby("pix")[["logd", "logf"]].transform("mean")
    dd = df["logd"].to_numpy() - means["logd"].to_numpy()
    df_ = df["logf"].to_numpy() - means["logf"].to_numpy()
    b = float(np.sum(dd * df_) / np.sum(df_ ** 2))
    sigma_eps = float(np.std(dd - b * df_))
    return b, sigma_eps


def _pixel_a(pix, log_dcomp, log_f, b):
    """Per pixel, `a_pix = median(log10 DCOMP90 - b log10 f)` over every
    one of its own sources -- a pandas groupby median, no loop over
    pixels. Returns `(pix_ids, a_pix)`, `pix_ids` sorted.
    """
    grouped = pd.Series(log_dcomp - b * log_f).groupby(pix)
    medians = grouped.median()
    return medians.index.to_numpy(dtype=np.int64), medians.to_numpy(dtype=np.float64)


def _pixel_limits_and_width(admitted, hpx_pix_512, fnu, dcomp90, origin, curated_bands,
                             delta_dex, width_dex, f50_2mass):
    """Every admitted pixel's `F_LIM_50_PIX_MJY` (the fixed point of the
    fitted `log10 DCOMP90` vs `log10 f` relation) and `W_DEX_PIX` (the
    source-level width carried to that fixed limit), plus the fitted `b`
    and `sigma_eps` per band -- module docstring, spec section 3.3. Fit
    and `a_pix` both use only the band's own detections (`ORIGIN_FNU ==
    1`): a pixel with none of its own takes its nearest detected pixel's
    row, same fill as `F_LIM_50_MED_MJY` (`_fill_unoccupied`), per band
    since a pixel detected in one Spitzer band need not be in another.
    Returns `(pix, f_lim_pix, w_dex_pix, b_slope, sigma_eps)`, `pix`
    sorted (`np.sort(admitted)`).
    """
    band_keys = [b.key for b in definitions.BANDS]
    pix = np.sort(admitted)
    n_pix = pix.size
    f_lim_pix = np.empty((n_pix, len(band_keys)), dtype=np.float64)
    w_dex_pix = np.empty((n_pix, len(band_keys)), dtype=np.float64)
    b_slope = np.full(len(band_keys), np.nan)
    sigma_eps = np.full(len(band_keys), np.nan)

    for jk, key in enumerate(limits_module.IRAC_MIPS_KEYS):
        j = band_keys.index(key)
        cb = curated_bands.index(key)
        detected = origin[:, cb] == 1
        pix_det = hpx_pix_512[detected]
        log_dcomp_det = np.log10(dcomp90[detected, cb])
        log_f_det = np.log10(fnu[detected, cb])
        b, s = _band_slope(pix_det, log_dcomp_det, log_f_det)
        b_slope[j] = b
        sigma_eps[j] = s

        occ_band, a_pix = _pixel_a(pix_det, log_dcomp_det, log_f_det, b)
        f_lim_occ = 10.0 ** ((a_pix - delta_dex[jk]) / (1.0 - b))
        n_dummy = np.zeros(occ_band.size, dtype=np.int32)
        _, _, f_lim_col, _ = _fill_unoccupied(admitted, occ_band, n_dummy, f_lim_occ[:, None])
        f_lim_pix[:, j] = f_lim_col[:, 0]
        w_dex_pix[:, j] = np.sqrt(width_dex[j] ** 2 + s ** 2) / (1.0 - b)

    for tk, key in enumerate(limits_module.TWOMASS_KEYS):
        j = band_keys.index(key)
        f_lim_pix[:, j] = f50_2mass[tk]
        w_dex_pix[:, j] = width_dex[j]

    return pix, f_lim_pix, w_dex_pix, b_slope, sigma_eps


def _fill_unoccupied(admitted, occupied, occupied_n_sources, occupied_medians):
    """Every admitted pixel's row: an occupied pixel keeps its own
    measured count and median; an admitted pixel with no sources takes
    the row of its nearest occupied pixel by angular distance between
    pixel centres -- a vectorised argmax of the unit-vector dot product
    (monotonic with angular distance) against every occupied pixel, no
    loop over pixels -- with its own `N_SOURCES = 0`.
    Returns `(pix, n_sources, f_lim_50_med_mjy, n_filled)`, `pix` sorted.
    """
    pix = np.sort(admitted)
    is_occupied = np.isin(pix, occupied)

    occ_order = np.argsort(occupied)
    occ_row = occ_order[np.searchsorted(occupied[occ_order], pix[is_occupied])]

    n_sources = np.zeros(pix.size, dtype=np.int32)
    f_lim_50_med_mjy = np.empty((pix.size, occupied_medians.shape[1]), dtype=np.float32)
    n_sources[is_occupied] = occupied_n_sources[occ_row]
    f_lim_50_med_mjy[is_occupied] = occupied_medians[occ_row]

    filled_mask = ~is_occupied
    if filled_mask.any():
        occ_vec = np.asarray(hp.pix2vec(NSIDE_512, occupied, nest=True)).T  # (n_occ, 3)
        filled_vec = np.asarray(hp.pix2vec(NSIDE_512, pix[filled_mask], nest=True)).T
        nearest = np.argmax(filled_vec @ occ_vec.T, axis=1)  # no loop over pixels
        f_lim_50_med_mjy[filled_mask] = occupied_medians[nearest]

    return pix, n_sources, f_lim_50_med_mjy, int(filled_mask.sum())


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
            # section 3.3): fit on the raw per-source flux/map-value pairs,
            # never on `limits()`'s already-rescaled `F_lim,50`.
            fnu, dcomp90, origin, curated_bands = _raw_source_arrays(config, region)
            delta_dex, width_dex, f50_2mass = _depth_fit_params(config, region)
            _, f_lim_50_pix_mjy, w_dex_pix_grid, b_slope, sigma_eps = _pixel_limits_and_width(
                admitted, hpx_pix_512, fnu, dcomp90, origin, curated_bands,
                delta_dex, width_dex, f50_2mass
            )

            band_keys = [b.key for b in definitions.BANDS]
            spitzer_cols = [band_keys.index(k) for k in limits_module.IRAC_MIPS_KEYS]
            shift = np.log10(f_lim_50_pix_mjy[:, spitzer_cols] / f_lim_50_med_mjy[:, spitzer_cols])
            print("depth_grid fit, %s: b=%s sigma_eps=%s"
                  % (region, np.round(b_slope, 3).tolist(), np.round(sigma_eps, 3).tolist()))
            print("  median log10(F_LIM_50_PIX/F_LIM_50_MED), Spitzer bands=%.3f, median W_DEX_PIX=%s"
                  % (float(np.median(shift)), np.round(np.median(w_dex_pix_grid, axis=0), 3).tolist()))

            out_path = config_module.product_path(
                config, "catalog", "sesna", "depth-grid", "hpx512", region=region
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with h5py.File(out_path, "w") as f:
                f.attrs["GRANULE"] = "hpx512"
                f.attrs["B_SLOPE"] = b_slope
                f.attrs["SIGMA_EPS_DEX"] = sigma_eps
                f.create_dataset("HPX_PIX_512", data=pix)
                f.create_dataset("N_SOURCES", data=n_sources)
                f.create_dataset("F_LIM_50_MED_MJY", data=f_lim_50_med_mjy)
                f.create_dataset("F_LIM_50_PIX_MJY", data=f_lim_50_pix_mjy.astype(np.float32))
                f.create_dataset("W_DEX_PIX", data=w_dex_pix_grid.astype(np.float32))

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
