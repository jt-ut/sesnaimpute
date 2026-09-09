"""Each arm's map column against the reddening of its own background
stars (SPEC_BMSTP_DRAFT.md section 3.1's "column row"; the measurement
`studies/column_at_depth.md`'s T1 makes by hand, here made a product:
owner ruling 2026-09-09).

Background 2MASS PSC stars at `Ks` 11-13 with clean photometry
(`ph_qual == 'AAA'`) are, in every region, dominated by stars beyond the
cloud (T1): their nside-512-pixel median `H - Ks` colour rises with the
cloud's own foreground column. Regressing that colour against the arm's
own map column (Herschel where covered, else Planck; the per-source `A_K`
each arm's own column module already writes, median per pixel over the
region's SESNA sources) gives a slope in `d(H-Ks)/dA_K`; dividing by
`A_H/A_K - 1` at the adopted hybrid law (`population.selection.
kappa_hybrid`) converts it to a scale on the map's own column, the same
`SCALE` T1 reports and W19's owner ruling asks to be applied in the
adoption (W36).

This is a survey-wide view of external 2MASS photometry read against
SESNA source positions and each arm's own already-derived column -- never
a SESNA flux -- so it belongs in `sky.derived` (memory:
`sesna-data-placement-rules`).
"""

import os

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.population import selection as selection_module
from sesnaimpute.sky.derived.twomass_counts import _csv_row_batch_size, _download_path

NSIDE = 512
ARMS = ("herschel", "planck")

#: T1's own star cut (`studies/column_at_depth.md` section 1): 2MASS PSC's
#: own clean-photometry grade, all three bands.
PH_QUAL_CLEAN = "AAA"
#: T1's own Ks window: complete and unconfused, and (per that study's own
#: measurement, reproduced per region below from TRILEGAL) dominated by
#: stars beyond the cloud in every region, so the pixel median is a
#: background star's colour.
KS_WINDOW = (11.0, 13.0)
#: T1's own per-pixel star-count floor for a usable median H-Ks.
MIN_STARS_PER_PIXEL = 20
#: brief W35: the arm's per-pixel column needs at least this many SESNA
#: sources for a stable median A_K.
MIN_SOURCES_PER_PIXEL = 5
#: brief W35: below this many usable (star AND column) pixels, the arm's
#: regression is not attempted for the region.
MIN_PIXELS = 20
#: brief W35, matching T1's own bootstrap: 200 draws over pixels, seed 0.
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 0

_H_IDX = selection_module.BAND_KEYS.index("H")
_KS_IDX = selection_module.BAND_KEYS.index("Ks")

#: The hybrid law's A_H/A_K at the ramp's two endpoints (diffuse-only,
#: dense-only): population.selection.kappa_ak at LAW_DIFFUSE and
#: LAW_DENSE. The two differ by (1.5551-1.5471)/(1.5551-1) = 1.4% in
#: (A_H/A_K - 1), the SCALE formula's denominator -- below every region's
#: own SCALE_SIGMA in the acceptance table (rule 3; reported in the
#: report, not asserted here as a runtime check).
_KAPPA_DIFFUSE_H = float(selection_module.kappa_ak(None, selection_module.LAW_DIFFUSE)[_H_IDX])
_KAPPA_DENSE_H = float(selection_module.kappa_ak(None, selection_module.LAW_DENSE)[_H_IDX])


def _star_pixel_stats(config, region):
    """This region's background-star pixels: `Ks` 11-13, clean photometry,
    nside-512 pixel (`healpy.ang2pix`, no frame conversion -- fp_psc's
    `glon`/`glat` are already this project's own Galactic grid), read in
    row batches from the widened download CSV (rule 10b; the same batching
    `sky.derived.twomass_counts` uses). Returns, for pixels with >=
    MIN_STARS_PER_PIXEL stars, sorted by pixel: (pix, median H-Ks, n_stars).
    """
    csv_path = _download_path(config, region)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"twomass_column_scale: no download CSV at {csv_path} -- run the "
            "'sesnaimpute.sky.download.twomass_counts.build' RUNBOOK line first")
    chunksize = _csv_row_batch_size(csv_path)
    pix_parts, hk_parts = [], []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if chunk.empty:
            continue
        sel = ((chunk["ph_qual"] == PH_QUAL_CLEAN)
               & (chunk["k_m"] >= KS_WINDOW[0]) & (chunk["k_m"] < KS_WINDOW[1]))
        if not sel.any():
            continue
        glon = chunk.loc[sel, "glon"].to_numpy(dtype=np.float64)
        glat = chunk.loc[sel, "glat"].to_numpy(dtype=np.float64)
        h_m = chunk.loc[sel, "h_m"].to_numpy(dtype=np.float64)
        k_m = chunk.loc[sel, "k_m"].to_numpy(dtype=np.float64)
        pix_parts.append(hp.ang2pix(NSIDE, glon, glat, nest=True, lonlat=True).astype(np.int64))
        hk_parts.append(h_m - k_m)
    if not pix_parts:
        return np.empty(0, np.int64), np.empty(0, np.float64), np.empty(0, np.int64)
    df = pd.DataFrame({"pix": np.concatenate(pix_parts), "hk": np.concatenate(hk_parts)})
    g = df.groupby("pix")["hk"]
    n_stars = g.size()
    med_hk = g.median()
    keep = n_stars.to_numpy() >= MIN_STARS_PER_PIXEL
    return (med_hk.index.to_numpy()[keep].astype(np.int64),
            med_hk.to_numpy()[keep], n_stars.to_numpy()[keep].astype(np.int64))


def _arm_column_path(config, region, arm):
    if arm == "herschel":
        return config_module.product_path(config, "sky/derived", "herschel", "column", "source", region=region)
    return config_module.product_path(config, "sky/derived", "planck", "column", "source", region=region)


def _arm_pixel_column(config, region, arm):
    """This region's per-pixel arm column: the median of the arm's own
    per-source `A_K` (Herschel only where `COVERED`; Planck's own arm is
    100% finite and positive by construction, `column.py`) over the
    region's SESNA sources landing in each nside-512 pixel
    (`granules.access.region_slice`'s own pixel, `per_source`'s own
    catalogue-row alignment). Returns pixels with >= MIN_SOURCES_PER_PIXEL
    sources, sorted by pixel: (pix, median A_K, n_sources)."""
    path = _arm_column_path(config, region, arm)
    runbook_line = "herschel_column" if arm == "herschel" else "planck_source_column"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"twomass_column_scale: {arm} arm missing for region {region!r} at {path!r} -- "
            f"run the sesnaimpute.sky.derived.{runbook_line} RUNBOOK line for it")
    rs = access.region_slice(config, region)
    pix = rs["hpx_pix_512"]
    if arm == "herschel":
        cols = access.per_source(config, region, path, ["A_K", "COVERED"])
        valid = cols["COVERED"].astype(bool) & np.isfinite(cols["A_K"]) & (cols["A_K"] > 0)
    else:
        cols = access.per_source(config, region, path, ["A_K"])
        valid = np.isfinite(cols["A_K"]) & (cols["A_K"] > 0)
    if not valid.any():
        return np.empty(0, np.int64), np.empty(0, np.float64), np.empty(0, np.int64)
    df = pd.DataFrame({"pix": pix[valid], "a_k": cols["A_K"][valid]})
    g = df.groupby("pix")["a_k"]
    n_src = g.size()
    med_ak = g.median()
    keep = n_src.to_numpy() >= MIN_SOURCES_PER_PIXEL
    return (med_ak.index.to_numpy()[keep].astype(np.int64),
            med_ak.to_numpy()[keep], n_src.to_numpy()[keep].astype(np.int64))


def _ols_slope_intercept(x, y):
    """Ordinary least squares, `y = intercept + slope * x` (brief W35)."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _bootstrap_scale_sigma(x, y, a_h_per_ak, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """`SCALE_SIGMA`: the std of `SCALE` over `n_bootstrap` pixel
    resamples (brief W35, T1's own method), the OLS slope formula
    evaluated on every draw at once (no loop over draws or pixels)."""
    n = x.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    xs, ys = x[idx], y[idx]
    sx, sy = xs.sum(axis=1), ys.sum(axis=1)
    sxx, sxy = (xs * xs).sum(axis=1), (xs * ys).sum(axis=1)
    denom = n * sxx - sx * sx
    slope_draws = np.where(denom != 0, (n * sxy - sx * sy) / np.where(denom != 0, denom, 1.0), np.nan)
    scale_draws = slope_draws / (a_h_per_ak - 1.0)
    return float(np.nanstd(scale_draws, ddof=1))


def region_arm_scale(config, region, arm, star_pix, star_med_hk, star_n):
    """This region and arm's `SCALE` row (brief W35 step 3): OLS of the
    background stars' median H-Ks on the arm's own per-pixel column, over
    pixels both sides have enough of; `SCALE = SLOPE / (A_H/A_K - 1)` at
    the hybrid law evaluated at the pixels' own median column; NaN below
    `MIN_PIXELS` usable pixels."""
    col_pix, col_med_ak, col_n = _arm_pixel_column(config, region, arm)
    common, i_star, i_col = np.intersect1d(star_pix, col_pix, assume_unique=True, return_indices=True)
    n_pixels = int(common.size)
    n_stars = int(star_n[i_star].sum()) if n_pixels else 0
    out = dict(scale=np.nan, scale_sigma=np.nan, slope=np.nan, intercept=np.nan,
               n_stars=n_stars, n_pixels=n_pixels)
    if n_pixels < MIN_PIXELS:
        return out
    x = col_med_ak[i_col]
    y = star_med_hk[i_star]
    slope, intercept = _ols_slope_intercept(x, y)
    median_column = float(np.median(x))
    w = selection_module.law_dense_weight(np.array([median_column]))
    kappa = selection_module.kappa_hybrid(config, w)[0]
    a_h_per_ak = float(kappa[_H_IDX] / kappa[_KS_IDX])
    scale = slope / (a_h_per_ak - 1.0)
    scale_sigma = _bootstrap_scale_sigma(x, y, a_h_per_ak)
    out.update(scale=scale, scale_sigma=scale_sigma, slope=slope, intercept=intercept)
    return out


def build_region(config, region):
    """Both arms' rows for one region."""
    star_pix, star_med_hk, star_n = _star_pixel_stats(config, region)
    return {arm: region_arm_scale(config, region, arm, star_pix, star_med_hk, star_n) for arm in ARMS}


def build(config, regions=None):
    """Writes `sky/derived/twomass/column-scale_twomass_region.hdf5`:
    `SCALE`/`SCALE_SIGMA`/`SLOPE_H_KS_PER_AK`/`INTERCEPT_H_KS`/`N_STARS`/
    `N_PIXELS` over `REGION` (thirty, `regions.REGIONS` order) x `ARM`
    (herschel, planck); updated in place for the requested regions only
    (rule 5c)."""
    names = regions or [r.name for r in regions_module.REGIONS]
    out_path = config_module.product_path(config, "sky/derived", "twomass", "column-scale", "region")
    with progress_module.Stage("sky.derived.twomass_column_scale") as st:
        n_regions = len(names)
        results = []
        for i, region in enumerate(names):
            results.append(build_region(config, region))
            st.tick(i + 1, n_regions, "regions")

        keys = (("SCALE", "scale"), ("SCALE_SIGMA", "scale_sigma"),
                ("SLOPE_H_KS_PER_AK", "slope"), ("INTERCEPT_H_KS", "intercept"),
                ("N_STARS", "n_stars"), ("N_PIXELS", "n_pixels"))
        rows = {}
        for name, key in keys:
            dtype = np.int64 if key in ("n_stars", "n_pixels") else np.float64
            rows[name] = np.array([[r[arm][key] for arm in ARMS] for r in results], dtype=dtype)
        tables_module.update_rows(out_path, names, rows, granule="region")
        with h5py.File(out_path, "a") as f:
            if "ARM" not in f:
                f.create_dataset("ARM", data=np.array([a.encode("utf-8") for a in ARMS]))

        n_below_min = sum(1 for r in results for arm in ARMS if r[arm]["n_pixels"] < MIN_PIXELS)
        print("twomass_column_scale: kappa_hybrid A_H/A_K at diffuse=%.4f, dense=%.4f "
              "(%.2f%% apart in A_H/A_K-1, below every region's own SCALE_SIGMA)"
              % (_KAPPA_DIFFUSE_H, _KAPPA_DENSE_H,
                 100.0 * (_KAPPA_DIFFUSE_H - _KAPPA_DENSE_H) / (_KAPPA_DIFFUSE_H - 1.0)), flush=True)
        for region, r in zip(names, results):
            for arm in ARMS:
                row = r[arm]
                if row["n_pixels"] < MIN_PIXELS:
                    print("twomass_column_scale: %-20s %-9s below MIN_PIXELS=%d "
                          "(n_pixels=%d, n_stars=%d) -> NaN"
                          % (region, arm, MIN_PIXELS, row["n_pixels"], row["n_stars"]), flush=True)
        st.done(out_path, regions=n_regions, arms_below_min_pixels=n_below_min)
    return out_path


if __name__ == "__main__":
    run(build)
