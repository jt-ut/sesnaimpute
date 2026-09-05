"""Survey depth: the 50%-completeness flux, per region and band
(SPEC_PRIORS.md section 1.3).

Intrinsic source counts rise steeply toward faint flux while completeness
falls, so the observed counts turn over, and the turnover is the survey's
depth. For each region and Spitzer band (I1-I4, M1) the turnover flux
`F_50` is read off the region's own detections (`ORIGIN_FNU == 1`) as the
mode of a Freedman-Diaconis-binned log10(flux) histogram, smoothed once
with a 3-bin box filter before the argmax; the reported quantity is the
offset `Delta = log10(median DCOMP90) - log10(F_50)`, so that a source's
50%-completeness limit becomes `DCOMP90 * 10**(-Delta)` (`limits.py`),
keeping each source's own DCOMP90 spatial pattern and moving only the
region's typical value. For each region and 2MASS band (J, H, Ks), which
carries no per-source completeness map, `F_50` itself is the limit.

A region-band's Delta ships as its own measurement, whatever its sign: a
negative Delta says the catalogue's detections in that band turn over
brighter than the completeness map's 90% level, and the region's own
turnover is still the limit. The only substitution is for a region whose
histogram is too sparse to have a turnover at all (fewer than
`MIN_RAW_BINS` natural Freedman-Diaconis bins), which takes its band's
median Delta over the regions that did resolve it; which regions were
substituted is recorded in `SUBSTITUTED`. The bootstrap uncertainty of
each Delta (`BOOTSTRAP_RESAMPLES` re-runs of the whole estimator on
resamples of the region's detections) is stored beside it in
`SIGMA_DELTA_DEX`; it is small (0.01-0.05 dex on every region measured)
and never decides anything.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run

IRAC_MIPS_KEYS = ("I1", "I2", "I3", "I4", "M1")
TWOMASS_KEYS = ("J", "H", "Ks")

SMOOTH_WINDOW = 3

# Resamples per region-band for the bootstrap sampling uncertainty on
# Delta -- re-running the whole estimator this many times is a build-time
# cost paid once.
BOOTSTRAP_RESAMPLES = 1000

# Fixed so the resolved/unresolved verdict is reproducible bit-for-bit
# from the catalogue alone.
BOOTSTRAP_SEED = 20260822

MIN_RAW_BINS = 4


def _freedman_diaconis_edges(x, min_bins=MIN_RAW_BINS, max_bins=20000):
    """Bin edges by the Freedman-Diaconis rule (width = 2*IQR*n**(-1/3),
    falling back to Scott's rule on a zero IQR). Also returns the raw,
    unclamped bin count `n_raw`, since a raw count below `MIN_RAW_BINS`
    is this module's unresolved test.
    """
    lo, hi = float(x.min()), float(x.max())
    q75, q25 = np.percentile(x, [75.0, 25.0])
    width = 2.0 * (q75 - q25) * x.size ** (-1.0 / 3.0)
    if not width > 0:
        width = 3.49 * float(np.std(x, ddof=1)) * x.size ** (-1.0 / 3.0)
    n_raw = int(np.ceil((hi - lo) / width)) if width > 0 else 0
    n_bins = int(np.clip(n_raw, min_bins, max_bins))
    return np.linspace(lo, hi, n_bins + 1), n_raw


def _smooth_counts(counts, window=SMOOTH_WINDOW):
    """3-bin box filter, edge-truncated: each bin is normalised by the
    count of real neighbours actually inside the window, not a fixed
    denominator.
    """
    if window <= 1 or counts.size < window:
        return counts
    kernel = np.ones(window)
    sums = np.convolve(counts, kernel, mode="same")
    overlap = np.convolve(np.ones_like(counts), kernel, mode="same")
    return sums / overlap


def _rollover_from_log10(x):
    """The turnover: Freedman-Diaconis histogram mode of an already-log10
    sample, 3-bin-smoothed before the argmax. Returns `(log10_f50,
    n_raw_bins)`; `n_raw_bins < MIN_RAW_BINS` marks an unresolved
    histogram.
    """
    edges, n_raw = _freedman_diaconis_edges(x)
    counts, _ = np.histogram(x, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = _smooth_counts(counts.astype(float))
    return float(centers[int(np.argmax(smoothed))]), n_raw


def _bootstrap_sigma_log10_f50(x, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """The estimator's own sampling uncertainty: resample the detections
    with replacement and re-run the whole estimator on each resample.
    """
    rng = np.random.default_rng(seed)
    n = x.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    vals = np.array([_rollover_from_log10(x[i])[0] for i in idx])
    return float(np.std(vals))


def _region_depths(curated_path):
    """One region's F_50 (2MASS) and Delta (Spitzer), with the resolved
    flag and the bootstrap uncertainty of Delta per Spitzer band.
    """
    with h5py.File(curated_path, "r") as f:
        fnu = f["FNU_MJY"][:]
        dcomp90 = f["DCOMP90_MJY"][:]
        origin = f["ORIGIN_FNU"][:]
        bands = [b.decode() if isinstance(b, bytes) else b for b in f.attrs["BANDS"]]

    delta_dex = np.full(len(IRAC_MIPS_KEYS), np.nan)
    sigma_delta_dex = np.full(len(IRAC_MIPS_KEYS), np.nan)
    resolved = np.zeros(len(IRAC_MIPS_KEYS), dtype=bool)
    for j, key in enumerate(IRAC_MIPS_KEYS):
        b = bands.index(key)
        detected = fnu[origin[:, b] == 1, b]
        log10_f50, n_raw = _rollover_from_log10(np.log10(detected))
        log10_dcomp_med = float(np.log10(np.median(dcomp90[:, b])))
        d = log10_dcomp_med - log10_f50
        sigma_boot = _bootstrap_sigma_log10_f50(np.log10(detected))
        delta_dex[j] = d
        sigma_delta_dex[j] = sigma_boot
        resolved[j] = n_raw >= MIN_RAW_BINS

    f50_2mass = np.full(len(TWOMASS_KEYS), np.nan)
    for j, key in enumerate(TWOMASS_KEYS):
        b = bands.index(key)
        detected = fnu[origin[:, b] == 1, b]
        log10_f50, _ = _rollover_from_log10(np.log10(detected))
        f50_2mass[j] = 10.0 ** log10_f50

    return delta_dex, sigma_delta_dex, resolved, f50_2mass


def build(config, regions=None):
    """Builds the region-granule survey-depths product for the given
    regions (default: every region in `regions.REGIONS`), parallelised
    over regions with joblib. Reads the curated catalogues from
    `curated.build`.
    """
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]

    curated_paths = []
    for region in regions:
        p = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"catalog.depths.build: curated catalogue missing for region {region!r} "
                f"at {p!r} -- run the curated-catalogue RUNBOOK line for it"
            )
        curated_paths.append(p)

    results = Parallel(n_jobs=-1)(delayed(_region_depths)(p) for p in curated_paths)
    delta_dex = np.array([r[0] for r in results])
    sigma_delta_dex = np.array([r[1] for r in results])
    resolved = np.array([r[2] for r in results])
    f50_2mass = np.array([r[3] for r in results])

    substituted = ~resolved
    for j in range(len(IRAC_MIPS_KEYS)):
        if resolved[:, j].any():
            band_median = np.median(delta_dex[resolved[:, j], j])
        else:
            band_median = np.nanmedian(delta_dex[:, j])
        delta_dex[substituted[:, j], j] = band_median

    band_keys = [b.key for b in definitions.BANDS]
    substituted_full = np.zeros((len(regions), len(definitions.BANDS)), dtype=bool)
    for j, key in enumerate(IRAC_MIPS_KEYS):
        substituted_full[:, band_keys.index(key)] = substituted[:, j]

    out_path = config_module.product_path(config, "catalog", "sesna", "depths", "region")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    region_bytes = np.array([r.encode("utf-8") for r in regions])
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "region"
        f.create_dataset("REGION", data=region_bytes)
        f.create_dataset("DELTA_DEX", data=delta_dex)
        f.create_dataset("SIGMA_DELTA_DEX", data=sigma_delta_dex)
        f.create_dataset("F50_2MASS_MJY", data=f50_2mass)
        f.create_dataset("SUBSTITUTED", data=substituted_full)


if __name__ == "__main__":
    run(build)
