"""Survey depth: the 50%-completeness flux, per region and band
(SPEC_PRIORS.md section 1.3).

For each region and Spitzer band (I1-I4, M1), every detection is first
put in units of its own map depth: `x = m - m90`, where `m =
-2.5 log10(F)` is the detection's magnitude and `m90 = -2.5
log10(DCOMP90)` is that same source's own completeness-map value. The
histogram of `x` over the sources with `ORIGIN_FNU == 1` (0.1-mag bins
from the bright end to 2 mag beyond the faintest detection) is fit with
power-law counts times an error-function completeness roll-off,
`N(x) = A * 10**(alpha*x) * C(x)`, `C(x) = 0.5 * erfc((x - x50) /
(sqrt(2) * w))`, with `A`, `alpha`, `x50` and `w` all free, by Poisson
likelihood on the whole histogram. Because each detection already carries
its own map value out before histogramming, the map's spatial spread
across the region no longer enters the fitted width -- only one width is
learned, per region and band, same as before.

`x = 0` is the map's own 90%-completeness point, so `x50` is directly the
survey's 90%-to-50% offset in magnitude, and `Delta = 0.4 * x50` (dex) is
the offset applied to every source's own `DCOMP90` to reach its
50%-completeness limit, `F_lim,50 = DCOMP90 * 10**(-Delta)` (`limits.py`).
Delta can come out negative: where a band's detections turn over
brighter than the map's own 90% level, `x50` and hence Delta land below
zero, and the fitted curve says so instead of being forced positive. The
roll-off's own 90% point, `x = delta_offset`, floats along with the
width; `delta_offset = x50 - 1.2816 * w` is reported alongside Delta and
recovers to the mag-space offset the ruling calls `delta`. The fit's
relative L1 distance from the histogram, restricted to the roll-off
region (bins fainter than `x = -1`), is reported alongside the fitted
width; a band whose histogram cannot be fit by a roll-off shows this as a
large residual rather than a hidden bias in Delta.

The three 2MASS bands (J, H, Ks) carry no per-source completeness map, so
the same model is fit directly on absolute magnitude `m`, with `m50` and
`w` both free; `F50_2MASS_MJY = 10**(-0.4 m50)` is the region's
50%-completeness flux and the analogous relative L1 residual (bins
fainter than `m50 - 1` mag) is reported beside it.

A Spitzer band is substituted -- taking its band's median Delta over the
regions that fit -- only when its fit fails to converge or its histogram
has fewer than `MIN_POPULATED_BINS` populated bins; which regions were
substituted is recorded in `SUBSTITUTED`. The bootstrap uncertainty of
each fitted Delta (`BOOTSTRAP_RESAMPLES` re-fits of the whole estimator
on resamples of the region's detections) is stored beside it in
`SIGMA_DELTA_DEX`. Each resample is drawn and fit one at a time, never as
a full `(BOOTSTRAP_RESAMPLES, n_detections)` index array, to keep the
per-band working set small.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import erfc

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.build import run

IRAC_MIPS_KEYS = ("I1", "I2", "I3", "I4", "M1")
TWOMASS_KEYS = ("J", "H", "Ks")

# Histogram bin width, in magnitude (SPEC_PRIORS.md section 1.3).
MAG_BIN = 0.1

# A histogram with fewer populated bins than this cannot constrain the
# model's three free shape parameters and is not fit.
MIN_POPULATED_BINS = 4

# 1/2.5: converts a magnitude offset to the dex offset of the flux it
# implies (the Pogson magnitude-flux relation).
DEX_PER_MAG = 0.4

# The standard normal distribution's one-sided 90% quantile, Phi^-1(0.9):
# an error-function completeness curve's 90% point sits `Z90 * w`
# brighter than its 50% point, so `delta_offset = m50 - Z90 * w`.
Z90 = 1.2816

# Resamples per region-band for the bootstrap sampling uncertainty on
# Delta -- re-fitting the whole estimator this many times is a build-time
# cost paid once.
BOOTSTRAP_RESAMPLES = 1000

# Fixed so the fit is reproducible bit-for-bit from the catalogue alone.
BOOTSTRAP_SEED = 20260822

_ALPHA_BOUNDS = (-2.0, 2.0)
_W_BOUNDS = (0.02, 3.0)


def _completeness(m, m50, w):
    """The error-function completeness roll-off, `C(m50) = 0.5`."""
    return 0.5 * erfc((m - m50) / (np.sqrt(2.0) * w))


def _model_free(m, alpha, m50, w):
    """`N(m)` shape (up to `A`) with `m50` and `w` both free: used both
    for the 2MASS bands (no per-source map to peg to) and for the
    Spitzer bands, where `m` is each detection's magnitude relative to
    its own map value and `m50` is therefore the survey's own
    90%-to-50% offset rather than an absolute magnitude.
    """
    return 10.0 ** (alpha * m) * _completeness(m, m50, w)


def _poisson_nll_and_amplitude(f, n):
    """The Poisson negative log likelihood of histogram counts `n` against
    shape `f`, with the amplitude `A` profiled out analytically (the
    score equation for `A` sets the model's total count to the data's).
    """
    f = np.clip(f, 1e-300, None)
    total_n = float(n.sum())
    total_f = float(f.sum())
    if not total_f > 0:
        return np.inf, np.nan
    amplitude = total_n / total_f
    mu = np.clip(amplitude * f, 1e-300, None)
    nll = float(np.sum(mu - n * np.log(mu)))
    return nll, amplitude


def _bin_histogram(m):
    """Bins a magnitude-like array into 0.1-mag bins from the bright end
    to 2 mag beyond the faintest value.
    """
    lo = float(m.min())
    hi = float(m.max()) + 2.0
    n_bins = max(1, int(np.ceil((hi - lo) / MAG_BIN)))
    edges = lo + MAG_BIN * np.arange(n_bins + 1)
    counts, _ = np.histogram(m, bins=edges)
    centers = edges[:-1] + 0.5 * MAG_BIN
    return centers, counts.astype(float)


def _mag_histogram(detected_flux):
    """The detected-flux histogram in absolute magnitude (2MASS, which
    has no per-source map to measure relative to).
    """
    return _bin_histogram(-2.5 * np.log10(detected_flux))


def _relative_mag(detected_flux, dcomp90_detected):
    """Each detection's magnitude relative to its own map value,
    `m - m90 = -2.5 log10(F / DCOMP90)`.
    """
    return -2.5 * np.log10(detected_flux / dcomp90_detected)


def _relative_mag_histogram(detected_flux, dcomp90_detected):
    """The Spitzer detected-flux histogram in magnitude relative to each
    source's own map value: `x = 0` is the map's own 90%-completeness
    point for every source, so the map's spatial spread across the
    region does not enter the histogram's width.
    """
    return _bin_histogram(_relative_mag(detected_flux, dcomp90_detected))


def _fit_free(centers, counts):
    """Poisson-likelihood fit of `(alpha, m50, w)`, all free (2MASS)."""
    m50_guess = centers[int(np.argmax(counts))] + 0.5

    def nll(theta):
        alpha, m50, w = theta
        f = _model_free(centers, alpha, m50, w)
        val, _ = _poisson_nll_and_amplitude(f, counts)
        return val

    x0 = np.array([0.3, m50_guess, 0.3])
    m50_bounds = (centers.min() - 2.0, centers.max() + 5.0)
    bounds = (_ALPHA_BOUNDS, m50_bounds, _W_BOUNDS)
    return minimize(nll, x0=x0, method="L-BFGS-B", bounds=bounds)


def _relative_l1_residual(centers, counts, model_values, reference_mag):
    """The relative L1 distance between the fitted model and the
    histogram, restricted to the roll-off region: bins fainter than
    `reference_mag - 1` mag.
    """
    mask = centers >= (reference_mag - 1.0)
    denom = float(counts[mask].sum())
    if not denom > 0:
        return float("nan")
    return float(np.sum(np.abs(model_values[mask] - counts[mask])) / denom)


def _bootstrap_sigma_delta_spitzer(detected_flux, dcomp90_detected, n_resamples=BOOTSTRAP_RESAMPLES,
                                    seed=BOOTSTRAP_SEED):
    """The estimator's own sampling uncertainty on Delta: resample the
    detections (flux and each one's own map value, paired) with
    replacement and re-fit the whole free model on each resample, one
    resample at a time -- never a full `(n_resamples, n_detections)`
    index array, which for a tens-of-thousands-detection band would
    itself be the size of the catalogue many times over.
    """
    rng = np.random.default_rng(seed)
    n = detected_flux.size
    deltas = np.full(n_resamples, np.nan)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        centers, counts = _relative_mag_histogram(detected_flux[idx], dcomp90_detected[idx])
        if np.count_nonzero(counts > 0) < MIN_POPULATED_BINS:
            continue
        res = _fit_free(centers, counts)
        if res.success:
            deltas[k] = DEX_PER_MAG * res.x[1]
    return float(np.nanstd(deltas))


def _fit_spitzer_band(detected_flux, dcomp90_detected):
    """One Spitzer band's fit, on each detection's magnitude relative to
    its own map value: Delta, its offset from the map's own 90% point
    (`delta_offset`), the fitted width, the bootstrap sigma, the fit
    residual, and whether it converged.
    """
    centers, counts = _relative_mag_histogram(detected_flux, dcomp90_detected)
    empty = dict(converged=False, delta=np.nan, delta_offset=np.nan, w=np.nan,
                 sigma=np.nan, residual=np.nan)
    if np.count_nonzero(counts > 0) < MIN_POPULATED_BINS:
        return empty

    res = _fit_free(centers, counts)
    if not res.success:
        return empty

    alpha_hat, m50_hat, w_hat = res.x
    model_values = _model_free(centers, alpha_hat, m50_hat, w_hat)
    _, amplitude = _poisson_nll_and_amplitude(model_values, counts)
    residual = _relative_l1_residual(centers, counts, amplitude * model_values, 0.0)
    delta = DEX_PER_MAG * m50_hat
    delta_offset = m50_hat - Z90 * w_hat
    sigma = _bootstrap_sigma_delta_spitzer(detected_flux, dcomp90_detected)
    return dict(converged=True, delta=delta, delta_offset=delta_offset, w=w_hat,
                sigma=sigma, residual=residual)


def _fit_twomass_band(detected_flux):
    """One 2MASS band's free fit: F50, and its fit residual (no map to
    peg to, so no Delta and no substitution logic).
    """
    centers, counts = _mag_histogram(detected_flux)
    if np.count_nonzero(counts > 0) < MIN_POPULATED_BINS:
        return dict(f50=np.nan, residual=np.nan)

    res = _fit_free(centers, counts)
    if not res.success:
        return dict(f50=np.nan, residual=np.nan)

    alpha_hat, m50_hat, w_hat = res.x
    model_values = _model_free(centers, alpha_hat, m50_hat, w_hat)
    _, amplitude = _poisson_nll_and_amplitude(model_values, counts)
    residual = _relative_l1_residual(centers, counts, amplitude * model_values, m50_hat)
    f50 = 10.0 ** (-DEX_PER_MAG * m50_hat)
    return dict(f50=f50, residual=residual)


def _region_depths(curated_path):
    """One region's fitted Delta, width, bootstrap sigma and fit residual
    per Spitzer band, and fitted F50 and fit residual per 2MASS band.
    """
    with h5py.File(curated_path, "r") as f:
        fnu = f["FNU_MJY"][:]
        dcomp90 = f["DCOMP90_MJY"][:]
        origin = f["ORIGIN_FNU"][:]
        bands = [b.decode() if isinstance(b, bytes) else b for b in f.attrs["BANDS"]]

    delta_dex = np.full(len(IRAC_MIPS_KEYS), np.nan)
    sigma_delta_dex = np.full(len(IRAC_MIPS_KEYS), np.nan)
    w_mag = np.full(len(IRAC_MIPS_KEYS), np.nan)
    fit_residual = np.full(len(IRAC_MIPS_KEYS), np.nan)
    converged = np.zeros(len(IRAC_MIPS_KEYS), dtype=bool)
    for j, key in enumerate(IRAC_MIPS_KEYS):
        b = bands.index(key)
        mask = origin[:, b] == 1
        detected = fnu[mask, b]
        dcomp90_detected = dcomp90[mask, b]
        result = _fit_spitzer_band(detected, dcomp90_detected)
        delta_dex[j] = result["delta"]
        sigma_delta_dex[j] = result["sigma"]
        w_mag[j] = result["w"]
        fit_residual[j] = result["residual"]
        converged[j] = result["converged"]

    f50_2mass = np.full(len(TWOMASS_KEYS), np.nan)
    fit_residual_2mass = np.full(len(TWOMASS_KEYS), np.nan)
    for j, key in enumerate(TWOMASS_KEYS):
        b = bands.index(key)
        detected = fnu[origin[:, b] == 1, b]
        result = _fit_twomass_band(detected)
        f50_2mass[j] = result["f50"]
        fit_residual_2mass[j] = result["residual"]

    return delta_dex, sigma_delta_dex, w_mag, fit_residual, converged, f50_2mass, fit_residual_2mass


def build(config, regions=None):
    """Builds the region-granule survey-depths product for the given
    regions (default: every region in `regions.REGIONS`), parallelised
    over regions with joblib (capped at `config.n_jobs` workers). Reads the
    curated catalogues from `curated.build`, and updates only the given
    regions' rows of the product (`tables.update_rows`).
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

    results = Parallel(n_jobs=config.n_jobs, prefer="threads")(delayed(_region_depths)(p) for p in curated_paths)
    delta_dex = np.array([r[0] for r in results])
    sigma_delta_dex = np.array([r[1] for r in results])
    w_mag = np.array([r[2] for r in results])
    fit_residual = np.array([r[3] for r in results])
    converged = np.array([r[4] for r in results])
    f50_2mass = np.array([r[5] for r in results])
    fit_residual_2mass = np.array([r[6] for r in results])

    substituted = ~converged
    for j in range(len(IRAC_MIPS_KEYS)):
        if converged[:, j].any():
            band_median = np.median(delta_dex[converged[:, j], j])
        else:
            band_median = np.nanmedian(delta_dex[:, j])
        delta_dex[substituted[:, j], j] = band_median

    band_keys = [b.key for b in definitions.BANDS]
    substituted_full = np.zeros((len(regions), len(definitions.BANDS)), dtype=bool)
    for j, key in enumerate(IRAC_MIPS_KEYS):
        substituted_full[:, band_keys.index(key)] = substituted[:, j]

    out_path = config_module.product_path(config, "catalog", "sesna", "depths", "region")
    tables_module.update_rows(
        out_path,
        regions,
        {
            "DELTA_DEX": delta_dex,
            "SIGMA_DELTA_DEX": sigma_delta_dex,
            "W_MAG": w_mag,
            "FIT_RESIDUAL": fit_residual,
            "F50_2MASS_MJY": f50_2mass,
            "FIT_RESIDUAL_2MASS": fit_residual_2mass,
            "SUBSTITUTED": substituted_full,
        },
        granule="region",
    )


if __name__ == "__main__":
    run(build)
