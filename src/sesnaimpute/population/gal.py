"""GAL: the background-galaxy number-counts law (SPEC_PRIORS.md section 5).

This module builds the one survey-wide, region-independent piece of the
background-galaxy prior that a consumer reads: `phi(S)`, the intrinsic
4.5um galaxy number-counts law (Fazio et al. 2004, ApJS 154, 39, Table
1) -- one smooth broken power law in cumulative counts N(>S), fitted
once (`build_counts_law`) and written to the `counts_gal_survey`
product's `LOG10_S_GRID`/`PHI_S`, alongside `PHI_S_POINT = PHI_S *
P_POINT`, the IRAC point-source retention `p(S)` measured in
studies/swire_vs_fazio.md against Fazio's own star counts (`bmstp.
sample_gal`, `bmstp.density` and `bmstp.template_weights` read the
point-source-corrected law). The full selection shape `r(a, log10 S) =
phi(S) . S . eps(a, S)` assembles this law with the extinction/selection
term elsewhere; this module never evaluates `eps`.

The star-galaxy split itself is chosen once, survey-wide
(`select_star_galaxy_split`): among the candidate rules, the one whose
own removed-star count, summed across the six SWIRE (Surace et al. 2005
DR2 release) fields, is closest to Fazio's own star columns transported
to each field's latitude (studies/swire_vs_fazio.md sec 4) -- not the
earlier criterion (surviving galaxy count against Fazio's galaxy law),
which is unreachable once `p(S) < 1`. The split and the counts law are
cross-checked against the star-subtracted SWIRE and S-COSMOS (Sanders
et al. 2007, ApJS 172, 86) counts at fixed fluxes and printed; neither
the split nor these checks are written to the product.

The library register never enters this quantity (SPEC_PRIORS.md section
0.3, C3): it supplies SED templates to the fitter only.
"""

import os

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute.build import run

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: The four IRAC bands GAL's counts law and split read (SPEC_PRIORS.md
#: section 5.1: "the 2MASS bands are excluded -- a galaxy bright enough
#: for 2MASS is a resolved nearby object, negligible at SESNA's depth").
IRAC_BAND_KEYS = ("I1", "I2", "I3", "I4")

#: The counts law's own coordinate band: 4.5um, matching Fazio et al.
#: 2004's Table 1 and SESNA's I2.
COORD_BAND_UM = 4.5

#: Fazio et al. 2004 (ApJS 154, 39, section 2)'s own Vega-system zero-point
#: flux densities, Jy -- Table 1's magnitude abscissa is defined by these,
#: never this project's own general-purpose Vega zero points.
FAZIO_VEGA_ZP_JY = {3.6: 277.5, 4.5: 179.5, 5.8: 116.6, 8.0: 63.1}

#: Table 1's three fields (star-subtracted, completeness-corrected galaxy
#: columns only -- never the `*_total` columns, which still carry stars).
FIELD_COLUMNS = ("bootes_galaxies", "egs_galaxies", "qso1700_galaxies")

#: The four brightest rows of Table 1 fix the analytic power-law tail used
#: to cumulate brighter than the table's own first row (module-local
#: convention, matching every published use of this table in this
#: pipeline).
N_BRIGHT_TAIL_FIT = 4

#: SWIRE's own 5-sigma depths at 3.6/4.5/5.8/8.0um, uJy (SPEC_PRIORS.md
#: section 5.1; Surace et al. 2005, SWIRE Data Release 2). 3-9x below
#: SESNA's survey-wide median 50%-limits, which is why SWIRE stands in for
#: the external galaxy population.
SWIRE_5SIGMA_UJY = {"I1": 4.0, "I2": 6.0, "I3": 48.0, "I4": 40.0}

#: The six SWIRE fields' combined two-band (3.6+4.5um) footprint, deg^2,
#: measured from the catalogue's own occupied 1-arcmin sky cells --
#: not the nominal "49 deg^2" of the release announcement, which is 9%
#: too large for the area every row here actually carries (Surace et al.
#: 2005 DR2 release; studies/swire_vs_fazio.md sec 1 "Footprint").
SWIRE_AREA_DEG2 = 44.8

#: The same six fields' own individual areas, deg^2, SWIRE_FIELD_FILES
#: order -- ELAIS-N1, ELAIS-N2, ELAIS-S1, Lockman, XMM-LSS, CDFS
#: (studies/swire_vs_fazio.md sec 3), summing to SWIRE_AREA_DEG2. Used
#: only to weight the per-field expected star count in
#: `expected_star_count` below.
SWIRE_FIELD_AREA_DEG2 = (8.84, 3.80, 6.11, 10.36, 8.37, 7.29)

#: The same six fields' own |galactic latitude|, deg, SWIRE_FIELD_FILES
#: order (field centres; studies/swire_vs_fazio.md sec 3) -- the target
#: latitudes the Fazio star-count law is transported to.
SWIRE_FIELD_ABS_B_DEG = (44.6, 42.1, 73.1, 52.2, 58.8, 54.6)

#: S-COSMOS's own field area, deg^2 (Sanders et al. 2007, ApJS 172, 86;
#: SPEC_PRIORS.md section 5.1), for the same purpose.
SCOSMOS_AREA_DEG2 = 2.0

SWIRE_FIELD_FILES = (
    "swire_elaisn1.csv", "swire_elaisn2.csv", "swire_elaiss1.csv",
    "swire_lockman.csv", "swire_xmmlss.csv", "swire_cdfs.csv",
)
SWIRE_FLUX_COLUMNS = ("flux_ap2_36", "flux_ap2_45", "flux_ap2_58", "flux_ap2_80")
SWIRE_STELL_COLUMNS = ("stell_36", "stell_45", "stell_58", "stell_80")
SWIRE_EXT_FL_COLUMNS = ("ext_fl_36", "ext_fl_45", "ext_fl_58", "ext_fl_80")

#: SWIRE's per-band extended-source flag, as the DR2 release's own IRSA
#: column documentation defines it (irsa.ipac.caltech.edu, SWIRE
#: Optical-IRAC-MIPS24 catalog definition, "ext_fl_*"): -1 "definitely
#: point-like", 0 "indeterminate", 1 "might be extended", 2 "clearly
#: extended". Star-galaxy separation reads this first, per band.
EXT_FL_POINT = -1
EXT_FL_EXTENDED = (1, 2)

#: The SExtractor point-source stellarity threshold (Bertin & Arnouts 1996,
#: A&AS 117, 393) -- SWIRE's own stellarity index is "generated by the
#: SExtractor software package" (same IRSA documentation), so its CLASS_STAR
#: convention is what breaks the flag's own "indeterminate" (0) category,
#: using the two best-PSF-sampled bands (3.6, 4.5um) only: the
#: documentation itself warns the index "is not reliable at low flux
#: levels", which is worst in 5.8/8.0um where non-detections are common.
STELLARITY_STAR_MIN = 0.9

#: Star-galaxy split candidates (SPEC_PRIORS.md section 5.1, owner ruling
#: 2026-09-05): the split adopted is the one that reproduces Fazio's
#: star-subtracted counts at 0.1-1mJy within the fitted cosmic-variance
#: spread, optical stellarity preferred where the release has it. The
#: SWIRE pull (`sky.download.swire.build.COLUMNS`) carries no optical
#: (measured off the optical image) stellarity column -- only the four
#: per-band IRAC stellarities (`stell_36` etc, SExtractor run on the IRAC
#: images themselves) and the per-band extended flag -- so the "optical
#: preferred" branch never fires here; the stellarity candidate below
#: reads IRAC 3.6um, the best-PSF-sampled band, instead.
SPLIT_STELLARITY_GRID = tuple(np.round(np.arange(0.50, 0.981, 0.02), 2))

#: The three fluxes SPEC_PRIORS.md 5.1 names for the split criterion,
#: 4.5um, mJy.
SPLIT_CRITERION_S_MJY = (0.1, 0.3, 1.0)


#: The shared flux grid PHI_S is tabulated on: 61 points, SWIRE's I2 depth
#: to Fazio's bright end (IMPLEMENTATION.md section 3).
N_S_GRID = 61

#: IRAC point-source retention p(S) = min(1, (S/S0)^-q): the fraction of
#: the intrinsic galaxy law PHI_S an IRAC point-source catalogue actually
#: detects, measured against Fazio+2004's own star counts transported to
#: each SWIRE field's latitude (studies/swire_vs_fazio.md sec 5), rms
#: 0.051 dex over 0.05-0.32mJy; held at 1 below SWIRE's own 90%
#: completeness (0.026mJy), where the galaxy angular-size distribution has
#: already crossed the IRAC PSF and has no mechanism to turn back down.
POINT_RETENTION_S0_MJY = 0.0703
POINT_RETENTION_Q = 0.83
POINT_RETENTION_RMS_DEX = 0.051

#: Fazio et al. 2004's own three fields' |galactic latitude|, deg, in
#: FIELD_COLUMNS/STAR_FIELD_COLUMNS order (studies/swire_vs_fazio.md sec
#: "Expected stars") -- the anchor points the star-count latitude
#: gradient below is measured between.
FAZIO_FIELD_ABS_B_DEG = (67.4, 60.0, 33.6)

#: The star-count latitude gradient those three fields themselves define
#: (studies/swire_vs_fazio.md sec "Expected stars": -0.0072 to -0.0144
#: dex/deg per magnitude across the three fields, mean below), used to
#: transport Fazio's pooled star counts to a SWIRE field's own latitude
#: where no TRILEGAL pointing exists (|b| > 40).
STAR_COUNT_LATITUDE_GRADIENT_DEX_PER_DEG = -0.0115

#: Fazio Table 1's star columns, star-subtracted counts' complement --
#: same table, same rows as FIELD_COLUMNS, the *_stars columns instead of
#: *_galaxies.
STAR_FIELD_COLUMNS = ("bootes_stars", "egs_stars", "qso1700_stars")

#: The flux points the star-galaxy split is graded at, mJy -- the faint
#: range where the threshold actually matters (studies/swire_vs_fazio.md
#: sec 4: brighter than ~0.25mJy no threshold restores the expected star
#: count, because the point-source deficit dominates there, not the
#: split).
SPLIT_CRITERION_STAR_S_MJY = (0.045, 0.071, 0.113, 0.179)


# ---------------------------------------------------------------------------
# 1. The counts law phi(S) -- Fazio et al. 2004, fit once, survey-wide
# ---------------------------------------------------------------------------

class BrokenPowerLaw:
    """A smooth broken power law for cumulative counts N(>S), deg^-2, in
    log-log:

        log10 N(>S) = logA - af*x - (ab-af)*D*log10(1 + 10**(x/D)),
        x = log10(S/S_b)

    with faint-end slope `af`, bright-end slope `ab`, break `S_b` and
    smoothness `D`. `differential` is its analytic derivative, `phi(S) =
    -dN/dS`, never a numerical one. A single power law (`fit_counts`'s
    own fallback when the break is not supported by the data) is the
    same class with `af == ab`: the break/smoothness terms then cancel
    out of the formula exactly, leaving a straight line in log-log, so
    no separate class is needed.

    `log_range`, when given (`fit_counts`'s own bound: the data's
    faintest and brightest tabulated bins), is where the formula was
    actually fitted. Below `log_range[0]` (fainter than Fazio's own
    faintest bin) every method here extrapolates by holding the running
    slope fixed at its own value AT that faintest bin -- a straight
    power-law continuation, not the smooth formula's own further
    curvature past data it was never fit to (owner, 2026-09-06)."""

    PARAM_NAMES = ("log10_A", "log10_S_break", "alpha_faint", "alpha_bright", "smoothness")

    def __init__(self, params, log_range=None):
        self.params = np.asarray(params, dtype=float)
        self.log_range = None if log_range is None else (float(log_range[0]), float(log_range[1]))

    def _raw_log10_cumulative(self, log10_S):
        logA, logSb, af, ab, D = self.params
        u = (log10_S - logSb) / D
        soft = np.logaddexp(0.0, u * np.log(10.0)) / np.log(10.0)
        return logA - af * (log10_S - logSb) - (ab - af) * D * soft

    def _raw_local_slope(self, log10_S):
        logA, logSb, af, ab, D = self.params
        u = (log10_S - logSb) / D
        w = 1.0 / (1.0 + 10.0 ** np.clip(-u, -300, 300))
        return af + (ab - af) * w

    def log10_cumulative(self, log10_S):
        log10_S = np.asarray(log10_S, dtype=float)
        if self.log_range is None:
            return self._raw_log10_cumulative(log10_S)
        xlo = self.log_range[0]
        below = log10_S < xlo
        val = self._raw_log10_cumulative(np.where(below, xlo, log10_S))
        slope_lo = self._raw_local_slope(np.asarray(xlo, dtype=float))
        return np.where(below, val + slope_lo * (xlo - log10_S), val)

    def cumulative(self, S):
        """N(>S), galaxies deg^-2 brighter than flux S (mJy)."""
        return 10.0 ** self.log10_cumulative(np.log10(np.asarray(S, dtype=float)))

    def local_slope(self, log10_S):
        """-d log10 N / d log10 S, the running cumulative slope, held at
        the faintest tabulated bin's own value below `log_range[0]`."""
        log10_S = np.asarray(log10_S, dtype=float)
        if self.log_range is None:
            return self._raw_local_slope(log10_S)
        xlo = self.log_range[0]
        return self._raw_local_slope(np.where(log10_S < xlo, xlo, log10_S))

    def differential(self, S):
        """phi(S) = -dN(>S)/dS, galaxies deg^-2 mJy^-1, positive everywhere."""
        S = np.asarray(S, dtype=float)
        return (self.cumulative(S) / S) * self.local_slope(np.log10(S))


def read_fazio_table(path, band_um=COORD_BAND_UM, columns=FIELD_COLUMNS):
    """One band's block of the Fazio et al. 2004 Table 1 CSV: `(mag,
    {column: log10 differential counts})`, the table's own 0.5-mag bins,
    absent field/magnitude combinations left NaN. `columns` defaults to
    the star-subtracted galaxy columns (FIELD_COLUMNS); pass
    STAR_FIELD_COLUMNS for the table's own star columns instead.
    """
    df = pd.read_csv(path)
    blk = df[df["band_um"] == band_um].sort_values("mag")
    if blk.empty:
        raise ValueError(f"gal: no rows for band_um={band_um!r} in {path!r}")
    return blk["mag"].to_numpy(dtype=float), {c: blk[c].to_numpy(dtype=float) for c in columns}


def cumulative_from_differential(mag, log10_n, band_um=COORD_BAND_UM,
                                  n_bright_fit=N_BRIGHT_TAIL_FIT):
    """Integrates log10 differential counts per magnitude into cumulative
    `N(>S)`: trapezoid over the table's own populated rows, plus an
    analytic power-law tail brighter than the first row (slope fit to the
    `n_bright_fit` brightest rows). Returns `(log10_S, log10_N, tail)`, `S`
    in mJy on Fazio's own Vega zero point.
    """
    ok = np.isfinite(log10_n)
    m = np.asarray(mag, dtype=float)[ok]
    n = 10.0 ** np.asarray(log10_n, dtype=float)[ok]
    if m.size < n_bright_fit + 2:
        raise ValueError("gal: too few populated Fazio magnitude rows to cumulate")
    slope = np.polyfit(m[:n_bright_fit], np.log10(n[:n_bright_fit]), 1)[0]
    if slope <= 0:
        raise ValueError("gal: Fazio bright-end counts must rise with magnitude")
    tail = n[0] / (np.log(10.0) * slope)
    cum = np.concatenate([[tail], tail + np.cumsum(0.5 * (n[1:] + n[:-1]) * np.diff(m))])
    zp_mjy = FAZIO_VEGA_ZP_JY[band_um] * 1e3
    log10_S = np.log10(zp_mjy * 10.0 ** (-0.4 * m))
    return log10_S, np.log10(cum), float(tail)


#: `fit_counts`'s own multi-start grid: at least eight break starts,
#: log-spaced (linear in log10 S) across Fazio's own tabulated range, and
#: several starting slope pairs -- Fazio's 4.5um counts break INSIDE the
#: tabulated data, near 50-100 uJy (owner, 2026-09-06): a single-start
#: constrained fit landing on the range's edge is an optimiser failure,
#: not evidence against a break, so many starts are tried and the best
#: (lowest rms) kept.
_BREAK_START_N = 8
_SLOPE_STARTS = ((0.3, 1.2), (0.6, 1.8), (1.0, 2.5))

#: The bar an in-range break must clear (owner, 2026-09-06): within this
#: many dex of the earlier, unconstrained-break fit's own rms (0.045,
#: `_REFERENCE_RMS_DEX`) -- otherwise the in-range search has genuinely
#: failed to find a break the data support, and `fit_counts` reverts to
#: that earlier fit instead of accepting a worse one.
_RMS_TOLERANCE_DEX = 0.01
_REFERENCE_RMS_DEX = 0.045


def _fit_one_start(x, y, logSb0, af0, ab0, break_bounds):
    """One `least_squares` run of `BrokenPowerLaw` from one starting
    break/slope pair, break bounded to `break_bounds`. Returns
    `(params, rms_dex)`."""
    p0 = [float(y.max()), float(logSb0), float(af0), float(ab0), 0.5]

    def resid(p):
        return BrokenPowerLaw(p).log10_cumulative(x) - y

    sol = least_squares(resid, p0, bounds=(
        [0.0, break_bounds[0], -10.0, 0.3, 0.05], [9.0, break_bounds[1], 10.0, 5.0, 4.0]))
    r = resid(sol.x)
    return sol.x, float(np.sqrt(np.mean(r ** 2)))


def fit_counts(log10_S, log10_N):
    """Least-squares fit of `BrokenPowerLaw` to a cumulative counts
    curve. First tries the break constrained INSIDE the data's own
    tabulated flux range (`x.min()`..`x.max()`, Fazio's faintest and
    brightest bins), from `_BREAK_START_N` log-spaced break starts times
    `_SLOPE_STARTS` slope-pair starts, keeping the best (lowest rms) of
    all of them (owner, 2026-09-06: one start landing on the edge is an
    optimiser failure, not evidence against an in-range break -- Fazio's
    own 4.5um counts break inside the tabulated data).

    If the best in-range fit's rms is not within `_RMS_TOLERANCE_DEX` of
    `_REFERENCE_RMS_DEX` (the earlier, unconstrained-break fit's own
    0.045 dex), no in-range break actually reproduces the data as well:
    reverts to that earlier fit (break free to sit below the range) --
    read only inside the flux grid actually built on it (`LOG10_S_GRID`,
    SWIRE's depth to Fazio's own bright end, which sits entirely inside
    `x.min()`..`x.max()`), where it reproduces Fazio's own counts to
    0.045 dex; the held-slope extrapolation below `x.min()` (module
    docstring) is never reached by that grid.

    Returns `(fit, stats)`, `stats` carrying the fit's own residual in
    dex, the data range, and whether the revert fired."""
    x, y = np.asarray(log10_S, dtype=float), np.asarray(log10_N, dtype=float)
    x_lo, x_hi = float(x.min()), float(x.max())

    break_starts = np.linspace(x_lo, x_hi, _BREAK_START_N)
    best_params, best_rms = None, np.inf
    for logSb0 in break_starts:
        for af0, ab0 in _SLOPE_STARTS:
            params, rms = _fit_one_start(x, y, logSb0, af0, ab0, (x_lo, x_hi))
            if rms < best_rms:
                best_params, best_rms = params, rms

    reverted = best_rms > _REFERENCE_RMS_DEX + _RMS_TOLERANCE_DEX
    if reverted:
        params, _rms = _fit_one_start(x, y, float(np.clip(x.mean(), x_lo - 5.0, x_hi)), 0.6, 1.8,
                                      (x_lo - 5.0, x_hi))
    else:
        params = best_params

    fit = BrokenPowerLaw(params, log_range=(x_lo, x_hi))
    r = fit.log10_cumulative(x) - y
    return fit, {"rms_dex": float(np.sqrt(np.mean(r ** 2))), "max_abs_dex": float(np.max(np.abs(r))),
                "log10_s_lo": x_lo, "log10_s_hi": x_hi, "single_power_law": False,
                "reverted": bool(reverted), "best_in_range_rms_dex": float(best_rms)}


def fit_all_variants(path):
    """The central (all-fields-pooled) fit and the two field-preferred
    variants, each `(fit, stats, (log10_S, log10_N))`. "mean" averages
    log10 of whichever galaxy columns are populated per magnitude row --
    the pooled fit `build_counts_law` adopts. "egs"/"qso1700" prefer that
    field, falling back to Bootes where it is silent -- their spread is
    the cosmic-variance measurement (`cosmic_variance_dex`).
    """
    mag, cols = read_fazio_table(path)
    stack = np.vstack([cols[c] for c in FIELD_COLUMNS])
    with np.errstate(invalid="ignore"):
        mean_log = np.nanmean(stack, axis=0)
    variants = {
        "mean": mean_log,
        "egs": np.where(np.isnan(cols["egs_galaxies"]), cols["bootes_galaxies"], cols["egs_galaxies"]),
        "qso1700": np.where(np.isnan(cols["qso1700_galaxies"]), cols["bootes_galaxies"], cols["qso1700_galaxies"]),
    }
    out = {}
    for key, log10_n in variants.items():
        log10_S, log10_N, _tail = cumulative_from_differential(mag, log10_n)
        fit, stats = fit_counts(log10_S, log10_N)
        out[key] = (fit, stats, (log10_S, log10_N))
    return out


def cosmic_variance_dex(variants):
    """The counts law's field-to-field spread (SPEC_PRIORS.md section 5.1,
    "about 0.1 dex"): the RMS difference in log10 N(>S) between the two
    field-preferred fits, over the flux range both actually constrain.
    """
    fit_egs = variants["egs"][0]
    fit_qso = variants["qso1700"][0]
    lo = max(fit_egs.log_range[0], fit_qso.log_range[0])
    hi = min(fit_egs.log_range[1], fit_qso.log_range[1])
    grid = np.linspace(lo, hi, 50)
    diff = fit_egs.log10_cumulative(grid) - fit_qso.log10_cumulative(grid)
    return float(np.sqrt(np.mean(diff ** 2)))


def fazio_bright_end_log10_s(fazio_path):
    """log10 S (mJy) of Table 1's own brightest coordinate-band row --
    the LOG10_S_GRID's bright edge (IMPLEMENTATION.md section 3)."""
    mag, _cols = read_fazio_table(fazio_path)
    zp_mjy = FAZIO_VEGA_ZP_JY[COORD_BAND_UM] * 1e3
    return float(np.log10(zp_mjy * 10.0 ** (-0.4 * mag.min())))


def build_log10_s_grid(fazio_path):
    """The 61-point LOG10_S_GRID, SWIRE's I2 5-sigma depth to Fazio's
    bright end (IMPLEMENTATION.md section 3)."""
    lo = np.log10(SWIRE_5SIGMA_UJY["I2"] / 1000.0)
    hi = fazio_bright_end_log10_s(fazio_path)
    return np.linspace(lo, hi, N_S_GRID)


def point_source_retention(s_mjy, s0=POINT_RETENTION_S0_MJY, q=POINT_RETENTION_Q):
    """p(S) = min(1, (S/S0)^-q), the IRAC point-source retention of the
    intrinsic galaxy law (module docstring; studies/swire_vs_fazio.md sec
    5, rms POINT_RETENTION_RMS_DEX dex over 0.05-0.32mJy)."""
    s = np.asarray(s_mjy, dtype=float)
    return np.minimum(1.0, (s / s0) ** (-q))


def build_counts_law(fazio_path):
    """The pooled fit, its cosmic-variance band, PHI_S on LOG10_S_GRID,
    and the point-source-corrected PHI_S_POINT = PHI_S * p(S)
    (studies/swire_vs_fazio.md sec 6, fix 1). Returns a dict ready for
    `write_counts`."""
    variants = fit_all_variants(fazio_path)
    fit, stats, _ = variants["mean"]
    cv_dex = cosmic_variance_dex(variants)
    log10_s_grid = build_log10_s_grid(fazio_path)
    phi_s = fit.differential(10.0 ** log10_s_grid)
    p_point = point_source_retention(10.0 ** log10_s_grid)
    phi_s_point = phi_s * p_point
    return dict(fit=fit, stats=stats, cosmic_variance_dex=cv_dex,
                log10_s_grid=log10_s_grid, phi_s=phi_s, p_point=p_point,
                phi_s_point=phi_s_point, variants=variants)


def write_counts(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fit = result["fit"]
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        for name, val in zip(BrokenPowerLaw.PARAM_NAMES, fit.params):
            f.create_dataset(name.upper(), data=np.float64(val))
        f.create_dataset("LOG10_S_GRID", data=result["log10_s_grid"].astype(np.float64))
        f.create_dataset("PHI_S", data=result["phi_s"].astype(np.float64))
        f.create_dataset("P_POINT", data=result["p_point"].astype(np.float64))
        f.create_dataset("PHI_S_POINT", data=result["phi_s_point"].astype(np.float64))
        f.create_dataset("COSMIC_VARIANCE_DEX", data=np.float64(result["cosmic_variance_dex"]))
        f.create_dataset("FIT_LOG10_S_MIN", data=np.float64(fit.log_range[0]))
        f.create_dataset("FIT_LOG10_S_MAX", data=np.float64(fit.log_range[1]))
        f.create_dataset("FIT_RMS_DEX", data=np.float64(result["stats"]["rms_dex"]))


# ---------------------------------------------------------------------------
# 2. The star-galaxy split -- SWIRE, once, survey-wide
# ---------------------------------------------------------------------------

def read_swire_catalogue(config):
    """The six SWIRE fields concatenated: `(flux_mjy, stell, ext_fl,
    field)`, the first three `(n, 4)` in `IRAC_BAND_KEYS` order, `field`
    `(n,)` the row's index into `SWIRE_FIELD_FILES`/`SWIRE_FIELD_AREA_DEG2`/
    `SWIRE_FIELD_ABS_B_DEG` (`select_star_galaxy_split`'s per-field star
    count needs it). Fluxes converted from SWIRE's own aperture-2 uJy to
    this project's mJy convention.
    """
    dest_dir = f"{config.data_root}/sky/download/swire"
    cols = list(SWIRE_FLUX_COLUMNS) + list(SWIRE_STELL_COLUMNS) + list(SWIRE_EXT_FL_COLUMNS)
    frames, field_idx = [], []
    for i, name in enumerate(SWIRE_FIELD_FILES):
        path = f"{dest_dir}/{name}"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"gal: no SWIRE field catalogue at {path!r} -- run the "
                f"'sesnaimpute.sky.download.swire.build' RUNBOOK line")
        df_i = pd.read_csv(path, usecols=cols)
        frames.append(df_i)
        field_idx.append(np.full(len(df_i), i, dtype=np.int8))
    df = pd.concat(frames, ignore_index=True)
    flux_mjy = df[list(SWIRE_FLUX_COLUMNS)].to_numpy(dtype=float) / 1000.0
    stell = df[list(SWIRE_STELL_COLUMNS)].to_numpy(dtype=float)
    ext_fl = df[list(SWIRE_EXT_FL_COLUMNS)].to_numpy(dtype=float)
    field = np.concatenate(field_idx)
    return flux_mjy, stell, ext_fl, field


def classify_galaxy_extended_flag(stell, ext_fl):
    """`(n,)` bool: which SWIRE rows are galaxies, by the release's own
    per-band extended flag (3.6, 4.5um only -- the two best-PSF-sampled
    bands), falling back to the SExtractor stellarity threshold only where
    the flag itself is "indeterminate" in both bands. Split candidate (b):
    the release's extended flag, as the earlier construction used it.
    """
    ext36, ext45 = ext_fl[:, 0], ext_fl[:, 1]
    st36, st45 = stell[:, 0], stell[:, 1]
    extended_override = np.isin(ext36, EXT_FL_EXTENDED) | np.isin(ext45, EXT_FL_EXTENDED)
    point_flagged = (~extended_override) & ((ext36 == EXT_FL_POINT) | (ext45 == EXT_FL_POINT))
    indeterminate = (~extended_override) & (~point_flagged)
    star_by_stellarity = indeterminate & (
        (np.isfinite(st36) & (st36 > STELLARITY_STAR_MIN))
        | (np.isfinite(st45) & (st45 > STELLARITY_STAR_MIN))
    )
    is_star = point_flagged | star_by_stellarity
    return ~is_star


def classify_galaxy_stellarity(stell, threshold):
    """`(n,)` bool: which SWIRE rows are galaxies by IRAC 3.6um
    stellarity alone -- the "optical stellarity where available, else
    IRAC 3.6um" rule (SPEC_PRIORS.md 5.1), with no optical column in this
    release's pull (see the module docstring). A row is a star if its
    3.6um stellarity is measured and at or above `threshold`; an
    unmeasured stellarity is kept as a galaxy, the same conservative
    default `classify_galaxy_extended_flag` uses for an unresolved flag.
    Split candidate (a), one point of `SPLIT_STELLARITY_GRID`.
    """
    st36 = stell[:, 0]
    is_star = np.isfinite(st36) & (st36 >= threshold)
    return ~is_star


def classify_galaxy_no_removal(n):
    """`(n,)` bool, all `True`: split candidate (c), no star removal at
    all -- the "also reported with no removal" branch of SPEC_PRIORS.md
    5.1, and the population `EPS_NO_REMOVAL` is built on.
    """
    return np.ones(n, dtype=bool)


def candidate_star_galaxy_splits(stell, ext_fl):
    """`{label: is_galaxy}` over every split candidate SPEC_PRIORS.md 5.1
    names: `"no_removal"` (c), `"extended_flag"` (b), and one
    `"stellarity_t=..."` (a) per `SPLIT_STELLARITY_GRID` point.
    """
    n = stell.shape[0]
    out = {
        "no_removal": classify_galaxy_no_removal(n),
        "extended_flag": classify_galaxy_extended_flag(stell, ext_fl),
    }
    for t in SPLIT_STELLARITY_GRID:
        out[f"stellarity_t={t:.2f}"] = classify_galaxy_stellarity(stell, t)
    return out


def fazio_pooled_star_counts(fazio_path):
    """`(log10_S, log10_N)`, Fazio's pooled (mean-of-three-fields) star
    `N(>S)`, the star-count analogue of `fit_all_variants`'s "mean"
    galaxy variant: mean of `log10` over `STAR_FIELD_COLUMNS`'s populated
    rows, cumulated the same way (`cumulative_from_differential`). The
    anchor `expected_star_count` transports to each SWIRE field's own
    latitude.
    """
    mag, cols = read_fazio_table(fazio_path, columns=STAR_FIELD_COLUMNS)
    stack = np.vstack([cols[c] for c in STAR_FIELD_COLUMNS])
    with np.errstate(invalid="ignore"):
        mean_log = np.nanmean(stack, axis=0)
    log10_S, log10_N, _tail = cumulative_from_differential(mag, mean_log)
    return log10_S, log10_N


def expected_star_count(fazio_path, s_mjy):
    """The total expected star COUNT (not density) above `s_mjy` summed
    across the six SWIRE fields: Fazio's pooled star `N(>S)`
    (`fazio_pooled_star_counts`) transported to each field's own
    |b| (`SWIRE_FIELD_ABS_B_DEG`) from the three Fazio fields' own mean
    |b| by `STAR_COUNT_LATITUDE_GRADIENT_DEX_PER_DEG`, times that field's
    own area (`SWIRE_FIELD_AREA_DEG2`) -- the "expected stars" of
    studies/swire_vs_fazio.md sec 2-3, summed rather than reported per
    field, since a raw count, not a fraction, is the quantity the split
    actually controls (sec 4).
    """
    log10_S, log10_N = fazio_pooled_star_counts(fazio_path)
    # cumulative_from_differential runs bright-to-faint (decreasing log10_S,
    # the table's own row order); np.interp needs its x-coordinate
    # increasing, so read it faint-to-bright here.
    order = np.argsort(log10_S)
    log10_n0_deg2 = float(np.interp(np.log10(s_mjy), log10_S[order], log10_N[order]))
    b_fazio_mean = float(np.mean(FAZIO_FIELD_ABS_B_DEG))
    b_field = np.asarray(SWIRE_FIELD_ABS_B_DEG, dtype=float)
    area_field = np.asarray(SWIRE_FIELD_AREA_DEG2, dtype=float)
    shift = STAR_COUNT_LATITUDE_GRADIENT_DEX_PER_DEG * (b_field - b_fazio_mean)
    density_field = 10.0 ** (log10_n0_deg2 + shift)
    return float(np.sum(density_field * area_field))


def select_star_galaxy_split(flux_mjy_i2, stell, ext_fl, fazio_path, tolerance_dex,
                              s_values=SPLIT_CRITERION_STAR_S_MJY):
    """Grades every `candidate_star_galaxy_splits` candidate on absolute
    star counts (studies/swire_vs_fazio.md sec 4, fix 4): each
    candidate's own removed-star count, summed across all six SWIRE
    fields, against `expected_star_count` -- Fazio's own star columns
    transported to each field's own latitude -- at `s_values`, in dex.
    Not the earlier criterion (the candidate's surviving GALAXY count
    against the Fazio galaxy law): that is unachievable once the
    point-source retention p(S) < 1, which is why the earlier search
    always landed on the stellarity grid's own edge.

    A candidate "passes" if every one of those `|dex|` is within
    `tolerance_dex`. The adopted split is the passing candidate whose
    counts are closest to the expectation (smallest mean `|dex|` -- the
    split that most nearly "restores the expected star count", not the
    least intervention); if none passes, the candidate with the smallest
    maximum `|dex|`, flagged as such.

    Returns `(adopted_label, is_galaxy_adopted, rows, none_passed)`,
    `rows` a list of dicts (`label`, `n_removed`, `ratio` per `s_values`,
    `dex` per `s_values`, `max_abs_dex`, `passed`), one per candidate, in
    `candidate_star_galaxy_splits` order.
    """
    exp_n = np.array([expected_star_count(fazio_path, s) for s in s_values])
    candidates = candidate_star_galaxy_splits(stell, ext_fl)

    rows = []
    for label, is_galaxy in candidates.items():
        is_star = ~is_galaxy
        obs_n = np.array([
            float(np.sum(is_star & np.isfinite(flux_mjy_i2) & (flux_mjy_i2 >= s)))
            for s in s_values
        ])
        obs_n_safe = np.where(obs_n > 0, obs_n, np.nan)
        dex = np.log10(obs_n_safe) - np.log10(exp_n)
        # a candidate with zero observed stars at every s_value (e.g.
        # "no_removal") cannot be graded -- infinite deviation, not NaN
        # (a NaN key breaks min()'s ordering: nan compares False against
        # everything, so the first-seen candidate would wrongly "win").
        abs_dex_filled = np.where(np.isfinite(dex), np.abs(dex), np.inf)
        rows.append(dict(
            label=label, n_removed=int(is_star.sum()),
            ratio=(obs_n / exp_n).tolist(), dex=abs_dex_filled.tolist(),
            max_abs_dex=float(np.max(abs_dex_filled)),
            passed=bool(np.all(abs_dex_filled <= tolerance_dex)),
        ))

    passing = [r for r in rows if r["passed"]]
    none_passed = len(passing) == 0
    pool = passing if not none_passed else rows
    best = min(pool, key=lambda r: r["max_abs_dex"])
    return best["label"], candidates[best["label"]], rows, none_passed


# ---------------------------------------------------------------------------
# report-only checks (SPEC_PRIORS.md section 5.1 last rows, "Checks")
# ---------------------------------------------------------------------------

def swire_band_cumulative(flux_mjy_band, is_galaxy, threshold_mjy):
    """SWIRE's own measured N(>S) at `threshold_mjy`, galaxies deg^-2 --
    the "against the four-band catalogue's own counts" check, and the
    grading criterion `select_star_galaxy_split` uses."""
    good = is_galaxy & np.isfinite(flux_mjy_band) & (flux_mjy_band >= threshold_mjy)
    return float(np.sum(good)) / SWIRE_AREA_DEG2


def star_ratio_per_field(flux_mjy_i2, is_star, field, fazio_path, s_mjy):
    """Report-only (studies/swire_vs_fazio.md sec 3's "star/exp" column,
    per SWIRE field, at N(>`s_mjy`)): the adopted split's own star count
    in each field against `expected_star_count`'s own per-field term
    (not its cross-field sum). Returns a list of `(field_name, obs, exp,
    ratio)`, `SWIRE_FIELD_FILES` order.
    """
    log10_S, log10_N = fazio_pooled_star_counts(fazio_path)
    order = np.argsort(log10_S)
    log10_n0 = float(np.interp(np.log10(s_mjy), log10_S[order], log10_N[order]))
    b_fazio_mean = float(np.mean(FAZIO_FIELD_ABS_B_DEG))
    rows = []
    for i, name in enumerate(SWIRE_FIELD_FILES):
        in_field = field == i
        obs = float(np.sum(is_star[in_field] & np.isfinite(flux_mjy_i2[in_field])
                            & (flux_mjy_i2[in_field] >= s_mjy)))
        shift = STAR_COUNT_LATITUDE_GRADIENT_DEX_PER_DEG * (SWIRE_FIELD_ABS_B_DEG[i] - b_fazio_mean)
        exp = (10.0 ** (log10_n0 + shift)) * SWIRE_FIELD_AREA_DEG2[i]
        rows.append((name, obs, exp, obs / exp if exp > 0 else float("nan")))
    return rows


def scosmos_8um_cumulative(config, threshold_mjy):
    """S-COSMOS's own measured 8um N(>S) at `threshold_mjy`, deg^-2, over
    rows the release's own overall quality flag passes (`flag == 0`) --
    S-COSMOS carries no native stellarity (S-L5), so no star-galaxy
    separation is applied here; it is the deep single-field cross-check,
    not the selection population.
    """
    path = f"{config.data_root}/sky/download/scosmos/scosmos_irac.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"gal: no S-COSMOS catalogue at {path!r} -- run the "
            f"'sesnaimpute.sky.download.scosmos.build' RUNBOOK line")
    df = pd.read_csv(path, usecols=["flux_c4_2", "flag"])
    flux_mjy = df["flux_c4_2"].to_numpy(dtype=float) / 1000.0
    good = (df["flag"].to_numpy(dtype=float) == 0) & np.isfinite(flux_mjy) & (flux_mjy >= threshold_mjy)
    return float(np.sum(good)) / SCOSMOS_AREA_DEG2


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Writes the survey-wide galaxy number-counts law (`phi(S)` on
    `LOG10_S_GRID`) and prints the star-galaxy split with its
    consistency checks against SWIRE and S-COSMOS. Survey-wide;
    `regions` is accepted and ignored.
    """
    st = progress.Stage("prior.gal")

    fazio_path = f"{config.data_root}/sky/download/fazio2004/fazio2004_table1_irac_counts.csv"
    if not os.path.exists(fazio_path):
        raise FileNotFoundError(
            f"gal: no Fazio 2004 table at {fazio_path!r} -- see the "
            f"'sky/download/fazio2004' RUNBOOK comment line: manual "
            f"acquisition, no reachable download URL")

    counts_result = build_counts_law(fazio_path)
    counts_path = config_module.product_path(config, "population", "gal", "counts", "survey")
    write_counts(counts_path, counts_result)
    fit = counts_result["fit"]
    stats = counts_result["stats"]
    print(f"gal: counts law: log10_A={fit.params[0]:.4f} log10_S_break={fit.params[1]:.4f} "
          f"alpha_faint={fit.params[2]:.4f} alpha_bright={fit.params[3]:.4f} "
          f"smoothness={fit.params[4]:.4f} rms={stats['rms_dex']:.4f} dex "
          f"data_range=[{stats['log10_s_lo']:.4f}, {stats['log10_s_hi']:.4f}] "
          f"reverted={stats['reverted']} best_in_range_rms={stats['best_in_range_rms_dex']:.4f} dex "
          f"cosmic_variance={counts_result['cosmic_variance_dex']:.4f} dex -> {counts_path}")
    if stats["reverted"]:
        print(f"gal: counts law: the best in-range break's own rms ({stats['best_in_range_rms_dex']:.4f} dex) "
              f"did not reach the earlier fit's {_REFERENCE_RMS_DEX:.4f} dex within {_RMS_TOLERANCE_DEX:.4f} dex "
              "-- reverted to the earlier fit (break free to sit below the data range); the law is read "
              "only inside the flux grid actually built on it (LOG10_S_GRID), which sits entirely inside "
              "the data range and where this fit reproduces Fazio's own counts")
    else:
        print(f"gal: counts law: the break sits inside Fazio's own tabulated range "
              f"(log10_S_break={fit.params[1]:.4f} in [{stats['log10_s_lo']:.4f}, {stats['log10_s_hi']:.4f}])")
    print(f"gal: counts law: below the data range's own faint edge (log10 S={stats['log10_s_lo']:.4f}), "
          "the law is extrapolated at that edge's own running slope, held fixed -- "
          "not the smooth formula's further curvature past data it was never fit to")

    flux_mjy_all, stell_all, ext_fl_all, field_all = read_swire_catalogue(config)
    print("gal: SWIRE pull carries no optical stellarity column (sky.download.swire.build.COLUMNS); "
          "the stellarity split candidate reads IRAC 3.6um in its place")

    split_label, is_galaxy_all, split_rows, none_passed = select_star_galaxy_split(
        flux_mjy_all[:, IRAC_BAND_KEYS.index("I2")], stell_all, ext_fl_all,
        fazio_path, counts_result["cosmic_variance_dex"])
    print("gal: star-galaxy split candidates (absolute star count against Fazio+2004's own star "
          f"columns, latitude-transported, at S={SPLIT_CRITERION_STAR_S_MJY} mJy, tolerance ="
          f" {counts_result['cosmic_variance_dex']:.4f} dex):")
    for r in split_rows:
        ratio_str = ", ".join(f"{s}mJy={ratio:.3f}" for s, ratio in zip(SPLIT_CRITERION_STAR_S_MJY, r["ratio"]))
        print(f"gal:   {r['label']}: n_removed={r['n_removed']} ratio(swire_star/fazio_exp*) [{ratio_str}] "
              f"max|dex|={r['max_abs_dex']:.4f} passed={r['passed']}")
    if none_passed:
        print(f"gal: no split candidate restores Fazio's expected star count within tolerance; "
              f"adopting the smallest max|dex| candidate: {split_label}")
    else:
        print(f"gal: adopted split: {split_label} (closest to the expected star count among candidates "
              "meeting the standard)")

    n_star = int((~is_galaxy_all).sum())
    print(f"gal: SWIRE: {flux_mjy_all.shape[0]} rows, {is_galaxy_all.sum()} classed galaxy under "
          f"{split_label}, {n_star} classed star")

    field_rows = star_ratio_per_field(flux_mjy_all[:, IRAC_BAND_KEYS.index("I2")], ~is_galaxy_all,
                                       field_all, fazio_path, min(SPLIT_CRITERION_STAR_S_MJY))
    pooled_obs = sum(r[1] for r in field_rows)
    pooled_exp = sum(r[2] for r in field_rows)
    print(f"gal: adopted split star/exp* at S>{min(SPLIT_CRITERION_STAR_S_MJY)}mJy per field: " +
          ", ".join(f"{name}={ratio:.2f}" for name, obs, exp, ratio in field_rows) +
          f"; pooled={pooled_obs / pooled_exp:.2f}")

    p_at = np.array([0.03, 0.07, 0.1, 0.3, 1.0])
    p_vals = point_source_retention(p_at)
    print("gal: point-source retention p(S) = min(1, (S/{:.4f})^-{:.2f}) at S={} mJy: p={}".format(
        POINT_RETENTION_S0_MJY, POINT_RETENTION_Q, p_at.tolist(), np.round(p_vals, 4).tolist()))
    below_s0 = 10.0 ** counts_result["log10_s_grid"] <= POINT_RETENTION_S0_MJY
    print(f"gal: PHI_S_POINT <= PHI_S everywhere: "
          f"{bool(np.all(counts_result['phi_s_point'] <= counts_result['phi_s']))}; "
          f"equal below {POINT_RETENTION_S0_MJY}mJy: "
          f"{bool(np.allclose(counts_result['phi_s_point'][below_s0], counts_result['phi_s'][below_s0]))}")

    s_grid = 10.0 ** counts_result["log10_s_grid"]
    d_log10_s = float(counts_result["log10_s_grid"][1] - counts_result["log10_s_grid"][0])
    a_gal_before = float(np.sum(counts_result["phi_s"] * s_grid * d_log10_s * np.log(10.0)))
    a_gal_after = float(np.sum(counts_result["phi_s_point"] * s_grid * d_log10_s * np.log(10.0)))
    print(f"gal: A_GAL (sum phi(S) S dS, deg^-2): before p(S)={a_gal_before:.1f}, after={a_gal_after:.1f}")

    swire_45_01 = swire_band_cumulative(flux_mjy_all[:, IRAC_BAND_KEYS.index("I2")], is_galaxy_all, 0.1)
    swire_45_1 = swire_band_cumulative(flux_mjy_all[:, IRAC_BAND_KEYS.index("I2")], is_galaxy_all, 1.0)
    fazio_45_01 = fit.cumulative(0.1)
    fazio_45_1 = fit.cumulative(1.0)
    print(f"gal: Fazio/SWIRE 4.5um N(>0.1mJy): fazio={fazio_45_01:.1f} swire={swire_45_01:.1f} "
          f"ratio={fazio_45_01 / swire_45_01:.3f}")
    print(f"gal: Fazio/SWIRE 4.5um N(>1mJy): fazio={fazio_45_1:.1f} swire={swire_45_1:.1f} "
          f"ratio={fazio_45_1 / swire_45_1:.3f}")

    swire_80_01 = swire_band_cumulative(flux_mjy_all[:, IRAC_BAND_KEYS.index("I4")], is_galaxy_all, 0.1)
    scosmos_80_01 = scosmos_8um_cumulative(config, 0.1)
    print(f"gal: S-COSMOS/SWIRE 8um N(>0.1mJy): scosmos={scosmos_80_01:.1f} swire={swire_80_01:.1f} "
          f"ratio={scosmos_80_01 / swire_80_01:.3f}" if swire_80_01 > 0 else
          "gal: S-COSMOS/SWIRE 8um N(>0.1mJy): swire count is zero, ratio undefined")

    st.done(counts_path, rms_dex=stats["rms_dex"], cosmic_variance_dex=counts_result["cosmic_variance_dex"])


if __name__ == "__main__":
    run(build)
