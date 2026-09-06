"""GAL: the background-galaxy prior (SPEC_PRIORS.md section 5).

A background galaxy sits behind the *entire* dust column on its sightline
(`a = A_s`, no separate placement) -- geometry alone separates it from a
cloud member, which never carries the full column. What this module builds
is therefore two survey machinery pieces, assembled into the shape
`r(a, log10 S) = phi(S) . S . eps(a, S)` only at prior-table evaluation time
(never stored here):

1. `phi(S)`, the intrinsic 4.5um galaxy number-counts law (Fazio et al.
   2004, ApJS 154, 39, Table 1): one smooth broken power law in cumulative
   counts N(>S), fitted once, region-independent (`build_counts_law`,
   written to the `counts/survey` product).
2. `EPS[n, n_x, n_s]`, the exact per-source selection: for each catalogued
   source, on the shared scaled-extinction ladder `X_LADDER` (`a_query =
   X_LADDER * A_s`, a background galaxy carrying the whole column) by
   `LOG10_S_GRID`, the fraction of an external, four-band galaxy
   population (SWIRE, Surace et al. 2005 DR2 release), its stars removed,
   that clears any two of the four IRAC bands at *that source's own*
   eight limits (`build_source_selection`, written to the `prior/
   selection/source` product, one file per region, SPEC_PRIORS.md
   section 1.3). Two comparison variants are kept at the region's own
   median 8-band limit, on the column-grid nodes, in the region-level
   `prior/region` product: `EPS_2BAND` requires *both* 3.6 and 4.5um --
   the SEDS-era two-band form; `EPS_NO_REMOVAL` is the four-band test
   with no star removal at all (SPEC_PRIORS.md section 5, "Checks").

The star-galaxy split itself is chosen once, survey-wide
(`select_star_galaxy_split`): among the candidate rules that reproduce
Fazio's star-subtracted counts at 0.1-1mJy within the fitted
cosmic-variance spread, the one removing the fewest SWIRE rows.

The library register never enters either quantity (SPEC_PRIORS.md section
0.3, C3): it supplies SED templates to the fitter only.
"""

import os

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid as column_grid_module
from sesnaimpute.prior import selection as selection_module

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

BAND_KEYS = tuple(b.key for b in definitions.BANDS)

#: The four IRAC bands GAL's selection reads (SPEC_PRIORS.md section 5.1:
#: "the 2MASS bands are excluded -- a galaxy bright enough for 2MASS is a
#: resolved nearby object, negligible at SESNA's depth").
IRAC_BAND_KEYS = ("I1", "I2", "I3", "I4")
IRAC_BAND_IDX = np.array([BAND_KEYS.index(b) for b in IRAC_BAND_KEYS])

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

#: The six SWIRE fields' combined sky area, deg^2 (Lonsdale et al. 2003,
#: PASP 115, 897; Surace et al. 2005 DR2 release -- "six blank fields,
#: 49 deg^2", SPEC_PRIORS.md section 5.1), for turning a raw SWIRE galaxy
#: count into a surface density for the report-only counts-law check.
SWIRE_AREA_DEG2 = 49.0

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
SPLIT_STELLARITY_GRID = tuple(np.round(np.arange(0.50, 0.951, 0.05), 2))

#: The three fluxes SPEC_PRIORS.md 5.1 names for the split criterion,
#: 4.5um, mJy.
SPLIT_CRITERION_S_MJY = (0.1, 0.3, 1.0)


#: The shared flux grid every eps table and PHI_S is tabulated on: 61
#: points, SWIRE's I2 depth to Fazio's bright end (IMPLEMENTATION.md
#: section 3).
N_S_GRID = 61


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
    -dN/dS`, never a numerical one.
    """

    PARAM_NAMES = ("log10_A", "log10_S_break", "alpha_faint", "alpha_bright", "smoothness")

    def __init__(self, params, log_range=None):
        self.params = np.asarray(params, dtype=float)
        self.log_range = None if log_range is None else (float(log_range[0]), float(log_range[1]))

    def log10_cumulative(self, log10_S):
        logA, logSb, af, ab, D = self.params
        u = (np.asarray(log10_S, dtype=float) - logSb) / D
        soft = np.logaddexp(0.0, u * np.log(10.0)) / np.log(10.0)
        return logA - af * (np.asarray(log10_S, dtype=float) - logSb) - (ab - af) * D * soft

    def cumulative(self, S):
        """N(>S), galaxies deg^-2 brighter than flux S (mJy)."""
        return 10.0 ** self.log10_cumulative(np.log10(np.asarray(S, dtype=float)))

    def local_slope(self, log10_S):
        """-d log10 N / d log10 S, the running cumulative slope."""
        logA, logSb, af, ab, D = self.params
        u = (np.asarray(log10_S, dtype=float) - logSb) / D
        w = 1.0 / (1.0 + 10.0 ** np.clip(-u, -300, 300))
        return af + (ab - af) * w

    def differential(self, S):
        """phi(S) = -dN(>S)/dS, galaxies deg^-2 mJy^-1, positive everywhere."""
        S = np.asarray(S, dtype=float)
        return (self.cumulative(S) / S) * self.local_slope(np.log10(S))


def read_fazio_table(path, band_um=COORD_BAND_UM):
    """One band's block of the Fazio et al. 2004 Table 1 CSV: `(mag,
    {field: log10 differential galaxy counts})`, the table's own 0.5-mag
    bins, absent field/magnitude combinations left NaN.
    """
    df = pd.read_csv(path)
    blk = df[df["band_um"] == band_um].sort_values("mag")
    if blk.empty:
        raise ValueError(f"gal: no rows for band_um={band_um!r} in {path!r}")
    return blk["mag"].to_numpy(dtype=float), {c: blk[c].to_numpy(dtype=float) for c in FIELD_COLUMNS}


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


def fit_counts(log10_S, log10_N):
    """Least-squares fit of `BrokenPowerLaw` to a cumulative counts curve.
    Returns `(fit, stats)`, `stats` carrying the fit's own residual in dex
    so its adequacy is a number, not a claim.
    """
    x, y = np.asarray(log10_S, dtype=float), np.asarray(log10_N, dtype=float)
    p0 = [y.max(), x.mean(), 0.6, 1.8, 0.5]

    def resid(p):
        return BrokenPowerLaw(p).log10_cumulative(x) - y

    sol = least_squares(resid, p0, bounds=([0.0, -5.0, 0.05, 0.3, 0.05], [9.0, 3.0, 3.0, 5.0, 4.0]))
    r = resid(sol.x)
    fit = BrokenPowerLaw(sol.x, log_range=(float(x.min()), float(x.max())))
    return fit, {"rms_dex": float(np.sqrt(np.mean(r ** 2))), "max_abs_dex": float(np.max(np.abs(r)))}


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


def build_counts_law(fazio_path):
    """The pooled fit, its cosmic-variance band, and PHI_S on
    LOG10_S_GRID. Returns a dict ready for `write_counts`."""
    variants = fit_all_variants(fazio_path)
    fit, stats, _ = variants["mean"]
    cv_dex = cosmic_variance_dex(variants)
    log10_s_grid = build_log10_s_grid(fazio_path)
    phi_s = fit.differential(10.0 ** log10_s_grid)
    return dict(fit=fit, stats=stats, cosmic_variance_dex=cv_dex,
                log10_s_grid=log10_s_grid, phi_s=phi_s, variants=variants)


def write_counts(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fit = result["fit"]
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        for name, val in zip(BrokenPowerLaw.PARAM_NAMES, fit.params):
            f.create_dataset(name.upper(), data=np.float64(val))
        f.create_dataset("LOG10_S_GRID", data=result["log10_s_grid"].astype(np.float64))
        f.create_dataset("PHI_S", data=result["phi_s"].astype(np.float64))
        f.create_dataset("COSMIC_VARIANCE_DEX", data=np.float64(result["cosmic_variance_dex"]))
        f.create_dataset("FIT_LOG10_S_MIN", data=np.float64(fit.log_range[0]))
        f.create_dataset("FIT_LOG10_S_MAX", data=np.float64(fit.log_range[1]))
        f.create_dataset("FIT_RMS_DEX", data=np.float64(result["stats"]["rms_dex"]))


def write_cdf_tables(path, cdf):
    """Appends the colour-CDF tables (`build_colour_cdf_tables`) to the
    survey-level GAL product at `path` (`write_counts`'s own file):
    `CDF_GRID_I1`/`I3`/`I4` (`N_COLOUR_GRID`,) f4 the three colour axes,
    `CDF_JOINT` (n_bin, N, N, N) f4, `CDF_PAIR_I1I3`/`I1I4`/`I3I4`
    (n_bin, N, N) f4, `CDF_MARGINAL_I1`/`I3`/`I4` (n_bin, N) f4.
    """
    with h5py.File(path, "a") as f:
        for name, arr in (("CDF_GRID_I1", cdf["g1"]), ("CDF_GRID_I3", cdf["g3"]),
                          ("CDF_GRID_I4", cdf["g4"]), ("CDF_JOINT", cdf["joint"]),
                          ("CDF_PAIR_I1I3", cdf["pair13"]), ("CDF_PAIR_I1I4", cdf["pair14"]),
                          ("CDF_PAIR_I3I4", cdf["pair34"]), ("CDF_MARGINAL_I1", cdf["marg1"]),
                          ("CDF_MARGINAL_I3", cdf["marg3"]), ("CDF_MARGINAL_I4", cdf["marg4"])):
            if name in f:
                del f[name]
            f.create_dataset(name, data=arr.astype(np.float32))


# ---------------------------------------------------------------------------
# 2. The selection population -- SWIRE, star-galaxy separated, subsampled
# ---------------------------------------------------------------------------

def read_swire_catalogue(config):
    """The six SWIRE fields concatenated: `(flux_mjy, stell, ext_fl)`, each
    `(n, 4)` in `IRAC_BAND_KEYS` order. Fluxes converted from SWIRE's own
    aperture-2 uJy to this project's mJy convention.
    """
    dest_dir = f"{config.data_root}/sky/download/swire"
    cols = list(SWIRE_FLUX_COLUMNS) + list(SWIRE_STELL_COLUMNS) + list(SWIRE_EXT_FL_COLUMNS)
    frames = []
    for name in SWIRE_FIELD_FILES:
        path = f"{dest_dir}/{name}"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"gal: no SWIRE field catalogue at {path!r} -- run the "
                f"'sesnaimpute.sky.download.swire.build' RUNBOOK line")
        frames.append(pd.read_csv(path, usecols=cols))
    df = pd.concat(frames, ignore_index=True)
    flux_mjy = df[list(SWIRE_FLUX_COLUMNS)].to_numpy(dtype=float) / 1000.0
    stell = df[list(SWIRE_STELL_COLUMNS)].to_numpy(dtype=float)
    ext_fl = df[list(SWIRE_EXT_FL_COLUMNS)].to_numpy(dtype=float)
    return flux_mjy, stell, ext_fl


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


def select_star_galaxy_split(flux_mjy_i2, stell, ext_fl, fit, cosmic_variance_dex,
                              s_values=SPLIT_CRITERION_S_MJY):
    """Grades every `candidate_star_galaxy_splits` candidate against the
    Fazio fit's own `N(>S)` at `s_values` (SPEC_PRIORS.md 5.1's
    criterion): each candidate's ratio of its own SWIRE galaxy count to
    the Fazio target at each flux, in dex. A candidate "passes" if every
    one of those `|dex|` is within `cosmic_variance_dex`. The adopted
    split is the passing candidate removing the fewest SWIRE rows (the
    least intervention that meets the standard); if none passes, the
    candidate with the smallest maximum `|dex|`, flagged as such.

    Returns `(adopted_label, is_galaxy_adopted, rows, none_passed)`,
    `rows` a list of dicts (`label`, `n_removed`, `ratio` per `s_values`,
    `dex` per `s_values`, `max_abs_dex`, `passed`), one per candidate, in
    `candidate_star_galaxy_splits` order.
    """
    fazio_n = np.array([fit.cumulative(s) for s in s_values])
    candidates = candidate_star_galaxy_splits(stell, ext_fl)

    rows = []
    for label, is_galaxy in candidates.items():
        swire_n = np.array([swire_band_cumulative(flux_mjy_i2, is_galaxy, s) for s in s_values])
        dex = np.log10(swire_n) - np.log10(fazio_n)
        rows.append(dict(
            label=label, n_removed=int((~is_galaxy).sum()),
            ratio=(swire_n / fazio_n).tolist(), dex=dex.tolist(),
            max_abs_dex=float(np.max(np.abs(dex))),
            passed=bool(np.all(np.abs(dex) <= cosmic_variance_dex)),
        ))

    passing = [r for r in rows if r["passed"]]
    none_passed = len(passing) == 0
    if not none_passed:
        best = min(passing, key=lambda r: r["n_removed"])
    else:
        best = min(rows, key=lambda r: r["max_abs_dex"])
    return best["label"], candidates[best["label"]], rows, none_passed


#: The joint-colour CDF grid: points per axis and the target spacing,
#: dex (SPEC_PRIORS.md section 5.1's exact-selection design, restated
#: here as three colours' orthant probability -- see module docstring
#: at `build_colour_cdf_tables`).
N_COLOUR_GRID = 100
COLOUR_GRID_DEX = 0.02


def _colour_of(flux_mjy, band_idx, log10_i2):
    """log10(flux) - log10(I2 flux) for `band_idx`, `-inf` where that
    band's own flux is not finite (never clears, and never counted as
    a real value in the CDF's support -- see `_axis_index`)."""
    f = flux_mjy[:, band_idx]
    finite = np.isfinite(f) & (f > 0)
    return np.where(finite, np.log10(np.where(finite, f, 1.0)) - log10_i2, -np.inf)


def _colour_axis(colour, n=N_COLOUR_GRID, dex=COLOUR_GRID_DEX):
    """`(n,)` ascending: a grid spanning the population's own 0.1-99.9th
    percentile colour range (padded two cells either side), the range a
    real threshold can fall in."""
    finite = colour[np.isfinite(colour)]
    lo, hi = np.percentile(finite, [0.1, 99.9])
    pad = 2.0 * dex
    return np.linspace(lo - pad, hi + pad, n)


def _axis_index(colour, axis):
    """Each colour's own grid cell (0..N-1): the index of the smallest
    grid point at or above it, so a cumulative count through cell `m`
    is exactly the count with colour `<= axis[m]` (the CDF's own
    definition, not one cell off it). `-inf` (missing) always lands in
    cell 0, so it is counted as "colour <= threshold" at every grid
    point -- a missing band never clears, matching the direct test."""
    idx = np.searchsorted(axis, colour, side="left")
    return np.clip(idx, 0, axis.size - 1)


def build_colour_cdf_tables(flux_mjy, is_galaxy, log10_s_grid):
    """The exact joint-colour CDF of the FULL classified population, per
    4.5um flux bin (SPEC_PRIORS.md section 5.1): within one bin, a
    galaxy's pass/fail depends on the source only through four
    thresholds on its own colours (module docstring's rescale), so
    "two of four clear" is an orthant probability of the bin's
    three-dimensional colour distribution (3.6, 5.8, 8.0 relative to
    4.5um) -- tabulated here once per survey, read by trilinear/
    bilinear/linear interpolation per source in `build_source_selection`,
    no population member touched per source. `is_galaxy` is the
    star-galaxy split's own per-row mask, `None` for no removal.

    Returns a dict: `g1`/`g3`/`g4` (`N_COLOUR_GRID`,) the three colour
    axes (I1, I3, I4 each relative to I2); `joint` (n_bin, N, N, N) f4
    the 3-D CDF F(c1,c3,c4); `pair13`/`pair14`/`pair34` (n_bin, N, N) f4
    the three pairwise 2-D CDFs; `marg1`/`marg3`/`marg4` (n_bin, N) f4
    the three 1-D marginal CDFs -- everything an exact inclusion-
    exclusion needs for "at least 2 of {I1,I3,I4} clear".
    """
    i1, i2, i3, i4 = (IRAC_BAND_KEYS.index(b) for b in ("I1", "I2", "I3", "I4"))
    coord = flux_mjy[:, i2]
    valid = np.isfinite(coord) & (coord > 0)
    if is_galaxy is not None:
        valid = valid & is_galaxy
    idx = np.flatnonzero(valid)
    log10_i2 = np.log10(coord[idx])
    edges = 0.5 * (log10_s_grid[1:] + log10_s_grid[:-1])
    sbin = np.clip(np.searchsorted(edges, log10_i2), 0, log10_s_grid.size - 1)

    flux = flux_mjy[idx]
    c1 = _colour_of(flux, i1, log10_i2)
    c3 = _colour_of(flux, i3, log10_i2)
    c4 = _colour_of(flux, i4, log10_i2)
    g1, g3, g4 = _colour_axis(c1), _colour_axis(c3), _colour_axis(c4)
    n1, n3, n4 = _axis_index(c1, g1), _axis_index(c3, g3), _axis_index(c4, g4)

    n_bin = log10_s_grid.size
    N = N_COLOUR_GRID
    joint = np.zeros((n_bin, N, N, N), dtype=np.float32)
    pair13 = np.zeros((n_bin, N, N), dtype=np.float32)
    pair14 = np.zeros((n_bin, N, N), dtype=np.float32)
    pair34 = np.zeros((n_bin, N, N), dtype=np.float32)
    marg1 = np.zeros((n_bin, N), dtype=np.float32)
    marg3 = np.zeros((n_bin, N), dtype=np.float32)
    marg4 = np.zeros((n_bin, N), dtype=np.float32)

    for j in range(n_bin):
        m = sbin == j
        n_m = int(m.sum())
        if n_m == 0:
            continue
        a1, a3, a4 = n1[m], n3[m], n4[m]
        hist3 = np.bincount((a1 * N + a3) * N + a4, minlength=N ** 3).reshape(N, N, N)
        joint[j] = (np.cumsum(np.cumsum(np.cumsum(hist3, 0), 1), 2) / n_m).astype(np.float32)
        h13 = np.bincount(a1 * N + a3, minlength=N * N).reshape(N, N)
        pair13[j] = (np.cumsum(np.cumsum(h13, 0), 1) / n_m).astype(np.float32)
        h14 = np.bincount(a1 * N + a4, minlength=N * N).reshape(N, N)
        pair14[j] = (np.cumsum(np.cumsum(h14, 0), 1) / n_m).astype(np.float32)
        h34 = np.bincount(a3 * N + a4, minlength=N * N).reshape(N, N)
        pair34[j] = (np.cumsum(np.cumsum(h34, 0), 1) / n_m).astype(np.float32)
        marg1[j] = (np.cumsum(np.bincount(a1, minlength=N)) / n_m).astype(np.float32)
        marg3[j] = (np.cumsum(np.bincount(a3, minlength=N)) / n_m).astype(np.float32)
        marg4[j] = (np.cumsum(np.bincount(a4, minlength=N)) / n_m).astype(np.float32)

    return dict(g1=g1, g3=g3, g4=g4, joint=joint, pair13=pair13, pair14=pair14, pair34=pair34,
                marg1=marg1, marg3=marg3, marg4=marg4)


def _interp1(table_row, axis, x):
    return np.interp(x, axis, table_row)


def _bracket(axis, x):
    xc = np.clip(x, axis[0], axis[-1])
    i = np.clip(np.searchsorted(axis, xc) - 1, 0, axis.size - 2)
    t = (xc - axis[i]) / (axis[i + 1] - axis[i])
    return i, t


def _interp2(table, axis_a, axis_b, xa, xb):
    ia, ta = _bracket(axis_a, xa)
    ib, tb = _bracket(axis_b, xb)
    out = np.zeros_like(xa, dtype=np.float64)
    for da in (0, 1):
        wa = ta if da else (1.0 - ta)
        for db in (0, 1):
            wb = tb if db else (1.0 - tb)
            out += wa * wb * table[ia + da, ib + db]
    return out


def _interp3(table, axis_a, axis_b, axis_c, xa, xb, xc):
    ia, ta = _bracket(axis_a, xa)
    ib, tb = _bracket(axis_b, xb)
    ic, tc = _bracket(axis_c, xc)
    out = np.zeros_like(xa, dtype=np.float64)
    for da in (0, 1):
        wa = ta if da else (1.0 - ta)
        for db in (0, 1):
            wb = tb if db else (1.0 - tb)
            for dc in (0, 1):
                wc = tc if dc else (1.0 - tc)
                out += wa * wb * wc * table[ia + da, ib + db, ic + dc]
    return out


def population_flux_bins(flux_mjy, log10_s_grid):
    """`(idx_valid, bin_idx)`: the FULL population's own valid rows (finite,
    positive I2 flux) and their 4.5um flux bin on `log10_s_grid` (bin edges
    the midpoints between grid points) -- no subsample cap; used where a
    consumer needs the population's own absolute flux (`build_region_
    selection`, `eps_at_survey_median`), not the colour-CDF tables
    `build_colour_cdf_tables` builds for the per-source kernel.
    """
    coord = flux_mjy[:, IRAC_BAND_KEYS.index("I2")]
    valid = np.isfinite(coord) & (coord > 0)
    idx_valid = np.flatnonzero(valid)
    edges = 0.5 * (log10_s_grid[1:] + log10_s_grid[:-1])
    bin_idx = np.clip(np.searchsorted(edges, np.log10(coord[idx_valid])), 0, log10_s_grid.size - 1)
    return idx_valid, bin_idx


# ---------------------------------------------------------------------------
# 3a. The per-source selection EPS[n, n_x, n_s] (SPEC_PRIORS.md section 1.3)
# ---------------------------------------------------------------------------

def region_median_irac_limit(config, region):
    """`(8,)` log10 mJy: the region's own median 8-band detection limit
    over its sources (`catalog.limits.limits`) -- the single limit the
    two comparison variants `EPS_2BAND`/`EPS_NO_REMOVAL` are evaluated
    at, on the column-grid nodes (no depth grouping).
    """
    lim = limits_module.limits(config, region)
    with np.errstate(divide="ignore"):
        return np.median(np.log10(lim), axis=0)


def population_eight_band(log10_flux_irac):
    """`(n_pop, 8)`: the subsampled SWIRE population's own four IRAC
    log10 fluxes placed in `definitions.BANDS` order, the three 2MASS
    slots and the 24um slot left non-finite so they never clear a band
    test (SPEC_PRIORS.md section 5.1: "the 2MASS bands are excluded").
    """
    n_pop = log10_flux_irac.shape[0]
    out = np.full((n_pop, len(BAND_KEYS)), -np.inf, dtype=np.float64)
    out[:, IRAC_BAND_IDX] = log10_flux_irac
    other = np.array([j for j in range(len(BAND_KEYS)) if j not in set(IRAC_BAND_IDX.tolist())])
    out[:, other] = np.nan
    return out


def source_selection_from_cdf(log10_lim_b, a_query_b, kappa_b, log10_s_grid, cdf):
    """`(n_batch, n_x, n_s)` f8: the exact two-of-four pass fraction from
    the colour-CDF tables (`build_colour_cdf_tables`), one S-bin at a
    time. Per band `i` in {I1, I3, I4}, the threshold `tau_i = log10_lim
    - log10(S) + 0.4 a kappa_i` is the colour a galaxy must clear (the
    module docstring's rescale); the 4.5um band's own threshold `tau_2`
    compares against colour zero. If 4.5um clears, the pass fraction is
    `1 - F(tau1, tau3, tau4)` (at least one of the other three clears);
    otherwise it is `P(>=2 of 3 clear)`, the fixed inclusion-exclusion
    `sum(pairwise survivals) - 2*(triple survival)`, both built from the
    tabulated 1-D/2-D/3-D CDFs by linear/bilinear/trilinear interpolation
    -- no population member is read here.
    """
    i1, i2, i3, i4 = (IRAC_BAND_IDX[IRAC_BAND_KEYS.index(b)] for b in ("I1", "I2", "I3", "I4"))
    n_b, n_x = a_query_b.shape[0], a_query_b.shape[1]
    n_s = log10_s_grid.size
    eps = np.zeros((n_b, n_x, n_s), dtype=np.float64)
    dim1 = 0.4 * a_query_b * kappa_b[:, :, i1]
    dim2 = 0.4 * a_query_b * kappa_b[:, :, i2]
    dim3 = 0.4 * a_query_b * kappa_b[:, :, i3]
    dim4 = 0.4 * a_query_b * kappa_b[:, :, i4]
    for j in range(n_s):
        s = log10_s_grid[j]
        tau1 = (log10_lim_b[:, i1, None] - s + dim1)
        tau2 = (log10_lim_b[:, i2, None] - s + dim2)
        tau3 = (log10_lim_b[:, i3, None] - s + dim3)
        tau4 = (log10_lim_b[:, i4, None] - s + dim4)

        f3d = _interp3(cdf["joint"][j], cdf["g1"], cdf["g3"], cdf["g4"], tau1, tau3, tau4)
        at_least1 = 1.0 - f3d

        f1 = _interp1(cdf["marg1"][j], cdf["g1"], tau1)
        f3 = _interp1(cdf["marg3"][j], cdf["g3"], tau3)
        f4 = _interp1(cdf["marg4"][j], cdf["g4"], tau4)
        f13 = _interp2(cdf["pair13"][j], cdf["g1"], cdf["g3"], tau1, tau3)
        f14 = _interp2(cdf["pair14"][j], cdf["g1"], cdf["g4"], tau1, tau4)
        f34 = _interp2(cdf["pair34"][j], cdf["g3"], cdf["g4"], tau3, tau4)
        pair_ab = 1.0 - f1 - f3 + f13
        pair_ac = 1.0 - f1 - f4 + f14
        pair_bc = 1.0 - f3 - f4 + f34
        triple = 1.0 - f1 - f3 - f4 + f13 + f14 + f34 - f3d
        at_least2 = pair_ab + pair_ac + pair_bc - 2.0 * triple

        eps[:, :, j] = np.where(tau2 <= 0.0, at_least1, at_least2)
    return np.clip(eps, 0.0, 1.0)


def build_source_selection(config, region, cdf, log10_s_grid, batch_budget_bytes=(512 << 20)):
    """Writes the region's exact per-source GAL selection
    (SPEC_PRIORS.md section 1.3): for every catalogued source, on
    `selection.X_LADDER` by `log10_s_grid`, the exact two-of-four pass
    fraction read from the survey-wide colour-CDF tables (`build_colour_
    cdf_tables`, `source_selection_from_cdf`) -- a background galaxy
    carries the entire column (SPEC_PRIORS.md section 5.2). Sources are
    batched (`sesnaimpute.batches.batches`) so no batch's working arrays
    exceed `batch_budget_bytes`. Returns the product path.
    """
    log10_lim = np.log10(limits_module.limits(config, region))
    n_source = log10_lim.shape[0]
    n_x = selection_module.X_LADDER.size
    n_s = log10_s_grid.size

    adopted_path = config_module.product_path(
        config, "sky/derived", "adopted", "column", "source", region=region)
    a_col = np.asarray(
        access.per_source(config, region, adopted_path, ["A_COL_K"])["A_COL_K"], dtype=np.float64)
    if a_col.shape[0] != n_source:
        raise ValueError(
            "gal: %r's column count (%d) does not match the region's %d sources"
            % (adopted_path, a_col.shape[0], n_source))

    path = config_module.product_path(config, "bms", "gal", "selection", "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row_bytes = len(BAND_KEYS) * 8 + n_x * 8 + n_x * len(BAND_KEYS) * 8 + n_x * n_s * 8 * 4
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.create_dataset("X_LADDER", data=selection_module.X_LADDER.astype(np.float64))
        f.create_dataset("LOG10_S_GRID", data=np.asarray(log10_s_grid, dtype=np.float64))
        ds_eps = f.create_dataset("EPS", shape=(n_source, n_x, n_s), dtype="f2")
        for start, stop in batches_module.batches(n_source, row_bytes, budget_bytes=batch_budget_bytes):
            lim_b = np.ascontiguousarray(log10_lim[start:stop])
            a_b = a_col[start:stop]
            a_query_b = np.ascontiguousarray(selection_module.X_LADDER[None, :] * a_b[:, None])
            w_dense_b = selection_module.law_dense_weight(a_query_b)
            kappa_b = np.ascontiguousarray(selection_module.kappa_hybrid(config, w_dense_b))
            eps = source_selection_from_cdf(lim_b, a_query_b, kappa_b, log10_s_grid, cdf)
            ds_eps[start:stop] = eps.astype("f2")
    return path


# ---------------------------------------------------------------------------
# 3b. The two comparison variants, region median limit, column-grid nodes
# ---------------------------------------------------------------------------

def build_region_selection(a_nodes, kappa4_all, log10_flux_irac, finite_irac,
                            bin_idx, n_bins, limit_log10):
    """`(eps, eps_2band)`, each `(K, n_node, n_bins)`: the fraction of the
    (already subsampled, star-galaxy-separated) SWIRE population that,
    dimmed through column `a_nodes[i]` in every IRAC band by
    `10**(-0.4*a*kappa_i(a))`, clears any two of the four bands (`eps`) or
    both 3.6 and 4.5um (`eps_2band`, the SEDS-era comparison) at row
    `k`'s own limit `limit_log10[k]`. `K` is 1 here: the region's own
    median 8-band limit, no depth grouping.

    Vectorised over galaxies and bands (one matmul against a
    galaxy-to-bin indicator per node); looped only over the column-grid
    nodes (183).
    """
    n_gal = log10_flux_irac.shape[0]
    n_node = a_nodes.size
    K = limit_log10.shape[0]

    indicator = np.zeros((n_gal, n_bins), dtype=np.float32)
    indicator[np.arange(n_gal), bin_idx] = 1.0
    bin_counts = indicator.sum(axis=0)
    safe_counts = np.where(bin_counts > 0, bin_counts, 1.0)

    eps = np.zeros((K, n_node, n_bins), dtype=np.float32)
    eps_2band = np.zeros((K, n_node, n_bins), dtype=np.float32)

    i1 = IRAC_BAND_KEYS.index("I1")
    i2 = IRAC_BAND_KEYS.index("I2")

    for i in range(n_node):
        dimmed = log10_flux_irac - 0.4 * a_nodes[i] * kappa4_all[i][None, :]         # (n_gal, 4)
        clears = finite_irac & (dimmed[None, :, :] >= limit_log10[:, None, :])       # (K, n_gal, 4)
        n_clear = clears.sum(axis=2)
        passed = (n_clear >= 2).astype(np.float32)                                   # (K, n_gal)
        passed_2band = (clears[:, :, i1] & clears[:, :, i2]).astype(np.float32)      # (K, n_gal)

        num = passed @ indicator          # (K, n_bins)
        num2 = passed_2band @ indicator   # (K, n_bins)
        eps[:, i, :] = np.where(bin_counts > 0, num / safe_counts, 0.0)
        eps_2band[:, i, :] = np.where(bin_counts > 0, num2 / safe_counts, 0.0)

    return eps, eps_2band, bin_counts


def write_region(path, a_nodes, log10_s_grid, eps_2band, eps_no_removal, median_limit_log10,
                  split_label, split_row):
    """The region-level comparison product: `EPS_2BAND` and
    `EPS_NO_REMOVAL`, one row each on `a_nodes` by `log10_s_grid`, both
    evaluated at the region's own median 8-band limit (no depth
    grouping). The per-source `EPS` lives in the sibling `prior/
    selection/source` product (`build_source_selection`).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "region"
        f.attrs["SPLIT"] = split_label
        f.attrs["SPLIT_CRITERION_S_MJY"] = np.array(SPLIT_CRITERION_S_MJY, dtype=np.float64)
        f.attrs["SPLIT_RATIO_SWIRE_OVER_FAZIO"] = np.array(split_row["ratio"], dtype=np.float64)
        f.create_dataset("A_NODES", data=a_nodes.astype(np.float64))
        f.create_dataset("LOG10_S_GRID", data=log10_s_grid.astype(np.float64))
        f.create_dataset("EPS_2BAND", data=eps_2band.astype(np.float32))
        f.create_dataset("EPS_NO_REMOVAL", data=eps_no_removal.astype(np.float32))
        f.create_dataset("MEDIAN_LOG10_FLIM", data=median_limit_log10.astype(np.float64))


# ---------------------------------------------------------------------------
# report-only checks (SPEC_PRIORS.md section 5.1 last rows, "Checks")
# ---------------------------------------------------------------------------

def regions_below_swire_depth(config, region_names):
    """`{region: {band: True}}` for `region_names` whose own median IRAC
    limit is fainter than SWIRE's 5-sigma depth in that band -- SWIRE
    would then not reach deep enough to characterise the population the
    region's own catalogue can see.
    """
    swire_mjy = np.array([SWIRE_5SIGMA_UJY[b] / 1000.0 for b in IRAC_BAND_KEYS])
    below = {}
    for region in region_names:
        median_irac_log10 = region_median_irac_limit(config, region)[IRAC_BAND_IDX]
        flagged = 10.0 ** median_irac_log10 < swire_mjy
        if flagged.any():
            below[region] = {b: bool(f) for b, f in zip(IRAC_BAND_KEYS, flagged) if f}
    return below


def swire_band_cumulative(flux_mjy_band, is_galaxy, threshold_mjy):
    """SWIRE's own measured N(>S) at `threshold_mjy`, galaxies deg^-2 --
    the "against the four-band catalogue's own counts" check."""
    good = is_galaxy & np.isfinite(flux_mjy_band) & (flux_mjy_band >= threshold_mjy)
    return float(np.sum(good)) / SWIRE_AREA_DEG2


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


def eps_at_survey_median(config, log10_flux_irac, finite_irac, bin_idx, log10_s_grid, a, s_mjy,
                          limit_log10):
    """`(eps, eps_2band)` at one `(a, S)` point, evaluated directly against
    `limit_log10` (the region's own median 8-band limit, IRAC bands only)
    -- the report-only sensitivity point named in the brief, not a
    stored table.
    """
    w = selection_module.law_dense_weight(np.array([a]))
    kappa4 = selection_module.kappa_hybrid(config, w)[0, IRAC_BAND_IDX]
    dimmed = log10_flux_irac - 0.4 * a * kappa4[None, :]
    clears = finite_irac & (dimmed >= limit_log10[None, :])
    i1, i2 = IRAC_BAND_KEYS.index("I1"), IRAC_BAND_KEYS.index("I2")
    passed = clears.sum(axis=1) >= 2
    passed_2band = clears[:, i1] & clears[:, i2]

    edges = 0.5 * (log10_s_grid[1:] + log10_s_grid[:-1])
    j = int(np.clip(np.searchsorted(edges, np.log10(s_mjy)), 0, log10_s_grid.size - 1))
    in_bin = bin_idx == j
    n = int(in_bin.sum())
    if n == 0:
        return float("nan"), float("nan")
    return float(passed[in_bin].mean()), float(passed_2band[in_bin].mean())


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Writes the survey-wide counts law and, for `regions` (default: all
    thirty), the per-region selection tables.
    """
    import numba
    numba.set_num_threads(max(1, int(config.n_jobs)))
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    fazio_path = f"{config.data_root}/sky/download/fazio2004/fazio2004_table1_irac_counts.csv"
    if not os.path.exists(fazio_path):
        raise FileNotFoundError(
            f"gal: no Fazio 2004 table at {fazio_path!r} -- see the "
            f"'sky/download/fazio2004' RUNBOOK comment line: manual "
            f"acquisition, no reachable download URL")

    counts_result = build_counts_law(fazio_path)
    counts_path = config_module.product_path(config, "bms", "gal", "counts", "survey")
    write_counts(counts_path, counts_result)
    fit = counts_result["fit"]
    print(f"gal: counts law: log10_A={fit.params[0]:.4f} log10_S_break={fit.params[1]:.4f} "
          f"alpha_faint={fit.params[2]:.4f} alpha_bright={fit.params[3]:.4f} "
          f"smoothness={fit.params[4]:.4f} rms={counts_result['stats']['rms_dex']:.4f} dex "
          f"cosmic_variance={counts_result['cosmic_variance_dex']:.4f} dex -> {counts_path}")

    flux_mjy_all, stell_all, ext_fl_all = read_swire_catalogue(config)
    print("gal: SWIRE pull carries no optical stellarity column (sky.download.swire.build.COLUMNS); "
          "the stellarity split candidate reads IRAC 3.6um in its place")

    split_label, is_galaxy_all, split_rows, none_passed = select_star_galaxy_split(
        flux_mjy_all[:, IRAC_BAND_KEYS.index("I2")], stell_all, ext_fl_all,
        fit, counts_result["cosmic_variance_dex"])
    print("gal: star-galaxy split candidates (Fazio N(>S) reproduction at "
          f"S={SPLIT_CRITERION_S_MJY} mJy, cosmic-variance band ="
          f" {counts_result['cosmic_variance_dex']:.4f} dex):")
    for r in split_rows:
        ratio_str = ", ".join(f"{s}mJy={ratio:.3f}" for s, ratio in zip(SPLIT_CRITERION_S_MJY, r["ratio"]))
        print(f"gal:   {r['label']}: n_removed={r['n_removed']} ratio(swire/fazio) [{ratio_str}] "
              f"max|dex|={r['max_abs_dex']:.4f} passed={r['passed']}")
    if none_passed:
        print(f"gal: no split candidate reproduces Fazio's counts within the cosmic-variance band; "
              f"adopting the smallest max|dex| candidate: {split_label}")
    else:
        print(f"gal: adopted split: {split_label} (least intervention among candidates meeting the standard)")
    split_row = next(r for r in split_rows if r["label"] == split_label)

    n_star = int((~is_galaxy_all).sum())
    print(f"gal: SWIRE: {flux_mjy_all.shape[0]} rows, {is_galaxy_all.sum()} classed galaxy under "
          f"{split_label}, {n_star} classed star")

    log10_s_grid = counts_result["log10_s_grid"]

    # The FULL population's own absolute fluxes, no subsample cap: for the
    # region-level comparison variants and report-only checks, which read a
    # galaxy's actual I1/I2 flux directly (not the colour-only rescale the
    # per-source kernel uses).
    idx_valid, bin_idx = population_flux_bins(flux_mjy_all[is_galaxy_all], log10_s_grid)
    flux_mjy = flux_mjy_all[is_galaxy_all][idx_valid]
    finite_irac = np.isfinite(flux_mjy) & (flux_mjy > 0)
    log10_flux_irac = np.where(finite_irac, np.log10(np.where(finite_irac, flux_mjy, 1.0)), -np.inf)
    frac_measured = finite_irac.mean(axis=0)
    print("gal: full population: %d galaxies; fraction with a measured flux: %s" % (
        flux_mjy.shape[0],
        ", ".join(f"{b}={frac_measured[j]:.3f}" for j, b in enumerate(IRAC_BAND_KEYS))))

    idx_valid_nr, bin_idx_nr = population_flux_bins(flux_mjy_all, log10_s_grid)
    flux_mjy_nr = flux_mjy_all[idx_valid_nr]
    finite_irac_nr = np.isfinite(flux_mjy_nr) & (flux_mjy_nr > 0)
    log10_flux_irac_nr = np.where(finite_irac_nr, np.log10(np.where(finite_irac_nr, flux_mjy_nr, 1.0)), -np.inf)

    # The per-source kernel's own population: not touched per source at
    # all -- the exact joint-colour CDF of the FULL population, once per
    # survey, per 4.5um flux bin (module docstring at `build_colour_cdf_
    # tables`); `build_source_selection` reads it by interpolation.
    import time as _time
    _t0 = _time.time()
    cdf = build_colour_cdf_tables(flux_mjy_all, is_galaxy_all, log10_s_grid)
    cdf_build_s = _time.time() - _t0
    write_cdf_tables(counts_path, cdf)
    cdf_bytes = sum(a.nbytes for a in cdf.values() if hasattr(a, "nbytes"))
    print("gal: colour-CDF tables: %d points/axis, %d S-bins, %.1f MB total, built in %.2f s -> %s" % (
        N_COLOUR_GRID, log10_s_grid.size, cdf_bytes / 1e6, cdf_build_s, counts_path))

    a_nodes = column_grid_module.nodes(config)
    w_nodes = selection_module.law_dense_weight(a_nodes)
    kappa4_all = selection_module.kappa_hybrid(config, w_nodes)[:, IRAC_BAND_IDX]

    below = regions_below_swire_depth(config, region_names)
    if below:
        print(f"gal: regions whose own reference limit is fainter than SWIRE's 5-sigma depth: {below}")
    else:
        print("gal: no region's reference limit falls below SWIRE's 5-sigma depths")

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

    max_2band_violation = 0.0
    max_monotone_violation = 0.0
    for region in region_names:
        median_limit_log10 = region_median_irac_limit(config, region)
        limit_log10 = median_limit_log10[IRAC_BAND_IDX][None, :]

        for a_check in (0.0, 2.0):
            for s_check in (0.05, 0.5):
                e4, e2 = eps_at_survey_median(config, log10_flux_irac, finite_irac, bin_idx, log10_s_grid,
                                               a_check, s_check, limit_log10[0])
                e4_nr, _e2_nr = eps_at_survey_median(config, log10_flux_irac_nr, finite_irac_nr, bin_idx_nr,
                                                      log10_s_grid, a_check, s_check, limit_log10[0])
                print(f"gal: {region} median-limit eps(a={a_check}, S={s_check}mJy): 4-band={e4:.4f} "
                      f"2-band={e2:.4f} no_removal={e4_nr:.4f} (adopted split: {split_label})")

        eps, eps_2band, _bin_counts = build_region_selection(
            a_nodes, kappa4_all, log10_flux_irac, finite_irac, bin_idx, log10_s_grid.size, limit_log10)
        eps_no_removal, _eps_2band_nr, _bin_counts_nr = build_region_selection(
            a_nodes, kappa4_all, log10_flux_irac_nr, finite_irac_nr, bin_idx_nr, log10_s_grid.size, limit_log10)

        max_2band_violation = max(max_2band_violation, float(np.max(eps_2band - eps)))
        d = np.diff(eps.astype(np.float64), axis=1)
        max_monotone_violation = max(max_monotone_violation, float(np.max(np.clip(d, 0.0, None))))

        S_lin = 10.0 ** log10_s_grid
        n_gal_region = float(np.trapz(counts_result["phi_s"] * eps[0, 0, :], S_lin))
        n_gal_region_nr = float(np.trapz(counts_result["phi_s"] * eps_no_removal[0, 0, :], S_lin))
        fazio_at_median_limit = float(fit.cumulative(10.0 ** median_limit_log10[IRAC_BAND_IDX[
            IRAC_BAND_KEYS.index("I2")]]))

        region_path = config_module.product_path(config, "bms", "gal", "prior", "region", region=region)
        write_region(region_path, a_nodes, log10_s_grid, eps_2band[0], eps_no_removal[0], median_limit_log10,
                     split_label, split_row)
        print(f"gal: {region}: N_GAL(a=0, median limit) adopted={n_gal_region:.1f} deg^-2 "
              f"no_removal={n_gal_region_nr:.1f} deg^-2 "
              f"Fazio N(>region median I2 limit)={fazio_at_median_limit:.1f} deg^-2 -> {region_path}")

        source_path = build_source_selection(config, region, cdf, log10_s_grid)
        print(f"gal: {region}: per-source selection (colour-CDF interpolation, "
              f"{selection_module.X_LADDER.size} x-nodes, {log10_s_grid.size} S-grid points) -> {source_path}")

    print(f"gal: acceptance: max(EPS_2BAND - EPS)={max_2band_violation:.6g} "
          f"(expect <= 0); max positive d(EPS)/d(node)={max_monotone_violation:.6g} (expect ~0)")


if __name__ == "__main__":
    run(build)
