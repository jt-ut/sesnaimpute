"""GAL: the background-galaxy number-counts law (SPEC_PRIORS.md section 5).

This module builds the one survey-wide, region-independent piece of the
background-galaxy prior that a consumer reads: `phi(S)`, the intrinsic
4.5um galaxy DIFFERENTIAL number-counts law -- one smooth broken power
law, break free, fitted directly to the differential counts themselves
(`fit_counts_differential`), never through an intermediate cumulative
fit and derivative (review ledger C3: the earlier fit was to cumulative
N(>S), consumed as the differential `phi(S) = -dN/dS`, with the break
bounded to the data's own range -- which is why it always landed on
that range's edge).

Owner ruling 2026-10-21 (review ledger C27, overruling C3's "SEDS
beside, never fit"): the law is fitted jointly to BOTH tabulations it
has -- Fazio et al. (2004, ApJS 154, 39, Table 1)'s wide-area bright end
and Ashby et al. (2013)'s SEDS (ApJ 769, 80, five full-depth fields,
1.46 deg^2) deep faint end, each star-subtracted, each weighted by its
own Poisson error plus its own measured field-to-field (cosmic-variance)
scatter -- never one dataset weighted down by hand (`build_joint_counts_
law`). SEDS carries no native star/galaxy classification (the ReadMe's
own byte-by-byte description has none), so its counts are star-
subtracted by Fazio's own star columns transported to each SEDS field's
own Galactic latitude, exactly as `expected_star_count` already
transports them to SWIRE's fields (`expected_star_phi_at_latitude`,
differential form). The two datasets are complementary, not redundant:
Fazio's three wide fields anchor the bright end SEDS's 1.46 deg^2 cannot
reach at adequate numbers, while SEDS's ~100x the source count anchors
the faint end at far smaller Poisson error than Fazio's own few-field
tabulation carries there -- so the weighting alone, with no hand
adjustment, lets each dataset govern where it is actually informative.

The fit is written to the `counts_gal_survey` product's
`LOG10_S_GRID`/`PHI_S`, alongside `PHI_S_POINT = PHI_S * P_POINT`, the
IRAC point-source retention `p(S)` MEASURED directly on SWIRE's own
per-band extended-source flag over 0.3-20mJy (`measure_point_fraction`,
review ledger C4: the earlier `p(S)` was calibrated over 0.056-0.28mJy
only and extrapolated 1.8 dex beyond it as an unmeasured power law) --
`bmstp.sample_gal` and `bmstp.density` read the point-source-corrected
law (`bmstp.template_weights` does not read this product). The full
selection shape `r(a, log10 S) = phi(S) . S . eps(a, S)` assembles this
law with the extinction/selection term elsewhere; this module never
evaluates `eps`.

The star-galaxy split itself is chosen once, survey-wide
(`select_star_galaxy_split`): among the candidate rules, the one whose
own removed-star count, summed across the six SWIRE (Surace et al. 2005
DR2 release) fields, is closest to Fazio's own star columns transported
to each field's latitude (studies/swire_vs_fazio.md sec 4) -- not the
earlier criterion (surviving galaxy count against Fazio's galaxy law),
which is unreachable once `p(S) < 1`. The split and the counts law are
cross-checked against the star-subtracted SWIRE and S-COSMOS (Sanders
et al. 2007, ApJS 172, 86) counts at fixed fluxes and printed; neither
the split nor these checks are written to the product. S-COSMOS's own
IRAC table carries no native point/extended-source or stellarity flag
(confirmed directly on `sky/download/scosmos/scosmos_irac.csv`: its
`fl_c1`/`fl_c3` and `fl_c2`/`fl_c4` columns are identical row for row --
a shared per-detector-pair quality flag, not a morphology classifier --
and `studies/S-L5_four_band_galaxy_catalogue.md` already records this),
so `p(S)` is measured on SWIRE alone; S-COSMOS continues to serve only
the existing deep single-field total-count cross-check.

The library register never enters this quantity (SPEC_PRIORS.md section
0.3, C3): it supplies SED templates to the fitter only.
"""

import gzip
import os

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run

_STEM = "counts_gal_survey"

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

#: IRAC point-source retention p(S): the fraction of SWIRE's adopted
#: galaxies at each 4.5um flux whose own per-band extended-source flag
#: (3.6, 4.5um -- `EXT_FL_EXTENDED`) never reads "might be" or "clearly"
#: extended, measured DIRECTLY over this range (`measure_point_fraction`,
#: review ledger C4) -- never extrapolated past it. Below
#: `POINT_FRACTION_S_LO_MJY`, p(S) is held at 1 (module docstring: the
#: galaxy angular-size distribution has already crossed the IRAC PSF
#: there and has no mechanism to turn back down; no measurement
#: contradicts this, so it is unchanged).
POINT_FRACTION_S_LO_MJY = 0.3
POINT_FRACTION_S_HI_MJY = 20.0
POINT_FRACTION_N_BINS = 12

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

#: Ashby et al. (2013, ApJ 769, 80)'s five SEDS fields: the per-source
#: fixed-width catalogue file (Ashby et al. 2013 ReadMe, "table[789].dat
#: table1[01].dat") and the field's own area, deg^2 (Table 1: UDS, ECDFS,
#: COSMOS, HDFN, EGS), now one of the counts law's own fit inputs
#: (review ledger C27, owner ruling 2026-10-21, overruling C3's "SEDS
#: beside, never fit"): the release carries no star-galaxy separation
#: (module docstring), so its counts are star-subtracted here before
#: entering the fit (`expected_star_phi_at_latitude`), never used raw.
ASHBY_SEDS_FIELDS = (
    ("table7.dat.gz", 0.32), ("table8.dat.gz", 0.35), ("table9.dat.gz", 0.19),
    ("table10.dat.gz", 0.25), ("table11.dat.gz", 0.35),
)
ASHBY_SEDS_AREA_DEG2 = float(sum(area for _name, area in ASHBY_SEDS_FIELDS))

#: `ASHBY_SEDS_FIELDS`'s own names, same order (Ashby et al. 2013 Table 1:
#: UDS, ECDFS, COSMOS, HDFN, EGS) -- the star subtraction's per-field key.
ASHBY_SEDS_FIELD_NAMES = ("UDS", "ECDFS", "COSMOS", "HDFN", "EGS")

#: `ASHBY_SEDS_FIELD_NAMES`'s own |Galactic latitude|, deg -- each field's
#: own centre (Ashby et al. 2013 ReadMe `table1.dat`: RA/Dec J2000,
#: "02 18 00 -05 10 17" etc.) transformed ICRS -> Galactic once, the
#: target latitude the Fazio star-count law is transported to here
#: (`expected_star_phi_at_latitude`), mirroring what `expected_star_count`
#: already does for `SWIRE_FIELD_ABS_B_DEG`.
ASHBY_SEDS_FIELD_ABS_B_DEG = (59.88, 54.44, 42.10, 54.80, 59.81)

#: Byte-offset columns (0-indexed, half-open) of the SEDS per-source
#: fixed-width table (Ashby et al. 2013 ReadMe, bytes 100-104 "[4.5]psf",
#: byte 152 "f4.5"): the PSF-fitted 4.5um AB magnitude and its own
#: quality-flag sum (0 = no known issue).
ASHBY_SEDS_COLSPECS = ((99, 104), (151, 152))
ASHBY_SEDS_COLUMN_NAMES = ("mag45psf", "f45")

#: The AB magnitude system's own zero-point flux density, Jy -- Ashby et
#: al. (2013)'s own magnitude convention (ReadMe byte-by-byte
#: description, "mag"), never Fazio's Vega system.
ASHBY_SEDS_AB_ZP_JY = 3631.0

#: Fazio Table 1's own published per-row uncertainty on that field's
#: `*_total` column, dex -- `FIELD_COLUMNS`/`STAR_FIELD_COLUMNS` row order,
#: no separate `_galaxies`-only error tabulated. Used as the nearest
#: stored uncertainty for the star-subtracted galaxy row too (disclosed,
#: `fazio_mean_variant_with_sigma`), rather than fabricating one.
FAZIO_ERR_COLUMNS = ("bootes_err", "egs_err", "qso1700_err")

#: The SEDS fit input's own bin width, log10 S dex (the owner's own
#: choice, review ledger C27).
SEDS_FIT_BIN_WIDTH_DEX = 0.25

#: A SEDS field-to-field-scatter bin needs at least this many RAW
#: (pre-subtraction) sources in a field before that field's own log10
#: density there is trusted for the scatter measurement -- too few and
#: the estimate is small-number noise, not real field-to-field spread.
SEDS_SCATTER_MIN_N_FIELD = 50

#: ...and needs at least this many of the five fields passing that count
#: (with a positive star-subtracted count) before the bin contributes an
#: RMS to `seds_differential_counts`'s own pooled scatter estimate at all.
SEDS_SCATTER_MIN_FIELDS = 3


# ---------------------------------------------------------------------------
# 1. The counts law phi(S) -- Fazio et al. 2004, fit once, survey-wide
# ---------------------------------------------------------------------------

class BrokenPowerLaw:
    """A smooth broken power law in log-log:

        log10 y(S) = logA - af*x - (ab-af)*D*log10(1 + 10**(x/D)),
        x = log10(S/S_b)

    with faint-end slope `af`, bright-end slope `ab`, break `S_b` and
    smoothness `D`. Fitted here directly to the DIFFERENTIAL galaxy
    counts phi(S), never to the cumulative N(>S) and then differentiated
    (review ledger C3 -- the predecessor fit was to N(>S), with the break
    bounded inside the data's own tabulated range, which is why it always
    landed on that range's edge; `fit_counts_differential` fits phi(S)
    itself with the break free of any such bound). A single power law
    (`af == ab`) is the same formula with the break/smoothness terms
    cancelling out exactly, leaving a straight line in log-log, so no
    separate class is needed."""

    PARAM_NAMES = ("log10_A", "log10_S_break", "alpha_faint", "alpha_bright", "smoothness")

    def __init__(self, params):
        self.params = np.asarray(params, dtype=float)

    def log10_value(self, log10_S):
        """The fitted log10 y at `log10_S` -- log10 phi(S) when this is a
        `fit_counts_differential` result, the quantity actually fitted."""
        logA, logSb, af, ab, D = self.params
        log10_S = np.asarray(log10_S, dtype=float)
        u = (log10_S - logSb) / D
        soft = np.logaddexp(0.0, u * np.log(10.0)) / np.log(10.0)
        return logA - af * (log10_S - logSb) - (ab - af) * D * soft

    def value(self, S):
        """phi(S), galaxies deg^-2 mJy^-1, directly -- the fitted
        quantity itself, no derivative step."""
        return 10.0 ** self.log10_value(np.log10(np.asarray(S, dtype=float)))

    def cumulative_numeric(self, S, log10_s_hi, n_grid=4001):
        """N(>S), galaxies deg^-2, by numerically integrating `value`
        from `S` up to `10**log10_s_hi` (Fazio's own tabulated bright
        edge) -- report-only checks need a cumulative count; no analytic
        cumulative form is kept once phi(S) is the directly fitted
        quantity (C3). `S` scalar or array, mJy."""
        s_arr = np.atleast_1d(np.asarray(S, dtype=float))
        out = np.empty_like(s_arr)
        for i, s0 in enumerate(s_arr):
            grid = np.linspace(np.log10(s0), float(log10_s_hi), n_grid)
            phi = self.value(10.0 ** grid)
            out[i] = np.trapz(phi * 10.0 ** grid * np.log(10.0), grid)
        return out if np.ndim(S) else float(out[0])


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


def fazio_differential_phi(mag, log10_n, band_um=COORD_BAND_UM):
    """Fazio et al. (2004) Table 1's own tabulated DIFFERENTIAL galaxy
    counts, converted directly to phi(S) = dN/dS (galaxies deg^-2
    mJy^-1) at each populated 0.5-mag row -- never through an
    intermediate cumulative integral (review ledger C3). Fazio et al.
    (2004, ApJS 154, 39, section 2) state Table 1's own units directly:
    "the logarithm of the observed differential number counts in units
    of number mag^-1 deg^-2" -- `n(mag)` tabulated in log10 is already
    PER MAGNITUDE; the table's 0.5-mag spacing is only its sampling
    step, not a bin width to divide out (review ledger C16: dividing by
    0.5 doubled this law survey-wide). So `dN/dmag = 10**n(mag)`
    directly, and on the mag-flux relation `mag = -2.5 log10(S /
    S_zp)`, `|dmag/dS| = 2.5 / (S ln10)`; `phi(S) = dN/dmag * |dmag/dS|`.
    Returns `(log10_S, log10_phi)`, ascending in `log10_S`, `S` in mJy on
    Fazio's own Vega zero point.
    """
    ok = np.isfinite(log10_n)
    m = np.asarray(mag, dtype=float)[ok]
    n_mag = np.asarray(log10_n, dtype=float)[ok]
    zp_mjy = FAZIO_VEGA_ZP_JY[band_um] * 1e3
    log10_S = np.log10(zp_mjy * 10.0 ** (-0.4 * m))
    dn_dmag = 10.0 ** n_mag
    s_mjy = 10.0 ** log10_S
    phi = dn_dmag * 2.5 / (s_mjy * np.log(10.0))
    order = np.argsort(log10_S)
    return log10_S[order], np.log10(phi)[order]


#: `fit_counts_differential`'s own multi-start grid: log-spaced break
#: starts and several starting slope pairs. The break is FREE (review
#: ledger C3): its search and bound extend `_BREAK_FREE_MARGIN_DEX` past
#: Fazio's own tabulated range on each side, so a break the data support
#: outside that range is not penalised relative to one inside it -- the
#: predecessor fit bounded the break TO the data range and so always
#: landed on that range's edge (not evidence against a break, an
#: artifact of the bound itself).
_BREAK_START_N = 10
_SLOPE_STARTS = ((0.3, 1.2), (0.6, 1.8), (1.0, 2.5), (1.5, 3.2))
_BREAK_FREE_MARGIN_DEX = 6.0


def fit_counts_differential(log10_S, log10_phi, sigma=None):
    """Least-squares fit of `BrokenPowerLaw` directly to DIFFERENTIAL
    galaxy counts phi(S) (review ledger C3: never through an intermediate
    cumulative fit), break free of any bound to the data's own tabulated
    range: multi-start over `_BREAK_START_N` log-spaced break starts
    (spanning the data's own range plus `_BREAK_FREE_MARGIN_DEX` dex on
    each side) times `_SLOPE_STARTS` slope-pair starts, keeping the
    lowest-cost result.

    `sigma`, one per point, dex: each point's residual is divided by its
    own `sigma` before the sum of squares is formed (review ledger C27,
    owner ruling 2026-10-21 -- the joint Fazio+SEDS fit's own weighting,
    each point by its own error, no dataset weighted down by hand).
    Defaults to all ones (every earlier, single-dataset caller of this
    function, unweighted, unchanged).

    Returns `(fit, stats)`, `stats` carrying the fit's own weighted
    ("chi-like", dimensionless) and unweighted (dex) rms, max unweighted
    residual, the data's own flux range (descriptive -- the fit is not
    bounded to it), and whether the fitted break landed inside that
    range.
    """
    x, y = np.asarray(log10_S, dtype=float), np.asarray(log10_phi, dtype=float)
    s = np.ones_like(x) if sigma is None else np.asarray(sigma, dtype=float)
    x_lo, x_hi = float(x.min()), float(x.max())
    lo_bound, hi_bound = x_lo - _BREAK_FREE_MARGIN_DEX, x_hi + _BREAK_FREE_MARGIN_DEX

    def resid(p):
        return (BrokenPowerLaw(p).log10_value(x) - y) / s

    break_starts = np.linspace(x_lo, x_hi, _BREAK_START_N)
    best_params, best_cost = None, np.inf
    for logSb0 in break_starts:
        for af0, ab0 in _SLOPE_STARTS:
            p0 = [float(y.max()), float(logSb0), float(af0), float(ab0), 0.5]
            sol = least_squares(resid, p0, bounds=(
                [0.0, lo_bound, -3.0, 0.1, 0.02], [12.0, hi_bound, 6.0, 8.0, 4.0]))
            r = resid(sol.x)
            cost = float(np.sum(r ** 2))
            if cost < best_cost:
                best_params, best_cost = sol.x, cost

    fit = BrokenPowerLaw(best_params)
    r_weighted = resid(best_params)
    r_unweighted = BrokenPowerLaw(best_params).log10_value(x) - y
    return fit, {"rms_dex": float(np.sqrt(np.mean(r_unweighted ** 2))),
                "rms_weighted": float(np.sqrt(np.mean(r_weighted ** 2))),
                "max_abs_dex": float(np.max(np.abs(r_unweighted))),
                "log10_s_lo": x_lo, "log10_s_hi": x_hi,
                "break_inside_data": bool(x_lo <= best_params[1] <= x_hi)}


def fit_all_variants(path):
    """The Fazio-alone central (all-fields-pooled) fit and the two
    field-preferred variants, each `(fit, stats, (log10_S, log10_phi))`,
    each unweighted (`fit_counts_differential`'s default). "mean" averages
    log10 of whichever galaxy columns are populated per magnitude row --
    the pooled fit `build_joint_counts_law` weights and adds SEDS to
    (review ledger C27); kept here as the Fazio-only comparator
    (`build()`'s "IDENTITY C3" print) and as the input the field-preferred
    variants below still fit. "egs"/"qso1700" prefer that field, falling
    back to Bootes where it is silent -- their spread is the
    cosmic-variance measurement (`cosmic_variance_dex`). Each variant
    is fitted directly to the differential counts (`fazio_differential_phi`,
    `fit_counts_differential`; review ledger C3).
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
        log10_S, log10_phi = fazio_differential_phi(mag, log10_n)
        fit, stats = fit_counts_differential(log10_S, log10_phi)
        out[key] = (fit, stats, (log10_S, log10_phi))
    return out


def fazio_mean_variant_with_sigma(path):
    """Fazio's pooled ("mean") differential counts AND that pooled row's
    own published uncertainty, row-aligned: `(log10_S, log10_phi,
    sigma_dex)`, ascending in `log10_S` -- `build_joint_counts_law`'s own
    Fazio-side fit input (review ledger C27). `sigma_dex` propagates
    `FAZIO_ERR_COLUMNS`'s own per-field, per-row published error (on that
    field's `*_total` column -- the nearest stored uncertainty for the
    star-subtracted `*_galaxies` column plotted here; no separate
    galaxy-only error is tabulated, disclosed, not fabricated) through the
    SAME arithmetic mean `fit_all_variants`'s "mean" variant forms:
    `sigma_row = sqrt(sum_i err_i^2) / n_populated`, over the fields
    populated at that row (error propagation of an arithmetic mean of
    independent fields).
    """
    mag, cols = read_fazio_table(path)
    df = pd.read_csv(path)
    blk = df[df["band_um"] == COORD_BAND_UM].sort_values("mag")
    val_stack = np.vstack([cols[c] for c in FIELD_COLUMNS])
    err_stack = np.vstack([blk[c].to_numpy(dtype=float) for c in FAZIO_ERR_COLUMNS])
    with np.errstate(invalid="ignore"):
        mean_log = np.nanmean(val_stack, axis=0)
        n_populated = np.sum(np.isfinite(val_stack), axis=0)
        sigma_row = np.sqrt(np.nansum(err_stack ** 2, axis=0)) / n_populated
    log10_S, log10_phi = fazio_differential_phi(mag, mean_log)
    # Reproduce fazio_differential_phi's own mask (isfinite(mean_log)) and
    # sort (ascending log10_S) so sigma_row lines up with (log10_S, log10_phi).
    ok = np.isfinite(mean_log)
    zp_mjy = FAZIO_VEGA_ZP_JY[COORD_BAND_UM] * 1e3
    log10_S_unsorted = np.log10(zp_mjy * 10.0 ** (-0.4 * np.asarray(mag, dtype=float)[ok]))
    order = np.argsort(log10_S_unsorted)
    sigma_dex = sigma_row[ok][order]
    return log10_S, log10_phi, sigma_dex


def cosmic_variance_dex(variants):
    """The counts law's field-to-field spread (SPEC_PRIORS.md section 5.1,
    "about 0.1 dex"): the RMS difference in log10 phi(S) between the two
    field-preferred fits, over the flux range both actually constrain.
    """
    fit_egs, stats_egs, _ = variants["egs"]
    fit_qso, stats_qso, _ = variants["qso1700"]
    lo = max(stats_egs["log10_s_lo"], stats_qso["log10_s_lo"])
    hi = min(stats_egs["log10_s_hi"], stats_qso["log10_s_hi"])
    grid = np.linspace(lo, hi, 50)
    diff = fit_egs.log10_value(grid) - fit_qso.log10_value(grid)
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


def measure_point_fraction(flux_mjy_i2, ext_fl, is_galaxy,
                            s_lo_mjy=POINT_FRACTION_S_LO_MJY, s_hi_mjy=POINT_FRACTION_S_HI_MJY,
                            n_bins=POINT_FRACTION_N_BINS):
    """The IRAC point-like fraction p(S), MEASURED directly on SWIRE's
    adopted galaxies (review ledger C4: never an extrapolated fit): over
    `n_bins` log-spaced 4.5um flux bins between `s_lo_mjy` and `s_hi_mjy`,
    the fraction of galaxies (`is_galaxy`, the adopted star/galaxy split,
    not this test) whose own per-band extended-source flag (3.6, 4.5um)
    never reads "might be" or "clearly" extended (`EXT_FL_EXTENDED`) --
    the same resolved test `classify_galaxy_extended_flag` applies for
    the split itself, read here on the galaxies it already kept, not on
    the star/galaxy decision.

    Returns `(log10_s_centers, p_point, n_per_bin)`, one row per bin that
    actually holds a galaxy; an empty bin is dropped, never filled or
    interpolated across silently.
    """
    ext36, ext45 = ext_fl[:, 0], ext_fl[:, 1]
    resolved = np.isin(ext36, EXT_FL_EXTENDED) | np.isin(ext45, EXT_FL_EXTENDED)
    point = ~resolved
    gal_mask = is_galaxy & np.isfinite(flux_mjy_i2) & (flux_mjy_i2 > 0)

    edges = np.geomspace(s_lo_mjy, s_hi_mjy, n_bins + 1)
    log10_centers, p_point_rows, n_rows = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = gal_mask & (flux_mjy_i2 >= lo) & (flux_mjy_i2 < hi)
        n = int(np.sum(m))
        if n == 0:
            continue
        log10_centers.append(0.5 * (np.log10(lo) + np.log10(hi)))
        p_point_rows.append(float(np.mean(point[m])))
        n_rows.append(n)
    return np.array(log10_centers, dtype=np.float64), np.array(p_point_rows, dtype=np.float64), \
        np.array(n_rows, dtype=np.int64)


def point_source_retention_measured(log10_s_grid, log10_centers, p_point, s_lo_mjy=POINT_FRACTION_S_LO_MJY):
    """p(S) on `log10_s_grid`: `measure_point_fraction`'s own measured
    curve, linearly interpolated in log10 S over its own measured range
    and held at its own edge value past it (no measurement there either
    way -- `fit_counts_differential`'s own held-value convention, not a
    further extrapolating formula); held at 1 below `log10(s_lo_mjy)`
    (module docstring: the galaxy angular-size distribution has already
    crossed the IRAC PSF there and nothing measured contradicts it)."""
    grid = np.asarray(log10_s_grid, dtype=float)
    lo_edge, hi_edge = float(log10_centers.min()), float(log10_centers.max())
    clipped = np.clip(grid, lo_edge, hi_edge)
    interp = np.interp(clipped, log10_centers, p_point)
    return np.where(grid < np.log10(s_lo_mjy), 1.0, interp)


def build_counts_law(fazio_path, config):
    """The joint Fazio+SEDS fit (review ledger C27, owner ruling
    2026-10-21), its cosmic-variance bands and PHI_S on LOG10_S_GRID. The
    point-source correction PHI_S_POINT = PHI_S * p(S) is added
    afterward in `build`, once SWIRE's own rows are read
    (`measure_point_fraction`; review ledger C4) -- this function never
    reads SWIRE. Returns a dict ready for `write_counts` once `p_point`/
    `phi_s_point` and the measured curve are added to it.

    `variants` (the Fazio-alone "mean"/"egs"/"qso1700" fits,
    unweighted) is still returned: `cosmic_variance_dex` (Fazio's own
    field-to-field spread) reads it, and `build()`'s "IDENTITY C3" print
    compares the joint fit's level against the Fazio-alone "mean" fit and
    a direct integral of Fazio's own table.
    """
    variants = fit_all_variants(fazio_path)
    cv_dex = cosmic_variance_dex(variants)
    fit, stats, seds_result, cv_fazio_dex = build_joint_counts_law(fazio_path, config)
    log10_s_grid = build_log10_s_grid(fazio_path)
    phi_s = fit.value(10.0 ** log10_s_grid)
    return dict(fit=fit, stats=stats, cosmic_variance_dex=cv_dex,
                log10_s_grid=log10_s_grid, phi_s=phi_s, variants=variants,
                seds_result=seds_result)


def write_counts(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fit = result["fit"]
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        for name, val in zip(BrokenPowerLaw.PARAM_NAMES, fit.params):
            upper = name.upper()
            build_module.write_dataset(f, upper, np.float64(val), *REGISTRY[(_STEM, upper)])
        for name, data in (
            ("LOG10_S_GRID", result["log10_s_grid"].astype(np.float64)),
            ("PHI_S", result["phi_s"].astype(np.float64)),
            ("P_POINT", result["p_point"].astype(np.float64)),
            ("PHI_S_POINT", result["phi_s_point"].astype(np.float64)),
            ("COSMIC_VARIANCE_DEX", np.float64(result["cosmic_variance_dex"])),
            ("FIT_LOG10_S_MIN", np.float64(result["stats"]["log10_s_lo"])),
            ("FIT_LOG10_S_MAX", np.float64(result["stats"]["log10_s_hi"])),
            ("FIT_RMS_DEX", np.float64(result["stats"]["rms_dex"])),
            ("SEDS_FIELD_SCATTER_DEX", np.float64(result["stats"]["seds_field_scatter_dex"])),
            ("FIT_N_FAZIO_ROWS", np.int64(result["stats"]["n_fazio"])),
            ("FIT_N_SEDS_BINS", np.int64(result["stats"]["n_seds"])),
            ("FIT_LOG10_S_SEDS_MIN", np.float64(result["stats"]["log10_s_seds_min"])),
            ("FIT_LOG10_S_SEDS_MAX", np.float64(result["stats"]["log10_s_seds_max"])),
            ("POINT_FRACTION_LOG10_S_CENTERS", result["point_log10_centers"].astype(np.float64)),
            ("POINT_FRACTION_MEASURED", result["point_measured"].astype(np.float64)),
            ("POINT_FRACTION_N_SWIRE", result["point_n_swire"].astype(np.int64)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])


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


def fazio_star_differential_phi(fazio_path):
    """Fazio's pooled (mean-of-three-fields) star DIFFERENTIAL counts,
    phi_star(S), deg^-2 mJy^-1 -- the star-count analogue of
    `fazio_pooled_star_counts` (cumulative) matching the quantity the
    counts-law fit actually uses (differential, review ledger C3):
    `fazio_differential_phi` applied to `STAR_FIELD_COLUMNS`'s own pooled
    mean. `expected_star_phi_at_latitude` transports this to a SEDS
    field's own latitude (review ledger C27), exactly as
    `expected_star_count` transports the cumulative form to SWIRE's.
    """
    mag, cols = read_fazio_table(fazio_path, columns=STAR_FIELD_COLUMNS)
    stack = np.vstack([cols[c] for c in STAR_FIELD_COLUMNS])
    with np.errstate(invalid="ignore"):
        mean_log = np.nanmean(stack, axis=0)
    return fazio_differential_phi(mag, mean_log)


def expected_star_phi_at_latitude(fazio_path, log10_s, abs_b_deg):
    """phi_star(S) at `log10_s` (array or scalar), transported from
    Fazio's own three fields' mean latitude to `abs_b_deg` by
    `STAR_COUNT_LATITUDE_GRADIENT_DEX_PER_DEG` -- the differential
    analogue of `expected_star_count`'s transport, read by
    `seds_differential_counts` for each SEDS field's own star
    subtraction (review ledger C27). Held at `fazio_star_differential_
    phi`'s own edge value beyond its tabulated range (no measurement
    either side of it), the same held-edge convention
    `point_source_retention_measured` uses.
    """
    log10_s_star, log10_phi_star = fazio_star_differential_phi(fazio_path)
    lo, hi = float(log10_s_star.min()), float(log10_s_star.max())
    clipped = np.clip(np.asarray(log10_s, dtype=float), lo, hi)
    interp = np.interp(clipped, log10_s_star, log10_phi_star)
    b_fazio_mean = float(np.mean(FAZIO_FIELD_ABS_B_DEG))
    shift = STAR_COUNT_LATITUDE_GRADIENT_DEX_PER_DEG * (float(abs_b_deg) - b_fazio_mean)
    return 10.0 ** (interp + shift)


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


def _scosmos_flux_mjy(config, column):
    """S-COSMOS's own measured flux, mJy, one IRAC channel column, over
    rows the release's own overall quality flag passes (`flag == 0`) --
    the one read `scosmos_8um_cumulative` (8um, `flux_c4_2`) and the
    per-bin 4.5um check in `build()` (`flux_c2_2`) both share.
    """
    path = f"{config.data_root}/sky/download/scosmos/scosmos_irac.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"gal: no S-COSMOS catalogue at {path!r} -- run the "
            f"'sesnaimpute.sky.download.scosmos.build' RUNBOOK line")
    df = pd.read_csv(path, usecols=[column, "flag"])
    flux_mjy = df[column].to_numpy(dtype=float) / 1000.0
    good = (df["flag"].to_numpy(dtype=float) == 0) & np.isfinite(flux_mjy)
    return flux_mjy[good]


def scosmos_8um_cumulative(config, threshold_mjy):
    """S-COSMOS's own measured 8um N(>S) at `threshold_mjy`, deg^-2 --
    S-COSMOS carries no native stellarity (S-L5), so no star-galaxy
    separation is applied here; it is the deep single-field cross-check,
    not the selection population.
    """
    flux_mjy = _scosmos_flux_mjy(config, "flux_c4_2")
    return float(np.sum(flux_mjy >= threshold_mjy)) / SCOSMOS_AREA_DEG2


def _read_ashby_seds_field_flux_mjy(path):
    """One SEDS field file's own quality-passing 4.5um fluxes, mJy (AB
    system, `ASHBY_SEDS_AB_ZP_JY`): rows whose per-source quality-flag
    sum reads clean (`f45 == 0`) and whose PSF magnitude is a valid
    detection (`mag45psf > 0` -- the release's own "0.00" code for no
    valid detection, ReadMe note 1). The one read `ashby_seds_field_
    fluxes` shares between the pooled cumulative check and the
    differential fit input (review ledger C27: no second copy of the
    binning).
    """
    with gzip.open(path, "rt") as fh:
        df = pd.read_fwf(fh, colspecs=list(ASHBY_SEDS_COLSPECS), names=list(ASHBY_SEDS_COLUMN_NAMES))
    good = (df["f45"] == 0) & (df["mag45psf"] > 0)
    mag = df.loc[good, "mag45psf"].to_numpy(dtype=float)
    return ASHBY_SEDS_AB_ZP_JY * 10.0 ** (-0.4 * mag) * 1e3


def ashby_seds_field_fluxes(config):
    """`{field_name: flux_mjy}`, `ASHBY_SEDS_FIELD_NAMES` keys -- each
    SEDS field's own quality-passing 4.5um fluxes
    (`_read_ashby_seds_field_flux_mjy`), the one read `ashby_seds_
    cumulative` (pooled total-count check) and `seds_differential_counts`
    (the fit's own per-bin, per-field input) both build on.
    """
    base = f"{config.data_root}/sky/download/ashby2013_seds"
    out = {}
    for (name, _area), field_name in zip(ASHBY_SEDS_FIELDS, ASHBY_SEDS_FIELD_NAMES):
        path = f"{base}/{name}"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"gal: no Ashby 2013 SEDS table at {path!r} -- made by the "
                f"RUNBOOKtp.sh line sesnaimpute.sky.download.ashby2013_seds.build")
        out[field_name] = _read_ashby_seds_field_flux_mjy(path)
    return out


def ashby_seds_cumulative(config, threshold_mjy):
    """Ashby et al. (2013, ApJ 769, 80) SEDS's own measured total 4.5um
    source count N(>`threshold_mjy`) over its five full-depth fields
    (`ASHBY_SEDS_AREA_DEG2` = 1.46 deg^2 combined), galaxies deg^-2 -- the
    RAW (star+galaxy) level, printed beside the fit as a sanity check on
    the catalogue read itself (its own published total source density at
    0.1mJy is about 1,709 deg^-2, review ledger C27). The release carries
    no star-galaxy separation (module docstring), so this over-counts by
    whatever stellar contamination its deep, high-latitude fields still
    carry at the flux read -- `seds_differential_counts` is the star-
    subtracted, differential form the fit itself uses (review ledger C27,
    overruling C3's "SEDS beside, never fit").
    """
    flux_mjy = np.concatenate(list(ashby_seds_field_fluxes(config).values()))
    n = int(np.sum(flux_mjy >= threshold_mjy))
    return n / ASHBY_SEDS_AREA_DEG2, n


def seds_differential_counts(config, fazio_path, bin_width_dex=SEDS_FIT_BIN_WIDTH_DEX):
    """SEDS's own star-subtracted differential galaxy counts, pooled over
    its five fields, in `bin_width_dex`-wide log10 S bins spanning the
    data's own observed flux range (review ledger C27, the counts law's
    joint-fit input).

    Per field: the RAW (star+galaxy) count per bin
    (`ashby_seds_field_fluxes`'s own selection); the expected star count
    per bin from Fazio's own star law transported to that field's own
    latitude (`expected_star_phi_at_latitude`, `ASHBY_SEDS_FIELD_ABS_B_
    DEG`), subtracted. Pooled across the five fields' combined area
    (`ASHBY_SEDS_AREA_DEG2`) for `phi_pool`, the star-subtracted
    differential density.

    `sigma_poisson_dex` is the POOLED RAW count's own Poisson error
    (`1 / (ln10 sqrt(N))`) -- never the star-subtracted count's: the
    expected-star term is a deterministic transport, not an independent
    draw, so it carries no counting noise of its own. `seds_field_
    scatter_dex` is the field-to-field (cosmic-variance) scatter the
    joint fit's own weighting needs (owner ruling 2026-10-21): at each
    bin where at least `SEDS_SCATTER_MIN_FIELDS` fields each carry at
    least `SEDS_SCATTER_MIN_N_FIELD` raw sources and a positive star-
    subtracted count, the RMS of that bin's own per-field log10 density
    around their mean; pooled into one scalar by a raw-count-weighted RMS
    over those bins (`per_bin_scatter_dex`/`per_bin_n_fields` carry every
    bin's own value for inspection, printed by `build()`).

    Returns a dict: `log10_s_centers`, `phi_pool`, `n_raw_pool`,
    `n_gal_pool`, `sigma_poisson_dex`, `seds_field_scatter_dex`,
    `per_bin_scatter_dex`, `per_bin_n_fields`, `bin_width_dex`.
    """
    field_fluxes = ashby_seds_field_fluxes(config)
    area_by_field = dict(zip(ASHBY_SEDS_FIELD_NAMES, (a for _n, a in ASHBY_SEDS_FIELDS)))
    all_flux = np.concatenate(list(field_fluxes.values()))
    lo = np.floor(np.log10(all_flux.min()) / bin_width_dex) * bin_width_dex
    hi = np.ceil(np.log10(all_flux.max()) / bin_width_dex) * bin_width_dex
    edges = np.arange(lo, hi + bin_width_dex / 2.0, bin_width_dex)
    centers = 0.5 * (edges[:-1] + edges[1:])
    s_center = 10.0 ** centers
    d_s = s_center * np.log(10.0) * bin_width_dex

    n_raw, n_star = {}, {}
    for field_name, abs_b in zip(ASHBY_SEDS_FIELD_NAMES, ASHBY_SEDS_FIELD_ABS_B_DEG):
        n_raw[field_name], _ = np.histogram(np.log10(field_fluxes[field_name]), bins=edges)
        phi_star_field = expected_star_phi_at_latitude(fazio_path, centers, abs_b)
        n_star[field_name] = phi_star_field * d_s * area_by_field[field_name]

    n_raw_pool = sum(n_raw.values())
    n_star_pool = sum(n_star.values())
    n_gal_pool = n_raw_pool - n_star_pool
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_pool = n_gal_pool / ASHBY_SEDS_AREA_DEG2 / d_s
    sigma_poisson_dex = 1.0 / (np.log(10.0) * np.sqrt(np.maximum(n_raw_pool, 1)))

    per_bin_scatter_dex = np.full(centers.size, np.nan)
    per_bin_n_fields = np.zeros(centers.size, dtype=np.int64)
    for i in range(centers.size):
        log_phis = []
        for field_name in ASHBY_SEDS_FIELD_NAMES:
            n_r = n_raw[field_name][i]
            n_g = n_r - n_star[field_name][i]
            if n_r >= SEDS_SCATTER_MIN_N_FIELD and n_g > 0:
                log_phis.append(np.log10(n_g / area_by_field[field_name] / d_s[i]))
        if len(log_phis) >= SEDS_SCATTER_MIN_FIELDS:
            arr = np.asarray(log_phis, dtype=float)
            per_bin_scatter_dex[i] = float(np.sqrt(np.mean((arr - arr.mean()) ** 2)))
            per_bin_n_fields[i] = len(log_phis)

    usable = np.isfinite(per_bin_scatter_dex)
    if usable.any():
        w = n_raw_pool[usable].astype(float)
        seds_field_scatter_dex = float(np.sqrt(np.sum(w * per_bin_scatter_dex[usable] ** 2) / np.sum(w)))
    else:
        seds_field_scatter_dex = 0.0

    return dict(log10_s_centers=centers, phi_pool=phi_pool, n_raw_pool=n_raw_pool,
                n_gal_pool=n_gal_pool, sigma_poisson_dex=sigma_poisson_dex,
                seds_field_scatter_dex=seds_field_scatter_dex,
                per_bin_scatter_dex=per_bin_scatter_dex, per_bin_n_fields=per_bin_n_fields,
                bin_width_dex=bin_width_dex)


def build_joint_counts_law(fazio_path, config):
    """The ONE counts-law fit this product adopts (owner ruling
    2026-10-21, review ledger C27, overruling C3's "SEDS beside, never
    fit"): Fazio et al. (2004)'s pooled differential counts (`fazio_mean_
    variant_with_sigma`) AND Ashby et al. (2013) SEDS's star-subtracted,
    pooled differential counts (`seds_differential_counts`, restricted to
    `LOG10_S_GRID`'s own faint edge and brighter -- SWIRE's I2 5-sigma
    depth, itself a decade or more brighter than SEDS's own completeness
    turnover, measured directly below as the peak of its pooled raw
    counts; the law is read only inside the tabulated grid in any case,
    module docstring), each point weighted by its own Poisson error PLUS
    that dataset's own field-to-field (cosmic-variance) scatter (Fazio:
    `cosmic_variance_dex`'s existing egs/qso1700 spread; SEDS: `seds_
    differential_counts`'s own re-measured five-field spread) -- no hand
    weighting of one dataset against the other.

    Returns `(fit, stats, seds_result, cv_fazio_dex)`; `stats` extends
    `fit_counts_differential`'s own dict with `n_fazio`, `n_seds`,
    `cv_fazio_dex`, `seds_field_scatter_dex` and `log10_s_lo_fit` (the
    SEDS-side faint cut actually applied).
    """
    log10_s_fazio, log10_phi_fazio, sigma_fazio_err = fazio_mean_variant_with_sigma(fazio_path)
    cv_fazio_dex = cosmic_variance_dex(fit_all_variants(fazio_path))
    sigma_fazio = np.sqrt(sigma_fazio_err ** 2 + cv_fazio_dex ** 2)

    seds_result = seds_differential_counts(config, fazio_path)
    log10_s_lo_fit = np.log10(SWIRE_5SIGMA_UJY["I2"] / 1000.0)
    turnover_log10_s = float(seds_result["log10_s_centers"][np.argmax(seds_result["n_raw_pool"])])
    print(f"gal: SEDS own pooled raw-count turnover (completeness) peaks at "
          f"S={1e3 * 10 ** turnover_log10_s:.3f}uJy; the fit's own SEDS-side faint cut sits at "
          f"S={1e3 * 10 ** log10_s_lo_fit:.3f}uJy ({log10_s_lo_fit - turnover_log10_s:+.2f} dex brighter)")
    valid = (seds_result["n_gal_pool"] > 0) & (seds_result["log10_s_centers"] >= log10_s_lo_fit)
    log10_s_seds = seds_result["log10_s_centers"][valid]
    log10_phi_seds = np.log10(seds_result["phi_pool"][valid])
    sigma_seds = np.sqrt(seds_result["sigma_poisson_dex"][valid] ** 2
                          + seds_result["seds_field_scatter_dex"] ** 2)

    x = np.concatenate([log10_s_fazio, log10_s_seds])
    y = np.concatenate([log10_phi_fazio, log10_phi_seds])
    sigma = np.concatenate([sigma_fazio, sigma_seds])
    fit, stats = fit_counts_differential(x, y, sigma=sigma)
    stats.update(n_fazio=int(log10_s_fazio.size), n_seds=int(log10_s_seds.size),
                 cv_fazio_dex=float(cv_fazio_dex),
                 seds_field_scatter_dex=float(seds_result["seds_field_scatter_dex"]),
                 log10_s_seds_min=float(log10_s_seds.min()) if log10_s_seds.size else float("nan"),
                 log10_s_seds_max=float(log10_s_seds.max()) if log10_s_seds.size else float("nan"))
    return fit, stats, seds_result, cv_fazio_dex


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Writes the survey-wide galaxy number-counts law (`phi(S)` on
    `LOG10_S_GRID`, fitted jointly to Fazio et al. (2004)'s and Ashby et
    al. (2013) SEDS's own differential counts, each star-subtracted and
    weighted by its own error, review ledger C27, owner ruling
    2026-10-21, overruling C3's "SEDS beside, never fit") and the
    measured IRAC point-source retention `P_POINT` (review ledger C4),
    and prints the star-galaxy split with its consistency checks against
    SWIRE and S-COSMOS. Survey-wide; `regions` is accepted and ignored.
    """
    st = progress.Stage("population.gal")

    fazio_path = f"{config.data_root}/sky/download/fazio2004/fazio2004_table1_irac_counts.csv"
    if not os.path.exists(fazio_path):
        raise FileNotFoundError(
            f"gal: no Fazio 2004 table at {fazio_path!r} -- see the "
            f"'sky/download/fazio2004' RUNBOOK comment line: manual "
            f"acquisition, no reachable download URL")

    counts_result = build_counts_law(fazio_path, config)
    fit = counts_result["fit"]
    stats = counts_result["stats"]
    seds_result = counts_result["seds_result"]
    print(f"gal: counts law (fitted JOINTLY to Fazio's and SEDS's own differential counts, break "
          f"free, review ledger C27): "
          f"log10_A={fit.params[0]:.4f} log10_S_break={fit.params[1]:.4f} "
          f"alpha_faint={fit.params[2]:.4f} alpha_bright={fit.params[3]:.4f} "
          f"smoothness={fit.params[4]:.4f} rms={stats['rms_dex']:.4f} dex (unweighted), "
          f"{stats['rms_weighted']:.4f} (weighted, dimensionless) "
          f"combined_data_range=[{stats['log10_s_lo']:.4f}, {stats['log10_s_hi']:.4f}] "
          f"break_inside_data={stats['break_inside_data']}")
    print(f"gal: joint fit provenance: Fazio rows={stats['n_fazio']} (own published *_err + "
          f"cv_fazio={stats['cv_fazio_dex']:.4f} dex), SEDS bins={stats['n_seds']} "
          f"(own Poisson + seds_field_scatter={stats['seds_field_scatter_dex']:.4f} dex), "
          f"SEDS's own fit-constrained flux range=[{stats['log10_s_seds_min']:.4f}, "
          f"{stats['log10_s_seds_max']:.4f}] log10 mJy")
    print("gal: SEDS field-to-field scatter, per bin (n_fields>=%d, n_raw>=%d per field; "
          "bins below are from seds_differential_counts, same bins the fit itself is built on):"
          % (SEDS_SCATTER_MIN_FIELDS, SEDS_SCATTER_MIN_N_FIELD))
    for i in range(seds_result["log10_s_centers"].size):
        if np.isfinite(seds_result["per_bin_scatter_dex"][i]):
            s_c = 1e3 * 10 ** seds_result["log10_s_centers"][i]
            print(f"gal:   S={s_c:8.2f}uJy  n_fields={int(seds_result['per_bin_n_fields'][i])}  "
                  f"scatter={seds_result['per_bin_scatter_dex'][i]:.4f} dex")

    # --- C3 identity, now read as a disclosure rather than an acceptance
    # bound: the joint fit also answers to SEDS (review ledger C27), so
    # its level is no longer expected to match a direct integral of
    # Fazio's OWN table to the earlier 0.01 dex (a Fazio-only-fit bound).
    # Reported beside the Fazio-ALONE "mean" fit's own identical check
    # (still exact to 0.01 dex, since that fit answers to Fazio alone),
    # to isolate how much of any difference is the SEDS pull.
    mag_tbl, cols_tbl = read_fazio_table(fazio_path)
    stack_tbl = np.vstack([cols_tbl[c] for c in FIELD_COLUMNS])
    with np.errstate(invalid="ignore"):
        mean_log_tbl = np.nanmean(stack_tbl, axis=0)
    log10_s_tbl, log10_phi_tbl = fazio_differential_phi(mag_tbl, mean_log_tbl)
    log10_s_grid = counts_result["log10_s_grid"]
    d_log10_s = float(log10_s_grid[1] - log10_s_grid[0])
    s_grid = 10.0 ** log10_s_grid
    log10_phi_direct = np.interp(log10_s_grid, log10_s_tbl, log10_phi_tbl)
    a_gal_fit = float(np.sum(counts_result["phi_s"] * s_grid * d_log10_s * np.log(10.0)))
    a_gal_direct = float(np.sum(10.0 ** log10_phi_direct * s_grid * d_log10_s * np.log(10.0)))
    identity_c3_dex = float(np.log10(a_gal_fit / a_gal_direct))
    fazio_only_fit, _fazio_only_stats, _ = counts_result["variants"]["mean"]
    a_gal_fazio_only = float(np.sum(fazio_only_fit.value(s_grid) * s_grid * d_log10_s * np.log(10.0)))
    identity_c3_fazio_only_dex = float(np.log10(a_gal_fazio_only / a_gal_direct))
    print(f"gal: IDENTITY C3: A_GAL from the JOINT fit over LOG10_S_GRID = {a_gal_fit:.1f} deg^-2; "
          f"from a direct integral of Fazio Table 1's own tabulated differential counts over the "
          f"same grid = {a_gal_direct:.1f} deg^-2; level difference = {identity_c3_dex:+.4f} dex "
          f"(the Fazio-ALONE fit's own identical check: {identity_c3_fazio_only_dex:+.4f} dex, "
          f"acceptance 0.01 dex -- the joint fit is not held to that bound, since it also answers "
          f"to SEDS)")

    # --- Ashby et al. (2013) SEDS, reported beside the joint fit's own
    # level, both RAW (sanity check on the catalogue read: SEDS's own
    # published total source density at 0.1mJy is about 1,709 deg^-2) and
    # star-subtracted (the fit's own input quantity, review ledger C27).
    ashby_density_by_check = {}
    for s_check in (0.1, 1.0):
        ashby_density, ashby_n = ashby_seds_cumulative(config, s_check)
        ashby_density_by_check[s_check] = ashby_density
        fazio_cum = float(fit.cumulative_numeric(s_check, stats["log10_s_hi"]))
        print(f"gal: SEDS RAW beside joint fit, N(>{s_check}mJy): "
              f"seds_raw={ashby_density:.1f} deg^-2 (n={ashby_n} of {ASHBY_SEDS_AREA_DEG2:.2f} deg^2, "
              f"no star-galaxy separation) joint_fit={fazio_cum:.1f} deg^-2 "
              f"ratio(fit/seds_raw)={fazio_cum / ashby_density:.3f}")
    seds_centers = seds_result["log10_s_centers"]
    seds_above = seds_centers >= np.log10(0.1)
    seds_n_gal_above_01 = float(np.sum(seds_result["n_gal_pool"][seds_above]))
    print(f"gal: SEDS star-subtracted N(>0.1mJy) = {seds_n_gal_above_01 / ASHBY_SEDS_AREA_DEG2:.1f} "
          f"deg^-2 (galaxy-only; compare to the RAW 1,709 deg^-2 catalogue-level check above, which "
          f"includes stars)")

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

    # --- C4: p(S), MEASURED directly on SWIRE's adopted galaxies over
    # 0.3-20mJy, never extrapolated. S-COSMOS carries no native
    # point/extended flag (module docstring) so it cannot contribute
    # this measurement; disclosed below, not substituted.
    point_log10_centers, point_measured, point_n_swire = measure_point_fraction(
        flux_mjy_all[:, IRAC_BAND_KEYS.index("I2")], ext_fl_all, is_galaxy_all)
    print(f"gal: IDENTITY C4: p(S) measured directly on SWIRE's {int(is_galaxy_all.sum())} adopted "
          f"galaxies over {POINT_FRACTION_S_LO_MJY}-{POINT_FRACTION_S_HI_MJY}mJy "
          f"({POINT_FRACTION_N_BINS} log-spaced bins, {int(point_n_swire.sum())} galaxies total in range):")
    for s_target in (1.0, 18.0):
        idx = int(np.argmin(np.abs(point_log10_centers - np.log10(s_target))))
        s_bin = 10.0 ** point_log10_centers[idx]
        print(f"gal:   p(S~{s_target}mJy) = {point_measured[idx]:.4f} at the measured bin centered "
              f"on {s_bin:.3f}mJy, n={int(point_n_swire[idx])} SWIRE galaxies")
    print("gal: S-COSMOS's own IRAC table carries no native point/extended-source flag (fl_c1/fl_c3 "
          "and fl_c2/fl_c4 are identical row for row, a per-detector-pair quality flag, not a "
          "morphology classifier; studies/S-L5_four_band_galaxy_catalogue.md already records this) "
          "-- p(S) is measured on SWIRE alone")

    p_point = point_source_retention_measured(log10_s_grid, point_log10_centers, point_measured)
    phi_s_point = counts_result["phi_s"] * p_point
    counts_result.update(p_point=p_point, phi_s_point=phi_s_point,
                          point_log10_centers=point_log10_centers, point_measured=point_measured,
                          point_n_swire=point_n_swire)
    counts_path = config_module.product_path(config, "population", "gal", "counts", "survey")
    write_counts(counts_path, counts_result)
    print(f"gal: wrote {counts_path}")

    print(f"gal: PHI_S_POINT <= PHI_S everywhere: "
          f"{bool(np.all(phi_s_point <= counts_result['phi_s']))}; "
          f"p(S) held at 1 below {POINT_FRACTION_S_LO_MJY}mJy: "
          f"{bool(np.all(p_point[s_grid < POINT_FRACTION_S_LO_MJY] == 1.0))}")

    a_gal_before = float(np.sum(counts_result["phi_s"] * s_grid * d_log10_s * np.log(10.0)))
    a_gal_after = float(np.sum(phi_s_point * s_grid * d_log10_s * np.log(10.0)))
    print(f"gal: A_GAL (sum phi(S) S dS, deg^-2): before p(S)={a_gal_before:.1f}, after={a_gal_after:.1f}")

    swire_45_01 = swire_band_cumulative(flux_mjy_all[:, IRAC_BAND_KEYS.index("I2")], is_galaxy_all, 0.1)
    swire_45_1 = swire_band_cumulative(flux_mjy_all[:, IRAC_BAND_KEYS.index("I2")], is_galaxy_all, 1.0)
    law_45_01 = float(fit.cumulative_numeric(0.1, stats["log10_s_hi"]))
    law_45_1 = float(fit.cumulative_numeric(1.0, stats["log10_s_hi"]))
    print(f"gal: law/SWIRE 4.5um N(>0.1mJy): law={law_45_01:.1f} swire={swire_45_01:.1f} "
          f"ratio={law_45_01 / swire_45_01:.3f}")
    print(f"gal: law/SWIRE 4.5um N(>1mJy): law={law_45_1:.1f} swire={swire_45_1:.1f} "
          f"ratio={law_45_1 / swire_45_1:.3f}")

    swire_80_01 = swire_band_cumulative(flux_mjy_all[:, IRAC_BAND_KEYS.index("I4")], is_galaxy_all, 0.1)
    scosmos_80_01 = scosmos_8um_cumulative(config, 0.1)
    print(f"gal: S-COSMOS/SWIRE 8um N(>0.1mJy): scosmos={scosmos_80_01:.1f} swire={swire_80_01:.1f} "
          f"ratio={scosmos_80_01 / swire_80_01:.3f}" if swire_80_01 > 0 else
          "gal: S-COSMOS/SWIRE 8um N(>0.1mJy): swire count is zero, ratio undefined")

    # --- Every bin's own residual (law/data), 0.25 dex, 6uJy to 1mJy
    # (the brief's own identities table): Fazio (its own tabulated
    # differential counts, interpolated), SEDS (star-subtracted, this
    # fit's own input), SWIRE (star-subtracted, the adopted split), and
    # S-COSMOS 4.5um (`flux_c2_2`, RAW -- no native stellarity, module
    # docstring, so never star-subtracted; a level cross-check only).
    # The edges are phase-locked to multiples of SEDS_FIT_BIN_WIDTH_DEX
    # (floor/ceil, not `log10(0.006)` itself) so these bin centers land
    # exactly on `seds_differential_counts`'s own bin centers -- any other
    # phase never matches them and every SEDS lookup below reads NaN.
    report_lo = np.floor(np.log10(0.006) / SEDS_FIT_BIN_WIDTH_DEX) * SEDS_FIT_BIN_WIDTH_DEX
    report_hi = np.ceil(np.log10(1.0) / SEDS_FIT_BIN_WIDTH_DEX) * SEDS_FIT_BIN_WIDTH_DEX
    report_edges = np.arange(report_lo, report_hi + SEDS_FIT_BIN_WIDTH_DEX / 2.0, SEDS_FIT_BIN_WIDTH_DEX)
    report_centers = 0.5 * (report_edges[:-1] + report_edges[1:])
    s_report = 10.0 ** report_centers
    d_s_report = s_report * np.log(10.0) * SEDS_FIT_BIN_WIDTH_DEX
    law_report = fit.value(s_report)

    log10_phi_fazio_report = np.interp(report_centers, log10_s_tbl, log10_phi_tbl,
                                        left=np.nan, right=np.nan)
    phi_fazio_report = 10.0 ** log10_phi_fazio_report

    swire_i2 = flux_mjy_all[:, IRAC_BAND_KEYS.index("I2")]
    swire_counts, _ = np.histogram(np.log10(swire_i2[is_galaxy_all & np.isfinite(swire_i2)]), bins=report_edges)
    phi_swire_report = swire_counts / SWIRE_AREA_DEG2 / d_s_report

    scosmos_flux_45 = _scosmos_flux_mjy(config, "flux_c2_2")
    scosmos_counts, _ = np.histogram(np.log10(scosmos_flux_45[scosmos_flux_45 > 0]), bins=report_edges)
    phi_scosmos_report = scosmos_counts / SCOSMOS_AREA_DEG2 / d_s_report

    seds_centers_all = seds_result["log10_s_centers"]
    phi_seds_report = np.full(report_centers.size, np.nan)
    for i, rc in enumerate(report_centers):
        j = int(np.argmin(np.abs(seds_centers_all - rc)))
        if abs(seds_centers_all[j] - rc) < 1e-6 and seds_result["n_gal_pool"][j] > 0:
            phi_seds_report[i] = seds_result["phi_pool"][j]

    print("gal: law vs data, per 0.25 dex bin, 6uJy-1mJy (ratio = law / data):")
    print("gal:   S(uJy)       law         fazio   ratio   seds(sub)  ratio   swire(sub)  ratio   "
          "scosmos(raw)  ratio")
    for i, rc in enumerate(report_centers):
        def _r(val):
            return f"{law_report[i] / val:7.3f}" if (np.isfinite(val) and val > 0) else "    -- "
        print(f"gal:   {1e3 * s_report[i]:8.2f}  {law_report[i]:10.4g}  "
              f"{phi_fazio_report[i]:10.4g} {_r(phi_fazio_report[i])}  "
              f"{phi_seds_report[i]:10.4g} {_r(phi_seds_report[i])}  "
              f"{phi_swire_report[i]:10.4g} {_r(phi_swire_report[i])}  "
              f"{phi_scosmos_report[i]:10.4g} {_r(phi_scosmos_report[i])}")

    # --- N(>0.1mJy), star-subtracted SEDS, vs SEDS's own published
    # total (RAW, 1,709 deg^-2) check above.
    print(f"gal: SEDS N(>0.1mJy): law={float(fit.cumulative_numeric(0.1, stats['log10_s_hi'])):.1f} "
          f"deg^-2; seds_star_subtracted={seds_n_gal_above_01 / ASHBY_SEDS_AREA_DEG2:.1f} deg^-2; "
          f"seds_raw={ashby_density_by_check[0.1]:.1f} deg^-2 (SEDS's own published total, for "
          f"reference, ~1709 deg^-2)")

    # --- The owner's own standard 0.5 dex bins (18-56uJy and its two
    # neighbours, matching c18_diag/reproduce.py's edges) and the
    # second-break decision: a second break is added ONLY if the 18-56uJy
    # bin's own residual exceeds its data's own combined (Poisson +
    # field-scatter) band with one break (the brief's own trigger).
    std_edges = np.arange(-2.25, 1.01, 0.5)
    std_centers = 0.5 * (std_edges[:-1] + std_edges[1:])
    s_std = 10.0 ** std_centers
    d_s_std = s_std * np.log(10.0) * 0.5
    law_std = fit.value(s_std)

    field_fluxes_for_std = ashby_seds_field_fluxes(config)
    n_raw_std = {}
    for field_name in ASHBY_SEDS_FIELD_NAMES:
        n_raw_std[field_name], _ = np.histogram(np.log10(field_fluxes_for_std[field_name]), bins=std_edges)
    n_raw_std_pool = sum(n_raw_std.values())
    n_star_std_pool = np.zeros_like(n_raw_std_pool, dtype=float)
    for field_name, abs_b in zip(ASHBY_SEDS_FIELD_NAMES, ASHBY_SEDS_FIELD_ABS_B_DEG):
        area_f = dict(zip(ASHBY_SEDS_FIELD_NAMES, (a for _n, a in ASHBY_SEDS_FIELDS)))[field_name]
        phi_star_f = expected_star_phi_at_latitude(fazio_path, std_centers, abs_b)
        n_star_std_pool += phi_star_f * d_s_std * area_f
    n_gal_std_pool = n_raw_std_pool - n_star_std_pool
    phi_std = n_gal_std_pool / ASHBY_SEDS_AREA_DEG2 / d_s_std
    sigma_poisson_std = 1.0 / (np.log(10.0) * np.sqrt(np.maximum(n_raw_std_pool, 1)))
    sigma_std_total = np.sqrt(sigma_poisson_std ** 2 + stats["seds_field_scatter_dex"] ** 2)

    print("gal: owner's standard 0.5 dex bins, law (after the joint fit) vs SEDS star-subtracted "
          "(18-56uJy and its two neighbours are the C16/C24f disclosure bins):")
    resid_18_56 = None
    sigma_18_56 = None
    for i in range(std_centers.size):
        lo_u, hi_u = 1e3 * 10 ** std_edges[i], 1e3 * 10 ** std_edges[i + 1]
        resid_dex = float(np.log10(law_std[i]) - np.log10(phi_std[i])) if phi_std[i] > 0 else float("nan")
        print(f"gal:   {lo_u:7.1f}-{hi_u:7.1f}uJy  law={law_std[i]:.4g}  seds_sub={phi_std[i]:.4g}  "
              f"sigma_seds={sigma_std_total[i]:.4f} dex  resid(law/seds)={resid_dex:+.4f} dex")
        if np.isclose(std_edges[i], np.log10(0.0178), atol=0.01):
            resid_18_56 = resid_dex
            sigma_18_56 = sigma_std_total[i]
    if resid_18_56 is not None:
        exceeds = abs(resid_18_56) > sigma_18_56
        print(f"gal: SECOND-BREAK CHECK at 18-56uJy: residual={resid_18_56:+.4f} dex, "
              f"SEDS's own band={sigma_18_56:.4f} dex -- "
              f"{'EXCEEDS the band' if exceeds else 'within the band'}; "
              f"{'a second break would be considered' if exceeds else 'no second break added (parsimony)'}")
    else:
        print("gal: SECOND-BREAK CHECK: could not locate the 18-56uJy standard bin edge; skipped")

    st.done(counts_path, rms_dex=stats["rms_dex"], cosmic_variance_dex=counts_result["cosmic_variance_dex"],
            identity_c3_dex=identity_c3_dex)


if __name__ == "__main__":
    run(build)
