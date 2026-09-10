"""The PAH-contamination probability curve `P(q)` (`SPEC_PRIORS.md` section
4; `IMPLEMENTATION.md` section 6, stage 2).

A source's aperture is contaminated by extended PAH nebular emission with
probability `P(q)`, `q = F_lim,8(s) / F_8,pred(s)` -- the source's own 8
micron completeness limit over its predicted photospheric 8 micron flux,
so `q` is large where the local field is faint relative to the star (a
nebulous, shallow position) and small where it is bright. `P(q)` is
measured once, survey-wide, directly from SESNA photometry: for every
source with measured 3.6, 4.5 and 8.0 micron photometry, an object counts
toward the curve if its own 8 micron flux exceeds its predicted
photospheric 8 micron flux by more than 3 sigma.

Two pieces are external to SESNA, both from the region's own TRILEGAL
retained field-star population (`population.field_stars`), which carries
intrinsic (undimmed) fluxes only (module `field_stars.py`'s own
docstring):

  - the photospheric colour relation, [4.5]-[8.0] as the median function of
    [3.6]-[4.5] in narrow bins of the latter -- what predicts a source's
    own photospheric 8 micron flux from its own 3.6 and 4.5 micron fluxes;
  - the population's constant [3.6]-[4.5] median -- what predicts a
    source's own photospheric 4.5 micron flux from its own 3.6 micron
    flux, for the 4.5 micron excess test that separates a circumstellar
    disk (whose excess grows smoothly with wavelength and shows already
    at 4.5 micron, where there is no PAH feature) from PAH contamination
    (which does not).

Every TRILEGAL retained star enters the colour relation with equal weight
(SPEC_PRIORS.md section 0.3, C3, "population-weighted"): the STAR anchor
reweighting that turns this same population into a count is a later
stage's own step and plays no part here.

De-reddening (SPEC_PRIORS.md section 4, "the photospheric prediction"): a
source's own observed 3.6 and 4.5 micron fluxes are corrected by its own
adopted column before its colour enters the TRILEGAL relation --
`F_i,0 = F_i * 10**(0.4*a*kappa_i(a))`, `a` the source's own `A_COL_K`
(`sky/derived/adopted/column_adopted_source`) and `kappa_i(a)` the
blended diffuse/dense extinction-curve ratio at that column
(`population.selection.kappa_hybrid`, SPEC_PRIORS.md section 1.3) -- the
relation itself is built from TRILEGAL's own undimmed fluxes, so a
reddened source colour would read the wrong point on it. The 8 micron and
4.5 micron predictions are made in this de-reddened frame and reddened
back with the source's own `kappa_8` or `kappa_4.5` before comparison to
the source's *observed* flux, so both excess tests live in the observed,
catalogued frame throughout.

Sigma on each excess test combines the catalogue's own flux error with a
relation term measured on the population itself, not the TRILEGAL
relation's own binned scatter: the robust width -- 1.4826 times the
median absolute deviation, the standard Gaussian-equivalent scaling -- of
the observed-minus-predicted residual, in magnitudes, on the shelf
population (`0.1 <= q <= 2`, no 4.5 micron excess), one number
survey-wide per test (`RESIDUAL_WIDTH_MAG` for [4.5]-[8.0],
`RESIDUAL_WIDTH_45_MAG` for [3.6]-[4.5]), converted to a flux-domain sigma
per source from that source's own predicted flux level. The shelf
population that defines the widths is itself selected by the 4.5 micron
excess test the width feeds, so the widths are measured twice: once with
no 4.5 micron cut at all, to get a first sigma for the 4.5 micron test and
flag its excess sources; then again on the shelf with that cut applied,
which is what is shipped and used for both tests' final sigma. Both
iterations are reported at build.

The shipped curve is measured on sources with no 4.5 micron excess (the
final iteration's flag); the curve on the 4.5-micron-excess sources is
reported beside it, and the same shipped construction is repeated per
region as a check (SPEC_PRIORS.md section 4, "Checks"). Both curves and
the region checks share one 40-bin log10 q grid, spanning the full
eligible population's own measured range. The shipped curve's floor --
the mean 8 micron excess fraction among eligible, no-4.5-excess sources
on the flat shelf `0.1 <= q <= 2.0`, measured per source, not
bin-quantized -- is subtracted and clipped at zero, since a probability
cannot be negative; the raw curve is kept beside it.

Reads SESNA photometry (the curated catalogues, their own detection
limits and adopted column) and the external TRILEGAL population -- never
a classification label (rule 7). Writes one survey-wide product,
`bms/pahc/curve_pahc_survey.hdf5`.

The bright end of q (log10 q from about -2.5 to -1.5, q 30-1000: sources
far brighter than their own 8 micron limit) holds no possible nebular-
light excess. Contamination adds nebular light of order the source's own
8 micron limit, `F_lim,8`, to the aperture -- an excess of order `q` in
flux-fraction units -- but the per-source 1 sigma near a bright star's
own photospheric prediction is set by the shelf's own residual scatter,
`s = 10**(0.4*RESIDUAL_WIDTH_MAG) - 1` (the flux-fraction equivalent of
the shelf's robust magnitude width), and a bright enough star's sigma
exceeds any excess nebular light alone could add. A 3 sigma excess from
nebular light therefore requires `q >= Q_MIN = EXCESS_SIGMA * s`; below
that, an observed 3 sigma 8 micron excess with no 4.5 micron excess is
circumstellar (a dusty evolved star, or a disc with an inner hole), not
contamination. Every `P_Q` bin whose upper edge in q lies below `Q_MIN`
is set to exactly zero (bins at or above `Q_MIN` are unchanged, floor and
all); each such bin's own measured, unfloored excess fraction is kept
instead in `P_Q_BRIGHT_EXCESS`, with its own `N_PER_BIN_BRIGHT_EXCESS` --
the fraction of bright field stars carrying circumstellar 8 micron
emission with no 4.5 micron excess.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic

from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.population import selection as selection_module

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
IDX_I1, IDX_I2, IDX_I4 = BAND_KEYS.index("I1"), BAND_KEYS.index("I2"), BAND_KEYS.index("I4")
ZERO_POINT_MJY = np.array([constants.VEGA_ZERO_POINT_MJY[k] for k in BAND_KEYS], dtype=np.float64)

#: An excess -- 8 micron over its photospheric prediction, or 4.5 micron
#: over its photospheric prediction -- is "more than 3 sigma"
#: (SPEC_PRIORS.md section 4, the `P(q)` row and "disks separated from
#: PAH" row).
EXCESS_SIGMA = 3.0

#: The colour relation's own bin width in [3.6]-[4.5], Vega mag
#: (SPEC_PRIORS.md section 4: "narrow bins of the latter"). Narrow enough
#: to resolve the relation's own curvature, wide enough that every bin
#: over the populated colour range holds many thousands of the pooled,
#: 30-region TRILEGAL retained population -- this module reports the
#: achieved per-bin counts.
COLOUR_BIN_WIDTH_MAG = 0.02

#: The colour relation is binned over the pooled population's own 0.5th
#: to 99.5th percentile range in [3.6]-[4.5]; a handful of extreme
#: outlier stars beyond that would otherwise force emptied bins at the
#: relation's own edges, where SPEC_PRIORS.md section 4's linear
#: interpolation holds the edge value regardless.
COLOUR_RANGE_PERCENTILE = (0.5, 99.5)

#: A colour bin below this count is dropped from the relation's own
#: interpolation knots (its median and width are not trusted); at
#: `COLOUR_BIN_WIDTH_MAG` over the populated range this is never binding
#: for the pooled, 30-region population.
MIN_STARS_PER_COLOUR_BIN = 50

#: Forty bins in log10 q (SPEC_PRIORS.md section 4, the `P(q)` row),
#: spanning the full eligible population's own measured range.
N_Q_BINS = 40

#: The floor's own flat shelf in q (SPEC_PRIORS.md section 4, the `P(q)`
#: row: "the floor being the flat shelf at 0.1 <= q <= 2").
FLOOR_Q_LO, FLOOR_Q_HI = 0.1, 2.0

#: `read()`'s own right-hand extrapolation rule, not the stored curve: a
#: bin below this count is too sparse to hold on its own beyond the
#: measured range -- the binomial error on a probability near 0.27 at
#: N = 25 is sqrt(0.27*0.73/25) = 0.09, already comparable to the
#: plateau's own bin-to-bin scatter, so a bin thinner than this can swing
#: the held value by tens of points of probability on a handful of stars.
MIN_PLATEAU_BIN_COUNT = 25

#: The q above which the curve sits on its own flat, high-q shelf (found
#: by reading the shipped curve, not a stored number): `read()` averages
#: the well-populated bins at and above this q, count-weighted, to get the
#: value it holds beyond the last well-measured bin.
PLATEAU_LOG10_Q_MIN = np.log10(15.0)

#: SESNA's own eligibility test for this measurement (SPEC_PRIORS.md
#: section 4, "the 1.64 M sources with 3.6, 4.5 and 8.0 micron
#: photometry"): a measured (`ORIGIN_FNU == 1`), finite, positive flux in
#: all three of I1, I2, I4.
ELIGIBLE_BAND_IDX = (IDX_I1, IDX_I2, IDX_I4)

#: The robust, Gaussian-equivalent scaling of the median absolute
#: deviation (`1 / Phi^-1(0.75)`, the standard estimator) -- turns the
#: shelf population's residual MAD into the relation term of sigma
#: (SPEC_PRIORS.md section 4, "the photospheric prediction").
MAD_TO_SIGMA = 1.4826

#: Regions are read and measured in chunks of `config.n_jobs` at a time
#: (CODING_RULES.md 10a: an 8 GB resident-memory budget for every worker
#: together; never all thirty regions' sources resident at once).


# ---------------------------------------------------------------------------
# 1. the photospheric colour relation, external, from pooled TRILEGAL stars
# ---------------------------------------------------------------------------

def _chunks(seq, size):
    """`seq` split into consecutive chunks of at most `size` items
    (CODING_RULES.md 10a: never hold every region's own arrays resident
    at once; process the thirty regions in chunks of the worker cap)."""
    seq = list(seq)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _field_star_path(config, region):
    path = config_module.product_path(
        config, "population", "trilegal", "field-stars", "region", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prior.pahc_curve: field-star product missing for region {region!r} "
            f"at {path!r} -- run the 'prior.field_stars' RUNBOOK line first")
    return path


def load_field_star_colours(config, region_names):
    """Reads the retained TRILEGAL population's own intrinsic fluxes
    (`population.field_stars`) for every region in `region_names`, pooled with
    equal weight per star, and returns `(c12, c48)`: `[3.6]-[4.5]` and
    `[4.5]-[8.0]`, Vega mag, on this project's own zero points
    (`constants.VEGA_ZERO_POINT_MJY`) -- the same zero points SESNA's own
    magnitudes below are read on.

    Each region's own 8-band flux array is read and reduced to its two
    colours inside the worker that reads it (`_one_region`), so only the
    two `(n_region,)` colour columns -- not the full flux array -- ever
    cross back to this process; those columns are written straight into
    two arrays preallocated to the survey total (CODING_RULES.md 10a),
    never pooled through a growing list and `np.concatenate` (no region's
    flux array, and no other region's colours, are held at once beyond one
    `n_jobs`-wide chunk).
    """
    def _one_region(region):
        with h5py.File(_field_star_path(config, region), "r") as f:
            fnu = f["FNU_MJY"][:]
        mag = -2.5 * np.log10(fnu / ZERO_POINT_MJY[None, :])
        return mag[:, IDX_I1] - mag[:, IDX_I2], mag[:, IDX_I2] - mag[:, IDX_I4]

    def _n_star(region):
        with h5py.File(_field_star_path(config, region), "r") as f:
            return f["FNU_MJY"].shape[0]

    counts = [_n_star(r) for r in region_names]
    c12_all = np.empty(sum(counts), dtype=np.float64)
    c48_all = np.empty(sum(counts), dtype=np.float64)

    cursor = 0
    for chunk in _chunks(region_names, config.n_jobs):
        for c12, c48 in Parallel(n_jobs=config.n_jobs)(delayed(_one_region)(r) for r in chunk):
            n = c12.size
            c12_all[cursor:cursor + n] = c12
            c48_all[cursor:cursor + n] = c48
            cursor += n

    finite = np.isfinite(c12_all) & np.isfinite(c48_all)
    return c12_all[finite], c48_all[finite]


def colour_relation(c12, c48, bin_width=COLOUR_BIN_WIDTH_MAG,
                     range_percentile=COLOUR_RANGE_PERCENTILE,
                     min_count=MIN_STARS_PER_COLOUR_BIN):
    """The binned relation: `(edges, medians, widths, counts)`, `edges`
    `(n_bin+1,)` fixed-width bins of `c12` over the population's own
    percentile range; per bin, `binned_statistic`'s own vectorised median
    of `c48` and the 16-84% half-width of `c48` (both empty-bin-safe,
    NaN where `counts < min_count`)."""
    lo, hi = np.percentile(c12, range_percentile)
    n_bin = max(1, int(np.ceil((hi - lo) / bin_width)))
    edges = np.linspace(lo, hi, n_bin + 1)

    counts, _, _ = binned_statistic(c12, c48, statistic="count", bins=edges)
    medians, _, _ = binned_statistic(c12, c48, statistic="median", bins=edges)
    p16, _, _ = binned_statistic(c12, c48, statistic=lambda x: np.percentile(x, 16.0), bins=edges)
    p84, _, _ = binned_statistic(c12, c48, statistic=lambda x: np.percentile(x, 84.0), bins=edges)
    widths = 0.5 * (p84 - p16)

    trusted = counts >= min_count
    medians = np.where(trusted, medians, np.nan)
    widths = np.where(trusted, widths, np.nan)
    return edges, medians, widths, counts.astype(np.int64)


def colour_relation_knots(edges, medians, widths):
    """`(centers, medians, widths)` restricted to the trusted (non-NaN)
    bins, sorted by `centers` -- the interpolation knots
    `predict_colour_48` reads."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(medians) & np.isfinite(widths)
    return centers[ok], medians[ok], widths[ok]


def predict_colour_48(c12, centers, medians, widths):
    """`[4.5]-[8.0]` predicted from `[3.6]-[4.5]` (`c12`) by linear
    interpolation between the relation's own trusted bin centres, edge
    values held beyond the relation's own measured range (SPEC_PRIORS.md
    section 4's own interpolation convention, applied here to the
    relation that feeds the prediction). Returns `(c48_pred, width)`."""
    c48_pred = np.interp(c12, centers, medians, left=medians[0], right=medians[-1])
    width = np.interp(c12, centers, widths, left=widths[0], right=widths[-1])
    return c48_pred, width


# ---------------------------------------------------------------------------
# 2. per-source excess tests and q, one region, fully vectorised
# ---------------------------------------------------------------------------

def _predicted_flux(f_in_mjy, colour_out_minus_in, zp_in, zp_out):
    """The flux a colour relation predicts in one band from an observed
    flux in another: `F_out = F_in * (F0_out/F0_in) * 10**(0.4*colour)`,
    the algebraic inverse of `colour = mag_in - mag_out` on this
    project's own Vega zero points."""
    return f_in_mjy * (zp_out / zp_in) * 10.0 ** (0.4 * colour_out_minus_in)


def _dereddened(f, a_col, kappa):
    """`F_i,0 = F_i * 10**(0.4*a*kappa_i(a))` (SPEC_PRIORS.md section 4,
    "the photospheric prediction"): the observed flux `f` undone of its
    own sightline dimming at column `a_col` and per-band ratio `kappa`."""
    return f * 10.0 ** (0.4 * a_col * kappa)


def _reddened(f0, a_col, kappa):
    """The inverse of `_dereddened`: a de-reddened-frame flux `f0`
    dimmed back to the observed frame at column `a_col`."""
    return f0 * 10.0 ** (-0.4 * a_col * kappa)


def region_measurement(config, region, colour_knots, colour45_median):
    """One region's eligible sources (SPEC_PRIORS.md section 4): `q` and
    the six arrays the survey-wide excess tests are built from --
    `(q, f2, f2_pred, f4, f4_pred, sigma2, sigma4)`, `(m,)` each, `m` the
    region's own eligible count -- vectorised over the region's sources,
    no Python loop. The excess flags themselves are not computed here:
    their sigma is a survey-wide constant (module docstring) not known
    until every region's residuals are pooled.
    """
    src_path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"prior.pahc_curve: curated catalogue missing for region {region!r} "
            f"at {src_path!r} -- run the 'catalog.curated' RUNBOOK line first")
    with h5py.File(src_path, "r") as f:
        fnu = f["FNU_MJY"][:]
        sigma = f["SIGMA_FNU_MJY"][:]
        origin = f["ORIGIN_FNU"][:]

    measured = np.all(origin[:, ELIGIBLE_BAND_IDX] == 1, axis=1)
    flux3 = fnu[:, ELIGIBLE_BAND_IDX]
    finite = np.all(np.isfinite(flux3) & (flux3 > 0), axis=1)
    eligible = measured & finite
    n_eligible = int(eligible.sum())
    empty = np.empty(0, dtype=np.float64)
    if n_eligible == 0:
        return empty, empty, empty, empty, empty, empty, empty

    f1 = fnu[eligible, IDX_I1]
    f2 = fnu[eligible, IDX_I2]
    f4 = fnu[eligible, IDX_I4]
    sigma2 = sigma[eligible, IDX_I2]
    sigma4 = sigma[eligible, IDX_I4]
    zp1, zp2, zp4 = ZERO_POINT_MJY[IDX_I1], ZERO_POINT_MJY[IDX_I2], ZERO_POINT_MJY[IDX_I4]

    # the source's own extinction column and its per-band dimming ratio
    # (SPEC_PRIORS.md section 1.3, "kappa_i(a)"), catalogue order --
    # extinction, not the gas column the young-star law was measured
    # on, since this is what the source's own starlight passes through
    #
    a_col_path = config_module.product_path(
        config, "sky/derived", "adopted", "extinction", "source", region=region)
    a_col_all = access.per_source(config, region, a_col_path, ["A_COL_K"])["A_COL_K"]
    a_col = np.asarray(a_col_all, dtype=np.float64)[eligible]
    kappa = selection_module.kappa_hybrid(config, selection_module.law_dense_weight(a_col))
    kappa1, kappa2, kappa4 = kappa[:, IDX_I1], kappa[:, IDX_I2], kappa[:, IDX_I4]

    # de-redden the observed 3.6 and 4.5 micron fluxes before the colour
    # relation reads them (module docstring, "De-reddening"): the
    # relation is built from TRILEGAL's own undimmed fluxes
    f1_0 = _dereddened(f1, a_col, kappa1)
    f2_0 = _dereddened(f2, a_col, kappa2)
    c12 = -2.5 * np.log10(f1_0 / zp1) + 2.5 * np.log10(f2_0 / zp2)

    # the photospheric prediction, in the de-reddened frame, reddened
    # back with kappa_8 to the observed frame the catalogue's own flux
    # and sigma live in (module docstring)
    centers, medians, widths = colour_knots
    c48_pred, _ = predict_colour_48(c12, centers, medians, widths)
    f4_pred_0 = _predicted_flux(f2_0, c48_pred, zp2, zp4)
    f4_pred = _reddened(f4_pred_0, a_col, kappa4)

    # the 4.5 micron excess's own prediction (SPEC_PRIORS.md section 4,
    # "disks separated from PAH"): the population's constant
    # photospheric [3.6]-[4.5], de-reddened frame, reddened back with
    # kappa_4.5
    f2_pred_0 = _predicted_flux(f1_0, colour45_median, zp1, zp2)
    f2_pred = _reddened(f2_pred_0, a_col, kappa2)

    f_lim8 = limits_module.limits(config, region)[eligible, IDX_I4]
    q = f_lim8 / f4_pred

    return q, f2, f2_pred, f4, f4_pred, sigma2, sigma4


# ---------------------------------------------------------------------------
# 2a. the survey-wide residual width and the excess tests it feeds
# ---------------------------------------------------------------------------

def mag_residual(f_obs, f_pred):
    """`mag(f_obs) - mag(f_pred) = -2.5*log10(f_obs/f_pred)`: the
    observed-minus-predicted residual, in magnitudes, that both the
    robust width and the excess test read (SPEC_PRIORS.md section 4)."""
    return -2.5 * np.log10(f_obs / f_pred)


def robust_width_mag(residual, shelf):
    """`RESIDUAL_WIDTH_MAG`/`RESIDUAL_WIDTH_45_MAG` (SPEC_PRIORS.md
    section 4): 1.4826 times the median absolute deviation of
    `residual[shelf]`, the robust, Gaussian-equivalent width of the
    observed-minus-predicted residual on the shelf population."""
    x = residual[shelf]
    med = np.median(x)
    return MAD_TO_SIGMA * float(np.median(np.abs(x - med)))


def flux_sigma_relation(f_pred, width_mag):
    """The relation term of sigma in flux units at `f_pred`'s own level:
    a magnitude width carried through `dF/dm = -F*ln(10)/2.5`
    (SPEC_PRIORS.md section 4)."""
    return f_pred * np.log(10.0) * 0.4 * width_mag


def excess_flags(f_obs, f_pred, sigma_meas, width_mag, valid):
    """One excess test (SPEC_PRIORS.md section 4, `EXCESS_SIGMA`): sigma
    is the catalogue's own measurement error combined in quadrature with
    the relation term at `width_mag`; `valid` masks sources with no
    finite prediction (kept False there, never counted an excess)."""
    sigma = np.sqrt(sigma_meas ** 2 + flux_sigma_relation(f_pred, width_mag) ** 2)
    excess = np.zeros(f_obs.shape, dtype=bool)
    excess[valid] = (f_obs[valid] - f_pred[valid]) / sigma[valid] > EXCESS_SIGMA
    return excess


# ---------------------------------------------------------------------------
# 3. the 40-bin curve: shipped, disk-excess, per-region check, floor
# ---------------------------------------------------------------------------

def binned_fraction(log10_q, excess, edges):
    """`(fraction, count)` per bin of `edges`: `binned_statistic`'s own
    vectorised mean of the excess flag and count, no Python loop over
    bins (rule 8). Empty bins report a zero fraction and a zero count."""
    counts, _, _ = binned_statistic(log10_q, excess.astype(np.float64), statistic="count", bins=edges)
    means, _, _ = binned_statistic(log10_q, excess.astype(np.float64), statistic="mean", bins=edges)
    counts = counts.astype(np.int64)
    fraction = np.where(counts > 0, means, 0.0)
    return fraction, counts


def q_min_bright_excess(width_mag, excess_sigma=EXCESS_SIGMA):
    """`Q_MIN` (module docstring, "the bright end of q"): the smallest q a
    3 sigma nebular-light excess can produce, given the shelf's own
    residual width. `s = 10**(0.4*width_mag) - 1` turns the shelf's
    robust magnitude width into a flux fraction -- the per-source 1 sigma
    at bright flux -- and `Q_MIN = excess_sigma * s` since nebular light
    of order `F_lim,8` (i.e. `q` in these units) must clear that sigma
    `excess_sigma` times over to register as a 3 sigma excess."""
    s = 10.0 ** (0.4 * width_mag) - 1.0
    return excess_sigma * s


def apply_bright_end_rule(curve, q_min):
    """Zeros every `P_Q` bin whose upper edge in q lies below `q_min`
    (module docstring): nebular light cannot produce a 3 sigma excess
    there, so the bin's measured excess fraction is not contamination.
    Those bins' own unfloored excess fraction and count move to
    `p_bright_excess`/`n_bright_excess` instead; bins at or above `q_min`
    are untouched (identity, rule 11)."""
    upper_q = 10.0 ** curve["edges"][1:]
    zeroed = upper_q < q_min
    p_a = curve["p_a"].copy()
    p_bright_excess = np.zeros_like(p_a)
    n_bright_excess = np.zeros_like(curve["n_a"])
    p_bright_excess[zeroed] = curve["p_a_raw"][zeroed]
    n_bright_excess[zeroed] = curve["n_a"][zeroed]
    p_a[zeroed] = 0.0
    curve = dict(curve)
    curve.update(p_a=p_a, p_bright_excess=p_bright_excess,
                 n_bright_excess=n_bright_excess, zeroed=zeroed, q_min=q_min)
    return curve


def build_curve(q, excess8, excess45, region_idx, n_region, n_bins=N_Q_BINS):
    """Assembles the 40-bin log10 q curve family: the shipped curve (no
    4.5 micron excess, floor-subtracted and raw), the disk-excess curve
    (with a 4.5 micron excess), the per-region check curves, and the
    floor. Returns a dict ready for `write_curve`.
    """
    finite_q = np.isfinite(q) & (q > 0)
    q, excess8, excess45, region_idx = (
        q[finite_q], excess8[finite_q], excess45[finite_q], region_idx[finite_q])
    log10_q = np.log10(q)
    edges = np.linspace(log10_q.min(), log10_q.max(), n_bins + 1)

    mask_a = ~excess45
    mask_b = excess45
    p_a_raw, n_a = binned_fraction(log10_q[mask_a], excess8[mask_a], edges)
    p_b_raw, n_b = binned_fraction(log10_q[mask_b], excess8[mask_b], edges)

    # the floor: the source-weighted (not bin-quantized) mean 8 micron
    # excess fraction on the shipped subsample's own flat shelf
    # (SPEC_PRIORS.md section 4, "the floor being the flat shelf at
    # 0.1 <= q <= 2")
    shelf = mask_a & (q >= FLOOR_Q_LO) & (q <= FLOOR_Q_HI)
    floor = float(np.mean(excess8[shelf])) if shelf.any() else 0.0
    p_a = np.clip(p_a_raw - floor, 0.0, None)

    p_region = np.zeros((n_region, n_bins), dtype=np.float64)
    n_region_bin = np.zeros((n_region, n_bins), dtype=np.int64)
    for r in range(n_region):
        sel = mask_a & (region_idx == r)
        if sel.any():
            p_region[r], n_region_bin[r] = binned_fraction(log10_q[sel], excess8[sel], edges)

    return dict(
        edges=edges, p_a=p_a, p_a_raw=p_a_raw, n_a=n_a, floor=floor,
        p_b_raw=p_b_raw, n_b=n_b, p_region=p_region, n_region_bin=n_region_bin,
        n_shipped=int(mask_a.sum()), n_disk_excess=int(mask_b.sum()),
        n_eligible=int(finite_q.sum()),
    )


# ---------------------------------------------------------------------------
# 4. write, read
# ---------------------------------------------------------------------------

def write_curve(path, curve, q_min, residual_width_mag):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.attrs["Q_MIN"] = float(q_min)
        f.attrs["RESIDUAL_WIDTH_MAG"] = float(residual_width_mag)
        f.create_dataset("LOG10_Q_EDGES", data=curve["edges"].astype(np.float64))
        f.create_dataset("P_Q", data=curve["p_a"].astype(np.float64))
        f.create_dataset("N_PER_BIN", data=curve["n_a"].astype(np.int64))
        f.create_dataset("P_Q_BRIGHT_EXCESS", data=curve["p_bright_excess"].astype(np.float64))
        f.create_dataset("N_PER_BIN_BRIGHT_EXCESS", data=curve["n_bright_excess"].astype(np.int64))


def read(config):
    """The shipped curve `P(q)` as a callable, linear in log10 q. Left of
    the measured range, the first bin's own value is held. Right of it,
    the value held is not necessarily the last bin's own value: a bin
    with fewer than `MIN_PLATEAU_BIN_COUNT` stars is too sparse to trust
    on its own (a couple of stars can put it far from the shelf), so
    every bin beyond the last bin that clears that count is folded into
    one plateau value -- the count-weighted mean `P` over the bins at
    `q` above `PLATEAU_LOG10_Q_MIN`, the curve's own flat high-q shelf --
    and that plateau is what both those bins and the right-hand fill
    hold. The stored bins and counts on disk are unchanged; only how
    this reader extrapolates beyond them."""
    path = config_module.product_path(config, "population", "pahc", "curve", "survey")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prior.pahc_curve: no PAHC curve product at {path!r} -- run the "
            f"'prior.pahc_curve' RUNBOOK line first")
    with h5py.File(path, "r") as f:
        edges = f["LOG10_Q_EDGES"][:]
        p_q = f["P_Q"][:]
        n_per_bin = f["N_PER_BIN"][:]
    centers = 0.5 * (edges[:-1] + edges[1:])

    on_plateau = (centers >= PLATEAU_LOG10_Q_MIN) & (n_per_bin > 0)
    plateau = (float(np.average(p_q[on_plateau], weights=n_per_bin[on_plateau]))
               if on_plateau.any() else float(p_q[-1]))

    well_measured = np.flatnonzero(n_per_bin >= MIN_PLATEAU_BIN_COUNT)
    p_q_read = p_q.copy()
    if well_measured.size:
        p_q_read[well_measured[-1] + 1:] = plateau
    else:
        p_q_read[:] = plateau

    return interp1d(centers, p_q_read, kind="linear", bounds_error=False,
                     fill_value=(float(p_q_read[0]), plateau))


# ---------------------------------------------------------------------------
# 5. build
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Measures `P(q)` over the pool of sources in `regions` (default:
    all thirty), writes `bms/pahc/curve_pahc_survey.hdf5`. The
    photospheric colour relation pools the TRILEGAL retained population
    of the same region set. The regions are read in chunks of at most
    `config.n_jobs` at a time (CODING_RULES.md 10a)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    st = progress.Stage("prior.pahc_curve")

    c12_star, c48_star = load_field_star_colours(config, region_names)
    colour_edges, colour_medians, colour_widths, colour_counts = colour_relation(c12_star, c48_star)
    knots = colour_relation_knots(colour_edges, colour_medians, colour_widths)
    colour45_median = float(np.median(c12_star))
    print(f"pahc_curve: {c12_star.size} pooled TRILEGAL retained stars, "
          f"{knots[0].size}/{colour_medians.size} trusted colour bins "
          f"(width {COLOUR_BIN_WIDTH_MAG} mag), "
          f"[3.6]-[4.5] median={colour45_median:.4f} mag", flush=True)

    chunks = list(_chunks(region_names, config.n_jobs))
    results = []
    for i_chunk, chunk in enumerate(chunks, start=1):
        results.extend(Parallel(n_jobs=config.n_jobs)(
            delayed(region_measurement)(config, region, knots, colour45_median)
            for region in chunk))
        st.tick(i_chunk, len(chunks), "region chunks")
    q = np.concatenate([r[0] for r in results]) if results else np.empty(0)
    f2 = np.concatenate([r[1] for r in results]) if results else np.empty(0)
    f2_pred = np.concatenate([r[2] for r in results]) if results else np.empty(0)
    f4 = np.concatenate([r[3] for r in results]) if results else np.empty(0)
    f4_pred = np.concatenate([r[4] for r in results]) if results else np.empty(0)
    sigma2 = np.concatenate([r[5] for r in results]) if results else np.empty(0)
    sigma4 = np.concatenate([r[6] for r in results]) if results else np.empty(0)
    region_idx = np.concatenate([
        np.full(r[0].size, i, dtype=np.int64) for i, r in enumerate(results)]) if results else np.empty(0, dtype=np.int64)
    n_eligible_total = q.size
    print(f"pahc_curve: {n_eligible_total} eligible sources "
          f"(measured I1, I2, I4 in {len(region_names)} regions)", flush=True)

    # the survey-wide relation width and the excess flags it feeds
    # (module docstring, "Sigma on each excess test"): the shelf that
    # defines the width is itself the 4.5 micron test's own no-excess
    # subsample, so the width is measured twice -- once with no 4.5
    # micron cut, to get a first sigma for that test; once more on the
    # cut shelf, shipped
    valid = np.isfinite(q) & (q > 0) & np.isfinite(f2_pred) & (f2_pred > 0) & np.isfinite(f4_pred) & (f4_pred > 0)
    resid48 = np.where(valid, mag_residual(f4, f4_pred), np.nan)
    resid45 = np.where(valid, mag_residual(f2, f2_pred), np.nan)
    shelf_q = valid & (q >= FLOOR_Q_LO) & (q <= FLOOR_Q_HI)

    width48_pass0 = robust_width_mag(resid48, shelf_q)
    width45_pass0 = robust_width_mag(resid45, shelf_q)
    excess45_pass0 = excess_flags(f2, f2_pred, sigma2, width45_pass0, valid)

    shelf_final = shelf_q & ~excess45_pass0
    width48 = robust_width_mag(resid48, shelf_final)
    width45 = robust_width_mag(resid45, shelf_final)
    excess45 = excess_flags(f2, f2_pred, sigma2, width45, valid)
    excess8 = excess_flags(f4, f4_pred, sigma4, width48, valid)

    print(f"pahc_curve: residual width pass 0 (no 4.5um cut, n={int(shelf_q.sum())}) "
          f"[4.5]-[8.0]={width48_pass0:.4f} [3.6]-[4.5]={width45_pass0:.4f} mag; "
          f"pass 1 (4.5um cut applied, shipped, n={int(shelf_final.sum())}) "
          f"[4.5]-[8.0]={width48:.4f} [3.6]-[4.5]={width45:.4f} mag", flush=True)

    curve = build_curve(q, excess8, excess45, region_idx, len(region_names))

    # the bright end of q: below Q_MIN a 3 sigma excess cannot be
    # nebular light, so P_Q is zeroed there and the measured fraction
    # moves to P_Q_BRIGHT_EXCESS (module docstring)
    q_min = q_min_bright_excess(width48)
    curve = apply_bright_end_rule(curve, q_min)
    n_zeroed = int(curve["zeroed"].sum())
    frac_eligible_zeroed = float(curve["n_a"][curve["zeroed"]].sum()) / curve["n_eligible"]

    out_path = config_module.product_path(config, "population", "pahc", "curve", "survey")
    write_curve(out_path, curve, q_min, width48)

    st.done(out_path, n_shipped=curve["n_shipped"], n_disk_excess=curve["n_disk_excess"],
            floor=curve["floor"], q_min=q_min, n_zeroed=n_zeroed)
    print(f"pahc_curve: shipped n={curve['n_shipped']} disk-excess n={curve['n_disk_excess']} "
          f"floor={curve['floor']:.6f} Q_MIN={q_min:.4f} bins_zeroed={n_zeroed}/{N_Q_BINS} "
          f"({frac_eligible_zeroed:.4%} of the eligible, no-4.5um-excess population) "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    run(build)
