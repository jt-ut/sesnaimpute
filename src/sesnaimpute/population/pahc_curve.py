"""The PAH-contamination probability curve `P(q)` (`SPEC_PRIORS.md` section
4; `IMPLEMENTATION.md` section 6, stage 2).

A source's aperture is contaminated by extended PAH nebular emission with
probability `P(q)`, `q = F_lim,8(s) / F_8,pred(s)` -- the source's own 8
micron completeness limit over its predicted photospheric 8 micron flux,
so `q` is large where the local field is faint relative to the star (a
nebulous, shallow position) and small where it is bright. `P(q)` is
measured once, survey-wide, directly from SESNA photometry, over every
source with measured (`ORIGIN_FNU == 1`) 3.6 and 4.5 micron photometry --
the denominator the curve is later applied against (the atlas partitions
every such star into STAR or PAHC). A source counts toward the numerator
only if it additionally has a measured 8 micron flux that exceeds its
predicted photospheric 8 micron flux by more than 3 sigma; a source with
no 8 micron measurement (98-99% of the faint-end population, where the
8 micron limit is close to the survey's 4.5 micron one) still contributes
its own `q` to the denominator and can never register an excess, since an
unmeasured flux cannot exceed anything. Restricting the denominator to
the rarer 8-micron-detected sources, as an earlier build did, measures
`P(q)` on a population increasingly unlike the one it is applied to as
`q` grows, and overstates the contaminated fraction there by roughly the
inverse of the 8 micron detection rate.

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

The curve `P(q)` is measured and shipped PER REGION (ledger C7,
`REVIEW_LEDGER_2026-10-08.md`; `verify_B.md` V7; `prior_5_PAHC.md`
section 4): the shelf excess rate (`0.1 <= q <= 2`, no 4.5 micron
excess) is regional by a factor of 8 -- Orion A 0.047, Pipe 0.0056,
Aquila 0.0054, against the one-survey-wide curve's own shipped shelf
values of 0.0033-0.0068 -- because nebular 8 micron brightness is a
property of the region (Povich et al. 2007, ApJ 660, 346), not a
survey-wide constant. One curve applied to every region therefore
starved PAHC's prior by about 1.4 nats in Orion A's shelf while handing
Pipe and Musca a nonzero probability where their own measurement
supports zero. Every region in the build's own region list gets its own
row, `p(excess | measured, bin)` by bin, with NO scaling by that bin's
own 8-micron-measured share `m(bin) = M_PER_BIN/N_PER_BIN`: the earlier
construction's `P_Q(bin) = m(bin) * p(excess | measured, bin)` is
equivalent to asserting that a source with no 8 micron measurement is
never contaminated, which is a depth statement, not a contamination
one, and understated the level by roughly 1.6 dex at high `q`, where
most of the eligible population sits (`prior_5_PAHC.md` section 1). The
shipped per-region value is the region's own measured-only excess
fraction, unscaled, with the floor below subtracted and clipped at
zero: `P_Q_REGION(region, bin) = clip(p(excess | measured, region, bin)
- F_FLOOR, 0)`; the same quantity before the floor subtraction is kept
beside it, `P_Q_REGION_RAW`.

The floor -- a false 3 sigma excess from a source that carries no real
nebular contamination -- is a property of the 8 micron band itself, not
of any one region's own nebular brightness, so it is measured once,
survey-wide in CONSTRUCTION but not in POPULATION: pooled only over the
three regions whose own measured shelf rate is lowest, Pipe, Musca and
Aquila (`QUIESCENT_FLOOR_REGIONS`), rather than over all thirty. The
earlier construction pooled the floor over every region, `F_NOISE =
0.0161`, about 2.5 times the quiescent regions' own rate (about 0.006),
because that pool is dominated by nebulous regions and so contains real
contamination, subtracting signal from every bin it is meant to
protect. `F_FLOOR` is the quiescent regions' own pooled shelf excess
count over their own pooled shelf 8-micron-measured count, one number,
subtracted from every region's curve alike.

The curve on sources WITH a 4.5 micron excess (disk-bearing or
otherwise circumstellar) is measured the same unscaled way,
`p(excess | measured, bin)` with no `m(bin)` factor, pooled survey-wide
and left unfloored, and reported beside the shipped curve
(SPEC_BMSTP_DRAFT.md section 5.3, "The curve on disc-excess sources is
reported beside it").

Reads SESNA photometry (the curated catalogues, their own detection
limits and adopted column) and the external TRILEGAL population -- never
a classification label (rule 7). Writes one product with a region axis,
`population/pahc/curve_pahc_survey.hdf5`; the earlier design's
`bms/pahc/curve_pahc_survey.hdf5` is a stale, superseded toy (225
sources, 1 region, 6 Sep) that this module does not read, write, or
delete (CODING_RULES_BMSTP.md: no unit edits the earlier design's
products under `bms/`; its removal is the rebuild's own housekeeping).

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
contamination. Every `P_Q_REGION` bin whose upper edge in q lies below
`Q_MIN` is set to exactly zero, in every region's row alike since
`Q_MIN` is not region-specific (bins at or above `Q_MIN` are unchanged,
floor and all); each such bin's own measured, unfloored, per-region
excess fraction is kept instead in `P_Q_REGION_BRIGHT_EXCESS`, with its
own `N_PER_BIN_REGION_BRIGHT_EXCESS` -- that region's own fraction of
bright field stars carrying circumstellar 8 micron emission with no 4.5
micron excess.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.population import selection as selection_module

_STEM = "curve_pahc_survey"

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

#: The regions whose pooled shelf excess rate anchors the noise floor
#: `F_FLOOR` (ledger C7, `REVIEW_LEDGER_2026-10-08.md`; WP-POP-4): the
#: three regions `verify_B.md` V7 and `prior_5_PAHC.md` section 4 measure
#: the lowest shelf rate on, about 0.006 pooled, against the
#: all-region pool's 0.0161 -- a region earns a place here by its own
#: measured rate, not by an independent quiescence criterion.
QUIESCENT_FLOOR_REGIONS = ("Pipe", "Musca", "Aquila")

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

#: The curve's own denominator, the population it is later applied to
#: (module docstring): a measured (`ORIGIN_FNU == 1`), finite, positive
#: flux in I1 and I2. The 4.5 micron excess test, the shelf, both
#: residual widths and `Q_MIN` are all defined on this same population.
ELIGIBLE_BAND_IDX = (IDX_I1, IDX_I2)

#: The numerator's own extra requirement (module docstring): a measured,
#: finite, positive flux in I4 as well, tested in `region_measurement`.
EXCESS_BAND_IDX = IDX_I4

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
    """One region's eligible sources (module docstring: every I1+I2
    measured source, the curve's own denominator): `q` and the seven
    arrays the survey-wide excess tests are built from -- `(q, f2,
    f2_pred, f4, f4_pred, sigma2, sigma4, have_8um)`, `(m,)` each, `m` the
    region's own eligible count -- vectorised over the region's sources,
    no Python loop. `have_8um` flags the subset with a measured I4 flux as
    well (`EXCESS_BAND_IDX`); `f4`/`sigma4` hold the catalogue's own
    substitute (a completeness-limit flux, `catalog.curated`'s own
    docstring) where it is False, read by no excess test. The excess
    flags themselves are not computed here: their sigma is a survey-wide
    constant (module docstring) not known until every region's residuals
    are pooled.
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
    flux2 = fnu[:, ELIGIBLE_BAND_IDX]
    finite = np.all(np.isfinite(flux2) & (flux2 > 0), axis=1)
    eligible = measured & finite
    n_eligible = int(eligible.sum())
    empty = np.empty(0, dtype=np.float64)
    empty_bool = np.empty(0, dtype=bool)
    if n_eligible == 0:
        return empty, empty, empty, empty, empty, empty, empty, empty_bool

    have_8um_all = ((origin[:, EXCESS_BAND_IDX] == 1)
                    & np.isfinite(fnu[:, EXCESS_BAND_IDX]) & (fnu[:, EXCESS_BAND_IDX] > 0))
    have_8um = have_8um_all[eligible]

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

    # q reads the predicted photospheric 8 micron flux, never the
    # observed one (module docstring; the atlas factor tables read it the
    # same way), so every eligible source has a q whether or not it has
    # an 8 micron measurement
    f_lim8 = limits_module.limits(config, region)[eligible, IDX_I4]
    q = f_lim8 / f4_pred

    return q, f2, f2_pred, f4, f4_pred, sigma2, sigma4, have_8um


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

def binned_measured_rate(log10_q, have_8um, excess8, edges):
    """`(n, m, p_measured)` per bin of `edges`, all `(n_bin,)`: `n` the
    denominator count (every source, module docstring), `m` the count
    with a measured 8 micron flux, and `p_measured` the excess fraction
    among that measured subset only (0 where `m == 0`, never read
    there). `binned_statistic`'s own vectorised sums and mean, no Python
    loop over bins (rule 8)."""
    n, _, _ = binned_statistic(log10_q, np.ones(log10_q.shape), statistic="sum", bins=edges)
    m, _, _ = binned_statistic(log10_q[have_8um], np.ones(have_8um.sum()), statistic="sum", bins=edges)
    p_measured = np.zeros(m.shape, dtype=np.float64)
    if have_8um.any():
        counts, _, _ = binned_statistic(log10_q[have_8um], excess8[have_8um].astype(np.float64),
                                         statistic="count", bins=edges)
        means, _, _ = binned_statistic(log10_q[have_8um], excess8[have_8um].astype(np.float64),
                                        statistic="mean", bins=edges)
        p_measured = np.where(counts > 0, means, 0.0)
    return n.astype(np.int64), m.astype(np.int64), p_measured


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
    """Zeros, in every region's row of `p_region`, the bins whose upper
    edge in q lies below `q_min` (module docstring): nebular light
    cannot produce a 3 sigma excess there, so a bin's measured excess
    fraction there is circumstellar, not contamination -- the same
    physical argument in every region, since `q_min` comes from the
    shelf's own survey-wide residual width, not from region. Those
    bins' own unfloored, per-region excess fraction and count move to
    `p_region_bright_excess`/`n_region_bright_excess` instead; bins at
    or above `q_min` are untouched (identity, rule 11)."""
    upper_q = 10.0 ** curve["edges"][1:]
    zeroed = upper_q < q_min
    p_region = curve["p_region"].copy()
    p_region_bright_excess = np.zeros_like(p_region)
    n_region_bright_excess = np.zeros_like(curve["n_region_bin"])
    p_region_bright_excess[:, zeroed] = curve["p_region_raw"][:, zeroed]
    n_region_bright_excess[:, zeroed] = curve["n_region_bin"][:, zeroed]
    p_region[:, zeroed] = 0.0
    curve = dict(curve)
    curve.update(p_region=p_region, p_region_bright_excess=p_region_bright_excess,
                 n_region_bright_excess=n_region_bright_excess, zeroed=zeroed, q_min=q_min)
    return curve


def build_curve(q, excess8, excess45, have_8um, region_idx, region_names, n_bins=N_Q_BINS):
    """Assembles the 40-bin log10 q curve family, per region (WP-POP-4,
    ledger C7): the shipped per-region curve (no 4.5 micron excess,
    floored and raw), the disk-excess curve (with a 4.5 micron excess,
    survey-wide, unfloored), and the floor. Returns a dict ready for
    `write_curve`.

    `F_FLOOR` is a property of the 8 micron band's own false-excess
    rate, not of any one region's nebular brightness: it is the pooled
    shelf excess count over the pooled shelf 8-micron-measured count,
    `0.1 <= q <= 2`, no 4.5 micron excess, restricted to
    `QUIESCENT_FLOOR_REGIONS` -- never the all-region pool, which is
    dominated by nebulous regions and so is itself contaminated (module
    docstring). Each region's own `P_Q_REGION` is that region's own
    measured-only excess fraction net of `F_FLOOR`, clipped at zero, with
    NO scaling by that bin's 8-micron-measured share: a source with no 8
    micron measurement is not assumed uncontaminated.
    """
    finite_q = np.isfinite(q) & (q > 0)
    q, excess8, excess45, have_8um, region_idx = (
        q[finite_q], excess8[finite_q], excess45[finite_q], have_8um[finite_q], region_idx[finite_q])
    log10_q = np.log10(q)
    edges = np.linspace(log10_q.min(), log10_q.max(), n_bins + 1)

    mask_a = ~excess45  # the shipped population: no 4.5 micron excess
    mask_b = excess45   # the disk-excess population, reported beside it
    shelf = mask_a & (q >= FLOOR_Q_LO) & (q <= FLOOR_Q_HI)

    floor_region_idx = [i for i, r in enumerate(region_names) if r in QUIESCENT_FLOOR_REGIONS]
    if not floor_region_idx:
        raise ValueError(
            "pahc_curve.build_curve: none of the floor regions %r are in this build's "
            "own region list %r -- the noise floor cannot be measured without at least "
            "one of them" % (QUIESCENT_FLOOR_REGIONS, tuple(region_names)))
    in_floor_region = np.isin(region_idx, floor_region_idx)
    floor_shelf_measured = shelf & have_8um & in_floor_region
    f_floor = float(np.mean(excess8[floor_shelf_measured])) if floor_shelf_measured.any() else 0.0

    n_region = len(region_names)
    p_region_raw = np.zeros((n_region, n_bins), dtype=np.float64)
    p_region = np.zeros((n_region, n_bins), dtype=np.float64)
    n_region_bin = np.zeros((n_region, n_bins), dtype=np.int64)
    m_region_bin = np.zeros((n_region, n_bins), dtype=np.int64)
    for r in range(n_region):
        sel = mask_a & (region_idx == r)
        n_r, m_r, p_measured_r = binned_measured_rate(log10_q[sel], have_8um[sel], excess8[sel], edges)
        p_region_raw[r] = p_measured_r
        p_region[r] = np.clip(p_measured_r - f_floor, 0.0, None)
        n_region_bin[r] = n_r
        m_region_bin[r] = m_r

    n_disk_excess_bin, m_disk_excess_bin, p_disk_excess = binned_measured_rate(
        log10_q[mask_b], have_8um[mask_b], excess8[mask_b], edges)

    return dict(
        edges=edges, region_names=tuple(region_names),
        f_floor=f_floor, floor_regions=tuple(region_names[i] for i in floor_region_idx),
        p_region=p_region, p_region_raw=p_region_raw,
        n_region_bin=n_region_bin, m_region_bin=m_region_bin,
        p_disk_excess=p_disk_excess, n_disk_excess_bin=n_disk_excess_bin,
        m_disk_excess_bin=m_disk_excess_bin,
        n_shipped=int(mask_a.sum()), n_disk_excess=int(mask_b.sum()),
        n_eligible=int(finite_q.sum()),
    )


# ---------------------------------------------------------------------------
# 4. write, read
# ---------------------------------------------------------------------------

def write_curve(path, curve, q_min, residual_width_mag, residual_width_45_mag):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.attrs["Q_MIN"] = float(q_min)
        f.attrs["RESIDUAL_WIDTH_MAG"] = float(residual_width_mag)
        f.attrs["RESIDUAL_WIDTH_45_MAG"] = float(residual_width_45_mag)
        f.attrs["F_FLOOR"] = float(curve["f_floor"])
        f.attrs["FLOOR_REGIONS"] = ", ".join(curve["floor_regions"])
        for name, data, extra in (
            ("LOG10_Q_EDGES", curve["edges"].astype(np.float64), {}),
            ("REGION", np.array(curve["region_names"], dtype=object),
             dict(dtype=h5py.string_dtype(encoding="utf-8"))),
            ("P_Q_REGION", curve["p_region"].astype(np.float64), {}),
            ("P_Q_REGION_RAW", curve["p_region_raw"].astype(np.float64), {}),
            ("N_PER_BIN_REGION", curve["n_region_bin"].astype(np.int64), {}),
            ("M_PER_BIN_REGION", curve["m_region_bin"].astype(np.int64), {}),
            ("P_Q_REGION_BRIGHT_EXCESS", curve["p_region_bright_excess"].astype(np.float64), {}),
            ("N_PER_BIN_REGION_BRIGHT_EXCESS", curve["n_region_bright_excess"].astype(np.int64), {}),
            ("P_Q_DISK_EXCESS", curve["p_disk_excess"].astype(np.float64), {}),
            ("N_PER_BIN_DISK_EXCESS", curve["n_disk_excess_bin"].astype(np.int64), {}),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)], **extra)


def _region_row(config, region, caller):
    """`(centers, p_q, n_per_bin)`, this region's own raw row of the
    shipped curve (`P_Q_REGION`/`N_PER_BIN_REGION`), straight off disk,
    no plateau correction. The one place that opens the curve product
    (`population/pahc/curve_pahc_survey.hdf5`) at all -- `read` and
    `read_raw_bins` both call this rather than each reading the file,
    so there is exactly one piece of code that knows the product's own
    dataset names and `REGION` axis. `caller` names the public function
    in a region-not-found error, for a useful traceback."""
    path = config_module.product_path(config, "population", "pahc", "curve", "survey")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prior.pahc_curve: no PAHC curve product at {path!r} -- run the "
            f"'prior.pahc_curve' RUNBOOK line first")
    with h5py.File(path, "r") as f:
        edges = f["LOG10_Q_EDGES"][:]
        region_names = [r.decode("utf-8") if isinstance(r, bytes) else r for r in f["REGION"][:]]
        if region not in region_names:
            raise ValueError(
                f"prior.pahc_curve.{caller}: region {region!r} is not one of the "
                f"{len(region_names)} regions this curve was built on {region_names!r} "
                f"-- rebuild 'prior.pahc_curve' with that region included")
        r = region_names.index(region)
        p_q = f["P_Q_REGION"][r]
        n_per_bin = f["N_PER_BIN_REGION"][r]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, p_q, n_per_bin


def read_raw_bins(config, region):
    """`(centers, p_q, n_per_bin)`, this region's own curve exactly as
    stored: the bin centers in log10 q, `P_Q_REGION`'s own row (no
    plateau correction -- that is `read`'s own read-time convenience for
    a continuous callable, not part of the stored curve), and
    `N_PER_BIN_REGION`'s own row, the count each bin's value rests on.
    For a consumer that needs the curve's own tabulated bins directly
    (a diagnostic report against them, or a density built from them),
    rather than a continuous lookup at an arbitrary q -- so that
    consumer never opens the curve product itself (`_region_row` is the
    one place that does)."""
    return _region_row(config, region, "read_raw_bins")


def read(config, region):
    """The shipped curve `P(q)` for one region, as a callable, linear in
    log10 q (WP-POP-4, ledger C7: the excess rate is regional by a
    factor of 8, so every consumer reads its own region's row, never a
    pooled one -- there is no region-less fallback). Left of the
    measured range, the first bin's own value is held. Right of it, the
    value held is not necessarily the last bin's own value: a bin with
    fewer than `MIN_PLATEAU_BIN_COUNT` sources, in THIS region, is too
    sparse to trust on its own, so every bin beyond the last one that
    clears that count, in this region's own row, is folded into one
    plateau value -- the count-weighted mean `P` over this region's bins
    at `q` above `PLATEAU_LOG10_Q_MIN` -- and that plateau is what both
    those bins and the right-hand fill hold. The stored bins and counts
    on disk are unchanged; only how this reader extrapolates beyond
    them."""
    centers, p_q, n_per_bin = _region_row(config, region, "read")

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
    """Measures `P(q)` per region over `regions` (default: all thirty),
    writes `population/pahc/curve_pahc_survey.hdf5` with one row per
    region (WP-POP-4, ledger C7). The photospheric colour relation and
    both residual widths still pool the TRILEGAL retained population and
    the shelf of the same region set (SPEC_BMSTP_DRAFT.md section 5.3:
    "measured once, survey-wide"), since neither is a contamination
    rate; only the contamination rate itself, and the floor that nets
    it, are now region-specific and quiescent-region-specific
    respectively. The floor needs at least one of
    `QUIESCENT_FLOOR_REGIONS` in `regions` (`build_curve` raises
    otherwise). The regions are read in chunks of at most
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
    have_8um = np.concatenate([r[7] for r in results]) if results else np.empty(0, dtype=bool)
    region_idx = np.concatenate([
        np.full(r[0].size, i, dtype=np.int64) for i, r in enumerate(results)]) if results else np.empty(0, dtype=np.int64)
    n_eligible_total = q.size
    n_have_8um = int(have_8um.sum())
    print(f"pahc_curve: {n_eligible_total} eligible sources (measured I1, I2 in "
          f"{len(region_names)} regions), {n_have_8um} ({n_have_8um / max(n_eligible_total, 1):.4%}) "
          f"also measured I4", flush=True)

    # the survey-wide relation width and the excess flags it feeds
    # (module docstring, "Sigma on each excess test"): the shelf that
    # defines the width is itself the 4.5 micron test's own no-excess
    # subsample, so the width is measured twice -- once with no 4.5
    # micron cut, to get a first sigma for that test; once more on the
    # cut shelf, shipped. The 4.5 micron test and its own shelf/width need
    # no 8 micron measurement and so run over the full eligible
    # population (`valid`); the 8 micron width and `excess8` can only be
    # measured where an 8 micron flux exists (`valid & have_8um`) -- the
    # same population, restricted by what a residual needs to exist at
    # all, not by a further choice.
    valid = np.isfinite(q) & (q > 0) & np.isfinite(f2_pred) & (f2_pred > 0) & np.isfinite(f4_pred) & (f4_pred > 0)
    valid8 = valid & have_8um
    resid48 = np.where(valid8, mag_residual(f4, f4_pred), np.nan)
    resid45 = np.where(valid, mag_residual(f2, f2_pred), np.nan)
    shelf_q = valid & (q >= FLOOR_Q_LO) & (q <= FLOOR_Q_HI)
    shelf_q_8 = shelf_q & have_8um

    width48_pass0 = robust_width_mag(resid48, shelf_q_8)
    width45_pass0 = robust_width_mag(resid45, shelf_q)
    excess45_pass0 = excess_flags(f2, f2_pred, sigma2, width45_pass0, valid)

    shelf_final = shelf_q & ~excess45_pass0
    width48 = robust_width_mag(resid48, shelf_final & have_8um)
    width45 = robust_width_mag(resid45, shelf_final)
    excess45 = excess_flags(f2, f2_pred, sigma2, width45, valid)
    excess8 = excess_flags(f4, f4_pred, sigma4, width48, valid8)

    print(f"pahc_curve: residual width pass 0 (no 4.5um cut, n={int(shelf_q.sum())}, "
          f"n_8um={int(shelf_q_8.sum())}) "
          f"[4.5]-[8.0]={width48_pass0:.4f} [3.6]-[4.5]={width45_pass0:.4f} mag; "
          f"pass 1 (4.5um cut applied, shipped, n={int(shelf_final.sum())}, "
          f"n_8um={int((shelf_final & have_8um).sum())}) "
          f"[4.5]-[8.0]={width48:.4f} [3.6]-[4.5]={width45:.4f} mag", flush=True)

    curve = build_curve(q, excess8, excess45, have_8um, region_idx, region_names)

    # the bright end of q: below Q_MIN a 3 sigma excess cannot be
    # nebular light, so P_Q_REGION is zeroed there (every region alike)
    # and the measured fraction moves to P_Q_REGION_BRIGHT_EXCESS
    # (module docstring)
    q_min = q_min_bright_excess(width48)
    curve = apply_bright_end_rule(curve, q_min)
    n_zeroed = int(curve["zeroed"].sum())
    frac_eligible_zeroed = float(curve["n_region_bin"][:, curve["zeroed"]].sum()) / curve["n_eligible"]

    out_path = config_module.product_path(config, "population", "pahc", "curve", "survey")
    write_curve(out_path, curve, q_min, width48, width45)

    # the sweep's own identity (WP-POP-4, ledger C7): each region's own
    # raw shelf rate (no floor subtracted, no m(bin) scaling), measured
    # directly from the per-source arrays the same way F_FLOOR is
    # measured, must reproduce the review's independent measurement
    # (`verify_B.md` V7) -- Orion A 0.047, Aquila 0.005
    shelf_q_mask = ~excess45 & (q >= FLOOR_Q_LO) & (q <= FLOOR_Q_HI)
    for check_region, target in (("Orion A", 0.047), ("Aquila", 0.005)):
        if check_region in region_names:
            r = region_names.index(check_region)
            region_shelf_measured = shelf_q_mask & have_8um & (region_idx == r)
            rate = float(np.mean(excess8[region_shelf_measured])) if region_shelf_measured.any() else float("nan")
            print(f"pahc_curve: identity -- {check_region} shelf rate (raw, no floor, no "
                  f"m(bin) scaling), n_measured={int(region_shelf_measured.sum())}: "
                  f"{rate:.4f} against the review's {target}", flush=True)

    st.done(out_path, n_shipped=curve["n_shipped"], n_disk_excess=curve["n_disk_excess"],
            f_floor=curve["f_floor"], floor_regions=curve["floor_regions"], q_min=q_min, n_zeroed=n_zeroed)
    print(f"pahc_curve: shipped n={curve['n_shipped']} disk-excess n={curve['n_disk_excess']} "
          f"f_floor={curve['f_floor']:.6f} (regions {curve['floor_regions']}) "
          f"Q_MIN={q_min:.4f} bins_zeroed={n_zeroed}/{N_Q_BINS} "
          f"({frac_eligible_zeroed:.4%} of the eligible, no-4.5um-excess population) "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    run(build)
