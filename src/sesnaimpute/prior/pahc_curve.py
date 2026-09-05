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
retained field-star population (`prior.field_stars`), which carries
intrinsic (undimmed) fluxes only (module `field_stars.py`'s own
docstring):

  - the photospheric colour relation, [4.5]-[8.0] as the median function of
    [3.6]-[4.5] in narrow bins of the latter -- what predicts a source's
    own photospheric 8 micron flux from its own 3.6 and 4.5 micron fluxes;
  - the population's [3.6]-[4.5] median and 16-84% half-width, a single
    pair of numbers standing in for "a photosphere's [3.6]-[4.5] is nearly
    constant" -- what predicts a source's own photospheric 4.5 micron flux
    from its own 3.6 micron flux, for the 4.5 micron excess test that
    separates a circumstellar disk (whose excess grows smoothly with
    wavelength and shows already at 4.5 micron, where there is no PAH
    feature) from PAH contamination (which does not).

Every TRILEGAL retained star enters the colour relation with equal weight
(SPEC_PRIORS.md section 0.3, C3, "population-weighted"): the STAR anchor
reweighting that turns this same population into a count is a later
stage's own step and plays no part here.

The shipped curve is measured on sources with no 4.5 micron excess; the
curve on the 4.5-micron-excess sources is reported beside it, and the
same shipped construction is repeated per region as a check
(SPEC_PRIORS.md section 4, "Checks"). Both curves and the region checks
share one 40-bin log10 q grid, spanning the full eligible population's own
measured range. The shipped curve's floor -- the mean 8 micron excess
fraction among eligible, no-4.5-excess sources on the flat shelf
`0.1 <= q <= 2.0`, measured per source, not bin-quantized -- is subtracted
and clipped at zero, since a probability cannot be negative; the raw curve
is kept beside it.

Reads only SESNA photometry (the curated catalogues, their own detection
limits) and the external TRILEGAL population -- never a classification
label (rule 7) and never a source's dust column: the [3.6]-[4.5] colour
this module reads off a source is its own *observed* colour, unreddened,
because IRAC's 3.6/4.5/8.0 micron bands sit close enough in wavelength
that their differential reddening is small next to the relation's own
scatter and the catalogue's own photometric error, both already inside
this module's sigma. Writes one survey-wide product,
`bms/pahc/curve_pahc_survey.hdf5`.
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
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module

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

#: SESNA's own eligibility test for this measurement (SPEC_PRIORS.md
#: section 4, "the 1.64 M sources with 3.6, 4.5 and 8.0 micron
#: photometry"): a measured (`ORIGIN_FNU == 1`), finite, positive flux in
#: all three of I1, I2, I4.
ELIGIBLE_BAND_IDX = (IDX_I1, IDX_I2, IDX_I4)


# ---------------------------------------------------------------------------
# 1. the photospheric colour relation, external, from pooled TRILEGAL stars
# ---------------------------------------------------------------------------

def load_field_star_colours(config, region_names):
    """Reads the retained TRILEGAL population's own intrinsic fluxes
    (`prior.field_stars`) for every region in `region_names`, pooled with
    equal weight per star, and returns `(c12, c48)`: `[3.6]-[4.5]` and
    `[4.5]-[8.0]`, Vega mag, on this project's own zero points
    (`constants.VEGA_ZERO_POINT_MJY`) -- the same zero points SESNA's own
    magnitudes below are read on.
    """
    def _one_region(region):
        path = config_module.product_path(
            config, "bms", "trilegal", "field-stars", "region", region=region)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"prior.pahc_curve: field-star product missing for region {region!r} "
                f"at {path!r} -- run the 'prior.field_stars' RUNBOOK line first")
        with h5py.File(path, "r") as f:
            return f["FNU_MJY"][:]

    parts = Parallel(n_jobs=-1)(delayed(_one_region)(r) for r in region_names)
    fnu = np.concatenate(parts, axis=0)
    mag = -2.5 * np.log10(fnu / ZERO_POINT_MJY[None, :])
    c12 = mag[:, IDX_I1] - mag[:, IDX_I2]
    c48 = mag[:, IDX_I2] - mag[:, IDX_I4]
    finite = np.isfinite(c12) & np.isfinite(c48)
    return c12[finite], c48[finite]


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


def region_measurement(config, region, colour_knots, colour45_median, colour45_width):
    """One region's eligible sources (SPEC_PRIORS.md section 4): `q`,
    the 8 micron excess flag, and the 4.5 micron excess flag, `(m,)` each,
    `m` the region's own eligible count -- vectorised over the region's
    sources, no Python loop.
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
    if n_eligible == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty(0, dtype=bool), np.empty(0, dtype=bool)

    f1 = fnu[eligible, IDX_I1]
    f2 = fnu[eligible, IDX_I2]
    f4 = fnu[eligible, IDX_I4]
    sigma2 = sigma[eligible, IDX_I2]
    sigma4 = sigma[eligible, IDX_I4]
    zp1, zp2, zp4 = ZERO_POINT_MJY[IDX_I1], ZERO_POINT_MJY[IDX_I2], ZERO_POINT_MJY[IDX_I4]

    # the source's own observed [3.6]-[4.5], unreddened (module docstring)
    c12 = -2.5 * np.log10(f1 / zp1) + 2.5 * np.log10(f2 / zp2)

    # the photospheric prediction (SPEC_PRIORS.md section 4, "the
    # photospheric prediction"): F_8,pred from F_1, F_2 through the
    # TRILEGAL colour relation
    centers, medians, widths = colour_knots
    c48_pred, width48 = predict_colour_48(c12, centers, medians, widths)
    f4_pred = _predicted_flux(f2, c48_pred, zp2, zp4)
    sigma_relation_48 = f4_pred * np.log(10.0) * 0.4 * width48
    sigma8 = np.sqrt(sigma4 ** 2 + sigma_relation_48 ** 2)
    excess8 = (f4 - f4_pred) / sigma8 > EXCESS_SIGMA

    # the 4.5 micron excess (SPEC_PRIORS.md section 4, "disks separated
    # from PAH"): the same construction, through the population's
    # constant photospheric [3.6]-[4.5]
    f2_pred = _predicted_flux(f1, colour45_median, zp1, zp2)
    sigma_relation_45 = f2_pred * np.log(10.0) * 0.4 * colour45_width
    sigma45 = np.sqrt(sigma2 ** 2 + sigma_relation_45 ** 2)
    excess45 = (f2 - f2_pred) / sigma45 > EXCESS_SIGMA

    f_lim8 = limits_module.limits(config, region)[eligible, IDX_I4]
    q = f_lim8 / f4_pred

    return q, excess8, excess45


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

def write_curve(path, curve, colour_edges, colour_medians, colour_widths,
                 colour45_median, colour45_width):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.create_dataset("LOG10_Q_EDGES", data=curve["edges"].astype(np.float64))
        f.create_dataset("P_Q", data=curve["p_a"].astype(np.float64))
        f.create_dataset("P_Q_RAW", data=curve["p_a_raw"].astype(np.float64))
        f.create_dataset("FLOOR", data=np.float64(curve["floor"]))
        f.create_dataset("N_PER_BIN", data=curve["n_a"].astype(np.int64))
        f.create_dataset("P_Q_DISK_EXCESS", data=curve["p_b_raw"].astype(np.float64))
        f.create_dataset("P_Q_REGION", data=curve["p_region"].astype(np.float64))
        f.create_dataset("N_PER_BIN_REGION", data=curve["n_region_bin"].astype(np.int64))

        cr = f.create_group("COLOUR_RELATION")
        cr.create_dataset("EDGES", data=colour_edges.astype(np.float64))
        cr.create_dataset("MEDIANS", data=colour_medians.astype(np.float64))
        cr.create_dataset("WIDTHS", data=colour_widths.astype(np.float64))

        f.create_dataset("COLOUR_45_MEDIAN", data=np.float64(colour45_median))
        f.create_dataset("COLOUR_45_WIDTH", data=np.float64(colour45_width))


def read(config):
    """The shipped curve `P(q)` as a callable, linear in log10 q, holding
    the edge value beyond the measured range (SPEC_PRIORS.md section 4:
    "Interpolated linearly in log10 q, end bins held beyond the measured
    range")."""
    path = config_module.product_path(config, "bms", "pahc", "curve", "survey")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prior.pahc_curve: no PAHC curve product at {path!r} -- run the "
            f"'prior.pahc_curve' RUNBOOK line first")
    with h5py.File(path, "r") as f:
        edges = f["LOG10_Q_EDGES"][:]
        p_q = f["P_Q"][:]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return interp1d(centers, p_q, kind="linear", bounds_error=False,
                     fill_value=(float(p_q[0]), float(p_q[-1])))


# ---------------------------------------------------------------------------
# 5. build
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Measures `P(q)` over the pool of sources in `regions` (default:
    all thirty), writes `bms/pahc/curve_pahc_survey.hdf5`. The
    photospheric colour relation and the 4.5 micron constant colour pool
    the TRILEGAL retained population of the same region set."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    c12_star, c48_star = load_field_star_colours(config, region_names)
    colour_edges, colour_medians, colour_widths, colour_counts = colour_relation(c12_star, c48_star)
    knots = colour_relation_knots(colour_edges, colour_medians, colour_widths)
    colour45_median = float(np.median(c12_star))
    p16, p84 = np.percentile(c12_star, [16.0, 84.0])
    colour45_width = float(0.5 * (p84 - p16))
    print(f"pahc_curve: {c12_star.size} pooled TRILEGAL retained stars, "
          f"{knots[0].size}/{colour_medians.size} trusted colour bins "
          f"(width {COLOUR_BIN_WIDTH_MAG} mag), "
          f"[3.6]-[4.5] median={colour45_median:.4f} width={colour45_width:.4f} mag", flush=True)

    results = Parallel(n_jobs=-1)(
        delayed(region_measurement)(config, region, knots, colour45_median, colour45_width)
        for region in region_names)
    q = np.concatenate([r[0] for r in results]) if results else np.empty(0)
    excess8 = np.concatenate([r[1] for r in results]) if results else np.empty(0, dtype=bool)
    excess45 = np.concatenate([r[2] for r in results]) if results else np.empty(0, dtype=bool)
    region_idx = np.concatenate([
        np.full(r[0].size, i, dtype=np.int64) for i, r in enumerate(results)]) if results else np.empty(0, dtype=np.int64)
    n_eligible_total = q.size
    print(f"pahc_curve: {n_eligible_total} eligible sources "
          f"(measured I1, I2, I4 in {len(region_names)} regions)", flush=True)

    curve = build_curve(q, excess8, excess45, region_idx, len(region_names))

    out_path = config_module.product_path(config, "bms", "pahc", "curve", "survey")
    write_curve(out_path, curve, colour_edges, colour_medians, colour_widths,
                colour45_median, colour45_width)

    print(f"pahc_curve: shipped n={curve['n_shipped']} disk-excess n={curve['n_disk_excess']} "
          f"floor={curve['floor']:.6f} -> {out_path}", flush=True)


if __name__ == "__main__":
    run(build)
