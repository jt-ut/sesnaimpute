"""The STAR per-tile-and-bin anchor weight `W` (SPEC_PRIORS.md section
2.1, table rows "W", "cluster exclusion", "faint end"; reading note
04_star_family.md section B, `fit_tile_weights`, `shrink_log_normal`,
`cluster_excluded_tiles`), from the tile geometry and the predicted
anchor histograms `population.anchor_tiles` already wrote, and the
young-star-subtracted observed anchor histogram this stage now computes
itself (brief U4, 2026-10-19): the star-anchor chain `population.
young_stars` -> `population.anchor_observed` -> `population.
anchor_weights` is now ONE stage. `region_tile_counts` calls `population.
young_stars.build_region` (the model's own young-star expectation) and
`population.anchor_observed.compute_region` (the young-star-and-cluster
subtraction) directly, in memory, per region; neither writes a product
of its own any more, and the Hunt & Reffert 2023 member table (1.29 M
rows) is read ONCE, for the union of every requested region's own
candidate clusters, by this module's `build` (it was 11-12 s per
region, every region, under the retired `population.anchor_observed`
stage).

    W(t, bin) = N_obs(t, bin) / N_pred(t, bin)

fitted separately on the Gaia `G` marginal, the 2MASS `Ks` marginal, and
the joint `(G, Ks)` grid where the SESNA-Gaia crossmatch supplies both
magnitudes for a source (section 2.1, "joint and marginal bins" -- a
joint `(G, Ks)` cell is used only where the region's pooled count across
its own tiles reaches `MIN_COUNTS` and the cell was therefore actually
fitted (`USE_JOINT` means fitted, not merely a nonzero observed count in
one tile); every other star takes the marginal weight, which by the same
ruling is never left at the uncalibrated 1.0 either (no star ever carries
that unfitted TRILEGAL-level weight).

C21 (2026-10-20): "the region's pooled count across its own tiles" above
was, until now, only ever a single tile's own count -- `fit_tile_weights`
gates each tile on ITS OWN low-count mask, and no joint `w_pool` was ever
passed in, so a joint cell with no ONE tile clearing `MIN_COUNTS` alone
was left unfitted even where the region's tiles, summed, clearly did
(a joint cell is two-dimensional and sparse; almost no single tile's own
crossmatch count reaches 25 in one `(G, Ks)` bin). `region_pooled_ratio`
now sums every unexcluded tile's own `N_GK_OBS`/`N_GK_PRED` first and
gates ONLY the pooled total, and `build_region` passes that as the
joint fit's own `w_pool` -- the same slot `fit_tile_weights` already
uses for the marginals' SURVEY-wide pool, here filled with this ONE
region's own pooled joint histogram instead (a survey-wide joint pool
has no counterpart; item 1 asks for the region level only). `JOINT_SOURCE`
records, per joint cell, which level actually supplied `W_JOINT`: 0 none,
1 a tile of this region on its own, 2 this region's tiles pooled. A cell
unmeasured at BOTH levels is where `population.star_population`'s
placement rule (owner ruling 2026-09-06, item 2) is actually reached now
-- previously it was reached far more often, at every joint cell no
single tile's own crossmatch count happened to clear the floor. Each
axis is fitted the SAME way: the
sky's own per-tile, per-bin counts (aggregated from `population.anchor_tiles`'
per-pixel `N_G_PRED`/`N_KS_PRED`/`N_GK_PRED` and `population.anchor_observed`'s
young-star-subtracted `N_G_SUB`/`N_KS_SUB`/`N_GK_SUB`) over the model's
own predicted count, each bin's Poisson error taken from its own observed
count, a bin under 25 observed counts pooled to the region value, and
every usable bin partially pooled toward the region in the log by the
closed-form two-level moment estimator `shrink_log_normal` (quarry
`anchor_reweighting.shrink_log_normal`, lifted unchanged).

Cluster exclusion (SPEC_BMSTP_DRAFT.md section 5.1, "clusters" row) is
now one label-free mask, evaluated per tile: the tile's own counting-
weighted pooled observed-over-predicted ratio, over every populated bin
on both anchors, departs from the region's MEDIAN of that same statistic
by more than `CLUSTER_EXCLUSION_BAND_DEX` (quarry
`anchor_reweighting.cluster_excluded_tiles`, lifted). A tile overlapping
a Hunt & Reffert 2023 bound cluster (A&A 673, A114; VizieR
J/A+A/673/A114, `sky/download/hunt_reffert2023/clusters.dat.gz`; their
`Type` "o" open or "g" globular, not the unbound moving groups "m")
within its own half-member radius `r50` (the measured radius holding
half its members, where it is dense enough to distort a tile's count)
plus half the tile's own side (`L_STAR_DEG/2`) is NO LONGER excluded on
that account: `population.anchor_observed` subtracts the cluster's own
Gaia members from the tile's observed histogram, bin by bin, before this
module ever forms a ratio, so the tile keeps its own (now
cluster-cleaned) weights instead of losing its evidence outright.
`cluster_excluded_tiles` (below) keeps computing the same geometric test
-- `population.anchor_observed` calls it to decide which clusters'
members are candidates for a region's subtraction, and this module still
calls it too, only to report how many tiles it WOULD have excluded under
the old rule, never to exclude them. Excluding whole tiles for any
catalogued cluster had removed 58% of the survey's tiles and every tile
of 14 regions (`studies/star_galaxy_level.md`), each of which fell back
to the survey-pooled weight even where its own anchors disagreed with it
by a factor of ~2.

Faint end, Gaia axis: `G_EDGES` stops at the Gaia anchor's own faint
edge; the log10 trend of the region-pooled weight across its faintest
three POPULATED bins (bins some tile actually measured) is fit by
least squares and stored as a root attribute, the disclosed
uncertainty on the Gaia faint-end extrapolation (unchanged by the C6
ruling below: the owner's ruling and `REVIEW_LEDGER_2026-10-08.md`'s
row C6 concern the Ks anchor only).

Faint end, Ks axis (ledger row C6, owner's ruling 2026-10-09: ANCHOR,
`bms_review/briefs/SWEEP_2026-10-09.md` WP-POP-3). The deepest Ks
anchor most of the survey carries is 2MASS's own Ks 14.3 (`KS_CUT_MAG`
below) -- 92 percent of the retained field-star population is fainter
than that, so its faint-end weight was TRILEGAL's own unanchored
luminosity-function slope there (`studies/faint_end_excess.md`),
predicting 1.6 to 4 times the observed count at one to three times a
pixel's own depth limit, STAR carrying 83 to 96 percent of the excess.
A region whose `KS_SOURCE` reaches 1 (`population.anchor_tiles`'
UKIDSS GPS or VHS extension to that region's own measured deep edge,
`_region_ks_deep_cut`, at most `KS_DEEP_CUT_MAG` -- about Ks 18, C28
2026-10-21, raised from 17) HAS a real anchor out there; this module's
`deep_slope_dex_per_mag` fits the
log10 region-pooled weight's slope across exactly those deep-sourced,
counting-floor-cleared bins (never the Gaia-axis rule's generic
"faintest three populated bins", which for a region with no deep
coverage would just be 2MASS's own last bins, already absorbed by the
completeness correction `P_KS` below, and never the SESNA catalogue's
own retained counts, which would make the prior read the data it is
meant to predict). `survey_pooled_weights` pools every region's own
deep-measured slope (a straight mean, printed with its own region
count) into ONE survey-wide TRANSPORTED slope. A region with no deep
coverage at all takes that transported slope as `FAINT_TREND_KS_DEX_
PER_MAG` (root attr `FAINT_TREND_KS_TRANSPORTED` discloses this), and
`_extend_shallow_ks_axis` appends synthetic bins to ITS OWN `KS_EDGES`/
`W_KS`/`W_REGION_KS`/`W_JOINT`/`W_REGION_JOINT`/`P_KS` from its own
faint edge out to `KS_DEEP_CUT_MAG`, each new bin's weight the region's
own last REAL bin stepped by the transported slope -- so a star this
faint, read generically by `population.star_population.star_weights`
off whatever `KS_EDGES`/`W_KS` this file carries, gets the transported
correction with no change to that module. The new bins carry no tile
or region evidence of their own (`POPULATED_KS`/`MEASURED_KS`/
`USE_JOINT` all False there): the weight is a disclosed extrapolation,
never a fit. A region whose own deep bins exist but do not clear the
counting floor falls back to the transported slope the same way.

Product, per region, `bms/anchors/weights_anchors_tile__<Region>.hdf5`:
`W_JOINT`/`USE_JOINT` (n_tile, n_G, n_Ks), `W_G` (n_tile, n_G), `W_KS`
(n_tile, n_Ks), `W_REGION_JOINT` (n_G, n_Ks), `W_REGION_G` (n_G),
`W_REGION_KS` (n_Ks), `JOINT_SOURCE` (n_G, n_Ks, int8 -- C21: 0 no joint
evidence at any level, 1 at least one tile of this region cleared
`MIN_COUNTS` on its own, 2 no tile did but this region's tiles pooled
did), `EXCLUDED` (n_tile, bool), `G_EDGES`, `KS_EDGES`,
`P_KS` (n_tile, n_Ks), `KS_M50` (n_tile), `KS_SCALE` (scalar -- the one
fixed instrument width every tile fits with, W38); root attrs
`GRANULE="tile"`, `FAINT_TREND_G_DEX_PER_MAG`, `FAINT_TREND_KS_DEX_PER_MAG`,
`FAINT_TREND_KS_TRANSPORTED` (bool -- C6 ruling: True where this
region's own `KS_SOURCE` never clears the counting floor and
`FAINT_TREND_KS_DEX_PER_MAG` is the OTHER, deep-covered regions'
transported slope rather than this region's own measurement; `n_Ks`
is then `_extend_shallow_ks_axis`'s own longer axis, out to
`KS_DEEP_CUT_MAG`, not `population.anchor_tiles`' original `KS_EDGES`
for this region).

W13 (SPEC_BMSTP_DRAFT.md 5.1's "N^{model->obs}" row), extended by W38 and
W46: 2MASS's raw predicted count `n_pred_ks`/the Ks axis of `n_pred_joint`
is NOT treated as complete to `KS_CUT_MAG` on the 2MASS-served bins
(`KS_SOURCE == 0`, `anchor_tiles.write_histograms`'s own per-bin dataset;
every bin where the region carries no deep survey at all) -- it is
scored there by `P_KS`, the same sigmoid
`anchor_tiles.gaia_detection_weight` uses for Gaia, but with a per-tile
free level `L` and centre `m50_t` fit by Poisson maximum likelihood to
the tile's OWN observed-over-predicted counts (`fit_ks_completeness`),
width held at the fixed instrument floor `KS_SCALE_MAG` (2MASS's
roll-off in a confused field is confusion-driven and has no published
per-field limit, so only its centre is measured per tile). `P_KS = 1`
on the deep, UKIDSS-served bins: the deep survey is treated complete
there relative to the model. This makes `W_KS` the level alone on the
2MASS-served bins, so the faint-end rule there extrapolates a level
rather than a falling completeness curve; on the deep bins `W_KS` is
the plain observed-over-model ratio, W37/W38's own reach. The Gaia axis
(`n_pred_g`, already scored by `gaia_detection_weight`) is untouched.

Owner ruling 2026-09-06. The survey-pooled weight (`survey_pooled_weights`,
item 1: the ratio of observed to predicted counts summed over every
region's own populated tiles, computed once over all thirty regions) is
folded into `W_G`/`W_KS`/`W_REGION_G`/`W_REGION_KS` as the fallback for a
bin no tile of this region measures, never written separately. The
Ks pool sums each bin by its own (lower, upper) edge over exactly the
regions that carry it (the union of the short 2MASS-only and long
UKIDSS-reaching axes, W46), not by array position, so no region's Ks
axis length ever drops it from the pool.
`POPULATED_G`/`POPULATED_KS` (n_G/n_Ks, bool): this region had at least
one of its own unmasked tiles in this bin -- the explicit flag
`faint_trend_dex_per_mag` and `star_population` read, replacing the old
`w_region != 1.0` sentinel (item 3).
"""

import os

import h5py
import numpy as np
import pandas as pd

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.population.anchor_tiles import KS_DEEP_CUT_MAG
from sesnaimpute.sky.derived import twomass_counts as twomass_counts_module

_STEM = "weights_anchors_tile"

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: SPEC_PRIORS.md section 2.1, "W" row: "bins under 25 counts take the
#: region value"; reading note 04B's `LF_DRIFT_MIN_COUNTS`.
MIN_COUNTS = 25.0

#: SPEC_PRIORS.md section 2.1, "cluster exclusion" row: a tile whose
#: pooled ratio departs from the region median by more than a factor 3
#: (0.477 dex) is flagged. `log10(3)` exactly, so the stated dex figure
#: is reproduced rather than re-typed.
CLUSTER_EXCLUSION_EXCESS_FACTOR = 3.0
CLUSTER_EXCLUSION_BAND_DEX = float(np.log10(CLUSTER_EXCLUSION_EXCESS_FACTOR))

#: SPEC_PRIORS.md section 2.1, "faint end" row: "the trend ... across the
#: faintest three populated bins ... is reported".
FAINT_TREND_N_BINS = 3

EXCLUSION_NONE = 0
EXCLUSION_RATIO = 1

#: Hunt & Reffert 2023 (A&A 673, A114) `clusters.dat.gz`'s own byte
#: layout (its ReadMe, "Byte-by-byte Description of file: clusters.dat"):
#: Name (1-20), Type (274, "o" open / "g" globular / "m" unbound moving
#: group), GLON (339-350, deg), GLAT (352-362, deg), r50 (364-374, deg,
#: the measured radius holding half the cluster's members, where a
#: cluster is dense enough to distort a tile's count; SPEC_PRIORS.md
#: section 2.1) -- 0-based half-open `pandas.read_fwf` colspecs.
_CLUSTER_COLSPECS = [(0, 20), (273, 274), (338, 350), (351, 362), (363, 374)]
_CLUSTER_NAMES = ["NAME", "TYPE", "GLON", "GLAT", "R50_DEG"]
#: decision 4(ii): only bound clusters count as a mask; Hunt & Reffert's
#: unbound moving groups ("m") are not clusters an anchor star avoids.
_CLUSTER_TYPES_KEPT = ("o", "g")

#: The per-tile 2MASS
#: completeness sigmoid's `m50` bounds. 17 is "complete" -- at the fixed
#: scale below, p(14.3; 17, 0.102) = 1 - 3.2e-11, past 1 - 1e-11. 8 is a
#: numerical guard only (a tile with no evidence at all pins `m50` at a
#: bound rather than the optimiser wandering; nothing in the survey
#: reads a tile that bright as physical).
KS_M50_NUMERICAL_GUARD_MAG = 8.0
KS_M50_HI_MAG = 17.0

#: studies/2mass_rolloff_width.md: the 2MASS PSC's own intrinsic Ks
#: roll-off, 0.074 dex (from the ExSup's Table 2 99% point and Figures
#: 3-5's digitised 50% point of the calibration-field completeness
#: curve), converted mag = 2.5 dex (a magnitude is -2.5 log10 flux) to a
#: Gaussian sigma of 0.185 mag, then to this module's logistic scale by
#: matching variances (`pi^2 s^2/3 = sigma^2`, so `s = sigma*sqrt(3)/pi`).
#: W38 (coordinator's refit, 2026-09-09): a 5-mag-bounded free width let
#: a level decline imitate a sigmoid (62/196 Cygnus X tiles, 80/311
#: Pipe, sat at the old `m50` floor with `s` 2-5); held at this measured
#: PSC value instead, `m50` alone fits every tile of both regions with
#: no lower bound needed.
KS_ROLLOFF_DEX = 0.074
KS_SCALE_MAG = 2.5 * KS_ROLLOFF_DEX * np.sqrt(3.0) / np.pi

#: brief W13: a tile needs this many bins clearing MIN_COUNTS on both
#: the observed and predicted side to carry its own completeness fit;
#: fewer, and it takes the region-summed fit.
KS_COMPLETENESS_MIN_BINS = 3

#: WP-POP-3 (C6, owner's ruling 2026-10-09: ANCHOR). A region without
#: its own deep-sourced evidence takes the OTHER regions' transported
#: faint-end slope, extrapolated in steps of this width from its own
#: faint edge out to `KS_DEEP_CUT_MAG` -- the same granularity as the
#: shared axis's own regular bins (module docstring).
KS_FAINT_EXTENSION_STEP_MAG = 0.5

#: The least number of deep-sourced (`KS_SOURCE == 1`), counting-floor-
#: cleared bins a region's own Ks axis needs for `deep_slope_dex_per_mag`
#: to fit a line through them at all.
DEEP_SLOPE_MIN_BINS = 2

#: C30 (briefs/round4_2026-10-20/C30_population_anchor.md) item 3: the
#: region's own 2MASS colour-cell giant model count (summed over the
#: five Ks bins and four J-Ks bins, non-excluded tiles) must clear this
#: floor before W_GIANT is fitted at all -- below it there is no split,
#: and a giant reads its bin's W_DWARF exactly like a dwarf (today's
#: construction). The brief's own proposal: 500, from C24i's diagnostic
#: (c18_diag/c24i_population_weights.py) finding Perseus and NGC 7129's
#: giant fits (W_GIANT errors 5-8% of the value there, vs <2% in the
#: regions with thousands of giant counts) unstable at a few hundred
#: giant counts -- measured per region and cited in this module's own
#: printed report (N_GIANT_MODEL), not re-derived here.
N_GIANT_MIN = 500.0


# ---------------------------------------------------------------------------
# lifted from the quarry unchanged (reading note 04B: "correct; lift")
# ---------------------------------------------------------------------------

def shrink_log_normal(f_hat, sig_comb, mask=None):
    """Partially pool the per-tile ratios toward their region, in the
    log, by the closed-form two-level moment estimator (quarry
    `anchor_reweighting.shrink_log_normal`): `y* = mu + B (y - mu)`,
    `B = tau2 / (s^2 + tau2)`. A perfectly measured tile (`s = 0`) keeps
    its own value (`B = 1`); a masked tile (`s = inf`) collapses exactly
    to the region scalar (`B = 0`) -- the mechanism `fit_tile_weights`
    (below) uses to give a low-count or cluster-excluded tile the
    region-pooled value rather than a separately coded fallback.
    """
    f_hat = np.asarray(f_hat, float)
    sig = np.asarray(sig_comb, float)
    n = f_hat.size
    mask = np.zeros(n, bool) if mask is None else np.asarray(mask, bool)

    bad = np.isfinite(sig) & (sig <= 0.0)
    if np.any(bad):
        raise ValueError(
            "shrink_log_normal: tiles %s carry a non-positive combined error; "
            "a tile measured to nothing keeps its own value and must not be "
            "filtered into the masked set" % np.flatnonzero(bad).tolist())

    usable = (np.isfinite(f_hat) & (f_hat > 0) & np.isfinite(sig) & (sig > 0)
              & ~mask)
    y = np.where(f_hat > 0, np.log(np.where(f_hat > 0, f_hat, 1.0)), 0.0)
    s = np.where(usable, sig / np.where(f_hat > 0, f_hat, 1.0), np.inf)
    inv_s2 = np.where(usable, 1.0 / np.where(usable, s, 1.0) ** 2, 0.0)

    sum_w0 = float(inv_s2.sum())
    if sum_w0 <= 0:
        raise ValueError(
            "shrink_log_normal: no tile carries a usable amplitude and error")
    mu0 = float(np.sum(inv_s2 * y) / sum_w0)
    q = float(np.sum(inv_s2 * (y - mu0) ** 2))
    n_eff = int(np.count_nonzero(usable))
    # the moment estimator's denominator, written as the sum over pairs it
    # actually is -- its textbook spelling subtracts two nearly equal sums,
    # and one tile measured far better than the rest makes them equal to
    # the last bit a float can hold.
    denom = 2.0 * float(np.sum(inv_s2 * (np.cumsum(inv_s2) - inv_s2))) / sum_w0
    tau2 = 0.0 if denom <= 0 else max(0.0, (q - (n_eff - 1)) / denom)

    w = np.where(usable, 1.0 / (np.where(usable, s, 1.0) ** 2 + tau2), 0.0)
    sum_w = float(w.sum())
    mu = float(np.sum(w * y) / sum_w) if sum_w > 0 else mu0
    var_mu = 1.0 / sum_w if sum_w > 0 else 1.0 / sum_w0

    b = tau2 / (s ** 2 + tau2) if tau2 > 0 else np.zeros(n)
    b = np.where(np.isfinite(b), b, 0.0)
    y_star = mu + b * (y - mu)
    sig_star = np.sqrt(tau2 * (1.0 - b) + (1.0 - b) ** 2 * var_mu)

    return {"f_rt": np.exp(y_star), "sig_frac": sig_star, "mu_log": mu,
            "mu_factor": float(np.exp(mu)), "tau2": float(tau2), "q": q,
            "n_eff": n_eff, "shrinkage_weight": b, "y": y, "s": s,
            "usable": usable, "var_mu": var_mu}


# ---------------------------------------------------------------------------
# the per-bin Poisson error and the per-axis pooled log10 ratio
# ---------------------------------------------------------------------------

def sigma_stat_frac(n_obs):
    """The ratio's fractional error, from the OBSERVED count's own
    Poisson statistics alone (SPEC_PRIORS.md section 2.1, "W" row:
    "Poisson error from the observed count")."""
    n_obs = np.asarray(n_obs, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_obs > 0, 1.0 / np.sqrt(np.where(n_obs > 0, n_obs, 1.0)),
                        np.inf)


def _log10_ratio_and_precision(n_obs, n_pred, min_counts=MIN_COUNTS):
    """One anchor axis's `(n_tile, n_bin)` log10 ratio and its inverse-
    variance precision, counting-floor-gated (reading note 04B's
    `_axis_log10_ratio_and_precision`, lifted with the Poisson-from-
    observed-count error this module uses throughout)."""
    n_obs = np.asarray(n_obs, float)
    n_pred = np.asarray(n_pred, float)
    usable = (n_obs >= min_counts) & (n_pred >= min_counts) & (n_pred > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(usable, n_obs / np.where(n_pred > 0, n_pred, 1.0), np.nan)
    usable = usable & np.isfinite(ratio) & (ratio > 0)
    log_r = np.where(usable, np.log10(np.where(ratio > 0, ratio, 1.0)), 0.0)
    sigma_log10 = sigma_stat_frac(n_obs) / np.log(10.0)
    precision = np.where(usable & np.isfinite(sigma_log10) & (sigma_log10 > 0),
                          1.0 / np.where(sigma_log10 > 0, sigma_log10, 1.0) ** 2, 0.0)
    return log_r, precision


def pooled_tile_log_ratio(n_obs_by_axis, n_pred_by_axis, min_counts=MIN_COUNTS):
    """Per tile, the counting-weighted pooled log10 observed-over-
    predicted ratio over every populated bin on the given axes, and the
    region's own MEDIAN of that statistic over evidence-carrying tiles
    (reading note 04B's `cluster_excluded_tiles`, lifted: the median,
    not the mean, so a rich cluster tile cannot drag the reference toward
    itself)."""
    log_r_parts, prec_parts = [], []
    for n_obs, n_pred in zip(n_obs_by_axis, n_pred_by_axis):
        log_r, precision = _log10_ratio_and_precision(n_obs, n_pred, min_counts)
        log_r_parts.append(log_r)
        prec_parts.append(precision)
    log_r_all = np.concatenate(log_r_parts, axis=1)
    prec_all = np.concatenate(prec_parts, axis=1)
    tile_precision = prec_all.sum(axis=1)
    has_evidence = tile_precision > 0
    safe_precision = np.where(has_evidence, tile_precision, 1.0)
    tile_log_ratio = (prec_all * log_r_all).sum(axis=1) / safe_precision
    region_log_ratio = (float(np.median(tile_log_ratio[has_evidence]))
                         if np.any(has_evidence) else 0.0)
    return tile_log_ratio, tile_precision, has_evidence, region_log_ratio


def region_pooled_ratio(n_obs, n_pred, excluded, min_counts=MIN_COUNTS):
    """C21 item 1: per bin, the REGION-pooled observed-over-predicted
    ratio and whether it clears the counting floor -- every unexcluded
    tile's own count summed FIRST, with no per-tile `low_counts` gate
    (unlike `fit_tile_weights`' own per-tile mask): a joint `(G, Ks)`
    cell is two-dimensional and sparse, so almost no single tile's own
    crossmatch count ever reaches `min_counts` alone while the region's
    tiles, summed, regularly do. Passed as `fit_tile_weights`' `w_pool`
    argument for the joint fit (`build_region`), filling the SAME
    all-tiles-masked fallback slot the marginals' SURVEY-wide pool fills,
    with this ONE region's own pooled joint histogram instead -- a
    joint cell has no survey-wide pooled counterpart (item 1 asks for
    the region level only). Returns `(ratio, populated)`, `(n_bin,)`
    each; `ratio` is NaN wherever `populated` is False, so a caller that
    forgets the flag still gets `fit_tile_weights`' own not-finite
    convention rather than a silently wrong number."""
    keep = ~np.asarray(excluded, dtype=bool)
    n_obs = np.asarray(n_obs, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    sum_obs = np.where(keep[:, None], n_obs, 0.0).sum(axis=0)
    sum_pred = np.where(keep[:, None], n_pred, 0.0).sum(axis=0)
    populated = (sum_obs >= min_counts) & (sum_pred >= min_counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(populated, sum_obs / np.where(sum_pred > 0, sum_pred, 1.0), np.nan)
    return ratio, populated


def cluster_excluded_by_ratio(n_obs_gaia, n_pred_gaia, n_obs_ks, n_pred_ks,
                               band_dex=CLUSTER_EXCLUSION_BAND_DEX):
    """Decision 4(i): a tile whose pooled ratio (both anchors) departs
    from the region median by more than `band_dex`, symmetric (an excess
    and a deficit are the same "not measuring the field" disagreement)."""
    tile_log_ratio, tile_precision, has_evidence, region_log_ratio = \
        pooled_tile_log_ratio((n_obs_gaia, n_obs_ks), (n_pred_gaia, n_pred_ks))
    excluded = has_evidence & (np.abs(tile_log_ratio - region_log_ratio) > band_dex)
    return dict(excluded=excluded, tile_log_ratio=tile_log_ratio,
                tile_precision=tile_precision, has_evidence=has_evidence,
                region_log_ratio=region_log_ratio, band_dex=float(band_dex))


# ---------------------------------------------------------------------------
# decision 4(ii): the Hunt & Reffert 2023 catalogue mask
# ---------------------------------------------------------------------------

def _read_hunt_reffert_clusters(config):
    """Bound clusters only (`Type` in "o" open, "g" globular; the unbound
    moving groups "m" are dropped), by their measured total radius
    `r50` (SPEC_PRIORS.md section 2.1, "cluster exclusion", as amended
    2026-09-05)."""
    path = f"{config.data_root}/sky/download/hunt_reffert2023/clusters.dat.gz"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.anchor_weights: Hunt & Reffert 2023 cluster table missing at "
            f"{path} -- run `sesnaimpute.sky.download.hunt_reffert2023.build` first")
    df = pd.read_fwf(path, colspecs=_CLUSTER_COLSPECS, names=_CLUSTER_NAMES,
                      header=None, dtype=str, compression="gzip")
    df["TYPE"] = df["TYPE"].str.strip()
    for col in ("GLON", "GLAT", "R50_DEG"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ok = (df["TYPE"].isin(_CLUSTER_TYPES_KEPT) & np.isfinite(df["GLON"])
          & np.isfinite(df["GLAT"]) & np.isfinite(df["R50_DEG"]))
    df = df[ok]
    return dict(name=df["NAME"].str.strip().to_numpy(dtype=str),
                glon=df["GLON"].to_numpy(dtype=np.float64),
                glat=df["GLAT"].to_numpy(dtype=np.float64),
                r50_deg=df["R50_DEG"].to_numpy(dtype=np.float64))


def _angular_sep_deg(l1_deg, b1_deg, l2_deg, b2_deg):
    """Great-circle separation, vectorised over both arguments'
    broadcastable shapes -- the small-angle Euclidean approximation
    fails near the Galactic poles some regions' tiles sit close to."""
    l1 = np.radians(l1_deg)
    b1 = np.radians(b1_deg)
    l2 = np.radians(l2_deg)
    b2 = np.radians(b2_deg)
    cos_sep = np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2)
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


def cluster_excluded_tiles(tile_l_deg, tile_b_deg, l_star_deg, clusters):
    """The geometric overlap test (SPEC_BMSTP_DRAFT.md 5.1 "clusters"
    row): a tile whose centre sits within a Hunt & Reffert 2023 bound
    cluster's own half-member radius `r50` PLUS half the tile's own side
    (`l_star_deg / 2`, this region's tile grid resolution -- "tile centre
    within r50 plus half the tile's own extent"). `clusters` already
    carries only `Type` "o"/"g" entries (`_read_hunt_reffert_clusters`).
    Returns the overlap mask and, per tile, the name of the nearest
    satisfying cluster (empty string where none).

    This module no longer excludes a flagged tile (module docstring):
    the mask is used here only to report how many tiles would have been
    excluded under the old rule, and by `population.anchor_observed` to
    pick which clusters' own members are candidates for a region's
    subtraction.
    """
    n_t = tile_l_deg.size
    half_extent = 0.5 * float(l_star_deg)
    sep = _angular_sep_deg(tile_l_deg[:, None], tile_b_deg[:, None],
                            clusters["glon"][None, :], clusters["glat"][None, :])
    margin = sep - (clusters["r50_deg"][None, :] + half_extent)
    inside = margin <= 0.0
    excluded = inside.any(axis=1)
    nearest = np.full(n_t, "", dtype=object)
    if clusters["name"].size:
        best = np.argmin(np.where(inside, margin, np.inf), axis=1)
        for t in np.flatnonzero(excluded):
            nearest[t] = str(clusters["name"][best[t]])
    return excluded, nearest


# ---------------------------------------------------------------------------
# the per-axis and joint weight fit
# ---------------------------------------------------------------------------

def fit_tile_weights(n_obs, n_pred, excluded, min_counts=MIN_COUNTS,
                      w_pool=None):
    """The per-tile, per-bin reweighting factor on one axis (`(n_tile,
    n_bin)`, the marginal `G`/`Ks` grid or the joint grid flattened to
    one bin axis): the sky's own count over the model's, shrunk toward
    the region-pooled value in the log by `shrink_log_normal`. A bin
    below `min_counts` on either side, or a ratio-excluded tile
    (`cluster_excluded_by_ratio`; cluster overlap no longer excludes a
    tile, module docstring), is masked into the pooled fit for that bin
    -- not deleted, only pooled (reading note 04B's `fit_tile_weights`)
    -- so its own stored weight IS the region-pooled value with no
    separate override, and its evidence never enters the pool other
    tiles shrink toward.

    Owner ruling 2026-09-06, item 1: a bin with NO unmasked tile at all
    (every tile in the region is below the counting floor or ratio-
    excluded) no longer stays at unity. It takes `w_pool[k]`, the
    SURVEY-POOLED ratio for this bin (`survey_pooled_weights`, summed
    over every region's own populated tiles) -- never 1.0, the
    uncalibrated TRILEGAL level. Where the pool itself carries no usable
    value for this bin either -- `w_pool` is `None` (the caller is
    computing the pool and has none yet), or `w_pool[k]` is itself not
    finite (`survey_pooled_weights` leaves a bin with zero summed
    predicted counts at NaN) or not positive, or the bin has no matching
    edge in the pool's own axis (`_pool_at_region_bins`) -- the bin is
    left NOT FINITE (`w[:, k] = w_region[k] = nan`), never at the
    initialised 1.0: a finite-but-uncalibrated weight would be invisible
    to the `isfinite` marginal-fallback test `star_population.star_weights`
    runs on it, and the ruling is "never 1.0", not "never 1.0 except when
    unmeasured."

    Returns `populated` (bin had at least one unmasked tile of its OWN
    region -- the explicit flag item 3 asks for, replacing the `w_region
    != 1.0` sentinel) and `pooled` (bin fell back to the survey pool)
    alongside the usual `w`/`b`/`w_region`.
    """
    n_t, n_bin = n_obs.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_pred > 0, n_obs / np.where(n_pred > 0, n_pred, 1.0), np.nan)
    sigma_frac = sigma_stat_frac(n_obs)
    with np.errstate(invalid="ignore"):
        sigma = ratio * sigma_frac
    low_counts = (n_obs < min_counts) | (n_pred < min_counts)
    unmeasured = ~np.isfinite(ratio) | (ratio <= 0)

    w = np.ones((n_t, n_bin))
    b = np.zeros((n_t, n_bin))
    w_region = np.ones(n_bin)
    populated = np.zeros(n_bin, dtype=bool)
    pooled = np.zeros(n_bin, dtype=bool)
    for k in range(n_bin):
        mask_k = low_counts[:, k] | unmeasured[:, k] | excluded
        if np.all(mask_k):
            # no tile in THIS region carries a usable count in this bin:
            # no region evidence to reweight or pool by, so fall back to
            # the survey-pooled ratio rather than pooling nothing into
            # something.
            if w_pool is not None and np.isfinite(w_pool[k]) and w_pool[k] > 0:
                w[:, k] = w_pool[k]
                w_region[k] = w_pool[k]
                pooled[k] = True
            else:
                # neither this region nor the survey pool (absent, or
                # itself NaN where the pool's own sum_pred is zero, or
                # unmatched by `_pool_at_region_bins`) has evidence for
                # this bin: leave it NOT FINITE rather than at the
                # initialised 1.0, so a downstream `isfinite` test (the
                # marginal fallback in `star_population.star_weights`)
                # reads it as unmeasured instead of as a calibrated
                # weight; no star is ever handed this uncalibrated level.
                w[:, k] = np.nan
                w_region[k] = np.nan
            continue
        populated[k] = True
        pooled_fit = shrink_log_normal(ratio[:, k], sigma[:, k], mask=mask_k)
        w[:, k] = pooled_fit["f_rt"]
        b[:, k] = pooled_fit["shrinkage_weight"]
        w_region[k] = pooled_fit["mu_factor"]
    return dict(w=w, b=b, w_region=w_region, ratio=ratio, populated=populated,
                pooled=pooled)


# ---------------------------------------------------------------------------
# C30 (briefs/round4_2026-10-20/C30_population_anchor.md) item 3: the
# per-population (giant/dwarf) split of the 2MASS colour-cell range,
# Ks 12.0-14.3 -- TWO unknown families, W_GIANT (one value per region)
# and W_DWARF (per tile and per Ks bin, through the SAME fit_tile_weights
# pooling/shrinkage `fit_ks` above already runs). Below N_GIANT_MIN there
# is no split: a giant reads its bin's W_KS exactly like a dwarf, and
# `fit_ks` (the ordinary, unsplit fit over the WHOLE Ks axis, computed
# first in every case) is never touched -- the identity the brief asks
# for (bit-identical to the baseline with the split disabled) holds by
# construction, not by a separate code path.
# ---------------------------------------------------------------------------

def _colour_cell_wls(obs, model_giant, model_dwarf):
    """Weighted least squares `obs = W_giant * model_giant + W_dwarf *
    model_dwarf` over flattened (Ks bin, J-Ks bin) cells, `sigma^2 =
    max(obs, 1)` (C24i's own construction, `c18_diag/
    c24i_population_weights.py`'s `wls`, reused exactly). Returns
    `(w_giant, w_dwarf, sigma_giant, sigma_dwarf)`."""
    s2 = np.maximum(obs, 1.0)
    a = np.column_stack([model_giant, model_dwarf]) / np.sqrt(s2)[:, None]
    y = obs / np.sqrt(s2)
    cov = np.linalg.inv(a.T @ a)
    w = cov @ a.T @ y
    e = np.sqrt(np.diag(cov))
    return float(w[0]), float(w[1]), float(e[0]), float(e[1])


def _read_twomass_cells(config, region, pixels):
    """The region's own 2MASS (J-Ks, Ks) colour-cell counts
    (`sky.derived.twomass_counts`'s `N_CELL`, `CELL_COMPLETE`,
    `CELL_KS_EDGES`, `CELL_JK_EDGES`): the observed side of the giant/
    dwarf split. Fails if this product's own pixel set disagrees with
    `pixels` (the tiles/histograms products' own set, already cross-
    checked against each other)."""
    path = config_module.product_path(config, "sky/derived", "twomass", "counts", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prior.anchor_weights: 2MASS counts missing for region {region!r} at {path} "
            "-- run `sesnaimpute.sky.derived.twomass_counts` first")
    with h5py.File(path, "r") as f:
        tm_pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        cell_ks_edges = np.asarray(f["CELL_KS_EDGES"][:], dtype=np.float64)
        cell_jk_edges = np.asarray(f["CELL_JK_EDGES"][:], dtype=np.float64)
        n_cell = np.asarray(f["N_CELL"][:], dtype=np.float64)
        cell_complete = np.asarray(f["CELL_COMPLETE"][:], dtype=bool)
    if not np.array_equal(tm_pix, pixels):
        raise ValueError(
            f"prior.anchor_weights: {region!r}'s 2MASS counts pixel set disagrees "
            "with the tiles/histograms products' own set -- rerun "
            "`sesnaimpute.sky.derived.twomass_counts` for this region")
    return cell_ks_edges, cell_jk_edges, n_cell, cell_complete


def fit_population_split(config, region, tiles, hist, excluded, n_obs_ks, n_pred_ks,
                          n_giant_min=N_GIANT_MIN):
    """`(w_giant, w_giant_err, n_giant_model, split, dwarf5)`: this
    region's own W_GIANT (a single scalar fit, C24i's region-pooled
    colour-cell construction) and, where `n_giant_model` (the region's
    own giant model count over the colour-cell range, non-excluded
    tiles) clears `n_giant_min`, the giant-corrected per-tile inputs
    `fit_tile_weights` re-fits the first `n_cell_k` Ks bins on
    (`dwarf5`, a dict with `n_obs_dwarf`/`n_pred_dwarf`/`n_clamped`/
    `n_cell_k`) -- `None` where `split` is False, in which case the
    CALLER makes no further change to its own, already-computed
    unsplit `fit_ks`.

    `n_pred_ks`/`n_obs_ks` are `build_region`'s own already-tile-
    aggregated, P_KS-scaled arrays (the SAME inputs the unsplit `fit_ks`
    call uses) -- giants and dwarfs share one P_KS (a 2MASS-detection
    completeness factor, not a stellar-population one), so splitting
    AFTER that scaling keeps every number on `fit_ks`'s own scale.
    """
    pixels = hist["pixels"]
    tile_of_pix = tiles["tile_of_pix"]
    n_tile = tiles["tile_l_deg"].size
    cell_ks_edges, cell_jk_edges, obs_cell_pix, cell_complete_pix = _read_twomass_cells(config, region, pixels)
    n_cell_k = cell_ks_edges.size - 1
    if not np.array_equal(hist["ks_edges"][: n_cell_k + 1], cell_ks_edges):
        raise ValueError(
            f"prior.anchor_weights: {region!r}'s KS_EDGES do not open on the "
            "2MASS colour cells' own CELL_KS_EDGES -- sky.derived.twomass_counts "
            "and population.anchor_tiles have drifted apart")
    if not (np.array_equal(hist["cell_ks_edges"], cell_ks_edges)
            and np.array_equal(hist["cell_jk_edges"], cell_jk_edges)):
        raise ValueError(
            f"prior.anchor_weights: {region!r}'s model colour cells "
            "(population.anchor_tiles) and observed colour cells "
            "(sky.derived.twomass_counts) disagree on their own grid")

    obs_cell_pix = np.where(cell_complete_pix, obs_cell_pix, 0.0)
    obs_cell_tile = _aggregate_to_tiles(obs_cell_pix, tile_of_pix, n_tile)
    giant_cell_tile = _aggregate_to_tiles(hist["n_giant_cell"], tile_of_pix, n_tile)
    dwarf_cell_tile = _aggregate_to_tiles(hist["n_dwarf_cell"], tile_of_pix, n_tile)

    keep = ~excluded
    obs_cell_region = obs_cell_tile[keep].sum(axis=0)
    giant_cell_region = giant_cell_tile[keep].sum(axis=0)
    dwarf_cell_region = dwarf_cell_tile[keep].sum(axis=0)
    n_giant_model = float(giant_cell_region.sum())
    split = n_giant_model > n_giant_min
    if not split:
        return float("nan"), float("nan"), n_giant_model, False, None

    w_giant, _w_dwarf0, e_giant, _e_dwarf0 = _colour_cell_wls(
        obs_cell_region.ravel(), giant_cell_region.ravel(), dwarf_cell_region.ravel())

    giant_tile_bin = giant_cell_tile.sum(axis=2)
    dwarf_tile_bin = dwarf_cell_tile.sum(axis=2)
    total_tile_bin = giant_tile_bin + dwarf_tile_bin
    giant_share = np.where(total_tile_bin > 0, giant_tile_bin / np.where(total_tile_bin > 0, total_tile_bin, 1.0), 0.0)
    n_pred_giant_5 = n_pred_ks[:, :n_cell_k] * giant_share
    n_pred_dwarf_5 = n_pred_ks[:, :n_cell_k] - n_pred_giant_5
    n_obs_dwarf_5_raw = n_obs_ks[:, :n_cell_k] - w_giant * n_pred_giant_5
    n_clamped = int(np.count_nonzero(n_obs_dwarf_5_raw < 0.0))
    n_obs_dwarf_5 = np.maximum(n_obs_dwarf_5_raw, 0.0)
    dwarf5 = dict(n_obs_dwarf=n_obs_dwarf_5, n_pred_dwarf=n_pred_dwarf_5,
                  n_clamped=n_clamped, n_cell_k=n_cell_k)
    return w_giant, e_giant, n_giant_model, True, dwarf5


# ---------------------------------------------------------------------------
# W13: the 2MASS anchor's own per-tile completeness, fit from the tile's
# own counts against the model's shape (SPEC_BMSTP_DRAFT.md 5.1)
# ---------------------------------------------------------------------------

def ks_completeness_weight(m, m50, s):
    """`p(m; m50, s)`, `anchor_tiles.gaia_detection_weight`'s sigmoid
    with a per-tile free centre in place of Cantat-Gaudin's fixed one
    -- the 2MASS PSC's own completeness curve, replacing the "complete
    to Ks=14.3" assumption (module docstring, "W13"). `s` is the one
    module-wide `KS_SCALE_MAG` (W38); kept as its own argument rather
    than a bare module read so the region-summed and per-tile fits
    below share one call."""
    m = np.asarray(m, dtype=np.float64)
    arg = (m - m50) / s
    return 1.0 / (1.0 + np.exp(np.minimum(arg, 700.0)))


#: W38: the profiled-likelihood grid step for `m50` -- five times finer
#: than `KS_SCALE_MAG`'s own transition width, so the search never
#: straddles a genuinely narrow roll-off. A gradient search started at
#: `m50 = KS_M50_HI_MAG` sees an exactly flat Poisson likelihood there
#: once `s` is this narrow (every 2MASS-served bin centre sits many
#: widths below it, so nudging `m50` from 17 to 15 changes not one
#: predicted bin count to machine precision) and never leaves the
#: bound -- the grid evaluates every candidate directly instead.
KS_M50_GRID_STEP_MAG = 0.02


def _fit_ks_completeness_batch(n_obs, n_pred, m_centers):
    """`(n_row,)` `m50` by Poisson maximum likelihood, one row per tile
    (or one row, the region-summed fit), over a grid spanning
    `KS_M50_NUMERICAL_GUARD_MAG` to `KS_M50_HI_MAG` (brief W13/W38).
    `L` is profiled out in closed form at every grid point -- the
    Poisson MLE of a pure scale factor on a fixed shape is the ratio of
    the observed to the (unit-`L`) predicted sums -- so the search is
    genuinely one-dimensional and vectorised over rows and grid points
    at once (CODING_RULES.md rule 8: no Python loop over tiles or grid
    points); `L` itself is discarded, the level being `W`'s job.
    """
    # KS_M50_HI_MAG appended exactly, not just approached by the step, so
    # the common "no roll-off observed" outcome lands on the true bound
    # to machine precision rather than one step short of it.
    grid = np.arange(KS_M50_NUMERICAL_GUARD_MAG, KS_M50_HI_MAG, KS_M50_GRID_STEP_MAG)
    grid = np.concatenate([grid, [KS_M50_HI_MAG]])
    p = ks_completeness_weight(m_centers[None, None, :], grid[None, :, None], KS_SCALE_MAG)
    mu_unit = p * n_pred[:, None, :]  # (n_row, n_grid, n_bin)
    sum_unit = mu_unit.sum(axis=2)  # (n_row, n_grid)
    n_obs_total = n_obs.sum(axis=1, keepdims=True)
    l_hat = np.where(sum_unit > 0, n_obs_total / np.where(sum_unit > 0, sum_unit, 1.0), 0.0)
    mu = np.maximum(l_hat[:, :, None] * mu_unit, 1e-300)
    nll = np.sum(mu - n_obs[:, None, :] * np.log(mu), axis=2)  # (n_row, n_grid)
    return grid[np.argmin(nll, axis=1)]


def fit_ks_completeness(n_obs_ks, n_pred_ks, ks_edges, served_mask):
    """Per tile `m50_t` (brief W13/W38), fit only on `served_mask` --
    the 2MASS-served bins (`KS_SOURCE == 0`; every bin where the region
    carries no deep survey at all, since then the whole axis is
    2MASS's). A tile with at least
    `KS_COMPLETENESS_MIN_BINS` served bins clearing `MIN_COUNTS` on
    both the observed and the predicted side fits its own counts;
    otherwise it takes the region-summed fit (the same model on the
    region's pooled histograms, same served bins). `P_KS` is 1 on every
    bin outside `served_mask` (module docstring, "W13"): the deep,
    UKIDSS-served bins are treated complete relative to the model there.
    Returns `(m50, p_ks, own_fit)`: `m50` and `own_fit` are `(n_tile,)`,
    `p_ks` is `(n_tile, n_bin)`.
    """
    m_centers = 0.5 * (ks_edges[:-1] + ks_edges[1:])
    n_tile, n_bin = n_obs_ks.shape
    served_idx = np.flatnonzero(served_mask)
    obs_s = n_obs_ks[:, served_idx]
    pred_s = n_pred_ks[:, served_idx]
    centers_s = m_centers[served_idx]
    clears = (obs_s >= MIN_COUNTS) & (pred_s >= MIN_COUNTS)
    own_fit = clears.sum(axis=1) >= KS_COMPLETENESS_MIN_BINS

    m50_region = _fit_ks_completeness_batch(obs_s.sum(axis=0, keepdims=True),
                                             pred_s.sum(axis=0, keepdims=True), centers_s)[0]
    m50 = np.full(n_tile, m50_region, dtype=np.float64)
    if np.any(own_fit):
        m50[own_fit] = _fit_ks_completeness_batch(obs_s[own_fit], pred_s[own_fit], centers_s)

    p_ks = np.ones((n_tile, n_bin), dtype=np.float64)
    p_ks[:, served_idx] = ks_completeness_weight(centers_s[None, :], m50[:, None], KS_SCALE_MAG)
    return m50, p_ks, own_fit


def ks_served_mask(ks_source):
    """`(n_bin,)` bool: this region's own 2MASS-served bins (W46 item 2)
    -- the bins `anchor_tiles`' own per-bin `KS_SOURCE` marks 2MASS-
    chosen (0). The complement is the UKIDSS-chosen bins, shared or
    deep, the completeness fit above never touches."""
    return np.asarray(ks_source) == 0


# ---------------------------------------------------------------------------
# owner ruling 2026-09-06, item 1: the survey-pooled fallback weight
# ---------------------------------------------------------------------------

def region_tile_counts(config, region, clusters, members_all, min_counts=MIN_COUNTS, cache=None):
    """One region's tile-level predicted/observed counts and exclusion
    mask, with no weight fit performed -- the read-and-aggregate half of
    `build_region`, split out so `survey_pooled_weights` (the once-over-
    all-thirty-regions pass, item 1) and the per-region fit share the
    same counts rather than reading the on-disk products twice with two
    slightly different code paths.

    The young-star expectation and the observed-anchor subtraction
    (brief U4: the retired `population.young_stars` and
    `population.anchor_observed` stages) are computed HERE, in memory,
    from the one histograms read below -- no `young-stars_anchors_
    hpx512` or `observed_anchors_hpx512` file is read or written any
    more. `cache` (set by `build`, a plain `{region: (young, obs)}`
    dict shared across this call and `survey_pooled_weights`' own pass
    over all thirty regions) makes sure each region's young-star
    quadrature and cluster-member subtraction run exactly ONCE per
    `build()` call, however many times `region_tile_counts` itself is
    invoked for that region."""
    from sesnaimpute.population import anchor_observed, young_stars

    tiles = _read_tiles(config, region)
    hist = _read_histograms(config, region)

    if not np.array_equal(tiles["pixels"], hist["pixels"]):
        raise ValueError(
            f"prior.anchor_weights: {region!r}'s tiles/histograms products "
            "disagree on their own pixel set -- rerun `prior.anchor_tiles` for this region")

    if cache is not None and region in cache:
        young, obs = cache[region]
    else:
        young = young_stars.build_region(config, region)
        obs = anchor_observed.compute_region(config, region, hist, young, tiles, clusters, members_all)
        young_stars.report(config, region, young)
        anchor_observed.report(region, obs)
        if cache is not None:
            cache[region] = (young, obs)

    n_tile = tiles["tile_l_deg"].size
    tile_of_pix = tiles["tile_of_pix"]
    n_g, n_ks = hist["g_edges"].size - 1, hist["ks_edges"].size - 1

    n_pred_g = _aggregate_to_tiles(hist["n_g_pred"], tile_of_pix, n_tile)
    n_pred_ks = _aggregate_to_tiles(hist["n_ks_pred"], tile_of_pix, n_tile)
    n_pred_joint = _aggregate_to_tiles(hist["n_gk_pred"], tile_of_pix, n_tile)
    n_obs_g = _aggregate_to_tiles(obs["n_g_sub"], tile_of_pix, n_tile)
    n_obs_ks = _aggregate_to_tiles(obs["n_ks_sub"], tile_of_pix, n_tile)
    n_obs_joint = _aggregate_to_tiles(obs["n_gk_sub"], tile_of_pix, n_tile)

    # W13/W38: the 2MASS anchor's own per-tile completeness (module
    # docstring) -- n_pred_ks/the Ks axis of n_pred_joint stop being
    # "complete to KS_CUT_MAG" and become the model's count times this
    # tile's own fitted sigmoid ON THE 2MASS-SERVED BINS (p_ks = 1 on
    # the deep, UKIDSS-served bins), BEFORE the ratio-exclusion test and
    # every fit below reads them. The Gaia axis (n_pred_g) is untouched.
    # a tile MEASURES a Ks bin only where the model has stars in it: on
    # the deep, UKIDSS-served bins that is exactly the tiles the deep
    # survey covers (`anchor_tiles` drops the predicted counts elsewhere),
    # so a tile without deep coverage ends its own Ks axis at the 2MASS
    # cut and never reads a deep-bin weight, pooled or not
    # (`star_population` takes its faint edge from this flag).
    measured_ks = n_pred_ks > 0.0
    served_mask = ks_served_mask(hist["ks_source"])
    ks_m50, p_ks, ks_own_fit = fit_ks_completeness(n_obs_ks, n_pred_ks, hist["ks_edges"], served_mask)
    n_pred_ks = n_pred_ks * p_ks
    n_pred_joint = n_pred_joint * p_ks[:, None, :]
    # the joint table's observed Ks is each survey source's own 2MASS
    # measurement (`population.anchor_observed`), complete only on the
    # 2MASS-served bins: on the deep bins an observed joint count would
    # set an incomplete detection set against a complete prediction, so
    # those joint bins carry no evidence and fall to the marginals
    # (`USE_JOINT` False there), exactly as they did when the axis ended
    # at the 2MASS cut.
    n_obs_joint = np.where(served_mask[None, None, :], n_obs_joint, 0.0)

    ratio_mask = cluster_excluded_by_ratio(n_obs_g, n_pred_g, n_obs_ks, n_pred_ks)
    excluded_ratio = ratio_mask["excluded"]
    # the catalogue overlap test no longer excludes a tile (module
    # docstring): population.anchor_observed has already subtracted
    # those clusters' own members from n_obs_g/n_obs_ks/n_obs_joint
    # above. `excluded_catalog` is kept only to report, per region, how
    # many tiles the old rule would have thrown away.
    excluded_catalog, nearest_cluster = cluster_excluded_tiles(
        tiles["tile_l_deg"], tiles["tile_b_deg"], tiles["l_star_deg"], clusters)
    excluded = excluded_ratio
    reason = np.where(excluded_ratio, EXCLUSION_RATIO, EXCLUSION_NONE).astype(np.int64)

    return dict(
        region=region, n_tile=n_tile, tiles=tiles, hist=hist, obs=obs,
        n_pred_g=n_pred_g, n_pred_ks=n_pred_ks, n_pred_joint=n_pred_joint,
        n_obs_g=n_obs_g, n_obs_ks=n_obs_ks, n_obs_joint=n_obs_joint,
        excluded=excluded, reason=reason, nearest_cluster=nearest_cluster,
        ratio_mask=ratio_mask, excluded_ratio=excluded_ratio,
        excluded_catalog=excluded_catalog, ks_source=hist["ks_source"],
        ks_m50=ks_m50, ks_own_fit=ks_own_fit, p_ks=p_ks, measured_ks=measured_ks)


def _pool_bins_by_edge(lower_list, upper_list, value_lists):
    """The union bin set over a survey of regions whose own axes may
    differ in length: every distinct `(lower, upper)` bin any
    region carries, summed from `value_lists` (one `(n_bin_r,)` array
    per region per entry, same region order as `lower_list`/`upper_list`)
    over exactly the regions whose own axis carries that bin, returned
    ordered by lower edge (ties, the one branch point where the short
    2MASS-only axis's last bin and the long axis's first deep bin both
    start at the same lower edge, broken by upper edge). No Python loop
    over bins: each region's own small bin set is scattered into the
    union axis by one vectorised `searchsorted`/`np.add.at` per region.

    Returns `(lower, upper, n_regions, [summed value arrays])`.
    """
    key_list = [np.round(lo, 4) * 1000.0 + np.round(up, 4)
                for lo, up in zip(lower_list, upper_list)]
    all_keys = np.concatenate(key_list)
    all_lower = np.concatenate([np.round(lo, 4) for lo in lower_list])
    all_upper = np.concatenate([np.round(up, 4) for up in upper_list])
    key_sorted, first_idx = np.unique(all_keys, return_index=True)
    lower_by_key = all_lower[first_idx]
    upper_by_key = all_upper[first_idx]
    n_bin = key_sorted.size

    sums = [np.zeros(n_bin) for _ in value_lists]
    n_regions = np.zeros(n_bin, dtype=np.int64)
    for r, key_r in enumerate(key_list):
        idx = np.searchsorted(key_sorted, key_r)
        for s, values in zip(sums, value_lists):
            np.add.at(s, idx, values[r])
        np.add.at(n_regions, idx, 1)

    order = np.lexsort((upper_by_key, lower_by_key))
    lower_out = lower_by_key[order]
    upper_out = upper_by_key[order]
    n_regions = n_regions[order]
    sums = [s[order] for s in sums]
    return lower_out, upper_out, n_regions, sums


def survey_pooled_weights(config, region_names, clusters, members_all, min_counts=MIN_COUNTS, cache=None):
    """The SURVEY-POOLED weight per magnitude bin, on the `G` and `Ks`
    marginals (owner ruling 2026-09-06, item 1): sum the observed and
    the predicted counts over EVERY region's own populated tiles (not
    cluster-excluded, both counts clearing `min_counts`) and take one
    ratio per bin -- computed once over all thirty regions' stored
    anchor products, never per region.

    The Gaia axis is shared by every region (one `G_EDGES` grid) and
    pools by array position. The Ks axis is not (W46; C28, 2026-10-21:
    a region's own deep edge is now ITS OWN measured completeness
    turnover, capped at `anchor_tiles.KS_DEEP_CUT_MAG` -- no two
    deep-covered regions need reach the same bin any more, and a region
    with no deep survey at all still reaches only 2MASS's own 14.3) --
    each Ks bin is keyed
    by its own `(lower, upper)` edge and summed over exactly the regions
    whose axis carries it (`_pool_bins_by_edge`), never by array
    position, so no region is skipped for its Ks axis length. `skipped`
    now names only a region whose upstream anchor product is missing.

    Also returns the WP-POP-3 (C6) TRANSPORTED Ks faint-end slope: the
    straight mean, over every region whose own deep-sourced bins clear
    `min_counts` (`deep_slope_dex_per_mag` on this region's OWN pooled
    ratio, not yet shrunk toward anything), of that region's own
    deep-range log10 slope -- one survey-wide number, disclosed with
    the regions that contributed it.

    Returns `(w_pool_g, w_pool_ks, g_edges, ks_lower, ks_upper, skipped,
    n_pool_ks, transported_ks_slope, deep_slope_regions)`, `n_pool_ks`
    the count of regions carrying each returned Ks bin.
    """
    sum_obs_g = sum_pred_g = None
    g_edges = None
    ks_lower_list, ks_upper_list, ks_obs_list, ks_pred_list = [], [], [], []
    skipped = []
    deep_slopes, deep_slope_regions = [], []
    for region in region_names:
        try:
            rc = region_tile_counts(config, region, clusters, members_all, min_counts, cache=cache)
        except (FileNotFoundError, KeyError) as exc:
            # flag, do not block: a region whose upstream anchor products
            # are missing or stale (an older schema without N_GK_PRED,
            # say) is left out of the survey pool and named in
            # `skipped`, rather than stopping every other region's build.
            print("prior.anchor_weights: survey pool skipping %r (%s: %s)"
                  % (region, type(exc).__name__, exc))
            skipped.append(region)
            continue
        if g_edges is None:
            g_edges = rc["hist"]["g_edges"]
        usable_g = (~rc["excluded"])[:, None] & (rc["n_obs_g"] >= min_counts) \
            & (rc["n_pred_g"] >= min_counts)
        og = np.where(usable_g, rc["n_obs_g"], 0.0).sum(axis=0)
        pg = np.where(usable_g, rc["n_pred_g"], 0.0).sum(axis=0)
        sum_obs_g = og if sum_obs_g is None else sum_obs_g + og
        sum_pred_g = pg if sum_pred_g is None else sum_pred_g + pg

        ks_edges_r = rc["hist"]["ks_edges"]
        usable_ks = (~rc["excluded"])[:, None] & (rc["n_obs_ks"] >= min_counts) \
            & (rc["n_pred_ks"] >= min_counts)
        ks_obs_r = np.where(usable_ks, rc["n_obs_ks"], 0.0).sum(axis=0)
        ks_pred_r = np.where(usable_ks, rc["n_pred_ks"], 0.0).sum(axis=0)
        ks_lower_list.append(ks_edges_r[:-1])
        ks_upper_list.append(ks_edges_r[1:])
        ks_obs_list.append(ks_obs_r)
        ks_pred_list.append(ks_pred_r)

        # WP-POP-3 (C6): this region's own deep-sourced, floor-cleared
        # bins, fit on the plain region-summed ratio (the slope needs
        # no per-tile shrinkage, only the level does).
        if np.any(rc["ks_source"] == 1):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_r = np.where(ks_pred_r > 0,
                                    ks_obs_r / np.where(ks_pred_r > 0, ks_pred_r, 1.0), np.nan)
            deep_populated_r = (ks_pred_r >= min_counts) & (ks_obs_r >= min_counts) \
                & np.isfinite(ratio_r) & (ratio_r > 0)
            slope_r = deep_slope_dex_per_mag(ratio_r, ks_edges_r, rc["ks_source"], deep_populated_r)
            if np.isfinite(slope_r):
                deep_slopes.append(slope_r)
                deep_slope_regions.append(region)

    with np.errstate(divide="ignore", invalid="ignore"):
        w_pool_g = np.where(sum_pred_g > 0,
                             sum_obs_g / np.where(sum_pred_g > 0, sum_pred_g, 1.0), np.nan)

    ks_lower, ks_upper, n_pool_ks, (sum_obs_ks, sum_pred_ks) = _pool_bins_by_edge(
        ks_lower_list, ks_upper_list, [ks_obs_list, ks_pred_list])
    with np.errstate(divide="ignore", invalid="ignore"):
        w_pool_ks = np.where(sum_pred_ks > 0,
                              sum_obs_ks / np.where(sum_pred_ks > 0, sum_pred_ks, 1.0), np.nan)

    transported_ks_slope = float(np.mean(deep_slopes)) if deep_slopes else float("nan")
    return (w_pool_g, w_pool_ks, g_edges, ks_lower, ks_upper, skipped, n_pool_ks,
            transported_ks_slope, deep_slope_regions)


def _pool_at_region_bins(pool_w, pool_lower, pool_upper, region_edges):
    """`(n_bin,)`: `survey_pooled_weights`' own Ks ratio for each of this
    region's own bins, matched by `(lower, upper)` edge -- not array
    position, since the pool's own axis is the union over all thirty
    regions' bins and one lower edge (14.0) names two different bins
    there (the short axis's 14.0-14.3 and the long axis's first deep
    bin, 14.0-14.5). `nan` where the pool carries no matching bin, so
    `fit_tile_weights`'s existing no-pool path applies.
    """
    region_edges = np.asarray(region_edges, dtype=np.float64)
    r_key = np.round(region_edges[:-1], 4) * 1000.0 + np.round(region_edges[1:], 4)
    p_key = np.round(np.asarray(pool_lower), 4) * 1000.0 + np.round(np.asarray(pool_upper), 4)
    order = np.argsort(p_key)
    p_key_sorted = p_key[order]
    w_sorted = np.asarray(pool_w)[order]
    idx = np.clip(np.searchsorted(p_key_sorted, r_key), 0, p_key_sorted.size - 1)
    matched = np.isclose(p_key_sorted[idx], r_key)
    out = np.full(r_key.size, np.nan)
    out[matched] = w_sorted[idx[matched]]
    return out


# ---------------------------------------------------------------------------
# the faint-end trend (decision 5)
# ---------------------------------------------------------------------------

def faint_trend_dex_per_mag(w_region, edges, populated, n_bins=FAINT_TREND_N_BINS):
    """The log10 slope of the region-pooled weight across the faintest
    `n_bins` POPULATED bins -- "populated" now the explicit per-bin flag
    `fit_tile_weights` returns (owner ruling 2026-09-06, item 3), not the
    `w_region != 1.0` sentinel: a bin measured to exactly 1.0 by real
    tile evidence is populated, and a bin that fell back to the survey
    pool is not, regardless of what number either lands on. `nan` where
    fewer than two populated bins exist to fit a slope through.
    """
    centres = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    populated = np.flatnonzero(np.asarray(populated, dtype=bool))
    if populated.size < 2:
        return float("nan")
    faintest = populated[np.argsort(centres[populated])][-n_bins:]
    x = centres[faintest]
    y = np.log10(w_region[faintest])
    if x.size < 2:
        return float("nan")
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def deep_slope_dex_per_mag(w_region_ks, ks_edges, ks_source, populated_ks,
                            min_bins=DEEP_SLOPE_MIN_BINS):
    """WP-POP-3 (C6 ruling: anchor). The log10(`W_REGION_KS`) slope
    across this region's own DEEP-sourced Ks bins alone (`ks_source ==
    1`: UKIDSS GPS or VVV, measured out to `KS_DEEP_CUT_MAG`) that also
    cleared the counting floor (`populated_ks`) -- never the 2MASS-only
    range, which the completeness correction `P_KS` already absorbs,
    and never a generic "faintest N populated bins" count that could
    reach into that range. `nan` where fewer than `min_bins` such bins
    exist: no deep coverage at all, or too little of it to clear the
    floor (module docstring's "Faint end, Ks axis").
    """
    centres = 0.5 * (np.asarray(ks_edges[:-1]) + np.asarray(ks_edges[1:]))
    mask = np.asarray(populated_ks, dtype=bool) & (np.asarray(ks_source) == 1)
    if np.count_nonzero(mask) < min_bins:
        return float("nan")
    slope, _ = np.polyfit(centres[mask], np.log10(np.asarray(w_region_ks)[mask]), 1)
    return float(slope)


def _extend_shallow_ks_axis(ks_edges, w_ks, w_region_ks, populated_ks, measured_ks,
                             w_joint, use_joint, w_region_joint, p_ks, joint_source,
                             slope_dex_per_mag, n_tile,
                             target_mag=KS_DEEP_CUT_MAG,
                             step_mag=KS_FAINT_EXTENSION_STEP_MAG):
    """WP-POP-3 (C6 ruling: anchor). This region's Ks axis carries no
    evidence past its own survey depth (no deep-sourced bin clears the
    counting floor). Append synthetic bins from this axis's own faint
    edge out to `target_mag` (the deep regions' own reach), each new
    bin's weight the region's own last REAL bin stepped by the
    TRANSPORTED survey-wide deep-region slope (module docstring).
    `populated_ks`/`measured_ks`/`use_joint` are False at every new bin
    (no tile or region evidence of its own there -- a disclosed
    extrapolation, never a fit), `joint_source` is 0 (none) there for
    the same reason (C21); `w_joint`/`w_region_joint` are padded
    with the same Ks marginal value at every Gaia bin, never read since
    `use_joint` stays False.

    Returns the ten Ks-indexed arrays unchanged where the axis already
    reaches `target_mag` or no transported slope is available (this
    region's own faint edge, or `nan`, respectively) -- the no-op path
    `build_region` also takes for a region with its own deep coverage.
    """
    edge0 = float(ks_edges[-1])
    if edge0 >= target_mag - 1e-9 or not np.isfinite(slope_dex_per_mag):
        return (ks_edges, w_ks, w_region_ks, populated_ks, measured_ks,
                w_joint, use_joint, w_region_joint, p_ks, joint_source)

    ref_candidates = np.flatnonzero(populated_ks)
    ref_idx = int(ref_candidates[-1]) if ref_candidates.size else int(populated_ks.size - 1)
    ref_centre = 0.5 * (ks_edges[ref_idx] + ks_edges[ref_idx + 1])
    ref_w_region = float(w_region_ks[ref_idx])
    if not np.isfinite(ref_w_region):
        # neither this region nor the survey pool has a usable value at
        # its own faint edge either -- nothing to step the slope from.
        return (ks_edges, w_ks, w_region_ks, populated_ks, measured_ks,
                w_joint, use_joint, w_region_joint, p_ks, joint_source)
    ref_w_tile = w_ks[:, ref_idx]

    n_new = max(1, int(np.ceil((target_mag - edge0) / step_mag)))
    new_edges = np.concatenate([edge0 + np.arange(1, n_new) * step_mag, [target_mag]])
    new_centres = 0.5 * (np.concatenate([[edge0], new_edges[:-1]]) + new_edges)
    n_new_bins = new_edges.size
    step = 10.0 ** (slope_dex_per_mag * (new_centres - ref_centre))

    new_w_region = ref_w_region * step
    new_w_tile = ref_w_tile[:, None] * step[None, :]
    n_g = w_joint.shape[1]

    return (
        np.concatenate([ks_edges, new_edges]),
        np.concatenate([w_ks, new_w_tile], axis=1),
        np.concatenate([w_region_ks, new_w_region]),
        np.concatenate([populated_ks, np.zeros(n_new_bins, dtype=bool)]),
        np.concatenate([measured_ks, np.zeros((n_tile, n_new_bins), dtype=bool)], axis=1),
        np.concatenate(
            [w_joint, np.broadcast_to(new_w_tile[:, None, :], (n_tile, n_g, n_new_bins))],
            axis=2),
        np.concatenate([use_joint, np.zeros((n_tile, n_g, n_new_bins), dtype=bool)], axis=2),
        np.concatenate(
            [w_region_joint, np.broadcast_to(new_w_region[None, :], (n_g, n_new_bins))],
            axis=1),
        np.concatenate([p_ks, np.ones((n_tile, n_new_bins), dtype=np.float64)], axis=1),
        np.concatenate([joint_source, np.zeros((n_g, n_new_bins), dtype=np.int8)], axis=1),
    )


# ---------------------------------------------------------------------------
# per-tile aggregation of the pixel-granule products
# ---------------------------------------------------------------------------

def _aggregate_to_tiles(per_pixel, tile_of_pix, n_tile):
    """`(n_tile,) + trailing`: `bincount(tile_of_pix, weights=...)` over
    every trailing (magnitude-bin) column at once, via one flattened
    index -- CODING_RULES.md rule 8, no Python loop over pixels or
    tiles."""
    per_pixel = np.asarray(per_pixel, dtype=np.float64)
    trailing = per_pixel.shape[1:]
    flat = per_pixel.reshape(per_pixel.shape[0], -1)
    n_k = flat.shape[1]
    combined = tile_of_pix[:, None] * n_k + np.arange(n_k)[None, :]
    out = np.bincount(combined.ravel(), weights=flat.ravel(), minlength=n_tile * n_k)
    return out.reshape((n_tile,) + trailing)


def _read_tiles(config, region):
    path = config_module.product_path(config, "population", "anchors", "tiles", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prior.anchor_weights: tiles product missing for region {region!r} at "
            f"{path} -- run `sesnaimpute.population.anchor_tiles` first")
    with h5py.File(path, "r") as f:
        return dict(
            pixels=np.asarray(f["HPX_PIX_512"][:], dtype=np.int64),
            tile_of_pix=np.asarray(f["TILE_ID"][:], dtype=np.int64),
            tile_l_deg=np.asarray(f["TILE_L_DEG"][:], dtype=np.float64),
            tile_b_deg=np.asarray(f["TILE_B_DEG"][:], dtype=np.float64),
            l_star_deg=float(f.attrs["L_STAR_DEG"]),
        )


def _read_histograms(config, region):
    """The one read of `population.anchor_tiles`' histograms product: the
    predicted-side fields `region_tile_counts` fits against, and the
    observed-side marginals (`N_G_OBS`/`N_KS_OBS`) the young-star and
    cluster-member subtraction below needs -- one open serving both,
    where the retired `population.anchor_observed` stage used to open
    the same file a second time (brief U4)."""
    path = config_module.product_path(config, "population", "anchors", "histograms", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prior.anchor_weights: histograms product missing for region {region!r} at "
            f"{path} -- run `sesnaimpute.population.anchor_tiles` first")
    with h5py.File(path, "r") as f:
        return dict(
            pixels=np.asarray(f["HPX_PIX_512"][:], dtype=np.int64),
            g_edges=np.asarray(f["G_EDGES"][:], dtype=np.float64),
            ks_edges=np.asarray(f["KS_EDGES"][:], dtype=np.float64),
            ks_source=np.asarray(f["KS_SOURCE"][:], dtype=np.int64),
            n_g_pred=np.asarray(f["N_G_PRED"][:], dtype=np.float64),
            n_ks_pred=np.asarray(f["N_KS_PRED"][:], dtype=np.float64),
            n_gk_pred=np.asarray(f["N_GK_PRED"][:], dtype=np.float64),
            n_g_obs=np.asarray(f["N_G_OBS"][:], dtype=np.float64),
            n_ks_obs=np.asarray(f["N_KS_OBS"][:], dtype=np.float64),
            # C30 item 3: the model's per-population colour-cell split
            # (population.anchor_tiles item 2), read alongside everything
            # else this one open already serves.
            cell_ks_edges=np.asarray(f["CELL_KS_EDGES"][:], dtype=np.float64),
            cell_jk_edges=np.asarray(f["CELL_JK_EDGES"][:], dtype=np.float64),
            n_giant_cell=np.asarray(f["N_GIANT_CELL"][:], dtype=np.float64),
            n_dwarf_cell=np.asarray(f["N_DWARF_CELL"][:], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# checks (SPEC_PRIORS.md section 2.3, report only)
# ---------------------------------------------------------------------------

def check_scatter_vs_counting_error(tile_log_ratio, tile_precision, has_evidence):
    """The scatter of the per-tile pooled ratio across tiles against its
    own counting error: `std(tile_log_ratio)` versus the median
    `1/sqrt(precision)` over evidence-carrying tiles."""
    if np.count_nonzero(has_evidence) < 2:
        return float("nan"), float("nan")
    observed_scatter = float(np.std(tile_log_ratio[has_evidence]))
    counting_error = float(np.median(1.0 / np.sqrt(tile_precision[has_evidence])))
    return observed_scatter, counting_error


def check_gaia_vs_2mass(n_obs_g, n_pred_g, n_obs_ks, n_pred_ks):
    """Per tile, Gaia's own pooled ratio against 2MASS's own, and the
    fraction of tiles disagreeing beyond 2 sigma of their combined
    counting error."""
    log_r_g, prec_g = _log10_ratio_and_precision(n_obs_g, n_pred_g)
    log_r_ks, prec_ks = _log10_ratio_and_precision(n_obs_ks, n_pred_ks)
    tile_prec_g = prec_g.sum(axis=1)
    tile_prec_ks = prec_ks.sum(axis=1)
    has_g = tile_prec_g > 0
    has_ks = tile_prec_ks > 0
    both = has_g & has_ks
    if not np.any(both):
        return float("nan")
    pooled_g = (prec_g * log_r_g).sum(axis=1)[both] / tile_prec_g[both]
    pooled_ks = (prec_ks * log_r_ks).sum(axis=1)[both] / tile_prec_ks[both]
    sigma_combined = np.sqrt(1.0 / tile_prec_g[both] + 1.0 / tile_prec_ks[both])
    disagree = np.abs(pooled_g - pooled_ks) > 2.0 * sigma_combined
    return float(np.mean(disagree))


# ---------------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------------

def build_region(config, region, clusters, members_all, w_pool_g, w_pool_ks, ks_pool_lower, ks_pool_upper,
                  transported_ks_slope, cache=None):
    rc = region_tile_counts(config, region, clusters, members_all, cache=cache)
    tiles, hist = rc["tiles"], rc["hist"]
    n_tile = rc["n_tile"]
    n_g, n_ks = hist["g_edges"].size - 1, hist["ks_edges"].size - 1
    n_pred_g, n_pred_ks, n_pred_joint = rc["n_pred_g"], rc["n_pred_ks"], rc["n_pred_joint"]
    n_obs_g, n_obs_ks, n_obs_joint = rc["n_obs_g"], rc["n_obs_ks"], rc["n_obs_joint"]
    excluded, reason, nearest_cluster = rc["excluded"], rc["reason"], rc["nearest_cluster"]
    ratio_mask = rc["ratio_mask"]
    excluded_ratio, excluded_catalog = rc["excluded_ratio"], rc["excluded_catalog"]
    ks_m50, ks_own_fit, p_ks = rc["ks_m50"], rc["ks_own_fit"], rc["p_ks"]
    ks_source = rc["ks_source"]
    measured_ks = rc["measured_ks"]

    # the pool's own Ks axis is the union over all thirty regions' bins
    #; this region's bins take the pool by (lower, upper) edge, not
    # array position, before the fit ever sees it.
    w_pool_ks_region = _pool_at_region_bins(w_pool_ks, ks_pool_lower, ks_pool_upper, hist["ks_edges"])

    fit_g = fit_tile_weights(n_obs_g, n_pred_g, excluded, w_pool=w_pool_g)
    fit_ks = fit_tile_weights(n_obs_ks, n_pred_ks, excluded, w_pool=w_pool_ks_region)

    # C30 item 3: the giant/dwarf split, Ks 12.0-14.3 only. `fit_ks` above
    # is the ordinary, unsplit fit over the WHOLE Ks axis -- computed
    # first, in every case, and never touched below unless `split` is
    # True: the brief's own "bit-identical with the split disabled"
    # identity holds by construction, not a separate code path.
    w_giant, w_giant_err, n_giant_model, split, dwarf5 = fit_population_split(
        config, region, tiles, hist, excluded, n_obs_ks, n_pred_ks)
    if split:
        n_cell_k = dwarf5["n_cell_k"]
        fit_dwarf = fit_tile_weights(dwarf5["n_obs_dwarf"], dwarf5["n_pred_dwarf"], excluded,
                                      w_pool=w_pool_ks_region[:n_cell_k])
        fit_ks["w"][:, :n_cell_k] = fit_dwarf["w"]
        fit_ks["b"][:, :n_cell_k] = fit_dwarf["b"]
        fit_ks["w_region"][:n_cell_k] = fit_dwarf["w_region"]
        fit_ks["ratio"][:, :n_cell_k] = fit_dwarf["ratio"]
        fit_ks["populated"][:n_cell_k] = fit_dwarf["populated"]
        fit_ks["pooled"][:n_cell_k] = fit_dwarf["pooled"]
        n_dwarf_clamped = dwarf5["n_clamped"]
    else:
        n_dwarf_clamped = 0

    n_obs_joint_flat = n_obs_joint.reshape(n_tile, n_g * n_ks)
    n_pred_joint_flat = n_pred_joint.reshape(n_tile, n_g * n_ks)
    # C21 item 1: the joint grid has no SURVEY-wide pooled counterpart
    # (item 1 asks for the region level only), but it DOES now have a
    # REGION-pooled one -- every unexcluded tile's own N_GK_OBS/N_GK_PRED
    # summed first, gated only at the pooled total
    # (`region_pooled_ratio`), filling the all-tiles-masked fallback
    # `fit_tile_weights` already has a slot for. A joint cell unmeasured
    # at BOTH levels is left NOT FINITE exactly as before.
    w_pool_joint_region, _ = region_pooled_ratio(n_obs_joint_flat, n_pred_joint_flat, excluded)
    fit_joint = fit_tile_weights(n_obs_joint_flat, n_pred_joint_flat, excluded, w_pool=w_pool_joint_region)
    w_joint = fit_joint["w"].reshape(n_tile, n_g, n_ks)
    w_region_joint = fit_joint["w_region"].reshape(n_g, n_ks)
    # per-bin source code (item 1): 0 none, 1 at least one tile of this
    # region cleared MIN_COUNTS on its own, 2 no tile did but this
    # region's tiles pooled did (`fit_tile_weights`' own `populated`/
    # `pooled` flags, mutually exclusive by construction).
    joint_tile_fitted = fit_joint["populated"].reshape(n_g, n_ks)
    joint_pool_fitted = fit_joint["pooled"].reshape(n_g, n_ks)
    joint_source = np.where(joint_tile_fitted, 1, np.where(joint_pool_fitted, 2, 0)).astype(np.int8)
    # decision 3, extended (C21 item 1): a TILE-fitted cell is used only
    # where this tile's own observed count is positive too (unchanged --
    # 1-24 crossmatched sources at a single tile, or n_pred < MIN_COUNTS
    # region-wide, still leave it unfitted at that level). A POOL-fitted
    # cell (no single tile cleared the floor, the region's tiles summed
    # did) is read by EVERY tile in the region regardless of its own
    # count: a tile's own zero crossmatch stars there is not evidence
    # against a value the region's own tiles, pooled, DID measure -- the
    # placement branch (star_population.star_weights' rule 5) is then
    # reached only where no joint information exists at any level.
    use_joint = np.where(joint_pool_fitted[None, :, :], True,
                          (n_obs_joint > 0.0) & joint_tile_fitted[None, :, :])

    faint_g = faint_trend_dex_per_mag(fit_g["w_region"], hist["g_edges"], fit_g["populated"])

    # WP-POP-3 (C6 ruling: anchor). This region's OWN deep-sourced
    # slope, measured only on its KS_SOURCE == 1 bins; a region with no
    # such bins, or too few clearing the counting floor, takes the
    # OTHER regions' transported value instead (module docstring).
    faint_ks_own = deep_slope_dex_per_mag(fit_ks["w_region"], hist["ks_edges"],
                                           ks_source, fit_ks["populated"])
    faint_ks_transported = not np.isfinite(faint_ks_own)
    faint_ks = transported_ks_slope if faint_ks_transported else faint_ks_own

    # The synthetic faint-end extension (module docstring): a no-op for
    # a region whose own KS_EDGES already reaches KS_DEEP_CUT_MAG, or
    # when no transported slope is available at all.
    (ks_edges_out, w_ks_out, w_region_ks_out, populated_ks_out, measured_ks_out,
     w_joint_out, use_joint_out, w_region_joint_out, p_ks_out, joint_source_out) = _extend_shallow_ks_axis(
        hist["ks_edges"], fit_ks["w"], fit_ks["w_region"], fit_ks["populated"], measured_ks,
        w_joint, use_joint, w_region_joint, p_ks, joint_source, faint_ks, n_tile,
        target_mag=KS_DEEP_CUT_MAG)
    n_ks_added = int(ks_edges_out.size) - int(hist["ks_edges"].size)

    # acceptance (CODING_RULES.md rule 11): the raw ratio, before
    # shrinkage, reproduces N_OBS_G exactly wherever the marginal is used
    # (USE_JOINT is False) and the observed count clears the pooling floor
    marginal_used = ~use_joint.any(axis=2)  # collapsed onto the G axis below
    usable_g = (n_obs_g >= MIN_COUNTS) & (n_pred_g >= MIN_COUNTS)
    check_mask = usable_g & marginal_used
    raw_ratio_g = np.where(n_pred_g > 0, n_obs_g / np.where(n_pred_g > 0, n_pred_g, 1.0), 0.0)
    identity_dev = np.abs(raw_ratio_g * n_pred_g - n_obs_g)
    max_identity_dev = float(identity_dev[check_mask].max()) if np.any(check_mask) else 0.0

    region_total_ratio_g = n_obs_g.sum(axis=0) / np.where(
        n_pred_g.sum(axis=0) > 0, n_pred_g.sum(axis=0), 1.0)
    pooled_bins = fit_g["populated"]
    if np.any(pooled_bins):
        rel_dev = np.abs(fit_g["w_region"][pooled_bins] - region_total_ratio_g[pooled_bins]) \
            / np.where(region_total_ratio_g[pooled_bins] > 0, region_total_ratio_g[pooled_bins], 1.0)
        max_region_pooled_dev = float(rel_dev.max())
    else:
        max_region_pooled_dev = float("nan")

    scatter_obs, scatter_expected = check_scatter_vs_counting_error(
        ratio_mask["tile_log_ratio"], ratio_mask["tile_precision"], ratio_mask["has_evidence"])
    frac_disagree_2sigma = check_gaia_vs_2mass(n_obs_g, n_pred_g, n_obs_ks, n_pred_ks)

    return dict(
        region=region, n_tile=n_tile,
        g_edges=hist["g_edges"], ks_edges=ks_edges_out,
        n_obs_g=n_obs_g, n_pred_g=n_pred_g, w_g=fit_g["w"], b_g=fit_g["b"], w_region_g=fit_g["w_region"],
        populated_g=fit_g["populated"], pooled_g=fit_g["pooled"],
        n_obs_ks=n_obs_ks, n_pred_ks=n_pred_ks, w_ks=w_ks_out, b_ks=fit_ks["b"], w_region_ks=w_region_ks_out,
        populated_ks=populated_ks_out, pooled_ks=fit_ks["pooled"], measured_ks=measured_ks_out,
        n_obs_joint=n_obs_joint, n_pred_joint=n_pred_joint, w_joint=w_joint_out,
        w_region_joint=w_region_joint_out, use_joint=use_joint_out, joint_source=joint_source_out,
        n_joint_tile_fitted=int(np.count_nonzero(joint_tile_fitted)),
        n_joint_pool_fitted=int(np.count_nonzero(joint_pool_fitted)),
        n_joint_bins=int(joint_tile_fitted.size),
        excluded=excluded, reason=reason, nearest_cluster=nearest_cluster,
        n_excluded_ratio=int(np.count_nonzero(excluded_ratio)),
        # the old rule's tile count, for the report only (module
        # docstring): no longer subtracted into `excluded`.
        n_previously_catalog_excluded=int(np.count_nonzero(excluded_catalog)),
        n_freed_by_subtraction=int(np.count_nonzero(excluded_catalog & ~excluded_ratio)),
        clusters_hit=sorted(set(nearest_cluster[excluded_catalog].tolist())),
        faint_trend_g=faint_g, faint_trend_ks=faint_ks,
        faint_trend_ks_transported=bool(faint_ks_transported), n_ks_added=n_ks_added,
        max_identity_dev=max_identity_dev, n_identity_checked=int(np.count_nonzero(check_mask)),
        max_region_pooled_dev=max_region_pooled_dev,
        scatter_observed=scatter_obs, scatter_expected=scatter_expected,
        frac_disagree_2sigma=frac_disagree_2sigma,
        ks_m50=ks_m50, ks_own_fit=ks_own_fit, p_ks=p_ks_out, ks_source=ks_source,
        w_giant=w_giant, w_giant_err=w_giant_err, n_giant_model=n_giant_model,
        split=bool(split), n_dwarf_clamped=n_dwarf_clamped,
    )


def _write_product(config, region, result):
    path = config_module.product_path(config, "population", "anchors", "weights", "tile", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "tile"
        f.attrs["FAINT_TREND_G_DEX_PER_MAG"] = result["faint_trend_g"]
        f.attrs["FAINT_TREND_KS_DEX_PER_MAG"] = result["faint_trend_ks"]
        # WP-POP-3 (C6 ruling: anchor): True where this region has no
        # deep-sourced evidence of its own and FAINT_TREND_KS_DEX_PER_MAG
        # is therefore the OTHER regions' transported slope (module
        # docstring, "Faint end, Ks axis"), not this region's own fit.
        f.attrs["FAINT_TREND_KS_TRANSPORTED"] = result["faint_trend_ks_transported"]
        # item 3: POPULATED_G/POPULATED_KS are the explicit per-bin flag,
        # replacing the `w_region != 1.0` sentinel -- this region's own
        # tile evidence, whether or not the bin fell back to the
        # survey-pooled value (item 1). W13/W38: P_KS/KS_M50/KS_SCALE are
        # the 2MASS anchor's own per-tile completeness sigmoid; the width
        # is now one fixed instrument value, not a per-tile fit.
        for name, data in (
            ("G_EDGES", result["g_edges"]),
            ("KS_EDGES", result["ks_edges"]),
            ("W_G", result["w_g"]),
            ("W_REGION_G", result["w_region_g"]),
            ("POPULATED_G", result["populated_g"]),
            ("W_KS", result["w_ks"]),
            ("W_REGION_KS", result["w_region_ks"]),
            ("POPULATED_KS", result["populated_ks"]),
            ("MEASURED_KS", result["measured_ks"].astype(np.bool_)),
            ("W_JOINT", result["w_joint"]),
            ("USE_JOINT", result["use_joint"]),
            ("W_REGION_JOINT", result["w_region_joint"]),
            ("JOINT_SOURCE", result["joint_source"]),
            ("EXCLUDED", result["excluded"]),
            ("P_KS", result["p_ks"]),
            ("KS_M50", result["ks_m50"]),
            ("KS_SCALE", np.float64(KS_SCALE_MAG)),
            # C30 item 3: the per-population (giant/dwarf) split at the
            # 2MASS colour-cell range -- one value per region, SPLIT False
            # (W_GIANT/W_GIANT_ERR NaN) where the region's own giant model
            # count never clears N_GIANT_MIN, in which case a giant reads
            # its bin's W_KS exactly like a dwarf (module docstring above
            # `fit_population_split`).
            ("W_GIANT", np.float64(result["w_giant"])),
            ("W_GIANT_ERR", np.float64(result["w_giant_err"])),
            ("N_GIANT_MODEL", np.float64(result["n_giant_model"])),
            ("SPLIT", np.bool_(result["split"])),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])
    return path


def build(config, regions=None):
    """Writes, per region (default: all thirty), `bms/anchors/weights_
    anchors_tile__<Region>.hdf5` (module docstring). Prints, per region,
    the tile count, the excluded-tile counts by reason, the pooled Gaia
    and 2MASS weights, the tile-weight range, the faint-end slopes, and
    the two section 2.3 checks.
    """
    from sesnaimpute.population import anchor_observed

    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    clusters = _read_hunt_reffert_clusters(config)

    # item 1: the survey pool, over ALL thirty regions, computed once,
    # before any per-region fit -- every region (even one being rebuilt
    # alone) needs the same pooled fallback, not a pool of itself.
    all_region_names = [r.name for r in regions_module.REGIONS]

    # brief U4: the Hunt & Reffert 2023 member table (1.29 M rows) is
    # read ONCE here, for the UNION of every one-of-thirty region's own
    # candidate clusters (the survey pool pass below always touches all
    # thirty regardless of `regions`), rather than once per region as
    # the retired `population.anchor_observed` stage did (11-12 s each,
    # every region, tiny or huge).
    all_touching_names = set()
    for region in all_region_names:
        try:
            tiles_r = _read_tiles(config, region)
        except FileNotFoundError as exc:
            # the same rule as the survey pool below: a region whose tiles
            # are absent is named and left out, never a stop for the rest
            print("population.anchor_weights: cluster scan skipping %r (%s)" % (region, exc))
            continue
        touching_r = anchor_observed._clusters_touching_region(
            tiles_r["tile_l_deg"], tiles_r["tile_b_deg"], tiles_r["l_star_deg"], clusters)
        all_touching_names.update(clusters["name"][touching_r].tolist())
    members_all = anchor_observed._read_cluster_members(config, sorted(all_touching_names))
    print("prior.anchor_weights: Hunt & Reffert member table read once for %d candidate "
          "cluster(s) (survey-wide union) -> %d member rows"
          % (len(all_touching_names), members_all["glon"].size))

    # young_stars.build_region/anchor_observed.compute_region run at
    # most once per region per `build()` call: the survey pool pass
    # below and the per-region fit pass further down both call
    # `region_tile_counts` for every requested region, and this cache
    # makes the second call a lookup, not a second computation.
    cache = {}

    (w_pool_g, w_pool_ks, pool_g_edges, ks_pool_lower, ks_pool_upper, pool_skipped, n_pool_ks,
     transported_ks_slope, deep_slope_regions) = survey_pooled_weights(
        config, all_region_names, clusters, members_all, cache=cache)
    # the pool pass held every region's own tile counts; only the regions
    # this call fits need them again (one region's working set at a time)
    for region in list(cache):
        if region not in region_names:
            del cache[region]
    print(
        "prior.anchor_weights: survey pool over %d regions (skipped %s): "
        "W_POOL_G=[%.3f,%.3f] W_POOL_KS=[%.3f,%.3f] "
        "ks_bins(lower)=%s n_regions_per_ks_bin=%s"
        % (len(all_region_names) - len(pool_skipped), pool_skipped or "[]",
           float(np.nanmin(w_pool_g)), float(np.nanmax(w_pool_g)),
           float(np.nanmin(w_pool_ks)), float(np.nanmax(w_pool_ks)),
           ks_pool_lower.tolist(), n_pool_ks.tolist()))
    # WP-POP-3 (C6 ruling: anchor): the ONE survey-wide Ks faint-end
    # slope transported to every region with no deep coverage of its
    # own, and exactly which regions' own deep (UKIDSS GPS or VVV)
    # evidence it was measured from.
    print(
        "prior.anchor_weights: C6 transported Ks faint-end slope=%.4f dex/mag "
        "from %d deep-covered region(s): %s"
        % (transported_ks_slope, len(deep_slope_regions), deep_slope_regions))

    for region in region_names:
        with progress.Stage("population.anchor_weights", region) as st:
            result = build_region(config, region, clusters, members_all, w_pool_g, w_pool_ks,
                                   ks_pool_lower, ks_pool_upper, transported_ks_slope, cache=cache)
            path = _write_product(config, region, result)
            st.done(path, n_tile=result["n_tile"], max_identity_dev=result["max_identity_dev"])

        # C21 item 1: how many of the joint (G, Ks) cells carry evidence
        # at each level -- a tile of this region on its own, this
        # region's tiles pooled, or neither (where star_population's
        # placement rule, item 2, is actually reached).
        print(
            "prior.anchor_weights: %s JOINT_SOURCE over %d cells: tile-fitted=%d "
            "pool-fitted=%d none=%d"
            % (region, result["n_joint_bins"], result["n_joint_tile_fitted"],
               result["n_joint_pool_fitted"],
               result["n_joint_bins"] - result["n_joint_tile_fitted"] - result["n_joint_pool_fitted"]))

        w_all = np.concatenate([result["w_g"].ravel(), result["w_ks"].ravel()])
        print(
            "prior.anchor_weights: %s tiles=%d excluded(ratio=%d) "
            "previously_catalog_excluded=%d freed_by_subtraction=%d clusters=%s "
            "W_region_G=[%.3f,%.3f] W_region_Ks=[%.3f,%.3f] W_range=[%.3f,%.3f] "
            "faint_slope_G=%.3f faint_slope_Ks=%.3f (transported=%s, ks_bins_added=%d) "
            "identity_max_dev=%.2e (n=%d) "
            "region_pooled_max_reldev=%.4f scatter_obs=%.3f scatter_expected=%.3f "
            "gaia_2mass_disagree_frac=%.3f -> %s"
            % (region, result["n_tile"], result["n_excluded_ratio"],
               result["n_previously_catalog_excluded"], result["n_freed_by_subtraction"],
               result["clusters_hit"] or "[]",
               float(result["w_region_g"].min()), float(result["w_region_g"].max()),
               float(result["w_region_ks"].min()), float(result["w_region_ks"].max()),
               float(w_all.min()), float(w_all.max()),
               result["faint_trend_g"], result["faint_trend_ks"],
               result["faint_trend_ks_transported"], result["n_ks_added"],
               result["max_identity_dev"], result["n_identity_checked"],
               result["max_region_pooled_dev"], result["scatter_observed"],
               result["scatter_expected"], result["frac_disagree_2sigma"], path))
        n_at_bound = int(np.count_nonzero(
            np.isclose(result["ks_m50"], KS_M50_HI_MAG, atol=1e-6)))
        ks_source = result["ks_source"]
        print(
            "anchor_weights: %s KS_SOURCE=%s (%d/%d bins UKIDSS) ks_m50 median=%.3f "
            "range=%.3f-%.3f ks_scale(fixed)=%.4f tiles_at_complete_bound=%d/%d"
            % (region, ks_source.tolist(), int(ks_source.sum()), ks_source.size,
               float(np.median(result["ks_m50"])),
               float(result["ks_m50"].min()), float(result["ks_m50"].max()),
               KS_SCALE_MAG, n_at_bound, result["n_tile"]))
        # C30 item 3: the giant/dwarf split -- W_GIANT a single region
        # scalar, W_DWARF the (now possibly giant-corrected) first
        # CELL_KS_EDGES bins of W_REGION_KS.
        n_cell_k = twomass_counts_module.CELL_KS_EDGES.size - 1
        w_dwarf_cells = result["w_region_ks"][:n_cell_k]
        print(
            "anchor_weights: %s C30 SPLIT=%s N_GIANT_MODEL=%.1f (floor %.0f) W_GIANT=%.3f+-%.3f "
            "W_DWARF(Ks 12.0-14.3)=%s n_dwarf_clamped_to_zero=%d"
            % (region, result["split"], result["n_giant_model"], N_GIANT_MIN,
               result["w_giant"], result["w_giant_err"],
               np.round(w_dwarf_cells, 3).tolist(), result["n_dwarf_clamped"]))


if __name__ == "__main__":
    run(build)
