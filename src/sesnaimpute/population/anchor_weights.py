"""The STAR per-tile-and-bin anchor weight `W` (SPEC_PRIORS.md section
2.1, table rows "W", "cluster exclusion", "faint end"; reading note
04_star_family.md section B, `fit_tile_weights`, `shrink_log_normal`,
`cluster_excluded_tiles`), from the tile geometry, the predicted anchor
histograms, and the young-star-subtracted observed anchor histograms
`population.anchor_tiles` and `population.anchor_observed` already wrote.

    W(t, bin) = N_obs(t, bin) / N_pred(t, bin)

fitted separately on the Gaia `G` marginal, the 2MASS `Ks` marginal, and
the joint `(G, Ks)` grid where the SESNA-Gaia crossmatch supplies both
magnitudes for a source (section 2.1, "joint and marginal bins" -- a
joint bin with a nonzero observed count is populated and used; elsewhere
the marginal weight applies). Each axis is fitted the SAME way: the
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

Faint end (decision 5): the tables stop at `G_EDGES`/`KS_EDGES`' own
faint edges (the anchors' own limits); the log10 trend of the region-
pooled weight across each anchor's faintest three POPULATED bins (bins
some tile actually measured) is fit by least squares and stored as a
root attribute, the disclosed uncertainty on the faint-end extrapolation.

Product, per region, `bms/anchors/weights_anchors_tile__<Region>.hdf5`:
`W_JOINT`/`USE_JOINT` (n_tile, n_G, n_Ks), `W_G` (n_tile, n_G), `W_KS`
(n_tile, n_Ks), `W_REGION_JOINT` (n_G, n_Ks), `W_REGION_G` (n_G),
`W_REGION_KS` (n_Ks), `EXCLUDED` (n_tile, bool), `G_EDGES`, `KS_EDGES`,
`P_KS` (n_tile, n_Ks), `KS_M50` (n_tile), `KS_SCALE` (scalar -- the one
fixed instrument width every tile fits with, W38); root attrs
`GRANULE="tile"`, `FAINT_TREND_G_DEX_PER_MAG`, `FAINT_TREND_KS_DEX_PER_MAG`.

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
bin no tile of this region measures, never written separately. W48: the
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

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run

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

#: W13/W38 (coordinator's refit, 2026-09-09): the per-tile 2MASS
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
    uncalibrated TRILEGAL level. `w_pool` is `None` only when a caller
    is computing the pool itself and has none yet to fall back on; in
    that one case the bin is dropped from the pool sum (see
    `survey_pooled_weights`), not set to unity.

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
            continue
        populated[k] = True
        pooled_fit = shrink_log_normal(ratio[:, k], sigma[:, k], mask=mask_k)
        w[:, k] = pooled_fit["f_rt"]
        b[:, k] = pooled_fit["shrinkage_weight"]
        w_region[k] = pooled_fit["mu_factor"]
    return dict(w=w, b=b, w_region=w_region, ratio=ratio, populated=populated,
                pooled=pooled)


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

def region_tile_counts(config, region, clusters, min_counts=MIN_COUNTS):
    """One region's tile-level predicted/observed counts and exclusion
    mask, with no weight fit performed -- the read-and-aggregate half of
    `build_region`, split out so `survey_pooled_weights` (the once-over-
    all-thirty-regions pass, item 1) and the per-region fit share the
    same counts rather than reading the on-disk products twice with two
    slightly different code paths."""
    tiles = _read_tiles(config, region)
    hist = _read_histograms(config, region)
    obs = _read_observed(config, region)

    if not (np.array_equal(tiles["pixels"], hist["pixels"])
            and np.array_equal(tiles["pixels"], obs["pixels"])):
        raise ValueError(
            f"prior.anchor_weights: {region!r}'s tiles/histograms/observed products "
            "disagree on their own pixel set -- rerun `prior.anchor_tiles` and "
            "`prior.anchor_observed` for this region")
    if not (np.array_equal(hist["g_edges"], obs["g_edges"])
            and np.array_equal(hist["ks_edges"], obs["ks_edges"])):
        raise ValueError(
            f"prior.anchor_weights: {region!r}'s histograms/observed products carry "
            "different G_EDGES/KS_EDGES -- rerun `prior.anchor_tiles` and "
            "`prior.anchor_observed` for this region")

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


def survey_pooled_weights(config, region_names, clusters, min_counts=MIN_COUNTS):
    """The SURVEY-POOLED weight per magnitude bin, on the `G` and `Ks`
    marginals (owner ruling 2026-09-06, item 1): sum the observed and
    the predicted counts over EVERY region's own populated tiles (not
    cluster-excluded, both counts clearing `min_counts`) and take one
    ratio per bin -- computed once over all thirty regions' stored
    anchor products, never per region.

    The Gaia axis is shared by every region (one `G_EDGES` grid) and
    pools by array position. The Ks axis is not (W46: fifteen regions
    reach 17.0, fifteen reach 2MASS's own 14.3) -- each Ks bin is keyed
    by its own `(lower, upper)` edge and summed over exactly the regions
    whose axis carries it (`_pool_bins_by_edge`), never by array
    position, so no region is skipped for its Ks axis length. `skipped`
    now names only a region whose upstream anchor product is missing.

    Returns `(w_pool_g, w_pool_ks, g_edges, ks_lower, ks_upper, skipped,
    n_pool_ks)`, `n_pool_ks` the count of regions carrying each returned
    Ks bin.
    """
    sum_obs_g = sum_pred_g = None
    g_edges = None
    ks_lower_list, ks_upper_list, ks_obs_list, ks_pred_list = [], [], [], []
    skipped = []
    for region in region_names:
        try:
            rc = region_tile_counts(config, region, clusters, min_counts)
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
        ks_lower_list.append(ks_edges_r[:-1])
        ks_upper_list.append(ks_edges_r[1:])
        ks_obs_list.append(np.where(usable_ks, rc["n_obs_ks"], 0.0).sum(axis=0))
        ks_pred_list.append(np.where(usable_ks, rc["n_pred_ks"], 0.0).sum(axis=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        w_pool_g = np.where(sum_pred_g > 0,
                             sum_obs_g / np.where(sum_pred_g > 0, sum_pred_g, 1.0), np.nan)

    ks_lower, ks_upper, n_pool_ks, (sum_obs_ks, sum_pred_ks) = _pool_bins_by_edge(
        ks_lower_list, ks_upper_list, [ks_obs_list, ks_pred_list])
    with np.errstate(divide="ignore", invalid="ignore"):
        w_pool_ks = np.where(sum_pred_ks > 0,
                              sum_obs_ks / np.where(sum_pred_ks > 0, sum_pred_ks, 1.0), np.nan)
    return w_pool_g, w_pool_ks, g_edges, ks_lower, ks_upper, skipped, n_pool_ks


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
        )


def _read_observed(config, region):
    path = config_module.product_path(config, "population", "anchors", "observed", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prior.anchor_weights: observed anchors missing for region {region!r} at "
            f"{path} -- run `sesnaimpute.population.anchor_observed` first")
    with h5py.File(path, "r") as f:
        return dict(
            pixels=np.asarray(f["HPX_PIX_512"][:], dtype=np.int64),
            g_edges=np.asarray(f["G_EDGES"][:], dtype=np.float64),
            ks_edges=np.asarray(f["KS_EDGES"][:], dtype=np.float64),
            n_g_sub=np.asarray(f["N_G_SUB"][:], dtype=np.float64),
            n_ks_sub=np.asarray(f["N_KS_SUB"][:], dtype=np.float64),
            n_gk_sub=np.asarray(f["N_GK_SUB"][:], dtype=np.float64),
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

def build_region(config, region, clusters, w_pool_g, w_pool_ks, ks_pool_lower, ks_pool_upper):
    rc = region_tile_counts(config, region, clusters)
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

    n_obs_joint_flat = n_obs_joint.reshape(n_tile, n_g * n_ks)
    n_pred_joint_flat = n_pred_joint.reshape(n_tile, n_g * n_ks)
    # the joint grid has no survey-pooled counterpart (item 1 asks for the
    # G/Ks marginals only); an all-masked joint bin stays at unity, as
    # before -- the marginal weights are what a star actually falls back
    # to (star_population.star_weights), the joint table is a refinement
    # only where a real crossmatch populates it.
    fit_joint = fit_tile_weights(n_obs_joint_flat, n_pred_joint_flat, excluded)
    w_joint = fit_joint["w"].reshape(n_tile, n_g, n_ks)
    w_region_joint = fit_joint["w_region"].reshape(n_g, n_ks)
    use_joint = n_obs_joint > 0.0  # decision 3: the joint bin is populated

    faint_g = faint_trend_dex_per_mag(fit_g["w_region"], hist["g_edges"], fit_g["populated"])
    faint_ks = faint_trend_dex_per_mag(fit_ks["w_region"], hist["ks_edges"], fit_ks["populated"])

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
        g_edges=hist["g_edges"], ks_edges=hist["ks_edges"],
        n_obs_g=n_obs_g, n_pred_g=n_pred_g, w_g=fit_g["w"], b_g=fit_g["b"], w_region_g=fit_g["w_region"],
        populated_g=fit_g["populated"], pooled_g=fit_g["pooled"],
        n_obs_ks=n_obs_ks, n_pred_ks=n_pred_ks, w_ks=fit_ks["w"], b_ks=fit_ks["b"], w_region_ks=fit_ks["w_region"],
        populated_ks=fit_ks["populated"], pooled_ks=fit_ks["pooled"], measured_ks=measured_ks,
        n_obs_joint=n_obs_joint, n_pred_joint=n_pred_joint, w_joint=w_joint,
        w_region_joint=w_region_joint, use_joint=use_joint,
        excluded=excluded, reason=reason, nearest_cluster=nearest_cluster,
        n_excluded_ratio=int(np.count_nonzero(excluded_ratio)),
        # the old rule's tile count, for the report only (module
        # docstring): no longer subtracted into `excluded`.
        n_previously_catalog_excluded=int(np.count_nonzero(excluded_catalog)),
        n_freed_by_subtraction=int(np.count_nonzero(excluded_catalog & ~excluded_ratio)),
        clusters_hit=sorted(set(nearest_cluster[excluded_catalog].tolist())),
        faint_trend_g=faint_g, faint_trend_ks=faint_ks,
        max_identity_dev=max_identity_dev, n_identity_checked=int(np.count_nonzero(check_mask)),
        max_region_pooled_dev=max_region_pooled_dev,
        scatter_observed=scatter_obs, scatter_expected=scatter_expected,
        frac_disagree_2sigma=frac_disagree_2sigma,
        ks_m50=ks_m50, ks_own_fit=ks_own_fit, p_ks=p_ks, ks_source=ks_source,
    )


def _write_product(config, region, result):
    path = config_module.product_path(config, "population", "anchors", "weights", "tile", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "tile"
        f.attrs["FAINT_TREND_G_DEX_PER_MAG"] = result["faint_trend_g"]
        f.attrs["FAINT_TREND_KS_DEX_PER_MAG"] = result["faint_trend_ks"]
        f.create_dataset("G_EDGES", data=result["g_edges"])
        f.create_dataset("KS_EDGES", data=result["ks_edges"])
        f.create_dataset("W_G", data=result["w_g"])
        f.create_dataset("W_REGION_G", data=result["w_region_g"])
        # item 3: the explicit per-bin flag, replacing the `w_region !=
        # 1.0` sentinel -- this region's own tile evidence, whether or
        # not the bin fell back to the survey-pooled value (item 1).
        f.create_dataset("POPULATED_G", data=result["populated_g"])
        f.create_dataset("W_KS", data=result["w_ks"])
        f.create_dataset("W_REGION_KS", data=result["w_region_ks"])
        f.create_dataset("POPULATED_KS", data=result["populated_ks"])
        f.create_dataset("MEASURED_KS", data=result["measured_ks"].astype(np.bool_))
        f.create_dataset("W_JOINT", data=result["w_joint"])
        f.create_dataset("USE_JOINT", data=result["use_joint"])
        f.create_dataset("W_REGION_JOINT", data=result["w_region_joint"])
        f.create_dataset("EXCLUDED", data=result["excluded"])
        # W13/W38: the 2MASS anchor's own per-tile completeness sigmoid;
        # the width is now one fixed instrument value, not a per-tile fit.
        f.create_dataset("P_KS", data=result["p_ks"])
        f.create_dataset("KS_M50", data=result["ks_m50"])
        f.create_dataset("KS_SCALE", data=np.float64(KS_SCALE_MAG))
    return path


def build(config, regions=None):
    """Writes, per region (default: all thirty), `bms/anchors/weights_
    anchors_tile__<Region>.hdf5` (module docstring). Prints, per region,
    the tile count, the excluded-tile counts by reason, the pooled Gaia
    and 2MASS weights, the tile-weight range, the faint-end slopes, and
    the two section 2.3 checks.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    clusters = _read_hunt_reffert_clusters(config)

    # item 1: the survey pool, over ALL thirty regions, computed once,
    # before any per-region fit -- every region (even one being rebuilt
    # alone) needs the same pooled fallback, not a pool of itself.
    all_region_names = [r.name for r in regions_module.REGIONS]
    w_pool_g, w_pool_ks, pool_g_edges, ks_pool_lower, ks_pool_upper, pool_skipped, n_pool_ks = \
        survey_pooled_weights(config, all_region_names, clusters)
    print(
        "prior.anchor_weights: survey pool over %d regions (skipped %s): "
        "W_POOL_G=[%.3f,%.3f] W_POOL_KS=[%.3f,%.3f] "
        "ks_bins(lower)=%s n_regions_per_ks_bin=%s"
        % (len(all_region_names) - len(pool_skipped), pool_skipped or "[]",
           float(np.nanmin(w_pool_g)), float(np.nanmax(w_pool_g)),
           float(np.nanmin(w_pool_ks)), float(np.nanmax(w_pool_ks)),
           ks_pool_lower.tolist(), n_pool_ks.tolist()))

    for region in region_names:
        with progress.Stage("prior.anchor_weights", region) as st:
            result = build_region(config, region, clusters, w_pool_g, w_pool_ks,
                                   ks_pool_lower, ks_pool_upper)
            path = _write_product(config, region, result)
            st.done(path, n_tile=result["n_tile"], max_identity_dev=result["max_identity_dev"])

        w_all = np.concatenate([result["w_g"].ravel(), result["w_ks"].ravel()])
        print(
            "prior.anchor_weights: %s tiles=%d excluded(ratio=%d) "
            "previously_catalog_excluded=%d freed_by_subtraction=%d clusters=%s "
            "W_region_G=[%.3f,%.3f] W_region_Ks=[%.3f,%.3f] W_range=[%.3f,%.3f] "
            "faint_slope_G=%.3f faint_slope_Ks=%.3f identity_max_dev=%.2e (n=%d) "
            "region_pooled_max_reldev=%.4f scatter_obs=%.3f scatter_expected=%.3f "
            "gaia_2mass_disagree_frac=%.3f -> %s"
            % (region, result["n_tile"], result["n_excluded_ratio"],
               result["n_previously_catalog_excluded"], result["n_freed_by_subtraction"],
               result["clusters_hit"] or "[]",
               float(result["w_region_g"].min()), float(result["w_region_g"].max()),
               float(result["w_region_ks"].min()), float(result["w_region_ks"].max()),
               float(w_all.min()), float(w_all.max()),
               result["faint_trend_g"], result["faint_trend_ks"],
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


if __name__ == "__main__":
    run(build)
