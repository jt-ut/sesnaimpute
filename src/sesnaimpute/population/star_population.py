"""Per-tile field-star placement and anchor weight (SPEC_PRIORS.md section
1.4's `u`, section 1.5, section 2.1's `star_reweight_factor` and "faint
end", section 2.2's shape population `a_i = A_s u_i` and "Per tile";
reading note 04_star_family.md section B's `star_reweight_factor`;
IMPLEMENTATION.md section 6, stage 9).

This is the placement-and-weight half of the STAR/AGB/PAHC family module:
where every retained field star (`population.field_stars`) sits in extinction
on a tile's own line of sight, and what per-star anchor weight
(`population.anchor_weights`) it carries there. The partition into STAR/AGB/
PAHC, the brightness unit `B`, and PAHC's contamination weight are a
second, later module.

Placement (spec section 2.2, "Per tile"). A field star carries a distance
but no sky position of its own (`population.field_stars`'s one TRILEGAL
pointing per region); a tile's own extinction profile is therefore
approximated by ONE mean profile: the source-weighted mean, over the
tile's own distinct nside-256 sightlines, of that sightline's own
`u(d) = a_of_d(d, pix, A_pix) / A_pix` (`sky.derived.profile.read(...)`),
on one shared distance grid (the profile product's own `DIST_PC` plus a
log-spaced tail from the map's edge to 10 kpc, spec section 1.4's far-
field tail). Every retained field star's `u_i` is then read off that ONE
mean profile at its own distance; `a_i = A_tile * u_i`, `A_tile` the mean
adopted column of the tile's own sources. One tile, one mean profile, one
column: the approximation this module makes.

Per-star weight (spec section 2.1's `star_reweight_factor`). Every
retained field star, placed at this tile's own `a_i`, predicts its own
observed `(G_obs, Ks_obs)` (`anchor_tiles.magnitudes_at_extinction`, the
project's one place a local column becomes the two anchor magnitudes);
the tile's own `W_JOINT` table (`population.anchor_weights`) supplies the
weight where the bin is populated by a real crossmatch (`USE_JOINT`),
else the `G` or `Ks` marginal the star's own predicted magnitude falls
inside. Where BOTH marginals apply and the joint bin does not, the star's
own PLACEMENT decides which one anchor stands in (owner ruling
2026-09-06, item 2, replacing the geometric mean this module used to
cite to a §2.1 phrase that is not there): a star in front of the cloud
(its own `u = A(d)/A(inf)` below `u_front`, this tile's mean profile
read at the cloud's own measured near edge, `profile.
cloud_front_edge_pc`) takes the Gaia weight at its own magnitude, a
star behind the cloud takes the 2MASS weight, and where that side's own
bin carries no finite weight at all -- unmeasured even after
`population.anchor_weights`' survey-pooled fallback -- the star takes the
OTHER anchor's weight instead. Recorded under its own `WEIGHT_RULE` code,
5, beyond the five the brief names, so the per-rule star counts this
module reports add up honestly. A star beyond both anchors' faint (or
both anchors' bright) edges reads its own tile's faintest (brightest)
POPULATED bin -- populated in the EXPLICIT per-bin sense
(`POPULATED_G`/`POPULATED_KS`, `population.anchor_weights.fit_tile_weights`'
own evidence flag, item 3) -- under the same joint-or-placement rule;
where the two axes disagree (one out on the faint side, the other on the
bright side -- a physically rare, intrinsically very red or very blue
star), the star is reported faint-end by priority, but each axis still
reads its own out-of-range direction's populated edge bin. A cluster-
excluded tile (`EXCLUDED`) reads its weight from the region-pooled table
(`W_REGION_*`) rather than its own tile row; `population.anchor_weights`' own
shrinkage already collapses an excluded tile's stored row to exactly this
value, so reading `W_REGION_*` directly makes that fact part of the code
a reader sees rather than something they have to trace through the
upstream build to learn.

WEIGHT_RULE codes, per star: 0 joint, 1 G marginal only, 2 Ks marginal
only, 3 faint end, 4 bright end, 5 both marginals (placement-selected:
front of the cloud reads Gaia, behind reads 2MASS).

Partition (spec section 3, "AGB"). Every retained field star is evolved
or not by one HR-diagram cut on TRILEGAL's own raw columns (`log g < 1,
log T_e < 3.6, log L > 3`, `evolved_selector`); an evolved star's tile
weight splits `w_AGB = F_dusty * w`, `w_STAR = w - w_AGB`, row by row, so
`w_STAR + w_AGB == w` exactly (reading note 04_star_family.md section C,
`partition_weights`). TRILEGAL carries no chemistry per star, so
`F_dusty` is the weighted mean over the two chemistries measured once,
survey-wide, off Riebel et al. (2012, ApJ 753, 71) per-star GRAMS fits
against the curated GRAMS library's own optical-depth floor
(`f_dusty_by_chemistry`, spec section 10 item 2); `f_C = 0.18` (Le Bertre
et al. 2003) is the fixed carbon-fraction weight.

Brightness units (C3, spec section 2.2, section 3, section 4). Every
star, evolved or not, carries `LOG10_B` (STAR): the median over the
eight bands of its own TRILEGAL flux over its matched atmosphere
template's reference flux (`population.field_stars`' own `TEMPLATE_INDEX`
into `sed_models/registers/sps_register.hdf5`) -- a unit change only
(C3). `LOG10_B_PAHC` is the same median, J/H/Ks only, against the PAHC
library's own continuum reference (`sed_models/registers/
pahc_register.hdf5`'s `library/pahc_fstar` table, nearest of its eleven
`T_EFF` nodes). An evolved star additionally carries `LOG10_B_AGB_C` and
`LOG10_B_AGB_O` (spec section 3's closed forms, `b_agb`); NaN for a
non-evolved star.

PAHC weights (spec section 4). Every retained star (the whole
population, `w` unreduced) carries `P_PAHC`, one nebular-contamination
probability per node of a small, region-wide grid of eight 8 micron
completeness limits (the 5/20/35/50/65/80/95/99th percentiles of the
region's own sources' 50%-completeness limit at I4, `catalog.limits.
limits`): `q = F_lim,8 / (F_i(8um) dimmed by the star's own tile
extinction)`, `P_PAHC = P(q)` from the measured curve
(`population.pahc_curve.read`).

Writes, per region, `bms/star/population_star_tile__<Region>.hdf5`: root
attrs `GRANULE="tile"`, `OMEGA_SIM_DEG2`, `F_DUSTY_O`, `F_DUSTY_C`,
`F_DUSTY_MEAN`, `F_C`; a `LIMIT8_GRID_MJY` (8,) dataset, the region's PAHC
limit grid; one HDF5 group `tile_<id>` per tile, attr
`OMEGA_POINTING_DEG2`, with datasets `STAR_INDEX` (row into
`population.field_stars`' retained group), `U`, `W`, `W_STAR`, `W_AGB`,
`IS_EVOLVED`, `LOG10_B`, `LOG10_B_PAHC`, `LOG10_B_AGB_C`, `LOG10_B_AGB_O`,
`P_PAHC` (n_star, 8). The population stores each
star's placement fraction `U` alone (owner, 2026-09-06): a star's own
extinction is `A_s * U`, `A_s` a real source's own adopted column, formed
at read time by the consumer that has a source to apply it to; this
module never stores a tile-mean extinction. The tile's own mean column
(`a_tile`, computed below) is still used internally, unstored, to place
each simulated star's predicted magnitude for the anchor-weight lookup
and the PAHC contamination weight -- the same per-tile approximation the
module docstring above already describes for `U` itself.
"""

import os
import threading

import h5py
import healpy as hp
import numpy as np
import pandas as pd
from astropy.io import fits
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.population import anchor_tiles, pahc_curve, selection
from sesnaimpute.sky.derived import profile as profile_module
from sesnaimpute.sky.download.trilegal import build as trilegal_download

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: The shared distance grid's far tail (module docstring, spec 1.4's tail):
#: the profile product's own map edge falls well short of TRILEGAL's
#: simulated depth, so the mean profile is extended by the profile
#: reader's own analytic far-field tail, sampled out to 10 kpc -- past
#: every region's own cloud distance and simulated population.
TAIL_MAX_PC = 10_000.0

#: Log-spaced tail node count: a resolution choice, not a physical
#: constant. The tail is one smooth analytic shape (an exponential disc
#: or vertical model, `sky.derived.profile`), so a modest sampling between
#: the map edge and `TAIL_MAX_PC` suffices.
TAIL_N_NODES = 64

#: WEIGHT_RULE codes (module docstring).
WEIGHT_RULE_JOINT = 0
WEIGHT_RULE_G_MARGINAL = 1
WEIGHT_RULE_KS_MARGINAL = 2
WEIGHT_RULE_FAINT_END = 3
WEIGHT_RULE_BRIGHT_END = 4
WEIGHT_RULE_BOTH_MARGINAL = 5
N_WEIGHT_RULES = 6

#: The project's own eight census bands, in the order every register and
#: `catalog.limits` share (module docstring, "Brightness units").
BAND_KEYS = tuple(b.key for b in definitions.BANDS)
IDX_I4 = BAND_KEYS.index("I4")
IDX_JHK = tuple(BAND_KEYS.index(k) for k in ("J", "H", "Ks"))

#: Evolved-star HR-diagram cut on TRILEGAL's own raw columns (SPEC_PRIORS.md
#: section 3): "evolved stars are selected by an HR-diagram cut ... log g
#: < 1, log T_e < 3.6, log L > 3".
EVOLVED_LOGG_MAX = 1.0
EVOLVED_LOGTE_MAX = 3.6
EVOLVED_LOGL_MIN = 3.0

#: Carbon fraction among dusty evolved stars (SPEC_PRIORS.md section 3
#: table): Le Bertre et al. 2003, A&A 403, 943 (126/689 carbon stars in
#: their sample); one Galaxy-wide value, band 0.18-0.47 recorded but not
#: applied (Ishihara et al. 2011's radial gradient).
F_C = 0.18

#: Riebel et al. 2012 (ApJ 753, 71, VizieR J/ApJ/753/71) fit one optical
#: depth per star, at 10.0um for their O-rich GRAMS fits and 11.3um for
#: their C-rich fits (table3.dat ReadMe, note 4) -- one column, the right
#: band already selected per star by the fit itself.
RIEBEL_GCL_COLSPEC = (33, 34)
RIEBEL_TAU_COLSPEC = (85, 92)

#: The curated GRAMS library's own detectability floor in that same
#: per-chemistry band (SPEC_PRIORS.md section 3): "tau_10 >= 0.0128
#: (O-rich) and tau_11.3 >= 0.02 (C-rich)".
TAU_FLOOR_O = 0.0128
TAU_FLOOR_C = 0.02

#: PAHC's own small grid of 8 micron completeness-limit values (SPEC_PRIORS.md
#: section 4, `IMPLEMENTATION.md` section 3): the percentiles of the
#: region's sources' own 50%-completeness limit at I4 the shape is
#: tabulated on.
PAHC_LIMIT_QUANTILES = (5.0, 20.0, 35.0, 50.0, 65.0, 80.0, 95.0, 99.0)
PAHC_LIMIT_MEDIAN_INDEX = PAHC_LIMIT_QUANTILES.index(50.0)

#: SPEC_BMSTP_DRAFT.md section 5.1, "faint end" row: below an anchor's
#: faint edge a star's weight extrapolates the region's own measured
#: faint-end trend (`population.anchor_weights.faint_trend_dex_per_mag`'s
#: `FAINT_TREND_G_DEX_PER_MAG`/`FAINT_TREND_KS_DEX_PER_MAG`) rather than
#: holding the faintest populated bin's weight flat; no floor and no
#: cap, since the trend is itself the measured correction to TRILEGAL's
#: slope. A region whose three faintest populated bins are flat measures
#: a trend of zero and this reduces exactly to the flat rule; a trend
#: with too few populated bins to fit (`nan`) is read as zero for the
#: same reason -- no evidence to extrapolate is not evidence of a trend.


# ---------------------------------------------------------------------------
# the shared distance grid and a tile's own mean profile
# ---------------------------------------------------------------------------

def shared_distance_grid(profile_obj):
    """The common distance grid (module docstring): the profile product's
    own `DIST_PC` (`distance_knots_pc`, plus the implicit `d=0` origin
    every profile starts at), extended by a log-spaced tail from the
    map's own edge to `TAIL_MAX_PC`."""
    knots = profile_obj.distance_knots_pc()
    d_edge = float(knots[-1])
    tail = np.geomspace(d_edge, TAIL_MAX_PC, TAIL_N_NODES)[1:]
    return np.concatenate([[0.0], knots, tail])


def tile_mean_u(profile_obj, dist_grid, pix256, a_pix, n_src):
    """The tile's own mean profile `u(d)` on `dist_grid`: the source-
    weighted mean, over the tile's distinct nside-256 sightlines
    (`pix256`, each's own source count `n_src`), of that sightline's own
    `u(d) = a_of_d(d, pix, A_pix) / A_pix` (spec section 1.4's `u`,
    section 2.2's "Per tile"). One `profile.u` call per sightline, over
    the whole grid at once -- never per star (module docstring)."""
    weight = n_src.astype(np.float64) / n_src.sum()
    mean_u = np.zeros(dist_grid.size, dtype=np.float64)
    for j in range(pix256.size):
        mean_u += weight[j] * profile_obj.u(
            dist_grid, hpx_pix=int(pix256[j]), total_column_ak=float(a_pix[j]))
    return mean_u


# ---------------------------------------------------------------------------
# the per-star weight: joint, marginal, placement-selected marginal, faint/bright end
# ---------------------------------------------------------------------------

def _populated_edge_indices(populated, n_bin):
    """`(bright_idx, faint_idx)`: the first and last bin index the
    region-pooled table actually measured -- the EXPLICIT `POPULATED_G`/
    `POPULATED_KS` flag `population.anchor_weights.fit_tile_weights` now
    returns (owner ruling 2026-09-06, item 3), not the `w_region != 1`
    sentinel. Falls back to the table's own structural edges where
    nothing at all is populated."""
    idx = np.flatnonzero(np.asarray(populated, dtype=bool))
    if idx.size == 0:
        return 0, n_bin - 1
    return int(idx[0]), int(idx[-1])


def star_weights(g_obs, ks_obs, g_edges, ks_edges, w_joint, use_joint,
                  w_g, w_ks, populated_g, populated_ks, u_star, u_front,
                  trend_g=0.0, trend_ks=0.0):
    """Per star, `(W, WEIGHT_RULE)` (module docstring). `w_joint`/`w_g`/
    `w_ks` are the tables this tile actually reads from -- its own, or
    (an excluded tile) the region-pooled `W_REGION_*`, chosen by the
    caller; `use_joint` is always the tile's own (a real crossmatch either
    did or did not populate a bin, independent of exclusion).
    `populated_g`/`populated_ks` are this tile's table's own explicit
    per-bin evidence flag (item 3); `u_star` is the star's own placement
    `A(d)/A(inf)` on this tile's mean profile (`tile_mean_u`); `u_front`
    is that SAME mean profile's own value at the cloud's measured near
    edge (`profile.cloud_front_edge_pc`, `_build_one_tile`'s
    `u_front_tile`) -- the front/behind boundary a star's own placement is
    compared against. `trend_g`/`trend_ks` are this region's own faint-end
    slopes (`FAINT_TREND_G_DEX_PER_MAG`/`FAINT_TREND_KS_DEX_PER_MAG`,
    SPEC_BMSTP_DRAFT.md section 5.1's "faint end" row, module docstring's
    faint-end constant).

    Owner ruling 2026-09-06, item 2: where both marginals are in range
    but the joint bin is not populated, the geometric mean is replaced by
    PLACEMENT -- a star in front of the cloud (`u_star` below `u_front`)
    takes the Gaia weight
    at its own magnitude; a star behind it takes the 2MASS weight. Where
    a star's own anchor is unmeasured in its bin even after the survey
    pool (`populated_*` false AND the stored weight itself is not
    finite -- `population.anchor_weights`' pool fallback already fills most of
    these), it takes the OTHER anchor's weight instead.

    Faint end (SPEC_BMSTP_DRAFT.md section 5.1): a star past an axis's
    own faint edge no longer reads that axis's faintest populated bin
    flat -- it extrapolates from it, `W(faint bin) * 10**(trend * (m -
    m_faint))`, `m` the star's own observed magnitude on that axis and
    `m_faint` the centre of the faintest populated bin. The bright end,
    the joint table, and every in-range star read their bin exactly as
    before (identity: W23's acceptance).
    """
    n_g, n_ks = w_g.size, w_ks.size
    bin_g_raw = np.digitize(g_obs, g_edges) - 1
    bin_ks_raw = np.digitize(ks_obs, ks_edges) - 1
    out_g_faint, out_g_bright = bin_g_raw >= n_g, bin_g_raw < 0
    out_ks_faint, out_ks_bright = bin_ks_raw >= n_ks, bin_ks_raw < 0
    in_range_g = ~out_g_faint & ~out_g_bright
    in_range_ks = ~out_ks_faint & ~out_ks_bright

    g_bright_idx, g_faint_idx = _populated_edge_indices(populated_g, n_g)
    ks_bright_idx, ks_faint_idx = _populated_edge_indices(populated_ks, n_ks)

    bin_g = np.where(out_g_faint, g_faint_idx,
                      np.where(out_g_bright, g_bright_idx, np.clip(bin_g_raw, 0, n_g - 1)))
    bin_ks = np.where(out_ks_faint, ks_faint_idx,
                       np.where(out_ks_bright, ks_bright_idx, np.clip(bin_ks_raw, 0, n_ks - 1)))

    joint_here = use_joint[bin_g, bin_ks]
    both_faint = out_g_faint & out_ks_faint
    both_bright = out_g_bright & out_ks_bright
    both_in_range = in_range_g & in_range_ks

    # the faint-end extrapolation (section 5.1): only where THIS axis is
    # past ITS OWN faint edge does its marginal value leave the flat
    # `w_g[bin_g]`/`w_ks[bin_ks]` it already reads everywhere else --
    # in range, or clamped at the bright edge, `bin_g`/`bin_ks` already
    # index the same bin the flat rule always did, so those stars' values
    # are the same expression, bit for bit, as before this brief.
    trend_g = trend_g if np.isfinite(trend_g) else 0.0
    trend_ks = trend_ks if np.isfinite(trend_ks) else 0.0
    g_centers = 0.5 * (np.asarray(g_edges[:-1]) + np.asarray(g_edges[1:]))
    ks_centers = 0.5 * (np.asarray(ks_edges[:-1]) + np.asarray(ks_edges[1:]))
    g_faint_extrap = w_g[g_faint_idx] * 10.0 ** (trend_g * (np.asarray(g_obs, float) - g_centers[g_faint_idx]))
    ks_faint_extrap = w_ks[ks_faint_idx] * 10.0 ** (trend_ks * (np.asarray(ks_obs, float) - ks_centers[ks_faint_idx]))
    g_val = np.where(out_g_faint, g_faint_extrap, w_g[bin_g])
    ks_val = np.where(out_ks_faint, ks_faint_extrap, w_ks[bin_ks])
    joint_val = w_joint[bin_g, bin_ks]

    # placement-selected marginal (item 2): front of the cloud reads
    # Gaia, behind reads 2MASS; if that side's own bin has no finite
    # weight at all (unmeasured even after the survey pool), fall to the
    # other anchor instead.
    g_measured = np.isfinite(g_val)
    ks_measured = np.isfinite(ks_val)
    front = np.asarray(u_star) < u_front
    placement_val = np.where(
        front,
        np.where(g_measured, g_val, ks_val),
        np.where(ks_measured, ks_val, g_val))

    # priority: fainter than BOTH anchors' faint edges, or brighter than
    # BOTH anchors' bright edges (module docstring); then joint or the
    # placement-selected marginal where both magnitudes are in range;
    # then whichever single marginal is in range. The physically
    # near-impossible remainder -- one axis out on the faint side, the
    # other on the bright side, e.g. an intrinsically extreme colour --
    # falls to the `default`, faint-end priority (module docstring).
    rule = np.select(
        [both_faint,
         both_bright,
         both_in_range & joint_here,
         both_in_range,
         in_range_g,
         in_range_ks],
        [WEIGHT_RULE_FAINT_END, WEIGHT_RULE_BRIGHT_END, WEIGHT_RULE_JOINT,
         WEIGHT_RULE_BOTH_MARGINAL, WEIGHT_RULE_G_MARGINAL, WEIGHT_RULE_KS_MARGINAL],
        default=np.where(out_g_faint | out_ks_faint,
                          WEIGHT_RULE_FAINT_END, WEIGHT_RULE_BRIGHT_END)).astype(np.int8)

    weight = np.select(
        [rule == WEIGHT_RULE_JOINT, rule == WEIGHT_RULE_G_MARGINAL,
         rule == WEIGHT_RULE_KS_MARGINAL, rule == WEIGHT_RULE_BOTH_MARGINAL],
        [joint_val, g_val, ks_val, placement_val],
        # faint end / bright end: the same joint-or-placement rule, at the
        # clamped populated edge bin.
        default=np.where(joint_here, joint_val, placement_val))
    return weight, rule, bin_g, bin_ks


# ---------------------------------------------------------------------------
# F_dusty, survey-wide, from Riebel+2012's per-star GRAMS fits (spec
# section 3, section 10 item 2)
# ---------------------------------------------------------------------------

def read_riebel_optical_depths(config):
    """`(gcl, tau)`, `(n,)` each: every Riebel et al. 2012 AGB candidate's
    own GRAMS chemistry class ("o"/"c") and fitted optical depth, read by
    byte position off the fixed-width `table3.dat.gz` (module docstring's
    `RIEBEL_*_COLSPEC`, `sky.download.riebel2012`'s own ReadMe)."""
    path = f"{config.data_root}/sky/download/riebel2012/table3.dat.gz"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.star_population: no Riebel+2012 table at %r -- run the "
            "sesnaimpute.sky.download.riebel2012 RUNBOOK line first" % path)
    df = pd.read_fwf(path, colspecs=[RIEBEL_GCL_COLSPEC, RIEBEL_TAU_COLSPEC],
                      names=["GCL", "TAU"], compression="gzip")
    return df["GCL"].to_numpy(dtype=str), df["TAU"].to_numpy(dtype=np.float64)


def f_dusty_by_chemistry(config):
    """`(f_dusty_o, f_dusty_c, n_o, n_c)` (SPEC_PRIORS.md section 3, "the
    value is the fraction of Riebel's stars whose fitted optical depth
    exceeds the on-disk floor, per chemistry"): the per-chemistry share
    of Riebel+2012's own per-star fits whose fitted `tau` clears the
    curated GRAMS library's own detectability floor, `TAU_FLOOR_O`/
    `TAU_FLOOR_C`, at that chemistry's own fitted band (10.0um O-rich,
    11.3um C-rich -- one column already carries the right band)."""
    gcl, tau = read_riebel_optical_depths(config)
    is_o, is_c = gcl == "o", gcl == "c"
    n_o, n_c = int(is_o.sum()), int(is_c.sum())
    f_o = float(np.mean(tau[is_o] >= TAU_FLOOR_O)) if n_o else float("nan")
    f_c = float(np.mean(tau[is_c] >= TAU_FLOOR_C)) if n_c else float("nan")
    return f_o, f_c, n_o, n_c


def agb_orich_l_sun(config):
    """`(L_O, n_model)` (SPEC_PRIORS.md section 3 table: "read live from
    the built GRAMS O-rich library (one shared luminosity, 4820 Lsun)"):
    every O-rich model in the curated GRAMS library
    (`sed_models/agb/parameters.fits`) shares one bolometric luminosity,
    so the class mean is exact to floating precision, not a fit. Fails
    loudly if a re-curated library ever ships a spread of O-rich
    luminosities, since `b_agb`'s closed form assumes one shared value."""
    path = f"{config.data_root}/sed_models/agb/parameters.fits"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.star_population: no AGB library parameters at %r -- "
            "the GRAMS library must be curated before this build" % path)
    with fits.open(path) as hdul:
        data = hdul[1].data
        chem = np.array([c.strip() for c in np.asarray(data["CHEM"]).astype(str)])
        l_sun = np.asarray(data["L_SUN"], dtype=np.float64)
    o_vals = l_sun[chem == "O"]
    if o_vals.size == 0:
        raise ValueError("prior.star_population: no CHEM=='O' rows in %r" % path)
    spread = float(o_vals.max() - o_vals.min())
    if spread > 1e-3 * float(o_vals.mean()):
        raise ValueError(
            "prior.star_population: the AGB library's O-rich L_SUN is not "
            "one shared value (spread %.6f Lsun) -- b_agb's closed form "
            "assumes it is" % spread)
    return float(o_vals.mean()), int(o_vals.size)


# ---------------------------------------------------------------------------
# the two libraries' own reference fluxes: STAR's sps match, PAHC's
# continuum table (spec section 2.2, section 4; C3, a unit change only)
# ---------------------------------------------------------------------------

def load_sps_reference_fluxes(config):
    """`(n_model, 8)` mJy, `BAND_KEYS` order: every sps atmosphere
    template's own reference flux (`sed_models/registers/
    sps_register.hdf5`), indexed later by `population.field_stars`'
    `TEMPLATE_INDEX` for `LOG10_B` (STAR, all eight bands)."""
    path = f"{config.data_root}/sed_models/registers/sps_register.hdf5"
    with h5py.File(path, "r") as f:
        models = f["models"]
        return np.column_stack(
            [np.asarray(models["F_REF_%s" % key], dtype=np.float64) for key in BAND_KEYS])


def load_pahc_continuum_reference(config):
    """`(teff_node, ref_jhk)`: the PAHC library's own continuum reference
    table (`sed_models/registers/pahc_register.hdf5`, `library/
    pahc_fstar`), eleven `T_EFF` nodes each carrying the underlying
    star's own J/H/Ks reference flux before any PAH template is added --
    the PAHC register's own reference fluxes (module docstring,
    `LOG10_B_PAHC`), not the atmosphere template's, since the register
    carries them. Its per-model `F_REF_J/H/Ks` (one row per PAH-template
    combination, thousands of rows) carry no independent T_EFF key to
    match a star against economically, so this reduced table -- the
    register's own built-in T_EFF grid -- is what a star's own T_eff is
    matched to."""
    path = f"{config.data_root}/sed_models/registers/pahc_register.hdf5"
    with h5py.File(path, "r") as f:
        grp = f["library/pahc_fstar"]
        teff_node = np.asarray(grp["T_EFF"], dtype=np.float64)
        ref_jhk = np.column_stack(
            [np.asarray(grp["J"], dtype=np.float64), np.asarray(grp["H"], dtype=np.float64),
             np.asarray(grp["Ks"], dtype=np.float64)])
    return teff_node, ref_jhk


# ---------------------------------------------------------------------------
# partition (spec section 3) and the per-star brightness units (C3,
# spec section 2.2, section 3, section 4)
# ---------------------------------------------------------------------------

def evolved_selector(log_g, log_teff, log_l):
    """Which retained field stars are evolved (module docstring's
    `EVOLVED_*` cut), on TRILEGAL's own raw `logg`, `logTe`, `logL`
    columns verbatim."""
    log_g = np.asarray(log_g, dtype=np.float64)
    log_teff = np.asarray(log_teff, dtype=np.float64)
    log_l = np.asarray(log_l, dtype=np.float64)
    return ((log_g < EVOLVED_LOGG_MAX) & (log_teff < EVOLVED_LOGTE_MAX)
            & (log_l > EVOLVED_LOGL_MIN))


def star_brightness_log10_b(fnu_mjy, template_index, f_ref_sps):
    """`LOG10_B` (STAR, module docstring): `log10` of the median over the
    eight bands of a star's own TRILEGAL flux over its matched sps
    template's reference flux -- a unit change only (C3), no floor
    correction (the floor guards a single band's log against a near-zero
    flux, not a same-model ratio of two positive fluxes;
    `population.field_stars.load_atmosphere_grid`'s own precedent)."""
    ratio = fnu_mjy / f_ref_sps[template_index]
    return np.log10(np.median(ratio, axis=1))


def pahc_brightness_log10_b(fnu_mjy, log_teff, teff_node, ref_jhk):
    """`LOG10_B_PAHC` (module docstring): `log10` of the median over
    J/H/Ks of a star's own TRILEGAL flux over the PAHC continuum
    reference at its nearest `T_EFF` node (`load_pahc_continuum_reference`),
    matched in log T_eff."""
    node_log_teff = np.log10(teff_node)
    star_log_teff = np.asarray(log_teff, dtype=np.float64)
    nearest = np.argmin(np.abs(star_log_teff[:, None] - node_log_teff[None, :]), axis=1)
    ratio = fnu_mjy[:, IDX_JHK] / ref_jhk[nearest]
    return np.log10(np.median(ratio, axis=1))


def agb_brightness_log10_b(dist_pc, log_l, l_o_lsun, is_evolved):
    """`(LOG10_B_AGB_C, LOG10_B_AGB_O)` (spec section 3's closed forms,
    `b_agb`): `2*log10(1kpc/d)` for the carbon-rich unit,
    `log10(L/L_O) + 2*log10(1kpc/d)` for the oxygen-rich unit; NaN for a
    non-evolved star (module docstring)."""
    d_kpc = np.asarray(dist_pc, dtype=np.float64) / 1000.0
    two_log_inv_d = 2.0 * np.log10(1.0 / d_kpc)
    log10_b_c = np.where(is_evolved, two_log_inv_d, np.nan)
    log_l_over_l_o = np.asarray(log_l, dtype=np.float64) - np.log10(l_o_lsun)
    log10_b_o = np.where(is_evolved, log_l_over_l_o + two_log_inv_d, np.nan)
    return log10_b_c, log10_b_o


def partition_weights(w, is_evolved, f_dusty_mean):
    """`(w_star, w_agb)` (spec section 3, reading note 04's
    `partition_weights`): `w_AGB = F_dusty * w` on the evolved rows,
    `w_STAR = w - w_AGB`, so `w_STAR + w_AGB == w` row by row -- a
    reweighting, never a filter."""
    w = np.asarray(w, dtype=np.float64)
    w_agb = np.where(is_evolved, f_dusty_mean * w, 0.0)
    return w - w_agb, w_agb


def pahc_limit_grid_mjy(config, region):
    """`(8,)` mJy (module docstring, `PAHC_LIMIT_QUANTILES`): the
    region's own sources' 50%-completeness limit at I4
    (`catalog.limits.limits`), at the eight percentiles PAHC's shape is
    tabulated on."""
    f_lim_i4 = limits_module.limits(config, region)[:, IDX_I4]
    return np.percentile(f_lim_i4, PAHC_LIMIT_QUANTILES)


def pahc_contamination_weight(fnu_8um, a_i, limit_grid_mjy, config, curve):
    """`P_PAHC` (spec section 4): per star and per grid limit,
    `q = F_lim,8 / (F_i(8um)` dimmed by the star's own tile extinction`)`,
    `P(q)` read off the measured curve (`population.pahc_curve.read`)."""
    w_dense = selection.law_dense_weight(a_i)
    kappa_8 = selection.kappa_hybrid(config, w_dense)[:, IDX_I4]
    dimmed_flux_8 = fnu_8um * 10.0 ** (-0.4 * a_i * kappa_8)
    q = limit_grid_mjy[None, :] / dimmed_flux_8[:, None]
    return curve(np.log10(q))


# ---------------------------------------------------------------------------
# per-region reads
# ---------------------------------------------------------------------------

def _read_field_stars(config, region):
    path = config_module.product_path(config, "population", "trilegal", "field-stars", "region", region=region)
    with h5py.File(path, "r") as f:
        stars = dict(
            dist_pc=f["DIST_PC"][:].astype(np.float64),
            g_proxy=f["G_PROXY"][:].astype(np.float64),
            ks_mag=f["KS_MAG"][:].astype(np.float64),
            k_g_diffuse=f["K_G_DIFFUSE"][:].astype(np.float64),
            k_g_dense=f["K_G_DENSE"][:].astype(np.float64),
            log_g=f["LOG_G"][:].astype(np.float64),
            log_teff=f["LOG_TEFF"][:].astype(np.float64),
            log_l=f["LOG_L"][:].astype(np.float64),
            fnu_mjy=f["FNU_MJY"][:].astype(np.float64),
            template_index=f["TEMPLATE_INDEX"][:].astype(np.int64),
            pointing_index=f["POINTING_INDEX"][:].astype(np.int64),
        )
        omega_sim_deg2 = float(f.attrs["OMEGA_SIM_DEG2"])
        n_raw = int(f.attrs["N_RAW"])
    return stars, omega_sim_deg2, n_raw


def _read_tiles(config, region):
    path = config_module.product_path(config, "population", "anchors", "tiles", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.star_population: tiles product missing for region %r at %s -- "
            "run the `prior.anchor_tiles` RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(
            pix512=np.asarray(f["HPX_PIX_512"][:], dtype=np.int64),
            tile_of_pix=np.asarray(f["TILE_ID"][:], dtype=np.int64),
            n_tile=int(f["TILE_L_DEG"].shape[0]),
            tile_omega_deg2=np.asarray(f["TILE_OMEGA_DEG2"][:], dtype=np.float64),
        )


def _read_weights(config, region):
    path = config_module.product_path(config, "population", "anchors", "weights", "tile", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.star_population: anchor weight table missing for region %r at %s "
            "-- run the `prior.anchor_weights` RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(
            w_joint=np.asarray(f["W_JOINT"][:], dtype=np.float64),
            use_joint=np.asarray(f["USE_JOINT"][:], dtype=bool),
            w_g=np.asarray(f["W_G"][:], dtype=np.float64),
            w_ks=np.asarray(f["W_KS"][:], dtype=np.float64),
            w_region_joint=np.asarray(f["W_REGION_JOINT"][:], dtype=np.float64),
            w_region_g=np.asarray(f["W_REGION_G"][:], dtype=np.float64),
            w_region_ks=np.asarray(f["W_REGION_KS"][:], dtype=np.float64),
            # owner ruling 2026-09-06, item 3: the explicit per-bin
            # evidence flag, read straight off `prior.anchor_weights`
            # rather than re-derived from a `!= 1` sentinel here.
            populated_g=np.asarray(f["POPULATED_G"][:], dtype=bool),
            populated_ks=np.asarray(f["POPULATED_KS"][:], dtype=bool),
            excluded=np.asarray(f["EXCLUDED"][:], dtype=bool),
            g_edges=np.asarray(f["G_EDGES"][:], dtype=np.float64),
            ks_edges=np.asarray(f["KS_EDGES"][:], dtype=np.float64),
            # section 5.1's "faint end" row: the region's own faint-slope
            # trend, the extrapolation `star_weights` now applies past
            # each axis's own faint edge.
            trend_g=float(f.attrs["FAINT_TREND_G_DEX_PER_MAG"]),
            trend_ks=float(f.attrs["FAINT_TREND_KS_DEX_PER_MAG"]),
        )


def _read_anchor_ratio(config, region):
    """The region-pooled observed-over-predicted anchor ratio, both
    anchors pooled (`bms/anchors/histograms_anchors_hpx512__<Region>`,
    `population.anchor_tiles`'s own raw predicted histograms; the tile
    weight's own reference point), for this module's report-only check
    against `star_weights`' reweighted retained population."""
    path = config_module.product_path(config, "population", "anchors", "histograms", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        n_obs = float(np.asarray(f["N_G_OBS"]).sum() + np.asarray(f["N_KS_OBS"]).sum())
        n_pred = float(np.asarray(f["N_G_PRED"]).sum() + np.asarray(f["N_KS_PRED"]).sum())
    return n_obs / n_pred if n_pred > 0 else float("nan")


def _region_source_geometry(config, region, tiles):
    """Per-source `HPX_PIX_256`, adopted column, and tile assignment
    (module docstring's Inputs): the tiles product's own `HPX_PIX_512 ->
    TILE_ID` map is the only place tile membership lives (no survey-wide
    tile-membership product exists yet)."""
    rs = access.region_slice(config, region)
    pix512 = np.asarray(rs["hpx_pix_512"], dtype=np.int64)
    pix256 = np.asarray(rs["hpx_pix_256"], dtype=np.int64)
    adopted_path = config_module.product_path(config, "sky/derived", "adopted", "column", "source", region=region)
    a_col = access.per_source(config, region, adopted_path, ["A_COL_K"])["A_COL_K"].astype(np.float64)

    order = np.argsort(tiles["pix512"])
    pix_sorted = tiles["pix512"][order]
    loc = np.searchsorted(pix_sorted, pix512)
    capped = np.minimum(loc, pix_sorted.size - 1) if pix_sorted.size else loc
    valid = pix_sorted.size and bool(np.all(pix_sorted[capped] == pix512))
    if not valid:
        raise ValueError(
            "prior.star_population: %r has source(s) whose HPX_PIX_512 is absent "
            "from the tiles product -- rerun `prior.anchor_tiles` for this region" % region)
    tile_of_source = tiles["tile_of_pix"][order][capped]
    return dict(pix256=pix256, a_col=a_col, tile=tile_of_source)


# ---------------------------------------------------------------------------
# per-tile build
# ---------------------------------------------------------------------------

def tile_centre_lb(pix256_t):
    """The tile's own centre in galactic (l, b) degrees: the plain mean
    over its distinct nside-256 sightlines' pixel centres (owner ruling
    2026-09-06's "by tile centre"). Longitude is unwrapped through 180 deg
    first, matching `sky.download.trilegal.build.region_pointings`'s own
    convention, so a tile never sits on the wrong side of a wrap."""
    l, b = hp.pix2ang(256, np.unique(pix256_t), nest=True, lonlat=True)
    l = np.where(l > 180.0, l - 360.0, l)
    return float(np.mean(l)), float(np.mean(b))


def nearest_pointing(tile_l, tile_b, pointing_l, pointing_b):
    """Index into `pointing_l`/`pointing_b` of the pointing nearest this
    tile's own centre, plain Euclidean in (l, b) degrees -- pointings are
    1.5 deg apart, far coarser than the angular distortion `cos(b)` would
    correct for."""
    d2 = (pointing_l - tile_l) ** 2 + (pointing_b - tile_b) ** 2
    return int(np.argmin(d2))


def _build_one_tile(config, t, geom, stars, weights, profile_obj, dist_grid, curve,
                     pointing_l, pointing_b, present_pointings, front_edge_pc):
    """One tile's placement, weight, partition and brightness units, built
    from ONE pointing's simulated stars only (owner ruling 2026-09-06:
    each tile is assigned to its nearest pointing by tile centre; spec
    section 2.2's "Per tile" population is that pointing's retained
    stars, not the whole region's). Returns the tile's own datasets plus
    its mean-profile array and a few report-only diagnostics (not written
    to the product).

    The nearest pointing is chosen only AMONG `present_pointings` (owner
    ruling: a tile takes its nearest pointing among the pointings actually
    present in the field-stars product, never from the region's full grid)
    -- `population.field_stars`' own W0f single-pointing fallback stores
    every star at `POINTING_INDEX` 0 even when the region's grid has more
    cells, so a tile assigned to an absent grid cell would draw an empty
    sample and every downstream quantity here would be built from zero
    stars. With every grid cell on disk, `present_pointings` is the whole
    grid and the mapping is unchanged."""
    in_tile = geom["tile"] == t
    pix256_t = geom["pix256"][in_tile]
    a_col_t = geom["a_col"][in_tile]
    sightlines, inv, n_src = np.unique(pix256_t, return_inverse=True, return_counts=True)
    a_pix = np.bincount(inv, weights=a_col_t) / n_src
    mean_u = tile_mean_u(profile_obj, dist_grid, sightlines, a_pix, n_src)
    a_tile = float(a_col_t.mean())
    # owner ruling 2026-09-06, item 1 (of the second brief): the front/
    # behind boundary is THIS tile's own mean profile read at the cloud's
    # measured near edge -- the same one-tile-one-mean-profile
    # approximation `u_i` below already makes, not a fixed 0.5.
    u_front_tile = float(np.interp(front_edge_pc, dist_grid, mean_u))

    tile_l, tile_b = tile_centre_lb(pix256_t)
    p_local = nearest_pointing(tile_l, tile_b, pointing_l[present_pointings], pointing_b[present_pointings])
    p_idx = int(present_pointings[p_local])
    star_index = np.flatnonzero(stars["pointing_index"] == p_idx).astype(np.int32)

    u_i = np.interp(stars["dist_pc"][star_index], dist_grid, mean_u)
    a_i = a_tile * u_i

    excluded = bool(weights["excluded"][t])
    w_joint = weights["w_region_joint"] if excluded else weights["w_joint"][t]
    w_g = weights["w_region_g"] if excluded else weights["w_g"][t]
    w_ks = weights["w_region_ks"] if excluded else weights["w_ks"][t]

    g_obs, ks_obs = anchor_tiles.magnitudes_at_extinction(
        a_i, stars["g_proxy"][star_index], stars["ks_mag"][star_index],
        stars["k_g_diffuse"][star_index], stars["k_g_dense"][star_index],
        stars["r_diffuse"], stars["r_dense"])
    w, rule, bin_g, bin_ks = star_weights(
        g_obs, ks_obs, weights["g_edges"], weights["ks_edges"],
        w_joint, weights["use_joint"][t], w_g, w_ks,
        weights["populated_g"], weights["populated_ks"], u_i, u_front_tile,
        trend_g=weights["trend_g"], trend_ks=weights["trend_ks"])

    # the partition (spec section 3): a REWEIGHTING of this tile's own W,
    # never a filter -- w_star + w_agb == w row by row.
    w_star, w_agb = partition_weights(w, stars["is_evolved"][star_index], stars["f_dusty_mean"])

    # PAHC's own weight (spec section 4): the star's own tile extinction
    # `a_i` dims its intrinsic 8um flux before the contrast `q` is formed.
    p_pahc = pahc_contamination_weight(
        stars["fnu_mjy"][star_index][:, IDX_I4], a_i, stars["limit_grid_mjy"], config, curve)

    return dict(
        tile=t, n_sightlines=int(sightlines.size), a_tile=a_tile,
        pointing_index=p_idx, tile_l=tile_l, tile_b=tile_b,
        star_index=star_index,
        u=u_i.astype(np.float32), u_front=u_front_tile,
        w=w.astype(np.float32), rule=rule,
        w_star=w_star.astype(np.float32), w_agb=w_agb.astype(np.float32),
        p_pahc=p_pahc.astype(np.float32),
        mean_u=mean_u, use_joint_any=bool(weights["use_joint"][t].any()),
        w_g=w_g, w_ks=w_ks, bin_g=bin_g, bin_ks=bin_ks,
    )


def build_region(config, region, f_dusty_o, f_dusty_c, l_o_lsun,
                  f_ref_sps, teff_node, ref_jhk, curve, st=None):
    stars_raw, omega_sim_deg2, n_raw = _read_field_stars(config, region)
    r_diffuse = float(selection.ak_per_av(config, 0.0))
    r_dense = float(selection.ak_per_av(config, 1.0))

    # region-level, placement-independent per-star quantities (module
    # docstring, "Partition", "Brightness units"): computed once, not per
    # tile, since neither the HR-diagram cut nor a unit-change ratio
    # depends on where a star sits on a tile's own profile.
    is_evolved = evolved_selector(stars_raw["log_g"], stars_raw["log_teff"], stars_raw["log_l"])
    f_dusty_mean = (1.0 - F_C) * f_dusty_o + F_C * f_dusty_c
    log10_b = star_brightness_log10_b(stars_raw["fnu_mjy"], stars_raw["template_index"], f_ref_sps)
    log10_b_pahc = pahc_brightness_log10_b(stars_raw["fnu_mjy"], stars_raw["log_teff"], teff_node, ref_jhk)
    log10_b_agb_c, log10_b_agb_o = agb_brightness_log10_b(
        stars_raw["dist_pc"], stars_raw["log_l"], l_o_lsun, is_evolved)
    limit_grid_mjy = pahc_limit_grid_mjy(config, region)

    stars = dict(stars_raw, r_diffuse=r_diffuse, r_dense=r_dense,
                 is_evolved=is_evolved, f_dusty_mean=f_dusty_mean,
                 log10_b=log10_b, log10_b_pahc=log10_b_pahc,
                 log10_b_agb_c=log10_b_agb_c, log10_b_agb_o=log10_b_agb_o,
                 limit_grid_mjy=limit_grid_mjy)

    tiles = _read_tiles(config, region)
    weights = _read_weights(config, region)
    geom = _region_source_geometry(config, region, tiles)

    profile_obj = profile_module.read(config, region)
    dist_grid = shared_distance_grid(profile_obj)
    front_edge_pc = profile_obj.cloud_front_edge_pc()

    # the region's own pointing grid (owner ruling 2026-09-06): each
    # tile below is assigned to its nearest pointing centre, in the same
    # order `prior.field_stars` used to number `POINTING_INDEX`.
    pointings = trilegal_download.region_pointings(config, region)
    pointing_l = np.array([p["l_deg"] for p in pointings])
    pointing_b = np.array([p["b_deg"] for p in pointings])
    # each tile's own retained sample is drawn from ONE pointing (owner
    # ruling 2026-09-06); the solid angle that sample covers is that
    # pointing's own queried area, not the region-total OMEGA_SIM_DEG2
    # (all pointings share one area today, `REGION_POINTINGS`).
    pointing_area = np.array([p["area_deg2"] for p in pointings])
    # owner ruling: a tile's nearest pointing is chosen only among the
    # pointings actually on disk (this region's own `POINTING_INDEX`
    # values), never the full grid -- `field_stars`' W0f fallback can
    # leave every star at index 0 while the grid still names every cell.
    present_pointings = np.unique(stars["pointing_index"])

    n_tile_total = tiles["n_tile"]
    _tick_lock = threading.Lock()
    _done_count = [0]

    def _one_tile(t):
        result = _build_one_tile(config, t, geom, stars, weights, profile_obj, dist_grid, curve,
                                  pointing_l, pointing_b, present_pointings, front_edge_pc)
        if st is not None:
            with _tick_lock:
                _done_count[0] += 1
                st.tick(_done_count[0], n_tile_total, "tiles")
        return result

    results = Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(_one_tile)(t) for t in range(n_tile_total))

    # report-only (blessing check, owner ruling 2026-09-06): the pointing a
    # tile would have used under the OLD one-pointing-per-region scheme --
    # nearest to the region's own historical centre -- against the one it
    # actually got, so `_report` can state what fraction of tiles moved.
    old_info = trilegal_download.REGION_POINTINGS[region]
    region_centre_pointing = nearest_pointing(
        old_info["l_deg"], old_info["b_deg"], pointing_l, pointing_b)

    return dict(region=region, n_star=stars["dist_pc"].size, n_raw=n_raw,
                omega_sim_deg2=omega_sim_deg2, dist_grid=dist_grid,
                tiles=results, n_tile=tiles["n_tile"],
                tile_omega_deg2=tiles["tile_omega_deg2"],
                region_anchor_ratio=_read_anchor_ratio(config, region),
                is_evolved=is_evolved, log10_b=log10_b, log10_b_pahc=log10_b_pahc,
                log10_b_agb_c=log10_b_agb_c, log10_b_agb_o=log10_b_agb_o,
                limit_grid_mjy=limit_grid_mjy, f_dusty_mean=f_dusty_mean,
                n_pointing=len(pointings), pointing_l=pointing_l, pointing_b=pointing_b,
                pointing_area=pointing_area,
                region_centre_pointing=region_centre_pointing)


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def write_region(config, region, result, f_dusty_o, f_dusty_c):
    path = config_module.product_path(config, "population", "star", "population", "tile", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "tile"
        f.attrs["OMEGA_SIM_DEG2"] = result["omega_sim_deg2"]
        f.attrs["F_DUSTY_O"] = f_dusty_o
        f.attrs["F_DUSTY_C"] = f_dusty_c
        f.attrs["F_DUSTY_MEAN"] = result["f_dusty_mean"]
        f.attrs["F_C"] = F_C
        f.create_dataset("LIMIT8_GRID_MJY", data=result["limit_grid_mjy"])
        for tile_result in result["tiles"]:
            grp = f.create_group("tile_%d" % tile_result["tile"])
            # section 5.1's Omega_sim = n_pointings * Omega_pointing: the
            # region-total solid angle stays the honest whole-simulation area
            # (anchor_tiles' per-pixel prediction still divides by it); each
            # tile group below carries the ONE pointing's own area instead,
            # since that pointing's retained sample is all a tile draws from.
            grp.attrs["OMEGA_POINTING_DEG2"] = float(result["pointing_area"][tile_result["pointing_index"]])
            grp.create_dataset("STAR_INDEX", data=tile_result["star_index"])
            grp.create_dataset("U", data=tile_result["u"])
            grp.create_dataset("W", data=tile_result["w"])
            grp.create_dataset("W_STAR", data=tile_result["w_star"])
            grp.create_dataset("W_AGB", data=tile_result["w_agb"])
            si = tile_result["star_index"]
            grp.create_dataset("IS_EVOLVED", data=result["is_evolved"][si].astype(np.int8))
            grp.create_dataset("LOG10_B", data=result["log10_b"][si].astype(np.float32))
            grp.create_dataset("LOG10_B_PAHC", data=result["log10_b_pahc"][si].astype(np.float32))
            grp.create_dataset("LOG10_B_AGB_C", data=result["log10_b_agb_c"][si].astype(np.float32))
            grp.create_dataset("LOG10_B_AGB_O", data=result["log10_b_agb_o"][si].astype(np.float32))
            grp.create_dataset("P_PAHC", data=tile_result["p_pahc"])
    return path


# ---------------------------------------------------------------------------
# report (CODING_RULES.md rule 10, 11, 13)
# ---------------------------------------------------------------------------

def _report(result):
    tiles = result["tiles"]
    n_tile = result["n_tile"]
    a_tile = np.array([t["a_tile"] for t in tiles])
    u300 = np.array([np.interp(300.0, result["dist_grid"], t["mean_u"]) for t in tiles])
    u1000 = np.array([np.interp(1000.0, result["dist_grid"], t["mean_u"]) for t in tiles])

    all_rule = np.concatenate([t["rule"] for t in tiles])
    rule_frac = np.bincount(all_rule, minlength=N_WEIGHT_RULES) / all_rule.size

    sigma_w = np.array([float(t["w"].sum()) for t in tiles])
    ratio_per_tile = sigma_w / result["n_raw"]
    omega = result["tile_omega_deg2"]
    sigma_w_over_n_raw = float(np.sum(ratio_per_tile * omega) / np.sum(omega))

    # algebraic acceptance 1: u non-decreasing and <= 1 on every tile's
    # own mean profile (spec section 1.4).
    max_decrease = max(
        float(np.max(-np.diff(t["mean_u"]))) if t["mean_u"].size > 1 else 0.0
        for t in tiles)
    max_decrease = max(0.0, max_decrease)
    max_over_one = max(0.0, float(np.max([np.max(t["mean_u"]) - 1.0 for t in tiles])))

    # algebraic acceptance 2: on a tile with USE_JOINT nowhere set, a
    # star's own single-marginal rule (1 or 2) must equal that marginal's
    # table entry at its own bin exactly.
    no_joint_tiles = [t for t in tiles if not t["use_joint_any"]]
    max_marginal_dev = 0.0
    n_marginal_checked = 0
    n_no_joint_stars = 0
    n_no_joint_non_marginal = 0
    for t in no_joint_tiles:
        is_g = t["rule"] == WEIGHT_RULE_G_MARGINAL
        is_ks = t["rule"] == WEIGHT_RULE_KS_MARGINAL
        n_marginal_checked += int(np.count_nonzero(is_g) + np.count_nonzero(is_ks))
        n_no_joint_stars += t["rule"].size
        # a tile with no populated joint bin can still see a star fall
        # into the "both marginals" placement-selected rule (rule 5) or a faint/
        # bright-end clamp that resolves to it (rule 3/4 with no joint):
        # honestly out of scope for a single-marginal-table identity.
        n_no_joint_non_marginal += int(np.count_nonzero(~is_g & ~is_ks))
        # independent re-read of the marginal table at the star's own
        # stored bin, rather than trusting `star_weights`' own assignment.
        dev_g = np.abs(t["w"][is_g] - t["w_g"][t["bin_g"][is_g]])
        dev_ks = np.abs(t["w"][is_ks] - t["w_ks"][t["bin_ks"][is_ks]])
        if dev_g.size:
            max_marginal_dev = max(max_marginal_dev, float(dev_g.max()))
        if dev_ks.size:
            max_marginal_dev = max(max_marginal_dev, float(dev_ks.max()))
    frac_no_joint_non_marginal = (n_no_joint_non_marginal / n_no_joint_stars
                                   if n_no_joint_stars else float("nan"))

    # tile-Omega-weighted combination of a per-tile sum into one region
    # count per deg2 (module docstring's Sigma w_STAR/Sigma w_AGB/Sigma
    # (w.P_PAHC); the same combination `sigma_w_over_n_raw` already uses
    # for the retained-fraction check, here against `OMEGA_SIM_DEG2`
    # rather than `N_RAW`, per the brief).
    def _combined_per_deg2(sums_per_tile):
        ratio = np.asarray(sums_per_tile) / result["omega_sim_deg2"]
        return float(np.sum(ratio * omega) / np.sum(omega))

    sigma_w_star = [float(t["w_star"].sum()) for t in tiles]
    sigma_w_agb = [float(t["w_agb"].sum()) for t in tiles]
    sigma_w_pahc_median = [float((t["w"] * t["p_pahc"][:, PAHC_LIMIT_MEDIAN_INDEX]).sum())
                            for t in tiles]
    star_per_deg2 = _combined_per_deg2(sigma_w_star)
    agb_per_deg2 = _combined_per_deg2(sigma_w_agb)
    pahc_per_deg2 = _combined_per_deg2(sigma_w_pahc_median)

    frac_evolved_stars = float(np.mean(result["is_evolved"]))
    frac_evolved_weight_per_tile = [
        float(t["w"][result["is_evolved"][t["star_index"]]].sum() / t["w"].sum()) for t in tiles]
    frac_evolved_weight = float(
        np.sum(np.asarray(frac_evolved_weight_per_tile) * omega) / np.sum(omega))

    log10_b = result["log10_b"]
    log10_b_median = float(np.median(log10_b))
    log10_b_p16, log10_b_p84 = (float(x) for x in np.percentile(log10_b, [16.0, 84.0]))

    # algebraic identity (spec section 3): w_star + w_agb == w row by row.
    max_partition_dev = max(
        float(np.max(np.abs((t["w_star"] + t["w_agb"]).astype(np.float64) - t["w"].astype(np.float64))))
        for t in tiles)

    frac_pointing_changed = float(np.mean(
        [t["pointing_index"] != result["region_centre_pointing"] for t in tiles]))

    return dict(
        n_tile=n_tile, a_tile_min=float(a_tile.min()), a_tile_max=float(a_tile.max()),
        n_pointing=result["n_pointing"], frac_pointing_changed=frac_pointing_changed,
        u300_median=float(np.median(u300)), u1000_median=float(np.median(u1000)),
        rule_frac=rule_frac, sigma_w_over_n_raw=sigma_w_over_n_raw,
        region_anchor_ratio=result["region_anchor_ratio"],
        max_decrease=max_decrease, max_over_one=max_over_one,
        n_no_joint_tiles=len(no_joint_tiles), n_marginal_checked=n_marginal_checked,
        max_marginal_dev=max_marginal_dev,
        frac_no_joint_non_marginal=frac_no_joint_non_marginal,
        star_per_deg2=star_per_deg2, agb_per_deg2=agb_per_deg2, pahc_per_deg2=pahc_per_deg2,
        frac_evolved_stars=frac_evolved_stars, frac_evolved_weight=frac_evolved_weight,
        log10_b_median=log10_b_median, log10_b_p16=log10_b_p16, log10_b_p84=log10_b_p84,
        max_partition_dev=max_partition_dev,
    )


def build(config, regions=None):
    """Writes the per-tile placement-and-weight product for `regions`
    (default: all thirty), one file per region (module docstring). The
    literature `F_dusty` (Riebel+2012 against the curated GRAMS floor)
    and the GRAMS O-rich library's own shared luminosity are survey-wide
    and read once, not per region (rule 9); likewise the two libraries'
    own reference-flux tables and the measured PAHC curve."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    f_dusty_o, f_dusty_c, n_riebel_o, n_riebel_c = f_dusty_by_chemistry(config)
    l_o_lsun, n_orich_models = agb_orich_l_sun(config)
    f_ref_sps = load_sps_reference_fluxes(config)
    teff_node, ref_jhk = load_pahc_continuum_reference(config)
    curve = pahc_curve.read(config)
    print(
        "star_population: F_dusty_O=%.4f (n=%d) F_dusty_C=%.4f (n=%d) "
        "L_O=%.2f Lsun (n_model=%d, sed_models/agb/parameters.fits CHEM=='O') f_C=%.2f"
        % (f_dusty_o, n_riebel_o, f_dusty_c, n_riebel_c, l_o_lsun, n_orich_models, F_C))

    for region in region_names:
        with progress.Stage("prior.star_population", region) as st:
            result = build_region(config, region, f_dusty_o, f_dusty_c, l_o_lsun,
                                   f_ref_sps, teff_node, ref_jhk, curve, st=st)
            path = write_region(config, region, result, f_dusty_o, f_dusty_c)
            rep = _report(result)
            st.done(path, n_tile=rep["n_tile"], n_pointing=rep["n_pointing"],
                    star_per_deg2=rep["star_per_deg2"])
        rule_str = " ".join("%d=%.3f" % (k, rep["rule_frac"][k]) for k in range(N_WEIGHT_RULES))
        print(
            "star_population: %s: n_tile=%d n_pointing=%d frac_pointing_changed=%.4f "
            "A_TILE_K=[%.3f,%.3f] "
            "median_u(300pc)=%.4f median_u(1kpc)=%.4f "
            "weight_rule_frac(0-5)=[%s] sigma_w_over_n_raw=%.4f region_anchor_ratio=%.4f "
            "u_max_decrease=%.2e u_max_over_one=%.2e "
            "no_joint_tiles=%d marginal_checked=%d marginal_max_dev=%.2e "
            "no_joint_non_marginal_frac=%.4f -> %s"
            % (region, rep["n_tile"], rep["n_pointing"], rep["frac_pointing_changed"],
               rep["a_tile_min"], rep["a_tile_max"],
               rep["u300_median"], rep["u1000_median"], rule_str,
               rep["sigma_w_over_n_raw"], rep["region_anchor_ratio"],
               rep["max_decrease"], rep["max_over_one"],
               rep["n_no_joint_tiles"], rep["n_marginal_checked"], rep["max_marginal_dev"],
               rep["frac_no_joint_non_marginal"],
               path))
        print(
            "star_population: %s: STAR/deg2=%.3f AGB/deg2=%.3f PAHC/deg2@median_limit=%.3f "
            "evolved_frac(stars)=%.4f evolved_frac(weight)=%.4f "
            "LOG10_B_median=%.3f [16-84%%]=[%.3f,%.3f] "
            "partition_identity_max_dev=%.2e"
            % (region, rep["star_per_deg2"], rep["agb_per_deg2"], rep["pahc_per_deg2"],
               rep["frac_evolved_stars"], rep["frac_evolved_weight"],
               rep["log10_b_median"], rep["log10_b_p16"], rep["log10_b_p84"],
               rep["max_partition_dev"]))


if __name__ == "__main__":
    run(build)
