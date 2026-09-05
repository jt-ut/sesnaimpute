"""Per-tile field-star placement and anchor weight (SPEC_PRIORS.md section
1.4's `u`, section 1.5, section 2.1's `star_reweight_factor` and "faint
end", section 2.2's shape population `a_i = A_s u_i` and "Per tile";
reading note 04_star_family.md section B's `star_reweight_factor`;
IMPLEMENTATION.md section 6, stage 9).

This is the placement-and-weight half of the STAR/AGB/PAHC family module:
where every retained field star (`prior.field_stars`) sits in extinction
on a tile's own line of sight, and what per-star anchor weight
(`prior.anchor_weights`) it carries there. The partition into STAR/AGB/
PAHC, the brightness unit `B`, and PAHC's contamination weight are a
second, later module.

Placement (spec section 2.2, "Per tile"). A field star carries a distance
but no sky position of its own (`prior.field_stars`'s one TRILEGAL
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
the tile's own `W_JOINT` table (`prior.anchor_weights`) supplies the
weight where the bin is populated by a real crossmatch (`USE_JOINT`),
else the `G` or `Ks` marginal the star's own predicted magnitude falls
inside. Where BOTH marginals apply and the joint bin does not, the two
marginals' geometric mean stands in (spec section 2.1, "if both, the
geometric mean") -- recorded under its own `WEIGHT_RULE` code, 5, beyond
the five the brief names, so the per-rule star counts this module reports
add up honestly. A star beyond both anchors' faint (or both anchors'
bright) edges reads its own tile's faintest (brightest) POPULATED bin --
populated in the region-pooled table's own sense (`W_REGION_G/KS/JOINT
!= 1`, `prior.anchor_weights`' own "no evidence anywhere" placeholder) --
under the same joint-or-marginal rule; where the two axes disagree (one
out on the faint side, the other on the bright side -- a physically rare,
intrinsically very red or very blue star), the star is reported faint-end
by priority, but each axis still reads its own out-of-range direction's
populated edge bin. A cluster-excluded tile (`EXCLUDED`) reads its weight
from the region-pooled table (`W_REGION_*`) rather than its own tile row;
`prior.anchor_weights`' own shrinkage already collapses an excluded
tile's stored row to exactly this value, so reading `W_REGION_*` directly
makes that fact part of the code a reader sees rather than something they
have to trace through the upstream build to learn.

WEIGHT_RULE codes, per star: 0 joint, 1 G marginal only, 2 Ks marginal
only, 3 faint end, 4 bright end, 5 both marginals (geometric mean).

Writes, per region, `bms/star/population_star_tile__<Region>.hdf5`: root
attrs `GRANULE="tile"`, `OMEGA_SIM_DEG2`; a `DIST_GRID` dataset, the
common distance grid every tile's mean profile and every star's `U` were
read off; one HDF5 group `tile_<id>` per tile with datasets `STAR_INDEX`
(row into `prior.field_stars`' retained group), `U`, `A`, `W`,
`WEIGHT_RULE`, and attrs `A_TILE_K`, `N_SIGHTLINES`.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import anchor_tiles, selection
from sesnaimpute.sky.derived import profile as profile_module

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
# the per-star weight: joint, marginal, geometric mean, faint/bright end
# ---------------------------------------------------------------------------

def _populated_edge_indices(w_region):
    """`(bright_idx, faint_idx)`: the first and last bin index the
    region-pooled table actually measured (`w_region != 1`, its own "no
    evidence anywhere" placeholder; `prior.anchor_weights.
    faint_trend_dex_per_mag` uses the same test). Falls back to the
    table's own structural edges where nothing at all is populated."""
    populated = np.flatnonzero(w_region != 1.0)
    if populated.size == 0:
        return 0, w_region.size - 1
    return int(populated[0]), int(populated[-1])


def star_weights(g_obs, ks_obs, g_edges, ks_edges, w_joint, use_joint,
                  w_g, w_ks, w_region_g, w_region_ks):
    """Per star, `(W, WEIGHT_RULE)` (module docstring). `w_joint`/`w_g`/
    `w_ks` are the tables this tile actually reads from -- its own, or
    (an excluded tile) the region-pooled `W_REGION_*`, chosen by the
    caller; `use_joint` is always the tile's own (a real crossmatch either
    did or did not populate a bin, independent of exclusion)."""
    n_g, n_ks = w_g.size, w_ks.size
    bin_g_raw = np.digitize(g_obs, g_edges) - 1
    bin_ks_raw = np.digitize(ks_obs, ks_edges) - 1
    out_g_faint, out_g_bright = bin_g_raw >= n_g, bin_g_raw < 0
    out_ks_faint, out_ks_bright = bin_ks_raw >= n_ks, bin_ks_raw < 0
    in_range_g = ~out_g_faint & ~out_g_bright
    in_range_ks = ~out_ks_faint & ~out_ks_bright

    g_bright_idx, g_faint_idx = _populated_edge_indices(w_region_g)
    ks_bright_idx, ks_faint_idx = _populated_edge_indices(w_region_ks)

    bin_g = np.where(out_g_faint, g_faint_idx,
                      np.where(out_g_bright, g_bright_idx, np.clip(bin_g_raw, 0, n_g - 1)))
    bin_ks = np.where(out_ks_faint, ks_faint_idx,
                       np.where(out_ks_bright, ks_bright_idx, np.clip(bin_ks_raw, 0, n_ks - 1)))

    joint_here = use_joint[bin_g, bin_ks]
    both_faint = out_g_faint & out_ks_faint
    both_bright = out_g_bright & out_ks_bright
    both_in_range = in_range_g & in_range_ks
    g_val = w_g[bin_g]
    ks_val = w_ks[bin_ks]
    joint_val = w_joint[bin_g, bin_ks]
    geomean_val = np.sqrt(g_val * ks_val)

    # priority: fainter than BOTH anchors' faint edges, or brighter than
    # BOTH anchors' bright edges (module docstring); then joint or the
    # geometric mean of both marginals where both magnitudes are in
    # range; then whichever single marginal is in range. The physically
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
        [joint_val, g_val, ks_val, geomean_val],
        # faint end / bright end: the same joint-or-marginal rule, at the
        # clamped populated edge bin.
        default=np.where(joint_here, joint_val, geomean_val))
    return weight, rule, bin_g, bin_ks


# ---------------------------------------------------------------------------
# per-region reads
# ---------------------------------------------------------------------------

def _read_field_stars(config, region):
    path = config_module.product_path(config, "bms", "trilegal", "field-stars", "region", region=region)
    with h5py.File(path, "r") as f:
        stars = dict(
            dist_pc=f["DIST_PC"][:].astype(np.float64),
            g_proxy=f["G_PROXY"][:].astype(np.float64),
            ks_mag=f["KS_MAG"][:].astype(np.float64),
            k_g_diffuse=f["K_G_DIFFUSE"][:].astype(np.float64),
            k_g_dense=f["K_G_DENSE"][:].astype(np.float64),
        )
        omega_sim_deg2 = float(f.attrs["OMEGA_SIM_DEG2"])
        n_raw = int(f.attrs["N_RAW"])
    return stars, omega_sim_deg2, n_raw


def _read_tiles(config, region):
    path = config_module.product_path(config, "bms", "anchors", "tiles", "hpx512", region=region)
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
    path = config_module.product_path(config, "bms", "anchors", "weights", "tile", region=region)
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
            excluded=np.asarray(f["EXCLUDED"][:], dtype=bool),
            g_edges=np.asarray(f["G_EDGES"][:], dtype=np.float64),
            ks_edges=np.asarray(f["KS_EDGES"][:], dtype=np.float64),
        )


def _read_anchor_ratio(config, region):
    """The region-pooled observed-over-predicted anchor ratio, both
    anchors pooled (`bms/anchors/histograms_anchors_hpx512__<Region>`,
    `prior.anchor_tiles`'s own raw predicted histograms; the tile
    weight's own reference point), for this module's report-only check
    against `star_weights`' reweighted retained population."""
    path = config_module.product_path(config, "bms", "anchors", "histograms", "hpx512", region=region)
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

def _build_one_tile(t, geom, stars, weights, profile_obj, dist_grid):
    """One tile's placement and weight for the WHOLE retained field-star
    population (spec section 2.2, "Per tile": the same population,
    deposited one by one, per tile). Returns the tile's own datasets plus
    its mean-profile array and a few report-only diagnostics (not written
    to the product)."""
    in_tile = geom["tile"] == t
    pix256_t = geom["pix256"][in_tile]
    a_col_t = geom["a_col"][in_tile]
    sightlines, inv, n_src = np.unique(pix256_t, return_inverse=True, return_counts=True)
    a_pix = np.bincount(inv, weights=a_col_t) / n_src
    mean_u = tile_mean_u(profile_obj, dist_grid, sightlines, a_pix, n_src)
    a_tile = float(a_col_t.mean())

    u_i = np.interp(stars["dist_pc"], dist_grid, mean_u)
    a_i = a_tile * u_i

    excluded = bool(weights["excluded"][t])
    w_joint = weights["w_region_joint"] if excluded else weights["w_joint"][t]
    w_g = weights["w_region_g"] if excluded else weights["w_g"][t]
    w_ks = weights["w_region_ks"] if excluded else weights["w_ks"][t]

    g_obs, ks_obs = anchor_tiles.magnitudes_at_extinction(
        a_i, stars["g_proxy"], stars["ks_mag"], stars["k_g_diffuse"], stars["k_g_dense"],
        stars["r_diffuse"], stars["r_dense"])
    w, rule, bin_g, bin_ks = star_weights(
        g_obs, ks_obs, weights["g_edges"], weights["ks_edges"],
        w_joint, weights["use_joint"][t], w_g, w_ks,
        weights["w_region_g"], weights["w_region_ks"])

    return dict(
        tile=t, n_sightlines=int(sightlines.size), a_tile=a_tile,
        star_index=np.arange(stars["dist_pc"].size, dtype=np.int32),
        u=u_i.astype(np.float32), a=a_i.astype(np.float32),
        w=w.astype(np.float32), rule=rule,
        mean_u=mean_u, use_joint_any=bool(weights["use_joint"][t].any()),
        w_g=w_g, w_ks=w_ks, bin_g=bin_g, bin_ks=bin_ks,
    )


def build_region(config, region):
    stars_raw, omega_sim_deg2, n_raw = _read_field_stars(config, region)
    r_diffuse = float(selection.ak_per_av(config, 0.0))
    r_dense = float(selection.ak_per_av(config, 1.0))
    stars = dict(stars_raw, r_diffuse=r_diffuse, r_dense=r_dense)

    tiles = _read_tiles(config, region)
    weights = _read_weights(config, region)
    geom = _region_source_geometry(config, region, tiles)

    profile_obj = profile_module.read(config, region)
    dist_grid = shared_distance_grid(profile_obj)

    results = Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(_build_one_tile)(t, geom, stars, weights, profile_obj, dist_grid)
        for t in range(tiles["n_tile"]))

    return dict(region=region, n_star=stars["dist_pc"].size, n_raw=n_raw,
                omega_sim_deg2=omega_sim_deg2, dist_grid=dist_grid,
                tiles=results, n_tile=tiles["n_tile"],
                tile_omega_deg2=tiles["tile_omega_deg2"],
                region_anchor_ratio=_read_anchor_ratio(config, region))


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def write_region(config, region, result):
    path = config_module.product_path(config, "bms", "star", "population", "tile", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "tile"
        f.attrs["OMEGA_SIM_DEG2"] = result["omega_sim_deg2"]
        f.create_dataset("DIST_GRID", data=result["dist_grid"])
        for tile_result in result["tiles"]:
            grp = f.create_group("tile_%d" % tile_result["tile"])
            grp.attrs["A_TILE_K"] = tile_result["a_tile"]
            grp.attrs["N_SIGHTLINES"] = tile_result["n_sightlines"]
            grp.create_dataset("STAR_INDEX", data=tile_result["star_index"])
            grp.create_dataset("U", data=tile_result["u"])
            grp.create_dataset("A", data=tile_result["a"])
            grp.create_dataset("W", data=tile_result["w"])
            grp.create_dataset("WEIGHT_RULE", data=tile_result["rule"])
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
        # into the "both marginals" geometric mean (rule 5) or a faint/
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

    return dict(
        n_tile=n_tile, a_tile_min=float(a_tile.min()), a_tile_max=float(a_tile.max()),
        u300_median=float(np.median(u300)), u1000_median=float(np.median(u1000)),
        rule_frac=rule_frac, sigma_w_over_n_raw=sigma_w_over_n_raw,
        region_anchor_ratio=result["region_anchor_ratio"],
        max_decrease=max_decrease, max_over_one=max_over_one,
        n_no_joint_tiles=len(no_joint_tiles), n_marginal_checked=n_marginal_checked,
        max_marginal_dev=max_marginal_dev,
        frac_no_joint_non_marginal=frac_no_joint_non_marginal,
    )


def build(config, regions=None):
    """Writes the per-tile placement-and-weight product for `regions`
    (default: all thirty), one file per region (module docstring)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        result = build_region(config, region)
        path = write_region(config, region, result)
        rep = _report(result)
        rule_str = " ".join("%d=%.3f" % (k, rep["rule_frac"][k]) for k in range(N_WEIGHT_RULES))
        print(
            "star_population: %s: n_tile=%d A_TILE_K=[%.3f,%.3f] "
            "median_u(300pc)=%.4f median_u(1kpc)=%.4f "
            "weight_rule_frac(0-5)=[%s] sigma_w_over_n_raw=%.4f region_anchor_ratio=%.4f "
            "u_max_decrease=%.2e u_max_over_one=%.2e "
            "no_joint_tiles=%d marginal_checked=%d marginal_max_dev=%.2e "
            "no_joint_non_marginal_frac=%.4f -> %s"
            % (region, rep["n_tile"], rep["a_tile_min"], rep["a_tile_max"],
               rep["u300_median"], rep["u1000_median"], rule_str,
               rep["sigma_w_over_n_raw"], rep["region_anchor_ratio"],
               rep["max_decrease"], rep["max_over_one"],
               rep["n_no_joint_tiles"], rep["n_marginal_checked"], rep["max_marginal_dev"],
               rep["frac_no_joint_non_marginal"],
               path))


if __name__ == "__main__":
    run(build)
