"""The STAR anchors' observed joint histogram and the young-star
subtraction (SPEC_PRIORS.md section 2.1, the "joint and marginal bins"
and "young stars in the anchors" rows), per nside-512 anchor pixel --
the same pixels and edges `population.anchor_tiles.write_histograms` and
`population.young_stars` already wrote.

N_GK_OBS. Read from `sky/derived/gaia/joint-counts_gaia-twomass_hpx512__
<Region>.hdf5` (`sesnaimpute.sky.derived.gaia_twomass_counts`) and placed
onto the histograms product's own `G_EDGES`/`KS_EDGES`: the joint count
is the Gaia archive's own Gaia-DR3-to-2MASS crossmatch, complete on the
whole pixel like the two marginals, not the survey's own sources (SPEC_
BMSTP_DRAFT.md section 5.1, "joint and marginal bins").

THE SUBTRACTION (section 2.1, "young stars in the anchors"). Before a
weight is formed elsewhere, each observed histogram has the model's own
expectation of young stars subtracted, floored at zero:

    N_G_SUB  = max(N_G_OBS  - N_G_YOUNG,  0)
    N_KS_SUB = max(N_KS_OBS - N_KS_YOUNG, 0)
    N_GK_SUB = max(N_GK_OBS - N_GK_YOUNG, 0)

`population.young_stars` carries only the two young-star marginals, not their
joint, so `N_GK_YOUNG` is formed here as the outer product of
`N_G_YOUNG` and `N_KS_YOUNG`, normalised to the pixel's own young-star
total (`N_YOUNG_TOTAL`, zero where the total is zero):

    N_GK_YOUNG[pix] = N_G_YOUNG[pix] (x) N_KS_YOUNG[pix] / N_YOUNG_TOTAL[pix]

This assumes a young star's `G` and `Ks` placements are independent
given the pixel, which they are not exactly -- both come from the same
drawn mass and the same drawn extinction. The exact joint is a later
refinement of `young_stars.py`, which would have to carry it directly
rather than leave it to be reconstructed from two marginals.

`FLOORED_G`/`FLOORED_KS`/`FLOORED_GK` (n_pix): the number of bins per
pixel where the floor actually triggered (`N_*_OBS < N_*_YOUNG`).
`SUBTRACTED_FRAC_G`/`SUBTRACTED_FRAC_KS` (n_pix): the pixel-total share
of the observed marginal removed by the subtraction, `(N_OBS.sum() -
N_SUB.sum()) / N_OBS.sum()` (zero where the pixel has no observed
counts at all).

THE CLUSTER SUBTRACTION (SPEC_BMSTP_DRAFT.md section 5.1, "clusters"
row). After the young-star subtraction above, the members of every Hunt
& Reffert 2023 bound cluster (A&A 673, A114; VizieR J/A+A/673/A114,
`Type` "o" open or "g" globular) whose centre lies within its own
half-member radius `r50` plus half a tile's side (`L_STAR_DEG/2`) of any
of the region's own tiles are ALSO subtracted, bin by bin, floored at
zero:

    N_G_SUB  = max(N_G_SUB  - N_G_CLUSTER,  0)
    N_KS_SUB = max(N_KS_SUB - N_KS_CLUSTER, 0)
    N_GK_SUB = max(N_GK_SUB - N_GK_CLUSTER, 0)

Unlike the young-star law, a cluster member is a real Gaia DR3 source
with its own measured `Gmag`, `BP-RP` and sky position
(`sky/download/hunt_reffert2023/members.dat`), so `N_G_CLUSTER` is an
exact per-member count -- placed at the member's own nside-512 pixel by
its own `(GLON, GLAT)`, not spread by a law -- rather than an integrated
expectation, and `N_GK_CLUSTER` is the true joint of each member's own
`(G, Ks)` pair (`_cluster_member_histograms`), not the young-star
marginals' outer-product approximation. A member enters only with
membership probability `Prob >= _MEMBER_PROB_MIN` (Hunt & Reffert 2023,
Sect. 4: every member within the fitted tidal radius, their own "good
member" flag, carries `Prob > 0.5`, so the catalogue's own recommended
cut is the same threshold). `Ks` is estimated from the member's own
`Gmag` and `BP-RP` through the Gaia DR2 photometric relationship `G -
Ks = -0.1885 + 2.092 x - 0.1345 x^2` (`x = BP-RP`; Gaia DR2 documentation,
Evans et al. 2018, A&A 616, A4, Table 5.8/5.9, "G-Ks = f(GBP-GRP)", fit
scatter 0.083 mag, tabulated for `0.25 < x < 5.5`, extrapolated outside
that range) -- the table carries `BP-RP` for essentially every member, so
the Gaia and 2MASS bins both subtract; a member with no finite `BP-RP`
subtracts from the Gaia bins only.

`CLUSTER_FLOORED_G`/`CLUSTER_FLOORED_KS`/`CLUSTER_FLOORED_GK` (n_pix):
the number of bins per pixel where this second floor triggered
(`N_*_SUB(pre-cluster) < N_*_CLUSTER`), reported the same way as the
young-star floor count. `population.anchor_weights.cluster_excluded_tiles`
no longer excludes a tile for this overlap (module docstring there): the
tile keeps its own (now cluster-cleaned) weights instead.

Product, per region, `bms/anchors/observed_anchors_hpx512__<Region>
.hdf5`: `HPX_PIX_512`, `G_EDGES`, `KS_EDGES`, `N_G_SUB`, `N_KS_SUB`,
`N_GK_SUB` (n_pix, n_G_bin, n_Ks_bin` for the joint array, `(n_pix,
n_bin)` for the marginals); root attr `GRANULE="hpx512"`.
"""

import os

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.population import anchor_weights as anchor_weights_module

#: `population.anchor_tiles.NSIDE`: the anchor histograms' own pixel
#: grain, matched here so a cluster member's own position lands on the
#: same pixel index the histograms use.
_NSIDE_HPX = 512

#: Hunt & Reffert 2023's own recommended "good member" cut (module
#: docstring): every member within the fitted tidal radius carries
#: `Prob > 0.5`, and a member below it is flagged low quality.
_MEMBER_PROB_MIN = 0.5

#: `sky/download/hunt_reffert2023/members.dat`'s own byte layout (its
#: ReadMe, "Byte-by-byte Description of file: members.dat"): Name (1-20),
#: Prob (53-72), GLON (166-189, deg), GLAT (191-213, deg), Gmag
#: (783-801, mag), BP-RP (843-865, mag) -- 0-based half-open
#: `pandas.read_fwf` colspecs.
_MEMBER_COLSPECS = [(0, 20), (52, 72), (165, 189), (190, 213), (782, 801), (842, 865)]
_MEMBER_NAMES = ["NAME", "PROB", "GLON", "GLAT", "GMAG", "BP_RP"]
#: rows read per `pandas.read_fwf` chunk while scanning `members.dat`
#: for the region's own candidate clusters (CODING_RULES.md rule 10b).
_MEMBER_CHUNK_ROWS = 200_000

#: Gaia DR2 documentation (Evans et al. 2018, A&A 616, A4), Table 5.8/5.9,
#: "G-Ks = f(GBP-GRP)": G - Ks = c0 + c1*x + c2*x^2, x = BP-RP, fit
#: scatter 0.083 mag, tabulated for 0.25 < x < 5.5 (module docstring).
_GAIA_TO_KS_C0 = -0.1885
_GAIA_TO_KS_C1 = 2.092
_GAIA_TO_KS_C2 = -0.1345


# ---------------------------------------------------------------------
# the cluster-member subtraction (module docstring, "THE CLUSTER
# SUBTRACTION")
# ---------------------------------------------------------------------

def ks_from_gaia(g_mag, bp_rp):
    """A cluster member's `Ks` estimated from its own `Gmag` and `BP-RP`
    through the Gaia DR2 `G-Ks = f(GBP-GRP)` relationship (module
    docstring's constants block). `nan` in, `nan` out: a member with no
    finite `BP-RP` in the catalogue yields no `Ks` estimate and is left
    out of the 2MASS-bin subtraction by the caller.
    """
    bp_rp = np.asarray(bp_rp, dtype=np.float64)
    g_minus_ks = _GAIA_TO_KS_C0 + _GAIA_TO_KS_C1 * bp_rp + _GAIA_TO_KS_C2 * bp_rp ** 2
    return np.asarray(g_mag, dtype=np.float64) - g_minus_ks


def _clusters_touching_region(tile_l_deg, tile_b_deg, l_star_deg, clusters):
    """Bound clusters (`clusters` already `Type` "o"/"g" only) whose
    centre lies within `r50 + L_STAR_DEG/2` of at least one of the
    region's own tiles -- the same geometric test
    `anchor_weights.cluster_excluded_tiles` applies per tile, used here
    to pick which clusters' members are even candidates for this
    region's subtraction, rather than to exclude a tile.
    """
    n_cl = clusters["name"].size
    if n_cl == 0 or tile_l_deg.size == 0:
        return np.zeros(n_cl, dtype=bool)
    sep = anchor_weights_module._angular_sep_deg(
        tile_l_deg[:, None], tile_b_deg[:, None], clusters["glon"][None, :], clusters["glat"][None, :])
    margin = sep - (clusters["r50_deg"][None, :] + 0.5 * float(l_star_deg))
    return (margin <= 0.0).any(axis=0)


def _read_cluster_members(config, cluster_names):
    """The Gaia DR3 member stars of `cluster_names` (probability >=
    `_MEMBER_PROB_MIN`), from `sky/download/hunt_reffert2023/members.dat`
    (module docstring's byte layout). Read and filtered in
    `_MEMBER_CHUNK_ROWS`-row batches (CODING_RULES.md rule 10b): the
    file carries every cluster's members (1.29 million rows) but a
    region's own candidate set is a small fraction of the clusters, so
    each chunk is filtered to that set before it is kept, and only the
    kept rows accumulate.
    """
    path = f"{config.data_root}/sky/download/hunt_reffert2023/members.dat"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.anchor_observed: Hunt & Reffert 2023 member table missing at "
            f"{path} -- run `sesnaimpute.sky.download.hunt_reffert2023.build` first")
    name_set = set(str(n) for n in cluster_names)
    if not name_set:
        return dict(glon=np.zeros(0), glat=np.zeros(0), gmag=np.zeros(0), bp_rp=np.zeros(0))
    kept = []
    reader = pd.read_fwf(path, colspecs=_MEMBER_COLSPECS, names=_MEMBER_NAMES,
                          header=None, dtype=str, chunksize=_MEMBER_CHUNK_ROWS)
    for chunk in reader:
        chunk = chunk[chunk["NAME"].str.strip().isin(name_set)]
        if len(chunk):
            kept.append(chunk)
    if not kept:
        return dict(glon=np.zeros(0), glat=np.zeros(0), gmag=np.zeros(0), bp_rp=np.zeros(0))
    df = pd.concat(kept, ignore_index=True)
    for col in ("PROB", "GLON", "GLAT", "GMAG", "BP_RP"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["PROB"] >= _MEMBER_PROB_MIN]
    return dict(glon=df["GLON"].to_numpy(dtype=np.float64),
                glat=df["GLAT"].to_numpy(dtype=np.float64),
                gmag=df["GMAG"].to_numpy(dtype=np.float64),
                bp_rp=df["BP_RP"].to_numpy(dtype=np.float64))


def _cluster_member_histograms(pixels, members, g_edges, ks_edges):
    """`(N_G_CLUSTER, N_KS_CLUSTER, N_GK_CLUSTER)`: every candidate
    member placed at its own nside-512 pixel by its own `(GLON, GLAT)`
    (module docstring) and binned by its own `Gmag` and estimated `Ks`
    -- an exact count, not a detection-weighted expectation, since these
    are real catalogued Gaia sources, not the raw TRILEGAL population.
    A member outside the region's own pixel set does not enter. Fully
    vectorised (CODING_RULES.md rule 8): one `healpy.ang2pix` call and
    three flattened `bincount`s, no Python loop over members.

    Returns the three histograms and `(n_matched, n_with_ks)`: how many
    candidate members fell inside the region's own pixels at all, and
    how many of those also carried a finite `BP-RP` and so entered the
    2MASS/joint subtraction.
    """
    n_pix = pixels.size
    n_g = g_edges.size - 1
    n_ks = ks_edges.size - 1
    zeros = (np.zeros((n_pix, n_g)), np.zeros((n_pix, n_ks)), np.zeros((n_pix, n_g, n_ks)))
    if members["glon"].size == 0 or n_pix == 0:
        return zeros + (0, 0)

    member_pix = hp.ang2pix(_NSIDE_HPX, members["glon"], members["glat"], nest=True, lonlat=True)
    idx_all = np.searchsorted(pixels, member_pix)
    capped = np.minimum(idx_all, n_pix - 1)
    in_region = pixels[capped] == member_pix
    n_matched = int(np.count_nonzero(in_region))
    if n_matched == 0:
        return zeros + (0, 0)

    pix_idx = capped[in_region]
    g_vals = members["gmag"][in_region]
    bp_rp = members["bp_rp"][in_region]
    ks_vals = ks_from_gaia(g_vals, bp_rp)
    has_ks = np.isfinite(bp_rp) & np.isfinite(ks_vals)
    n_with_ks = int(np.count_nonzero(has_ks))

    finite_g = np.isfinite(g_vals) & (g_vals >= g_edges[0]) & (g_vals < g_edges[-1])
    g_idx = np.searchsorted(g_edges, g_vals, side="right") - 1
    flat_g = pix_idx[finite_g] * n_g + g_idx[finite_g]
    n_g_cluster = np.bincount(flat_g, minlength=n_pix * n_g).reshape(n_pix, n_g).astype(np.float64)

    in_ks_range = has_ks & (ks_vals >= ks_edges[0]) & (ks_vals < ks_edges[-1])
    ks_idx = np.searchsorted(ks_edges, ks_vals, side="right") - 1
    flat_ks = pix_idx[in_ks_range] * n_ks + ks_idx[in_ks_range]
    n_ks_cluster = np.bincount(flat_ks, minlength=n_pix * n_ks).reshape(n_pix, n_ks).astype(np.float64)

    # a member's own (G, Ks) joint bin -- the same 2-D digitised bincount
    # `N_GK_OBS` used before this brief, now only needed here (that
    # histogram's own producer moved to `sky.derived.gaia_twomass_counts`).
    in_gk_range = finite_g & in_ks_range
    flat_gk = (pix_idx[in_gk_range] * n_g + g_idx[in_gk_range]) * n_ks + ks_idx[in_gk_range]
    n_gk_cluster = np.bincount(flat_gk, minlength=n_pix * n_g * n_ks).reshape(n_pix, n_g, n_ks).astype(np.float64)

    return n_g_cluster, n_ks_cluster, n_gk_cluster, n_matched, n_with_ks


# ---------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------

def _read_histograms(config, region):
    path = config_module.product_path(config, "population", "anchors", "histograms", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.anchor_observed: anchor histograms missing for region %r at "
            "%s -- run the 'prior.anchor_tiles' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(
            pixels=np.asarray(f["HPX_PIX_512"][:], dtype=np.int64),
            g_edges=np.asarray(f["G_EDGES"][:], dtype=np.float64),
            ks_edges=np.asarray(f["KS_EDGES"][:], dtype=np.float64),
            n_g_obs=np.asarray(f["N_G_OBS"][:], dtype=np.float64),
            n_ks_obs=np.asarray(f["N_KS_OBS"][:], dtype=np.float64),
        )


def _read_young_stars(config, region, pixels):
    path = config_module.product_path(config, "population", "anchors", "young-stars", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.anchor_observed: young-star anchor expectation missing for region %r at "
            "%s -- run the 'prior.young_stars' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        yp = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        n_g_young = np.asarray(f["N_G_YOUNG"][:], dtype=np.float64)
        n_ks_young = np.asarray(f["N_KS_YOUNG"][:], dtype=np.float64)
        n_young_total = np.asarray(f["N_YOUNG_TOTAL"][:], dtype=np.float64)
    if not np.array_equal(pixels, yp):
        raise ValueError(
            "prior.anchor_observed: %r's young-star product pixel set disagrees with "
            "the anchor histograms' own pixel set -- rerun 'prior.anchor_tiles' or "
            "'prior.young_stars' for this region" % region)
    return n_g_young, n_ks_young, n_young_total


def _read_joint_counts(config, region):
    """`sky/derived/gaia/joint-counts_gaia-twomass_hpx512__<Region>.hdf5`
    (`sesnaimpute.sky.derived.gaia_twomass_counts`): the Gaia archive's
    own Gaia-2MASS crossmatch, per pixel, `G` bin and `Ks` bin."""
    path = f"{config.data_root}/sky/derived/gaia/joint-counts_gaia-twomass_hpx512__{region}.hdf5"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "population.anchor_observed: Gaia-2MASS joint counts missing for region %r at "
            "%s -- run the 'sesnaimpute.sky.derived.gaia_twomass_counts' RUNBOOK line first"
            % (region, path))
    with h5py.File(path, "r") as f:
        return dict(
            pixels=np.asarray(f["HPX_PIX_512"][:], dtype=np.int64),
            g_edges=np.asarray(f["G_EDGES"][:], dtype=np.float64),
            ks_edges=np.asarray(f["KS_EDGES"][:], dtype=np.float64),
            n_gk=np.asarray(f["N_GK"][:], dtype=np.float64),
        )


def _place_joint_on_ks_axis(n_gk_joint, joint_ks_edges, hist_ks_edges):
    """`n_gk_joint` (n_pix, n_g, n_ks_joint), whose Ks axis stops at the
    2MASS marginal's own cut (14.3), placed onto the histograms
    product's own Ks axis -- which a region with a UKIDSS deep extension
    runs past that cut (`population.anchor_tiles._combine_ks_axis`). A
    histogram bin whose own edges match one of the joint product's takes
    that column; a deep bin past 14.3 has no archive crossmatch at all
    and is left at zero, exactly as an unmeasured bin already reads to
    `population.anchor_weights` (`USE_JOINT` false there). Vectorised
    (CODING_RULES.md rule 8): one edge comparison, no Python loop over
    pixels or bins.
    """
    n_pix, n_g, _ = n_gk_joint.shape
    out = np.zeros((n_pix, n_g, hist_ks_edges.size - 1), dtype=np.float64)
    match = np.isclose(hist_ks_edges[:-1, None], joint_ks_edges[None, :-1], atol=1e-6)
    hist_idx, joint_idx = np.nonzero(match)
    out[:, :, hist_idx] = n_gk_joint[:, :, joint_idx]
    return out


def build_region(config, region):
    hist = _read_histograms(config, region)
    pixels, g_edges, ks_edges = hist["pixels"], hist["g_edges"], hist["ks_edges"]
    n_pix = pixels.size

    n_g_young, n_ks_young, n_young_total = _read_young_stars(config, region, pixels)

    joint = _read_joint_counts(config, region)
    if not np.array_equal(pixels, joint["pixels"]):
        raise ValueError(
            "population.anchor_observed: %r's Gaia-2MASS joint counts pixel set disagrees "
            "with the anchor histograms' own pixel set -- rerun "
            "'sesnaimpute.sky.derived.gaia_twomass_counts' or 'population.anchor_tiles' for "
            "this region" % region)
    if not np.array_equal(g_edges, joint["g_edges"]):
        raise ValueError(
            "population.anchor_observed: %r's Gaia-2MASS joint counts G_EDGES disagree with "
            "the anchor histograms' own -- rerun 'sesnaimpute.sky.derived.gaia_twomass_counts' "
            "for this region" % region)

    n_gk_obs = _place_joint_on_ks_axis(joint["n_gk"], joint["ks_edges"], ks_edges)
    n_entering = int(joint["n_gk"].sum())

    n_young_total_safe = np.where(n_young_total > 0, n_young_total, 1.0)
    n_gk_young = (n_g_young[:, :, None] * n_ks_young[:, None, :]
                  / n_young_total_safe[:, None, None])
    n_gk_young = np.where(n_young_total[:, None, None] > 0, n_gk_young, 0.0)

    n_g_obs, n_ks_obs = hist["n_g_obs"], hist["n_ks_obs"]
    n_g_sub = np.maximum(n_g_obs - n_g_young, 0.0)
    n_ks_sub = np.maximum(n_ks_obs - n_ks_young, 0.0)
    n_gk_sub = np.maximum(n_gk_obs - n_gk_young, 0.0)

    floored_g = np.count_nonzero(n_g_obs < n_g_young, axis=1).astype(np.int64)
    floored_ks = np.count_nonzero(n_ks_obs < n_ks_young, axis=1).astype(np.int64)
    floored_gk = np.count_nonzero(n_gk_obs < n_gk_young, axis=(1, 2)).astype(np.int64)

    # the cluster-member subtraction (module docstring, "THE CLUSTER
    # SUBTRACTION"): applied on top of the young-star-subtracted
    # histogram, same floor-at-zero mechanism.
    tiles = anchor_weights_module._read_tiles(config, region)
    if not np.array_equal(pixels, tiles["pixels"]):
        raise ValueError(
            "prior.anchor_observed: %r's tiles product pixel set disagrees with the "
            "anchor histograms' own pixel set -- rerun `prior.anchor_tiles` for this "
            "region" % region)
    clusters = anchor_weights_module._read_hunt_reffert_clusters(config)
    touching = _clusters_touching_region(
        tiles["tile_l_deg"], tiles["tile_b_deg"], tiles["l_star_deg"], clusters)
    cluster_names = clusters["name"][touching]
    members = _read_cluster_members(config, cluster_names)
    n_g_cluster, n_ks_cluster, n_gk_cluster, n_members_matched, n_members_with_ks = \
        _cluster_member_histograms(pixels, members, g_edges, ks_edges)

    n_g_pre_cluster, n_ks_pre_cluster, n_gk_pre_cluster = n_g_sub, n_ks_sub, n_gk_sub
    n_g_sub = np.maximum(n_g_pre_cluster - n_g_cluster, 0.0)
    n_ks_sub = np.maximum(n_ks_pre_cluster - n_ks_cluster, 0.0)
    n_gk_sub = np.maximum(n_gk_pre_cluster - n_gk_cluster, 0.0)

    cluster_floored_g = np.count_nonzero(n_g_pre_cluster < n_g_cluster, axis=1).astype(np.int64)
    cluster_floored_ks = np.count_nonzero(n_ks_pre_cluster < n_ks_cluster, axis=1).astype(np.int64)
    cluster_floored_gk = np.count_nonzero(n_gk_pre_cluster < n_gk_cluster, axis=(1, 2)).astype(np.int64)

    total_g_obs = n_g_obs.sum(axis=1)
    total_ks_obs = n_ks_obs.sum(axis=1)
    subtracted_frac_g = np.where(
        total_g_obs > 0, (total_g_obs - n_g_sub.sum(axis=1)) / np.where(total_g_obs > 0, total_g_obs, 1.0), 0.0)
    subtracted_frac_ks = np.where(
        total_ks_obs > 0, (total_ks_obs - n_ks_sub.sum(axis=1)) / np.where(total_ks_obs > 0, total_ks_obs, 1.0), 0.0)

    return dict(
        pixels=pixels, g_edges=g_edges, ks_edges=ks_edges,
        n_gk_obs=n_gk_obs, n_g_sub=n_g_sub, n_ks_sub=n_ks_sub, n_gk_sub=n_gk_sub,
        floored_g=floored_g, floored_ks=floored_ks, floored_gk=floored_gk,
        subtracted_frac_g=subtracted_frac_g, subtracted_frac_ks=subtracted_frac_ks,
        n_entering=n_entering,
        total_g_obs=float(total_g_obs.sum()), total_ks_obs=float(total_ks_obs.sum()),
        n_clusters_touching=int(np.count_nonzero(touching)),
        n_members_matched=n_members_matched, n_members_with_ks=n_members_with_ks,
        n_g_cluster_total=float(n_g_cluster.sum()), n_ks_cluster_total=float(n_ks_cluster.sum()),
        cluster_floored_g=cluster_floored_g, cluster_floored_ks=cluster_floored_ks,
        cluster_floored_gk=cluster_floored_gk,
    )


def _write_product(config, region, result):
    path = config_module.product_path(config, "population", "anchors", "observed", "hpx512", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.create_dataset("HPX_PIX_512", data=result["pixels"].astype(np.int64))
        f.create_dataset("G_EDGES", data=result["g_edges"])
        f.create_dataset("KS_EDGES", data=result["ks_edges"])
        f.create_dataset("N_G_SUB", data=result["n_g_sub"].astype(np.float32))
        f.create_dataset("N_KS_SUB", data=result["n_ks_sub"].astype(np.float32))
        f.create_dataset("N_GK_SUB", data=result["n_gk_sub"].astype(np.float32))
    return path


def build(config, regions=None):
    """Writes, per region (default: all thirty), `bms/anchors/observed_
    anchors_hpx512__<Region>.hdf5` (module docstring). Prints, per
    region, the number of archive crossmatch rows entering `N_GK_OBS`,
    the fraction of pixels carrying any floored bin, the region-total
    subtracted fraction in `G` and `Ks`, and the cluster-member
    subtraction's own candidate-cluster count, matched-member count and
    floored-bin count.
    """
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in names:
        with progress.Stage("prior.anchor_observed", region) as st:
            result = build_region(config, region)
            path = _write_product(config, region, result)

            any_floored = ((result["floored_g"] > 0) | (result["floored_ks"] > 0)
                            | (result["floored_gk"] > 0))
            frac_floored_pix = float(any_floored.mean()) if any_floored.size else float("nan")
            any_cluster_floored = ((result["cluster_floored_g"] > 0)
                                    | (result["cluster_floored_ks"] > 0)
                                    | (result["cluster_floored_gk"] > 0))
            n_cluster_floored_pix = int(np.count_nonzero(any_cluster_floored))

            total_g_obs, total_ks_obs = result["total_g_obs"], result["total_ks_obs"]
            region_frac_g = ((total_g_obs - float(result["n_g_sub"].sum())) / total_g_obs
                              if total_g_obs > 0 else float("nan"))
            region_frac_ks = ((total_ks_obs - float(result["n_ks_sub"].sum())) / total_ks_obs
                               if total_ks_obs > 0 else float("nan"))
            st.done(path, n_entering=result["n_entering"], frac_pixels_floored=frac_floored_pix)
        print(
            "prior.anchor_observed: %s N_GK_OBS entering=%d "
            "frac_pixels_floored=%.4f region_subtracted_frac G=%.4f Ks=%.4f "
            "clusters_touching=%d members_matched=%d members_with_ks=%d "
            "N_G_CLUSTER_sum=%.1f N_KS_CLUSTER_sum=%.1f cluster_floored_pixels=%d -> %s"
            % (region, result["n_entering"], frac_floored_pix,
               region_frac_g, region_frac_ks, result["n_clusters_touching"],
               result["n_members_matched"], result["n_members_with_ks"],
               result["n_g_cluster_total"], result["n_ks_cluster_total"],
               n_cluster_floored_pix, path))


if __name__ == "__main__":
    run(build)
