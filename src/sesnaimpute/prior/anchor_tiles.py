"""The STAR anchor tiles and their two anchors' observed/predicted
magnitude histograms, per nside-512 pixel (SPEC_PRIORS.md section 2.1,
table rows "tile", "anchor counts", "N^{model->obs}" and "Gaia dimming";
reading note 04_star_family.md section B, `read_anchor_counts` through
`pass one -> fractional_gradient -> l_star_optimum -> tiles_from_hpx512`).

Scope. This module answers, per region: what tile a source's nside-512
pixel belongs to, and what the two anchors observed and predicted in
every pixel's own magnitude bins, pooled into one first-pass
observed-over-predicted ratio. It does NOT fit the per-tile-and-bin
weight `W` (SPEC_PRIORS.md section 2.1's `W` row), subtract the young-
star contribution from the observed histograms (the "young stars in the
anchors" row), exclude cluster tiles (the "cluster exclusion" row), or
extrapolate the faint end (the "faint end" row) -- those read the
products this module writes and are a later stage's build.

`pointing_mixture` (reading note 04 section B) is dropped: every region
carries exactly one TRILEGAL pointing (`prior.field_stars`), so the
inverse-square-distance blend across pointings is the identity and
mixing buys nothing.

The Gaia dimming coefficient. `prior.field_stars` carries each raw
star's `K_G_DIFFUSE`/`K_G_DENSE`, Danielski et al. 2018's own `A_G/A_V`
coefficient under the diffuse-ISM and dense-cloud laws (its docstring,
"the per-star Gaia extinction coefficient `k_G = A_G/A_V`"). A star's
local extinction `a_local = A_pix * u` is carried in `A_K` throughout
this project, so this module converts each pure-law coefficient to
`A_G/A_K` by the SAME law's own `A_K/A_V` ratio (`prior.selection.
ak_per_av` at ramp weight 0 and 1) before blending by the section 1.3
ramp -- the K-normalise-then-blend convention `prior.selection.
kappa_hybrid` already uses for the other seven bands. Re-expanding that
blend back through the hybrid law's own `A_K/A_V(w)` (the quarry
`anchor_reweighting.hybrid_kg`'s two-step recipe) and then multiplying
by `a_local / (A_K/A_V)(w)` cancels the re-expansion factor exactly, so
`G_obs = G + a_local * kappa_G(w)` with `kappa_G(w)` already the blended
`A_G/A_K` is the same number in one step.

Products, per region:

  `tiles_anchors_hpx512__<Region>.hdf5` -- `HPX_PIX_512`, `TILE_ID`
  (one row per occupied pixel) and, one row per tile, `TILE_L_DEG`,
  `TILE_B_DEG`, `TILE_N_PIX`, `TILE_OMEGA_DEG2`; root attrs
  `GRANULE="hpx512"`, `L_STAR_DEG`.

  `histograms_anchors_hpx512__<Region>.hdf5` -- `HPX_PIX_512`,
  `A_PIX_K`, `OMEGA_PIX_DEG2`, `G_EDGES`, `KS_EDGES`, `N_G_OBS`,
  `N_G_PRED`, `N_KS_OBS`, `N_KS_PRED` (n_pix, n_bins), `N_GK_PRED`
  (n_pix, n_G_bins, n_Ks_bins) -- the same raw stars, same weights and
  Gaia detection weight as `N_G_PRED`, 2-D digitised on `(G_obs, Ks_obs)`
  onto `G_EDGES`/`KS_EDGES` (SPEC_PRIORS.md section 2.1, "joint and
  marginal bins") -- `RATIO_PASS1`; root attr `GRANULE="hpx512"`.
"""

import os

import h5py
import healpy as hp
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import selection
from sesnaimpute.sky.derived import profile as profile_module

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

NSIDE = 512

#: SPEC_PRIORS.md section 2.1, anchor-counts row: "Gaia DR3 at G < 19".
GAIA_G_CUT_MAG = 19.0

#: Cantat-Gaudin et al. 2023's own published Gaia DR3 50%-completeness
#: limit and sigmoid roll-off width, applied as the anchor's detection
#: probability (spec 2.1: "the published 50% completeness limit ... and
#: roll-off width ... applied as a sigmoid p_G(m)").
GAIA_G_LIM_MAG = 20.5
GAIA_G_ROLLOFF_MAG = 0.5

#: SPEC_PRIORS.md section 2.1, anchor-counts row: "2MASS PSC at
#: Ks < 14.3" -- must equal `sky.derived.twomass_counts.MAG_EDGES[-1]`.
KS_CUT_MAG = 14.3

#: The tile-size bias-variance optimum's own constant (quarry
#: `zone_grid.l_star_optimum`): counting noise on a tile's ratio goes as
#: `1/(Sigma_obs * L^2)`, representativity as `(g*L)^2/12` (the variance
#: of a uniform slope across a tile of side L); minimising their sum in
#: L gives `L* = (12 / (Sigma_obs * g^2))^(1/4)`.
TILE_VARIANCE_TWELVE = 12.0

#: Candidate tile HEALPix nsides `L*` is quantised onto -- every power of
#: two from the whole sky (nside 1) down to the native anchor pixel
#: (nside 512), so a tile is always a whole group of nside-512 pixels.
_CANDIDATE_TILE_NSIDES = np.array([1 << k for k in range(10)], dtype=np.int64)


# ---------------------------------------------------------------------------
# the anchor observables: a raw star's own magnitude at a pixel's column
# ---------------------------------------------------------------------------

def gaia_detection_weight(g_obs):
    """`p_G(G_obs)`, Cantat-Gaudin et al. 2023's published 50% limit and
    roll-off width as a sigmoid (SPEC_PRIORS.md section 2.1)."""
    g_obs = np.asarray(g_obs, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(GAIA_G_LIM_MAG - g_obs) / GAIA_G_ROLLOFF_MAG))


def anchor_observables(dist_pc, g_proxy, ks_mag, k_g_diffuse, k_g_dense,
                        profile_obj, parent_hpx256, a_pix_k, r_diffuse, r_dense):
    """`(G_obs, Ks_obs)` for the region's raw population placed at one
    pixel's own column `a_pix_k`, on its own parent nside-256 sightline.

    `profile.a_of_d(d, hpx_pix=parent, total_column_ak=a_pix_k)` already
    returns the star's own local extinction `a_local = a_pix_k * u(d)`
    (SPEC_PRIORS.md section 1.4): `u` never has to be read back out and
    multiplied through separately. `Ks_obs` dims by `a_local` exactly
    (the project's own currency); `G_obs` dims by `a_local` through the
    star's own diffuse/dense `A_G/A_K` coefficient, blended by the
    section 1.3 ramp at that same local column (module docstring).
    """
    a_local = profile_obj.a_of_d(dist_pc, hpx_pix=parent_hpx256, total_column_ak=a_pix_k)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = selection.law_dense_weight(a_local)
    kappa_g = (1.0 - w) * (k_g_diffuse / r_diffuse) + w * (k_g_dense / r_dense)
    g_obs = g_proxy + a_local * kappa_g
    ks_obs = ks_mag + a_local
    return g_obs, ks_obs


# ---------------------------------------------------------------------------
# pass one: the pooled ratio field, its gradient, and the tile size
# ---------------------------------------------------------------------------

def pooled_ratio_pass1(n_obs, n_pred):
    """Per pixel, the pooled observed-over-predicted ratio over every
    bin both anchors populate, counting-weighted (SPEC_PRIORS.md section
    2.1's "cluster exclusion" row names this exact construction; run
    here per pixel, before any tile exists, rather than per tile).

    Each populated bin's precision is the inverse variance of its log10
    ratio under combined Poisson shot noise from the observed and the
    predicted side, `sigma_log10 = sqrt(1/n_obs + 1/n_pred) / ln(10)`
    (quarry `anchor_reweighting.sigma_stat_frac`, delta-method into the
    log). Pooling in the log keeps a bin found twice too bright and a
    bin found twice too faint symmetric. Returns the linear ratio
    (`nan` where a pixel has no populated bin on either anchor) and each
    pixel's total populated-bin observed count, `fractional_gradient`'s
    own counting weight.
    """
    n_obs = np.asarray(n_obs, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    populated = (n_obs > 0) & (n_pred > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(populated, n_obs / np.where(n_pred > 0, n_pred, 1.0), 1.0)
        log_r = np.log10(ratio)
        sigma_log10 = np.sqrt(1.0 / np.where(n_obs > 0, n_obs, 1.0)
                               + 1.0 / np.where(n_pred > 0, n_pred, 1.0)) / np.log(10.0)
        precision = np.where(populated, 1.0 / sigma_log10 ** 2, 0.0)
    pixel_precision = precision.sum(axis=1)
    has_evidence = pixel_precision > 0
    safe_precision = np.where(has_evidence, pixel_precision, 1.0)
    pooled_log10 = (precision * log_r).sum(axis=1) / safe_precision
    ratio_pass1 = np.where(has_evidence, 10.0 ** pooled_log10, np.nan)
    weight_pix = np.where(populated, n_obs, 0.0).sum(axis=1)
    return ratio_pass1, weight_pix


def fractional_gradient(l_deg, b_deg, ratio, weight):
    """The measured ratio field's fractional gradient, per degree
    (quarry `anchor_reweighting.fractional_gradient`, lifted unchanged).

    A count-weighted linear fit of the field on each sky axis
    separately, each slope divided by the weighted mean of the field --
    a fractional gradient, since the representativity term
    `l_star_optimum` reads it into is itself a fractional variance. The
    reported gradient is the larger of the two axes': a tile has one
    side, sized against the steeper direction.
    """
    l_deg = np.asarray(l_deg, dtype=np.float64)
    b_deg = np.asarray(b_deg, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    ok = np.isfinite(ratio) & np.isfinite(weight) & (weight > 0)
    if np.count_nonzero(ok) < 3:
        return {"g_l": 0.0, "g_b": 0.0, "g": 0.0, "f_mean": float("nan"),
                "n_cells": int(np.count_nonzero(ok))}
    w = weight[ok]
    y = ratio[ok]
    f_mean = float(np.sum(w * y) / np.sum(w))
    out = {"f_mean": f_mean, "n_cells": int(np.count_nonzero(ok))}
    # longitude is circular: unwrap about the first measured pixel so a
    # region crossing l=0 is a local field, not a 360-degree jump
    l_unwrapped = (l_deg[0] + ((l_deg - l_deg[0] + 180.0) % 360.0) - 180.0
                   if l_deg.size else l_deg)
    for name, x_all in (("g_l", l_unwrapped), ("g_b", b_deg)):
        x = x_all[ok]
        x_bar = np.sum(w * x) / np.sum(w)
        var = np.sum(w * (x - x_bar) ** 2)
        if var <= 0 or f_mean == 0.0:
            out[name] = 0.0
        else:
            cov = np.sum(w * (x - x_bar) * (y - f_mean))
            out[name] = float(abs(cov / var) / abs(f_mean))
    out["g"] = float(max(out["g_l"], out["g_b"]))
    return out


def l_star_optimum(sigma_obs_deg2, g_fractional_per_deg):
    """`L* = (12 / (Sigma_obs * g^2))^(1/4)` degrees (quarry
    `zone_grid.l_star_optimum`, lifted; see `TILE_VARIANCE_TWELVE`). A
    flat field (`g == 0`) or an unmeasured density (`Sigma_obs == 0`)
    returns `inf`: there is no scale at which refining helps, and the
    tile clips to the whole region."""
    sigma = float(sigma_obs_deg2)
    g = float(g_fractional_per_deg)
    if sigma < 0.0:
        raise ValueError("l_star_optimum: Sigma_obs must be non-negative")
    denom = sigma * g * g
    if denom <= 0.0:
        return float("inf")
    return (TILE_VARIANCE_TWELVE / denom) ** 0.25


def tiles_from_hpx512(pixels, sigma_obs_deg2, g_fractional_per_deg, floor_deg, omega_pix_deg2):
    """Whole groups of the region's own occupied nside-512 pixels, sized
    by `l_star_optimum` and floored by `floor_deg` (quarry
    `zone_grid.tiles_from_hpx512`'s HEALPix branch, lifted): `L*` is
    quantised UP to the coarsest power-of-two HEALPix nside whose own
    pixel size is still >= `L*` floored, so a tile is a whole group of
    nside-512 pixels never finer than the map's own counting grain.
    Grouping by `pixel // (512/tile_nside)**2` is exact NESTED-scheme
    bit-parentage. A region occupying only part of one such coarse pixel
    collapses to that one tile, without a separate clip step, because
    only the region's own occupied pixels are grouped.

    `floor_deg` (this module: `hp.nside2resol(512)` in degrees, the
    anchor counts' own binning grain) is the finest a tile can be
    quantised to; reading note 04B's fuller "map floor" (tied to the
    ~1-degree Edenhofer sightline grid) is not carried here, since
    quantising to whole nside-512 pixels already prevents anything finer
    than what this module's own inputs resolve.
    """
    l_opt = l_star_optimum(sigma_obs_deg2, g_fractional_per_deg)
    l_floored = max(l_opt, float(floor_deg))
    resolutions_deg = np.degrees([hp.nside2resol(int(ns)) for ns in _CANDIDATE_TILE_NSIDES])
    eligible = _CANDIDATE_TILE_NSIDES[resolutions_deg >= l_floored]
    tile_nside = int(eligible[-1]) if eligible.size else 1
    factor = (NSIDE // tile_nside) ** 2
    parent = pixels // factor
    tile_pixels, tile_of_pix = np.unique(parent, return_inverse=True)
    tile_l_deg, tile_b_deg = hp.pix2ang(tile_nside, tile_pixels, nest=True, lonlat=True)
    tile_n_pix = np.bincount(tile_of_pix, minlength=tile_pixels.size).astype(np.int64)
    tile_omega_deg2 = tile_n_pix.astype(np.float64) * float(omega_pix_deg2)
    l_star_deg = float(np.degrees(hp.nside2resol(tile_nside)))
    return dict(
        tile_of_pix=tile_of_pix.astype(np.int64),
        tile_l_deg=np.asarray(tile_l_deg, dtype=np.float64),
        tile_b_deg=np.asarray(tile_b_deg, dtype=np.float64),
        tile_n_pix=tile_n_pix,
        tile_omega_deg2=tile_omega_deg2,
        l_star_deg=l_star_deg,
        l_star_optimum_deg=l_opt,
        n_tiles=int(tile_pixels.size),
    )


# ---------------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------------

def _read_field_stars_raw(config, region):
    path = config_module.product_path(config, "bms", "trilegal", "field-stars", "region", region=region)
    with h5py.File(path, "r") as f:
        omega_sim_deg2 = float(f.attrs["OMEGA_SIM_DEG2"])
        raw = f["RAW"]
        out = dict(
            dist_pc=raw["DIST_PC"][:].astype(np.float64),
            g_proxy=raw["G_PROXY"][:].astype(np.float64),
            ks_mag=raw["KS_MAG"][:].astype(np.float64),
            k_g_diffuse=raw["K_G_DIFFUSE"][:].astype(np.float64),
            k_g_dense=raw["K_G_DENSE"][:].astype(np.float64),
        )
    return out, omega_sim_deg2


def _region_pixel_columns(config, region):
    """The region's occupied nside-512 pixels (the granule map's own
    set, sorted unique), each pixel's parent nside-256 sightline (its
    member sources' own `HPX_PIX_256`, which is single-valued per
    nside-512 pixel by construction), and each pixel's mean adopted
    column `A_PIX_K` over its own SESNA sources.
    """
    rs = access.region_slice(config, region)
    pix512_src = np.asarray(rs["hpx_pix_512"], dtype=np.int64)
    pix256_src = np.asarray(rs["hpx_pix_256"], dtype=np.int64)
    adopted_path = config_module.product_path(config, "sky/derived", "adopted", "column", "source", region=region)
    a_col_src = access.per_source(config, region, adopted_path, ["A_COL_K"])["A_COL_K"].astype(np.float64)

    pixels, first_idx, inverse = np.unique(pix512_src, return_index=True, return_inverse=True)
    parent256 = pix256_src[first_idx]
    n_src_per_pix = np.bincount(inverse, minlength=pixels.size).astype(np.float64)
    a_pix = np.bincount(inverse, weights=a_col_src, minlength=pixels.size) / n_src_per_pix
    return pixels, parent256, a_pix


def _read_anchor_counts(config, region, pixels):
    """The two anchors' own per-pixel histograms, restricted to the
    module's magnitude cuts: Gaia's bins with upper edge <= 19
    (`GAIA_G_CUT_MAG`), 2MASS's whole grid (already built to close at
    `KS_CUT_MAG`). Fails if either product's own pixel set disagrees
    with the granule map's (`_region_pixel_columns`) -- the join every
    later array in this module assumes.
    """
    gaia_path = config_module.product_path(config, "sky/derived", "gaia", "counts", "hpx512", region=region)
    twomass_path = config_module.product_path(config, "sky/derived", "twomass", "counts", "hpx512", region=region)
    with h5py.File(gaia_path, "r") as f:
        gaia_pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        gaia_edges_full = np.asarray(f["MAG_EDGES"][:], dtype=np.float64)
        gaia_n = np.asarray(f["N"][:], dtype=np.float64)
    with h5py.File(twomass_path, "r") as f:
        ks_pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        ks_edges = np.asarray(f["MAG_EDGES"][:], dtype=np.float64)
        ks_n = np.asarray(f["N"][:], dtype=np.float64)
    if not (np.array_equal(pixels, gaia_pix) and np.array_equal(pixels, ks_pix)):
        raise ValueError(
            f"anchor_tiles: {region!r}'s granule-map pixel set disagrees with the "
            "gaia_counts/twomass_counts products' own pixel set -- rerun those two "
            "RUNBOOK lines for this region")
    if abs(float(ks_edges[-1]) - KS_CUT_MAG) > 1e-6:
        raise ValueError(
            f"anchor_tiles: {twomass_path!r}'s MAG_EDGES ends at {ks_edges[-1]!r}, "
            f"not the spec cut {KS_CUT_MAG} -- rerun sky.derived.twomass_counts")
    n_g_edges = int(np.searchsorted(gaia_edges_full, GAIA_G_CUT_MAG + 1e-9))
    g_edges = gaia_edges_full[:n_g_edges]
    n_g_obs = gaia_n[:, : n_g_edges - 1]
    return g_edges, n_g_obs, ks_edges, ks_n


def _predicted_histograms(config, profile_obj, parent256, a_pix, raw, g_edges, ks_edges,
                           r_diffuse, r_dense, weight_star):
    """`(N_G_PRED, N_KS_PRED, N_GK_PRED)`: every raw star's own anchor
    observables at each pixel's own column and parent sightline
    (`anchor_observables`), binned and weighted by the Gaia detection
    probability (Ks carries none -- 2MASS's cut is treated as complete
    to `KS_CUT_MAG`), then scaled by each star's `Ω_pix/Ω_sim` share.
    `N_GK_PRED` (n_pix, n_G_bin, n_Ks_bin) is the same population, same
    Gaia-weight, digitised jointly on `(G_obs, Ks_obs)` onto the same two
    edge arrays (SPEC_PRIORS.md section 2.1, "joint and marginal bins").
    Vectorised over the whole raw population per pixel; the Python loop
    is over pixels only, run in threads (profile evaluation and
    histogramming are numpy/C and release the GIL, and the raw
    population and profile are shared read-only rather than repickled
    per pixel).
    """
    dist_pc = raw["dist_pc"]
    g_proxy = raw["g_proxy"]
    ks_mag = raw["ks_mag"]
    k_g_diffuse = raw["k_g_diffuse"]
    k_g_dense = raw["k_g_dense"]

    def _one_pixel(parent, a_p):
        g_obs, ks_obs = anchor_observables(
            dist_pc, g_proxy, ks_mag, k_g_diffuse, k_g_dense,
            profile_obj, int(parent), float(a_p), r_diffuse, r_dense)
        p_g = gaia_detection_weight(g_obs)
        n_g = np.histogram(g_obs, bins=g_edges, weights=p_g)[0]
        n_ks = np.histogram(ks_obs, bins=ks_edges)[0]
        n_gk = np.histogram2d(g_obs, ks_obs, bins=[g_edges, ks_edges], weights=p_g)[0]
        return n_g, n_ks, n_gk

    results = Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(_one_pixel)(parent256[i], a_pix[i]) for i in range(parent256.size))
    n_g_pred = np.stack([r[0] for r in results]) * weight_star
    n_ks_pred = np.stack([r[1] for r in results]) * weight_star
    n_gk_pred = np.stack([r[2] for r in results]) * weight_star
    return n_g_pred, n_ks_pred, n_gk_pred


def _acceptance_check_one_pixel(profile_obj, parent256, a_pix, raw, g_edges, r_diffuse, r_dense,
                                 weight_star, n_g_pred_row, pixel_index=0):
    """CODING_RULES.md rule 11: the algebraic identity the brief names,
    on one pixel -- the stored histogram must equal a direct re-sum of
    `1[10 <= G_obs < 19] * p_G * Ω_pix/Ω_sim` over the raw population,
    to 1e-9 relative. Returns `(max_abs_rel_dev, n_stars_below_g10)`.
    """
    g_obs, _ = anchor_observables(
        raw["dist_pc"], raw["g_proxy"], raw["ks_mag"], raw["k_g_diffuse"], raw["k_g_dense"],
        profile_obj, int(parent256[pixel_index]), float(a_pix[pixel_index]), r_diffuse, r_dense)
    p_g = gaia_detection_weight(g_obs)
    in_range = (g_obs >= g_edges[0]) & (g_obs < GAIA_G_CUT_MAG)
    direct_sum = float(np.sum(p_g[in_range]) * weight_star)
    stored_sum = float(n_g_pred_row.sum())
    denom = max(abs(stored_sum), 1e-30)
    rel_dev = abs(direct_sum - stored_sum) / denom
    n_below = int(np.count_nonzero(g_obs < g_edges[0]))
    return rel_dev, n_below


def build_region(config, region):
    raw, omega_sim_deg2 = _read_field_stars_raw(config, region)
    pixels, parent256, a_pix = _region_pixel_columns(config, region)
    g_edges, n_g_obs, ks_edges, n_ks_obs = _read_anchor_counts(config, region, pixels)

    profile_obj = profile_module.read(config, region)
    r_diffuse = float(selection.ak_per_av(config, 0.0))
    r_dense = float(selection.ak_per_av(config, 1.0))
    omega_pix_deg2 = float(hp.nside2pixarea(NSIDE, degrees=True))
    weight_star = omega_pix_deg2 / omega_sim_deg2

    n_g_pred, n_ks_pred, n_gk_pred = _predicted_histograms(
        config, profile_obj, parent256, a_pix, raw, g_edges, ks_edges, r_diffuse, r_dense, weight_star)

    rel_dev, n_below_g10 = _acceptance_check_one_pixel(
        profile_obj, parent256, a_pix, raw, g_edges, r_diffuse, r_dense, weight_star, n_g_pred[0])

    n_obs_pooled = np.concatenate([n_g_obs, n_ks_obs], axis=1)
    n_pred_pooled = np.concatenate([n_g_pred, n_ks_pred], axis=1)
    ratio_pass1, weight_pix = pooled_ratio_pass1(n_obs_pooled, n_pred_pooled)

    l_deg, b_deg = hp.pix2ang(NSIDE, pixels, nest=True, lonlat=True)
    grad = fractional_gradient(l_deg, b_deg, ratio_pass1, weight_pix)

    total_obs = float(n_g_obs.sum() + n_ks_obs.sum())
    sigma_obs_deg2 = total_obs / (pixels.size * omega_pix_deg2)
    floor_deg = float(np.degrees(hp.nside2resol(NSIDE)))
    tiles = tiles_from_hpx512(pixels, sigma_obs_deg2, grad["g"], floor_deg, omega_pix_deg2)

    return dict(
        region=region, pixels=pixels, a_pix=a_pix, omega_pix_deg2=omega_pix_deg2,
        g_edges=g_edges, ks_edges=ks_edges,
        n_g_obs=n_g_obs, n_g_pred=n_g_pred, n_ks_obs=n_ks_obs, n_ks_pred=n_ks_pred,
        n_gk_pred=n_gk_pred,
        ratio_pass1=ratio_pass1, tiles=tiles,
        sigma_obs_deg2=sigma_obs_deg2, gradient=grad,
        n_raw=raw["dist_pc"].size, omega_sim_deg2=omega_sim_deg2,
        acceptance_rel_dev=rel_dev, acceptance_n_below_g10=n_below_g10,
    )


def write_tiles(config, region, result):
    pixels, tiles = result["pixels"], result["tiles"]
    path = config_module.product_path(config, "bms", "anchors", "tiles", "hpx512", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.attrs["L_STAR_DEG"] = tiles["l_star_deg"]
        f.create_dataset("HPX_PIX_512", data=pixels.astype(np.int64))
        f.create_dataset("TILE_ID", data=tiles["tile_of_pix"].astype(np.int64))
        f.create_dataset("TILE_L_DEG", data=tiles["tile_l_deg"])
        f.create_dataset("TILE_B_DEG", data=tiles["tile_b_deg"])
        f.create_dataset("TILE_N_PIX", data=tiles["tile_n_pix"])
        f.create_dataset("TILE_OMEGA_DEG2", data=tiles["tile_omega_deg2"])


def write_histograms(config, region, result):
    path = config_module.product_path(config, "bms", "anchors", "histograms", "hpx512", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_pix = result["pixels"].size
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.create_dataset("HPX_PIX_512", data=result["pixels"].astype(np.int64))
        f.create_dataset("A_PIX_K", data=result["a_pix"].astype(np.float32))
        f.create_dataset("OMEGA_PIX_DEG2", data=np.full(n_pix, result["omega_pix_deg2"], dtype=np.float64))
        f.create_dataset("G_EDGES", data=result["g_edges"])
        f.create_dataset("KS_EDGES", data=result["ks_edges"])
        f.create_dataset("N_G_OBS", data=result["n_g_obs"])
        f.create_dataset("N_G_PRED", data=result["n_g_pred"])
        f.create_dataset("N_KS_OBS", data=result["n_ks_obs"])
        f.create_dataset("N_KS_PRED", data=result["n_ks_pred"])
        f.create_dataset("N_GK_PRED", data=result["n_gk_pred"])
        f.create_dataset("RATIO_PASS1", data=result["ratio_pass1"])


def build(config, regions=None):
    """Writes the tiles and histograms products for `regions` (default:
    all thirty), one file pair per region."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        result = build_region(config, region)
        write_tiles(config, region, result)
        write_histograms(config, region, result)
        tiles = result["tiles"]
        ratio_g = result["n_g_obs"].sum() / max(result["n_g_pred"].sum(), 1e-30)
        ratio_ks = result["n_ks_obs"].sum() / max(result["n_ks_pred"].sum(), 1e-30)
        print(f"anchor_tiles: {region}: {result['pixels'].size} pixels, "
              f"L*={tiles['l_star_deg']:.4f} deg (optimum {tiles['l_star_optimum_deg']:.4f}), "
              f"{tiles['n_tiles']} tiles, gradient g={result['gradient']['g']:.4f}/deg, "
              f"obs/pred Gaia={ratio_g:.3f} 2MASS={ratio_ks:.3f}, "
              f"acceptance rel.dev={result['acceptance_rel_dev']:.2e} "
              f"({result['acceptance_n_below_g10']} raw stars brighter than G={result['g_edges'][0]})")


if __name__ == "__main__":
    run(build)
