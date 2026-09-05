"""The YSO law count and the per-sightline shape (SPEC_PRIORS.md section 6.1,
6.3; IMPLEMENTATION.md section 3, the YSO row).

Two products, independent of each other and of the mass-based selection
(section 6.2, built elsewhere):

`law_count` (section 6.1): disk-bearing young stars form in proportion to
the square of the cloud's own column, `N_law = kappa * (d_r*pi/180)^2 *
max(A_s - p_r, 0)^2`, with `kappa` Lada, Lombardi et al. 2013's
area-sampled Orion A+B level (their section 4.1, 12 young stars per pc^2
per mag^2 of A_K on the Megeath+2012 Class I+II census at NICEST's 3'
resolution), transferred to the two beams by the measured beam term:
10.0 at Herschel's 36.3", 12.9 at Planck's 5.03'. `p_r`, the region's own
diffuse pedestal, is the 10th percentile of its adopted columns -- the
cloud's own column is what the law was fitted to, not the total column
with two kiloparsecs of unrelated diffuse dust folded in. `build` writes
one 30-row product with `p_r`, the region distance and the pc^2/deg^2
factor it multiplies, and the pedestal-removed fraction of the law count
(section 6.4 item 4).

The shape (section 6.3): a young star sits at a depth drawn from the
cloud's own gas density, `p(u) ~ rho_gas(d(u))^(1/2)` where
`u = A(d)/A(inf)` (Parmentier & Pfalzner 2013's 3-D form of the observed
Sigma_YSO ~ Sigma_gas^2 surface law), on the profile's own piecewise-linear
cells; the far-field tail past the 3-D map's edge (section 1.4) is one
further cell carrying the residual column, so `u` reaches 1 on every
sightline. The brightness axis is the same placement read on distance:
`log10 B = -2 log10(d / 1 kpc)`, Gaussian in `log10 B` about a
rho_gas^(3/2)-weighted ridge fitted along the same ray, width the ridge's
own conditional residual. Per column-grid node, the `a`-marginal folds
in the column-measurement kernel (section 1.2) as a scale mixture over
its quadrature, `p(a|A) = sum_q w_q * p_u(a/T_q) / T_q`. `build` writes
one product per region, sightline granule: `U_EDGES`/`P_U` (the shape
itself), `RIDGE_*` (the closed-form conditional brightness density), and
`A_GRID`/`A_MARGINAL` (the node marginals the prior table blends between).
`A_GRID` per node spans 0 to the weighted `(1 - A_GRID_MASS_TOL)` point of
the node's own kernel quadrature in `T`, not the quadrature's full
numerical extent (most of which carries negligible weight); the bound
this leaves on `A_MARGINAL` mass beyond the grid is printed once at
build (`_node_quadrature`'s `worst_excluded_frac`, a property of the
kernel and column grid alone) and must stay under `A_GRID_MASS_TOL`.

No selection (section 6.2) and no library enter either product (C3):
neither reads a template register or an IMF.
"""

import os

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid
from sesnaimpute.prior import kernel as kernel_module
from sesnaimpute.sky.derived import herschel_column as sky_herschel_column
from sesnaimpute.sky.derived import planck_column as sky_planck_column

# ====================================================================
# 6.1 -- the law count
# ====================================================================

#: Lada, Lombardi et al. 2013 section 4.1: 12 young stars pc^-2 mag^-2 of
#: A_K at NICEST's 3' resolution, fixed to Megeath+2012's Orion A+B Class
#: I+II census, transferred to the survey's two column-measurement beams
#: by the measured beam term (SPEC_PRIORS.md section 6.1's "three routes"
#: table): 36.3" (Herschel Gould Belt Survey) and 5.03' (Planck).
KAPPA_HERSCHEL = 10.0
KAPPA_PLANCK = 12.9

#: The region's own 10th percentile of adopted column, `p_r` (SPEC_PRIORS.md
#: section 6.1): the cloud's own column is the total column less this
#: diffuse pedestal, floored at zero.
PEDESTAL_PERCENTILE = 10.0

#: Pokhrel+2020's cloud-to-cloud scatter of the normalisation: the
#: uncertainty on any one region's law level, reported and never
#: marginalised (SPEC_PRIORS.md section 6.1).
LAW_BAND_DEX = 0.36

#: Provenance codes of the adopted column (SPEC_PRIORS.md section 1.1):
#: 0 Herschel, 1 Planck.
PROVENANCE_HERSCHEL = 0
PROVENANCE_PLANCK = 1

_ADOPTED_COLUMN_CACHE = {}


def pc2_per_deg2(d_r_pc):
    """The pc^2-per-deg^2 conversion at distance `d_r_pc`: one degree on
    the sky subtends `d_r_pc * pi/180` pc there, so its area factor is
    that length squared (SPEC_PRIORS.md section 6.1)."""
    return (np.asarray(d_r_pc, dtype=float) * np.pi / 180.0) ** 2


def _adopted_columns(config, region):
    """`(a_col, provenance)`, every source of `region`, in catalogue row
    order -- cached, since both `law_count` (per call) and the law
    product build read the same array."""
    key = (id(config), region)
    cached = _ADOPTED_COLUMN_CACHE.get(key)
    if cached is not None:
        return cached
    path = config_module.product_path(config, "sky/derived", "adopted",
                                       "column", "source", region=region)
    cols = access.per_source(config, region, path,
                              ["A_COL_K", "A_COL_PROVENANCE"])
    out = (np.asarray(cols["A_COL_K"], dtype=float),
           np.asarray(cols["A_COL_PROVENANCE"]))
    _ADOPTED_COLUMN_CACHE[key] = out
    return out


def pedestal_k(config, region):
    """`p_r`: the 10th percentile of `region`'s own adopted-column
    distribution (SPEC_PRIORS.md section 6.1)."""
    a_col, _ = _adopted_columns(config, region)
    return float(np.percentile(a_col, PEDESTAL_PERCENTILE))


def law_count(config, region, a_col, provenance):
    """`N_law`, young stars deg^-2, for arrays of adopted column and arm
    (SPEC_PRIORS.md section 6.1):

        N_law = kappa_arm * (d_r*pi/180)^2 * max(a_col - p_r, 0)^2
    """
    a_col = np.asarray(a_col, dtype=float)
    provenance = np.asarray(provenance)
    p_r = pedestal_k(config, region)
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    kappa = np.where(provenance == PROVENANCE_HERSCHEL,
                      KAPPA_HERSCHEL, KAPPA_PLANCK)
    a_cloud = np.maximum(a_col - p_r, 0.0)
    return kappa * pc2_per_deg2(d_r_pc) * a_cloud ** 2


def _law_row(config, region):
    """One region's law-product row, plus the pedestal-removed fraction
    of the law count (SPEC_PRIORS.md section 6.4 item 4): the built law
    count (with the pedestal subtracted) against what the same sources
    would give the total column, unsubtracted."""
    a_col, provenance = _adopted_columns(config, region)
    p_r = pedestal_k(config, region)
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    pc2 = float(pc2_per_deg2(d_r_pc))
    kappa = np.where(provenance == PROVENANCE_HERSCHEL,
                      KAPPA_HERSCHEL, KAPPA_PLANCK)
    n_with_pedestal = law_count(config, region, a_col, provenance)
    n_without_pedestal = kappa * pc2 * a_col ** 2
    removed_frac = 1.0 - float(np.sum(n_with_pedestal)
                                / np.sum(n_without_pedestal))
    return dict(region=region, pedestal_k=p_r, d_r_pc=float(d_r_pc),
                pc2_per_deg2=pc2, pedestal_removed_frac=removed_frac)


def _write_law_product(config, regions, rows):
    """Writes the 30-row law product, in place for `regions`
    (CODING_RULES.md 5c), plus the three region-independent constants
    (`KAPPA_HERSCHEL`, `KAPPA_PLANCK`, `LAW_BAND_DEX`) as the same
    root-level scalar datasets every build has always written -- created
    once, since their value never depends on which regions ran."""
    path = config_module.product_path(config, "bms", "yso", "law", "region")
    tables_module.update_rows(
        path, regions,
        {
            "PEDESTAL_K": np.array([r["pedestal_k"] for r in rows], dtype=np.float64),
            "D_R_PC": np.array([r["d_r_pc"] for r in rows], dtype=np.float64),
            "PC2_PER_DEG2": np.array([r["pc2_per_deg2"] for r in rows], dtype=np.float64),
            "PEDESTAL_REMOVED_FRAC": np.array(
                [r["pedestal_removed_frac"] for r in rows], dtype=np.float64),
        },
        granule="region")
    with tables_module.open_product(path, granule="region") as f:
        for name, value in (("KAPPA_HERSCHEL", KAPPA_HERSCHEL),
                             ("KAPPA_PLANCK", KAPPA_PLANCK),
                             ("LAW_BAND_DEX", LAW_BAND_DEX)):
            if name not in f:
                f.create_dataset(name, data=np.float64(value))
    return path


# ====================================================================
# 6.1 (area form) -- the law count integrated over an anchor pixel's area
# ====================================================================
#
# SPEC_PRIORS.md section 2.1's "young stars in the anchors" row needs "the
# law count integrated over the tile", not a source-sampled mean: the
# anchor subtraction is an AREA count, and `N_law` is convex (squared) in
# column, so the mean of SOURCE-level law counts under-runs the true area
# integral -- SESNA's own sources avoid the densest gas, so they are a
# biased-low sample of the pixel's own column field. Section 6.4 item 1's
# per-cloud check needs the same integral, so it is served once here.

#: The anchor histograms' own HEALPix resolution (`prior.anchor_tiles`,
#: `prior.young_stars`): the nside this integral is evaluated at.
NSIDE_ANCHOR = 512

_KERNEL_CACHE = {}


def _load_kernel(config):
    """The composed column kernel (`prior.kernel.load`), cached per config
    like this module's other survey-wide reads (`_ADOPTED_COLUMN_CACHE`):
    `law_area_integral` is called once per region and the kernel's own
    build reads every region's adopted columns, so an uncached reload
    would cost O(regions^2)."""
    key = id(config)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = kernel_module.load(config)
    return _KERNEL_CACHE[key]


def _map_block_a_k(path):
    """One HGBS map's own valid cells, block-reduced to
    `sky.derived.planck_column.BLOCK_TARGET_ARCSEC` (~12", already
    well-sampled relative to Herschel's own 36.3" beam -- the same cell
    size `planck_column` degrades HGBS maps to for its calibration,
    reused rather than the map's raw few-arcsec pixel grid, which a timed
    survey build cannot afford to carry through a coordinate transform
    whole): `(hpx512, a_k)`, the block's own nside-512 NESTED galactic
    pixel and its `A_K` (`sky.derived.herschel_column`'s currency),
    vectorised over the whole map at once (rule 8)."""
    data, wcs, pixscale = sky_herschel_column._open_hgbs_map(path)
    finite = np.isfinite(data) & (data > 0)
    f = max(1, int(round(sky_planck_column.BLOCK_TARGET_ARCSEC / pixscale)))
    bs, bn, _ny2, _nx2 = sky_planck_column._block_reduce(data, finite, f)
    bmean = np.where(bn > 0, bs / np.maximum(bn, 1), np.nan)
    by, bx = np.mgrid[0:bmean.shape[0], 0:bmean.shape[1]]
    sel = bn > 0
    if not sel.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    xc = bx[sel] * f + (f - 1) / 2.0
    yc = by[sel] * f + (f - 1) / 2.0
    ra, dec = wcs.wcs_pix2world(xc.astype(np.float64), yc.astype(np.float64), 0)
    gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    hpx = hp.ang2pix(NSIDE_ANCHOR, gal.l.deg, gal.b.deg, nest=True, lonlat=True)
    a_k = bmean[sel].astype(np.float64) * sky_herschel_column.NH2_TO_AK
    return hpx.astype(np.int64), a_k


def _region_bbox_icrs(hpx_pix_512):
    """(ra_min, ra_max, dec_min, dec_max) of a set of nside-512 NESTED
    galactic pixel centres -- the same bbox-overlap screen
    `sky.derived.herschel_column._boxes_overlap` runs on, so only maps
    that can possibly touch this pixel set are opened."""
    l_deg, b_deg = hp.pix2ang(NSIDE_ANCHOR, hpx_pix_512, nest=True, lonlat=True)
    icrs = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic").icrs
    ra, dec = icrs.ra.deg, icrs.dec.deg
    return float(ra.min()), float(ra.max()), float(dec.min()), float(dec.max())


def _herschel_pixel_stats(config, pix_sorted, pedestal_k_value):
    """`(sum_sq, count)`, each `(n_pix,)`, aligned to `pix_sorted`
    (ascending): the sum and count of `max(A_cell - p_r, 0)^2` over every
    HGBS map cell (`_map_block_a_k`) whose own nside-512 pixel is in the
    set, over every map overlapping the set's own footprint. Maps are the
    largest independent iterator here (rule 8) and are parallelised with
    joblib; a pixel that two overlapping map reductions both touch is not
    deduplicated -- disclosed: HGBS reductions rarely overlap, and the
    effect on a pixel mean is second order against the gain this area
    integral makes over the source-sampled mean it replaces."""
    maps = sky_herschel_column._map_list(config)
    headers = {m["name"]: sky_herschel_column._map_header(m["path"]) for m in maps}
    rbox = _region_bbox_icrs(pix_sorted)
    candidates = [m for m in maps
                  if sky_herschel_column._boxes_overlap(headers[m["name"]]["bbox"], rbox, pad=0.1)]

    results = Parallel(n_jobs=-1)(delayed(_map_block_a_k)(m["path"]) for m in candidates)

    n_pix = pix_sorted.size
    sum_sq = np.zeros(n_pix)
    count = np.zeros(n_pix)
    for hpx, a_k in results:
        if hpx.size == 0:
            continue
        loc = np.searchsorted(pix_sorted, hpx)
        capped = np.minimum(loc, max(n_pix - 1, 0))
        matched = (n_pix > 0) & (pix_sorted[capped] == hpx)
        if not matched.any():
            continue
        idx = capped[matched]
        a_cloud = np.maximum(a_k[matched] - pedestal_k_value, 0.0)
        sum_sq += np.bincount(idx, weights=a_cloud ** 2, minlength=n_pix)
        count += np.bincount(idx, weights=np.ones(idx.size), minlength=n_pix)
    return sum_sq, count


def _planck_parent_column(config, parent256):
    """`A_K` of every requested nside-256 pixel (`parent256`), from the
    survey-wide Planck sightline product
    (`sky.derived.planck_column.build_column`) -- the fallback arm's own
    column, for pixels the Herschel maps do not reach."""
    path = config_module.product_path(config, "sky/derived", "planck",
                                       "column", "sightline")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.yso: Planck sightline column missing at %s -- run the "
            "'sky.derived.planck_column' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)
    order = np.argsort(sl_pix)
    sl_pix_sorted, a_k_sorted = sl_pix[order], a_k[order]
    loc = np.searchsorted(sl_pix_sorted, parent256)
    capped = np.minimum(loc, max(sl_pix_sorted.size - 1, 0))
    matched = (sl_pix_sorted.size > 0) & (sl_pix_sorted[capped] == parent256)
    if not np.all(matched):
        raise ValueError(
            "prior.yso: an anchor pixel's parent nside-256 sightline is "
            "absent from the Planck column product")
    return a_k_sorted[capped]


def law_area_integral(config, region, hpx_pix_512):
    """`N_law` integrated over each requested nside-512 pixel's own area
    (SPEC_PRIORS.md section 2.1's "young stars in the anchors" row, "the
    law count integrated over the tile"; section 6.4 item 1's per-cloud
    check reuses it). Returns young stars deg^-2, aligned to
    `hpx_pix_512`, in `law_count`'s own convention (a consumer multiplies
    by the pixel's own solid angle for a count) -- but genuinely
    area-averaged over the column MAP, not over the pixel's own catalogued
    sources, which avoid the densest gas and under-run a convex (squared)
    law.

    Herschel-covered pixels (any HGBS map cell falls inside,
    `_herschel_pixel_stats` at the map's own beam-scale resolution):

        kappa_H * mean_over_map_cells[ max(A_cell - p_r, 0)^2 ]

    No coverage-fraction weighting: a pixel only partly inside the HGBS
    mosaic still carries Gaia/2MASS detections across its WHOLE area,
    since the anchor histograms are all-sky (SPEC_PRIORS.md section 2.1)
    and never gated on the Herschel footprint; the covered cells' own mean
    is the best estimate available for the pixel's whole area and stands
    for it, rather than being scaled down by a coverage fraction that
    would zero out real young stars the anchors do see.

    Elsewhere, the Planck sightline column `A_P` of the parent nside-256
    pixel, with the sub-beam variance `prior.kernel.Kernel.second_moment`
    supplies (the true-column dispersion the Planck beam hides,
    SPEC_PRIORS.md section 1.2):

        kappa_P * ((A_P - p_r)^2 + Var(T | A_P))     for A_P > p_r, else 0
    """
    hpx_pix_512 = np.asarray(hpx_pix_512, dtype=np.int64)
    order = np.argsort(hpx_pix_512)
    pix_sorted = hpx_pix_512[order]

    p_r = pedestal_k(config, region)
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    pc2 = pc2_per_deg2(d_r_pc)
    kappa_h_full = KAPPA_HERSCHEL * pc2
    kappa_p_full = KAPPA_PLANCK * pc2

    sum_sq, count = _herschel_pixel_stats(config, pix_sorted, p_r)
    herschel_covered = count > 0
    herschel_value = kappa_h_full * np.divide(
        sum_sq, count, out=np.zeros_like(sum_sq), where=herschel_covered)

    a_p = _planck_parent_column(config, pix_sorted >> 2)
    sigma2_sub = _load_kernel(config).second_moment(a_p)
    a_cloud_p = a_p - p_r
    planck_value = np.where(a_cloud_p > 0.0,
                             kappa_p_full * (a_cloud_p ** 2 + sigma2_sub), 0.0)

    value_sorted = np.where(herschel_covered, herschel_value, planck_value)
    out = np.empty_like(value_sorted)
    out[order] = value_sorted
    return out


# ====================================================================
# 6.3 -- the shape: embedding density, ridge, kernel-convolved marginal
# ====================================================================

#: `rho_YSO ~ rho_gas^alpha` (Parmentier & Pfalzner 2013, A&A 549, A132):
#: the 3-D exponent behind the observed Sigma_YSO ~ Sigma_gas^2 surface
#: law. The embedding density per unit `u` is `rho_gas^(alpha-1)`; the
#: ridge weight is `rho_gas^alpha` (SPEC_PRIORS.md section 6.3).
ALPHA_YSO = 1.5

#: The brightness axis's reference distance (SPEC_PRIORS.md section 6.3):
#: `log10 B(d) = -2 log10(d / 1 kpc)`, the fitter's library reference.
D_REF_PC = 1000.0

#: A density floor guarding the alpha-1 power against an exact zero cell
#: (none occur in the measured profiles; kept as the same defensive floor
#: the profile's own `RHO_K_PER_PC` construction uses).
_RHO_FLOOR = np.finfo(np.float64).tiny

#: The two column-measurement arms the kernel is tabulated at
#: (SPEC_PRIORS.md section 1.1/1.2).
MAP_CLASSES = ("herschel", "planck")

#: The `a`-marginal's own tabulation resolution (IMPLEMENTATION.md
#: section 3, YSO row: "per node ... A_GRID (n_node, 256)").
N_A_GRID = 256

#: `A_GRID`'s own tabulation fidelity bar (IMPLEMENTATION.md section 2's
#: column-grid bar, 0.002 relative L1, restated for this axis): the grid
#: spans only out to the node's kernel-quadrature weighted (1 - tol)
#: point of `T`, not the quadrature's full numerical extent, most of
#: which (section 1.2's wide `_t_grid` window) carries negligible weight.
A_GRID_MASS_TOL = 1.0e-4

#: Sightlines per joblib block (rule 8: parallelise the largest
#: iterator; a block amortises the kernel-quadrature arrays' overhead
#: across a handful of sightlines per dispatched task).
SIGHTLINE_BLOCK = 16


def _profile_path(config, region):
    return config_module.product_path(config, "sky/derived", "edenhofer",
                                       "profile", "sightline", region=region)


def _load_profile_arrays(config, region):
    """The raw sightline-profile arrays this shape is built from
    (`sky.derived.profile.build`'s own product, SPEC_PRIORS.md section
    1.4): `A_CUM_K` already normalised so its own edge never exceeds
    `A_INF_K`, and the far-field tail's residual column and e-folding
    scale past that edge."""
    path = _profile_path(config, region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.yso: sightline profile missing for region %r at %s -- "
            "run the 'sky.derived.profile' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(
            hpx_pix_256=np.asarray(f["HPX_PIX_256"][:], dtype=np.int64),
            dist_pc=np.asarray(f["DIST_PC"][:], dtype=np.float64),
            a_cum_k=np.asarray(f["A_CUM_K"][:], dtype=np.float64),
            a_inf_k=np.asarray(f["A_INF_K"][:], dtype=np.float64),
            rho_k_per_pc=np.asarray(f["RHO_K_PER_PC"][:], dtype=np.float64),
            tail_residual_k=np.asarray(f["TAIL_RESIDUAL_K"][:], dtype=np.float64),
            tail_efold_pc=np.asarray(f["TAIL_EFOLD_PC"][:], dtype=np.float64),
        )


def embedding_and_ridge(profile):
    """The whole region's `u` edges, embedding density, and a-B ridge in
    one vectorised pass -- no Python loop over sightlines, cells or nodes
    (SPEC_PRIORS.md section 6.3).

    The 3-D map's own cells (`RHO_K_PER_PC`, `n_d-1` of them) are joined
    by one more cell carrying the far-field tail's residual column, so
    `u` reaches 1 on every sightline (section 1.4). That cell's own
    characteristic density and depth stand in for its exponential decay
    (`TAIL_RESIDUAL_K`, `TAIL_EFOLD_PC`): the mean of an exponential
    population with that e-folding scale sits one scale length past the
    map's edge, and its density there is the residual column spread over
    that same scale length -- an analytic stand-in disclosed here, not a
    second placement law.

    Returns a dict of `(n_sl, ...)` arrays: `u_edges` (n_sl, n_cell+1),
    `p_u` (n_sl, n_cell) normalised over u in [0, 1], `u_median` (report,
    section 6.3's "spread of the per-sightline u medians"), and the ridge
    `intercept`/`slope`/`resid_sigma`/`corr` (n_sl,).
    """
    dist_pc = profile["dist_pc"]
    a_cum = profile["a_cum_k"]
    a_inf = profile["a_inf_k"]
    tail_residual = profile["tail_residual_k"]
    tail_efold = profile["tail_efold_pc"]
    n_sl, n_d = a_cum.shape

    d_widths = np.diff(dist_pc)                                   # (n_d-1,)
    rho_map = np.maximum(profile["rho_k_per_pc"], _RHO_FLOOR)      # (n_sl, n_d-1)
    rho_tail = np.maximum(tail_residual / tail_efold, _RHO_FLOOR)  # (n_sl,)
    rho_full = np.concatenate([rho_map, rho_tail[:, None]], axis=1)   # (n_sl, n_d)
    width_full = np.concatenate(
        [np.broadcast_to(d_widths, (n_sl, n_d - 1)), tail_efold[:, None]],
        axis=1)                                                    # (n_sl, n_d)

    # u = A(d)/A(inf) (section 1.4): the map's own cumulative edges,
    # followed by the sightline's total column at u = 1 exactly, so the
    # residual cell closes the support.
    a_edges = np.concatenate([a_cum, a_inf[:, None]], axis=1)      # (n_sl, n_d+1)
    u_edges = a_edges / a_inf[:, None]
    u_edges[:, -1] = 1.0
    u_widths = np.diff(u_edges, axis=1)                            # (n_sl, n_d)
    a_rep = 0.5 * (a_edges[:, :-1] + a_edges[:, 1:])                # (n_sl, n_d)

    d_mid_map = 0.5 * (dist_pc[:-1] + dist_pc[1:])                  # (n_d-1,)
    d_mid_tail = dist_pc[-1] + tail_efold                           # (n_sl,)
    d_mid = np.concatenate(
        [np.broadcast_to(d_mid_map, (n_sl, n_d - 1)), d_mid_tail[:, None]],
        axis=1)                                                    # (n_sl, n_d)
    log10_b = -2.0 * np.log10(d_mid / D_REF_PC)

    # the embedding density per unit u (section 6.3): rho_gas^(alpha-1).
    density_raw = rho_full ** (ALPHA_YSO - 1.0)
    mass = np.sum(density_raw * u_widths, axis=1, keepdims=True)
    p_u = density_raw / mass

    # the median u per sightline (report, section 6.3): interpolated off
    # the same piecewise-constant density's own cumulative mass.
    cdf = np.concatenate(
        [np.zeros((n_sl, 1)), np.cumsum(density_raw * u_widths, axis=1) / mass],
        axis=1)                                                    # (n_sl, n_d+1)
    idx = np.clip(np.sum(cdf <= 0.5, axis=1) - 1, 0, n_d - 1)
    cdf_lo = np.take_along_axis(cdf, idx[:, None], axis=1)[:, 0]
    cdf_hi = np.take_along_axis(cdf, (idx + 1)[:, None], axis=1)[:, 0]
    u_lo = np.take_along_axis(u_edges, idx[:, None], axis=1)[:, 0]
    u_hi = np.take_along_axis(u_edges, (idx + 1)[:, None], axis=1)[:, 0]
    frac = np.where(cdf_hi > cdf_lo,
                     (0.5 - cdf_lo) / np.maximum(cdf_hi - cdf_lo, 1e-300), 0.0)
    u_median = u_lo + frac * (u_hi - u_lo)

    # the ridge (section 6.3): the rho^alpha-weighted least-squares line
    # of log10 B on a -- the same population weight the embedding density
    # is built from, so the two are one placement read on two axes.
    w = rho_full ** ALPHA_YSO * width_full
    wn = w / np.sum(w, axis=1, keepdims=True)
    mean_a = np.sum(wn * a_rep, axis=1)
    mean_b = np.sum(wn * log10_b, axis=1)
    var_a = np.sum(wn * (a_rep - mean_a[:, None]) ** 2, axis=1)
    var_b = np.sum(wn * (log10_b - mean_b[:, None]) ** 2, axis=1)
    cov = np.sum(wn * (a_rep - mean_a[:, None]) * (log10_b - mean_b[:, None]), axis=1)
    slope = cov / var_a
    intercept = mean_b - slope * mean_a
    resid_var = np.maximum(var_b - slope * cov, 0.0)
    corr = np.where(var_b > 0, cov / np.sqrt(var_a * var_b), np.nan)

    return dict(u_edges=u_edges, p_u=p_u, u_median=u_median,
                ridge_intercept=intercept, ridge_slope=slope,
                ridge_resid_sigma=np.sqrt(resid_var), ridge_corr=corr)


def _majority_map_class(config, region, sl_pix):
    """Per sightline, the map class (`herschel`/`planck`) of the majority
    of its own sources' adopted-column arm (SPEC_PRIORS.md section 6.3's
    tabulation, "map class from the sightline's majority arm"). A
    sightline the granule map admits by Spitzer coverage alone, with no
    catalogued source of its own, has no arm to poll and falls back to
    Planck, the all-sky arm (section 1.1)."""
    src_pix = access.region_slice(config, region)["hpx_pix_256"]
    _, provenance = _adopted_columns(config, region)
    loc = np.searchsorted(sl_pix, src_pix)
    capped = np.minimum(loc, max(sl_pix.size - 1, 0))
    if src_pix.size and not np.all(sl_pix[capped] == src_pix):
        raise ValueError(
            "prior.yso: %r has a source whose own pixel is absent from its "
            "region's sightline profile" % region)
    is_herschel = (provenance == PROVENANCE_HERSCHEL).astype(float)
    n_herschel = np.bincount(loc, weights=is_herschel, minlength=sl_pix.size)
    n_total = np.bincount(loc, minlength=sl_pix.size)
    majority_herschel = (n_total > 0) & (n_herschel * 2 > n_total)
    return np.where(majority_herschel, "herschel", "planck")


def _t_cutoff(t, w, tol):
    """`(t_cut, excluded_frac)` per row: `t_cut` is the smallest `T`
    beyond which the row's kernel-quadrature weight is below `tol` of
    its total -- the weighted `(1 - tol)` point of `T`; `excluded_frac`
    is the weight actually left beyond it, `<= tol` by construction. `t`
    is already ascending (`kernel._t_grid`'s own construction), so its
    cumulative weight is a proper CDF and the cutoff is the first grid
    point clearing `1 - tol`."""
    cdf = np.cumsum(w, axis=1)
    cdf = cdf / cdf[:, -1:]
    idx = np.argmax(cdf >= (1.0 - tol), axis=1)
    rows = np.arange(t.shape[0])
    return t[rows, idx], 1.0 - cdf[rows, idx]


def _node_quadrature(config, nodes_arr):
    """`(a_grid, quad_by_class, worst_excluded_frac)`: the composed
    column kernel's quadrature at every column-grid node, for both map
    classes (SPEC_PRIORS.md section 1.2) -- built once for the whole
    build, since the kernel depends only on the node's column value and
    the arm, never on the sightline (so this fidelity number is a
    property of the kernel and the column grid alone, the same for
    every region). `a_grid[k]` runs 0 to the larger of the two arms'
    weighted `(1 - A_GRID_MASS_TOL)` point of `T` at node `k`: the
    quadrature's own `_t_grid` window (section 1.2) is widened for the
    low-column, one-sided tail and spans 100-150x the node's column, but
    almost all of that width carries negligible weight, so tabulating
    `a` out to the raw numerical extent (the old `t.max()`) put most of
    `A_GRID`'s 256 points where the density is zero. Spanning to the
    mass cutoff instead concentrates the grid where the density lives.

    `worst_excluded_frac`, printed by `build` (Part 0's required number):
    the largest, over every node and arm, of the kernel-quadrature weight
    left beyond that arm's own cutoff -- an exact upper bound (not a
    grid-resolution estimate) on the share of any sightline's
    `A_MARGINAL` mass that falls beyond `A_GRID`'s edge, since a
    sightline's marginal at `a > t_cut` can only be built from the
    excluded quadrature components (T_q > t_cut) and each contributes at
    most its own weight. Guaranteed `< A_GRID_MASS_TOL` by the cutoff's
    own construction.
    """
    kern = kernel_module.load(config)
    n_node = nodes_arr.size
    # a small (n_node * 2) bounded loop over a fixed, survey-wide design
    # grid, not over sources or sightlines; `Kernel.nodes` has no batched
    # form (rule 8), so the iterator itself is parallelised with joblib
    # threads, sharing the one loaded `Kernel` rather than re-pickling it.
    jobs = [(map_class, k) for map_class in MAP_CLASSES for k in range(n_node)]
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(kern.nodes)(float(nodes_arr[k]), map_class) for map_class, k in jobs)
    quad_by_class = {}
    t_cut = np.zeros(n_node)
    worst_excluded_frac = 0.0
    for i, map_class in enumerate(MAP_CLASSES):
        block = results[i * n_node:(i + 1) * n_node]
        t = np.stack([r[0] for r in block], axis=0)
        w = np.stack([r[1] for r in block], axis=0)
        quad_by_class[map_class] = (t, w)
        cut, excluded = _t_cutoff(t, w, A_GRID_MASS_TOL)
        t_cut = np.maximum(t_cut, cut)
        worst_excluded_frac = max(worst_excluded_frac, float(excluded.max()))
    a_grid = np.linspace(np.zeros(n_node), t_cut, N_A_GRID, axis=1)
    return a_grid, quad_by_class, worst_excluded_frac


def _sightline_block(u_edges_blk, p_u_blk, map_class_blk, a_grid, quad_by_class):
    """One block's `A_MARGINAL`: for every sightline in the block,
    vectorised over every column-grid node, every kernel-quadrature
    point and every `a`-grid point at once (no Python loop below the
    sightline) -- `p(a|A) = sum_q w_q * p_u(a/T_q) / T_q`
    (SPEC_PRIORS.md section 6.3), `p_u` the sightline's own
    piecewise-constant embedding density located by one vectorised
    `searchsorted` into its `u` edges.
    """
    n_blk, n_node, n_grid = u_edges_blk.shape[0], a_grid.shape[0], a_grid.shape[1]
    out = np.empty((n_blk, n_node, n_grid), dtype=np.float32)
    for i in range(n_blk):
        edges = u_edges_blk[i]
        p_u = p_u_blk[i]
        t_q, w_q = quad_by_class[map_class_blk[i]]           # (n_node, n_quad)
        x = a_grid[:, None, :] / t_q[:, :, None]              # (n_node, n_quad, n_grid)
        bin_idx = np.searchsorted(edges, x.ravel(), side="right").reshape(x.shape) - 1
        bin_idx = np.clip(bin_idx, 0, p_u.size - 1)
        valid = (x >= 0.0) & (x <= 1.0)
        density = np.where(valid, p_u[bin_idx] / t_q[:, :, None], 0.0)
        marginal = np.sum(w_q[:, :, None] * density, axis=1)  # (n_node, n_grid)
        # algebraic acceptance (SPEC_PRIORS.md section 6.3 tabulation):
        # every row integrates to 1 on its own a_grid, by construction
        # (the T-quadrature's own weight sums to 1, section 1.2, and
        # A_GRID's cutoff leaves only A_GRID_MASS_TOL of it out).
        norm = np.trapz(marginal, a_grid, axis=1)[:, None]
        out[i] = (marginal / norm).astype(np.float32)
    return out


def _write_shape_product(path, hpx_pix_256, sightline_id, embed, a_grid, a_marginal):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        f.create_dataset("HPX_PIX_256", data=hpx_pix_256)
        f.create_dataset("SIGHTLINE_ID", data=sightline_id)
        f.create_dataset("U_EDGES", data=embed["u_edges"])
        f.create_dataset("P_U", data=embed["p_u"])
        f.create_dataset("U_MEDIAN", data=embed["u_median"])
        f.create_dataset("RIDGE_INTERCEPT", data=embed["ridge_intercept"])
        f.create_dataset("RIDGE_SLOPE", data=embed["ridge_slope"])
        f.create_dataset("RIDGE_WIDTH", data=embed["ridge_resid_sigma"])
        f.create_dataset("RIDGE_CORR", data=embed["ridge_corr"])
        f.create_dataset("A_GRID", data=a_grid.astype(np.float32))
        f.create_dataset("A_MARGINAL", data=a_marginal)


def _sightline_id_lookup(config, region, sl_pix):
    """`SIGHTLINE_ID` per profile row, matched against the granule map's
    own per-source column (IMPLEMENTATION.md section 1). A row the
    granule map admits by coverage alone, with no catalogued source
    sitting in it, carries no survey-wide sightline id at all and is
    marked -1 -- informational only; the prior table joins every product
    through `HPX_PIX_256` (`granules.access.per_source`), never through
    this id."""
    rs = access.region_slice(config, region)
    src_pix, src_sid = rs["hpx_pix_256"], rs["sightline_id"]
    order = np.argsort(src_pix)
    src_pix_sorted, src_sid_sorted = src_pix[order], src_sid[order]
    loc = np.searchsorted(src_pix_sorted, sl_pix)
    capped = np.minimum(loc, max(src_pix_sorted.size - 1, 0))
    matched = (src_pix_sorted.size > 0) & (src_pix_sorted[capped] == sl_pix)
    return np.where(matched, src_sid_sorted[capped], -1).astype(np.int64)


def build_shape(config, region, nodes_arr, a_grid, quad_by_class):
    """Writes one region's `prior/yso/prior_yso_sightline` product
    (SPEC_PRIORS.md section 6.3): the embedding shape and a-B ridge for
    every occupied sightline, and the kernel-convolved a-marginal at
    every column-grid node, blocked over sightlines via joblib.
    """
    profile = _load_profile_arrays(config, region)
    sl_pix = profile["hpx_pix_256"]
    embed = embedding_and_ridge(profile)
    map_class = _majority_map_class(config, region, sl_pix)
    sightline_id = _sightline_id_lookup(config, region, sl_pix)

    n_sl = sl_pix.size
    starts = list(range(0, n_sl, SIGHTLINE_BLOCK))
    blocks = Parallel(n_jobs=-1)(
        delayed(_sightline_block)(
            embed["u_edges"][s:s + SIGHTLINE_BLOCK],
            embed["p_u"][s:s + SIGHTLINE_BLOCK],
            map_class[s:s + SIGHTLINE_BLOCK],
            a_grid, quad_by_class)
        for s in starts)
    a_marginal = (np.concatenate(blocks, axis=0) if blocks
                  else np.empty((0, nodes_arr.size, N_A_GRID), dtype=np.float32))

    path = config_module.product_path(config, "bms", "yso", "prior",
                                       "sightline", region=region)
    _write_shape_product(path, sl_pix, sightline_id, embed, a_grid, a_marginal)
    return path, embed["u_median"], embed["ridge_resid_sigma"]


def build(config, regions=None):
    """Writes, per region, the YSO shape product (section 6.3), and one
    30-row (or subset) law product (section 6.1) over `regions`
    (default: all thirty). Prints `A_GRID`'s fidelity number once
    (Part 0, a property of the kernel and column grid alone, not of any
    one region): the worst-case fraction of any node's kernel-quadrature
    weight left beyond `A_GRID`'s cutoff, which bounds how much of any
    sightline's `A_MARGINAL` mass can fall beyond the grid; must stay
    under `A_GRID_MASS_TOL`."""
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    nodes_arr = column_grid.nodes(config)
    a_grid, quad_by_class, worst_excluded_frac = _node_quadrature(config, nodes_arr)
    print("prior.yso: A_GRID worst-node kernel-quadrature mass beyond the "
          "grid = %.3e (bar %.1e)" % (worst_excluded_frac, A_GRID_MASS_TOL))

    law_rows = []
    for region in names:
        build_shape(config, region, nodes_arr, a_grid, quad_by_class)
        law_rows.append(_law_row(config, region))
    _write_law_product(config, names, law_rows)


if __name__ == "__main__":
    run(build)
