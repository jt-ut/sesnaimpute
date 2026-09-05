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

No selection (section 6.2) and no library enter either product (C3):
neither reads a template register or an IMF.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid
from sesnaimpute.prior import kernel as kernel_module

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


def _write_law_product(config, rows):
    path = config_module.product_path(config, "bms", "yso", "law", "region")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "region"
        f.create_dataset("REGION", data=np.array(
            [r["region"] for r in rows], dtype="S64"))
        f.create_dataset("PEDESTAL_K", data=np.array(
            [r["pedestal_k"] for r in rows], dtype=np.float64))
        f.create_dataset("D_R_PC", data=np.array(
            [r["d_r_pc"] for r in rows], dtype=np.float64))
        f.create_dataset("PC2_PER_DEG2", data=np.array(
            [r["pc2_per_deg2"] for r in rows], dtype=np.float64))
        f.create_dataset("PEDESTAL_REMOVED_FRAC", data=np.array(
            [r["pedestal_removed_frac"] for r in rows], dtype=np.float64))
        f.create_dataset("KAPPA_HERSCHEL", data=np.float64(KAPPA_HERSCHEL))
        f.create_dataset("KAPPA_PLANCK", data=np.float64(KAPPA_PLANCK))
        f.create_dataset("LAW_BAND_DEX", data=np.float64(LAW_BAND_DEX))
    return path


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


def _node_quadrature(config, nodes_arr):
    """`(a_grid, quad_by_class)`: the composed column kernel's quadrature
    at every column-grid node, for both map classes (SPEC_PRIORS.md
    section 1.2) -- built once for the whole build, since the kernel
    depends only on the node's column value and the arm, never on the
    sightline. `a_grid[k]` runs 0 to the largest `T` either arm's
    quadrature places any weight on at node `k` (IMPLEMENTATION.md
    section 3: "from 0 to the node's own column times the kernel's
    largest T/A"), which is exactly that quadrature's own `t.max()`
    since `T/A * A = T`; every `a = T*u` with `u <= 1` is then inside
    the grid by construction.
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
    t_max = np.zeros(n_node)
    for i, map_class in enumerate(MAP_CLASSES):
        block = results[i * n_node:(i + 1) * n_node]
        t = np.stack([r[0] for r in block], axis=0)
        w = np.stack([r[1] for r in block], axis=0)
        quad_by_class[map_class] = (t, w)
        t_max = np.maximum(t_max, t.max(axis=1))
    a_grid = np.linspace(np.zeros(n_node), t_max, N_A_GRID, axis=1)
    return a_grid, quad_by_class


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
        # every row integrates to 1 on its own a_grid, by construction.
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
    (default: all thirty)."""
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    nodes_arr = column_grid.nodes(config)
    a_grid, quad_by_class = _node_quadrature(config, nodes_arr)

    law_rows = []
    for region in names:
        build_shape(config, region, nodes_arr, a_grid, quad_by_class)
        law_rows.append(_law_row(config, region))
    _write_law_product(config, law_rows)


if __name__ == "__main__":
    run(build)
