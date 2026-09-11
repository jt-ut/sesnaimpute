"""The cloud-class population sample: YSO, per sightline (SPEC_BMSTP_DRAFT.md
sec. 5.5 "Marks", "Weight"; sec. 2 "the common grid", "region distance and
depth"; IMPLEMENTATION_BMSTP_DRAFT.md sec. 1.2 P3). H2S has no sampler of
its own: sec. 5.6 "Marks" says its `x` is YSO's and its brightness is a
separable region Gaussian, so `bmstp.shapes` reads YSO's own `X_MARGINAL`
and the region's `(LOGSIG_MEAN, LOGSIG_STD)` rather than sampling anything.

A young star's depth and brightness are both functions of its distance
along the sightline, so `h_YSO(x, F_4.5)` is NOT an outer product (sec.
5.5 "Marks"): `GRID_YSO[x, F] = Sigma_k,sub w_k,sub * 1[x = x_k,sub]
* P_ref(F - delta_k,sub) (x) N(sigma_d)` -- each depth sub-sample adds one
shifted-and-smoothed copy of the survey-wide template marginal `P_ref`
(`p_ref_f45`, at the library's own 1 kpc reference distance, unplaced) to
its own row of the grid, `delta_k,sub = -2 log10(d_k,sub / 1 kpc)` the
brightness shift that sub-sample's own distance implies. `bmstp.shapes`
forms this per sightline by grouping sub-samples into their own `log10 x`
row first (mathematically identical, since the shift-then-smooth
convolution is linear and additive over sub-samples in the same row) and
convolving that row's own binned-and-smoothed shift distribution with
`P_ref` in one pass. `X_MARGINAL` (`p(x)` below) is unchanged in value.

`p(x)`, per sightline: reuses `population.yso`'s own vetted embedding-
density construction (`_load_profile_arrays`, `embedding_and_ridge`) at its
FULL resolution -- the profile product it points at, sec. 5.5's "the
profile's cells" -- restricted to the cells whose distance range overlaps
the region's cloud interval `[d_front, d_back]` (sec. 2's "region distance
and depth", REWRITTEN after W24: `cloud_interval_pc` DOUBLES the region
depth product's own `D_LO_PC`/`D_HI_PC` half-widths about `D_PEAK_PC`,
W24b); a cell partly inside counts its own inside fraction. The matching
distance edges are formed the same way `population.yso`'s own tail cell is
(`d_mid_tail = dist_pc[-1] + tail_efold`, sec. 6.3 there): the map's own
`DIST_PC` points, plus one more edge `dist_pc[-1] + 2 * tail_efold` past
the map's reach, so the tail cell's midpoint distance matches
`population.yso`'s exactly. `_cell_subsamples` is `p(x)`'s own machinery
one level down: the raw, unbinned sub-samples (weight, depth, distance)
`p(x)`, `shift_kernel` and `GRID_YSO`'s row-by-row construction all share.

`shift_kernel`'s `K(delta)`, per region (sec. 5.5): the
region's FALLBACK sightline's own depth draw turned into the distribution
of the brightness shift every template rides, `K` independent of the
template -- every template's brightness is `c_theta + delta` -- and
carrying the old cloud-depth-and-distance-uncertainty widening term
itself, so that term is never applied a second time. `template_weights
.build_yso` reads `K` for its own conditional table (sec. 5.5 "Template
weights"); `bmstp.atlas._yso_register`'s Monte Carlo draws `delta` from
the same `K`.
"""

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.bmstp import grid, template_weights
from sesnaimpute.population import yso as yso_module

#: Sub-samples per profile cell, laid evenly along the cell's own segment
#: in `log10 x` (sec. 5.5 "Marks": "in practice, a fixed number of
#: sub-samples per cell").
N_SUB = 16

#: The common grid's own `log10 x` floor (sec. 2), guarding `log10(0)` at
#: a sightline's nearest cell edge (`u = 0` at `d = 0`).
_X_FLOOR = 10.0 ** grid.LOG10_X_EDGES[0]

_SUB_T = (np.arange(N_SUB, dtype=np.float64) + 0.5) / N_SUB  # (N_SUB,)


def _region_profile(config, region):
    """The region's full-resolution embedding density and matching
    distance edges, one vectorised call over every sightline
    (`embedding_and_ridge`, unmodified): `u_edges`/`d_edges` (n_sl,
    n_d+1), `p_u` (n_sl, n_d), `hpx_pix_256` (n_sl,)."""
    profile = yso_module._load_profile_arrays(config, region)
    embed = yso_module.embedding_and_ridge(profile)
    dist_pc = profile["dist_pc"]
    n_d = dist_pc.size
    n_sl = embed["u_edges"].shape[0]
    d_edges = np.empty((n_sl, n_d + 1), dtype=np.float64)
    d_edges[:, :n_d] = dist_pc[None, :]
    d_edges[:, n_d] = dist_pc[-1] + 2.0 * profile["tail_efold_pc"]
    return dict(u_edges=embed["u_edges"], p_u=embed["p_u"], d_edges=d_edges,
                hpx_pix_256=profile["hpx_pix_256"])


def sightline_count(config, region):
    """The region's number of sightlines -- P3's grain axis length."""
    return _region_profile(config, region)["hpx_pix_256"].size


def cloud_interval_pc(config, region):
    """`(d_front, d_back)`, the region's CLOUD INTERVAL (sec. 2 "region
    distance and depth", REWRITTEN after W24, W24b): the dust structure's
    peak distance minus TWICE its lower half-width, to its peak plus TWICE
    its upper half-width -- `d_front = D_PEAK - 2 (D_PEAK - D_LO)`, `d_back
    = D_PEAK + 2 (D_HI - D_PEAK)` -- read directly off the region depth
    product's own `D_PEAK_PC`/`D_LO_PC`/`D_HI_PC` (the bare 16-84 interval
    alone held only two thirds of the structure's own dust by construction
    and cut 19-84% of the YSO placement's mass, W24; the doubled interval
    holds ~95% of it). Floored at the profile's own first cell
    (`DIST_PC[0]`, 0 pc) and capped at its own last reachable edge (the
    map's own edge plus twice the SMALLEST sightline's tail e-folding
    scale -- the same construction `_region_profile` uses to close every
    sightline's own support), so the interval never reaches past what
    every sightline of the region can represent."""
    depth_path = config_module.product_path(config, "sky/derived", "edenhofer", "depth", "region")
    with h5py.File(depth_path, "r") as f:
        names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in f["REGION"][:]]
        if region not in names:
            raise ValueError("sample_cloud.cloud_interval_pc: region %r has no row in %s"
                              % (region, depth_path))
        i = names.index(region)
        d_peak = float(f["D_PEAK_PC"][i])
        d_lo = float(f["D_LO_PC"][i])
        d_hi = float(f["D_HI_PC"][i])
    d_front = d_peak - 2.0 * (d_peak - d_lo)
    d_back = d_peak + 2.0 * (d_hi - d_peak)

    profile_path = config_module.product_path(
        config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    with h5py.File(profile_path, "r") as f:
        dist_first = float(f["DIST_PC"][0])
        dist_last = float(f["DIST_PC"][-1])
        tail_efold_min = float(np.min(f["TAIL_EFOLD_PC"][:]))
    d_front = max(d_front, dist_first)
    d_back = min(d_back, dist_last + 2.0 * tail_efold_min)
    return d_front, d_back


def _bin1d(values, w, edges, sigma_cells):
    """One axis of `bmstp.grid.bin`, alone: the weighted 1-D histogram of
    `values` on `edges`, normalised to `1 - mass_outside`, Gaussian-
    smoothed by `sigma_cells` cells (`mode="constant"`: mass pushed past
    an edge is mass outside the grid, never wrapped), then floored at
    `grid.FLOOR` of its own peak (sec. 2 "minimum widths", "the floor") --
    the two independent pieces of YSO's separable shape are each built
    this way, once per axis."""
    values = np.asarray(values, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    total_weight = w.sum()
    h, _ = np.histogram(values, bins=edges, weights=w)
    if total_weight <= 0:
        return np.full(h.shape, grid.FLOOR), 1.0
    mass_outside = float((total_weight - h.sum()) / total_weight)
    h = h / total_weight
    mass_before = float(h.sum())
    h = gaussian_filter1d(h, sigma=sigma_cells, mode="constant")
    mass_outside += mass_before - float(h.sum())
    h = np.maximum(h, grid.FLOOR * h.max())
    return h, mass_outside


def _cell_subsamples(loaded, row, d_front, d_back):
    """The raw, unbinned sub-samples a sightline track's own cells
    (restricted to `[d_front, d_back]`, a cell partly inside counting its
    inside fraction) supply (sec. 5.5 "Marks", the shift kernel): `N_SUB`
    equally-weighted sub-samples laid along each surviving cell's own
    `log10 x` segment, mass `p_k * du_k` split `N_SUB` ways as before, each
    sub-sample's own distance `d_k,sub` interpolated LINEARLY in `d` along
    the same cell at the same fractional position (the cell's own `(u, d)`
    segment is monotonic in both). Returns `(log10x_nudged, w, d_sub,
    removed_frac)`, all `(n_cell_kept * N_SUB,)` flat except the scalar
    `removed_frac` -- `sample_x` bins `log10x_nudged` into `p_x`;
    `shift_kernel` and `bmstp.shapes._build_one_sightline` turn `d_sub`
    into the brightness shift `delta = -2 log10(d_sub / 1 kpc)`."""
    u_edges = loaded["u_edges"][row]
    d_edges = loaded["d_edges"][row]
    p_u = loaded["p_u"][row]
    u_lo, u_hi = u_edges[:-1], u_edges[1:]
    d_lo, d_hi = d_edges[:-1], d_edges[1:]
    # u is non-decreasing (sec. 5.5); the top cell's edges can differ from
    # 1.0 by a float64 rounding residual (~1e-16) with the wrong sign, so
    # the restriction below clips the increment at zero rather than
    # leaving it to go negative -- `population.yso.restrict_and_
    # renormalize` (moved from here, unchanged, so this product's own
    # numbers do not move): the interval's own restriction of `p_u`,
    # renormalised later by `_bin1d`'s own division, not here.
    mass, inside_frac, removed_frac = yso_module.restrict_and_renormalize(
        p_u, u_lo, u_hi, d_lo, d_hi, d_front, d_back)
    removed_frac = float(removed_frac)

    keep = inside_frac > 0.0
    mass_k = (mass * inside_frac)[keep]
    log10x_lo = np.log10(np.maximum(u_lo[keep], _X_FLOOR))
    log10x_hi = np.log10(np.maximum(u_hi[keep], _X_FLOOR))
    # a cell straddling an interval edge keeps only its own inside
    # fraction's worth of mass (above) but, without this clip, would
    # still spread its sub-samples' own DISTANCE across the cell's full
    # (unrestricted) span -- clipped to `[d_front, d_back]` so every
    # sub-sample's `d_k,sub` (and the brightness shift it implies) stays
    # inside the cloud interval the weight already restricts it to.
    d_lo_k = np.clip(d_lo[keep], d_front, d_back)
    d_hi_k = np.clip(d_hi[keep], d_front, d_back)

    log10x = log10x_lo[:, None] + _SUB_T[None, :] * (log10x_hi - log10x_lo)[:, None]
    d_sub = d_lo_k[:, None] + _SUB_T[None, :] * (d_hi_k - d_lo_k)[:, None]
    w = np.broadcast_to((mass_k / N_SUB)[:, None], log10x.shape)
    # edge convention (sec. 2): a mark exactly on a cell edge belongs to
    # the cell below it (`bmstp.grid.bin`'s own nudge, repeated here since
    # this is a standalone 1-D bin, not a call to `grid.bin`).
    log10x_nudged = np.nextafter(log10x.ravel(), -np.inf)
    return log10x_nudged, w.ravel(), d_sub.ravel(), removed_frac


def sample_x(loaded, row, d_front, d_back):
    """YSO's own-sightline depth mark, `p(x)` (sec. 5.5 "Marks"): the
    profile's native cells restricted to those whose distance range
    overlaps `[d_front, d_back]`, laid in `N_SUB` sub-samples per cell
    (`_cell_subsamples`) and binned on the common `log10 x` grid with the
    usual one-cell smoothing (sec. 2 "minimum widths"). Returns `(p_x,
    mass_outside, removed_frac)`: `removed_frac` is the fraction of the
    sightline's own (pre-restriction) mass the cloud-interval restriction
    removed, report-only (sec. 5.5's "1-9 percent median, up to 84
    percent")."""
    log10x_nudged, w, _d_sub, removed_frac = _cell_subsamples(loaded, row, d_front, d_back)
    p_x, mass_outside = _bin1d(log10x_nudged, w, grid.LOG10_X_EDGES, sigma_cells=1.0)
    return p_x, mass_outside, removed_frac


def restrict_old_x_marginal(loaded, row, old_x_marginal, d_front, d_back):
    """The acceptance check's own reference: the OLD (pre-4.5B)
    `X_MARGINAL`, read off disk before this build overwrites it,
    restricted to the cloud interval and renormalised. Each of the common
    grid's 128 `log10 x` cells' own distance range is approximated by
    inverting the sightline's native `u(d)` mapping (`loaded`'s
    `u_edges`/`d_edges`, both monotonic) at the grid's own edges, then the
    same inside-fraction rule `sample_x` applies at native resolution.
    Report-only: the two constructions restrict at different resolutions
    (native profile cells there, the common grid's own cells here), so
    exact agreement is not expected."""
    u_edges_native = loaded["u_edges"][row]
    d_edges_native = loaded["d_edges"][row]
    grid_x_edges = 10.0 ** grid.LOG10_X_EDGES
    d_at_grid_edges = np.interp(grid_x_edges, u_edges_native, d_edges_native)
    d_lo, d_hi = d_at_grid_edges[:-1], d_at_grid_edges[1:]
    overlap = np.clip(np.minimum(d_hi, d_back) - np.maximum(d_lo, d_front), 0.0, None)
    width = np.maximum(d_hi - d_lo, 1e-300)
    inside_frac = overlap / width
    restricted = np.asarray(old_x_marginal, dtype=np.float64) * inside_frac
    total = float(restricted.sum())
    return restricted / total if total > 0 else restricted


def p_ref_f45(config):
    """`P_ref`, the census-weighted YSO template marginal AT THE
    LIBRARY'S OWN 1 kpc REFERENCE DISTANCE (sec. 5.5 "Joint shape",
    the shift-kernel rule): `Sigma_theta w_theta delta(F - c_theta)` binned ONCE
    on the common `LOG10_F45` grid -- unplaced at any region's distance,
    unwidened by any cloud depth. `w_theta` is
    `template_weights.yso_population_weight`'s survey-wide census weight
    (Dunham et al. 2015's own census density over each template's `log10
    f_ref,4.5,theta` divided by the library's density of templates in the
    same quantity, times inclination uniform in cos i and the
    evolutionary-class census, sec. 1.4, owner's ruling 2026-09-09),
    `c_theta` `template_weights._c_theta`'s own `log10 F_REF_I2,theta`
    (floored at the register's `FLOOR_LINEAR`, sec. 3.5). Every region's
    own distance and every sightline's own cloud depth enter only through
    `shift_kernel`'s convolution kernel, never here: survey-wide,
    independent of region. Returns `p_ref`, summing to the survey's own
    on-grid census share (`Sigma w_theta = 1` by construction, sec. 5.5;
    what falls outside the grid at `c_theta` alone is off by construction,
    before any placement)."""
    reg = template_weights._read_register(config, "yso")
    c_theta = template_weights._c_theta(reg)
    _names_w, w_theta = template_weights.yso_population_weight(config)
    h, _ = np.histogram(c_theta, bins=grid.LOG10_F45_EDGES, weights=w_theta)
    return h


def _fallback_profile(config, region):
    """The `sky.derived.profile` product's own `fallback` group -- the
    region's source-weighted mean direction, sec. 1.4 there -- read into
    the SAME one-sightline shape `_load_profile_arrays` returns for an
    ordinary sightline, so `embedding_and_ridge` (unmodified) can be
    reused directly. `A_INF_K` has no fallback twin: `A_COL_SIGHTLINE_K`
    IS that quantity, already source-weighted (`w @ a_inf` at build, the
    profile module's own `fallback` dict); likewise `RESIDUAL_K` stands
    in for `TAIL_RESIDUAL_K` and the fallback's own scalar `TAIL_EFOLD_PC`
    for the per-sightline array. `RHO_K_PER_PC` is rebuilt off the
    fallback's own `A_CUM_K`/`DIST_PC`, the same finite-difference the
    profile build itself uses (sec. 1.4, `rho_k_per_pc = diff(A_CUM_K) /
    diff(DIST_PC)`)."""
    path = config_module.product_path(
        config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    with h5py.File(path, "r") as f:
        fb = f["fallback"]
        dist_pc = np.asarray(fb["DIST_PC"][:], dtype=np.float64)
        a_cum_k = np.asarray(fb["A_CUM_K"][:], dtype=np.float64)
        a_inf_k = np.array([float(fb["A_COL_SIGHTLINE_K"][()])], dtype=np.float64)
        tail_residual_k = np.array([float(fb["RESIDUAL_K"][()])], dtype=np.float64)
        tail_efold_pc = np.array([float(fb["TAIL_EFOLD_PC"][()])], dtype=np.float64)
    rho_k_per_pc = (np.diff(a_cum_k) / np.diff(dist_pc))[None, :]
    return dict(hpx_pix_256=np.array([-1], dtype=np.int64), dist_pc=dist_pc,
                a_cum_k=a_cum_k[None, :], a_inf_k=a_inf_k, rho_k_per_pc=rho_k_per_pc,
                tail_residual_k=tail_residual_k, tail_efold_pc=tail_efold_pc)


def sigma_d_dex(region):
    """The region's own distance-uncertainty width in `log10 F_4.5`
    (sec. 5.5 "Marks"): `sigma_d = 2 sigma_d,pc / (d_r ln
    10)` -- the one formula `shift_kernel` and `bmstp.shapes.build_cloud`
    (the per-row Gaussian smoothing `GRID_YSO`'s construction applies, the
    SAME width `shift_kernel` convolves `K` with) share, so it lives in
    exactly one place."""
    return 2.0 * float(region.sigma_pc) / (float(region.d_r_pc) * np.log(10.0))


def exact_gaussian_kernel(sigma_dex):
    """The `(2*half_width+1,)` EXACT discrete Gaussian smoothing kernel
    of width `sigma_dex` (floored at one common-grid cell, sec. 2's
    "minimum widths"), each entry the Gaussian's own CDF difference
    across a destination cell offset from the source cell's own center
    (`build_yso`'s Gaussian construction, generalised to a fixed kernel
    since every cell of a uniform grid sees the SAME offset-only
    dependence) -- truncated at +-4 sigma, same as before. NEVER
    `scipy.ndimage.gaussian_filter1d`, whose own discrete kernel is not
    exact enough for the acceptance identity (sec. 9, ruling 11): summed
    against a one-hot spike it differs from the exact CDF by ~0.016 at
    one cell's width, far past float rounding."""
    D = grid.D_LOG10_F45
    sigma = max(float(sigma_dex), D)
    sigma_cells = sigma / D
    half_width = int(np.ceil(4.0 * sigma_cells))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    z_lo = (offsets - 0.5) * D / sigma
    z_hi = (offsets + 0.5) * D / sigma
    return ndtr(z_hi) - ndtr(z_lo)


def shift_kernel(config, region, d_front, d_back):
    """`K(delta)`, the shift kernel (sec. 5.5, ruling 1): the region's
    FALLBACK sightline's own depth draw (`_fallback_profile`,
    `_cell_subsamples`, the SAME cloud interval every sightline uses --
    ruling: no source counts from P1 enter here) turned into the
    distribution of the brightness shift `delta = -2 log10(d / 1 kpc)`
    each sub-sample implies, weight `w_k,sub` carried through, binned RAW
    on the common `LOG10_F45` grid's 0.1 dex cells (`np.histogram`, no
    smoothing) then convolved EXACTLY (`exact_gaussian_kernel`) by the
    region's own distance uncertainty `sigma_d = 2 sigma_d,pc / (d_r ln
    10)` (floored at one cell, sec. 2's "minimum widths"). `K` is
    independent of the template: every template's brightness is `c_theta
    + delta`, so it carries the OLD cloud-depth widening term itself --
    that term must not be applied again downstream. Returns `(K,
    mass_outside)`."""
    r = regions_module.REGIONS_BY_NAME[region]
    fb_profile = _fallback_profile(config, region)
    embed = yso_module.embedding_and_ridge(fb_profile)
    dist_pc = fb_profile["dist_pc"]
    n_d = dist_pc.size
    d_edges = np.empty((1, n_d + 1), dtype=np.float64)
    d_edges[:, :n_d] = dist_pc[None, :]
    d_edges[:, n_d] = dist_pc[-1] + 2.0 * fb_profile["tail_efold_pc"][0]
    loaded_fb = dict(u_edges=embed["u_edges"], p_u=embed["p_u"], d_edges=d_edges)

    _log10x, w, d_sub, _removed = _cell_subsamples(loaded_fb, 0, d_front, d_back)
    delta = -2.0 * np.log10(d_sub / 1000.0)
    sigma_d = sigma_d_dex(r)
    total_weight = float(w.sum())
    raw, _ = np.histogram(delta, bins=grid.LOG10_F45_EDGES, weights=w)
    kernel = exact_gaussian_kernel(sigma_d)
    K = np.convolve(raw, kernel, mode="same")
    mass_outside = float(1.0 - K.sum() / total_weight) if total_weight > 0 else 1.0
    K = K / total_weight if total_weight > 0 else K
    return K, mass_outside
