"""The cloud-class population sample: YSO, per sightline (SPEC_BMSTP_DRAFT.md
sec. 5.5 "Marks", "Weight"; sec. 2 "the common grid", "region distance and
depth"; IMPLEMENTATION_BMSTP_DRAFT.md sec. 1.2 P3). H2S has no sampler of
its own: sec. 5.6 "Marks" says its `x` is YSO's and its brightness is a
separable region Gaussian, so `bmstp.shapes` reads YSO's own `X_MARGINAL`
and the region's `(LOGSIG_MEAN, LOGSIG_STD)` rather than sampling anything.

`h_YSO(x, F_4.5) = p(x) * p(F_4.5)` is a strict outer product (sec. 5.5
"Marks"): the two marks are independent, so each is built as its own 1-D
shape and the joint grid is formed by `np.outer` at write time.

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
`population.yso`'s exactly.

`p(F_4.5)`, per region (the SAME shape at every sightline of the region):
the YSO register's own templates, weighted by
`template_weights.yso_population_weight` (Dunham et al. 2015's own census
density over each template's `log10 f_ref,4.5,theta` divided by the
library's density of templates in the same quantity, times inclination
uniform in cos i and the evolutionary-class census, sec. 1.4, W54, owner's
ruling 2026-09-09) -- called directly here rather than re-derived, the one
place that weight is formed -- each at `log10 F_4.5 = log10
F_REF_I2,theta - 2 log10(d_r / 1 kpc)`, widened by the cloud's own depth
and the region's distance uncertainty as a Gaussian in `log10 F_4.5`.
"""

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

from sesnaimpute import config as config_module
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


def sample_x(loaded, row, d_front, d_back):
    """YSO's own-sightline depth mark, `p(x)` (sec. 5.5 "Marks"): the
    profile's native cells restricted to those whose distance range
    overlaps `[d_front, d_back]` -- a cell partly inside counts its
    inside fraction -- `x = u(d)` over the surviving cells, mass `p_k *
    du_k` as before, laid in `N_SUB` equally-weighted sub-samples along
    each cell's own `log10 x` segment before the common grid's one-cell
    smoothing (sec. 5.5, sec. 2 "minimum widths"). Returns `(p_x,
    mass_outside, removed_frac)`: `removed_frac` is the fraction of the
    sightline's own (pre-restriction) mass the cloud-interval restriction
    removed, report-only (sec. 5.5's "1-9 percent median, up to 84
    percent")."""
    u_edges = loaded["u_edges"][row]
    d_edges = loaded["d_edges"][row]
    p_u = loaded["p_u"][row]
    u_lo, u_hi = u_edges[:-1], u_edges[1:]
    d_lo, d_hi = d_edges[:-1], d_edges[1:]
    # u is non-decreasing (sec. 5.5); the top cell's edges can differ from
    # 1.0 by a float64 rounding residual (~1e-16) with the wrong sign, so
    # the increment is clipped at zero rather than left to go negative.
    mass = p_u * np.maximum(u_hi - u_lo, 0.0)  # (n_cell,)

    overlap = np.clip(np.minimum(d_hi, d_back) - np.maximum(d_lo, d_front), 0.0, None)
    width = np.maximum(d_hi - d_lo, 1e-300)
    inside_frac = overlap / width
    removed_frac = float(1.0 - np.sum(mass * inside_frac))

    keep = inside_frac > 0.0
    mass_k = (mass * inside_frac)[keep]
    log10x_lo = np.log10(np.maximum(u_lo[keep], _X_FLOOR))
    log10x_hi = np.log10(np.maximum(u_hi[keep], _X_FLOOR))

    log10x = log10x_lo[:, None] + _SUB_T[None, :] * (log10x_hi - log10x_lo)[:, None]
    w = np.broadcast_to((mass_k / N_SUB)[:, None], log10x.shape)
    # edge convention (sec. 2): a mark exactly on a cell edge belongs to
    # the cell below it (`bmstp.grid.bin`'s own nudge, repeated here since
    # this is a standalone 1-D bin, not a call to `grid.bin`).
    log10x_nudged = np.nextafter(log10x.ravel(), -np.inf)
    p_x, mass_outside = _bin1d(log10x_nudged, w.ravel(), grid.LOG10_X_EDGES, sigma_cells=1.0)
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


def sample_f45(config, region, d_r_pc, sigma_d_pc, d_front, d_back):
    """YSO's own brightness mark, `p(F_4.5)` (sec. 5.5 "Marks"), the same
    shape at every sightline of the region: the YSO register's templates,
    weighted by `template_weights.yso_population_weight` (Dunham et al.
    2015's own census density over each template's `log10
    f_ref,4.5,theta` divided by the library's density of templates in the
    same quantity, times inclination uniform in cos i and the
    evolutionary-class census, sec. 1.4, W54, owner's ruling 2026-09-09)
    -- called directly rather than re-derived, so the
    weight formula lives in exactly one place for `build_yso`, this
    function and `bmstp.atlas._yso_register` alike. Each template placed
    at `log10 F_4.5 = log10 F_REF_I2,theta - 2 log10(d_r / 1 kpc)`,
    widened as a Gaussian in `log10 F_4.5` by the cloud's own depth and
    the region's distance uncertainty together, `2 log10(d_back/d_front)
    + 2 sigma_d/(d_r ln 10)` (sec. 5.5 "Marks", using the DOUBLED cloud
    interval, W24b). Returns `(p_f45, mass_outside, width_dex,
    mass_above_top)`: `mass_above_top` is the raw (pre-widening) weight
    above the grid's own top edge (sec. 9's 0.1% bar)."""
    reg = template_weights._read_register(config, "yso")
    names, f_ref_i2 = reg["names"], reg["f_ref"]["I2"]

    # the register's own FREFRAW convention (sec. 3.5): F_REF is raw and
    # must be floored at FLOOR_LINEAR before a log, the same rule
    # `fittp.sweep._register` applies -- 703 of 200,000 YSO templates
    # carry an I2 reference flux of exactly zero (an edge-on disc's own
    # 4.5 micron flux driven to numerical zero), never `-inf` from here.
    register_path = f"{config.inputs['sed_models']}/registers/yso_register.hdf5"
    with h5py.File(register_path, "r") as f:
        floor_linear = f["models/FLOOR_LINEAR"][:].astype(np.float64)

    names_w, weight = template_weights.yso_population_weight(config)
    if names_w.size != names.size or not np.all(names_w == names):
        raise ValueError(
            "sample_cloud.sample_f45: yso_population_weight's row order "
            "disagrees with the yso register")

    log10_f45_theta = (np.log10(np.maximum(f_ref_i2, floor_linear))
                        - 2.0 * np.log10(float(d_r_pc) / 1000.0))

    width_dex = (2.0 * np.log10(float(d_back) / float(d_front))
                 + 2.0 * float(sigma_d_pc) / (float(d_r_pc) * np.log(10.0)))
    sigma_cells = max(1.0, width_dex / grid.D_LOG10_F45)

    p_f45, mass_outside = _bin1d(log10_f45_theta, weight, grid.LOG10_F45_EDGES, sigma_cells)
    above_top = grid.mass_above_top(log10_f45_theta, weight)
    return p_f45, mass_outside, float(width_dex), above_top
