"""The cloud-class population sample: YSO, per sightline (SPEC_BMSTP_DRAFT.md
sec. 5.5 "Marks", "Weight"; IMPLEMENTATION_BMSTP_DRAFT.md sec. 3 row 1.5).
H2S has no sampler of its own: sec. 5.6's "Marks" says its `x` is YSO's and
its brightness is a separable region Gaussian, so `bmstp.shapes` reads
YSO's own `X_MARGINAL` and the region's `(LOGSIG_MEAN, LOGSIG_STD)`
(`population/h2s/prior_h2s_region__R.hdf5`) rather than sampling anything.

Reuses `population.yso`'s own vetted embedding-density construction
(`_load_profile_arrays`, `embedding_and_ridge`) at its FULL resolution
(before that module's own storage coarsening to 32 cells) -- the profile
product it points at, sec. 5.5's "the profile's cells `(d_k, u_k)`" --
rather than re-deriving `rho_gas^(alpha-1)` here. `embedding_and_ridge`
gives `u` edges and the density per unit `u`; the matching DISTANCE edges
are formed the same way its own tail cell is (`d_mid_tail = dist_pc[-1] +
tail_efold`, sec. 6.3): the map's own `DIST_PC` points, plus one more edge
`dist_pc[-1] + 2 * tail_efold` past the map's reach, so the tail cell's
midpoint distance matches `population.yso`'s exactly.
"""

import numpy as np

from sesnaimpute.bmstp import grid
from sesnaimpute.population import yso as yso_module

#: Sub-samples per profile cell, laid evenly along the cell's own segment
#: in `(log10 x, log10 B)` (sec. 5.5 "Marks": "in practice, a fixed number
#: of sub-samples per cell"; IMPLEMENTATION_BMSTP_DRAFT.md sec. 3 row 1.5).
N_SUB = 16

#: The common grid's own `log10 x` floor (sec. 2), guarding `log10(0)` at
#: a sightline's nearest cell edge (`u = 0` at `d = 0`).
_X_FLOOR = 10.0 ** grid.LOG10_X_EDGES[0]

#: The distance, in pc, at which `log10 B = -2 log10(d / D_REF_PC)` reaches
#: the star family's brightness axis's own top edge (sec. 2: origin -6.0,
#: 120 cells of 0.1 dex, so the top edge is +6.0) -- the matching floor on
#: distance, guarding `log10(0)` at `d = 0`.
_B_TOP = grid.LOG10_B_ORIGIN_TEMPLATE + grid.N_B * grid.D_LOG10_B
_D_FLOOR_PC = yso_module.D_REF_PC * 10.0 ** (-_B_TOP / 2.0)

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


def sample_yso(loaded, row):
    """YSO's `(x, log10_b, w)` on sightline `row` of `loaded`
    (`_region_profile`'s output): every profile cell's mass `p_k * du_k`
    laid, in `N_SUB` equally-weighted sub-samples, along its own segment
    of the `(log10 x, log10 B)` plane between `(u_k, d_k)` and `(u_{k+1},
    d_{k+1})` -- linear in `log10 x` and in `log10 B = -2 log10(d /
    D_REF_PC)` at once, sec. 5.5 "Marks": "a fixed number of sub-samples
    per cell". Both endpoints are floored at the common grid's own edges
    before the log (`_X_FLOOR`, `_D_FLOOR_PC`), so the sightline's nearest
    cell (`u = 0` at `d = 0`) never takes a `log10(0)`."""
    u_edges = loaded["u_edges"][row]
    d_edges = loaded["d_edges"][row]
    p_u = loaded["p_u"][row]
    u_lo, u_hi = u_edges[:-1], u_edges[1:]
    d_lo, d_hi = d_edges[:-1], d_edges[1:]
    mass = p_u * (u_hi - u_lo)  # (n_cell,)

    log10x_lo = np.log10(np.maximum(u_lo, _X_FLOOR))
    log10x_hi = np.log10(np.maximum(u_hi, _X_FLOOR))
    b_lo = -2.0 * np.log10(np.maximum(d_lo, _D_FLOOR_PC) / yso_module.D_REF_PC)
    b_hi = -2.0 * np.log10(np.maximum(d_hi, _D_FLOOR_PC) / yso_module.D_REF_PC)

    log10x = log10x_lo[:, None] + _SUB_T[None, :] * (log10x_hi - log10x_lo)[:, None]
    log10b = b_lo[:, None] + _SUB_T[None, :] * (b_hi - b_lo)[:, None]
    w = np.broadcast_to((mass / N_SUB)[:, None], log10x.shape)

    x = 10.0 ** log10x.ravel()
    return x, log10b.ravel(), w.ravel()
