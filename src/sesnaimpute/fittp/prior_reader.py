"""The reader of SPEC_BMSTP_DRAFT.md section 4: the fitter's one contract
with the prior, `bmstp`'s three tables (section 4.1) folded into the cell
sum of section 4.2. A library module: no `progress` use, no runbook line
(IMPLEMENTATION_BMSTP_DRAFT.md section 4 row 2.1).

`load` opens one region's class: P1's per-source rows, the class's shape
grid (P2 for STAR/AGB, P2's STAR grid for PAHC, P3 for YSO/H2S, P4 for
GAL), the class's template weights (P5) and `C_THETA`, and the population
column kernel (`population.kernel.Kernel`, section 2 "the column kernel").
`prepare` blurs a block's grain shapes by each source's own kernel
(`bmstp.grid.blur`). `ln_prior` is the cell sum itself, in numba: per
template the cell window holding `a_hat +/- 5 sigma_a`, found in O(1)
from the grid's own geometric spacing (section 2) rather than a scan or a
tabulated `a_hat` grid, the Gaussian's mass `M_i` there by an erf
difference, the gather along the conditional brightness line and the dot
product with `M`, the Jacobian, then the weight factors and the log sky
density (section 1.3, 1.4, 4.2).
"""

import math
import os

import h5py
import numba
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute.bmstp import grid
from sesnaimpute.population import kernel as kernel_module

#: class -> (shape-grid source, its dataset, template-weight library, its
#: granule) -- IMPLEMENTATION_BMSTP_DRAFT.md section 1.2 P2-P5.
_SHAPE = {
    "STAR": ("star", "GRID_STAR"),
    "AGB": ("star", "GRID_AGB"),
    "PAHC": ("star", "GRID_STAR"),
    "YSO": ("cloud", "GRID_YSO"),
    "H2S": ("cloud", "GRID_H2S"),  # on the common grid since the common-axis rule, section 4.1 P3
    "GAL": ("gal", "GRID"),
}
_LIB = {"STAR": ("sps", "region"), "AGB": ("agb", "region"), "PAHC": ("pahc", "region"),
        "YSO": ("yso", "region"), "H2S": ("h2shock", "region"), "GAL": ("galz", "survey")}

_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))

#: SPEC_BMSTP_DRAFT.md section 4.2: a window at most this many cells wide
#: is summed by exact per-cell erf differences; a wider one reads the
#: source's own a'-grid table instead, by window width alone -- whatever
#: sigma_a is (W6d item 2, ruling: window width, not sigma_a, decides).
N_EXACT = 8
#: the a'-grid step's coarse limit, section 4.2 ("a fine grid of a_hat
#: (0.01 mag)"). The per-source step is min(this, sigma_a / 10), so a
#: narrow Gaussian's table stays accurate (interpolation error
#: (step / sigma_a)^2 / 8 <= 0.125% at the sigma_a/10 floor) even though
#: its window can still be wide near the grid's low-extinction end, where
#: cells are far narrower than sigma_a itself (W6d ruling).
A_STAR_TABLE_STEP_MAX = 0.01


class Prior(object):
    """One region/class's P1-P5 read, held for repeated `prepare`/`ln_prior`
    calls over the class's batches (section 4)."""

    def __init__(self, a_col, a_col_sig, arm, zp_sig, grain, density, p1_columns,
                 grid_all, x_edges, b_edges, model_name, c_theta, factors, kernel):
        self.a_col = a_col
        self.a_col_sig = a_col_sig
        self.arm = arm
        self.zp_sig = zp_sig
        self.grain = grain
        self.density = density
        self.p1_columns = p1_columns
        self.grid_all = grid_all
        self.x_edges = x_edges
        self.dlx = float(x_edges[1] - x_edges[0])
        self.b_edges = b_edges
        self.b_origin = float(b_edges[0])
        self.dlb = float(b_edges[1] - b_edges[0])
        self.model_name = model_name
        self.c_theta = c_theta
        self.factors = factors  # list of dict(W, C_F, D_F, normalised)
        self.kernel = kernel


def load(config, region, cls):
    """`Prior` for `region`'s class `cls` (SPEC_BMSTP_DRAFT.md section 4.1):
    P1's rows, the class's shape grid(s), its weight table `C_THETA` and
    factors (P5), the column kernel. Rule 5b: a class's P5 file must
    already exist -- a missing one fails with one sentence naming the
    RUNBOOK line that builds it, never a silent uniform-weight read.
    """
    p1_path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    with h5py.File(p1_path, "r") as f:
        a_col = f["A_COL_K"][:].astype(np.float64)
        a_col_sig = f["A_COL_SIG_K"][:].astype(np.float64)
        arm = f["ARM"][:]
        zp_sig = f["ZP_SIG_K"][:].astype(np.float64)
        density = f["DENSITY_%s" % cls][:].astype(np.float64)
        tile = f["TILE"][:]
        sightline = f["SIGHTLINE_ROW"][:]
        p1_columns = {"D_PAHC": f["D_PAHC"][:].astype(np.float64)}

    shape_src, dset = _SHAPE[cls]
    if shape_src == "star":
        # STAR/AGB: the common axis (section 2), read straight off P2 --
        # one origin, one width, shared with every other class.
        path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
        with h5py.File(path, "r") as f:
            x_edges = f["LOG10_X_EDGES"][:]
            b_edges = f["LOG10_F45_EDGES"][:]
            grid_all = f[dset][:]
        grain = tile
    elif shape_src == "cloud":
        path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
        with h5py.File(path, "r") as f:
            # the common axis for both cloud classes since the common-axis rule: P3's
            # `GRID_YSO`/`GRID_H2S` are already on it (section 4.1, 5.6),
            # the H2S template's Sigma-to-4.5-micron conversion `C_THETA`
            # (P5, `bmstp.template_weights.h2shock_conversion`) folded in
            # at the shape stage, not at this read -- no private axis.
            x_edges = f["LOG10_X_EDGES"][:]
            b_edges = f["LOG10_F45_EDGES"][:]
            grid_all = f[dset][:]
        grain = sightline
    else:  # gal: one survey-wide grid, no grain axis, on the common axis too
        path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
        with h5py.File(path, "r") as f:
            x_edges = f["LOG10_X_EDGES"][:]
            b_edges = f["LOG10_F45_EDGES"][:]
            grid_all = f["GRID"][:][None, :, :]
        grain = np.zeros(a_col.shape[0], dtype=np.int64)

    lib, granule = _LIB[cls]
    weight_path = config_module.product_path(
        config, "bmstp", "weights", lib, granule, region=(region if granule == "region" else None))
    if not os.path.exists(weight_path):
        # rule 5b: the only existence check, one sentence naming the
        # RUNBOOK line that makes it -- no silent uniform-weight fallback.
        raise RuntimeError(
            "fittp.prior_reader.load [%s/%s]: missing %s -- run RUNBOOKtp.sh's "
            "'PY sesnaimpute.bmstp.template_weights' line first" % (region, cls, weight_path))
    with h5py.File(weight_path, "r") as f:
        model_name = f["MODEL_NAME"][:]
        c_theta = f["C_THETA"][:]
        # the factor tables' cell axis is the common one (P5, section 4.1).
        b_centers_w = f["LOG10_F45_CENTERS"][:]
        n_factor = sum(1 for k in f.keys() if k.startswith("factor_"))
        factors = []
        for k in range(n_factor):
            grp = f["factor_%d" % k]
            factors.append(dict(W=grp["W"][:].astype(np.float64), C_F=grp["C_F"][:],
                                 D_F=grp.attrs.get("D_F", ""),
                                 b_centers=b_centers_w))

    kernel = kernel_module.Kernel.read(config)
    return Prior(a_col, a_col_sig, arm, zp_sig, grain, density, p1_columns,
                 grid_all, x_edges, b_edges, model_name, c_theta, factors, kernel)


def prepare(reader, rows):
    """`h (n_block, 128, 120)` float32 (SPEC_BMSTP_DRAFT.md section 4.2,
    9): each of `rows`' sources, its grain's shape blurred along
    `log10 x` by its own column kernel (`grid.blur`), floored at 1e-6 of
    its peak cell and renormalised to sum to one -- a block of ~50
    sources, never a whole batch (the 600 MB per-batch footprint of the
    unblurred grid held at once, IMPLEMENTATION_BMSTP_DRAFT.md section 9).
    """
    rows = np.asarray(rows)
    a_col = reader.a_col[rows]
    a_col_sig = reader.a_col_sig[rows]
    zp_sig = reader.zp_sig[rows]
    arm = reader.arm[rows]
    grain = reader.grain[rows]
    w, mu, sigma = reader.kernel.mixture(a_col, a_col_sig, arm, zp_sigma_k=zp_sig)
    n = rows.size
    n_x, n_b = reader.grid_all.shape[1], reader.grid_all.shape[2]
    h = np.empty((n, n_x, n_b), dtype=np.float32)
    for k in range(n):
        H = reader.grid_all[grain[k]].astype(np.float64)
        H_s, _ = grid.blur(H, float(w[k]), float(mu[k, 0]), float(sigma[k, 0]),
                            float(mu[k, 1]), float(sigma[k, 1]))
        H_s = np.maximum(H_s, grid.FLOOR * H_s.max())
        H_s = H_s / H_s.sum()
        h[k] = H_s.astype(np.float32)
    return h


def _truncated_mean(a_hat, sigma_a, a_floor):
    """`a*`: the mean of `N(a_hat, sigma_a)` truncated to `a >= 0`
    (SPEC_BMSTP_DRAFT.md section 4.2's `a*`), broadcasting `sigma_a`
    (n,) against `a_hat` (n, m). Far enough below the truncation point
    (`a_hat` some tens of `sigma_a` negative) both the numerator `phi(z)`
    and the denominator `Phi(z)` underflow float64 to zero together, and
    the floor on `Phi` alone would then read the ratio as zero and return
    `a_hat` itself -- a large negative extinction -- instead of the
    truncated normal's own asymptotic mean, which tends to the truncation
    point as `a_hat -> -infinity`: where both underflow, `a*` is read at
    `a_floor` (the grid's own low edge, the same point `_cell_sum`'s
    low-edge fallback reads these templates at) instead."""
    z = a_hat / sigma_a[:, None]
    phi = np.exp(-0.5 * z * z) / _SQRT2PI
    big_phi_raw = 0.5 * (1.0 + _erf_np(z / _SQRT2))
    big_phi = np.maximum(big_phi_raw, 1e-300)
    a_star = a_hat + sigma_a[:, None] * phi / big_phi
    underflow = (phi == 0.0) & (big_phi_raw <= 1e-300)
    return np.where(underflow, a_floor, a_star)


def _erf_np(x):
    from scipy.special import erf
    return erf(x)


def _factor_ln(reader, rows, a_hat, log10_b_hat, slope, sigma_a, model_index):
    """`Sum_f ln PI_f[theta](log10 B_hat_theta(a*) + C_F_f[theta] +
    D_F_f[s])` (section 4.2); `reader.factors` is empty only for a class
    with no factor tables of its own (never for a missing P5 file --
    `load` fails on that, rule 5b)."""
    n, m = a_hat.shape
    if not reader.factors:
        return np.zeros((n, m), dtype=np.float64)
    a_floor = reader.a_col[rows][:, None] * (10.0 ** reader.x_edges[0])
    a_star = _truncated_mean(a_hat, sigma_a, a_floor)
    b_star = log10_b_hat + slope[:, None] * (a_star - a_hat)
    total = np.zeros((n, m), dtype=np.float64)
    for f in reader.factors:
        w_theta = f["W"][model_index]          # (m, 120)
        c_f = f["C_F"][model_index]            # (m,)
        b_centers = f["b_centers"]
        d_f = 0.0
        if f["D_F"]:
            d_f = reader.p1_columns[f["D_F"]][rows][:, None]
        arg = b_star + c_f[None, :] + d_f       # (n, m)
        dlb = float(b_centers[1] - b_centers[0])
        pos = (arg - b_centers[0]) / dlb
        pos = np.where(np.isfinite(pos), pos, 0.0)
        j0 = np.clip(np.floor(pos).astype(np.int64), 0, b_centers.size - 2)
        frac = np.clip(pos - j0, 0.0, 1.0)
        rows_idx = np.arange(m)[None, :]
        v0 = w_theta[rows_idx, j0]
        v1 = w_theta[rows_idx, j0 + 1]
        pi = np.maximum(v0 * (1.0 - frac) + v1 * frac, 1e-300)
        total += np.log(pi)
    return total


def _build_a_star_tables(a_col, x_edges, sigma_a, a_hat):
    """The hybrid cell-mass table of SPEC_BMSTP_DRAFT.md section 4.2, one
    per source: on an `a'` grid of step `min(A_STAR_TABLE_STEP_MAX, sigma_a
    / 10)` covering this block's templates' `a_hat` range (`(n,)` `a_hat`
    rows, one per source) clipped to `[-5 sigma_a, A_COL_K * 10^x_max + 5
    sigma_a]`, the cell mass `M_i(a')` and the cell's truncated-normal mean
    `a*_i(a')` (both `(n_ap, n_x)`), by one vectorised `scipy.special.erf`
    and `exp` pass over the whole grid (no erf/exp inside the per-template
    kernel loop, W6d item 2 as ruled: the exact/table choice is by window
    width alone, so every source gets a table regardless of `sigma_a`; the
    `sigma_a / 10` step floor keeps a narrow Gaussian's interpolation error
    at `(step / sigma_a)^2 / 8 <= 0.125%` even on the rare narrow-`sigma_a`
    template whose window is still wide because it sits near the grid's
    low-extinction end, where cells are far narrower than `sigma_a`),
    plus each `a'` bin's own cell window `i_lo, i_hi` from the grid's
    geometric spacing. Returns flat float32 `(sum(n_ap), n_x)` M and a*
    tables holding every source's rows back to back (no padding to the
    block's widest range, finding 3: one narrow-`sigma_a` source no
    longer sets the allocation for the whole block), flat int32
    `(sum(n_ap),)` window tables, an int64 `(n,)` per-source offset into
    those flat tables, float64 `(n,)` grid origins and int32 `(n,)` real
    lengths.
    """
    from scipy.special import erf
    n = sigma_a.size
    n_x = x_edges.size - 1
    x0 = float(x_edges[0])
    dlx = float(x_edges[1] - x_edges[0])
    x_max = float(x_edges[-1])
    a_min = np.zeros(n, dtype=np.float64)
    step = np.zeros(n, dtype=np.float64)
    n_ap = np.zeros(n, dtype=np.int64)
    edges_per_source = [None] * n
    for s in range(n):
        if sigma_a[s] <= 0.0 or a_col[s] <= 0.0:
            continue
        lo_bound = -5.0 * sigma_a[s]
        hi_bound = a_col[s] * 10.0 ** x_max + 5.0 * sigma_a[s]
        row = a_hat[s]
        amin = max(lo_bound, float(np.min(row)))
        amax = min(hi_bound, float(np.max(row)))
        st = min(A_STAR_TABLE_STEP_MAX, sigma_a[s] / 10.0)
        if amax <= amin:
            amax = amin + st
        step[s] = st
        n_ap[s] = int(np.ceil((amax - amin) / st)) + 1
        a_min[s] = amin
        edges_per_source[s] = a_col[s] * 10.0 ** x_edges
    offset = np.zeros(n, dtype=np.int64)
    if n:
        offset[1:] = np.cumsum(n_ap)[:-1]
    total_ap = int(n_ap.sum())
    m_tab = np.zeros((total_ap, n_x), dtype=np.float32)
    a_tab = np.zeros((total_ap, n_x), dtype=np.float32)
    ilo_tab = np.zeros(total_ap, dtype=np.int32)
    ihi_tab = np.zeros(total_ap, dtype=np.int32)
    for s in range(n):
        if n_ap[s] == 0:
            continue
        off = offset[s]
        ap = a_min[s] + step[s] * np.arange(n_ap[s])
        edges = edges_per_source[s]
        z = (edges[None, :] - ap[:, None]) / sigma_a[s]
        cdf = 0.5 * (1.0 + erf(z / _SQRT2))
        phi = np.exp(-0.5 * z * z) / _SQRT2PI
        mass = cdf[:, 1:] - cdf[:, :-1]
        mass_safe = np.maximum(mass, 1e-300)
        a_star = ap[:, None] + sigma_a[s] * (phi[:, :-1] - phi[:, 1:]) / mass_safe
        # the grid's low edge is not a truncation boundary of its own:
        # the kernel's mass below it, M_below (= cdf at the true edge,
        # `cdf[:, 0]`, since that cdf already integrates from -infinity),
        # is placed in the lowest cell (SPEC_BMSTP_DRAFT.md 4.2's
        # low-edge statement) and READ AT THAT CELL -- its own geometric
        # centre a_c0 = sqrt(a_edge_lo * a_1), not the cell's own
        # truncated mean a*_in (mass[:, 0]/a_star[:, 0] above, computed
        # exactly as every other cell). Folding `M_below / a_c0` into the
        # stored (mass, a*) pair at a*_in -- mass_eff = M_in + M_below *
        # a*_in / a_c0 -- makes the same `dens * mass / a*` gather every
        # other cell uses reproduce `dens0 * (M_in / a*_in + M_below /
        # a_c0)` for cell 0 too.
        a_c0 = math.sqrt(edges[0] * edges[1])
        mass[:, 0] = mass[:, 0] + cdf[:, 0] * (a_star[:, 0] / a_c0)
        # cells the window never reaches carry mass ~ 0 and an a* the
        # kernel never gathers (its own window index selects only cells
        # inside +/-5 sigma); clip before the float32 cast so those unused
        # far cells (mass_safe's 1e-300 floor dividing a near-zero
        # numerator swing) cannot overflow it.
        a_star = np.clip(a_star, -1e30, 1e30)
        m_tab[off:off + n_ap[s], :] = mass.astype(np.float32)
        a_tab[off:off + n_ap[s], :] = a_star.astype(np.float32)
        log10_ak = math.log10(a_col[s])
        lo_a = ap - 5.0 * sigma_a[s]
        hi_a = ap + 5.0 * sigma_a[s]
        lx_hi = np.log10(np.maximum(hi_a, 1e-300)) - log10_ak
        ihi = np.clip(np.floor((lx_hi - x0) / dlx), 0, n_x - 1).astype(np.int64)
        lx_lo = np.log10(np.maximum(lo_a, 1e-300)) - log10_ak
        ilo = np.where(lo_a > 0.0, np.clip(np.floor((lx_lo - x0) / dlx), 0, n_x - 1), 0.0).astype(np.int64)
        ilo_tab[off:off + n_ap[s]] = ilo.astype(np.int32)
        ihi_tab[off:off + n_ap[s]] = ihi.astype(np.int32)
    return m_tab, a_tab, ilo_tab, ihi_tab, a_min, step, n_ap.astype(np.int32), offset


@numba.njit(cache=True, fastmath=True, error_model="numpy")
def _cell_index(a_val, log10_ak, x0, dlx, n_x):
    """The `log10 x` cell holding extinction `a_val` at this source's
    `A_COL_K` (its column an O(1) inverse of the grid's own geometric
    spacing, `log10 x_i = x0 + i * dlx`): clamped to `[0, n_x - 1]`, and to
    0 for `a_val <= 0` (the grid's cells start above `x = 0`, so anything
    at or below zero extinction sits at the grid's own low edge)."""
    if a_val <= 0.0:
        return 0
    lx = math.log10(a_val) - log10_ak
    idx = int(math.floor((lx - x0) / dlx))
    if idx < 0:
        idx = 0
    elif idx > n_x - 1:
        idx = n_x - 1
    return idx


@numba.njit(cache=True, fastmath=True, error_model="numpy")
def _ln_half_erfc(z):
    """`ln[(1/2) erfc(z)]`, stable for large `z` (`fittp.likelihood`'s
    `_ln_one_minus_c_kernel`'s own three branches, section 6.2's
    asymptotic series for the scaled complementary error function): used
    for the Gaussian's tail mass beyond the grid's first cell edge, so
    that a template whose window never reaches positive extinction, or
    whose cells all floor below `1e-6`, still reads a finite prior
    (section 1.3: no hypothesis is ever at `-inf`)."""
    ln_half = -0.6931471805599453
    sqrt_pi = 1.7724538509055159
    if z < 0.0:
        return math.log1p(-0.5 * math.erfc(-z))
    elif z < 5.0:
        return ln_half + math.log(math.erfc(z))
    else:
        ln_erfcx = -math.log(z * sqrt_pi) - math.log(1.0 + 1.0 / (2.0 * z * z))
        return ln_half + ln_erfcx - z * z


@numba.njit(cache=True, fastmath=True, error_model="numpy", parallel=True)
def _cell_sum(a_col, x_edges, sigma_a, a_hat, log10_b_hat, slope, c_theta, h,
              b_origin, dlb, dlx, a_edges_buf,
              m_tab, a_tab, ilo_tab, ihi_tab, a_min_tab, step_tab, n_ap_tab, offset_tab):
    """The cell sum of SPEC_BMSTP_DRAFT.md section 4.2, per source and
    template: the cell window `[i_lo, i_hi]` holding `a_hat +/- 5 sigma_a`
    found in O(1) from the grid's own geometric spacing (no scan of the
    other 125 cells); in each cell the Gaussian's mass `M_i` and the
    brightness argument `a*_i`, the cell's own truncated-normal mean --
    `a_hat` itself for a narrow Gaussian, the cell's midpoint for a wide
    one, so `h`'s gather and the Jacobian `1 / a*_i` both sit at the
    mass's own mean within the cell. A window at most `N_EXACT` cells
    wide still pays one erf and one exp per cell exactly; a wider one --
    by window width alone, whatever `sigma_a` is (W6d ruling: a narrow
    Gaussian near the grid's low-extinction end can still have a wide
    window, since cells there are far narrower than `sigma_a`) -- instead
    gathers `M_i`, `a*_i` from the source's own `a'`-grid table
    (`_build_a_star_tables`, W6d item 2) by linear interpolation in `a'`
    -- no erf, no exp, no per-template log10 for that window's mass, only
    the one log10 pair that still decides which path a given template's
    window takes (measured negligible next to the erf/exp it replaces,
    W6d report). `m_tab`/`a_tab`/`ilo_tab`/`ihi_tab` are flat, one
    source's rows at `offset_tab[s] : offset_tab[s] + n_ap_tab[s]`, no
    padding to the block's widest range (finding 3). The dot with `M` runs over cells
    above 1e-6. `A_COL_K` and `ln 10` in the Jacobian, common to every
    template at a source, are dropped. No `(n_source x n_model x cells)`
    intermediate. `a_edges_buf` is `(n, n_x+1)` scratch, one row per
    source. The outer source loop is plain and serial (the fitter's
    harness calls this one source at a time); `prange` is the inner loop
    over templates, so a single source's read still uses every core
    (W6d item 3).

    Cell 0's own mass `M_0` and mean `a*_0` are formed exactly as every
    other cell's, over its own bounds `[a_edges[0], a_edges[1]]` -- the
    grid's low edge is not itself a truncation boundary, but that does
    not move cell 0's own mean below it. The kernel's mass below that
    edge, `M_below`, does not vanish (section 4.2's low-edge statement:
    it is placed in the lowest cell) but it is READ AT THAT CELL, at its
    own geometric centre `a_c0 = sqrt(a_edges[0] * a_edges[1])`, not at
    `a*_0`: cell 0's term is `dens_0 * (M_0 / a*_0 + M_below / a_c0)`,
    `dens_0` gathered once at `a*_0`. The table path
    (`_build_a_star_tables`) and this exact path agree on this.

    Two edge cases never see a cleared cell and are read as a single
    substitute cell instead, so no template's prior is ever `-inf`
    (section 1.3): a window whose own low bound never reaches positive
    extinction, or whose cells all floor below the skip while its low
    bound still sits at the grid's lowest cell, is read at the grid's
    first cell and `a* = a_0`, its own lower edge, times the Gaussian's
    tail mass beyond `a_0` -- and that same lowest cell's own mass
    (`_build_a_star_tables` and the exact per-cell sum below both apply
    this) already carries whatever Gaussian mass lies below `a_0`, since
    the grid's low edge is not itself a truncation boundary. A window
    whose own low bound already clears the grid's TOP edge is read at
    the mirror point instead -- the top cell's own floored density at
    its own upper edge, `a*` there, times the tail mass beyond that edge
    -- because mass above the grid is mass outside and is never wrapped
    onto the grid's low end (section 2); reading it at `a_0` instead
    misprices the Jacobian `1 / a*` by the ratio of the two edges."""
    n, m = a_hat.shape
    n_x = x_edges.size - 1
    n_b = h.shape[2]
    x0 = x_edges[0]
    out = np.full((n, m), -np.inf, dtype=np.float32)
    sqrt2 = 1.4142135623730951
    sqrt2pi = 2.5066282746310002  # sqrt(2 pi), the normal density's normalisation
    for s in range(n):
        AK = a_col[s]
        sig = sigma_a[s]
        if sig <= 0.0 or AK <= 0.0:
            continue
        log10_ak = math.log10(AK)
        a_edges = a_edges_buf[s]
        for i in range(n_x + 1):
            a_edges[i] = AK * 10.0 ** x_edges[i]
        inv_sig = 1.0 / sig
        n_ap = n_ap_tab[s]
        a_min = a_min_tab[s]
        a_step = step_tab[s]
        off = offset_tab[s]
        for th in numba.prange(m):
            ah = a_hat[s, th]
            lo_a = ah - 5.0 * sig
            hi_a = ah + 5.0 * sig
            total = 0.0
            lbh = log10_b_hat[s, th]
            sl = slope[s]
            ct = c_theta[th]
            in_grid = hi_a > 0.0
            i_lo = 0
            i_hi = -1
            if in_grid:
                i_lo = _cell_index(lo_a, log10_ak, x0, dlx, n_x)
                i_hi = _cell_index(hi_a, log10_ak, x0, dlx, n_x)
            kpos = (ah - a_min) / a_step if n_ap > 0 else 0.0
            use_table = n_ap > 0 and (i_hi - i_lo + 1) > N_EXACT
            if not in_grid:
                pass  # the +/-5 sigma window never reaches positive extinction
            elif use_table:
                # the hybrid table path: M_i, a*_i by linear interpolation in a'
                k0 = int(math.floor(kpos))
                if k0 < 0:
                    k0 = 0
                    frac_k = 0.0
                else:
                    frac_k = kpos - k0
                t_lo = ilo_tab[off + k0]
                t_hi = ihi_tab[off + k0]
                for i in range(t_lo, t_hi + 1):
                    mi = m_tab[off + k0, i] * (1.0 - frac_k) + m_tab[off + k0 + 1, i] * frac_k
                    if mi >= 1e-6:
                        a_star = a_tab[off + k0, i] * (1.0 - frac_k) + a_tab[off + k0 + 1, i] * frac_k
                        bval = lbh + sl * (a_star - ah) + ct
                        bpos = (bval - b_origin) / dlb - 0.5
                        j0 = int(math.floor(bpos))
                        frac = bpos - j0
                        if j0 < 0:
                            j0 = 0
                            frac = 0.0
                        elif j0 >= n_b - 1:
                            j0 = n_b - 2
                            frac = 1.0
                        dens = (h[s, i, j0] * (1.0 - frac) + h[s, i, j0 + 1] * frac) / (dlx * dlb)
                        total += dens * mi / a_star
            else:
                z_prev = (a_edges[i_lo] - ah) * inv_sig
                cdf_prev = 0.5 * (1.0 + math.erf(z_prev / sqrt2))
                phi_prev = math.exp(-0.5 * z_prev * z_prev) / sqrt2pi
                for i in range(i_lo, i_hi + 1):
                    z_next = (a_edges[i + 1] - ah) * inv_sig
                    cdf_next = 0.5 * (1.0 + math.erf(z_next / sqrt2))
                    phi_next = math.exp(-0.5 * z_next * z_next) / sqrt2pi
                    # M_i, a*_i: this cell's own mass and truncated-normal
                    # mean within its own bounds (section 4.2), cell 0
                    # included -- the grid's low edge is not itself a
                    # truncation boundary, so cell 0's own bounds are
                    # [a_edges[0], a_edges[1]] like any other cell.
                    mi = cdf_next - cdf_prev
                    if mi >= 1e-6:
                        a_star = ah + sig * (phi_prev - phi_next) / mi
                        if i == 0:
                            # the kernel's mass below the grid's low edge,
                            # M_below = cdf_prev (it already integrates
                            # from -infinity), is placed in the lowest
                            # cell (section 4.2's low-edge statement) and
                            # READ AT THAT CELL -- its own geometric
                            # centre a_c0 = sqrt(a_edges[0]*a_edges[1]),
                            # not a*_i. Folding M_below/a_c0 into the
                            # (mi, a_star) pair at a_star=a*_i reproduces
                            # dens0 * (M_i/a*_i + M_below/a_c0) below.
                            a_c0 = math.sqrt(a_edges[0] * a_edges[1])
                            mi = mi + cdf_prev * (a_star / a_c0)
                        bval = lbh + sl * (a_star - ah) + ct
                        bpos = (bval - b_origin) / dlb - 0.5
                        j0 = int(math.floor(bpos))
                        frac = bpos - j0
                        if j0 < 0:
                            j0 = 0
                            frac = 0.0
                        elif j0 >= n_b - 1:
                            j0 = n_b - 2
                            frac = 1.0
                        dens = (h[s, i, j0] * (1.0 - frac) + h[s, i, j0 + 1] * frac) / (dlx * dlb)
                        total += dens * mi / a_star
                    cdf_prev = cdf_next
                    phi_prev = phi_next
                    z_prev = z_next
            if total > 0.0:
                out[s, th] = np.log(total)
            elif in_grid and lo_a >= a_edges[n_x]:
                # section 1.3: no template's prior is -inf. The window's
                # own low bound already clears the grid's top edge: mass
                # above the grid is mass outside, never wrapped back onto
                # it (section 2), so this reads as the top cell's own
                # floored density at its own upper edge, times the
                # Gaussian's tail mass beyond that edge -- the mirror of
                # the low-edge fallback below, not that fallback's a_0
                # and cell 0 (which would misprice the Jacobian by the
                # ratio of the two edges).
                a_top = a_edges[n_x]
                z_top = (a_top - ah) * inv_sig
                ln_tail = _ln_half_erfc(z_top / sqrt2)
                bval = lbh + sl * (a_top - ah) + ct
                bpos = (bval - b_origin) / dlb - 0.5
                j0 = int(math.floor(bpos))
                frac = bpos - j0
                if j0 < 0:
                    j0 = 0
                    frac = 0.0
                elif j0 >= n_b - 1:
                    j0 = n_b - 2
                    frac = 1.0
                dens = (h[s, n_x - 1, j0] * (1.0 - frac) + h[s, n_x - 1, j0 + 1] * frac) / (dlx * dlb)
                if dens > 0.0:
                    out[s, th] = math.log(dens / a_top) + ln_tail
            else:
                # Either the window never reached positive extinction or
                # every cell in it floored below 1e-6: read the grid's
                # first cell (i = 0) at a* = a_0, its own lower edge,
                # times the Gaussian's tail mass beyond a_0.
                a0 = a_edges[0]
                z0 = (a0 - ah) * inv_sig
                ln_tail = _ln_half_erfc(z0 / sqrt2)
                bval = lbh + sl * (a0 - ah) + ct
                bpos = (bval - b_origin) / dlb - 0.5
                j0 = int(math.floor(bpos))
                frac = bpos - j0
                if j0 < 0:
                    j0 = 0
                    frac = 0.0
                elif j0 >= n_b - 1:
                    j0 = n_b - 2
                    frac = 1.0
                dens = (h[s, 0, j0] * (1.0 - frac) + h[s, 0, j0 + 1] * frac) / (dlx * dlb)
                if dens > 0.0:
                    out[s, th] = math.log(dens / a0) + ln_tail
    return out


def ln_prior(reader, rows, h, a_hat, log10_b_hat, slope, sigma_a, model_index):
    """`(n, m)` float32: `ln <Lambda_C>_s(theta)` of SPEC_BMSTP_DRAFT.md
    section 4.2, plus `ln A_C(s)` (section 1.3) -- everything the fitter's
    evidence sum needs from the prior. `rows` indexes `reader`'s per-source
    arrays; `h` is `prepare(rows)`'s blurred grid for the same sources, in
    the same order; `a_hat`, `log10_b_hat`, `slope`, `sigma_a` are
    `fittp.likelihood.fit`'s unconstrained mark, its conditional slope and
    the fit's own `sigma_a`, all in `A_K`; `model_index` locates each of
    the `m` templates in the class's `C_THETA` and weight-factor tables
    (identity order where the library is read whole).
    """
    rows = np.asarray(rows)
    a_col = reader.a_col[rows]
    density = reader.density[rows]
    model_index = np.asarray(model_index)
    m = a_hat.shape[1]
    n_x = reader.x_edges.size - 1
    c_theta = reader.c_theta[model_index] if reader.c_theta.size else np.zeros(m)
    a_edges_buf = np.empty((rows.size, n_x + 1), dtype=np.float64)
    sigma_a = np.asarray(sigma_a, dtype=np.float64)
    a_hat64 = np.asarray(a_hat, dtype=np.float64)
    m_tab, a_tab, ilo_tab, ihi_tab, a_min_tab, step_tab, n_ap_tab, offset_tab = _build_a_star_tables(
        a_col, reader.x_edges, sigma_a, a_hat64)
    core = _cell_sum(a_col, reader.x_edges, sigma_a,
                      a_hat64, np.asarray(log10_b_hat, dtype=np.float64),
                      np.asarray(slope, dtype=np.float64), np.asarray(c_theta, dtype=np.float64),
                      h, reader.b_origin, reader.dlb, reader.dlx, a_edges_buf,
                      m_tab, a_tab, ilo_tab, ihi_tab, a_min_tab, step_tab, n_ap_tab, offset_tab)
    factor_term = _factor_ln(reader, rows, np.asarray(a_hat, dtype=np.float64),
                              np.asarray(log10_b_hat, dtype=np.float64),
                              np.asarray(slope, dtype=np.float64),
                              np.asarray(sigma_a, dtype=np.float64), model_index)
    return (core + np.log(density)[:, None] + factor_term).astype(np.float32)
