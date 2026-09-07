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
    "H2S": ("cloud", None),  # formed at load, section 4.1 P3
    "GAL": ("gal", "GRID"),
}
_LIB = {"STAR": ("sps", "region"), "AGB": ("agb", "region"), "PAHC": ("pahc", "region"),
        "YSO": ("yso", "survey"), "H2S": ("h2shock", "survey"), "GAL": ("galz", "survey")}

_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


class Prior(object):
    """One region/class's P1-P5 read, held for repeated `prepare`/`ln_prior`
    calls over the class's batches (section 4)."""

    def __init__(self, a_col, a_col_sig, arm, zp_sig, grain, density, p1_columns,
                 grid_all, x_edges, b_edges, model_name, c_theta, factors, kernel, has_weights):
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
        self.has_weights = has_weights


def load(config, region, cls):
    """`Prior` for `region`'s class `cls` (SPEC_BMSTP_DRAFT.md section 4.1):
    P1's rows, the class's shape grid(s), its weight table `C_THETA` and
    factors (P5), the column kernel. If the class's P5 file is not yet
    built the read proceeds with `C_THETA = 0` and no factors (uniform
    weights), disclosed by `has_weights=False` -- the read's code path is
    unchanged either way.
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
        path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
        with h5py.File(path, "r") as f:
            x_edges = f["LOG10_X_EDGES"][:]
            b_edges = f["LOG10_B_EDGES"][:]
            grid_all = f[dset][:]
        grain = tile
    elif shape_src == "cloud":
        path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
        with h5py.File(path, "r") as f:
            x_edges = f["LOG10_X_EDGES"][:]
            b_edges = f["LOG10_B_EDGES"][:]
            if cls == "YSO":
                grid_all = f["GRID_YSO"][:]
            else:
                # H2S's grid, section 4.1 P3: the sightline's log10 x
                # marginal of GRID_YSO times the region's knot-brightness
                # Gaussian on the B axis, formed here (not stored).
                x_marg = f["X_MARGINAL"][:].astype(np.float64)
                logsig_mean = float(f.attrs["LOGSIG_MEAN"])
                logsig_std = float(f.attrs["LOGSIG_STD"])
                b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
                z = (b_centers - logsig_mean) / logsig_std
                b_pdf = np.exp(-0.5 * z * z)
                b_pdf /= b_pdf.sum()
                grid_all = (x_marg[:, :, None] * b_pdf[None, None, :]).astype(np.float32)
        grain = sightline
    else:  # gal: one survey-wide grid, no grain axis
        path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
        with h5py.File(path, "r") as f:
            x_edges = f["LOG10_X_EDGES"][:]
            b_edges = f["LOG10_B_EDGES"][:]
            grid_all = f["GRID"][:][None, :, :]
        grain = np.zeros(a_col.shape[0], dtype=np.int64)

    lib, granule = _LIB[cls]
    weight_path = config_module.product_path(
        config, "bmstp", "weights", lib, granule, region=(region if granule == "region" else None))
    has_weights = os.path.exists(weight_path)
    if has_weights:
        with h5py.File(weight_path, "r") as f:
            model_name = f["MODEL_NAME"][:]
            c_theta = f["C_THETA"][:]
            b_centers_w = f["LOG10_B_CENTERS"][:]
            n_factor = sum(1 for k in f.keys() if k.startswith("factor_"))
            factors = []
            for k in range(n_factor):
                grp = f["factor_%d" % k]
                factors.append(dict(W=grp["W"][:].astype(np.float64), C_F=grp["C_F"][:],
                                     D_F=grp.attrs.get("D_F", ""),
                                     b_centers=b_centers_w))
    else:
        model_name = np.array([], dtype="S1")
        c_theta = np.zeros(0)
        factors = []

    kernel = kernel_module.Kernel.read(config)
    return Prior(a_col, a_col_sig, arm, zp_sig, grain, density, p1_columns,
                 grid_all, x_edges, b_edges, model_name, c_theta, factors, kernel, has_weights)


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


def _truncated_mean(a_hat, sigma_a):
    """`a*`: the mean of `N(a_hat, sigma_a)` truncated to `a >= 0`
    (SPEC_BMSTP_DRAFT.md section 4.2's `a*`), broadcasting `sigma_a`
    (n,) against `a_hat` (n, m)."""
    z = a_hat / sigma_a[:, None]
    phi = np.exp(-0.5 * z * z) / _SQRT2PI
    big_phi = 0.5 * (1.0 + _erf_np(z / _SQRT2))
    big_phi = np.maximum(big_phi, 1e-300)
    return a_hat + sigma_a[:, None] * phi / big_phi


def _erf_np(x):
    from scipy.special import erf
    return erf(x)


def _factor_ln(reader, rows, a_hat, log10_b_hat, slope, sigma_a, model_index):
    """`Sum_f ln PI_f[theta](log10 B_hat_theta(a*) + C_F_f[theta] +
    D_F_f[s])` (section 4.2), zero where the class's P5 product is not
    yet built (`reader.has_weights` False, disclosed by the caller)."""
    n, m = a_hat.shape
    if not reader.has_weights or not reader.factors:
        return np.zeros((n, m), dtype=np.float64)
    a_star = _truncated_mean(a_hat, sigma_a)
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


@numba.njit(cache=True, fastmath=True, error_model="numpy", parallel=True)
def _cell_sum(a_col, x_edges, sigma_a, a_hat, log10_b_hat, slope, c_theta, h,
              b_origin, dlb, dlx, a_edges_buf):
    """The cell sum of SPEC_BMSTP_DRAFT.md section 4.2, per source and
    template: the cell window `[i_lo, i_hi]` holding `a_hat +/- 5 sigma_a`
    found in O(1) from the grid's own geometric spacing (no table, no
    scan of the other 125 cells); in each cell the Gaussian's mass `M_i`
    by an erf difference and the brightness argument `a*_i`, the cell's
    own truncated-normal mean `a_hat + sigma_a * (phi(alpha_i) -
    phi(beta_i)) / M_i` with `alpha_i`, `beta_i` the cell's edges in
    sigma units (section 4.2) -- `a_hat` itself for a narrow Gaussian,
    the cell's midpoint for a wide one, so `h`'s gather and the Jacobian
    `1 / a*_i` both sit at the mass's own mean within the cell, not the
    cell's geometric center. The dot with `M` runs over cells above
    1e-6. `A_COL_K` and `ln 10` in the Jacobian, common to every template
    at a source, are dropped. No `(n_source x n_model x cells)`
    intermediate. `a_edges_buf` is `(n, n_x+1)` scratch, one row per
    source: passed in rather than allocated per `prange` iteration, since
    numba's auto-parallelisation can hoist a loop-invariant-shaped
    `np.empty` out of the parallel loop and share one buffer across
    threads -- a real race this reader hit at n > 1 sources, silently
    wrong answers, not a crash."""
    n, m = a_hat.shape
    n_x = x_edges.size - 1
    n_b = h.shape[2]
    x0 = x_edges[0]
    out = np.full((n, m), -np.inf, dtype=np.float32)
    sqrt2 = 1.4142135623730951
    sqrt2pi = 2.5066282746310002  # sqrt(2 pi), the normal density's normalisation
    for s in numba.prange(n):
        AK = a_col[s]
        sig = sigma_a[s]
        if sig <= 0.0 or AK <= 0.0:
            continue
        log10_ak = math.log10(AK)
        a_edges = a_edges_buf[s]
        for i in range(n_x + 1):
            a_edges[i] = AK * 10.0 ** x_edges[i]
        inv_sig = 1.0 / sig
        for th in range(m):
            ah = a_hat[s, th]
            lo_a = ah - 5.0 * sig
            hi_a = ah + 5.0 * sig
            if hi_a <= 0.0:
                continue  # the +/-5 sigma window never reaches positive extinction
            i_lo = _cell_index(lo_a, log10_ak, x0, dlx, n_x)
            i_hi = _cell_index(hi_a, log10_ak, x0, dlx, n_x)
            total = 0.0
            lbh = log10_b_hat[s, th]
            sl = slope[s]
            ct = c_theta[th]
            z_prev = (a_edges[i_lo] - ah) * inv_sig
            cdf_prev = 0.5 * (1.0 + math.erf(z_prev / sqrt2))
            phi_prev = math.exp(-0.5 * z_prev * z_prev) / sqrt2pi
            for i in range(i_lo, i_hi + 1):
                z_next = (a_edges[i + 1] - ah) * inv_sig
                cdf_next = 0.5 * (1.0 + math.erf(z_next / sqrt2))
                phi_next = math.exp(-0.5 * z_next * z_next) / sqrt2pi
                mi = cdf_next - cdf_prev
                if mi >= 1e-6:
                    # a*_i: the Gaussian's mean within cell i (section 4.2)
                    a_star = ah + sig * (phi_prev - phi_next) / mi
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
    core = _cell_sum(a_col, reader.x_edges, np.asarray(sigma_a, dtype=np.float64),
                      np.asarray(a_hat, dtype=np.float64), np.asarray(log10_b_hat, dtype=np.float64),
                      np.asarray(slope, dtype=np.float64), np.asarray(c_theta, dtype=np.float64),
                      h, reader.b_origin, reader.dlb, reader.dlx, a_edges_buf)
    factor_term = _factor_ln(reader, rows, np.asarray(a_hat, dtype=np.float64),
                              np.asarray(log10_b_hat, dtype=np.float64),
                              np.asarray(slope, dtype=np.float64),
                              np.asarray(sigma_a, dtype=np.float64), model_index)
    return (core + np.log(density)[:, None] + factor_term).astype(np.float32)
