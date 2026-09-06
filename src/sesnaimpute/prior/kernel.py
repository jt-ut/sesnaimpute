"""The column kernel `p(T | A_measured)` (SPEC_PRIORS.md section 1.2):
log-normal in the true column, `log10 T ~ Normal(log10 A_s + mu, sigma)`.
`mu` and `sigma` are the mean and standard deviation of `log10(T/A)` under
the sub-beam measurement's conditional distribution at the arm's stated
beam, tabulated on the column grid and interpolated linearly in `log A`
between nodes. The source's own measurement uncertainty and, for
Herschel, the field zero point add to `sigma` in quadrature at read time.
Every convolution along a scaled extinction axis is a shift by `mu` and a
Gaussian smoothing of width `sigma`; no quadrature.
"""

import os

import h5py
import numpy as np
from scipy.special import erf

from sesnaimpute import config as config_module

#: How far past `A_s`, in sigma, consumers take the kernel's tail.
X_TAIL_SIGMAS = 3.0

#: The three beams the sub-beam conditional tables are tabulated at.
_TABULATED = (("L108", 108.0), ("L302", 301.8), ("L821", 821.0))
#: A conditioning-column bin with fewer counts than this in the persisted
#: KA x KD histogram carries no resolvable quantile.
_MIN_KERNEL_COUNTS = 200.0

#: The two beams the sub-beam term is ever evaluated at (spec 1.2), and
#: the fixed order/codes the tabulated product's arm axis uses.
STATED_BEAM_ARCSEC = {"herschel": 36.3, "planck": 301.52072}
_ARM_ORDER = ("herschel", "planck")
_ARM_CODE = {"herschel": 0, "planck": 1}

_LN16 = float(np.log(16.0))
_LN10 = float(np.log(10.0))
_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


def _conditional_table(kern2d, ka_col, kd_col):
    """`(table, ci)`: per occupied KA bin (>= `_MIN_KERNEL_COUNTS` counts),
    `(A_HERSCHEL_SMOOTH_K, median_d, q16, q84, q99)` read off the bin's own
    KD cumulative distribution."""
    counts = kern2d.astype(np.float64)
    tot = counts.sum(axis=1)
    good = tot >= _MIN_KERNEL_COUNTS
    cdf = np.cumsum(counts[good], axis=1) / tot[good, None]
    q = np.array([np.interp([0.50, 0.16, 0.84, 0.99], cdf[i], kd_col)
                  for i in range(cdf.shape[0])])
    table = np.column_stack([ka_col[good], q])
    ci = {"A_HERSCHEL_SMOOTH_K": 0, "median_d": 1, "q16": 2, "q84": 3, "q99": 4}
    return table, ci


def _load_subbeam_and_scaling(config):
    """`(regions, conditional, scalars)`: the per-region beam-rescaling and
    offset-rescaling scalars and the column-conditional kernel tables at
    the three tabulated beams, from `sky/derived/herschel/subbeam/region`
    (`sky.derived.subbeam.build`)."""
    path = config_module.product_path(config, "sky/derived", "herschel",
                                      "subbeam", "region")
    with h5py.File(path, "r") as f:
        regions = [r.decode() if isinstance(r, bytes) else r
                  for r in f["REGION"][:]]
        ka_edges, kd_edges = f["KA_EDGES"][:], f["KD_EDGES"][:]
        ka_col = np.exp(0.5 * (ka_edges[:-1] + ka_edges[1:]))
        kd_col = 0.5 * (kd_edges[:-1] + kd_edges[1:])
        scales, qs = f["SCALES"][:], f["QS"][:]
        i302 = int(np.argmin(np.abs(scales - 301.8)))
        i_med = int(np.argmin(np.abs(qs - 0.50)))
        cond_quant = f["COND_QUANTILES"][:]
        beta = f["BETA"][:]
        comp108, comp302, comp821 = (f["COMPLETION_L108"][:],
                                     f["COMPLETION_L302"][:],
                                     f["COMPLETION_L821"][:])
        w_36p3 = f["W_ABS_36P3"][:]
        offset_p = f["OFFSET_EXPONENT"][:]
        kern = {lab: f["COND_KERNEL_%s" % lab][:] for lab, _ in _TABULATED}

    scalars, conditional = {}, {}
    for i, reg in enumerate(regions):
        scalars[reg] = dict(
            beta=float(beta[i]), W_abs_36p3=float(w_36p3[i]),
            completion_L108=float(comp108[i]), completion_L302=float(comp302[i]),
            completion_L821=float(comp821[i]),
            med_off_302=float(cond_quant[i, i302, i_med]),
            offset_powerlaw_p=float(offset_p[i]))
        conditional[reg] = {lab: _conditional_table(kern[lab][i], ka_col, kd_col)
                            for lab, _ in _TABULATED}
    return regions, conditional, scalars


def _load_sigma_zp_herschel(config):
    """`SIGMA_ZP_K`: the Herschel field zero point (SPEC_PRIORS.md section
    1.2), from `sky.derived.herschel_column.build`."""
    path = config_module.product_path(config, "sky/derived", "herschel",
                                      "sigma", "survey")
    with h5py.File(path, "r") as f:
        return float(f["SIGMA_ZP_K"][()])


class _RegionKernel(object):
    """One region's sub-beam parameters: the raw conditional tables (three
    tabulated beams) plus the region's own beam- and offset-rescaling
    exponents, giving the composite (Gaussian-core / exponential-tail)
    shape of `s = ln T - ln column` at any beam and conditioning column."""

    _QUANTILE_COLUMNS = ("median_d", "q16", "q84", "q99")

    def __init__(self, tables, scalars):
        self._sorted = {}
        for lab, (table, ci) in tables.items():
            x = np.log(table[:, ci["A_HERSCHEL_SMOOTH_K"]])
            order = np.argsort(x)
            self._sorted[lab] = (x[order],
                                 {n: table[order, ci[n]]
                                  for n in self._QUANTILE_COLUMNS})
        self.beta = scalars["beta"]
        self.W_abs_36p3 = scalars["W_abs_36p3"]
        self.completion = {"L108": scalars["completion_L108"],
                            "L302": scalars["completion_L302"],
                            "L821": scalars["completion_L821"]}
        self.med_off_302 = scalars["med_off_302"]
        self.offset_p = scalars["offset_powerlaw_p"]

    def w_abs(self, L):
        return self.W_abs_36p3 * (L / 36.3) ** ((self.beta - 2.0) / 2.0)

    def offset(self, L):
        return self.med_off_302 * (L / 301.8) ** self.offset_p

    def _nearest_ref(self, L):
        labs = [lab for lab, _ in _TABULATED]
        Ls = np.array([Lr for _, Lr in _TABULATED])
        i = int(np.argmin(np.abs(np.log(L) - np.log(Ls))))
        return labs[i], float(Ls[i])

    def params_batch(self, columns, beam_fwhm_arcsec):
        """`(mu, sigma, s_break, k)`, each `(n_col,)`: the composite
        kernel's shape for `s = ln T - ln column` at this region and beam.
        Below 36.3" (no column-binned tail data) this is a plain,
        column-independent Gaussian (`s_break = +inf`)."""
        columns = np.asarray(columns, dtype=float)
        L = float(beam_fwhm_arcsec)
        n = columns.size
        if L <= 36.3:
            return (np.full(n, self.offset(L)), np.full(n, self.w_abs(L)),
                    np.full(n, float("inf")), np.ones(n))

        lab, L_ref = self._nearest_ref(L)
        x, ys = self._sorted[lab]
        ln_column = np.log(columns)
        q = {name: np.interp(ln_column, x, ys[name])
             for name in self._QUANTILE_COLUMNS}

        mu0 = q["median_d"]
        sigma0 = np.maximum(0.5 * (q["q84"] - q["q16"]), 1.0e-6)
        s_break0 = q["q84"]
        denom = q["q99"] - q["q84"]
        with np.errstate(divide="ignore", invalid="ignore"):
            k0 = np.where(denom > 1.0e-6, _LN16 / denom, _LN16 / 1.0e-6)

        F = (self.w_abs(L) / self.w_abs(L_ref)) * self.completion[lab]
        R = (L / L_ref) ** self.offset_p

        mu = mu0 * R
        sigma = sigma0 * F
        s_break = mu + (s_break0 - mu0) * F
        k = k0 / F
        return mu, sigma, s_break, k


def _composite_moments(mu, sigma, s_break, k):
    """`(mean, var)` of `s` under the Gaussian-core / exponential-tail
    composite, in closed form from its own shape parameters. `s_break =
    +inf` (no tail data at this beam) is the plain Gaussian, `(mu,
    sigma**2)`."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    s_break = np.asarray(s_break, dtype=float)
    k = np.asarray(k, dtype=float)
    finite = np.isfinite(s_break)
    mean = np.where(finite, 0.0, mu)
    var = np.where(finite, 0.0, sigma ** 2)
    if np.any(finite):
        m, s, b, kk = mu[finite], sigma[finite], s_break[finite], k[finite]
        z_break = (b - m) / s
        gt = np.exp(-0.5 * z_break * z_break)
        phi = 0.5 * (erf(z_break / _SQRT2) + 1.0)
        core_mass = s * _SQRT2PI * phi
        tail_mass = gt / kk
        norm = core_mass + tail_mass
        e1 = (m * core_mass - s * s * gt + tail_mass * b + gt / (kk * kk)) / norm
        e2 = (core_mass * (m * m + s * s) - gt * s * s * (m + b)
              + tail_mass * b * b + 2.0 * gt * b / (kk * kk)
              + 2.0 * gt / (kk * kk * kk)) / norm
        mean[finite] = e1
        var[finite] = np.maximum(e2 - e1 * e1, 1.0e-12)
    return mean, var


class Kernel(object):
    """`log10 T ~ Normal(log10 A_s + mu, sigma) | A_s`, `mu`/`sigma`
    tabulated per arm on the column grid (SPEC_PRIORS.md section 1.2).
    Built by `build(config)`, loaded by `read(config)`.
    """

    def __init__(self, a_nodes, mu, sigma, zp_herschel_k):
        self._a_nodes = np.asarray(a_nodes, dtype=float)
        self._ln_nodes = np.log(self._a_nodes)
        self._mu = np.asarray(mu, dtype=float)          # (n_arm, n_node)
        self._sigma = np.asarray(sigma, dtype=float)     # (n_arm, n_node)
        self.zp_herschel_k = float(zp_herschel_k)

    @classmethod
    def read(cls, config):
        path = config_module.product_path(config, "bms", "sesna", "kernel", "survey")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "prior.kernel: no kernel product at %s -- run the "
                "'prior.kernel' RUNBOOK line first" % path)
        with h5py.File(path, "r") as f:
            return cls(f["A_NODES"][:], f["MU"][:], f["SIGMA"][:],
                       float(f["ZP_HERSCHEL_K"][()]))

    def _interp(self, table, a_col, arm_idx):
        """Linear interpolation of `table[arm_idx[i], :]` in `log A` at
        `a_col[i]`, clamped at the grid ends."""
        ln_a = np.log(np.clip(a_col, self._a_nodes[0], self._a_nodes[-1]))
        i = np.clip(np.searchsorted(self._ln_nodes, ln_a) - 1,
                    0, self._ln_nodes.size - 2)
        span = self._ln_nodes[i + 1] - self._ln_nodes[i]
        t = (ln_a - self._ln_nodes[i]) / span
        lo = table[arm_idx, i]
        hi = table[arm_idx, i + 1]
        return lo + t * (hi - lo)

    def _arm_index(self, map_class):
        mc = np.asarray(map_class)
        idx = np.zeros(mc.shape, dtype=np.intp)
        for arm in _ARM_ORDER:
            idx[mc == arm] = _ARM_CODE[arm]
        return idx

    def width_dex(self, a_col, map_class):
        """`sigma_subbeam(A, arm)` alone: the shapes' per-node smoothing
        width, no per-source term."""
        a_col = np.asarray(a_col, dtype=float)
        return self._interp(self._sigma, a_col, self._arm_index(map_class))

    def shift_dex(self, a_col, map_class):
        """`mu(A, arm)`."""
        a_col = np.asarray(a_col, dtype=float)
        return self._interp(self._mu, a_col, self._arm_index(map_class))

    def params(self, a_col, sigma_col, map_class):
        """`(mu, sigma)`, each `(n,)`: `log10 T ~ Normal(log10 a_col + mu,
        sigma)`, `sigma` composing the sub-beam width with the source's
        own measurement uncertainty and, for Herschel, the field zero
        point, both converted to dex at `a_col`."""
        a_col = np.asarray(a_col, dtype=float)
        sigma_col = np.asarray(sigma_col, dtype=float)
        mc = np.asarray(map_class)
        arm_idx = self._arm_index(mc)
        mu = self._interp(self._mu, a_col, arm_idx)
        width = self._interp(self._sigma, a_col, arm_idx)
        sigma_col_dex = sigma_col / (a_col * _LN10)
        zp_dex = np.where(mc == "herschel",
                          self.zp_herschel_k / (a_col * _LN10), 0.0)
        sigma = np.sqrt(width * width + sigma_col_dex * sigma_col_dex
                        + zp_dex * zp_dex)
        return mu, sigma

    def _broadcast_t(self, t, n):
        t = np.asarray(t, dtype=float)
        if t.ndim == 1:
            t = np.broadcast_to(t[np.newaxis, :], (n, t.shape[0]))
        return t

    def pdf(self, t, a_col, sigma_col, map_class):
        """`p(t | a_col)` in `T` (per unit `A_K`): `(n, m)`, `t` broadcast
        against the `n` sources."""
        a_col = np.asarray(a_col, dtype=float)
        mu, sigma = self.params(a_col, sigma_col, map_class)
        tt = self._broadcast_t(t, a_col.size)
        loc = np.log10(a_col) + mu
        z = (np.log10(tt) - loc[:, np.newaxis]) / sigma[:, np.newaxis]
        dens_log10t = (np.exp(-0.5 * z * z)
                      / (sigma[:, np.newaxis] * _SQRT2PI))
        return dens_log10t / (tt * _LN10)

    def cdf(self, t, a_col, sigma_col, map_class):
        """`P(T <= t | a_col)`: `(n, m)`, same broadcasting as `pdf`."""
        a_col = np.asarray(a_col, dtype=float)
        mu, sigma = self.params(a_col, sigma_col, map_class)
        tt = self._broadcast_t(t, a_col.size)
        loc = np.log10(a_col) + mu
        z = (np.log10(tt) - loc[:, np.newaxis]) / sigma[:, np.newaxis]
        return 0.5 * (1.0 + erf(z / _SQRT2))


def build(config, regions=None):
    """Tabulates `mu(A, arm)` and `sigma_subbeam(A, arm)` -- the mean and
    standard deviation of `log10(T/A)` under the sub-beam measurement's
    conditional distribution, equal-weight over the region mixture -- on
    every node of the column grid, for both arms at their stated beam, and
    writes `bms/sesna/kernel_sesna_survey.hdf5`. Survey-wide; `regions` is
    accepted and ignored.
    """
    from sesnaimpute.prior import column_grid

    regions_list, conditional, scalars = _load_subbeam_and_scaling(config)
    zp = _load_sigma_zp_herschel(config)
    a_nodes = column_grid.nodes(config)

    region_kernels = [_RegionKernel(conditional[r], scalars[r])
                      for r in regions_list]
    n_reg = len(region_kernels)

    mu_rows, sigma_rows = [], []
    for arm in _ARM_ORDER:
        beam = STATED_BEAM_ARCSEC[arm]
        means = np.empty((n_reg, a_nodes.size))
        variances = np.empty((n_reg, a_nodes.size))
        for i, rk in enumerate(region_kernels):
            mu_r, sigma_r, s_break_r, k_r = rk.params_batch(a_nodes, beam)
            means[i], variances[i] = _composite_moments(
                mu_r, sigma_r, s_break_r, k_r)
        mean_mix = means.mean(axis=0)
        var_mix = (variances + means * means).mean(axis=0) - mean_mix * mean_mix
        mu_rows.append(mean_mix / _LN10)
        sigma_rows.append(np.sqrt(np.maximum(var_mix, 1.0e-12)) / _LN10)

    MU = np.stack(mu_rows, axis=0)
    SIGMA = np.stack(sigma_rows, axis=0)
    codes = np.array([_ARM_CODE[arm] for arm in _ARM_ORDER], dtype=np.int64)

    out_path = config_module.product_path(config, "bms", "sesna", "kernel", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("A_NODES", data=a_nodes.astype(np.float64))
        f.create_dataset("MU", data=MU.astype(np.float64))
        f.create_dataset("SIGMA", data=SIGMA.astype(np.float64))
        f.create_dataset("MAP_CLASS_CODES", data=codes)
        f.create_dataset("ZP_HERSCHEL_K", data=np.float64(zp))

    print("kernel: %d arms x %d nodes -> %s" % (len(_ARM_ORDER), a_nodes.size, out_path),
          flush=True)


if __name__ == "__main__":
    from sesnaimpute.build import run
    run(build)
