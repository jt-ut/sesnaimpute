"""The column kernel `p(T | A_measured)` (SPEC_PRIORS.md section 1.2): a
mixture of two log-normals in the true column,

    log10 T ~ w * Normal(log10 A_s + mu_1, sigma_1)
              + (1 - w) * Normal(log10 A_s + mu_2, sigma_2)

`w`, `mu_1`, `mu_2`, `sigma_1`, `sigma_2` are the sub-beam stage's fitted
structural mixture (`sesnaimpute.sky.derived.subbeam`, its noise-corrected
forward model), pooled over regions and tabulated on the column grid, for
each arm at its own stated beam. The Planck arm reads the 302 arcsec
sub-beam table. The Herschel arm's own stated beam (36.3 arcsec) is the
finest map there is -- no sub-beam data exists below it -- so its
structural term is a point mass at `T = A_s` (zero width, zero shift):
its kernel is the per-source measurement uncertainty and the field zero
point alone, not an extrapolation. Both are added to each component's
sigma in quadrature, at the source's own column, converted to dex.
"""

import os

import h5py
import numpy as np
from scipy.special import erf

from sesnaimpute import config as config_module

#: How far past `A_s`, in sigma, consumers take the kernel's tail.
X_TAIL_SIGMAS = 3.0

#: The beam the sub-beam mixture table is pooled from for the Planck arm
#: (index into the sub-beam product's beam axis, order L108/L302/L821).
_POOL_BEAM_LABEL = "L302"
_POOL_BEAM_INDEX = 1

#: The two beams the kernel is ever evaluated at (spec 1.2), and the fixed
#: order/codes the tabulated product's arm axis uses.
STATED_BEAM_ARCSEC = {"herschel": 36.3, "planck": 301.52072}
_ARM_ORDER = ("herschel", "planck")
_ARM_CODE = {"herschel": 0, "planck": 1}

_LN10 = float(np.log(10.0))
_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


def _load_sigma_zp_herschel(config):
    """`SIGMA_ZP_K`: the Herschel field zero point (SPEC_PRIORS.md section
    1.2), from `sky.derived.herschel_column.build`."""
    path = config_module.product_path(config, "sky/derived", "herschel",
                                      "sigma", "survey")
    with h5py.File(path, "r") as f:
        return float(f["SIGMA_ZP_K"][()])


def _mixture_mean_var(w, mu1, sigma1, mu2, sigma2):
    """`(mean, var)` of the two-component log-normal mixture in log10 T,
    exactly, from its own component parameters."""
    mean = w * mu1 + (1.0 - w) * mu2
    d1, d2 = mu1 - mean, mu2 - mean
    var = w * (sigma1 * sigma1 + d1 * d1) + (1.0 - w) * (sigma2 * sigma2 + d2 * d2)
    return mean, var


class Kernel(object):
    """The column kernel: a two-component log-normal mixture in `log10 T`
    per arm, tabulated on the column grid (SPEC_PRIORS.md section 1.2).
    Built by `build(config)`, loaded by `read(config)`.
    """

    def __init__(self, a_nodes, w, mu, sigma, zp_herschel_k):
        self._a_nodes = np.asarray(a_nodes, dtype=float)
        self._ln_nodes = np.log(self._a_nodes)
        self._w = np.asarray(w, dtype=float)          # (n_arm, n_node)
        self._mu = np.asarray(mu, dtype=float)         # (n_arm, n_node, 2)
        self._sigma = np.asarray(sigma, dtype=float)    # (n_arm, n_node, 2)
        self.zp_herschel_k = float(zp_herschel_k)

    @classmethod
    def read(cls, config):
        path = config_module.product_path(config, "bms", "sesna", "kernel", "survey")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "prior.kernel: no kernel product at %s -- run the "
                "'prior.kernel' RUNBOOK line first" % path)
        with h5py.File(path, "r") as f:
            return cls(f["A_NODES"][:], f["MIX_W"][:], f["MIX_MU"][:],
                       f["MIX_SIGMA"][:], float(f["ZP_HERSCHEL_K"][()]))

    def _interp_idx(self, a_col):
        """`(i, t)`: the node bracket and fractional position in `log A` for
        linear interpolation, clamped at the grid ends."""
        ln_a = np.log(np.clip(a_col, self._a_nodes[0], self._a_nodes[-1]))
        i = np.clip(np.searchsorted(self._ln_nodes, ln_a) - 1,
                    0, self._ln_nodes.size - 2)
        span = self._ln_nodes[i + 1] - self._ln_nodes[i]
        t = (ln_a - self._ln_nodes[i]) / span
        return i, t

    def _arm_index(self, map_class):
        mc = np.asarray(map_class)
        idx = np.zeros(mc.shape, dtype=np.intp)
        for arm in _ARM_ORDER:
            idx[mc == arm] = _ARM_CODE[arm]
        return idx

    def _structural(self, a_col, arm_idx):
        """`(w, mu, sigma)` at `a_col`: `w` shape `(n,)`, `mu`/`sigma` shape
        `(n, 2)`, the pooled structural mixture alone, no per-source term."""
        i, t = self._interp_idx(a_col)
        w = self._w[arm_idx, i] + t * (self._w[arm_idx, i + 1] - self._w[arm_idx, i])
        mu = (self._mu[arm_idx, i, :]
              + t[:, np.newaxis] * (self._mu[arm_idx, i + 1, :] - self._mu[arm_idx, i, :]))
        sigma = (self._sigma[arm_idx, i, :]
                 + t[:, np.newaxis] * (self._sigma[arm_idx, i + 1, :]
                                        - self._sigma[arm_idx, i, :]))
        return w, mu, sigma

    def width_dex(self, a_col, map_class):
        """The structural mixture's own standard deviation in log10 T at
        `a_col`, no per-source term -- the shapes' per-node smoothing
        width."""
        a_col = np.asarray(a_col, dtype=float)
        w, mu, sigma = self._structural(a_col, self._arm_index(map_class))
        _, var = _mixture_mean_var(w, mu[:, 0], sigma[:, 0], mu[:, 1], sigma[:, 1])
        return np.sqrt(np.maximum(var, 0.0))

    def shift_dex(self, a_col, map_class):
        """The structural mixture's own mean in log10 T at `a_col`, no
        per-source term."""
        a_col = np.asarray(a_col, dtype=float)
        w, mu, sigma = self._structural(a_col, self._arm_index(map_class))
        mean, _ = _mixture_mean_var(w, mu[:, 0], sigma[:, 0], mu[:, 1], sigma[:, 1])
        return mean

    def mixture(self, a_col, sigma_col, map_class):
        """`(w, mu, sigma)`: `w (n,)`, `mu (n, 2)`, `sigma (n, 2)` -- the
        pooled structural mixture at `a_col`, with the source's own
        measurement uncertainty and, for Herschel, the field zero point
        added to each component's width in quadrature, both converted to
        dex at `a_col`."""
        a_col = np.asarray(a_col, dtype=float)
        sigma_col = np.asarray(sigma_col, dtype=float)
        mc = np.asarray(map_class)
        arm_idx = self._arm_index(mc)
        w, mu, sigma0 = self._structural(a_col, arm_idx)
        sigma_col_dex = sigma_col / (a_col * _LN10)
        zp_dex = np.where(mc == "herschel", self.zp_herschel_k / (a_col * _LN10), 0.0)
        extra_var = sigma_col_dex * sigma_col_dex + zp_dex * zp_dex
        sigma = np.sqrt(sigma0 * sigma0 + extra_var[:, np.newaxis])
        return w, mu, sigma

    def params(self, a_col, sigma_col, map_class):
        """`(mu, sigma)`, each `(n,)`: the mixture's exact overall mean and
        standard deviation in log10 T at `a_col`, per-source terms
        included -- what a consumer that treats the kernel as a single
        Gaussian needs."""
        w, mu, sigma = self.mixture(a_col, sigma_col, map_class)
        mean, var = _mixture_mean_var(w, mu[:, 0], sigma[:, 0], mu[:, 1], sigma[:, 1])
        return mean, np.sqrt(np.maximum(var, 0.0))

    def _broadcast_t(self, t, n):
        t = np.asarray(t, dtype=float)
        if t.ndim == 1:
            t = np.broadcast_to(t[np.newaxis, :], (n, t.shape[0]))
        return t

    def pdf(self, t, a_col, sigma_col, map_class):
        """`p(t | a_col)` in `T` (per unit `A_K`): `(n, m)`, `t` broadcast
        against the `n` sources; the mixture density."""
        a_col = np.asarray(a_col, dtype=float)
        w, mu, sigma = self.mixture(a_col, sigma_col, map_class)
        tt = self._broadcast_t(t, a_col.size)
        log10t = np.log10(tt)
        dens = np.zeros_like(log10t)
        wk = (w, 1.0 - w)
        for k, wc in enumerate(wk):
            loc = np.log10(a_col) + mu[:, k]
            z = (log10t - loc[:, np.newaxis]) / sigma[:, k][:, np.newaxis]
            dens += (wc[:, np.newaxis] * np.exp(-0.5 * z * z)
                     / (sigma[:, k][:, np.newaxis] * _SQRT2PI))
        return dens / (tt * _LN10)

    def cdf(self, t, a_col, sigma_col, map_class):
        """`P(T <= t | a_col)`: `(n, m)`, same broadcasting as `pdf`; the
        mixture CDF."""
        a_col = np.asarray(a_col, dtype=float)
        w, mu, sigma = self.mixture(a_col, sigma_col, map_class)
        tt = self._broadcast_t(t, a_col.size)
        log10t = np.log10(tt)
        out = np.zeros_like(log10t)
        wk = (w, 1.0 - w)
        for k, wc in enumerate(wk):
            loc = np.log10(a_col) + mu[:, k]
            z = (log10t - loc[:, np.newaxis]) / sigma[:, k][:, np.newaxis]
            out += wc[:, np.newaxis] * 0.5 * (1.0 + erf(z / _SQRT2))
        return out


def _pool_planck_mixture(subbeam_path, a_nodes):
    """Pools the sub-beam stage's fitted 302 arcsec mixture over regions,
    weighting each region's five parameters at a KA bin by that region's
    own bin counts there (a weighted average of the fitted numbers, not a
    refit of the pooled histogram -- simpler, and the per-region fits
    already share one forward model and one KA grid), skipping regions
    with too few counts to have a fit; interpolates the pooled numbers in
    `log A` onto `a_nodes`, clamped at the ends, skipping KA bins with no
    pooled value anywhere.

    Returns `(w, mu1, mu2, sigma1, sigma2)`, each `(len(a_nodes),)`, in
    natural-log units of `s = ln(T / A)` (converted to log10 by the
    caller).
    """
    with h5py.File(subbeam_path, "r") as f:
        w_r = f["MIX_W"][:, _POOL_BEAM_INDEX, :]
        mu1_r = f["MIX_MU1"][:, _POOL_BEAM_INDEX, :]
        mu2_r = f["MIX_MU2"][:, _POOL_BEAM_INDEX, :]
        sig1_r = f["MIX_SIG1"][:, _POOL_BEAM_INDEX, :]
        sig2_r = f["MIX_SIG2"][:, _POOL_BEAM_INDEX, :]
        ka_cent = f["MIX_KA_CENTRES"][:]
        counts_r = f["COND_KERNEL_%s" % _POOL_BEAM_LABEL][:].sum(axis=2).astype(np.float64)

    valid = np.isfinite(w_r) & (counts_r > 0)
    wt = np.where(valid, counts_r, 0.0)
    tot = wt.sum(axis=0)

    def pool(x):
        s = np.where(valid, x * wt, 0.0).sum(axis=0)
        out = np.full(tot.shape, np.nan)
        ok = tot > 0
        out[ok] = s[ok] / tot[ok]
        return out

    pooled = [pool(x) for x in (w_r, mu1_r, mu2_r, sig1_r, sig2_r)]
    finite = np.isfinite(pooled[0])
    ln_ka = ka_cent[finite]
    ln_nodes = np.log(a_nodes)
    return tuple(np.interp(ln_nodes, ln_ka, p[finite]) for p in pooled)


def build(config, regions=None):
    """Tabulates the pooled two-component log-normal mixture (weight, the
    two means, the two widths, all in log10 T) on every node of the
    column grid, for both arms at their own stated beam, and writes
    `bms/sesna/kernel_sesna_survey.hdf5`. The Planck arm pools the sub-beam
    stage's fitted 302 arcsec mixture over regions (`_pool_planck_mixture`);
    the Herschel arm has no sub-beam data at its own 36.3 arcsec beam, so
    its structural term is a point mass (`w = 0.5`, both means and both
    widths zero) -- disclosed, not extrapolated (SPEC_PRIORS.md section
    1.2). Survey-wide; `regions` is accepted and ignored.
    """
    from sesnaimpute.prior import column_grid

    subbeam_path = config_module.product_path(config, "sky/derived", "herschel",
                                              "subbeam", "region")
    zp = _load_sigma_zp_herschel(config)
    a_nodes = column_grid.nodes(config)
    n_node = a_nodes.size

    w_p, mu1_p, mu2_p, sig1_p, sig2_p = _pool_planck_mixture(subbeam_path, a_nodes)

    W = np.empty((len(_ARM_ORDER), n_node))
    MU = np.empty((len(_ARM_ORDER), n_node, 2))
    SIGMA = np.empty((len(_ARM_ORDER), n_node, 2))

    i_h, i_p = _ARM_CODE["herschel"], _ARM_CODE["planck"]
    W[i_h] = 0.5
    MU[i_h] = 0.0
    SIGMA[i_h] = 0.0

    W[i_p] = w_p
    MU[i_p, :, 0] = mu1_p / _LN10
    MU[i_p, :, 1] = mu2_p / _LN10
    SIGMA[i_p, :, 0] = sig1_p / _LN10
    SIGMA[i_p, :, 1] = sig2_p / _LN10

    codes = np.array([_ARM_CODE[arm] for arm in _ARM_ORDER], dtype=np.int64)

    out_path = config_module.product_path(config, "bms", "sesna", "kernel", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("A_NODES", data=a_nodes.astype(np.float64))
        f.create_dataset("MIX_W", data=W.astype(np.float64))
        f.create_dataset("MIX_MU", data=MU.astype(np.float64))
        f.create_dataset("MIX_SIGMA", data=SIGMA.astype(np.float64))
        f.create_dataset("MAP_CLASS_CODES", data=codes)
        f.create_dataset("ZP_HERSCHEL_K", data=np.float64(zp))

    print("kernel: %d arms x %d nodes (mixture) -> %s"
          % (len(_ARM_ORDER), n_node, out_path), flush=True)


if __name__ == "__main__":
    from sesnaimpute.build import run
    run(build)
