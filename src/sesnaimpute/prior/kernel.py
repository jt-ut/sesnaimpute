"""The column kernel `p(T | A_measured)` (SPEC_PRIORS.md section 1.2): a
mixture of two log-normals in the true column,

    log10 T ~ w * Normal(log10 A_s + mu_1, sigma_1)
              + (1 - w) * Normal(log10 A_s + mu_2, sigma_2)

`w`, `mu_1`, `mu_2`, `sigma_1`, `sigma_2` are the sub-beam stage's fitted
structural mixture (`sesnaimpute.sky.derived.subbeam`, its noise-corrected
forward model), tabulated on the column grid, for each arm at its own
stated beam. The Planck arm is built from the sub-beam stage's own
SURVEY-POOLED 302 arcsec fit (`MIX_POOLED_*`: the regions' raw histograms
summed count-for-count, then refit with the same forward model and a
count-weighted pooled noise level) -- not an average of the regions'
separately-fitted parameters, which is not the pooled distribution.
Planck is used only where no Herschel map covers a source, so one
survey-wide pooled fit, not a per-region one, is the right object for it.
The Herschel arm's own stated beam (36.3 arcsec) is the finest map there
is -- no sub-beam data exists below it -- so its structural term is a
point mass at `T = A_s` (zero width, zero shift): its kernel is the
per-source measurement uncertainty and the field zero point alone, not
an extrapolation. Both are added to each component's sigma in quadrature,
at the source's own column, converted to dex.

The zero point is one systematic per field (owner, 2026-09-06;
`sky.derived.herschel_column.field_zeropoints`), not one survey constant:
what folds into a Herschel-arm source's width is its own field's
`ZP_SIGMA_FIELD` -- the uncertainty OF the per-field offset, not the
offset itself, which the column stage never adds to `A_COL_K` (it is
carried only as a sigma term; `sky.derived.column.merge_region` copies
the Herschel arm's value through unchanged). `Kernel.mixture`/`.params`
take an optional `field` array; when a source's field is not given, or
is a field with no fit, the survey-wide RMS of the per-field values
(`ZP_HERSCHEL_K`) stands in, so every existing call site is unaffected.
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
    """`SIGMA_ZP_K`: the survey-wide RMS of the per-field Herschel zero
    points (SPEC_PRIORS.md section 1.2), from
    `sky.derived.herschel_column.write_field_zeropoint`. Kept as its own
    reader, unchanged, for the one scalar a field-less caller falls back
    to; `_load_herschel_field_zeropoints` reads the per-field table."""
    path = config_module.product_path(config, "sky/derived", "herschel",
                                      "sigma", "survey")
    with h5py.File(path, "r") as f:
        return float(f["SIGMA_ZP_K"][()])


def _load_herschel_field_zeropoints(config):
    """`(names, zp_sigma_field)`: the per-field Herschel zero-point
    uncertainty (`ZP_SIGMA_FIELD`, keyed by `FIELD_NAME`) from the same
    product `_load_sigma_zp_herschel` reads its scalar from. Empty lists
    if the product predates the per-field measurement (owner, 2026-09-06)
    -- a caller then gets the survey-wide scalar for every source, exactly
    as before."""
    path = config_module.product_path(config, "sky/derived", "herschel", "sigma", "survey")
    with h5py.File(path, "r") as f:
        if "FIELD_NAME" not in f:
            return [], np.empty(0, dtype=np.float64)
        names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["FIELD_NAME"][:]]
        zp_sigma_field = np.asarray(f["ZP_SIGMA_FIELD"][:], dtype=np.float64)
    return names, zp_sigma_field


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

    def __init__(self, a_nodes, w, mu, sigma, zp_herschel_k, field_names=(), zp_sigma_field=()):
        self._a_nodes = np.asarray(a_nodes, dtype=float)
        self._ln_nodes = np.log(self._a_nodes)
        self._w = np.asarray(w, dtype=float)          # (n_arm, n_node)
        self._mu = np.asarray(mu, dtype=float)         # (n_arm, n_node, 2)
        self._sigma = np.asarray(sigma, dtype=float)    # (n_arm, n_node, 2)
        self.zp_herschel_k = float(zp_herschel_k)
        #: field name -> ZP_SIGMA_FIELD (mag), the per-field zero-point
        #: uncertainty (owner, 2026-09-06); empty if the kernel product
        #: predates it, so every lookup falls back to `zp_herschel_k`.
        self._zp_sigma_by_field = dict(zip(field_names, (float(v) for v in zp_sigma_field)))

    @classmethod
    def read(cls, config):
        path = config_module.product_path(config, "bms", "sesna", "kernel", "survey")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "prior.kernel: no kernel product at %s -- run the "
                "'prior.kernel' RUNBOOK line first" % path)
        with h5py.File(path, "r") as f:
            field_names = ([x.decode("utf-8") if isinstance(x, bytes) else str(x)
                            for x in f["ZP_FIELD_NAME"][:]] if "ZP_FIELD_NAME" in f else [])
            zp_sigma_field = f["ZP_SIGMA_FIELD"][:] if "ZP_SIGMA_FIELD" in f else []
            return cls(f["A_NODES"][:], f["MIX_W"][:], f["MIX_MU"][:],
                       f["MIX_SIGMA"][:], float(f["ZP_HERSCHEL_K"][()]),
                       field_names, zp_sigma_field)

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
        """`(n,)` intp, `_ARM_CODE[arm]` per entry of `map_class`, which
        arrives in either of two forms: the arm's own name (a majority-
        vote map class such as `prior.yso`'s `_majority_map_class` or
        `prior.star_shapes`'s per-tile `MAP_CLASS`, as `str` or, read
        back from HDF5, fixed-length `bytes`), or already the numeric
        provenance code (`sky.derived.column`'s `A_COL_PROVENANCE`, 0/1,
        exactly `_ARM_CODE`'s own values). Comparing a numeric or `bytes`
        array against the `str` literals below silently returns a single
        scalar `False` (a numpy `FutureWarning`, "elementwise comparison
        failed"), which left every source's index at its `np.zeros`
        default -- arm 0, Herschel, regardless of the source's real arm.
        Every branch here is an exact, same-dtype comparison instead."""
        mc = np.asarray(map_class)
        if np.issubdtype(mc.dtype, np.integer):
            idx = mc.astype(np.intp)
            bad = (idx < 0) | (idx > max(_ARM_CODE.values()))
            if np.any(bad):
                raise ValueError(
                    "prior.kernel: map_class carries a numeric provenance code outside "
                    "%r" % (sorted(_ARM_CODE.values()),))
            return idx
        if mc.dtype.kind == "S":
            mc = mc.astype("U")
        resolved = np.zeros(mc.shape, dtype=bool)
        idx = np.zeros(mc.shape, dtype=np.intp)
        for arm in _ARM_ORDER:
            hit = mc == arm
            idx[hit] = _ARM_CODE[arm]
            resolved |= hit
        if not np.all(resolved):
            raise ValueError(
                "prior.kernel: map_class carries a value outside %r" % (_ARM_ORDER,))
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

    def _zp_herschel_dex(self, a_col, arm_idx, field):
        """The zero-point term folded into a Herschel-arm source's width,
        in dex at `a_col`: that source's own field's `ZP_SIGMA_FIELD` where
        `field` names one and the kernel has a fit for it, else the
        survey-wide RMS `zp_herschel_k` (owner, 2026-09-06). Zero for a
        Planck-arm source, as before. `field` is optional and defaults to
        the old, field-less behaviour -- every existing caller is
        unaffected."""
        is_h = arm_idx == _ARM_CODE["herschel"]
        if field is None or not self._zp_sigma_by_field:
            zp_ak = np.where(is_h, self.zp_herschel_k, 0.0)
        else:
            field = np.asarray(field)
            if field.dtype.kind == "S":
                field = field.astype("U")
            zp_ak = np.array([self._zp_sigma_by_field.get(str(fld), self.zp_herschel_k)
                              for fld in field], dtype=float)
            zp_ak = np.where(is_h, zp_ak, 0.0)
        return zp_ak / (a_col * _LN10)

    def mixture(self, a_col, sigma_col, map_class, field=None):
        """`(w, mu, sigma)`: `w (n,)`, `mu (n, 2)`, `sigma (n, 2)` -- the
        pooled structural mixture at `a_col`, with the source's own
        measurement uncertainty and, for Herschel, the field zero point
        added to each component's width in quadrature, both converted to
        dex at `a_col`. `field`, one field name per source, is optional;
        omitting it (every current call site does) uses the survey-wide
        zero point for every Herschel source, as before."""
        a_col = np.asarray(a_col, dtype=float)
        sigma_col = np.asarray(sigma_col, dtype=float)
        arm_idx = self._arm_index(map_class)
        w, mu, sigma0 = self._structural(a_col, arm_idx)
        sigma_col_dex = sigma_col / (a_col * _LN10)
        # the same arm index `_arm_index` already resolved, not a second,
        # independently-typed string comparison against `map_class`
        # (the bug this fix removes: `mc == "herschel"` silently failed
        # for a numeric or bytes `map_class`, zeroing the zero point).
        zp_dex = self._zp_herschel_dex(a_col, arm_idx, field)
        extra_var = sigma_col_dex * sigma_col_dex + zp_dex * zp_dex
        sigma = np.sqrt(sigma0 * sigma0 + extra_var[:, np.newaxis])
        return w, mu, sigma

    def params(self, a_col, sigma_col, map_class, field=None):
        """`(mu, sigma)`, each `(n,)`: the mixture's exact overall mean and
        standard deviation in log10 T at `a_col`, per-source terms
        included -- what a consumer that treats the kernel as a single
        Gaussian needs. `field` is the same optional per-source field name
        `mixture` takes."""
        w, mu, sigma = self.mixture(a_col, sigma_col, map_class, field=field)
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
    """Reads the sub-beam stage's own survey-pooled 302 arcsec mixture fit
    (`MIX_POOLED_*`: the regions' raw histograms summed count-for-count and
    refit with the same forward model and a count-weighted pooled noise
    level -- the pooled distribution is the count-weighted mixture of the
    regions' distributions, not an average of their fitted parameters) and
    interpolates its five numbers in `log A` onto `a_nodes`, clamped at the
    ends, skipping KA bins with no pooled fit.

    Returns `(w, mu1, mu2, sigma1, sigma2)`, each `(len(a_nodes),)`, in
    natural-log units of `s = ln(T / A)` (converted to log10 by the
    caller).
    """
    with h5py.File(subbeam_path, "r") as f:
        w_p = f["MIX_POOLED_W"][_POOL_BEAM_INDEX, :]
        mu1_p = f["MIX_POOLED_MU1"][_POOL_BEAM_INDEX, :]
        mu2_p = f["MIX_POOLED_MU2"][_POOL_BEAM_INDEX, :]
        sig1_p = f["MIX_POOLED_SIG1"][_POOL_BEAM_INDEX, :]
        sig2_p = f["MIX_POOLED_SIG2"][_POOL_BEAM_INDEX, :]
        ka_cent = f["MIX_KA_CENTRES"][:]

    pooled = (w_p, mu1_p, mu2_p, sig1_p, sig2_p)
    finite = np.isfinite(pooled[0])
    ln_ka = ka_cent[finite]
    ln_nodes = np.log(a_nodes)
    return tuple(np.interp(ln_nodes, ln_ka, p[finite]) for p in pooled)


def build(config, regions=None):
    """Tabulates the pooled two-component log-normal mixture (weight, the
    two means, the two widths, all in log10 T) on every node of the
    column grid, for both arms at their own stated beam, and writes
    `bms/sesna/kernel_sesna_survey.hdf5`. The Planck arm reads the sub-beam
    stage's own survey-pooled 302 arcsec mixture fit (`_pool_planck_mixture`,
    `MIX_POOLED_*` -- the regions' histograms summed and refit, not their
    fitted parameters averaged); the Herschel arm has no sub-beam data at
    its own 36.3 arcsec beam, so
    its structural term is a point mass (`w = 0.5`, both means and both
    widths zero) -- disclosed, not extrapolated (SPEC_PRIORS.md section
    1.2). Survey-wide; `regions` is accepted and ignored.
    """
    from sesnaimpute.prior import column_grid

    subbeam_path = config_module.product_path(config, "sky/derived", "herschel",
                                              "subbeam", "region")
    zp = _load_sigma_zp_herschel(config)
    zp_field_names, zp_sigma_field = _load_herschel_field_zeropoints(config)
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
        if zp_field_names:
            f.create_dataset("ZP_FIELD_NAME", data=np.array([n.encode("utf-8") for n in zp_field_names]))
            f.create_dataset("ZP_SIGMA_FIELD", data=np.asarray(zp_sigma_field, dtype=np.float64))

    print("kernel: %d arms x %d nodes (mixture), zp_herschel_k=%.4f, %d fields -> %s"
          % (len(_ARM_ORDER), n_node, zp, len(zp_field_names), out_path), flush=True)


if __name__ == "__main__":
    from sesnaimpute.build import run
    run(build)
