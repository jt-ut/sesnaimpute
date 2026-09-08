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
`sky.derived.herschel_column.field_zeropoints`), not one survey constant,
and a measured systematic left unapplied is an error of its own size: the
column stage (`sky.derived.column.merge_region`) now subtracts the
field's own offset from a Herschel-arm source's `A_COL_K` and writes the
offset's uncertainty as that source's own `ZP_SIGMA_K` (0 for a
Planck-arm source). `Kernel.mixture`/`.params` take that
same per-source `ZP_SIGMA_K` directly (mag, already resolved to the
source's field, or 0) rather than a field name -- simpler than having the
kernel carry its own field lookup, since the column product has already
done the join. Omitting it (every call site not yet updated) falls back
to the survey-wide RMS `ZP_HERSCHEL_K` for every Herschel source, exactly
the pre-fix behaviour.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import progress

#: The beam the sub-beam mixture table is pooled from for the Planck arm
#: (index into the sub-beam product's beam axis, order L108/L302/L821).
_POOL_BEAM_INDEX = 1

#: The two beams the kernel is ever evaluated at (spec 1.2), and the fixed
#: order/codes the tabulated product's arm axis uses.
_ARM_ORDER = ("herschel", "planck")
_ARM_CODE = {"herschel": 0, "planck": 1}

_LN10 = float(np.log(10.0))


def _load_sigma_zp_herschel(config):
    """`SIGMA_ZP_K`: the survey-wide RMS of the per-field Herschel zero
    points (SPEC_PRIORS.md section 1.2), from
    `sky.derived.herschel_column.write_field_zeropoint` -- the fallback a
    caller with no per-source `ZP_SIGMA_K` of its own gets."""
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
        #: the survey-wide RMS of the per-field zero points -- the
        #: fallback for a Herschel source whose call site does not yet
        #: pass its own `ZP_SIGMA_K` (owner, 2026-09-06).
        self.zp_herschel_k = float(zp_herschel_k)

    @classmethod
    def read(cls, config):
        path = config_module.product_path(config, "population", "sesna", "kernel", "survey")
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
        """`(n,)` intp, `_ARM_CODE[arm]` per entry of `map_class`, which
        arrives in either of two forms: the arm's own name (a majority-
        vote map class such as `population.yso`'s `_majority_map_class` or
        `population.star_shapes`'s per-tile `MAP_CLASS`, as `str` or, read
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

    def _zp_herschel_dex(self, a_col, arm_idx, zp_sigma_k):
        """The zero-point term folded into a Herschel-arm source's width,
        in dex at `a_col`: the source's own `ZP_SIGMA_K` (mag, already
        resolved to its field by `sky.derived.column.merge_region`) where
        given, else the survey-wide RMS `zp_herschel_k` (owner,
        2026-09-06). Zero for a Planck-arm source either way. `zp_sigma_k`
        is optional so every existing caller (none pass it yet) is
        unaffected."""
        is_h = arm_idx == _ARM_CODE["herschel"]
        if zp_sigma_k is None:
            zp_ak = np.where(is_h, self.zp_herschel_k, 0.0)
        else:
            zp_ak = np.where(is_h, np.asarray(zp_sigma_k, dtype=float), 0.0)
        return zp_ak / (a_col * _LN10)

    def mixture(self, a_col, sigma_col, map_class, zp_sigma_k=None):
        """`(w, mu, sigma)`: `w (n,)`, `mu (n, 2)`, `sigma (n, 2)` -- the
        pooled structural mixture at `a_col`, with the source's own
        measurement uncertainty and, for Herschel, the zero point's own
        uncertainty added to each component's width in quadrature, both
        converted to dex at `a_col`. `zp_sigma_k`, one per source (mag,
        0 for Planck-arm), is optional; omitting it (every call site not
        yet wired) uses the survey-wide zero point for every Herschel
        source, as before the per-field fix."""
        a_col = np.asarray(a_col, dtype=float)
        sigma_col = np.asarray(sigma_col, dtype=float)
        arm_idx = self._arm_index(map_class)
        w, mu, sigma0 = self._structural(a_col, arm_idx)
        sigma_col_dex = sigma_col / (a_col * _LN10)
        # the same arm index `_arm_index` already resolved, not a second,
        # independently-typed string comparison against `map_class`
        # (the bug this fix removes: `mc == "herschel"` silently failed
        # for a numeric or bytes `map_class`, zeroing the zero point).
        zp_dex = self._zp_herschel_dex(a_col, arm_idx, zp_sigma_k)
        extra_var = sigma_col_dex * sigma_col_dex + zp_dex * zp_dex
        sigma = np.sqrt(sigma0 * sigma0 + extra_var[:, np.newaxis])
        return w, mu, sigma

    def params(self, a_col, sigma_col, map_class, zp_sigma_k=None):
        """`(mu, sigma)`, each `(n,)`: the mixture's exact overall mean and
        standard deviation in log10 T at `a_col`, per-source terms
        included -- what a consumer that treats the kernel as a single
        Gaussian needs. `zp_sigma_k` is the same optional per-source
        zero-point uncertainty `mixture` takes."""
        w, mu, sigma = self.mixture(a_col, sigma_col, map_class, zp_sigma_k=zp_sigma_k)
        mean, var = _mixture_mean_var(w, mu[:, 0], sigma[:, 0], mu[:, 1], sigma[:, 1])
        return mean, np.sqrt(np.maximum(var, 0.0))



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
    from sesnaimpute.population import column_grid

    st = progress.Stage("prior.kernel")
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

    out_path = config_module.product_path(config, "population", "sesna", "kernel", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("A_NODES", data=a_nodes.astype(np.float64))
        f.create_dataset("MIX_W", data=W.astype(np.float64))
        f.create_dataset("MIX_MU", data=MU.astype(np.float64))
        f.create_dataset("MIX_SIGMA", data=SIGMA.astype(np.float64))
        f.create_dataset("ZP_HERSCHEL_K", data=np.float64(zp))

    st.done(out_path, n_arms=len(_ARM_ORDER), n_node=n_node, zp_herschel_k=float(zp))
    print("kernel: %d arms x %d nodes (mixture), zp_herschel_k=%.4f -> %s"
          % (len(_ARM_ORDER), n_node, zp, out_path), flush=True)


if __name__ == "__main__":
    from sesnaimpute.build import run
    run(build)
