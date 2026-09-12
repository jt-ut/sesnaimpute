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
The Herschel arm carries its own structural term at its own stated beam,
36.3 arcsec (`W_ABS_36P3`): the sub-beam stage tabulates a two-scale
mixture only at 108/302/821 arcsec, none of them the Herschel arm's own
beam, so its 36.3 arcsec mixture is the 108 arcsec pooled mixture's own
shape, stretched by the stage's own completion factor and the region's
own absolute-width ratio between the two beams, then recentred so
`E[T / A_beam] = 1` in linear units at every node -- the beam column is
the mean of its own pencils (`_pool_herschel_ref_mixture`). The
per-source measurement uncertainty and the field zero point are added to
each component's sigma in quadrature, at the source's own column,
converted to dex, for both arms alike.

The kernel is also class-conditional: `Kernel.mixture`'s `exponent`
keyword reweights the mixture by `T ** exponent`, the star-gas law's own
column exponent (SPEC_BMSTP_DRAFT.md section 5.5) -- where members of a
class form in proportion to a power of the column, that class's own
member sits off the beam mean by that same power, within the beam.

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

    def mixture(self, a_col, sigma_col, map_class, zp_sigma_k=None, exponent=0.0):
        """`(w, mu, sigma)`: `w (n,)`, `mu (n, 2)`, `sigma (n, 2)` -- the
        pooled structural mixture at `a_col`, with the source's own
        measurement uncertainty and, for Herschel, the zero point's own
        uncertainty added to each component's width in quadrature, both
        converted to dex at `a_col`. `zp_sigma_k`, one per source (mag,
        0 for Planck-arm), is optional; omitting it (every call site not
        yet wired) uses the survey-wide zero point for every Herschel
        source, as before the per-field fix.

        `exponent` is the star-gas law's own column exponent, `gamma` in
        `p(T | A_beam, C) propto T**gamma * p(T | A_beam)`
        (SPEC_BMSTP_DRAFT.md section 5.5): a class whose members form in
        proportion to `A**gamma` of the column sits, within the beam, in
        the pencils that carry `T**gamma` more of that class's members,
        not at the beam's own mean pencil. For one component,
        `log10 T = log10 A + y`, `y ~ N(mu, sigma**2)` in dex, weighting
        by `T**gamma` is an exponential tilt of `y` by `gamma * ln 10`,
        exact in closed form: `sigma' = sigma` (a tilted Gaussian is a
        Gaussian of the same width), `mu' = mu + gamma * sigma**2 * ln 10`,
        and the component keeps its normalising mass, `w' propto
        w * exp(gamma * ln10 * mu + (gamma * ln10 * sigma)**2 / 2)`,
        renormalised over the two components. Applied after the
        measurement and zero-point terms are folded into `sigma` (they
        are already part of the same lognormal by then). `exponent = 0.0`
        (the default) recovers the unweighted kernel exactly."""
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
        if exponent == 0.0:
            return w, mu, sigma
        c = exponent * _LN10
        mu_tilt = mu + c * sigma * sigma
        log_wt = c * mu + (c * sigma) ** 2 / 2.0
        log_wt -= log_wt.max(axis=1, keepdims=True)
        wt = np.stack([w, 1.0 - w], axis=1) * np.exp(log_wt)
        w_tilt = wt[:, 0] / wt.sum(axis=1)
        return w_tilt, mu_tilt, sigma

    def params(self, a_col, sigma_col, map_class, zp_sigma_k=None):
        """`(mu, sigma)`, each `(n,)`: the mixture's exact overall mean and
        standard deviation in log10 T at `a_col`, per-source terms
        included -- what a consumer that treats the kernel as a single
        Gaussian needs. `zp_sigma_k` is the same optional per-source
        zero-point uncertainty `mixture` takes. Always the unweighted
        kernel (`mixture`'s `exponent = 0.0`), its own meaning kept."""
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


#: The sub-beam stage's own ladder beam nearest the Herschel arm's own
#: 36.3 arcsec beam (index into the sub-beam product's beam axis, order
#: L108/L302/L821) -- the two-scale mixture SHAPE the Herschel arm's own
#: reference-beam term is stretched from.
_REF_SHAPE_BEAM_INDEX = 0


def _pool_herschel_ref_mixture(subbeam_path, a_nodes):
    """Forms the Herschel arm's own survey-pooled two-lognormal structural
    mixture at its stated 36.3 arcsec beam.

    The sub-beam stage tabulates a two-scale mixture only at 108, 302 and
    821 arcsec (`MIX_POOLED_*`), none of them the Herschel arm's own beam,
    and stores no pooled mixture at the reference scale, so this mixture
    is formed from the per-region fits by the stage's own rule: the 108
    arcsec pooled mixture's own SHAPE (`MIX_POOLED_*` at
    `_REF_SHAPE_BEAM_INDEX`, the nearest tabulated beam) is stretched by
    one region-independent factor, `W_ABS_36P3 / (W_ABS_L108 /
    COMPLETION_L108)` -- the region's own absolute pencil-to-36.3-beam
    width divided by its own two-scale sigma at 108 arcsec, the stage's
    own `completion factor` (module docstring) chained with the beams'
    own absolute-width ratio -- pooled across regions count-weighted by
    each region's own counts in the 108 arcsec conditional histogram, the
    same weighting the stage's own survey pool uses. One factor for every
    node: a power-law spectrum stretches the whole log-column
    distribution by one factor across column, per the stage's own model,
    not a per-node one.

    Returns `(w, mu1, mu2, sigma1, sigma2, factor)`: the first five each
    `(len(a_nodes),)`, in natural-log units of `s = ln(T / A)` (converted
    to log10 by the caller, and NOT yet recentred to `E[T / A] = 1`);
    `factor` the pooled stretch, for the report.
    """
    with h5py.File(subbeam_path, "r") as f:
        w_r = f["MIX_POOLED_W"][_REF_SHAPE_BEAM_INDEX, :]
        mu1_r = f["MIX_POOLED_MU1"][_REF_SHAPE_BEAM_INDEX, :]
        mu2_r = f["MIX_POOLED_MU2"][_REF_SHAPE_BEAM_INDEX, :]
        sig1_r = f["MIX_POOLED_SIG1"][_REF_SHAPE_BEAM_INDEX, :]
        sig2_r = f["MIX_POOLED_SIG2"][_REF_SHAPE_BEAM_INDEX, :]
        ka_cent = f["MIX_KA_CENTRES"][:]
        w_abs_ref = f["W_ABS_36P3"][:]
        w_abs_108 = f["W_ABS_L108"][:]
        completion_108 = f["COMPLETION_L108"][:]
        counts_108 = f["COND_KERNEL_L108"][:].sum(axis=(1, 2)).astype(np.float64)

    stretch = completion_108 * (w_abs_ref / w_abs_108)
    ok = np.isfinite(stretch) & (counts_108 > 0)
    factor = float(np.sum(counts_108[ok] * stretch[ok]) / np.sum(counts_108[ok]))

    pooled = (w_r, mu1_r, mu2_r, sig1_r, sig2_r)
    finite = np.isfinite(pooled[0])
    ln_ka = ka_cent[finite]
    ln_nodes = np.log(a_nodes)
    w, mu1, mu2, sig1, sig2 = (np.interp(ln_nodes, ln_ka, p[finite]) for p in pooled)
    return w, mu1 * factor, mu2 * factor, sig1 * factor, sig2 * factor, factor


def build(config, regions=None):
    """Tabulates the pooled two-component log-normal mixture (weight, the
    two means, the two widths, all in log10 T) on every node of the
    column grid, for both arms at their own stated beam, and writes
    `bms/sesna/kernel_sesna_survey.hdf5`. The Planck arm reads the sub-beam
    stage's own survey-pooled 302 arcsec mixture fit (`_pool_planck_mixture`,
    `MIX_POOLED_*` -- the regions' histograms summed and refit, not their
    fitted parameters averaged); the Herschel arm reads its own 36.3 arcsec
    mixture (`_pool_herschel_ref_mixture`), then recentred so
    `E[T / A_beam] = 1` in linear units at every node -- the beam column is
    the mean of its own pencils -- by one uniform per-node shift on both
    components' means (SPEC_BMSTP_DRAFT.md section 2). Survey-wide;
    `regions` is accepted and ignored.
    """
    from sesnaimpute.population import column_grid

    st = progress.Stage("prior.kernel")
    subbeam_path = config_module.product_path(config, "sky/derived", "herschel",
                                              "subbeam", "region")
    zp = _load_sigma_zp_herschel(config)
    a_nodes = column_grid.nodes(config)
    n_node = a_nodes.size

    w_p, mu1_p, mu2_p, sig1_p, sig2_p = _pool_planck_mixture(subbeam_path, a_nodes)
    w_h, mu1_h, mu2_h, sig1_h, sig2_h, herschel_factor = _pool_herschel_ref_mixture(
        subbeam_path, a_nodes)

    W = np.empty((len(_ARM_ORDER), n_node))
    MU = np.empty((len(_ARM_ORDER), n_node, 2))
    SIGMA = np.empty((len(_ARM_ORDER), n_node, 2))

    i_h, i_p = _ARM_CODE["herschel"], _ARM_CODE["planck"]
    W[i_h] = w_h
    MU[i_h, :, 0] = mu1_h / _LN10
    MU[i_h, :, 1] = mu2_h / _LN10
    SIGMA[i_h, :, 0] = sig1_h / _LN10
    SIGMA[i_h, :, 1] = sig2_h / _LN10

    # E[T / A_beam] = 1 (the beam column is the mean of its own pencils):
    # one uniform dex shift per node on both components' means, solved so
    # the mixture's own linear-space mean is exactly one (a lognormal
    # component's own linear mean is `10 ** (mu + sigma**2 * ln10 / 2)`;
    # shifting both means by the same amount scales both components, and
    # so the whole mixture, by the same factor).
    e_raw = (W[i_h] * 10.0 ** (MU[i_h, :, 0] + SIGMA[i_h, :, 0] ** 2 * _LN10 / 2.0)
             + (1.0 - W[i_h]) * 10.0 ** (MU[i_h, :, 1] + SIGMA[i_h, :, 1] ** 2 * _LN10 / 2.0))
    herschel_offset = -np.log10(e_raw)
    MU[i_h, :, 0] += herschel_offset
    MU[i_h, :, 1] += herschel_offset

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

    st.done(out_path, n_arms=len(_ARM_ORDER), n_node=n_node, zp_herschel_k=float(zp),
            herschel_stretch=herschel_factor)
    print("kernel: %d arms x %d nodes (mixture), zp_herschel_k=%.4f, "
          "herschel_stretch=%.4f -> %s"
          % (len(_ARM_ORDER), n_node, zp, herschel_factor, out_path), flush=True)


if __name__ == "__main__":
    from sesnaimpute.build import run
    run(build)
