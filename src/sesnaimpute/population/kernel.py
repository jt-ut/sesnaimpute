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

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.special import erf

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
_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))

#: R4's calibration grid for `CLOUD_SIGMA_HERSCHEL_DEX` (this brief): 0.03
#: to 0.80 dex in 0.01-dex steps, the profile likelihood's own domain.
_CLOUD_SIGMA_GRID_LO = 0.03
_CLOUD_SIGMA_GRID_HI = 0.80
_CLOUD_SIGMA_GRID_STEP = 0.01

#: The half-drop in log-likelihood (a chi-square difference of 1 for one
#: profiled parameter) that bounds `CLOUD_SIGMA_HERSCHEL_DEX`'s own 68%
#: interval (this brief, R4).
_CLOUD_SIGMA_DLOGLIKE = 0.5

#: The two regions with a Herschel arm whose Class 0/I/flat protostars
#: calibrate the cloud-class structural width (R4): HOPS's own Orion A,
#: eHOPS's own Aquila (`sky.derived.protostars`).
_PROTOSTAR_REGIONS = ("Orion A", "Aquila")
_PROTOSTAR_CLASSES = (b"0", b"I", b"flat")

#: A numerical floor on `u` before `log10` (the sightline's own inner
#: edge, `u = 0` at `d = 0`) -- a guard against `log10(0)`, not a
#: physical constant.
_U_LOG_FLOOR = 1.0e-6


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


def _norm_cdf(z):
    """`Phi(z)`, the standard normal CDF, off the exact `erf`."""
    return 0.5 * (1.0 + erf(z / _SQRT2))


def _rescale_structural_to_sigma_cloud(w, mu, sigma0, sigma_cloud):
    """R4: the two-component structural mixture (`mu`, `sigma0`, last axis
    size 2; `w` the same leading shape), with both components' sigma
    scaled by ONE factor so the mixture's own structural width (the
    mean-subtracted second moment `_mixture_mean_var` gives from the
    UNSCALED `mu`/`sigma0`) equals `sigma_cloud` -- `sigma_cloud`
    broadcasts against `w` (a scalar per source, or a trial grid the
    caller has already shaped to broadcast against `w`'s own axis, e.g.
    `(n_grid, 1)` against `w`'s `(n_proto,)`). Scaling leaves the
    component MEANS untouched, so the between-component spread `d0`,
    `d1` (mean differences) is exactly what it was before scaling; only
    the within-component variance needs the scale factor `k`:

        sigma_cloud**2 = k**2 * (w*sigma0_0**2 + (1-w)*sigma0_1**2)
                          + (w*d0**2 + (1-w)*d1**2)

    `k` clipped at 0 where `sigma_cloud` cannot be reached by scaling
    alone (the between-component spread already exceeds it) -- a
    degenerate (zero-width) component there, not a failure: the
    measurement and zero-point terms `mixture` adds next keep the total
    width positive. Recentred to mean one afterward (the build's own
    Herschel recentring, generalised to whatever shape `w` carries): a
    lognormal component's width sets its own linear mean, so scaling
    sigma without re-centring would leave `E[T/A_beam] != 1`. Returns
    `(mu_new, sigma_new)`, the same trailing shape as `mu`/`sigma0`.
    """
    mu0, mu1 = mu[..., 0], mu[..., 1]
    s0, s1 = sigma0[..., 0], sigma0[..., 1]
    mean0 = w * mu0 + (1.0 - w) * mu1
    d0, d1 = mu0 - mean0, mu1 - mean0
    between = w * d0 * d0 + (1.0 - w) * d1 * d1
    within = w * s0 * s0 + (1.0 - w) * s1 * s1
    k = np.sqrt(np.clip((sigma_cloud * sigma_cloud - between) / within, 0.0, None))
    s0n, s1n = s0 * k, s1 * k
    lin_mean = (w * 10.0 ** (mu0 + s0n * s0n * _LN10 / 2.0)
                + (1.0 - w) * 10.0 ** (mu1 + s1n * s1n * _LN10 / 2.0))
    offset = -np.log10(lin_mean)
    mu_new = np.stack([mu0 + offset, mu1 + offset], axis=-1)
    sigma_new = np.stack([s0n, s1n], axis=-1)
    return mu_new, sigma_new


class Kernel(object):
    """The column kernel: a two-component log-normal mixture in `log10 T`
    per arm, tabulated on the column grid (SPEC_PRIORS.md section 1.2).
    Built by `build(config)`, loaded by `read(config)`.
    """

    def __init__(self, a_nodes, w, mu, sigma, zp_herschel_k, cloud_sigma_herschel_dex):
        self._a_nodes = np.asarray(a_nodes, dtype=float)
        self._ln_nodes = np.log(self._a_nodes)
        self._w = np.asarray(w, dtype=float)          # (n_arm, n_node)
        self._mu = np.asarray(mu, dtype=float)         # (n_arm, n_node, 2)
        self._sigma = np.asarray(sigma, dtype=float)    # (n_arm, n_node, 2)
        #: the survey-wide RMS of the per-field zero points -- the
        #: fallback for a Herschel source whose call site does not yet
        #: pass its own `ZP_SIGMA_K` (owner, 2026-09-06).
        self.zp_herschel_k = float(zp_herschel_k)
        #: R4: the Herschel arm's cloud-class (`exponent > 0`) structural
        #: width, dex, calibrated in `build` on the HOPS/eHOPS protostars.
        self.cloud_sigma_herschel_dex = float(cloud_sigma_herschel_dex)

    @classmethod
    def read(cls, config):
        path = config_module.product_path(config, "population", "sesna", "kernel", "survey")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "prior.kernel: no kernel product at %s -- run the "
                "'prior.kernel' RUNBOOK line first" % path)
        with h5py.File(path, "r") as f:
            return cls(f["A_NODES"][:], f["MIX_W"][:], f["MIX_MU"][:],
                       f["MIX_SIGMA"][:], float(f["ZP_HERSCHEL_K"][()]),
                       float(f["CLOUD_SIGMA_HERSCHEL_DEX"][()]))

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
        (the default) recovers the unweighted kernel exactly.

        R4: for `exponent != 0.0`, the Herschel arm's structural width is
        first replaced (`_rescale_structural_to_sigma_cloud`) by the
        protostar-calibrated `self.cloud_sigma_herschel_dex`, before the
        measurement/zero-point terms below or this tilt -- the sub-beam
        stage's own structural width is a lower bound toward cores, not
        the cloud classes' own scale. `cloud_sigma_herschel_dex` is fit in
        `build` (`_fit_cloud_sigma_herschel`) on the HOPS (Orion A) and
        eHOPS (Aquila) Class 0/I/flat protostars: each protostar's own
        foreground `A_V`, converted to `A_K`, against its own nside-256
        sightline's beam column, profiled against the region's own
        cloud-interval placement of a member star. SESNA's own YSO fits
        validate or replace this number per region once they exist.
        `exponent = 0.0` and the Planck arm are untouched by this."""
        a_col = np.asarray(a_col, dtype=float)
        sigma_col = np.asarray(sigma_col, dtype=float)
        arm_idx = self._arm_index(map_class)
        w, mu, sigma0 = self._structural(a_col, arm_idx)
        if exponent != 0.0:
            is_h = arm_idx == _ARM_CODE["herschel"]
            if np.any(is_h):
                mu = mu.copy()
                sigma0 = sigma0.copy()
                mu[is_h], sigma0[is_h] = _rescale_structural_to_sigma_cloud(
                    w[is_h], mu[is_h], sigma0[is_h], self.cloud_sigma_herschel_dex)
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


def _interp_structural(a_nodes, ln_nodes, w_arm, mu_arm, sigma_arm, a_col):
    """`(w, mu, sigma)` at `a_col`: linear interpolation in `log A` of one
    arm's own tabulated structural mixture (`w_arm` `(n_node,)`, `mu_arm`/
    `sigma_arm` `(n_node, 2)`), clamped at the grid ends -- the same
    interpolation `Kernel._structural` applies per source, shared here so
    R4's fit (run inside `build`, before a `Kernel` exists to read) and
    `Kernel._structural` agree exactly."""
    ln_a = np.log(np.clip(a_col, a_nodes[0], a_nodes[-1]))
    i = np.clip(np.searchsorted(ln_nodes, ln_a) - 1, 0, ln_nodes.size - 2)
    span = ln_nodes[i + 1] - ln_nodes[i]
    t = (ln_a - ln_nodes[i]) / span
    w = w_arm[i] + t * (w_arm[i + 1] - w_arm[i])
    mu = mu_arm[i, :] + t[:, np.newaxis] * (mu_arm[i + 1, :] - mu_arm[i, :])
    sigma = sigma_arm[i, :] + t[:, np.newaxis] * (sigma_arm[i + 1, :] - sigma_arm[i, :])
    return w, mu, sigma


def _match_protostars_to_beam(config):
    """R4's sample: every HOPS/eHOPS (`sky.derived.protostars`) Class 0,
    I or flat protostar with a finite positive `AV_FOREGROUND_MAG`,
    matched to its own nside-256 sightline's adopted column
    (`sky.derived.column.build_sightline`'s survey-wide product, `A_K`
    the EXTINCTION column at that granule, SPEC section 3.2) -- kept only
    where that sightline is on the Herschel arm (`PROVENANCE` 0, the same
    code `_ARM_CODE['herschel']` uses). Returns a dict of aligned arrays
    (`region`, `av_mag`, `a_beam`, `sigma_beam`, `pix256`) and the three
    drop counts this brief's report prints, in the order checked: no
    finite positive `A_V`, no sightline row at that pixel at all, no
    Herschel arm there (a Planck-arm pixel, or a region/pixel the
    protostar view assigns no SESNA footprint to).
    """
    proto_path = f"{config.data_root}/sky/derived/protostars/protostars_survey.hdf5"
    with h5py.File(proto_path, "r") as f:
        region = f["REGION"][:]
        cls = f["CLASS"][:]
        ra_deg = np.asarray(f["RA_DEG"][:], dtype=np.float64)
        dec_deg = np.asarray(f["DEC_DEG"][:], dtype=np.float64)
        av_mag = np.asarray(f["AV_FOREGROUND_MAG"][:], dtype=np.float64)

    class_ok = np.isin(cls, np.array(_PROTOSTAR_CLASSES))
    av_ok = np.isfinite(av_mag) & (av_mag > 0.0)
    n_dropped_no_av = int(np.count_nonzero(class_ok & ~av_ok))
    keep = class_ok & av_ok

    idx = np.where(keep)[0]
    gal = SkyCoord(ra=ra_deg[idx] * u.deg, dec=dec_deg[idx] * u.deg, frame="icrs").galactic
    pix256 = hp.ang2pix(256, gal.l.deg, gal.b.deg, nest=True, lonlat=True)

    sl_path = config_module.product_path(config, "sky/derived", "adopted", "column", "sightline")
    with h5py.File(sl_path, "r") as f:
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        sl_a_k = np.asarray(f["A_K"][:], dtype=np.float64)
        sl_sigma = np.asarray(f["SIGMA_A_K"][:], dtype=np.float64)
        sl_prov = np.asarray(f["PROVENANCE"][:])

    order = np.argsort(sl_pix)
    sl_pix_sorted = sl_pix[order]
    loc = np.searchsorted(sl_pix_sorted, pix256)
    capped = np.minimum(loc, max(sl_pix_sorted.size - 1, 0))
    found = (sl_pix_sorted.size > 0) & (sl_pix_sorted[capped] == pix256)
    n_dropped_no_sightline = int(np.count_nonzero(~found))

    is_h = np.zeros(pix256.shape, dtype=bool)
    is_h[found] = sl_prov[order[capped[found]]] == _ARM_CODE["herschel"]
    region_ok = np.isin(region[idx], np.array([r.encode("utf-8") for r in _PROTOSTAR_REGIONS]))
    n_dropped_no_herschel_arm = int(np.count_nonzero(found & (~is_h | ~region_ok)))

    keep2 = found & is_h & region_ok
    rows = order[capped[keep2]]
    return dict(
        region=region[idx][keep2], av_mag=av_mag[idx][keep2],
        a_beam=sl_a_k[rows], sigma_beam=sl_sigma[rows], pix256=pix256[keep2],
        n_dropped_no_av=n_dropped_no_av,
        n_dropped_no_sightline=n_dropped_no_sightline,
        n_dropped_no_herschel_arm=n_dropped_no_herschel_arm,
    )


def _cloud_cells_for_pixels(config, yso_module, region, pix256):
    """`(log10x_mid, mass)`, each `(len(pix256), n_cell)`: the region's
    cloud-interval-restricted embedding density (`population.yso.
    embedding_and_ridge`/`restrict_and_renormalize`/`cloud_interval_pc`,
    full profile resolution -- the same restriction `population.
    young_stars.sightline_lookup` applies for the field-star deduction),
    read off at each of `pix256`'s own nside-256 sightline row. `mass`
    sums to 1 per row (cells outside the cloud interval carry zero);
    `log10x_mid` is each cell's own `log10 u` at the midpoint of its
    (floored) `u` span -- `x = u = A(d)/A(inf)`, the fraction of the
    sightline's total column in front of a member at that depth.
    """
    profile = yso_module._load_profile_arrays(config, region)
    embed = yso_module.embedding_and_ridge(profile)
    dist_pc = profile["dist_pc"]
    n_d = dist_pc.size
    n_sl = embed["u_edges"].shape[0]
    d_edges = np.empty((n_sl, n_d + 1), dtype=np.float64)
    d_edges[:, :n_d] = dist_pc[None, :]
    d_edges[:, n_d] = dist_pc[-1] + 2.0 * profile["tail_efold_pc"]
    u_edges = embed["u_edges"]
    p_u = embed["p_u"]
    u_lo, u_hi = u_edges[:, :-1], u_edges[:, 1:]
    d_lo, d_hi = d_edges[:, :-1], d_edges[:, 1:]

    d_front, d_back = yso_module.cloud_interval_pc(config, region)
    mass, inside_frac, removed_frac = yso_module.restrict_and_renormalize(
        p_u, u_lo, u_hi, d_lo, d_hi, d_front, d_back)
    mass_restricted = (mass * inside_frac) / np.maximum(1.0 - removed_frac, 1e-300)[:, None]
    log10x_mid = 0.5 * (np.log10(np.maximum(u_lo, _U_LOG_FLOOR))
                         + np.log10(np.maximum(u_hi, _U_LOG_FLOOR)))

    sl_pix = profile["hpx_pix_256"]
    order = np.argsort(sl_pix)
    sl_pix_sorted = sl_pix[order]
    loc = np.searchsorted(sl_pix_sorted, pix256)
    capped = np.minimum(loc, max(sl_pix_sorted.size - 1, 0))
    matched = (sl_pix_sorted.size > 0) & (sl_pix_sorted[capped] == pix256)
    if not np.all(matched):
        raise ValueError(
            "population.kernel: %d of %d protostar sightlines of region %r have no "
            "row in the embedding profile -- run the 'population.yso' RUNBOOK line"
            % (int(np.count_nonzero(~matched)), pix256.size, region))
    rows = order[capped]
    return log10x_mid[rows], mass_restricted[rows]


def _fit_cloud_sigma_herschel(config, a_nodes, w_h, mu_h, sigma_h, zp_herschel_k):
    """R4: `CLOUD_SIGMA_HERSCHEL_DEX`, one survey-pooled number replacing
    the sub-beam stage's own (core-biased) extrapolated structural width
    for the cloud classes (`Kernel.mixture`'s `exponent > 0`), calibrated
    on the HOPS (Orion A) + eHOPS (Aquila) Class 0/I/flat protostars
    (`_match_protostars_to_beam`). For a cloud member on a protostar's own
    sightline, `log10 r_p = log10 x + y`: `x` from that sightline's own
    cloud-interval `p(u)` (`_cloud_cells_for_pixels`), `y` from the
    Herschel structural mixture with both components' sigma scaled to a
    trial `sigma_cloud` (`_rescale_structural_to_sigma_cloud`), the
    source's own measurement term (`SIGMA_A_K`) and the survey zero point
    folded in, reweighted by `T**2` (`Kernel.mixture`'s own exponent,
    SPEC_BMSTP_DRAFT.md 5.5) -- the same arithmetic `mixture` performs,
    inlined here since no `Kernel` exists yet inside `build`. The sample
    log-likelihood -- protostars independent, `p(u)`'s cells and the
    kernel's two components summed in closed form (Gaussian in
    `log10 r`) -- is profiled over a fixed `sigma_cloud` grid; the
    maximiser is `CLOUD_SIGMA_HERSCHEL_DEX`, its 68% interval where the
    log-likelihood is within 0.5 of the maximum (one profiled parameter).
    """
    from sesnaimpute.population import selection as selection_module
    from sesnaimpute.population import yso as yso_module

    match = _match_protostars_to_beam(config)
    a_beam = match["a_beam"]
    n_proto = a_beam.size
    ln_nodes = np.log(a_nodes)

    a_p = match["av_mag"] * selection_module.ak_per_av(
        config, selection_module.law_dense_weight(a_beam))
    log10_r_p = np.log10(a_p / a_beam)

    w0, mu0, sigma0 = _interp_structural(a_nodes, ln_nodes, w_h, mu_h, sigma_h, a_beam)
    # the source's measurement term and the survey zero point, in quadrature,
    # converted to dex at a_beam -- exactly `Kernel.mixture`'s own combination
    # (`sigma_col_dex**2 + zp_dex**2`), every matched protostar the Herschel
    # arm by construction (`_match_protostars_to_beam`).
    extra_var = ((match["sigma_beam"] / (a_beam * _LN10)) ** 2
                 + (zp_herschel_k / (a_beam * _LN10)) ** 2)

    log10x_parts, mass_parts = [], []
    for region in _PROTOSTAR_REGIONS:
        sel = match["region"] == region.encode("utf-8")
        if not np.any(sel):
            continue
        log10x_r, mass_r = _cloud_cells_for_pixels(config, yso_module, region, match["pix256"][sel])
        log10x_parts.append((sel, log10x_r, mass_r))
    n_u_max = max(p[1].shape[1] for p in log10x_parts)
    log10x = np.zeros((n_proto, n_u_max))
    mass = np.zeros((n_proto, n_u_max))
    for sel, log10x_r, mass_r in log10x_parts:
        n_u = log10x_r.shape[1]
        rows = np.where(sel)[0]
        log10x[rows[:, None], np.arange(n_u)[None, :]] = log10x_r
        mass[rows[:, None], np.arange(n_u)[None, :]] = mass_r

    n_grid = int(round((_CLOUD_SIGMA_GRID_HI - _CLOUD_SIGMA_GRID_LO) / _CLOUD_SIGMA_GRID_STEP)) + 1
    grid = _CLOUD_SIGMA_GRID_LO + _CLOUD_SIGMA_GRID_STEP * np.arange(n_grid, dtype=np.float64)
    loglike = np.empty(n_grid, dtype=np.float64)
    exponent = 2.0
    c = exponent * _LN10

    def _tilted(mu_resc, sigma_resc):
        sigma_tot = np.sqrt(sigma_resc ** 2 + extra_var[:, None])
        mu_tilt = mu_resc + c * sigma_tot * sigma_tot
        log_wt = c * mu_resc + (c * sigma_tot) ** 2 / 2.0
        log_wt -= log_wt.max(axis=1, keepdims=True)
        wt = np.stack([w0, 1.0 - w0], axis=1) * np.exp(log_wt)
        w_tilt = wt[:, 0] / wt.sum(axis=1)
        return w_tilt, mu_tilt, sigma_tot

    for g, sigma_cloud in enumerate(grid):
        mu_resc, sigma_resc = _rescale_structural_to_sigma_cloud(w0, mu0, sigma0, sigma_cloud)
        w_tilt, mu_tilt, sigma_tot = _tilted(mu_resc, sigma_resc)
        off0 = (log10_r_p[:, None] - log10x - mu_tilt[:, 0:1]) / sigma_tot[:, 0:1]
        off1 = (log10_r_p[:, None] - log10x - mu_tilt[:, 1:2]) / sigma_tot[:, 1:2]
        dens_cell = (w_tilt[:, None] * np.exp(-0.5 * off0 * off0) / (sigma_tot[:, 0:1] * _SQRT2PI)
                     + (1.0 - w_tilt[:, None]) * np.exp(-0.5 * off1 * off1)
                     / (sigma_tot[:, 1:2] * _SQRT2PI))
        density_p = np.sum(mass * dens_cell, axis=1)
        loglike[g] = float(np.sum(np.log(np.maximum(density_p, 1e-300))))

    i_max = int(np.argmax(loglike))
    sigma_cloud_best = float(grid[i_max])
    within_1sigma = grid[loglike >= loglike[i_max] - _CLOUD_SIGMA_DLOGLIKE]
    p16, p84 = float(within_1sigma.min()), float(within_1sigma.max())

    # Item 3(c)/3(d)'s own report numbers, all at the fitted width.
    mu_resc, sigma_resc = _rescale_structural_to_sigma_cloud(w0, mu0, sigma0, sigma_cloud_best)
    w_tilt, mu_tilt, sigma_tot = _tilted(mu_resc, sigma_resc)
    mean0 = log10_r_p - mu_tilt[:, 0]
    mean1 = log10_r_p - mu_tilt[:, 1]
    p_reach = (w_tilt * _norm_cdf(-mean0 / sigma_tot[:, 0])
               + (1.0 - w_tilt) * _norm_cdf(-mean1 / sigma_tot[:, 1]))
    frac_below_reach = float(np.mean(p_reach < 0.01)) if n_proto else float("nan")

    def _pooled_cdf(z):
        off0 = (z - log10x - mu_tilt[:, 0:1]) / sigma_tot[:, 0:1]
        off1 = (z - log10x - mu_tilt[:, 1:2]) / sigma_tot[:, 1:2]
        cdf_cell = w_tilt[:, None] * _norm_cdf(off0) + (1.0 - w_tilt[:, None]) * _norm_cdf(off1)
        return float(np.mean(np.sum(mass * cdf_cell, axis=1)))

    def _solve(target, lo=-6.0, hi=3.0, iters=60):
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if _pooled_cdf(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    pred_median = _solve(0.5) if n_proto else float("nan")
    pred_p84 = _solve(0.84) if n_proto else float("nan")
    emp_median = float(np.median(log10_r_p)) if n_proto else float("nan")
    emp_p84 = float(np.percentile(log10_r_p, 84.0)) if n_proto else float("nan")

    return dict(
        sigma_cloud=sigma_cloud_best, p16=p16, p84=p84, grid=grid, loglike=loglike,
        n_protostars=n_proto, a_beam_dataset="sky/derived/adopted/column/sightline: A_K",
        n_dropped_no_herschel_arm=match["n_dropped_no_herschel_arm"],
        n_dropped_no_av=match["n_dropped_no_av"],
        n_dropped_no_sightline=match["n_dropped_no_sightline"],
        pred_median=pred_median, pred_p84=pred_p84,
        emp_median=emp_median, emp_p84=emp_p84, frac_below_reach=frac_below_reach,
    )


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

    # R5: the Planck arm recentred exactly as the Herschel arm is above --
    # the same uniform per-node dex shift on both components' means, so
    # `E[T / A_beam] = 1` in linear units at every node there too; the
    # Planck arm's sigma's and the two means' relative offset untouched.
    e_raw_p = (W[i_p] * 10.0 ** (MU[i_p, :, 0] + SIGMA[i_p, :, 0] ** 2 * _LN10 / 2.0)
               + (1.0 - W[i_p]) * 10.0 ** (MU[i_p, :, 1] + SIGMA[i_p, :, 1] ** 2 * _LN10 / 2.0))
    planck_offset = -np.log10(e_raw_p)
    MU[i_p, :, 0] += planck_offset
    MU[i_p, :, 1] += planck_offset
    e_check_p = (W[i_p] * 10.0 ** (MU[i_p, :, 0] + SIGMA[i_p, :, 0] ** 2 * _LN10 / 2.0)
                 + (1.0 - W[i_p]) * 10.0 ** (MU[i_p, :, 1] + SIGMA[i_p, :, 1] ** 2 * _LN10 / 2.0))

    # R4: the cloud-class (`exponent > 0`) Herschel structural width,
    # calibrated on the HOPS/eHOPS protostars (Orion A, Aquila) rather
    # than extrapolated from the sub-beam stage's own core-biased lower
    # bound.
    fit = _fit_cloud_sigma_herschel(config, a_nodes, W[i_h], MU[i_h], SIGMA[i_h], zp)

    out_path = config_module.product_path(config, "population", "sesna", "kernel", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("A_NODES", data=a_nodes.astype(np.float64))
        f.create_dataset("MIX_W", data=W.astype(np.float64))
        f.create_dataset("MIX_MU", data=MU.astype(np.float64))
        f.create_dataset("MIX_SIGMA", data=SIGMA.astype(np.float64))
        f.create_dataset("ZP_HERSCHEL_K", data=np.float64(zp))
        f.create_dataset("CLOUD_SIGMA_HERSCHEL_DEX", data=np.float64(fit["sigma_cloud"]))
        f.create_dataset("CLOUD_SIGMA_HERSCHEL_P16", data=np.float64(fit["p16"]))
        f.create_dataset("CLOUD_SIGMA_HERSCHEL_P84", data=np.float64(fit["p84"]))
        f.create_dataset("CLOUD_SIGMA_GRID_DEX", data=fit["grid"].astype(np.float64))
        f.create_dataset("CLOUD_SIGMA_LOGLIKE", data=fit["loglike"].astype(np.float64))
        f.create_dataset("N_PROTOSTARS_FIT", data=np.int64(fit["n_protostars"]))

    st.done(out_path, n_arms=len(_ARM_ORDER), n_node=n_node, zp_herschel_k=float(zp),
            herschel_stretch=herschel_factor, cloud_sigma_herschel_dex=fit["sigma_cloud"],
            n_protostars_fit=fit["n_protostars"])
    print("kernel: %d arms x %d nodes (mixture), zp_herschel_k=%.4f, "
          "herschel_stretch=%.4f -> %s"
          % (len(_ARM_ORDER), n_node, zp, herschel_factor, out_path), flush=True)
    print("kernel: Planck arm recentred, max |E[T/A_beam] - 1| = %.3e"
          % float(np.max(np.abs(e_check_p - 1.0))), flush=True)
    print("kernel: R4 cloud_sigma_herschel_dex = %.3f dex (68%% interval %.3f-%.3f dex), "
          "N_PROTOSTARS_FIT = %d, A_beam dataset = %s"
          % (fit["sigma_cloud"], fit["p16"], fit["p84"], fit["n_protostars"], fit["a_beam_dataset"]),
          flush=True)
    print("kernel: R4 protostars dropped -- no Herschel arm: %d, no finite A_V: %d, "
          "no sightline: %d" % (fit["n_dropped_no_herschel_arm"], fit["n_dropped_no_av"],
                                 fit["n_dropped_no_sightline"]), flush=True)
    print("kernel: R4 check -- predicted/empirical median log10 r = %.4f/%.4f, "
          "predicted/empirical p84 log10 r = %.4f/%.4f, frac P(T>=a_p)<0.01 = %.4f"
          % (fit["pred_median"], fit["emp_median"], fit["pred_p84"], fit["emp_p84"],
             fit["frac_below_reach"]), flush=True)


if __name__ == "__main__":
    from sesnaimpute.build import run
    run(build)
