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
member sits off the beam mean by that same power, within the beam. For
the cloud classes (YSO, H2S) that exponent is `CLOUD_GAMMA_HERSCHEL`
(owner, 2026-09-12 evening ruling 1): the star-gas law's own exponent
(Pokhrel+2020 1.8-2.3, Lada+2013 2.04 +/- 0.01) is measured at the maps'
resolution, and applying it WITHIN a beam is an extrapolation with no
published support, so `gamma` is instead measured jointly with the
cloud-class structural width `CLOUD_SIGMA_HERSCHEL_DEX` by a 2-D
maximum-likelihood fit on the same HOPS/eHOPS protostars
(`_fit_cloud_gamma_sigma_herschel`, `build`) -- `gamma = 0` (no beam
tilt at all) if the fit's own 68% interval for `gamma` includes 0
(parsimony), else the fitted `gamma`. A caller passes this per-source
column's own class exponent -- `0.0` for every other class, the
product's `cloud_gamma_herschel` for YSO/H2S (`fittp.prior_reader`'s
`CLOUD` flag, `atlas.protostars._verdict`) -- never a literal `2.0`.

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

#: the calibration grid for `CLOUD_SIGMA_HERSCHEL_DEX`: 0.03
#: to 0.80 dex in 0.01-dex steps, the profile likelihood's own domain.
_CLOUD_SIGMA_GRID_LO = 0.03
_CLOUD_SIGMA_GRID_HI = 0.80
_CLOUD_SIGMA_GRID_STEP = 0.01

#: The half-drop in log-likelihood (a chi-square difference of 1 for one
#: profiled parameter) that bounds `CLOUD_SIGMA_HERSCHEL_DEX`'s own 68%
#: interval when it is refit ALONE, `gamma` fixed at 0 (the parsimony
#: branch of `_fit_cloud_gamma_sigma_herschel`).
_CLOUD_SIGMA_DLOGLIKE = 0.5

#: the calibration grid for `CLOUD_GAMMA_HERSCHEL`, the within-beam
#: tilt exponent (owner, 2026-09-12 evening ruling 1): 0.0 to 3.0 in
#: 0.1 steps, fit jointly with `sigma_cloud` on the same 2-D grid.
_CLOUD_GAMMA_GRID_LO = 0.0
_CLOUD_GAMMA_GRID_HI = 3.0
_CLOUD_GAMMA_GRID_STEP = 0.1

#: The half-drop in log-likelihood bounding the JOINT 2-D 68% region for
#: `(gamma, sigma_cloud)` -- a chi-square difference of 2.30 for two
#: fitted parameters, halved. `CLOUD_GAMMA_HERSCHEL`'s own 68% interval
#: is this joint region's projection onto the gamma axis (the maximum
#: over `sigma_cloud` at each `gamma`, "the marginal over the other"),
#: not a separate one-parameter profile.
_CLOUD_JOINT_DLOGLIKE = 1.15

#: The two regions with a Herschel arm whose Class 0/I/flat protostars
#: calibrate the cloud-class structural width: HOPS's own Orion A,
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


def _cloud_single_component(sigma_cloud, n):
    """The cloud-class Herschel structural term is ONE
    lognormal component of width `sigma_cloud` dex, the same at every
    column -- not the sub-beam stage's two-component mixture rescaled.
    Scaling both of that mixture's components by one factor is
    ill-conditioned wherever they are nearly degenerate (their means
    coincide, so the between-component term vanishes and the whole
    target width has to come from the scale factor alone, blowing one
    component's sigma up arbitrarily); a single fixed-width component has
    no such failure mode. `w = 1`; both `mu`/`sigma` "components" are
    identical (the second is unused) so the ordinary two-component
    machinery downstream (`_mixture_mean_var`, `mixture`'s own tilt)
    treats it exactly as the one component it is. `mu = -sigma_cloud**2 *
    ln10 / 2` recentres it to mean one (`E[T/A_beam] = 1`) exactly, a
    lognormal's own mean-one condition. Returns `(w, mu, sigma)` broadcast
    to `n` sources: `w` `(n,)`, `mu`/`sigma` `(n, 2)`."""
    mu0 = -0.5 * sigma_cloud * sigma_cloud * _LN10
    w = np.ones(n, dtype=float)
    mu = np.full((n, 2), mu0, dtype=float)
    sigma = np.full((n, 2), float(sigma_cloud), dtype=float)
    return w, mu, sigma


class Kernel(object):
    """The column kernel: a two-component log-normal mixture in `log10 T`
    per arm, tabulated on the column grid (SPEC_PRIORS.md section 1.2).
    Built by `build(config)`, loaded by `read(config)`.
    """

    def __init__(self, a_nodes, w, mu, sigma, zp_herschel_k, cloud_sigma_herschel_dex,
                 cloud_gamma_herschel):
        self._a_nodes = np.asarray(a_nodes, dtype=float)
        self._ln_nodes = np.log(self._a_nodes)
        self._w = np.asarray(w, dtype=float)          # (n_arm, n_node)
        self._mu = np.asarray(mu, dtype=float)         # (n_arm, n_node, 2)
        self._sigma = np.asarray(sigma, dtype=float)    # (n_arm, n_node, 2)
        #: the survey-wide RMS of the per-field zero points -- the
        #: fallback for a Herschel source whose call site does not yet
        #: pass its own `ZP_SIGMA_K` (owner, 2026-09-06).
        self.zp_herschel_k = float(zp_herschel_k)
        #: the Herschel arm's cloud-class (`exponent > 0`) structural
        #: width, dex, calibrated in `build` (`_fit_cloud_gamma_sigma_
        #: herschel`) jointly with `cloud_gamma_herschel` on the
        #: HOPS/eHOPS protostars.
        self.cloud_sigma_herschel_dex = float(cloud_sigma_herschel_dex)
        #: the star-gas law's own within-beam tilt exponent for the cloud
        #: classes (YSO, H2S), fit jointly with `cloud_sigma_herschel_dex`
        #: on the same protostars (owner, 2026-09-12 evening ruling 1;
        #: `_fit_cloud_gamma_sigma_herschel`) -- 0 if that fit's own 68%
        #: interval for gamma includes 0 (parsimony), else the fitted
        #: value. A caller passing `exponent` to `mixture` for a cloud
        #: class reads this, never a literal `2.0`.
        self.cloud_gamma_herschel = float(cloud_gamma_herschel)

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
                       float(f["CLOUD_SIGMA_HERSCHEL_DEX"][()]),
                       float(f["CLOUD_GAMMA_HERSCHEL"][()]))

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

        For `exponent != 0.0`, the Herschel arm's structural term is
        replaced entirely (`_cloud_single_component`) by ONE lognormal
        component of width `self.cloud_sigma_herschel_dex` dex, the same
        at every column, recentred to mean one -- before the measurement/
        zero-point terms below or this tilt -- rather than the sub-beam
        stage's own two-component mixture (a lower bound toward cores,
        not the cloud classes' own scale) rescaled: scaling both of that
        mixture's components by one factor is ill-conditioned wherever
        they are nearly degenerate, so a single fixed-width component
        replaces it outright instead. `cloud_sigma_herschel_dex` and the
        caller's own `exponent` (`cloud_gamma_herschel` for a cloud
        class) are fit JOINTLY in `build`
        (`_fit_cloud_gamma_sigma_herschel`) with this same one-component
        construction on the HOPS (Orion A) and eHOPS (Aquila) Class
        0/I/flat protostars: each protostar's own foreground `A_V`,
        converted to `A_K`, against its matched SESNA source's own
        adopted extinction column (the reader's own beam column, not the
        nside-256 sightline mean), profiled on a 2-D grid against the
        region's own cloud-interval placement of a member star. The
        published star-gas law exponent (Pokhrel+2020, Lada+2013) is
        measured at the maps' resolution, not within a beam, so `gamma`
        is measured here rather than assumed; SESNA's own YSO fits
        validate or replace both numbers per region once they exist.
        `exponent = 0.0` and the Planck arm are untouched by this."""
        a_col = np.asarray(a_col, dtype=float)
        sigma_col = np.asarray(sigma_col, dtype=float)
        arm_idx = self._arm_index(map_class)
        w, mu, sigma0 = self._structural(a_col, arm_idx)
        if exponent != 0.0:
            is_h = arm_idx == _ARM_CODE["herschel"]
            n_h = int(np.count_nonzero(is_h))
            if n_h:
                w = w.copy()
                mu = mu.copy()
                sigma0 = sigma0.copy()
                w[is_h], mu[is_h], sigma0[is_h] = _cloud_single_component(
                    self.cloud_sigma_herschel_dex, n_h)
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


#: The nearest-neighbour match radius to the curated catalogue, this
#: brief -- exactly `atlas.protostars.MATCH_RADIUS_ARCSEC`: inside it a
#: protostar takes that source as its SESNA counterpart.
_MATCH_RADIUS_ARCSEC = 2.0


def _pix512_galactic(ra_deg, dec_deg):
    """Each position's nside-512 galactic NESTED pixel -- the granule
    map's own pixelisation, the same one `atlas.protostars._pix512_
    galactic` uses for its own catalogue-match standin (reproduced here,
    not imported: `population` may not import `atlas`)."""
    gal = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    return hp.ang2pix(512, gal.l.deg, gal.b.deg, nest=True, lonlat=True)


def _match_one_region_to_catalogue(config, region, ra_deg, dec_deg):
    """Each protostar's own SESNA source in `region` -- the SAME rule
    `atlas.protostars._match_to_catalogue` applies (nearest within
    `_MATCH_RADIUS_ARCSEC`; failing that, the nearest catalogued source
    sharing the protostar's own nside-512 pixel; failing that, excluded),
    reproduced here rather than imported (`population` may not import
    `atlas`) -- but reading `sky.derived.adopted.extinction.source`'s own
    `A_COL_K`/`A_COL_SIG_K`/`A_COL_PROVENANCE` at the matched source's
    row (`granules.access.per_source`'s own `source`-granule join: one
    row per catalogued source of `region`, in catalogue-row order, so no
    separate name-alignment check is needed the way `atlas.protostars`
    needs one against a later `bmstp` product), not `atlas.protostars`'s
    own `bmstp.density.table.source` -- this fit runs inside
    `population`, before any `bmstp` product exists.

    Returns `(a_col, a_col_sig, prov, matched)`, each `(len(ra_deg),)`;
    `matched` False where no catalogued source of `region` is within
    reach (`a_col`/`a_col_sig`/`prov` undefined there).
    """
    from sesnaimpute.granules import access

    cat_path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(cat_path, "r") as f:
        ra_cat = np.asarray(f["RA_DEG"][:], dtype=np.float64)
        dec_cat = np.asarray(f["DEC_DEG"][:], dtype=np.float64)

    ext_path = config_module.product_path(config, "sky/derived", "adopted", "extinction", "source", region=region)
    cols = access.per_source(config, region, ext_path, ["A_COL_K", "A_COL_SIG_K", "A_COL_PROVENANCE"])
    a_col_cat = np.asarray(cols["A_COL_K"], dtype=np.float64)
    a_col_sig_cat = np.asarray(cols["A_COL_SIG_K"], dtype=np.float64)
    prov_cat = np.asarray(cols["A_COL_PROVENANCE"])

    coord_proto = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    coord_cat = SkyCoord(ra=ra_cat * u.deg, dec=dec_cat * u.deg, frame="icrs")
    idx_nn, sep2d, _ = coord_proto.match_to_catalog_sky(coord_cat)
    direct = sep2d.arcsec <= _MATCH_RADIUS_ARCSEC

    cat_row = np.full(ra_deg.size, -1, dtype=np.int64)
    cat_row[direct] = idx_nn[direct]

    # The sightline stand-in (`atlas.protostars._match_to_catalogue`,
    # unchanged rule): the loop below is over the DISTINCT nside-512
    # pixels among the protostars a direct match missed (typically a
    # handful), each iteration itself vectorised over its own candidates.
    need_standin = ~direct
    if np.any(need_standin):
        pix_proto = _pix512_galactic(ra_deg, dec_deg)
        pix_cat = _pix512_galactic(ra_cat, dec_cat)
        order = np.argsort(pix_cat)
        pix_cat_sorted = pix_cat[order]
        for p in np.unique(pix_proto[need_standin]):
            lo = np.searchsorted(pix_cat_sorted, p, side="left")
            hi = np.searchsorted(pix_cat_sorted, p, side="right")
            if hi == lo:
                continue  # no catalogued source shares this pixel: stays excluded
            cand = order[lo:hi]
            rows_in_pix = np.where(need_standin & (pix_proto == p))[0]
            sep = coord_cat[cand][:, None].separation(coord_proto[rows_in_pix][None, :]).arcsec
            nearest = cand[np.argmin(sep, axis=0)]
            cat_row[rows_in_pix] = nearest

    matched = cat_row >= 0
    a_col = np.full(ra_deg.size, np.nan, dtype=np.float64)
    a_col_sig = np.full(ra_deg.size, np.nan, dtype=np.float64)
    prov = np.full(ra_deg.size, -1, dtype=np.int64)
    a_col[matched] = a_col_cat[cat_row[matched]]
    a_col_sig[matched] = a_col_sig_cat[cat_row[matched]]
    prov[matched] = prov_cat[cat_row[matched]].astype(np.int64)
    return a_col, a_col_sig, prov, matched


def _match_protostars_to_beam(config):
    """The calibration sample: every HOPS/eHOPS (`sky.derived.protostars`) Class 0,
    I or flat protostar with a finite positive `AV_FOREGROUND_MAG`,
    matched (`_match_one_region_to_catalogue`) to its own SESNA source in
    its region. `A_beam` is that SOURCE's own adopted EXTINCTION column,
    `A_COL_K` from `sky.derived.adopted.extinction.source` (SPEC section
    3.2) -- the reader's own beam column, the 36 arcsec-map value at the
    SOURCE's position, not the nside-256 sightline mean the two arms are
    pooled on (they differ by ~0.3 dex at a protostar): the quantity
    `a_p` is compared to (`xi_hat = a_p / A_beam`) must divide by the same
    column a consumer elsewhere divides by. Kept only where the matched
    source is on the Herschel arm (`A_COL_PROVENANCE` 0, the same code
    `_ARM_CODE['herschel']` uses). Returns a dict of aligned arrays
    (`region`, `av_mag`, `a_beam`, `sigma_beam`, `pix256`) and the three
    drop counts the stage prints, in the order checked: no
    finite positive `A_V`, no SESNA source match in its own region (or a
    region/pixel the protostar view assigns no SESNA footprint to), no
    Herschel arm at the matched source (a Planck-arm source).
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
    region_k = region[idx]
    ra_k, dec_k = ra_deg[idx], dec_deg[idx]
    region_ok = np.isin(region_k, np.array([r.encode("utf-8") for r in _PROTOSTAR_REGIONS]))

    a_col = np.full(idx.size, np.nan, dtype=np.float64)
    a_col_sig = np.full(idx.size, np.nan, dtype=np.float64)
    prov = np.full(idx.size, -1, dtype=np.int64)
    matched = np.zeros(idx.size, dtype=bool)
    for r in _PROTOSTAR_REGIONS:
        sel = region_ok & (region_k == r.encode("utf-8"))
        if not np.any(sel):
            continue
        a_col_r, a_col_sig_r, prov_r, matched_r = _match_one_region_to_catalogue(
            config, r, ra_k[sel], dec_k[sel])
        rows = np.where(sel)[0]
        a_col[rows], a_col_sig[rows], prov[rows], matched[rows] = a_col_r, a_col_sig_r, prov_r, matched_r

    n_dropped_no_catalogue_match = int(np.count_nonzero(region_ok & ~matched))
    is_h = matched & (prov == _ARM_CODE["herschel"])
    n_dropped_no_herschel_arm = int(np.count_nonzero(~region_ok | (region_ok & matched & ~is_h)))

    keep2 = region_ok & matched & is_h
    gal = SkyCoord(ra=ra_k[keep2] * u.deg, dec=dec_k[keep2] * u.deg, frame="icrs").galactic
    pix256 = hp.ang2pix(256, gal.l.deg, gal.b.deg, nest=True, lonlat=True)
    return dict(
        region=region_k[keep2], av_mag=av_mag[idx][keep2],
        a_beam=a_col[keep2], sigma_beam=a_col_sig[keep2], pix256=pix256,
        n_dropped_no_av=n_dropped_no_av,
        n_dropped_no_sightline=n_dropped_no_catalogue_match,
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
    n_sl = embed["xi_edges"].shape[0]
    d_edges = np.empty((n_sl, n_d + 1), dtype=np.float64)
    d_edges[:, :n_d] = dist_pc[None, :]
    d_edges[:, n_d] = dist_pc[-1] + 2.0 * profile["tail_efold_pc"]
    xi_edges = embed["xi_edges"]
    p_u = embed["p_u"]
    u_lo, u_hi = xi_edges[:, :-1], xi_edges[:, 1:]
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


def _fit_cloud_gamma_sigma_herschel(config, zp_herschel_k):
    """`(CLOUD_GAMMA_HERSCHEL, CLOUD_SIGMA_HERSCHEL_DEX)`, one
    survey-pooled pair for the cloud classes' within-beam tilt and
    Herschel structural term (`Kernel.mixture`'s `exponent`, `sigma0`),
    calibrated JOINTLY on the HOPS (Orion A) + eHOPS (Aquila) Class
    0/I/flat protostars (`_match_protostars_to_beam`) -- owner, 2026-09-
    12 evening ruling 1: the published star-gas law exponent (Pokhrel+2020
    1.8-2.3, Lada+2013 2.04 +/- 0.01) is measured AT THE MAPS' RESOLUTION;
    applying it WITHIN a beam is an extrapolation with no published
    support, so `gamma` is measured here, not assumed. This same
    one-component construction the fit and `Kernel.mixture` share
    (`_cloud_single_component`: one lognormal of width `sigma_cloud` dex,
    the same at every column, recentred to mean one) -- not the sub-beam
    stage's own two-component mixture rescaled, which is ill-conditioned
    wherever its two components are nearly degenerate. `A_beam`
    (`_match_protostars_to_beam`) is the matched SESNA SOURCE's own
    adopted extinction column, `A_COL_K` from `sky.derived.adopted.
    extinction.source` -- the reader's own beam column, the 36 arcsec-map
    value at the source's position -- not the nside-256 sightline mean
    (they differ by ~0.3 dex at a protostar): the ratio `xi_hat = a_p /
    A_beam` must divide by the same column a consumer elsewhere divides
    by. For a cloud member on a protostar's own sightline, `log10 ξi_hat =
    log10 ξ + y`: `x` from that sightline's own cloud-interval `p(u)`
    (`_cloud_cells_for_pixels`), `y` from the one-component structural
    term at a trial `sigma_cloud`, the source's own measurement term
    (`A_COL_SIG_K`) and the survey zero point folded in, reweighted by
    `T**gamma` (`Kernel.mixture`'s own tilt, SPEC_BMSTP_DRAFT.md 5.5) --
    the same arithmetic `mixture` performs, inlined here since no
    `Kernel` exists yet inside `build`. The sample log-likelihood --
    protostars independent, `p(u)`'s cells summed in closed form
    (Gaussian in `log10 r`) -- is profiled on a 2-D grid, `sigma_cloud`
    x `gamma` (module constants); the joint maximiser is the fitted pair.
    PARSIMONY RULE (owner): `gamma`'s own 68% interval -- the joint
    region's projection onto the gamma axis, `Delta loglike <=
    _CLOUD_JOINT_DLOGLIKE` maximised over `sigma_cloud` at each `gamma`
    -- is checked first; if it includes 0, `gamma` is fixed at 0 and
    `sigma_cloud` is REFIT alone (one profiled parameter, `Delta loglike
    <= _CLOUD_SIGMA_DLOGLIKE`) rather than read off the joint grid's own
    gamma=0 row, so the adopted width is the best one-parameter fit at
    the adopted (zero) tilt, not a slice of the two-parameter surface.
    Otherwise both the joint maximum-likelihood `gamma` and `sigma_cloud`
    are adopted.
    """
    from sesnaimpute.population import selection as selection_module
    from sesnaimpute.population import yso as yso_module

    match = _match_protostars_to_beam(config)
    a_beam = match["a_beam"]
    n_proto = a_beam.size

    a_p = match["av_mag"] * selection_module.ak_per_av(
        config, selection_module.law_dense_weight(a_beam))
    log10_xi_hat = np.log10(a_p / a_beam)

    # the source's measurement term and the survey zero point, in quadrature,
    # converted to dex at a_beam -- exactly `Kernel.mixture`'s own combination
    # (`sigma_col_dex**2 + zp_dex**2`), every matched protostar the Herschel
    # arm by construction (`_match_protostars_to_beam`).
    extra_var = ((match["sigma_beam"] / (a_beam * _LN10)) ** 2
                 + (zp_herschel_k / (a_beam * _LN10)) ** 2)

    log10x_parts = []
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

    n_sigma = int(round((_CLOUD_SIGMA_GRID_HI - _CLOUD_SIGMA_GRID_LO) / _CLOUD_SIGMA_GRID_STEP)) + 1
    sigma_grid = _CLOUD_SIGMA_GRID_LO + _CLOUD_SIGMA_GRID_STEP * np.arange(n_sigma, dtype=np.float64)
    n_gamma = int(round((_CLOUD_GAMMA_GRID_HI - _CLOUD_GAMMA_GRID_LO) / _CLOUD_GAMMA_GRID_STEP)) + 1
    gamma_grid = _CLOUD_GAMMA_GRID_LO + _CLOUD_GAMMA_GRID_STEP * np.arange(n_gamma, dtype=np.float64)

    # The one-component construction's second "component" carries weight
    # exactly 0 (`_cloud_single_component`'s own `w = 1`), so the tilt's
    # renormalised weight is exactly 1 too, at every (sigma, gamma): the
    # cloud-class kernel really is a single lognormal, `mu' = mu0 + gamma
    # ln10 sigma_tot**2`, `sigma' = sigma_tot` unchanged -- this fit reads
    # that single Gaussian directly rather than carrying the (always-zero)
    # second component through the grid. `sigma_tot` and `mu0` do not
    # depend on `gamma`, so they are formed once for the whole sigma grid;
    # only the python loop over `gamma` (31 points) remains, each an array
    # pass over (n_sigma, n_proto, n_u).
    sigma_tot_grid = np.sqrt(sigma_grid[:, None] ** 2 + extra_var[None, :])  # (n_sigma, n_proto)
    mu0_grid = -0.5 * sigma_grid * sigma_grid * _LN10  # (n_sigma,)

    loglike2d = np.empty((n_gamma, n_sigma), dtype=np.float64)
    for gi, gamma in enumerate(gamma_grid):
        c = gamma * _LN10
        mu_tilt = mu0_grid[:, None] + c * sigma_tot_grid ** 2  # (n_sigma, n_proto)
        off = ((log10_xi_hat[None, :, None] - log10x[None, :, :] - mu_tilt[:, :, None])
               / sigma_tot_grid[:, :, None])
        dens_cell = np.exp(-0.5 * off * off) / (sigma_tot_grid[:, :, None] * _SQRT2PI)
        density_p = np.sum(mass[None, :, :] * dens_cell, axis=2)  # (n_sigma, n_proto)
        loglike2d[gi, :] = np.sum(np.log(np.maximum(density_p, 1e-300)), axis=1)

    flat_max = int(np.argmax(loglike2d))
    gi_max, si_max = np.unravel_index(flat_max, loglike2d.shape)
    gamma_mle, sigma_mle = float(gamma_grid[gi_max]), float(sigma_grid[si_max])
    loglike_max = float(loglike2d[gi_max, si_max])

    gamma_profile = loglike2d.max(axis=1)  # (n_gamma,): best sigma at each gamma
    within_gamma = gamma_grid[gamma_profile >= loglike_max - _CLOUD_JOINT_DLOGLIKE]
    gamma_p16, gamma_p84 = float(within_gamma.min()), float(within_gamma.max())
    sigma_profile = loglike2d.max(axis=0)  # (n_sigma,): best gamma at each sigma
    within_sigma = sigma_grid[sigma_profile >= loglike_max - _CLOUD_JOINT_DLOGLIKE]
    sigma_p16_joint, sigma_p84_joint = float(within_sigma.min()), float(within_sigma.max())

    loglike_at = {}
    for g_report in (0.0, 1.0, 2.0):
        gi_report = int(round((g_report - _CLOUD_GAMMA_GRID_LO) / _CLOUD_GAMMA_GRID_STEP))
        loglike_at[g_report] = float(gamma_profile[gi_report])

    parsimony_zero = gamma_p16 <= 0.0 <= gamma_p84
    if parsimony_zero:
        gamma_adopted = 0.0
        loglike_zero = loglike2d[0, :]  # gamma_grid[0] == 0.0 exactly
        si_zero = int(np.argmax(loglike_zero))
        sigma_adopted = float(sigma_grid[si_zero])
        loglike_zero_max = float(loglike_zero[si_zero])
        within_zero = sigma_grid[loglike_zero >= loglike_zero_max - _CLOUD_SIGMA_DLOGLIKE]
        sigma_p16, sigma_p84 = float(within_zero.min()), float(within_zero.max())
        loglike_adopted = loglike_zero_max
    else:
        gamma_adopted, sigma_adopted = gamma_mle, sigma_mle
        sigma_p16, sigma_p84 = sigma_p16_joint, sigma_p84_joint
        loglike_adopted = loglike_max

    # The report's own numbers, all at the adopted (gamma, sigma) -- the
    # scalar two-component machinery (`_cloud_single_component`) so this
    # matches `Kernel.mixture`'s own arithmetic exactly.
    c_adopted = gamma_adopted * _LN10
    w0, mu0, sigma0 = _cloud_single_component(sigma_adopted, n_proto)
    sigma_tot = np.sqrt(sigma0 ** 2 + extra_var[:, None])
    mu_tilt = mu0 + c_adopted * sigma_tot * sigma_tot
    log_wt = c_adopted * mu0 + (c_adopted * sigma_tot) ** 2 / 2.0
    log_wt -= log_wt.max(axis=1, keepdims=True)
    wt = np.stack([w0, 1.0 - w0], axis=1) * np.exp(log_wt)
    w_tilt = wt[:, 0] / wt.sum(axis=1)

    mean0 = log10_xi_hat - mu_tilt[:, 0]
    mean1 = log10_xi_hat - mu_tilt[:, 1]
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
    emp_median = float(np.median(log10_xi_hat)) if n_proto else float("nan")
    emp_p84 = float(np.percentile(log10_xi_hat, 84.0)) if n_proto else float("nan")

    return dict(
        gamma=gamma_adopted, gamma_mle=gamma_mle, gamma_p16=gamma_p16, gamma_p84=gamma_p84,
        parsimony_zero=parsimony_zero,
        sigma_cloud=sigma_adopted, sigma_mle=sigma_mle, p16=sigma_p16, p84=sigma_p84,
        gamma_grid=gamma_grid, sigma_grid=sigma_grid, loglike2d=loglike2d,
        loglike_max=loglike_max, loglike_adopted=loglike_adopted, loglike_at=loglike_at,
        n_protostars=n_proto, a_beam_dataset="sky/derived/adopted/extinction/source: A_COL_K",
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

    # the Planck arm recentred exactly as the Herschel arm is above --
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

    # the cloud-class within-beam tilt (`gamma`) and Herschel structural
    # width (`sigma_cloud`), fit JOINTLY on the HOPS/eHOPS protostars
    # (Orion A, Aquila) rather than assuming the published star-gas law's
    # own exponent within a beam (owner, 2026-09-12 evening ruling 1).
    fit = _fit_cloud_gamma_sigma_herschel(config, zp)

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
        f.create_dataset("CLOUD_GAMMA_HERSCHEL", data=np.float64(fit["gamma"]))
        f.create_dataset("CLOUD_GAMMA_HERSCHEL_P16", data=np.float64(fit["gamma_p16"]))
        f.create_dataset("CLOUD_GAMMA_HERSCHEL_P84", data=np.float64(fit["gamma_p84"]))
        f.create_dataset("CLOUD_SIGMA_GRID_DEX", data=fit["sigma_grid"].astype(np.float64))
        f.create_dataset("CLOUD_GAMMA_GRID", data=fit["gamma_grid"].astype(np.float64))
        f.create_dataset("CLOUD_GAMMA_SIGMA_LOGLIKE", data=fit["loglike2d"].astype(np.float64))
        f.create_dataset("N_PROTOSTARS_FIT", data=np.int64(fit["n_protostars"]))

    st.done(out_path, n_arms=len(_ARM_ORDER), n_node=n_node, zp_herschel_k=float(zp),
            herschel_stretch=herschel_factor, cloud_sigma_herschel_dex=fit["sigma_cloud"],
            cloud_gamma_herschel=fit["gamma"], n_protostars_fit=fit["n_protostars"])
    print("kernel: %d arms x %d nodes (mixture), zp_herschel_k=%.4f, "
          "herschel_stretch=%.4f -> %s"
          % (len(_ARM_ORDER), n_node, zp, herschel_factor, out_path), flush=True)
    print("kernel: Planck arm recentred, max |E[T/A_beam] - 1| = %.3e"
          % float(np.max(np.abs(e_check_p - 1.0))), flush=True)
    print("kernel: joint fit -- gamma_mle=%.2f (68%% interval %.2f-%.2f), "
          "sigma_mle=%.3f dex; parsimony_zero=%s -> adopted gamma=%.2f, sigma=%.3f dex "
          "(68%% interval %.3f-%.3f dex), N_PROTOSTARS_FIT=%d, A_beam dataset=%s"
          % (fit["gamma_mle"], fit["gamma_p16"], fit["gamma_p84"], fit["sigma_mle"],
             fit["parsimony_zero"], fit["gamma"], fit["sigma_cloud"], fit["p16"], fit["p84"],
             fit["n_protostars"], fit["a_beam_dataset"]), flush=True)
    print("kernel: log-likelihood at the joint maximum = %.4f; profiled over sigma at "
          "gamma = 0, 1, 2: %.4f, %.4f, %.4f; at the adopted (gamma, sigma) = %.4f"
          % (fit["loglike_max"], fit["loglike_at"][0.0], fit["loglike_at"][1.0],
             fit["loglike_at"][2.0], fit["loglike_adopted"]), flush=True)
    print("kernel: protostars dropped -- no Herschel arm: %d, no finite A_V: %d, "
          "no SESNA catalogue match: %d" % (fit["n_dropped_no_herschel_arm"], fit["n_dropped_no_av"],
                                             fit["n_dropped_no_sightline"]), flush=True)
    print("kernel: check (adopted gamma, sigma) -- predicted/empirical median log10 r = %.4f/%.4f, "
          "predicted/empirical p84 log10 r = %.4f/%.4f, frac P(T>=a_p)<0.01 = %.4f"
          % (fit["pred_median"], fit["emp_median"], fit["pred_p84"], fit["emp_p84"],
             fit["frac_below_reach"]), flush=True)


if __name__ == "__main__":
    from sesnaimpute.build import run
    run(build)
