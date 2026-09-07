"""The closed-form SED fit at the unconstrained optimum (SPEC_BMSTP_DRAFT.md
section 6.1), for every template of a library against a block of sources at
once. A library module: no `progress` use, no runbook line
(IMPLEMENTATION_BMSTP_DRAFT.md section 4 row 2.2).

`prepare` builds, once per block of sources, the per-source design `X`, its
normal matrix and projector, and the two numbers (`sigma_a`, the conditional
slope) the prior read needs (section 1.3). `fit` then reduces every
template to two matrix products per source: `chi2_min = r^T P r` and the
unconstrained `(Av_hat, SC_hat) = r^T M^T`. This module returns those
unconstrained marks only -- the clamp to `[0, 75 / (A_K/A_V)]` with `SC_hat`
re-solved for the reported marks and the flux prediction, and the
non-detection term of section 6.2, are a further unit's work.
"""

import numpy as np

from sesnaimpute import definitions
from sesnaimpute.population import selection as population_selection

#: The eight SESNA bands, in the order every array here uses
#: (`definitions.BANDS`; matches `catalog.limits.limits`'s column order).
BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: `catalog.limits.limits`'s two band groups: the five Spitzer bands carry
#: a per-source DCOMP90 map rescaled by the region's fitted Delta; the
#: three 2MASS bands carry no per-source map, so every source in a region
#: shares its F50 flux.
IRAC_MIPS_KEYS = ("I1", "I2", "I3", "I4", "M1")
TWOMASS_KEYS = ("J", "H", "Ks")

#: SPEC_BMSTP_DRAFT.md section 2 -- `log10 B = -2 SC`, the design's constant
#: gray-scale column.
GRAY_COLUMN = -2.0

#: SPEC_PRIORS.md 1.3 / `population.selection.MIN_BANDS` -- SESNA's own
#: two-of-eight catalogue-inclusion rule: a source with fewer detected
#: bands has no two-parameter fit and is flagged, not fitted.
MIN_DETECTED_BANDS = 2


class Batch:
    """The per-source quantities the closed form needs, computed once per
    block and reused for every template: `log10_f_obs` and its weight,
    `P` and `M` (SPEC_BMSTP_DRAFT.md section 6.1), `sigma_a` and the
    conditional slope (section 1.3), and the per-band detection limit and
    roll-off width (section 6.2), all at each source of the block.
    """

    __slots__ = ("log10_f_obs", "weight", "P", "M", "sigma_a_ak",
                 "slope_sc_av", "ak_per_av", "f_lim50", "width_dex",
                 "n_detected", "flagged")

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


def _f_lim50(dcomp90, delta_dex, f50_2mass):
    """The per-source 50%-completeness flux, all eight bands
    (`catalog.limits.limits`'s formula, reproduced here on an
    already-read batch rather than re-opening the curated file): the
    Spitzer bands rescale each source's own DCOMP90 map value by the
    region's fitted Delta; the three 2MASS bands, which carry no
    per-source map, take the region's F50 flux unchanged.
    """
    n = dcomp90.shape[0]
    out = np.empty((n, N_BANDS), dtype=np.float64)
    for j, key in enumerate(BAND_KEYS):
        if key in TWOMASS_KEYS:
            out[:, j] = f50_2mass[TWOMASS_KEYS.index(key)]
        else:
            out[:, j] = dcomp90[:, j] * 10.0 ** (-delta_dex[IRAC_MIPS_KEYS.index(key)])
    return out


def prepare(config, flux, sigma, origin, ak_col, dcomp90, delta_dex, f50_2mass, width_dex):
    """Source batch preparation (SPEC_BMSTP_DRAFT.md section 6.1): from one
    block's curated rows -- fluxes, uncertainties and `ORIGIN_FNU`, the
    same read `fittp.cascade` uses -- plus the source's own column
    `ak_col` (`AK_SESNA`), its DCOMP90 map value, and the region's depth
    constants (`delta_dex`, `f50_2mass`, `width_dex` from `catalog.depths`,
    read once per region by the caller), builds the per-source design,
    weight and projector every template's fit reuses.
    """
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    detected = np.asarray(origin) == 1
    n = flux.shape[0]

    # log10 f_obs with the small-error log transform and its weight
    # (section 6.1): sigma_log = sigma_f / (f ln 10), weight 1/sigma_log^2
    # for detected bands, zero elsewhere (undetected bands never enter the
    # sum: see the zero row/column of P below).
    safe_flux = np.where(detected & (flux > 0), flux, 1.0)
    log10_f_obs = np.log10(safe_flux)
    sigma_log = sigma / (safe_flux * np.log(10.0))
    weight = np.where(detected & (sigma_log > 0), 1.0 / sigma_log ** 2, 0.0)

    # the blended extinction law at the source's own column (section 2):
    # kappa_hybrid is K-normalised (kappa_Ks = 1); multiplying by the same
    # blend's A_K/A_V ratio turns it into the A_V-normalised design column
    # section 6.1 fits in, and the same ratio converts the fitted Av_hat
    # back to a_hat (A_K) below.
    w_ramp = population_selection.law_dense_weight(ak_col)
    kappa_k = population_selection.kappa_hybrid(config, w_ramp)
    ak_per_av = population_selection.ak_per_av(config, w_ramp)
    kappa_v = kappa_k * ak_per_av[:, None]

    design = np.empty((n, N_BANDS, 2), dtype=np.float64)
    design[:, :, 0] = -0.4 * kappa_v
    design[:, :, 1] = GRAY_COLUMN

    wx = design * weight[:, :, None]
    xtwx = np.einsum("nbk,nbl->nkl", wx, design)

    n_detected = detected.sum(axis=1)
    det_xtwx = np.linalg.det(xtwx)
    flagged = (n_detected < MIN_DETECTED_BANDS) | (det_xtwx <= 0)
    xtwx_safe = xtwx.copy()
    xtwx_safe[flagged] = np.eye(2)
    xtwx_inv = np.linalg.inv(xtwx_safe)

    m_mat = np.einsum("nkl,nbl->nkb", xtwx_inv, wx)          # (XtWX)^-1 XtW, (n, 2, 8)
    p_mat = -np.einsum("nbk,nkc->nbc", wx, m_mat)
    diag = np.arange(N_BANDS)
    p_mat[:, diag, diag] += weight

    # sigma_a^2 = [(XtWX)^-1]_aa (section 1.3), converted A_V -> A_K by the
    # same ratio; the conditional slope d SC / d A_V from the same matrix.
    sigma_a_ak = np.sqrt(xtwx_inv[:, 0, 0]) * ak_per_av
    slope_sc_av = xtwx_inv[:, 1, 0] / xtwx_inv[:, 0, 0]

    f_lim50 = _f_lim50(np.asarray(dcomp90, dtype=np.float64), delta_dex, f50_2mass)
    width = np.broadcast_to(np.asarray(width_dex, dtype=np.float64), (n, N_BANDS)).copy()

    return Batch(log10_f_obs=log10_f_obs,
                 weight=weight.astype(np.float32),
                 P=p_mat, M=m_mat,
                 sigma_a_ak=sigma_a_ak, slope_sc_av=slope_sc_av, ak_per_av=ak_per_av,
                 f_lim50=f_lim50.astype(np.float32), width_dex=width.astype(np.float32),
                 n_detected=n_detected.astype(np.int8), flagged=flagged)


def fit(batch, log10_f_ref):
    """The closed-form fit at the UNCONSTRAINED optimum (SPEC_BMSTP_DRAFT.md
    section 6.1), two matrix products per source over every template of
    `log10_f_ref` (`(m, 8)` float32) at once: `chi2_min = r^T P r` and
    `(Av_hat, SC_hat) = r^T M^T`, `r = log10_f_obs - log10_f_ref`. Returns
    `chi2_min, a_hat, log10_b_hat`, each `(n, m)` float32, with
    `a_hat = Av_hat * (A_K/A_V)_s` and `log10_b_hat = -2 SC_hat` the marks
    the prior read integrates over (section 1.3) -- NOT the clamped marks
    of the reported record, and without the non-detection term of section
    6.2. Rows flagged at `prepare` (fewer than two detected bands, or a
    singular `XtWX`) are NaN.
    """
    log10_f_ref = np.asarray(log10_f_ref, dtype=np.float32)
    # log10_f_obs stays float64 (CODING_RULES_BMSTP.md rule 3's "float32
    # template arrays" is about the template side): subtracting a float32
    # template from it promotes r to float64, so the closed form's own
    # precision is not spent on this per-source, per-block-sized array.
    r = batch.log10_f_obs[:, None, :] - log10_f_ref[None, :, :]

    chi2_min = np.einsum("nmb,nbc,nmc->nm", r, batch.P, r).astype(np.float32)
    marks = np.einsum("nmb,nkb->nmk", r, batch.M)
    a_hat = (marks[:, :, 0] * batch.ak_per_av[:, None]).astype(np.float32)
    log10_b_hat = (-2.0 * marks[:, :, 1]).astype(np.float32)

    chi2_min[batch.flagged] = np.nan
    a_hat[batch.flagged] = np.nan
    log10_b_hat[batch.flagged] = np.nan
    return chi2_min, a_hat, log10_b_hat
