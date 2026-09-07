"""The closed-form SED fit (SPEC_BMSTP_DRAFT.md section 6.1) and the
non-detection term (section 6.2), for every template of a library against a
block of sources at once. A library module: no `progress` use, no runbook
line (IMPLEMENTATION_BMSTP_DRAFT.md section 4 row 2.2).

`prepare` builds, once per block of sources, the per-source design `X`, its
normal matrix and projector, and the numbers the prior read needs (section
1.3: `sigma_a`, the conditional slope). `fit` then reduces every template to
two matrix products per source for the UNCONSTRAINED optimum the prior read
integrates over, the CLAMPED marks for the reported record and the flux
prediction, and the non-detection term at those clamped marks.
"""

import math
from collections import namedtuple

import numba
import numpy as np

from sesnaimpute import definitions
from sesnaimpute.catalog import limits as catalog_limits
from sesnaimpute.population import selection as population_selection

#: The eight SESNA bands, in the order every array here uses
#: (`definitions.BANDS`; matches `catalog.limits.limits`'s column order).
BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: SPEC_BMSTP_DRAFT.md section 2 -- `log10 B = -2 SC`, the design's constant
#: gray-scale column.
GRAY_COLUMN = -2.0

#: SPEC_PRIORS.md 1.3 / `population.selection.MIN_BANDS` -- SESNA's own
#: two-of-eight catalogue-inclusion rule: a source with fewer detected
#: bands has no two-parameter fit and is flagged, not fitted.
MIN_DETECTED_BANDS = 2

#: SPEC_BMSTP_DRAFT.md section 6.1 -- the reported mark's ceiling, in A_K
#: magnitudes; the design's own A_V amplitude is clamped to this converted
#: by the source's own (A_K/A_V).
AV_CLAMP_MAX_AK = 75.0

_SQRT2 = np.float32(np.sqrt(2.0))

#: Per-block working set (rule 10b/10a), in float32-`(n_block, n_model, 8)`
#: equivalents: the chi2 quadratic form's `R @ P` (1), and the
#: non-detection term's `log10_fhat`/`z` (float32, 1 each) plus the numba
#: kernel's float64 copy and output (2 float32-equivalents each, for the
#: float64 precision the z=0 identity needs) -- 7 in all; `block_size`
#: sizes a block so that many such arrays fit `budget_mb`.
NONDET_BUFFERS = 7


#: SPEC_BMSTP_DRAFT.md section 6.2 -- above this `z`, `_ln_one_minus_c_kernel`
#: switches from the direct `ln erfc(z)` (exact, no underflow risk yet: erfc
#: does not underflow to zero until z ~ 27) to the asymptotic series, so the
#: switch is a speed/simplicity choice, not a stability one.
_ERFC_DIRECT_MAX_Z = 5.0


@numba.njit(parallel=True, cache=True)
def _ln_one_minus_c_kernel(z_flat, out):
    """`ln[1 - C(z)]` for every element of the flattened `z` (section 6.2's
    erf roll-off, `z = (log10 f_hat - log10 F_lim50) / (sqrt(2) w)`),
    `@njit(parallel=True)` over templates via `prange`: for `z < 0` (fainter
    than the limit) `log1p(-0.5 erfc(-z))` has no cancellation to guard
    against; for `0 <= z < 5`, `math.erfc` itself has not yet underflowed,
    so `ln(1/2) + ln erfc(z)` is exact; beyond that, the scaled
    complementary error function's own asymptotic series, `ln erfcx(z) ~=
    -ln(z sqrt(pi)) - ln(1 + 1/(2 z^2))`, avoids `erfc`'s eventual
    underflow (z ~ 27) with the leading-order tail term the spec's own
    "ten times the limit costs about 5 nats" case sits well inside.
    """
    n = z_flat.shape[0]
    ln_half = -0.6931471805599453
    sqrt_pi = 1.7724538509055159
    for i in numba.prange(n):
        zi = z_flat[i]
        if zi < 0.0:
            out[i] = math.log1p(-0.5 * math.erfc(-zi))
        elif zi < _ERFC_DIRECT_MAX_Z:
            out[i] = ln_half + math.log(math.erfc(zi))
        else:
            ln_erfcx = -math.log(zi * sqrt_pi) - math.log(1.0 + 1.0 / (2.0 * zi * zi))
            out[i] = ln_half + ln_erfcx - zi * zi
    return out


def _ln_one_minus_c(z):
    """`ln[1 - C(z)]`, section 6.2 (`_ln_one_minus_c_kernel`'s docstring for
    the three branches), on an array of any shape: at `z = 0` this is
    `ln(1/2)` to float64 precision (`math.erfc(0.0) == 1.0` exactly), so the
    identity check holds to 1e-10 with no float32 rounding in the way.
    """
    z = np.asarray(z, dtype=np.float64)
    shape = z.shape
    z_flat = np.ascontiguousarray(z.reshape(-1))
    out = np.empty_like(z_flat)
    _ln_one_minus_c_kernel(z_flat, out)
    return out.reshape(shape)


def _ln_nondet(batch, log10_f_ref, av_clamped32, sc_clamped32):
    """The non-detection term of section 6.2 at the CLAMPED marks: the
    predicted log-flux in every band (float32) for every source and
    template at once, masked to each source's own undetected, limited
    bands and summed. One batched call into `_ln_one_minus_c`'s numba
    kernel beats gathering each source's own undetected-band slice and
    calling it once per source (measured: ~2.5x slower for a 27-source,
    200,000-template block -- the per-source Python-level overhead
    dominates the kernel's own, embarrassingly parallel, per-element cost).
    """
    log10_fhat = (log10_f_ref[None, :, :]
                  + batch.ext_col[:, None, :] * av_clamped32[:, :, None]
                  + np.float32(GRAY_COLUMN) * sc_clamped32[:, :, None])
    z = (log10_fhat - batch.log10_f_lim50[:, None, :]) / (_SQRT2 * batch.width_dex[:, None, :])
    term = _ln_one_minus_c(z)
    term = np.where(batch.nondet_mask[:, None, :], term, 0.0)
    return term.sum(axis=2).astype(np.float32)


class Batch:
    """The per-source quantities the closed form and the non-detection term
    need, computed once per block and reused for every template:
    `log10_f_obs` and its weight, `P` and `M` (section 6.1), `sigma_a` and
    the conditional slope (section 1.3), the extinction design column
    (`ext_col`) and the two scalars the clamp's re-solve uses (`s0`,
    `w_sum`), and the per-band `F_LIM_50` and roll-off width `WIDTH_DEX`
    (section 6.2).
    """

    __slots__ = ("log10_f_obs", "weight", "P", "M", "ext_col", "s0", "w_sum",
                 "sigma_a_ak", "slope_sc_av", "ak_per_av",
                 "log10_f_lim50", "width_dex", "nondet_mask",
                 "n_detected", "flagged")

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


Fit = namedtuple("Fit", (
    "chi2_min",              # (n, m) f4 -- at the unconstrained optimum
    "a_hat",                 # (n, m) f8 -- unconstrained, for the prior read's identity
    "log10_b_hat",           # (n, m) f8 -- unconstrained
    "a_hat_clamped",         # (n, m) f4 -- for the reported record and flux prediction
    "log10_b_hat_clamped",   # (n, m) f4
    "frac_clamped",          # (n,)   f4 -- FRAC_CLAMPED, fraction of templates clamped
    "ln_nondet",             # (n, m) f4 -- section 6.2, at the clamped marks
))


def block_size(n_model, budget_mb=512, extra_buffers=0):
    """Sources per block a `budget_mb`-MB budget holds for a library of
    `n_model` templates (rule 10a): `NONDET_BUFFERS` float32
    `(n_block, n_model, 8)` arrays are this module's own working set (the
    chi2 quadratic form's `R @ P` and the non-detection term's `log10_fhat`,
    `z` and `term`), not the much smaller `(n_block, n_model)` outputs.
    `extra_buffers` counts a caller's own float32-`(n_block, n_model, 8)`-
    equivalent arrays that live for the same block (e.g. `fittp.sweep`'s
    `log10_flux`/`flux_theta`, two float64 arrays = 4 such equivalents) so
    the block's real total working set, not just this module's share,
    stays inside `budget_mb`.
    """
    row_bytes = n_model * N_BANDS * 4 * (NONDET_BUFFERS + extra_buffers)
    return max(1, (budget_mb << 20) // max(1, row_bytes))


def prepare(config, region, start, stop, flux, sigma, origin, ak_col, width_dex):
    """Source batch preparation (SPEC_BMSTP_DRAFT.md section 6.1): from one
    block's curated rows -- fluxes, uncertainties and `ORIGIN_FNU`, the
    same read `fittp.cascade` uses, for source rows `[start, stop)` of
    `region` -- plus the source's own column `ak_col` (`AK_SESNA`) and the
    region's roll-off width `width_dex` (`WIDTH_DEX`, all eight bands, from
    `catalog.depths`, read once per region by the caller), builds the
    per-source design, weight and projector every template's fit reuses.
    `F_LIM_50` is `catalog.limits.limits`'s own per-source, per-band value
    (the package's one definition), sliced to this block's rows.
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
    ext_col = -0.4 * kappa_v

    design = np.empty((n, N_BANDS, 2), dtype=np.float64)
    design[:, :, 0] = ext_col
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

    # the clamp's re-solve (fit()) needs only these two per-source scalars:
    # S0 = sum_b W_b X_b0 (the extinction column's weighted sum) and the
    # summed weight, because the gray column is the same constant in every
    # band (see fit()'s docstring for the algebra).
    s0 = (weight * ext_col).sum(axis=1)
    w_sum = weight.sum(axis=1)

    f_lim50 = catalog_limits.limits(config, region)[start:stop]
    log10_f_lim50 = np.log10(f_lim50)
    width = np.broadcast_to(np.asarray(width_dex, dtype=np.float64), (n, N_BANDS))
    # section 6.2: a band with no measurement and no limit is excluded;
    # SESNA's curated substitution always supplies one or the other, so
    # this only guards a non-finite F_LIM_50.
    nondet_mask = (~detected) & np.isfinite(f_lim50)

    return Batch(log10_f_obs=log10_f_obs, weight=weight.astype(np.float32),
                 P=p_mat, M=m_mat, ext_col=ext_col.astype(np.float32),
                 s0=s0, w_sum=w_sum,
                 sigma_a_ak=sigma_a_ak, slope_sc_av=slope_sc_av, ak_per_av=ak_per_av,
                 log10_f_lim50=log10_f_lim50.astype(np.float32),
                 width_dex=width.astype(np.float32), nondet_mask=nondet_mask,
                 n_detected=n_detected.astype(np.int8), flagged=flagged)


def fit(batch, log10_f_ref):
    """The closed-form fit of SPEC_BMSTP_DRAFT.md section 6.1 over every
    template of `log10_f_ref` (`(m, 8)` float32) at once, `r = log10_f_obs -
    log10_f_ref`:

    - `chi2_min = r^T P r` and the UNCONSTRAINED `(Av_hat, SC_hat) = r^T
      M^T` at every source, kept in float64 (`a_hat`, `log10_b_hat`) since
      the prior read's own identity needs it; `chi2_min` is float32.
    - The CLAMPED marks, `Av` restricted to `[0, 75/(A_K/A_V)_s]`: because
      the gray column is one constant in every band, the least-squares
      residual is orthogonal to both design columns, so re-solving `SC` at
      fixed `Av` needs no per-band array at all --
      `SC_clamped = SC_hat + (Av_hat - Av_clamped) * S0 / (gray * W_sum)`,
      `S0 = sum_b W_b X_b0` (`prepare`'s `s0`). `FRAC_CLAMPED` is the
      fraction of `m` templates where the clamp engaged, per source.
    - The non-detection term of section 6.2 at the CLAMPED marks: the
      template's model flux `f_hat_i` in every undetected, limited band,
      compared to the source's own `F_LIM_50,i` through the region-band
      roll-off width `WIDTH_DEX`, `ln[1 - C_i(f_hat_i)]` via the scaled
      complementary error function (`_ln_one_minus_c`), gathered per
      source to only its own undetected bands (`_ln_nondet`).

    Rows flagged at `prepare` (fewer than two detected bands, or a singular
    `XtWX`) are NaN throughout.
    """
    log10_f_ref = np.asarray(log10_f_ref, dtype=np.float32)
    r = batch.log10_f_obs[:, None, :] - log10_f_ref[None, :, :]     # (n, m, 8) float64

    # r^T P r as two (m, 8) @ (8, 8) matmuls per source (P symmetric),
    # batched over sources: faster than a single three-index einsum
    # because it dispatches to a per-source BLAS gemm; kept in float64 so
    # chi2_min matches the direct lstsq residual to the fit's own
    # precision, not float32's.
    rp = np.matmul(r, batch.P)
    chi2_min = (r * rp).sum(axis=2).astype(np.float32)

    marks = np.einsum("nmb,nkb->nmk", r, batch.M)
    av_hat = marks[:, :, 0]
    sc_hat = marks[:, :, 1]
    a_hat = av_hat * batch.ak_per_av[:, None]
    log10_b_hat = -2.0 * sc_hat

    av_max = AV_CLAMP_MAX_AK / batch.ak_per_av
    av_clamped = np.clip(av_hat, 0.0, av_max[:, None])
    clamp_engaged = av_hat != av_clamped
    resolve_ratio = batch.s0 / (GRAY_COLUMN * batch.w_sum)
    sc_clamped = sc_hat + (av_hat - av_clamped) * resolve_ratio[:, None]

    a_hat_clamped = (av_clamped * batch.ak_per_av[:, None]).astype(np.float32)
    log10_b_hat_clamped = (-2.0 * sc_clamped).astype(np.float32)
    frac_clamped = clamp_engaged.mean(axis=1).astype(np.float32)

    av_clamped32 = av_clamped.astype(np.float32)
    sc_clamped32 = sc_clamped.astype(np.float32)
    ln_nondet = _ln_nondet(batch, log10_f_ref, av_clamped32, sc_clamped32)

    chi2_min[batch.flagged] = np.nan
    a_hat[batch.flagged] = np.nan
    log10_b_hat[batch.flagged] = np.nan
    a_hat_clamped[batch.flagged] = np.nan
    log10_b_hat_clamped[batch.flagged] = np.nan
    ln_nondet[batch.flagged] = np.nan

    return Fit(chi2_min=chi2_min, a_hat=a_hat, log10_b_hat=log10_b_hat,
               a_hat_clamped=a_hat_clamped, log10_b_hat_clamped=log10_b_hat_clamped,
               frac_clamped=frac_clamped, ln_nondet=ln_nondet)
