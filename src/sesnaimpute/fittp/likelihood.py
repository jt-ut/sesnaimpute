"""The closed-form SED fit (SPEC_BMSTP_DRAFT.md section 6.1) and the
non-detection term (section 6.2), for every template of a library against a
block of sources at once. A library module: no `progress` use, no runbook
line (IMPLEMENTATION_BMSTP_DRAFT.md section 4 row 2.2).

`prepare` builds, once per block of sources and the block's own class/
library, the per-band weight at variance `sigma_i^2 = sigma_log,i^2 +
sigma_cal,i^2 + sigma_lib,L^2` (section 6.1; `sigma_lib,L` read from `fittp.
library_resolution`'s check product for the class named), a band SESNA
marks detected but whose flux is <= 0 counted as unmeasured rather than a
fabricated datum, and a non-finite sigma on a detected band flagging the
source rather than carrying a NaN normalisation into every template's
likelihood. `fit` then solves the diffuse/dense extinction-law blend of
section 2 self-consistently against the fit's own extinction mark (never
the sightline column or SESNA's published `AK`, spec section 2's ruling):
starting from the pure diffuse design, it fits every template, reads the
source's own extinction back from the result, rebuilds the design at the
blend that mark implies, and refits, until the source's own mark stops
moving by more than 0.01 mag or four rounds have run -- the design a
source's whole library shares needs one blend weight, so each round's
mark is every template's own fitted extinction taken together (the
median), not any one template alone. Every template is then reduced to
two matrix products at that converged design for the UNCONSTRAINED
optimum the prior read integrates over, the CLAMPED marks for the
reported record and the flux prediction, and the non-detection term (plus
section 6.1's per-source normalisation) at those clamped marks.
"""

import math
from collections import namedtuple
from functools import lru_cache

import h5py
import numba
import numpy as np

from sesnaimpute import config as config_module
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

#: SPEC_BMSTP_DRAFT.md section 6.1 -- the absolute-calibration systematic
#: per band, dex, added in quadrature to the statistical log-flux error;
#: `BAND_KEYS`' own order (2MASS J, H, Ks; IRAC I1-I4; MIPS M1).
SIGMA_CAL_DEX = np.array([
    0.010, 0.010, 0.010,          # 2MASS -- Skrutskie et al. 2006, AJ 131, 1163
    0.013, 0.013, 0.013, 0.013,   # IRAC -- Reach et al. 2005, astro-ph/0507139
    0.017,                        # MIPS 24 um -- Engelbracht et al. 2007, PASP 119, 994
], dtype=np.float64)

#: SPEC_BMSTP_DRAFT.md section 2, owner's ruling 2026-09-10 -- the
#: self-consistent extinction-law solve's own stopping rule: converged once
#: the source's own mark moves less than this between rounds, in A_K
#: magnitudes, or after this many rounds, whichever comes first.
LAW_ITER_TOL_MAG = 0.01
LAW_ITER_MAX = 4

_SQRT2 = np.float32(np.sqrt(2.0))

#: Per-block working set (rule 10b/10a), in float32-`(n_block, n_model, 8)`
#: equivalents: `r` (float64, held across the chi2 and marks products, 2)
#: plus `R @ P` (float64, transient during the chi2 product, 2) or, once
#: freed, the non-detection term's `log10_fhat`/`z` (float32, 1 each), the
#: numba kernel's own float64 output (2, for the float64 precision the
#: z=0 identity needs -- the kernel casts each element to float64 inside
#: the loop, so no separate float64 copy of the input `z` array is held)
#: and its masked copy (2) -- 8 in all at the non-detection term's own
#: peak (`r` still live, `log10_fhat` + `z` + kernel output + masked
#: copy); `block_size` sizes a block so that many such arrays fit
#: `budget_mb`. The extinction-law solve's own per-round design (`(n,
#: 8, 8)`/`(n, 8, 2)`) carries no model-axis and is negligible beside
#: these, so it adds nothing here even run up to `LAW_ITER_MAX` times.
NONDET_BUFFERS = 8


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
    `z_flat` is float32 (the block's own working dtype); each element is
    cast to float64 here, per scalar, so the z=0 identity keeps double
    precision without a separate float64 copy of the whole array.
    """
    n = z_flat.shape[0]
    ln_half = -0.6931471805599453
    sqrt_pi = 1.7724538509055159
    for i in numba.prange(n):
        zi = np.float64(z_flat[i])
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
    `ln(1/2)` to float64 precision (`math.erfc(0.0) == 1.0` exactly
    regardless of `z`'s own dtype), so the identity check holds to 1e-10
    with no float32 rounding in the way. `z` keeps its own dtype (float32
    from the block's working set) into the kernel -- only the kernel's
    per-element arithmetic and its output are float64, so no full-array
    float64 copy of `z` is made here.
    """
    z = np.asarray(z)
    shape = z.shape
    z_flat = np.ascontiguousarray(z.reshape(-1))
    out = np.empty(z_flat.shape, dtype=np.float64)
    _ln_one_minus_c_kernel(z_flat, out)
    return out.reshape(shape)


def _ln_nondet(batch, ext_col32, log10_f_ref, av_clamped32, sc_clamped32):
    """The non-detection term of section 6.2 at the CLAMPED marks: the
    predicted log-flux in every band (float32) for every source and
    template at once, masked to each source's own undetected, limited
    bands and summed. `ext_col32` is the extinction-law solve's own
    converged design column (float32, section 2). One batched call into
    `_ln_one_minus_c`'s numba kernel beats gathering each source's own
    undetected-band slice and calling it once per source (measured: ~2.5x
    slower for a 27-source, 200,000-template block -- the per-source
    Python-level overhead dominates the kernel's own, embarrassingly
    parallel, per-element cost).
    """
    log10_fhat = (log10_f_ref[None, :, :]
                  + ext_col32[:, None, :] * av_clamped32[:, :, None]
                  + np.float32(GRAY_COLUMN) * sc_clamped32[:, :, None])
    z = (log10_fhat - batch.log10_f_lim50[:, None, :]) / (_SQRT2 * batch.width_dex[:, None, :])
    term = _ln_one_minus_c(z)
    term = np.where(batch.nondet_mask[:, None, :], term, 0.0)
    return term.sum(axis=2).astype(np.float32)


class Batch:
    """The per-source quantities the closed form and the non-detection term
    need: `log10_f_obs` (float64, section 6.1) and its weight, `flagged`
    (fewer than two valid detections, or a non-finite sigma on a detected
    band, section 6.1/16), the per-band `F_LIM_50` and roll-off width
    `WIDTH_DEX` (section 6.2), `ln_norm_term` (section 6.1's per-source
    Gaussian normalisation, over only the bands with a finite variance),
    and `config` (the extinction-law solve's own need, `fit`'s docstring).
    `fit` fills in, once its self-consistent solve has converged, the
    design's own extinction column (`ext_col`) and the source's `sigma_a`,
    conditional slope and (A_K/A_V) at that design (section 1.3) -- one
    number per source, since every template of a source shares the one
    design the solve converges to.
    """

    __slots__ = ("log10_f_obs", "weight", "config",
                 "log10_f_lim50", "width_dex", "nondet_mask",
                 "n_detected", "flagged", "ln_norm_term",
                 "ext_col", "s0", "w_sum", "sigma_a_ak", "slope_sc_av", "ak_per_av")

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
    "ln_nondet",             # (n, m) f4 -- section 6.2 (clamped marks) plus
                             # section 6.1's per-source ln_norm_term, the one
                             # sum `fittp.sweep` already adds unscaled into ln L_hat
    "n_law_iter",            # (n,)   i1 -- N_LAW_ITER, rounds the law solve ran
))


@lru_cache(maxsize=None)
def _sigma_lib_by_class(path):
    """`{class code: sigma_lib,L dex}`, section 6.1's one number per LIBRARY,
    from `fittp.library_resolution`'s check product (its `LIBRARY`/
    `SIGMA_LIB_DEX` datasets). Cached on the product's own path: the file
    is tiny (six numbers) but `prepare` is called once per block, so an
    open+read per block would otherwise repeat needlessly (rule 9).
    """
    with h5py.File(path, "r") as f:
        libs = np.char.decode(f["LIBRARY"][:].astype("S"), "utf-8")
        sigma = f["SIGMA_LIB_DEX"][:]
    return dict(zip(libs.tolist(), sigma.tolist()))


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


def prepare(config, region, cls, start, stop, flux, sigma, origin, width_dex):
    """Source batch preparation (SPEC_BMSTP_DRAFT.md section 6.1): from one
    block's curated rows -- fluxes, uncertainties and `ORIGIN_FNU`, the
    same read `fittp.cascade` uses, for source rows `[start, stop)` of
    `region` -- plus the region's roll-off width `width_dex` (`WIDTH_DEX`,
    all eight bands, from `catalog.depths`, read once per region by the
    caller), builds the per-source weight and the fit's other design-
    independent numbers. `F_LIM_50` is `catalog.limits.limits`'s own
    per-source, per-band value (the package's one definition), sliced to
    this block's rows. `cls` (the class this block's library belongs to,
    e.g. "YSO") selects the one `sigma_lib,L` number the class's library
    carries (`fittp.library_resolution`'s check product; the YSO register
    is one set, so every YSO subclass shares it, section 1.4). The
    extinction-law design itself is not built here: it depends on the
    fit's own extinction mark (section 2), so `fit` builds and converges
    it once the per-template residual exists.
    """
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = flux.shape[0]

    lib_path = config_module.product_path(config, "fittp", "check", "library-resolution", "survey")
    sigma_lib_l = _sigma_lib_by_class(lib_path)[cls]

    # (15, R2 D2): SESNA marks a band detected (ORIGIN_FNU == 1) but its
    # flux can still be <= 0; such a band carries no measurement and is
    # excluded from "detected" here, so it never enters the fit as a
    # fabricated 1 mJy datum -- it falls through to nondet_mask below and
    # is scored by the non-detection term at the source's own limit
    # instead, exactly like a genuine non-detection.
    origin_detected = np.asarray(origin) == 1
    detected = origin_detected & (flux > 0)

    # section 6.1: the per-band variance is the statistical log-flux error
    # in quadrature with the band's absolute-calibration systematic
    # (SIGMA_CAL_DEX) and the class's library resolution (sigma_lib,L,
    # one number, the same in every band): sigma_log = sigma_f/(f ln 10),
    # weight 1/sigma_i^2 for detected bands, zero elsewhere (undetected
    # bands never enter the sum: see the zero row/column of P in `fit`).
    safe_flux = np.where(detected, flux, 1.0)
    log10_f_obs = np.log10(safe_flux)
    sigma_log = sigma / (safe_flux * np.log(10.0))
    sigma2 = sigma_log ** 2 + SIGMA_CAL_DEX[None, :] ** 2 + sigma_lib_l ** 2
    weight = np.where(detected & (sigma_log > 0), 1.0 / sigma2, 0.0)

    # (16, R2 D3): a non-finite sigma on a detected band already carries
    # zero weight (sigma_log > 0 is False for NaN), but sigma2 itself is
    # still NaN; summing it into ln_norm_term regardless silently zeroed
    # the source's whole posterior. Excluded from the sum instead, and the
    # source flagged (sweep.py then records and excludes it, exactly as a
    # too-few-bands source already is).
    finite_sigma2 = np.isfinite(sigma2)
    flagged_sigma = (detected & ~finite_sigma2).any(axis=1)
    ln_norm_term = -0.5 * np.where(detected & finite_sigma2, np.log(sigma2), 0.0).sum(axis=1)

    n_detected = detected.sum(axis=1)
    flagged = (n_detected < MIN_DETECTED_BANDS) | flagged_sigma

    f_lim50 = catalog_limits.limits(config, region)[start:stop]
    log10_f_lim50 = np.log10(f_lim50)
    width = np.broadcast_to(np.asarray(width_dex, dtype=np.float64), (n, N_BANDS))
    # section 6.2: a band with no measurement and no limit is excluded;
    # SESNA's curated substitution always supplies one or the other, so
    # this only guards a non-finite F_LIM_50. Row 15's unmeasured
    # detections fall in here too, since `detected` already excludes them.
    nondet_mask = (~detected) & np.isfinite(f_lim50)

    return Batch(log10_f_obs=log10_f_obs, weight=weight, config=config,
                 log10_f_lim50=log10_f_lim50.astype(np.float32),
                 width_dex=width.astype(np.float32), nondet_mask=nondet_mask,
                 n_detected=n_detected.astype(np.int8), flagged=flagged,
                 ln_norm_term=ln_norm_term)


def fit(batch, log10_f_ref):
    """The closed-form fit of SPEC_BMSTP_DRAFT.md section 6.1 over every
    template of `log10_f_ref` (`(m, 8)` float32) at once, `r = log10_f_obs -
    log10_f_ref`, with the extinction law of section 2 solved
    self-consistently first (owner's ruling 2026-09-10, row 13):

    - The design a source's whole library shares needs one blend weight,
      so the solve starts at the pure diffuse design (`a = 0`) and, each
      round, fits every template at that design, takes the MEDIAN of
      every template's own unconstrained `a_hat` (`(m,)` values collapsed
      to the source's own extinction, since no one hypothesis is
      privileged over the others), sets `w = law_dense_weight(that
      median)`, and rebuilds the design at `w` -- until the median moves
      by less than `LAW_ITER_TOL_MAG` between rounds or `LAW_ITER_MAX`
      rounds have run; `n_law_iter` is the round each source's own
      median first stopped moving (`LAW_ITER_MAX` if it never did).
      Never `AK_SESNA` or the sightline column (row 13; `classify.py`'s
      claim that `AK_SESNA` is unread by `fittp` is now true).
    - At the converged design, `chi2_min = r^T P r` and the UNCONSTRAINED
      `(Av_hat, SC_hat) = r^T M^T` at every source, `r`/`P`/`M` kept
      float64 (`P` is a projector built from cancelling O(weight)
      ~1e3-1e4 terms; a float32 `P` leaves `P @ X` at ~1e-4 instead of
      ~0, which fails identity (i) -- measured): `a_hat`, `log10_b_hat`
      float64; `chi2_min` float32. Both products are one batched matmul
      per source (`M` stored `(n, 8, 2)` so `matmul(r, M)` needs no
      per-call transpose) rather than a general three-index einsum -- a
      per-source BLAS gemm dispatch either way, but einsum's own generic
      loop was markedly slower (measured).
    - The CLAMPED marks, `Av` restricted to `[0, 75/(A_K/A_V)_s]`: because
      the gray column is one constant in every band, the least-squares
      residual is orthogonal to both design columns, so re-solving `SC` at
      fixed `Av` needs no per-band array at all --
      `SC_clamped = SC_hat + (Av_hat - Av_clamped) * S0 / (gray * W_sum)`,
      `S0 = sum_b W_b X_b0`. `FRAC_CLAMPED` is the fraction of `m`
      templates where the clamp engaged, per source.
    - The non-detection term of section 6.2 at the CLAMPED marks: the
      template's model flux `f_hat_i` in every undetected, limited band,
      compared to the source's own `F_LIM_50,i` through the region-band
      roll-off width `WIDTH_DEX`, `ln[1 - C_i(f_hat_i)]` via the scaled
      complementary error function (`_ln_one_minus_c`), gathered per
      source to only its own undetected bands (`_ln_nondet`), plus
      `batch.ln_norm_term` (section 6.1's `-1/2 Sum_i ln sigma_i^2`, one
      number per source): `ln_nondet` is exactly the one field
      `fittp.sweep` adds unscaled into `ln L_hat = -1/2 chi2_min +
      ln_nondet`, so both section 6.2's term and section 6.1's
      normalisation ride in it without any change to that formula.

    `batch.flagged` (fewer than two detected bands or a non-finite sigma,
    `prepare`'s own flags) is widened here by a singular design at the
    converged blend, and `batch.ext_col`/`sigma_a_ak`/`slope_sc_av`/
    `ak_per_av` are filled in at that same converged design for
    `fittp.sweep`'s prior-read conversion and flux reconstruction. Rows
    flagged are NaN throughout.
    """
    config = batch.config
    log10_f_ref = np.asarray(log10_f_ref, dtype=np.float32)
    log10_f_ref64 = log10_f_ref.astype(np.float64)
    r = batch.log10_f_obs[:, None, :] - log10_f_ref64[None, :, :]     # (n, m, 8) float64
    n = r.shape[0]
    weight = batch.weight
    diag = np.arange(N_BANDS)

    a_hat_source = np.zeros(n, dtype=np.float64)   # section 2: start at the diffuse design
    n_law_iter = np.full(n, LAW_ITER_MAX, dtype=np.int8)
    still_open = np.ones(n, dtype=bool)

    for it in range(1, LAW_ITER_MAX + 1):
        w_ramp = population_selection.law_dense_weight(a_hat_source)          # (n,)
        kappa_k = population_selection.kappa_hybrid(config, w_ramp)           # (n, 8)
        ak_per_av = population_selection.ak_per_av(config, w_ramp)            # (n,)
        ext_col = -0.4 * kappa_k * ak_per_av[:, None]                         # (n, 8)

        design = np.empty((n, N_BANDS, 2), dtype=np.float64)
        design[:, :, 0] = ext_col
        design[:, :, 1] = GRAY_COLUMN

        wx = design * weight[:, :, None]
        xtwx = np.einsum("nbk,nbl->nkl", wx, design)
        det_xtwx = np.linalg.det(xtwx)
        design_flagged = batch.flagged | (det_xtwx <= 0)
        xtwx_safe = xtwx.copy()
        xtwx_safe[design_flagged] = np.eye(2)
        xtwx_inv = np.linalg.inv(xtwx_safe)

        m_mat = np.einsum("nkl,nbl->nkb", xtwx_inv, wx)          # (XtWX)^-1 XtW, (n, 2, 8)
        marks = np.matmul(r, np.ascontiguousarray(m_mat.transpose(0, 2, 1)))
        av_hat = marks[:, :, 0]
        sc_hat = marks[:, :, 1]
        a_hat = av_hat * ak_per_av[:, None]      # (n, m)

        # every reader of law_dense_weight elsewhere in the package (gaia,
        # atlas, the prior) evaluates it at a physical, non-negative
        # extinction; the unconstrained a_hat can dip below zero on a
        # near-zero-extinction source's fit noise, which the ramp's own
        # log(a/LAW_RAMP_LO) has no value for -- floored at zero before
        # driving the next round's design, never before the reported marks.
        a_hat_med = np.median(np.maximum(a_hat, 0.0), axis=1)
        delta = np.abs(a_hat_med - a_hat_source)
        newly_converged = still_open & (delta < LAW_ITER_TOL_MAG)
        n_law_iter[newly_converged] = it
        still_open &= ~newly_converged
        a_hat_source = a_hat_med
        if not still_open.any():
            break

    p_mat = -np.einsum("nbk,nkc->nbc", wx, m_mat)
    p_mat[:, diag, diag] += weight

    # sigma_a^2 = [(XtWX)^-1]_aa (section 1.3), converted A_V -> A_K by the
    # same ratio; the conditional slope d SC / d A_V from the same matrix,
    # at the converged design.
    sigma_a_ak = np.sqrt(xtwx_inv[:, 0, 0]) * ak_per_av
    slope_sc_av = xtwx_inv[:, 1, 0] / xtwx_inv[:, 0, 0]

    # the clamp's re-solve needs only these two per-source scalars:
    # S0 = sum_b W_b X_b0 (the extinction column's weighted sum) and the
    # summed weight, because the gray column is the same constant in every
    # band (see the docstring above for the algebra).
    s0 = (weight * ext_col).sum(axis=1)
    w_sum = weight.sum(axis=1)

    batch.flagged = design_flagged
    batch.ext_col = ext_col.astype(np.float32)
    batch.s0, batch.w_sum = s0, w_sum
    batch.sigma_a_ak, batch.slope_sc_av, batch.ak_per_av = sigma_a_ak, slope_sc_av, ak_per_av

    rp = np.matmul(r, p_mat)
    chi2_min = (r * rp).sum(axis=2).astype(np.float32)
    log10_b_hat = -2.0 * sc_hat

    av_max = AV_CLAMP_MAX_AK / ak_per_av
    av_clamped = np.clip(av_hat, 0.0, av_max[:, None])
    clamp_engaged = av_hat != av_clamped
    resolve_ratio = s0 / (GRAY_COLUMN * w_sum)
    sc_clamped = sc_hat + (av_hat - av_clamped) * resolve_ratio[:, None]

    a_hat_clamped = (av_clamped * ak_per_av[:, None]).astype(np.float32)
    log10_b_hat_clamped = (-2.0 * sc_clamped).astype(np.float32)
    frac_clamped = clamp_engaged.mean(axis=1).astype(np.float32)

    # only the non-detection term's flux prediction needs float32 (its
    # own (n, m, 8) working set, section 6.2) -- cast once here, not
    # threaded back through the clamp above.
    av_clamped32 = av_clamped.astype(np.float32)
    sc_clamped32 = sc_clamped.astype(np.float32)
    ln_nondet = _ln_nondet(batch, batch.ext_col, log10_f_ref, av_clamped32, sc_clamped32)
    ln_nondet = ln_nondet + batch.ln_norm_term[:, None].astype(np.float32)

    chi2_min[batch.flagged] = np.nan
    a_hat[batch.flagged] = np.nan
    log10_b_hat[batch.flagged] = np.nan
    a_hat_clamped[batch.flagged] = np.nan
    log10_b_hat_clamped[batch.flagged] = np.nan
    ln_nondet[batch.flagged] = np.nan

    return Fit(chi2_min=chi2_min, a_hat=a_hat, log10_b_hat=log10_b_hat,
               a_hat_clamped=a_hat_clamped, log10_b_hat_clamped=log10_b_hat_clamped,
               frac_clamped=frac_clamped, ln_nondet=ln_nondet, n_law_iter=n_law_iter)
