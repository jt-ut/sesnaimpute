"""The closed-form SED fit (SPEC_BMSTP_DRAFT.md section 6.1) and the
non-detection term (section 6.2), for every template of a library against
ONE SOURCE at a time -- the fitter's own unit of work (PARALLEL brief): the
model axis (`m` templates) is the only array axis anywhere in this module;
nothing here knows about any other source. A library module: no `progress`
use, no runbook line (IMPLEMENTATION_BMSTP_DRAFT.md section 4 row 2.2).

`prepare` builds, once per source and the source's own class/library, the
per-band weight at variance `sigma_i^2 = sigma_log,i^2 + sigma_cal,i^2 +
sigma_lib,L^2` (section 6.1; `sigma_lib,L` read once by the caller from
`fittp.library_resolution`'s check product for the class named and handed
in as a plain number), a band SESNA marks detected but whose flux is <= 0
counted as unmeasured rather than a fabricated datum, and a non-finite
sigma on a detected band flagging the source rather than carrying a NaN
normalisation into every template's likelihood. `fit` then solves the
diffuse/dense extinction-law blend of section 2 self-consistently against
the fit's own extinction mark (never the sightline column or SESNA's
published `AK`, spec section 2's ruling): starting from the pure diffuse
design, it fits every template, reads the source's own extinction back
from the result (the median over every template's own unconstrained
a_hat), rebuilds the design at the blend that mark implies, and refits,
until the source's own mark stops moving by more than 0.01 mag or four
rounds have run -- a plain loop with a `break`, one design and one set of
convergence numbers per source, no mask arrays: those existed only to let
one block-wide loop serve many sources whose own rounds finished at
different times, and a whole block was fitted at once only because the
block existed (PARALLEL brief). Every template is then reduced to two
matrix products at that converged design for the UNCONSTRAINED optimum
the prior read integrates over, the CLAMPED marks for the reported record
and the flux prediction, and the non-detection term (plus section 6.1's
per-source normalisation) at those clamped marks.
"""

import math
from collections import namedtuple
from functools import lru_cache

import h5py
import numba
import numpy as np

from sesnaimpute import definitions
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
#: magnitudes, or after this many rounds, whichever comes first. A GUARD,
#: not a stopping rule the fit relies on for accuracy: the chi2 solve
#: itself is closed-form and always succeeds, this loop is a fixed point on
#: the extinction-law choice alone (PARALLEL brief).
LAW_ITER_TOL_MAG = 0.01
LAW_ITER_MAX = 4

_SQRT2 = np.float32(np.sqrt(2.0))

#: The per-source working set this module holds at its own peak, in
#: float32-`(n_model, 8)` equivalents (PARALLEL brief item 8's memory
#: disclosure, `fittp.sweep.build_region_class`'s printed line): `r`
#: (float64, held across the chi2 and marks products, 2 equivalents) plus
#: `r @ p_mat` (float64, transient during the chi2 product, 2) or, once
#: freed, the non-detection term's `log10_fhat`/`z` (float32, 1 each), the
#: numba kernel's own float64 output (2, for the float64 precision the
#: z=0 identity needs) and its masked copy (2) -- 8 in all at the
#: non-detection term's own peak (`r` still live, `log10_fhat` + `z` +
#: kernel output + masked copy). Ten in this table's own count: the
#: register `log10_f_ref` (`(m, 8)` float32, shared read-only across every
#: worker by fork, not a per-worker allocation) is excluded, matching
#: `fittp.sweep`'s own disclosure formula (`n_model * 8 * 4 *
#: WORKER_WORKING_SET_EQUIV` bytes per worker, about 64 MB at YSO's
#: 200,000 templates with one worker's own r/rp/nondet buffers live at
#: once).
WORKER_WORKING_SET_EQUIV = 10


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
    `z_flat` is float32 (this module's own working dtype); each element is
    cast to float64 here, per scalar, so the z=0 identity keeps double
    precision without a separate float64 copy of the whole array. `prange`
    parallelises over templates within one source's own call; the fitter's
    parallel axis is now the source (a multiprocessing pool over sources,
    `fittp.sweep`), so each worker pins `numba.set_num_threads(1)` before
    its first task and this loop runs single-threaded there -- oversub-
    scription, not this kernel, decides whether it threads at all.
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
    with no float32 rounding in the way. `z` keeps its own dtype (float32)
    into the kernel -- only the kernel's per-element arithmetic and its
    output are float64, so no full-array float64 copy of `z` is made here.
    """
    z = np.asarray(z)
    shape = z.shape
    z_flat = np.ascontiguousarray(z.reshape(-1))
    out = np.empty(z_flat.shape, dtype=np.float64)
    _ln_one_minus_c_kernel(z_flat, out)
    return out.reshape(shape)


def _ln_nondet(ext_col32, log10_f_ref, av_clamped32, sc_clamped32,
               log10_f_lim50, width_dex, nondet_mask):
    """The non-detection term of section 6.2 at the CLAMPED marks, for one
    source's own `m` templates: the predicted log-flux in every band
    (float32) for every template, masked to this source's own undetected,
    limited bands and summed. `ext_col32` is the extinction-law solve's own
    converged design column for this source, `(8,)` float32 (section 2).
    One batched call into `_ln_one_minus_c`'s numba kernel beats a
    per-template Python-level call (measured on the block form this
    replaces: ~2.5x slower -- the per-call overhead dominates the kernel's
    own, embarrassingly parallel, per-element cost).
    """
    log10_fhat = (log10_f_ref
                  + ext_col32[None, :] * av_clamped32[:, None]
                  + np.float32(GRAY_COLUMN) * sc_clamped32[:, None])       # (m, 8)
    z = (log10_fhat - log10_f_lim50[None, :]) / (_SQRT2 * width_dex[None, :])
    term = _ln_one_minus_c(z)
    term = np.where(nondet_mask[None, :], term, 0.0)
    return term.sum(axis=1).astype(np.float32)


class Batch:
    """One source's own quantities the closed form and the non-detection
    term need: `log10_f_obs` (`(8,)` float64, section 6.1) and its weight,
    `flagged` (fewer than two valid detections, or a non-finite sigma on a
    detected band, section 6.1/16), the per-band `F_LIM_50` and roll-off
    width `WIDTH_DEX` (`(8,)`, section 6.2), `ln_norm_term` (section 6.1's
    per-source Gaussian normalisation, over only the bands with a finite
    variance), and `config` (the extinction-law solve's own need, `fit`'s
    docstring). `fit` fills in, once its self-consistent solve has
    converged, the design's own extinction column (`ext_col`, `(8,)`) and
    the source's `sigma_a`, conditional slope and (A_K/A_V) at that design
    (section 1.3) -- one number each, since every template of a source
    shares the one design the solve converges to.
    """

    __slots__ = ("log10_f_obs", "weight", "config",
                 "log10_f_lim50", "width_dex", "nondet_mask",
                 "n_detected", "flagged", "ln_norm_term",
                 "ext_col", "s0", "w_sum", "sigma_a_ak", "slope_sc_av", "ak_per_av")

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


Fit = namedtuple("Fit", (
    "chi2_min",              # (m,) f4 -- at the unconstrained optimum
    "a_hat",                 # (m,) f8 -- unconstrained, for the prior read's identity
    "log10_b_hat",           # (m,) f8 -- unconstrained
    "a_hat_clamped",         # (m,) f4 -- for the reported record and flux prediction
    "log10_b_hat_clamped",   # (m,) f4
    "frac_clamped",          # f4    -- FRAC_CLAMPED, fraction of templates clamped
    "ln_nondet",              # (m,) f4 -- section 6.2 (clamped marks) plus
                              # section 6.1's per-source ln_norm_term, the one
                              # sum `fittp.sweep` already adds unscaled into ln L_hat
    "n_law_iter",             # int -- N_LAW_ITER, rounds the law solve ran
))


@lru_cache(maxsize=None)
def sigma_lib_by_class(path):
    """`{class code: sigma_lib,L dex}`, section 6.1's one number per LIBRARY,
    from `fittp.library_resolution`'s check product (its `LIBRARY`/
    `SIGMA_LIB_DEX` datasets). Called once by `fittp.sweep` in the parent,
    before the pool is created, and handed into `prepare` as a plain
    number -- not re-opened per source or per worker (rule 9); `lru_cache`
    on the product's own path keeps a second class's own read of the same
    tiny (six-number) file from repeating the open.
    """
    with h5py.File(path, "r") as f:
        libs = np.char.decode(f["LIBRARY"][:].astype("S"), "utf-8")
        sigma = f["SIGMA_LIB_DEX"][:]
    return dict(zip(libs.tolist(), sigma.tolist()))


def prepare(config, flux, sigma, origin, sigma_lib_l, f_lim50, width_dex):
    """One source's own fit inputs (SPEC_BMSTP_DRAFT.md section 6.1): its
    fluxes, uncertainties and `ORIGIN_FNU` (the same read `fittp.cascade`
    uses, each `(8,)`), this class's one `sigma_lib,L` number
    (`sigma_lib_by_class`, read once in the parent) and this source's own
    `F_LIM_50` and region roll-off width `WIDTH_DEX` (`(8,)`, sliced once
    from the region-wide arrays the caller loaded, never read from disk
    here). Builds the per-band weight and the fit's other
    design-independent numbers. The extinction-law design itself is not
    built here: it depends on the fit's own extinction mark (section 2), so
    `fit` builds and converges it once the per-template residual exists.
    """
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

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
    sigma2 = sigma_log ** 2 + SIGMA_CAL_DEX ** 2 + sigma_lib_l ** 2
    weight = np.where(detected & (sigma_log > 0), 1.0 / sigma2, 0.0)

    # (16, R2 D3): a non-finite sigma on a detected band already carries
    # zero weight (sigma_log > 0 is False for NaN), but sigma2 itself is
    # still NaN; summing it into ln_norm_term regardless silently zeroed
    # the source's whole posterior. Excluded from the sum instead, and the
    # source flagged (sweep.py then records and excludes it, exactly as a
    # too-few-bands source already is).
    finite_sigma2 = np.isfinite(sigma2)
    flagged_sigma = bool((detected & ~finite_sigma2).any())
    ln_norm_term = float(-0.5 * np.where(detected & finite_sigma2, np.log(sigma2), 0.0).sum())

    n_detected = int(detected.sum())
    flagged = (n_detected < MIN_DETECTED_BANDS) or flagged_sigma

    f_lim50 = np.asarray(f_lim50, dtype=np.float64)
    log10_f_lim50 = np.log10(f_lim50)
    width = np.asarray(width_dex, dtype=np.float64)
    # section 6.2: a band with no measurement and no limit is excluded;
    # SESNA's curated substitution always supplies one or the other, so
    # this only guards a non-finite F_LIM_50. Row 15's unmeasured
    # detections fall in here too, since `detected` already excludes them.
    nondet_mask = (~detected) & np.isfinite(f_lim50)

    return Batch(log10_f_obs=log10_f_obs, weight=weight, config=config,
                 log10_f_lim50=log10_f_lim50.astype(np.float32),
                 width_dex=width.astype(np.float32), nondet_mask=nondet_mask,
                 n_detected=n_detected, flagged=flagged,
                 ln_norm_term=ln_norm_term)


def fit(batch, log10_f_ref):
    """The closed-form fit of SPEC_BMSTP_DRAFT.md section 6.1 over every
    template of `log10_f_ref` (`(m, 8)` float32, this class's shared
    register) at this ONE source, `r = log10_f_obs - log10_f_ref`, with the
    extinction law of section 2 solved self-consistently first (owner's
    ruling 2026-09-10, row 13):

    - The design this source's whole library shares needs one blend
      weight, so the solve starts at the pure diffuse design (`a = 0`)
      and, each round, fits every template at that design, takes the
      MEDIAN of every template's own unconstrained `a_hat` (`(m,)` values
      collapsed to this source's own extinction, since no one hypothesis
      is privileged over the others), sets `w = law_dense_weight(that
      median)`, and rebuilds the design at `w` -- a plain loop with a
      `break` once the median moves by less than `LAW_ITER_TOL_MAG`
      between rounds, or after `LAW_ITER_MAX` rounds, whichever comes
      first; `n_law_iter` is the round this source's own median stopped
      moving (`LAW_ITER_MAX` if it never did). No per-source mask array:
      one source, one design, one convergence flag (PARALLEL brief -- the
      block form's `still_open`/`newly_converged` bookkeeping existed only
      to let many sources' rounds, finishing at different times, share one
      loop). Never `AK_SESNA` or the sightline column (row 13;
      `classify.py`'s claim that `AK_SESNA` is unread by `fittp` is now
      true).
    - At the converged design, `chi2_min = r^T P r` and the UNCONSTRAINED
      `(Av_hat, SC_hat) = r M^T` at every template, `r`/`P`/`M` kept
      float64 (`P` is a projector built from cancelling O(weight)
      ~1e3-1e4 terms; a float32 `P` leaves `P @ X` at ~1e-4 instead of
      ~0, which fails identity (i) -- measured): `a_hat`, `log10_b_hat`
      float64; `chi2_min` float32. The design solve is one plain 2x2
      matrix inverse per source (`xtwx`, `(2, 2)`), and the marks are one
      `(m, 8) @ (8, 2)` matmul -- no per-call transpose, `M` stored
      `(2, 8)` so `r @ M.T` needs none either.
    - The CLAMPED marks, `Av` restricted to `[0, 75/(A_K/A_V)_s]`: because
      the gray column is one constant in every band, the least-squares
      residual is orthogonal to both design columns, so re-solving `SC` at
      fixed `Av` needs no per-band array at all --
      `SC_clamped = SC_hat + (Av_hat - Av_clamped) * S0 / (gray * W_sum)`,
      `S0 = sum_b W_b X_b0`. `FRAC_CLAMPED` is the fraction of `m`
      templates where the clamp engaged.
    - The non-detection term of section 6.2 at the CLAMPED marks: the
      template's model flux `f_hat_i` in every undetected, limited band,
      compared to this source's own `F_LIM_50,i` through the region-band
      roll-off width `WIDTH_DEX`, `ln[1 - C_i(f_hat_i)]` via the scaled
      complementary error function (`_ln_one_minus_c`), gathered to only
      this source's own undetected bands (`_ln_nondet`), plus
      `batch.ln_norm_term` (section 6.1's `-1/2 Sum_i ln sigma_i^2`, one
      number): `ln_nondet` is exactly the one field `fittp.sweep` adds
      unscaled into `ln L_hat = -1/2 chi2_min + ln_nondet`, so both
      section 6.2's term and section 6.1's normalisation ride in it
      without any change to that formula.

    `batch.flagged` (fewer than two detected bands or a non-finite sigma,
    `prepare`'s own flags) is widened here by a singular design at the
    converged blend, and `batch.ext_col`/`sigma_a_ak`/`slope_sc_av`/
    `ak_per_av` are filled in at that same converged design for
    `fittp.sweep`'s prior-read conversion and flux reconstruction. Every
    output is NaN throughout if the source ends up flagged.
    """
    config = batch.config
    log10_f_ref = np.asarray(log10_f_ref, dtype=np.float32)
    log10_f_ref64 = log10_f_ref.astype(np.float64)
    r = batch.log10_f_obs[None, :] - log10_f_ref64     # (m, 8) float64
    m = r.shape[0]
    weight = batch.weight                              # (8,) float64
    diag = np.arange(N_BANDS)

    a_source = 0.0     # section 2: start at the diffuse design
    n_law_iter = LAW_ITER_MAX
    source_flagged = bool(batch.flagged)

    for it in range(1, LAW_ITER_MAX + 1):
        w_ramp = population_selection.law_dense_weight(a_source)
        kappa_k = population_selection.kappa_hybrid(config, w_ramp)          # (8,)
        ak_per_av = float(population_selection.ak_per_av(config, w_ramp))
        ext_col = -0.4 * kappa_k * ak_per_av                                 # (8,)

        design = np.empty((N_BANDS, 2), dtype=np.float64)
        design[:, 0] = ext_col
        design[:, 1] = GRAY_COLUMN

        wx = design * weight[:, None]                    # (8, 2)
        xtwx = wx.T @ design                              # (2, 2)
        det_xtwx = xtwx[0, 0] * xtwx[1, 1] - xtwx[0, 1] * xtwx[1, 0]
        source_flagged = bool(batch.flagged) or det_xtwx <= 0.0
        xtwx_safe = xtwx if not source_flagged else np.eye(2)
        xtwx_inv = np.linalg.inv(xtwx_safe)

        m_mat = xtwx_inv @ wx.T                           # (2, 8), (XtWX)^-1 XtW
        marks = r @ m_mat.T                               # (m, 2)
        av_hat = marks[:, 0]
        sc_hat = marks[:, 1]
        a_hat = av_hat * ak_per_av                        # (m,)

        # every reader of law_dense_weight elsewhere in the package (gaia,
        # atlas, the prior) evaluates it at a physical, non-negative
        # extinction; the unconstrained a_hat can dip below zero on a
        # near-zero-extinction source's fit noise, which the ramp's own
        # log(a/LAW_RAMP_LO) has no value for -- floored at zero before
        # driving the next round's design, never before the reported marks.
        a_med = float(np.median(np.maximum(a_hat, 0.0)))
        converged = abs(a_med - a_source) < LAW_ITER_TOL_MAG
        a_source = a_med
        n_law_iter = it
        if converged:
            break

    p_mat = -(wx @ m_mat)                                 # (8, 8)
    p_mat[diag, diag] += weight

    # sigma_a^2 = [(XtWX)^-1]_aa (section 1.3), converted A_V -> A_K by the
    # same ratio; the conditional slope d SC / d A_V from the same matrix,
    # at the converged design.
    sigma_a_ak = float(np.sqrt(xtwx_inv[0, 0]) * ak_per_av)
    slope_sc_av = float(xtwx_inv[1, 0] / xtwx_inv[0, 0])

    # the clamp's re-solve needs only these two scalars: S0 = sum_b W_b
    # X_b0 (the extinction column's weighted sum) and the summed weight,
    # because the gray column is the same constant in every band (see the
    # docstring above for the algebra).
    s0 = float((weight * ext_col).sum())
    w_sum = float(weight.sum())

    batch.flagged = source_flagged
    batch.ext_col = ext_col.astype(np.float32)
    batch.s0, batch.w_sum = s0, w_sum
    batch.sigma_a_ak, batch.slope_sc_av, batch.ak_per_av = sigma_a_ak, slope_sc_av, ak_per_av

    rp = r @ p_mat
    chi2_min = (r * rp).sum(axis=1).astype(np.float32)
    log10_b_hat = -2.0 * sc_hat

    av_max = AV_CLAMP_MAX_AK / ak_per_av
    av_clamped = np.clip(av_hat, 0.0, av_max)
    clamp_engaged = av_hat != av_clamped
    resolve_ratio = s0 / (GRAY_COLUMN * w_sum)
    sc_clamped = sc_hat + (av_hat - av_clamped) * resolve_ratio

    a_hat_clamped = (av_clamped * ak_per_av).astype(np.float32)
    log10_b_hat_clamped = (-2.0 * sc_clamped).astype(np.float32)
    frac_clamped = np.float32(clamp_engaged.mean())

    # only the non-detection term's flux prediction needs float32 (its
    # own (m, 8) working set, section 6.2) -- cast once here, not
    # threaded back through the clamp above.
    av_clamped32 = av_clamped.astype(np.float32)
    sc_clamped32 = sc_clamped.astype(np.float32)
    ln_nondet = _ln_nondet(batch.ext_col, log10_f_ref, av_clamped32, sc_clamped32,
                            batch.log10_f_lim50, batch.width_dex, batch.nondet_mask)
    ln_nondet = ln_nondet + np.float32(batch.ln_norm_term)

    if batch.flagged:
        chi2_min = np.full(m, np.nan, dtype=np.float32)
        a_hat = np.full(m, np.nan, dtype=np.float64)
        log10_b_hat = np.full(m, np.nan, dtype=np.float64)
        a_hat_clamped = np.full(m, np.nan, dtype=np.float32)
        log10_b_hat_clamped = np.full(m, np.nan, dtype=np.float32)
        ln_nondet = np.full(m, np.nan, dtype=np.float32)

    return Fit(chi2_min=chi2_min, a_hat=a_hat, log10_b_hat=log10_b_hat,
               a_hat_clamped=a_hat_clamped, log10_b_hat_clamped=log10_b_hat_clamped,
               frac_clamped=frac_clamped, ln_nondet=ln_nondet, n_law_iter=n_law_iter)
