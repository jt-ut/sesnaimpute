"""`fit_batch`: one region, one class (one library register), one batch
of catalogue source rows -- the chi-squared fit over every model of the
register, folded with the library's sampling weight and the census prior
into a per-subclass evidence, a predictive flux moment, and a top-K
record (`10_POSTERIOR.md` section 1's `L_hat`, `w_h`, `lambda~_C`, read
once per model at that model's own best-fit `(a, log10 B)` -- the
"struck paragraph" quadrature is NOT adopted, see that page).

Lifted, not redesigned, from two owner-tuned quarry modules:

- `sesnacomplete.sed_fit.batched.fit_models_batched`: the closed-form
  chi-squared profiled over extinction and a free gray scale
  (`sedfitter.fitting_routines.linear_regression`/`optimal_scaling`/
  `chi_squared`), the per-source hybrid extinction law blended by the
  ramp weight, the A_V clamp converted from the project's A_K bound at
  that ramp weight, and the zero-weight guard (a legitimate zero model
  flux must not poison a band the fit already excluded).
- `sesnacomplete.sed_fit.streaming.SourceAccumulator`: the log-sum-exp
  evidence per subclass, the predictive flux mean/covariance under the
  full posterior weight, and the top-K kept models.

ONE DELIBERATE SIMPLIFICATION FROM THE QUARRY, forced by the new
register format: `fit_models_batched` has two branches, FREE_SCALE and
APERTURE_DEPENDENT (a fit over a trial-distance grid, argmin'd away).
This project's registers carry no distance axis -- library curation
already resolved the aperture dependence into one `F_REF_<band>` per
model at one reference distance, 1 kpc (see a register's own
`APERTURE_CONVENTION` root attribute) -- so every class fits on the
FREE_SCALE branch only; the aperture-dependent branch is not lifted.
This matches `10_POSTERIOR.md` section 1: `log10 B = -2*SC` is the
fitter's own gray scale, not a physical distance.

A SECOND SIMPLIFICATION: the quarry batches MODELS (`DEFAULT_MODEL_
BATCH_SIZE = 10000`) to bound an `(n_model, n_band)` temporary under
`sedfitter`'s `astropy.units.Quantity` wrapper. Nothing here builds that
wrapper -- the fit algebra (`sedfitter.fitting_routines.linear_
regression`/`optimal_scaling`/`chi_squared`) is reimplemented in this
module as `_linear_regression_block`/`_optimal_scaling_block`/
`_chi_squared_block`, plain float64 arithmetic with one extra leading
source axis -- so the model axis is never batched within a source. The
evidence log-sum-exp therefore needs no running-max rescale either: it
is one `exp()` over the source's whole model axis, not folded across
successive model batches.

`fit_batch` itself works over BLOCKS of sources, not one source at a
time: every per-source quantity (the hybrid law, the fit, the prior
read, the evidence fold, the top-K) is one array expression over a
`(n_block, n_model[, n_band])` working set, `n_block` sized by
`sesnaimpute.batches.batches` to keep that set under `config.
fit_block_budget_mb` (owner ruling 2026-09-06, default 512 MB,
CODING_RULES.md 10a/10b) -- large for small registers (STAR), small for
large ones (YSO), but never a Python loop over sources. See the block-
size formula's own comment, below `fit_batch`'s `keep = min(topk,
n_model)` line, for the measured array census (`tracemalloc`) the
budget is actually bounding. The `gamma`/`psi` hooks are the one
exception: their signature takes one source at a time, so they are
called in a plain loop over each block's rows.

THE BLOCKS OF ONE BATCH RUN IN PARALLEL, ON THREADS. By the time
`fit_batch` is called, the caller has already called `prior.
prepare(rows)` once for this whole batch (`fit.run._fit_one`'s own
docstring) -- so every block's `prior.log_density` call below is a pure
read against that one already-built tabulation, never a write, and
every block writes into its own disjoint `evidence[blk]`/`flux_mean
[blk]`/... slice of the arrays this function preallocates. Two blocks
therefore share no mutable state, so they are dispatched across
`config.n_jobs` THREADS (`joblib.Parallel(prefer="threads")`), not
processes: threads share the one `prior`/`gamma`/`psi` objects and the
register arrays by reference, with no pickling and no per-worker
rebuild. This only speeds anything up because the block's own numpy
calls (the fit, the prior read, the evidence fold, all array-at-once)
release the GIL while they run; the one piece that does NOT -- the
`gamma`/`psi` hooks' plain Python loop over each block's rows -- still
runs under the GIL like any interpreted loop, so a block dominated by
many sources with few models (where that per-row loop, not the array
work, is the bottleneck) will not speed up in proportion to `n_jobs`.
"""

import os
import threading

import h5py
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute.prior import selection

# ---------------------------------------------------------------------
# constants (CODING_RULES.md rule 3: every number cited)
# ---------------------------------------------------------------------

#: `sedfitter.source.Source`'s own `valid` vocabulary (`sed_fit.hook`'s
#: module docstring; `Source.get_log_fluxes`): 1 = a real flux
#: measurement, 3 = an upper limit (a violation adds the step penalty
#: below), 0 = excluded (zero weight, no residual comparison at all).
#: This project's catalogue never carries sedfitter's other codes (2 =
#: lower limit, 4 = pre-logged flux, 9 = ignored-but-plotted).
VALID_DETECTION = 1
VALID_UPPER_LIMIT = 3
VALID_EXCLUDED = 0

#: `sesnaimpute.catalog.curated`'s own `ORIGIN_FNU` vocabulary (that
#: module's docstring): 1 = a real flux measurement; 2, 90, 91 = a
#: survey/completeness bound substituted for a non-detection (the 2MASS
#: global bound, the source's own DCOMP90, or a sky-neighbour's DCOMP90)
#: -- catalogued as an upper limit at that substituted flux, never a
#: detection. `SIGMA_FNU_MJY` at any of these three already carries
#: `catalog.curated.UPPER_LIMIT_SIGMA = 0.99`, the confidence `c` a
#: violated limit's step penalty `-2 ln(1 - c)` (`sedfitter.fitting_
#: routines.chi_squared`) needs -- read straight off the catalogue, per
#: source per band, never re-declared as a fitter-side constant.
ORIGIN_DETECTED = 1
ORIGIN_UPPER_LIMIT = (2, 90, 91)

#: The project-wide A_V clamp, in A_K magnitudes (transcribed from
#: `sesnacomplete.constants.AK_MIN`/`AK_MAX`, not yet ported to
#: `sesnaimpute.constants`): 0 because negative extinction is
#: impossible; 75 because the largest adopted column measured anywhere
#: in the survey is 47.131, with room to spare. A physics guard on the
#: fit's own A_V solve, converted to that source's own ramp-blended A_V
#: at the point of use (`av_range_for_law_weight`'s lift, below) --
#: never the census prior's own plausible range.
AK_MIN = 0.0
AK_MAX = 75.0

#: `w_h ~ RHO_KDE1^(-alpha)`, normalised within one register file
#: (`sesnacomplete.sed_models_register.io.weights`; Q80 / `sesnacomplete.
#: sed_fit.fit._W_H_ALPHA_RHO`): alpha=1 is the ratified production
#: value, safe against a duplicated-model blowup because the KDE
#: bandwidth already bounds any one model's outlier weight. Not a
#: per-call knob.
ALPHA_RHO = 1.0

#: Kept models per source, ranked by full posterior weight
#: (`sesnacomplete.sed_fit.streaming.DEFAULT_NKEEP`, the documented
#: production default) -- a fallback only; `fit_batch` reads the live
#: value off `config.fit_topk` (owner ruling 2026-09-06, `config.py`'s
#: `[fit] topk`).
TOPK = 5

#: class code -> the library register key: `definitions.CLASS_REGISTER`,
#: shared with `fit.terms` so both modules read one class's register
#: under the same key.
CLASS_REGISTER = definitions.CLASS_REGISTER

_BAND_KEYS = tuple(b.key for b in definitions.BANDS)
_N_BAND = len(_BAND_KEYS)

#: `sedfitter`'s free-scale design vector for the gray-scale nuisance:
#: a constant -2 in every band, exactly (`sed_fit.fit`'s own docstring,
#: Q1) -- `10_POSTERIOR.md` section 1's `log10 B = -2*SC`, `SC` the
#: fitted coefficient on this vector.
_SC_LAW = np.full(_N_BAND, -2.0, dtype=np.float64)


def block_size(n_model, block_budget_mb):
    """Sources per block a `block_budget_mb`-MB budget holds for a register
    of `n_model` models -- the same row-bytes formula `fit_batch` bounds
    its own blocks by (see that function's own MEMORY DRIVER comment,
    below `keep = min(topk, n_model)`), exposed here so `fit.run` can
    print the number before a job's first batch is even prepared."""
    row_bytes = n_model * _N_BAND * 8 * 16
    return max(1, (block_budget_mb << 20) // max(1, row_bytes))


def _register_arrays(config, cls):
    """Everything `fit_batch` needs off one class's register, read once
    per call: the unit-state template flux (`FLOOR_LINEAR`-floored, per
    the register's own `FREFRAW` convention), the library-sampling
    weight `ln w_h`, the model->subclass index, and the two pure
    extinction laws' design vectors, reordered to `definitions.BANDS`
    order off the register's own `bands/BAND` column (never assumed
    positional).
    """
    reg_path = os.path.join(config.inputs["sed_models"], "registers",
                             f"{CLASS_REGISTER[cls]}_register.hdf5")
    if not os.path.isfile(reg_path):
        raise ValueError(
            f"fit.sweep: no register for class {cls!r} at {reg_path!r} -- "
            f"run the sed_models_register RUNBOOK line that builds it")
    with h5py.File(reg_path, "r") as f:
        n_model = f["models"]["MODEL_NAME"].shape[0]
        f_ref = np.empty((n_model, _N_BAND), dtype=np.float64)
        for j, key in enumerate(_BAND_KEYS):
            f_ref[:, j] = np.asarray(f["models"][f"F_REF_{key}"][:], dtype=np.float64)
        floor_linear = np.asarray(f["models"]["FLOOR_LINEAR"][:], dtype=np.float64)
        rho_kde1 = np.asarray(f["models"]["RHO_KDE1"][:], dtype=np.float64)
        subclass_raw = f["models"]["SUBCLASS"][:]

        band_order = [b.decode() if isinstance(b, bytes) else b for b in f["bands"]["BAND"][:]]
        reorder = [band_order.index(key) for key in _BAND_KEYS]
        av_law_draine = np.asarray(f["bands"]["AV_LAW_DRAINE"][:], dtype=np.float64)[reorder]
        av_law_whitney = np.asarray(f["bands"]["AV_LAW_WHITNEY"][:], dtype=np.float64)[reorder]

    subclass_names = np.array(
        [s.decode() if isinstance(s, bytes) else s for s in subclass_raw])
    subclass_order = definitions.SUBCLASSES_OF[cls]
    sub_to_idx = {s: i for i, s in enumerate(subclass_order)}
    unknown = set(subclass_names.tolist()) - set(sub_to_idx)
    if unknown:
        raise ValueError(
            f"fit.sweep: register {reg_path!r} carries SUBCLASS value(s) "
            f"{sorted(unknown)!r} not in definitions.SUBCLASSES_OF[{cls!r}] "
            f"= {subclass_order!r}")
    subclass_idx = np.array([sub_to_idx[s] for s in subclass_names], dtype=np.intp)

    # FREFRAW convention (sed_models_register.io): F_REF is raw, floor
    # before any log -- a legitimate zero model flux (a cold envelope
    # genuinely emits nothing at J) must not become -inf here.
    template_log = np.log10(np.maximum(f_ref, floor_linear[:, None]))

    weight_rho = rho_kde1 ** (-ALPHA_RHO)
    ln_w_h = np.log(weight_rho) - np.log(np.sum(weight_rho))

    return {
        "n_model": n_model, "template_log": template_log, "ln_w_h": ln_w_h,
        "subclass_idx": subclass_idx, "n_sub": len(subclass_order),
        "av_law_draine": av_law_draine, "av_law_whitney": av_law_whitney,
    }


def _catalog_bands(config, region, rows):
    """`(fnu, sigma_fnu, origin)`, each `(n_source, 8)`, `definitions.
    BANDS` order -- the curated catalogue's own on-disk order, which
    `sesnaimpute.catalog.curated.build` already writes in that order.
    """
    path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        curated_bands = [b.decode() if isinstance(b, bytes) else b for b in f.attrs["BANDS"]]
        if list(curated_bands) != list(_BAND_KEYS):
            raise ValueError(
                f"fit.sweep: {path!r}'s own band order {curated_bands!r} "
                f"does not match definitions.BANDS order {_BAND_KEYS!r}")
        fnu = np.asarray(f["FNU_MJY"][:], dtype=np.float64)[rows]
        sigma_fnu = np.asarray(f["SIGMA_FNU_MJY"][:], dtype=np.float64)[rows]
        origin = np.asarray(f["ORIGIN_FNU"][:])[rows]
    return fnu, sigma_fnu, origin


def _hybrid_av_law_block(config, w, av_law_draine, av_law_whitney):
    """`hybrid_av_law` and `av_range_for_law_weight`'s `ak_per_av`, both
    evaluated for a whole block of sources' ramp weights `w` `(n_block,)`
    at once -- `selection.ak_per_av` already vectorises over `w`, so the
    round-trip blend is one array expression; the `w<=0`/`w>=1` endpoints
    are restored bit-exact by `np.where`, same as the scalar function.
    Returns `(av_law, ak_per_av_w)`, `(n_block, n_band)` and `(n_block,)`.
    """
    w = np.asarray(w, dtype=np.float64)
    r_diffuse = selection.ak_per_av(config, 0.0)
    r_dense = selection.ak_per_av(config, 1.0)
    kappa_diffuse = av_law_draine / (-0.4 * r_diffuse)
    kappa_dense = av_law_whitney / (-0.4 * r_dense)
    kappa_s = (1.0 - w)[:, None] * kappa_diffuse[None, :] + w[:, None] * kappa_dense[None, :]
    ak_per_av_w = np.asarray(selection.ak_per_av(config, w), dtype=np.float64)
    av_law = -0.4 * ak_per_av_w[:, None] * kappa_s
    av_law = np.where((w <= 0.0)[:, None], av_law_draine[None, :], av_law)
    av_law = np.where((w >= 1.0)[:, None], av_law_whitney[None, :], av_law)
    return av_law, ak_per_av_w


def _source_log_fluxes_block(flux, sigma, origin):
    """`(valid, weight, log_flux, log_error)`, each `(n_block, 8)` --
    `_source_log_fluxes`'s algebra applied to a whole block at once
    (purely elementwise, so the block axis changes nothing but shape).
    """
    valid = np.where(np.isin(origin, ORIGIN_UPPER_LIMIT), VALID_UPPER_LIMIT,
                     np.where(origin == ORIGIN_DETECTED, VALID_DETECTION, VALID_EXCLUDED))
    weight = np.zeros_like(flux)
    log_flux = np.zeros_like(flux)
    log_error = np.zeros_like(flux)

    det = valid == VALID_DETECTION
    log_flux[det] = np.log10(flux[det]) - 0.5 * (sigma[det] / flux[det]) ** 2 / np.log(10.0)
    log_error[det] = np.abs(sigma[det] / flux[det]) / np.log(10.0)
    weight[det] = 1.0 / log_error[det] ** 2

    lim = valid == VALID_UPPER_LIMIT
    log_flux[lim] = np.log10(flux[lim])
    log_error[lim] = sigma[lim]

    return valid, weight, log_flux, log_error


def _linear_regression_block(data, weights, pattern1, pattern2):
    """`sedfitter.fitting_routines.linear_regression`, one extra leading
    source axis: `data` is `(n_block, n_model, n_band)`, `weights`/
    `pattern1` are `(n_block, n_band)` (the per-source hybrid law and its
    fit weights), `pattern2` is `(n_band,)` (the fixed gray-scale law,
    the same design vector for every source). Each source's `(m11, m12,
    m22)` normal-equation entries are independent of every other
    source's -- the block axis is a pure broadcast, not a new sum -- so
    this is bit-identical to calling the scalar routine once per source.
    Returns `(av_hat, sc_hat)`, each `(n_block, n_model)`.
    """
    c1 = np.sum(data * pattern1[:, None, :] * weights[:, None, :], axis=2)
    c2 = np.sum(data * pattern2[None, None, :] * weights[:, None, :], axis=2)
    m11 = np.sum(pattern1 * pattern1 * weights, axis=1)
    m12 = np.sum(pattern1 * pattern2[None, :] * weights, axis=1)
    m22 = np.sum(pattern2[None, :] * pattern2[None, :] * weights, axis=1)
    inv_det = 1.0 / (m11 * m22 - m12 * m12)
    p1 = (m22[:, None] * c1 - m12[:, None] * c2) * inv_det[:, None]
    p2 = (m11[:, None] * c2 - m12[:, None] * c1) * inv_det[:, None]
    return p1, p2


def _optimal_scaling_block(data, weights, pattern1):
    """`sedfitter.fitting_routines.optimal_scaling`, one extra leading
    source axis: `data` is `(n_block, n_model, n_band)`, `weights` is
    `(n_block, n_band)`, `pattern1` is `(n_band,)`. Returns `(n_block,
    n_model)`.
    """
    num = np.sum(data * pattern1[None, None, :] * weights[:, None, :], axis=2)
    denom = np.sum(pattern1[None, :] * pattern1[None, :] * weights, axis=1)
    return num / denom[:, None]


def _chi_squared_block(valid, data, error, weight, model):
    """`sedfitter.fitting_routines.chi_squared`, one extra leading source
    axis: `valid`/`error`/`weight` are `(n_block, n_band)`, `data`/
    `model` are `(n_block, n_model, n_band)`. The `valid == 2` (lower
    limit) branch is omitted -- this catalogue never carries that code
    (module docstring at `VALID_UPPER_LIMIT`) -- everything else is the
    same elementwise algebra, per source, broadcast over the block.
    Returns `(n_block, n_model)`.
    """
    chi2_array = (data - model) ** 2 * weight[:, None, :]

    zero = np.broadcast_to((valid == VALID_EXCLUDED)[:, None, :], chi2_array.shape)
    chi2_array[zero] = 0.0

    lim = (valid == VALID_UPPER_LIMIT)[:, None, :]
    reset = lim & (model > data)
    penalty = -2.0 * np.log(1.0 - error)
    chi2_array = np.where(reset, penalty[:, None, :], chi2_array)

    chi2_array[np.isinf(chi2_array)] = 1.0e30
    return np.sum(chi2_array, axis=2)


def _fit_model_grid_block(valid, weight, log_flux, log_error, template_log,
                          av_law, av_min, av_max):
    """`_fit_one_model_grid`, one extra leading source axis: the
    closed-form free-scale fit for a whole block of sources against the
    class's whole model register at once. `av_min`/`av_max` are
    `(n_block,)`. Returns `(av_hat, sc_hat, chi2, model)`, `av_hat`/
    `sc_hat`/`chi2` `(n_block, n_model)`, `model` `(n_block, n_model,
    n_band)`.
    """
    residual = log_flux[:, None, :] - template_log[None, :, :]
    residual = np.array(residual)  # writable; broadcast subtraction already copies

    zero_weight = weight == 0.0
    bad = None
    if zero_weight.any():
        bad = zero_weight[:, None, :] & ~np.isfinite(residual)
        if not bad.any():
            bad = None
    sign = None
    if bad is not None:
        sign = np.sign(residual[bad])
        sign[np.isnan(sign)] = 0.0
        residual[bad] = 0.0

    av_hat, sc_hat = _linear_regression_block(residual, weight, av_law, _SC_LAW)

    reset_lo = av_hat < av_min[:, None]
    reset_hi = av_hat > av_max[:, None]
    av_hat = np.where(reset_lo, av_min[:, None], np.where(reset_hi, av_max[:, None], av_hat))
    reset = reset_lo | reset_hi
    if reset.any():
        sc_hat = sc_hat.copy()
        sc_hat_full = _optimal_scaling_block(
            residual - av_hat[:, :, None] * av_law[:, None, :], weight, _SC_LAW)
        sc_hat[reset] = sc_hat_full[reset]

    model = av_hat[:, :, None] * av_law[:, None, :] + sc_hat[:, :, None] * _SC_LAW[None, None, :]

    if bad is not None:
        residual[bad] = model[bad] + sign

    chi2 = _chi_squared_block(valid, residual, log_error, weight, model)
    return av_hat, sc_hat, chi2, model


def fit_batch(config, region, cls, rows, prior, gamma=None, psi=None, stage=None):
    """One region, one class, one batch of catalogue rows: the full
    model-grid fit and posterior fold for each source in `rows`.

    `prior` is a `sesnaimpute.prior.callable.SourcePrior` for `region`.
    Its `prepare(rows)` must already have been called by the caller --
    once per batch, shared across every class fit against that same
    batch (`SourcePrior`'s own contract; the tabulation GAL/YSO/H2S need
    is class-independent).

    `gamma`/`psi` are the Gaia-congruence and colour-cascade hooks
    (`10_POSTERIOR.md` section 1's `Gamma`, `Psi`), left as arguments
    defaulting to zero (a parallel unit's own scope): each, if given, is
    called as `f(row, model_index, a, log10_b) -> ln value, (n_model,)`.

    `stage` is the caller's own `sesnaimpute.progress.Stage` for this
    whole class job (owner ruling 2026-09-06): when given, one block's
    completion calls `stage.tick(done, n_blocks, "blocks")` -- the
    helper's own ten-second throttle means this only actually prints
    when one batch's blocks are running long, sharing the same
    throttle clock as the caller's own per-batch tick.

    Returns a dict of plain numpy arrays, row-aligned to `rows`:
    `EVIDENCE` `(n_source, n_sub)` (natural-log, this class's own
    subclasses in `definitions.SUBCLASSES_OF[cls]` order), `FLUX_MEAN`
    `(n_source, 8)`, `FLUX_COV` `(n_source, 8, 8)`, `TOPK_MODEL`/
    `TOPK_A`/`TOPK_LOG10B`/`TOPK_LNL`/`TOPK_CHI2` `(n_source, topk)`,
    `TOPK_FLUX` `(n_source, topk, 8)` (each kept model's own fitted --
    scaled, reddened -- flux in `definitions.BANDS` order, mJy),
    `N_DETECTED` `(n_source,)`. `topk` is `config.fit_topk` (owner ruling
    2026-09-06).
    """
    if cls not in CLASS_REGISTER:
        raise ValueError(f"fit.sweep.fit_batch: unknown class {cls!r}, "
                         f"must be one of {sorted(CLASS_REGISTER)}")
    rows = np.asarray(rows, dtype=np.intp)
    n_source = rows.size

    reg = _register_arrays(config, cls)
    n_model, n_sub = reg["n_model"], reg["n_sub"]
    template_log = reg["template_log"]
    ln_w_h = reg["ln_w_h"]
    subclass_idx = reg["subclass_idx"]
    av_law_draine, av_law_whitney = reg["av_law_draine"], reg["av_law_whitney"]

    fnu, sigma_fnu, origin = _catalog_bands(config, region, rows)
    a_col_k = np.asarray(prior.table["A_COL_K"], dtype=np.float64)[rows]
    ramp_w = selection.law_dense_weight(a_col_k)

    prior_cls = cls.lower()
    needs_model_index = prior_cls in ("gal", "h2s")
    model_index_1d = np.arange(n_model, dtype=np.intp) if needs_model_index else None

    topk = int(getattr(config, "fit_topk", TOPK))

    evidence = np.full((n_source, n_sub), -np.inf, dtype=np.float64)
    flux_mean = np.zeros((n_source, _N_BAND), dtype=np.float64)
    flux_cov = np.zeros((n_source, _N_BAND, _N_BAND), dtype=np.float64)
    topk_model = np.full((n_source, topk), -1, dtype=np.int64)
    topk_a = np.full((n_source, topk), np.nan, dtype=np.float64)
    topk_log10b = np.full((n_source, topk), np.nan, dtype=np.float64)
    topk_lnl = np.full((n_source, topk), -np.inf, dtype=np.float64)
    topk_chi2 = np.full((n_source, topk), np.inf, dtype=np.float64)
    topk_flux = np.full((n_source, topk, _N_BAND), np.nan, dtype=np.float64)
    n_detected = np.zeros(n_source, dtype=np.int64)

    # one-hot subclass membership (n_model, n_sub): turns the per-source
    # `np.bincount(subclass_idx, weights=...)` into one matmul per block.
    onehot = np.zeros((n_model, n_sub), dtype=np.float64)
    onehot[np.arange(n_model), subclass_idx] = 1.0

    model_index_full = np.arange(n_model, dtype=np.intp)
    keep = min(topk, n_model)

    # MEMORY DRIVER (owner ruling 2026-09-06, item 4), measured with
    # `tracemalloc` on STAR/NGC 7129 (n_model=4066), one job,
    # single-threaded, n=300 source rows split into two blocks: traced
    # Python-heap peak 1.17 GB against a naive two-block estimate of
    # ~625 MB at "8 arrays of (n_block, n_model, 8) float64" -- roughly
    # 1.9x. The flux-covariance step's `outer = np.einsum("bki,bkj->bij",
    # weighted_flux, model_fluxes_mjy, optimize=True)` was the first
    # suspect -- a per-source (8, 8) reduction over the WHOLE model axis
    # k looks like it could materialise a (n_block, n_model, 8, 8)
    # intermediate before summing over k -- but CHECKED, NOT ASSUMED:
    # `np.einsum_path` on that exact call (n_block=245, n_model=4066)
    # reports its own largest intermediate as the (n_block, 8, 8) OUTPUT
    # itself (1.568e4 elements), so it was never the driver. It is still
    # replaced below by an explicit `np.matmul` (a batched `W^T @ F`),
    # the owner-directed form, unambiguously BLAS-dispatched where
    # `einsum`'s own C loop is not guaranteed to be -- but this is a
    # robustness/performance change, not a fix for the excess peak. The
    # remaining five genuinely-alive (n_block, n_model, 8) arrays
    # (`residual`, `model`, `chi2_array`'s transient, `model_fluxes_mjy`,
    # `weighted_flux`) do not explain the measured 1.9x either; the
    # 16-array multiplier below is an empirical safety margin (it
    # measurably dropped STAR/NGC 7129's n_jobs=4 peak from 6.3 GB to
    # 4.1 GB), not a fully-identified array census. The true remaining
    # driver is still open.
    # row_bytes = n_model * n_band(8) * 8 bytes/float64 * 16
    # "array-equivalents", bounded under `config.fit_block_budget_mb`
    # (`config.py`'s `[fit] block_budget_mb`, default 512 MB).
    row_bytes = n_model * _N_BAND * 8 * 16

    def _process_block(start, stop):
        blk = slice(start, stop)
        n_blk = stop - start
        rows_blk = rows[blk]
        w_blk = ramp_w[blk]

        av_law_blk, ak_per_av_blk = _hybrid_av_law_block(
            config, w_blk, av_law_draine, av_law_whitney)
        av_min_blk = AK_MIN / ak_per_av_blk
        av_max_blk = AK_MAX / ak_per_av_blk

        valid, weight, log_flux, log_error = _source_log_fluxes_block(
            fnu[blk], sigma_fnu[blk], origin[blk])
        n_detected[blk] = np.sum(valid == VALID_DETECTION, axis=1)

        av_hat, sc_hat, chi2, model = _fit_model_grid_block(
            valid, weight, log_flux, log_error, template_log,
            av_law_blk, av_min_blk, av_max_blk)

        a_hat = av_hat * ak_per_av_blk[:, None]   # A_K, each source's own ramp law
        log10_b_hat = -2.0 * sc_hat               # 10_POSTERIOR.md section 1
        ln_l = -0.5 * chi2

        # the prior read: still the source's own fitted (a_hat,
        # log10_b_hat) per model, no quadrature -- one call for the
        # whole block, since `SourcePrior.log_density` already accepts
        # `rows` (n,) and `a`/`log10_b` (n, m) (its own docstring).
        ln_lambda = prior.log_density(
            prior_cls, rows_blk, a_hat, log10_b_hat,
            model_index=model_index_1d)

        if gamma is None:
            ln_gamma = np.zeros((n_blk, n_model), dtype=np.float64)
        else:
            # gamma's signature takes one row at a time; not vectorised.
            ln_gamma = np.stack([
                np.asarray(gamma(int(rows_blk[j]), model_index_full,
                                  a_hat[j], log10_b_hat[j]))
                for j in range(n_blk)])
        if psi is None:
            ln_psi = np.zeros((n_blk, n_model), dtype=np.float64)
        else:
            # psi's signature takes one row at a time; not vectorised.
            ln_psi = np.stack([
                np.asarray(psi(int(rows_blk[j]), model_index_full,
                                a_hat[j], log10_b_hat[j]))
                for j in range(n_blk)])

        ln_w_full = ln_w_h[None, :] + ln_lambda + ln_l + ln_gamma + ln_psi

        finite = np.isfinite(ln_w_full)
        m = np.where(finite.any(axis=1),
                     np.max(np.where(finite, ln_w_full, -np.inf), axis=1), 0.0)
        lin = np.where(finite, np.exp(ln_w_full - m[:, None]), 0.0)

        sub_sum = lin.dot(onehot)
        with np.errstate(divide="ignore"):
            evidence[blk] = np.where(
                sub_sum > 0.0, m[:, None] + np.log(np.where(sub_sum > 0.0, sub_sum, 1.0)),
                -np.inf)

        total = lin.sum(axis=1)
        has_total = total > 0.0
        safe_total = np.where(has_total, total, 1.0)
        model_fluxes_mjy = 10.0 ** (model + template_log[None, :, :])
        weighted_flux = lin[:, :, None] * model_fluxes_mjy
        mean_flux = weighted_flux.sum(axis=1) / safe_total[:, None]
        # Sum_k w_k f_k f_k^T, per source: a batched matmul, `(W^T @ F)`
        # per block-row -- `np.swapaxes(weighted_flux, 1, 2)` is
        # `(n_block, 8, n_model)`, `model_fluxes_mjy` is `(n_block,
        # n_model, 8)`, `np.matmul` contracts the shared `n_model` axis
        # straight to `(n_block, 8, 8)` via BLAS batched-gemm -- the same
        # arithmetic the earlier `np.einsum("bki,bkj->bij", ...,
        # optimize=True)` form computed. Checked, not assumed:
        # `np.einsum_path("bki,bkj->bij", ...)` on that form reports its
        # own largest intermediate as the `(n_block, 8, 8)` OUTPUT
        # itself (1.568e4 elements at n_block=245, n_model=4066) -- so
        # this einsum was never materialising the `(n_block, n_model, 8,
        # 8)` array this module's block-size comment (below) suspected.
        # `matmul` is kept over `einsum` anyway, as the owner-directed
        # form for this contraction: it is unambiguously dispatched to
        # BLAS, where `einsum`'s own C loop is not guaranteed to be.
        outer = np.matmul(np.swapaxes(weighted_flux, 1, 2), model_fluxes_mjy) / safe_total[:, None, None]
        cov = outer - np.einsum("bi,bj->bij", mean_flux, mean_flux)
        flux_mean[blk] = np.where(has_total[:, None], mean_flux, 0.0)
        flux_cov[blk] = np.where(has_total[:, None, None], cov, 0.0)

        if n_model > keep:
            part = np.argpartition(-ln_w_full, keep - 1, axis=1)[:, :keep]
            part_vals = np.take_along_axis(ln_w_full, part, axis=1)
            local_order = np.argsort(-part_vals, axis=1)
            order = np.take_along_axis(part, local_order, axis=1)
        else:
            order = np.argsort(-ln_w_full, axis=1)
        order_top = order[:, :keep]
        any_finite = finite.any(axis=1)

        topk_model[blk, :keep] = np.where(any_finite[:, None], order_top, -1)
        topk_a[blk, :keep] = np.where(
            any_finite[:, None], np.take_along_axis(a_hat, order_top, axis=1), np.nan)
        topk_log10b[blk, :keep] = np.where(
            any_finite[:, None], np.take_along_axis(log10_b_hat, order_top, axis=1), np.nan)
        topk_lnl[blk, :keep] = np.where(
            any_finite[:, None], np.take_along_axis(ln_l, order_top, axis=1), -np.inf)
        topk_chi2[blk, :keep] = np.where(
            any_finite[:, None], np.take_along_axis(chi2, order_top, axis=1), np.inf)
        topk_flux_blk = np.take_along_axis(
            model_fluxes_mjy, order_top[:, :, None], axis=1)
        topk_flux[blk, :keep, :] = np.where(
            any_finite[:, None, None], topk_flux_blk, np.nan)

        if stage is not None:
            with _tick_lock:
                _blocks_done[0] += 1
                done = _blocks_done[0]
                if done == 1:
                    # Force the very first completed block through
                    # `stage`'s own ten-second throttle (`progress.Stage.
                    # tick`'s `now - self._last >= _MIN_INTERVAL_S`
                    # check): a long class (H2S killed under `capped.sh`
                    # before its first throttled tick, owner ruling
                    # 2026-09-06) must show life immediately, not after
                    # ten seconds of silence.
                    stage._last = 0.0
            stage.tick(done, n_blocks, "blocks")

    block_budget_mb = int(getattr(config, "fit_block_budget_mb", 512))
    blocks = list(batches_module.batches(n_source, row_bytes, budget_bytes=block_budget_mb << 20))
    n_blocks = len(blocks)
    _tick_lock = threading.Lock()
    _blocks_done = [0]
    if len(blocks) > 1:
        # `config.n_jobs` Python threads is the parallelism (module
        # docstring); each thread's own BLAS calls (the fit's normal
        # equations, `lin.dot(onehot)`, the flux-covariance einsum) must
        # stay single-threaded here or BLAS spawns its own thread pool
        # per call, `n_jobs` of them at once, oversubscribing the
        # machine's cores and inflating memory (measured: STAR/NGC 7129
        # peaked at 6.5 GB uncapped vs 2.7 GB single-threaded, for the
        # same one-block-at-a-time work) -- `threadpool_limits(1)`
        # forces BLAS to do exactly the same arithmetic on the one
        # thread that called it.
        with threadpool_limits(1):
            Parallel(n_jobs=config.n_jobs, prefer="threads")(
                delayed(_process_block)(start, stop) for start, stop in blocks)
    else:
        for start, stop in blocks:
            _process_block(start, stop)

    return {
        "EVIDENCE": evidence, "FLUX_MEAN": flux_mean, "FLUX_COV": flux_cov,
        "TOPK_MODEL": topk_model, "TOPK_A": topk_a, "TOPK_LOG10B": topk_log10b,
        "TOPK_LNL": topk_lnl, "TOPK_CHI2": topk_chi2, "TOPK_FLUX": topk_flux,
        "N_DETECTED": n_detected,
    }
