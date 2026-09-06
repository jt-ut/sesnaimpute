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
wrapper (`sedfitter.fitting_routines` is called directly on plain
float64), and one source's whole `(n_model, n_band)` array is at most
200,000 x 8 float64 = 12.8 MB (YSO, the largest register) -- far under
CODING_RULES.md 10a's ceiling -- so models are never batched within a
source; only rule 10b's source-batching (about ten thousand at a time,
the caller's concern) applies. The evidence log-sum-exp therefore needs
no running-max rescale either: it is one `exp()` over the source's whole
model axis, not folded across successive model batches.
"""

import os

import h5py
import numpy as np
from sedfitter import fitting_routines as sedfit_algebra

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
#: production default).
TOPK = 5

#: class code (`definitions.CLASSES`) -> the library register key
#: (`sed_models/registers/<key>_register.hdf5`). GAL/H2S transcribed
#: from `sesnaimpute.prior.callable._LIBRARY_KEY`; STAR/PAHC/AGB/YSO
#: from the register file names their own modules already read
#: (`sesnaimpute.prior.field_stars`/`star_population`: `sps_register.
#: hdf5`, `pahc_register.hdf5`; the on-disk `agb_register.hdf5`,
#: `yso_register.hdf5`, the latter a pooled register over YSO's five
#: sub-grids).
CLASS_REGISTER = {
    "STAR": "sps", "AGB": "agb", "PAHC": "pahc",
    "GAL": "galz", "YSO": "yso", "H2S": "h2shock",
}

_BAND_KEYS = tuple(b.key for b in definitions.BANDS)
_N_BAND = len(_BAND_KEYS)

#: `sedfitter`'s free-scale design vector for the gray-scale nuisance:
#: a constant -2 in every band, exactly (`sed_fit.fit`'s own docstring,
#: Q1) -- `10_POSTERIOR.md` section 1's `log10 B = -2*SC`, `SC` the
#: fitted coefficient on this vector.
_SC_LAW = np.full(_N_BAND, -2.0, dtype=np.float64)


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


def hybrid_av_law(config, w, av_law_draine, av_law_whitney):
    """The `av_law` design vector (log10-flux-per-A_V, per band) for one
    source at ramp weight `w`, blended between the register's own two
    pure laws.

    Lifted from `sesnacomplete.sed_fit.fit.hybrid_av_law`: undo each pure
    vector's own magnitude-to-flux scaling to recover its K-normalised
    dimming vector (`kappa = av_law / (-0.4 * (A_K/A_V)_pure)`), blend
    the two `kappa`s convexly at `w`, then re-apply the hybrid curve's
    own `(A_K/A_V)` at that weight (`sesnaimpute.prior.selection.
    ak_per_av`'s harmonic blend). At `w = 0` or `w = 1` the pure endpoint
    vector is returned directly (bit-exact), not through the round trip.
    """
    if w <= 0.0:
        return av_law_draine
    if w >= 1.0:
        return av_law_whitney
    r_diffuse = selection.ak_per_av(config, 0.0)
    r_dense = selection.ak_per_av(config, 1.0)
    kappa_diffuse = av_law_draine / (-0.4 * r_diffuse)
    kappa_dense = av_law_whitney / (-0.4 * r_dense)
    kappa_s = (1.0 - w) * kappa_diffuse + w * kappa_dense
    ak_per_av_s = float(selection.ak_per_av(config, w))
    return -0.4 * ak_per_av_s * kappa_s


def av_range_for_law_weight(config, w):
    """`(av_min, av_max)`: the project's A_K bound (`AK_MIN`/`AK_MAX`),
    converted to A_V through this source's own ramp weight's `(A_K/A_V)`
    -- lifted from `sesnacomplete.sed_fit.fit.av_range_for_law_weight`.
    The bound is declared in A_K because the law blends per source, so
    one nominal A_V number would be a different physical limit at every
    weight; converted here, `AK_MAX` is the same physical limit at every
    `w`, in that weight's own A_V.
    """
    ak_per_av_w = float(selection.ak_per_av(config, w))
    return AK_MIN / ak_per_av_w, AK_MAX / ak_per_av_w


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


def _source_log_fluxes(flux, sigma, origin):
    """`(valid, weight, log_flux, log_error)`, each `(8,)`: one source's
    `ORIGIN_FNU` codes translated to `sedfitter`'s `valid` vocabulary,
    then `sedfitter.source.Source.get_log_fluxes`'s own algebra
    (transcribed, not called -- there is no `Source` object here, only
    plain arrays)."""
    valid = np.where(np.isin(origin, ORIGIN_UPPER_LIMIT), VALID_UPPER_LIMIT,
                     np.where(origin == ORIGIN_DETECTED, VALID_DETECTION, VALID_EXCLUDED))
    weight = np.zeros(_N_BAND, dtype=np.float64)
    log_flux = np.zeros(_N_BAND, dtype=np.float64)
    log_error = np.zeros(_N_BAND, dtype=np.float64)

    det = valid == VALID_DETECTION
    log_flux[det] = np.log10(flux[det]) - 0.5 * (sigma[det] / flux[det]) ** 2 / np.log(10.0)
    log_error[det] = np.abs(sigma[det] / flux[det]) / np.log(10.0)
    weight[det] = 1.0 / log_error[det] ** 2

    lim = valid == VALID_UPPER_LIMIT
    log_flux[lim] = np.log10(flux[lim])
    log_error[lim] = sigma[lim]  # the confidence c; weight stays 0 (Q5)

    return valid, weight, log_flux, log_error


def _fit_one_model_grid(valid, weight, log_flux, log_error, template_log, av_law, av_min, av_max):
    """The closed-form free-scale fit (`sesnacomplete.sed_fit.batched.
    fit_models_batched`'s FREE_SCALE branch, transcribed): profiled A_V
    and gray scale, and the chi-squared at that optimum, for every model
    of `template_log` at once (the vectorised axis). Returns `(av_hat,
    sc_hat, chi2)`, each `(n_model,)`.
    """
    residual = log_flux[None, :] - template_log

    # the zero-weight guard (batched.py's module docstring): a
    # legitimate zero model flux in a band the fit already excluded
    # (weight == 0) must contribute nothing, not NaN.
    zero_weight = weight == 0.0
    bad = None
    if zero_weight.any():
        bad = zero_weight[None, :] & ~np.isfinite(residual)
        if not bad.any():
            bad = None
    sign = None
    if bad is not None:
        sign = np.sign(residual[bad])
        sign[np.isnan(sign)] = 0.0
        residual[bad] = 0.0

    av_hat, sc_hat = sedfit_algebra.linear_regression(residual, weight, av_law, _SC_LAW)

    reset_lo = av_hat < av_min
    reset_hi = av_hat > av_max
    av_hat = np.where(reset_lo, av_min, np.where(reset_hi, av_max, av_hat))
    reset = reset_lo | reset_hi
    if reset.any():
        sc_hat = sc_hat.copy()
        sc_hat[reset] = sedfit_algebra.optimal_scaling(
            residual[reset] - av_hat[reset][:, None] * av_law[None, :], weight, _SC_LAW)

    model = av_hat[:, None] * av_law[None, :] + sc_hat[:, None] * _SC_LAW[None, :]

    if bad is not None:
        residual[bad] = model[bad] + sign

    chi2 = sedfit_algebra.chi_squared(valid, residual, log_error, weight, model)
    return av_hat, sc_hat, chi2, model


def fit_batch(config, region, cls, rows, prior, gamma=None, psi=None):
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

    Returns a dict of plain numpy arrays, row-aligned to `rows`:
    `EVIDENCE` `(n_source, n_sub)` (natural-log, this class's own
    subclasses in `definitions.SUBCLASSES_OF[cls]` order), `FLUX_MEAN`
    `(n_source, 8)`, `FLUX_COV` `(n_source, 8, 8)`, `TOPK_MODEL`/
    `TOPK_A`/`TOPK_LOG10B`/`TOPK_LNL` `(n_source, TOPK)`, `N_DETECTED`
    `(n_source,)`.
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

    evidence = np.full((n_source, n_sub), -np.inf, dtype=np.float64)
    flux_mean = np.zeros((n_source, _N_BAND), dtype=np.float64)
    flux_cov = np.zeros((n_source, _N_BAND, _N_BAND), dtype=np.float64)
    topk_model = np.full((n_source, TOPK), -1, dtype=np.int64)
    topk_a = np.full((n_source, TOPK), np.nan, dtype=np.float64)
    topk_log10b = np.full((n_source, TOPK), np.nan, dtype=np.float64)
    topk_lnl = np.full((n_source, TOPK), -np.inf, dtype=np.float64)
    n_detected = np.zeros(n_source, dtype=np.int64)

    for i in range(n_source):
        row = int(rows[i])
        w = float(ramp_w[i])
        av_law = hybrid_av_law(config, w, av_law_draine, av_law_whitney)
        ak_per_av_w = float(selection.ak_per_av(config, w))
        av_min, av_max = av_range_for_law_weight(config, w)

        valid, weight, log_flux, log_error = _source_log_fluxes(
            fnu[i], sigma_fnu[i], origin[i])
        n_detected[i] = int(np.sum(valid == VALID_DETECTION))

        av_hat, sc_hat, chi2, model = _fit_one_model_grid(
            valid, weight, log_flux, log_error, template_log, av_law, av_min, av_max)

        a_hat = av_hat * ak_per_av_w          # A_K, this source's own ramp law
        log10_b_hat = -2.0 * sc_hat           # 10_POSTERIOR.md section 1
        ln_l = -0.5 * chi2

        rows_probe = np.array([row], dtype=np.intp)
        ln_lambda = prior.log_density(
            prior_cls, rows_probe, a_hat[None, :], log10_b_hat[None, :],
            model_index=model_index_1d)[0]

        ln_gamma = (np.zeros(n_model) if gamma is None
                   else np.asarray(gamma(row, np.arange(n_model), a_hat, log10_b_hat)))
        ln_psi = (np.zeros(n_model) if psi is None
                 else np.asarray(psi(row, np.arange(n_model), a_hat, log10_b_hat)))

        ln_w_full = ln_w_h + ln_lambda + ln_l + ln_gamma + ln_psi

        finite = np.isfinite(ln_w_full)
        if finite.any():
            m = float(np.max(ln_w_full[finite]))
            lin = np.where(finite, np.exp(ln_w_full - m), 0.0)
            sub_sum = np.bincount(subclass_idx, weights=lin, minlength=n_sub)
            with np.errstate(divide="ignore"):
                evidence[i] = np.where(sub_sum > 0.0, m + np.log(sub_sum), -np.inf)

            total = float(lin.sum())
            if total > 0.0:
                model_fluxes_mjy = 10.0 ** (model + template_log)
                mean_flux = (lin[:, None] * model_fluxes_mjy).sum(axis=0) / total
                outer = np.einsum("k,ki,kj->ij", lin, model_fluxes_mjy, model_fluxes_mjy) / total
                flux_mean[i] = mean_flux
                flux_cov[i] = outer - np.outer(mean_flux, mean_flux)

            keep = min(TOPK, n_model)
            best = np.argpartition(-ln_w_full, keep - 1)[:keep] if n_model > keep else np.arange(n_model)
            order = best[np.argsort(-ln_w_full[best])]
            topk_model[i, :keep] = order
            topk_a[i, :keep] = a_hat[order]
            topk_log10b[i, :keep] = log10_b_hat[order]
            topk_lnl[i, :keep] = ln_l[order]

    return {
        "EVIDENCE": evidence, "FLUX_MEAN": flux_mean, "FLUX_COV": flux_cov,
        "TOPK_MODEL": topk_model, "TOPK_A": topk_a, "TOPK_LOG10B": topk_log10b,
        "TOPK_LNL": topk_lnl, "N_DETECTED": n_detected,
    }
