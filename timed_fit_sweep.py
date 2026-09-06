"""The timed acceptance run for `sesnaimpute.fit.sweep.fit_batch`
(CODING_RULES.md 10, "time it on real data"; the fit-sweep brief).

For one region, 200 catalogue sources, class STAR (the `sps` register,
4,066 models) and class YSO (the pooled `yso` register, 200,000 models):
`prior.prepare` on those 200 rows once (shared across both classes), then
`fit_batch` timed per class with both census hooks at their zero default.
Reports wall time per source per class, peak resident memory (read this
script's own stderr under `capped.sh`, printed after the process exits),
and the survey extrapolation to 8.66e6 sources, all six census classes,
on 4 cores.

Then the brief's own algebraic acceptance, for one source of the batch
and one model of each class's register:

- the chi-squared at the fit's own returned `(a_hat, log10_b_hat)`
  equals a direct recomputation of `chi_squared` from the source's raw
  fluxes and that model's template, to 1e-9;
- that chi-squared is <= the chi-squared at 20 random `(a, log10 b)`
  points drawn in a box around it (the fit is at least a local optimum
  of the grid the closed-form solve searches);
- the evidence for one subclass equals the log-sum-exp of that
  subclass's own per-model `ln_w_h + ln_lambda + ln_L` terms, recomputed
  directly from the same per-model arrays `fit_batch` computed, to 1e-9.

Run under `capped.sh` (CODING_RULES.md 10a):
    ./capped.sh /usr/local/bin/python3.9 timed_fit_sweep.py <config> <region>
"""

import sys
import time

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute.fit import sweep
from sesnaimpute.prior import selection
from sesnaimpute.prior.callable import SourcePrior

N_SOURCES = 200
N_MODEL_LABEL = {"STAR": 4066, "YSO": 200000}
SURVEY_N_SOURCES = 8.66e6
N_CLASSES = 6
N_CORES = 4
N_RANDOM_ACCEPTANCE = 20


def _direct_chi2(valid, log_flux, log_error, weight, template_row, av, sc, av_law):
    """One model's chi-squared at one trial `(av, sc)`, recomputed from
    the raw per-band arrays with no reference to `sweep`'s own vectorised
    solve -- `sedfitter.fitting_routines.chi_squared`'s own formula,
    evaluated by hand at one point."""
    model = av * av_law + sc * sweep._SC_LAW
    residual = log_flux - template_row
    chi2 = (residual - model) ** 2 * weight
    chi2 = np.where(valid == 0, 0.0, chi2)
    for j in np.where(valid == 2)[0]:
        if model[j] < residual[j]:
            chi2[j] = -2.0 * np.log(1.0 - log_error[j])
    for j in np.where(valid == 3)[0]:
        if model[j] > residual[j]:
            chi2[j] = -2.0 * np.log(1.0 - log_error[j])
    chi2[np.isinf(chi2)] = 1.0e30
    return float(np.sum(chi2))


def acceptance(config, region, cls, row, seed=0):
    rng = np.random.default_rng(seed)
    reg = sweep._register_arrays(config, cls)
    fnu, sigma_fnu, origin = sweep._catalog_bands(config, region, np.array([row]))
    valid, weight, log_flux, log_error = sweep._source_log_fluxes(fnu[0], sigma_fnu[0], origin[0])

    prior = SourcePrior(config, region)
    prior.prepare(np.array([row], dtype=np.intp))
    a_col_k = float(np.asarray(prior.table["A_COL_K"])[row])
    w = float(selection.law_dense_weight(a_col_k))
    av_law = sweep.hybrid_av_law(config, w, reg["av_law_draine"], reg["av_law_whitney"])
    av_min, av_max = sweep.av_range_for_law_weight(config, w)

    av_hat, sc_hat, chi2, model = sweep._fit_one_model_grid(
        valid, weight, log_flux, log_error, reg["template_log"], av_law, av_min, av_max)

    # -- test 1: one model's chi2 at (a_hat, sc_hat) matches a scratch
    #    recomputation to 1e-9.
    m = 0
    direct = _direct_chi2(valid, log_flux, log_error, weight, reg["template_log"][m],
                          av_hat[m], sc_hat[m], av_law)
    test1 = abs(direct - float(chi2[m]))

    # -- test 2: the fitted (av, sc) is <= 20 random points nearby.
    span_av = max(0.5, 0.1 * abs(float(av_hat[m])))
    span_sc = 0.5
    worse = 0
    for _ in range(N_RANDOM_ACCEPTANCE):
        av_try = np.clip(float(av_hat[m]) + rng.uniform(-span_av, span_av), av_min, av_max)
        sc_try = float(sc_hat[m]) + rng.uniform(-span_sc, span_sc)
        c_try = _direct_chi2(valid, log_flux, log_error, weight, reg["template_log"][m],
                             av_try, sc_try, av_law)
        if c_try < float(chi2[m]) - 1.0e-9:
            worse += 1
    test2_violations = worse

    # -- test 3: the evidence for one subclass equals the direct
    #    log-sum-exp of that subclass's own per-model terms.
    a_hat = av_hat * float(selection.ak_per_av(config, w))
    log10_b_hat = -2.0 * sc_hat
    ln_l = -0.5 * chi2
    needs_mi = cls.lower() in ("gal", "h2s")
    mi = np.arange(reg["n_model"], dtype=np.intp) if needs_mi else None
    ln_lambda = prior.log_density(cls.lower(), np.array([row], dtype=np.intp),
                                  a_hat[None, :], log10_b_hat[None, :], model_index=mi)[0]
    ln_w_full = reg["ln_w_h"] + ln_lambda + ln_l
    sub0 = int(reg["subclass_idx"][0])
    members = reg["subclass_idx"] == sub0
    terms = ln_w_full[members]
    finite = np.isfinite(terms)
    direct_ev = -np.inf
    if finite.any():
        mx = float(np.max(terms[finite]))
        direct_ev = mx + float(np.log(np.sum(np.exp(terms[finite] - mx))))

    out = sweep.fit_batch(config, region, cls, np.array([row], dtype=np.intp), prior)
    fit_ev = float(out["EVIDENCE"][0, sub0])
    test3 = (0.0 if (np.isneginf(direct_ev) and np.isneginf(fit_ev))
            else abs(direct_ev - fit_ev))

    return {"chi2_direct_diff": test1, "worse_than_random": test2_violations,
           "evidence_direct_diff": test3}


def main(config_path, region):
    config = config_module.load(config_path)

    catalog_path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(catalog_path, "r") as f:
        n_total = int(f.attrs["N_SOURCES"])
    n = min(N_SOURCES, n_total)
    rows = np.arange(n, dtype=np.intp)

    prior = SourcePrior(config, region)
    t0 = time.time()
    prior.prepare(rows)
    t_prepare = time.time() - t0
    print(f"timed_fit_sweep: {region}: prior.prepare({n} rows) wall={t_prepare:.3f}s", flush=True)

    per_class_wall = {}
    for cls in ("STAR", "YSO"):
        t0 = time.time()
        out = sweep.fit_batch(config, region, cls, rows, prior)
        wall = time.time() - t0
        per_class_wall[cls] = wall
        per_source = wall / n
        print(f"timed_fit_sweep: {region}: {cls}: n_source={n} "
             f"n_model={N_MODEL_LABEL[cls]} wall={wall:.3f}s "
             f"per_source={per_source * 1000.0:.2f}ms "
             f"n_detected_mean={float(np.mean(out['N_DETECTED'])):.2f}", flush=True)

    total_per_source_two_classes = sum(per_class_wall.values()) / n
    survey_hours_two_classes = (total_per_source_two_classes * SURVEY_N_SOURCES
                                / N_CORES / 3600.0)
    survey_hours_six_classes = survey_hours_two_classes * (N_CLASSES / 2.0)
    print(f"timed_fit_sweep: {region}: survey extrapolation (8.66e6 sources, "
         f"{N_CORES} cores): STAR+YSO alone = {survey_hours_two_classes:.3g} hours; "
         f"scaled x{N_CLASSES}/2 classes (STAR/YSO stand in for the other four's "
         f"own model counts, NOT measured) = {survey_hours_six_classes:.3g} hours",
         flush=True)

    for cls in ("STAR", "YSO"):
        acc = acceptance(config, region, cls, int(rows[0]))
        print(f"timed_fit_sweep: {region}: {cls}: acceptance: "
             f"chi2_direct_diff={acc['chi2_direct_diff']:.3e} (bar 1e-9) "
             f"worse_than_random={acc['worse_than_random']}/{N_RANDOM_ACCEPTANCE} (bar 0) "
             f"evidence_direct_diff={acc['evidence_direct_diff']:.3e} (bar 1e-9)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
