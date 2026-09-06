"""The posterior: the prior table's six counts joined with the fit's six
evidence files into the class posterior, the global subclass posterior,
the argmax-committed flux imputation and the two entropies (`10_POSTERIOR.md`
section 1's `P(C | D_s) prop N_C . Sum_theta`; `reading/06_fitter_and_impute.md`
section A's "the class posterior" and "N_C re-estimable" rows -- the old
`bms_posterior.impute`'s `class_posterior`/`global_subclass_posterior`/
`entropy`/`apply_psi`, lifted as pure arithmetic without its provenance
checks, additive-decomposition verifier or ablation-variant machinery).

For one region: `N_C(s)`, six counts per source, comes from the prior
table (`prior.table.read`, already the per-source amplitudes the posterior
page names). `EV_C(s)`, the class's total evidence, is the log-sum-exp
over one evidence file's own `EVIDENCE` dataset (natural-log evidence per
subclass, `fit.sweep.fit_batch`'s own return, summed over the class's
subclasses -- `Sum_theta` restricted to one class -- exactly as the old
`ClassEvidence.ln_ev` was `logsumexp(ln_ev_sub)`). `P(C | D_s) = N_C . EV_C
/ Sum_C' N_C' . EV_C'` is computed by the same exact log-sum-exp shift the
old `class_posterior` used: shifting every class's ln-evidence by the
source's own max before exponentiating cancels in the ratio, so only a
class genuinely ~745 nats below the source's best class underflows to
zero. The global subclass posterior multiplies each class's own posterior
by its own softmax over that class's `EVIDENCE` columns (the same shift,
scoped to one class). The argmax-committed class (lowest index wins an
exact tie, `np.argmax`'s own convention) and its own `FLUX_MEAN`/
`FLUX_COV` are read off that one evidence file as the imputation -- no
cross-class mixture, `reading/06_fitter_and_impute.md`'s "argmax-commit"
row. Psi is wired at `BETA = 0` (the old value, inert today: `beta * ln
Psi` is the identity), so no `T`/`qbar` term is read.
"""

import os
import time

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.fit.run import CLASSES, evidence_path
from sesnaimpute.prior import table as table_module

#: `CLASSMAP` in its own stored order: one row per subclass, the order
#: `global_subclass_posterior`'s output columns follow.
SUBCLASSES = tuple((s.cls, s.name) for s in definitions.CLASSMAP)
SUBCLASS_INDEX = {key: i for i, key in enumerate(SUBCLASSES)}
N_SUBCLASS = len(SUBCLASSES)

#: The color-cascade concordance dial (`10_POSTERIOR.md` section 1's
#: `Psi^beta`): 0 today, `reading/06_fitter_and_impute.md`'s own citation
#: of the old value -- the identity, `EV_SUB *= Psi**0 == 1`.
BETA = 0.0

N_BANDS = 8


def _log_sum_exp(ln_x, axis):
    """`ln(sum(exp(ln_x)))` along `axis`, by the exact max-shift
    (module docstring): an all `-inf` slice stays `-inf`, not `nan`."""
    m = np.max(ln_x, axis=axis, keepdims=True)
    finite = np.isfinite(m)
    shifted = np.where(finite, ln_x - np.where(finite, m, 0.0), -np.inf)
    with np.errstate(divide="ignore"):
        summed = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    out = np.squeeze(m + summed, axis=axis)
    return np.where(np.squeeze(finite, axis=axis), out, -np.inf)


def _read_evidence(config, region):
    """One dict per class: `EVIDENCE` (n, n_subclass_of_class),
    `FLUX_MEAN` (n, 8), `FLUX_COV` (n, 8, 8), `N_DETECTED` (n), `NAME`
    order asserted against the prior table's own (the join's row-order
    check, `IMPLEMENTATION.md` section 5's accessor discipline)."""
    out = {}
    for cls in CLASSES:
        path = evidence_path(config, region, cls)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "impute.posterior: no evidence file for %r class %r at %s -- "
                "run the 'fit.run' RUNBOOK line first" % (region, cls, path))
        with h5py.File(path, "r") as f:
            if f.attrs.get("GRANULE") != "source":
                raise ValueError("impute.posterior: %s declares GRANULE %r, expected 'source'"
                                 % (path, f.attrs.get("GRANULE")))
            if f.attrs.get("CLASS") != cls:
                raise ValueError("impute.posterior: %s declares CLASS %r, expected %r"
                                 % (path, f.attrs.get("CLASS"), cls))
            out[cls] = {key: f[key][:] for key in
                       ("EVIDENCE", "FLUX_MEAN", "FLUX_COV", "N_DETECTED")}
    return out


def class_posterior(n_class, ln_ev_class):
    """`P_CLASS` (n, 6): `N_C . EV_C / Sum_C' N_C' . EV_C'` by the exact
    log-sum-exp shift (module docstring)."""
    shift = np.max(ln_ev_class, axis=1, keepdims=True)
    finite = np.isfinite(shift)
    weighted = np.where(finite, n_class * np.exp(ln_ev_class - np.where(finite, shift, 0.0)), 0.0)
    total = weighted.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError(
            "impute.posterior: class_posterior: %d source(s) have zero total weighted "
            "evidence across every class" % int(np.count_nonzero(total <= 0.0)))
    return weighted / total


def global_subclass_posterior(p_class, ln_ev_sub_by_class):
    """`P_SUBCLASS` (n, N_SUBCLASS), `SUBCLASSES` order, each row summing
    to 1: `P(sub) = P_CLASS[class(sub)] . P(sub | class(sub))`,
    `P(sub | class)` the same exact-shift softmax scoped to one class's
    own subclasses (module docstring)."""
    n = p_class.shape[0]
    out = np.zeros((n, N_SUBCLASS), dtype=np.float64)
    for ci, cls in enumerate(CLASSES):
        ln_ev_sub = ln_ev_sub_by_class[cls]
        shift = np.max(ln_ev_sub, axis=1, keepdims=True)
        finite = np.isfinite(shift)
        weighted = np.where(finite, np.exp(ln_ev_sub - np.where(finite, shift, 0.0)), 0.0)
        denom = weighted.sum(axis=1, keepdims=True)
        cond = np.divide(weighted, denom, out=np.zeros_like(weighted), where=denom > 0.0)
        subclasses = definitions.SUBCLASSES_OF[cls]
        for si, sub in enumerate(subclasses):
            out[:, SUBCLASS_INDEX[(cls, sub)]] = p_class[:, ci] * cond[:, si]
    return out


def entropy(p):
    """Natural-log Shannon entropy along the last axis, `0 . ln 0 := 0`
    (module docstring)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * np.log(p), 0.0)
    return -terms.sum(axis=-1)


def apply_psi(ln_ev_sub, beta):
    """`ln(EV_SUB) += beta * ln(Psi)`; at `BETA = 0` (module docstring)
    this is the identity and needs no `T`/`qbar` term, matching the old
    `apply_psi`'s own beta==0 short-circuit. A nonzero beta is not wired
    in this port (no `T`/`qbar` in the evidence files) and is refused by
    name rather than silently ignored."""
    if beta == 0.0:
        return ln_ev_sub
    raise NotImplementedError(
        "impute.posterior.apply_psi: beta != 0 needs the color-cascade T/qbar "
        "terms, not carried by this port's evidence files")


def _committed_flux(committed, flux_mean_stack, flux_cov_stack):
    """`(FLUX_IMPUTED, FLUX_IMPUTED_COV)`: the committed class's own
    predictive mean and covariance, gathered per source (module
    docstring's "argmax-commit", no cross-class mixture)."""
    n = committed.shape[0]
    rows = np.arange(n)
    flux_mean = flux_mean_stack[rows, committed, :]
    flux_cov = flux_cov_stack[rows, committed, :, :]
    return flux_mean, flux_cov


def _build_one(config, region):
    t0 = time.time()
    prior = table_module.read(config, region)
    name = prior["NAME"]
    n_sources = name.shape[0]

    n_class = np.stack([prior["N_%s" % cls] for cls in CLASSES], axis=1).astype(np.float64)
    evidence = _read_evidence(config, region)

    ln_ev_sub_by_class = {}
    ln_ev_class = np.empty((n_sources, len(CLASSES)), dtype=np.float64)
    flux_mean_stack = np.empty((n_sources, len(CLASSES), N_BANDS), dtype=np.float64)
    flux_cov_stack = np.empty((n_sources, len(CLASSES), N_BANDS, N_BANDS), dtype=np.float64)
    for ci, cls in enumerate(CLASSES):
        ev = evidence[cls]
        if ev["EVIDENCE"].shape[0] != n_sources:
            raise ValueError(
                "impute.posterior: %r's %r evidence has %d rows, prior table has %d sources"
                % (region, cls, ev["EVIDENCE"].shape[0], n_sources))
        ln_ev_sub = apply_psi(ev["EVIDENCE"], BETA)
        ln_ev_sub_by_class[cls] = ln_ev_sub
        ln_ev_class[:, ci] = _log_sum_exp(ln_ev_sub, axis=1)
        flux_mean_stack[:, ci, :] = ev["FLUX_MEAN"]
        flux_cov_stack[:, ci, :, :] = ev["FLUX_COV"]
    n_detected = evidence[CLASSES[0]]["N_DETECTED"]

    p_class = class_posterior(n_class, ln_ev_class)
    p_subclass = global_subclass_posterior(p_class, ln_ev_sub_by_class)
    committed = np.argmax(p_class, axis=1).astype(np.int32)
    flux_imputed, flux_imputed_cov = _committed_flux(committed, flux_mean_stack, flux_cov_stack)
    entropy_class = entropy(p_class)
    entropy_subclass = entropy(p_subclass)

    out_path = _write(config, region, name, p_class, p_subclass, committed,
                      flux_imputed, flux_imputed_cov, entropy_class, entropy_subclass,
                      n_detected)
    wall_s = time.time() - t0
    print("impute.posterior: %s: %d sources, wall=%.1fs -> %s"
         % (region, n_sources, wall_s, out_path), flush=True)
    return out_path


def _output_path(config, region):
    return config_module.product_path(config, "bms", "impute", "posterior", "source", region=region)


def _write(config, region, name, p_class, p_subclass, committed, flux_imputed,
          flux_imputed_cov, entropy_class, entropy_subclass, n_detected):
    out_path = _output_path(config, region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.create_dataset("NAME", data=name)
        f.create_dataset("P_CLASS", data=p_class.astype(np.float32))
        f.create_dataset("P_SUBCLASS", data=p_subclass.astype(np.float32))
        f.create_dataset("COMMITTED_CLASS", data=committed)
        f.create_dataset("FLUX_IMPUTED", data=flux_imputed.astype(np.float32))
        f.create_dataset("FLUX_IMPUTED_COV", data=flux_imputed_cov.astype(np.float32))
        f.create_dataset("ENTROPY_CLASS", data=entropy_class.astype(np.float32))
        f.create_dataset("ENTROPY_SUBCLASS", data=entropy_subclass.astype(np.float32))
        f.create_dataset("N_DETECTED", data=n_detected)
    return out_path


def build(config, regions=None):
    """Writes the per-region imputed posterior for `regions` (default:
    all thirty), one product per region, from that region's prior table
    and six evidence files (module docstring)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        _build_one(config, region)


if __name__ == "__main__":
    run(build)
