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
row. Psi is wired at `config.fit_beta` (0 by default -- `beta * ln Psi`
is then the identity), so no `T`/`qbar` term is read unless a nonzero
beta is configured.

THE IMPUTATION RULE (owner ruling 2026-09-06) is a SELECTION, never a
MIXTURE. Within one class's library, the candidate flux is that class's
own evidence-weighted average of its fitted models' fluxes (`fit.sweep.
fit_batch`'s `FLUX_MEAN`, verified here: `weighted_flux.sum(axis=1) /
total`, `weighted_flux = lin * model_fluxes_mjy`, `lin` proportional to
each model's own evidence -- exactly the evidence-weighted average of
that one library's own fitted fluxes, nothing else). In whichever bands
the source has a real measurement (`catalog.curated`'s `ORIGIN_FNU ==
1`), the candidate keeps the source's own catalogued flux in place of
the model average -- only the undetected bands are imputed. This gives
six per-class candidates (`CANDIDATE_FLUX` below, `CLASSES` order). The
official `FLUX_IMPUTED` is simply the MAP class's own candidate
(`argmax P(C | D)`, `COMMITTED_CLASS`) -- never an average, weighted
sum, or any other combination ACROSS the six candidates. No cross-class
mixture is introduced anywhere in this module.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.fit.run import CLASSES, evidence_path
from sesnaimpute.fit.sweep import ORIGIN_DETECTED
from sesnaimpute.prior import table as table_module
from sesnaimpute.progress import Stage

#: `CLASSMAP` in its own stored order: one row per subclass, the order
#: `global_subclass_posterior`'s output columns follow.
SUBCLASSES = tuple((s.cls, s.name) for s in definitions.CLASSMAP)
SUBCLASS_INDEX = {key: i for i, key in enumerate(SUBCLASSES)}
N_SUBCLASS = len(SUBCLASSES)

#: The color-cascade concordance dial (`10_POSTERIOR.md` section 1's
#: `Psi^beta`): a fallback only -- `_build_one` reads the live value off
#: `config.fit_beta` (owner ruling 2026-09-06, `config.py`'s `[fit]
#: beta`), 0 today, so `EV_SUB *= Psi**0 == 1`.
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


def _read_evidence(config, region, classes):
    """One dict per class in `classes` (not necessarily all six --
    `build`'s own `--classes` restriction, owner ruling 2026-09-06):
    `EVIDENCE` (n, n_subclass_of_class), `FLUX_MEAN` (n, 8), `FLUX_COV`
    (n, 8, 8), `N_DETECTED` (n), `NAME` order asserted against the prior
    table's own (the join's row-order check, `IMPLEMENTATION.md` section
    5's accessor discipline). Missing files for classes NOT in `classes`
    are never touched; a class named in `classes` with no file is still
    a hard error."""
    out = {}
    for cls in classes:
        path = evidence_path(config, region, cls)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "impute.posterior: no evidence file for %r class %r at %s -- "
                "run the 'fit.run' RUNBOOK line first, or drop it from --classes"
                % (region, cls, path))
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


def _own_flux_detected(config, region):
    """`(FNU_MJY, DETECTED)`, each `(n_source, 8)`, `definitions.BANDS`
    order, read straight from the curated catalogue (the same band-order
    assertion `fit.sweep._catalog_bands` makes): `DETECTED` is
    `ORIGIN_FNU == ORIGIN_DETECTED` (`fit.sweep`'s own vocabulary, 1 = a
    real flux measurement) -- the mask the imputation rule uses to keep a
    source's own measured flux in a detected band (module docstring)."""
    path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    band_keys = tuple(b.key for b in definitions.BANDS)
    with h5py.File(path, "r") as f:
        curated_bands = [b.decode() if isinstance(b, bytes) else b for b in f.attrs["BANDS"]]
        if list(curated_bands) != list(band_keys):
            raise ValueError(
                "impute.posterior: %r's own band order %r does not match "
                "definitions.BANDS order %r" % (path, curated_bands, band_keys))
        fnu = np.asarray(f["FNU_MJY"][:], dtype=np.float64)
        origin = np.asarray(f["ORIGIN_FNU"][:])
    return fnu, origin == ORIGIN_DETECTED


def class_posterior(n_class, ln_ev_class, strict=True):
    """`P_CLASS` (n, n_classes): `N_C . EV_C / Sum_C' N_C' . EV_C'` by the
    exact log-sum-exp shift (module docstring). `n_classes` is however
    many classes `build` was given (all six, or a `--classes` subset).

    A source with zero total weighted evidence across every given class
    is a hard error when `strict` (the full six-class run: no genuine
    source should fit none of them). Restricted to a subset, this is an
    expected, flagged outcome -- a source whose true class was left out
    of `classes` -- so `strict=False` (`build`'s own subset case) instead
    prints a warning and falls back to a uniform posterior over the
    given classes for just those rows (flag, don't block)."""
    shift = np.max(ln_ev_class, axis=1, keepdims=True)
    finite = np.isfinite(shift)
    weighted = np.where(finite, n_class * np.exp(ln_ev_class - np.where(finite, shift, 0.0)), 0.0)
    total = weighted.sum(axis=1, keepdims=True)
    zero = total <= 0.0
    if np.any(zero):
        n_zero = int(np.count_nonzero(zero))
        if strict:
            raise ValueError(
                "impute.posterior: class_posterior: %d source(s) have zero total weighted "
                "evidence across every class" % n_zero)
        print("impute.posterior: WARNING: %d source(s) have zero total weighted evidence "
              "across every one of the %d given class(es) -- their class(es) were likely "
              "left out of --classes; falling back to a uniform posterior for those rows"
              % (n_zero, ln_ev_class.shape[1]), flush=True)
        n_c = ln_ev_class.shape[1]
        uniform = np.full_like(weighted, 1.0 / n_c)
        weighted = np.where(zero, uniform, weighted)
        total = np.where(zero, 1.0, total)
    return weighted / total


def global_subclass_posterior(p_class, ln_ev_sub_by_class, classes):
    """`P_SUBCLASS` (n, N_SUBCLASS), `SUBCLASSES` order, each row summing
    to 1 (or to the restricted classes' total mass if `classes` omits
    some -- their subclass columns simply stay zero): `P(sub) =
    P_CLASS[class(sub)] . P(sub | class(sub))`, `P(sub | class)` the same
    exact-shift softmax scoped to one class's own subclasses (module
    docstring). `classes` gives `p_class`'s own column order."""
    n = p_class.shape[0]
    out = np.zeros((n, N_SUBCLASS), dtype=np.float64)
    for ci, cls in enumerate(classes):
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


def _committed_flux(committed, candidate_flux, flux_cov_stack):
    """`(FLUX_IMPUTED, FLUX_IMPUTED_COV)`: the MAP class's own candidate
    flux and predictive covariance, gathered per source -- a SELECTION,
    reading one of the stacked candidates off by index, never combining
    them (module docstring's "SELECTION, never MIXTURE")."""
    n = committed.shape[0]
    rows = np.arange(n)
    flux_mean = candidate_flux[rows, committed, :]
    flux_cov = flux_cov_stack[rows, committed, :, :]
    return flux_mean, flux_cov


def _build_one(config, region, classes):
    class_list = list(classes)
    st = Stage("impute.posterior", region)
    print("impute.posterior: %s: classes used=%s" % (region, class_list), flush=True)

    prior = table_module.read(config, region)
    name = prior["NAME"]
    n_sources = name.shape[0]
    n_class = np.stack([prior["N_%s" % cls] for cls in class_list], axis=1).astype(np.float64)
    evidence = _read_evidence(config, region, class_list)

    ln_ev_sub_by_class = {}
    ln_ev_class = np.empty((n_sources, len(class_list)), dtype=np.float64)
    flux_mean_stack = np.empty((n_sources, len(class_list), N_BANDS), dtype=np.float64)
    flux_cov_stack = np.empty((n_sources, len(class_list), N_BANDS, N_BANDS), dtype=np.float64)
    n_detected_by_class = {}
    for ci, cls in enumerate(class_list):
        ev = evidence[cls]
        if ev["EVIDENCE"].shape[0] != n_sources:
            raise ValueError(
                "impute.posterior: %r's %r evidence has %d rows, prior table has %d sources"
                % (region, cls, ev["EVIDENCE"].shape[0], n_sources))
        ln_ev_sub = apply_psi(ev["EVIDENCE"], config.fit_beta)
        ln_ev_sub_by_class[cls] = ln_ev_sub
        ln_ev_class[:, ci] = _log_sum_exp(ln_ev_sub, axis=1)
        flux_mean_stack[:, ci, :] = ev["FLUX_MEAN"]
        flux_cov_stack[:, ci, :, :] = ev["FLUX_COV"]
        n_detected_by_class[cls] = ev["N_DETECTED"]

    # N_DETECTED is a per-source catalogue property, not a per-class one
    # (module docstring) -- every class's evidence file must agree, and
    # this was previously unasserted (reading page section 4).
    n_detected = n_detected_by_class[class_list[0]]
    for cls in class_list[1:]:
        if not np.array_equal(n_detected_by_class[cls], n_detected):
            raise ValueError(
                "impute.posterior: %r's N_DETECTED disagrees between class %r and %r "
                "-- a per-source catalogue property must be identical across every "
                "class's evidence file" % (region, class_list[0], cls))

    # THE SIX (or fewer, restricted) CANDIDATES (module docstring): each
    # class's own evidence-weighted FLUX_MEAN, with the source's own
    # measured flux substituted back into any band it was actually
    # detected in -- computed once, per class, never combined across
    # classes.
    own_flux, own_detected = _own_flux_detected(config, region)
    candidate_flux = np.where(own_detected[:, None, :], own_flux[:, None, :], flux_mean_stack)

    p_class = class_posterior(n_class, ln_ev_class, strict=(set(class_list) == set(CLASSES)))
    p_subclass = global_subclass_posterior(p_class, ln_ev_sub_by_class, class_list)
    committed = np.argmax(p_class, axis=1).astype(np.int32)
    flux_imputed, flux_imputed_cov = _committed_flux(committed, candidate_flux, flux_cov_stack)
    entropy_class = entropy(p_class)
    entropy_subclass = entropy(p_subclass)

    out_path = _write(config, region, name, class_list, p_class, p_subclass, committed,
                      candidate_flux, flux_imputed, flux_imputed_cov,
                      entropy_class, entropy_subclass, n_detected)

    mean_p = p_class.mean(axis=0)
    committed_counts = np.bincount(committed, minlength=len(class_list))
    numbers = {"n_sources": n_sources}
    for ci, cls in enumerate(class_list):
        numbers["mean_P_%s" % cls] = float(mean_p[ci])
        numbers["n_%s" % cls] = int(committed_counts[ci])
    st.done(out_path, **numbers)
    return out_path


def _output_path(config, region):
    return config_module.product_path(config, "bms", "impute", "posterior", "source", region=region)


def _write(config, region, name, class_list, p_class, p_subclass, committed,
          candidate_flux, flux_imputed, flux_imputed_cov, entropy_class,
          entropy_subclass, n_detected):
    out_path = _output_path(config, region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.attrs["CLASSES"] = np.array(class_list, dtype="S")
        f.create_dataset("NAME", data=name)
        f.create_dataset("P_CLASS", data=p_class.astype(np.float32))
        f.create_dataset("P_SUBCLASS", data=p_subclass.astype(np.float32))
        f.create_dataset("COMMITTED_CLASS", data=committed)
        # the six (or restricted) per-class candidates, CLASSES order --
        # a SELECTION, not a mixture (module docstring): FLUX_IMPUTED is
        # exactly CANDIDATE_FLUX[:, argmax(P_CLASS), :].
        f.create_dataset("CANDIDATE_FLUX", data=candidate_flux.astype(np.float32))
        f.create_dataset("FLUX_IMPUTED", data=flux_imputed.astype(np.float32))
        f.create_dataset("FLUX_IMPUTED_COV", data=flux_imputed_cov.astype(np.float32))
        f.create_dataset("ENTROPY_CLASS", data=entropy_class.astype(np.float32))
        f.create_dataset("ENTROPY_SUBCLASS", data=entropy_subclass.astype(np.float32))
        f.create_dataset("N_DETECTED", data=n_detected)
    return out_path


def build(config, regions=None, classes=None):
    """Writes the per-region imputed posterior for `regions` (default:
    all thirty), one product per region, from that region's prior table
    and evidence files (module docstring). `classes` (default: all six,
    `fit.run.CLASSES`) restricts the posterior to the classes with an
    evidence file -- an explicit list, never an auto-detected one, so a
    missing class is a deliberate choice, reported at every region's own
    print line (owner ruling 2026-09-06)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    class_list = classes if classes is not None else list(CLASSES)
    for region in region_names:
        _build_one(config, region, class_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="restrict to these classes' evidence files "
                             "(default: all six -- tolerates a subset "
                             "when the others' evidence has not been built yet)")
    args = parser.parse_args()
    cfg = config_module.load(args.config)
    build(cfg, regions=args.regions, classes=args.classes)
