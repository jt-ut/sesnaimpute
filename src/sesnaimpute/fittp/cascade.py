"""The colour cascade on the survey's measured photometry (SPEC_BMSTP_DRAFT.md
section 6.5), and the P10 comparison product it feeds (section 7.3;
IMPLEMENTATION_BMSTP_DRAFT.md section 1.3, P10; section 4 row 2.5).

Gutermuth+2009's colour-cut cascade (`gutcolors.prob.classify_prob`) is run
on every source's curated, measured photometry with both of its masks set
to the real detection pattern (`valid = detected = ORIGIN_FNU == 1`, the
pattern `fit.psi.verdicts` reads): it may form colours only from measured
fluxes, and it follows the branch logic the survey actually saw. The
catalogue's own substitution scheme already carries a completeness-limit
value in every undetected band, so no masking of the flux array itself is
needed here; only the mask tells the cascade which bands to trust.

Its eleven verdict probabilities (`crisp.LABELS` order) are grouped into a
class score `PSI_C(s)` over the fitter's six classes.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.batches import batches
from sesnaimpute.build import run
from sesnaimpute.gutcolors import crisp
from sesnaimpute.gutcolors import prob as gc_prob

#: The fitter's six classes, in the order every P-product's class axis
#: uses (IMPLEMENTATION_BMSTP_DRAFT.md section 1).
CLASSES = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")

#: Each class's own verdict set (SPEC_BMSTP_DRAFT.md section 6.5): `PSI_C(s)`
#: is the probability that the cascade's verdict on this source is one an
#: object of class C would itself receive, so a class's set is the set of
#: verdicts its own population receives from the colour cuts, and two
#: classes' sets may overlap. A bare AGB photosphere reads as a diskless
#: star under Gutermuth's cuts; a dusty AGB envelope reads as a disc
#: source -- Gutermuth et al. 2009 (ApJS 184, 18) and Megeath et al. 2012
#: (AJ 144, 192) both note dusty AGB stars as the principal Class II
#: contaminant, so AGB's set spans all three and overlaps YSO's set in
#: CLASS_II and TRANSITION_DISK.
CONCORDANT_LABELS = {
    "STAR": ("DISKLESS_STAR",),
    "AGB": ("DISKLESS_STAR", "CLASS_II", "TRANSITION_DISK"),
    "PAHC": ("PAH_APERTURE",),
    "GAL": ("AGN", "PAH_GALAXY", "GENERIC_GALAXY"),
    "YSO": ("DEEPLY_EMBEDDED", "CLASS_I", "CLASS_II", "TRANSITION_DISK"),
    "H2S": ("SHOCK_BLOB",),
}

#: One display string per class, for the `LABEL_SETS` attr.
LABEL_SETS = tuple(",".join(CONCORDANT_LABELS[c]) for c in CLASSES)

UNCLASSIFIED_INDEX = crisp.LABEL_INDEX["UNCLASSIFIED"]

#: (11, 6) 0/1 label-to-class matrix. Columns need not be disjoint (AGB
#: overlaps YSO in CLASS_II and TRANSITION_DISK) and need not cover every
#: label (UNCLASSIFIED belongs to none), so a `PSI_CLASS` row does not in
#: general sum to 1 -- see `psi_class`.
GROUP_MATRIX = np.zeros((len(crisp.LABELS), len(CLASSES)), dtype=np.float64)
for _cls, _labels in CONCORDANT_LABELS.items():
    for _lab in _labels:
        GROUP_MATRIX[crisp.LABEL_INDEX[_lab], CLASSES.index(_cls)] = 1.0


def psi_class(prob):
    """`PSI_C(s)`, `(n, 6)`: column `c` is the summed probability of class
    `c`'s own verdict set (`CONCORDANT_LABELS`) -- not a normalised
    distribution, since the sets overlap and do not cover UNCLASSIFIED. A
    source whose most probable verdict is itself UNCLASSIFIED -- the
    detected bands allow no verdict at all (SPEC_BMSTP_DRAFT.md section
    6.5) -- gets a uniform row instead: the cascade's silence there says
    nothing about any class.
    """
    psi = prob @ GROUP_MATRIX
    no_verdict = np.argmax(prob, axis=1) == UNCLASSIFIED_INDEX
    psi[no_verdict] = 1.0 / len(CLASSES)
    return psi


def verdict_code(prob):
    """The SESNA `CLASS` code (`crisp.CLASS_CODE`) of each row's most
    probable verdict; -100 (UNCLASSIFIED's own code) where that verdict
    is itself UNCLASSIFIED."""
    return crisp.CLASS_CODE[np.argmax(prob, axis=1)].astype(np.int16)


def confusion_by_detected_count(verdict_idx, n_detected):
    """`(11, 7)` verdict counts by detected-band count 2..8
    (IMPLEMENTATION_BMSTP_DRAFT.md P10's `CONFUSION_MEASURED`): its 11x6
    shape names six band-count columns, but SESNA's own two-of-eight
    retention admits seven values (2 through 8), so the table is stored
    (11, 7) instead, one column per count.
    """
    table = np.zeros((len(crisp.LABELS), 7), dtype=np.int64)
    for k in range(2, 9):
        sel = n_detected == k
        if sel.any():
            table[:, k - 2] = np.bincount(verdict_idx[sel], minlength=len(crisp.LABELS))
    return table


#: Per-source working-set estimate for the batch loop (rule 10b): the raw
#: flux and sigma reads, (8,) float64 each, plus the cascade's own row-level
#: intermediates (feature dict, 45-column gate table) at a generous margin.
ROW_BYTES = 4096


def build_region(config, region, st):
    """Runs the cascade on `region`'s curated, measured photometry in
    source batches (CODING_RULES_BMSTP.md rule 10b), returning the P10
    MEASURED-half arrays for the whole region.
    """
    path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        n = f["NAME"].shape[0]
        name = f["NAME"][:]

    p_verdict = np.empty((n, len(crisp.LABELS)), dtype=np.float32)
    psi = np.empty((n, len(CLASSES)), dtype=np.float32)
    verdict = np.empty(n, dtype=np.int16)
    n_detected = np.empty(n, dtype=np.int8)

    bounds = list(batches(n, ROW_BYTES))
    for i, (start, stop) in enumerate(bounds):
        with h5py.File(path, "r") as f:
            flux = np.asarray(f["FNU_MJY"][start:stop], dtype=float)
            sigma = np.asarray(f["SIGMA_FNU_MJY"][start:stop], dtype=float)
            origin = np.asarray(f["ORIGIN_FNU"][start:stop])
        detected = origin == 1
        # the real pattern (fit.psi.verdicts): only the survey's own
        # detections may be used, and only they count as detected.
        prob = gc_prob.classify_prob(flux, sigma, valid=detected, detected=detected).prob
        p_verdict[start:stop] = prob.astype(np.float32)
        psi[start:stop] = psi_class(prob).astype(np.float32)
        verdict[start:stop] = verdict_code(prob)
        n_detected[start:stop] = detected.sum(axis=1).astype(np.int8)
        st.tick(i + 1, len(bounds), "batches")

    verdict_idx = np.argmax(p_verdict, axis=1)
    confusion = confusion_by_detected_count(verdict_idx, n_detected)
    return dict(name=name, p_verdict=p_verdict, psi=psi, verdict=verdict,
                n_detected=n_detected, confusion=confusion)


def write_region(path, result):
    """Writes the P10 MEASURED half. The imputed half (`P_VERDICT_IMPUTED`,
    `VERDICT_IMPUTED`) is written later by the classify stage and is left
    absent here, not empty.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("NAME", data=result["name"])
        f.create_dataset("P_VERDICT_MEASURED", data=result["p_verdict"])
        f.create_dataset("PSI_CLASS", data=result["psi"])
        f.create_dataset("VERDICT_MEASURED", data=result["verdict"])
        f.create_dataset("N_DETECTED", data=result["n_detected"])
        f.attrs["GRANULE"] = "source"
        f.attrs["LABELS"] = np.array(crisp.LABELS, dtype="S20")
        f.attrs["CLASSES"] = np.array(CLASSES, dtype="S8")
        f.attrs["LABEL_SETS"] = np.array(LABEL_SETS, dtype="S64")
        f.attrs["CONFUSION_MEASURED"] = result["confusion"]


def _classify_path(config, region):
    return config_module.product_path(
        config, "fittp", "classification", "posterior", "source", region=region)


def build_region_imputed(config, region, st, name, n_detected, verdict_measured):
    """The imputed half (spec sec 6.5, 7.3; IMPLEMENTATION_BMSTP_DRAFT.md
    row 2.5): the cascade run on `classify`'s `FLUX_IMPUTED` with
    `valid = detected = all` -- every band usable, since the imputed SED
    is the MAP class's own candidate, not a measurement, so it carries no
    sigma of its own (`sigma = 0`, the crisp limit of `classify_prob`).
    Runs only once `classify` has written the region (rule 5b: otherwise
    None, the caller leaves the imputed half absent, not a failure --
    the measured half stands on its own).

    `confusion_yso` (spec sec 7.3's second table) is built from the
    MEASURED verdict (`verdict_measured`, this region's own
    `VERDICT_MEASURED`), not the imputed one: the imputed verdict is the
    MAP class's own SED read back (module docstring, sec 6.5), so it
    cannot serve as independent evidence against `P(YSO) > 0.5`.
    `confusion_verdict_map` is the other sec 7.3 table and keeps the
    imputed verdict, which is exactly what it checks -- the library's
    idea of the class against the cascade's idea of the class.
    """
    path = _classify_path(config, region)
    if not os.path.exists(path):
        print("fittp.cascade [%s]: no classify product yet (%s) -- run "
              "'PY sesnaimpute.fittp.classify' after the fit loop, then "
              "'PY sesnaimpute.fittp.cascade' again for the imputed half"
              % (region, path))
        return None

    with h5py.File(path, "r") as f:
        classify_name = f["NAME"][:]
        map_class = f["MAP_CLASS"][:]
        p_yso = f["P_YSO"][:]
        n = f["FLUX_IMPUTED"].shape[0]
    if not np.array_equal(classify_name, name):
        raise ValueError("fittp.cascade [%s]: classify's NAME does not row-align "
                          "with the cascade's own" % region)

    p_verdict_imp = np.empty((n, len(crisp.LABELS)), dtype=np.float32)
    all_true = None
    bounds = list(batches(n, ROW_BYTES))
    for i, (start, stop) in enumerate(bounds):
        with h5py.File(path, "r") as f:
            flux = np.asarray(f["FLUX_IMPUTED"][start:stop], dtype=float)
        sigma = np.zeros_like(flux)
        detected = np.ones(flux.shape, dtype=bool)
        prob = gc_prob.classify_prob(flux, sigma, valid=detected, detected=detected).prob
        p_verdict_imp[start:stop] = prob.astype(np.float32)
        st.tick(i + 1, len(bounds), "batches (imputed)")

    verdict_idx = np.argmax(p_verdict_imp, axis=1)
    verdict_imp = crisp.CLASS_CODE[verdict_idx].astype(np.int16)
    confusion_imputed = confusion_by_detected_count(verdict_idx, n_detected)

    # verdict (imputed) vs MAP class -- spec sec 7.3's first confusion table.
    confusion_verdict_map = np.zeros((len(crisp.LABELS), len(CLASSES)), dtype=np.int64)
    for ci in range(len(CLASSES)):
        sel = map_class == ci
        if sel.any():
            confusion_verdict_map[:, ci] = np.bincount(verdict_idx[sel], minlength=len(crisp.LABELS))

    # cascade's YSO set (MEASURED verdict) vs P(YSO) > 0.5 -- spec sec
    # 7.3's second table, the honest colour-cut YSO set.
    yso_label_idx = [crisp.LABEL_INDEX[lab] for lab in CONCORDANT_LABELS["YSO"]]
    cascade_yso = np.isin(verdict_measured, yso_label_idx)
    pyso_half = p_yso > 0.5
    confusion_yso_measured = np.array([
        [int((~cascade_yso & ~pyso_half).sum()), int((~cascade_yso & pyso_half).sum())],
        [int((cascade_yso & ~pyso_half).sum()), int((cascade_yso & pyso_half).sum())],
    ], dtype=np.int64)

    return dict(p_verdict=p_verdict_imp, verdict=verdict_imp,
                confusion_imputed=confusion_imputed,
                confusion_verdict_imputed_vs_map=confusion_verdict_map,
                confusion_cascade_yso_measured_vs_pyso=confusion_yso_measured)


def write_region_imputed(path, imputed):
    with h5py.File(path, "a") as f:
        for name in ("P_VERDICT_IMPUTED", "VERDICT_IMPUTED"):
            if name in f:
                del f[name]
        f.create_dataset("P_VERDICT_IMPUTED", data=imputed["p_verdict"])
        f.create_dataset("VERDICT_IMPUTED", data=imputed["verdict"])
        f.attrs["CONFUSION_IMPUTED"] = imputed["confusion_imputed"]
        # sec 7.3's first table: imputed verdict against the MAP class.
        f.attrs["CONFUSION_VERDICT_IMPUTED_VS_MAP"] = imputed["confusion_verdict_imputed_vs_map"]
        # sec 7.3's second table: MEASURED verdict's YSO set against P(YSO) > 0.5.
        f.attrs["CONFUSION_CASCADE_YSO_MEASURED_VS_PYSO"] = imputed["confusion_cascade_yso_measured_vs_pyso"]


def build(config, regions=None):
    """Writes `fittp/classification/cascade_classification_source[__R].hdf5`
    for `regions` (default all thirty), one file per region
    (IMPLEMENTATION_BMSTP_DRAFT.md section 1.3, P10).
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        with progress.Stage("fittp.cascade", region) as st:
            result = build_region(config, region, st)
            path = config_module.product_path(
                config, "fittp", "classification", "cascade", "source", region=region)
            write_region(path, result)

            n = result["name"].shape[0]
            # each PSI_CLASS column is the sum of its own label set's
            # P_VERDICT_MEASURED probabilities, by construction.
            label_sum_err = 0.0
            classified = result["verdict"] != -100
            for c, labels in CONCORDANT_LABELS.items():
                idx = [crisp.LABEL_INDEX[lab] for lab in labels]
                expect = result["p_verdict"][:, idx].sum(axis=1)
                err = np.max(np.abs(result["psi"][classified, CLASSES.index(c)]
                                     - expect[classified])) if classified.any() else 0.0
                label_sum_err = max(label_sum_err, float(err))
            p_defined = classified
            p_sum_err = float(np.max(np.abs(result["p_verdict"][p_defined].sum(axis=1) - 1.0))) \
                if p_defined.any() else 0.0
            uniform_ok = bool(np.allclose(
                result["psi"][~p_defined], 1.0 / len(CLASSES), atol=1e-6)) if (~p_defined).any() else True

            st.done(path, n=n, n_classified=int(p_defined.sum()),
                     psi_column_err=label_sum_err, p_verdict_sum_err=p_sum_err,
                     unclassified_rows_uniform=uniform_ok)
            print(f"fittp.cascade {region}: verdict counts by detected-band count "
                  f"(rows=verdict in LABELS order, cols=2..8):\n{result['confusion']}")

        with progress.Stage("fittp.cascade.imputed", region) as st:
            imputed = build_region_imputed(config, region, st, result["name"],
                                            result["n_detected"], result["verdict"])
            if imputed is not None:
                write_region_imputed(path, imputed)
                st.done(path, n=imputed["p_verdict"].shape[0])
                print(f"fittp.cascade {region}: imputed-verdict-vs-MAP confusion (rows=verdict, "
                      f"cols=MAP class {CLASSES}):\n{imputed['confusion_verdict_imputed_vs_map']}")
                print(f"fittp.cascade {region}: MEASURED-verdict cascade YSO set vs P(YSO)>0.5 "
                      f"[[not-not, not-yso],[cascade-not, cascade-yso]]:"
                      f"\n{imputed['confusion_cascade_yso_measured_vs_pyso']}")
            else:
                st.done(None, n=0)


if __name__ == "__main__":
    run(build)
