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
class posterior `PSI_C(s)` over the fitter's six classes.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.batches import batches
from sesnaimpute.build import run
from sesnaimpute.gutcolors import crisp
from sesnaimpute.gutcolors import prob as gc_prob

#: The fitter's six classes, in the order every P-product's class axis
#: uses (IMPLEMENTATION_BMSTP_DRAFT.md section 1).
CLASSES = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")

#: Which of Gutermuth's eleven verdict categories (`crisp.LABELS`) count
#: toward each of the six classes (SPEC_BMSTP_DRAFT.md section 6.5, "the
#: eleven labels are grouped by class"). AGB has no category of its own in
#: Gutermuth's scheme -- a dusty evolved-star photosphere is not a case his
#: cascade separates -- so it is assigned the UNCLASSIFIED verdict rather
#: than an empty set: every one of the eleven labels is covered exactly
#: once below, so `PSI_C(s)` sums to 1 by construction for a classified
#: source. (This differs from `fit.psi.CONCORDANT_LABELS`, which leaves
#: AGB's set empty and forces its per-model concordance to 1: that Psi is
#: an independent per-class factor on a likelihood, not a distribution
#: over classes, so it need not partition the eleven labels.)
CONCORDANT_LABELS = {
    "STAR": ("DISKLESS_STAR",),
    "AGB": ("UNCLASSIFIED",),
    "PAHC": ("PAH_APERTURE",),
    "GAL": ("AGN", "PAH_GALAXY", "GENERIC_GALAXY"),
    "YSO": ("DEEPLY_EMBEDDED", "CLASS_I", "CLASS_II", "TRANSITION_DISK"),
    "H2S": ("SHOCK_BLOB",),
}

UNCLASSIFIED_INDEX = crisp.LABEL_INDEX["UNCLASSIFIED"]

#: (11, 6) one-hot label-to-class grouping matrix.
GROUP_MATRIX = np.zeros((len(crisp.LABELS), len(CLASSES)), dtype=np.float64)
for _cls, _labels in CONCORDANT_LABELS.items():
    for _lab in _labels:
        GROUP_MATRIX[crisp.LABEL_INDEX[_lab], CLASSES.index(_cls)] = 1.0


def psi_class(prob):
    """`PSI_C(s)`, `(n, 6)`: `prob` (n, 11) grouped by class through
    `GROUP_MATRIX`. A source whose most probable verdict is itself
    UNCLASSIFIED -- the detected bands allow no verdict at all
    (SPEC_BMSTP_DRAFT.md section 6.5) -- gets a uniform row instead: the
    cascade's silence there says nothing about any class, not just the
    AGB slot the grouping above would otherwise hand it.
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


def build_region(config, region):
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

    for start, stop in batches(n, ROW_BYTES):
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
        f.attrs["CONFUSION_MEASURED"] = result["confusion"]


def build(config, regions=None):
    """Writes `fittp/classification/cascade_fittp_source[__R].hdf5` for
    `regions` (default all thirty), one file per region
    (IMPLEMENTATION_BMSTP_DRAFT.md section 1.3, P10). The product's
    directory ("classification") and its filename's source token
    ("fittp", the subpackage's own name, not a data provenance) differ, as
    P11's atlas product does too -- `config.product_path` assumes the two
    are the same, so the path is built directly here instead.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        result = build_region(config, region)
        stem = f"cascade_fittp_source__{region}"
        path = f"{config.data_root}/fittp/classification/{stem}.hdf5"
        write_region(path, result)

        n = result["name"].shape[0]
        psi_sum_err = np.max(np.abs(result["psi"].sum(axis=1) - 1.0))
        p_defined = result["verdict"] != -100
        p_sum_err = np.max(np.abs(result["p_verdict"][p_defined].sum(axis=1) - 1.0)) \
            if p_defined.any() else 0.0
        argmax_class = np.argmax(result["psi"], axis=1)
        argmax_verdict = np.argmax(result["p_verdict"], axis=1)
        verdict_class = np.argmax(GROUP_MATRIX[argmax_verdict], axis=1)
        n_defined = int(p_defined.sum())
        n_match = int(np.sum(argmax_class[p_defined] == verdict_class[p_defined])) \
            if n_defined else 0
        uniform_ok = np.allclose(
            result["psi"][~p_defined], 1.0 / len(CLASSES), atol=1e-6) if (~p_defined).any() else True
        print(f"cascade {region}: n={n} PSI_CLASS max|sum-1|={psi_sum_err:.2e} "
              f"P_VERDICT max|sum-1|={p_sum_err:.2e} "
              f"argmax-class-matches-verdict={n_match}/{n_defined} "
              f"unclassified-rows-uniform={uniform_ok}")
        print(f"cascade {region}: verdict counts by detected-band count (rows=verdict, "
              f"cols=2..8):\n{result['confusion']}")


if __name__ == "__main__":
    run(build)
