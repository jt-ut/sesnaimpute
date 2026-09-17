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


def confusion_verdict_map_by_count(verdict_idx, map_class, n_detected, n_classes):
    """`(7, 11, n_classes)`: sec 7.3's imputed-verdict-vs-MAP table, one
    slice per detected-band count 2..8, so the two axes can still be
    crossed with the count after the fact instead of only their region
    total (R3 D4 -- the summed-over-count table hides exactly the split
    sec 6.5 says matters, the cascade abstaining on every two-band source)."""
    table = np.zeros((7, len(crisp.LABELS), n_classes), dtype=np.int64)
    for k in range(2, 9):
        sel_k = n_detected == k
        for ci in range(n_classes):
            sel = sel_k & (map_class == ci)
            if sel.any():
                table[k - 2, :, ci] = np.bincount(verdict_idx[sel], minlength=len(crisp.LABELS))
    return table


def confusion_yso_by_count(cascade_yso, pyso_half, n_detected):
    """`(7, 2, 2)`: sec 7.3's cascade-YSO-set-vs-`P(YSO)>0.5` table, one
    2x2 slice per detected-band count 2..8, rows/cols `[not, yso]` (R3 D4)."""
    table = np.zeros((7, 2, 2), dtype=np.int64)
    for k in range(2, 9):
        sel = n_detected == k
        table[k - 2, 0, 0] = int((sel & ~cascade_yso & ~pyso_half).sum())
        table[k - 2, 0, 1] = int((sel & ~cascade_yso & pyso_half).sum())
        table[k - 2, 1, 0] = int((sel & cascade_yso & ~pyso_half).sum())
        table[k - 2, 1, 1] = int((sel & cascade_yso & pyso_half).sum())
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


def _label_set_vote(verdict_idx):
    """`(n, 6)`: one reading's own unit, split equally over the classes
    whose `CONCORDANT_LABELS` set holds `verdict_idx`'s own most probable
    label (BATCH0917 item 4) -- `GROUP_MATRIX` row `v` is already that
    label's 0/1 membership over the six classes (module docstring, `psi_
    class`), so the row divided by its own sum is the split; a row that
    sums to zero (UNCLASSIFIED, which no class's set holds) is a clean
    ABSTAIN, left at zero rather than divided."""
    row = GROUP_MATRIX[verdict_idx]
    row_sum = row.sum(axis=1, keepdims=True)
    return np.divide(row, row_sum, out=np.zeros_like(row), where=row_sum > 0)


def _prior_leaning_vote(config, region, hpx512_source):
    """`(n,)` int64, `CLASSES` index of `argmax_C N_CAT_C` at the source's
    own pixel (BATCH0917 item 4, reading 1): `bmstp.atlas`'s own prior
    atlas (`bmstp/atlas/prior_atlas_hpx512__<R>.hdf5`), the source's pixel
    found the same way `fittp.atlas.build_region` finds it -- a sorted
    search on the atlas's own `HPX_PIX_512` against `hpx512_source`
    (`bmstp.density`'s own `HPX_512` column, P1). -1 (ABSTAIN) where the
    source's pixel carries no row there (should not occur inside the
    admitted footprint, but read defensively rather than assumed)."""
    path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        n_cat = np.stack([np.asarray(f["N_CAT_%s" % c][:], dtype=np.float64) for c in CLASSES], axis=1)
    order = np.argsort(pix)
    pix_sorted = pix[order]
    n_cat_sorted = n_cat[order]
    n = hpx512_source.shape[0]
    leaning = np.full(n, -1, dtype=np.int64)
    if pix_sorted.size:
        idx = np.clip(np.searchsorted(pix_sorted, hpx512_source), 0, pix_sorted.size - 1)
        valid = pix_sorted[idx] == hpx512_source
        leaning[valid] = np.argmax(n_cat_sorted[idx[valid]], axis=1)
    return leaning


def _gaia_leaning_vote(config, region, name):
    """`(n,)` int64, `CLASSES` index of `argmax_C TOPK_LN_GAMMA[:, 0]` over
    the six fit files (BATCH0917 item 4, reading 2): -1 (ABSTAIN) where the
    six values are all equal (no Gaia datum -- `fittp.gaia.GaiaTerm.
    ln_gamma`'s own convention for a source with no counterpart is
    `ln Gamma = 0` for every model of every class, so this falls out of
    the same equality test) or where any of the six is not finite (a
    flagged source's `TOPK_LN_GAMMA` is NaN, module docstring's `_empty_
    row`)."""
    n = name.shape[0]
    vals = np.empty((n, len(CLASSES)), dtype=np.float64)
    for ci, cls in enumerate(CLASSES):
        path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
        with h5py.File(path, "r") as f:
            cls_name = f["NAME"][:]
            vals[:, ci] = np.asarray(f["TOPK_LN_GAMMA"][:, 0], dtype=np.float64)
        if not np.array_equal(cls_name, name):
            raise ValueError("fittp.cascade [%s]: %s's NAME does not row-align "
                              "with the cascade's own" % (region, path))
    abstain = np.any(~np.isfinite(vals), axis=1) | np.all(vals == vals[:, [0]], axis=1)
    leaning = np.where(abstain, -1, np.argmax(vals, axis=1))
    return leaning


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

    # PSI_VOTES' own Gaia-leaning reading (BATCH0917 item 4) needs this
    # run's own fit files, which carry TOPK_LN_GAMMA only once THIS
    # region's sweep has (re)written them: an older classify product left
    # on disk from a PRIOR run (rule 5c: this line re-runs in place) would
    # otherwise read a stale fit file that has no such column. Same
    # "not ready yet" treatment as the classify guard above, not a version
    # stamp -- the fit files this call is ABOUT to read are simply missing
    # the column it needs.
    for cls in CLASSES:
        fit_path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
        with h5py.File(fit_path, "r") as f:
            has_gamma = "TOPK_LN_GAMMA" in f
        if not has_gamma:
            print("fittp.cascade [%s]: %s has no TOPK_LN_GAMMA yet (stale, pre-BATCH0917) -- "
                  "run the fit loop (PY sesnaimpute.fittp.sweep) and classify again, then "
                  "'PY sesnaimpute.fittp.cascade' again for the imputed half"
                  % (region, fit_path))
            return None

    with h5py.File(path, "r") as f:
        classify_name = f["NAME"][:]
        map_class = f["MAP_CLASS"][:]
        p_yso = f["P_YSO"][:]
        n = f["FLUX_IMPUTED"].shape[0]
    if not np.array_equal(classify_name, name):
        raise ValueError("fittp.cascade [%s]: classify's NAME does not row-align "
                          "with the cascade's own" % region)

    # PSI_VOTES' own measured-half reading (BATCH0917 item 4, reading 3):
    # this region's own P_VERDICT_MEASURED, already on disk on the CASCADE
    # product (not `path` above, the classify one) -- `write_region`, the
    # first cascade run this region made, earlier in this same `build()` call.
    cascade_path = config_module.product_path(
        config, "fittp", "classification", "cascade", "source", region=region)
    with h5py.File(cascade_path, "r") as f:
        p_verdict_measured = np.asarray(f["P_VERDICT_MEASURED"][:])
    verdict_idx_measured = np.argmax(p_verdict_measured, axis=1)

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

    # verdict (imputed) vs MAP class -- spec sec 7.3's first confusion
    # table, per detected-band count (R3 D4).
    confusion_verdict_map = confusion_verdict_map_by_count(
        verdict_idx, map_class, n_detected, len(CLASSES))

    # cascade's YSO set (MEASURED verdict) vs P(YSO) > 0.5 -- spec sec
    # 7.3's second table, the honest colour-cut YSO set, per detected-band
    # count (R3 D4). `verdict_measured` is SESNA's own CLASS code
    # (`crisp.CLASS_CODE`); `CONCORDANT_LABELS["YSO"]` names `crisp.LABELS`
    # indices, so the comparison goes through `CLASS_CODE` explicitly
    # rather than relying on the two vocabularies' rows coinciding by
    # construction (module note; any reordering of `constants.
    # GUTERMUTH_LABELS` would otherwise break this silently).
    yso_label_idx = [crisp.LABEL_INDEX[lab] for lab in CONCORDANT_LABELS["YSO"]]
    yso_class_codes = crisp.CLASS_CODE[yso_label_idx]
    cascade_yso = np.isin(verdict_measured, yso_class_codes)
    pyso_half = p_yso > 0.5
    confusion_yso_measured = confusion_yso_by_count(cascade_yso, pyso_half, n_detected)

    # PSI_VOTES, ENTROPY_PSI_VOTES (BATCH0917 item 4): four readings, each
    # casting one unit (split equally where a reading's own set is not a
    # single class), the prior atlas and the fit files' own Gaia term
    # already on disk beside the posterior product this run just opened.
    density_path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    with h5py.File(density_path, "r") as f:
        density_name = f["NAME"][:]
        hpx512_source = np.asarray(f["HPX_512"][:], dtype=np.int64)
    if not np.array_equal(density_name, name):
        raise ValueError("fittp.cascade [%s]: bmstp.density's NAME does not row-align "
                          "with the cascade's own" % region)

    prior_leaning = _prior_leaning_vote(config, region, hpx512_source)
    gaia_leaning = _gaia_leaning_vote(config, region, name)

    votes = np.zeros((n, len(CLASSES)), dtype=np.float64)
    for leaning in (prior_leaning, gaia_leaning):
        cast = leaning >= 0
        votes[cast, leaning[cast]] += 1.0
    votes += _label_set_vote(verdict_idx_measured)
    votes += _label_set_vote(verdict_idx)
    row_sum = votes.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_votes = votes / row_sum[:, None]
        entropy = -np.nansum(np.where(p_votes > 0, p_votes * np.log(p_votes), 0.0), axis=1)
    entropy = np.where(row_sum > 0, entropy, np.nan)

    return dict(p_verdict=p_verdict_imp, verdict=verdict_imp,
                confusion_imputed=confusion_imputed,
                confusion_verdict_imputed_vs_map=confusion_verdict_map,
                confusion_cascade_yso_measured_vs_pyso=confusion_yso_measured,
                psi_votes=votes.astype(np.float32), entropy_psi_votes=entropy.astype(np.float32))


def write_region_imputed(path, imputed):
    with h5py.File(path, "a") as f:
        for name in ("P_VERDICT_IMPUTED", "VERDICT_IMPUTED", "PSI_VOTES", "ENTROPY_PSI_VOTES"):
            if name in f:
                del f[name]
        f.create_dataset("P_VERDICT_IMPUTED", data=imputed["p_verdict"])
        f.create_dataset("VERDICT_IMPUTED", data=imputed["verdict"])
        # BATCH0917 item 4: (n, 6) CLASSES-order votes (0-4 per row) and the
        # row's own entropy (nats), normalised to 1, NaN where nothing voted.
        f.create_dataset("PSI_VOTES", data=imputed["psi_votes"])
        f.create_dataset("ENTROPY_PSI_VOTES", data=imputed["entropy_psi_votes"])
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
