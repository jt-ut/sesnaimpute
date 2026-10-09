"""The colour cascade on the survey's measured photometry (SPEC_BMSTP_DRAFT.md
section 6.5), and the P10 product it feeds (IMPLEMENTATION_BMSTP_DRAFT.md
section 1.3, P10; section 4 row 2.5). The product stores the verdicts and
their probabilities; the confusion tables of section 7.3 are computed and
printed by `fittp.check`, not stored here.

Gutermuth+2009's colour-cut cascade (`gutcolors.prob.classify_prob`) is run
on every source's curated, measured photometry with both of its masks set
to the real detection pattern (`valid = detected = ORIGIN_FNU == 1`, the
pattern `fit.psi.verdicts` reads): it may form colours only from measured
fluxes, and it follows the branch logic the survey actually saw. The
catalogue's own substitution scheme already carries a completeness-limit
value in every undetected band, so no masking of the flux array itself is
needed here; only the mask tells the cascade which bands to trust.
"""

import os

import h5py
import numpy as np
from sesnaimpute import definitions

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.batches import batches
from sesnaimpute.gutcolors import crisp
from sesnaimpute.gutcolors import prob as gc_prob

#: `config.product_path`'s own stem for this module's one product
#: (`attrs_registry.REGISTRY`'s key, CODING_RULES_BMSTP.md rule 5).
_STEM = "cascade_classification_source"

#: The fitter's six classes, in the order every P-product's class axis
#: uses (IMPLEMENTATION_BMSTP_DRAFT.md section 1).
CLASSES = tuple(c.code for c in definitions.CLASSES)

#: Each class's own verdict set (SPEC_BMSTP_DRAFT.md section 6.5): the set
#: of verdicts an object of class C itself receives from the colour cuts,
#: so a class's set is the set of verdicts its own population receives,
#: and two classes' sets may overlap. A bare AGB photosphere reads as a
#: diskless star under Gutermuth's cuts; a dusty AGB envelope reads as a
#: disc source -- Gutermuth et al. 2009 (ApJS 184, 18) and Megeath et al.
#: 2012 (AJ 144, 192) both note dusty AGB stars as the principal Class II
#: contaminant, so AGB's set spans all three and overlaps YSO's set in
#: CLASS_II and TRANSITION_DISK. Its only use in this module is the vote
#: split (`_label_set_vote`): a reading's own unit is split equally over
#: the classes whose set holds the cascade's most probable verdict.
CONCORDANT_LABELS = {
    "STAR": ("DISKLESS_STAR",),
    "AGB": ("DISKLESS_STAR", "CLASS_II", "TRANSITION_DISK"),
    "PAHC": ("PAH_APERTURE",),
    "GAL": ("AGN", "PAH_GALAXY", "GENERIC_GALAXY"),
    "YSO": ("DEEPLY_EMBEDDED", "CLASS_I", "CLASS_II", "TRANSITION_DISK"),
    "H2S": ("SHOCK_BLOB",),
}

#: `UNITS`/`READING` for every dataset here (CODING_RULES_BMSTP.md rule 5),
#: measured and imputed halves both, live in `attrs_registry.REGISTRY`,
#: keyed by `(_STEM, name)`.


def verdict_code(prob):
    """The SESNA `CLASS` code (`crisp.CLASS_CODE`) of each row's most
    probable verdict; -100 (UNCLASSIFIED's own code) where that verdict
    is itself UNCLASSIFIED."""
    return crisp.CLASS_CODE[np.argmax(prob, axis=1)].astype(np.int16)


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
        verdict[start:stop] = verdict_code(prob)
        n_detected[start:stop] = detected.sum(axis=1).astype(np.int8)
        st.tick(i + 1, len(bounds), "batches")

    return dict(name=name, p_verdict=p_verdict, verdict=verdict, n_detected=n_detected)


def write_region(path, result):
    """Writes the P10 MEASURED half. The imputed half (`VERDICT_IMPUTED`,
    `PSI_VOTES`, `ENTROPY_PSI_VOTES`) is written later, once `classify`
    has run, and is left absent here, not empty.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, field in (("NAME", "name"), ("P_VERDICT_MEASURED", "p_verdict"),
                           ("VERDICT_MEASURED", "verdict"), ("N_DETECTED", "n_detected")):
            build_module.write_dataset(f, key, result[field], *REGISTRY[(_STEM, key)])
        f.attrs["GRANULE"] = "source"
        f.attrs["LABELS"] = np.array(crisp.LABELS, dtype="S20")
        f.attrs["CLASSES"] = np.array(CLASSES, dtype="S8")


def _classify_path(config, region):
    return config_module.product_path(
        config, "fittp", "classification", "posterior", "source", region=region)


#: (11, 6) 0/1 label-to-class matrix built from `CONCORDANT_LABELS`: row
#: `v` is label `v`'s own membership over the six classes. Private to
#: `_label_set_vote`, its only reader.
_LABEL_CLASS_MATRIX = np.zeros((len(crisp.LABELS), len(CLASSES)), dtype=np.float64)
for _cls, _labels in CONCORDANT_LABELS.items():
    for _lab in _labels:
        _LABEL_CLASS_MATRIX[crisp.LABEL_INDEX[_lab], CLASSES.index(_cls)] = 1.0


def _label_set_vote(verdict_idx):
    """`(n, 6)`: one reading's own unit, split equally over the classes
    whose `CONCORDANT_LABELS` set holds `verdict_idx`'s own most probable
    label -- `_LABEL_CLASS_MATRIX` row `v` is already that label's 0/1
    membership over the six classes, so the row divided by its own sum is
    the split; a row that sums to zero (UNCLASSIFIED, which no class's
    set holds) is a clean ABSTAIN, left at zero rather than divided."""
    row = _LABEL_CLASS_MATRIX[verdict_idx]
    row_sum = row.sum(axis=1, keepdims=True)
    return np.divide(row, row_sum, out=np.zeros_like(row), where=row_sum > 0)


def _prior_leaning_vote(config, region, hpx512_source):
    """`(n,)` int64, `CLASSES` index of `argmax_C N_CAT_C` at the source's
    own pixel (spec sec 6.5's vote readings): `bmstp.atlas`'s own prior
    atlas (`bmstp/atlas/prior_atlas_hpx512__<R>.hdf5`), the source's pixel
    found the same way `fittp.atlas.build_region` finds it -- a sorted
    search on the atlas's own `HPX_PIX_512` against `hpx512_source`
    (`bmstp.density`'s own `HPX_512` column, P1). -1 (ABSTAIN) where the
    source's pixel carries no row there (should not occur inside the
    admitted footprint, but read defensively rather than assumed)."""
    path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    if not os.path.exists(path):
        raise RuntimeError(
            "fittp.cascade --imputed [%s]: missing %s -- run RUNBOOKtp.sh's "
            "'PY sesnaimpute.bmstp.atlas' line first" % (region, path))
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
    """`(n,)` int64, `CLASSES` index of `argmax_C TOPK_LN_GAMMA_ML[:, 0]` over
    the six fit files (spec sec 6.5's vote readings): -1 (ABSTAIN) where the
    six values are all equal (no Gaia datum -- `fittp.gaia.GaiaTerm.
    ln_gamma`'s own convention for a source with no counterpart is
    `ln Gamma = 0` for every model of every class, so this falls out of
    the same equality test) or where any of the six is not finite (a
    flagged source's `TOPK_LN_GAMMA_ML` is NaN, `fittp.sweep._empty_row`)."""
    n = name.shape[0]
    vals = np.empty((n, len(CLASSES)), dtype=np.float64)
    for ci, cls in enumerate(CLASSES):
        path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
        if not os.path.exists(path):
            raise RuntimeError(
                "fittp.cascade --imputed [%s]: missing %s -- run RUNBOOKtp.sh's "
                "'PY sesnaimpute.fittp.sweep --classes \"%s\"' line first" % (region, path, cls))
        with h5py.File(path, "r") as f:
            cls_name = f["NAME"][:]
            vals[:, ci] = np.asarray(f["TOPK_LN_GAMMA_ML"][:, 0], dtype=np.float64)
        if not np.array_equal(cls_name, name):
            raise ValueError("fittp.cascade --imputed [%s]: %s's NAME does not row-align "
                              "with the cascade's own" % (region, path))
    abstain = np.any(~np.isfinite(vals), axis=1) | np.all(vals == vals[:, [0]], axis=1)
    leaning = np.where(abstain, -1, np.argmax(vals, axis=1))
    return leaning


def build_region_imputed(config, region, st):
    """The imputed half (spec sec 6.5, 7.3; IMPLEMENTATION_BMSTP_DRAFT.md
    row 2.5): the cascade run on `classify`'s `LOG10_FLUX_IMPUTED`, `10**x`
    the median flux in mJy (SPEC_BMSTP_DRAFT.md sec 7.1), with `valid =
    detected = all` -- every band usable, since the imputed SED is the MAP
    class's own candidate, not a measurement, so it carries no sigma of
    its own (`sigma = 0`, the crisp limit of `classify_prob`).
    Reads the measured half (`NAME`, `P_VERDICT_MEASURED`) from the
    cascade product already on disk, and `classify`'s own
    `LOG10_FLUX_IMPUTED`; a missing input fails with one
    sentence naming the RUNBOOK line that makes it (rule 5b), nothing
    checks its content. The sec 7.3 confusion tables this half used to
    carry (imputed verdict against the MAP class; the MEASURED verdict's
    YSO set against `P(YSO) > 0.5`) are computed and printed by
    `fittp.check` from this product and `classify`'s own `MAP_CLASS`/
    `P_YSO`, not stored here.

    PSI_VOTES/ENTROPY_PSI_VOTES (spec sec 6.5): four readings, each
    casting one unit split over the classes it implicates, or abstaining
    -- the prior atlas's own leaning at the source's pixel
    (`_prior_leaning_vote`), the Gaia term's own leaning
    (`_gaia_leaning_vote`, `TOPK_LN_GAMMA_ML`), and the two cascade verdicts
    above (`_label_set_vote`).
    """
    cascade_path = config_module.product_path(
        config, "fittp", "classification", "cascade", "source", region=region)
    if not os.path.exists(cascade_path):
        raise RuntimeError(
            "fittp.cascade --imputed [%s]: missing %s -- run RUNBOOKtp.sh's "
            "'PY sesnaimpute.fittp.cascade' line first" % (region, cascade_path))
    with h5py.File(cascade_path, "r") as f:
        name = f["NAME"][:]
        p_verdict_measured = np.asarray(f["P_VERDICT_MEASURED"][:])
    verdict_idx_measured = np.argmax(p_verdict_measured, axis=1)

    path = _classify_path(config, region)
    if not os.path.exists(path):
        raise RuntimeError(
            "fittp.cascade --imputed [%s]: missing %s -- run RUNBOOKtp.sh's "
            "'PY sesnaimpute.fittp.classify' line first" % (region, path))
    with h5py.File(path, "r") as f:
        classify_name = f["NAME"][:]
        n = f["LOG10_FLUX_IMPUTED"].shape[0]
    if not np.array_equal(classify_name, name):
        raise ValueError("fittp.cascade --imputed [%s]: classify's NAME does not row-align "
                          "with the cascade's own" % region)

    p_verdict_imp = np.empty((n, len(crisp.LABELS)), dtype=np.float32)
    bounds = list(batches(n, ROW_BYTES))
    for i, (start, stop) in enumerate(bounds):
        with h5py.File(path, "r") as f:
            log10_flux = np.asarray(f["LOG10_FLUX_IMPUTED"][start:stop], dtype=float)
        # section 7.1's median flux, 10**x, where the linear mean was read
        # before (identity 7: the imputed cascade half's verdicts move).
        flux = 10.0 ** log10_flux
        sigma = np.zeros_like(flux)
        detected = np.ones(flux.shape, dtype=bool)
        prob = gc_prob.classify_prob(flux, sigma, valid=detected, detected=detected).prob
        p_verdict_imp[start:stop] = prob.astype(np.float32)
        st.tick(i + 1, len(bounds), "batches (imputed)")

    verdict_idx = np.argmax(p_verdict_imp, axis=1)
    verdict_imp = crisp.CLASS_CODE[verdict_idx].astype(np.int16)

    # PSI_VOTES, ENTROPY_PSI_VOTES (spec sec 6.5): four readings, each
    # casting one unit (split equally where a reading's own set is not a
    # single class), the prior atlas and the fit files' own Gaia term
    # already on disk beside the posterior product this run just opened.
    density_path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    if not os.path.exists(density_path):
        raise RuntimeError(
            "fittp.cascade --imputed [%s]: missing %s -- run RUNBOOKtp.sh's "
            "'PY sesnaimpute.bmstp.density' line first" % (region, density_path))
    with h5py.File(density_path, "r") as f:
        density_name = f["NAME"][:]
        hpx512_source = np.asarray(f["HPX_512"][:], dtype=np.int64)
    if not np.array_equal(density_name, name):
        raise ValueError("fittp.cascade --imputed [%s]: bmstp.density's NAME does not row-align "
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
        # divided by its own maximum, ln of the number of classes, so the
        # value reads 0 (the voters agreed) to 1 (they spread evenly).
        entropy = -np.nansum(np.where(p_votes > 0, p_votes * np.log(p_votes), 0.0),
                              axis=1) / np.log(len(CLASSES))
    entropy = np.where(row_sum > 0, entropy, np.nan)

    return dict(verdict=verdict_imp,
                psi_votes=votes.astype(np.float32), entropy_psi_votes=entropy.astype(np.float32))


def write_region_imputed(path, imputed):
    with h5py.File(path, "a") as f:
        for name in ("VERDICT_IMPUTED", "PSI_VOTES", "ENTROPY_PSI_VOTES"):
            if name in f:
                del f[name]
        build_module.write_dataset(f, "VERDICT_IMPUTED", imputed["verdict"], *REGISTRY[(_STEM, "VERDICT_IMPUTED")])
        # (n, 6) CLASSES-order votes (0-4 per row) and the row's own
        # entropy (nats), normalised to 1, NaN where nothing voted.
        build_module.write_dataset(f, "PSI_VOTES", imputed["psi_votes"], *REGISTRY[(_STEM, "PSI_VOTES")])
        build_module.write_dataset(f, "ENTROPY_PSI_VOTES", imputed["entropy_psi_votes"],
                                    *REGISTRY[(_STEM, "ENTROPY_PSI_VOTES")])


def _path(config, region):
    return config_module.product_path(
        config, "fittp", "classification", "cascade", "source", region=region)


def build(config, regions=None, imputed=False):
    """Writes `fittp/classification/cascade_classification_source[__R].hdf5`
    for `regions` (default all thirty), one file per region
    (IMPLEMENTATION_BMSTP_DRAFT.md section 1.3, P10).

    Without `imputed` (the `--imputed` CLI flag): the MEASURED half only,
    reading the region's curated catalogue alone -- never a fit file or the
    posterior product. With it: the IMPUTED half only (`VERDICT_IMPUTED`,
    `PSI_VOTES`/`ENTROPY_PSI_VOTES`), reading the measured half from the
    cascade product already on disk, `classify`'s own `LOG10_FLUX_IMPUTED`,
    the prior atlas and the fit files' own Gaia term. A missing input fails
    with one sentence naming the RUNBOOK line that makes it (rule 5b);
    nothing checks its content. The sec 7.3 confusion tables are printed
    by `fittp.check`, not here.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        if not imputed:
            with progress.Stage("fittp.cascade", region) as st:
                result = build_region(config, region, st)
                path = _path(config, region)
                write_region(path, result)

                n = result["name"].shape[0]
                classified = result["verdict"] != -100
                p_sum_err = float(np.max(np.abs(result["p_verdict"][classified].sum(axis=1) - 1.0))) \
                    if classified.any() else 0.0
                st.done(path, n=n, n_classified=int(classified.sum()), p_verdict_sum_err=p_sum_err)
        else:
            with progress.Stage("fittp.cascade.imputed", region) as st:
                result = build_region_imputed(config, region, st)
                path = _path(config, region)
                write_region_imputed(path, result)
                st.done(path, n=result["verdict"].shape[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--imputed", action="store_true",
                         help="the imputed half only (VERDICT_IMPUTED, PSI_VOTES/"
                              "ENTROPY_PSI_VOTES): reads the measured half already on disk and "
                              "classify's own LOG10_FLUX_IMPUTED. Without this flag: the measured "
                              "half only, never opening a fit file or the posterior product.")
    args = parser.parse_args()
    cfg = config_module.load(args.config)
    build(cfg, regions=args.regions, imputed=args.imputed)
