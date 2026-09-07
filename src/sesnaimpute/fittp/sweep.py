"""One {region, class}: the class evidence, subclass evidence and top-K
posterior record of SPEC_BMSTP_DRAFT.md section 1.3
(IMPLEMENTATION_BMSTP_DRAFT.md section 1.3 P7, section 4 row 2.4).

Per source and template `theta`, the template's log posterior weight is
the three-factor sum of section 1.3:

    ln w_theta = ln <Lambda_C>_s(theta)   -- prior_reader.ln_prior, section 4.2
               + ln L_hat_s(theta)         -- likelihood.fit's chi2 and non-detection term
               + ln Gamma_s(theta)         -- gaia.GaiaTerm.ln_gamma, section 6.4

(the library-sampling weight is already folded into the prior's own
template-weight factors, section 6.3, so no separate library weight is
added here). The class evidence is `logsumexp` over every template; the
subclass evidence restricts that sum to one subclass's templates
(`-inf` where a subclass has none).

Region and class are read once (the register, the prior product, the
Gaia term); sources are then swept in batches of `[fit] batch_size`, and
each batch in blocks of `~[fit] block_budget_mb` sources
(`likelihood.block_size`): per block, the fit, the prior read, the
per-source Gaia term, the fold to evidence, the evidence-weighted flux
moments and the top-K record. Every block's own `(n_block, n_model, 8)`
working set is discarded once its block's row of results is written.
Each batch's own rows are written straight to their own part file
(`<product>.partN`, rule 10b); no batch or region array, and no part
file's read at join time, ever holds the whole region's P7 at once.
"""

import os
import time

import h5py
import numpy as np
import threadpoolctl
from scipy.special import logsumexp

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.fittp import likelihood
from sesnaimpute.fittp import prior_reader
from sesnaimpute.fittp.gaia import GaiaTerm

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: the six classes, in the fitter's own class-axis order
#: (IMPLEMENTATION_BMSTP_DRAFT.md section 1).
CLASSES = tuple(c.code for c in definitions.CLASSES)

#: `_block_result`'s `log10_flux`/`flux_theta` are two float32
#: `(n_block, n_model, 8)` arrays -- 2 float32-equivalents (W9: both built
#: from the fit's own float32 marks and the float32 register/design
#: column, so no float64 upcast is needed here) -- not counted by
#: `likelihood.block_size`'s own `NONDET_BUFFERS`; passed to `block_size`
#: as `extra_buffers` so the block's real working set stays inside
#: `[fit] block_budget_mb` (W7 review finding 6).
SWEEP_EXTRA_BUFFERS = 2

#: The datasets every P7 part file and the joined product carry, in write
#: order.
_PART_KEYS = ("NAME", "LN_EVIDENCE", "FLUX_MEAN", "FLUX_COV", "TOPK_MODEL", "TOPK_A_K",
              "TOPK_LOG10_B", "TOPK_CHI2", "TOPK_LN_L", "TOPK_LN_PRIOR", "TOPK_FLUX",
              "OCCAM_GAP", "FRAC_CLAMPED", "N_DETECTED")


def _register(config, cls):
    """`(template_log, subclass_idx, n_sub)` for `cls`'s own library
    register (`definitions.CLASS_REGISTER`): `template_log` is `(n_model,
    8)` float32 `log10 F_REF` (floored at the register's own
    `FLOOR_LINEAR`, its FREFRAW convention -- a genuinely dark band is not
    `-inf`), in `definitions.BANDS` order; `subclass_idx` locates each
    model in `definitions.SUBCLASSES_OF[cls]`'s own order (P7's
    `LN_EVIDENCE` column order).
    """
    key = definitions.CLASS_REGISTER[cls]
    path = os.path.join(config.inputs["sed_models"], "registers", "%s_register.hdf5" % key)
    with h5py.File(path, "r") as f:
        n_model = f["models"]["MODEL_NAME"].shape[0]
        f_ref = np.empty((n_model, N_BANDS), dtype=np.float64)
        for j, bkey in enumerate(BAND_KEYS):
            f_ref[:, j] = np.asarray(f["models"]["F_REF_%s" % bkey][:], dtype=np.float64)
        floor_linear = np.asarray(f["models"]["FLOOR_LINEAR"][:], dtype=np.float64)
        subclass_raw = f["models"]["SUBCLASS"][:]
    subclass_names = [s.decode() if isinstance(s, bytes) else s for s in subclass_raw]
    subclass_order = definitions.SUBCLASSES_OF[cls]
    sub_to_idx = {s: i for i, s in enumerate(subclass_order)}
    unknown = set(subclass_names) - set(sub_to_idx)
    if unknown:
        raise ValueError("fittp.sweep: register %r carries SUBCLASS %r not in "
                          "definitions.SUBCLASSES_OF[%r] = %r"
                          % (path, sorted(unknown), cls, subclass_order))
    subclass_idx = np.array([sub_to_idx[s] for s in subclass_names], dtype=np.intp)
    template_log = np.log10(np.maximum(f_ref, floor_linear[:, None])).astype(np.float32)
    return template_log, subclass_idx, len(subclass_order)


def _width_dex(config, region):
    """The region's own non-detection roll-off width, all eight bands
    (`catalog.depths`' `WIDTH_DEX`, section 6.2), read once per region."""
    path = config_module.product_path(config, "catalog", "sesna", "depths", "region")
    with h5py.File(path, "r") as f:
        region_names = [r.decode() if isinstance(r, bytes) else r for r in f["REGION"][:]]
        ridx = region_names.index(region)
        return np.asarray(f["WIDTH_DEX"][ridx, :], dtype=np.float64)


def _catalog_block(config, region, start, stop):
    """One block's own curated rows: fluxes, uncertainties, `ORIGIN_FNU`
    and the source's own column `AK_SESNA` (the same read `likelihood.
    prepare` and `fittp.cascade` both use)."""
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        flux = f["FNU_MJY"][start:stop]
        sigma = f["SIGMA_FNU_MJY"][start:stop]
        origin = f["ORIGIN_FNU"][start:stop]
        ak = f["AK_SESNA"][start:stop]
    return flux, sigma, origin, ak


def _n_sources(config, region):
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        return f["NAME"].shape[0]


def _block_result(config, region, cls, reader, gaia_term, template_log, subclass_idx,
                   n_sub, width_dex, topk, start, stop, timing):
    """One block's own P7 rows (section 1.3): the fit, the prior read, the
    per-source Gaia term, the fold to `ln w_theta`, the evidence-weighted
    flux moments and the top-K record -- one `(n_block, n_model[, 8])`
    working set, discarded on return. `timing` accumulates this block's
    own wall time by stage (rule 17's per-{region, class} split).
    """
    n_model = template_log.shape[0]
    t = time.perf_counter()
    flux, sigma, origin, ak = _catalog_block(config, region, start, stop)
    batch = likelihood.prepare(config, region, cls, start, stop, flux, sigma, origin, ak, width_dex)
    timing["prepare"] += time.perf_counter() - t

    t = time.perf_counter()
    fit = likelihood.fit(batch, template_log)
    timing["fit"] += time.perf_counter() - t

    rows = np.arange(start, stop)
    t = time.perf_counter()
    h = prior_reader.prepare(reader, rows)
    # d(log10 B)/d(a_K) from the fit's own d(SC)/d(A_V) (batch.slope_sc_av):
    # log10_B = -2*SC, a_K = ak_per_av * A_V, so d(log10_B)/d(a_K) =
    # -2 * slope_sc_av / ak_per_av (likelihood.fit's docstring, section 1.3).
    slope_log10b_per_ak = -2.0 * batch.slope_sc_av / batch.ak_per_av
    ln_lambda = prior_reader.ln_prior(reader, rows, h, fit.a_hat, fit.log10_b_hat,
                                       slope_log10b_per_ak, batch.sigma_a_ak,
                                       np.arange(n_model))
    timing["ln_prior"] += time.perf_counter() - t

    ln_l = -0.5 * fit.chi2_min.astype(np.float64) + fit.ln_nondet.astype(np.float64)

    n_block = stop - start
    model_index = np.arange(n_model)
    # Gaia term, vectorised over the whole block at once (W9: no Python
    # loop over sources -- CODING_RULES_BMSTP.md rule 8) -- the fit's own
    # clamped marks are already float32 (likelihood.fit); gaia.ln_gamma
    # upcasts internally where the astrometric algebra needs it.
    t = time.perf_counter()
    ln_gamma = gaia_term.ln_gamma(rows, model_index,
                                   fit.a_hat_clamped, fit.log10_b_hat_clamped, cls.lower())
    timing["ln_gamma"] += time.perf_counter() - t

    ln_w = ln_lambda.astype(np.float64) + ln_l + ln_gamma
    ln_w[batch.flagged] = -np.inf

    # The fold: each model belongs to exactly one subclass (`_register`'s
    # own check), so the per-subclass logsumexp values already partition
    # every column ln w_theta touches; the region total is their own
    # logsumexp rather than a second full (n_block, n_model) reduction
    # over the same elements (W9 -- the two passes were the same sum,
    # done twice).
    t = time.perf_counter()
    ln_evidence64 = np.full((n_block, n_sub), -np.inf, dtype=np.float64)
    for k in range(n_sub):
        mask = subclass_idx == k
        if mask.any():
            ln_evidence64[:, k] = logsumexp(ln_w[:, mask], axis=1)
    ln_evidence = ln_evidence64.astype(np.float32)
    ev_total = logsumexp(ln_evidence64, axis=1)
    timing["fold"] += time.perf_counter() - t

    t = time.perf_counter()
    with np.errstate(invalid="ignore"):
        p_theta = np.exp(ln_w - ev_total[:, None])
    p_theta = np.where(np.isfinite(p_theta), p_theta, 0.0)

    # the template's fitted flux at its clamped marks: recovering the
    # A_V-unit extinction the design column (batch.ext_col) was built in
    # from the reported a_K mark (fit.a_hat_clamped = A_V_clamped *
    # ak_per_av, likelihood.fit's own docstring) -- algebraically the
    # same log10-flux likelihood.fit's own non-detection term evaluates.
    # All four inputs are already float32 (the register, the design
    # column, the fit's own clamped marks): no float64 temporary carries
    # this value (W9 -- was an unneeded float64 upcast of a float32-exact
    # quantity, doubling the block's largest working set).
    av_clamped = fit.a_hat_clamped / batch.ak_per_av[:, None].astype(np.float32)
    log10_flux = (template_log[None, :, :]
                  + batch.ext_col[:, None, :] * av_clamped[:, :, None]
                  + fit.log10_b_hat_clamped[:, :, None])
    flux_theta = np.power(np.float32(10.0), log10_flux)  # (n_block, n_model, 8), float32
    timing["moments"] += time.perf_counter() - t

    t = time.perf_counter()
    flux_mean = np.einsum("nm,nmb->nb", p_theta, flux_theta)
    flux_m2 = np.einsum("nm,nma,nmb->nab", p_theta, flux_theta, flux_theta)
    flux_cov = flux_m2 - flux_mean[:, :, None] * flux_mean[:, None, :]
    timing["moments"] += time.perf_counter() - t

    t = time.perf_counter()
    k_keep = min(topk, n_model)
    order = np.argpartition(-ln_w, k_keep - 1, axis=1)[:, :k_keep]
    row_idx = np.arange(n_block)[:, None]
    order = order[row_idx, np.argsort(-ln_w[row_idx, order], axis=1)]

    occam_gap = (ev_total - ln_w.max(axis=1)).astype(np.float32)

    topk_model = np.full((n_block, topk), -1, dtype=np.int32)
    topk_a_k = np.full((n_block, topk), np.nan, dtype=np.float32)
    topk_log10_b = np.full((n_block, topk), np.nan, dtype=np.float32)
    topk_chi2 = np.full((n_block, topk), np.nan, dtype=np.float32)
    topk_ln_l = np.full((n_block, topk), np.nan, dtype=np.float32)
    topk_ln_prior = np.full((n_block, topk), np.nan, dtype=np.float32)
    topk_flux = np.full((n_block, topk, N_BANDS), np.nan, dtype=np.float32)

    good = ~batch.flagged
    topk_model[good, :k_keep] = order[good].astype(np.int32)
    topk_a_k[good, :k_keep] = fit.a_hat_clamped[row_idx, order][good]
    topk_log10_b[good, :k_keep] = fit.log10_b_hat_clamped[row_idx, order][good]
    topk_chi2[good, :k_keep] = fit.chi2_min[row_idx, order][good]
    topk_ln_l[good, :k_keep] = ln_l[row_idx, order][good].astype(np.float32)
    topk_ln_prior[good, :k_keep] = ln_lambda[row_idx, order][good]
    topk_flux[good, :k_keep, :] = flux_theta[row_idx, order, :][good].astype(np.float32)
    occam_gap[~good] = np.nan
    timing["topk"] += time.perf_counter() - t

    return dict(
        ln_evidence=ln_evidence,
        flux_mean=flux_mean.astype(np.float32), flux_cov=flux_cov.astype(np.float32),
        topk_model=topk_model, topk_a_k=topk_a_k, topk_log10_b=topk_log10_b,
        topk_chi2=topk_chi2, topk_ln_l=topk_ln_l, topk_ln_prior=topk_ln_prior,
        topk_flux=topk_flux, occam_gap=occam_gap,
        frac_clamped=fit.frac_clamped, n_detected=batch.n_detected,
        zero_ext_count=int((fit.a_hat[good] < 0.0).sum()) if good.any() else 0,
        n_templates_checked=int(good.sum()) * n_model,
    )


def _part_path(path, bi):
    return "%s.part%d" % (path, bi)


def _batch_result(config, region, cls, reader, gaia_term, template_log, subclass_idx,
                   n_sub, width_dex, topk, block, bstart, bstop, timing):
    """One batch's own P7 rows, `[bstart, bstop)`, folded block by block
    (rule 10b): a batch-sized array, never a region-sized one. `timing`
    accumulates this batch's wall time by stage.
    """
    m = bstop - bstart
    ln_evidence = np.empty((m, n_sub), dtype=np.float32)
    flux_mean = np.empty((m, N_BANDS), dtype=np.float32)
    flux_cov = np.empty((m, N_BANDS, N_BANDS), dtype=np.float32)
    topk_model = np.empty((m, topk), dtype=np.int32)
    topk_a_k = np.empty((m, topk), dtype=np.float32)
    topk_log10_b = np.empty((m, topk), dtype=np.float32)
    topk_chi2 = np.empty((m, topk), dtype=np.float32)
    topk_ln_l = np.empty((m, topk), dtype=np.float32)
    topk_ln_prior = np.empty((m, topk), dtype=np.float32)
    topk_flux = np.empty((m, topk, N_BANDS), dtype=np.float32)
    occam_gap = np.empty(m, dtype=np.float32)
    frac_clamped = np.empty(m, dtype=np.float32)
    n_detected = np.empty(m, dtype=np.int8)

    zero_ext_count = 0
    n_templates_checked = 0
    for start in range(bstart, bstop, block):
        stop = min(start + block, bstop)
        r = _block_result(config, region, cls, reader, gaia_term, template_log,
                           subclass_idx, n_sub, width_dex, topk, start, stop, timing)
        sl = slice(start - bstart, stop - bstart)
        ln_evidence[sl] = r["ln_evidence"]
        flux_mean[sl] = r["flux_mean"]
        flux_cov[sl] = r["flux_cov"]
        topk_model[sl] = r["topk_model"]
        topk_a_k[sl] = r["topk_a_k"]
        topk_log10_b[sl] = r["topk_log10_b"]
        topk_chi2[sl] = r["topk_chi2"]
        topk_ln_l[sl] = r["topk_ln_l"]
        topk_ln_prior[sl] = r["topk_ln_prior"]
        topk_flux[sl] = r["topk_flux"]
        occam_gap[sl] = r["occam_gap"]
        frac_clamped[sl] = r["frac_clamped"]
        n_detected[sl] = r["n_detected"]
        zero_ext_count += r["zero_ext_count"]
        n_templates_checked += r["n_templates_checked"]

    with h5py.File(config_module.product_path(config, "catalog", "sesna", "sources", "source",
                                               region=region), "r") as f:
        name = f["NAME"][bstart:bstop]

    return dict(
        name=name, ln_evidence=ln_evidence, flux_mean=flux_mean, flux_cov=flux_cov,
        topk_model=topk_model, topk_a_k=topk_a_k, topk_log10_b=topk_log10_b,
        topk_chi2=topk_chi2, topk_ln_l=topk_ln_l, topk_ln_prior=topk_ln_prior,
        topk_flux=topk_flux, occam_gap=occam_gap, frac_clamped=frac_clamped,
        n_detected=n_detected, zero_ext_count=zero_ext_count,
        n_templates_checked=n_templates_checked,
    )


def _write_part(part_path, batch):
    with h5py.File(part_path, "w") as f:
        for key, field in zip(_PART_KEYS,
                               ("name", "ln_evidence", "flux_mean", "flux_cov", "topk_model",
                                "topk_a_k", "topk_log10_b", "topk_chi2", "topk_ln_l",
                                "topk_ln_prior", "topk_flux", "occam_gap", "frac_clamped",
                                "n_detected")):
            f.create_dataset(key, data=batch[field])


def build_region_class(config, region, cls, st, limit=None):
    """Sweeps one {region, class}'s whole region (or, with `limit`, its
    first `limit` catalogue rows only -- a timing/acceptance device, never
    a default) in batches of `[fit] batch_size`, each batch in blocks of
    `likelihood.block_size` sources (section 1.3; IMPLEMENTATION_BMSTP_
    DRAFT.md section 4 row 2.4). Each batch's own rows are written
    straight to their own part file (rule 10b); the caller joins the
    parts once every batch is done. Returns the part file list and the
    summary numbers for the caller's join and report.
    """
    template_log, subclass_idx, n_sub = _register(config, cls)
    n_model = template_log.shape[0]
    reader = prior_reader.load(config, region, cls)
    gaia_term = GaiaTerm(config, region)
    width_dex = _width_dex(config, region)
    topk = config.fit_topk
    block = likelihood.block_size(n_model, config.fit_block_budget_mb,
                                   extra_buffers=SWEEP_EXTRA_BUFFERS)

    n_source = _n_sources(config, region)
    if limit is not None:
        n_source = min(n_source, limit)
    batch_size = config.fit_batch_size
    batch_bounds = [(s, min(s + batch_size, n_source)) for s in range(0, n_source, batch_size)]

    path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    part_paths = []
    zero_ext_count = 0
    n_templates_checked = 0
    # Per-{region, class} wall-time split (rule 17): the stages a batch
    # passes through, plus the part-file write, each block/batch adds its
    # own share into these totals.
    timing = dict.fromkeys(
        ("prepare", "fit", "ln_prior", "ln_gamma", "fold", "moments", "topk", "write"), 0.0)

    # BLAS's own thread pool is capped to 1 for the fit's small (m,8)@(8,8)
    # gemms (W9a: memory-bound, 1 thread ~10% faster than 4) while numba's
    # separate erfc kernel keeps its own 4 threads (fittp.likelihood).
    with threadpoolctl.threadpool_limits(1, user_api="blas"):
        for bi, (bstart, bstop) in enumerate(batch_bounds):
            batch = _batch_result(config, region, cls, reader, gaia_term, template_log,
                                   subclass_idx, n_sub, width_dex, topk, block, bstart, bstop,
                                   timing)
            part_path = _part_path(path, bi)
            t = time.perf_counter()
            _write_part(part_path, batch)
            timing["write"] += time.perf_counter() - t
            part_paths.append(part_path)
            zero_ext_count += batch["zero_ext_count"]
            n_templates_checked += batch["n_templates_checked"]
            st.tick(bi + 1, len(batch_bounds), "batches")

    zero_ext_frac = zero_ext_count / n_templates_checked if n_templates_checked else float("nan")
    density_file = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    lib, granule = ("sps", "region") if cls == "STAR" else \
        {"AGB": ("agb", "region"), "PAHC": ("pahc", "region"), "YSO": ("yso", "survey"),
         "H2S": ("h2shock", "survey"), "GAL": ("galz", "survey")}[cls]
    weights_file = config_module.product_path(
        config, "bmstp", "weights", lib, granule, region=(region if granule == "region" else None))
    return dict(
        path=path, part_paths=part_paths, n_source=n_source, n_model=n_model,
        subclasses=definitions.SUBCLASSES_OF[cls], library=definitions.CLASS_REGISTER[cls],
        zero_ext_frac=zero_ext_frac, density_file=density_file, weights_file=weights_file,
        timing=timing,
    )


def join_parts(summary, topk):
    """Joins one {region, class}'s part files into the final P7 product,
    one part's rows at a time, dataset by dataset (rule 10b: never a
    region-sized array); removes the part files once written.
    """
    path = summary["path"]
    n_source = summary["n_source"]
    with h5py.File(path, "w") as out:
        for key in _PART_KEYS:
            with h5py.File(summary["part_paths"][0], "r") as pf0:
                shape = (n_source,) + pf0[key].shape[1:]
                dtype = pf0[key].dtype
            out.create_dataset(key, shape=shape, dtype=dtype)
        offset = 0
        for part_path in summary["part_paths"]:
            with h5py.File(part_path, "r") as pf:
                m = pf["NAME"].shape[0]
                for key in _PART_KEYS:
                    out[key][offset:offset + m] = pf[key][:]
            offset += m
        out.attrs["GRANULE"] = "source"
        out.attrs["CLASS"] = summary.get("cls")
        out.attrs["SUBCLASSES"] = np.array(summary["subclasses"], dtype="S8")
        out.attrs["LIBRARY"] = summary["library"]
        out.attrs["N_MODEL"] = summary["n_model"]
        out.attrs["K"] = topk
        out.attrs["WEIGHTS_FILE"] = summary["weights_file"]
        out.attrs["DENSITY_FILE"] = summary["density_file"]
    for part_path in summary["part_paths"]:
        os.remove(part_path)


def build(config, regions=None, classes=None, limit=None):
    """Writes `fittp/fit/<CLS>_fit_source__R.hdf5` for every {region,
    class} pair (default all thirty regions, all six classes; section
    1.3, IMPLEMENTATION_BMSTP_DRAFT.md P7). `limit` restricts every
    region to its first `limit` catalogue rows -- a timing/acceptance
    device for a class whose prior read does not fit the run budget on
    the whole region, never a default.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    class_codes = classes if classes is not None else list(CLASSES)
    for region in region_names:
        for cls in class_codes:
            with progress.Stage("fittp.sweep.%s" % cls, region) as st:
                summary = build_region_class(config, region, cls, st, limit=limit)
                summary["cls"] = cls
                join_parts(summary, config.fit_topk)
                with h5py.File(summary["path"], "r") as f:
                    occam = np.asarray(f["OCCAM_GAP"][:])
                occam_finite = occam[np.isfinite(occam)]
                occam_median = float(np.median(occam_finite)) if occam_finite.size else float("nan")
                # Rule 17's per-{region, class} wall-time split (W9): where the
                # sweep's own time goes, stage by stage, so a future pass reads
                # the long pole straight off the done line instead of profiling.
                split = " ".join("%s=%.1fs" % (k, v) for k, v in summary["timing"].items())
                st.done(summary["path"], n=summary["n_source"], n_model=summary["n_model"],
                        zero_ext_frac=summary["zero_ext_frac"], occam_gap_median=occam_median,
                        n_batches=len(summary["part_paths"]), split=split)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="default: all six -- every {region, class} pair is swept")
    parser.add_argument("--limit", type=int, default=None,
                         help="sweep only the first LIMIT catalogue rows of each region "
                              "(a timing/acceptance device, never a default)")
    args = parser.parse_args()
    cfg = config_module.load(args.config)
    build(cfg, regions=args.regions, classes=args.classes, limit=args.limit)
