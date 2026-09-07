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
working set is discarded once its block's row of results is written; no
batch or region ever holds an `(n, m, 8)` array.
"""

import os

import h5py
import numpy as np
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
                   n_sub, width_dex, topk, start, stop):
    """One block's own P7 rows (section 1.3): the fit, the prior read, the
    per-source Gaia term, the fold to `ln w_theta`, the evidence-weighted
    flux moments and the top-K record -- one `(n_block, n_model[, 8])`
    working set, discarded on return.
    """
    n_model = template_log.shape[0]
    flux, sigma, origin, ak = _catalog_block(config, region, start, stop)
    batch = likelihood.prepare(config, region, start, stop, flux, sigma, origin, ak, width_dex)
    fit = likelihood.fit(batch, template_log)

    rows = np.arange(start, stop)
    h = prior_reader.prepare(reader, rows)
    # d(log10 B)/d(a_K) from the fit's own d(SC)/d(A_V) (batch.slope_sc_av):
    # log10_B = -2*SC, a_K = ak_per_av * A_V, so d(log10_B)/d(a_K) =
    # -2 * slope_sc_av / ak_per_av (likelihood.fit's docstring, section 1.3).
    slope_log10b_per_ak = -2.0 * batch.slope_sc_av / batch.ak_per_av
    ln_lambda = prior_reader.ln_prior(reader, rows, h, fit.a_hat, fit.log10_b_hat,
                                       slope_log10b_per_ak, batch.sigma_a_ak,
                                       np.arange(n_model))

    ln_l = -0.5 * fit.chi2_min.astype(np.float64) + fit.ln_nondet.astype(np.float64)

    n_block = stop - start
    model_index = np.arange(n_model)
    ln_gamma = np.empty((n_block, n_model), dtype=np.float64)
    a_clamped64 = fit.a_hat_clamped.astype(np.float64)
    b_clamped64 = fit.log10_b_hat_clamped.astype(np.float64)
    for i in range(n_block):
        ln_gamma[i] = gaia_term.ln_gamma(start + i, model_index,
                                          a_clamped64[i], b_clamped64[i], cls.lower())

    ln_w = ln_lambda.astype(np.float64) + ln_l + ln_gamma
    ln_w[batch.flagged] = -np.inf

    ev_total = logsumexp(ln_w, axis=1)
    ln_evidence = np.full((n_block, n_sub), -np.inf, dtype=np.float32)
    for k in range(n_sub):
        mask = subclass_idx == k
        if mask.any():
            ln_evidence[:, k] = logsumexp(ln_w[:, mask], axis=1).astype(np.float32)

    with np.errstate(invalid="ignore"):
        p_theta = np.exp(ln_w - ev_total[:, None])
    p_theta = np.where(np.isfinite(p_theta), p_theta, 0.0)

    # the template's fitted flux at its clamped marks: recovering the
    # A_V-unit extinction the design column (batch.ext_col) was built in
    # from the reported a_K mark (fit.a_hat_clamped = A_V_clamped *
    # ak_per_av, likelihood.fit's own docstring) -- algebraically the
    # same log10-flux likelihood.fit's own non-detection term evaluates.
    av_clamped = a_clamped64 / batch.ak_per_av[:, None]
    log10_flux = (template_log[None, :, :].astype(np.float64)
                  + batch.ext_col.astype(np.float64)[:, None, :] * av_clamped[:, :, None]
                  + b_clamped64[:, :, None])
    flux_theta = np.power(10.0, log10_flux)  # (n_block, n_model, 8), this block only

    flux_mean = np.einsum("nm,nmb->nb", p_theta, flux_theta)
    flux_m2 = np.einsum("nm,nma,nmb->nab", p_theta, flux_theta, flux_theta)
    flux_cov = flux_m2 - flux_mean[:, :, None] * flux_mean[:, None, :]

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


def build_region_class(config, region, cls, st, limit=None):
    """Sweeps one {region, class}'s whole region (or, with `limit`, its
    first `limit` catalogue rows only -- a timing/acceptance device, never
    a default) in batches of `[fit] batch_size`, each batch in blocks of
    `likelihood.block_size` sources, joined in catalogue-row order
    (section 1.3; IMPLEMENTATION_BMSTP_DRAFT.md section 4 row 2.4).
    Returns the P7 arrays for the rows swept plus the zero-extinction
    fraction and the Occam-gap values, for the caller's report.
    """
    template_log, subclass_idx, n_sub = _register(config, cls)
    n_model = template_log.shape[0]
    reader = prior_reader.load(config, region, cls)
    gaia_term = GaiaTerm(config, region)
    width_dex = _width_dex(config, region)
    topk = config.fit_topk
    block = likelihood.block_size(n_model, config.fit_block_budget_mb)

    n_source = _n_sources(config, region)
    if limit is not None:
        n_source = min(n_source, limit)
    batch_size = config.fit_batch_size
    batch_bounds = [(s, min(s + batch_size, n_source)) for s in range(0, n_source, batch_size)]

    ln_evidence = np.empty((n_source, n_sub), dtype=np.float32)
    flux_mean = np.empty((n_source, N_BANDS), dtype=np.float32)
    flux_cov = np.empty((n_source, N_BANDS, N_BANDS), dtype=np.float32)
    topk_model = np.empty((n_source, topk), dtype=np.int32)
    topk_a_k = np.empty((n_source, topk), dtype=np.float32)
    topk_log10_b = np.empty((n_source, topk), dtype=np.float32)
    topk_chi2 = np.empty((n_source, topk), dtype=np.float32)
    topk_ln_l = np.empty((n_source, topk), dtype=np.float32)
    topk_ln_prior = np.empty((n_source, topk), dtype=np.float32)
    topk_flux = np.empty((n_source, topk, N_BANDS), dtype=np.float32)
    occam_gap = np.empty(n_source, dtype=np.float32)
    frac_clamped = np.empty(n_source, dtype=np.float32)
    n_detected = np.empty(n_source, dtype=np.int8)

    zero_ext_count = 0
    n_templates_checked = 0

    for bi, (bstart, bstop) in enumerate(batch_bounds):
        for start in range(bstart, bstop, block):
            stop = min(start + block, bstop)
            r = _block_result(config, region, cls, reader, gaia_term, template_log,
                               subclass_idx, n_sub, width_dex, topk, start, stop)
            ln_evidence[start:stop] = r["ln_evidence"]
            flux_mean[start:stop] = r["flux_mean"]
            flux_cov[start:stop] = r["flux_cov"]
            topk_model[start:stop] = r["topk_model"]
            topk_a_k[start:stop] = r["topk_a_k"]
            topk_log10_b[start:stop] = r["topk_log10_b"]
            topk_chi2[start:stop] = r["topk_chi2"]
            topk_ln_l[start:stop] = r["topk_ln_l"]
            topk_ln_prior[start:stop] = r["topk_ln_prior"]
            topk_flux[start:stop] = r["topk_flux"]
            occam_gap[start:stop] = r["occam_gap"]
            frac_clamped[start:stop] = r["frac_clamped"]
            n_detected[start:stop] = r["n_detected"]
            zero_ext_count += r["zero_ext_count"]
            n_templates_checked += r["n_templates_checked"]
        st.tick(bi + 1, len(batch_bounds), "batches")

    with h5py.File(config_module.product_path(config, "catalog", "sesna", "sources", "source",
                                               region=region), "r") as f:
        name = f["NAME"][:n_source]

    zero_ext_frac = zero_ext_count / n_templates_checked if n_templates_checked else float("nan")
    density_file = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    lib, granule = ("sps", "region") if cls == "STAR" else \
        {"AGB": ("agb", "region"), "PAHC": ("pahc", "region"), "YSO": ("yso", "survey"),
         "H2S": ("h2shock", "survey"), "GAL": ("galz", "survey")}[cls]
    weights_file = config_module.product_path(
        config, "bmstp", "weights", lib, granule, region=(region if granule == "region" else None))
    return dict(
        name=name, ln_evidence=ln_evidence, flux_mean=flux_mean, flux_cov=flux_cov,
        topk_model=topk_model, topk_a_k=topk_a_k, topk_log10_b=topk_log10_b,
        topk_chi2=topk_chi2, topk_ln_l=topk_ln_l, topk_ln_prior=topk_ln_prior,
        topk_flux=topk_flux, occam_gap=occam_gap, frac_clamped=frac_clamped,
        n_detected=n_detected, n_model=n_model, subclasses=definitions.SUBCLASSES_OF[cls],
        library=definitions.CLASS_REGISTER[cls], zero_ext_frac=zero_ext_frac,
        density_file=density_file, weights_file=weights_file,
    )


def write_region_class(path, result, cls, topk):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("NAME", data=result["name"])
        f.create_dataset("LN_EVIDENCE", data=result["ln_evidence"])
        f.create_dataset("FLUX_MEAN", data=result["flux_mean"])
        f.create_dataset("FLUX_COV", data=result["flux_cov"])
        f.create_dataset("TOPK_MODEL", data=result["topk_model"])
        f.create_dataset("TOPK_A_K", data=result["topk_a_k"])
        f.create_dataset("TOPK_LOG10_B", data=result["topk_log10_b"])
        f.create_dataset("TOPK_CHI2", data=result["topk_chi2"])
        f.create_dataset("TOPK_LN_L", data=result["topk_ln_l"])
        f.create_dataset("TOPK_LN_PRIOR", data=result["topk_ln_prior"])
        f.create_dataset("TOPK_FLUX", data=result["topk_flux"])
        f.create_dataset("OCCAM_GAP", data=result["occam_gap"])
        f.create_dataset("FRAC_CLAMPED", data=result["frac_clamped"])
        f.create_dataset("N_DETECTED", data=result["n_detected"])
        f.attrs["GRANULE"] = "source"
        f.attrs["CLASS"] = cls
        f.attrs["SUBCLASSES"] = np.array(result["subclasses"], dtype="S8")
        f.attrs["LIBRARY"] = result["library"]
        f.attrs["N_MODEL"] = result["n_model"]
        f.attrs["K"] = topk
        f.attrs["WEIGHTS_FILE"] = result["weights_file"]
        f.attrs["DENSITY_FILE"] = result["density_file"]


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
                result = build_region_class(config, region, cls, st, limit=limit)
                path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
                write_region_class(path, result, cls, config.fit_topk)
                occam_finite = result["occam_gap"][np.isfinite(result["occam_gap"])]
                occam_median = float(np.median(occam_finite)) if occam_finite.size else float("nan")
                st.done(path, n=result["name"].shape[0], n_model=result["n_model"],
                        zero_ext_frac=result["zero_ext_frac"], occam_gap_median=occam_median)


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
