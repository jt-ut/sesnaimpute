"""The catalogue's own resolution: a source's `(a, log10 B)` posterior
width from its detected bands alone (`10_POSTERIOR.md` section 1, "the
fitter profiles `(a, B)` to `(a-hat, B-hat)` and reports the local
curvature `V_s`"; `IMPLEMENTATION.md` section 3, the STAR/AGB/PAHC
storage row's amendment: "the cell size is half the survey-median width
of a source's own (a, log10 B) posterior, formed from the catalogue's
flux errors alone").

For one source with detected bands `i` (`ORIGIN_FNU == 1`, a measured
flux, not a filled non-detection), the model log10 flux in band `i` is
`log10 B - 0.4 * kappa_i(a) * a` plus a band constant, so the gradient of
that model with respect to `x = (a, log10 B)` is `g_i = (-0.4 kappa_i(a),
1)`. Summing the per-band Fisher information `(1/sigma_i**2) g_i g_i^T`
over the source's own detected bands gives the same 2x2 curvature the
fitter's nuisance quadrature forms from the SED fit
(`sed_fit.nodes.nuisance_curvature`, read for this module's own citation,
not imported: the census forms it here from the catalogue alone, before
any fit exists). Its inverse is `V_s`; the diagonal is the source's own
`sigma_a**2`, `sigma_logB**2`. A source with fewer than two detected
bands leaves the 2x2 sum singular (rank <= 1) and is skipped -- it has
no width to contribute.

`kappa_i(a)` is the blended law at the source's own adopted column
(`prior.selection.kappa_hybrid` at `prior.selection.law_dense_weight(a)`,
SPEC_PRIORS.md section 1.3): the same law every other census module uses
at the same column, not a fit-time quantity.

Chunked one region at a time (joblib over regions, `config.n_jobs`
workers, CODING_RULES.md 10a): a region's own `(n_source, 8)` flux and
error columns are the largest array this module ever holds, never the
survey's 8.66 million rows at once.

Writes `bms/sesna/posterior-width_sesna_survey.hdf5`: the survey-wide
median and 16th/84th percentile of `sigma_a` (K magnitudes) and
`sigma_logB`, and the per-region median of each.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import selection

# ---------------------------------------------------------------------------
# constants block
# ---------------------------------------------------------------------------

#: `ln(10)`: converts a fractional flux error to a log10-flux error
#: (`d(log10 F) = dF / (F * ln 10)`).
_LN10 = np.log(10.0)

#: `catalog.curated`'s own code for "measured detection" in `ORIGIN_FNU`
#: (catalog/curated.py module docstring): only these bands carry a real
#: flux-error measurement, so only these enter the curvature sum.
ORIGIN_MEASURED = 1

#: A source needs at least this many detected bands to constrain both
#: `a` and `log10 B` (`10_POSTERIOR.md` section 1: two free nuisances);
#: the same floor `prior.selection.MIN_BANDS` states for SESNA's own
#: catalog inclusion test.
MIN_DETECTED_BANDS = selection.MIN_BANDS

#: The percentiles this product reports beside the median
#: (`IMPLEMENTATION.md` section 3's amendment: "16/84%").
_PERCENTILES = (16.0, 50.0, 84.0)


# ---------------------------------------------------------------------------
# one region's own curvature, vectorised over its sources
# ---------------------------------------------------------------------------

def region_widths(config, region):
    """`(sigma_a, sigma_logb)` for `region`'s sources with >= `MIN_
    DETECTED_BANDS` measured bands (module docstring): the per-band
    Fisher information summed over detected bands, inverted per source.
    Vectorised over the region's own sources (rule 8); a region's own
    `(n_source, 8)` columns are the only per-source arrays this module
    holds (rule 10a)."""
    cat_path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(cat_path):
        raise FileNotFoundError(
            "prior.posterior_width: no curated catalogue for region %r at %s -- "
            "run the 'catalog.curated' RUNBOOK line first" % (region, cat_path))
    with h5py.File(cat_path, "r") as f:
        fnu = f["FNU_MJY"][:].astype(np.float64)
        sigma_fnu = f["SIGMA_FNU_MJY"][:].astype(np.float64)
        origin = f["ORIGIN_FNU"][:]

    adopted_path = config_module.product_path(config, "sky/derived", "adopted",
                                               "column", "source", region=region)
    a_col = access.per_source(config, region, adopted_path, ["A_COL_K"])["A_COL_K"].astype(np.float64)

    # the blended law at each source's own adopted column
    # (prior.selection.kappa_hybrid, module docstring).
    w_dense = selection.law_dense_weight(a_col)
    kappa = selection.kappa_hybrid(config, w_dense)  # (n_source, 8)

    detected = origin == ORIGIN_MEASURED
    n_det = np.count_nonzero(detected, axis=1)
    usable = n_det >= MIN_DETECTED_BANDS

    # sigma_i, the catalogue's log10-flux error (module docstring); only
    # finite, positive fluxes and errors are informative, and only in a
    # detected band (rule 6: no silent fallback -- an undetected band's
    # error simply drops out of the sum via `weight`).
    finite = np.isfinite(fnu) & (fnu > 0.0) & np.isfinite(sigma_fnu) & (sigma_fnu > 0.0)
    sigma_log10 = np.full_like(fnu, np.inf)
    sigma_log10[finite] = sigma_fnu[finite] / fnu[finite] / _LN10
    weight = np.where(detected & finite, 1.0 / sigma_log10 ** 2, 0.0)  # (n_source, 8)

    g_a = -0.4 * kappa  # (n_source, 8): d(model log10 flux) / d(a)
    m11 = np.sum(weight * g_a * g_a, axis=1)
    m12 = np.sum(weight * g_a, axis=1)          # d/d(log10 B) term is 1
    m22 = np.sum(weight, axis=1)
    det_m = m11 * m22 - m12 * m12

    ok = usable & np.isfinite(det_m) & (det_m > 0.0)
    sigma_a = np.full(fnu.shape[0], np.nan)
    sigma_logb = np.full(fnu.shape[0], np.nan)
    # V_s = M^{-1}; the diagonal only is needed (module docstring).
    sigma_a[ok] = np.sqrt(m22[ok] / det_m[ok])
    sigma_logb[ok] = np.sqrt(m11[ok] / det_m[ok])
    return sigma_a[ok], sigma_logb[ok]


# ---------------------------------------------------------------------------
# survey-wide aggregation
# ---------------------------------------------------------------------------

def survey_width(config, regions=None):
    """`(per_region, survey)`: `per_region` is `{region: (sigma_a, sigma_
    logb)}` valid-source arrays; `survey` is the concatenation. One
    joblib task per region (`config.n_jobs` workers, CODING_RULES.md
    10a): the region loop is the only parallel axis, so no region's own
    columns are ever resident alongside another's beyond the worker
    pool's own width."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    results = Parallel(n_jobs=config.n_jobs)(
        delayed(region_widths)(config, region) for region in region_names)
    per_region = dict(zip(region_names, results))
    survey_sigma_a = np.concatenate([r[0] for r in results]) if results else np.array([])
    survey_sigma_logb = np.concatenate([r[1] for r in results]) if results else np.array([])
    return per_region, (survey_sigma_a, survey_sigma_logb)


def _percentiles(values):
    if values.size == 0:
        return (float("nan"),) * 3
    p16, p50, p84 = np.percentile(values, _PERCENTILES)
    return float(p16), float(p50), float(p84)


# ---------------------------------------------------------------------------
# write / read
# ---------------------------------------------------------------------------

def write(config, per_region, survey):
    """Writes `bms/sesna/posterior-width_sesna_survey.hdf5` (module
    docstring): survey-wide median and 16/84% of `sigma_a`, `sigma_logb`,
    and the per-region median of each, in the region order `regions.
    REGIONS` gives for the regions present in `per_region`."""
    survey_sigma_a, survey_sigma_logb = survey
    a_p16, a_p50, a_p84 = _percentiles(survey_sigma_a)
    b_p16, b_p50, b_p84 = _percentiles(survey_sigma_logb)

    region_order = [r.name for r in regions_module.REGIONS if r.name in per_region]
    region_median_a = np.array([_percentiles(per_region[r][0])[1] for r in region_order])
    region_median_b = np.array([_percentiles(per_region[r][1])[1] for r in region_order])
    region_n = np.array([per_region[r][0].size for r in region_order], dtype=np.int64)

    out_path = config_module.product_path(config, "bms", "sesna", "posterior-width", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.create_dataset("SIGMA_A_MEDIAN", data=np.float64(a_p50))
        f.create_dataset("SIGMA_A_P16", data=np.float64(a_p16))
        f.create_dataset("SIGMA_A_P84", data=np.float64(a_p84))
        f.create_dataset("SIGMA_LOGB_MEDIAN", data=np.float64(b_p50))
        f.create_dataset("SIGMA_LOGB_P16", data=np.float64(b_p16))
        f.create_dataset("SIGMA_LOGB_P84", data=np.float64(b_p84))
        f.create_dataset("REGION", data=np.array([r.encode("utf-8") for r in region_order]))
        f.create_dataset("SIGMA_A_MEDIAN_REGION", data=region_median_a.astype(np.float64))
        f.create_dataset("SIGMA_LOGB_MEDIAN_REGION", data=region_median_b.astype(np.float64))
        f.create_dataset("N_SOURCE_WIDTH_REGION", data=region_n)
    return out_path, (a_p16, a_p50, a_p84), (b_p16, b_p50, b_p84)


def read(config):
    """`(sigma_a_median, sigma_logb_median)`, the survey-wide values
    `prior.star_shapes` smooths and grids on (`IMPLEMENTATION.md`
    section 3's amendment)."""
    path = config_module.product_path(config, "bms", "sesna", "posterior-width", "survey")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.posterior_width: no survey posterior width at %s -- "
            "run the 'prior.posterior_width' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        return float(f["SIGMA_A_MEDIAN"][()]), float(f["SIGMA_LOGB_MEDIAN"][()])


def build(config, regions=None):
    """Computes and writes the survey posterior width (module docstring).
    `regions` selects which regions' sources are measured; the product
    is, like `prior.column_grid`, one survey-wide file regardless."""
    per_region, survey = survey_width(config, regions=regions)
    path, (a_p16, a_p50, a_p84), (b_p16, b_p50, b_p84) = write(config, per_region, survey)
    n_total = int(sum(v[0].size for v in per_region.values()))
    print(
        "posterior_width: %d region(s), %d source(s) with a width -> "
        "sigma_a[16,50,84]=(%.4f, %.4f, %.4f) A_K "
        "sigma_logb[16,50,84]=(%.4f, %.4f, %.4f) -> %s"
        % (len(per_region), n_total, a_p16, a_p50, a_p84, b_p16, b_p50, b_p84, path),
        flush=True)
    region_a = {r: _percentiles(v[0])[1] for r, v in per_region.items()}
    region_b = {r: _percentiles(v[1])[1] for r, v in per_region.items()}
    lo_a, hi_a = min(region_a.values()), max(region_a.values())
    lo_b, hi_b = min(region_b.values()), max(region_b.values())
    print(
        "posterior_width: per-region median sigma_a range (%.4f, %.4f) A_K, "
        "median sigma_logb range (%.4f, %.4f)" % (lo_a, hi_a, lo_b, hi_b),
        flush=True)


if __name__ == "__main__":
    run(build)
