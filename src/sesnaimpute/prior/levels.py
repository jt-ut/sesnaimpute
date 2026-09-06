"""The six per-region level factors (SPEC_PRIORS.md section 0.2, the
paragraph "The levels are normalised to the survey"; IMPLEMENTATION.md
section 6, stage 12a): each class's count is calibrated on something
outside SESNA, and their sum was never held to SESNA's own total. This
module fits, per region, one non-negative factor per class that brings
the six classes' own predicted count patterns -- positions and counts
only, never a label -- into a Poisson fit of the region's catalogued
source density per occupied nside-512 pixel.

Per pixel `i`, `n_i` is the pixel's own catalogued source count (the
granule map's own `N_SOURCE_ROWS`) and each class `C`'s predicted count
before the level correction, `P_C,i`, is:

    STAR, AGB, PAHC, GAL, H2S: the mean of the class's per-source count
    (`prior.counts_star_family`, `prior.counts_cloud`; catalogued-rows-
    per-deg**2) over the pixel's own catalogued sources, times the
    pixel's own solid angle -- the approximation that the pixel's own
    sources sample its own column field.

    YSO: `prior.yso.law_area_integral`, the young-star law integrated
    over the pixel's own area from the column MAP (not the sources),
    times the pixel's own mean `EPS_YSO` and solid angle. The law rises
    with the column squared, so the source-sampled mean used by the
    other five classes would under-run it; the map integral is exact,
    the mean selected fraction is not (SESNA sources have already
    passed the selection `EPS_YSO` is the fraction of).

The six factors `f_C` maximise the Poisson likelihood of `n_i` given
`mu_i = Sum_C f_C . P_C,i`, fitted as `scipy.optimize.minimize` on the
six log-factors (so `f_C > 0` by construction) with the analytic
gradient, started at `f_C = 1`. Standard errors and the correlation
matrix come from the observed (Fisher) information in `f`-space itself,
not in the log-factors the optimizer runs in: `mu_i` is linear in `f`,
so the Poisson log-likelihood's Hessian in `f`-space is the plain
weighted Gram matrix `P^T diag(n / mu**2) P`.

Writes one 30-row product, `bms/table/levels_table_region.hdf5`: root
attr `GRANULE = "region"`; `F_STAR` ... `F_H2S`, `SIGMA_F_STAR` ...
`SIGMA_F_H2S`, `CORR` (6x6), `TOTAL_OBSERVED`, `TOTAL_BEFORE`,
`TOTAL_AFTER`, `DEVIANCE_BEFORE`, `DEVIANCE_AFTER`, `N_PIXELS`.
"""

import os
import time

import h5py
import healpy as hp
import numpy as np
from scipy.optimize import minimize

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import yso as yso_module

#: The six classes every catalogued source is one of (spec section 0.2),
#: in the fixed order every array and every `F_*`/`SIGMA_F_*` column of
#: this module shares.
CLASSES = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")

#: STAR/AGB/PAHC/GAL/H2S's own per-source count product ("source"
#: granule); YSO rides `prior.yso.law_area_integral` instead (module
#: docstring).
_STAR_FAMILY_CLASSES = ("STAR", "AGB", "PAHC", "GAL")

#: HEALPix nside 512 is the equal-area tessellation every occupied pixel
#: in this module shares (the same nside `prior.anchor_tiles`'s own
#: tiles product and `prior.yso.NSIDE_ANCHOR` use); the pixel's own solid
#: angle is one HEALPix constant, not a per-pixel read.
NSIDE = 512
OMEGA_PIX_DEG2 = float(hp.nside2pixarea(NSIDE, degrees=True))

#: The Poisson-precision bar the fitted total is graded against (brief
#: item 2): `TOTAL_AFTER` must land within this many `sqrt(TOTAL_OBSERVED)`
#: of `TOTAL_OBSERVED`.
TOTAL_MATCH_SIGMA = 3.0


# ---------------------------------------------------------------------------
# per-region reads: the occupied pixels, their catalogued source counts,
# and every source's own class counts (brief item 1)
# ---------------------------------------------------------------------------

def occupied_pixels(config, region):
    """`(pixels, n_i)`, ascending pixel order: the region's own occupied
    nside-512 pixels and each one's catalogued source count, read from
    the granule map's `association/region_healpix512` group (the one
    place a pixel's own catalogued row count is booked, independent of
    any downstream product's own source join)."""
    path = config_module.product_path(config, "granules", "sesna", "granule-map", "source")
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["region/REGION"][:]]
        if region not in names:
            raise ValueError("prior.levels: region %r is absent from granule map %s" % (region, path))
        code = int(f["region/REGION_CODE"][names.index(region)])
        assoc_pix = np.asarray(f["association/region_healpix512/HPX_PIX_512"][:], dtype=np.int64)
        assoc_code = np.asarray(f["association/region_healpix512/REGION_CODE"][:], dtype=np.int64)
        assoc_n = np.asarray(f["association/region_healpix512/N_SOURCE_ROWS"][:], dtype=np.int64)
    mask = (assoc_code == code) & (assoc_n > 0)
    order = np.argsort(assoc_pix[mask])
    pixels, n_i = assoc_pix[mask][order], assoc_n[mask][order]

    tiles_path = config_module.product_path(config, "bms", "anchors", "tiles", "hpx512", region=region)
    if not os.path.exists(tiles_path):
        raise FileNotFoundError(
            "prior.levels: no tiles product for region %r at %s -- run the "
            "'prior.anchor_tiles' RUNBOOK line first" % (region, tiles_path))
    with h5py.File(tiles_path, "r") as f:
        tiles_pixels = np.sort(np.asarray(f["HPX_PIX_512"][:], dtype=np.int64))
    if not np.array_equal(pixels, tiles_pixels):
        raise ValueError(
            "prior.levels: %r's occupied pixel set (granule map) disagrees with the "
            "tiles product's own pixel set -- both should list the same nside-512 "
            "pixels" % region)
    return pixels, n_i.astype(np.float64)


def per_source_counts(config, region):
    """`(src_pix, values)`: every catalogued source's own nside-512
    pixel, catalogue-row order, and its own STAR/AGB/PAHC/GAL/H2S counts
    plus `EPS_YSO` (`prior.counts_star_family`'s and `prior.counts_cloud`'s
    "source"-granule products)."""
    rs = access.region_slice(config, region)
    src_pix = np.asarray(rs["hpx_pix_512"], dtype=np.int64)

    star_path = config_module.product_path(config, "bms", "table", "counts-star-family",
                                            "source", region=region)
    star = access.per_source(config, region, star_path, ["N_STAR", "N_AGB", "N_PAHC", "N_GAL"])

    cloud_path = config_module.product_path(config, "bms", "table", "counts-cloud",
                                             "source", region=region)
    cloud = access.per_source(config, region, cloud_path, ["N_H2S", "EPS_YSO"])

    values = {
        "STAR": np.asarray(star["N_STAR"], dtype=np.float64),
        "AGB": np.asarray(star["N_AGB"], dtype=np.float64),
        "PAHC": np.asarray(star["N_PAHC"], dtype=np.float64),
        "GAL": np.asarray(star["N_GAL"], dtype=np.float64),
        "H2S": np.asarray(cloud["N_H2S"], dtype=np.float64),
        "EPS_YSO": np.asarray(cloud["EPS_YSO"], dtype=np.float64),
    }
    return src_pix, values


# ---------------------------------------------------------------------------
# each class's predicted count per pixel, before the level correction
# ---------------------------------------------------------------------------

def _pixel_means(pixels, src_pix, values):
    """`(n_pix,)`: the grouped mean of `values` (one per catalogued
    source) onto `pixels` (the region's own occupied set) -- every
    source's own index into `pixels` by `searchsorted` (a source's own
    pixel is occupied by construction, so this is an exact match, never a
    nearest lookup), summed and counted by `np.bincount` (CODING_RULES.md
    rule 8: no Python loop over sources or pixels)."""
    idx = np.searchsorted(pixels, src_pix)
    n_pix = pixels.size
    counts = np.bincount(idx, minlength=n_pix)
    sums = np.bincount(idx, weights=values, minlength=n_pix)
    if np.any(counts == 0):
        raise ValueError(
            "prior.levels: an occupied pixel has zero catalogued sources landing in it "
            "by the source-to-pixel join -- a join bug, not a normal condition")
    return sums / counts


#: The four IRAC bands whose own coverage fraction bounds a mosaic-edge
#: pixel's real surveyed area (the fifth band the coverage product
#: carries, MIPS M1, is not part of SESNA's own two-of-eight selection
#: footprint and is left out); the maximum of the four is the pixel's own
#: best-covered footprint, since a source only needs SOME band pair to be
#: catalogued.
_COVERAGE_BANDS = ("I1", "I2", "I3", "I4")


def covered_fraction(config, region, pixels):
    """`(n_pix,)`: each of `pixels`' own maximum coverage fraction across
    the four IRAC bands (`sky.derived.spitzer.coverage`'s own per-pixel
    product) -- a mosaic-edge pixel's own solid angle `Omega_i` is the
    full HEALPix cell regardless of how much of it the survey actually
    covers, so every predicted count must be scaled down by this fraction
    or the level fit reads the edge's own shortfall of catalogued sources
    as a lower class level than the region's interior really has."""
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage",
                                      "hpx512", region=region)
    with h5py.File(path, "r") as f:
        bands = [b.decode() if isinstance(b, bytes) else b for b in f["BANDS"][:]]
        irac_idx = [bands.index(b) for b in _COVERAGE_BANDS]
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac_max = np.max(np.asarray(f["FRAC"][:, irac_idx], dtype=np.float64), axis=1)
    order = np.argsort(cov_pix)
    cov_pix_sorted, frac_sorted = cov_pix[order], frac_max[order]
    loc = np.searchsorted(cov_pix_sorted, pixels)
    capped = np.minimum(loc, cov_pix_sorted.size - 1) if cov_pix_sorted.size else loc
    matched = cov_pix_sorted.size > 0 and np.all(cov_pix_sorted[capped] == pixels)
    if not matched:
        raise ValueError(
            "prior.levels: %r has an occupied nside-512 pixel absent from its own "
            "Spitzer coverage product %s -- every occupied pixel must carry a "
            "covered fraction" % (region, path))
    return frac_sorted[capped]


def restrict_to_covered(config, region, pixels, n_i, src_pix):
    """`(pixels, n_i, frac, src_keep)`: drops any occupied pixel with zero
    coverage across the four IRAC bands and reports, as a boolean mask
    aligned to `src_pix`, which catalogued sources land in a pixel that
    survives -- `covered_fraction`'s own denominator is exactly zero on a
    dropped pixel, so no level factor can explain a catalogued source
    there; these are sources selected through some other band pair
    entirely outside the IRAC mosaic (SESNA needs only two of eight
    bands), not the partly-covered mosaic-edge case the correction
    targets, and the fit cannot use a pixel whose own prediction is
    identically zero for every class."""
    frac = covered_fraction(config, region, pixels)
    keep_pix = frac > 0.0
    pixels_kept, n_i_kept, frac_kept = pixels[keep_pix], n_i[keep_pix], frac[keep_pix]

    idx = np.searchsorted(pixels_kept, src_pix)
    capped = np.clip(idx, 0, max(pixels_kept.size - 1, 0))
    src_keep = (pixels_kept.size > 0) & (pixels_kept[capped] == src_pix)
    return pixels_kept, n_i_kept, frac_kept, src_keep


def predicted_patterns(config, region, pixels, src_pix, values):
    """`{class: (n_pix,)}`: each of the six classes' own predicted count
    in every occupied pixel before the level correction (module
    docstring), scaled by the pixel's own covered fraction
    (`covered_fraction`) so a partly-covered mosaic-edge pixel's own
    prediction is not built on its full, uncovered solid angle."""
    frac = covered_fraction(config, region, pixels)
    patterns = {}
    for cls in _STAR_FAMILY_CLASSES + ("H2S",):
        patterns[cls] = _pixel_means(pixels, src_pix, values[cls]) * OMEGA_PIX_DEG2 * frac

    n_law_deg2 = yso_module.law_area_integral(config, region, pixels)
    mean_eps_yso = _pixel_means(pixels, src_pix, values["EPS_YSO"])
    patterns["YSO"] = n_law_deg2 * mean_eps_yso * OMEGA_PIX_DEG2 * frac
    return patterns


# ---------------------------------------------------------------------------
# the Poisson fit (brief item 2)
# ---------------------------------------------------------------------------

def _neg_log_likelihood_and_grad(theta, p_matrix, n):
    """`(nll, grad)`: the Poisson negative log-likelihood of `n` given
    `mu = p_matrix @ exp(theta)` (`theta` the six log-factors) and its
    analytic gradient in `theta` -- the `log(n_i!)` term is dropped (a
    constant, does not move the optimum)."""
    f = np.exp(theta)
    mu = np.maximum(p_matrix @ f, 1e-300)
    nll = float(np.sum(mu - n * np.log(mu)))
    resid = 1.0 - n / mu                       # d(nll)/d(mu_i)
    grad = (p_matrix.T @ resid) * f             # chain rule through mu = P @ exp(theta)
    return nll, grad


def fit_levels(patterns, n_i):
    """`(f, cov, mu)`: the six non-negative level factors maximising the
    region's own Poisson likelihood (module docstring), the observed-
    information covariance of `f` itself, and the fitted `mu_i`."""
    p_matrix = np.column_stack([patterns[c] for c in CLASSES])
    n_i = np.asarray(n_i, dtype=np.float64)

    theta0 = np.zeros(len(CLASSES))
    result = minimize(_neg_log_likelihood_and_grad, theta0, args=(p_matrix, n_i),
                      jac=True, method="L-BFGS-B")
    if not result.success:
        raise ValueError("prior.levels: the Poisson fit did not converge (%s)" % result.message)
    f = np.exp(result.x)

    mu = np.maximum(p_matrix @ f, 1e-300)
    # the observed information directly in f-space (mu is linear in f,
    # so this is the exact Hessian of the Poisson -log-likelihood in f,
    # not a delta-method transform of the log-factor Hessian):
    info = (p_matrix.T * (n_i / mu**2)) @ p_matrix
    cov = np.linalg.inv(info)
    return f, cov, mu


def poisson_deviance(n_i, mu_i):
    """The total Poisson deviance (a chi**2-like goodness-of-fit
    statistic) of `n_i` against `mu_i`, summed over the region's own
    occupied pixels: `2 . Sum_i [n_i ln(n_i / mu_i) - (n_i - mu_i)]`,
    with the `n_i ln(n_i/mu_i)` term taken as 0 where `n_i = 0`."""
    n_i = np.asarray(n_i, dtype=np.float64)
    mu_i = np.asarray(mu_i, dtype=np.float64)
    log_term = np.where(n_i > 0, n_i * np.log(n_i / mu_i), 0.0)
    return float(2.0 * np.sum(log_term - (n_i - mu_i)))


# ---------------------------------------------------------------------------
# write and read
# ---------------------------------------------------------------------------

def _output_path(config):
    return config_module.product_path(config, "bms", "table", "levels", "region")


def read(config, region):
    """The region's own row of the levels product: `F_*`/`SIGMA_F_*` as a
    dict, plus `CORR`, the totals and the deviances."""
    path = _output_path(config)
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["REGION"][:]]
        if region not in names:
            raise ValueError("prior.levels: %r has no row in %s" % (region, path))
        i = names.index(region)
        out = {"F_%s" % c: float(f["F_%s" % c][i]) for c in CLASSES}
        out.update({"SIGMA_F_%s" % c: float(f["SIGMA_F_%s" % c][i]) for c in CLASSES})
        out["CORR"] = np.asarray(f["CORR"][i], dtype=np.float64)
        for key in ("TOTAL_OBSERVED", "TOTAL_BEFORE", "TOTAL_AFTER",
                   "DEVIANCE_BEFORE", "DEVIANCE_AFTER", "N_PIXELS"):
            out[key] = float(f[key][i])
    return out


# ---------------------------------------------------------------------------
# report (rules 10, 11, 13)
# ---------------------------------------------------------------------------

def report(region, n_pixels, wall_s, f, sigma, corr, total_observed, total_before,
          total_after, dev_before, dev_after, mosaic_area_deg2, covered_area_deg2):
    lines = ["prior.levels: %s: %d occupied nside-512 pixels, wall=%.1fs"
            % (region, n_pixels, wall_s),
            "prior.levels: %s: mosaic area=%.4g deg^2, covered-fraction-weighted area=%.4g "
            "deg^2 (max of the four IRAC bands' own coverage fraction per pixel)"
            % (region, mosaic_area_deg2, covered_area_deg2)]
    for i, cls in enumerate(CLASSES):
        lines.append("prior.levels: %s: F_%s=%.4f +/- %.4f" % (region, cls, f[i], sigma[i]))
    i_yso, i_pahc = CLASSES.index("YSO"), CLASSES.index("PAHC")
    corr_yso_pahc = float(corr[i_yso, i_pahc])
    lines.append("prior.levels: %s: corr(YSO, PAHC)=%.3f -- %s"
                 % (region, corr_yso_pahc,
                    "the two classes separate in this fit" if abs(corr_yso_pahc) < 0.5
                    else "the two classes are poorly separated in this fit"))
    lines.append("prior.levels: %s: total observed=%.6g, total before=%.6g, total after=%.6g "
                "(bar: |after-observed|/sqrt(observed) < %.1f)"
                % (region, total_observed, total_before, total_after, TOTAL_MATCH_SIGMA))
    lines.append("prior.levels: %s: deviance before=%.6g, deviance after=%.6g (%d pixels)"
                % (region, dev_before, dev_after, n_pixels))
    return lines


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _build_one(config, region):
    t0 = time.time()
    all_pixels, all_n_i = occupied_pixels(config, region)
    src_pix, values = per_source_counts(config, region)
    pixels, n_i, frac, src_keep = restrict_to_covered(config, region, all_pixels, all_n_i, src_pix)
    n_dropped_pixels = all_pixels.size - pixels.size
    n_dropped_sources = int(np.count_nonzero(~src_keep))
    if n_dropped_pixels:
        print("prior.levels: %s: dropped %d of %d occupied pixels (%d catalogued sources) with "
             "zero coverage across the four IRAC bands -- sources caught by some other band "
             "pair entirely outside the IRAC mosaic, not usable by an IRAC-coverage correction"
             % (region, n_dropped_pixels, all_pixels.size, n_dropped_sources), flush=True)
    src_pix, values = src_pix[src_keep], {k: v[src_keep] for k, v in values.items()}
    patterns = predicted_patterns(config, region, pixels, src_pix, values)

    f, cov, mu_after = fit_levels(patterns, n_i)
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)

    p_matrix = np.column_stack([patterns[c] for c in CLASSES])
    mu_before = p_matrix @ np.ones(len(CLASSES))

    total_observed = float(np.sum(n_i))
    total_before = float(np.sum(mu_before))
    total_after = float(np.sum(mu_after))
    dev_before = poisson_deviance(n_i, mu_before)
    dev_after = poisson_deviance(n_i, mu_after)

    if abs(total_after - total_observed) / np.sqrt(total_observed) >= TOTAL_MATCH_SIGMA:
        raise ValueError(
            "prior.levels: %r's fitted total %.6g misses the observed total %.6g by more "
            "than %.1f sigma of Poisson counting noise -- the six classes' own patterns "
            "cannot jointly reproduce the region's catalogued source density"
            % (region, total_after, total_observed, TOTAL_MATCH_SIGMA))

    wall_s = time.time() - t0
    mosaic_area_deg2 = pixels.size * OMEGA_PIX_DEG2
    covered_area_deg2 = float(np.sum(frac)) * OMEGA_PIX_DEG2
    for line in report(region, pixels.size, wall_s, f, sigma, corr, total_observed,
                       total_before, total_after, dev_before, dev_after,
                       mosaic_area_deg2, covered_area_deg2):
        print(line, flush=True)

    rows = {"F_%s" % c: [f[i]] for i, c in enumerate(CLASSES)}
    rows.update({"SIGMA_F_%s" % c: [sigma[i]] for i, c in enumerate(CLASSES)})
    rows["CORR"] = corr[None, :, :]
    rows["TOTAL_OBSERVED"] = [total_observed]
    rows["TOTAL_BEFORE"] = [total_before]
    rows["TOTAL_AFTER"] = [total_after]
    rows["DEVIANCE_BEFORE"] = [dev_before]
    rows["DEVIANCE_AFTER"] = [dev_after]
    rows["N_PIXELS"] = [float(pixels.size)]
    return rows


def build(config, regions=None):
    """Writes the per-region level factors for `regions` (default: all
    thirty) into the one 30-row `bms/table/levels_table_region.hdf5`
    product, `regions`' own rows only (CODING_RULES.md rule 5c)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    path = _output_path(config)
    for region in region_names:
        rows = _build_one(config, region)
        tables_module.update_rows(path, [region], rows, granule="region")
        print("prior.levels: %s -> %s" % (region, path), flush=True)


if __name__ == "__main__":
    run(build)
