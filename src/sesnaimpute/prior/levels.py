"""The one per-region level factor (SPEC_PRIORS.md section 0.2, the
paragraph "The levels are normalised to the survey"; IMPLEMENTATION.md
section 6, stage 12a): each class's count is calibrated on something
outside SESNA, and their sum was never held to SESNA's own total. This
module fits, per region, the single non-negative scalar that brings the
sum of the six classes' own predicted count patterns -- positions and
counts only, never a label -- to the region's own catalogued source
count: every catalogued source is one of the six classes, so the six
counts, integrated over the region, must equal the number of sources
catalogued there. The same factor multiplies every class, so no source
moves between classes and no source's class probability changes; the
per-class source-sum / area-integral figures (below) are printed
SAMPLING diagnostics of how well each class's own spatial pattern
follows the catalogue's -- not a level check, since the level factor
`f` above is the only thing this module fits -- never a second
correction. For YSO in particular the figure runs above 1 in cluster
regions by construction: the law rises with the column squared, so
sources concentrated in the densest pixels sum to more than the
pixel-mean-times-area integral would predict there.

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

The one factor `f = sum(n_i) / sum_i sum_C P_C,i` is the closed-form
Poisson-likelihood maximiser for a single multiplicative scalar on a
linear predictor, so no iteration is needed. The Poisson deviance before
and after is reported as the goodness-of-fit diagnostic; the per-class
source-sum / area-integral SAMPLING diagnostic (not a level check; YSO's
own runs above 1 in cluster regions by construction) is reported per
class (module `read`'s `RATIO_<class>`).

Writes one 30-row product, `bms/table/levels_table_region.hdf5`: root
attr `GRANULE = "region"`; `F_REGION`, `RATIO_STAR` ... `RATIO_H2S`,
`TOTAL_OBSERVED`, `TOTAL_BEFORE`, `DEVIANCE_BEFORE`, `DEVIANCE_AFTER`,
`N_PIXELS`.

Memory: this module's own per-source working set is exactly what the
fit needs, six counts and a pixel index, `values.values()` plus
`src_pix` in `per_source_counts` -- measured at 64 bytes/source (5.2 MB
on Perseus's 80,496 sources), not a source of region-to-region growth.
Measured instead: `predicted_patterns`'s YSO term calls `prior.yso.
law_area_integral`, which reads the Herschel Gould Belt Survey FITS
maps overlapping the region's own sky footprint through a joblib worker
pool (`prior.yso._herschel_pixel_stats`, budgeted at up to 6 GB across
workers) -- a cost set by how much HGBS-mapped sky the region's own
bounding box overlaps, not by its source count or its occupied-pixel
count. Isolating that one call (`_herschel_pixel_stats` +
`_planck_parent_column`) on Perseus's 705 occupied pixels alone
reproduces essentially the whole region's peak (924 of 1,032/1,090 MB
measured for the full build): the "13 kB/source" figure is an artefact
of dividing this fixed, region-footprint-driven cost by Perseus's own
source count, not a real per-source rate, and it does not extrapolate
by source count to Cygnus X -- a region's own HGBS overlap does not
grow with its catalogue size (Cygnus X, at ~1.4 kpc, is likely mostly
or entirely outside the nearby-cloud HGBS footprint that drives this
cost at all). Splitting the YSO call into smaller pixel batches was
tried and rejected: `_herschel_pixel_stats` does not cache maps across
calls, so each extra call re-opens and re-reduces the same FITS files,
trading a bounded memory win for an unbounded, multiplicative wall-time
loss (a Perseus run in 64-pixel batches did not finish in 120 s, against
~11 s unbatched). The actual fix belongs in `prior.yso._herschel_
pixel_stats` (a tighter pool budget, or streaming the block-reduce so a
cutout's own float32 array is not held whole) -- outside this module's
own file and this pass's scope.
"""

import os
import time

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import progress
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
    cloud = access.per_source(config, region, cloud_path, ["N_H2S", "N_YSO", "EPS_YSO"])

    values = {
        "STAR": np.asarray(star["N_STAR"], dtype=np.float64),
        "AGB": np.asarray(star["N_AGB"], dtype=np.float64),
        "PAHC": np.asarray(star["N_PAHC"], dtype=np.float64),
        "GAL": np.asarray(star["N_GAL"], dtype=np.float64),
        "H2S": np.asarray(cloud["N_H2S"], dtype=np.float64),
        "YSO": np.asarray(cloud["N_YSO"], dtype=np.float64),
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

def region_factor(patterns, n_i):
    """`(f, mu_before)`: the one scalar per region (SPEC_PRIORS.md 0.2, owner
    2026-09-06) that makes the six counts, integrated over the region,
    sum to the number of catalogued sources: `f = sum(n_i) / sum_i sum_C
    P_C,i`. The same factor multiplies every class, so no source moves
    between classes and no class probability changes; the factor and the
    total before it are the diagnostic of the calibrations' absolute level."""
    p_matrix = np.column_stack([patterns[c] for c in CLASSES])
    mu_before = p_matrix @ np.ones(len(CLASSES))
    total_before = float(np.sum(mu_before))
    if total_before <= 0.0:
        raise ValueError("prior.levels: the six counts integrate to zero over the region")
    f = float(np.sum(n_i)) / total_before
    return f, mu_before


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
    """The region's own row of the levels product as a dict: F_REGION (the one
    scalar), TOTAL_OBSERVED, TOTAL_BEFORE, DEVIANCE_BEFORE/AFTER, N_PIXELS and
    the per-class diagnostic ratios RATIO_<class>."""
    with h5py.File(_output_path(config), "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["REGION"][:]]
        i = names.index(region)
        keys = ["F_REGION", "TOTAL_OBSERVED", "TOTAL_BEFORE", "DEVIANCE_BEFORE",
                "DEVIANCE_AFTER", "N_PIXELS"] + ["RATIO_%s" % c for c in CLASSES]
        out = {k: float(f[k][i]) for k in keys}
    if not np.isfinite(out["F_REGION"]):
        raise ValueError("prior.levels: no factor built for region %r -- run the RUNBOOK line "
                         "sesnaimpute.prior.levels for it" % region)
    return out


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

    f, mu_before = region_factor(patterns, n_i)
    mu_after = f * mu_before
    total_observed = float(np.sum(n_i))
    total_before = float(np.sum(mu_before))
    dev_before = poisson_deviance(n_i, mu_before)
    dev_after = poisson_deviance(n_i, mu_after)

    # the per-class SAMPLING diagnostic (reported, never enforced, not a
    # level check -- that is `f` above): the prior class probability
    # summed over the region's sources against the class's integrated
    # count after the factor; a ratio away from 1 says the count's
    # spatial pattern does not follow the catalogue's source density.
    # YSO's own ratio runs above 1 in cluster regions by construction
    # (the law rises with the column squared, so sources concentrated in
    # the densest pixels sum to more than the pixel mean predicts).
    n_tot = np.zeros(src_pix.size)
    for c in CLASSES:
        n_tot += values[c]
    ratios = {}
    for c in CLASSES:
        prob_sum = float(np.sum(np.where(n_tot > 0, values[c] / np.where(n_tot > 0, n_tot, 1.0), 0.0)))
        integrated = f * float(np.sum(patterns[c]))
        ratios[c] = prob_sum / integrated if integrated > 0 else np.nan

    wall_s = time.time() - t0
    mosaic_area_deg2 = pixels.size * OMEGA_PIX_DEG2
    covered_area_deg2 = float(np.sum(frac)) * OMEGA_PIX_DEG2
    print("prior.levels: %s: %d pixels, %.1f s; factor f = %.4f (counts integrate to %.1f "
          "against %d catalogued sources); deviance before/after %.0f/%.0f; mosaic %.2f deg2, "
          "covered-weighted %.2f deg2" % (region, pixels.size, wall_s, f, total_before,
          int(total_observed), dev_before, dev_after, mosaic_area_deg2, covered_area_deg2), flush=True)
    print("prior.levels: %s: source-sum / area-integral per class (sampling diagnostic, "
          "not a level check; YSO's own > 1 in cluster regions by construction): %s"
          % (region, ", ".join("%s %.2f" % (c, ratios[c]) for c in CLASSES)), flush=True)

    rows = {"F_REGION": [f], "TOTAL_OBSERVED": [total_observed], "TOTAL_BEFORE": [total_before],
            "DEVIANCE_BEFORE": [dev_before], "DEVIANCE_AFTER": [dev_after],
            "N_PIXELS": [float(pixels.size)]}
    rows.update({"RATIO_%s" % c: [ratios[c]] for c in CLASSES})
    return rows


def build(config, regions=None):
    """Writes the per-region level factors for `regions` (default: all
    thirty) into the one 30-row `bms/table/levels_table_region.hdf5`
    product, `regions`' own rows only (CODING_RULES.md rule 5c)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    path = _output_path(config)
    for region in region_names:
        with progress.Stage("prior.levels", region) as st:
            rows = _build_one(config, region)
            tables_module.update_rows(path, [region], rows, granule="region")
            st.done(path, f_region=rows["F_REGION"][0], n_pixels=int(rows["N_PIXELS"][0]))


if __name__ == "__main__":
    run(build)
