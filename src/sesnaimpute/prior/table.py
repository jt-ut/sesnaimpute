"""The prior table (SPEC_PRIORS.md section 0.2; IMPLEMENTATION.md section
5): the join, one row per catalogue source, in catalogue order, of every
scalar the fitter's callable needs to assemble `N_C(s)` and `lambda~_C`
for the six classes at that source. The table carries the numbers a
consumer reads and nothing else: no provenance, no manifest, one root
attribute `GRANULE` plus the two per-region scalars the H2S class is
conditioned by (`ETA`, `EPS_EXT`).

The table does no science of its own: every count, normaliser, node
bracket and ridge parameter already lives in an upstream product
(`prior.counts_star_family`'s STAR/AGB/PAHC/GAL counts and normalisers;
`prior.counts_cloud`'s YSO/H2S counts, normalisers and ridge; the adopted
column product; the curated catalogue; the granule map). This module only
reads each one in catalogue-row order and writes it once, so the fitter
opens one file per region instead of five.

`NODE_LO`/`NODE_W` are computed independently in `prior.counts_star_family`
and `prior.counts_cloud`, both from the same adopted column
(IMPLEMENTATION.md section 2's node bracket). Build asserts the two
products agree; a disagreement is a bug in one of them, not a join
choice, so it fails rather than picking one. Depth groups no longer
exist (SPEC_PRIORS.md section 1.3): every class's selection is now
exact per source.
"""

import os
import time

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid as column_grid_module
from sesnaimpute.prior import counts_star_family as counts_star_family_module
from sesnaimpute.prior import levels as levels_module
from sesnaimpute.prior import star_shapes as star_shapes_module

#: The fixed-seed row subset the re-read check (brief item 1) compares
#: against every column's own source array -- small enough to check every
#: column, not just the three the region-wide algebraic check covers.
CHECK_SEED = 0
N_CHECK_ROWS = 20

#: Every dataset the fitter reads off the table, in the order
#: IMPLEMENTATION.md section 5 lists them.
CONDITIONING_COLUMNS = (
    "A_COL_K", "A_COL_SIG_K", "A_COL_PROVENANCE",
    "HPX_PIX_512", "HPX_PIX_256", "HPX256_ROW",
    "TILE_ID", "F_LIM_50_MJY",
    "NODE_LO", "NODE_W",
)
COUNT_COLUMNS = ("N_STAR", "N_AGB", "N_PAHC", "N_GAL", "N_YSO", "N_H2S")
NORMALISER_COLUMNS = ("Z_STAR", "Z_AGB", "Z_PAHC", "Z_GAL", "Z_YSO", "Z_H2S")

#: Each count column's own class, the `prior.levels` factor it is scaled
#: by before it is written (SPEC_PRIORS.md section 0.2's "The levels are
#: normalised to the survey"), and the root attribute that factor is
#: carried under.
_COUNT_CLASS = dict(zip(COUNT_COLUMNS, levels_module.CLASSES))
LEVEL_ATTRS = ("F_REGION",)
RIDGE_COLUMNS = ("RIDGE_INTERCEPT", "RIDGE_SLOPE", "RIDGE_WIDTH")
YSO_DIAGNOSTIC_COLUMNS = ("EPS_YSO", "EPS_YSO_3MYR", "N_YSO_3MYR", "N_LAW",
                         "M_LIM_8UM_1MYR", "IMF_FRAC_ABOVE_MLIM")
ALL_COLUMNS = (("NAME",) + CONDITIONING_COLUMNS + COUNT_COLUMNS + NORMALISER_COLUMNS
              + RIDGE_COLUMNS + YSO_DIAGNOSTIC_COLUMNS)

#: The per-region root attributes the fitter reads beside the columns
#: (SPEC_PRIORS.md section 7's H2S amplitude): computed once in
#: `prior.counts_cloud` and copied through here.
REGION_ATTRS = ("ETA", "EPS_EXT")

_STAR_FAMILY_COLUMNS = ("TILE_ID", "NODE_LO", "NODE_W",
                        "N_STAR", "N_AGB", "N_PAHC", "N_GAL",
                        "Z_STAR", "Z_AGB", "Z_PAHC", "Z_GAL")
_CLOUD_COLUMNS = ("NODE_LO", "NODE_W", "N_YSO", "N_H2S", "Z_YSO", "Z_H2S") \
    + RIDGE_COLUMNS + YSO_DIAGNOSTIC_COLUMNS


# ---------------------------------------------------------------------------
# per-region reads (brief items 1-4: NAME, the conditioning scalars, the
# six counts, the six normalisers, the YSO ridge and diagnostics)
# ---------------------------------------------------------------------------

def _name_and_limits(config, region):
    """`NAME` (the curated catalogue's own identifier) and the source's
    own 50%-completeness limit vector in all eight bands (`catalog.
    limits.limits`), both in catalogue row order."""
    curated_path = config_module.product_path(config, "catalog", "sesna", "sources",
                                               "source", region=region)
    name = access.per_source(config, region, curated_path, ["NAME"])["NAME"]
    f_lim_50_mjy = limits_module.limits(config, region)
    return name, f_lim_50_mjy


def _adopted_column(config, region):
    """`A_COL_K`, `A_COL_SIG_K`, `A_COL_PROVENANCE` (`sky.derived.
    column`'s per-source product), the conditioning column every class
    brackets against (SPEC_PRIORS.md section 0.2)."""
    path = config_module.product_path(config, "sky/derived", "adopted",
                                       "column", "source", region=region)
    return access.per_source(config, region, path,
                             ["A_COL_K", "A_COL_SIG_K", "A_COL_PROVENANCE"])


def _star_family(config, region):
    """STAR/AGB/PAHC/GAL's counts, normalisers, tile, node bracket, depth
    group and common-mode shift (`prior.counts_star_family`'s product)."""
    path = config_module.product_path(config, "bms", "table", "counts-star-family",
                                       "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.table: no counts-star-family product for region %r at %s -- "
            "run the 'prior.counts_star_family' RUNBOOK line first" % (region, path))
    return access.per_source(config, region, path, _STAR_FAMILY_COLUMNS)


def _cloud(config, region):
    """YSO/H2S's counts, normalisers, node bracket, depth group, ridge and
    diagnostics (`prior.counts_cloud`'s product), plus its three per-
    region attributes copied to the table's own root."""
    path = config_module.product_path(config, "bms", "table", "counts-cloud",
                                       "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.table: no counts-cloud product for region %r at %s -- "
            "run the 'prior.counts_cloud' RUNBOOK line first" % (region, path))
    cols = access.per_source(config, region, path, _CLOUD_COLUMNS)
    with h5py.File(path, "r") as f:
        region_attrs = {key: float(f.attrs[key]) for key in REGION_ATTRS}
    return cols, region_attrs


def _assert_node_agree(region, star, cloud):
    """The one bug the join cannot fix by picking a side
    (IMPLEMENTATION.md section 5): `NODE_LO`/`NODE_W` computed
    independently in `prior.counts_star_family` and `prior.counts_cloud`
    from the same adopted column must already agree exactly (up to the
    two products' own float32 storage)."""
    node_lo_sf = np.rint(star["NODE_LO"]).astype(np.int64)
    node_lo_cl = np.rint(cloud["NODE_LO"]).astype(np.int64)
    if not np.array_equal(node_lo_sf, node_lo_cl):
        raise ValueError(
            "prior.table: %r's counts-star-family and counts-cloud products disagree on "
            "NODE_LO -- both are computed from the same adopted column, so this is a bug "
            "in one of the two upstream builds" % region)

    if not np.allclose(star["NODE_W"], cloud["NODE_W"], atol=1e-4):
        raise ValueError(
            "prior.table: %r's counts-star-family and counts-cloud products disagree on "
            "NODE_W -- both are computed from the same adopted column, so this is a bug "
            "in one of the two upstream builds" % region)


def _assert_row_counts(region, n_sources, arrays):
    """Every input's own row count against the region's catalogued source
    count (IMPLEMENTATION.md section 5's "row count ... asserted at
    build"): `access.per_source` already refuses a mismatched column, so
    this is the belt for the one array (`F_LIM_50_MJY`, read directly off
    `catalog.limits`, not through the accessor) that bypasses it."""
    for name, arr in arrays.items():
        if arr.shape[0] != n_sources:
            raise ValueError(
                "prior.table: %r's %s has %d rows, region has %d catalogued sources"
                % (region, name, arr.shape[0], n_sources))


# ---------------------------------------------------------------------------
# write and read
# ---------------------------------------------------------------------------

def _output_path(config, region):
    return config_module.product_path(config, "bms", "table", "prior", "source", region=region)


def _write(config, region, name, f_lim_50_mjy, rs, adopted, star, cloud, region_attrs, level_factors):
    out_path = _output_path(config, region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        for key in REGION_ATTRS:
            f.attrs[key] = region_attrs[key]
        for key in LEVEL_ATTRS:
            f.attrs[key] = level_factors[key]

        f.create_dataset("NAME", data=name)

        f.create_dataset("A_COL_K", data=adopted["A_COL_K"].astype(np.float32))
        f.create_dataset("A_COL_SIG_K", data=adopted["A_COL_SIG_K"].astype(np.float32))
        f.create_dataset("A_COL_PROVENANCE", data=adopted["A_COL_PROVENANCE"])
        f.create_dataset("HPX_PIX_512", data=rs["hpx_pix_512"].astype(np.int64))
        f.create_dataset("HPX_PIX_256", data=rs["hpx_pix_256"].astype(np.int64))
        f.create_dataset("HPX256_ROW", data=rs["hpx256_row"].astype(np.int64))
        f.create_dataset("TILE_ID", data=star["TILE_ID"].astype(np.int32))
        f.create_dataset("F_LIM_50_MJY", data=f_lim_50_mjy.astype(np.float32))
        f.create_dataset("NODE_LO", data=np.rint(star["NODE_LO"]).astype(np.int32))
        f.create_dataset("NODE_W", data=star["NODE_W"].astype(np.float32))

        for key in COUNT_COLUMNS:
            src = star if key in ("N_STAR", "N_AGB", "N_PAHC", "N_GAL") else cloud
            factor = level_factors["F_REGION"]
            f.create_dataset(key, data=(src[key].astype(np.float64) * factor).astype(np.float32))
        for key in NORMALISER_COLUMNS:
            src = star if key in ("Z_STAR", "Z_AGB", "Z_PAHC", "Z_GAL") else cloud
            f.create_dataset(key, data=src[key].astype(np.float32))
        for key in RIDGE_COLUMNS + YSO_DIAGNOSTIC_COLUMNS:
            f.create_dataset(key, data=cloud[key].astype(np.float32))
    return out_path


def _recompute_z_columns(config, region, out_path, n_sources):
    """Overwrites `Z_STAR` ... `Z_H2S` in the just-written table with
    `prior.callable._z_by_quadrature`'s own value, called once per class
    for every source in the region (owner ruling, 2026-09-06, step
    C2a): `Z` is the normaliser of the density the fitter's callable
    actually reads, so it is computed by that SAME routine, here, once
    per survey source, rather than left as the counts stage's own
    smoothed-density normaliser or recomputed per fitter batch. A local
    import of `prior.callable` (not a module-level one): `callable`
    itself imports this module to read the table it is completing, so a
    top-level import here would be circular -- by the time this
    function actually runs the table this class needs already exists
    on disk (`_write`, just above in `_build_one`), because `SourcePrior.
    __init__` reads it. `_z_by_quadrature` already chunks over sources
    internally (`CODING_RULES.md` 10a), so the whole region's source
    count is passed in one call per class. Returns `{cls: Z array}` so
    the caller's re-read check compares against the SAME values written,
    not a second recomputation."""
    from sesnaimpute.prior import callable as callable_module

    all_rows = np.arange(n_sources, dtype=np.intp)
    z_by_class = {}
    for cls in callable_module.CLASSES:
        prior = callable_module.SourcePrior(config, region, cls)
        # STAR/AGB/PAHC/H2S's own `selection` gathers from `prepare`'s
        # batch tabulation (their EPS read); this is the whole region in
        # one `prepare` call, not `_z_by_quadrature`'s own internal
        # source-chunking (that only bounds the QUERY-point working set,
        # not this EPS gather, which is small survey-wide -- module note
        # on `prepare`).
        prior.prepare(all_rows)
        z_by_class[cls] = callable_module._z_by_quadrature(
            prior, cls, all_rows).astype(np.float32)
    with h5py.File(out_path, "r+") as f:
        for cls in callable_module.CLASSES:
            f["Z_%s" % cls.upper()][...] = z_by_class[cls]
    return z_by_class


def read(config, region):
    """Every dataset and per-region attribute of the region's prior table
    (IMPLEMENTATION.md section 5), as a plain dict keyed by dataset name
    (root attrs under their own names)."""
    path = _output_path(config, region)
    out = {}
    with h5py.File(path, "r") as f:
        for key in REGION_ATTRS:
            out[key] = float(f.attrs[key])
        for key in ALL_COLUMNS:
            out[key] = f[key][:]
    return out


# ---------------------------------------------------------------------------
# report (rules 10, 11, 13): row counts, the six region totals beside the
# catalogued source count, the algebraic acceptance check
# ---------------------------------------------------------------------------

def _mosaic_area_deg2(config, region):
    """The region's own mosaic area, deg**2, from the I2-band coverage
    fraction summed over its nside-512 pixels (`sky.derived.coverage`) --
    the area the region-totals check needs to turn a per-source count
    density into an expected catalogued count."""
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage",
                                       "hpx512", region=region)
    with h5py.File(path, "r") as f:
        bands = [b.decode() if isinstance(b, bytes) else b for b in f["BANDS"][:]]
        frac = f["FRAC"][:, bands.index("I2")]
    pix_area_deg2 = hp.nside2pixarea(512, degrees=True)
    return float(np.sum(frac) * pix_area_deg2)


def _region_totals(config, region, n_sources, out):
    """Sum of each count over the region's sources times the region's
    mosaic area per source (`area_deg2 / n_sources`): the six expected
    catalogued counts per region -- algebraically `mean(N_C) *
    area_deg2`, a sanity scale against the region's own catalogued source
    count, not an identity (most catalogued sources are field stars)."""
    area_deg2 = _mosaic_area_deg2(config, region)
    area_per_source = area_deg2 / n_sources
    return {key: float(np.sum(out[key]) * area_per_source) for key in COUNT_COLUMNS}, area_deg2


def _algebraic_check(region, out, adopted, star, cloud, level_factors):
    """The join reproduces each input's column bit for bit (rule 11), over
    every source in the region: `N_STAR` and `N_YSO` are level-scaled
    (SPEC_PRIORS.md section 0.2), so the comparand is the source product's
    own value times the region's own `F_REGION` before the max abs diff is
    taken; `A_COL_K` is never scaled and compares directly."""
    factor = level_factors["F_REGION"]
    checks = {
        "N_STAR": float(np.max(np.abs(
            out["N_STAR"] - (star["N_STAR"].astype(np.float64) * factor).astype(np.float32)))),
        "N_YSO": float(np.max(np.abs(
            out["N_YSO"] - (cloud["N_YSO"].astype(np.float64) * factor).astype(np.float32)))),
        "A_COL_K": float(np.max(np.abs(out["A_COL_K"] - adopted["A_COL_K"].astype(np.float32)))),
    }
    return checks


def _fixed_seed_rows(n_sources):
    """`N_CHECK_ROWS` fixed-seed row indices into the region's own
    catalogue order, ascending, so every run checks the identical rows
    regardless of platform or thread count."""
    n = min(N_CHECK_ROWS, n_sources)
    return np.sort(np.random.RandomState(CHECK_SEED).choice(n_sources, size=n, replace=False))


def _reread_check(region, out, name, f_lim_50_mjy, rs, adopted, star, cloud, level_factors, rows,
                  z_by_class):
    """Every column of the table re-read from its own source array at a
    fixed-seed subset of rows and compared against what the build wrote
    (brief item 2's check for this stage: the module docstring says the
    table does no science, only a join, so every column must reproduce
    its source exactly). The six counts are compared against the source
    value times the region's own `F_REGION`; every other column is the
    source value through the same dtype cast `_write` applies. `Z_*` is
    the one exception (step C2a): its source is no longer the counts
    stage's own column but `_recompute_z_columns`'s own `z_by_class`, the
    SAME array just written, not a second recomputation -- comparing the
    table against a fresh quadrature call would only re-check floating
    determinism, not the join. Returns the worst absolute difference
    across every numeric column (bar 0)."""
    factor = level_factors["F_REGION"]
    expected = {
        "A_COL_K": adopted["A_COL_K"].astype(np.float32),
        "A_COL_SIG_K": adopted["A_COL_SIG_K"].astype(np.float32),
        "A_COL_PROVENANCE": adopted["A_COL_PROVENANCE"],
        "HPX_PIX_512": rs["hpx_pix_512"].astype(np.int64),
        "HPX_PIX_256": rs["hpx_pix_256"].astype(np.int64),
        "HPX256_ROW": rs["hpx256_row"].astype(np.int64),
        "TILE_ID": star["TILE_ID"].astype(np.int32),
        "F_LIM_50_MJY": f_lim_50_mjy.astype(np.float32),
        "NODE_LO": np.rint(star["NODE_LO"]).astype(np.int32),
        "NODE_W": star["NODE_W"].astype(np.float32),
    }
    for key in ("N_STAR", "N_AGB", "N_PAHC", "N_GAL"):
        expected[key] = (star[key].astype(np.float64) * factor).astype(np.float32)
    for key in ("N_YSO", "N_H2S"):
        expected[key] = (cloud[key].astype(np.float64) * factor).astype(np.float32)
    for key in NORMALISER_COLUMNS:
        expected[key] = z_by_class[key[2:].lower()]
    for key in RIDGE_COLUMNS + YSO_DIAGNOSTIC_COLUMNS:
        expected[key] = cloud[key].astype(np.float32)

    worst = 0.0
    for key, arr in expected.items():
        table_val = np.asarray(out[key])[rows].astype(np.float64)
        src_val = np.asarray(arr)[rows].astype(np.float64)
        diff = float(np.max(np.abs(table_val - src_val)))
        if diff > 0.0:
            raise ValueError(
                "prior.table: %r's re-read check: column %s differs from its source "
                "product at a fixed-seed row by %.3g (bar 0)" % (region, key, diff))
        worst = max(worst, diff)

    if not np.array_equal(np.asarray(out["NAME"])[rows], np.asarray(name)[rows]):
        raise ValueError("prior.table: %r's re-read check: NAME differs from the curated "
                         "catalogue at a fixed-seed row" % region)
    return worst


#: The AGB photospheric-ratio diagnostic (below) is report-only, so it is
#: evaluated on a fixed-seed subsample rather than every source -- the
#: same reasoning `prior.selection`'s own Monte Carlo subsample uses
#: (IMPLEMENTATION.md section 4): measured at ~7 ms/source unsampled
#: (bicubic shape evaluation per source, batched but still per-source
#: work), which would run to hours at Cygnus X's 3.3M sources; capping
#: the sample bounds the cost regardless of the region's own size (brief
#: item 3's fix for the one part of this stage that scales with n_source).
AGB_RATIO_SEED = 0
AGB_RATIO_MAX_SOURCES = 10000


def _agb_photospheric_ratio(config, region, adopted, star, n_sources):
    """SPEC_PRIORS.md section 3's reported lower bound: on a fixed-seed
    subsample of at most `AGB_RATIO_MAX_SOURCES` sources, the AGB count
    recomputed with `prior.star_selection`'s `EPS_AGB_PHOTOSPHERE` (the
    bare-photosphere selection) in place of the dusty `EPS_AGB`, as the
    ratio to the dusty `N_AGB` this table actually carries over the SAME
    subsample -- one printed number per region (brief item 2), never a
    table column. Mirrors `prior.counts_star_family.family_counts`'s own
    two-component-mixture quadrature exactly (the shape's stored tile
    array at its two bracketing width nodes per component, the selection
    read with the per-source, per-component kernel shift folded into ITS
    query point instead of the density's, the two components combined by
    the mixture weight `w`) -- with only the selection array swapped for
    the photospheric one; not batched, since the subsample is already
    capped."""
    n_sample = min(AGB_RATIO_MAX_SOURCES, n_sources)
    rows = np.sort(np.random.RandomState(AGB_RATIO_SEED).choice(n_sources, size=n_sample, replace=False))

    shape = star_shapes_module.read(config, region, "agb")
    sel_path = config_module.product_path(config, "bms", "star", "selection", "source", region=region)
    with h5py.File(sel_path, "r") as f:
        eps_photo = f["EPS_AGB_PHOTOSPHERE"][rows].astype(np.float64)
        x_ladder = f["X_LADDER"][:].astype(np.float64)
        b_grid = f["LOG10_B_GRID_AGB"][:].astype(np.float64)
    wb = counts_star_family_module.selection_on_shape_grid(shape, x_ladder, b_grid)

    a_col = adopted["A_COL_K"].astype(np.float64)[rows]
    sigma_col = adopted["A_COL_SIG_K"].astype(np.float64)[rows]
    map_class = np.asarray(adopted["A_COL_PROVENANCE"])[rows]
    tile_id = star["TILE_ID"].astype(np.int64)[rows]
    amp = counts_star_family_module.family_amplitude(config, region, "agb", {"tile_id": tile_id})

    kern = shape.kern
    w_mix, mu_mix, sigma_mix = kern.mixture(a_col, sigma_col, map_class)
    log_shape_nodes = np.log(shape.shape_nodes)

    z_photo = np.zeros(n_sample, dtype=np.float64)
    for k in range(2):
        i_lo, t_w = column_grid_module.bracket(np.log(sigma_mix[:, k]), log_shape_nodes)
        i_hi = np.minimum(i_lo + 1, shape.shape_nodes.size - 1)
        d_lo = shape.density_table[tile_id, i_lo]
        d_hi = shape.density_table[tile_id, i_hi]
        dens_k = (1.0 - t_w)[:, None, None] * d_lo + t_w[:, None, None] * d_hi
        eps_grid_k = counts_star_family_module._shift_and_interp_eps(
            eps_photo, x_ladder, wb, shape.x_centers, mu_mix[:, k])
        weight_k = w_mix if k == 0 else (1.0 - w_mix)
        z_photo += weight_k * (dens_k * eps_grid_k).sum(axis=(1, 2))

    n_dusty_sample = float(np.sum(star["N_AGB"][rows]))
    return float(np.sum(amp * z_photo)) / n_dusty_sample if n_dusty_sample > 0.0 else float("nan")


def _assert_class_probabilities_identity(config, region, out, level_factors, n_sources):
    """The two identities SPEC_PRIORS.md section 0.2 states (the paragraph
    "The levels are normalised to the survey"), both recomputed directly
    from this table's own written columns (brief item 1's check for the
    levels stage: "recompute the region total of the six scaled counts
    ... directly from the table"), never re-derived from the upstream
    counts products:

    1. The class probability the fitter uses, `N_C(s) / Sum_C' N_C'(s)`,
       sums over the region's own sources to exactly the source count
       (each source's own six probabilities already sum to one) --
       asserted, bar 0.
    2. The six already-levelled counts, unscaled by `F_REGION` to recover
       `prior.levels.predicted_patterns`'s own "before" values (EPS_YSO
       is never scaled) and integrated by the SAME per-pixel rule
       `prior.levels.region_factor` fit against, sum to the region's own
       catalogued, IRAC-covered source count exactly -- this is the
       identity the level factor was built to satisfy; recomputing it
       from the table catches a join bug the factor's own report cannot
       see. Asserted, bar 0 (float32 storage roundoff only)."""
    n_c = np.stack([out[key].astype(np.float64) for key in COUNT_COLUMNS], axis=1)
    denom = np.sum(n_c, axis=1)
    if np.any(denom <= 0.0):
        raise ValueError(
            "prior.table: %r has %d source(s) with zero total count across all six "
            "classes -- the class probability is undefined there"
            % (region, int(np.count_nonzero(denom <= 0.0))))
    class_prob_sum = np.sum(n_c / denom[:, None], axis=0)
    total = float(np.sum(class_prob_sum))
    if abs(total - n_sources) > max(1e-3, 1e-6 * n_sources):
        raise ValueError(
            "prior.table: %r's six class probabilities sum to %.6g, not the %d "
            "catalogued sources exactly" % (region, total, n_sources))

    factor = level_factors["F_REGION"]
    src_pix = out["HPX_PIX_512"].astype(np.int64)
    all_pixels, all_n_i = levels_module.occupied_pixels(config, region)
    pixels, n_i, _frac, src_keep = levels_module.restrict_to_covered(config, region, all_pixels, all_n_i, src_pix)
    values = {cls: (out[key].astype(np.float64) / factor)[src_keep]
             for key, cls in zip(COUNT_COLUMNS, levels_module.CLASSES)}
    values["EPS_YSO"] = out["EPS_YSO"].astype(np.float64)[src_keep]
    patterns = levels_module.predicted_patterns(config, region, pixels, src_pix[src_keep], values)

    # Flagged, not asserted (CODING_RULES.md standing direction: flag a
    # data-quality finding, do not block the build on it): the region's
    # own `prior.levels` deviance-after is already reported far above its
    # naive Poisson expectation (real pixel-to-pixel scatter the six
    # patterns do not capture), so a per-class miss here by a few sigma
    # of the naive sqrt(class total) is the SAME overdispersion, not a
    # fresh bug -- printed for every class so the miss is visible, never
    # silently dropped.
    class_prob_sum_covered = np.sum((n_c / denom[:, None])[src_keep], axis=0)
    for i, key in enumerate(COUNT_COLUMNS):
        cls = _COUNT_CLASS[key]
        integrated = factor * float(np.sum(patterns[cls]))
        observed = float(class_prob_sum_covered[i])
        n_sigma = abs(observed - integrated) / np.sqrt(max(integrated, 1.0))
        print("prior.table: %s: %s prob-sum %.1f against integrated %.1f (%.1f sigma) -- a diagnostic "
              "of the count's spatial pattern, reported not enforced" % (region, cls, observed, integrated, n_sigma), flush=True)

    total_after = factor * float(sum(np.sum(p) for p in patterns.values()))
    observed_total = float(np.sum(n_i))
    diff = abs(total_after - observed_total)
    if diff > max(1e-2, 1e-6 * observed_total):
        raise ValueError(
            "prior.table: %r's six scaled counts, integrated over the region directly from "
            "the table, sum to %.6g against %d catalogued (IRAC-covered) sources -- the levels "
            "identity does not hold" % (region, total_after, int(observed_total)))
    print("prior.table: %s: levels identity from the table: six scaled counts integrate to "
          "%.4f against %d catalogued sources (diff %.3g, bar 0)"
          % (region, total_after, int(observed_total), diff), flush=True)
    return diff


def report(region, n_sources, wall_s, totals, area_deg2, checks):
    lines = [
        "prior.table: %s: %d catalogued sources, wall=%.1fs" % (region, n_sources, wall_s),
        "prior.table: %s: mosaic area=%.4g deg^2" % (region, area_deg2),
    ]
    for key in COUNT_COLUMNS:
        lines.append("prior.table: %s: expected catalogued %s=%.4g vs %d catalogued sources"
                     % (region, key, totals[key], n_sources))
    for key, dev in checks.items():
        lines.append("prior.table: %s: algebraic check %s: max abs diff=%.3g (bar 0)"
                     % (region, key, dev))
    return lines


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _build_one(config, region):
    st = progress.Stage("prior.table", region)
    t0 = time.time()
    rs = access.region_slice(config, region)
    n_sources = rs["n_sources"]

    name, f_lim_50_mjy = _name_and_limits(config, region)
    adopted = _adopted_column(config, region)
    star = _star_family(config, region)
    cloud, region_attrs = _cloud(config, region)
    level_factors = levels_module.read(config, region)

    _assert_node_agree(region, star, cloud)
    _assert_row_counts(region, n_sources, {
        "NAME": name, "F_LIM_50_MJY": f_lim_50_mjy,
        "A_COL_K": adopted["A_COL_K"], "N_STAR": star["N_STAR"], "N_YSO": cloud["N_YSO"],
    })

    out_path = _write(config, region, name, f_lim_50_mjy, rs, adopted, star, cloud,
                      region_attrs, level_factors)

    # `Z_STAR` ... `Z_H2S`: overwritten here with `prior.callable`'s own
    # `_z_by_quadrature`, once per class for every source in the region
    # (owner ruling, 2026-09-06, step C2a) -- `_write` above still puts
    # the counts stage's own Z there first only because the column has
    # to exist (and `SourcePrior.__init__`, which this needs, reads the
    # table) before it can be overwritten in place.
    t_z0 = time.time()
    z_by_class = _recompute_z_columns(config, region, out_path, n_sources)
    wall_z_s = time.time() - t_z0

    out = read(config, region)
    wall_build_s = time.time() - t0

    t1 = time.time()
    _assert_class_probabilities_identity(config, region, out, level_factors, n_sources)

    rows = _fixed_seed_rows(n_sources)
    worst = _reread_check(region, out, name, f_lim_50_mjy, rs, adopted, star, cloud, level_factors,
                          rows, z_by_class)
    print("prior.table: %s: re-read check, %d fixed-seed rows, every column: worst abs diff=%.3g (bar 0)"
          % (region, rows.size, worst), flush=True)
    print("prior.table: %s: Z recompute (prior.callable._z_by_quadrature, %d sources x 6 classes): "
         "%.1fs" % (region, n_sources, wall_z_s), flush=True)

    agb_ratio = _agb_photospheric_ratio(config, region, adopted, star, n_sources)
    print("prior.table: %s: AGB photospheric-selection lower bound: N_AGB(photosphere)/N_AGB(dusty) = %.4f"
          % (region, agb_ratio), flush=True)
    wall_check_s = time.time() - t1

    totals, area_deg2 = _region_totals(config, region, n_sources, out)
    checks = _algebraic_check(region, out, adopted, star, cloud, level_factors)
    wall_s = time.time() - t0
    st.done(out_path, n_sources=n_sources, area_deg2=area_deg2)
    for line in report(region, n_sources, wall_s, totals, area_deg2, checks):
        print(line, flush=True)
    print("prior.table: %s: wall split: build+write %.1fs, report-only checks %.1fs (of %.1fs total)"
          % (region, wall_build_s, wall_check_s, wall_s), flush=True)
    return out_path


def build(config, regions=None):
    """Writes the per-region prior table for `regions` (default: all
    thirty), one product per region, built once from the upstream
    per-source products (module docstring)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        _build_one(config, region)


if __name__ == "__main__":
    run(build)
