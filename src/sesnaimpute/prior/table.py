"""The prior table (SPEC_PRIORS.md section 0.2; IMPLEMENTATION.md section
5): the join, one row per catalogue source, in catalogue order, of every
scalar the fitter's callable needs to assemble `N_C(s)` and `lambda~_C`
for the six classes at that source. The table carries the numbers a
consumer reads and nothing else: no provenance, no manifest, one root
attribute `GRANULE` plus the three per-region scalars the YSO/H2S classes
are conditioned by (`M_LIM_8UM_1MYR`, `ETA`, `EPS_EXT`).

The table does no science of its own: every count, normaliser, node
bracket and ridge parameter already lives in an upstream product
(`prior.counts_star_family`'s STAR/AGB/PAHC/GAL counts and normalisers;
`prior.counts_cloud`'s YSO/H2S counts, normalisers and ridge; the adopted
column product; the curated catalogue; the granule map). This module only
reads each one in catalogue-row order and writes it once, so the fitter
opens one file per region instead of five.

`NODE_LO`, `NODE_W` and `GROUP` are computed independently in
`prior.counts_star_family` and `prior.counts_cloud`, both from the same
adopted column and the same region depth groups (IMPLEMENTATION.md
section 2's node bracket and section 1.3's nearest depth-group). Build
asserts the two products agree; a disagreement is a bug in one of them,
not a join choice, so it fails rather than picking one.
"""

import os
import time

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access

#: Every dataset the fitter reads off the table, in the order
#: IMPLEMENTATION.md section 5 lists them.
CONDITIONING_COLUMNS = (
    "A_COL_K", "A_COL_SIG_K", "A_COL_PROVENANCE",
    "HPX_PIX_512", "HPX_PIX_256", "HPX256_ROW",
    "TILE_ID", "GROUP", "S_SHIFT", "F_LIM_50_MJY",
    "NODE_LO", "NODE_W",
)
COUNT_COLUMNS = ("N_STAR", "N_AGB", "N_PAHC", "N_GAL", "N_YSO", "N_H2S")
NORMALISER_COLUMNS = ("Z_STAR", "Z_AGB", "Z_PAHC", "Z_GAL", "Z_YSO", "Z_H2S")
RIDGE_COLUMNS = ("RIDGE_INTERCEPT", "RIDGE_SLOPE", "RIDGE_WIDTH")
YSO_DIAGNOSTIC_COLUMNS = ("EPS_YSO", "EPS_YSO_3MYR", "N_YSO_3MYR", "N_LAW")
ALL_COLUMNS = (("NAME",) + CONDITIONING_COLUMNS + COUNT_COLUMNS + NORMALISER_COLUMNS
              + RIDGE_COLUMNS + YSO_DIAGNOSTIC_COLUMNS)

#: The per-region root attributes the fitter reads beside the columns
#: (SPEC_PRIORS.md section 6.1's law count, section 7's H2S amplitude):
#: computed once in `prior.counts_cloud` and copied through here.
REGION_ATTRS = ("M_LIM_8UM_1MYR", "ETA", "EPS_EXT")

_STAR_FAMILY_COLUMNS = ("TILE_ID", "GROUP", "S_SHIFT", "NODE_LO", "NODE_W",
                        "N_STAR", "N_AGB", "N_PAHC", "N_GAL",
                        "Z_STAR", "Z_AGB", "Z_PAHC", "Z_GAL")
_CLOUD_COLUMNS = ("GROUP", "NODE_LO", "NODE_W", "N_YSO", "N_H2S", "Z_YSO", "Z_H2S") \
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


def _assert_node_group_agree(region, star, cloud):
    """The one bug the join cannot fix by picking a side
    (IMPLEMENTATION.md section 5): `NODE_LO`/`NODE_W`/`GROUP` computed
    independently in `prior.counts_star_family` and `prior.counts_cloud`
    from the same adopted column and the same region depth groups must
    already agree exactly (up to the two products' own float32
    storage)."""
    node_lo_sf = np.rint(star["NODE_LO"]).astype(np.int64)
    node_lo_cl = np.rint(cloud["NODE_LO"]).astype(np.int64)
    if not np.array_equal(node_lo_sf, node_lo_cl):
        raise ValueError(
            "prior.table: %r's counts-star-family and counts-cloud products disagree on "
            "NODE_LO -- both are computed from the same adopted column, so this is a bug "
            "in one of the two upstream builds" % region)

    group_sf = np.rint(star["GROUP"]).astype(np.int64)
    group_cl = np.rint(cloud["GROUP"]).astype(np.int64)
    if not np.array_equal(group_sf, group_cl):
        raise ValueError(
            "prior.table: %r's counts-star-family and counts-cloud products disagree on "
            "GROUP -- both are computed from the same region depth groups, so this is a bug "
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


def _write(config, region, name, f_lim_50_mjy, rs, adopted, star, cloud, region_attrs):
    out_path = _output_path(config, region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        for key in REGION_ATTRS:
            f.attrs[key] = region_attrs[key]

        f.create_dataset("NAME", data=name)

        f.create_dataset("A_COL_K", data=adopted["A_COL_K"].astype(np.float32))
        f.create_dataset("A_COL_SIG_K", data=adopted["A_COL_SIG_K"].astype(np.float32))
        f.create_dataset("A_COL_PROVENANCE", data=adopted["A_COL_PROVENANCE"])
        f.create_dataset("HPX_PIX_512", data=rs["hpx_pix_512"].astype(np.int64))
        f.create_dataset("HPX_PIX_256", data=rs["hpx_pix_256"].astype(np.int64))
        f.create_dataset("HPX256_ROW", data=rs["hpx256_row"].astype(np.int64))
        f.create_dataset("TILE_ID", data=star["TILE_ID"].astype(np.int32))
        f.create_dataset("GROUP", data=np.rint(star["GROUP"]).astype(np.int32))
        f.create_dataset("S_SHIFT", data=star["S_SHIFT"].astype(np.float32))
        f.create_dataset("F_LIM_50_MJY", data=f_lim_50_mjy.astype(np.float32))
        f.create_dataset("NODE_LO", data=np.rint(star["NODE_LO"]).astype(np.int32))
        f.create_dataset("NODE_W", data=star["NODE_W"].astype(np.float32))

        for key in COUNT_COLUMNS:
            src = star if key in ("N_STAR", "N_AGB", "N_PAHC", "N_GAL") else cloud
            f.create_dataset(key, data=src[key].astype(np.float32))
        for key in NORMALISER_COLUMNS:
            src = star if key in ("Z_STAR", "Z_AGB", "Z_PAHC", "Z_GAL") else cloud
            f.create_dataset(key, data=src[key].astype(np.float32))
        for key in RIDGE_COLUMNS + YSO_DIAGNOSTIC_COLUMNS:
            f.create_dataset(key, data=cloud[key].astype(np.float32))
    return out_path


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


def _algebraic_check(region, out, adopted, star, cloud):
    """The join reproduces each input's column bit for bit (rule 11): max
    abs diff against the source products for `N_STAR`, `N_YSO`,
    `A_COL_K`."""
    checks = {
        "N_STAR": float(np.max(np.abs(out["N_STAR"] - star["N_STAR"].astype(np.float32)))),
        "N_YSO": float(np.max(np.abs(out["N_YSO"] - cloud["N_YSO"].astype(np.float32)))),
        "A_COL_K": float(np.max(np.abs(out["A_COL_K"] - adopted["A_COL_K"].astype(np.float32)))),
    }
    return checks


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
    t0 = time.time()
    rs = access.region_slice(config, region)
    n_sources = rs["n_sources"]

    name, f_lim_50_mjy = _name_and_limits(config, region)
    adopted = _adopted_column(config, region)
    star = _star_family(config, region)
    cloud, region_attrs = _cloud(config, region)

    _assert_node_group_agree(region, star, cloud)
    _assert_row_counts(region, n_sources, {
        "NAME": name, "F_LIM_50_MJY": f_lim_50_mjy,
        "A_COL_K": adopted["A_COL_K"], "N_STAR": star["N_STAR"], "N_YSO": cloud["N_YSO"],
    })

    out_path = _write(config, region, name, f_lim_50_mjy, rs, adopted, star, cloud, region_attrs)
    out = read(config, region)
    wall_s = time.time() - t0

    totals, area_deg2 = _region_totals(config, region, n_sources, out)
    checks = _algebraic_check(region, out, adopted, star, cloud)
    for line in report(region, n_sources, wall_s, totals, area_deg2, checks):
        print(line, flush=True)
    print("prior.table: %s -> %s" % (region, out_path), flush=True)
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
