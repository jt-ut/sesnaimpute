"""The per-pixel column distribution `Q_pix` and its node representation
(C23 brief, `briefs/round4_2026-10-20/C23_within_pixel_column.md`).

The finding the brief ships (C18 follow-up 7, measured Gaia-free): the
field-star model used to dim every simulated star by `u * A_PIX_K`,
`A_PIX_K` being a pixel's MEAN source column -- but real stars see the
pixel's own internal structure, and the model's missing red tail (Ks
12-14, `A_K` > the pixel mean) is what drags the Gaia-2MASS joint weight
down. This module replaces the single pixel-mean column with a small,
explicit representation of the pixel's own column DISTRIBUTION, so every
consumer that used to dim at one value instead averages over this
distribution's own nodes.

Per nside-512 pixel (the union of every anchor pixel, `population.
anchor_tiles._region_pixel_columns`'s own SESNA-source-bearing set, and
every pixel `bmstp.atlas` admits, `catalog.limits.completeness`'s own
`HPX_PIX_512` -- a strict superset of the anchor set, measured: every
anchor pixel is in the admitted set, never the reverse), the column
distribution `Q_pix`:

  - where the HGBS map covers the pixel (`population.yso.
    _herschel_cell_matches`, the SAME cell reader `population.yso.
    law_area_integral` reads its own area integral from): the
    area-weighted SET of the pixel's own 36" cells' `A_K` -- every
    matched cell carries equal weight (the maps are already block-
    reduced to one common cell size, `sky.derived.planck_column.
    BLOCK_TARGET_ARCSEC`, so "area-weighted" is a plain equal-weight
    set, no further weighting term);
  - elsewhere, a lognormal about the pixel's own `A_PIX_K` (below) with
    the sightline's own Planck sub-beam width (`population.kernel.Kernel.
    read(config).params(a_col, 0, "planck")`, the SAME structural width
    `population.yso._kernel_second_moment_planck` reads off `sky.derived.
    subbeam`'s own fitted mixture): `T = A_PIX_K * 10**(mu + sigma*Z)`,
    `Z` standard normal, `mu`/`sigma` the kernel's own exact mixture mean
    and width in log10 T at that column.

`A_PIX_K`, the per-pixel anchor the distribution is centred on: the
SESNA-source mean (`anchor_tiles._region_pixel_columns`'s own
construction) where the pixel carries at least one SESNA source; for the
admitted-but-sourceless pixels in `bmstp.atlas`'s own wider set (no
mean to form), the pixel's own parent nside-256 sightline's adopted
extinction column (`sky/derived/adopted/extinction_adopted_sightline.hdf5`,
the SAME product `bmstp.atlas._pixel_column` reads for exactly those
pixels) -- never a new definition, the two existing per-pixel/per-
sightline column readings this project already has, joined where each
applies.

Represented as `N_COL_NODES` equal-weight quantile nodes (`_quantile_
nodes`): the distribution's own value at quantile `(i + 0.5) / N`, for
`i = 0 .. N-1` -- every consumer that used to read one pixel value reads
`N_COL_NODES` instead, weight `1 / N_COL_NODES` each (`population.
anchor_tiles._pixel_block`, `population.star_population._build_one_tile`,
`bmstp.atlas`'s STAR/AGB/PAHC quadrature).

Identity (the brief's own): at `N_COL_NODES = 1`, the quantile asked for
is `0.5` -- the median -- which is NOT in general `A_PIX_K` on the HGBS
branch (a real map's own median cell value) or on the lognormal branch
(`A_PIX_K * 10**mu`, mu the kernel's own structural recentring, not
zero). `N_COL_NODES = 1` is therefore special-cased to the single node
`A_PIX_K` itself (`build_region`'s own `if n_col_nodes == 1`): the
"reproduces the C22 products bit-identically" identity the brief asks
for is a MECHANISM check -- that averaging over nodes with weight `1/N`
at `N=1`, node value `A_PIX_K`, is bit-identical to the old direct
`A_PIX_K` dimming every one of the three consumers used -- not a claim
that the real (N>1) HGBS/lognormal construction happens to equal
`A_PIX_K`.

Writes, per region, `population/anchors/column-nodes_anchors_hpx512__
<Region>.hdf5`: `HPX_PIX_512`, `A_PIX_K`, `HERSCHEL_COVERED`, `NODE_A_K`
(n_pix, N_COL_NODES); root attrs `GRANULE="hpx512"`, `N_COL_NODES`.
"""

import os

import h5py
import numpy as np
from scipy.special import ndtri

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.population import anchor_tiles, kernel as kernel_module, yso
from sesnaimpute.sky.derived import column as column_module

_STEM = "column-nodes_anchors_hpx512"

#: The brief proposes 8 and asks for the smaller of 8/16 that passes the
#: tail-share identity at 1e-3 -- measured on Vela D (coordinator's own
#: convergence check, this unit's report): 8 and 16 nodes gave the same
#: tail shares to well inside 1e-3, so 8 is kept (the smaller node count
#: that passes).
N_COL_NODES = 8


def _equal_weight_quantiles(n):
    """The `n` equal-weight quantile points `(i + 0.5) / n`, `i = 0..n-1`
    -- the probability each of `n` nodes stands for is split evenly, and
    a node's own value is the distribution's value at the middle of its
    own share (module docstring)."""
    return (np.arange(n, dtype=np.float64) + 0.5) / float(n)


def _herschel_pixel_cell_values(config, pix_sorted, n_pix):
    """`[array_of_cell_A_K, ...]`, one entry per pixel of `pix_sorted`
    (ascending), the pixel's own matched HGBS cells (empty where none
    matched) -- grouped from `population.yso._herschel_cell_matches`'s
    flat `(idx, a_k)` pair (the SAME cell reader the law integral
    reads), never a second map read."""
    idx, a_k = yso._herschel_cell_matches(config, pix_sorted)
    out = [[] for _ in range(n_pix)]
    if idx.size:
        order = np.argsort(idx, kind="stable")
        idx_sorted, a_k_sorted = idx[order], a_k[order]
        starts = np.searchsorted(idx_sorted, np.arange(n_pix), side="left")
        ends = np.searchsorted(idx_sorted, np.arange(n_pix), side="right")
        for p in range(n_pix):
            if ends[p] > starts[p]:
                out[p] = a_k_sorted[starts[p]:ends[p]]
    return out


def column_nodes_for_pixel(a_pix_k, cell_values, mu, sigma, n_col_nodes):
    """`(n_col_nodes,)`: one pixel's own column nodes (module docstring).
    `cell_values` is that pixel's own matched HGBS cell set (empty ->
    the lognormal branch); `mu`/`sigma` the Planck kernel's own mixture
    mean/width in log10 T at `a_pix_k`, used only on the lognormal
    branch. `n_col_nodes == 1` is the brief's own mechanism-check
    special case (module docstring's "Identity"): the single node is
    `a_pix_k` itself, neither branch evaluated."""
    if n_col_nodes == 1:
        return np.array([a_pix_k], dtype=np.float64)
    qs = _equal_weight_quantiles(n_col_nodes)
    if len(cell_values):
        return np.quantile(np.asarray(cell_values, dtype=np.float64), qs)
    z = ndtri(qs)
    return a_pix_k * 10.0 ** (mu + sigma * z)


def _sightline_a_k_lookup(config, region):
    """`(sightline_pix_sorted, a_k_sorted)`: every nside-256 sightline of
    `region`'s own adopted extinction column (`sky/derived/adopted/
    extinction_adopted_sightline.hdf5`, the SAME product `bmstp.atlas.
    _pixel_column` reads) -- the fallback `A_PIX_K`-equivalent for an
    admitted pixel with no SESNA source of its own."""
    path = config_module.product_path(config, "sky/derived", "adopted", "extinction", "sightline")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "population.column_nodes: no extinction sightline column at %s -- run the "
            "'sesnaimpute.sky.derived.column' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        mine = np.asarray(f["REGION_CODE"][:]) == column_module._region_codes(config, [region])[region]
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)[mine]
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)[mine]
    order = np.argsort(sl_pix)
    return sl_pix[order], a_k[order]


def _lookup_sightline(sl_pix_sorted, a_k_sorted, parent256):
    loc = np.searchsorted(sl_pix_sorted, parent256)
    capped = np.minimum(loc, max(sl_pix_sorted.size - 1, 0))
    found = (sl_pix_sorted.size > 0) & (sl_pix_sorted[capped] == parent256)
    if not np.all(found):
        raise ValueError(
            "population.column_nodes: %d admitted pixel(s) have no sightline column in the "
            "adopted extinction product" % int(np.sum(~found)))
    return a_k_sorted[capped]


def build_region(config, region, n_col_nodes=N_COL_NODES):
    """Every pixel `bmstp.atlas` admits for `region` (`catalog.limits.
    completeness`'s own `HPX_PIX_512`, a superset of the anchor pixel
    set), its own `A_PIX_K`, HGBS coverage flag, and `n_col_nodes` column
    nodes."""
    reader = limits_module.completeness(config, region)
    pixels = np.unique(np.asarray(reader.hpx_pix_512, dtype=np.int64))
    n_pix = pixels.size
    parent256 = pixels // 4

    # the SESNA-source mean, for the pixels that carry a source of their
    # own (anchor_tiles' own construction, module docstring).
    anchor_pix, _anchor_parent256, anchor_a_pix = anchor_tiles._region_pixel_columns(config, region)
    loc = np.searchsorted(anchor_pix, pixels)
    capped = np.minimum(loc, max(anchor_pix.size - 1, 0))
    has_source = (anchor_pix.size > 0) & (anchor_pix[capped] == pixels)
    a_pix_k = np.empty(n_pix, dtype=np.float64)
    a_pix_k[has_source] = anchor_a_pix[capped[has_source]]

    # the sourceless remainder: the parent nside-256 sightline's own
    # adopted extinction column (module docstring).
    if np.any(~has_source):
        sl_pix_sorted, sl_a_k_sorted = _sightline_a_k_lookup(config, region)
        a_pix_k[~has_source] = _lookup_sightline(sl_pix_sorted, sl_a_k_sorted, parent256[~has_source])

    cell_values = _herschel_pixel_cell_values(config, np.sort(pixels), n_pix)
    # `pixels` is already ascending (np.unique's own return), so
    # `cell_values`'s own order (built on `np.sort(pixels)`) matches it.
    herschel_covered = np.array([len(c) > 0 for c in cell_values], dtype=bool)

    mu = np.zeros(n_pix, dtype=np.float64)
    sigma = np.zeros(n_pix, dtype=np.float64)
    if n_col_nodes > 1 and np.any(~herschel_covered):
        kern = kernel_module.Kernel.read(config)
        m, s = kern.params(a_pix_k[~herschel_covered], np.zeros(int(np.sum(~herschel_covered))),
                            np.full(int(np.sum(~herschel_covered)), "planck"))
        mu[~herschel_covered] = m
        sigma[~herschel_covered] = s

    node_a_k = np.empty((n_pix, n_col_nodes), dtype=np.float64)
    for i in range(n_pix):
        node_a_k[i] = column_nodes_for_pixel(
            a_pix_k[i], cell_values[i], mu[i], sigma[i], n_col_nodes)

    return dict(region=region, pixels=pixels, a_pix_k=a_pix_k,
                herschel_covered=herschel_covered, node_a_k=node_a_k,
                n_col_nodes=n_col_nodes)


def write_region(config, region, result):
    path = config_module.product_path(config, "population", "anchors", "column-nodes", "hpx512", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.attrs["N_COL_NODES"] = result["n_col_nodes"]
        for name, data in (
            ("HPX_PIX_512", result["pixels"].astype(np.int64)),
            ("A_PIX_K", result["a_pix_k"].astype(np.float32)),
            ("HERSCHEL_COVERED", result["herschel_covered"].astype(np.bool_)),
            ("NODE_A_K", result["node_a_k"].astype(np.float32)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])
    return path


_READ_CACHE = {}


def read_region(config, region):
    """`(pixels, a_pix_k, herschel_covered, node_a_k)` of `region`'s own
    column-nodes product, cached per `(data_root, region)` (`population.
    anchor_tiles._region_pixel_columns`'s own per-region reader
    precedent): `population.anchor_tiles`, `population.star_population`
    and `bmstp.atlas` each read this once per region, not once per
    pixel/tile/piece."""
    key = (config.data_root, region)
    cached = _READ_CACHE.get(key)
    if cached is not None:
        return cached
    path = config_module.product_path(config, "population", "anchors", "column-nodes", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "population.column_nodes: no column-nodes product for region %r at %s -- run "
            "the 'sesnaimpute.population.column_nodes' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        pixels = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        a_pix_k = np.asarray(f["A_PIX_K"][:], dtype=np.float64)
        herschel_covered = np.asarray(f["HERSCHEL_COVERED"][:], dtype=bool)
        node_a_k = np.asarray(f["NODE_A_K"][:], dtype=np.float64)
    _READ_CACHE.clear()
    out = (pixels, a_pix_k, herschel_covered, node_a_k)
    _READ_CACHE[key] = out
    return out


def nodes_for_pixels(config, region, pix512):
    """`(n_col_nodes,)`-wide `NODE_A_K` rows aligned to `pix512` (any
    order, any subset of the region's own column-nodes product): the one
    join every consumer (`anchor_tiles`, `star_population`, `bmstp.
    atlas`) uses, so a pixel absent from the product raises rather than
    silently substituting a value."""
    pixels, _a_pix_k, _herschel_covered, node_a_k = read_region(config, region)
    order = np.argsort(pixels)
    pix_sorted = pixels[order]
    loc = np.searchsorted(pix_sorted, pix512)
    capped = np.minimum(loc, max(pix_sorted.size - 1, 0))
    found = (pix_sorted.size > 0) & (pix_sorted[capped] == pix512)
    if not np.all(found):
        raise ValueError(
            "population.column_nodes: %d pixel(s) absent from the column-nodes product -- "
            "rerun 'sesnaimpute.population.column_nodes' for this region" % int(np.sum(~found)))
    return node_a_k[order][capped]


def build(config, regions=None):
    """Writes the column-nodes product for `regions` (default: all
    thirty), one file per region."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        result = build_region(config, region)
        path = write_region(config, region, result)
        n_cov = int(result["herschel_covered"].sum())
        print("population.column_nodes: %s: %d pixels, N_COL_NODES=%d, "
              "herschel_covered=%d/%d, A_PIX_K 16/50/84=%.3f/%.3f/%.3f -> %s"
              % (region, result["pixels"].size, result["n_col_nodes"], n_cov, result["pixels"].size,
                 *np.percentile(result["a_pix_k"], (16, 50, 84)), path))


if __name__ == "__main__":
    run(build)
