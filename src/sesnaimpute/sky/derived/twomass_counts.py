"""2MASS PSC anchor counts per nside-512 HEALPix pixel (SPEC_BMSTP_DRAFT.md
section 5.1's anchor row: 2MASS PSC from 9.0 in half-magnitude bins,
read against the STAR weight's `N^obs` row). The grid runs half-magnitude
bins from 9.0 to 14.3 -- ten whole bins (9.0-14.0) plus one partial
closing bin (14.0-14.3) that carries the anchor exactly to the cut,
rather than a full magnitude past it.

Derive never fetches, but this is the one computing half of the 2MASS
pair: `sky.download.twomass_counts.build` could not get IRSA to bin by
nside-512 pixel server-side (no HEALPix ADQL function there -- see that
module's docstring), so it wrote verbatim rows of 2MASS's own real Ks
point sources (`glon`, `glat`, `k_m`, `cc_flg`, `rd_flg`, `ph_qual`,
`j_m`, `h_m`), and this module does the binning the Gaia sibling's TAP
query did for itself: `healpy.ang2pix(512, glon, glat, nest=True,
lonlat=True)` -- no frame conversion, since `fp_psc`'s `glon`/`glat` are
already this project's own Galactic nside-512 grid -- then a half-
magnitude histogram per pixel, restricted to the region's own occupied
pixel set from the granule map. Alongside the histogram it reports, per
region, what fraction of the counted sources carry a `cc_flg` confusion
code (`CONFUSION_CC_CODES`) -- report only, not written to the product.
"""

import os

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import batches as batches_module
from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.granules import access

_STEM = "counts_twomass_hpx512"

NSIDE = 512

#: Row-batch memory budget for the per-source download CSV (rule 10b):
#: unlike the Gaia sibling, this download is one row per 2MASS detection,
#: not one row per (pixel, bin) -- a dense region's file can carry
#: millions of rows (measured 4.2 GB resident reading it whole at Cygnus
#: X). The histogram accumulation (`np.add.at`) is exact and order-free,
#: so summing it one batch at a time changes nothing it computes.
ROW_BATCH_BUDGET_BYTES = 512 << 20
#: Ten half-mag bins (9.0-14.0) plus the closing partial bin to the spec
#: cut (SPEC_PRIORS.md section 2.1, "2MASS PSC ... Ks < 14.3").
MAG_EDGES = np.append(np.arange(9.0, 14.0001, 0.5), 14.3)
#: `cc_flg`'s five non-zero codes -- the download module's own constants
#: block and docstring quote the Explanatory Supplement establishing
#: that every one marks a real, retained source, not an artifact; this
#: view reports what fraction of the counted (real-detection) sources
#: carry each, so the confusion the old clean-photometry cut used to
#: discard is on record.
CONFUSION_CC_CODES = ("p", "c", "d", "s", "b")

# ---------------------------------------------------------------------------
# C30 item 1 (briefs/round4_2026-10-20/C30_population_anchor.md): the
# colour-cell counts beside the Ks-only histogram above -- C24e's own
# construction (c18_diag/c24e_colour_cells.py), carried into the product so
# `population.anchor_tiles`/`anchor_weights` can fit a per-population
# anchor weight without a diagnostic re-derivation of these counts.
# ---------------------------------------------------------------------------

#: The anchor's own five Ks bins at its 2MASS-only range, 12.0-14.3
#: (`anchor_tiles.KS_CUT_MAG`) -- the colour split only ever applies here,
#: never to the deep (UKIDSS/VHS) bins past it, which stay Ks-only.
CELL_KS_EDGES = np.array([12.0, 12.5, 13.0, 13.5, 14.0, 14.3])
#: Interior J-Ks edges only (C24e): bin 0 is "J-Ks < CELL_JK_EDGES[0]",
#: bin -1 (the reddest) is "J-Ks > CELL_JK_EDGES[-1]" -- a row with no
#: usable J (`ph_qual`'s J character 'U'/'X', or a non-finite `j_m`)
#: counts in the reddest bin: a J non-detection is truly redder than the
#: last edge, never assumed blue.
CELL_JK_EDGES = np.array([0.6, 1.0, 1.5])
#: C24e's own J-completeness margin: a (pixel, Ks bin, J-Ks bin) cell
#: counts only where this pixel's own 2MASS J 50%-completeness magnitude
#: clears the cell's own faintest possible J (Ks_hi + (J-Ks)_hi) by this
#: many magnitudes -- found every cell complete in the five regions
#: measured 2026-10-21, kept anyway: it is data, not an assumption.
CELL_J_COMPLETE_MARGIN_MAG = 0.3


def _read_j50(config, region, pixels):
    """2MASS J 50%-completeness magnitude per `pixels` (nside-512),
    from this region's own `catalog.depths` per-nside-128-pixel product
    (`F50_2MASS_MJY[:, 0]`, `catalog.depths.TWOMASS_KEYS` J-first order),
    read at each pixel's own NESTED parent (`pix512 >> 4`). 2MASS is
    all-sky, so every parent is expected present; a pixel missing from
    that product (should not happen) takes the region's own median
    instead, named in the printed report rather than silently assumed.
    """
    path = config_module.product_path(config, "catalog", "sesna", "depths", "hpx128", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"twomass_counts derive: no per-pixel depths product at {path} -- run the "
            "'sesnaimpute.catalog.depths' RUNBOOK line first")
    with h5py.File(path, "r") as f:
        p128 = np.asarray(f["HPX_PIX_128"][:], dtype=np.int64)
        f50_j_mjy = np.asarray(f["F50_2MASS_MJY"][:, 0], dtype=np.float64)
    zp_j_jy = definitions.BANDS_BY_KEY["J"].vega_zero_point_jy
    j50_128 = -2.5 * np.log10(f50_j_mjy / (zp_j_jy * 1.0e3))
    parent = pixels.astype(np.int64) >> 4
    loc = np.searchsorted(p128, parent)
    ok = (loc < p128.size) & (p128[np.minimum(loc, p128.size - 1)] == parent)
    j50 = np.where(ok, j50_128[np.minimum(loc, p128.size - 1)], np.nanmedian(j50_128))
    n_missing = int((~ok).sum())
    if n_missing:
        print(f"twomass_counts derive: {region}: {n_missing} pixel(s) missing from the "
              "per-pixel depths product -- took the region's own median J50 instead")
    return j50


def _colour_cell_matrix(pixels, csv_path, j50_pix):
    """`(N_CELL, CELL_COMPLETE)`, each `(n_pix, 5, 4)`: the same downloaded
    2MASS rows as `_counts_matrix`, restricted to `CELL_KS_EDGES` (the
    anchor's own 12.0-14.3 range) and split by J-Ks colour onto
    `CELL_JK_EDGES`. `CELL_COMPLETE` is C24e's own J-completeness test,
    per cell: `Ks_hi + (J-Ks)_hi <= J50 - CELL_J_COMPLETE_MARGIN_MAG`,
    the reddest (open-ended) bin standing in its own last finite edge
    for "(J-Ks)_hi" (C24e: "the red bin: Ks_hi + 1.5 <= J50 - 0.3, so a
    J non-detection is truly redder than 1.5"). Read in the same row-
    batched passes as `_counts_matrix` (rule 10b); a second pass over the
    same CSV, since the two matrices bin on different magnitude grids
    and keeping them independent keeps each one simple and exactly
    re-derivable from the download alone.
    """
    n_pix = pixels.size
    n_k = CELL_KS_EDGES.size - 1
    n_c = CELL_JK_EDGES.size + 1
    n_cell = np.zeros((n_pix, n_k, n_c), dtype=np.int64)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"twomass_counts derive: no download CSV at {csv_path} -- run the "
            "'sesnaimpute.sky.download.twomass_counts.build' RUNBOOK line first")
    chunksize = _csv_row_batch_size(csv_path)
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if chunk.empty:
            continue
        glon = chunk["glon"].to_numpy(dtype=np.float64)
        glat = chunk["glat"].to_numpy(dtype=np.float64)
        k_m = chunk["k_m"].to_numpy(dtype=np.float64)
        j_m = chunk["j_m"].to_numpy(dtype=np.float64)
        ph_qual_j = chunk["ph_qual"].astype(str).str[0]
        row_pix = hp.ang2pix(NSIDE, glon, glat, nest=True, lonlat=True).astype(np.int64)
        loc = np.searchsorted(pixels, row_pix)
        in_set = (loc < n_pix) & (pixels[np.minimum(loc, n_pix - 1)] == row_pix)
        kb = np.searchsorted(CELL_KS_EDGES, k_m, side="right") - 1
        in_kb = (kb >= 0) & (kb < n_k)
        j_bad = ph_qual_j.isin(["U", "X"]).to_numpy() | ~np.isfinite(j_m)
        jk = np.where(j_bad, np.inf, j_m - k_m)
        cb = np.minimum(np.searchsorted(CELL_JK_EDGES, jk, side="right"), n_c - 1)
        keep = in_set & in_kb
        np.add.at(n_cell, (loc[keep], kb[keep], cb[keep]), 1)
    ks_hi = CELL_KS_EDGES[1:]
    jk_hi = np.append(CELL_JK_EDGES, CELL_JK_EDGES[-1])
    complete = (ks_hi[None, :, None] + jk_hi[None, None, :]
                <= (j50_pix[:, None, None] - CELL_J_COMPLETE_MARGIN_MAG))
    return n_cell, complete


def _download_path(config, region):
    return f"{config.data_root}/sky/download/twomass_counts/counts_twomass_hpx512__{region}.csv"


def _region_pixels(config, region):
    """The region's own occupied nside-512 pixels, sorted, from the granule map."""
    return np.unique(access.region_slice(config, region)["hpx_pix_512"])


def _csv_row_batch_size(csv_path, budget_bytes=ROW_BATCH_BUDGET_BYTES):
    """Rows per batch (rule 10b) for `csv_path`: its own first data row's
    byte width sets `batches.batches`' per-row footprint, so a dense
    region's file (millions of detection rows) is never read whole."""
    with open(csv_path, "r") as f:
        f.readline()  # header
        first = f.readline()
    if not first:
        return 1
    with open(csv_path, "r") as f:
        n_rows = sum(1 for _ in f) - 1
    row_bytes = len(first.encode("utf-8"))
    _, stop = next(batches_module.batches(max(n_rows, 1), row_bytes, budget_bytes=budget_bytes))
    return stop


def _counts_matrix(pixels, csv_path):
    """`(n_pix, n_bins)` int64: each downloaded row placed at its own
    Galactic nside-512 pixel and Ks half-mag bin, restricted to `pixels`
    (the region's own occupied set). Read in row batches (rule 10b): the
    histogram this accumulates into is exact and order-independent, so
    no batching changes any count. Alongside it, the counted rows' own
    `cc_flg` Ks character is tallied per `CONFUSION_CC_CODES` -- report
    only, read by nothing, so it cannot change the histogram above."""
    n_pix, n_bins = pixels.size, MAG_EDGES.size - 1
    n = np.zeros((n_pix, n_bins), dtype=np.int64)
    code_counts = {c: 0 for c in CONFUSION_CC_CODES}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"twomass_counts derive: no download CSV at {csv_path} -- run the "
            "'sesnaimpute.sky.download.twomass_counts.build' RUNBOOK line first")
    chunksize = _csv_row_batch_size(csv_path)
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if chunk.empty:
            continue
        glon = chunk["glon"].to_numpy(dtype=np.float64)
        glat = chunk["glat"].to_numpy(dtype=np.float64)
        k_m = chunk["k_m"].to_numpy(dtype=np.float64)
        row_pix = hp.ang2pix(NSIDE, glon, glat, nest=True, lonlat=True).astype(np.int64)
        bin_idx = np.searchsorted(MAG_EDGES, k_m, side="right") - 1
        loc = np.searchsorted(pixels, row_pix)
        in_set = (loc < n_pix) & (pixels[np.minimum(loc, n_pix - 1)] == row_pix)
        in_bin = (bin_idx >= 0) & (bin_idx < n_bins)
        keep = in_set & in_bin
        np.add.at(n, (loc[keep], bin_idx[keep]), 1)
        cc_ks_kept = chunk["cc_flg"].str[2].to_numpy()[keep]
        for code in CONFUSION_CC_CODES:
            code_counts[code] += int((cc_ks_kept == code).sum())
    return n, code_counts


def _write_region(config, region, pixels, n, n_cell, cell_complete):
    out_path = config_module.product_path(config, "sky/derived", "twomass", "counts", "hpx512", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        for name, data in (
            ("HPX_PIX_512", pixels.astype(np.int64)),
            ("MAG_EDGES", MAG_EDGES.astype(np.float64)),
            ("N", n.astype(np.int64)),
            ("CELL_KS_EDGES", CELL_KS_EDGES.astype(np.float64)),
            ("CELL_JK_EDGES", CELL_JK_EDGES.astype(np.float64)),
            ("N_CELL", n_cell.astype(np.int64)),
            ("CELL_COMPLETE", cell_complete.astype(np.bool_)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])


def build(config, regions=None):
    """Writes `sky/derived/twomass/counts_twomass_hpx512__<Region>.hdf5`
    for each requested region (default: all thirty). Reports, per
    region, the fraction of the counted (real-Ks-detection) sources
    carrying a `cc_flg` confusion code -- the population the old clean-
    photometry cut used to exclude (SPEC_BMSTP_DRAFT.md 5.1)."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    written = []
    with progress_module.Stage("sky.derived.twomass_counts") as st:
        n_regions = len(regions)
        total_counts = 0
        for i, region in enumerate(regions):
            pixels = _region_pixels(config, region)
            n, code_counts = _counts_matrix(pixels, _download_path(config, region))
            j50_pix = _read_j50(config, region, pixels)
            n_cell, cell_complete = _colour_cell_matrix(pixels, _download_path(config, region), j50_pix)
            _write_region(config, region, pixels, n, n_cell, cell_complete)
            written.append(region)
            n_total = int(n.sum())
            total_counts += n_total
            n_confused = sum(code_counts.values())
            frac = n_confused / n_total if n_total else 0.0
            by_code = ", ".join(f"{c}={code_counts[c]/n_total:.4f}" if n_total else f"{c}=0.0000"
                                 for c in CONFUSION_CC_CODES)
            n_cell_total = int(n_cell.sum())
            incomplete = int(n_cell.size - cell_complete.sum())
            print(f"twomass_counts derive: {region}: {pixels.size} pixels, {n_total} total counts, "
                  f"{frac:.4f} confusion-flagged ({by_code})")
            print(f"twomass_counts derive: {region}: colour cells (Ks 12.0-14.3): {n_cell_total} rows, "
                  f"{incomplete}/{n_cell.size} (pixel, Ks bin, J-Ks bin) cells J-incomplete")
            st.tick(i + 1, n_regions, "regions")
        st.done(None, regions=n_regions, total_counts=total_counts)
    return dict(regions_written=written)


if __name__ == "__main__":
    build_module.run(build)
