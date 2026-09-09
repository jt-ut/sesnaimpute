"""UKIDSS GPS clean-photometry counts per nside-512 HEALPix pixel
(SPEC_BMSTP_DRAFT.md 5.1's STAR anchor: this is the 2MASS anchor
counts product's own shape, `sky.derived.twomass_counts`, carried past
its 14.3 cut to K = 17.0 -- W38 reads this to extend the STAR weight's
faint 2MASS bins with GPS's deeper reach where GPS covers the region
(Lucas et al. 2008, MNRAS 391, 136: K approx 18 5-sigma, |b| lesssim
5 deg, l 15-107 deg and 141-230 deg)).

Derive never fetches: `sky.download.ukidss_gps.build` already applied
the clean-photometry quality cut (WFCAM error bits, merged class, W34
section 2b) and the null-sentinel/magnitude-range cuts at query time, so
this module's only computation is the nside-512, half-magnitude
histogram, exactly `twomass_counts`'s own binning:
`healpy.ang2pix(512, l, b, nest=True, lonlat=True)` (no frame
conversion -- GPS's `l`/`b` are already Galactic) then a half-mag
histogram per pixel, restricted to the region's own occupied pixel set
from the granule map -- the same set `twomass_counts` restricts to,
since both read `access.region_slice` for it, W37's pixel-identity
check.

Photometric system. WFCAM K (UKIDSS/GPS) is calibrated to the UKIRT
system (Hewett, Warren, Leggett & Hodgkin 2006, MNRAS 367, 454), and its
transformation to 2MASS Ks (Hodgkin, Irwin, Hewett & Warren 2009, MNRAS
394, 675) carries a J-or-H colour term of order 0.01-0.03 mag (Hewett
et al. 2006 table 5's synthetic-photometry coefficients on a typical
GPS-field J-K); this download carries K alone, not J or H, so the term
cannot be evaluated per source. Left unapplied: at most 0.03 mag against
this histogram's 0.5-mag bins is a few percent of one bin width, not a
correction the counts feel.
"""

import os

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import batches as batches_module
from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access

NSIDE = 512

#: Row-batch memory budget (rule 10b): like `twomass_counts`'s download,
#: this is one row per GPS detection, and a dense region's file can carry
#: millions of rows (Cygnus X measured ~9M, W37 report) -- never read
#: whole. The histogram accumulation (`np.add.at`) is exact and
#: order-free, so batching changes nothing it computes.
ROW_BATCH_BUDGET_BYTES = 512 << 20
#: Sixteen half-magnitude bins, 9.0-17.0: the download's own K < 17.5
#: cut (`sky.download.ukidss_gps.build.K_CEILING`) gives this top edge a
#: half-bin of headroom, as `twomass_counts.MAG_EDGES` does against its
#: 14.3 cut.
MAG_EDGES = np.arange(9.0, 17.0001, 0.5)


def _download_path(config, region):
    return f"{config.data_root}/sky/download/ukidss_gps/sources_ukidss_hpx512__{region}.csv"


def _region_pixels(config, region):
    """The region's own occupied nside-512 pixels, sorted, from the
    granule map -- the same call `twomass_counts` makes, so the two
    products' `HPX_PIX_512` are identical by construction (W37's
    pixel-identity check)."""
    return np.unique(access.region_slice(config, region)["hpx_pix_512"])


def _csv_row_batch_size(csv_path, budget_bytes=ROW_BATCH_BUDGET_BYTES):
    """Rows per batch (rule 10b) for `csv_path`: its own first data row's
    byte width sets `batches.batches`' per-row footprint."""
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
    """`(n_pix, n_bins)` int64: each downloaded clean-photometry row
    placed at its own Galactic nside-512 pixel and K half-mag bin,
    restricted to `pixels` (the region's own occupied set). Read in row
    batches (rule 10b)."""
    n_pix, n_bins = pixels.size, MAG_EDGES.size - 1
    n = np.zeros((n_pix, n_bins), dtype=np.int64)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"ukidss_counts derive: no download CSV at {csv_path} -- run the "
            "'sesnaimpute.sky.download.ukidss_gps.build' RUNBOOK line first")
    chunksize = _csv_row_batch_size(csv_path)
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if chunk.empty:
            continue
        l_deg = chunk["l"].to_numpy(dtype=np.float64)
        b_deg = chunk["b"].to_numpy(dtype=np.float64)
        k_mag = chunk["k_1AperMag3"].to_numpy(dtype=np.float64)
        row_pix = hp.ang2pix(NSIDE, l_deg, b_deg, nest=True, lonlat=True).astype(np.int64)
        bin_idx = np.searchsorted(MAG_EDGES, k_mag, side="right") - 1
        loc = np.searchsorted(pixels, row_pix)
        in_set = (loc < n_pix) & (pixels[np.minimum(loc, n_pix - 1)] == row_pix)
        in_bin = (bin_idx >= 0) & (bin_idx < n_bins)
        keep = in_set & in_bin
        np.add.at(n, (loc[keep], bin_idx[keep]), 1)
    return n


def _write_region(config, region, pixels, n):
    out_path = config_module.product_path(config, "sky/derived", "ukidss", "counts", "hpx512", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.attrs["N_SOURCES"] = int(n.sum())
        f.create_dataset("HPX_PIX_512", data=pixels.astype(np.int64))
        f.create_dataset("MAG_EDGES", data=MAG_EDGES.astype(np.float64))
        f.create_dataset("N", data=n.astype(np.int64))


def build(config, regions=None):
    """Writes `sky/derived/ukidss/counts_ukidss_hpx512__<Region>.hdf5` for
    each requested region (default: all thirty). A region whose download
    is header-only (outside GPS's footprint) writes `N` all zero, with
    `N_SOURCES = 0` -- the coverage record W37's acceptance check reads."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    written = []
    with progress_module.Stage("sky.derived.ukidss_counts") as st:
        n_regions = len(regions)
        total_counts = 0
        for i, region in enumerate(regions):
            pixels = _region_pixels(config, region)
            n = _counts_matrix(pixels, _download_path(config, region))
            _write_region(config, region, pixels, n)
            written.append(region)
            total_counts += int(n.sum())
            print(f"ukidss_counts derive: {region}: {pixels.size} pixels, {int(n.sum())} total counts")
            st.tick(i + 1, n_regions, "regions")
        st.done(None, regions=n_regions, total_counts=total_counts)
    return dict(regions_written=written)


if __name__ == "__main__":
    build_module.run(build)
