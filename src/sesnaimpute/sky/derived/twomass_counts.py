"""2MASS PSC anchor counts per nside-512 HEALPix pixel (SPEC_PRIORS.md
section 2.1: "2MASS PSC at Ks < 14.3, clean photometry flag ...
Half-magnitude histograms per nside-512 pixel", read against
`04_star_family.md` section B's `read_anchor_counts` row). The grid runs
half-magnitude bins from 9.0 to the spec cut at 14.3 -- ten whole bins
(9.0-14.0) plus one partial closing bin (14.0-14.3) that carries the
anchor exactly to the cut, rather than a full magnitude past it.

Derive never fetches, but this is the one computing half of the 2MASS
pair: `sky.download.twomass_counts.build` could not get IRSA to bin by
nside-512 pixel server-side (no HEALPix ADQL function there -- see that
module's docstring), so it wrote verbatim clean-photometry rows
(`glon`, `glat`, `k_m`), and this module does the binning the Gaia
sibling's TAP query did for itself: `healpy.ang2pix(512, glon, glat,
nest=True, lonlat=True)` -- no frame conversion, since `fp_psc`'s
`glon`/`glat` are already this project's own Galactic nside-512 grid --
then a half-magnitude histogram per pixel, restricted to the region's
own occupied pixel set from the granule map.
"""

import os

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import batches as batches_module
from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access

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
    """`(n_pix, n_bins)` int64: each downloaded clean-photometry row
    placed at its own Galactic nside-512 pixel and Ks half-mag bin,
    restricted to `pixels` (the region's own occupied set). Read in row
    batches (rule 10b): the histogram this accumulates into is exact and
    order-independent, so no batching changes any count."""
    n_pix, n_bins = pixels.size, MAG_EDGES.size - 1
    n = np.zeros((n_pix, n_bins), dtype=np.int64)
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
    return n


def _write_region(config, region, pixels, n):
    out_path = config_module.product_path(config, "sky/derived", "twomass", "counts", "hpx512", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.create_dataset("HPX_PIX_512", data=pixels.astype(np.int64))
        f.create_dataset("MAG_EDGES", data=MAG_EDGES.astype(np.float64))
        f.create_dataset("N", data=n.astype(np.int64))


def build(config, regions=None):
    """Writes `sky/derived/twomass/counts_twomass_hpx512__<Region>.hdf5`
    for each requested region (default: all thirty)."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    written = []
    for region in regions:
        pixels = _region_pixels(config, region)
        n = _counts_matrix(pixels, _download_path(config, region))
        _write_region(config, region, pixels, n)
        written.append(region)
        print(f"twomass_counts derive: {region}: {pixels.size} pixels, {int(n.sum())} total counts")
    return dict(regions_written=written)


if __name__ == "__main__":
    build_module.run(build)
