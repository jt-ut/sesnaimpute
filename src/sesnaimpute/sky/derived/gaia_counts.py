"""Gaia DR3 anchor counts, reshaped onto the region's exact nside-512
pixel set (SPEC_PRIORS.md section 2.1: "Half-magnitude histograms per
nside-512 pixel" -- read as the eleven 1-mag `G` bins of
`04_star_family.md` section B's `read_anchor_counts` row and the old
`fetch_external.gaia_source_counts.build.healpix512` module).

Derive never fetches: this reads
`sky/download/gaia_counts/counts_gaia_hpx512__<Region>.csv` (already a
per-pixel, per-bin count table -- the TAP service's own GROUP BY, see
that module's docstring) and the granule map, and does exactly one
computation -- restricting the download's bounding-box superset of
pixels to the region's own occupied nside-512 set, which is a join, not
a science computation. The result is deposited into a dense
`(n_pix, n_bins)` array, zero where the download carried no row for a
(pixel, bin) pair -- 2MASS-style but with the Gaia magnitude grid.
"""

import os

import h5py
import numpy as np

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access

MAG_EDGES = np.arange(10.0, 22.0, 1.0)  # the old module's G_MAG_EDGES: eleven 1-mag bins, 10-21


def _download_path(config, region):
    return f"{config.data_root}/sky/download/gaia_counts/counts_gaia_hpx512__{region}.csv"


def _region_pixels(config, region):
    """The region's own occupied nside-512 pixels, sorted, from the granule map."""
    return np.unique(access.region_slice(config, region)["hpx_pix_512"])


def _counts_matrix(pixels, csv_path):
    """`(n_pix, n_bins)` int64: the download's per-pixel-per-bin rows
    restricted to `pixels` (the region's own set) and deposited by bin
    index `G_BIN_LO - 10`."""
    n_pix, n_bins = pixels.size, MAG_EDGES.size - 1
    n = np.zeros((n_pix, n_bins), dtype=np.int64)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"gaia_counts derive: no download CSV at {csv_path} -- run the "
            "'sesnaimpute.sky.download.gaia_counts.build' RUNBOOK line first")
    table = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
    if table.size == 0:
        return n
    row_pix = table["hpx_pix_512"].astype(np.int64)
    bin_idx = table["g_bin_lo"].astype(np.int64) - 10
    row_n = table["n"].astype(np.int64)
    loc = np.searchsorted(pixels, row_pix)
    in_set = (loc < n_pix) & (pixels[np.minimum(loc, n_pix - 1)] == row_pix)
    in_bin = (bin_idx >= 0) & (bin_idx < n_bins)
    keep = in_set & in_bin
    np.add.at(n, (loc[keep], bin_idx[keep]), row_n[keep])
    return n


def _write_region(config, region, pixels, n):
    out_path = config_module.product_path(config, "sky/derived", "gaia", "counts", "hpx512", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.create_dataset("HPX_PIX_512", data=pixels.astype(np.int64))
        f.create_dataset("MAG_EDGES", data=MAG_EDGES.astype(np.float64))
        f.create_dataset("N", data=n.astype(np.int64))


def build(config, regions=None):
    """Writes `sky/derived/gaia/counts_gaia_hpx512__<Region>.hdf5` for
    each requested region (default: all thirty)."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    written = []
    with progress_module.Stage("sky.derived.gaia_counts") as st:
        n_regions = len(regions)
        total_counts = 0
        for i, region in enumerate(regions):
            pixels = _region_pixels(config, region)
            n = _counts_matrix(pixels, _download_path(config, region))
            _write_region(config, region, pixels, n)
            written.append(region)
            total_counts += int(n.sum())
            print(f"gaia_counts derive: {region}: {pixels.size} pixels, {int(n.sum())} total counts")
            st.tick(i + 1, n_regions, "regions")
        st.done(None, regions=n_regions, total_counts=total_counts)
    return dict(regions_written=written)


if __name__ == "__main__":
    build_module.run(build)
