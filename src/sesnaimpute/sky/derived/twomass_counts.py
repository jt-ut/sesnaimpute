"""2MASS PSC anchor counts per nside-512 HEALPix pixel (SPEC_PRIORS.md
section 2.1: "2MASS PSC at Ks < 14.3, clean photometry flag ...
Half-magnitude histograms per nside-512 pixel", read against
`04_star_family.md` section B's `read_anchor_counts` row and the old
`fetch_external.twomass_source_counts.build.healpix512` module's
half-mag, 9.0-15.5 histogram).

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

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access

NSIDE = 512
MAG_EDGES = np.arange(9.0, 15.5001, 0.5)  # the old module's KS_HIST_EDGES: thirteen half-mag bins, 9-15.5


def _download_path(config, region):
    return f"{config.data_root}/sky/download/twomass_counts/counts_twomass_hpx512__{region}.csv"


def _region_pixels(config, region):
    """The region's own occupied nside-512 pixels, sorted, from the granule map."""
    return np.unique(access.region_slice(config, region)["hpx_pix_512"])


def _counts_matrix(pixels, csv_path):
    """`(n_pix, n_bins)` int64: each downloaded clean-photometry row
    placed at its own Galactic nside-512 pixel and Ks half-mag bin,
    restricted to `pixels` (the region's own occupied set)."""
    n_pix, n_bins = pixels.size, MAG_EDGES.size - 1
    n = np.zeros((n_pix, n_bins), dtype=np.int64)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"twomass_counts derive: no download CSV at {csv_path} -- run the "
            "'sesnaimpute.sky.download.twomass_counts.build' RUNBOOK line first")
    table = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
    if table.size == 0:
        return n
    glon = table["glon"].astype(np.float64)
    glat = table["glat"].astype(np.float64)
    k_m = table["k_m"].astype(np.float64)
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
