"""The NICEST star-colour extinction column, per source (the extinction
column is formed at the reference map's own resolution, its 3' beam,
not coarser).

Juvela & Montillaud (2016, A&A 585, A38) built `NICEST_AJ_M1_FWHM3.0.fits`
from 2MASS star colours reddened by the whole sightline (NICEST, model
M1, 3.0' FWHM), HEALPix nested NSIDE=2048, Galactic `A_J`. `A_K` here is
that `A_J` over the adopted diffuse law's own `A_J/A_K`
(`population.selection.extinction_k` at `LAW_DIFFUSE`) -- the same
K-currency every other adopted-column product uses, so a per-nside-1024-
cell factor against the emission column needs no further
conversion.

Per source: `A_K` is the map value at the source's own nside-2048
nested pixel (its Galactic position, `catalog/sesna/sources_sesna_source`'s
`GAL_L_DEG`/`GAL_B_DEG`), and `HPX_PIX_1024` is the source's nside-1024
nested pixel -- the 3.4' beam cell the extinction column's factor is formed on.
"""

import os

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.population import selection as selection_module

NSIDE_MAP = 2048
NSIDE_CELL = 1024


def _map_path(config):
    return f"{config.data_root}/sky/download/juvela2016/NICEST_AJ_M1_FWHM3.0.fits"


def _load_map(config):
    """The NICEST `A_J` map, nested, converted in place to `A_K` with
    the adopted diffuse law's own `A_J/A_K` (module docstring)."""
    path = _map_path(config)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "sky.derived.juvela_extinction: no NICEST map at %r -- run the "
            "'sesnaimpute.sky.download.juvela2016.build' RUNBOOK line first" % path)
    a_j = hp.read_map(path, nest=True, dtype=np.float64)
    k = selection_module.extinction_k(config, selection_module.LAW_DIFFUSE)
    band_keys = selection_module.BAND_KEYS
    aj_over_ak = k[band_keys.index("J")] / k[band_keys.index("Ks")]
    return a_j / aj_over_ak


def _region_positions(config, region):
    """`(l_deg, b_deg)`, one row per catalogued source of `region`, in
    catalogue-row order (`GAL_L_DEG`/`GAL_B_DEG` from the region's own
    curated source file, already in that order -- `granules.access`'s
    `CATALOG_ROW` is a plain `range(n_sources)` for a per-region file)."""
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "sky.derived.juvela_extinction: no curated source catalogue for region %r "
            "at %r -- run the 'sesnaimpute.catalog.curated' RUNBOOK line first" % (region, path))
    cols = access.per_source(config, region, path, ["GAL_L_DEG", "GAL_B_DEG"])
    return cols["GAL_L_DEG"], cols["GAL_B_DEG"]


def _write_source(config, region, a_k, hpx_pix_1024):
    out_path = config_module.product_path(config, "sky/derived", "juvela", "extinction", "source", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        build_module.write_dataset(f, "A_K", a_k.astype(np.float32), *REGISTRY[("extinction_juvela_source", "A_K")])
        build_module.write_dataset(f, "HPX_PIX_1024", hpx_pix_1024.astype(np.int64),
                                    *REGISTRY[("extinction_juvela_source", "HPX_PIX_1024")])


def build(config, regions=None):
    """Writes `sky/derived/juvela/extinction_juvela_source__<Region>.hdf5`
    for each requested region (default: all thirty)."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    a_k_map = _load_map(config)  # full nside-2048 A_K map, ~400 MB float64

    with progress_module.Stage("sky.derived.juvela_extinction") as st:
        n_regions = len(regions)
        for i, region in enumerate(regions):
            l_deg, b_deg = _region_positions(config, region)
            pix2048 = hp.ang2pix(NSIDE_MAP, l_deg, b_deg, nest=True, lonlat=True)
            pix1024 = hp.ang2pix(NSIDE_CELL, l_deg, b_deg, nest=True, lonlat=True)
            a_k_source = a_k_map[pix2048]
            _write_source(config, region, a_k_source, pix1024)
            st.tick(i + 1, n_regions, "regions")
        st.done(None, regions=n_regions)


if __name__ == "__main__":
    run(build)
