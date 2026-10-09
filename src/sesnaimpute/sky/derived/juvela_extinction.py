"""The NICEST star-colour extinction column, per source and per
sightline (the extinction column is formed at the reference
map's own resolution, its 3' beam, not coarser).

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
nested pixel -- the 3.4' beam cell the extinction column's factor is formed on. Per
sightline: `A_K` is the mean of the map's own nside-2048 pixels over
*every* child of the sightline's nside-256 pixel (64 of them, the whole
map footprint under the beam), not only the ones holding a catalogued
source -- the star map sees the whole sightline, not just where SESNA
has sources. Nested HEALPix makes this a reshape: a nside-256 pixel's
2048-side children are the contiguous block `[p*64, p*64+64)` (three
dyadic refinements, 256->512->1024->2048, each x4), so
`map.reshape(-1, 64).mean(axis=1)` gives every nside-256 pixel's mean in
one vectorised pass.
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
from sesnaimpute.sky.derived import profile as profile_module

NSIDE_MAP = 2048
NSIDE_CELL = 1024
CHILDREN_PER_SIGHTLINE = (NSIDE_MAP // 256) ** 2  # 64: nside-256 -> nside-2048


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


def _admitted_sightlines_by_region(config):
    """`{region: sorted unique HPX_PIX_256}` for every admitted sightline
    of every region -- the granule map's own admission
    (`granules.build`'s `association/region_healpix256` group, read
    through `sky.derived.profile`'s own `_admitted_sightlines`): a pixel
    is admitted by coverage, fixed once the granule map is built, not by
    any later column's values, so this never depends on the gas column
    `sky.derived.column` writes."""
    out = {}
    for region in [r.name for r in regions_module.REGIONS]:
        pix, _ = profile_module._admitted_sightlines(config, region)
        out[region] = pix
    return out


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
    for each requested region (default: all thirty), and rebuilds the
    one survey file `sky/derived/juvela/extinction_juvela_sightline.hdf5`
    over the admitted sightlines of all thirty regions (the sightline
    product has no region axis to update in place, so every run of this
    stage writes it whole)."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    else:
        print("sky.derived.juvela_extinction: sightline survey file is not per-region "
              "(rule 5c) -- rebuilt whole every run, cheap once the map is loaded")
    a_k_map = _load_map(config)  # full nside-2048 A_K map, ~400 MB float64
    sightline_a_k_by_pix256 = a_k_map.reshape(-1, CHILDREN_PER_SIGHTLINE).mean(axis=1)

    all_regions = [r.name for r in regions_module.REGIONS]
    admitted_by_region = _admitted_sightlines_by_region(config)
    sl_pix, sl_a_k = [], []
    with progress_module.Stage("sky.derived.juvela_extinction") as st:
        n_regions = len(all_regions)
        for i, region in enumerate(all_regions):
            admitted_pix256 = admitted_by_region[region]
            sl_pix.append(admitted_pix256)
            sl_a_k.append(sightline_a_k_by_pix256[admitted_pix256])

            if region in regions:
                l_deg, b_deg = _region_positions(config, region)
                pix2048 = hp.ang2pix(NSIDE_MAP, l_deg, b_deg, nest=True, lonlat=True)
                pix1024 = hp.ang2pix(NSIDE_CELL, l_deg, b_deg, nest=True, lonlat=True)
                a_k_source = a_k_map[pix2048]
                _write_source(config, region, a_k_source, pix1024)
            st.tick(i + 1, n_regions, "regions")

        pix_all = np.concatenate(sl_pix)
        a_k_all = np.concatenate(sl_a_k)
        out_path = config_module.product_path(config, "sky/derived", "juvela", "extinction", "sightline")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs["GRANULE"] = "sightline"
            build_module.write_dataset(f, "HPX_PIX_256", pix_all.astype(np.int64),
                                        *REGISTRY[("extinction_juvela_sightline", "HPX_PIX_256")])
            build_module.write_dataset(f, "A_K", a_k_all.astype(np.float32),
                                        *REGISTRY[("extinction_juvela_sightline", "A_K")])
        st.done(out_path, regions=n_regions, sightlines=pix_all.size)


if __name__ == "__main__":
    run(build)
