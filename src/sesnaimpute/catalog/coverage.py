"""The atlas's coverage, from the catalogue's own IRAC footprint
(SPEC_BMSTP_DRAFT.md sec. 3.3 "the coverage", sec. 8's "weighted by the
pixel's surveyed fraction"): the fraction of each admitted nside-512 pixel
over which SESNA holds a measured IRAC detection.

`sky.derived.coverage`'s Spitzer field-mask union is the wrong denominator
for two regions -- Pipe and Auriga-California -- because their native mask
files are missing mosaics outright, not mis-masked
(`studies/count_discrepancy.md`, cause 1): the sources in the "uncovered"
pixels carry measured IRAC fluxes at the same rate as the rest, and 99.9%
of them fall outside the bounds of every field mask in the file. The atlas
needs only the fraction of each admitted pixel the catalogue was extracted
on, and the catalogue defines that exactly where it holds sources with
measured IRAC photometry -- a `catalog` product (SESNA only, no external
bytes), computed the same way for all thirty regions.

`FRAC` for an admitted pixel is the fraction of its 16 nside-2048 children
(nested HEALPix, `child = pix * 16 + 0..15`) that hold at least one
catalogued source with a measured flux (`ORIGIN_FNU == 1`) in any of the
four IRAC bands (I1-I4); MIPS and 2MASS do not count, so a MIPS-only patch
(2MASS + M1 pass the survey's two-of-eight rule with no IRAC band among
the two) reads FRAC = 0, matching sec. 8's IRAC-defined catalogue. Sources
are read from `catalog.sesna.sources` in batches (rule 10b): nothing holds
every source of a region at once.
"""

import os

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import depth_grid as depth_grid_module

NSIDE_512 = 512
NSIDE_2048 = 2048
#: nside-2048 children per admitted nside-512 pixel: (2048/512)**2.
CHILD_FACTOR = (NSIDE_2048 // NSIDE_512) ** 2
_HPX512_PIXEL_DEG2 = 41252.96 / (12 * NSIDE_512 ** 2)

#: I1-I4 columns of `ORIGIN_FNU`, in `definitions.BANDS`' fixed order
#: (J, H, Ks, I1, I2, I3, I4, M1) -- the catalogue's own IRAC columns.
IRAC_COLS = slice(3, 7)

#: Row-batch memory budget for the per-source read (rule 10b): GAL_L_DEG,
#: GAL_B_DEG (f8 each) plus the four IRAC `ORIGIN_FNU` columns (i1 each).
ROW_BATCH_BUDGET_BYTES = 512 << 20
_ROW_BYTES = 8 + 8 + 4


def _admitted_pixels(config, region):
    """The region's admitted nside-512 pixels, sorted ascending -- the
    granule map's own admission rule (every nside-512 child of the
    region's source-bearing nside-256 pixels), computed by
    `catalog.depth_grid.admitted_pixels_512` and imported rather than
    re-derived, so coverage does not need the depth grid's product (which
    itself now reads coverage's own `FRAC`, sec. 3.3)."""
    return depth_grid_module.admitted_pixels_512(config, region)


def _child_occupancy(config, region, pix):
    """`counts`, int64 `(n_pix, CHILD_FACTOR)`: for each admitted pixel's
    16 nside-2048 children, the number of catalogued sources with a
    measured IRAC flux landing in that child (module docstring). Batched
    over sources (rule 10b), accumulated with `np.add.at`."""
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.coverage: no curated catalogue for %s at %s -- run "
            "the 'catalog.curated' RUNBOOKtp.sh line first" % (region, path))
    counts = np.zeros(pix.size * CHILD_FACTOR, dtype=np.int64)
    with h5py.File(path, "r") as f:
        n_source = f["GAL_L_DEG"].shape[0]
        batch_list = list(batches_module.batches(n_source, _ROW_BYTES, budget_bytes=ROW_BATCH_BUDGET_BYTES))
        for start, stop in batch_list:
            gl = np.asarray(f["GAL_L_DEG"][start:stop], dtype=np.float64)
            gb = np.asarray(f["GAL_B_DEG"][start:stop], dtype=np.float64)
            origin_irac = np.asarray(f["ORIGIN_FNU"][start:stop, IRAC_COLS])
            has_irac = np.any(origin_irac == 1, axis=1)
            if not np.any(has_irac):
                continue
            child2048 = hp.ang2pix(NSIDE_2048, gl[has_irac], gb[has_irac], nest=True, lonlat=True)
            parent512 = child2048 // CHILD_FACTOR
            offset = child2048 % CHILD_FACTOR
            loc = np.minimum(np.searchsorted(pix, parent512), pix.size - 1)
            in_admitted = pix[loc] == parent512
            flat_idx = loc[in_admitted] * CHILD_FACTOR + offset[in_admitted]
            np.add.at(counts, flat_idx, 1)
    return counts.reshape(pix.size, CHILD_FACTOR), batch_list


def _hole_rate(pix, occupied_flat):
    """`(n_holes, n_occupied)`: an empty nside-2048 child whose eight
    neighbours (`healpy.get_all_neighbours`, nested) are all occupied is a
    "hole" -- a covered cell the source density failed to mark. Reported,
    not patched."""
    offsets = np.arange(CHILD_FACTOR, dtype=np.int64)
    child_all = (pix[:, None] * CHILD_FACTOR + offsets[None, :]).ravel()  # already sorted
    n_occupied = int(np.sum(occupied_flat))
    empty = child_all[~occupied_flat]
    if empty.size == 0:
        return 0, n_occupied
    neighbours = hp.get_all_neighbours(NSIDE_2048, empty, nest=True)  # (8, n_empty)
    loc = np.searchsorted(child_all, neighbours)
    loc = np.clip(loc, 0, child_all.size - 1)
    found = (child_all[loc] == neighbours) & (neighbours >= 0)
    neighbour_occupied = found & occupied_flat[loc]
    n_holes = int(np.sum(np.all(neighbour_occupied, axis=0)))
    return n_holes, n_occupied


def build_region(config, region, st):
    """Writes `catalog/sesna/coverage_sesna_hpx512__R.hdf5`: `HPX_PIX`,
    the region's admitted pixels; `FRAC`, each pixel's IRAC-catalogued
    child fraction (module docstring); root attrs `GRANULE = "hpx512"`,
    `AREA_DEG2 = Sigma FRAC * Omega_pix`."""
    pix = _admitted_pixels(config, region)
    counts, batch_list = _child_occupancy(config, region, pix)
    occupied = counts > 0
    frac = occupied.mean(axis=1)
    n_holes, n_occupied = _hole_rate(pix, occupied.ravel())
    hole_rate = n_holes / n_occupied if n_occupied else 0.0
    area_deg2 = float(np.sum(frac) * _HPX512_PIXEL_DEG2)

    print("catalog.coverage %s: sparse-footprint check -- %d holes / %d occupied "
          "nside-2048 children = %.4f" % (region, n_holes, n_occupied, hole_rate))

    out_path = config_module.product_path(config, "catalog", "sesna", "coverage", "hpx512", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.attrs["AREA_DEG2"] = area_deg2
        f.create_dataset("HPX_PIX", data=pix)
        f.create_dataset("FRAC", data=frac.astype(np.float32))
    return out_path, pix.size, area_deg2, hole_rate, len(batch_list)


def build(config, regions=None):
    """`build(config, regions=None)`: per region, `build_region` (rule 5c's
    per-region product, one file per region)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    with progress_module.Stage("catalog.coverage") as st:
        out_path = None
        for i, region in enumerate(region_names):
            out_path, n_pix, area_deg2, hole_rate, n_batches = build_region(config, region, st)
            st.tick(i + 1, len(region_names), "regions")
        st.done(out_path, regions=len(region_names))


if __name__ == "__main__":
    run(build)
