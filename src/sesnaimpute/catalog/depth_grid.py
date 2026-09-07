"""The depth grid: median 50%-completeness limit per nside-512 pixel
(SPEC_BMSTP_DRAFT.md section 3.3), the object the prior atlas reads at a
pixel's median limits instead of a per-source one (SPEC_PRIORS.md
section 1.3, "the atlas evaluates the same code at a pixel's median
limits"). One row per ADMITTED pixel (IMPLEMENTATION_BMSTP_DRAFT.md P6),
not merely per occupied one.

Per region, admission is the granule map's own rule (`granules/build.py`):
every nside-512 child of one of the region's source-bearing nside-256
pixels. For an OCCUPIED admitted pixel (carrying at least one catalogued
source), `F_LIM_50_MED_MJY` is the median, over the pixel's own sources,
of each band's 50%-completeness limit -- `catalog.limits.limits`, the one
limits reader every consumer uses, evaluated once for the whole region
and grouped by pixel with a pandas groupby (vectorised: no loop over
pixels). An admitted pixel with no sources of its own (a mosaic-support
sibling of an occupied nside-256 pixel) takes the row of its nearest
occupied pixel by angular distance between pixel centres (a vectorised
argmax of the unit-vector dot product against every occupied pixel, no
loop over pixels), with its own `N_SOURCES = 0` so a consumer can tell a
filled row from a measured one.
"""

import os

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access as access_module

NSIDE_512 = 512

# Row-batch memory budget for the per-source limits array before its
# grouping by pixel (CODING_RULES.md rule 10b); at 8 bands of float32
# plus one int64 pixel id this admits far more than any region's source
# count in one batch.
ROW_BATCH_BUDGET_BYTES = 512 << 20

# Ten pixels for the fixed-seed identity check the brief asks for.
CHECK_SEED = 20260907
N_CHECK_PIXELS = 10


def _admitted_pixels_512(config, region):
    """Every nside-512 child of the region's own source-bearing nside-256
    pixels -- the granule map's admission rule (`granules/build.py`): a
    pixel/region association, not a per-source column, so read directly
    rather than through `granules.access.per_source`.
    """
    path = config_module.product_path(config, "granules", "sesna", "granule-map", "source")
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["region/REGION"][:]]
        code = f["region/REGION_CODE"][names.index(region)]
        g = f["association/region_healpix256"]
        pix256 = np.asarray(g["HPX_PIX_256"][:], dtype=np.int64)
        region_code = np.asarray(g["REGION_CODE"][:])
        n_source_rows = np.asarray(g["N_SOURCE_ROWS"][:])
    source_pix256 = pix256[(region_code == code) & (n_source_rows > 0)]
    children = (source_pix256[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).reshape(-1)
    return np.unique(children)


def _pixel_medians(hpx_pix_512, source_limits):
    """One region's per-pixel source count and median 50%-completeness
    limit, over every band, grouped by occupied pixel -- a single pandas
    groupby, no loop over pixels.
    """
    band_keys = [b.key for b in definitions.BANDS]
    df = pd.DataFrame(source_limits, columns=band_keys)
    df["HPX_PIX_512"] = hpx_pix_512
    grouped = df.groupby("HPX_PIX_512", sort=True)
    n_sources = grouped.size()
    medians = grouped[band_keys].median()
    return (
        medians.index.to_numpy(dtype=np.int64),
        n_sources.to_numpy(dtype=np.int32),
        medians.to_numpy(dtype=np.float32),
    )


def _fill_unoccupied(admitted, occupied, occupied_n_sources, occupied_medians):
    """Every admitted pixel's row: an occupied pixel keeps its own
    measured count and median; an admitted pixel with no sources takes
    the row of its nearest occupied pixel by angular distance between
    pixel centres -- a vectorised argmax of the unit-vector dot product
    (monotonic with angular distance) against every occupied pixel, no
    loop over pixels -- with its own `N_SOURCES = 0`.
    Returns `(pix, n_sources, f_lim_50_med_mjy, n_filled)`, `pix` sorted.
    """
    pix = np.sort(admitted)
    is_occupied = np.isin(pix, occupied)

    occ_order = np.argsort(occupied)
    occ_row = occ_order[np.searchsorted(occupied[occ_order], pix[is_occupied])]

    n_sources = np.zeros(pix.size, dtype=np.int32)
    f_lim_50_med_mjy = np.empty((pix.size, occupied_medians.shape[1]), dtype=np.float32)
    n_sources[is_occupied] = occupied_n_sources[occ_row]
    f_lim_50_med_mjy[is_occupied] = occupied_medians[occ_row]

    filled_mask = ~is_occupied
    if filled_mask.any():
        occ_vec = np.asarray(hp.pix2vec(NSIDE_512, occupied, nest=True)).T  # (n_occ, 3)
        filled_vec = np.asarray(hp.pix2vec(NSIDE_512, pix[filled_mask], nest=True)).T
        nearest = np.argmax(filled_vec @ occ_vec.T, axis=1)  # no loop over pixels
        f_lim_50_med_mjy[filled_mask] = occupied_medians[nearest]

    return pix, n_sources, f_lim_50_med_mjy, int(filled_mask.sum())


def _check_identity(hpx_pix_512, source_limits, occupied, occupied_medians,
                     n_admitted, n_filled, region):
    """Prints, for ten occupied pixels chosen by a fixed seed, a direct
    numpy median over the per-source limits against the stored row --
    the brief's own identity check, run from the build itself -- plus
    the region's admitted/occupied/filled pixel counts.
    """
    rng = np.random.default_rng(CHECK_SEED)
    n_pick = min(N_CHECK_PIXELS, occupied.size)
    picked = rng.choice(occupied.size, size=n_pick, replace=False)
    print("depth_grid identity, %s, %d pixels (admitted=%d occupied=%d filled=%d):"
          % (region, n_pick, n_admitted, occupied.size, n_filled))
    for i in picked:
        p = occupied[i]
        direct = np.median(source_limits[hpx_pix_512 == p, :], axis=0).astype(np.float32)
        stored = occupied_medians[i]
        print("  pixel %d: direct median == stored row -> %s" %
              (p, bool(np.allclose(direct, stored, equal_nan=True))))


def build(config, regions=None):
    """Builds the per-region depth-grid product for the given regions
    (default: every region in `regions.REGIONS`).
    """
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]

    with progress_module.Stage("catalog.depth_grid") as st:
        n_done = 0
        out_paths = []
        n_admitted_total = n_occupied_total = n_filled_total = 0
        for region in regions:
            rs = access_module.region_slice(config, region)
            source_limits = limits_module.limits(config, region)

            hpx_pix_512_parts, limits_parts = [], []
            row_bytes = 8 + 8 * 4
            for start, stop in batches_module.batches(
                rs["n_sources"], row_bytes, budget_bytes=ROW_BATCH_BUDGET_BYTES
            ):
                hpx_pix_512_parts.append(rs["hpx_pix_512"][start:stop])
                limits_parts.append(source_limits[start:stop])
            hpx_pix_512 = np.concatenate(hpx_pix_512_parts)
            all_limits = np.concatenate(limits_parts)

            occupied, occupied_n_sources, occupied_medians = _pixel_medians(hpx_pix_512, all_limits)
            admitted = _admitted_pixels_512(config, region)
            pix, n_sources, f_lim_50_med_mjy, n_filled = _fill_unoccupied(
                admitted, occupied, occupied_n_sources, occupied_medians
            )
            _check_identity(hpx_pix_512, all_limits, occupied, occupied_medians,
                             admitted.size, n_filled, region)

            out_path = config_module.product_path(
                config, "catalog", "sesna", "depth-grid", "hpx512", region=region
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with h5py.File(out_path, "w") as f:
                f.attrs["GRANULE"] = "hpx512"
                f.create_dataset("HPX_PIX_512", data=pix)
                f.create_dataset("N_SOURCES", data=n_sources)
                f.create_dataset("F_LIM_50_MED_MJY", data=f_lim_50_med_mjy)

            out_paths.append(out_path)
            n_admitted_total += int(admitted.size)
            n_occupied_total += int(occupied.size)
            n_filled_total += int(n_filled)
            n_done += 1
            st.tick(n_done, len(regions), "regions")

        st.done(out_paths[-1] if out_paths else None, regions=len(regions),
                admitted=n_admitted_total, occupied=n_occupied_total, filled=n_filled_total)


if __name__ == "__main__":
    run(build)
