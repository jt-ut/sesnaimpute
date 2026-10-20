"""The admitted-pixel axis and the low-column selection (SPEC_BMSTP_
DRAFT.md section 3.3): library code only, no RUNBOOK line and no product
of its own (COMPLETENESS_2026-10-20.md: "NEW catalog.completeness ...
replaces ... catalog.depth_grid's limit/width role" -- a genuine
per-pixel, per-band measured completeness curve needs no region-wide
counts fit and no per-source DCOMP90 shift standing in for it; see
`catalog.completeness.build` and `catalog.limits.completeness`).

`admitted_pixels_512` is the ADMITTED pixel axis every per-pixel product
in this package shares (IMPLEMENTATION_BMSTP_DRAFT.md P6): every nside-512
child of one of a region's source-bearing nside-256 pixels, not merely an
occupied one. `_low_column_selection` (with its own `_pixel_coverage`/
`_pixel_column` reads) is the "complete low-column" source mask
(`studies/star_galaxy_level.md` section 1): well-covered, low-extinction
sources, the reference population a region-wide fit or summary statistic
should use rather than the full, extinction-biased source set. Both are
kept here as plain functions for whichever consumer still needs them
(the old per-source/per-pixel limit-and-width product, `limits_sesna_
source`/`depth-grid_sesna_hpx512`, and the code that built them from a
region-wide Spitzer counts fit, are deleted outright, not kept beside
this).
"""

import os

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute.sky.derived.column import _region_codes

NSIDE_256 = 256
NSIDE_512 = 512

# Coverage floor for a "well-covered" admitted pixel (studies/
# star_galaxy_level.md section 1: "COVERAGE > 0.9, the any-IRAC union").
COVERAGE_FLOOR = 0.9

# The low-column quartile of the well-covered set (same section).
COLUMN_PERCENTILE = 25.0

# Below this many low-column-pixel sources, a region-wide fit on this
# selection falls back to every one of the region's sources (flagged).
MIN_LOW_COLUMN_SOURCES = 500


def admitted_pixels_512(config, region):
    """Every nside-512 child of the region's own source-bearing nside-256
    pixels -- the granule map's admission rule (`granules/build.py`): a
    pixel carries at least one SESNA source, full stop.

    Computed directly from the curated catalogue's own positions
    (`catalog.curated`'s `GAL_L_DEG`/`GAL_B_DEG`), not from the granule
    map: `granules.build`'s own admission table (`association/
    region_healpix256`) restricted to this region and to `N_SOURCE_ROWS >
    0` is exactly `unique(hp.ang2pix(256, gl, gb, nest=True))` for this
    region's own catalogued sources (`granules/build.py`'s
    `_region_arrays`/`source_pix256`; every association row carries
    sources, so the filter is a no-op kept for the statement) -- so this
    is the same pixel set, bit for bit, read one RUNBOOK line earlier
    (`catalog.curated`, not `granules.build`) to keep a cold-build reader
    off the granule map it would otherwise need before that map exists.
    """
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.depth_grid: no curated catalogue for %s at %s -- run the "
            "'sesnaimpute.catalog.curated' RUNBOOKtp.sh line first" % (region, path))
    with h5py.File(path, "r") as f:
        gl = np.asarray(f["GAL_L_DEG"][:], dtype=np.float64)
        gb = np.asarray(f["GAL_B_DEG"][:], dtype=np.float64)
    pix256 = hp.ang2pix(NSIDE_256, gl, gb, nest=True, lonlat=True).astype(np.int64)
    source_pix256 = np.unique(pix256)
    children = (source_pix256[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).reshape(-1)
    return np.unique(children)


def _pixel_coverage(config, region, pix):
    """Each admitted pixel's IRAC coverage fraction: `sky/derived/spitzer/
    coverage_spitzer_hpx512`'s own per-band `FRAC` (I1 I2 I3 I4 M1, that
    product's own column order), the fraction of the pixel on a covered
    mosaic pixel. The four IRAC bands' `max` is taken as the any-IRAC
    union fraction -- a LOWER BOUND on the true union (the product stores
    only each band's own fraction, not the union mask itself), exact
    where the four IRAC footprints coincide, which is nearly everywhere
    since 3.6/5.8 um share one camera array and 4.5/8.0 um the other. The
    Spitzer product covers a superset of the admitted pixels (every pixel
    a mosaic touches), so a pixel absent from it (never touched by a
    mosaic) reads FRAC = 0, the same join `bmstp.atlas._coverage` and
    `population.knot_rate._coverage_frac` perform.
    """
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.depth_grid: no Spitzer mosaic coverage for %s at %s -- run the "
            "'sesnaimpute.sky.derived.coverage' RUNBOOKtp.sh line first" % (region, path))
    with h5py.File(path, "r") as f:
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)
    union = frac[:, :4].max(axis=1) if frac.size else np.empty(0, dtype=np.float64)
    order = np.argsort(cov_pix)
    loc = np.minimum(np.searchsorted(cov_pix[order], pix), max(cov_pix.size - 1, 0))
    hit = order[loc]
    found = cov_pix[hit] == pix if cov_pix.size else np.zeros(pix.size, dtype=bool)
    out = np.zeros(pix.size, dtype=np.float64)
    out[found] = union[hit[found]]
    return out


def _pixel_column(config, region, pix):
    """Each admitted pixel's own extinction column, from the nside-256
    sightline it is a child of (survey-wide product, one row per region and
    pixel: this region's own rows are read;
    `granules/build.py`'s own `HPX_PIX_256 = HPX_PIX_512 // 4`) -- the same
    join `bmstp.atlas._pixel_column` performs. The low-column selection is
    a starlight (IRAC) selection, so it reads the extinction column, not
    the gas column the young-star law was measured on.
    """
    parent256 = pix // 4
    path = config_module.product_path(config, "sky/derived", "adopted", "extinction", "sightline")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.depth_grid: no extinction sightline column at %s -- run the "
            "'sesnaimpute.sky.derived.column' RUNBOOKtp.sh line first" % path)
    code = _region_codes(config, [region])[region]
    with h5py.File(path, "r") as f:
        own = np.asarray(f["REGION_CODE"][:], dtype=np.int64) == code
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)[own]
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)[own]
    order = np.argsort(sl_pix)
    loc = np.minimum(np.searchsorted(sl_pix[order], parent256), sl_pix.size - 1)
    hit = order[loc]
    found = sl_pix[hit] == parent256
    if not np.all(found):
        raise ValueError("catalog.depth_grid: %d admitted pixel(s) have no sightline column in %s"
                          % (int(np.sum(~found)), path))
    return a_k[hit]


def _low_column_selection(config, region, hpx_pix_512, admitted):
    """The region's own "complete low-column" source mask
    (`studies/star_galaxy_level.md` section 1): IRAC-union coverage above
    `COVERAGE_FLOOR` at the source's own admitted pixel, and that pixel's
    own adopted column below the `COLUMN_PERCENTILE` of the well-covered
    set's own column. Falls back to every one of the region's sources,
    flagged, if fewer than `MIN_LOW_COLUMN_SOURCES` sources pass. Returns
    `(mask, n_selected, fallback)`.
    """
    pix = np.sort(admitted)
    cov = _pixel_coverage(config, region, pix)
    col = _pixel_column(config, region, pix)
    well_covered = cov > COVERAGE_FLOOR
    if well_covered.any():
        threshold = float(np.percentile(col[well_covered], COLUMN_PERCENTILE))
        low_column_pix = pix[well_covered & (col < threshold)]
    else:
        low_column_pix = np.empty(0, dtype=np.int64)
    mask = np.isin(hpx_pix_512, low_column_pix)
    n_selected = int(mask.sum())
    if n_selected < MIN_LOW_COLUMN_SOURCES:
        return np.ones(hpx_pix_512.shape[0], dtype=bool), n_selected, True
    return mask, n_selected, False
