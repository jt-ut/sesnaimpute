"""The Spitzer completeness-footprint coverage fraction per pixel, per band
(IMPLEMENTATION.md section 1, SPEC_PRIORS.md section 2.1's coverage
denominator for the anchor counts).

`build(config, regions=None)` reads each field's own on-map footprint from
Rob Gutermuth's measured completeness maps (`catalog.completeness.
field_maps`, `sky/download/sesna_completeness/<Region>/`): a pixel is
covered where any of the region's own fields for that band is on-map
(COMPLETENESS_2026-10-20.md rule 4: "the footprint is the coverage source;
the mosaic mask file and its line are retired" -- the native Spitzer mosaic
masks this module used to read from `sky.download.spitzer_coverage` are no
longer acquired at all). For each region and each of the five Spitzer bands
(I1-I4, M1; the three 2MASS bands are all-sky and are not part of this
product), it measures the fraction of every nside-512 HEALPix pixel any of
the region's own completeness fields touch that falls on an on-map 30-arcsec
completeness pixel, and writes one `hpx512` product per region, schema
unchanged from the mosaic-mask version -- the coverage fraction per band the
depth grid, atlas and knot rate read. Pixel admission (which pixels carry a
SESNA source) is `granules.build`'s own job, from the curated catalogue, not
this product's.
"""

import functools
import os

import h5py
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from sesnaimpute import build as build_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog import completeness as completeness_module
from sesnaimpute.config import product_path
from sesnaimpute.definitions import BANDS_BY_KEY

# The five Spitzer bands the completeness maps carry; the three 2MASS bands
# are all-sky (fraction 1 everywhere) and are not columns of this product.
SPITZER_BANDS = ("I1", "I2", "I3", "I4", "M1")
for _band in SPITZER_BANDS:
    assert BANDS_BY_KEY[_band].survey in ("IRAC", "MIPS")

# The coarse grain the field-overlap search batches candidates on before
# dividing down to the published nside-512 pixel: not an output resolution
# of this product (there is no nside-256 "sightline" product any more).
NSIDE_CANDIDATE256 = 256
NSIDE_HPX512 = 512

# Candidate-grain padding and supersampling density, carried over unchanged
# from the mosaic-mask version of this stage: a field's corner-to-center
# angular radius padded 10% (TAN-projection corner distortion) plus a fixed
# 0.75 deg margin (about three nside-256 grain widths) sets the disc of
# nside-256 grains a field can touch; each candidate nside-512 grain is
# supersampled at 16x its own side (256 sample points/512-grain), which the
# mosaic-mask version's own convergence check found reproduces the exact
# overlap to <0.1% on every region checked, half that density already.
CANDIDATE_PAD_FACTOR = 1.1
CANDIDATE_PAD_MARGIN_DEG = 0.75
_SUPERSAMPLE_PER_SIDE = 16
SAMPLING_NSIDE = NSIDE_HPX512 * _SUPERSAMPLE_PER_SIDE

# Batches candidate nside-256 grains so one region's supersample array
# (up to 1024 points/grain) never exceeds a few hundred thousand points --
# bounds the astropy coordinate-transform working set for a large-mosaic
# region without changing the result (batches are just concatenated).
PIX256_BATCH = 256


def _fields_for_region(config, region):
    """`{band: [(astropy.wcs.WCS, on_map), ...]}` for every completeness
    field this region has for that band (`catalog.completeness.field_maps`):
    `on_map` (ny, nx) bool is the field's own on-map footprint (module
    docstring's coverage source). One `field_maps` call per band reads that
    band's own FITS files once; the per-region completeness product itself
    is not read here (this stage runs earlier in the RUNBOOK, before
    `catalog.completeness`, and only needs the raw field footprints)."""
    return {band: [(m.wcs, m.on_map) for m in completeness_module.field_maps(config, region, band)]
            for band in SPITZER_BANDS}


def _flatten_fields(fields_by_band):
    """`{band: [(wcs_params, on_map), ...]}`: the same fields
    `_fields_for_region` returns, with each field's `astropy.wcs.WCS`
    reduced to a tuple of plain numbers (`pieces.py`'s module docstring:
    only numpy arrays bound into a piece function are memory-mapped to the
    workers; a `WCS` object bound directly is pickled whole, with every
    batch). `on_map` is already a plain numpy array -- no run-length
    encoding is needed here (unlike the mosaic masks this replaces): a
    field's own 30-arcsec footprint is orders of magnitude smaller than a
    full-resolution Spitzer mosaic, so the flattened dict this returns is
    cheap to pickle once per batch."""
    out = {}
    for band, fields in fields_by_band.items():
        flat = []
        for wcs, on_map in fields:
            wcs_params = (float(wcs.wcs.crpix[0]), float(wcs.wcs.crpix[1]),
                          float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1]),
                          str(wcs.wcs.ctype[0]), str(wcs.wcs.ctype[1]),
                          float(wcs.wcs.cd[0, 0]), float(wcs.wcs.cd[0, 1]),
                          float(wcs.wcs.cd[1, 0]), float(wcs.wcs.cd[1, 1]))
            flat.append((wcs_params, on_map))
        out[band] = flat
    return out


def _wcs_from_params(wcs_params):
    """Rebuilds one field's `astropy.wcs.WCS` from the plain numbers
    `_flatten_fields` reduced it to."""
    crpix1, crpix2, crval1, crval2, ctype1, ctype2, cd11, cd12, cd21, cd22 = wcs_params
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [crpix1, crpix2]
    wcs.wcs.cdelt = [1.0, 1.0]
    wcs.wcs.crval = [crval1, crval2]
    wcs.wcs.ctype = [ctype1, ctype2]
    wcs.wcs.cd = [[cd11, cd12], [cd21, cd22]]
    return wcs


def _band_covered(ra_deg, dec_deg, fields):
    """Boolean, one entry per `(ra_deg, dec_deg)` sample point: True where
    the point's NEAREST pixel (COMPLETENESS_2026-10-20.md rule 1, the same
    convention `catalog.completeness.lookup` uses) is on-map in any of this
    band's fields. `fields` is `_flatten_fields`'s per-field tuples; the
    `WCS` is rebuilt here, once per field per call, from the numbers a
    piece actually receives memory-mapped."""
    covered = np.zeros(ra_deg.shape, dtype=bool)
    for wcs_params, on_map in fields:
        wcs = _wcs_from_params(wcs_params)
        px, py = wcs.wcs_world2pix(ra_deg, dec_deg, 0)
        ny, nx = on_map.shape
        col = np.rint(px).astype(np.int64)
        row = np.rint(py).astype(np.int64)
        inbounds = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
        idx = np.flatnonzero(inbounds)
        if idx.size == 0:
            continue
        covered[idx] |= on_map[row[idx], col[idx]]
    return covered


def _candidate_pix256(fields_by_band):
    """Sorted, unique nside-256 NESTED galactic grains any of the region's
    completeness fields can touch: the union, over every field of every
    band, of a disc centered on that field's own ICRS sky center (its
    central pixel, converted to galactic), radius the field's own
    corner-to-center angular half-diagonal padded by `CANDIDATE_PAD_FACTOR`
    plus `CANDIDATE_PAD_MARGIN_DEG`."""
    pix_parts = []
    for band in SPITZER_BANDS:
        for wcs, on_map in fields_by_band[band]:
            ny, nx = on_map.shape
            cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
            xs = np.array([0.0, nx - 1.0, 0.0, nx - 1.0, cx])
            ys = np.array([0.0, 0.0, ny - 1.0, ny - 1.0, cy])
            ra, dec = wcs.wcs_pix2world(xs, ys, 0)
            center = SkyCoord(ra=ra[4] * u.deg, dec=dec[4] * u.deg, frame="icrs")
            corners = SkyCoord(ra=ra[:4] * u.deg, dec=dec[:4] * u.deg, frame="icrs")
            radius_deg = float(np.max(center.separation(corners).deg))
            gal = center.galactic
            vec = hp.ang2vec(float(gal.l.deg), float(gal.b.deg), lonlat=True)
            radius_rad = np.radians(radius_deg * CANDIDATE_PAD_FACTOR + CANDIDATE_PAD_MARGIN_DEG)
            pix_parts.append(hp.query_disc(NSIDE_CANDIDATE256, vec, radius_rad, nest=True, inclusive=True))
    if not pix_parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(pix_parts)).astype(np.int64)


def _children(pix, factor):
    offsets = np.arange(factor, dtype=np.int64)
    return (pix[:, None] * factor + offsets[None, :]).ravel().astype(np.int64)


def _supersampled_fraction(pix256_batch, fields_by_band):
    """`(frac512, pix512)`, `frac512` `(n_pix, n_band)` float64, for one
    batch of nside-256 candidate grains: the mean, over each nside-512
    child's own `SAMPLING_NSIDE` grandchildren, of the per-band coverage
    boolean."""
    pix512 = _children(pix256_batch, (NSIDE_HPX512 // NSIDE_CANDIDATE256) ** 2)
    hi_factor = (SAMPLING_NSIDE // NSIDE_HPX512) ** 2
    hi_pix = _children(pix512, hi_factor)
    gl, gb = hp.pix2ang(SAMPLING_NSIDE, hi_pix, nest=True, lonlat=True)
    icrs = SkyCoord(l=gl * u.deg, b=gb * u.deg, frame="galactic").icrs
    ra_hi, dec_hi = icrs.ra.deg, icrs.dec.deg

    n512 = pix512.size
    frac512 = np.zeros((n512, len(SPITZER_BANDS)), dtype=np.float64)
    for j, band in enumerate(SPITZER_BANDS):
        fields = fields_by_band[band]
        if not fields:
            continue  # band never observed in this region: fraction stays 0
        covered_hi = _band_covered(ra_hi, dec_hi, fields)
        frac512[:, j] = covered_hi.reshape(n512, hi_factor).mean(axis=1)
    return frac512, pix512


def _coverage_piece(piece, pix256, fields_by_band):
    """One `PIX256_BATCH`-sized piece: the candidate-grain batch's own
    `(start, stop)` bounds into `pix256`, the region-wide candidate array
    (indices, not data, as `pieces.py` asks)."""
    start, stop = piece
    return _supersampled_fraction(pix256[start:stop], fields_by_band)


def _region_coverage(fields_by_band, n_jobs, stage=None):
    """`(pix512, frac512)` for one region: the nside-512 grains any of the
    region's completeness fields touch and their per-band covered fraction.
    The candidate nside-256 grains (`NSIDE_CANDIDATE256`, only a batching
    unit here, not a published resolution) are cut into `PIX256_BATCH`-sized
    pieces -- the same batching the astropy coordinate-transform working-
    set bound always needed -- and run on `sesnaimpute.pieces`' shared
    pool (owner ruling 2026-10-09: the region loop itself, in `build`
    below, stays sequential; this is where one region's work is spread
    over many small pieces). Each piece is independent (no partial sum to
    reduce), so running them out of order and concatenating in piece
    order reproduces the serial result exactly -- no float rounding at
    all, not even the ~1e-13 COMMON.md allows for a changed reduction
    order."""
    pix256 = _candidate_pix256(fields_by_band)
    if pix256.size == 0:
        empty = np.empty((0, len(SPITZER_BANDS)), dtype=np.float32)
        return pix256, empty
    pieces = [(start, min(start + PIX256_BATCH, pix256.size))
              for start in range(0, pix256.size, PIX256_BATCH)]
    # Bound into the piece function as its arrays and plain numbers
    # (`_flatten_fields`), not as `fields_by_band`'s `WCS` objects.
    fields_flat = _flatten_fields(fields_by_band)
    fn = functools.partial(_coverage_piece, pix256=pix256, fields_by_band=fields_flat)
    results = pieces_module.run(fn, pieces, n_jobs, stage=stage, unit="pixel batches")
    pix512 = np.concatenate([p512 for _f512, p512 in results])
    frac512 = np.concatenate([f512 for f512, _p512 in results]).astype(np.float32)
    return pix512, frac512


def _write(config, region, granule, pix, frac):
    out_path = product_path(config, "sky/derived", "spitzer", "coverage", granule, region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stem = f"coverage_spitzer_{granule}"
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = granule
        build_module.write_dataset(f, "HPX_PIX", pix.astype(np.int64), *REGISTRY[(stem, "HPX_PIX")],
                                    compression="gzip", compression_opts=4)
        build_module.write_dataset(f, "FRAC", frac.astype(np.float32), *REGISTRY[(stem, "FRAC")],
                                    compression="gzip", compression_opts=4)
        build_module.write_dataset(f, "BANDS", np.array(SPITZER_BANDS, dtype="S8"), *REGISTRY[(stem, "BANDS")])


def _build_one_region(config, region, n_jobs, stage):
    fields_by_band = _fields_for_region(config, region)
    pix512, frac512 = _region_coverage(fields_by_band, n_jobs, stage=stage)
    _write(config, region, "hpx512", pix512, frac512)
    return dict(region=region, n_hpx512=int(pix512.size),
                mean_frac=frac512.mean(axis=0) if frac512.size else np.zeros(len(SPITZER_BANDS)))


def build(config, regions=None):
    """Writes each region's `coverage_spitzer_hpx512` product. Regions run
    one after another (owner ruling 2026-10-09: regions are sequential,
    only the work inside one region is spread over workers); each
    region's own candidate-grain batches run on the shared pool
    `sesnaimpute.pieces.run` gives every stage, with the region's
    completeness-field footprints (`fields_by_band`) bound once per region
    as the pieces' shared, read-only input."""
    names = regions or [r.name for r in regions_module.REGIONS]
    with progress_module.Stage("sky.derived.coverage") as st:
        rows = [_build_one_region(config, name, config.n_jobs, st) for name in names]
        st.done(None, regions=len(names),
                hpx512_pixels=sum(r["n_hpx512"] for r in rows))


if __name__ == "__main__":
    run(build)
