"""The Spitzer mosaic coverage fraction per pixel, per band (IMPLEMENTATION.md
section 1, SPEC_PRIORS.md section 2.1's coverage denominator for the anchor
counts).

`build(config, regions=None)` reads the native per-field coverage masks SESNA
already carries at `sky/download/spitzer_coverage/spitzer_coverage_masks.hdf5`
(no download module: these are SESNA's own mosaics and already live in the
data root under the tree `IMPLEMENTATION.md` section 1a names for external
bytes). For each region and each of the five Spitzer bands (I1-I4, M1; the
three 2MASS bands are all-sky and are not part of this product), it measures
the fraction of every nside-512 and every nside-256 HEALPix pixel the
region's own mosaics touch that falls on a covered mask pixel, and writes one
`hpx512` and one `sightline` product per region.
"""

import os

import h5py
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from joblib import Parallel, delayed

from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.config import product_path
from sesnaimpute.definitions import BANDS_BY_KEY

# The five Spitzer bands the native masks carry; the three 2MASS bands are
# all-sky (fraction 1 everywhere) and are not columns of this product.
SPITZER_BANDS = ("I1", "I2", "I3", "I4", "M1")
for _band in SPITZER_BANDS:
    assert BANDS_BY_KEY[_band].survey in ("IRAC", "MIPS")

NSIDE_SIGHTLINE = 256
NSIDE_HPX512 = 512

# Candidate-grain padding and supersampling density, lifted from the old
# sesnacomplete.fetch_external.healpix_coverage_fraction quarry: a field's
# corner-to-center angular radius padded 10% (TAN-projection corner
# distortion) plus a fixed 0.75 deg margin (about three nside-256 grain
# widths) sets the disc of nside-256 grains a field can touch; each
# candidate nside-512 grain is supersampled at 16x its own side (256 sample
# points/512-grain), which the quarry's own convergence check found
# reproduces the exact overlap to <0.1% on every region checked, half that
# density already.
CANDIDATE_PAD_FACTOR = 1.1
CANDIDATE_PAD_MARGIN_DEG = 0.75
_SUPERSAMPLE_PER_SIDE = 16
SAMPLING_NSIDE = NSIDE_HPX512 * _SUPERSAMPLE_PER_SIDE

# Batches candidate nside-256 grains so one region's supersample array
# (up to 1024 points/grain) never exceeds a few hundred thousand points --
# bounds the astropy coordinate-transform working set for a large-mosaic
# region without changing the result (batches are just concatenated).
PIX256_BATCH = 256


def _masks_path(config):
    return f"{config.data_root}/sky/download/spitzer_coverage/spitzer_coverage_masks.hdf5"


def _field_masks(masks_path, region):
    """`{band: [(astropy.wcs.WCS, bool mask array), ...]}` for `region`'s
    own per-field masks (`MASKS/<region>/<band>/field_*`), which carry a
    real CD-matrix WCS; the `ALLBAND/union_*` datasets are a different,
    lower-fidelity reprojection and are not read here."""
    out = {band: [] for band in SPITZER_BANDS}
    with h5py.File(masks_path, "r") as f:
        group = f["MASKS"][region]
        for band in SPITZER_BANDS:
            if band not in group:
                continue
            band_group = group[band]
            for key in band_group.keys():
                if not key.startswith("field_"):
                    continue
                dataset = band_group[key]
                mask = np.asarray(dataset[()], dtype=bool)
                attrs = dataset.attrs
                wcs = WCS(naxis=2)
                wcs.wcs.crpix = [attrs["CRPIX1"], attrs["CRPIX2"]]
                wcs.wcs.cdelt = [1.0, 1.0]
                wcs.wcs.crval = [attrs["CRVAL1"], attrs["CRVAL2"]]
                wcs.wcs.ctype = [str(attrs["CTYPE1"]), str(attrs["CTYPE2"])]
                wcs.wcs.cd = [[attrs["CD1_1"], attrs["CD1_2"]],
                              [attrs["CD2_1"], attrs["CD2_2"]]]
                out[band].append((wcs, mask))
    return out


def _band_covered(ra_deg, dec_deg, fields):
    """Boolean, one entry per `(ra_deg, dec_deg)` sample point: True where
    the point falls on a covered pixel of any of this band's field masks
    (nearest-pixel WCS lookup, the same test the old anchor code used)."""
    covered = np.zeros(ra_deg.shape, dtype=bool)
    for wcs, mask in fields:
        px, py = wcs.wcs_world2pix(ra_deg, dec_deg, 0)
        ny, nx = mask.shape
        col = np.rint(px).astype(np.int64)
        row = np.rint(py).astype(np.int64)
        inbounds = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
        idx = np.flatnonzero(inbounds)
        if idx.size == 0:
            continue
        covered[idx] |= mask[row[idx], col[idx]]
    return covered


def _candidate_pix256(fields_by_band):
    """Sorted, unique nside-256 NESTED galactic grains any of the region's
    field masks can touch: the union, over every field of every band, of a
    disc centered on that field's own ICRS sky center (its central pixel,
    converted to galactic), radius the field's own corner-to-center angular
    half-diagonal padded by `CANDIDATE_PAD_FACTOR` plus
    `CANDIDATE_PAD_MARGIN_DEG`."""
    pix_parts = []
    for band in SPITZER_BANDS:
        for wcs, mask in fields_by_band[band]:
            ny, nx = mask.shape
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
            pix_parts.append(hp.query_disc(NSIDE_SIGHTLINE, vec, radius_rad, nest=True, inclusive=True))
    if not pix_parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(pix_parts)).astype(np.int64)


def _children(pix, factor):
    offsets = np.arange(factor, dtype=np.int64)
    return (pix[:, None] * factor + offsets[None, :]).ravel().astype(np.int64)


def _supersampled_fraction(pix256_batch, fields_by_band):
    """`(frac256, frac512)`, each `(n_pix, n_band)` float64, for one batch
    of nside-256 candidate grains: the mean, over each grain's own
    `SAMPLING_NSIDE` children, of the per-band coverage boolean -- both
    nsides read off the same hi-res sample array, so the nside-512 rows
    aggregate exactly into their nside-256 parent."""
    pix512 = _children(pix256_batch, (NSIDE_HPX512 // NSIDE_SIGHTLINE) ** 2)
    hi_factor = (SAMPLING_NSIDE // NSIDE_HPX512) ** 2
    hi_pix = _children(pix512, hi_factor)
    gl, gb = hp.pix2ang(SAMPLING_NSIDE, hi_pix, nest=True, lonlat=True)
    icrs = SkyCoord(l=gl * u.deg, b=gb * u.deg, frame="galactic").icrs
    ra_hi, dec_hi = icrs.ra.deg, icrs.dec.deg

    n512 = pix512.size
    n256 = pix256_batch.size
    frac512 = np.zeros((n512, len(SPITZER_BANDS)), dtype=np.float64)
    frac256 = np.zeros((n256, len(SPITZER_BANDS)), dtype=np.float64)
    for j, band in enumerate(SPITZER_BANDS):
        fields = fields_by_band[band]
        if not fields:
            continue  # band never observed in this region: fraction stays 0
        covered_hi = _band_covered(ra_hi, dec_hi, fields)
        frac512[:, j] = covered_hi.reshape(n512, hi_factor).mean(axis=1)
        frac256[:, j] = covered_hi.reshape(n256, 4 * hi_factor).mean(axis=1)
    return frac256, frac512, pix512


def _region_coverage(fields_by_band):
    """`(pix256, frac256, pix512, frac512)` for one region: candidate
    grains at both resolutions and their per-band covered fraction,
    batched over `PIX256_BATCH` nside-256 grains at a time."""
    pix256 = _candidate_pix256(fields_by_band)
    if pix256.size == 0:
        empty = np.empty((0, len(SPITZER_BANDS)), dtype=np.float32)
        return pix256, empty, pix256, empty
    frac256_parts, frac512_parts, pix512_parts = [], [], []
    for start in range(0, pix256.size, PIX256_BATCH):
        batch = pix256[start:start + PIX256_BATCH]
        f256, f512, p512 = _supersampled_fraction(batch, fields_by_band)
        frac256_parts.append(f256)
        frac512_parts.append(f512)
        pix512_parts.append(p512)
    frac256 = np.concatenate(frac256_parts).astype(np.float32)
    pix512 = np.concatenate(pix512_parts)
    frac512 = np.concatenate(frac512_parts).astype(np.float32)
    return pix256, frac256, pix512, frac512


def _write(config, region, granule, pix, frac):
    out_path = product_path(config, "sky/derived", "spitzer", "coverage", granule, region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = granule
        f.create_dataset("HPX_PIX", data=pix.astype(np.int64), compression="gzip", compression_opts=4)
        f.create_dataset("FRAC", data=frac.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("BANDS", data=np.array(SPITZER_BANDS, dtype="S8"))


def _build_one_region(config, region, masks_path):
    fields_by_band = _field_masks(masks_path, region)
    pix256, frac256, pix512, frac512 = _region_coverage(fields_by_band)
    _write(config, region, "sightline", pix256, frac256)
    _write(config, region, "hpx512", pix512, frac512)
    return dict(region=region, n_sightline=int(pix256.size), n_hpx512=int(pix512.size),
                mean_frac=frac512.mean(axis=0) if frac512.size else np.zeros(len(SPITZER_BANDS)))


def build(config, regions=None):
    """Writes each region's `coverage_spitzer_hpx512` and
    `coverage_spitzer_sightline` products. Regions run in a joblib pool;
    each worker reads only its own region's field masks."""
    masks_path = _masks_path(config)
    if not os.path.exists(masks_path):
        raise FileNotFoundError(
            "sky.derived.coverage.build: no native Spitzer mosaic mask file at %s -- "
            "these are SESNA's own mosaics and are expected already staged in the "
            "data root; nothing acquires them" % masks_path)
    names = regions or [r.name for r in regions_module.REGIONS]
    Parallel(n_jobs=-1)(delayed(_build_one_region)(config, name, masks_path) for name in names)


if __name__ == "__main__":
    run(build)
