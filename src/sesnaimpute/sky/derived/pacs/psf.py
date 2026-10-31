"""The PACS point-spread function: the release templates and encircled-energy tables, the input maps a mosaic is built
from, and the reader of a mosaic's PSF (the local mixture of the input maps' templates on the output grid).

Depends on: grid. Replaces: sky/derived/pacs_psf.py (all of it but `CachedPairedPSF`/`PairedPSF`, which only served
non-mosaic maps; their interpolation core `_cached_responses` is kept), pacs_fit._template_density, and the input-map
primitives pacs_mosaic._read_layers/_good/valid_at/_smooth_lattice/_lattice_at (here because MosaicPSF.weights needs
them and psf precedes mosaic). The build of a mosaic's PSF tables is mosaic.py's (`mosaic.build_mosaic_psf`: it
measures the tables with the aperture core and the TAN area factors, both mosaic's).

The release (Calzoletti, ESAC, 16 May 2017, HERSCHEL-HSC-DOC-2181 "PACS Photometer PSFs and EEFs as Ancillary Data
Products: Release note"), distributing the PSFs and EEFs of PICC-ME-TN-033 v2.2 (Bocchio, Lutz et al. 2016, "PACS
photometer point spread function"): Vesta templates, OD345, array-to-map angles +42 and -42, pixfrac 0.1, files in
PSF_Data/ and EEF_Data/ under https://archives.esac.esa.int/hsa/legacy/ADP/PSF/PACS/PACS-P/, placed once by hand in
`<herschel_pacs_maps>/calibration/psf/` (derived stages run without network).

Normalisation. The release images are 201 x 201 at 1 arcsec, flux-normalised so the signal inside r = 60 arcsec is 1
after the mean of the 61-70 arcsec annulus is subtracted everywhere (release note sec 2.2; PICC-ME-TN-033 sec 9). That
subtracts the same constant c per unit area from the true PSF, so the true cumulative energy is
EEF(r) = a V(r) + c pi r^2 with V the image's own cumulative sum. The release's EEF tables carry both: column 3 is the
encircled fraction of the total out to 1000 arcsec and column 5 is V(r) out to 60 arcsec; a least squares of column 3
on column 5 and pi r^2 over the 100 radii closes to 1e-12 and gives a = 0.911 (blue) and 0.906 (green), the same for
every speed and mode of a band. Scaling each image by a makes its amplitude the total flux; the removed pedestal c is
absorbed by the fitted local background.

A map's templates: a pair, one per scan direction, each placed on the sky at the spacecraft position angle of the
observation that made it; a parallel map holding one scan direction has that direction's template alone
(`published_components`).

Store of a mosaic's PSF (under the mosaic directory, mosaic.py): `psf/psf.json` (sub-group keys, per input map {path,
wcs_header, shape, smooth, smooth_step, scale, sub, bbox, hdus}, extent, subpixel_steps, integration_order, fbar,
enclosed_sub, aperture_fraction_sub, sub_beams, beam_arcsec, enclosed_fraction, aperture_fraction,
broad_over_plain_total, table_half), `psf/sub<k>.npy` (float32 (n_nodes, n_nodes, 1, count, count)), `psf/table.npy`
(the fbar-weighted mean, same shape); each input map's smoothed error is the TiledLayer `<mosaic>/<smooth>`.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits
from numba import njit
from scipy.ndimage import map_coordinates

from sesnaimpute.definitions import PACS_APERTURES as APERTURES
from sesnaimpute.sky.derived.pacs import grid
from sesnaimpute.sky.derived.pacs.grid import TiledLayer
from sesnaimpute.sky.download.herschel_pacs.index import scan_angles as _index_scan_angles

# (band, mode, speed arcsec/s) -> the release's band tag, speed tag, EEF file, and the beam FWHM (geometric mean of the
# two fitted axes, arcsec) that sets the blend radius, the fitting box and the extended-source threshold. Axes (minor x
# major, arcsec), PICC-ME-TN-033 v2.2:
#   parallel (mode 31): Table 6, the equal-weight sum of the +42 and -42 scans (the beam of a finished map):
#       P70  60"/s 8.80x9.60   20"/s 5.74x6.26;   P100 60"/s 9.73x10.69  20"/s 6.98x7.42
#   scan map (mode 1): Table 4 (OD160, prime mode, one scan direction; a finished two-direction map is 3-4 per cent
#     broader):  P70  20"/s 5.41x5.72   60"/s 5.70x9.05;   P100 20"/s 6.66x6.89   60"/s 6.84x9.81
#   160 um (red): templates for the scan speeds only (20, 60 "/s) and EEF tables for red 20, 20para and 60para; parallel
#     maps use the scan template at their speed with the parallel EEF table, scan-map 60 "/s the 60para table. Beams:
#     Gaussian fits to the release templates (+42/-42 mean), 2026-10-27.
_COMBINATIONS = {
    ("P70", 31, 60): ("blu", "60para", "EEF_blu_60para_21.txt", np.sqrt(8.80 * 9.60)),
    ("P70", 31, 20): ("blu", "20para", "EEF_blu_20para_21.txt", np.sqrt(5.74 * 6.26)),
    ("P100", 31, 60): ("grn", "60para", "EEF_grn_60para_21.txt", np.sqrt(9.73 * 10.69)),
    ("P100", 31, 20): ("grn", "20para", "EEF_grn_20para_21.txt", np.sqrt(6.98 * 7.42)),
    ("P70", 1, 20): ("blu", "20", "EEF_blu_20.txt", np.sqrt(5.41 * 5.72)),
    ("P70", 1, 60): ("blu", "60", "EEF_blu_60_22.txt", np.sqrt(5.70 * 9.05)),
    ("P100", 1, 20): ("grn", "20", "EEF_grn_20.txt", np.sqrt(6.66 * 6.89)),
    ("P100", 1, 60): ("grn", "60", "EEF_grn_60_22.txt", np.sqrt(6.84 * 9.81)),
    ("P160", 31, 60): ("red", "60", "EEF_red_60para_21.txt", 11.4),
    ("P160", 31, 20): ("red", "20", "EEF_red_20para_21.txt", 10.3),
    ("P160", 1, 20): ("red", "20", "EEF_red_20.txt", 10.3),
    ("P160", 1, 60): ("red", "60", "EEF_red_60para_21.txt", 11.4),
}
# The two array-to-map angles of the OD345 templates; scan-map observations use scan angles of 0/45/135 degrees, and
# the +-42 pair is the nearest the release offers.
_ARRAY_TO_MAP_ANGLES = ("+42", "-42")
_DIRECTION_TAGS = (("nominal", "+42"), ("orthogonal", "-42"))
# The release image: 201 x 201, 1 arcsec pixels, centred on pixel (101, 101).
_SIZE = 201
_CENTRE = 100
_NORMALISATION_RADIUS = 60.0
#: per-position memories of a MosaicPSF (last recently used): mixture fractions, and response patches
WEIGHTS_KEPT = 8
PATCHES_KEPT = 4


# ----------------------------------------------------------------------------------------------------------------------
# the release templates and encircled energies

def _key(record) -> tuple:
    key = (str(record.band), int(record.mode), int(round(float(record.scan_speed))))
    if key not in _COMBINATIONS:
        raise ValueError(f"No PACS PSF release for band/mode/speed {key}; supported: {sorted(_COMBINATIONS)}")
    return key


def beam_fwhm_arcsec(record) -> float:
    """The beam FWHM (geometric mean of the release's fitted axes) of a map's (band, mode, speed), arcsec. Reads
    nothing. Identity: `pacs_psf.beam_fwhm_arcsec` (pacs_psf.py:122) for an input map."""
    return float(_COMBINATIONS[_key(record)][3])


def psf_directory(config) -> str:
    """The PSF release directory: <herschel_pacs_maps>/calibration/psf."""
    return str(Path(config.inputs["herschel_pacs_maps"]) / "calibration" / "psf")


def _eef_table(eef_path) -> np.ndarray:
    rows = [line.split() for line in Path(eef_path).read_text().splitlines() if line.strip() and not line.startswith("#")]
    return np.array([[float(v) for v in row] for row in rows if len(row) == 5])


def _eef_factors(eef_path) -> Tuple[float, float]:
    """(a, c) of the least squares EEF(r) = a V(r) + c pi r^2 (band-level, see the module docstring)."""
    table = _eef_table(eef_path)
    radius, eef, vesta = table[:, 0], table[:, 2], table[:, 4]
    design = np.column_stack([vesta, np.pi * radius ** 2])
    (factor, pedestal), *_ = np.linalg.lstsq(design, eef, rcond=None)
    return float(factor), float(pedestal)


def _release_fraction(config, record, annulus_share: bool) -> float:
    """The fraction of a point source's total flux the eHOPS aperture returns (with or without the PSF wing the sky
    annulus removes), from the release EEF table of the map's band, mode and speed (log-radius interpolation)."""
    r_ap, r_in, r_out = APERTURES[record.band]
    table = _eef_table(Path(psf_directory(config)) / _COMBINATIONS[_key(record)][2])
    radius, eef = table[:, 0], table[:, 2]
    ee = lambda r: float(np.interp(np.log(r), np.log(radius), eef))
    frac = ee(r_ap)
    if annulus_share:
        frac -= r_ap ** 2 / (r_out ** 2 - r_in ** 2) * (ee(r_out) - ee(r_in))
    return frac


def _single_template_fraction(config, record, tag: str, annulus_share: bool) -> float:
    """The aperture fraction of a one-direction parallel map: the band's a and c applied to this template's OWN
    cumulative V(r) (the release image summed within r on its 1-arcsec grid, bilinear on an 8x finer sampling),
    EEF(r) = a V(r) + c pi r^2, then the eHOPS aperture and annulus share as `_release_fraction`."""
    r_ap, r_in, r_out = APERTURES[record.band]
    a, c = _eef_factors(Path(psf_directory(config)) / _COMBINATIONS[_key(record)][2])
    template = PublishedPSF.read(config, record, tag, position_angle_deg=0.0)
    raw = template.image / template.total_flux_factor
    n = 8
    lattice = np.linspace(-.5 + .5 / n, 2 * _CENTRE + .5 - .5 / n, _SIZE * n)
    yy, xx = np.meshgrid(lattice, lattice, indexing="ij")
    fine = map_coordinates(raw, [yy, xx], order=1, mode="nearest", prefilter=False)
    rr = np.hypot(xx - _CENTRE, yy - _CENTRE)
    ee = lambda r: a * float(fine[rr <= r].sum()) / n ** 2 + c * np.pi * r ** 2
    frac = ee(r_ap)
    if annulus_share:
        frac -= r_ap ** 2 / (r_out ** 2 - r_in ** 2) * (ee(r_out) - ee(r_in))
    return frac


def _fraction(config, record, annulus_share: bool) -> float:
    single = _single_direction_tag(config, record)
    if single is not None:
        if record.band == "P100":
            raise ValueError("the +42 and -42 P100 templates differ by about 3 percent inside 8 arcsec, so the "
                             "single-template aperture fraction is unverified at that band")
        return _single_template_fraction(config, record, single, annulus_share)
    return _release_fraction(config, record, annulus_share)


def aperture_eef(config, record) -> float:
    """The fraction of the total flux inside the eHOPS aperture radius (definitions.PACS_APERTURES[band][0]), no sky:
    the divisor of every aperture sum (the normalisation of a mosaic's PSF tables). Reads the release EEF table of the
    map's (band, mode, speed), or the one-direction template. Identity: `pacs_psf.aperture_eef` (pacs_psf.py:171)."""
    return _fraction(config, record, annulus_share=False)


def aperture_fraction(config, record) -> float:
    """The fraction of the total flux the aperture keeps after the annulus sky (`pacs_psf.aperture_fraction`,
    pacs_psf.py:155): e.g. P70 parallel-60 0.712, scan-20 0.756; P100 0.545 / 0.632."""
    return _fraction(config, record, annulus_share=True)


@dataclass(frozen=True)
class PublishedPSF:
    """One release template (`pacs_psf.PublishedPSF`): `image` (201, 201) float64 scaled to total flux,
    `position_angle_deg`, `total_flux_factor` (a), `path`."""
    image: np.ndarray
    position_angle_deg: float
    total_flux_factor: float
    path: str

    @classmethod
    def read(cls, config, record, array_to_map_angle: str, *, position_angle_deg: float) -> "PublishedPSF":
        """Reads the release template of the map's (band, mode, speed) at `array_to_map_angle` ("+42" or "-42"),
        scaled to total flux, placed at the spacecraft position angle. Identity: `pacs_psf.PublishedPSF.read`."""
        band_tag, speed_tag, eef_name, _ = _COMBINATIONS[_key(record)]
        folder = Path(psf_directory(config))
        path = folder / f"psf20_{band_tag}_{speed_tag}_vesta_od345_ama{array_to_map_angle}_0p10.fits.gz"
        if array_to_map_angle not in _ARRAY_TO_MAP_ANGLES or not np.isfinite(position_angle_deg):
            raise ValueError("Require a release array-to-map angle and a finite position angle")
        with fits.open(path) as hdus:
            header = hdus["image"].header
            data = np.array(hdus["image"].data, dtype=float)
        if (data.shape != (_SIZE, _SIZE) or not np.all(np.isfinite(data))
                or not np.allclose([header["CDELT1"], header["CDELT2"]], [-1 / 3600, 1 / 3600])
                or (header["CRPIX1"], header["CRPIX2"]) != (101., 101.)):
            raise ValueError(f"Unexpected published PSF geometry: {path}")
        yy, xx = np.indices(data.shape)
        inside = np.hypot(xx - _CENTRE, yy - _CENTRE) <= _NORMALISATION_RADIUS
        # the release images sum to 1 within 60 arcsec to +-5e-4 (red: 0.9995-1.0003); +-1e-3 accepts them
        if not np.isclose(data[inside].sum(), 1., atol=1e-3):
            raise ValueError(f"Release 60 arcsec normalization not satisfied: {path}")
        factor = _eef_factors(folder / eef_name)[0]
        data *= factor
        data.flags.writeable = False
        return cls(image=data, position_angle_deg=float(position_angle_deg), total_flux_factor=factor, path=str(path))

    def density(self, east_arcsec: np.ndarray, north_arcsec: np.ndarray) -> np.ndarray:
        """Flux per square arcsec at sky offsets (arcsec east, north); +Z is up, template x increases westward. Refuses
        offsets beyond the released +-100 arcsec (`PublishedPSF.density`)."""
        a = np.deg2rad(self.position_angle_deg)
        east0 = east_arcsec * np.cos(a) - north_arcsec * np.sin(a)
        north0 = east_arcsec * np.sin(a) + north_arcsec * np.cos(a)
        x, y = _CENTRE - east0, _CENTRE + north0
        if np.any((x < 0) | (x > 200) | (y < 0) | (y > 200)):
            raise ValueError("Requested support exceeds published PSF; wings must not be invented")
        return map_coordinates(self.image, [y, x], order=1, mode="nearest", prefilter=False)


def template_density(components, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """The mean over the scan components of the template's flux per square arcsec at sky offsets (arcsec; east,
    north); zero beyond the released +-100 arcsec (the mosaic PSF's and the zero-image check's unit source).
    Identity: `pacs_fit._template_density` (pacs_fit.py:1481)."""
    out = 0.0
    for c in components:
        a = np.deg2rad(c.position_angle_deg)
        east0 = east * np.cos(a) - north * np.sin(a)
        north0 = east * np.sin(a) + north * np.cos(a)
        out = out + map_coordinates(c.image, [100.0 + north0, 100.0 - east0], order=1, mode="constant", cval=0.0,
                                    prefilter=False)
    return out / len(components)


def _map_header(config, record):
    """The map's image header: the record's stored copy, else read from the file."""
    stored = getattr(record, "wcs_header", "")
    if stored:
        return fits.Header.fromstring(stored)
    path = Path(config.inputs["herschel_pacs_maps"]) / record.path
    with fits.open(path, mode="denywrite", memmap=True) as hdus:
        header = hdus[0].header.copy()
        header.extend(hdus[record.image_hdu].header, update=True, strip=False)
    return header


_SCANS = {}


def _scan_angles(config) -> dict:
    key = config.inputs["herschel_pacs_maps"]
    if key not in _SCANS:
        _SCANS[key] = _index_scan_angles(config)
    return _SCANS[key]


def _parallel_orientations(config, record, header) -> dict:
    """Every contributing observation's (direction, angle) of a parallel map: {'nominal'|'orthogonal': [angles]}. A
    Level 3 stack of several parallel pairs holds several angles per direction (Cepheus Flare: 235.7 and 208.4 deg).
    Fails if a direction or angle is missing or none is found."""
    scans = _scan_angles(config)
    contexts = [scans[str(o)] + (f"HSA metadata of {o}",) for o in record.obsids if str(o) in scans]
    orientations = {}
    for direction, angle, source in [(str(header.get("SCANDIR", "")).strip().lower(),
                                      float(header.get("POSANGLE", np.nan)), "map header")] + contexts:
        if direction not in ("nominal", "orthogonal") or not np.isfinite(angle):
            raise ValueError(f"Missing scan direction/position angle in {source}")
        angles = orientations.setdefault(direction, [])
        if not any(abs(angle - a) < 1e-6 for a in angles):
            angles.append(angle)
    if not orientations:
        raise ValueError("No scan direction found for this parallel map")
    return orientations


def _single_direction_tag(config, record) -> Optional[str]:
    """The release template tag ("+42" nominal, "-42" orthogonal) of a parallel map holding ONE scan direction, else
    None (a two-direction parallel map or a scan map)."""
    if _key(record)[1] == 1:
        return None
    orientations = _parallel_orientations(config, record, _map_header(config, record))
    if len(orientations) != 1:
        return None
    return dict(_DIRECTION_TAGS)[next(iter(orientations))]


def _direction_component(config, record, tag: str, angles) -> PublishedPSF:
    """One scan direction's template: the release image at the direction's one angle, or, for a stack whose
    observations of that direction were taken at several angles, the mean of the template rotated to each, in the frame
    of the first angle. Samples beyond the released +-100 arcsec are zero."""
    first = PublishedPSF.read(config, record, tag, position_angle_deg=angles[0])
    if len(angles) == 1:
        return first
    yy, xx = np.indices(first.image.shape, dtype=float)
    east0, north0 = _CENTRE - xx, yy - _CENTRE
    a0 = np.deg2rad(angles[0])
    east = east0 * np.cos(a0) + north0 * np.sin(a0)       # invert the first angle's rotation
    north = -east0 * np.sin(a0) + north0 * np.cos(a0)
    image = np.zeros_like(first.image)
    for angle in angles:
        a = np.deg2rad(angle)
        e = east * np.cos(a) - north * np.sin(a)
        n = east * np.sin(a) + north * np.cos(a)
        image += map_coordinates(first.image, [_CENTRE + n, _CENTRE - e], order=1, mode="constant", cval=0.0)
    image /= len(angles)
    image.flags.writeable = False
    return PublishedPSF(image=image, position_angle_deg=float(angles[0]), total_flux_factor=first.total_flux_factor,
                        path=first.path)


def published_components(config, record) -> List[PublishedPSF]:
    """The release templates of a map, each at the spacecraft position angle of its own scan: the +42/-42 pair, or for
    a parallel map holding one scan direction that direction's template alone. Parallel maps: the header's SCANDIR and
    POSANGLE and the HSA per-observation metadata (the download's tap_pacs_scan__<Region>.csv); nominal = +42,
    orthogonal = -42. Scan maps: both templates at the mean position angle of the contributing observations. A map
    PIXFRAC other than 0.1, a missing context or an ambiguous angle fails. Reads the map header (record) and the
    release. Identity: `pacs_psf.published_components_for_map` (pacs_psf.py:526); its order is the PSF group key's
    (mosaic.group_key)."""
    key = _key(record)
    header = _map_header(config, record)
    if not np.isclose(float(header.get("PIXFRAC", np.nan)), .1):
        raise ValueError("Published templates are for pixfrac=.1 maps")
    if key[1] == 1:
        scans = _scan_angles(config)
        contexts = [scans[str(o)] + (f"HSA metadata of {o}",) for o in record.obsids if str(o) in scans]
        angles = [a for _, a, _ in contexts] or [float(header.get("POSANGLE", np.nan))]
        angle = float(np.mean(angles))
        return tuple(PublishedPSF.read(config, record, a, position_angle_deg=angle) for a in _ARRAY_TO_MAP_ANGLES)
    orientations = _parallel_orientations(config, record, header)
    return tuple(_direction_component(config, record, tag, orientations[d])
                 for d, tag in _DIRECTION_TAGS if d in orientations)


# ----------------------------------------------------------------------------------------------------------------------
# the input maps of a mosaic

def good_pixels(image: np.ndarray, error: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Where an input map's pixel is usable: finite image and error, error > 0, coverage > 0 (`pacs_mosaic._good`)."""
    return np.isfinite(image) & np.isfinite(error) & (error > 0) & (coverage > 0)


_INPUT_LAYERS = {}
_SMOOTH_DENSE = {}


def _input_layers(path: str, hdus: Tuple[int, int, int]):
    """The (image, error, coverage) arrays of the input map at `path`, memory-mapped and kept open for the life of the
    process (valid_at reads single pixels of them). Truly read-only ("denywrite", shared pages): astropy's "readonly"
    maps copy-on-write, which a node with strict overcommit (Stampede3) charges in full per worker per map (65.7 GB over
    115 workers in job 3661155)."""
    if path not in _INPUT_LAYERS:
        h = fits.open(path, mode="denywrite", memmap=True)
        _INPUT_LAYERS[path] = (h, tuple(h[i].data for i in hdus))
    return _INPUT_LAYERS[path][1]


def _valid_at(layers, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Whether the four interpolation neighbours of input pixel (px, py) the resampling weights are good (a neighbour
    of bilinear weight zero not asked). `layers`: (image, error, coverage) arrays. Identity: `pacs_mosaic.valid_at`."""
    img, err, cov = layers
    ny, nx = img.shape
    xi = np.clip(np.floor(px).astype(np.int64), 0, max(nx - 2, 0))
    yi = np.clip(np.floor(py).astype(np.int64), 0, max(ny - 2, 0))
    fx, fy = np.clip(px - xi, 0, 1), np.clip(py - yi, 0, 1)
    valid = np.ones(px.shape, bool)
    for dy, by in ((0, 1 - fy), (1, fy)):
        for dx, bx in ((0, 1 - fx), (1, fx)):
            jj, ii = np.minimum(yi + dy, ny - 1), np.minimum(xi + dx, nx - 1)
            valid &= good_pixels(img[jj, ii], err[jj, ii], cov[jj, ii]) | (bx * by == 0)
    return valid


def _smooth_dense(directory: str) -> np.ndarray:
    """An input map's smoothed-error samples as one dense float64 array, kept per process (they are small:
    (ny / step) x (nx / step)); keyed by the store's index time, so a rebuilt store is read again."""
    key = (directory, os.stat(os.path.join(directory, "index.json")).st_mtime_ns)
    if key not in _SMOOTH_DENSE:
        for old in [k for k in _SMOOTH_DENSE if k[0] == directory]:
            del _SMOOTH_DENSE[old]
        _SMOOTH_DENSE[key] = TiledLayer.open(directory, "r").to_dense()
    return _SMOOTH_DENSE[key]


def _smooth_at(samples: np.ndarray, step: int, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """FROZEN 6. The smoothed error at input pixel (px, py): bilinear over the valid samples (NaN where none).
    Identity: `pacs_mosaic._lattice_at` (pacs_mosaic.py:244)."""
    nj, ni = samples.shape
    u, v = px / step, py / step
    i0 = np.clip(np.floor(u).astype(np.int64), 0, ni - 2) if ni > 1 else np.zeros(u.shape, np.int64)
    j0 = np.clip(np.floor(v).astype(np.int64), 0, nj - 2) if nj > 1 else np.zeros(v.shape, np.int64)
    fu, fv = np.clip(u - i0, 0, 1), np.clip(v - j0, 0, 1)
    num = np.zeros(px.shape)
    den = np.zeros(px.shape)
    for dj, bj in ((0, 1 - fv), (1, fv)):
        for di, bi in ((0, 1 - fu), (1, fu)):
            jj, ii = np.minimum(j0 + dj, nj - 1), np.minimum(i0 + di, ni - 1)
            node = samples[jj, ii]
            b = np.where(np.isfinite(node), bj * bi, 0.0)
            num += b * np.where(np.isfinite(node), node, 0.0)
            den += b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


@dataclass
class InputMap:
    """One input map of a mosaic: `record` (MapRecord), `wcs` (celestial, `grid.celestial_wcs(record.wcs_header)`),
    `maps_root` (config's herschel_pacs_maps directory), `smooth` (its smoothed error, TiledLayer float64 fill NaN, in
    INPUT pixels sampled every `smooth_step`), `smooth_step` (FROZEN 6: `grid.smooth_step_px(record.pixel_scale_arcsec)`).
    The FITS file is opened read-only ("denywrite", shared pages)."""
    record: object
    wcs: object
    maps_root: str
    smooth: Optional[TiledLayer] = None
    smooth_step: int = 0

    @property
    def path(self) -> str:
        """The FITS file's absolute path."""
        return os.path.join(self.maps_root, self.record.path)

    def layers(self, rows: slice, cols: slice) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """float64 copies of the map's (image, error, coverage) window rows x cols (`pacs_mosaic._read_layers`)."""
        rec = self.record
        with fits.open(self.path, mode="denywrite", memmap=True) as hdus:
            return tuple(np.array(hdus[i].data[rows, cols], dtype=np.float64)
                         for i in (rec.image_hdu, rec.error_hdu, rec.coverage_hdu))

    def valid_at(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Whether the mosaic uses the map at input pixel (px, py): the four interpolation neighbours good (a neighbour
        of bilinear weight zero not asked). Identity: `pacs_mosaic.valid_at` (pacs_mosaic.py:197)."""
        rec = self.record
        return _valid_at(_input_layers(self.path, (rec.image_hdu, rec.error_hdu, rec.coverage_hdu)), px, py)

    def smoothed_error_at(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """FROZEN 6. The smoothed error at input pixel (px, py) by bilinear interpolation over the valid samples (NaN
        where none). Identity: `pacs_mosaic._lattice_at` (pacs_mosaic.py:244), bit-identical."""
        return _smooth_at(_smooth_dense(self.smooth.directory), self.smooth_step, px, py)


def build_smoothed_error(input_map: InputMap, out_dir: str) -> TiledLayer:
    """FROZEN 6. The median of the map's valid error pixels in a square window of grid.SMOOTH_WINDOW_ARCSEC, sampled
    every window_px // 2 input pixels (sample (j, i) at pixel (i * step, j * step)); NaN where the window has no valid
    pixel. Reads the input FITS band by band (memory: one window of rows). Writes `out_dir` (TiledLayer float64, fill
    NaN). Identity: `pacs_mosaic._smooth_lattice` (pacs_mosaic.py:214): every sample bit-identical."""
    rec = input_map.record
    win = grid.smooth_window_px(rec.pixel_scale_arcsec)
    half, step = win // 2, grid.smooth_step_px(rec.pixel_scale_arcsec)
    ny, nx = rec.shape
    nj, ni = (ny - 1) // step + 2, (nx - 1) // step + 2
    samples = np.full((nj, ni), np.nan)
    with fits.open(input_map.path, mode="denywrite", memmap=True) as hdus:
        eh, ch, ih = hdus[rec.error_hdu].data, hdus[rec.coverage_hdu].data, hdus[rec.image_hdu].data
        for j in range(nj):
            r0, r1 = max(0, j * step - half), min(ny, j * step + half + 1)
            if r1 <= r0:
                continue
            err = np.array(eh[r0:r1], dtype=np.float64)
            err[~good_pixels(np.array(ih[r0:r1], dtype=np.float64), err, np.array(ch[r0:r1], dtype=np.float64))] = np.nan
            padded = np.pad(err, ((0, 0), (half, max(0, (ni - 1) * step + win - nx - half))), constant_values=np.nan)
            view = np.lib.stride_tricks.sliding_window_view(padded, (err.shape[0], win), axis=(0, 1))
            cut = view[0, ::step][:ni].reshape(ni, -1)
            ok = np.isfinite(cut).any(axis=1)
            row = np.full(ni, np.nan)
            row[ok] = np.nanmedian(cut[ok], axis=1)
            samples[j] = row
    return TiledLayer.from_dense(samples, out_dir, fill=np.nan)


# ----------------------------------------------------------------------------------------------------------------------
# the mosaic PSF (reader; mosaic.build_mosaic_psf writes it)

@njit(cache=True)
def _cached_responses(table, source_x, source_y, pixel_x, pixel_y, map_nx, map_ny, extent, steps):
    """The response of a (nodes, nodes, components, count, count) table at pixels around sources: bilinear between the
    spatial nodes (FROZEN 14: g = (s + 0.5) / n * (nodes - 1)) and between the sub-pixel samples (`pacs_psf.
    _cached_responses`)."""
    n_source, n_pixel = pixel_x.shape
    n_component = table.shape[2]
    out = np.empty((n_source, n_component, n_pixel), dtype=np.float64)
    spatial_ny, spatial_nx = table.shape[:2]
    last = table.shape[-1] - 1
    for row in range(n_source):
        sx, sy = source_x[row], source_y[row]
        if not (-.5 <= sx <= map_nx - .5 and -.5 <= sy <= map_ny - .5):
            raise ValueError('Source lies outside calibrated response-cache map')
        gx = (sx + .5) / map_nx * (spatial_nx - 1)
        gy = (sy + .5) / map_ny * (spatial_ny - 1)
        ix, iy = min(int(gx), spatial_nx - 2), min(int(gy), spatial_ny - 2)
        fx, fy = gx - ix, gy - iy
        for pixel in range(n_pixel):
            dx = (pixel_x[row, pixel] - sx + extent) * steps
            dy = (pixel_y[row, pixel] - sy + extent) * steps
            if not (0 <= dx <= last and 0 <= dy <= last):
                raise ValueError('Cutout exceeds response-cache displacement support')
            jx, jy = min(int(dx), last - 1), min(int(dy), last - 1)
            ux, uy = dx - jx, dy - jy
            for member in range(n_component):
                value = 0.0
                for cy in range(2):
                    wy = fy if cy else 1 - fy
                    for cx in range(2):
                        wx = fx if cx else 1 - fx
                        im = table[iy + cy, ix + cx, member]
                        local = ((1 - ux) * (1 - uy) * im[jy, jx] + ux * (1 - uy) * im[jy, jx + 1]
                                 + ux * uy * im[jy + 1, jx + 1] + (1 - ux) * uy * im[jy + 1, jx])
                        value += wx * wy * local
                out[row, member, pixel] = value
    return out


def psf_store(mosaic_dir: str) -> str:
    """The PSF store of a mosaic: <mosaic_dir>/psf."""
    return os.path.join(mosaic_dir, "psf")


class MosaicPSF:
    """The PSF of a mosaic: at an output position the weight-fraction mixture sum_k f_k P_k over the sub-groups of the
    input maps valid there, f from w_i = 1 / (smoothed error_i)^2 (the mosaic's own weights) summed per sub-group. P_k is
    sub-group k's template (mean of the scan columns) on the north-up output grid, convolved with the input pixel's box
    and the resampling's bilinear kernel, sampled at the output pixel centres and normalised to the release EEF inside
    the aperture radius. `table` (the weight-mean over the mosaic) serves the detection kernel; `evaluate_xy` is the
    exact local mixture (the injection template); `enclosed_at` / `aperture_fraction_at` the mixture of the sub-groups'
    enclosed fraction (no sky) and aperture fraction (after the annulus sky), interpolated between the spatial nodes;
    `beam_at` the mixture of the sub-groups' beams. A map counts exactly where the mosaic uses it (`InputMap.valid_at`).
    `maps_root`: the input-map directory at read time (psf.json stores no absolute path). Read from `<mosaic>/psf/`; the
    tables memory-mapped. Picklable by (directory, maps_root, tables). Identity: `pacs_psf.MosaicPSF`
    (pacs_psf.py:627-793), every method bit-identical. Attributes: directory, maps_root, meta (psf.json), shape, extent,
    subpixel_steps, integration_order, n_sub, table_half, out_wcs, fbar, enclosed, aperture_fraction_sub, sub_beams,
    subs and table (with `tables`)."""

    def __init__(self, directory: str, maps_root: str, tables: bool = True) -> None:
        """Opens `<directory>/psf/` (psf.json; with `tables` the sub<k>.npy and table.npy memory-mapped) and each input
        map's smoothed error."""
        self.directory = directory
        pdir = psf_store(directory)
        with open(os.path.join(pdir, "psf.json")) as fh:
            self.meta = json.load(fh)
        m = self.meta
        self.shape = tuple(m["shape"])
        self.extent, self.subpixel_steps, self.integration_order = m["extent"], m["subpixel_steps"], m["integration_order"]
        self.n_sub = m["n_sub"]
        self.table_half = m.get("table_half")
        self.out_wcs = grid.celestial_wcs(m["out_header"])
        self._maps = []
        for q in m["maps"]:
            self._maps.append(dict(wcs=grid.celestial_wcs(q["wcs_header"]), shape=tuple(q["shape"]),
                                   samples=_smooth_dense(os.path.join(directory, q["smooth"])), step=q["smooth_step"],
                                   scale=q["scale"], sub=q["sub"], bbox=q["bbox"], path=q["path"], hdus=tuple(q["hdus"])))
        self.maps_root = str(maps_root)
        self.fbar = np.array(m["fbar"])
        self.enclosed = np.array(m["enclosed_sub"])                          # (n_sub, nodes, nodes)
        self.aperture_fraction_sub = np.array(m["aperture_fraction_sub"])
        self.sub_beams = np.array(m["sub_beams"])
        if tables:
            self.subs = [np.load(os.path.join(pdir, f"sub{k}.npy"), mmap_mode="r") for k in range(self.n_sub)]
            self.table = np.load(os.path.join(pdir, "table.npy"), mmap_mode="r")

    def __getstate__(self):
        return {"directory": self.directory, "maps_root": self.maps_root, "tables": hasattr(self, "table")}

    def __setstate__(self, state):
        self.__init__(state["directory"], state["maps_root"], state["tables"])

    def weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(n, n_sub) mixture fractions at output positions; a position no map covers gets fbar
        (`MosaicPSF.weights`, pacs_psf.py:679)."""
        sx, sy = np.asarray(x, float), np.asarray(y, float)
        key = (sx.tobytes(), sy.tobytes())
        cache = self.__dict__.setdefault("_weights_kept", OrderedDict())
        if key in cache:
            cache.move_to_end(key)
            return cache[key].copy()
        w = self.map_weights(sx, sy)
        total = w.sum(axis=1)
        f = np.where(total[:, None] > 0, w / np.where(total > 0, total, 1.0)[:, None], self.fbar[None, :])
        cache[key] = f.copy()
        if len(cache) > WEIGHTS_KEPT:
            cache.popitem(last=False)
        return f

    def map_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(n, n_sub) the summed weights 1/(smoothed error)^2 of each sub-group's input maps the mosaic uses at the
        output positions, unnormalised (zero where none of the sub-group's maps contributes)."""
        sx, sy = np.asarray(x, float), np.asarray(y, float)
        w = np.zeros((sx.size, self.n_sub))
        for q in self._maps:
            x0, x1, y0, y1 = q["bbox"]
            idx = np.flatnonzero((sx >= x0 - 1) & (sx <= x1) & (sy >= y0 - 1) & (sy <= y1))
            if not idx.size:
                continue
            ra, dec = self.out_wcs.all_pix2world(sx[idx], sy[idx], 0)
            px, py = q["wcs"].all_world2pix(ra, dec, 0)
            ny, nx = q["shape"]
            inside = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (px <= nx - 1) & (py >= 0) & (py <= ny - 1)
            s = _smooth_at(q["samples"], q["step"], np.where(inside, px, 0.0), np.where(inside, py, 0.0)) * q["scale"]
            ok = inside & np.isfinite(s) & (s > 0)
            at = np.flatnonzero(ok)
            if at.size:
                ok[at] = _valid_at(_input_layers(os.path.join(self.maps_root, q["path"]), q["hdus"]), px[at], py[at])
            w[idx[ok], q["sub"]] += 1.0 / s[ok] ** 2
        return w

    def _node_values(self, nodes: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(n, n_sub) of the per-sub-group node values `nodes` (n_sub, N, N) at the positions, bilinear between the
        spatial nodes with the weights `_cached_responses` interpolates the tables by."""
        n = nodes.shape[-1]
        ix, fx = grid.psf_node_cell(grid.psf_node_coord(x, self.shape[1], n, clip=True), n)
        iy, fy = grid.psf_node_cell(grid.psf_node_coord(y, self.shape[0], n, clip=True), n)
        v =(nodes[:, iy, ix] * (1 - fx) * (1 - fy) + nodes[:, iy, ix + 1] * fx * (1 - fy)
             + nodes[:, iy + 1, ix] * (1 - fx) * fy + nodes[:, iy + 1, ix + 1] * fx * fy)
        return v.T

    def evaluate_xy(self, source_x: np.ndarray, source_y: np.ndarray, pixel_x: np.ndarray, pixel_y: np.ndarray,
                    mixture: Optional[np.ndarray] = None) -> np.ndarray:
        """(n, 1, ...) the local-mixture response at the pixels around each source; `mixture` (n, n_sub): the
        sub-group fractions to use in place of the local `weights` (a one-hot row gives one sub-group alone)."""
        sx, sy = np.asarray(source_x, dtype=float), np.asarray(source_y, dtype=float)
        x, y = np.broadcast_arrays(np.asarray(pixel_x, dtype=float), np.asarray(pixel_y, dtype=float))
        if (sx.ndim != 1 or sy.shape != sx.shape or x.ndim != 3 or x.shape[0] != sx.size
                or not all(np.all(np.isfinite(a)) for a in (sx, sy, x, y))):
            raise ValueError("Expected finite source vectors and (n,ny,nx) pixel coordinates")
        if not sx.size:
            return np.empty((0, 1, *x.shape[1:]), dtype=np.float64)
        key = None
        if mixture is None:
            # the same injections' template is asked for twice (pasted into the image, then applied to the detection
            # map): a small last-recently-used memory of the patches, bit-identical to computing them again
            key = (sx.tobytes(), sy.tobytes(), x.shape, np.ascontiguousarray(x).tobytes(), np.ascontiguousarray(y).tobytes())
            kept = self.__dict__.setdefault("_patches_kept", OrderedDict())
            if key in kept:
                kept.move_to_end(key)
                return kept[key].copy()
        f = self.weights(sx, sy) if mixture is None else mixture
        xr, yr = x.reshape(sx.size, -1), y.reshape(sx.size, -1)
        out = np.zeros((sx.size, 1, xr.shape[1]))
        for k in range(self.n_sub):
            if f[:, k].any():
                out += f[:, k, None, None] * _cached_responses(self.subs[k], sx, sy, xr, yr, self.shape[1],
                                                               self.shape[0], self.extent, self.subpixel_steps)
        out = out.reshape((sx.size, 1, *x.shape[1:]))
        if key is not None:
            kept[key] = out.copy()
            if len(kept) > PATCHES_KEPT:
                kept.popitem(last=False)
        return out

    def evaluate_mean_xy(self, source_x: np.ndarray, source_y: np.ndarray, pixel_x: np.ndarray,
                         pixel_y: np.ndarray) -> np.ndarray:
        """The weight-mean table's response (one component) -- the detection kernel's."""
        sx, sy = np.asarray(source_x, dtype=float), np.asarray(source_y, dtype=float)
        x, y = np.broadcast_arrays(np.asarray(pixel_x, dtype=float), np.asarray(pixel_y, dtype=float))
        out = _cached_responses(self.table, sx, sy, x.reshape(sx.size, -1), y.reshape(sx.size, -1), self.shape[1],
                                self.shape[0], self.extent, self.subpixel_steps)
        return out.reshape((sx.size, self.table.shape[2], *x.shape[1:]))

    def enclosed_at(self, x: np.ndarray, y: np.ndarray, mixture: Optional[np.ndarray] = None) -> np.ndarray:
        """The mixture's fraction of the total flux inside the aperture radius (no sky) at output positions."""
        f = self.weights(x, y) if mixture is None else mixture
        return (f * self._node_values(self.enclosed, x, y)).sum(axis=1)

    def aperture_fraction_at(self, x: np.ndarray, y: np.ndarray, mixture: Optional[np.ndarray] = None) -> np.ndarray:
        """The mixture's aperture fraction after the annulus sky at output positions."""
        f = self.weights(x, y) if mixture is None else mixture
        return (f * self._node_values(self.aperture_fraction_sub, x, y)).sum(axis=1)

    def beam_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The mixture's beam FWHM (arcsec) at output positions: the weights times the sub-groups' beams."""
        return self.weights(x, y) @ self.sub_beams

    def beam_at_sky(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        """`beam_at` at sky positions (a position off the grid gets the mosaic's mean fractions' beam)."""
        x, y = self.out_wcs.all_world2pix(np.asarray(ra_deg, float), np.asarray(dec_deg, float), 0)
        return self.beam_at(np.where(np.isfinite(x), x, -1e9), np.where(np.isfinite(y), y, -1e9))


def mosaic_scalar(mosaic_dir: str, name: str) -> float:
    """A weight-mean scalar of the mosaic PSF (beam_arcsec, aperture_fraction, enclosed_fraction) from psf.json."""
    with open(os.path.join(psf_store(mosaic_dir), "psf.json")) as fh:
        return float(json.load(fh)[name])
