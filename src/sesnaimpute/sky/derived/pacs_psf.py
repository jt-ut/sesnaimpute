"""The PACS point-spread functions the far-infrared photometry fits (SPEC_PACS
section 4 and 11.4): the PACS team's Vesta templates for P70 (blue1, 70 um) and
P100 (blue2, 100 um, "grn" in the release's names), for parallel mode (31) at
20 and 60 arcsec/s and scan-map mode (1) at 20 and 60 arcsec/s, each with its
two scan angles, normalised to total flux.

This is a library the photometry and completeness stages import, not a stage:
it builds no product and reads the templates from
`<herschel_pacs_maps>/calibration/psf/` (placed there once by hand from the
release below; derived stages run without network).

What a map needs is a pair of templates, one per scan direction, each placed
on the sky at the spacecraft position angle of the observation that made it.
A parallel map that holds only one scan direction (Taurus: two Level 2 maps of one
observation each) is fitted with that direction's template alone, at its own
position angle, and its aperture fraction comes from that template's own cumulative V(r) with the
band's a and c (P70 and P160; a one-direction P100 map raises, the two P100 templates differing by
about 3 per cent inside 8 arcsec).
`published_components_for_map(config, record)` returns that pair (or the one);
`CachedPairedPSF.from_map(...)` integrates the pair over the map's pixels on a
grid of source positions and pixel offsets once per map, and
`CachedPairedPSF.evaluate_xy(...)` interpolates it for any set of sources. The
two scan columns are fitted unconstrained and their SUM is the flux estimator;
no scan-weight mixture is assumed.

Normalisation. The release images are 201 x 201 at 1 arcsec, flux-normalised so
the signal inside r = 60 arcsec is 1 after the mean of the 61-70 arcsec annulus
is subtracted everywhere (release note HERSCHEL-HSC-DOC-2181 section 2.2;
PICC-ME-TN-033 section 9: "background at r = 61-70 arcsec ... total signal
inside r = 60 arcsec is normalized"). That subtracts the same constant c per
unit area from the true PSF, so the true cumulative energy is
EEF(r) = a V(r) + c pi r^2 with V the image's own cumulative sum. The release's
EEF tables carry both: column 3 is the encircled fraction of the total out to
1000 arcsec (pixfrac 0.1) and column 5 is V(r) out to 60 arcsec. A least
squares of column 3 on column 5 and pi r^2 over the 100 radii closes to 1e-12,
which gives a = 0.911 (blue) and 0.906 (green), the SAME for every speed and
mode of that band (only the inner EEF shape differs between files), and c =
2.188e-6 / 1.998e-6 per square arcsec. Scaling each image by a makes its
amplitude the total flux; the removed pedestal c is absorbed by the fitted
local background. This confirms the old code's 0.911 for P70 parallel 60"/s
(it came from EEF_blu_60para_21.txt) and extends it: P100 is 0.906.
"""

import dataclasses
import json
from collections import OrderedDict
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from numba import njit
from scipy.ndimage import map_coordinates

from sesnaimpute.sky.download.herschel_pacs.index import MapRecord
from sesnaimpute.definitions import PACS_APERTURES as APERTURES

# The PACS team's release (Calzoletti, ESAC, 16 May 2017, HERSCHEL-HSC-DOC-2181
# "PACS Photometer PSFs and EEFs as Ancillary Data Products: Release note"),
# distributing the PSFs and EEFs of PICC-ME-TN-033 v2.2 (Bocchio, Lutz et al.
# 2016, "PACS photometer point spread function", 16 Nov 2015; the technical
# report bolopsf_22.pdf). Templates: Vesta, OD345 (array-to-map angles +42 and
# -42), non-recentered, projected with pixfrac 0.1 (the Standard Pipeline value,
# which the maps carry as PIXFRAC). Calibration context: Balog et al. 2014,
# Exp. Astron. 37, 129. Files in PSF_Data/ and EEF_Data/ under
# https://archives.esac.esa.int/hsa/legacy/ADP/PSF/PACS/PACS-P/

# (band, mode, speed arcsec/s) -> the release's speed tag and EEF file, and the
# beam FWHM (geometric mean of the two fitted axes, arcsec) that sets the blend
# radius, the fitting box and the extended-source threshold. Axes (minor x
# major, arcsec) and their source, PICC-ME-TN-033 v2.2:
#   parallel (mode 31): Table 6, the equal-weight sum of the +42 and -42 scans
#     (the beam of a finished map):
#       P70  60"/s 8.80x9.60   20"/s 5.74x6.26
#       P100 60"/s 9.73x10.69  20"/s 6.98x7.42
#   scan map (mode 1): Table 4 (OD160, prime mode, one scan direction; Table 6
#     has no prime-mode equal-weight sum, so a finished two-direction map is
#     3-4 per cent broader than this; the fast prime beams are the elongated
#     5.70x9.05 and 6.84x9.81):
#       P70  20"/s 5.41x5.72   60"/s 5.70x9.05
#       P100 20"/s 6.66x6.89   60"/s 6.84x9.81
_COMBINATIONS = {
    ("P70", 31, 60): ("blu", "60para", "EEF_blu_60para_21.txt", np.sqrt(8.80 * 9.60)),
    ("P70", 31, 20): ("blu", "20para", "EEF_blu_20para_21.txt", np.sqrt(5.74 * 6.26)),
    ("P100", 31, 60): ("grn", "60para", "EEF_grn_60para_21.txt", np.sqrt(9.73 * 10.69)),
    ("P100", 31, 20): ("grn", "20para", "EEF_grn_20para_21.txt", np.sqrt(6.98 * 7.42)),
    ("P70", 1, 20): ("blu", "20", "EEF_blu_20.txt", np.sqrt(5.41 * 5.72)),
    ("P70", 1, 60): ("blu", "60", "EEF_blu_60_22.txt", np.sqrt(5.70 * 9.05)),
    ("P100", 1, 20): ("grn", "20", "EEF_grn_20.txt", np.sqrt(6.66 * 6.89)),
    ("P100", 1, 60): ("grn", "60", "EEF_grn_60_22.txt", np.sqrt(6.84 * 9.81)),
    # 160 um (red): the release has red templates for the scan speeds only (20, 60 "/s) and EEF tables for
    # red 20, 20para and 60para; parallel maps use the scan template at their speed with the parallel EEF
    # table, scan-map 60 "/s uses the 60para EEF table. The reported 160 um flux does not depend on these
    # (it is eHOPS's aperture with their constant); they enter the blend fit, the bright-peak finder and
    # the extended flag. Beams: Gaussian fits to the release templates (+42/-42 mean), 2026-10-27.
    ("P160", 31, 60): ("red", "60", "EEF_red_60para_21.txt", 11.4),
    ("P160", 31, 20): ("red", "20", "EEF_red_20para_21.txt", 10.3),
    ("P160", 1, 20): ("red", "20", "EEF_red_20.txt", 10.3),
    ("P160", 1, 60): ("red", "60", "EEF_red_60para_21.txt", 11.4),
}

# The two array-to-map angles of the OD345 templates; scan-map observations
# use scan angles of 0/45/135 degrees, and the +-42 pair is the nearest the
# release offers (the 20"/s prime beam is round to 6 per cent, so the choice
# moves a template by a few per cent; at 60"/s it is an approximation).
_ARRAY_TO_MAP_ANGLES = ("+42", "-42")

# The release image: 201 x 201, 1 arcsec pixels, centred on pixel (101, 101).
_SIZE = 201
_CENTRE = 100
_NORMALISATION_RADIUS = 60.0


def _key(record):
    key = (str(record.band), int(record.mode), int(round(float(record.scan_speed))))
    if key not in _COMBINATIONS:
        raise ValueError(f"No PACS PSF release for band/mode/speed {key}; "
                         f"supported: {sorted(_COMBINATIONS)}")
    return key


def beam_fwhm_arcsec(record):
    """The beam FWHM (arcsec) for this map's band, mode and scan speed.

    The geometric mean of the two fitted Gaussian axes of the combined beam
    (fast-scan beams are elongated, e.g. P70 parallel 60"/s is 8.80 x 9.60
    arcsec, 9.19 as returned); the axes and their tables are in the comment on
    `_COMBINATIONS`. A mosaic record: the mosaic's weight-mean beam.
    """
    if getattr(record, "mosaic_dir", ""):
        return mosaic_scalar(record, "beam_arcsec")
    return float(_COMBINATIONS[_key(record)][3])


def psf_directory(config):
    """Where the templates live: <herschel_pacs_maps>/calibration/psf."""
    return Path(config.inputs["herschel_pacs_maps"]) / "calibration" / "psf"


# The eHOPS apertures: definitions.PACS_APERTURES (imported above as APERTURES).


def _eef_columns(eef_path):
    rows = [line.split() for line in Path(eef_path).read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    table = np.array([[float(v) for v in row] for row in rows if len(row) == 5])
    return table[:, 0], table[:, 2], table[:, 4]       # radius, total-flux EEF, 60-arcsec-normalised EEF


def aperture_fraction(config, record):
    """The aperture correction the measurement divides by (owner ruling 2026-10-27: one canonical flux,
    the TOTAL flux of a point source, the PACS calibration's own definition): the fraction of a point
    source's total flux the eHOPS aperture returns after the annulus sky, from the release EEF table of
    this map's band, mode and speed (e.g. P70 parallel-60 0.712, scan-20 0.756; P100 0.545 / 0.632)."""
    if getattr(record, "mosaic_dir", ""):
        return mosaic_scalar(record, "aperture_fraction")
    single = _single_direction_tag(config, record)
    if single is not None:
        if record.band == "P100":
            raise ValueError("the +42 and -42 P100 templates differ by about 3 percent inside 8 arcsec, so the "
                             "single-template aperture fraction is unverified at that band")
        return _single_template_fraction(config, record, single)
    return physical_aperture_fraction(config, record, "total", True)


def aperture_eef(config, record):
    """The fraction of a point source's total flux inside the eHOPS aperture radius, with no sky
    subtraction, from the same release EEF table (or one-direction template) `aperture_fraction` reads:
    the normalisation the map's point-spread table is held to (`pacs_fit.Tile`), so that a PSF fit's
    amplitude, an injected artificial source and the aperture photometry all mean the same total flux."""
    if getattr(record, "mosaic_dir", ""):
        return mosaic_scalar(record, "enclosed_fraction")
    single = _single_direction_tag(config, record)
    if single is not None:
        if record.band == "P100":
            raise ValueError("the +42 and -42 P100 templates differ by about 3 percent inside 8 arcsec, so the "
                             "single-template aperture fraction is unverified at that band")
        return _single_template_fraction(config, record, single, annulus_share=False)
    return physical_aperture_fraction(config, record, "total", False)


def _single_template_fraction(config, record, tag, annulus_share=True):
    """The aperture fraction of a one-direction parallel map: the band's a and c (the EEF least squares
    of total_flux_factor) applied to this template's OWN cumulative V(r) (the release image summed within
    r on its 1-arcsec grid, bilinear on an 8x finer sampling so sub-pixel radii are smooth),
    EEF(r) = a V(r) + c pi r^2, then the eHOPS aperture and annulus share exactly as
    physical_aperture_fraction does."""
    r_ap, r_in, r_out = APERTURES[record.band]
    eef_path = psf_directory(config) / _COMBINATIONS[_key(record)][2]
    a, c = _eef_factors(eef_path)
    template = PublishedPSF.read(config, record, tag, position_angle_deg=0.0)
    raw = template.image / template.total_flux_factor
    n = 8
    grid = np.linspace(-.5 + .5 / n, 2 * _CENTRE + .5 - .5 / n, _SIZE * n)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    fine = map_coordinates(raw, [yy, xx], order=1, mode="nearest", prefilter=False)
    rr = np.hypot(xx - _CENTRE, yy - _CENTRE)
    ee = lambda r: a * float(fine[rr <= r].sum()) / n ** 2 + c * np.pi * r ** 2
    frac = ee(r_ap)
    if annulus_share:
        frac -= r_ap ** 2 / (r_out ** 2 - r_in ** 2) * (ee(r_out) - ee(r_in))
    return frac


def physical_aperture_fraction(config, record, normalisation, annulus_share):
    """For the record only (the study table, spec section 12): the fraction of a point source's flux the
    eHOPS aperture returns for this map's band, mode and speed from the release EEF table, on the "total"
    or "60arcsec" normalisation, with or without the PSF wing the sky annulus removes."""
    r_ap, r_in, r_out = APERTURES[record.band]
    radius, eef_total, eef_60 = _eef_columns(psf_directory(config) / _COMBINATIONS[_key(record)][2])
    eef = eef_total if normalisation == "total" else eef_60
    ee = lambda r: float(np.interp(np.log(r), np.log(radius), eef))
    frac = ee(r_ap)
    if annulus_share:
        frac -= r_ap ** 2 / (r_out ** 2 - r_in ** 2) * (ee(r_out) - ee(r_in))
    return frac


def total_flux_factor(eef_path):
    """The factor a that turns a release image into total flux (see module
    docstring): least squares of the EEF (column 3, total-flux fraction) on
    the Vesta cumulative (column 5) and pi r^2, over the radii <= 60 arcsec.
    """
    return _eef_factors(eef_path)[0]


def _eef_factors(eef_path):
    """(a, c) of the same least squares: EEF(r) = a V(r) + c pi r^2 (band-level, see module docstring)."""
    rows = [line.split() for line in Path(eef_path).read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    table = np.array([[float(v) for v in row] for row in rows if len(row) == 5])
    radius, eef, vesta = table[:, 0], table[:, 2], table[:, 4]
    design = np.column_stack([vesta, np.pi * radius ** 2])
    (factor, pedestal), *_ = np.linalg.lstsq(design, eef, rcond=None)
    return float(factor), float(pedestal)


@dataclass(frozen=True)
class PublishedPSF:
    image: np.ndarray
    path: str
    position_angle_deg: float
    total_flux_factor: float = float("nan")

    @classmethod
    def read(cls, config, record, array_to_map_angle, *, position_angle_deg):
        """One release component ("+42" or "-42") of this map's combination,
        scaled to total flux, placed at the given spacecraft position angle."""
        band_tag, speed_tag, eef_name, _ = _COMBINATIONS[_key(record)]
        folder = psf_directory(config)
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
        factor = total_flux_factor(folder / eef_name)
        data *= factor
        data.flags.writeable = False
        return cls(data, str(path), float(position_angle_deg), factor)

    def density(self, east_arcsec, north_arcsec):
        """Flux per square arcsec; +Z is up, template x increases westward."""
        a = np.deg2rad(self.position_angle_deg)
        east0 = east_arcsec * np.cos(a) - north_arcsec * np.sin(a)
        north0 = east_arcsec * np.sin(a) + north_arcsec * np.cos(a)
        x, y = _CENTRE - east0, _CENTRE + north0
        if np.any((x < 0) | (x > 200) | (y < 0) | (y > 200)):
            raise ValueError('Requested support exceeds published PSF; wings must not be invented')
        return map_coordinates(self.image, [y, x], order=1, mode='nearest', prefilter=False)


@njit(cache=True)
def _paired_quadrature(east, north, images, cosine, sine, nodes, weights):
    """Fused pixel integration; no quadrature-sized temporary image arrays."""
    n_position, n_pixel = east.shape[1:]
    out = np.zeros((n_position, images.shape[0], n_pixel), dtype=np.float64)
    for row in range(n_position):
        for pixel in range(n_pixel):
            area = 0.0
            for corner in range(4):
                nxt = (corner + 1) % 4
                area += (east[corner, row, pixel] * north[nxt, row, pixel]
                         - north[corner, row, pixel] * east[nxt, row, pixel])
            area = 0.5 * abs(area)
            for iu in range(nodes.size):
                u = nodes[iu]
                for iv in range(nodes.size):
                    v = nodes[iv]
                    coeff = ((.5 - u) * (.5 - v), (.5 + u) * (.5 - v),
                             (.5 + u) * (.5 + v), (.5 - u) * (.5 + v))
                    e, n = 0.0, 0.0
                    for corner in range(4):
                        e += coeff[corner] * east[corner, row, pixel]
                        n += coeff[corner] * north[corner, row, pixel]
                    weight = weights[iu] * weights[iv] * area
                    for member in range(images.shape[0]):
                        x = 100.0 - (e * cosine[member] - n * sine[member])
                        y = 100.0 + (e * sine[member] + n * cosine[member])
                        if x < 0 or x > 200 or y < 0 or y > 200:
                            raise ValueError('Requested support exceeds published PSF')
                        ix, iy = min(int(x), 199), min(int(y), 199)
                        fx, fy = x - ix, y - iy
                        value = ((1 - fx) * (1 - fy) * images[member, iy, ix]
                                 + fx * (1 - fy) * images[member, iy, ix + 1]
                                 + fx * fy * images[member, iy + 1, ix + 1]
                                 + (1 - fx) * fy * images[member, iy + 1, ix])
                        out[row, member, pixel] += weight * value
    return out


@dataclass(frozen=True)
class PairedPSF:
    """Separate published component columns with shared quadrature geometry.

    Output is (n_position, n_component, ny, nx), n_component 1 or 2, without mixture weights or
    renormalization: each pixel holds the template's flux integrated over the
    pixel by Gauss-Legendre quadrature on the pixel's corner-interpolated sky
    offsets.
    """
    components: tuple

    def __post_init__(self):
        if len(self.components) not in (1, 2) or not all(isinstance(c, PublishedPSF)
                                               for c in self.components):
            raise ValueError('PairedPSF requires one or two published components')

    def __call__(self, wcs, source, pixel_x, pixel_y, integration_order):
        if not isinstance(integration_order, int) or integration_order < 1:
            raise ValueError('integration_order must be a positive integer')
        x, y = np.broadcast_arrays(np.asarray(pixel_x, float), np.asarray(pixel_y, float))
        if x.ndim != 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError('Pixel coordinates must be finite (n_position,ny,nx) arrays')
        east, north = [], []
        for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)):
            e, n = source.spherical_offsets_to(wcs.pixel_to_world(x + dx, y + dy))
            east.append(e.arcsec)
            north.append(n.arcsec)
        shape = (4, x.shape[0], x.shape[1] * x.shape[2])
        east = np.ascontiguousarray(east).reshape(shape)
        north = np.ascontiguousarray(north).reshape(shape)
        nodes, weights = np.polynomial.legendre.leggauss(integration_order)
        angles = np.deg2rad([c.position_angle_deg for c in self.components])
        images = np.stack([c.image for c in self.components])
        out = _paired_quadrature(east, north, images, np.cos(angles), np.sin(angles),
                                 nodes / 2, weights / 2)
        return out.reshape((x.shape[0], len(self.components), x.shape[1], x.shape[2]))


@njit(cache=True)
def _cached_responses(table, source_x, source_y, pixel_x, pixel_y,
                      map_nx, map_ny, extent, steps):
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


@dataclass(frozen=True)
class CachedPairedPSF:
    """Map-specific interpolation of separately integrated published components.

    The table holds the exact PairedPSF quadrature on a deterministic grid of
    source positions and pixel offsets; interpolation never rescales template
    sums. Memory is table.nbytes (construction refuses tables above 1 GiB).
    float32 halves the table (99 to 49 MB on Musca) and changes no pixel by
    more than 3e-8 of the peak. Fewer source nodes would build faster (3 x 3:
    8 s, 18 MB) but moved Musca's 3.2 arcsec pixels by 3e-3 of the peak, so
    the default stays 5 x 5; a 1.6 arcsec scan map moved only 1.6e-5.
    """
    table: np.ndarray
    shape: tuple
    extent: float
    subpixel_steps: int
    integration_order: int

    @classmethod
    def from_map(cls, wcs, shape, components, *, half_size=15, integration_order=8,
                 spatial_nodes=5, subpixel_steps=16, dtype=np.float32):
        for value in (half_size, integration_order, spatial_nodes, subpixel_steps):
            if not isinstance(value, int) or value < 1:
                raise ValueError('Cache geometry must use positive integer dimensions')
        if spatial_nodes < 2 or len(shape) != 2 or min(shape) <= 0:
            raise ValueError('Cache requires at least two spatial nodes and a positive map shape')
        extent = half_size + .5
        count = int(2 * extent * subpixel_steps) + 1
        nbytes = spatial_nodes ** 2 * len(components) * count ** 2 * np.dtype(dtype).itemsize
        if nbytes > 1024 ** 3:
            raise ValueError('Response table exceeds 1 GiB budget')
        offsets = np.linspace(-extent, extent, count)
        yy, xx = np.meshgrid(offsets, offsets, indexing='ij')
        table = np.empty((spatial_nodes, spatial_nodes, len(components), count, count), dtype=dtype)
        psf = PairedPSF(tuple(components))
        for iy, sy in enumerate(np.linspace(-.5, shape[0] - .5, spatial_nodes)):
            for ix, sx in enumerate(np.linspace(-.5, shape[1] - .5, spatial_nodes)):
                source = wcs.pixel_to_world(np.array([sx]), np.array([sy]))[:, None, None]
                table[iy, ix] = psf(wcs, source, (xx + sx)[None], (yy + sy)[None],
                                    integration_order)[0]
        table.flags.writeable = False
        return cls(table, tuple(shape), extent, subpixel_steps, integration_order)

    def scaled(self, factor):
        """The same cache with every response multiplied by `factor` (a normalisation, `pacs_fit.Tile`)."""
        table = (self.table * self.table.dtype.type(factor)).astype(self.table.dtype)
        table.flags.writeable = False
        return dataclasses.replace(self, table=table)

    def evaluate_xy(self, source_x, source_y, pixel_x, pixel_y):
        sx, sy = np.asarray(source_x, dtype=float), np.asarray(source_y, dtype=float)
        x, y = np.broadcast_arrays(np.asarray(pixel_x, dtype=float), np.asarray(pixel_y, dtype=float))
        if (sx.ndim != 1 or sy.shape != sx.shape or x.ndim != 3 or x.shape[0] != sx.size
                or not all(np.all(np.isfinite(a)) for a in (sx, sy, x, y))):
            raise ValueError('Expected finite source vectors and (n,ny,nx) pixel coordinates')
        if not sx.size:
            return np.empty((0, self.table.shape[2], *x.shape[1:]), dtype=np.float64)
        out = _cached_responses(self.table, sx, sy, x.reshape(sx.size, -1), y.reshape(sx.size, -1),
                                self.shape[1], self.shape[0], self.extent, self.subpixel_steps)
        return out.reshape((sx.size, self.table.shape[2], *x.shape[1:]))

    def __call__(self, wcs, source, pixel_x, pixel_y, integration_order):
        if integration_order != self.integration_order:
            raise ValueError('Requested quadrature differs from the response-cache definition')
        sx, sy = wcs.world_to_pixel(source)
        return self.evaluate_xy(np.asarray(sx).reshape(-1), np.asarray(sy).reshape(-1),
                                pixel_x, pixel_y)


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


def _scan_angles(config):
    key = config.inputs["herschel_pacs_maps"]
    if key not in _SCANS:
        from sesnaimpute.sky.download.herschel_pacs.index import scan_angles
        _SCANS[key] = scan_angles(config)
    return _SCANS[key]


def _parallel_orientations(config, record, header):
    """Every contributing observation's (direction, angle) of a parallel map: {'nominal'|'orthogonal':
    [angles]}. A Level 3 stack of several parallel pairs holds several angles per direction (Cepheus
    Flare: 235.7 and 208.4 deg). Fails if a direction or angle is missing or none is found."""
    scans = _scan_angles(config)
    contexts = [scans[str(o)] + (f"HSA metadata of {o}",) for o in record.obsids if str(o) in scans]
    orientations = {}
    for direction, angle, source in [(str(header.get('SCANDIR', '')).strip().lower(),
                                      float(header.get('POSANGLE', np.nan)), 'map header')] + contexts:
        if direction not in ('nominal', 'orthogonal') or not np.isfinite(angle):
            raise ValueError(f'Missing scan direction/position angle in {source}')
        angles = orientations.setdefault(direction, [])
        if not any(abs(angle - a) < 1e-6 for a in angles):
            angles.append(angle)
    if not orientations:
        raise ValueError('No scan direction found for this parallel map')
    return orientations


_DIRECTION_TAGS = (('nominal', '+42'), ('orthogonal', '-42'))


def _single_direction_tag(config, record):
    """The release template tag ("+42" nominal, "-42" orthogonal) when this is a parallel map holding
    ONE scan direction, else None (a two-direction parallel map or a scan map)."""
    if _key(record)[1] == 1:
        return None
    orientations = _parallel_orientations(config, record, _map_header(config, record))
    if len(orientations) != 1:
        return None
    return dict(_DIRECTION_TAGS)[next(iter(orientations))]


def published_components_for_map(config, record):
    """The release templates for this map, each placed at the spacecraft position angle of its own
    scan: the +42 and -42 pair, or for a parallel map holding only one scan direction (nominal or
    orthogonal) that direction's template alone (one-tuple).

    Parallel maps: the map header gives one SCANDIR (nominal/orthogonal) and
    POSANGLE, the HSA's per-observation metadata (the download's cached
    tap_pacs_scan__<Region>.csv) both; nominal is the +42 template, orthogonal the -42. Scan-map
    maps carry no scan-direction label: both templates take the mean position
    angle of the contributing observations. A mismatch of the map's PIXFRAC
    with the templates' 0.1, a missing context, or an ambiguous angle fails
    explicitly.
    """
    key = _key(record)
    header = _map_header(config, record)
    if not np.isclose(float(header.get('PIXFRAC', np.nan)), .1):
        raise ValueError('Published templates are for pixfrac=.1 maps')
    if key[1] == 1:
        # each contributing observation's scan direction and angle, from the download's cached HSA
        # metadata table (the IRSA map header names only one scan of a parallel pair)
        scans = _scan_angles(config)
        contexts = [scans[str(o)] + (f"HSA metadata of {o}",) for o in record.obsids if str(o) in scans]
        angles = [a for _, a, _ in contexts] or [float(header.get('POSANGLE', np.nan))]
        angle = float(np.mean(angles))
        return tuple(PublishedPSF.read(config, record, a, position_angle_deg=angle)
                     for a in _ARRAY_TO_MAP_ANGLES)
    orientations = _parallel_orientations(config, record, header)
    return tuple(_direction_component(config, record, tag, orientations[d])
                 for d, tag in _DIRECTION_TAGS if d in orientations)


def _direction_component(config, record, tag, angles):
    """One scan direction's template: the release image at the direction's one angle, or, for a stack
    whose observations of that direction were taken at several angles, the mean of the template rotated
    to each, expressed in the frame of the first angle (the map is the mean of its observations; their
    coverages are about equal). Samples beyond the released +-100 arcsec are zero."""
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
        image += map_coordinates(first.image, [_CENTRE + n, _CENTRE - e], order=1, mode='constant', cval=0.0)
    image /= len(angles)
    image.flags.writeable = False
    return PublishedPSF(image, first.path, float(angles[0]), first.total_flux_factor)


# ----------------------------------------------------------------------------
# the mosaic's point-spread function (briefs/pacs_mosaic_2026-10-29/BRIEF.md sections 6-8)

@dataclass(frozen=True)
class MosaicRecord(MapRecord):
    """The map record of one band's mosaic of a region (`pacs_fit.mosaic_record`): `Tile` opens it like any map.
    `path` is a unique key (the product's folder + "/mosaic", relative to the data root, so `pacs_fit.prepared_dir`
    is unique); `mosaic_dir` is the product folder."""
    mosaic_dir: str = ""


_MOSAIC_SCALARS = {}
_INPUT_LAYERS = {}


def mosaic_scalar(record, name):
    """A weight-mean scalar of the mosaic PSF (beam_arcsec, aperture_fraction, enclosed_fraction) from its
    `_psf/psf.json`; read again when the file has been rebuilt (keyed by its modification time)."""
    path = os.path.join(record.mosaic_dir, "_psf", "psf.json")
    key = (path, os.stat(path).st_mtime_ns)
    if key not in _MOSAIC_SCALARS:
        for old in [k for k in _MOSAIC_SCALARS if k[0] == path]:
            del _MOSAIC_SCALARS[old]
        with open(path) as fh:
            _MOSAIC_SCALARS[key] = json.load(fh)
    return float(_MOSAIC_SCALARS[key][name])


def _input_layers(root, q):
    """The (image, error, coverage) arrays of input map `q` (a `psf.json` map entry), memory-mapped and kept open
    for the life of the process: `MosaicPSF.weights` reads single pixels of them. Truly read-only ("denywrite", shared
    pages): astropy's "readonly" maps copy-on-write, which a node with strict overcommit (Stampede3) charges in full
    per worker per map (65.7 GB over 115 workers in job 3661155)."""
    path = os.path.join(root, q["path"])
    if path not in _INPUT_LAYERS:
        hdus = fits.open(path, mode="denywrite", memmap=True)
        _INPUT_LAYERS[path] = (hdus, tuple(hdus[i].data for i in q["hdus"]))
    return _INPUT_LAYERS[path][1]


#: per-position memories of a MosaicPSF (last recently used): mixture fractions, and response patches (256 sources
#: of 47 x 47 pixels, 4.5 MB, each)
WEIGHTS_KEPT = 8
PATCHES_KEPT = 4


class MosaicPSF:
    """The point-spread function of a mosaic: at an output position the weight-fraction mixture over the input
    maps valid there, sum_i f_i P_i, with f_i = w_i / sum w and w_i = 1 / (smoothed error_i)^2 (the mosaic's own
    weights, evaluated from each map's stored smoothed-error lattice; no per-pixel layer is stored). P_i is map i's
    published template pair (mean of the scan columns, as `Tile` injects it) expressed on the north-up output grid,
    convolved with the input pixel's box and the bilinear interpolation kernel of the resampling (a separable
    triangle of half-width one INPUT pixel along the input axes, so a quadratic B-spline once the box is in),
    sampled at the output pixel centres, and normalised exactly as `Tile` normalises an ordinary map's table
    (total-flux EEF inside the aperture radius). Maps of one template group, pixel scale and orientation share a
    sub-group; each sub-group's response table lives in `<mosaic>/_psf/sub<k>.npy`, memory-mapped.

    `table` (the weight-mean over the mosaic, one component) with `extent`, `subpixel_steps` serves the
    fitting core as for `CachedPairedPSF`; `evaluate_xy` is the exact local mixture; `enclosed_at` and
    `aperture_fraction_at` the same mixture of the sub-groups' enclosed fraction (no sky) and aperture fraction
    (after the annulus sky), each measured on every spatial node's table and interpolated between the nodes as the
    tables are (a fixed pixel radius covers less sky off the tangent point); `beam_at` the mixture of the sub-groups'
    beam FWHM. A map counts in the mixture exactly where the mosaic uses it (`pacs_mosaic.valid_at`).
    `maps_root`: the input-map directory, taken from the config by the caller at read time (`psf.json` stores no
    absolute path, so a PSF built on another machine reads here).
    Build with `pacs_fit.build_mosaic_psf`."""

    def __init__(self, directory, maps_root, tables=True):
        from astropy.wcs import WCS
        self.directory = directory
        pdir = os.path.join(directory, "_psf")
        with open(os.path.join(pdir, "psf.json")) as fh:
            self.meta = json.load(fh)
        m = self.meta
        self.shape = tuple(m["shape"])
        self.extent, self.subpixel_steps, self.integration_order = m["extent"], m["subpixel_steps"], m["integration_order"]
        self.n_sub = m["n_sub"]
        self.out_wcs = WCS(fits.Header.fromstring(m["out_header"])).celestial
        self._maps = []
        for q in m["maps"]:
            self._maps.append(dict(wcs=WCS(fits.Header.fromstring(q["wcs_header"])).celestial, shape=tuple(q["shape"]),
                                   lat=np.load(os.path.join(directory, q["lattice"])), step=q["lattice_step"],
                                   scale=q["scale"], sub=q["sub"], bbox=q["bbox"], path=q["path"], hdus=q["hdus"]))
        self.maps_root = str(maps_root)                       # the config's input-map root at read time
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

    def weights(self, source_x, source_y):
        """(n, n_sub) mixture fractions f at output pixel positions (zero-origin x, y): the weight 1/(smoothed
        error)^2 of every input map the mosaic uses there (inside its footprint, a finite smoothed error and the four
        interpolation neighbours good: `pacs_mosaic.resample_block`'s validity, read from single pixels of the input
        map), summed per sub-group and normalised; a position no map covers gets the mosaic's mean fractions."""
        sx, sy = np.asarray(source_x, float), np.asarray(source_y, float)
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

    def map_weights(self, source_x, source_y):
        """(n, n_sub) the summed weights 1/(smoothed error)^2 of each sub-group's input maps the mosaic uses at the
        positions, unnormalised (zero where none of the sub-group's maps contributes; `weights` normalises them)."""
        from sesnaimpute.sky.derived.pacs_mosaic import _lattice_at, valid_at
        sx, sy = np.asarray(source_x, float), np.asarray(source_y, float)
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
            lat = _lattice_at(q["lat"], q["step"], np.where(inside, px, 0.0), np.where(inside, py, 0.0)) * q["scale"]
            ok = inside & np.isfinite(lat) & (lat > 0)
            at = np.flatnonzero(ok)
            if at.size:
                ok[at] = valid_at(_input_layers(self.maps_root, q), px[at], py[at])
            w[idx[ok], q["sub"]] += 1.0 / lat[ok] ** 2
        return w

    def _node_values(self, nodes, source_x, source_y):
        """(n, n_sub) of the per-sub-group node values `nodes` (n_sub, N, N) at the positions, bilinear between the
        spatial nodes with the weights `_cached_responses` interpolates the tables by."""
        sx, sy = np.asarray(source_x, float), np.asarray(source_y, float)
        n = nodes.shape[-1]
        gx = np.clip((sx + .5) / self.shape[1] * (n - 1), 0.0, n - 1.0)
        gy = np.clip((sy + .5) / self.shape[0] * (n - 1), 0.0, n - 1.0)
        ix, iy = np.minimum(gx.astype(np.int64), n - 2), np.minimum(gy.astype(np.int64), n - 2)
        fx, fy = gx - ix, gy - iy
        v = (nodes[:, iy, ix] * (1 - fx) * (1 - fy) + nodes[:, iy, ix + 1] * fx * (1 - fy)
             + nodes[:, iy + 1, ix] * (1 - fx) * fy + nodes[:, iy + 1, ix + 1] * fx * fy)
        return v.T

    def evaluate_xy(self, source_x, source_y, pixel_x, pixel_y, mixture=None):
        """The response at the pixels around each source; `mixture` (n, n_sub): the sub-group fractions to use in
        place of the local `weights` (a one-hot row gives one sub-group's table alone)."""
        sx, sy = np.asarray(source_x, dtype=float), np.asarray(source_y, dtype=float)
        x, y = np.broadcast_arrays(np.asarray(pixel_x, dtype=float), np.asarray(pixel_y, dtype=float))
        if (sx.ndim != 1 or sy.shape != sx.shape or x.ndim != 3 or x.shape[0] != sx.size
                or not all(np.all(np.isfinite(a)) for a in (sx, sy, x, y))):
            raise ValueError('Expected finite source vectors and (n,ny,nx) pixel coordinates')
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

    def evaluate_mean_xy(self, source_x, source_y, pixel_x, pixel_y):
        """The response of the weight-mean table `table` (one component) as `CachedPairedPSF.evaluate_xy`."""
        sx, sy = np.asarray(source_x, dtype=float), np.asarray(source_y, dtype=float)
        x, y = np.broadcast_arrays(np.asarray(pixel_x, dtype=float), np.asarray(pixel_y, dtype=float))
        out = _cached_responses(self.table, sx, sy, x.reshape(sx.size, -1), y.reshape(sx.size, -1), self.shape[1],
                                self.shape[0], self.extent, self.subpixel_steps)
        return out.reshape((sx.size, self.table.shape[2], *x.shape[1:]))

    def enclosed_at(self, source_x, source_y, mixture=None):
        """The mixture's enclosed fraction inside the aperture radius (no sky) at each position (`mixture`: the
        fractions `weights` returns there, when the caller has them)."""
        f = self.weights(source_x, source_y) if mixture is None else mixture
        return (f * self._node_values(self.enclosed, source_x, source_y)).sum(axis=1)

    def aperture_fraction_at(self, source_x, source_y, mixture=None):
        """The mixture's aperture fraction (after the annulus sky) at each position."""
        f = self.weights(source_x, source_y) if mixture is None else mixture
        return (f * self._node_values(self.aperture_fraction_sub, source_x, source_y)).sum(axis=1)

    def beam_at(self, source_x, source_y):
        """The mixture's beam FWHM (arcsec) at each output pixel position: the weights times the sub-groups' beams."""
        return self.weights(source_x, source_y) @ self.sub_beams

    def beam_at_sky(self, ra_deg, dec_deg):
        """`beam_at` at sky positions (a position off the grid gets the mosaic's mean fractions' beam)."""
        x, y = self.out_wcs.all_world2pix(np.asarray(ra_deg, float), np.asarray(dec_deg, float), 0)
        return self.beam_at(np.where(np.isfinite(x), x, -1e9), np.where(np.isfinite(y), y, -1e9))
