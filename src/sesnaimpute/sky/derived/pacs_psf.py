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
`published_components_for_map(config, record)` returns that pair;
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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from numba import njit
from scipy.ndimage import map_coordinates

# The PACS team's release (Calzoletti, ESAC, 16 May 2017, HERSCHEL-HSC-DOC-2181
# "PACS Photometer PSFs and EEFs as Ancillary Data Products: Release note"),
# distributing the PSFs and EEFs of PICC-ME-TN-033 v2.2 (Bocchio, Lutz et al.
# 2016, "PACS photometer point spread function", 16 Nov 2015; the technical
# report bolopsf_22.pdf). Templates: Vesta, OD345 (array-to-map angles +42 and
# -42), non-recentered, projected with pixfrac 0.1 (the Standard Pipeline value,
# which the maps carry as PIXFRAC). Calibration context: Balog et al. 2014,
# Exp. Astron. 37, 129. Files in PSF_Data/ and EEF_Data/ under
RELEASE_URL = "https://archives.esac.esa.int/hsa/legacy/ADP/PSF/PACS/PACS-P/"

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
    `_COMBINATIONS`.
    """
    return float(_COMBINATIONS[_key(record)][3])


def psf_directory(config):
    """Where the templates live: <herschel_pacs_maps>/calibration/psf."""
    return Path(config.inputs["herschel_pacs_maps"]) / "calibration" / "psf"


def total_flux_factor(eef_path):
    """The factor a that turns a release image into total flux (see module
    docstring): least squares of the EEF (column 3, total-flux fraction) on
    the Vesta cumulative (column 5) and pi r^2, over the radii <= 60 arcsec.
    """
    rows = [line.split() for line in Path(eef_path).read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    table = np.array([[float(v) for v in row] for row in rows if len(row) == 5])
    radius, eef, vesta = table[:, 0], table[:, 2], table[:, 4]
    design = np.column_stack([vesta, np.pi * radius ** 2])
    (factor, _), *_ = np.linalg.lstsq(design, eef, rcond=None)
    return float(factor)


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
        if not np.isclose(data[inside].sum(), 1., atol=2e-4):
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

    Output is (n_position, n_component, ny, nx), without mixture weights or
    renormalization: each pixel holds the template's flux integrated over the
    pixel by Gauss-Legendre quadrature on the pixel's corner-interpolated sky
    offsets.
    """
    components: tuple

    def __post_init__(self):
        if len(self.components) != 2 or not all(isinstance(c, PublishedPSF)
                                               for c in self.components):
            raise ValueError('PairedPSF requires two published components')

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
        return out.reshape((x.shape[0], 2, x.shape[1], x.shape[2]))


@njit(cache=True)
def _cached_responses(table, source_x, source_y, pixel_x, pixel_y,
                      map_nx, map_ny, extent, steps):
    n_source, n_pixel = pixel_x.shape
    out = np.empty((n_source, 2, n_pixel), dtype=np.float64)
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
            for member in range(2):
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
        nbytes = spatial_nodes ** 2 * 2 * count ** 2 * np.dtype(dtype).itemsize
        if nbytes > 1024 ** 3:
            raise ValueError('Response table exceeds 1 GiB budget')
        offsets = np.linspace(-extent, extent, count)
        yy, xx = np.meshgrid(offsets, offsets, indexing='ij')
        table = np.empty((spatial_nodes, spatial_nodes, 2, count, count), dtype=dtype)
        psf = PairedPSF(tuple(components))
        for iy, sy in enumerate(np.linspace(-.5, shape[0] - .5, spatial_nodes)):
            for ix, sx in enumerate(np.linspace(-.5, shape[1] - .5, spatial_nodes)):
                source = wcs.pixel_to_world(np.array([sx]), np.array([sy]))[:, None, None]
                table[iy, ix] = psf(wcs, source, (xx + sx)[None], (yy + sy)[None],
                                    integration_order)[0]
        table.flags.writeable = False
        return cls(table, tuple(shape), extent, subpixel_steps, integration_order)

    def evaluate_xy(self, source_x, source_y, pixel_x, pixel_y):
        sx, sy = np.asarray(source_x, dtype=float), np.asarray(source_y, dtype=float)
        x, y = np.broadcast_arrays(np.asarray(pixel_x, dtype=float), np.asarray(pixel_y, dtype=float))
        if (sx.ndim != 1 or sy.shape != sx.shape or x.ndim != 3 or x.shape[0] != sx.size
                or not all(np.all(np.isfinite(a)) for a in (sx, sy, x, y))):
            raise ValueError('Expected finite source vectors and (n,ny,nx) pixel coordinates')
        if not sx.size:
            return np.empty((0, 2, *x.shape[1:]), dtype=np.float64)
        out = _cached_responses(self.table, sx, sy, x.reshape(sx.size, -1), y.reshape(sx.size, -1),
                                self.shape[1], self.shape[0], self.extent, self.subpixel_steps)
        return out.reshape((sx.size, 2, *x.shape[1:]))

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
    with fits.open(path, mode="readonly", memmap=True) as hdus:
        header = hdus[0].header.copy()
        header.extend(hdus[record.image_hdu].header, update=True, strip=False)
    return header


def published_components_for_map(config, record):
    """The two release templates (+42 and -42 scan) for this map, each placed
    at the spacecraft position angle of its own scan.

    Parallel maps: the map header gives one SCANDIR (nominal/orthogonal) and
    POSANGLE, the retained observation contexts (maps_root/<obsid>/hpacs*obs*)
    the other; nominal is the +42 template, orthogonal the -42. Scan-map
    maps carry no scan-direction label: both templates take the mean position
    angle of the contributing observations. A mismatch of the map's PIXFRAC
    with the templates' 0.1, a missing context, or an ambiguous angle fails
    explicitly.
    """
    key = _key(record)
    header = _map_header(config, record)
    if not np.isclose(float(header.get('PIXFRAC', np.nan)), .1):
        raise ValueError('Published templates are for pixfrac=.1 maps')
    root = Path(config.inputs["herschel_pacs_maps"])
    contexts = []
    for obsid in record.obsids:
        for path in sorted((root / str(obsid)).glob(f'hpacs{obsid}obs*.fits.gz')):
            with fits.open(path, mode='readonly') as hdus:
                h = hdus[0].header
                if str(h.get('OBS_ID', h.get('OBSID', ''))) != str(obsid):
                    raise ValueError(f'Observation context ID differs from indexed scan: {path}')
                contexts.append((str(h.get('SCANDIR', '')).strip().lower(),
                                 float(h.get('POSANGLE', np.nan)), path))
    if key[1] == 1:
        angles = [a for _, a, _ in contexts] or [float(header.get('POSANGLE', np.nan))]
        angle = float(np.mean(angles))
        return tuple(PublishedPSF.read(config, record, a, position_angle_deg=angle)
                     for a in _ARRAY_TO_MAP_ANGLES)
    orientations = {}
    for direction, angle, source in [(str(header.get('SCANDIR', '')).strip().lower(),
                                      float(header.get('POSANGLE', np.nan)), 'map header')] + contexts:
        if direction not in ('nominal', 'orthogonal') or not np.isfinite(angle):
            raise ValueError(f'Missing scan direction/position angle in {source}')
        if direction in orientations and abs(orientations[direction] - angle) > 1e-8:
            raise ValueError(f'Ambiguous {direction} scan orientation in retained metadata')
        orientations[direction] = angle
    if set(orientations) != {'nominal', 'orthogonal'}:
        raise ValueError('Both original nominal/orthogonal orientations must be retained')
    return tuple(PublishedPSF.read(config, record, a, position_angle_deg=orientations[d])
                 for d, a in (('nominal', '+42'), ('orthogonal', '-42')))
