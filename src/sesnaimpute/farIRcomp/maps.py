"""Local PACS map records and read-only, memory-mapped access."""
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
import json
import re

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from sesnaimpute import config as config_module


@dataclass(frozen=True)
class MapRecord:
    map_id: str
    obsids: tuple
    band: str
    path: str
    image_hdu: int
    error_hdu: int
    coverage_hdu: int
    shape: tuple
    unit: str
    wcs_header: str
    mode: int
    level: str
    mapper: str
    scan_speed: float = float('nan')
    scan_angles: tuple = ()
    position_angle: float = float('nan')
    polygon: str = ''
    error_unit: str = ''
    error_kind: str = 'unknown'
    scan_angle_frame: str = 'unknown'


def raw_root(config):
    return checked_path(Path(config.data_root), 'sky/download/herschel_pacs')


def checked_path(root, relative):
    """Resolve inside root; reject traversal and symlinked output ancestors."""
    root = Path(root).absolute()
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Expected a relative path without traversal')
    for node in (root, *root.parents, path, *path.parents):
        if node.is_symlink():
            raise ValueError(f'Symlink is not permitted in PACS write path: {node}')
    path.resolve().relative_to(root.resolve())
    return path


def index_path(config, region):
    path = Path(config_module.product_path(config, 'sky/derived', 'pacs',
                                          'index', 'region', region))
    return checked_path(Path(config.data_root), str(path.relative_to(config.data_root)))


def write_index(config, region, records):
    """Write column datasets. Parent supplies the registry's attribute entries."""
    from sesnaimpute.attrs_registry import REGISTRY
    path = index_path(config, region)
    rows = sorted(records, key=lambda r: r.map_id)
    if len({r.map_id for r in rows}) != len(rows):
        raise ValueError('Duplicate logical map identity in regional index')
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(MapRecord)]
    attrs = {name: REGISTRY[('index_pacs_region', name.upper())] for name in names}
    with h5py.File(path, 'w') as out:
        for name in names:
            values = [getattr(r, name) for r in rows]
            if name in ('image_hdu', 'error_hdu', 'coverage_hdu', 'mode'):
                data = np.asarray(values, dtype=np.int32)
            elif name in ('scan_speed', 'position_angle'):
                data = np.asarray(values, dtype=np.float64)
            else:
                if name in ('obsids', 'shape', 'scan_angles'):
                    values = [json.dumps(v) for v in values]
                data = np.asarray(values, dtype=h5py.string_dtype('utf-8'))
            ds = out.create_dataset(name.upper(), data=data)
            ds.attrs['UNITS'], ds.attrs['READING'] = attrs[name]
    return path


def read_index(config, region):
    records = []
    with h5py.File(index_path(config, region), 'r') as src:
        for i in range(len(src['MAP_ID'])):
            row = {}
            for f in fields(MapRecord):
                v = src[f.name.upper()][i]
                if isinstance(v, bytes):
                    v = v.decode('utf-8')
                if f.name in ('obsids', 'shape', 'scan_angles'):
                    v = tuple(json.loads(v))
                row[f.name] = v
            records.append(MapRecord(**row))
    return sorted(records, key=lambda r: r.map_id)


@dataclass
class OpenMap:
    image: np.ndarray
    error: np.ndarray
    coverage: np.ndarray
    wcs: WCS
    record: MapRecord


_WCS_KEY = re.compile(r'^(CTYPE[12]|CUNIT[12]|CRPIX[12]|CRVAL[12]|CDELT[12]|CROTA[12]|'
                      r'(CD|PC)[12]_[12]|PV[12]_\d+|RADESYS|EQUINOX|LONPOLE|LATPOLE|'
                      r'[AB]P?_ORDER|[AB]P?_\d+_\d+)$')


def _check_layers(hdus, indices, science_header, shape):
    """Check storage and optional layer WCS before touching any image data."""
    reference = WCS(science_header).celestial
    ny, nx = shape
    points = np.array([[0., 0.], [nx - 1., 0.], [0., ny - 1.],
                       [nx - 1., ny - 1.], [(nx - 1.) / 2, (ny - 1.) / 2]])
    expected = reference.pixel_to_world(points[:, 0], points[:, 1]).icrs
    for index in indices:
        hdu = hdus[index]
        if isinstance(hdu, fits.CompImageHDU):
            raise ValueError('Internally compressed FITS layers cannot be memory mapped')
        header = hdu.header
        if header.get('NAXIS') != 2 or (header.get('NAXIS2'), header.get('NAXIS1')) != tuple(shape):
            raise ValueError('Map layers have inconsistent dimensions')
        # Herschel may store WCS only on science; absent auxiliary WCS inherits it.
        explicit = [key for key in header if _WCS_KEY.match(key)]
        if explicit:
            combined = science_header.copy()
            for key in explicit:
                combined[key] = header[key]
            actual = WCS(combined).celestial.pixel_to_world(points[:, 0], points[:, 1]).icrs
            separation = actual.separation(expected).deg
            if not np.all(np.isfinite(separation)) or np.max(separation) > 1e-8:
                raise ValueError(f'Layer {hdu.name} WCS is not aligned with science')


@contextmanager
def open_map(config, record):
    """Open once; arrays are borrowed views valid only inside the context."""
    path = checked_path(raw_root(config), record.path)
    if path.suffix.lower() in ('.gz', '.zip'):
        raise ValueError('Compressed maps must be unpacked before memory-mapped access')
    with fits.open(path, mode='readonly', memmap=True) as hdus:
        science_header = fits.Header.fromstring(record.wcs_header)
        _check_layers(hdus, (record.image_hdu, record.error_hdu, record.coverage_hdu),
                      science_header, record.shape)
        arrays = [hdus[i].data for i in (record.image_hdu, record.error_hdu,
                                       record.coverage_hdu)]
        if any(a is None or a.shape != tuple(record.shape) for a in arrays):
            raise ValueError(f'Inconsistent map layers: {path}')
        for array in arrays:
            array.flags.writeable = False
        yield OpenMap(*arrays, WCS(fits.Header.fromstring(record.wcs_header)).celestial, record)


def inspect_map(config, path, *, obsids, mode, level, polygon=''):
    """Inspect an uncompressed FITS map, rejecting unknown band/layer semantics.

    Header metadata is preserved in the record; scan angles and position angle
    are not interchangeable, and missing orientation is deliberately explicit.
    """
    path = Path(path)
    relative = str(path.resolve().relative_to(raw_root(config).resolve()))
    with fits.open(path, mode='readonly', memmap=True) as hdus:
        by_name = {h.name.lower(): i for i, h in enumerate(hdus)}
        if 'error' not in by_name and 'stdev' in by_name:
            stdev = hdus[by_name['stdev']].header
            if str(stdev.get('META_1', '')).strip().lower() == (
                    'the standard deviation of the pixel mean flux.'):
                by_name['error'] = by_name['stdev']
        if not all(n in by_name for n in ('image', 'error', 'coverage')):
            raise ValueError(f'Missing image/error/coverage layers: {path}')
        ih, eh, ch = (by_name[n] for n in ('image', 'error', 'coverage'))
        header = hdus[0].header.copy()
        header.extend(hdus[ih].header, update=True, strip=False)
        band_value = str(header.get('BLUEBAND', header.get('FILTER', header.get('BAND', '')))).lower()
        wavelength = header.get('WAVELNTH')
        if wavelength in (70, 100):
            band_value = str(int(wavelength))
        if band_value in ('blue1', 'bs', 'blu', '70'):
            band = 'P70'
        elif band_value in ('blue2', 'bl', 'green', 'grn', '100'):
            band = 'P100'
        else:
            raise ValueError(f'Unknown PACS filter {band_value!r}: {path}')
        unit = str(header.get('QTTY____', header.get('BUNIT', '')))
        error_header = hdus[eh].header
        error_unit = str(error_header.get('BUNIT', error_header.get('QTTY____', '')))
        # An extension called ERROR is not evidence for standard-deviation semantics.
        error_kind = 'unknown'
        if (hdus[eh].name.lower() == 'stdev' and
                str(error_header.get('META_1', '')).strip().lower() ==
                'the standard deviation of the pixel mean flux.'):
            error_kind = 'standard_deviation_of_mean'
        shape = (int(header['NAXIS2']), int(header['NAXIS1']))
        _check_layers(hdus, (ih, eh, ch), header, shape)
        map_id = str(header.get('FILENAME', '')).strip()
        if not map_id:
            raise ValueError('Map lacks a stable archive FILENAME identity')
        contributing = tuple(str(header[k]) for k in header
                             if k.startswith('OBSID') and k[5:].isdigit())
        return MapRecord(map_id, contributing or tuple(str(v) for v in obsids), band, relative,
                         ih, eh, ch, shape, unit, header.tostring(), mode, level,
                         str(header.get('MAPPER', 'unknown')),
                         scan_speed=float(header.get('SCANVELO', float('nan'))),
                         position_angle=float(header.get('POSANGLE', float('nan'))),
                         polygon=polygon,
                         error_unit=error_unit, error_kind=error_kind)
