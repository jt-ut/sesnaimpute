"""The Herschel PACS download index: one record per blue-channel map on disk.

`build` (this package) fetches the maps; this module reads what is ON DISK
and says which maps serve each region. A map is described by the header it
carries (band from BLUEBAND / WAVELNTH, observing mode, scan speed, the
observations merged into it, pixel scale, which extension is the image,
the noise and the coverage), never by archive catalogue membership.

`read_index(config, region)` returns the region's `MapRecord`s, read from
`<herschel_pacs_maps>/index_herschel_pacs__<Region>.csv`; a region with no
PACS coverage has an index with a header and no rows. `maps_root(config)`
is the maps directory, `[inputs] herschel_pacs_maps`, no fallback. The
downstream stages (pacs_fit, pacs_completeness, pacs_photometry) read maps
only through these two functions and the record.

Per-observation footprints come from the archive's cached TAP response
(`tap_pacs_obs__<Region>.csv`, written by `build`); `region_observations`
keeps those whose footprint polygon touches one of the region's admitted
nside-512 pixels.
"""

import csv
import dataclasses
import glob
import json
import os
import re

import h5py
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.path import Path as MplPath

from sesnaimpute.config import product_path

NSIDE = 512
# PACS observing modes of the archive: 1 = scan map, 31 = SPIRE/PACS parallel.
SCAN_MODE, PARALLEL_MODE = 1, 31


@dataclasses.dataclass(frozen=True)
class MapRecord:
    map_id: str            # the archive's own FILENAME of the map
    obsids: tuple          # every observation merged into the map
    band: str              # "P70", "P100" or "P160", from WAVELNTH / the product name / BLUEBAND
    path: str              # relative to maps_root
    image_hdu: int
    error_hdu: int         # the documented stDev layer, else an "error" layer
    coverage_hdu: int
    shape: tuple           # (ny, nx)
    unit: str              # image unit, Jy/pixel
    wcs_header: str        # primary + image header cards (80-char cards): WCS, PIXFRAC, POSANGLE, SCANDIR...
    mode: int              # 1 scan map, 31 parallel
    level: str             # "L20", "L25" or "L30"
    mapper: str            # "JScanam", "Unimap" or "HPF"
    scan_speed: float = float("nan")      # arcsec/s
    scan_angles: tuple = ()
    position_angle: float = float("nan")
    polygon: tuple = ()    # map extent corners, ((ra, dec), ...) ICRS degrees
    error_unit: str = ""
    error_kind: str = "unknown"   # "standard_deviation_of_mean" when documented
    scan_angle_frame: str = "unknown"
    pixel_scale_arcsec: float = float("nan")
    source: str = ""       # "IRSA" or "HSA", where the bytes came from


_TUPLE_FIELDS = ("obsids", "shape", "scan_angles", "polygon")
_FLOAT_FIELDS = ("scan_speed", "position_angle", "pixel_scale_arcsec")
_INT_FIELDS = ("image_hdu", "error_hdu", "coverage_hdu", "mode")


def maps_root(config):
    """The maps directory: `[inputs] herschel_pacs_maps`, no fallback."""
    return config.inputs["herschel_pacs_maps"]


def safe_name(region):
    """The region name as it appears in a download file name."""
    return re.sub(r"[^A-Za-z0-9+-]+", "_", region)


def index_path(config, region):
    return f"{maps_root(config)}/index_herschel_pacs__{safe_name(region)}.csv"


def tap_path(config, region):
    return f"{maps_root(config)}/tap_pacs_obs__{safe_name(region)}.csv"


def scan_path(config, region):
    return f"{maps_root(config)}/tap_pacs_scan__{safe_name(region)}.csv"


def scan_angles(config):
    """{obsid: (scan direction, position angle deg)} from every region's cached scan table."""
    out = {}
    for path in sorted(glob.glob(f"{maps_root(config)}/tap_pacs_scan__*.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["scandir"] and row["posangle"]:
                    out[row["observation_id"]] = (row["scandir"].strip().lower(), float(row["posangle"]))
    return out


def write_index(config, region, records):
    """Writes the region's index; an empty record list writes the header only."""
    names = [f.name for f in dataclasses.fields(MapRecord)]
    path = index_path(config, region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(names)
        for rec in sorted(records, key=lambda r: r.map_id):
            writer.writerow([json.dumps(getattr(rec, n)) if n in _TUPLE_FIELDS
                             else getattr(rec, n) for n in names])
    return path


def read_index(config, region):
    """The region's `MapRecord`s (possibly none), ordered by map_id."""
    out = []
    with open(index_path(config, region), newline="") as src:
        for row in csv.DictReader(src):
            for n in _TUPLE_FIELDS:
                row[n] = _tuplify(json.loads(row[n]))
            for n in _FLOAT_FIELDS:
                row[n] = float(row[n])
            for n in _INT_FIELDS:
                row[n] = int(row[n])
            out.append(MapRecord(**row))
    return sorted(out, key=lambda r: r.map_id)


def _tuplify(x):
    return tuple(_tuplify(v) for v in x) if isinstance(x, list) else x


_WCS_KEY = re.compile(r"^(NAXIS[12]?|CTYPE[12]|CUNIT[12]|CRPIX[12]|CRVAL[12]|CDELT[12]|"
                      r"CROTA[12]|(CD|PC)[12]_[12]|PV[12]_\d+|RADESYS|EQUINOX|LONPOLE|LATPOLE)$")


def inspect_map(path, root, source=""):
    """Reads a map's header into a `MapRecord`; raises ValueError when the
    band is neither 70 nor 100 um or the image/coverage/noise layers are
    missing. The noise layer is the documented stDev (standard deviation of
    the pixel mean flux) when the file has one; an undocumented "error"
    extension is used only when there is no stDev (the HPF L2 map of
    1342188101 carries both, and its "error" is not the pixel-mean noise).
    """
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    with fits.open(path, mode="readonly", memmap=True) as hdus:
        names = {h.name.lower(): i for i, h in enumerate(hdus)}
        if not all(n in names for n in ("image", "coverage")):
            raise ValueError(f"no image/coverage layers: {path}")
        ih, ch = names["image"], names["coverage"]
        eh, kind = None, "unknown"
        if "stdev" in names:
            eh = names["stdev"]
            # Level 3 files name the layer stDev without its META_1 description
            kind = "standard_deviation_of_mean"
        elif "error" in names:
            eh = names["error"]
        if eh is None:
            raise ValueError(f"no noise layer: {path}")
        head0, himg = hdus[0].header, hdus[ih].header
        wave = head0.get("WAVELNTH")
        blue = str(head0.get("BLUEBAND", "")).lower()
        if wave == 160 or re.search(r"MAPR", os.path.basename(path)):
            band = "P160"                          # the red camera (its header also carries BLUEBAND)
        elif wave in (70, 100):
            band = f"P{int(wave)}"
        elif blue == "blue1":
            band = "P70"
        elif blue == "blue2":
            band = "P100"
        else:
            raise ValueError(f"unknown PACS blue band {blue!r} / {wave!r}: {path}")
        omode = str(head0.get("OBS_MODE", "")).lower()
        mode = SCAN_MODE if omode.startswith("scan") else PARALLEL_MODE if omode.startswith("parallel") else None
        if mode is None:
            raise ValueError(f"unknown observing mode {omode!r}: {path}")
        shape = (int(himg["NAXIS2"]), int(himg["NAXIS1"]))
        wcs_cards = fits.Header([c for c in himg.cards if _WCS_KEY.match(c.keyword)])
        full = head0.copy()                       # primary + image cards, as the PSF stage reads them
        full.extend(himg, update=True, strip=False)
        wcs = WCS(wcs_cards).celestial
        ny, nx = shape
        corners = wcs.pixel_to_world([-0.5, nx - 0.5, nx - 0.5, -0.5],
                                     [-0.5, -0.5, ny - 0.5, ny - 0.5]).icrs
        polygon = tuple((float(a), float(d)) for a, d in zip(corners.ra.deg, corners.dec.deg))
        scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0)
        obsids = tuple(str(head0[k]) for k in head0 if k.startswith("OBSID") and k[5:].isdigit())
        if not obsids:
            obsids = (os.path.basename(path).split("_")[1],)
        mapper = str(head0.get("MAPPER", "unknown"))
        level = {"level2": "L20", "level2_5": "L25", "level3": "L30"}.get(
            rel.split(os.sep)[1] if rel.count(os.sep) >= 2 else "", "")
        return MapRecord(
            map_id=str(head0.get("FILENAME", "")).strip() or os.path.basename(path),
            obsids=obsids, band=band, path=rel, image_hdu=ih, error_hdu=eh,
            coverage_hdu=ch, shape=shape,
            unit=str(himg.get("QTTY____", himg.get("BUNIT", ""))),
            wcs_header=full.tostring(sep=""), mode=mode, level=level,
            mapper=mapper, scan_speed=float(head0.get("SCANVELO", float("nan"))),
            position_angle=float(head0.get("POSANGLE", float("nan"))),
            polygon=polygon,
            error_unit=str(hdus[eh].header.get("BUNIT", hdus[eh].header.get("QTTY____", ""))),
            error_kind=kind, pixel_scale_arcsec=scale, source=source)


def map_rank(rec):
    """Preference among maps of one observation (spec 11.2): 0 JScanam Level 3 (the stack of several
    observations: at least as deep everywhere, deeper where they overlap), 1 JScanam L2.5, 2 Unimap,
    3 everything else (the L2 photProject maps)."""
    m = rec.mapper.lower()
    if m.startswith("jscanam"):
        return 0 if rec.level == "L30" else 1 if rec.level == "L25" else 3
    if m.startswith("unimap"):
        return 2
    return 3


def choose(obs_ids, candidates):
    """Picks the maps that serve `obs_ids`. `candidates` is a list of
    (key, rank, obsids) -- every map that exists, the observations merged
    into it. Each observation takes its best-ranked map; a map that already
    contains an observation is never taken again for it. Returns
    (chosen keys in pick order, observations no candidate contains)."""
    best = {}
    for key, rank, ids in candidates:
        for o in set(ids) & set(obs_ids):
            if o not in best or (rank, key) < (best[o][0], best[o][1]):
                best[o] = (rank, key)
    by_key = {k: (r, set(i)) for k, r, i in candidates}
    chosen, covered = [], set()
    for o in sorted(best, key=lambda x: (best[x][0], x)):
        if o in covered:
            continue
        key = best[o][1]
        chosen.append(key)
        covered |= by_key[key][1]
    return chosen, sorted(set(obs_ids) - set(best))


def disk_maps(config):
    """Every blue-channel map on disk, as records (header inspection only)."""
    root = maps_root(config)
    out = []
    for f in sorted(glob.glob(f"{root}/*/level*/*/*.fits")):
        try:
            out.append(inspect_map(f, root))
        except ValueError as exc:
            print(f"index: skipped {f}: {exc}")
    return out


def region_index_records(config, region):
    """The maps on disk that serve the region's observations."""
    obs = {o["obsid"] for o in region_observations(config, region)}
    recs = {r.path: r for r in disk_maps(config)}
    out = []
    for red in (False, True):                     # one map per observation per camera
        cands = [(r.path, map_rank(r), r.obsids) for r in recs.values() if (r.band == "P160") == red]
        chosen, _ = choose(obs, cands)
        out += [recs[k] for k in chosen]
    return out


# ----- the region's observations -------------------------------------------

_NUM = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def parse_polygon(text):
    """The archive's polygon_fov string, '{(ra , dec),(ra , dec),...}' in
    radians, as an (n, 2) array of ICRS degrees; None when empty."""
    vals = np.array(_NUM.findall(text or ""), dtype=float)
    if len(vals) < 6:
        return None
    return np.degrees(vals.reshape(-1, 2))


def region_pixels(config, region):
    """The region's admitted nside-512 pixels (`catalog.depth_grid.admitted_pixels_512`'s rule, on the raw
    delivery's own Galactic positions through `catalog.curated.read_galactic`, so this download runs before
    the catalogue stages): centres and corners in ICRS degrees, (n,2) and (n,4,2)."""
    from sesnaimpute.catalog import curated, depth_grid
    pix = depth_grid.admitted_pixels_512_from(*curated.read_galactic(config, region)[1:])
    lon, lat = hp.pix2ang(NSIDE, pix, nest=True, lonlat=True)
    centre = SkyCoord(lon * u.deg, lat * u.deg, frame="galactic").icrs
    xyz = hp.boundaries(NSIDE, pix, step=1, nest=True)       # (n, 3, 4)
    cl, cb = hp.vec2ang(np.moveaxis(xyz, 1, 2).reshape(-1, 3), lonlat=True)
    corner = SkyCoord(cl * u.deg, cb * u.deg, frame="galactic").icrs
    return (np.column_stack([centre.ra.deg, centre.dec.deg]),
            np.stack([corner.ra.deg.reshape(-1, 4), corner.dec.deg.reshape(-1, 4)], axis=-1))


def _plane(ra, dec, ra0, dec0):
    """Gnomonic offsets (degrees) about (ra0, dec0)."""
    r, d, r0, d0 = map(np.radians, (ra, dec, ra0, dec0))
    c = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(r - r0)
    x = np.cos(d) * np.sin(r - r0) / c
    y = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(r - r0)) / c
    return np.degrees(x), np.degrees(y)


def polygon_touches(poly, centres, corners, ra0, dec0):
    """True when the ICRS polygon `poly` (n,2 deg) overlaps any pixel."""
    px, py = _plane(poly[:, 0], poly[:, 1], ra0, dec0)
    cx, cy = _plane(centres[:, 0], centres[:, 1], ra0, dec0)
    pad = 0.07                      # a nside-512 pixel's half-diagonal, degrees
    near = ((cx > px.min() - pad) & (cx < px.max() + pad)
            & (cy > py.min() - pad) & (cy < py.max() + pad))
    if not near.any():
        return False
    ring = MplPath(np.column_stack([px, py]))
    idx = np.flatnonzero(near)
    qx, qy = _plane(corners[idx, :, 0], corners[idx, :, 1], ra0, dec0)
    pts = np.concatenate([np.column_stack([cx[idx], cy[idx]])[:, None, :],
                          np.stack([qx, qy], axis=-1)], axis=1)       # (m, 5, 2)
    if ring.contains_points(pts.reshape(-1, 2)).any():
        return True
    for q in np.stack([qx, qy], axis=-1):                               # edge crossing
        if MplPath(q).intersects_path(ring, filled=True):
            return True
    return False


def read_tap(config, region):
    """The cached TAP rows as dicts with obsid (str), mode, ra, dec, polygon."""
    out = []
    csv.field_size_limit(10**9)
    with open(tap_path(config, region), newline="") as src:
        for row in csv.DictReader(src):
            out.append(dict(obsid=row["observation_id"].strip(),
                            mode=int(row["observing_mode_oid"]),
                            ra=float(row["ra"]), dec=float(row["dec"]),
                            polygon=parse_polygon(row["polygon_fov"]),
                            target=row.get("target_name", ""),
                            state=row.get("obsstate", "")))
    return out


def region_observations(config, region):
    """The cached TAP observations whose footprint polygon touches one of
    the region's admitted nside-512 pixels. An observation with no polygon
    in the archive is kept when its centre is within 0.5 deg of a pixel
    (its map header decides later)."""
    rows = read_tap(config, region)
    if not rows:
        return []
    centres, corners = region_pixels(config, region)
    keep = []
    for r in rows:
        if r["polygon"] is None:
            d = np.hypot((centres[:, 0] - r["ra"]) * np.cos(np.radians(r["dec"])),
                         centres[:, 1] - r["dec"])
            if (d < 0.5).any():
                keep.append(dict(r, nopolygon=True))
        elif polygon_touches(r["polygon"], centres, corners, r["ra"], r["dec"]):
            keep.append(dict(r, nopolygon=False))
    return keep
