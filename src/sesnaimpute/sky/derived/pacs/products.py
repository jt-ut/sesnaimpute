"""The PACS build's three products, with UNITS and READING on every dataset, and the CHECKS table they carry.

    completeness_pacs_source__<Region>.hdf5   per covered cell and band: CURVE, FLUX90, SCATTER, BIAS, CELL_SIGMA,
                                              CELL_OFFSET, ...; SOURCE_ROW per SESNA source; MAP_SHAPE/MAP_WCS/
                                              MAP_CELL_STEP/CELL_ID per band
    sources_pacs_region__<Region>.hdf5        the region's PACS sources
    photometry_pacs_source__<Region>.hdf5     per SESNA source and band: FNU, SIGMA, ORIGIN, ...

Depends on: grid, psf, mosaic, detect, inject, photometry (types only). Replaces: pacs_completeness.write_product/
write_sources/_read_band/_held_path/_read_held, pacs_photometry.write_region's writer part/_DTYPES/_FILL/_NAMES. Paths:
config.product_path(config, "sky/derived", "pacs", <quantity>, <granule>, region) exactly as today. Datasets: written
through build.write_dataset (UNITS, READING) with the same names, dtypes, shapes, chunking and fill values as main
0683690, so the products are byte-identical except for the added CHECKS group and one sentence added to the READING of
FNU_<B>_MJY (the empty-sky offset: FNU excludes it).

CHECKS (new, FIX_SPEC.md sec 11): a group `CHECKS` in each product holding one row per check run on it: NAME (str),
BAND (str), VALUE (float64), BAR_LOW, BAR_HIGH (float64, NaN = none), VERDICT (str: PASS, FAIL, INFO), DETAIL (str; the
number of cases behind the value is its first words, "n=<count>; ...").

What write_completeness reads of a BandCompleteness (the carrier of the contract; an empty band has detection.cell_grid
None and zero rows):
  curve, flux90, n_pos, scatter, bias; detection.cells (CellStats: ids, ra, dec, sigma, offset, n_sigma, m_sigma,
  m_offset), detection.src_row, detection.cell_bin, detection.r_match, detection.shift (east, north, n_pairs),
  detection.cell_grid (CellGrid: wcs, shape, step_x, step_y); stats: map_id, map_obsid (str), rms_inj, rms_align (float).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from sesnaimpute import build as build_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.config import product_path
from sesnaimpute.definitions import COMPLETENESS_STORAGE_SCALE as STORAGE_SCALE
from sesnaimpute.definitions import PACS_BANDS as BANDS
from sesnaimpute.definitions import PACS_POOL_ARCSEC as POOL_ARCSEC
from sesnaimpute.definitions import PACS_POOL_WIDENINGS as POOL_WIDENINGS
from sesnaimpute.sky.derived.pacs import inject, photometry
from sesnaimpute.sky.derived.pacs.grid import CellGrid
from sesnaimpute.sky.derived.pacs.inject import BandCompleteness
from sesnaimpute.sky.derived.pacs.photometry import ForcedPhotometry, Sources

#: PACS has no Vega system: zero point 1 Jy, m = -2.5 log10(F / 1 Jy) (the CURVE dataset's attribute ZERO_POINT_JY)
ZERO_POINT_JY = 1.0

#: appended to the READING of FNU_<B>_MJY: what a reader comparing with HOPS, eHOPS or HPPSC must know
FNU_OFFSET_READING = ("FNU excludes the empty-sky offset CELL_OFFSET; add it back for HOPS/HPPSC conventions "
                      "(at 160 um FNU alone reads 0.74 of HOPS).")


@dataclass(frozen=True)
class CheckRow:
    """One check's result, as stored in a product's CHECKS group: name, band ("" for region-wide), value, bar_low,
    bar_high (NaN = none), verdict ("PASS" | "FAIL" | "INFO"), detail (one line, plain words)."""
    name: str
    band: str
    value: float
    bar_low: float
    bar_high: float
    verdict: str
    detail: str


# ----------------------------------------------------------------------------------------------------------------------
# paths

def completeness_path(config, region: str) -> str:
    """The completeness product's path (product_path(config, "sky/derived", "pacs", "completeness", "source", region))."""
    return product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)


def sources_path(config, region: str) -> str:
    """The sources product's path (quantity "sources", granule "region")."""
    return product_path(config, "sky/derived", "pacs", "sources", "region", region=region)


def photometry_path(config, region: str) -> str:
    """The photometry product's path (quantity "photometry", granule "source")."""
    return product_path(config, "sky/derived", "pacs", "photometry", "source", region=region)


# ----------------------------------------------------------------------------------------------------------------------
# the completeness product

def _strings(values, n_rows: int = None) -> np.ndarray:
    """A byte-string array as the old writer made it (empty: one-byte strings)."""
    return np.array(list(values) or [""], dtype="S")[:len(values)]


def _band_arrays(b: BandCompleteness) -> dict:
    """The arrays of one band's datasets, from the BandCompleteness (the old band dict's keys, `_band_dict`/`_empty_band`)."""
    det = b.detection
    n_plane = b.curve.shape[1]
    empty = det is None or det.cell_grid is None
    if empty:
        z = lambda dt: np.zeros(0, dt)
        return dict(curve=np.zeros((0, n_plane), np.uint16), source_row=np.asarray(det.src_row, np.int32),
                    envelope=np.zeros(n_plane, np.uint16), n_pos=z(np.int32), n_sigma=z(np.int16), flux90=z(np.float32),
                    bias=np.zeros((0, n_plane), np.float32), scatter=np.zeros((0, n_plane), np.float32),
                    sky_bin=z(np.int8), cell_m_sigma=z(np.float32), cell_m_offset=z(np.float32), r_match=z(np.float32),
                    rms_inj=np.nan, rms_align=np.nan, cell_ra=z(float), cell_dec=z(float), cell_sigma=z(np.float32),
                    cell_offset=z(np.float32), cell_id=z(np.int64), cell_map_index=z(np.int32), cell_map=z("S1"),
                    map_id=z("S1"), map_obsid=z("S1"), map_shift=np.zeros((0, 2), np.float32), map_pairs=z(np.int32),
                    map_wcs=z("S1"), map_shape=np.zeros((0, 2), np.int32), map_cell_step=np.zeros((0, 2)))
    n_row = b.curve.shape[0]
    cells, grid = det.cells, det.cell_grid
    map_id, map_obsid = str(b.stats["map_id"]), str(b.stats["map_obsid"])
    envelope = b.curve.max(axis=0).astype(np.uint16) if n_row else np.zeros(n_plane, np.uint16)
    return dict(curve=b.curve, source_row=np.asarray(det.src_row).astype(np.int32), envelope=envelope, n_pos=b.n_pos,
                n_sigma=np.asarray(cells.n_sigma).astype(np.int16), flux90=b.flux90, bias=b.bias, scatter=b.scatter,
                sky_bin=np.asarray(det.cell_bin).astype(np.int8), cell_m_sigma=cells.m_sigma, cell_m_offset=cells.m_offset,
                r_match=np.asarray([det.r_match], np.float32), rms_inj=float(b.stats.get("rms_inj", np.nan)),
                rms_align=float(b.stats.get("rms_align", np.nan)), cell_ra=cells.ra, cell_dec=cells.dec,
                cell_sigma=cells.sigma, cell_offset=cells.offset, cell_id=cells.ids,
                cell_map_index=np.zeros(n_row, np.int32), cell_map=np.array([map_id] * n_row or [""], dtype="S")[:n_row],
                map_id=np.array([map_id], dtype="S"), map_obsid=np.array([map_obsid], dtype="S"),
                map_shift=np.array([[det.shift[0], det.shift[1]]], np.float32), map_pairs=np.array([det.shift[2]], np.int32),
                map_wcs=np.array([grid.wcs.to_header_string()], dtype="S"),
                map_shape=np.array([grid.shape], dtype=np.int32).reshape(1, 2),
                map_cell_step=np.array([(grid.step_x, grid.step_y)], dtype=np.float64).reshape(1, 2))


def write_completeness(path: str, region: str, mag: np.ndarray, bands: Dict[str, BandCompleteness]) -> None:
    """Writes the completeness product. Identity: `pacs_completeness.write_product` (pacs_completeness.py:1758) --
    every dataset and attribute byte-identical (FROZEN 1/4: CELL_ID and row order)."""
    stem = "completeness_pacs_source"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    snr = inject.snr_grid()
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.attrs["REGION"] = region
        for b in BANDS:
            d = _band_arrays(bands[b])
            w = lambda name, data, **kw: build_module.write_dataset(f, name, data, *REGISTRY[(stem, name)], **kw)
            ds = w("CURVE_" + b, d["curve"])
            ds.attrs["ZERO_POINT_JY"] = ZERO_POINT_JY
            ds.attrs["POOL_ARCSEC"] = POOL_ARCSEC
            ds.attrs["POOL_WIDENINGS"] = np.asarray(POOL_WIDENINGS)
            ds.attrs["SNR_GRID"] = snr
            ds.attrs["R_MATCH_INJECTED_RMS_ARCSEC"] = float(d["rms_inj"])
            ds.attrs["R_MATCH_ALIGNMENT_POOLED_RMS_ARCSEC"] = float(d["rms_align"])
            w("MAG_GRID_" + b, mag)
            w("ENVELOPE_" + b, d["envelope"])
            w("COMPLETED_FROM_" + b, np.full(d["curve"].shape[0], np.nan, np.float32))
            w("N_POS_" + b, d["n_pos"])
            w("N_SIGMA_POS_" + b, d["n_sigma"])
            w("FLUX90_%s_MJY" % b, d["flux90"])
            w("BIAS_%s_MJY" % b, d["bias"])
            w("SCATTER_%s_MJY" % b, d["scatter"])
            w("SKY_BIN_" + b, d["sky_bin"])
            w("CELL_SIGMA_MATCHED_" + b, d["cell_m_sigma"])
            w("CELL_OFFSET_MATCHED_" + b, d["cell_m_offset"])
            w("MAP_R_MATCH_%s_ARCSEC" % b, np.asarray(d["r_match"], np.float32))
            w("CELL_RA_" + b, d["cell_ra"])
            w("CELL_DEC_" + b, d["cell_dec"])
            w("CELL_ID_" + b, d["cell_id"])
            w("CELL_MAP_INDEX_" + b, d["cell_map_index"])
            w("CELL_MAP_" + b, d["cell_map"])
            w("CELL_SIGMA_%s_MJY" % b, d["cell_sigma"])
            w("CELL_OFFSET_%s_MJY" % b, d["cell_offset"])
            w("MAP_ID_" + b, d["map_id"])
            w("MAP_OBSID_" + b, d["map_obsid"])
            w("MAP_SHIFT_%s_ARCSEC" % b, d["map_shift"])
            w("MAP_N_PAIRS_" + b, d["map_pairs"])
            w("MAP_WCS_" + b, d["map_wcs"])
            w("MAP_SHAPE_" + b, d["map_shape"])
            w("MAP_CELL_STEP_" + b, d["map_cell_step"])
        w = lambda name, data: build_module.write_dataset(f, name, data, *REGISTRY[(stem, name)])
        w("SOURCE_ROW", np.stack([np.asarray(bands[b].detection.src_row) for b in BANDS], axis=1).astype(np.int32))


# ----------------------------------------------------------------------------------------------------------------------
# the sources product

def write_sources(path: str, region: str, bands: Dict[str, BandCompleteness], sources: Sources) -> None:
    """Writes the sources product. Identity: `pacs_completeness.write_sources` (pacs_completeness.py:1953).

    One row per PACS source (the bands' blind detections merged across bands): its position (from the sharpest detecting
    band), and per band whether it detects the source, the flux at the position and its uncertainty, the offset
    subtracted, the detection's S/N, the map and the extended flag; the merge radii and the 160-only count as attributes of
    RA_DEG."""
    stem = "sources_pacs_region"
    c = sources.columns
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "region"
        f.attrs["REGION"] = region
        w = lambda name, data: build_module.write_dataset(f, name, data, *REGISTRY[(stem, name)])
        ds = w("RA_DEG", np.asarray(c["RA_DEG"], np.float64))
        for key in c:
            if key.startswith("MERGE_RADIUS_"):
                ds.attrs[key] = float(c[key])
        ds.attrs["N_160_ONLY"] = int((c["POSITION_BAND"] == BANDS.index("P160")).sum())
        w("DEC_DEG", np.asarray(c["DEC_DEG"], np.float64))
        w("POSITION_BAND", c["POSITION_BAND"].astype(np.int8))
        for b in BANDS:
            w("DETECTED_" + b, c["DETECTED_" + b])
            w("FNU_%s_MJY" % b, c["FNU_%s_MJY" % b])
            w("SIGMA_%s_MJY" % b, c["SIGMA_%s_MJY" % b])
            w("CELL_OFFSET_%s_MJY" % b, c["CELL_OFFSET_%s_MJY" % b])
            w("SNR_" + b, np.asarray(c["SNR_" + b], np.float32))
            w("MAP_INDEX_" + b, c["MAP_INDEX_" + b])
            w("EXTENDED_" + b, c["EXTENDED_" + b])
            w("RA_%s_DEG" % b, np.asarray(c["RA_%s_DEG" % b], np.float64))
            w("DEC_%s_DEG" % b, np.asarray(c["DEC_%s_DEG" % b], np.float64))


def read_sources(path: str) -> Sources:
    """The sources product as the in-memory table `photometry.cross_match` reads (the datasets as stored; the merge radii
    from the attributes of RA_DEG)."""
    cols: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for name, ds in f.items():
            if isinstance(ds, h5py.Dataset):
                cols[name] = ds[()]
        for key, v in f["RA_DEG"].attrs.items():
            if key.startswith("MERGE_RADIUS_"):
                cols[key] = np.array(v)
    return Sources(cols)


# ----------------------------------------------------------------------------------------------------------------------
# the photometry product

def write_photometry(path: str, region: str, table: ForcedPhotometry) -> None:
    """Writes the photometry product (chunks (min(n, 65536),), fillvalue per column). Identity: `pacs_photometry.
    write_region`'s file plus `cross_match`'s in-place columns (pacs_photometry.py:322-370, 178-261), byte-identical.
    The READING of FNU_<B>_MJY ends with FNU_OFFSET_READING. Where the cross-match found PACS sources, ORIGIN_<B> carries
    MATCH_RADIUS_ARCSEC and ORIGIN_P70 the EXPECTED_CHANCE_COINCIDENCES."""
    stem = "photometry_pacs_source"
    cols = table.columns
    names = cols["NAME"]
    n = names.size
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        build_module.write_dataset(f, "NAME", names, *REGISTRY[(stem, "NAME")])
        for b in BANDS:
            for key, pattern in photometry.PHOTOMETRY_NAMES.items():
                name = pattern.format(b=b)
                ds = build_module.write_dataset(
                    f, name, None, *REGISTRY[(stem, name)], shape=(n,), dtype=photometry.PHOTOMETRY_DTYPES[key],
                    fillvalue=photometry.PHOTOMETRY_FILL[key], chunks=(min(n, 65536),) if n else None)
                if n:
                    ds[:] = cols[name]
                if key == "FNU":
                    ds.attrs["READING"] = ds.attrs["READING"] + " " + FNU_OFFSET_READING
        if "_CHANCE" in cols:
            for b in BANDS:
                f["ORIGIN_" + b].attrs["MATCH_RADIUS_ARCSEC"] = photometry.MATCH_RADIUS_ARCSEC
            f["ORIGIN_P70"].attrs["EXPECTED_CHANCE_COINCIDENCES"] = float(cols["_CHANCE"])


# ----------------------------------------------------------------------------------------------------------------------
# the CHECKS table

#: (dataset, dtype, units, reading) of the CHECKS columns
_CHECK_COLUMNS = (
    ("NAME", "S", "name", "The check's short name; the check is defined where the build code defines it (checks.py)."),
    ("BAND", "S", "band", "The PACS band the check was run on (P70, P100, P160), or empty for a whole-region check."),
    ("VALUE", "f8", "as named", "The measured number the check reports (a share in per cent, a ratio, a deviation: "
     "the DETAIL says which)."),
    ("BAR_LOW", "f8", "as VALUE", "The lowest VALUE that passes; NaN where the check has no lower bar."),
    ("BAR_HIGH", "f8", "as VALUE", "The highest VALUE that passes; NaN where the check has no upper bar."),
    ("VERDICT", "S", "verdict", "PASS where VALUE is within the bar, FAIL where it is not, INFO for a number reported "
     "with no bar."),
    ("DETAIL", "S", "text", "One line in plain words: how many cases stand behind VALUE (n=...) and what was compared."),
)


def write_checks(path: str, rows: Sequence[CheckRow]) -> None:
    """Appends `rows` to the product's CHECKS group (created on first use; UNITS/READING on each column)."""
    rows = list(rows)
    if not rows:
        return
    cols = dict(NAME=[r.name for r in rows], BAND=[r.band for r in rows], VALUE=[r.value for r in rows],
                BAR_LOW=[r.bar_low for r in rows], BAR_HIGH=[r.bar_high for r in rows], VERDICT=[r.verdict for r in rows],
                DETAIL=[r.detail for r in rows])
    text = h5py.string_dtype("utf-8")
    with h5py.File(path, "r+") as f:
        group = f.require_group("CHECKS")
        for name, kind, units, reading in _CHECK_COLUMNS:
            if kind == "S":
                data = np.array(cols[name], dtype=object)
            else:
                data = np.array(cols[name], dtype=np.float64)
            if name in group:
                ds = group[name]
                m = ds.shape[0]
                ds.resize((m + len(rows),))
                ds[m:] = data
            else:
                ds = build_module.write_dataset(group, name, data, units, reading, maxshape=(None,), chunks=(max(len(rows), 64),),
                                                **({"dtype": text} if kind == "S" else {}))


def read_checks(path: str) -> List[CheckRow]:
    """The CHECKS rows of a product."""
    with h5py.File(path, "r") as f:
        if "CHECKS" not in f:
            return []
        g = f["CHECKS"]
        s = lambda k: [v.decode() if isinstance(v, bytes) else str(v) for v in g[k][()]]
        return [CheckRow(n, b, float(v), float(lo), float(hi), ver, det) for n, b, v, lo, hi, ver, det in
                zip(s("NAME"), s("BAND"), g["VALUE"][()], g["BAR_LOW"][()], g["BAR_HIGH"][()], s("VERDICT"), s("DETAIL"))]


# ----------------------------------------------------------------------------------------------------------------------
# reading a band back

def read_completeness_band(path: str, band: str) -> dict:
    """One band's datasets of the completeness product (`pacs_completeness._read_band`): curve, mag, flux90, n_pos,
    scatter, bias, sigma (CELL_SIGMA), offset (CELL_OFFSET), sky_bin, cell_id, cell_map, map_id, map_wcs, map_shape,
    map_shift, source_row, r_match (median over maps), flux (mJy of the plane), and `centres` (n, 2): each cell's centre in
    arcsec in the pooling frame (maps 1e7 arcsec apart), what the pooling matrices are built from. Pooling matrices
    themselves are built by the caller (they depend on the pooling rule)."""
    with h5py.File(path, "r") as f:
        g = lambda name: f[name][:]
        d = dict(curve=g("CURVE_" + band), mag=g("MAG_GRID_" + band), flux90=g("FLUX90_%s_MJY" % band),
                 n_pos=g("N_POS_" + band), scatter=g("SCATTER_%s_MJY" % band), bias=g("BIAS_%s_MJY" % band),
                 sigma=g("CELL_SIGMA_%s_MJY" % band), offset=g("CELL_OFFSET_%s_MJY" % band),
                 sky_bin=g("SKY_BIN_" + band).astype(np.int64), cell_id=g("CELL_ID_" + band),
                 cell_map=g("CELL_MAP_INDEX_" + band), map_id=g("MAP_ID_" + band), map_wcs=g("MAP_WCS_" + band),
                 map_shape=g("MAP_SHAPE_" + band), map_shift=g("MAP_SHIFT_%s_ARCSEC" % band),
                 source_row=g("SOURCE_ROW")[:, BANDS.index(band)])
        d["r_match"] = float(np.nanmedian(f["MAP_R_MATCH_%s_ARCSEC" % band][:]))
        zero_point = float(f["CURVE_" + band].attrs["ZERO_POINT_JY"])
    d["flux"] = 1000.0 * zero_point * 10.0 ** (-d["mag"].astype(np.float64) / 2.5)
    xy = np.zeros((d["cell_id"].size, 2))
    for m in range(d["map_id"].size):
        rows = np.flatnonzero(d["cell_map"] == m)
        if rows.size:
            cells = CellGrid.from_wcs(WCS(fits.Header.fromstring(d["map_wcs"][m].decode())).celestial, tuple(d["map_shape"][m]))
            xy[rows] = cells.centres_arcsec(d["cell_id"][rows]) + np.array([m * 1e7, 0.0])
    d["centres"] = xy
    return d
