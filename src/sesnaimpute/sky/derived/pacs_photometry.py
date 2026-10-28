"""Herschel PACS 70 and 100 micron flux at every SESNA source position
(SPEC_PACS_FARIR_DRAFT.md sections 4, 7b and 11.3 step 3).

`build(config, regions=None)` measures, for every source of
`catalog/sesna/sources_sesna_source__<Region>.hdf5`, the flux in each PACS
band on the map the region's completeness stage chose for the source's cell
(`sky.derived.pacs_completeness`, the deepest map there), from the maps the
download index lists (`sky.download.herschel_pacs.index.read_index`), and
writes `sky/derived/pacs/photometry_pacs_source__<Region>.hdf5`: one row per
source in the catalogue's own order, with its flux, its noise, the origin
code the fitter reads (1 detected, 2 upper bound from an unseparable blend,
90 observed but not detected, 91 not covered), the contributing observation
ids, the blend bookkeeping (how many detections lie within a beam, and the
brightest of them) and the extended-source flag.

Everything numerical is `sky.derived.pacs_fit`'s: the eHOPS aperture photometry and the fitted width. This
stage only decides which sources go to which map tile, runs the singleton pass, applies the blend rule and turns
the measurements into products:

- flux: FNU is the aperture flux minus its cell's empty-sky offset, `CELL_OFFSET_<B>_MJY` (stored per source;
  adding it back gives the HOPS/eHOPS convention).
- noise: a detected source's sigma is the photometric error the completeness stage MEASURED for a source of
  its flux in its cell, by recovering artificial sources of known flux injected into the map: the cell's
  `SCATTER_<B>_MJY` at the source's flux (log-linear between the flux steps). A source that is not detected
  carries the noise of its 30-arcsec cell, `CELL_SIGMA_<B>_MJY`, the same number the detection rule uses.
- detection (RULINGS_4): flux - CELL_OFFSET >= 5 x CELL_SIGMA, a fixed precision floor, AND RELIABILITY >= 0.8,
  the chance that the flux is the source's own rather than the field's (Sutherland & Saunders 1992), from the
  completeness stage's measured densities (origin 1). Above the floor with RELIABILITY < 0.8: origin 2, the flux an
  upper bound. Below the floor: origin 90. A source the completeness stage found no cell for, or whose aperture has
  too little valid data, is 91.
- blends (the far-infrared survey convention; planner ruling, owner-approved 2026-10-28): a source within one
  beam of another source detected on the map is an unseparable blend, origin 2, its own aperture flux (offset
  subtracted, so it holds the neighbour's light) as the upper bound and the pipeline's upper-limit sentinel as
  its sigma, unless it is itself detected and the brightest of the detections within that beam.
- extended: a detected source whose fitted width exceeds 1.5 x the beam.

Work is split by map tile (one piece per band and map), the tile's sources
being transformed to map pixels in one call; the product is preallocated and
each tile's rows are written as they arrive.
"""

import functools
import os
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog.curated import UPPER_LIMIT_SIGMA
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived import pacs_fit
from sesnaimpute.sky.derived.pacs_fit import DETECTION_SNR
from sesnaimpute.sky.download.herschel_pacs.index import read_index

BANDS = ("P70", "P100", "P160")

#: the pipeline's origin codes for a flux (`ORIGIN_FNU` in the source table)
ORIGIN_DETECTED, ORIGIN_UPPER_BOUND, ORIGIN_UNDETECTED, ORIGIN_NOT_COVERED = 1, 2, 90, 91

#: a detection fitted wider than this many beam widths is flagged extended
#: (SPEC 11.3 step 3); the width is `pacs_fit.Tile.fitted_fwhm`, the beam
#: broadened by a Gaussian chosen by chi-square
EXTENDED_BEAM_FACTOR = 1.5
#: a source within this many beam FWHM of another source's detection is an unseparable blend (ORIGIN 2) unless it is
#: the brighter member: the far-infrared survey convention (HOPS/eHOPS flag close pairs the same way; planner ruling,
#: owner-approved 2026-10-28)
BLEND_RADIUS_BEAMS = 1.0


# ---------------------------------------------------------------------------
# positions

def _unit_vectors(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])



# ---------------------------------------------------------------------------
# the singleton pass and the calibrated sigma

#: bins of injected S/N (F / CELL_SIGMA) in which the completeness stage's C2 checks the measured error: < 8, 8-30,
#: 30-120, >= 120
SNR_BIN_EDGES = (8.0, 30.0, 120.0)
SNR_BIN_CENTRES = (5.7, 15.5, 60.0, 240.0)


def singleton_pass(tile, ra, dec, shift, sigma_cell, offset):
    """The aperture measurement of every source of `ra`, `dec` on `tile` and the catalogue's detection of
    it: returns `(x, y, single, ok, detected)`, the map pixels (the map's registration `shift` applied),
    the aperture fit, whether the source has a usable cell noise and fit, and whether its flux minus the
    cell's empty-sky offset reaches DETECTION_SNR x the cell noise. The completeness stage runs this same
    function on the same inputs to know which sources the catalogue detects."""
    x, y = tile.xy(*pacs_fit.to_map_frame(ra, dec, shift))
    single = tile.measure(x, y)
    ok = single.covered & np.isfinite(sigma_cell) & (sigma_cell > 0)
    detected = ok & (single.flux - offset >= DETECTION_SNR * sigma_cell)
    return x, y, single, ok, detected


#: the reliability rule (RULINGS_4, Sutherland & Saunders 1992): a source above the 5-sigma floor is ORIGIN 1 only
#: if the chance that its flux is its own, q / (q + n), is at least R_MIN; the densities are tabulated in log10 of
#: FNU (mJy) on this grid
R_MIN = 0.8
LOGF_MIN, LOGF_MAX, LOGF_STEP = -1.0, 7.0, 0.1


def logf_edges():
    """The edges (log10 mJy) of the reliability tables' flux bins."""
    return np.round(np.arange(LOGF_MIN, LOGF_MAX + LOGF_STEP / 2, LOGF_STEP), 6)


def logf_bin(fnu):
    """The reliability-table bin of each flux (mJy); -1 where not positive and finite."""
    fnu = np.asarray(fnu, float)
    out = np.full(fnu.shape, -1, np.int64)
    good = np.isfinite(fnu) & (fnu > 0)
    n_f = logf_edges().size - 1
    out[good] = np.clip(np.floor((np.log10(fnu[good]) - LOGF_MIN) / LOGF_STEP).astype(np.int64), 0, n_f - 1)
    return out


def reliability(fnu, table, r_grid):
    """RELIABILITY of each flux `fnu` (mJy) read from its table's row of `r_grid` (n_table, n_f); NaN where the
    flux is not positive or the table is -1."""
    k = logf_bin(fnu)
    t = np.asarray(table, np.int64)
    out = np.full(k.shape, np.nan)
    use = (k >= 0) & (t >= 0)
    out[use] = r_grid[t[use], k[use]]
    return out


def blend_rule(x, y, fnu, detected, ok, beam_px):
    """The far-infrared survey convention: a source within BLEND_RADIUS_BEAMS of another `detected` source is an
    unseparable blend unless it is itself detected and the brightest (by `fnu`) of the detections within that
    radius. Returns (blend (bool), how many other detections lie within the radius, the index of the brightest of
    them or -1)."""
    n = x.size
    blend = np.zeros(n, bool)
    n_near = np.zeros(n, np.int16)
    brightest_of = np.full(n, -1, np.int64)
    d_idx = np.flatnonzero(detected)
    if d_idx.size == 0:
        return blend, n_near, brightest_of
    tree = cKDTree(np.c_[x[d_idx], y[d_idx]])
    near_lists = tree.query_ball_point(np.c_[x, y], BLEND_RADIUS_BEAMS * beam_px)
    for i in np.flatnonzero(ok):
        others = [d_idx[j] for j in near_lists[i] if d_idx[j] != i]
        if not others:
            continue
        n_near[i] = len(others)
        brightest = max(others, key=lambda j: fnu[j])
        brightest_of[i] = brightest
        blend[i] = not (detected[i] and fnu[i] >= fnu[brightest])
    return blend, n_near, brightest_of


def calibrated_sigma(scatter, flux_grid, flux):
    """The photometric error of sources of flux `flux` (n,) mJy from their cells' measured scatter at the
    grid planes: `scatter` (n, n_plane), `flux_grid` (n_plane,) mJy, bright to faint. Log-linear in flux
    between the two planes that bracket the flux, clamped to the grid ends; where either bracketing plane
    holds NaN the value of the nearest plane of the same row that is finite; NaN for a row with none."""
    asc = np.log10(flux_grid[::-1])
    rows = np.asarray(scatter, float)[:, ::-1]
    n, p = rows.shape
    t = np.log10(np.maximum(np.asarray(flux, float), 1e-300))
    hi = np.clip(np.searchsorted(asc, t), 1, p - 1)
    lo = hi - 1
    w = np.clip((t - asc[lo]) / (asc[hi] - asc[lo]), 0.0, 1.0)
    r = np.arange(n)
    both = np.isfinite(rows[r, lo]) & np.isfinite(rows[r, hi])
    out = np.where(both, rows[r, lo] * (1 - w) + rows[r, hi] * w, np.nan)
    need = np.flatnonzero(~both)
    if need.size:
        finite = np.isfinite(rows[need])
        away = np.where(finite, np.abs(np.arange(p)[None, :] - (lo[need] + w[need])[:, None]), np.inf)
        k = np.argmin(away, axis=1)
        out[need] = np.where(finite.any(axis=1), rows[need, k], np.nan)
    return out


# ---------------------------------------------------------------------------
# one tile

def _measure_tile(piece, config, records, ra, dec, cell_sigma, cell_offset, shifts, flux_grid, cell_table, r_grid):
    """All of one map tile's sources: `piece = (band, map_index, rows, scatter)`, the source rows (ascending) the
    completeness stage assigned to that map and, for each, its cell's measured aperture SCATTER (n, n_plane, mJy).
    `cell_table[band]` (n,) is each source's reliability table (-1 none) and `r_grid[band]` the tables
    (RELIABILITY_GRID of the completeness product). Returns `(band, map_index, rows, columns, n_nan, n_blend_2,
    n_reliability_2)`: the product's per-source arrays for those rows, the detections with no measured error, and
    the sources above the floor turned into upper bounds by the blend rule and by the reliability rule."""
    band, map_index, rows, scatter = piece
    record = records[map_index]
    n = rows.size
    sigma_cell = cell_sigma[band][rows]
    # the cell's empty-sky offset (sky structure the aperture's sky leaves), subtracted from every flux measured
    # in the cell, exactly as the completeness stage does for its positions
    offset = cell_offset[band][rows]
    table = cell_table[band][rows]

    flux = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    origin = np.full(n, ORIGIN_NOT_COVERED, dtype=np.int8)
    extended = np.zeros(n, dtype=bool)
    sigma_annulus = np.full(n, np.nan)
    rel = np.full(n, np.nan)
    n_demoted = n_unreliable = 0

    with pacs_fit.open_tile(config, record) as tile:
        # the map's registration to SESNA, as the completeness stage measured and stored it
        shift = shifts.get((BANDS[band], _tile_key(record)), (0.0, 0.0))
        # eHOPS aperture photometry for every source (HOPS/eHOPS-convention annulus sky); FNU is it minus the
        # cell's empty-sky offset, and the detection is decided on that, the statistic the completeness measures
        x, y, single, ok, floor = singleton_pass(tile, ra[rows], dec[rows], shift, sigma_cell, offset)
        sigma_annulus = single.sigma
        flux[ok] = single.flux[ok] - offset[ok]
        # an undetected source carries its cell's noise; a detected one gets the measured error below
        sigma[ok] = sigma_cell[ok]
        origin[ok] = ORIGIN_UNDETECTED
        # above the 5-sigma floor: ORIGIN 1 if the flux is reliably the source's own (RELIABILITY >= R_MIN), else
        # ORIGIN 2 with the measured flux as the bound (the flux is there, probably not the source's)
        rel[floor] = reliability(flux[floor], table[floor], r_grid[band])
        detected = floor & (rel >= R_MIN)
        unreliable = floor & ~detected
        n_unreliable = int(unreliable.sum())
        origin[detected] = ORIGIN_DETECTED
        origin[unreliable] = ORIGIN_UPPER_BOUND
        sigma[unreliable] = UPPER_LIMIT_SIGMA

        # Blends (the far-infrared survey convention), applied after, on the ORIGIN 1 sources: a source within
        # BLEND_RADIUS_BEAMS of another detection is an unseparable blend, ORIGIN 2 with its own aperture flux as the
        # upper bound, unless it is itself detected and the brightest of the detections within that radius.
        blend, n_blend, brightest = blend_rule(x, y, flux, detected, ok, tile.beam_px)
        group_id = np.where(brightest >= 0, rows[np.maximum(brightest, 0)], -1).astype(np.int32)
        n_demoted = int((blend & detected).sum())
        origin[blend] = ORIGIN_UPPER_BOUND
        sigma[blend] = UPPER_LIMIT_SIGMA

        # Extended: detections only.
        det = np.flatnonzero(origin == ORIGIN_DETECTED)
        if det.size:
            extended[det] = tile.fitted_fwhm(x[det], y[det]) > EXTENDED_BEAM_FACTOR * tile.beam_arcsec

    # a detected source's error, measured by injection: its cell's SCATTER at its flux (log-linear between the
    # flux steps)
    n_nan = 0
    det = np.flatnonzero(origin == ORIGIN_DETECTED)
    if det.size:
        sigma[det] = calibrated_sigma(scatter[det], flux_grid, flux[det])
        n_nan += int(np.isnan(sigma[det]).sum())

    # the offset subtracted from each reported flux (adding it back gives the HOPS/eHOPS convention)
    subtracted = np.where(ok, offset, np.nan)
    columns = dict(FNU=flux, SIGMA=sigma, ORIGIN=origin, LOCAL_RMS=sigma_cell.astype(np.float32), CELL_OFFSET=subtracted,
                   RELIABILITY=rel.astype(np.float32),
                   SIGMA_ANNULUS=np.asarray(sigma_annulus, np.float32),
                   N_BLEND=n_blend, BLEND_GROUP=group_id, EXTENDED=extended)
    # a source with no usable measurement keeps the cell noise only where a cell exists
    columns["LOCAL_RMS"][origin == ORIGIN_NOT_COVERED] = np.nan
    return band, map_index, rows, columns, n_nan, n_demoted, n_unreliable


# ---------------------------------------------------------------------------
# the region

def _runs(rows):
    """Half-open `(start, stop, offset)` runs of consecutive values in the
    ascending `rows` (offset = where the run begins in `rows`)."""
    if rows.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [rows.size]])
    return [(int(rows[a]), int(rows[b - 1]) + 1, int(a)) for a, b in zip(starts, stops)]


_DTYPES = dict(FNU=np.float64, SIGMA=np.float64, CELL_OFFSET=np.float64, RELIABILITY=np.float32, ORIGIN=np.int8, LOCAL_RMS=np.float32,
               N_BLEND=np.int16, BLEND_GROUP=np.int32, EXTENDED=bool, SIGMA_ANNULUS=np.float32)
_FILL = dict(FNU=np.nan, SIGMA=np.nan, CELL_OFFSET=np.nan, RELIABILITY=np.nan, ORIGIN=ORIGIN_NOT_COVERED, LOCAL_RMS=np.nan,
             N_BLEND=0, BLEND_GROUP=-1, EXTENDED=False, SIGMA_ANNULUS=np.nan)
_NAMES = dict(FNU="FNU_{b}_MJY", SIGMA="SIGMA_{b}_MJY", CELL_OFFSET="CELL_OFFSET_{b}_MJY", RELIABILITY="RELIABILITY_{b}", ORIGIN="ORIGIN_{b}", LOCAL_RMS="LOCAL_RMS_{b}_MJY",
              N_BLEND="N_BLEND_{b}", BLEND_GROUP="BLEND_GROUP_{b}", EXTENDED="EXTENDED_{b}",
              SIGMA_ANNULUS="SIGMA_ANNULUS_{b}_MJY")


def _tile_key(record):
    return ",".join(str(o) for o in record.obsids)


def write_region(out_path, names, ra, dec, records, cell_sigma, cell_offset, cell_obsid, config, n_jobs, stage=None,
                 shifts=None, cell_row=None, scatter=None, flux_grid=None, cell_table=None, r_grid=None):
    """Measures every source on its cell's map and writes the product at
    `out_path`. `cell_sigma[b]` (n,) float64 is the source's cell noise (NaN:
    no cell, not covered) and `cell_obsid[b]` (n,) bytes the cell's map, the
    comma-joined observation ids of the `records` entry that holds it;
    `cell_row[b]` (n,) is the source's row in the completeness product (-1: none) and
    `scatter[b]` (n_cell, n_plane) that product's measured aperture error per cell and flux plane (mJy);
    `cell_table[b]` (n,) the source's reliability table and `r_grid[b]` the tables. Returns the per-band origin
    counts."""
    n = names.size
    key_to_map = {}
    for i, rec in enumerate(records):
        key_to_map.setdefault((rec.band, _tile_key(rec)), i)
    pieces = []
    for b, band in enumerate(BANDS):
        sel = np.flatnonzero(np.isfinite(cell_sigma[b]))
        if sel.size == 0:
            continue
        for key in np.unique(cell_obsid[b][sel]):
            i = key_to_map.get((band, key.decode()))
            if i is None:
                raise ValueError("pacs_photometry: completeness cell map %r is not in the %s index" % (key, band))
            rows = sel[cell_obsid[b][sel] == key]
            pieces.append((b, i, rows, scatter[b][cell_row[b][rows]]))
    obsid_len = max([len(_tile_key(r)) for r in records] + [1])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    counts = {}
    with h5py.File(out_path, "w") as f:
        build_module.write_dataset(f, "NAME", names, *REGISTRY[("photometry_pacs_source", "NAME")])
        dsets = {}
        for b, band in enumerate(BANDS):
            for col, pattern in _NAMES.items():
                name = pattern.format(b=band)
                dsets[b, col] = build_module.write_dataset(
                    f, name, None, *REGISTRY[("photometry_pacs_source", name)],
                    shape=(n,), dtype=_DTYPES[col], fillvalue=_FILL[col], chunks=(min(n, 65536),) if n else None)
            name = "OBSID_" + band
            dsets[b, "OBSID"] = build_module.write_dataset(
                f, name, None, *REGISTRY[("photometry_pacs_source", name)],
                shape=(n,), dtype="S%d" % obsid_len, fillvalue=b"", chunks=(min(n, 65536),) if n else None)
            counts[band] = {code: 0 for code in (1, 2, 90, 91)}
            counts[band][ORIGIN_NOT_COVERED] = n
        fn_args = dict(config=config, records=records, ra=ra, dec=dec, cell_sigma=np.vstack(cell_sigma),
                       cell_offset=np.vstack(cell_offset), cell_table=np.vstack(cell_table), r_grid=r_grid,
                       shifts=shifts or {}, flux_grid=flux_grid)
        fn = functools.partial(_measure_tile, **fn_args)
        n_nan = {band: 0 for band in BANDS}
        demoted = {band: 0 for band in BANDS}
        unreliable = {band: 0 for band in BANDS}
        for band_index, map_index, rows, cols, nan_sigma, n_dem, n_unrel in pieces_module.iterate(fn, pieces, n_jobs, stage=stage, unit="map tiles"):
            band = BANDS[band_index]
            n_nan[band] += nan_sigma
            demoted[band] += n_dem
            unreliable[band] += n_unrel
            obsid = _tile_key(records[map_index]).encode()
            for start, stop, off in _runs(rows):
                width = stop - start
                for col, data in cols.items():
                    dsets[band_index, col][start:stop] = data[off:off + width]
                dsets[band_index, "OBSID"][start:stop] = obsid
            counts[band][ORIGIN_NOT_COVERED] -= rows.size
            for code in (1, 2, 90, 91):
                counts[band][code] += int(np.sum(cols["ORIGIN"] == code))
    for band in BANDS:
        print("  %s: ORIGIN 1 %d; ORIGIN 2 %d: %d above the floor but unreliable (RELIABILITY < %.1f), %d blends of "
              "which %d were reliable detections on their own" % (band, counts[band][ORIGIN_DETECTED],
                                                                  counts[band][ORIGIN_UPPER_BOUND], unreliable[band], R_MIN,
                                                                  counts[band][ORIGIN_UPPER_BOUND] - unreliable[band],
                                                                  demoted[band]), flush=True)
        if n_nan[band]:
            print("  %s: %d detected sources have no measured photometric error in their cell (SIGMA NaN)"
                  % (band, n_nan[band]), flush=True)
    return counts


def _read_cells(config, region, n_source):
    """The completeness stage's per-source cell noise, cell offset, map and cell row, per band, and the
    cells' measured photometric errors: `(cell_sigma, cell_offset, cell_obsid, shifts, cell_row,
    scatter, flux_grid, cell_table, r_grid)`, lists over BANDS (`shifts` a dict, `flux_grid` the one grid in
    mJy). A source with `SOURCE_ROW` -1 has no cell (noise NaN, offset 0)."""
    path = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
    sigma = [np.full(n_source, np.nan) for _ in BANDS]
    offset = [np.zeros(n_source) for _ in BANDS]
    table = [np.full(n_source, -1, np.int64) for _ in BANDS]
    r_grid = [np.ones((0, logf_edges().size - 1)) for _ in BANDS]
    row = [np.full(n_source, -1, dtype=np.int64) for _ in BANDS]
    shifts = {}
    obsid = [np.full(n_source, b"", dtype="S1") for _ in BANDS]
    scatter = [np.zeros((0, 1), np.float32) for _ in BANDS]
    flux_grid = None
    if not os.path.exists(path):
        raise FileNotFoundError("pacs_photometry needs the completeness product first: %s" % path)
    with h5py.File(path, "r") as f:
        source_row = f["SOURCE_ROW"][:]
        for b, band in enumerate(BANDS):
            if "CELL_SIGMA_%s_MJY" % band not in f:
                continue
            grid = 1000.0 * 10.0 ** (-f["MAG_GRID_" + band][:].astype(np.float64) / 2.5) * f["CURVE_" + band].attrs["ZERO_POINT_JY"]
            flux_grid = grid if flux_grid is None else flux_grid
            cell_sigma = f["CELL_SIGMA_%s_MJY" % band][:]
            if cell_sigma.size == 0:
                continue
            rows = source_row[:, b]
            have = rows >= 0
            cell_obs = f["CELL_OBSID_%s" % band][:]
            sigma[b][have] = cell_sigma[rows[have]]
            offset[b][have] = f["CELL_OFFSET_%s_MJY" % band][:][rows[have]]
            table[b][have] = f["CELL_TABLE_" + band][:][rows[have]]
            r_grid[b] = f["RELIABILITY_GRID_" + band][:]
            row[b] = np.where(have, rows, -1).astype(np.int64)
            scatter[b] = f["SCATTER_%s_MJY" % band][:]
            for key, sh in zip(f["MAP_OBSID_" + band][:], f["MAP_SHIFT_%s_ARCSEC" % band][:]):
                shifts[(band, key.decode())] = (float(sh[0]), float(sh[1]))
            obsid[b] = np.where(have, cell_obs[np.where(have, rows, 0)], b"")
    return sigma, offset, obsid, shifts, row, scatter, flux_grid, table, r_grid


def _build_one_region(config, region, st):
    src_path = product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(src_path, "r") as f:
        names = f["NAME"][:]
        ra = f["RA_DEG"][:].astype(np.float64)
        dec = f["DEC_DEG"][:].astype(np.float64)
    out_path = product_path(config, "sky/derived", "pacs", "photometry", "source", region=region)
    records = read_index(config, region)
    if records:
        cell_sigma, cell_offset, cell_obsid, shifts, cell_row, scatter, flux_grid, cell_table, r_grid = _read_cells(
            config, region, names.size)
    else:
        cell_sigma = [np.full(names.size, np.nan) for _ in BANDS]
        cell_offset = [np.zeros(names.size) for _ in BANDS]
        cell_table = [np.full(names.size, -1, np.int64) for _ in BANDS]
        r_grid = [np.ones((0, logf_edges().size - 1)) for _ in BANDS]
        shifts = {}
        cell_obsid = [np.full(names.size, b"", dtype="S1") for _ in BANDS]
        cell_row = [np.full(names.size, -1, dtype=np.int64) for _ in BANDS]
        scatter = [np.zeros((0, 1), np.float32) for _ in BANDS]
        flux_grid = None
    t0 = time.time()
    counts = write_region(out_path, names, ra, dec, records, cell_sigma, cell_offset, cell_obsid,
                          config, config.n_jobs, stage=st, shifts=shifts, cell_row=cell_row, scatter=scatter,
                          flux_grid=flux_grid, cell_table=cell_table, r_grid=r_grid)
    wall = time.time() - t0
    numbers = {"sources": int(names.size), "sources_per_s": names.size / max(wall, 1e-9)}
    for band in BANDS:
        for code, label in ((1, "detected"), (2, "bound"), (90, "undetected"), (91, "uncovered")):
            numbers["%s_%s" % (band, label)] = counts[band][code]
    return out_path, numbers


def build(config, regions=None):
    """Writes each region's `photometry_pacs_source` product (all-91 where no
    PACS map covers the region). Regions run one after another; inside a
    region the map tiles are the pieces."""
    names = regions or [r.name for r in regions_module.REGIONS]
    t0 = time.time()
    for name in names:
        with progress_module.Stage("sky.derived.pacs_photometry", name) as st:
            path, numbers = _build_one_region(config, name, st)
            st.done(path, **numbers)
    progress_module.summary("sky.derived.pacs_photometry", len(names), t0)


# ---------------------------------------------------------------------------
# acceptance (printed, by function call; nothing is written)

#: HOPS (Fischer et al. 2016, ApJS 224, 5) 70 micron flux of the sources, one row per HOPS target, in mJy, with
#: the target's position, read from the VizieR tables of the HOPS download
HOPS_FILE = "J_ApJS_224_5.vot"
HOPS_MATCH_ARCSEC = 2.0
#: the bright-source comparison of the flux convention (C4)
C4_MIN_F70_MJY = 200.0
#: the bins of HOPS 70 micron flux, mJy, of the detection comparison (C5)
C5_EDGES_MJY = (40.0, 70.0, 100.0, 200.0, 500.0, 1000.0, np.inf)


def _read_hops(config):
    """`(ra deg, dec deg, F70 mJy)` of the HOPS targets with a measured 70 micron flux (flag 1 or 2) and a
    position in the HOPS tables."""
    from astropy.io.votable import parse
    path = os.path.join(config.data_root, "sky", "download", "hops", HOPS_FILE)
    tables = {t.name: t.array for t in parse(path).resources[0].tables}
    pos = {}
    for name in ("table1", "table7", "table8", "table9"):
        a = tables["J/ApJS/224/5/" + name]
        for h, r, d in zip(a["HOPS"], a["RAJ2000"], a["DEJ2000"]):
            pos[int(h)] = (float(r), float(d))
    t2 = tables["J/ApJS/224/5/table2"]
    f70 = np.asarray(t2["F70"].filled(np.nan), float) * 1000.0
    keep = np.isfinite(f70) & np.array([int(h) in pos for h in t2["HOPS"]])
    ra = np.array([pos[int(h)][0] for h in t2["HOPS"][keep]])
    dec = np.array([pos[int(h)][1] for h in t2["HOPS"][keep]])
    return ra, dec, f70[keep]


def _match_hops(config, region):
    """The HOPS targets matched to the region's SESNA sources within HOPS_MATCH_ARCSEC (nearest source):
    `(hops index, source row)` arrays, plus the HOPS arrays."""
    ra, dec, f70 = _read_hops(config)
    s_ra, s_dec = _region_positions(config, region)
    tree = cKDTree(_unit_vectors(s_ra, s_dec))
    chord = 2.0 * np.sin(np.radians(HOPS_MATCH_ARCSEC / 3600.0) / 2.0)
    dist, row = tree.query(_unit_vectors(ra, dec), distance_upper_bound=chord)
    found = np.isfinite(dist)
    return np.flatnonzero(found), row[found], f70


def _region_positions(config, region):
    with h5py.File(product_path(config, "catalog", "sesna", "sources", "source", region=region), "r") as f:
        return f["RA_DEG"][:].astype(np.float64), f["DEC_DEG"][:].astype(np.float64)


def _c4_ratio(config, region, path):
    """Median (our 70 micron flux / HOPS's) over the HOPS targets of at least C4_MIN_F70_MJY that the product
    at `path` detects (ORIGIN 1), and how many. The flux is put on HOPS's convention (no empty-sky offset
    subtracted): FNU plus the product's CELL_OFFSET where it stores one (an earlier product subtracted none)."""
    h, row, f70 = _match_hops(config, region)
    with h5py.File(path, "r") as f:
        flux, origin = f["FNU_P70_MJY"][:][row], f["ORIGIN_P70"][:][row]
        if "CELL_OFFSET_P70_MJY" in f:
            flux = flux + f["CELL_OFFSET_P70_MJY"][:][row]
    use = (f70[h] >= C4_MIN_F70_MJY) & (origin == ORIGIN_DETECTED) & np.isfinite(flux)
    return float(np.median(flux[use] / f70[h][use])) if use.any() else float("nan"), int(use.sum()), int(((f70[h] >= C4_MIN_F70_MJY)).sum())


def acceptance(config, region, reference_product=None):
    """Prints C4 and C5 of PACS_COMPLETENESS_PLAN.md for the region's finished photometry product (and the
    completeness product it read). C4: the median of our 70 micron flux over HOPS's for HOPS targets of at
    least 200 mJy, next to the same number for `reference_product` (the earlier product's path), the plan's bar
    being agreement within 2 %. C5: the HOPS targets binned by their 70 micron flux: the fraction detected
    (ORIGIN 1) among those the band covers, against the mean of the completeness curve at their SESNA
    source's cell and their flux, with binomial errors. Nothing is written."""
    path = product_path(config, "sky/derived", "pacs", "photometry", "source", region=region)
    ratio, n, n_all = _c4_ratio(config, region, path)
    print("C4 %s: median (ours + CELL_OFFSET) / HOPS at 70 micron (HOPS's convention), HOPS targets >= %g mJy detected by us: %.4f (%d of %d targets)"
          % (region, C4_MIN_F70_MJY, ratio, n, n_all))
    if reference_product is not None:
        old, n_old, _ = _c4_ratio(config, region, reference_product)
        print("   the reference product: %.4f (%d targets); difference %+.2f %% (bar: within 2 %%: %s)"
              % (old, n_old, 100 * (ratio / old - 1), "pass" if abs(ratio / old - 1) <= 0.02 else "FAIL"))
    h, row, f70 = _match_hops(config, region)
    with h5py.File(path, "r") as f:
        o70 = f["ORIGIN_P70"][:][row]
    bright = f70[h] >= C4_MIN_F70_MJY
    print("   HOPS targets >= %g mJy by final 70 micron ORIGIN (sanity, not a bar): %s"
          % (C4_MIN_F70_MJY, "  ".join("%d: %d" % (c, int((o70[bright] == c).sum())) for c in (1, 2, 90, 91))))
    comp = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
    with h5py.File(path, "r") as f:
        origin = f["ORIGIN_P70"][:][row]
    with h5py.File(comp, "r") as f:
        cell = f["SOURCE_ROW"][:][row, 0]
        curve = f["CURVE_P70"][:]
        grid = 1000.0 * f["CURVE_P70"].attrs["ZERO_POINT_JY"] * 10.0 ** (-f["MAG_GRID_P70"][:].astype(float) / 2.5)
    flux = f70[h]
    covered = (origin == ORIGIN_DETECTED) | (origin == ORIGIN_UNDETECTED)
    # the curve at each target's cell, interpolated in log flux (the grid runs bright to faint)
    expected = np.full(h.size, np.nan)
    for i in np.flatnonzero(covered & (cell >= 0)):
        expected[i] = np.interp(np.log10(flux[i]), np.log10(grid[::-1]), curve[cell[i], ::-1] / 600.0)
    for lo, hi in zip(C5_EDGES_MJY[:-1], C5_EDGES_MJY[1:]):
        k = covered & np.isfinite(expected) & (flux >= lo) & (flux < hi)
        if not k.any():
            print("C5 %s: HOPS 70 micron %g-%g mJy: none covered" % (region, lo, hi))
            continue
        obs = float(np.mean(origin[k] == ORIGIN_DETECTED))
        pred = float(np.mean(expected[k]))
        err = float(np.sqrt(np.sum(expected[k] * (1 - expected[k]))) / k.sum())
        print("C5 %s: HOPS 70 micron %g-%g mJy: %d targets, detected %.3f +- %.3f, curve predicts %.3f +- %.3f"
              % (region, lo, hi, int(k.sum()), obs, np.sqrt(max(obs * (1 - obs), 1e-9) / k.sum()), pred, err))


if __name__ == "__main__":
    run(build)
