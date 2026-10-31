"""Artificial-source injection, pooling and the completeness/error estimator of a band, block by block.

Depends on: grid, psf, mosaic, detect. Replaces (pacs_completeness.py): snr_bin, snr_grid, canonical_grid, Injector,
_inject_batches, _inject_pairs, _injection_state, _map_first, map_sizing, injection_blocks, injection_pairs, tile_order,
_chunks, part_counts, round_task_list, _real_check (the computation stays here, `_real_map_ratios`; the CHECKS rows are
checks.py's), cell_centres_arcsec (-> CellGrid.centres_arcsec), pool_matrices, pooled_counts, pooled_median_scatter(_one),
_snr_brackets, _at_flux, curve_on_flux_grid, error_on_flux_grid, flux90, pool_product, cell_errors, pool_dir,
_write_pool_table, pool_tasks, _pool_block, _assemble_tile, _band_dict (the injection half), _empty_band.

Method (unchanged from main 0683690). Artificial point sources are injected in signal-to-noise: at S/N plane k each
source's flux is SNR_GRID[k] x the CELL_SIGMA of the cell it lands in. The work of a band is cut by its cell grid into
injection blocks of INJECTION_BLOCK_LINES cell lines (FROZEN 3); every (block, plane, half, batch) has its own rng,
default_rng([seed_key, plane, 2 * batch + half, block]) (FROZEN 2-3), so the records do not depend on the cut into pieces,
on K or on n_jobs. A batch is measured tile by tile: its positions are grouped by the 256-row band of the grid they fall
in (the old strip), and inside a band by the tile column; each group is measured on a private Window of its tile plus the
margin mg (the reach of a patch, an aperture and annulus at a peak up to the recovery radius away, and the peak search),
holding the real pixels plus every patch of the band that reaches the window, pasted in the band's order. A worker
therefore holds one tile window, whatever the grid's size, and the pixels it measures are those of the old full-width
strip.

The estimator. CURVE at flux F is the pooled recovered fraction per S/N plane read at F / the cell's own CELL_SIGMA (the
neighbourhood's shape, the cell's own scale); BIAS and SCATTER the pooled median and 1.4826 x MAD of dflux / CELL_SIGMA
over the covered injections farther than BLEND_RADIUS_BEAMS from every detection, read the same way and times the cell's
CELL_SIGMA. How cells pool is a PARAMETER, `PoolingRule` (FIX_SPEC.md sec 8 and 12; science M1, M3):

  pooling_class    "none" (main's: every cell within the radius), "window" (only cells with CELL_SIGMA within x1.5 of
                   the target's), "window_edge" (window AND the same coverage-edge class: within-cell coverage max/min >
                   2), "window_gradient" (window, and the injections split by their own |ln r| <= / > 0.03, r = the
                   aperture box's mean coverage over the cell's mean coverage: two SCATTER/BIAS per cell)
  short_cell_rule  "nan" (main's: SCATTER NaN where x3 holds fewer than MIN_INJECTIONS far injections on a plane), "widen"
                   (keep widening x4, x5, ... inside the class until MIN_INJECTIONS; when no factor reaches it, the
                   all-cell pool: the edge class over the map without the noise window where the rule has an edge
                   class, else the whole map; never NaN, never a dropped read)
  curve_class      "cell" (main's), "gradient" (two curves per coverage-edge cell, its injections split by |ln r| <= / >
                   0.03, read by the source's own r; other cells carry their cell curve in both)

The pooling class and the short rule apply to BIAS/SCATTER only; CURVE pools as main's (in S/N, no noise window) unless
curve_class says otherwise. `PoolingRule.legacy()` reproduces main 0683690 exactly and is the identity gate's setting.

Store (transient, the band's work directory `records_dir`): injection records `inject_r<r>_s<k>of<K>.hdf5` (one table per
shard: row int32, plane int8, half int8, batch int32, block int32, near bool, recovered bool, dflux float32, flux float32,
det_flux float32, det_dist float32, lnr float32; the designated round-0 piece of each block also carries the per-cell
coverage table cell_row, cell_cov_mean, cell_cov_max, cell_cov_min; scalars seconds_inject, seconds_aperture,
n_detected, check), written by the caller (stages.write_record; here `_write_record` until round 3); the band's record
table `table.hdf5` (both halves sorted by row with per-row offsets, far-injection counts per (row, plane), the per-row
coverage, the held-out half and the statistics; `build_table`) that the pooling blocks read; no pickles.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from sesnaimpute.definitions import COMPLETENESS_STORAGE_SCALE as STORAGE_SCALE
from sesnaimpute.definitions import PACS_APERTURES as APERTURES
from sesnaimpute.definitions import PACS_BLEND_RADIUS_BEAMS as BLEND_RADIUS_BEAMS
from sesnaimpute.definitions import PACS_MAX_EXTRA_BATCHES as MAX_EXTRA_BATCHES
from sesnaimpute.definitions import PACS_MIN_INJECTIONS as MIN_INJECTIONS
from sesnaimpute.definitions import PACS_POOL_ARCSEC as POOL_ARCSEC
from sesnaimpute.definitions import PACS_POOL_WIDENINGS as POOL_WIDENINGS
from sesnaimpute.definitions import PACS_SNR_BIN_EDGES as SNR_BIN_EDGES
from sesnaimpute.sky.derived.pacs import detect, mosaic
from sesnaimpute.sky.derived.pacs.detect import BandDetection, CellStats, Kernel, Peaks
from sesnaimpute.sky.derived.pacs.grid import CELL_ARCSEC, INJECTION_BLOCK_LINES, TILE, CellGrid
from sesnaimpute.sky.derived.pacs.mosaic import Mosaic, Window

# ----------------------------------------------------------------------------------------------------------------------
# constants (moved unchanged from pacs_completeness.py; deleted there in round 3)

#: PACS has no Vega system: zero point 1 Jy, m = -2.5 log10(F / 1 Jy)
ZERO_POINT_JY = 1.0
#: ONE flux grid per band for the whole survey, bright to faint
MAG_MIN = -3.5
MAG_MAX = 5.5
MAG_STEP = 0.25
MAD_TO_SIGMA = 1.4826
#: no two artificial sources of a batch closer than the injected patch's half-width plus the annulus's outer radius,
#: drawn at one per this many squared separations of valid area
SEPARATION_AREAS = 2.0
#: log10 S/N of the injection planes, SNR_LOG_MIN to SNR_LOG_MAX in SNR_LOG_STEP (0.1 dex, the flux grid's 0.25 mag)
SNR_LOG_MIN = -0.5
SNR_LOG_MAX = 3.0
SNR_LOG_STEP = 0.1
#: the first round of batches aims at this multiple of MIN_INJECTIONS in an interior cell's pool
INTERIOR_SAFETY = 1.3
#: the recovered fraction FLUX90 is read at
FLUX90_FRACTION = 0.9
#: a block's jobs are cut into pieces of about this many of the block's cells per (plane, half) job
INJECTION_PIECE_ROWS = 64
#: the pooling is cut into square blocks of POOL_BLOCK_CELLS cells
POOL_BLOCK_CELLS = 8
#: the real-map report: this many empty-sky positions per half
REAL_CHECK_POSITIONS = 200
#: the real-map report's seed plane
REAL_CHECK_SEED = 99999
#: the recovery radius is MATCH_SIGMAS x the detection's position scatter at the injection's S/N (detect's constant;
#: R3: swap to detect.MATCH_SIGMAS once detect.py defines it)
MATCH_SIGMAS = 3.0
#: the catalogue's detection threshold in S/N (R3: swap to the detect/mosaic constant)
DETECTION_SNR = 5.0
#: the pooling parameters ruled by the science track (FIX_SPEC sec 8, 12; M1, M3)
NOISE_WINDOW = 1.5
EDGE_RATIO = 2.0
GRADIENT_BOUNDARY = 0.03
#: the all-cell pool is used with fewer than MIN_INJECTIONS far injections only when the whole map holds fewer (a
#: median and a MAD need two values); never reached on the gate set
MAP_MIN = 2
#: the S/N planes pooled (36)
N_PLANES = int(round((SNR_LOG_MAX - SNR_LOG_MIN) / SNR_LOG_STEP)) + 1

#: the record columns and their dtypes
RECORD_COLUMNS = (("row", np.int32), ("plane", np.int8), ("half", np.int8), ("batch", np.int32), ("block", np.int32),
                  ("near", np.bool_), ("recovered", np.bool_), ("dflux", np.float32), ("flux", np.float32),
                  ("det_flux", np.float32), ("det_dist", np.float32), ("lnr", np.float32))
#: the per-cell coverage columns of the designated round-0 piece of each block
CELL_COLUMNS = (("cell_row", np.int64), ("cell_cov_mean", np.float64), ("cell_cov_max", np.float64),
                ("cell_cov_min", np.float64))
#: the held-out half kept for the acceptance (C1, C2)
HELD_KEYS = ("row", "plane", "near", "recovered", "dflux", "flux", "det_flux", "det_dist", "lnr")


# ----------------------------------------------------------------------------------------------------------------------
# the S/N planes and the flux grid

def snr_grid() -> np.ndarray:
    """The injection S/N planes: 10 ** arange(SNR_LOG_MIN, SNR_LOG_MAX + step/2, SNR_LOG_STEP) (`snr_grid`)."""
    n = int(round((SNR_LOG_MAX - SNR_LOG_MIN) / SNR_LOG_STEP)) + 1
    return 10.0 ** (SNR_LOG_MIN + SNR_LOG_STEP * np.arange(n))


def flux_grid() -> Tuple[np.ndarray, np.ndarray]:
    """(mag, flux mJy): the one survey-wide flux grid, m = -2.5 log10(F / 1 Jy) from MAG_MIN to MAG_MAX in MAG_STEP,
    bright to faint (`canonical_grid`)."""
    n = int(round((MAG_MAX - MAG_MIN) / MAG_STEP)) + 1
    mag = MAG_MIN + MAG_STEP * np.arange(n)
    return mag, 1000.0 * ZERO_POINT_JY * 10.0 ** (-mag / 2.5)


def snr_bin(snr: np.ndarray) -> np.ndarray:
    """The C2 S/N bin of each value (np.digitize on definitions.PACS_SNR_BIN_EDGES)."""
    return np.digitize(np.asarray(snr, float), SNR_BIN_EDGES)


def injection_pairs() -> List[Tuple[int, int]]:
    """The (S/N plane, half) pairs of the injections, in the order of the count columns (col = 2 plane + half)."""
    return [(p, half) for p in range(snr_grid().size) for half in (0, 1)]


# ----------------------------------------------------------------------------------------------------------------------
# small helpers

class _CellTable:
    """A per-row value looked up by cell id (NaN for a cell that holds no row): what the old n_cells-sized arrays
    were, without an n_cells-sized array (a search on the ascending row ids)."""

    def __init__(self, ids: np.ndarray, values: np.ndarray):
        self.ids = ids
        self.values = values

    def __getitem__(self, cid):
        cid = np.asarray(cid, np.int64)
        n = self.ids.size
        if n == 0:
            return np.full(cid.shape, np.nan)
        k = np.minimum(np.searchsorted(self.ids, cid), n - 1)
        return np.where(self.ids[k] == cid, self.values[k], np.nan)


def _rows_of(ids: np.ndarray, cid: np.ndarray) -> np.ndarray:
    """The row of each cell id (ids ascending), -1 where the cell holds no row."""
    cid = np.asarray(cid, np.int64)
    n = ids.size
    if n == 0:
        return np.full(cid.shape, -1, np.int64)
    k = np.minimum(np.searchsorted(ids, cid), n - 1)
    return np.where((cid >= 0) & (ids[k] == cid), k, -1)


def _f32(v) -> np.ndarray:
    """The float32 value as stored in the product, as float64 (the injections' fluxes and offsets are computed from
    the stored float32 CELL_SIGMA / CELL_OFFSET, as main's `tile_inputs`)."""
    return np.asarray(v, np.float32).astype(np.float64)


def _cell_coverage(cov_layer, cells: CellGrid, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cell of `ids`: (mean coverage over all its pixels, max, min over its positive pixels), the cell's pixels
    [ceil(ix step_x - 0.5), ceil((ix + 1) step_x - 0.5)) x the same in y, clipped to the grid; NaN read as 0. `cov_layer`
    is read one cell line at a time (`window`). The coverage-edge class is max / min > EDGE_RATIO (M1, M3)."""
    ids = np.asarray(ids, np.int64)
    ny, nx = cells.shape
    mean, cmax, cmin = (np.full(ids.size, np.nan) for _ in range(3))
    if ids.size == 0:
        return mean, cmax, cmin
    iy, ix = cells.line_column(ids)
    for line in np.unique(iy):
        sel = np.flatnonzero(iy == line)
        ya = int(np.ceil(line * cells.step_y - 0.5))
        yb = min(int(np.ceil((line + 1) * cells.step_y - 0.5)), ny)
        cx = ix[sel]
        xa = np.clip(np.ceil(cx * cells.step_x - 0.5).astype(np.int64), 0, nx)
        xb = np.minimum(np.ceil((cx + 1) * cells.step_x - 0.5).astype(np.int64), nx)
        c0, c1 = int(xa.min()), int(xb.max())
        if yb <= ya or c1 <= c0:
            continue
        a = np.nan_to_num(np.asarray(cov_layer.window(ya, yb, c0, c1), np.float64))
        pos = a > 0
        csum = a.sum(0)
        cmx = a.max(0)
        cmn = np.where(pos, a, np.inf).min(0)
        for j, (u, v) in enumerate(zip(xa - c0, xb - c0)):
            if v <= u:
                continue
            mean[sel[j]] = csum[u:v].sum() / ((yb - ya) * (v - u))
            cmax[sel[j]] = cmx[u:v].max()
            cmin[sel[j]] = cmn[u:v].min()
    return mean, cmax, cmin


def _box_mean(cov: np.ndarray, y0: int, x0: int, ix: np.ndarray, iy: np.ndarray, rb: int, shape) -> np.ndarray:
    """The mean coverage over the (2 rb + 1)^2 box centred on grid pixel (ix, iy), from the window `cov` whose first
    pixel is (y0, x0); pixels outside the grid count as zero (the box of the aperture photometry, M3's r)."""
    ny, nx = shape
    a = np.nan_to_num(np.asarray(cov, np.float64))
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    xa = np.clip(np.clip(ix - rb, 0, nx) - x0, 0, a.shape[1])
    xb = np.clip(np.clip(ix + rb + 1, 0, nx) - x0, 0, a.shape[1])
    ya = np.clip(np.clip(iy - rb, 0, ny) - y0, 0, a.shape[0])
    yb = np.clip(np.clip(iy + rb + 1, 0, ny) - y0, 0, a.shape[0])
    return (s[yb, xb] - s[ya, xb] - s[yb, xa] + s[ya, xa]) / float((2 * rb + 1) ** 2)


# ----------------------------------------------------------------------------------------------------------------------
# the injections

@dataclass(frozen=True)
class InjectionTask:
    """One piece of an injection round: `block` (FROZEN 3 block id), `rows` (first row, last row + 1 of the block's
    covered cells), `jobs` [(plane, half, batch seeds, rows or None)], `check` (run the real-map check on this piece)."""
    block: int
    rows: Tuple[int, int]
    jobs: Tuple[tuple, ...]
    check: bool


class Injector:
    """The injections of one band's mosaic over the covered cells (`BandDetection.cells`): candidate positions per
    row's cell (rng draws), validity, thinning to SEPARATION_AREAS x separation^2, fluxes at an S/N plane x the cell's
    CELL_SIGMA, and the measurement of each batch tile by tile on private Windows (paste, aperture, window_peaks,
    recovered = nearest peak within the recovery radius and brighter than any real source there; near = within
    BLEND_RADIUS_BEAMS x beam of a real detection). Per-worker memory: one tile window + per-row arrays (no
    n_cells-sized array: the old row_of_cell is a search on the sorted row ids).
    `layers`: the band's detection layers (detect.open_detection) that `detect.window_peaks` reads (contract change:
    the contract's signature has no way to reach them).
    Identity: `pacs_completeness.Injector` (pacs_completeness.py:832-1017): bit-identical records for the same
    (seed_key, plane, batch, half, block)."""

    def __init__(self, m: Mosaic, detection: BandDetection, seed_key: int, layers: Optional[dict] = None) -> None:
        self.m = m
        self.seed_key = int(seed_key)
        self.layers = layers
        self.cells = detection.cell_grid
        self.ids = np.asarray(detection.cells.ids, np.int64)
        n = self.ids.size
        self.sigma = _f32(detection.cells.sigma)
        self.offset = _f32(detection.cells.offset)
        self.m_sigma = _f32(detection.cells.m_sigma)
        self.m_offset = _f32(detection.cells.m_offset)
        # the per-cell tables window_peaks reads (indexed by cell id)
        self.offset_cell = _CellTable(self.ids, self.offset)
        self.m_sigma_cell = _CellTable(self.ids, self.m_sigma)
        self.m_offset_cell = _CellTable(self.ids, self.m_offset)
        self.shape = tuple(int(v) for v in _shape_of(m))
        self.band = _band_of(m)
        self.pixel_arcsec = float(m.pixel_arcsec)
        self.h = int(m.injection_half_width())
        self.r_out_px = APERTURES[self.band][2] / self.pixel_arcsec
        self.separation_px = self.h + self.r_out_px
        self.pos_k_px = float(detection.pos_k_arcsec) / self.pixel_arcsec
        peaks = detection.peaks
        self.det_xy = np.c_[np.asarray(peaks.x, float), np.asarray(peaks.y, float)]
        self.det_flux = np.asarray(peaks.fnu, float)
        self.n_detected = int(self.det_flux.size)
        self.det_tree = cKDTree(self.det_xy) if self.n_detected else None
        self.cell_px = float(self.cells.step_x * self.cells.step_y)
        # the per-row coverage (M3's r): computed per cell line when first needed
        self._cov = np.full((3, n), np.nan)
        self._cov_done = np.zeros(n, bool)
        self._ops = _operations(m)
        self.valid_fraction = 1.0
        self.valid_fraction = (float(self._valid(*self._draw(np.random.default_rng(0), np.arange(n), 20000)).mean())
                               if n else 0.0)

    @property
    def n_rows(self) -> int:
        """The band's rows (covered cells)."""
        return int(self.ids.size)

    # -- positions
    def _draw(self, rng, rows, n):
        """n positions, each uniform inside the cell of a randomly chosen row of `rows`."""
        x0, x1, y0, y1 = self.cells.bounds(self.ids[rows[rng.integers(0, rows.size, n)]])
        return x0 + (x1 - x0) * rng.random(n), y0 + (y1 - y0) * rng.random(n)

    def _valid(self, x, y):
        """Positions inside a row's cell, on a pixel of usable data (the reader's error mask)."""
        cid = self.cells.cell_ids(x, y)
        ok = cid >= 0
        ok[ok] = _rows_of(self.ids, cid[ok]) >= 0
        ok[ok] = np.isfinite(self.m.pixel_sigma(x[ok], y[ok]))
        return ok

    def _thin(self, x, y, limit):
        """Greedy selection, in order, of at most `limit` positions no two closer than the separation."""
        n = x.size
        if n == 0:
            return np.zeros(0, np.int64)
        pairs = cKDTree(np.c_[x, y]).query_pairs(self.separation_px, output_type="ndarray").reshape(-1, 2)
        both = np.concatenate([pairs, pairs[:, ::-1]])
        both = both[np.argsort(both[:, 0], kind="stable")]
        edges = np.searchsorted(both[:, 0], np.arange(n + 1))
        removed = np.zeros(n, bool)
        chosen = []
        for i in range(n):
            if removed[i]:
                continue
            chosen.append(i)
            if len(chosen) >= limit:
                break
            removed[both[edges[i]:edges[i + 1], 1]] = True
        return np.asarray(chosen, np.int64)

    def positions(self, rng, rows):
        """One batch's positions over the valid pixels of the cells of `rows`, one per SEPARATION_AREAS squared
        separations: (x, y, distance in pixels to the nearest PACS source of the un-injected map, inf if none)."""
        target = max(int(round(rows.size * self.cell_px * self.valid_fraction / (SEPARATION_AREAS * self.separation_px ** 2))), 1)
        x, y = self._draw(rng, rows, int(2.5 * target) + 8)
        ok = self._valid(x, y)
        x, y = x[ok], y[ok]
        keep = self._thin(x, y, target)
        x, y = x[keep], y[keep]
        return x, y, self._peak_distance(x, y)

    def _peak_distance(self, x, y):
        if self.det_tree is None:
            return np.full(np.asarray(x).size, np.inf)
        d, _ = self.det_tree.query(np.c_[x, y])
        return d

    def flux_at(self, x, y, snr):
        """The flux (mJy) of an artificial source of S/N `snr` at each (x, y): snr x CELL_SIGMA of its cell."""
        return snr * self.sigma[_rows_of(self.ids, self.cells.cell_ids(x, y))]

    def recovery_radius_px(self, snr, x, y):
        """MATCH_SIGMAS x the detection's position scatter at S/N `snr`, between one pixel and half the local beam."""
        half_beam = 0.5 * self.m.beam_px_at(x, y)
        if not np.isfinite(self.pos_k_px):
            return half_beam
        return np.clip(MATCH_SIGMAS * self.pos_k_px / max(snr, 1e-3), 1.0, half_beam)

    def batch(self, rng: np.random.Generator, rows: np.ndarray, snr: float) -> dict:
        """One batch: positions (`positions`), fluxes, the measurement (`measure_positions`). Returns the record
        columns of the module docstring (row .. lnr; plane/half/batch/block are the caller's) plus n and seconds."""
        x, y, _d = self.positions(rng, rows)
        return self.measure_positions(x, y, self.flux_at(x, y, snr), radius_px=self.recovery_radius_px(snr, x, y))

    # -- coverage (M3's r)
    def _row_coverage(self, rows: np.ndarray) -> np.ndarray:
        """(3, n) cell mean / max / min coverage of `rows`, computed per cell line on first use."""
        rows = np.asarray(rows, np.int64)
        todo = np.unique(rows[~self._cov_done[rows]])
        if todo.size:
            self._cov[:, todo] = np.vstack(_cell_coverage(self._ops.coverage, self.cells, self.ids[todo]))
            self._cov_done[todo] = True
        return self._cov[:, rows]

    # -- the measurement
    def measure_positions(self, x: np.ndarray, y: np.ndarray, flux: np.ndarray, radius_px: Optional[np.ndarray] = None
                          ) -> dict:
        """Injects `flux` (mJy, scalar or one per position) at the grid pixels (x, y), measures the forced aperture
        flux at each and runs the blind detection around each (`detect.window_peaks` within `radius_px`, half the
        local beam if None). Grouped by 256-row band (the old strip) and, inside a band, by tile column: each group
        on a private Window of rows [band - mg, band + 256 + mg) and columns [column - mg, column + 256 + mg), holding
        every patch of the band that reaches it, pasted in the band's order (the pixels of the old full-width strip).
        RECOVERED: a peak within the radius, and where a source of the un-injected map lies within it the injection is
        the brighter. Returns dict(row, near, recovered, dflux (forced flux - cell offset - injected), flux, det_flux,
        det_dist, lnr (ln of the aperture box's mean coverage over the cell's mean coverage)) for the injections on
        covered data, n, seconds. (`pacs_completeness.Injector.measure_positions`, pacs_completeness.py:942; the
        distance to the nearest real source, an argument there, is computed here.)"""
        ops = self._ops
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = x.size
        out = {k: np.zeros(0, dt) for k, dt in RECORD_COLUMNS if k not in ("plane", "half", "batch", "block")}
        out.update(n=0, seconds=dict(inject=0.0, aperture=0.0))
        if n == 0:
            return out
        d = self._peak_distance(x, y)
        beam = self.m.beam_px_at(x, y)
        r = 0.5 * beam if radius_px is None else np.broadcast_to(np.asarray(radius_px, float), x.shape)
        flux = np.broadcast_to(np.asarray(flux, float), x.shape)
        rows = _rows_of(self.ids, self.cells.cell_ids(x, y))
        offset = self.offset[rows]
        near = d < BLEND_RADIUS_BEAMS * beam
        brighter = np.ones(n, bool)
        if self.det_tree is not None:
            lists = self.det_tree.query_ball_point(np.c_[x, y], r)
            for k, l in enumerate(lists):
                if l:
                    brighter[k] = flux[k] >= self.det_flux[l].max()
        flux_out, covered = np.full(n, np.nan), np.zeros(n, bool)
        found, det_flux, det_dist = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan)
        box = np.full(n, np.nan)
        t_inject = t_ap = 0.0
        ny, nx = self.shape
        rb = int(np.ceil(self.r_out_px)) + 1
        mg = self.h + int(np.ceil(self.r_out_px)) + 1 + int(np.ceil(r.max())) + 2
        band = np.floor(y / TILE).astype(np.int64)
        for b in np.unique(band):
            k = np.flatnonzero(band == b)
            a0, a1 = max(int(b) * TILE - mg, 0), min((int(b) + 1) * TILE + mg, ny)
            t0 = time.time()
            pcx, pcy, patches = ops.template_patches(x[k], y[k], flux[k])
            side_y, side_x = patches.shape[1:]
            t_inject += time.time() - t0
            col = np.floor(x[k] / TILE).astype(np.int64)
            for c in np.unique(col):
                kk = k[col == c]
                c0, c1 = max(int(c) * TILE - mg, 0), min((int(c) + 1) * TILE + mg, nx)
                t0 = time.time()
                win = ops.window(a0, a1, c0, c1)
                reach = (pcx < c1) & (pcx + side_x > c0) & (pcy < a1) & (pcy + side_y > a0)
                ops.paste(win, pcx[reach], pcy[reach], patches[reach])
                t_inject += time.time() - t0
                t0 = time.time()
                a = ops.aperture(x[kk], y[kk], win)
                flux_out[kk], covered[kk] = a.flux, a.covered
                f_, _xc, _yc, df_, dd_ = ops.peaks(self, x[kk], y[kk], r[kk], flux[kk], win)
                found[kk], det_flux[kk], det_dist[kk] = f_, df_, dd_
                cov = ops.coverage.window(a0, a1, c0, c1)
                box[kk] = _box_mean(cov, a0, c0, np.rint(x[kk]).astype(np.int64), np.rint(y[kk]).astype(np.int64),
                                    rb, self.shape)
                t_ap += time.time() - t0
        recovered = covered & found & brighter
        keep = covered
        with np.errstate(all="ignore"):
            lnr = np.log(box / self._row_coverage(rows)[0])
        out.update(row=rows[keep].astype(np.int32), near=near[keep], recovered=recovered[keep],
                   dflux=(flux_out - offset - flux).astype(np.float32)[keep],
                   flux=flux.astype(np.float32)[keep], det_flux=det_flux.astype(np.float32)[keep],
                   det_dist=det_dist.astype(np.float32)[keep], lnr=lnr.astype(np.float32)[keep], n=int(keep.sum()),
                   seconds=dict(inject=t_inject, aperture=t_ap))
        return out


def _shape_of(m) -> Tuple[int, int]:
    return tuple(m.grid.shape) if hasattr(m, "grid") and hasattr(m.grid, "shape") else tuple(m.shape)


def _band_of(m) -> str:
    return m.band if hasattr(m, "band") else m.record.band


class _Operations:
    """The mosaic and detection operations the injection uses (window, template_patches, paste, aperture, peaks,
    coverage), bound once per Injector: the contract's mosaic/detect functions, or the round-2 adapter to the old
    modules (`_OldMosaic`; R3: swap)."""

    def __init__(self, m):
        self.m = m
        self.window = m.window
        self.template_patches = m.template_patches
        self.aperture = lambda x, y, win: m.aperture(x, y, window=win)
        self.coverage = m.layers["coverage"]
        self.paste = mosaic.paste

    def peaks(self, inj: "Injector", x, y, r, flux, win):
        """(found, xc, yc, fnu, distance) of the nearest peak within r (contract change: the contract's docstring
        names the fifth return value snr; the identity is the old distance)."""
        return detect.window_peaks(self.m, inj.layers, x, y, r, inj.m_offset_cell, inj.m_sigma_cell, inj.offset_cell,
                                   flux=flux, window=win)


def _operations(m) -> "_Operations":
    return m.operations() if isinstance(m, _OldMosaic) else _Operations(m)   # R3: swap (drop the adapter branch)


def _real_map_ratios(inj: Injector, flux_mjy: float, half: int, rows: Optional[np.ndarray] = None) -> np.ndarray:
    """The catalogue's measurement in its own regime: REAL_CHECK_POSITIONS empty-sky positions (farther than one
    beam from every detected source) of `rows`, the aperture flux after injecting `flux_mjy` minus before, over
    `flux_mjy` (`pacs_completeness._real_check`). The window spans the positions plus the patch and aperture reach;
    every position's patch is pasted, as the old whole-map injection did. Returns the ratios."""
    ops = inj._ops
    rng = np.random.default_rng([inj.seed_key, REAL_CHECK_SEED, half])
    x, y, d = inj.positions(rng, np.arange(inj.ids.size) if rows is None else rows)
    far = d >= BLEND_RADIUS_BEAMS * inj.m.beam_px_at(x, y)
    x, y = x[far][:REAL_CHECK_POSITIONS], y[far][:REAL_CHECK_POSITIONS]
    if x.size == 0:
        return np.zeros(0)
    ny, nx = inj.shape
    mg = inj.h + int(np.ceil(inj.r_out_px)) + 3
    y0, y1 = max(int(np.floor(y.min())) - mg, 0), min(int(np.ceil(y.max())) + mg + 1, ny)
    x0, x1 = max(int(np.floor(x.min())) - mg, 0), min(int(np.ceil(x.max())) + mg + 1, nx)
    before = ops.aperture(x, y, ops.window(y0, y1, x0, x1))
    win = ops.window(y0, y1, x0, x1)
    pcx, pcy, patches = ops.template_patches(x, y, np.full(x.size, float(flux_mjy)))
    ops.paste(win, pcx, pcy, patches)
    after = ops.aperture(x, y, win)
    ok = before.covered & after.covered
    return (after.flux[ok] - before.flux[ok]) / flux_mjy


_INJECTOR = {}


def _injector(m: Mosaic, detection: BandDetection, layers: Optional[dict] = None) -> Injector:
    """The process-local Injector of a band (seed_key 1: the band's one mosaic is map 0, FROZEN 2), built on the first
    piece a worker takes, kept (one band at a time)."""
    key = (_mosaic_key(m), int(np.asarray(detection.cells.ids).size))
    if _INJECTOR.get("key") != key:
        _INJECTOR.clear()
        _INJECTOR.update(key=key, injector=Injector(m, detection, 1, layers=layers))
    return _INJECTOR["injector"]


def _mosaic_key(m) -> str:
    for name in ("key", "dir", "directory"):
        v = getattr(m, name, None)
        if isinstance(v, str):
            return v
    return str(id(m))


def first_batches(config, m: Mosaic, detection: BandDetection) -> int:
    """The batches of round 0 per (plane, half) and block: ceil(INTERIOR_SAFETY x MIN_INJECTIONS / interior pool
    density) (`_map_first`, pacs_completeness.py:1170)."""
    inj = _injector(m, detection)
    W = pool_matrices(detection.cell_grid, inj.ids, np.zeros(inj.ids.size, np.int8),
                      PoolingRule.legacy(), widenings=POOL_WIDENINGS[:1])[0]
    interior = (inj.cell_px * inj.valid_fraction / (SEPARATION_AREAS * inj.separation_px ** 2)) * float(np.median(np.diff(W.indptr)))
    return int(np.ceil(INTERIOR_SAFETY * MIN_INJECTIONS / max(interior, 1e-9)))


def injection_blocks(ids: np.ndarray, nx: int) -> List[Tuple[int, int, int]]:
    """The injection blocks of a band whose rows hold the cells `ids` (ascending): [(block, first row, last row + 1)]."""
    blk = np.asarray(ids, np.int64) // nx // INJECTION_BLOCK_LINES
    b, first = np.unique(blk, return_index=True)
    return [(int(b[i]), int(first[i]), int(first[i + 1]) if i + 1 < b.size else int(blk.size)) for i in range(b.size)]


def _chunks(jobs, n_rows):
    """A block's jobs cut into pieces of about INJECTION_PIECE_ROWS of the block's cells per job."""
    k = int(min(len(jobs), max(1, np.ceil(n_rows / INJECTION_PIECE_ROWS))))
    return [[jobs[i] for i in c] for c in np.array_split(np.arange(len(jobs)), k) if c.size]


def round_tasks(detection: BandDetection, first: int, r: int, load: Callable[[int], np.ndarray]) -> List[InjectionTask]:
    """The ordered pieces of round `r` ([] when the band is complete): round 0 = `first` batches in every block for every
    (plane, half); each later round one more batch per pair over the cells within the pooling radius of a cell whose
    pooled count (all earlier rounds) is short of MIN_INJECTIONS, up to MAX_EXTRA_BATCHES. `load(q)`: the (rows x pairs)
    counts of round q's records (`record_counts`; dense or sparse). Identity: `pacs_completeness.round_task_list`
    (pacs_completeness.py:1258) -- same tasks in the same order (the old (tile index, inputs) fields drop: one mosaic per
    band)."""
    pairs = injection_pairs()
    ids = np.asarray(detection.cells.ids, np.int64)
    blocks = injection_blocks(ids, detection.cell_grid.nx)
    W = pool_matrices(detection.cell_grid, ids, np.zeros(ids.size, np.int8), PoolingRule.legacy(),
                      widenings=POOL_WIDENINGS[:1])[0]
    count = np.zeros((ids.size, len(pairs)))
    tasks = []
    for i, blk in enumerate(blocks):
        jobs = [(p, half, tuple(range(first)), None) for p, half in pairs]
        for j, c in enumerate(_chunks(jobs, blk[2] - blk[1])):
            tasks.append(InjectionTask(blk[0], (blk[1], blk[2]), tuple(c), i == 0 and j == 0))
    for extra in range(min(r, MAX_EXTRA_BATCHES + 1)):
        if not tasks:
            return []
        q = load(extra)
        count += q.toarray() if hasattr(q, "toarray") else np.asarray(q)
        tasks = []
        if extra == MAX_EXTRA_BATCHES:
            break
        short = (W @ count) < MIN_INJECTIONS
        act = short.any(axis=0)
        if not act.any():
            continue
        reach = (W @ short.astype(float)) > 0
        for b, r0, r1 in blocks:
            jobs = [(p, half, (first + extra,), r0 + np.flatnonzero(reach[r0:r1, i]))
                    for i, (p, half) in enumerate(pairs) if act[i] and reach[r0:r1, i].any()]
            for c in (_chunks(jobs, r1 - r0) if jobs else []):
                tasks.append(InjectionTask(b, (r0, r1), tuple(c), False))
    return tasks if r <= MAX_EXTRA_BATCHES else []


def inject_task(config, m: Mosaic, detection: BandDetection, task: InjectionTask, layers: Optional[dict] = None) -> dict:
    """One piece's injections (`_inject_pairs`): the record columns (RECORD_COLUMNS) concatenated over its jobs in job
    then batch order, seconds_inject / seconds_aperture, n_detected, `check` (the real-map report's mean ratio per half
    when `task.check`, else empty; the CHECKS rows are checks.py's), and, on the block's designated round-0 piece (its
    first job is (plane 0, half 0, seeds from 0, the whole block)), the block's per-cell coverage (CELL_COLUMNS).
    Every value is a numpy array: the caller writes the dict as one record file. `layers`: see `Injector`."""
    inj = _injector(m, detection, layers)
    snr = snr_grid()
    r0, r1 = task.rows
    out = {k: [] for k, _ in RECORD_COLUMNS}
    seconds = dict(inject=0.0, aperture=0.0)
    check = np.zeros(0)
    if task.check:
        f_check = 10.0 * DETECTION_SNR * float(np.median(inj.sigma))
        check = np.array([float(v.mean()) if v.size else np.nan
                          for v in (_real_map_ratios(inj, f_check, half, np.arange(r0, r1)) for half in (0, 1))])
    for p, half, seeds, rows in task.jobs:
        rr = np.arange(r0, r1) if rows is None else np.asarray(rows, np.int64)
        for seed in seeds:
            res = inj.batch(np.random.default_rng([inj.seed_key, p, 2 * seed + half, task.block]), rr, snr[p])
            k = res["row"].size
            for name, _ in RECORD_COLUMNS:
                if name == "plane":
                    out[name].append(np.full(k, p, np.int8))
                elif name == "half":
                    out[name].append(np.full(k, half, np.int8))
                elif name == "batch":
                    out[name].append(np.full(k, seed, np.int32))
                elif name == "block":
                    out[name].append(np.full(k, task.block, np.int32))
                else:
                    out[name].append(res[name])
            for s in seconds:
                seconds[s] += res["seconds"][s]
    rec = {k: (np.concatenate(out[k]).astype(dt) if out[k] else np.zeros(0, dt)) for k, dt in RECORD_COLUMNS}
    first = task.jobs[0] if task.jobs else None
    if first is not None and first[0] == 0 and first[1] == 0 and first[2][:1] == (0,) and first[3] is None:
        rows_b = np.arange(r0, r1)
        mean, cmax, cmin = inj._row_coverage(rows_b)
        rec.update(cell_row=rows_b.astype(np.int64), cell_cov_mean=mean, cell_cov_max=cmax, cell_cov_min=cmin)
    rec.update(seconds_inject=np.float64(seconds["inject"]), seconds_aperture=np.float64(seconds["aperture"]),
               n_detected=np.int64(inj.n_detected), check=check)
    return rec


def record_counts(records: dict, n_rows: int) -> np.ndarray:
    """(rows x pairs) injection counts of a set of records (`part_counts`, dense int64 per band); col = 2 plane + half."""
    col = 2 * np.asarray(records["plane"], np.int64) + np.asarray(records["half"], np.int64)
    key = np.asarray(records["row"], np.int64) * (2 * N_PLANES) + col
    return np.bincount(key, minlength=n_rows * 2 * N_PLANES).reshape(n_rows, 2 * N_PLANES).astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# record files (R3: swap to stages.write_record / read_record: stages is to the right of inject in the import order)

def _write_record(path: str, tables: Dict[str, np.ndarray], attrs: Optional[dict] = None) -> dict:
    """An HDF5 record file written atomically (temporary name with pid, fsync, os.replace), one contiguous dataset per
    key at the root, then its receipt `<path>.receipt.json` {size, sha256} (also atomic). Returns the receipt."""
    tmp = "%s.tmp%d" % (path, os.getpid())
    with h5py.File(tmp, "w") as f:
        for k, v in tables.items():
            f.create_dataset(k, data=np.asarray(v))
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    with open(tmp, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    receipt = dict(path=os.path.basename(path), size=os.path.getsize(path), sha256=h.hexdigest())
    rtmp = "%s.receipt.json.tmp%d" % (path, os.getpid())
    with open(rtmp, "w") as f:
        json.dump(receipt, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(rtmp, path + ".receipt.json")
    return receipt


def _read_record(path: str, keys: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in (f.keys() if keys is None else keys) if k in f}


def _record_files(records_dir: str) -> List[str]:
    """The injection record files of every round, round by round, shard by shard; raises when a round's shards are
    mixed or missing."""
    found = {}
    for p in glob.glob(os.path.join(records_dir, "inject_r*_s*of*.hdf5")):
        mm = re.fullmatch(r"inject_r(\d+)_s(\d+)of(\d+)\.hdf5", os.path.basename(p))
        if mm:
            found.setdefault(int(mm.group(1)), {}).setdefault(int(mm.group(3)), set()).add(int(mm.group(2)))
    out = []
    for r in sorted(found):
        if len(found[r]) != 1:
            raise ValueError("%s: round %d has shardings %s mixed" % (records_dir, r, sorted(found[r])))
        (k, have), = found[r].items()
        if have != set(range(k)):
            raise ValueError("%s: round %d is missing shard(s) %s of %d" % (records_dir, r, sorted(set(range(k)) - have), k))
        out += [os.path.join(records_dir, "inject_r%d_s%dof%d.hdf5" % (r, s, k)) for s in range(k)]
    return out


TABLE_NAME = "table.hdf5"


def build_table(records_dir: str, n_row: int, sigma: Optional[np.ndarray] = None) -> str:
    """The band's record table, written once before its pooling blocks (contract change: a public step the stages
    call; `pool_block` builds it when absent): both halves' records sorted by row (a stable sort of the records taken in
    (half, plane, block, batch) order, so the table does not depend on the order the pieces finished) with `offsets`
    (n_row + 1); far-injection counts per (row, plane) of each half and gradient class (`far_h<half>_g<class>`); the
    per-row coverage (from the designated round-0 pieces); the held-out half (`held/<key>`, in the same order) and the
    statistics (attrs); with `sigma` (the rows' CELL_SIGMA), the map-wide fallbacks of the "widen" rule (`fallback`:
    per record set (both halves, build half) x gradient class (0, 1, any) x edge class (0, 1, any) x plane: n, median,
    1.4826 MAD of the far injections' dflux / CELL_SIGMA). Written to a temporary name and renamed (any number of
    callers). Returns its path. (`pacs_completeness._write_pool_table`.)"""
    path = os.path.join(records_dir, TABLE_NAME)
    if os.path.exists(path):
        return path
    files = _record_files(records_dir)
    cols = {k: [] for k, _ in RECORD_COLUMNS}
    cov = np.full((3, n_row), np.nan)
    seconds = dict(inject=0.0, aperture=0.0)
    check, n_detected = None, 0
    for p in files:
        rec = _read_record(p)
        for k, dt in RECORD_COLUMNS:
            cols[k].append(np.asarray(rec[k], dt))
        if "cell_row" in rec and rec["cell_row"].size:
            cov[:, rec["cell_row"]] = np.vstack([rec["cell_cov_mean"], rec["cell_cov_max"], rec["cell_cov_min"]])
        seconds["inject"] += float(rec.get("seconds_inject", 0.0))
        seconds["aperture"] += float(rec.get("seconds_aperture", 0.0))
        n_detected = int(rec.get("n_detected", n_detected))
        if check is None and np.asarray(rec.get("check", np.zeros(0))).size:
            check = np.asarray(rec["check"], float)
    cols = {k: (np.concatenate(cols[k]) if cols[k] else np.zeros(0, dt)) for k, dt in RECORD_COLUMNS}
    entry = np.lexsort((cols["batch"], cols["block"], cols["plane"], cols["half"]))
    cols = {k: v[entry] for k, v in cols.items()}
    held_sel = cols["half"] == 1
    held = {k: cols[k][held_sel] for k in HELD_KEYS}
    order = np.argsort(cols["row"], kind="stable")
    table = {k: cols[k][order] for k in ("row", "plane", "half", "near", "recovered", "dflux", "lnr")}
    count = np.bincount(table["row"].astype(np.int64), minlength=n_row)
    offsets = np.concatenate([[0], np.cumsum(count)]).astype(np.int64)
    far = ~table["near"]
    gcls = _gradient_class(table["lnr"])
    counts = {}
    for half in (0, 1):
        for g in (0, 1):
            s = far & (table["half"] == half) & (gcls == g)
            key = table["row"][s].astype(np.int64) * N_PLANES + table["plane"][s].astype(np.int64)
            counts["far_h%d_g%d" % (half, g)] = np.bincount(key, minlength=n_row * N_PLANES).reshape(n_row, N_PLANES).astype(np.int32)
    snr = snr_grid()
    near_plane = np.zeros((2, snr.size, 2))
    for half in (0, 1):
        s = cols["half"] == half
        nr = cols["near"][s]
        pl = cols["plane"][s].astype(np.int64)
        near_plane[half, :, 0] = np.bincount(pl[nr], minlength=snr.size)
        near_plane[half, :, 1] = np.bincount(pl[nr & cols["recovered"][s]], minlength=snr.size)
    stats = dict(n_row=int(n_row), n_detected=n_detected, n_build=int((~held_sel).sum()), n_held=int(held_sel.sum()),
                 seconds=seconds, n_by_path=dict(far=int((~cols["near"]).sum()), near=int(cols["near"].sum())),
                 check=[float(v) for v in (check if check is not None else [np.nan, np.nan])])
    tmp = "%s.tmp%d" % (path, os.getpid())
    with h5py.File(tmp, "w") as f:
        for k, v in table.items():
            f.create_dataset(k, data=v)
        f.create_dataset("offsets", data=offsets)
        for k, v in counts.items():
            f.create_dataset(k, data=v)
        f.create_dataset("cell_cov", data=cov)
        f.create_dataset("near_plane", data=near_plane)
        if sigma is not None:
            f.create_dataset("fallback", data=_map_fallbacks(table, _f32(sigma), cov))
        g = f.create_group("held")
        for k, v in held.items():
            g.create_dataset(k, data=v)
        f.attrs["stats"] = json.dumps(stats)
    with open(tmp, "rb+") as fh:
        os.fsync(fh.fileno())
    try:
        os.replace(tmp, path)
    except OSError:
        if not os.path.exists(path):
            raise
    return path


def _map_fallbacks(table: dict, sigma: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """(2 sets, 3 gradient, 3 edge, N_PLANES, 3) n, median, 1.4826 MAD of z = dflux / CELL_SIGMA over the far records
    of the whole map: sets both halves / build half; gradient class 0, 1, any; edge class (cell coverage max/min >
    EDGE_RATIO) 0, 1, any. The last steps of the "widen" rule: the class over the map, then the map."""
    out = np.full((2, 3, 3, N_PLANES, 3), np.nan)
    far = ~table["near"]
    row = table["row"].astype(np.int64)
    z = table["dflux"].astype(np.float64) / sigma[row]
    gcl = _gradient_class(table["lnr"])
    with np.errstate(invalid="ignore", divide="ignore"):
        ecl = (cov[1] / cov[2] > EDGE_RATIO).astype(np.int8)[row]
    plane = table["plane"].astype(np.int64)
    for si, halves in enumerate(((0, 1), (0,))):
        s_set = far & np.isin(table["half"], halves)
        for gi in range(3):
            s_g = s_set if gi == 2 else s_set & (gcl == gi)
            for ei in range(3):
                sel = s_g if ei == 2 else s_g & (ecl == ei)
                pl, zz = plane[sel], z[sel]
                for p in range(N_PLANES):
                    v = zz[pl == p]
                    out[si, gi, ei, p, 0] = v.size
                    if v.size:
                        med = 0.5 * (np.partition(v, (v.size - 1) // 2)[(v.size - 1) // 2] + np.partition(v, v.size // 2)[v.size // 2])
                        a = np.abs(v - med)
                        mad = 0.5 * (np.partition(a, (a.size - 1) // 2)[(a.size - 1) // 2] + np.partition(a, a.size // 2)[a.size // 2])
                        out[si, gi, ei, p, 1:] = med, MAD_TO_SIGMA * mad
    return out


def held_and_stats(detection: BandDetection, records_dir: str) -> Tuple[dict, dict]:
    """(held-out half, statistics) of a band, what `assemble_band` takes: the record table built when absent
    (`build_table` with the band's CELL_SIGMA, so the "widen" rule's map-wide values are present), then read."""
    build_table(records_dir, np.asarray(detection.cells.ids).size, detection.cells.sigma)
    return read_held(records_dir)


def read_held(records_dir: str) -> Tuple[dict, dict]:
    """(held-out half, statistics) of a band from its record table (`build_table`): what `assemble_band` takes
    (contract change: public; the old held/meta came from meta.pkl)."""
    with h5py.File(os.path.join(records_dir, TABLE_NAME), "r") as f:
        held = {k: f["held"][k][()] for k in f["held"]}
        stats = json.loads(f.attrs["stats"])
        stats["near_plane"] = f["near_plane"][()]
    return held, stats


class _Table:
    """The record table of a band opened read-only, its datasets memory-mapped (contiguous HDF5 datasets)."""

    def __init__(self, path: str):
        self.path = path
        self.f = h5py.File(path, "r")
        self.data = {}
        for k in self.f.keys():
            if isinstance(self.f[k], h5py.Dataset):
                self.data[k] = self._map(self.f[k])

    def _map(self, ds):
        off = ds.id.get_offset()
        if off is None or ds.chunks is not None or ds.size == 0:
            return ds[()]
        return np.memmap(self.path, mode="r", dtype=ds.dtype, offset=off, shape=ds.shape)

    def rows(self, keys: Sequence[str], rows: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """The records of `rows` (ascending, distinct) as dict of arrays and each record's index into `rows`."""
        off = self.data["offsets"]
        rows = np.asarray(rows, np.int64)
        n = np.asarray(off[rows + 1]) - np.asarray(off[rows])
        # contiguous runs of rows read as slices
        out = {}
        runs = _runs(rows)
        for k in keys:
            arr = self.data[k]
            out[k] = (np.concatenate([np.asarray(arr[off[s]:off[e + 1]]) for s, e in runs])
                      if runs else np.zeros(0, arr.dtype))
        return out, np.repeat(np.arange(rows.size), n)


def _runs(rows: np.ndarray) -> List[Tuple[int, int]]:
    """[(first, last)] of the runs of consecutive values of ascending `rows`."""
    if rows.size == 0:
        return []
    br = np.flatnonzero(np.diff(rows) != 1)
    starts = np.concatenate([[rows[0]], rows[br + 1]])
    ends = np.concatenate([rows[br], [rows[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


_TABLE = {}


def _open_table(records_dir: str, n_row: int, sigma: Optional[np.ndarray] = None) -> _Table:
    path = build_table(records_dir, n_row, sigma)
    if _TABLE.get("path") != path:
        _TABLE.clear()
        _TABLE.update(path=path, table=_Table(path))
    return _TABLE["table"]


# ----------------------------------------------------------------------------------------------------------------------
# pooling and the estimator

POOLING_CLASSES = ("none", "window", "window_edge", "window_gradient")
#: PRODUCTION estimator (owner's ruling 2026-10-31): pool only cells with CELL_SIGMA within x1.5 of the target's
#: (the window restores the ruled calibration of noisy cells on their own residuals, FIX_SPEC sec 8, 12); the
#: coverage-edge class is NOT used (measured to add <= 0.013 to the noisy-bin scatter ratio)
PRODUCTION_POOLING_CLASS = "window"
#: PRODUCTION short-cell rule (owner 2026-10-31): widen inside the window until MIN_INJECTIONS, else the all-cell pool;
#: SCATTER is never NaN and the acceptance never drops a read (FIX_SPEC sec 12 item 2)
PRODUCTION_SHORT_CELL_RULE = "widen"
#: PRODUCTION curve (owner 2026-10-31): two curves per coverage-edge cell by |ln r| <= / > 0.03, each source read on
#: its own r (M3: recovery depends on |ln r| in edge cells, -0.10 at the transition); other cells one curve
PRODUCTION_CURVE_CLASS = "gradient"
SHORT_RULES = ("nan", "widen")
CURVE_CLASSES = ("cell", "gradient")


@dataclass(frozen=True)
class PoolingRule:
    """How a cell's injections are pooled (FIX_SPEC.md sec 8, 12): `radius_arcsec`, `widenings`, `min_injections`;
    `noise_window` (pool only cells with CELL_SIGMA within x this of the target's; None = no window), `edge_class`
    (pool coverage-edge cells -- within-cell coverage max/min > `edge_ratio` -- only with edge cells), `short_rule`
    ("nan": SCATTER NaN where short, main's; "widen": widen beyond the last widening inside the class until
    min_injections; when no factor reaches it, the all-cell pool (the edge class over the map without the window
    where the rule has an edge class, else the whole map)), `gradient_class`
    (the errors' injections split by |ln r| <= / > `gradient_boundary`: two SCATTER/BIAS per cell), `curve_class`
    ("cell", main's; "gradient": two curves per coverage-edge cell by |ln r|). The three named parameters of the
    brief map onto these fields through `named` (contract change: gradient_class, gradient_boundary, curve_class and
    `named` are additions)."""
    radius_arcsec: float
    widenings: Tuple[int, ...]
    min_injections: int
    noise_window: Optional[float] = None
    edge_class: bool = False
    edge_ratio: float = 2.0
    short_rule: str = "nan"
    gradient_class: bool = False
    gradient_boundary: float = GRADIENT_BOUNDARY
    curve_class: str = "cell"

    @classmethod
    def legacy(cls) -> "PoolingRule":
        """Main 0683690's rule: definitions.PACS_POOL_ARCSEC, PACS_POOL_WIDENINGS, PACS_MIN_INJECTIONS, no window,
        no class, short = NaN. The identity gate's setting."""
        return cls(float(POOL_ARCSEC), tuple(POOL_WIDENINGS), int(MIN_INJECTIONS))

    @classmethod
    def named(cls, pooling_class: str = "none", short_cell_rule: str = "nan", curve_class: str = "cell") -> "PoolingRule":
        """The rule from the brief's parameter names: pooling_class in POOLING_CLASSES, short_cell_rule in SHORT_RULES,
        curve_class in CURVE_CLASSES; ("none", "nan", "cell") = `legacy()`."""
        if pooling_class not in POOLING_CLASSES or short_cell_rule not in SHORT_RULES or curve_class not in CURVE_CLASSES:
            raise ValueError("pooling_class %r, short_cell_rule %r, curve_class %r" % (pooling_class, short_cell_rule, curve_class))
        return cls(float(POOL_ARCSEC), tuple(POOL_WIDENINGS), int(MIN_INJECTIONS),
                   noise_window=None if pooling_class == "none" else NOISE_WINDOW,
                   edge_class=pooling_class == "window_edge", edge_ratio=EDGE_RATIO, short_rule=short_cell_rule,
                   gradient_class=pooling_class == "window_gradient", gradient_boundary=GRADIENT_BOUNDARY,
                   curve_class=curve_class)

    @classmethod
    def production(cls) -> "PoolingRule":
        """The owner's ruled estimator (PRODUCTION_*): window, widen, gradient curve. `legacy()` stays for the
        identity gate only."""
        return cls.named(PRODUCTION_POOLING_CLASS, PRODUCTION_SHORT_CELL_RULE, PRODUCTION_CURVE_CLASS)

    @property
    def is_legacy(self) -> bool:
        return (self.noise_window is None and not self.edge_class and self.short_rule == "nan"
                and not self.gradient_class and self.curve_class == "cell")


def _gradient_class(lnr: np.ndarray, boundary: float = GRADIENT_BOUNDARY) -> np.ndarray:
    """0 where |ln r| <= boundary (or r unknown), 1 beyond (M3)."""
    with np.errstate(invalid="ignore"):
        return (np.abs(np.asarray(lnr, float)) > boundary).astype(np.int8)


def pool_classes(detection: BandDetection, rule: PoolingRule, edge_ratio_per_cell: Optional[np.ndarray] = None
                 ) -> np.ndarray:
    """Each row's pooling class under `rule`: 1 for a coverage-edge cell (`edge_ratio_per_cell` = within-cell coverage
    max/min > rule.edge_ratio) when rule.edge_class, else 0 (all 0 for the legacy rule)."""
    n = np.asarray(detection.cells.ids).size
    if not rule.edge_class or edge_ratio_per_cell is None:
        return np.zeros(n, np.int8)
    with np.errstate(invalid="ignore"):
        return (np.asarray(edge_ratio_per_cell, float) > rule.edge_ratio).astype(np.int8)


def _keep(rows_i: np.ndarray, cols_j: np.ndarray, classes: np.ndarray, rule: PoolingRule,
          sigma: Optional[np.ndarray]) -> np.ndarray:
    """The pairs (target i, member j) a rule keeps: same class, and CELL_SIGMA within the noise window (the target
    always pools with itself)."""
    k = classes[rows_i] == classes[cols_j]
    if rule.noise_window is not None and sigma is not None:
        with np.errstate(all="ignore"):
            q = sigma[cols_j] / sigma[rows_i]
        k &= ((q >= 1.0 / rule.noise_window) & (q <= rule.noise_window)) | (rows_i == cols_j)
    return k


def pool_matrices(cells: CellGrid, ids: np.ndarray, classes: np.ndarray, rule: PoolingRule,
                  widenings: Optional[Sequence[int]] = None, sigma: Optional[np.ndarray] = None) -> list:
    """Sparse (n, n) 0/1 pooling matrices, one per widening of `rule` (or `widenings`): (i, j) = 1 where cell centres
    (CellGrid.centres_arcsec) lie within radius x widening, the classes match and (rule.noise_window) sigma[j] /
    sigma[i] lies within the window; a cell pools with itself. Identity (legacy rule): `pacs_completeness.pool_matrices`.
    Contract change: `widenings` and `sigma` (the window needs the cells' CELL_SIGMA) are added keywords."""
    centres = cells.centres_arcsec(np.asarray(ids, np.int64))
    n = len(centres)
    tree = cKDTree(centres)
    classes = np.asarray(classes)
    sig = None if sigma is None else np.asarray(sigma, float)
    out = []
    for k in (rule.widenings if widenings is None else widenings):
        pairs = tree.query_pairs(k * rule.radius_arcsec, output_type="ndarray").reshape(-1, 2)
        r = np.concatenate([np.arange(n), pairs[:, 0], pairs[:, 1]])
        c = np.concatenate([np.arange(n), pairs[:, 1], pairs[:, 0]])
        if not (rule.noise_window is None and not rule.edge_class):
            keep = _keep(r, c, classes, rule, sig)
            r, c = r[keep], c[keep]
        out.append(csr_matrix((np.ones(r.size), (r, c)), shape=(n, n)))
    return out


def _pooled_counts(Ws, row, plane, recovered, n_row, n_plane, min_n):
    """Pooled injections and recovered injections per (cell, plane) on the narrowest of `Ws` reaching `min_n`, else the
    widest (`pacs_completeness.pooled_counts`)."""
    key = row.astype(np.int64) * n_plane + plane
    tot = np.bincount(key, minlength=n_row * n_plane).reshape(n_row, n_plane).astype(np.float64)
    rec = np.bincount(key[recovered], minlength=n_row * n_plane).reshape(n_row, n_plane).astype(np.float64)
    pt, pr = Ws[-1] @ tot, Ws[-1] @ rec
    for W in reversed(Ws[:-1]):
        t, r = W @ tot, W @ rec
        use = t >= min_n
        pt, pr = np.where(use, t, pt), np.where(use, r, pr)
    return pt, pr


def _pooled_median_scatter_one(W, row, value, n_row, min_n):
    """The pooled median and scatter of each row of `W` (m rows over the n_row cells)."""
    n_out = W.shape[0]
    out_med = np.full(n_out, np.nan)
    out_sc = np.full(n_out, np.nan)
    if row.size == 0:
        return out_med, out_sc
    order = np.argsort(row, kind="stable")
    r, v = row[order], value[order].astype(np.float64)
    count = np.bincount(r, minlength=n_row)
    start = np.cumsum(count) - count
    pair_c = np.repeat(np.arange(n_out), np.diff(W.indptr))
    pair_n = W.indices
    k = count[pair_n]
    total = int(k.sum())
    if total == 0:
        return out_med, out_sc
    seg = np.repeat(pair_c, k)
    within = np.arange(total) - np.repeat(np.cumsum(k) - k, k)
    vals = v[np.repeat(start[pair_n], k) + within]
    n_c = np.bincount(seg, minlength=n_out)
    first = np.cumsum(n_c) - n_c

    def segmented_median(x):
        xs = x[np.lexsort((x, seg))]
        lo = np.minimum(first + (n_c - 1) // 2, xs.size - 1)
        hi = np.minimum(first + n_c // 2, xs.size - 1)
        return np.where(n_c > 0, 0.5 * (xs[lo] + xs[hi]), np.nan)

    med = segmented_median(vals)
    mad = segmented_median(np.abs(vals - med[seg]))
    good = n_c >= min_n
    out_med[good] = med[good]
    out_sc[good] = MAD_TO_SIGMA * mad[good]
    return out_med, out_sc


def _pooled_median_scatter(Ws, row, value, n_row, min_n):
    """Per output cell, the median and 1.4826 x MAD of the pooled records' values on the narrowest of `Ws` holding at
    least `min_n` records (one number, or one per level of `Ws`); NaN where none does
    (`pacs_completeness.pooled_median_scatter`)."""
    n_out = Ws[0].shape[0]
    med = np.full(n_out, np.nan)
    sc = np.full(n_out, np.nan)
    mins = list(min_n) if np.ndim(min_n) else [min_n] * len(Ws)
    for W, mn in zip(Ws, mins):
        need = np.flatnonzero(np.isnan(med))
        if need.size == 0:
            break
        m, s_ = _pooled_median_scatter_one(W[need], row, value, n_row, mn)
        fill = np.isfinite(m)
        med[need[fill]], sc[need[fill]] = m[fill], s_[fill]
    return med, sc


def _snr_brackets(sigma_rows, flux_grid_mjy):
    """Per (row, flux plane): the two S/N planes bracketing F / CELL_SIGMA, the upper one's weight (linear in log S/N;
    held at the grid's ends) and F / CELL_SIGMA over the grid's largest S/N (> 1 only beyond the bright end)."""
    n_x = snr_grid().size
    with np.errstate(all="ignore"):
        lx = np.log10(flux_grid_mjy[None, :] / np.asarray(sigma_rows, float)[:, None])
    t = np.clip((lx - SNR_LOG_MIN) / SNR_LOG_STEP, 0.0, n_x - 1.0)
    t = np.where(np.isfinite(t), t, 0.0)
    k0 = np.minimum(np.floor(t).astype(np.int64), n_x - 2)
    w = t - k0
    beyond = np.maximum(10.0 ** (lx - SNR_LOG_MAX), 1.0)
    return k0, w, np.where(np.isfinite(beyond), beyond, 1.0)


def _at_flux(values_x, k0, w):
    """`values_x` (n_row, n_snr) read at the bracketing S/N planes of every (row, flux plane)."""
    r = np.arange(values_x.shape[0])[:, None]
    lo, hi = values_x[r, k0], values_x[r, k0 + 1]
    with np.errstate(invalid="ignore"):
        mid = (1.0 - w) * lo + w * hi
    return np.where(w <= 0.0, lo, np.where(w >= 1.0, hi, mid))


def curve_on_flux_grid(Ws: list, row: np.ndarray, plane: np.ndarray, recovered: np.ndarray, sigma_rows: np.ndarray,
                       flux_grid_mjy: np.ndarray, n_row: int) -> Tuple[np.ndarray, np.ndarray]:
    """(fraction (n_out, n_flux), pooled counts): the pooled recovered fraction per S/N plane read at F / CELL_SIGMA,
    linear in log S/N (`curve_on_flux_grid`). The rows of `Ws` are the output cells, `sigma_rows` theirs."""
    n_x = snr_grid().size
    tot, got = _pooled_counts(Ws, row, plane.astype(np.int64), recovered, n_row, n_x, MIN_INJECTIONS)
    with np.errstate(all="ignore"):
        fx = np.where(tot > 0, got / tot, np.nan)
    k0, w, _ = _snr_brackets(sigma_rows, flux_grid_mjy)
    fraction = _at_flux(fx, k0, w)
    r = np.arange(tot.shape[0])[:, None]
    count = np.where(w <= 0.0, tot[r, k0], np.where(w >= 1.0, tot[r, k0 + 1], np.minimum(tot[r, k0], tot[r, k0 + 1])))
    return np.nan_to_num(fraction, nan=0.0), count


def error_on_flux_grid(Ws: list, row: np.ndarray, plane: np.ndarray, dflux: np.ndarray, sigma_rows: np.ndarray,
                       flux_grid_mjy: np.ndarray, n_row: int, rule: PoolingRule,
                       sigma_out: Optional[np.ndarray] = None,
                       fallback: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       level_min: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(bias, scatter) (n_out, n_flux): pooled median and 1.4826 x MAD of dflux / CELL_SIGMA (the injection's own cell)
    per S/N plane, on the narrowest of `Ws` holding rule.min_injections records (the rule's ladder: under "widen" the
    caller passes the extended ladder, `pool_block`), read at F / the output row's CELL_SIGMA and times it; beyond the
    grid's brightest S/N in proportion to the flux. `level_min`: the count each level needs (default rule.min_injections
    for every level). Where no level holds enough:
    `fallback` (median, scatter) per (output row, S/N plane) (the "widen" rule's map-wide values), else NaN. Contract
    change: `fallback` and `level_min` are added keywords.
    Identity (legacy rule): `error_on_flux_grid` + `pooled_median_scatter` bit-identical."""
    n_x = snr_grid().size
    sig = np.asarray(sigma_rows, float)
    sig_out = sig if sigma_out is None else np.asarray(sigma_out, float)
    med_x = np.full((Ws[0].shape[0], n_x), np.nan)
    sc_x = np.full((Ws[0].shape[0], n_x), np.nan)
    z = dflux.astype(np.float64) / sig[row]
    for p in range(n_x):
        sel = plane == p
        med_x[:, p], sc_x[:, p] = _pooled_median_scatter(Ws, row[sel], z[sel], n_row,
                                                         rule.min_injections if level_min is None else level_min)
    if fallback is not None:
        miss = np.isnan(med_x)
        med_x[miss], sc_x[miss] = fallback[0][miss], fallback[1][miss]
    k0, w, beyond = _snr_brackets(sig_out, flux_grid_mjy)
    scale = sig_out[:, None] * beyond
    return (_at_flux(med_x, k0, w) * scale).astype(np.float32), (_at_flux(sc_x, k0, w) * scale).astype(np.float32)


def flux90(fraction: np.ndarray, flux_grid_mjy: np.ndarray, level: float = FLUX90_FRACTION) -> np.ndarray:
    """Per row, the faintest flux at which the recovered fraction still reaches `level`: from the grid's bright end,
    the plane before the first one below `level`, log-linearly interpolated to the crossing; the faintest plane's flux
    if none falls below; NaN if even the brightest does not reach it (`flux90`)."""
    n, p = fraction.shape
    below = fraction < level
    k = np.where(below.any(axis=1), below.argmax(axis=1), p)
    lf = np.log10(flux_grid_mjy)
    r = np.arange(n)
    km = np.clip(k - 1, 0, p - 1)
    kk = np.clip(k, 0, p - 1)
    with np.errstate(all="ignore"):
        t = (fraction[r, km] - level) / (fraction[r, km] - fraction[r, kk])
        inter = 10.0 ** (lf[km] + t * (lf[kk] - lf[km]))
    out = np.where(k >= p, flux_grid_mjy[-1], inter)
    return np.where(k == 0, np.nan, out)


def _pool_halo(rule: PoolingRule) -> int:
    return int(np.ceil(rule.widenings[-1] * rule.radius_arcsec / CELL_ARCSEC)) + 1


class _BandCells:
    """Per-process cell-level state of a band for the widening (the rows' centres, tree, CELL_SIGMA, classes)."""

    def __init__(self, detection: BandDetection, cov: np.ndarray, rule: PoolingRule):
        self.ids = np.asarray(detection.cells.ids, np.int64)
        self.centres = detection.cell_grid.centres_arcsec(self.ids)
        self.tree = cKDTree(self.centres) if self.ids.size else None
        self.sigma = _f32(detection.cells.sigma)
        with np.errstate(all="ignore"):
            self.edge_ratio = cov[1] / cov[2]
        self.classes = pool_classes(detection, rule, self.edge_ratio)
        lo, hi = self.centres.min(0), self.centres.max(0)
        self.diameter = float(np.hypot(*(hi - lo))) if self.ids.size else 0.0


_CELLS = {}


def _band_cells(detection, table: _Table, rule: PoolingRule) -> _BandCells:
    key = (table.path, rule)
    if _CELLS.get("key") != key:
        _CELLS.clear()
        _CELLS.update(key=key, cells=_BandCells(detection, np.asarray(table.data["cell_cov"]), rule))
    return _CELLS["cells"]


def _ladder_extension(bc: _BandCells, out_rows: np.ndarray, short: np.ndarray, counts: List[np.ndarray],
                      rule: PoolingRule) -> Tuple[list, dict]:
    """The levels beyond the rule's widenings for the output rows short on some plane (`short` (n_out, n_planes), any
    record set of `counts`): x(w+1), x(w+2), ... inside the class (and noise window) until every short plane of the row
    reaches min_injections in every set, or the radius spans the map. Returns ([csr (n_out, n_row) per level],
    statistics: rows completed at each level, rows left to the all-cell pool (`_fallback_values`)). A level's matrix holds rows only for the rows still short when it is reached."""
    n_row = bc.ids.size
    levels, hist = [], {}
    need = short.copy()
    k = rule.widenings[-1]
    cs = [np.asarray(c, np.float64) for c in counts]
    while need.any():
        k += 1
        act = np.flatnonzero(need.any(1))
        lists = bc.tree.query_ball_point(bc.centres[out_rows[act]], k * rule.radius_arcsec)
        cnt = np.array([len(l) for l in lists])
        cols_j = np.concatenate([np.asarray(l, np.int64) for l in lists]) if cnt.sum() else np.zeros(0, np.int64)
        rows_l = np.repeat(act, cnt)
        keep = _keep(out_rows[rows_l], cols_j, bc.classes, rule, bc.sigma)
        W = csr_matrix((np.ones(int(keep.sum())), (rows_l[keep], cols_j[keep])), shape=(out_rows.size, n_row))
        st = np.zeros_like(need)
        for c in cs:
            st |= (W @ c) < rule.min_injections
        done = need.any(1) & ~(need & st).any(1)
        hist["x%d" % k] = int(done.sum())
        need = need & st
        levels.append(W)
        if k * rule.radius_arcsec > bc.diameter:
            break
    hist["to_map_fallback_rows"] = int(need.any(1).sum())
    hist["to_map_fallback_planes"] = int(need.sum())
    return levels, hist


def _fallback_values(fb: np.ndarray, set_index: int, g: Optional[int], edge_out: Optional[np.ndarray], n_out: int,
                     min_n: int) -> Tuple[np.ndarray, np.ndarray]:
    """(median, scatter) (n_out, N_PLANES) of the "widen" rule's all-cell pool from the table's map-wide values: the
    row's edge class over the map (no window) where the rule has an edge class and it holds min_n, else the whole map
    (down to MAP_MIN injections where the whole map holds fewer than min_n); NaN only for a map with < MAP_MIN."""
    gi = 2 if g is None else g
    med = np.full((n_out, N_PLANES), np.nan)
    sc = np.full((n_out, N_PLANES), np.nan)
    whole = fb[set_index, gi, 2]
    ok_w = whole[:, 0] >= MAP_MIN            # the whole map: min_n wherever it holds them, at least MAP_MIN
    med[:, ok_w], sc[:, ok_w] = whole[ok_w, 1], whole[ok_w, 2]
    if edge_out is not None:
        for e in (0, 1):
            rows = np.flatnonzero(edge_out == e)
            cls = fb[set_index, gi, e]
            ok = cls[:, 0] >= min_n
            if rows.size and ok.any():
                med[np.ix_(rows, np.flatnonzero(ok))] = cls[ok, 1][None, :]
                sc[np.ix_(rows, np.flatnonzero(ok))] = cls[ok, 2][None, :]
    return med, sc


def pool_block(detection: BandDetection, records_dir: str, block: Tuple[int, int, int, int], rule: PoolingRule,
               flux_grid_mjy: np.ndarray) -> dict:
    """One square pooling block of POOL_BLOCK_CELLS cells (cell lines i0:i1, columns j0:j1): its rows' curve, flux90,
    n_pos, pooled totals, errors (scatter, bias from both halves: the product's) and build_errors (the build half's:
    what C2 reads), from the records of the cells within the widest pooling reach (13 cells for the legacy rule; under
    "widen" also the records of the cells the extended ladder reaches, and the table's map-wide values last). Reads the
    band's record table memory-mapped (`build_table`). Under rule.gradient_class the errors carry a leading class axis
    (2, n, n_flux); under curve_class "gradient" the curve, flux90, n_pos and totals too.
    Identity (legacy rule): `pacs_completeness._pool_block`."""
    i0, i1, j0, j1 = block
    cg = detection.cell_grid
    nx = cg.nx
    ids = np.asarray(detection.cells.ids, np.int64)
    n_all = ids.size
    sigma_all = _f32(detection.cells.sigma)
    tab = _open_table(records_dir, n_all, sigma_all)
    halo = _pool_halo(rule)
    ey0, ey1 = max(i0 - halo, 0), i1 + halo
    ex0, ex1 = max(j0 - halo, 0), min(j1 + halo, nx)
    lines = np.arange(ey0, ey1, dtype=np.int64)
    a = np.searchsorted(ids, lines * nx + ex0)
    b = np.searchsorted(ids, lines * nx + ex1)
    ext = np.concatenate([np.arange(x, y) for x, y in zip(a, b)] or [np.zeros(0, np.int64)]).astype(np.int64)
    ext_ids = ids[ext]
    inside = ((ext_ids // nx >= i0) & (ext_ids // nx < i1) & (ext_ids % nx >= j0) & (ext_ids % nx < j1))
    out_local = np.flatnonzero(inside)
    out_rows = ext[out_local]
    cov = np.asarray(tab.data["cell_cov"])
    with np.errstate(invalid="ignore", divide="ignore"):
        edge_ratio = cov[1] / cov[2]
    classes_all = pool_classes(detection, rule, edge_ratio)
    Ws_full = pool_matrices(cg, ext_ids, classes_all[ext], rule, sigma=sigma_all[ext])
    Ws = [W[out_local] for W in Ws_full]
    gsets = (0, 1) if rule.gradient_class else (None,)
    sets = (("errors", (0, 1)), ("build_errors", (0,)))
    extra, hist = [], {}
    if rule.short_rule == "widen" and out_rows.size:
        counts = []
        for _name, halves in sets:
            for g in gsets:
                gs = (0, 1) if g is None else (g,)
                counts.append(sum(np.asarray(tab.data["far_h%d_g%d" % (h, gg)], np.int64) for h in halves for gg in gs))
        Wlast = csr_matrix((Ws[-1].data, ext[Ws[-1].indices], Ws[-1].indptr), shape=(out_rows.size, n_all))
        short = np.zeros((out_rows.size, N_PLANES), bool)
        for c in counts:
            short |= (Wlast @ c) < rule.min_injections
        if short.any():
            extra, hist = _ladder_extension(_band_cells(detection, tab, rule), out_rows, short, counts, rule)
    loaded = np.union1d(ext, np.concatenate([W.indices for W in extra])) if extra else ext
    keys = ("plane", "half", "near", "recovered", "dflux", "lnr")
    rec, rec_row = tab.rows(keys, loaded)
    rec["row"] = rec_row
    sig = sigma_all[loaded]
    sig_out = sigma_all[out_rows]
    n_loaded = loaded.size
    pos_ext = np.searchsorted(loaded, ext)

    def remap(W, cols_map):
        return csr_matrix((W.data, cols_map[W.indices], W.indptr), shape=(W.shape[0], n_loaded))

    ladder = [remap(W, pos_ext) for W in Ws]
    level_min = [rule.min_injections] * len(Ws)
    if extra:
        pos_all = np.searchsorted(loaded, np.arange(n_all))
        ladder += [remap(W, pos_all) for W in extra]
        level_min += [rule.min_injections] * len(extra)
    bm = rec["half"] == 0
    build = {k: rec[k][bm] for k in ("row", "plane", "recovered", "near", "dflux", "lnr")}
    # the curve: main's pooling (in S/N, no window, no class); under curve_class "gradient" a coverage-edge cell's two
    # curves pool the edge cells' injections of each |ln r| class (M3); the held-out half's curve the same way (C1)
    if rule.noise_window is None and not rule.edge_class:
        curve_Ws = ladder[:len(Ws)]
    else:
        curve_Ws = [remap(W, pos_ext)[out_local] for W in
                    pool_matrices(cg, ext_ids, np.zeros(ext.size, np.int8), PoolingRule.legacy())]
    if rule.curve_class == "gradient":
        edge_cls = (edge_ratio > rule.edge_ratio).astype(np.int8)
        edge_rule = PoolingRule(rule.radius_arcsec, rule.widenings, rule.min_injections, edge_class=True,
                                edge_ratio=rule.edge_ratio)
        edge_W = [remap(W, pos_ext)[out_local] for W in pool_matrices(cg, ext_ids, edge_cls[ext], edge_rule)]
        is_edge = edge_cls[out_rows].astype(bool)

    def curve_of(r):
        o = _pool_product(curve_Ws, r, flux_grid_mjy, sig, n_loaded, sig_out)
        if rule.curve_class == "gradient":
            gb = _gradient_class(r["lnr"], rule.gradient_boundary)
            parts = [_pool_product(edge_W, {k: v[gb == g] for k, v in r.items()}, flux_grid_mjy, sig, n_loaded,
                                   sig_out) for g in (0, 1)]
            for k in ("curve", "flux90", "n_pos", "pooled_total"):
                e = is_edge.reshape((-1,) + (1,) * (o[k].ndim - 1))
                o[k] = np.stack([np.where(e, q[k], o[k]) for q in parts])
        return o

    out = curve_of(build)
    hm = ~bm
    held_c = curve_of({k: rec[k][hm] for k in ("row", "plane", "recovered", "near", "dflux", "lnr")})
    out["held_curve"], out["held_total"] = held_c["curve"], held_c["pooled_total"]
    fb = np.asarray(tab.data["fallback"]) if (rule.short_rule == "widen" and "fallback" in tab.data) else None
    edge_out = classes_all[out_rows] if rule.edge_class else None

    def errors(r, set_index):
        e = []
        for g in gsets:
            rr = r if g is None else {k: v[_gradient_class(r["lnr"], rule.gradient_boundary) == g] for k, v in r.items()}
            fbv = None if fb is None else _fallback_values(fb, set_index, g, edge_out, out_rows.size, rule.min_injections)
            e.append(_cell_errors(ladder, flux_grid_mjy, sig, rr, rule, sig_out, fbv, level_min))
        return e[0] if len(e) == 1 else {k: np.stack([q[k] for q in e]) for k in ("scatter", "bias")}

    out["errors"] = errors({k: rec[k] for k in ("row", "plane", "near", "dflux", "lnr")}, 0)
    out["build_errors"] = errors({k: build[k] for k in ("row", "plane", "near", "dflux", "lnr")}, 1)
    out["rows"] = out_rows
    out["widen"] = hist
    return out


def _pool_product(Ws, rec, flux_grid_mjy, sigma_rows, n_row, sigma_out):
    """The curve (encoded), FLUX90, the injections behind the faintest plane and the pooled totals (`pool_product`)."""
    fraction, count = curve_on_flux_grid(Ws, rec["row"], rec["plane"], rec["recovered"], sigma_out, flux_grid_mjy, n_row)
    return dict(curve=np.rint(fraction * STORAGE_SCALE).astype(np.uint16),
                flux90=flux90(fraction, flux_grid_mjy).astype(np.float32), n_pos=count[:, -1].astype(np.int32),
                pooled_total=count)


def _cell_errors(Ws, flux_grid_mjy, sigma_rows, rec, rule, sigma_out, fallback=None, level_min=None):
    """SCATTER and BIAS over the records farther than BLEND_RADIUS_BEAMS from every detection (`cell_errors`)."""
    n_row = len(sigma_rows)
    sel = ~rec["near"].astype(bool)
    bias, scatter = error_on_flux_grid(Ws, rec["row"][sel].astype(np.int64), rec["plane"][sel].astype(np.int64),
                                       rec["dflux"][sel], sigma_rows, flux_grid_mjy, n_row, rule, sigma_out=sigma_out,
                                       fallback=fallback, level_min=level_min)
    return dict(scatter=scatter, bias=bias)


def pool_tasks(detection: BandDetection) -> List[Tuple[int, int, int, int]]:
    """The pooling blocks holding rows, ascending (`pool_tasks`, POOL_BLOCK_CELLS = 8): (i0, i1, j0, j1)."""
    ids = np.asarray(detection.cells.ids, np.int64)
    if ids.size == 0:
        return []
    cy, cx = detection.cell_grid.line_column(ids)
    b = np.unique(np.c_[cy // POOL_BLOCK_CELLS, cx // POOL_BLOCK_CELLS], axis=0)
    n = POOL_BLOCK_CELLS
    return [(int(i * n), int((i + 1) * n), int(j * n), int((j + 1) * n)) for i, j in b]


@dataclass
class BandCompleteness:
    """A band's assembled completeness (the old band.pkl, now band.hdf5): curve (n_row, n_flux) uint16 (fraction x
    COMPLETENESS_STORAGE_SCALE), flux90, n_pos, scatter, bias (n_row, n_flux) float32, `held` (the held-out half's
    records with build_errors and the per-map stats, for C1/C2), `stats` (counts, pooled totals, widening), plus the
    BandDetection it extends. Under a gradient rule the class axis leads (see `pool_block`)."""
    detection: BandDetection
    curve: np.ndarray
    flux90: np.ndarray
    n_pos: np.ndarray
    scatter: np.ndarray
    bias: np.ndarray
    held: dict
    stats: dict


def assemble_band(detection: BandDetection, pooled: Sequence[dict], held: dict, stats: dict,
                  flux_grid_mjy: np.ndarray) -> BandCompleteness:
    """The band from its pooling blocks (`_assemble_tile` + `_band_dict`): rows in cell-id order (FROZEN 4); every row
    filled exactly once. `held`, `stats`: `held_and_stats`. `held` gains build_errors (C2), and the held-out half's own
    curve and pooled totals built exactly as the stored curve (`curve`, `curve_total`: C1, per class under a gradient
    curve)."""
    n_row = np.asarray(detection.cells.ids).size
    n_plane = flux_grid_mjy.size
    if not pooled:
        raise ValueError("no pooling blocks")
    p0 = pooled[0]
    cl = p0["curve"].shape[:-2]                       # (2,) under curve_class "gradient", else ()
    el = p0["errors"]["scatter"].shape[:-2]           # (2,) under a gradient pooling class, else ()

    def arr(shape_lead, tail, dt, fill):
        return np.full(tuple(shape_lead) + tuple(tail), fill, dt)

    curve = arr(cl, (n_row, n_plane), np.uint16, 0)
    f90 = arr(cl, (n_row,), np.float32, np.nan)
    n_pos = arr(cl, (n_row,), np.int32, 0)
    tot = arr(cl, (n_row, n_plane), np.float64, 0.0)
    h_curve = arr(cl, (n_row, n_plane), np.uint16, 0)
    h_tot = arr(cl, (n_row, n_plane), np.float64, 0.0)
    errs = {k: arr(el, (n_row, n_plane), np.float32, np.nan) for k in ("scatter", "bias")}
    build_errs = {k: arr(el, (n_row, n_plane), np.float32, np.nan) for k in ("scatter", "bias")}
    filled = np.zeros(n_row, bool)
    widen = {}
    for o in pooled:
        r = o["rows"]
        assert not filled[r].any()
        filled[r] = True
        curve[..., r, :], f90[..., r], n_pos[..., r], tot[..., r, :] = o["curve"], o["flux90"], o["n_pos"], o["pooled_total"]
        h_curve[..., r, :], h_tot[..., r, :] = o["held_curve"], o["held_total"]
        for k in errs:
            errs[k][..., r, :] = o["errors"][k]
            build_errs[k][..., r, :] = o["build_errors"][k]
        for k, v in o.get("widen", {}).items():
            widen[k] = widen.get(k, 0) + v
    assert filled.all()
    held = dict(held)
    held["build_errors"] = build_errs
    held["curve"], held["curve_total"] = h_curve, h_tot
    stats = dict(stats, min_pool=float(tot.min()) if tot.size else np.nan,
                 median_pool=float(np.median(tot)) if tot.size else np.nan,
                 short=float((tot < MIN_INJECTIONS).mean()) if tot.size else np.nan, widen=widen,
                 nan_cellplanes=int((~np.isfinite(errs["scatter"])).sum()),
                 nan_build_cellplanes=int((~np.isfinite(build_errs["scatter"])).sum()))
    held["stats"] = [stats]
    return BandCompleteness(detection, curve, f90, n_pos, errs["scatter"], errs["bias"], held, stats)


def empty_band(n_src: int, n_flux: int) -> BandCompleteness:
    """A band no mosaic reaches: no rows, SOURCE_ROW -1 (`_empty_band`)."""
    z = lambda dt: np.zeros(0, dt)
    cells = CellStats(ids=z(np.int64), n_sigma=z(np.int16), sigma=z(np.float32), offset=z(np.float32), sky=z(float),
                      ra=z(float), dec=z(float), m_sigma=z(np.float32), m_offset=z(np.float32), covered=z(bool))
    peaks = Peaks(x=z(float), y=z(float), fnu=z(float), cell=z(np.int64), row=z(np.int64), snr=z(float), bin=z(np.int8),
                  ra=z(float), dec=z(float), neg_count=z(np.int64))
    det = BandDetection(cells=cells, cell_grid=CellGrid(None, (0, 0), 1.0, 1.0, 0, 0), shift=(0.0, 0.0, 0),
                        src_row=np.full(n_src, -1, np.int64), cell_bin=z(np.int8), peaks=peaks, pos_k_arcsec=np.nan,
                        r_match=np.nan, kernel=Kernel(0, np.zeros((1, 1)), np.zeros((1, 1)), 0.0), n_src=int(n_src))
    nan = np.zeros((0, n_flux), np.float32)
    return BandCompleteness(det, np.zeros((0, n_flux), np.uint16), z(np.float32), z(np.int32), nan, nan.copy(), None,
                            dict(stats=[]))


# ----------------------------------------------------------------------------------------------------------------------
# ROUND-2 ADAPTER to the old modules (R3: swap -- delete this section once mosaic.py and detect.py are filled)

class _OldMosaic:
    """The contract's Mosaic surface the injection uses, on the old prepared `pacs_fit.Tile` of a mosaic record:
    window, template_patches, aperture (with a Window), pixel_sigma, beam_px_at, injection_half_width, pixel_arcsec,
    band, grid.shape, layers["coverage"], key. Picklable by (config, record). R3: swap."""

    def __init__(self, config, record):
        self.config = config
        self.record = record
        self.key = record.path
        self._tile = None

    def __getstate__(self):
        return dict(config=self.config, record=self.record, key=self.key, _tile=None)

    @property
    def tile(self):
        if self._tile is None:
            from sesnaimpute.sky.derived import pacs_fit           # R3: swap
            self._tile = pacs_fit.cached_tile(self.config, self.record)
        return self._tile

    @property
    def shape(self):
        return tuple(self.record.shape)

    @property
    def band(self):
        return self.record.band

    @property
    def pixel_arcsec(self):
        return self.tile.pixel_arcsec

    @property
    def layers(self):
        return {"coverage": _DenseLayer(os.path.join(self.record.mosaic_dir, "coverage.npy"), 0.0)}

    def injection_half_width(self):
        return self.tile.injection_half_width()

    def beam_px_at(self, x, y):
        return self.tile.beam_px_at(x, y)

    def pixel_sigma(self, x, y):
        return self.tile.pixel_sigma(x, y)

    def window(self, y0, y1, x0, x1):
        t = self.tile
        return Window(y0, x0, np.array(t.image[y0:y1, x0:x1]), t.error[y0:y1, x0:x1])

    def template_patches(self, x, y, flux):
        parts = list(self.tile._template_patches(x, y, flux))
        if not parts:
            h = self.tile.injection_half_width()
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2 * h + 1, 2 * h + 1))
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))

    def aperture(self, x, y, window=None, chunk=2048):
        """`pacs_fit.Tile.aperture` on a Window (rows and columns offset; the old code offset rows only)."""
        from sesnaimpute.sky.derived import pacs_fit                # R3: swap
        from sesnaimpute.sky.derived.pacs_psf import MosaicPSF
        t = self.tile
        if window is None:
            image, error, r0, c0 = t.image, t.error, 0, 0
        else:
            image, error, r0, c0 = window.image, window.error, window.y0, window.x0
        r_ap, r_in, r_out = (r / t.pixel_arcsec for r in APERTURES[t.record.band])
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        n = x.size
        rb = int(np.ceil(r_out)) + 1
        flux, sky, sd, area = (np.full(n, np.nan) for _ in range(4))
        cov = np.zeros(n, bool)
        sd_eef = np.full(n, t.enclosed_fraction)
        for lo in range(0, n, chunk):
            sl = slice(lo, lo + chunk)
            xs, ys = x[sl], y[sl]
            f, s_, d_, a_, c_ = pacs_fit._aperture_core(image, error, xs - c0, ys - r0, r_ap, r_in, r_out, 5)
            mix = t._psf.weights(xs, ys) if isinstance(t._psf, MosaicPSF) else None
            eef, fraction = t._corrections_at(xs, ys, mix)
            amp = np.where(c_, f / fraction, 0.0)
            if not c_.any():
                cov[sl] = c_
                continue
            model = t._outer_light(xs, ys, rb, mix)
            for _ in range(2):
                f, s_, d_, a_, c_ = pacs_fit._aperture_core_wing(image, error, xs - c0, ys - r0, r_ap, r_in, r_out, 5,
                                                                 model, np.where(c_ & np.isfinite(amp), amp, 0.0))
                amp = np.where(c_, f / eef, 0.0)
            flux[sl], sky[sl], sd[sl], area[sl], cov[sl] = f / eef, s_, d_, a_, c_
            sd_eef[sl] = eef
        return mosaic.ApertureFit(flux, sky, sd * np.sqrt(area) / sd_eef, cov)

    def operations(self) -> "_Operations":
        ops = _Operations.__new__(_Operations)
        ops.m = self
        ops.window = self.window
        ops.template_patches = self.template_patches
        ops.aperture = lambda x, y, win: self.aperture(x, y, window=win)
        ops.coverage = self.layers["coverage"]
        ops.paste = _old_paste
        ops.peaks = lambda inj, x, y, r, flux, win: _old_window_peaks(self, inj, x, y, r, flux, win)
        return ops


class _DenseLayer:
    """A dense .npy layer read through the TiledLayer `window` surface (fill outside the grid). R3: swap."""

    def __init__(self, path, fill):
        self.a = np.load(path, mmap_mode="r")
        self.fill = fill

    def window(self, y0, y1, x0, x1):
        ny, nx = self.a.shape
        out = np.full((y1 - y0, x1 - x0), self.fill, self.a.dtype)
        a0, a1, b0, b1 = max(y0, 0), min(y1, ny), max(x0, 0), min(x1, nx)
        if a1 > a0 and b1 > b0:
            out[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = self.a[a0:a1, b0:b1]
        return out


def _old_paste(window: Window, cx, cy, patches) -> None:
    """`pacs_fit._paste` of each patch, in order, into the window (grid coordinates). R3: swap -> mosaic.paste."""
    ny, nx = window.image.shape
    for i in range(len(cx)):
        p = patches[i]
        x0, y0 = int(cx[i]) - window.x0, int(cy[i]) - window.y0
        side_y, side_x = p.shape
        a0, b0 = max(x0, 0), max(y0, 0)
        a1, b1 = min(x0 + side_x, nx), min(y0 + side_y, ny)
        if a1 <= a0 or b1 <= b0:
            continue
        window.image[b0:b1, a0:a1] += p[b0 - y0:b1 - y0, a0 - x0:a1 - x0].astype(np.float32)


def _old_window_peaks(om: _OldMosaic, inj: Injector, x, y, radius_px, flux, window: Window):
    """`pacs_completeness.window_peaks` with the found peak's flux measured on a Window (`_OldMosaic.aperture`). R3:
    swap -> detect.window_peaks."""
    from sesnaimpute.sky.derived import pacs_completeness as pc   # R3: swap
    from sesnaimpute.sky.derived import pacs_fit
    tile = om.tile
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    out = (np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    if n == 0:
        return out
    ms = pc.matched_state(tile)
    rad = np.broadcast_to(np.asarray(radius_px, float), (n,))[:, None, None]
    h = int(np.ceil(rad.max())) + 2
    off = np.arange(-h, h + 1)
    cx, cy = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
    px = np.ascontiguousarray(np.broadcast_to(cx[:, None, None] + off[None, None, :], (n, off.size, off.size)))
    py = np.ascontiguousarray(np.broadcast_to(cy[:, None, None] + off[None, :, None], (n, off.size, off.size)))
    dist = np.hypot(px - x[:, None, None], py - y[:, None, None])
    ny, nx = tile.shape
    use = (dist <= rad + 1.5) & (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    v = np.full(px.shape, np.nan)
    v[use] = ms["m"][py[use], px[use]]
    if flux is not None:
        v = v + pc._injected_delta(tile, x, y, np.broadcast_to(np.asarray(flux, float), x.shape), px, py)
    cid = tile.grid.cell_ids(px, py)
    good = cid >= 0
    v = np.where(good, v - inj.m_offset_cell[np.maximum(cid, 0)], np.nan)
    thr = np.where(good, pacs_fit.DETECTION_SNR * inj.m_sigma_cell[np.maximum(cid, 0)], np.nan)
    pk = pc._local_peaks(v, thr) & (dist <= rad)
    found, xc, yc, fnu, dd = out
    cand = np.where(pk, dist, np.inf).reshape(n, -1)
    best = np.argmin(cand, axis=1)
    has = np.isfinite(cand[np.arange(n), best])
    for i in np.flatnonzero(has):
        iy, ix = divmod(int(best[i]), off.size)
        dy, dx = pc._parabola(v[i], iy, ix)
        xc[i], yc[i] = px[i, iy, ix] + dx, py[i, iy, ix] + dy
    if has.any():
        cidh = tile.grid.cell_ids(xc[has], yc[has])
        offh = np.where(cidh >= 0, inj.offset_cell[np.maximum(cidh, 0)], np.nan)
        a = om.aperture(xc[has], yc[has], window=window)
        fnu[has] = np.where(a.covered, a.flux - offh, np.nan)
        dd[has] = np.hypot(xc[has] - x[has], yc[has] - y[has])
        found[has] = dd[has] <= rad[has, 0, 0]
    return found, xc, yc, fnu, dd


def _old_band_detection(state: dict) -> BandDetection:
    """A BandDetection from the old DETECT state (`pacs_stages.detect`'s detect.pkl dict, one map). R3: swap."""
    g = state["geometry"][0][2]
    cg = CellGrid(g.wcs, tuple(g.shape), float(g.step_x), float(g.step_y), int(g.nx), int(g.ny))
    cols = state["cols"]
    ids = np.asarray(state["cell_id"], np.int64)
    cells = CellStats(ids=ids, n_sigma=np.asarray(cols["n_sigma"]), sigma=np.asarray(state["cell_sigma"]),
                      offset=np.asarray(state["cell_offset"]), sky=np.asarray(cols["sky"]), ra=np.asarray(cols["ra"]),
                      dec=np.asarray(cols["dec"]), m_sigma=np.asarray(state["m_sigma_rows"]),
                      m_offset=np.asarray(state["m_offset_rows"]), covered=np.ones(ids.size, bool))
    p = state["peaks"]
    peaks = Peaks(x=np.asarray(p["x"], float), y=np.asarray(p["y"], float), fnu=np.asarray(p["fnu"], float),
                  cell=ids[np.asarray(p["row"], np.int64)] if ids.size else np.zeros(0, np.int64),
                  row=np.asarray(p["row"], np.int64), snr=np.asarray(p["snr"], float), bin=np.asarray(p["bin"]),
                  ra=np.asarray(p["ra"], float), dec=np.asarray(p["dec"], float), neg_count=np.zeros(0, np.int64))
    sh = state["shifts"][0]
    return BandDetection(cells=cells, cell_grid=cg, shift=(float(sh[0]), float(sh[1]), int(sh[2])),
                         src_row=np.asarray(state["src_row"]), cell_bin=np.asarray(state["cell_bin"]), peaks=peaks,
                         pos_k_arcsec=float(state["pos_k"]), r_match=float(state["r_match"][0]),
                         kernel=Kernel(0, np.zeros((1, 1)), np.zeros((1, 1)), 0.0), n_src=int(state["n_src"]))
