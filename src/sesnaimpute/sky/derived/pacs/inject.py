"""Artificial-source injection, pooling and the completeness/error estimator of a band, block by block.

Round-1 skeleton (CONTRACT.md): signatures only. Depends on: grid, psf, mosaic, detect. Replaces (pacs_completeness.py):
snr_bin, snr_grid, canonical_grid, Injector, _inject_batches, _inject_pairs, _injection_state, _map_first, map_sizing,
injection_blocks, injection_pairs, tile_order, _chunks, part_counts, round_task_list, _real_check (-> checks.py),
cell_centres_arcsec (-> CellGrid.centres_arcsec), pool_matrices, pooled_counts, pooled_median_scatter(_one),
_snr_brackets, _at_flux, curve_on_flux_grid, error_on_flux_grid, flux90, pool_product, cell_errors, pool_dir,
_write_pool_table, pool_tasks, _pool_block, _assemble_tile, _band_dict, _empty_band.

The estimator's pooling class and the short-cell rule are PARAMETERS (`PoolingRule`): the science track (FIX_SPEC.md
sec 8 and 12) fixes their values; until then `PoolingRule.legacy()` reproduces main 0683690 exactly (radius 120 arcsec,
widenings x1/x2/x3, no noise window, no edge class, SCATTER NaN where short) and is the identity gate's setting.

Store (transient, the band's work directory): injection records `inject_r<r>_s<k>of<K>.hdf5` (one table per shard:
row int32, plane int8, half int8, batch int32, block int32, near bool, recovered bool, dflux float32, flux float32,
det_flux float32, det_dist float32; attrs round, shard, K) and pooling records `pool_s<k>of<K>.hdf5`; the assembled band
`band.hdf5` (curve, flux90, scatter, bias, n_pos, held-out table). All written by stages.write_record (atomic, receipt);
no pickles. The injection rng is default_rng([seed_key, plane, 2 * batch + half, block]) (FROZEN 2-3): records are
independent of K, n_jobs and the piece cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sesnaimpute.sky.derived.pacs import detect, grid, mosaic
from sesnaimpute.sky.derived.pacs.detect import BandDetection
from sesnaimpute.sky.derived.pacs.grid import CellGrid
from sesnaimpute.sky.derived.pacs.mosaic import Mosaic, Window

ROUND2 = "round 2"


# ----------------------------------------------------------------------------------------------------------------------
# the S/N planes and the flux grid

def snr_grid() -> np.ndarray:
    """The injection S/N planes: 10 ** arange(SNR_LOG_MIN, SNR_LOG_MAX + step/2, SNR_LOG_STEP) (`snr_grid`)."""
    raise NotImplementedError(ROUND2)


def flux_grid() -> Tuple[np.ndarray, np.ndarray]:
    """(mag, flux mJy): the one survey-wide flux grid, m = -2.5 log10(F / 1 Jy) from MAG_MIN to MAG_MAX in MAG_STEP,
    bright to faint (`canonical_grid`)."""
    raise NotImplementedError(ROUND2)


def snr_bin(snr: np.ndarray) -> np.ndarray:
    """The C2 S/N bin of each value (np.digitize on definitions.PACS_SNR_BIN_EDGES)."""
    raise NotImplementedError(ROUND2)


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
    CELL_SIGMA, and the measurement of each batch on a private Window of the block plus the margin mg (paste, aperture,
    window_peaks, recovered = nearest peak within the recovery radius and brighter than any real source there; near =
    within BLEND_RADIUS_BEAMS x beam of a real detection). Per-worker memory: one block window + per-row arrays (no
    n_cells-sized array: the old row_of_cell becomes a search on the sorted row ids).
    Identity: `pacs_completeness.Injector` (pacs_completeness.py:832-1017): bit-identical records for the same
    (seed_key, plane, batch, half, block)."""

    def __init__(self, m: Mosaic, detection: BandDetection, seed_key: int) -> None:
        raise NotImplementedError(ROUND2)

    def batch(self, rng: np.random.Generator, rows: np.ndarray, snr: float) -> dict:
        """One batch: positions (`positions`), fluxes, the measurement (`measure_positions`). Returns the record
        columns of the module docstring plus seconds."""
        raise NotImplementedError(ROUND2)

    def measure_positions(self, x: np.ndarray, y: np.ndarray, flux: np.ndarray, radius_px: Optional[np.ndarray] = None
                          ) -> dict:
        """Pastes the sources into a private window of the rows they span plus mg, measures forced aperture flux and
        runs the blind detection there (`Injector.measure_positions`, pacs_completeness.py:949)."""
        raise NotImplementedError(ROUND2)


def first_batches(config, m: Mosaic, detection: BandDetection) -> int:
    """The batches of round 0 per (plane, half) and block: ceil(INTERIOR_SAFETY x MIN_INJECTIONS / interior pool
    density) (`_map_first`, pacs_completeness.py:1170)."""
    raise NotImplementedError(ROUND2)


def round_tasks(detection: BandDetection, first: int, r: int, load: Callable[[int], np.ndarray]) -> List[InjectionTask]:
    """The ordered pieces of round `r` ([] when the band is complete): round 0 = `first` batches in every block for every
    (plane, half); each later round one more batch per pair over the cells within the pooling radius of a cell whose
    pooled count (all earlier rounds) is short of MIN_INJECTIONS, up to MAX_EXTRA_BATCHES. `load(q)`: the (rows x pairs)
    counts of round q's records. Identity: `pacs_completeness.round_task_list` (pacs_completeness.py:1258) -- same
    tasks in the same order (the old (tile index, inputs) fields drop: one mosaic per band)."""
    raise NotImplementedError(ROUND2)


def inject_task(config, m: Mosaic, detection: BandDetection, task: InjectionTask) -> dict:
    """One piece's injections (`_inject_pairs`): the record columns, concatenated over its jobs, plus seconds and the
    real-map check rows when `task.check` (checks.real_map_check)."""
    raise NotImplementedError(ROUND2)


def record_counts(records: dict, n_rows: int) -> np.ndarray:
    """(rows x pairs) injection counts of a set of records (`part_counts`, dense int64 per band)."""
    raise NotImplementedError(ROUND2)


# ----------------------------------------------------------------------------------------------------------------------
# pooling and the estimator

@dataclass(frozen=True)
class PoolingRule:
    """How a cell's injections are pooled (FIX_SPEC.md sec 8, 12): `radius_arcsec`, `widenings`, `min_injections`;
    `noise_window` (pool only cells with CELL_SIGMA within x this of the target's; None = no window), `edge_class`
    (pool coverage-edge cells -- within-cell coverage max/min > `edge_ratio` -- only with edge cells), `short_rule`
    ("nan": SCATTER NaN where short, main's; "widen": widen beyond the last widening inside the class until
    min_injections, then pool the class over the map; SCATTER never NaN)."""
    radius_arcsec: float
    widenings: Tuple[int, ...]
    min_injections: int
    noise_window: Optional[float] = None
    edge_class: bool = False
    edge_ratio: float = 2.0
    short_rule: str = "nan"

    @classmethod
    def legacy(cls) -> "PoolingRule":
        """Main 0683690's rule: definitions.PACS_POOL_ARCSEC, PACS_POOL_WIDENINGS, PACS_MIN_INJECTIONS, no window,
        no class, short = NaN. The identity gate's setting."""
        raise NotImplementedError(ROUND2)


def pool_classes(detection: BandDetection, rule: PoolingRule, edge_ratio_per_cell: Optional[np.ndarray] = None
                 ) -> np.ndarray:
    """Each row's pooling class under `rule` (all 0 for the legacy rule)."""
    raise NotImplementedError(ROUND2)


def pool_matrices(cells: CellGrid, ids: np.ndarray, classes: np.ndarray, rule: PoolingRule) -> list:
    """Sparse (n, n) 0/1 pooling matrices, one per widening: (i, j) = 1 where cell centres (CellGrid.centres_arcsec) lie
    within radius x widening and the classes match. Identity (legacy rule): `pacs_completeness.pool_matrices`."""
    raise NotImplementedError(ROUND2)


def curve_on_flux_grid(Ws: list, row: np.ndarray, plane: np.ndarray, recovered: np.ndarray, sigma_rows: np.ndarray,
                       flux_grid_mjy: np.ndarray, n_row: int) -> Tuple[np.ndarray, np.ndarray]:
    """(fraction (n_row, n_flux), pooled counts): the pooled recovered fraction per S/N plane read at F / CELL_SIGMA,
    linear in log S/N (`curve_on_flux_grid`)."""
    raise NotImplementedError(ROUND2)


def error_on_flux_grid(Ws: list, row: np.ndarray, plane: np.ndarray, dflux: np.ndarray, sigma_rows: np.ndarray,
                       flux_grid_mjy: np.ndarray, n_row: int, rule: PoolingRule,
                       sigma_out: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(bias, scatter) (n_row, n_flux): pooled median and 1.4826 x MAD of dflux / CELL_SIGMA over the far injections,
    read at F / the row's CELL_SIGMA, times it; short cells per `rule.short_rule`. Identity (legacy rule):
    `error_on_flux_grid` + `pooled_median_scatter` bit-identical."""
    raise NotImplementedError(ROUND2)


def flux90(fraction: np.ndarray, flux_grid_mjy: np.ndarray, level: float = 0.9) -> np.ndarray:
    """The faintest flux at which the curve still reaches `level` (`flux90`)."""
    raise NotImplementedError(ROUND2)


def pool_block(detection: BandDetection, records_dir: str, block: Tuple[int, int, int, int], rule: PoolingRule,
               flux_grid_mjy: np.ndarray) -> dict:
    """One square pooling block of POOL_BLOCK_CELLS cells (cell lines i0:i1, columns j0:j1): its rows' curve, flux90,
    n_pos, scatter, bias from the build half's records of the cells within the widest pooling reach (13 cells for the
    legacy rule). Reads the injection records memory-mapped. Identity: `pacs_completeness._pool_block`."""
    raise NotImplementedError(ROUND2)


def pool_tasks(detection: BandDetection) -> List[Tuple[int, int, int, int]]:
    """The pooling blocks holding rows, ascending (`pool_tasks`, POOL_BLOCK_CELLS = 8)."""
    raise NotImplementedError(ROUND2)


@dataclass
class BandCompleteness:
    """A band's assembled completeness (the old band.pkl, now band.hdf5): curve (n_row, n_flux) uint16 (fraction x
    COMPLETENESS_STORAGE_SCALE), flux90, n_pos, scatter, bias (n_row, n_flux) float32, `held` (the held-out half's
    records, for C1/C2), `stats` (counts per round), plus the BandDetection it extends."""
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
    """The band from its pooling blocks (`_assemble_tile` + `_band_dict`): rows in cell-id order (FROZEN 4)."""
    raise NotImplementedError(ROUND2)


def empty_band(n_src: int, n_flux: int) -> BandCompleteness:
    """A band no mosaic reaches: no rows, SOURCE_ROW -1 (`_empty_band`)."""
    raise NotImplementedError(ROUND2)
