"""The blind detection of PACS sources on a band's mosaic, tile by tile: the matched (aperture-filtered) image and
its noise, the cell statistics, the registration to SESNA, the peaks, the matching radius.

Round-1 skeleton (CONTRACT.md): signatures only. Depends on: grid, psf, mosaic. Replaces (pacs_completeness.py):
filtered, _detection_kernel, detection_strips/open_detection/detection_strip/_finish_piece/_strip_piece/_layer_strip/
_prepare_stage/matched_state, cell_fits/cell_statistics/ring/block_sigma/cell_offset/_tile_reference/_piece/
detection_cell_stats, _local_peaks/_parabola/_block_peaks/_assemble_peaks/window_peaks/_injected_delta, _calibrate,
tile_inputs, sky_bins, geometry_cells, _tile_geometry, _band_detect; and pacs_fit.registration/registration_pairs,
select_deeper. Dead and not carried: prepare_detection, encode_curves.

Store (transient, under the region's work directory, stages.work_dir): `<band>/detect/{m,ratio,norm}/` TiledLayer
float32 fill NaN and `<band>/detect/{valid,pix_valid}/` TiledLayer bool fill False, on the band's PixelGrid; the band's
detection record `<band>/detect.hdf5` (stages.write_record: the tables of `BandDetection` and the kernel; no pickle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sesnaimpute.sky.derived.pacs import grid, mosaic
from sesnaimpute.sky.derived.pacs.grid import CellGrid, TiledLayer
from sesnaimpute.sky.derived.pacs.mosaic import Mosaic, Window

ROUND2 = "round 2"

#: the detection layers, their dtype and fill
DETECTION_LAYERS = {"m": (np.float32, np.nan), "ratio": (np.float32, np.nan), "norm": (np.float32, np.nan),
                    "valid": (np.bool_, False), "pix_valid": (np.bool_, False)}


@dataclass(frozen=True)
class Kernel:
    """FROZEN 11. The detection kernel: half-width `h` (= Mosaic.injection_half_width()), the template on its
    footprint `kc` and the flipped footprint `fc` (side 2h+1), and `s_full` the zero-sum template's sum of squares."""
    h: int
    kc: np.ndarray
    fc: np.ndarray
    s_full: float


def detection_kernel(m: Mosaic) -> Kernel:
    """FROZEN 11. The weight-mean PSF table (MosaicPSF.evaluate_mean_xy) at the grid centre (ny // 2, nx // 2), on its
    footprint (> ZERO_SUM_FOOTPRINT of its peak), zero-summed. Identity: `pacs_completeness._detection_kernel`
    (pacs_completeness.py:497) bit-identical."""
    raise NotImplementedError(ROUND2)


def detect_tile(m: Mosaic, kernel: Kernel, iy: int, ix: int, layers: Dict[str, TiledLayer]) -> bool:
    """The detection layers on one tile: read the tile plus a halo of h pixels (Mosaic.tile_window), correlate the
    image and the error with the kernel (float32 temporaries; the halo makes the tile's interior equal to a whole-grid
    correlation), write m, ratio, norm, valid, pix_valid for the tile's own pixels (layers open "r+"; nothing written for
    a tile with no valid pixel). Returns whether it wrote. Identity: `pacs_completeness.detection_strip`
    (pacs_completeness.py:546) per pixel: m, ratio, norm within 1e-6 relative (FFT summation order), valid and
    pix_valid identical."""
    raise NotImplementedError(ROUND2)


def build_detection_layers(config, m: Mosaic, work_dir: str, n_jobs: Optional[int] = None) -> Dict[str, TiledLayer]:
    """`detect_tile` over the mosaic's present tiles (joblib, one piece = a run of tiles; per-worker memory one tile
    with halo, independent of the grid size and the worker count), then each layer's write_index. Writes
    `<work_dir>/detect/<layer>/`. Returns the layers open "r"."""
    raise NotImplementedError(ROUND2)


def open_detection(work_dir: str) -> Dict[str, TiledLayer]:
    """The detection layers of a band, open read-only."""
    raise NotImplementedError(ROUND2)


def filtered(m: Mosaic, x: np.ndarray, y: np.ndarray, offset_cell: np.ndarray, window: Optional[Window] = None
             ) -> np.ndarray:
    """The matched-filter flux at positions minus the cell offset, NaN where not covered
    (`pacs_completeness.filtered`)."""
    raise NotImplementedError(ROUND2)


# ----------------------------------------------------------------------------------------------------------------------
# cell statistics

@dataclass
class CellStats:
    """Per covered cell of a band (rows ordered by cell id; one mosaic per band, so map index 0): `ids` int64,
    `n_sigma`, `sigma` (CELL_SIGMA, mJy), `offset` (CELL_OFFSET, mJy), `sky`, `ra`, `dec` (cell centre, ICRS deg),
    `m_sigma`, `m_offset` (the matched image's), `covered` bool."""
    ids: np.ndarray
    n_sigma: np.ndarray
    sigma: np.ndarray
    offset: np.ndarray
    sky: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    m_sigma: np.ndarray
    m_offset: np.ndarray
    covered: np.ndarray


def cell_reference(config, m: Mosaic, cells: np.ndarray, ra: np.ndarray, dec: np.ndarray) -> dict:
    """The mosaic's reference noise (median cell sigma over up to REFERENCE_CELLS evenly spread cells, seed 0) and its
    registration shift (east, north, n_pairs) against the SESNA positions on it. Identity: `pacs_completeness.
    _tile_reference` (pacs_completeness.py:235); the shift is `registration` below."""
    raise NotImplementedError(ROUND2)


def registration(m: Mosaic, ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[float, float, int]:
    """FROZEN 9. The map's astrometric offset from SESNA (east arcsec, north arcsec, n_pairs): median of PACS minus
    SESNA over mutual nearest pairs of bright peaks (Mosaic.neighbours) and SESNA positions within the local beam;
    fewer than MIN_REGISTRATION_PAIRS: (0, 0, n). Identity: `pacs_fit.registration` + `registration_pairs`."""
    raise NotImplementedError(ROUND2)


def cell_block(config, m: Mosaic, detection: Dict[str, TiledLayer], ids: np.ndarray, reference: dict) -> dict:
    """One cell-statistics block (FROZEN 3: the cells of one `CellGrid.stat_block` id plus a one-cell border): the
    cells' fits at N_POS positions each (`CellGrid.sample_pixels(ids, N_POS, seed)`, seed = map index + 1 = 1, FROZEN 2),
    the clean statistics, the 3 x 3 ring noise and offsets, the matched image's level and MAD. Reads the mosaic window of
    the block plus the fit reach, and the detection layer m. Identity: `pacs_completeness._piece` + cell_fits/
    cell_statistics/block_sigma/cell_offset/detection_cell_stats, bit-identical per cell."""
    raise NotImplementedError(ROUND2)


def cell_statistics(config, m: Mosaic, detection: Dict[str, TiledLayer], shift: Tuple[float, float],
                    reference: dict, n_jobs: Optional[int] = None) -> CellStats:
    """`cell_block` over every stat block of the grid holding a covered tile, in block order (FROZEN 3), joblib; then the
    covered rows ordered by cell id. Cells are enumerated over the whole grid (C sec 5 item 8) but a block with no
    present tile is skipped (it has no covered cell). Identity: the rows, CELL_SIGMA, CELL_OFFSET, m_sigma, m_offset of
    `_band_detect` (pacs_completeness.py:1556-1590) bit-identical."""
    raise NotImplementedError(ROUND2)


# ----------------------------------------------------------------------------------------------------------------------
# peaks

@dataclass
class Peaks:
    """The band's PACS detections: grid pixel x, y, fnu (mJy), cell id, row (into CellStats), snr, sky bin, ra, dec
    (ICRS, SESNA frame: the registration shift removed), and the inverted-map peak count per sky bin (`neg_count`)."""
    x: np.ndarray
    y: np.ndarray
    fnu: np.ndarray
    cell: np.ndarray
    row: np.ndarray
    snr: np.ndarray
    bin: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    neg_count: np.ndarray


def block_peaks(config, m: Mosaic, detection: Dict[str, TiledLayer], ids: np.ndarray, cell_tables: Tuple[np.ndarray, ...]
                ) -> dict:
    """The detections of one stat block: 8-neighbour local maxima of the matched S/N >= DETECTION_SNR inside the block's
    cells (parabolic sub-pixel position), their aperture flux, and the same on the negated map. `cell_tables` = (offset,
    sigma, m_offset, m_sigma) indexed by cell id (n_cells float64 -- replaced in round 2 by a per-block slice).
    Identity: `pacs_completeness._block_peaks` (pacs_completeness.py:792)."""
    raise NotImplementedError(ROUND2)


def window_peaks(m: Mosaic, detection: Dict[str, TiledLayer], x: np.ndarray, y: np.ndarray, radius_px: np.ndarray,
                 m_offset_cell: np.ndarray, m_sigma_cell: np.ndarray, offset_cell: np.ndarray,
                 flux: Optional[np.ndarray] = None, window: Optional[Window] = None) -> Tuple[np.ndarray, ...]:
    """The blind detection run on the pixels around positions (the injection recovery test): (found, x, y, fnu, snr)
    of the nearest peak within `radius_px`. With `window` (an injected private buffer) the matched image is updated by
    the injected sources' own correlation (`_injected_delta`, mode "full", reach 2h). Identity: `pacs_completeness.
    window_peaks` (pacs_completeness.py:745)."""
    raise NotImplementedError(ROUND2)


def sky_bins(cell_sky: np.ndarray) -> np.ndarray:
    """Each cell's sky-brightness bin (SKY_BINS quantiles over the band's cells) (`pacs_completeness.sky_bins`)."""
    raise NotImplementedError(ROUND2)


# ----------------------------------------------------------------------------------------------------------------------
# the band

@dataclass
class BandDetection:
    """Everything of one band before its injections (the old detect.pkl, now detect.hdf5): `cells` (CellStats),
    `cell_grid`, `shift` (east, north, n_pairs), `src_row` (each SESNA source's row or -1; FROZEN 9 + 4), `cell_bin`,
    `peaks`, `pos_k_arcsec` (position scatter x S/N), `r_match` (matching radius at S/N 5, arcsec), `kernel`,
    `n_src`. Map index is 0 throughout (one mosaic per band); FROZEN 4 keys use it."""
    cells: CellStats
    cell_grid: CellGrid
    shift: Tuple[float, float, int]
    src_row: np.ndarray
    cell_bin: np.ndarray
    peaks: Peaks
    pos_k_arcsec: float
    r_match: float
    kernel: Kernel
    n_src: int


def matching_radius(config, m: Mosaic, detection: Dict[str, TiledLayer], cells: CellStats) -> Tuple[float, np.ndarray]:
    """(pos_k, offsets): the recovery radius constant from bright injections at CALIBRATION_SNR (seeded as in
    `pacs_completeness._calibrate`, pacs_completeness.py:1479), per tile; r_match = MATCH_SIGMAS x pos_k / 5."""
    raise NotImplementedError(ROUND2)


def detect_band(config, region: str, band: str, box: Optional[Sequence[float]], ra: np.ndarray, dec: np.ndarray,
                work_dir: str, n_jobs: Optional[int] = None) -> Optional[BandDetection]:
    """The DETECT stage of one band: open the mosaic, build the detection layers, reference and registration (first
    pass unregistered), cell statistics (registered SESNA positions, FROZEN 9), each source's row (FROZEN 4 key search),
    peaks, matching radius. None when the band's mosaic reaches no source. Writes the detection layers; the caller
    (stages) writes detect.hdf5. Identity: `pacs_completeness._band_detect` (pacs_completeness.py:1529): every array of
    its returned dict bit-identical (map index 0)."""
    raise NotImplementedError(ROUND2)
