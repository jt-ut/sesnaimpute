"""The blind detection of PACS sources on a band's mosaic, tile by tile: the matched (aperture-filtered) image and
its noise, the cell statistics, the registration to SESNA, the peaks, the matching radius.

Depends on: grid, psf, mosaic. Replaces (pacs_completeness.py): filtered, _detection_kernel, detection_strips/
open_detection/detection_strip/_finish_piece/_strip_piece/_layer_strip/_prepare_stage/matched_state, cell_fits/
cell_statistics/ring/block_sigma/cell_offset/_tile_reference/_piece/detection_cell_stats, _local_peaks/_parabola/
_block_peaks/_assemble_peaks/window_peaks/_injected_delta, _calibrate, tile_inputs, sky_bins, geometry_cells,
_tile_geometry, _band_detect; and pacs_fit.registration/registration_pairs, select_deeper. Dead and not carried:
prepare_detection, encode_curves.

The work unit is the 256-pixel tile (grid.TILE). The detection layers are computed per present tile of the mosaic with
a halo of the kernel's half-width h (P70 <= 24, P100 <= 27, P160 <= 15 px), so a tile's pixels equal a whole-grid
correlation's; the cell statistics and the peaks run per cell-statistics block (FROZEN 3) whose pixels touch a present
tile; the matching radius's bright injections are drawn once for the band (their seeds are the band's, FROZEN 2) and
searched per tile. Every id, seed, block and key is a function of the cell id over the whole grid (CONTRACT sec 6).

Store (transient, under the band's work directory `work_dir`): `<work_dir>/detect/{m,ratio,norm}/` TiledLayer float32
fill NaN and `<work_dir>/detect/{valid,pix_valid}/` TiledLayer bool fill False, on the band's PixelGrid; the detection
kernel (h, kc, fc, s_full) in the m layer's index.json "meta" (JSON floats round-trip float64 exactly). The band's
detection record `detect.hdf5` is the caller's (stages.write_record of `BandDetection`).
"""

from __future__ import annotations

import functools
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from sesnaimpute import pieces as pieces_module
from sesnaimpute.definitions import PACS_APERTURES as APERTURES
from sesnaimpute.sky.derived.pacs import grid, mosaic
from sesnaimpute.sky.derived.pacs.grid import TILE, CellGrid, TiledLayer
from sesnaimpute.sky.derived.pacs.mosaic import Mosaic, Window

#: the detection layers, their dtype and fill
DETECTION_LAYERS = {"m": (np.float32, np.nan), "ratio": (np.float32, np.nan), "norm": (np.float32, np.nan),
                    "valid": (np.bool_, False), "pix_valid": (np.bool_, False)}

# the tuning constants of the detection, moved unchanged in value from pacs_completeness.py / pacs_fit.py (CONTRACT sec 3)
#: positions per cell measured for the cell statistics (spec section 5)
N_POS = 200
#: a cell needs this many valid positions to get a row; fewer means the cell is not covered
MIN_VALID_POSITIONS = 100
#: a cell's own noise estimate counts in its 3 x 3 block's median only with this many clean positions
MIN_CLEAN_POSITIONS = 50
#: a cell's matched-image level and noise need this many finite pixels
MIN_MATCHED_PIXELS = 20
MAD_TO_SIGMA = 1.4826
#: the map's reference noise (the fallback for a block with no clean cell) is the median over up to this many evenly
#: spread cells of the grid
REFERENCE_CELLS = 400
#: the cells are ranked by their annulus sky and cut into this many equal bins (the spurious rate per bin)
SKY_BINS = 5
#: the detection kernel's footprint: where the template exceeds this fraction of its peak
ZERO_SUM_FOOTPRINT = 0.01
#: a PACS source is a local maximum of the matched image reaching this many times its cell noise
DETECTION_SNR = 5.0
#: the recovery radius is MATCH_SIGMAS x the detection's position scatter at the injection's S/N, measured on bright
#: injections at CALIBRATION_SNR (seeds [map index + 1, CALIBRATION_SEED_TAG, k])
MATCH_SIGMAS = 3.0
CALIBRATION_SNR = (30.0, 100.0)
CALIBRATION_SEED_TAG = 7777
#: the matching radius is quoted at this S/N (r_match)
MATCH_RADIUS_SNR = 5.0
#: injections: no two artificial sources closer than the injected patch's half-width plus the sky annulus's outer
#: radius, drawn at one per this many squared separations of valid area (inject.py uses the same rule)
SEPARATION_AREAS = 2.0
#: the valid fraction of a band's cells is probed with this many draws (seed 0)
VALID_PROBE_DRAWS = 20000
#: a map is registered to SESNA only with at least this many unique bright-peak / SESNA pairs
MIN_REGISTRATION_PAIRS = 10
#: the injected sources' detection-map change is computed this many sources at a time
DELTA_CHUNK = 256
#: the reference cells' fits are cut into pieces of this many cells (x N_POS = a whole number of the aperture's 2048
#: chunks, so a piece's positions are chunked as the one-piece call chunked them)
REFERENCE_PIECE_CELLS = 256


# ----------------------------------------------------------------------------------------------------------------------
# small helpers

class _CellLookup:
    """A per-cell table indexed like an n_cells array (`table[cell ids]`), NaN for a cell it does not hold; holds only
    the cells a piece needs (sorted `ids`, float64 `values`), so a worker's memory does not grow with the grid."""

    def __init__(self, ids: np.ndarray, values: np.ndarray) -> None:
        self.ids = np.asarray(ids, np.int64)
        self.values = np.asarray(values, np.float64)

    def __getitem__(self, cid) -> np.ndarray:
        cid = np.asarray(cid, np.int64)
        if self.ids.size == 0:
            return np.full(cid.shape, np.nan)
        k = np.clip(np.searchsorted(self.ids, cid), 0, self.ids.size - 1)
        return np.where(self.ids[k] == cid, self.values[k], np.nan)


def _lookups(row_ids: np.ndarray, columns: Sequence[np.ndarray], want: np.ndarray) -> Tuple[_CellLookup, ...]:
    """`_CellLookup`s of `columns` (aligned with the sorted `row_ids`) restricted to the cells `want`."""
    want = np.unique(np.asarray(want, np.int64))
    if row_ids.size == 0 or want.size == 0:
        return tuple(_CellLookup(np.zeros(0, np.int64), np.zeros(0)) for _ in columns)
    k = np.clip(np.searchsorted(row_ids, want), 0, row_ids.size - 1)
    hit = row_ids[k] == want
    return tuple(_CellLookup(want[hit], np.asarray(c, np.float64)[k[hit]]) for c in columns)


def _rows_as_stored(v: np.ndarray) -> np.ndarray:
    """A row column as the stages carry it: float32 (as stored in the product), read back as float64."""
    return np.asarray(v).astype(np.float32).astype(np.float64)


def _inside(path: str, root: str) -> str:
    real, r = os.path.realpath(path), os.path.realpath(root)
    assert real == r or real.startswith(r + os.sep), "%s resolves outside %s" % (path, root)
    return real


def _tile_key(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The tile of each position's nearest pixel, as one int64 key."""
    return (np.floor(y + 0.5).astype(np.int64) // TILE) * (1 << 31) + np.floor(x + 0.5).astype(np.int64) // TILE


# ----------------------------------------------------------------------------------------------------------------------
# the detection layers

@dataclass(frozen=True)
class Kernel:
    """FROZEN 11. The detection kernel: half-width `h` (= Mosaic.injection_half_width()), the template on its
    footprint `kc` and the flipped footprint `fc` (side 2h+1), and `s_full` the zero-sum template's sum of squares."""
    h: int
    kc: np.ndarray
    fc: np.ndarray
    s_full: float

    def to_meta(self) -> dict:
        return dict(h=int(self.h), kc=self.kc.tolist(), fc=self.fc.tolist(), s_full=float(self.s_full))

    @classmethod
    def from_meta(cls, d: dict) -> "Kernel":
        return cls(int(d["h"]), np.asarray(d["kc"], np.float64), np.asarray(d["fc"], np.float64), float(d["s_full"]))


def detection_kernel(m: Mosaic) -> Kernel:
    """FROZEN 11. The weight-mean PSF table (MosaicPSF.evaluate_mean_xy) at the grid centre (ny // 2, nx // 2), on its
    footprint (> ZERO_SUM_FOOTPRINT of its peak), zero-summed. Identity: `pacs_completeness._detection_kernel`
    (pacs_completeness.py:497) bit-identical."""
    h = int(m.injection_half_width())
    side = 2 * h + 1
    cy, cx = m.grid.detection_kernel_centre()
    offs = np.arange(side, dtype=float)
    px = np.broadcast_to((cx - h + offs)[None, None, :], (1, side, side))
    py = np.broadcast_to((cy - h + offs)[None, :, None], (1, side, side))
    k = m.psf.evaluate_mean_xy(np.array([float(cx)]), np.array([float(cy)]), px, py).mean(axis=1)[0]
    foot = k > ZERO_SUM_FOOTPRINT * k.max()
    kf = np.where(foot, k, 0.0)
    s_full = float(((k[foot] - k[foot].mean()) ** 2).sum())
    return Kernel(h, np.ascontiguousarray(kf[::-1, ::-1]), np.ascontiguousarray(foot[::-1, ::-1].astype(float)), s_full)


#: the correlations' dtype (float64: see `_matched`; float32 was measured in the R2 gate)
FFT_DTYPE = np.float64


def _matched(image: np.ndarray, error: np.ndarray, kernel: Kernel, dtype=None):
    """The detection rule on one window (the old `detection_strip` body): D = valid pixels in the footprint, B = sum of
    the template over them, S_valid = of its square less B^2 / D; the zero-sum response A - B C / D scaled by
    sqrt(S_full / S_valid), defined where D >= 2 and S_valid > 1e-9 S_full. Returns (m, ratio, norm, ok, valid) on the
    window (correlations "same"; pixels outside the grid are NaN in the window and enter as zero, as the old strips'
    edges did). `dtype` of the correlations: FFT_DTYPE by default."""
    dt = FFT_DTYPE if dtype is None else dtype
    kc, fc, s_full = kernel.kc.astype(dt), kernel.fc.astype(dt), kernel.s_full

    def corr(a, c):
        return fftconvolve(a, c, mode="same")
    valid = np.isfinite(image) & np.isfinite(error)
    v = valid.astype(dt)
    dd = np.rint(corr(v, fc))
    b = corr(v, kc)
    s_valid = corr(v, kc * kc)
    del v
    s_valid -= np.divide(b * b, dd, out=np.zeros_like(b), where=dd > 0)
    ok = valid & (dd >= 2) & (s_valid > 1e-9 * s_full)
    ratio = np.divide(b, dd, out=np.zeros_like(b), where=ok)
    del b, dd
    norm = np.sqrt(np.divide(s_full, s_valid, out=np.zeros_like(s_valid), where=ok))
    del s_valid
    iv = np.where(valid, image, 0.0).astype(dt)
    m = corr(iv, kc)
    c = corr(iv, fc)
    del iv
    c *= ratio
    m -= c
    del c
    m *= norm
    m = np.where(ok, m, np.nan)
    return m, ratio, norm, ok, valid


def detect_tile(m: Mosaic, kernel: Kernel, iy: int, ix: int, layers: Dict[str, TiledLayer]) -> bool:
    """The detection layers on one tile: read the tile plus a halo of h pixels (Mosaic.tile_window), correlate the
    image and the error with the kernel (the halo makes the tile's interior equal to a whole-grid correlation), write
    m, ratio, norm, valid, pix_valid for the tile's own pixels (layers open "r+"; nothing written for a tile with no
    valid pixel). Returns whether it wrote. Identity: `pacs_completeness.detection_strip` (pacs_completeness.py:546)
    per pixel: m, ratio, norm within 1e-6 relative (FFT summation order), valid and pix_valid identical."""
    h = kernel.h
    w = m.tile_window(iy, ix, h)
    y0, y1, x0, x1 = m.grid.tile_bounds(iy, ix)
    own = (slice(h, h + y1 - y0), slice(h, h + x1 - x0))
    if not (np.isfinite(w.image[own]) & np.isfinite(w.error[own])).any():
        return False
    mm, ratio, norm, ok, valid = _matched(w.image, w.error, kernel)
    for name, a in (("m", mm), ("ratio", ratio), ("norm", norm), ("valid", ok), ("pix_valid", valid)):
        layers[name][y0:y1, x0:x1] = a[own]
    return True


def _detect_piece(tiles, m, kernel, directory):
    """One worker's run of tiles (`build_detection_layers`)."""
    layers = {name: TiledLayer.open(os.path.join(directory, name), "r+") for name in DETECTION_LAYERS}
    return sum(bool(detect_tile(m, kernel, iy, ix, layers)) for iy, ix in tiles)


def build_detection_layers(config, m: Mosaic, work_dir: str, n_jobs: Optional[int] = None) -> Dict[str, TiledLayer]:
    """`detect_tile` over the mosaic's present tiles (joblib, one piece = a run of tiles; per-worker memory one tile
    with halo, independent of the grid size and the worker count), then each layer's write_index. Writes
    `<work_dir>/detect/<layer>/` (an earlier detect/ there is removed first). Returns the layers open "r"."""
    n_jobs = int(n_jobs or getattr(config, "n_jobs", 1) or 1)
    d = os.path.join(work_dir, "detect")
    _inside(d, work_dir)
    if os.path.exists(d):
        shutil.rmtree(d)
    kernel = detection_kernel(m)
    header = m.grid.header_text()
    for name, (dt, fill) in DETECTION_LAYERS.items():
        TiledLayer.create(os.path.join(d, name), m.grid.shape, dt, fill, wcs_header=header,
                          meta=dict(kernel=kernel.to_meta()) if name == "m" else None)
    tiles = list(m.present_tiles())
    pieces = [tiles[a:b] for a, b in pieces_module.slices(len(tiles), n_jobs)] if tiles else []
    pieces_module.run(functools.partial(_detect_piece, m=m, kernel=kernel, directory=d), pieces, n_jobs,
                      unit="tile runs (detection layers)")
    for name in DETECTION_LAYERS:
        TiledLayer.open(os.path.join(d, name), "r+").write_index()
    return open_detection(work_dir)


def open_detection(work_dir: str) -> Dict[str, TiledLayer]:
    """The detection layers of a band, open read-only."""
    d = os.path.join(work_dir, "detect")
    return {name: TiledLayer.open(os.path.join(d, name), "r") for name in DETECTION_LAYERS}


_KERNELS: Dict[str, Kernel] = {}


def _kernel_of(detection: Dict[str, TiledLayer]) -> Kernel:
    """The detection kernel stored with the layers (the m layer's meta), process-local cache by directory."""
    key = detection["m"].directory
    if key not in _KERNELS:
        _KERNELS.clear()
        _KERNELS[key] = Kernel.from_meta(detection["m"].meta["kernel"])
    return _KERNELS[key]


def filtered(m: Mosaic, x: np.ndarray, y: np.ndarray, offset_cell: np.ndarray, window: Optional[Window] = None
             ) -> np.ndarray:
    """The aperture flux (Mosaic.aperture, FROZEN 8) at positions minus the CELL_OFFSET of the position's cell
    (`offset_cell` indexed by cell id, NaN where the cell holds no row); NaN where not covered. `window`: the pixels to
    measure on (an injected private buffer). Identity: `pacs_completeness.filtered`."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    cid = m.cells.cell_ids(x, y)
    off = np.where(cid >= 0, offset_cell[np.maximum(cid, 0)], np.nan)
    a = m.aperture(x, y, window=window)
    return np.where(a.covered, a.flux - off, np.nan)


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


def _cell_fits(m: Mosaic, ids: np.ndarray, seed: int):
    """The aperture measurement at N_POS positions in each cell of `ids` (FROZEN 2 draws). Returns b (flux mJy), gain
    (1 where covered), the annulus sigma, valid, clean (farther than one beam from every bright neighbour) and the
    annulus sky, each (n_cell, N_POS). Identity: `pacs_completeness.cell_fits`."""
    x, y = m.cells.sample_pixels(ids, N_POS, seed)
    a = m.aperture(x.ravel(), y.ravel())
    shape = x.shape
    gain = np.where(a.covered, 1.0, np.nan)
    valid = (a.covered & np.isfinite(a.flux)).reshape(shape)
    clean = ~m.near_bright(x, y)
    return (a.flux.reshape(shape), gain.reshape(shape), a.annulus_sigma.reshape(shape), valid, clean,
            a.sky.reshape(shape))


def _clean_statistics(b, gain, pixel_sigma, valid, clean):
    """Per cell: n_valid, n_clean, clean sigma (1.4826 MAD about the offset), median gain, median annulus sigma, offset
    (the clean median), over the valid positions farther than one beam from every bright neighbour. Identity:
    `pacs_completeness.cell_statistics`."""
    use = valid & clean
    bb = np.where(use, b, np.nan)
    with np.errstate(all="ignore"):
        offset = np.nanmedian(bb, axis=1)
        sigma = MAD_TO_SIGMA * np.nanmedian(np.abs(bb - offset[:, None]), axis=1)
        g = np.nanmedian(np.where(valid, gain, np.nan), axis=1)
        ps = np.nanmedian(np.where(valid, pixel_sigma, np.nan), axis=1)
    return valid.sum(axis=1), use.sum(axis=1), sigma, g, ps, offset


def ring(cells: CellGrid, ids: np.ndarray) -> np.ndarray:
    """The cells of the grid in the 3 x 3 block around each of `ids`, flattened, unique, ascending."""
    cy, cx = np.divmod(np.asarray(ids, np.int64), cells.nx)
    ny = (cy[:, None] + np.repeat([-1, 0, 1], 3)[None, :])
    nx = (cx[:, None] + np.tile([-1, 0, 1], 3)[None, :])
    ok = (ny >= 0) & (ny < cells.ny) & (nx >= 0) & (nx < cells.nx)
    return np.unique((ny * cells.nx + nx)[ok])


def _neighbourhood(cells: CellGrid, ids: np.ndarray, all_ids: np.ndarray):
    cy, cx = np.divmod(np.asarray(ids, np.int64), cells.nx)
    ny = cy[:, None] + np.repeat([-1, 0, 1], 3)[None, :]
    nx = cx[:, None] + np.tile([-1, 0, 1], 3)[None, :]
    inside = (ny >= 0) & (ny < cells.ny) & (nx >= 0) & (nx < cells.nx)
    nb = np.where(inside, ny * cells.nx + nx, -1)
    k = np.clip(np.searchsorted(all_ids, nb), 0, all_ids.size - 1)
    return inside, nb, k


def block_sigma(cells: CellGrid, ids, all_ids, mad_sigma, n_clean):
    """Noise of each cell of `ids`: the median of the MAD sigmas of the cells of its 3 x 3 block (those in `all_ids`,
    sorted, with at least MIN_CLEAN_POSITIONS clean positions); (sigma, clean positions behind it), NaN where no cell of
    the block qualifies. Identity: `pacs_completeness.block_sigma`."""
    inside, nb, k = _neighbourhood(cells, ids, all_ids)
    found = inside & (all_ids[k] == nb)
    good = found & (n_clean[k] >= MIN_CLEAN_POSITIONS) & np.isfinite(mad_sigma[k]) & (mad_sigma[k] > 0)
    with np.errstate(all="ignore"):
        sigma = np.nanmedian(np.where(good, mad_sigma[k], np.nan), axis=1)
    return sigma, np.where(good, n_clean[k], 0).sum(axis=1)


def cell_offset(cells: CellGrid, ids, all_ids, offset, n_clean):
    """Offset of each cell of `ids`: its own clean median, or with fewer than MIN_CLEAN_POSITIONS clean positions the
    median of its 3 x 3 block's qualifying cells, else 0. Identity: `pacs_completeness.cell_offset`."""
    k = np.searchsorted(all_ids, ids)
    own = np.where((n_clean[k] >= MIN_CLEAN_POSITIONS) & np.isfinite(offset[k]), offset[k], np.nan)
    if np.isnan(own).any():
        inside, nbid, kk = _neighbourhood(cells, ids, all_ids)
        good = inside & (all_ids[kk] == nbid) & (n_clean[kk] >= MIN_CLEAN_POSITIONS) & np.isfinite(offset[kk])
        with np.errstate(all="ignore"):
            block = np.nanmedian(np.where(good, offset[kk], np.nan), axis=1)
        own = np.where(np.isnan(own), np.nan_to_num(block, nan=0.0), own)
    return own


def _matched_cell_stats(m: Mosaic, layer: TiledLayer, ids: np.ndarray):
    """Per cell of `ids`: the matched image's level (median) and noise (1.4826 MAD) over the cell's finite pixels, and
    their count; one window read for all the cells. Identity: `pacs_completeness.detection_cell_stats`."""
    ny_, nx_ = m.grid.shape
    x0, x1, y0, y1 = m.cells.bounds(ids)
    a0 = np.maximum(np.ceil(x0).astype(np.int64), 0)
    a1 = np.minimum(np.floor(x1).astype(np.int64) + 1, nx_)
    b0 = np.maximum(np.ceil(y0).astype(np.int64), 0)
    b1 = np.minimum(np.floor(y1).astype(np.int64) + 1, ny_)
    level, mad, cnt = np.full(ids.size, np.nan), np.full(ids.size, np.nan), np.zeros(ids.size, np.int64)
    if ids.size == 0:
        return level, mad, cnt
    Y0, X0 = int(b0.min()), int(a0.min())
    w = layer.window(Y0, int(b1.max()), X0, int(a1.max()))
    for i in range(ids.size):
        v = w[b0[i] - Y0:b1[i] - Y0, a0[i] - X0:a1[i] - X0]
        v = v[np.isfinite(v)]
        cnt[i] = v.size
        if v.size >= MIN_MATCHED_PIXELS:
            level[i] = np.median(v)
            mad[i] = MAD_TO_SIGMA * np.median(np.abs(v - level[i]))
    return level, mad, cnt


def _reference_piece(ids, m):
    """The reference's fits on one run of its cells: (n_valid, n_clean, mad) per cell (seed 0)."""
    b, gain, ps, valid, clean, _sky = _cell_fits(m, ids, 0)
    n_valid, n_clean, mad, _, _, _ = _clean_statistics(b, gain, ps, valid, clean)
    return n_valid, n_clean, mad


def _cell_reference(config, m: Mosaic, cells: np.ndarray, n_jobs: int) -> dict:
    used = np.asarray(cells, np.int64)
    if used.size == 0:
        return dict(n_used=0, shift=(0.0, 0.0, 0), median=np.nan)
    centres = used[np.unique(np.linspace(0, used.size - 1, min(REFERENCE_CELLS, used.size)).astype(np.int64))]
    all_ids = ring(m.cells, centres)
    runs = [all_ids[a:a + REFERENCE_PIECE_CELLS] for a in range(0, all_ids.size, REFERENCE_PIECE_CELLS)]
    parts = pieces_module.run(functools.partial(_reference_piece, m=m), runs, n_jobs, unit="cell runs (reference)")
    n_valid = np.concatenate([p[0] for p in parts])
    n_clean = np.concatenate([p[1] for p in parts])
    mad = np.concatenate([p[2] for p in parts])
    sigma, _ = block_sigma(m.cells, centres, all_ids, mad, n_clean)
    c = np.searchsorted(all_ids, centres)
    good = np.isfinite(sigma) & (sigma > 0) & (n_valid[c] == N_POS)
    # the old build applies no registration (pacs_completeness.py:237: "the PACS products keep Herschel's own
    # astrometry"), so FROZEN 9's shift is (0, 0, 0) here; `registration` is the function that would set it
    return dict(n_used=int(used.size), shift=(0.0, 0.0, 0), median=float(np.median(sigma[good])) if good.any() else np.nan)


def cell_reference(config, m: Mosaic, cells: np.ndarray, ra: np.ndarray, dec: np.ndarray) -> dict:
    """The mosaic's reference noise (median cell sigma over up to REFERENCE_CELLS evenly spread cells of `cells`, seed
    0) and its registration shift. Identity: `pacs_completeness._tile_reference` (pacs_completeness.py:235): the shift
    is (0, 0, 0) there (no registration in the old build; `ra`, `dec` are not read), and so here."""
    return _cell_reference(config, m, cells, int(getattr(config, "n_jobs", 1) or 1))


def registration(m: Mosaic, ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[float, float, int]:
    """FROZEN 9. The map's astrometric offset from SESNA (east arcsec, north arcsec, n_pairs): median of PACS minus
    SESNA over mutual nearest pairs of bright peaks (Mosaic.neighbours) and SESNA positions within the local beam;
    fewer than MIN_REGISTRATION_PAIRS: (0, 0, n). Identity: `pacs_fit.registration` + `registration_pairs`. Not
    applied by the build (see `cell_reference`)."""
    nb = m.neighbours()
    ra_deg, dec_deg = np.asarray(ra_deg, float), np.asarray(dec_deg, float)
    if nb.x.size == 0 or ra_deg.size == 0:
        delta = np.zeros((0, 2))
    else:
        sky = m.grid.wcs.pixel_to_world(nb.x, nb.y).icrs
        pra, pde = sky.ra.deg, sky.dec.deg
        ra0, de0 = np.median(pra), np.median(pde)

        def plane(r, d):
            return np.c_[((r - ra0 + 180.0) % 360.0 - 180.0) * np.cos(np.radians(de0)) * 3600.0, (d - de0) * 3600.0]
        peaks, sesna = plane(pra, pde), plane(ra_deg, dec_deg)
        beam_px_max = float(np.max(m.psf.sub_beams)) / m.pixel_arcsec
        d1, i1 = cKDTree(sesna).query(peaks, distance_upper_bound=beam_px_max * m.pixel_arcsec)
        ok = np.isfinite(d1)
        ok[ok] = d1[ok] < m.beam_at(nb.x[ok], nb.y[ok])
        if not ok.any():
            delta = np.zeros((0, 2))
        else:
            _, back = cKDTree(peaks).query(sesna[i1[ok]])
            mutual = np.flatnonzero(ok)[back == np.flatnonzero(ok)]
            delta = peaks[mutual] - sesna[i1[mutual]]
    if delta.shape[0] < MIN_REGISTRATION_PAIRS:
        return 0.0, 0.0, int(delta.shape[0])
    return float(np.median(delta[:, 0])), float(np.median(delta[:, 1])), int(delta.shape[0])


def cell_block(config, m: Mosaic, detection: Dict[str, TiledLayer], ids: np.ndarray, reference: dict) -> dict:
    """One cell-statistics block (FROZEN 3: the cells of one `CellGrid.stat_block` id plus a one-cell border): the
    cells' fits at N_POS positions each (`CellGrid.sample_pixels(ids, N_POS, seed)`, seed = map index + 1 = 1, FROZEN 2),
    the clean statistics, the 3 x 3 ring noise and offsets, the matched image's level and MAD. Reads the mosaic around
    the block's positions and one window of the detection layer m. Identity: `pacs_completeness._piece` + cell_fits/
    cell_statistics/block_sigma/cell_offset/detection_cell_stats, bit-identical per cell (the m-derived columns to the
    m layer's 1e-6)."""
    cells = m.cells
    ids = np.asarray(ids, np.int64)
    all_ids = ring(cells, ids)
    b, gain, ps, valid, clean, sky = _cell_fits(m, all_ids, 1)
    m_off, m_mad, m_clean = _matched_cell_stats(m, detection["m"], all_ids)
    n_valid, n_clean, mad, _g, _pps, off = _clean_statistics(b, gain, ps, valid, clean)
    sigma, n_sigma = block_sigma(cells, ids, all_ids, mad, n_clean)
    offset = cell_offset(cells, ids, all_ids, off, n_clean)
    # a block with no clean cell (wholly within a beam of bright sources) takes the reference noise; N_SIGMA_POS 0 marks it
    sigma = np.where(np.isfinite(sigma), sigma, reference["median"])
    m_sigma, _ = block_sigma(cells, ids, all_ids, m_mad, m_clean)
    with np.errstate(all="ignore"):
        m_sigma = np.where(np.isfinite(m_sigma), m_sigma,
                           np.nanmedian(m_sigma) if np.isfinite(m_sigma).any() else np.nan)
    m_offset = cell_offset(cells, ids, all_ids, m_off, m_clean)
    k = np.searchsorted(all_ids, ids)
    n_valid = n_valid[k]
    with np.errstate(all="ignore"):
        cell_sky = np.nanmedian(np.where(valid, sky, np.nan), axis=1)[k]
    covered = (n_valid >= MIN_VALID_POSITIONS) & np.isfinite(sigma) & (sigma > 0)
    ra, dec = grid.from_map_frame(*cells.centres_icrs(ids), reference["shift"][:2])
    return dict(ids=ids, covered=covered, n_sigma=np.minimum(n_sigma, np.iinfo(np.int16).max).astype(np.int16),
                sigma=sigma, offset=offset, sky=cell_sky, ra=ra, dec=dec, m_sigma=m_sigma, m_offset=m_offset)


def _stat_blocks(m: Mosaic) -> List[np.ndarray]:
    """The cell-statistics blocks (FROZEN 3) in ascending block id whose pixels touch a present tile, each its cells
    ascending. Every cell of the grid is enumerated (C sec 5 item 8); a block touching no present tile has no valid
    pixel, so no cell of it can be covered (a covered position needs 90 % of its aperture on valid pixels) and it is
    skipped. Built from the block geometry, not by masking all n_cells per block."""
    cells = m.cells
    present = set(map(tuple, m.present_tiles()))
    if not present:
        return []
    bc = grid.BLOCK_CELLS
    out = []
    for by in range(-(-cells.ny // bc)):
        lines = np.arange(by * bc, min((by + 1) * bc, cells.ny), dtype=np.int64)
        for bx in range(-(-cells.nx // bc)):
            cols = np.arange(bx * bc, min((bx + 1) * bc, cells.nx), dtype=np.int64)
            ids = (lines[:, None] * cells.nx + cols[None, :]).ravel()
            x0, x1, y0, y1 = cells.bounds(ids[[0, -1]])
            px0, px1 = int(np.floor(x0[0])), int(np.ceil(x1[1]))
            py0, py1 = int(np.floor(y0[0])), int(np.ceil(y1[1]))
            ty = range(max(py0, 0) // TILE, max(min(py1, m.grid.ny - 1), 0) // TILE + 1)
            tx = range(max(px0, 0) // TILE, max(min(px1, m.grid.nx - 1), 0) // TILE + 1)
            if any((iy, ix) in present for iy in ty for ix in tx):
                out.append(ids)
    return out


def cell_statistics(config, m: Mosaic, detection: Dict[str, TiledLayer], shift: Tuple[float, float],
                    reference: dict, n_jobs: Optional[int] = None) -> CellStats:
    """`cell_block` over every stat block of the grid holding a covered tile, in block order (FROZEN 3), joblib; then the
    covered rows ordered by cell id. Cells are enumerated over the whole grid (C sec 5 item 8) but a block with no
    present tile is skipped (it has no covered cell). Identity: the rows, CELL_SIGMA, CELL_OFFSET, m_sigma, m_offset of
    `_band_detect` (pacs_completeness.py:1556-1590) bit-identical (m_sigma, m_offset to the m layer's 1e-6)."""
    n_jobs = int(n_jobs or getattr(config, "n_jobs", 1) or 1)
    old = tuple(reference.get("shift", (0.0, 0.0, 0)))
    ref = dict(reference, shift=(float(shift[0]), float(shift[1])) + old[2:])
    blocks = _stat_blocks(m)
    results = pieces_module.run(functools.partial(cell_block, config, m, detection, reference=ref), blocks, n_jobs,
                                unit="cell blocks")
    names = ("covered", "n_sigma", "sigma", "offset", "sky", "ra", "dec", "m_sigma", "m_offset")
    if not results:
        z = np.zeros(0)
        return CellStats(np.zeros(0, np.int64), np.zeros(0, np.int16), z, z, z, z, z, z, z, np.zeros(0, bool))
    ids = np.concatenate([r["ids"] for r in results])
    order = np.argsort(ids)
    d = {k: np.concatenate([r[k] for r in results])[order] for k in names}
    keep = d["covered"]
    return CellStats(ids=ids[order][keep], **{k: d[k][keep] for k in names})


def sky_bins(cell_sky: np.ndarray) -> np.ndarray:
    """Each cell's sky-brightness bin (SKY_BINS equal bins by rank of the annulus sky over the band's cells, 0
    faintest; PACS maps have no absolute zero level). Identity: `pacs_completeness.sky_bins` for one map."""
    cell_sky = np.asarray(cell_sky)
    if cell_sky.size == 0:
        return np.zeros(0, np.int64)
    rank = np.argsort(np.argsort(np.nan_to_num(cell_sky, nan=np.nanmedian(cell_sky)), kind="stable"))
    return np.minimum(rank * SKY_BINS // max(cell_sky.size, 1), SKY_BINS - 1).astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# peaks

@dataclass
class Peaks:
    """The band's PACS detections: grid pixel x, y, fnu (mJy), cell id, row (into CellStats), snr, sky bin, ra, dec
    (ICRS, SESNA frame: the registration shift removed), and the inverted-map peak count per sky bin (`neg_count`,
    SKY_BINS entries: local maxima of the negated matched image reaching DETECTION_SNR, by the sky bin of their cell;
    information only, the old build counted none)."""
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


def _local_peaks(v, thr):
    """8-neighbour local maxima (along the last two axes) of `v` that reach `thr`; NaN is never a peak."""
    vv = np.where(np.isfinite(v), v, -np.inf)
    size = (1,) * (v.ndim - 2) + (3, 3)
    mx = maximum_filter(vv, size=size, mode="constant", cval=-np.inf)
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (vv >= mx) & (v >= thr)


def _parabola(v, iy, ix):
    """The sub-pixel peak of the 2-D array `v` at its local maximum (iy, ix): a parabola through the pixel and its two
    neighbours on each axis, as offsets (dy, dx) clipped to half a pixel."""
    c, l, r, d, u = v[iy, ix], v[iy, ix - 1], v[iy, ix + 1], v[iy - 1, ix], v[iy + 1, ix]
    with np.errstate(all="ignore"):
        dx = 0.5 * (l - r) / (l - 2 * c + r)
        dy = 0.5 * (d - u) / (d - 2 * c + u)
    return float(np.clip(np.nan_to_num(dy), -0.5, 0.5)), float(np.clip(np.nan_to_num(dx), -0.5, 0.5))


def block_peaks(config, m: Mosaic, detection: Dict[str, TiledLayer], ids: np.ndarray, cell_tables: Tuple[np.ndarray, ...]
                ) -> dict:
    """The detections of one stat block: 8-neighbour local maxima of the matched image less its cell level reaching
    DETECTION_SNR x its cell noise inside the block's cells (parabolic sub-pixel position), their aperture flux
    (`filtered`); and the local maxima of the negated image (count only, `neg_cell`). `cell_tables` = (offset, sigma,
    m_offset, m_sigma), each indexed by cell id (an n_cells array, or a `_CellLookup` holding the block's ring: the
    per-block slice). Identity: `pacs_completeness._block_peaks` (pacs_completeness.py:792)."""
    offset_cell, _sigma_cell, m_off, m_sig = cell_tables
    cells = m.cells
    ids = np.asarray(ids, np.int64)
    x0, x1, y0, y1 = cells.bounds(ids)
    ny, nx = m.grid.shape
    ix0, ix1 = max(int(np.floor(x0.min())) - 1, 0), min(int(np.ceil(x1.max())) + 1, nx - 1)
    iy0, iy1 = max(int(np.floor(y0.min())) - 1, 0), min(int(np.ceil(y1.max())) + 1, ny - 1)
    gx, gy = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1))
    cid = cells.cell_ids(gx, gy)
    core = np.isin(cid, ids)
    good = cid >= 0
    mm = detection["m"].window(iy0, iy1 + 1, ix0, ix1 + 1)
    v = np.where(good, mm - m_off[np.maximum(cid, 0)], np.nan)
    thr = np.where(good, DETECTION_SNR * m_sig[np.maximum(cid, 0)], np.nan)

    def detect(vv):
        pk = _local_peaks(vv, thr) & core
        pk[0, :] = pk[-1, :] = False
        pk[:, 0] = pk[:, -1] = False
        return np.argwhere(pk)

    where = detect(v)
    neg = detect(-v)
    xs, ys = [], []
    for iy, ix in where:
        dy, dx = _parabola(v, iy, ix)
        xs.append(gx[iy, ix] + dx)
        ys.append(gy[iy, ix] + dy)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    fnu = filtered(m, xs, ys, offset_cell) if xs.size else np.zeros(0)
    ok = np.isfinite(fnu)
    return dict(x=xs[ok], y=ys[ok], fnu=fnu[ok], cell=cells.cell_ids(xs[ok], ys[ok]),
                neg_cell=cid[neg[:, 0], neg[:, 1]] if neg.size else np.zeros(0, np.int64))


def _block_peaks_piece(ids, config, m, detection, row_ids, columns):
    """block_peaks with the block's ring of the row tables (`_lookups`)."""
    return block_peaks(config, m, detection, ids, _lookups(row_ids, columns, ring(m.cells, ids)))


def _injected_delta(m: Mosaic, kernel: Kernel, rect: dict, x, y, flux, wpx, wpy):
    """The change of the detection map at the window pixels (wpx, wpy) (n, side, side) of each injection, from that
    injection's own template: the operator of the detection rule with each pixel's own valid-footprint factors applied
    to the template on the valid pixels. `rect`: the detection layers ratio, norm, pix_valid read over a rectangle
    (origin y0, x0) holding every pixel needed. Identity: `pacs_completeness._injected_delta`."""
    h = kernel.h
    side = 2 * h + 1
    ny, nx = m.grid.shape
    Y0, X0 = rect["y0"], rect["x0"]
    offs = np.arange(side, dtype=float)
    cx = np.rint(x).astype(np.int64) - h
    cy = np.rint(y).astype(np.int64) - h
    out = np.zeros(wpx.shape)
    for lo in range(0, x.size, DELTA_CHUNK):
        sl = slice(lo, lo + DELTA_CHUNK)
        px = np.broadcast_to(cx[sl][:, None, None] + offs[None, None, :], (cx[sl].size, side, side))
        py = np.broadcast_to(cy[sl][:, None, None] + offs[None, :, None], (cx[sl].size, side, side))
        resp = m.psf.evaluate_xy(x[sl], y[sl], px, py).mean(axis=1) * flux[sl][:, None, None]
        # only the injected light on valid pixels enters, as for the real image
        ipx, ipy = px.astype(np.int64), py.astype(np.int64)
        on = (ipx >= 0) & (ipx < nx) & (ipy >= 0) & (ipy < ny)
        pv = np.zeros(resp.shape, bool)
        pv[on] = rect["pix_valid"][ipy[on] - Y0, ipx[on] - X0]
        resp = np.where(pv, resp, 0.0)
        full_k = fftconvolve(resp, kernel.kc[None], mode="full", axes=(1, 2))
        full_f = fftconvolve(resp, kernel.fc[None], mode="full", axes=(1, 2))
        # full index of map pixel p: p - (patch origin) + h
        iy = wpy[sl] - cy[sl][:, None, None] + h
        ix = wpx[sl] - cx[sl][:, None, None] + h
        inside = (iy >= 0) & (iy < full_k.shape[1]) & (ix >= 0) & (ix < full_k.shape[2])
        inside &= (wpx[sl] >= 0) & (wpx[sl] < nx) & (wpy[sl] >= 0) & (wpy[sl] < ny)
        vals = np.zeros(iy.shape)
        k = np.nonzero(inside)
        my, mx = wpy[sl][k] - Y0, wpx[sl][k] - X0
        vals[k] = (full_k[k[0], iy[k], ix[k]] - rect["ratio"][my, mx] * full_f[k[0], iy[k], ix[k]]) * rect["norm"][my, mx]
        out[sl] = vals
    return out


def _search_group(m, detection, kernel, x, y, rad, m_offset_cell, m_sigma_cell, flux):
    """`_peak_search` on positions of one tile: one rectangle read of the detection layers around them."""
    n = x.size
    hh = int(np.ceil(rad.max())) + 2
    off = np.arange(-hh, hh + 1)
    cx, cy = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
    px = np.ascontiguousarray(np.broadcast_to(cx[:, None, None] + off[None, None, :], (n, off.size, off.size)))
    py = np.ascontiguousarray(np.broadcast_to(cy[:, None, None] + off[None, :, None], (n, off.size, off.size)))
    dist = np.hypot(px - x[:, None, None], py - y[:, None, None])
    ny, nx = m.grid.shape
    r3 = rad[:, None, None]
    use = (dist <= r3 + 1.5) & (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    reach = max(hh, kernel.h) + 1
    Y0, Y1 = int(cy.min()) - reach, int(cy.max()) + reach + 1
    X0, X1 = int(cx.min()) - reach, int(cx.max()) + reach + 1
    rect = dict(y0=Y0, x0=X0, m=detection["m"].window(Y0, Y1, X0, X1))
    v = np.full(px.shape, np.nan)
    v[use] = rect["m"][py[use] - Y0, px[use] - X0]
    if flux is not None:
        for name in ("ratio", "norm", "pix_valid"):
            rect[name] = detection[name].window(Y0, Y1, X0, X1)
        v = v + _injected_delta(m, kernel, rect, x, y, flux, px, py)
    cid = m.cells.cell_ids(px, py)
    good = cid >= 0
    v = np.where(good, v - m_offset_cell[np.maximum(cid, 0)], np.nan)
    thr = np.where(good, DETECTION_SNR * m_sigma_cell[np.maximum(cid, 0)], np.nan)
    pk = _local_peaks(v, thr) & (dist <= r3)
    xc, yc, dd = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    cand = np.where(pk, dist, np.inf).reshape(n, -1)
    best = np.argmin(cand, axis=1)
    has = np.isfinite(cand[np.arange(n), best])
    for i in np.flatnonzero(has):
        iy, ix = divmod(int(best[i]), off.size)
        dy, dx = _parabola(v[i], iy, ix)
        xc[i], yc[i] = px[i, iy, ix] + dx, py[i, iy, ix] + dy
    dd[has] = np.hypot(xc[has] - x[has], yc[has] - y[has])
    return has, xc, yc, dd


def _peak_search(m, detection, kernel, x, y, rad, m_offset_cell, m_sigma_cell, flux=None):
    """For each position: the nearest local maximum of the matched image (plus each injection's own change when
    `flux` is given), less its cell level, reaching DETECTION_SNR x its cell noise, within `rad` (pixels; pixels within
    rad + 1.5 enter the 8-neighbour test); parabolic sub-pixel position. Grouped by tile: each group reads one rectangle
    of the layers. Returns (has, xc, yc, distance). Per position independent of the grouping (a wider window adds only
    pixels outside rad + 1.5, which are NaN in the test, as in the old one-window call)."""
    n = x.size
    has, xc, yc, dd = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if n == 0:
        return has, xc, yc, dd
    key = _tile_key(x, y)
    for k in np.unique(key):
        g = np.flatnonzero(key == k)
        r = _search_group(m, detection, kernel, x[g], y[g], rad[g], m_offset_cell, m_sigma_cell,
                          None if flux is None else flux[g])
        has[g], xc[g], yc[g], dd[g] = r
    return has, xc, yc, dd


def window_peaks(m: Mosaic, detection: Dict[str, TiledLayer], x: np.ndarray, y: np.ndarray, radius_px: np.ndarray,
                 m_offset_cell: np.ndarray, m_sigma_cell: np.ndarray, offset_cell: np.ndarray,
                 flux: Optional[np.ndarray] = None, window: Optional[Window] = None) -> Tuple[np.ndarray, ...]:
    """The blind detection run on the pixels around positions (the injection recovery test): (found, x, y, fnu,
    distance) of the nearest peak within `radius_px` (one radius, or one per position); found = a peak within the
    radius; fnu = `filtered` there, on `window` when given (an injected private buffer). With `flux` the matched image
    is updated by the injected sources' own correlation (`_injected_delta`, mode "full", reach 2h). The per-cell tables
    are indexed by cell id (arrays or `_CellLookup`s). Reads the layers per tile of the positions (one rectangle each).
    Identity: `pacs_completeness.window_peaks` (pacs_completeness.py:745)."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = x.size
    found, fnu = np.zeros(n, bool), np.full(n, np.nan)
    if n == 0:
        return found, np.full(0, np.nan), np.full(0, np.nan), fnu, np.full(0, np.nan)
    rad = np.broadcast_to(np.asarray(radius_px, float), (n,)).astype(float)
    fl = None if flux is None else np.broadcast_to(np.asarray(flux, float), (n,)).astype(float)
    has, xc, yc, dd = _peak_search(m, detection, _kernel_of(detection), x, y, rad, m_offset_cell, m_sigma_cell, fl)
    if has.any():
        fnu[has] = filtered(m, xc[has], yc[has], offset_cell, window=window)
        found[has] = dd[has] <= rad[has]
    return found, xc, yc, fnu, dd


# ----------------------------------------------------------------------------------------------------------------------
# the matching radius

def _draw(cells: CellGrid, ids: np.ndarray, rng, rows: np.ndarray, n: int):
    """n positions, each uniform inside the cell of a randomly chosen row of `rows` (`Injector._draw`)."""
    x0, x1, y0, y1 = cells.bounds(ids[rows[rng.integers(0, rows.size, n)]])
    return x0 + (x1 - x0) * rng.random(n), y0 + (y1 - y0) * rng.random(n)


def _valid(m: Mosaic, row_ids: np.ndarray, x, y):
    """Positions inside a row's cell, on a pixel of valid data (`Injector._valid`)."""
    cid = m.cells.cell_ids(x, y)
    ok = cid >= 0
    if row_ids.size:
        k = np.clip(np.searchsorted(row_ids, cid[ok]), 0, row_ids.size - 1)
        ok[ok] = row_ids[k] == cid[ok]
    else:
        ok[:] = False
    px = np.rint(x).astype(np.int64)
    py = np.rint(y).astype(np.int64)
    ny, nx = m.grid.shape
    ok &= (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    ok[ok] = np.isfinite(m.pixel_sigma(px[ok].astype(float), py[ok].astype(float)))
    return ok


def _thin(x, y, limit, separation_px):
    """Greedy selection, in order, of at most `limit` positions no two closer than `separation_px`
    (`Injector._thin`)."""
    n = x.size
    if n == 0:
        return np.zeros(0, np.int64)
    pairs = cKDTree(np.c_[x, y]).query_pairs(separation_px, output_type="ndarray").reshape(-1, 2)
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


def _calibration_piece(piece, m, detection, row_ids, columns):
    """The bright injections of one tile: covered (the aperture's coverage, which depends only on which pixels are
    valid, so the injected light does not change it), and the distance of the found peak (`_peak_search` with the
    injections' own change). Returns (covered, distance)."""
    x, y, flux, rad = piece
    covered = m.aperture(x, y).covered
    kernel = _kernel_of(detection)
    reach = int(np.ceil(rad.max())) + 3
    cx, cy = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
    want = m.cells.cells_in_window(int(cy.min()) - reach, int(cy.max()) + reach + 1,
                                   int(cx.min()) - reach, int(cx.max()) + reach + 1)
    m_offset, m_sigma = _lookups(row_ids, columns, want)
    _has, _xc, _yc, dd = _peak_search(m, detection, kernel, x, y, rad, m_offset, m_sigma, flux)
    return covered, dd


def matching_radius(config, m: Mosaic, detection: Dict[str, TiledLayer], cells: CellStats,
                    peaks: Optional["Peaks"] = None, n_jobs: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """(pos_k, offsets): the recovery radius constant from bright injections at CALIBRATION_SNR, seeded as in
    `pacs_completeness._calibrate` (pacs_completeness.py:1479: rng [map index + 1, 7777, k], the valid-fraction probe
    rng 0, positions farther than a beam from every PACS source of `peaks`), drawn once for the band and searched per
    tile; offsets = found distance (arcsec) x S/N; pos_k = sqrt(median(offsets^2) / ln 2); r_match = MATCH_SIGMAS x pos_k
    / 5. `peaks`, `n_jobs`: added to the round-1 signature (the old Injector held the peaks; REPORT_R2_B.md)."""
    n_jobs = int(n_jobs or getattr(config, "n_jobs", 1) or 1)
    cg = m.cells
    ids = np.asarray(cells.ids, np.int64)
    n = ids.size
    if n == 0:
        return np.nan, np.zeros(0)
    sigma = _rows_as_stored(cells.sigma)
    columns = (_rows_as_stored(cells.m_offset), _rows_as_stored(cells.m_sigma))
    separation_px = m.injection_half_width() + APERTURES[m.band][2] / m.pixel_arcsec
    det_tree = None
    if peaks is not None and np.asarray(peaks.x).size:
        det_tree = cKDTree(np.c_[np.asarray(peaks.x, float), np.asarray(peaks.y, float)])
    cell_px = float(cg.step_x * cg.step_y)
    rows = np.arange(n)
    probe = _draw(cg, ids, np.random.default_rng(0), rows, VALID_PROBE_DRAWS)
    valid_fraction = float(_valid(m, ids, *probe).mean())
    pieces, snrs = [], []
    for k, snr in enumerate(CALIBRATION_SNR):
        rng = np.random.default_rng([1, CALIBRATION_SEED_TAG, k])
        target = max(int(round(n * cell_px * valid_fraction / (SEPARATION_AREAS * separation_px ** 2))), 1)
        x, y = _draw(cg, ids, rng, rows, int(2.5 * target) + 8)
        ok = _valid(m, ids, x, y)
        x, y = x[ok], y[ok]
        keep = _thin(x, y, target, separation_px)
        x, y = x[keep], y[keep]
        d = np.full(x.size, np.inf) if det_tree is None else det_tree.query(np.c_[x, y])[0]
        far = d >= m.beam_px_at(x, y)
        x, y = x[far], y[far]
        if x.size == 0:
            continue
        cid = cg.cell_ids(x, y)
        kk = np.clip(np.searchsorted(ids, cid), 0, n - 1)
        flux = snr * np.where(ids[kk] == cid, sigma[kk], np.nan)
        rad = m.beam_px_at(x, y)
        key = _tile_key(x, y)
        for t in np.unique(key):
            g = np.flatnonzero(key == t)
            pieces.append((x[g], y[g], flux[g], rad[g]))
            snrs.append(snr)
    res = pieces_module.run(functools.partial(_calibration_piece, m=m, detection=detection, row_ids=ids,
                                              columns=columns), pieces, n_jobs, unit="tiles (matching radius)")
    out = []
    for (covered, dd), snr in zip(res, snrs):
        dd = dd[covered]
        good = np.isfinite(dd)
        out.append(dd[good] * m.pixel_arcsec * snr)
    d_inj = np.concatenate(out) if out else np.zeros(0)
    pos_k = float(np.sqrt(np.median(d_inj ** 2) / np.log(2.0))) if d_inj.size else np.nan
    return pos_k, d_inj


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


def _source_rows(cells: CellStats, src_cell: np.ndarray) -> np.ndarray:
    """Each source's row (FROZEN 4 key search, map index 0) or -1: the cell it falls in, where that cell has a row with
    a finite positive CELL_SIGMA (`select_deeper` with one map; `_band_detect` pacs_completeness.py:1580-1595)."""
    n_src = src_cell.size
    keys = CellGrid.peak_key(0, cells.ids)
    n_row = keys.size
    src_row = np.full(n_src, -1, np.int64)
    idx = np.flatnonzero(src_cell >= 0)
    if idx.size == 0 or n_row == 0:
        return src_row
    q = CellGrid.peak_key(0, src_cell[idx])
    k = np.clip(np.searchsorted(keys, q), 0, n_row - 1)
    hit = keys[k] == q
    sig = np.where(hit, cells.sigma[k], np.nan)
    sel = hit & np.isfinite(sig) & (sig > 0)
    src_row[idx[sel]] = np.searchsorted(keys, q[sel])
    return src_row


def _band_peaks(config, m, detection, cells: CellStats, cell_bin: np.ndarray, shift, n_jobs) -> Peaks:
    """`block_peaks` over the stat blocks (block order), assembled: rows (FROZEN 4 key search), S/N, sky bin, ICRS
    position (`_assemble_peaks` for one map), the inverted-map counts per sky bin."""
    row_ids = cells.ids
    columns = (_rows_as_stored(cells.offset), _rows_as_stored(cells.sigma), _rows_as_stored(cells.m_offset),
               _rows_as_stored(cells.m_sigma))
    blocks = _stat_blocks(m) if row_ids.size else []
    found = pieces_module.run(functools.partial(_block_peaks_piece, config=config, m=m, detection=detection,
                                                row_ids=row_ids, columns=columns), blocks, n_jobs,
                              unit="cell blocks (PACS sources)")

    def cat(key, dt):
        return np.concatenate([d[key] for d in found]).astype(dt) if found else np.zeros(0, dt)
    x, y, f, c, neg = cat("x", float), cat("y", float), cat("fnu", float), cat("cell", np.int64), cat("neg_cell", np.int64)
    keys = CellGrid.peak_key(0, row_ids)
    n_row = keys.size
    q = CellGrid.peak_key(0, c)
    row = np.clip(np.searchsorted(keys, q), 0, max(n_row - 1, 0))
    ok = (keys[row] == q) if n_row else np.zeros(q.size, bool)
    x, y, f, c, row = x[ok], y[ok], f[ok], c[ok], row[ok]
    snr = f / cells.sigma.astype(np.float32)[row].astype(float)
    ra, dec = np.full(x.size, np.nan), np.full(x.size, np.nan)
    if x.size:
        wr, wd = grid.pixel_to_sky(m.grid.wcs, x, y)
        ra, dec = grid.from_map_frame(wr, wd, shift[:2])
    if n_row:
        nk = np.clip(np.searchsorted(row_ids, neg), 0, n_row - 1)
        neg_count = np.bincount(cell_bin[nk[row_ids[nk] == neg]], minlength=SKY_BINS).astype(np.int64)
        bins = cell_bin[row]
    else:
        neg_count, bins = np.zeros(SKY_BINS, np.int64), np.zeros(0, np.int64)
    return Peaks(x=x, y=y, fnu=f, cell=c, row=row, snr=snr, bin=bins, ra=ra, dec=dec, neg_count=neg_count)


def detect_band(config, region: str, band: str, box: Optional[Sequence[float]], ra: np.ndarray, dec: np.ndarray,
                work_dir: str, n_jobs: Optional[int] = None) -> Optional[BandDetection]:
    """The DETECT stage of one band: open the mosaic, build the detection layers, reference and registration (first
    pass unregistered), cell statistics (registered SESNA positions, FROZEN 9), each source's row (FROZEN 4 key search),
    peaks, matching radius. None when the band has no mosaic or its grid no cell. Writes the detection layers; the caller
    (stages) writes detect.hdf5. Identity: `pacs_completeness._band_detect` (pacs_completeness.py:1529): every array of
    its returned dict bit-identical (map index 0), the m-derived ones (m_sigma, m_offset, peak positions, pos_k) to the
    m layer's 1e-6."""
    n_jobs = int(n_jobs or getattr(config, "n_jobs", 1) or 1)
    try:
        m = mosaic.open_mosaic(config, region, band, box)
    except FileNotFoundError:
        return None
    if m.cells.n_cells == 0:
        return None
    ra = np.asarray(ra, np.float64)
    dec = np.asarray(dec, np.float64)
    t0 = time.time()
    detection = build_detection_layers(config, m, work_dir, n_jobs)
    t1 = time.time()
    reference = _cell_reference(config, m, np.arange(m.cells.n_cells, dtype=np.int64), n_jobs)
    shift = reference["shift"]
    print("  %s %s: registration east %+.2f north %+.2f arcsec from %d pairs (not applied)" % ((region, band) + shift),
          flush=True)
    src_cell = m.cells.cell_ids(*m.xy(*grid.to_map_frame(ra, dec, shift[:2])))
    cells = cell_statistics(config, m, detection, shift[:2], reference, n_jobs)
    t2 = time.time()
    src_row = _source_rows(cells, src_cell)
    cell_bin = sky_bins(cells.sky)
    peaks = _band_peaks(config, m, detection, cells, cell_bin, shift, n_jobs)
    t3 = time.time()
    pos_k, d_inj = matching_radius(config, m, detection, cells, peaks, n_jobs)
    t4 = time.time()
    r_match = MATCH_SIGMAS * pos_k / MATCH_RADIUS_SNR
    print("  recovery radius: %.1f x position scatter, the scatter x S/N = %.2f arcsec from %d bright injections (radius "
          "%.2f arcsec at S/N 5, %.2f at S/N 30)" % (MATCH_SIGMAS, pos_k, d_inj.size, r_match, MATCH_SIGMAS * pos_k / 30.0),
          flush=True)
    print("  PACS detections %d; by sky bin (faint to bright) %s; inverted-map peaks %s; %d cells; seconds: layers %.1f, "
          "cells %.1f, peaks %.1f, radius %.1f" % (peaks.fnu.size, [int((peaks.bin == j).sum()) for j in range(SKY_BINS)],
                                                   peaks.neg_count.tolist(), cells.ids.size, t1 - t0, t2 - t1, t3 - t2,
                                                   t4 - t3), flush=True)
    return BandDetection(cells=cells, cell_grid=m.cells, shift=(float(shift[0]), float(shift[1]), int(shift[2])),
                         src_row=src_row, cell_bin=cell_bin, peaks=peaks, pos_k_arcsec=pos_k, r_match=r_match,
                         kernel=_kernel_of(detection), n_src=int(ra.size))
