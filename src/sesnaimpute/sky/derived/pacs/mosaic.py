"""One co-added PACS mosaic per band per region, built tile by tile, and the measurement engine on it.

Round-1 skeleton (CONTRACT.md): signatures only. Depends on: grid, psf. Replaces: sky/derived/pacs_mosaic.py (the
co-add, zero levels, product), and the measurement half of pacs_fit.py (Tile -> `Mosaic` + `Window`: aperture photometry,
the PSF fit, bright neighbours, fitted widths, injection patches; open_map/_load_layers/mosaic_layer_plan/
mosaic_layer_strip/prepared copies -> the reader masks of `Mosaic.window`). "Tile" in the old sense (one whole prepared
map) is retired; a tile here is always the 256-pixel storage and work unit.

Store `<data_root>/sky/derived/pacs/mosaic/<Region_safe>/<band>[__slice_<ra>_<dec>_<w>x<h>]/`:
    image/ error/ coverage/ weight/   TiledLayer float32 (fill NaN, NaN, 0, 0) on the band's PixelGrid; image/index.json
                                      "meta" holds META once (region, band, shape, window_arcsec, pixel_arcsec, resampler,
                                      maps [{path, group, smooth, smooth_step, window_px, zero_level_jy_per_pixel,
                                      zero_level_mjy_per_beam}], dropped, groups, pairs, group_keys, box, floors); every
                                      layer's index.json holds the WCS header
    smooth/<input map>/               each input map's smoothed error (TiledLayer float64, input pixels)
    psf/                              the mosaic PSF (psf.py)
No mosaic.hdf5, no prepared copies (the mJy scale and the error mask are the reader's, FROZEN 12).
UNITS and READING of each layer: index.json "meta" ("units", "reading" per layer, the texts of pacs_mosaic.LAYERS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sesnaimpute.sky.derived.pacs import grid, psf
from sesnaimpute.sky.derived.pacs.grid import CellGrid, PixelGrid, TiledLayer

ROUND2 = "round 2"

#: the layers of a mosaic, in store order
LAYERS = ("image", "error", "coverage", "weight")


# ----------------------------------------------------------------------------------------------------------------------
# inputs and groups (pacs_mosaic.py:94-131)

def input_set(records: Sequence, band: str) -> Tuple[list, list]:
    """(kept, dropped): the band's maps (each file once), minus any map whose observation set is a subset of another
    kept map's; `dropped` = [(record, host record)]. Identity: `pacs_mosaic.input_set` -- same kept list, same order
    (sorted by path): META's map order, hence FROZEN 7's sub-group numbering."""
    raise NotImplementedError(ROUND2)


def group_key(config, record) -> tuple:
    """A map's PSF group key: its published templates and position angles (`pacs_mosaic.group_key`)."""
    raise NotImplementedError(ROUND2)


def mosaic_dir(config, region: str, band: str, box: Optional[Sequence[float]] = None) -> str:
    """The mosaic's directory (the path of `pacs_mosaic.product_dir`: <data_root>/sky/derived/pacs/mosaic/<safe>/<band>
    [__slice_%.3f_%.3f_%.2fx%.2f])."""
    raise NotImplementedError(ROUND2)


# ----------------------------------------------------------------------------------------------------------------------
# resampling and zero levels (pacs_mosaic.py:267-458)

def pixel_area_arcsec2(wcs) -> float:
    """The pixel area at the reference pixel, arcsec^2 (`pacs_mosaic.pixel_area_arcsec2`)."""
    raise NotImplementedError(ROUND2)


def tan_area_factor(wcs, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The TAN pixel's area at zero-origin (x, y) over the reference pixel's, (1 + rho^2)^-1.5
    (`pacs_mosaic.tan_area_factor`)."""
    raise NotImplementedError(ROUND2)


def map_bbox(input_map: "psf.InputMap", pixel_grid: PixelGrid) -> Optional[Tuple[int, int, int, int]]:
    """The output-pixel bounding box (x0, x1, y0, y1) of an input map's footprint, clipped to the grid; None if
    outside. Identity: `pacs_mosaic._bbox_in_output` (stored in psf.json; read by MosaicPSF.weights)."""
    raise NotImplementedError(ROUND2)


def resample_window(input_map: "psf.InputMap", pixel_grid: PixelGrid, y0: int, y1: int, x0: int, x1: int,
                    with_smooth: bool = True) -> Optional[Dict[str, np.ndarray]]:
    """One input map resampled onto output pixels [y0, y1) x [x0, x1) (a tile): None when no output pixel falls inside
    the map, else {valid, image, error (Jy per output pixel), coverage, factor, smooth}. Identity: `pacs_mosaic.
    resample_block` (pacs_mosaic.py:290) -- each pixel's value depends only on its own coordinates, so a tile equals
    the same pixels of a strip bit for bit."""
    raise NotImplementedError(ROUND2)


def overlap_tile(pixel_grid: PixelGrid, inputs: Sequence["psf.InputMap"], iy: int, ix: int
                 ) -> Tuple[Dict[str, int], Dict[Tuple[str, str], Tuple[int, np.ndarray]]]:
    """Pass 1 over one tile: each map's valid pixel count, and per pair of maps (n common valid pixels, float32
    differences R_i - R_j / factor). Identity: `pacs_mosaic._pair_strip` over the tile's pixels; the zero-level solve's
    medians take every common pixel of the grid (the concatenation order does not change a median)."""
    raise NotImplementedError(ROUND2)


def solve_zero_levels(paths: Sequence[str], npix: Dict[str, int], pair_n: dict, pair_d: dict) -> Dict[str, float]:
    """One constant o_i per map: weighted least squares on o_i - o_j = d_ij per connected set, weighted-mean zero.
    Identity: `pacs_mosaic.solve_zero_levels` (pacs_mosaic.py:392), global by design."""
    raise NotImplementedError(ROUND2)


def coadd_tile(pixel_grid: PixelGrid, inputs: Sequence["psf.InputMap"], offsets: Dict[str, float], iy: int, ix: int,
               layers: Dict[str, TiledLayer]) -> bool:
    """Pass 2 over one tile: the weighted co-add of every input map overlapping the tile (zero level removed), written
    into the four layers (open "r+"; the tile written only where it holds a covered pixel). Returns whether it did.
    Identity: `pacs_mosaic._strip` per pixel -- image = sum(w I)/sum(w), error = sqrt(sum(w^2 e^2))/sum(w), NaN/0 where
    sum(w) = 0; float32 values equal to the strip build's to 1e-6 relative (summation order over maps is META order in
    both, so equal in practice)."""
    raise NotImplementedError(ROUND2)


def build_mosaic(config, region: str, band: str, box: Optional[Sequence[float]] = None,
                 n_jobs: Optional[int] = None) -> "Mosaic":
    """Builds the band's mosaic: input set, PixelGrid.build (FROZEN 13), each map's smoothed error (psf.
    build_smoothed_error), pass 1 per tile (joblib over the tiles any map's bbox touches), zero levels, pass 2 per tile,
    the layers' index.json (META in image/), then the mosaic PSF (psf.build_mosaic_psf). Reads the index and the input
    FITS; writes the store above (realpath inside <data_root>/sky/derived). Returns it opened.
    Identity: every covered pixel equal to `pacs_mosaic.build_mosaic`'s within 1e-6 relative; shape, WCS header text
    and META map order identical; absent tiles are exactly the tiles with no covered pixel."""
    raise NotImplementedError(ROUND2)


# ----------------------------------------------------------------------------------------------------------------------
# the opened mosaic and its reader

@dataclass
class Window:
    """A rectangle of the mosaic as the measurements read it: rows y0 .. y0 + image.shape[0], columns x0 ..
    x0 + image.shape[1] of the grid (may reach outside the grid: those pixels are NaN). `image`, `error` float32 mJy
    per pixel, `error` NaN where unusable (FROZEN 12). A private buffer: injections paste into `image`."""
    y0: int
    x0: int
    image: np.ndarray
    error: np.ndarray


@dataclass(frozen=True)
class Floors:
    """FROZEN 12. The reader's error mask: `coverage` = MIN_COVERAGE_FRACTION x median positive coverage over
    `PixelGrid.floor_sample_rows()`; `error_fraction` = ERROR_FLOOR_FRACTION (applied to 1 / sqrt(weight))."""
    coverage: float
    error_fraction: float


@dataclass(frozen=True)
class Neighbours:
    """Blind bright sources of a mosaic: grid pixels x, y, flux (mJy), snr (`pacs_fit.Neighbours`)."""
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    snr: np.ndarray


@dataclass(frozen=True)
class PositionFit:
    """Forced PSF fits at n positions (`pacs_fit.PositionFit`): flux, gain, sigma (mJy), covered, n_nuisance."""
    flux: np.ndarray
    gain: np.ndarray
    sigma: np.ndarray
    covered: np.ndarray
    n_nuisance: np.ndarray


@dataclass(frozen=True)
class ApertureFit:
    """Forced aperture photometry at n positions (`pacs_fit.ApertureFit`): flux (mJy, total), sky (mJy/pixel),
    annulus_sigma (mJy), covered (>= 90 % aperture weight and >= half the annulus valid)."""
    flux: np.ndarray
    sky: np.ndarray
    annulus_sigma: np.ndarray
    covered: np.ndarray


class Mosaic:
    """An opened mosaic: `dir`, `grid` (PixelGrid), `cells` (CellGrid), `layers` {name: TiledLayer, mode "r"}, `meta`,
    `psf` (psf.MosaicPSF), `floors` (Floors), `record` (a MapRecord-like description: map_id, band, obsids, shape,
    wcs_header, pixel_scale_arcsec), `band`, `pixel_arcsec`, `beam_arcsec`, `aperture_fraction`, `enclosed_fraction`,
    `half` (fit box half-width, px), `table_half`. Replaces pacs_mosaic.Mosaic + pacs_fit.Tile/OpenMap/open_map/
    open_prepared/cached_tile/mosaic_record. Picklable by (config, region, band, box)."""

    def __init__(self, config, region: str, band: str, box: Optional[Sequence[float]] = None,
                 psf_tables: bool = True) -> None:
        """Opens the store (layers read-only, PSF tables memory-mapped when `psf_tables`) and computes `floors` once
        (or reads them from META). Reads: the mosaic store, the region's PACS index (obsids)."""
        raise NotImplementedError(ROUND2)

    # the reader
    def window(self, y0: int, y1: int, x0: int, x1: int) -> Window:
        """The pixels [y0, y1) x [x0, x1) as the measurements read them: image x 1e3 (mJy, float32), error x 1e3 with
        NaN where coverage <= floors.coverage, error <= 0, error <= floors.error_fraction x 1e3 / sqrt(weight) or the
        image is not finite (FROZEN 12). Identity: equals the old prepared layers (`mosaic_layer_strip`) bit for bit on
        those pixels."""
        raise NotImplementedError(ROUND2)

    def tile_window(self, iy: int, ix: int, halo: int) -> Window:
        """`window` of tile (iy, ix) plus `halo` pixels on every side."""
        raise NotImplementedError(ROUND2)

    def present_tiles(self) -> List[Tuple[int, int]]:
        """The tiles holding any covered pixel (the image layer's present list)."""
        raise NotImplementedError(ROUND2)

    # geometry
    def xy(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sky -> grid pixel (`grid.sky_to_pixel`)."""
        raise NotImplementedError(ROUND2)

    def beam_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The local beam FWHM, arcsec (MosaicPSF.beam_at)."""
        raise NotImplementedError(ROUND2)

    def beam_px_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """`beam_at` in pixels."""
        raise NotImplementedError(ROUND2)

    def injection_half_width(self) -> int:
        """Half-width (px) of an injected patch: int(min(table_half, half + ceil(beam_px))) (pacs_fit.py:1119); also the
        detection kernel's half-width h."""
        raise NotImplementedError(ROUND2)

    # measurements (each on a Window; positions in grid pixels)
    def aperture(self, x: np.ndarray, y: np.ndarray, window: Optional[Window] = None, chunk: int = 2048) -> ApertureFit:
        """FROZEN 8. Forced eHOPS aperture photometry: `_aperture_core` (5 x 5 sub-pixels, mmm-clipped annulus sky),
        then two passes of `_aperture_core_wing` with the source's own outer light (scaled to the previous pass's flux)
        removed from the annulus; flux = sum / EEF at the position (MosaicPSF mixture). `window`: the pixels to measure
        on (default: read the positions' bounding box plus the aperture reach). Identity: `pacs_fit.Tile.aperture`
        (pacs_fit.py:943) bit-identical on the same pixels; the numba cores move here unchanged."""
        raise NotImplementedError(ROUND2)

    def neighbours(self) -> Neighbours:
        """The mosaic's blind bright sources (PSF-fit S/N >= NEIGHBOUR_SNR), found per tile with a halo of the fit
        reach. Identity: `pacs_fit.Tile._find_neighbours` (pacs_fit.py:820): same list, sorted as there. Stored in
        META["neighbours"] after the first computation (replaces prepared/tile.pkl)."""
        raise NotImplementedError(ROUND2)

    def psf_fit(self, x: np.ndarray, y: np.ndarray, window: Optional[Window] = None,
                use_nuisance: bool = True) -> PositionFit:
        """The weighted linear PSF fit with bright neighbours as nuisance columns (`Tile._fit`/`measure`)."""
        raise NotImplementedError(ROUND2)

    def fitted_fwhm(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Best-fit total FWHM (arcsec) of a detection (`Tile.fitted_fwhm`); EXTENDED where > 1.5 x beam."""
        raise NotImplementedError(ROUND2)

    def near_bright(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Within one beam of a bright-neighbour peak (`Tile.near_bright`)."""
        raise NotImplementedError(ROUND2)

    def pixel_sigma(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The masked error (mJy/px) at the nearest pixel, NaN off the data (`Tile.pixel_sigma`)."""
        raise NotImplementedError(ROUND2)

    def template_patches(self, x: np.ndarray, y: np.ndarray, flux: np.ndarray,
                         mixture: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cx, cy, patches): the injection template (local MosaicPSF mixture, mean over components) times `flux` on
        (2h+1)^2 patches, lower-left pixels (cx, cy) (`Tile._template_patches`, pacs_fit.py:1138)."""
        raise NotImplementedError(ROUND2)


def paste(window: Window, cx: np.ndarray, cy: np.ndarray, patches: np.ndarray) -> None:
    """Adds patches into `window.image` at grid lower-left pixels (cx, cy), clipped to the window
    (`pacs_fit._paste`, pacs_fit.py:1198; float32 addition order as there)."""
    raise NotImplementedError(ROUND2)


def open_mosaic(config, region: str, band: str, box: Optional[Sequence[float]] = None) -> Mosaic:
    """The built mosaic opened (`Mosaic(config, region, band, box)`)."""
    raise NotImplementedError(ROUND2)
