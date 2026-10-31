"""The PACS build's grids, its tile store and its one set of coordinate conversions.

    PixelGrid    the output pixel grid of one band's mosaic: TAN, north up, PIXEL_ARCSEC per band, origin at the box
                 corner (the old `pacs_mosaic.make_grid`)
    CellGrid     the 30-arcsec cell grid over a pixel grid: y-major cell ids, the seeds, the blocks, the peak keys (the
                 old `pacs_fit.MapGrid` and the id arithmetic of `pacs_completeness`)
    TiledLayer   one 2-D layer stored as 256 x 256 tiles, only the tiles holding a non-fill pixel on disk
    conversions  out-pixel <-> sky <-> in-pixel, arcsec <-> pixels, the registration frame, the PSF node lookup, cell id
                 <-> (line, column): each written once, here

Vocabulary (one meaning per word, CONTRACT.md): grid = the output pixel grid; cell = the 30-arcsec cell; tile = the
256 x 256-pixel storage and work unit; block = the cell-line block of the injections (INJECTION_BLOCK_LINES lines of
cells) and the square cell block of the cell statistics (BLOCK_CELLS x BLOCK_CELLS cells); node = a PSF node; plane = an
S/N plane; piece = one work item. The tile index is storage only: cell ids, the WCS origin and the grid shape do not
depend on it.

FROZEN -- the identity items. Every product number of the old build depends on these; the consolidated build keeps them
exactly (FIX_SPEC.md sec 11, readings G4 sec 6 and C sec 5). Each has a named constant below with its one-line reason:

  1  CELL_ID            cell id = iy * nx + ix over the whole grid, nx = ceil(grid nx / step_x)
  2  SEEDS              a cell's draws: counter = id << SEED_COUNTER_SHIFT + draw, SplitMix64-mixed with the seed
  3  BLOCKS             injection block = id // nx // INJECTION_BLOCK_LINES; cell-statistics block =
                        (iy // BLOCK_CELLS) * (nx // BLOCK_CELLS + 1) + ix // BLOCK_CELLS
  4  PEAK_KEY           a row's key = map index << PEAK_KEY_SHIFT + cell id
  5  PSF_NODE_COUNT     n_nodes = max(PSF_SPATIAL_NODES, ceil(max(nx, ny) * pix / 3600 / MOSAIC_PSF_NODE_SPACING_DEG) + 1),
                        made odd; the centre node (n_nodes**2 // 2) carries the EEF normalisation
  6  SMOOTH_STEP        an input map's smoothed error is sampled every window_px // 2 input pixels (window_px =
                        max(3, round(SMOOTH_WINDOW_ARCSEC / scale) | 1)) and read by bilinear interpolation
  7  SUBGROUP_KEY       a PSF sub-group key = (group, round(pixel scale, SUBGROUP_SCALE_DECIMALS),
                        round(angle * 4) / 4 % 90), keys in first-appearance order over the maps in META order
  8  APERTURE_RECIPE    APERTURE_SUBPIXELS x APERTURE_SUBPIXELS sub-pixels, then OUTER_LIGHT_PASSES passes with the
                        source's own outer light removed from the annulus
  9  REGISTRATION       SESNA positions are moved into a map's frame (`to_map_frame`) before any cell lookup
  10 REGION_PAD         REGION_PAD_ARCSEC = 412: one nside-512 pixel around the region's source box
  11 DETECTION_KERNEL   the detection kernel is the PSF at grid pixel (ny // 2, nx // 2)
  12 READER_FLOORS      a pixel's error is unusable at coverage <= MIN_COVERAGE_FRACTION x median positive coverage
                        (sampled over whole rows at stride max(1, ny * nx // FLOOR_SAMPLE_PIXELS)), error <= 0, error <=
                        ERROR_FLOOR_FRACTION / sqrt(weight), or a non-finite image
  13 WCS_ORIGIN         TAN at the footprint centre (GRID_RECENTRE_ITERATIONS re-centrings), crpix = (0.5 - xmin,
                        0.5 - ymin), nx = ceil(xmax - xmin) + 1 (a sky box: crpix = ((n + 1) / 2) about the box centre)
  14 PSF_NODE_LAYOUT    node positions linspace(-0.5, n - 0.5, n_nodes) per axis over the grid; a position's node
                        coordinate (s + 0.5) / n * (n_nodes - 1)

Nothing in this module reads a product of the build except through TiledLayer; it imports only numpy, astropy, healpy
and sesnaimpute.definitions.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from sesnaimpute.definitions import (PACS_BLOCK_CELLS, PACS_CELL_ARCSEC, PACS_PIXEL_ARCSEC)

# ----------------------------------------------------------------------------------------------------------------------
# FROZEN identity constants (see the module docstring for the item each serves)

#: 1 CELL_ID -- cell side, arcsec: the cell grid's step (definitions.PACS_CELL_ARCSEC, SESNA's own 30 arcsec)
CELL_ARCSEC = PACS_CELL_ARCSEC
#: 2 SEEDS -- the draw counter of a cell is id << 32 + draw: the injection positions depend on the cell id only
SEED_COUNTER_SHIFT = 32
#: 2 SEEDS -- SplitMix64 constants of the counter mixing (the draws are bit-reproducible only with these)
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
#: 3 BLOCKS -- lines of cells per injection block: the block id seeds the injection rng (`default_rng([key, plane,
#: 2 * seed + half, block])`), so the block partition is fixed by the grid, never by the worker count
INJECTION_BLOCK_LINES = 16
#: 3 BLOCKS -- side, in cells, of the cell-statistics block (definitions.PACS_BLOCK_CELLS); block ids use the stride
#: nx // BLOCK_CELLS + 1, which sets the piece order of the cell stage
BLOCK_CELLS = PACS_BLOCK_CELLS
#: 4 PEAK_KEY -- a completeness row is keyed map << 40 + cell id (rows sorted by key; cell ids must stay below 2**40)
PEAK_KEY_SHIFT = 40
#: 5 PSF_NODE_COUNT -- the least PSF node count per axis (3 x 3 nodes at Orion A pass the normalisation check)
PSF_SPATIAL_NODES = 3
#: 5 PSF_NODE_COUNT -- the most distance between PSF nodes, deg (Aquila at 6.4 deg read +0.30 %: the enclosed
#: fraction is bilinear between nodes in a function curved on that scale)
MOSAIC_PSF_NODE_SPACING_DEG = 4.25
#: 6 SMOOTH_STEP -- side of the median window of an input map's error, arcsec (sets every mosaic weight)
SMOOTH_WINDOW_ARCSEC = 60.0
#: 7 SUBGROUP_KEY -- a sub-group's scan angle is rounded to this many degrees (angle % 90)
SUBGROUP_ANGLE_STEP_DEG = 0.25
#: 7 SUBGROUP_KEY -- a sub-group's pixel scale is rounded to this many decimals (arcsec)
SUBGROUP_SCALE_DECIMALS = 3
#: 8 APERTURE_RECIPE -- sub-pixels per pixel axis of the aperture weights (the eHOPS recipe as built)
APERTURE_SUBPIXELS = 5
#: 8 APERTURE_RECIPE -- passes of the annulus sky with the source's own outer light removed
OUTER_LIGHT_PASSES = 2
#: 10 REGION_PAD -- one nside-512 pixel (412 arcsec) around the region's source bounding box. Note: on main 0683690 no
#: code applies it (G1 sec 3: `pacs_completeness.REGION_PAD_ARCSEC` was referenced by nothing; the cell stage enumerates
#: every cell of the grid, C sec 5 item 8); defined once here, unused by the identity path
REGION_PAD_ARCSEC = 412.0
#: 12 READER_FLOORS -- coverage floor as a fraction of the median positive coverage (below it the error layer collapses
#: to meaningless tiny values: Musca, 0.2 % of pixels at coverage < 0.05 x median carry errors of 1e-7 mJy)
MIN_COVERAGE_FRACTION = 0.05
#: 12 READER_FLOORS -- an error at or below this fraction of the error its own weight expects (1 / sqrt(weight)) is
#: unusable (a few collapsed values would otherwise carry all the weight)
ERROR_FLOOR_FRACTION = 0.25
#: 12 READER_FLOORS -- the coverage floor's median is taken from about this many pixels (whole rows at a stride that
#: depends on the grid shape)
FLOOR_SAMPLE_PIXELS = 8_000_000
#: 13 WCS_ORIGIN -- re-centrings of the tangent point on the footprint before the final grid
GRID_RECENTRE_ITERATIONS = 3

#: the frozen items by number, for CONTRACT.md and the round-3 audit: number -> (name, one-line reason)
FROZEN = {
    1: ("CELL_ID", "cell id = iy * nx + ix seeds the draws and keys every completeness row"),
    2: ("SEEDS", "counter = id << 32 + draw, SplitMix64: the injection positions are a function of the cell id"),
    3: ("BLOCKS", "injection block id = id // nx // 16 enters the rng; cell-statistics block stride nx // 16 + 1 orders pieces"),
    4: ("PEAK_KEY", "row key = map << 40 + cell id orders the CURVE rows and resolves SOURCE_ROW"),
    5: ("PSF_NODE_COUNT", "odd n_nodes from the grid extent; the centre node normalises the EEF"),
    6: ("SMOOTH_STEP", "smoothed error sampled every window_px // 2 input px, bilinear: sets every mosaic weight"),
    7: ("SUBGROUP_KEY", "(group, scale, angle to 0.25 deg) in first-appearance order: the sub<k> table numbering"),
    8: ("APERTURE_RECIPE", "5 x 5 sub-pixels and two outer-light passes: every aperture flux"),
    9: ("REGISTRATION", "SESNA positions shifted into the map frame before the cell lookup: SOURCE_ROW"),
    10: ("REGION_PAD", "412 arcsec = one nside-512 pixel around the source box (unused on main)"),
    11: ("DETECTION_KERNEL", "the kernel is the PSF at (ny // 2, nx // 2): every detection and S/N"),
    12: ("READER_FLOORS", "coverage, error and image floors of the error mask: which pixels count"),
    13: ("WCS_ORIGIN", "crpix = (0.5 - xmin, 0.5 - ymin) after three re-centrings: every pixel coordinate"),
    14: ("PSF_NODE_LAYOUT", "nodes at linspace(-0.5, n - 0.5) and lookup (s + 0.5) / n * (n_nodes - 1): every PSF value"),
}

# ----------------------------------------------------------------------------------------------------------------------
# storage constants (not identity: the tile index is storage only)

#: side of a storage and work tile, pixels
TILE = 256
#: the value an absent tile reads as, per layer name (image, error: no data; coverage, weight: zero; detection masks)
LAYER_FILL = {"image": np.nan, "error": np.nan, "coverage": 0.0, "weight": 0.0,
              "m": np.nan, "ratio": np.nan, "norm": np.nan, "valid": False, "pix_valid": False}
#: the HEALPix resolution and scheme of the tile index (granules/build.py: nested, galactic)
INDEX_NSIDE = 512

ArrayLike = Union[np.ndarray, float, Sequence[float]]


# ----------------------------------------------------------------------------------------------------------------------
# conversions (each written once)

def celestial_wcs(header_text: str) -> WCS:
    """The celestial WCS of a FITS header string (a map's `wcs_header`, a mosaic's WCS_HEADER)."""
    return WCS(fits.Header.fromstring(header_text)).celestial


def pixel_scale_arcsec(wcs: WCS) -> float:
    """The mean pixel side of `wcs`, arcsec (as `pacs_fit.Tile` and `build_mosaic_psf` take it)."""
    return float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0


def pixel_to_sky(wcs: WCS, x: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-origin pixel (x, y) -> (ra, dec) deg, one vectorised `all_pix2world` call."""
    ra, dec = wcs.all_pix2world(np.asarray(x, float), np.asarray(y, float), 0)
    return np.asarray(ra), np.asarray(dec)


def sky_to_pixel(wcs: WCS, ra_deg: ArrayLike, dec_deg: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """(ra, dec) deg -> zero-origin pixel (x, y), one vectorised `all_world2pix` call."""
    x, y = wcs.all_world2pix(np.asarray(ra_deg, float), np.asarray(dec_deg, float), 0)
    return np.asarray(x), np.asarray(y)


def out_to_in(out_wcs: WCS, in_wcs: WCS, x: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Output-grid pixel (x, y) -> an input map's pixel (px, py), through the sky (the resampling's and the PSF
    weights' path: all_pix2world then all_world2pix)."""
    return sky_to_pixel(in_wcs, *pixel_to_sky(out_wcs, x, y))


def in_to_out(in_wcs: WCS, out_wcs: WCS, px: ArrayLike, py: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """An input map's pixel (px, py) -> output-grid pixel (x, y), through the sky."""
    return sky_to_pixel(out_wcs, *pixel_to_sky(in_wcs, px, py))


def arcsec_to_px(arcsec: ArrayLike, pixel_arcsec: float) -> ArrayLike:
    """A length in arcsec as pixels of side `pixel_arcsec` (radii of apertures, beams, separations)."""
    return np.asarray(arcsec, float) / pixel_arcsec if np.ndim(arcsec) else float(arcsec) / pixel_arcsec


def px_to_arcsec(px: ArrayLike, pixel_arcsec: float) -> ArrayLike:
    """A length in pixels of side `pixel_arcsec` as arcsec."""
    return np.asarray(px, float) * pixel_arcsec if np.ndim(px) else float(px) * pixel_arcsec


def to_map_frame(ra_deg: ArrayLike, dec_deg: ArrayLike, shift: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """FROZEN 9. SESNA positions in a map's own astrometric frame: plus the map's registration shift (`shift` =
    (east, north) arcsec, PACS minus SESNA). Bit-identical to `pacs_fit.to_map_frame`."""
    east, north = shift
    dec_deg = np.asarray(dec_deg, float)
    return (np.asarray(ra_deg, float) + east / 3600.0 / np.cos(np.radians(dec_deg)),
            dec_deg + north / 3600.0)


def from_map_frame(ra_deg: ArrayLike, dec_deg: ArrayLike, shift: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """The inverse of `to_map_frame`: a position in the map's frame back to the SESNA frame (`pacs_fit.from_map_frame`)."""
    east, north = shift
    dec_deg = np.asarray(dec_deg, float) - north / 3600.0
    return np.asarray(ra_deg, float) - east / 3600.0 / np.cos(np.radians(dec_deg)), dec_deg


def psf_node_count(nx: int, ny: int, pixel_arcsec: float) -> int:
    """FROZEN 5. The PSF nodes per axis of a grid of nx x ny pixels: at least PSF_SPATIAL_NODES, at most
    MOSAIC_PSF_NODE_SPACING_DEG apart, made odd (the centre node normalises). `pacs_fit.py:1640-1641`."""
    n = max(PSF_SPATIAL_NODES, int(np.ceil(max(nx, ny) * pixel_arcsec / 3600.0 / MOSAIC_PSF_NODE_SPACING_DEG)) + 1)
    return n + 1 - n % 2


def psf_node_positions(size: int, n_nodes: int) -> np.ndarray:
    """FROZEN 14. The node positions along one grid axis of `size` pixels: linspace(-0.5, size - 0.5, n_nodes)."""
    return np.linspace(-.5, size - .5, n_nodes)


def psf_node_coord(s: ArrayLike, size: int, n_nodes: int, clip: bool = False) -> ArrayLike:
    """FROZEN 14. A grid position's coordinate in node units along one axis: (s + 0.5) / size * (n_nodes - 1);
    `clip` to [0, n_nodes - 1] (as `MosaicPSF._node_values`; `_cached_responses` raises outside instead)."""
    g = (np.asarray(s, float) + .5) / size * (n_nodes - 1)
    return np.clip(g, 0.0, n_nodes - 1.0) if clip else g


def psf_node_cell(g: ArrayLike, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """The lower node index and the bilinear fraction of node coordinates `g` (>= 0): i = min(int(g), n_nodes - 2),
    f = g - i (`_cached_responses` and `_node_values` both)."""
    g = np.asarray(g, float)
    i = np.minimum(g.astype(np.int64), n_nodes - 2)
    return i, g - i


def smooth_window_px(pixel_scale_arcsec_: float) -> int:
    """FROZEN 6. The odd side, in input pixels, of an input map's error median window: max(3, round(60 / scale) | 1)."""
    return max(3, int(round(SMOOTH_WINDOW_ARCSEC / pixel_scale_arcsec_)) | 1)


def smooth_step_px(pixel_scale_arcsec_: float) -> int:
    """FROZEN 6. The smoothed error's sample spacing, input pixels: max(1, window_px // 2)."""
    return max(1, smooth_window_px(pixel_scale_arcsec_) // 2)


def subgroup_key(group: int, pixel_scale_arcsec_: float, angle_deg: float) -> Tuple[int, float, float]:
    """FROZEN 7. The PSF sub-group key of an input map: (group, scale to 3 decimals, angle % 90 to 0.25 deg)
    (`pacs_fit.build_mosaic_psf`, `pacs_fit.py:1604`; `angle_deg` already taken % 90 there)."""
    return (group, round(pixel_scale_arcsec_, SUBGROUP_SCALE_DECIMALS), round(angle_deg * 4) / 4 % 90.0)


def cell_line_column(ids: ArrayLike, nx: int) -> Tuple[np.ndarray, np.ndarray]:
    """FROZEN 1. (iy, ix) of cell ids on a cell grid nx cells wide (`np.divmod(ids, nx)`)."""
    return np.divmod(np.asarray(ids, np.int64), nx)


def mix64(counter: np.ndarray) -> np.ndarray:
    """FROZEN 2. SplitMix64 mixing with intentional uint64 wraparound (`pacs_fit._mix64`)."""
    value = counter + np.uint64(SPLITMIX_GAMMA)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(SPLITMIX_M1)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(SPLITMIX_M2)
    value ^= value >> np.uint64(31)
    return value


def uniform(counter: np.ndarray) -> np.ndarray:
    """FROZEN 2. Uniform float64 values from the upper 53 mixed counter bits (`pacs_fit._uniform`)."""
    return (mix64(counter) >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)


def hpx512_galactic(ra_deg: ArrayLike, dec_deg: ArrayLike) -> np.ndarray:
    """The nside-512 nested galactic HEALPix pixel of ICRS positions (the survey's pixelisation, granules/build.py)."""
    from astropy.coordinates import SkyCoord
    from scipy.special import erfc  # noqa: F401 -- import before healpy for native-library initialisation order
    import healpy as hp
    g = SkyCoord(np.atleast_1d(np.asarray(ra_deg, float)), np.atleast_1d(np.asarray(dec_deg, float)),
                 unit="deg", frame="icrs").galactic
    return hp.ang2pix(INDEX_NSIDE, g.l.deg, g.b.deg, nest=True, lonlat=True).astype(np.int64)


def _tan(ra: float, dec: float, pix_arcsec: float, crpix: Tuple[float, float] = (1.0, 1.0)) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-pix_arcsec / 3600.0, pix_arcsec / 3600.0]
    w.wcs.crpix = list(crpix)
    w.wcs.cunit = ["deg", "deg"]
    return w


# ----------------------------------------------------------------------------------------------------------------------
# the pixel grid

@dataclass(frozen=True)
class PixelGrid:
    """The output pixel grid of one band's mosaic: `wcs` (celestial TAN, north up), `shape` (ny, nx), `pixel_arcsec`,
    `box` (ra, dec, width_deg, height_deg) or None (the union of the input footprints).

    FROZEN 13: built exactly as `pacs_mosaic.make_grid` (same header text, same shape)."""
    wcs: WCS = field(compare=False)
    shape: Tuple[int, int]
    pixel_arcsec: float
    box: Optional[Tuple[float, float, float, float]] = None

    @classmethod
    def build(cls, records: Sequence, band: str, box: Optional[Sequence[float]] = None) -> "PixelGrid":
        """The grid spanning the union of `records`' corner polygons (`record.polygon`, the maps' own pixel edges), or
        the sky `box` about (ra, dec) (width along RA on the sky). Pixel = definitions.PACS_PIXEL_ARCSEC[band]."""
        pix = PACS_PIXEL_ARCSEC[band]
        if box is not None:
            ra0, dec0, wd, hd = box
            nx, ny = int(np.ceil(wd * 3600.0 / pix)), int(np.ceil(hd * 3600.0 / pix))
            return cls(_tan(ra0, dec0, pix, crpix=((nx + 1) / 2.0, (ny + 1) / 2.0)), (ny, nx), pix, tuple(box))
        pts = np.concatenate([np.array(r.polygon, dtype=float) for r in records])
        ra0 = float(np.degrees(np.angle(np.exp(1j * np.radians(pts[:, 0])).mean())) % 360.0)
        dec0 = float(pts[:, 1].mean())
        for _ in range(GRID_RECENTRE_ITERATIONS):          # re-centre the tangent point on the footprint
            w = _tan(ra0, dec0, pix)
            x, y = w.all_world2pix(pts[:, 0], pts[:, 1], 0)
            cx, cy = (x.min() + x.max()) / 2.0, (y.min() + y.max()) / 2.0
            ra0, dec0 = (float(v) for v in w.all_pix2world(cx, cy, 0))
        w = _tan(ra0, dec0, pix)
        x, y = w.all_world2pix(pts[:, 0], pts[:, 1], 0)
        nx, ny = int(np.ceil(x.max() - x.min())) + 1, int(np.ceil(y.max() - y.min())) + 1
        return cls(_tan(ra0, dec0, pix, crpix=(0.5 - x.min(), 0.5 - y.min())), (ny, nx), pix, None)

    @classmethod
    def from_header(cls, header_text: str, shape: Sequence[int], pixel_arcsec: Optional[float] = None,
                    box: Optional[Sequence[float]] = None) -> "PixelGrid":
        """The grid of a stored WCS header string and shape (pixel side from the WCS when not given)."""
        wcs = celestial_wcs(header_text)
        pix = pixel_scale_arcsec(wcs) if pixel_arcsec is None else float(pixel_arcsec)
        return cls(wcs, (int(shape[0]), int(shape[1])), pix, None if box is None else tuple(box))

    @classmethod
    def from_meta(cls, meta: dict, header_text: str) -> "PixelGrid":
        """The grid of a mosaic's META (keys `shape`, `pixel_arcsec`, `box`, as `pacs_mosaic.build_mosaic` writes) and
        its WCS_HEADER string."""
        return cls.from_header(header_text, meta["shape"], meta["pixel_arcsec"], meta.get("box"))

    def to_meta(self) -> dict:
        """The META keys of the grid as `pacs_mosaic.build_mosaic` writes them: {shape: [ny, nx], pixel_arcsec, box}."""
        return dict(shape=[int(self.shape[0]), int(self.shape[1])], pixel_arcsec=self.pixel_arcsec,
                    box=None if self.box is None else list(self.box))

    def header_text(self) -> str:
        """The WCS_HEADER string as `pacs_mosaic.build_mosaic` writes it (relax=True, NAXIS1/2 set, no separators)."""
        header = self.wcs.to_header(relax=True)
        header["NAXIS1"], header["NAXIS2"] = int(self.shape[1]), int(self.shape[0])
        return header.tostring(sep="")

    @property
    def ny(self) -> int:
        return int(self.shape[0])

    @property
    def nx(self) -> int:
        return int(self.shape[1])

    def cell_grid(self) -> "CellGrid":
        """The 30-arcsec cell grid of this pixel grid."""
        return CellGrid.from_wcs(self.wcs, self.shape)

    def tile_shape(self, tile: int = TILE) -> Tuple[int, int]:
        """(tile rows, tile columns) covering the grid."""
        return -(-self.ny // tile), -(-self.nx // tile)

    def tile_bounds(self, iy: int, ix: int, tile: int = TILE) -> Tuple[int, int, int, int]:
        """(y0, y1, x0, x1) of tile (iy, ix) clipped to the grid, half-open."""
        return iy * tile, min((iy + 1) * tile, self.ny), ix * tile, min((ix + 1) * tile, self.nx)

    def psf_node_count(self) -> int:
        """FROZEN 5: the grid's PSF node count per axis."""
        return psf_node_count(self.nx, self.ny, self.pixel_arcsec)

    def psf_nodes(self) -> list:
        """FROZEN 14: [(sx, sy)] of the PSF nodes, y-major (the order `build_mosaic_psf` writes the tables in)."""
        n = self.psf_node_count()
        return [(sx, sy) for sy in psf_node_positions(self.ny, n) for sx in psf_node_positions(self.nx, n)]

    def detection_kernel_centre(self) -> Tuple[int, int]:
        """FROZEN 11: (cy, cx) = (ny // 2, nx // 2), where the detection kernel is evaluated."""
        return self.ny // 2, self.nx // 2

    def floor_sample_rows(self) -> np.ndarray:
        """FROZEN 12: the rows the coverage floor's median is sampled on: arange(0, ny, max(1, ny * nx // 8e6))."""
        return np.arange(0, self.ny, max(1, int(self.ny * self.nx // FLOOR_SAMPLE_PIXELS)))


# ----------------------------------------------------------------------------------------------------------------------
# the cell grid

@dataclass(frozen=True)
class CellGrid:
    """The 30-arcsec cell grid of one pixel grid (exactly `pacs_fit.MapGrid`): y-major cell ids, cells aligned with the
    pixel axes, edge cells clipped to the image (pixel centres are integers, edges at -0.5, n - 0.5). `nx`, `ny` are
    CELL counts; `shape` is the pixel grid's (ny, nx); `step_x`, `step_y` the cell side in pixels (30 / scale).

    FROZEN 1-4: ids, seeds, blocks and keys are functions of the cell id over the whole grid, never of the tiles."""
    wcs: object = field(compare=False)
    shape: Tuple[int, int]
    step_x: float
    step_y: float
    nx: int
    ny: int

    @classmethod
    def from_wcs(cls, wcs: WCS, shape: Sequence[int]) -> "CellGrid":
        """The cell grid of a pixel grid; rejects a skewed or distorted WCS (`MapGrid.from_wcs`)."""
        wcs = wcs.celestial
        scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=np.float64) * 3600.0
        if not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError("invalid map pixel scale")
        matrix = wcs.pixel_scale_matrix
        cosine = np.dot(matrix[:, 0], matrix[:, 1]) / np.prod(scales / 3600.0)
        if abs(cosine) > 1e-6 or wcs.has_distortion:
            raise ValueError("skewed or distorted WCS needs explicit cell geometry")
        sx, sy = CELL_ARCSEC / scales
        shape = tuple(int(v) for v in shape)
        return cls(wcs, shape, float(sx), float(sy), int(np.ceil(shape[1] / sx)), int(np.ceil(shape[0] / sy)))

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell_ids(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """FROZEN 1. int64 cell ids of pixel positions, -1 outside the image."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        inside = (np.isfinite(x) & np.isfinite(y) & (x >= -0.5) & (x < self.shape[1] - 0.5)
                  & (y >= -0.5) & (y < self.shape[0] - 0.5))
        out = np.full(x.shape, -1, dtype=np.int64)
        ix = np.floor((x[inside] + 0.5) / self.step_x).astype(np.int64)
        iy = np.floor((y[inside] + 0.5) / self.step_y).astype(np.int64)
        out[inside] = iy * self.nx + ix
        return out

    def line_column(self, ids: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """(iy, ix) of cell ids."""
        return cell_line_column(ids, self.nx)

    def bounds(self, ids: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """x0, x1, y0, y1 pixel edges of the cells, clipped to the image."""
        ids = np.asarray(ids)
        x0 = (ids % self.nx) * self.step_x - 0.5
        y0 = (ids // self.nx) * self.step_y - 0.5
        return (x0, np.minimum(x0 + self.step_x, self.shape[1] - 0.5),
                y0, np.minimum(y0 + self.step_y, self.shape[0] - 0.5))

    def centres_icrs(self, ids: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """ra, dec (deg, ICRS) of the cells' clipped centres."""
        x0, x1, y0, y1 = self.bounds(ids)
        sky = self.wcs.pixel_to_world((x0 + x1) / 2, (y0 + y1) / 2).icrs
        return sky.ra.deg, sky.dec.deg

    def centres_arcsec(self, ids: ArrayLike) -> np.ndarray:
        """(n, 2) arcsec position of each cell's clipped centre in the grid's own pixel frame (the pooling frame,
        `pacs_completeness.cell_centres_arcsec`)."""
        x0, x1, y0, y1 = self.bounds(ids)
        return np.c_[(x0 + x1) / 2 * (CELL_ARCSEC / self.step_x), (y0 + y1) / 2 * (CELL_ARCSEC / self.step_y)]

    def sample_pixels(self, ids: ArrayLike, n_positions: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """FROZEN 2. (n_ids, n) uniform x and y in each cell, reproducible per (seed, cell, draw) whatever the batching
        of the cells: counter = id << 32 + draw (2 * n draws, x even, y odd), key = mix64(seed)."""
        ids = np.asarray(ids)
        x0, x1, y0, y1 = self.bounds(ids)
        counter = ids.astype(np.uint64)[:, None] << np.uint64(SEED_COUNTER_SHIFT)
        draws = np.arange(2 * n_positions, dtype=np.uint64)[None, :]
        key = mix64(np.array([seed], dtype=np.uint64))[0]
        u = uniform((counter + draws) ^ key)
        return (x0[:, None] + (x1 - x0)[:, None] * u[:, 0::2],
                y0[:, None] + (y1 - y0)[:, None] * u[:, 1::2])

    def injection_block(self, ids: ArrayLike) -> np.ndarray:
        """FROZEN 3. The injection block of each cell: id // nx // INJECTION_BLOCK_LINES."""
        return np.asarray(ids, np.int64) // self.nx // INJECTION_BLOCK_LINES

    def stat_block(self, ids: ArrayLike) -> np.ndarray:
        """FROZEN 3. The cell-statistics block of each cell: (iy // 16) * (nx // 16 + 1) + ix // 16."""
        cy, cx = self.line_column(ids)
        return (cy // BLOCK_CELLS) * (self.nx // BLOCK_CELLS + 1) + cx // BLOCK_CELLS

    @staticmethod
    def peak_key(map_index: ArrayLike, ids: ArrayLike) -> np.ndarray:
        """FROZEN 4. The row key map << 40 + cell id (int64)."""
        return np.asarray(map_index).astype(np.int64) * (1 << PEAK_KEY_SHIFT) + np.asarray(ids, np.int64)

    def cells_in_window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Ascending ids of the cells that hold any pixel of the half-open pixel window [y0, y1) x [x0, x1) (a tile's
        cells; a cell straddling two tiles belongs to both)."""
        if y1 <= y0 or x1 <= x0:
            return np.zeros(0, np.int64)
        iy0, iy1 = int(np.floor((y0 + 0.5) / self.step_y)), int(np.floor((y1 - 1 + 0.5) / self.step_y))
        ix0, ix1 = int(np.floor((x0 + 0.5) / self.step_x)), int(np.floor((x1 - 1 + 0.5) / self.step_x))
        iy = np.arange(max(iy0, 0), min(iy1, self.ny - 1) + 1, dtype=np.int64)
        ix = np.arange(max(ix0, 0), min(ix1, self.nx - 1) + 1, dtype=np.int64)
        return (iy[:, None] * self.nx + ix[None, :]).ravel()


# ----------------------------------------------------------------------------------------------------------------------
# the tile store

def _fill_bits(dtype: np.dtype, fill) -> np.ndarray:
    """The fill value of `dtype` as its unsigned-integer bit pattern (NaN fills compare by bits, so a stored tile is
    byte-exact and -0.0 is not taken for 0)."""
    return np.array(fill, dtype=dtype).view(_bits_dtype(dtype))


def _bits_dtype(dtype: np.dtype) -> np.dtype:
    return np.dtype("u%d" % np.dtype(dtype).itemsize) if np.dtype(dtype).kind != "b" else np.dtype("u1")


def _fill_json(fill):
    if isinstance(fill, (bool, np.bool_)):
        return bool(fill)
    f = float(fill)
    return "nan" if np.isnan(f) else f


def _fill_from_json(v, dtype):
    if np.dtype(dtype).kind == "b":
        return bool(v)
    return np.nan if v == "nan" else float(v)


def _norm_axis(k, n: int):
    """An axis key -> ('slice', start, stop, step) or ('int', i) or ('array', ndarray)."""
    if isinstance(k, (int, np.integer)):
        i = int(k)
        if i < -n or i >= n:
            raise IndexError("index %d out of bounds for axis of size %d" % (i, n))
        return ("int", i % n)
    if isinstance(k, slice):
        return ("slice",) + k.indices(n)
    a = np.asarray(k)
    if a.dtype.kind in "iu":
        return ("array", a)
    raise TypeError("TiledLayer index must be ints, slices or integer arrays, not %r" % (k,))


class TiledLayer:
    """One 2-D layer of the PACS build stored as TILE x TILE tiles (FIX_SPEC.md sec 9).

    Layout of `<layer dir>/`:
        index.json            {"shape": [ny, nx], "tile": 256, "dtype": "float32", "fill": "nan" | 0.0 | false,
                               "wcs_header": str or null, "meta": {...} or null,
                               "present": [[iy, ix], ...] (ascending), "hpx512": [int, ...] (one per present tile:
                               the nside-512 nested galactic pixel under the tile's centre; -1 without a WCS)}
        tiles/<iy>_<ix>.npy   a full TILE x TILE array of `dtype` (pixels beyond the grid hold the fill), written only
                              when some pixel's bits differ from the fill's

    An absent tile reads as the fill. Writes are atomic per tile (a temporary file in tiles/, fsync, os.replace), so
    parallel workers writing DISTINCT tiles are safe; two workers must never write parts of the same tile. Writers open
    the layer with mode "r+"; the present list in index.json is refreshed by `write_index()`, which the one parent calls
    after its workers end (a reader opened in mode "r" trusts index.json; mode "r+" looks at the tiles/ directory).
    The tile index is storage only: the grid's shape, WCS and cell ids are not changed by it."""

    def __init__(self, directory: str, shape: Tuple[int, int], dtype, fill, tile: int = TILE,
                 wcs_header: Optional[str] = None, meta: Optional[dict] = None, present=None, mode: str = "r"):
        self.directory = directory
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = np.dtype(dtype)
        self.fill = fill
        self.tile = int(tile)
        self.wcs_header = wcs_header
        self.meta = meta
        self.mode = mode
        self._bits = _fill_bits(self.dtype, fill)
        self._present = set(map(tuple, present or ()))
        self.tile_rows, self.tile_cols = -(-self.shape[0] // self.tile), -(-self.shape[1] // self.tile)

    # -- opening and creating ------------------------------------------------------------------------------------------

    @classmethod
    def create(cls, directory: str, shape: Sequence[int], dtype=np.float32, fill=None, tile: int = TILE,
               wcs_header: Optional[str] = None, meta: Optional[dict] = None) -> "TiledLayer":
        """An empty layer at `directory` (created; an existing index or tile there is an error). `fill` defaults to
        LAYER_FILL[basename(directory)], else NaN for floats, 0 for integers and False for booleans. Returns it
        open in mode "r+"; `write_index()` writes index.json."""
        if fill is None:
            fill = LAYER_FILL.get(os.path.basename(os.path.normpath(directory)),
                                  False if np.dtype(dtype).kind == "b" else np.nan if np.dtype(dtype).kind == "f" else 0)
        tiles = os.path.join(directory, "tiles")
        if os.path.exists(os.path.join(directory, "index.json")) or (os.path.isdir(tiles) and os.listdir(tiles)):
            raise FileExistsError("a tiled layer already exists at %s" % directory)
        os.makedirs(tiles, exist_ok=True)
        layer = cls(directory, tuple(shape), dtype, fill, tile, wcs_header, meta, (), mode="r+")
        layer.write_index()
        return layer

    @classmethod
    def open(cls, directory: str, mode: str = "r") -> "TiledLayer":
        """The layer stored at `directory`; mode "r" (index.json's present list) or "r+" (writable; presence read
        from tiles/)."""
        if mode not in ("r", "r+"):
            raise ValueError("mode must be 'r' or 'r+'")
        with open(os.path.join(directory, "index.json")) as f:
            ix = json.load(f)
        layer = cls(directory, tuple(ix["shape"]), ix["dtype"], _fill_from_json(ix["fill"], ix["dtype"]), ix["tile"],
                    ix.get("wcs_header"), ix.get("meta"), [tuple(p) for p in ix["present"]], mode=mode)
        if mode == "r+":
            layer._present = set(layer._scan())
        return layer

    @classmethod
    def from_dense(cls, path_or_array: Union[str, np.ndarray], out_dir: str, fill=None, tile: int = TILE,
                   wcs_header: Optional[str] = None, meta: Optional[dict] = None) -> "TiledLayer":
        """A tiled copy of a dense 2-D array or .npy file (memory-mapped, read one tile row at a time) at `out_dir`;
        index.json written. Returns the layer open in mode "r"."""
        a = np.load(path_or_array, mmap_mode="r") if isinstance(path_or_array, str) else path_or_array
        if a.ndim != 2:
            raise ValueError("a tiled layer is 2-D")
        layer = cls.create(out_dir, a.shape, a.dtype, fill, tile, wcs_header, meta)
        t = layer.tile
        for iy in range(layer.tile_rows):
            strip = np.asarray(a[iy * t:(iy + 1) * t])
            for ix in range(layer.tile_cols):
                part = strip[:, ix * t:(ix + 1) * t]
                if layer._any_nonfill(part):
                    buf = layer._fill_tile()
                    buf[:part.shape[0], :part.shape[1]] = part
                    layer._write_tile(iy, ix, buf)
        layer.write_index()
        return cls.open(out_dir, "r")

    # -- properties ----------------------------------------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return 2

    def present(self) -> list:
        """The present tiles, [(iy, ix)] ascending (mode "r+": as on disk now)."""
        if self.mode == "r+":
            self._present = set(self._scan())
        return sorted(self._present)

    def tile_bounds(self, iy: int, ix: int) -> Tuple[int, int, int, int]:
        """(y0, y1, x0, x1) of tile (iy, ix) clipped to the grid."""
        t = self.tile
        return iy * t, min((iy + 1) * t, self.shape[0]), ix * t, min((ix + 1) * t, self.shape[1])

    def tile_path(self, iy: int, ix: int) -> str:
        return os.path.join(self.directory, "tiles", "%d_%d.npy" % (iy, ix))

    # -- reading -------------------------------------------------------------------------------------------------------

    def _has(self, iy: int, ix: int) -> bool:
        if self.mode == "r+":
            return os.path.exists(self.tile_path(iy, ix))
        return (iy, ix) in self._present

    def _fill_tile(self) -> np.ndarray:
        return np.full((self.tile, self.tile), self.fill, dtype=self.dtype)

    def read_tile(self, iy: int, ix: int) -> np.ndarray:
        """The full TILE x TILE array of tile (iy, ix) (the fill where absent); a private copy."""
        if not (0 <= iy < self.tile_rows and 0 <= ix < self.tile_cols) or not self._has(iy, ix):
            return self._fill_tile()
        a = np.load(self.tile_path(iy, ix))
        if a.shape != (self.tile, self.tile) or a.dtype != self.dtype:
            raise ValueError("tile %s has shape %s dtype %s" % (self.tile_path(iy, ix), a.shape, a.dtype))
        return a

    def window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """The pixels [y0, y1) x [x0, x1) as a new array; coordinates may reach outside the grid (read as the fill:
        what a tile's halo needs)."""
        out = np.full((max(y1 - y0, 0), max(x1 - x0, 0)), self.fill, dtype=self.dtype)
        if out.size == 0:
            return out
        t = self.tile
        ya, yb = max(y0, 0), min(y1, self.shape[0])
        xa, xb = max(x0, 0), min(x1, self.shape[1])
        if yb <= ya or xb <= xa:
            return out
        for iy in range(ya // t, (yb - 1) // t + 1):
            for ix in range(xa // t, (xb - 1) // t + 1):
                if not self._has(iy, ix):
                    continue
                a = self.read_tile(iy, ix)
                ty0, ty1 = max(ya, iy * t), min(yb, (iy + 1) * t)
                tx0, tx1 = max(xa, ix * t), min(xb, (ix + 1) * t)
                out[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0] = a[ty0 - iy * t:ty1 - iy * t, tx0 - ix * t:tx1 - ix * t]
        return out

    def _points(self, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        yy, xx = np.broadcast_arrays(np.asarray(yy, np.int64), np.asarray(xx, np.int64))
        ny, nx = self.shape
        if ((yy < -ny) | (yy >= ny) | (xx < -nx) | (xx >= nx)).any():
            raise IndexError("index out of bounds for a layer of shape %s" % (self.shape,))
        yy, xx = yy % ny, xx % nx
        out = np.full(yy.shape, self.fill, dtype=self.dtype)
        t = self.tile
        key = (yy // t) * self.tile_cols + xx // t
        flat_key, flat_y, flat_x = key.ravel(), yy.ravel(), xx.ravel()
        res = out.ravel()
        for k in np.unique(flat_key):
            iy, ix = divmod(int(k), self.tile_cols)
            if not self._has(iy, ix):
                continue
            sel = np.flatnonzero(flat_key == k)
            a = self.read_tile(iy, ix)
            res[sel] = a[flat_y[sel] - iy * t, flat_x[sel] - ix * t]
        return res.reshape(yy.shape)

    def __getitem__(self, key) -> np.ndarray:
        """numpy indexing of the full layer: (slices or ints, any step) or a pair of integer arrays (point reads,
        broadcast together); absent tiles read as the fill. Always a new array."""
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            i = [j for j, k in enumerate(key) if k is Ellipsis][0]
            key = key[:i] + (slice(None),) * (2 - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (2 - len(key))
        if len(key) != 2:
            raise IndexError("a tiled layer is 2-D")
        ky, kx = _norm_axis(key[0], self.shape[0]), _norm_axis(key[1], self.shape[1])
        if ky[0] == "array" or kx[0] == "array":
            yy = ky[1] if ky[0] in ("array", "int") else np.arange(*ky[1:])
            xx = kx[1] if kx[0] in ("array", "int") else np.arange(*kx[1:])
            if ky[0] == "slice" or kx[0] == "slice":
                raise TypeError("mixing an integer array with a slice is not supported by TiledLayer")
            return self._points(yy, xx)

        def rng(k):
            if k[0] == "int":
                return k[1], k[1] + 1, None
            start, stop, step = k[1:]
            idx = np.arange(start, stop, step)
            if idx.size == 0:
                return 0, 0, idx
            return int(idx.min()), int(idx.max()) + 1, (None if step == 1 else idx)
        y0, y1, yi = rng(ky)
        x0, x1, xi = rng(kx)
        w = self.window(y0, y1, x0, x1)
        if yi is not None:
            w = w[yi - y0] if yi.size else w[:0]
        if xi is not None:
            w = w[:, xi - x0] if xi.size else w[:, :0]
        if ky[0] == "int":
            w = w[0]
            return w[0] if kx[0] == "int" else w
        if kx[0] == "int":
            return w[:, 0]
        return w

    def iter_tiles(self, halo: int = 0, tiles: Optional[Sequence[Tuple[int, int]]] = None
                   ) -> Iterator[Tuple[int, int, np.ndarray]]:
        """(iy, ix, array) over the present tiles (or `tiles`), the array (TILE + 2 halo) square: rows
        iy * TILE - halo .. (iy + 1) * TILE + halo, columns likewise, the fill outside the grid and in absent tiles."""
        t, h = self.tile, int(halo)
        for iy, ix in (self.present() if tiles is None else tiles):
            yield iy, ix, self.window(iy * t - h, (iy + 1) * t + h, ix * t - h, (ix + 1) * t + h)

    def to_dense(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """The whole layer as a dense array (`out`: a preallocated (ny, nx) array or memmap to fill)."""
        a = np.full(self.shape, self.fill, dtype=self.dtype) if out is None else out
        if out is not None:
            a[...] = self.fill
        for iy, ix in self.present():
            y0, y1, x0, x1 = self.tile_bounds(iy, ix)
            a[y0:y1, x0:x1] = self.read_tile(iy, ix)[:y1 - y0, :x1 - x0]
        return a

    # -- writing -------------------------------------------------------------------------------------------------------

    def _any_nonfill(self, a: np.ndarray) -> bool:
        return bool((np.ascontiguousarray(a).view(_bits_dtype(self.dtype)) != self._bits).any())

    def _write_tile(self, iy: int, ix: int, a: np.ndarray) -> None:
        path = self.tile_path(iy, ix)
        tmp = os.path.join(os.path.dirname(path), ".%d_%d.npy.tmp%d" % (iy, ix, os.getpid()))
        with open(tmp, "wb") as f:
            np.save(f, np.ascontiguousarray(a, dtype=self.dtype))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        self._present.add((iy, ix))

    def _drop_tile(self, iy: int, ix: int) -> None:
        try:
            os.remove(self.tile_path(iy, ix))
        except FileNotFoundError:
            pass
        self._present.discard((iy, ix))

    def __setitem__(self, key, value) -> None:
        """Writes `value` into the pixels of a (slice, slice) key of step 1 (ints taken as one-wide slices); each tile
        touched is read, updated and written back atomically, or removed when it then holds only the fill."""
        if self.mode != "r+":
            raise PermissionError("the layer is open read-only")
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("TiledLayer assignment takes (rows, columns)")
        bounds = []
        for k, n in zip(key, self.shape):
            if isinstance(k, (int, np.integer)):
                k = slice(int(k) % n, int(k) % n + 1)
            if not isinstance(k, slice):
                raise TypeError("TiledLayer assignment takes slices")
            a, b, s = k.indices(n)
            if s != 1:
                raise ValueError("TiledLayer assignment takes step-1 slices")
            bounds.append((a, max(a, b)))
        (y0, y1), (x0, x1) = bounds
        if y1 <= y0 or x1 <= x0:
            return
        v = np.broadcast_to(np.asarray(value, dtype=self.dtype), (y1 - y0, x1 - x0))
        t = self.tile
        for iy in range(y0 // t, (y1 - 1) // t + 1):
            for ix in range(x0 // t, (x1 - 1) // t + 1):
                ty0, ty1 = max(y0, iy * t), min(y1, (iy + 1) * t)
                tx0, tx1 = max(x0, ix * t), min(x1, (ix + 1) * t)
                part = v[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]
                full = (ty1 - ty0 == t) and (tx1 - tx0 == t)
                if full:
                    buf = np.array(part, dtype=self.dtype)
                elif self._has(iy, ix):
                    buf = self.read_tile(iy, ix)
                    buf[ty0 - iy * t:ty1 - iy * t, tx0 - ix * t:tx1 - ix * t] = part
                else:
                    if not self._any_nonfill(part):
                        continue
                    buf = self._fill_tile()
                    buf[ty0 - iy * t:ty1 - iy * t, tx0 - ix * t:tx1 - ix * t] = part
                if self._any_nonfill(buf):
                    self._write_tile(iy, ix, buf)
                elif self._has(iy, ix):
                    self._drop_tile(iy, ix)

    def _scan(self) -> list:
        out = []
        for n in os.listdir(os.path.join(self.directory, "tiles")):
            if n.endswith(".npy") and not n.startswith("."):
                iy, ix = n[:-4].split("_")
                out.append((int(iy), int(ix)))
        return sorted(out)

    def _tile_centres_hpx(self, present: list) -> list:
        if not self.wcs_header or not present:
            return [-1] * len(present)
        wcs = celestial_wcs(self.wcs_header)
        b = np.array([self.tile_bounds(iy, ix) for iy, ix in present], float)
        cy, cx = (b[:, 0] + b[:, 1] - 1) / 2.0, (b[:, 2] + b[:, 3] - 1) / 2.0
        ra, dec = pixel_to_sky(wcs, cx, cy)
        return [int(v) for v in hpx512_galactic(ra, dec)]

    def write_index(self) -> None:
        """Writes index.json (atomically) with the present tiles as found in tiles/ now and their nside-512 pixels.
        Called by the one parent after every writer has ended."""
        present = self._scan()
        self._present = set(present)
        ix = dict(shape=list(self.shape), tile=self.tile, dtype=self.dtype.name, fill=_fill_json(self.fill),
                  wcs_header=self.wcs_header, meta=self.meta, present=[list(p) for p in present],
                  hpx512=self._tile_centres_hpx(present))
        path = os.path.join(self.directory, "index.json")
        tmp = path + ".tmp%d" % os.getpid()
        with open(tmp, "w") as f:
            json.dump(ix, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def disk_bytes(self) -> int:
        """Bytes of the present tile files."""
        return sum(os.path.getsize(self.tile_path(iy, ix)) for iy, ix in self.present())

    def __repr__(self) -> str:
        return "TiledLayer(%r, shape=%s, dtype=%s, fill=%r, tiles=%d/%d)" % (
            self.directory, self.shape, self.dtype, self.fill, len(self._present), self.tile_rows * self.tile_cols)
