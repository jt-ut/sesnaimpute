"""The PACS point-spread function: the release templates and encircled-energy tables, the input maps a mosaic is built
from, and the mosaic's PSF (the local mixture of the input maps' templates on the output grid).

Round-1 skeleton (CONTRACT.md): signatures only; bodies are written in round 2 from the old code named in each docstring.
Depends on: grid. Replaces: sky/derived/pacs_psf.py (all of it but `CachedPairedPSF`/`PairedPSF`, which only served
non-mosaic maps), pacs_fit.build_mosaic_psf/_mosaic_node/_node_table/_kernel_weights/_template_density/_jacobian/
_table_aperture/_table_integral/mosaic_psf, and the input-map primitives pacs_mosaic._read_layers/_good/valid_at/
_smooth_lattice/_lattice_at (moved here because MosaicPSF.weights needs them and psf precedes mosaic).

Store (under a mosaic directory, see mosaic.py): `psf/psf.json` (the PSF's data only: sub-group keys, per input map
{path, wcs_header, shape, smooth, smooth_step, scale, sub, bbox, hdus}, extent, subpixel_steps, integration_order, fbar,
enclosed_sub, aperture_fraction_sub, sub_beams, beam_arcsec, enclosed_fraction, aperture_fraction,
broad_over_plain_total, table_half), `psf/sub<k>.npy` (float32 (n_nodes, n_nodes, 1, count, count)), `psf/table.npy`
(the fbar-weighted mean, same shape). Unchanged from today but for the folder name (`_psf` -> `psf`) and the smoothed
error's key (`lattice` -> `smooth`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sesnaimpute.sky.derived.pacs import grid
from sesnaimpute.sky.derived.pacs.grid import PixelGrid, TiledLayer

ROUND2 = "round 2"


# ----------------------------------------------------------------------------------------------------------------------
# the release templates and encircled energies (pacs_psf.py:114-583)

def beam_fwhm_arcsec(record) -> float:
    """The beam FWHM (geometric mean of the release's fitted axes) of a map's (band, mode, speed).
    Input: a MapRecord. Output: arcsec. Reads nothing. Identity: `pacs_psf.beam_fwhm_arcsec` (pacs_psf.py:122)."""
    raise NotImplementedError(ROUND2)


def psf_directory(config) -> str:
    """The PSF release directory under config.inputs["herschel_pacs_maps"] (`pacs_psf.psf_directory`)."""
    raise NotImplementedError(ROUND2)


def aperture_eef(config, record) -> float:
    """The fraction of the total flux inside the eHOPS aperture radius (definitions.PACS_APERTURES[band][0]), no sky:
    the divisor of every aperture sum. Reads the release EEF table of the map's (band, mode, speed).
    Identity: `pacs_psf.aperture_eef` (pacs_psf.py:171), bit-identical."""
    raise NotImplementedError(ROUND2)


def aperture_fraction(config, record) -> float:
    """The fraction of the total flux the aperture keeps after the annulus sky (`pacs_psf.aperture_fraction`,
    pacs_psf.py:155), bit-identical."""
    raise NotImplementedError(ROUND2)


@dataclass(frozen=True)
class PublishedPSF:
    """One release template (`pacs_psf.PublishedPSF`): `image` (201, 201) float64, `position_angle_deg`,
    `total_flux_factor`, `path`."""
    image: np.ndarray
    position_angle_deg: float
    total_flux_factor: float
    path: str

    @classmethod
    def read(cls, config, record, array_to_map_angle: str, *, position_angle_deg: float) -> "PublishedPSF":
        """Reads the release template of the map's (band, mode, speed) at `array_to_map_angle` ("+42" or "-42").
        Identity: `pacs_psf.PublishedPSF.read` (pacs_psf.py:251)."""
        raise NotImplementedError(ROUND2)

    def density(self, east_arcsec: np.ndarray, north_arcsec: np.ndarray) -> np.ndarray:
        """The template's surface density at sky offsets (arcsec east, north), normalised to unit total flux
        (`PublishedPSF.density`, pacs_psf.py:276)."""
        raise NotImplementedError(ROUND2)


def published_components(config, record) -> List[PublishedPSF]:
    """The one or two release templates of a map (the +42/-42 scan columns) at the map's scan angles. Reads the
    map header (scan angles) and the release. Identity: `pacs_psf.published_components_for_map` (pacs_psf.py:526) --
    its order is the PSF group key's order (mosaic.group_key)."""
    raise NotImplementedError(ROUND2)


# ----------------------------------------------------------------------------------------------------------------------
# the input maps of a mosaic

def good_pixels(image: np.ndarray, error: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Where an input map's pixel is usable: finite image and error, error > 0, coverage > 0 (`pacs_mosaic._good`)."""
    raise NotImplementedError(ROUND2)


@dataclass
class InputMap:
    """One input map of a mosaic: `record` (MapRecord), `wcs` (celestial, `grid.celestial_wcs(record.wcs_header)`),
    `maps_root` (config's herschel_pacs_maps directory), `smooth` (its smoothed error, TiledLayer float64 fill NaN, in
    INPUT pixels sampled every `smooth_step`), `smooth_step` (FROZEN 6: `grid.smooth_step_px(record.pixel_scale_arcsec)`).
    The FITS file is opened read-only ("denywrite", shared pages; pacs_psf.py:609-621 explains why not "readonly")."""
    record: object
    wcs: object
    maps_root: str
    smooth: Optional[TiledLayer] = None
    smooth_step: int = 0

    def layers(self, rows: slice, cols: slice) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """float64 copies of the map's (image, error, coverage) window rows x cols (`pacs_mosaic._read_layers`)."""
        raise NotImplementedError(ROUND2)

    def valid_at(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Whether the mosaic uses the map at input pixel (px, py): the four interpolation neighbours good (a neighbour
        of bilinear weight zero not asked). Identity: `pacs_mosaic.valid_at` (pacs_mosaic.py:197)."""
        raise NotImplementedError(ROUND2)

    def smoothed_error_at(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """FROZEN 6. The smoothed error at input pixel (px, py) by bilinear interpolation over the valid samples (NaN
        where none). Identity: `pacs_mosaic._lattice_at` (pacs_mosaic.py:244), bit-identical."""
        raise NotImplementedError(ROUND2)


def build_smoothed_error(input_map: InputMap, out_dir: str) -> TiledLayer:
    """FROZEN 6. The median of the map's valid error pixels in a square window of grid.SMOOTH_WINDOW_ARCSEC, sampled
    every window_px // 2 input pixels (sample (j, i) at pixel (i * step, j * step)); NaN where the window has no valid
    pixel. Reads the input FITS band by band (memory: one window of rows). Writes `out_dir` (TiledLayer float64, fill
    NaN). Identity: `pacs_mosaic._smooth_lattice` (pacs_mosaic.py:214): every sample bit-identical."""
    raise NotImplementedError(ROUND2)


# ----------------------------------------------------------------------------------------------------------------------
# the mosaic PSF

@dataclass(frozen=True)
class PsfInput:
    """What `build_mosaic_psf` needs of one input map of a mosaic (the mosaic's META entry, in META order): the
    map, its PSF group index (mosaic.group_key order), the smoothed error's store directory relative to the mosaic
    and its step."""
    input_map: InputMap
    group: int
    smooth_dir: str
    smooth_step: int


def build_mosaic_psf(config, region: str, band: str, pixel_grid: PixelGrid, inputs: Sequence[PsfInput],
                     image: TiledLayer, mosaic_dir: str, n_jobs: Optional[int] = None) -> "MosaicPSF":
    """Builds the mosaic's PSF tables. Inputs: the grid, the input maps in META order, the mosaic image layer (the
    fbar sample reads its finiteness at 40 x 40 grid points). Writes `<mosaic_dir>/psf/psf.json`, `sub<k>.npy`,
    `table.npy`; joblib over (sub-group, node) at n_jobs. Returns it opened.
    Identity: `pacs_fit.build_mosaic_psf` (pacs_fit.py:1581-1676) bit-identical: FROZEN 5 (node count, odd, centre
    node normalises with `aperture_eef / centre's table aperture`), FROZEN 7 (sub-group keys, first appearance, angle
    from the Jacobian at the grid centre ((nx-1)/2, (ny-1)/2)), FROZEN 14 (node layout `PixelGrid.psf_nodes()`), the
    table size (half, table_half = min(half + ceil(2 beam_px), 65 // pix), extent = table_half + 0.5, count =
    int(2 * extent * 8) + 1), and the fbar sample."""
    raise NotImplementedError(ROUND2)


class MosaicPSF:
    """The PSF of a mosaic: at an output position the weight-fraction mixture sum_k f_k P_k over the sub-groups of
    the input maps valid there (f from 1 / smoothed error^2, the mosaic's own weights). Read from `<mosaic>/psf/`; the
    tables memory-mapped. Picklable by (directory, maps_root, tables). Identity: `pacs_psf.MosaicPSF`
    (pacs_psf.py:627-793), every method bit-identical (the node lookup is `grid.psf_node_coord` / `psf_node_cell`)."""

    directory: str
    maps_root: str
    shape: Tuple[int, int]
    extent: float
    subpixel_steps: int
    n_sub: int
    table_half: int

    def __init__(self, directory: str, maps_root: str, tables: bool = True) -> None:
        """Opens `<directory>/psf/` (psf.json; with `tables` the sub<k>.npy and table.npy memory-mapped) and each input
        map's smoothed error."""
        raise NotImplementedError(ROUND2)

    def weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(n, n_sub) mixture fractions at output positions; a position no map covers gets fbar
        (`MosaicPSF.weights`, pacs_psf.py:679)."""
        raise NotImplementedError(ROUND2)

    def evaluate_xy(self, source_x: np.ndarray, source_y: np.ndarray, pixel_x: np.ndarray, pixel_y: np.ndarray,
                    mixture: Optional[np.ndarray] = None) -> np.ndarray:
        """(n, 1, n_pix) the local-mixture response at the pixels around each source (`MosaicPSF.evaluate_xy`,
        pacs_psf.py:734) -- the injection template."""
        raise NotImplementedError(ROUND2)

    def evaluate_mean_xy(self, source_x: np.ndarray, source_y: np.ndarray, pixel_x: np.ndarray,
                         pixel_y: np.ndarray) -> np.ndarray:
        """The weight-mean table's response (`MosaicPSF.evaluate_mean_xy`, pacs_psf.py:767) -- the detection kernel's."""
        raise NotImplementedError(ROUND2)

    def enclosed_at(self, x: np.ndarray, y: np.ndarray, mixture: Optional[np.ndarray] = None) -> np.ndarray:
        """The mixture's fraction of the total flux inside the aperture radius at output positions (pacs_psf.py:775)."""
        raise NotImplementedError(ROUND2)

    def aperture_fraction_at(self, x: np.ndarray, y: np.ndarray, mixture: Optional[np.ndarray] = None) -> np.ndarray:
        """The mixture's aperture fraction after the annulus sky at output positions (pacs_psf.py:781)."""
        raise NotImplementedError(ROUND2)

    def beam_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The mixture's beam FWHM (arcsec) at output positions (pacs_psf.py:786)."""
        raise NotImplementedError(ROUND2)

    def beam_at_sky(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        """`beam_at` at sky positions (pacs_psf.py:790)."""
        raise NotImplementedError(ROUND2)


def mosaic_scalar(mosaic_dir: str, name: str) -> float:
    """A weight-mean scalar of the mosaic PSF (beam_arcsec, aperture_fraction, enclosed_fraction) from psf.json
    (`pacs_psf.mosaic_scalar`)."""
    raise NotImplementedError(ROUND2)
