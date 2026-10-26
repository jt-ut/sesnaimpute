"""The one PACS measurement code path, shared by `pacs_completeness` and
`pacs_photometry` (SPEC_PACS_FARIR_DRAFT.md sections 4, 5, 7b and 11.3).

STATUS: interface committed first so the photometry stage can code against
it; bodies follow in the next commit. Everything below is the public surface;
the names, argument order and return fields will not change.

What it does, for ONE map tile (one PACS map, one band):

  with open_tile(config, record) as tile:
      tile.xy(ra, dec)                       sky -> map pixel, one vectorised WCS call
      tile.neighbours                        blind bright sources (11.3 step 1)
      tile.fit_positions(x, y)               forced fit, nuisance columns included
      tile.fit_groups(x, y, group)           simultaneous fit of blend groups
      tile.grid                              the 30-arcsec cell grid on this map
      tile.pixel_sigma(x, y)                 local propagated pixel noise (deepest-map choice)

The fit at a position is linear least squares of two scan-direction PSF
columns for the target (flux = the sum of their two amplitudes), two columns
for every bright neighbour whose PSF reaches the fitting box (a nuisance,
dropped where it lies within one beam of the target, because the target's own
column then carries it) and one constant background, on a square box of
half-width FIT_BOX_FWHM x beam FWHM, weighted by the map's error layer. The
"error" it returns is the PROPAGATED PIXEL NOISE only (the error layer
carried through the linear operator); the cell sigma that sets detection is
measured by the completeness stage from the scatter of these same fits at
random empty-sky positions, never added to this term.

All fluxes are mJy. All pixel coordinates are zero-origin map pixels (x
along the fastest axis).
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

# TEMPORARY (unit DOWNLOAD / unit PSF not landed): the old package supplies the
# map record and index reader.  Replace this ONE line by
#   from sesnaimpute.sky.download.herschel_pacs.index import MapRecord, read_index, maps_root
from sesnaimpute.farIRcomp.maps import MapRecord, read_index  # noqa: F401
# TEMPORARY: replace this ONE line by
#   from sesnaimpute.sky.derived.pacs_psf import (published_components_for_map, CachedPairedPSF, beam_fwhm_arcsec)
from sesnaimpute.farIRcomp.psf import published_components_for_map, CachedPairedPSF  # noqa: F401

#: detection threshold in sigma (owner 2026-10-24); defines recovery in the completeness
DETECTION_SNR = 5.0
#: fitting-box half-width in beam FWHM (spec 11.3 step 1)
FIT_BOX_FWHM = 3.0
#: a pixel is used only where the coverage layer exceeds this
MIN_COVERAGE = 0.0
#: bright-neighbour finding: single-PSF fit S/N against the propagated pixel noise
NEIGHBOUR_SNR = 5.0
#: completeness cell side (owner 2026-10-24: SESNA's own)
CELL_ARCSEC = 30.0


@dataclass(frozen=True)
class Neighbours:
    """Blind bright sources on one tile: x, y (map pixels), flux (mJy), snr."""
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    snr: np.ndarray


@dataclass(frozen=True)
class PositionFit:
    """Forced fits at n positions.  Rows with covered False carry NaN.

    flux      (n,) mJy, summed amplitude of the two scan columns
    gain      (n,) response of `flux` to a unit-flux injected beam (0.5/0.5 scan mix)
    sigma     (n,) propagated pixel noise on `flux`, mJy
    covered   (n,) bool, the position lies on valid data and the fit is well posed
    n_nuisance (n,) int, bright-neighbour sources included as nuisance columns
    """
    flux: np.ndarray
    gain: np.ndarray
    sigma: np.ndarray
    covered: np.ndarray
    n_nuisance: np.ndarray


@dataclass(frozen=True)
class GroupFit:
    """Simultaneous fits of blend groups, one row per member (input order).

    flux        (n,) mJy per member
    sigma       (n,) propagated pixel noise on each member's flux in the group fit
    inflation   (n,) sigma / (propagated sigma of a singleton fit at the same position):
                the correlation inflation photometry applies to the cell sigma
    max_abs_rho (n,) largest |correlation| of the member's fitted flux with any other
                member of its group (1.0 when the group fit is singular)
    group_flux  (n,) sum of the group's member fluxes (same value on every member)
    group_sigma (n,) propagated pixel noise of that sum (full covariance)
    covered     (n,) bool
    """
    flux: np.ndarray
    sigma: np.ndarray
    inflation: np.ndarray
    max_abs_rho: np.ndarray
    group_flux: np.ndarray
    group_sigma: np.ndarray
    covered: np.ndarray


class MapGrid:
    """30-arcsec cell grid on a map: y-major cell ids, aligned with pixel axes.

    cell_ids(x, y) -> int64 ids (-1 outside); bounds(ids) -> x0, x1, y0, y1;
    centres_icrs(ids) -> ra, dec (deg); sample_pixels(ids, n, seed) -> x, y
    (n_ids, n) uniform in each cell, reproducible per (tile seed, cell, draw).
    """


class Tile:
    """One open map tile with its fitting machinery (build with `open_tile`)."""

    def xy(self, ra_deg, dec_deg):
        """ICRS degrees -> zero-origin map pixels (x, y); one vectorised WCS call."""
        raise NotImplementedError

    def pixel_sigma(self, x, y):
        """Error-layer value (mJy/pixel) at the nearest pixel, NaN off the data.
        The photometry/completeness stages use the cell sigma for the deepest-map
        choice; this is the cheap map-level proxy for candidates not in a cell."""
        raise NotImplementedError

    def fit_positions(self, x, y, batch=512):
        """Forced fit at each (x, y) -> PositionFit.  Same code for real sources
        and for random completeness positions."""
        raise NotImplementedError

    def fit_groups(self, x, y, group, batch=64):
        """Simultaneous fit of blend groups -> GroupFit.  `group` (n,) integer
        group label per row; rows with the same label are fitted together."""
        raise NotImplementedError

    def fitted_fwhm(self, x, y):
        """Best-fit total FWHM (arcsec) of a source at each position: the beam
        broadened by a Gaussian chosen on a small grid by chi-square.  Intended
        for detections only (photometry's EXTENDED flag = this > 1.5 x beam)."""
        raise NotImplementedError


@contextmanager
def open_tile(config, record):
    """Open `record` read-only (memory-mapped), build the PSF response table once
    per process, find the bright neighbours, and yield a `Tile`."""
    raise NotImplementedError
    yield


def select_deeper(tile_index, position_indices, local_rms, covered, selected_tile, selected_rms):
    """Deepest-map choice across overlapping maps (lowest sigma wins; ties go to
    the smaller tile ordinal).  Updates `selected_tile` (int32, init -1) and
    `selected_rms` (float64, init +inf) in place at `position_indices`."""
    raise NotImplementedError


def admitted_sample_points(admitted):
    """ICRS RA/Dec (n_pixel, 256) of the 256 nested children at nside 8192 of each
    admitted nside-512 Galactic NESTED pixel."""
    raise NotImplementedError


def load_sesna_candidates(config, centre_ra_deg, centre_dec_deg, radius_deg):
    """Every curated SESNA source (all regions' catalogues) inside a sky cap, as
    (SourceCandidates(ra_deg, dec_deg, source_row, catalogue), audit_dict)."""
    raise NotImplementedError
