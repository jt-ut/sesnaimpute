"""PACS fluxes of the region: the cross-band merge of the bands' detections into PACS sources, the forced photometry at
every SESNA position, and the one SESNA-PACS cross-match (ORIGIN 1 detected, 3 censored bound, 90 undetected, 91 not
covered).

Round-1 skeleton (CONTRACT.md): signatures only. Depends on: grid, psf, mosaic, detect, inject. Replaces:
pacs_completeness.merge_sources/_measure_at/_pair_scatter/_unit/_chord/_sep_arcsec, pacs_photometry.singleton_pass/
calibrated_sigma/_measure_tile/cross_match/print_offsets/_runs/_build_one_region/build/_read_cells/_unit_vectors/
_sep_arcsec/_region_positions. The aperture recipe itself is `mosaic.Mosaic.aperture` (written once there: detect and
inject, earlier in the order, measure with it too). The spherical helpers are written once here. Unused today and not
carried: write_region's flux_grid/scatter/cell_row parameters, singleton_pass's fifth return value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from sesnaimpute.sky.derived.pacs import detect, inject, mosaic
from sesnaimpute.sky.derived.pacs.inject import BandCompleteness
from sesnaimpute.sky.derived.pacs.mosaic import Mosaic

ROUND2 = "round 2"

#: the pipeline's origin codes for a flux (`ORIGIN_<B>` in the products)
ORIGIN_DETECTED, ORIGIN_BOUND, ORIGIN_UNDETECTED, ORIGIN_NOT_COVERED = 1, 3, 90, 91


def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """(n, 3) unit vectors of ICRS positions (written once; was pacs_photometry._unit_vectors and
    pacs_completeness._unit)."""
    raise NotImplementedError(ROUND2)


def sep_arcsec(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angular separation (arcsec) of unit vectors (was written twice)."""
    raise NotImplementedError(ROUND2)


def chord(arcsec: float) -> float:
    """The chord length of an angle in arcsec (kd-tree radius on unit vectors)."""
    raise NotImplementedError(ROUND2)


def calibrated_sigma(scatter: np.ndarray, flux_grid_mjy: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """A detection's SIGMA: the row's SCATTER curve read at its flux (`pacs_photometry.calibrated_sigma`); x CELL_SIGMA
    by the caller. NaN only where SCATTER is NaN (the legacy short-cell rule)."""
    raise NotImplementedError(ROUND2)


@dataclass
class Sources:
    """The region's PACS sources (the sources product's columns): position band, ra, dec, per band fnu, sigma, snr,
    detected, extended, map position, row; MATCH_RADIUS; CHANCE."""
    columns: Dict[str, np.ndarray]


def merge_sources(config, bands: Dict[str, BandCompleteness], mosaics: Dict[str, Mosaic], flux_grid_mjy: np.ndarray,
                  n_jobs: Optional[int] = None) -> Sources:
    """The bands' blind detections merged across bands (planner ruling 2026-10-28): pairs within the pair's merge radius
    are one source (each band once, nearest first, brightest first); position = the sharpest detecting band's; each
    detecting band's flux = Mosaic.aperture at its own detection position; SIGMA = CELL_SIGMA x calibrated_sigma.
    Identity: `pacs_completeness.merge_sources` (pacs_completeness.py:1844) bit-identical."""
    raise NotImplementedError(ROUND2)


@dataclass
class ForcedPhotometry:
    """Per SESNA source and band: FNU, SIGMA (cell), OFFSET, CELL row, map position, covered."""
    columns: Dict[str, np.ndarray]


def forced_photometry(config, region: str, mosaics: Dict[str, Mosaic], bands: Dict[str, BandCompleteness],
                      ra: np.ndarray, dec: np.ndarray, n_jobs: Optional[int] = None) -> ForcedPhotometry:
    """Forced aperture flux at every SESNA position (registration shift applied, FROZEN 9), per tile of positions
    (pieces of >= PHOTOMETRY_PIECE_MIN sources), SIGMA = the cell's CELL_SIGMA. Identity: `pacs_photometry.write_region`
    + `_measure_tile` (pacs_photometry.py:143, 322) bit-identical."""
    raise NotImplementedError(ROUND2)


def cross_match(photometry: ForcedPhotometry, sources: Sources, ra: np.ndarray, dec: np.ndarray,
                mosaics: Dict[str, Mosaic]) -> Tuple[ForcedPhotometry, dict]:
    """The one SESNA-PACS association: nearest neighbour within MATCH_RADIUS_ARCSEC per band; a sole match takes the
    source's FNU/SIGMA (ORIGIN 1), a SESNA source within one beam of a detection it does not own is ORIGIN 3 (censored
    bound), else 90 (covered) or 91; CONFUSED, EXTENDED, PACS_SOURCE, MATCH_SEPARATION columns; and the expected chance
    coincidences (CHANCE_DENSITY_ARCSEC). Returns the updated table and the counts. Identity: `pacs_photometry.
    cross_match` (pacs_photometry.py:178)."""
    raise NotImplementedError(ROUND2)


def registration_offsets(photometry: ForcedPhotometry, mosaics: Dict[str, Mosaic]) -> dict:
    """Per input map, the median PACS - SESNA offset of its associated pairs (>= MIN_OFFSET_PAIRS)
    (`pacs_photometry.print_offsets`, now returned for checks.py)."""
    raise NotImplementedError(ROUND2)
