"""Every acceptance check of the PACS build, each returning `products.CheckRow`s that the stage stores in the product's
CHECKS table (and prints): the build-time checks that stop a build (zero image, normalisation), the information-only
real-map check, and the region checks after FINISH (C1, C2, the detection report, C4/C5 against HOPS, the HPPSC
checks, the chance coincidences).

Round-1 skeleton (CONTRACT.md): signatures only. Depends on: grid, psf, mosaic, detect, inject, photometry, products.
Replaces: pacs_completeness.self_check/_real_check/_acceptance_c1/_acceptance_c2/_report_detection/acceptance/
_binomial_error/_verdict/_robust/_at_injected_flux, pacs_checks.py (all: representative_maps, _aperture_sum,
template_in_aperture, unit_source, expected_enclosed, check_positions, _real_sources, normalisation_check),
pacs_photometry.acceptance/_c4_ratio/_read_hops/_match_hops/_read_hppsc/hppsc_checks, and the print-only reporting of
finish_region. The two normalisation paths (self_check and normalisation_check) stay two checks with distinct names
(they test different things: the zero-image sky term vs the PSF divisor), run from ONE place (stages, stage "psf").
Bars are today's (RELEASE_TOLERANCE 0.015, TIE 0.003, SUM 0.01, zero check 0.005, ...), named constants here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from sesnaimpute.sky.derived.pacs import inject, mosaic, photometry, products
from sesnaimpute.sky.derived.pacs.inject import BandCompleteness
from sesnaimpute.sky.derived.pacs.mosaic import Mosaic
from sesnaimpute.sky.derived.pacs.products import CheckRow

ROUND2 = "round 2"


def zero_image_check(config, m: Mosaic) -> List[CheckRow]:
    """ZERO_IMAGE: the aperture sum of injected unit sources on a zero image over (injected x EEF) at
    ZERO_CHECK_POSITIONS positions; bar ZERO_CHECK_TOLERANCE; FAIL stops the build. Works on a private zero Window per
    position (no whole-grid array, D sec 1). Identity: the numbers `pacs_completeness.self_check` prints."""
    raise NotImplementedError(ROUND2)


def normalisation_check(config, m: Mosaic, n_jobs: Optional[int] = None) -> List[CheckRow]:
    """NORMALISATION (real/divisor, injected/real, patch sums, template/release EEF) per sub-group at the check
    positions; bars TIE 0.003, SUM 0.01 (rims only where the sub-group contributes), RELEASE 0.015. Identity: the values
    `pacs_checks.normalisation_check` computes (its `.rows`)."""
    raise NotImplementedError(ROUND2)


def real_map_check(m: Mosaic, injector: "inject.Injector", flux_mjy: float, half: int,
                   rows: Optional[np.ndarray] = None) -> List[CheckRow]:
    """REAL_MAP (information only): injected sources on the real map recovered as on the zero image, on a private
    Window (no whole-grid copy-on-write map). Identity: `pacs_completeness._real_check`'s numbers."""
    raise NotImplementedError(ROUND2)


def completeness_c1(band: str, completeness: dict, held: dict) -> List[CheckRow]:
    """C1: the held-out half's recovered fraction against the build half's CURVE, per flux bin (binomial error);
    `completeness` = products.read_completeness_band. Identity: `_acceptance_c1`'s values."""
    raise NotImplementedError(ROUND2)


def completeness_c2(band: str, completeness: dict, held: dict) -> List[CheckRow]:
    """C2: the held-out (dflux / SIGMA) scatter per S/N bin (definitions.PACS_SNR_BIN_EDGES); no read is dropped (a NaN
    SIGMA is counted and reported as its own row). Identity: `_acceptance_c2`'s values on the non-NaN reads."""
    raise NotImplementedError(ROUND2)


def detection_report(band: str, completeness: dict, held: dict) -> List[CheckRow]:
    """DETECTIONS (information): detections and inverted-map peaks per sky bin (`_report_detection`)."""
    raise NotImplementedError(ROUND2)


def hops_checks(config, region: str, photometry_path: str, reference_product: Optional[str] = None) -> List[CheckRow]:
    """C4 (ours/HOPS 70 um ratio above C4_MIN_F70_MJY, and against `reference_product`) and C5 (recovery per flux bin
    C5_EDGES_MJY). Identity: `pacs_photometry.acceptance`'s values."""
    raise NotImplementedError(ROUND2)


def hppsc_checks(config, region: str, sources_path: str, photometry_path: str,
                 box: Optional[Sequence[float]] = None) -> List[CheckRow]:
    """HPPSC recovery (>= HPPSC_RECOVERY_BAR above HPPSC_RECOVERY_SNR), flux ratio (HPPSC_FLUX_BAR), offsets, coverage
    (`pacs_photometry.hppsc_checks`)."""
    raise NotImplementedError(ROUND2)


def chance_check(counts: dict) -> List[CheckRow]:
    """CHANCE (information): expected chance coincidences of the association per band (was the
    EXPECTED_CHANCE_COINCIDENCES attribute and a print)."""
    raise NotImplementedError(ROUND2)


def region_checks(config, region: str, bands: Dict[str, BandCompleteness], box: Optional[Sequence[float]] = None,
                  reference_product: Optional[str] = None) -> List[CheckRow]:
    """Every check after FINISH, in one list (C1, C2, detections, C4/C5, HPPSC, chance), for products.write_checks."""
    raise NotImplementedError(ROUND2)
