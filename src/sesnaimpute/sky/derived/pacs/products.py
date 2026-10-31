"""The PACS build's three products, with UNITS and READING on every dataset, and the CHECKS table they carry.

    completeness_pacs_source__<Region>.hdf5   per covered cell and band: CURVE, FLUX90, SCATTER, BIAS, CELL_SIGMA,
                                              CELL_OFFSET, ...; SOURCE_ROW per SESNA source; MAP_SHAPE/MAP_WCS/
                                              MAP_CELL_STEP/CELL_ID per band
    sources_pacs_region__<Region>.hdf5        the region's PACS sources
    photometry_pacs_source__<Region>.hdf5     per SESNA source and band: FNU, SIGMA, ORIGIN, ...

Round-1 skeleton (CONTRACT.md): signatures only. Depends on: grid, psf, mosaic, detect, inject, photometry (types only).
Replaces: pacs_completeness.write_product/write_sources/_read_band/_held_path/_read_held, pacs_photometry.write_region's
writer part/_DTYPES/_FILL/_NAMES. Paths: config.product_path(config, "sky/derived", "pacs", <quantity>, <granule>,
region) exactly as today. Datasets: written through build.write_dataset (UNITS, READING) with the same names, dtypes,
shapes, chunking and fill values as main 0683690, so the products are byte-identical except for the added CHECKS group.

CHECKS (new, FIX_SPEC.md sec 11): a group `CHECKS` in each product holding one row per check run on it: NAME (str),
BAND (str), VALUE (float64), BAR_LOW, BAR_HIGH (float64, NaN = none), VERDICT (str: PASS, FAIL, INFO), DETAIL (str).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from sesnaimpute.sky.derived.pacs import inject, photometry
from sesnaimpute.sky.derived.pacs.inject import BandCompleteness
from sesnaimpute.sky.derived.pacs.photometry import ForcedPhotometry, Sources

ROUND2 = "round 2"


@dataclass(frozen=True)
class CheckRow:
    """One check's result, as stored in a product's CHECKS group: name, band ("" for region-wide), value, bar_low,
    bar_high (NaN = none), verdict ("PASS" | "FAIL" | "INFO"), detail (one line, plain words)."""
    name: str
    band: str
    value: float
    bar_low: float
    bar_high: float
    verdict: str
    detail: str


def completeness_path(config, region: str) -> str:
    """The completeness product's path (product_path(config, "sky/derived", "pacs", "completeness", "source", region))."""
    raise NotImplementedError(ROUND2)


def sources_path(config, region: str) -> str:
    """The sources product's path (quantity "sources", granule "region")."""
    raise NotImplementedError(ROUND2)


def photometry_path(config, region: str) -> str:
    """The photometry product's path (quantity "photometry", granule "source")."""
    raise NotImplementedError(ROUND2)


def write_completeness(path: str, region: str, mag: np.ndarray, bands: Dict[str, BandCompleteness]) -> None:
    """Writes the completeness product. Identity: `pacs_completeness.write_product` (pacs_completeness.py:1758) --
    every dataset and attribute byte-identical (FROZEN 1/4: CELL_ID and row order)."""
    raise NotImplementedError(ROUND2)


def write_sources(path: str, region: str, bands: Dict[str, BandCompleteness], sources: Sources) -> None:
    """Writes the sources product. Identity: `pacs_completeness.write_sources` (pacs_completeness.py:1953)."""
    raise NotImplementedError(ROUND2)


def write_photometry(path: str, region: str, table: ForcedPhotometry) -> None:
    """Writes the photometry product (chunks (min(n, 65536),), fillvalue per column). Identity: `pacs_photometry.
    write_region`'s file plus `cross_match`'s in-place columns (pacs_photometry.py:322-370, 178-261), byte-identical."""
    raise NotImplementedError(ROUND2)


def write_checks(path: str, rows: Sequence[CheckRow]) -> None:
    """Appends `rows` to the product's CHECKS group (created on first use; UNITS/READING on each column)."""
    raise NotImplementedError(ROUND2)


def read_checks(path: str) -> List[CheckRow]:
    """The CHECKS rows of a product."""
    raise NotImplementedError(ROUND2)


def read_completeness_band(path: str, band: str) -> dict:
    """One band's datasets of the completeness product (`pacs_completeness._read_band`)."""
    raise NotImplementedError(ROUND2)
