"""The PACS build's stages and its ONE command line:

    python -m sesnaimpute.sky.derived.pacs.stages <mosaic|psf|detect|inject|pool|finish> --config C --region R
           [--band B] [--box RA DEC W H] [--round r --shard k --of K] [--n-jobs N]
    python -m sesnaimpute.sky.derived.pacs.stages run --config C --region R [--box ...]     (all stages, one process)

Stages, per region (per band where marked):
    mosaic  (band)  mosaic.build_mosaic without the PSF                       -> mosaic store
    psf     (band)  psf.build_mosaic_psf; checks zero_image + normalisation   -> psf/, CHECKS rows kept in the work dir
    detect  (band)  detect.detect_band                                        -> detection layers, detect.hdf5
    inject  (band, round, shard of K)  inject.round_tasks[shard::K] -> inject.inject_task
                                                                              -> inject_r<r>_s<k>of<K>.hdf5
    pool    (band, shard of K)  inject.pool_tasks[shard::K] -> inject.pool_block; the last shard's assembly
                                                                              -> pool_s<k>of<K>.hdf5, band.hdf5
    finish          photometry.merge_sources, forced_photometry, cross_match; products.write_*; checks.region_checks
                    -> products.write_checks; the work directory cleared

Resumable from the STAGE TABLE: a stage (or shard) is done when its record file exists and matches its receipt
(`<file>.receipt.json`: size and sha256 written after the atomic rename -- lost-write detection); K is recorded in every
shard record and a rerun with another K is refused. No pickles; no prepared copies; the work directory is
`<data_root>/sky/derived/pacs/work/<Region_safe>[__slice_...]/`.

Round-1 skeleton (CONTRACT.md): signatures only. Depends on: every module before it. Replaces: pacs_stages.py,
pacs_stages_cli.py, pacs_completeness.build/preflight/finish_region/_region_inputs/_git_state, pacs_photometry.build,
and the RUNBOOK entries (`pacs_completeness`, `pacs_photometry` lines -> one `stages run` line). Removed: run_region's
separate path, the preflight duplicate of prepare, claim-finish, the plan CLI (driver.py plans).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from sesnaimpute.sky.derived.pacs import checks, detect, grid, inject, mosaic, photometry, products, psf

ROUND2 = "round 2"

#: the stages, in order
STAGES = ("mosaic", "psf", "detect", "inject", "pool", "finish")
#: the stages run per band
BAND_STAGES = ("mosaic", "psf", "detect", "inject", "pool")


@dataclass(frozen=True)
class Receipt:
    """What a finished record file must be: `path`, `size` (bytes), `sha256`; written as `<path>.receipt.json` after the
    file's atomic rename."""
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class StageRow:
    """One row of the stage table: stage, band ("" for region-wide), round (-1 if none), shard, of (K), path, done
    (record and receipt agree), problem ("" or what is wrong: missing, size, checksum, mixed K)."""
    stage: str
    band: str
    round: int
    shard: int
    of: int
    path: str
    done: bool
    problem: str


def region_inputs(config, region: str) -> tuple:
    """(ra, dec) float64 of the region's SESNA sources in curated order (`catalog.curated.read_positions`)."""
    raise NotImplementedError(ROUND2)


def work_dir(config, region: str, box: Optional[Sequence[float]] = None, band: Optional[str] = None) -> str:
    """The region's (band's) work directory: <data_root>/sky/derived/pacs/work/<safe>[__slice_...][/<band>]
    (realpath inside <data_root>/sky/derived)."""
    raise NotImplementedError(ROUND2)


def write_record(path: str, tables: Dict[str, np.ndarray], attrs: Optional[dict] = None) -> Receipt:
    """Writes an HDF5 record file atomically (temporary name with pid, fsync, os.replace), then its receipt (size,
    sha256; also atomic). Returns the receipt."""
    raise NotImplementedError(ROUND2)


def read_record(path: str) -> Dict[str, np.ndarray]:
    """Reads a record file after `verify`; raises when the record does not match its receipt."""
    raise NotImplementedError(ROUND2)


def verify(path: str) -> str:
    """"" when the record exists and matches its receipt, else the problem (missing, no receipt, size, checksum)."""
    raise NotImplementedError(ROUND2)


def stage_table(config, region: str, box: Optional[Sequence[float]] = None) -> List[StageRow]:
    """Every stage (and shard) of the region with its state, from the work directory's records and receipts."""
    raise NotImplementedError(ROUND2)


def run_stage(config, stage: str, region: str, band: Optional[str] = None, box: Optional[Sequence[float]] = None,
              round_: Optional[int] = None, shard: int = 0, of: int = 1, n_jobs: Optional[int] = None) -> StageRow:
    """Runs one stage (one shard) unless its stage row is done; returns its row. Seeds and records do not depend on
    (shard, of, n_jobs) (FROZEN 2-3)."""
    raise NotImplementedError(ROUND2)


def run_region(config, region: str, box: Optional[Sequence[float]] = None, n_jobs: Optional[int] = None) -> None:
    """Every stage of the region in one process (K = 1): what the RUNBOOK line and a laptop build run. Identity: the
    products equal those of main 0683690's `pacs_completeness.build` + `finish_region` (except the added CHECKS)."""
    raise NotImplementedError(ROUND2)


def build(config, regions: Optional[Sequence[str]] = None, box: Optional[Sequence[float]] = None) -> None:
    """The RUNBOOK entry (`build.run(build)`): `run_region` for each region in turn (regions sequential)."""
    raise NotImplementedError(ROUND2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The command line of the module docstring. Exit 0 done (or already done), 1 failure, 2 usage."""
    raise NotImplementedError(ROUND2)


if __name__ == "__main__":
    raise SystemExit(main())
