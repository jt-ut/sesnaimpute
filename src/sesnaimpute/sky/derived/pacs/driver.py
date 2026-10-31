"""The PACS build on a cluster: plans each band's node count K and time limits from the detection records and the
measured rates, and EMITS the region's whole slurm chain (one script, every job with its afterok dependency); nothing
is run by hand between phases. Laptop builds use `stages run` instead.

Round-1 skeleton (CONTRACT.md): signatures only. Depends on: stages (and through it every module). Replaces:
pacs_sizing.py (rates, node_cap, plan_band, wall_limit), pacs_driver.py (launchers, share_cmd, _run_shares,
inject_band, pool_band; claim_finish removed), scripts/submit_pacs_region.sh's phase logic (the job bodies and the
environment: thread variables, malloc variables, numpy assert stay in the emitted script). The system facts (partitions,
CPUs, account, roots) are a `System` value read from the config, not hard-coded paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sesnaimpute.sky.derived.pacs import stages

ROUND2 = "round 2"


@dataclass(frozen=True)
class System:
    """A cluster: name, partition, cpus per node, most nodes per job, most queued jobs per user, account, software root
    (PYTHONPATH), environment script, the measured per-task rates at `ref_cpus` (pacs_sizing's ROUND0_S, LATER_S,
    P100_INJECT_S, POOL_TABLES_MIN, POOL_SHARD_MIN, PREPARE_MIN, FINISH_MIN), target minutes per job."""
    name: str
    partition: str
    cpus: int
    max_nodes: int
    max_queued: int
    account: str
    software: str
    env_script: str
    ref_cpus: int
    rates: Dict[str, float]
    target_minutes: float


@dataclass(frozen=True)
class BandPlan:
    """One band's plan: K nodes for inject and pool, the wall limits of each job ("HH:MM:SS"), the round-0 task count."""
    band: str
    k: int
    inject_limit: str
    pool_limit: str
    detect_limit: str
    n_tasks: int


@dataclass(frozen=True)
class Job:
    """One job of the chain: name, stage, band, nodes, wall limit, the command lines (one srun step per shard), and
    the names of the jobs it waits for (afterok)."""
    name: str
    stage: str
    band: str
    nodes: int
    limit: str
    commands: tuple
    after: tuple


def load_system(config, name: str) -> System:
    """The cluster's facts from the config (a [system.<name>] section)."""
    raise NotImplementedError(ROUND2)


def plan_band(config, region: str, band: str, system: System, box: Optional[Sequence[float]] = None) -> BandPlan:
    """K and limits from the band's detect record (round-0 task count) and the system's rates
    (`pacs_sizing.plan_band` + node_cap + wall_limit)."""
    raise NotImplementedError(ROUND2)


def chain(config, config_path: str, region: str, system: System, box: Optional[Sequence[float]] = None) -> List[Job]:
    """The region's jobs in order: per band mosaic -> psf -> detect -> inject (K nodes, rounds in sequence inside the
    job) -> pool (K nodes) ; finish after the three bands' pools. Inject/pool sizing needs the detect record, so the
    chain's last job before inject is a `plan` step that writes the inject/pool jobs and submits them (the only
    self-submission; everything else is afterok)."""
    raise NotImplementedError(ROUND2)


def emit(jobs: Sequence[Job], system: System) -> str:
    """The submission script text for `jobs` (sbatch lines with --dependency=afterok, the job bodies with the
    environment of scripts/submit_pacs_region.sh)."""
    raise NotImplementedError(ROUND2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python -m sesnaimpute.sky.derived.pacs.driver --config C --region R --system S [--box ...] [--out FILE]`:
    writes the chain script; submits nothing."""
    raise NotImplementedError(ROUND2)


if __name__ == "__main__":
    raise SystemExit(main())
