"""The PACS build on a cluster or the laptop. On a cluster it plans each band's node count K and time limits from the
detection records and the measured rates, and EMITS the chain as script files (every job with its afterok dependency); it
submits nothing itself. On the laptop it runs the same job commands in turn.

    python -m sesnaimpute.sky.derived.pacs.driver --config C --system S --region R [--region R2 ...] [--box ...]
           [--out-dir D] [--target-minutes M] [--tasks P70=N,P100=N,P160=N]
        writes <out-dir>/pacs_phase1.sh (the small regions: every region whose largest band is < SMALL_MPX
        megapixels) and <out-dir>/pacs_phase2.sh (the wide ones), each holding every region of the phase in the step
        it is in, and prints their paths (out-dir default: the system's slurm_dir)
    python -m sesnaimpute.sky.derived.pacs.driver laptop --config C --region R [--region R2 ...] [--box ...] [--n-jobs N]
        the same job commands, run in turn on this machine (K = 1, local launcher), resumable like the chain
    python -m sesnaimpute.sky.derived.pacs.driver inject-band|pool-band --config C --region R --band B --of K
           [--box ...] [--launcher srun|local] [--cpus N]
        the bodies of the inject and pool steps (what the emitted jobs run)

Phases (owner 2026-10-31): phase 1 = the small regions, phase 2 = the wide ones, submitted by the owner after phase 1's
CHECKS pass. Inside a phase every region goes through two steps, because K needs the detection records and TACC refuses
`sbatch` from a compute node ("Job submission is not allowed from this host", smoke job 3661756), so no job can submit
the rest of the chain:
    step "detect": prepare (1 node: per band `stages mosaic` -- which converts the region's stored dense mosaic to the
                   tile store and verifies it bit for bit, or builds it -- `stages psf`, `stages detect`)
    step "rest":   (written once the region's detect rows are all done; K of each band from its detect record)
                   per band with maps one job on K nodes: the injection rounds (the K shares of a round as srun steps,
                   a barrier between rounds) then the pooling (K shares, then the assembly) -> finish (1 node, afterok
                   every band: the empty pooling record of a band without maps, `stages finish`, which writes and
                   verifies the products and their CHECKS, then `catalog.curated` for the region)
The owner runs the driver and `bash`es the phase file on a LOGIN node; after the detect jobs end the same driver command
rewrites the phase file with the rest step. A job whose rows are done does nothing, so a failed job is resubmitted by
running the driver again. The file submits region by region and waits (sleeping, on the login node) until the user's
queued jobs leave room under the system's limit for the next region's jobs.

Sizing (unchanged from the retired pacs_sizing): the work of a band = round-0 task count N0 x (ROUND0_S +
MAX_EXTRA_BATCHES x LATER_S) node-seconds at REF_CPUS cores (scaled by ref_cpus / cpus); K = max(1, ceil(work /
(target x 60))), at most min(nodes per job, (nodes per user - 3) // 3); -t = 3 x the expected minutes rounded up to 30
min, at least 30; an injection on K > 1 nodes gets at least INJECT_FLOOR (Cygnus X P70, K = 4: TIMEOUT at 2:00:10, then
40 min more). Pool minutes = (POOL_TABLES_MIN + POOL_SHARD_MIN / K) x max(1, work / P100_INJECT_S); a band's job limit is
the injection's plus the pooling's. Prepare DETECT_LIMIT; finish 3 x FINISH_MIN.

Replaces: pacs_sizing.py, pacs_driver.py (launchers, inject_band, pool_band; claim_finish removed),
scripts/submit_pacs_region.sh (phases, job bodies and environment: thread variables, malloc variables, ulimit, the
environment script, the numpy assert). The system facts (partition, CPUs, caps, account, software root, environment
script, slurm directory) are a `System` read from the config file's `[system.<name>]` section (defaults for the
non-path facts below), not hard-coded paths. Not carried from the retired preflight: its git-clean refusal (the stage
rows record the code commit instead; no currency gates).
"""

from __future__ import annotations

import argparse
import configparser
import dataclasses
import datetime
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sesnaimpute import config as config_module
from sesnaimpute.definitions import PACS_BANDS as BANDS
from sesnaimpute.definitions import PACS_MAX_EXTRA_BATCHES as MAX_EXTRA_BATCHES
from sesnaimpute.sky.derived.pacs import mosaic, stages
from sesnaimpute.sky.derived.pacs.grid import PixelGrid
from sesnaimpute.sky.download.herschel_pacs.index import read_index

#: the measured rates (pacs_sizing.py, Orion A on Stampede3 spr, 112 cores, jobs 3661406-3661441)
RATES = dict(ROUND0_S=1812.0 / 3873.0,                              # node-s per round-0 task
             LATER_S=4200.0 / (MAX_EXTRA_BATCHES * 3873.0),         # node-s per round-0 task and later round
             P100_INJECT_S=(9.6 + 12.3) * 60.0,                     # P100's measured injection node-s
             POOL_TABLES_MIN=2.7, POOL_SHARD_MIN=4.5,               # P100 pooling on one node
             PREPARE_MIN=40.0, FINISH_MIN=40.0)                     # one node each
REF_CPUS = 112
TARGET_MIN = 45.0
#: the prepare job (every band's mosaic, checks and detection, one node): Cygnus X took 10 + 36 + 4 + 7 min (jobs
#: 3662364-3662367), so one flat limit
DETECT_LIMIT = "04:00:00"
#: the least -t of an injection on more than one node (a wide region)
INJECT_FLOOR = "03:00:00"
#: a region is "small" (phase 1) when its largest band's grid holds fewer pixels than this, in millions (owner
#: 2026-10-31: IRAS 20050+2720, NGC 7129 and S140 -- 0.2, 3.7 and 7.8 Mpx; the next, L988, is 61 Mpx)
SMALL_MPX = 20.0
#: the cluster python's numpy (the job refuses another)
NUMPY_VERSION = "1.24.4"
#: seconds the phase file sleeps between looks at the queue while it waits for room
QUEUE_POLL_S = 120
#: per system, the non-path facts (TACC queue limits); the [system.<name>] section may override any of them
SYSTEM_DEFAULTS = {
    "stampede3": dict(partition="spr", cpus=112, max_nodes=64, max_queued=36, user_nodes=96, sbatch_extra=""),
    "vista": dict(partition="gg", cpus=144, max_nodes=32, max_queued=40, user_nodes=96, sbatch_extra="--propagate=NONE"),
}
#: the section keys every system must name (paths and account: never defaulted)
SYSTEM_REQUIRED = ("account", "software", "env_script", "slurm_dir")
STAGES_MODULE = "sesnaimpute.sky.derived.pacs.stages"
DRIVER_MODULE = "sesnaimpute.sky.derived.pacs.driver"
CURATED_MODULE = "sesnaimpute.catalog.curated"


@dataclass(frozen=True)
class System:
    """A cluster: name, partition, cpus per node, most nodes per job, most queued jobs per user, account, software root
    (PYTHONPATH), environment script, the measured per-task rates at `ref_cpus` (ROUND0_S, LATER_S, P100_INJECT_S,
    POOL_TABLES_MIN, POOL_SHARD_MIN, PREPARE_MIN, FINISH_MIN), target minutes per job, the slurm directory (job logs and
    the phase files), nodes per user, extra sbatch options, and the config file the jobs read."""
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
    slurm_dir: str = ""
    user_nodes: int = 96
    sbatch_extra: tuple = ()
    config_path: str = ""


@dataclass(frozen=True)
class BandPlan:
    """One band's plan: K nodes for its injection and pooling, the wall limits ("HH:MM:SS"), the round-0 task count."""
    band: str
    k: int
    inject_limit: str
    pool_limit: str
    detect_limit: str
    n_tasks: int


@dataclass(frozen=True)
class Job:
    """One job of the chain: name, stage (prepare, band, finish), band, nodes, wall limit, the command lines (run in
    turn; a failure fails the job; each a tuple of words), and the names of the jobs it waits for (afterok)."""
    name: str
    stage: str
    band: str
    nodes: int
    limit: str
    commands: tuple
    after: tuple


def load_system(config_path: str, name: str) -> System:
    """The cluster's facts from the config FILE's [system.<name>] section (the loaded Config keeps no such section),
    over SYSTEM_DEFAULTS for the non-path facts. Required keys: account, software, env_script, slurm_dir. Optional:
    partition, cpus, max_nodes, max_queued, user_nodes, sbatch_extra, ref_cpus, target_minutes, and any rate as
    rate_<NAME>."""
    parser = configparser.ConfigParser()
    if not parser.read(config_path):
        raise FileNotFoundError(config_path)
    section = "system." + name
    if section not in parser:
        raise ValueError("%s has no [%s] section (keys %s)" % (config_path, section, ", ".join(SYSTEM_REQUIRED)))
    s = dict(SYSTEM_DEFAULTS.get(name, {}))
    s.update(parser[section])
    missing = [k for k in SYSTEM_REQUIRED + ("partition", "cpus", "max_nodes", "max_queued", "user_nodes") if k not in s]
    if missing:
        raise ValueError("[%s] in %s lacks %s" % (section, config_path, ", ".join(missing)))
    rates = dict(RATES)
    for k, v in s.items():
        if k.startswith("rate_"):
            rates[k[5:].upper()] = float(v)
    return System(name, s["partition"], int(s["cpus"]), int(s["max_nodes"]), int(s["max_queued"]), s["account"],
                  s["software"], s["env_script"], int(s.get("ref_cpus", REF_CPUS)), rates,
                  float(s.get("target_minutes", TARGET_MIN)), s["slurm_dir"], int(s["user_nodes"]),
                  tuple(shlex.split(s.get("sbatch_extra", ""))), os.path.abspath(config_path))


def wall_limit(expected_min: float) -> str:
    """`-t HH:MM:SS`: 3 x the expected minutes rounded up to 30 min, at least 30."""
    m = max(1, math.ceil(3.0 * expected_min / 30.0)) * 30
    return "%02d:%02d:00" % (m // 60, m % 60)


def _minutes(limit: str) -> int:
    h, m, _s = (int(x) for x in limit.split(":"))
    return 60 * h + m


def _limit(minutes: int) -> str:
    return "%02d:%02d:00" % (minutes // 60, minutes % 60)


def node_cap(system: System) -> int:
    """The most nodes one band's job takes: the system's per-job cap and a third of the user's nodes (the three
    bands' jobs run together)."""
    return max(1, min(system.max_nodes, (system.user_nodes - len(BANDS)) // len(BANDS)))


def _plan(band: str, n0: int, system: System) -> BandPlan:
    r = system.rates
    scale = system.ref_cpus / system.cpus
    work = n0 * (r["ROUND0_S"] + MAX_EXTRA_BATCHES * r["LATER_S"]) * scale
    k = min(node_cap(system), max(1, math.ceil(work / (system.target_minutes * 60.0))))
    inject_limit = wall_limit(work / k / 60.0)
    if k > 1 and _minutes(inject_limit) < _minutes(INJECT_FLOOR):
        inject_limit = INJECT_FLOOR
    pool_min = (r["POOL_TABLES_MIN"] + r["POOL_SHARD_MIN"] / k) * max(1.0, work / (r["P100_INJECT_S"] * scale))
    return BandPlan(band, k, inject_limit, wall_limit(pool_min), DETECT_LIMIT, int(n0))


def plan_band(config, region: str, band: str, system: System, box: Optional[Sequence[float]] = None) -> BandPlan:
    """K and limits from the band's detect record (round-0 task count) and the system's rates (INJECT_FLOOR for
    K > 1)."""
    return _plan(band, stages._round_task_count(config, region, band, 0, box), system)


def largest_band_mpx(config, region: str, box: Optional[Sequence[float]] = None) -> float:
    """The pixel count, in millions, of the region's largest band grid (`PixelGrid.build` over the band's input set
    from the map index; no mosaic needed). 0 for a region without maps."""
    records = read_index(config, region)
    sizes = [0]
    for b in BANDS:
        kept, _dropped = mosaic.input_set(records, b)
        if kept:
            g = PixelGrid.build(kept, b, box)
            sizes.append(g.ny * g.nx)
    return max(sizes) / 1e6


def _tag(region: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", region)


def _words(python: str, module: str, verb: Optional[str], config_path: str, region: str, box,
           n_jobs: Optional[int], *extra) -> tuple:
    words = [python, "-W", "ignore", "-m", module] + ([verb] if verb else []) + ["--config", config_path, "--region", region]
    if box is not None:
        words += ["--box"] + ["%r" % float(v) for v in box]
    if n_jobs is not None:
        words += ["--n-jobs", str(n_jobs)]
    return tuple(words + [str(x) for x in extra])


def _detect_done(config, region, box) -> bool:
    return all(stages._row_state(config, region, box, stages._key("detect", b)).done for b in BANDS)


def _chain(config, config_path: str, region: str, system: Optional[System], box=None,
           tasks: Optional[Dict[str, int]] = None, laptop_jobs: Optional[int] = None) -> List[Job]:
    """The region's jobs for the step it is in (see `chain`). `laptop_jobs`: the laptop form (this interpreter, the
    local launcher, K = 1, that many workers)."""
    tag = _tag(region)
    laptop = laptop_jobs is not None
    python = sys.executable if laptop else "python"
    c = laptop_jobs if laptop else system.cpus
    w = lambda module, verb, *extra: _words(python, module, verb, config_path, region, box, c, *extra)
    if tasks is None and config is not None and stages._row_state(config, region, box, stages._FINISH).done:
        return []
    if tasks is None and (config is None or not _detect_done(config, region, box)):
        commands = tuple(w(STAGES_MODULE, s, "--band", b) for b in BANDS for s in ("mosaic", "psf", "detect"))
        return [Job("pacs_%s_prepare" % tag, "prepare", "", 1, DETECT_LIMIT, commands, ())]
    jobs = []
    mapped = {r.band for r in read_index(config, region)} if config is not None else set(BANDS)
    for b in BANDS:
        if b not in mapped:             # no map of the band: its empty pooling record is written by the finish job
            continue
        if laptop:
            k, limit = 1, ""
        else:
            p = _plan(b, tasks[b], system) if tasks is not None else plan_band(config, region, b, system, box)
            k, limit = p.k, _limit(_minutes(p.inject_limit) + _minutes(p.pool_limit))
        launch = ("--launcher", "local") if laptop else ("--launcher", "srun", "--cpus", c)
        jobs.append(Job("pacs_%s_%s" % (tag, b), "band", b, k, limit,
                        (w(DRIVER_MODULE, "inject-band", "--band", b, "--of", k, *launch),
                         w(DRIVER_MODULE, "pool-band", "--band", b, "--of", k, *launch)), ()))
    curated = (python, "-W", "ignore", "-m", CURATED_MODULE, config_path, "--regions", region)
    empty = tuple(w(STAGES_MODULE, "pool", "--band", b) for b in BANDS if b not in mapped)
    jobs.append(Job("pacs_%s_finish" % tag, "finish", "", 1,
                    "" if laptop else wall_limit(system.rates["FINISH_MIN"]),
                    empty + (w(STAGES_MODULE, "finish"), curated), tuple(j.name for j in jobs)))
    return jobs


def chain(config, config_path: str, region: str, system: System, box: Optional[Sequence[float]] = None) -> List[Job]:
    """The region's jobs for the step it is in: while any band's detect row is not done, step "detect" (one prepare
    job: every band's mosaic, psf and detect stages on 1 node); once all are done, step "rest" (per band one job on K
    nodes: the injection rounds then the pooling; finish afterok every band: the products, their CHECKS, then
    catalog.curated for the region), K from the detect records; [] once finish is done."""
    return _chain(config, config_path, region, system, box)


def _step(jobs: Sequence[Job]) -> str:
    return "rest" if any(j.stage == "band" for j in jobs) else "detect"


def _code() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run(("git", "-C", here, "rev-parse", "--short", "HEAD"), capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def emit(regions: Dict[str, List[Job]], system: System, phase: str = "") -> str:
    """The submission script for the regions' jobs ({region: jobs}, each region in the step it is in): sbatch lines
    with --dependency=afterok, region by region, waiting for room under the system's queue limit before each region;
    the job bodies with the environment of the retired submit_pacs_region.sh. Run with bash on a login node (`--dry-run`
    prints the sbatch lines); sbatch runs the same file as `<file> --job <name>`."""
    q = shlex.quote
    jobs = [j for js in regions.values() for j in js]
    out = ["#!/bin/bash",
           "# PACS chain %s on %s (%s, %d cores per node), emitted %s by %s (code %s)."
           % (phase, system.name, system.partition, system.cpus, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
              DRIVER_MODULE, _code()),
           "# Run on a LOGIN node (a compute node cannot submit, job 3661756):  bash THIS_FILE [--dry-run]",
           "# sbatch runs this same file as the body of each job:  THIS_FILE --job NAME",
           "# Regions (step) and jobs (name, nodes, -t, afterok):"]
    for region, js in regions.items():
        out.append("#   %s (step %s)" % (region, _step(js)))
        out += ["#     %-34s %3d  %s  %s" % (j.name, j.nodes, j.limit, ",".join(j.after) or "-") for j in js]
    if any(_step(js) == "detect" for js in regions.values()):
        out.append("# Next: once the prepare jobs have ended, run the driver again (same command): it writes the rest step.")
    out += ["", "SELF=$(cd \"$(dirname \"$0\")\" && pwd)/$(basename \"$0\")", "SLURM_DIR=%s" % q(system.slurm_dir),
            "SOFTWARE=%s" % q(system.software), "",
            "if [ \"${1:-}\" = \"--job\" ]; then",
            "    JOB=$2",
            "    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
            "    # glibc: a fixed mmap threshold so the workers' large temporaries go back to the system (job 3661209)",
            "    export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2",
            "    ulimit -u $(ulimit -Hu)",
            "    source %s" % q(system.env_script),
            "    export PYTHONPATH=$SOFTWARE/src",
            "    cd \"$SLURM_DIR\"",
            "    python -c \"import numpy, sys; sys.exit(0 if numpy.__version__ == '%s' else 1)\" || { echo \"numpy is not %s\"; exit 1; }"
            % (NUMPY_VERSION, NUMPY_VERSION),
            "    python -W ignore -c \"import sesnaimpute; print('code:', sesnaimpute.__file__)\"",
            "    ( cd \"$SOFTWARE\" && git log --oneline -1 )",
            "    echo \"job $JOB on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST\"",
            "    case \"$JOB\" in"]
    for j in jobs:
        out.append("        %s)" % q(j.name))
        out += ["            %s || exit 1" % " ".join(q(x) for x in c) for c in j.commands]
        out.append("            ;;")
    out += ["        *) echo \"unknown job $JOB\" >&2; exit 2 ;;", "    esac", "    exit 0", "fi", "",
            "DRY=0; [ \"${1:-}\" = \"--dry-run\" ] && DRY=1",
            "[ $DRY -eq 1 ] || mkdir -p \"$SLURM_DIR\"",
            "# room N: wait until N more jobs fit under the queue limit (%d on %s)" % (system.max_queued, system.name),
            "room() {",
            "    [ $DRY -eq 1 ] && return 0",
            "    while [ $(( $(squeue -u \"$USER\" -h 2>/dev/null | wc -l) + $1 )) -gt %d ]; do" % system.max_queued,
            "        echo \"waiting for room for $1 job(s) under %d queued ($(date +%%H:%%M))\"; sleep %d" % (system.max_queued, QUEUE_POLL_S),
            "    done",
            "}",
            "# submit NAME NODES TIME DEPENDENCY-IDS(colon-joined, or empty) -> prints the job id",
            "submit() {",
            "    local name=$1 nodes=$2 time=$3 dep=$4",
            "    local cmd=(sbatch --parsable -J \"$name\" -o \"$SLURM_DIR/%%x_%%j.out\" -e \"$SLURM_DIR/%%x_%%j.err\" -p %s -N $nodes -n $nodes -c %d"
            % (q(system.partition), system.cpus),
            "               -t $time -A %s%s)" % (q(system.account), "".join(" " + q(x) for x in system.sbatch_extra)),
            "    [ -z \"$dep\" ] || cmd+=(--dependency=\"afterok:$dep\")",
            "    cmd+=(\"$SELF\" --job \"$name\")",
            "    if [ $DRY -eq 1 ]; then",
            "        { printf '%q ' \"${cmd[@]}\"; echo; } >&2",
            "        echo \"$name\"",
            "    else",
            "        local id",
            "        id=$(\"${cmd[@]}\" | cut -d';' -f1 | grep -E '^[0-9]+$' | tail -1)",
            "        [ -n \"$id\" ] || { echo \"submission of $name failed: stopping (cancel this region's earlier jobs: squeue -u $USER)\" >&2; kill -TERM $$; exit 1; }",
            "        echo \"$id\"",
            "    fi",
            "}"]
    var = {j.name: "ID_%d" % i for i, j in enumerate(jobs)}
    for region, js in regions.items():
        out.append("room %d   # %s" % (len(js), region))
        for j in js:
            dep = ":".join("${%s}" % var[a] for a in j.after)
            out.append("%s=$(submit %s %d %s \"%s\")" % (var[j.name], q(j.name), j.nodes, j.limit, dep))
            out.append("echo \"%s %s nodes=%d -t %s ID=${%s}\"" % (j.stage.upper(), j.name, j.nodes, j.limit, var[j.name]))
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------------------------------
# the job bodies of inject and pool: K shares per round on a node each

class LocalLauncher:
    """The shares as subprocesses of this machine, run together."""
    def wrap(self, cmd):
        return cmd

    def run(self, cmds):
        procs = [subprocess.Popen(self.wrap(c)) for c in cmds]
        return [p.wait() for p in procs]


class SrunLauncher(LocalLauncher):
    """The shares as steps of the batch allocation, one node each, the node's cores to the share alone."""
    def __init__(self, cpus):
        self.cpus = cpus

    def wrap(self, cmd):
        return ["srun", "--nodes=1", "--ntasks=1", "--exclusive", "--cpus-per-task=%d" % self.cpus, "--export=ALL"] + cmd


def _share(config_path, region, box, n_jobs, *extra) -> list:
    return list(_words(sys.executable, STAGES_MODULE, None, config_path, region, box, n_jobs, *extra))


def _run_shares(launch, cmds, what, shards):
    codes = launch.run(cmds)
    failed = [i for i, c in enumerate(codes) if c != 0]
    if failed:
        raise RuntimeError("%s: shares %s failed (exit codes %s)" % (what, [shards[i] for i in failed], [codes[i] for i in failed]))


def inject_band(config, config_path, region, band, k, box, launch, n_jobs=None) -> None:
    """INJECT of a band over `k` nodes: rounds 0 .. MAX_EXTRA_BATCHES, each round's shards not done started together,
    stopping at the first round without tasks."""
    for r in range(MAX_EXTRA_BATCHES + 1):
        n = stages._round_task_count(config, region, band, r, box)
        if not n:
            print("inject-band %s %s: round %d has no tasks, the injections are complete" % (region, band, r), flush=True)
            return
        stages._refuse_mixed_k(config, region, box, band, "inject", r, k)
        todo = [s for s in range(k) if not stages._row_state(config, region, box, stages._key("inject", band, r, s, k)).done]
        print("inject-band %s %s: round %d, %d tasks, %d of %d shards to run" % (region, band, r, n, len(todo), k), flush=True)
        _run_shares(launch, [_share(config_path, region, box, n_jobs, "inject", "--band", band, "--round", r, "--shard", s,
                                    "--of", k) for s in todo], "inject %s round %d" % (band, r), todo)


def pool_band(config, config_path, region, band, k, box, launch, n_jobs=None) -> None:
    """POOL of a band over `k` nodes: the shards not done started together, then the assembly (one process)."""
    stages._refuse_mixed_k(config, region, box, band, "pool", -1, k)
    todo = [s for s in range(k) if not stages._row_state(config, region, box, stages._key("pool", band, None, s, k)).done]
    print("pool-band %s %s: %d of %d shards to run" % (region, band, len(todo), k), flush=True)
    _run_shares(launch, [_share(config_path, region, box, n_jobs, "pool", "--band", band, "--shard", s, "--of", k)
                         for s in todo], "pool %s" % band, todo)
    _run_shares(LocalLauncher(), [_share(config_path, region, box, n_jobs, "pool", "--band", band, "--of", k)],
                "pool %s assembly" % band, ["assembly"])


def laptop(config, config_path: str, regions: Sequence[str], box=None, n_jobs: Optional[int] = None) -> None:
    """The chain on this machine: for each region in turn, the commands of its jobs (prepare, then each band's
    injection and pooling, then finish and catalog.curated) run one after another (K = 1, local launcher). A command
    that fails stops the run; a rerun resumes from the stage rows."""
    n_jobs = n_jobs or config.n_jobs
    for region in regions:
        for _ in range(2):                          # the detect step, then the rest
            jobs = _chain(config, config_path, region, None, box, laptop_jobs=n_jobs)
            for j in jobs:
                for c in j.commands:
                    print("driver laptop %s: %s" % (j.name, " ".join(shlex.quote(x) for x in c)), flush=True)
                    if subprocess.run(list(c)).returncode:
                        raise RuntimeError("driver laptop: %s failed (%s)" % (j.name, " ".join(c[:6])))
            if any(j.stage == "finish" for j in jobs):
                break


# ----------------------------------------------------------------------------------------------------------------------
# the command line

def _tasks_arg(text: str) -> Dict[str, int]:
    out = {}
    for kv in text.split(","):
        b, n = kv.split("=")
        if b not in BANDS:
            raise argparse.ArgumentTypeError("unknown band %s" % b)
        out[b] = int(n)
    if set(out) != set(BANDS):
        raise argparse.ArgumentTypeError("--tasks needs every band of %s" % (BANDS,))
    return out


def _box(v):
    return None if v is None else tuple(v)


def _band_main(verb, argv) -> int:
    p = argparse.ArgumentParser(prog="python -m %s %s" % (DRIVER_MODULE, verb))
    p.add_argument("--config", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--band", required=True, choices=BANDS)
    p.add_argument("--of", type=int, required=True)
    p.add_argument("--box", type=float, nargs=4, default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--launcher", choices=("srun", "local"), default="local")
    p.add_argument("--cpus", type=int, default=1)
    a = p.parse_args(argv)
    config = config_module.load(a.config)
    launch = SrunLauncher(a.cpus) if a.launcher == "srun" else LocalLauncher()
    try:
        (inject_band if verb == "inject-band" else pool_band)(config, a.config, a.region, a.band, a.of, _box(a.box),
                                                              launch, a.n_jobs)
    except Exception as e:          # noqa: BLE001 -- the job's exit code
        print("%s %s %s: FAILED: %s" % (verb, a.region, a.band, e), file=sys.stderr, flush=True)
        return 1
    return 0


def _laptop_main(argv) -> int:
    p = argparse.ArgumentParser(prog="python -m %s laptop" % DRIVER_MODULE)
    p.add_argument("--config", required=True)
    p.add_argument("--region", action="append", required=True)
    p.add_argument("--box", type=float, nargs=4, default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    a = p.parse_args(argv)
    try:
        laptop(config_module.load(a.config), os.path.abspath(a.config), a.region, _box(a.box), a.n_jobs)
    except Exception as e:          # noqa: BLE001
        print("driver laptop: FAILED: %s" % e, file=sys.stderr, flush=True)
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python -m sesnaimpute.sky.derived.pacs.driver --config C --system S --region R ...`: writes the phase files
    (pacs_phase1.sh: the small regions; pacs_phase2.sh: the wide ones), each region in the step it is in; submits
    nothing. `laptop`: the chain on this machine. `inject-band` / `pool-band`: the job bodies."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("inject-band", "pool-band"):
        return _band_main(argv[0], argv[1:])
    if argv and argv[0] == "laptop":
        return _laptop_main(argv[1:])
    p = argparse.ArgumentParser(prog="python -m " + DRIVER_MODULE, description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--region", action="append", required=True, help="repeat for several regions")
    p.add_argument("--box", type=float, nargs=4, default=None)
    p.add_argument("--out-dir", default=None, help="where the phase files go (default: the system's slurm_dir)")
    p.add_argument("--target-minutes", type=float, default=None)
    p.add_argument("--tasks", type=_tasks_arg, default=None,
                   help="P70=N,P100=N,P160=N: round-0 task counts instead of the detect records (writes the rest step; a "
                        "dry run of the sizing)")
    a = p.parse_args(argv)
    system = load_system(a.config, a.system)
    if a.target_minutes is not None:
        system = dataclasses.replace(system, target_minutes=a.target_minutes)
    config = config_module.load(a.config)
    box = _box(a.box)
    out_dir = a.out_dir or system.slurm_dir
    os.makedirs(out_dir, exist_ok=True)
    phases: Dict[str, Dict[str, List[Job]]] = {"phase1": {}, "phase2": {}}
    for region in a.region:
        mpx = largest_band_mpx(config, region, box)
        phase = "phase1" if mpx < SMALL_MPX else "phase2"
        jobs = _chain(config, system.config_path, region, system, box, a.tasks)
        if not jobs:
            print("%s: finish is done; nothing to submit" % region)
            continue
        phases[phase][region] = jobs
        for j in jobs:
            print("%s %-7s %-8s %-34s nodes %3d  -t %s  afterok %s" % (phase, _step(jobs), j.stage, j.name, j.nodes,
                                                                        j.limit, ",".join(j.after) or "-"))
    for phase, regions in phases.items():
        if not regions:
            continue
        path = os.path.join(out_dir, "pacs_%s.sh" % phase)
        with open(path, "w") as f:
            f.write(emit(regions, system, phase))
        os.chmod(path, 0o755)
        print("wrote %s (%d regions; run it with bash on a login node; --dry-run to see the sbatch lines)"
              % (path, len(regions)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
