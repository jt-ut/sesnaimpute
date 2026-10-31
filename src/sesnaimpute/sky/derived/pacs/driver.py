"""The PACS build on a cluster: plans each band's node count K and time limits from the detection records and the
measured rates, and EMITS the region's slurm chain as a script file (every job with its afterok dependency); it submits
nothing itself. Laptop builds use `stages all` instead.

    python -m sesnaimpute.sky.derived.pacs.driver --config C --system S --region R [--region R2 ...] [--box ...]
           [--out-dir D] [--target-minutes M] [--tasks P70=N,P100=N,P160=N]
        writes <out-dir>/pacs_<TAG>_<phase>.sh per region (out-dir default: the system's slurm_dir) and prints its path
    python -m sesnaimpute.sky.derived.pacs.driver inject-band|pool-band --config C --region R --band B --of K
           [--box ...] [--launcher srun|local] [--cpus N]
        the bodies of the inject and pool jobs (what the emitted script's jobs run)

Two phases, because K needs the detection records and TACC refuses `sbatch` from a compute node ("Job submission is not
allowed from this host", smoke job 3661756), so no job can submit the rest of the chain:
    phase "detect": prepare (1 node: `stages mosaic` + `stages psf` of every band) -> detect per band (1 node each,
                    afterok prepare)
    phase "rest":   (written once every band's detect row is done, the K of each band read from its detect record)
                    inject per band (K nodes; rounds in sequence inside the job, the K shares of a round as srun steps,
                    a barrier between rounds) -> pool per band (K nodes, afterok its inject; the K shares, then the
                    assembly) -> finish (1 node, afterok every pool)
The owner runs the driver and `bash`es the script it writes on a LOGIN node, twice per region (after the detect jobs
end, the same driver command writes the "rest" script). A job whose rows are done does nothing, so a failed job is
resubmitted by running the driver again (it writes the phase the region is in).

Sizing (pacs_sizing's, unchanged): the work of a band = round-0 task count N0 x (ROUND0_S + MAX_EXTRA_BATCHES x LATER_S)
node-seconds at REF_CPUS cores (scaled by ref_cpus / cpus); K = max(1, ceil(work / (target x 60))), at most
min(nodes per job, (nodes per user - 3) // 3); -t = 3 x the expected minutes rounded up to 30 min, at least 30; an
inject job on K > 1 nodes (a wide region) gets at least INJECT_FLOOR (Cygnus X P70, K = 4: TIMEOUT at 2:00:10, then 40
min more). Pool minutes = (POOL_TABLES_MIN + POOL_SHARD_MIN / K) x max(1, work / P100_INJECT_S) (kept: conservative now
that no tables step exists). Detect -t DETECT_LIMIT; prepare and finish 3 x 40 min.

Replaces: pacs_sizing.py (rates, node_cap, plan_band, wall_limit), pacs_driver.py (launchers, share_cmd, _run_shares,
inject_band, pool_band; claim_finish removed), scripts/submit_pacs_region.sh (phases, job bodies and environment:
thread variables, malloc variables, ulimit, the environment script, the numpy assert). The system facts (partition,
CPUs, caps, account, software root, environment script, slurm directory) are a `System` read from the config file's
`[system.<name>]` section (defaults for the non-path facts below), not hard-coded paths.
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
from sesnaimpute.sky.derived.pacs import stages

#: the measured rates (pacs_sizing.py, Orion A on Stampede3 spr, 112 cores, jobs 3661406-3661441)
RATES = dict(ROUND0_S=1812.0 / 3873.0,                              # node-s per round-0 task
             LATER_S=4200.0 / (MAX_EXTRA_BATCHES * 3873.0),         # node-s per round-0 task and later round
             P100_INJECT_S=(9.6 + 12.3) * 60.0,                     # P100's measured injection node-s
             POOL_TABLES_MIN=2.7, POOL_SHARD_MIN=4.5,               # P100 pooling on one node
             PREPARE_MIN=40.0, FINISH_MIN=40.0)                     # one node each
REF_CPUS = 112
TARGET_MIN = 45.0
#: a detect job writes its record only at its end: one flat limit (about 11 x Orion A's P70)
DETECT_LIMIT = "04:00:00"
#: the least -t of an inject job on more than one node (a wide region)
INJECT_FLOOR = "03:00:00"
#: the cluster python's numpy (the job refuses another)
NUMPY_VERSION = "1.24.4"
#: per system, the non-path facts (TACC queue limits); the [system.<name>] section may override any of them
SYSTEM_DEFAULTS = {
    "stampede3": dict(partition="spr", cpus=112, max_nodes=64, max_queued=36, user_nodes=96, sbatch_extra=""),
    "vista": dict(partition="gg", cpus=144, max_nodes=32, max_queued=40, user_nodes=96, sbatch_extra="--propagate=NONE"),
}
#: the section keys every system must name (paths and account: never defaulted)
SYSTEM_REQUIRED = ("account", "software", "env_script", "slurm_dir")
STAGES_MODULE = "sesnaimpute.sky.derived.pacs.stages"
DRIVER_MODULE = "sesnaimpute.sky.derived.pacs.driver"


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
    """One job of the chain: name, stage, band, nodes, wall limit, the command lines (run in turn; a failure fails the
    job), and the names of the jobs it waits for (afterok)."""
    name: str
    stage: str
    band: str
    nodes: int
    limit: str
    commands: tuple
    after: tuple


@dataclass(frozen=True)
class _Site:
    """The system facts `System` does not hold (contract change asked for): slurm directory, nodes per user, extra
    sbatch options, the config file the jobs read."""
    slurm_dir: str
    user_nodes: int
    sbatch_extra: tuple
    config_path: str


_SITES: Dict[str, _Site] = {}


def load_system(config, name: str) -> System:
    """The cluster's facts from the config FILE's [system.<name>] section (`config` is the config path: the loaded
    Config keeps no such section), over SYSTEM_DEFAULTS for the non-path facts. Required keys: account, software,
    env_script, slurm_dir. Optional: partition, cpus, max_nodes, max_queued, user_nodes, sbatch_extra, ref_cpus,
    target_minutes, and any rate as rate_<NAME>."""
    if not isinstance(config, str):
        raise TypeError("load_system needs the config file's path (the loaded Config keeps no [system.*] section)")
    parser = configparser.ConfigParser()
    if not parser.read(config):
        raise FileNotFoundError(config)
    section = "system." + name
    if section not in parser:
        raise ValueError("%s has no [%s] section (keys %s)" % (config, section, ", ".join(SYSTEM_REQUIRED)))
    s = dict(SYSTEM_DEFAULTS.get(name, {}))
    s.update(parser[section])
    missing = [k for k in SYSTEM_REQUIRED + ("partition", "cpus", "max_nodes", "max_queued", "user_nodes") if k not in s]
    if missing:
        raise ValueError("[%s] in %s lacks %s" % (section, config, ", ".join(missing)))
    rates = dict(RATES)
    for k, v in s.items():
        if k.startswith("rate_"):
            rates[k[5:].upper()] = float(v)
    _SITES[name] = _Site(s["slurm_dir"], int(s["user_nodes"]), tuple(shlex.split(s.get("sbatch_extra", ""))),
                         os.path.abspath(config))
    return System(name, s["partition"], int(s["cpus"]), int(s["max_nodes"]), int(s["max_queued"]), s["account"],
                  s["software"], s["env_script"], int(s.get("ref_cpus", REF_CPUS)), rates,
                  float(s.get("target_minutes", TARGET_MIN)))


def wall_limit(expected_min: float) -> str:
    """`-t HH:MM:SS`: 3 x the expected minutes rounded up to 30 min, at least 30 (pacs_sizing.wall_limit)."""
    m = max(1, math.ceil(3.0 * expected_min / 30.0)) * 30
    return "%02d:%02d:00" % (m // 60, m % 60)


def _minutes(limit: str) -> int:
    h, m, _s = (int(x) for x in limit.split(":"))
    return 60 * h + m


def node_cap(system: System) -> int:
    """The most nodes one band's job takes: the system's per-job cap and a third of the user's nodes (the three
    bands' jobs run together)."""
    user = _SITES[system.name].user_nodes if system.name in _SITES else 96
    return max(1, min(system.max_nodes, (user - len(BANDS)) // len(BANDS)))


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
    """K and limits from the band's detect record (round-0 task count) and the system's rates
    (`pacs_sizing.plan_band` + node_cap + wall_limit; INJECT_FLOOR for K > 1)."""
    return _plan(band, stages._round_task_count(config, region, band, 0, box), system)


def _tag(region: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", region)


def _cmd(module: str, verb: str, config_path: str, region: str, box, n_jobs: Optional[int], *extra) -> str:
    words = ["python", "-W", "ignore", "-m", module, verb, "--config", config_path, "--region", region]
    if box is not None:
        words += ["--box"] + ["%g" % v for v in box]
    if n_jobs is not None:
        words += ["--n-jobs", str(n_jobs)]
    return " ".join(shlex.quote(w) for w in words + [str(x) for x in extra])


def _detect_done(config, region, box) -> bool:
    return all(stages._row_state(config, region, box, stages._key("detect", b)).done for b in BANDS)


def _chain(config, config_path: str, region: str, system: System, box=None, tasks: Optional[Dict[str, int]] = None
           ) -> List[Job]:
    tag, c = _tag(region), system.cpus
    if tasks is None and config is not None and stages._row_state(config, region, box, stages._FINISH).done:
        return []
    if tasks is None and (config is None or not _detect_done(config, region, box)):
        prepare = Job("pacs_%s_prepare" % tag, "prepare", "", 1, wall_limit(system.rates["PREPARE_MIN"]),
                      tuple(_cmd(STAGES_MODULE, s, config_path, region, box, c, "--band", b) for b in BANDS
                            for s in ("mosaic", "psf")), ())
        return [prepare] + [Job("pacs_%s_%s_detect" % (tag, b), "detect", b, 1, DETECT_LIMIT,
                                (_cmd(STAGES_MODULE, "detect", config_path, region, box, c, "--band", b),), (prepare.name,))
                            for b in BANDS]
    jobs, pools = [], []
    for b in BANDS:
        p = _plan(b, tasks[b], system) if tasks is not None else plan_band(config, region, b, system, box)
        inj = Job("pacs_%s_%s_inject" % (tag, b), "inject", b, p.k, p.inject_limit,
                  (_cmd(DRIVER_MODULE, "inject-band", config_path, region, box, c, "--band", b, "--of", p.k,
                        "--launcher", "srun", "--cpus", c),), ())
        pool = Job("pacs_%s_%s_pool" % (tag, b), "pool", b, p.k, p.pool_limit,
                   (_cmd(DRIVER_MODULE, "pool-band", config_path, region, box, c, "--band", b, "--of", p.k,
                         "--launcher", "srun", "--cpus", c),), (inj.name,))
        jobs += [inj, pool]
        pools.append(pool.name)
    jobs.append(Job("pacs_%s_finish" % tag, "finish", "", 1, wall_limit(system.rates["FINISH_MIN"]),
                    (_cmd(STAGES_MODULE, "finish", config_path, region, box, c),), tuple(pools)))
    return jobs


def chain(config, config_path: str, region: str, system: System, box: Optional[Sequence[float]] = None) -> List[Job]:
    """The region's jobs for the phase it is in: while any band's detect row is not done, phase "detect" (prepare =
    every band's mosaic and psf stages on 1 node; detect per band afterok prepare); once all are done, phase "rest"
    (per band inject on K nodes, pool on K nodes afterok inject; finish afterok every pool), K from the detect records;
    [] once finish is done. (No `plan` job: a compute node cannot submit; the driver is rerun on the login node.)"""
    return _chain(config, config_path, region, system, box)


def _phase(jobs: Sequence[Job]) -> str:
    return "rest" if any(j.stage == "inject" for j in jobs) else "detect"


def _code() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run(("git", "-C", here, "rev-parse", "--short", "HEAD"), capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def emit(jobs: Sequence[Job], system: System) -> str:
    """The submission script text for `jobs` (sbatch lines with --dependency=afterok, the job bodies with the
    environment of scripts/submit_pacs_region.sh). Run with bash on a login node; `--dry-run` prints the sbatch lines;
    sbatch runs the same file as `<file> --job <name>`."""
    site = _SITES.get(system.name)
    if site is None:
        raise ValueError("emit: load_system(%r) first" % system.name)
    q = shlex.quote
    phase = _phase(jobs) if jobs else "none"
    out = ["#!/bin/bash",
           "# PACS chain, phase %s, on %s (%s, %d cores per node), emitted %s by %s (code %s)."
           % (phase, system.name, system.partition, system.cpus, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
              DRIVER_MODULE, _code()),
           "# Run on a LOGIN node (a compute node cannot submit, job 3661756):  bash THIS_FILE [--dry-run]",
           "# sbatch runs this same file as the body of each job:  THIS_FILE --job NAME",
           "# Jobs (name, nodes, -t, afterok):"]
    out += ["#   %-34s %3d  %s  %s" % (j.name, j.nodes, j.limit, ",".join(j.after) or "-") for j in jobs]
    if phase == "detect":
        out.append("# Next: once the detect jobs have ended, run the driver again (same command): it writes phase rest.")
    out += ["", "SELF=$(cd \"$(dirname \"$0\")\" && pwd)/$(basename \"$0\")", "SLURM_DIR=%s" % q(site.slurm_dir),
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
        out += ["            %s || exit 1" % c for c in j.commands]
        out.append("            ;;")
    out += ["        *) echo \"unknown job $JOB\" >&2; exit 2 ;;", "    esac", "    exit 0", "fi", "",
            "DRY=0; [ \"${1:-}\" = \"--dry-run\" ] && DRY=1",
            "QUEUED=0; [ $DRY -eq 1 ] || QUEUED=$(squeue -u \"$USER\" -h 2>/dev/null | wc -l)",
            "[ $((%d + QUEUED)) -le %d ] || { echo \"refused: %d jobs (+ $QUEUED queued) against %d on %s\" >&2; exit 1; }"
            % (len(jobs), system.max_queued, len(jobs), system.max_queued, system.name),
            "[ $DRY -eq 1 ] || mkdir -p \"$SLURM_DIR\"",
            "# submit NAME NODES TIME DEPENDENCY-IDS(colon-joined, or empty) -> prints the job id",
            "submit() {",
            "    local name=$1 nodes=$2 time=$3 dep=$4",
            "    local cmd=(sbatch --parsable -J \"$name\" -o \"$SLURM_DIR/%%x_%%j.out\" -e \"$SLURM_DIR/%%x_%%j.err\" -p %s -N $nodes -n $nodes -c %d"
            % (q(system.partition), system.cpus),
            "               -t $time -A %s%s)" % (q(system.account), "".join(" " + q(x) for x in site.sbatch_extra)),
            "    [ -z \"$dep\" ] || cmd+=(--dependency=\"afterok:$dep\")",
            "    cmd+=(\"$SELF\" --job \"$name\")",
            "    if [ $DRY -eq 1 ]; then",
            "        { printf '%q ' \"${cmd[@]}\"; echo; } >&2",
            "        echo \"$name\"",
            "    else",
            "        local id",
            "        id=$(\"${cmd[@]}\" | cut -d';' -f1 | grep -E '^[0-9]+$' | tail -1)",
            "        [ -n \"$id\" ] || { echo \"submission of $name failed: stopping (cancel this chain's earlier jobs: squeue -u $USER)\" >&2; kill -TERM $$; exit 1; }",
            "        echo \"$id\"",
            "    fi",
            "}"]
    var = {j.name: "ID_%d" % i for i, j in enumerate(jobs)}
    for j in jobs:
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
    cmd = [sys.executable, "-W", "ignore", "-m", STAGES_MODULE] + [str(x) for x in extra] + ["--config", config_path,
                                                                                           "--region", region]
    if box is not None:
        cmd += ["--box"] + ["%r" % float(v) for v in box]
    if n_jobs is not None:
        cmd += ["--n-jobs", str(n_jobs)]
    return cmd


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
    box = None if a.box is None else tuple(a.box)
    try:
        (inject_band if verb == "inject-band" else pool_band)(config, a.config, a.region, a.band, a.of, box, launch, a.n_jobs)
    except Exception as e:          # noqa: BLE001 -- the job's exit code
        print("%s %s %s: FAILED: %s" % (verb, a.region, a.band, e), file=sys.stderr, flush=True)
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python -m sesnaimpute.sky.derived.pacs.driver --config C --region R --system S [--box ...] [--out-dir D]`:
    writes each region's chain script for the phase it is in; submits nothing. `inject-band` / `pool-band`: the job
    bodies."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("inject-band", "pool-band"):
        return _band_main(argv[0], argv[1:])
    p = argparse.ArgumentParser(prog="python -m " + DRIVER_MODULE, description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--region", action="append", required=True, help="repeat for several regions (one file each)")
    p.add_argument("--box", type=float, nargs=4, default=None)
    p.add_argument("--out-dir", default=None, help="where the scripts go (default: the system's slurm_dir)")
    p.add_argument("--target-minutes", type=float, default=None)
    p.add_argument("--tasks", type=_tasks_arg, default=None,
                   help="P70=N,P100=N,P160=N: round-0 task counts instead of the detect records (writes phase rest; a dry "
                        "run of the sizing)")
    a = p.parse_args(argv)
    system = load_system(a.config, a.system)
    if a.target_minutes is not None:
        system = dataclasses.replace(system, target_minutes=a.target_minutes)
    config = config_module.load(a.config)
    box = None if a.box is None else tuple(a.box)
    out_dir = a.out_dir or _SITES[system.name].slurm_dir
    os.makedirs(out_dir, exist_ok=True)
    for region in a.region:
        jobs = _chain(config, os.path.abspath(a.config), region, system, box, a.tasks)
        if not jobs:
            print("%s: finish is done; nothing to submit" % region)
            continue
        path = os.path.join(out_dir, "pacs_%s_%s.sh" % (_tag(region), _phase(jobs)))
        with open(path, "w") as f:
            f.write(emit(jobs, system))
        os.chmod(path, 0o755)
        for j in jobs:
            print("%-8s %-34s nodes %3d  -t %s  afterok %s" % (j.stage, j.name, j.nodes, j.limit, ",".join(j.after) or "-"))
        print("wrote %s (run it with bash on a login node; --dry-run to see the sbatch lines)" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
