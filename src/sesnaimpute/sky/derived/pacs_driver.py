"""The drivers of the multi-node PACS jobs (`pacs_stages` has the stages, `pacs_sizing` the node counts): one job holds
K nodes and runs a band's stage over all of them, every share (`inject --round r --shard k --of K`, `pool --shard k
--of K`) on a node of its own, the K shares of a round started together and the next round only after all have ended
(round r + 1 is planned from the counts of every share of round r).

A `Launcher` starts the shares: `SrunLauncher` as job steps of the batch allocation (one node each), `LocalLauncher`
as subprocesses of this machine (the laptop checks run the same driver code). A share whose file exists is not
started again, so a job that was killed resumes at the first file missing. A share that fails fails the job, after
the other shares of its round have ended.
"""
import os
import subprocess
import sys

from sesnaimpute.sky.derived import pacs_completeness as pc
from sesnaimpute.sky.derived import pacs_stages as stages

MODULE = "sesnaimpute.sky.derived.pacs_stages"
FINISH_MARKER = "finish_submitted"


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


def launcher(name, cpus):
    return SrunLauncher(cpus) if name == "srun" else LocalLauncher()


def share_cmd(stage, config_path, region, box, n_jobs, band=None, round_=None, shard=None, of=None):
    """The command line of one share (or single-node step) of a stage."""
    cmd = [sys.executable, "-W", "ignore", "-m", MODULE, stage, "--config", config_path, "--region", region]
    if band is not None:
        cmd += ["--band", band]
    if round_ is not None:
        cmd += ["--round", str(round_)]
    if shard is not None:
        cmd += ["--shard", str(shard), "--of", str(of)]
    if box is not None:
        cmd += ["--box"] + ["%g" % v for v in box]
    if n_jobs is not None:
        cmd += ["--n-jobs", str(n_jobs)]
    return cmd


def _run_shares(launch, cmds, what, shards=None):
    """Runs the commands together (`shards`: the shard number of each, for the message); raises when one failed."""
    codes = launch.run(cmds)
    failed = [i for i, c in enumerate(codes) if c != 0]
    if failed:
        label = lambda i: i if shards is None else shards[i]
        raise RuntimeError("%s: shares %s failed (exit codes %s)" % (what, [label(i) for i in failed], [codes[i] for i in failed]))


def inject_band(config, config_path, region, band, k, box, launch, n_jobs=None, through_round=pc.MAX_EXTRA_BATCHES):
    """INJECT of a band over `k` nodes: rounds 0 .. `through_round`, each with its shares started together, stopping
    at the first round without pieces."""
    d = stages._band_dir(config, region, band, box)
    for r in range(through_round + 1):
        n = len(stages.round_tasks(config, region, band, r, box=box))
        if not n:
            print("inject-band %s %s: round %d has no tasks, the injections are complete" % (region, band, r), flush=True)
            return
        todo = [s for s in range(k) if not os.path.exists(os.path.join(d, stages._shard_name("inject_r%d" % r, s, k)))]
        print("inject-band %s %s: round %d, %d tasks, %d of %d shards to run" % (region, band, r, n, len(todo), k), flush=True)
        _run_shares(launch, [share_cmd("inject", config_path, region, box, n_jobs, band, r, s, k) for s in todo],
                    "inject %s round %d" % (band, r), todo)
    print("inject-band %s %s: through round %d" % (region, band, through_round), flush=True)


def pool_band(config, config_path, region, band, k, box, launch, n_jobs=None):
    """POOL of a band over `k` nodes: the injection tables on one node, the `k` pooling shards together, then the
    band's ASSEMBLE on one node."""
    d = stages._band_dir(config, region, band, box)
    one = lambda stage: [share_cmd(stage, config_path, region, box, n_jobs, band)]
    if not os.path.exists(os.path.join(d, "band.pkl")):
        _run_shares(launch, one("pool-tables"), "pool-tables %s" % band)
        todo = [s for s in range(k) if not os.path.exists(os.path.join(d, stages._shard_name("pool", s, k)))]
        print("pool-band %s %s: %d of %d shards to run" % (region, band, len(todo), k), flush=True)
        _run_shares(launch, [share_cmd("pool", config_path, region, box, n_jobs, band, None, s, k) for s in todo],
                    "pool %s" % band, todo)
    _run_shares(launch, one("assemble"), "assemble %s" % band)


def claim_finish(config, region, box):
    """True for exactly one caller once every band's `band.pkl` exists: it creates the marker (O_EXCL) and is the one
    to submit the finish job. False when a band is not assembled yet or the finish was claimed already."""
    for b in pc.BANDS:
        if not os.path.exists(os.path.join(stages._band_dir(config, region, b, box), "band.pkl")):
            return False
    try:
        fd = os.open(os.path.join(stages.stage_dir(config, region, box), FINISH_MARKER), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.fsync(fd)
    os.close(fd)
    return True
