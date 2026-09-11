"""The one entry-point convention every build module follows.

Every build module exposes `build(config, regions=None)` and ends with:

    if __name__ == "__main__":
        run(build)

`run` parses `CONFIG [--regions R1 R2 ...]` from the command line, loads the
config, and calls `build_fn(config, regions=None or the given list)`. The
command line defaults to all thirty regions; per-region products are one
file per region, so a subset run never touches the others.

This module also holds a stage's worker pool alive for the stage's whole
duration. joblib's loky backend retires a worker idle for 300 s
(`joblib/_parallel_backends.py:531`, hardcoded); on this machine a freshly
spawned worker's startup imports die intermittently in dyld, so a mid-stage
respawn is a crash risk, not just a cost. `_StagePoolLoky` raises that
default to a day and is registered as joblib's *default* backend
(`make_default=True`) at import time, not set through `parallel_config` —
a call site that leaves `backend` unspecified (every plain
`Parallel(n_jobs=...)` in the package) then resolves to this pool with
joblib's own default-backend path, which is what preserves the
`prefer="threads"` fallback several call sites rely on for shared-memory
work; a `parallel_config(backend=...)` context instead marks the backend
"explicit" and turns that fallback off package-wide, which is the wrong
trade for one hardening fix. The net effect is the same pool serving every
`Parallel` call in the stage instead of one torn down and respawned
between phases or between regions.
"""

import argparse
import sys

from joblib import register_parallel_backend
from joblib._parallel_backends import LokyBackend
from joblib.externals.loky import process_executor as _loky_workers

from sesnaimpute import config as config_module


def _keep_worker():
    """Runs once in every worker the stage pool starts: a worker's memory
    grows by design (it holds the pools of the region it works on), and
    loky would otherwise read that growth as a leak, retire the worker
    after 300 MB and spawn a replacement -- a fresh spawn is the one
    event this machine's loader does not survive. The retirement rule is
    loky's own module constant, read in the worker at each check, so it is
    raised here, in the worker, to a size no task reaches."""
    if not hasattr(_loky_workers, "_MAX_MEMORY_LEAK_SIZE"):
        raise AttributeError("sesnaimpute.build: this joblib's loky has no _MAX_MEMORY_LEAK_SIZE; "
                             "find its worker-retirement rule and raise it here")
    _loky_workers._MAX_MEMORY_LEAK_SIZE = sys.maxsize


class _StagePoolLoky(LokyBackend):
    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None,
                  idle_worker_timeout=86400, initializer=_keep_worker, **backend_args):
        return super().configure(n_jobs=n_jobs, parallel=parallel, prefer=prefer,
                                  require=require, idle_worker_timeout=idle_worker_timeout,
                                  initializer=initializer, **backend_args)


register_parallel_backend("stage_pool", _StagePoolLoky, make_default=True)


def run(build_fn):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args(sys.argv[1:])
    config = config_module.load(args.config)
    build_fn(config, regions=args.regions)
