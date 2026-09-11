"""The one entry-point convention every build module follows.

Every build module exposes `build(config, regions=None)` and ends with:

    if __name__ == "__main__":
        run(build)

`run` parses `CONFIG [--regions R1 R2 ...]` from the command line, loads the
config, and calls `build_fn(config, regions=None or the given list)`. The
command line defaults to all thirty regions; per-region products are one
file per region, so a subset run never touches the others.

`run` also holds the stage's worker pool alive for the stage's whole
duration. joblib's loky backend retires a worker idle for 300 s
(`joblib/_parallel_backends.py:531`, hardcoded); on this machine a freshly
spawned worker's startup imports die intermittently in dyld, so a mid-stage
respawn is a crash risk, not just a cost. `_StagePoolLoky` raises that
default to a day and is registered once; `build_fn` runs inside one
`parallel_config(backend="stage_pool")` set before it starts and never
changed inside it, so every `Parallel` call the stage makes, at any nesting
depth or phase, draws from the one pool created for its first call instead
of tearing one down and respawning between phases or between regions.
"""

import argparse
import sys

from joblib import parallel_config, register_parallel_backend
from joblib._parallel_backends import LokyBackend

from sesnaimpute import config as config_module


class _StagePoolLoky(LokyBackend):
    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None,
                  idle_worker_timeout=86400, **backend_args):
        return super().configure(n_jobs=n_jobs, parallel=parallel, prefer=prefer,
                                  require=require, idle_worker_timeout=idle_worker_timeout,
                                  **backend_args)


register_parallel_backend("stage_pool", _StagePoolLoky)


def run(build_fn):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args(sys.argv[1:])
    config = config_module.load(args.config)
    with parallel_config(backend="stage_pool"):
        build_fn(config, regions=args.regions)
