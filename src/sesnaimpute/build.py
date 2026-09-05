"""The one entry-point convention every build module follows.

Every build module exposes `build(config, regions=None)` and ends with:

    if __name__ == "__main__":
        run(build)

`run` parses `CONFIG [--regions R1 R2 ...]` from the command line, loads the
config, and calls `build_fn(config, regions=None or the given list)`. The
command line defaults to all thirty regions; per-region products are one
file per region, so a subset run never touches the others.
"""

import argparse
import sys

from sesnaimpute import config as config_module


def run(build_fn):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args(sys.argv[1:])
    config = config_module.load(args.config)
    build_fn(config, regions=args.regions)
