"""The region shapes page, one stage of its own for the runbook: the
prior's mass in the nuisance plane with the survey's selection included,
`atlas.shapes.build_region_prior` (SPEC_BMSTP_DRAFT.md sec. 8). The page's
construction lives beside the per-source page in `atlas.shapes`; this
module is its command line, so the runbook's `--from`/`--to` can name it.
"""
import argparse
import dataclasses

from sesnaimpute import config as config_module
from sesnaimpute.atlas import shapes


def build(config, regions=None):
    shapes.build_region_prior(config, regions=regions)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    config = config_module.load(args.config)
    if args.n_jobs is not None:
        config = dataclasses.replace(config, n_jobs=args.n_jobs)
        print("%s: n_jobs=%d from --n-jobs (config says %d)"
              % ("sesnaimpute.atlas.shapes_region", args.n_jobs, config_module.load(args.config).n_jobs))
    build(config, regions=args.regions)


if __name__ == "__main__":
    _main()
