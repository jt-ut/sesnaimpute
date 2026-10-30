"""The command line of the staged PACS completeness (`pacs_stages`): one job of a region = one invocation.

    python -m sesnaimpute.sky.derived.pacs_stages <stage> --config CFG --region R [options]

Stages (each skips work whose output file exists, so a resubmitted job resumes):
  preflight    the preflight checks and the mosaics of every band (`pacs_completeness.preflight`)
  detect       DETECT of --band
  inject       INJECT round --round of --band, shard --shard of --of
  inject-band  INJECT of --band over --of nodes (`pacs_driver.inject_band`): every round 0 .. --through-round, the
               shares of a round started together on a node each, the next round after all have ended
  pool-tables  the injection tables of --band, written once before its POOL shards
  pool         POOL shard --shard of --of of --band
  pool-band    POOL of --band over --of nodes (`pacs_driver.pool_band`): tables, the shards together, assemble
  assemble     ASSEMBLE of --band
  finish       FINISH of the region (product, merge, sources, photometry, acceptances)
  run          every stage in one process (`run_region`, --of shards)
  plan         from --band's DETECT file, the work, the node count K and the expected minutes of its injection and
               pooling jobs on --system (`pacs_sizing`); the last line is `PLAN <band> K=.. INJECT_T=.. POOL_T=..`
  claim-finish exit 0 for the one caller that finds every band assembled and claims the finish job, 3 otherwise

Common options: --box RA DEC W H (a slice, as pacs_mosaic names it), --n-jobs N (the pool size for this invocation),
--launcher srun|local with --cpus (how inject-band and pool-band start their shares: job steps, or subprocesses).

The hook at the bottom of pacs_stages.py calls `main`:

    if __name__ == "__main__":
        from sesnaimpute.sky.derived.pacs_stages_cli import main
        main()
"""
import argparse
import dataclasses
import sys

from sesnaimpute import config as config_module
from sesnaimpute.sky.derived import pacs_sizing

BANDS = ("P70", "P100", "P160")
STAGES = ("preflight", "detect", "inject", "inject-band", "pool-tables", "pool", "pool-band", "assemble", "finish", "run", "plan",
          "claim-finish")
NEEDS_BAND = ("detect", "inject", "inject-band", "pool-tables", "pool", "pool-band", "assemble", "plan")
NEEDS_SHARD = ("inject", "pool")


def _parser():
    p = argparse.ArgumentParser(prog="python -m sesnaimpute.sky.derived.pacs_stages", description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--config", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--band", choices=BANDS)
    p.add_argument("--round", type=int, default=None, dest="round_")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=1, help="the number of shards K (default 1)")
    p.add_argument("--box", type=float, nargs=4, default=None, metavar=("RA", "DEC", "W", "H"))
    p.add_argument("--n-jobs", type=int, default=None, help="pool size for this invocation, over the config's [run] n_jobs")
    p.add_argument("--reuse-mosaics", action="store_true", help="preflight: mosaics already built are intended")
    p.add_argument("--reference-product", default=None, help="preflight and finish: earlier photometry product")
    p.add_argument("--system", choices=sorted(pacs_sizing.SYSTEMS), default="stampede3", help="plan: the cluster")
    p.add_argument("--target-minutes", type=float, default=pacs_sizing.TARGET_MIN, help="plan: wanted minutes per job")
    p.add_argument("--through-round", type=int, default=None, help="inject-band: the last round to run (default MAX_EXTRA_BATCHES)")
    p.add_argument("--launcher", choices=("srun", "local"), default="local", help="inject-band, pool-band: how shares start")
    p.add_argument("--cpus", type=int, default=1, help="srun launcher: cores of a node, for each step")
    return p


def _check(args, parser):
    if args.stage in NEEDS_BAND and args.band is None:
        parser.error("%s needs --band" % args.stage)
    if args.stage == "inject" and args.round_ is None:
        parser.error("inject needs --round")
    if args.of < 1:
        parser.error("--of must be at least 1")
    if args.stage in NEEDS_SHARD:
        if args.shard is None:
            parser.error("%s needs --shard (0 .. --of - 1)" % args.stage)
        if not 0 <= args.shard < args.of:
            parser.error("--shard %d is outside 0 .. %d" % (args.shard, args.of - 1))


def _load_config(args):
    config = config_module.load(args.config)
    if args.n_jobs is not None:
        config = dataclasses.replace(config, n_jobs=args.n_jobs)
        print("pacs_stages: n_jobs=%d from --n-jobs" % args.n_jobs, flush=True)
    return config


def _plan(stages, config, region, box, bands, system, target_min):
    from sesnaimpute.sky.derived import pacs_completeness
    assert pacs_sizing.MAX_EXTRA_BATCHES == pacs_completeness.MAX_EXTRA_BATCHES
    for b in bands:
        p = pacs_sizing.plan_band(len(stages.round_tasks(config, region, b, 0, box=box)), system, target_min)
        print(pacs_sizing.format_plan(b, p))
        print("PLAN %s K=%d INJECT_T=%s POOL_T=%s" % (b, p["k"], p["inject_t"], p["pool_t"]), flush=True)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    _check(args, parser)
    config = _load_config(args)
    from sesnaimpute.sky.derived import pacs_stages as stages
    box = None if args.box is None else tuple(args.box)
    region, band, kw = args.region, args.band, dict(box=box)
    if args.stage == "preflight":
        from sesnaimpute.sky.derived import pacs_completeness
        pacs_completeness.preflight(config, region, box=box, reuse_mosaics=args.reuse_mosaics,
                                    reference_product=args.reference_product)
    elif args.stage == "detect":
        stages.detect(config, region, band, **kw)
    elif args.stage == "inject":
        stages.inject(config, region, band, args.round_, args.shard, args.of, **kw)
    elif args.stage == "inject-band":
        from sesnaimpute.sky.derived import pacs_completeness, pacs_driver
        through = pacs_completeness.MAX_EXTRA_BATCHES if args.through_round is None else args.through_round
        pacs_driver.inject_band(config, args.config, region, band, args.of, box, pacs_driver.launcher(args.launcher, args.cpus),
                                n_jobs=args.n_jobs, through_round=through)
    elif args.stage == "pool-band":
        from sesnaimpute.sky.derived import pacs_driver
        pacs_driver.pool_band(config, args.config, region, band, args.of, box, pacs_driver.launcher(args.launcher, args.cpus),
                              n_jobs=args.n_jobs)
    elif args.stage == "claim-finish":
        from sesnaimpute.sky.derived import pacs_driver
        return 0 if pacs_driver.claim_finish(config, region, box) else 3
    elif args.stage == "pool-tables":
        stages.pool_tables(config, region, band, **kw)
    elif args.stage == "pool":
        stages.pool(config, region, band, args.shard, args.of, **kw)
    elif args.stage == "assemble":
        stages.assemble(config, region, band, **kw)
    elif args.stage == "finish":
        # the store is cleared only once every step has run, so a FINISH that fails in the photometry resumes
        stages.finish(config, region, reference_product=args.reference_product, clear=False, **kw)
        from sesnaimpute.sky.derived import pacs_completeness
        pacs_completeness.finish_region(config, region, box=box, reference_product=args.reference_product)
        stages._clear_store(config, region, box)
        stages.clear_prepared(config, region, box)
    elif args.stage == "run":
        stages.run_region(config, region, n_shards=args.of, **kw)
    else:
        _plan(stages, config, region, box, (band,), args.system, args.target_minutes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
