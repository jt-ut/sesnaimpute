"""The command line of the staged PACS completeness (`pacs_stages`): one job of a region = one invocation.

    python -m sesnaimpute.sky.derived.pacs_stages <stage> --config CFG --region R [options]

Stages (each skips work whose output file exists, so a resubmitted job resumes):
  preflight    the preflight checks and the mosaics of every band (`pacs_completeness.preflight`)
  detect       DETECT of --band
  inject       INJECT round --round of --band, shard --shard of --of
  inject-rest  INJECT rounds --round (default 1) .. MAX_EXTRA_BATCHES of --band, every shard of --of in turn, in
               this one process, stopping at the first round with no tasks (the rounds after round 0 are one
               small batch each, behind a barrier: one job runs them all instead of 40 jobs queueing)
  pool-tables  the injection tables of --band, written once before its POOL shards
  pool         POOL shard --shard of --of of --band
  assemble     ASSEMBLE of --band
  finish       FINISH of the region (product, merge, sources, photometry, acceptances)
  run          every stage in one process (`run_region`, --of shards)
  plan         from the DETECT files, the tasks of round 0 per band and the shard count K that gives ~--target-hours
               per job at --rate tasks per second (the rate is an input: measure it on the system)

Common options: --box RA DEC W H (a slice, as pacs_mosaic names it), --n-jobs N (the pool size for this invocation).
The hook at the bottom of pacs_stages.py calls `main`:

    if __name__ == "__main__":
        from sesnaimpute.sky.derived.pacs_stages_cli import main
        main()
"""
import argparse
import dataclasses
import math
import sys

from sesnaimpute import config as config_module

BANDS = ("P70", "P100", "P160")
STAGES = ("preflight", "detect", "inject", "inject-rest", "pool-tables", "pool", "assemble", "finish", "run", "plan")
NEEDS_BAND = ("detect", "inject", "inject-rest", "pool-tables", "pool", "assemble")
NEEDS_SHARD = ("inject", "pool")
# tasks per second of one injection job on a whole node: the P70 round 0 of Orion A on Stampede3 spr (112 workers)
# ran ~1.1-1.5 per second (job 3661209); an input of `plan`, never a constant of the staging
PLAN_RATE = 1.2


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
    p.add_argument("--rate", type=float, default=PLAN_RATE, help="plan: tasks per second of one job")
    p.add_argument("--target-hours", type=float, default=1.0, help="plan: wanted hours per job")
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


def _inject_rest(stages, config, region, band, first, n_shards, box):
    from sesnaimpute.sky.derived import pacs_completeness
    for r in range(first, pacs_completeness.MAX_EXTRA_BATCHES + 1):
        n = len(stages.round_tasks(config, region, band, r, box=box))
        if not n:
            print("inject-rest %s %s: round %d has no tasks, the injections are complete" % (region, band, r), flush=True)
            return
        print("inject-rest %s %s: round %d, %d tasks, %d shards" % (region, band, r, n, n_shards), flush=True)
        for k in range(n_shards):
            stages.inject(config, region, band, r, k, n_shards, box=box)
    print("inject-rest %s %s: MAX_EXTRA_BATCHES reached" % (region, band), flush=True)


def _plan(stages, config, region, box, bands, rate, target_hours):
    print("plan %s: round 0 tasks per band; K for ~%.1f h per job at %.2f tasks/s per job (the rate is an input)"
          % (region, target_hours, rate))
    for b in bands:
        try:
            n = len(stages.round_tasks(config, region, b, 0, box=box))
        except FileNotFoundError:
            print("  %-5s no detect file: run `detect --band %s` first" % (b, b))
            continue
        k = max(1, math.ceil(n / (rate * 3600.0 * target_hours)))
        print("  %-5s %6d tasks in round 0 -> K = %d (%.2f h per job)" % (b, n, k, n / k / rate / 3600.0))


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
    elif args.stage == "inject-rest":
        _inject_rest(stages, config, region, band, 1 if args.round_ is None else args.round_, args.of, box)
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
    elif args.stage == "run":
        stages.run_region(config, region, n_shards=args.of, **kw)
    else:
        _plan(stages, config, region, box, (band,) if band else BANDS, args.rate, args.target_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
