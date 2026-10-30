"""How many nodes a band's heavy PACS jobs (injection, pooling) take, from the band's work, so that every job runs
about the same time (`pacs_stages` is the staging, `scripts/submit_pacs_region.sh` the submission that uses this).

The work of a band is its round-0 task count N0 (`len(round_tasks(..., 0))`, in `detect.pkl`) times the node-seconds a
round-0 task costs plus the same task's share of the 40 later rounds, measured on Orion A (Stampede3 spr, 112 workers
per node, jobs 3661406-3661441):

    P160 round 0, K = 4:   6.8-8.3 min per share, mean 7.55 min  -> 4 x 7.55 x 60 = 1812 node-s
    P160 rounds 1-40:      ~70 min on one node                    -> 4200 node-s
    P160 tasks:            3873 in every round (edge and hole cells never reach MIN_INJECTIONS, so all 40 later
                           rounds run); N0 = 3873 is taken for round 0, whose pieces cover the same blocks
    ROUND0_S = 1812 / 3873 = 0.468 node-s per task;  LATER_S = 4200 / (40 x 3873) = 0.0271 node-s per task and round

The later rounds are charged on N0, which is more than the later rounds hold (P100: 12.3 min measured, 22 min charged)
and the measured later-round time includes one process start per share and round, which the one-share-per-node layout
pays once: both make K a little high, never low. Pooling (P100: tables 2.7 min once, 4.5 min per share at K = 1) is
scaled by the band's work over P100's measured injection work (576 s + 738 s), not below 1. A system with more cores
per node runs the work proportionally faster (`cpus / 112`).

K = max(1, ceil(work / (rate x TARGET))), at most the nodes one job may hold on the system and a third of the user's
nodes (the three bands' jobs run together). `-t` of a job = 3 x its expected minutes rounded up to 30 min, at least 30
(TACC encumbers the whole limit against the balance).

    python -m sesnaimpute.sky.derived.pacs_sizing --system stampede3 --tasks P70=5026,P100=1231,P160=3873
"""
import argparse
import math

BANDS = ("P70", "P100", "P160")
MAX_EXTRA_BATCHES = 40                      # pacs_completeness.MAX_EXTRA_BATCHES (the round rule is not changed here)
REF_CPUS = 112
ROUND0_S = 1812.0 / 3873.0                  # node-s per round-0 task at REF_CPUS
LATER_S = 4200.0 / (MAX_EXTRA_BATCHES * 3873.0)          # node-s per round-0 task and later round at REF_CPUS
P100_INJECT_S = (9.6 + 12.3) * 60.0         # P100's measured injection node-s (Orion A)
POOL_TABLES_MIN, POOL_SHARD_MIN = 2.7, 4.5  # P100 on one node (Orion A)
DETECT_MIN = {"P70": 21.3, "P100": 6.9, "P160": 12.1}    # one node (Orion A)
PREPARE_MIN, FINISH_MIN = 40.0, 40.0        # one node; NOT measured on Orion A (mosaics reused, photometry): -t 2 h
TARGET_MIN = 45.0
#: per system: workers per node, nodes per job, nodes per user (TACC queue limits)
SYSTEMS = {"stampede3": dict(cpus=112, per_job=64, per_user=96), "vista": dict(cpus=144, per_job=32, per_user=96)}


def wall_limit(expected_min):
    """`-t HH:MM:SS`: 3 x the expected minutes rounded up to 30 min, at least 30."""
    m = max(1, math.ceil(3.0 * expected_min / 30.0)) * 30
    return "%02d:%02d:00" % (m // 60, m % 60)


def node_cap(system):
    s = SYSTEMS[system]
    return max(1, min(s["per_job"], (s["per_user"] - len(BANDS)) // len(BANDS)))


def plan_band(n0, system, target_min=TARGET_MIN):
    """`dict(n0, work, k, inject_min, pool_min, inject_t, pool_t)` of a band with `n0` tasks in round 0; `work` in
    node-seconds at the system's cores."""
    scale = REF_CPUS / SYSTEMS[system]["cpus"]
    work = n0 * (ROUND0_S + MAX_EXTRA_BATCHES * LATER_S) * scale
    k = min(node_cap(system), max(1, math.ceil(work / (target_min * 60.0))))
    inject_min = work / k / 60.0
    pool_min = (POOL_TABLES_MIN + POOL_SHARD_MIN / k) * max(1.0, work / (P100_INJECT_S * scale))
    return dict(n0=n0, work=work, k=k, inject_min=inject_min, pool_min=pool_min,
                inject_t=wall_limit(inject_min), pool_t=wall_limit(pool_min))


def format_plan(band, p):
    return ("plan %-4s: %6d round-0 tasks, work %7.0f node-s (%.1f node-min), K = %d, expected inject %.1f min (-t %s), "
            "pool %.1f min (-t %s)" % (band, p["n0"], p["work"], p["work"] / 60.0, p["k"], p["inject_min"], p["inject_t"],
                                       p["pool_min"], p["pool_t"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--tasks", help="BAND=N0,... round-0 task counts")
    ap.add_argument("--times", action="store_true", help="print the -t of the prepare, detect and finish jobs and stop")
    ap.add_argument("--target-minutes", type=float, default=TARGET_MIN)
    args = ap.parse_args(argv)
    if args.times:
        print("TIMES PREPARE_T=%s FINISH_T=%s %s" % (wall_limit(PREPARE_MIN), wall_limit(FINISH_MIN),
                                                    " ".join("DETECT_%s_T=%s" % (b, wall_limit(DETECT_MIN[b])) for b in BANDS)))
        return
    for kv in args.tasks.split(","):
        b, n = kv.split("=")
        assert b in BANDS, b
        p = plan_band(int(n), args.system, args.target_minutes)
        print(format_plan(b, p))
        print("PLAN %s K=%d INJECT_T=%s POOL_T=%s" % (b, p["k"], p["inject_t"], p["pool_t"]))


if __name__ == "__main__":
    main()
