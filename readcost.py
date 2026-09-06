"""The acceptance run for `sesnaimpute.prior.callable`'s per-batch
tabulation (`prior.callable.SourcePrior.prepare`): for NGC 7129 and
Perseus, `prepare` on a batch of ten thousand sources (or all, if the
region has fewer), then per class the read cost of one already-prepared
source read five thousand times (microseconds per read -- CODING_RULES.md
10, "time it on real data"), and the normalisation check
(`10_POSTERIOR.md` section 3) on five random sources of the batch per
class, on a 200x200 `(a, log10 B)` grid, within 0.02 for the tabulated
classes and 1e-3 for YSO (`prior.callable.check`'s own bar). Scope fixed
by the owner 2026-09-06: no larger run, no survey extrapolation.

Run under `capped.sh` (CODING_RULES.md 10a):
    ./capped.sh /usr/local/bin/python3.9 readcost.py <config> <region>...
"""

import sys
import time

import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute.prior import callable as callable_module

#: The batch `prepare` tabulates (CODING_RULES.md 10b): about ten
#: thousand sources, or all of a smaller region.
_BATCH_N = 10000

#: The owner's scoped-down acceptance figure (2026-09-06): one already-
#: prepared source, read this many times per class.
_N_READS = 5000

#: The owner's scoped-down normalisation check: this many random sources
#: of the batch, per class.
_N_CHECK_SOURCES = 5

#: The owner's scoped-down check grid: `prior.callable._CHECK_GRID_N`
#: (400, `check`'s own default) overridden to this for this run only, so
#: the existing grid-building helpers (`_family_grid`, `_extinction_grid`,
#: `_integrate`, `_integrate_yso`) are reused verbatim rather than
#: reimplemented.
_CHECK_GRID_N_SCOPED = 200


def read_cost(prior, row, cls, n_reads=_N_READS):
    """`(wall_s, us_per_read)`: one already-prepared source, `cls`'s own
    read, called once vectorised over `n_reads` query points (rule 8: a
    single array call, not a Python loop) -- the production read shape,
    `log10 B` swept over a plausible range at the source's own `a` scale."""
    # `a_probe`/`b_probe` at shape `(1, n_reads)` -- `rows_probe` is ONE
    # source (n=1): `log_density`'s own broadcasting reads a length-m 1-D
    # array as m SOURCES, not m query points of one source (the same trap
    # `check`'s own read-cost probe avoids), so the query axis must stay
    # explicitly 2-D here.
    a_val = max(float(prior.table["A_COL_K"][row]), 1.0e-3)
    a_probe = np.full((1, n_reads), a_val)
    b_probe = np.linspace(-2.0, 2.0, n_reads)[None, :]
    mi_probe = None
    if cls in ("gal", "h2s"):
        fref_size = prior.gal_fref.size if cls == "gal" else prior.h2s_fref.size
        mi_probe = (np.arange(n_reads, dtype=np.intp) % fref_size)[None, :]
    rows_probe = np.array([row], dtype=np.intp)
    t0 = time.time()
    prior.log_density(cls, rows_probe, a_probe, b_probe, model_index=mi_probe)
    wall = time.time() - t0
    return wall, wall / n_reads * 1.0e6


def normalisation_check(prior, rows, n_sources=_N_CHECK_SOURCES, seed=0):
    """`worst`, per class: `abs(integral - 1)` over `n_sources` random rows
    of the prepared batch `rows`, on the owner's scoped-down 200x200 grid
    -- reuses `check`'s own grid-building helpers with `_CHECK_GRID_N`
    overridden for this call only."""
    saved_grid_n = callable_module._CHECK_GRID_N
    callable_module._CHECK_GRID_N = _CHECK_GRID_N_SCOPED
    try:
        rng = np.random.default_rng(seed)
        n = min(n_sources, rows.size)
        probe_rows = rng.choice(rows, size=n, replace=False)

        worst = {cls: 0.0 for cls in callable_module.CLASSES}
        for row in probe_rows:
            a_col = float(prior.table["A_COL_K"][row])

            for cls in callable_module.FAMILY_CLASSES:
                shape = prior.shapes[cls]
                a_grid, b_grid = callable_module._family_grid(shape, a_col)
                a_grid = a_grid[a_grid > 0.0]
                integral = callable_module._integrate(prior, cls, row, a_grid, b_grid)
                worst[cls] = max(worst[cls], abs(integral - 1.0))

            sl_rows = prior.table["HPX256_ROW"][row:row + 1].astype(np.intp)
            node_lo = prior.table["NODE_LO"][row:row + 1].astype(np.intp)
            node_hi = np.minimum(node_lo + 1, prior.yso_shape.n_node - 1)
            t_lo, _ = prior.yso_shape._gather_quadrature(sl_rows, node_lo)
            t_hi, _ = prior.yso_shape._gather_quadrature(sl_rows, node_hi)
            t_max_row = max(float(t_lo.max()), float(t_hi.max()))

            a_grid_gal = callable_module._extinction_grid(a_col, t_max_row * 3.0)
            b_grid_gal = prior.gal_log10_s_grid - np.log10(prior.gal_fref[0])
            integral = callable_module._integrate(prior, "gal", row, a_grid_gal, b_grid_gal, model_index=0)
            worst["gal"] = max(worst["gal"], abs(integral - 1.0))

            a_grid_cloud = callable_module._extinction_grid(a_col, t_max_row * 3.0)
            integral = callable_module._integrate_yso(prior, row, a_grid_cloud)
            worst["yso"] = max(worst["yso"], abs(integral - 1.0))

            log10_sigma_lo = prior.h2s_logsig_mean - 6 * prior.h2s_logsig_std
            log10_sigma_hi = prior.h2s_logsig_mean + 6 * prior.h2s_logsig_std
            b_grid_h2s = (np.linspace(log10_sigma_lo, log10_sigma_hi, callable_module._CHECK_GRID_N)
                         - np.log10(prior.h2s_fref[0]))
            integral = callable_module._integrate(prior, "h2s", row, a_grid_cloud, b_grid_h2s, model_index=0)
            worst["h2s"] = max(worst["h2s"], abs(integral - 1.0))
        return worst
    finally:
        callable_module._CHECK_GRID_N = saved_grid_n


def run_region(config, region):
    t_prep0 = time.time()
    prior = callable_module.SourcePrior(config, region)
    n_batch = min(_BATCH_N, prior.n_source)
    rng = np.random.default_rng(0)
    batch_rows = rng.choice(prior.n_source, size=n_batch, replace=False)
    prior.prepare(batch_rows)
    t_prep = time.time() - t_prep0
    print("readcost: %s: prepare() on %d of %d sources -> %.3gs"
          % (region, n_batch, prior.n_source, t_prep), flush=True)

    probe_row = int(batch_rows[0])
    print("readcost: %s: read cost (one prepared source x %d reads per class)"
          % (region, _N_READS), flush=True)
    for cls in callable_module.CLASSES:
        wall, us = read_cost(prior, probe_row, cls)
        print("readcost: %s: %s: %d reads in %.4gs -> %.4g us/read"
              % (region, cls, _N_READS, wall, us), flush=True)

    worst = normalisation_check(prior, batch_rows)
    print("readcost: %s: normalisation check (%d random sources of the batch, "
          "%dx%d grid; bar 0.02 all but yso 1e-3)"
          % (region, _N_CHECK_SOURCES, _CHECK_GRID_N_SCOPED, _CHECK_GRID_N_SCOPED), flush=True)
    for cls in callable_module.CLASSES:
        bar = 1.0e-3 if cls == "yso" else 0.02
        flag = "" if worst[cls] <= bar else "  ** EXCEEDS BAR **"
        print("readcost: %s: %s: worst |integral - 1| = %.4g (bar %.4g)%s"
              % (region, cls, worst[cls], bar, flag), flush=True)


if __name__ == "__main__":
    cfg = config_module.load(sys.argv[1])
    for _region in sys.argv[2:]:
        run_region(cfg, _region)
