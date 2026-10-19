"""Within-region parallelism (owner ruling 2026-10-09): a stage runs its regions
one after another and, inside a region, spreads the work over many small pieces
-- far more pieces than workers, so pieces of uneven cost even out and no core
waits on the one expensive chunk. One region's data is in memory at a time.

`run(fn, pieces, n_jobs)` is the one pool every stage uses: `fn(piece)` over
`pieces` on `n_jobs` loky worker processes, results returned in the order of
`pieces` (so a stage that sums partial results sums them in one fixed order),
each worker pinned to one numba and one BLAS thread (the pool is the
parallelism; a threaded kernel inside a worker would oversubscribe the
machine, as `fittp.sweep` already pins). Progress is counted in the parent,
where the count is true.

Region-wide read-only inputs go in `fn` itself (a `functools.partial` or a
closure): joblib pickles `fn` once per batch of tasks, not once per piece,
and memory-maps large numpy arrays to the workers. A piece should carry
indices or slice bounds, not data.

Only NUMPY ARRAYS bound into `fn` are memory-mapped. Any other large object
bound into it (a run-length mask, a tree, a dict of arrays, an open file)
is PICKLED and sent to every worker with every batch -- measured on
sky.derived.coverage's Cygnus X masks as 35 s of a 38 s stage. Bind such an
object as its underlying numpy arrays, or have each worker build or open
it once (a module-level per-process cache keyed on the region, cleared
when the region changes, as bmstp.sample_star does), never per piece.
"""

import os

from joblib import Parallel, delayed

#: pieces per worker a stage should aim for: enough that the last piece to
#: finish is a small fraction of the stage, few enough that per-piece
#: overhead (a few ms) stays negligible
PIECES_PER_WORKER = 20

_PINNED = False


def _pin_threads():
    global _PINNED
    if _PINNED:
        return
    import numba
    import threadpoolctl
    numba.set_num_threads(1)
    threadpoolctl.threadpool_limits(1, user_api="blas")
    _PINNED = True


def _call_pinned(fn, piece):
    _pin_threads()
    return fn(piece)


def n_pieces(n_items, n_jobs, min_items=1):
    """How many pieces to cut `n_items` into for `n_jobs` workers:
    `PIECES_PER_WORKER * n_jobs`, but never pieces smaller than `min_items`
    and never more pieces than items."""
    n = max(1, int(n_jobs)) * PIECES_PER_WORKER
    n = min(n, max(1, n_items // max(1, int(min_items))))
    return max(1, min(n, max(1, n_items)))


def slices(n_items, n_jobs, min_items=1):
    """Half-open `(start, stop)` bounds cutting `range(n_items)` into
    `n_pieces(...)` contiguous pieces of near-equal size."""
    k = n_pieces(n_items, n_jobs, min_items)
    edges = [round(i * n_items / k) for i in range(k + 1)]
    return [(edges[i], edges[i + 1]) for i in range(k) if edges[i + 1] > edges[i]]


def run(fn, pieces, n_jobs, stage=None, unit="pieces"):
    """`[fn(p) for p in pieces]`, computed on `n_jobs` worker processes,
    in the order of `pieces`. `stage` (a `progress.Stage`) is ticked in the
    parent as results arrive. `n_jobs == 1` runs in this process."""
    pieces = list(pieces)
    n = len(pieces)
    out = []
    if int(n_jobs) <= 1 or n <= 1:
        for i, p in enumerate(pieces):
            out.append(fn(p))
            if stage is not None:
                stage.tick(i + 1, n, unit)
        return out
    gen = Parallel(n_jobs=int(n_jobs), return_as="generator")(
        delayed(_call_pinned)(fn, p) for p in pieces)
    for i, r in enumerate(gen):
        out.append(r)
        if stage is not None:
            stage.tick(i + 1, n, unit)
    return out


def iterate(fn, pieces, n_jobs, stage=None, unit="pieces"):
    """Generator twin of `run`: yields `fn(p)` for each `p` in `pieces`, one
    at a time, in the same piece order, instead of collecting every result
    into a list first -- a caller that reduces as results arrive (summing,
    or writing into a preallocated array) never holds more than the pieces
    still in flight, where `run`'s own `out` list holds the whole region's
    worth of raw per-piece results at once. `n_jobs == 1` runs in this
    process."""
    pieces = list(pieces)
    n = len(pieces)
    if int(n_jobs) <= 1 or n <= 1:
        for i, p in enumerate(pieces):
            yield fn(p)
            if stage is not None:
                stage.tick(i + 1, n, unit)
        return
    gen = Parallel(n_jobs=int(n_jobs), return_as="generator")(
        delayed(_call_pinned)(fn, p) for p in pieces)
    for i, r in enumerate(gen):
        yield r
        if stage is not None:
            stage.tick(i + 1, n, unit)
