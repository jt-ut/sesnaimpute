"""Progress messages for the build stages: one line at the start of a stage, one
line per unit of work once a stage has run past a few seconds, one line at the end
with the product path, the key numbers, the wall time and the peak resident memory.
Every stage prints through this module so a run's log reads as one record.

Convention (every stage module, no exceptions): `progress.Stage` is the only way
a stage prints its start, progress and end -- no bare `print(...)` progress line,
no module's own timer, no `psutil` or other peak-memory read; `Stage` already
measures peak resident (`peak_mb`). The `name` passed to `Stage` is the stage's
own RUNBOOK module name with the leading `sesnaimpute.` dropped, e.g.
`Stage("bmstp.density", region)` for the RUNBOOK line `PY sesnaimpute.bmstp.density`,
`Stage("population.field_stars", region)` for `PY sesnaimpute.population.field_stars`
-- never a name left over from an earlier design (`prior.*`, `fit.*`, `impute.*`).
A region loop opens one `Stage` per region (`st.tick` inside it, at most once
every ten seconds and at the last unit) and calls `st.done(path, **numbers)`
with the few numbers a reader of the log needs to know that region did its job;
once every region is done, the stage prints one summary line with `summary`:
`<name>: done, <R> regions, <T> s, peak <P> MB`, so a log reader sees both the
per-region and the whole-stage picture without re-deriving either. A stage with
no region axis calls `Stage` once, with no region, and needs no summary line.
Diagnostic prints that state a science result -- an identity check, a
cross-check against an independent number, a disclosure of what a computation
could not measure -- are not progress lines and print on their own, outside
`Stage`; but a diagnostic the owner has ruled out is deleted, never kept
printing behind a flag nobody sets. The wrapper `capped.sh` prints its own
`capped: peak resident N MB, exit E` line after the process exits; no stage
prints a peak line that duplicates it.
"""

import resource
import sys
import time

_MIN_INTERVAL_S = 10.0


def peak_mb():
    """The peak resident set size of this process so far, in MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def summary(name, n_regions, t0):
    """The one line a multi-region stage prints after its last region's `Stage`
    has called `done`: `<name>: done, <n_regions> regions, <T> s, peak <P> MB`,
    `t0` being the `time.time()` taken before the region loop started."""
    print(f"{name}: done, {n_regions} regions, {time.time() - t0:.0f} s, "
          f"peak {peak_mb():.0f} MB", flush=True)


class Stage:
    """`with Stage("bmstp.density", region) as st:` ... `st.tick(i, n, "sources")`
    ... `st.done(path, key=value, ...)`. `name` is the stage's own RUNBOOK module
    name with `sesnaimpute.` dropped; see the module docstring's convention
    paragraph. A multi-region stage follows its last region's `done` with one
    `progress.summary(name, n_regions, t0)` call."""

    def __init__(self, name, region=None):
        self.name = name
        self.region = region
        self.t0 = time.time()
        self._last = self.t0
        where = f" [{region}]" if region else ""
        print(f"{name}{where}: start", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            print(f"{self.name}: FAILED after {time.time() - self.t0:.1f} s ({exc_type.__name__}: {exc})",
                  flush=True)
        return False

    def tick(self, i, n, unit="items"):
        """Print progress for unit `i` of `n`, at most every ten seconds and at the last."""
        now = time.time()
        if now - self._last >= _MIN_INTERVAL_S or i == n:
            elapsed = now - self.t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (n - i) / rate if rate > 0 else float("nan")
            print(f"{self.name}: {i}/{n} {unit}, {elapsed:.0f} s elapsed, ~{eta:.0f} s left, "
                  f"peak {peak_mb():.0f} MB", flush=True)
            self._last = now

    def done(self, path=None, **numbers):
        keys = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in numbers.items())
        where = f" -> {path}" if path else ""
        print(f"{self.name}: done in {time.time() - self.t0:.1f} s, peak {peak_mb():.0f} MB{where} {keys}",
              flush=True)
