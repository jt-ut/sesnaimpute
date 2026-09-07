"""Progress messages for the build stages: one line at the start of a stage, one
line per unit of work once a stage has run past a few seconds, one line at the end
with the product path, the key numbers, the wall time and the peak resident memory.
Every stage prints through this module so a run's log reads as one record.
"""

import resource
import sys
import time

_MIN_INTERVAL_S = 10.0


def _peak_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


class Stage:
    """`with Stage("prior.star_selection", region) as st:` ... `st.tick(i, n, "sources")`
    ... `st.done(path, key=value, ...)`."""

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
                  f"peak {_peak_mb():.0f} MB", flush=True)
            self._last = now

    def done(self, path=None, **numbers):
        keys = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in numbers.items())
        where = f" -> {path}" if path else ""
        print(f"{self.name}: done in {time.time() - self.t0:.1f} s, peak {_peak_mb():.0f} MB{where} {keys}",
              flush=True)
