"""Row-batch boundaries for per-source stages (CODING_RULES.md rule 10b).

Every stage that reads, computes, or writes one row per source works in
batches: a batch is read, computed, and written before the next is
touched, so nothing holds every source of a region at once. `batches`
turns a row count and a per-row byte footprint into the (start, stop)
slices that keep one batch's resident footprint under a fixed budget.
"""

def batches(n_rows, row_bytes, budget_bytes=512 << 20):
    """Yields half-open `(start, stop)` index slices covering
    `range(n_rows)`, each holding as many rows as fit in `budget_bytes`
    at `row_bytes` bytes per row (at least one row per batch, even if a
    single row exceeds the budget).
    """
    rows_per_batch = max(1, int(budget_bytes // max(1, row_bytes)))
    for start in range(0, n_rows, rows_per_batch):
        yield start, int(min(start + rows_per_batch, n_rows))
