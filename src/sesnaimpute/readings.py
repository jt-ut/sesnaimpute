"""`UNITS`/`READING`: the two attributes CODING_RULES_BMSTP.md rule 5
(amended by `briefs/LOGFLUX.md` section 4) puts on every dataset a fitter
or posterior product writes, and no other attribute beyond those a
consumer already reads.
"""


def set_readings(group, readings):
    """Sets `UNITS` (a short string) and `READING` (one sentence) on every
    dataset named in `readings`, a `{dataset name: (units, reading)}` map
    -- a name not present in `group` is skipped, since not every product
    this is called on writes every dataset a shared map describes (e.g. a
    part file carries a subset of the joined product's own datasets)."""
    for name, (units, reading) in readings.items():
        if name in group:
            group[name].attrs["UNITS"] = units
            group[name].attrs["READING"] = reading
