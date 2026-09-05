"""Row-level updates for the thirty-region products (CODING_RULES.md 5c):
a build given a subset of regions writes only those regions' rows (or,
for a per-region-group product, only those regions' groups) and leaves
the rest of the file exactly as it was.
"""

import os
from contextlib import contextmanager

import h5py
import numpy as np

from sesnaimpute.regions import REGIONS

ALL_REGIONS = [r.name for r in REGIONS]
ROW_IDX = {name: i for i, name in enumerate(ALL_REGIONS)}


@contextmanager
def open_product(path, granule="region"):
    """Opens the shared product at `path` in append mode if it already
    exists -- so a caller's own per-row or per-group upsert leaves every
    other row or group untouched -- and in create mode otherwise; sets
    the root `GRANULE` attribute either way."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    mode = "a" if os.path.exists(path) else "w"
    with h5py.File(path, mode) as f:
        f.attrs["GRANULE"] = granule
        yield f


def _fill_value(dtype):
    """The placeholder a freshly created row carries before any build
    has written it: NaN for floats, -1 for integers, False for bool."""
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.bool_):
        return False
    if np.issubdtype(dtype, np.integer):
        return -1
    raise ValueError(f"tables.update_rows: no fill value for dtype {dtype}")


def _widen_to_canonical(f):
    """Upgrades a file whose `REGION` dataset does not already list all
    thirty regions in `ALL_REGIONS` order (a product last written before
    this module existed, holding only whichever regions its own last
    build touched) to the canonical thirty-row shape: every dataset is
    rebuilt at that shape, each old row carried to its own region's slot
    by name, every other slot taking that dataset's own placeholder. A
    file already in canonical shape is left untouched."""
    if "REGION" not in f:
        return
    existing = [r.decode() if isinstance(r, bytes) else r for r in f["REGION"][:]]
    if existing == ALL_REGIONS:
        return
    old_idx = {name: i for i, name in enumerate(existing)}
    for name in list(f.keys()):
        if name == "REGION":
            continue
        old = np.asarray(f[name][()])
        if old.ndim == 0 or old.shape[0] != len(existing):
            continue  # a global scalar (e.g. a region-independent constant), not a row
        new = np.full((len(ALL_REGIONS),) + old.shape[1:],
                       _fill_value(old.dtype), dtype=old.dtype)
        for region, i in old_idx.items():
            if region in ROW_IDX:
                new[ROW_IDX[region]] = old[i]
        del f[name]
        f.create_dataset(name, data=new)
    del f["REGION"]
    f.create_dataset("REGION", data=np.array(
        [r.encode("utf-8") for r in ALL_REGIONS], dtype="S64"))


def update_rows(path, regions, rows, granule="region", attrs=None):
    """Writes `rows` (dataset name -> array, first axis aligned to
    `regions`) into the thirty-row product at `path`: creates the file
    with all thirty regions' rows, every dataset filled with its own
    placeholder, when the file or a dataset is absent, and writes only
    the given regions' rows -- the rest of the file is left exactly as
    it was (CODING_RULES.md 5c). `attrs`, if given, are set as root
    attributes.
    """
    regions = list(regions)
    row_idx = ROW_IDX
    with open_product(path, granule) as f:
        _widen_to_canonical(f)
        if "REGION" not in f:
            f.create_dataset("REGION", data=np.array(
                [r.encode("utf-8") for r in ALL_REGIONS], dtype="S64"))
        for name, values in rows.items():
            values = np.asarray(values)
            if name not in f:
                shape = (len(ALL_REGIONS),) + values.shape[1:]
                f.create_dataset(name, data=np.full(
                    shape, _fill_value(values.dtype), dtype=values.dtype))
            dset = f[name]
            for i, region in enumerate(regions):
                dset[row_idx[region]] = values[i]
        if attrs:
            for key, value in attrs.items():
                f.attrs[key] = value
