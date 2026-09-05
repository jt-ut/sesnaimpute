"""Region-axis products: create-once, update-in-place (CODING_RULES.md
rule 5c).

A product that is one file with a region axis (thirty rows) is created
whole the first time any build writes it and, after that, updated in
place: a build given a subset of regions overwrites only those rows and
leaves the rest of the file untouched. `update_rows` is the one place
that does this for every module shipping such a table, so a region list
never rewrites all thirty.
"""

import os

import h5py
import numpy as np

from sesnaimpute import regions as regions_module


def _fill_value(dtype):
    """The absent-row fill for a dataset's dtype: `False` for a boolean
    dataset, `NaN` otherwise.
    """
    if np.issubdtype(dtype, np.bool_):
        return False
    return np.nan


def update_rows(path, regions, rows, granule="region"):
    """Writes `rows` (a `{dataset name: array}` map whose first axis
    matches `regions`) into the region-axis product at `path`.

    Creates the file if it is absent, with all thirty rows: `REGION` from
    `sesnaimpute.regions.REGIONS`, in that order, every dataset in `rows`
    NaN-filled (`False`-filled if boolean), and the root attribute
    `GRANULE` set to `granule`. A dataset named in `rows` that the file
    does not yet carry (an older build of the same product, before a
    dataset was added) is created the same way. Either way, only the
    rows for `regions` are then written.
    """
    all_regions = [r.name for r in regions_module.REGIONS]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "a") as f:
        if "REGION" not in f:
            f.attrs["GRANULE"] = granule
            f.create_dataset("REGION", data=np.array([r.encode("utf-8") for r in all_regions]))

        file_regions = [r.decode() if isinstance(r, bytes) else r for r in f["REGION"][:]]
        row_index = [file_regions.index(r) for r in regions]

        for name, arr in rows.items():
            arr = np.asarray(arr)
            if name not in f:
                shape = (len(file_regions),) + arr.shape[1:]
                f.create_dataset(name, data=np.full(shape, _fill_value(arr.dtype), dtype=arr.dtype))
            dset = f[name]
            for k, i in enumerate(row_index):
                dset[i, ...] = arr[k]
