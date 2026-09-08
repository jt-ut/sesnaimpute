"""Identity check for the `population` copy (W0, `SPEC_PRIORS.md`): every
dataset of each `population/` product must equal, element for element, the
same-named dataset of the `bms/` product it was copied from -- the science
is unchanged, only the area differs (README "The one rule of W0"). Report
only: this never repairs a mismatch, and raises only when one of the two
files is missing, naming the RUNBOOK line that builds it.

Two products the earlier design stopped writing (`prior/gal.py`'s
per-region comparison file and `prior/h2s.py`'s per-source selection file,
both retired in favour of an on-the-fly lookup) are not compared: neither
`bms/` nor `population/` ever builds them.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute import progress
from sesnaimpute.build import run
from sesnaimpute.batches import batches

#: (source, quantity, granule, per_region, module) for every product this
#: copy is required to match, in RUNBOOK order.
PRODUCTS = [
    ("sesna", "column-grid", "survey", False, "sesnaimpute.population.column_grid"),
    ("trilegal", "field-stars", "region", True, "sesnaimpute.population.field_stars"),
    ("pahc", "curve", "survey", False, "sesnaimpute.population.pahc_curve"),
    ("anchors", "tiles", "hpx512", True, "sesnaimpute.population.anchor_tiles"),
    ("anchors", "histograms", "hpx512", True, "sesnaimpute.population.anchor_tiles"),
    ("sesna", "kernel", "survey", False, "sesnaimpute.population.kernel"),
    ("yso", "law", "region", False, "sesnaimpute.population.yso"),
    ("yso", "prior", "sightline", True, "sesnaimpute.population.yso"),
    ("h2s", "law-blurred", "source", True, "sesnaimpute.population.h2s"),
    ("h2s", "prior", "region", True, "sesnaimpute.population.h2s"),
    ("gal", "counts", "survey", False, "sesnaimpute.population.gal"),
    ("anchors", "young-stars", "hpx512", True, "sesnaimpute.population.young_stars"),
    ("anchors", "observed", "hpx512", True, "sesnaimpute.population.anchor_observed"),
    ("anchors", "weights", "tile", True, "sesnaimpute.population.anchor_weights"),
    ("star", "population", "tile", True, "sesnaimpute.population.star_population"),
]

#: Dataset byte size above which a comparison reads and compares in
#: row-batches instead of loading the whole array (CODING_RULES 10b).
BATCH_BYTES = 512 << 20


def _require(path, module):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "population.check: missing %s -- run RUNBOOKtp.sh's `PY %s` line first" % (path, module))


def _equal_dataset(a, b):
    """`(equal, max_abs_diff, first_diff_index)` for one dataset pair,
    NaN-aware and batched over axis 0 once a side exceeds `BATCH_BYTES`
    (rule 10b: nothing holds a large source-granule array whole)."""
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False, None, None
    n = a.shape[0] if a.ndim else 1
    row_bytes = max(1, a.nbytes // max(1, n))
    is_float = a.dtype.kind == "f"
    for start, stop in batches(n, row_bytes, BATCH_BYTES):
        av = a[start:stop] if a.ndim else a[()]
        bv = b[start:stop] if b.ndim else b[()]
        if is_float:
            same = np.array_equal(av, bv, equal_nan=True)
        else:
            same = np.array_equal(av, bv)
        if not same:
            if is_float:
                fa, fb = av.astype(float), bv.astype(float)
                nan_mismatch = np.isnan(fa) != np.isnan(fb)
                diff = np.where(nan_mismatch, np.inf, np.abs(fa - fb))
            else:
                diff = (av != bv)
            flat = np.argmax(diff)
            idx = start + int(flat // (av.size // av.shape[0])) if av.ndim > 1 else start + int(flat)
            max_abs = float(np.max(diff)) if is_float else float("nan")
            return False, max_abs, idx
    return True, None, None


def _compare_file(pop_path, bms_path, label, module):
    _require(pop_path, module)
    _require(bms_path, module)
    with h5py.File(pop_path, "r") as fp, h5py.File(bms_path, "r") as fb:
        names = sorted(set(fp.keys()) | set(fb.keys()))
        # descend into groups (e.g. star_population's per-tile groups)
        stack = list(names)
        all_equal = True
        while stack:
            name = stack.pop(0)
            item_p = fp.get(name)
            item_b = fb.get(name)
            if isinstance(item_p, h5py.Group) or isinstance(item_b, h5py.Group):
                keys = sorted(set((item_p or {}).keys()) | set((item_b or {}).keys()))
                stack.extend(f"{name}/{k}" for k in keys)
                continue
            if item_p is None or item_b is None:
                print(f"population.check: {label} {name}: no (missing on one side)", flush=True)
                all_equal = False
                continue
            equal, max_abs, idx = _equal_dataset(item_p[()], item_b[()])
            if equal:
                print(f"population.check: {label} {name}: yes", flush=True)
            else:
                print(f"population.check: {label} {name}: no max_abs_diff={max_abs} first_diff_index={idx}",
                      flush=True)
                all_equal = False
        attrs_p = dict(fp.attrs)
        attrs_b = dict(fb.attrs)
        for key in sorted(set(attrs_p) | set(attrs_b)):
            if key not in attrs_p or key not in attrs_b or not np.array_equal(attrs_p.get(key), attrs_b.get(key)):
                print(f"population.check: {label} attrs/{key}: no ({attrs_p.get(key)!r} vs {attrs_b.get(key)!r})",
                      flush=True)
                all_equal = False
    return all_equal


def build(config, regions=None):
    """For each product in `PRODUCTS`, opens the `population/` build and its
    `bms/` twin and prints one line per dataset (and per differing root
    attribute): equal, or the max absolute difference and first differing
    index. Survey-granule products are checked once; per-region products
    are checked for every region in `regions` (default: all thirty)."""
    regions = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    with progress.Stage("population.check") as st:
        n_checked = 0
        for source, quantity, granule, per_region, module in PRODUCTS:
            region_list = regions if per_region else [None]
            for region in region_list:
                pop_path = config_module.product_path(config, "population", source, quantity, granule, region=region)
                bms_path = config_module.product_path(config, "bms", source, quantity, granule, region=region)
                label = f"{quantity}_{source}_{granule}" + (f"__{region}" if region else "")
                _compare_file(pop_path, bms_path, label, module)
                n_checked += 1
        st.done(n_products=n_checked, n_regions=len(regions))


if __name__ == "__main__":
    run(build)
