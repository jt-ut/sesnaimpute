"""The one-time in-place migration (CODING_RULES_BMSTP.md rule 5,
briefs/ATTRS.md section 2): walks every product file the writers under
`catalog/`, `sky/derived/`, `population/` and `bmstp/` produce and sets
`UNITS`/`READING` on every dataset from the SAME table those writers use
(`sesnaimpute.attrs_registry.REGISTRY`), without touching any value. A
dataset the registry does not know is an error naming the file and
dataset, not a silent skip. Idempotent: a dataset already carrying its
own correct two attributes is left untouched. Not a RUNBOOK line --
every future build writes its own attributes itself
(`sesnaimpute.build.write_dataset`); this migrates files already on
disk from before that write path existed.

Usage: `python3.9 -m sesnaimpute.build.attrs --root <data root>`.
`sky/download/`, `granules/`, `bms/` and `fittp/` carry no dataset this
pass's registry describes (the first two write no attributed dataset
at all under the current design; the last two are out of scope) and
are not walked.
"""

import argparse
import os

import h5py

from sesnaimpute.attrs_registry import REGISTRY

#: The areas the writers in scope for this migration write to
#: (briefs/ATTRS.md section 1's five source directories, minus `atlas/`,
#: which writes figures, not HDF5 datasets).
AREAS = ("catalog", "sky/derived", "population", "bmstp")

#: Every product stem the current writers can produce (REGISTRY's own
#: first key element) -- a file on disk whose stem is not in this set is
#: not "a product file the writers above produce" (a retired module's
#: output, a superseded granule, a dated backup copy, an old design's
#: file sharing an area directory) and is left untouched, not opened.
KNOWN_STEMS = frozenset(stem for stem, _name in REGISTRY)


def stem_of(filename):
    """The product stem a REGISTRY key's first element names: `filename`
    (a basename, with or without its `.hdf5` extension) stripped of that
    extension and of any trailing `__<Region>` suffix. A region name
    never itself contains "__" (rule 5a's own naming convention uses it
    only as the one quantity/region separator), so splitting on the
    first occurrence is exact."""
    base = filename[:-5] if filename.endswith(".hdf5") else filename
    return base.split("__", 1)[0]


def dataset_key(stem, dataset):
    """The REGISTRY key for `dataset` (an open h5py Dataset): its full
    path under the file root where a fixed-named enclosing group (e.g.
    `fallback`, `RAW`) carries a meaning of its own distinct from the
    same leaf name elsewhere in the file, else its bare leaf name where
    the enclosing group is itself parameterised per instance (e.g.
    `tile_3`, `factor_0`) and every such group shares one reading.
    Tried full-path first so a fixed-group dataset is never shadowed by
    an unrelated top-level entry of the same leaf name."""
    full = dataset.name.lstrip("/")
    if (stem, full) in REGISTRY:
        return full
    return full.rsplit("/", 1)[-1]


def set_attrs_on_file(path):
    """Sets `UNITS`/`READING` on every dataset of the HDF5 file at
    `path`, read from REGISTRY by the file's own stem (`stem_of`) and
    each dataset's own key (`dataset_key`), and removes any other
    attribute a dataset carries (CODING_RULES_BMSTP.md rule 5: exactly
    those two, nothing else -- a file built before this rule's own
    write path existed can carry a stale one, e.g. an old `UNIT`/
    `DEFINITION` pair). Raises naming the file and dataset if REGISTRY
    has no entry for it. Returns how many datasets this call actually
    changed (0 on a fully up-to-date, idempotent re-run)."""
    stem = stem_of(os.path.basename(path))
    written = [0]
    with h5py.File(path, "a") as f:
        def visit(_name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            key = dataset_key(stem, obj)
            entry = REGISTRY.get((stem, key))
            if entry is None:
                raise KeyError(
                    "build.attrs: %r dataset %r (product stem %r) has no "
                    "attrs_registry.REGISTRY entry" % (path, obj.name, stem))
            units, reading = entry
            stale = [k for k in obj.attrs if k not in ("UNITS", "READING")]
            changed = obj.attrs.get("UNITS") != units or obj.attrs.get("READING") != reading or bool(stale)
            if changed:
                for k in stale:
                    del obj.attrs[k]
                obj.attrs["UNITS"] = units
                obj.attrs["READING"] = reading
                written[0] += 1
        f.visititems(visit)
    return written[0]


def iter_product_files(root):
    """Every `.hdf5` file under `root`'s own `AREAS`, in a fixed (sorted)
    order so a run's own log is reproducible."""
    for area in AREAS:
        area_dir = os.path.join(root, area)
        if not os.path.isdir(area_dir):
            continue
        for dirpath, _dirnames, filenames in sorted(os.walk(area_dir)):
            for name in sorted(filenames):
                if name.endswith(".hdf5"):
                    yield os.path.join(dirpath, name)


def run(root):
    all_paths = list(iter_product_files(root))
    in_scope, skipped = [], []
    for path in all_paths:
        stem = stem_of(os.path.basename(path))
        (in_scope if stem in KNOWN_STEMS else skipped).append(path)
    print("build.attrs: %d file(s) found under %s; %d match a known product stem, "
          "%d left untouched (not a current writer's own product)"
          % (len(all_paths), root, len(in_scope), len(skipped)), flush=True)
    n_written_files = 0
    n_datasets_written = 0
    for i, path in enumerate(in_scope, start=1):
        written = set_attrs_on_file(path)
        if written:
            n_written_files += 1
            n_datasets_written += written
        if i % 200 == 0 or i == len(in_scope):
            print("build.attrs: %d/%d in-scope files scanned" % (i, len(in_scope)), flush=True)
    print("build.attrs: done -- %d in-scope files scanned, %d file(s) had a dataset newly set, "
          "%d dataset attribute-set(s) written, %d file(s) skipped as not a current product"
          % (len(in_scope), n_written_files, n_datasets_written, len(skipped)), flush=True)
    return dict(n_files_found=len(all_paths), n_in_scope=len(in_scope), n_skipped=len(skipped),
                n_written_files=n_written_files, n_datasets_written=n_datasets_written,
                skipped_paths=skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run(args.root)
