"""Reports what is on disk under `sky/` against what `DATA.md` records.

Run by hand: `python -m sesnaimpute.sky.data_check CONFIG`. Lists every
directory under `<data_root>/sky/download/` and every product file under
`<data_root>/sky/derived/`, parses the product column of every markdown
table in `<data_root>/DATA.md`, and prints two lists: on disk but not
named there, and named there but not on disk. It prints and stops, always
exiting 0; nothing else in the pipeline reads `DATA.md` (IMPLEMENTATION.md
section 1c).
"""

import os
import sys

from sesnaimpute import config as config_module


def _disk_products(data_root):
    """One entry per source directory under `sky/download/`, one entry
    per file under `sky/derived/`, as paths relative to `data_root`.
    """
    products = set()

    download_dir = os.path.join(data_root, "sky", "download")
    if os.path.isdir(download_dir):
        for name in sorted(os.listdir(download_dir)):
            if os.path.isdir(os.path.join(download_dir, name)):
                products.add(f"sky/download/{name}")

    derived_dir = os.path.join(data_root, "sky", "derived")
    if os.path.isdir(derived_dir):
        for root, _dirs, files in os.walk(derived_dir):
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), data_root)
                products.add(rel)

    return products


def _data_md_products(data_root):
    """The product column of every markdown table in `DATA.md`, backticks
    and trailing slashes stripped.
    """
    entries = set()
    with open(os.path.join(data_root, "DATA.md")) as f:
        for line in f:
            if not line.startswith("|"):
                continue
            cell = line.split("|")[1].strip().strip("`").rstrip("/")
            if not cell or cell == "product" or set(cell) == {"-"}:
                continue
            entries.add(cell)
    return entries


def main(config_path):
    config = config_module.load(config_path)
    on_disk = _disk_products(config.data_root)
    in_data_md = _data_md_products(config.data_root)

    print("on disk, not in DATA.md:")
    for entry in sorted(on_disk - in_data_md):
        print(f"  {entry}")

    print("in DATA.md, not on disk:")
    for entry in sorted(in_data_md - on_disk):
        print(f"  {entry}")


if __name__ == "__main__":
    main(sys.argv[1])
