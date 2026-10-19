"""One-time ascii -> HDF5 conversion for the TRILEGAL files already on
disk (T6, 2026-10-19): the owner is not re-pulling the 931 `.dat` files
already drawn ("I'm not re-downloading all of the trilegal data"), so
this module reads each one and writes the identical typed schema
`sky.download.trilegal.build.fetch_region_part` now writes for a fresh
pull (`build.write_ascii_table_h5`), verifying the conversion per file:
the row count, then every column, equal to the ascii parsed at the
column's own stored dtype (`np.array_equal`, same cast). The `.dat`
files are NOT deleted -- the owner decides that later.

Converts every `.dat` under the directory (default: `<data_root>/
sky/download/trilegal`, the fetch's own area; `--source-dir` overrides
it, for converting a copy of a few regions only). Never run from
RUNBOOKtp.sh: by hand, once --

    python -m sesnaimpute.sky.download.trilegal.convert <root.cfg>
"""

import argparse
import glob
import os
import re

import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute.sky.download.trilegal import build as trilegal_build


def _region_label(dat_path):
    """The region/file-group one `.dat` belongs to, for the one summary
    line per region (T6 item 2): the filename with its `_part<n>` and
    `_pt<n>` suffixes and extension stripped -- `build._file_names`'s
    own naming convention, read backwards. The one colour-query file
    (`colour_gaiaDR2_2mass_perseus.dat`) has neither suffix and prints
    under its own bare name."""
    stem = os.path.splitext(os.path.basename(dat_path))[0]
    stem = re.sub(r"_part\d+$", "", stem)
    stem = re.sub(r"_pt\d+$", "", stem)
    return stem


def convert_file(dat_path):
    """Converts one `.dat` to the `.h5` beside it and verifies it against
    the ascii it was parsed from. Returns `(h5_path, n_rows, n_bytes_dat,
    n_bytes_h5)`."""
    header_line, names, df = trilegal_build.read_ascii_table(dat_path)
    h5_path = dat_path[:-4] + ".h5"
    trilegal_build.write_ascii_table_h5(h5_path, header_line, names, df)

    h5_df = trilegal_build.read_region_part_h5(h5_path)
    if len(df) != len(h5_df):
        raise ValueError(f"trilegal convert: {h5_path}: row count {len(h5_df)} != "
                          f"{len(df)} parsed from {dat_path}")
    for name in names:
        clean = trilegal_build.sanitize_column_name(name)
        expect = df[name].to_numpy(dtype=trilegal_build.column_dtype(clean))
        got = h5_df[name].to_numpy()
        if not np.array_equal(expect, got):
            raise ValueError(f"trilegal convert: {h5_path}: column {name!r} differs "
                              f"from the ascii parsed at its stored dtype")
    return h5_path, len(df), os.path.getsize(dat_path), os.path.getsize(h5_path)


def convert(config, source_dir=None):
    """Converts every `.dat` under `source_dir` (default: `<data_root>/
    sky/download/trilegal`) to the schema `build.fetch_region_part` now
    writes for a fresh pull, verified per file (`convert_file`), printing
    one line per region (T6 item 2). The `.dat` files are left in place.
    """
    source_dir = source_dir or f"{config.data_root}/sky/download/trilegal"
    dat_paths = sorted(glob.glob(f"{source_dir}/*.dat"))
    if not dat_paths:
        raise FileNotFoundError(f"trilegal convert: no .dat files under {source_dir}")

    by_region = {}
    for path in dat_paths:
        by_region.setdefault(_region_label(path), []).append(path)

    with progress_module.Stage("sky.download.trilegal.convert") as st:
        n_done = 0
        n_rows_total = 0
        bytes_dat_total = 0
        bytes_h5_total = 0
        for region in sorted(by_region):
            n_rows_region = 0
            bytes_dat_region = 0
            bytes_h5_region = 0
            for path in by_region[region]:
                _, n_rows, n_bytes_dat, n_bytes_h5 = convert_file(path)
                n_rows_region += n_rows
                bytes_dat_region += n_bytes_dat
                bytes_h5_region += n_bytes_h5
                n_done += 1
                st.tick(n_done, len(dat_paths), "files")
            print(f"trilegal convert: {region}: {len(by_region[region])} file(s), "
                  f"{n_rows_region} rows verified, {bytes_dat_region} dat bytes -> "
                  f"{bytes_h5_region} h5 bytes ({bytes_h5_region / bytes_dat_region:.3f}x)")
            n_rows_total += n_rows_region
            bytes_dat_total += bytes_dat_region
            bytes_h5_total += bytes_h5_region
        st.done(source_dir, files=n_done, rows=n_rows_total,
                dat_bytes=bytes_dat_total, h5_bytes=bytes_h5_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--source-dir", default=None,
                         help="override the directory of .dat files to convert "
                              "(testing only; the default is the config's own "
                              "sky/download/trilegal)")
    args = parser.parse_args()
    config = config_module.load(args.config)
    convert(config, source_dir=args.source_dir)


if __name__ == "__main__":
    main()
