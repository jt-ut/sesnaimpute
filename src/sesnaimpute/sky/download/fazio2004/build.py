"""The Fazio et al. (2004) IRAC galaxy number-count table.

Fazio G.G. et al. 2004, ApJS 154, 39, Table 1 (IRAC differential galaxy
source counts). No CDS/VizieR or arXiv URL serves these exact bytes
programmatically -- the old `fetch_external.fazio2004.build.frozen_inputs`
module never contacts the network either, and instead reuses a pre-staged
copy. This module does the same: it copies the four accepted files
(the computational CSV, the arXiv source tarball, its loose TeX source,
and the published Table 1 transcription) verbatim, byte-for-byte, from
the staged location `_STAGED_DIR` below, which is where an already
completed acquisition of this family lives.

Feeds SPEC_PRIORS.md section 5.1 (GAL; the Fazio galaxy-count fit).
"""

import os
import shutil

from sesnaimpute.build import run

# Pre-staged location this module copies from -- there is no reachable
# download URL for these bytes (see module docstring).
_STAGED_DIR = "/Users/jtaylor/Dropbox/Research/SESNA_Complete/sky/download/fazio2004"

_FILES = (
    "fazio2004_table1_irac_counts.csv",
    "0405595.tar.gz",
    "ms.tex",
    "Fazio 2004 (ApJS 154, 39, Table 1).txt",
)


def build(config, regions=None, _limit=None):
    """Copies the four accepted Fazio 2004 files verbatim from
    `_STAGED_DIR`. `regions` is accepted for interface uniformity and
    ignored: this is a survey-wide product. `_limit` is a rehearsal
    knob: when set, only the first `_limit` files are copied.
    """
    dest_dir = f"{config.data_root}/sky/download/fazio2004"
    os.makedirs(dest_dir, exist_ok=True)
    names = _FILES if _limit is None else _FILES[:_limit]
    for name in names:
        src_path = f"{_STAGED_DIR}/{name}"
        dest_path = f"{dest_dir}/{name}"
        shutil.copyfile(src_path, dest_path)
        n_bytes = os.path.getsize(dest_path)
        print(f"fazio2004 build: {src_path} -> {dest_path} ({n_bytes} bytes)")


if __name__ == "__main__":
    run(build)
