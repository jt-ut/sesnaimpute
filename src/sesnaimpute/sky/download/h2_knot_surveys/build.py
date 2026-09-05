"""Five external H2-shock-knot surveys, staged bytes copied verbatim.

Copies the already-acquired raw tables of the H2 knot/outflow literature to
the canonical location `<data_root>/sky/download/h2_knot_surveys/`, keeping
every file's name and its place in the tree:

    vizier/J_ApJ_767_147/{ReadMe,table2.dat}
        Giannini T. et al. 2013, ApJ 767, 147 -- Spitzer-IRAC survey of H2
        jets and knots in the Vela-D molecular cloud (69 knots, 2.12 um and
        per-band IRAC photometry).
    vizier/J_A+A_496_153/{ReadMe,tableb1.dat,...}
        Davis C.J. et al. 2009, A&A 496, 153 -- a census of H2 outflows
        along the Orion A molecular ridge (147 knots with measured proper
        motions in tableb1.dat).
    vizier/J_A+A_392_239/J_A+A_392_239.vot
        Stanke T., McCaughrean M.J. & Zinnecker H. 2002, A&A 392, 239 -- an
        IR survey of H2 outflows in Orion A. SPEC_PRIORS.md section 7 rules
        this an EXCLUDED comparison sample (flow centroids, not individual
        knot positions): the bytes are carried for that comparison only, no
        knot table is built from it.
    vizier/J_MNRAS_454_2586/{ReadMe,tableb1.dat,tablec1.dat,tabled1.dat}
        Froebrich D. et al. 2015, MNRAS 454, 2586 -- the UWISH2 extended
        H2-emission-source catalogue. VizieR's own tabled1.dat is an
        excerpt only (its ReadMe: "This table is only an excerpt of the
        first few objects", 69 of the survey's 33,200 features); the full
        table is the transcribed appendix below.
    vizier/J_A+A_426_171/{ReadMe,table2.dat}, vizier/J_ApJ_844_38/{ReadMe,table1-3.dat}
        Two further VizieR H2-outflow tables staged alongside the five
        surveys SPEC_PRIORS.md section 7 names (Khanzadyan T. et al. 2004,
        A&A 426, 171, H2 flows in rho Ophiuchi A; Wolf-Chase G. et al.
        2017, ApJ 844, 38, MHOs toward 22 high-mass-outflow regions). No
        product in section 7 reads either: the bytes are carried, unused.
    walawender/*.html, walawender/*.mhtml
        Walawender J., Bally J. & Reipurth B. 2005, AJ 129, 2308 (Perseus,
        NOAO survey table pages) and Walawender J., Bally J., Di Francesco
        J. & Jorgensen J. 2005, AJ 130, 1795 (Barnard 1) -- IOPscience
        blocks automated access to article table pages (no VizieR/CDS
        machine-readable version exists for either paper), so these are
        the owner's saved journal table-page views (html for the Perseus
        paper, mhtml devtools snapshots for the Barnard 1 paper).
    uwish2/appendix_large.pdf, uwish2/stv1729_Supplementary_Data.zip
        Froebrich et al. 2015's institutional-access PDF appendix (Table
        D1, the full 33,200-feature UWISH2 catalogue behind the VizieR
        excerpt above) and the paper's zipped online supplement.
    uwish2/tabled1_transcribed.hdf5
        The old package's transcription of that PDF appendix (Froebrich+
        2015 Table D1, all 33,200 features) -- copied verbatim from
        `sky/download/h2-shock-surveys/parsed/uwish2_tabled1.hdf5`. Rule 14
        forbids opening a PDF; this transcription is treated as the
        acquired bytes for Table D1, made once by the predecessor package
        and not remade here.

`PRODUCT.json` and `MANIFEST.json` files anywhere in the staged tree are
bookkeeping and are not copied.

Feeds SPEC_PRIORS.md section 7 (H2S), via `sky.derived.knots`.
"""

import os
import shutil

from sesnaimpute.build import run

#: Pre-staged location this module copies from -- there is no reachable
#: download URL for any of these five surveys (see module docstring).
_STAGED_DIR = "/Users/jtaylor/Dropbox/Research/SESNA_Complete/sky/download/h2-shock-surveys"

_SKIP_NAMES = ("PRODUCT.json", "MANIFEST.json")


def _copy_raw_tree(dest_dir):
    """Copies every file under `_STAGED_DIR/raw/`, verbatim, to `dest_dir`,
    keeping each file's path relative to `raw/` and skipping bookkeeping
    files. Returns `(n_files, n_bytes)`."""
    src_root = f"{_STAGED_DIR}/raw"
    n_files = 0
    n_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(src_root):
        rel_dir = os.path.relpath(dirpath, src_root)
        for name in sorted(filenames):
            if name in _SKIP_NAMES:
                continue
            src_path = os.path.join(dirpath, name)
            dest_path = os.path.join(dest_dir, rel_dir, name) if rel_dir != "." \
                else os.path.join(dest_dir, name)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copyfile(src_path, dest_path)
            file_bytes = os.path.getsize(dest_path)
            n_files += 1
            n_bytes += file_bytes
            print(f"h2_knot_surveys build: {src_path} -> {dest_path} ({file_bytes} bytes)")
    return n_files, n_bytes


def _copy_transcribed_uwish2(dest_dir):
    """Copies the old package's PDF-appendix transcription (Froebrich+2015
    Table D1) to `uwish2/tabled1_transcribed.hdf5`, verbatim."""
    src_path = f"{_STAGED_DIR}/parsed/uwish2_tabled1.hdf5"
    dest_path = f"{dest_dir}/uwish2/tabled1_transcribed.hdf5"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copyfile(src_path, dest_path)
    file_bytes = os.path.getsize(dest_path)
    print(f"h2_knot_surveys build: {src_path} -> {dest_path} ({file_bytes} bytes)")
    return file_bytes


def build(config, regions=None):
    """Copies the staged `raw/` tree (five surveys plus two further VizieR
    tables staged alongside them) and the transcribed UWISH2 appendix
    verbatim to `<data_root>/sky/download/h2_knot_surveys/`. `regions` is
    accepted for interface uniformity and ignored: this is a survey-wide
    product.
    """
    dest_dir = f"{config.data_root}/sky/download/h2_knot_surveys"
    os.makedirs(dest_dir, exist_ok=True)
    n_files, n_bytes = _copy_raw_tree(dest_dir)
    transcribed_bytes = _copy_transcribed_uwish2(dest_dir)
    n_files += 1
    n_bytes += transcribed_bytes
    print(f"h2_knot_surveys build: {n_files} files, {n_bytes} bytes total")


if __name__ == "__main__":
    run(build)
