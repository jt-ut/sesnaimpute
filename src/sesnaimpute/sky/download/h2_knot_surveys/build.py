"""Four H2-shock-knot survey tables, fetched verbatim from CDS, plus two
manual acquisitions with no reachable machine-readable source.

Fetches each catalogue's ReadMe and every data file the ReadMe's own File
Summary lists, verbatim, into `<dest>/<catalogue>/`:

    J_ApJ_767_147/{ReadMe,table2.dat}
        Giannini T. et al. 2013, ApJ 767, 147 -- Spitzer-IRAC survey of H2
        jets and knots in the Vela-D molecular cloud (69 knots, 2.12 um and
        per-band IRAC photometry, table2.dat).
    J_A+A_496_153/{ReadMe,table2.dat,table3.dat,tableb1.dat}
        Davis C.J. et al. 2009, A&A 496, 153 -- a census of H2 outflows
        along the Orion A molecular ridge (147 knots with measured proper
        motions in tableb1.dat).
    J_MNRAS_454_2586/{ReadMe,tableb1.dat,tablec1.dat,tabled1.dat}
        Froebrich D. et al. 2015, MNRAS 454, 2586 -- the UWISH2 extended
        H2-emission-source catalogue. VizieR's own tabled1.dat is an
        excerpt only (its ReadMe: "This table is only an excerpt of the
        first few objects", 69 of the survey's 33,200 features) -- it does
        not carry the full per-feature surface brightness, area and class
        columns, so the PDF-appendix transcription below is not retired.
    J_A+A_392_239/{ReadMe,table3.dat,notes.dat}
        Stanke T., McCaughrean M.J. & Zinnecker H. 2002, A&A 392, 239 -- an
        IR survey of H2 outflows in Orion A. SPEC_PRIORS.md section 7 rules
        this an EXCLUDED comparison sample (flow centroids, not individual
        knot positions): the bytes are carried for that comparison only, no
        knot table is built from it.

Two further acquisitions have no reachable CDS/VizieR table and stay
manual, placed by hand at the destinations below:

    walawender/*.html, *.mhtml -- Walawender J., Bally J. & Reipurth B.
        2005, AJ 129, 2308 (Perseus) and Walawender J., Bally J.,
        Di Francesco J. & Jorgensen J. 2005, AJ 130, 1795 (Barnard 1).
        Neither paper has a VizieR/CDS entry (`J/AJ/129/2308` and
        `J/AJ/130/1795` both 404 on cdsarc.cds.unistra.fr as of 2026-09-05)
        and IOPscience blocks automated access to the article table pages,
        so these are the owner's saved journal table-page views. Manual
        acquisition -- see the RUNBOOK comment line naming both journal
        URLs.
    uwish2/appendix_large.pdf, uwish2/tabled1_transcribed.hdf5
        Froebrich et al. 2015's institutional-access PDF appendix (Table
        D1, the full 33,200-feature UWISH2 catalogue) and its transcription
        into HDF5 (made once by the predecessor package). VizieR's
        `J/MNRAS/454/2586` does not carry these columns for the full
        catalogue (see above), so this transcription is the acquired bytes
        for Table D1. Manual acquisition -- Rule 14 forbids opening a PDF.

Feeds SPEC_PRIORS.md section 7 (H2S), via `sky.derived.knots`.
"""

import os

from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch


#: CDS catalogue -> (base URL, files its own ReadMe File Summary lists).
_CDS_CATALOGUES = {
    "J_ApJ_767_147": (
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/767/147",
        ("ReadMe", "table2.dat"),
    ),
    "J_A+A_496_153": (
        "https://cdsarc.cds.unistra.fr/ftp/J/A+A/496/153",
        ("ReadMe", "table2.dat", "table3.dat", "tableb1.dat"),
    ),
    "J_MNRAS_454_2586": (
        "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/454/2586",
        ("ReadMe", "tableb1.dat", "tablec1.dat", "tabled1.dat"),
    ),
    "J_A+A_392_239": (
        "https://cdsarc.cds.unistra.fr/ftp/J/A+A/392/239",
        ("ReadMe", "table3.dat", "notes.dat"),
    ),
}

_WALAWENDER_FILES = (
    "Walawender2005(AJ 129, 2308)_tbl2.html",
    "Walawender2005(AJ 129, 2308)_tbl3.html",
    "Walawender2005(AJ 130, 1795)_tbl3.mhtml",
)


def _fetch_cds(dest_dir):
    """Fetches every CDS catalogue's ReadMe and listed data files. Returns
    the number of files fetched."""
    n_files = 0
    for catalogue, (base_url, files) in _CDS_CATALOGUES.items():
        for name in files:
            fetch(f"{base_url}/{name}", f"{dest_dir}/vizier/{catalogue}/{name}")
            n_files += 1
    return n_files


def _require_manual(dest_dir):
    """The two hand-placed acquisitions (Walawender html/mhtml table pages;
    the UWISH2 PDF-appendix transcription) must already be at their
    destinations; a missing one fails naming where to put it. Returns the
    number of files present."""
    wanted = [f"{dest_dir}/walawender/{name}" for name in _WALAWENDER_FILES]
    wanted.append(f"{dest_dir}/uwish2/tabled1_transcribed.hdf5")
    missing = [w for w in wanted if not os.path.isfile(w)]
    if missing:
        raise FileNotFoundError(
            "h2_knot_surveys: hand-placed input missing -- acquire it by hand "
            f"(see this module's docstring) and place it at {missing[0]!r}")
    return len(wanted)


def build(config, regions=None):
    """Fetches the four CDS-reachable knot-survey catalogues into
    `<data_root>/sky/download/h2_knot_surveys/` and requires the two hand-placed
    acquisitions (Walawender: no CDS entry; UWISH2 Table D1: a PDF appendix) there.
    `regions` is accepted for interface uniformity and ignored: this is a
    survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/h2_knot_surveys"
    n_fetched = _fetch_cds(dest_dir)
    n_manual = _require_manual(dest_dir)
    print(f"h2_knot_surveys build: {n_fetched} files fetched from CDS, "
          f"{n_manual} hand-placed files present")


if __name__ == "__main__":
    run(build)
