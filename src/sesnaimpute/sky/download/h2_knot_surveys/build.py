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

Also extracts UWISH2's own Table C1 from that same PDF appendix -- the
survey's image list, one row per WFCAM detector-array image, with the
image centre in equatorial and galactic coordinates, seeing, zero-point
and one-pixel background noise: the survey's own footprint record
(`population.knot_rate` reads it as the UWISH2 image-square footprint
test). Table C1 is typeset portrait (Table D1 is landscape), so
pdfplumber's own line assembly works directly; the data glyphs still sit
in per-page Type3 subset fonts with no ToUnicode map, resolved the same
way `sky.derived.knots`'s own transcription note describes for Table D1
-- each font's `/Encoding /Differences` array (glyph names `aNNN`, NNN
the ASCII codepoint). Signs are never assumed: the '+' glyph decodes to
an arbitrary page-varying substitute, so every row's declination and
galactic-latitude sign is resolved from the row's OWN astrometry (all
four sign combinations tried, the one whose (RA, Dec) reproduces the
printed (l, b) kept). Writes `uwish2/uwish2_tablec1.csv`.

Feeds SPEC_PRIORS.md section 7 (H2S), via `sky.derived.knots`.
"""

import csv
import os
import re

import numpy as np
import pdfplumber
from pdfminer.pdftypes import resolve1

from sesnaimpute import progress as progress_module
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


def _fetch_cds(dest_dir, st=None):
    """Fetches every CDS catalogue's ReadMe and listed data files, ticking
    `st` (a `progress.Stage`) per file if given. Returns the number of
    files fetched."""
    all_names = [(catalogue, name) for catalogue, (_, files) in _CDS_CATALOGUES.items() for name in files]
    n_total = len(all_names)
    n_files = 0
    for catalogue, name in all_names:
        base_url = _CDS_CATALOGUES[catalogue][0]
        fetch(f"{base_url}/{name}", f"{dest_dir}/vizier/{catalogue}/{name}")
        n_files += 1
        if st is not None:
            st.tick(n_files, n_total, "files")
    return n_files


def _require_manual(dest_dir):
    """The three hand-placed acquisitions (Walawender html/mhtml table
    pages; the UWISH2 PDF appendix, which carries both Table D1's
    transcription and Table C1) must already be at their destinations; a
    missing one fails naming where to put it. Returns the number of
    files present."""
    wanted = [f"{dest_dir}/walawender/{name}" for name in _WALAWENDER_FILES]
    wanted.append(f"{dest_dir}/uwish2/tabled1_transcribed.hdf5")
    wanted.append(f"{dest_dir}/uwish2/appendix_large.pdf")
    missing = [w for w in wanted if not os.path.isfile(w)]
    if missing:
        raise FileNotFoundError(
            "h2_knot_surveys: hand-placed input missing -- acquire it by hand "
            f"(see this module's docstring) and place it at {missing[0]!r}")
    return len(wanted)


# ------------------------------------------------------------- UWISH2 C1

_TABLEC1_COLS = ["Tile", "Image", "RAdeg", "DEdeg", "GLON", "GLAT",
                  "Seeing", "mapzp", "e_mapzp", "Noise"]
_TABLEC1_GLYPH = re.compile(r"a(\d+)\Z")
_TABLEC1_SCORE = re.compile(r"w\d{8}_\d{5}_[wxyz]")
# Tile Image RA Dec GLON GLAT Seeing zp e_zp Noise -- the sign characters
# are whatever the page's font subset produced, matched as "not a digit".
_TABLEC1_ROW = re.compile(
    r"(H2_\S+)\s+(w\d{8}_\d{5}_[wxyz])\s+"
    r"(\d+\.\d+)\s+([^\d\s]?)(\d+\.\d+)\s+"
    r"(\d+\.\d+)\s+([^\d\s]?)(\d+\.\d+)\s+"
    r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")


def _tablec1_font_maps(page):
    out = {}
    res = resolve1(page.page_obj.resources) or {}
    for name, ref in (resolve1(res.get("Font")) or {}).items():
        font = resolve1(ref)
        enc = resolve1(font.get("Encoding"))
        if not isinstance(enc, dict) or "Differences" not in enc:
            continue
        mapping, code = {}, None
        for item in resolve1(enc["Differences"]):
            if isinstance(item, int):
                code = item
                continue
            glyph = item.name if hasattr(item, "name") else str(item)
            hit = _TABLEC1_GLYPH.match(glyph)
            if hit and code is not None:
                mapping[code] = chr(int(hit.group(1)))
            if code is not None:
                code += 1
        if mapping:
            out[name] = mapping
    return out


def _tablec1_decode(line, mapping):
    return "".join(c if c == " " else mapping.get(ord(c), c) for c in line)


def _tablec1_best_map(lines, maps):
    best, score = None, 0
    for mapping in maps.values():
        s = sum(len(_TABLEC1_SCORE.findall(_tablec1_decode(l, mapping))) for l in lines)
        if s > score:
            best, score = mapping, s
    return best, score


def _tablec1_repair_signs(rows):
    """Resolves each row's declination and galactic-latitude sign from
    its own astrometry: the (RA, Dec) that reproduces the printed (l, b)
    on the sky (module docstring)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if not rows:
        return rows, np.array([])
    ra = np.array([r[2] for r in rows])
    dec = np.array([abs(r[3]) for r in rows])
    lon = np.array([r[4] for r in rows])
    lat = np.array([abs(r[5]) for r in rows])

    best_sep = np.full(len(rows), np.inf)
    best_sd = np.ones(len(rows))
    best_sb = np.ones(len(rows))
    for sd in (1.0, -1.0):
        gal = SkyCoord(ra * u.deg, sd * dec * u.deg, frame="icrs").galactic
        for sb in (1.0, -1.0):
            sep = SkyCoord(lon * u.deg, sb * lat * u.deg,
                            frame="galactic").separation(gal).arcsec
            take = sep < best_sep
            best_sep[take] = sep[take]
            best_sd[take] = sd
            best_sb[take] = sb

    for i, r in enumerate(rows):
        r[3] = best_sd[i] * dec[i]
        r[5] = best_sb[i] * lat[i]
    return rows, best_sep


def _extract_uwish2_tablec1(pdf_path):
    """Table C1: portrait pages near the front of the appendix PDF, one
    row per WFCAM image (module docstring). Returns `(rows, pages_hit,
    sign_residual_arcsec)`, `rows` de-duplicated on the `Image` column."""
    pdf = pdfplumber.open(pdf_path)
    rows, pages_hit, misses = [], 0, 0
    for page in pdf.pages:
        # Table C1 is one contiguous portrait block near the front; stop
        # once it has ended rather than decoding the 400+ landscape D1 pages.
        if pages_hit and misses > 15:
            break
        chars = page.chars
        if not chars or sum(not c.get("upright", True) for c in chars) > 100:
            misses += 1
            continue
        text = page.extract_text() or ""
        if not text:
            misses += 1
            continue
        lines = text.split("\n")
        mapping, score = _tablec1_best_map(lines, _tablec1_font_maps(page))
        if not mapping or score == 0:
            misses += 1
            continue
        hit = 0
        for line in lines:
            m = _TABLEC1_ROW.search(_tablec1_decode(line, mapping))
            if not m:
                continue
            g = m.groups()
            rows.append([g[0], g[1], float(g[2]), float(g[4]), float(g[5]),
                         float(g[7]), float(g[8]), float(g[9]), float(g[10]),
                         float(g[11])])
            hit += 1
        if hit:
            pages_hit += 1
            misses = 0
        else:
            misses += 1

    rows, sep = _tablec1_repair_signs(rows)
    seen, uniq = set(), []
    for r in rows:
        if r[1] in seen:
            continue
        seen.add(r[1])
        uniq.append(r)
    return uniq, pages_hit, sep


def _write_uwish2_tablec1(dest_dir):
    """Extracts UWISH2 Table C1 from the raw PDF appendix and writes
    `uwish2/uwish2_tablec1.csv`. Returns `(out_path, n_rows)`."""
    pdf_path = f"{dest_dir}/uwish2/appendix_large.pdf"
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(
            "h2_knot_surveys: hand-placed input missing -- acquire it by hand "
            f"(see this module's docstring) and place it at {pdf_path!r}")
    out_path = f"{dest_dir}/uwish2/uwish2_tablec1.csv"
    uniq, pages_hit, sep = _extract_uwish2_tablec1(pdf_path)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_TABLEC1_COLS)
        w.writerows(uniq)
    median_sep = float(np.median(sep)) if sep.size else float("nan")
    max_sep = float(sep.max()) if sep.size else float("nan")
    print("h2_knot_surveys uwish2 Table C1: pages_with_C1=%d rows=%d "
          "sign-repair residual median %.3f\" max %.2f\""
          % (pages_hit, len(uniq), median_sep, max_sep))
    return out_path, len(uniq)


def build(config, regions=None):
    """Fetches the four CDS-reachable knot-survey catalogues into
    `<data_root>/sky/download/h2_knot_surveys/`, requires the three
    hand-placed acquisitions (Walawender: no CDS entry; UWISH2 Table D1's
    transcription and the PDF appendix Table D1 and C1 come from) there,
    and extracts UWISH2's own Table C1 (the survey's image footprint)
    from that PDF into `uwish2/uwish2_tablec1.csv`. `regions` is accepted
    for interface uniformity and ignored: this is a survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/h2_knot_surveys"
    with progress_module.Stage("sky.download.h2_knot_surveys") as st:
        n_fetched = _fetch_cds(dest_dir, st)
        n_manual = _require_manual(dest_dir)
        _c1_path, n_c1 = _write_uwish2_tablec1(dest_dir)
        print(f"h2_knot_surveys build: {n_fetched} files fetched from CDS, "
              f"{n_manual} hand-placed files present, "
              f"{n_c1} UWISH2 Table C1 rows extracted")
        st.done(dest_dir, cds_files=n_fetched, manual_files=n_manual, tablec1_rows=n_c1)


if __name__ == "__main__":
    run(build)
