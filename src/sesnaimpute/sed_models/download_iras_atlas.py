"""
sesnaimpute/sed_models/download_iras_atlas.py
====================================================================
ONE download line: the IRAS 60 and 100 um fluxes of the Brown et al. 2014
atlas galaxies (the far-infrared anchors galaxy_curate extends the Brown
spectra with; their spectra end at 35 um).

Canonical invocation::

    python3.9 -m sesnaimpute.sed_models.download_iras_atlas <root.cfg path, or a sed_models data-root directory>

For each galaxy in the atlas summary catalogue (129 positions), a VizieR
cone search (CDS mirror vizier.cds.unistra.fr) within 90 arcsec of the
atlas position against three IRAS catalogues:

    PSC   II/125        Point Source Catalog              Fnu_60/100, q_Fnu_60/100
    FSC   II/156A       Faint Source Catalog              Fnu60/100,  q_Fnu60/100
    RBGS  J/AJ/126/1607 Revised Bright Galaxy Sample      F60um/F100um, u_F60um/u_F100um
                        (Sanders et al. 2003; integrated galaxy totals)

and keeps, per catalogue, the NEAREST match with its separation, name,
the 60 and 100 um fluxes (Jy) and their quality (PSC, FSC: 1 upper limit,
2 moderate, 3 high) or limit flag (RBGS). Raw values only; which one anchors
a galaxy is galaxy_curate.choose_iras_anchor's decision.

Output, ONE small table, written once:

    <sed_models>/downloads/galaxy/brown2014_galsedatlas/iras_fluxes.csv

A present file is final: the line returns without querying (never
re-query). UNITS: *_F60_JY, *_F100_JY are Jy; *_SEP_ARCSEC arcsec;
empty field = no match / no value. READING: one row per atlas galaxy in
the summary catalogue's order; PSC_Q60/Q100 and FSC_Q60/Q100 are the
catalogues' own 1-3 quality integers (1 = upper limit); RBGS_U60/U100 is
":" when the RBGS value is flagged confusion-uncertain, empty otherwise.
"""

import argparse
import csv
import os
import sys

import numpy as np

from sesnaimpute.sed_models import paths

SUMMARY_CSV = os.path.join("catalogs", "hlsp_galsedatlas_multi_multi_summary_multi_v1_cat.csv")
OUT_NAME = "iras_fluxes.csv"
VIZIER_SERVER = "vizier.cds.unistra.fr"
RADIUS_ARCSEC = 90.0

# catalogue key -> (VizieR id, id column, {output field: VizieR column})
CATALOGUES = {
    "PSC": ("II/125", "IRAS",
            {"F60": "Fnu_60", "Q60": "q_Fnu_60", "F100": "Fnu_100", "Q100": "q_Fnu_100"}),
    "FSC": ("II/156A", "IRAS",
            {"F60": "Fnu60", "Q60": "q_Fnu60", "F100": "Fnu100", "Q100": "q_Fnu100"}),
    "RBGS": ("J/AJ/126/1607", "Name",
             {"F60": "F60um", "U60": "u_F60um", "F100": "F100um", "U100": "u_F100um"}),
}
FIELDS = {"PSC": ("F60", "Q60", "F100", "Q100"), "FSC": ("F60", "Q60", "F100", "Q100"),
          "RBGS": ("F60", "U60", "F100", "U100")}


CHUNK = 15


def _query(vizier, coords, cat, tries=4):
    import time
    from astropy import units as u
    for k in range(tries):
        try:
            return vizier.query_region(coords, radius=RADIUS_ARCSEC * u.arcsec, catalog=cat)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def downloads_dir():
    return os.path.join(paths.input_dir("sed_models"), "downloads", "galaxy",
                        "brown2014_galsedatlas")


def _cell(v):
    if np.ma.is_masked(v):
        return ""
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6g}"
    return str(v).strip()


def read_atlas_positions(directory):
    with open(os.path.join(directory, SUMMARY_CSV)) as f:
        rows = [{k: v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return [(r["Name"].lower().replace(" ", "-"), r["Name"], float(r["RA(J2000)"]),
             float(r["Decl(J2000)"]), r["cz"]) for r in rows]


def fetch_iras_table():
    """Query once, write iras_fluxes.csv; returns its path. A present file is
    returned untouched with no query."""
    directory = downloads_dir()
    out_path = os.path.join(directory, OUT_NAME)
    if os.path.exists(out_path):
        return out_path
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    Vizier.VIZIER_SERVER = VIZIER_SERVER

    atlas = read_atlas_positions(directory)
    coords = SkyCoord([a[2] for a in atlas] * u.deg, [a[3] for a in atlas] * u.deg)
    best = {key: {} for key in CATALOGUES}   # key -> galaxy index -> (sep, row)
    for key, (cat, idcol, colmap) in CATALOGUES.items():
        for lo in range(0, len(atlas), CHUNK):
            v = Vizier(columns=["**", "_r"], row_limit=-1)
            result = _query(v, coords[lo:lo + CHUNK], cat)
            if not len(result):
                continue
            t = result[0]
            # VizieR's _r column carries its unit in the column, arcsec by default
            scale = u.Unit(t["_r"].unit if t["_r"].unit is not None else "arcsec").to(u.arcsec)
            for row in t:
                i = lo + int(row["_q"]) - 1      # _q is 1-based into this chunk
                sep = float(row["_r"]) * scale
                if i not in best[key] or sep < best[key][i][0]:
                    best[key][i] = (sep, row)
    header = ["SLUG", "NAME", "RA_DEG", "DEC_DEG", "CZ_KMS"]
    for key in CATALOGUES:
        header += [f"{key}_ID", f"{key}_SEP_ARCSEC"]
        for f in FIELDS[key]:
            header.append(f"{key}_{f}" + ("_JY" if f.startswith("F") else ""))
    rows = []
    for i, (slug, name, ra, dec, cz) in enumerate(atlas):
        r = [slug, name, ra, dec, cz]
        for key, (cat, idcol, colmap) in CATALOGUES.items():
            if i in best[key]:
                sep, row = best[key][i]
                r += [_cell(row[idcol]), f"{sep:.2f}"] + [_cell(row[colmap[f]]) for f in FIELDS[key]]
            else:
                r += [""] * (2 + len(FIELDS[key]))
        rows.append(r)
    real = os.path.realpath(directory)
    assert real.startswith(os.path.realpath(paths.input_dir("sed_models"))), real
    tmp = out_path + ".part"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, out_path)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python3.9 -m sesnaimpute.sed_models.download_iras_atlas",
        description="Download the IRAS 60/100 um fluxes of the Brown 2014 atlas galaxies.")
    parser.add_argument("root", help="root.cfg path, or the sed_models data-root directory")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    from sesnaimpute.sed_models import build_galz
    build_galz.ingest(args.root)
    print(fetch_iras_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
