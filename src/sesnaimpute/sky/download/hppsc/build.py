"""The Herschel/PACS Point Source Catalogue (HPPSC; Marton et al. 2017, arXiv:1705.05693).

The three band lists (IRSA `ppsc_70`, `ppsc_100`, `ppsc_160`: blue, green, red) and the observation table
(`ppsc_obstbl`, the paper's Table 8, which maps it was built from), whole and verbatim as CSV, from the IRSA TAP
service (`sky.download._tap.query_csv`, paged on the row key `cntr`). Flux in mJy from a 6 / 6 / 12 arcsec aperture,
aperture-corrected, not colour-corrected; positions on the Herschel pointing (1.2 arcsec, Sect. 5.1.3).

VALIDATION-ONLY (planner ruling, owner-approved 2026-10-29): the PACS source list is the pipeline's own detection;
HPPSC is the external check of its positions, fluxes and recovery (`sky.derived.pacs.checks.hppsc_checks`). It
enters no product.
"""

import os

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._tap import query_csv

#: dest file -> IRSA table
TABLES = {"ppsc_70.csv": "ppsc_70", "ppsc_100.csv": "ppsc_100", "ppsc_160.csv": "ppsc_160", "ppsc_obstbl.csv": "ppsc_obstbl"}
#: the band lists' columns (Marton et al. 2017 Table 7); `cntr` first, the paging key
COLUMNS = ("cntr", "name", "band", "ra", "dec", "raerr", "decerr", "flux", "snr", "snrnoise", "stn", "rms", "strn",
           "fratio", "fwhmxfit", "fwhmyfit", "flag_elong", "edgeflag", "flag_blend", "warmat", "obsid", "ssomapflag")


def download_dir(config):
    return f"{config.data_root}/sky/download/hppsc"


def build(config, regions=None):
    """Fetches the HPPSC band lists and observation table to `sky/download/hppsc/` (present files are kept).
    `regions` is accepted for interface uniformity and ignored: a survey-wide product."""
    dest = download_dir(config)
    os.makedirs(dest, exist_ok=True)
    with progress_module.Stage("sky.download.hppsc") as st:
        for i, (name, table) in enumerate(TABLES.items()):
            path = f"{dest}/{name}"
            if table == "ppsc_obstbl":
                n, nb = _whole(table, path)
            else:
                n, nb = query_csv(table, COLUMNS, "cntr", path)
            print(f"sky.download.hppsc: {table} -> {path}: {n} rows, {nb} bytes", flush=True)
            st.tick(i + 1, len(TABLES), "tables")
        st.done(dest, files=len(TABLES))


def _whole(table, path):
    """A small table (the observation table) in one query."""
    import urllib.parse
    import urllib.request
    from sesnaimpute.sky.download._tap import SYNC_URL
    if os.path.exists(path):
        print(f"tap: {path} present, skipped")
        return sum(1 for _ in open(path)) - 1, os.path.getsize(path)
    params = urllib.parse.urlencode({"QUERY": f"SELECT * FROM {table}", "FORMAT": "csv", "LANG": "ADQL"}).encode()
    with urllib.request.urlopen(SYNC_URL, data=params, timeout=600) as resp:
        text = resp.read().decode("utf-8")
    with open(path, "w") as out:
        out.write(text)
    return len(text.splitlines()) - 1, os.path.getsize(path)


if __name__ == "__main__":
    run(build)
