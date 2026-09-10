"""Dunham et al. 2015 (ApJS 220, 11) Spitzer c2d + Gould Belt YSO census:
2,966 excess-selected YSOs in 18 nearby (125-950 pc) clouds, complete to
the hydrogen-burning limit in the nearby clouds. Fetches `ReadMe`,
`table1.dat` (18 clouds), `table2.dat` (2,966 YSOs, bolometric
properties), `table3.dat` (observed 2MASS/Spitzer flux densities) and
`table4.dat` (the same, extinction corrected) verbatim from the CDS
distribution into `sky/download/dunham2015/`.

Feeds the YSO brightness reference (`sesnaimpute.sky.derived.dunham_yso`):
the matched census the young-star law was fit to (Pokhrel et al. 2020),
so its dereddened 4.5 micron flux is the population density over
brightness `bmstp/sample_cloud.py`'s `sample_f45` predicts (sec 1.4).
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/220/11"
FILENAMES = ("ReadMe", "table1.dat", "table2.dat", "table3.dat", "table4.dat")


def build(config, regions=None):
    """Fetches the five files verbatim into `sky/download/dunham2015/`.
    `regions` is accepted for interface uniformity and ignored: this is a
    survey-wide product. A file already present is skipped (`_fetch.fetch`).
    """
    dest_dir = f"{config.data_root}/sky/download/dunham2015"
    with progress_module.Stage("sky.download.dunham2015") as st:
        for name in FILENAMES:
            fetch(f"{BASE_URL}/{name}", f"{dest_dir}/{name}")
        st.done(dest_dir, files=len(FILENAMES))


if __name__ == "__main__":
    run(build)
