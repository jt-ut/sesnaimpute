"""Rebull et al. 2010 (ApJS 186, 259) the Taurus Spitzer survey: new
candidate Taurus members selected using sensitive mid-infrared
photometry, VizieR J/ApJS/186/259. Fetches the ReadMe, `table6.dat` (215
previously-identified Taurus members' derived properties: position via
the Spitzer source name, our evolutionary `Class`) and `table7.dat` (148
new candidate members, the same columns) verbatim from the CDS
distribution into `sky/download/rebull2010/`.

Feeds the excess-selected young-star census (`sesnaimpute.sky.derived.
yso_census`), owner's 2026-10-20 ruling: a region (Taurus) fitted
directly on its own published census of members WITH excess (Class I,
flat or II -- Class III carries no excess and is not ingested) rather
than the pooled coefficient.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/186/259"
FILENAMES = ("ReadMe", "table6.dat", "table7.dat")


def build(config, regions=None):
    """Fetches the ReadMe, `table6.dat` and `table7.dat` verbatim into
    `sky/download/rebull2010/`. `regions` is accepted for interface
    uniformity and ignored: this is a survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/rebull2010"
    with progress_module.Stage("sky.download.rebull2010") as st:
        for i, name in enumerate(FILENAMES):
            fetch(f"{BASE_URL}/{name}", f"{dest_dir}/{name}")
            st.tick(i + 1, len(FILENAMES), "files")
        st.done(dest_dir, files=len(FILENAMES))


if __name__ == "__main__":
    run(build)
