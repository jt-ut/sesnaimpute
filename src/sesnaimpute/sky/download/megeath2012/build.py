"""Megeath et al. 2012 (AJ 144, 192) Spitzer survey of Orion A and B. I.:
a census of 3,479 dusty young stellar objects (YSOs) found by mid-IR
colour excess in the Orion A and B molecular clouds, with IRAC/MIPS/2MASS
photometry and a protostar-vs-disk classification. VizieR J/AJ/144/192.
Fetches the ReadMe and `table4.dat` (3,479 rows) verbatim from the CDS
distribution into `sky/download/megeath2012/`.

Feeds the excess-selected young-star census (`sesnaimpute.sky.derived.
yso_census`), owner's 2026-10-20 ruling: a second region (Orion A and B)
fitted directly on its own published census rather than the pooled
coefficient.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/AJ/144/192"
FILENAMES = ("ReadMe", "table4.dat")


def build(config, regions=None):
    """Fetches the ReadMe and `table4.dat` verbatim into
    `sky/download/megeath2012/`. `regions` is accepted for interface
    uniformity and ignored: this is a survey-wide product. A file
    already present is skipped (`_fetch.fetch`).
    """
    dest_dir = f"{config.data_root}/sky/download/megeath2012"
    with progress_module.Stage("sky.download.megeath2012") as st:
        for name in FILENAMES:
            fetch(f"{BASE_URL}/{name}", f"{dest_dir}/{name}")
        st.done(dest_dir, files=len(FILENAMES))


if __name__ == "__main__":
    run(build)
