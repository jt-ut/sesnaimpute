"""The Gaia DR3 open-cluster catalogue of Hunt & Reffert 2023.

Hunt E.L., Reffert S. 2023, "Improving the open cluster census. II. An
all-sky cluster catalogue with Gaia DR3", A&A 673, A114. VizieR
J/A+A/673/A114. Fetches the ReadMe and the main cluster table
(`clusters.dat.gz`, one row per cluster, carrying RAdeg, DEdeg, r50, rc,
rt, rtot) verbatim, byte-for-byte, gzipped as distributed.

Feeds SPEC_PRIORS.md section 2.1 (STAR: cluster exclusion mask, ratio and
catalogue test against the tidal radius `rt`).
"""

from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

_BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/673/A114"
_FILES = ("ReadMe", "clusters.dat.gz")


def build(config, regions=None):
    """Fetches the ReadMe and the main cluster table, verbatim. `regions`
    is accepted for interface uniformity and ignored: this is a survey-
    wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/hunt_reffert2023"
    for name in _FILES:
        fetch(f"{_BASE_URL}/{name}", f"{dest_dir}/{name}")


if __name__ == "__main__":
    run(build)
