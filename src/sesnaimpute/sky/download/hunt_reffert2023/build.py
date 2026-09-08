"""The Gaia DR3 open-cluster catalogue of Hunt & Reffert 2023.

Hunt E.L., Reffert S. 2023, "Improving the open cluster census. II. An
all-sky cluster catalogue with Gaia DR3", A&A 673, A114. VizieR
J/A+A/673/A114. Fetches the ReadMe, the main cluster table
(`clusters.dat.gz`, one row per cluster, carrying RAdeg, DEdeg, r50, rc,
rt, rtot), and the member-star table (`members.dat`, one row per member
star of a catalogued cluster, carrying the cluster Name, Gaia DR3 source
id, membership probability `Prob`, RA/Dec, GLON/GLAT, and G/BP/RP
photometry -- the ReadMe's "Byte-by-byte Description of file:
members.dat"), each verbatim, byte-for-byte, in whatever compression the
archive currently serves it in (`clusters.dat.gz` gzipped; `members.dat`
plain -- the archive does not gzip it, unlike the cluster table).

Feeds SPEC_PRIORS.md section 2.1 (STAR: cluster exclusion mask, ratio and
catalogue test against the tidal radius `rt`) and SPEC_BMSTP_DRAFT.md
section 5.1's "clusters" row (STAR: cluster-member subtraction from the
observed anchor histogram, `population.anchor_observed`).
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

_BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/673/A114"
_FILES = ("ReadMe", "clusters.dat.gz", "members.dat")


def build(config, regions=None):
    """Fetches the ReadMe and the main cluster table, verbatim. `regions`
    is accepted for interface uniformity and ignored: this is a survey-
    wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/hunt_reffert2023"
    with progress_module.Stage("sky.download.hunt_reffert2023") as st:
        for i, name in enumerate(_FILES):
            fetch(f"{_BASE_URL}/{name}", f"{dest_dir}/{name}")
            st.tick(i + 1, len(_FILES), "files")
        st.done(dest_dir, files=len(_FILES))


if __name__ == "__main__":
    run(build)
