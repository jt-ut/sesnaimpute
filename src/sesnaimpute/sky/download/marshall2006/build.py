"""Marshall et al. 2006's 2MASS-red-giant 3-D extinction map.

Marshall D. J., Robin A. C., Reyle C., Schultheis M., Picaud S. 2006,
"Modelling the Galactic interstellar extinction distribution in three
dimensions", A&A 453, 635. VizieR J/A+A/453/635. Fetches the ReadMe and
the per-cell A_Ks(d) table (`table1.dat.gz`, one row per 15-arcmin
(l, b) cell, carrying A_Ks and its error at up to 33 distance bins)
verbatim, gzipped as distributed -- the same URL and bytes
briefs/round4_2026-10-20/C24_locate_red_tail.md's own investigation
pulled under c18_diag/c24_marshall_table1.dat.gz /
c24_marshall_ReadMe.txt.

Feeds briefs/round4_2026-10-20/C25_far_column.md: the far-field column a
star beyond the Edenhofer 2023 cube's edge sees, on every admitted
sightline this map covers (|l| <= 100, |b| <= 10).
"""

from sesnaimpute.build import run
from sesnaimpute import progress as progress_module
from sesnaimpute.sky.download._fetch import fetch

_BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/453/635"
_FILES = ("ReadMe", "table1.dat.gz")


def build(config, regions=None):
    """Fetches the ReadMe and the per-cell A_Ks(d) table, verbatim.
    `regions` is accepted for interface uniformity and ignored: this is a
    survey-wide product, one file pair, never one per region."""
    dest_dir = f"{config.data_root}/sky/download/marshall2006"
    with progress_module.Stage("sky.download.marshall2006") as st:
        for i, name in enumerate(_FILES):
            fetch(f"{_BASE_URL}/{name}", f"{dest_dir}/{name}")
            st.tick(i + 1, len(_FILES), "files")
        st.done(dest_dir, files=len(_FILES))


if __name__ == "__main__":
    run(build)
