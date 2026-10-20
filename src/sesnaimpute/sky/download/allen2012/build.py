"""Allen et al. 2012 (ApJ 750, 125) Spitzer imaging of the Cepheus OB3b
cluster: young stellar objects classified by SED shape into class I,
class II, transition disk and class III, VizieR J/ApJ/750/125 (four
tables: `table1.dat` class I, 56 rows; `table2.dat` class II, 1,014 rows;
`table3.dat` transition disk, 65 rows; `table4.dat` class III, 1,440
rows). Fetches the ReadMe and all four tables verbatim from the CDS
distribution into `sky/download/allen2012/`.

Feeds the excess-selected young-star census (`sesnaimpute.sky.derived.
yso_census`), owner's 2026-10-20 ruling: a region (Cepheus OB3b) fitted
directly on its own published census of DISK-BEARING members (class I,
class II, transition disk -- class III carries no excess and is not
ingested) rather than the pooled coefficient.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/750/125"
FILENAMES = ("ReadMe", "table1.dat", "table2.dat", "table3.dat", "table4.dat")


def build(config, regions=None):
    """Fetches the ReadMe and the four class tables verbatim into
    `sky/download/allen2012/`. `regions` is accepted for interface
    uniformity and ignored: this is a survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/allen2012"
    with progress_module.Stage("sky.download.allen2012") as st:
        for i, name in enumerate(FILENAMES):
            fetch(f"{BASE_URL}/{name}", f"{dest_dir}/{name}")
            st.tick(i + 1, len(FILENAMES), "files")
        st.done(dest_dir, files=len(FILENAMES))


if __name__ == "__main__":
    run(build)
