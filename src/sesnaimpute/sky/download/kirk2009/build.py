"""Kirk et al. 2009 (ApJS 185, 198) the Spitzer survey of interstellar
clouds in the Gould belt. II. The Cepheus flare observed with IRAC and
MIPS: 133 young stellar object candidates (the paper's own count; the
VizieR tables carry 143 rows, note in DATA.md), VizieR J/ApJS/185/198.
Fetches the ReadMe, `table1.dat` (6 dark-cloud associations), `table6.dat`
(143 YSO candidates: positions via the Spitzer catalog name, IRAC/MIPS
fluxes) and `table7.dat` (143 rows, the SAME running index: derived
properties -- cloud, spectral-index slope `alpha` and class) verbatim
from the CDS distribution into `sky/download/kirk2009/`.

Feeds the excess-selected young-star census (`sesnaimpute.sky.derived.
yso_census`), owner's 2026-10-20 ruling: a region (Cepheus Flare) fitted
directly on its own published census rather than the pooled coefficient.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/185/198"
FILENAMES = ("ReadMe", "table1.dat", "table6.dat", "table7.dat")


def build(config, regions=None):
    """Fetches the ReadMe, `table1.dat`, `table6.dat` and `table7.dat`
    verbatim into `sky/download/kirk2009/`. `regions` is accepted for
    interface uniformity and ignored: this is a survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/kirk2009"
    with progress_module.Stage("sky.download.kirk2009") as st:
        for i, name in enumerate(FILENAMES):
            fetch(f"{BASE_URL}/{name}", f"{dest_dir}/{name}")
            st.tick(i + 1, len(FILENAMES), "files")
        st.done(dest_dir, files=len(FILENAMES))


if __name__ == "__main__":
    run(build)
