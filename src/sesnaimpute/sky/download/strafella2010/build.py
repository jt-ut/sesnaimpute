"""Strafella et al. 2010 (ApJ 719, 9) the Spitzer-IRAC point-source
catalog of the Vela-D cloud: a photometric catalog of 170,299 point
sources, with a 637-candidate IRAC YSO catalogue (`table9.dat`, positions
in decimal degrees, IRAC/MIPS flux densities and the spectral slope
`alpha` for 2-20 micron), VizieR J/ApJ/719/9. Fetches the ReadMe and
`table9.dat` verbatim from the CDS distribution into
`sky/download/strafella2010/`.

Feeds the excess-selected young-star census (`sesnaimpute.sky.derived.
yso_census`), owner's 2026-10-20 ruling: a region (Vela D) fitted
directly on its own published census rather than the pooled coefficient.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/719/9"
FILENAMES = ("ReadMe", "table9.dat")


def build(config, regions=None):
    """Fetches the ReadMe and `table9.dat` verbatim into
    `sky/download/strafella2010/`. `regions` is accepted for interface
    uniformity and ignored: this is a survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/strafella2010"
    with progress_module.Stage("sky.download.strafella2010") as st:
        for name in FILENAMES:
            fetch(f"{BASE_URL}/{name}", f"{dest_dir}/{name}")
        st.done(dest_dir, files=len(FILENAMES))


if __name__ == "__main__":
    run(build)
