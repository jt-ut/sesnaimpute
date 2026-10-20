"""Kryukova et al. 2014 (AJ 148, 11) the dependence of protostellar
luminosity on environment in the Cygnus-X star-forming complex: 2,007
protostar candidates with IRAC/MIPS/2MASS (or UKIDSS) photometry and the
spectral index `alpha`, VizieR J/AJ/148/11, `table3.dat`. PROTOSTARS
ONLY (no disk-bearing/Class II members are catalogued here). Fetches the
ReadMe and `table3.dat` verbatim from the CDS distribution into
`sky/download/kryukova2014/`.

Feeds the excess-selected young-star census (`sesnaimpute.sky.derived.
yso_census`), owner's 2026-10-20 ruling: ingested with `SELECTION =
"protostar"` (Cygnus X, protostars only) -- the law unit (not this
module) decides how a protostar-only census enters the region fit.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/AJ/148/11"
FILENAMES = ("ReadMe", "table3.dat")


def build(config, regions=None):
    """Fetches the ReadMe and `table3.dat` verbatim into
    `sky/download/kryukova2014/`. `regions` is accepted for interface
    uniformity and ignored: this is a survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/kryukova2014"
    with progress_module.Stage("sky.download.kryukova2014") as st:
        for name in FILENAMES:
            fetch(f"{BASE_URL}/{name}", f"{dest_dir}/{name}")
        st.done(dest_dir, files=len(FILENAMES))


if __name__ == "__main__":
    run(build)
