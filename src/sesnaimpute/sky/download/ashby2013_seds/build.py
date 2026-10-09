"""SEDS: the Spitzer Extended Deep Survey, Ashby et al. 2013.

Ashby M.L.N. et al. 2013, "SEDS: The Spitzer Extended Deep Survey.
Survey Design, Photometry, and Deep IRAC Source Counts", ApJ 769, 80.
VizieR J/ApJ/769/80. Fetches the ReadMe and the five full-depth fields'
per-source catalogues (`table7.dat.gz` .. `table11.dat.gz`, 1.46 deg^2
combined), verbatim, gzipped as distributed.

Feeds `population.gal.ashby_seds_cumulative` (SPEC_BMSTP sec 5.4 GAL, a
level cross-check beside the Fazio 2004 counts, review ledger C3).
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

_BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/769/80"
_FILES = ("ReadMe", "table1.dat", "table7.dat.gz", "table8.dat.gz", "table9.dat.gz",
          "table10.dat.gz", "table11.dat.gz")


def build(config, regions=None):
    """Fetches the ReadMe and the per-field SEDS source-count tables, verbatim.
    `regions` is accepted for interface uniformity and ignored: this is a
    survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/ashby2013_seds"
    with progress_module.Stage("sky.download.ashby2013_seds") as st:
        for i, name in enumerate(_FILES):
            fetch(f"{_BASE_URL}/{name}", f"{dest_dir}/{name}")
            st.tick(i + 1, len(_FILES), "files")
        st.done(dest_dir, files=len(_FILES))


if __name__ == "__main__":
    run(build)
