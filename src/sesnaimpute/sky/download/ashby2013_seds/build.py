"""The SEDS galaxy source-catalogue tables, fetched verbatim from CDS.

Ashby M.L.N. et al. 2013, ApJ 769, 80 -- the Spitzer Extended Deep Survey
(SEDS): survey design, photometry, and deep IRAC source counts in five
extragalactic fields (UDS, ECDFS, COSMOS, HDFN, EGS). VizieR carries the
paper as `J/ApJ/769/80` and its directory is reachable.

Fetches the ReadMe and every file its own File Summary lists, verbatim,
into `<data_root>/sky/download/ashby2013_seds/`:

    ReadMe          this file
    table1.dat      the five SEDS fields
    table7.dat.gz   full-depth source catalog, UDS
    table8.dat.gz   full-depth source catalog, ECDFS
    table9.dat.gz   full-depth source catalog, COSMOS
    table10.dat.gz  full-depth source catalog, HDFN
    table11.dat.gz  full-depth source catalog, EGS

Feeds SPEC_PRIORS.md section 5 (GAL), via
`sesnaimpute.sky.derived.all_sky.ashby2013_seds` colour-distribution
product.
"""

from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

#: VizieR base URL for this catalogue.
_BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/769/80"

#: Every file the ReadMe's own File Summary lists.
_FILES = (
    "ReadMe",
    "table1.dat",
    "table7.dat.gz",
    "table8.dat.gz",
    "table9.dat.gz",
    "table10.dat.gz",
    "table11.dat.gz",
)


def build(config, regions=None):
    """Fetches the SEDS ReadMe and its five data tables verbatim from
    VizieR to `<data_root>/sky/download/ashby2013_seds/`. `regions` is
    accepted for interface uniformity and ignored: this is a survey-wide
    product.
    """
    dest_dir = f"{config.data_root}/sky/download/ashby2013_seds"
    for name in _FILES:
        fetch(f"{_BASE_URL}/{name}", f"{dest_dir}/{name}")
    print(f"ashby2013_seds build: {len(_FILES)} files fetched from CDS")


if __name__ == "__main__":
    run(build)
