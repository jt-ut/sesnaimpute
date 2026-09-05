"""The per-star GRAMS radiative-transfer fits of Riebel et al. 2012.

Riebel D., Srinivasan S., Sargent B., Meixner M. 2012, "The mass-loss
return from evolved stars to the Large Magellanic Cloud. VI. Luminosities
and mass-loss rates on population scales", ApJ 753, 71. VizieR
J/ApJ/753/71. Fetches the ReadMe and the per-star fit table
(`table3.dat.gz`, one row per AGB/RSG candidate, carrying the fitted
optical depth `tau` and the O-rich/C-rich chemistry classification `GCl`)
verbatim, gzipped as distributed.

Feeds SPEC_PRIORS.md section 3 (AGB: `F_dusty`).
"""

from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

_BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/753/71"
_FILES = ("ReadMe", "table3.dat.gz")


def build(config, regions=None):
    """Fetches the ReadMe and the per-star GRAMS fit table, verbatim.
    `regions` is accepted for interface uniformity and ignored: this is a
    survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/riebel2012"
    for name in _FILES:
        fetch(f"{_BASE_URL}/{name}", f"{dest_dir}/{name}")


if __name__ == "__main__":
    run(build)
