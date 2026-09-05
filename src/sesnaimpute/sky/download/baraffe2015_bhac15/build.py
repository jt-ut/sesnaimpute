"""The two remaining BHAC15 isochrone filter tables, 2MASS and Gaia.

Baraffe I., Homeier D., Allard F., Chabrier G. 2015, "New evolutionary
models for pre-main sequence and main sequence low-mass stars down to the
hydrogen-burning limit", A&A 577, A42. Author distribution, Lyon:
https://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/. Fetches
`BHAC15_iso.2mass` and `BHAC15_iso.GAIA` verbatim, beside the Spitzer
table (`BHAC15_iso.SPITZER`) already on disk from an earlier acquisition.
The directory lists the 2MASS file as `BHAC15_iso.2mass` (lower-case
extension), not `BHAC15_iso.2MASS`; fetched under that name.

Feeds SPEC_PRIORS.md section 2.1 (STAR: young-star subtraction) and
section 6.2 (YSO: selection in the 2MASS bands).
"""

from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

_BASE_URL = "https://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir"
_FILES = ("BHAC15_iso.2mass", "BHAC15_iso.GAIA")


def build(config, regions=None):
    """Fetches the 2MASS and Gaia BHAC15 isochrone tables, verbatim.
    `regions` is accepted for interface uniformity and ignored: this is a
    survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/baraffe2015_bhac15"
    for name in _FILES:
        fetch(f"{_BASE_URL}/{name}", f"{dest_dir}/{name}")


if __name__ == "__main__":
    run(build)
