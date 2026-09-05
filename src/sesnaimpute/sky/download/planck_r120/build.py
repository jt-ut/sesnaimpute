"""The Planck 2013 R1.20 thermal-dust model map.

Planck Collaboration, "Planck 2013 results. XI. All-sky model of thermal
dust emission" (2014, A&A 571, A11), release R1.20. Fetches
`HFI_CompMap_ThermalDustModel_2048_R1.20.fits` (all-sky HEALPix nside-2048
TAU353, TEMP, BETA, RADIANCE) verbatim from the ESA Planck Legacy Archive.

Feeds SPEC_PRIORS.md section 1.1 (columns; the Planck-elsewhere branch of
the adopted dust column where Herschel does not cover a sightline).
"""

from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

_URL = (
    "http://pla.esac.esa.int/pla/aio/product-action?"
    "MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits"
)
_FILENAME = "HFI_CompMap_ThermalDustModel_2048_R1.20.fits"


def build(config, regions=None, _limit=None):
    """Fetches the R1.20 thermal-dust FITS map, verbatim. `regions` is
    accepted for interface uniformity and ignored: this is a survey-wide
    all-sky product. `_limit` is a rehearsal knob only: it is not used
    here since this source is already a single file.
    """
    dest_dir = f"{config.data_root}/sky/download/planck_r120"
    fetch(_URL, f"{dest_dir}/{_FILENAME}")


if __name__ == "__main__":
    run(build)
