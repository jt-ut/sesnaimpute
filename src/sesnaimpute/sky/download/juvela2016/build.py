"""The Juvela & Montillaud 2016 (A&A 585, A38) all-sky NICEST extinction
map: 2MASS star-colour extinction (`A_J`, model M1, 3.0' FWHM), HEALPix
nested NSIDE=2048, Galactic coordinates. Fetches the one FITS file
verbatim from the author distribution
(http://www.interstellarmedium.org/Extinction/data/).

Feeds W49's extinction column (`sesnaimpute.sky.derived.juvela_extinction`):
a star-colour map of the whole sightline, read against the adopted
(gas) column's beam to set the scale where the emission map runs low.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

URL = "http://www.interstellarmedium.org/Extinction/data/NICEST_AJ_M1_FWHM3.0.fits"
FILENAME = "NICEST_AJ_M1_FWHM3.0.fits"


def build(config, regions=None):
    """Fetches `NICEST_AJ_M1_FWHM3.0.fits` verbatim into
    `sky/download/juvela2016/`. `regions` is accepted for interface
    uniformity and ignored: this is a survey-wide product. A file
    already present is skipped (`_fetch.fetch`).
    """
    dest_path = f"{config.data_root}/sky/download/juvela2016/{FILENAME}"
    with progress_module.Stage("sky.download.juvela2016") as st:
        fetch(URL, dest_path)
        st.done(dest_path, files=1)


if __name__ == "__main__":
    run(build)
