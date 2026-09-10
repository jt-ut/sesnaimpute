"""The Edenhofer et al. 2023/2024 3-D dust map and its extinction curve.

Edenhofer G. et al. 2024, A&A 685, A82, "A parsec-scale galactic 3D dust
map out to 1.25 kpc from the Sun" (Zenodo 10.5281/zenodo.8187943, the
primary map and the auxiliary less-data validation map, both HEALPix
nside-256 posterior mean/std extinction-density cubes), together with the
Zhang, Green & Rix (2023, "ZGR23") extinction curve it is calibrated
against (Zenodo 10.5281/zenodo.7692680). Fetches both FITS cubes and the
curve ascii file verbatim.

Feeds SPEC_PRIORS.md section 1.4 (profiles; region distance, depth and
pedestal from the emission-anchored extinction profile). The much larger
posterior-samples cubes `sky.derived.edenhofer_samples` reads are a
manual one-off acquisition, not fetched here -- see the comment above
that stage's own RUNBOOKtp.sh line.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

# filename -> upstream URL, lifted from the old
# fetch_external.edenhofer.build.profile_inputs ACCEPTED_FILES table,
# smallest file first for the rehearsal knob below.
_FILES = {
    "zgr23_extinction_curve.txt": "https://zenodo.org/api/records/7811871/files/extinction_curve.txt/content",
    "mean_and_std_healpix.fits": "https://zenodo.org/api/records/8187943/files/mean_and_std_healpix.fits/content",
    "validation_with_less_data_but_2kpc_mean_and_std_healpix.fits": (
        "https://zenodo.org/api/records/8187943/files/"
        "validation_with_less_data_but_2kpc_mean_and_std_healpix.fits/content"
    ),
}


def build(config, regions=None, _limit=None):
    """Fetches the two 3-D dust map FITS cubes and the ZGR23 extinction
    curve, verbatim. `regions` is accepted for interface uniformity and
    ignored: this is a survey-wide product. `_limit` is a rehearsal
    knob: when set, only the first `_limit` files (smallest first) are
    fetched.
    """
    dest_dir = f"{config.data_root}/sky/download/edenhofer2023"
    names = list(_FILES)
    if _limit is not None:
        names = names[:_limit]
    with progress_module.Stage("sky.download.edenhofer2023") as st:
        for i, name in enumerate(names):
            fetch(_FILES[name], f"{dest_dir}/{name}")
            st.tick(i + 1, len(names), "files")
        st.done(dest_dir, files=len(names))


if __name__ == "__main__":
    run(build)
