"""The Edenhofer et al. 2023/2024 3-D dust map and its extinction curve.

Edenhofer G. et al. 2024, A&A 685, A82, "A parsec-scale galactic 3D dust
map out to 1.25 kpc from the Sun" (Zenodo 10.5281/zenodo.8187943, the
primary map and the auxiliary less-data validation map, both HEALPix
nside-256 posterior mean/std extinction-density cubes), together with the
Zhang, Green & Rix (2023, "ZGR23") extinction curve it is calibrated
against (Zenodo 10.5281/zenodo.7692680). Fetches both FITS cubes and the
curve ascii file verbatim, plus -- while `sky.derived.edenhofer_samples`
(repair-list row 8) still has a region left to do -- the two much larger
posterior-samples cubes that stage needs and nothing else reads; once
every region's `SIGMA_SAMPLES_K` product exists, the samples are no
longer fetched (they are an intermediate for that stage, not a product
of their own, and the coordinator deletes them once it has run).

Feeds SPEC_PRIORS.md section 1.4 (profiles; region distance, depth and
pedestal from the emission-anchored extinction profile).
"""

import os

from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
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

# The 12 posterior samples behind the mean/std layers above (19.5/24.7 GB):
# optional, fetched only while `sky.derived.edenhofer_samples` still needs
# them (`_samples_needed` below).
_OPTIONAL_SAMPLE_FILES = {
    "samples_healpix.fits": "https://zenodo.org/api/records/8187943/files/samples_healpix.fits/content",
    "validation_with_less_data_but_2kpc_samples_healpix.fits": (
        "https://zenodo.org/api/records/8187943/files/"
        "validation_with_less_data_but_2kpc_samples_healpix.fits/content"
    ),
}


def _samples_needed(config):
    """False once every region's `SIGMA_SAMPLES_K` product
    (`sky.derived.edenhofer_samples`) exists: the samples are only that
    stage's own intermediate, so nothing left needs them fetched."""
    return not all(
        os.path.exists(config_module.product_path(
            config, "sky/derived", "edenhofer", "profile-sigma-samples", "sightline", region=r.name))
        for r in regions_module.REGIONS)


def build(config, regions=None, _limit=None):
    """Fetches the two 3-D dust map FITS cubes and the ZGR23 extinction
    curve, verbatim, plus the two posterior-samples cubes while
    `sky.derived.edenhofer_samples` still needs them (module docstring).
    `regions` is accepted for interface uniformity and ignored: this is
    a survey-wide product. `_limit` is a rehearsal knob: when set, only
    the first `_limit` files (smallest first) of the required set are
    fetched.
    """
    dest_dir = f"{config.data_root}/sky/download/edenhofer2023"
    names = list(_FILES)
    if _limit is not None:
        names = names[:_limit]
    if _samples_needed(config):
        names = names + list(_OPTIONAL_SAMPLE_FILES)
    urls = {**_FILES, **_OPTIONAL_SAMPLE_FILES}
    with progress_module.Stage("sky.download.edenhofer2023") as st:
        for i, name in enumerate(names):
            fetch(urls[name], f"{dest_dir}/{name}")
            st.tick(i + 1, len(names), "files")
        st.done(dest_dir, files=len(names))


if __name__ == "__main__":
    run(build)
