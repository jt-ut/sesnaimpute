"""The MIST v1.2 solar-metallicity, non-rotating basic isochrone set.

Choi J., Dotter A., Conroy C., Cantiello M., Paxton B., Johnson B. D.
2016, "Mesa Isochrones and Stellar Tracks (MIST). I. Solar-scaled Models",
ApJ 823, 102. Packaged-model distribution:
https://waps.cfa.harvard.edu/MIST/model_grids.html. Fetches the
`[Fe/H] = 0.00`, `v/vcrit = 0.0` basic isochrone tarball, unpacks the one
solar-metallicity file it carries, and discards the tarball (the other
metallicities in the same archive are not this design's isochrone).

Extends the pre-main-sequence mass-luminosity relation past BHAC15's
1.4 M⊙ top (`sesnaimpute.population.yso_mass`), which SPEC_BMSTP_DRAFT.md
sections 3.5 and 10 need for YSO templates whose luminosity exceeds
BHAC15's covered range.
"""

import os
import tarfile

from sesnaimpute import progress
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

_URL = "https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.0_basic_isos.txz"
_MEMBER = "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_basic.iso"


def build(config, regions=None):
    """Fetches the MIST v1.2 basic isochrone tarball, unpacks the one
    solar-metallicity file it carries to `sky/download/mist2016/`, and
    deletes the tarball. `regions` is accepted for interface uniformity
    and ignored: this is a survey-wide product. Skips the fetch and the
    unpack if the target file is already present (rule 5a).
    """
    with progress.Stage("sky.download.mist2016") as st:
        dest_dir = f"{config.data_root}/sky/download/mist2016"
        dest_file = f"{dest_dir}/{_MEMBER}"
        if os.path.exists(dest_file):
            st.done(dest_file, skipped=1)
            return
        archive_path = f"{dest_dir}/{os.path.basename(_URL)}"
        fetch(_URL, archive_path)
        with tarfile.open(archive_path, "r:xz") as tf:
            member = next(m for m in tf.getmembers() if os.path.basename(m.name) == _MEMBER)
            member.name = os.path.basename(member.name)
            tf.extract(member, path=dest_dir)
        os.remove(archive_path)
        st.done(dest_file, skipped=0)


if __name__ == "__main__":
    run(build)
