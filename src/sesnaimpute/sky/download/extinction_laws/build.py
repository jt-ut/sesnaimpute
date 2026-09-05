"""The two extinction-law curves SPEC_PRIORS.md section 1.3's hybrid
`kappa_i(a)` blends between.

Each curve is a `(wavelength_um, opacity_cm2_per_g)` table (`<law>.par`)
plus a sibling `<law>.info` file naming which columns of the `.par` hold
wavelength and opacity. Both were assembled for the SED fitter and have
lived, unversioned, in the predecessor package's data tree ever since --
there is no reachable download URL for either; this module copies the two
files each law ships verbatim, byte-for-byte, from that tree, the same
pattern `sky.download.fazio2004.build` uses for another URL-less source.

Citations, transcribed from that tree's own `extinction/README.md`:

    `draine_rv3.1` is the Weingartner & Draine (2001) / Draine (2003)
    R_V=3.1 carbonaceous-silicate dust model, downloaded from Draine's own
    site and reduced to [wavelength, K_ext] (K_ext = K_abs / (1 - albedo))
    with wavelength sorted ascending.

    `whitney.r550` is the dense-cloud curve of Indebetouw et al. (2005,
    ApJ 619, 931): a Kim, Martin & Hendry (1994) Galactic ISM grain model
    with the mid-infrared opacities adjusted to the IRAC measurements, as
    distributed with Robitaille's `using_the_models.ipynb` notebook and
    commonly called the Whitney law (owner, 2026-09-05). The R_V = 5.5
    tag is the notebook's file name; no R_V is stated in the file.

Both are used exactly as published, with no renormalisation.

Feeds SPEC_PRIORS.md section 1.3 (the diffuse/dense-cloud law blend
`kappa_i(a)`, `prior.selection.LAW_DIFFUSE`/`LAW_DENSE`).
"""

import os
import shutil

from sesnaimpute.build import run

#: Pre-staged location this module copies from -- there is no reachable
#: download URL for either law (see module docstring).
_STAGED_DIR = (
    "/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/sesna-complete/"
    "src/sesnacomplete/data/extinction"
)

#: The two laws SPEC_PRIORS.md 1.3 blends between, and the two files each
#: ships (the tabulated curve and its column key).
LAWS = ("draine_rv3.1", "whitney.r550")


def _files(law):
    return (f"{law}.par", f"{law}.info")


def build(config, regions=None, _limit=None):
    """Copies each law's `.par`/`.info` pair verbatim from `_STAGED_DIR`
    to `<data_root>/sky/download/extinction_laws/<law>/`. `regions` is
    accepted for interface uniformity and ignored: this is a survey-wide
    product. `_limit` is a rehearsal knob: when set, only the first
    `_limit` laws are copied.
    """
    laws = LAWS if _limit is None else LAWS[:_limit]
    n_files = 0
    n_bytes = 0
    for law in laws:
        dest_dir = f"{config.data_root}/sky/download/extinction_laws/{law}"
        os.makedirs(dest_dir, exist_ok=True)
        for name in _files(law):
            src_path = f"{_STAGED_DIR}/{law}/{name}"
            dest_path = f"{dest_dir}/{name}"
            shutil.copyfile(src_path, dest_path)
            file_bytes = os.path.getsize(dest_path)
            n_files += 1
            n_bytes += file_bytes
            print(f"extinction_laws build: {src_path} -> {dest_path} ({file_bytes} bytes)")
    print(f"extinction_laws build: {n_files} files, {n_bytes} bytes total, "
          f"{len(laws)} laws")


if __name__ == "__main__":
    run(build)
