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

    `whitney.r550` is a Kim, Martin & Hendry (1994) Galactic ISM curve
    with mid-IR properties modified per Indebetouw et al. (2005); it was
    distributed with Robitaille's `using_the_models.ipynb` notebook.

Both are used exactly as published, with no renormalisation. Note the
`whitney.r550` attribution is less certain than `draine_rv3.1`'s: the law
directory's own `.README` only *guesses* an origin paper ("I believe this
law came from..."), and neither source file states an R_V value -- "r550"
is inferred, not published, to mean R_V=5.5. Neither file names Whitney et
al. 2003. This module carries the tree's own citation as written; it does
not strengthen it.

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
