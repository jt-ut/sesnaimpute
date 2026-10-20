"""Rebull et al. 2011 (ApJS 193, 25) the North American and Pelican
Nebulae (NAN) complex. II. MIPS observations, VizieR J/ApJS/193/25
(2,076 YSO candidates total: 1,286 MIPS-based, 806 more from IRAC alone)
with Guieu et al. 2009 (ApJ 697, 787) the NAN complex. I. IRAC
observations, VizieR J/ApJ/697/787 (the Paper I IRAC-only candidate list
Rebull 2011's own `cr5` flags against) -- the NAN excess-selected YSO
census, bundled here as one download module since they are two parts of
the same survey and Rebull 2011's own table4.dat cross-references
Guieu 2009's objects directly (its `cr5` criterion). Fetches both
catalogues' ReadMe and data files verbatim from the CDS distribution.

Feeds the excess-selected young-star census (`sesnaimpute.sky.derived.
yso_census`), owner's 2026-10-20 ruling: a region (North America Nebula)
fitted directly on its own published census rather than the pooled
coefficient.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

#: CDS catalogue -> (base URL, files its own ReadMe File Summary lists).
_CATALOGUES = {
    "rebull2011": (
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/193/25",
        ("ReadMe", "table3.dat", "table4.dat", "table8.dat", "table9.dat"),
    ),
    "guieu2009": (
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/697/787",
        ("ReadMe", "table4.dat"),
    ),
}


def build(config, regions=None):
    """Fetches both catalogues' ReadMe and data files, verbatim, into
    `sky/download/rebull2011/<catalogue>/`. `regions` is accepted for
    interface uniformity and ignored: this is a survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/rebull2011"
    all_names = [(cat, name) for cat, (_, files) in _CATALOGUES.items() for name in files]
    with progress_module.Stage("sky.download.rebull2011") as st:
        for i, (cat, name) in enumerate(all_names):
            base_url = _CATALOGUES[cat][0]
            fetch(f"{base_url}/{name}", f"{dest_dir}/{cat}/{name}")
            st.tick(i + 1, len(all_names), "files")
        st.done(dest_dir, files=len(all_names))


if __name__ == "__main__":
    run(build)
