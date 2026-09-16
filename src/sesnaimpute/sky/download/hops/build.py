"""The HOPS and eHOPS Herschel-confirmed protostar catalogues.

Furlan E. et al. 2016, "The Herschel Orion Protostar Survey: Fitting the
Spectral Energy Distributions of Class 0 and I Protostars", ApJS 224, 5
(VizieR `J/ApJS/224/5`, table1, 330 protostars in Orion); Pokhrel R. et al.
2023, "Extension of HOPS out to 500 pc (eHOPS). I. Aquila", ApJS 266, 32
(VizieR `J/ApJS/266/32`, the `ehops` table, 172 protostars in Aquila).
Fetches each catalogue's one VOTable verbatim, byte-for-byte.

Report-only, VALIDATION-ONLY (SPEC_BMSTP_DRAFT.md sec. 5.5, sec. 9):
neither catalogue enters the prior or the posterior; `sky.derived.protostars`
and `atlas.protostars` read them for an independent overlay check.

The two files already exist verbatim at `archive/sky_pre_wave2/hops/download/`
in the data root (its `MANIFEST.md` records the exact VizieR request and
table names this module's URLs reproduce); this module copies them from
there byte-for-byte when present, and otherwise fetches from VizieR
(`sesnaimpute.sky.download._fetch.fetch`).
"""

import os
import shutil

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

_VIZIER_BASE = "https://vizier.cds.unistra.fr/viz-bin/votable"

#: dest file name -> the VizieR `-source` table this module's request
#: names (verified against the archive's MANIFEST.md: HOPS's protostar
#: classification/fit-parameter table is `table1`; eHOPS's is the whole-
#: catalogue table named `ehops`, not `table1`).
_FILES = {
    "J_ApJS_224_5.vot": "J/ApJS/224/5/table1",
    "J_ApJS_266_32.vot": "J/ApJS/266/32/ehops",
}


def build(config, regions=None):
    """Copies (or, failing that, fetches) the HOPS and eHOPS VOTables to
    `sky/download/hops/`. `regions` is accepted for interface uniformity
    and ignored: this is a survey-wide product.
    """
    dest_dir = f"{config.data_root}/sky/download/hops"
    # The archive's own verbatim copy of both VOTables (MANIFEST.md), used
    # when present so the build is a copy, not a re-fetch.
    archive_dir = f"{config.data_root}/archive/sky_pre_wave2/hops/download"
    os.makedirs(dest_dir, exist_ok=True)
    with progress_module.Stage("sky.download.hops") as st:
        for i, (name, source_table) in enumerate(_FILES.items()):
            dest_path = f"{dest_dir}/{name}"
            archive_path = f"{archive_dir}/{name}"
            if os.path.exists(dest_path):
                print(f"sky.download.hops: {dest_path} present, skipped")
            elif os.path.exists(archive_path):
                shutil.copyfile(archive_path, dest_path)
                print(f"sky.download.hops: {archive_path} -> {dest_path} "
                      f"({os.path.getsize(dest_path)} bytes)")
            else:
                url = f"{_VIZIER_BASE}?-source={source_table}&-out.max=unlimited"
                fetch(url, dest_path)
            st.tick(i + 1, len(_FILES), "files")
        st.done(dest_dir, files=len(_FILES))


if __name__ == "__main__":
    run(build)
