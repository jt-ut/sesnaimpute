"""Pulls the S-COSMOS IRAC catalogue (Sanders et al. 2007, ApJS 172, 86)
from IRSA's TAP service, as the deep single-field cross-check SPEC_PRIORS
section 5.1 names beside SWIRE. Download never computes: this module
selects position, the four-band aperture-2 fluxes and their uncertainties,
and the per-band data-quality flags plus the SExtractor flag -- the
closest this release ships to star-galaxy separation, since S-COSMOS's own
IRAC table carries no native stellarity index (S-L5) -- and writes them
verbatim, upstream column names kept, to one CSV.

Table name discovered against IRSA's TAP_SCHEMA: `scosmos_irac_0407`.
"""

import os

from sesnaimpute import build as build_module
from sesnaimpute.sky.download import _tap

TABLE = "scosmos_irac_0407"

# cntr (the page cursor) first, then position, the four-band 1.9" aperture
# fluxes with uncertainties (aperture 2 of 4, matching SWIRE's aperture-2
# pick), and the per-band/overall quality flags.
COLUMNS = (
    "cntr", "ra", "dec",
    "flux_c1_2", "err_c1_2",
    "flux_c2_2", "err_c2_2",
    "flux_c3_2", "err_c3_2",
    "flux_c4_2", "err_c4_2",
    "fl_c1", "fl_c2", "fl_c3", "fl_c4", "flag",
)


def build(config, regions=None):
    """Writes `sky/download/scosmos/scosmos_irac.csv`. `regions` is
    accepted for the standard `build` signature and ignored: S-COSMOS is
    one survey-level field, not tied to SESNA's own region list.
    """
    dest_dir = f"{config.data_root}/sky/download/scosmos"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = f"{dest_dir}/scosmos_irac.csv"
    n_rows, n_bytes = _tap.query_csv(TABLE, COLUMNS, "cntr", dest_path)
    print(f"scosmos build: {TABLE} -> {dest_path}: {n_rows} rows, {n_bytes} bytes")


if __name__ == "__main__":
    build_module.run(build)
