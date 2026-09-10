"""The per-source 50%-completeness detection limits (SPEC_PRIORS.md
section 1.3; SPEC_BMSTP_DRAFT.md section 3.3).

`limits(config, region)` is the only limits reader in the package: every
consumer of a source's detection limit -- the likelihood's non-detection
terms, the field-star retention, the depth groups, every class build --
calls this function, never `DCOMP90_MJY`, `depths.py`'s products or
`depth_grid.py`'s products directly, so a single fit reaches every
consumer identically.

The values come from `catalog.depth_grid`'s per-source product
(`limits_sesna_source__<Region>.hdf5`, `F_LIM_50_MJY`): the region's
counts-based turnover, fit on the catalogued sources' own absolute
fluxes rather than on each source's map value, shifted to each source's
own `DCOMP90` for the five Spitzer bands, and the region's constant 2MASS
50% flux for the three 2MASS bands (`catalog/depth_grid.py`'s module
docstring). This is the same fit `bmstp.atlas` reads for the pixel grid,
so the likelihood, the retention and the atlas price the same
completeness (`depths.py`'s per-source `DELTA_DEX` rule is report-only).
"""

import numpy as np
import h5py

from sesnaimpute import config as config_module

IRAC_MIPS_KEYS = ("I1", "I2", "I3", "I4", "M1")
TWOMASS_KEYS = ("J", "H", "Ks")


def limits(config, region):
    """Returns `(n, 8)` mJy detection limits for every source in
    `region`, in `definitions.BANDS` order, reading `catalog.depth_grid`'s
    per-source limits product (`depth_grid.build`).
    """
    path = config_module.product_path(config, "catalog", "sesna", "limits", "source", region=region)
    with h5py.File(path, "r") as f:
        return np.asarray(f["F_LIM_50_MJY"][:], dtype=np.float64)
