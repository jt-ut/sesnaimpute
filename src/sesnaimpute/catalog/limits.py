"""The per-source 50%-completeness detection limits (SPEC_PRIORS.md
section 1.3).

`limits(config, region)` is the only limits reader in the package: every
consumer of a source's detection limit -- the selection test, the depth
groups, every class build -- calls this function, never `DCOMP90_MJY` or
`depths.py`'s products directly, so a single rescaling reaches every
consumer identically.

For the five Spitzer bands (I1, I2, I3, I4, M1), the limit is the
source's own `DCOMP90` rescaled by its region-band offset:
`F_lim,50 = DCOMP90 * 10**(-Delta)`. For the three 2MASS bands (J, H, Ks),
which carry no per-source completeness map, the limit is the region's
`F_50` flux directly, alike for every source in the region.
"""

import numpy as np
import h5py

from sesnaimpute import config as config_module
from sesnaimpute import definitions

IRAC_MIPS_KEYS = ("I1", "I2", "I3", "I4", "M1")
TWOMASS_KEYS = ("J", "H", "Ks")


def limits(config, region):
    """Returns `(n, 8)` mJy detection limits for every source in
    `region`, in `definitions.BANDS` order, reading the curated catalogue
    (`curated.build`) and the survey-depths product (`depths.build`).
    """
    band_keys = [b.key for b in definitions.BANDS]

    curated_path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region
    )
    with h5py.File(curated_path, "r") as f:
        dcomp90 = f["DCOMP90_MJY"][:]
        curated_bands = [b.decode() if isinstance(b, bytes) else b for b in f.attrs["BANDS"]]

    depths_path = config_module.product_path(config, "catalog", "sesna", "depths", "region")
    with h5py.File(depths_path, "r") as f:
        depth_regions = [r.decode() if isinstance(r, bytes) else r for r in f["REGION"][:]]
        if region not in depth_regions:
            raise ValueError(f"catalog.limits: {region!r} not in {depths_path!r}:/REGION")
        ridx = depth_regions.index(region)
        delta_dex = f["DELTA_DEX"][ridx, :]
        f50_2mass = f["F50_2MASS_MJY"][ridx, :]

    n = dcomp90.shape[0]
    out = np.empty((n, len(band_keys)), dtype=np.float64)
    for j, key in enumerate(band_keys):
        cb = curated_bands.index(key)
        if key in TWOMASS_KEYS:
            out[:, j] = f50_2mass[TWOMASS_KEYS.index(key)]
        else:
            delta = delta_dex[IRAC_MIPS_KEYS.index(key)]
            out[:, j] = dcomp90[:, cb] * 10.0 ** (-delta)
    return out
