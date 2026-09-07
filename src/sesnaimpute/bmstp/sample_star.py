"""The star-family population samples: STAR and AGB, one tile at a time
(SPEC_BMSTP_DRAFT.md sec. 5.1 "Marks", "Weight"; sec. 5.2 "Marks", "Weight";
IMPLEMENTATION_BMSTP_DRAFT.md sec. 3 row 1.3).

Reads `population.star_population`'s per-tile product
(`population/star/population_star_tile__R.hdf5`): one root group per tile,
`tile_<id>`, holding `U` (scaled extinction), `LOG10_B` (the STAR reference),
`LOG10_B_AGB_O`/`LOG10_B_AGB_C` (the AGB references, NaN off the evolved
subset), `W_STAR`/`W_AGB` (the reweighting of sec. 5.2), and `IS_EVOLVED`.
PAHC has no sampler of its own: it reads STAR's grid (`GRID_STAR`) unchanged.
"""

import h5py
import numpy as np

from sesnaimpute import config as config_module

_SKIP_KEYS = ("DIST_GRID", "LIMIT8_GRID_MJY")


def _path(config, region):
    return config_module.product_path(
        config, "population", "star", "population", "tile", region=region)


def tile_ids(config, region):
    """The region's tile ids (the root groups `tile_<id>` of the per-tile
    star population product), sorted -- one grain axis for P2, read once
    per region without loading any tile's arrays."""
    with h5py.File(_path(config, region), "r") as f:
        ids = sorted(int(name.split("_")[1]) for name in f.keys() if name not in _SKIP_KEYS)
    return np.array(ids, dtype=np.int64)


def sample_star(config, region, tile_id):
    """STAR's `(x, log10_b, w)` on tile `tile_id` (sec. 5.1 "Marks",
    "Weight"): `x = U`, `log10 B` against the STAR reference (`LOG10_B`),
    `w = W_STAR` (the field-star share of the reweighting, sec. 5.2)."""
    with h5py.File(_path(config, region), "r") as f:
        grp = f[f"tile_{tile_id}"]
        x = grp["U"][()].astype(np.float64)
        log10_b = grp["LOG10_B"][()].astype(np.float64)
        w = grp["W_STAR"][()].astype(np.float64)
    return x, log10_b, w


def sample_agb(config, region, tile_id):
    """AGB's `(x, log10_b, w)` on tile `tile_id` (sec. 5.2 "Marks",
    "Weight"): the evolved subset only (`LOG10_B_AGB_*` is NaN elsewhere),
    each star's `x = U` placed twice -- once at its O-rich reference, once
    at its C-rich reference -- weighted by the DUSTY subset's carbon
    share, `f_C F_dusty,C / (f_C F_dusty,C + (1 - f_C) F_dusty,O)` (sec.
    5.2, not the bare carbon fraction `F_C`: `W_AGB` is already the dusty
    subset, whose chemistry mix is `F_dusty` by chemistry, not `F_C`
    itself), from the population product's own `F_C`/`F_DUSTY_C`/
    `F_DUSTY_MEAN` attributes (`F_DUSTY_MEAN` is that same denominator,
    `population.star_population`'s own `(1-F_C)*F_DUSTY_O + F_C*F_DUSTY_C`)."""
    with h5py.File(_path(config, region), "r") as f:
        f_c = float(f.attrs["F_C"])
        f_dusty_c = float(f.attrs["F_DUSTY_C"])
        f_dusty_mean = float(f.attrs["F_DUSTY_MEAN"])
        grp = f[f"tile_{tile_id}"]
        evolved = grp["IS_EVOLVED"][()].astype(bool)
        u = grp["U"][()].astype(np.float64)[evolved]
        log10_b_o = grp["LOG10_B_AGB_O"][()].astype(np.float64)[evolved]
        log10_b_c = grp["LOG10_B_AGB_C"][()].astype(np.float64)[evolved]
        w_agb = grp["W_AGB"][()].astype(np.float64)[evolved]
    carbon_share = f_c * f_dusty_c / f_dusty_mean
    x = np.concatenate([u, u])
    log10_b = np.concatenate([log10_b_o, log10_b_c])
    w = np.concatenate([w_agb * (1.0 - carbon_share), w_agb * carbon_share])
    return x, log10_b, w
