"""The star-family population samples: STAR and AGB, one tile at a time
(SPEC_BMSTP_DRAFT.md sec. 5.1 "Marks", "Weight"; sec. 5.2 "Marks", "Weight";
sec. 2 "brightness"; IMPLEMENTATION_BMSTP_DRAFT.md sec. 1.2 P2).

Reads `population.star_population`'s per-tile product
(`population/star/population_star_tile__R.hdf5`): one root group per tile,
`tile_<id>`, holding `STAR_INDEX` (the row into the region's retained
field-star sample, `population.field_stars`'s `field-stars_trilegal_region`
product), `U` (scaled extinction), `W_STAR`/`W_AGB` (the reweighting of sec.
5.2) and `IS_EVOLVED`. PAHC has no sampler of its own: it reads STAR's grid
(`GRID_STAR`) unchanged.

The brightness mark is TRILEGAL's own dereddened 4.5 micron flux (sec. 2's
`F_4.5`), never a gray factor against a template: STAR reads it straight off
the retained sample's `FNU_MJY`; AGB turns the star's own luminosity and
distance into a shell flux through the AGB library's own flux-to-luminosity
ratio, per chemistry (sec. 5.2).
"""

import functools

import h5py
import numpy as np
from astropy.io import fits

from sesnaimpute import config as config_module
from sesnaimpute import definitions

_SKIP_KEYS = ("DIST_GRID", "LIMIT8_GRID_MJY")

#: 4.5 micron's index in the eight-band order every register and product
#: shares (J H Ks I1 I2 I3 I4 M1, sec. 2's `F_4.5`).
IDX_I2 = tuple(b.key for b in definitions.BANDS).index("I2")

#: The 16th-84th percentile half-width, in dex, above which the brief asks
#: the AGB per-chemistry ratio's spread to be flagged in the report (sec.
#: 5.2: "if it exceeds one cell (0.1 dex)").
AGB_RATIO_SPREAD_FLAG_DEX = 0.1


def _tile_path(config, region):
    return config_module.product_path(
        config, "population", "star", "population", "tile", region=region)


def _field_stars_path(config, region):
    return config_module.product_path(
        config, "population", "trilegal", "field-stars", "region", region=region)


def tile_ids(config, region):
    """The region's tile ids (the root groups `tile_<id>` of the per-tile
    star population product), sorted -- one grain axis for P2, read once
    per region without loading any tile's arrays."""
    with h5py.File(_tile_path(config, region), "r") as f:
        ids = sorted(int(name.split("_")[1]) for name in f.keys() if name not in _SKIP_KEYS)
    return np.array(ids, dtype=np.int64)


def sample_star(config, region, tile_id):
    """STAR's `(x, log10 F_4.5, w)` on tile `tile_id` (sec. 5.1 "Marks",
    "Weight"): `x = U`; `F_4.5` is TRILEGAL's own dereddened 4.5 micron
    flux of the star at its own distance -- the retained field-stars
    product's `FNU_MJY` at the I2 column, read through `STAR_INDEX`;
    `w = W_STAR` (the field-star share of the reweighting, sec. 5.2)."""
    with h5py.File(_tile_path(config, region), "r") as f:
        grp = f[f"tile_{tile_id}"]
        x = grp["U"][()].astype(np.float64)
        star_index = grp["STAR_INDEX"][()].astype(np.int64)
        w = grp["W_STAR"][()].astype(np.float64)
    with h5py.File(_field_stars_path(config, region), "r") as f:
        fnu_i2 = f["FNU_MJY"][:, IDX_I2].astype(np.float64)
    log10_f45 = np.log10(fnu_i2[star_index])
    return x, log10_f45, w


@functools.lru_cache(maxsize=None)
def agb_log10_ratio_stats(data_root):
    """Per chemistry, the AGB library's own ratio of its 4.5 micron
    reference flux (`F_REF_I2`, mJy at the register's 1 kpc reference) to
    its bolometric luminosity (`L_SUN`, `sed_models/agb/parameters.fits`,
    sec. 5.2): the median `log10` ratio (`F_4.5 = L (1kpc/d)^2 * r_chem`
    needs a point value, and a median commutes with `log10`), and the
    16th-84th percentile half-width in dex -- the spread the brief asks to
    be measured and stored (P2 attrs `F45_PER_L_SPREAD_DEX_O/C`). A join
    by `MODEL_NAME` between the register and the library's own curated
    parameters table, vectorised (`np.searchsorted` on the sorted
    parameters table, never a per-template python loop, rule 8), cached:
    survey-wide, the same for every region and tile."""
    register_path = f"{data_root}/sed_models/registers/agb_register.hdf5"
    with h5py.File(register_path, "r") as f:
        reg_names = np.char.decode(f["models/MODEL_NAME"][:].astype("S"), "utf-8")
        f_ref_i2 = f["models/F_REF_I2"][:].astype(np.float64)
    params_path = f"{data_root}/sed_models/agb/parameters.fits"
    with fits.open(params_path) as hdul:
        data = hdul[1].data
        p_names = np.char.strip(np.asarray(data["MODEL_NAME"]).astype(str))
        l_sun = np.asarray(data["L_SUN"], dtype=np.float64)
        chem = np.char.strip(np.asarray(data["CHEM"]).astype(str))

    order = np.argsort(p_names)
    p_sorted = p_names[order]
    pos = np.clip(np.searchsorted(p_sorted, reg_names), 0, order.size - 1)
    matched = p_sorted[pos] == reg_names
    if not matched.all():
        raise ValueError(
            "sample_star.agb_log10_ratio_stats: %d/%d AGB register templates "
            "have no row in %s by MODEL_NAME"
            % (int((~matched).sum()), matched.size, params_path))

    l_sun_matched = l_sun[order][pos]
    chem_matched = chem[order][pos]
    log10_r = np.log10(f_ref_i2) - np.log10(l_sun_matched)

    stats = {}
    for label in ("O", "C"):
        sel = chem_matched == label
        p16, p84 = np.percentile(log10_r[sel], [16.0, 84.0])
        stats[label] = dict(
            log10_r=float(np.median(log10_r[sel])),
            spread_dex=float(0.5 * (p84 - p16)),
            n=int(sel.sum()))
    return stats


def sample_agb(config, region, tile_id):
    """AGB's `(x, log10 F_4.5, w)` on tile `tile_id` (sec. 5.2 "Marks",
    "Weight"): the evolved subset only, each star's `x = U` placed twice
    -- once at the O-rich chemistry's implied shell flux, once at the
    C-rich -- weighted by the DUSTY subset's carbon share,
    `f_C F_dusty,C / (f_C F_dusty,C + (1 - f_C) F_dusty,O)` (sec. 5.2, not
    the bare carbon fraction `F_C`: `W_AGB` is already the dusty subset,
    whose chemistry mix is `F_dusty` by chemistry). `F_4.5 = L_i (1 kpc /
    d_i)^2 * r_chem`, `L_i`/`d_i` the star's own TRILEGAL luminosity and
    distance (the retained field-stars product's `LOG_L`/`DIST_PC`, read
    through `STAR_INDEX`), `r_chem` this chemistry's own median flux-to-
    luminosity ratio (`agb_log10_ratio_stats`)."""
    with h5py.File(_tile_path(config, region), "r") as f:
        f_c = float(f.attrs["F_C"])
        f_dusty_c = float(f.attrs["F_DUSTY_C"])
        f_dusty_mean = float(f.attrs["F_DUSTY_MEAN"])
        grp = f[f"tile_{tile_id}"]
        evolved = grp["IS_EVOLVED"][()].astype(bool)
        star_index = grp["STAR_INDEX"][()].astype(np.int64)[evolved]
        u = grp["U"][()].astype(np.float64)[evolved]
        w_agb = grp["W_AGB"][()].astype(np.float64)[evolved]
    with h5py.File(_field_stars_path(config, region), "r") as f:
        dist_pc_all = f["DIST_PC"][:].astype(np.float64)
        log_l_all = f["LOG_L"][:].astype(np.float64)
    dist_pc = dist_pc_all[star_index]
    log_l = log_l_all[star_index]

    ratio = agb_log10_ratio_stats(config.data_root)
    two_log_inv_d = 2.0 * np.log10(1000.0 / dist_pc)
    log10_f45_o = log_l + two_log_inv_d + ratio["O"]["log10_r"]
    log10_f45_c = log_l + two_log_inv_d + ratio["C"]["log10_r"]

    carbon_share = f_c * f_dusty_c / f_dusty_mean
    x = np.concatenate([u, u])
    log10_f45 = np.concatenate([log10_f45_o, log10_f45_c])
    w = np.concatenate([w_agb * (1.0 - carbon_share), w_agb * carbon_share])
    return x, log10_f45, w
