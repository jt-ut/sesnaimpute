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
import os

import h5py
import numpy as np
from astropy.io import fits

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute.bmstp import grid
from sesnaimpute.sky.derived import profile as profile_module

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



# ---------------------------------------------------------------------------
# the field-star depth mark's own width -- the map's propagated
# column sigma at the star's distance, quantised to N_WIDTH_CLASSES
# geometric classes between the one-cell floor and the tile's own
# cloud-interval-span cap (SPEC_BMSTP_DRAFT.md sec. 2, sec. 5.1 "Marks").
# A TRILEGAL field star carries a distance but no sky position of its own
# (module docstring), so the tile's own centre stands in for one: every
# star of a tile reads the SAME representative sightline (the region's
# admitted profile sightline nearest the tile's centre).
# ---------------------------------------------------------------------------

_PROFILE_CACHE = {}


def _cached_profile(config, region):
    """The region's profile object, cached per `(data_root, region)` so a
    per-tile call (`bmstp.shapes`' `Parallel` dispatches many tiles to
    each worker) does not reopen and re-parse the whole sightline product
    per tile."""
    key = (config.data_root, region)
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = profile_module.read(config, region)
    return _PROFILE_CACHE[key]


def _tile_centre_lb(config, region, tile_id):
    """The STAR anchor tile's own centre, `(l_deg, b_deg)`
    (`population.anchor_tiles`'s `TILE_L_DEG`/`TILE_B_DEG`)."""
    path = config_module.product_path(config, "population", "anchors", "tiles", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        return float(f["TILE_L_DEG"][tile_id]), float(f["TILE_B_DEG"][tile_id])


def tile_width_classes(config, region, tile_id):
    """`(row, sigma_classes_dex)` (the depth-width rule): the tile's own
    representative sightline (`_RegionProfile.row_of_lb` at the tile's
    own centre) and its `grid.N_WIDTH_CLASSES` geometric width classes in
    `log10 x`, floored at one grid cell (`grid._X_CELL_WIDTH`) and capped
    at that sightline's own cloud-interval span, `log10 u(d_back) -
    log10 u(d_front)` (`sample_cloud.cloud_interval_pc`'s region-level
    front/back distances, read against the sightline's own total column
    `A_INF_K` so a `d_back` past the map's own edge -- the doubled cloud
    interval routinely reaches it, sec. 2 -- still sees the analytic
    far-field tail rather than a flat extrapolation)."""
    from sesnaimpute.bmstp import sample_cloud  # deferred: sample_cloud
    # imports template_weights, which imports this module -- a module-load
    # cycle a top-level import here would create.
    profile_obj = _cached_profile(config, region)
    tile_l, tile_b = _tile_centre_lb(config, region, tile_id)
    row, pix = profile_obj.row_of_lb(tile_l, tile_b)
    d_front, d_back = sample_cloud.cloud_interval_pc(config, region)
    a_inf_row = float(profile_obj.a_inf[row])
    u_front = float(profile_obj.u(d_front, hpx_pix=pix, total_column_ak=a_inf_row))
    u_back = float(profile_obj.u(d_back, hpx_pix=pix, total_column_ak=a_inf_row))
    floor_dex = float(grid._X_CELL_WIDTH)
    cap_dex = max(floor_dex, np.log10(max(u_back, 1e-300)) - np.log10(max(u_front, 1e-300)))
    sigma_classes_dex = np.geomspace(floor_dex, cap_dex, grid.N_WIDTH_CLASSES)
    return row, sigma_classes_dex


_SIGMA_SAMPLES_CACHE = {}


def _sigma_samples_product(config, region):
    """The region's `SIGMA_RATIO_SAMPLES` product beside its
    `SIGMA_SAMPLES_K`: the 12 released Edenhofer posterior samples' own
    standard deviation of the depth mark's ratio `u(d) = A(d) / A_inf`,
    read in place of the profile's fully correlated `SIGMA_COR_K` sum --
    an upper bound absent a stated correlation length
    (`bms_review/studies/edenhofer_kernel.md`). No fallback: a region
    without this product fails here by name rather than silently reading
    the bound."""
    key = (config.data_root, region)
    if key not in _SIGMA_SAMPLES_CACHE:
        path = config_module.product_path(config, "sky/derived", "edenhofer", "profile-sigma-samples",
                                           "sightline", region=region)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "sample_star.star_width_class: %s missing -- run the "
                "sesnaimpute.sky.derived.edenhofer_samples RUNBOOK line for it" % path)
        with h5py.File(path, "r") as f:
            _SIGMA_SAMPLES_CACHE[key] = (f["HPX_PIX_256"][:].astype(np.int64),
                                          f["DIST_PC"][:].astype(np.float64),
                                          f["SIGMA_RATIO_SAMPLES"][:].astype(np.float64))
    return _SIGMA_SAMPLES_CACHE[key]


def _sigma_samples_at(config, region, hpx_pix, dist_pc):
    """`sigma_x(d)`, the ratio's own across-sample spread, from
    `SIGMA_RATIO_SAMPLES`, interpolated on that product's own distance
    axis, at sightline `hpx_pix`."""
    hpx, dist, sigma = _sigma_samples_product(config, region)
    order = np.argsort(hpx)
    j = order[np.searchsorted(hpx[order], hpx_pix)]
    if hpx[j] != hpx_pix:
        raise ValueError("sample_star.star_width_class: hpx_pix %d has no row in %s's SIGMA_RATIO_SAMPLES"
                          % (hpx_pix, region))
    return np.interp(dist_pc, dist, sigma[j])


def star_width_class(config, region, dist_pc, row, sigma_classes_dex):
    """Per star, the width-class index (0..N_WIDTH_CLASSES-1) nearest its
    own `sigma_x` (the depth-width rule): `sigma_x(d) / (u(d) ln 10)`,
    `u(d)` the depth mark itself, `A(d) / A_inf` on the tile's
    representative sightline, and `sigma_x(d)` its `SIGMA_RATIO_SAMPLES`
    -- the ratio's own across-sample spread, not the numerator's
    (`sky.derived.edenhofer_samples`'s module docstring), both at the
    star's own distance, clipped to the class range before the
    nearest-class lookup (in log space, since the classes are
    geometric)."""
    profile_obj = _cached_profile(config, region)
    dist_pc = np.asarray(dist_pc, dtype=np.float64)
    hpx_pix = int(profile_obj.hpx[row])
    a_inf_row = float(profile_obj.a_inf[row])
    u_d = profile_obj.u(dist_pc, hpx_pix=hpx_pix, total_column_ak=a_inf_row)
    sigma_x_d = _sigma_samples_at(config, region, hpx_pix, dist_pc)
    sigma_x = sigma_x_d / (np.maximum(u_d, 1e-12) * np.log(10.0))
    sigma_x = np.clip(sigma_x, sigma_classes_dex[0], sigma_classes_dex[-1])
    idx = np.argmin(np.abs(np.log(sigma_x)[:, None] - np.log(sigma_classes_dex)[None, :]), axis=1)
    return idx.astype(np.int64)


def star_distances(config, region, tile_id):
    """Each retained field star's own TRILEGAL distance (`d_i`, sec. 5.1 "Marks"):
    `STAR_INDEX` into the field-stars product's own `DIST_PC`, the SAME
    join `sample_star` uses for `F_4.5`."""
    with h5py.File(_tile_path(config, region), "r") as f:
        star_index = f[f"tile_{tile_id}"]["STAR_INDEX"][()].astype(np.int64)
    with h5py.File(_field_stars_path(config, region), "r") as f:
        dist_pc = f["DIST_PC"][:].astype(np.float64)
    return dist_pc[star_index]


def agb_star_distances(config, region, tile_id):
    """AGB's own distance array row-aligned with `sample_agb`'s
    `(x, log10_f45, w)`: the evolved subset's own distance, duplicated
    (O-rich draw, then C-rich) the same way `sample_agb` duplicates `u`
    (AGB follows STAR, sec. 5.2)."""
    with h5py.File(_tile_path(config, region), "r") as f:
        grp = f[f"tile_{tile_id}"]
        evolved = grp["IS_EVOLVED"][()].astype(bool)
        star_index = grp["STAR_INDEX"][()].astype(np.int64)[evolved]
    with h5py.File(_field_stars_path(config, region), "r") as f:
        dist_pc_all = f["DIST_PC"][:].astype(np.float64)
    d = dist_pc_all[star_index]
    return np.concatenate([d, d])


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
