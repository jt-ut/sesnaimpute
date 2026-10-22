"""The star-family population samples: STAR and AGB, one tile at a time
(SPEC_BMSTP_DRAFT.md sec. 5.1 "Marks", "Weight"; sec. 5.2 "Marks", "Weight";
sec. 2 "brightness"; IMPLEMENTATION_BMSTP_DRAFT.md sec. 1.2 P2).

Reads `population.star_population`'s per-tile product
(`population/star/population_star_tile__R.hdf5`): one root group per tile,
`tile_<id>`, holding `STAR_INDEX` (the row into the region's retained
field-star sample, `population.field_stars`'s `field-stars_trilegal_region`
product), `XI` (the depth fraction ξ), `W_STAR`/`W_AGB` (the reweighting of sec.
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


_FIELD_STAR_CACHE = {}


def _field_star_column(config, region, name):
    """One column of the region's retained field-stars product, read ONCE
    per `(data root, region, column)` and cached for the worker process's
    lifetime: `"F45"` (the `FNU_MJY` I2 column, dereddened, log10 mJy),
    `"DIST_PC"` or `"LOG_L"` (float64). This is the same per-process
    memoization `_cached_profile` already applies to the sightline product,
    here for a per-tile caller (`sample_star`/`sample_agb`/`star_distances`/
    `agb_star_distances`, `bmstp.shapes`' own per-tile piece) that would
    otherwise reread the region's WHOLE field-star catalogue (Cygnus X:
    2.4 GB) on every one of a region's tiles (rule 9, CODING_RULES_BMSTP.md:
    never touch the full catalogue inside a per-item loop). Each caller
    reads only the columns it uses, so a worker never holds (or widens) a
    column no tile of its own needs."""
    key = (config.data_root, region)
    if _FIELD_STAR_CACHE.get("key") != key:
        _FIELD_STAR_CACHE.clear()   # one region's columns per worker at a time
        _FIELD_STAR_CACHE["key"] = key
    if name not in _FIELD_STAR_CACHE:
        with h5py.File(_field_stars_path(config, region), "r") as f:
            if name == "F45":
                col = np.log10(f["FNU_MJY"][:, IDX_I2].astype(np.float64))
            else:
                col = f[name][:].astype(np.float64)
        _FIELD_STAR_CACHE[name] = col
    return _FIELD_STAR_CACHE[name]


def tile_ids(config, region):
    """The region's tile ids (the root groups `tile_<id>` of the per-tile
    star population product), sorted -- one grain axis for P2, read once
    per region without loading any tile's arrays.

    Identified by the group's own name pattern (`tile_` followed by
    nothing but digits), never by a skip-list of the product's other,
    non-tile top-level names: C21 (commit 4f66043, `RULE_COUNT_FRAC`/
    `RULE_WEIGHT_FRAC`) and C21 item 3 (commit 701fe92, `KS_EDGES_2MASS`/
    `N_KS_OBS_2MASS`/`FINAL_POP_KS_HIST`, removed again by C23) each added
    a further root-level name to `population/star/population_star_tile__
    R.hdf5` that this function's own skip-list (`DIST_GRID`,
    `LIMIT8_GRID_MJY`) never learned about, so `int(name.split('_')[1])`
    broke on the first one of those (`'RULE_COUNT_FRAC'.split('_')[1] ==
    'COUNT'`) the moment the product was rebuilt (C23, found running the
    baseline before this brief's own change). `bmstp.atlas` and `bmstp.
    template_weights` already read this SAME product's own tile axis this
    way (`k.startswith('tile_')`); this is that one correct pattern, not
    a third one."""
    with h5py.File(_tile_path(config, region), "r") as f:
        ids = sorted(int(name[len("tile_"):]) for name in f.keys()
                     if name.startswith("tile_") and name[len("tile_"):].isdigit())
    return np.array(ids, dtype=np.int64)


def sample_star(config, region, tile_id):
    """STAR's `(x, log10 F_4.5, w)` on tile `tile_id` (sec. 5.1 "Marks",
    "Weight"): `x = U`; `F_4.5` is TRILEGAL's own dereddened 4.5 micron
    flux of the star at its own distance -- the retained field-stars
    product's `FNU_MJY` at the I2 column, read through `STAR_INDEX`;
    `w = W_STAR` (the field-star share of the reweighting, sec. 5.2)."""
    with h5py.File(_tile_path(config, region), "r") as f:
        grp = f[f"tile_{tile_id}"]
        x = grp["XI"][()].astype(np.float64)
        star_index = grp["STAR_INDEX"][()].astype(np.int64)
        w = grp["W_STAR"][()].astype(np.float64)
    log10_f45_all = _field_star_column(config, region, "F45")
    log10_f45 = log10_f45_all[star_index]
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
        _PROFILE_CACHE.clear()   # one region's profile per worker at a time
        _PROFILE_CACHE[key] = profile_module.read(config, region)
    return _PROFILE_CACHE[key]


def _tile_centre_lb(config, region, tile_id):
    """The STAR anchor tile's own centre, `(l_deg, b_deg)`
    (`population.anchor_tiles`'s `TILE_L_DEG`/`TILE_B_DEG`)."""
    path = config_module.product_path(config, "population", "anchors", "tiles", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        return float(f["TILE_L_DEG"][tile_id]), float(f["TILE_B_DEG"][tile_id])


def tile_width_classes(config, region, tile_id):
    """`(row, sigma_classes_dex, d_front, d_back, u_front)` (the
    depth-width rule): the tile's own representative sightline
    (`_RegionProfile.row_of_lb` at the tile's own centre) and its
    `grid.N_WIDTH_CLASSES` geometric width classes in `log10 x`, floored
    at one grid cell (`grid._X_CELL_WIDTH`) and capped at that
    sightline's own cloud-interval span, `log10 u(d_back) - log10
    u(d_front)` (`sample_cloud.cloud_interval_pc`'s region-level
    front/back distances, read against the sightline's own total column
    `A_INF_K` so a `d_back` past the map's own edge -- the doubled cloud
    interval routinely reaches it, sec. 2 -- still sees the analytic
    far-field tail rather than a flat extrapolation). `d_front`/`d_back`
    and `u_front` (this SAME representative sightline's own profile read
    at the cloud interval's front edge) are returned beside the width
    classes so a caller building the STAR depth mark (C5) reads the
    identical front edge and foreground share this function already
    computed, rather than a second, independent read of the profile."""
    from sesnaimpute.bmstp import sample_cloud  # deferred: sample_cloud
    # imports template_weights, which imports this module -- a module-load
    # cycle a top-level import here would create.
    profile_obj = _cached_profile(config, region)
    tile_l, tile_b = _tile_centre_lb(config, region, tile_id)
    row, pix = profile_obj.row_of_lb(tile_l, tile_b)
    d_front, d_back = sample_cloud.cloud_interval_pc(config, region)
    a_inf_row = profile_module.sightline_total_column(config, region, pix)
    u_front = float(profile_obj.u(d_front, hpx_pix=pix, total_column_ak=a_inf_row))
    u_back = float(profile_obj.u(d_back, hpx_pix=pix, total_column_ak=a_inf_row))
    floor_dex = float(grid._X_CELL_WIDTH)
    cap_dex = max(floor_dex, np.log10(max(u_back, 1e-300)) - np.log10(max(u_front, 1e-300)))
    sigma_classes_dex = np.geomspace(floor_dex, cap_dex, grid.N_WIDTH_CLASSES)
    return row, sigma_classes_dex, d_front, d_back, u_front


_SIGMA_SAMPLES_CACHE = {}


def _sigma_samples_product(config, region):
    """The region's `SIGMA_RATIO_SAMPLES` product: the 12 released
    Edenhofer posterior samples' own standard deviation of the depth
    mark's ratio `u(d) = A(d) / A_inf`, read in place of a correlated-sum
    bound -- there is no stated correlation length
    (`bms_review/studies/edenhofer_kernel.md`). No fallback: a region
    without this product fails here by name rather than silently reading
    a bound."""
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


def star_width_class(config, region, dist_pc, row, sigma_classes_dex, d_front, d_back):
    """Per star, the width-class index (0..N_WIDTH_CLASSES-1) nearest its
    own `sigma_x` (the depth-width rule): `sigma_x(d) / (u(d) ln 10)`,
    `u(d)` the depth mark itself, `A(d) / A_inf` on the tile's
    representative sightline, and `sigma_x(d)` its `SIGMA_RATIO_SAMPLES`
    -- the ratio's own across-sample spread, not the numerator's
    (`sky.derived.edenhofer_samples`'s module docstring), both at the
    star's own distance, clipped to the class range before the
    nearest-class lookup (in log space, since the classes are
    geometric). Consequence of the C5 ruling: a star the interval logic
    PINS (`dist_pc > d_back` or `dist_pc < d_front` -- `shapes._star_
    depth_mark`'s own test) has no distance uncertainty in `x` left to
    carry -- the interval has already decided which side of the wall
    it's on, so the map's own `sigma_x(d)` no longer describes its mark;
    what pencil-over-beam spread such a star DOES carry is the reader's
    own column kernel, applied later, not this stage's. A pinned star
    therefore takes the FLOOR width, class 0 (`sigma_classes_dex[0]`,
    one grid cell); only a star genuinely inside the interval keeps the
    profile's own `sigma_x` class computed above."""
    profile_obj = _cached_profile(config, region)
    dist_pc = np.asarray(dist_pc, dtype=np.float64)
    hpx_pix = int(profile_obj.hpx[row])
    a_inf_row = profile_module.sightline_total_column(config, region, hpx_pix)
    u_d = profile_obj.u(dist_pc, hpx_pix=hpx_pix, total_column_ak=a_inf_row)
    sigma_x_d = _sigma_samples_at(config, region, hpx_pix, dist_pc)
    sigma_x = sigma_x_d / (np.maximum(u_d, 1e-12) * np.log(10.0))
    sigma_x = np.clip(sigma_x, sigma_classes_dex[0], sigma_classes_dex[-1])
    idx = np.argmin(np.abs(np.log(sigma_x)[:, None] - np.log(sigma_classes_dex)[None, :]), axis=1)
    pinned = (dist_pc > d_back) | (dist_pc < d_front)
    idx = np.where(pinned, 0, idx)
    return idx.astype(np.int64)


def star_distances(config, region, tile_id):
    """Each retained field star's own TRILEGAL distance (`d_i`, sec. 5.1 "Marks"):
    `STAR_INDEX` into the field-stars product's own `DIST_PC`, the SAME
    join `sample_star` uses for `F_4.5`."""
    with h5py.File(_tile_path(config, region), "r") as f:
        star_index = f[f"tile_{tile_id}"]["STAR_INDEX"][()].astype(np.int64)
    dist_pc = _field_star_column(config, region, "DIST_PC")
    return dist_pc[star_index]


def _agb_register_log10_ratio(sed_models):
    """`(names, chem, log10_r)` in the AGB register's own row order: every
    template's own `log10(F_REF_I2 / L_SUN)` join (`F_REF_I2` the
    register's 4.5 micron reference flux at 1 kpc, floor-free here --
    the register's own rows are already the curated, non-zero set;
    `L_SUN` from `sed_models/agb/parameters.fits`'s curated parameters
    table, sec. 5.2). A join by `MODEL_NAME`, vectorised
    (`np.searchsorted` on the sorted parameters table, never a
    per-template python loop, rule 8). Shared by `agb_log10_ratio_stats`
    (the per-chemistry median/spread, reporting only) and
    `agb_log10_ratio_per_template` (every row kept, the spec's
    template-dependent offset, sec. 5.2, FINDINGS.md S10) so the join
    itself lives in exactly one place."""
    register_path = f"{sed_models}/registers/agb_register.hdf5"
    with h5py.File(register_path, "r") as f:
        reg_names = np.char.decode(f["models/MODEL_NAME"][:].astype("S"), "utf-8")
        f_ref_i2 = f["models/F_REF_I2"][:].astype(np.float64)
    params_path = f"{sed_models}/agb/parameters.fits"
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
            "sample_star._agb_register_log10_ratio: %d/%d AGB register templates "
            "have no row in %s by MODEL_NAME"
            % (int((~matched).sum()), matched.size, params_path))

    l_sun_matched = l_sun[order][pos]
    chem_matched = chem[order][pos]
    log10_r = np.log10(f_ref_i2) - np.log10(l_sun_matched)
    return reg_names, chem_matched, log10_r


@functools.lru_cache(maxsize=None)
def agb_log10_ratio_stats(sed_models):
    """Per chemistry, the AGB library's own ratio of its 4.5 micron
    reference flux (`F_REF_I2`, mJy at the register's 1 kpc reference) to
    its bolometric luminosity (`L_SUN`, `sed_models/agb/parameters.fits`,
    sec. 5.2): the median `log10` ratio (`F_4.5 = L (1kpc/d)^2 * r_chem`
    needs a point value, and a median commutes with `log10`), and the
    16th-84th percentile half-width in dex -- the spread the brief asks to
    be measured and stored (P2 attrs `F45_PER_L_SPREAD_DEX_O/C`), REPORTING
    ONLY (FINDINGS.md S10): `sample_agb_shape` below carries each
    template's OWN ratio into the brightness shape itself, this function's
    median is no longer the brightness construction's input. Cached:
    survey-wide, the same for every region and tile."""
    _reg_names, chem_matched, log10_r = _agb_register_log10_ratio(sed_models)
    stats = {}
    for label in ("O", "C"):
        sel = chem_matched == label
        p16, p84 = np.percentile(log10_r[sel], [16.0, 84.0])
        stats[label] = dict(
            log10_r=float(np.median(log10_r[sel])),
            spread_dex=float(0.5 * (p84 - p16)),
            n=int(sel.sum()))
    return stats


@functools.lru_cache(maxsize=None)
def agb_log10_ratio_per_template(sed_models):
    """`(names, chem, log10_r)` at full AGB-register resolution, in the
    register's own row order -- the SAME order `bmstp.template_weights.
    agb_tau_ratio` returns its per-template population weight in (both
    read the register directly, sec 1.4), so `sample_agb_shape` can pair
    a template's own flux-to-luminosity ratio with its own population
    weight by position, no second join. The spec's template-dependent
    offset (SPEC_BMSTP_DRAFT.md sec 5.2; FINDINGS.md S10: "the coupling
    enters through the template's own F/L, never a separate factor")."""
    return _agb_register_log10_ratio(sed_models)


def agb_star_distances(config, region, tile_id):
    """AGB's own distance array row-aligned with `sample_agb`'s
    `(x, log10_f45, w, template_index)`: the evolved subset's own
    distance, repeated once per AGB template within each chemistry, in
    the SAME row order `sample_agb` builds (all O-rich templates for
    every evolved star, then all C-rich -- `agb_log10_ratio_per_template`'s
    own per-chemistry row order) -- AGB follows STAR on distance (sec.
    5.2). `bmstp.atlas`'s own AGB member assembly reads this the same way
    `bmstp.shapes._build_one_tile` does."""
    with h5py.File(_tile_path(config, region), "r") as f:
        grp = f[f"tile_{tile_id}"]
        evolved = grp["IS_EVOLVED"][()].astype(bool)
        star_index = grp["STAR_INDEX"][()].astype(np.int64)[evolved]
    dist_pc_all = _field_star_column(config, region, "DIST_PC")
    d = dist_pc_all[star_index]
    _names, chem, _log10_r = agb_log10_ratio_per_template(config.inputs["sed_models"])
    n_o = int(np.sum(chem == "O"))
    n_c = int(np.sum(chem == "C"))
    return np.concatenate([np.repeat(d, n_o), np.repeat(d, n_c)])


def sample_agb(config, region, tile_id):
    """AGB's JOINT measure over (template, brightness) on tile `tile_id`,
    as `(x, log10 F_4.5, w, template_index)` (sec. 5.2 "Marks", "Weight";
    FINDINGS.md S10, coordinator's joint-measure ruling): the evolved
    subset only, each star's `x = U` placed once per AGB template of a
    chemistry -- never a single chemistry-median point -- at THAT
    TEMPLATE's own implied shell flux, `template_index` naming which
    register row (so a caller can read that template's own `F_REF`/
    `FLOOR_LINEAR` straight off the register, no second lookup table).
    `F_4.5 = L_i (1 kpc / d_i)^2 * r_chem,theta`, `L_i`/`d_i` the star's
    own TRILEGAL luminosity and distance (the retained field-stars
    product's `LOG_L`/`DIST_PC`, read through `STAR_INDEX`), `r_chem,theta`
    THIS TEMPLATE's own flux-to-luminosity ratio
    (`agb_log10_ratio_per_template`) -- identity: at fixed `L_i`/`d_i`,
    two templates' own rows differ in `log10 F_4.5` by EXACTLY their own
    `log10_r` difference. Mass is the star's own AGB weight times the
    chemistry's own dusty-mix share times that template's own population
    share within its chemistry (`template_weights.agb_tau_ratio`'s
    `ratio`, the shell optical-depth weight -- unchanged, reused here,
    never recomputed: "the shell weights stay what they are"); summed
    over one chemistry's templates this recovers exactly that chemistry's
    own share of `w_agb` (`ratio` sums to 1 within each chemistry, sec.
    5.2), so the population total is unchanged (identity 2).

    This ONE measure is read two ways by its two callers, the joint's
    marginal and its conditional, never a second crossing of either
    against the other:
    `bmstp.shapes._build_one_tile` sums it over template (keeping `x`,
    `log10_f45`, `w`) into the brightness SHAPE, `GRID_AGB`;
    `bmstp.template_weights.build_agb` sums the SAME measure, region-wide
    over every tile, over EVERYTHING BUT template and brightness bin,
    then normalises over template at each bin (`_normalise_over_theta`)
    into the brightness-CONDITIONAL template weight, P5's `W`;
    `bmstp.atlas`'s AGB member assembly reads every row directly as one
    catalogued-count member (`x`, `log10_f45`, `w`, and `template_index`
    for its own `F_REF` rescaling) -- no cross product left to build
    there (coordinator's ruling: "no second crossing")."""
    # local import: `template_weights` imports this module at its own top
    # level (`agb_tau_ratio`'s row-order check against the register), so
    # importing it back at module scope here would be circular.
    from sesnaimpute.bmstp import template_weights as template_weights_module

    with h5py.File(_tile_path(config, region), "r") as f:
        f_c = float(f.attrs["F_C"])
        f_dusty_c = float(f.attrs["F_DUSTY_C"])
        f_dusty_mean = float(f.attrs["F_DUSTY_MEAN"])
        grp = f[f"tile_{tile_id}"]
        evolved = grp["IS_EVOLVED"][()].astype(bool)
        star_index = grp["STAR_INDEX"][()].astype(np.int64)[evolved]
        u = grp["XI"][()].astype(np.float64)[evolved]
        w_agb = grp["W_AGB"][()].astype(np.float64)[evolved]
    dist_pc_all = _field_star_column(config, region, "DIST_PC")
    log_l_all = _field_star_column(config, region, "LOG_L")
    dist_pc = dist_pc_all[star_index]
    log_l = log_l_all[star_index]

    names, chem, log10_r = agb_log10_ratio_per_template(config.inputs["sed_models"])
    names_pop, chem_pop, _log10_tau, pop_ratio = template_weights_module.agb_tau_ratio(config)
    if not (names_pop.size == names.size and np.array_equal(names_pop, names)
            and np.array_equal(chem_pop, chem)):
        raise ValueError(
            "sample_star.sample_agb: template_weights.agb_tau_ratio's row "
            "order disagrees with agb_log10_ratio_per_template's AGB register join")
    model_row = np.arange(names.size, dtype=np.int64)

    two_log_inv_d = 2.0 * np.log10(1000.0 / dist_pc)
    carbon_share = f_c * f_dusty_c / f_dusty_mean

    x_parts, f45_parts, w_parts, idx_parts = [], [], [], []
    for label, chem_share in (("O", 1.0 - carbon_share), ("C", carbon_share)):
        sel = chem == label
        n_tmpl = int(sel.sum())
        # (n_star, n_tmpl): star index varies down the rows, template
        # index across the columns -- `.ravel()` (C order) then walks
        # template-fastest within a star, the SAME order
        # `agb_star_distances`'s `np.repeat(d, n_tmpl)` repeats `d` in.
        log10_f45_tmpl = log_l[:, None] + two_log_inv_d[:, None] + log10_r[sel][None, :]
        w_tmpl = (w_agb * chem_share)[:, None] * pop_ratio[sel][None, :]
        n_star = u.size
        x_parts.append(np.repeat(u, n_tmpl))
        f45_parts.append(log10_f45_tmpl.ravel())
        w_parts.append(w_tmpl.ravel())
        idx_parts.append(np.tile(model_row[sel], n_star))

    x = np.concatenate(x_parts)
    log10_f45 = np.concatenate(f45_parts)
    w = np.concatenate(w_parts)
    template_index = np.concatenate(idx_parts)
    return x, log10_f45, w, template_index
