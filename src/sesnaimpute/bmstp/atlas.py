"""P6, the prior atlas (SPEC_BMSTP_DRAFT.md sec. 8; IMPLEMENTATION_BMSTP_DRAFT.md
sec. 1.2 P6, sec. 3 row 1.9).

Per admitted nside-512 pixel, a fixed-seed Monte Carlo sample of each class's
population, dimmed at the pixel's own column through the blended law
(`population.selection.kappa_hybrid`), counted by the two-of-eight test at the
pixel's own median 50% limits (`catalog.depth_grid`'s `F_LIM_50_MED_MJY`). The
selection appears here and nowhere else in the atlas (sec. 1.2): the prior itself
is unthinned.

This build writes STAR/AGB/PAHC (sec. 5.1-5.3, partitioning the field population:
a star is a STAR or a PAHC member of the Monte Carlo, never both, weighted
`W_STAR*(1-P_PAHC)`/`W_STAR*P_PAHC`) and GAL (sec. 5.4: SWIRE's four IRAC fluxes
per galaxy, S from the counts law's own node, colours from a galaxy measured at
that node, at `x=1`). YSO and H2S (sec. 5.5-5.6) did not fit this unit's budget on
top of the other four classes' own read chains; their `N_CAT_*`/`SHARE_*` columns
are written as NaN and excluded from the total-count check, which is therefore
reported for STAR+AGB+PAHC+GAL only, not the full six.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.population import selection as selection_module
from sesnaimpute.bmstp import sample_gal

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)
IDX_I4 = BAND_KEYS.index("I4")
CLASSES = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")

#: Monte Carlo draw size per pixel and class -- SPEC_BMSTP_DRAFT.md sec. 8:
#: "A sample of 1e4 per class holds the Monte Carlo error under 1%."
N_MC = 10_000

#: A fixed seed for the per-tile resample (reproducible, not a science
#: constant): one `RandomState` per tile, seeded by this value plus the
#: tile id, so two runs draw the same members.
MC_SEED = 137

#: Two of eight bands clear -- the survey's own catalogue rule (sec. 1.2),
#: the same constant `population.selection.MIN_BANDS` sets.
MIN_BANDS_CLEAR = selection_module.MIN_BANDS

#: joblib worker cap (CODING_RULES_BMSTP.md rule 10a).
_MAX_N_JOBS = 4

_HPX512_PIXEL_DEG2 = 41252.96 / (12 * 512 ** 2)


def _depth_grid(config, region):
    """The admitted pixel axis and its median 50% limits (`catalog.depth_grid`,
    sec. 3.3): `(pix, f_lim_50_med_mjy)`."""
    path = config_module.product_path(config, "catalog", "sesna", "depth-grid", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        f_lim = np.asarray(f["F_LIM_50_MED_MJY"][:], dtype=np.float64)
    return pix, f_lim


def _coverage(config, region, pix):
    """The IRAC coverage fraction at each admitted pixel (sec. 3.3's
    "coverage"): the mean of `sky.derived.coverage`'s four-band `FRAC` at
    `hpx512` granule, 0 where a pixel is absent from that product (never
    observed in any Spitzer band)."""
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64).mean(axis=1)
    order = np.argsort(cov_pix)
    loc = np.searchsorted(cov_pix[order], pix)
    loc = np.minimum(loc, cov_pix.size - 1)
    hit = order[loc]
    found = cov_pix[hit] == pix
    out = np.zeros(pix.size, dtype=np.float64)
    out[found] = frac[hit[found]]
    return out


def _pixel_column(config, pix):
    """The pixel's own column and arm, from the sightline it is a child of
    (SPEC_BMSTP_DRAFT.md sec. 8: "placed at the pixel (its column, its
    tile or sightline, its arm)"). Nested HEALPix, confirmed from
    `granules.build`'s own `HPX_PIX_256 = HPX_PIX_512 // 4`: every
    admitted nside-512 pixel's parent nside-256 sightline is `pix // 4`.
    `sky/derived/adopted/column_adopted_sightline.hdf5` (survey-wide, no
    region argument) carries `A_K`/`PROVENANCE` at that granule for every
    source-bearing sightline, so every admitted pixel resolves -- no NaN."""
    parent256 = pix // 4
    path = config_module.product_path(config, "sky/derived", "adopted", "column", "sightline")
    with h5py.File(path, "r") as f:
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)
        prov = np.asarray(f["PROVENANCE"][:])
    order = np.argsort(sl_pix)
    loc = np.searchsorted(sl_pix[order], parent256)
    loc = np.minimum(loc, sl_pix.size - 1)
    hit = order[loc]
    found = sl_pix[hit] == parent256
    if not np.all(found):
        raise ValueError("bmstp.atlas: %d admitted pixel(s) have no sightline column in %s"
                          % (int(np.sum(~found)), path))
    return a_k[hit], prov[hit]


def _pixel_tile(config, region, pix):
    """`TILE_ID` at each admitted pixel (`population/anchors/tiles/
    hpx512__R.hdf5`, the star-family tile definition), -1 where the pixel
    carries no tile."""
    path = config_module.product_path(config, "population", "anchors", "tiles", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        t_pix_id = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        t_tile_id = np.asarray(f["TILE_ID"][:], dtype=np.int64)
    order = np.argsort(t_pix_id)
    loc = np.searchsorted(t_pix_id[order], pix)
    loc = np.minimum(loc, t_pix_id.size - 1)
    hit = order[loc]
    found = t_pix_id[hit] == pix
    tile = np.full(pix.size, -1, dtype=np.int64)
    tile[found] = t_tile_id[hit[found]]
    return tile


def _draw_members(rng, weight, n_mc):
    """`(idx, total)`: `n_mc` indices into the tile's retained field-star
    arrays, drawn with replacement in proportion to `weight` -- the fixed-
    seed Monte Carlo sample of sec. 8. Each draw stands for `total / n_mc`
    objects, `total = sum(weight)`, so the accepted fraction over the draw
    times `total / OMEGA_SIM_DEG2` recovers the class's own catalogued
    density. `None` where the tile carries no weight at all (e.g. no
    evolved star for AGB)."""
    total = float(np.sum(weight, dtype=np.float64))
    if total <= 0.0 or weight.size == 0:
        return None, 0.0
    p = weight / total
    idx = rng.choice(weight.size, size=n_mc, replace=True, p=p)
    return idx, total


def _accepted_fraction(a_col, u, flux0, f_lim, config):
    """`(frac, mc_error)` per pixel: `flux0` (n_mc, 8) undimmed, `u` (n_mc,)
    the member's own placement fraction, `a_col` (n_pix,) the pixel's own
    column, `f_lim` (n_pix, 8) the pixel's own median limits. `a = a_col *
    u` (sec. 8's "its own extinction"), dimmed through the blended law
    (`population.selection.kappa_hybrid`), counted where at least
    `MIN_BANDS_CLEAR` of eight bands clear; `mc_error` is the binomial
    standard error of the accepted fraction at `N_MC` draws."""
    n_mc = u.size
    a = a_col[:, None] * u[None, :]  # (n_pix, n_mc)
    w_ramp = selection_module.law_dense_weight(a)  # (n_pix, n_mc)
    kappa = selection_module.kappa_hybrid(config, w_ramp)  # (n_pix, n_mc, 8)
    dimming = 0.4 * a[:, :, None] * kappa  # (n_pix, n_mc, 8)
    flux = flux0[None, :, :] * 10.0 ** (-dimming)  # (n_pix, n_mc, 8)
    n_clear = np.sum(flux >= f_lim[:, None, :], axis=2)  # (n_pix, n_mc)
    accepted = (n_clear >= MIN_BANDS_CLEAR).astype(np.float64)
    frac = accepted.mean(axis=1)
    mc_error = np.sqrt(np.clip(frac * (1.0 - frac), 0.0, None) / n_mc)
    return frac, mc_error


def _pahc_weight(limit8_grid, p_pahc, x):
    """`P_PAHC` interpolated to the flux limit `x` (a scalar mJy, the
    tile's own mean I4 limit -- an approximation of sec. 4's per-pixel
    contrast, disclosed rather than resampled per pixel, since `P_PAHC`
    is tabulated on `LIMIT8_GRID_MJY`'s eight region-wide quantile nodes,
    not per pixel): vectorised over stars via `searchsorted`, no
    per-star loop."""
    j = np.clip(np.searchsorted(limit8_grid, x), 1, limit8_grid.size - 1)
    lo, hi = j - 1, j
    frac = (x - limit8_grid[lo]) / (limit8_grid[hi] - limit8_grid[lo])
    return p_pahc[:, lo] + frac * (p_pahc[:, hi] - p_pahc[:, lo])


def _build_one_tile(config, region, tile_id, pix_in_tile, a_col_in_tile, f_lim_in_tile):
    """One tile's `(frac_star, frac_agb, frac_pahc, mc_star, mc_agb,
    mc_pahc, density_star, density_agb, density_pahc)`, over its own
    admitted pixels, from the star-family population's own retained
    sample (`population/star/population_star_tile__R.hdf5`'s `tile_<id>`
    group): STAR/AGB/PAHC all draw from the SAME TRILEGAL flux table
    (sec. 8's members list), each with its own weight column and its own
    Monte Carlo resample, so the three densities carry independent
    binomial noise rather than the same draw reweighted after the fact."""
    star_path = config_module.product_path(
        config, "population", "star", "population", "tile", region=region)
    field_path = config_module.product_path(
        config, "population", "trilegal", "field-stars", "region", region=region)
    with h5py.File(star_path, "r") as f:
        omega_sim = float(f.attrs["OMEGA_SIM_DEG2"])
        limit8_grid = np.asarray(f["LIMIT8_GRID_MJY"][()], dtype=np.float64)
        grp = f[f"tile_{tile_id}"]
        star_index = np.asarray(grp["STAR_INDEX"][()], dtype=np.int64)
        u = np.asarray(grp["U"][()], dtype=np.float64)
        w_star = np.asarray(grp["W_STAR"][()], dtype=np.float64)
        w_agb = np.asarray(grp["W_AGB"][()], dtype=np.float64)
        p_pahc_grid = np.asarray(grp["P_PAHC"][()], dtype=np.float64)  # (n_star, 8)
    with h5py.File(field_path, "r") as f:
        flux0_all = np.asarray(f["FNU_MJY"][star_index], dtype=np.float64)  # (n_star, 8)

    tile_i4_limit = float(np.mean(f_lim_in_tile[:, IDX_I4])) if pix_in_tile.size else float(limit8_grid[len(limit8_grid) // 2])
    p_pahc = _pahc_weight(limit8_grid, p_pahc_grid, tile_i4_limit)

    # STAR and PAHC partition the field population (spec sec. 5.1, 5.3;
    # coordinator ruling): a star is EITHER a STAR member or a PAHC member
    # of the Monte Carlo, weighted `W_STAR*(1-P_PAHC)` / `W_STAR*P_PAHC`,
    # so `N_CAT_STAR + N_CAT_PAHC` never exceeds the field-star count.
    w_star_only = w_star * (1.0 - p_pahc)
    w_pahc_only = w_star * p_pahc

    rng = np.random.RandomState(MC_SEED + tile_id)
    out = {}
    for cls, weight in (("STAR", w_star_only), ("AGB", w_agb), ("PAHC", w_pahc_only)):
        idx, total = _draw_members(rng, weight, N_MC)
        density = total / omega_sim  # objects deg^-2, sec. 5.1/5.2's Omega_sim
        if idx is None:
            frac = np.zeros(pix_in_tile.size)
            mc_error = np.zeros(pix_in_tile.size)
        else:
            frac, mc_error = _accepted_fraction(
                a_col_in_tile, u[idx], flux0_all[idx], f_lim_in_tile, config)
        out[cls] = (frac, mc_error, density)
    return out


_ZP_MJY = {b.key: b.vega_zero_point_jy * 1000.0 for b in definitions.BANDS}


def _gal_members(config, rng, n_mc):
    """`(flux0, u)`, GAL's Monte Carlo sample (sec. 5.4): `S` drawn from
    the counts law's own tabulated `log10 S` node
    (`bmstp.sample_gal.sample`'s `phi(S).S` weight, the same law
    `bmstp.shapes.build_gal` bins), the three IRAC colours from a galaxy
    measured at that node (`sky/derived/swire/galaxies_swire_survey.hdf5`'s
    finite-colour subset, its own `NODE` axis; a node with no measured
    galaxy borrows its nearest node that has one) -- "draw S from the law
    and colours from the node's galaxies" (coordinator ruling). The three
    Vega-magnitude colours and `S` (already I2's own flux) give I1/I3/I4
    through `definitions.BANDS`' own Vega zero points; J, H, Ks, M1 are
    unmeasured for a galaxy and held at zero flux, so the two-of-eight
    test runs on the four IRAC bands only (disclosed). `x = 1`: sec. 5.4's
    "whole column"."""
    x_law, log10_s_grid, w_law = sample_gal.sample(config)
    node_draw = rng.choice(log10_s_grid.size, size=n_mc, replace=True, p=w_law / w_law.sum())
    s_draw = 10.0 ** log10_s_grid[node_draw]

    gal_path = config_module.product_path(config, "sky/derived", "swire", "galaxies", "survey")
    with h5py.File(gal_path, "r") as f:
        node = np.asarray(f["NODE"][:], dtype=np.int64)
        c12 = np.asarray(f["COLOUR_I1I2"][:], dtype=np.float64)
        c23 = np.asarray(f["COLOUR_I2I3"][:], dtype=np.float64)
        c24 = np.asarray(f["COLOUR_I2I4"][:], dtype=np.float64)
    finite = (node >= 0) & np.isfinite(c12) & np.isfinite(c23) & np.isfinite(c24)
    node, c12, c23, c24 = node[finite], c12[finite], c23[finite], c24[finite]

    order = np.argsort(node, kind="stable")
    counts = np.bincount(node[order], minlength=log10_s_grid.size)
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    node_ids = np.arange(log10_s_grid.size)
    has = counts > 0
    nearest = node_ids.copy()
    if not has.all():
        have_idx = node_ids[has]
        nearest[~has] = have_idx[np.argmin(np.abs(node_ids[~has, None] - have_idx[None, :]), axis=1)]
    src_node = nearest[node_draw]
    within = np.minimum((rng.random(n_mc) * counts[src_node]).astype(np.int64), counts[src_node] - 1)
    gal_row = order[starts[src_node] + within]

    flux = np.zeros((n_mc, N_BANDS), dtype=np.float64)
    i1, i2, i3, i4 = (BAND_KEYS.index(k) for k in ("I1", "I2", "I3", "I4"))
    flux[:, i2] = s_draw
    flux[:, i1] = s_draw * (_ZP_MJY["I1"] / _ZP_MJY["I2"]) * 10.0 ** (-0.4 * c12[gal_row])
    flux[:, i3] = s_draw * (_ZP_MJY["I3"] / _ZP_MJY["I2"]) * 10.0 ** (0.4 * c23[gal_row])
    flux[:, i4] = s_draw * (_ZP_MJY["I4"] / _ZP_MJY["I2"]) * 10.0 ** (0.4 * c24[gal_row])
    u = np.ones(n_mc, dtype=np.float64)
    return flux, u, float(w_law.sum())


def build_region(config, region):
    """Writes `bmstp/atlas/prior_atlas_hpx512__R.hdf5` for one region: the
    admitted pixel axis (`catalog.depth_grid`), its column and coverage,
    and STAR/AGB/PAHC's `N_CAT_*`/`SHARE_*` from the per-tile Monte Carlo
    selection above. GAL/YSO/H2S columns are NaN (module docstring)."""
    with progress.Stage("bmstp.atlas", region) as st:
        pix, f_lim = _depth_grid(config, region)
        n_pix = pix.size
        coverage = _coverage(config, region, pix)
        a_col, _arm = _pixel_column(config, pix)
        tile_of_pix = _pixel_tile(config, region, pix)

        n_cat = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in CLASSES}
        mc_err = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in ("STAR", "AGB", "PAHC")}

        star_path = config_module.product_path(
            config, "population", "star", "population", "tile", region=region)
        with h5py.File(star_path, "r") as f:
            tile_ids_present = sorted(int(k.split("_")[1]) for k in f.keys()
                                       if k.startswith("tile_"))

        # only pixels with a tile enter the family draw; a pixel outside
        # the star-family footprint keeps NaN there, reported, not filled
        # (GAL, below, does not need a tile and runs on every pixel).
        usable = tile_of_pix >= 0
        tiles_here = sorted(set(int(t) for t in tile_of_pix[usable]) & set(tile_ids_present))

        n_jobs = min(int(config.n_jobs), _MAX_N_JOBS)

        def _one(tile_id):
            m = usable & (tile_of_pix == tile_id)
            return tile_id, m, _build_one_tile(config, region, tile_id, pix[m], a_col[m], f_lim[m])

        results = Parallel(n_jobs=n_jobs)(delayed(_one)(t) for t in tiles_here)
        for i, (tile_id, m, out) in enumerate(results):
            for cls in ("STAR", "AGB", "PAHC"):
                frac, mc_error, density = out[cls]
                n_cat[cls][m] = density * frac
                mc_err[cls][m] = mc_error
            st.tick(i + 1, len(tiles_here), "tiles")

        # GAL, sec. 5.4: one region-wide Monte Carlo sample (fixed seed,
        # not per tile -- GAL has no tile), evaluated at every admitted
        # pixel's own column and limits with the shared `_accepted_fraction`.
        gal_rng = np.random.RandomState(MC_SEED)
        gal_flux, gal_u, density_gal = _gal_members(config, gal_rng, N_MC)
        frac_gal, mc_gal = _accepted_fraction(a_col, gal_u, gal_flux, f_lim, config)
        n_cat["GAL"] = density_gal * frac_gal
        mc_err["GAL"] = mc_gal

        built = ("STAR", "AGB", "PAHC", "GAL")
        built_total = np.nansum([n_cat[c] for c in built], axis=0)
        share = {c: n_cat[c] / built_total for c in built}

        n_source = access.region_slice(config, region)["n_sources"]
        area_deg2 = n_pix * _HPX512_PIXEL_DEG2
        total_predicted_built = float(np.nansum(built_total) * _HPX512_PIXEL_DEG2)
        ratio = {c: float(np.nansum(n_cat[c]) * _HPX512_PIXEL_DEG2) / n_source
                 if n_source else float("nan") for c in built}
        ratio_built = total_predicted_built / n_source if n_source else float("nan")

        path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "hpx512"
            f.attrs["N_MC"] = N_MC
            f.attrs["DENSITY_GAL"] = density_gal
            f.attrs["TOTAL_PREDICTED"] = total_predicted_built
            f.attrs["TOTAL_OBSERVED"] = float(n_source)
            for c in built:
                f.attrs[f"RATIO_{c}"] = ratio[c]
            f.attrs["RATIO_BUILT"] = ratio_built
            f.create_dataset("HPX_PIX_512", data=pix)
            f.create_dataset("A_COL_K", data=a_col.astype(np.float32))
            f.create_dataset("COVERAGE", data=coverage.astype(np.float32))
            f.create_dataset("F_LIM_50_MED_MJY", data=f_lim.astype(np.float32))
            for c in CLASSES:
                f.create_dataset(f"N_CAT_{c}", data=n_cat[c].astype(np.float32))
            for c in built:
                f.create_dataset(f"SHARE_{c}", data=share[c].astype(np.float32))
            for c in ("YSO", "H2S"):
                f.create_dataset(f"SHARE_{c}", data=np.full(n_pix, np.nan, dtype=np.float32))

        def _max_mc_err(c):
            frac_c = n_cat[c] / (built_total + 1e-300)
            return float(np.nanmax(mc_err[c][frac_c > 0.1])) if np.any(frac_c > 0.1) else 0.0

        max_mc_err = max((_max_mc_err(c) for c in built), default=0.0) if n_pix else 0.0
        st.done(path, n_pix=n_pix, n_tile=len(tiles_here), area_deg2=area_deg2,
                total_predicted_built=total_predicted_built, total_observed=n_source,
                ratio_star=ratio["STAR"], ratio_agb=ratio["AGB"], ratio_pahc=ratio["PAHC"],
                ratio_gal=ratio["GAL"], ratio_built=ratio_built,
                density_gal_deg2=density_gal, mc_error_max_where_frac_gt_0p1=max_mc_err)
    return path


def build(config, regions=None):
    """`build(config, regions=None)`: per region, `build_region` (rule
    5c's per-region product, one file per region)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region(config, region)


if __name__ == "__main__":
    run(build)
