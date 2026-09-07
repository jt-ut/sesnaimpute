"""P6, the prior atlas (SPEC_BMSTP_DRAFT.md sec. 8; IMPLEMENTATION_BMSTP_DRAFT.md
sec. 1.2 P6, sec. 3 row 1.9).

Per admitted nside-512 pixel, a fixed-seed Monte Carlo sample of each class's
population, dimmed at the pixel's own column through the blended law
(`population.selection.kappa_hybrid`), counted by the two-of-eight test at the
pixel's own median 50% limits (`catalog.depth_grid`'s `F_LIM_50_MED_MJY`). The
selection appears here and nowhere else in the atlas (sec. 1.2): the prior itself
is unthinned.

This build writes the STAR/AGB/PAHC family (sec. 5.1-5.3), the first and
highest-value class the brief names. GAL, YSO and H2S (sec. 5.4-5.6) did not fit
this unit's 15-minute budget on top of the family's own read chain (see the
module docstring's "What did not run" in the delivery report); their `N_CAT_*`/
`SHARE_*` columns are written as NaN and excluded from the total-count check,
which is therefore reported for STAR+AGB+PAHC only, not the full six.
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


def _pixel_column(config, region, pix):
    """`A_COL_K` at each admitted pixel: `population.anchor_tiles`'s own
    per-pixel column (`population/anchors/histograms/hpx512__R.hdf5`,
    `A_PIX_K`), the same nside-512 pixel axis the star-family tile
    definition (`.../anchors/tiles/hpx512__R.hdf5`) uses -- the "column map
    at the pixel" the brief names, joined by pixel id (a pixel absent from
    the anchor product, outside the star-family footprint, is dropped, not
    zero-filled, since a column of zero is not a measurement)."""
    path = config_module.product_path(config, "population", "anchors", "histograms", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        a_pix_id = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        a_pix_k = np.asarray(f["A_PIX_K"][:], dtype=np.float64)
    order = np.argsort(a_pix_id)
    loc = np.searchsorted(a_pix_id[order], pix)
    loc = np.minimum(loc, a_pix_id.size - 1)
    hit = order[loc]
    found = a_pix_id[hit] == pix
    a_col = np.full(pix.size, np.nan, dtype=np.float64)
    a_col[found] = a_pix_k[hit[found]]
    return a_col, found


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

    rng = np.random.RandomState(MC_SEED + tile_id)
    out = {}
    for cls, weight in (("STAR", w_star), ("AGB", w_agb), ("PAHC", p_pahc)):
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


def build_region(config, region):
    """Writes `bmstp/atlas/prior_atlas_hpx512__R.hdf5` for one region: the
    admitted pixel axis (`catalog.depth_grid`), its column and coverage,
    and STAR/AGB/PAHC's `N_CAT_*`/`SHARE_*` from the per-tile Monte Carlo
    selection above. GAL/YSO/H2S columns are NaN (module docstring)."""
    with progress.Stage("bmstp.atlas", region) as st:
        pix, f_lim = _depth_grid(config, region)
        n_pix = pix.size
        coverage = _coverage(config, region, pix)
        a_col, has_col = _pixel_column(config, region, pix)
        tile_of_pix = _pixel_tile(config, region, pix)

        n_cat = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in CLASSES}
        mc_err = {c: np.full(n_pix, np.nan, dtype=np.float64) for c in ("STAR", "AGB", "PAHC")}

        star_path = config_module.product_path(
            config, "population", "star", "population", "tile", region=region)
        with h5py.File(star_path, "r") as f:
            tile_ids_present = sorted(int(k.split("_")[1]) for k in f.keys()
                                       if k.startswith("tile_"))

        # only pixels with both a tile and a column enter the family draw
        # (sec. 8's placement needs both); a pixel outside the star-family
        # footprint keeps NaN, reported, not filled.
        usable = has_col & (tile_of_pix >= 0)
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

        family_total = np.nansum([n_cat["STAR"], n_cat["AGB"], n_cat["PAHC"]], axis=0)
        share = {c: n_cat[c] / family_total for c in ("STAR", "AGB", "PAHC")}

        n_source = access.region_slice(config, region)["n_sources"]
        area_deg2 = n_pix * _HPX512_PIXEL_DEG2
        total_predicted_family = float(np.nansum(family_total) * _HPX512_PIXEL_DEG2)
        ratio = {c: float(np.nansum(n_cat[c]) * _HPX512_PIXEL_DEG2) / n_source
                 if n_source else float("nan") for c in ("STAR", "AGB", "PAHC")}
        ratio_family = total_predicted_family / n_source if n_source else float("nan")

        path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "hpx512"
            f.attrs["N_MC"] = N_MC
            f.attrs["TOTAL_PREDICTED"] = total_predicted_family
            f.attrs["TOTAL_OBSERVED"] = float(n_source)
            for c in ("STAR", "AGB", "PAHC"):
                f.attrs[f"RATIO_{c}"] = ratio[c]
            f.attrs["RATIO_STAR_AGB_PAHC"] = ratio_family
            f.create_dataset("HPX_PIX_512", data=pix)
            f.create_dataset("A_COL_K", data=a_col.astype(np.float32))
            f.create_dataset("COVERAGE", data=coverage.astype(np.float32))
            f.create_dataset("F_LIM_50_MED_MJY", data=f_lim.astype(np.float32))
            for c in CLASSES:
                f.create_dataset(f"N_CAT_{c}", data=n_cat[c].astype(np.float32))
            for c in ("STAR", "AGB", "PAHC"):
                f.create_dataset(f"SHARE_{c}", data=share[c].astype(np.float32))
            for c in ("GAL", "YSO", "H2S"):
                f.create_dataset(f"SHARE_{c}", data=np.full(n_pix, np.nan, dtype=np.float32))

        max_mc_err = float(np.nanmax([mc_err[c][n_cat[c] / (family_total + 1e-300) > 0.1]
                                       for c in ("STAR", "AGB", "PAHC")
                                       if np.any(n_cat[c] / (family_total + 1e-300) > 0.1)] or [0.0])
                            ) if n_pix else 0.0
        st.done(path, n_pix=n_pix, n_tile=len(tiles_here), area_deg2=area_deg2,
                total_predicted_family=total_predicted_family, total_observed=n_source,
                ratio_star=ratio["STAR"], ratio_agb=ratio["AGB"], ratio_pahc=ratio["PAHC"],
                ratio_family=ratio_family, mc_error_max_where_frac_gt_0p1=max_mc_err)
    return path


def build(config, regions=None):
    """`build(config, regions=None)`: per region, `build_region` (rule
    5c's per-region product, one file per region)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region(config, region)


if __name__ == "__main__":
    run(build)
