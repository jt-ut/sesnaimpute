"""The shape-grid builder: bins each class's population sample onto the
common grid (`bmstp.grid`) and writes the shape-grid products of
IMPLEMENTATION_BMSTP_DRAFT.md sec. 1.2 (SPEC_BMSTP_DRAFT.md sec. 2, sec.
4.1's "shape grids", sec. 5.1-5.6).

Writes P2 (the star-family grids, per tile), P3 (the cloud-class grid,
per sightline: YSO's `GRID_YSO`/`X_MARGINAL`, plus the region's H2S
Gaussian carried as two numbers, sec. 5.6 "Marks" -- H2S's own grid is
the outer product of `X_MARGINAL` and that Gaussian, formed at read, not
stored), and P4 (the galaxy grid, survey-wide).
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.bmstp import grid, sample_cloud, sample_gal, sample_star

#: joblib worker cap for the grain loop (CODING_RULES_BMSTP.md rule 10a):
#: `root.cfg`'s own `[run] n_jobs`, never more than four.
_MAX_N_JOBS = 4


def _sigma_b_min_star_family(region):
    """The region's distance uncertainty carried onto `-2 log10 d`
    (SPEC_BMSTP_DRAFT.md sec. 2 "minimum widths"), from `constants.REGIONS`
    (`regions.Region.d_r_pc`, `sigma_pc`; no separate region-depth product
    is read here): `log10 B` for the star family runs `-2 log10(d/1 kpc)`,
    so to first order `sigma(log10 B) = 2 * sigma_pc / (d_r_pc * ln 10)`."""
    r = regions_module.REGIONS_BY_NAME[region]
    return 2.0 * r.sigma_pc / (r.d_r_pc * np.log(10.0))


def _build_one_tile(config, region, tile_id, sigma_b_min):
    """One tile's `(GRID_STAR, MASS_OUTSIDE_STAR, GRID_AGB,
    MASS_OUTSIDE_AGB, sum_check)`, `sum_check` the max, over the two
    grids, of the pre-floor identity `|H.sum() - (1 - mass_outside)|`
    (an identity of `grid.bin`, SPEC_BMSTP_DRAFT.md sec. 9's normalisation
    check for this class)."""
    x_s, b_s, w_s = sample_star.sample_star(config, region, tile_id)
    h_star, mo_star = grid.bin(x_s, b_s, w_s, grid.LOG10_B_ORIGIN_TEMPLATE, sigma_b_min)
    x_a, b_a, w_a = sample_star.sample_agb(config, region, tile_id)
    h_agb, mo_agb = grid.bin(x_a, b_a, w_a, grid.LOG10_B_ORIGIN_TEMPLATE, sigma_b_min)
    sum_check = max(abs(h_star.sum() - (1.0 - mo_star)), abs(h_agb.sum() - (1.0 - mo_agb)))
    return h_star.astype(np.float32), mo_star, h_agb.astype(np.float32), mo_agb, sum_check


def build_star_family(config, region):
    """Writes P2, `bmstp/shape/star_shape_tile__R.hdf5`: `GRID_STAR` and
    `GRID_AGB` per tile (sec. 5.1, 5.2), `MASS_OUTSIDE_STAR`/`_AGB`, the
    grid edges and the star-count-per-tile attribute. PAHC has no grid of
    its own: a reader loads `GRID_STAR` for it (sec. 5.3 "Grain")."""
    with progress.Stage("bmstp.shapes.star_family", region) as st:
        ids = sample_star.tile_ids(config, region)
        n_tile = ids.size
        sigma_b_min = _sigma_b_min_star_family(region)

        n_jobs = min(int(config.n_jobs), _MAX_N_JOBS)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_build_one_tile)(config, region, int(t), sigma_b_min) for t in ids)
        for i in range(n_tile):
            st.tick(i + 1, n_tile, "tiles")

        grid_star = np.stack([r[0] for r in results])
        mass_outside_star = np.array([r[1] for r in results], dtype=np.float32)
        grid_agb = np.stack([r[2] for r in results])
        mass_outside_agb = np.array([r[3] for r in results], dtype=np.float32)
        max_sum_check = max(r[4] for r in results) if results else 0.0

        # star counts per tile (module docstring's `N_STARS_PER_TILE`):
        # the tile's own retained-star row count, read once more cheaply
        # than re-opening every tile group inside `_build_one_tile`.
        n_stars_per_tile = _n_stars_per_tile(config, region, ids)

        path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "tile"
            f.attrs["FLOOR"] = grid.FLOOR
            f.attrs["N_STARS_PER_TILE"] = n_stars_per_tile
            f.create_dataset("LOG10_X_EDGES", data=grid.LOG10_X_EDGES)
            f.create_dataset("LOG10_B_EDGES", data=grid.log10_b_edges(grid.LOG10_B_ORIGIN_TEMPLATE))
            f.attrs["LOG10_B_ORIGIN"] = grid.LOG10_B_ORIGIN_TEMPLATE
            f.create_dataset("TILE_ID", data=ids.astype(np.int32))
            f.create_dataset("GRID_STAR", data=grid_star)
            f.create_dataset("GRID_AGB", data=grid_agb)
            f.create_dataset("MASS_OUTSIDE_STAR", data=mass_outside_star)
            f.create_dataset("MASS_OUTSIDE_AGB", data=mass_outside_agb)

        st.done(path, n_tile=int(n_tile),
                mass_outside_star_max=float(mass_outside_star.max()) if n_tile else 0.0,
                mass_outside_agb_max=float(mass_outside_agb.max()) if n_tile else 0.0,
                sum_check_max=float(max_sum_check))
    return path, n_tile, mass_outside_star, mass_outside_agb, max_sum_check


def _n_stars_per_tile(config, region, ids):
    import h5py as _h5py
    path = config_module.product_path(
        config, "population", "star", "population", "tile", region=region)
    with _h5py.File(path, "r") as f:
        return np.array([f[f"tile_{int(t)}"]["U"].shape[0] for t in ids], dtype=np.int32)


def _build_one_sightline(loaded, row, sigma_b_min):
    """One sightline's `(GRID_YSO, MASS_OUTSIDE_YSO, sum_check)`,
    `sum_check` the pre-floor identity `|H.sum() - (1 - mass_outside)|`
    (an identity of `grid.bin`)."""
    x, log10_b, w = sample_cloud.sample_yso(loaded, row)
    h, mass_outside = grid.bin(x, log10_b, w, grid.LOG10_B_ORIGIN_TEMPLATE, sigma_b_min)
    sum_check = abs(h.sum() - (1.0 - mass_outside))
    return h.astype(np.float32), mass_outside, sum_check


def _yso_x_marginal_check(loaded, row, x_marginal):
    """On one sightline, the built `X_MARGINAL` (already blurred by one
    cell along `log10 x`, `mode="constant"`, `grid.bin`'s own minimum
    width) against the profile's own `p(u) du` rebinned onto
    `LOG10_X_EDGES` (sec. 5.5's acceptance): the cell masses redistributed
    onto the common grid's bins by linear interpolation of the cumulative
    mass on the profile's native `log10 u` edges (the "rebin"), then
    smoothed by the SAME one-cell, zero-padded Gaussian `grid.bin` applies
    -- so the reference carries the same edge treatment the built grid
    does, not a bare unsmoothed rebin. Compared cumulative-sum to
    cumulative-sum against `X_MARGINAL`'s own cumulative sum, per the
    brief. Returns the max absolute difference of the two cumulative
    arrays."""
    u_edges = loaded["u_edges"][row]
    p_u = loaded["p_u"][row]
    mass = p_u * np.diff(u_edges)
    log10u_edges = np.log10(np.maximum(u_edges, sample_cloud._X_FLOOR))
    ref_cum_at_edges = np.interp(grid.LOG10_X_EDGES, log10u_edges,
                                  np.concatenate([[0.0], np.cumsum(mass)]))
    ref_bins = np.diff(ref_cum_at_edges)
    ref_bins = gaussian_filter1d(ref_bins, sigma=1.0, mode="constant")
    ref_cum = np.concatenate([[0.0], np.cumsum(ref_bins)])
    grid_cum = np.concatenate([[0.0], np.cumsum(x_marginal)])
    return float(np.max(np.abs(ref_cum - grid_cum)))


def build_cloud(config, region):
    """Writes P3, `bmstp/shape/cloud_shape_sightline__R.hdf5`: `GRID_YSO`
    and its `log10 x` marginal per sightline (sec. 5.5), `MASS_OUTSIDE_YSO`,
    and the region's H2S brightness Gaussian (`LOGSIG_MEAN`, `LOGSIG_STD`,
    sec. 5.6) read from `population/h2s/prior_h2s_region__R.hdf5` and
    carried as attributes -- H2S has no grid of its own (sec. 5.6
    "Marks": separable, `X_MARGINAL` times this Gaussian, formed at read)."""
    with progress.Stage("bmstp.shapes.cloud", region) as st:
        loaded = sample_cloud._region_profile(config, region)
        n_sl = loaded["hpx_pix_256"].size
        sigma_b_min = _sigma_b_min_star_family(region)

        n_jobs = min(int(config.n_jobs), _MAX_N_JOBS)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_build_one_sightline)(loaded, row, sigma_b_min) for row in range(n_sl))
        for i in range(n_sl):
            st.tick(i + 1, n_sl, "sightlines")

        grid_yso = np.stack([r[0] for r in results]) if n_sl else np.zeros((0, grid.LOG10_X_EDGES.size - 1, grid.N_B), dtype=np.float32)
        mass_outside_yso = np.array([r[1] for r in results], dtype=np.float32)
        max_sum_check = max((r[2] for r in results), default=0.0)
        x_marginal = grid_yso.sum(axis=2).astype(np.float32)

        h2s_path = config_module.product_path(config, "population", "h2s", "prior", "region", region=region)
        with h5py.File(h2s_path, "r") as f:
            logsig_mean = float(f["LOGSIG_MEAN"][()])
            logsig_std = float(f["LOGSIG_STD"][()])

        # sec. 5.5's fixed-seed check: one sightline picked reproducibly,
        # its built `X_MARGINAL` against the profile's own `p(u) du`
        # rebinned onto the common grid.
        pick = int(np.random.RandomState(0).randint(n_sl)) if n_sl else 0
        x_marginal_check = _yso_x_marginal_check(loaded, pick, x_marginal[pick]) if n_sl else 0.0

        path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "sightline"
            f.attrs["FLOOR"] = grid.FLOOR
            f.attrs["N_SUB"] = sample_cloud.N_SUB
            f.attrs["LOGSIG_MEAN"] = logsig_mean
            f.attrs["LOGSIG_STD"] = logsig_std
            f.attrs["LOG10_B_ORIGIN"] = grid.LOG10_B_ORIGIN_TEMPLATE
            f.create_dataset("LOG10_X_EDGES", data=grid.LOG10_X_EDGES)
            f.create_dataset("LOG10_B_EDGES", data=grid.log10_b_edges(grid.LOG10_B_ORIGIN_TEMPLATE))
            f.create_dataset("HPX_PIX_256", data=loaded["hpx_pix_256"])
            f.create_dataset("GRID_YSO", data=grid_yso)
            f.create_dataset("X_MARGINAL", data=x_marginal)
            f.create_dataset("MASS_OUTSIDE_YSO", data=mass_outside_yso)

        st.done(path, n_sightline=int(n_sl),
                mass_outside_yso_max=float(mass_outside_yso.max()) if n_sl else 0.0,
                sum_check_max=float(max_sum_check),
                x_marginal_check_sightline=pick,
                x_marginal_check_max_abs_diff=float(x_marginal_check))
    return path, n_sl, mass_outside_yso, max_sum_check, x_marginal_check


def build_gal(config):
    """Writes P4, `bmstp/shape/gal_shape_survey.hdf5`: the one
    survey-wide GAL grid (sec. 5.4 "Grain"). No region distance applies to
    a survey-wide flux axis, so the brightness-axis smoothing is the
    one-cell floor only (`sigma_b_min = 0`, `grid.bin`'s own minimum)."""
    with progress.Stage("bmstp.shapes.gal") as st:
        x, log10_b, w = sample_gal.sample(config)
        origin = float(log10_b[0])
        h, mass_outside = grid.bin(x, log10_b, w, origin, 0.0)
        sum_check = abs(h.sum() - (1.0 - mass_outside))
        density_gal = float(w.sum())

        path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            f.attrs["FLOOR"] = grid.FLOOR
            f.attrs["LOG10_B_ORIGIN"] = origin
            f.attrs["DENSITY_GAL"] = density_gal
            f.attrs["COSMIC_VARIANCE_DEX"] = _gal_cosmic_variance_dex(config)
            f.create_dataset("LOG10_X_EDGES", data=grid.LOG10_X_EDGES)
            f.create_dataset("LOG10_B_EDGES", data=grid.log10_b_edges(origin))
            f.create_dataset("GRID", data=h.astype(np.float32))

        st.done(path, mass_outside=float(mass_outside), sum_check=float(sum_check),
                density_gal_deg2=density_gal)
    return path, mass_outside, sum_check


def _gal_cosmic_variance_dex(config):
    path = config_module.product_path(config, "population", "gal", "counts", "survey")
    with h5py.File(path, "r") as f:
        return float(f["COSMIC_VARIANCE_DEX"][()])


def build(config, regions=None):
    """Per region, P2 (the star family) and P3 (the cloud class); once,
    P4 (GAL, survey-wide, built when `regions` includes the survey's
    first region or the file is absent -- IMPLEMENTATION_BMSTP_DRAFT.md
    sec. 1.2 P4)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    for region in region_names:
        build_star_family(config, region)
        build_cloud(config, region)

    gal_path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
    first_region = regions_module.REGIONS[0].name
    if first_region in region_names or not os.path.exists(gal_path):
        build_gal(config)


if __name__ == "__main__":
    run(build)
