"""The shape-grid builder: bins each class's population sample onto the
common `(log10 x, log10 F_4.5)` grid (`bmstp.grid`) and writes the
shape-grid products of IMPLEMENTATION_BMSTP_DRAFT.md sec. 1.2
(SPEC_BMSTP_DRAFT.md sec. 2, sec. 4.1's "shape grids", sec. 5.1-5.6).

Writes P2 (the star-family grids, per tile), P3 (the cloud-class grid,
per sightline: YSO's `GRID_YSO`/`X_MARGINAL`, and H2S's own
`GRID_H2S` on the SAME common grid -- `GRID_H2S[x, F] = p_x(x) . [L_Sigma
(*) K_c](F)`, `p_x` the sightline's own `X_MARGINAL`, `L_Sigma` the
region's knot lognormal (`LOGSIG_MEAN`/`LOGSIG_STD`, kept as attributes),
`K_c` the distribution of the h2shock templates' own Sigma-to-4.5-micron
conversions under their (uniform) weights -- one convolution, survey-wide
in its template content, computed once per region and broadcast over
sightlines through `p_x` alone, sec. 5.6 "Marks"), and P4 (the galaxy
grid, survey-wide). Every brightness axis is the one common
`LOG10_F45_EDGES` (sec. 2): no per-shape origin, not even H2S's.

Before each product is overwritten, this module reads whatever the SAME
path already holds (rule: acceptance is read before the write that would
erase it) and reports the identity the brief names against the freshly
built product.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.bmstp import grid, sample_cloud, sample_gal, sample_star, template_weights
from sesnaimpute.population import h2s as h2s_module


# ---------------------------------------------------------------------------
# P2 -- the star-family grids, per tile (sec. 5.1, 5.2)
# ---------------------------------------------------------------------------

def _build_one_tile(config, region, tile_id):
    """One tile's `(GRID_STAR, MASS_OUTSIDE_STAR, GRID_AGB,
    MASS_OUTSIDE_AGB, sum_check, above_star_w, total_star_w, above_agb_w,
    total_agb_w)`, `sum_check` the max, over the two grids, of the
    pre-floor identity `|H.sum() - (1 - mass_outside)|` (an identity of
    `grid.bin`, SPEC_BMSTP_DRAFT.md sec. 9's normalisation check for this
    class). `above_*_w`/`total_*_w` are each class's own RAW (unnormalised)
    weight above the grid's top edge and its own total weight, summed
    across tiles by the caller into the region's single mass-above-top
    fraction (sec. 2, sec. 9's 0.1% bar, W24b -- distinct from
    `MASS_OUTSIDE_*`, which also counts the mass the retention limit
    drops at the bottom edge)."""
    x_s, f45_s, w_s = sample_star.sample_star(config, region, tile_id)
    x_a, f45_a, w_a = sample_star.sample_agb(config, region, tile_id)

    # W61 (sec. 2 "minimum widths", sec. 5.1 "Marks"): the field-star
    # depth mark's own width is the map's propagated column sigma at the
    # star's distance, not a fixed one cell -- AGB follows STAR (the same
    # stars, sec. 5.2), so both read the SAME tile width classes.
    row, sigma_classes_dex = sample_star.tile_width_classes(config, region, tile_id)
    sigma_classes_cells = sigma_classes_dex / grid._X_CELL_WIDTH
    class_s = sample_star.star_width_class(
        config, region, sample_star.star_distances(config, region, tile_id), row, sigma_classes_dex)
    class_a = sample_star.star_width_class(
        config, region, sample_star.agb_star_distances(config, region, tile_id), row, sigma_classes_dex)

    h_star, mo_star = grid.bin_star_widths(x_s, f45_s, w_s, class_s, sigma_classes_cells)
    h_agb, mo_agb = grid.bin_star_widths(x_a, f45_a, w_a, class_a, sigma_classes_cells)
    sum_check = max(abs(h_star.sum() - (1.0 - mo_star)), abs(h_agb.sum() - (1.0 - mo_agb)))
    above_star_w = float(w_s[f45_s > grid.LOG10_F45_EDGES[-1]].sum())
    above_agb_w = float(w_a[f45_a > grid.LOG10_F45_EDGES[-1]].sum())
    return (h_star.astype(np.float32), mo_star, h_agb.astype(np.float32), mo_agb, sum_check,
            above_star_w, float(w_s.sum()), above_agb_w, float(w_a.sum()))


def _read_old_star_product(path):
    """The current P2's own `x`-marginals (`GRID_STAR`/`GRID_AGB` summed
    over the whole brightness axis), read before this build overwrites
    the file (STAR's acceptance identity, sec. 9: the depth placement is
    unchanged, only the brightness axis was redefined, so summing away
    that axis must reproduce the SAME `x` shape). `None` if no product
    exists yet (first build)."""
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        tile_id = f["TILE_ID"][()].astype(np.int64)
        x_marginal_star = f["GRID_STAR"][()].sum(axis=2)
        x_marginal_agb = f["GRID_AGB"][()].sum(axis=2)
    return dict(tile_id=tile_id, x_marginal_star=x_marginal_star, x_marginal_agb=x_marginal_agb)


def build_star_family(config, region):
    """Writes P2, `bmstp/shape/star_shape_tile__R.hdf5`: `GRID_STAR` and
    `GRID_AGB` per tile (sec. 5.1, 5.2) on the common `LOG10_F45_EDGES`,
    `MASS_OUTSIDE_STAR`/`_AGB`, `ON_GRID_STAR`/`_AGB` (`1 - mass_outside`
    per tile, sec. 2's ON-GRID FRACTION for the density stage to scale by,
    W26), the grid edges, the star-count-per-tile attribute, and AGB's own
    per-chemistry flux-to-luminosity ratio spread (`F45_PER_L_SPREAD_DEX_O/C`,
    sec. 5.2). PAHC has no grid of its own: a reader loads `GRID_STAR` for
    it (sec. 5.3 "Grain")."""
    path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
    old = _read_old_star_product(path)

    with progress.Stage("bmstp.shapes.star_family", region) as st:
        ids = sample_star.tile_ids(config, region)
        n_tile = ids.size

        # worker count is `root.cfg`'s own `[run] n_jobs` (CODING_RULES_BMSTP.md
        # rule 10a): the owner sets it to what the machine's memory allows.
        n_jobs = int(config.n_jobs)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_build_one_tile)(config, region, int(t)) for t in ids)
        for i in range(n_tile):
            st.tick(i + 1, n_tile, "tiles")

        grid_star = np.stack([r[0] for r in results])
        mass_outside_star = np.array([r[1] for r in results], dtype=np.float32)
        grid_agb = np.stack([r[2] for r in results])
        mass_outside_agb = np.array([r[3] for r in results], dtype=np.float32)
        max_sum_check = max(r[4] for r in results) if results else 0.0
        on_grid_star = 1.0 - mass_outside_star
        on_grid_agb = 1.0 - mass_outside_agb

        # the region's own mass-above-top-edge fraction per class (sec. 2,
        # sec. 9's 0.1% bar, W24b): the tiles' own raw weights summed
        # first, then divided -- NOT a mean of per-tile fractions, since
        # tiles hold unequal star counts.
        above_star_w = sum(r[5] for r in results)
        total_star_w = sum(r[6] for r in results)
        above_agb_w = sum(r[7] for r in results)
        total_agb_w = sum(r[8] for r in results)
        above_top_star = above_star_w / total_star_w if total_star_w > 0 else 0.0
        above_top_agb = above_agb_w / total_agb_w if total_agb_w > 0 else 0.0

        # star counts per tile (module docstring's `N_STARS_PER_TILE`):
        # the tile's own retained-star row count, read once more cheaply
        # than re-opening every tile group inside `_build_one_tile`.
        n_stars_per_tile = _n_stars_per_tile(config, region, ids)

        # AGB's own flux-to-luminosity ratio spread by chemistry (sec.
        # 5.2): survey-wide, cached, computed once regardless of region.
        agb_ratio = sample_star.agb_log10_ratio_stats(config.data_root)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "tile"
            f.attrs["FLOOR"] = grid.FLOOR
            f.attrs["N_STARS_PER_TILE"] = n_stars_per_tile
            f.attrs["F45_PER_L_SPREAD_DEX_O"] = agb_ratio["O"]["spread_dex"]
            f.attrs["F45_PER_L_SPREAD_DEX_C"] = agb_ratio["C"]["spread_dex"]
            f.create_dataset("LOG10_X_EDGES", data=grid.LOG10_X_EDGES)
            f.create_dataset("LOG10_F45_EDGES", data=grid.LOG10_F45_EDGES)
            f.create_dataset("TILE_ID", data=ids.astype(np.int32))
            f.create_dataset("GRID_STAR", data=grid_star)
            f.create_dataset("GRID_AGB", data=grid_agb)
            f.create_dataset("MASS_OUTSIDE_STAR", data=mass_outside_star)
            f.create_dataset("MASS_OUTSIDE_AGB", data=mass_outside_agb)
            f.create_dataset("ON_GRID_STAR", data=on_grid_star.astype(np.float32))
            f.create_dataset("ON_GRID_AGB", data=on_grid_agb.astype(np.float32))

        # STAR's acceptance identity (sec. 9): the new `x`-marginal against
        # the OLD product's own, read above before the overwrite.
        max_x_marginal_dev = float("nan")
        if old is not None and np.array_equal(old["tile_id"], ids):
            new_x_marginal_star = grid_star.sum(axis=2)
            max_x_marginal_dev = float(
                np.max(np.abs(old["x_marginal_star"] - new_x_marginal_star)))

        # STAR's own F_4.5 range against the field-stars product's own
        # flux column (sec. 9's "must agree, since it is the same
        # numbers"): report only, the field-stars flux is what `GRID_STAR`
        # was binned from directly.
        f45_lo, f45_hi = _field_star_f45_range(config, region)

        for chem in ("O", "C"):
            if agb_ratio[chem]["spread_dex"] > sample_star.AGB_RATIO_SPREAD_FLAG_DEX:
                print(f"bmstp.shapes.star_family: {region}: AGB chemistry={chem} "
                      f"flux-to-luminosity ratio spread {agb_ratio[chem]['spread_dex']:.3f} dex "
                      f"exceeds the {sample_star.AGB_RATIO_SPREAD_FLAG_DEX} dex flag bar "
                      f"(n={agb_ratio[chem]['n']}); the chemistry median is used anyway "
                      f"(sec. 5.2)")

        st.done(path, n_tile=int(n_tile),
                mass_outside_star_max=float(mass_outside_star.max()) if n_tile else 0.0,
                mass_outside_agb_max=float(mass_outside_agb.max()) if n_tile else 0.0,
                sum_check_max=float(max_sum_check),
                x_marginal_max_dev=max_x_marginal_dev,
                f45_range_field_stars=(f45_lo, f45_hi),
                mass_above_top_star=float(above_top_star),
                mass_above_top_agb=float(above_top_agb),
                on_grid_star_range=(float(on_grid_star.min()), float(on_grid_star.max())) if n_tile else (0.0, 0.0),
                on_grid_agb_range=(float(on_grid_agb.min()), float(on_grid_agb.max())) if n_tile else (0.0, 0.0))
    return (path, n_tile, mass_outside_star, mass_outside_agb, max_sum_check, max_x_marginal_dev,
            above_top_star, above_top_agb, on_grid_star, on_grid_agb)


def _n_stars_per_tile(config, region, ids):
    path = config_module.product_path(
        config, "population", "star", "population", "tile", region=region)
    with h5py.File(path, "r") as f:
        return np.array([f[f"tile_{int(t)}"]["U"].shape[0] for t in ids], dtype=np.int32)


def _field_star_f45_range(config, region):
    """`(log10 f45 min, max)` over the region's retained field-stars
    sample's own 4.5 micron flux (sec. 9's STAR check): the same column
    `sample_star.sample_star` reads through `STAR_INDEX`, read here
    directly and unindexed since every retained star contributes to some
    tile."""
    path = config_module.product_path(
        config, "population", "trilegal", "field-stars", "region", region=region)
    with h5py.File(path, "r") as f:
        fnu_i2 = f["FNU_MJY"][:, sample_star.IDX_I2].astype(np.float64)
    log10_f45 = np.log10(fnu_i2)
    return float(log10_f45.min()), float(log10_f45.max())


# ---------------------------------------------------------------------------
# P3 -- the cloud-class grid, per sightline (sec. 5.5)
# ---------------------------------------------------------------------------

def _build_one_sightline(loaded, row, p_ref, kernel_1d, d_front, d_back):
    """One sightline's `(GRID_YSO, X_MARGINAL, MASS_OUTSIDE_YSO,
    removed_frac)` (sec. 5.5 "Marks"): every depth
    sub-sample (`sample_cloud._cell_subsamples`, weight `w_k,sub`, depth
    `x_k,sub`, distance `d_k,sub`) adds its own shifted-and-smoothed copy
    of `p_ref` to its own `log10 x` row -- grouped by row first (linear
    in the sub-samples, so summing the row's own raw weighted `delta =
    -2 log10(d_k,sub / 1 kpc)` histogram before the ONE EXACT Gaussian
    smoothing (`kernel_1d`, `sample_cloud.exact_gaussian_kernel`) and the
    ONE convolution with `p_ref` reproduces the per-sub-sample sum
    exactly). `X_MARGINAL` (`p_x`) is unchanged in value
    (`sample_cloud.sample_x`, its own one-cell-smoothed bin).
    `MASS_OUTSIDE_YSO` is the combined shortfall of `GRID_YSO`'s own
    total against the sightline's intended mass, `sum(w_k,sub) *
    p_ref.sum()`."""
    p_x, _mo_x, removed_frac = sample_cloud.sample_x(loaded, row, d_front, d_back)
    log10x_nudged, w_sub, d_sub, _removed = sample_cloud._cell_subsamples(loaded, row, d_front, d_back)
    n_x = grid.LOG10_X_EDGES.size - 1
    n_b = grid.LOG10_F45_EDGES.size - 1

    x_idx = np.digitize(log10x_nudged, grid.LOG10_X_EDGES) - 1
    on_x = (x_idx >= 0) & (x_idx < n_x)
    delta = -2.0 * np.log10(d_sub / 1000.0)
    # the row's own raw (unsmoothed) weighted delta histogram (sec. 5.5,
    # the shift kernel K, restricted to this one row's own sub-samples):
    # a single 2-D histogram call over every kept sub-sample at once.
    k_row, _, _ = np.histogram2d(
        x_idx[on_x], delta[on_x], bins=[np.arange(n_x + 1), grid.LOG10_F45_EDGES],
        weights=w_sub[on_x])
    # the EXACT Gaussian smoothing (never `gaussian_filter1d`, sec. 9's
    # ruling 11 identity: see `exact_gaussian_kernel`'s own docstring),
    # one vectorised row-wise convolution for every x row at once.
    k_row = ndimage.convolve1d(k_row, weights=kernel_1d, axis=1, mode="constant")

    # the row's own convolution with p_ref (sec. 5.5): both
    # are histograms on the SAME 0.1 dex grid sharing the same origin, so
    # `np.convolve`'s left-edge index convention adds their bin indices
    # directly -- the full convolution's own index n = j + m corresponds
    # to the common grid's index n + LOG10_F45_EDGES[0]/D_LOG10_F45 (an
    # exact integer, the grid runs from -4.0 in steps of 0.1).
    conv = fftconvolve(k_row, p_ref[None, :], mode="full", axes=1)
    start = int(round(-grid.LOG10_F45_EDGES[0] / grid.D_LOG10_F45))
    grid_yso = np.maximum(conv[:, start:start + n_b], 0.0)
    # the one-cell smoothing along x (sec. 2 "minimum widths") applies as
    # now: `sample_x`'s own p_x already carries it; GRID_YSO gets its own
    # pass here since its x-axis rows were built from the raw (unsmoothed)
    # sub-sample scatter above, not from p_x.
    grid_yso = gaussian_filter1d(grid_yso, sigma=1.0, axis=0, mode="constant")

    total_intended = float(w_sub.sum()) * float(p_ref.sum())
    total_actual = float(grid_yso.sum())
    mass_outside_yso = (0.0 if total_intended <= 0.0
                         else float(np.clip(1.0 - total_actual / total_intended, 0.0, 1.0)))
    peak = grid_yso.max()
    grid_yso = np.maximum(grid_yso, grid.FLOOR * peak) if peak > 0 else np.full_like(grid_yso, grid.FLOOR)
    return grid_yso.astype(np.float32), p_x.astype(np.float32), mass_outside_yso, removed_frac


def build_cloud(config, region):
    """Writes P3, `bmstp/shape/cloud_shape_sightline__R.hdf5`: `GRID_YSO`
    (sec. 5.5, the shift-kernel rule -- each depth sub-sample's own shifted copy of
    `P_ref`, `bmstp.sample_cloud.p_ref_f45`, summed by `log10 x` row, NOT
    an outer product) and its `log10 x` marginal per sightline,
    `MASS_OUTSIDE_YSO`, `ON_GRID_YSO` (`1 - mass_outside_yso` per
    sightline, sec. 2's ON-GRID FRACTION, W26's own read), the cloud
    interval `D_FRONT_PC`/`D_BACK_PC` (DOUBLED about the region's own peak
    distance, W24b), and the region's H2S brightness Gaussian (`LOGSIG_MEAN`,
    `LOGSIG_STD`, sec. 5.6) computed here by transporting the UWISH2 knot
    survey's surface-brightness sample to the region's own distance
    (`population.h2s.transport_log10_sigma`,
    `population.h2s.region_sigma_lognormal`) -- H2S has no grid of its own
    (sec. 5.6 "Marks": separable, `X_MARGINAL` times this Gaussian, formed
    at read on the common `LOG10_F45_EDGES` through the template's own
    `C_THETA` offset, never a class-specific origin)."""
    p3_path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    old_x_marginal = None
    if os.path.exists(p3_path):
        with h5py.File(p3_path, "r") as f:
            old_x_marginal = f["X_MARGINAL"][()]

    with progress.Stage("bmstp.shapes.cloud", region) as st:
        loaded = sample_cloud._region_profile(config, region)
        n_sl = loaded["hpx_pix_256"].size
        d_front, d_back = sample_cloud.cloud_interval_pc(config, region)
        r = regions_module.REGIONS_BY_NAME[region]

        # P_ref (sec. 5.5), survey-wide, unplaced -- built
        # once, not per sightline or region; every sub-sample's own row
        # convolves it with that row's own shift kernel below.
        p_ref = sample_cloud.p_ref_f45(config)
        sigma_d = sample_cloud.sigma_d_dex(r)
        kernel_1d = sample_cloud.exact_gaussian_kernel(sigma_d)

        # worker count is `root.cfg`'s own `[run] n_jobs` (CODING_RULES_BMSTP.md
        # rule 10a): the owner sets it to what the machine's memory allows.
        n_jobs = int(config.n_jobs)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_build_one_sightline)(loaded, row, p_ref, kernel_1d, d_front, d_back)
            for row in range(n_sl))
        for i in range(n_sl):
            st.tick(i + 1, n_sl, "sightlines")

        n_b = grid.LOG10_F45_EDGES.size - 1
        n_x = grid.LOG10_X_EDGES.size - 1
        grid_yso = (np.stack([r_[0] for r_ in results]) if n_sl
                    else np.zeros((0, n_x, n_b), dtype=np.float32))
        x_marginal = (np.stack([r_[1] for r_ in results]) if n_sl
                      else np.zeros((0, n_x), dtype=np.float32))
        mass_outside_yso = (np.array([r_[2] for r_ in results], dtype=np.float64)
                             if n_sl else np.zeros((0,))).astype(np.float32)
        removed_frac = np.array([r_[3] for r_ in results], dtype=np.float64)
        on_grid_yso = (1.0 - mass_outside_yso).astype(np.float32)

        # H2S's brightness lognormal (sec. 5.6 "Marks"): the UWISH2 knot
        # survey's reference sample, transported from each knot's own
        # field distance to THIS region's distance.
        log10_sb_native, area_pc2 = h2s_module.uwish2_reference_knots(config)
        log10_sigma = h2s_module.transport_log10_sigma(log10_sb_native, area_pc2, r.d_r_pc)
        logsig_mean, logsig_std = h2s_module.region_sigma_lognormal(log10_sigma)

        # GRID_H2S (sec. 5.6 "Marks"): GRID_H2S[x, F] = p_x(x)
        # . [L_Sigma (*) K_c](F), one marginal per region (the template
        # content and the region's own lognormal are both region-wide, not
        # per-sightline), broadcast over sightlines through X_MARGINAL
        # alone below. `log10 Sigma` itself is never placed on the common
        # grid (its own scale, ~-8 dex here, sits nowhere near the grid's
        # `log10 F_4.5` range) -- only `log10 Sigma + c_theta` is a
        # brightness, so `[L_Sigma (*) K_c](F)` is built the SAME way
        # `template_weights.build_h2shock`'s own `p(F_k) = Sum_theta
        # w_theta L_Sigma(F_k - c_theta)` is: each template's own EXACT
        # Gaussian kernel (`exact_gaussian_kernel(LOGSIG_STD)`, floored at
        # one cell, sec. 2's "minimum widths") placed at its own `log10
        # Sigma_mean + c_theta` nearest cell and summed with the (uniform,
        # sec. 5.6, disclosed) template weight -- the SAME formula, so this
        # marginal and P5's own conditional table agree by construction.
        names_h2s, c_theta_h2s = template_weights.h2shock_conversion(config)
        n_model_h2s = names_h2s.size
        w_uniform_h2s = np.full(n_model_h2s, 1.0 / n_model_h2s, dtype=np.float64)
        sigma_kernel = sample_cloud.exact_gaussian_kernel(logsig_std)
        half_width_sigma = (sigma_kernel.size - 1) // 2
        idx_center_h2s = np.round(
            (logsig_mean + c_theta_h2s - grid.LOG10_F45_EDGES[0]) / grid.D_LOG10_F45).astype(np.int64)
        f45_marginal_h2s = np.zeros(n_b, dtype=np.float64)
        for m_i in range(sigma_kernel.size):
            m = m_i - half_width_sigma
            cell_idx = idx_center_h2s + m
            valid = (cell_idx >= 0) & (cell_idx < n_b)
            if not np.any(valid):
                continue
            np.add.at(f45_marginal_h2s, cell_idx[valid], w_uniform_h2s[valid] * sigma_kernel[m_i])

        # K_c (report/identity only, below): the templates' own conversions
        # under the uniform weights, a point-mass histogram on the common
        # grid's 0.1 dex cells -- not used to build f45_marginal_h2s above
        # (the per-template placement already sums it in exactly), only to
        # convolve against the OLD reader's construction for rule 11's
        # identity.
        c_hist, _ = np.histogram(c_theta_h2s, bins=grid.LOG10_F45_EDGES, weights=w_uniform_h2s)

        grid_h2s = (x_marginal[:, :, None] * f45_marginal_h2s[None, None, :]).astype(np.float32)
        on_grid_frac_h2s = float(f45_marginal_h2s.sum())
        mass_outside_h2s = (np.clip(1.0 - x_marginal.astype(np.float64).sum(axis=1) * on_grid_frac_h2s,
                                     0.0, 1.0).astype(np.float32)
                             if n_sl else np.zeros((0,), dtype=np.float32))
        on_grid_h2s = (1.0 - mass_outside_h2s).astype(np.float32)

        # sec. 5.5's acceptance identity: the current (pre-overwrite)
        # X_MARGINAL restricted to the cloud interval and renormalised,
        # against the freshly built one -- both read above/built above,
        # compared per sightline.
        max_x_marginal_dev = float("nan")
        if old_x_marginal is not None and old_x_marginal.shape[0] == n_sl:
            ref = np.stack([
                sample_cloud.restrict_old_x_marginal(loaded, row, old_x_marginal[row], d_front, d_back)
                for row in range(n_sl)]) if n_sl else np.zeros((0, n_x))
            max_x_marginal_dev = float(np.max(np.abs(ref - x_marginal))) if n_sl else 0.0

        path = p3_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "sightline"
            f.attrs["FLOOR"] = grid.FLOOR
            f.attrs["N_SUB"] = sample_cloud.N_SUB
            f.attrs["LOGSIG_MEAN"] = logsig_mean
            f.attrs["LOGSIG_STD"] = logsig_std
            f.attrs["D_FRONT_PC"] = float(d_front)
            f.attrs["D_BACK_PC"] = float(d_back)
            f.create_dataset("LOG10_X_EDGES", data=grid.LOG10_X_EDGES)
            f.create_dataset("LOG10_F45_EDGES", data=grid.LOG10_F45_EDGES)
            f.create_dataset("HPX_PIX_256", data=loaded["hpx_pix_256"])
            f.create_dataset("GRID_YSO", data=grid_yso)
            f.create_dataset("X_MARGINAL", data=x_marginal)
            f.create_dataset("MASS_OUTSIDE_YSO", data=mass_outside_yso)
            f.create_dataset("ON_GRID_YSO", data=on_grid_yso)
            f.create_dataset("GRID_H2S", data=grid_h2s)
            f.create_dataset("MASS_OUTSIDE_H2S", data=mass_outside_h2s)
            f.create_dataset("ON_GRID_H2S", data=on_grid_h2s)

        # the region's own brightness marginal (sec. 9's report): P_ref
        # convolved with the region's own shift kernel K (sec. 5.5
        # ruling 1, `shift_kernel` -- the SAME K `template_weights
        # .build_yso` reads for its conditional table), peak and 16-84%.
        k_region, mo_k = sample_cloud.shift_kernel(config, region, d_front, d_back)
        conv_region = np.convolve(p_ref, k_region, mode="full")
        start = int(round(-grid.LOG10_F45_EDGES[0] / grid.D_LOG10_F45))
        p_region_marginal = np.maximum(conv_region[start:start + n_b], 0.0)
        b_centers = grid._B_CENTERS
        peak_f45 = float(b_centers[int(np.argmax(p_region_marginal))])
        cdf = np.cumsum(p_region_marginal) / np.sum(p_region_marginal)
        p16_f45, p84_f45 = (float(np.interp(q, cdf, b_centers)) for q in (0.16, 0.84))

        # the joint's own depth-brightness correlation (sec. 9's report,
        # the shift-kernel rule acceptance): 0 before (a strict outer product), negative
        # after (deeper is fainter) -- the Pearson correlation of `log10
        # x` and `log10 F_4.5` over one sightline's own on-grid joint mass.
        def _joint_corr(h2d):
            mass = h2d.astype(np.float64)
            total = mass.sum()
            if total <= 0:
                return float("nan")
            x_c, f_c = grid._X_CENTERS, grid._B_CENTERS
            px = mass.sum(axis=1) / total
            pf = mass.sum(axis=0) / total
            mx, mf = float(np.sum(px * x_c)), float(np.sum(pf * f_c))
            vx = float(np.sum(px * (x_c - mx) ** 2))
            vf = float(np.sum(pf * (f_c - mf) ** 2))
            cov = float(np.sum(mass / total * (x_c[:, None] - mx) * (f_c[None, :] - mf)))
            return cov / np.sqrt(vx * vf) if vx > 0 and vf > 0 else float("nan")

        row46_corr = _joint_corr(grid_yso[46]) if n_sl > 46 else float("nan")
        med_col = int(np.argsort(x_marginal.sum(axis=1))[n_sl // 2]) if n_sl else -1
        med_corr = _joint_corr(grid_yso[med_col]) if n_sl else float("nan")


        # the knots' own 16-84% range in log10 F_4.5 (sec. 9's report,
        # the common-axis rule's effect number): the region-wide K_c (*) L_Sigma marginal,
        # the same for every sightline (only p_x varies by row).
        h2s_p16_f45 = h2s_p84_f45 = float("nan")
        if float(f45_marginal_h2s.sum()) > 0:
            cdf_h2s = np.cumsum(f45_marginal_h2s) / np.sum(f45_marginal_h2s)
            h2s_p16_f45, h2s_p84_f45 = (float(np.interp(q, cdf_h2s, b_centers)) for q in (0.16, 0.84))

        st.done(path, n_sightline=int(n_sl),
                mass_outside_yso_max=float(mass_outside_yso.max()) if n_sl else 0.0,
                on_grid_yso_range=(float(on_grid_yso.min()), float(on_grid_yso.max())) if n_sl else (0.0, 0.0),
                removed_frac_median=float(np.median(removed_frac)) if n_sl else 0.0,
                removed_frac_max=float(removed_frac.max()) if n_sl else 0.0,
                x_marginal_max_dev=max_x_marginal_dev,
                f45_peak=peak_f45, f45_p16=p16_f45, f45_p84=p84_f45,
                sigma_d_dex=sigma_d, mass_outside_k=float(mo_k),
                row46_corr=row46_corr, median_col_corr=med_corr,
                d_front_pc=float(d_front), d_back_pc=float(d_back),
                on_grid_h2s_range=(float(on_grid_h2s.min()), float(on_grid_h2s.max())) if n_sl else (0.0, 0.0),
                h2s_f45_p16=h2s_p16_f45, h2s_f45_p84=h2s_p84_f45)
    return (path, n_sl, mass_outside_yso, removed_frac, max_x_marginal_dev,
            (peak_f45, p16_f45, p84_f45), on_grid_yso, (d_front, d_back))


# ---------------------------------------------------------------------------
# P4 -- the galaxy grid, survey-wide (sec. 5.4)
# ---------------------------------------------------------------------------

def _rebin_conservative(old_edges, values, new_edges):
    """Conservative regridding of a piecewise-constant histogram from
    `old_edges` (its own bin edges, uniform-density within each bin -- what
    a stored histogram value means) to `new_edges`: exact, since a
    piecewise-constant function's cumulative sum is piecewise LINEAR, so
    linear interpolation of that cumulative sum at the new edges recovers
    it exactly there, whatever the new grid's own phase (`np.interp` on
    `old_edges`'s own cumulative sum, then differenced) -- the same
    technique this module used to check `X_MARGINAL` against a native
    profile before the 4.5B redesign."""
    old_cum = np.concatenate([[0.0], np.cumsum(values)])
    new_cum = np.interp(new_edges, old_edges, old_cum, left=0.0, right=old_cum[-1])
    return np.diff(new_cum)


def build_gal(config):
    """Writes P4, `bmstp/shape/gal_shape_survey.hdf5`: the one
    survey-wide GAL grid on the common `LOG10_F45_EDGES` (sec. 5.4
    "Grain"): a delta at `log10 x = 0` (the whole column) times the
    counts law directly in `F_4.5 = S` -- the law's own tabulated range
    (-2.2 to +1.3 dex) sits well inside the common grid, so no per-shape
    margin is needed any more -- plus the attr `ON_GRID_GAL` (`1 -
    mass_outside`, sec. 2's ON-GRID FRACTION, W26's own read)."""
    path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
    old = None
    if os.path.exists(path):
        # the current product is already on the common `LOG10_F45_EDGES`
        # schema -- there is no pre-4.5B `LOG10_B_EDGES` product to
        # migrate -- so the conservative regrid below only ever runs
        # across the grid's own top-edge move, never a per-shape-origin
        # phase shift.
        with h5py.File(path, "r") as f:
            old = dict(grid=f["GRID"][()], log10_f45_edges=f["LOG10_F45_EDGES"][()])

    with progress.Stage("bmstp.shapes.gal") as st:
        x, log10_b, w = sample_gal.sample(config)
        h, mass_outside = grid.bin(x, log10_b, w)
        sum_check = abs(h.sum() - (1.0 - mass_outside))
        above_top_gal = grid.mass_above_top(log10_b, w)
        on_grid_gal = 1.0 - mass_outside
        # `A_GAL`, sec. 5.4 "Sky density": `sample_gal.density` (the `ln
        # 10` integral), the same function P1 (`bmstp.density`) and P6
        # (`bmstp.atlas`) read -- NOT the shape weight `w.sum()`, short by
        # `ln 10`.
        density_gal = sample_gal.density(config)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            f.attrs["FLOOR"] = grid.FLOOR
            f.attrs["DENSITY_GAL"] = density_gal
            f.attrs["COSMIC_VARIANCE_DEX"] = _gal_cosmic_variance_dex(config)
            f.attrs["ON_GRID_GAL"] = float(on_grid_gal)
            f.create_dataset("LOG10_X_EDGES", data=grid.LOG10_X_EDGES)
            f.create_dataset("LOG10_F45_EDGES", data=grid.LOG10_F45_EDGES)
            f.create_dataset("GRID", data=h.astype(np.float32))

        # sec. 9's identity: the current (pre-overwrite) P4's own `log10 S`
        # marginal, conservatively rebinned onto the common axis, against
        # the freshly built one.
        max_dev = float("nan")
        if old is not None:
            old_b_marginal = old["grid"].sum(axis=0)
            old_b_edges = old["log10_f45_edges"]
            ref = _rebin_conservative(old_b_edges, old_b_marginal, grid.LOG10_F45_EDGES)
            new_b_marginal = h.sum(axis=0)
            max_dev = float(np.max(np.abs(ref - new_b_marginal)))

        st.done(path, mass_outside=float(mass_outside), sum_check=float(sum_check),
                density_gal_deg2=density_gal, f45_marginal_max_dev=max_dev,
                mass_above_top_gal=float(above_top_gal), on_grid_gal=float(on_grid_gal))
    return path, mass_outside, sum_check, max_dev, above_top_gal, on_grid_gal


def _gal_cosmic_variance_dex(config):
    path = config_module.product_path(config, "population", "gal", "counts", "survey")
    with h5py.File(path, "r") as f:
        return float(f["COSMIC_VARIANCE_DEX"][()])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

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
