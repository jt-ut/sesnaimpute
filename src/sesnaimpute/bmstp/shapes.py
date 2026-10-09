"""The shape-grid builder: bins each class's population sample onto the
common `(log10 ξ, log10 F_4.5)` grid (`bmstp.grid`) and writes the
shape-grid products of IMPLEMENTATION_BMSTP_DRAFT.md sec. 1.2
(SPEC_BMSTP_DRAFT.md sec. 2, sec. 4.1's "shape grids", sec. 5.1-5.6).

Writes P2 (the star-family grids, per tile), P3 (the cloud-class grid,
per sightline: YSO's `GRID_YSO`/`XI_MARGINAL`, and H2S's own
`GRID_H2S` on the SAME common grid -- `GRID_H2S[ξ, F] = p_ξ(ξ) . [L_Sigma
(*) K_c](F)`, `p_ξ` the sightline's own `XI_MARGINAL`, `L_Sigma` the
region's knot lognormal (`LOGSIG_MEAN`/`LOGSIG_STD`, kept as attributes),
`K_c` the distribution of the h2shock templates' own Sigma-to-4.5-micron
conversions under their (uniform) weights -- one convolution, survey-wide
in its template content, computed once per region and broadcast over
sightlines through `p_ξ` alone, sec. 5.6 "Marks"), and P4 (the galaxy
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

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.bmstp import grid, sample_cloud, sample_gal, sample_star, template_weights
from sesnaimpute.population import h2s as h2s_module
from sesnaimpute.sky.derived import profile as profile_module

_STAR_STEM = "star_shape_tile"
_CLOUD_STEM = "cloud_shape_sightline"
_GAL_STEM = "gal_shape_survey"


# ---------------------------------------------------------------------------
# P2 -- the star-family grids, per tile (sec. 5.1, 5.2)
# ---------------------------------------------------------------------------

def _star_depth_mark(x, dist_pc, d_front, d_back, u_front):
    """The STAR depth mark from the cloud interval (owner's ruling, C5,
    `bms_review/REVIEW_LEDGER_2026-10-08.md`): a star beyond the
    interval's back edge (`dist_pc > d_back`) sits at the full column
    exactly (`x = 1`), never at the profile's own `u(d)` read there --
    the 3-D map's radial smearing (Edenhofer 2023's own stated radial
    resolution; Leike 2020 measures the same bias) otherwise spreads
    background stars forward into partial extinction they do not
    physically carry (prior_1 concern 2). A star in front of the front
    edge (`dist_pc < d_front`) takes the foreground share, `u_front`
    (`sample_star.tile_width_classes`' own profile read at `d_front` on
    the tile's representative sightline) -- the SAME value for every
    foreground star, since the map cannot resolve a finer distance
    there. Only a star inside the interval keeps its own partial value
    straight off the profile: `x` unchanged."""
    x = np.array(x, dtype=np.float64, copy=True)
    dist_pc = np.asarray(dist_pc, dtype=np.float64)
    x[dist_pc > d_back] = 1.0
    x[dist_pc < d_front] = u_front
    return x


def _build_one_tile(config, region, tile_id):
    """One tile's `(GRID_STAR, MASS_OUTSIDE_STAR, GRID_AGB,
    MASS_OUTSIDE_AGB, sum_check, above_star_w, total_star_w, above_agb_w,
    total_agb_w)`, `sum_check` the max, over the two grids, of the
    pre-floor identity `|H.sum() - (1 - mass_outside)|` (`mass_outside`
    from `grid.bin_star_widths`'s own INDEPENDENT bookkeeping, not read
    back off `H.sum()`, so this is a genuine check, SPEC_BMSTP_DRAFT.md
    sec. 9's normalisation check for this class). `above_*_w`/`total_*_w`
    are each class's own RAW (unnormalised)
    weight above the grid's top edge and its own total weight, summed
    across tiles by the caller into the region's single mass-above-top
    fraction (sec. 2, sec. 9's 0.1% bar, W24b -- distinct from
    `MASS_OUTSIDE_*`, which also counts the mass the retention limit
    drops at the bottom edge)."""
    x_s, f45_s, w_s = sample_star.sample_star(config, region, tile_id)
    x_a, f45_a, w_a = sample_star.sample_agb(config, region, tile_id)

    # (sec. 2 "minimum widths", sec. 5.1 "Marks"): the field-star
    # depth mark's own width is the map's propagated column sigma at the
    # star's distance, not a fixed one cell -- AGB follows STAR (the same
    # stars, sec. 5.2), so both read the SAME tile width classes. A star
    # the interval pins (below) takes the floor class instead
    # (`star_width_class`'s own docstring, the C5 consequence).
    row, sigma_classes_dex, d_front, d_back, u_front = sample_star.tile_width_classes(
        config, region, tile_id)
    sigma_classes_cells = sigma_classes_dex / grid._X_CELL_WIDTH
    dist_s = sample_star.star_distances(config, region, tile_id)
    dist_a = sample_star.agb_star_distances(config, region, tile_id)
    class_s = sample_star.star_width_class(config, region, dist_s, row, sigma_classes_dex, d_front, d_back)
    class_a = sample_star.star_width_class(config, region, dist_a, row, sigma_classes_dex, d_front, d_back)

    # the STAR depth mark from the cloud interval (owner's ruling, C5):
    # AGB follows STAR, the SAME rule on the SAME stars' distances (sec.
    # 5.2) -- see `_star_depth_mark`'s own docstring.
    x_s = _star_depth_mark(x_s, dist_s, d_front, d_back, u_front)
    x_a = _star_depth_mark(x_a, dist_a, d_front, d_back, u_front)

    h_star, mo_star = grid.bin_star_widths(x_s, f45_s, w_s, class_s, sigma_classes_cells)
    h_agb, mo_agb = grid.bin_star_widths(x_a, f45_a, w_a, class_a, sigma_classes_cells)
    sum_check = max(abs(h_star.sum() - (1.0 - mo_star)), abs(h_agb.sum() - (1.0 - mo_agb)))
    above_star_w = float(w_s[f45_s > grid.LOG10_F45_EDGES[-1]].sum())
    above_agb_w = float(w_a[f45_a > grid.LOG10_F45_EDGES[-1]].sum())
    return (h_star.astype(np.float32), mo_star, h_agb.astype(np.float32), mo_agb, sum_check,
            above_star_w, float(w_s.sum()), above_agb_w, float(w_a.sum()))


def build_star_family(config, region):
    """Writes P2, `bmstp/shape/star_shape_tile__R.hdf5`: `GRID_STAR` and
    `GRID_AGB` per tile (sec. 5.1, 5.2) on the common `LOG10_F45_EDGES`,
    `MASS_OUTSIDE_STAR`/`_AGB`, `ON_GRID_STAR`/`_AGB` (`1 - mass_outside`
    per tile, sec. 2's ON-GRID FRACTION for the density stage to scale by,
    W26), the grid edges, the star-count-per-tile attribute, and AGB's own
    per-chemistry flux-to-luminosity ratio spread (`F45_PER_L_SPREAD_DEX_O/C`,
    sec. 5.2). PAHC has no grid of its own: a reader loads `GRID_STAR` for
    it (sec. 5.3 "Grain"). `GRID_STAR`/`GRID_AGB` carry no `N_EFF` dataset
    and no `FLOOR` attribute, like every shape product (`bmstp.grid`'s
    module docstring)."""
    path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)

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
        agb_ratio = sample_star.agb_log10_ratio_stats(config.inputs["sed_models"])

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "tile"
            f.attrs["N_STARS_PER_TILE"] = n_stars_per_tile
            f.attrs["F45_PER_L_SPREAD_DEX_O"] = agb_ratio["O"]["spread_dex"]
            f.attrs["F45_PER_L_SPREAD_DEX_C"] = agb_ratio["C"]["spread_dex"]
            for name, data in (
                ("LOG10_XI_EDGES", grid.LOG10_XI_EDGES),
                ("LOG10_F45_EDGES", grid.LOG10_F45_EDGES),
                ("TILE_ID", ids.astype(np.int32)),
                ("GRID_STAR", grid_star),
                ("GRID_AGB", grid_agb),
                ("MASS_OUTSIDE_STAR", mass_outside_star),
                ("MASS_OUTSIDE_AGB", mass_outside_agb),
                ("ON_GRID_STAR", on_grid_star.astype(np.float32)),
                ("ON_GRID_AGB", on_grid_agb.astype(np.float32)),
            ):
                build_module.write_dataset(f, name, data, *REGISTRY[(_STAR_STEM, name)])

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
                f45_range_field_stars=(f45_lo, f45_hi),
                mass_above_top_star=float(above_top_star),
                mass_above_top_agb=float(above_top_agb),
                on_grid_star_range=(float(on_grid_star.min()), float(on_grid_star.max())) if n_tile else (0.0, 0.0),
                on_grid_agb_range=(float(on_grid_agb.min()), float(on_grid_agb.max())) if n_tile else (0.0, 0.0))
    return (path, n_tile, mass_outside_star, mass_outside_agb, max_sum_check,
            above_top_star, above_top_agb, on_grid_star, on_grid_agb)


def _n_stars_per_tile(config, region, ids):
    path = config_module.product_path(
        config, "population", "star", "population", "tile", region=region)
    with h5py.File(path, "r") as f:
        return np.array([f[f"tile_{int(t)}"]["XI"].shape[0] for t in ids], dtype=np.int32)


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

def _region_peak_pc(config, region):
    """This region's own single `D_PEAK_PC`
    (`sky/derived/edenhofer/depth_edenhofer_region`), the SAME aggregate
    dust-structure peak `sample_cloud.cloud_interval_pc`'s cloud interval
    and `fittp.gaia`'s cloud-anchored Normal are already built from (sec.
    2 "region distance and depth") -- the baseline
    `_sightline_dust_peaks_pc` offsets each sightline's own peak against,
    read directly off the same product rather than threaded through
    another stage's return value."""
    depth_path = config_module.product_path(config, "sky/derived", "edenhofer", "depth", "region")
    with h5py.File(depth_path, "r") as f:
        names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in f["REGION"][:]]
        return float(f["D_PEAK_PC"][names.index(region)])


def _sightline_dust_peaks_pc(config, region, hpx_pix_256, d_r_pc, d_front, d_back, d_peak_region_pc):
    """Each of this region's own sightlines' own 3-D-map dust-structure
    peak distance (sec. 5.5 "Marks", the shift-kernel rule's own
    distance): `sky.derived.profile.structure_depth` -- the SAME
    peak-finder that stage already runs once, on the region's own
    sightline-weighted mean density, for the D_PEAK_PC the depth mark
    (`sample_cloud.cloud_interval_pc`) and the Gaia term
    (`fittp.gaia._cloud_normal_params_mas`) both already read -- run here
    PER SIGHTLINE on that sightline's own `RHO_K_PER_PC` row, target the
    region's own literature distance (the same target `measure_region`
    uses for the aggregate peak), so a footprint whose line of sight
    crosses more than one structure (Aquila's Serpens-Main and
    Aquila-Rift components) returns a different peak for the sightlines
    on each side rather than the one region-wide blend. The search is
    restricted to `[d_front, d_back]`, the SAME cloud interval every
    other sub-sample of this sightline is already restricted to
    (`sample_cloud.cloud_interval_pc`), so a peak outside the region's
    own cloud (foreground cirrus, an unrelated background clump) is
    never returned. Falls back to the region's own `d_peak_region_pc` on
    a sightline whose own density carries no separable local maximum
    inside that window (`structure_depth` returns a NaN peak only then;
    a peak found but failing its own fractional-depth bar is still a
    real distance and is kept). Returns `d_peak_pc`, `(n_sl,)`, aligned
    to `hpx_pix_256`."""
    path = config_module.product_path(
        config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    with h5py.File(path, "r") as f:
        hpx_profile = f["HPX_PIX_256"][:]
        dist_pc = f["DIST_PC"][:]
        rho = f["RHO_K_PER_PC"][:]
    if hpx_profile.shape != hpx_pix_256.shape or not np.array_equal(hpx_profile, hpx_pix_256):
        raise ValueError(
            "bmstp.shapes._sightline_dust_peaks_pc: HPX_PIX_256 order mismatch between "
            "the profile product and the region's own embedding, region %r" % region)
    radii = 0.5 * (dist_pc[:-1] + dist_pc[1:])
    in_window = (radii >= d_front) & (radii <= d_back)
    radii_w = radii[in_window]
    n_sl = rho.shape[0]
    d_peak_pc = np.empty(n_sl, dtype=np.float64)
    for row in range(n_sl):
        found = (profile_module.structure_depth(radii_w, rho[row][in_window], d_r_pc)
                 if radii_w.size >= 3 else {"d_peak_pc": np.nan})
        peak = found["d_peak_pc"]
        d_peak_pc[row] = peak if np.isfinite(peak) else d_peak_region_pc
    return d_peak_pc


def _yso_kernel_placement(x_idx, delta, weights, on_x, n_x, kernel_1d, p_ref, n_b):
    """One sub-sample weight column shifted-and-smoothed through the
    kernel-placement pipeline (sec. 5.5, the shift-kernel rule): the raw
    weighted `(x, delta)` histogram, the ONE EXACT Gaussian smoothing
    (`kernel_1d`) and the ONE convolution with `p_ref`, BEFORE the
    common grid's own one-cell `x`-axis smoothing (sec. 2 "minimum
    widths") -- what `GRID_YSO` itself is built from."""
    k_row, _, _ = np.histogram2d(
        x_idx[on_x], delta[on_x], bins=[np.arange(n_x + 1), grid.LOG10_F45_EDGES],
        weights=weights[on_x])
    k_row = ndimage.convolve1d(k_row, weights=kernel_1d, axis=1, mode="constant")
    conv = fftconvolve(k_row, p_ref[None, :], mode="full", axes=1)
    start = int(round(-grid.LOG10_F45_EDGES[0] / grid.D_LOG10_F45))
    return np.maximum(conv[:, start:start + n_b], 0.0)


def _build_one_sightline(loaded, row, p_ref, kernel_1d, d_front, d_back, peak_offset_dex=0.0):
    """One sightline's `(GRID_YSO, XI_MARGINAL, MASS_OUTSIDE_YSO,
    removed_frac)` (sec. 5.5 "Marks"): every depth sub-sample
    (`sample_cloud._cell_subsamples`, weight `w_k,sub`, depth `x_k,sub`,
    distance `d_k,sub`) adds its own shifted-and-smoothed copy of `p_ref`
    to its own `log10 ξ` row -- grouped by row first (linear in the
    sub-samples, so summing the row's own raw weighted `delta = -2
    log10(d_k,sub / 1 kpc) + peak_offset_dex` histogram before the ONE
    EXACT Gaussian smoothing (`kernel_1d`, `sample_cloud.
    exact_gaussian_kernel`) and the ONE convolution with `p_ref`
    reproduces the per-sub-sample sum exactly). `peak_offset_dex`
    (`_sightline_dust_peaks_pc`, this sightline's own 3-D-map
    dust-structure peak against the region's single D_PEAK_PC) carries
    the sightline's own distance into the shift alongside `d_sub`'s
    within-structure depth variation, rather than letting every
    sightline of a multi-structure footprint share the one region-wide
    reference the depth mark and the Gaia term already peak-anchor
    (sec. 2 "region distance and depth"); it is zero for a single-
    structure region, where this reduces to the prior behaviour exactly.
    `XI_MARGINAL` (`p_x`) is unchanged in value (`sample_cloud.sample_x`,
    its own one-cell-smoothed bin) and is built from the SAME `d_front`/
    `d_back` cloud interval, untouched by `peak_offset_dex`, so the depth
    mark this sweep's other packages own is not altered here.
    A stored shape is normalised to one over the CLASS'S OWN population
    (sec. 4.1's "shape grids": "normalised to one over cells", true of
    every class alike): the sightline's cloud-interval-restricted
    sub-sample weight, `total_weight = sum(w_k,sub)` (`sample_x`'s own
    denominator, `sample_cloud._bin1d`, so `GRID_YSO`'s `x`-marginal
    agrees with `XI_MARGINAL` by construction), never `sum(w_k,sub) *
    p_ref.sum()` (a quantity that shrinks with `GRID_YSO`'s own
    shortfall and so cannot measure it). `GRID_YSO` is divided by
    `total_weight` here, the same division `_bin1d` already applies to
    build `p_x`. `MASS_OUTSIDE_YSO` is the shortfall against that
    invariant (1, after the division): THE EDGE ξ = 1 (`bmstp.grid`'s
    module docstring) reflects whatever of the one-cell `x` smoothing's
    own mass reaches `log10 ξ > 0` back onto the support (`grid.fold_wall`)
    rather than dropping it, exactly as `grid.bin` does for every other
    class, so only mass a sub-sample's own smoothing carries past the
    array's true `log10 ξ` edges, or `p_ref`'s own off-grid template
    share (`sample_cloud.p_ref_f45`'s docstring), is ever shortfall.
    `GRID_YSO` carries its own true zeros: no per-shape floor is baked in
    here."""
    p_x, _mo_x, removed_frac = sample_cloud.sample_x(loaded, row, d_front, d_back)
    log10_xi_nudged, w_sub, d_sub, _removed = sample_cloud._cell_subsamples(loaded, row, d_front, d_back)
    n_x = grid.LOG10_XI_EDGES.size - 1
    n_b = grid.LOG10_F45_EDGES.size - 1

    x_idx = np.digitize(log10_xi_nudged, grid.LOG10_XI_EDGES) - 1
    on_x = (x_idx >= 0) & (x_idx < n_x)
    delta = -2.0 * np.log10(d_sub / 1000.0) + peak_offset_dex
    # the sub-samples' own raw weights through the shift-kernel placement
    # -- identically what `GRID_YSO` is, before its own one-cell
    # `x`-smoothing (sec. 2 "minimum widths").
    grid_yso = _yso_kernel_placement(x_idx, delta, w_sub, on_x, n_x, kernel_1d, p_ref, n_b)

    # the one-cell smoothing along x (sec. 2 "minimum widths"): `sample_x`'s
    # own p_x already carries it; GRID_YSO gets its own pass here since its
    # x-axis rows were built from the raw (unsmoothed) sub-sample scatter
    # above, not from p_x.
    grid_yso = gaussian_filter1d(grid_yso, sigma=1.0, axis=0, mode="constant",
                                  truncate=grid._truncate_for(1.0, n_x))

    # the invariant GRID_YSO is normalised against: the sightline's own
    # cloud-interval-restricted sub-sample weight, the SAME denominator
    # `sample_cloud._bin1d` divides by to build `p_x` (sec. 4.1's "shape
    # grids", "normalised to one over cells") -- never `total_weight *
    # p_ref.sum()`, which is derived from `GRID_YSO`'s own incomplete
    # construction and shrinks in lockstep with it, so it could not
    # measure a shortfall (the defect this replaces).
    total_weight = float(w_sub.sum())
    # THE EDGE ξ = 1: `log10 ξ = 0` reflects -- the shift-kernel convolution's
    # own padding dex is folded back onto the support (`grid.fold_wall`)
    # rather than dropped, so `mass_outside_yso` below counts only what
    # the smoothing carries past the array's true `log10 ξ` edges, or what
    # `p_ref` itself never placed on the grid.
    grid_yso = grid.fold_wall(grid_yso)
    total_actual = float(grid_yso.sum())
    if total_weight <= 0.0:
        mass_outside_yso = 1.0
    else:
        mass_outside_yso = float(np.clip(1.0 - total_actual / total_weight, 0.0, 1.0))
        grid_yso = grid_yso / total_weight
    return (grid_yso.astype(np.float32), p_x.astype(np.float32), mass_outside_yso, removed_frac)


def build_cloud(config, region):
    """Writes P3, `bmstp/shape/cloud_shape_sightline__R.hdf5`: `GRID_YSO`
    (sec. 5.5, the shift-kernel rule -- each depth sub-sample's own shifted copy of
    `P_ref`, `bmstp.sample_cloud.p_ref_f45`, summed by `log10 ξ` row, NOT
    an outer product) and its `log10 ξ` marginal per sightline,
    `MASS_OUTSIDE_YSO`, `ON_GRID_YSO` (`1 - mass_outside_yso` per
    sightline, sec. 2's ON-GRID FRACTION, W26's own read), the cloud
    interval `D_FRONT_PC`/`D_BACK_PC` (DOUBLED about the region's own peak
    distance, W24b), `D_PEAK_SIGHTLINE_PC` (each sightline's own 3-D-map
    dust-structure peak, `_sightline_dust_peaks_pc`, that anchors
    `GRID_YSO`'s brightness shift alongside the within-structure depth
    `d_sub` already carries, sec. 2 "region distance and depth"), and the
    region's H2S brightness Gaussian (`LOGSIG_MEAN`,
    `LOGSIG_STD`, sec. 5.6) computed here by transporting the UWISH2 knot
    survey's surface-brightness sample to the region's own distance
    (`population.h2s.transport_log10_sigma`,
    `population.h2s.region_sigma_lognormal`) -- H2S has no grid of its own
    (sec. 5.6 "Marks": separable, `XI_MARGINAL` times this Gaussian, formed
    at read on the common `LOG10_F45_EDGES` through the template's own
    `C_THETA` offset, never a class-specific origin). Neither grid carries
    an `N_EFF` dataset or a `FLOOR` attribute, per `bmstp.grid`'s module docstring."""
    p3_path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)

    with progress.Stage("bmstp.shapes.cloud", region) as st:
        loaded = sample_cloud._region_profile(config, region)
        n_sl = loaded["hpx_pix_256"].size
        d_front, d_back = sample_cloud.cloud_interval_pc(config, region)
        r = regions_module.REGIONS_BY_NAME[region]

        # the YSO brightness-shift distance per sightline (sec. 2 "region
        # distance and depth"): each sightline's own 3-D-map dust peak,
        # read against the region's single D_PEAK_PC the depth mark and
        # the Gaia term already use, so a footprint crossing more than
        # one structure (Aquila's Serpens-Main and Aquila-Rift
        # components) shifts its own sightlines by their own distance
        # rather than the one region-wide blend.
        d_peak_region_pc = _region_peak_pc(config, region)
        d_peak_sightline_pc = (
            _sightline_dust_peaks_pc(config, region, loaded["hpx_pix_256"], r.d_r_pc,
                                      d_front, d_back, d_peak_region_pc)
            if n_sl else np.zeros((0,), dtype=np.float64))
        peak_offset_dex = (-2.0 * np.log10(d_peak_sightline_pc / d_peak_region_pc)
                            if n_sl else np.zeros((0,), dtype=np.float64))

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
            delayed(_build_one_sightline)(loaded, row, p_ref, kernel_1d, d_front, d_back,
                                           peak_offset_dex[row])
            for row in range(n_sl))
        for i in range(n_sl):
            st.tick(i + 1, n_sl, "sightlines")

        n_b = grid.LOG10_F45_EDGES.size - 1
        n_x = grid.LOG10_XI_EDGES.size - 1
        grid_yso = (np.stack([r_[0] for r_ in results]) if n_sl
                    else np.zeros((0, n_x, n_b), dtype=np.float32))
        xi_marginal = (np.stack([r_[1] for r_ in results]) if n_sl
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

        # GRID_H2S (sec. 5.6 "Marks"): GRID_H2S[ξ, F] = p_ξ(ξ)
        # . [L_Sigma (*) K_c](F), one marginal per region (the template
        # content and the region's own lognormal are both region-wide, not
        # per-sightline), broadcast over sightlines through XI_MARGINAL
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

        grid_h2s = (xi_marginal[:, :, None] * f45_marginal_h2s[None, None, :]).astype(np.float32)
        on_grid_frac_h2s = float(f45_marginal_h2s.sum())
        mass_outside_h2s = (np.clip(1.0 - xi_marginal.astype(np.float64).sum(axis=1) * on_grid_frac_h2s,
                                     0.0, 1.0).astype(np.float32)
                             if n_sl else np.zeros((0,), dtype=np.float32))
        on_grid_h2s = (1.0 - mass_outside_h2s).astype(np.float32)

        path = p3_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "sightline"
            f.attrs["N_SUB"] = sample_cloud.N_SUB
            f.attrs["LOGSIG_MEAN"] = logsig_mean
            f.attrs["LOGSIG_STD"] = logsig_std
            f.attrs["D_FRONT_PC"] = float(d_front)
            f.attrs["D_BACK_PC"] = float(d_back)
            for name, data in (
                ("LOG10_XI_EDGES", grid.LOG10_XI_EDGES),
                ("LOG10_F45_EDGES", grid.LOG10_F45_EDGES),
                ("HPX_PIX_256", loaded["hpx_pix_256"]),
                ("D_PEAK_SIGHTLINE_PC", d_peak_sightline_pc),
                ("GRID_YSO", grid_yso),
                ("XI_MARGINAL", xi_marginal),
                ("MASS_OUTSIDE_YSO", mass_outside_yso),
                ("ON_GRID_YSO", on_grid_yso),
                ("GRID_H2S", grid_h2s),
                ("MASS_OUTSIDE_H2S", mass_outside_h2s),
                ("ON_GRID_H2S", on_grid_h2s),
            ):
                build_module.write_dataset(f, name, data, *REGISTRY[(_CLOUD_STEM, name)])

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
        med_col = int(np.argsort(xi_marginal.sum(axis=1))[n_sl // 2]) if n_sl else -1
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
                f45_peak=peak_f45, f45_p16=p16_f45, f45_p84=p84_f45,
                sigma_d_dex=sigma_d, mass_outside_k=float(mo_k),
                row46_corr=row46_corr, median_col_corr=med_corr,
                d_front_pc=float(d_front), d_back_pc=float(d_back),
                on_grid_h2s_range=(float(on_grid_h2s.min()), float(on_grid_h2s.max())) if n_sl else (0.0, 0.0),
                h2s_f45_p16=h2s_p16_f45, h2s_f45_p84=h2s_p84_f45,
                d_peak_region_pc=float(d_peak_region_pc),
                d_peak_sightline_range=(float(d_peak_sightline_pc.min()), float(d_peak_sightline_pc.max()))
                if n_sl else (0.0, 0.0),
                peak_offset_dex_range=(float(peak_offset_dex.min()), float(peak_offset_dex.max()))
                if n_sl else (0.0, 0.0))
    return (path, n_sl, mass_outside_yso, removed_frac,
            (peak_f45, p16_f45, p84_f45), on_grid_yso, (d_front, d_back))


# ---------------------------------------------------------------------------
# P4 -- the galaxy grid, survey-wide (sec. 5.4)
# ---------------------------------------------------------------------------

def build_gal(config):
    """Writes P4, `bmstp/shape/gal_shape_survey.hdf5`: the one
    survey-wide GAL grid on the common `LOG10_F45_EDGES` (sec. 5.4
    "Grain"): a delta at `log10 ξ = 0` (the whole column) times the
    counts law directly in `F_4.5 = S` -- the law's own tabulated range
    (-2.2 to +1.3 dex) sits well inside the common grid, so no per-shape
    margin is needed -- plus the attr `ON_GRID_GAL` (`1 -
    mass_outside`, sec. 2's ON-GRID FRACTION, W26's own read). No `N_EFF`
    dataset and no `FLOOR` attribute, like every shape product
    (`bmstp.grid`'s module docstring)."""
    path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")

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
            f.attrs["DENSITY_GAL"] = density_gal
            f.attrs["COSMIC_VARIANCE_DEX"] = _gal_cosmic_variance_dex(config)
            f.attrs["ON_GRID_GAL"] = float(on_grid_gal)
            for name, data in (
                ("LOG10_XI_EDGES", grid.LOG10_XI_EDGES),
                ("LOG10_F45_EDGES", grid.LOG10_F45_EDGES),
                ("GRID", h.astype(np.float32)),
            ):
                build_module.write_dataset(f, name, data, *REGISTRY[(_GAL_STEM, name)])

        st.done(path, mass_outside=float(mass_outside), sum_check=float(sum_check),
                density_gal_deg2=density_gal,
                mass_above_top_gal=float(above_top_gal), on_grid_gal=float(on_grid_gal))
    return path, mass_outside, sum_check, above_top_gal, on_grid_gal


def _gal_cosmic_variance_dex(config):
    path = config_module.product_path(config, "population", "gal", "counts", "survey")
    with h5py.File(path, "r") as f:
        return float(f["COSMIC_VARIANCE_DEX"][()])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Per region, P2 (the star family) and P3 (the cloud class); once per
    call, P4 (GAL, survey-wide, IMPLEMENTATION_BMSTP_DRAFT.md sec. 1.2
    P4) -- rebuilt every time this driver runs, regardless of `regions`,
    since it is not a per-region product."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    for region in region_names:
        build_star_family(config, region)
        build_cloud(config, region)

    build_gal(config)


if __name__ == "__main__":
    run(build)
