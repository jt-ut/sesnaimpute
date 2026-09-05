"""The exact-selection tables for STAR, AGB and PAHC (SPEC_PRIORS.md
section 1.3's "exact selection tables per depth group"; section 2.1's
`eps_s(a_i, B_i)` on TRILEGAL's own fluxes; section 3 "Selection" on GRAMS
colours with the photospheric bound; section 4's PAHC weight `w * P(q)`).

`prior.selection.PassFractionModel.fit` needs, for one class, an exact
`population_eps(delta_5) -> (n_a, n_b)` at each depth group's own
five-band residual: the class's own external population (SPEC_PRIORS.md
0.3, C3), dimmed and scaled exactly, clearing >= 2 of 8 bands at the
group's own shifted limit vector, at every column-grid node `a` and every
point of a class-specific `log10 B` grid.

The population's own placement, per source, is the tile it sits in
(`prior.star_population`'s per-tile mean profile). A depth group's own
table is a region-wide object, evaluated a few hundred times per region
per class -- too expensive to repeat once per tile -- so this module
places the WHOLE region's stars on ONE fallback profile: the tile-weighted
mean of the population product's own per-tile `U` (reading note
04_star_family.md section C: "port uses the region's fallback placement
for the tables while shapes are per tile. State the approximation" --
this is that approximation, and each tile's own total anchor weight
(`Sigma W` over the tile) stands in for its source count, since no
second read of the tile geometry is needed to already have that sum on
hand).

The two-of-eight test (`prior.selection.epsilon`) is evaluated exactly,
without ever tabulating a `(star, node, log10 B)` cube: a star's own
pass/fail at a query node `A` and a query `log10 B` depends on `log10 B`
only through a single additive shift (rescaling a star's flux by
`10**(log10 B - log10 B_i)` shifts every band's dimmed log-flux by the
same amount), so each star's own two-of-eight threshold reduces to one
critical brightness per (star, node): the second-smallest, over the eight
bands, of `log10 F_lim,band - L(star, node, band)`, where `L` folds in
the star's own flux, its own dimming at `a_i = A * u_i`, and its own
`B_i` (the module's `L_i(node, band)`, `_build_population_eps`'s
docstring works the algebra). The class's exact `eps(node, log10 B)` is
then the weighted fraction of the population whose critical brightness is
at or below the query `log10 B` -- an exact weighted CDF, one matrix
product per depth-group query, vectorised over every star, node and band
at once.

AGB draws its SED from GRAMS (SPEC_PRIORS.md section 3, "Selection"): each
evolved star is matched to its nearest curated GRAMS model by chemistry
(O-rich to the nearest model luminosity, since the shipped O-rich library
shares one luminosity to within 0.1% -- SPEC_PRIORS.md section 3's `L_O`
-- so the match is near-degenerate and only breaks the rare tie; C-rich to
the nearest model in log10 luminosity, TRILEGAL's own `log_L` being the
only per-star axis available to match a chemistry-blind population
against a library that, unlike the O-rich branch, spans a real luminosity
grid), then that model's own 100-kpc convolved photometry is rescaled to
the star's own brightness by the same `B`-unit-change rule the shape uses
(`b_agb`): `flux_at_star = flux_100kpc * (100 kpc / 1 kpc)**2 *
10**(log10 B_i - log10 B_model_at_1kpc)`, `log10 B_model_at_1kpc` being
zero for C-rich (`b_agb`'s own closed form carries no luminosity term)
and `log10(L_model / L_O)` for O-rich. A second AGB table, `EPS_
PHOTOSPHERE`, is built the same way but with the star's own TRILEGAL
flux standing in for the library flux -- the lower bound SPEC_PRIORS.md
section 3 calls for, sharing the dusty table's own depth groups so the
two sit "beside" each other in one product.

PAHC's population weight is `W * P_PAHC[:, j]` at the region's median 8
micron completeness limit (`j`, the index of the 50% quantile in `star_
population.PAHC_LIMIT_QUANTILES`), on the star's own TRILEGAL fluxes and
`LOG10_B_PAHC` (SPEC_PRIORS.md section 4).

`K` is grown from the region's already-built depth-groups product
(`prior.depth_groups`) by doubling until `prior.selection.fit_by_
residual`'s 1% worst-node bar is met or `K_SEARCH_ABS_CAP`; never
refused, always reported.

Writes one product per region, `bms/star/selection_star_region.hdf5`
(SPEC_PRIORS.md section 1.3, `IMPLEMENTATION.md` section 4): root attr
`GRANULE="region"`; one group per class ("star", "agb", "pahc"), each
carrying `EPS` (K, n_node, n_b), `GROUP_CENTRES` (K, 5), `REF_LOG10_
FLIM` (8,), `LOG10_B_GRID` (n_b,), and attrs `K`, `WORST_RESIDUAL`; the
"agb" group additionally carries `EPS_PHOTOSPHERE` beside `EPS`.
"""

import functools
import os

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.prior import column_grid, depth_groups, selection, star_population

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: A class's own `log10 B` grid: `IMPLEMENTATION.md` section 4 / the
#: brief -- the population's weighted 0.1-99.9% range in 32 points.
N_B_GRID = 32
B_GRID_PCT_LO = 0.1
B_GRID_PCT_HI = 99.9

#: Rule 9: never exact-evaluate a class's full population (order 1e4-1e6
#: stars) at K x n_node x n_b points without bound. Measured on real data
#: (NGC 7129): one query point (one depth-group centre or one residual
#: sample) costs about 0.03 ms per star at n_node=183, n_b=32; a `K`
#: search issues on the order of 10**3 query points in the worst case
#: (a handful of doublings, each fitting K centres and grading 256
#: residual samples), so a cap of 2,000 keeps a class's whole search
#: under a minute even then, well inside the 20,000-star ceiling the
#: brief allows -- a fixed-seed subsample is a proper Monte Carlo
#: reduction of the weighted sum (every kept star keeps its own weight).
SUBSAMPLE_CAP = 2_000
SUBSAMPLE_SEED = 0

#: Every AGB library flux is quoted at this many kpc from the GRAMS
#: convolution (`sed_models/agb/flux.fits`'s own `DISTANCE` header,
#: read live at build time -- this is the fallback only if that header
#: is ever missing).
_FALLBACK_AGB_REF_DISTANCE_KPC = 100.0


# ---------------------------------------------------------------------------
# the region's fallback population: one placement, one weight per class
# ---------------------------------------------------------------------------

def _tile_group_names(f):
    names = [k for k in f.keys() if k.startswith("tile_")]
    return sorted(names, key=lambda s: int(s.split("_")[1]))


def region_population(config, region):
    """The region-level fallback population for the exact-selection
    tables (module docstring): one placement `U_REGION` per star (the
    tile-weighted mean of the population product's own per-tile `U`,
    each tile weighted by its own `Sigma W`), and one total weight per
    star for each of STAR, AGB and PAHC (the sum, over tiles, of that
    tile's own `W_STAR` / `W_AGB` / `W * P_PAHC[:, median]`) -- the same
    reduction from "per tile" to "per region" for both the placement and
    the weight, stated once here rather than separately per class.
    """
    pop_path = config_module.product_path(config, "bms", "star", "population", "tile", region=region)
    if not os.path.exists(pop_path):
        raise FileNotFoundError(
            "prior.star_selection: no field-star population for region %r at %s "
            "-- run the `prior.star_population` RUNBOOK line first" % (region, pop_path))
    with h5py.File(pop_path, "r") as f:
        tile_names = _tile_group_names(f)
        if not tile_names:
            raise ValueError("prior.star_selection: %r carries no tile_* groups" % pop_path)
        g0 = f[tile_names[0]]
        n_star = g0["STAR_INDEX"].shape[0]
        is_evolved = g0["IS_EVOLVED"][:].astype(bool)
        log10_b = g0["LOG10_B"][:].astype(np.float64)
        log10_b_pahc = g0["LOG10_B_PAHC"][:].astype(np.float64)
        log10_b_agb_c = g0["LOG10_B_AGB_C"][:].astype(np.float64)
        log10_b_agb_o = g0["LOG10_B_AGB_O"][:].astype(np.float64)
        median_idx = star_population.PAHC_LIMIT_MEDIAN_INDEX

        u_num = np.zeros(n_star, dtype=np.float64)
        u_den = 0.0
        w_star_total = np.zeros(n_star, dtype=np.float64)
        w_agb_total = np.zeros(n_star, dtype=np.float64)
        w_pahc_total = np.zeros(n_star, dtype=np.float64)
        for name in tile_names:
            g = f[name]
            w_tile = g["W"][:].astype(np.float64)
            tile_weight = float(w_tile.sum())
            u_num += tile_weight * g["U"][:].astype(np.float64)
            u_den += tile_weight
            w_star_total += g["W_STAR"][:].astype(np.float64)
            w_agb_total += g["W_AGB"][:].astype(np.float64)
            w_pahc_total += w_tile * g["P_PAHC"][:, median_idx].astype(np.float64)
        u_region = u_num / u_den if u_den > 0 else u_num

    field_path = config_module.product_path(config, "bms", "trilegal", "field-stars", "region", region=region)
    with h5py.File(field_path, "r") as f:
        flux = f["FNU_MJY"][:].astype(np.float64)
        log_l = f["LOG_L"][:].astype(np.float64)
    if flux.shape[0] != n_star:
        raise ValueError(
            "prior.star_selection: %r's field-star count (%d) does not match "
            "%r's population count (%d)" % (field_path, flux.shape[0], pop_path, n_star))

    return dict(
        u_region=u_region, flux=flux, log_l=log_l, is_evolved=is_evolved,
        log10_b=log10_b, log10_b_pahc=log10_b_pahc,
        log10_b_agb_c=log10_b_agb_c, log10_b_agb_o=log10_b_agb_o,
        w_star_total=w_star_total, w_agb_total=w_agb_total, w_pahc_total=w_pahc_total)


# ---------------------------------------------------------------------------
# the GRAMS library: chemistry, luminosity, and 8-band photometry at the
# library's own reference distance
# ---------------------------------------------------------------------------

def agb_library(config):
    """`(chem, l_sun, flux_ref, ref_distance_kpc)`: the curated GRAMS
    library's own per-model chemistry and luminosity (`sed_models/agb/
    parameters.fits`) and 8-band photometry (`sed_models/agb/convolved/
    <band>.fits`, the sedfitter `convolve_model_dir` format), aligned to
    `parameters.fits`'s own row order by `MODEL_NAME`. `ref_distance_kpc`
    is read from `flux.fits`'s own `DISTANCE` header (cm), the distance
    every convolved flux is quoted at (SPEC_PRIORS.md section 3,
    "Selection").
    """
    root = f"{config.data_root}/sed_models/agb"
    params_path = f"{root}/parameters.fits"
    if not os.path.isfile(params_path):
        raise FileNotFoundError(
            "prior.star_selection: no AGB library at %r -- the GRAMS library "
            "must be curated before this build" % params_path)
    with fits.open(params_path) as hdul:
        data = hdul[1].data
        model_name = np.array([s.strip() for s in np.asarray(data["MODEL_NAME"]).astype(str)])
        chem = np.array([c.strip() for c in np.asarray(data["CHEM"]).astype(str)])
        l_sun = np.asarray(data["L_SUN"], dtype=np.float64)

    flux_header_path = f"{root}/flux.fits"
    with fits.open(flux_header_path) as hdul:
        distance_cm = float(hdul[0].header["DISTANCE"])
    ref_distance_kpc = (distance_cm * u.cm).to(u.kpc).value if distance_cm > 0 \
        else _FALLBACK_AGB_REF_DISTANCE_KPC

    flux_ref = np.empty((model_name.size, N_BANDS), dtype=np.float64)
    for j, key in enumerate(BAND_KEYS):
        conv_path = f"{root}/convolved/{key}.fits"
        with fits.open(conv_path) as hdul:
            t = hdul[1].data
            names_j = np.array([s.strip() for s in np.asarray(t["MODEL_NAME"]).astype(str)])
            flux_j = np.asarray(t["TOTAL_FLUX"], dtype=np.float64).reshape(-1)
        order = {name: i for i, name in enumerate(names_j)}
        try:
            take = np.array([order[name] for name in model_name])
        except KeyError as exc:
            raise ValueError(
                "prior.star_selection: %r is missing model %r that "
                "%r carries" % (conv_path, exc.args[0], params_path))
        flux_ref[:, j] = flux_j[take]

    return chem, l_sun, flux_ref, ref_distance_kpc


def _nearest_1d(values, targets):
    """`(n,)` int: the index into `values` nearest each of `targets`, by
    plain absolute difference -- exact for a 1-D match, no k-d tree
    needed. `values` need not be sorted; sorted internally once.
    """
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    order = np.argsort(values)
    sorted_vals = values[order]
    pos = np.clip(np.searchsorted(sorted_vals, targets), 1, sorted_vals.size - 1)
    left, right = sorted_vals[pos - 1], sorted_vals[pos]
    take_left = (targets - left) <= (right - targets)
    return order[np.where(take_left, pos - 1, pos)]


def agb_matched_flux(config, log_l_evolved, log10_b_agb_o, log10_b_agb_c):
    """`(flux_o, flux_c)`, each `(n_evolved, 8)` mJy: every evolved star's
    nearest GRAMS model, by chemistry, rescaled to the star's own AGB
    brightness (module docstring's `b_agb`-consistent rescaling).
    """
    chem, l_sun, flux_ref, ref_distance_kpc = agb_library(config)
    is_o, is_c = chem == "O", chem == "C"
    l_o_sun, n_orich_models = star_population.agb_orich_l_sun(config)

    l_star = 10.0 ** np.asarray(log_l_evolved, dtype=np.float64)
    idx_o_within = _nearest_1d(l_sun[is_o], l_star)
    idx_c_within = _nearest_1d(np.log10(l_sun[is_c]), np.log10(l_star))
    model_o = flux_ref[is_o][idx_o_within]
    model_c = flux_ref[is_c][idx_c_within]
    l_model_o = l_sun[is_o][idx_o_within]

    # (100 kpc / 1 kpc)**2 brings the library's own reference distance to
    # 1 kpc; the further 10**(log10_B_i - log10_B_model_at_1kpc) matches
    # the star's own brightness (module docstring's rescaling rule).
    to_1kpc = ref_distance_kpc ** 2
    log10_b_model_o_at_1kpc = np.log10(l_model_o / l_o_sun)  # b_agb's O term, C's is 0
    flux_o = model_o * to_1kpc * 10.0 ** (log10_b_agb_o - log10_b_model_o_at_1kpc)[:, None]
    flux_c = model_c * to_1kpc * 10.0 ** (log10_b_agb_c)[:, None]
    return flux_o, flux_c


# ---------------------------------------------------------------------------
# the exact selection, algebraically reduced to a per-(star, node)
# critical brightness (module docstring)
# ---------------------------------------------------------------------------

def _delta5_to_f_lim8(delta_5, ref_log10_flim_8):
    """The full 8-band `F_lim` (mJy) at a depth group's own shifted
    limits `ref_log10_flim_8 + Delta` on `selection.BANDS_DEPTH`; the
    three 2MASS bands stay at the region's own reference
    (`selection.split_common_mode`'s own convention: they carry no
    `Delta` component)."""
    delta_5 = np.asarray(delta_5, dtype=np.float64)
    delta_8 = np.zeros(selection.N_BANDS, dtype=np.float64)
    delta_8[selection._BAND_DEPTH_IDX] = delta_5
    return 10.0 ** (np.asarray(ref_log10_flim_8, dtype=np.float64) + delta_8)


def _weighted_percentile(values, weights, pct):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[finite], weights[finite]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return float(np.interp(pct / 100.0, cum, values))


def log10_b_grid_for(log10_b, weight, n=N_B_GRID):
    """The class's own `log10 B` grid (module docstring): `n` points
    linear between the population's own weighted 0.1% and 99.9% `log10
    B`.
    """
    lo = _weighted_percentile(log10_b, weight, B_GRID_PCT_LO)
    hi = _weighted_percentile(log10_b, weight, B_GRID_PCT_HI)
    return np.linspace(lo, hi, int(n))


def subsample_population(weight, cap=SUBSAMPLE_CAP, seed=SUBSAMPLE_SEED):
    """The nonzero-weight rows, fixed-seed subsampled to at most `cap`
    (rule 9) -- each kept row keeps its own weight, a standard Monte
    Carlo reduction of a weighted sum.
    """
    idx = np.flatnonzero(np.asarray(weight, dtype=np.float64) > 0.0)
    n_nonzero = idx.size
    if n_nonzero > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx, n_nonzero


def build_population_eps(config, flux, log10_b, weight, u_region, a_nodes,
                          b_grid, ref_log10_flim, cap=SUBSAMPLE_CAP, seed=SUBSAMPLE_SEED):
    """`(population_eps, n_used, n_nonzero)`: `population_eps(delta_5) ->
    (n_points, n_a, n_b)`, the class's exact selection (module docstring)
    on this population, subsampled per `subsample_population`.
    """
    idx, n_nonzero = subsample_population(weight, cap=cap, seed=seed)
    flux_s = np.asarray(flux, dtype=np.float64)[idx]
    log10b_s = np.asarray(log10_b, dtype=np.float64)[idx]
    w_s = np.asarray(weight, dtype=np.float64)[idx]
    u_s = np.asarray(u_region, dtype=np.float64)[idx]
    n_star = idx.size
    a_nodes = np.asarray(a_nodes, dtype=np.float64)
    n_node = a_nodes.size
    n_b = np.asarray(b_grid, dtype=np.float64).size
    w_sum = float(w_s.sum())

    a_i = a_nodes[None, :] * u_s[:, None]                       # (n_star, n_node)
    w_dense = selection.law_dense_weight(a_i)
    kappa = selection.kappa_hybrid(config, w_dense)             # (n_star, n_node, 8)
    # L(star, node, band): the part of the two-of-eight test that does
    # not depend on the query log10 B (module docstring's algebra).
    l_star_node_band = (np.log10(flux_s)[:, None, :]
                        - 0.4 * a_i[:, :, None] * kappa
                        - log10b_s[:, None, None])

    def population_eps(delta_5):
        delta_5 = np.atleast_2d(np.asarray(delta_5, dtype=np.float64))
        n_points = delta_5.shape[0]
        out = np.empty((n_points, n_node, n_b), dtype=np.float64)
        for p in range(n_points):
            f_lim = _delta5_to_f_lim8(delta_5[p], ref_log10_flim)
            m = np.log10(f_lim)[None, None, :] - l_star_node_band  # (n_star, n_node, 8)
            b_crit = np.partition(m, selection.MIN_BANDS - 1, axis=-1)[..., selection.MIN_BANDS - 1]
            cond = b_crit[:, :, None] <= np.asarray(b_grid, dtype=np.float64)[None, None, :]
            out[p] = w_s.dot(cond.reshape(n_star, -1)).reshape(n_node, n_b) / w_sum
        return out

    return population_eps, n_star, n_nonzero


# ---------------------------------------------------------------------------
# per-region, per-class table
# ---------------------------------------------------------------------------

def _fit_one(config, region, fit_dg, k_start, n_sources, a_nodes,
             population_eps, b_grid):
    knots, model, residual = selection.fit_by_residual(
        fit_dg, region, population_eps, a_nodes, b_grid, population_eps,
        n_sources, k_start)
    return knots, model, residual


def build_region(config, region):
    pop = region_population(config, region)
    a_nodes = column_grid.nodes(config)

    dg_path = config_module.product_path(config, "bms", "sesna", "depth-groups", "region")
    knots0 = depth_groups.DepthGroups.read(dg_path, region)
    k_start = knots0.n_groups
    n_sources = int(limits_module.limits(config, region).shape[0])
    fit_dg = functools.partial(depth_groups.fit_depth_groups, config)

    b_grid_star = log10_b_grid_for(pop["log10_b"], pop["w_star_total"])
    peps_star, n_used_star, n_nz_star = build_population_eps(
        config, pop["flux"], pop["log10_b"], pop["w_star_total"], pop["u_region"],
        a_nodes, b_grid_star, knots0.ref_log10_flim)

    b_grid_pahc = log10_b_grid_for(pop["log10_b_pahc"], pop["w_pahc_total"])
    peps_pahc, n_used_pahc, n_nz_pahc = build_population_eps(
        config, pop["flux"], pop["log10_b_pahc"], pop["w_pahc_total"], pop["u_region"],
        a_nodes, b_grid_pahc, knots0.ref_log10_flim)

    evolved = pop["is_evolved"]
    n_evolved = int(evolved.sum())
    flux_o, flux_c = agb_matched_flux(
        config, pop["log_l"][evolved], pop["log10_b_agb_o"][evolved], pop["log10_b_agb_c"][evolved])
    w_agb_evolved = pop["w_agb_total"][evolved]
    f_c = star_population.F_C
    weight_o, weight_c = (1.0 - f_c) * w_agb_evolved, f_c * w_agb_evolved
    u_evolved = pop["u_region"][evolved]
    log10_b_o, log10_b_c = pop["log10_b_agb_o"][evolved], pop["log10_b_agb_c"][evolved]

    log10_b_agb_combined = np.concatenate([log10_b_o, log10_b_c])
    weight_agb_combined = np.concatenate([weight_o, weight_c])
    u_agb_combined = np.concatenate([u_evolved, u_evolved])
    b_grid_agb = log10_b_grid_for(log10_b_agb_combined, weight_agb_combined)

    flux_dusty = np.concatenate([flux_o, flux_c], axis=0)
    peps_agb, n_used_agb, n_nz_agb = build_population_eps(
        config, flux_dusty, log10_b_agb_combined, weight_agb_combined, u_agb_combined,
        a_nodes, b_grid_agb, knots0.ref_log10_flim)

    # the three classes' own K-searches are independent of one another
    # (rule 8: the largest iterator here, since each search issues its
    # own few hundred exact-evaluation query points) -- run them together.
    (knots_star, model_star, res_star), (knots_pahc, model_pahc, res_pahc), \
        (knots_agb, model_agb, res_agb) = Parallel(n_jobs=min(config.n_jobs, 3), prefer="threads")(
            delayed(_fit_one)(config, region, fit_dg, k_start, n_sources, a_nodes, peps, bg)
            for peps, bg in ((peps_star, b_grid_star), (peps_pahc, b_grid_pahc), (peps_agb, b_grid_agb)))

    # the photospheric lower bound (SPEC_PRIORS.md section 3): the same
    # evolved population, weights and brightness axis, the star's own
    # TRILEGAL flux standing in for the GRAMS model -- fit at the SAME
    # group centres already found for the dusty table, so the two share
    # one (K, n_a, n_b) shape.
    flux_photo = np.concatenate([pop["flux"][evolved], pop["flux"][evolved]], axis=0)
    peps_agb_photo, _, _ = build_population_eps(
        config, flux_photo, log10_b_agb_combined, weight_agb_combined, u_agb_combined,
        a_nodes, b_grid_agb, knots0.ref_log10_flim)
    model_agb_photo = selection.PassFractionModel.fit(peps_agb_photo, knots_agb, a_nodes, b_grid_agb)

    return dict(
        star=dict(model=model_star, residual=res_star, n_used=n_used_star, n_nonzero=n_nz_star),
        pahc=dict(model=model_pahc, residual=res_pahc, n_used=n_used_pahc, n_nonzero=n_nz_pahc),
        agb=dict(model=model_agb, model_photo=model_agb_photo, residual=res_agb,
                  n_used=n_used_agb, n_nonzero=n_nz_agb, n_evolved=n_evolved),
        pop=pop, a_nodes=a_nodes)


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def _write_class(g, entry, with_photosphere=False):
    model = entry["model"]
    g.create_dataset("EPS", data=model.eps_table.astype("f4"))
    g.create_dataset("GROUP_CENTRES", data=np.asarray(model.knots.group_centres, dtype="f8"))
    g.create_dataset("REF_LOG10_FLIM", data=np.asarray(model.knots.ref_log10_flim, dtype="f8"))
    g.create_dataset("LOG10_B_GRID", data=model.b_grid.astype("f8"))
    g.attrs["K"] = int(model.eps_table.shape[0])
    g.attrs["WORST_RESIDUAL"] = float(entry["residual"]["l1_rel_worst_node"])
    if with_photosphere:
        g.create_dataset("EPS_PHOTOSPHERE", data=entry["model_photo"].eps_table.astype("f4"))


def write_region(config, region, result):
    path = config_module.product_path(config, "bms", "star", "selection", "region", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "region"
        _write_class(f.create_group("star"), result["star"])
        _write_class(f.create_group("agb"), result["agb"], with_photosphere=True)
        _write_class(f.create_group("pahc"), result["pahc"])
    return path


# ---------------------------------------------------------------------------
# report (rules 10, 11, 13): K, residual, algebraic acceptance, the
# evaluate() round trip
# ---------------------------------------------------------------------------

def _acceptance(model):
    """`(max_increase_in_a, max_decrease_in_b)`: how far `EPS` violates
    non-increasing-in-`a` / non-decreasing-in-`log10 B`, at every (k, b)
    or (k, a) slice -- zero when the table is fully monotone.
    """
    eps = model.eps_table
    d_a = np.diff(eps, axis=1)
    max_increase_a = float(np.max(np.clip(d_a, 0.0, None))) if d_a.size else 0.0
    d_b = np.diff(eps, axis=2)
    max_decrease_b = float(np.max(np.clip(-d_b, 0.0, None))) if d_b.size else 0.0
    return max_increase_a, max_decrease_b


def _evaluate_round_trip(model):
    """`evaluate()` at the median group centre, at node 0 exactly
    (`node_w=0`) and `s=0`, against the stored table at the same point --
    the acceptance identity the brief names."""
    k_mid = model.eps_table.shape[0] // 2
    b_mid = model.b_grid.size // 2
    centre = model.knots.group_centres[k_mid:k_mid + 1]
    direct = float(model.eps_table[k_mid, 0, b_mid])
    evaluated = float(model.evaluate(
        node_lo=np.array([0]), node_w=np.array([0.0]),
        log10_b=np.array([model.b_grid[b_mid]]), s=np.array([0.0]), delta_5=centre)[0])
    return abs(evaluated - direct)


def _eps_at(model, a_query, b_query, a_nodes, k_mid):
    node_lo, node_w = column_grid.bracket(np.array([a_query]), a_nodes)
    centre = model.knots.group_centres[k_mid:k_mid + 1]
    return float(model.evaluate(node_lo=node_lo, node_w=node_w,
                                log10_b=np.array([b_query]), s=np.array([0.0]), delta_5=centre)[0])


def report(region, result):
    a_nodes = result["a_nodes"]
    lines = []
    for cls in ("star", "pahc"):
        entry = result[cls]
        model = entry["model"]
        k_mid = model.eps_table.shape[0] // 2
        b_med = float(np.median(model.b_grid))
        inc_a, dec_b = _acceptance(model)
        round_trip = _evaluate_round_trip(model)
        lines.append(
            "star_selection: %s %s: K=%d (n_used=%d/%d nonzero-weight) worst_residual=%.4f "
            "eps(a=0)=%.4f eps(a=2)=%.4f at median group, median log10B "
            "monotone_a_viol=%.2e monotone_b_viol=%.2e evaluate_round_trip=%.2e"
            % (region, cls, model.eps_table.shape[0], entry["n_used"], entry["n_nonzero"],
               entry["residual"]["l1_rel_worst_node"],
               _eps_at(model, 0.0, b_med, a_nodes, k_mid),
               _eps_at(model, 2.0, b_med, a_nodes, k_mid),
               inc_a, dec_b, round_trip))
    entry = result["agb"]
    for name, model in (("dusty", entry["model"]), ("photosphere", entry["model_photo"])):
        k_mid = model.eps_table.shape[0] // 2
        b_med = float(np.median(model.b_grid))
        inc_a, dec_b = _acceptance(model)
        round_trip = _evaluate_round_trip(model)
        lines.append(
            "star_selection: %s agb(%s): K=%d (n_used=%d/%d nonzero-weight, n_evolved=%d) "
            "worst_residual=%.4f eps(a=0)=%.4f eps(a=2)=%.4f at median group, median log10B "
            "monotone_a_viol=%.2e monotone_b_viol=%.2e evaluate_round_trip=%.2e"
            % (region, name, model.eps_table.shape[0], entry["n_used"], entry["n_nonzero"],
               entry["n_evolved"], entry["residual"]["l1_rel_worst_node"],
               _eps_at(model, 0.0, b_med, a_nodes, k_mid),
               _eps_at(model, 2.0, b_med, a_nodes, k_mid),
               inc_a, dec_b, round_trip))
    return lines


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _build_one(config, region):
    result = build_region(config, region)
    path = write_region(config, region, result)
    for line in report(region, result):
        print(line, flush=True)
    print("star_selection: %s -> %s" % (region, path), flush=True)
    return path


def build(config, regions=None):
    """Writes the exact-selection tables for `regions` (default: all
    thirty), one product per region, parallel over regions (rule 8; the
    per-region cost -- three classes' worth of depth-group searches -- is
    the largest iterator here)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(_build_one)(config, region) for region in region_names)


if __name__ == "__main__":
    run(build)
