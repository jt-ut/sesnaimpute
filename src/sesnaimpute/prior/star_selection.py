"""The exact per-source selection for STAR, AGB and PAHC (SPEC_PRIORS.md
section 1.3's exact selection evaluated at each source's own eight
limits; section 2.1's `eps_s(a_i, B_i)` on TRILEGAL's own fluxes;
section 3 "Selection" on GRAMS colours with the photospheric bound;
section 4's PAHC weight `w * P(q)`).

`prior.selection.pass_fractions_binned`/`pass_fractions_binned_multi`
need, for one class, the class's own external population (SPEC_PRIORS.md
0.3, C3) and, per source, the query extinctions `a = x * A_s` on the
shared scaled-extinction ladder `selection.X_LADDER`, `x = a / A_s`, and
the hybrid law's per-band dimming at each query extinction. No depth
grouping and no common-mode split: every source's own eight limits and
own column enter directly. The population's colour distribution is
conditioned on brightness: each member is counted only within its own
`log10 B` bin on the class's 24-point grid, not across the whole
population (SPEC_PRIORS.md 1.3).

Members per class: STAR is the whole field-star population weighted by
`W_STAR`; PAHC is the same population weighted by `W * P_PAHC` at the
region's median 8 micron completeness limit; AGB dusty is the evolved
stars matched to the nearest GRAMS model by chemistry, weighted by the
O/C split of `W_AGB`; AGB photosphere is the same evolved stars and
weights with the star's own TRILEGAL flux standing in for the GRAMS
flux, the lower bound SPEC_PRIORS.md section 3 calls for. STAR and PAHC
share one field-star population, so they are drawn as ONE fixed-seed
member subsample (`shared_subsample`, the union of rows either class
weights nonzero) and run through the compiled kernel together
(`_FusedPair`, `pass_fractions_binned_multi`): each member's 8-band flux
is read once and both classes' critical brightness computed from that
one read.

Writes one product per region, row-aligned with the curated catalogue,
`bms/star/selection_star_source.hdf5`: root attr `GRANULE="source"`,
`X_LADDER` (n_x,) f8, `LOG10_B_GRID_STAR`/`_PAHC`/`_AGB` (n_b,) f8, and
`EPS_STAR`/`EPS_PAHC`/`EPS_AGB`/`EPS_AGB_PHOTOSPHERE` (n, n_x, n_b) f2.
Sources are batched (`sesnaimpute.batches.batches`) so no batch's working
arrays exceed the byte budget.
"""

import os

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.prior import selection, star_population

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: The shared scaled-extinction ladder and brightness-grid size (module
#: docstring): re-exported from `prior.selection`, the one place they
#: are defined.
X_LADDER = selection.X_LADDER
N_B_GRID = selection.N_B_GRID
SUBSAMPLE_CAP = selection.SUBSAMPLE_CAP
SUBSAMPLE_SEED = 0

#: A class's own `log10 B` grid: the population's weighted 0.1-99.9%
#: range in `N_B_GRID` points.
B_GRID_PCT_LO = 0.1
B_GRID_PCT_HI = 99.9

#: The per-batch working-array budget (`sesnaimpute.batches.batches`).
BATCH_BUDGET_BYTES = 512 << 20

#: Every AGB library flux is quoted at this many kpc from the GRAMS
#: convolution (`sed_models/agb/flux.fits`'s own `DISTANCE` header, read
#: live at build time -- this is the fallback only if that header is
#: ever missing).
_FALLBACK_AGB_REF_DISTANCE_KPC = 100.0


# ---------------------------------------------------------------------------
# the region's population: one weight per star for each of STAR, AGB, PAHC
# ---------------------------------------------------------------------------

def _tile_group_names(f):
    names = [k for k in f.keys() if k.startswith("tile_")]
    return sorted(names, key=lambda s: int(s.split("_")[1]))


def region_population(config, region):
    """The region-level population for the exact per-source selection
    (module docstring): TRILEGAL's own fluxes and luminosities, and one
    total weight per star for each of STAR, AGB and PAHC (the sum, over
    tiles, of that tile's own `W_STAR` / `W_AGB` / `W * P_PAHC[:,
    median]`).
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

        w_star_total = np.zeros(n_star, dtype=np.float64)
        w_agb_total = np.zeros(n_star, dtype=np.float64)
        w_pahc_total = np.zeros(n_star, dtype=np.float64)
        for name in tile_names:
            g = f[name]
            w_tile = g["W"][:].astype(np.float64)
            w_star_total += g["W_STAR"][:].astype(np.float64)
            w_agb_total += g["W_AGB"][:].astype(np.float64)
            w_pahc_total += w_tile * g["P_PAHC"][:, median_idx].astype(np.float64)

    field_path = config_module.product_path(config, "bms", "trilegal", "field-stars", "region", region=region)
    with h5py.File(field_path, "r") as f:
        flux = f["FNU_MJY"][:].astype(np.float64)
        log_l = f["LOG_L"][:].astype(np.float64)
    if flux.shape[0] != n_star:
        raise ValueError(
            "prior.star_selection: %r's field-star count (%d) does not match "
            "%r's population count (%d)" % (field_path, flux.shape[0], pop_path, n_star))

    return dict(
        flux=flux, log_l=log_l, is_evolved=is_evolved,
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
    brightness (module docstring's rescaling rule).
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
# brightness grid and subsample
# ---------------------------------------------------------------------------

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
    -- each kept row keeps its own weight, a standard Monte Carlo
    reduction of a weighted sum.
    """
    idx = np.flatnonzero(np.asarray(weight, dtype=np.float64) > 0.0)
    n_nonzero = idx.size
    if n_nonzero > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx, n_nonzero


def shared_subsample(weight_a, weight_b, cap=SUBSAMPLE_CAP, seed=SUBSAMPLE_SEED):
    """As `subsample_population`, but over the union of rows with a
    nonzero weight in EITHER `weight_a` or `weight_b` -- the one member
    draw two classes share (module docstring, STAR/PAHC fusion)."""
    weight_a = np.asarray(weight_a, dtype=np.float64)
    weight_b = np.asarray(weight_b, dtype=np.float64)
    idx = np.flatnonzero((weight_a > 0.0) | (weight_b > 0.0))
    n_nonzero = idx.size
    if n_nonzero > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx, n_nonzero


def bin_of_pop_for(log10_b, b_grid):
    """`(n,)` int64: the index of `b_grid`'s own nearest point to each
    member's `log10_b` (`prior.gal`'s own bin-by-own-value pattern) --
    the cell edges are the grid's own midpoints, so a member always
    lands in the single bin its value is closest to."""
    edges = 0.5 * (b_grid[1:] + b_grid[:-1])
    return np.clip(np.searchsorted(edges, log10_b), 0, b_grid.size - 1).astype(np.int64)


def _log10_finite(flux):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.asarray(flux, dtype=np.float64))


# ---------------------------------------------------------------------------
# per-region build: one population per class, one batched pass over sources
# ---------------------------------------------------------------------------

class _ClassPopulation:
    """One class's subsampled population, ready for `selection.
    pass_fractions_binned`: `log10_flux` (n_used, 8), `log10_b`
    (n_used,), `weight` (n_used,), the class's own `b_grid` (N_B_GRID,),
    and `bin_of_pop` (n_used,) int -- each member's own nearest `b_grid`
    point, the colour distribution conditioned on brightness
    (SPEC_PRIORS.md 1.3)."""

    def __init__(self, flux, log10_b, weight):
        log10_b = np.asarray(log10_b, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)
        b_grid = log10_b_grid_for(log10_b, weight)
        idx, n_nonzero = subsample_population(weight)
        self.log10_flux = np.ascontiguousarray(_log10_finite(flux[idx]))
        self.log10_b = np.ascontiguousarray(log10_b[idx])
        self.weight = np.ascontiguousarray(weight[idx])
        self.b_grid = np.ascontiguousarray(b_grid)
        self.bin_of_pop = np.ascontiguousarray(bin_of_pop_for(self.log10_b, self.b_grid))
        self.n_used = idx.size
        self.n_nonzero = n_nonzero


class _FusedPair:
    """STAR and PAHC together, drawn from ONE shared member subsample
    (module docstring) so `selection.pass_fractions_binned_multi` reads
    each member's 8-band flux once for both classes: `log10_flux`
    (n_used, 8) shared; `log10_b_pop`, `weight`, `bin_of_pop` each
    (n_used, 2) [STAR, PAHC]; `b_grid` (N_B_GRID, 2). `n_used_a`/
    `n_used_b`, `n_nonzero_a`/`n_nonzero_b` are each class's own
    nonzero-weight count for the report; `n_used`/`n_nonzero` are the
    shared draw's own size and its union nonzero count."""

    def __init__(self, flux, log10_b_a, weight_a, log10_b_b, weight_b):
        log10_b_a = np.asarray(log10_b_a, dtype=np.float64)
        log10_b_b = np.asarray(log10_b_b, dtype=np.float64)
        weight_a = np.asarray(weight_a, dtype=np.float64)
        weight_b = np.asarray(weight_b, dtype=np.float64)
        b_grid_a = log10_b_grid_for(log10_b_a, weight_a)
        b_grid_b = log10_b_grid_for(log10_b_b, weight_b)
        idx, n_nonzero = shared_subsample(weight_a, weight_b)

        self.log10_flux = np.ascontiguousarray(_log10_finite(flux[idx]))
        self.log10_b_pop = np.ascontiguousarray(
            np.stack([log10_b_a[idx], log10_b_b[idx]], axis=1))
        self.weight = np.ascontiguousarray(np.stack([weight_a[idx], weight_b[idx]], axis=1))
        self.bin_of_pop = np.ascontiguousarray(np.stack(
            [bin_of_pop_for(log10_b_a[idx], b_grid_a),
             bin_of_pop_for(log10_b_b[idx], b_grid_b)], axis=1))
        self.b_grid = np.ascontiguousarray(np.stack([b_grid_a, b_grid_b], axis=1))
        self.b_grid_a, self.b_grid_b = b_grid_a, b_grid_b
        self.n_used = idx.size
        self.n_nonzero = n_nonzero
        self.n_nonzero_a = int(np.count_nonzero(weight_a > 0.0))
        self.n_nonzero_b = int(np.count_nonzero(weight_b > 0.0))


def _class_populations(config, pop):
    """`{"star_pahc": _FusedPair, "agb": ..., "agb_photo": ...}`
    (module docstring's four member sets; STAR and PAHC fused onto one
    shared member draw)."""
    star_pahc = _FusedPair(pop["flux"], pop["log10_b"], pop["w_star_total"],
                            pop["log10_b_pahc"], pop["w_pahc_total"])

    evolved = pop["is_evolved"]
    flux_o, flux_c = agb_matched_flux(
        config, pop["log_l"][evolved], pop["log10_b_agb_o"][evolved], pop["log10_b_agb_c"][evolved])
    w_agb_evolved = pop["w_agb_total"][evolved]
    f_c = star_population.F_C
    weight_o, weight_c = (1.0 - f_c) * w_agb_evolved, f_c * w_agb_evolved
    log10_b_o = pop["log10_b_agb_o"][evolved]
    log10_b_c = pop["log10_b_agb_c"][evolved]
    flux_photo_evolved = pop["flux"][evolved]

    log10_b_combined = np.concatenate([log10_b_o, log10_b_c])
    weight_combined = np.concatenate([weight_o, weight_c])
    flux_dusty = np.concatenate([flux_o, flux_c], axis=0)
    flux_photo = np.concatenate([flux_photo_evolved, flux_photo_evolved], axis=0)

    agb = _ClassPopulation(flux_dusty, log10_b_combined, weight_combined)
    # the photospheric lower bound (SPEC_PRIORS.md section 3): the same
    # evolved population, weights and brightness axis, the star's own
    # TRILEGAL flux standing in for the GRAMS model -- shares the dusty
    # table's own `log10 B` grid and bin assignment so the two products
    # sit beside one another, and (since weight and log10_b are
    # identical) the SAME subsample, drawn with the same seed.
    idx_photo, _ = subsample_population(weight_combined)
    agb_photo = _ClassPopulation.__new__(_ClassPopulation)
    agb_photo.log10_flux = np.ascontiguousarray(_log10_finite(flux_photo[idx_photo]))
    agb_photo.log10_b = agb.log10_b
    agb_photo.weight = agb.weight
    agb_photo.b_grid = agb.b_grid
    agb_photo.bin_of_pop = agb.bin_of_pop
    agb_photo.n_used = agb.n_used
    agb_photo.n_nonzero = agb.n_nonzero

    return dict(star_pahc=star_pahc, agb=agb, agb_photo=agb_photo)


def _row_bytes(n_x, n_b):
    """The per-source working-array footprint one batch holds: the
    source's own limits and query-extinction/kappa arrays, plus the
    fused STAR+PAHC pass's `(n_x, n_b, 2)` f4 output and the two AGB
    passes' `(n_x, n_b)` f4 outputs, before each is cast to `f2` on
    write."""
    return (N_BANDS * 8 + n_x * 8 + n_x * N_BANDS * 8
            + (2 * n_x * n_b + 2 * n_x * n_b) * 4)


def build_and_write_region(config, region):
    """Computes and writes the region's exact per-source selection
    (module docstring), one batch of sources at a time so no batch's
    working arrays exceed `BATCH_BUDGET_BYTES`."""
    pop = region_population(config, region)
    classes = _class_populations(config, pop)

    log10_lim = _log10_finite(limits_module.limits(config, region))
    n_source = log10_lim.shape[0]

    adopted_path = config_module.product_path(
        config, "sky/derived", "adopted", "column", "source", region=region)
    a_col = np.asarray(
        access.per_source(config, region, adopted_path, ["A_COL_K"])["A_COL_K"], dtype=np.float64)
    if a_col.shape[0] != n_source:
        raise ValueError(
            "prior.star_selection: %r's column count (%d) does not match "
            "the region's %d sources" % (adopted_path, a_col.shape[0], n_source))

    x_ladder = X_LADDER
    n_x = x_ladder.size
    n_b = N_B_GRID

    path = config_module.product_path(config, "bms", "star", "selection", "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.create_dataset("X_LADDER", data=x_ladder.astype("f8"))
        f.create_dataset("LOG10_B_GRID_STAR", data=classes["star_pahc"].b_grid_a.astype("f8"))
        f.create_dataset("LOG10_B_GRID_PAHC", data=classes["star_pahc"].b_grid_b.astype("f8"))
        f.create_dataset("LOG10_B_GRID_AGB", data=classes["agb"].b_grid.astype("f8"))
        ds_star = f.create_dataset("EPS_STAR", shape=(n_source, n_x, n_b), dtype="f2")
        ds_pahc = f.create_dataset("EPS_PAHC", shape=(n_source, n_x, n_b), dtype="f2")
        ds_agb = f.create_dataset("EPS_AGB", shape=(n_source, n_x, n_b), dtype="f2")
        ds_agb_photo = f.create_dataset("EPS_AGB_PHOTOSPHERE", shape=(n_source, n_x, n_b), dtype="f2")

        row_bytes = _row_bytes(n_x, n_b)
        for start, stop in batches_module.batches(n_source, row_bytes, budget_bytes=BATCH_BUDGET_BYTES):
            lim_b = np.ascontiguousarray(log10_lim[start:stop])
            a_b = a_col[start:stop]
            a_query_b = np.ascontiguousarray(x_ladder[None, :] * a_b[:, None])
            w_dense_b = selection.law_dense_weight(a_query_b)
            kappa_b = np.ascontiguousarray(selection.kappa_hybrid(config, w_dense_b))

            fp = classes["star_pahc"]
            eps2 = selection.pass_fractions_binned_multi(
                lim_b, a_query_b, kappa_b, fp.log10_flux, fp.log10_b_pop,
                fp.weight, fp.bin_of_pop, fp.b_grid)
            ds_star[start:stop] = eps2[..., 0].astype("f2")
            ds_pahc[start:stop] = eps2[..., 1].astype("f2")

            # AGB and AGB-photosphere: unchanged -- marginal (not
            # brightness-conditioned), unfused. Their population is a
            # handful of GRAMS-matched stars (tens of members), far too
            # few for the 24-bin conditioning STAR/PAHC now use: most
            # bins would hold zero or one member and read as a hard
            # zero rather than the population's real, smooth density.
            cp = classes["agb"]
            eps = selection.pass_curves(
                lim_b, a_query_b, kappa_b, cp.log10_flux, cp.log10_b, cp.weight, cp.b_grid)
            ds_agb[start:stop] = eps.astype("f2")

            cp = classes["agb_photo"]
            eps = selection.pass_curves(
                lim_b, a_query_b, kappa_b, cp.log10_flux, cp.log10_b, cp.weight, cp.b_grid)
            ds_agb_photo[start:stop] = eps.astype("f2")

    return path, classes, n_source


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(region, classes, n_source, path):
    lines = []
    fp = classes["star_pahc"]
    lines.append(
        "star_selection: %s star+pahc (fused): n_used=%d shared draw, "
        "star nonzero=%d, pahc nonzero=%d, n_source=%d"
        % (region, fp.n_used, fp.n_nonzero_a, fp.n_nonzero_b, n_source))
    for name in ("agb", "agb_photo"):
        cp = classes[name]
        lines.append(
            "star_selection: %s %s: n_used=%d/%d nonzero-weight, n_source=%d"
            % (region, name, cp.n_used, cp.n_nonzero, n_source))
    lines.append("star_selection: %s -> %s" % (region, path))
    return lines


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _build_one(config, region):
    path, classes, n_source = build_and_write_region(config, region)
    for line in report(region, classes, n_source, path):
        print(line, flush=True)
    return path


def build(config, regions=None):
    """Writes the exact per-source selection tables for `regions`
    (default: all thirty), one product per region. `numba`'s thread
    count is set from `config.n_jobs` once; each region's four classes
    are computed and written batch by batch (module docstring)."""
    import numba
    numba.set_num_threads(max(1, int(config.n_jobs)))
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        _build_one(config, region)


if __name__ == "__main__":
    run(build)
