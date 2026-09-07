"""The template-weight tables P5 (SPEC_BMSTP_DRAFT.md sec 1.4, 4.1, 4.2,
5.1-5.6; IMPLEMENTATION_BMSTP_DRAFT.md sec 1.2 P5, sec 3 row 1.7):
`pi_C(theta, B) = p_C(shape of theta | B) / rho_C(theta)`, one factor per
population statement over a library's own templates, stored per band cell
on the common brightness axis so the fitter's read (sec 4.2) is a per-
template gather along one row.

Each factor function returns `(W, C_F, D_F, normalised, source)`: `W`
`(n_model, 120)` float32 on `LOG10_B_CENTERS`; `C_F` `(n_model,)` f8, the
factor's own per-template offset; `D_F` the density-table column name a
per-source offset comes from at read time, or `""`; `normalised` True if
the factor sums to 1 over templates at every cell (a shape statement),
False if it is a probability carried as-is (`contrast`). `rho_C(theta)` is
the register's own `RHO_KDE1` (`fit.terms.library_weights`'s own column,
sec 3.5: "supplied with the library").

Survey products (galz, yso, h2shock) are built once; region products
(sps, agb, pahc) once per region named on the command line. PAHC is
regional because its `type` factor borrows the region's own sps type
histogram at each PAHC template's nearest sps atmosphere match (owner
ruling: PAHC's library carries no atmosphere-type axis of its own).
"""

import os

import h5py
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.bmstp import grid
from sesnaimpute.build import run
from sesnaimpute.population import star_population

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: Chabrier (2003, PASP 115, 763, eq. 17-18) system IMF: lognormal in
#: log10 M below 1 Msun, power law above (spec sec 5.5, sec 10 item 1).
CHABRIER_LOG_MC = np.log10(0.2)     # dex, the lognormal centre, Msun
CHABRIER_SIGMA_DEX = 0.55           # dex, the lognormal width below 1 Msun
CHABRIER_SLOPE = 1.35               # dN/dlog M ~ M^-CHABRIER_SLOPE above 1 Msun

#: A normalised factor's value in an empty cell, and the floor every
#: normalised template-weight factor is renormalised against (spec sec 2,
#: "the floor": no hypothesis at -inf from the prior); shared with
#: `bmstp.grid`'s shape floor, same number, same citation.
FACTOR_FLOOR = grid.FLOOR

#: A probability factor (not a distribution over templates) is floored
#: and capped at the same `FACTOR_FLOOR`/`1 - FACTOR_FLOOR` for the same
#: reason (spec sec 2).
PROB_CAP = 1.0 - FACTOR_FLOOR

#: Carbon fraction f_C (spec sec 5.2, Le Bertre et al. 2003), also on the
#: star_population product's own attrs; used here to mix the two
#: chemistries' Riebel tau distributions.
F_C = star_population.F_C

_LOG10_B_ORIGIN_YSO = grid.LOG10_B_ORIGIN_TEMPLATE
_LOG10_B_ORIGIN_STAR = grid.LOG10_B_ORIGIN_TEMPLATE

#: The five YSO sub-grid directories under `sed_models/yso/` (different
#: geometries, different parameter sets): read only to join each
#: sub-grid's own `parameters.fits` (inclination) to the pooled
#: register by `MODEL_NAME` -- a file lookup, not a weighting.
YSO_SUBGRIDS = (("c0", "C0"), ("cI", "CI"), ("cII", "CII"),
                ("cIII", "CIII"), ("td", "TD"))

_REGISTER_FILE = {key: "%s_register.hdf5" % key for key in definitions.CLASS_REGISTER.values()}


# ---------------------------------------------------------------------------
# registers and grids
# ---------------------------------------------------------------------------

def _read_register(config, cls):
    """The register's `MODEL_NAME`, `RHO_KDE1` (`rho_C(theta)`, sec 3.5),
    `SUBCLASS` (report-only diagnostic label, sec 5.5's check) and
    reference fluxes in every band, in the register's own row order.
    """
    path = f"{config.inputs['sed_models']}/registers/{_REGISTER_FILE[cls]}"
    with h5py.File(path, "r") as f:
        m = f["models"]
        names = np.char.decode(m["MODEL_NAME"][:].astype("S"), "utf-8")
        rho = m["RHO_KDE1"][:].astype(np.float64)
        subclass = np.char.decode(m["SUBCLASS"][:].astype("S"), "utf-8")
        f_ref = {b: m[f"F_REF_{b}"][:].astype(np.float64)
                 for b in ("I1", "I2", "I3", "I4", "J", "H", "Ks", "M1")}
    return dict(names=names, rho=rho, f_ref=f_ref, subclass=subclass)


def _log10_b_centers(origin):
    edges = grid.log10_b_edges(origin)
    return 0.5 * (edges[:-1] + edges[1:])


def _normalise_over_theta(w):
    """`w` `(n_model, n_b)`, normalised over the model axis at every
    brightness cell; a cell with no weight at all (an empty class, spec
    sec 5.1 "empty cells uniform") is set uniform."""
    col_sum = w.sum(axis=0, keepdims=True)
    empty = (col_sum[0] <= 0)
    out = np.divide(w, col_sum, out=np.zeros_like(w), where=~empty[None, :])
    if empty.any():
        out[:, empty] = 1.0 / w.shape[0]
    return out


def _broadcast(vec, n_b):
    """A per-template value that is the same at every brightness cell
    (spec: several factors are brightness-independent) as `(n_model, n_b)`.
    """
    return np.repeat(vec[:, None], n_b, axis=1)


def _floor_normalised(w):
    """A normalised factor floored at `FACTOR_FLOOR` of its own cell
    maximum and renormalised (spec sec 2, "the floor"): the fraction of
    cells x templates that sat at zero before the floor is also
    returned, float64 throughout (the cast to float32 is the writer's
    job, spec sec 9's bar applies here)."""
    cell_max = w.max(axis=0, keepdims=True)
    frac_zero = float(np.mean(w <= 0.0))
    floored = np.maximum(w, FACTOR_FLOOR * cell_max)
    return floored / floored.sum(axis=0, keepdims=True), frac_zero


def _floor_probability(p):
    """A probability factor (not renormalised: sec 1.4) floored at
    `FACTOR_FLOOR` and capped at `PROB_CAP` (spec sec 2); the fraction
    of entries that sat at zero before the floor is also returned."""
    frac_zero = float(np.mean(p <= 0.0))
    return np.clip(p, FACTOR_FLOOR, PROB_CAP), frac_zero


# ---------------------------------------------------------------------------
# P5 writer
# ---------------------------------------------------------------------------

def _write_library(config, lib, granule, names, c_theta, log10_b_centers, factors,
                    region=None, extra_attrs=None):
    """One `bmstp/weights/<lib>_weights_<granule>[__R].hdf5` file (P5):
    `MODEL_NAME`, `C_THETA`, `LOG10_B_CENTERS`, one `factor_<k>/` group per
    entry of `factors` (`name -> (W, C_F, D_F, normalised, source)`)."""
    path = config_module.product_path(config, "bmstp", "weights", lib, granule, region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = granule
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
        f.create_dataset("MODEL_NAME", data=np.char.encode(names, "utf-8"))
        f.create_dataset("C_THETA", data=c_theta.astype(np.float64))
        f.create_dataset("LOG10_B_CENTERS", data=log10_b_centers.astype(np.float64))
        for i, (name, (w, c_f, d_f, normalised, source)) in enumerate(factors.items()):
            grp = f.create_group(f"factor_{i}")
            grp.create_dataset("W", data=w.astype(np.float32))
            grp.create_dataset("C_F", data=c_f.astype(np.float64))
            grp.attrs["NAME"] = name
            grp.attrs["D_F"] = d_f
            grp.attrs["NORMALISED"] = bool(normalised)
            grp.attrs["SOURCE"] = source
    return path


# ---------------------------------------------------------------------------
# the contamination curve reader, shared by pahc `contrast` and sps
# `uncontaminated` (spec sec 5.1, sec 5.3: "read exactly as ... below")
# ---------------------------------------------------------------------------

def _read_pahc_curve(config):
    path = config_module.product_path(config, "population", "pahc", "curve", "survey")
    with h5py.File(path, "r") as f:
        edges = f["LOG10_Q_EDGES"][:].astype(np.float64)
        p_q = f["P_Q"][:].astype(np.float64)
    return 0.5 * (edges[:-1] + edges[1:]), p_q


def _p_at_neg_log10_q(neg_log10_q_query, centers, p_q):
    """`P(q)` at `-log10 q = neg_log10_q_query` (spec sec 5.3): linear
    interpolation on the curve's own bin centers; outside the tabulated
    `log10 q` range `np.interp`'s own clamping holds the nearest
    measured bin's value at each end (constant extrapolation, never the
    shelf -- the shelf subtraction is the curve product's own business,
    not repeated here)."""
    log10_q = -neg_log10_q_query
    return np.interp(log10_q, centers, p_q)


def _pahc_contrast_row(config, n_b_centers):
    """The one row `P(-log10 q = LOG10_B_CENTERS[k])`, the same for every
    template (spec sec 5.3: template enters only through `C_F`)."""
    centers, p_q = _read_pahc_curve(config)
    return _p_at_neg_log10_q(n_b_centers, centers, p_q)


# ---------------------------------------------------------------------------
# YSO: imf, inclination (survey, sec 5.5)
# ---------------------------------------------------------------------------

def _read_yso_subgrid_inclination(config, subdir):
    path = f"{config.inputs['sed_models']}/yso/{subdir}/parameters.fits"
    with fits.open(path) as hdul:
        d = hdul[1].data
        names = np.char.strip(d["MODEL_NAME"].astype(str))
        incl_deg = d["inclination"].astype(np.float64)
    return names, incl_deg


def _chabrier_dn_dlogm(m_star):
    """`dN/dlog10 M` (unnormalised: the per-template weight only needs to
    be proportional, since every consumer normalises over templates,
    sec 5.5, sec 10 item 1): the Chabrier 2003 lognormal below 1 Msun,
    continuous onto the `M^-CHABRIER_SLOPE` power law above it."""
    log_m = np.log10(m_star)
    lognormal = np.exp(-(log_m - CHABRIER_LOG_MC) ** 2 / (2.0 * CHABRIER_SIGMA_DEX ** 2))
    c_join = np.exp(-(0.0 - CHABRIER_LOG_MC) ** 2 / (2.0 * CHABRIER_SIGMA_DEX ** 2))
    powerlaw = c_join * np.power(m_star, -CHABRIER_SLOPE)
    return np.where(m_star <= 1.0, lognormal, powerlaw)


def build_yso(config):
    with progress.Stage("bmstp.template_weights.yso") as st:
        reg = _read_register(config, "yso")
        names, rho = reg["names"], reg["rho"]
        n_model = names.size

        mass_path = config_module.product_path(config, "population", "yso", "mass", "survey")
        if not os.path.isfile(mass_path):
            raise FileNotFoundError(
                f"template_weights.yso: missing {mass_path}; run "
                "sesnaimpute.population.yso_mass first")
        with h5py.File(mass_path, "r") as f:
            mass_names = np.char.decode(f["MODEL_NAME"][:].astype("S"), "utf-8")
            m_star = f["M_STAR"][:].astype(np.float64)
        n_matched_mass = int(np.sum(mass_names == names)) if mass_names.size == n_model else 0
        if n_matched_mass != n_model:
            raise ValueError(
                f"template_weights.yso: mass table join n_matched={n_matched_mass} "
                f"!= n_register={n_model}")

        incl_names, incl_deg = [], []
        for subdir, _label in YSO_SUBGRIDS:
            n, i = _read_yso_subgrid_inclination(config, subdir)
            incl_names.append(n)
            incl_deg.append(i)
        incl_names = np.concatenate(incl_names)
        incl_deg = np.concatenate(incl_deg)
        n_matched_incl = int(np.sum(incl_names == names)) if incl_names.size == n_model else 0
        if n_matched_incl != n_model:
            raise ValueError(
                f"template_weights.yso: inclination join n_matched={n_matched_incl} "
                f"!= n_register={n_model}")

        log10_b_centers = _log10_b_centers(_LOG10_B_ORIGIN_YSO)
        n_b = log10_b_centers.size

        # imf and inclination share the plain argument log10 B (no C_F,
        # no D_F): one stored factor, divided once by rho as the
        # register stores it (spec sec 1.4: no sub-grid subdivides the
        # library's own density here -- the subclass posterior is the
        # only place the YSO set is subdivided, sec 5.5).
        psi = _chabrier_dn_dlogm(m_star)         # imf: dN/dlog10 M, Chabrier 2003
        incl_raw = np.sin(np.radians(incl_deg))  # uniform in cos i (spec sec 5.5)
        shape_raw = psi * incl_raw / rho

        population_w = _broadcast(shape_raw, n_b)
        population_w, frac_zero = _floor_normalised(population_w)

        factors = {
            "population": (population_w, np.zeros(n_model), "", True,
                            "population.yso_mass Chabrier 2003; yso sub-grid "
                            "parameters.fits inclination"),
        }
        c_theta = np.zeros(n_model)
        path = _write_library(config, "yso", "survey", names, c_theta, log10_b_centers, factors,
                               extra_attrs={"COMPONENTS": "imf,inclination"})
        col_sum = population_w.sum(axis=0)
        print(f"template_weights.yso: factor=population max|colsum-1|="
              f"{float(np.max(np.abs(col_sum - 1.0))):.3g} floored_fraction={frac_zero:.4f}",
              flush=True)

        # report only (spec sec 5.5's check): the share of the built
        # weight held by each register SUBCLASS value, a diagnostic of
        # the library's coverage, never used in the weight itself. The
        # factor carries no brightness dependence, so the share is the
        # same at every cell; reported once.
        subclass = reg["subclass"]
        for label in np.unique(subclass):
            share = float(population_w[subclass == label, 0].sum())
            print(f"template_weights.yso: subclass={label} weight_share={share:.4f}", flush=True)
        st.done(path, n_model=n_model, floored_fraction=frac_zero)


# ---------------------------------------------------------------------------
# GALZ: colour (survey, sec 5.4)
# ---------------------------------------------------------------------------

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _gal_b_origin(config):
    shape_path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
    if os.path.isfile(shape_path):
        with h5py.File(shape_path, "r") as f:
            return float(f.attrs["LOG10_B_ORIGIN"]), "bmstp/shape/gal_shape_survey.hdf5"
    counts_path = config_module.product_path(config, "population", "gal", "counts", "survey")
    with h5py.File(counts_path, "r") as f:
        origin = float(f["LOG10_S_GRID"][0])
    return origin, "population/gal/counts_gal_survey.hdf5 LOG10_S_GRID[0] (gal_shape_survey absent)"


def _node_kde(colour_theta_f32, c_gal, s_gal):
    """One node's `(dens, min_z)` over every template (spec sec 5.4): the
    Gaussian kernel density in `COLOUR_I1I2`, bandwidth `s_gal` per
    galaxy, evaluated at `colour_theta_f32`, and each template's distance
    in bandwidths to its nearest galaxy (for the beyond-3-bandwidths
    report). float32 throughout (a density, not a stored science number);
    one `(n_model, galaxy batch)` block at a time, batch sized to keep
    that block under 512 MB with its one same-shaped working array
    (rule 10b)."""
    n_model = colour_theta_f32.size
    n_gal = c_gal.size
    if n_gal == 0:
        # no galaxies at this S node: uniform density (normalised below);
        # every template counts as beyond three bandwidths from nothing.
        return np.ones(n_model, dtype=np.float64), np.zeros(n_model, dtype=np.float32)
    c_gal = c_gal.astype(np.float32)
    s_gal = s_gal.astype(np.float32)
    dens = np.zeros(n_model, dtype=np.float64)
    min_z = np.full(n_model, np.inf, dtype=np.float32)
    # two same-shaped float32 arrays (z, gauss) are alive at once, and up
    # to four nodes run concurrently (rule 10a): budget each block at
    # 512 MB / (2 arrays * 4 workers) so the whole stage stays in budget.
    batch_gal = max(1, int((512 << 20) // (4 * n_model * 2 * 4)))
    ct = colour_theta_f32[:, None]
    for g0 in range(0, n_gal, batch_gal):
        g1 = min(g0 + batch_gal, n_gal)
        sb = s_gal[None, g0:g1]
        z = (ct - c_gal[None, g0:g1]) / sb
        gauss = np.exp(-0.5 * z * z, dtype=np.float32)
        gauss /= (sb * np.float32(_SQRT_2PI))
        dens += gauss.sum(axis=1, dtype=np.float64)
        np.abs(z, out=z)
        np.minimum(min_z, z.min(axis=1), out=min_z)
    return dens / n_gal, min_z


def build_galz(config):
    with progress.Stage("bmstp.template_weights.galz") as st:
        reg = _read_register(config, "galz")
        names, rho = reg["names"], reg["rho"]
        n_model = names.size
        colour_theta = np.log10(reg["f_ref"]["I1"]) - np.log10(reg["f_ref"]["I2"])
        colour_theta_f32 = colour_theta.astype(np.float32)
        c_theta = np.log10(reg["f_ref"]["I2"])

        swire_path = config_module.product_path(config, "sky/derived", "swire", "galaxies", "survey")
        with h5py.File(swire_path, "r") as f:
            node = f["NODE"][:]
            colour = f["COLOUR_I1I2"][:].astype(np.float64)
            sigma = f["SIGMA_COLOUR_I1I2"][:].astype(np.float64)
            log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
        n_node = log10_s_grid.size
        finite = np.isfinite(colour) & np.isfinite(sigma) & (sigma > 0)

        origin, origin_source = _gal_b_origin(config)
        log10_b_centers = _log10_b_centers(origin)
        n_b = log10_b_centers.size

        # one (n_model x galaxy-batch) KDE block per node (sec 5.4); the
        # 61 nodes are independent, so they run on threads, capped at 4
        # workers (CODING_RULES.md rule 10a), each still batching its own
        # galaxies under the 512 MB block (rule 10b).
        node_gal = [(colour[finite & (node == k)], sigma[finite & (node == k)])
                    for k in range(n_node)]
        n_jobs = min(4, config.n_jobs)
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_node_kde)(colour_theta_f32, c_gal, s_gal) for c_gal, s_gal in node_gal)
        node_density = np.empty((n_node, n_model), dtype=np.float64)
        beyond3_fraction = np.empty(n_node, dtype=np.float64)
        for k, (dens, min_z) in enumerate(results):
            node_density[k] = dens
            beyond3_fraction[k] = 1.0 if node_gal[k][0].size == 0 else float(np.mean(min_z > 3.0))
            st.tick(k + 1, n_node, "S nodes")

        node_w = _normalise_over_theta((node_density / rho[None, :]).T).T  # (n_node, n_model)

        # interpolate the per-node, per-template normalised density onto
        # log10_b_centers (already origin + 0.1(k+1/2), sec 2 -- no
        # second origin add), vectorised over every template at once: a
        # linear combination of the two bracketing nodes' own
        # theta-normalised vectors, clamped at the node edges (which
        # keeps the per-cell sum at 1, sec 5.4's "normalised over theta
        # at each S": a convex combination of two vectors that each
        # already sum to 1 sums to 1 too).
        s_query = np.clip(log10_b_centers, log10_s_grid[0], log10_s_grid[-1])
        hi = np.clip(np.searchsorted(log10_s_grid, s_query), 1, n_node - 1)
        lo = hi - 1
        frac = (s_query - log10_s_grid[lo]) / (log10_s_grid[hi] - log10_s_grid[lo])
        w = ((1.0 - frac)[:, None] * node_w[lo] + frac[:, None] * node_w[hi]).T  # (n_model, n_b)
        w, frac_zero = _floor_normalised(w)

        # the varying-cell range (brief item 1's check): a cell whose
        # s_query clips to a grid edge repeats that edge node's
        # per-template vector exactly, so it is identical across the
        # brightness axis to the first (or last) cell; the varying span
        # is the cells that differ from both edges.
        same_lo = np.all(w == w[:, [0]], axis=0)
        same_hi = np.all(w == w[:, [-1]], axis=0)
        varying = np.where(~(same_lo | same_hi))[0]
        varying_range = (int(varying.min()), int(varying.max())) if varying.size else (-1, -1)

        factors = {
            "colour": (w, c_theta, "", True,
                       "sky/derived/swire/galaxies_swire_survey.hdf5 COLOUR_I1I2 KDE"),
        }
        path = _write_library(config, "galz", "survey", names, c_theta, log10_b_centers, factors)
        col_sum = w.sum(axis=0)
        print(f"template_weights.galz: origin={origin:.4f} ({origin_source}); "
              f"max|colsum-1|={float(np.max(np.abs(col_sum - 1.0))):.3g}; "
              f"floored_fraction={frac_zero:.4f}; varying cells={varying_range[0]}-{varying_range[1]}; "
              "beyond-3-bandwidths fraction per node: "
              + ",".join(f"{v:.3f}" for v in beyond3_fraction), flush=True)
        st.done(path, n_model=n_model, n_node=n_node,
                 beyond3_mean=float(np.mean(beyond3_fraction)), floored_fraction=frac_zero)
    return beyond3_fraction


# ---------------------------------------------------------------------------
# SPS: type, uncontaminated (per region, sec 5.1)
# ---------------------------------------------------------------------------

def _region_star_stars(config, region):
    """`(template_index, weight, log10_b)` over every retained field star
    of `region`'s tiles (`population.star_population`'s `W_STAR`, `LOG10_B`
    per tile; `population.field_stars`'s `TEMPLATE_INDEX` per retained
    star, joined by `STAR_INDEX`)."""
    field_path = config_module.product_path(
        config, "population", "trilegal", "field-stars", "region", region=region)
    tile_path = config_module.product_path(
        config, "population", "star", "population", "tile", region=region)
    with h5py.File(field_path, "r") as f:
        template_index = f["TEMPLATE_INDEX"][:]
    template_idx_all, weight_all, log10_b_all = [], [], []
    with h5py.File(tile_path, "r") as f:
        for key in f:
            if not key.startswith("tile_"):
                continue
            grp = f[key]
            si = grp["STAR_INDEX"][:]
            template_idx_all.append(template_index[si])
            weight_all.append(grp["W_STAR"][:].astype(np.float64))
            log10_b_all.append(grp["LOG10_B"][:].astype(np.float64))
    return (np.concatenate(template_idx_all), np.concatenate(weight_all),
            np.concatenate(log10_b_all))


def _weighted_type_histogram(template_idx, weight, log10_b, n_model, origin):
    edges = grid.log10_b_edges(origin)
    idx_b = np.clip(np.digitize(log10_b, edges) - 1, 0, edges.size - 2)
    h = np.zeros((n_model, edges.size - 1), dtype=np.float64)
    np.add.at(h, (template_idx, idx_b), weight)
    return h


def _sps_raw_type_histogram(config, region):
    """`(names, rho, h, weight)`: the sps register's own `MODEL_NAME`/
    `RHO_KDE1` and the region's raw, weighted type histogram (the
    population count per sps template per brightness cell, BEFORE the
    `1/rho` division and the per-cell normalisation, spec sec 5.1) --
    shared by `build_sps` (which normalises it) and `build_pahc` (which
    reads it at each PAHC template's matched sps row, owner ruling)."""
    reg = _read_register(config, "sps")
    names, rho = reg["names"], reg["rho"]
    template_idx, weight, log10_b = _region_star_stars(config, region)
    h = _weighted_type_histogram(template_idx, weight, log10_b, names.size, _LOG10_B_ORIGIN_STAR)
    return names, rho, h, weight


def build_sps(config, region):
    with progress.Stage("bmstp.template_weights.sps", region) as st:
        reg = _read_register(config, "sps")
        names, rho = reg["names"], reg["rho"]
        n_model = names.size
        log10_b_centers = _log10_b_centers(_LOG10_B_ORIGIN_STAR)

        _names_check, _rho_check, h, weight = _sps_raw_type_histogram(config, region)
        type_w = _normalise_over_theta(h / rho[:, None])
        type_w, frac_zero_type = _floor_normalised(type_w)

        n_b_row = _pahc_contrast_row(config, log10_b_centers)
        unc_raw = 1.0 - np.repeat(n_b_row[None, :], n_model, axis=0)  # 1 - P(q), sec 5.1
        unc_raw, frac_zero_unc = _floor_probability(unc_raw)
        c_f_unc = np.log10(reg["f_ref"]["I4"])  # the sps template's own 8um photosphere

        factors = {
            "type": (type_w, np.zeros(n_model), "", True,
                     f"population.star_population {region} W_STAR/LOG10_B"),
            "uncontaminated": (unc_raw, c_f_unc, "D_PAHC", False,
                                "population.pahc.curve_pahc_survey P_Q (1 - P(q))"),
        }
        c_theta = np.zeros(n_model)
        path = _write_library(config, "sps", "region", names, c_theta, log10_b_centers, factors,
                               region=region)
        retained_weighted = float(weight.sum())
        histogram_weighted = float(h.sum())
        col_sum = type_w.sum(axis=0)
        _curve_centers, curve_p_q = _read_pahc_curve(config)
        print(f"template_weights.sps [{region}]: retained weighted count={retained_weighted:.4f} "
              f"histogram sum={histogram_weighted:.4f} "
              f"max|colsum-1|={float(np.max(np.abs(col_sum - 1.0))):.3g} "
              f"floored_fraction type={frac_zero_type:.4f} uncontaminated={frac_zero_unc:.4f}; "
              f"P(q) at axis ends={float(n_b_row[0]):.4f}/{float(n_b_row[-1]):.4f} "
              f"vs curve end bins={float(curve_p_q[0]):.4f}/{float(curve_p_q[-1]):.4f}",
              flush=True)
        st.done(path, n_model=n_model, n_star=weight.size,
                 retained_weighted=retained_weighted)


# ---------------------------------------------------------------------------
# AGB: tau (per region, but brightness- and thus region-independent, sec
# 5.2: "the same weight at every B cell")
# ---------------------------------------------------------------------------

def build_agb(config, region):
    with progress.Stage("bmstp.template_weights.agb", region) as st:
        reg = _read_register(config, "agb")
        names, rho = reg["names"], reg["rho"]
        n_model = names.size

        grid_path = f"{config.inputs['sed_models']}/agb/parameters.fits"
        with fits.open(grid_path) as hdul:
            d = hdul[1].data
            grid_names = np.char.strip(d["MODEL_NAME"].astype(str))
            tau = d["TAU"].astype(np.float64)
            chem = np.char.strip(d["CHEM"].astype(str))
        n_matched = int(np.sum(grid_names == names)) if grid_names.size == n_model else 0
        if n_matched != n_model:
            raise ValueError(f"template_weights.agb: join n_matched={n_matched} "
                              f"!= n_register={n_model}")

        gcl, riebel_tau = star_population.read_riebel_optical_depths(config)
        log10_tau_o = np.log10(riebel_tau[gcl == "o"])
        log10_tau_c = np.log10(riebel_tau[gcl == "c"])
        edges = np.linspace(-3.0, 2.0, 61)
        h_o, _ = np.histogram(log10_tau_o, bins=edges, density=True)
        h_c, _ = np.histogram(log10_tau_c, bins=edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        log10_tau_theta = np.log10(tau)
        p_o = np.interp(log10_tau_theta, centers, h_o, left=0.0, right=0.0)
        p_c = np.interp(log10_tau_theta, centers, h_c, left=0.0, right=0.0)
        p_mix = np.where(chem == "O", (1.0 - F_C) * p_o, F_C * p_c)

        log10_b_centers = _log10_b_centers(_LOG10_B_ORIGIN_STAR)
        n_b = log10_b_centers.size
        tau_raw = p_mix / rho
        tau_w = _normalise_over_theta(_broadcast(tau_raw, n_b))
        tau_w, frac_zero = _floor_normalised(tau_w)

        factors = {
            "tau": (tau_w, np.zeros(n_model), "", True,
                    "sky.download.riebel2012 table3.dat.gz TAU by CHEM"),
        }
        c_theta = np.zeros(n_model)
        path = _write_library(config, "agb", "region", names, c_theta, log10_b_centers, factors,
                               region=region)
        col_sum = tau_w.sum(axis=0)
        print(f"template_weights.agb [{region}]: n_riebel_o={log10_tau_o.size} "
              f"n_riebel_c={log10_tau_c.size} max|colsum-1|="
              f"{float(np.max(np.abs(col_sum - 1.0))):.3g} floored_fraction={frac_zero:.4f}",
              flush=True)
        st.done(path, n_model=n_model, floored_fraction=frac_zero)


# ---------------------------------------------------------------------------
# PAHC: type, contrast (per region, sec 5.3; the `type` match is an owner
# ruling, PAHC's aperture library carrying no atmosphere axis of its own)
# ---------------------------------------------------------------------------

def _read_teff_logg(path, logg_col):
    with fits.open(path) as hdul:
        d = hdul[1].data
        names = np.char.strip(d["MODEL_NAME"].astype(str))
        teff = d["T_EFF"].astype(np.float64)
        logg = d[logg_col].astype(np.float64)
    return names, teff, logg


def _match_pahc_to_sps(config, sps_names):
    """`(sps_index, distance)`, one row per PAHC register template, in
    the PAHC register's own row order: the nearest sps atmosphere
    template in `(log10 T_EFF, LOGG)` (Euclidean, `cKDTree`, both
    libraries' own `parameters.fits`; owner ruling -- PAHC's SED shape
    IS the star family's, spec sec 5.3, but its library has no
    atmosphere-type axis, so its `type` weight borrows the nearest sps
    template's). Row order is checked against each register's own
    `MODEL_NAME`, not assumed."""
    from scipy.spatial import cKDTree

    sps_path = f"{config.inputs['sed_models']}/sps/parameters.fits"
    pahc_path = f"{config.inputs['sed_models']}/pahc/parameters.fits"
    sps_p_names, sps_teff, sps_logg = _read_teff_logg(sps_path, "LOG[G]")
    pahc_p_names, pahc_teff, pahc_logg = _read_teff_logg(pahc_path, "LOGG")

    reg_pahc_names = _read_register(config, "pahc")["names"]
    if not (sps_p_names.size == sps_names.size and np.array_equal(sps_p_names, sps_names)):
        raise ValueError("template_weights.pahc: sps parameters.fits row order disagrees "
                          "with the sps register")
    if not (pahc_p_names.size == reg_pahc_names.size
            and np.array_equal(pahc_p_names, reg_pahc_names)):
        raise ValueError("template_weights.pahc: pahc parameters.fits row order disagrees "
                          "with the pahc register")

    sps_points = np.column_stack([np.log10(sps_teff), sps_logg])
    pahc_points = np.column_stack([np.log10(pahc_teff), pahc_logg])
    dist, idx = cKDTree(sps_points).query(pahc_points)
    return idx, dist


def build_pahc(config, region):
    with progress.Stage("bmstp.template_weights.pahc", region) as st:
        reg = _read_register(config, "pahc")
        names, rho = reg["names"], reg["rho"]
        n_model = names.size
        log10_b_centers = _log10_b_centers(_LOG10_B_ORIGIN_STAR)

        sps_names, sps_rho, h_sps, _weight = _sps_raw_type_histogram(config, region)
        sps_idx, dist = _match_pahc_to_sps(config, sps_names)
        # the region's raw STAR type histogram (the population count, not
        # yet divided by rho or normalised) at each PAHC template's
        # matched sps row, then PAHC's own 1/rho and per-cell
        # normalisation (owner ruling: a proper weight over the PAHC
        # library, not a copy of the sps one).
        raw_type = h_sps[sps_idx, :] / rho[:, None]
        type_w = _normalise_over_theta(raw_type)
        type_w, frac_zero_type = _floor_normalised(type_w)

        row = _pahc_contrast_row(config, log10_b_centers)
        contrast_w = np.repeat(row[None, :], n_model, axis=0)
        contrast_w, frac_zero_contrast = _floor_probability(contrast_w)
        c_f = np.log10(reg["f_ref"]["I4"])  # +log10 f_ref,8,theta (spec sec 5.3)

        factors = {
            "type": (type_w, np.zeros(n_model), "", True,
                     f"population.star_population {region} W_STAR/LOG10_B, "
                     "nearest sps template in (log10 T_EFF, LOGG)"),
            "contrast": (contrast_w, c_f, "D_PAHC", False,
                         "population.pahc.curve_pahc_survey P_Q"),
        }
        c_theta = np.zeros(n_model)
        path = _write_library(config, "pahc", "region", names, c_theta, log10_b_centers, factors,
                               region=region)
        col_sum = type_w.sum(axis=0)
        n_matched_sps_used = int(np.unique(sps_idx).size)
        _curve_centers, curve_p_q = _read_pahc_curve(config)
        print(f"template_weights.pahc [{region}]: match n=median {float(np.median(dist)):.4f} "
              f"max {float(dist.max()):.4f} (log10 T_EFF, LOGG); "
              f"{n_matched_sps_used}/{sps_names.size} sps templates ever matched; "
              f"max|colsum-1|={float(np.max(np.abs(col_sum - 1.0))):.3g} "
              f"floored_fraction type={frac_zero_type:.4f} contrast={frac_zero_contrast:.4f}; "
              f"contrast at axis ends={float(row[0]):.4f}/{float(row[-1]):.4f} "
              f"vs curve end bins={float(curve_p_q[0]):.4f}/{float(curve_p_q[-1]):.4f}", flush=True)
        st.done(path, n_model=n_model, n_matched_sps_used=n_matched_sps_used)


# ---------------------------------------------------------------------------
# H2SHOCK: uniform (survey, sec 5.6)
# ---------------------------------------------------------------------------

def build_h2shock(config):
    with progress.Stage("bmstp.template_weights.h2shock") as st:
        reg = _read_register(config, "h2shock")
        names = reg["names"]
        n_model = names.size
        log10_b_centers = _log10_b_centers(_LOG10_B_ORIGIN_STAR)

        grid_path = f"{config.inputs['sed_models']}/h2shock/parameters.fits"
        with fits.open(grid_path) as hdul:
            cols = hdul[1].columns.names
            d = hdul[1].data
            grid_names = np.char.strip(d["MODEL_NAME"].astype(str))
            if "I_H2_1_0_S1" in cols:
                # the H2 1-0 S(1) surface brightness, erg/s/cm2/sr, joined
                # on MODEL_NAME (spec sec 3.5, sec 5.6): the Ks stand-in
                # is dropped now that the shock grid carries this column.
                i_ref = d["I_H2_1_0_S1"].astype(np.float64)
                c_theta_source = "I_H2_1_0_S1, parameters.fits"
                n_matched = int(np.sum(grid_names == names)) if grid_names.size == n_model else 0
            else:
                i_ref = reg["f_ref"]["Ks"]  # stand-in until the line column exists (spec sec 3.5)
                c_theta_source = "Ks reference flux (stand-in)"
                n_matched = n_model  # no per-template join needed for this branch
        if n_matched != n_model:
            raise ValueError(f"template_weights.h2shock: join n_matched={n_matched} "
                              f"!= n_register={n_model}")

        w = np.full((n_model, log10_b_centers.size), 1.0 / n_model, dtype=np.float64)
        c_theta = np.log10(i_ref)

        factors = {
            "uniform": (w, np.zeros(n_model), "", True, "no external distribution (spec sec 5.6)"),
        }
        path = _write_library(config, "h2shock", "survey", names, c_theta, log10_b_centers, factors,
                               extra_attrs={"C_THETA_SOURCE": c_theta_source})
        st.done(path, n_model=n_model)
        print(f"template_weights.h2shock: C_THETA_SOURCE={c_theta_source} "
              f"log10(I_H2_1_0_S1) range min={c_theta.min():.4f} "
              f"median={np.median(c_theta):.4f} max={c_theta.max():.4f}", flush=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Survey products once (galz, yso, h2shock); sps, pahc and agb once
    per region in `regions` (default: all of `regions.REGIONS`, rule 5c).
    PAHC is per region (not survey) because its `type` factor borrows the
    region's own sps type histogram (owner ruling)."""
    region_list = regions if regions else [r.name for r in regions_module.REGIONS]

    build_yso(config)
    build_galz(config)
    build_h2shock(config)
    for region in region_list:
        field_path = config_module.product_path(
            config, "population", "trilegal", "field-stars", "region", region=region)
        if not os.path.isfile(field_path):
            print(f"template_weights: skipping {region}, no population field-star product "
                  f"({field_path})", flush=True)
            continue
        build_sps(config, region)
        build_pahc(config, region)
        build_agb(config, region)


if __name__ == "__main__":
    run(build)
