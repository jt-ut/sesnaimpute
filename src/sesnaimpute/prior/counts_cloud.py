"""The per-source YSO and H2S counts on the cloud law (SPEC_PRIORS.md
sections 6.2, 6.4 item 1, and 7; IMPLEMENTATION.md section 5): the
"counts" half of the prior table's per-source join for the two classes
that ride the young-star law count, `prior.yso.law_count`.

For every catalogued source `s`, this module reads the region's own
per-source tabulated pieces -- the YSO mass-based selection `g(a)`
(`prior.yso_selection`) and the H2S selection `eps(a, Sigma)`
(`prior.h2s`), both exact per source on the shared scaled-extinction
ladder `X_LADDER` (`a = X_LADDER * A_s`, IMPLEMENTATION.md section 4: no
depth groups), and the sightline's own exact extinction marginal
`p(a | A_s)` (`prior.yso.YsoShape`, the log-normal closed form) -- and
combines them:

    EPS_YSO(s)      = Integral da g(a) p(a | A_s)
    N_YSO(s)        = N_law(s) * EPS_YSO(s)
    EPS_H2S(s)      = Integral da dlog10(Sigma) eps(a, Sigma) p(a | A_s) p_r(log10 Sigma)
    N_H2S(s)        = N_LAW_BLURRED(s) * ETA_r * EPS_EXT * EPS_H2S(s)

Both integrals are exact in the extinction axis: `p(a | A_s)` integrates
in closed form (`YsoShape.cdf_exact`), so its mass in each of the
source's own `X_LADDER * A_s` bins is an exact difference of two
`cdf_exact` calls at that source's own adopted column and measurement
uncertainty; the selection's own tabulated curve is read at each bin's
midpoint, weighted by that bin's exact mass, and summed. The `log10
Sigma` axis is a plain quadrature on the region's own 41-point grid, as
`prior.h2s.source_pass_fraction` already does for its own (point-
evaluated, not marginal-integrated) report.

No selection lives inside the YSO shape (section 6.2: `Z_YSO = 1`); H2S's
own selection is the count's own normaliser (section 7: `Z_H2S =
EPS_H2S`), since nothing else in this module's `Lambda_H2S` still needs
renormalising against it.

Sources sharing a sightline do NOT share one column: `A_COL_K` is a
per-source adopted value, and both the closed-form marginal and the
selection tables read it exactly, per source -- there is no shared grid
and nothing to deduplicate, so `bin_mass_exact` is a plain vectorised
`(n_source, n_x)` closed-form evaluation, batched over sources
(`sesnaimpute.batches.batches`) so one batch's transient array stays
bounded regardless of a region's own source count (CODING_RULES.md rule
10b).

The `NODE_LO`/`NODE_W` bracket (`prior.column_grid.bracket`) is still
computed and written to the output product, since other classes' node
blend (IMPLEMENTATION.md section 2) reads it from the prior table -- it
plays no part in this module's own extinction integral, which uses each
source's exact column directly.

The YSO and H2S selection products are both built on `prior.selection.
X_LADDER`, the one scaled-extinction ladder every class's exact
selection shares (IMPLEMENTATION.md section 4); this module checks the
two copies agree before combining them.
"""

import os

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid as column_grid_module
from sesnaimpute.prior import yso as yso_module

#: Per-batch memory budget for `bin_mass_exact`'s closed-form evaluation
#: (CODING_RULES.md rule 10b): well under the 8 GB ceiling regardless of
#: a region's own source count.
BIN_MASS_BUDGET_BYTES = 256 << 20

#: The anchor pixelisation the section 6.4 item 1 check integrates over
#: (`prior.yso.NSIDE_ANCHOR`, the same nside `prior.young_stars` uses for
#: "young stars in the anchors").
_OMEGA_PIX512_DEG2 = hp.nside2pixarea(yso_module.NSIDE_ANCHOR, degrees=True)


# ---------------------------------------------------------------------
# per-region reads
# ---------------------------------------------------------------------

def _adopted_columns(config, region):
    """`(a_col, sigma_col, provenance)`, every source of `region`, in
    catalogue row order (the same adopted-column product
    `prior.yso.law_count` reads)."""
    path = config_module.product_path(config, "sky/derived", "adopted",
                                       "column", "source", region=region)
    cols = access.per_source(config, region, path,
                              ["A_COL_K", "A_COL_SIG_K", "A_COL_PROVENANCE"])
    return (np.asarray(cols["A_COL_K"], dtype=float),
            np.asarray(cols["A_COL_SIG_K"], dtype=float),
            np.asarray(cols["A_COL_PROVENANCE"]))


def _yso_selection_tables(config, region):
    """`X_LADDER`, `G_1MYR`, `G_3MYR`, `M_LIM_8UM_1MYR`, each exact per
    source (`prior.yso_selection.build_and_write_region`'s own
    per-source product)."""
    path = config_module.product_path(config, "bms", "yso", "selection", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.counts_cloud: no YSO selection product for region %r at %s "
            "-- run the 'prior.yso_selection' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(x_ladder=np.asarray(f["X_LADDER"][:], dtype=np.float64),
                    g_1myr=np.asarray(f["G_1MYR"][:], dtype=np.float64),
                    g_3myr=np.asarray(f["G_3MYR"][:], dtype=np.float64),
                    m_lim_8um_1myr=np.asarray(f["M_LIM_8UM_1MYR"][:], dtype=np.float64))


def _h2s_selection_source(config, region):
    """`X_LADDER`, `EPS[n, n_x, n_sigma]` (`prior.h2s.
    build_and_write_source_selection`'s own per-source product)."""
    path = config_module.product_path(config, "bms", "h2s", "selection", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.counts_cloud: no H2S selection product for region %r at %s "
            "-- run the 'prior.h2s' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(x_ladder=np.asarray(f["X_LADDER"][:], dtype=np.float64),
                    eps=np.asarray(f["EPS"][:], dtype=np.float64))


def _h2s_region_product(config, region):
    """`ETA`, `EPS_EXT`, the brightness lognormal and its own `log10
    Sigma` grid (`prior.h2s.build`'s own per-region product)."""
    path = config_module.product_path(config, "bms", "h2s", "prior", "region", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.counts_cloud: no H2S region product for region %r at %s "
            "-- run the 'prior.h2s' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(
            eta=float(f["ETA"][()]), eps_ext=float(f["EPS_EXT"][()]),
            logsig_mean=float(f["LOGSIG_MEAN"][()]), logsig_std=float(f["LOGSIG_STD"][()]),
            log10_sigma_grid=np.asarray(f["LOG10_SIGMA_GRID"][:], dtype=np.float64),
        )


def _n_law_blurred(config, region):
    """`N_LAW_BLURRED_DEG2`, every source (`prior.h2s.build_law_blurred`'s
    own per-source product)."""
    path = config_module.product_path(config, "bms", "h2s", "law-blurred", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.counts_cloud: no H2S law-blurred product for region %r at %s "
            "-- run the 'prior.h2s' RUNBOOK line first" % (region, path))
    cols = access.per_source(config, region, path, ["N_LAW_BLURRED_DEG2"])
    return np.asarray(cols["N_LAW_BLURRED_DEG2"], dtype=np.float64)


def _ridge(config, region):
    """`RIDGE_INTERCEPT`/`RIDGE_SLOPE`/`RIDGE_WIDTH`, one row per
    sightline (`prior.yso.build_shape`'s own product; `YsoShape.read`
    does not carry these, so they are read directly here)."""
    path = config_module.product_path(config, "bms", "yso", "prior", "sightline", region=region)
    with h5py.File(path, "r") as f:
        return dict(intercept=np.asarray(f["RIDGE_INTERCEPT"][:], dtype=np.float64),
                    slope=np.asarray(f["RIDGE_SLOPE"][:], dtype=np.float64),
                    width=np.asarray(f["RIDGE_WIDTH"][:], dtype=np.float64))


# ---------------------------------------------------------------------
# the exact extinction-axis integral: closed-form cdf differences at
# each source's own column, batched over sources
# ---------------------------------------------------------------------

def bin_mass_exact(shape, rows, a_col, sigma_col, x_ladder):
    """`(n_src, n_x - 1)`: the exact mass `p(a | A_s)` places in each of
    the source's own `X_LADDER * A_s` bins (SPEC_PRIORS.md sections
    6.2/7; IMPLEMENTATION.md section 4's shared scaled-extinction
    ladder) -- the closed-form cdf (`YsoShape.cdf_exact`) at each
    source's own adopted column and measurement uncertainty, evaluated
    at every ladder point (scaled by that source's own column) and
    differenced, batched over sources so one batch's transient `(batch,
    n_x)` array stays under `BIN_MASS_BUDGET_BYTES`."""
    n_src = rows.size
    n_x = x_ladder.size
    out = np.empty((n_src, n_x - 1), dtype=np.float64)
    row_bytes = 2 * n_x * 8  # the (batch, n_x) cdf array, float64
    for start, stop in batches_module.batches(n_src, row_bytes, BIN_MASS_BUDGET_BYTES):
        nb = stop - start
        rows_rep = np.repeat(rows[start:stop], n_x)
        a_col_rep = np.repeat(a_col[start:stop], n_x)
        sigma_rep = np.repeat(sigma_col[start:stop], n_x)
        a_rep = np.tile(x_ladder, nb) * a_col_rep
        cdf = shape.cdf_exact(a_rep, rows_rep, a_col_rep, sigma_rep).reshape(nb, n_x)
        out[start:stop] = np.diff(cdf, axis=1)
    return out


def eps_yso_from_bin_mass(bin_mass, g):
    """`Integral da g(a) p(a | A_s)`, per source (SPEC_PRIORS.md section
    6.2): the source's own tabulated curve `g[n, :]` read at each bin's
    midpoint, weighted by that bin's own exact marginal mass, summed."""
    g_mid = 0.5 * (g[:, :-1] + g[:, 1:])
    return np.sum(bin_mass * g_mid, axis=1)


def eps_h2s_from_bin_mass(bin_mass, eps_h2s, log10_sigma_grid, logsig_mean, logsig_std):
    """`Integral da dlog10(Sigma) eps(a, Sigma) p(a | A_s) p_r(log10
    Sigma)`, per source (SPEC_PRIORS.md section 7): the source's own
    `eps(a, Sigma)` table (`X_LADDER` already starts at `a = 0`, so no
    edge needs prepending), read at each `a`-bin's midpoint and weighted
    by that bin's own exact marginal mass -- a loop over the small,
    fixed `a`-bin axis (never over sources) -- then integrated over
    `log10 Sigma` by a plain quadrature on the region's own 41-point
    grid (as `prior.h2s.source_pass_fraction` already does for its own
    point-evaluated report)."""
    n_bins = bin_mass.shape[1]
    n_sigma = log10_sigma_grid.size
    eps_vs_sigma = np.zeros((bin_mass.shape[0], n_sigma), dtype=np.float64)
    for b in range(n_bins):
        eps_vs_sigma += bin_mass[:, b][:, None] * 0.5 * (eps_h2s[:, b, :] + eps_h2s[:, b + 1, :])

    density = np.exp(-0.5 * ((log10_sigma_grid - logsig_mean) / logsig_std) ** 2)
    density = density / np.trapz(density, log10_sigma_grid)
    return np.trapz(eps_vs_sigma * density[None, :], log10_sigma_grid, axis=1)


# ---------------------------------------------------------------------
# section 6.4 item 1 (report only): the law count over the region's own
# anchor pixels, beside Pokhrel+2020's Table 2 total where recorded
# ---------------------------------------------------------------------

def law_area_check(config, region):
    """`N_law` integrated over every occupied nside-512 pixel of `region`
    (SPEC_PRIORS.md section 6.4 item 1: "young stars in the whole region
    before selection") -- `prior.yso.law_area_integral`, times each equal-
    area pixel's own solid angle, summed."""
    rs = access.region_slice(config, region)
    pixels = np.unique(rs["hpx_pix_512"])
    per_pixel_deg2 = yso_module.law_area_integral(config, region, pixels)
    return float(np.sum(per_pixel_deg2) * _OMEGA_PIX512_DEG2)


# ---------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------

def build_region(config, region):
    """Computes every per-source dataset for one region (module
    docstring)."""
    rs = access.region_slice(config, region)
    n_src = rs["n_sources"]
    sightline_row = np.asarray(rs["hpx256_row"], dtype=np.intp)

    a_col, sigma_col, provenance = _adopted_columns(config, region)
    nodes = column_grid_module.nodes(config)
    node_lo, node_w = column_grid_module.bracket(a_col, nodes)

    n_law = yso_module.law_count(config, region, a_col, provenance)
    shape = yso_module.YsoShape.read(config, region)
    sel = _yso_selection_tables(config, region)
    h2s_sel = _h2s_selection_source(config, region)
    h2s = _h2s_region_product(config, region)
    n_law_blurred = _n_law_blurred(config, region)
    ridge = _ridge(config, region)

    if not np.array_equal(sel["x_ladder"], h2s_sel["x_ladder"]):
        raise ValueError(
            "prior.counts_cloud: %r's YSO and H2S X_LADDER disagree -- "
            "both are prior.selection.X_LADDER and should be identical" % region)
    x_ladder = sel["x_ladder"]

    bin_mass = bin_mass_exact(shape, sightline_row, a_col, sigma_col, x_ladder)
    eps_yso = eps_yso_from_bin_mass(bin_mass, sel["g_1myr"])
    eps_yso_3myr = eps_yso_from_bin_mass(bin_mass, sel["g_3myr"])
    eps_h2s = eps_h2s_from_bin_mass(bin_mass, h2s_sel["eps"],
                                    h2s["log10_sigma_grid"], h2s["logsig_mean"],
                                    h2s["logsig_std"])

    n_yso = n_law * eps_yso
    n_yso_3myr = n_law * eps_yso_3myr
    n_h2s = n_law_blurred * h2s["eta"] * h2s["eps_ext"] * eps_h2s

    return dict(
        sightline_row=sightline_row, node_lo=node_lo, node_w=node_w,
        n_yso=n_yso, n_yso_3myr=n_yso_3myr, eps_yso=eps_yso, eps_yso_3myr=eps_yso_3myr,
        n_law=n_law,
        ridge_intercept=ridge["intercept"][sightline_row],
        ridge_slope=ridge["slope"][sightline_row],
        ridge_width=ridge["width"][sightline_row],
        n_h2s=n_h2s, eps_h2s=eps_h2s,
        z_yso=np.ones(n_src, dtype=np.float64), z_h2s=eps_h2s,
        m_lim_8um_1myr=sel["m_lim_8um_1myr"], eta=h2s["eta"], eps_ext=h2s["eps_ext"],
    )


def _write_product(path, out):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    datasets = {
        "SIGHTLINE_ROW": out["sightline_row"], "NODE_LO": out["node_lo"],
        "NODE_W": out["node_w"],
        "N_YSO": out["n_yso"], "N_YSO_3MYR": out["n_yso_3myr"],
        "EPS_YSO": out["eps_yso"], "EPS_YSO_3MYR": out["eps_yso_3myr"],
        "N_LAW": out["n_law"], "M_LIM_8UM_1MYR": out["m_lim_8um_1myr"],
        "RIDGE_INTERCEPT": out["ridge_intercept"], "RIDGE_SLOPE": out["ridge_slope"],
        "RIDGE_WIDTH": out["ridge_width"],
        "N_H2S": out["n_h2s"], "EPS_H2S": out["eps_h2s"],
        "Z_YSO": out["z_yso"], "Z_H2S": out["z_h2s"],
    }
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.attrs["ETA"] = out["eta"]
        f.attrs["EPS_EXT"] = out["eps_ext"]
        f.attrs["KAPPA_HERSCHEL"] = yso_module.KAPPA_HERSCHEL
        f.attrs["KAPPA_PLANCK"] = yso_module.KAPPA_PLANCK
        for name, arr in datasets.items():
            f.create_dataset(name, data=np.asarray(arr, dtype=np.float32))


def build(config, regions=None):
    """Writes, per region, `bms/table/counts-cloud_table_source` (module
    docstring); prints the section 6.4 item 1 check (`law_area_check`)
    beside Pokhrel+2020's Table 2 total where the S-D38 study record
    carries it. `regions` default: all thirty."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    for region in region_names:
        out = build_region(config, region)
        path = config_module.product_path(config, "bms", "table", "counts-cloud",
                                          "source", region=region)
        _write_product(path, out)
        n_law_total = law_area_check(config, region)
        print("prior.counts_cloud: %s: %d sources, N_law region total=%.4g "
              "young stars (section 6.4 item 1, against Pokhrel+2020's Table 2 "
              "total per the S-D38 study record) -> %s"
              % (region, out["sightline_row"].size, n_law_total, path))


if __name__ == "__main__":
    run(build)
