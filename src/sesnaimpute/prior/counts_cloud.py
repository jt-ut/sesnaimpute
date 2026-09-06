"""The per-source YSO and H2S counts on the cloud law (SPEC_PRIORS.md
sections 6.2, 6.4 item 1, and 7; IMPLEMENTATION.md section 5): the
"counts" half of the prior table's per-source join for the two classes
that ride the young-star law count, `prior.yso.law_count`.

For every catalogued source `s`, this module reads the region's own
tabulated pieces -- the YSO mass-based selection `g_k(a)` per depth group
(`prior.yso_selection`), the H2S selection `eps_k(a, Sigma)` per depth
group (`prior.h2s`), and the sightline's own exact extinction marginal
`p(a | A_s)` (`prior.yso.YsoShape`, the log-normal closed form) -- and
combines them:

    EPS_YSO(s)      = Integral da g_k(a) p(a | A_s)
    N_YSO(s)        = N_law(s) * EPS_YSO(s)
    EPS_H2S(s)      = Integral da dlog10(Sigma) eps_k(a, Sigma) p(a | A_s) p_r(log10 Sigma)
    N_H2S(s)        = N_LAW_BLURRED(s) * ETA_r * EPS_EXT * EPS_H2S(s)

Both integrals are exact in the extinction axis: `p(a | A_s)` integrates
in closed form (`YsoShape.cdf_exact`), so its mass in any `a` bin is an
exact difference of two `cdf_exact` calls at each source's own adopted
column and measurement uncertainty; the group's own curve, tabulated
only at the shared grid's nodes, is read at each bin's midpoint by the
linear blend between its two bracketing nodes (the same
node-interpolation convention every class shares, IMPLEMENTATION.md
section 2). The `log10 Sigma` axis is a plain quadrature on the region's
own 41-point grid, as `prior.h2s.source_pass_fraction` already does for
its own (point-evaluated, not marginal-integrated) report.

No selection lives inside the YSO shape (section 6.2: `Z_YSO = 1`); H2S's
own selection is the count's own normaliser (section 7: `Z_H2S =
EPS_H2S`), since nothing else in this module's `Lambda_H2S` still needs
renormalising against it.

Sources sharing a sightline do NOT share one column: `A_COL_K` is a
per-source adopted value, and the closed form reads it exactly, not
through a shared node table -- there is no quadrature left to
deduplicate, so `bin_mass_exact` is a plain vectorised `(n_source,
n_edges)` closed-form evaluation, batched over sources
(`sesnaimpute.batches.batches`) so one batch's transient array stays
bounded regardless of a region's own source count (CODING_RULES.md rule
10b).

The `NODE_LO`/`NODE_W` bracket (`prior.column_grid.bracket`) is still
computed and written to the output product, since other classes' node
blend (IMPLEMENTATION.md section 2) reads it from the prior table -- it
plays no part in this module's own extinction integral, which uses each
source's exact column directly.

The YSO `A_GRID` (`prior.yso_selection`, a zero prepended to
`column_grid.nodes`) and the H2S `A_NODES`-plus-zero grid
(`prior.h2s`, the same `column_grid.nodes`) are the same array by
construction; the exact marginal's bin masses are computed once and
read by both integrals.
"""

import os

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid as column_grid_module
from sesnaimpute.prior import depth_groups as depth_groups_module
from sesnaimpute.prior import selection as selection_module
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


def _group_index(config, region, depth_groups):
    """Each source's own nearest depth-group centre (SPEC_PRIORS.md
    section 1.3): the region's 8-band limits split into the common-mode
    depth shift and the five-band residual `Delta`, then the nearest of
    the region's `K` cluster centres."""
    log10_flim_8 = np.log10(limits_module.limits(config, region))
    _s, delta5 = selection_module.split_common_mode(log10_flim_8, depth_groups.ref_log10_flim)
    return depth_groups.assign_group(delta5)


def _yso_selection_tables(config, region):
    """`A_GRID`, `G_1MYR`, `G_3MYR` (`prior.yso_selection.build`'s own
    per-region product)."""
    path = config_module.product_path(config, "bms", "yso", "selection", "region", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.counts_cloud: no YSO selection product for region %r at %s "
            "-- run the 'prior.yso_selection' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(a_grid=np.asarray(f["A_GRID"][:], dtype=np.float64),
                    g_1myr=np.asarray(f["G_1MYR"][:], dtype=np.float64),
                    g_3myr=np.asarray(f["G_3MYR"][:], dtype=np.float64))


def _m_lim_8um_1myr(config, region):
    """The region's own row of the YSO literature-check summary
    (SPEC_PRIORS.md section 6.2's Gutermuth one-band form,
    `prior.yso_selection.build`'s 30-row product)."""
    path = config_module.product_path(config, "bms", "yso", "summary", "region")
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["REGION"][:]]
        if region not in names:
            raise ValueError("prior.counts_cloud: %r has no row in %s" % (region, path))
        return float(f["M_LIM_8UM_1MYR"][names.index(region)])


def _h2s_region_product(config, region):
    """`ETA`, `EPS_EXT`, the brightness lognormal, `A_NODES` and `EPS`
    (`prior.h2s.build`'s own per-region product)."""
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
            a_nodes=np.asarray(f["A_NODES"][:], dtype=np.float64),
            eps=np.asarray(f["EPS"][:], dtype=np.float64),
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

def bin_mass_exact(shape, a_edges, rows, a_col, sigma_col):
    """`(n_src, n_edges - 1)`: the exact mass `p(a | A_s)` places in each
    of `a_edges`'s bins, one source per row (SPEC_PRIORS.md sections
    6.2/7's "a sum over grid bins ... cdf_exact differences") -- the
    closed-form cdf (`YsoShape.cdf_exact`) at each source's own adopted
    column and measurement uncertainty, evaluated at every edge and
    differenced, batched over sources so one batch's transient
    `(batch, n_edges)` array stays under `BIN_MASS_BUDGET_BYTES`."""
    n_src = rows.size
    n_edges = a_edges.size
    out = np.empty((n_src, n_edges - 1), dtype=np.float64)
    row_bytes = 2 * n_edges * 8  # the (batch, n_edges) cdf array, float64
    for start, stop in batches_module.batches(n_src, row_bytes, BIN_MASS_BUDGET_BYTES):
        nb = stop - start
        rows_rep = np.repeat(rows[start:stop], n_edges)
        a_col_rep = np.repeat(a_col[start:stop], n_edges)
        sigma_rep = np.repeat(sigma_col[start:stop], n_edges)
        a_rep = np.tile(a_edges, nb)
        cdf = shape.cdf_exact(a_rep, rows_rep, a_col_rep, sigma_rep).reshape(nb, n_edges)
        out[start:stop] = np.diff(cdf, axis=1)
    return out


def eps_yso_from_bin_mass(bin_mass, g_row):
    """`Integral da g_k(a) p(a | A_s)`, per source (SPEC_PRIORS.md section
    6.2): the group's own tabulated curve read at each bin's midpoint (the
    linear blend between its two bracketing nodes), weighted by that
    bin's own exact marginal mass, summed."""
    g_mid = 0.5 * (g_row[:, :-1] + g_row[:, 1:])
    return np.sum(bin_mass * g_mid, axis=1)


def eps_h2s_from_bin_mass(bin_mass, group_idx, eps_h2s, log10_sigma_grid,
                           logsig_mean, logsig_std):
    """`Integral da dlog10(Sigma) eps_k(a, Sigma) p(a | A_s) p_r(log10
    Sigma)`, per source (SPEC_PRIORS.md section 7): the group's own
    `eps_k(a, Sigma)` table, with an `a = 0` edge prepended (flat
    continuation below the tabulated floor, mirroring the YSO `A_GRID`'s
    own 0-node), read at each `a`-bin's midpoint and weighted by that
    bin's own exact marginal mass -- a loop over the small, fixed `a`-bin
    axis (never over sources) -- then integrated over `log10 Sigma` by a
    plain quadrature on the region's own 41-point grid (as
    `prior.h2s.source_pass_fraction` already does for its own point-
    evaluated report)."""
    eps_extended = np.concatenate([eps_h2s[:, :1, :], eps_h2s], axis=1)  # (K, n_node+1, n_sigma)
    n_bins = bin_mass.shape[1]
    n_sigma = log10_sigma_grid.size
    eps_vs_sigma = np.zeros((bin_mass.shape[0], n_sigma), dtype=np.float64)
    for b in range(n_bins):
        edge_lo = eps_extended[group_idx, b, :]
        edge_hi = eps_extended[group_idx, b + 1, :]
        eps_vs_sigma += bin_mass[:, b][:, None] * 0.5 * (edge_lo + edge_hi)

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

    depth_path = config_module.product_path(config, "bms", "sesna", "depth-groups", "region")
    depth_groups = depth_groups_module.DepthGroups.read(depth_path, region)
    group_idx = _group_index(config, region, depth_groups)

    n_law = yso_module.law_count(config, region, a_col, provenance)
    shape = yso_module.YsoShape.read(config, region)
    sel = _yso_selection_tables(config, region)
    m_lim = _m_lim_8um_1myr(config, region)
    h2s = _h2s_region_product(config, region)
    n_law_blurred = _n_law_blurred(config, region)
    ridge = _ridge(config, region)

    a_edges = sel["a_grid"]
    a_edges_h2s = np.concatenate(([0.0], h2s["a_nodes"]))
    if not np.array_equal(a_edges, a_edges_h2s):
        raise ValueError(
            "prior.counts_cloud: %r's YSO A_GRID and H2S A_NODES-plus-zero "
            "grid disagree -- both are built from column_grid.nodes and "
            "should be identical" % region)

    bin_mass = bin_mass_exact(shape, a_edges, sightline_row, a_col, sigma_col)
    eps_yso = eps_yso_from_bin_mass(bin_mass, sel["g_1myr"][group_idx])
    eps_yso_3myr = eps_yso_from_bin_mass(bin_mass, sel["g_3myr"][group_idx])
    eps_h2s = eps_h2s_from_bin_mass(bin_mass, group_idx, h2s["eps"],
                                    h2s["log10_sigma_grid"], h2s["logsig_mean"],
                                    h2s["logsig_std"])

    n_yso = n_law * eps_yso
    n_yso_3myr = n_law * eps_yso_3myr
    n_h2s = n_law_blurred * h2s["eta"] * h2s["eps_ext"] * eps_h2s

    return dict(
        sightline_row=sightline_row, node_lo=node_lo, node_w=node_w, group=group_idx,
        n_yso=n_yso, n_yso_3myr=n_yso_3myr, eps_yso=eps_yso, eps_yso_3myr=eps_yso_3myr,
        n_law=n_law,
        ridge_intercept=ridge["intercept"][sightline_row],
        ridge_slope=ridge["slope"][sightline_row],
        ridge_width=ridge["width"][sightline_row],
        n_h2s=n_h2s, eps_h2s=eps_h2s,
        z_yso=np.ones(n_src, dtype=np.float64), z_h2s=eps_h2s,
        m_lim_8um_1myr=m_lim, eta=h2s["eta"], eps_ext=h2s["eps_ext"],
    )


def _write_product(path, out):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    datasets = {
        "SIGHTLINE_ROW": out["sightline_row"], "NODE_LO": out["node_lo"],
        "NODE_W": out["node_w"], "GROUP": out["group"],
        "N_YSO": out["n_yso"], "N_YSO_3MYR": out["n_yso_3myr"],
        "EPS_YSO": out["eps_yso"], "EPS_YSO_3MYR": out["eps_yso_3myr"],
        "N_LAW": out["n_law"],
        "RIDGE_INTERCEPT": out["ridge_intercept"], "RIDGE_SLOPE": out["ridge_slope"],
        "RIDGE_WIDTH": out["ridge_width"],
        "N_H2S": out["n_h2s"], "EPS_H2S": out["eps_h2s"],
        "Z_YSO": out["z_yso"], "Z_H2S": out["z_h2s"],
    }
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.attrs["M_LIM_8UM_1MYR"] = out["m_lim_8um_1myr"]
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
