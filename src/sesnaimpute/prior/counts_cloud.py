"""The per-source YSO and H2S counts on the cloud law (SPEC_PRIORS.md
sections 6.2, 6.4 item 1, and 7; IMPLEMENTATION.md section 5): the
"counts" half of the prior table's per-source join for the two classes
that ride the young-star law count, `prior.yso.law_count`.

For every catalogued source `s`, this module reads that source's OWN
exact selection -- the YSO mass-based `g_s(a)` (`prior.yso_selection`'s
`G_1MYR`/`G_3MYR`, on the shared ladder `X_LADDER . A_s`) and the H2S
`eps_s(a, Sigma)` (`prior.h2s`'s `EPS`, on the same ladder by the
region's own `LOG10_SIGMA_GRID`) -- and the sightline's own exact
extinction marginal `p(a | A_s)` (`prior.yso.YsoShape`, the log-normal
closed form), and combines them:

    EPS_YSO(s)      = Integral da g_s(a) p(a | A_s)
    N_YSO(s)        = N_law(s) * EPS_YSO(s)
    EPS_H2S(s)      = Integral da dlog10(Sigma) eps_s(a, Sigma) p(a | A_s) p_r(log10 Sigma)
    N_H2S(s)        = N_LAW_BLURRED(s) * ETA_r * EPS_EXT * EPS_H2S(s)

Both integrals are exact in the extinction axis: `p(a | A_s)` integrates
in closed form (`YsoShape.cdf_exact`), so its mass in any `a` bin is an
exact difference of two `cdf_exact` calls at each source's own adopted
column and measurement uncertainty; each source's own ladder cells
(`X_LADDER . A_s`, a value at every ladder point, no group lookup) are
read at each bin's midpoint by the linear blend between the two columns
bracketing it. The `log10 Sigma` axis is a plain quadrature on the
region's own 41-point grid, as `prior.h2s.source_pass_fraction` already
does for its own (point-evaluated, not marginal-integrated) report.

No selection lives inside the YSO shape (section 6.2: `Z_YSO = 1`); H2S's
own selection is the count's own normaliser (section 7: `Z_H2S =
EPS_H2S`), since nothing else in this module's `Lambda_H2S` still needs
renormalising against it.

Sources sharing a sightline do NOT share one column: `A_COL_K` is a
per-source adopted value, and the closed form reads it exactly, not
through a shared node table -- `_bin_mass_batch` is a plain vectorised
`(batch, n_edges)` closed-form evaluation. `build_region` runs its whole
per-source pipeline in source batches (`sesnaimpute.batches.batches`):
every per-source array, the H2S selection's own `EPS` table above all
(`n_source, 8, 41)`, is read from its own product file by row-range slice
inside the batch loop and never held for the whole region at once, so
memory stays bounded regardless of a region's own source count.

The `NODE_LO`/`NODE_W` bracket (`prior.column_grid.bracket`) is still
computed and written to the output product, since other classes' node
blend reads it from the prior table -- it plays no part in this module's
own extinction integral, which uses each source's exact column directly.

`X_LADDER` (`prior.selection`) is the same shared ladder both the YSO
and H2S per-source selection products are tabulated on; the exact
marginal's bin masses are computed once, at that one ladder scaled by
each source's own column, and read by both integrals.
"""

import os

import h5py
import healpy as hp
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid as column_grid_module
from sesnaimpute.prior import yso as yso_module

#: Per-batch memory budget for `build_region`'s own source loop: every
#: per-source array it reads has to fit `_build_row_bytes(...)` times the
#: batch size under this ceiling, well inside the 8 GB process limit.
BUILD_REGION_BUDGET_BYTES = 512 << 20

#: The anchor pixelisation the section 6.4 item 1 check integrates over
#: (`prior.yso.NSIDE_ANCHOR`, the same nside `prior.young_stars` uses for
#: "young stars in the anchors").
_OMEGA_PIX512_DEG2 = hp.nside2pixarea(yso_module.NSIDE_ANCHOR, degrees=True)

#: Row-identity columns `granules.access.per_source` would need to permute
#: a "source"-granule file's rows into catalogue order; none of this
#: module's per-source products carry one (checked at read time,
#: `_check_no_permutation`), so a batch's own disk row range already is
#: its catalogue-row range and a direct `h5py` slice needs no reordering.
_ROW_IDENTITY_COLUMNS = ("CATALOG_ROW", "ROWINDEX", "ROW")


def _check_no_permutation(f, path):
    found = [c for c in _ROW_IDENTITY_COLUMNS if c in f]
    if found:
        raise ValueError(
            "prior.counts_cloud: %s carries a row-identity column %s -- "
            "batched reads assume disk row order is already catalogue-row "
            "order (granules.access.per_source's fast path); this file "
            "needs the permuting read instead" % (path, found))


# ---------------------------------------------------------------------
# per-region reads (small, whole-region: one row per sightline or one row
# per region, never one row per source)
# ---------------------------------------------------------------------

def _h2s_region_product(config, region):
    """`ETA`, `EPS_EXT`, the brightness lognormal, `LOG10_SIGMA_GRID`
    (`prior.h2s.build`'s own per-region product; the tabulated per-source
    `A_NODES`/`EPS` it used to carry now live in the per-source selection
    product, read by `build_region` in its own batch loop)."""
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
# each source's own column and own ladder, one already-loaded batch
# ---------------------------------------------------------------------

def _bin_mass_batch(shape, x_ladder, rows, a_col, sigma_col, map_class, zp_sigma_k=None):
    """`(batch, n_x - 1)`: the exact mass `p(a | A_s)` places in each of
    this batch's own ladder bins, `a = X_LADDER . A_s` (SPEC_PRIORS.md
    sections 6.2/7) -- the closed-form cdf (`YsoShape.cdf_exact`) at each
    source's own adopted column and measurement uncertainty, evaluated at
    every ladder point (scaled by that source's own column) and
    differenced. Operates on one already-loaded batch (`build_region`'s
    own source loop, item 3): the caller owns the batch size, since the
    dominant cost per expanded (source, ladder-point) row is not this
    function's own small `(batch, n_x)` cdf array but `cdf_exact`'s
    internal per-cell arrays, each `(expanded_batch, n_cell)` wide where
    `n_cell` is the region's own embedding-profile cell count. `map_class`
    is the SOURCE's own arm (`A_COL_PROVENANCE`), required -- never the
    sightline's block-averaged one (fixed defect, owner 2026-09-06).
    `zp_sigma_k`, one per source (mag, 0 for Planck-arm), is the
    Herschel field zero point's own uncertainty; omitting it falls back
    to the survey-wide RMS."""
    n_x = x_ladder.size
    nb = rows.size
    rows_rep = np.repeat(rows, n_x)
    a_col_rep = np.repeat(a_col, n_x)
    sigma_rep = np.repeat(sigma_col, n_x)
    map_class_rep = np.repeat(map_class, n_x)
    zp_rep = np.repeat(zp_sigma_k, n_x) if zp_sigma_k is not None else None
    a_rep = (x_ladder[None, :] * a_col[:, None]).reshape(-1)
    cdf = shape.cdf_exact(a_rep, rows_rep, a_col_rep, sigma_rep, map_class_rep,
                          zp_sigma_k=zp_rep).reshape(nb, n_x)
    return np.diff(cdf, axis=1)


def eps_yso_from_bin_mass(bin_mass, g_row):
    """`Integral da g_s(a) p(a | A_s)`, per source (SPEC_PRIORS.md section
    6.2): this source's own tabulated ladder curve read at each bin's
    midpoint (the linear blend between its two bracketing ladder points),
    weighted by that bin's own exact marginal mass, summed."""
    g_mid = 0.5 * (g_row[:, :-1] + g_row[:, 1:])
    return np.sum(bin_mass * g_mid, axis=1)


def eps_h2s_from_bin_mass(bin_mass, eps_h2s, log10_sigma_grid, logsig_mean, logsig_std):
    """`Integral da dlog10(Sigma) eps_s(a, Sigma) p(a | A_s) p_r(log10
    Sigma)`, per source (SPEC_PRIORS.md section 7): this source's own
    `eps_s(a, Sigma)` table, read at each `a`-bin's midpoint and weighted
    by that bin's own exact marginal mass -- a loop over the small, fixed
    `a`-bin axis (never over sources) -- then integrated over `log10
    Sigma` by a plain quadrature on the region's own 41-point grid."""
    n_bins = bin_mass.shape[1]
    n_sigma = log10_sigma_grid.size
    eps_vs_sigma = np.zeros((bin_mass.shape[0], n_sigma), dtype=np.float64)
    for b in range(n_bins):
        edge_lo = eps_h2s[:, b, :]
        edge_hi = eps_h2s[:, b + 1, :]
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

#: The names, in write order, of `build_region`'s own output datasets --
#: every per-source column of `bms/table/counts-cloud_table_source`
#: (module docstring), preallocated once per region and filled by batch.
_OUTPUT_COLUMNS = (
    "SIGHTLINE_ROW", "NODE_LO", "NODE_W", "N_YSO", "N_YSO_3MYR",
    "EPS_YSO", "EPS_YSO_3MYR", "N_LAW", "RIDGE_INTERCEPT", "RIDGE_SLOPE",
    "RIDGE_WIDTH", "N_H2S", "EPS_H2S", "Z_YSO", "Z_H2S",
    "M_LIM_8UM_1MYR", "IMF_FRAC_ABOVE_MLIM",
)


def _build_row_bytes(n_cell, n_x, n_sigma):
    """The per-source byte budget `build_region`'s own batch loop sizes
    itself against (item 3): the H2S selection's own `EPS` row, `n_x *
    n_sigma` float64 (8 x 41 x 8 = 2,624 bytes at this survey's ladder and
    sigma-grid widths) -- the largest single per-source array read --
    plus the closed-form extinction integral's own working set for one
    source (`_bin_mass_batch`'s docstring: `n_x` expanded rows, each
    holding about 9 arrays of width `n_cell + 1` for one live mixture
    component)."""
    eps_row_bytes = n_x * n_sigma * 8
    closed_form_row_bytes = n_x * (9 * (n_cell + 1) * 8 + 2 * 8)
    return eps_row_bytes + closed_form_row_bytes


def build_region(config, region):
    """Computes and writes one region's `bms/table/counts-cloud_table_
    source` product (module docstring), end to end on the same source
    chunks throughout: every per-source array -- the adopted columns, the
    YSO selection `G_1MYR`/`G_3MYR`, the H2S selection `EPS`, the blurred
    law count, the ridge -- is read from its own product file by `h5py`
    row-range slice inside the batch loop, never as a whole-region array,
    and each batch's own results are written into the (preallocated)
    output datasets before the next batch is read. `EPS`, at `(n_source,
    8, 41)` float64, is by far the largest of them (`_build_row_bytes`):
    at Cygnus X's 3,313,391 sources a whole-region read would be about
    8.7 GB by itself, over the 8 GB ceiling before anything else runs.
    Returns `(n_src, path)`.
    """
    rs = access.region_slice(config, region)
    n_src = rs["n_sources"]
    sightline_row = np.asarray(rs["hpx256_row"], dtype=np.intp)

    shape = yso_module.YsoShape.read(config, region)
    h2s = _h2s_region_product(config, region)
    ridge = _ridge(config, region)
    nodes = column_grid_module.nodes(config)
    n_cell = shape.p_u.shape[1]

    adopted_path = config_module.product_path(config, "sky/derived", "adopted",
                                               "column", "source", region=region)
    yso_sel_path = config_module.product_path(config, "bms", "yso", "selection",
                                               "source", region=region)
    h2s_sel_path = config_module.product_path(config, "bms", "h2s", "selection",
                                               "source", region=region)
    law_blur_path = config_module.product_path(config, "bms", "h2s", "law-blurred",
                                                "source", region=region)
    for path, label in ((yso_sel_path, "YSO selection"), (h2s_sel_path, "H2S selection"),
                        (law_blur_path, "H2S law-blurred")):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "prior.counts_cloud: no %s product for region %r at %s -- "
                "run its RUNBOOK line first" % (label, region, path))

    out_path = config_module.product_path(config, "bms", "table", "counts-cloud",
                                          "source", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(adopted_path, "r") as fa, \
         h5py.File(yso_sel_path, "r") as fy, \
         h5py.File(h2s_sel_path, "r") as fh, \
         h5py.File(law_blur_path, "r") as fl, \
         h5py.File(out_path, "w") as fo:

        for f, path in ((fa, adopted_path), (fy, yso_sel_path),
                        (fh, h2s_sel_path), (fl, law_blur_path)):
            _check_no_permutation(f, path)

        x_ladder = np.asarray(fy["X_LADDER"][:], dtype=np.float64)
        x_ladder_h2s = np.asarray(fh["X_LADDER"][:], dtype=np.float64)
        if not np.array_equal(x_ladder, x_ladder_h2s):
            raise ValueError(
                "prior.counts_cloud: %r's YSO and H2S per-source selection "
                "products disagree on X_LADDER -- both should be `prior."
                "selection.X_LADDER`" % region)
        n_x = x_ladder.size
        n_sigma = h2s["log10_sigma_grid"].size

        fo.attrs["GRANULE"] = "source"
        fo.attrs["ETA"] = h2s["eta"]
        fo.attrs["EPS_EXT"] = h2s["eps_ext"]
        fo.attrs["KAPPA_HERSCHEL"] = yso_module.KAPPA_HERSCHEL
        fo.attrs["KAPPA_PLANCK"] = yso_module.KAPPA_PLANCK
        dsets = {name: fo.create_dataset(name, shape=(n_src,), dtype=np.float32)
                 for name in _OUTPUT_COLUMNS}

        row_bytes = _build_row_bytes(n_cell, n_x, n_sigma)
        spans = list(batches_module.batches(n_src, row_bytes, BUILD_REGION_BUDGET_BYTES))

        has_zp_sigma_k = "ZP_SIGMA_K" in fa

        def _one_batch(start, stop):
            rows = sightline_row[start:stop]
            a_col = np.asarray(fa["A_COL_K"][start:stop], dtype=np.float64)
            sigma_col = np.asarray(fa["A_COL_SIG_K"][start:stop], dtype=np.float64)
            provenance = np.asarray(fa["A_COL_PROVENANCE"][start:stop])
            # the Herschel field zero point's own uncertainty (owner,
            # 2026-09-06), 0 for Planck-arm sources; zeros (the old,
            # field-less behaviour) for a region whose adopted column
            # product predates it.
            zp_sigma_k = (np.asarray(fa["ZP_SIGMA_K"][start:stop], dtype=np.float64)
                         if has_zp_sigma_k else np.zeros(stop - start, dtype=np.float64))
            node_lo, node_w = column_grid_module.bracket(a_col, nodes)
            n_law = yso_module.law_count(config, region, a_col, provenance)

            g_1myr = np.asarray(fy["G_1MYR"][start:stop], dtype=np.float64)
            g_3myr = np.asarray(fy["G_3MYR"][start:stop], dtype=np.float64)
            m_lim = np.asarray(fy["M_LIM_8UM_1MYR"][start:stop], dtype=np.float64)
            imf_frac = np.asarray(fy["IMF_FRAC_ABOVE_MLIM"][start:stop], dtype=np.float64)
            eps_table = np.asarray(fh["EPS"][start:stop], dtype=np.float64)
            n_law_blurred = np.asarray(fl["N_LAW_BLURRED_DEG2"][start:stop], dtype=np.float64)

            bin_mass = _bin_mass_batch(shape, x_ladder, rows, a_col, sigma_col, provenance, zp_sigma_k=zp_sigma_k)
            eps_yso = eps_yso_from_bin_mass(bin_mass, g_1myr)
            eps_yso_3myr = eps_yso_from_bin_mass(bin_mass, g_3myr)
            eps_h2s = eps_h2s_from_bin_mass(bin_mass, eps_table, h2s["log10_sigma_grid"],
                                            h2s["logsig_mean"], h2s["logsig_std"])

            n_yso = n_law * eps_yso
            n_yso_3myr = n_law * eps_yso_3myr
            n_h2s = n_law_blurred * h2s["eta"] * h2s["eps_ext"] * eps_h2s
            nb = stop - start

            values = dict(
                SIGHTLINE_ROW=rows, NODE_LO=node_lo, NODE_W=node_w,
                N_YSO=n_yso, N_YSO_3MYR=n_yso_3myr, EPS_YSO=eps_yso,
                EPS_YSO_3MYR=eps_yso_3myr, N_LAW=n_law,
                RIDGE_INTERCEPT=ridge["intercept"][rows], RIDGE_SLOPE=ridge["slope"][rows],
                RIDGE_WIDTH=ridge["width"][rows],
                N_H2S=n_h2s, EPS_H2S=eps_h2s,
                Z_YSO=np.ones(nb, dtype=np.float64), Z_H2S=eps_h2s,
                M_LIM_8UM_1MYR=m_lim, IMF_FRAC_ABOVE_MLIM=imf_frac,
            )
            # each batch owns a disjoint row range, so writing here (rather
            # than collecting every batch's arrays back on the main thread
            # first) never holds more than one batch's output in memory.
            for name, arr in values.items():
                dsets[name][start:stop] = np.asarray(arr, dtype=np.float32)

        Parallel(n_jobs=max(1, config.n_jobs), prefer="threads")(
            delayed(_one_batch)(start, stop) for start, stop in spans)

    return n_src, out_path


def build(config, regions=None):
    """Writes, per region, `bms/table/counts-cloud_table_source` (module
    docstring); prints the section 6.4 item 1 check (`law_area_check`)
    beside Pokhrel+2020's Table 2 total where the S-D38 study record
    carries it. `regions` default: all thirty."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    for region in region_names:
        n_src, path = build_region(config, region)
        n_law_total = law_area_check(config, region)
        print("prior.counts_cloud: %s: %d sources, N_law region total=%.4g "
              "young stars (section 6.4 item 1, against Pokhrel+2020's Table 2 "
              "total per the S-D38 study record) -> %s"
              % (region, n_src, n_law_total, path))


if __name__ == "__main__":
    run(build)
