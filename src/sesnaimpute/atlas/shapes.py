"""The prior at a source: its shape and, beneath it, the class share of the
prior at each grid cell, one page per region (SPEC_BMSTP_DRAFT.md sec.
1.1's factorisation, sec. 1.4's template weights, sec. 2's common grid,
sec. 4.1-4.2, sec. 5.1-5.6).
Report-only: nothing written here is read by the fitter or by any other
`bmstp`/`fittp` stage.

Per region, ONE source from the density table -- the one at the region's
median `A_COL_K` (`_select_sources`'s first choice; the 99th-percentile
source is not drawn: the normalised shapes differ between
sightlines only through the profile, and the fitter's actual class
decision needs the levels the old page's per-source bar chart kept apart
from the shapes it plotted). Six panels per row (GAL, YSO, H2S, STAR,
PAHC, AGB), sharing ONE pair of axes, `log10 x` and `log10 F_4.5` in mJy
(sec. 2: one common brightness axis for every class, H2S included since
W58's rule -- its template conversion folded in at the shape stage).

Row 1, unchanged in content and pixel output from the previous page's own
median-source row: the normalised shape `h_C(x, F_4.5)` the fitter reads
for that source -- `fittp.prior_reader.load`/`prepare`, which blurs the
class's stored grain shape by the source's own column kernel (sec. 4.2) --
`log10`, one common colour scale (its own norm is still taken over BOTH
the median and the (no-longer-drawn) 99th-percentile source's panels, so
the colour mapping, and so the pixels, match the prior page's exactly).

Row 2, the class share: for each class and grid cell,
`S_C(x, F) = A_C(s) h_C(x, F) f_C(F; s) / Sum_C' A_C'(s) h_C'(x, F) f_C'(F; s)`
(sec. 1.1's product the fitter actually compares between classes at a
source's own depth and brightness, sec. 1.4's template weights), on a
LINEAR 0-1 colour scale, cells masked blank where every class's own
SHAPE sits at its stored floor there (`h_C(x, F) <= bmstp.grid.FLOOR *
max(h_C)` for every class in the sum, sec. 2's floor, 1e-6 of the
class's own peak) -- a total-sized mask never fires, since GAL's own
`A_C(s)` is the largest of the six and its floored shape is not
negligible next to the grid's overall maximum, so it would read as
GAL share 1 everywhere no other class has real density; masking on the
shape's own floor is what "at least one class has real density here"
means. `f_C` is 1 for every class whose weight-table factors
are all normalised over theta (GAL's `colour`, YSO's `population`, AGB's
`tau`, H2S's `uniform` sec 1.4: `Sum_theta pi_C(theta, F) = 1` at every F
by construction, so multiplying it in and summing over theta is exactly 1)
except STAR and PAHC, whose second factor (`uncontaminated`/`contrast`,
sec. 5.1/5.3) is a probability, not normalised:
`f_STAR(F; s) = Sum_theta pi_STAR(theta|F) [1 - P(q_s,theta)]`,
`f_PAHC(F; s) = Sum_theta pi_PAHC(theta|F) P(q_s,theta)`, `pi_C(theta|F)`
read directly off the class's own `type` factor table (P5 factor 0,
already the sec. 1.4 axis, `Sum_theta pi_C(theta, F) = 1` at every F) and
`P` read exactly as `fittp.prior_reader._factor_ln` reads it, `arg =
b_star + C_F + D_PAHC(s)` with `b_star = F - C_THETA[theta]` (sec. 2's
`log10 F_4.5 = log10 Bhat + C_THETA` line, solved for `log10 Bhat` at the
query brightness `F`) -- `bmstp.template_weights._read_pahc_curve`/
`_p_at_neg_log10_q`, imported not re-derived. H2S's own `f_C` is 1 (its
`uniform` factor is normalised over theta, sec 5.6), the same STAR/PAHC
carve-out above does not apply to it, so it is included in row 2's
denominator sum like every other class since W58's rule
(`briefs/reports/W57.md` disclosed the earlier exclusion; removed here).
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.fittp import prior_reader
from sesnaimpute.bmstp import grid
from sesnaimpute.bmstp import template_weights

PAGE_W_IN, PAGE_H_IN = 16.0, 9.0

#: The old page's panel order (W18's brief).
CLASS_ORDER = ("GAL", "YSO", "H2S", "STAR", "PAHC", "AGB")

#: Every class reads the one common brightness axis (sec. 2, H2S included
#: since W58): `log10 F_4.5` in mJy, `bmstp.grid.LOG10_F45_EDGES` -- the
#: row's one outside y-axis label below.
_SHARED_Y_LABEL = "log10 F_4.5   [mJy]"

#: The one shared caption below the page, the spec's own term for `x`
#: (sec. 0 vocabulary): every panel's short per-panel `log10 x` xlabel
#: stays as it was (row 1 pixel-identical), this is the figure-level
#: definition instead.
_X_CAPTION = ("log10 x   (scaled extinction x = a / A_s: the extinction "
              "in front of the object as a fraction of the sightline's column)")

_ARM_NAME = {0: "Herschel", 1: "Planck"}


def _select_sources(a_col):
    """`(idx_median, idx_p99)`: the catalogued source closest to the
    region's median `A_COL_K` and closest to its 99th percentile."""
    idx_median = int(np.argmin(np.abs(a_col - np.median(a_col))))
    idx_p99 = int(np.argmin(np.abs(a_col - np.percentile(a_col, 99.0))))
    return idx_median, idx_p99


def _read_density_table(config, region):
    """P1's per-source rows this page needs: `NAME`, `A_COL_K`, `ARM`,
    the star-family/cloud grain indices, `D_PAHC` (sec. 4.1, row 2's
    `f_STAR`/`f_PAHC` read point), the six `DENSITY_<C>` (sec.
    4.1), and the file's `KAPPA_HERSCHEL`/`KAPPA_PLANCK` attributes
    (sec. 5.5's law, the caption line)."""
    path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    with h5py.File(path, "r") as f:
        d = dict(
            name=f["NAME"][:], a_col=f["A_COL_K"][:].astype(np.float64),
            arm=f["ARM"][:].astype(np.int64), tile=f["TILE"][:].astype(np.int64),
            sightline=f["SIGHTLINE_ROW"][:].astype(np.int64),
            d_pahc=f["D_PAHC"][:].astype(np.float64),
            kappa_herschel=float(f.attrs["KAPPA_HERSCHEL"]),
            kappa_planck=float(f.attrs["KAPPA_PLANCK"]),
        )
        for c in ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S"):
            d[c] = f["DENSITY_%s" % c][:].astype(np.float64)
    return d


def _on_grid_star(config, region):
    """`(ON_GRID_STAR, ON_GRID_AGB)` per tile (P2, sec. 2's ON-GRID
    FRACTION, W26): PAHC reads STAR's own (sec. 5.3 "Grain": the shape IS
    STAR's)."""
    path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
    with h5py.File(path, "r") as f:
        return f["ON_GRID_STAR"][:].astype(np.float64), f["ON_GRID_AGB"][:].astype(np.float64)


def _on_grid_yso(config, region):
    """`ON_GRID_YSO` per sightline (P3, W26)."""
    path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    with h5py.File(path, "r") as f:
        return f["ON_GRID_YSO"][:].astype(np.float64)


def _on_grid_h2s(config, region):
    """`ON_GRID_H2S` per sightline (P3, W58: `GRID_H2S` is on the common
    grid now, so its own on-grid fraction is measured, not fixed at 1)."""
    path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    with h5py.File(path, "r") as f:
        return f["ON_GRID_H2S"][:].astype(np.float64)


def _on_grid_gal(config):
    """`ON_GRID_GAL`, the one survey-wide attr (P4, W26): read directly,
    not recomputed -- `bmstp.shapes.build_gal` is this number's own build."""
    path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
    with h5py.File(path, "r") as f:
        return float(f.attrs["ON_GRID_GAL"])


def _class_on_grid(cls, dtab, on_grid_star, on_grid_agb, on_grid_yso, on_grid_h2s, on_grid_gal, src_idx):
    if cls in ("STAR", "PAHC"):
        return float(on_grid_star[dtab["tile"][src_idx]])
    if cls == "AGB":
        return float(on_grid_agb[dtab["tile"][src_idx]])
    if cls == "YSO":
        return float(on_grid_yso[dtab["sightline"][src_idx]])
    if cls == "H2S":
        # W58: GRID_H2S is on the common grid now, its own on-grid
        # fraction measured per sightline (sec. 4.1), same as YSO's.
        return float(on_grid_h2s[dtab["sightline"][src_idx]])
    return float(on_grid_gal)  # GAL


def _panel_arrays(config, region, cls, rows):
    """`(density, x_edges, b_edges, mass, reader)`: the shape `h_C` the
    fitter reads for each of `rows`' sources -- `fittp.prior_reader.load`/
    `prepare`, which blurs the class's stored grain shape by the source's
    own column kernel (sec. 4.2), floors it and renormalises it to sum to
    one -- converted from a per-cell mass to a density per dex² (`/ (dlx *
    dlb)`, both axes `log10`, sec. 2). `mass` is `prepare`'s own post-blur
    sum (sec. 9's "shape normalisation after blur", 1 +/- 1e-3 by
    construction -- a DIFFERENT check from the panel's own on-grid
    fraction, which is the grain's raw (pre-blur) population weight the
    grid ever admitted). `reader` is returned too -- row 2's `f_STAR`/
    `f_PAHC` (sec. 1.4) read the SAME P5 weight table this call already
    opened, never a second `load`."""
    reader = prior_reader.load(config, region, cls)
    h = prior_reader.prepare(reader, np.asarray(rows))
    density = h.astype(np.float64) / (reader.dlx * reader.dlb)
    mass = density.sum(axis=(1, 2)) * reader.dlx * reader.dlb
    return density, reader.x_edges, reader.b_edges, mass, reader


#: Row 2's denominator classes (sec. 1.1/1.4): every class, since W58 --
#: H2S is on the common grid at read time now (P3's `GRID_H2S`, sec. 4.1),
#: so it is no longer left out of the share's total.
_SHARE_CLASSES = ("GAL", "YSO", "H2S", "STAR", "PAHC", "AGB")


def _factor_marginal(panel_b_centers, reader, cls, d_pahc_s, curve):
    """`f_C(F; s)` for STAR/PAHC (module docstring, sec. 1.4, 5.1, 5.3):
    `Sum_theta pi_C(theta|F) x {1-P, P}(-log10 q)`. `pi_C(theta|F)` is P5
    factor 0 (`type`, `template_weights.build_sps`/`build_pahc`'s own
    dict order), already normalised over theta at every `F` cell (sec.
    1.4), read directly off its `F` axis -- no per-template offset, since
    `type`'s own `C_F = C_THETA` cancels sec. 2's `log10 F_4.5 = log10
    Bhat + C_THETA` line exactly. `P` is factor 1 (`uncontaminated`/
    `contrast`)'s own curve, read exactly as `fittp.prior_reader.
    _factor_ln` reads any factor, `arg = b_star + C_F + D_PAHC(s)`, with
    `b_star = F - C_THETA[theta]` (the same line solved for `log10 Bhat`
    at the query brightness `F`, since row 2 has no fitted `a_hat`/slope
    to place a* away from F) -- `template_weights._read_pahc_curve`/
    `_p_at_neg_log10_q`, imported not re-derived, not the stored/floored
    P5 table (which fixes the argument at F, sec. 1.4, wrong for the
    per-template shift row 2 needs)."""
    type_factor, contrast_factor = reader.factors[0], reader.factors[1]
    pi_theta_f = type_factor["W"]          # (n_model, n_b), sum_theta = 1 at every F
    c_theta = reader.c_theta               # (n_model,)
    c_f = contrast_factor["C_F"]           # (n_model,)
    centers, p_q = curve
    arg = panel_b_centers[None, :] - c_theta[:, None] + c_f[:, None] + d_pahc_s
    p_val = template_weights._p_at_neg_log10_q(arg, centers, p_q)
    term = (1.0 - p_val) if cls == "STAR" else p_val
    return (pi_theta_f * term).sum(axis=0)


def _build_region_data(config, region):
    dtab = _read_density_table(config, region)
    # sec. "row 1 pixel-identical": `_select_sources`'s SECOND choice
    # (the 99th-percentile source) is still computed here, alongside the
    # first, purely so row 1's colour norm below spans the SAME two-source
    # panel set the previous page's norm did -- its own row is not drawn
    # (module docstring).
    idx_median, idx_p99 = _select_sources(dtab["a_col"])
    rows = np.array([idx_median, idx_p99])
    on_grid_star, on_grid_agb = _on_grid_star(config, region)
    on_grid_yso = _on_grid_yso(config, region)
    on_grid_h2s = _on_grid_h2s(config, region)
    on_grid_gal = _on_grid_gal(config)

    panels = {}
    readers = {}
    for cls in CLASS_ORDER:
        density2, x_edges, b_edges, mass2, reader = _panel_arrays(config, region, cls, rows)
        readers[cls] = reader
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
        for i, src_idx in enumerate(rows):
            d = density2[i]
            pi, pj = np.unravel_index(int(np.argmax(d)), d.shape)
            on_grid = _class_on_grid(cls, dtab, on_grid_star, on_grid_agb, on_grid_yso, on_grid_h2s,
                                      on_grid_gal, src_idx)
            panels[(i, cls)] = dict(
                density=d, x_edges=x_edges, b_edges=b_edges,
                x_centers=x_centers, b_centers=b_centers,
                mass=float(mass2[i]), on_grid=on_grid,
                peak_x=float(x_centers[pi]), peak_b=float(b_centers[pj]),
                intensity=float(dtab[cls][src_idx]))

    # Row 2, the class share at the median source (module docstring, sec.
    # 1.1/1.4): Lambda_C(x, F) = A_C(s) h_C(x, F) f_C(F; s) for every
    # class (H2S included since W58, its shape on the common grid now),
    # summed to the total and divided back into each --
    # the SAME product the fitter's evidence sum compares between classes
    # (sec. 4.2), not the shape alone. A cell is blanked where every
    # class's own SHAPE sits at its stored floor there (sec. 2, `bmstp.
    # grid.FLOOR`), never by the total's own size: the total is never
    # small in absolute terms (GAL's own A_C(s) times its floored shape
    # is not negligible next to the grid's overall maximum, since GAL's
    # peak is the largest of the six), so a total-sized mask never fires
    # and every off-support cell reads as GAL share 1 -- masking on the
    # shape's own floor is what "at least one class has real density
    # here" actually means.
    d_pahc_s = float(dtab["d_pahc"][idx_median])
    curve = template_weights._read_pahc_curve(config)
    lam = {}
    at_floor = None
    for cls in _SHARE_CLASSES:
        p = panels[(0, cls)]
        h_c = p["density"]
        if cls in ("STAR", "PAHC"):
            f_c = _factor_marginal(p["b_centers"], readers[cls], cls, d_pahc_s, curve)
        else:
            f_c = np.ones(h_c.shape[1], dtype=np.float64)
        lam[cls] = p["intensity"] * h_c * f_c[None, :]
        # "no real density": a class counts as absent from a cell where
        # its shape is within one decade of its own floor (sec. 2's
        # FLOOR, 1e-6 of the peak). The one-cell blur's far tails and the
        # float32 store leave cells a little above the exact floor with
        # no support behind them; exactly-at-floor masking then never
        # fires and GAL's own A_C(s) times its floor colours every empty
        # cell as GAL share 1. A decade above the floor is still five
        # decades below the class's own peak: nothing the fitter would
        # count as support.
        floor_c = h_c <= (10.0 * grid.FLOOR * h_c.max())
        at_floor = floor_c if at_floor is None else (at_floor & floor_c)
    total = sum(lam.values())
    masked = at_floor
    share = {}
    for cls in _SHARE_CLASSES:
        s = np.divide(lam[cls], total, out=np.full_like(total, np.nan), where=~masked)
        share[cls] = s
    return dtab, idx_median, panels, share, masked


def _print_numbers(region, dtab, idx_median, panels, share, masked):
    name = dtab["name"][idx_median].decode("utf-8")
    print("atlas.shapes [%s] median source %s: A_COL_K=%.4g mag arm=%s"
          % (region, name, dtab["a_col"][idx_median], _ARM_NAME[int(dtab["arm"][idx_median])]))
    for cls in CLASS_ORDER:
        p = panels[(0, cls)]
        print("atlas.shapes [%s] row1/%s: mass=%.6f on_grid=%.6f peak_log10x=%.4g peak_log10F45=%.4g "
              "A_C(s)=%.4g deg^-2" % (region, cls, p["mass"], p["on_grid"], p["peak_x"], p["peak_b"], p["intensity"]))
    valid = ~masked
    print("atlas.shapes [%s] row2: masked fraction=%.4f (every class's own shape at its stored floor "
          "there, H2S included in the sum since W58, module docstring)" % (region, float(np.mean(masked))))
    for cls in _SHARE_CLASSES:
        s = share[cls]
        smax = float(np.nanmax(s)) if valid.any() else float("nan")
        smean = float(np.nanmean(s)) if valid.any() else float("nan")
        print("atlas.shapes [%s] row2/%s: share max=%.4f mean(valid cells)=%.4f" % (region, cls, smax, smean))


def _draw_figure(config, region, dtab, idx_median, panels, share, masked):
    plot_style.apply_style()
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))

    margin_l, margin_r, margin_t, margin_b = 0.55, 1.05, 0.85, 0.55
    row_gap, col_gap = 0.90, 0.14
    usable_w = PAGE_W_IN - margin_l - margin_r
    shape_w = (usable_w - 5 * col_gap) / 6.0
    row_h = (PAGE_H_IN - margin_t - margin_b - row_gap) / 2.0

    # row 1's colour norm spans BOTH sources' panels `_build_region_data`
    # computed (median and the no-longer-drawn 99th-percentile one) --
    # the SAME set the previous page's own norm spanned, so row 1's
    # pixels match it exactly (module docstring, "row 1 pixel-identical").
    all_log = np.concatenate([np.log10(p["density"]).ravel() for p in panels.values()])
    norm1 = Normalize(vmin=float(all_log.min()), vmax=float(all_log.max()))
    cmap1 = plt.get_cmap("viridis")
    cmap2 = plt.get_cmap("viridis").copy()
    cmap2.set_bad(color=(0.0, 0.0, 0.0, 0.0))  # masked cells left blank
    norm2 = Normalize(vmin=0.0, vmax=1.0)
    im1 = im2 = None

    src_idx = idx_median
    a_col_src = dtab["a_col"][src_idx]
    for i in range(2):
        y0 = margin_b + (1 - i) * (row_h + row_gap)
        for c, cls in enumerate(CLASS_ORDER):
            x0 = margin_l + c * (shape_w + col_gap)
            ax = fig.add_axes([x0 / PAGE_W_IN, y0 / PAGE_H_IN, shape_w / PAGE_W_IN, row_h / PAGE_H_IN])
            p = panels[(0, cls)]
            if i == 0:
                im1 = ax.imshow(np.log10(p["density"]).T, origin="lower", aspect="auto",
                                 extent=[p["x_edges"][0], p["x_edges"][-1], p["b_edges"][0], p["b_edges"][-1]],
                                 cmap=cmap1, norm=norm1)
                ax.text(0.02, 0.03, "mass=%.4f\non_grid=%.4f\npeak=(%.2f, %.2f)"
                        % (p["mass"], p["on_grid"], p["peak_x"], p["peak_b"]),
                        transform=ax.transAxes, fontsize=5.5, color="white", va="bottom")
            else:
                s = share[cls]
                im2 = ax.imshow(np.ma.masked_invalid(s).T, origin="lower", aspect="auto",
                                 extent=[p["x_edges"][0], p["x_edges"][-1], p["b_edges"][0], p["b_edges"][-1]],
                                 cmap=cmap2, norm=norm2)
            # the y-axis label is set ONCE per row, outside the panels
            # (below), not per panel -- the previous page's six repeated
            # labels overlapped their neighbours. Every panel still shows
            # its own numeric y-ticks (the values, not the label text).
            ax.set_xlabel("log10 x", fontsize=6.5)
            ax.tick_params(labelsize=6)
            xt = np.array([-3.0, -1.0, 1.0])
            ax.set_xticks(xt)
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xt)
            ax2.set_xticklabels(["%.3g" % (a_col_src * 10.0 ** v) for v in xt], fontsize=5)
            ax2.tick_params(length=2, pad=1, labelsize=5)
            if c == 0:
                ax2.set_xlabel("a  [A_K mag]", fontsize=5.5, labelpad=1)
            if i == 0:
                title = "%s  A_C(s) = %.3g deg$^{-2}$" % (cls, p["intensity"])
            else:
                title = "%s share" % cls
            ax.set_title(title, fontsize=8.5 if i == 0 else 9, weight="bold", pad=16)
            if c == 0:
                fig.text(x0 / PAGE_W_IN - 0.30 / PAGE_W_IN, (y0 + 0.5 * row_h) / PAGE_H_IN,
                          _SHARED_Y_LABEL, rotation=90, va="center", ha="center", fontsize=7)

    cax1_rect = [(margin_l + 6 * shape_w + 5 * col_gap + 0.15) / PAGE_W_IN,
                 (margin_b + row_h + row_gap) / PAGE_H_IN, 0.22 / PAGE_W_IN, row_h / PAGE_H_IN]
    cax1 = fig.add_axes(cax1_rect)
    fig.colorbar(im1, cax=cax1, label="log10 h_C(x, F_4.5)   prior shape density [dex$^{-2}$]")

    cax2_rect = [(margin_l + 6 * shape_w + 5 * col_gap + 0.15) / PAGE_W_IN,
                 margin_b / PAGE_H_IN, 0.22 / PAGE_W_IN, row_h / PAGE_H_IN]
    cax2 = fig.add_axes(cax2_rect)
    fig.colorbar(im2, cax=cax2, label="class share of the prior at (x, F_4.5)")

    r = regions_module.REGIONS_BY_NAME[region]
    name_med = dtab["name"][idx_median].decode("utf-8")
    arm_med = _ARM_NAME[int(dtab["arm"][idx_median])]
    caption = (
        "%s -- median source %s (A_COL_K=%.3g mag, %s arm); "
        "d_r=%.0f±%.0f pc; KAPPA_HERSCHEL=%.1f, KAPPA_PLANCK=%.1f"
        % (region, name_med, dtab["a_col"][idx_median], arm_med,
           r.d_r_pc, r.sigma_pc, dtab["kappa_herschel"], dtab["kappa_planck"]))
    fig.suptitle(caption, fontsize=9.5, y=1.0 - 0.15 / PAGE_H_IN)
    fig.text(margin_l / PAGE_W_IN, 0.10 / PAGE_H_IN, _X_CAPTION, fontsize=6.5, ha="left")

    out_dir = f"{config.data_root}/bmstp/atlas/figures"
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in ("png", "pdf"):
        path = f"{out_dir}/prior-shapes_{region}.{fmt}"
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths


def build_region(config, region):
    with progress.Stage("atlas.shapes", region) as st:
        dtab, idx_median, panels, share, masked = _build_region_data(config, region)
        _print_numbers(region, dtab, idx_median, panels, share, masked)
        paths = _draw_figure(config, region, dtab, idx_median, panels, share, masked)
        row1_panels = [p for (i, _c), p in panels.items() if i == 0]
        mass_min = min(p["mass"] for p in row1_panels)
        mass_max = max(p["mass"] for p in row1_panels)
        on_grid_min = min(p["on_grid"] for p in row1_panels)
        on_grid_max = max(p["on_grid"] for p in row1_panels)
        st.done(paths[0], n_panel=len(row1_panels), mass_min=mass_min, mass_max=mass_max,
                on_grid_min=on_grid_min, on_grid_max=on_grid_max,
                masked_fraction=float(np.mean(masked)))
    return paths


def build(config, regions=None):
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region(config, region)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()
    config = config_module.load(args.config)
    build(config, regions=args.regions)


if __name__ == "__main__":
    _main()
