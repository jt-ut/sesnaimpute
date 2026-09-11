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
its template conversion folded in at the shape stage).

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
source's own depth and brightness, sec. 1.4's template weights), computed
in EVERY cell, floors included, on a LINEAR 0-1 colour scale: the drawn
quantity is this share SMOOTHED by the evidence behind its own cell's
winner (owner's ruling 2026-09-10) -- the class with the largest `S_C`
there carries `N` effective members in that cell (`bmstp.grid.n_eff`'s
Kish count, read off the shape product's own `N_EFF_<C>` dataset beside
the grid it was built from, GAL's `n_eff_analytic` giving +inf on its own
support), and every class's share is replaced by the smoothed share
`S_smooth_C = (N . S_C + 1/6) / (N + 1)`, one pseudo-member spread evenly
over the six classes: a cell with no member behind its winner draws flat
at one sixth for every class, a cell with ten members' worth of evidence
reads within a tenth of a member's worth of the raw share, and GAL's
infinite-evidence cells draw the share as is -- a count of evidence
standing in for the floor test, never a threshold or a cut. `f_C` is 1 for every class whose weight-table factors
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
query brightness `F`) -- `population.pahc_curve.read`, the same accessor
`template_weights._pahc_contrast_row` reads, imported not re-derived. H2S's own `f_C` is 1 (its
`uniform` factor is normalised over theta, sec 5.6), the same STAR/PAHC
carve-out above does not apply to it, so it is included in row 2's
denominator sum like every other class
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
from sesnaimpute.population import pahc_curve

PAGE_W_IN, PAGE_H_IN = 16.0, 9.0

#: The old page's panel order (W18's brief).
CLASS_ORDER = ("GAL", "YSO", "H2S", "STAR", "PAHC", "AGB")

#: Every class reads the one common brightness axis (sec. 2, H2S included
#: ): `log10 F_4.5` in mJy, `bmstp.grid.LOG10_F45_EDGES` -- the
#: row's one outside y-axis label below.
_SHARED_Y_LABEL = plot_style.label("log10 F_4.5", "mJy")

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
    """`ON_GRID_H2S` per sightline (P3: `GRID_H2S` is on the common
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


def _n_eff_star(config, region):
    """`(N_EFF_STAR, N_EFF_AGB)` per tile, read off the same P2 file
    `_on_grid_star` already opens (W66e) -- PAHC reads STAR's own (sec.
    5.3 "Grain")."""
    path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
    with h5py.File(path, "r") as f:
        return f["N_EFF_STAR"][:].astype(np.float64), f["N_EFF_AGB"][:].astype(np.float64)


def _n_eff_yso(config, region):
    """`N_EFF_YSO` per sightline, off the same P3 file `_on_grid_yso`
    opens (W66e)."""
    path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    with h5py.File(path, "r") as f:
        return f["N_EFF_YSO"][:].astype(np.float64)


def _n_eff_h2s(config, region):
    """`N_EFF_H2S`, one brightness-only vector for the whole region (P3,
    W66e): H2S carries no `x`-axis evidence of its own (module
    docstring), so every sightline and every `x` column share it."""
    path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    with h5py.File(path, "r") as f:
        return f["N_EFF_H2S"][:].astype(np.float64)


def _n_eff_gal(config):
    """`N_EFF_GAL`, the one survey-wide `(x, F)` grid (P4, W66e): +inf on
    the law's own raw support, 0 off it."""
    path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
    with h5py.File(path, "r") as f:
        return f["N_EFF_GAL"][:].astype(np.float64)


def _class_on_grid(cls, dtab, on_grid_star, on_grid_agb, on_grid_yso, on_grid_h2s, on_grid_gal, src_idx):
    if cls in ("STAR", "PAHC"):
        return float(on_grid_star[dtab["tile"][src_idx]])
    if cls == "AGB":
        return float(on_grid_agb[dtab["tile"][src_idx]])
    if cls == "YSO":
        return float(on_grid_yso[dtab["sightline"][src_idx]])
    if cls == "H2S":
        # GRID_H2S is on the common grid, its own on-grid
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


#: Row 2's denominator classes (sec. 1.1/1.4): every class --
#: H2S is on the common grid at read time now (P3's `GRID_H2S`, sec. 4.1),
#: so it is in the share's total.
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
    to place a* away from F) -- `population.pahc_curve.read`, imported
    not re-derived, not the stored/floored
    P5 table (which fixes the argument at F, sec. 1.4, wrong for the
    per-template shift row 2 needs)."""
    type_factor, contrast_factor = reader.factors[0], reader.factors[1]
    pi_theta_f = type_factor["W"]          # (n_model, n_b), sum_theta = 1 at every F
    c_theta = reader.c_theta               # (n_model,)
    c_f = contrast_factor["C_F"]           # (n_model,)
    arg = panel_b_centers[None, :] - c_theta[:, None] + c_f[:, None] + d_pahc_s
    p_val = curve(-arg)                    # curve's own x-axis is log10 q, arg is -log10 q
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
    n_eff_star, n_eff_agb = _n_eff_star(config, region)
    n_eff_yso = _n_eff_yso(config, region)
    n_eff_h2s_f = _n_eff_h2s(config, region)
    n_eff_gal = _n_eff_gal(config)

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
    # class (H2S included, its shape on the common grid), summed to the
    # total and divided back into each -- the SAME product the fitter's
    # evidence sum compares between classes (sec. 4.2), not the shape
    # alone. Computed in every cell now, floors included (owner's ruling
    # 2026-09-10): `at_floor` below is kept only to report row 2's old
    # floor-excluded mean share for comparison, never to blank a cell.
    d_pahc_s = float(dtab["d_pahc"][idx_median])
    curve = pahc_curve.read(config)
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
        floor_c = h_c <= (10.0 * grid.FLOOR * h_c.max())
        at_floor = floor_c if at_floor is None else (at_floor & floor_c)
    total = sum(lam.values())
    share = {cls: lam[cls] / total for cls in _SHARE_CLASSES}

    # The smoothed share (module docstring, owner's ruling 2026-09-10):
    # each cell's winner (largest raw `S_C` there) lends its own per-cell
    # effective member count `N` (`bmstp.grid.n_eff`/`n_eff_analytic`,
    # W66e's `N_EFF_<C>`, read off the SAME grain -- tile for STAR/PAHC,
    # tile for AGB, sightline for YSO -- the source's own row 1 panel
    # already read); H2S has no `x`-axis evidence of its own so its one
    # brightness vector is shared by every `x` column, and GAL's single
    # survey-wide grid needs no grain index at all.
    tile_med, sl_med = int(dtab["tile"][idx_median]), int(dtab["sightline"][idx_median])
    n_x, n_b = share["GAL"].shape
    n_eff_map = {
        "STAR": n_eff_star[tile_med], "PAHC": n_eff_star[tile_med],
        "AGB": n_eff_agb[tile_med], "YSO": n_eff_yso[sl_med],
        "H2S": np.broadcast_to(n_eff_h2s_f[None, :], (n_x, n_b)),
        "GAL": n_eff_gal,
    }
    share_stack = np.stack([share[cls] for cls in _SHARE_CLASSES])
    n_eff_stack = np.stack([n_eff_map[cls] for cls in _SHARE_CLASSES])
    winner_idx = np.argmax(share_stack, axis=0)
    n_winner = np.take_along_axis(n_eff_stack, winner_idx[None, :, :], axis=0)[0]
    # `S_smooth_C = (N.S_C + 1/6)/(N+1)`: GAL's +inf cells take the limit
    # N -> infinity of that ratio, which is `S_C` itself, computed
    # directly rather than through an infinity-over-infinity division.
    finite = np.isfinite(n_winner)
    n_safe = np.where(finite, n_winner, 0.0)
    smoothed_stack = (n_safe[None, :, :] * share_stack + 1.0 / 6.0) / (n_safe[None, :, :] + 1.0)
    smoothed_stack = np.where(finite[None, :, :], smoothed_stack, share_stack)
    share_smooth = {cls: smoothed_stack[k] for k, cls in enumerate(_SHARE_CLASSES)}
    return dtab, idx_median, panels, share, share_smooth, at_floor, winner_idx, n_winner


def _print_numbers(region, dtab, idx_median, panels, share, share_smooth, at_floor, winner_idx, n_winner):
    name = dtab["name"][idx_median].decode("utf-8")
    print("atlas.shapes [%s] median source %s: A_COL_K=%.4g mag arm=%s"
          % (region, name, dtab["a_col"][idx_median], _ARM_NAME[int(dtab["arm"][idx_median])]))
    for cls in CLASS_ORDER:
        p = panels[(0, cls)]
        print("atlas.shapes [%s] row1/%s: mass=%.6f on_grid=%.6f peak_log10x=%.4g peak_log10F45=%.4g "
              "A_C(s)=%.4g deg^-2" % (region, cls, p["mass"], p["on_grid"], p["peak_x"], p["peak_b"], p["intensity"]))
    valid = ~at_floor
    for k, cls in enumerate(_SHARE_CLASSES):
        s, s_smooth = share[cls], share_smooth[cls]
        mean_before = float(np.mean(s[valid])) if valid.any() else float("nan")
        mean_after = float(np.mean(s_smooth))
        wins = winner_idx == k
        wins_evidenced = float(np.mean(wins & (n_winner >= 1.0)))
        print("atlas.shapes [%s] row2/%s: share mean(old floor-excluded)=%.4f mean_smoothed(every cell)=%.4f "
              "wins_with_N>=1_frac=%.4f" % (region, cls, mean_before, mean_after, wins_evidenced))


def _draw_figure(config, region, dtab, idx_median, panels, share_smooth):
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
    cmap2 = plt.get_cmap("viridis")
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
                # `A_C(s)` moves into this corner text (spelled "area
                # density" here, never `A_C`) because the symbol collides
                # with the top axis's extinction `A` (owner's ruling
                # 2026-09-10); the row-1 title above carries the class
                # name alone.
                ax.text(0.02, 0.03, "mass=%.4f\non_grid=%.4f\npeak=(%.2f, %.2f)\narea density = %.3g deg^-2"
                        % (p["mass"], p["on_grid"], p["peak_x"], p["peak_b"], p["intensity"]),
                        transform=ax.transAxes, fontsize=5.5, color="white", va="bottom")
            else:
                s = share_smooth[cls]
                im2 = ax.imshow(s.T, origin="lower", aspect="auto",
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
            # The top axis ticks at DECADES OF `a` itself (not of `x`,
            # which under a non-power-of-ten `A_s` lands off-decade and
            # prints overlapping labels like 0.000313/3.13): pick the
            # integer powers of ten within the panel's own `log10 x`
            # range, place them at `log10 x = k - log10(A_s)`, and label
            # them as powers of ten so no label overruns its own panel.
            log_a_col = np.log10(a_col_src)
            xlim = ax.get_xlim()
            k_lo = int(np.ceil(xlim[0] + log_a_col))
            k_hi = int(np.floor(xlim[1] + log_a_col))
            decades = np.arange(k_lo, k_hi + 1)
            ax2.set_xticks(decades - log_a_col)
            ax2.set_xticklabels(["$10^{%d}$" % k for k in decades], fontsize=5)
            ax2.tick_params(length=2, pad=1, labelsize=5)
            if c == 0:
                # A plain text annotation, not `set_xlabel` -- an actual
                # twin-axis xlabel makes matplotlib's own title-placement
                # (`_update_title_position`) push THAT panel's title
                # higher than its five neighbours to clear it (owner's
                # ruling 2026-09-10); a right-aligned annotation at the
                # top axis's own right end carries the same information
                # without registering as an axis label.
                ax2.text(1.0, 1.05, plot_style.label("a", "mag"), transform=ax2.transAxes,
                         fontsize=5.5, ha="right", va="bottom")
            ax.set_title(cls, fontsize=8.5 if i == 0 else 9, pad=16)
            if c == 0:
                fig.text(x0 / PAGE_W_IN - 0.30 / PAGE_W_IN, (y0 + 0.5 * row_h) / PAGE_H_IN,
                          _SHARED_Y_LABEL, rotation=90, va="center", ha="center", fontsize=7)

    cax1_rect = [(margin_l + 6 * shape_w + 5 * col_gap + 0.15) / PAGE_W_IN,
                 (margin_b + row_h + row_gap) / PAGE_H_IN, 0.22 / PAGE_W_IN, row_h / PAGE_H_IN]
    cax1 = fig.add_axes(cax1_rect)
    fig.colorbar(im1, cax=cax1, label=plot_style.label("Prior Shape Density", "log10 dex$^{-2}$"))

    cax2_rect = [(margin_l + 6 * shape_w + 5 * col_gap + 0.15) / PAGE_W_IN,
                 margin_b / PAGE_H_IN, 0.22 / PAGE_W_IN, row_h / PAGE_H_IN]
    cax2 = fig.add_axes(cax2_rect)
    fig.colorbar(im2, cax=cax2, label=plot_style.label("Prior Fractional Share", None))

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
        dtab, idx_median, panels, share, share_smooth, at_floor, winner_idx, n_winner = \
            _build_region_data(config, region)
        _print_numbers(region, dtab, idx_median, panels, share, share_smooth, at_floor, winner_idx, n_winner)
        paths = _draw_figure(config, region, dtab, idx_median, panels, share_smooth)
        row1_panels = [p for (i, _c), p in panels.items() if i == 0]
        mass_min = min(p["mass"] for p in row1_panels)
        mass_max = max(p["mass"] for p in row1_panels)
        on_grid_min = min(p["on_grid"] for p in row1_panels)
        on_grid_max = max(p["on_grid"] for p in row1_panels)
        st.done(paths[0], n_panel=len(row1_panels), mass_min=mass_min, mass_max=mass_max,
                on_grid_min=on_grid_min, on_grid_max=on_grid_max,
                old_floor_cell_fraction=float(np.mean(at_floor)))
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
