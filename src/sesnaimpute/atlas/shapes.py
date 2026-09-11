"""The prior at a source, one page per region (SPEC_BMSTP_DRAFT.md sec.
1.1's factorisation, sec. 1.4's template weights, sec. 2's common grid,
sec. 4.1-4.2, sec. 5.1-5.6). Report-only: nothing written here is read
by the fitter or by any other `bmstp`/`fittp` stage. The vocabulary and
the two rows' probability statements are `atlas.captions`'s, imported
and printed on the page, never restated here.

Per region, ONE source: the one at the region's median `A_COL_K`
(`_select_source`). Six panels per row (GAL, YSO, H2S, STAR, PAHC, AGB),
sharing ONE pair of axes, `log10 x` and `log10 F_4.5` in mJy (sec. 2: one
common brightness axis for every class, H2S included since its template
conversion folded in at the shape stage).

Row 1 draws `P(C, cell | s) = Lambda_C(cell) / sum over classes and cells
of Lambda(cell)`, the joint probability that the source is of class C
AND lies in that cell, on ONE log colour scale shared by all six panels
(owner's ruling 2026-09-11): the class's intensity `A_C(s)` and template
weight `f_C` are multiplied into the shape before the panels are
compared, since a shape alone (unit mass) cannot be compared across
classes whose intensities differ by orders of magnitude.

Row 2 draws `P(C | cell, s) = Lambda_C(cell) / sum over classes of
Lambda(cell)`, the class share of the prior at that cell, on the LINEAR
0-1 scale, in EVERY cell, with no smoothing, no evidence count and no
footprint: the effective-count machinery (`N_EFF`, the smoothed share,
the per-cell winner) that used to stand in for a floor test is gone, in
favour of the one common floor below deciding what an empty cell means.

Both rows apply W66ae's rules at the read: (1) support -- `x = a/A_s <=
1` by definition, so a cell with `log10 x > 0` is outside the prior,
drawn blank (masked out of the colour scale) with the `x = 1` line
marked, and excluded from both rows' sums; `grid.py` does not yet expose
this support restriction as a reader-side rule (W66ae is being coded in
parallel), so it is applied here directly on the cell edges. (2) the
common floor -- today's stored shape is floored at `grid.FLOOR` of its
OWN peak, so an empty cell is decided by intensity alone; this page
instead floors the prior DENSITY `Lambda_C` at one value common to every
class, `Lambda_floor(s) = grid.FLOOR * max over classes and cells of
Lambda_C(cell; s)` (support cells only), computed locally here for the
same reason -- `grid.py` does not yet expose it either. Where every
class was below this common floor the six classes are now exactly
equal, one sixth apiece, and the likelihood is left to decide.

The colourbars keep W66p's labels through `plot_style.label`, updated to
the rows' new quantities; the page prints, below the rows,
`captions.SHAPES_ROW1`, `captions.SHAPES_ROW2` and
`captions.vocabulary_block()` in full, wrapped to the page width, with
the page grown to fit them.
"""

import argparse
import os
import textwrap

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
from sesnaimpute.population import pahc_curve
from sesnaimpute.atlas import captions

PAGE_W_IN = 16.0

#: The old page's panel order (W18's brief).
CLASS_ORDER = ("GAL", "YSO", "H2S", "STAR", "PAHC", "AGB")

#: Every class reads the one common brightness axis (sec. 2, H2S included
#: ): `log10 F_4.5` in mJy, `bmstp.grid.LOG10_F45_EDGES` -- the
#: row's one outside y-axis label below.
_SHARED_Y_LABEL = plot_style.label("log10 F_4.5", "mJy")

_ARM_NAME = {0: "Herschel", 1: "Planck"}

#: The caption block's own type size and wrap width, chosen so the
#: wrapped lines stay well inside the page's usable width at this font
#: (a generous under-estimate of the page's own character capacity, so
#: the block never overruns the page horizontally) -- the page instead
#: grows in the one free dimension, its height, to fit the line count
#: this font and width produce.
_CAPTION_FONTSIZE = 7.5
_CAPTION_LINESPACING = 1.3
_CAPTION_CHARS_PER_LINE = 160
_CAPTION_LINE_HEIGHT_IN = _CAPTION_FONTSIZE * _CAPTION_LINESPACING / 72.0
_CAPTION_TOP_PAD_IN = 0.20
_CAPTION_BOTTOM_PAD_IN = 0.15


def _select_source(a_col):
    """The catalogued source closest to the region's median `A_COL_K`
    -- the one source this page draws (module docstring)."""
    return int(np.argmin(np.abs(a_col - np.median(a_col))))


def _read_density_table(config, region):
    """P1's per-source rows this page needs: `NAME`, `A_COL_K`, `ARM`,
    the star-family/cloud grain indices, `D_PAHC` (sec. 4.1, row 1's
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


def _class_on_grid(cls, dtab, on_grid_star, on_grid_agb, on_grid_yso, on_grid_h2s, on_grid_gal, src_idx):
    if cls in ("STAR", "PAHC"):
        return float(on_grid_star[dtab["tile"][src_idx]])
    if cls == "AGB":
        return float(on_grid_agb[dtab["tile"][src_idx]])
    if cls == "YSO":
        return float(on_grid_yso[dtab["sightline"][src_idx]])
    if cls == "H2S":
        return float(on_grid_h2s[dtab["sightline"][src_idx]])
    return float(on_grid_gal)  # GAL


def _panel_shape(config, region, cls, idx_median):
    """`(density, mass, reader)`: the shape `h_C` the fitter reads for
    the median source -- `fittp.prior_reader.load`/`prepare`, which
    blurs the class's stored grain shape by the source's own column
    kernel (sec. 4.2), floors it at its OWN peak (the per-class floor
    the common floor below replaces at the read) and renormalises it to
    sum to one -- converted from a per-cell mass to a density per dex^2
    (`/ (dlx * dlb)`, both axes `log10`, sec. 2). `mass` is `prepare`'s
    own post-blur sum over the WHOLE grid (sec. 9's "shape normalisation
    after blur", 1 +/- 1e-3 by construction): a blur-identity check,
    distinct from the support restriction the read below applies."""
    reader = prior_reader.load(config, region, cls)
    h = prior_reader.prepare(reader, np.array([idx_median]))[0]
    density = h.astype(np.float64) / (reader.dlx * reader.dlb)
    mass = float(density.sum() * reader.dlx * reader.dlb)
    return density, mass, reader


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
    at the query brightness `F`, since this page has no fitted `a_hat`/
    slope to place a* away from F) -- `population.pahc_curve.read`,
    imported not re-derived, not the stored/floored P5 table (which
    fixes the argument at F, sec. 1.4, wrong for the per-template shift
    this needs)."""
    type_factor, contrast_factor = reader.factors[0], reader.factors[1]
    pi_theta_f = type_factor["W"]          # (n_model, n_b), sum_theta = 1 at every F
    c_theta = reader.c_theta               # (n_model,)
    c_f = contrast_factor["C_F"]           # (n_model,)
    arg = panel_b_centers[None, :] - c_theta[:, None] + c_f[:, None] + d_pahc_s
    p_val = curve(-arg)                    # curve's own x-axis is log10 q, arg is -log10 q
    term = (1.0 - p_val) if cls == "STAR" else p_val
    return (pi_theta_f * term).sum(axis=0)


def _build_region_data(config, region):
    """Reads the median source's per-class shapes, forms `Lambda_C =
    A_C(s) h_C f_C(F; s)` for every class (sec. 1.1/1.4, module
    docstring), applies W66ae's support rule and common floor at the
    read, and returns the two rows' probabilities plus the row-1 corner
    numbers."""
    dtab = _read_density_table(config, region)
    idx_median = _select_source(dtab["a_col"])
    on_grid_star, on_grid_agb = _on_grid_star(config, region)
    on_grid_yso = _on_grid_yso(config, region)
    on_grid_h2s = _on_grid_h2s(config, region)
    on_grid_gal = _on_grid_gal(config)
    d_pahc_s = float(dtab["d_pahc"][idx_median])
    curve = pahc_curve.read(config)

    lam = {}
    mass_c = {}
    on_grid_c = {}
    intensity_c = {}
    x_edges = b_edges = None
    for cls in CLASS_ORDER:
        density, mass, reader = _panel_shape(config, region, cls, idx_median)
        x_edges, b_edges = reader.x_edges, reader.b_edges
        b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
        if cls in ("STAR", "PAHC"):
            f_c = _factor_marginal(b_centers, reader, cls, d_pahc_s, curve)
        else:
            f_c = np.ones(density.shape[1], dtype=np.float64)
        intensity_c[cls] = float(dtab[cls][idx_median])
        lam[cls] = intensity_c[cls] * density * f_c[None, :]
        mass_c[cls] = mass
        on_grid_c[cls] = _class_on_grid(cls, dtab, on_grid_star, on_grid_agb, on_grid_yso, on_grid_h2s,
                                         on_grid_gal, idx_median)

    # W66ae rule 1, the support: `x = a / A_s <= 1` by definition, so a
    # cell whose LEFT edge already sits at or past `log10 x = 0` is
    # entirely outside the prior (the edge convention that places a mark
    # exactly at `x = 1` in the cell below it, `bmstp.grid.bin`, leaves
    # cell 95 -- the last cell below the edge -- as the last cell IN the
    # support). Both rows below sum over `support` only; the excluded
    # cells are drawn blank (`_draw_figure`).
    outside = x_edges[:-1] >= 0.0
    support = ~outside

    # W66ae rule 2, the one common floor, formed locally (module
    # docstring): `Lambda_floor(s) = grid.FLOOR * max` over classes and
    # SUPPORT cells of the raw `Lambda_C`, then every class's `Lambda_C`
    # is floored at this one shared value -- a cell where every class
    # was below it now reads exactly equal across classes.
    lambda_stack = np.stack([lam[cls] for cls in CLASS_ORDER])
    lambda_floor = float(grid.FLOOR * lambda_stack[:, support, :].max())
    lam_floored = {cls: np.where(support[:, None], np.maximum(lam[cls], lambda_floor), 0.0)
                   for cls in CLASS_ORDER}
    all_below_floor = np.all(lambda_stack <= lambda_floor, axis=0) & support[:, None]
    floor_fraction = float(all_below_floor.sum()) / float(support.sum() * lambda_stack.shape[2])

    # Row 1 (module docstring): `P(C, cell | s)`, the joint over all six
    # classes and every support cell, summing to 1 over that whole set.
    total_lambda = sum(lam_floored[cls] for cls in CLASS_ORDER)
    grand_total = float(total_lambda[support, :].sum())
    joint = {cls: lam_floored[cls] / grand_total for cls in CLASS_ORDER}

    # Row 2 (module docstring): `P(C | cell, s)`, the class share of
    # `Lambda` at each cell -- `denom` is never zero on the support,
    # since every class there is floored at `lambda_floor > 0`.
    denom = np.where(support[:, None], total_lambda, 1.0)
    share = {cls: lam_floored[cls] / denom for cls in CLASS_ORDER}

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    p_total = {}
    peak = {}
    for cls in CLASS_ORDER:
        p_total[cls] = float(joint[cls][support, :].sum())
        masked = np.where(support[:, None], joint[cls], -np.inf)
        pi, pj = np.unravel_index(int(np.argmax(masked)), masked.shape)
        peak[cls] = (float(x_centers[pi]), float(b_centers[pj]))

    return dict(dtab=dtab, idx_median=idx_median, x_edges=x_edges, b_edges=b_edges,
                support=support, joint=joint, share=share, mass_c=mass_c, on_grid_c=on_grid_c,
                intensity_c=intensity_c, p_total=p_total, peak=peak,
                lambda_floor=lambda_floor, floor_fraction=floor_fraction)


def _print_numbers(region, data):
    dtab, idx_median = data["dtab"], data["idx_median"]
    name = dtab["name"][idx_median].decode("utf-8")
    print("atlas.shapes [%s] median source %s: A_COL_K=%.4g mag arm=%s Lambda_floor(s)=%.4g "
          "floor_cell_fraction=%.4f"
          % (region, name, dtab["a_col"][idx_median], _ARM_NAME[int(dtab["arm"][idx_median])],
             data["lambda_floor"], data["floor_fraction"]))
    for cls in CLASS_ORDER:
        px, pb = data["peak"][cls]
        print("atlas.shapes [%s] %s: mass=%.6f on_grid=%.6f A_C(s)=%.4g deg^-2 "
              "P(C|s)=%.6f peak_log10x=%.4g peak_log10F45=%.4g"
              % (region, cls, data["mass_c"][cls], data["on_grid_c"][cls], data["intensity_c"][cls],
                 data["p_total"][cls], px, pb))


def _caption_block():
    """The page's caption, wrapped to `_CAPTION_CHARS_PER_LINE`: the two
    rows' probability statements and the shared vocabulary, read from
    `captions` and never restated here (CODING_RULES_BMSTP.md rule 4).
    Returns `(text, n_lines)` -- `n_lines` is what `_draw_figure` grows
    the page's height by, so the block never overlaps the rows above it."""
    sections = [textwrap.wrap(captions.SHAPES_ROW1, width=_CAPTION_CHARS_PER_LINE),
                textwrap.wrap(captions.SHAPES_ROW2, width=_CAPTION_CHARS_PER_LINE)]
    vocab_lines = []
    for line in captions.vocabulary_block().split("\n"):
        vocab_lines.extend(textwrap.wrap(line, width=_CAPTION_CHARS_PER_LINE) or [""])
    sections.append(vocab_lines)
    text = "\n\n".join("\n".join(sec) for sec in sections)
    n_lines = sum(len(sec) for sec in sections) + 2 * (len(sections) - 1)
    return text, n_lines


def _draw_figure(config, region, data):
    plot_style.apply_style()
    dtab, idx_median = data["dtab"], data["idx_median"]
    x_edges, b_edges, support = data["x_edges"], data["b_edges"], data["support"]
    joint, share = data["joint"], data["share"]

    caption_text, caption_lines = _caption_block()
    caption_block_h = (caption_lines * _CAPTION_LINE_HEIGHT_IN
                        + _CAPTION_TOP_PAD_IN + _CAPTION_BOTTOM_PAD_IN)

    margin_l, margin_r, margin_t = 0.55, 1.05, 0.85
    row_gap, col_gap = 0.90, 0.14
    row_h = 3.2
    page_h = margin_t + 2 * row_h + row_gap + caption_block_h
    page_w = PAGE_W_IN
    usable_w = page_w - margin_l - margin_r
    shape_w = (usable_w - 5 * col_gap) / 6.0

    fig = plot_style.new_sized_figure(page_w, page_h)

    # Row 1's colour scale (module docstring): one log norm over the
    # joint `P(C, cell | s)`, spanning all six classes and every SUPPORT
    # cell (the excluded cells never enter the norm, since they carry no
    # probability). Row 2's colour scale is fixed, linear 0-1 (module
    # docstring: the class share is a probability, not a density).
    log_joint = {cls: np.where(support[:, None], np.log10(np.where(support[:, None], joint[cls], 1.0)), np.nan)
                 for cls in CLASS_ORDER}
    all_log = np.concatenate([log_joint[cls][support, :].ravel() for cls in CLASS_ORDER])
    norm1 = Normalize(vmin=float(all_log.min()), vmax=float(all_log.max()))
    norm2 = Normalize(vmin=0.0, vmax=1.0)
    cmap1 = plt.get_cmap("viridis").copy()
    cmap1.set_bad("white")
    cmap2 = plt.get_cmap("viridis").copy()
    cmap2.set_bad("white")
    share_masked = {cls: np.where(support[:, None], share[cls], np.nan) for cls in CLASS_ORDER}
    im1 = im2 = None

    src_idx = idx_median
    a_col_src = dtab["a_col"][src_idx]
    for i in range(2):
        y0 = page_h - margin_t - (i + 1) * row_h - i * row_gap
        for c, cls in enumerate(CLASS_ORDER):
            x0 = margin_l + c * (shape_w + col_gap)
            ax = fig.add_axes([x0 / page_w, y0 / page_h, shape_w / page_w, row_h / page_h])
            extent = [x_edges[0], x_edges[-1], b_edges[0], b_edges[-1]]
            if i == 0:
                im1 = ax.imshow(log_joint[cls].T, origin="lower", aspect="auto",
                                 extent=extent, cmap=cmap1, norm=norm1)
                ax.text(0.02, 0.03, "mass=%.4f\non_grid=%.4f\narea density = %.3g deg^-2"
                        % (data["mass_c"][cls], data["on_grid_c"][cls], data["intensity_c"][cls]),
                        transform=ax.transAxes, fontsize=5.5, color="white", va="bottom")
            else:
                im2 = ax.imshow(share_masked[cls].T, origin="lower", aspect="auto",
                                 extent=extent, cmap=cmap2, norm=norm2)
            # W66ae's support boundary, `log10 x = 0` (`x = 1`), marked
            # on every panel of both rows.
            ax.axvline(0.0, color="white", lw=0.7, linestyle="--", alpha=0.85)
            ax.set_xlabel("log10 x", fontsize=6.5)
            ax.tick_params(labelsize=6)
            ax.set_xticks(np.array([-3.0, -1.0, 1.0]))
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            log_a_col = np.log10(a_col_src)
            xlim = ax.get_xlim()
            k_lo = int(np.ceil(xlim[0] + log_a_col))
            k_hi = int(np.floor(xlim[1] + log_a_col))
            decades = np.arange(k_lo, k_hi + 1)
            ax2.set_xticks(decades - log_a_col)
            ax2.set_xticklabels(["$10^{%d}$" % k for k in decades], fontsize=5)
            ax2.tick_params(length=2, pad=1, labelsize=5)
            if c == 0:
                ax2.text(1.0, 1.05, plot_style.label("a", "mag"), transform=ax2.transAxes,
                         fontsize=5.5, ha="right", va="bottom")
            ax.set_title(cls, fontsize=8.5 if i == 0 else 9, pad=16)
            if c == 0:
                fig.text(x0 / page_w - 0.30 / page_w, (y0 + 0.5 * row_h) / page_h,
                          _SHARED_Y_LABEL, rotation=90, va="center", ha="center", fontsize=7)

    cax1_rect = [(margin_l + 6 * shape_w + 5 * col_gap + 0.15) / page_w,
                 (page_h - margin_t - row_h) / page_h, 0.22 / page_w, row_h / page_h]
    cax1 = fig.add_axes(cax1_rect)
    fig.colorbar(im1, cax=cax1, label=plot_style.label("P(C, cell | s)", None))

    cax2_rect = [(margin_l + 6 * shape_w + 5 * col_gap + 0.15) / page_w,
                 (caption_block_h) / page_h, 0.22 / page_w, row_h / page_h]
    cax2 = fig.add_axes(cax2_rect)
    fig.colorbar(im2, cax=cax2, label=plot_style.label("P(C | cell, s)", None))

    r = regions_module.REGIONS_BY_NAME[region]
    name_med = dtab["name"][idx_median].decode("utf-8")
    arm_med = _ARM_NAME[int(dtab["arm"][idx_median])]
    caption = (
        "%s -- median source %s (A_COL_K=%.3g mag, %s arm); "
        "d_r=%.0f+/-%.0f pc; KAPPA_HERSCHEL=%.1f, KAPPA_PLANCK=%.1f"
        % (region, name_med, dtab["a_col"][idx_median], arm_med,
           r.d_r_pc, r.sigma_pc, dtab["kappa_herschel"], dtab["kappa_planck"]))
    fig.suptitle(caption, fontsize=9.5, y=1.0 - 0.15 / page_h)

    # The page's own vocabulary and row statements (module docstring),
    # anchored so their top line sits just below the rows -- the page
    # was grown, above, by exactly the height this block needs.
    fig.text(margin_l / page_w, (caption_block_h - _CAPTION_TOP_PAD_IN) / page_h, caption_text,
              fontsize=_CAPTION_FONTSIZE, va="top", ha="left", linespacing=_CAPTION_LINESPACING)

    out_dir = f"{config.data_root}/bmstp/atlas/figures"
    paths = []
    for fmt in ("png", "pdf"):
        paths.append(f"{out_dir}/prior-shapes_{region}.{fmt}")
    return fig, paths


def build_region(config, region):
    with progress.Stage("atlas.shapes", region) as st:
        data = _build_region_data(config, region)
        _print_numbers(region, data)
        fig, paths = _draw_figure(config, region, data)
        os.makedirs(f"{config.data_root}/bmstp/atlas/figures", exist_ok=True)
        for path in paths:
            fig.savefig(path, dpi=150)
        plt.close(fig)
        mass_min = min(data["mass_c"].values())
        mass_max = max(data["mass_c"].values())
        on_grid_min = min(data["on_grid_c"].values())
        on_grid_max = max(data["on_grid_c"].values())
        st.done(paths[0], n_panel=len(CLASS_ORDER), mass_min=mass_min, mass_max=mass_max,
                on_grid_min=on_grid_min, on_grid_max=on_grid_max,
                lambda_floor=data["lambda_floor"], floor_cell_fraction=data["floor_fraction"])
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
