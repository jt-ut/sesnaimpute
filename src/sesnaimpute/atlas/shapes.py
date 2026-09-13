r"""The prior at a source, one page per region (SPEC_BMSTP_DRAFT.md sec.
1.1's factorisation, sec. 1.4's template weights, sec. 2's common grid,
sec. 4.1-4.2, sec. 5.1-5.6). Report-only: nothing written here is read
by the fitter or by any other `bmstp`/`fittp` stage. The vocabulary and
the two rows' probability statements are `atlas.captions`'s, imported
and printed on the page, never restated here.

Per region, ONE source: the one at the region's median `A_COL_K`
(`_select_source`). Row 2 carries the six class panels (GAL, YSO, H2S,
STAR, PAHC, AGB); row 1 carries ONE panel, the total over classes, in
the first column, the other five slots empty. Every panel shares ONE
pair of axes, `log10 ξ` (from -3.0 to the median source's own
blurred read's last populated cell, never less than the edge ξ = 1 at
`log10 ξ = 0`, drawn dashed on every panel, sec. 2, 4.2) and
`log10 F_4.5` in mJy (sec. 2: one common brightness axis for
every class, H2S included since its template conversion folded in at
the shape stage), the bold y-axis label set as the axes' own ylabel.
Because the six panels of a row share that one brightness axis, only the
row's LEFTMOST panel carries y tick labels: a repeated set overhangs the
narrow column gap and reads as stray small numbers inside the panel to
its left. The
title is the region and what the page is; the median source's own name,
sightline column `A_s`, arm and distance sit in the caption block
instead.

Row 1 draws `P(cell | s) = sum over classes of Lambda_C(cell) / sum over
classes and cells of Lambda(cell)`, the probability that the source lies
in that cell AT ALL, summed over the six classes -- where the prior
expects a source at this position -- on the same log colour scale the
six class panels below would otherwise share: each class's intensity
`A_C(s)` and template weight `f_C` are multiplied into its shape before
the classes are summed, since a shape alone (unit mass) cannot be
compared, or added, across classes whose intensities differ by orders of
magnitude.

Row 2 draws `P(C | cell, s) = Lambda_C(cell) / sum over classes of
Lambda(cell)`, the class share of the prior at that cell, on the LINEAR
0-1 scale, in EVERY cell, with no smoothing, no evidence count and no
footprint: the one common floor below is what decides what an empty
cell means. Each class panel is drawn at that share's own colour and an
opacity set by the six shares' entropy -- solid where the prior decides
the class, faded to white (the axes' own background) where it is blind,
every class at the common floor and the six shares equal.

Both rows apply two rules at the read. (1) Support -- `x = a/A_s <= 1` by
definition, so a cell with `log10 ξ > 0` is outside the prior's SUPPORT
and excluded from both rows' SUMS (the normalisation, the floor search,
the printed peaks); it is drawn, since the read
(`fittp.prior_reader`'s own change of variables, `lambda_grids`'s
`blur=True`) carries a real, non-zero density there -- a pencil column
above the beam mean, in the measured coordinate's one-dex padding, never
folded back -- so both rows draw it, past the `x = 1` line marked on
every panel. (2) One common floor on the prior DENSITY `Lambda_C`,
common to every class at the source rather than per class:
`Lambda_floor(s) = grid.FLOOR * max over classes and SUPPORT cells of
Lambda_C(cell; s)`; where every class was below it the six classes read
exactly equal, one sixth apiece, and the likelihood is left to decide --
applied over the WHOLE array, support and padding alike, so an empty
padding cell reads the same floor too. The support is
`bmstp.grid.N_XI_SUPPORT`, the floor `fittp.prior_reader.common_floor` --
the same two definitions the fitter's read uses, imported, never
restated.

This page draws the MEASURED depth fraction, the fitter's own
$\hat\xi$ (owner's ruling): its x-axis title reads `$log_{10}\,\hat\xi$`
(row 2 only -- row 1's one panel carries no x-axis title) and its
colourbars read `$P(\hat\xi, F_{4.5} \mid s)$` (row 1, inset at its own
panel's right edge) and `$P(C \mid \hat\xi, F_{4.5}, s)$` (row 2, its
scale and position unchanged). `_XI_LABEL` and its own colourbar wording
stay the region page's, in the true depth fraction `x`, until a
package-wide rename. The page prints, below the rows, in order,
`captions.SHAPES_ROW1` (the top panel), `captions.SHAPES_ROW2` (the
class share, its fade rule folded in), `captions.SHAPES_MEASURED` (what
$\hat\xi$ is and why mass can lie past 1) and the source's own distance
line -- no vocabulary block. The
page's height grows by that block at the line pitch matplotlib actually
sets for it (`_CAPTION_LINE_HEIGHT_IN`, above the nominal
fontsize * linespacing because of the mathtext lines) plus a top and a
bottom margin, so its last line sits inside the page rather than under
the bottom edge.
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.fittp import prior_reader
from sesnaimpute.bmstp import grid
from sesnaimpute.bmstp import template_weights
from sesnaimpute.population import pahc_curve
from sesnaimpute.atlas import captions

PAGE_W_IN = 16.0

#: The page's panel order.
CLASS_ORDER = ("GAL", "YSO", "H2S", "STAR", "PAHC", "AGB")

#: Every class reads the one common brightness axis (sec. 2, H2S included
#: ): `log10 F_4.5` in mJy, `bmstp.grid.LOG10_F45_EDGES` -- the row's one
#: outside y-axis label below, real mathtext subscripts throughout, bold
#: by `plot_style.apply_style`'s own `axes.labelweight` since it is set
#: as the axes' own ylabel, not a plain figure text.
#: `axes.labelweight` (`plot_style.apply_style`) bolds a `set_xlabel`/
#: `set_ylabel` Text artist whole, but not the mathtext runs inside it
#: (matplotlib does not carry a font-weight rcParam into mathtext), so
#: the math itself is wrapped in `\mathbf{...}` here to read bold too --
#: the plain unit suffix (e.g. `[mJy]`) still gets its bold from the
#: rcParam alone.
_SHARED_Y_LABEL = plot_style.label(r"$\mathbf{log_{10}\,F_{4.5}}$", "mJy")
_XI_LABEL = r"$\mathbf{log_{10}\,\xi}$"
#: This page's own x-axis title (owner's ruling, item 3): the page draws
#: the MEASURED depth fraction the fitter reads, $\hat\xi$, distinct from
#: `_XI_LABEL` above, the region page's true depth fraction -- untouched
#: until the package-wide rename.
_XI_LABEL_MEASURED = r"$\mathbf{log_{10}\,\hat{\xi}}$"

_ARM_NAME = {0: "Herschel", 1: "Planck"}

#: Panel titles, axis labels and colourbar labels at a size a reader
#: sees on a slide; tick labels smaller; the vocabulary block stays at
#: its own small size (`_CAPTION_FONTSIZE` below).
_LABEL_FONTSIZE = 13
_TICK_FONTSIZE = 10

#: The page title and its one-line subtitle (owner's list): bigger than
#: the panel titles, since a reader meets the page here first.
_TITLE_FONTSIZE = 18
_SUBTITLE_FONTSIZE = 15

#: The array's own low edge (sec. 2): the panel axis never extends
#: further left than this.
_LOG10_XI_MIN = -3.0
#: THE EDGE ξ = 1 (sec. 2, 4.2): drawn dashed on every panel. The panel's own
#: `log10 ξ` axis extends at least this far right, and further where the
#: median source's own blurred read still carries a non-negligible share
#: of some class's mass past it -- a pencil column above the beam mean,
#: real in the measured coordinate and never folded back (`_panel_xi_max`).
_LOG10_XI_EDGE = 0.0

#: The caption block's own type size and wrap width, chosen so the
#: wrapped lines stay well inside the page's usable width at this font
#: (a generous under-estimate of the page's own character capacity, so
#: the block never overruns the page horizontally) -- the page instead
#: grows in the one free dimension, its height, to fit the line count
#: this font and width produce.
_CAPTION_FONTSIZE = 8.0
_CAPTION_LINESPACING = 1.3
_CAPTION_CHARS_PER_LINE = 160
#: The pitch matplotlib actually sets for these lines, inches, measured
#: from the rendered block: more than `fontsize * linespacing / 72`,
#: because a line carrying mathtext ($F_{4.5}$) is taller than a plain
#: one and matplotlib leads the next line from that taller extent.
_CAPTION_LINE_HEIGHT_IN = 0.156
_CAPTION_TOP_PAD_IN = 0.20
#: The page's bottom margin: clear white below the block's last line.
_CAPTION_BOTTOM_PAD_IN = 0.25

#: Room, inches, for row 2's own x-axis tick labels and "log10 ξ" label
#: below its panels, ahead of the caption strip.
AXIS_LABEL_MARGIN_IN = 0.35


def _select_source(a_col):
    """The catalogued source closest to the region's median `A_COL_K`
    -- the one source this page draws (module docstring)."""
    return int(np.argmin(np.abs(a_col - np.median(a_col))))


def _read_density_table(config, region):
    """P1's per-source rows this page needs: `NAME`, `A_COL_K` (the
    source's own sightline column `A_s`), `ARM`, the star-family/cloud
    grain indices, `D_PAHC` (sec. 4.1, row 1's `f_STAR`/`f_PAHC` read
    point), and the six `DENSITY_<C>` (sec. 4.1)."""
    path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    with h5py.File(path, "r") as f:
        d = dict(
            name=f["NAME"][:], a_col=f["A_COL_K"][:].astype(np.float64),
            arm=f["ARM"][:].astype(np.int64), tile=f["TILE"][:].astype(np.int64),
            sightline=f["SIGHTLINE_ROW"][:].astype(np.int64),
            d_pahc=f["D_PAHC"][:].astype(np.float64),
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
    kernel (sec. 4.2) and renormalises it to sum to one -- converted
    from a per-cell mass to a density per dex^2 (`/ (dlx * dlb)`, both
    axes `log10`, sec. 2). `mass` is `prepare`'s own post-blur sum over
    the WHOLE grid (sec. 9's "shape normalisation after blur", 1 +/-
    1e-3 by construction): a blur-identity check, distinct from the
    support restriction the read below applies."""
    reader = prior_reader.load(config, region, cls)
    h = prior_reader.prepare(reader, np.array([idx_median]))[0]
    density = h.astype(np.float64) / (reader.dlx * reader.dlb)
    mass = float(density.sum() * reader.dlx * reader.dlb)
    return density, mass, reader


#: `_factor_marginal`'s own (n_query, n_model, n_b) working set for
#: STAR/PAHC (arg, p_val, term, the `pi_theta_f * term` product) -- the
#: same 512 MB budget and array count `bmstp.atlas._above_factor_batch_size`
#: applies to the identical call, so many rows at once (rule 10b) rather
#: than the whole `rows` axis, which for a template register of several
#: thousand can otherwise exceed the machine's ceiling.
_LAMBDA_BATCH_BUDGET_BYTES = 512 * 1024 * 1024
_N_FACTOR_TEMP_ARRAYS = 4


def _lambda_batch_size(n_model, n_b):
    row_bytes = max(1, n_model) * n_b * 8 * _N_FACTOR_TEMP_ARRAYS
    return max(1, _LAMBDA_BATCH_BUDGET_BYTES // row_bytes)


def _prepare_unblurred(reader, rows):
    """`h (n_block, n_x, n_b)` float32: the same construction as
    `prior_reader.prepare` with no column-kernel blur -- each row's own
    raw, STORED grain shape (the DISTANCE coordinate, `x = a / A_s`, zero
    above the edge ξ = 1 by construction, `bmstp.grid`'s module docstring)
    renormalised to sum to one over the whole array, exactly as
    `prepare` renormalises its own blurred result (`blur=False`, this
    brief's item 1)."""
    grain = reader.grain[rows]
    raw = reader.grid_all[grain].astype(np.float64)
    totals = raw.reshape(raw.shape[0], -1).sum(axis=1)
    safe = np.where(totals > 0.0, totals, 1.0)
    h = np.where(totals[:, None, None] > 0.0, raw / safe[:, None, None], raw)
    return h.astype(np.float32)


def lambda_grids(config, region, rows, blur=True):
    """`(Lambda (n_rows, 6, n_x, n_b) float64, lambda_floor (n_rows,),
    class_order)`: every row's own six-class prior density `Lambda_C(cell;
    s) = A_C(s) h_C(cell; s) f_C(F; s)` (sec. 1.1/1.4), floored at the one
    common floor per row (`prior_reader.common_floor`, the common-floor
    rule) -- the construction `_build_region_data` used inline for the
    page's one median source, lifted here so a caller can form it for many
    rows at once (`prior_reader.prepare` already batches over rows, rule
    8: no python loop over sources, only over the six classes and, for
    STAR/PAHC, over row BATCHES sized to `_lambda_batch_size`, rule 10b).

    `blur=True` (default) reads `h_C` through `prior_reader.prepare`, the
    fitter's own column-kernel blur into the MEASURED coordinate, where a
    class's mass past `log10 ξ = 0` is real (a pencil column above the
    beam mean, `fittp.prior_reader` module docstring) and stays in the
    returned array, over its WHOLE extent, not folded back or dropped.
    `blur=False` reads the UNBLURRED stored shape instead
    (`_prepare_unblurred`): the DISTANCE coordinate, zero above the edge ξ = 1
    by construction, no read-side change of variables. Either way the one
    common floor (below) is applied over the whole array, so an empty
    cell reads the same floor for every class whether or not it sits past
    the edge ξ = 1."""
    rows = np.asarray(rows)
    n_rows = rows.size
    dtab = _read_density_table(config, region)
    d_pahc_rows = dtab["d_pahc"][rows].astype(np.float64).reshape(-1, 1, 1)
    curve = pahc_curve.read(config)

    lam = {}
    for cls in CLASS_ORDER:
        reader = prior_reader.load(config, region, cls)
        b_centers = 0.5 * (reader.b_edges[:-1] + reader.b_edges[1:])
        if cls in ("STAR", "PAHC"):
            batch = _lambda_batch_size(reader.c_theta.size, b_centers.size)
        else:
            batch = n_rows if n_rows else 1
        intensity = dtab[cls][rows].astype(np.float64)
        parts = []
        for start in range(0, n_rows, batch):
            sl = slice(start, start + batch)
            h = prior_reader.prepare(reader, rows[sl]) if blur else _prepare_unblurred(reader, rows[sl])
            density = h.astype(np.float64) / (reader.dlx * reader.dlb)  # (n_batch, n_x, n_b)
            if cls in ("STAR", "PAHC"):
                f_c = template_weights._factor_marginal(b_centers, reader, cls, d_pahc_rows[sl], curve)
            else:
                f_c = np.ones((density.shape[0], density.shape[-1]), dtype=np.float64)
            parts.append(intensity[sl, None, None] * density * f_c[:, None, :])
        lam[cls] = np.concatenate(parts, axis=0) if parts else np.empty((0,) + density.shape[1:])

    # The one common floor, the fitter's own rule (`prior_reader.
    # common_floor`): `Lambda_floor(s)` per row from the six classes' own
    # peaks over the SUPPORT cells of the raw `Lambda_C` -- the search
    # stays support-only regardless of `blur` (the unblurred read is zero
    # in the padding by construction, so widening the search there would
    # cost nothing but would break comparability with the fitter's own
    # floor definition).
    support = np.zeros(grid.LOG10_XI_EDGES.size - 1, dtype=bool)
    support[:grid.N_XI_SUPPORT] = True
    peaks = [lam[cls][:, support, :].reshape(n_rows, -1).max(axis=1) for cls in CLASS_ORDER]
    lambda_floor = prior_reader.common_floor(peaks)  # (n_rows,)
    # The floor now applies over the WHOLE array:
    # the padding past the edge ξ = 1 is real mass under `blur=True` and is
    # drawn, not zeroed; only the unblurred read is genuinely zero there,
    # so flooring it simply reproduces the common floor.
    lambda_all = np.stack(
        [np.maximum(lam[cls], lambda_floor[:, None, None]) for cls in CLASS_ORDER],
        axis=1)  # (n_rows, 6, n_x, n_b)
    return lambda_all, lambda_floor, CLASS_ORDER


def _panel_xi_max(xi_edges, densities):
    """The panels' own `log10 ξ` upper limit (module docstring): never
    less than THE EDGE ξ = 1 (`_LOG10_XI_EDGE`), and no further right than the
    last cell, across the six classes' own blurred reads (`_panel_shape`),
    that still carries a non-negligible share of that class's own mass --
    `grid.FLOOR` of its own peak row, the same relative floor the read
    applies everywhere else."""
    last_idx = -1
    for density in densities:
        row_mass = density.sum(axis=1)
        peak = float(row_mass.max())
        if peak <= 0.0:
            continue
        above = np.nonzero(row_mass > grid.FLOOR * peak)[0]
        if above.size:
            last_idx = max(last_idx, int(above[-1]))
    if last_idx < 0:
        return _LOG10_XI_EDGE
    return max(_LOG10_XI_EDGE, float(xi_edges[last_idx + 1]))


def _build_region_data(config, region):
    """Reads the median source's per-class shapes, forms `Lambda_C =
    A_C(s) h_C f_C(F; s)` for every class (sec. 1.1/1.4, module
    docstring) through `lambda_grids`, applies the support rule and common
    floor at the read, and returns the two rows' probabilities, row 1's
    one total-over-classes panel, row 2's fade opacity, plus the row-1
    corner numbers."""
    dtab = _read_density_table(config, region)
    idx_median = _select_source(dtab["a_col"])
    on_grid_star, on_grid_agb = _on_grid_star(config, region)
    on_grid_yso = _on_grid_yso(config, region)
    on_grid_h2s = _on_grid_h2s(config, region)
    on_grid_gal = _on_grid_gal(config)

    mass_c = {}
    on_grid_c = {}
    intensity_c = {}
    densities = []
    xi_edges = b_edges = None
    for cls in CLASS_ORDER:
        density, mass, reader = _panel_shape(config, region, cls, idx_median)
        densities.append(density)
        xi_edges, b_edges = reader.xi_edges, reader.b_edges
        intensity_c[cls] = float(dtab[cls][idx_median])
        mass_c[cls] = mass
        on_grid_c[cls] = _class_on_grid(cls, dtab, on_grid_star, on_grid_agb, on_grid_yso, on_grid_h2s,
                                         on_grid_gal, idx_median)
    log10_xi_max = _panel_xi_max(xi_edges, densities)

    lambda_all, lambda_floor_rows, class_order = lambda_grids(config, region, np.array([idx_median]))
    lambda_floor = float(lambda_floor_rows[0])
    lam_floored = {cls: lambda_all[0, i] for i, cls in enumerate(class_order)}

    # The support: `x = a / A_s <= 1` by definition; the grid's own
    # `N_XI_SUPPORT` is the count of cells inside it (`bmstp.grid`). Both
    # rows below sum over `support` only; the excluded cells are drawn
    # blank (`_draw_figure`).
    support = np.zeros(xi_edges.size - 1, dtype=bool)
    support[:grid.N_XI_SUPPORT] = True

    # A cell is at the common floor for every class exactly where
    # `lambda_grids` clamped it there (`np.maximum` returns the floor
    # itself, bit for bit, whenever the raw density was at or below it).
    all_below_floor = np.all(
        np.stack([lam_floored[cls] for cls in CLASS_ORDER]) == lambda_floor, axis=0) & support[:, None]
    floor_fraction = float(all_below_floor.sum()) / float(support.sum() * lam_floored[CLASS_ORDER[0]].shape[1])

    # Row 1 (module docstring): `P(C, cell | s)`, the joint over all six
    # classes and every support cell, summing to 1 over that whole set.
    total_lambda = sum(lam_floored[cls] for cls in CLASS_ORDER)
    grand_total = float(total_lambda[support, :].sum())
    joint = {cls: lam_floored[cls] / grand_total for cls in CLASS_ORDER}

    # Row 2 (module docstring): `P(C | cell, s)`, the class share of
    # `Lambda` at each cell, over the WHOLE array (padding included) --
    # `denom` is never zero anywhere, since every
    # class at every cell is floored at `lambda_floor > 0`.
    denom = np.where(total_lambda > 0.0, total_lambda, 1.0)
    share = {cls: lam_floored[cls] / denom for cls in CLASS_ORDER}

    # Row 1's one panel (item 1): the total over classes -- exactly the
    # six `joint` arrays summed, since `P(cell | s) = sum over classes of
    # Lambda_C(cell) / sum over classes and cells of Lambda(cell)`.
    joint_total = sum(joint[cls] for cls in CLASS_ORDER)

    # Row 2's fade (item 2): per cell, over the WHOLE array (padding
    # included), the entropy of the six shares `p_C` and the opacity it
    # sets -- 1 where one class takes all, 0 where the six shares are
    # equal (every class at the common floor, the prior blind there).
    share_stack = np.stack([share[cls] for cls in CLASS_ORDER])  # (6, n_x, n_b)
    log_share = np.where(share_stack > 0.0, np.log(share_stack), 0.0)
    entropy = -(share_stack * log_share).sum(axis=0)
    alpha = 1.0 - entropy / np.log(6.0)
    alpha_low_fraction = float((alpha[support, :] < 0.05).sum()) / float(support.sum() * alpha.shape[1])

    xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    p_total = {}
    peak = {}
    for cls in CLASS_ORDER:
        p_total[cls] = float(joint[cls][support, :].sum())
        masked = np.where(support[:, None], joint[cls], -np.inf)
        pi, pj = np.unravel_index(int(np.argmax(masked)), masked.shape)
        peak[cls] = (float(xi_centers[pi]), float(b_centers[pj]))

    return dict(dtab=dtab, idx_median=idx_median, xi_edges=xi_edges, b_edges=b_edges,
                support=support, joint=joint, share=share, joint_total=joint_total, alpha=alpha,
                alpha_low_fraction=alpha_low_fraction, mass_c=mass_c, on_grid_c=on_grid_c,
                intensity_c=intensity_c, p_total=p_total, peak=peak,
                lambda_floor=lambda_floor, floor_fraction=floor_fraction, log10_xi_max=log10_xi_max)


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


def _caption_block(d_r_pc, sigma_pc):
    """The page's raw (unwrapped) caption, in order: the top-panel
    statement, the class-share statement (its own fade rule folded in),
    the measured-depth-fraction line and one distance line -- read from
    `captions` and never restated here (CODING_RULES_BMSTP.md rule 4)
    apart from the distance line itself. The median source's name/
    column/arm sit in the subtitle instead, and the vocabulary is off
    the page entirely (owner's list). Wrapping and the height it needs
    are `captions.caption_layout`'s, the one definition this page and
    `atlas.render`'s own caption strip both call."""
    distance_line = r"region distance $d_r$ = %.0f ± %.0f pc" % (d_r_pc, sigma_pc)
    return "\n\n".join([captions.SHAPES_ROW1, captions.SHAPES_ROW2, captions.SHAPES_MEASURED,
                         distance_line])


def _draw_figure(config, region, data):
    plot_style.apply_style()
    dtab, idx_median = data["dtab"], data["idx_median"]
    xi_edges, b_edges, support = data["xi_edges"], data["b_edges"], data["support"]
    joint, share = data["joint"], data["share"]
    joint_total, alpha = data["joint_total"], data["alpha"]
    log10_xi_max = data["log10_xi_max"]

    # The source's own name, sightline column and arm move out of the
    # title into the subtitle below it; the distance is the caption
    # block's one line instead; `KAPPA_*` is dropped entirely.
    r = regions_module.REGIONS_BY_NAME[region]
    name_med = dtab["name"][idx_median].decode("utf-8")
    arm_med = _ARM_NAME[int(dtab["arm"][idx_median])]
    subtitle_text = r"Source: %s ($\mathbf{A_s}$ = %.3g mag, %s)" % (name_med, dtab["a_col"][idx_median], arm_med)
    caption_text, caption_block_h = captions.caption_layout(
        _caption_block(r.d_r_pc, r.sigma_pc), _CAPTION_CHARS_PER_LINE,
        _CAPTION_LINE_HEIGHT_IN, _CAPTION_TOP_PAD_IN, _CAPTION_BOTTOM_PAD_IN)

    # `margin_t` clears, in order from the page's top edge: the title
    # (`_TITLE_FONTSIZE`), the one-line subtitle (`_SUBTITLE_FONTSIZE`),
    # then row 1's own panel titles (`_LABEL_FONTSIZE`, `pad=18` points)
    # -- more than the panel-titles-only margin this page used before
    # the subtitle existed.
    margin_l, margin_r, margin_t = 0.75, 1.05, 1.35
    row_gap, col_gap = 0.90, 0.14
    row_h = 3.2
    # Row 2's own x-axis tick labels and "log10 ξ" label draw BELOW its
    # axes at a fixed offset matplotlib chooses, not inside `row_h` --
    # without this margin the caption strip's own top edge would sit
    # exactly at row 2's bottom edge and those labels would overlap the
    # caption text. Reserved the same way `atlas.render`'s own
    # `MARGIN_BOTTOM_IN` reserves room for its bottom row's tick labels,
    # ahead of its own caption strip.
    page_h = margin_t + 2 * row_h + row_gap + AXIS_LABEL_MARGIN_IN + caption_block_h
    page_w = PAGE_W_IN
    usable_w = page_w - margin_l - margin_r
    shape_w = (usable_w - 5 * col_gap) / 6.0

    fig = plot_style.new_sized_figure(page_w, page_h)

    # Row 1's colour scale (module docstring): a LogNorm set from the
    # joint `P(C, cell | s)` over the SUPPORT cells alone (the printed
    # peaks' own range) -- both rows DRAW the whole read's extent
    # (a class's mass past the edge ξ = 1 is real under the blur), the colour scale
    # itself unchanged, so a padding cell simply reads on the same bar. The
    # one total panel below reads on this same scale, unchanged by the sum
    # (owner's ruling): a cell whose classes sum past `vmax` simply clips.
    all_vals = np.concatenate([joint[cls][support, :].ravel() for cls in CLASS_ORDER])
    norm1 = LogNorm(vmin=float(all_vals.min()), vmax=float(all_vals.max()))
    norm2 = Normalize(vmin=0.0, vmax=1.0)
    cmap1 = plt.get_cmap("viridis").copy()
    cmap1.set_bad("white")
    cmap2 = plt.get_cmap("viridis").copy()
    cmap2.set_bad("white")
    # Row 2's colourbar reads off this `ScalarMappable`, not a drawn image
    # -- every class panel below is its own RGBA composite (colour AND a
    # per-cell alpha), not a single `cmap`/`norm` image a colorbar can key
    # off directly.
    sm2 = ScalarMappable(norm=norm2, cmap=cmap2)
    extent = [xi_edges[0], xi_edges[-1], b_edges[0], b_edges[-1]]

    # Row 1: ONE panel, column 0 -- the total over classes, where the
    # prior expects a source at this position (item 1). The other five
    # slots of the row are empty.
    y0_row1 = page_h - margin_t - row_h
    ax1 = fig.add_axes([margin_l / page_w, y0_row1 / page_h, shape_w / page_w, row_h / page_h])
    im1 = ax1.imshow(joint_total.T, origin="lower", aspect="auto", extent=extent, cmap=cmap1, norm=norm1)
    ax1.axvline(_LOG10_XI_EDGE, color="0.35", lw=0.9, linestyle="--", alpha=0.9)
    ax1.set_xlim(_LOG10_XI_MIN, log10_xi_max)
    # No x-axis title on row 1 -- row 2 below carries it; the tick marks
    # and tick labels, and the y axis, stay.
    ax1.tick_params(labelsize=_TICK_FONTSIZE, labelleft=True)
    ax1.set_xticks(np.array([-3.0, -2.0, -1.0, 0.0]))
    # Left-anchored at the panel's own left edge, not centred over it --
    # centred, the title (wider than the one narrow panel) would hang off
    # the page's left edge; left-anchored it runs rightward into the
    # row's five empty slots instead, where nothing else is drawn.
    ax1.set_title("where the prior expects a source at this position",
                  fontsize=_LABEL_FONTSIZE, fontweight="bold", pad=18, loc="left")
    ax1.set_ylabel(_SHARED_Y_LABEL, fontsize=_LABEL_FONTSIZE)

    # Row 2: the six class panels, each an RGBA composite over the axes'
    # white background -- colour = viridis at `norm2` of that class's own
    # share, alpha = the fade `alpha` the six shares' entropy sets (item 2):
    # solid where the prior decides the class, faded to white where it is
    # blind (every class at the common floor, the six shares equal).
    y0_row2 = page_h - margin_t - 2 * row_h - row_gap
    for c, cls in enumerate(CLASS_ORDER):
        x0 = margin_l + c * (shape_w + col_gap)
        ax = fig.add_axes([x0 / page_w, y0_row2 / page_h, shape_w / page_w, row_h / page_h])
        ax.set_facecolor("white")
        rgba = cmap2(norm2(share[cls].T))
        rgba[..., 3] = alpha.T
        ax.imshow(rgba, origin="lower", aspect="auto", extent=extent)
        # THE EDGE ξ = 1, `log10 ξ = 0` (`x = 1`), marked on every panel -- grey,
        # not white, since the panel's own axis extends past it into the
        # padding, where the read now draws real mass and a white line
        # would be lost against a bright cell there.
        ax.axvline(_LOG10_XI_EDGE, color="0.35", lw=0.9, linestyle="--", alpha=0.9)
        # The panel's own axis: never less than the edge ξ = 1, extended to
        # the median source's own blurred read's last populated cell
        # where the column kernel carries mass past it (`_panel_xi_max`).
        ax.set_xlim(_LOG10_XI_MIN, log10_xi_max)
        ax.set_xlabel(_XI_LABEL_MEASURED, fontsize=_LABEL_FONTSIZE)
        # The six panels of a row share one brightness axis, so only
        # the leftmost carries its numbers; repeated on every panel
        # they overhang the column gap into the panel to the left.
        ax.tick_params(labelsize=_TICK_FONTSIZE, labelleft=(c == 0))
        ax.set_xticks(np.array([-3.0, -2.0, -1.0, 0.0]))
        ax.set_title(cls, fontsize=_LABEL_FONTSIZE, pad=18)
        if c == 0:
            # Bold as `plot_style.apply_style`'s own `axes.labelweight`
            # requires -- set as the axes' own ylabel, never a plain
            # figure text.
            ax.set_ylabel(_SHARED_Y_LABEL, fontsize=_LABEL_FONTSIZE)

    # Row 1's colour bar, inset at its own panel's right edge (item 1),
    # rather than out past the row's five empty slots.
    cax1_rect = [(margin_l + shape_w + 0.15) / page_w, y0_row1 / page_h, 0.22 / page_w, row_h / page_h]
    cax1 = fig.add_axes(cax1_rect)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label(r"$\mathbf{P(\hat{\xi}, F_{4.5} \mid s)}$", fontsize=_LABEL_FONTSIZE)
    cbar1.ax.tick_params(labelsize=_TICK_FONTSIZE)

    cax2_rect = [(margin_l + 6 * shape_w + 5 * col_gap + 0.15) / page_w,
                 (AXIS_LABEL_MARGIN_IN + caption_block_h) / page_h, 0.22 / page_w, row_h / page_h]
    cax2 = fig.add_axes(cax2_rect)
    cbar2 = fig.colorbar(sm2, cax=cax2)
    cbar2.set_label(r"$\mathbf{P(C \mid \hat{\xi}, F_{4.5}, s)}$", fontsize=_LABEL_FONTSIZE)
    cbar2.ax.tick_params(labelsize=_TICK_FONTSIZE)

    # The title names the page; the median source's own name, column and
    # arm are the subtitle below it (`figure.titleweight` bolds the
    # suptitle already; the subtitle, a plain `fig.text`, is bolded
    # explicitly).
    fig.suptitle("%s Prior Parameter Plane" % region,
                 fontsize=_TITLE_FONTSIZE, y=1.0 - 0.15 / page_h)
    fig.text(0.5, 1.0 - 0.50 / page_h, subtitle_text,
              fontsize=_SUBTITLE_FONTSIZE, fontweight="bold", ha="center", va="top")

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
                lambda_floor=data["lambda_floor"], floor_cell_fraction=data["floor_fraction"],
                alpha_low_fraction=data["alpha_low_fraction"])
    return paths


def build(config, regions=None):
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region(config, region)


# ====================================================================
# The region page: where in the nuisance
# plane the prior puts the sky's selected mass, and what class favours
# it there. Six class panels, one per class in `CLASS_ORDER`, `bmstp.
# atlas` P6's own per-cell region grid read verbatim (rule 5: no
# recomputation of the sums the product carries) -- `N_CAT_CELL_<C>`,
# the region's expected number of selected objects of class C per
# parameter cell, summing over cells to `RATIO_C * TOTAL_OBSERVED`.
# `N_CELL_<C>`'s own total sets the catalogable fraction the caption and
# the printed numbers carry; its per-cell array is not drawn. Never read
# by `bmstp`/`fittp` (module docstring's own rule).
# ====================================================================

#: I2 (4.5 micron)'s index into `catalog.depth_grid`'s own band axis --
#: the same construction `bmstp.atlas.IDX_I2` uses.
_BAND_KEYS = tuple(b.key for b in definitions.BANDS)
_IDX_I2 = _BAND_KEYS.index("I2")



def _read_prior_atlas(config, region):
    """P6's per-cell region grids and totals this page needs:
    `N_CAT_CELL_<C>`, `N_CELL_<C>` (both `(128, 110)` on `grid.
    LOG10_XI_EDGES` by `grid.LOG10_F45_EDGES`, `bmstp.atlas`'s module
    docstring), `RATIO_<C>` and `TOTAL_OBSERVED` (the region's own
    total-count check, sec. 9) -- read as the product stores them, no
    recomputation of the sums it already carries (rule 5)."""
    path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        n_cat_cell = {c: f["N_CAT_CELL_%s" % c][:].astype(np.float64) for c in CLASS_ORDER}
        n_cell = {c: f["N_CELL_%s" % c][:].astype(np.float64) for c in CLASS_ORDER}
        ratio = {c: float(f.attrs["RATIO_%s" % c]) for c in CLASS_ORDER}
        total_observed = float(f.attrs["TOTAL_OBSERVED"])
    return n_cat_cell, n_cell, ratio, total_observed


def _region_f_lim_i2_mjy(config, region):
    """The region's median 4.5 micron 50% limit (I2), `catalog.
    depth_grid`'s own per-pixel `F_LIM_50_MED_MJY` (rule 5: read, not
    rebuilt) -- the horizontal line every panel of this page marks."""
    path = config_module.product_path(config, "catalog", "sesna", "depth-grid", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        f_lim = f["F_LIM_50_MED_MJY"][:, _IDX_I2].astype(np.float64)
    return float(np.median(f_lim))


def _populated_xi_range(cell, xi_edges):
    """The `log10 ξ` range where `cell`'s own row mass (summed over
    brightness) exceeds `grid.FLOOR` of its own peak row -- the same
    relative floor `_panel_xi_max` applies at the per-source page."""
    row_mass = cell.sum(axis=1)
    peak = float(row_mass.max())
    if peak <= 0.0:
        return float("nan"), float("nan")
    above = np.nonzero(row_mass > grid.FLOOR * peak)[0]
    if not above.size:
        return float("nan"), float("nan")
    return float(xi_edges[above[0]]), float(xi_edges[above[-1] + 1])


def _build_region_prior_data(config, region):
    """P6's per-cell region grid and totals and the region's I2 50%
    limit -- everything `_draw_region_figure`/`_print_region_numbers`
    need, read once."""
    n_cat_cell, n_cell, ratio, total_observed = _read_prior_atlas(config, region)
    xi_edges, b_edges = grid.LOG10_XI_EDGES, grid.LOG10_F45_EDGES

    n_cat_total = {c: float(n_cat_cell[c].sum()) for c in CLASS_ORDER}
    n_cell_total = {c: float(n_cell[c].sum()) for c in CLASS_ORDER}
    catalogable_fraction = {
        c: (n_cat_total[c] / n_cell_total[c] if n_cell_total[c] > 0 else float("nan"))
        for c in CLASS_ORDER}

    populated_x = {c: _populated_xi_range(n_cat_cell[c], xi_edges) for c in CLASS_ORDER}

    log10_f_lim_med = float(np.log10(_region_f_lim_i2_mjy(config, region)))

    return dict(n_cat_cell=n_cat_cell, n_cell=n_cell, ratio=ratio, total_observed=total_observed,
                n_cat_total=n_cat_total, n_cell_total=n_cell_total,
                catalogable_fraction=catalogable_fraction, populated_x=populated_x,
                xi_edges=xi_edges, b_edges=b_edges, log10_f_lim_med=log10_f_lim_med)


def _print_region_numbers(region, data):
    for c in CLASS_ORDER:
        print("atlas.shapes.region [%s] %s: N_cat=%.6g RATIO=%.6f (RATIO*TOTAL_OBSERVED=%.6g) "
              "N_cell=%.6g catalogable_fraction=%.6f populated_log10x=[%.4g, %.4g]"
              % (region, c, data["n_cat_total"][c], data["ratio"][c],
                 data["ratio"][c] * data["total_observed"], data["n_cell_total"][c],
                 data["catalogable_fraction"][c], data["populated_x"][c][0], data["populated_x"][c][1]))
    print("atlas.shapes.region [%s] I2 50%% limit log10 F45 = %.4g"
          % (region, data["log10_f_lim_med"]))


def _draw_region_figure(config, region, data):
    plot_style.apply_style()
    xi_edges, b_edges = data["xi_edges"], data["b_edges"]
    n_cat_cell = data["n_cat_cell"]

    totals_lines = [captions.SHAPES_REGION_TOTALS.format(
        cls=c, n_cat=data["n_cat_total"][c], n_cell=data["n_cell_total"][c],
        ratio=data["catalogable_fraction"][c]) for c in CLASS_ORDER]
    caption_text, caption_block_h = captions.caption_layout(
        "\n\n".join([captions.SHAPES_REGION_ROW1, "Region totals:\n" + "\n".join(totals_lines)]),
        _CAPTION_CHARS_PER_LINE, _CAPTION_LINE_HEIGHT_IN, _CAPTION_TOP_PAD_IN, _CAPTION_BOTTOM_PAD_IN)

    # `margin_t` reserves room for the suptitle, THEN the row's own
    # class-name titles (`pad=18` points) below it -- the per-source
    # page's own convention.
    margin_l, margin_r, margin_t = 0.75, 1.05, 1.35
    col_gap = 0.30
    row_h = 3.2
    page_h = margin_t + row_h + AXIS_LABEL_MARGIN_IN + caption_block_h
    page_w = PAGE_W_IN
    usable_w = page_w - margin_l - margin_r
    shape_w = (usable_w - 5 * col_gap) / 6.0

    fig = plot_style.new_sized_figure(page_w, page_h)

    # ONE shared log scale for the six panels (module docstring), floored
    # at `grid.FLOOR` of the row's own peak cell -- the same relative
    # floor the per-source page applies to its own prior density
    # (`Lambda_floor`). No masking below the floor: the array is drawn as
    # it is, and the norm clips a cell at or below the floor to the scale's
    # own low end, white on `"YlGnBu"`; a zero cell, undefined on a log
    # scale, is masked and drawn the same white through `cmap.set_bad`.
    all_cat = np.concatenate([n_cat_cell[c].ravel() for c in CLASS_ORDER])
    cat_peak = float(all_cat.max())
    cat_floor = grid.FLOOR * cat_peak
    norm = LogNorm(vmin=cat_floor, vmax=cat_peak)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")

    extent = [xi_edges[0], xi_edges[-1], b_edges[0], b_edges[-1]]
    y0 = page_h - margin_t - row_h
    im = None
    for c, cls in enumerate(CLASS_ORDER):
        x0 = margin_l + c * (shape_w + col_gap)
        ax = fig.add_axes([x0 / page_w, y0 / page_h, shape_w / page_w, row_h / page_h])
        arr = np.ma.masked_less_equal(n_cat_cell[cls], 0.0)
        im = ax.imshow(arr.T, origin="lower", aspect="auto", extent=extent, cmap=cmap, norm=norm)

        # THE EDGE ξ = 1, drawn on every panel (module docstring): the region
        # grid is zero above it (no per-source blur padding at this
        # population level), so the panel's own axis runs only to the
        # wall. The dotted line is the region's median I2 50% limit.
        ax.axvline(_LOG10_XI_EDGE, color="0.35", lw=0.9, linestyle="--", alpha=0.9)
        ax.axhline(data["log10_f_lim_med"], color="0.25", lw=0.8, linestyle=":", alpha=0.9)

        # a hair past the edge ξ = 1, so the edge ξ = 1's own line is drawn inside the
        # axes rather than on its spine; the grid is zero beyond it
        ax.set_xlim(_LOG10_XI_MIN, _LOG10_XI_EDGE + 0.08)
        ax.set_xlabel(_XI_LABEL, fontsize=_LABEL_FONTSIZE)
        ax.tick_params(labelsize=_TICK_FONTSIZE, labelleft=(c == 0))
        ax.set_xticks(np.array([-3.0, -2.0, -1.0, 0.0]))
        ax.set_title(cls, fontsize=_LABEL_FONTSIZE, pad=18)
        if c == 0:
            ax.set_ylabel(_SHARED_Y_LABEL, fontsize=_LABEL_FONTSIZE)

    cax_rect = [(margin_l + 6 * shape_w + 5 * col_gap + 0.15) / page_w,
                y0 / page_h, 0.22 / page_w, row_h / page_h]
    cax = fig.add_axes(cax_rect)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("objects per cell", fontsize=_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)

    fig.suptitle("%s Prior Selection Count" % region,
                 fontsize=_TITLE_FONTSIZE, y=1.0 - 0.15 / page_h)

    fig.text(margin_l / page_w, (caption_block_h - _CAPTION_TOP_PAD_IN) / page_h, caption_text,
              fontsize=_CAPTION_FONTSIZE, va="top", ha="left", linespacing=_CAPTION_LINESPACING)

    out_dir = f"{config.data_root}/bmstp/atlas/figures"
    paths = []
    for fmt in ("png", "pdf"):
        paths.append(f"{out_dir}/prior-shapes-region_{region}.{fmt}")
    return fig, paths


def build_region_prior_one(config, region):
    with progress.Stage("atlas.shapes.region", region) as st:
        data = _build_region_prior_data(config, region)
        _print_region_numbers(region, data)
        fig, paths = _draw_region_figure(config, region, data)
        os.makedirs(f"{config.data_root}/bmstp/atlas/figures", exist_ok=True)
        for path in paths:
            fig.savefig(path, dpi=150)
        plt.close(fig)
        st.done(paths[0], n_panel=len(CLASS_ORDER))
    return paths


def build_region_prior(config, regions=None):
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region_prior_one(config, region)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()
    build(config_module.load(args.config), regions=args.regions)


if __name__ == "__main__":
    _main()
