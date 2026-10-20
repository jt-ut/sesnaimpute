"""The survey's depth, one page per region (SPEC_BMSTP_DRAFT.md sec. 3.3,
"the depth grid"; sec. 8's coverage fraction): left, each of the eight
bands' own completeness curve at the region's median pixel; right, the
catalogue's own IRAC coverage fraction across the region's footprint.

`catalog.depth_grid` writes, per admitted nside-512 pixel, `F_LIM_50_PIX_
MJY` (the pixel's own 50%-completeness flux limit, every band) and `W_DEX_
PIX` (the roll-off width in dex, every band); `fittp.likelihood`'s own
detection-probability roll-off (sec. 6.2) is the erf form `C(z) = 0.5
erfc(-z)`, `z = (log10 f - log10 F_lim50) / (sqrt(2) w)` -- equivalently
the standard normal CDF of `(log10 f - log10 F_lim50) / w`, evaluated here
directly rather than through that module's own log-space kernel (which
returns `ln[1 - C(z)]` for the fit, not `C(z)` itself). The median pixel is
the one whose own 4.5 um (`I2`) limit is the region's median -- a single,
reproducible pixel, not an average over pixels, so its eight bands' limits
and widths are a real, self-consistent set a reader could look up.

`sky.derived.coverage` writes `FRAC`, each Spitzer band's own fraction of
the pixel on a covered mosaic pixel; the any-IRAC union taken here (the
four IRAC bands' `max`, sec. 8's "the fraction of each sky pixel the IRAC
catalog covers") is drawn on the same tangent-plane footprint and Gaussian
reprojection every atlas page uses (`atlas.render._footprint_geometry`/
`_reproject`), so this page's map lines up with the rest of the deck's sky
figures pixel for pixel.
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize
from scipy.special import erf, erfinv

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.atlas.render import (
    LABEL_FONTSIZE, TICK_FONTSIZE, _add_panel, _align, _footprint_geometry,
    _panel_colorbar, _read_depth_grid)
from sesnaimpute.catalog import limits as limits_module

PAGE_W_IN = 16.0
PAGE_H_IN = 6.0

_TITLE_FONTSIZE = 18
_STATEMENT_FONTSIZE = 9

_SQRT2 = float(np.sqrt(2.0))

#: One nside-512 HEALPix pixel's solid angle, deg^2 (41252.96 deg^2 of
#: full sky over 12 * nside^2 equal-area pixels), the area the covered-
#: area print multiplies the admitted footprint's own union fraction by.
_HPX512_PIXEL_DEG2 = 41252.96 / (12 * 512 ** 2)

#: The eight bands, `definitions.BANDS`' own fixed order (J, H, Ks, I1-I4,
#: M1), each with the plain wavelength word this page's direct labels use.
_BAND_LABELS = ("J", "H", "Ks", "3.6", "4.5", "5.8", "8.0", "24")
_I2_INDEX = tuple(b.key for b in definitions.BANDS).index("I2")

_STATEMENT = (
    "Left: the probability that a source of a given flux is detected in each band at a "
    "typical Orion A pixel, from the survey's own limits; right: the fraction of each sky "
    "pixel the IRAC catalog covers. A source is selected when two of the eight bands detect it."
)


def _completeness(log10_f, log10_f50, w_dex):
    """`C(z) = 0.5 (1 + erf(z))`, `z = (log10_f - log10_f50) / (sqrt(2)
    w_dex)` -- `fittp.likelihood`'s own roll-off (module docstring),
    evaluated directly on an array of trial fluxes rather than through
    that module's log-space, non-detection-only kernel."""
    z = (log10_f - log10_f50) / (_SQRT2 * w_dex)
    return 0.5 * (1.0 + erf(z))


def _read_coverage(config, region):
    """`sky/derived/spitzer/coverage_spitzer_hpx512`'s own `HPX_PIX` and
    per-band `FRAC` (I1 I2 I3 I4 M1, that product's own column order),
    reduced to the any-IRAC union fraction -- the four IRAC bands' `max`,
    a LOWER BOUND on the true union (the product stores only each band's
    own fraction, not the union mask itself), exact where the four IRAC
    footprints coincide, which is nearly everywhere since 3.6/5.8 um
    share one camera array and 4.5/8.0 um the other. The product carries
    no `AREA_DEG2` attribute; `build_region` computes the covered area
    from this fraction restricted to the region's own admitted pixels."""
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.figure_depth: no Spitzer mosaic coverage for %s at %s -- run the "
            "'sesnaimpute.sky.derived.coverage' RUNBOOKtp.sh line first" % (region, path))
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)
    union = frac[:, :4].max(axis=1) if frac.size else np.empty(0, dtype=np.float64)
    order = np.argsort(pix)
    return pix[order], union[order]


def _read_depth_pixels(config, region):
    """Every admitted pixel's own measured 50%-completeness flux (n, 8)
    and local erf-width (n, 8), from `catalog.limits.completeness`'s
    reader object (`fifty_percent_pixel`/`local_width_dex_pixel` --
    COMPLETENESS_2026-10-20.md "The reader"), sorted by pixel to match
    `_read_depth_grid`'s footprint ordering; replaces the retired
    `catalog.depth_grid` counts fit (`F_LIM_50_PIX_MJY`/`W_DEX_PIX`)."""
    reader = limits_module.completeness(config, region)
    pix = reader.hpx_pix_512
    rows = np.arange(pix.shape[0])
    band_keys = tuple(b.key for b in definitions.BANDS)
    f_lim = np.empty((pix.shape[0], len(band_keys)), dtype=np.float64)
    w_dex = np.empty((pix.shape[0], len(band_keys)), dtype=np.float64)
    for j, band in enumerate(band_keys):
        flux50 = reader.fifty_percent_pixel(band, rows)
        width = reader.local_width_dex_pixel(band, rows)
        unsurveyed = ~np.isfinite(flux50)
        f_lim[:, j] = np.where(unsurveyed, np.inf, flux50)
        w_dex[:, j] = np.where(unsurveyed | ~np.isfinite(width), 0.0, width)
    order = np.argsort(pix)
    return pix[order], f_lim[order], w_dex[order]


def _median_pixel_row(pix, f_lim):
    """The row whose own 4.5 um (`I2`) limit is closest to the region's
    median 4.5 um limit -- one actual pixel's own eight-band set, not an
    average over pixels (module docstring)."""
    i2 = f_lim[:, _I2_INDEX]
    median = float(np.median(i2))
    row = int(np.argmin(np.abs(i2 - median)))
    return row, pix[row]


#: The completeness level each direct label sits at, on its own curve
#: (module docstring: every band's curve saturates toward 1 by the
#: plot's right edge, so a label at the shared right EDGE would stack
#: on top of every other band's; a label placed at a fixed completeness
#: instead sits at that band's OWN flux -- `log10_F50 + z90 sqrt(2) w`
#: -- which differs band to band, so eight labels land at eight
#: different x positions along their own curves).
_LABEL_COMPLETENESS = 0.90


def _label_positions(log10_f50, w_dex):
    """Each label's `(x, y)` on its own curve at `_LABEL_COMPLETENESS`
    (`erfinv`, the exact inverse of `_completeness`'s `erf`), then a
    small alternating vertical offset for any pair whose labels would
    otherwise overlap (`collision_gap` dex apart in x), so two bands
    with nearby limits (e.g. `H`/`Ks`) still print two readable words."""
    z90 = float(erfinv(2.0 * _LABEL_COMPLETENESS - 1.0))
    x = log10_f50 + z90 * _SQRT2 * w_dex
    y = np.full(x.shape, _LABEL_COMPLETENESS)
    order = np.argsort(x)
    collision_gap = 0.18
    offset = 0.10
    toggle = False
    last_x = None
    for idx in order:
        if last_x is not None and (x[idx] - last_x) < collision_gap:
            toggle = not toggle
        else:
            toggle = False
        y[idx] = _LABEL_COMPLETENESS + (offset if toggle else 0.0)
        last_x = x[idx]
    return x, y


def _draw_completeness_panel(fig, rect_in, page_w, page_h, f_lim_row, w_row):
    """Panel 1: each band's own completeness curve at the median pixel,
    `_completeness` against `log10` flux (mJy), viridis in wavelength
    order, a dotted 50% line, each line labelled directly on its own
    curve near its right end (`_label_positions`)."""
    ax = fig.add_axes([rect_in[0] / page_w, rect_in[1] / page_h,
                        rect_in[2] / page_w, rect_in[3] / page_h])
    log10_f50 = np.log10(f_lim_row)
    x_min = float(np.min(log10_f50 - 5.0 * w_row))
    x_max = float(np.max(log10_f50 + 5.0 * w_row))
    x_grid = np.linspace(x_min, x_max, 600)

    cmap = plt.get_cmap("viridis")
    n_bands = len(_BAND_LABELS)
    colors = [cmap(i / (n_bands - 1)) for i in range(n_bands)]

    for j in range(n_bands):
        y = _completeness(x_grid, log10_f50[j], w_row[j])
        ax.plot(x_grid, y, color=colors[j], lw=1.8, zorder=3)

    label_x, label_y = _label_positions(log10_f50, w_row)
    outline = [path_effects.withStroke(linewidth=2.5, foreground="white")]
    for j in range(n_bands):
        ax.text(label_x[j], label_y[j], _BAND_LABELS[j], fontsize=10.5, fontweight="bold",
                color=colors[j], va="center", ha="center", zorder=4, path_effects=outline)

    ax.axhline(0.5, color="grey", lw=1.0, linestyle=":", zorder=1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.03, 1.15)
    ax.set_xlabel(plot_style.label(r"$\mathbf{log_{10}\,F}$", "mJy"), fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$\mathbf{completeness}$", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_title("Completeness at the Median Pixel", fontsize=LABEL_FONTSIZE, pad=6)
    return ax


def build_region(config, region, formats=("png", "pdf")):
    """Writes `catalog/sesna/figures/depth_sesna_region__<region>.png/.pdf`
    and prints the median pixel's eight 50% limits and widths and the
    covered area (module docstring)."""
    with progress.Stage("catalog.figure_depth", region) as st:
        plot_style.apply_style()

        depth_pix, f_lim, w_dex = _read_depth_pixels(config, region)
        row, pix_id = _median_pixel_row(depth_pix, f_lim)
        f_lim_row = f_lim[row]
        w_row = w_dex[row]

        cov_pix, frac = _read_coverage(config, region)
        footprint_pix = _read_depth_grid(config, region)
        frac_aligned = _align(footprint_pix, cov_pix, frac, fill=0.0)
        area_deg2 = float(np.sum(frac_aligned) * _HPX512_PIXEL_DEG2)

        band_words = " ".join(
            "%s=%.4g/%.3g" % (label, f, w) for label, f, w in zip(_BAND_LABELS, f_lim_row, w_row))
        print("catalog.figure_depth [%s]: median pixel HPX_PIX_512=%d 50%% limits [mJy]/widths [dex] "
              "%s" % (region, pix_id, band_words))
        print("catalog.figure_depth [%s]: covered area = %.4g deg^2 over %d admitted pixels"
              % (region, area_deg2, footprint_pix.size))

        fig = plot_style.new_sized_figure(PAGE_W_IN, PAGE_H_IN)

        left_rect = (0.95, 1.05, 6.6, 4.1)
        _draw_completeness_panel(fig, left_rect, PAGE_W_IN, PAGE_H_IN, f_lim_row, w_row)

        geom = _footprint_geometry(footprint_pix)
        aspect = geom["n_y"] / geom["n_x"]
        right_h = 4.3
        right_w = min(6.3, right_h / aspect)
        right_h = right_w * aspect
        right_x = 9.6 + (6.3 - right_w) / 2.0
        right_y = 1.0 + (4.3 - right_h) / 2.0
        right_rect = (right_x, right_y, right_w, right_h)
        grid = np.zeros(geom["shape"], dtype=np.float64)
        loc = np.minimum(np.searchsorted(footprint_pix, geom["grid_pix"]), footprint_pix.size - 1)
        found = footprint_pix[loc] == geom["grid_pix"]
        value_at = np.zeros(geom["grid_pix"].size, dtype=np.float64)
        value_at[found] = frac_aligned[loc[found]]
        grid_flat = value_at
        grid_flat[~found] = np.nan
        grid = grid_flat.reshape(geom["shape"])

        ax2, im2 = _add_panel(fig, right_rect, PAGE_W_IN, PAGE_H_IN, geom["wcs"], grid, "viridis",
                               norm=Normalize(0.0, 1.0), title="Coverage")
        cbar2 = _panel_colorbar(fig, ax2, im2, label="fraction covered", ticks=(0.0, 0.5, 1.0))
        cbar2.ax.yaxis.label.set_fontweight("bold")

        fig.suptitle("%s Survey Depth" % region, fontsize=_TITLE_FONTSIZE, y=0.985)
        fig.text(0.5, 0.06, _STATEMENT, fontsize=_STATEMENT_FONTSIZE, ha="center", va="top",
                  wrap=True)

        out_dir = os.path.join(config.data_root, "catalog", "sesna", "figures")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for fmt in formats:
            path = os.path.join(out_dir, "depth_sesna_region__%s.%s" % (region, fmt))
            fig.savefig(path, dpi=150)
            paths.append(path)
        plt.close(fig)
        st.done(paths[0], median_pixel=pix_id, area_deg2=area_deg2)
    return paths


def build(config, regions=None):
    """`build(config, regions=None)`: per region, `build_region` (default:
    every region `regions.REGIONS` names)."""
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
