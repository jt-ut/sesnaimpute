"""The sky atlas figures (SPEC_BMSTP_DRAFT.md sec. 8; IMPLEMENTATION_
BMSTP_DRAFT.md sec. 1.2 P6, sec. 1.3 P11). Per region, one figure
reprojecting the admitted nside-512 pixels onto a tangent-plane display
grid of 1' pixels centred on the region: the pixel's nearest nside-512
value (healpy `ang2pix`), then a Gaussian smoothing of one nside-512
pixel width (6.9') so pixel edges do not show. Nothing at display
resolution is stored -- the figure is rendered at plot time only. The
granule map's own frame is galactic (`granules/build.py`'s `ang2pix(...,
gl, gb, ...)`); the display grid and its RA/Dec ticks are equatorial, so
every lookup crosses frames once through astropy, never through a local
approximation.

Every panel is drawn at the same size and shares the region's own
display-grid aspect: the column `A_K` (log scale), the total predicted
catalogued density `Sigma_C N_CAT_C` (deg^-2, hatched where the surveyed
IRAC-coverage fraction is below 0.5), the prior share `SHARE_C` per
class in class order, and -- when the posterior atlas (P11) exists --
the posterior mean `MEAN_P_C` per class plus `N_YSO_ABOVE_HALF`: 15
panels with a posterior, 8 without. Panels fill row-major, in that
order, into the grid `page_geometry` picks (sec. below); the region's
total-count ratio (`RATIO_<CLS>`) sits in the figure's caption line
rather than any panel title.

Colour maps and scales follow the convention the earlier package's own
sky-atlas figure used (`sesnacomplete.bms_prior.validation.sky_atlas`):
the measured column on `magma`, every class quantity (share, posterior
mean) on `viridis`, its own colour scale per panel (a shared 0-1 bar
hides the spatial variation of the low-share classes, YSO first among
them). That earlier figure's own `page_geometry` insets a colour bar
into the panel when the footprint leaves clear room for one and sizes a
3x2 class block plus one wide dust panel to fill a page that grows to
fit its content; here every panel is an equal member of one grid with
its own colour-bar strip (no panel is large enough to inset one), so
`page_geometry` below generalises that rule to N equal panels on a
FIXED page (a slide, 16x9in by default, `--page WxH` to override): for
columns `c = 1..N` (rows `r = ceil(N/c)`), it picks the `c` that
maximises the common panel scale the page allows, exactly as the
earlier module picked the class block's shape from a short list of
candidates.
"""

import argparse
import os

import astropy.units as u
import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import gaussian_filter

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute import regions as regions_module

NSIDE = 512
CLASSES = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")

#: The display grid's own pixel size (sec. 8: "the 1' grid of the
#: current atlas").
PIXEL_ARCMIN = 1.0

#: One nside-512 pixel width -- sqrt of the pixel's own solid angle
#: (sec. 8's smoothing scale, "6.9'"), so the nearest-value reprojection's
#: pixel edges are blurred out rather than shown as steps.
SMOOTH_ARCMIN = float(np.sqrt(hp.nside2pixarea(NSIDE, degrees=True)) * 60.0)

#: A border of this many display pixels beyond the admitted footprint's
#: own bounding box, so the smoothed edge is not clipped by the frame.
PAD_DISPLAY_PIXELS = 6

# ---------------------------------------------------------------------------
# page geometry -- every panel the same size, as large as a fixed page
# allows, generalising the earlier package's own
# `sesnacomplete.bms_prior.validation.sky_atlas.page_geometry`.
# ---------------------------------------------------------------------------

#: The page a region's figure is drawn on, inches -- a slide by default;
#: `--page WxH` overrides both numbers together.
PAGE_WIDTH_IN, PAGE_HEIGHT_IN = 16.0, 9.0

#: Page furniture, inches: room outside the panel grid for the left
#: column's Dec tick labels, the bottom row's RA tick labels, and the
#: two-line suptitle above.
MARGIN_LEFT_IN, MARGIN_RIGHT_IN = 0.55, 0.15
MARGIN_TOP_IN, MARGIN_BOTTOM_IN = 0.95, 0.55
GAP_X_IN, GAP_Y_IN = 0.10, 0.12

#: A panel's own colour-bar strip (with its tick labels) and title
#: strip, each as a fraction of the panel's own height `s` -- so, like
#: the earlier module's inset bar, they cost more page space on a
#: bigger panel but never crowd a small one. Sized for a thin bar plus
#: three two/three-character ticks and a two-line class-name title.
BAR_WIDTH_FRACTION = 0.30
TITLE_HEIGHT_FRACTION = 0.15

#: Selection ties (within this fraction of the best scale found) go to
#: the wider grid, which reads better than a tall stack -- the same
#: rule the earlier module used to break ties between its own two
#: candidate class-block shapes.
GRID_TIE_FRACTION = 0.02


def page_geometry(aspect, n_panels, page_w=PAGE_WIDTH_IN, page_h=PAGE_HEIGHT_IN):
    """The `(cols, rows, panel_w, panel_h, bar_w, title_h)` that lays
    `n_panels` equal panels of display-grid aspect `aspect` (width over
    height) as large as the fixed `page_w` x `page_h` page allows, all
    in inches.

    For every candidate column count `c` (1..n_panels; rows `r =
    ceil(n_panels / c)`), the common panel height `s` is capped by the
    page width split `c` ways (each column costs a panel width `s *
    aspect` plus its own colour-bar strip `s * BAR_WIDTH_FRACTION`) and
    by the page height split `r` ways (each row costs a panel height
    `s` plus its own title strip `s * TITLE_HEIGHT_FRACTION`); the `c`
    that maximises `s` is picked, ties going to the wider grid.
    """
    candidates = []
    for cols in range(1, n_panels + 1):
        rows = -(-n_panels // cols)  # ceil
        usable_w = page_w - MARGIN_LEFT_IN - MARGIN_RIGHT_IN - (cols - 1) * GAP_X_IN
        usable_h = page_h - MARGIN_TOP_IN - MARGIN_BOTTOM_IN - (rows - 1) * GAP_Y_IN
        if usable_w <= 0.0 or usable_h <= 0.0:
            continue
        s = min(usable_w / (cols * (aspect + BAR_WIDTH_FRACTION)),
                usable_h / (rows * (1.0 + TITLE_HEIGHT_FRACTION)))
        if s > 0.0:
            candidates.append((cols, rows, s))
    if not candidates:
        raise ValueError(
            "page_geometry: no arrangement of %d panels fits a %.3gx%.3gin "
            "page at aspect %.3g" % (n_panels, page_w, page_h, aspect))
    s_max = max(s for _, _, s in candidates)
    cols, rows, s = max(
        (c for c in candidates if c[2] >= s_max * (1.0 - GRID_TIE_FRACTION)),
        key=lambda c: c[0])
    panel_h = s
    panel_w = s * aspect
    bar_w = s * BAR_WIDTH_FRACTION
    title_h = s * TITLE_HEIGHT_FRACTION
    return dict(cols=cols, rows=rows, panel_w=panel_w, panel_h=panel_h,
                bar_w=bar_w, title_h=title_h)


def _panel_rect(geom, index, page_w, page_h):
    """The panel axes' `(x, y, w, h)` in inches from the page's lower
    left, for panel `index` filled row-major into `geom`'s grid, and the
    grid's own occupied width/height (for centring the grid on the
    page when one dimension of `page_geometry` does not bind)."""
    cols, rows = geom["cols"], geom["rows"]
    panel_w, panel_h = geom["panel_w"], geom["panel_h"]
    bar_w, title_h = geom["bar_w"], geom["title_h"]
    content_w = cols * (panel_w + bar_w) + (cols - 1) * GAP_X_IN
    content_h = rows * (panel_h + title_h) + (rows - 1) * GAP_Y_IN
    usable_w = page_w - MARGIN_LEFT_IN - MARGIN_RIGHT_IN
    usable_h = page_h - MARGIN_TOP_IN - MARGIN_BOTTOM_IN
    origin_x = MARGIN_LEFT_IN + 0.5 * (usable_w - content_w)
    origin_y = MARGIN_BOTTOM_IN + 0.5 * (usable_h - content_h)
    row, col = divmod(index, cols)
    x = origin_x + col * (panel_w + bar_w + GAP_X_IN)
    y = origin_y + (rows - 1 - row) * (panel_h + title_h + GAP_Y_IN)
    return x, y, panel_w, panel_h


def _bar_rect(panel_rect, geom):
    """The colour-bar strip's `(x, y, w, h)` in inches, immediately to
    the right of `panel_rect` inside the panel's own reserved `bar_w`."""
    x, y, w, h = panel_rect
    bar_w = geom["bar_w"]
    return x + w + 0.22 * bar_w, y + 0.05 * h, 0.24 * bar_w, 0.90 * h


def _frac(rect, page_w, page_h):
    """`rect` (inches) as a figure-fraction `[x, y, w, h]`."""
    x, y, w, h = rect
    return [x / page_w, y / page_h, w / page_w, h / page_h]


def _outer_rows(n_panels, cols):
    """The row index of the bottom-most panel in each column, for the
    row-major fill of `n_panels` into `cols` columns -- the panel whose
    RA tick labels are shown, since a partial last row can leave a
    column's own bottom panel above the grid's nominal last row."""
    last_row = {}
    for i in range(n_panels):
        row, col = divmod(i, cols)
        last_row[col] = row
    return last_row


# ---------------------------------------------------------------------------
# reading, reprojection, rendering
# ---------------------------------------------------------------------------

def _read_prior(path):
    """The prior atlas (P6), sorted by pixel for the reprojection's
    `searchsorted` lookup."""
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        a_k = np.asarray(f["A_COL_K"][:], dtype=np.float64)
        coverage = np.asarray(f["COVERAGE"][:], dtype=np.float64)
        n_cat = np.stack([np.asarray(f["N_CAT_%s" % c][:], dtype=np.float64) for c in CLASSES], axis=1)
        share = np.stack([np.asarray(f["SHARE_%s" % c][:], dtype=np.float64) for c in CLASSES], axis=1)
        attrs = dict(f.attrs)
    order = np.argsort(pix)
    return dict(pix=pix[order], a_k=a_k[order], coverage=coverage[order],
                n_cat=n_cat[order], share=share[order], attrs=attrs)


def _read_posterior(path):
    """The posterior atlas (P11), sorted by pixel."""
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        mean_p = np.stack([np.asarray(f["MEAN_P_%s" % c][:], dtype=np.float64) for c in CLASSES], axis=1)
        n_yso_half = np.asarray(f["N_YSO_ABOVE_HALF"][:], dtype=np.float64)
    order = np.argsort(pix)
    return dict(pix=pix[order], mean_p=mean_p[order], n_yso_half=n_yso_half[order])


def _region_centre_icrs(pix):
    """`(ra0, dec0)` in ICRS degrees: the mean unit vector of the
    admitted footprint's own galactic pixel positions, rotated once to
    ICRS -- the tangent point of the display WCS and the figure's RA/Dec
    ticks. A vector mean commutes with the (linear, orthogonal) frame
    rotation, so this equals the ICRS mean directly."""
    l, b = hp.pix2ang(NSIDE, pix, nest=True, lonlat=True)
    vec = hp.ang2vec(l, b, lonlat=True)
    mean_vec = vec.mean(axis=0)
    mean_vec /= np.linalg.norm(mean_vec)
    l0 = np.degrees(np.arctan2(mean_vec[1], mean_vec[0])) % 360.0
    b0 = np.degrees(np.arcsin(np.clip(mean_vec[2], -1.0, 1.0)))
    centre = SkyCoord(l=l0 * u.deg, b=b0 * u.deg, frame="galactic").icrs
    return float(centre.ra.deg), float(centre.dec.deg)


def _display_wcs(ra0, dec0, n_x, n_y):
    """A gnomonic (TAN) WCS in ICRS, `n_x` x `n_y` pixels of
    `PIXEL_ARCMIN` each, centred at `(ra0, dec0)`."""
    w = WCS(naxis=2)
    w.wcs.crpix = [n_x / 2.0 + 0.5, n_y / 2.0 + 0.5]
    w.wcs.cdelt = [-PIXEL_ARCMIN / 60.0, PIXEL_ARCMIN / 60.0]
    w.wcs.crval = [ra0, dec0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _grid_size(ra0, dec0, pix):
    """`(n_x, n_y)`: the smallest display grid, plus `PAD_DISPLAY_PIXELS`,
    that covers every admitted pixel's own ICRS position under the
    tangent projection centred at `(ra0, dec0)`."""
    l, b = hp.pix2ang(NSIDE, pix, nest=True, lonlat=True)
    radec = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
    w0 = _display_wcs(ra0, dec0, 1, 1)
    x, y = w0.wcs_world2pix(radec.ra.deg, radec.dec.deg, 0)
    n_x = max(int(np.ceil(2 * (np.max(np.abs(x)) + PAD_DISPLAY_PIXELS))), 12)
    n_y = max(int(np.ceil(2 * (np.max(np.abs(y)) + PAD_DISPLAY_PIXELS))), 12)
    return n_x, n_y


def _grid_pixels(wcs, n_x, n_y):
    """Every display cell's nside-512 galactic pixel: the cell's own
    RA/Dec (equatorial, from the display WCS) converted once to galactic
    for the `ang2pix` lookup, since the granule map's grid is galactic."""
    yy, xx = np.mgrid[0:n_y, 0:n_x]
    ra, dec = wcs.wcs_pix2world(xx.ravel(), yy.ravel(), 0)
    gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    return hp.ang2pix(NSIDE, gal.l.deg, gal.b.deg, nest=True, lonlat=True)


def _reproject(pix_sorted, values, grid_pix_flat, shape):
    """The nearest-nside-512 value at each display cell, then a Gaussian
    smoothing of `SMOOTH_ARCMIN` (one nside-512 pixel width),
    edge-normalised by the same smoothing of the footprint's own 0/1
    mask -- so a cell near the admitted boundary is the mean of only the
    admitted neighbours it has, rather than being pulled toward zero by
    empty sky (sec. 8: "so pixel edges do not show"). A source-less
    admitted pixel's `NaN` (the posterior atlas's own convention, P11)
    is excluded from the weight the same way an unadmitted cell is --
    one `NaN` input would otherwise spread through the whole smoothed
    output, since the Gaussian filter has no NaN-aware mode."""
    loc = np.minimum(np.searchsorted(pix_sorted, grid_pix_flat), pix_sorted.size - 1)
    found = pix_sorted[loc] == grid_pix_flat
    value_at = values[loc]
    valid = found & np.isfinite(value_at)
    data = np.zeros(grid_pix_flat.size, dtype=np.float64)
    data[valid] = value_at[valid]
    data = data.reshape(shape)
    mask = valid.astype(np.float64).reshape(shape)

    sigma = SMOOTH_ARCMIN / PIXEL_ARCMIN
    num = gaussian_filter(data, sigma, mode="constant", cval=0.0)
    den = gaussian_filter(mask, sigma, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.5] = np.nan
    return out


def _add_panel(fig, rect, page_w, page_h, wcs, data, cmap, norm=None, title="",
               show_dec=True, show_ra=True, title_size=9):
    """One sky panel at `rect` (inches, lower-left origin), on the
    shared display WCS. `show_dec`/`show_ra` gate the tick *labels*
    only (every panel keeps its ticks and grid, since every panel is
    the same field, sec. 8's "shared sky frame") -- only the grid's
    left column and each column's own bottom-most panel need the
    numbers repeated."""
    ax = fig.add_axes(_frac(rect, page_w, page_h), projection=wcs)
    im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=title_size, pad=4)
    for i in (0, 1):
        # `set_axislabel("")` alone is not enough: WCSAxes treats an
        # empty label as "unset" and redraws its own default
        # ("pos.eq.ra"/"pos.eq.dec") unless auto-labelling is off too.
        ax.coords[i].set_auto_axislabel(False)
        ax.coords[i].set_axislabel("")
        ax.coords[i].set_ticklabel(size=7)
        ax.coords[i].set_ticks(number=3)
    ax.coords[1].set_ticklabel_visible(show_dec)
    ax.coords[0].set_ticklabel_visible(show_ra)
    ax.coords.grid(color="white", alpha=0.3, linestyle="solid", linewidth=0.4)
    return ax, im


def _log_norm(grid):
    """A `LogNorm` spanning the grid's own finite positive range -- the
    log colour scale sec. 8 asks for the column and density panels."""
    finite = grid[np.isfinite(grid) & (grid > 0)]
    if finite.size == 0:
        return LogNorm(vmin=1e-6, vmax=1.0)
    return LogNorm(vmin=float(finite.min()), vmax=float(finite.max()))


def _colorbar(fig, im, rect, page_w, page_h):
    """A thin colour bar in its own reserved strip at `rect` (inches),
    scaled to that panel's own data range -- every panel's bar is the
    same width relative to its panel, sized by `page_geometry`'s own
    `bar_w`, rather than a bar that costs a fixed share of a fixed-size
    panel."""
    cax = fig.add_axes(_frac(rect, page_w, page_h))
    cbar = fig.colorbar(im, cax=cax)
    cbar.locator = MaxNLocator(nbins=3)
    cbar.formatter = FuncFormatter(lambda v, _pos: "%.2g" % v)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=7, length=2, pad=1.0)
    cbar.outline.set_linewidth(0.5)
    return cbar


def build_region(config, region, formats, page_w=PAGE_WIDTH_IN, page_h=PAGE_HEIGHT_IN):
    prior_path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    if not os.path.exists(prior_path):
        print("atlas.render [%s]: no prior atlas -- run RUNBOOKtp.sh's "
              "'PY sesnaimpute.bmstp.atlas' line first, skipped" % region, flush=True)
        return None
    prior = _read_prior(prior_path)

    post_path = config_module.product_path(config, "fittp", "atlas", "posterior", "hpx512", region=region)
    posterior = _read_posterior(post_path) if os.path.exists(post_path) else None

    with progress.Stage("atlas.render", region) as st:
        ra0, dec0 = _region_centre_icrs(prior["pix"])
        n_x, n_y = _grid_size(ra0, dec0, prior["pix"])
        wcs = _display_wcs(ra0, dec0, n_x, n_y)
        grid_pix = _grid_pixels(wcs, n_x, n_y)
        shape = (n_y, n_x)

        col_grid = _reproject(prior["pix"], prior["a_k"], grid_pix, shape)
        density_grid = _reproject(prior["pix"], prior["n_cat"].sum(axis=1), grid_pix, shape)
        coverage_grid = _reproject(prior["pix"], prior["coverage"], grid_pix, shape)
        share_grids = [_reproject(prior["pix"], prior["share"][:, i], grid_pix, shape)
                       for i in range(len(CLASSES))]

        has_post = posterior is not None
        if has_post:
            post_share_grids = [_reproject(posterior["pix"], posterior["mean_p"][:, i], grid_pix, shape)
                                 for i in range(len(CLASSES))]
            nyso_grid = _reproject(posterior["pix"], posterior["n_yso_half"], grid_pix, shape)

        # The panels, row-major fill order: column, predicted count,
        # the six prior shares, the six posterior means (if present),
        # the YSO count -- 15 panels with a posterior, 8 without.
        ratios = [(cls, float(prior["attrs"].get("RATIO_%s" % cls, np.nan))) for cls in CLASSES]
        low_coverage = coverage_grid < 0.5
        panels = [
            dict(data=col_grid, cmap="magma", norm=_log_norm(col_grid),
                 title="column $A_K$ (mag)", hatch=None),
            dict(data=density_grid, cmap="viridis", norm=_log_norm(density_grid),
                 title="predicted catalogued\nsources (deg$^{-2}$)",
                 hatch=low_coverage if np.any(low_coverage) else None),
        ]
        panels += [dict(data=share_grids[i], cmap="viridis", norm=None,
                        title="%s\nprior share" % cls, hatch=None)
                   for i, cls in enumerate(CLASSES)]
        if has_post:
            panels += [dict(data=post_share_grids[i], cmap="viridis", norm=None,
                            title="%s\nposterior mean P" % cls, hatch=None)
                       for i, cls in enumerate(CLASSES)]
            panels.append(dict(data=nyso_grid, cmap="magma", norm=None,
                               title="N($P$(YSO)$>$0.5)", hatch=None))
        n_panels = len(panels)

        # `page_geometry` (ported and generalised from the earlier
        # package's `sesnacomplete.bms_prior.validation.sky_atlas`) picks
        # the column/row split that gives the largest common panel size
        # on the fixed page.
        aspect = n_x / float(n_y)
        geom = page_geometry(aspect, n_panels, page_w, page_h)
        cols = geom["cols"]
        last_row_of_col = _outer_rows(n_panels, cols)

        plot_style.apply_style()
        fig = plot_style.new_sized_figure(page_w, page_h)

        axes_ims = []
        for i, p in enumerate(panels):
            row, col = divmod(i, cols)
            rect = _panel_rect(geom, i, page_w, page_h)
            show_dec = col == 0
            show_ra = row == last_row_of_col[col]
            ax, im = _add_panel(fig, rect, page_w, page_h, wcs, p["data"], p["cmap"],
                                 norm=p["norm"], title=p["title"],
                                 show_dec=show_dec, show_ra=show_ra)
            if p["hatch"] is not None:
                # Hatch only where the admitted footprint itself is
                # low-coverage: `coverage_grid` is already NaN outside
                # the footprint (sec. 8's own reprojection mask), and a
                # NaN comparison is False, so this never hatches the
                # white area outside the admitted pixels.
                ax.contourf(p["hatch"].astype(float), levels=[0.5, 1.5],
                            hatches=["//"], colors="none")
            _colorbar(fig, im, _bar_rect(rect, geom), page_w, page_h)
            axes_ims.append((ax, im))

        caption = None if has_post else \
            "posterior atlas (P11) absent for this region -- prior atlas only"

        total_predicted = float(prior["attrs"].get("TOTAL_PREDICTED", np.nan))
        total_observed = float(prior["attrs"].get("TOTAL_OBSERVED", np.nan))
        surveyed_area = float(prior["attrs"].get("SURVEYED_AREA_DEG2", np.nan))
        ratio_po = total_predicted / total_observed if total_observed else float("nan")
        ratio_line = "total-count ratio: " + ", ".join(
            "%s %.3g" % (cls, ratio) for cls, ratio in ratios)
        title = ("%s -- predicted/observed = %.4g/%.4g = %.3f, surveyed area %.4g deg$^2$%s"
                  % (region, total_predicted, total_observed, ratio_po, surveyed_area,
                     "" if caption is None else " (%s)" % caption))
        fig.suptitle(title + "\n" + ratio_line, fontsize=11, y=1.0 - 0.10 / page_h)

        out_dir = os.path.join(config.data_root, "bmstp", "atlas", "figures")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for fmt in formats:
            path = os.path.join(out_dir, "prior-atlas_%s.%s" % (region, fmt))
            # No `bbox_inches="tight"`: that re-crops to content and
            # drifts the saved size away from the requested page size.
            fig.savefig(path, dpi=150)
            paths.append(path)
        plt.close(fig)

        st.done(paths[0], n_x=n_x, n_y=n_y, has_posterior=int(has_post),
                cols=geom["cols"], rows=geom["rows"], panel_scale_in=geom["panel_h"])
    return paths


def build(config, regions=None, formats=("png", "pdf"), page=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN)):
    """Per region, one prior-atlas figure (and its posterior row where
    P11 exists) written under `bmstp/atlas/figures/` -- rendering, not a
    build (no product is stored at display resolution), so a region
    without a prior atlas yet is skipped rather than failed."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region(config, region, formats, page_w=page[0], page_h=page[1])


def _parse_page(text):
    w, _, h = text.lower().partition("x")
    return float(w), float(h)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--page", default="%gx%g" % (PAGE_WIDTH_IN, PAGE_HEIGHT_IN),
                        help="page size in inches, WxH (default: a 16x9 slide)")
    args = parser.parse_args()
    config = config_module.load(args.config)
    formats = tuple(s.strip() for s in args.formats.split(",") if s.strip())
    build(config, regions=args.regions, formats=formats, page=_parse_page(args.page))


if __name__ == "__main__":
    _main()
