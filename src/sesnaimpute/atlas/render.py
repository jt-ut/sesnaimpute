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

Top row: the column `A_K` (log scale) and the total predicted catalogued
density `Sigma_C N_CAT_C` (deg^-2), hatched where the surveyed
(IRAC-coverage) fraction is below 0.5. Middle row: the prior share
`SHARE_C` per class, in class order, each on its own colour scale (a
shared 0-1 bar hides the spatial variation of the low-share classes,
YSO first among them); the region's total-count ratio (`RATIO_<CLS>`)
moves to the figure's caption line rather than the panel title. Bottom
row, when the posterior atlas (P11) exists: the posterior mean `MEAN_P_C`
per class, likewise each on its own scale, plus a seventh small panel,
`N_YSO_ABOVE_HALF`.

Colour maps and scales follow the convention the earlier package's own
sky-atlas figure used (`sesnacomplete.bms_prior.validation.sky_atlas`):
the measured dust column on `magma`, every class quantity (share,
posterior mean, raw density) on `viridis`, and a panel's colour bar
inset into the panel itself rather than beside it, since an inset bar
costs the layout nothing while a row of six side-by-side bars would
eat the space the maps need.
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


def _add_panel(fig, spec, wcs, data, cmap, vmin=None, vmax=None, norm=None, title="",
               show_dec=True, show_ra=True, title_size=8):
    """One sky panel on the shared display WCS. `show_dec`/`show_ra`
    gate the tick *labels* only (every panel keeps its ticks and grid,
    since every panel is the same field, sec. 8's "shared sky frame") --
    only the leftmost panel of a row and the figure's bottom row need
    the numbers repeated. No colorbar is added here: `fig.colorbar`
    creates a new axes appended after every existing one, so a colorbar
    added panel-by-panel ends up drawn *underneath* a later column's
    opaque background and its tail is painted over -- every colorbar in
    this figure is added in one pass after all panels exist instead."""
    ax = fig.add_subplot(spec, projection=wcs)
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_title(title, fontsize=title_size)
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


def _inset_colorbar(fig, ax, im):
    """A colour bar inset into the panel itself, ticks on its left so
    the labels stay inside the panel's own box rather than spilling
    into the next column -- the earlier package's sky-atlas convention
    (`sesnacomplete.bms_prior.validation.sky_atlas._inset_colorbar`): a
    bar drawn beside a panel costs its width from every column on the
    page, so a row of six narrow class panels has no room for six
    side-by-side bars, but an inset bar costs the layout nothing. A
    white translucent backing keeps the bar and its ticks legible over
    the panel's own image; `%.2g` keeps a tick's own exponent (for a
    share as small as YSO's) inside the tick label itself, rather than
    a separate offset annotation that has nowhere narrow to sit."""
    x, y, w, h = 0.58, 0.08, 0.06, 0.34
    ax.add_patch(plt.Rectangle((x - 0.30, y - 0.05), w + 0.33, h + 0.10,
                                transform=ax.transAxes, facecolor="white",
                                alpha=0.78, edgecolor="none", zorder=4))
    cax = ax.inset_axes([x, y, w, h], zorder=5)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.locator = MaxNLocator(nbins=2)
    cbar.formatter = FuncFormatter(lambda v, _pos: "%.2g" % v)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=5.5, length=2, pad=1.0)
    cbar.outline.set_linewidth(0.5)
    return cbar


def build_region(config, region, formats):
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

        # Figure about 16 wide; height set from the display grid's own
        # aspect ratio so a panel's box is close to square rather than
        # letterboxed (every panel shares one equal-aspect WCS). One
        # flat 9-column grid for the whole figure (row 1's two maps each
        # span 3 of the 9 columns -- about a third of the figure width;
        # rows 2-3's six class panels each take one of those columns, so
        # a share panel and its posterior match in size); each class
        # panel's colour bar sits inset inside the panel itself (column 6
        # is unused spacing), and the N(P(YSO)>0.5) panel sits in column
        # 7 at the same size as the six, with column 8 for its own bar.
        n_rows = 3 if has_post else 2
        aspect = n_y / float(n_x)
        fig_w = 16.0
        col_widths = [1, 1, 1, 1, 1, 1, 0.35, 1, 0.35]
        unit_w = fig_w / sum(col_widths)
        row1_h = 3 * unit_w * aspect + 1.3
        row23_h = unit_w * aspect + 0.9
        height_ratios = [row1_h] + [row23_h] * (n_rows - 1)
        fig_h = max(sum(height_ratios) + 1.0, 8.0)
        # The package house style (`sesnaimpute.plot_style`) before any
        # panel is drawn, so titles/labels come out bold in its font;
        # `new_sized_figure` is the module's own exception for a page
        # size set by the data (here, the region's footprint aspect)
        # rather than a fixed choice from `FIGURE_SIZES`.
        plot_style.apply_style()
        fig = plot_style.new_sized_figure(fig_w, fig_h)
        gs = fig.add_gridspec(n_rows, 9, width_ratios=col_widths, height_ratios=height_ratios,
                               wspace=0.65, hspace=0.7)

        ax_col, im_col = _add_panel(fig, gs[0, 0:3], wcs, col_grid, "magma",
                                     norm=_log_norm(col_grid), title="column A_K (mag)",
                                     show_dec=True, show_ra=False, title_size=10)
        ax_dens, im_dens = _add_panel(fig, gs[0, 3:6], wcs, density_grid, "viridis",
                                       norm=_log_norm(density_grid),
                                       title="predicted catalogued sources (deg$^{-2}$)",
                                       show_dec=False, show_ra=False, title_size=10)
        # Hatch only where the admitted footprint itself is low-coverage:
        # `coverage_grid` is already NaN outside the footprint (sec. 8's
        # own reprojection mask), and a NaN comparison is False, so this
        # never hatches the white area outside the admitted pixels.
        low_coverage = coverage_grid < 0.5
        if np.any(low_coverage):
            ax_dens.contourf(low_coverage.astype(float), levels=[0.5, 1.5],
                              hatches=["//"], colors="none")

        share_row, post_row = 1, 2

        # Every class quantity (share, posterior mean) on its own colour
        # scale, inset into its own panel: a shared 0-1 bar hides the
        # spatial variation of the low-share classes (YSO first among
        # them), which is exactly what a reader needs to see here.
        ratios = []
        for i, cls in enumerate(CLASSES):
            ratios.append((cls, float(prior["attrs"].get("RATIO_%s" % cls, np.nan))))
            ax, im = _add_panel(fig, gs[share_row, i], wcs, share_grids[i], "viridis",
                                 title="%s\nprior share" % cls,
                                 show_dec=(i == 0), show_ra=(not has_post), title_size=9)
            _inset_colorbar(fig, ax, im)

        if has_post:
            for i, cls in enumerate(CLASSES):
                ax, im = _add_panel(fig, gs[post_row, i], wcs, post_share_grids[i], "viridis",
                                     title="%s\nposterior mean P" % cls,
                                     show_dec=(i == 0), show_ra=True, title_size=9)
                _inset_colorbar(fig, ax, im)
            ax_nyso, im_nyso = _add_panel(fig, gs[post_row, 7], wcs, nyso_grid, "magma",
                                          title="N(P(YSO)>0.5)", show_dec=False, show_ra=True, title_size=9)
            caption = None
        else:
            caption = "posterior atlas (P11) absent for this region -- prior atlas only"

        # Every colorbar added last, after every panel axes exists:
        # `fig.colorbar` appends a new axes on top of whatever already
        # exists, so adding one between two image panels leaves it
        # underneath (hence painted over by) any panel created after it.
        # (Columns 6 and 8 of the grid, once the shared row bars' and
        # N(P(YSO)>0.5)'s home, are left as plain spacing now that every
        # class panel carries its own inset bar.)
        fig.colorbar(im_col, ax=ax_col, fraction=0.046, pad=0.05)
        fig.colorbar(im_dens, ax=ax_dens, fraction=0.046, pad=0.05)
        if has_post:
            fig.colorbar(im_nyso, ax=ax_nyso, fraction=0.046, pad=0.05)

        total_predicted = float(prior["attrs"].get("TOTAL_PREDICTED", np.nan))
        total_observed = float(prior["attrs"].get("TOTAL_OBSERVED", np.nan))
        surveyed_area = float(prior["attrs"].get("SURVEYED_AREA_DEG2", np.nan))
        ratio_po = total_predicted / total_observed if total_observed else float("nan")
        ratio_line = "total-count ratio: " + ", ".join(
            "%s %.3g" % (cls, ratio) for cls, ratio in ratios)
        title = ("%s -- predicted/observed = %.4g/%.4g = %.3f, surveyed area %.4g deg$^2$%s"
                  % (region, total_predicted, total_observed, ratio_po, surveyed_area,
                     "" if caption is None else " (%s)" % caption))
        fig.suptitle(title + "\n" + ratio_line, fontsize=12)

        out_dir = os.path.join(config.data_root, "bmstp", "atlas", "figures")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for fmt in formats:
            path = os.path.join(out_dir, "prior-atlas_%s.%s" % (region, fmt))
            # No `bbox_inches="tight"`: that re-crops to content and
            # drifts the saved size away from the intended ~16x12in.
            fig.savefig(path, dpi=150)
            paths.append(path)
        plt.close(fig)

        st.done(paths[0], n_x=n_x, n_y=n_y, has_posterior=int(has_post))
    return paths


def build(config, regions=None, formats=("png", "pdf")):
    """Per region, one prior-atlas figure (and its posterior row where
    P11 exists) written under `bmstp/atlas/figures/` -- rendering, not a
    build (no product is stored at display resolution), so a region
    without a prior atlas yet is skipped rather than failed."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region(config, region, formats)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()
    config = config_module.load(args.config)
    formats = tuple(s.strip() for s in args.formats.split(",") if s.strip())
    build(config, regions=args.regions, formats=formats)


if __name__ == "__main__":
    _main()
