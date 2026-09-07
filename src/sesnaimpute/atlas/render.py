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
`SHARE_C` per class, in class order, 0-1, titled with the region's
total-count ratio (`RATIO_<CLS>`). Bottom row, when the posterior atlas
(P11) exists: the posterior share `MEAN_P_C` per class on the same 0-1
scale, plus a seventh small panel, `N_YSO_ABOVE_HALF`.
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
from scipy.ndimage import gaussian_filter

from sesnaimpute import config as config_module
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


def _add_panel(fig, gs, row, col, wcs, data, cmap, vmin=None, vmax=None, norm=None, title=""):
    """One sky panel on the shared display WCS: RA/Dec tick marks and
    tick labels (sec. 8's "shared sky frame with RA/Dec ticks"), no
    repeated per-panel axis-name text -- fifteen copies of "pos.eq.ra"/
    "pos.eq.dec" would only crowd the tick numbers they sit beside."""
    ax = fig.add_subplot(gs[row, col], projection=wcs)
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.16)
    ax.set_title(title, fontsize=8)
    for i in (0, 1):
        # `set_axislabel("")` alone is not enough: WCSAxes treats an
        # empty label as "unset" and redraws its own default
        # ("pos.eq.ra"/"pos.eq.dec") unless auto-labelling is off too.
        ax.coords[i].set_auto_axislabel(False)
        ax.coords[i].set_axislabel("")
        ax.coords[i].set_ticklabel(size=6)
        ax.coords[i].set_ticks(number=3)
    ax.coords.grid(color="white", alpha=0.3, linestyle="solid", linewidth=0.4)
    return ax


def _log_norm(grid):
    """A `LogNorm` spanning the grid's own finite positive range -- the
    log colour scale sec. 8 asks for the column and density panels."""
    finite = grid[np.isfinite(grid) & (grid > 0)]
    if finite.size == 0:
        return LogNorm(vmin=1e-6, vmax=1.0)
    return LogNorm(vmin=float(finite.min()), vmax=float(finite.max()))


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

        # Row heights matched to the display grid's own aspect ratio
        # (WCSAxes holds equal aspect on the sky): a row-1 panel spans
        # 3 of the 6 regular columns, so it needs 3x a regular row's
        # height to fill its box without letterboxing.
        n_rows = 3 if has_post else 2
        aspect = n_y / float(n_x)
        fig_w = 16.0
        unit_w = fig_w / 6.7
        height_ratios = [3, 1, 1] if has_post else [3, 1]
        fig_h = unit_w * aspect * sum(height_ratios) + 2.5
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(n_rows, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.7],
                               height_ratios=height_ratios, hspace=0.55, wspace=0.95)

        _add_panel(fig, gs, 0, slice(0, 3), wcs, col_grid, "cividis",
                   norm=_log_norm(col_grid), title="column $A_K$ (mag)")
        ax_dens = _add_panel(fig, gs, 0, slice(3, 6), wcs, density_grid, "magma",
                              norm=_log_norm(density_grid),
                              title=r"$\Sigma_C N_{CAT,C}$ (deg$^{-2}$)")
        low_coverage = (coverage_grid < 0.5) | ~np.isfinite(coverage_grid)
        if np.any(low_coverage) and not np.all(low_coverage):
            ax_dens.contourf(low_coverage.astype(float), levels=[0.5, 1.5],
                              hatches=["//"], colors="none")

        for i, cls in enumerate(CLASSES):
            ratio = float(prior["attrs"].get("RATIO_%s" % cls, np.nan))
            _add_panel(fig, gs, 1, i, wcs, share_grids[i], "viridis", vmin=0.0, vmax=1.0,
                       title="%s share (ratio %.3g)" % (cls, ratio))

        if has_post:
            for i, cls in enumerate(CLASSES):
                _add_panel(fig, gs, 2, i, wcs, post_share_grids[i], "viridis", vmin=0.0, vmax=1.0,
                           title="%s posterior mean" % cls)
            _add_panel(fig, gs, 2, 6, wcs, nyso_grid, "magma",
                       title="N(P(YSO)>0.5)")
            caption = ""
        else:
            caption = "posterior atlas (P11) absent for this region -- prior atlas only"

        total_predicted = float(prior["attrs"].get("TOTAL_PREDICTED", np.nan))
        total_observed = float(prior["attrs"].get("TOTAL_OBSERVED", np.nan))
        surveyed_area = float(prior["attrs"].get("SURVEYED_AREA_DEG2", np.nan))
        ratio_po = total_predicted / total_observed if total_observed else float("nan")
        title = ("%s -- predicted/observed = %.4g/%.4g = %.3f, surveyed area %.4g deg$^2$"
                  % (region, total_predicted, total_observed, ratio_po, surveyed_area))
        if caption:
            title = title + "\n" + caption
        fig.suptitle(title, fontsize=12)

        out_dir = os.path.join(config.data_root, "bmstp", "atlas", "figures")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for fmt in formats:
            path = os.path.join(out_dir, "prior-atlas_%s.%s" % (region, fmt))
            fig.savefig(path, dpi=150, bbox_inches="tight")
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
