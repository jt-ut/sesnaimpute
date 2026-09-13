"""The sky atlas figures (SPEC_BMSTP_DRAFT.md sec. 8; IMPLEMENTATION_
BMSTP_DRAFT.md sec. 1.2 P6, sec. 1.3 P11). Three figures per region: the
prior atlas's two pages (`bmstp/atlas/figures/prior-atlas-intrinsic_<R>`
and `prior-atlas-selection_<R>`, from P6) and the posterior atlas's one
(`fittp/atlas/figures/posterior-atlas_<R>`, from P11), each reprojecting
the SAME footprint -- `catalog.depth_grid`'s admitted
nside-512 pixels (P6 and P11 are themselves both built on that axis) --
onto its own tangent-plane display grid of 1' pixels centred on the
region: the pixel's nearest nside-512 value (healpy `ang2pix`), then a
Gaussian smoothing of one nside-512 pixel width (6.9') so pixel edges do
not show. Nothing at display resolution is stored -- the figures are
rendered at plot time only. The granule map's own frame is galactic
(`granules/build.py`'s `ang2pix(..., gl, gb, ...)`); the display grid and
its RA/Dec ticks are equatorial, so every lookup crosses frames once
through astropy, never through a local approximation. Reading the same
depth-grid footprint for both figures (rather than each atlas's own
axis) makes their display grids identical pixel for pixel.

Every panel of a figure is drawn at the same size and shares the
region's own display-grid aspect. The prior atlas is two pages per
region, one code path (`_build_prior_page`) parameterised by `view`:
`prior-atlas-intrinsic_<R>` reads the product's `N_ABOVE_C` (the density
above the pixel's own 4.5 um 50% completeness limit, SPEC_BMSTP_DRAFT.md
sec. 8) for the column-B density panel `Sigma_C N_ABOVE_C` and the six
class panels `N_ABOVE_C / Sigma N_ABOVE`; `prior-atlas-
selection_<R>` reads `N_CAT_C` (catalogued density) for the same slots,
`Sigma_C N_CAT_C` hatched where the surveyed IRAC-coverage fraction is
below 0.5. `atlas.captions` states which probability each class panel
draws and the shared vocabulary; both are printed below the panel grid,
the page growing taller to hold them rather than shrinking the type.
The posterior figure carries the column `A_K`, the posterior
mean `MEAN_P_C` per class in class order, and `N_YSO_ABOVE_HALF` -- 8
panels. An admitted pixel with no sources of its own (P11's
`N_SOURCES == 0`, or a pixel P11 omits) is painted flat neutral grey on
every posterior-derived panel, masked out before the Gaussian smoothing
so it never bleeds into an occupied neighbour's own value, rather than
dropped as transparent; the posterior figure's caption says so. All
three pages' 8 panels fill the SAME fixed four-column, two-stacked-panel
grid (A/B | C/D | E/F | G/H: A = column, B = the row-1 counterpart --
the density panel for a prior page, the posterior YSO count for the
posterior page -- C..H the six classes in the same column pairing),
each page's own panel scale derived from its region's aspect
(`_atlas_page_size` below). The prior selection page's predicted/
observed ratio and bright-source ratios, and the posterior figure's
`N(P(YSO)>0.5)` count and two-band fraction, sit in each page's own
caption line rather than any panel title; the intrinsic page's caption
names the region only, since none of those selection-side numbers
describe it.

Colour maps and scales follow the convention the earlier package's own
sky-atlas figure used (`sesnacomplete.bms_prior.validation.sky_atlas`):
the measured column on `magma`, every class quantity (share, posterior
mean) on `viridis`. A prior page's six class panels draw a PROBABILITY,
of order one where a class holds the pixel, but each panel gets its OWN
LINEAR colour bar spanning its own finite range, ticked automatically
(`_panel_norm`): a shared 0-1 bar paints a rare class's whole spatial
structure in one flat colour near the bottom of the scale, hiding the
very structure the panel exists to show. The column
and density panels keep their own log scale. That earlier figure's own `page_geometry` insets a colour bar
into the panel when the footprint leaves clear room for one and sizes a
3x2 class block plus one wide dust panel to fill a page that grows to
fit its content; here every panel of both figures is an equal member of
the same fixed 4x2 grid, its own colour-bar strip inset into its own
right edge (`_panel_colorbar`) rather than a free-floating rectangle,
and the page itself (`_atlas_page_size`) is sized to that grid's own
content at the region's own aspect, so the panels always fill the page
with no letterboxing.
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
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter
from scipy.ndimage import gaussian_filter

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.atlas import captions

NSIDE = 512
CLASSES = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")

#: Panel titles and colourbar labels at a size a reader sees on a slide;
#: tick labels smaller -- the two atlas pages' own convention;
#: `atlas.protostars`'s own `_colorbar` path is unaffected, it is not
#: one of these two pages.
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 10

#: The two prior pages' own page title, bold, larger than a panel title
#: so the page's own subject reads first.
TITLE_FONTSIZE = 18

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
# page geometry -- both atlas figures share one fixed 4x2 panel grid,
# sized to its own content (`_atlas_page_size`) so the panels fill the
# page at every region's own aspect, each panel's own colour bar inset
# into its own right edge (`_panel_colorbar`); `_colorbar`'s
# free-floating-rectangle path serves `atlas.protostars`'s own layout.
# ---------------------------------------------------------------------------

#: The page `atlas.protostars` draws on, inches; the two atlas pages size
#: themselves to their content (`_atlas_page_size`).
PAGE_WIDTH_IN, PAGE_HEIGHT_IN = 16.0, 9.0

#: Page furniture, inches: room outside the panel grid for the left
#: column's Dec tick labels, the bottom row's RA tick labels, and the
#: page title plus the group heading above the panel titles.
#: Widened over its earlier 0.55 in for the 13 pt panel titles/labels at
#: a narrow region's own aspect, whose Dec tick labels otherwise overhang
#: the left edge by 0.03-0.17 in.
MARGIN_LEFT_IN, MARGIN_RIGHT_IN = 0.75, 0.15
MARGIN_TOP_IN, MARGIN_BOTTOM_IN = 0.85, 0.55
GAP_X_IN, GAP_Y_IN = 0.10, 0.12

#: `MARGIN_TOP_IN`'s own budget for the prior page's title (top pad,
#: then the `TITLE_FONTSIZE` line, measured down from the page's own
#: top edge, `va="top"`); the group heading sits just above the panel
#: grid's own top edge (`HEADING_ABOVE_GRID_IN`, `va="bottom"`), so it
#: reads as the heading of the three class columns beneath it. The
#: posterior figure's own suptitle is unaffected, since it draws by its
#: own fraction of the page height rather than from these pads.
TITLE_TOP_PAD_IN = 0.14
HEADING_ABOVE_GRID_IN = 0.06

#: The left column (the column and density panels) meets the three
#: class columns at this gap rather than `GAP_X_IN`, so the rule drawn
#: through it (`_build_prior_page`) has room; the other two inter-
#: column gaps keep `GAP_X_IN`.
SEPARATOR_GAP_IN = 0.35

#: A panel's own colour-bar strip (with its tick labels) and title
#: strip, each as a fraction of the panel's own height `s` -- so, like
#: the earlier module's inset bar, they cost more page space on a
#: bigger panel but never crowd a small one. Sized for a thin bar plus
#: three two/three-character ticks and, where a panel's colour bar
#: carries one, a single rotated word of label text.
BAR_WIDTH_FRACTION = 0.18
TITLE_HEIGHT_FRACTION = 0.20


def _panel_rect(geom, index, page_w, page_h):
    """The panel axes' `(x, y, w, h)` in inches from the page's lower
    left, for panel `index` filled row-major into `geom`'s grid (as
    `_atlas_page_size` lays it out), and the grid's own occupied width/
    height (for centring the grid on the page). `geom["col_gaps"]`
    holds the `cols - 1` inter-column gaps in order (the prior page's
    own first gap is `SEPARATOR_GAP_IN`, wider than the rest, for the
    separator rule; the posterior figure's are all `GAP_X_IN`)."""
    cols, rows = geom["cols"], geom["rows"]
    panel_w, panel_h = geom["panel_w"], geom["panel_h"]
    bar_w, title_h = geom["bar_w"], geom["title_h"]
    col_gaps = geom["col_gaps"]
    content_w = cols * (panel_w + bar_w) + sum(col_gaps)
    content_h = rows * (panel_h + title_h) + (rows - 1) * GAP_Y_IN
    usable_w = page_w - MARGIN_LEFT_IN - MARGIN_RIGHT_IN
    usable_h = page_h - MARGIN_TOP_IN - MARGIN_BOTTOM_IN
    origin_x = MARGIN_LEFT_IN + 0.5 * (usable_w - content_w)
    origin_y = MARGIN_BOTTOM_IN + 0.5 * (usable_h - content_h)
    row, col = divmod(index, cols)
    x = origin_x + col * (panel_w + bar_w) + sum(col_gaps[:col])
    y = origin_y + (rows - 1 - row) * (panel_h + title_h + GAP_Y_IN)
    return x, y, panel_w, panel_h


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

def _read_prior(path, region, require_above):
    """The prior atlas (P6), sorted by pixel for the reprojection's
    `searchsorted` lookup. `N_ABOVE_<C>` (the intrinsic view's own
    per-pixel density above the pixel's own 4.5 um 50% completeness limit,
    SPEC_BMSTP_DRAFT.md sec. 8's rule) is required only for the intrinsic
    page (`require_above`; the posterior page and the selection page read
    this product only for `A_COL_K`/`N_CAT_C`) -- a product that lacks it
    fails here, by name, rather than the intrinsic page silently falling
    back to some other quantity."""
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        a_k = np.asarray(f["A_COL_K"][:], dtype=np.float64)
        coverage = np.asarray(f["COVERAGE"][:], dtype=np.float64)
        n_cat = np.stack([np.asarray(f["N_CAT_%s" % c][:], dtype=np.float64) for c in CLASSES], axis=1)
        above = None
        if require_above:
            missing = [c for c in CLASSES if ("N_ABOVE_%s" % c) not in f]
            if missing:
                raise KeyError(
                    "atlas.render [%s]: prior atlas carries no N_ABOVE_%s -- run "
                    "RUNBOOKtp.sh's 'PY sesnaimpute.bmstp.atlas' line to rebuild it"
                    % (region, missing[0]))
            above = np.stack([np.asarray(f["N_ABOVE_%s" % c][:], dtype=np.float64) for c in CLASSES], axis=1)
        attrs = dict(f.attrs)
    order = np.argsort(pix)
    return dict(pix=pix[order], a_k=a_k[order], coverage=coverage[order],
                n_cat=n_cat[order],
                above=above[order] if above is not None else None,
                attrs=attrs)


def _read_posterior(path):
    """The posterior atlas (P11), sorted by pixel; `n_sources` is the
    per-pixel count a source-less pixel (grey on the figure) reads as 0."""
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        n_sources = np.asarray(f["N_SOURCES"][:], dtype=np.int64)
        mean_p = np.stack([np.asarray(f["MEAN_P_%s" % c][:], dtype=np.float64) for c in CLASSES], axis=1)
        n_yso_half = np.asarray(f["N_YSO_ABOVE_HALF"][:], dtype=np.float64)
    order = np.argsort(pix)
    return dict(pix=pix[order], n_sources=n_sources[order],
                mean_p=mean_p[order], n_yso_half=n_yso_half[order])


def _read_depth_grid(config, region):
    """The admitted-pixel footprint (`catalog.depth_grid`, sec. 3.3) --
    the SAME footprint P6 and P11 are themselves built on (sec. 8), read
    here directly so the prior and posterior figures share it exactly,
    sorted by pixel."""
    path = config_module.product_path(config, "catalog", "sesna", "depth-grid", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
    return np.sort(pix)


def _align(footprint_pix, src_pix, values, fill):
    """`values` (aligned to the sorted `src_pix`) reindexed onto the
    sorted `footprint_pix`, `fill` where `footprint_pix` holds a pixel
    `src_pix` lacks -- so every panel of a figure reads the identical
    admitted-pixel axis regardless of which product supplied the data."""
    loc = np.minimum(np.searchsorted(src_pix, footprint_pix), src_pix.size - 1)
    found = src_pix[loc] == footprint_pix
    out = np.full((footprint_pix.size,) + values.shape[1:], fill,
                  dtype=np.result_type(values, fill))
    out[found] = values[loc[found]]
    return out


def _footprint_geometry(pix):
    """The display WCS, grid-pixel lookup and shape for the admitted
    footprint `pix` -- computed once from the depth grid and shared by
    every panel of both figures, so their display grids coincide."""
    ra0, dec0 = _region_centre_icrs(pix)
    n_x, n_y = _grid_size(ra0, dec0, pix)
    wcs = _display_wcs(ra0, dec0, n_x, n_y)
    grid_pix = _grid_pixels(wcs, n_x, n_y)
    return dict(n_x=n_x, n_y=n_y, wcs=wcs, grid_pix=grid_pix, shape=(n_y, n_x))


def _gap_grid(pix_sorted, gap_1d, grid_pix_flat, shape):
    """Per display cell, whether its nearest admitted pixel (the same
    nearest lookup `_reproject` makes) is source-less (`gap_1d`, aligned
    to `pix_sorted`) -- computed before any smoothing, so a source-less
    pixel's own cells are painted flat grey rather than shown as
    whatever a smoothed occupied neighbour bleeds into them (sec. 8,
    "mask before smoothing ... paint the mask grey")."""
    loc = np.minimum(np.searchsorted(pix_sorted, grid_pix_flat), pix_sorted.size - 1)
    found = pix_sorted[loc] == grid_pix_flat
    return (found & gap_1d[loc]).reshape(shape)


def _two_band_fraction(config, region):
    """The fraction of the region's classified sources detected in
    exactly two of the eight bands (P8's `N_DETECTED`; SPEC_BMSTP_DRAFT.md
    "the two-band population", reported beside every YSO count). P8
    carries no such attribute (`fittp.classify`'s own `st.done` line
    computes it the same way, then prints it, without storing it), so it
    is read here as a direct count over that one small column, never
    P8's large per-source arrays (`CANDIDATE_FLUX`, `FLUX_IMPUTED_COV`)."""
    path = config_module.product_path(config, "fittp", "classification", "posterior", "source", region=region)
    with h5py.File(path, "r") as f:
        n_detected = np.asarray(f["N_DETECTED"][:])
    return float(np.mean(n_detected == 2)) if n_detected.size else float("nan")


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
               show_dec=True, show_ra=True, title_size=LABEL_FONTSIZE, grey_grid=None):
    """One sky panel at `rect` (inches, lower-left origin), on the
    shared display WCS. `show_dec`/`show_ra` gate the tick *labels*
    only (every panel keeps its ticks and grid, since every panel is
    the same field, sec. 8's "shared sky frame") -- only the grid's
    left column and each column's own bottom-most panel need the
    numbers repeated. `grey_grid`, where given, paints those display
    cells flat neutral grey over the panel's own smoothed value (sec.
    8: a source-less admitted pixel, never dropped as transparent)."""
    ax = fig.add_axes(_frac(rect, page_w, page_h), projection=wcs)
    im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm)
    if grey_grid is not None and np.any(grey_grid):
        overlay = np.zeros(grey_grid.shape + (4,))
        overlay[..., :3] = 0.65
        overlay[..., 3] = grey_grid.astype(float)
        ax.imshow(overlay, origin="lower")
    ax.set_title(title, fontsize=title_size, pad=4)
    for i in (0, 1):
        # `set_axislabel("")` alone is not enough: WCSAxes treats an
        # empty label as "unset" and redraws its own default
        # ("pos.eq.ra"/"pos.eq.dec") unless auto-labelling is off too.
        ax.coords[i].set_auto_axislabel(False)
        ax.coords[i].set_axislabel("")
        ax.coords[i].set_ticklabel(size=TICK_FONTSIZE)
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
    """A thin colour bar in its own free-floating reserved strip at
    `rect` (inches), scaled to that panel's own data range, for
    `atlas.protostars`'s own layout, which computes its own `rect`; the
    two atlas pages use `_panel_colorbar`'s inset bar."""
    cax = fig.add_axes(_frac(rect, page_w, page_h))
    cbar = fig.colorbar(im, cax=cax)
    cbar.locator = MaxNLocator(nbins=3)
    cbar.formatter = FuncFormatter(lambda v, _pos: "%.2g" % v)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=7, length=2, pad=1.0)
    cbar.outline.set_linewidth(0.5)
    return cbar


def _log_tick_values(vmin, vmax):
    """Candidate tick values at 1, 2, 5 x 10^k inside `[vmin, vmax]`:
    the 2s dropped once there are more than four candidates, then every
    non-decade value dropped too if still more than four -- so a log
    bar never carries more ticks than a reader can place, and a panel
    whose range spans less than a decade still gets at least one
    intermediate value alongside its two ends."""
    if not (vmin and vmax and vmax > vmin > 0):
        return [v for v in (vmin, vmax) if v]
    k_min = int(np.floor(np.log10(vmin)))
    k_max = int(np.ceil(np.log10(vmax)))
    tol_lo, tol_hi = vmin * (1 - 1e-9), vmax * (1 + 1e-9)
    candidates = [(m, m * 10.0 ** k) for k in range(k_min, k_max + 1) for m in (1, 2, 5)]
    candidates = [(m, v) for m, v in candidates if tol_lo <= v <= tol_hi]
    if len(candidates) > 4:
        candidates = [(m, v) for m, v in candidates if m != 2]
    if len(candidates) > 4:
        candidates = [(m, v) for m, v in candidates if m == 1]
    values = sorted(v for _, v in candidates)
    return values if values else [vmin, vmax]


def _format_log_tick(v):
    """One log-bar tick label with no exponent notation: an integer
    value prints as an integer (`5000`, not `5e+03`), a fraction prints
    with the digits it needs (`0.1`, not `1e-01`)."""
    if v == 0:
        return "0"
    rounded = round(v)
    if abs(v - rounded) <= 1e-6 * abs(v) and rounded != 0:
        return "%d" % rounded
    return ("%f" % v).rstrip("0").rstrip(".")


def _panel_colorbar(fig, ax, im, label=None, log=False, ticks=None):
    """The panel's own colour bar, inset into its own right edge and
    spanning exactly its height -- axes coordinates work for WCSAxes, so
    no free-floating bar rectangle is needed for either atlas page.
    `ticks`, for a panel on a fixed scale shared with other panels,
    labels exactly those values; a panel on its own auto-ranged scale
    (e.g. a prior page's class panels, `_panel_norm`) passes `None` and
    gets the automatic ticks below.

    `log`, for a `LogNorm`-scaled panel, ticks at `_log_tick_values`'s
    1/2/5-per-decade candidates, minor ticks unlabelled and of zero
    length, every label plain (`_format_log_tick`, no exponent
    notation).  A linear panel ticks on `MaxNLocator(nbins=3, steps=[1,
    2, 2.5, 5, 10])`; where its range's top sits below 0.01, the ticks
    are labelled in units of one power of ten (`1 2 3`, say) and that
    multiplier is printed once above the bar, so no tick label reads
    like `2e-04`."""
    cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
    cbar = fig.colorbar(im, cax=cax)
    multiplier_text = None
    if ticks is not None:
        cbar.locator = FixedLocator(list(ticks))
        cbar.formatter = FuncFormatter(lambda v, _pos: "%g" % v)
    elif log:
        vmin, vmax = im.norm.vmin, im.norm.vmax
        cbar.locator = FixedLocator(_log_tick_values(vmin, vmax))
        cbar.formatter = FuncFormatter(lambda v, _pos: _format_log_tick(v))
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
        cbar.ax.tick_params(which="minor", length=0)
    else:
        vmax = im.norm.vmax
        cbar.locator = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10])
        if vmax and vmax > 0 and vmax < 0.01:
            k = int(np.floor(np.log10(vmax)))
            scale = 10.0 ** k
            cbar.formatter = FuncFormatter(lambda v, _pos, _scale=scale: "%g" % (v / _scale))
            multiplier_text = r"$\times 10^{%d}$" % k
        else:
            cbar.formatter = FuncFormatter(lambda v, _pos: "%g" % v)
    cbar.update_ticks()
    if multiplier_text is not None:
        cbar.ax.text(0.5, 1.04, multiplier_text, transform=cbar.ax.transAxes,
                      ha="center", va="bottom", fontsize=TICK_FONTSIZE)
    # Tick and label sizes a reader sees on a slide, the two atlas
    # pages' own convention.
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=2, pad=1.0)
    cbar.outline.set_linewidth(0.5)
    if label:
        cbar.set_label(label, fontsize=LABEL_FONTSIZE)
    return cbar


#: An atlas page's own panel height, inches -- the one free scale of its
#: fixed four-column, two-stacked-panel layout (below), shared by the
#: prior and the posterior figure alike: the page itself is sized to
#: this content, not the other way around, so the panels always fill
#: the page regardless of a region's own footprint aspect.
ATLAS_PANEL_HEIGHT_IN = 3.0


def _atlas_page_size(aspect, cols=4, rows=2, panel_h=ATLAS_PANEL_HEIGHT_IN, col_gaps=None):
    """`(page_w, page_h, geom)` for an atlas page (the prior figure or
    the posterior figure, both fixed `cols`x`rows` grids): `cols`x`rows`
    equal panels at the region's own `aspect`, each `panel_h` tall plus
    its own colour-bar strip and title strip, the page margins (left/
    right Dec and RA tick room, top title room, bottom RA-label room)
    added once around that content -- so the panel grid always fills the
    page exactly, with no letterboxing in either dimension. `col_gaps`,
    the `cols - 1` inter-column gaps in order, defaults to `GAP_X_IN`
    throughout (the posterior figure's own uniform grid); the prior
    page passes its own wider first gap for the separator rule."""
    if col_gaps is None:
        col_gaps = [GAP_X_IN] * (cols - 1)
    panel_w = panel_h * aspect
    bar_w = panel_h * BAR_WIDTH_FRACTION
    title_h = panel_h * TITLE_HEIGHT_FRACTION
    content_w = cols * (panel_w + bar_w) + sum(col_gaps)
    content_h = rows * (panel_h + title_h) + (rows - 1) * GAP_Y_IN
    page_w = MARGIN_LEFT_IN + MARGIN_RIGHT_IN + content_w
    page_h = MARGIN_TOP_IN + MARGIN_BOTTOM_IN + content_h
    geom = dict(cols=cols, rows=rows, panel_w=panel_w, panel_h=panel_h,
                bar_w=bar_w, title_h=title_h, col_gaps=col_gaps)
    return page_w, page_h, geom


#: The caption block's own type size and line pitch, inches -- chosen
#: small enough that `atlas.captions.vocabulary_block()`'s eleven terms
#: plus the two panel statements read as body text, not a footnote, at
#: the wrap width `_caption_layout` derives from the page.
CAPTION_FONT_SIZE = 8.0
CAPTION_LINE_HEIGHT_IN = 0.16
CAPTION_TOP_PAD_IN = 0.12
CAPTION_BOTTOM_PAD_IN = 0.25

#: Rough characters-per-inch for `CAPTION_FONT_SIZE` DejaVu Sans -- used
#: only to size the page's own caption strip, not to typeset it exactly
#: (matplotlib wraps nothing on its own), so the page grows enough that
#: the caption never overruns the bottom margin.
CAPTION_CHARS_PER_IN = 15.0


def _panel_norm(grid):
    """A LINEAR scale for one class panel's `P(C | ...)` colour bar,
    spanning that panel's OWN finite range (`vmin`/`vmax` at the grid's
    own nanmin/nanmax; `vmin` to `vmin + 1e-6` where the range is
    degenerate, so `Normalize` never divides by zero) rather than the
    shared 0-1 scale the two prior pages used to fix on every class
    panel: a rare class's whole probability range can sit within one
    tenth of that shared scale, painting its spatial structure in a
    single flat colour, so each panel is left to find its own range and
    its own automatic ticks, at the cost of the six panels no longer
    being directly comparable to each other."""
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1e-6)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return Normalize(vmin=vmin, vmax=vmax)


#: The two prior pages differ only in which stored quantity feeds the
#: density and class panels, and in what the caption says about it
#: (`atlas.captions`, never restated here) -- everything else in
#: `_build_prior_page` is the one shared code path the rules ask for.
VIEWS = {
    "intrinsic": dict(
        density_label="Prior Density",
        page_title="%s Prior Atlas (Intrinsic)",
        group_heading="P(class | pixel, above the 4.5 µm limit)",
        class_caption=captions.ATLAS_INTRINSIC_CLASS,
        total_caption=captions.ATLAS_INTRINSIC_TOTAL,
        coverage_outline=False,
    ),
    "selection": dict(
        density_label="Prior Density",
        page_title="%s Prior Atlas",
        group_heading="P(class | pixel, selected)",
        class_caption=captions.ATLAS_SELECTION_CLASS,
        total_caption=captions.ATLAS_SELECTION_TOTAL,
        coverage_outline=True,
    ),
}


def _caption_block(view, extra_line=None):
    """The text a prior page prints below its panel grid: the view's own
    two `atlas.captions` statements, then `extra_line` if given (the
    selection page's own total-count-check lines). The shared
    vocabulary itself is not printed here -- `captions.write_vocabulary`
    writes it once to its own file alongside the figures."""
    spec = VIEWS[view]
    parts = [spec["class_caption"], spec["total_caption"]]
    if extra_line:
        parts.append(extra_line)
    return "\n\n".join(parts)


def _caption_layout(text, page_w):
    """`(wrapped_text, height_in)`: `text` hard-wrapped at the page's own
    content width (an unwrapped paragraph would run off the page's right
    edge on a narrow region), and the height, inches, the page must grow
    by to hold exactly that wrapped text (rules: "the page growing to
    fit") -- `captions.caption_layout`, the one definition this page and
    `atlas.shapes`'s own caption strip both call, so the drawn block and
    the reserved strip always agree because both come from the one wrap."""
    wrap_chars = max(int((page_w - MARGIN_LEFT_IN - MARGIN_RIGHT_IN) * CAPTION_CHARS_PER_IN), 20)
    return captions.caption_layout(text, wrap_chars, CAPTION_LINE_HEIGHT_IN,
                                    CAPTION_TOP_PAD_IN, CAPTION_BOTTOM_PAD_IN)


def _require_depth_grid(config, region):
    """The depth-grid footprint, or `None` with a skip message -- both
    figures need it (sec. 8), P6 and P11 are each built on it already."""
    depth_path = config_module.product_path(config, "catalog", "sesna", "depth-grid", "hpx512", region=region)
    if not os.path.exists(depth_path):
        print("atlas.render [%s]: no depth grid -- run RUNBOOKtp.sh's "
              "'PY sesnaimpute.catalog.depth_grid' line first, skipped" % region, flush=True)
        return None
    return _read_depth_grid(config, region)


def _build_prior_page(config, region, formats, view):
    """Writes one prior-atlas page (8 panels: column, the class-summed
    density, the six class panels) under `bmstp/atlas/figures/` --
    `prior-atlas-intrinsic_<R>` (`A_C`, before the survey's selection) or
    `prior-atlas-selection_<R>` (`N_CAT_C`, after it), `view` selecting
    which stored quantity feeds the density and class panels (`VIEWS`)
    so the two pages are one code path rather than two copies that could
    drift apart."""
    spec = VIEWS[view]
    prior_path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    if not os.path.exists(prior_path):
        print("atlas.render [%s]: no prior atlas -- run RUNBOOKtp.sh's "
              "'PY sesnaimpute.bmstp.atlas' line first, skipped" % region, flush=True)
        return None
    footprint_pix = _require_depth_grid(config, region)
    if footprint_pix is None:
        return None
    prior = _read_prior(prior_path, region, require_above=(view == "intrinsic"))

    with progress.Stage("atlas.render.prior.%s" % view, region) as st:
        geom_grid = _footprint_geometry(footprint_pix)
        wcs, grid_pix, shape = geom_grid["wcs"], geom_grid["grid_pix"], geom_grid["shape"]

        a_k = _align(footprint_pix, prior["pix"], prior["a_k"], np.nan)
        coverage = _align(footprint_pix, prior["pix"], prior["coverage"], np.nan)
        # The view's own per-class quantity (A_C or N_CAT_C) and its
        # class-summed total, from which BOTH the density panel and the
        # class panels' P(C | ...) = class / total are drawn (sec. 8,
        # `atlas.captions.ATLAS_INTRINSIC_*`/`ATLAS_SELECTION_*`).
        class_values = prior["above"] if view == "intrinsic" else prior["n_cat"]
        total = class_values.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total[:, None] > 0, class_values / total[:, None], np.nan)
        total_aligned = _align(footprint_pix, prior["pix"], total, np.nan)
        share_aligned = _align(footprint_pix, prior["pix"], share, np.nan)

        col_grid = _reproject(footprint_pix, a_k, grid_pix, shape)
        density_grid = _reproject(footprint_pix, total_aligned, grid_pix, shape)
        coverage_grid = (_reproject(footprint_pix, coverage, grid_pix, shape)
                          if spec["coverage_outline"] else None)
        share_grids = [_reproject(footprint_pix, share_aligned[:, i], grid_pix, shape)
                       for i in range(len(CLASSES))]

        # Fixed layout: four columns of two
        # stacked panels, A/B | C/D | E/F | G/H -- A = column, B = the
        # view's density, C = GAL, D = STAR, E = PAHC, F = AGB, G = YSO,
        # H = H2S. Row-major fill into a 4-column grid puts the top row
        # at [A, C, E, G] and the bottom row at [B, D, F, H], which is
        # exactly this column pairing. Each class panel draws on its OWN
        # linear scale, spanning its own pixel's range (`_panel_norm`),
        # not a scale shared across the six, so a rare class's spatial
        # structure is visible rather than flattened to one colour.
        def _class_panel(cls):
            idx = CLASSES.index(cls)
            # The share as computed, on that panel's own linear scale:
            # nothing is clipped, and the colour bar's automatic ticks
            # read off whatever range this class actually spans.
            # No per-panel colour-bar label: the probability the six
            # class panels draw is stated once, in the group heading
            # above them (`spec["group_heading"]`), not repeated six
            # times beside each bar.
            return dict(data=share_grids[idx], cmap="viridis", norm=_panel_norm(share_grids[idx]), title=cls,
                        cbar_label=None, hatch=None, cbar_ticks=None)

        col_panel = dict(data=col_grid, cmap="magma", norm=_log_norm(col_grid),
                          title=plot_style.label(r"Column $\mathbf{A_K}$", "mag"), cbar_label=None, hatch=None,
                          cbar_ticks=None)
        density_title = plot_style.label(spec["density_label"], "deg$^{-2}$").replace(" [", "\n[")
        hatch = None
        if coverage_grid is not None and np.any(coverage_grid < 0.5):
            hatch = coverage_grid
        density_panel = dict(data=density_grid, cmap="viridis", norm=_log_norm(density_grid),
                              title=density_title, cbar_label=None, hatch=hatch, cbar_ticks=None)
        panels = [col_panel, _class_panel("GAL"), _class_panel("PAHC"), _class_panel("YSO"),
                  density_panel, _class_panel("STAR"), _class_panel("AGB"), _class_panel("H2S")]
        n_panels = len(panels)

        # The selection page's own caption lines, in addition to the
        # shared class/total statements: the total-count check, the
        # surveyed area and the per-class total-count ratio -- each its
        # own line here rather than in the suptitle, which has no room
        # for them on a narrow region. `bright3`/`bright10` stay in the
        # product and the stage's own log, off this page.
        extra_lines = []
        if view == "selection":
            total_predicted = float(prior["attrs"].get("TOTAL_PREDICTED", np.nan))
            total_observed = float(prior["attrs"].get("TOTAL_OBSERVED", np.nan))
            surveyed_area = float(prior["attrs"].get("SURVEYED_AREA_DEG2", np.nan))
            ratio_po = total_predicted / total_observed if total_observed else float("nan")
            ratios = [(cls, float(prior["attrs"].get("RATIO_%s" % cls, np.nan))) for cls in CLASSES]
            area_label = plot_style.label("surveyed area", "deg$^{2}$")
            # The total-count check (sec. 8, sec. 9): the prior's own
            # normalization against the survey's count, the statement
            # `captions.TOTAL_COUNT_CHECK`'s, never restated here.
            extra_lines.append(captions.TOTAL_COUNT_CHECK.format(value=ratio_po))
            extra_lines.append("%s = %.4g" % (area_label, surveyed_area))
            extra_lines.append("Per-class total-count ratio: " + ", ".join(
                "%s %.3g" % (cls, ratio) for cls, ratio in ratios))

        # The panel grid is sized to the fixed 4x2 layout's content
        # first; the caption block (below the grid) then grows the
        # page's OWN height by exactly what it needs, so the grid itself
        # never shrinks to make room.
        aspect = geom_grid["n_x"] / float(geom_grid["n_y"])
        page_w, page_h, geom = _atlas_page_size(
            aspect, col_gaps=[SEPARATOR_GAP_IN, GAP_X_IN, GAP_X_IN])
        extra_block = "\n".join(extra_lines) if extra_lines else None
        caption_text, caption_h = _caption_layout(_caption_block(view, extra_line=extra_block), page_w)
        page_h_total = page_h + caption_h
        cols = geom["cols"]
        last_row_of_col = _outer_rows(n_panels, cols)

        plot_style.apply_style()
        fig = plot_style.new_sized_figure(page_w, page_h_total)

        for i, p in enumerate(panels):
            row, col = divmod(i, cols)
            x, y, w, h = _panel_rect(geom, i, page_w, page_h)
            rect = (x, y + caption_h, w, h)  # shifted up to clear the caption strip
            ax, im = _add_panel(fig, rect, page_w, page_h_total, wcs, p["data"], p["cmap"],
                                 norm=p["norm"], title=p["title"],
                                 show_dec=col == 0, show_ra=row == last_row_of_col[col])
            if p["hatch"] is not None:
                # The surveyed-coverage floor, drawn on the selection
                # page's density panel only, as one thin contour line
                # outlining the well-covered footprint -- `coverage_grid`
                # is already NaN outside the admitted footprint (sec. 8's
                # own reprojection mask), so the line never crosses into
                # the white area outside it.
                ax.contour(p["hatch"], levels=[0.5], colors="white", linewidths=0.8)
            _panel_colorbar(fig, ax, im, label=p["cbar_label"],
                             log=isinstance(p["norm"], LogNorm), ticks=p["cbar_ticks"])

        fig.text(MARGIN_LEFT_IN / page_w, (caption_h - CAPTION_TOP_PAD_IN) / page_h_total,
                  caption_text, fontsize=CAPTION_FONT_SIZE, va="top", ha="left")

        # The separator rule, between the left column (col_panel/
        # density_panel, index 0) and the three class columns (indices
        # 1-3): one thin vertical line centred in `SEPARATOR_GAP_IN`,
        # from the bottom of the lower panel to the top of the upper
        # panel's title strip -- both rows share that span regardless of
        # column, since every panel in the grid is the same height.
        left_rect = _panel_rect(geom, 0, page_w, page_h)
        col1_rect = _panel_rect(geom, 1, page_w, page_h)
        col3_rect = _panel_rect(geom, 3, page_w, page_h)
        bottom_rect = _panel_rect(geom, 4, page_w, page_h)
        content_h = geom["rows"] * (geom["panel_h"] + geom["title_h"]) + (geom["rows"] - 1) * GAP_Y_IN
        sep_x_in = left_rect[0] + geom["panel_w"] + geom["bar_w"] + SEPARATOR_GAP_IN / 2.0
        sep_y0_in = bottom_rect[1] + caption_h
        sep_y1_in = bottom_rect[1] + content_h + caption_h
        fig.add_artist(Line2D([sep_x_in / page_w, sep_x_in / page_w],
                               [sep_y0_in / page_h_total, sep_y1_in / page_h_total],
                               transform=fig.transFigure, color="0.6", linewidth=0.8))

        # The page title at the top of the page and the group heading
        # centred over the three class columns, just above the panel
        # grid's own top edge -- both bold, independent of the region's
        # own aspect.
        heading_x_in = 0.5 * (col1_rect[0] + col3_rect[0] + geom["panel_w"] + geom["bar_w"])
        fig.text(0.5, (page_h_total - TITLE_TOP_PAD_IN) / page_h_total,
                  spec["page_title"] % region, fontsize=TITLE_FONTSIZE, fontweight="bold",
                  va="top", ha="center")
        fig.text(heading_x_in / page_w, (sep_y1_in + HEADING_ABOVE_GRID_IN) / page_h_total,
                  spec["group_heading"], fontsize=LABEL_FONTSIZE, fontweight="bold",
                  va="bottom", ha="center")

        out_dir = os.path.join(config.data_root, "bmstp", "atlas", "figures")
        os.makedirs(out_dir, exist_ok=True)
        # The shared vocabulary, off both pages' own caption and onto
        # its own file -- one write per page built, an idempotent
        # overwrite of the same file.
        captions.write_vocabulary(out_dir)
        paths = []
        for fmt in formats:
            path = os.path.join(out_dir, "prior-atlas-%s_%s.%s" % (view, region, fmt))
            # No `bbox_inches="tight"`: that re-crops to content and
            # drifts the saved size away from the requested page size.
            fig.savefig(path, dpi=150)
            paths.append(path)
        plt.close(fig)

        st.done(paths[0], n_x=geom_grid["n_x"], n_y=geom_grid["n_y"], n_admitted=footprint_pix.size,
                cols=geom["cols"], rows=geom["rows"], panel_scale_in=geom["panel_h"])
    return paths


def build_prior_intrinsic_region(config, region, formats):
    """The prior atlas's intrinsic page (`_build_prior_page`, view
    "intrinsic"): the sky density and class shares before the survey's
    selection, `A_C` from the product's own `N_CAT_C`."""
    return _build_prior_page(config, region, formats, "intrinsic")


def build_prior_selection_region(config, region, formats):
    """The prior atlas's selection page (`_build_prior_page`, view
    "selection"): the same layout after the survey's selection, `N_C`
    from the product's own `N_CAT_C`, plus the coverage outline and the
    predicted/observed and bright-source ratios the selection alone
    determines."""
    return _build_prior_page(config, region, formats, "selection")


def build_posterior_region(config, region, formats):
    """Writes the posterior-atlas figure (P11, 8 panels: column, the six
    posterior means, the posterior YSO count) under `fittp/atlas/figures/`,
    on the SAME footprint as the prior figure's own (`catalog.depth_grid`)
    and the SAME fixed 4x2 layout the prior figure uses, so the two
    pages read as one pair. The column panel reuses P6's own `A_COL_K`
    (P11 carries no column), so this figure needs the prior atlas too."""
    post_path = config_module.product_path(config, "fittp", "atlas", "posterior", "hpx512", region=region)
    if not os.path.exists(post_path):
        print("atlas.render [%s]: no posterior atlas -- run RUNBOOKtp.sh's "
              "'PY sesnaimpute.fittp.atlas' line first, skipped" % region, flush=True)
        return None
    prior_path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    if not os.path.exists(prior_path):
        print("atlas.render [%s]: posterior atlas present but no prior atlas for its "
              "column panel -- run RUNBOOKtp.sh's 'PY sesnaimpute.bmstp.atlas' line first, skipped"
              % region, flush=True)
        return None
    footprint_pix = _require_depth_grid(config, region)
    if footprint_pix is None:
        return None
    posterior = _read_posterior(post_path)
    prior = _read_prior(prior_path, region, require_above=False)

    with progress.Stage("atlas.render.posterior", region) as st:
        geom_grid = _footprint_geometry(footprint_pix)
        wcs, grid_pix, shape = geom_grid["wcs"], geom_grid["grid_pix"], geom_grid["shape"]

        a_k = _align(footprint_pix, prior["pix"], prior["a_k"], np.nan)
        n_sources = _align(footprint_pix, posterior["pix"], posterior["n_sources"], 0)
        gap_1d = n_sources == 0
        mean_p = _align(footprint_pix, posterior["pix"], posterior["mean_p"], np.nan)
        n_yso_half = _align(footprint_pix, posterior["pix"],
                             posterior["n_yso_half"].astype(np.float64), np.nan)

        col_grid = _reproject(footprint_pix, a_k, grid_pix, shape)
        mean_grids = [_reproject(footprint_pix, mean_p[:, i], grid_pix, shape)
                      for i in range(len(CLASSES))]
        nyso_grid = _reproject(footprint_pix, n_yso_half, grid_pix, shape)
        gap_grid = _gap_grid(footprint_pix, gap_1d, grid_pix, shape)

        # Fixed layout, the same one the
        # prior figure uses: four columns of two stacked panels, A/B |
        # C/D | E/F | G/H -- A = column, B = the posterior YSO count in
        # the prior figure's "density" slot, C = GAL, D = STAR, E =
        # PAHC, F = AGB, G = YSO, H = H2S. Every posterior-derived panel
        # (not the column, which P6 defines for every admitted pixel
        # regardless of source count) is painted grey at a source-less
        # pixel.
        def _class_panel(cls):
            idx = CLASSES.index(cls)
            return dict(data=mean_grids[idx], cmap="viridis", norm=None, title=cls,
                        cbar_label=plot_style.label("Posterior Fractional Share", None),
                        grey=gap_grid)

        col_panel = dict(data=col_grid, cmap="magma", norm=_log_norm(col_grid),
                          title=plot_style.label(r"Column $\mathbf{A_K}$", "mag"), cbar_label=None, grey=None)
        # N_YSO_ABOVE_HALF (P11) is a raw per-pixel COUNT of P(YSO)>0.5
        # sources, never divided by the pixel's own solid angle -- unlike
        # N_CAT_C (deg^-2 already), so this title carries no unit rather
        # than a false "deg^-2" (the
        # nearest true description of what the panel draws).
        nyso_panel = dict(data=nyso_grid, cmap="magma", norm=None,
                           title=plot_style.label("Posterior YSO Count", None),
                           cbar_label=None, grey=gap_grid)
        panels = [col_panel, _class_panel("GAL"), _class_panel("PAHC"), _class_panel("YSO"),
                  nyso_panel, _class_panel("STAR"), _class_panel("AGB"), _class_panel("H2S")]
        n_panels = len(panels)

        # The page ITSELF is sized to the fixed 4x2 layout's content,
        aspect = geom_grid["n_x"] / float(geom_grid["n_y"])
        page_w, page_h, geom = _atlas_page_size(aspect)
        cols = geom["cols"]
        last_row_of_col = _outer_rows(n_panels, cols)

        plot_style.apply_style()
        fig = plot_style.new_sized_figure(page_w, page_h)

        for i, p in enumerate(panels):
            row, col = divmod(i, cols)
            rect = _panel_rect(geom, i, page_w, page_h)
            ax, im = _add_panel(fig, rect, page_w, page_h, wcs, p["data"], p["cmap"],
                                 norm=p["norm"], title=p["title"],
                                 show_dec=col == 0, show_ra=row == last_row_of_col[col],
                                 grey_grid=p["grey"])
            _panel_colorbar(fig, ax, im, label=p["cbar_label"], log=isinstance(p["norm"], LogNorm))

        n_yso_total = int(posterior["n_yso_half"].sum())
        two_band_frac = _two_band_fraction(config, region)
        title = ("%s -- posterior atlas: N($P$(YSO)$>$0.5) = %d, two-band fraction = %.3f"
                  % (region, n_yso_total, two_band_frac))
        fig.suptitle(title + "\ngray: no sources in pixel", fontsize=LABEL_FONTSIZE, y=1.0 - 0.10 / page_h)

        out_dir = os.path.join(config.data_root, "fittp", "atlas", "figures")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for fmt in formats:
            path = os.path.join(out_dir, "posterior-atlas_%s.%s" % (region, fmt))
            fig.savefig(path, dpi=150)
            paths.append(path)
        plt.close(fig)

        st.done(paths[0], n_x=geom_grid["n_x"], n_y=geom_grid["n_y"], n_admitted=footprint_pix.size,
                n_gap=int(gap_1d.sum()), cols=geom["cols"], rows=geom["rows"],
                panel_scale_in=geom["panel_h"])
    return paths


def build(config, regions=None, formats=("png", "pdf")):
    """Per region, the prior atlas's two pages (`bmstp/atlas/figures/`,
    when P6 exists: intrinsic then selection) and the posterior-atlas
    figure (`fittp/atlas/figures/`, when P11 exists), on the same
    depth-grid footprint -- rendering, not a build (no product is stored
    at display resolution), so a region missing an input is skipped
    rather than failed."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_prior_intrinsic_region(config, region, formats)
        build_prior_selection_region(config, region, formats)
        build_posterior_region(config, region, formats)


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
