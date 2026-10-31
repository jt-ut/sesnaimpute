"""One co-added PACS mosaic per band per region, from the region's downloaded maps.

    build_mosaic(config, region, band, box=None) -> Mosaic     writes the product, returns it opened
    open_mosaic(config, region, band, box=None)  -> Mosaic     opens a built product (layers memory-mapped)

The product lives in <data_root>/sky/derived/pacs/mosaic/<Region>/<band>/ (a sky box adds "__slice_<ra>_<dec>_<w>x<h>" to the band
folder): one float32 .npy per layer (image, error, coverage, weight), `_smooth/` (each input map's smoothed-error lattice, a part
of the product: U2 recomputes a map's weight at any position from it) and `mosaic.hdf5`, which holds each layer's UNITS
and READING, the header of the mosaic's celestial WCS and META (per input map: path, PSF group, lattice file and step,
zero level; the PSF groups; the pairwise zero-level differences and their residuals).

Method (briefs/pacs_mosaic_2026-10-29/BRIEF.md, M1-M4 with amendment A1):
  M1  input set = the band's maps minus any map whose observation set is a subset of another map's.
  M2  TAN grid at the footprint centre, north up, pixel = the band's finest input pixel, spanning the union of the
      input footprints (or a given sky box).
  M3a each input map's arbitrary zero level is replaced by one constant o_i per map (`solve_zero_levels`):
      d_ij = median(R_i - R_j) over the output pixels both maps cover (pairs with >= MIN_COMMON_PIXELS), then weighted
      least squares on o_i - o_j = d_ij (weight = pixels in common), each connected set of maps with weighted-mean
      zero; a map with no qualifying overlap gets 0.  o_i is a constant surface brightness: the differences are divided by,
      and o_i multiplied by, the output pixel's own area factor (`tan_area_factor`), so o_i is in Jy per reference-pixel
      area.  Every common pixel enters the median (no subsample).
  M3  each input (minus o_i) is resampled onto the grid by bilinear interpolation of its surface brightness (Jy per pixel
      divided by the input pixel area, times the output pixel's own area, `tan_area_factor`), which conserves flux.  The weight of map i is
      1 / (smoothed error_i)^2, the error layer median-filtered over WINDOW_ARCSEC; pixels with zero or non-finite
      error, zero coverage or a non-finite image have weight 0 and are ignored by the filter.
        image = sum(w_i I_i) / sum(w_i);   error = sqrt(sum(w_i^2 err_i^2)) / sum(w_i)
  M4  the PSF groups (maps with identical published-template set and scan angles,
      `pacs_psf.published_components_for_map`) are listed in META; no weight-fraction layers are stored.

The mosaic is built strip by strip of PRODUCT rows into memory-mapped files, joblib over strips at the config's
n_jobs; each worker reads only the input windows that overlap its strip, so its memory does not depend on the mosaic
size or the worker count.
"""

import json
import os
import re
import warnings

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap

from sesnaimpute.sky.derived import pacs_psf
from sesnaimpute.sky.download.herschel_pacs.index import map_rank, maps_root, read_index
from sesnaimpute.definitions import PACS_PIXEL_ARCSEC as PIXEL_ARCSEC

warnings.filterwarnings("ignore", message=".*datfix.*")

#: Side (arcsec) of the square window of the median filter that smooths each map's error layer.
WINDOW_ARCSEC = 60.0
#: Rows of the mosaic one task builds.
STRIP_ROWS = 64
#: Least common valid output pixels for a pair of maps to enter the zero-level solve.
MIN_COMMON_PIXELS = 2000
#: The band's finest input pixel (arcsec): definitions.PACS_PIXEL_ARCSEC, imported above as PIXEL_ARCSEC.

LAYERS = {
    "image": ("Jy per output pixel", "The mosaic's flux in each pixel, the weighted mean of the input maps resampled onto this grid. "
              "Not a number where no input map has a valid pixel."),
    "error": ("Jy per output pixel", "The 1-sigma noise of the mosaic pixel, propagated from the input maps' own per-pixel noise "
              "layers and the weights. It treats neighbouring input pixels as fully correlated through the resampling "
              "(see the mosaic's READING)."),
    "coverage": ("input coverage units", "The sum over the contributing maps of their resampled coverage layer (the number of "
                 "exposure samples that fell in the pixel); 0 where no input map is valid."),
    "weight": ("(Jy per output pixel)^-2", "The total weight, the sum over the contributing maps of 1 / (smoothed noise)^2; 0 where "
               "no input map is valid."),
}

_READING = (
    "PACS {band} mosaic of {region}. Every input map of the band that is not contained in another map's observations "
    "is resampled onto this one grid (tangent projection, north up, {pix} arcsec pixels). Resampling interpolates the "
    "surface brightness (flux per unit area) bilinearly between the four nearest input pixel centres and multiplies by "
    "the output pixel's own area (a tangent-projection pixel shrinks away from the grid centre), so the flux of a "
    "source is conserved everywhere on the grid. The maps are averaged with weights "
    "1/(smoothed noise)^2, where the smoothed noise of a map is its own per-pixel noise layer median-filtered over a "
    "square window of {win} arcsec; a weight therefore follows how deep the map is, never the signal in the pixel. "
    "Pixels where a map has zero or undefined noise, zero coverage or an undefined image get weight 0 in that map. "
    "The image is sum(w*I)/sum(w); the error is sqrt(sum(w^2*e^2))/sum(w) from the maps' real per-pixel noise, with "
    "the four interpolation neighbours of a pixel taken as fully correlated, so it is not reduced by the interpolation. "
    "Each input map first has its own arbitrary zero level (PACS maps have no defined absolute zero) replaced by one "
    "constant surface brightness, solved from all overlaps between maps (least squares on the median differences of "
    "overlapping maps, each taken over every pixel the two maps share); in each output pixel that constant is multiplied by the "
    "pixel's own area, like the flux, so it changes no point-source flux measured against the local background. Units: Jy per output pixel. "
    "Pixels with no valid input are not-a-number.")


# ----------------------------------------------------------------------------
# M1: the input set

def _label(rec):
    return f"{rec.map_id} [{os.path.basename(rec.path)}] obs {','.join(rec.obsids)}"


def input_set(records, band):
    """(kept, dropped): the band's maps (each file once), minus any map whose observation set is a subset of
    another kept map's. Of maps with identical observation sets the best-ranked (then first by path) is kept.
    `dropped` is a list of (dropped record, the record that contains it)."""
    recs = sorted({r.path: r for r in records if r.band == band}.values(), key=lambda r: r.path)
    order = sorted(recs, key=lambda r: (-len(set(r.obsids)), map_rank(r), r.path))
    kept, dropped = [], []
    for r in order:
        host = next((k for k in kept if set(r.obsids) <= set(k.obsids)), None)
        if host is None:
            kept.append(r)
        else:
            dropped.append((r, host))
    kept.sort(key=lambda r: r.path)
    return kept, dropped


def print_input_set(region, band, kept, dropped):
    print(f"pacs_mosaic {region} {band}: {len(kept)} maps kept, {len(dropped)} dropped (observation set contained in another map's)")
    for r in kept:
        print(f"  kept    {_label(r)}")
    for r, host in dropped:
        print(f"  dropped {_label(r)}  (inside {host.map_id} [{os.path.basename(host.path)}])")


def group_key(config, rec):
    """The PSF component group of a map: its published templates and their position angles."""
    comps = pacs_psf.published_components_for_map(config, rec)
    return tuple((c.path, round(float(c.position_angle_deg), 6), round(float(c.image.sum()), 9),
                  round(float((c.image ** 2).sum()), 9)) for c in comps)


# ----------------------------------------------------------------------------
# M2: the grid

def _celestial(rec):
    return WCS(fits.Header.fromstring(rec.wcs_header)).celestial


def _tan(ra, dec, pix_arcsec, nx=1, ny=1, crpix=(1.0, 1.0)):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-pix_arcsec / 3600.0, pix_arcsec / 3600.0]
    w.wcs.crpix = list(crpix)
    w.wcs.cunit = ["deg", "deg"]
    return w


def _corners(rec):
    return np.array(rec.polygon, dtype=float)


def make_grid(records, band, box=None):
    """The mosaic's celestial WCS and shape (ny, nx). `box` = (ra, dec, width_deg, height_deg): the sky box about
    (ra, dec) (width along RA on the sky) instead of the union of the footprints."""
    pix = PIXEL_ARCSEC[band]
    if box is not None:
        ra0, dec0, wd, hd = box
        nx, ny = int(np.ceil(wd * 3600.0 / pix)), int(np.ceil(hd * 3600.0 / pix))
        return _tan(ra0, dec0, pix, crpix=((nx + 1) / 2.0, (ny + 1) / 2.0)), (ny, nx)
    pts = np.concatenate([_corners(r) for r in records])
    ra0 = float(np.degrees(np.angle(np.exp(1j * np.radians(pts[:, 0])).mean())) % 360.0)
    dec0 = float(pts[:, 1].mean())
    for _ in range(3):                                   # re-centre the tangent point on the footprint
        w = _tan(ra0, dec0, pix)
        x, y = w.all_world2pix(pts[:, 0], pts[:, 1], 0)
        cx, cy = (x.min() + x.max()) / 2.0, (y.min() + y.max()) / 2.0
        ra0, dec0 = (float(v) for v in w.all_pix2world(cx, cy, 0))
    w = _tan(ra0, dec0, pix)
    x, y = w.all_world2pix(pts[:, 0], pts[:, 1], 0)
    nx, ny = int(np.ceil(x.max() - x.min())) + 1, int(np.ceil(y.max() - y.min())) + 1
    return _tan(ra0, dec0, pix, crpix=(0.5 - x.min(), 0.5 - y.min())), (ny, nx)


# ----------------------------------------------------------------------------
# the smoothed error of one map (lattice of window medians, stage A)

def _window_px(rec):
    return max(3, int(round(WINDOW_ARCSEC / rec.pixel_scale_arcsec)) | 1)      # odd


def _lattice_path(outdir, rec):
    return os.path.join(outdir, "_smooth", os.path.splitext(rec.path)[0].replace("/", "__") + ".npy")


def _read_layers(root, rec, rows=None, cols=None):
    """The map's image, error, coverage (float64 copies of the window rows x cols, as slices)."""
    rows = rows or slice(0, rec.shape[0])
    cols = cols or slice(0, rec.shape[1])
    with fits.open(os.path.join(root, rec.path), mode="denywrite", memmap=True) as hdus:
        return tuple(np.array(hdus[i].data[rows, cols], dtype=np.float64)
                     for i in (rec.image_hdu, rec.error_hdu, rec.coverage_hdu))


def _good(img, err, cov):
    return np.isfinite(img) & np.isfinite(err) & (err > 0) & (cov > 0)


def valid_at(layers, px, py):
    """Whether the mosaic uses the input map at input pixel coordinates (px, py) (inside the map): the four
    interpolation neighbours `resample_block` weights are good (`_good`; a neighbour of weight zero is not asked).
    `layers`: the map's (image, error, coverage) arrays (memory-mapped; only the neighbours are read)."""
    img, err, cov = layers
    ny, nx = img.shape
    xi = np.clip(np.floor(px).astype(np.int64), 0, max(nx - 2, 0))
    yi = np.clip(np.floor(py).astype(np.int64), 0, max(ny - 2, 0))
    fx, fy = np.clip(px - xi, 0, 1), np.clip(py - yi, 0, 1)
    valid = np.ones(px.shape, bool)
    for dy, by in ((0, 1 - fy), (1, fy)):
        for dx, bx in ((0, 1 - fx), (1, fx)):
            jj, ii = np.minimum(yi + dy, ny - 1), np.minimum(xi + dx, nx - 1)
            valid &= _good(img[jj, ii], err[jj, ii], cov[jj, ii]) | (bx * by == 0)
    return valid


def _smooth_lattice(root, rec, outdir):
    """The median of the map's valid error pixels in a square window of WINDOW_ARCSEC, evaluated on a lattice with
    a spacing of half a window (node (j, i) at pixel (i * step, j * step)); NaN where the window has no valid pixel.
    Done band by band, so the memory is one window of rows."""
    win = _window_px(rec)
    half, step = win // 2, max(1, win // 2)
    ny, nx = rec.shape
    nj, ni = (ny - 1) // step + 2, (nx - 1) // step + 2
    lat = np.full((nj, ni), np.nan)
    with fits.open(os.path.join(root, rec.path), mode="denywrite", memmap=True) as hdus:
        eh, ch, ih = hdus[rec.error_hdu].data, hdus[rec.coverage_hdu].data, hdus[rec.image_hdu].data
        for j in range(nj):
            r0, r1 = max(0, j * step - half), min(ny, j * step + half + 1)
            if r1 <= r0:
                continue
            err = np.array(eh[r0:r1], dtype=np.float64)
            err[~_good(np.array(ih[r0:r1], dtype=np.float64), err, np.array(ch[r0:r1], dtype=np.float64))] = np.nan
            padded = np.pad(err, ((0, 0), (half, max(0, (ni - 1) * step + win - nx - half))), constant_values=np.nan)
            view = np.lib.stride_tricks.sliding_window_view(padded, (err.shape[0], win), axis=(0, 1))
            cut = view[0, ::step][:ni].reshape(ni, -1)
            ok = np.isfinite(cut).any(axis=1)
            row = np.full(ni, np.nan)
            row[ok] = np.nanmedian(cut[ok], axis=1)
            lat[j] = row
    path = _lattice_path(outdir, rec)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, lat)
    return path


def _lattice_at(lat, step, px, py):
    """The lattice at input pixel coordinates by bilinear interpolation over the valid nodes (NaN where none)."""
    nj, ni = lat.shape
    u, v = px / step, py / step
    i0 = np.clip(np.floor(u).astype(np.int64), 0, ni - 2) if ni > 1 else np.zeros(u.shape, np.int64)
    j0 = np.clip(np.floor(v).astype(np.int64), 0, nj - 2) if nj > 1 else np.zeros(v.shape, np.int64)
    fu, fv = np.clip(u - i0, 0, 1), np.clip(v - j0, 0, 1)
    num = np.zeros(px.shape)
    den = np.zeros(px.shape)
    for dj, bj in ((0, 1 - fv), (1, fv)):
        for di, bi in ((0, 1 - fu), (1, fu)):
            jj, ii = np.minimum(j0 + dj, nj - 1), np.minimum(i0 + di, ni - 1)
            node = lat[jj, ii]
            b = np.where(np.isfinite(node), bj * bi, 0.0)
            num += b * np.where(np.isfinite(node), node, 0.0)
            den += b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ----------------------------------------------------------------------------
# M3: resampling one map onto a block of the output grid

def pixel_area_arcsec2(wcs):
    """The pixel area at the reference pixel (arcsec^2)."""
    return float(abs(np.linalg.det(wcs.pixel_scale_matrix)) * 3600.0 ** 2)


def tan_area_factor(wcs, x, y):
    """The area of the tangent-projection pixel at zero-origin (x, y) over the reference pixel's: cos^3 of its
    angular distance from the tangent point, (1 + rho^2)^(-3/2) with rho the intermediate radius in radians (exact
    for TAN; 0.6 % at 3.7 deg)."""
    xi, eta = _intermediate(wcs, x, y)
    return (1.0 + xi ** 2 + eta ** 2) ** -1.5


def _intermediate(wcs, x, y):
    """The intermediate world coordinates (radians) of zero-origin pixels (x, y) of a TAN `wcs`."""
    if not all(c.endswith("-TAN") for c in wcs.wcs.ctype):
        raise ValueError("the mosaic grid is not a tangent projection")
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    m = wcs.pixel_scale_matrix
    dx, dy = x + 1.0 - wcs.wcs.crpix[0], y + 1.0 - wcs.wcs.crpix[1]
    return np.radians(m[0, 0] * dx + m[0, 1] * dy), np.radians(m[1, 0] * dx + m[1, 1] * dy)


def resample_block(root, rec, in_wcs, out_wcs, out_area, x0, x1, y0, y1, lattice=None, layers=None):
    """Map `rec` resampled onto output pixels x0:x1, y0:y1 (zero-origin, half-open). Returns None when no output
    pixel falls inside the map, else a dict of arrays over the block: `valid` (bool: the four interpolation
    neighbours are valid), `image`, `error` (Jy per output pixel), `coverage`, `factor` (the pixel's own area over the reference
    pixel's), and with `lattice` = (lat, step)
    `smooth` (smoothed error, Jy per output pixel; NaN falls back to nothing: such pixels are invalid). `layers`: the
    map's (image, error, coverage) arrays in memory, in place of the file (checks on a synthetic map)."""
    ny, nx = rec.shape
    X, Y = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(y0, y1, dtype=float))
    ra, dec = out_wcs.all_pix2world(X.ravel(), Y.ravel(), 0)
    px, py = (a.reshape(X.shape) for a in in_wcs.all_world2pix(ra, dec, 0))
    inside = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (px <= nx - 1) & (py >= 0) & (py <= ny - 1)
    if not inside.any():
        return None
    ix0, ix1 = int(np.floor(px[inside].min())), int(np.floor(px[inside].max())) + 2
    iy0, iy1 = int(np.floor(py[inside].min())), int(np.floor(py[inside].max())) + 2
    ix1, iy1 = min(ix1, nx), min(iy1, ny)
    img, err, cov = (_read_layers(root, rec, slice(iy0, iy1), slice(ix0, ix1)) if layers is None
                     else tuple(np.array(a[iy0:iy1, ix0:ix1], dtype=np.float64) for a in layers))
    good = _good(img, err, cov)
    wx, wy = np.where(inside, px, 0.0) - ix0, np.where(inside, py, 0.0) - iy0
    xi = np.clip(np.floor(wx).astype(np.int64), 0, max(img.shape[1] - 2, 0))
    yi = np.clip(np.floor(wy).astype(np.int64), 0, max(img.shape[0] - 2, 0))
    fx, fy = np.clip(wx - xi, 0, 1), np.clip(wy - yi, 0, 1)
    scale = out_area * tan_area_factor(out_wcs, X, Y) / pixel_area_arcsec2(in_wcs)   # the output pixel's own area
    acc = {k: np.zeros(X.shape) for k in ("image", "error", "coverage")}
    valid = inside.copy()
    for dy, by in ((0, 1 - fy), (1, fy)):
        for dx, bx in ((0, 1 - fx), (1, fx)):
            jj, ii = np.minimum(yi + dy, img.shape[0] - 1), np.minimum(xi + dx, img.shape[1] - 1)
            b = bx * by
            g = good[jj, ii]
            valid &= g | (b == 0)
            for k, a in (("image", img), ("error", err), ("coverage", cov)):
                acc[k] += b * np.where(g, a[jj, ii], 0.0)
    acc["image"] *= scale
    acc["error"] *= scale
    acc["factor"] = tan_area_factor(out_wcs, X, Y)         # the output pixel's own area over the reference pixel's
    acc["valid"] = valid
    if lattice is not None:
        lat, step = lattice
        s = _lattice_at(lat, step, np.where(inside, px, 0.0), np.where(inside, py, 0.0)) * scale
        acc["smooth"] = s
        acc["valid"] = valid & np.isfinite(s) & (s > 0)
    return acc


def _bbox_in_output(rec, in_wcs, out_wcs, shape):
    """Output pixel bounding box (x0, x1, y0, y1) of the map's footprint, clipped to the grid; None if outside."""
    ny, nx = rec.shape
    cx = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
    cy = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    ra, dec = in_wcs.all_pix2world(cx, cy, 0)
    x, y = out_wcs.all_world2pix(ra, dec, 0)
    x0, x1 = max(int(np.floor(x.min())) - 1, 0), min(int(np.ceil(x.max())) + 2, shape[1])
    y0, y1 = max(int(np.floor(y.min())) - 1, 0), min(int(np.ceil(y.max())) + 2, shape[0])
    return None if (x1 <= x0 or y1 <= y0) else (x0, x1, y0, y1)


def _blocks(root, outdir, recs, out_wcs, out_area, shape, y0, y1):
    """[(rec, (bx0, bx1, by0, by1), resampled dict)] for the maps overlapping rows y0:y1 (valid as in resample_block)."""
    out = []
    for rec in recs:
        in_wcs = _celestial(rec)
        box = _bbox_in_output(rec, in_wcs, out_wcs, shape)
        if box is None or box[3] <= y0 or box[2] >= y1:
            continue
        bx0, bx1, by0, by1 = box
        by0, by1 = max(by0, y0), min(by1, y1)
        lat = np.load(_lattice_path(outdir, rec))
        step = max(1, _window_px(rec) // 2)
        r = resample_block(root, rec, in_wcs, out_wcs, out_area, bx0, bx1, by0, by1, (lat, step))
        if r is not None:
            out.append((rec, (bx0, bx1, by0, by1), r))
    return out


def _pair_strip(task):
    """First pass over rows y0:y1: the valid pixel count of each map, and for each pair of maps the count of
    common valid pixels and all their differences R_i - R_j in surface brightness (Jy per reference-pixel area), float32."""
    root, outdir, recs, out_header, shape, y0, y1 = task
    out_wcs = WCS(fits.Header.fromstring(out_header))
    blocks = _blocks(root, outdir, recs, out_wcs, pixel_area_arcsec2(out_wcs), shape, y0, y1)
    counts = {rec.path: int(r["valid"].sum()) for rec, _, r in blocks}
    pairs = {}
    for a in range(len(blocks)):
        for b in range(a + 1, len(blocks)):
            (ra_, (ax0, ax1, ay0, ay1), A), (rb_, (bx0, bx1, by0, by1), B) = blocks[a], blocks[b]
            x0, x1, yy0, yy1 = max(ax0, bx0), min(ax1, bx1), max(ay0, by0), min(ay1, by1)
            if x1 <= x0 or yy1 <= yy0:
                continue
            sa = (slice(yy0 - ay0, yy1 - ay0), slice(x0 - ax0, x1 - ax0))
            sb = (slice(yy0 - by0, yy1 - by0), slice(x0 - bx0, x1 - bx0))
            both = A["valid"][sa] & B["valid"][sb]
            n = int(both.sum())
            if n == 0:
                continue
            d = ((A["image"][sa] - B["image"][sb]) / A["factor"][sa])[both].astype(np.float32)
            pairs[(ra_.path, rb_.path)] = (n, d)
    return counts, pairs


def solve_zero_levels(paths, npix, pair_n, pair_d):
    """One constant o_i per map from the pairwise differences d_ij = median(R_i - R_j): weighted least squares on
    o_i - o_j = d_ij with weight n_ij, separately for each connected set of maps, each set with weighted-mean zero
    (weight = the map's valid pixel count). `pair_n`, `pair_d`: {(path_i, path_j): n, d} for the qualifying pairs.
    A map without a qualifying pair gets 0. Returns {path: o}."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    index = {p: k for k, p in enumerate(paths)}
    off = {p: 0.0 for p in paths}
    if not pair_n:
        return off
    ii = np.array([index[a] for a, _ in pair_n])
    jj = np.array([index[b] for _, b in pair_n])
    graph = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(paths),) * 2)
    _, label = connected_components(graph, directed=False)
    for c in np.unique(label[np.unique(np.concatenate([ii, jj]))]):
        members = np.flatnonzero(label == c)
        local = {m: k for k, m in enumerate(members)}
        keys = [k for k in pair_n if label[index[k[0]]] == c]
        A = np.zeros((len(keys) + 1, len(members)))
        y = np.zeros(len(keys) + 1)
        sw = np.zeros(len(keys) + 1)
        for r, k in enumerate(keys):
            A[r, local[index[k[0]]]], A[r, local[index[k[1]]]] = 1.0, -1.0
            y[r], sw[r] = pair_d[k], np.sqrt(pair_n[k])
        A[-1] = [npix[paths[m]] for m in members]
        A[-1] /= A[-1].sum()
        sw[-1] = 1e3 * sw[:-1].max()
        o = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        for m, v in zip(members, o):
            off[paths[m]] = float(v)
    return off


def _strip(task):
    """Builds rows y0:y1 of every layer in place (memory-mapped, r+); each map's zero level is subtracted."""
    root, outdir, recs, offsets, out_header, shape, y0, y1 = task
    out_wcs = WCS(fits.Header.fromstring(out_header))
    out_area = pixel_area_arcsec2(out_wcs)
    ny, nx = shape
    rows = y1 - y0
    sw = np.zeros((rows, nx))
    swi = np.zeros((rows, nx))
    sw2e2 = np.zeros((rows, nx))
    cov = np.zeros((rows, nx))
    for rec, (bx0, bx1, by0, by1), r in _blocks(root, outdir, recs, out_wcs, out_area, shape, y0, y1):
        ok = r["valid"]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(ok, 1.0 / r["smooth"] ** 2, 0.0)
        sl = (slice(by0 - y0, by1 - y0), slice(bx0, bx1))
        sw[sl] += w
        swi[sl] += w * np.where(ok, r["image"] - offsets[rec.path] * r["factor"], 0.0)
        sw2e2[sl] += (w * np.where(ok, r["error"], 0.0)) ** 2
        cov[sl] += np.where(ok, r["coverage"], 0.0)
    has = sw > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        image = np.where(has, swi / sw, np.nan)
        error = np.where(has, np.sqrt(sw2e2) / sw, np.nan)
    for name, a in (("image", image), ("error", error), ("coverage", cov), ("weight", sw)):
        m = np.load(os.path.join(outdir, name + ".npy"), mmap_mode="r+")
        m[y0:y1] = a.astype(np.float32)
        m.flush()
        del m
    return y0


# ----------------------------------------------------------------------------
# the product

def product_dir(config, region, band, box=None):
    safe = re.sub(r"[^A-Za-z0-9+-]+", "_", region)
    return os.path.join(config.data_root, "sky", "derived", "pacs", "mosaic", safe,
                        band + ("" if box is None else "__slice_%.3f_%.3f_%.2fx%.2f" % tuple(box)))


def _assert_inside_scratch_derived(config, path):
    """The output directory must really lie inside this root's own sky/derived (no symlink in between)."""
    derived = os.path.join(os.path.realpath(config.data_root), "sky", "derived")
    assert os.path.realpath(os.path.join(config.data_root, "sky", "derived")) == derived, "sky/derived is a symlink"
    real = os.path.realpath(path)
    assert real == derived or real.startswith(derived + os.sep), f"{path} resolves outside {derived}"


class Mosaic:
    """An opened mosaic: `layers` {name: memory-mapped float32 array}, `wcs`, `shape`, `meta` (see build_mosaic),
    `groups` (list of lists of input paths per PSF group), `dir`."""

    def __init__(self, directory):
        self.dir = directory
        with h5py.File(os.path.join(directory, "mosaic.hdf5"), "r") as f:
            self.wcs = WCS(fits.Header.fromstring(f["WCS_HEADER"][()].decode()))
            self.meta = json.loads(f["META"][()].decode())
        self.shape = tuple(self.meta["shape"])
        self.groups = self.meta["groups"]
        self.layers = {name: np.load(os.path.join(directory, name + ".npy"), mmap_mode="r")
                       for name in ("image", "error", "coverage", "weight")}


def open_mosaic(config, region, band, box=None):
    return Mosaic(product_dir(config, region, band, box))


def build_mosaic(config, region, band, box=None, n_jobs=None, verbose=True):
    """Builds the band's mosaic of the region (see the module docstring) and returns it opened.
    META (json in mosaic.hdf5): `maps` = per input map {path, group, lattice (file relative to the product), lattice_step
    (input pixels), window_px, zero_level_jy_per_pixel (Jy per reference-pixel area), zero_level_mjy_per_beam}; `groups`, `group_keys`; `pairs` =
    [{a, b, d, n, residual}] (Jy per output pixel; residual = d - (o_a - o_b))."""
    root = maps_root(config)
    kept, dropped = input_set(read_index(config, region), band)
    print_input_set(region, band, kept, dropped)
    for r in kept:
        assert r.unit.strip().lower() == "jy/pixel", f"{r.path}: the map's unit is {r.unit!r}, not Jy/pixel"
    wcs, shape = make_grid(kept, band, box)
    in_wcs = {r.path: _celestial(r) for r in kept}
    kept = [r for r in kept if _bbox_in_output(r, in_wcs[r.path], wcs, shape) is not None]
    if not kept:
        raise ValueError(f"no {band} map of {region} overlaps the mosaic grid")
    keys, group_of = [], {}
    for r in kept:
        k = group_key(config, r)
        if k not in keys:
            keys.append(k)
        group_of[r.path] = keys.index(k)
    groups = [[r.path for r in kept if group_of[r.path] == g] for g in range(len(keys))]
    outdir = product_dir(config, region, band, box)
    os.makedirs(outdir, exist_ok=True)
    _assert_inside_scratch_derived(config, outdir)
    n_jobs = n_jobs or config.n_jobs
    pool = Parallel(n_jobs=n_jobs)
    pool(delayed(_smooth_lattice)(root, r, outdir) for r in kept)
    ny, nx = shape
    header = wcs.to_header(relax=True)
    header["NAXIS1"], header["NAXIS2"] = nx, ny
    header_text = header.tostring(sep="")
    # pass 1: valid pixel counts and pairwise differences, strip by strip
    starts = list(range(0, ny, STRIP_ROWS))
    res = pool(delayed(_pair_strip)((root, outdir, kept, header_text, shape, y, min(y + STRIP_ROWS, ny))) for y in starts)
    npix = {r.path: 0 for r in kept}
    pair_n, pair_vals = {}, {}
    for counts, pairs in res:
        for k, v in counts.items():
            npix[k] += v
        for k, (n, d) in pairs.items():
            pair_n[k] = pair_n.get(k, 0) + n
            pair_vals.setdefault(k, []).append(d)
    pair_n = {k: n for k, n in pair_n.items() if n >= MIN_COMMON_PIXELS}
    pair_d = {k: float(np.median(np.concatenate(pair_vals[k]))) for k in pair_n}
    offsets = solve_zero_levels([r.path for r in kept], npix, pair_n, pair_d)
    beam = {r.path: pacs_psf.beam_fwhm_arcsec(r) for r in kept}
    to_mjy = {p: 1e3 * 1.1331 * (beam[p] / PIXEL_ARCSEC[band]) ** 2 for p in beam}       # Jy/pixel -> mJy/beam
    pairs_meta = [dict(a=a, b=b, d=pair_d[(a, b)], n=pair_n[(a, b)], residual=pair_d[(a, b)] - (offsets[a] - offsets[b]))
                  for a, b in pair_n]
    mean_conv = float(np.mean(list(to_mjy.values())))
    resid = np.array([q["residual"] for q in pairs_meta]) * mean_conv
    print(f"pacs_mosaic {region} {band}: zero levels (mJy/beam) "
          + ", ".join(f"{os.path.basename(p)[:9]}:{offsets[p] * to_mjy[p]:.1f}" for p in offsets))
    print(f"  pair residual d_ij - (o_i - o_j): {len(pairs_meta)} pairs, rms {np.sqrt(np.mean(resid ** 2)) if resid.size else 0:.1f} "
          f"max {np.abs(resid).max() if resid.size else 0:.1f} mJy/beam")
    for name in ["image", "error", "coverage", "weight"]:
        _assert_inside_scratch_derived(config, outdir)
        m = open_memmap(os.path.join(outdir, name + ".npy"), mode="w+", dtype=np.float32, shape=shape)
        m.flush()
        del m
    pool(delayed(_strip)((root, outdir, kept, offsets, header_text, shape, y, min(y + STRIP_ROWS, ny))) for y in starts)
    meta = dict(region=region, band=band, shape=[ny, nx], window_arcsec=WINDOW_ARCSEC, pixel_arcsec=PIXEL_ARCSEC[band],
                resampler="bilinear interpolation of surface brightness, times the output pixel's own area",
                maps=[dict(path=r.path, group=group_of[r.path],
                           lattice=os.path.relpath(_lattice_path(outdir, r), outdir), lattice_step=max(1, _window_px(r) // 2),
                           window_px=_window_px(r), zero_level_jy_per_pixel=offsets[r.path],
                           zero_level_mjy_per_beam=offsets[r.path] * to_mjy[r.path]) for r in kept],
                dropped=[r.path for r, _ in dropped], groups=groups, pairs=pairs_meta,
                group_keys=[[list(c) for c in k] for k in keys], box=None if box is None else list(box))
    _assert_inside_scratch_derived(config, outdir)
    with h5py.File(os.path.join(outdir, "mosaic.hdf5"), "w") as f:
        f.create_dataset("WCS_HEADER", data=np.bytes_(header_text))
        f.create_dataset("META", data=np.bytes_(json.dumps(meta)))
        reading = _READING.format(band=band, region=region, pix=PIXEL_ARCSEC[band], win=int(WINDOW_ARCSEC))
        for name, (units, text) in LAYERS.items():
            d = f.create_dataset(name, data=np.bytes_(name + ".npy"))
            d.attrs["UNITS"], d.attrs["READING"] = units, text + " " + reading
    return Mosaic(outdir)
