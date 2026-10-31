"""The one PACS measurement code path, shared by `pacs_completeness` and
`pacs_photometry` (SPEC_PACS_FARIR_DRAFT.md sections 4, 5, 7b and 11.3).

For ONE map tile (one PACS map, one band):

  with open_tile(config, record) as tile:
      tile.xy(ra, dec)                   sky -> map pixel, one vectorised WCS call
      tile.neighbours                    blind bright sources (11.3 step 1)
      tile.fitted_fwhm(x, y)             width of a detection (EXTENDED flag)
      tile.grid                          the 30-arcsec cell grid on this map

The PSF fit at a position (used only to find the bright neighbours, `neighbours`, and the width of a detection) is
a weighted linear least squares of the scan-direction PSF columns on a box around it, with the HOPS/eHOPS annulus
sky subtracted. The catalogue's flux is the eHOPS aperture photometry (`measure`); there is no joint fit of close
pairs: a source within one beam of a detection is an unseparable blend (`pacs_photometry`).

All fluxes are mJy. Pixel coordinates are zero-origin map pixels (x along the
fast axis). A tile holds native-byte-order float32 copies of the image and error
layers (the FITS data are big-endian and numba needs native order); the file
itself is opened read-only and memory-mapped.
"""

import os
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import numba
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from scipy.special import erfc  # import before healpy for native-library initialisation order
import healpy as hp

from sesnaimpute.sky.derived.pacs_psf import (APERTURES, CachedPairedPSF, MosaicPSF, MosaicRecord, aperture_eef, aperture_fraction,
                                              beam_fwhm_arcsec, published_components_for_map)
from sesnaimpute.sky.download.herschel_pacs.index import MapRecord, maps_root, read_index  # noqa: F401
from sesnaimpute.definitions import PACS_CELL_ARCSEC as CELL_ARCSEC

#: detection threshold in sigma (owner 2026-10-24); defines recovery in the completeness
DETECTION_SNR = 5.0
#: rows per strip of the once-per-map preparation (`prepare_map`): every full-map operation runs strip by strip, so
#: its memory is a few strips of the map's width, whatever the map's size
PREPARE_STRIP_ROWS = 256
#: fitting-box half-width in beam FWHM (spec 11.3 step 1)
FIT_BOX_FWHM = 3.0
#: a pixel is also dropped if its error is below this fraction of the map's median error (a mosaic's: of the error its own
#: weight expects)
#: (the layer holds a few collapsed values, 1e-7 of the median, that would carry all the weight)
ERROR_FLOOR_FRACTION = 0.25
#: a pixel is used only where the coverage layer exceeds this fraction of the map's median
#: positive coverage (below it the error layer collapses to meaningless tiny values: measured
#: on Musca, 0.2 % of pixels with coverage < 0.05 of the median carry errors of 1e-7 mJy)
MIN_COVERAGE_FRACTION = 0.05
#: bright-neighbour finding: single-PSF fit S/N against the propagated pixel noise
NEIGHBOUR_SNR = 5.0
#: a blind local maximum must reach this pixel S/N before it is fitted (a 5-sigma
#: fit has a peak pixel near 1.6 sigma at PACS sampling; this only skips the
#: noise floor, it is not a detection rule)
NEIGHBOUR_PEAK_SNR = 1.0
#: a neighbour's PSF is modelled out to this many beam FWHM beyond the fitting box edge
NEIGHBOUR_REACH_FWHM = 2.0
#: at most this many nuisance sources (nearest first) enter one fit
MAX_NUISANCE = 6
#: the published PSF images span +-100 arcsec; a square response table is limited to
#: this half-width (the table corner is sqrt(2) times larger) -- pixels farther from a
#: source than this carry no modelled wing
PSF_TABLE_HALF_ARCSEC = 65.0
#: response-table sampling: sub-pixel steps and spatial nodes over the map
PSF_SUBPIXEL_STEPS = 8
PSF_SPATIAL_NODES = 3
#: a mosaic's PSF nodes per axis: at least PSF_SPATIAL_NODES, more where the mosaic is wider, so that nodes are at most
#: this far apart (Orion A, 8.2 deg, passes the normalisation check at 3 x 3 nodes 4.1 deg apart; Aquila's 12.8 deg at
#: 6.4 deg read real/divisor +0.30 %: the enclosed fraction between nodes is bilinear in a function curved on that scale)
MOSAIC_PSF_NODE_SPACING_DEG = 4.25
#: a fit is trusted only if this fraction of the target beam lies on valid pixels
MIN_BEAM_FRACTION = 0.9

#: a map is registered to SESNA only with at least this many unique bright-peak / SESNA pairs
MIN_REGISTRATION_PAIRS = 10


def to_map_frame(ra_deg, dec_deg, shift):
    """SESNA positions in a map's own astrometric frame: plus the map's registration shift
    (`shift` = (east, north) arcsec, PACS minus SESNA; see `registration`)."""
    east, north = shift
    dec_deg = np.asarray(dec_deg, float)
    return (np.asarray(ra_deg, float) + east / 3600.0 / np.cos(np.radians(dec_deg)),
            dec_deg + north / 3600.0)


def from_map_frame(ra_deg, dec_deg, shift):
    """The inverse of `to_map_frame`: a position in the map's frame back to the SESNA frame."""
    east, north = shift
    dec_deg = np.asarray(dec_deg, float) - north / 3600.0
    return np.asarray(ra_deg, float) - east / 3600.0 / np.cos(np.radians(dec_deg)), dec_deg


def registration_pairs(tile, ra_deg, dec_deg):
    """The pairs `registration` uses: the tile's blind bright peaks (S/N >= 5) paired one-to-one with their nearest
    SESNA position within one beam (each the other's nearest). Returns PACS minus SESNA (n_pairs, 2), east and
    north arcsec."""
    nb = tile.neighbours
    ra_deg, dec_deg = np.asarray(ra_deg, float), np.asarray(dec_deg, float)
    if nb.x.size == 0 or ra_deg.size == 0:
        return np.zeros((0, 2))
    sky = tile.wcs.pixel_to_world(nb.x, nb.y).icrs
    pra, pde = sky.ra.deg, sky.dec.deg
    ra0, de0 = np.median(pra), np.median(pde)
    def plane(r, d):
        return np.c_[((r - ra0 + 180.0) % 360.0 - 180.0) * np.cos(np.radians(de0)) * 3600.0, (d - de0) * 3600.0]
    peaks, sesna = plane(pra, pde), plane(ra_deg, dec_deg)
    d1, i1 = cKDTree(sesna).query(peaks, distance_upper_bound=tile.beam_px_max * tile.pixel_arcsec)
    ok = np.isfinite(d1)
    ok[ok] = d1[ok] < tile.beam_at(nb.x[ok], nb.y[ok])          # within the peak's own (local) beam
    if not ok.any():
        return np.zeros((0, 2))
    _, back = cKDTree(peaks).query(sesna[i1[ok]])
    mutual = np.flatnonzero(ok)[back == np.flatnonzero(ok)]
    return peaks[mutual] - sesna[i1[mutual]]


def registration(tile, ra_deg, dec_deg):
    """The map's astrometric offset from SESNA, (east arcsec, north arcsec, n_pairs): the median of PACS minus
    SESNA over `registration_pairs`. Fewer than MIN_REGISTRATION_PAIRS pairs: no shift. Planner 2026-10-27:
    Mon OB1's map 0640_p0927 sat 3.3 arcsec west of SESNA and a fixed-position fit there lost half the flux of
    its 6-arcsec beam; one shift per map keeps the fit fixed-position and linear."""
    delta = registration_pairs(tile, ra_deg, dec_deg)
    if delta.shape[0] < MIN_REGISTRATION_PAIRS:
        return 0.0, 0.0, int(delta.shape[0])
    return float(np.median(delta[:, 0])), float(np.median(delta[:, 1])), int(delta.shape[0])


# ----------------------------------------------------------------------------
# results

@dataclass(frozen=True)
class Neighbours:
    """Blind bright sources on one tile: map pixels x, y, flux (mJy), snr."""
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    snr: np.ndarray


@dataclass(frozen=True)
class PositionFit:
    """Forced fits at n positions.  Rows with covered False carry NaN.

    flux       (n,) mJy, summed amplitude of the scan columns (two; one for a one-direction map)
    gain       (n,) response of `flux` to a unit-flux injected beam (50/50 scan mix)
    sigma      (n,) propagated pixel noise on `flux`, mJy
    covered    (n,) bool, the position lies on valid data and the fit is well posed
    n_nuisance (n,) int, bright-neighbour sources included as nuisance columns
    """
    flux: np.ndarray
    gain: np.ndarray
    sigma: np.ndarray
    covered: np.ndarray
    n_nuisance: np.ndarray


@dataclass(frozen=True)
class ApertureFit:
    """Forced aperture photometry at n positions (the eHOPS apertures, pacs_psf.APERTURES).
    Rows with covered False carry NaN.

    flux        (n,) mJy: aperture sum minus the annulus sky (taken after the source's own point-spread outer
                light is removed from the annulus), divided by the fraction of the total flux inside the
                aperture radius (pacs_psf.aperture_eef), so a point source returns its flux (gain 1)
    sky         (n,) mJy per pixel, the clipped annulus sky (IDL mmm)
    annulus_sigma (n,) mJy: the annulus pixel scatter (after clipping) x sqrt(aperture area in pixels),
                corrected like the flux -- the per-source noise eHOPS-style tools quote; not used for
                detection
    covered     (n,) bool: >= 90 % of the aperture weight and >= half of the annulus on valid pixels
    """
    flux: np.ndarray
    sky: np.ndarray
    annulus_sigma: np.ndarray
    covered: np.ndarray


# ----------------------------------------------------------------------------
# the numerical core

@numba.njit(cache=True)
def _box_templates(table, extent, steps, map_nx, map_ny, sx, sy, x0, y0, side, out, row):
    """Pixel response of the scan columns (one or two, table.shape[2]) of a source at (sx, sy) on the
    box whose lower-left pixel is (x0, y0): nearest spatial node of the response table, bilinear
    in pixel displacement. Writes out[row .. row+n_component-1]; zero beyond the table."""
    nsy, nsx = table.shape[0], table.shape[1]
    nk = table.shape[2]
    last = table.shape[-1] - 1
    ix = min(max(int(round((sx + 0.5) / map_nx * (nsx - 1))), 0), nsx - 1)
    iy = min(max(int(round((sy + 0.5) / map_ny * (nsy - 1))), 0), nsy - 1)
    for j in range(side):
        dy = (y0 + j - sy + extent) * steps
        for i in range(side):
            p = j * side + i
            dx = (x0 + i - sx + extent) * steps
            if dx < 0 or dx > last or dy < 0 or dy > last:
                for m in range(nk):
                    out[row + m, p] = 0.0
                continue
            jx = min(int(dx), last - 1)
            jy = min(int(dy), last - 1)
            ux = dx - jx
            uy = dy - jy
            for m in range(nk):
                im = table[iy, ix, m]
                out[row + m, p] = ((1 - ux) * (1 - uy) * im[jy, jx] + ux * (1 - uy) * im[jy, jx + 1]
                                   + ux * uy * im[jy + 1, jx + 1] + (1 - ux) * uy * im[jy + 1, jx])


@numba.njit(cache=True)
def _mix_templates(subs, fw, extent, steps, map_nx, map_ny, sx, sy, x0, y0, side, out, row, tmp):
    """`_box_templates` of the mixture sum_k fw[k] x subs[k] (the response tables of a mosaic's sub-groups, or the one
    table of a map with fw = [1]); `tmp` is scratch of the same shape as the rows written."""
    nk = subs[0].shape[2]
    for m in range(nk):
        for p in range(side * side):
            out[row + m, p] = 0.0
    for k in range(len(subs)):
        if fw[k] > 0:
            _box_templates(subs[k], extent, steps, map_nx, map_ny, sx, sy, x0, y0, side, tmp, 0)
            for m in range(nk):
                for p in range(side * side):
                    out[row + m, p] += fw[k] * tmp[m, p]


@numba.njit(cache=True)
def _fit_core(img, err, subs, fw, extent, steps, tx, ty, tn, ux, uy, un, cx, cy, half, min_beam, r_in, r_out):
    """Weighted linear fits, one per row. Row i: tn[i] target sources at (tx, ty), un[i]
    nuisance sources at (ux, uy), box centred on pixel (cx[i], cy[i]) of half-width `half`.
    The background is the HOPS/eHOPS convention (owner ruling 2026-10-27): the clipped (IDL mmm) sky of
    the valid pixels in the annulus r_in..r_out (pixels) around the first target, subtracted from the
    data; every PSF column has its own mean over the same annulus subtracted, so a point source returns
    its full flux (the share of its wing the annulus removes is in the model); no free constant.
    The point-spread table is the mixture of `subs` with the row's fractions fw[i] (the local mixture of a mosaic).
    Returns per-member flux, gain, covariance and a validity flag."""
    n, mmax = tx.shape
    side = 2 * half + 1
    npix = side * side
    map_ny, map_nx = img.shape
    kmax = ux.shape[1]
    nk = subs[0].shape[2]                # scan columns per source: 1 (one-direction map) or 2
    tmp = np.empty((nk, side * side))
    cmax = nk * (mmax + kmax)
    abuf = np.empty(npix)
    flux = np.full((n, mmax), np.nan)
    gain = np.full((n, mmax), np.nan)
    cov = np.full((n, mmax, mmax), np.nan)
    valid = np.zeros(n, dtype=np.bool_)
    X = np.empty((cmax, npix))
    w = np.empty(npix)
    d = np.empty(npix)
    e = np.empty(npix)
    ok = np.zeros(npix, dtype=np.bool_)
    A = np.empty((cmax, cmax))
    B = np.empty((cmax, mmax))
    L = np.empty((mmax, npix))
    scale = np.empty(cmax)
    for i in range(n):
        mt = tn[i]
        kn = un[i]
        nc = nk * (mt + kn)
        x0 = cx[i] - half
        y0 = cy[i] - half
        for m in range(mt):
            _mix_templates(subs, fw[i], extent, steps, map_nx, map_ny, tx[i, m], ty[i, m], x0, y0, side, X, nk * m, tmp)
        for k in range(kn):
            _mix_templates(subs, fw[i], extent, steps, map_nx, map_ny, ux[i, k], uy[i, k], x0, y0, side, X,
                           nk * (mt + k), tmp)
        for j in range(side):
            yy = y0 + j
            for ii in range(side):
                p = j * side + ii
                xx = x0 + ii
                good = False
                if 0 <= xx < map_nx and 0 <= yy < map_ny:
                    v = img[yy, xx]
                    s = err[yy, xx]
                    good = np.isfinite(v) and np.isfinite(s) and s > 0
                ok[p] = good
                if good:
                    d[p] = v
                    e[p] = s
                    w[p] = 1.0 / (s * s)
                else:
                    d[p] = 0.0
                    e[p] = 0.0
                    w[p] = 0.0
        # the beam of each target must sit on valid pixels
        enough = True
        for m in range(mt):
            tot = 0.0
            got = 0.0
            for p in range(npix):
                t = X[nk * m, p]
                for q in range(1, nk):
                    t += X[nk * m + q, p]
                tot += t
                if ok[p]:
                    got += t
            if tot <= 0 or got < min_beam * tot:
                enough = False
        if not enough:
            continue
        # the annulus around the first target: clipped sky of the data, plain mean of each column
        ax = tx[i, 0]
        ay = ty[i, 0]
        na = 0
        n_ann = 0
        for j in range(side):
            for ii in range(side):
                dd = np.hypot(x0 + ii - ax, y0 + j - ay)
                if r_in <= dd <= r_out:
                    n_ann += 1
                    p = j * side + ii
                    if ok[p]:
                        abuf[na] = d[p]
                        na += 1
        if na < 10 or 2 * na < n_ann:
            continue
        sky, _ = _mmm_sky(abuf[:na])
        for c in range(nc):
            sm = 0.0
            for j in range(side):
                for ii in range(side):
                    dd = np.hypot(x0 + ii - ax, y0 + j - ay)
                    p = j * side + ii
                    if r_in <= dd <= r_out and ok[p]:
                        sm += X[c, p]
            mean = sm / na
            for p in range(npix):
                X[c, p] -= mean
        for p in range(npix):
            if ok[p]:
                d[p] -= sky
        for a in range(nc):
            for b in range(a, nc):
                s = 0.0
                for p in range(npix):
                    s += w[p] * X[a, p] * X[b, p]
                A[a, b] = s
                A[b, a] = s
        bad = False
        for a in range(nc):
            if A[a, a] <= 0:
                bad = True
                break
            scale[a] = np.sqrt(A[a, a])
        if bad:
            continue
        for a in range(nc):
            for b in range(nc):
                A[a, b] = A[a, b] / (scale[a] * scale[b])
        for a in range(nc):
            for m in range(mt):
                B[a, m] = 0.0
        for m in range(mt):
            for q in range(nk):
                B[nk * m + q, m] = 1.0 / scale[nk * m + q]
        # Gauss-Jordan with partial pivoting on the equilibrated normal matrix
        singular = False
        for c in range(nc):
            piv = c
            best = abs(A[c, c])
            for r in range(c + 1, nc):
                if abs(A[r, c]) > best:
                    best = abs(A[r, c])
                    piv = r
            if best < 1e-10:
                singular = True
                break
            if piv != c:
                for q in range(nc):
                    t = A[c, q]
                    A[c, q] = A[piv, q]
                    A[piv, q] = t
                for q in range(mt):
                    t = B[c, q]
                    B[c, q] = B[piv, q]
                    B[piv, q] = t
            inv = 1.0 / A[c, c]
            for q in range(nc):
                A[c, q] *= inv
            for q in range(mt):
                B[c, q] *= inv
            for r in range(nc):
                if r != c:
                    f = A[r, c]
                    if f != 0.0:
                        for q in range(nc):
                            A[r, q] -= f * A[c, q]
                        for q in range(mt):
                            B[r, q] -= f * B[c, q]
        if singular:
            continue
        for m in range(mt):
            for p in range(npix):
                s = 0.0
                for a in range(nc):
                    s += B[a, m] / scale[a] * X[a, p]
                L[m, p] = s * w[p]
        for m in range(mt):
            f = 0.0
            g = 0.0
            for p in range(npix):
                f += L[m, p] * d[p]
                t = X[nk * m, p]
                for q in range(1, nk):
                    t += X[nk * m + q, p]
                g += L[m, p] * (1.0 / nk) * t
            flux[i, m] = f
            gain[i, m] = g
            for m2 in range(m, mt):
                c2 = 0.0
                for p in range(npix):
                    c2 += L[m, p] * L[m2, p] * e[p] * e[p]
                cov[i, m, m2] = c2
                cov[i, m2, m] = c2
        valid[i] = True
    return flux, gain, cov, valid


# ----------------------------------------------------------------------------
# cell grid

def _mix64(counter):
    """SplitMix64 mixing with intentional uint64 wraparound."""
    value = counter + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value


def _uniform(counter):
    """Uniform float64 values from the upper 53 mixed counter bits."""
    return (_mix64(counter) >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)


@dataclass(frozen=True)
class MapGrid:
    """The 30-arcsec cell grid of one map: y-major cell ids, cells aligned with the pixel
    axes, edge cells clipped to the image (pixel centres are integers, edges at -0.5, n-0.5).

    cell_ids(x, y)            int64 ids, -1 outside the image
    bounds(ids)               x0, x1, y0, y1
    centres_icrs(ids)         ra, dec in degrees
    sample_pixels(ids, n, s)  (n_ids, n) uniform x and y in each cell, reproducible per
                              (seed, cell, draw) whatever the batching of the cells
    """
    wcs: object
    shape: tuple
    step_x: float
    step_y: float
    nx: int
    ny: int

    @classmethod
    def from_wcs(cls, wcs, shape):
        wcs = wcs.celestial
        scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=np.float64) * 3600.0
        if not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError("invalid map pixel scale")
        matrix = wcs.pixel_scale_matrix
        cosine = np.dot(matrix[:, 0], matrix[:, 1]) / np.prod(scales / 3600.0)
        if abs(cosine) > 1e-6 or wcs.has_distortion:
            raise ValueError("skewed or distorted WCS needs explicit cell geometry")
        sx, sy = CELL_ARCSEC / scales
        shape = tuple(int(v) for v in shape)
        return cls(wcs, shape, float(sx), float(sy),
                   int(np.ceil(shape[1] / sx)), int(np.ceil(shape[0] / sy)))

    @property
    def n_cells(self):
        return self.nx * self.ny

    def cell_ids(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        inside = (np.isfinite(x) & np.isfinite(y) & (x >= -0.5) & (x < self.shape[1] - 0.5)
                  & (y >= -0.5) & (y < self.shape[0] - 0.5))
        out = np.full(x.shape, -1, dtype=np.int64)
        ix = np.floor((x[inside] + 0.5) / self.step_x).astype(np.int64)
        iy = np.floor((y[inside] + 0.5) / self.step_y).astype(np.int64)
        out[inside] = iy * self.nx + ix
        return out

    def bounds(self, cell_ids):
        ids = np.asarray(cell_ids)
        x0 = (ids % self.nx) * self.step_x - 0.5
        y0 = (ids // self.nx) * self.step_y - 0.5
        return (x0, np.minimum(x0 + self.step_x, self.shape[1] - 0.5),
                y0, np.minimum(y0 + self.step_y, self.shape[0] - 0.5))

    def centres_icrs(self, cell_ids):
        x0, x1, y0, y1 = self.bounds(cell_ids)
        sky = self.wcs.pixel_to_world((x0 + x1) / 2, (y0 + y1) / 2).icrs
        return sky.ra.deg, sky.dec.deg

    def sample_pixels(self, cell_ids, n_positions, seed):
        ids = np.asarray(cell_ids)
        x0, x1, y0, y1 = self.bounds(ids)
        counter = ids.astype(np.uint64)[:, None] << np.uint64(32)
        draws = np.arange(2 * n_positions, dtype=np.uint64)[None, :]
        key = _mix64(np.array([seed], dtype=np.uint64))[0]
        u = _uniform((counter + draws) ^ key)
        return (x0[:, None] + (x1 - x0)[:, None] * u[:, 0::2],
                y0[:, None] + (y1 - y0)[:, None] * u[:, 1::2])


# ----------------------------------------------------------------------------
# map access

@dataclass
class OpenMap:
    image: np.ndarray
    error: np.ndarray
    coverage: np.ndarray
    wcs: WCS
    record: MapRecord


@contextmanager
def open_map(config, record):
    """Open a map read-only and memory-mapped; the three layers are borrowed views valid
    only inside the context."""
    if getattr(record, "mosaic_dir", ""):         # a mosaic: its layers are the product's memory-mapped .npy files
        arrays = [np.load(os.path.join(record.mosaic_dir, n + ".npy"), mmap_mode="r") for n in ("image", "error", "coverage")]
        if any(a.shape != tuple(record.shape) for a in arrays):
            raise ValueError("inconsistent mosaic layers: %s" % record.mosaic_dir)
        yield OpenMap(*arrays, WCS(fits.Header.fromstring(record.wcs_header)).celestial, record)
        return
    path = Path(maps_root(config)) / record.path
    if path.suffix.lower() in (".gz", ".zip"):
        raise ValueError("compressed maps must be unpacked before memory-mapped access")
    with fits.open(path, mode="denywrite", memmap=True) as hdus:
        header = fits.Header.fromstring(record.wcs_header)
        arrays = [hdus[i].data for i in (record.image_hdu, record.error_hdu, record.coverage_hdu)]
        if any(a is None or a.shape != tuple(record.shape) for a in arrays):
            raise ValueError("inconsistent map layers: %s" % path)
        for a in arrays:
            a.flags.writeable = False
        yield OpenMap(*arrays, WCS(header).celestial, record)


# ----------------------------------------------------------------------------
# the tile

@numba.njit(cache=True)
def _mmm_sky(v):
    """The IDL mmm sky of the values v: 3-sigma clipping about the median to convergence (at most 20
    passes), then 3 median - 2 mean where the mean exceeds the median (a positively skewed sky, as
    from faint sources), else the mean. Returns (sky, clipped standard deviation)."""
    v = np.sort(v)
    lo, hi = 0, v.size
    for _ in range(20):
        w = v[lo:hi]
        if w.size < 3:
            break
        med = w[w.size // 2]
        sd = np.std(w)
        nlo = lo + np.searchsorted(w, med - 3.0 * sd)
        nhi = lo + np.searchsorted(w, med + 3.0 * sd, side="right")
        if nlo == lo and nhi == hi:
            break
        lo, hi = nlo, nhi
    w = v[lo:hi]
    med = np.median(w)
    mean = np.mean(w)
    sky = 3.0 * med - 2.0 * mean if mean > med else mean
    return sky, np.std(w)


@numba.njit(cache=True)
def _aperture_core(img, err, xs, ys, r_ap, r_in, r_out, nsub):
    """Aperture sum (sub-pixel weights, nsub x nsub samples per pixel) minus the mmm annulus sky (pixel
    centres in the annulus), at each (x, y) in pixels. Returns raw flux, sky, annulus scatter, aperture
    area (pixels), covered."""
    return _aperture_core_wing(img, err, xs, ys, r_ap, r_in, r_out, nsub, np.zeros((1, 1, 1)), np.zeros(xs.size))


@numba.njit(cache=True)
def _aperture_core_wing(img, err, xs, ys, r_ap, r_in, r_out, nsub, model, amp):
    """`_aperture_core` with the source's own outer light removed from the annulus before the sky is taken: the
    annulus pixel at (yy, xx) enters as img - amp[i] x model[i, yy - cy + rb, xx - cx + rb] (`model` the unit-flux
    point-spread response on the (2 rb + 1)^2 box around the pixel (cx, cy) nearest the position, rb =
    ceil(r_out) + 1; not read where amp[i] is 0)."""
    n = xs.size
    ny, nx = img.shape
    out_f = np.full(n, np.nan)
    out_s = np.full(n, np.nan)
    out_r = np.full(n, np.nan)
    out_a = np.full(n, np.nan)
    cov = np.zeros(n, dtype=np.bool_)
    rb = int(np.ceil(r_out)) + 1
    buf = np.empty((2 * rb + 1) ** 2)
    for i in range(n):
        x0 = xs[i]
        y0 = ys[i]
        cx = int(np.floor(x0 + 0.5))
        cy = int(np.floor(y0 + 0.5))
        total_w = 0.0
        good_w = 0.0
        acc = 0.0
        nb = 0
        n_ann = 0
        for yy in range(cy - rb, cy + rb + 1):
            for xx in range(cx - rb, cx + rb + 1):
                dc = np.hypot(xx - x0, yy - y0)
                inside = 0 <= xx < nx and 0 <= yy < ny
                valid = False
                if inside:
                    v = img[yy, xx]
                    e = err[yy, xx]
                    valid = np.isfinite(v) and np.isfinite(e) and e > 0
                if dc <= r_ap + 0.75:
                    if dc <= r_ap - 0.75:
                        wgt = 1.0
                    else:
                        c = 0
                        for a in range(nsub):
                            for b in range(nsub):
                                if np.hypot(xx - 0.5 + (a + 0.5) / nsub - x0, yy - 0.5 + (b + 0.5) / nsub - y0) <= r_ap:
                                    c += 1
                        wgt = c / (nsub * nsub)
                    total_w += wgt
                    if valid:
                        good_w += wgt
                        acc += wgt * img[yy, xx]
                if r_in <= dc <= r_out:
                    n_ann += 1
                    if valid:
                        v = img[yy, xx]
                        if amp[i] != 0.0:
                            v -= amp[i] * model[i, yy - cy + rb, xx - cx + rb]
                        buf[nb] = v
                        nb += 1
        if total_w <= 0 or good_w < 0.9 * total_w or nb < max(10, n_ann // 2):
            continue
        sky, sd = _mmm_sky(buf[:nb])
        # the missing aperture weight is filled at the sky level (at most 10 per cent of the aperture)
        out_f[i] = acc * total_w / good_w - sky * total_w
        out_s[i] = sky
        out_r[i] = sd
        out_a[i] = total_w
        cov[i] = True
    return out_f, out_s, out_r, out_a, cov


class Tile:
    """One open map tile with its fitting machinery (build with `open_tile`)."""

    def __init__(self, record, wcs, config, image, error, psf=None, neighbours=None, image_path=None):
        """`image`, `error`: the map's layers in mJy per pixel with unusable pixels NaN in the error (`_load_layers`);
        `psf`, `neighbours`: the prepared point-spread cache and bright-neighbour list (`prepare_map`), built here when
        None; `image_path`: the prepared image file, re-opened by `writable`."""
        self.wcs = wcs
        self.shape = tuple(record.shape)
        scale = proj_plane_pixel_scales(wcs)
        self.pixel_arcsec = float(np.mean(scale)) * 3600.0
        self.beam_arcsec = beam_fwhm_arcsec(record)
        self.record = record
        self.aperture_fraction = aperture_fraction(config, record)
        # the fraction of the total flux inside the aperture radius (the measurement removes the source's own
        # outer light from the annulus, so no annulus share is taken off)
        self.enclosed_fraction = aperture_eef(config, record)
        # the PSF fit's amplitude is already the total flux, the products' convention (owner 2026-10-27)
        self.flux_scale = 1.0
        self.beam_px = self.beam_arcsec / self.pixel_arcsec
        # the box holds the eHOPS sky annulus (pacs_psf.APERTURES) and at least FIT_BOX_FWHM beams
        _, ann_in, ann_out = APERTURES[record.band]
        self.ann_in_px, self.ann_out_px = ann_in / self.pixel_arcsec, ann_out / self.pixel_arcsec
        self.half = int(np.ceil(max(FIT_BOX_FWHM * self.beam_px, self.ann_out_px + 1.0)))
        self.table_half = int(min(self.half + np.ceil(NEIGHBOUR_REACH_FWHM * self.beam_px),
                                  PSF_TABLE_HALF_ARCSEC // self.pixel_arcsec))
        self.reach = float(self.table_half)            # nuisance inclusion radius, pixels
        self.grid = MapGrid.from_wcs(wcs, self.shape)
        self.image, self.error = image, error
        self._image_path = image_path
        if psf is None and getattr(record, "mosaic_dir", ""):
            # the mosaic's mixture PSF (built by `build_mosaic_psf`) is already on the output grid and normalised
            self._psf = MosaicPSF(record.mosaic_dir, maps_root(config))
            self.psf_scale = 1.0
            if self._psf.extent != self.table_half + 0.5:
                raise ValueError("the mosaic PSF table was built for another table half-width")
        elif psf is None:
            components = published_components_for_map(config, record)
            self._psf = CachedPairedPSF.from_map(
                wcs, self.shape, components, half_size=self.table_half, integration_order=8,
                spatial_nodes=PSF_SPATIAL_NODES, subpixel_steps=PSF_SUBPIXEL_STEPS)
            # one normalisation for every use of the table (fit amplitude, artificial source): the release EEF's
            # total-flux fraction inside the aperture radius, the one the aperture photometry divides by
            self.psf_scale = 1.0 / self._enclosed_over_eef(config)
            self._psf = self._psf.scaled(self.psf_scale)
        else:
            self._psf = psf
        self._tree = None
        self.neighbours = self._find_neighbours() if neighbours is None else neighbours
        if self.neighbours.x.size:
            self._tree = cKDTree(np.c_[self.neighbours.x, self.neighbours.y])

    def beam_at(self, x, y):
        """The beam FWHM (arcsec) at map pixel positions: a mosaic's local mixture of its input maps' beams
        (`MosaicPSF.beam_at`), a map's one beam."""
        x = np.asarray(x, float)
        if isinstance(self._psf, MosaicPSF):
            return self._psf.beam_at(x.ravel(), np.asarray(y, float).ravel()).reshape(x.shape)
        return np.full(x.shape, self.beam_arcsec)

    def beam_px_at(self, x, y):
        """`beam_at` in pixels."""
        return self.beam_at(x, y) / self.pixel_arcsec

    @property
    def beam_px_max(self):
        """The widest beam (pixels) the tile's mixture holds."""
        return (float(self._psf.sub_beams.max()) if isinstance(self._psf, MosaicPSF) else self.beam_arcsec) / self.pixel_arcsec

    @property
    def beam_px_min(self):
        """The narrowest beam (pixels) the tile's mixture holds."""
        return (float(self._psf.sub_beams.min()) if isinstance(self._psf, MosaicPSF) else self.beam_arcsec) / self.pixel_arcsec

    @contextmanager
    def writable(self):
        """Inside the context the tile's image is the prepared file memory-mapped copy-on-write, so that `injected`
        can add sources to it (the prepared image is otherwise read-only, shared by every worker); the private pages
        are dropped on exit. For the one-off real-map check; the injections of `Injector.measure_positions` work on
        a private buffer of rows instead."""
        real = self.image
        self.image = np.load(self._image_path, mmap_mode="c")
        try:
            yield self
        finally:
            self.image = real

    def _enclosed_over_eef(self, config):
        """The point-spread table's flux inside the aperture radius (no sky subtracted), for a source of unit
        amplitude on an empty image, over the release EEF's total-flux fraction there (`pacs_psf.aperture_eef`):
        the mean over a 3 x 3 sub-pixel lattice at the valid position nearest the map centre (on a step-2 lattice;
        ties by row, then column) whose aperture and annulus lie wholly on valid data. The table is divided by it
        (P70 scan maps: 0.996). Works on a window around that position only."""
        r_ap, r_in, r_out = (r / self.pixel_arcsec for r in APERTURES[self.record.band])
        reach = int(np.ceil(r_out)) + 2
        h = int(np.ceil(self._psf.extent)) + 1
        margin = max(reach, h)
        ny, nx = self.shape
        y0 = x0 = None
        cy0, cx0 = ny / 2, nx / 2
        # lattice points margin, margin + 2, ... in growing squares about the centre: within radius R of the centre
        # every lattice point lies in the square of half-side R, so the first valid point found by distance inside
        # it is the nearest valid point of the whole lattice
        r_max = np.hypot(ny, nx)
        R = 16.0
        while y0 is None:
            ys = np.arange(margin, ny - margin, 2)
            xs = np.arange(margin, nx - margin, 2)
            ys = ys[np.abs(ys - cy0) <= R]
            xs = xs[np.abs(xs - cx0) <= R]
            if ys.size and xs.size:
                gy, gx = np.meshgrid(ys, xs, indexing="ij")
                dist = np.hypot(gy - cy0, gx - cx0).ravel()
                inside = dist <= R
                order = np.lexsort((gx.ravel()[inside], gy.ravel()[inside], dist[inside]))
                for k in order:
                    yy, xx = int(gy.ravel()[inside][k]), int(gx.ravel()[inside][k])
                    if np.isfinite(self.error[yy - reach:yy + reach + 1, xx - reach:xx + reach + 1]).all():
                        y0, x0 = yy, xx
                        break
            if R > r_max:
                break
            R *= 2.0
        if y0 is None:
            raise ValueError("%s: no position with a wholly valid aperture and annulus" % self.record.map_id)
        sub = np.arange(3) / 3.0
        x = (x0 + sub[None, :] + 0 * sub[:, None]).ravel()
        y = (y0 + sub[:, None] + 0 * sub[None, :]).ravel()
        # a window holding each source's patch and the aperture and annulus around it
        w = max(self.injection_half_width(), int(np.ceil(r_out)) + 1) + 3
        ya, yb, xa, xb = max(y0 - w, 0), min(y0 + w + 1, ny), max(x0 - w, 0), min(x0 + w + 1, nx)
        err = np.asarray(self.error[ya:yb, xa:xb])
        enclosed = []
        for i in range(x.size):            # one source at a time: the nine positions overlap
            img = np.zeros((yb - ya, xb - xa), np.float32)
            for cx, cy, resp in self._template_patches(x[i:i + 1], y[i:i + 1], 1.0):
                _paste(img, int(cx[0]) - xa, int(cy[0]) - ya, resp[0])
            f, sky, _sd, area, cov = _aperture_core(img, err, x[i:i + 1] - xa, y[i:i + 1] - ya, r_ap, r_in, r_out, 5)
            if not cov.all():
                raise ValueError("%s: the normalisation position is not covered" % self.record.map_id)
            enclosed.append(float((f + sky * area)[0]))
        return float(np.mean(enclosed)) / aperture_eef(config, self.record)

    # -- geometry ---------------------------------------------------------
    def xy(self, ra_deg, dec_deg):
        """ICRS degrees -> zero-origin map pixels (x, y); one vectorised WCS call."""
        x, y = self.wcs.all_world2pix(np.asarray(ra_deg, float), np.asarray(dec_deg, float), 0)
        return np.asarray(x), np.asarray(y)

    def _call(self, tx, ty, tn, ux, uy, un, cx, cy, half):
        psf = self._psf
        if isinstance(psf, MosaicPSF):               # the local mixture of the sub-groups' tables at each target
            subs, fw = tuple(psf.subs), psf.weights(tx[:, 0], ty[:, 0])
        else:
            subs, fw = (psf.table,), np.ones((tx.shape[0], 1))
        return _fit_core(self.image, self.error, subs, fw, psf.extent, psf.subpixel_steps,
                         tx, ty, tn, ux, uy, un, cx, cy, half, MIN_BEAM_FRACTION,
                         self.ann_in_px, self.ann_out_px)

    # -- bright neighbours (11.3 step 1) -------------------------------------
    def _find_neighbours(self):
        """Blind, deterministic: local maxima (window one beam) whose single-PSF fit at the
        peak, position refined by a 3x3 parabola, reaches S/N >= NEIGHBOUR_SNR against the
        propagated pixel noise."""
        r = max(1, int(round(self.beam_px_min)))         # a mosaic: the narrowest beam, wider ones thinned below
        ny, nx = self.shape
        cols = {k: [] for k in ("py", "px", "v0", "left", "right", "down", "up")}
        # strip by strip, each with r + 1 rows of margin (the filter's reach and the 3 x 3 below): the same peaks
        # and values as the whole map at once
        for s0 in range(0, ny, PREPARE_STRIP_ROWS):
            s1 = min(s0 + PREPARE_STRIP_ROWS, ny)
            a0, a1 = max(s0 - r - 1, 0), min(s1 + r + 1, ny)
            err = np.asarray(self.error[a0:a1])
            img = np.where(np.isfinite(err), np.asarray(self.image[a0:a1]), -np.inf).astype(np.float32)
            peak = (img == maximum_filter(img, size=2 * r + 1, mode="constant", cval=-np.inf))
            snr = np.zeros(img.shape, dtype=np.float32)
            np.divide(img, err, out=snr, where=np.isfinite(err))
            peak &= snr >= NEIGHBOUR_PEAK_SNR
            peak[:s0 - a0] = False
            peak[s1 - a0:] = False
            if s0 == 0:
                peak[:1] = False
            if s1 == ny:
                peak[ny - 1 - a0:] = False
            peak[:, :1] = peak[:, -1:] = False
            qy, qx = np.nonzero(peak)
            cols["py"].append(qy + a0)
            cols["px"].append(qx)
            cols["v0"].append(img[qy, qx].astype(float))
            cols["left"].append(np.maximum(img[qy, qx - 1], 0))
            cols["right"].append(np.maximum(img[qy, qx + 1], 0))
            cols["down"].append(np.maximum(img[qy - 1, qx], 0))
            cols["up"].append(np.maximum(img[qy + 1, qx], 0))
        py, px, v0, left, right, down, up = (np.concatenate(cols[k]) for k in ("py", "px", "v0", "left", "right", "down", "up"))
        if px.size == 0:
            return Neighbours(*(np.empty(0) for _ in range(4)))
        if isinstance(self._psf, MosaicPSF):
            # a peak must be the maximum within ITS beam: where the local beam is wider than the filter's window, a
            # brighter peak within that window (Chebyshev, as the filter) removes it
            wide = np.flatnonzero(np.round(self.beam_px_at(px, py)) > r)
            if wide.size:
                radius = np.round(self.beam_px_at(px[wide], py[wide]))
                near = cKDTree(np.c_[px, py]).query_ball_point(np.c_[px[wide], py[wide]], radius, p=np.inf)
                drop = np.array([bool(len(l) > 1 and v0[l].max() > v0[i]) for i, l in zip(wide, near)])
                keep = np.ones(px.size, bool)
                keep[wide[drop]] = False
                py, px, v0, left, right, down, up = (a[keep] for a in (py, px, v0, left, right, down, up))
        # parabolic sub-pixel refinement from the 3x3 around each peak
        with np.errstate(all="ignore"):
            dx = 0.5 * (right - left) / (2 * v0 - left - right)
            dy = 0.5 * (up - down) / (2 * v0 - down - up)
        dx = np.clip(np.nan_to_num(dx), -0.5, 0.5)
        dy = np.clip(np.nan_to_num(dy), -0.5, 0.5)
        x, y = px + dx, py + dy
        fit = self._fit(x, y, use_nuisance=False)
        keep = fit.covered & (fit.flux > 0) & (fit.flux >= NEIGHBOUR_SNR * fit.sigma)
        return Neighbours(x[keep], y[keep], fit.flux[keep], (fit.flux / fit.sigma)[keep])

    # -- forced fits --------------------------------------------------------
    def _nuisance(self, x, y, own_exclude=()):
        """Nearest bright neighbours (up to MAX_NUISANCE) within the reach of each position,
        except those within one beam of any exclusion point (the target's own column carries
        them). Returns ux, uy (n, K) and counts (n,)."""
        n = x.size
        ux = np.zeros((n, MAX_NUISANCE))
        uy = np.zeros((n, MAX_NUISANCE))
        count = np.zeros(n, dtype=np.int64)
        if self._tree is None or n == 0:
            return ux, uy, count
        dist, idx = self._tree.query(np.c_[x, y], k=MAX_NUISANCE, distance_upper_bound=self.reach)
        dist = dist.reshape(n, -1)
        idx = idx.reshape(n, -1)
        k = dist.shape[1]
        have = np.isfinite(dist)
        nx_, ny_ = self.neighbours.x, self.neighbours.y
        safe = np.where(have, idx, 0)
        beam = self.beam_px_at(x, y)[:, None]                  # the target's own (local) beam
        near = have & (dist < beam)
        for ex, ey in own_exclude:
            near |= have & (np.hypot(nx_[safe] - ex[:, None], ny_[safe] - ey[:, None]) < beam)
        use = have & ~near
        order = np.argsort(~use, axis=1, kind="stable")
        use = np.take_along_axis(use, order, axis=1)
        safe = np.take_along_axis(safe, order, axis=1)
        ux[:, :k] = np.where(use, nx_[safe], 0.0)
        uy[:, :k] = np.where(use, ny_[safe], 0.0)
        return ux, uy, use.sum(axis=1)

    def _fit(self, x, y, use_nuisance=True, batch=512):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.size
        out = [np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan),
               np.zeros(n, dtype=bool), np.zeros(n, dtype=np.int64)]
        ny, nx = self.shape
        cx = np.rint(x).astype(np.int64)
        cy = np.rint(y).astype(np.int64)
        inside = np.isfinite(x) & np.isfinite(y) & (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        rows = np.flatnonzero(inside)
        for start in range(0, rows.size, batch):
            r = rows[start:start + batch]
            m = r.size
            if use_nuisance:
                ux, uy, un = self._nuisance(x[r], y[r], own_exclude=[(x[r], y[r])])
            else:
                ux, uy, un = np.zeros((m, 1)), np.zeros((m, 1)), np.zeros(m, dtype=np.int64)
            flux, gain, cov, valid = self._call(x[r, None], y[r, None], np.ones(m, dtype=np.int64),
                                                ux, uy, un, cx[r], cy[r], self.half)
            out[0][r] = flux[:, 0] * self.flux_scale
            out[1][r] = gain[:, 0]
            out[2][r] = np.sqrt(np.maximum(cov[:, 0, 0], 0)) * self.flux_scale
            out[3][r] = valid
            out[4][r] = un
        return PositionFit(*out)

    def measure(self, x, y, image=None, row0=0):
        """The measurement of isolated sources and of the completeness positions (final ruling, spec
        section 12): eHOPS aperture photometry -> PositionFit (flux mJy on eHOPS's normalisation, gain 1,
        sigma = the annulus scatter statistic, covered, no nuisance). `image`, `row0`: see `aperture`."""
        a = self.aperture(x, y, image=image, row0=row0)
        return PositionFit(a.flux, np.where(a.covered, 1.0, np.nan), a.annulus_sigma, a.covered,
                           np.zeros(a.flux.size, dtype=np.int64))

    def aperture(self, x, y, chunk=2048, image=None, row0=0):
        """Forced aperture photometry at each (x, y) -> ApertureFit (the eHOPS apertures; the photometry of
        isolated sources and the completeness positions alike). The clipped (mmm) annulus sky rejects the
        pixels where a bright source's own outer light stands above the noise, so the annulus would remove
        less of that light than the encircled-energy table's average and bright sources would read up to
        2.5 % high, by an amount set by the local noise. So the sky is taken from the annulus after the
        source's own point-spread outer light, scaled to its flux, is subtracted (the flux from the previous
        pass: first with the table's aperture-and-annulus fraction, then twice with the outer light removed),
        and the sum is divided by the fraction of the total flux inside the aperture radius alone. `image`: the rows
        `row0` onward of the map's image, a private buffer to measure on in place of the tile's (the error layer is
        then the same rows, a view); positions are in map coordinates either way."""
        if image is None:
            image, error = self.image, self.error
        else:
            error = self.error[row0:row0 + image.shape[0]]
        r_ap, r_in, r_out = (r / self.pixel_arcsec for r in APERTURES[self.record.band])
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        n = x.size
        rb = int(np.ceil(r_out)) + 1
        offs = np.arange(-rb, rb + 1, dtype=float)
        flux = np.full(n, np.nan)
        sky = np.full(n, np.nan)
        sd = np.full(n, np.nan)
        area = np.full(n, np.nan)
        cov = np.zeros(n, bool)
        sd_eef = np.full(n, self.enclosed_fraction)
        for lo in range(0, n, chunk):
            sl = slice(lo, lo + chunk)
            xs, ys = x[sl], y[sl]
            f, s_, d_, a_, c_ = _aperture_core(image, error, xs, ys - row0, r_ap, r_in, r_out, 5)
            mix = self._psf.weights(xs, ys) if isinstance(self._psf, MosaicPSF) else None
            eef, fraction = self._corrections_at(xs, ys, mix)
            amp = np.where(c_, f / fraction, 0.0)
            if not c_.any():
                cov[sl] = c_
                continue
            model = self._outer_light(xs, ys, rb, mix)
            for _ in range(2):
                f, s_, d_, a_, c_ = _aperture_core_wing(image, error, xs, ys - row0, r_ap, r_in, r_out, 5,
                                                        model, np.where(c_ & np.isfinite(amp), amp, 0.0))
                amp = np.where(c_, f / eef, 0.0)
            flux[sl], sky[sl], sd[sl], area[sl], cov[sl] = f / eef, s_, d_, a_, c_
            sd_eef[sl] = eef
        return ApertureFit(flux, sky, sd * np.sqrt(area) / sd_eef, cov)

    def _corrections_at(self, xs, ys, mix=None):
        """(encircled-energy fraction, aperture fraction) that divide the aperture sum at these map positions:
        the map's two constants, or for a mosaic the local mixture of its sub-groups' (`MosaicPSF`; `mix`: the
        mixture fractions there)."""
        if isinstance(self._psf, MosaicPSF):
            return self._psf.enclosed_at(xs, ys, mix), self._psf.aperture_fraction_at(xs, ys, mix)
        return self.enclosed_fraction, self.aperture_fraction

    #: sub-pixel steps per pixel of the cached outer-light patterns
    OUTER_LIGHT_STEPS = 16

    #: patterns of a mosaic's outer-light cache kept at once (one per sub-group and third x third of the map)
    OUTER_LIGHT_KEPT = 64

    def _outer_pattern(self, k, node, rb):
        """The unit-flux response on the (2 rb + 1)^2 box of the sub-pixel lattice (b, a) of third x third `node`, of
        sub-group `k` alone (a map: the one table), kept in a least-recently-used cache."""
        from collections import OrderedDict
        steps = self.OUTER_LIGHT_STEPS
        ny, nx = self.shape
        if getattr(self, "_outer_cache", None) is None or self._outer_cache[0] != rb:
            self._outer_cache = (rb, OrderedDict())
        cache = self._outer_cache[1]
        if (k, node) in cache:
            cache.move_to_end((k, node))
            return cache[(k, node)]
        offs = np.arange(-rb, rb + 1, dtype=float)
        u = (np.arange(steps) + 0.5) / steps - 0.5
        cy0 = float(np.floor((node // 3 + 0.5) / 3 * ny))
        cx0 = float(np.floor((node % 3 + 0.5) / 3 * nx))
        sx = np.repeat(cx0 + u[None, :], steps, axis=0).ravel()      # [b, a] -> x offset u[a]
        sy = np.repeat(cy0 + u[:, None], steps, axis=1).ravel()      # [b, a] -> y offset u[b]
        px = np.broadcast_to(cx0 + offs[None, None, :], (sx.size, offs.size, offs.size))
        py = np.broadcast_to(cy0 + offs[None, :, None], (sx.size, offs.size, offs.size))
        if isinstance(self._psf, MosaicPSF):
            one = np.zeros((sx.size, self._psf.n_sub))
            one[:, k] = 1.0
            resp = self._psf.evaluate_xy(sx, sy, px, py, one).astype(np.float32)
        else:
            resp = self._psf.evaluate_xy(sx, sy, px, py)
        cache[(k, node)] = resp.mean(axis=1).reshape(steps, steps, offs.size, offs.size)
        if len(cache) > self.OUTER_LIGHT_KEPT:
            cache.popitem(last=False)
        return cache[(k, node)]

    def _outer_light(self, x, y, rb, mix=None):
        """The unit-flux point-spread response on the (2 rb + 1)^2 pixel box around the pixel nearest each (x, y)
        (the box `_aperture_core_wing` reads), looked up from patterns computed once: on a OUTER_LIGHT_STEPS x
        OUTER_LIGHT_STEPS lattice of sub-pixel positions, in each third x third of the map (the table's own spatial
        nodes). The lattice step (1/16 pixel) moves the pattern in the annulus by far less than the noise; it enters
        only the sky estimate. A mosaic holds a pattern per sub-group (at most OUTER_LIGHT_KEPT at once) and each
        position takes the mixture of them with its own fractions `mix` (n, n_sub)."""
        steps = self.OUTER_LIGHT_STEPS
        ny, nx = self.shape
        cx = np.floor(x + 0.5)
        cy = np.floor(y + 0.5)
        a = np.clip(np.floor((x - cx + 0.5) * steps).astype(np.int64), 0, steps - 1)
        b = np.clip(np.floor((y - cy + 0.5) * steps).astype(np.int64), 0, steps - 1)
        node = 3 * np.clip((3 * y / ny).astype(np.int64), 0, 2) + np.clip((3 * x / nx).astype(np.int64), 0, 2)
        if mix is None:
            out = np.empty((x.size, 2 * rb + 1, 2 * rb + 1))
            for nd in np.unique(node):
                sel = node == nd
                out[sel] = self._outer_pattern(0, int(nd), rb)[b[sel], a[sel]]
            return out
        out = np.zeros((x.size, 2 * rb + 1, 2 * rb + 1))
        for k in range(mix.shape[1]):
            for nd in np.unique(node[mix[:, k] > 0]):
                sel = np.flatnonzero((node == nd) & (mix[:, k] > 0))
                out[sel] += mix[sel, k, None, None] * self._outer_pattern(k, int(nd), rb)[b[sel], a[sel]]
        return out

    def fitted_fwhm(self, x, y):
        """Best-fit total FWHM (arcsec) of a source at each position (NaN where the fit is not
        possible): the beam broadened by a Gaussian of width chosen on a small grid by
        chi-square, two scan columns plus a constant background, no nuisance columns. Meant
        for detections only (photometry's EXTENDED flag is this > 1.5 x beam)."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        beam_px = self.beam_px_at(x, y)                      # the position's own (local) beam
        half = self.half
        side = 2 * half + 1
        psf = self._psf
        subs, fw = ((tuple(psf.subs), psf.weights(x, y)) if isinstance(psf, MosaicPSF)
                    else ((psf.table,), np.ones((x.size, 1))))
        out = np.full(x.size, np.nan)
        tmp = np.zeros((subs[0].shape[2], side * side))
        for i in range(x.size):
            cx, cy = int(round(x[i])), int(round(y[i]))
            y0, x0 = cy - half, cx - half
            if x0 < 0 or y0 < 0 or cx + half >= self.shape[1] or cy + half >= self.shape[0]:
                continue
            d = self.image[y0:y0 + side, x0:x0 + side].astype(float).ravel()
            e = self.error[y0:y0 + side, x0:x0 + side].astype(float).ravel()
            ok = np.isfinite(d) & np.isfinite(e)
            if ok.sum() < 0.8 * d.size:
                continue
            t = np.zeros_like(tmp)
            _mix_templates(subs, fw[i], psf.extent, psf.subpixel_steps, self.shape[1], self.shape[0],
                           x[i], y[i], x0, y0, side, t, 0, tmp)
            best, best_chi = np.nan, np.inf
            for s in np.array([0.0, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]) * beam_px[i] / 4.0:   # Gaussian sigma, pixels
                cols = [gaussian_filter(c.reshape(side, side), s).ravel() if s > 0 else c for c in t]
                M = np.c_[(*cols, np.ones(side * side))][ok] / e[ok, None]
                coef = np.linalg.lstsq(M, d[ok] / e[ok], rcond=None)[0]
                chi = np.sum((M @ coef - d[ok] / e[ok]) ** 2)
                if chi < best_chi:
                    best_chi, best = chi, s
            out[i] = np.hypot(beam_px[i] * self.pixel_arcsec, 2.3548 * best * self.pixel_arcsec)
        return out

    def near_bright(self, x, y):
        """True where a position lies within one beam of a bright-neighbour peak (step 1 list)."""
        x = np.asarray(x, float)
        if self._tree is None:
            return np.zeros(x.shape, dtype=bool)
        d, _ = self._tree.query(np.c_[x.ravel(), np.asarray(y, float).ravel()], k=1,
                                distance_upper_bound=self.beam_px_max)
        return (d < self.beam_px_at(x, y).ravel()).reshape(x.shape)

    def pixel_sigma(self, x, y):
        """Error-layer value (mJy/pixel) at the nearest pixel, NaN off the data."""
        cx = np.rint(np.asarray(x)).astype(np.int64)
        cy = np.rint(np.asarray(y)).astype(np.int64)
        ok = (cx >= 0) & (cx < self.shape[1]) & (cy >= 0) & (cy < self.shape[0])
        out = np.full(cx.shape, np.nan)
        out[ok] = self.error[cy[ok], cx[ok]]
        return out

    # -- artificial sources (the completeness stage's injection) ------------
    def injection_half_width(self):
        """Half-width in pixels of the square patch an artificial source is added on: the fitting box
        plus one beam (the box of every fit and aperture of a source reaches this far), never beyond the
        point-spread table's support."""
        return int(min(self.table_half, self.half + np.ceil(self.beam_px)))

    def _add_sources(self, x, y, flux, chunk=256, mixture=None):
        """Adds, for each (x, y) in map pixels, `flux` (mJy) times the band's point-spread template to
        the image in place (the mean of the template's scan columns, whose total over the plane is 1,
        evaluated by `CachedPairedPSF.evaluate_xy`, the table the fit's own columns come from). Returns
        the original pixels as `(x0, y0, patch)` triples so `_restore` can put them back."""
        saved = []
        for cx, cy, resp in self._template_patches(x, y, flux, chunk, mixture):
            for i in range(cx.size):
                r = _paste(self.image, int(cx[i]), int(cy[i]), resp[i], keep=True)
                if r is not None:
                    saved.append(r)
        return saved

    def _template_patches(self, x, y, flux, chunk=256, mixture=None):
        """Per chunk of sources: the patches' lower-left pixels (cx, cy) and the template times `flux` on them,
        (n, side, side), side = 2 x `injection_half_width` + 1. `mixture` (n, n_sub): a mosaic's sub-group fractions
        in place of the local ones (a one-hot row injects one sub-group's table alone; checks only)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        flux = np.broadcast_to(np.asarray(flux, float), x.shape)
        h = self.injection_half_width()
        side = 2 * h + 1
        offs = np.arange(side, dtype=float)
        for lo in range(0, x.size, chunk):
            sl = slice(lo, lo + chunk)
            cx = np.rint(x[sl]).astype(np.int64) - h
            cy = np.rint(y[sl]).astype(np.int64) - h
            px = np.broadcast_to(cx[:, None, None] + offs[None, None, :], (cx.size, side, side))
            py = np.broadcast_to(cy[:, None, None] + offs[None, :, None], (cx.size, side, side))
            resp = (self._psf.evaluate_xy(x[sl], y[sl], px, py) if mixture is None
                    else self._psf.evaluate_xy(x[sl], y[sl], px, py, np.asarray(mixture)[sl]))
            yield cx, cy, resp.mean(axis=1) * flux[sl][:, None, None]

    def _restore(self, saved):
        for a0, b0, patch in reversed(saved):          # last added, first undone: patches may overlap
            self.image[b0:b0 + patch.shape[0], a0:a0 + patch.shape[1]] = patch

    @contextmanager
    def injected(self, x, y, flux, mixture=None):
        """Inside the context the tile's image holds artificial point sources of total flux `flux` (mJy,
        scalar or per source) at the map pixels (x, y), each the band's point-spread template; on exit the
        real image is back, pixel for pixel. `mixture`: see `_template_patches`. Every measurement made inside (`measure`, ...)
        sees the sources, because they all read the tile's image."""
        saved = self._add_sources(x, y, flux, mixture=mixture)
        try:
            yield self
        finally:
            self._restore(saved)

    @contextmanager
    def blank_image(self):
        """Inside the context the tile's image is zero everywhere (the error layer is untouched), for
        measuring what the machinery returns for a source on empty sky; restored on exit."""
        real = self.image
        self.image = np.zeros_like(real)
        try:
            yield self
        finally:
            self.image = real

    @contextmanager
    def inverted(self):
        """Inside the context the tile's image is negated (the error layer untouched): a measurement made on
        it is the measurement's response to the map's noise with every real positive source turned into a
        negative one; restored on exit."""
        real = self.image
        self.image = -real
        try:
            yield self
        finally:
            self.image = real


def _paste(image, cx, cy, patch, keep=False):
    """Add `patch` (side x side) into `image` with its lower-left pixel at (cx, cy), clipped to the image; with
    `keep`, returns the original pixels as `(x0, y0, pixels)` for `Tile._restore` (None if nothing overlaps)."""
    ny, nx = image.shape
    side_y, side_x = patch.shape
    a0, b0 = max(cx, 0), max(cy, 0)
    a1, b1 = min(cx + side_x, nx), min(cy + side_y, ny)
    if a1 <= a0 or b1 <= b0:
        return None
    saved = (a0, b0, np.array(image[b0:b1, a0:a1])) if keep else None
    image[b0:b1, a0:a1] += patch[b0 - cy:b1 - cy, a0 - cx:a1 - cx].astype(np.float32)
    return saved


def prepared_dir(config, record):
    """Where `prepare_map` writes the map's prepared layers and state: one directory per map FILE under the run's
    PACS products, named from its path in the download area (`record.path`, unique; the map id is not: one map id
    can name several files, e.g. a level-3 mosaic re-released with more observations)."""
    return os.path.join(config.data_root, "sky", "derived", "pacs", "prepared",
                        os.path.splitext(record.path)[0].replace("/", "__"))


def _load_layers(omap, out_image, out_error):
    """The map's image and error in mJy per pixel into `out_image`, `out_error` (float32, map shape; memory-mapped
    files in `prepare_map`), the error NaN where a pixel is unusable: coverage at most MIN_COVERAGE_FRACTION of the
    median positive coverage, error not above ERROR_FLOOR_FRACTION of the median usable error, or the image not
    finite. The medians are over every 17th value in row order, as one pass over the map would take them; all of
    it strip by strip (PREPARE_STRIP_ROWS)."""
    record = omap.record
    factor = {"jy/pixel": 1e3}.get(record.unit.strip().lower())
    if factor is None:
        raise ValueError("unexpected map unit %r" % record.unit)
    f32 = np.float32(factor)
    ny = record.shape[0]
    strips = [(a, min(a + PREPARE_STRIP_ROWS, ny)) for a in range(0, ny, PREPARE_STRIP_ROWS)]

    def every_17th(select):
        picked, count = [], 0
        for a, b in strips:
            v = select(a, b)
            k = (count + np.arange(v.size)) % 17 == 0
            picked.append(v[k])
            count += v.size
        return np.concatenate(picked) if picked else np.zeros(0)
    positive = every_17th(lambda a, b: (lambda c: c[c > 0])(np.asarray(omap.coverage[a:b])))
    floor = MIN_COVERAGE_FRACTION * (np.median(positive) if positive.size else 0.0)

    def usable_error(a, b):
        err = np.asarray(omap.error[a:b], dtype=np.float32) * f32
        return err[(np.asarray(omap.coverage[a:b]) > floor) & (err > 0)]
    usable = every_17th(usable_error)
    error_floor = ERROR_FLOOR_FRACTION * (np.median(usable) if usable.size else 0.0)
    for a, b in strips:
        image = np.asarray(omap.image[a:b], dtype=np.float32) * f32
        error = np.asarray(omap.error[a:b], dtype=np.float32) * f32
        good = (np.asarray(omap.coverage[a:b]) > floor) & (error > 0)
        bad = ~good | ~(error > error_floor) | ~np.isfinite(image)
        error[bad] = np.nan
        out_image[a:b] = image
        out_error[a:b] = error


def _is_mosaic(record):
    return bool(getattr(record, "mosaic_dir", ""))


#: a mosaic's usable-pixel floors are taken from about this many sampled pixels (whole rows at a regular stride), so
#: that they do not cost a pass over a mosaic of any size
MOSAIC_FLOOR_SAMPLE = 8_000_000


def mosaic_layer_plan(config, record):
    """The first step of a mosaic's `prepare_map`: creates the prepared `image.npy` and `error.npy` (memory-mapped, to be
    filled by `mosaic_layer_strip`) and returns `(coverage floor, error fraction, strips)`: the coverage floor as
    `_load_layers` defines it (a fraction of the median positive coverage) but taken from whole rows at a regular stride
    (MOSAIC_FLOOR_SAMPLE pixels); the error fraction ERROR_FLOOR_FRACTION, which `mosaic_layer_strip` applies to each
    pixel's own expected error (its weight layer), not to a mosaic-wide median, so that the deepest overlaps are kept; and
    the strips `(first row, last row + 1)` of PREPARE_STRIP_ROWS rows. The mosaic is NOT copied serially: the strips are
    independent, any number run at once."""
    from numpy.lib.format import open_memmap
    d = prepared_dir(config, record)
    os.makedirs(d, exist_ok=True)
    shape = tuple(record.shape)
    for name in ("image", "error"):
        open_memmap(os.path.join(d, name + ".npy"), mode="w+", dtype=np.float32, shape=shape).flush()
    ny, nx = shape
    stride = max(1, int(ny * nx // MOSAIC_FLOOR_SAMPLE))
    with open_map(config, record) as omap:
        rows = np.arange(0, ny, stride)
        cov = np.concatenate([np.asarray(omap.coverage[a:a + 1]).ravel() for a in rows])
    positive = cov[cov > 0]
    floor = MIN_COVERAGE_FRACTION * (np.median(positive) if positive.size else 0.0)
    return float(floor), ERROR_FLOOR_FRACTION, [(a, min(a + PREPARE_STRIP_ROWS, ny)) for a in range(0, ny, PREPARE_STRIP_ROWS)]


def mosaic_layer_strip(config, record, a, b, floor, error_fraction):
    """Rows a..b of a mosaic's prepared layers (`mosaic_layer_plan`), the rule of `_load_layers` (mJy, error NaN where
    unusable) with the error floor local: a pixel is unusable where its error is not above `error_fraction` of the error
    its own weight expects, 1 / sqrt(weight) (the mosaic's weight layer, the error layer's units)."""
    from numpy.lib.format import open_memmap
    d = prepared_dir(config, record)
    f32 = np.float32(1e3)
    with open_map(config, record) as omap:
        image = np.asarray(omap.image[a:b], dtype=np.float32) * f32
        error = np.asarray(omap.error[a:b], dtype=np.float32) * f32
        good = (np.asarray(omap.coverage[a:b]) > floor) & (error > 0)
    weight = np.load(os.path.join(record.mosaic_dir, "weight.npy"), mmap_mode="r")
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = f32 / np.sqrt(np.asarray(weight[a:b], dtype=np.float32))
    error[~good | ~(error > error_fraction * expected) | ~np.isfinite(image)] = np.nan
    for name, v in (("image", image), ("error", error)):
        m = open_memmap(os.path.join(d, name + ".npy"), mode="r+")
        m[a:b] = v
        m.flush()
        del m


def finish_prepare(config, record):
    """The last step of `prepare_map`: the tile's point-spread cache, normalisation and neighbour list as `tile.pkl`
    (the layers are already written). Returns the prepared `Tile` (as `open_prepared`)."""
    d = prepared_dir(config, record)
    with open_map(config, record) as omap:
        wcs = omap.wcs
    tile = Tile(record, wcs, config, np.load(os.path.join(d, "image.npy"), mmap_mode="r"),
                np.load(os.path.join(d, "error.npy"), mmap_mode="r"))
    with open(os.path.join(d, "tile.pkl"), "wb") as fh:
        pickle.dump(dict(psf=tile._psf, psf_scale=tile.psf_scale, neighbours=tile.neighbours), fh)
    return open_prepared(config, record)


def prepare_map(config, record):
    """Prepare `record`'s map once, for every worker of the run: its layers (`_load_layers`) as `image.npy` and
    `error.npy`, and its point-spread cache, normalisation and bright-neighbour list as `tile.pkl`, in
    `prepared_dir`. Every full-map step works strip by strip, so the memory is the same for any map size. A mosaic is
    prepared by `mosaic_layer_plan`, `mosaic_layer_strip` and `finish_prepare` (the completeness stage runs the strips in
    parallel); here they run in this process. Returns the prepared `Tile` (as `open_prepared`)."""
    from numpy.lib.format import open_memmap
    if _is_mosaic(record):
        floor, error_fraction, strips = mosaic_layer_plan(config, record)
        for a, b in strips:
            mosaic_layer_strip(config, record, a, b, floor, error_fraction)
        return finish_prepare(config, record)
    d = prepared_dir(config, record)
    os.makedirs(d, exist_ok=True)
    shape = tuple(record.shape)
    image = open_memmap(os.path.join(d, "image.npy"), mode="w+", dtype=np.float32, shape=shape)
    error = open_memmap(os.path.join(d, "error.npy"), mode="w+", dtype=np.float32, shape=shape)
    with open_map(config, record) as omap:
        _load_layers(omap, image, error)
    image.flush()
    error.flush()
    del image, error
    return finish_prepare(config, record)


def open_prepared(config, record):
    """The `Tile` of `record` from its prepared files (`prepare_map`): the layers memory-mapped (shared by every
    worker; the image read-only: injections work on private buffers of rows, `Tile.writable` is the one copy-on-write exception), the point-spread
    cache and neighbour list unpickled. Raises FileNotFoundError when the map was not prepared."""
    d = prepared_dir(config, record)
    with open(os.path.join(d, "tile.pkl"), "rb") as fh:
        state = pickle.load(fh)
    path = os.path.join(d, "image.npy")
    wcs = WCS(fits.Header.fromstring(record.wcs_header)).celestial
    tile = Tile(record, wcs, config, np.load(path, mmap_mode="r"), np.load(os.path.join(d, "error.npy"), mmap_mode="r"),
                psf=state["psf"], neighbours=state["neighbours"], image_path=path)
    tile.psf_scale = state["psf_scale"]
    return tile


@contextmanager
def open_tile(config, record):
    """The prepared `Tile` of `record` (`open_prepared`)."""
    yield open_prepared(config, record)


_CACHE = {}


def cached_tile(config, record):
    """The process-local prepared Tile for `record` (`open_prepared`), kept (one tile at a time) so the pieces of
    a tile share it."""
    key = record.path
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = open_prepared(config, record)
    return _CACHE[key]


# ----------------------------------------------------------------------------
# across tiles and the catalogue

def select_deeper(tile_index, position_indices, local_rms, covered, selected_tile, selected_rms):
    """Deepest-map choice across overlapping maps: update `selected_tile` (int32, init -1)
    and `selected_rms` (float64, init +inf) in place at the unique `position_indices`. The
    lowest positive finite rms among covered candidates wins; equal rms goes to the smaller
    tile ordinal, whatever the processing order."""
    idx = np.asarray(position_indices)
    rms = np.asarray(local_rms, dtype=np.float64)
    replace = np.asarray(covered, dtype=bool) & np.isfinite(rms) & (rms > 0)
    old_t, old_r = selected_tile[idx], selected_rms[idx]
    replace &= (rms < old_r) | ((rms == old_r) & (tile_index < old_t))
    selected_tile[idx[replace]] = tile_index
    selected_rms[idx[replace]] = rms[replace]


def admitted_sample_points(admitted):
    """ICRS RA/Dec (n_pixel, 256) of the 256 nested children at nside 8192 of each admitted
    nside-512 Galactic NESTED pixel."""
    admitted = np.asarray(admitted)
    children = admitted.astype(np.int64)[:, None] * 256 + np.arange(256)[None, :]
    lon, lat = hp.pix2ang(8192, children, nest=True, lonlat=True)
    icrs = SkyCoord(l=lon, b=lat, unit="deg", frame="galactic").icrs
    return icrs.ra.deg, icrs.dec.deg


@dataclass(frozen=True)
class SourceCandidates:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    source_row: np.ndarray
    catalogue: np.ndarray


def load_sesna_candidates(config, centre_ra_deg, centre_dec_deg, radius_deg, chunk_rows=100000):
    """Every curated SESNA source (all regions' catalogues, so neighbours across region edges
    are found) inside a sky cap, as (SourceCandidates, audit dict)."""
    centre = np.array([np.cos(np.radians(centre_dec_deg)) * np.cos(np.radians(centre_ra_deg)),
                       np.cos(np.radians(centre_dec_deg)) * np.sin(np.radians(centre_ra_deg)),
                       np.sin(np.radians(centre_dec_deg))])
    chord = 2 * np.sin(np.radians(radius_deg) / 2)
    from sesnaimpute import regions as regions_module
    from sesnaimpute.catalog import curated
    ras, decs, rows, names = [], [], [], []
    n_read = 0
    for reg in regions_module.REGIONS:
        # the curated rows' positions from the delivery (catalog.curated.read_positions), not from the curated product,
        # which is built after the PACS stages
        _name, ra, dec = curated.read_positions(config, reg.name)
        fname = "sources_sesna_source__%s.hdf5" % reg.name.replace(" ", "_")
        n_read += ra.size
        for start in range(0, ra.size, chunk_rows):
            r, d = ra[start:start + chunk_rows], dec[start:start + chunk_rows]
            v = np.c_[np.cos(np.radians(d)) * np.cos(np.radians(r)),
                      np.cos(np.radians(d)) * np.sin(np.radians(r)), np.sin(np.radians(d))]
            keep = np.linalg.norm(v - centre, axis=1) <= chord
            ras.append(r[keep])
            decs.append(d[keep])
            rows.append(np.arange(start, start + r.size, dtype=np.int64)[keep])
            names.append(np.full(int(keep.sum()), fname, dtype=object))
    cat = lambda parts, empty: np.concatenate(parts) if parts else empty
    out = SourceCandidates(cat(ras, np.empty(0)), cat(decs, np.empty(0)),
                           cat(rows, np.empty(0, dtype=np.int64)), cat(names, np.empty(0, dtype=object)))
    return out, {"catalogues": len(regions_module.REGIONS), "rows_read": n_read, "rows_retained": out.ra_deg.size}


# ----------------------------------------------------------------------------
# the mosaic: its point-spread function and its map record

def _bspline_box_kernel(u):
    """Bilinear interpolation (triangle, half-width 1) convolved with the unit input-pixel box: the quadratic
    B-spline, support +-1.5, unit integral."""
    a = np.abs(u)
    return np.where(a <= 0.5, 0.75 - a ** 2, np.where(a <= 1.5, 0.5 * (1.5 - a) ** 2, 0.0))


def _kernel_weights(jac, steps, broad):
    """Weights on the fine output-pixel lattice (step 1/steps) of the input pixel's footprint: `broad` = box +
    bilinear triangle (the resampled map's response to a delta), else the box alone. `jac`: d(input pixel)/d(output
    pixel). Sum 1."""
    inv = np.linalg.inv(jac)
    half = int(np.ceil(1.5 * np.abs(inv).sum(axis=1).max())) + 1
    o = np.arange(-half * steps, half * steps + 1) / steps
    yy, xx = np.meshgrid(o, o, indexing="ij")
    u = jac[0, 0] * xx + jac[0, 1] * yy
    v = jac[1, 0] * xx + jac[1, 1] * yy
    if broad:
        w = _bspline_box_kernel(u) * _bspline_box_kernel(v)
    else:
        delta = np.abs(jac).sum(axis=1) / steps / 2.0                 # anti-aliased box edge, one fine step wide
        w = (np.clip((0.5 - np.abs(u)) / delta[0] + 0.5, 0, 1) * np.clip((0.5 - np.abs(v)) / delta[1] + 0.5, 0, 1))
    return w / w.sum(), half


def _template_density(components, east, north):
    """Mean over the scan components of the template's flux per square arcsec at these sky offsets (arcsec;
    east, north); zero beyond the released +-100 arcsec."""
    from scipy.ndimage import map_coordinates
    out = 0.0
    for c in components:
        a = np.deg2rad(c.position_angle_deg)
        east0 = east * np.cos(a) - north * np.sin(a)
        north0 = east * np.sin(a) + north * np.cos(a)
        out = out + map_coordinates(c.image, [100.0 + north0, 100.0 - east0], order=1, mode="constant", cval=0.0,
                                    prefilter=False)
    return out / len(components)


def _node_table(components, out_wcs, node, jac, extent, steps, out_area, broad):
    """One spatial node's response, (count, count): the unit-flux template (mean of the scan columns) on the
    north-up output grid around output pixel `node` = (x, y), convolved with the input footprint (`_kernel_weights`),
    sampled at output-pixel offsets -extent..extent in 1/steps; per output pixel (flux), at the node's own output
    pixel area (`out_area` is the reference pixel's; `pacs_mosaic.tan_area_factor`), as the mosaic's pixels hold it."""
    from scipy.signal import fftconvolve
    weights, half = _kernel_weights(jac, steps, broad)
    count = int(2 * extent * steps) + 1
    pad = half
    big = count + 2 * pad * steps
    o = (np.arange(big) - (big - 1) / 2) / steps
    yy, xx = np.meshgrid(o, o, indexing="ij")
    source = out_wcs.pixel_to_world(np.array([node[0]]), np.array([node[1]]))
    east, north = source.spherical_offsets_to(out_wcs.pixel_to_world(xx + node[0], yy + node[1]))
    density = _template_density(components, east.arcsec, north.arcsec)
    conv = fftconvolve(density, weights, mode="same")
    from sesnaimpute.sky.derived.pacs_mosaic import tan_area_factor
    return conv[pad * steps:pad * steps + count, pad * steps:pad * steps + count] * (out_area * float(tan_area_factor(out_wcs, *node)))


def _table_aperture(table, extent, steps, band, pixel_arcsec):
    """The aperture sums of a unit-flux source of a one-node table `table` (count, count), as `Tile` takes them:
    the mean over a 3 x 3 sub-pixel lattice of the aperture sum with the sky annulus's own scatter-free clipped
    sky (`_aperture_core` on the noise-free image). Returns (enclosed: aperture sum before sky,
    aperture_fraction: after sky)."""
    from scipy.ndimage import map_coordinates
    r_ap, r_in, r_out = (r / pixel_arcsec for r in APERTURES[band])
    w = int(np.ceil(r_out)) + 3
    offs = np.arange(-w, w + 1, dtype=float)
    sub = np.arange(3) / 3.0
    enclosed, fraction = [], []
    for sy in sub:
        for sx in sub:
            yy, xx = np.meshgrid(offs - sy, offs - sx, indexing="ij")
            img = map_coordinates(table, [(yy + extent) * steps, (xx + extent) * steps], order=1, mode="nearest",
                                  prefilter=False).astype(np.float32)
            f, s, _sd, area, cov = _aperture_core(img, np.ones_like(img), np.array([w + sx]), np.array([w + sy]),
                                                  r_ap, r_in, r_out, 5)
            if not cov.all():
                raise ValueError("the aperture of the mosaic table is not covered")
            enclosed.append(float((f + s * area)[0]))
            fraction.append(float(f[0]))
    return float(np.mean(enclosed)), float(np.mean(fraction))


def _table_integral(table, extent, steps, band, pixel_arcsec):
    """The flux fraction of a one-node table inside the aperture radius, integrated on the table's fine lattice (cells
    of 1/steps output pixel, anti-aliased edge): the continuous counterpart of `_table_aperture`'s enclosed sum, which
    samples the table at output pixel centres (a circle of 4 pixels radius on a 3.2 arcsec grid: +-0.2 %)."""
    r_ap = APERTURES[band][0] / pixel_arcsec
    o = np.arange(table.shape[0]) / steps - extent
    xx, yy = np.meshgrid(o, o)
    w = np.clip((r_ap - np.hypot(xx, yy)) * steps + 0.5, 0.0, 1.0)
    return float((table * w).sum() / steps ** 2)


def _jacobian(out_wcs, in_wcs, x, y):
    """d(input pixel)/d(output pixel) at output pixel (x, y), by central differences through the sky."""
    cols = []
    for dx, dy in ((1.0, 0.0), (0.0, 1.0)):
        ra, dec = out_wcs.all_pix2world([x - dx / 2, x + dx / 2], [y - dy / 2, y + dy / 2], 0)
        px, py = in_wcs.all_world2pix(ra, dec, 0)
        cols.append([px[1] - px[0], py[1] - py[0]])
    return np.array(cols).T


def _mosaic_node(components, out_wcs, in_wcs, node, extent, steps, out_area, band, pix, path, index, centre):
    """One (sub-group, spatial node) of `build_mosaic_psf`, a worker's unit: the node's response table written
    (unscaled, float32) into node `index` of the sub-group's table file `path`, and its aperture sums. Returns
    (enclosed, aperture fraction) of the broad table and, at the centre node, the plain table's enclosed sum (the
    normalisation) and the broad-over-plain total."""
    jac = _jacobian(out_wcs, in_wcs, node[0], node[1])
    broad = _node_table(components, out_wcs, node, jac, extent, steps, out_area, True)
    e_broad, a_broad = _table_aperture(broad, extent, steps, band, pix)
    e_plain = total = None
    if centre:
        plain = _node_table(components, out_wcs, node, jac, extent, steps, out_area, False)
        e_plain = _table_integral(plain, extent, steps, band, pix)       # continuous: the release EEF is a continuous integral
        total = float(broad.sum() / plain.sum())
    table = np.load(path, mmap_mode="r+")
    table[index // table.shape[1], index % table.shape[1], 0] = broad
    table.flush()
    return e_broad, a_broad, e_plain, total


def build_mosaic_psf(config, region, band, box=None):
    """Writes `<mosaic>/_psf/` for the built mosaic (`pacs_mosaic.build_mosaic`): psf.json, one response table
    per sub-group (`MosaicPSF`) and the weight-mean table `table.npy`. Sub-groups: the mosaic's PSF groups split
    by input pixel scale and axis orientation (the kernel depends on both). Tables are float32, 3 x 3 spatial
    nodes (PSF_SPATIAL_NODES) over the mosaic, PSF_SUBPIXEL_STEPS per pixel, the half-width `Tile` uses. The
    (sub-group, node) tables are built in parallel (config.n_jobs); the enclosed and aperture fractions are
    measured on every node's table, the sub-groups' beams stored for `MosaicPSF.beam_at`."""
    import json
    from joblib import Parallel, delayed
    from numpy.lib.format import open_memmap
    from sesnaimpute.sky.derived import pacs_mosaic
    mdir = pacs_mosaic.product_dir(config, region, band, box)
    pacs_mosaic._assert_inside_scratch_derived(config, mdir)
    m = pacs_mosaic.Mosaic(mdir)
    recs = {r.path: r for r in read_index(config, region)}
    out_wcs = m.wcs.celestial
    ny, nx = m.shape
    pix = float(np.mean(proj_plane_pixel_scales(out_wcs))) * 3600.0       # as Tile takes it (sets the table size)
    out_area = pacs_mosaic.pixel_area_arcsec2(out_wcs)
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    keys, entries = [], []
    for q in m.meta["maps"]:
        rec = recs[q["path"]]
        in_wcs = pacs_mosaic._celestial(rec)
        jac = _jacobian(out_wcs, in_wcs, cx, cy)
        angle = float(np.degrees(np.arctan2(jac[1, 0], jac[0, 0])) % 90.0)
        key = (q["group"], round(rec.pixel_scale_arcsec, 3), round(angle * 4) / 4 % 90.0)
        if key not in keys:
            keys.append(key)
        bbox = pacs_mosaic._bbox_in_output(rec, in_wcs, out_wcs, m.shape) or (0, 0, 0, 0)
        entries.append(dict(path=q["path"], wcs_header=rec.wcs_header, shape=list(rec.shape), lattice=q["lattice"],
                            lattice_step=q["lattice_step"], scale=pacs_mosaic.pixel_area_arcsec2(out_wcs) / pacs_mosaic.pixel_area_arcsec2(in_wcs),
                            sub=keys.index(key), bbox=[int(v) for v in bbox],
                            hdus=[rec.image_hdu, rec.error_hdu, rec.coverage_hdu]))
    n_sub = len(keys)
    rep = [next(r for r, e in zip(m.meta["maps"], entries) if e["sub"] == k) for k in range(n_sub)]
    rep_rec = [recs[q["path"]] for q in rep]
    beams = np.array([beam_fwhm_arcsec(r) for r in rep_rec])
    # stage 1: the mixture fractions over the mosaic (a regular sample of its pixels) -> mean beam and table size
    pdir = os.path.join(mdir, "_psf")
    os.makedirs(pdir, exist_ok=True)
    meta = dict(region=region, rep=[entries.index(next(e for e in entries if e["sub"] == k)) for k in range(n_sub)],
                shape=[ny, nx], n_sub=n_sub, keys=[list(k) for k in keys], maps=entries, out_header=m.wcs.to_header(relax=True).tostring(sep=""),
                extent=0.0, subpixel_steps=PSF_SUBPIXEL_STEPS, integration_order=8, fbar=[1.0 / n_sub] * n_sub,
                enclosed_sub=[1.0] * n_sub, aperture_fraction_sub=[1.0] * n_sub, sub_beams=[float(b) for b in beams])
    with open(os.path.join(pdir, "psf.json"), "w") as fh:
        json.dump(meta, fh)
    probe = MosaicPSF(mdir, maps_root(config), tables=False)
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 40), np.linspace(0, ny - 1, 40))
    f = probe.weights(gx.ravel(), gy.ravel())
    valid = np.isfinite(np.asarray(m.layers["image"])[np.rint(gy.ravel()).astype(int), np.rint(gx.ravel()).astype(int)])
    fbar = f[valid].mean(axis=0) if valid.any() else f.mean(axis=0)
    beam = float(fbar @ beams)
    beam_px = beam / pix
    half = int(np.ceil(max(FIT_BOX_FWHM * beam_px, APERTURES[band][2] / pix + 1.0)))
    table_half = int(min(half + np.ceil(NEIGHBOUR_REACH_FWHM * beam_px), PSF_TABLE_HALF_ARCSEC // pix))
    extent = table_half + 0.5
    steps = PSF_SUBPIXEL_STEPS
    count = int(2 * extent * steps) + 1
    n_nodes = max(PSF_SPATIAL_NODES, int(np.ceil(max(nx, ny) * pix / 3600.0 / MOSAIC_PSF_NODE_SPACING_DEG)) + 1)
    n_nodes += 1 - n_nodes % 2              # odd: the normalisation is taken at the centre node (`centre` below)
    nodes = [(sx, sy) for sy in np.linspace(-.5, ny - .5, n_nodes) for sx in np.linspace(-.5, nx - .5, n_nodes)]
    mean_table = np.zeros((n_nodes, n_nodes, 1, count, count))
    paths = [os.path.join(pdir, f"sub{k}.npy") for k in range(n_sub)]
    comps = []
    for k in range(n_sub):
        pacs_mosaic._assert_inside_scratch_derived(config, paths[k])
        open_memmap(paths[k], mode="w+", dtype=np.float32, shape=(n_nodes, n_nodes, 1, count, count)).flush()
        comps.append(published_components_for_map(config, rep_rec[k]))
    in_wcs = [pacs_mosaic._celestial(r) for r in rep_rec]
    centre = (n_nodes ** 2) // 2
    done = Parallel(n_jobs=config.n_jobs)(
        delayed(_mosaic_node)(comps[k], out_wcs, in_wcs[k], node, extent, steps, out_area, band, pix, paths[k], i, i == centre)
        for k in range(n_sub) for i, node in enumerate(nodes))
    enclosed = np.zeros((n_sub, n_nodes, n_nodes))
    aperture = np.zeros((n_sub, n_nodes, n_nodes))
    totals = []
    for k in range(n_sub):
        res = done[k * n_nodes ** 2:(k + 1) * n_nodes ** 2]
        scale = aperture_eef(config, rep_rec[k]) / res[centre][2]            # Tile's normalisation of an ordinary table
        totals.append(res[centre][3])
        enclosed[k] = np.array([r[0] for r in res]).reshape(n_nodes, n_nodes) * scale
        aperture[k] = np.array([r[1] for r in res]).reshape(n_nodes, n_nodes) * scale
        table = np.load(paths[k], mmap_mode="r+")
        table *= np.float32(scale)
        table.flush()
        mean_table += fbar[k] * table
        del table
    np.save(os.path.join(pdir, "table.npy"), mean_table.astype(np.float32))
    ctr = (n_nodes // 2, n_nodes // 2)
    meta.update(extent=extent, fbar=[float(v) for v in fbar], enclosed_sub=enclosed.tolist(), aperture_fraction_sub=aperture.tolist(),
                beam_arcsec=beam, enclosed_fraction=float(fbar @ enclosed[(slice(None), *ctr)]),
                aperture_fraction=float(fbar @ aperture[(slice(None), *ctr)]), broad_over_plain_total=totals, table_half=table_half)
    with open(os.path.join(pdir, "psf.json"), "w") as fh:
        json.dump(meta, fh)
    return MosaicPSF(mdir, maps_root(config))


def mosaic_psf(config, region, band, box=None):
    """The built mosaic's `MosaicPSF` without its tables (weights, fractions and beams only): `beam_at_sky` is the
    local beam at a sky position."""
    from sesnaimpute.sky.derived import pacs_mosaic
    return MosaicPSF(pacs_mosaic.product_dir(config, region, band, box), maps_root(config), tables=False)


def mosaic_record(config, region, band, box=None, rebuild_psf=False):
    """The `MosaicRecord` of the band's built mosaic of the region (builds its PSF when `_psf/` is absent or
    `rebuild_psf`). `path` = the product folder under the data root + "/mosaic"."""
    from sesnaimpute.sky.derived import pacs_mosaic
    mdir = pacs_mosaic.product_dir(config, region, band, box)
    m = pacs_mosaic.Mosaic(mdir)
    if rebuild_psf or not os.path.exists(os.path.join(mdir, "_psf", "psf.json")):
        build_mosaic_psf(config, region, band, box)
    import h5py
    with h5py.File(os.path.join(mdir, "mosaic.hdf5"), "r") as f:
        header = f["WCS_HEADER"][()].decode()
    paths = {q["path"] for q in m.meta["maps"]}
    obsids = tuple(sorted({str(o) for r in read_index(config, region) if r.path in paths for o in r.obsids}))
    return MosaicRecord(map_id="mosaic_%s_%s" % (region.replace(" ", "_"), band), obsids=obsids, band=band,
                        path=os.path.relpath(mdir, config.data_root) + "/mosaic", image_hdu=0, error_hdu=1, coverage_hdu=2,
                        shape=tuple(m.shape), unit="Jy/pixel", wcs_header=header, mode=1, level="L25", mapper="mosaic",
                        scan_speed=20.0, pixel_scale_arcsec=pacs_mosaic.PIXEL_ARCSEC[band], source="mosaic",
                        polygon=(), mosaic_dir=mdir)
