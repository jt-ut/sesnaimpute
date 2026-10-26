"""The one PACS measurement code path, shared by `pacs_completeness` and
`pacs_photometry` (SPEC_PACS_FARIR_DRAFT.md sections 4, 5, 7b and 11.3).

For ONE map tile (one PACS map, one band):

  with open_tile(config, record) as tile:
      tile.xy(ra, dec)                   sky -> map pixel, one vectorised WCS call
      tile.neighbours                    blind bright sources (11.3 step 1)
      tile.fit_positions(x, y)           forced fit, nuisance columns included
      tile.fit_groups(x, y, group)       simultaneous fit of blend groups
      tile.fitted_fwhm(x, y)             width of a detection (EXTENDED flag)
      tile.grid                          the 30-arcsec cell grid on this map

The fit at a position is a weighted linear least squares of two scan-direction
PSF columns for the target (flux = the sum of their two amplitudes), two
columns for every bright neighbour whose PSF reaches the fitting box (a
nuisance, dropped where it lies within one beam of a target because that
target's own column then carries it) and one constant background, on a square
box of half-width FIT_BOX_FWHM x beam FWHM, weighted by the map's error layer.
The sigma it returns is the PROPAGATED PIXEL NOISE only (the error layer carried
through the linear operator). The cell sigma that sets detection is measured by
the completeness stage from the scatter of these same fits at random empty-sky
positions, never added to this term.

Because the estimator is linear, an injected beam of flux F raises the fitted
flux by exactly gain x F; `gain` is the response to a 50/50 mix of the two
scan columns (the fit sums the two amplitudes, so the mix only enters the gain).

All fluxes are mJy. Pixel coordinates are zero-origin map pixels (x along the
fast axis). A tile holds native-byte-order float32 copies of the image and error
layers (the FITS data are big-endian and numba needs native order); the file
itself is opened read-only and memory-mapped.
"""

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

# TEMPORARY (unit DOWNLOAD and unit PSF have not landed): the old package supplies
# the map record, the index reader and the PSF. Replace these two lines by
#   from sesnaimpute.sky.download.herschel_pacs.index import MapRecord, read_index, maps_root
#   from sesnaimpute.sky.derived.pacs_psf import (CachedPairedPSF, published_components_for_map,
#                                                 beam_fwhm_arcsec)
from sesnaimpute.farIRcomp.maps import MapRecord, read_index  # noqa: F401
from sesnaimpute.farIRcomp.psf import CachedPairedPSF, published_components_for_map

#: detection threshold in sigma (owner 2026-10-24); defines recovery in the completeness
DETECTION_SNR = 5.0
#: completeness cell side, arcsec (owner 2026-10-24: SESNA's own)
CELL_ARCSEC = 30.0
#: fitting-box half-width in beam FWHM (spec 11.3 step 1)
FIT_BOX_FWHM = 3.0
#: a pixel is also dropped if its error is below this fraction of the map's median error
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
#: a fit is trusted only if this fraction of the target beam lies on valid pixels
MIN_BEAM_FRACTION = 0.9
#: beam FWHM, arcsec, used only until unit PSF's `beam_fwhm_arcsec` lands:
#: PICC-ME-TN-033 v2.2 Table 5 widest P70 parallel-60 scan major axis; P100 by ratio of wavelengths
_FALLBACK_BEAM = {"P70": 12.74, "P100": 12.74 * 100.0 / 70.0}


def beam_fwhm_arcsec(record):
    """Beam FWHM of a map's band and mode, arcsec (unit PSF's function once it lands)."""
    return _FALLBACK_BEAM[record.band]


def maps_root(config):
    """Directory of the downloaded PACS maps: config [inputs] herschel_pacs_maps, no fallback."""
    return Path(config.inputs["herschel_pacs_maps"])


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

    flux       (n,) mJy, summed amplitude of the two scan columns
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
class GroupFit:
    """Simultaneous fits of blend groups, one row per member (input order).

    flux        (n,) mJy per member
    sigma       (n,) propagated pixel noise on each member's flux in the group fit
    inflation   (n,) sigma / (propagated sigma of a singleton fit at the same
                position): the correlation inflation photometry applies to the cell sigma
    max_abs_rho (n,) largest |correlation| of the member's fitted flux with any other
                member of its group (1.0 when the group fit is singular)
    group_flux  (n,) sum of the group's member fluxes (same on every member)
    group_sigma (n,) propagated pixel noise of that sum (full covariance)
    covered     (n,) bool
    """
    flux: np.ndarray
    sigma: np.ndarray
    inflation: np.ndarray
    max_abs_rho: np.ndarray
    group_flux: np.ndarray
    group_sigma: np.ndarray
    covered: np.ndarray


# ----------------------------------------------------------------------------
# the numerical core

@numba.njit(cache=True)
def _box_templates(table, extent, steps, map_nx, map_ny, sx, sy, x0, y0, side, out, row):
    """Pixel response of the two scan columns of a source at (sx, sy) on the box whose
    lower-left pixel is (x0, y0): nearest spatial node of the response table, bilinear
    in pixel displacement. Writes out[row], out[row+1]; zero beyond the table."""
    nsy, nsx = table.shape[0], table.shape[1]
    last = table.shape[-1] - 1
    ix = min(max(int(round((sx + 0.5) / map_nx * (nsx - 1))), 0), nsx - 1)
    iy = min(max(int(round((sy + 0.5) / map_ny * (nsy - 1))), 0), nsy - 1)
    for j in range(side):
        dy = (y0 + j - sy + extent) * steps
        for i in range(side):
            p = j * side + i
            dx = (x0 + i - sx + extent) * steps
            if dx < 0 or dx > last or dy < 0 or dy > last:
                out[row, p] = 0.0
                out[row + 1, p] = 0.0
                continue
            jx = min(int(dx), last - 1)
            jy = min(int(dy), last - 1)
            ux = dx - jx
            uy = dy - jy
            for m in range(2):
                im = table[iy, ix, m]
                out[row + m, p] = ((1 - ux) * (1 - uy) * im[jy, jx] + ux * (1 - uy) * im[jy, jx + 1]
                                   + ux * uy * im[jy + 1, jx + 1] + (1 - ux) * uy * im[jy + 1, jx])


@numba.njit(cache=True)
def _fit_core(img, err, table, extent, steps, tx, ty, tn, ux, uy, un, cx, cy, half, min_beam):
    """Weighted linear fits, one per row. Row i: tn[i] target sources at (tx, ty), un[i]
    nuisance sources at (ux, uy), box centred on pixel (cx[i], cy[i]) of half-width `half`.
    Returns per-member flux, gain, covariance and a validity flag."""
    n, mmax = tx.shape
    side = 2 * half + 1
    npix = side * side
    map_ny, map_nx = img.shape
    kmax = ux.shape[1]
    cmax = 2 * (mmax + kmax) + 1
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
        nc = 2 * (mt + kn) + 1
        x0 = cx[i] - half
        y0 = cy[i] - half
        for m in range(mt):
            _box_templates(table, extent, steps, map_nx, map_ny, tx[i, m], ty[i, m], x0, y0, side, X, 2 * m)
        for k in range(kn):
            _box_templates(table, extent, steps, map_nx, map_ny, ux[i, k], uy[i, k], x0, y0, side, X,
                           2 * (mt + k))
        for p in range(npix):
            X[nc - 1, p] = 1.0
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
                t = X[2 * m, p] + X[2 * m + 1, p]
                tot += t
                if ok[p]:
                    got += t
            if tot <= 0 or got < min_beam * tot:
                enough = False
        if not enough:
            continue
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
            B[2 * m, m] = 1.0 / scale[2 * m]
            B[2 * m + 1, m] = 1.0 / scale[2 * m + 1]
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
                g += L[m, p] * 0.5 * (X[2 * m, p] + X[2 * m + 1, p])
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
    path = maps_root(config) / record.path
    if path.suffix.lower() in (".gz", ".zip"):
        raise ValueError("compressed maps must be unpacked before memory-mapped access")
    with fits.open(path, mode="readonly", memmap=True) as hdus:
        header = fits.Header.fromstring(record.wcs_header)
        arrays = [hdus[i].data for i in (record.image_hdu, record.error_hdu, record.coverage_hdu)]
        if any(a is None or a.shape != tuple(record.shape) for a in arrays):
            raise ValueError("inconsistent map layers: %s" % path)
        for a in arrays:
            a.flags.writeable = False
        yield OpenMap(*arrays, WCS(header).celestial, record)


# ----------------------------------------------------------------------------
# the tile

class Tile:
    """One open map tile with its fitting machinery (build with `open_tile`)."""

    def __init__(self, omap, config):
        record = omap.record
        self.record = record
        self.wcs = omap.wcs
        self.shape = tuple(record.shape)
        scale = proj_plane_pixel_scales(omap.wcs)
        self.pixel_arcsec = float(np.mean(scale)) * 3600.0
        self.beam_arcsec = beam_fwhm_arcsec(record)
        self.beam_px = self.beam_arcsec / self.pixel_arcsec
        self.half = int(np.ceil(FIT_BOX_FWHM * self.beam_px))
        self.table_half = int(min(self.half + np.ceil(NEIGHBOUR_REACH_FWHM * self.beam_px),
                                  PSF_TABLE_HALF_ARCSEC // self.pixel_arcsec))
        self.reach = float(self.table_half)            # nuisance inclusion radius, pixels
        self.grid = MapGrid.from_wcs(omap.wcs, self.shape)
        # unit conversion to mJy; image and error layers share the unit
        factor = {"jy/pixel": 1e3}.get(record.unit.strip().lower())
        if factor is None:
            raise ValueError("unexpected map unit %r" % record.unit)
        cover = np.asarray(omap.coverage)
        positive = cover[cover > 0]
        floor = MIN_COVERAGE_FRACTION * (np.median(positive[::17]) if positive.size else 0.0)
        image = np.asarray(omap.image, dtype=np.float32) * np.float32(factor)
        error = np.asarray(omap.error, dtype=np.float32) * np.float32(factor)
        usable = (cover > floor) & (error > 0)
        error_floor = ERROR_FLOOR_FRACTION * (np.median(error[usable][::17]) if usable.any() else 0.0)
        bad = ~usable | ~(error > error_floor) | ~np.isfinite(image)
        error[bad] = np.nan
        self.image, self.error = image, error
        components = published_components_for_map(config, record)
        self._psf = CachedPairedPSF.from_map(
            omap.wcs, self.shape, components, half_size=self.table_half, integration_order=8,
            spatial_nodes=PSF_SPATIAL_NODES, subpixel_steps=PSF_SUBPIXEL_STEPS)
        self._tree = None
        self.neighbours = self._find_neighbours()
        if self.neighbours.x.size:
            self._tree = cKDTree(np.c_[self.neighbours.x, self.neighbours.y])

    # -- geometry ---------------------------------------------------------
    def xy(self, ra_deg, dec_deg):
        """ICRS degrees -> zero-origin map pixels (x, y); one vectorised WCS call."""
        x, y = self.wcs.all_world2pix(np.asarray(ra_deg, float), np.asarray(dec_deg, float), 0)
        return np.asarray(x), np.asarray(y)

    def _call(self, tx, ty, tn, ux, uy, un, cx, cy, half):
        psf = self._psf
        return _fit_core(self.image, self.error, psf.table, psf.extent, psf.subpixel_steps,
                         tx, ty, tn, ux, uy, un, cx, cy, half, MIN_BEAM_FRACTION)

    # -- bright neighbours (11.3 step 1) -------------------------------------
    def _find_neighbours(self):
        """Blind, deterministic: local maxima (window one beam) whose single-PSF fit at the
        peak, position refined by a 3x3 parabola, reaches S/N >= NEIGHBOUR_SNR against the
        propagated pixel noise."""
        r = max(1, int(round(self.beam_px)))
        img = np.where(np.isfinite(self.error), self.image, -np.inf).astype(np.float32)
        peak = (img == maximum_filter(img, size=2 * r + 1, mode="constant", cval=-np.inf))
        snr = np.zeros(img.shape, dtype=np.float32)
        np.divide(img, self.error, out=snr, where=np.isfinite(self.error))
        peak &= snr >= NEIGHBOUR_PEAK_SNR
        peak[:1] = peak[-1:] = False
        peak[:, :1] = peak[:, -1:] = False
        py, px = np.nonzero(peak)
        if px.size == 0:
            return Neighbours(*(np.empty(0) for _ in range(4)))
        # parabolic sub-pixel refinement from the 3x3 around each peak
        v0 = img[py, px].astype(float)
        left, right = np.maximum(img[py, px - 1], 0), np.maximum(img[py, px + 1], 0)
        down, up = np.maximum(img[py - 1, px], 0), np.maximum(img[py + 1, px], 0)
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
        near = have & (dist < self.beam_px)
        for ex, ey in own_exclude:
            near |= have & (np.hypot(nx_[safe] - ex[:, None], ny_[safe] - ey[:, None]) < self.beam_px)
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
            out[0][r] = flux[:, 0]
            out[1][r] = gain[:, 0]
            out[2][r] = np.sqrt(np.maximum(cov[:, 0, 0], 0))
            out[3][r] = valid
            out[4][r] = un
        return PositionFit(*out)

    def fit_positions(self, x, y, batch=512):
        """Forced fit at each (x, y) -> PositionFit. The same code for real sources and for
        random completeness positions; bright neighbours enter as nuisance columns."""
        return self._fit(x, y, use_nuisance=True, batch=batch)

    def fit_groups(self, x, y, group):
        """Simultaneous fit of blend groups -> GroupFit. `group` (n,) holds an integer label
        per row; rows sharing a label are fitted together, every member at its own fixed
        position with two free scan columns, plus nuisance columns for the tile's bright
        neighbours that are not within one beam of any member."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        group = np.asarray(group)
        n = x.size
        res = {k: np.full(n, np.nan) for k in ("flux", "sigma", "rho", "gflux", "gsigma")}
        covered = np.zeros(n, dtype=bool)
        single = self._fit(x, y)
        _, inverse = np.unique(group, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, n]
        sizes = np.diff(starts)
        for size in np.unique(sizes):
            members = np.array([order[starts[g]:starts[g + 1]] for g in np.flatnonzero(sizes == size)])
            # one box per group, centred on the group centroid, large enough for every member
            gx = x[members].mean(axis=1)
            gy = y[members].mean(axis=1)
            spread = np.ceil(np.max(np.hypot(x[members] - gx[:, None], y[members] - gy[:, None]), axis=1))
            for lo in range(0, len(members), 64):
                sel = slice(lo, lo + 64)
                mem = members[sel]
                half = self.half + int(spread[sel].max())
                cxg = np.rint(gx[sel]).astype(np.int64)
                cyg = np.rint(gy[sel]).astype(np.int64)
                ux, uy, un = self._nuisance(gx[sel], gy[sel],
                                            own_exclude=[(x[mem[:, k]], y[mem[:, k]]) for k in range(size)])
                flux, gain, cov, valid = self._call(x[mem], y[mem], np.full(len(mem), size, dtype=np.int64),
                                                    ux, uy, un, cxg, cyg, half)
                sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0))
                with np.errstate(all="ignore"):
                    corr = np.abs(cov / (sd[:, :, None] * sd[:, None, :]))
                    corr[:, np.arange(size), np.arange(size)] = 0.0
                    rho = np.nanmax(corr, axis=2) if size > 1 else np.zeros_like(sd)
                rho = np.where(np.isfinite(rho), rho, 1.0)
                gsum = np.nansum(flux, axis=1)
                gsig = np.sqrt(np.maximum(np.nansum(cov, axis=(1, 2)), 0))
                for k in range(size):
                    i = mem[:, k]
                    res["flux"][i] = flux[:, k]
                    res["sigma"][i] = sd[:, k]
                    res["rho"][i] = np.where(valid, rho[:, k], 1.0)
                    res["gflux"][i] = np.where(valid, gsum, np.nan)
                    res["gsigma"][i] = np.where(valid, gsig, np.nan)
                    covered[i] = valid
        with np.errstate(all="ignore"):
            inflation = res["sigma"] / single.sigma
        return GroupFit(res["flux"], res["sigma"], inflation, res["rho"],
                        res["gflux"], res["gsigma"], covered)

    def fitted_fwhm(self, x, y):
        """Best-fit total FWHM (arcsec) of a source at each position (NaN where the fit is not
        possible): the beam broadened by a Gaussian of width chosen on a small grid by
        chi-square, two scan columns plus a constant background, no nuisance columns. Meant
        for detections only (photometry's EXTENDED flag is this > 1.5 x beam)."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        widths = np.array([0.0, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]) * self.beam_px / 4.0  # Gaussian sigma, pixels
        half = self.half
        side = 2 * half + 1
        psf = self._psf
        out = np.full(x.size, np.nan)
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
            t = np.zeros((2, side * side))
            _box_templates(psf.table, psf.extent, psf.subpixel_steps, self.shape[1], self.shape[0],
                           x[i], y[i], x0, y0, side, t, 0)
            best, best_chi = np.nan, np.inf
            for s in widths:
                cols = [gaussian_filter(c.reshape(side, side), s).ravel() if s > 0 else c for c in t]
                M = np.c_[cols[0], cols[1], np.ones(side * side)][ok] / e[ok, None]
                coef = np.linalg.lstsq(M, d[ok] / e[ok], rcond=None)[0]
                chi = np.sum((M @ coef - d[ok] / e[ok]) ** 2)
                if chi < best_chi:
                    best_chi, best = chi, s
            out[i] = np.hypot(self.beam_arcsec, 2.3548 * best * self.pixel_arcsec)
        return out

    def pixel_sigma(self, x, y):
        """Error-layer value (mJy/pixel) at the nearest pixel, NaN off the data."""
        cx = np.rint(np.asarray(x)).astype(np.int64)
        cy = np.rint(np.asarray(y)).astype(np.int64)
        ok = (cx >= 0) & (cx < self.shape[1]) & (cy >= 0) & (cy < self.shape[0])
        out = np.full(cx.shape, np.nan)
        out[ok] = self.error[cy[ok], cx[ok]]
        return out


@contextmanager
def open_tile(config, record):
    """Open `record` read-only (memory-mapped), build the PSF response table, find the bright
    neighbours, and yield a `Tile`."""
    with open_map(config, record) as omap:
        yield Tile(omap, config)


_CACHE = {}


def cached_tile(config, record):
    """The process-local Tile for `record`: built on the first call in a worker and kept (one tile
    at a time) so the pieces of a tile share its PSF table and neighbour list."""
    key = record.path
    if key not in _CACHE:
        _CACHE.clear()
        with open_tile(config, record) as tile:
            _CACHE[key] = tile        # arrays are native copies: valid after the file closes
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
    paths = sorted((Path(config.data_root) / "catalog" / "sesna").glob("sources_sesna_source__*.hdf5"))
    ras, decs, rows, names = [], [], [], []
    n_read = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            ra, dec = handle["RA_DEG"], handle["DEC_DEG"]
            n_read += ra.size
            for start in range(0, ra.size, chunk_rows):
                r, d = ra[start:start + chunk_rows], dec[start:start + chunk_rows]
                v = np.c_[np.cos(np.radians(d)) * np.cos(np.radians(r)),
                          np.cos(np.radians(d)) * np.sin(np.radians(r)), np.sin(np.radians(d))]
                keep = np.linalg.norm(v - centre, axis=1) <= chord
                ras.append(r[keep])
                decs.append(d[keep])
                rows.append(np.arange(start, start + r.size, dtype=np.int64)[keep])
                names.append(np.full(int(keep.sum()), path.name, dtype=object))
    cat = lambda parts, empty: np.concatenate(parts) if parts else empty
    out = SourceCandidates(cat(ras, np.empty(0)), cat(decs, np.empty(0)),
                           cat(rows, np.empty(0, dtype=np.int64)), cat(names, np.empty(0, dtype=object)))
    return out, {"catalogues": len(paths), "rows_read": n_read, "rows_retained": out.ra_deg.size}
