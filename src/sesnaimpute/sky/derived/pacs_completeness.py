"""The Herschel PACS completeness of the SESNA sources, per region and band, MEASURED by recovering
artificial point sources (PACS_COMPLETENESS_PLAN.md v2; SPEC_PACS_FARIR_DRAFT.md sections 5 and 11.3).

`build(config, regions=None)` writes one product per region,
`sky/derived/pacs/completeness_pacs_source__<Region>.hdf5`, in EXACTLY the layout of
`catalog/sesna/completeness_sesna_source__<Region>.hdf5` for the bands P70, P100 and P160
(`CURVE_<B>`, `MAG_GRID_<B>`, `SOURCE_ROW`, `ENVELOPE_<B>`, `COMPLETED_FROM_<B>`, `N_POS_<B>`), plus per-cell
columns aligned with the CURVE rows (`CELL_RA/DEC/ID/MAP_INDEX/OBSID/MAP_<B>`, `CELL_SIGMA_<B>_MJY`,
`CELL_OFFSET_<B>_MJY`, `N_SIGMA_POS_<B>`, `FLUX90_<B>_MJY`, `BIAS_<B>_MJY`, `SCATTER_<B>_MJY`,
`SCATTER_BLEND_<B>_MJY`) that the photometry stage reads.

Method. Nothing here is derived from a noise model: completeness is the recovered fraction of artificial
point sources injected into the real map and measured by the catalogue's own rule.

1. The detection rule's inputs. Per 30-arcsec cell on each map tile, 200 positions are drawn uniformly
   (real sources are not excluded) and measured with the catalogue's aperture photometry (`pacs_fit`).
   CELL_OFFSET is the median flux over the cell's clean positions (sky structure the aperture's sky
   leaves), CELL_SIGMA the median over the cell's 3 x 3 block of 1.4826 x the median absolute deviation
   about each cell's offset. They have one use: a flux is a detection when flux - CELL_OFFSET >= 5 x
   CELL_SIGMA of the cell. The photometry stage applies exactly this rule to the SESNA sources.
2. The real detections. On each map tile the SESNA sources it measures (the deepest map of each source)
   are measured and the rule applied (`pacs_photometry.singleton_pass`, the photometry's own pass).
3. Injection. For each plane of MAG_GRID, batches of artificial sources (the band's point-spread
   template, total flux F, added to a copy of the map at positions uniform over the tile's valid pixels
   inside the region's cells, at most one per 50 beam areas and never closer than the sky annulus's
   outer radius, so injected sources do not blend with each other) are measured through the catalogue's
   own path: the aperture where the position is farther than BLEND_RADIUS_BEAMS x beam from every
   detected source, the blend-group fit (the injected source one more member of the group of the
   detected sources within that radius) otherwise. RECOVERED means flux - CELL_OFFSET >= 5 CELL_SIGMA of
   the injected position's cell; no flux tolerance. An injection within half a beam of a detected source
   is dropped from every count (a SESNA source there is an unseparable blend). Batches alternate by
   seed parity: even seeds build the product, odd seeds are the held-out half kept in memory for the
   acceptance checks (`acceptance`).
4. Pooling. A cell's counts are the sums over the cells of its map whose centres lie within POOL_ARCSEC of
   its own. CURVE is the pooled recovered fraction; FLUX90 the faintest flux at which it still reaches
   0.9; BIAS and SCATTER the median and 1.4826 x MAD of (recovered - injected flux) over the pooled
   recovered injections of the aperture path, SCATTER_BLEND the same for the group path.

Every covered cell of every map inside the region's padded source bounding box has a CURVE row (rows ordered
by map, then cell); where maps overlap, SOURCE_ROW points each source at its cell on the deepest map (lowest
cell sigma). The flux grid is one per band for the
whole survey: m = -2.5 log10(F / 1 Jy) from MAG_MIN to MAG_MAX in MAG_STEP, bright to faint. A region a band
does not cover gets the same product with no rows and SOURCE_ROW -1.

Run: `PY sesnaimpute.sky.derived.pacs_completeness $CONFIG [--regions ...]`.
"""

import functools
import os
import time

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived import pacs_fit
from sesnaimpute.sky.derived.pacs_photometry import BLEND_RADIUS_BEAMS, BLEND_RHO_LIMIT, _blend_components, singleton_pass
from sesnaimpute.sky.derived.pacs_psf import APERTURES

BANDS = ("P70", "P100", "P160")
#: positions per cell (spec section 5)
N_POS = 200
#: a cell needs this many valid positions to get a curve; fewer means the cell is not covered
MIN_VALID_POSITIONS = 100
#: the stored curve is the fraction times this (the SESNA convention)
STORAGE_SCALE = 600
#: PACS has no Vega system: zero point 1 Jy, m = -2.5 log10(F / 1 Jy)
ZERO_POINT_JY = 1.0
#: ONE grid per band for the whole survey, bright to faint (6 mJy is below the 5-sigma flux of the deepest
#: cell, 25 Jy above the roll-off of every cell)
MAG_MIN = -3.5
MAG_MAX = 5.5
MAG_STEP = 0.25
#: a piece of the cell statistics is one tile's square block of BLOCK_CELLS x BLOCK_CELLS cells (plus a
#: one-cell border whose fits feed the 3 x 3 noise median of the block's edge cells)
BLOCK_CELLS = 16
MAD_TO_SIGMA = 1.4826
#: the map's reference noise (the fallback for a cell with no clean neighbour) is the median over up to this
#: many evenly spread cells of the tile
REFERENCE_CELLS = 400

#: injections: at most one artificial source per this many beam areas per batch
SPARSE_BEAM_AREAS = 50.0
#: an injection this close (in beams) to a detected source is dropped from every count
DROP_BEAMS = 0.5
#: a cell's counts are pooled over cells whose centres lie within this radius of its own; every
#: (cell, plane) of the build half has at least MIN_INJECTIONS injections behind it
POOL_ARCSEC = 120.0
MIN_INJECTIONS = 100
#: the first round of batches per plane and half aims at this multiple of MIN_INJECTIONS in the pooled
#: interior; further batches, restricted to the cells whose pooled count is short, follow until none is
INTERIOR_SAFETY = 1.3
MAX_EXTRA_BATCHES = 40
#: the recovered fraction FLUX90 is read at
FLUX90_FRACTION = 0.9


def canonical_grid():
    """(magnitude planes, flux planes in mJy), bright to faint."""
    n = int(round((MAG_MAX - MAG_MIN) / MAG_STEP)) + 1
    mag = MAG_MIN + MAG_STEP * np.arange(n)
    return mag, 1000.0 * ZERO_POINT_JY * 10.0 ** (-mag / 2.5)


# ----------------------------------------------------------------------------
# the cell statistic

def cell_fits(tile, ids, seed):
    """Forced fits at N_POS uniform positions in each cell of `ids` on `tile`. Returns b (flux
    mJy), gain, pixel sigma (propagated), valid, clean (farther than one beam from every bright
    neighbour), each (n_cell, N_POS)."""
    x, y = tile.grid.sample_pixels(ids, N_POS, seed)
    fit = tile.measure(x.ravel(), y.ravel())
    shape = x.shape
    valid = (fit.covered & np.isfinite(fit.flux) & np.isfinite(fit.gain) & (fit.gain > 0)).reshape(shape)
    clean = ~tile.near_bright(x, y)
    return (fit.flux.reshape(shape), fit.gain.reshape(shape), fit.sigma.reshape(shape), valid, clean)


#: a cell's own noise estimate counts in its block's median only with this many clean positions
MIN_CLEAN_POSITIONS = 50


def cell_statistics(b, gain, pixel_sigma, valid, clean):
    """Per cell: n_valid, n_clean, clean sigma, median gain, median propagated pixel sigma, offset. Over
    the valid positions farther than one beam from every bright neighbour (so a real source in the cell
    sets neither): the offset is the median of b, the sigma 1.4826 x the median absolute deviation about
    it. The offset is sky structure on the 30-arcsec scale that the fit's constant background does not
    absorb (planner's split-half test, Musca P70: the two halves' offsets correlate at 0.93 across
    cells, rms 0.57 sigma), so it is a bias, subtracted from every flux in the cell, not noise."""
    use = valid & clean
    bb = np.where(use, b, np.nan)
    with np.errstate(all="ignore"):
        offset = np.nanmedian(bb, axis=1)
        sigma = MAD_TO_SIGMA * np.nanmedian(np.abs(bb - offset[:, None]), axis=1)
        g = np.nanmedian(np.where(valid, gain, np.nan), axis=1)
        ps = np.nanmedian(np.where(valid, pixel_sigma, np.nan), axis=1)
    return valid.sum(axis=1), use.sum(axis=1), sigma, g, ps, offset


def ring(grid, ids):
    """The cells of the grid in the 3 x 3 block around each of `ids`, flattened and unique."""
    cy, cx = np.divmod(np.asarray(ids, np.int64), grid.nx)
    ny = (cy[:, None] + np.repeat([-1, 0, 1], 3)[None, :])
    nx = (cx[:, None] + np.tile([-1, 0, 1], 3)[None, :])
    ok = (ny >= 0) & (ny < grid.ny) & (nx >= 0) & (nx < grid.nx)
    return np.unique((ny * grid.nx + nx)[ok])


def block_sigma(grid, ids, all_ids, mad_sigma, n_clean):
    """Noise of each cell of `ids`: the median of the MAD sigmas of the cells of its 3 x 3 block (those
    in `all_ids`, sorted, with at least MIN_CLEAN_POSITIONS clean positions). One cell's 200 positions
    hold only ~10 independent beams, so its own MAD carries ~27 % sampling error (review 2026-10-26:
    adjacent cells differ by 0.68-1.46, flat with separation); nine cells bring it to ~10 % at a
    90-arcsec scale. Returns (sigma, number of clean positions behind it); NaN where no cell of the
    block qualifies."""
    cy, cx = np.divmod(np.asarray(ids, np.int64), grid.nx)
    ny = cy[:, None] + np.repeat([-1, 0, 1], 3)[None, :]
    nx = cx[:, None] + np.tile([-1, 0, 1], 3)[None, :]
    inside = (ny >= 0) & (ny < grid.ny) & (nx >= 0) & (nx < grid.nx)
    nb = np.where(inside, ny * grid.nx + nx, -1)
    k = np.clip(np.searchsorted(all_ids, nb), 0, all_ids.size - 1)
    found = inside & (all_ids[k] == nb)
    good = found & (n_clean[k] >= MIN_CLEAN_POSITIONS) & np.isfinite(mad_sigma[k]) & (mad_sigma[k] > 0)
    with np.errstate(all="ignore"):
        sigma = np.nanmedian(np.where(good, mad_sigma[k], np.nan), axis=1)
    return sigma, np.where(good, n_clean[k], 0).sum(axis=1)


def cell_offset(grid, ids, all_ids, offset, n_clean):
    """Offset of each cell of `ids`: its own clean median (the offsets are local: 0.93 split-half
    correlation), or, with fewer than MIN_CLEAN_POSITIONS clean positions, the median of its 3 x 3
    block's qualifying cells, else 0."""
    k = np.searchsorted(all_ids, ids)
    own = np.where((n_clean[k] >= MIN_CLEAN_POSITIONS) & np.isfinite(offset[k]), offset[k], np.nan)
    if np.isnan(own).any():
        cy, cx = np.divmod(np.asarray(ids, np.int64), grid.nx)
        ny = cy[:, None] + np.repeat([-1, 0, 1], 3)[None, :]
        nx = cx[:, None] + np.tile([-1, 0, 1], 3)[None, :]
        inside = (ny >= 0) & (ny < grid.ny) & (nx >= 0) & (nx < grid.nx)
        nbid = np.where(inside, ny * grid.nx + nx, -1)
        kk = np.clip(np.searchsorted(all_ids, nbid), 0, all_ids.size - 1)
        good = inside & (all_ids[kk] == nbid) & (n_clean[kk] >= MIN_CLEAN_POSITIONS) & np.isfinite(offset[kk])
        with np.errstate(all="ignore"):
            block = np.nanmedian(np.where(good, offset[kk], np.nan), axis=1)
        own = np.where(np.isnan(own), np.nan_to_num(block, nan=0.0), own)
    return own


def _tile_reference(piece, config, records):
    """Per map tile, one pieces.run piece each, before the cell blocks: its registration to SESNA
    (`pacs_fit.registration`) and its median cell sigma over up to REFERENCE_CELLS cells spread evenly over
    the cells the region USES there (wholly on valid data), the fallback sigma of a block with no clean
    cell. `piece = (tile index, used cells, SESNA ra, dec on the tile)`."""
    t, used, sra, sdec = piece
    tile = pacs_fit.cached_tile(config, records[t])
    shift = pacs_fit.registration(tile, sra, sdec)
    centres = used[np.unique(np.linspace(0, used.size - 1, min(REFERENCE_CELLS, used.size)).astype(np.int64))]
    all_ids = ring(tile.grid, centres)
    b, gain, ps, valid, clean = cell_fits(tile, all_ids, 0)
    _, n_clean, mad, _, _, _ = cell_statistics(b, gain, ps, valid, clean)
    sigma, _ = block_sigma(tile.grid, centres, all_ids, mad, n_clean)
    c = np.searchsorted(all_ids, centres)
    good = np.isfinite(sigma) & (sigma > 0) & (valid[c].sum(axis=1) == N_POS)
    return dict(n_used=int(used.size), shift=shift, median=float(np.median(sigma[good])) if good.any() else np.nan)


def encode_curves(counts, n_total):
    """Fraction x STORAGE_SCALE, rounded, uint16 (0 where nothing was injected)."""
    return np.rint(counts * STORAGE_SCALE / np.maximum(n_total, 1)).astype(np.uint16)


def _piece(piece, config, records, refs):
    """The detection rule's inputs for one tile's square block of cells: the arrays the parent assembles.
    The block's border cells are fitted too, only for the noise median."""
    tile_index, ids = piece
    record = records[tile_index]
    tile = pacs_fit.cached_tile(config, record)
    all_ids = ring(tile.grid, ids)
    b, gain, ps, valid, clean = cell_fits(tile, all_ids, tile_index + 1)
    n_valid, n_clean, mad, g, pps, off = cell_statistics(b, gain, ps, valid, clean)
    sigma, n_sigma = block_sigma(tile.grid, ids, all_ids, mad, n_clean)
    offset = cell_offset(tile.grid, ids, all_ids, off, n_clean)
    ref = refs[tile_index]
    # a block with no clean cell (wholly within a beam of bright sources) takes the tile's median noise;
    # N_SIGMA_POS 0 marks it
    sigma = np.where(np.isfinite(sigma), sigma, ref["median"])
    k = np.searchsorted(all_ids, ids)
    n_valid = n_valid[k]
    covered = (n_valid >= MIN_VALID_POSITIONS) & np.isfinite(sigma) & (sigma > 0)
    ra, dec = pacs_fit.from_map_frame(*tile.grid.centres_icrs(ids), ref["shift"][:2])
    return dict(tile=tile_index, ids=ids, covered=covered,
                n_sigma=np.minimum(n_sigma, np.iinfo(np.int16).max).astype(np.int16),
                sigma=sigma, offset=offset, ra=ra, dec=dec)


# ----------------------------------------------------------------------------
# pooling

def cell_centres_arcsec(grid, ids):
    """(n, 2) arcsec position of the centre of each cell of `ids` in the map's own pixel frame."""
    x0, x1, y0, y1 = grid.bounds(ids)
    return np.c_[(x0 + x1) / 2 * (pacs_fit.CELL_ARCSEC / grid.step_x), (y0 + y1) / 2 * (pacs_fit.CELL_ARCSEC / grid.step_y)]


def pool_matrix(centres_arcsec, radius_arcsec=None):
    """Sparse (n, n) 0/1 matrix: entry (i, j) is 1 where cell j's centre lies within POOL_ARCSEC of cell i's
    (a cell pools with itself). Cells of different maps must be given far-apart centres."""
    radius = POOL_ARCSEC if radius_arcsec is None else radius_arcsec
    n = len(centres_arcsec)
    pairs = cKDTree(centres_arcsec).query_pairs(radius, output_type="ndarray").reshape(-1, 2)
    r = np.concatenate([np.arange(n), pairs[:, 0], pairs[:, 1]])
    c = np.concatenate([np.arange(n), pairs[:, 1], pairs[:, 0]])
    return csr_matrix((np.ones(r.size), (r, c)), shape=(n, n))


def pooled_counts(W, row, plane, recovered, n_row, n_plane):
    """The pooled number of injections and of recovered injections at every (cell, plane): sums, over the
    cells W pools with each, of the counts of the injections (`row` the cell's row, `plane`, `recovered`)."""
    key = row.astype(np.int64) * n_plane + plane
    tot = np.bincount(key, minlength=n_row * n_plane).reshape(n_row, n_plane).astype(np.float64)
    rec = np.bincount(key[recovered], minlength=n_row * n_plane).reshape(n_row, n_plane).astype(np.float64)
    return W @ tot, W @ rec


def pooled_median_scatter(W, row, value, n_row, min_n=MIN_INJECTIONS):
    """Per cell, the median and 1.4826 x the median absolute deviation about it of the `value`s of the
    records (at cell `row`) of the cells W pools with it; NaN where fewer than `min_n` records."""
    out_med = np.full(n_row, np.nan)
    out_sc = np.full(n_row, np.nan)
    if row.size == 0:
        return out_med, out_sc
    order = np.argsort(row, kind="stable")
    r, v = row[order], value[order].astype(np.float64)
    count = np.bincount(r, minlength=n_row)
    start = np.cumsum(count) - count
    pair_c = np.repeat(np.arange(n_row), np.diff(W.indptr))
    pair_n = W.indices
    k = count[pair_n]
    total = int(k.sum())
    if total == 0:
        return out_med, out_sc
    seg = np.repeat(pair_c, k)
    within = np.arange(total) - np.repeat(np.cumsum(k) - k, k)
    vals = v[np.repeat(start[pair_n], k) + within]
    n_c = np.bincount(seg, minlength=n_row)
    first = np.cumsum(n_c) - n_c

    def segmented_median(x):
        xs = x[np.lexsort((x, seg))]
        lo = np.minimum(first + (n_c - 1) // 2, xs.size - 1)
        hi = np.minimum(first + n_c // 2, xs.size - 1)
        return np.where(n_c > 0, 0.5 * (xs[lo] + xs[hi]), np.nan)

    med = segmented_median(vals)
    mad = segmented_median(np.abs(vals - med[seg]))
    good = n_c >= min_n
    out_med[good] = med[good]
    out_sc[good] = MAD_TO_SIGMA * mad[good]
    return out_med, out_sc


def flux90(fraction, flux_grid, level=FLUX90_FRACTION):
    """Per row, the faintest flux at which the recovered fraction still reaches `level`: reading the grid from
    its bright end, the plane before the first one below `level`, log-linearly interpolated to the point
    where the fraction crosses it; the faintest plane's flux if no plane falls below; NaN if even the
    brightest plane does not reach it."""
    n, p = fraction.shape
    below = fraction < level
    k = np.where(below.any(axis=1), below.argmax(axis=1), p)
    lf = np.log10(flux_grid)
    r = np.arange(n)
    km = np.clip(k - 1, 0, p - 1)
    kk = np.clip(k, 0, p - 1)
    with np.errstate(all="ignore"):
        t = (fraction[r, km] - level) / (fraction[r, km] - fraction[r, kk])
        inter = 10.0 ** (lf[km] + t * (lf[kk] - lf[km]))
    out = np.where(k >= p, flux_grid[-1], inter)
    return np.where(k == 0, np.nan, out)


# ----------------------------------------------------------------------------
# injection

class Injector:
    """Artificial point sources injected into one map tile and measured through the catalogue's own path.

    `inp` carries what the stage knows of the tile: its rows' cell ids (`ids`), their `sigma` and `offset`
    (the float32 values stored in the product), and the SESNA sources the photometry will measure on this
    tile (`src_ra`, `src_dec`, `src_sigma`, `src_offset`) with the map's registration `shift`.
    """

    def __init__(self, tile, inp, seed_key):
        self.tile = tile
        self.seed_key = seed_key
        grid = tile.grid
        self.ids = np.asarray(inp["ids"], np.int64)
        n = self.ids.size
        self.row_of_cell = np.full(grid.n_cells, -1, np.int32)
        self.row_of_cell[self.ids] = np.arange(n, dtype=np.int32)
        self.sigma = np.full(grid.n_cells, np.nan)
        self.offset = np.full(grid.n_cells, np.nan)
        self.sigma[self.ids] = inp["sigma"]
        self.offset[self.ids] = inp["offset"]
        self.beam_px = tile.beam_px
        self.beam_area_px = np.pi / (4 * np.log(2)) * tile.beam_px ** 2
        self.separation_px = APERTURES[tile.record.band][2] / tile.pixel_arcsec
        # the real detections: the photometry's own singleton pass on the sources it measures on this tile
        self.det_tree = None
        self.det_label = np.zeros(0, np.int64)
        self.members = {}
        self.src_xy = np.zeros((0, 2))
        self.n_detected = 0
        if inp["src_ra"].size:
            x, y, _single, _ok, det = singleton_pass(tile, inp["src_ra"], inp["src_dec"], inp["shift"],
                                                     inp["src_sigma"], inp["src_offset"])
            self.n_detected = int(det.sum())
            self.src_xy = np.c_[x, y]
            if det.any():
                self.det_tree = cKDTree(self.src_xy[det])
            # blend groups of the real sources, as the photometry forms them
            label, _size = _blend_components(inp["src_ra"], inp["src_dec"], det, BLEND_RADIUS_BEAMS * tile.beam_arcsec)
            self.det_label = label[det]
            order = np.argsort(label, kind="stable")
            starts = np.r_[0, np.flatnonzero(np.diff(label[order])) + 1, label.size]
            self.members = {int(label[order[a]]): order[a:b] for a, b in zip(starts[:-1], starts[1:])}
        # candidate positions are drawn inside the cell of a random row; the share on valid pixels sets the density
        self.cell_px = float(grid.step_x * grid.step_y)
        self.valid_fraction = 1.0
        probe = self._draw(np.random.default_rng(0), np.arange(n), 20000)
        self.valid_fraction = float(self._valid(*probe).mean()) if n else 0.0

    def _draw(self, rng, rows, n):
        """n positions, each uniform inside the cell of a randomly chosen row of `rows`."""
        x0, x1, y0, y1 = self.tile.grid.bounds(self.ids[rows[rng.integers(0, rows.size, n)]])
        return x0 + (x1 - x0) * rng.random(n), y0 + (y1 - y0) * rng.random(n)

    def _valid(self, x, y):
        """Positions inside a row's cell, on a pixel of valid data."""
        cid = self.tile.grid.cell_ids(x, y)
        ok = cid >= 0
        ok[ok] = self.row_of_cell[cid[ok]] >= 0
        px = np.rint(x).astype(np.int64)
        py = np.rint(y).astype(np.int64)
        ny, nx = self.tile.shape
        ok &= (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
        ok[ok] = np.isfinite(self.tile.error[py[ok], px[ok]])
        return ok

    def _thin(self, x, y, limit):
        """Greedy selection, in order, of at most `limit` positions no two closer than the sky annulus's outer
        radius (so injected sources do not blend with each other)."""
        n = x.size
        if n == 0:
            return np.zeros(0, np.int64)
        pairs = cKDTree(np.c_[x, y]).query_pairs(self.separation_px, output_type="ndarray").reshape(-1, 2)
        both = np.concatenate([pairs, pairs[:, ::-1]])
        both = both[np.argsort(both[:, 0], kind="stable")]
        edges = np.searchsorted(both[:, 0], np.arange(n + 1))
        removed = np.zeros(n, bool)
        chosen = []
        for i in range(n):
            if removed[i]:
                continue
            chosen.append(i)
            if len(chosen) >= limit:
                break
            removed[both[edges[i]:edges[i + 1], 1]] = True
        return np.asarray(chosen, np.int64)

    def positions(self, rng, rows):
        """One batch's positions: uniform over the valid pixels of the cells of `rows`, at most one per
        SPARSE_BEAM_AREAS beam areas, none within DROP_BEAMS of a detected source. Returns (x, y, distance in
        pixels to the nearest detected source, inf if none)."""
        target = max(int(round(rows.size * self.cell_px * self.valid_fraction / (SPARSE_BEAM_AREAS * self.beam_area_px))), 1)
        x, y = self._draw(rng, rows, int(2.5 * target) + 8)
        ok = self._valid(x, y)
        x, y = x[ok], y[ok]
        keep = self._thin(x, y, target)
        x, y = x[keep], y[keep]
        if self.det_tree is None:
            return x, y, np.full(x.size, np.inf)
        d, _ = self.det_tree.query(np.c_[x, y])
        far = d >= DROP_BEAMS * self.beam_px
        return x[far], y[far], d[far]

    def batch(self, rng, rows, flux):
        """Injects one batch of `flux` (mJy) sources, measures each through the catalogue's own path, restores
        the map. Returns dict(row, path (1 aperture, 2 group), recovered, dflux (recovered - injected flux, mJy)
        for the injections that lie on covered data; n; the seconds spent in each step and the number of
        injections on each path)."""
        tile = self.tile
        x, y, d = self.positions(rng, rows)
        n = x.size
        out = dict(row=np.zeros(0, np.int32), path=np.zeros(0, np.int8), recovered=np.zeros(0, bool),
                   dflux=np.zeros(0, np.float32), n=0, seconds=dict(inject=0.0, aperture=0.0, group=0.0),
                   count=dict(aperture=0, group=0))
        if n == 0:
            return out
        cid = tile.grid.cell_ids(x, y)
        sigma, offset = self.sigma[cid], self.offset[cid]
        group = d < BLEND_RADIUS_BEAMS * self.beam_px
        flux_out = np.full(n, np.nan)
        covered = np.zeros(n, bool)
        separable = np.ones(n, bool)
        t_inject = t_ap = t_gr = 0.0
        t0 = time.time()
        saved = tile._add_sources(x, y, flux)
        t_inject += time.time() - t0
        try:
            ap = np.flatnonzero(~group)
            if ap.size:
                t0 = time.time()
                m = tile.measure(x[ap], y[ap])
                t_ap = time.time() - t0
                flux_out[ap], covered[ap] = m.flux, m.covered
            gr = np.flatnonzero(group)
            if gr.size:
                # each injected source is one more member of the group of the real sources the photometry
                # would fit together with the detected sources within the blend radius of it
                t0 = time.time()
                xs, ys, lab, last, used = [], [], [], [], 0
                for j in gr:
                    near = self.det_tree.query_ball_point([x[j], y[j]], BLEND_RADIUS_BEAMS * self.beam_px)
                    mem = np.concatenate([self.members[int(l)] for l in np.unique(self.det_label[near])])
                    xs += [self.src_xy[mem, 0], [x[j]]]
                    ys += [self.src_xy[mem, 1], [y[j]]]
                    lab.append(np.full(mem.size + 1, j))
                    used += mem.size + 1
                    last.append(used - 1)
                gfit = tile.fit_groups(np.concatenate(xs), np.concatenate(ys), np.concatenate(lab))
                last = np.asarray(last)
                flux_out[gr] = gfit.flux[last]
                covered[gr] = gfit.covered[last]
                separable[gr] = gfit.max_abs_rho[last] <= BLEND_RHO_LIMIT
                t_gr = time.time() - t0
        finally:
            t0 = time.time()
            tile._restore(saved)
            t_inject += time.time() - t0
        with np.errstate(invalid="ignore"):
            recovered = (covered & separable & np.isfinite(sigma) & (sigma > 0)
                         & (flux_out - offset >= pacs_fit.DETECTION_SNR * sigma))
        keep = covered
        out.update(row=self.row_of_cell[cid[keep]], path=np.where(group, 2, 1).astype(np.int8)[keep],
                   recovered=recovered[keep], dflux=(flux_out - flux).astype(np.float32)[keep], n=int(keep.sum()),
                   seconds=dict(inject=t_inject, aperture=t_ap, group=t_gr),
                   count=dict(aperture=int((~group)[keep].sum()), group=int(group[keep].sum())))
        return out


def _inject_half(injector, W, plane, flux, half, n_row, interior_pooled):
    """Batches of one plane and one half (0 build, 1 held out: the seed's parity), enough that every cell's
    pooled count reaches MIN_INJECTIONS where the cells allow: a first round sized for the interior, then
    batches restricted to the cells within the pooling radius of a short cell. Returns the injections, the
    seconds spent and the injections per path."""
    first = int(np.ceil(INTERIOR_SAFETY * MIN_INJECTIONS / max(interior_pooled, 1e-9)))
    rows_all = np.arange(n_row)
    recs = []
    count = np.zeros(n_row)
    seconds = dict(inject=0.0, aperture=0.0, group=0.0)
    n_by_path = dict(aperture=0, group=0)
    for seed in range(first + MAX_EXTRA_BATCHES):
        if seed < first:
            rows = rows_all
        else:
            short = (W @ count) < MIN_INJECTIONS
            if not short.any():
                break
            rows = np.flatnonzero((W @ short.astype(float)) > 0)
        r = injector.batch(np.random.default_rng([injector.seed_key, plane, 2 * seed + half]), rows, flux)
        recs.append(r)
        count += np.bincount(r["row"], minlength=n_row)
        for k in seconds:
            seconds[k] += r["seconds"][k]
        for k in n_by_path:
            n_by_path[k] += r["count"][k]
    cat = lambda key, dt: np.concatenate([r[key] for r in recs]) if recs else np.zeros(0, dt)
    return (dict(row=cat("row", np.int32), path=cat("path", np.int8), recovered=cat("recovered", bool),
                 dflux=cat("dflux", np.float32)), seconds, n_by_path)


def pool_product(W, rec, flux_grid, n_row):
    """The pooled recovered fraction (encoded), its 0.9 flux, the pooled injection count at the faintest
    plane and the pooled photometric bias and scatter of the injections `rec` (dict row, plane, path,
    recovered, dflux) on the cells W pools."""
    n_plane = flux_grid.size
    tot, got = pooled_counts(W, rec["row"], rec["plane"].astype(np.int64), rec["recovered"], n_row, n_plane)
    with np.errstate(all="ignore"):
        fraction = np.where(tot > 0, got / tot, 0.0)
    bias = np.full((n_row, n_plane), np.nan, np.float32)
    scatter = np.full((n_row, n_plane), np.nan, np.float32)
    scatter_blend = np.full((n_row, n_plane), np.nan, np.float32)
    for p in range(n_plane):
        here = (rec["plane"] == p) & rec["recovered"]
        for path, med_out, sc_out in ((1, bias, scatter), (2, None, scatter_blend)):
            sel = here & (rec["path"] == path)
            med, sc = pooled_median_scatter(W, rec["row"][sel], rec["dflux"][sel], n_row)
            if med_out is not None:
                med_out[:, p] = med
            sc_out[:, p] = sc
    return dict(curve=encode_curves(got, tot), flux90=flux90(fraction, flux_grid).astype(np.float32),
                n_pos=tot[:, -1].astype(np.int32), bias=bias, scatter=scatter, scatter_blend=scatter_blend,
                pooled_total=tot)


SELF_CHECK_FLUX_MJY = 500.0
SELF_CHECK_POSITIONS = 20
SELF_CHECK_TOLERANCE = 0.01


def self_check(config, record, strict=True):
    """Injects SELF_CHECK_FLUX_MJY at SELF_CHECK_POSITIONS empty-sky positions of `record`'s map into a ZERO
    image (not the real one) and measures them through the aperture path: every recovered flux must be within
    1 % of the injected flux, else the template's normalisation and the aperture fraction disagree. Prints the
    worst ratio; raises ValueError if the check fails and `strict`, else prints that it failed."""
    tile = pacs_fit.cached_tile(config, record)
    rng = np.random.default_rng(0)
    ny, nx = tile.shape
    margin = tile.injection_half_width() + 2
    sep = APERTURES[record.band][2] / tile.pixel_arcsec
    reach = int(np.ceil(APERTURES[record.band][2] / tile.pixel_arcsec)) + 2
    xs, ys = [], []
    for _ in range(100000):
        x, y = rng.uniform(margin, nx - margin), rng.uniform(margin, ny - margin)
        iy, ix = int(round(y)), int(round(x))
        # empty sky: the whole aperture and annulus on valid data
        if np.isfinite(tile.error[iy - reach:iy + reach + 1, ix - reach:ix + reach + 1]).all() and all(np.hypot(x - a, y - b) > sep for a, b in zip(xs, ys)):
            xs.append(x)
            ys.append(y)
            if len(xs) == SELF_CHECK_POSITIONS:
                break
    x, y = np.array(xs), np.array(ys)
    with tile.blank_image(), tile.injected(x, y, SELF_CHECK_FLUX_MJY):
        m = tile.measure(x, y)
    ratio = m.flux[m.covered] / SELF_CHECK_FLUX_MJY
    print("  self-check %s: %g mJy injected at %d empty-sky positions of a zero image, aperture flux / injected: "
          "min %.4f max %.4f (%d of %d on covered data)"
          % (record.map_id, SELF_CHECK_FLUX_MJY, x.size, ratio.min(), ratio.max(), ratio.size, x.size), flush=True)
    if ratio.size == 0 or np.abs(ratio - 1).max() > SELF_CHECK_TOLERANCE:
        if strict:
            raise ValueError("the injected template and the aperture fraction disagree by more than 1 %")
        print("  self-check %s FAILED: the injected template and the aperture fraction disagree by more than 1 %%"
              % record.band, flush=True)


def _inject_tile(piece, config, records, flux_grid):
    """Everything the injection measures on one map tile: `piece = (tile index, inputs)` (see `Injector`).
    Returns `(tile index, dict)`: the pooled arrays of the tile's rows for the product (the build half), the
    held-out half's injections, and the statistics the parent prints."""
    t, inp = piece
    tile = pacs_fit.cached_tile(config, records[t])
    inj = Injector(tile, inp, t + 1)
    n_row = inj.ids.size
    W = pool_matrix(cell_centres_arcsec(tile.grid, inj.ids))
    # injections one batch puts into the pool of an interior cell: density x the pool's cell count
    interior = (inj.cell_px * inj.valid_fraction / (SPARSE_BEAM_AREAS * inj.beam_area_px)) * float(np.median(np.diff(W.indptr)))
    keys = ("row", "plane", "path", "recovered", "dflux")
    stores = ({k: [] for k in keys}, {k: [] for k in keys})
    seconds = dict(inject=0.0, aperture=0.0, group=0.0)
    n_by_path = dict(aperture=0, group=0)
    for p in range(flux_grid.size):
        for half in (0, 1):
            r, sec, cnt = _inject_half(inj, W, p, flux_grid[p], half, n_row, interior)
            for k in ("row", "path", "recovered", "dflux"):
                stores[half][k].append(r[k])
            stores[half]["plane"].append(np.full(r["row"].size, p, np.int8))
            for k in seconds:
                seconds[k] += sec[k]
            for k in n_by_path:
                n_by_path[k] += cnt[k]
    build, held = ({k: np.concatenate(v) for k, v in s.items()} for s in stores)
    out = pool_product(W, build, flux_grid, n_row)
    tot = out.pop("pooled_total")
    out["held"] = held
    out["stats"] = dict(map_id=records[t].map_id, n_row=n_row, n_detected=inj.n_detected,
                        n_build=int(build["row"].size), n_held=int(held["row"].size),
                        min_pool=float(tot.min()), median_pool=float(np.median(tot)),
                        short=float((tot < MIN_INJECTIONS).mean()), seconds=seconds, n_by_path=n_by_path)
    return t, out


# ----------------------------------------------------------------------------
# one band of one region

def _tile_geometry(record):
    wcs = WCS(fits.Header.fromstring(record.wcs_header)).celestial
    return pacs_fit.MapGrid.from_wcs(wcs, record.shape), wcs


#: the cells kept per map: every cell inside the region's source bounding box padded by one nside-512
#: pixel (planner 2026-10-27: catalog.completeness maps the admitted pixels' sample points to cells itself)
REGION_PAD_ARCSEC = 412.0


def tile_inputs(cell_map, cell_id, sigma, offset, source_row, shift, ra, dec):
    """What `Injector` needs per map tile, from the band's rows: `{tile: dict}` for the tiles that hold rows.
    `cell_map`, `cell_id`, `sigma`, `offset` are the rows' map index, cell id and the detection rule's inputs
    (float32 as stored), `source_row` (n_source,) each source's row (-1 none), `shift[t]` the map's
    registration (as stored: float32), `ra`, `dec` the SESNA positions. The sources of a tile are the ones
    whose deepest map it is: the ones the photometry stage measures there, in its order."""
    on = source_row >= 0
    source_tile = np.full(source_row.size, -1, np.int64)
    source_tile[on] = cell_map[source_row[on]]
    out = {}
    for t in np.unique(cell_map):
        rows = np.flatnonzero(cell_map == t)
        s = np.flatnonzero(source_tile == t)
        out[int(t)] = dict(ids=cell_id[rows], sigma=sigma[rows].astype(np.float64), offset=offset[rows].astype(np.float64),
                           src_ra=ra[s], src_dec=dec[s], src_sigma=sigma[source_row[s]].astype(np.float64),
                           src_offset=offset[source_row[s]].astype(np.float64),
                           shift=(float(np.float32(shift[t][0])), float(np.float32(shift[t][1]))), row0=int(rows[0]))
    return out


def _print_tile_stats(s):
    sec, n = s["seconds"], s["n_by_path"]
    per = lambda k, c: 1e4 * sec[k] / max(c, 1)
    print("  %s: %d cells, %d detected sources; injections build %d held-out %d; pooled per (cell, plane) over %g arcsec: "
          "min %d median %d (%.1f %% of (cell, plane) below %d); seconds per 10000 injections: aperture %.1f group %.1f "
          "(%d aperture, %d group injections) adding and restoring %.1f"
          % (s["map_id"], s["n_row"], s["n_detected"], s["n_build"], s["n_held"], POOL_ARCSEC, s["min_pool"],
             s["median_pool"], 100 * s["short"], MIN_INJECTIONS, per("aperture", n["aperture"]), per("group", n["group"]),
             n["aperture"], n["group"], per("inject", n["aperture"] + n["group"])), flush=True)


def _band_product(config, records, ra, dec, n_jobs, stage, flux_grid):
    """Measure every covered cell of every map of one band inside the region's padded source bounding box;
    choose each source's deepest map; inject artificial sources on every map. Returns the band's datasets
    (and the held-out half, the printed statistics), or None when no map reaches a source."""
    n_src = ra.size
    n_plane = flux_grid.size

    def tile_cells(record, shift):
        # the map's cells inside the sources' bounding box (map frame), and each source's cell
        grid, wcs = _tile_geometry(record)
        sx, sy = (np.asarray(v) for v in wcs.all_world2pix(*pacs_fit.to_map_frame(ra, dec, shift), 0))
        src_cell = grid.cell_ids(sx, sy)
        on = np.isfinite(sx) & np.isfinite(sy)
        pad = REGION_PAD_ARCSEC / (abs(wcs.proj_plane_pixel_scales()[0].to_value("arcsec")))
        if not on.any():
            return grid, wcs, src_cell, np.empty(0, np.int64)
        x0 = max(int(np.floor((sx[on].min() - pad + 0.5) / grid.step_x)), 0)
        x1 = min(int(np.floor((sx[on].max() + pad + 0.5) / grid.step_x)), grid.nx - 1)
        y0 = max(int(np.floor((sy[on].min() - pad + 0.5) / grid.step_y)), 0)
        y1 = min(int(np.floor((sy[on].max() + pad + 0.5) / grid.step_y)), grid.ny - 1)
        if x1 < x0 or y1 < y0 or not (src_cell >= 0).any():
            return grid, wcs, src_cell, np.empty(0, np.int64)
        cy, cx = np.meshgrid(np.arange(y0, y1 + 1), np.arange(x0, x1 + 1), indexing="ij")
        return grid, wcs, src_cell, (cy * grid.nx + cx).ravel().astype(np.int64)

    # first pass, unregistered: the cells and sources of each tile, for its noise reference and registration
    first = [tile_cells(r, (0.0, 0.0)) for r in records]
    used = [(t, first[t][3], ra[first[t][2] >= 0], dec[first[t][2] >= 0])
            for t in range(len(records)) if first[t][3].size]
    if not used:
        return None
    refs = pieces_module.run(functools.partial(_tile_reference, config=config, records=records),
                             used, n_jobs, stage=stage, unit="tiles")
    refs = dict(zip([u[0] for u in used], refs))
    for t in sorted(refs):
        e, n, k = refs[t]["shift"]
        print("  %s: registration east %+.2f north %+.2f arcsec from %d pairs%s"
              % (records[t].map_id, e, n, k, "  ABOVE 1.5 ARCSEC" if np.hypot(e, n) > 1.5 else ""), flush=True)
    # second pass, registered: the cells, the pieces
    pieces, geometry = [], []
    for t, record in enumerate(records):
        grid, wcs, src_cell, cells = tile_cells(record, refs[t]["shift"][:2] if t in refs else (0.0, 0.0))
        geometry.append((src_cell, cells, grid, wcs))
        if t not in refs or cells.size == 0:
            continue
        cy, cx = np.divmod(cells, grid.nx)
        block = (cy // BLOCK_CELLS) * (grid.nx // BLOCK_CELLS + 1) + cx // BLOCK_CELLS
        for blk in np.unique(block):
            pieces.append((t, cells[block == blk]))
    if not pieces:
        return None
    # the detection rule's inputs: cell offset and noise
    results = pieces_module.run(functools.partial(_piece, config=config, records=records, refs=refs),
                                pieces, n_jobs, stage=stage, unit="cell blocks")
    names = ("covered", "n_sigma", "sigma", "offset", "ra", "dec")
    # the rows: every covered cell of every map, ordered by (map, cell id)
    parts = []
    for t in range(len(records)):
        mine = [r for r in results if r["tile"] == t]
        if not mine:
            continue
        ids = np.concatenate([r["ids"] for r in mine])
        order = np.argsort(ids)
        d = {k: np.concatenate([r[k] for r in mine])[order] for k in names}
        keep = d["covered"]
        parts.append((t, ids[order][keep], {k: v[keep] for k, v in d.items()}))
    cell_map = np.concatenate([np.full(c.size, t, np.int32) for t, c, _ in parts]) if parts else np.zeros(0, np.int32)
    cell_id = np.concatenate([c for _, c, _ in parts]) if parts else np.zeros(0, np.int64)
    cols = {k: (np.concatenate([d[k] for _, _, d in parts]) if parts else np.zeros(0)) for k in names}
    n_row = cell_id.size
    # each source's deepest covering map (lowest cell sigma), and its row
    keys = cell_map.astype(np.int64) * (1 << 40) + cell_id
    sel_t = np.full(n_src, -1, np.int32)
    sel_r = np.full(n_src, np.inf)
    for t, (src_cell, cells, _g, _w) in enumerate(geometry):
        idx = np.flatnonzero(src_cell >= 0)
        if idx.size == 0 or n_row == 0:
            continue
        q = np.int64(t) * (1 << 40) + src_cell[idx]
        k = np.clip(np.searchsorted(keys, q), 0, n_row - 1)
        hit = keys[k] == q
        sig = np.where(hit, cols["sigma"][k], np.nan)
        pacs_fit.select_deeper(t, idx, sig, hit, sel_t, sel_r)
    src_row = np.full(n_src, -1, np.int64)
    w = np.flatnonzero(sel_t >= 0)
    src_row[w] = np.searchsorted(keys, sel_t[w].astype(np.int64) * (1 << 40) + geometry_cells(geometry, sel_t, w))
    shifts = np.array([refs[t]["shift"] if t in refs else (0.0, 0.0, 0) for t in range(len(records))], dtype=float)
    cell_sigma = cols["sigma"].astype(np.float32)
    cell_offset = cols["offset"].astype(np.float32)

    # artificial sources: one piece per map tile, the biggest first
    inputs = tile_inputs(cell_map, cell_id, cell_sigma, cell_offset, src_row, shifts, ra, dec)
    todo = sorted(inputs.items(), key=lambda kv: -kv[1]["ids"].size)
    injected = pieces_module.run(functools.partial(_inject_tile, config=config, records=records, flux_grid=flux_grid),
                                 todo, n_jobs, stage=stage, unit="maps (artificial sources)")
    curve = np.zeros((n_row, n_plane), np.uint16)
    n_pos = np.zeros(n_row, np.int32)
    flux90_ = np.full(n_row, np.nan, np.float32)
    bias, scatter, scatter_blend = (np.full((n_row, n_plane), np.nan, np.float32) for _ in range(3))
    held = {k: [] for k in ("row", "plane", "path", "recovered", "dflux")}
    stats = []
    for t, out in injected:
        r0 = inputs[t]["row0"]
        sl = slice(r0, r0 + inputs[t]["ids"].size)
        curve[sl], n_pos[sl], flux90_[sl] = out["curve"], out["n_pos"], out["flux90"]
        bias[sl], scatter[sl], scatter_blend[sl] = out["bias"], out["scatter"], out["scatter_blend"]
        for k in held:
            held[k].append(out["held"][k] + r0 if k == "row" else out["held"][k])
        stats.append(out["stats"])
    held = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in held.items()}
    for s in stats:
        _print_tile_stats(s)
    envelope = curve.max(axis=0).astype(np.uint16) if n_row else np.zeros(n_plane, np.uint16)
    return dict(curve=curve, source_row=src_row.astype(np.int32), envelope=envelope, n_pos=n_pos,
                n_sigma=cols["n_sigma"].astype(np.int16), flux90=flux90_, bias=bias, scatter=scatter,
                scatter_blend=scatter_blend, held=held, stats=stats,
                cell_ra=cols["ra"], cell_dec=cols["dec"], cell_sigma=cell_sigma, cell_offset=cell_offset,
                cell_id=cell_id, cell_map_index=cell_map,
                cell_obsid=np.array([",".join(records[t].obsids) for t in cell_map] or [""], dtype="S")[:n_row],
                cell_map=np.array([records[t].map_id for t in cell_map] or [""], dtype="S")[:n_row],
                n_candidate=sum(g[1].size for g in geometry),
                map_id=np.array([r.map_id for r in records], dtype="S"),
                map_obsid=np.array([",".join(r.obsids) for r in records], dtype="S"),
                map_shift=shifts[:, :2].astype(np.float32), map_pairs=shifts[:, 2].astype(np.int32),
                map_wcs=np.array([g[3].to_header_string() for g in geometry], dtype="S"),
                map_shape=np.array([g[2].shape for g in geometry], dtype=np.int32).reshape(len(records), 2),
                map_cell_step=np.array([(g[2].step_x, g[2].step_y) for g in geometry], dtype=np.float64).reshape(len(records), 2))


def geometry_cells(geometry, sel_t, w):
    """The selected map's cell id of each source in `w`."""
    out = np.empty(w.size, np.int64)
    for t in np.unique(sel_t[w]):
        k = sel_t[w] == t
        out[k] = geometry[t][0][w[k]]
    return out


def _empty_band(n_src, n_plane):
    z = lambda dt: np.zeros(0, dt)
    nan = lambda: np.zeros((0, n_plane), np.float32)
    return dict(curve=np.zeros((0, n_plane), np.uint16), source_row=np.full(n_src, -1, np.int32),
                envelope=np.zeros(n_plane, np.uint16), n_pos=z(np.int32), n_sigma=z(np.int16), flux90=z(np.float32),
                bias=nan(), scatter=nan(), scatter_blend=nan(), held=None, stats=[], cell_ra=z(float),
                cell_dec=z(float), cell_sigma=z(np.float32), cell_offset=z(np.float32), cell_id=z(np.int64),
                cell_map_index=z(np.int32), cell_obsid=z("S1"), cell_map=z("S1"), n_candidate=0, map_id=z("S1"),
                map_obsid=z("S1"), map_shift=np.zeros((0, 2), np.float32), map_pairs=z(np.int32), map_wcs=z("S1"),
                map_shape=np.zeros((0, 2), np.int32), map_cell_step=np.zeros((0, 2)))


# ----------------------------------------------------------------------------
# writing

def write_product(path, region, mag, bands):
    """Write the region product at `path` (the final shapes are known here: nothing grows)."""
    stem = "completeness_pacs_source"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.attrs["REGION"] = region
        for b in BANDS:
            d = bands[b]
            w = lambda name, data, **kw: build_module.write_dataset(f, name, data, *REGISTRY[(stem, name)], **kw)
            ds = w("CURVE_" + b, d["curve"])
            ds.attrs["ZERO_POINT_JY"] = ZERO_POINT_JY
            w("MAG_GRID_" + b, mag)
            w("ENVELOPE_" + b, d["envelope"])
            w("COMPLETED_FROM_" + b, np.full(d["curve"].shape[0], np.nan, np.float32))
            w("N_POS_" + b, d["n_pos"])
            w("N_SIGMA_POS_" + b, d["n_sigma"])
            w("FLUX90_%s_MJY" % b, d["flux90"])
            w("BIAS_%s_MJY" % b, d["bias"])
            w("SCATTER_%s_MJY" % b, d["scatter"])
            w("SCATTER_BLEND_%s_MJY" % b, d["scatter_blend"])
            w("CELL_RA_" + b, d["cell_ra"])
            w("CELL_DEC_" + b, d["cell_dec"])
            w("CELL_ID_" + b, d["cell_id"])
            w("CELL_MAP_INDEX_" + b, d["cell_map_index"])
            w("CELL_OBSID_" + b, d["cell_obsid"])
            w("CELL_MAP_" + b, d["cell_map"])
            w("CELL_SIGMA_%s_MJY" % b, d["cell_sigma"])
            w("CELL_OFFSET_%s_MJY" % b, d["cell_offset"])
            w("MAP_ID_" + b, d["map_id"])
            w("MAP_OBSID_" + b, d["map_obsid"])
            w("MAP_SHIFT_%s_ARCSEC" % b, d["map_shift"])
            w("MAP_N_PAIRS_" + b, d["map_pairs"])
            w("MAP_WCS_" + b, d["map_wcs"])
            w("MAP_SHAPE_" + b, d["map_shape"])
            w("MAP_CELL_STEP_" + b, d["map_cell_step"])
        w = lambda name, data: build_module.write_dataset(f, name, data, *REGISTRY[(stem, name)])
        w("SOURCE_ROW", np.stack([bands[b]["source_row"] for b in BANDS], axis=1).astype(np.int32))


def _build_region(config, region, out_path, ra, dec, n_jobs, stage=None, maps=None):
    """Measure and write one region. `ra`/`dec` are the region's source positions in the curated order; `maps`
    (observation ids) restricts the maps used to the ones holding one of them (development on a few maps).
    Returns the per-band products (for the caller's checks, with each band's held-out half under "held")."""
    mag, flux_grid = canonical_grid()
    out = {}
    for b in BANDS:
        records = [r for r in pacs_fit.read_index(config, region) if r.band == b
                   and (maps is None or any(str(o) in maps for o in r.obsids))]
        scan = [r for r in records if r.mode == 1]
        if scan:
            self_check(config, scan[0], strict=b != "P160")
        out[b] = _band_product(config, records, ra, dec, n_jobs, stage, flux_grid) if records else None
        if out[b] is None:
            out[b] = _empty_band(ra.size, mag.size)
    write_product(out_path, region, mag, out)
    return out


def _region_inputs(config, region):
    """The region's source positions (NAME order of the curated source table; the redesign's
    catalog.curated.read_positions gives the same rows from the delivery)."""
    with h5py.File(product_path(config, "catalog", "sesna", "sources", "source", region=region), "r") as f:
        return f["RA_DEG"][:].astype(np.float64), f["DEC_DEG"][:].astype(np.float64)


def build(config, regions=None, maps=None):
    """Writes each region's `completeness_pacs_source` product, regions one after another. Returns
    `{region: {band: held-out injections}}` for `acceptance` (the command line discards it)."""
    names = regions or [r.name for r in regions_module.REGIONS]
    t0 = time.time()
    held = {}
    for region in names:
        with progress_module.Stage("sky.derived.pacs_completeness", region) as st:
            ra, dec = _region_inputs(config, region)
            path = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
            bands = _build_region(config, region, path, ra, dec, config.n_jobs, st, maps)
            held[region] = {b: bands[b]["held"] for b in BANDS}
            st.done(path, **{"cells_" + b: int(bands[b]["curve"].shape[0]) for b in BANDS})
    progress_module.summary("sky.derived.pacs_completeness", len(names), t0)
    return held


# ----------------------------------------------------------------------------
# acceptance (printed, by function call; nothing is written)

def _read_band(f, band):
    """One band of the product, as a dict of arrays, with the pooling matrix rebuilt from its cells."""
    g = lambda name: f[name][:]
    d = dict(curve=g("CURVE_" + band), mag=g("MAG_GRID_" + band), flux90=g("FLUX90_%s_MJY" % band),
             n_pos=g("N_POS_" + band), scatter=g("SCATTER_%s_MJY" % band), scatter_blend=g("SCATTER_BLEND_%s_MJY" % band),
             bias=g("BIAS_%s_MJY" % band), sigma=g("CELL_SIGMA_%s_MJY" % band), offset=g("CELL_OFFSET_%s_MJY" % band),
             cell_id=g("CELL_ID_" + band), cell_map=g("CELL_MAP_INDEX_" + band), map_id=g("MAP_ID_" + band),
             map_wcs=g("MAP_WCS_" + band), map_shape=g("MAP_SHAPE_" + band), map_shift=g("MAP_SHIFT_%s_ARCSEC" % band),
             source_row=g("SOURCE_ROW")[:, BANDS.index(band)])
    d["flux"] = 1000.0 * ZERO_POINT_JY * 10.0 ** (-d["mag"].astype(np.float64) / 2.5)
    xy = np.zeros((d["cell_id"].size, 2))
    for m in range(d["map_id"].size):
        rows = np.flatnonzero(d["cell_map"] == m)
        if rows.size:
            grid = pacs_fit.MapGrid.from_wcs(WCS(fits.Header.fromstring(d["map_wcs"][m].decode())).celestial,
                                             tuple(d["map_shape"][m]))
            xy[rows] = cell_centres_arcsec(grid, d["cell_id"][rows]) + np.array([m * 1e7, 0.0])
    d["W"] = pool_matrix(xy)
    return d


def _binomial_error(fraction, n):
    """One-sigma binomial error of a fraction from n trials, the fraction held off 0 and 1 by half a trial."""
    p = np.clip(fraction, 0.5 / np.maximum(n, 1), 1 - 0.5 / np.maximum(n, 1))
    return np.sqrt(p * (1 - p) / np.maximum(n, 1))


def _acceptance_c1(band, d, held):
    n_row, n_plane = d["curve"].shape
    tot, got = pooled_counts(d["W"], held["row"], held["plane"].astype(np.int64), held["recovered"], n_row, n_plane)
    with np.errstate(all="ignore"):
        fh = np.where(tot > 0, got / tot, np.nan)
    fb = d["curve"].astype(float) / STORAGE_SCALE
    use = (tot >= MIN_INJECTIONS) & (d["n_pos"][:, None] >= MIN_INJECTIONS)
    diff = np.abs(fb - fh)
    sig = _binomial_error(fh, tot)
    sig_both = np.sqrt(sig ** 2 + _binomial_error(fb, d["n_pos"][:, None].astype(float)) ** 2)
    n = int(use.sum())
    if n == 0:
        print("  C1 %s: no (cell, plane) has %d injections in both halves" % (band, MIN_INJECTIONS))
        return
    within_error, within_abs = use & (diff <= sig), use & (diff <= 0.03)
    print("  C1 %s: %d (cell, plane) with >= %d injections in both halves (%.1f %% of all): within the held-out half's "
          "binomial error %.1f %%, within 0.03 absolute %.1f %%, both %.1f %%; within the two halves' combined "
          "error %.1f %% (about 68 %% for an exact curve)"
          % (band, n, MIN_INJECTIONS, 100 * n / diff.size, 100 * within_error.sum() / n, 100 * within_abs.sum() / n,
             100 * (within_error & within_abs).sum() / n, 100 * (use & (diff <= sig_both)).sum() / n))
    worst = np.argsort(np.where(use, diff / np.maximum(sig, 1e-9), -1).ravel())[::-1][:5]
    for k in worst:
        r, p = divmod(int(k), n_plane)
        print("      worst: map %s cell %d plane m=%.2f: curve %.3f held-out %.3f (%d injections)"
              % (d["map_id"][d["cell_map"][r]].decode(), d["cell_id"][r], d["mag"][p], fb[r, p], fh[r, p], tot[r, p]))
    f90 = flux90(np.where(tot > 0, got / np.maximum(tot, 1), 0.0), d["flux"])
    both = np.isfinite(f90) & np.isfinite(d["flux90"])
    rel = np.abs(f90 - d["flux90"]) / d["flux90"]
    ok = both & (tot[:, -1] >= MIN_INJECTIONS)
    print("      FLUX90 of the two halves within 10 %%: %.1f %% of %d cells (median difference %.1f %%)"
          % (100 * np.mean(rel[ok] <= 0.10) if ok.any() else np.nan, int(ok.sum()), 100 * np.median(rel[ok]) if ok.any() else np.nan))
    for k in np.argsort(np.where(ok, rel, -1))[::-1][:5]:
        if ok[k]:
            print("      worst FLUX90: map %s cell %d: %.1f vs %.1f mJy"
                  % (d["map_id"][d["cell_map"][k]].decode(), d["cell_id"][k], d["flux90"][k], f90[k]))


def _acceptance_c2(band, d, held):
    row, plane = held["row"], held["plane"].astype(np.int64)
    rec = held["recovered"]
    flux = d["flux"][plane]
    median_sigma = np.array([np.median(d["sigma"][d["cell_map"] == m]) if (d["cell_map"] == m).any() else np.nan
                             for m in range(d["map_id"].size)])
    noisy = d["sigma"] > 2 * median_sigma[d["cell_map"]]
    above = rec & np.isfinite(d["flux90"][row]) & (flux > d["flux90"][row])
    for path, name, table in ((1, "aperture", d["scatter"]), (2, "group fit", d["scatter_blend"])):
        for label, cls in (("quiet cells", ~noisy[row]), ("noisy cells (CELL_SIGMA > 2 x map median)", noisy[row])):
            sel = above & (held["path"] == path) & cls
            allrec = rec & (held["path"] == path) & cls & np.isfinite(table[row, plane])
            if sel.sum() < 30:
                print("  C2 %s %s, %s: %d recovered injections above FLUX90 (too few)" % (band, name, label, int(sel.sum())))
                continue
            bias = np.median(held["dflux"][sel] / flux[sel])
            z = held["dflux"][allrec] / table[row, plane][allrec]
            za = held["dflux"][sel & np.isfinite(table[row, plane])] / table[row, plane][sel & np.isfinite(table[row, plane])]
            robust = lambda v: MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))) if v.size else np.nan
            print("  C2 %s %s, %s: %d recovered above FLUX90, median (recovered - injected)/injected %+.2f %%; "
                  "(recovered - injected)/SCATTER: %d injections, robust scatter %.3f (above FLUX90 %.3f), median %+.3f"
                  % (band, name, label, int(sel.sum()), 100 * bias, int(allrec.sum()), robust(z), robust(za), np.median(z) if z.size else np.nan))


def _inverted_tile(piece, config, records):
    """C3 on one map tile: the detection rule applied to the negated map, at random positions and at the
    SESNA positions the photometry measures there, next to the real map's detection counts."""
    t, inp = piece
    tile = pacs_fit.cached_tile(config, records[t])
    inj = Injector(tile, inp, t + 1)
    n_row = inj.ids.size
    ok_real = singleton_pass(tile, inp["src_ra"], inp["src_dec"], inp["shift"], inp["src_sigma"], inp["src_offset"])[3] \
        if inp["src_ra"].size else np.zeros(0, bool)
    n_det, n_undet = inj.n_detected, int(ok_real.sum()) - inj.n_detected
    x, y = inj._draw(np.random.default_rng(t), np.arange(n_row), 40000)
    v = inj._valid(x, y)
    x, y = x[v], y[v]
    cid = tile.grid.cell_ids(x, y)
    with tile.inverted():
        m = tile.measure(x, y)
        random_pass = m.covered & (m.flux - inj.offset[cid] >= pacs_fit.DETECTION_SNR * inj.sigma[cid])
        n_random = int(m.covered.sum())
        if inp["src_ra"].size:
            _, _, _, ok_inv, det_inv = singleton_pass(tile, inp["src_ra"], inp["src_dec"], inp["shift"],
                                                      inp["src_sigma"], inp["src_offset"])
        else:
            ok_inv, det_inv = np.zeros(0, bool), np.zeros(0, bool)
    return dict(map_id=records[t].map_id, n_random=n_random, random_pass=int(random_pass.sum()),
                n_sesna=int(ok_inv.sum()), sesna_pass=int(det_inv.sum()), n_undetected=n_undet, n_detected=n_det)


def _acceptance_c3(config, region, band, f, d, records, n_jobs):
    ra, dec = _region_inputs(config, region)
    index = {r.map_id: r for r in records if r.band == band}
    maps = [index[m.decode()] for m in d["map_id"]]
    inputs = tile_inputs(d["cell_map"], d["cell_id"], d["sigma"], d["offset"], d["source_row"].astype(np.int64),
                         d["map_shift"], ra, dec)
    res = pieces_module.run(functools.partial(_inverted_tile, config=config, records=maps),
                            sorted(inputs.items()), n_jobs)
    for r in res:
        rate_r = r["random_pass"] / max(r["n_random"], 1)
        rate_s = r["sesna_pass"] / max(r["n_sesna"], 1)
        print("  C3 %s %s: inverted map: %d random positions, %.4f %% pass the rule; %d SESNA positions, %.4f %% pass; "
              "covered undetected %d, detected %d: rate x undetected / detected = %.4f (random), %.4f (SESNA positions)"
              % (band, r["map_id"], r["n_random"], 100 * rate_r, r["n_sesna"], 100 * rate_s, r["n_undetected"],
                 r["n_detected"], rate_r * r["n_undetected"] / max(r["n_detected"], 1),
                 rate_s * r["n_undetected"] / max(r["n_detected"], 1)))


def acceptance(config, region, held_out):
    """Prints C1-C3 of PACS_COMPLETENESS_PLAN.md for the region's product and the held-out half kept from the
    build that wrote it (`build`'s return value for the region). C1: the stored curve and 0.9 flux against the
    held-out half, per band. C2: the held-out recovered injections' bias and scatter against the stored
    SCATTER. C3: the detection rule on the negated map. Nothing is written."""
    path = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
    records = pacs_fit.read_index(config, region)
    with h5py.File(path, "r") as f:
        for band in BANDS:
            held = held_out.get(band)
            if held is None or f["CURVE_" + band].shape[0] == 0:
                continue
            d = _read_band(f, band)
            print("acceptance %s %s: %d cells, %d held-out injections" % (region, band, d["curve"].shape[0], held["row"].size), flush=True)
            _acceptance_c1(band, d, held)
            _acceptance_c2(band, d, held)
            _acceptance_c3(config, region, band, f, d, records, config.n_jobs)


if __name__ == "__main__":
    run(build)
