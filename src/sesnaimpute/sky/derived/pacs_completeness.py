"""The Herschel PACS completeness of the SESNA sources, per region and band, MEASURED by recovering
artificial point sources (PACS_COMPLETENESS_PLAN.md v2; SPEC_PACS_FARIR_DRAFT.md sections 5 and 11.3).

`build(config, regions=None)` writes one product per region,
`sky/derived/pacs/completeness_pacs_source__<Region>.hdf5`, in EXACTLY the layout of
`catalog/sesna/completeness_sesna_source__<Region>.hdf5` for the bands P70, P100 and P160
(`CURVE_<B>`, `MAG_GRID_<B>`, `SOURCE_ROW`, `ENVELOPE_<B>`, `COMPLETED_FROM_<B>`, `N_POS_<B>`), plus per-cell
columns aligned with the CURVE rows (`CELL_RA/DEC/ID/MAP_INDEX/MAP_<B>`, `CELL_SIGMA_<B>_MJY`,
`CELL_OFFSET_<B>_MJY`, `N_SIGMA_POS_<B>`, `FLUX90_<B>_MJY`, `BIAS_<B>_MJY`, `SCATTER_<B>_MJY`, `SKY_BIN_<B>`) that
the photometry stage reads; and the region's PACS source table, `sources_pacs_region__<Region>.hdf5` (RULINGS_5).

Method. Nothing here is derived from a noise model: completeness is the recovered fraction of artificial
point sources injected into the real map and found by the catalogue's own blind detection. No SESNA source enters
the completeness; SESNA enters only the photometry's cross-match.

1. The detection's inputs. Per 30-arcsec cell of the mosaic, 200 positions are drawn uniformly and measured
   with the catalogue's aperture photometry (`pacs_fit`). CELL_OFFSET is the median flux over the cell's clean
   positions (sky structure the aperture's sky leaves), CELL_SIGMA the median over the cell's 3 x 3 block of
   1.4826 x the median absolute deviation about each cell's offset.
2. The blind detection (`_block_peaks`). The aperture-filtered map (the estimator at every pixel, minus the cell's
   offset); a PACS source is a local maximum (8 neighbours) reaching 5 x CELL_SIGMA, refined by the 3 x 3 centroid,
   its flux the estimator there; the same on the negated map gives the spurious rate per sky bin. The matching
   radius is MATCH_SIGMAS x the rms offset of a found position, injected sources and the real sources' alignment
   against SESNA in quadrature (`_calibrate`, `_assemble_peaks`).
3. Injection, in signal-to-noise. For each plane of the S/N grid (`snr_grid`: 0.3 to 1000 x CELL_SIGMA, 0.1 dex),
   batches of artificial sources (the band's point-spread template, held to the release EEF's total-flux
   normalisation by `pacs_fit.Tile`) of flux S/N x CELL_SIGMA of the cell each lands in are added to a copy of the
   map at positions uniform over the tile's valid pixels inside the region's cells, no two closer than the injected
   patch's half-width plus the sky annulus's outer radius (no injected light in another's aperture or annulus).
   RECOVERED: the blind detection, run on the pixels around the injection (`window_peaks`), finds a source within
   the matching radius; where a source of the real map lies within the radius, only if the injection is the
   brighter. Batches alternate by seed parity: even seeds build the product, odd seeds are the held-out half kept in
   memory for `acceptance`.
4. Pooling in S/N. Recovery depends on flux over the cell's noise, so a cell's counts at an S/N plane are the sums
   over the cells of its map within POOL_ARCSEC (widened where short), whatever their noise. CURVE at flux F is the
   pooled fraction read at F / the cell's own CELL_SIGMA (linear in log S/N): the shape is the neighbourhood's, the
   scale and offset the cell's own. FLUX90 is the faintest flux at which it still reaches 0.9; BIAS and SCATTER the
   median and 1.4826 x MAD of (forced flux at the injected position - injected flux) / CELL_SIGMA over all pooled
   covered injections farther than BLEND_RADIUS_BEAMS from every PACS source of the real map, detected or not, read
   the same way and multiplied by the cell's CELL_SIGMA.

Each band's only map is its co-added mosaic of the region (`sky.derived.pacs_mosaic`, one map per band: every input map
resampled onto one grid, so no position is detected or counted twice); the detection kernel is the mosaic's weight-mean
point-spread function and the injections use the local mixture of the input maps' functions at each position
(`pacs_psf.MosaicPSF`). `build(..., box=)` runs a slice of it. Every covered cell of the mosaic inside the region's padded
source bounding box has a CURVE row (rows ordered by cell); SOURCE_ROW points each source at its cell. The observation
ids of the mosaic's inputs are in MAP_OBSID_<B> (one row); there is no per-cell observation id. The flux grid is one per band for the
whole survey: m = -2.5 log10(F / 1 Jy) from MAG_MIN to MAG_MAX in MAG_STEP, bright to faint. A region a band
does not cover gets the same product with no rows and SOURCE_ROW -1.

Run: `PY sesnaimpute.sky.derived.pacs_completeness $CONFIG [--regions ...]`.
"""

import functools
import os
import pickle
import time

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.sparse import csr_matrix
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived import pacs_fit
from sesnaimpute.sky.derived.pacs_photometry import calibrated_sigma
from sesnaimpute.sky.derived import pacs_psf
from sesnaimpute.definitions import COMPLETENESS_STORAGE_SCALE as STORAGE_SCALE
from sesnaimpute.definitions import PACS_APERTURES as APERTURES
from sesnaimpute.definitions import PACS_BANDS as BANDS
from sesnaimpute.definitions import PACS_BLEND_RADIUS_BEAMS as BLEND_RADIUS_BEAMS
from sesnaimpute.definitions import PACS_BLOCK_CELLS as BLOCK_CELLS
from sesnaimpute.definitions import PACS_MAX_EXTRA_BATCHES as MAX_EXTRA_BATCHES
from sesnaimpute.definitions import PACS_MIN_INJECTIONS as MIN_INJECTIONS
from sesnaimpute.definitions import PACS_POOL_ARCSEC as POOL_ARCSEC
from sesnaimpute.definitions import PACS_POOL_WIDENINGS as POOL_WIDENINGS
from sesnaimpute.definitions import PACS_SNR_BIN_CENTRES as SNR_BIN_CENTRES
from sesnaimpute.definitions import PACS_SNR_BIN_EDGES as SNR_BIN_EDGES
#: positions per cell (spec section 5)
N_POS = 200
#: a cell needs this many valid positions to get a curve; fewer means the cell is not covered
MIN_VALID_POSITIONS = 100
#: PACS has no Vega system: zero point 1 Jy, m = -2.5 log10(F / 1 Jy)
ZERO_POINT_JY = 1.0
#: ONE grid per band for the whole survey, bright to faint (6 mJy is below the 5-sigma flux of the deepest
#: cell, 25 Jy above the roll-off of every cell)
MAG_MIN = -3.5
MAG_MAX = 5.5
MAG_STEP = 0.25
MAD_TO_SIGMA = 1.4826
#: the map's reference noise (the fallback for a cell with no clean neighbour) is the median over up to this
#: many evenly spread cells of the tile
REFERENCE_CELLS = 400

#: injections: no two artificial sources of a batch closer than the injected patch's half-width plus the sky
#: annulus's outer radius (so no injected light reaches another's aperture or annulus), drawn at one per this
#: many squared separations of valid area
SEPARATION_AREAS = 2.0
#: artificial sources are injected in signal-to-noise: at plane k each source's flux is SNR_GRID[k] times the
#: CELL_SIGMA of the cell it lands in, so a plane is one point on the detection curve's own axis for every cell.
#: log10 S/N from SNR_LOG_MIN to SNR_LOG_MAX in SNR_LOG_STEP (0.1 dex, the flux grid's 0.25 mag)
SNR_LOG_MIN = -0.5
SNR_LOG_MAX = 3.0
SNR_LOG_STEP = 0.1
#: the first round of batches per plane and half aims at this multiple of MIN_INJECTIONS in the pooled
#: interior; further batches, restricted to the cells whose pooled count is short, follow until none is
INTERIOR_SAFETY = 1.3
#: the recovered fraction FLUX90 is read at
FLUX90_FRACTION = 0.9


def snr_bin(snr):
    """The error bin (0-3) of an injected S/N."""
    return np.digitize(np.asarray(snr, float), SNR_BIN_EDGES)
#: the cells of each map are ranked by their annulus sky and cut into this many equal bins (the spurious rate); a
#: bin's value pools the faintest S/N plane over every map of the band (PACS maps have no absolute zero level,
#: so the sky is compared within a map)
SKY_BINS = 5


def snr_grid():
    """The S/N planes the artificial sources are injected at, faint to bright."""
    n = int(round((SNR_LOG_MAX - SNR_LOG_MIN) / SNR_LOG_STEP)) + 1
    return 10.0 ** (SNR_LOG_MIN + SNR_LOG_STEP * np.arange(n))


def canonical_grid():
    """(magnitude planes, flux planes in mJy), bright to faint."""
    n = int(round((MAG_MAX - MAG_MIN) / MAG_STEP)) + 1
    mag = MAG_MIN + MAG_STEP * np.arange(n)
    return mag, 1000.0 * ZERO_POINT_JY * 10.0 ** (-mag / 2.5)


# ----------------------------------------------------------------------------
# the cell statistic

def cell_fits(tile, ids, seed):
    """The catalogue's aperture measurement at N_POS uniform positions in each cell of `ids` on `tile`. Returns
    b (flux mJy), gain (1 where covered), the annulus sigma, valid, clean (farther than one beam from every bright
    neighbour) and the annulus sky (mJy per pixel), each (n_cell, N_POS)."""
    x, y = tile.grid.sample_pixels(ids, N_POS, seed)
    a = tile.aperture(x.ravel(), y.ravel())
    shape = x.shape
    gain = np.where(a.covered, 1.0, np.nan)
    valid = (a.covered & np.isfinite(a.flux)).reshape(shape)
    clean = ~tile.near_bright(x, y)
    return (a.flux.reshape(shape), gain.reshape(shape), a.annulus_sigma.reshape(shape), valid, clean,
            a.sky.reshape(shape))


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
    """Per map tile, one pieces.run piece each, before the cell blocks: its median cell sigma over up to REFERENCE_CELLS cells spread evenly over
    the cells the region USES there (wholly on valid data), the fallback sigma of a block with no clean
    cell. `piece = (tile index, used cells, SESNA ra, dec on the tile)`."""
    t, used, _sra, _sdec = piece
    tile = pacs_fit.cached_tile(config, records[t])
    # the PACS products keep Herschel's own astrometry: no registration to SESNA (the catalogue's cross-match fits
    # the offset of each observation itself)
    shift = (0.0, 0.0, 0)
    centres = used[np.unique(np.linspace(0, used.size - 1, min(REFERENCE_CELLS, used.size)).astype(np.int64))]
    all_ids = ring(tile.grid, centres)
    b, gain, ps, valid, clean, _sky = cell_fits(tile, all_ids, 0)
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
    b, gain, ps, valid, clean, sky = cell_fits(tile, all_ids, tile_index + 1)
    # the detection map's own level and noise per cell, from its pixels
    m_off, m_mad, m_clean = detection_cell_stats(tile, all_ids)
    n_valid, n_clean, mad, g, pps, off = cell_statistics(b, gain, ps, valid, clean)
    sigma, n_sigma = block_sigma(tile.grid, ids, all_ids, mad, n_clean)
    offset = cell_offset(tile.grid, ids, all_ids, off, n_clean)
    ref = refs[tile_index]
    # a block with no clean cell (wholly within a beam of bright sources) takes the tile's median noise;
    # N_SIGMA_POS 0 marks it
    sigma = np.where(np.isfinite(sigma), sigma, ref["median"])
    m_sigma, _ = block_sigma(tile.grid, ids, all_ids, m_mad, m_clean)
    with np.errstate(all="ignore"):
        m_sigma = np.where(np.isfinite(m_sigma), m_sigma, np.nanmedian(m_sigma) if np.isfinite(m_sigma).any() else np.nan)
    m_offset = cell_offset(tile.grid, ids, all_ids, m_off, m_clean)
    k = np.searchsorted(all_ids, ids)
    n_valid = n_valid[k]
    with np.errstate(all="ignore"):
        cell_sky = np.nanmedian(np.where(valid, sky, np.nan), axis=1)[k]
    covered = (n_valid >= MIN_VALID_POSITIONS) & np.isfinite(sigma) & (sigma > 0)
    ra, dec = pacs_fit.from_map_frame(*tile.grid.centres_icrs(ids), ref["shift"][:2])
    return dict(tile=tile_index, ids=ids, covered=covered,
                n_sigma=np.minimum(n_sigma, np.iinfo(np.int16).max).astype(np.int16),
                sigma=sigma, offset=offset, sky=cell_sky, ra=ra, dec=dec, m_sigma=m_sigma, m_offset=m_offset)


# ----------------------------------------------------------------------------
# pooling

def cell_centres_arcsec(grid, ids):
    """(n, 2) arcsec position of the centre of each cell of `ids` in the map's own pixel frame."""
    x0, x1, y0, y1 = grid.bounds(ids)
    return np.c_[(x0 + x1) / 2 * (pacs_fit.CELL_ARCSEC / grid.step_x), (y0 + y1) / 2 * (pacs_fit.CELL_ARCSEC / grid.step_y)]


def pool_matrices(centres_arcsec, widenings=POOL_WIDENINGS):
    """The pooling matrices for radius POOL_ARCSEC x each of `widenings`: sparse (n, n) 0/1, entry (i, j) 1 where
    cell j's centre lies within the radius of cell i's (a cell pools with itself). No noise window: the counts are
    pooled in S/N, where cells of different noise share one curve. Cells of different maps must be given far-apart
    centres."""
    n = len(centres_arcsec)
    tree = cKDTree(centres_arcsec)
    out = []
    for k in widenings:
        pairs = tree.query_pairs(k * POOL_ARCSEC, output_type="ndarray").reshape(-1, 2)
        r = np.concatenate([np.arange(n), pairs[:, 0], pairs[:, 1]])
        c = np.concatenate([np.arange(n), pairs[:, 1], pairs[:, 0]])
        out.append(csr_matrix((np.ones(r.size), (r, c)), shape=(n, n)))
    return out


def pooled_counts(Ws, row, plane, recovered, n_row, n_plane):
    """The pooled number of injections and of recovered injections at every (cell, plane): sums over the cells
    pooled with each (`row` the injection's cell row, `plane`, `recovered`), on the narrowest of the matrices
    `Ws` whose total reaches MIN_INJECTIONS, else the widest."""
    key = row.astype(np.int64) * n_plane + plane
    tot = np.bincount(key, minlength=n_row * n_plane).reshape(n_row, n_plane).astype(np.float64)
    rec = np.bincount(key[recovered], minlength=n_row * n_plane).reshape(n_row, n_plane).astype(np.float64)
    pt, pr = Ws[-1] @ tot, Ws[-1] @ rec
    for W in reversed(Ws[:-1]):
        t, r = W @ tot, W @ rec
        use = t >= MIN_INJECTIONS
        pt, pr = np.where(use, t, pt), np.where(use, r, pr)
    return pt, pr


def _pooled_median_scatter_one(W, row, value, n_row, min_n):
    """The pooled median and scatter (`pooled_median_scatter`) of each row of `W` (m rows over the n_row cells)."""
    n_out = W.shape[0]
    out_med = np.full(n_out, np.nan)
    out_sc = np.full(n_out, np.nan)
    if row.size == 0:
        return out_med, out_sc
    order = np.argsort(row, kind="stable")
    r, v = row[order], value[order].astype(np.float64)
    count = np.bincount(r, minlength=n_row)
    start = np.cumsum(count) - count
    pair_c = np.repeat(np.arange(n_out), np.diff(W.indptr))
    pair_n = W.indices
    k = count[pair_n]
    total = int(k.sum())
    if total == 0:
        return out_med, out_sc
    seg = np.repeat(pair_c, k)
    within = np.arange(total) - np.repeat(np.cumsum(k) - k, k)
    vals = v[np.repeat(start[pair_n], k) + within]
    n_c = np.bincount(seg, minlength=n_out)
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


def pooled_median_scatter(Ws, row, value, n_row, min_n=MIN_INJECTIONS):
    """Per cell, the median and 1.4826 x the median absolute deviation about it of the `value`s of the records
    (at cell `row`) of the cells pooled with it, on the narrowest of `Ws` that holds at least `min_n` records;
    NaN where none does. The rows of `Ws` are the output cells (all `n_row` of the records' cells, or a block's cells
    with the records of the block's whole neighbourhood: `n_row` then counts the cells the records' `row` index)."""
    n_out = Ws[0].shape[0]
    med = np.full(n_out, np.nan)
    sc = np.full(n_out, np.nan)
    for W in Ws:
        # a wider radius only for the cells the narrower ones left without a value
        need = np.flatnonzero(np.isnan(med))
        if need.size == 0:
            break
        m, s_ = _pooled_median_scatter_one(W[need], row, value, n_row, min_n)
        fill = np.isfinite(m)
        med[need[fill]], sc[need[fill]] = m[fill], s_[fill]
    return med, sc


def _snr_brackets(sigma_rows, flux_grid):
    """For every (row, flux plane): the two S/N planes bracketing F / CELL_SIGMA of the row, the weight of the
    upper one (linear in log S/N; held at the grid's ends), and F / CELL_SIGMA over the grid's largest S/N
    (> 1 only beyond the bright end)."""
    n_x = snr_grid().size
    with np.errstate(all="ignore"):
        lx = np.log10(flux_grid[None, :] / np.asarray(sigma_rows, float)[:, None])
    t = np.clip((lx - SNR_LOG_MIN) / SNR_LOG_STEP, 0.0, n_x - 1.0)
    t = np.where(np.isfinite(t), t, 0.0)
    k0 = np.minimum(np.floor(t).astype(np.int64), n_x - 2)
    w = t - k0
    beyond = np.maximum(10.0 ** (lx - SNR_LOG_MAX), 1.0)
    return k0, w, np.where(np.isfinite(beyond), beyond, 1.0)


def _at_flux(values_x, k0, w):
    """`values_x` (n_row, n_snr) read at the bracketing S/N planes of every (row, flux plane)."""
    r = np.arange(values_x.shape[0])[:, None]
    lo, hi = values_x[r, k0], values_x[r, k0 + 1]
    with np.errstate(invalid="ignore"):
        mid = (1.0 - w) * lo + w * hi
    return np.where(w <= 0.0, lo, np.where(w >= 1.0, hi, mid))


def curve_on_flux_grid(Ws, row, plane, recovered, sigma_rows, flux_grid, n_row):
    """The recovered fraction on the flux grid from injections made in S/N: per S/N plane the pooled counts of the
    cells within the pooling radius (`pooled_counts`), read for each row at F / its own CELL_SIGMA. Returns
    (fraction, injections behind it: the smaller of the two bracketing planes' pooled counts), (n_out, n_plane), the
    rows of `Ws` being the output cells and `sigma_rows` theirs (all `n_row` cells, or a block's within its
    neighbourhood's records)."""
    n_x = snr_grid().size
    tot, got = pooled_counts(Ws, row, plane.astype(np.int64), recovered, n_row, n_x)
    with np.errstate(all="ignore"):
        fx = np.where(tot > 0, got / tot, np.nan)
    k0, w, _ = _snr_brackets(sigma_rows, flux_grid)
    fraction = _at_flux(fx, k0, w)
    r = np.arange(tot.shape[0])[:, None]
    count = np.where(w <= 0.0, tot[r, k0], np.where(w >= 1.0, tot[r, k0 + 1], np.minimum(tot[r, k0], tot[r, k0 + 1])))
    return np.nan_to_num(fraction, nan=0.0), count


def error_on_flux_grid(Ws, row, plane, dflux, sigma_rows, flux_grid, n_row, min_n=MIN_INJECTIONS, sigma_out=None):
    """BIAS-like and SCATTER-like arrays on the flux grid from injections made in S/N: per S/N plane, the pooled
    median and 1.4826 x MAD of (recovered - injected flux) / CELL_SIGMA of the injection's own cell, read for each
    row at F / its own CELL_SIGMA and multiplied by that CELL_SIGMA; beyond the grid's brightest S/N both grow in
    proportion to the flux (the brightest plane's value relative to its flux). (n_out, n_plane) each; `sigma_rows` the
    noise of the cells the records' `row` index, `sigma_out` that of the output cells (the rows of `Ws`; default all)."""
    n_x = snr_grid().size
    sig = np.asarray(sigma_rows, float)
    sig_out = sig if sigma_out is None else np.asarray(sigma_out, float)
    med_x = np.full((Ws[0].shape[0], n_x), np.nan)
    sc_x = np.full((Ws[0].shape[0], n_x), np.nan)
    z = dflux.astype(np.float64) / sig[row]
    for p in range(n_x):
        sel = plane == p
        med_x[:, p], sc_x[:, p] = pooled_median_scatter(Ws, row[sel], z[sel], n_row, min_n)
    k0, w, beyond = _snr_brackets(sig_out, flux_grid)
    scale = sig_out[:, None] * beyond
    return (_at_flux(med_x, k0, w) * scale).astype(np.float32), (_at_flux(sc_x, k0, w) * scale).astype(np.float32)


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

# ----------------------------------------------------------------------------
# blind detection (RULINGS_5): the catalogue's PACS sources are the peaks of the aperture-filtered map

#: the detection kernel's footprint: where the template exceeds this fraction of its peak
ZERO_SUM_FOOTPRINT = 0.01
#: the matching radius is MATCH_SIGMAS x the rms offset of a detected position: the injections at CALIBRATION_SNR in
#: quadrature with the alignment scatter of the map's registration pairs (`pacs_fit.registration_pairs`), the map's own
#: where it has enough pairs, else the band's pooled
MATCH_SIGMAS = 3.0
CALIBRATION_SNR = (30.0, 100.0)


def filtered(tile, x, y, offset_cell, image=None, row0=0):
    """The catalogue's flux at map pixels (x, y): the aperture estimator (aperture minus annulus sky, `Tile.aperture`)
    minus the CELL_OFFSET of the position's cell (`offset_cell`, per grid cell, NaN where the cell holds no row); NaN
    where not covered. `image`, `row0`: the buffer of rows to measure on (`Tile.aperture`)."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    cid = tile.grid.cell_ids(x, y)
    off = np.where(cid >= 0, offset_cell[np.maximum(cid, 0)], np.nan)
    a = tile.aperture(x, y, image=image, row0=row0)
    return np.where(a.covered, a.flux - off, np.nan)


def _detection_kernel(tile):
    """The detection kernel of the tile: the band's own point-spread template (`Tile._add_sources`'s, at the map
    centre; a mosaic's weight-mean table, `MosaicPSF.table`, not the local mixture at the centre pixel) on its footprint (where it exceeds ZERO_SUM_FOOTPRINT of its peak). Returns (h, kc, fc, s_full): the
    half-width, the template on the footprint and the footprint as correlated (flipped), and the zero-sum template's
    sum of squares over the footprint."""
    h = tile.injection_half_width()
    side = 2 * h + 1
    cy, cx = tile.shape[0] // 2, tile.shape[1] // 2
    offs = np.arange(side, dtype=float)
    px = np.broadcast_to((cx - h + offs)[None, None, :], (1, side, side))
    py = np.broadcast_to((cy - h + offs)[None, :, None], (1, side, side))
    psf = tile._psf
    evaluate = psf.evaluate_mean_xy if isinstance(psf, pacs_psf.MosaicPSF) else psf.evaluate_xy
    k = evaluate(np.array([float(cx)]), np.array([float(cy)]), px, py).mean(axis=1)[0]
    foot = k > ZERO_SUM_FOOTPRINT * k.max()
    kf = np.where(foot, k, 0.0)
    s_full = float(((k[foot] - k[foot].mean()) ** 2).sum())
    return h, kf[::-1, ::-1], foot[::-1, ::-1].astype(float), s_full


#: the detection image's strips: the stage cuts a map into at least this many strips per worker (see `detection_strips`)
STRIPS_PER_WORKER = 4

_DETECTION_LAYERS = (("m", np.float32), ("ratio", np.float32), ("norm", np.float32), ("valid", bool), ("pix_valid", bool))


def detection_strips(tile, n_jobs=1):
    """The strips `prepare_detection` computes (`(first row, last row + 1)`): at most PREPARE_STRIP_ROWS rows (a worker's
    memory is a strip of the map's width, whatever the map's size), cut finer so that the map gives at least
    STRIPS_PER_WORKER x `n_jobs` strips where it is tall enough, and never fewer than 8 kernel half-widths (the strip's
    two overlap margins cost at most the strip itself)."""
    h = tile.injection_half_width()
    ny = tile.shape[0]
    rows = max(8 * h, min(pacs_fit.PREPARE_STRIP_ROWS, -(-ny // (STRIPS_PER_WORKER * max(int(n_jobs), 1)))))
    return [(s0, min(s0 + rows, ny)) for s0 in range(0, ny, rows)]


def open_detection(tile, config):
    """The first step of `prepare_detection`: the kernel written beside the prepared layers and the layers created
    (memory-mapped, to be filled strip by strip by `detection_strip`). Returns the strips."""
    from numpy.lib.format import open_memmap
    d = pacs_fit.prepared_dir(config, tile.record)
    h, kc, fc, s_full = _detection_kernel(tile)
    np.savez(os.path.join(d, "kernel.npz"), h=h, kc=kc, fc=fc, s_full=s_full)
    for name, dt in _DETECTION_LAYERS:
        open_memmap(os.path.join(d, name + ".npy"), mode="w+", dtype=dt, shape=tile.shape).flush()
    return detection_strips(tile, config.n_jobs)


def detection_strip(tile, config, s0, s1):
    """One strip of `prepare_detection`, rows s0..s1 written into the layers (`open_detection`), computed with the
    kernel's half-width of rows either side; strips are independent, so any number run at once."""
    from numpy.lib.format import open_memmap
    d = pacs_fit.prepared_dir(config, tile.record)
    k = np.load(os.path.join(d, "kernel.npz"))
    h, kc, fc, s_full = int(k["h"]), k["kc"], k["fc"], float(k["s_full"])
    out = {name: open_memmap(os.path.join(d, name + ".npy"), mode="r+") for name, _ in _DETECTION_LAYERS}
    ny = tile.shape[0]

    def corr(a, c):
        return fftconvolve(a, c, mode="same")
    a0, a1 = max(s0 - h, 0), min(s1 + h, ny)
    image = np.asarray(tile.image[a0:a1])
    valid = np.isfinite(image) & np.isfinite(np.asarray(tile.error[a0:a1]))
    # D = valid pixels in the footprint, B = sum of the template over them, E = of its square, A and C the
    # image's sums weighted by the template and plain: the zero-sum response is A - B C / D, its kernel's sum of
    # squares E - B^2 / D (FFT round-off: D is a count, S_valid a sum of squares held >= 0)
    v = valid.astype(float)
    dd = np.rint(corr(v, fc))
    b = corr(v, kc)
    s_valid = corr(v, kc * kc)
    del v
    s_valid -= np.divide(b * b, dd, out=np.zeros_like(b), where=dd > 0)
    ok = valid & (dd >= 2) & (s_valid > 1e-9 * s_full)
    ratio = np.divide(b, dd, out=np.zeros_like(b), where=ok)
    del b, dd
    norm = np.sqrt(np.divide(s_full, s_valid, out=np.zeros_like(s_valid), where=ok))
    del s_valid
    iv = np.where(valid, image, 0.0).astype(float)
    m = corr(iv, kc)
    c = corr(iv, fc)
    del iv
    c *= ratio
    m -= c
    del c
    m *= norm
    m = np.where(ok, m, np.nan)
    r0, r1 = s0 - a0, s1 - a0
    out["m"][s0:s1] = m[r0:r1]
    out["ratio"][s0:s1] = ratio[r0:r1]
    out["norm"][s0:s1] = norm[r0:r1]
    out["valid"][s0:s1] = ok[r0:r1]
    out["pix_valid"][s0:s1] = valid[r0:r1]
    for a in out.values():
        a.flush()


def prepare_detection(tile, config):
    """The detection image of the tile's real image, written once per map beside its prepared layers
    (`pacs_fit.prepared_dir`): the image correlated with the band's own point-spread template made zero-sum over
    its footprint (the template less its mean there, DAOFIND's lowered kernel), so that a smooth local background
    and its gradient give zero and a point source on the slope of a ridge is still a peak. At map edges and gaps
    the kernel is the template on the footprint's VALID pixels made zero-sum over those pixels, scaled by
    sqrt(S_full / S_valid) (S the kernel's sum of squares over the footprint, over its valid pixels) so that white
    pixel noise gives the interior's noise there (the factor is 1 in the interior); defined at every valid pixel
    whose valid footprint holds at least two pixels. Written: `m.npy` (NaN where not defined), the per-pixel
    valid-footprint mean factor `ratio.npy` and scale `norm.npy` (float32, 0 where not defined), `valid.npy` (where
    m is defined), `pix_valid.npy` (the pixels that enter), `kernel.npz`. Computed strip by strip
    (`detection_strips`), each strip with the kernel's half-width of rows either side, so the memory is a few strips
    whatever the map's size; the stage runs the strips of all maps in parallel (`_prepare_stage`), this function
    runs them in this process."""
    for s0, s1 in open_detection(tile, config):
        detection_strip(tile, config, s0, s1)


_MATCHED = {}


def matched_state(tile):
    """The tile's detection image and its factors (`prepare_detection`), memory-mapped read-only: one copy per map
    shared by every worker. Process-local, one map at a time. `m` the detection map (NaN where not valid), `kc` the
    template on the footprint and `fc` the footprint as correlated, `ratio`, `norm`, `valid`, `pix_valid` as
    written (`_injected_delta` applies the identical operator to an injection's template)."""
    key = tile.record.path
    if key not in _MATCHED:
        _MATCHED.clear()
        d = os.path.dirname(tile._image_path)
        k = np.load(os.path.join(d, "kernel.npz"))
        _MATCHED[key] = dict(h=int(k["h"]), kc=k["kc"], fc=k["fc"],
                             **{name: np.load(os.path.join(d, name + ".npy"), mmap_mode="r")
                                for name in ("m", "ratio", "norm", "valid", "pix_valid")})
    return _MATCHED[key]


def _layer_strip(task, config):
    """Rows of one mosaic's prepared layers (`pacs_fit.mosaic_layer_strip`)."""
    record, a, b, floor, error_fraction = task
    pacs_fit.mosaic_layer_strip(config, record, a, b, floor, error_fraction)
    return record.map_id


def _finish_piece(record, config):
    """Finish one mosaic's preparation (`pacs_fit.finish_prepare`) and create its detection layers (`open_detection`).
    Returns `(record, its detection strips)`."""
    tile = pacs_fit.finish_prepare(config, record)
    strips = open_detection(tile, config)
    pacs_fit._CACHE.clear()
    return record, strips


def _strip_piece(task, config):
    """One detection strip of one map (`detection_strip`), the map's tile shared by the strips a worker takes."""
    record, s0, s1 = task
    detection_strip(pacs_fit.cached_tile(config, record), config, s0, s1)
    return record.map_id


def _prepare_stage(config, records, n_jobs, stage):
    """The band mosaics prepared for the run, every step in strips over all workers: the mosaic's layers in mJy (the
    coverage floor from a sample of rows, the error floor from each pixel's own weight, `pacs_fit.mosaic_layer_plan`; strips of PREPARE_STRIP_ROWS rows in parallel), the
    point-spread cache and detection layers (one task per mosaic), then the detection strips (STRIPS_PER_WORKER x
    n_jobs of them where the mosaic is tall enough, `detection_strips`)."""
    plans = [(r, pacs_fit.mosaic_layer_plan(config, r)) for r in records]
    tasks = [(r, a, b, fl, ef) for r, (fl, ef, strips) in plans for a, b in strips]
    pieces_module.run(functools.partial(_layer_strip, config=config), tasks, n_jobs, stage=stage, unit="strips (mosaic layers)")
    prepared = pieces_module.run(functools.partial(_finish_piece, config=config), records, n_jobs, stage=stage,
                                 unit="mosaics (prepared)")
    strips = [(r, s0, s1) for r, ss in prepared for s0, s1 in ss]
    pieces_module.run(functools.partial(_strip_piece, config=config), strips, n_jobs, stage=stage,
                      unit="strips (detection image)")


def detection_cell_stats(tile, ids):
    """Per cell of `ids`: the detection map's level (median) and noise (1.4826 x the median absolute deviation) over
    the cell's valid pixels, and how many pixels; the noise is the block median over the 3 x 3 cells (as CELL_SIGMA)."""
    ms = matched_state(tile)
    grid = tile.grid
    x0, x1, y0, y1 = grid.bounds(ids)
    level, mad, cnt = np.full(ids.size, np.nan), np.full(ids.size, np.nan), np.zeros(ids.size, np.int64)
    for i in range(ids.size):
        a0, a1 = max(int(np.ceil(x0[i])), 0), min(int(np.floor(x1[i])) + 1, tile.shape[1])
        b0, b1 = max(int(np.ceil(y0[i])), 0), min(int(np.floor(y1[i])) + 1, tile.shape[0])
        v = ms["m"][b0:b1, a0:a1]
        v = v[np.isfinite(v)]
        cnt[i] = v.size
        if v.size >= 20:
            level[i] = np.median(v)
            mad[i] = MAD_TO_SIGMA * np.median(np.abs(v - level[i]))
    return level, mad, cnt


def _local_peaks(v, thr):
    """8-neighbour local maxima (along the last two axes) of `v` that reach `thr`; NaN is never a peak."""
    vv = np.where(np.isfinite(v), v, -np.inf)
    size = (1,) * (v.ndim - 2) + (3, 3)
    mx = maximum_filter(vv, size=size, mode="constant", cval=-np.inf)
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (vv >= mx) & (v >= thr)


def _parabola(v, iy, ix):
    """The sub-pixel peak of the 2-D array `v` at its local maximum (iy, ix): a parabola through the pixel and its two
    neighbours on each axis (as `pacs_fit.Tile._find_neighbours`), as offsets (dy, dx) clipped to half a pixel."""
    c, l, r, d, u = v[iy, ix], v[iy, ix - 1], v[iy, ix + 1], v[iy - 1, ix], v[iy + 1, ix]
    with np.errstate(all="ignore"):
        dx = 0.5 * (l - r) / (l - 2 * c + r)
        dy = 0.5 * (d - u) / (d - 2 * c + u)
    return float(np.clip(np.nan_to_num(dy), -0.5, 0.5)), float(np.clip(np.nan_to_num(dx), -0.5, 0.5))


def _injected_delta(tile, x, y, flux, wpx, wpy):
    """The change of the detection map at the window pixels (wpx, wpy) (n, side, side) of each injection, from that
    injection's own template (the detection map is linear in the image: its change is the operator of `matched_state`,
    with each pixel's own valid-footprint factors, applied to the template on the valid pixels)."""
    ms = matched_state(tile)
    h = ms["h"]
    side = 2 * h + 1
    ny, nx = tile.shape
    offs = np.arange(side, dtype=float)
    cx = np.rint(x).astype(np.int64) - h
    cy = np.rint(y).astype(np.int64) - h
    out = np.zeros(wpx.shape)
    for lo in range(0, x.size, 256):
        sl = slice(lo, lo + 256)
        px = np.broadcast_to(cx[sl][:, None, None] + offs[None, None, :], (cx[sl].size, side, side))
        py = np.broadcast_to(cy[sl][:, None, None] + offs[None, :, None], (cx[sl].size, side, side))
        resp = tile._psf.evaluate_xy(x[sl], y[sl], px, py).mean(axis=1) * flux[sl][:, None, None]
        # only the injected light on valid pixels enters, as for the real image
        ipx, ipy = px.astype(np.int64), py.astype(np.int64)
        on = (ipx >= 0) & (ipx < nx) & (ipy >= 0) & (ipy < ny)
        pv = np.zeros(resp.shape, bool)
        pv[on] = ms["pix_valid"][ipy[on], ipx[on]]
        resp = np.where(pv, resp, 0.0)
        full_k = fftconvolve(resp, ms["kc"][None], mode="full", axes=(1, 2))
        full_f = fftconvolve(resp, ms["fc"][None], mode="full", axes=(1, 2))
        # full index of map pixel p: p - (patch origin) + h
        iy = wpy[sl] - cy[sl][:, None, None] + h
        ix = wpx[sl] - cx[sl][:, None, None] + h
        inside = (iy >= 0) & (iy < full_k.shape[1]) & (ix >= 0) & (ix < full_k.shape[2])
        inside &= (wpx[sl] >= 0) & (wpx[sl] < nx) & (wpy[sl] >= 0) & (wpy[sl] < ny)
        vals = np.zeros(iy.shape)
        k = np.nonzero(inside)
        my, mx = wpy[sl][k], wpx[sl][k]
        vals[k] = (full_k[k[0], iy[k], ix[k]] - ms["ratio"][my, mx] * full_f[k[0], iy[k], ix[k]]) * ms["norm"][my, mx]
        out[sl] = vals
    return out


def window_peaks(tile, x, y, radius_px, m_offset_cell, m_sigma_cell, offset_cell, flux=None, image=None, row0=0):
    """For each map position (x, y): the PACS source the blind detection finds within `radius_px` (one, or one per position) of it, on the
    PSF-matched map (`matched_state`, plus each injection's own change when `flux` is given: the map then holds those
    injections) on the pixels within radius_px + 1.5 (enough for the 8-neighbour test): a local maximum reaching 5 x
    the matched map's own cell noise above its cell offset (`m_sigma_cell`, `m_offset_cell`), the nearest one, its
    position by the parabolic sub-pixel peak, its flux the catalogue's aperture estimator there (`filtered`, on the
    tile's current image, or `image`, the buffer of rows from `row0` (`Tile.aperture`)). Returns (found, xc, yc, fnu, distance)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    out = (np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    if n == 0:
        return out
    ms = matched_state(tile)
    rad = np.broadcast_to(np.asarray(radius_px, float), (n,))[:, None, None]       # one radius, or one per position
    h = int(np.ceil(rad.max())) + 2
    off = np.arange(-h, h + 1)
    cx, cy = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
    px = np.ascontiguousarray(np.broadcast_to(cx[:, None, None] + off[None, None, :], (n, off.size, off.size)))
    py = np.ascontiguousarray(np.broadcast_to(cy[:, None, None] + off[None, :, None], (n, off.size, off.size)))
    dist = np.hypot(px - x[:, None, None], py - y[:, None, None])
    ny, nx = tile.shape
    use = (dist <= rad + 1.5) & (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    v = np.full(px.shape, np.nan)
    v[use] = ms["m"][py[use], px[use]]
    if flux is not None:
        v = v + _injected_delta(tile, x, y, np.broadcast_to(np.asarray(flux, float), x.shape), px, py)
    cid = tile.grid.cell_ids(px, py)
    good = cid >= 0
    v = np.where(good, v - m_offset_cell[np.maximum(cid, 0)], np.nan)
    thr = np.where(good, pacs_fit.DETECTION_SNR * m_sigma_cell[np.maximum(cid, 0)], np.nan)
    pk = _local_peaks(v, thr) & (dist <= rad)
    found, xc, yc, fnu, dd = out
    cand = np.where(pk, dist, np.inf).reshape(n, -1)
    best = np.argmin(cand, axis=1)
    has = np.isfinite(cand[np.arange(n), best])
    for i in np.flatnonzero(has):
        iy, ix = divmod(int(best[i]), off.size)
        dy, dx = _parabola(v[i], iy, ix)
        xc[i], yc[i] = px[i, iy, ix] + dx, py[i, iy, ix] + dy
    if has.any():
        fnu[has] = filtered(tile, xc[has], yc[has], offset_cell, image=image, row0=row0)
        dd[has] = np.hypot(xc[has] - x[has], yc[has] - y[has])
        found[has] = dd[has] <= rad[has, 0, 0]
    return found, xc, yc, fnu, dd


def _block_peaks(piece, config, records, cell_tables):
    """The blind detection on one block of cells of one map tile (`piece = (tile index, cell ids)`, the cell stage's
    blocks): the local maxima of the detection map (`matched_state`), less its cell level, reaching 5 x its cell
    noise, inside the block's cells; position by the parabolic sub-pixel peak; flux the catalogue's aperture
    estimator there. `cell_tables[t]` = (offset, sigma, detection level, detection noise) per grid cell. Returns
    (tile, dict(x, y, fnu, cell))."""
    t, cells = piece
    tile = pacs_fit.cached_tile(config, records[t])
    offset_cell, _sigma_cell, m_off, m_sig = cell_tables[t]
    ms = matched_state(tile)
    grid = tile.grid
    x0, x1, y0, y1 = grid.bounds(cells)
    ny, nx = tile.shape
    ix0, ix1 = max(int(np.floor(x0.min())) - 1, 0), min(int(np.ceil(x1.max())) + 1, nx - 1)
    iy0, iy1 = max(int(np.floor(y0.min())) - 1, 0), min(int(np.ceil(y1.max())) + 1, ny - 1)
    gx, gy = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1))
    cid = grid.cell_ids(gx, gy)
    core = np.isin(cid, cells)
    good = cid >= 0
    v = np.where(good, ms["m"][iy0:iy1 + 1, ix0:ix1 + 1] - m_off[np.maximum(cid, 0)], np.nan)
    thr = np.where(good, pacs_fit.DETECTION_SNR * m_sig[np.maximum(cid, 0)], np.nan)

    def detect(vv):
        pk = _local_peaks(vv, thr) & core
        pk[0, :] = pk[-1, :] = False
        pk[:, 0] = pk[:, -1] = False
        return np.argwhere(pk)

    where = detect(v)
    xs, ys = [], []
    for iy, ix in where:
        dy, dx = _parabola(v, iy, ix)
        xs.append(gx[iy, ix] + dx)
        ys.append(gy[iy, ix] + dy)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    fnu = filtered(tile, xs, ys, offset_cell) if xs.size else np.zeros(0)
    ok = np.isfinite(fnu)
    return t, dict(x=xs[ok], y=ys[ok], fnu=fnu[ok], cell=grid.cell_ids(xs[ok], ys[ok]))


class Injector:
    """Artificial point sources injected into one map tile and measured through the catalogue's own path.

    `inp` carries what the stage knows of the tile: its rows' cell ids (`ids`), their `sigma` and `offset`
    (the float32 values stored in the product), the PACS sources the blind detection found on the un-injected map
    (`peak_x`, `peak_y`, `peak_fnu`) and the matching radius in pixels (`r_match_px`). No SESNA source enters.
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
        self.m_sigma = np.full(grid.n_cells, np.nan)
        self.m_offset = np.full(grid.n_cells, np.nan)
        self.m_sigma[self.ids] = inp.get("m_sigma", np.full(n, np.nan))
        self.m_offset[self.ids] = inp.get("m_offset", np.full(n, np.nan))
        matched_state(tile)                 # the detection image of the un-injected map, before any injection
        self.beam_px = tile.beam_px              # the mosaic-wide mean; every radius uses `beam_px_at`
        self.separation_px = tile.injection_half_width() + APERTURES[tile.record.band][2] / tile.pixel_arcsec
        # the PACS sources of the un-injected map (the blind detection), and the matching radius
        self.pos_k_px = float(inp.get("pos_k_arcsec", np.nan)) / tile.pixel_arcsec
        self.det_tree = None
        self.det_xy = np.c_[np.asarray(inp.get("peak_x", np.zeros(0)), float), np.asarray(inp.get("peak_y", np.zeros(0)), float)]
        self.det_flux = np.asarray(inp.get("peak_fnu", np.zeros(0)), float)
        self.n_detected = int(self.det_flux.size)
        if self.n_detected:
            self.det_tree = cKDTree(self.det_xy)
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
        """Greedy selection, in order, of at most `limit` positions no two closer than the injected patch's
        half-width plus the sky annulus's outer radius: no injected light reaches another injection's aperture or
        annulus (closer pairs read up to 7 % high at 160 micron)."""
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
        """One batch's positions: uniform over the valid pixels of the cells of `rows`, aiming at one per
        SEPARATION_AREAS squared separations. Returns (x, y, distance in pixels to the nearest PACS source of the
        un-injected map, inf if none)."""
        target = max(int(round(rows.size * self.cell_px * self.valid_fraction / (SEPARATION_AREAS * self.separation_px ** 2))), 1)
        x, y = self._draw(rng, rows, int(2.5 * target) + 8)
        ok = self._valid(x, y)
        x, y = x[ok], y[ok]
        keep = self._thin(x, y, target)
        x, y = x[keep], y[keep]
        if self.det_tree is None:
            return x, y, np.full(x.size, np.inf)
        d, _ = self.det_tree.query(np.c_[x, y])
        return x, y, d

    def flux_at(self, x, y, snr):
        """The flux (mJy) of an artificial source of S/N `snr` at each (x, y): snr x CELL_SIGMA of its cell."""
        return snr * self.sigma[self.tile.grid.cell_ids(x, y)]

    def batch(self, rng, rows, snr):
        """Injects one batch of sources of S/N `snr` at random positions of the cells of `rows`, measures each
        through the catalogue's own path, restores the map (`measure_positions`)."""
        x, y, d = self.positions(rng, rows)
        return self.measure_positions(x, y, d, self.flux_at(x, y, snr), radius_px=self.recovery_radius_px(snr, x, y))

    def beam_px_at(self, x, y):
        """The local beam (pixels) at map pixel positions (`Tile.beam_px_at`)."""
        return self.tile.beam_px_at(x, y)

    def recovery_radius_px(self, snr, x, y):
        """The recovery radius at injection S/N `snr` at each map position (x, y): MATCH_SIGMAS x the detection's
        measured position scatter there (the scatter x S/N constant over S/N), between one pixel and half the local beam."""
        half_beam = 0.5 * self.beam_px_at(x, y)
        if not np.isfinite(self.pos_k_px):
            return half_beam
        return np.clip(MATCH_SIGMAS * self.pos_k_px / max(snr, 1e-3), 1.0, half_beam)

    def measure_positions(self, x, y, d, flux, radius_px=None):
        """Injects `flux` (mJy, scalar or one per position) at the map pixels (x, y), `d` the distance (pixels) of
        each to the nearest PACS source of the un-injected map, runs the blind detection around each
        (`window_peaks` within `radius_px`, the recovery radius at the batch's S/N, a scalar or one per position; half the
        local beam if None), restores the
        map. RECOVERED: a PACS source is found within the radius, and where a source of the un-injected map lies within it the injection is the
        brighter (an injection on an existing peak counts only as the brighter member). Also the forced aperture
        flux at the injected position (the photometric error, SCATTER). Returns dict(row, near (within one
        BLEND_RADIUS_BEAMS of a PACS source of the un-injected map), recovered, dflux (forced flux minus the cell
        offset minus the injected flux, mJy), flux (injected, mJy), det_flux (the found source's flux, NaN if none),
        det_dist (its distance in pixels)) for the injections on covered data; n; the seconds spent."""
        tile = self.tile
        n = x.size
        out = dict(row=np.zeros(0, np.int32), near=np.zeros(0, bool), recovered=np.zeros(0, bool),
                   dflux=np.zeros(0, np.float32), flux=np.zeros(0, np.float32), det_flux=np.zeros(0, np.float32),
                   det_dist=np.zeros(0, np.float32), n=0, seconds=dict(inject=0.0, aperture=0.0))
        if n == 0:
            return out
        beam = self.beam_px_at(x, y)
        r = 0.5 * beam if radius_px is None else np.broadcast_to(np.asarray(radius_px, float), x.shape)
        flux = np.broadcast_to(np.asarray(flux, float), x.shape)
        cid = tile.grid.cell_ids(x, y)
        offset = self.offset[cid]
        near = d < BLEND_RADIUS_BEAMS * beam
        # an existing PACS source within the radius: the injection counts only if brighter than it
        brighter = np.ones(n, bool)
        if self.det_tree is not None:
            lists = self.det_tree.query_ball_point(np.c_[x, y], r)
            for k, l in enumerate(lists):
                if l:
                    brighter[k] = flux[k] >= self.det_flux[l].max()
        # the batch is applied in bands of rows (pacs_fit.PREPARE_STRIP_ROWS): its positions are thinned so that no
        # injected light reaches another injection's aperture or annulus, so each band's sources measure exactly as
        # with the whole batch in. Each band works on a private copy of its rows widened by the reach of a patch,
        # an aperture and annulus (the outer-light box included) at a found peak up to the recovery radius away,
        # and the window of the peak search; the copy is dropped after the band, so a worker holds one band's rows
        # whatever the map's size and never writes the shared image
        flux_out, covered = np.full(n, np.nan), np.zeros(n, bool)
        found, det_flux, det_dist = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan)
        t_inject = t_ap = 0.0
        strip = pacs_fit.PREPARE_STRIP_ROWS
        ny = tile.shape[0]
        mg = (tile.injection_half_width() + int(np.ceil(APERTURES[tile.record.band][2] / tile.pixel_arcsec)) + 1
              + int(np.ceil(r.max())) + 2)
        band = np.floor(y / strip).astype(np.int64)
        for b in np.unique(band):
            k = np.flatnonzero(band == b)
            a0, a1 = max(int(b) * strip - mg, 0), min((int(b) + 1) * strip + mg, ny)
            t0 = time.time()
            buf = np.array(tile.image[a0:a1])
            for cx, cy, resp in tile._template_patches(x[k], y[k], flux[k]):
                for i in range(cx.size):
                    pacs_fit._paste(buf, int(cx[i]), int(cy[i]) - a0, resp[i])
            t_inject += time.time() - t0
            t0 = time.time()
            m = tile.measure(x[k], y[k], image=buf, row0=a0)
            flux_out[k], covered[k] = m.flux, m.covered
            f_, _xc, _yc, df_, dd_ = window_peaks(tile, x[k], y[k], r[k], self.m_offset, self.m_sigma, self.offset,
                                                  flux=flux[k], image=buf, row0=a0)
            found[k], det_flux[k], det_dist[k] = f_, df_, dd_
            t_ap += time.time() - t0
        recovered = covered & found & brighter
        keep = covered
        out.update(row=self.row_of_cell[cid[keep]], near=near[keep], recovered=recovered[keep],
                   dflux=(flux_out - offset - flux).astype(np.float32)[keep],
                   flux=flux.astype(np.float32)[keep], det_flux=det_flux.astype(np.float32)[keep],
                   det_dist=det_dist.astype(np.float32)[keep], n=int(keep.sum()),
                   seconds=dict(inject=t_inject, aperture=t_ap))
        return out


def _inject_batches(injector, plane, snr, half, seeds, rows, block):
    """Batches of one S/N plane and one half (0 build, 1 held out: the seed's parity) over the cells of `rows` (global
    row numbers, within one block of the map), one batch per entry of `seeds`. The seeds are (map, S/N plane, batch
    and half, block): the same for any cut of the work into pieces and any number of workers. Returns the injections
    (`row` the global row), the seconds spent and the injections far from / near (within BLEND_RADIUS_BEAMS of) a
    detection."""
    recs = []
    seconds = dict(inject=0.0, aperture=0.0)
    n_by_path = dict(far=0, near=0)
    for seed in seeds:
        r = injector.batch(np.random.default_rng([injector.seed_key, plane, 2 * seed + half, block]), rows, snr)
        recs.append(r)
        for k in seconds:
            seconds[k] += r["seconds"][k]
        n_by_path["near"] += int(r["near"].sum())
        n_by_path["far"] += int(r["near"].size - r["near"].sum())
    cat = lambda key, dt: np.concatenate([r[key] for r in recs]) if recs else np.zeros(0, dt)
    return (dict(row=cat("row", np.int32), near=cat("near", bool), recovered=cat("recovered", bool),
                 dflux=cat("dflux", np.float32), flux=cat("flux", np.float32), det_flux=cat("det_flux", np.float32),
                 det_dist=cat("det_dist", np.float32)), seconds, n_by_path)


def pool_product(Ws, rec, flux_grid, sigma_rows, n_row, sigma_out=None):
    """The curve from the build half's injections made in S/N (`rec`: row, plane (S/N), recovered): the recovered
    fraction (encoded), its 0.9 flux, and the injections behind the faintest flux plane (`curve_on_flux_grid`)."""
    fraction, count = curve_on_flux_grid(Ws, rec["row"], rec["plane"], rec["recovered"],
                                         sigma_rows if sigma_out is None else sigma_out, flux_grid, n_row)
    return dict(curve=np.rint(fraction * STORAGE_SCALE).astype(np.uint16),
                flux90=flux90(fraction, flux_grid).astype(np.float32), n_pos=count[:, -1].astype(np.int32),
                pooled_total=count)


def cell_errors(Ws, flux_grid, sigma_rows, rec, sigma_out=None):
    """The error of a reported flux from injections (`rec`: row, plane (S/N), near, dflux = measured - cell offset -
    injected): per S/N plane, the pooled median and 1.4826 x MAD of dflux / CELL_SIGMA over ALL covered injections
    (detected or not) farther than BLEND_RADIUS_BEAMS from every detection (`error_on_flux_grid`; a source within
    it is a blend, reported as an upper bound) of the cells within the pooling radius, read for each row at
    F / its own CELL_SIGMA and multiplied by it: SCATTER and BIAS (n_row, n_plane)."""
    n_row = len(sigma_rows)
    sel = ~rec["near"].astype(bool)
    bias, scatter = error_on_flux_grid(Ws, rec["row"][sel].astype(np.int64), rec["plane"][sel].astype(np.int64),
                                       rec["dflux"][sel], sigma_rows, flux_grid, n_row, sigma_out=sigma_out)
    return dict(scatter=scatter, bias=bias)


# ----------------------------------------------------------------------------
# self-checks

ZERO_CHECK_FLUX_MJY = 500.0
ZERO_CHECK_POSITIONS = 20
ZERO_CHECK_TOLERANCE = 0.005
REAL_CHECK_POSITIONS = 200


def self_check(config, record):
    """The template's normalisation, on a ZERO image: ZERO_CHECK_FLUX_MJY injected at ZERO_CHECK_POSITIONS empty-sky
    positions of `record`'s map, the aperture SUM with no sky subtraction divided by the injected flux times the
    map's encircled-energy fraction at the aperture radius (total-flux normalisation; for a mosaic the local mixture
    of its input maps' REAL unit sources, resampled, `pacs_checks.expected_enclosed`: absolute, independent of the
    PSF build's own fractions): every position must be
    within 0.5 %, else the template is not normalised to total flux. Prints the worst ratios and, for
    information, the share of the aperture flux the sky term of the catalogue's own measurement removes on a
    zero image (the clipped mode of the monotonic wing, not the mean the aperture fraction assumes). Raises
    ValueError on failure."""
    tile = pacs_fit.cached_tile(config, record)
    rng = np.random.default_rng(0)
    ny, nx = tile.shape
    margin = tile.injection_half_width() + 2
    r_ap, r_in, r_out = (r / tile.pixel_arcsec for r in APERTURES[record.band])
    reach = int(np.ceil(r_out)) + 2
    # no injected light in another position's aperture or annulus (the Injector's own separation)
    separation = tile.injection_half_width() + r_out
    xs, ys = [], []
    for _ in range(100000):
        x, y = rng.uniform(margin, nx - margin), rng.uniform(margin, ny - margin)
        iy, ix = int(round(y)), int(round(x))
        # empty sky: the whole aperture and annulus on valid data
        if np.isfinite(tile.error[iy - reach:iy + reach + 1, ix - reach:ix + reach + 1]).all() and all(np.hypot(x - a, y - b) > separation for a, b in zip(xs, ys)):
            xs.append(x)
            ys.append(y)
            if len(xs) == ZERO_CHECK_POSITIONS:
                break
    x, y = np.array(xs), np.array(ys)
    with tile.blank_image(), tile.injected(x, y, ZERO_CHECK_FLUX_MJY):
        f, sky, _sd, area, cov = pacs_fit._aperture_core(tile.image, tile.error, x, y, r_ap, r_in, r_out, 5)
    if isinstance(tile._psf, pacs_psf.MosaicPSF):
        # a mosaic has no single release table, and its own enclosed fractions are what the injection was normalised
        # by (they would cancel): the expected aperture sum is recomputed on a REAL unit source synthesised in each
        # contributing input map's pixels and resampled (pacs_checks.expected_enclosed), tied to the release EEF
        from sesnaimpute.sky.derived import pacs_checks
        eef, tie = pacs_checks.expected_enclosed(config, record, tile, x, y, tile._psf.weights(x, y))
        if tie > pacs_checks.RELEASE_TOLERANCE:
            raise ValueError("%s: the synthetic real source is %.2f %% off the release encircled energy (template, pixel "
                             "area or EEF file)" % (record.band, 100 * tie))
    else:
        eef = pacs_psf.physical_aperture_fraction(config, record, "total", False)
    expected = ZERO_CHECK_FLUX_MJY * eef
    expected = (expected if np.ndim(expected) == 0 else expected[cov])
    ratio = (f + sky * area)[cov] / expected
    sky_share = (sky * area)[cov] / expected
    print("  self-check (zero image) %s %s: %d positions, aperture sum / (injected x EEF_total(r_ap) = %.4f on average): min %.4f "
          "max %.4f; the catalogue's sky term on a zero image removes %.2f %% of it (information only)"
          % (record.band, record.map_id, ratio.size, float(np.mean(eef)), ratio.min() if ratio.size else np.nan,
             ratio.max() if ratio.size else np.nan, 100 * sky_share.mean() if ratio.size else np.nan), flush=True)
    if ratio.size == 0 or np.abs(ratio - 1).max() > ZERO_CHECK_TOLERANCE:
        raise ValueError("%s: the injected template is not normalised to total flux within 0.5 %%" % record.band)


def _real_check(injector, flux_mjy, half, rows=None):
    """The catalogue's own measurement in its own regime: REAL_CHECK_POSITIONS empty-sky positions of the real map
    (farther than one beam from every detected source), the aperture flux after injecting `flux_mjy` minus before,
    over `flux_mjy`. Returns the ratios. `rows`: the rows the positions are drawn from (default all)."""
    tile = injector.tile
    rng = np.random.default_rng([injector.seed_key, 99999, half])
    x, y, d = injector.positions(rng, np.arange(injector.ids.size) if rows is None else rows)
    far = d >= BLEND_RADIUS_BEAMS * injector.beam_px_at(x, y)
    x, y = x[far][:REAL_CHECK_POSITIONS], y[far][:REAL_CHECK_POSITIONS]
    before = tile.measure(x, y)
    with tile.writable(), tile.injected(x, y, flux_mjy):
        after = tile.measure(x, y)
    ok = before.covered & after.covered
    return (after.flux[ok] - before.flux[ok]) / flux_mjy


#: The injection work of a map is cut by the map's own cell grid into BLOCKS of INJECTION_BLOCK_LINES lines of cells
#: (block = cell line // INJECTION_BLOCK_LINES: fixed by the grid, never by the number of workers), and each block's
#: work into its (S/N plane, half) pairs: a piece is one block and a few pairs, about INJECTION_PIECE_ROWS of the
#: block's cells per pair of a piece (the largest blocks one pair per piece). Every (block, plane, half, batch) has
#: its own seeds, so the injections do not depend on the cut into pieces or on n_jobs. The batches go in rounds over
#: all the blocks at once: the first round (sized for the interior of the map) and then one round per extra batch,
#: the cells it covers chosen from the whole map's counts, so a block's edge is pooled with its neighbours' injections.
INJECTION_BLOCK_LINES = 16
INJECTION_PIECE_ROWS = 64
#: the pooling is cut into square blocks of POOL_BLOCK_CELLS cells, each pooled from the injections of the cells
#: within the widest pooling radius of it (a shared memory-mapped table), identically to the whole map
POOL_BLOCK_CELLS = 8

_INJECTION_STATE = {}


def _injection_state(config, record, inp, t):
    """The process-local Injector of map `t`, built on the first piece of the map a worker takes, kept (one map at
    a time) for the map's further pieces."""
    key = record.path
    if key not in _INJECTION_STATE:
        _INJECTION_STATE.clear()
        _INJECTION_STATE[key] = Injector(pacs_fit.cached_tile(config, record), inp, t + 1)
    return _INJECTION_STATE[key]


def _map_first(task, config, records):
    """The number of batches of the first round of a map: sized so that an interior cell's pool holds
    INTERIOR_SAFETY x MIN_INJECTIONS. Returns `(tile index, batches)`."""
    t, inp = task
    inj = _injection_state(config, records[t], inp, t)
    W = pool_matrices(cell_centres_arcsec(inj.tile.grid, inj.ids), widenings=POOL_WIDENINGS[:1])[0]
    # injections one batch puts into the pool of an interior cell: density x the pool's cell count
    interior = (inj.cell_px * inj.valid_fraction / (SEPARATION_AREAS * inj.separation_px ** 2)) * float(np.median(np.diff(W.indptr)))
    return t, int(np.ceil(INTERIOR_SAFETY * MIN_INJECTIONS / max(interior, 1e-9)))


def injection_blocks(ids, nx):
    """The injection blocks of a map whose rows hold the cells `ids` (ascending): `[(block, first row, last row + 1)]`."""
    blk = np.asarray(ids, np.int64) // nx // INJECTION_BLOCK_LINES
    b, first = np.unique(blk, return_index=True)
    return [(int(b[i]), int(first[i]), int(first[i + 1]) if i + 1 < b.size else int(blk.size)) for i in range(b.size)]


def _inject_pairs(piece, config, records):
    """The injections of one piece: `piece = (tile index, inputs, (block, first row, last row + 1), jobs, real
    check?)`, `jobs` the `(S/N plane, half, batch seeds, rows or None: the block's)` of the piece (see `Injector`,
    `round_task_list`): the batches of each over the block's cells (`_inject_batches`); on the piece flagged for it, the
    real-map self-check. Returns `(tile index, dict)`."""
    t, inp, (b, r0, r1), jobs, do_check = piece
    record = records[t]
    inj = _injection_state(config, record, inp, t)
    snr = snr_grid()
    check = None
    if do_check:
        # the real-map report (information, never a stop: a crowded map has few empty positions and one in bright
        # structured emission moves the mean; the normalisation is checked on a zero image by `self_check` and the
        # flux scale by the bias of every injection, `acceptance` C2): F = 10 x the map's median 5-sigma flux
        f_check = 10.0 * pacs_fit.DETECTION_SNR * float(np.median(inj.sigma[inj.ids]))
        check = []
        for half in (0, 1):
            r = _real_check(inj, f_check, half, np.arange(r0, r1))
            check.append(float(r.mean()) if r.size else np.nan)
            q = np.percentile(r, [16, 50, 84]) if r.size else np.full(3, np.nan)
            print("  real-map report %s %s half %d: %d empty positions, F = %.0f mJy, injected-minus-before / F: mean %.4f, "
                  "16/50/84 %%: %.4f %.4f %.4f" % (record.band, record.map_id, half, r.size, f_check,
                                                    r.mean() if r.size else np.nan, *q), flush=True)
    done = []
    for p, half, seeds, rows in jobs:
        r, sec, cnt = _inject_batches(inj, p, snr[p], half, seeds, np.arange(r0, r1) if rows is None else rows, b)
        done.append((p, half, b, seeds[0], r, sec, cnt))
    return t, dict(pairs=done, check=check, n_detected=inj.n_detected)


def _chunks(jobs, n_rows):
    """A block's jobs cut into pieces of about INJECTION_PIECE_ROWS of the block's cells per job."""
    k = int(min(len(jobs), max(1, np.ceil(n_rows / INJECTION_PIECE_ROWS))))
    return [[jobs[i] for i in c] for c in np.array_split(np.arange(len(jobs)), k) if c.size]


def injection_pairs():
    """The (S/N plane, half) pairs of the injections, in the order of the count columns."""
    return [(p, half) for p in range(snr_grid().size) for half in (0, 1)]


def tile_order(inputs):
    """The maps of a band as the injections and the pooling take them: from the largest."""
    return [t for t, _ in sorted(inputs.items(), key=lambda kv: -kv[1]["ids"].size)]


def map_sizing(config, records, inputs, n_jobs, stage):
    """`{tile: batches of the first round}` (`_map_first`) of every map of a band."""
    return dict(pieces_module.run(functools.partial(_map_first, config=config, records=records),
                                  [(t, inputs[t]) for t in tile_order(inputs)], n_jobs, stage=stage,
                                  unit="maps (artificial sources, sizing)"))


def part_counts(inputs, done):
    """What a set of pieces' outputs (`done`, `[(tile, part)]`) add to the maps' injection counts, `{tile: sparse matrix
    (cells x `injection_pairs`)}`: the counts `round_task_list` reads for the rounds before the one it plans."""
    pairs = injection_pairs()
    acc = {}
    for t, part in done:
        rows, cols = acc.setdefault(t, ([], []))
        for p, half, _b, _s, res, _sec, _cnt in part["pairs"]:
            rows.append(res["row"])
            cols.append(np.full(res["row"].size, pairs.index((p, half))))
    out = {}
    for t, (rows, cols) in acc.items():
        row, col = np.concatenate(rows or [np.zeros(0, np.int64)]), np.concatenate(cols or [np.zeros(0, np.int64)])
        out[t] = csr_matrix((np.ones(row.size, np.int64), (row, col)), shape=(len(inputs[t]["ids"]), len(pairs)))
    return out


def round_task_list(records, inputs, nx, first, r, load):
    """The injections of one map of a band in rounds of pieces over all maps and blocks (`injection_blocks`): round 0
    makes each map's first batches in every block for every (S/N plane, half) pair; each further round one more batch
    per pair, over the cells within the pooling radius of a cell whose pool (the WHOLE map's, from the injections of
    all earlier rounds) is still short of MIN_INJECTIONS, until none is or MAX_EXTRA_BATCHES have been made. Returns the
    ordered pieces of round `r` (`_inject_pairs`; [] when the band's injections are complete); `load(q)` gives
    `{tile: counts}` (`part_counts`) of the pieces' outputs of round `q < r` (called only for a round that had pieces).
    `nx`: each map's cell grid width, `first`: `map_sizing`."""
    pairs = injection_pairs()
    tiles = tile_order(inputs)
    st, tasks = {}, []
    for t in tiles:
        ids = inputs[t]["ids"]
        blocks = injection_blocks(ids, nx[t])
        W = pool_matrices(cell_centres_arcsec(_tile_geometry(records[t])[0], ids), widenings=POOL_WIDENINGS[:1])[0]
        st[t] = dict(blocks=blocks, W=W, count=np.zeros((len(ids), len(pairs))), first=first[t])
        for i, blk in enumerate(blocks):
            jobs = [(p, half, tuple(range(first[t])), None) for p, half in pairs]
            for j, c in enumerate(_chunks(jobs, blk[2] - blk[1])):
                tasks.append((t, inputs[t], blk, c, i == 0 and j == 0))
    for extra in range(min(r, MAX_EXTRA_BATCHES + 1)):
        if not tasks:
            return []
        for t, m in load(extra).items():
            st[t]["count"] += m.toarray()
        tasks = []
        if extra == MAX_EXTRA_BATCHES:
            break
        for t in tiles:
            s_ = st[t]
            short = (s_["W"] @ s_["count"]) < MIN_INJECTIONS
            act = short.any(axis=0)
            if not act.any():
                continue
            reach = (s_["W"] @ short.astype(float)) > 0
            for blk in s_["blocks"]:
                b, r0, r1 = blk
                jobs = [(p, half, (s_["first"] + extra,), r0 + np.flatnonzero(reach[r0:r1, i]))
                        for i, (p, half) in enumerate(pairs) if act[i] and reach[r0:r1, i].any()]
                for c in (_chunks(jobs, r1 - r0) if jobs else []):
                    tasks.append((t, inputs[t], blk, c, False))
    return tasks if r <= MAX_EXTRA_BATCHES else []


_POOL_KEYS = ("row", "plane", "half", "near", "recovered", "dflux")


def pool_dir(config, record):
    """Where a map's injection table is written for the pooling (beside its prepared layers), removed after."""
    return os.path.join(pacs_fit.prepared_dir(config, record), "pool")


def _write_pool_table(t, inp, parts, records, config, d=None):
    """A map's injections, from its pieces' outputs (`parts`), written as the memory-mapped table the pooling tasks
    read: the records of both halves sorted by row (a stable sort of the records taken in (half, S/N plane, block, batch)
    order, so the table does not depend on the order the pieces finished), with `offsets` (the first record of
    every row). Written to `d` (default `pool_dir`). Returns `(table directory, the held-out half's injections, the
    statistics the parent prints)`."""
    ids = np.asarray(inp["ids"], np.int64)
    n_row = ids.size
    snr = snr_grid()
    keys = ("row", "plane", "near", "recovered", "dflux", "flux", "det_flux", "det_dist")
    stores = ({k: [] for k in keys}, {k: [] for k in keys})
    seconds = dict(inject=0.0, aperture=0.0)
    n_by_path = dict(far=0, near=0)
    near_plane = np.zeros((2, snr.size, 2))
    entries = sorted((e for part in parts for e in part["pairs"]), key=lambda e: (e[1], e[0], e[2], e[3]))
    for p, half, b, _seed, r, sec, cnt in entries:
        for k in ("row", "near", "recovered", "dflux", "flux", "det_flux", "det_dist"):
            stores[half][k].append(r[k])
        stores[half]["plane"].append(np.full(r["row"].size, p, np.int8))
        for k in seconds:
            seconds[k] += sec[k]
        for k in n_by_path:
            n_by_path[k] += cnt[k]
        nr = r["near"]
        near_plane[half, p] += (nr.sum(), (r["recovered"] & nr).sum())
    build, held = ({k: np.concatenate(v) for k, v in s.items()} for s in stores)
    cols = {k: np.concatenate([build[k], held[k]]) for k in ("row", "plane", "near", "recovered", "dflux")}
    cols["half"] = np.concatenate([np.zeros(build["row"].size, np.int8), np.ones(held["row"].size, np.int8)])
    order = np.argsort(cols["row"], kind="stable")
    d = d or pool_dir(config, records[t])
    os.makedirs(d, exist_ok=True)
    for k in _POOL_KEYS:
        np.save(os.path.join(d, k + ".npy"), cols[k][order])
    count = np.bincount(cols["row"], minlength=n_row)
    np.save(os.path.join(d, "offsets.npy"), np.concatenate([[0], np.cumsum(count)]).astype(np.int64))
    np.save(os.path.join(d, "ids.npy"), ids)
    np.save(os.path.join(d, "sigma.npy"), np.asarray(inp["sigma"], np.float64))
    check = next(part["check"] for part in parts if part["check"] is not None)
    stats = dict(map_id=records[t].map_id, n_row=n_row, n_detected=parts[0]["n_detected"],
                 n_build=int(build["row"].size), n_held=int(held["row"].size), seconds=seconds, n_by_path=n_by_path,
                 near_plane=near_plane, check=check)
    return d, held, stats


def pool_tasks(t, d, ids, nx):
    """The pooling tasks of a map: square blocks of POOL_BLOCK_CELLS cells holding rows,
    `(tile index, table directory, nx, cell line range, cell column range)`."""
    ids = np.asarray(ids, np.int64)
    cy, cx = ids // nx, ids % nx
    b = np.unique(np.c_[cy // POOL_BLOCK_CELLS, cx // POOL_BLOCK_CELLS], axis=0)
    n = POOL_BLOCK_CELLS
    return [(t, d, nx, int(i * n), int((i + 1) * n), int(j * n), int((j + 1) * n)) for i, j in b]


_POOL_TABLE = {}


def _pool_block(task, records, flux_grid):
    """The pooled curve, flux 0.9, injections behind it and errors (`pool_product`, `cell_errors`) of the cells of one
    pooling block, from the injections of the cells within the widest pooling radius of it, read from the map's
    memory-mapped table (`_write_pool_table`) and pooled with the matrices of exactly those cells: each cell's value
    is the whole map's pooling, whatever the cut. Returns `(tile index, dict of the block's global rows and
    arrays)`."""
    t, d, nx, cy0, cy1, cx0, cx1 = task
    if _POOL_TABLE.get("dir") != d:
        _POOL_TABLE.clear()
        _POOL_TABLE.update(dir=d, **{k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r")
                                     for k in _POOL_KEYS + ("offsets", "ids", "sigma")})
        _POOL_TABLE["grid"] = _tile_geometry(records[t])[0]
    tab = _POOL_TABLE
    ids = tab["ids"]
    halo = int(np.ceil(POOL_WIDENINGS[-1] * POOL_ARCSEC / pacs_fit.CELL_ARCSEC)) + 1
    ey0, ey1 = max(cy0 - halo, 0), cy1 + halo
    ex0, ex1 = max(cx0 - halo, 0), min(cx1 + halo, nx)
    lines = np.arange(ey0, ey1, dtype=np.int64)
    a = np.searchsorted(ids, lines * nx + ex0)
    b = np.searchsorted(ids, lines * nx + ex1)
    n_line = b - a
    total = int(n_line.sum())
    ext = np.repeat(a - (np.cumsum(n_line) - n_line), n_line) + np.arange(total)
    off = tab["offsets"]
    per_row = np.asarray(off[ext + 1]) - np.asarray(off[ext])
    rec = {k: np.concatenate([np.asarray(tab[k][off[x]:off[y]]) for x, y in zip(a, b) if y > x] or [np.zeros(0, tab[k].dtype)])
           for k in _POOL_KEYS if k != "row"}
    rec["row"] = np.repeat(np.arange(total), per_row)
    ext_ids = np.asarray(ids[ext])
    inside = ((ext_ids // nx >= cy0) & (ext_ids // nx < cy1) & (ext_ids % nx >= cx0) & (ext_ids % nx < cx1))
    out_local = np.flatnonzero(inside)
    Ws_full = pool_matrices(cell_centres_arcsec(tab["grid"], ext_ids))
    Ws = [W[out_local] for W in Ws_full]
    sig = np.asarray(tab["sigma"])[ext]
    sig_out = sig[out_local]
    bm = rec["half"] == 0
    build = {k: rec[k][bm] for k in ("row", "plane", "recovered", "near", "dflux")}
    out = pool_product(Ws, build, flux_grid, sig, total, sigma_out=sig_out)
    out["errors"] = cell_errors(Ws, flux_grid, sig, rec, sigma_out=sig_out)
    out["build_errors"] = cell_errors(Ws, flux_grid, sig, build, sigma_out=sig_out)
    out["rows"] = ext[out_local]
    return t, out


def _assemble_tile(inp, held, stats, blocks, flux_grid):
    """A map's product from its pooling blocks' outputs (`blocks`) and the held-out half: the arrays of the map's rows
    for the product (the build half) with the held-out injections and the statistics the parent prints."""
    n_row = np.asarray(inp["ids"]).size
    n_plane = flux_grid.size
    out = dict(curve=np.zeros((n_row, n_plane), np.uint16), flux90=np.full(n_row, np.nan, np.float32),
               n_pos=np.zeros(n_row, np.int32),
               errors={k: np.full((n_row, n_plane), np.nan, np.float32) for k in ("scatter", "bias")})
    build_errors = {k: np.full((n_row, n_plane), np.nan, np.float32) for k in ("scatter", "bias")}
    tot = np.zeros((n_row, n_plane))
    filled = np.zeros(n_row, bool)
    for o in blocks:
        r = o["rows"]
        assert not filled[r].any()
        filled[r] = True
        out["curve"][r], out["flux90"][r], out["n_pos"][r], tot[r] = o["curve"], o["flux90"], o["n_pos"], o["pooled_total"]
        for k in out["errors"]:
            out["errors"][k][r] = o["errors"][k]
            build_errors[k][r] = o["build_errors"][k]
    assert filled.all()
    held = dict(held)
    held["build_errors"] = build_errors
    out["held"] = held
    out["stats"] = dict(stats, min_pool=float(tot.min()), median_pool=float(np.median(tot)),
                        short=float((tot < MIN_INJECTIONS).mean()))
    return out


# ----------------------------------------------------------------------------
# one band of one region

def _tile_geometry(record):
    wcs = WCS(fits.Header.fromstring(record.wcs_header)).celestial
    return pacs_fit.MapGrid.from_wcs(wcs, record.shape), wcs


def _assemble_peaks(found, records, geometry, shifts, keys, cell_sigma, cell_bin, ra, dec):
    """The band's PACS sources from the blocks' detections: map index, pixel position, flux, the row of their cell
    (`keys` the rows' map x cell keys), S/N, sky bin, ICRS position; and the inverted-map peak count per sky bin."""
    t_, x_, y_, f_, c_ = [], [], [], [], []
    n_row = keys.size
    for t, d in found:
        t_.append(np.full(d["x"].size, t, np.int32))
        x_.append(d["x"])
        y_.append(d["y"])
        f_.append(d["fnu"])
        c_.append(d["cell"])
    cat = lambda v, dt: np.concatenate(v).astype(dt) if v else np.zeros(0, dt)
    m, x, y, f, c = cat(t_, np.int32), cat(x_, float), cat(y_, float), cat(f_, float), cat(c_, np.int64)
    q = m.astype(np.int64) * (1 << 40) + c
    row = np.clip(np.searchsorted(keys, q), 0, max(n_row - 1, 0))
    ok = (keys[row] == q) if n_row else np.zeros(m.size, bool)
    m, x, y, f, row = m[ok], x[ok], y[ok], f[ok], row[ok]
    snr = f / cell_sigma[row].astype(float)
    pra, pde = np.full(m.size, np.nan), np.full(m.size, np.nan)
    for t in np.unique(m):
        sel = np.flatnonzero(m == t)
        _src_cell, _cells, grid, wcs = geometry[t]
        wr, wd = wcs.all_pix2world(x[sel], y[sel], 0)
        pra[sel], pde[sel] = pacs_fit.from_map_frame(np.asarray(wr), np.asarray(wd), shifts[t][:2])
    return dict(map=m, x=x, y=y, fnu=f, row=row, snr=snr, bin=cell_bin[row], ra=pra, dec=pde)


def _calibrate(piece, config, records):
    """The injected-vs-detected position scatter on one map (`piece = (tile index, inputs)`): one batch at each S/N
    of CALIBRATION_SNR (own seeds, positions farther than a beam from every PACS source of the map), the blind
    detection within one beam of each; returns (tile index, the found sources' distances in arcsec times their S/N, the
    position-scatter constant)."""
    t, inp = piece
    tile = pacs_fit.cached_tile(config, records[t])
    inj = Injector(tile, inp, t + 1)
    out = []
    rows = np.arange(inj.ids.size)
    for k, snr in enumerate(CALIBRATION_SNR):
        x, y, d = inj.positions(np.random.default_rng([t + 1, 7777, k]), rows)
        far = d >= inj.beam_px_at(x, y)
        x, y, d = x[far], y[far], d[far]
        r = inj.measure_positions(x, y, d, inj.flux_at(x, y, snr), radius_px=inj.beam_px_at(x, y))
        good = np.isfinite(r["det_dist"])
        out.append(r["det_dist"][good] * tile.pixel_arcsec * snr)
    return t, (np.concatenate(out) if out else np.zeros(0))


def tile_inputs(cell_map, cell_id, sigma, offset):
    """What `Injector` needs per map tile, from the band's rows: `{tile: dict}` for the tiles that hold rows.
    `cell_map`, `cell_id`, `sigma`, `offset` are the rows' map index, cell id, noise and offset (float32 as
    stored). The blind detection's sources and the matching radius are added by the caller."""
    out = {}
    for t in np.unique(cell_map):
        rows = np.flatnonzero(cell_map == t)
        out[int(t)] = dict(ids=cell_id[rows], sigma=sigma[rows].astype(np.float64), offset=offset[rows].astype(np.float64),
                           row0=int(rows[0]))
    return out


def _print_tile_stats(s, flux_grid):
    sec, n = s["seconds"], s["n_by_path"]
    per = lambda k, c: 1e4 * sec[k] / max(c, 1)
    total = n["far"] + n["near"]
    print("  %s: %d cells, %d detected sources; injections build %d held-out %d; pooled per (cell, plane): "
          "min %d median %d (%.1f %% of (cell, plane) below %d after widening); seconds per 10000 injections: aperture "
          "%.1f, adding and restoring %.1f; %d within a beam of a detection (of %d); real-map report mean ratio %.4f / %.4f"
          % (s["map_id"], s["n_row"], s["n_detected"], s["n_build"], s["n_held"], s["min_pool"],
             s["median_pool"], 100 * s["short"], MIN_INJECTIONS, per("aperture", total), per("inject", total),
             n["near"], total, s["check"][0], s["check"][1]), flush=True)
    bp = s["near_plane"].sum(axis=0)
    if bp[:, 0].sum():
        snr = snr_grid()
        every = range(0, snr.size, 4)
        print("      within a beam of a detection: recovered fraction (the brighter member only) by S/N: %s"
              % " ".join("%.3g:%.2f" % (snr[p], bp[p, 1] / max(bp[p, 0], 1)) for p in every), flush=True)


def _band_detect(config, records, ra, dec, n_jobs, stage, flux_grid, detect_only=False):
    """Measure every covered cell of every map of one band inside the region's padded source bounding box;
    choose each source's deepest map; detect the PACS sources and fix the matching radius: everything of the band
    before its artificial sources (stage DETECT of `pacs_stages`). Returns the band's state (what `_band_dict` and the
    injection stages read), or None when no map reaches a source; with `detect_only` the band's datasets from the
    detection alone. `records`: the band's one mosaic (`pacs_stages.run_region`)."""
    n_src = ra.size
    n_plane = flux_grid.size

    def tile_cells(record, shift):
        # every cell of the map (the PACS product's footprint is the maps' own coverage, no SESNA; the cell stage
        # keeps the covered ones), and each SESNA source's cell (SOURCE_ROW only, the catalogue's lookup)
        grid, wcs = _tile_geometry(record)
        sx, sy = (np.asarray(v) for v in wcs.all_world2pix(*pacs_fit.to_map_frame(ra, dec, shift), 0))
        src_cell = grid.cell_ids(sx, sy)
        return grid, wcs, src_cell, np.arange(grid.n_cells, dtype=np.int64)

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
        print("  %s: registration east %+.2f north %+.2f arcsec from %d pairs%s%s"
              % (records[t].map_id, e, n, k, "  ABOVE 1.5 ARCSEC" if np.hypot(e, n) > 1.5 else "", ""), flush=True)
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
    names = ("covered", "n_sigma", "sigma", "offset", "sky", "ra", "dec", "m_sigma", "m_offset")
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

    inputs = tile_inputs(cell_map, cell_id, cell_sigma, cell_offset)
    m_sigma_rows = cols["m_sigma"].astype(np.float32)
    m_offset_rows = cols["m_offset"].astype(np.float32)
    for t, inp in inputs.items():
        rows = inp["row0"] + np.arange(inp["ids"].size)
        inp["m_sigma"], inp["m_offset"] = m_sigma_rows[rows].astype(float), m_offset_rows[rows].astype(float)
    cell_bin = sky_bins(cell_map, cols["sky"])

    # the blind detection of the PACS sources on every map, on the cell stage's blocks, and on the negated maps
    cell_tables = {}
    for t, inp in inputs.items():
        grid = geometry[t][2]
        tabs = [np.full(grid.n_cells, np.nan) for _ in range(4)]
        for arr, key in zip(tabs, ("offset", "sigma", "m_offset", "m_sigma")):
            arr[inp["ids"]] = inp[key]
        cell_tables[t] = tuple(tabs)
    blocks = [(t, c) for t, c in pieces if t in inputs]
    found = pieces_module.run(functools.partial(_block_peaks, config=config, records=records, cell_tables=cell_tables),
                              blocks, n_jobs, stage=stage, unit="cell blocks (PACS sources)")
    peaks = _assemble_peaks(found, records, geometry, shifts, keys, cell_sigma, cell_bin, ra, dec)
    for t, inp in inputs.items():
        mine = peaks["map"] == t
        inp["peak_x"], inp["peak_y"], inp["peak_fnu"] = peaks["x"][mine], peaks["y"][mine], peaks["fnu"][mine]
    if detect_only:
        # the PACS source detection alone (no artificial sources): what the merge needs
        print("  PACS detections %d; by sky bin (faint to bright) %s"
              % (peaks["fnu"].size, [int((peaks["bin"] == j).sum()) for j in range(SKY_BINS)]), flush=True)
        return dict(peaks=peaks, records=records, cell_map_index=cell_map, cell_id=cell_id, cell_offset=cell_offset,
                    map_shift=shifts[:, :2].astype(np.float32), scatter=np.full((n_row, n_plane), np.nan, np.float32),
                    held=None,
                    curve=np.zeros((n_row, 0), np.uint16))

    # the matching radius: the injected-vs-detected position scatter (bright injections) and the real sources'
    # alignment scatter against SESNA, in quadrature
    calib = pieces_module.run(functools.partial(_calibrate, config=config, records=records),
                              sorted(inputs.items()), n_jobs, stage=stage, unit="maps (matching radius)")
    d_inj = np.concatenate([c for _, c in calib]) if calib else np.zeros(0)
    # the recovery radius: 3 x the detection's own position scatter at the injection's S/N, measured on the bright
    # injections as a scatter x S/N constant (the scatter goes as 1/S/N), between one pixel and half a beam
    pos_k = np.sqrt(np.median(d_inj ** 2) / np.log(2.0)) if d_inj.size else np.nan
    r_match = np.full(len(records), np.nan)
    for t, inp in inputs.items():
        inp["pos_k_arcsec"] = pos_k
        r_match[t] = MATCH_SIGMAS * pos_k / 5.0
    rms_inj = pos_k
    rms_align = np.nan
    print("  recovery radius: %.1f x position scatter, the scatter x S/N = %.2f arcsec from %d bright injections (radius "
          "%.2f arcsec at S/N 5, %.2f at S/N 30; floor one pixel, cap half a beam)"
          % (MATCH_SIGMAS, pos_k, d_inj.size, MATCH_SIGMAS * pos_k / 5.0, MATCH_SIGMAS * pos_k / 30.0), flush=True)
    print("  PACS detections %d; by sky bin (faint to bright) %s"
          % (peaks["fnu"].size, [int((peaks["bin"] == j).sum()) for j in range(SKY_BINS)]), flush=True)

    return dict(records=records, n_src=n_src, geometry=geometry, cols=cols, cell_map=cell_map, cell_id=cell_id, n_row=n_row,
                src_row=src_row, shifts=shifts, cell_sigma=cell_sigma, cell_offset=cell_offset, m_sigma_rows=m_sigma_rows,
                m_offset_rows=m_offset_rows, cell_bin=cell_bin, inputs=inputs, peaks=peaks, pos_k=pos_k, r_match=r_match,
                rms_inj=rms_inj, rms_align=rms_align)


def _band_dict(s, injected, flux_grid):
    """The band's datasets from its state (`_band_detect`) and `injected`, `[(tile index, _assemble_tile's output)]` of
    its maps (`tile_order`): the curves, errors and the held-out half, the PACS sources' uncertainty."""
    records, inputs, peaks, geometry, cols = s["records"], s["inputs"], s["peaks"], s["geometry"], s["cols"]
    n_row, n_plane = s["n_row"], flux_grid.size
    curve = np.zeros((n_row, n_plane), np.uint16)
    n_pos = np.zeros(n_row, np.int32)
    flux90_ = np.full(n_row, np.nan, np.float32)
    cell_keys = ("scatter", "bias")
    errors = {k: np.full((n_row, n_plane), np.nan, np.float32) for k in cell_keys}
    build_errors = {k: np.full((n_row, n_plane), np.nan, np.float32) for k in cell_keys}
    held = {k: [] for k in ("row", "plane", "near", "recovered", "dflux", "flux", "det_flux", "det_dist")}
    stats = []
    for t, out in injected:
        r0 = inputs[t]["row0"]
        sl = slice(r0, r0 + inputs[t]["ids"].size)
        curve[sl], n_pos[sl], flux90_[sl] = out["curve"], out["n_pos"], out["flux90"]
        for k in cell_keys:
            errors[k][sl] = out["errors"][k]
            build_errors[k][sl] = out["held"]["build_errors"][k]
        for k in held:
            held[k].append(out["held"][k] + r0 if k == "row" else out["held"][k])
        stats.append(out["stats"])
    held = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in held.items()}
    held["stats"] = stats
    held["build_errors"] = build_errors
    # the PACS sources' uncertainty: the measured SCATTER of their cell at their flux
    peaks["sigma"] = calibrated_sigma(errors["scatter"][peaks["row"]], flux_grid, peaks["fnu"]) if peaks["fnu"].size else np.zeros(0)
    for s_ in stats:
        _print_tile_stats(s_, flux_grid)
    envelope = curve.max(axis=0).astype(np.uint16) if n_row else np.zeros(n_plane, np.uint16)
    return dict(curve=curve, source_row=s["src_row"].astype(np.int32), envelope=envelope, n_pos=n_pos,
                peaks=peaks, r_match=s["r_match"], rms_inj=s["rms_inj"], rms_align=s["rms_align"],
                cell_m_sigma=s["m_sigma_rows"], cell_m_offset=s["m_offset_rows"],
                sky_bin=s["cell_bin"].astype(np.int8), n_sigma=cols["n_sigma"].astype(np.int16), flux90=flux90_, bias=errors["bias"], scatter=errors["scatter"],
                held=held, stats=stats,
                cell_ra=cols["ra"], cell_dec=cols["dec"], cell_sigma=s["cell_sigma"], cell_offset=s["cell_offset"],
                cell_id=s["cell_id"], cell_map_index=s["cell_map"],
                cell_map=np.array([records[t].map_id for t in s["cell_map"]] or [""], dtype="S")[:n_row],
                n_candidate=sum(g[1].size for g in geometry),
                map_id=np.array([r.map_id for r in records], dtype="S"),
                map_obsid=np.array([",".join(r.obsids) for r in records], dtype="S"),
                map_shift=s["shifts"][:, :2].astype(np.float32), map_pairs=s["shifts"][:, 2].astype(np.int32),
                records=records,
                map_wcs=np.array([g[3].to_header_string() for g in geometry], dtype="S"),
                map_shape=np.array([g[2].shape for g in geometry], dtype=np.int32).reshape(len(records), 2),
                map_cell_step=np.array([(g[2].step_x, g[2].step_y) for g in geometry], dtype=np.float64).reshape(len(records), 2))


def sky_bins(cell_map, cell_sky):
    """Per row, its within-map sky bin: the cells of each map cut into SKY_BINS equal bins by rank of their annulus
    sky (PACS maps have no absolute zero level), 0 faintest."""
    bins = np.zeros(cell_map.size, np.int64)
    for m in np.unique(cell_map):
        rows = np.flatnonzero(cell_map == m)
        rank = np.argsort(np.argsort(np.nan_to_num(cell_sky[rows], nan=np.nanmedian(cell_sky[rows])), kind="stable"))
        bins[rows] = np.minimum(rank * SKY_BINS // max(rows.size, 1), SKY_BINS - 1)
    return bins


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
                peaks=dict(map=z(np.int32), x=z(float), y=z(float), fnu=z(float), row=z(np.int64), snr=z(float),
                           bin=z(np.int8), ra=z(float), dec=z(float), sigma=z(float), chance=z(float), r_match=z(float)),
                r_match=z(np.float32), rms_inj=np.nan, rms_align=np.nan, sky_bin=z(np.int8),
                cell_m_sigma=z(np.float32), cell_m_offset=z(np.float32), records=[],
                envelope=np.zeros(n_plane, np.uint16), n_pos=z(np.int32), n_sigma=z(np.int16), flux90=z(np.float32),
                bias=nan(), scatter=nan(),
                held=None, stats=[], cell_ra=z(float),
                cell_dec=z(float), cell_sigma=z(np.float32), cell_offset=z(np.float32), cell_id=z(np.int64),
                cell_map_index=z(np.int32), cell_map=z("S1"), n_candidate=0, map_id=z("S1"),
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
            ds.attrs["POOL_ARCSEC"] = POOL_ARCSEC
            ds.attrs["POOL_WIDENINGS"] = np.asarray(POOL_WIDENINGS)
            ds.attrs["SNR_GRID"] = snr_grid()
            ds.attrs["R_MATCH_INJECTED_RMS_ARCSEC"] = float(d["rms_inj"])
            ds.attrs["R_MATCH_ALIGNMENT_POOLED_RMS_ARCSEC"] = float(d["rms_align"])
            w("MAG_GRID_" + b, mag)
            w("ENVELOPE_" + b, d["envelope"])
            w("COMPLETED_FROM_" + b, np.full(d["curve"].shape[0], np.nan, np.float32))
            w("N_POS_" + b, d["n_pos"])
            w("N_SIGMA_POS_" + b, d["n_sigma"])
            w("FLUX90_%s_MJY" % b, d["flux90"])
            w("BIAS_%s_MJY" % b, d["bias"])
            w("SCATTER_%s_MJY" % b, d["scatter"])
            w("SKY_BIN_" + b, d["sky_bin"])
            w("CELL_SIGMA_MATCHED_" + b, d["cell_m_sigma"])
            w("CELL_OFFSET_MATCHED_" + b, d["cell_m_offset"])
            w("MAP_R_MATCH_%s_ARCSEC" % b, np.asarray(d["r_match"], np.float32))
            w("CELL_RA_" + b, d["cell_ra"])
            w("CELL_DEC_" + b, d["cell_dec"])
            w("CELL_ID_" + b, d["cell_id"])
            w("CELL_MAP_INDEX_" + b, d["cell_map_index"])
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


#: the cross-band merge (planner ruling, owner-approved 2026-10-28): detections of two bands closer than MERGE_SIGMAS x
#: the measured per-source position scatter between those bands (bright common sources, S/N >= MERGE_MIN_SNR, paired
#: within half the redder beam) are one PACS source; fewer than MERGE_MIN_PAIRS pairs: half the redder beam
MERGE_SIGMAS = 3.0
MERGE_MIN_SNR = 20.0
MERGE_MIN_PAIRS = 5


def _unit(ra_deg, dec_deg):
    ra, dec = np.radians(np.asarray(ra_deg, float)), np.radians(np.asarray(dec_deg, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _chord(arcsec):
    return 2.0 * np.sin(np.radians(np.asarray(arcsec, float) / 3600.0) / 2.0)


def _sep_arcsec(u, v):
    return np.degrees(2.0 * np.arcsin(np.clip(np.linalg.norm(u - v, axis=-1) / 2.0, 0.0, 1.0))) * 3600.0


def _pair_scatter(pa, pb, beam_red):
    """The per-source position scatter (2-D rms, robust, arcsec) between two bands' bright detections (S/N >=
    MERGE_MIN_SNR) paired within half the redder band's local beam at the first band's detection (`beam_red`: its
    mosaic's `MosaicPSF.beam_at_sky`), and the number of pairs."""
    a = pa["snr"] >= MERGE_MIN_SNR
    b = pb["snr"] >= MERGE_MIN_SNR
    if not a.any() or not b.any():
        return np.nan, 0
    ua, ub = _unit(pa["ra"][a], pa["dec"][a]), _unit(pb["ra"][b], pb["dec"][b])
    d, j = cKDTree(ub).query(ua, distance_upper_bound=_chord(beam_red.sub_beams.max() / 2.0))
    ok = np.isfinite(d)
    ok[ok] = _sep_arcsec(ua[ok], ub[j[ok]]) <= beam_red.beam_at_sky(pa["ra"][a][ok], pa["dec"][a][ok]) / 2.0
    if not ok.any():
        return np.nan, 0
    sep = _sep_arcsec(ua[ok], ub[j[ok]])
    return float(np.sqrt(np.median(sep ** 2) / np.log(2.0))), int(ok.sum())


def _measure_at(piece, config, records):
    """The catalogue's flux of PACS sources at given sky positions on one map of one band (`piece = (band, tile index,
    ra, dec, shift, offset per grid cell)`): the aperture estimator minus the cell offset, the cell id, and whether
    the fitted width exceeds 1.5 beams. Returns (band, tile index, fnu, cell id, extended)."""
    band, t, ra, dec, shift, offset_cell = piece
    tile = pacs_fit.cached_tile(config, records[band][t])
    x, y = tile.xy(*pacs_fit.to_map_frame(ra, dec, shift))
    fnu = filtered(tile, x, y, offset_cell)
    ext = tile.fitted_fwhm(x, y) > 1.5 * tile.beam_at(x, y)
    return band, t, fnu, tile.grid.cell_ids(x, y), ext


def merge_sources(config, bands, n_jobs, stage, flux_grid):
    """The region's PACS sources: the bands' blind detections merged across bands (planner ruling 2026-10-28).
    Detections of two bands within the pair's merge radius are one source (each band at most once, nearest first,
    brightest detections first); the position is the sharpest detecting band's (70, else 100, else 160); every
    detecting band's flux is the aperture estimator at that band's own detection position on its detecting map; the SESNA
    matching radius is the position band's map radius, a 160-only source's with the 70-160 merge radius in
    quadrature; CHANCE the SESNA density within LOCAL_DENSITY_ARCSEC times the radius's area. Returns the table."""
    pk = {b: bands[b]["peaks"] for b in BANDS}
    recs = {b: bands[b].get("records", []) for b in BANDS}
    psf = {b: pacs_psf.MosaicPSF(recs[b][0].mosaic_dir, pacs_fit.maps_root(config), tables=False) for b in BANDS if recs[b]}
    radius = {}                         # arcsec; None: half the redder band's local beam at the detection
    for a, b in (("P70", "P100"), ("P70", "P160"), ("P100", "P160")):
        if a not in psf or b not in psf:
            continue
        rms, n = _pair_scatter(pk[a], pk[b], psf[b])
        radius[(a, b)] = MERGE_SIGMAS * rms if n >= MERGE_MIN_PAIRS else None
        print("  merge radius %s-%s: %s" % (a, b, "%.2f arcsec (3 x rms %.2f from %d pairs)" % (radius[(a, b)], rms, n)
              if n >= MERGE_MIN_PAIRS else "half the redder band's local beam (%d pairs)" % n), flush=True)
    pos_band, members, pos_u = [], [], []
    for bi, b in enumerate(BANDS):
        p = pk[b]
        if p["fnu"].size == 0:
            continue
        u = _unit(p["ra"], p["dec"])
        order = np.argsort(-p["snr"])
        tree = cKDTree(np.array(pos_u)) if pos_u else None
        half_beam = 0.5 * psf[b].beam_at_sky(p["ra"], p["dec"])        # the default radius, at each detection of the band
        for i in order:
            best = None
            if tree is not None:
                reach = {a: (half_beam[i] if radius[(a, b)] is None else radius[(a, b)]) for a in BANDS[:bi] if (a, b) in radius}
                rmax = max(reach.values(), default=0.0)
                for k in tree.query_ball_point(u[i], _chord(rmax)):
                    if b in members[k] or (BANDS[pos_band[k]], b) not in radius:
                        continue
                    sep = _sep_arcsec(u[i], pos_u[k])
                    if sep <= reach[BANDS[pos_band[k]]] and (best is None or sep < best[1]):
                        best = (k, sep)
            if best is not None:
                members[best[0]][b] = int(i)
            else:
                pos_band.append(bi)
                members.append({b: int(i)})
                pos_u.append(u[i])
    n = len(members)
    pos_band = np.array(pos_band, np.int8)
    src = {b: np.array([m.get(b, -1) for m in members], np.int64) for b in BANDS}
    pra = np.array([pk[BANDS[pb]]["ra"][src[BANDS[pb]][k]] for k, pb in enumerate(pos_band)]) if n else np.zeros(0)
    pde = np.array([pk[BANDS[pb]]["dec"][src[BANDS[pb]][k]] for k, pb in enumerate(pos_band)]) if n else np.zeros(0)
    # each detecting band's flux at that band's own detection position (as HPPSC measures it; owner-approved
    # 2026-10-29), on that band's detecting map
    pieces, want = [], {}
    for b in BANDS:
        det = np.flatnonzero(src[b] >= 0)
        maps = pk[b]["map"][src[b][det]] if det.size else np.zeros(0, np.int32)
        cm, cid, off = bands[b]["cell_map_index"], bands[b]["cell_id"], bands[b]["cell_offset"]
        for t in np.unique(maps):
            k = det[maps == t]
            grid = _tile_geometry(recs[b][t])[0]
            oc = np.full(grid.n_cells, np.nan)
            rows = cm == t
            oc[cid[rows]] = off[rows]
            own = src[b][k]
            pieces.append((b, int(t), pk[b]["ra"][own], pk[b]["dec"][own], tuple(bands[b]["map_shift"][t]), oc))
            want[(b, int(t))] = k
    measured = pieces_module.run(functools.partial(_measure_at, config=config, records=recs), pieces, n_jobs,
                                 stage=stage, unit="PACS sources (flux at the merged position)") if pieces else []
    table = dict(ra=pra, dec=pde, position_band=pos_band, radius=radius, n=n)
    for b in BANDS:
        table["fnu_" + b] = np.full(n, np.nan)
        table["sigma_" + b] = np.full(n, np.nan)
        table["offset_" + b] = np.full(n, np.nan)
        table["extended_" + b] = np.zeros(n, bool)
        table["map_" + b] = np.where(src[b] >= 0, pk[b]["map"][np.maximum(src[b], 0)] if pk[b]["map"].size else -1, -1).astype(np.int32)
        table["snr_" + b] = np.where(src[b] >= 0, pk[b]["snr"][np.maximum(src[b], 0)] if pk[b]["snr"].size else np.nan, np.nan)
        table["detected_" + b] = src[b] >= 0
        table["ra_" + b] = np.where(src[b] >= 0, pk[b]["ra"][np.maximum(src[b], 0)] if pk[b]["ra"].size else np.nan, np.nan)
        table["dec_" + b] = np.where(src[b] >= 0, pk[b]["dec"][np.maximum(src[b], 0)] if pk[b]["dec"].size else np.nan, np.nan)
    for b, t, fnu, cid, ext in measured:
        k = want[(b, t)]
        keys = bands[b]["cell_map_index"].astype(np.int64) * (1 << 40) + bands[b]["cell_id"]
        q = np.int64(t) * (1 << 40) + cid
        row = np.clip(np.searchsorted(keys, q), 0, max(keys.size - 1, 0))
        hit = (cid >= 0) & (keys[row] == q)
        table["fnu_" + b][k] = fnu
        table["extended_" + b][k] = ext
        table["offset_" + b][k[hit]] = bands[b]["cell_offset"][row[hit]]
        if hit.any():
            table["sigma_" + b][k[hit]] = calibrated_sigma(bands[b]["scatter"][row[hit]], flux_grid, fnu[hit])
    only160 = int(((pos_band == BANDS.index("P160"))).sum())
    print("  PACS sources (merged) %d: position from P70 %d, P100 %d, P160 %d (160-only); detected in P70 %d, P100 %d, "
          "P160 %d" % (n, int((pos_band == 0).sum()), int((pos_band == 1).sum()), only160,
                       int(table["detected_P70"].sum()), int(table["detected_P100"].sum()), int(table["detected_P160"].sum())),
          flush=True)
    return table


def write_sources(path, region, bands, merged):
    """Write the region's PACS source table (the bands' blind detections merged across bands, `merge_sources`) at
    `path`: one row per PACS source, its position (from the sharpest detecting band), the SESNA matching radius, the
    expected chance matches, and per band whether it detects the source, the flux at the position and its
    uncertainty, the offset subtracted, the detection's S/N, the map and the extended flag; the merge radii, each
    band's spurious rate per sky bin and the 160-only count as attributes."""
    stem = "sources_pacs_region"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "region"
        f.attrs["REGION"] = region
        w = lambda name, data: build_module.write_dataset(f, name, data, *REGISTRY[(stem, name)])
        ds = w("RA_DEG", np.asarray(merged["ra"], np.float64))
        for (a, b), r in merged["radius"].items():
            ds.attrs["MERGE_RADIUS_%s_%s_ARCSEC" % (a, b)] = np.nan if r is None else float(r)
        ds.attrs["N_160_ONLY"] = int((merged["position_band"] == BANDS.index("P160")).sum())
        w("DEC_DEG", np.asarray(merged["dec"], np.float64))
        w("POSITION_BAND", merged["position_band"].astype(np.int8))
        for b in BANDS:
            w("DETECTED_" + b, merged["detected_" + b])
            w("FNU_%s_MJY" % b, merged["fnu_" + b])
            w("SIGMA_%s_MJY" % b, merged["sigma_" + b])
            w("CELL_OFFSET_%s_MJY" % b, merged["offset_" + b])
            w("SNR_" + b, np.asarray(merged["snr_" + b], np.float32))
            w("MAP_INDEX_" + b, merged["map_" + b])
            w("EXTENDED_" + b, merged["extended_" + b])
            w("RA_%s_DEG" % b, np.asarray(merged["ra_" + b], np.float64))
            w("DEC_%s_DEG" % b, np.asarray(merged["dec_" + b], np.float64))


def _region_inputs(config, region):
    """The region's source positions in the curated order, from `catalog.curated.read_positions` (the delivery: the
    curated product is built after the PACS stages)."""
    from sesnaimpute.catalog import curated
    _name, ra, dec = curated.read_positions(config, region)
    return ra.astype(np.float64), dec.astype(np.float64)


def build(config, regions=None, box=None, detect_only=False):
    """Writes each region's `completeness_pacs_source` product, regions one after another, through the stages of
    `pacs_stages.run_region` (one process; a rerun after a kill resumes at the first stage file missing). Returns
    `{region: {band: held-out injections}}` for `acceptance` (the command line discards it)."""
    from sesnaimpute.sky.derived import pacs_stages
    names = regions or [r.name for r in regions_module.REGIONS]
    t0 = time.time()
    held = {}
    for region in names:
        with progress_module.Stage("sky.derived.pacs_completeness", region) as st:
            path = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
            bands = pacs_stages.run_region(config, region, box, detect_only=detect_only, stage=st)
            held[region] = {b: bands[b]["held"] for b in BANDS}
            st.done(path, **{"cells_" + b: int(bands[b]["curve"].shape[0]) for b in BANDS})
    progress_module.summary("sky.derived.pacs_completeness", len(names), t0)
    return held


# ----------------------------------------------------------------------------
# the preflight (a Vista development-queue job before every production submission)

def _git_state():
    """(commit, dirty, upstream commit or None) of the checkout this module is imported from."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    git = lambda *a: subprocess.run(("git", "-C", here) + a, capture_output=True, text=True)
    head = git("rev-parse", "HEAD")
    if head.returncode:
        raise ValueError("the code at %s is not a git checkout: the preflight needs the pushed commit" % here)
    up = git("rev-parse", "@{upstream}")
    return head.stdout.strip(), bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip()), \
        (up.stdout.strip() if not up.returncode else None)


def preflight(config, region, box=None, reuse_mosaics=False, reference_product=None):
    """Everything a production submission would otherwise find out hours in, in one job (laptop or Vista development
    queue; the mosaics it builds are the ones the production job reuses): (1) the code is a clean checkout at its
    upstream (pushed) commit; (2) no mosaic product of the region exists unless `reuse_mosaics` (intended); (3) the
    inputs: HPPSC lists and observation table of every band, the HOPS table, the region's source positions, the
    `reference_product` (earlier photometry product, when given); (4) `write_product` and `write_sources` on empty
    bands (every dataset name registered); (5) the mosaic of every band, its PSF, the layers and detection kernels
    (`_prepare_stage`); (6) every band's `self_check` and `pacs_checks.normalisation_check`, all before any injection.
    Raises ValueError at the first failure, prints PREFLIGHT PASS at the end."""
    import tempfile
    from sesnaimpute.sky.derived import pacs_checks, pacs_mosaic, pacs_photometry
    t0 = time.time()
    commit, dirty, upstream = _git_state()
    print("preflight: code %s%s, upstream %s" % (commit, " (UNCOMMITTED CHANGES)" if dirty else "", upstream), flush=True)
    if dirty or upstream != commit:
        raise ValueError("the code is not the pushed commit (uncommitted changes, or HEAD differs from its upstream)")
    for b in BANDS:
        mdir = pacs_mosaic.product_dir(config, region, b, box)
        if os.path.exists(os.path.join(mdir, "mosaic.hdf5")) and not reuse_mosaics:
            raise ValueError("a mosaic product exists at %s: move it aside, or run with reuse_mosaics=True if it is intended" % mdir)
    for b in BANDS:
        h = pacs_photometry._read_hppsc(config, b)
        if not (h["ra"].size and h["obsids"]):
            raise ValueError("HPPSC %s list or observation table is empty" % b)
    # the HOPS table must read (the photometry's C4/C5); HOPS covers Orion only, so most regions match none
    n_hops = pacs_photometry._match_hops(config, region)[0].size
    print("preflight: HOPS table read, %d targets in %s (none outside Orion: C4/C5 then report nothing)" % (n_hops, region), flush=True)
    ra, dec = _region_inputs(config, region)
    if not ra.size:
        raise ValueError("no source positions for %s" % region)
    if reference_product is not None:
        pacs_photometry._c4_ratio(config, region, reference_product)
    print("preflight: inputs present (%d source positions) %.0f s" % (ra.size, time.time() - t0), flush=True)
    mag, _grid = canonical_grid()
    with tempfile.TemporaryDirectory(dir=os.path.join(config.data_root, "sky", "derived")) as tmp:
        bands = {b: _empty_band(5, mag.size) for b in BANDS}
        write_product(os.path.join(tmp, "c.hdf5"), region, mag, bands)
        merged = dict(ra=np.zeros(0), dec=np.zeros(0), position_band=np.zeros(0, np.int8), radius={("P70", "P100"): 1.0}, n=0)
        for b in BANDS:
            for k in ("fnu", "sigma", "offset", "snr", "ra", "dec"):
                merged[k + "_" + b] = np.zeros(0)
            merged["extended_" + b] = np.zeros(0, bool)
            merged["map_" + b] = np.zeros(0, np.int32)
            merged["detected_" + b] = np.zeros(0, bool)
        write_sources(os.path.join(tmp, "s.hdf5"), region, bands, merged)
    print("preflight: write_product and write_sources names registered %.0f s" % (time.time() - t0), flush=True)
    records = {}
    for b in BANDS:
        if not [r for r in pacs_fit.read_index(config, region) if r.band == b]:
            continue
        if not os.path.exists(os.path.join(pacs_mosaic.product_dir(config, region, b, box), "mosaic.hdf5")):
            pacs_mosaic.build_mosaic(config, region, b, box=box, n_jobs=config.n_jobs)
        records[b] = pacs_fit.mosaic_record(config, region, b, box)
        print("preflight: %s mosaic and PSF built %.0f s" % (b, time.time() - t0), flush=True)
    _prepare_stage(config, list(records.values()), config.n_jobs, None)
    print("preflight: layers and detection kernels prepared %.0f s" % (time.time() - t0), flush=True)
    for b, rec in records.items():
        self_check(config, rec)
        pacs_checks.normalisation_check(config, rec)
    print("PREFLIGHT PASS (%s, %s, code %s) %.0f s" % (region, ", ".join(records), commit[:7], time.time() - t0), flush=True)


# ----------------------------------------------------------------------------
# acceptance (printed, by function call; nothing is written)

def _read_band(f, band):
    """One band of the product, as a dict of arrays, with the pooling matrices rebuilt from its cells."""
    g = lambda name: f[name][:]
    d = dict(curve=g("CURVE_" + band), mag=g("MAG_GRID_" + band), flux90=g("FLUX90_%s_MJY" % band),
             n_pos=g("N_POS_" + band), scatter=g("SCATTER_%s_MJY" % band),
             bias=g("BIAS_%s_MJY" % band), sigma=g("CELL_SIGMA_%s_MJY" % band), offset=g("CELL_OFFSET_%s_MJY" % band),
             sky_bin=g("SKY_BIN_" + band).astype(np.int64), cell_id=g("CELL_ID_" + band), cell_map=g("CELL_MAP_INDEX_" + band),
             map_id=g("MAP_ID_" + band), map_wcs=g("MAP_WCS_" + band), map_shape=g("MAP_SHAPE_" + band),
             map_shift=g("MAP_SHIFT_%s_ARCSEC" % band), source_row=g("SOURCE_ROW")[:, BANDS.index(band)])
    d["r_match"] = float(np.nanmedian(f["MAP_R_MATCH_%s_ARCSEC" % band][:]))
    d["flux"] = 1000.0 * ZERO_POINT_JY * 10.0 ** (-d["mag"].astype(np.float64) / 2.5)
    xy = np.zeros((d["cell_id"].size, 2))
    for m in range(d["map_id"].size):
        rows = np.flatnonzero(d["cell_map"] == m)
        if rows.size:
            grid = pacs_fit.MapGrid.from_wcs(WCS(fits.Header.fromstring(d["map_wcs"][m].decode())).celestial,
                                             tuple(d["map_shape"][m]))
            xy[rows] = cell_centres_arcsec(grid, d["cell_id"][rows]) + np.array([m * 1e7, 0.0])
    d["Ws"] = pool_matrices(xy)
    return d


def _binomial_error(fraction, n):
    """One-sigma binomial error of a fraction from n trials, the fraction held off 0 and 1 by half a trial."""
    p = np.clip(fraction, 0.5 / np.maximum(n, 1), 1 - 0.5 / np.maximum(n, 1))
    return np.sqrt(p * (1 - p) / np.maximum(n, 1))


def _verdict(value, low, high=None):
    return "pass" if value >= low and (high is None or value <= high) else "FAIL"


def _acceptance_c1(band, d, held):
    """C1: the held-out half's curve, built exactly as the stored one (S/N pooling, read at each cell's own noise),
    against the stored curve: of the (cell, flux plane) on the curve's transition (either half between 0.02 and
    0.98) with at least MIN_INJECTIONS held-out injections behind them, the share whose difference lies within the
    combined 1-sigma binomial error of the two halves (60-75 %), within 2 sigma (>= 93 %), within 0.03 absolute
    (>= 95 %); and the share of cells whose FLUX90 of the two halves agree within 10 % (>= 90 %)."""
    n_row, n_plane = d["curve"].shape
    fh, tot = curve_on_flux_grid(d["Ws"], held["row"].astype(np.int64), held["plane"].astype(np.int64),
                                 held["recovered"].astype(bool), d["sigma"], d["flux"], n_row)
    fb = d["curve"].astype(float) / STORAGE_SCALE
    # the comparison is of two independent halves of equal size, so the difference carries both halves' binomial
    # error; only the curve's transition is tested (a plane at 0 or 1 in both halves agrees trivially)
    transition = ((fh > 0.02) & (fh < 0.98)) | ((fb > 0.02) & (fb < 0.98))
    use = (tot >= MIN_INJECTIONS) & (d["n_pos"][:, None] >= MIN_INJECTIONS) & transition
    diff = np.abs(fb - fh)
    sig = np.hypot(_binomial_error(fh, tot), _binomial_error(fb, tot))
    n = int(use.sum())
    if n == 0:
        print("  C1 %s: no (cell, plane) has %d held-out injections" % (band, MIN_INJECTIONS))
        return
    p1, p2, pa = (100 * (use & c).sum() / n for c in (diff <= sig, diff <= 2 * sig, diff <= 0.03))
    print("  C1 %s: %d (cell, plane) on the transition with >= %d held-out injections (%.1f %% of all): within 1 sigma %.1f %% (bar >= 60: %s), "
          "within 2 sigma %.1f %% (bar >= 93: %s); within 0.03 absolute %.1f %% (information)"
          % (band, n, MIN_INJECTIONS, 100 * n / diff.size, p1, _verdict(p1, 60), p2, _verdict(p2, 93), pa))
    worst = np.argsort(np.where(use, diff / np.maximum(sig, 1e-9), -1).ravel())[::-1][:5]
    for k in worst:
        r, p = divmod(int(k), n_plane)
        print("      worst: map %s cell %d plane m=%.2f: curve %.3f held-out %.3f (%d injections)"
              % (d["map_id"][d["cell_map"][r]].decode(), d["cell_id"][r], d["mag"][p], fb[r, p], fh[r, p], tot[r, p]))
    f90 = flux90(fh, d["flux"])
    ok = np.isfinite(f90) & np.isfinite(d["flux90"]) & (d["n_pos"] >= MIN_INJECTIONS)
    rel = np.abs(f90 - d["flux90"]) / d["flux90"]
    pf = 100 * np.mean(rel[ok] <= 0.10) if ok.any() else np.nan
    print("      FLUX90 of the two halves within 10 %%: %.1f %% of %d cells (bar >= 90: %s; median difference %.1f %%)"
          % (pf, int(ok.sum()), _verdict(pf, 90) if ok.any() else "no cells", 100 * np.median(rel[ok]) if ok.any() else np.nan))


def _robust(v):
    return MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))) if v.size else np.nan


def _at_injected_flux(values, flux_grid, row, flux):
    """values (n_row, n_plane) on the flux grid read for each injection at its own row and injected flux,
    linearly in log flux between the bracketing planes (held at the grid's ends)."""
    lf = np.log10(flux_grid[::-1])
    t = np.interp(np.log10(flux), lf, np.arange(lf.size, dtype=float))
    k0 = np.minimum(np.floor(t).astype(np.int64), lf.size - 2)
    w = t - k0
    v = values[:, ::-1]
    return (1 - w) * v[row, k0] + w * v[row, k0 + 1]


def _acceptance_c2(band, d, held):
    """C2 (RULINGS_3 D3), the held-out half against the error built from the BUILD half alone (`held["build_errors"]`).
    Aperture path, per noise class and S/N bin, inside the flux grid: the robust scatter of held-out dflux /
    SCATTER_build read at the injection's own cell and flux (expected 1.02 +/- 0.05, bar 1.00 +/- 0.10 where >= 200
    residuals), over the injections farther than BLEND_RADIUS_BEAMS from every detection (a source within it is an
    upper bound). Bias above FLUX90 inside the flux grid: |median dflux / injected| < 1 %."""
    e = held["build_errors"]
    bins_of_plane = snr_bin(snr_grid())
    row = held["row"].astype(np.int64)
    plane = held["plane"].astype(np.int64)
    k = bins_of_plane[plane]
    flux = held["flux"].astype(np.float64)
    median_sigma = np.array([np.median(d["sigma"][d["cell_map"] == m]) if (d["cell_map"] == m).any() else np.nan
                             for m in range(d["map_id"].size)])
    noisy = d["sigma"] > 2 * median_sigma[d["cell_map"]]
    in_grid = lambda f: (f >= d["flux"].min()) & (f <= d["flux"].max())
    ap = ~held["near"].astype(bool) & in_grid(flux)
    s_build = np.full(row.size, np.nan)
    s_build[ap] = _at_injected_flux(e["scatter"], d["flux"], row[ap], flux[ap])
    for label, cls in (("quiet", ~noisy[row]), ("noisy", noisy[row])):
        for j in range(len(SNR_BIN_CENTRES)):
            sel = ap & cls & (k == j) & np.isfinite(s_build) & (s_build > 0)
            z = held["dflux"][sel] / s_build[sel]
            verdict = _verdict(_robust(z), 0.90, 1.10) if sel.sum() >= 200 else "fewer than 200"
            print("  C2 %s aperture %s, S/N bin %d: %d held-out residuals, normalised scatter %.3f (expected 1.02 +/- 0.05, "
                  "bar 1.00 +/- 0.10: %s)" % (band, label, j, int(sel.sum()), _robust(z), verdict))
    f90 = d["flux90"][row]
    above = ap & np.isfinite(f90) & (flux > f90) & in_grid(flux)
    if above.sum():
        bias = 100 * np.median(held["dflux"][above] / flux[above])
        print("  C2 %s aperture bias above FLUX90: %d, median residual / injected %+.2f %% (bar |.| < 1 %%: %s)"
              % (band, int(above.sum()), bias, _verdict(-abs(bias), -1)))
    # the same on the detected sources' fluxes (RULINGS_5: the flux of the PACS source the blind detection found)
    det = held["recovered"].astype(bool) & np.isfinite(held["det_flux"])
    ddet = held["det_flux"].astype(np.float64) - flux
    for label, cls in (("quiet", ~noisy[row]), ("noisy", noisy[row])):
        for j in range(len(SNR_BIN_CENTRES)):
            sel = ap & det & cls & (k == j) & np.isfinite(s_build) & (s_build > 0)
            z = ddet[sel] / s_build[sel]
            verdict = _verdict(_robust(z), 0.90, 1.10) if sel.sum() >= 200 else "fewer than 200"
            print("  C2 %s detected-source flux %s, S/N bin %d: %d held-out residuals, normalised scatter %.3f (bar 1.00 "
                  "+/- 0.10: %s)" % (band, label, j, int(sel.sum()), _robust(z), verdict))
    above_d = above & det
    if above_d.sum():
        bias = 100 * np.median(ddet[above_d] / flux[above_d])
        print("  C2 %s detected-source flux bias above FLUX90: %d, median residual / injected %+.2f %% (bar |.| < 1 %%: %s)"
              % (band, int(above_d.sum()), bias, _verdict(-abs(bias), -1)))


def _report_detection(band, d, held):
    """RULINGS_5's detection checks: recovery of the held-out injections at S/N >= 20 (bar >= 99 %), the matching
    radius, and the curve's floor (its value at the faintest flux plane, the chance of lying within the radius of an
    existing PACS source) by sky bin."""
    snr = snr_grid()[held["plane"].astype(np.int64)]
    bright = snr >= 20.0
    frac = float(held["recovered"][bright].mean()) if bright.any() else np.nan
    print("  detection %s: matching radius (median over maps) %.2f arcsec; held-out injections at S/N >= 20 recovered %.4f of %d (bar >= 0.99: "
          "%s)" % (band, d["r_match"], frac, int(bright.sum()), _verdict(frac, 0.99)))
    floor = d["curve"][:, -1].astype(np.float64) / STORAGE_SCALE
    print("  detection %s: the curve's floor (faintest plane) by sky bin, faint to bright: %s" % (band, "  ".join(
        "%.4f" % (np.median(floor[d["sky_bin"] == j]) if (d["sky_bin"] == j).any() else np.nan) for j in range(SKY_BINS))))


def _held_path(config, region, band, box=None):
    from sesnaimpute.sky.derived import pacs_stages
    return os.path.join(pacs_stages.stage_dir(config, region, box), "held_%s.pkl" % band)


def _read_held(config, region, box=None):
    """`{band: held-out half}` from the stage store's `held_<band>.pkl` files (the bands that have one)."""
    held = {}
    for band in BANDS:
        path = _held_path(config, region, band, box)
        if os.path.exists(path):
            with open(path, "rb") as f:
                held[band] = pickle.load(f)
    return held


def acceptance(config, region, held_out=None, box=None):
    """Prints C1-C3 for the region's product and the held-out half kept from the build that wrote it
    (`held_out`: `{band: held-out injections}`, `build`'s return value for the region; None reads the stage
    store's `held_<band>.pkl`, `box` naming the store of a development slice). C1: the stored curve and 0.9 flux
    against the held-out half. C2: the held-out recovered injections' bias and scatter against the stored SCATTER
    (forced and detected-source fluxes); the detection checks (`_report_detection`). Nothing is written."""
    if held_out is None:
        held_out = _read_held(config, region, box)
    path = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
    with h5py.File(path, "r") as f:
        for band in BANDS:
            held = held_out.get(band)
            if held is None or f["CURVE_" + band].shape[0] == 0:
                continue
            d = _read_band(f, band)
            print("acceptance %s %s: %d cells, %d held-out injections" % (region, band, d["curve"].shape[0], held["row"].size), flush=True)
            _acceptance_c1(band, d, held)
            _acceptance_c2(band, d, held)
            _report_detection(band, d, held)


def finish_region(config, region, box=None, reference_product=None):
    """What a region script ran after the completeness product was written: the completeness acceptance (the
    held-out halves read from the stage store), `pacs_photometry.build` and `pacs_photometry.acceptance`
    (`reference_product`: the earlier photometry product's path for C4); the held-out halves are removed once the
    completeness acceptance and the photometry have run (a rerun after a failure repeats the whole of it). Returns
    nothing; the products are the files."""
    from sesnaimpute.sky.derived import pacs_photometry
    acceptance(config, region, None, box)
    pacs_photometry.build(config, regions=[region], box=box)
    pacs_photometry.acceptance(config, region, reference_product=reference_product, box=box)
    for band in BANDS:
        path = _held_path(config, region, band, box)
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    run(build)
