"""The Herschel PACS completeness of the SESNA sources, per region and band, MEASURED by recovering
artificial point sources (PACS_COMPLETENESS_PLAN.md v2; SPEC_PACS_FARIR_DRAFT.md sections 5 and 11.3).

`build(config, regions=None)` writes one product per region,
`sky/derived/pacs/completeness_pacs_source__<Region>.hdf5`, in EXACTLY the layout of
`catalog/sesna/completeness_sesna_source__<Region>.hdf5` for the bands P70, P100 and P160
(`CURVE_<B>`, `MAG_GRID_<B>`, `SOURCE_ROW`, `ENVELOPE_<B>`, `COMPLETED_FROM_<B>`, `N_POS_<B>`), plus per-cell
columns aligned with the CURVE rows (`CELL_RA/DEC/ID/MAP_INDEX/OBSID/MAP_<B>`, `CELL_SIGMA_<B>_MJY`,
`CELL_OFFSET_<B>_MJY`, `N_SIGMA_POS_<B>`, `FLUX90_<B>_MJY`, `BIAS_<B>_MJY`, `SCATTER_<B>_MJY`,
`SCATTER_BLEND_<B>_MJY`, `CONTAMINATION_<B>`) that the photometry stage reads.

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
3. Injection, in signal-to-noise. For each plane of the S/N grid (`snr_grid`: 0.3 to 1000 x CELL_SIGMA, 0.1 dex),
   batches of artificial sources (the band's point-spread template, held to the release EEF's total-flux
   normalisation by `pacs_fit.Tile`) of flux S/N x CELL_SIGMA of the cell each lands in are added to a copy of the
   map at positions uniform over the tile's valid pixels inside the region's cells, no two closer than the injected
   patch's half-width plus the sky annulus's outer radius (no injected light in another's aperture or annulus),
   and measured through the catalogue's own path: the aperture where the position is farther than
   BLEND_RADIUS_BEAMS x beam from every detected source, the blend-group fit (the injected source one more member
   of the group of the detected sources within that radius) otherwise. RECOVERED means flux - CELL_OFFSET >= 5
   CELL_SIGMA of the injected position's cell; no flux tolerance. An injection within half a beam of a detected
   source is dropped from every count (a SESNA source there is an unseparable blend). Batches alternate by seed
   parity: even seeds build the product, odd seeds are the held-out half kept in memory for `acceptance`.
4. Pooling in S/N. Recovery depends on flux over the cell's noise, so a cell's counts at an S/N plane are the sums
   over the cells of its map within POOL_ARCSEC (widened where short), whatever their noise. CURVE at flux F is the
   pooled fraction read at F / the cell's own CELL_SIGMA (linear in log S/N): the shape is the neighbourhood's, the
   scale and offset the cell's own. FLUX90 is the faintest flux at which it still reaches 0.9; BIAS and SCATTER the
   median and 1.4826 x MAD of (recovered - injected flux) / CELL_SIGMA over all pooled covered aperture-path
   injections, detected or not, read the same way and multiplied by the cell's CELL_SIGMA; SCATTER_BLEND the same for the group
   path's targeted injections.
5. CONTAMINATION: the fraction of the faintest-S/N injections the rule counts as detections, the chance that
   unrelated emission in the aperture passes the rule, pooled over the band's maps in SKY_BINS bins of within-map
   sky rank.

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
from sesnaimpute.sky.derived.pacs_photometry import (BLEND_RADIUS_BEAMS, BLEND_RHO_LIMIT, SNR_BIN_CENTRES, SNR_BIN_EDGES,
                                                     _blend_components, singleton_pass)
from sesnaimpute.sky.derived import pacs_psf
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

#: injections: no two artificial sources of a batch closer than the injected patch's half-width plus the sky
#: annulus's outer radius (so no injected light reaches another's aperture or annulus), drawn at one per this
#: many squared separations of valid area
SEPARATION_AREAS = 2.0
#: an injection this close (in beams) to a detected source is dropped from every count
DROP_BEAMS = 0.5
#: artificial sources are injected in signal-to-noise: at plane k each source's flux is SNR_GRID[k] times the
#: CELL_SIGMA of the cell it lands in, so a plane is one point on the detection curve's own axis for every cell.
#: log10 S/N from SNR_LOG_MIN to SNR_LOG_MAX in SNR_LOG_STEP (0.1 dex, the flux grid's 0.25 mag)
SNR_LOG_MIN = -0.5
SNR_LOG_MAX = 3.0
SNR_LOG_STEP = 0.1
#: a cell's counts at an S/N plane are pooled over the cells of its map whose centres lie within this radius of
#: its own; where the pooled count is still short of MIN_INJECTIONS the radius is widened to 2 x, then 3 x
POOL_ARCSEC = 120.0
POOL_WIDENINGS = (1, 2, 3)
MIN_INJECTIONS = 100
#: the blend path: per detected source, this many injections per plane and half, uniform in the ring from DROP_BEAMS
#: to BLEND_RADIUS_BEAMS beams around it, measured through the group path; they set SCATTER_BLEND only. A cell's
#: SCATTER_BLEND needs this many recovered ones in its pool (one source's worth: MIN_INJECTIONS would leave it
#: empty almost everywhere, the group path being rare)
BLEND_TARGETED = 40
MIN_BLEND_INJECTIONS = 30
#: the first round of batches per plane and half aims at this multiple of MIN_INJECTIONS in the pooled
#: interior; further batches, restricted to the cells whose pooled count is short, follow until none is
INTERIOR_SAFETY = 1.3
MAX_EXTRA_BATCHES = 40
#: the recovered fraction FLUX90 is read at
FLUX90_FRACTION = 0.9


def snr_bin(snr):
    """The error bin (0-3) of an injected S/N."""
    return np.digitize(np.asarray(snr, float), SNR_BIN_EDGES)
#: CONTAMINATION: the cells of each map are ranked by their annulus sky and cut into this many equal bins; a
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

def cell_fits(tile, ids, seed, psf=False):
    """The catalogue's aperture measurement at N_POS uniform positions in each cell of `ids` on `tile`. Returns
    b (flux mJy), gain (1 where covered), the annulus sigma, valid, clean (farther than one beam from every bright
    neighbour), the annulus sky (mJy per pixel) and, with `psf`, the single PSF-fit flux at the same positions
    (NaN where that fit is not covered; else None), each (n_cell, N_POS)."""
    x, y = tile.grid.sample_pixels(ids, N_POS, seed)
    a = tile.aperture(x.ravel(), y.ravel())
    shape = x.shape
    gain = np.where(a.covered, 1.0, np.nan)
    valid = (a.covered & np.isfinite(a.flux)).reshape(shape)
    clean = ~tile.near_bright(x, y)
    psf_flux = None
    if psf:
        fit = tile.fit_positions(x.ravel(), y.ravel())
        psf_flux = np.where(fit.covered & np.isfinite(fit.flux), fit.flux, np.nan).reshape(shape)
    return (a.flux.reshape(shape), gain.reshape(shape), a.annulus_sigma.reshape(shape), valid, clean,
            a.sky.reshape(shape), psf_flux)


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
    b, gain, ps, valid, clean, _sky, _psf = cell_fits(tile, all_ids, 0)
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
    b, gain, ps, valid, clean, sky, psf_flux = cell_fits(tile, all_ids, tile_index + 1, psf=True)
    n_valid, n_clean, mad, g, pps, off = cell_statistics(b, gain, ps, valid, clean)
    sigma, n_sigma = block_sigma(tile.grid, ids, all_ids, mad, n_clean)
    offset = cell_offset(tile.grid, ids, all_ids, off, n_clean)
    # the joint PSF fit's own empty-sky offset, built exactly as the aperture's (its clean positions' median); the two
    # estimators have different nulls, so each flux has its own offset subtracted
    psf_use = clean & np.isfinite(psf_flux)
    with np.errstate(all="ignore"):
        off_psf = np.nanmedian(np.where(psf_use, psf_flux, np.nan), axis=1)
    offset_psf = cell_offset(tile.grid, ids, all_ids, off_psf, psf_use.sum(axis=1))
    ref = refs[tile_index]
    # a block with no clean cell (wholly within a beam of bright sources) takes the tile's median noise;
    # N_SIGMA_POS 0 marks it
    sigma = np.where(np.isfinite(sigma), sigma, ref["median"])
    k = np.searchsorted(all_ids, ids)
    n_valid = n_valid[k]
    with np.errstate(all="ignore"):
        cell_sky = np.nanmedian(np.where(valid, sky, np.nan), axis=1)[k]
    covered = (n_valid >= MIN_VALID_POSITIONS) & np.isfinite(sigma) & (sigma > 0)
    ra, dec = pacs_fit.from_map_frame(*tile.grid.centres_icrs(ids), ref["shift"][:2])
    return dict(tile=tile_index, ids=ids, covered=covered,
                n_sigma=np.minimum(n_sigma, np.iinfo(np.int16).max).astype(np.int16),
                sigma=sigma, offset=offset, offset_psf=offset_psf, sky=cell_sky, ra=ra, dec=dec)


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
    NaN where none does."""
    med = np.full(n_row, np.nan)
    sc = np.full(n_row, np.nan)
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
    (fraction, injections behind it: the smaller of the two bracketing planes' pooled counts), (n_row, n_plane)."""
    n_x = snr_grid().size
    tot, got = pooled_counts(Ws, row, plane.astype(np.int64), recovered, n_row, n_x)
    with np.errstate(all="ignore"):
        fx = np.where(tot > 0, got / tot, np.nan)
    k0, w, _ = _snr_brackets(sigma_rows, flux_grid)
    fraction = _at_flux(fx, k0, w)
    r = np.arange(n_row)[:, None]
    count = np.where(w <= 0.0, tot[r, k0], np.where(w >= 1.0, tot[r, k0 + 1], np.minimum(tot[r, k0], tot[r, k0 + 1])))
    return np.nan_to_num(fraction, nan=0.0), count


def error_on_flux_grid(Ws, row, plane, dflux, sigma_rows, flux_grid, n_row, min_n=MIN_INJECTIONS):
    """BIAS-like and SCATTER-like arrays on the flux grid from injections made in S/N: per S/N plane, the pooled
    median and 1.4826 x MAD of (recovered - injected flux) / CELL_SIGMA of the injection's own cell, read for each
    row at F / its own CELL_SIGMA and multiplied by that CELL_SIGMA; beyond the grid's brightest S/N both grow in
    proportion to the flux (the brightest plane's value relative to its flux). (n_row, n_plane) each."""
    n_x = snr_grid().size
    sig = np.asarray(sigma_rows, float)
    med_x = np.full((n_row, n_x), np.nan)
    sc_x = np.full((n_row, n_x), np.nan)
    z = dflux.astype(np.float64) / sig[row]
    for p in range(n_x):
        sel = plane == p
        med_x[:, p], sc_x[:, p] = pooled_median_scatter(Ws, row[sel], z[sel], n_row, min_n)
    k0, w, beyond = _snr_brackets(sig, flux_grid)
    scale = sig[:, None] * beyond
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
        self.offset_psf = np.full(grid.n_cells, np.nan)
        self.sigma[self.ids] = inp["sigma"]
        self.offset[self.ids] = inp["offset"]
        self.offset_psf[self.ids] = inp["offset_psf"]
        self.beam_px = tile.beam_px
        self.beam_area_px = np.pi / (4 * np.log(2)) * tile.beam_px ** 2
        self.separation_px = tile.injection_half_width() + APERTURES[tile.record.band][2] / tile.pixel_arcsec
        # the real detections: the photometry's own singleton pass on the sources it measures on this tile
        self.det_tree = None
        self.det_xy = np.zeros((0, 2))
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
                self.det_xy = self.src_xy[det]
                self.det_tree = cKDTree(self.det_xy)
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
        SEPARATION_AREAS squared separations, none within DROP_BEAMS of a detected source. Returns (x, y, distance
        in pixels to the nearest detected source, inf if none)."""
        target = max(int(round(rows.size * self.cell_px * self.valid_fraction / (SEPARATION_AREAS * self.separation_px ** 2))), 1)
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

    def blend_positions(self, rng):
        """The targeted blend set of one round: one position per detected source, uniform in area in the ring
        from DROP_BEAMS to BLEND_RADIUS_BEAMS beams around it, on valid pixels, at most one inside any
        separation disc, none within DROP_BEAMS of any detected source. Returns (x, y, distance to the nearest
        detected source in pixels)."""
        n = self.det_xy.shape[0]
        if n == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        r = self.beam_px * np.sqrt(rng.uniform(DROP_BEAMS ** 2, BLEND_RADIUS_BEAMS ** 2, n))
        th = rng.uniform(0, 2 * np.pi, n)
        x, y = self.det_xy[:, 0] + r * np.cos(th), self.det_xy[:, 1] + r * np.sin(th)
        ok = self._valid(x, y)
        x, y = x[ok], y[ok]
        perm = rng.permutation(x.size)
        x, y = x[perm], y[perm]
        keep = self._thin(x, y, x.size)
        x, y = x[keep], y[keep]
        d, _ = self.det_tree.query(np.c_[x, y])
        far = d >= DROP_BEAMS * self.beam_px
        return x[far], y[far], d[far]

    def flux_at(self, x, y, snr):
        """The flux (mJy) of an artificial source of S/N `snr` at each (x, y): snr x CELL_SIGMA of its cell."""
        return snr * self.sigma[self.tile.grid.cell_ids(x, y)]

    def batch(self, rng, rows, snr):
        """Injects one batch of sources of S/N `snr` at random positions of the cells of `rows`, measures each
        through the catalogue's own path, restores the map (`measure_positions`)."""
        x, y, d = self.positions(rng, rows)
        return self.measure_positions(x, y, d, self.flux_at(x, y, snr))

    def measure_positions(self, x, y, d, flux):
        """Injects `flux` (mJy, scalar or one per position) at the map pixels (x, y), `d` the distance (pixels) of
        each to the nearest detected source, measures each through the catalogue's own path, restores the map.
        Returns dict(row, path (1 aperture, 2 group), recovered, separable (the group fit's correlation within
        BLEND_RHO_LIMIT; always True on the aperture path), dflux (recovered - injected flux, mJy), flux
        (injected, mJy) for the injections on covered data (dflux is measured flux minus the cell offset minus the
        injected flux, the reported flux's error); n; the seconds spent in each step and the number of
        injections on each path)."""
        tile = self.tile
        n = x.size
        out = dict(row=np.zeros(0, np.int32), path=np.zeros(0, np.int8), recovered=np.zeros(0, bool),
                   separable=np.zeros(0, bool), dflux=np.zeros(0, np.float32), flux=np.zeros(0, np.float32),
                   sigma_formal=np.zeros(0, np.float32), n=0,
                   seconds=dict(inject=0.0, aperture=0.0, group=0.0), count=dict(aperture=0, group=0))
        if n == 0:
            return out
        flux = np.broadcast_to(np.asarray(flux, float), x.shape)
        cid = tile.grid.cell_ids(x, y)
        sigma = self.sigma[cid]
        group = d < BLEND_RADIUS_BEAMS * self.beam_px
        # each path's flux minus its own estimator's empty-sky offset (aperture; joint PSF fit)
        offset = np.where(group, self.offset_psf[cid], self.offset[cid])
        sigma_formal = np.full(n, np.nan)
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
                sigma_formal[gr] = gfit.sigma[last]
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
                   recovered=recovered[keep], separable=separable[keep],
                   dflux=(flux_out - offset - flux).astype(np.float32)[keep],
                   sigma_formal=sigma_formal.astype(np.float32)[keep],
                   flux=flux.astype(np.float32)[keep], n=int(keep.sum()),
                   seconds=dict(inject=t_inject, aperture=t_ap, group=t_gr),
                   count=dict(aperture=int((~group)[keep].sum()), group=int(group[keep].sum())))
        return out


def _inject_half(injector, Ws, plane, snr, half, n_row, interior_pooled):
    """Batches of one S/N plane and one half (0 build, 1 held out: the seed's parity), enough that every cell's
    pooled count reaches MIN_INJECTIONS where the cells allow: a first round sized for the interior, then
    batches restricted to the cells within the pooling radius of a short cell. Returns the injections, the
    seconds spent and the injections per path."""
    W = Ws[0]
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
        r = injector.batch(np.random.default_rng([injector.seed_key, plane, 2 * seed + half]), rows, snr)
        recs.append(r)
        count += np.bincount(r["row"], minlength=n_row)
        for k in seconds:
            seconds[k] += r["seconds"][k]
        for k in n_by_path:
            n_by_path[k] += r["count"][k]
    cat = lambda key, dt: np.concatenate([r[key] for r in recs]) if recs else np.zeros(0, dt)
    return (dict(row=cat("row", np.int32), path=cat("path", np.int8), recovered=cat("recovered", bool),
                 separable=cat("separable", bool), dflux=cat("dflux", np.float32), flux=cat("flux", np.float32),
                 sigma_formal=cat("sigma_formal", np.float32)), seconds, n_by_path)


def _inject_blend(injector, plane, snr, half):
    """The targeted blend set of one S/N plane and one half: BLEND_TARGETED rounds, each one position per detected
    source in the ring around it, measured through the group path. Returns the injections (row, recovered,
    dflux), the seconds spent in the group fits and how many were made."""
    recs = []
    seconds = 0.0
    for k in range(BLEND_TARGETED):
        rng = np.random.default_rng([injector.seed_key, plane, 2 * (1000 + k) + half])
        x, y, d = injector.blend_positions(rng)
        r = injector.measure_positions(x, y, d, injector.flux_at(x, y, snr))
        recs.append(r)
        seconds += r["seconds"]["group"]
    cat = lambda key, dt: np.concatenate([r[key] for r in recs]) if recs else np.zeros(0, dt)
    return dict(row=cat("row", np.int32), recovered=cat("recovered", bool), separable=cat("separable", bool),
                dflux=cat("dflux", np.float32), flux=cat("flux", np.float32),
                sigma_formal=cat("sigma_formal", np.float32)), seconds


def pool_product(Ws, rec, flux_grid, sigma_rows, n_row):
    """The curve from the build half's injections made in S/N (`rec`: row, plane (S/N), recovered): the recovered
    fraction (encoded), its 0.9 flux, and the injections behind the faintest flux plane (`curve_on_flux_grid`)."""
    fraction, count = curve_on_flux_grid(Ws, rec["row"], rec["plane"], rec["recovered"], sigma_rows, flux_grid, n_row)
    return dict(curve=np.rint(fraction * STORAGE_SCALE).astype(np.uint16),
                flux90=flux90(fraction, flux_grid).astype(np.float32), n_pos=count[:, -1].astype(np.int32),
                pooled_total=count)


def _robust_scatter(v):
    return MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))) if v.size else np.nan


def cell_errors(Ws, flux_grid, sigma_rows, rec, blend):
    """The error of a reported flux from injections (`rec`, `blend`: row, plane (S/N), path, separable, dflux =
    measured - own offset - injected, sigma_formal). Aperture path (`error_on_flux_grid`): per S/N plane, the
    pooled median and 1.4826 x MAD of dflux / CELL_SIGMA over ALL covered aperture-path injections (detected or not)
    of the cells within the pooling radius, read for each row at F / its own CELL_SIGMA and multiplied by it:
    SCATTER and BIAS (n_row, n_plane). Joint fit, for the map: SCATTER_BLEND[bin] = the robust scatter of
    dflux / sigma_formal over all separable targeted injections in each S/N bin, N_SCATTER_BLEND[bin]."""
    n_row = len(sigma_rows)
    sel = (rec["path"] == 1) & rec["separable"].astype(bool)
    bias, scatter = error_on_flux_grid(Ws, rec["row"][sel].astype(np.int64), rec["plane"][sel].astype(np.int64),
                                       rec["dflux"][sel], sigma_rows, flux_grid, n_row)
    n_bin = len(SNR_BIN_CENTRES)
    bins_of_plane = snr_bin(snr_grid())
    bsel = blend["separable"].astype(bool) & np.isfinite(blend["sigma_formal"]) & (blend["sigma_formal"] > 0)
    bk = bins_of_plane[blend["plane"][bsel].astype(np.int64)]
    bz = blend["dflux"][bsel].astype(np.float64) / blend["sigma_formal"][bsel].astype(np.float64)
    scatter_blend = np.array([_robust_scatter(bz[bk == j]) for j in range(n_bin)])
    n_blend = np.array([int((bk == j).sum()) for j in range(n_bin)])
    return dict(scatter=scatter, bias=bias, scatter_blend=scatter_blend.astype(np.float32),
                n_scatter_blend=n_blend.astype(np.int32))


# ----------------------------------------------------------------------------
# self-checks

ZERO_CHECK_FLUX_MJY = 500.0
ZERO_CHECK_POSITIONS = 20
ZERO_CHECK_TOLERANCE = 0.005
REAL_CHECK_POSITIONS = 200
REAL_CHECK_TOLERANCE = 0.01


def self_check(config, record):
    """The template's normalisation, on a ZERO image: ZERO_CHECK_FLUX_MJY injected at ZERO_CHECK_POSITIONS empty-sky
    positions of `record`'s map, the aperture SUM with no sky subtraction divided by the injected flux times the
    map's encircled-energy fraction at the aperture radius (total-flux normalisation): every position must be
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
    eef = pacs_psf.physical_aperture_fraction(config, record, "total", False)
    expected = ZERO_CHECK_FLUX_MJY * eef
    ratio = (f + sky * area)[cov] / expected
    sky_share = (sky * area)[cov] / expected
    print("  self-check (zero image) %s %s: %d positions, aperture sum / (injected x EEF_total(r_ap) = %.4f): min %.4f "
          "max %.4f; the catalogue's sky term on a zero image removes %.2f %% of it (information only)"
          % (record.band, record.map_id, ratio.size, eef, ratio.min() if ratio.size else np.nan,
             ratio.max() if ratio.size else np.nan, 100 * sky_share.mean() if ratio.size else np.nan), flush=True)
    if ratio.size == 0 or np.abs(ratio - 1).max() > ZERO_CHECK_TOLERANCE:
        raise ValueError("%s: the injected template is not normalised to total flux within 0.5 %%" % record.band)


def _real_check(injector, flux_mjy, half):
    """The catalogue's own measurement in its own regime: REAL_CHECK_POSITIONS empty-sky positions of the real map
    (farther than one beam from every detected source), the aperture flux after injecting `flux_mjy` minus before,
    over `flux_mjy`. Returns the ratios."""
    tile = injector.tile
    rng = np.random.default_rng([injector.seed_key, 99999, half])
    x, y, d = injector.positions(rng, np.arange(injector.ids.size))
    far = d >= BLEND_RADIUS_BEAMS * injector.beam_px
    x, y = x[far][:REAL_CHECK_POSITIONS], y[far][:REAL_CHECK_POSITIONS]
    before = tile.measure(x, y)
    with tile.injected(x, y, flux_mjy):
        after = tile.measure(x, y)
    ok = before.covered & after.covered
    return (after.flux[ok] - before.flux[ok]) / flux_mjy


#: the injection work of a map is cut into pieces of its (S/N plane, half) pairs, about INJECTION_PIECE_ROWS rows x
#: all pairs of work each (a map of that many rows is one piece; the largest maps one pair per piece), so a large
#: mosaic spreads over every worker; every pair keeps its own seeds, so the injections do not depend on the cut
INJECTION_PIECE_ROWS = 512

_INJECTION_STATE = {}


def _injection_state(config, record, inp, t):
    """The process-local Injector of map `t`, its pooling matrix (POOL_ARCSEC) over all rows and the first round's
    expected pooled count of an interior cell: built on the first piece of the map a worker takes, kept (one map at
    a time) for the map's further pieces."""
    key = record.path
    if key not in _INJECTION_STATE:
        _INJECTION_STATE.clear()
        tile = pacs_fit.cached_tile(config, record)
        inj = Injector(tile, inp, t + 1)
        Ws = pool_matrices(cell_centres_arcsec(tile.grid, inj.ids), widenings=POOL_WIDENINGS[:1])
        # injections one batch puts into the pool of an interior cell: density x the pool's cell count
        interior = (inj.cell_px * inj.valid_fraction / (SEPARATION_AREAS * inj.separation_px ** 2)) * float(np.median(np.diff(Ws[0].indptr)))
        _INJECTION_STATE[key] = (inj, Ws, interior)
    return _INJECTION_STATE[key]


def _inject_pairs(piece, config, records):
    """The injections of one piece: `piece = (tile index, inputs, [(S/N plane, half), ...], real check?)` (see
    `Injector`, `injection_pieces`): for each pair, the batches over the map's cells (`_inject_half`) and around
    its detected sources (`_inject_blend`); on the piece flagged for it, the real-map self-check. Returns
    `(tile index, dict)`."""
    t, inp, pairs, do_check = piece
    record = records[t]
    inj, Ws, interior = _injection_state(config, record, inp, t)
    n_row = inj.ids.size
    snr = snr_grid()
    check = None
    if do_check:
        # the self-check on the real map: F = 10 x the map's median 5-sigma flux; the mean ratio within 1 % in both halves
        f_check = 10.0 * pacs_fit.DETECTION_SNR * float(np.median(inj.sigma[inj.ids]))
        check = []
        for half in (0, 1):
            r = _real_check(inj, f_check, half)
            check.append(float(r.mean()) if r.size else np.nan)
            print("  self-check (real map) %s %s half %d: %d empty positions, F = %.0f mJy, injected-minus-before / F: mean %.4f, "
                  "16/50/84 %%: %.4f %.4f %.4f" % (record.band, record.map_id, half, r.size, f_check, r.mean(),
                                                    *np.percentile(r, [16, 50, 84])), flush=True)
            if r.size == 0 or abs(r.mean() - 1) > REAL_CHECK_TOLERANCE:
                raise ValueError("%s: on the real map %s the mean recovered flux is %.4f of the injected flux (bar: within 1 %%)"
                                 % (record.band, record.map_id, r.mean() if r.size else np.nan))
    done = []
    for p, half in pairs:
        r, sec, cnt = _inject_half(inj, Ws, p, snr[p], half, n_row, interior)
        b, bsec = _inject_blend(inj, p, snr[p], half)
        done.append((p, half, r, sec, cnt, b, bsec))
    return t, dict(pairs=done, check=check, n_detected=inj.n_detected)


def _finish_tile(piece, records, flux_grid):
    """A map's product from its pieces' injections: `piece = (tile index, inputs, [the pieces' outputs])`, taken in
    (S/N plane, half) order as one process makes them, pooled (`pool_product`, `cell_errors`) over the map's rows.
    Returns `(tile index, dict)`: the pooled arrays of the tile's rows for the product (the build half), the
    held-out half's injections, and the statistics the parent prints."""
    t, inp, parts = piece
    ids = np.asarray(inp["ids"], np.int64)
    n_row = ids.size
    sigma_rows = np.asarray(inp["sigma"], np.float64)
    Ws = pool_matrices(cell_centres_arcsec(_tile_geometry(records[t])[0], ids))
    snr = snr_grid()
    keys = ("row", "plane", "path", "recovered", "separable", "dflux", "flux", "sigma_formal")
    bkeys = ("row", "plane", "recovered", "separable", "dflux", "flux", "sigma_formal")
    stores = ({k: [] for k in keys}, {k: [] for k in keys})
    bstores = ({k: [] for k in bkeys}, {k: [] for k in bkeys})
    seconds = dict(inject=0.0, aperture=0.0, group=0.0)
    n_by_path = dict(aperture=0, group=0)
    blend_seconds = 0.0
    blend_plane = np.zeros((2, snr.size, 2))
    by_pair = {(p, h): (r, sec, cnt, b, bsec) for part in parts for p, h, r, sec, cnt, b, bsec in part["pairs"]}
    assert len(by_pair) == 2 * snr.size
    for p in range(snr.size):
        for half in (0, 1):
            r, sec, cnt, b, bsec = by_pair[(p, half)]
            for k in ("row", "path", "recovered", "separable", "dflux", "flux", "sigma_formal"):
                stores[half][k].append(r[k])
            stores[half]["plane"].append(np.full(r["row"].size, p, np.int8))
            for k in seconds:
                seconds[k] += sec[k]
            for k in n_by_path:
                n_by_path[k] += cnt[k]
            for k in ("row", "recovered", "separable", "dflux", "flux", "sigma_formal"):
                bstores[half][k].append(b[k])
            bstores[half]["plane"].append(np.full(b["row"].size, p, np.int8))
            blend_seconds += bsec
            blend_plane[half, p] += (b["row"].size, b["recovered"].sum())
    build, held = ({k: np.concatenate(v) for k, v in s.items()} for s in stores)
    blend_build, blend_held = ({k: np.concatenate(v) for k, v in s.items()} for s in bstores)
    out = pool_product(Ws, build, flux_grid, sigma_rows, n_row)
    tot = out.pop("pooled_total")
    both = {k: np.concatenate([build[k], held[k]]) for k in build}
    blend_both = {k: np.concatenate([blend_build[k], blend_held[k]]) for k in blend_build}
    out["errors"] = cell_errors(Ws, flux_grid, sigma_rows, both, blend_both)          # the product: both halves
    held["build_errors"] = cell_errors(Ws, flux_grid, sigma_rows, build, blend_build)  # for the held-out check only
    held["blend"] = blend_held
    out["held"] = held
    # the faintest S/N plane of the build half: the rule firing on a source of almost no flux (CONTAMINATION)
    floor = build["plane"] == 0
    out["floor"] = dict(row=build["row"][floor], recovered=build["recovered"][floor])
    check = next(part["check"] for part in parts if part["check"] is not None)
    out["stats"] = dict(map_id=records[t].map_id, n_row=n_row, n_detected=parts[0]["n_detected"],
                        n_build=int(build["row"].size), n_held=int(held["row"].size),
                        n_source=int(inp["src_ra"].size),
                        min_pool=float(tot.min()), median_pool=float(np.median(tot)),
                        short=float((tot < MIN_INJECTIONS).mean()), seconds=seconds, n_by_path=n_by_path,
                        blend_seconds=blend_seconds, blend_plane=blend_plane,
                        blend_built=int(blend_build["row"].size), blend_held=int(blend_held["row"].size),
                        check=check)
    return t, out


def injection_pieces(records, inputs):
    """The injection pieces of a band, map by map from the largest (`(tile index, inputs, pairs, real check?)`, the
    check on each map's first piece), and each map's piece count."""
    pairs = [(p, half) for p in range(snr_grid().size) for half in (0, 1)]
    todo, counts = [], {}
    for t, inp in sorted(inputs.items(), key=lambda kv: -kv[1]["ids"].size):
        k = int(min(len(pairs), max(1, np.ceil(np.asarray(inp["ids"]).size / INJECTION_PIECE_ROWS))))
        chunks = [c for c in np.array_split(np.arange(len(pairs)), k) if c.size]
        counts[t] = len(chunks)
        todo += [(t, inp, [pairs[i] for i in c], j == 0) for j, c in enumerate(chunks)]
    return todo, counts


def _inject_tile(piece, config, records, flux_grid, cells=None):
    """Everything the injection measures on one map tile, in this process: `piece = (tile index, inputs)`.
    Returns `(tile index, dict)` as `_finish_tile`."""
    t, inp = piece
    todo, _ = injection_pieces(records, {t: inp})
    parts = [_inject_pairs(p, config, records)[1] for p in todo]
    return _finish_tile((t, inp, parts), records, flux_grid)


# ----------------------------------------------------------------------------
# one band of one region

def _tile_geometry(record):
    wcs = WCS(fits.Header.fromstring(record.wcs_header)).celestial
    return pacs_fit.MapGrid.from_wcs(wcs, record.shape), wcs


#: the cells kept per map: every cell inside the region's source bounding box padded by one nside-512
#: pixel (planner 2026-10-27: catalog.completeness maps the admitted pixels' sample points to cells itself)
REGION_PAD_ARCSEC = 412.0


def tile_inputs(cell_map, cell_id, sigma, offset, offset_psf, source_row, shift, ra, dec):
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
                           offset_psf=offset_psf[rows].astype(np.float64),
                           src_ra=ra[s], src_dec=dec[s], src_sigma=sigma[source_row[s]].astype(np.float64),
                           src_offset=offset[source_row[s]].astype(np.float64),
                           shift=(float(np.float32(shift[t][0])), float(np.float32(shift[t][1]))), row0=int(rows[0]))
    return out


def _print_tile_stats(s, flux_grid):
    sec, n = s["seconds"], s["n_by_path"]
    per = lambda k, c: 1e4 * sec[k] / max(c, 1)
    print("  %s: %d cells, %d detected sources; injections build %d held-out %d; pooled per (cell, plane): "
          "min %d median %d (%.1f %% of (cell, plane) below %d after widening); seconds per 10000 injections: aperture "
          "%.1f group %.1f (%d aperture, %d group injections) adding and restoring %.1f; real-map self-check mean ratio "
          "%.4f / %.4f"
          % (s["map_id"], s["n_row"], s["n_detected"], s["n_build"], s["n_held"], s["min_pool"],
             s["median_pool"], 100 * s["short"], MIN_INJECTIONS, per("aperture", n["aperture"]), per("group", n["group"]),
             n["aperture"], n["group"], per("inject", n["aperture"] + n["group"]), s["check"][0], s["check"][1]), flush=True)
    bp = s["blend_plane"].sum(axis=0)
    if bp[:, 0].sum():
        snr = snr_grid()
        every = range(0, snr.size, 4)
        print("      blend path: %d targeted injections build, %d held-out, %.1f s in group fits (%.1f s per 10000); "
              "recovered fraction by S/N: %s"
              % (s["blend_built"], s["blend_held"], s["blend_seconds"],
                 1e4 * s["blend_seconds"] / max(s["blend_built"] + s["blend_held"], 1),
                 " ".join("%.3g:%.2f" % (snr[p], bp[p, 1] / max(bp[p, 0], 1)) for p in every)), flush=True)


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
    names = ("covered", "n_sigma", "sigma", "offset", "offset_psf", "sky", "ra", "dec")
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
    cell_offset_psf = cols["offset_psf"].astype(np.float32)

    # artificial sources: each map cut into pieces of its (S/N plane, half) pairs, map by map from the biggest;
    # then each map pooled whole
    inputs = tile_inputs(cell_map, cell_id, cell_sigma, cell_offset, cell_offset_psf, src_row, shifts, ra, dec)
    todo, counts = injection_pieces(records, inputs)
    done = pieces_module.run(functools.partial(_inject_pairs, config=config, records=records),
                             todo, n_jobs, stage=stage, unit="map pieces (artificial sources)")
    by_tile = {}
    for t, part in done:
        by_tile.setdefault(t, []).append(part)
    assert all(len(by_tile[t]) == counts[t] for t in counts)
    injected = pieces_module.run(functools.partial(_finish_tile, records=records, flux_grid=flux_grid),
                                 [(t, inputs[t], by_tile[t]) for t in counts], n_jobs, stage=stage,
                                 unit="maps (pooling)")
    curve = np.zeros((n_row, n_plane), np.uint16)
    n_pos = np.zeros(n_row, np.int32)
    flux90_ = np.full(n_row, np.nan, np.float32)
    n_bin = len(SNR_BIN_CENTRES)
    cell_keys = ("scatter", "bias")
    map_keys = ("scatter_blend", "n_scatter_blend")
    errors = {k: np.full((n_row, n_plane), np.nan, np.float32) for k in cell_keys}
    build_errors = {k: np.full((n_row, n_plane), np.nan, np.float32) for k in cell_keys}
    errors.update({k: np.full((len(records), n_bin), np.nan, np.float32) for k in map_keys})
    build_errors.update({k: np.full((len(records), n_bin), np.nan, np.float32) for k in map_keys})
    held = {k: [] for k in ("row", "plane", "path", "recovered", "separable", "dflux", "flux")}
    held_blend = {k: [] for k in ("row", "plane", "recovered", "separable", "dflux", "flux")}
    floor_row, floor_rec = [], []
    stats = []
    for t, out in injected:
        r0 = inputs[t]["row0"]
        sl = slice(r0, r0 + inputs[t]["ids"].size)
        curve[sl], n_pos[sl], flux90_[sl] = out["curve"], out["n_pos"], out["flux90"]
        for k in cell_keys:
            errors[k][sl] = out["errors"][k]
            build_errors[k][sl] = out["held"]["build_errors"][k]
        for k in map_keys:
            errors[k][t] = out["errors"][k]
            build_errors[k][t] = out["held"]["build_errors"][k]
        for k in held:
            held[k].append(out["held"][k] + r0 if k == "row" else out["held"][k])
        for k in held_blend:
            held_blend[k].append(out["held"]["blend"][k] + r0 if k == "row" else out["held"]["blend"][k])
        floor_row.append(out["floor"]["row"] + r0)
        floor_rec.append(out["floor"]["recovered"])
        stats.append(out["stats"])
    held = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in held.items()}
    held["blend"] = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in held_blend.items()}
    held["stats"] = stats
    held["build_errors"] = build_errors
    contamination, bin_values, bin_counts = _contamination(cell_map, cols["sky"],
                                                           np.concatenate(floor_row) if floor_row else np.zeros(0, np.int64),
                                                           np.concatenate(floor_rec) if floor_rec else np.zeros(0, bool))
    print("  CONTAMINATION by within-map sky bin (faint to bright): %s"
          % "  ".join("%.4f (%d)" % (v, c) for v, c in zip(bin_values, bin_counts)), flush=True)
    for s_ in stats:
        _print_tile_stats(s_, flux_grid)
    envelope = curve.max(axis=0).astype(np.uint16) if n_row else np.zeros(n_plane, np.uint16)
    return dict(curve=curve, source_row=src_row.astype(np.int32), envelope=envelope, n_pos=n_pos,
                n_scatter_blend=errors["n_scatter_blend"].astype(np.int32),
                cell_offset_psf=cell_offset_psf, contamination=contamination, contamination_bins=bin_values, contamination_counts=bin_counts,
                n_sigma=cols["n_sigma"].astype(np.int16), flux90=flux90_, bias=errors["bias"], scatter=errors["scatter"],
                scatter_blend=errors["scatter_blend"], held=held, stats=stats,
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


def _contamination(cell_map, cell_sky, row, recovered):
    """CONTAMINATION per row: the cells of each map cut into SKY_BINS equal bins by rank of their annulus sky (PACS
    maps have no absolute zero level); per bin, over every map of the band, the fraction of the faintest-S/N
    artificial sources (`row`, `recovered`) the rule counted as detections. Returns (per row, per bin, injections
    per bin)."""
    n_row = cell_map.size
    bins = np.zeros(n_row, np.int64)
    for m in np.unique(cell_map):
        rows = np.flatnonzero(cell_map == m)
        rank = np.argsort(np.argsort(np.nan_to_num(cell_sky[rows], nan=np.nanmedian(cell_sky[rows])), kind="stable"))
        bins[rows] = np.minimum(rank * SKY_BINS // max(rows.size, 1), SKY_BINS - 1)
    n = np.bincount(bins[row], minlength=SKY_BINS).astype(np.float64)
    k = np.bincount(bins[row][recovered], minlength=SKY_BINS).astype(np.float64)
    with np.errstate(all="ignore"):
        value = np.where(n > 0, k / n, np.nan)
    return value[bins].astype(np.float32), value, n.astype(np.int64)


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
                contamination=z(np.float32), contamination_bins=np.full(SKY_BINS, np.nan),
                contamination_counts=np.zeros(SKY_BINS, np.int64),
                envelope=np.zeros(n_plane, np.uint16), n_pos=z(np.int32), n_sigma=z(np.int16), flux90=z(np.float32),
                bias=nan(), scatter=nan(),
                scatter_blend=np.zeros((0, 4), np.float32),
                n_scatter_blend=np.zeros((0, 4), np.int32), cell_offset_psf=z(np.float32), held=None, stats=[], cell_ra=z(float),
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
            ds.attrs["POOL_ARCSEC"] = POOL_ARCSEC
            ds.attrs["POOL_WIDENINGS"] = np.asarray(POOL_WIDENINGS)
            ds.attrs["SNR_GRID"] = snr_grid()
            w("MAG_GRID_" + b, mag)
            w("ENVELOPE_" + b, d["envelope"])
            w("COMPLETED_FROM_" + b, np.full(d["curve"].shape[0], np.nan, np.float32))
            w("N_POS_" + b, d["n_pos"])
            w("N_SIGMA_POS_" + b, d["n_sigma"])
            w("FLUX90_%s_MJY" % b, d["flux90"])
            w("BIAS_%s_MJY" % b, d["bias"])
            w("SCATTER_%s_MJY" % b, d["scatter"])
            ds = w("SCATTER_BLEND_" + b, d["scatter_blend"])
            ds.attrs["SNR_BIN_EDGES"] = np.asarray(SNR_BIN_EDGES)
            ds.attrs["SNR_BIN_CENTRES"] = np.asarray(SNR_BIN_CENTRES)
            w("N_SCATTER_BLEND_" + b, d["n_scatter_blend"])
            w("CELL_OFFSET_PSF_%s_MJY" % b, d["cell_offset_psf"])
            ds = w("CONTAMINATION_" + b, d["contamination"])
            ds.attrs["SKY_BINS"] = SKY_BINS
            ds.attrs["BIN_VALUES"] = np.asarray(d["contamination_bins"], np.float64)
            ds.attrs["BIN_INJECTIONS"] = np.asarray(d["contamination_counts"], np.int64)
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
        if records:
            self_check(config, (scan or records)[0])
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
    """One band of the product, as a dict of arrays, with the pooling matrices rebuilt from its cells."""
    g = lambda name: f[name][:]
    d = dict(curve=g("CURVE_" + band), mag=g("MAG_GRID_" + band), flux90=g("FLUX90_%s_MJY" % band),
             n_pos=g("N_POS_" + band), scatter=g("SCATTER_%s_MJY" % band), scatter_blend=g("SCATTER_BLEND_" + band),
             bias=g("BIAS_%s_MJY" % band), sigma=g("CELL_SIGMA_%s_MJY" % band), offset=g("CELL_OFFSET_%s_MJY" % band),
             contamination=g("CONTAMINATION_" + band), offset_psf=g("CELL_OFFSET_PSF_%s_MJY" % band), cell_id=g("CELL_ID_" + band), cell_map=g("CELL_MAP_INDEX_" + band),
             map_id=g("MAP_ID_" + band), map_wcs=g("MAP_WCS_" + band), map_shape=g("MAP_SHAPE_" + band),
             map_shift=g("MAP_SHIFT_%s_ARCSEC" % band), source_row=g("SOURCE_ROW")[:, BANDS.index(band)])
    d["contamination_bins"] = f["CONTAMINATION_" + band].attrs["BIN_VALUES"]
    d["contamination_counts"] = f["CONTAMINATION_" + band].attrs["BIN_INJECTIONS"]
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
    SCATTER_build read at the injection's own cell and flux (expected 1.02 +/- 0.05, bar 1.00 +/- 0.10 where >= 200 residuals). Joint fit, per map and
    bin: held-out dflux / (sigma_formal x SCATTER_BLEND_build[map, bin]), the same expectation and bar. Bias above
    FLUX90 inside the flux grid, both paths: |median dflux / injected| < 1 %."""
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
    ap = (held["path"] == 1) & held["separable"].astype(bool) & in_grid(flux)
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
    b = held["blend"]
    if b["row"].size:
        br = b["row"].astype(np.int64)
        bk = bins_of_plane[b["plane"].astype(np.int64)]
        bf = b["flux"].astype(np.float64)
        sf = b["sigma_formal"].astype(np.float64)
        bmap = d["cell_map"][br]
        sb = e["scatter_blend"][bmap, bk]
        good = b["separable"].astype(bool) & np.isfinite(sf) & (sf > 0) & np.isfinite(sb) & (sb > 0)
        for m in np.unique(bmap):
            for j in range(len(SNR_BIN_CENTRES)):
                sel = good & (bmap == m) & (bk == j)
                z = b["dflux"][sel] / (sf[sel] * sb[sel])
                verdict = _verdict(_robust(z), 0.90, 1.10) if sel.sum() >= 200 else "fewer than 200"
                print("  C2 %s joint fit map %s, S/N bin %d: %d held-out residuals, normalised scatter %.3f (expected 1.02 "
                      "+/- 0.05, bar 1.00 +/- 0.10: %s)" % (band, d["map_id"][m].decode(), j, int(sel.sum()), _robust(z), verdict))
        bf90 = d["flux90"][br]
        above = b["separable"].astype(bool) & np.isfinite(bf90) & (bf > bf90) & in_grid(bf)
        if above.sum():
            bias = 100 * np.median(b["dflux"][above] / bf[above])
            print("  C2 %s joint-fit bias above FLUX90: %d, median residual / injected %+.2f %% (bar |.| < 1 %%: %s)"
                  % (band, int(above.sum()), bias, _verdict(-abs(bias), -1)))


def _report_c3(band, d, held):
    """C3 (a report, no bar): CONTAMINATION by sky bin, and the number of the region's detections expected to be
    unrelated flux: the sum of CONTAMINATION over the covered SESNA sources against the detected count."""
    print("  C3 %s: CONTAMINATION by within-map sky bin, faint to bright: %s"
          % (band, "  ".join("%.4f (%d injections)" % (v, c) for v, c in zip(d["contamination_bins"], d["contamination_counts"]))))
    covered = d["source_row"] >= 0
    expected = float(np.nansum(d["contamination"][d["source_row"][covered]]))
    detected = sum(s["n_detected"] for s in held.get("stats", []))
    print("  C3 %s: %d covered SESNA sources, %d detected; expected detections from unrelated flux %.1f = %.2f %% of the "
          "detected (report)" % (band, int(covered.sum()), detected, expected, 100 * expected / max(detected, 1)))


def acceptance(config, region, held_out):
    """Prints C1-C3 for the region's product and the held-out half kept from the build that wrote it (`build`'s
    return value for the region). C1: the stored curve and 0.9 flux against the held-out half. C2: the held-out
    recovered injections' bias and scatter against the stored SCATTER. C3: the CONTAMINATION report. Nothing is
    written."""
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
            _report_c3(band, d, held)


if __name__ == "__main__":
    run(build)
