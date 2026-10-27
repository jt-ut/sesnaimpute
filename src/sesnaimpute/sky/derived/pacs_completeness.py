"""The Herschel PACS completeness of the SESNA sources, per region and band
(SPEC_PACS_FARIR_DRAFT.md sections 5 and 11.3 step 2).

`build(config, regions=None)` writes one product per region,
`sky/derived/pacs/completeness_pacs_source__<Region>.hdf5`, in EXACTLY the layout of
`catalog/sesna/completeness_sesna_source__<Region>.hdf5` for the bands P70 and P100
(`CURVE_<B>`, `MAG_GRID_<B>`, `SOURCE_ROW`, `PIXEL_CURVE_<B>`, `PIXEL_COVERED_FRAC_<B>`,
`ENVELOPE_<B>`, `COMPLETED_FROM_<B>`, `N_POS_<B>`, `HPX_PIX_512`), plus per-cell columns aligned
with the CURVE rows (`CELL_RA_<B>`, `CELL_DEC_<B>`, `CELL_OBSID_<B>`, `CELL_MAP_<B>`,
`CELL_SIGMA_<B>_MJY`, `CELL_GAIN_<B>`) that the photometry stage reads, and the same cell statistics of
the single PSF fit, the estimator the photometry uses for the members of blend groups
(`CELL_OFFSET_PSF_<B>_MJY`, `CELL_SIGMA_PSF_<B>_MJY`, `CELL_TAIL_FACTOR_PSF_<B>`, `CELL_GAIN_PSF_<B>`,
`CURVE_PSF_<B>`, rows aligned with the CURVE rows).

Method. The PACS measurement (`pacs_fit`) is linear in the map, so a point source of flux F at a
position p is measured as g F + b_p, where g is the fit's gain and b_p its value on the real, un-injected
map there (background, confusion and noise together). Per 30-arcsec cell on each map tile, 200
positions are drawn uniformly (no exclusion of real sources: confusion is part of the completeness),
each gets one forced fit with the bright neighbours of the tile as nuisance columns, and:
  cell offset = the median of b_p over the cell's clean positions: sky structure the fit's constant
  background does not absorb, subtracted from every flux measured in the cell (sources and positions);
  cell sigma = the median over the cell's 3 x 3 block of 1.4826 x the median absolute deviation of b_p
  about each cell's offset (or the cell's own such sigma when it exceeds the block median by more than
  OWN_SIGMA_FACTOR, so a lone structured cell is not given its quiet neighbours' noise), times the tail factor of the cell's noise-state group (cells of the map at a
  similar sigma / map median), matched as deep in the negative tail as the group's sample allows (one cell holds only ~10 independent beams), times one factor per
  map tile that matches the negative tail of b_p / sigma over the tile's empty sky to a
  Gaussian's (the noise IS the scatter of the extracted flux on empty sky, not added to the propagated
  pixel noise; the factor corrects the MAD for the noise's heavier-than-Gaussian tails);
  completeness at flux F = the fraction of positions with g F + b_p - offset >= 5 x cell sigma, i.e.
  the empirical distribution of (5 sigma + offset - b_p)/g evaluated at every plane of the grid.
The sigma of a cell is what the photometry stage calls the uncertainty of a source in that cell, so a
measured source and its completeness curve use one noise.

Two estimators, each with its own offset, sigma, tail factors and curve, from the SAME random positions:
the aperture (`Tile.measure`, isolated sources) and the single PSF fit (`Tile.fit_positions`, the members
of blend groups, whose joint fit carries its own background). A PSF fit absorbs part of the structure that
the aperture leaves in its offset, so a blend member is judged on the PSF estimator's offset and noise
(the `_PSF_` datasets), an isolated source on the aperture's.

Where maps overlap, each source and each of the 256 sample points per admitted pixel takes the cell of
the deepest map (lowest cell sigma); only cells that win somewhere get a CURVE row. The flux grid is one
per band for the whole survey: m = -2.5 log10(F / 1 Jy) from MAG_MIN to MAG_MAX in MAG_STEP, bright to
faint. A region a band does not cover gets the same product with no rows, SOURCE_ROW -1 and zero pixel
curves.

Run: `PY sesnaimpute.sky.derived.pacs_completeness $CONFIG [--regions ...]`.
"""

import functools
import os

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from sesnaimpute import build as build_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived import pacs_fit

BANDS = ("P70", "P100", "P160")
#: positions per cell (spec section 5)
N_POS = 200
#: a cell needs this many valid positions to get a curve; fewer means the cell is not covered
MIN_VALID_POSITIONS = 100
#: the stored curve is the fraction times this (the SESNA convention, lossless for 200 positions)
STORAGE_SCALE = 600
#: PACS has no Vega system: zero point 1 Jy, m = -2.5 log10(F / 1 Jy)
ZERO_POINT_JY = 1.0
#: ONE grid per band for the whole survey, bright to faint. 25 Jy brightward is above every
#: cell's roll-off (the sigma of the noisiest cell, beside Musca's 10 Jy source, is under 1 Jy, so
#: its thresholds sit below 8 Jy); 6 mJy faintward is below the 5-sigma flux of the deepest cell
#: (sigma ~ 12 mJy, curve zero a few sigma under 60 mJy). Range checked on every build (printed).
MAG_MIN = -3.5
MAG_MAX = 5.5
MAG_STEP = 0.25
#: a piece is one tile's square block of BLOCK_CELLS x BLOCK_CELLS cells (plus a one-cell border whose
#: fits feed the 3 x 3 noise median of the block's edge cells; about 30 % extra fits)
BLOCK_CELLS = 16
MAD_TO_SIGMA = 1.4826
#: the noise's tail correction is measured on this many evenly spread cells (with their 3 x 3 blocks) of each tile
TAIL_CELLS = 400
#: the noise-state groups of a map's cells (planner 2026-10-27): cell sigma / the map's median cell sigma
#: binned at these edges; one tail factor per group. The heavy tails live in the noisy cells, scattered
#: singly over a field (Cygnus X: 115 cells, 6 adjacent), so neighbours on the sky are the wrong group
NOISE_BIN_EDGES = (1.0, 1.5, 2.0, 3.0, 5.0)
#: a group's factor is matched at the deepest quantile its sample supports: this many clean positions
#: beyond it, but never deeper than the 5-sigma Gaussian probability
TAIL_MIN_BEYOND = 50
TAIL_DEEPEST_P = 2.87e-7
#: a group with fewer clean positions than this cannot reach even 3 sigma and joins its quieter neighbour
GROUP_MIN_POSITIONS = 37000


def canonical_grid():
    """(magnitude planes, flux planes in mJy), bright to faint."""
    n = int(round((MAG_MAX - MAG_MIN) / MAG_STEP)) + 1
    mag = MAG_MIN + MAG_STEP * np.arange(n)
    return mag, 1000.0 * ZERO_POINT_JY * 10.0 ** (-mag / 2.5)


# ----------------------------------------------------------------------------
# the cell statistic

def cell_fits(tile, ids, seed, estimator="measure"):
    """Forced measurements at N_POS uniform positions in each cell of `ids` on `tile`: the aperture
    (`estimator` "measure") or the single PSF fit ("fit_positions"), at the same positions (the seed
    fixes them). Returns b (flux mJy), gain, pixel sigma (propagated), valid, clean (farther than one beam
    from every bright neighbour), each (n_cell, N_POS)."""
    x, y = tile.grid.sample_pixels(ids, N_POS, seed)
    fit = getattr(tile, estimator)(x.ravel(), y.ravel())
    shape = x.shape
    valid = (fit.covered & np.isfinite(fit.flux) & np.isfinite(fit.gain) & (fit.gain > 0)).reshape(shape)
    clean = ~tile.near_bright(x, y)
    return (fit.flux.reshape(shape), fit.gain.reshape(shape), fit.sigma.reshape(shape), valid, clean)


#: a cell takes its own clean MAD sigma instead of its block's median when it exceeds the block median by
#: more than this factor: 1 + 3 x 0.27, three of the one-cell MAD's ~27 % sampling errors (see block_sigma)
OWN_SIGMA_FACTOR = 1.81
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


def thresholds(b, gain, valid, sigma, offset):
    """(5 sigma + offset - b)/g at every position, NaN where invalid: a source of that flux there is
    measured g F + b - offset, the cell's offset subtracted as for a real source. The curve uses ALL
    valid positions."""
    with np.errstate(all="ignore"):
        return ((pacs_fit.DETECTION_SNR * sigma[:, None] + offset[:, None] - np.where(valid, b, np.nan))
                / np.where(valid, gain, np.nan))


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


def cell_sigma(grid, ids, all_ids, mad_sigma, n_clean):
    """Noise of each cell of `ids`: its block median (`block_sigma`), or its own clean MAD sigma when that
    exceeds the block median by more than OWN_SIGMA_FACTOR (a lone structured cell: its neighbours' median
    would understate it, a two-sided excess and false detections). A cell with fewer than
    MIN_CLEAN_POSITIONS clean positions keeps the block value. Returns (sigma, positions behind the block,
    number of cells that took their own)."""
    block, n_block = block_sigma(grid, ids, all_ids, mad_sigma, n_clean)
    k = np.searchsorted(all_ids, ids)
    own = np.where((n_clean[k] >= MIN_CLEAN_POSITIONS) & np.isfinite(mad_sigma[k]) & (mad_sigma[k] > 0),
                   mad_sigma[k], np.nan)
    with np.errstate(invalid="ignore"):
        switch = np.isfinite(block) & np.isfinite(own) & (own > OWN_SIGMA_FACTOR * block)
    return np.where(switch, own, block), n_block, int(switch.sum())


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
    (`pacs_fit.registration`) and its median cell sigma over up to TAIL_CELLS cells spread evenly over the
    cells the region USES there (wholly on valid data), the reference for the noise-state groups and the
    fallback sigma. `piece = (tile index, used cells, SESNA ra, dec on the tile)`."""
    t, used, sra, sdec = piece
    tile = pacs_fit.cached_tile(config, records[t])
    shift = pacs_fit.registration(tile, sra, sdec)
    centres = used[np.unique(np.linspace(0, used.size - 1, min(TAIL_CELLS, used.size)).astype(np.int64))]
    all_ids = ring(tile.grid, centres)
    b, gain, ps, valid, clean = cell_fits(tile, all_ids, 0)
    _, n_clean, mad, _, _, _ = cell_statistics(b, gain, ps, valid, clean)
    sigma, _, _ = cell_sigma(tile.grid, centres, all_ids, mad, n_clean)
    c = np.searchsorted(all_ids, centres)
    good = np.isfinite(sigma) & (sigma > 0) & (valid[c].sum(axis=1) == N_POS)
    median = float(np.median(sigma[good])) if good.any() else np.nan
    # the PSF estimator's own reference, from the same cells
    b, gain, ps, valid, clean = cell_fits(tile, all_ids, 0, "fit_positions")
    _, n_clean, mad, _, _, _ = cell_statistics(b, gain, ps, valid, clean)
    sigma, _, _ = cell_sigma(tile.grid, centres, all_ids, mad, n_clean)
    good = np.isfinite(sigma) & (sigma > 0) & (valid[c].sum(axis=1) == N_POS)
    return dict(n_used=int(used.size), shift=shift, median=median,
                median_psf=float(np.median(sigma[good])) if good.any() else np.nan)


def noise_group(sigma, median):
    """The noise-state group of each cell, 0 (quietest) .. len(NOISE_BIN_EDGES)."""
    return np.digitize(sigma / median, NOISE_BIN_EDGES)


def tail_factors(stats, refs, records):
    """{tile: k per noise group} from the first pass's per-group statistics. Groups of a map with fewer
    than TAIL_CELLS used cells pool with the same group of every map of the band in the region with the
    same observing mode and scan speed; a group short of GROUP_MIN_POSITIONS joins its quieter neighbour.
    k = q(p) / z(p): q the p-quantile of (b - offset) / sigma over the group's clean positions (wholly
    valid cells), z the Gaussian point at p = max(TAIL_DEEPEST_P, TAIL_MIN_BEYOND / N) -- the deepest
    point the sample supports, since the tails are Gaussian to 3 sigma and heavy beyond. Returns also
    the table rows (tile, group, N, matched sigma, k) for printing."""
    from scipy.stats import norm
    n_group = len(NOISE_BIN_EDGES) + 1
    def merged(tiles, g):
        n = sum(stats[t]["n"][g] for t in tiles)
        low = np.sort(np.concatenate([stats[t]["low"][g] for t in tiles])) if tiles else np.zeros(0)
        return n, low[:TAIL_MIN_BEYOND * 4]
    out, rows = {}, []
    for t in stats:
        kind = (records[t].mode, records[t].scan_speed)
        tiles = [t] if refs[t]["n_used"] >= TAIL_CELLS else [u for u in stats if (records[u].mode, records[u].scan_speed) == kind]
        per = [merged(tiles, g) for g in range(n_group)]
        # from the noisiest group down, a short group joins its quieter neighbour; a short quietest
        # remainder joins the group above it
        groups, cur = [], []
        for g in reversed(range(n_group)):
            cur.insert(0, g)
            if sum(per[m][0] for m in cur) >= GROUP_MIN_POSITIONS:
                groups.insert(0, cur)
                cur = []
        if cur:
            groups = [cur + groups[0]] + groups[1:] if groups else [cur]
        k = np.ones(n_group)
        for members in groups:
            n = sum(per[m][0] for m in members)
            low = np.sort(np.concatenate([per[m][1] for m in members]))
            p = max(TAIL_DEEPEST_P, TAIL_MIN_BEYOND / max(n, 1))
            zp = norm.ppf(p)
            r = min(max(int(round(p * n)) - 1, 0), low.size - 1) if low.size else 0
            k[members] = float(low[r] / zp) if low.size and low[r] < 0 else 1.0
            rows.append((t, members, n, -zp, float(k[members[0]])))
        out[t] = k
    return out, rows


def curve_counts(threshold, flux_grid):
    """(n_cell, n_plane) number of positions with threshold <= plane flux (valid ones only; NaN
    thresholds never count). Monotone in the plane by construction."""
    with np.errstate(invalid="ignore"):
        return (threshold[:, :, None] <= flux_grid[None, None, :]).sum(axis=1)


def encode_curves(counts, n_valid):
    """Fraction x STORAGE_SCALE, rounded, uint16."""
    return np.rint(counts * STORAGE_SCALE / np.maximum(n_valid, 1)[:, None]).astype(np.uint16)


def _estimator(tile, ids, all_ids, seed, estimator, median):
    """One estimator's cell statistics for the block `ids` (`all_ids` its ring): the arrays of its fits and
    its sigma (rule B, then the tile median where no cell is clean), offset and group."""
    b, gain, ps, valid, clean = cell_fits(tile, all_ids, seed, estimator)
    n_valid, n_clean, mad, g, pps, off = cell_statistics(b, gain, ps, valid, clean)
    sigma, n_sigma, n_switch = cell_sigma(tile.grid, ids, all_ids, mad, n_clean)
    offset = cell_offset(tile.grid, ids, all_ids, off, n_clean)
    # a block with no clean cell (wholly within a beam of bright sources) takes the tile's median noise;
    # N_SIGMA_POS 0 marks it
    sigma = np.where(np.isfinite(sigma), sigma, median)
    k = np.searchsorted(all_ids, ids)
    return dict(b=b[k], gain=gain[k], valid=valid[k], clean=clean[k], n_valid=n_valid[k], g=g[k], pps=pps[k],
                sigma=sigma, n_sigma=n_sigma, offset=offset, n_switch=n_switch, group=noise_group(sigma, median))


def _first_pass(e):
    """The per-noise-group statistics of (b - offset) / sigma over one estimator's clean positions in wholly
    valid cells: the count and the TAIL_MIN_BEYOND * 4 lowest values."""
    whole = e["valid"].sum(axis=1) == N_POS
    z = (e["b"] - e["offset"][:, None]) / e["sigma"][:, None]
    n, low = [], []
    for gr in range(len(NOISE_BIN_EDGES) + 1):
        zz = z[(e["group"] == gr) & whole][(e["valid"] & e["clean"])[(e["group"] == gr) & whole]]
        n.append(int(zz.size))
        low.append(np.sort(zz)[:TAIL_MIN_BEYOND * 4])
    return n, low


def _second_pass(e, tail_by_group, flux_grid):
    """One estimator's arrays for the parent: sigma scaled by the cell's group factor, offset, gain, curve."""
    tail = tail_by_group[e["group"]]
    sigma = e["sigma"] * tail
    curve = encode_curves(curve_counts(thresholds(e["b"], e["gain"], e["valid"], sigma, e["offset"]), flux_grid),
                          e["n_valid"])
    return dict(sigma=sigma, offset=e["offset"], gain=e["g"], curve=curve, tail=tail.astype(np.float32))


def _piece(piece, config, records, flux_grid, refs, factors=None, factors_psf=None):
    """One tile's square block of cells, for both estimators (aperture, and the single PSF fit at the same
    random positions). First pass (`factors` None): the per-noise-group statistics of (b - offset) / sigma
    over the block's clean positions in wholly valid cells -- the count and the TAIL_MIN_BEYOND * 4 lowest
    values -- for each estimator. Second pass: the arrays the parent assembles, sigma scaled by the cell's
    group factor. Both passes draw the same random positions (seeded per cell). The block's border cells
    are fitted too, only for the noise median."""
    tile_index, ids = piece
    record = records[tile_index]
    tile = pacs_fit.cached_tile(config, record)
    all_ids = ring(tile.grid, ids)
    ref = refs[tile_index]
    ap = _estimator(tile, ids, all_ids, tile_index + 1, "measure", ref["median"])
    psf = _estimator(tile, ids, all_ids, tile_index + 1, "fit_positions", ref["median_psf"])
    if factors is None:
        n, low = _first_pass(ap)
        n_psf, low_psf = _first_pass(psf)
        return dict(tile=tile_index, n=n, low=low, n_psf=n_psf, low_psf=low_psf,
                    n_switch=ap["n_switch"], n_switch_psf=psf["n_switch"], n_cell=int(ids.size))
    a = _second_pass(ap, factors[tile_index], flux_grid)
    p = _second_pass(psf, factors_psf[tile_index], flux_grid)
    sigma, g = a["sigma"], a["gain"]
    covered = (ap["n_valid"] >= MIN_VALID_POSITIONS) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(g)
    ra, dec = pacs_fit.from_map_frame(*tile.grid.centres_icrs(ids), ref["shift"][:2])
    return dict(tile=tile_index, ids=ids, covered=covered, n_valid=ap["n_valid"].astype(np.int16),
                n_sigma=np.minimum(ap["n_sigma"], np.iinfo(np.int16).max).astype(np.int16), tail=a["tail"],
                sigma=sigma, offset=a["offset"], gain=g, pixel_sigma=ap["pps"], curve=a["curve"], ra=ra, dec=dec,
                sigma_psf=p["sigma"], offset_psf=p["offset"], gain_psf=p["gain"], curve_psf=p["curve"],
                tail_psf=p["tail"])


# ----------------------------------------------------------------------------
# one band of one region

def _tile_geometry(record):
    wcs = WCS(fits.Header.fromstring(record.wcs_header)).celestial
    return pacs_fit.MapGrid.from_wcs(wcs, record.shape), wcs


#: the cells kept per map: every cell inside the region's source bounding box padded by one nside-512
#: pixel (planner 2026-10-27: catalog.completeness maps the admitted pixels' sample points to cells itself)
REGION_PAD_ARCSEC = 412.0


def _band_product(config, records, ra, dec, n_jobs, stage, flux_grid):
    """Measure every covered cell of every map of one band inside the region's padded source bounding box;
    choose each source's deepest map. Returns the band's datasets, or None when no map reaches a source."""
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
    # pass 1: each map's per-noise-group tail statistics; then one factor per group
    first_pass = pieces_module.run(functools.partial(_piece, config=config, records=records, flux_grid=flux_grid,
                                                     refs=refs), pieces, n_jobs, stage=stage, unit="cell blocks (tails)")
    n_group = len(NOISE_BIN_EDGES) + 1
    edges = ("<1",) + tuple("%g-%g" % e for e in zip(NOISE_BIN_EDGES[:-1], NOISE_BIN_EDGES[1:])) + (">%g" % NOISE_BIN_EDGES[-1],)
    out_factors = []
    for label, kn, kl in (("aperture", "n", "low"), ("PSF fit", "n_psf", "low_psf")):
        stats = {}
        for r in first_pass:
            st = stats.setdefault(r["tile"], dict(n=[0] * n_group, low=[np.zeros(0)] * n_group))
            for gr in range(n_group):
                st["n"][gr] += r[kn][gr]
                st["low"][gr] = np.sort(np.r_[st["low"][gr], r[kl][gr]])[:TAIL_MIN_BEYOND * 4]
        factors_e, rows = tail_factors(stats, refs, records)
        out_factors.append(factors_e)
        for t, members, n, zlev, k in rows:
            print("  %s (%s): noise group %s (sigma/median %s): N %d, matched at %.2f sigma, k %.3f%s"
                  % (records[t].map_id, label, members, ",".join(edges[m] for m in members), n, zlev, k,
                     "  (pooled by mode)" if refs[t]["n_used"] < TAIL_CELLS else ""), flush=True)
    factors, factors_psf = out_factors
    print("  cells on their own noise (rule B): aperture %d, PSF fit %d of %d"
          % (sum(r["n_switch"] for r in first_pass), sum(r["n_switch_psf"] for r in first_pass),
             sum(r["n_cell"] for r in first_pass)), flush=True)
    # pass 2: the curves, sigma scaled by the cell's group factor (the same random positions)
    fn = functools.partial(_piece, config=config, records=records, flux_grid=flux_grid, refs=refs, factors=factors,
                           factors_psf=factors_psf)
    results = pieces_module.run(fn, pieces, n_jobs, stage=stage, unit="cell blocks")
    names = ("covered", "n_valid", "n_sigma", "sigma", "offset", "gain", "pixel_sigma", "curve", "ra", "dec", "tail",
             "sigma_psf", "offset_psf", "gain_psf", "curve_psf", "tail_psf")
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
    envelope = cols["curve"].max(axis=0).astype(np.uint16) if n_row else np.zeros(n_plane, np.uint16)
    shifts = np.array([refs[t]["shift"] if t in refs else (0.0, 0.0, 0) for t in range(len(records))], dtype=float)
    return dict(curve=cols["curve"].astype(np.uint16).reshape(n_row, n_plane), source_row=src_row.astype(np.int32),
                envelope=envelope, n_pos=cols["n_valid"].astype(np.int16), n_sigma=cols["n_sigma"].astype(np.int16),
                cell_ra=cols["ra"], cell_dec=cols["dec"], cell_sigma=cols["sigma"].astype(np.float32),
                cell_offset=cols["offset"].astype(np.float32), cell_gain=cols["gain"].astype(np.float32),
                cell_tail=cols["tail"].astype(np.float32), cell_id=cell_id,
                curve_psf=cols["curve_psf"].astype(np.uint16).reshape(n_row, n_plane),
                cell_sigma_psf=cols["sigma_psf"].astype(np.float32), cell_offset_psf=cols["offset_psf"].astype(np.float32),
                cell_gain_psf=cols["gain_psf"].astype(np.float32), cell_tail_psf=cols["tail_psf"].astype(np.float32), cell_map_index=cell_map,
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
    return dict(curve=np.zeros((0, n_plane), np.uint16), source_row=np.full(n_src, -1, np.int32),
                envelope=np.zeros(n_plane, np.uint16), n_pos=z(np.int16), n_sigma=z(np.int16), cell_ra=z(float),
                cell_dec=z(float), cell_sigma=z(np.float32), cell_offset=z(np.float32), cell_tail=z(np.float32),
                cell_gain=z(np.float32), curve_psf=np.zeros((0, n_plane), np.uint16), cell_sigma_psf=z(np.float32),
                cell_offset_psf=z(np.float32), cell_gain_psf=z(np.float32), cell_tail_psf=z(np.float32),
                cell_id=z(np.int64), cell_map_index=z(np.int32), cell_obsid=z("S1"),
                cell_map=z("S1"), n_candidate=0, map_id=z("S1"), map_obsid=z("S1"), map_shift=np.zeros((0, 2), np.float32),
                map_pairs=z(np.int32), map_wcs=z("S1"), map_shape=np.zeros((0, 2), np.int32),
                map_cell_step=np.zeros((0, 2)))


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
            w("CELL_RA_" + b, d["cell_ra"])
            w("CELL_DEC_" + b, d["cell_dec"])
            w("CELL_ID_" + b, d["cell_id"])
            w("CELL_MAP_INDEX_" + b, d["cell_map_index"])
            w("CELL_OBSID_" + b, d["cell_obsid"])
            w("CELL_MAP_" + b, d["cell_map"])
            w("CELL_SIGMA_%s_MJY" % b, d["cell_sigma"])
            w("CELL_OFFSET_%s_MJY" % b, d["cell_offset"])
            w("CELL_TAIL_FACTOR_" + b, d["cell_tail"])
            w("CELL_GAIN_" + b, d["cell_gain"])
            ds = w("CURVE_PSF_" + b, d["curve_psf"])
            ds.attrs["ZERO_POINT_JY"] = ZERO_POINT_JY
            w("CELL_SIGMA_PSF_%s_MJY" % b, d["cell_sigma_psf"])
            w("CELL_OFFSET_PSF_%s_MJY" % b, d["cell_offset_psf"])
            w("CELL_TAIL_FACTOR_PSF_" + b, d["cell_tail_psf"])
            w("CELL_GAIN_PSF_" + b, d["cell_gain_psf"])
            w("MAP_ID_" + b, d["map_id"])
            w("MAP_OBSID_" + b, d["map_obsid"])
            w("MAP_SHIFT_%s_ARCSEC" % b, d["map_shift"])
            w("MAP_N_PAIRS_" + b, d["map_pairs"])
            w("MAP_WCS_" + b, d["map_wcs"])
            w("MAP_SHAPE_" + b, d["map_shape"])
            w("MAP_CELL_STEP_" + b, d["map_cell_step"])
        w = lambda name, data: build_module.write_dataset(f, name, data, *REGISTRY[(stem, name)])
        w("SOURCE_ROW", np.stack([bands[b]["source_row"] for b in BANDS], axis=1).astype(np.int32))


def _build_region(config, region, out_path, ra, dec, n_jobs, stage=None):
    """Measure and write one region. `ra`/`dec` are the region's source positions in the curated order.
    Returns the per-band products (for the caller's checks)."""
    mag, flux_grid = canonical_grid()
    out = {}
    for b in BANDS:
        records = [r for r in pacs_fit.read_index(config, region) if r.band == b]
        out[b] = _band_product(config, records, ra, dec, n_jobs, stage, flux_grid) if records else None
        if out[b] is None:
            out[b] = _empty_band(ra.size, mag.size)
    write_product(out_path, region, mag, out)
    return out


def _region_inputs(config, region):
    """The region's source positions (NAME order of the curated source table; the redesign's
    catalog.curated.read_positions gives the same rows from the delivery)."""
    with h5py.File(product_path(config, "catalog", "sesna", "sources", "source", region=region), "r") as f:
        return f["RA_DEG"][:], f["DEC_DEG"][:]


def build(config, regions=None):
    """Writes each region's `completeness_pacs_source` product, regions one after another."""
    names = regions or [r.name for r in regions_module.REGIONS]
    t0 = __import__("time").time()
    for region in names:
        with progress_module.Stage("sky.derived.pacs_completeness", region) as st:
            ra, dec = _region_inputs(config, region)
            path = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
            bands = _build_region(config, region, path, ra, dec, config.n_jobs, st)
            st.done(path, **{"cells_" + b: int(bands[b]["curve"].shape[0]) for b in BANDS})
    progress_module.summary("sky.derived.pacs_completeness", len(names), t0)


if __name__ == "__main__":
    run(build)
