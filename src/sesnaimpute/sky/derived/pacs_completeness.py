"""The Herschel PACS completeness of the SESNA sources, per region and band
(SPEC_PACS_FARIR_DRAFT.md sections 5 and 11.3 step 2).

`build(config, regions=None)` writes one product per region,
`sky/derived/pacs/completeness_pacs_source__<Region>.hdf5`, in EXACTLY the layout of
`catalog/sesna/completeness_sesna_source__<Region>.hdf5` for the bands P70 and P100
(`CURVE_<B>`, `MAG_GRID_<B>`, `SOURCE_ROW`, `PIXEL_CURVE_<B>`, `PIXEL_COVERED_FRAC_<B>`,
`ENVELOPE_<B>`, `COMPLETED_FROM_<B>`, `N_POS_<B>`, `HPX_PIX_512`), plus per-cell columns aligned
with the CURVE rows (`CELL_RA_<B>`, `CELL_DEC_<B>`, `CELL_OBSID_<B>`, `CELL_MAP_<B>`,
`CELL_SIGMA_<B>_MJY`, `CELL_GAIN_<B>`) that the photometry stage reads.

Method. The PACS measurement (`pacs_fit`) is linear in the map, so a point source of flux F at a
position p is measured as g F + b_p, where g is the fit's gain and b_p its value on the real, un-injected
map there (background, confusion and noise together). Per 30-arcsec cell on each map tile, 200
positions are drawn uniformly (no exclusion of real sources: confusion is part of the completeness),
each gets one forced fit with the bright neighbours of the tile as nuisance columns, and:
  cell sigma = the median over the cell's 3 x 3 block of 1.4826 x the median of |b_p| (scatter about
  zero, since a detection counts from zero) over each cell's clean positions (one cell holds only ~10 independent beams), times one factor per
  map tile that matches the negative tail of b_p / sigma over the tile's empty sky to a
  Gaussian's (the noise IS the scatter of the extracted flux on empty sky, not added to the propagated
  pixel noise; the factor corrects the MAD for the noise's heavier-than-Gaussian tails);
  completeness at flux F = the fraction of positions with g F + b_p >= 5 x cell sigma, i.e. the
  empirical distribution of (5 sigma - b_p)/g evaluated at every plane of the grid.
The sigma of a cell is what the photometry stage calls the uncertainty of a source in that cell, so a
measured source and its completeness curve use one noise.

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

BANDS = ("P70", "P100")
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
#: one-sided Gaussian probability beyond 3 sigma: the negative tail of b / sigma over a tile's clean
#: positions is matched to it (real sources only ADD flux, so the negative side is pure noise)
TAIL_PROBABILITY = 0.00135


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
    fit = tile.fit_positions(x.ravel(), y.ravel())
    shape = x.shape
    valid = (fit.covered & np.isfinite(fit.flux) & np.isfinite(fit.gain) & (fit.gain > 0)).reshape(shape)
    clean = ~tile.near_bright(x, y)
    return (fit.flux.reshape(shape), fit.gain.reshape(shape), fit.sigma.reshape(shape), valid, clean)


#: a cell's own noise estimate counts in its block's median only with this many clean positions
MIN_CLEAN_POSITIONS = 50


def cell_statistics(b, gain, pixel_sigma, valid, clean):
    """Per cell: n_valid, n_clean, clean sigma, median gain, median propagated pixel sigma. The clean
    sigma is 1.4826 x the median of |b| over the valid positions farther than one beam from every bright
    neighbour (so a real source in the cell does not set the noise): the scatter about ZERO, not about
    the cell's own median, because a detection is counted from zero and each cell's empty-sky fluxes
    carry a background offset of their own (Musca P70: +-0.5 sigma, 16-84 %), which is part of a
    source's error."""
    use = valid & clean
    bb = np.where(use, b, np.nan)
    with np.errstate(all="ignore"):
        sigma = MAD_TO_SIGMA * np.nanmedian(np.abs(bb), axis=1)
        g = np.nanmedian(np.where(valid, gain, np.nan), axis=1)
        ps = np.nanmedian(np.where(valid, pixel_sigma, np.nan), axis=1)
    return valid.sum(axis=1), use.sum(axis=1), sigma, g, ps


def thresholds(b, gain, valid, sigma):
    """(5 sigma - b)/g at every position, NaN where invalid. The curve uses ALL valid positions."""
    with np.errstate(all="ignore"):
        return (pacs_fit.DETECTION_SNR * sigma[:, None] - np.where(valid, b, np.nan)) / np.where(valid, gain, np.nan)


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


def _tile_tail_factor(piece, config, records):
    """(k, the tile's median block sigma before k) for one map tile, one pieces.run piece per tile
    before the cell blocks. k scales the block-median sigma so that the negative tail of b / sigma
    over the tile's clean empty-sky positions matches a Gaussian's (real sources only ADD flux, so the
    negative side is pure noise): k = -q / 3, q the TAIL_PROBABILITY quantile over the clean positions
    of TAIL_CELLS evenly spread cells (their 3 x 3 blocks fitted too). Musca: P70 1.07, P100 1.01-1.12."""
    tile = pacs_fit.cached_tile(config, records[piece])
    centres = np.unique(np.linspace(0, tile.grid.n_cells - 1, TAIL_CELLS).astype(np.int64))
    all_ids = ring(tile.grid, centres)
    b, gain, ps, valid, clean = cell_fits(tile, all_ids, 0)
    _, n_clean, mad, _, _ = cell_statistics(b, gain, ps, valid, clean)
    sigma, _ = block_sigma(tile.grid, centres, all_ids, mad, n_clean)
    c = np.searchsorted(all_ids, centres)
    good = np.isfinite(sigma) & (sigma > 0)
    z = (b[c][good] / sigma[good, None])[(valid & clean)[c][good]]
    return (float(-np.quantile(z, TAIL_PROBABILITY) / 3.0) if z.size else 1.0,
            float(np.median(sigma[good])) if good.any() else np.nan)


def curve_counts(threshold, flux_grid):
    """(n_cell, n_plane) number of positions with threshold <= plane flux (valid ones only; NaN
    thresholds never count). Monotone in the plane by construction."""
    with np.errstate(invalid="ignore"):
        return (threshold[:, :, None] <= flux_grid[None, None, :]).sum(axis=1)


def encode_curves(counts, n_valid):
    """Fraction x STORAGE_SCALE, rounded, uint16."""
    return np.rint(counts * STORAGE_SCALE / np.maximum(n_valid, 1)[:, None]).astype(np.uint16)


def _piece(piece, config, records, flux_grid, tails):
    """One tile's square block of cells -> the arrays the parent assembles. The block's border cells are
    fitted too (same per-cell random positions as in their own piece), only for the noise median."""
    tile_index, ids = piece
    record = records[tile_index]
    tile = pacs_fit.cached_tile(config, record)
    all_ids = ring(tile.grid, ids)
    b, gain, ps, valid, clean = cell_fits(tile, all_ids, tile_index + 1)
    n_valid, n_clean, mad, g, pps = cell_statistics(b, gain, ps, valid, clean)
    sigma, n_sigma = block_sigma(tile.grid, ids, all_ids, mad, n_clean)
    tail, tile_median = tails[tile_index]
    # a block with no clean cell (wholly within a beam of bright sources) takes the tile's median noise;
    # N_SIGMA_POS 0 marks it
    sigma = np.where(np.isfinite(sigma), sigma, tile_median) * tail
    k = np.searchsorted(all_ids, ids)
    b, gain, valid, n_valid, g, pps = b[k], gain[k], valid[k], n_valid[k], g[k], pps[k]
    covered = (n_valid >= MIN_VALID_POSITIONS) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(g)
    curve = encode_curves(curve_counts(thresholds(b, gain, valid, sigma), flux_grid), n_valid)
    ra, dec = tile.grid.centres_icrs(ids)
    return dict(tile=tile_index, ids=ids, covered=covered, n_valid=n_valid.astype(np.int16),
                n_sigma=np.minimum(n_sigma, np.iinfo(np.int16).max).astype(np.int16), tail=tail,
                sigma=sigma, gain=g, pixel_sigma=pps, curve=curve, ra=ra, dec=dec)


# ----------------------------------------------------------------------------
# one band of one region

def _tile_geometry(record):
    wcs = WCS(fits.Header.fromstring(record.wcs_header)).celestial
    return pacs_fit.MapGrid.from_wcs(wcs, record.shape), wcs


def _band_product(config, records, ra, dec, sample_ra, sample_dec, n_jobs, stage, flux_grid):
    """Measure every candidate cell of every map of one band and choose the deepest. Returns a dict
    of the band's datasets, or None when the band has no map."""
    n_src = ra.size
    n_plane = flux_grid.size
    n_adm = sample_ra.shape[0]
    sample_ra, sample_dec = sample_ra.ravel(), sample_dec.ravel()
    # candidate cells of each tile: those holding a source or an admitted sample point
    pieces, geometry = [], []
    for t, record in enumerate(records):
        grid, wcs = _tile_geometry(record)
        sx, sy = (np.asarray(v) for v in wcs.all_world2pix(ra, dec, 0))
        px, py = (np.asarray(v) for v in wcs.all_world2pix(sample_ra, sample_dec, 0))
        src_cell, smp_cell = grid.cell_ids(sx, sy), grid.cell_ids(px, py)
        cells = np.unique(np.concatenate([src_cell[src_cell >= 0], smp_cell[smp_cell >= 0]]))
        geometry.append((src_cell, smp_cell, cells))
        cy, cx = np.divmod(cells, grid.nx)
        block = (cy // BLOCK_CELLS) * (grid.nx // BLOCK_CELLS + 1) + cx // BLOCK_CELLS
        for blk in np.unique(block):
            pieces.append((t, cells[block == blk]))
    if not pieces:
        return None
    tails = pieces_module.run(functools.partial(_tile_tail_factor, config=config, records=records),
                              sorted({t for t, _ in pieces}), n_jobs, stage=stage, unit="tiles")
    tails = dict(zip(sorted({t for t, _ in pieces}), tails))
    fn = functools.partial(_piece, config=config, records=records, flux_grid=flux_grid, tails=tails)
    results = pieces_module.run(fn, pieces, n_jobs, stage=stage, unit="cell blocks")
    # per tile arrays over its candidate cells
    tile_data = []
    for t, (_s, _p, cells) in enumerate(geometry):
        mine = [r for r in results if r["tile"] == t]
        # pieces are square blocks, so their cells interleave: put the tile's arrays back in the
        # sorted order of `cells`, which the searchsorted lookups below rely on
        order = np.argsort(np.concatenate([r["ids"] for r in mine])) if mine else None
        out = {k: (np.concatenate([r[k] for r in mine])[order] if mine else None) for k in
               ("covered", "n_valid", "n_sigma", "sigma", "gain", "pixel_sigma", "curve", "ra", "dec")}
        tile_data.append(out)
        if mine:
            print("  %s: noise tail factor %.3f" % (records[t].map_id, mine[0]["tail"]), flush=True)
    # deepest map at every source and every sample point
    sel_src_t = np.full(n_src, -1, np.int32)
    sel_src_r = np.full(n_src, np.inf)
    sel_smp_t = np.full(n_adm * 256, -1, np.int32)
    sel_smp_r = np.full(n_adm * 256, np.inf)
    for t, (src_cell, smp_cell, cells) in enumerate(geometry):
        d = tile_data[t]
        if d["covered"] is None:
            continue
        for cell_of, sel_t, sel_r in ((src_cell, sel_src_t, sel_src_r), (smp_cell, sel_smp_t, sel_smp_r)):
            idx = np.flatnonzero(cell_of >= 0)
            k = np.searchsorted(cells, cell_of[idx])
            pacs_fit.select_deeper(t, idx, d["sigma"][k], d["covered"][k], sel_t, sel_r)
    # the rows: cells that win at least one source or sample, ordered by (tile, cell)
    keys = []
    for t, (src_cell, smp_cell, cells) in enumerate(geometry):
        for cell_of, sel_t in ((src_cell, sel_src_t), (smp_cell, sel_smp_t)):
            w = np.flatnonzero(sel_t == t)
            keys.append(np.int64(t) * (1 << 40) + cell_of[w])
    keys = np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64)

    def rows_of(cell_of, sel_t):
        out = np.full(sel_t.size, -1, dtype=np.int64)
        w = np.flatnonzero(sel_t >= 0)
        out[w] = np.searchsorted(keys, sel_t[w].astype(np.int64) * (1 << 40) + cell_of[w])
        return out

    src_row = np.full(n_src, -1, dtype=np.int64)
    smp_row = np.full(n_adm * 256, -1, dtype=np.int64)
    for t, (src_cell, smp_cell, _c) in enumerate(geometry):
        for cell_of, sel_t, row in ((src_cell, sel_src_t, src_row), (smp_cell, sel_smp_t, smp_row)):
            w = np.flatnonzero(sel_t == t)
            row[w] = np.searchsorted(keys, np.int64(t) * (1 << 40) + cell_of[w])
    n_row = keys.size
    tile_of_row = (keys >> 40).astype(np.int64)
    cell_of_row = keys & ((1 << 40) - 1)
    curve = np.zeros((n_row, n_plane), np.uint16)
    cols = {k: np.zeros(n_row, dt) for k, dt in (("n_valid", np.int16), ("n_sigma", np.int16), ("sigma", np.float32),
                                                  ("gain", np.float32), ("pixel_sigma", np.float32),
                                                  ("ra", np.float64), ("dec", np.float64))}
    for t, (_s, _p, cells) in enumerate(geometry):
        r = np.flatnonzero(tile_of_row == t)
        if r.size == 0:
            continue
        k = np.searchsorted(cells, cell_of_row[r])
        curve[r] = tile_data[t]["curve"][k]
        for name in cols:
            cols[name][r] = tile_data[t][name][k]
    # area-averaged curve over each admitted nside-512 pixel
    rows2d = smp_row.reshape(n_adm, 256)
    pixel_curve = np.zeros((n_adm, n_plane), np.float32)
    covered_frac = (rows2d >= 0).mean(axis=1).astype(np.float32)
    for lo in range(0, n_adm, 400):
        r = rows2d[lo:lo + 400]
        ok = r >= 0
        total = (curve[np.where(ok, r, 0)].astype(np.float32) * ok[:, :, None]).sum(axis=1)
        pixel_curve[lo:lo + 400] = total / (STORAGE_SCALE * np.maximum(ok.sum(axis=1), 1)[:, None])
    envelope = curve.max(axis=0) if n_row else np.zeros(n_plane, np.uint16)
    obsid = np.array([",".join(records[t].obsids) for t in tile_of_row], dtype="S64")
    map_id = np.array([records[t].map_id for t in tile_of_row], dtype="S96")
    return dict(curve=curve, source_row=src_row.astype(np.int32), pixel_curve=pixel_curve,
                covered_frac=covered_frac, envelope=envelope, n_pos=cols["n_valid"], n_sigma=cols["n_sigma"],
                cell_ra=cols["ra"], cell_dec=cols["dec"], cell_sigma=cols["sigma"],
                cell_gain=cols["gain"], cell_pixel_sigma=cols["pixel_sigma"],
                cell_obsid=obsid, cell_map=map_id, n_candidate=sum(g[2].size for g in geometry))


def _empty_band(n_src, n_adm, n_plane):
    return dict(curve=np.zeros((0, n_plane), np.uint16), source_row=np.full(n_src, -1, np.int32),
                pixel_curve=np.zeros((n_adm, n_plane), np.float32),
                covered_frac=np.zeros(n_adm, np.float32), envelope=np.zeros(n_plane, np.uint16),
                n_pos=np.zeros(0, np.int16), n_sigma=np.zeros(0, np.int16), cell_ra=np.zeros(0), cell_dec=np.zeros(0),
                cell_sigma=np.zeros(0, np.float32), cell_gain=np.zeros(0, np.float32),
                cell_pixel_sigma=np.zeros(0, np.float32), cell_obsid=np.zeros(0, "S64"),
                cell_map=np.zeros(0, "S96"), n_candidate=0)


# ----------------------------------------------------------------------------
# writing

def write_product(path, region, admitted, mag, bands):
    """Write the region product at `path` (the final shapes are known here: nothing grows)."""
    stem = "completeness_pacs_source"
    n_src = bands[BANDS[0]]["source_row"].size
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
            w("PIXEL_CURVE_" + b, d["pixel_curve"])
            w("PIXEL_COVERED_FRAC_" + b, d["covered_frac"])
            w("ENVELOPE_" + b, d["envelope"])
            w("COMPLETED_FROM_" + b, np.full(d["curve"].shape[0], np.nan, np.float32))
            w("N_POS_" + b, d["n_pos"])
            w("N_SIGMA_POS_" + b, d["n_sigma"])
            w("CELL_RA_" + b, d["cell_ra"])
            w("CELL_DEC_" + b, d["cell_dec"])
            w("CELL_OBSID_" + b, d["cell_obsid"])
            w("CELL_MAP_" + b, d["cell_map"])
            w("CELL_SIGMA_%s_MJY" % b, d["cell_sigma"])
            w("CELL_GAIN_" + b, d["cell_gain"])
        w = lambda name, data: build_module.write_dataset(f, name, data, *REGISTRY[(stem, name)])
        w("SOURCE_ROW", np.stack([bands[b]["source_row"] for b in BANDS], axis=1).astype(np.int32))
        w("HPX_PIX_512", np.asarray(admitted, dtype=np.int64))


def _build_region(config, region, out_path, ra, dec, admitted, n_jobs, stage=None):
    """Measure and write one region. `ra`/`dec` are the region's source positions, `admitted` its
    nside-512 pixels. Returns the per-band products (for the caller's checks)."""
    mag, flux_grid = canonical_grid()
    sample_ra, sample_dec = pacs_fit.admitted_sample_points(admitted)
    out = {}
    for b in BANDS:
        records = [r for r in pacs_fit.read_index(config, region) if r.band == b]
        out[b] = (_band_product(config, records, ra, dec, sample_ra, sample_dec, n_jobs, stage, flux_grid)
                  if records else None)
        if out[b] is None:
            out[b] = _empty_band(ra.size, len(admitted), mag.size)
    write_product(out_path, region, admitted, mag, out)
    return out


def _region_inputs(config, region):
    with h5py.File(product_path(config, "catalog", "sesna", "sources", "source", region=region), "r") as f:
        ra, dec = f["RA_DEG"][:], f["DEC_DEG"][:]
    with h5py.File(product_path(config, "catalog", "sesna", "completeness", "source", region=region), "r") as f:
        admitted = f["HPX_PIX_512"][:]
    return ra, dec, admitted


def build(config, regions=None):
    """Writes each region's `completeness_pacs_source` product, regions one after another."""
    names = regions or [r.name for r in regions_module.REGIONS]
    t0 = __import__("time").time()
    for region in names:
        with progress_module.Stage("sky.derived.pacs_completeness", region) as st:
            ra, dec, admitted = _region_inputs(config, region)
            path = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
            bands = _build_region(config, region, path, ra, dec, admitted, config.n_jobs, st)
            st.done(path, **{"cells_" + b: int(bands[b]["curve"].shape[0]) for b in BANDS})
    progress_module.summary("sky.derived.pacs_completeness", len(names), t0)


if __name__ == "__main__":
    run(build)
