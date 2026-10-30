"""Herschel PACS 70 and 100 micron flux at every SESNA source position
(SPEC_PACS_FARIR_DRAFT.md sections 4, 7b and 11.3 step 3).

`build(config, regions=None, box=None)` measures, for every source of
the curated positions of the region (`catalog.curated.read_positions`: the delivery's rows in the curated order), the flux in each PACS
band on the band's one co-added mosaic of the region (`sky.derived.pacs_mosaic`; the map the completeness stage
(`sky.derived.pacs_completeness`) chose for the source's cell, a row of its MAP_* tables; `box` opens a development
slice of the mosaic), and
writes `sky/derived/pacs/photometry_pacs_source__<Region>.hdf5`: one row per
source in the catalogue's own order, with its flux, its noise, the origin
code the fitter reads (1 detected, 3 bound from a blend or a confusion,
90 observed but not detected, 91 not covered), the blend bookkeeping (how many detections lie within a beam, and the
brightest of them) and the extended-source flag.

Everything numerical is `sky.derived.pacs_fit`'s: the eHOPS aperture photometry and the fitted width. This
stage only decides which sources go to which map tile, runs the singleton pass, applies the blend rule and turns
the measurements into products:

- flux: FNU is the aperture flux minus its cell's empty-sky offset, `CELL_OFFSET_<B>_MJY` (stored per source;
  adding it back gives the HOPS/eHOPS convention).
- noise: a detected source's sigma is the photometric error the completeness stage MEASURED for a source of
  its flux in its cell, by recovering artificial sources of known flux injected into the map: the cell's
  `SCATTER_<B>_MJY` at the source's flux (log-linear between the flux steps). A source that is not detected
  carries the noise of its 30-arcsec cell, `CELL_SIGMA_<B>_MJY`, the same number the detection rule uses.
- detection: flux - CELL_OFFSET >= 5 x CELL_SIGMA (origin 1), else origin 90; a source the completeness stage
  found no cell for, or whose aperture has too little valid data, is 91.
- blends (the far-infrared survey convention; planner ruling, owner-approved 2026-10-28): a source within one
  beam of another source detected on the map is an unseparable blend, origin 3, its own aperture flux (offset
  subtracted, so it holds the neighbour's light) as the bound and the measurement's own error as
  its sigma, unless it is itself detected and the brightest of the detections within that beam.
- extended: a detected source whose fitted width exceeds 1.5 x the beam.

Work is split by map and by position: each band's mosaic has its sources cut, in order of declination, into pieces of
at least PHOTOMETRY_PIECE_MIN sources (`pieces.slices`), run over all the workers; a worker transforms a piece's sources to
map pixels in one call and shares the prepared mosaic with its other pieces. The product is preallocated and
each piece's rows are written as they arrive.
"""

import functools
import os
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived import pacs_fit
from sesnaimpute.sky.derived.pacs_fit import DETECTION_SNR
from sesnaimpute.sky.download.herschel_pacs.index import read_index

BANDS = ("P70", "P100", "P160")
#: least sources in one photometry piece (a piece costs a few ms of overhead; a source about 3 ms)
PHOTOMETRY_PIECE_MIN = 2000

#: the pipeline's origin codes for a flux (`ORIGIN_FNU` in the source table)
ORIGIN_DETECTED, ORIGIN_BOUND, ORIGIN_UNDETECTED, ORIGIN_NOT_COVERED = 1, 3, 90, 91

#: a detection fitted wider than this many beam widths is flagged extended
#: (SPEC 11.3 step 3); the width is `pacs_fit.Tile.fitted_fwhm`, the beam
#: broadened by a Gaussian chosen by chi-square
EXTENDED_BEAM_FACTOR = 1.5
#: a source within this many beam FWHM of another source's detection is an unseparable blend (ORIGIN 3) unless it is
#: the brighter member: the far-infrared survey convention (HOPS/eHOPS flag close pairs the same way; planner ruling,
#: owner-approved 2026-10-28)
BLEND_RADIUS_BEAMS = 1.0

#: the SESNA association of the region's PACS sources (planner ruling, owner-approved 2026-10-29: Rob Gutermuth's
#: band-merging recipe as eHOPS applies it, Pokhrel et al. 2023 Sect. III.2 and Appendix C): nearest neighbour within a
#: fixed radius in every band, set by the two catalogues' astrometric accuracy (Gutermuth et al. 2009 Sect. 2.3: 1 arcsec
#: IRAC/2MASS, 1.3 arcsec MIPS 24 um; eHOPS: 2 arcsec for PACS on the Herschel header astrometry, no registration)
MATCH_RADIUS_ARCSEC = 2.0
#: the SESNA density behind the expected chance coincidences is counted within this radius of each PACS source
CHANCE_DENSITY_ARCSEC = 60.0
#: an input map's median PACS - SESNA offset is printed where it holds at least this many associated pairs
MIN_OFFSET_PAIRS = 5


# ---------------------------------------------------------------------------
# positions

def _unit_vectors(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])



# ---------------------------------------------------------------------------
# the singleton pass and the calibrated sigma

#: bins of injected S/N (F / CELL_SIGMA) in which the completeness stage's C2 checks the measured error: < 8, 8-30,
#: 30-120, >= 120
SNR_BIN_EDGES = (8.0, 30.0, 120.0)
SNR_BIN_CENTRES = (5.7, 15.5, 60.0, 240.0)


def singleton_pass(tile, ra, dec, shift, sigma_cell, offset):
    """The aperture measurement of every source of `ra`, `dec` on `tile` and the catalogue's detection of
    it: returns `(x, y, single, ok, detected)`, the map pixels (the map's registration `shift` applied),
    the aperture fit, whether the source has a usable cell noise and fit, and whether its flux minus the
    cell's empty-sky offset reaches DETECTION_SNR x the cell noise. The completeness stage runs this same
    function on the same inputs to know which sources the catalogue detects."""
    x, y = tile.xy(*pacs_fit.to_map_frame(ra, dec, shift))
    single = tile.measure(x, y)
    ok = single.covered & np.isfinite(sigma_cell) & (sigma_cell > 0)
    detected = ok & (single.flux - offset >= DETECTION_SNR * sigma_cell)
    return x, y, single, ok, detected


def calibrated_sigma(scatter, flux_grid, flux):
    """The photometric error of sources of flux `flux` (n,) mJy from their cells' measured scatter at the
    grid planes: `scatter` (n, n_plane), `flux_grid` (n_plane,) mJy, bright to faint. Log-linear in flux
    between the two planes that bracket the flux, clamped to the grid ends; where either bracketing plane
    holds NaN the value of the nearest plane of the same row that is finite; NaN for a row with none."""
    asc = np.log10(flux_grid[::-1])
    rows = np.asarray(scatter, float)[:, ::-1]
    n, p = rows.shape
    t = np.log10(np.maximum(np.asarray(flux, float), 1e-300))
    hi = np.clip(np.searchsorted(asc, t), 1, p - 1)
    lo = hi - 1
    w = np.clip((t - asc[lo]) / (asc[hi] - asc[lo]), 0.0, 1.0)
    r = np.arange(n)
    both = np.isfinite(rows[r, lo]) & np.isfinite(rows[r, hi])
    out = np.where(both, rows[r, lo] * (1 - w) + rows[r, hi] * w, np.nan)
    need = np.flatnonzero(~both)
    if need.size:
        finite = np.isfinite(rows[need])
        away = np.where(finite, np.abs(np.arange(p)[None, :] - (lo[need] + w[need])[:, None]), np.inf)
        k = np.argmin(away, axis=1)
        out[need] = np.where(finite.any(axis=1), rows[need, k], np.nan)
    return out


# ---------------------------------------------------------------------------
# one tile

def _measure_tile(piece, config, records, ra, dec, cell_sigma, cell_offset, shifts, flux_grid):
    """All of one map tile's SESNA sources: `piece = (band, map_index, rows)`, the source rows (ascending) the
    completeness stage assigned to that map: the forced aperture flux at each position (the value an ORIGIN 90 source
    carries) and its cell's noise; ORIGIN 90 where covered, 91 where not. The cross-match with the PACS sources
    (`cross_match`) sets ORIGIN 1 and 2 after. Returns `(band, map_index, rows, columns)`."""
    band, map_index, rows = piece
    record = records[map_index]
    n = rows.size
    sigma_cell = cell_sigma[band][rows]
    offset = cell_offset[band][rows]
    flux = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    origin = np.full(n, ORIGIN_NOT_COVERED, dtype=np.int8)
    subtracted = np.full(n, np.nan)
    tile = pacs_fit.cached_tile(config, record)
    # the map's registration to SESNA, as the completeness stage measured and stored it
    shift = shifts.get((BANDS[band], map_index), (0.0, 0.0))
    x, y, single, ok, _floor = singleton_pass(tile, ra[rows], dec[rows], shift, sigma_cell, offset)
    sigma_annulus = single.sigma
    flux[ok] = single.flux[ok] - offset[ok]
    sigma[ok] = sigma_cell[ok]
    origin[ok] = ORIGIN_UNDETECTED
    subtracted[ok] = offset[ok]
    columns = dict(FNU=flux, SIGMA=sigma, ORIGIN=origin, LOCAL_RMS=sigma_cell.astype(np.float32), CELL_OFFSET=subtracted,
                   SIGMA_ANNULUS=np.asarray(sigma_annulus, np.float32), N_BLEND=np.zeros(n, np.int16),
                   PACS_SOURCE=np.full(n, -1, np.int32), MATCH_SEPARATION=np.full(n, np.nan, np.float32),
                   EXTENDED=np.zeros(n, bool), CONFUSED=np.zeros(n, bool))
    columns["LOCAL_RMS"][origin == ORIGIN_NOT_COVERED] = np.nan
    return band, map_index, rows, columns


def _sep_arcsec(u, v):
    return np.degrees(2.0 * np.arcsin(np.clip(np.linalg.norm(u - v, axis=-1) / 2.0, 0.0, 1.0))) * 3600.0


def cross_match(path, config, region, ra, dec, box=None):
    """The SESNA association of the region's PACS sources (`sources_pacs_region`, merged across bands; planner ruling,
    owner-approved 2026-10-29: Rob Gutermuth's band-merging recipe as eHOPS applies it) on the photometry product at
    `path` (rewritten in place). Each SESNA source takes the nearest PACS source within MATCH_RADIUS_ARCSEC. If that PACS
    source has this SESNA source alone within the radius: ORIGIN 1 in every band that detects it, FNU/SIGMA the PACS
    source's. If it has two or more SESNA sources within the radius (a blend): ORIGIN 3 in every detecting band, the
    bound that band's flux and SIGMA that detection's own error. A band that does not detect the PACS source keeps the forced measurement (ORIGIN 90 or 91),
    unless a detection of that band lies within one beam of the SESNA source (the band's local beam there, from its mosaic's
    input maps; the per-band confusion flag): then
    ORIGIN 3 (CONFUSED true) with the nearest such detection's flux as the bound. Returns per-band counts {m1, m2, confused}, the
    expected chance coincidences (the local SESNA density within
    CHANCE_DENSITY_ARCSEC x the radius's area, summed over the PACS sources) and the ORIGIN 1 pairs' offsets PACS - SESNA
    keyed by position band index: an (n, 4) array of east, north (arcsec) and the PACS position (ra, dec deg)."""
    src = product_path(config, "sky/derived", "pacs", "sources", "region", region=region)
    with h5py.File(src, "r") as f:
        n = f["RA_DEG"].shape[0]
        pra, pde, pb = f["RA_DEG"][:], f["DEC_DEG"][:], f["POSITION_BAND"][:].astype(int)
        cols = {b: dict(det=f["DETECTED_" + b][:], fnu=f["FNU_%s_MJY" % b][:], sig=f["SIGMA_%s_MJY" % b][:],
                        off=f["CELL_OFFSET_%s_MJY" % b][:], ext=f["EXTENDED_" + b][:], map=f["MAP_INDEX_" + b][:],
                        ra=f["RA_%s_DEG" % b][:], dec=f["DEC_%s_DEG" % b][:])
                for b in BANDS}
    out = {b: dict(m1=0, m2=0, confused=0) for b in BANDS}
    if n == 0 or ra.size == 0:
        return out, 0.0, {}
    su = _unit_vectors(ra, dec)
    pu = _unit_vectors(pra, pde)
    chord = 2.0 * np.sin(np.radians(MATCH_RADIUS_ARCSEC / 3600.0) / 2.0)
    stree, ptree = cKDTree(su), cKDTree(pu)
    claim = stree.query_ball_point(pu, chord)                       # SESNA sources within the radius, per PACS source
    n_local = stree.query_ball_point(pu, 2.0 * np.sin(np.radians(CHANCE_DENSITY_ARCSEC / 3600.0) / 2.0), return_length=True)
    chance = float(np.sum(n_local / (np.pi * CHANCE_DENSITY_ARCSEC ** 2)) * np.pi * MATCH_RADIUS_ARCSEC ** 2)
    dist, near = ptree.query(su, distance_upper_bound=chord)        # each SESNA source's nearest PACS source
    pairs = {}
    with h5py.File(path, "r+") as f:
        data = {b: {c: f[_NAMES[c].format(b=b)][:] for c in ("FNU", "SIGMA", "ORIGIN", "CELL_OFFSET", "N_BLEND",
                                                             "PACS_SOURCE", "MATCH_SEPARATION", "EXTENDED", "CONFUSED")} for b in BANDS}
        for i in np.flatnonzero(np.isfinite(dist)):
            k = int(near[i])
            sole = len(claim[k]) == 1
            sep = float(_sep_arcsec(su[i], pu[k]))
            if sole:
                cosd = np.cos(np.radians(dec[i]))
                pairs.setdefault(int(pb[k]), []).append(
                    ((pra[k] - ra[i]) * cosd * 3600.0, (pde[k] - dec[i]) * 3600.0, pra[k], pde[k]))
            for b in BANDS:
                c, d = cols[b], data[b]
                if not c["det"][k]:
                    continue
                d["FNU"][i], d["CELL_OFFSET"][i] = c["fnu"][k], c["off"][k]
                d["PACS_SOURCE"][i], d["MATCH_SEPARATION"][i], d["N_BLEND"][i] = k, sep, len(claim[k])
                if sole:
                    d["ORIGIN"][i], d["SIGMA"][i], d["EXTENDED"][i] = ORIGIN_DETECTED, c["sig"][k], c["ext"][k]
                    out[b]["m1"] += 1
                else:
                    d["ORIGIN"][i], d["SIGMA"][i] = ORIGIN_BOUND, c["sig"][k]
                    out[b]["m2"] += 1
        # the per-band confusion flag (owner-approved 2026-10-29): a covered source with no detection of the band
        # associated, but the band's detection (at its own position) within one beam of it, cannot be called
        # undetected: ORIGIN 3, CONFUSED, the nearest such detection's flux the bound
        for b in BANDS:
            c, d = cols[b], data[b]
            det = np.flatnonzero(c["det"])
            out[b]["confused"] = 0
            if det.size == 0:
                continue
            psf = pacs_fit.mosaic_psf(config, region, b, box)
            bu = _unit_vectors(c["ra"][det], c["dec"][det])
            bd, bj = cKDTree(bu).query(su, distance_upper_bound=2.0 * np.sin(np.radians(psf.sub_beams.max() / 3600.0) / 2.0))
            within = np.isfinite(bd)
            within[within] = bd[within] < 2.0 * np.sin(np.radians(psf.beam_at_sky(ra[within], dec[within]) / 3600.0) / 2.0)
            for i in np.flatnonzero(within & (d["ORIGIN"] == ORIGIN_UNDETECTED)):
                k = int(det[bj[i]])
                d["FNU"][i], d["CELL_OFFSET"][i], d["SIGMA"][i] = c["fnu"][k], c["off"][k], c["sig"][k]
                d["ORIGIN"][i], d["PACS_SOURCE"][i], d["N_BLEND"][i] = ORIGIN_BOUND, k, len(claim[k])
                d["CONFUSED"][i] = True
                d["MATCH_SEPARATION"][i] = float(_sep_arcsec(su[i], bu[bj[i]]))
                out[b]["confused"] += 1
        for b in BANDS:
            for c, v in data[b].items():
                f[_NAMES[c].format(b=b)][...] = v
            f["ORIGIN_" + b].attrs["MATCH_RADIUS_ARCSEC"] = MATCH_RADIUS_ARCSEC
        f["ORIGIN_P70"].attrs["EXPECTED_CHANCE_COINCIDENCES"] = chance
    offsets = {key: np.array(v, float) for key, v in pairs.items()}
    return out, chance, offsets


def print_offsets(config, region, offsets, box=None):
    """The median offsets PACS - SESNA of the associated pairs, one line per INPUT map of each band's mosaic (the check
    eHOPS made per observation, Pokhrel et al. 2023 Appendix C: a fraction of the 2 arcsec radius, so no registration
    is needed): the pairs (`offsets`, from `cross_match`) whose PACS position falls inside the input map's footprint
    (its WCS and image shape), for each input map with at least MIN_OFFSET_PAIRS of them. Information only."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from sesnaimpute.sky.derived import pacs_mosaic
    index = {r.path: r for r in read_index(config, region)}
    for bi, b in enumerate(BANDS):
        pairs = offsets.get(bi)
        mdir = pacs_mosaic.product_dir(config, region, b, box)
        if pairs is None or not os.path.exists(os.path.join(mdir, "mosaic.hdf5")):
            print("  %s-positioned: no associated pairs" % b, flush=True)
            continue
        rows = []
        for q in pacs_mosaic.Mosaic(mdir).meta["maps"]:
            rec = index[q["path"]]
            wcs = WCS(fits.Header.fromstring(rec.wcs_header)).celestial
            x, y = wcs.all_world2pix(pairs[:, 2], pairs[:, 3], 0)
            inside = (x >= -0.5) & (x < rec.shape[1] - 0.5) & (y >= -0.5) & (y < rec.shape[0] - 0.5)
            if inside.sum() >= MIN_OFFSET_PAIRS:
                rows.append((os.path.basename(q["path"]), int(inside.sum()), float(np.median(pairs[inside, 0])),
                             float(np.median(pairs[inside, 1]))))
        if not rows:
            print("  %s-positioned: no input map with >= %d associated pairs" % (b, MIN_OFFSET_PAIRS), flush=True)
            continue
        mag = np.array([np.hypot(e, nn) for _, _, e, nn in rows])
        print("  %s-positioned: median offset PACS - SESNA inside each of %d input maps of the mosaic: |offset| median %.2f, "
              "max %.2f arcsec (radius %.1f)" % (b, len(rows), np.median(mag), mag.max(), MATCH_RADIUS_ARCSEC), flush=True)
        for (name, npair, e, nn), r in zip(rows, mag):
            print("    %s: %d pairs, east %+.2f north %+.2f (|%.2f|)" % (name, npair, e, nn, r), flush=True)


# ---------------------------------------------------------------------------
# the region

def _runs(rows):
    """Half-open `(start, stop, offset)` runs of consecutive values in the
    ascending `rows` (offset = where the run begins in `rows`)."""
    if rows.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [rows.size]])
    return [(int(rows[a]), int(rows[b - 1]) + 1, int(a)) for a, b in zip(starts, stops)]


_DTYPES = dict(CONFUSED=bool, FNU=np.float64, SIGMA=np.float64, CELL_OFFSET=np.float64, ORIGIN=np.int8, LOCAL_RMS=np.float32,
               N_BLEND=np.int16, PACS_SOURCE=np.int32, MATCH_SEPARATION=np.float32, EXTENDED=bool, SIGMA_ANNULUS=np.float32)
_FILL = dict(CONFUSED=False, FNU=np.nan, SIGMA=np.nan, CELL_OFFSET=np.nan, ORIGIN=ORIGIN_NOT_COVERED, LOCAL_RMS=np.nan,
             N_BLEND=0, PACS_SOURCE=-1, MATCH_SEPARATION=np.nan, EXTENDED=False, SIGMA_ANNULUS=np.nan)
_NAMES = dict(CONFUSED="CONFUSED_{b}", FNU="FNU_{b}_MJY", SIGMA="SIGMA_{b}_MJY", CELL_OFFSET="CELL_OFFSET_{b}_MJY", ORIGIN="ORIGIN_{b}", LOCAL_RMS="LOCAL_RMS_{b}_MJY",
              N_BLEND="N_BLEND_{b}", PACS_SOURCE="PACS_SOURCE_{b}", MATCH_SEPARATION="MATCH_SEPARATION_{b}_ARCSEC",
              EXTENDED="EXTENDED_{b}",
              SIGMA_ANNULUS="SIGMA_ANNULUS_{b}_MJY")


def write_region(out_path, names, ra, dec, records, cell_sigma, cell_offset, cell_map, config, n_jobs, stage=None,
                 shifts=None, cell_row=None, scatter=None, flux_grid=None):
    """Measures every source on its cell's map and writes the product at
    `out_path`. `records` is the band's mosaics (one per band, `pacs_fit.mosaic_record`); `cell_sigma[b]` (n,) float64
    is the source's cell noise (NaN: no cell, not covered) and `cell_map[b]` (n,) int the cell's map, its row in the
    completeness product's MAP_* tables of the band (the band's `records` entries in order);
    `cell_row[b]` (n,) is the source's row in the completeness product (-1: none) and
    `scatter`, `flux_grid` are kept for the caller's interface. Returns the per-band origin counts of the forced
    measurement (the cross-match follows, `cross_match`)."""
    n = names.size
    by_band = {band: [i for i, rec in enumerate(records) if rec.band == band] for band in BANDS}
    pieces = []
    for b, band in enumerate(BANDS):
        sel = np.flatnonzero(np.isfinite(cell_sigma[b]))
        if sel.size == 0:
            continue
        for m in np.unique(cell_map[b][sel]):
            if not 0 <= m < len(by_band[band]):
                raise ValueError("pacs_photometry: completeness cell map %d is not among the %s mosaics" % (m, band))
            on_map = sel[cell_map[b][sel] == m]
            on_map = on_map[np.argsort(dec[on_map], kind="stable")]
            for a, z in pieces_module.slices(on_map.size, n_jobs, PHOTOMETRY_PIECE_MIN):
                pieces.append((b, by_band[band][m], np.sort(on_map[a:z])))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    counts = {}
    with h5py.File(out_path, "w") as f:
        build_module.write_dataset(f, "NAME", names, *REGISTRY[("photometry_pacs_source", "NAME")])
        dsets = {}
        for b, band in enumerate(BANDS):
            for col, pattern in _NAMES.items():
                name = pattern.format(b=band)
                dsets[b, col] = build_module.write_dataset(
                    f, name, None, *REGISTRY[("photometry_pacs_source", name)],
                    shape=(n,), dtype=_DTYPES[col], fillvalue=_FILL[col], chunks=(min(n, 65536),) if n else None)
            counts[band] = {code: 0 for code in (1, 3, 90, 91)}
            counts[band][ORIGIN_NOT_COVERED] = n
        fn_args = dict(config=config, records=records, ra=ra, dec=dec, cell_sigma=np.vstack(cell_sigma),
                       cell_offset=np.vstack(cell_offset),
                       shifts=shifts or {}, flux_grid=flux_grid)
        fn = functools.partial(_measure_tile, **fn_args)
        for band_index, map_index, rows, cols in pieces_module.iterate(fn, pieces, n_jobs, stage=stage, unit="map tiles"):
            band = BANDS[band_index]
            for start, stop, off in _runs(rows):
                width = stop - start
                for col, data in cols.items():
                    dsets[band_index, col][start:stop] = data[off:off + width]
            counts[band][ORIGIN_NOT_COVERED] -= rows.size
            for code in (1, 3, 90, 91):
                counts[band][code] += int(np.sum(cols["ORIGIN"] == code))
    return counts


def _read_cells(config, region, n_source):
    """The completeness stage's per-source cell noise, cell offset, map and cell row, per band, and the
    cells' measured photometric errors: `(cell_sigma, cell_offset, cell_map, shifts, cell_row,
    scatter, flux_grid)`, lists over BANDS (`shifts` a dict, `flux_grid` the one grid in
    mJy). A source with `SOURCE_ROW` -1 has no cell (noise NaN, offset 0)."""
    path = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
    sigma = [np.full(n_source, np.nan) for _ in BANDS]
    offset = [np.zeros(n_source) for _ in BANDS]
    row = [np.full(n_source, -1, dtype=np.int64) for _ in BANDS]
    shifts = {}
    cell_map = [np.full(n_source, -1, dtype=np.int64) for _ in BANDS]
    scatter = [np.zeros((0, 1), np.float32) for _ in BANDS]
    flux_grid = None
    if not os.path.exists(path):
        raise FileNotFoundError("pacs_photometry needs the completeness product first: %s" % path)
    with h5py.File(path, "r") as f:
        source_row = f["SOURCE_ROW"][:]
        for b, band in enumerate(BANDS):
            if "CELL_SIGMA_%s_MJY" % band not in f:
                continue
            grid = 1000.0 * 10.0 ** (-f["MAG_GRID_" + band][:].astype(np.float64) / 2.5) * f["CURVE_" + band].attrs["ZERO_POINT_JY"]
            flux_grid = grid if flux_grid is None else flux_grid
            cell_sigma = f["CELL_SIGMA_%s_MJY" % band][:]
            if cell_sigma.size == 0:
                continue
            rows = source_row[:, b]
            have = rows >= 0
            cell_map_index = f["CELL_MAP_INDEX_%s" % band][:]
            sigma[b][have] = cell_sigma[rows[have]]
            offset[b][have] = f["CELL_OFFSET_%s_MJY" % band][:][rows[have]]
            row[b] = np.where(have, rows, -1).astype(np.int64)
            scatter[b] = f["SCATTER_%s_MJY" % band][:]
            for m, sh in enumerate(f["MAP_SHIFT_%s_ARCSEC" % band][:]):
                shifts[(band, m)] = (float(sh[0]), float(sh[1]))
            cell_map[b] = np.where(have, cell_map_index[np.where(have, rows, 0)], -1).astype(np.int64)
    return sigma, offset, cell_map, shifts, row, scatter, flux_grid


def _build_one_region(config, region, st, box=None):
    from sesnaimpute.catalog import curated
    names, ra, dec = curated.read_positions(config, region)
    names = np.asarray(names).astype("S")
    ra, dec = ra.astype(np.float64), dec.astype(np.float64)
    out_path = product_path(config, "sky/derived", "pacs", "photometry", "source", region=region)
    from sesnaimpute.sky.derived import pacs_mosaic
    records = [pacs_fit.mosaic_record(config, region, b, box) for b in BANDS
               if os.path.exists(os.path.join(pacs_mosaic.product_dir(config, region, b, box), "mosaic.hdf5"))]
    if records:
        cell_sigma, cell_offset, cell_map, shifts, cell_row, scatter, flux_grid = _read_cells(
            config, region, names.size)
    else:
        cell_sigma = [np.full(names.size, np.nan) for _ in BANDS]
        cell_offset = [np.zeros(names.size) for _ in BANDS]
        shifts = {}
        cell_map = [np.full(names.size, -1, dtype=np.int64) for _ in BANDS]
        cell_row = [np.full(names.size, -1, dtype=np.int64) for _ in BANDS]
        scatter = [np.zeros((0, 1), np.float32) for _ in BANDS]
        flux_grid = None
    t0 = time.time()
    counts = write_region(out_path, names, ra, dec, records, cell_sigma, cell_offset, cell_map,
                          config, config.n_jobs, stage=st, shifts=shifts, cell_row=cell_row, scatter=scatter,
                          flux_grid=flux_grid)
    if records:
        matched, chance, offsets = cross_match(out_path, config, region, ra, dec, box)
        with h5py.File(out_path, "r") as f:
            for b in BANDS:
                o = f["ORIGIN_" + b][:]
                counts[b] = {code: int((o == code).sum()) for code in (1, 3, 90, 91)}
                print("  %s: association (%.1f arcsec): ORIGIN 1 %d, ORIGIN 3 %d (blend %d, confusion %d: the band's detection within one "
                      "beam, none associated), ORIGIN 90 %d, ORIGIN 91 %d" % (b, MATCH_RADIUS_ARCSEC, counts[b][1], counts[b][3],
                                                                            counts[b][3] - matched[b]["confused"], matched[b]["confused"],
                                                                            counts[b][90], counts[b][91]), flush=True)
        print("  expected chance coincidences (local SESNA density x pi (%.1f arcsec)^2, summed over the region's PACS "
              "sources): %.1f" % (MATCH_RADIUS_ARCSEC, chance), flush=True)
        print_offsets(config, region, offsets, box)
    wall = time.time() - t0
    numbers = {"sources": int(names.size), "sources_per_s": names.size / max(wall, 1e-9)}
    for band in BANDS:
        for code, label in ((1, "detected"), (3, "bound"), (90, "undetected"), (91, "uncovered")):
            numbers["%s_%s" % (band, label)] = counts[band][code]
    return out_path, numbers


def build(config, regions=None, box=None):
    """Writes each region's `photometry_pacs_source` product (all-91 where no
    PACS map covers the region), on the band mosaics of the region (`box`: a development slice of them). Regions run
    one after another; inside a region the pieces are the sources of a mosaic."""
    names = regions or [r.name for r in regions_module.REGIONS]
    t0 = time.time()
    for name in names:
        with progress_module.Stage("sky.derived.pacs_photometry", name) as st:
            path, numbers = _build_one_region(config, name, st, box)
            st.done(path, **numbers)
    progress_module.summary("sky.derived.pacs_photometry", len(names), t0)


# ---------------------------------------------------------------------------
# acceptance (printed, by function call; nothing is written)

#: HOPS (Fischer et al. 2016, ApJS 224, 5) 70 micron flux of the sources, one row per HOPS target, in mJy, with
#: the target's position, read from the VizieR tables of the HOPS download
HOPS_FILE = "J_ApJS_224_5.vot"
HOPS_MATCH_ARCSEC = 2.0
#: the bright-source comparison of the flux convention (C4)
C4_MIN_F70_MJY = 200.0
#: the bins of HOPS 70 micron flux, mJy, of the detection comparison (C5)
C5_EDGES_MJY = (40.0, 70.0, 100.0, 200.0, 500.0, 1000.0, np.inf)


def _read_hops(config):
    """`(ra deg, dec deg, F70 mJy)` of the HOPS targets with a measured 70 micron flux (flag 1 or 2) and a
    position in the HOPS tables."""
    from astropy.io.votable import parse
    path = os.path.join(config.data_root, "sky", "download", "hops", HOPS_FILE)
    tables = {t.name: t.array for t in parse(path).resources[0].tables}
    pos = {}
    for name in ("table1", "table7", "table8", "table9"):
        a = tables["J/ApJS/224/5/" + name]
        for h, r, d in zip(a["HOPS"], a["RAJ2000"], a["DEJ2000"]):
            pos[int(h)] = (float(r), float(d))
    t2 = tables["J/ApJS/224/5/table2"]
    f70 = np.asarray(t2["F70"].filled(np.nan), float) * 1000.0
    keep = np.isfinite(f70) & np.array([int(h) in pos for h in t2["HOPS"]])
    ra = np.array([pos[int(h)][0] for h in t2["HOPS"][keep]])
    dec = np.array([pos[int(h)][1] for h in t2["HOPS"][keep]])
    return ra, dec, f70[keep]


def _match_hops(config, region):
    """The HOPS targets matched to the region's SESNA sources within HOPS_MATCH_ARCSEC (nearest source):
    `(hops index, source row)` arrays, plus the HOPS arrays."""
    ra, dec, f70 = _read_hops(config)
    s_ra, s_dec = _region_positions(config, region)
    tree = cKDTree(_unit_vectors(s_ra, s_dec))
    chord = 2.0 * np.sin(np.radians(HOPS_MATCH_ARCSEC / 3600.0) / 2.0)
    dist, row = tree.query(_unit_vectors(ra, dec), distance_upper_bound=chord)
    found = np.isfinite(dist)
    return np.flatnonzero(found), row[found], f70


def _region_positions(config, region):
    from sesnaimpute.catalog import curated
    _name, ra, dec = curated.read_positions(config, region)
    return ra.astype(np.float64), dec.astype(np.float64)


def _c4_ratio(config, region, path):
    """Median (our 70 micron flux / HOPS's) over the HOPS targets of at least C4_MIN_F70_MJY that the product
    at `path` detects (ORIGIN 1), and how many. The flux is put on HOPS's convention (no empty-sky offset
    subtracted): FNU plus the product's CELL_OFFSET where it stores one (an earlier product subtracted none)."""
    h, row, f70 = _match_hops(config, region)
    with h5py.File(path, "r") as f:
        flux, origin = f["FNU_P70_MJY"][:][row], f["ORIGIN_P70"][:][row]
        if "CELL_OFFSET_P70_MJY" in f:
            flux = flux + f["CELL_OFFSET_P70_MJY"][:][row]
    use = (f70[h] >= C4_MIN_F70_MJY) & (origin == ORIGIN_DETECTED) & np.isfinite(flux)
    return float(np.median(flux[use] / f70[h][use])) if use.any() else float("nan"), int(use.sum()), int(((f70[h] >= C4_MIN_F70_MJY)).sum())


def acceptance(config, region, reference_product=None, box=None):
    """Prints C4 and C5 of PACS_COMPLETENESS_PLAN.md for the region's finished photometry product (and the
    completeness product it read). C4: the median of our 70 micron flux over HOPS's for HOPS targets of at
    least 200 mJy, next to the same number for `reference_product` (the earlier product's path), the plan's bar
    being agreement within 2 %. C5: the HOPS targets binned by their 70 micron flux: the fraction detected
    (ORIGIN 1) among those the band covers, against the mean of the completeness curve at their SESNA
    source's cell and their flux, with binomial errors. Then the HPPSC checks (`hppsc_checks`). Nothing is written."""
    path = product_path(config, "sky/derived", "pacs", "photometry", "source", region=region)
    ratio, n, n_all = _c4_ratio(config, region, path)
    print("C4 %s: median (ours + CELL_OFFSET) / HOPS at 70 micron (HOPS's convention), HOPS targets >= %g mJy detected by us: %.4f (%d of %d targets)"
          % (region, C4_MIN_F70_MJY, ratio, n, n_all))
    if reference_product is not None:
        old, n_old, _ = _c4_ratio(config, region, reference_product)
        print("   the reference product: %.4f (%d targets); difference %+.2f %% (bar: within 2 %%: %s)"
              % (old, n_old, 100 * (ratio / old - 1), "pass" if abs(ratio / old - 1) <= 0.02 else "FAIL"))
    with h5py.File(path, "r") as f:
        for b in BANDS:
            o3 = f["ORIGIN_" + b][:] == ORIGIN_BOUND
            conf = int(np.sum(o3 & f["CONFUSED_" + b][:]))
            print("   %s ORIGIN 3: %d (blend %d, confusion %d)" % (b, int(o3.sum()), int(o3.sum()) - conf, conf))
    h, row, f70 = _match_hops(config, region)
    with h5py.File(path, "r") as f:
        o70 = f["ORIGIN_P70"][:][row]
    bright = f70[h] >= C4_MIN_F70_MJY
    print("   HOPS targets >= %g mJy by final 70 micron ORIGIN: %s"
          % (C4_MIN_F70_MJY, "  ".join("%d: %d" % (c, int((o70[bright] == c).sum())) for c in (1, 3, 90, 91))))
    comp = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
    with h5py.File(path, "r") as f:
        origin = f["ORIGIN_P70"][:][row]
    with h5py.File(comp, "r") as f:
        cell = f["SOURCE_ROW"][:][row, 0]
        curve = f["CURVE_P70"][:]
        grid = 1000.0 * f["CURVE_P70"].attrs["ZERO_POINT_JY"] * 10.0 ** (-f["MAG_GRID_P70"][:].astype(float) / 2.5)
    flux = f70[h]
    covered = (origin == ORIGIN_DETECTED) | (origin == ORIGIN_UNDETECTED)
    # the curve at each target's cell, interpolated in log flux (the grid runs bright to faint)
    expected = np.full(h.size, np.nan)
    for i in np.flatnonzero(covered & (cell >= 0)):
        expected[i] = np.interp(np.log10(flux[i]), np.log10(grid[::-1]), curve[cell[i], ::-1] / 600.0)
    for lo, hi in zip(C5_EDGES_MJY[:-1], C5_EDGES_MJY[1:]):
        k = covered & np.isfinite(expected) & (flux >= lo) & (flux < hi)
        if not k.any():
            print("C5 %s: HOPS 70 micron %g-%g mJy: none covered" % (region, lo, hi))
            continue
        obs = float(np.mean(origin[k] == ORIGIN_DETECTED))
        pred = float(np.mean(expected[k]))
        err = float(np.sqrt(np.sum(expected[k] * (1 - expected[k]))) / k.sum())
        print("C5 %s: HOPS 70 micron %g-%g mJy: %d targets, detected %.3f +- %.3f, curve predicts %.3f +- %.3f"
              % (region, lo, hi, int(k.sum()), obs, np.sqrt(max(obs * (1 - obs), 1e-9) / k.sum()), pred, err))
    hppsc_checks(config, region, box)


#: HPPSC (Marton et al. 2017; `sky.download.hppsc`) band list per band, and its band name in the observation table
HPPSC_FILES = {"P70": ("ppsc_70.csv", "blue"), "P100": ("ppsc_100.csv", "green"), "P160": ("ppsc_160.csv", "red")}
#: the flux comparison uses common sources with HPPSC's structure-noise S/N at least this
HPPSC_FLUX_SNR = 10.0
#: the bars (planner ruling, owner-approved 2026-10-29): flux ratio within the 5 % calibration; recovery of HPPSC
#: sources at or above our FLUX90 at their position at least 90 % within binomial error
HPPSC_FLUX_BAR = 0.05
HPPSC_RECOVERY_BAR = 0.90
#: the recovery check counts HPPSC's point-like (fitted width at most this many of our beams) entries of HPPSC S/N at
#: least HPPSC_RECOVERY_SNR (owner ruling 2026-10-29: HPPSC also lists cloud knots wider than the beam and entries of
#: low significance that a point-source detection is not meant to find; the full list is printed as information)
HPPSC_POINT_BEAMS = 1.5
HPPSC_RECOVERY_SNR = 10.0


def _read_hppsc(config, band):
    """HPPSC's band list as arrays: ra, dec (deg), flux (mJy), snr (structure-noise S/N), edge flag, fwhm (arcsec: the
    geometric mean of its fitted widths, NaN where it gives none); and the observation table's obsids for the band."""
    import csv
    from sesnaimpute.sky.download.hppsc.build import download_dir
    name, colour = HPPSC_FILES[band]
    cols = {k: [] for k in ("ra", "dec", "flux", "snr", "edgeflag")}
    widths = {"fwhmxfit": [], "fwhmyfit": []}
    with open(os.path.join(download_dir(config), name)) as fh:
        for r in csv.DictReader(fh):
            for k in cols:
                cols[k].append(r[k])
            for k in ("fwhmxfit", "fwhmyfit"):
                widths[k].append(float(r[k]) if r[k] not in ("", None) else np.nan)
    out = {k: np.asarray(v, float) for k, v in cols.items() if k != "edgeflag"}
    out["edge"] = np.asarray(cols["edgeflag"]) == "t"
    out["fwhm"] = np.sqrt(np.asarray(widths["fwhmxfit"]) * np.asarray(widths["fwhmyfit"]))
    with open(os.path.join(download_dir(config), "ppsc_obstbl.csv")) as fh:
        out["obsids"] = {int(r["obsid"]) for r in csv.DictReader(fh) if r["band"] == colour}
    return out


def hppsc_checks(config, region, mosaic_box=None):
    """The external check of the region's PACS source list against HPPSC, per band (planner ruling, owner-approved
    2026-10-29; HPPSC enters no product): (1) how many of our maps HPPSC was built from (its observation table, the
    paper's Table 8); (2) the offsets of common sources, HPPSC to our nearest detection in the band within one beam,
    at that band's own detection position (RA_<b>_DEG, on the same map's pointing as HPPSC's), as information;
    (3) the median flux ratio ours / HPPSC over common sources (within one beam, HPPSC S/N >= HPPSC_FLUX_SNR), ours on
    HPPSC's convention (FNU plus the cell offset: HPPSC subtracts only its annulus sky), bar within HPPSC_FLUX_BAR;
    (4) of the HPPSC sources inside our footprint with flux at least our FLUX90 at their position (the completeness
    product read as a function of position, `catalog.completeness._PacsBand`), fitted width at most HPPSC_POINT_BEAMS
    beams and HPPSC S/N at least HPPSC_RECOVERY_SNR, the fraction our detection finds in the band within one beam, bar
    HPPSC_RECOVERY_BAR within binomial error (the same over all of them, and over the point-like ones, as information).
    `mosaic_box`: a development slice's mosaics (default the whole region's). Common sources are matched within one beam,
    not the association's 2 arcsec, so that the checks measure detection and not the two catalogues' astrometry
    (HPPSC's is 1.5-1.7 arcsec, and the 100 micron maps are separate observations; planner ruling, owner-approved
    2026-10-29). HPPSC states no reliability of its own: a false HPPSC source counts against (4). Nothing is written."""
    from sesnaimpute.catalog.completeness import _PacsBand
    comp = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
    src = product_path(config, "sky/derived", "pacs", "sources", "region", region=region)
    with h5py.File(src, "r") as f:
        mine = {b: dict(det=f["DETECTED_" + b][:], fnu=f["FNU_%s_MJY" % b][:], off=f["CELL_OFFSET_%s_MJY" % b][:],
                        ra=f["RA_%s_DEG" % b][:], dec=f["DEC_%s_DEG" % b][:]) for b in BANDS}
    with h5py.File(comp, "r") as fc:
        for b in BANDS:
            if "CELL_ID_" + b not in fc or fc["CELL_ID_" + b].shape[0] == 0:
                print("HPPSC %s %s: the band has no cells" % (region, b))
                continue
            h = _read_hppsc(config, b)
            ours = {int(o) for m in fc["MAP_OBSID_" + b][:] for o in m.decode().split(",") if o}
            print("HPPSC %s %s: coverage: %d of the %d observations of our mosaic are among HPPSC's observations (its Table 8)"
                  % (region, b, len(ours & h["obsids"]), len(ours)))
            cra, cde = fc["CELL_RA_" + b][:], fc["CELL_DEC_" + b][:]
            box = ((h["ra"] > cra.min() - 0.1) & (h["ra"] < cra.max() + 0.1) & (h["dec"] > cde.min() - 0.1)
                   & (h["dec"] < cde.max() + 0.1))
            hra, hde, hflux, hsnr = h["ra"][box], h["dec"][box], h["flux"][box], h["snr"][box]
            row = _PacsBand(fc, b).rows(hra, hde)
            inside = row >= 0
            flux90 = np.where(inside, fc["FLUX90_%s_MJY" % b][:][np.maximum(row, 0)], np.nan)
            det = np.flatnonzero(mine[b]["det"])
            # each band's own detection position (the same map as HPPSC's), not the merged one
            pra, pde = mine[b]["ra"], mine[b]["dec"]
            pu = _unit_vectors(np.nan_to_num(pra), np.nan_to_num(pde))
            if det.size == 0 or not inside.any():
                print("HPPSC %s %s: %d HPPSC sources inside our footprint, %d detections of ours: no comparison"
                      % (region, b, int(inside.sum()), det.size))
                continue
            tree = cKDTree(pu[det])
            hu = _unit_vectors(hra, hde)
            psf = pacs_fit.mosaic_psf(config, region, b, mosaic_box)
            beam = psf.beam_at_sky(hra, hde)                      # the band's local beam at each HPPSC source
            dist, j = tree.query(hu, distance_upper_bound=2.0 * np.sin(np.radians(psf.sub_beams.max() / 3600.0) / 2.0))
            near = np.isfinite(dist) & inside & (np.nan_to_num(dist, posinf=9.0) < 2.0 * np.sin(np.radians(beam / 3600.0) / 2.0))
            k = det[np.minimum(j, det.size - 1)]
            sep = np.where(near, np.degrees(2.0 * np.arcsin(np.clip(np.nan_to_num(dist, posinf=0.0) / 2.0, 0, 1))) * 3600.0, np.nan)
            cosd = np.cos(np.radians(hde))
            east = (pra[k] - hra) * cosd * 3600.0
            north = (pde[k] - hde) * 3600.0
            if near.any():
                print("HPPSC %s %s: offsets ours - HPPSC of %d common sources (within one local beam, median %.1f arcsec): separation "
                      "median %.2f, 68th %.2f, 90th %.2f arcsec; median east %+.2f north %+.2f; within %.1f arcsec %.3f"
                      % (region, b, int(near.sum()), np.median(beam[near]), np.median(sep[near]), np.percentile(sep[near], 68),
                         np.percentile(sep[near], 90), np.median(east[near]), np.median(north[near]), MATCH_RADIUS_ARCSEC,
                         float(np.mean(sep[near] <= MATCH_RADIUS_ARCSEC))))
            common = near                   # within one beam (planner ruling, owner-approved 2026-10-29)
            ours_flux = mine[b]["fnu"][k] + np.nan_to_num(mine[b]["off"][k])
            use = common & (hsnr >= HPPSC_FLUX_SNR) & np.isfinite(ours_flux) & (hflux > 0)
            if use.any():
                ratio = float(np.median(ours_flux[use] / hflux[use]))
                print("HPPSC %s %s: flux ours (FNU + cell offset) / HPPSC, median over %d common sources with HPPSC S/N >= "
                      "%g: %.3f (bar within %.0f %%: %s)" % (region, b, int(use.sum()), HPPSC_FLUX_SNR, ratio,
                                                           100 * HPPSC_FLUX_BAR,
                                                           "pass" if abs(ratio - 1.0) <= HPPSC_FLUX_BAR else "FAIL"))
            else:
                print("HPPSC %s %s: no common sources with HPPSC S/N >= %g" % (region, b, HPPSC_FLUX_SNR))
            above = inside & np.isfinite(flux90) & (hflux >= flux90)
            point = h["fwhm"][box] <= HPPSC_POINT_BEAMS * beam
            for label, pool, ruled in (("all", above, False), ("point-like", above & point, False),
                                       ("point-like with HPPSC S/N >= %g" % HPPSC_RECOVERY_SNR,
                                        above & point & (hsnr >= HPPSC_RECOVERY_SNR), True)):
                if not pool.any():
                    print("HPPSC %s %s: recovery (%s): no HPPSC source inside our footprint at or above our FLUX90"
                          % (region, b, label))
                    continue
                n_pool = int(pool.sum())
                frac = float(np.mean(common[pool]))
                err = np.sqrt(HPPSC_RECOVERY_BAR * (1 - HPPSC_RECOVERY_BAR) / n_pool)
                verdict = ("bar >= %.2f: %s" % (HPPSC_RECOVERY_BAR, "pass" if frac + err >= HPPSC_RECOVERY_BAR else "FAIL")
                           if ruled else "information")
                print("HPPSC %s %s: recovery (%s) of the %d HPPSC sources inside our footprint at or above our FLUX90 "
                      "there: %.3f (binomial error at the bar %.3f; %s); HPPSC edge-flagged among them %d"
                      % (region, b, label, n_pool, frac, err, verdict, int(np.sum(h["edge"][box][pool]))))


if __name__ == "__main__":
    run(build)
