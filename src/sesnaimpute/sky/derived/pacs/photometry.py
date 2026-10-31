"""PACS fluxes of the region: the cross-band merge of the bands' detections into PACS sources, the forced photometry at
every SESNA position, and the one SESNA-PACS cross-match (ORIGIN 1 detected, 3 censored bound, 90 undetected, 91 not
covered).

Depends on: grid, psf, mosaic, detect, inject. Replaces: pacs_completeness.merge_sources/_measure_at/_pair_scatter/_unit/
_chord/_sep_arcsec, pacs_photometry.singleton_pass/calibrated_sigma/_measure_tile/cross_match/print_offsets/_runs/
_build_one_region/build/_read_cells/_unit_vectors/_sep_arcsec/_region_positions. The aperture recipe itself is
`mosaic.Mosaic.aperture` (written once there). The spherical helpers are written once here. Unused today and not carried:
write_region's flux_grid/scatter/cell_row parameters, singleton_pass's fifth return value.

The tables are dicts of numpy arrays keyed by the NAME OF THE PRODUCT DATASET (products.py writes them as they are):
`Sources.columns` has RA_DEG, DEC_DEG, POSITION_BAND and, per band B, DETECTED_B, FNU_B_MJY, SIGMA_B_MJY,
CELL_OFFSET_B_MJY, SNR_B, MAP_INDEX_B, EXTENDED_B, RA_B_DEG, DEC_B_DEG, plus MERGE_RADIUS_<A>_<B>_ARCSEC (0-d, NaN = half
the redder band's local beam) for each band pair both bands of which have a mosaic. `ForcedPhotometry.columns` has NAME and,
per band B, FNU_B_MJY, SIGMA_B_MJY, ORIGIN_B, LOCAL_RMS_B_MJY, CELL_OFFSET_B_MJY, SIGMA_ANNULUS_B_MJY, N_BLEND_B,
PACS_SOURCE_B, MATCH_SEPARATION_B_ARCSEC, EXTENDED_B, CONFUSED_B at the dtypes the product stores; `cross_match` adds the
private keys _CHANCE (0-d, the expected chance coincidences; present only when the region has PACS sources) and
_PAIRS_<B> ((n, 4): east, north in arcsec and the PACS position, per position band) for `registration_offsets`.

What this module reads of its inputs (the contract's BandCompleteness is the carrier):
  BandCompleteness.detection.peaks (Peaks: ra, dec, fnu, snr, row), .cells (CellStats: ids, offset, sigma), .src_row,
  .shift (east, north, n_pairs); BandCompleteness.scatter. Mosaic: .psf (MosaicPSF: beam_at_sky, sub_beams), .cells
  (CellGrid), .xy, .aperture, .fitted_fwhm, .beam_at, .inputs (the input maps: record, wcs). An empty band (no mosaic) is
  simply absent from `mosaics`.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from sesnaimpute import pieces as pieces_module
from sesnaimpute.definitions import PACS_BANDS as BANDS
from sesnaimpute.sky.derived.pacs import detect, grid
from sesnaimpute.sky.derived.pacs.inject import BandCompleteness
from sesnaimpute.sky.derived.pacs.mosaic import Mosaic

#: the pipeline's origin codes for a flux (`ORIGIN_<B>` in the products)
ORIGIN_DETECTED, ORIGIN_BOUND, ORIGIN_UNDETECTED, ORIGIN_NOT_COVERED = 1, 3, 90, 91

#: least sources in one photometry piece (a piece costs a few ms of overhead; a source about 3 ms)
PHOTOMETRY_PIECE_MIN = 2000
#: the SESNA association of the region's PACS sources (planner ruling, owner-approved 2026-10-29: Rob Gutermuth's
#: band-merging recipe as eHOPS applies it, Pokhrel et al. 2023 Sect. III.2 and Appendix C): nearest neighbour within a
#: fixed radius in every band, set by the two catalogues' astrometric accuracy (Gutermuth et al. 2009 Sect. 2.3: 1 arcsec
#: IRAC/2MASS, 1.3 arcsec MIPS 24 um; eHOPS: 2 arcsec for PACS on the Herschel header astrometry, no registration)
MATCH_RADIUS_ARCSEC = 2.0
#: the SESNA density behind the expected chance coincidences is counted within this radius of each PACS source
CHANCE_DENSITY_ARCSEC = 60.0
#: an input map's median PACS - SESNA offset is reported where it holds at least this many associated pairs
MIN_OFFSET_PAIRS = 5
#: a detection is EXTENDED where its fitted width exceeds this many local beams
EXTENDED_BEAM_FACTOR = 1.5
#: the cross-band merge (planner ruling, owner-approved 2026-10-28): detections of two bands closer than MERGE_SIGMAS x the
#: measured per-source position scatter between those bands (bright common sources, S/N >= MERGE_MIN_SNR, paired within half
#: the redder beam) are one PACS source; fewer than MERGE_MIN_PAIRS pairs: half the redder beam
MERGE_SIGMAS = 3.0
MERGE_MIN_SNR = 20.0
MERGE_MIN_PAIRS = 5
#: the three band pairs of the merge, in the order they are tried
_PAIRS = (("P70", "P100"), ("P70", "P160"), ("P100", "P160"))

#: dtype, fill and dataset name of each forced-photometry column (the product's)
PHOTOMETRY_DTYPES = dict(CONFUSED=bool, FNU=np.float64, SIGMA=np.float64, CELL_OFFSET=np.float64, ORIGIN=np.int8,
                         LOCAL_RMS=np.float32, N_BLEND=np.int16, PACS_SOURCE=np.int32, MATCH_SEPARATION=np.float32,
                         EXTENDED=bool, SIGMA_ANNULUS=np.float32)
PHOTOMETRY_FILL = dict(CONFUSED=False, FNU=np.nan, SIGMA=np.nan, CELL_OFFSET=np.nan, ORIGIN=ORIGIN_NOT_COVERED,
                       LOCAL_RMS=np.nan, N_BLEND=0, PACS_SOURCE=-1, MATCH_SEPARATION=np.nan, EXTENDED=False,
                       SIGMA_ANNULUS=np.nan)
PHOTOMETRY_NAMES = dict(CONFUSED="CONFUSED_{b}", FNU="FNU_{b}_MJY", SIGMA="SIGMA_{b}_MJY", CELL_OFFSET="CELL_OFFSET_{b}_MJY",
                        ORIGIN="ORIGIN_{b}", LOCAL_RMS="LOCAL_RMS_{b}_MJY", N_BLEND="N_BLEND_{b}",
                        PACS_SOURCE="PACS_SOURCE_{b}", MATCH_SEPARATION="MATCH_SEPARATION_{b}_ARCSEC",
                        EXTENDED="EXTENDED_{b}", SIGMA_ANNULUS="SIGMA_ANNULUS_{b}_MJY")


# ---------------------------------------------------------------------------
# spherical helpers (written once)

def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """(n, 3) unit vectors of ICRS positions (written once; was pacs_photometry._unit_vectors and
    pacs_completeness._unit)."""
    ra, dec = np.radians(np.asarray(ra_deg, float)), np.radians(np.asarray(dec_deg, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def sep_arcsec(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angular separation (arcsec) of unit vectors (was written twice)."""
    return np.degrees(2.0 * np.arcsin(np.clip(np.linalg.norm(u - v, axis=-1) / 2.0, 0.0, 1.0))) * 3600.0


def chord(arcsec: float) -> float:
    """The chord length of an angle in arcsec (kd-tree radius on unit vectors)."""
    return 2.0 * np.sin(np.radians(np.asarray(arcsec, float) / 3600.0) / 2.0)


def calibrated_sigma(scatter: np.ndarray, flux_grid_mjy: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """A detection's SIGMA: the row's SCATTER curve read at its flux (`pacs_photometry.calibrated_sigma`); x CELL_SIGMA
    by the caller. NaN only where SCATTER is NaN (the legacy short-cell rule).

    `scatter` (n, n_plane), `flux_grid_mjy` (n_plane,) mJy, bright to faint. Log-linear in flux between the two planes
    that bracket the flux, clamped to the grid ends; where either bracketing plane holds NaN the value of the nearest
    plane of the same row that is finite; NaN for a row with none."""
    asc = np.log10(flux_grid_mjy[::-1])
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
# the PACS sources: the bands' detections merged across bands

@dataclass
class Sources:
    """The region's PACS sources (the sources product's columns, keyed by dataset name; see the module docstring):
    position band, ra, dec, per band fnu, sigma, snr, detected, extended, map position; the merge radii."""
    columns: Dict[str, np.ndarray]


def _pair_scatter(pa, pb, mosaic_red: Mosaic) -> Tuple[float, int]:
    """The per-source position scatter (2-D rms, robust, arcsec) between two bands' bright detections (S/N >=
    MERGE_MIN_SNR) paired within half the redder band's local beam at the first band's detection, and the number of
    pairs. `pa`, `pb`: the bands' Peaks."""
    a = pa.snr >= MERGE_MIN_SNR
    b = pb.snr >= MERGE_MIN_SNR
    if not a.any() or not b.any():
        return np.nan, 0
    ua, ub = unit_vectors(pa.ra[a], pa.dec[a]), unit_vectors(pb.ra[b], pb.dec[b])
    psf = mosaic_red.psf
    d, j = cKDTree(ub).query(ua, distance_upper_bound=chord(psf.sub_beams.max() / 2.0))
    ok = np.isfinite(d)
    ok[ok] = sep_arcsec(ua[ok], ub[j[ok]]) <= psf.beam_at_sky(pa.ra[a][ok], pa.dec[a][ok]) / 2.0
    if not ok.any():
        return np.nan, 0
    sep = sep_arcsec(ua[ok], ub[j[ok]])
    return float(np.sqrt(np.median(sep ** 2) / np.log(2.0))), int(ok.sum())


def _filtered(m, x, y, offset_cell):
    """The detection estimator at map pixels (detect.filtered)."""
    return detect.filtered(m, x, y, offset_cell)


def _measure_at(piece, mosaics: Dict[str, Mosaic]):
    """The catalogue's flux of PACS sources at given sky positions on one band's mosaic (`piece = (band, ra, dec, shift,
    offset per grid cell)`): the aperture estimator minus the cell offset, the cell id, and whether the fitted width
    exceeds EXTENDED_BEAM_FACTOR beams. Returns (band, fnu, cell id, extended)."""
    band, ra, dec, shift, offset_cell = piece
    m = mosaics[band]
    x, y = m.xy(*grid.to_map_frame(ra, dec, shift))
    fnu = _filtered(m, x, y, offset_cell)
    ext = m.fitted_fwhm(x, y) > EXTENDED_BEAM_FACTOR * m.beam_at(x, y)
    return band, fnu, m.cells.cell_ids(x, y), ext


def merge_sources(config, bands: Dict[str, BandCompleteness], mosaics: Dict[str, Mosaic], flux_grid_mjy: np.ndarray,
                  n_jobs: Optional[int] = None) -> Sources:
    """The bands' blind detections merged across bands (planner ruling 2026-10-28): pairs within the pair's merge radius
    are one source (each band once, nearest first, brightest first); position = the sharpest detecting band's; each
    detecting band's flux = Mosaic.aperture at its own detection position; SIGMA = CELL_SIGMA x calibrated_sigma.
    Identity: `pacs_completeness.merge_sources` (pacs_completeness.py:1844) bit-identical.

    `bands` has an entry per band (an empty one for a band with no map); `mosaics` only the bands that have one."""
    pk = {b: bands[b].detection.peaks for b in BANDS}
    radius = {}                         # arcsec; None: half the redder band's local beam at the detection
    for a, b in _PAIRS:
        if a not in mosaics or b not in mosaics:
            continue
        rms, n = _pair_scatter(pk[a], pk[b], mosaics[b])
        radius[(a, b)] = MERGE_SIGMAS * rms if n >= MERGE_MIN_PAIRS else None
    pos_band, members, pos_u = [], [], []
    for bi, b in enumerate(BANDS):
        p = pk[b]
        if p.fnu.size == 0:
            continue
        u = unit_vectors(p.ra, p.dec)
        order = np.argsort(-p.snr)
        tree = cKDTree(np.array(pos_u)) if pos_u else None
        half_beam = 0.5 * mosaics[b].psf.beam_at_sky(p.ra, p.dec)        # the default radius, at each detection of the band
        for i in order:
            best = None
            if tree is not None:
                reach = {a: (half_beam[i] if radius[(a, b)] is None else radius[(a, b)]) for a in BANDS[:bi] if (a, b) in radius}
                rmax = max(reach.values(), default=0.0)
                for k in tree.query_ball_point(u[i], chord(rmax)):
                    if b in members[k] or (BANDS[pos_band[k]], b) not in radius:
                        continue
                    sep = sep_arcsec(u[i], pos_u[k])
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
    pra = np.array([pk[BANDS[pb]].ra[src[BANDS[pb]][k]] for k, pb in enumerate(pos_band)]) if n else np.zeros(0)
    pde = np.array([pk[BANDS[pb]].dec[src[BANDS[pb]][k]] for k, pb in enumerate(pos_band)]) if n else np.zeros(0)
    # each detecting band's flux at that band's own detection position (as HPPSC measures it; owner-approved 2026-10-29)
    pieces, want = [], {}
    for b in BANDS:
        det = np.flatnonzero(src[b] >= 0)
        if det.size == 0:
            continue
        cells = bands[b].detection.cells
        oc = np.full(mosaics[b].cells.n_cells, np.nan)
        oc[cells.ids] = cells.offset
        own = src[b][det]
        shift = (float(bands[b].detection.shift[0]), float(bands[b].detection.shift[1]))
        pieces.append((b, pk[b].ra[own], pk[b].dec[own], shift, oc))
        want[b] = det
    measured = pieces_module.run(functools.partial(_measure_at, mosaics=mosaics), pieces, n_jobs or 1) if pieces else []
    columns: Dict[str, np.ndarray] = dict(RA_DEG=np.asarray(pra, np.float64), DEC_DEG=np.asarray(pde, np.float64),
                                          POSITION_BAND=pos_band)
    for (a, b), r in radius.items():
        columns["MERGE_RADIUS_%s_%s_ARCSEC" % (a, b)] = np.array(np.nan if r is None else float(r))
    fnu = {b: np.full(n, np.nan) for b in BANDS}
    sigma = {b: np.full(n, np.nan) for b in BANDS}
    offset = {b: np.full(n, np.nan) for b in BANDS}
    extended = {b: np.zeros(n, bool) for b in BANDS}
    for b, f, cid, ext in measured:
        k = want[b]
        ids = bands[b].detection.cells.ids.astype(np.int64)
        row = np.clip(np.searchsorted(ids, cid), 0, max(ids.size - 1, 0))
        hit = (cid >= 0) & (ids[row] == cid)
        fnu[b][k] = f
        extended[b][k] = ext
        offset[b][k[hit]] = bands[b].detection.cells.offset[row[hit]]
        if hit.any():
            sigma[b][k[hit]] = calibrated_sigma(bands[b].scatter[row[hit]], flux_grid_mjy, f[hit])
    for b in BANDS:
        s = src[b] >= 0
        at = np.maximum(src[b], 0)
        columns["DETECTED_" + b] = s
        columns["FNU_%s_MJY" % b] = fnu[b]
        columns["SIGMA_%s_MJY" % b] = sigma[b]
        columns["CELL_OFFSET_%s_MJY" % b] = offset[b]
        columns["SNR_" + b] = np.asarray(np.where(s, pk[b].snr[at] if pk[b].snr.size else np.nan, np.nan), np.float32)
        columns["MAP_INDEX_" + b] = np.where(s, 0, -1).astype(np.int32)       # one mosaic per band
        columns["EXTENDED_" + b] = extended[b]
        columns["RA_%s_DEG" % b] = np.asarray(np.where(s, pk[b].ra[at] if pk[b].ra.size else np.nan, np.nan), np.float64)
        columns["DEC_%s_DEG" % b] = np.asarray(np.where(s, pk[b].dec[at] if pk[b].dec.size else np.nan, np.nan), np.float64)
    return Sources(columns)


# ---------------------------------------------------------------------------
# forced photometry at the SESNA positions

@dataclass
class ForcedPhotometry:
    """Per SESNA source and band: FNU, SIGMA (cell), OFFSET, CELL row, map position, covered (the product's columns,
    keyed by dataset name; see the module docstring)."""
    columns: Dict[str, np.ndarray]


def _blank_photometry(names: np.ndarray) -> ForcedPhotometry:
    n = names.size
    cols: Dict[str, np.ndarray] = {"NAME": names}
    for b in BANDS:
        for key, pattern in PHOTOMETRY_NAMES.items():
            cols[pattern.format(b=b)] = np.full(n, PHOTOMETRY_FILL[key], dtype=PHOTOMETRY_DTYPES[key])
    return ForcedPhotometry(cols)


def _measure_piece(rows, mosaic_b: Mosaic, x, y, sigma_cell, offset):
    """All of one piece of SESNA sources on a band's mosaic (`rows` the source rows, ascending): the forced aperture flux
    at each position (the value an ORIGIN 90 source carries) and its cell's noise; ORIGIN 90 where covered, 91 where not.
    The cross-match sets ORIGIN 1 and 3 after. Returns (rows, columns)."""
    fit = mosaic_b.aperture(x[rows], y[rows])
    sc, off = sigma_cell[rows], offset[rows]
    ok = fit.covered & np.isfinite(sc) & (sc > 0)
    n = rows.size
    flux = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    origin = np.full(n, ORIGIN_NOT_COVERED, dtype=np.int8)
    subtracted = np.full(n, np.nan)
    flux[ok] = fit.flux[ok] - off[ok]
    sigma[ok] = sc[ok]
    origin[ok] = ORIGIN_UNDETECTED
    subtracted[ok] = off[ok]
    local = sc.astype(np.float32)
    local[origin == ORIGIN_NOT_COVERED] = np.nan
    return rows, dict(FNU=flux, SIGMA=sigma, ORIGIN=origin, LOCAL_RMS=local, CELL_OFFSET=subtracted,
                      SIGMA_ANNULUS=np.asarray(fit.annulus_sigma, np.float32))


def forced_photometry(config, region: str, mosaics: Dict[str, Mosaic], bands: Dict[str, BandCompleteness],
                      ra: np.ndarray, dec: np.ndarray, n_jobs: Optional[int] = None) -> ForcedPhotometry:
    """Forced aperture flux at every SESNA position (registration shift applied, FROZEN 9), per tile of positions
    (pieces of >= PHOTOMETRY_PIECE_MIN sources), SIGMA = the cell's CELL_SIGMA. Identity: `pacs_photometry.write_region`
    + `_measure_tile` (pacs_photometry.py:143, 322) bit-identical.

    A source with no cell in a band (SOURCE_ROW -1) is ORIGIN 91 there. The pieces are the sources of the band's cells in
    the order of the 256-pixel tile their position falls in, cut into runs of >= PHOTOMETRY_PIECE_MIN (compact windows; the
    aperture of a position does not depend on the other positions of its piece). `ra`, `dec` are the region's curated
    positions, row for row (the NAME column is read from the same delivery)."""
    from sesnaimpute.catalog import curated
    names = np.asarray(curated.read_positions(config, region)[0]).astype("S")
    ra, dec = np.asarray(ra, np.float64), np.asarray(dec, np.float64)
    if names.size != ra.size:
        raise ValueError("forced_photometry: %d positions for %d curated names" % (ra.size, names.size))
    out = _blank_photometry(names)
    for b in BANDS:
        if b not in mosaics or b not in bands:
            continue
        d = bands[b].detection
        rows = np.asarray(d.src_row)
        have = rows >= 0
        sigma_cell = np.full(ra.size, np.nan)
        offset = np.zeros(ra.size)
        sigma_cell[have] = d.cells.sigma[rows[have]]
        offset[have] = d.cells.offset[rows[have]]
        sel = np.flatnonzero(np.isfinite(sigma_cell))
        if sel.size == 0:
            continue
        shift = (float(d.shift[0]), float(d.shift[1]))
        m = mosaics[b]
        x = np.full(ra.size, np.nan)
        y = np.full(ra.size, np.nan)
        x[sel], y[sel] = m.xy(*grid.to_map_frame(ra[sel], dec[sel], shift))
        tile = 256
        key = (np.floor(np.nan_to_num(y[sel]) / tile).astype(np.int64) * (1 << 20)
               + np.floor(np.nan_to_num(x[sel]) / tile).astype(np.int64))
        on_map = sel[np.argsort(key, kind="stable")]
        pieces = [np.sort(on_map[a:z]) for a, z in pieces_module.slices(on_map.size, n_jobs or 1, PHOTOMETRY_PIECE_MIN)]
        fn = functools.partial(_measure_piece, mosaic_b=m, x=x, y=y, sigma_cell=sigma_cell, offset=offset)
        for rws, cols in pieces_module.iterate(fn, pieces, n_jobs or 1):
            for col, data in cols.items():
                out.columns[PHOTOMETRY_NAMES[col].format(b=b)][rws] = data
    return out


# ---------------------------------------------------------------------------
# the one cross-match

def cross_match(photometry: ForcedPhotometry, sources: Sources, ra: np.ndarray, dec: np.ndarray,
                mosaics: Dict[str, Mosaic]) -> Tuple[ForcedPhotometry, dict]:
    """The one SESNA-PACS association: nearest neighbour within MATCH_RADIUS_ARCSEC per band; a sole match takes the
    source's FNU/SIGMA (ORIGIN 1), a SESNA source within one beam of a detection it does not own is ORIGIN 3 (censored
    bound), else 90 (covered) or 91; CONFUSED, EXTENDED, PACS_SOURCE, MATCH_SEPARATION columns; and the expected chance
    coincidences (CHANCE_DENSITY_ARCSEC). Returns the updated table and the counts. Identity: `pacs_photometry.
    cross_match` (pacs_photometry.py:178).

    If a PACS source has this SESNA source alone within the radius: ORIGIN 1 in every band that detects it, FNU/SIGMA the
    PACS source's. If two or more SESNA sources (a blend): ORIGIN 3 in every detecting band, the bound that band's flux and
    SIGMA that detection's own error. A band that does not detect the PACS source keeps the forced measurement (ORIGIN 90
    or 91), unless a detection of that band lies within one local beam of the SESNA source: then ORIGIN 3 (CONFUSED true)
    with the nearest such detection's flux as the bound. The table is updated in place. The counts: {band: {m1, m2,
    confused}, "chance": expected chance coincidences, "origin": {band: {code: n}}}. Added to the table: _CHANCE and
    _PAIRS_<band> (see the module docstring)."""
    c = sources.columns
    ra, dec = np.asarray(ra, np.float64), np.asarray(dec, np.float64)
    n = c["RA_DEG"].shape[0]
    pra, pde, pb = c["RA_DEG"], c["DEC_DEG"], c["POSITION_BAND"].astype(int)
    cols = {b: dict(det=c["DETECTED_" + b], fnu=c["FNU_%s_MJY" % b], sig=c["SIGMA_%s_MJY" % b], off=c["CELL_OFFSET_%s_MJY" % b],
                    ext=c["EXTENDED_" + b], ra=c["RA_%s_DEG" % b], dec=c["DEC_%s_DEG" % b]) for b in BANDS}
    counts: dict = {b: dict(m1=0, m2=0, confused=0) for b in BANDS}
    data = {b: {key: photometry.columns[PHOTOMETRY_NAMES[key].format(b=b)]
                for key in ("FNU", "SIGMA", "ORIGIN", "CELL_OFFSET", "N_BLEND", "PACS_SOURCE", "MATCH_SEPARATION", "EXTENDED",
                            "CONFUSED")} for b in BANDS}
    if n == 0 or ra.size == 0:
        counts["chance"] = 0.0
        counts["origin"] = _origin_counts(photometry)
        return photometry, counts
    su = unit_vectors(ra, dec)
    pu = unit_vectors(pra, pde)
    radius = chord(MATCH_RADIUS_ARCSEC)
    stree, ptree = cKDTree(su), cKDTree(pu)
    claim = stree.query_ball_point(pu, radius)                      # SESNA sources within the radius, per PACS source
    n_local = stree.query_ball_point(pu, chord(CHANCE_DENSITY_ARCSEC), return_length=True)
    chance = float(np.sum(n_local / (np.pi * CHANCE_DENSITY_ARCSEC ** 2)) * np.pi * MATCH_RADIUS_ARCSEC ** 2)
    dist, near = ptree.query(su, distance_upper_bound=radius)       # each SESNA source's nearest PACS source
    pairs: Dict[int, list] = {}
    for i in np.flatnonzero(np.isfinite(dist)):
        k = int(near[i])
        sole = len(claim[k]) == 1
        sep = float(sep_arcsec(su[i], pu[k]))
        if sole:
            cosd = np.cos(np.radians(dec[i]))
            pairs.setdefault(int(pb[k]), []).append(
                ((pra[k] - ra[i]) * cosd * 3600.0, (pde[k] - dec[i]) * 3600.0, pra[k], pde[k]))
        for b in BANDS:
            cb, d = cols[b], data[b]
            if not cb["det"][k]:
                continue
            d["FNU"][i], d["CELL_OFFSET"][i] = cb["fnu"][k], cb["off"][k]
            d["PACS_SOURCE"][i], d["MATCH_SEPARATION"][i], d["N_BLEND"][i] = k, sep, len(claim[k])
            if sole:
                d["ORIGIN"][i], d["SIGMA"][i], d["EXTENDED"][i] = ORIGIN_DETECTED, cb["sig"][k], cb["ext"][k]
                counts[b]["m1"] += 1
            else:
                d["ORIGIN"][i], d["SIGMA"][i] = ORIGIN_BOUND, cb["sig"][k]
                counts[b]["m2"] += 1
    # the per-band confusion flag (owner-approved 2026-10-29): a covered source with no detection of the band associated, but
    # the band's detection (at its own position) within one local beam of it, cannot be called undetected: ORIGIN 3,
    # CONFUSED, the nearest such detection's flux the bound
    for b in BANDS:
        cb, d = cols[b], data[b]
        det = np.flatnonzero(cb["det"])
        if det.size == 0:
            continue
        psf = mosaics[b].psf
        bu = unit_vectors(cb["ra"][det], cb["dec"][det])
        bd, bj = cKDTree(bu).query(su, distance_upper_bound=chord(psf.sub_beams.max()))
        within = np.isfinite(bd)
        within[within] = bd[within] < chord(psf.beam_at_sky(ra[within], dec[within]))
        for i in np.flatnonzero(within & (d["ORIGIN"] == ORIGIN_UNDETECTED)):
            k = int(det[bj[i]])
            d["FNU"][i], d["CELL_OFFSET"][i], d["SIGMA"][i] = cb["fnu"][k], cb["off"][k], cb["sig"][k]
            d["ORIGIN"][i], d["PACS_SOURCE"][i], d["N_BLEND"][i] = ORIGIN_BOUND, k, len(claim[k])
            d["CONFUSED"][i] = True
            d["MATCH_SEPARATION"][i] = float(sep_arcsec(su[i], bu[bj[i]]))
            counts[b]["confused"] += 1
    photometry.columns["_CHANCE"] = np.array(chance)
    for bi, b in enumerate(BANDS):
        photometry.columns["_PAIRS_" + b] = np.array(pairs[bi], float).reshape(-1, 4) if bi in pairs else np.zeros((0, 4))
    counts["chance"] = chance
    counts["origin"] = _origin_counts(photometry)
    return photometry, counts


def _origin_counts(photometry: ForcedPhotometry) -> dict:
    return {b: {code: int((photometry.columns["ORIGIN_" + b] == code).sum()) for code in (1, 3, 90, 91)} for b in BANDS}


# ---------------------------------------------------------------------------
# registration offsets

def _input_footprints(m) -> List[tuple]:
    """(name, WCS, (ny, nx)) of each input map of a mosaic."""
    return [(os.path.basename(i.record.path), i.wcs, tuple(i.record.shape)) for i in m.inputs]


def registration_offsets(photometry: ForcedPhotometry, mosaics: Dict[str, Mosaic]) -> dict:
    """Per input map, the median PACS - SESNA offset of its associated pairs (>= MIN_OFFSET_PAIRS)
    (`pacs_photometry.print_offsets`, now returned for checks.py). The pairs are those of `cross_match` (sole matches,
    ORIGIN 1) whose PACS position falls inside the input map's footprint, grouped by the band that gave the PACS position.

    Returns {position band: [(input map name, pairs, median east, median north), ...]} (arcsec); a band with no associated
    pairs, no mosaic, or no input map with MIN_OFFSET_PAIRS has an empty list."""
    result: dict = {}
    for b in BANDS:
        pairs = photometry.columns.get("_PAIRS_" + b)
        result[b] = []
        if pairs is None or pairs.shape[0] == 0 or b not in mosaics:
            continue
        for name, wcs, shape in _input_footprints(mosaics[b]):
            x, y = wcs.all_world2pix(pairs[:, 2], pairs[:, 3], 0)
            inside = (x >= -0.5) & (x < shape[1] - 0.5) & (y >= -0.5) & (y < shape[0] - 0.5)
            if inside.sum() >= MIN_OFFSET_PAIRS:
                result[b].append((name, int(inside.sum()), float(np.median(pairs[inside, 0])),
                                  float(np.median(pairs[inside, 1]))))
    return result
