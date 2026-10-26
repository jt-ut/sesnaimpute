"""Herschel PACS 70 and 100 micron flux at every SESNA source position
(SPEC_PACS_FARIR_DRAFT.md sections 4, 7b and 11.3 step 3).

`build(config, regions=None)` measures, for every source of
`catalog/sesna/sources_sesna_source__<Region>.hdf5`, the flux in each PACS
band on the map the region's completeness stage chose for the source's cell
(`sky.derived.pacs_completeness`, the deepest map there), from the maps the
download index lists (`sky.download.herschel_pacs.index.read_index`), and
writes `sky/derived/pacs/photometry_pacs_source__<Region>.hdf5`: one row per
source in the catalogue's own order, with its flux, its noise, the origin
code the fitter reads (1 detected, 2 upper bound from an unseparable blend,
90 observed but not detected, 91 not covered), the contributing observation
ids, the blend bookkeeping and the extended-source flag.

Everything numerical is `sky.derived.pacs_fit`'s: the forced two-column PSF
fit with the bright neighbours as nuisance columns, the blend-group fit, the
fitted width. This stage only decides which sources go to which map tile,
runs the singleton pass, forms the blend groups and turns the fits into
products:

- noise: a source's sigma is the noise of its 30-arcsec cell,
  `CELL_SIGMA_<B>_MJY` of the completeness product (the scatter of the same
  fit at random empty-sky positions in the cell); the fit's propagated pixel
  noise is never added to it. A member of a blend group carries the cell
  noise times the ratio of its propagated error in the group fit to that of a
  singleton fit at the same position (the correlation inflation).
- detection: flux >= 5 sigma (origin 1), else origin 90; a source the
  completeness stage found no cell for, or whose fit has no data, is 91.
- blends: SESNA sources within one beam of a source detected in the singleton
  pass are fitted together with it; a member whose fitted flux has
  |correlation| > 0.9 with another member's cannot be separated and gets
  origin 2, flux = max(0, group flux) + 5 x group sigma, and the pipeline's
  upper-limit sentinel as its sigma.
- extended: a detected source whose fitted width exceeds 1.5 x the beam.

Work is split by map tile (one piece per band and map), the tile's sources
being transformed to map pixels in one call; the product is preallocated and
each tile's rows are written as they arrive.
"""

import functools
import os
import time

import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog.curated import UPPER_LIMIT_SIGMA
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived import pacs_fit
from sesnaimpute.sky.derived.pacs_fit import DETECTION_SNR
from sesnaimpute.sky.derived.pacs_psf import beam_fwhm_arcsec
from sesnaimpute.sky.download.herschel_pacs.index import read_index

BANDS = ("P70", "P100")

#: the pipeline's origin codes for a flux (`ORIGIN_FNU` in the source table)
ORIGIN_DETECTED, ORIGIN_UPPER_BOUND, ORIGIN_UNDETECTED, ORIGIN_NOT_COVERED = 1, 2, 90, 91

#: two members of a blend group whose fitted fluxes correlate more strongly than
#: this cannot be told apart (owner 2026-10-26, SPEC 11.1 item 4c)
BLEND_RHO_LIMIT = 0.9
#: a detection fitted wider than this many beam widths is flagged extended
#: (SPEC 11.3 step 3); the width is `pacs_fit.Tile.fitted_fwhm`, the beam
#: broadened by a Gaussian chosen by chi-square
EXTENDED_BEAM_FACTOR = 1.5
#: a source's group neighbours are the sources within one beam FWHM of a
#: detected source (SPEC 4: "within one beam of each other")
BLEND_RADIUS_BEAMS = 1.0


# ---------------------------------------------------------------------------
# blend groups

def _unit_vectors(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _blend_components(ra_deg, dec_deg, detected, radius_arcsec):
    """Connected components of the graph that joins two sources closer than
    `radius_arcsec` when at least one of them is detected. Returns `(label,
    size)` per source: label is the component number, size how many sources
    share it (1 for a source with no detected source within the radius)."""
    n = ra_deg.size
    chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
    tree = cKDTree(_unit_vectors(ra_deg, dec_deg))
    pairs = tree.query_pairs(chord, output_type="ndarray")
    if pairs.size:
        pairs = pairs[detected[pairs[:, 0]] | detected[pairs[:, 1]]]
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    size = np.bincount(label)[label]
    return label, size


# ---------------------------------------------------------------------------
# one tile

def _measure_tile(piece, config, records, ra, dec, cell_sigma):
    """All of one map tile's sources: `piece = (band, map_index, rows)`, the
    source rows (ascending) the completeness stage assigned to that map.
    Returns `(band, map_index, rows, columns)`, `columns` the product's
    per-source arrays for those rows."""
    band, map_index, rows = piece
    record = records[map_index]
    n = rows.size
    sigma_cell = cell_sigma[band][rows]
    beam = beam_fwhm_arcsec(record)

    flux = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    origin = np.full(n, ORIGIN_NOT_COVERED, dtype=np.int8)
    n_blend = np.zeros(n, dtype=np.int16)
    group_id = np.full(n, -1, dtype=np.int32)
    extended = np.zeros(n, dtype=bool)

    with pacs_fit.open_tile(config, record) as tile:
        x, y = tile.xy(ra[rows], dec[rows])
        single = tile.fit_positions(x, y)
        ok = single.covered & np.isfinite(sigma_cell) & (sigma_cell > 0)
        flux[ok] = single.flux[ok]
        sigma[ok] = sigma_cell[ok]
        detected = ok & (flux >= DETECTION_SNR * sigma_cell)
        origin[ok] = np.where(detected[ok], ORIGIN_DETECTED, ORIGIN_UNDETECTED)
        n_blend[ok] = 1

        # Blend groups around the sources detected in the singleton pass.
        label, size = _blend_components(ra[rows], dec[rows], detected, BLEND_RADIUS_BEAMS * beam)
        grouped = size > 1
        if grouped.any():
            g = np.flatnonzero(grouped)
            gfit = tile.fit_groups(x[g], y[g], label[g])
            # the group's identifier is the lowest catalogue row among its members
            lowest = np.full(label.max() + 1, np.iinfo(np.int64).max)
            np.minimum.at(lowest, label[g], rows[g])
            group_id[g] = lowest[label[g]]
            n_blend[g] = size[g]
            good = gfit.covered & ok[g]
            # the empirical-over-propagated noise ratio of each member's singleton
            # fit, averaged over the group, scales the group-total sigma the same way
            ratio = np.where(ok[g], sigma_cell[g] / np.where(single.sigma[g] > 0, single.sigma[g], np.nan), np.nan)
            ratio_sum = np.bincount(label[g], weights=np.nan_to_num(ratio))
            ratio_n = np.bincount(label[g], weights=np.isfinite(ratio).astype(float))
            ratio_mean = (ratio_sum / np.maximum(ratio_n, 1))[label[g]]
            member_sigma = sigma_cell[g] * gfit.inflation
            unseparable = good & (gfit.max_abs_rho > BLEND_RHO_LIMIT)
            separable = good & ~unseparable
            gi = g[separable]
            flux[gi] = gfit.flux[separable]
            sigma[gi] = member_sigma[separable]
            origin[gi] = np.where(flux[gi] >= DETECTION_SNR * sigma[gi], ORIGIN_DETECTED, ORIGIN_UNDETECTED)
            gu = g[unseparable]
            flux[gu] = (np.maximum(0.0, gfit.group_flux[unseparable])
                        + DETECTION_SNR * gfit.group_sigma[unseparable] * ratio_mean[unseparable])
            sigma[gu] = UPPER_LIMIT_SIGMA
            origin[gu] = ORIGIN_UPPER_BOUND

        # Extended: detections only (a group's separable detections included).
        det = np.flatnonzero(origin == ORIGIN_DETECTED)
        if det.size:
            extended[det] = tile.fitted_fwhm(x[det], y[det]) > EXTENDED_BEAM_FACTOR * beam

    columns = dict(FNU=flux, SIGMA=sigma, ORIGIN=origin, LOCAL_RMS=sigma_cell.astype(np.float32),
                   N_BLEND=n_blend, BLEND_GROUP=group_id, EXTENDED=extended)
    # a source with no usable fit keeps the cell noise only where a cell exists
    columns["LOCAL_RMS"][origin == ORIGIN_NOT_COVERED] = np.nan
    return band, map_index, rows, columns


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


_DTYPES = dict(FNU=np.float64, SIGMA=np.float64, ORIGIN=np.int8, LOCAL_RMS=np.float32,
               N_BLEND=np.int16, BLEND_GROUP=np.int32, EXTENDED=bool)
_FILL = dict(FNU=np.nan, SIGMA=np.nan, ORIGIN=ORIGIN_NOT_COVERED, LOCAL_RMS=np.nan,
             N_BLEND=0, BLEND_GROUP=-1, EXTENDED=False)
_NAMES = dict(FNU="FNU_{b}_MJY", SIGMA="SIGMA_{b}_MJY", ORIGIN="ORIGIN_{b}", LOCAL_RMS="LOCAL_RMS_{b}_MJY",
              N_BLEND="N_BLEND_{b}", BLEND_GROUP="BLEND_GROUP_{b}", EXTENDED="EXTENDED_{b}")


def _tile_key(record):
    return ",".join(str(o) for o in record.obsids)


def write_region(out_path, names, ra, dec, records, cell_sigma, cell_obsid, config, n_jobs, stage=None):
    """Measures every source on its cell's map and writes the product at
    `out_path`. `cell_sigma[b]` (n,) float64 is the source's cell noise (NaN:
    no cell, not covered) and `cell_obsid[b]` (n,) bytes the cell's map, the
    comma-joined observation ids of the `records` entry that holds it. Returns
    the per-band origin counts."""
    n = names.size
    key_to_map = {}
    for i, rec in enumerate(records):
        key_to_map.setdefault((rec.band, _tile_key(rec)), i)
    pieces = []
    for b, band in enumerate(BANDS):
        sel = np.flatnonzero(np.isfinite(cell_sigma[b]))
        if sel.size == 0:
            continue
        for key in np.unique(cell_obsid[b][sel]):
            i = key_to_map.get((band, key.decode()))
            if i is not None:
                pieces.append((b, i, sel[cell_obsid[b][sel] == key]))
    obsid_len = max([len(_tile_key(r)) for r in records] + [1])

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
            name = "OBSID_" + band
            dsets[b, "OBSID"] = build_module.write_dataset(
                f, name, None, *REGISTRY[("photometry_pacs_source", name)],
                shape=(n,), dtype="S%d" % obsid_len, fillvalue=b"", chunks=(min(n, 65536),) if n else None)
            counts[band] = {code: 0 for code in (1, 2, 90, 91)}
            counts[band][ORIGIN_NOT_COVERED] = n
        fn_args = dict(config=config, records=records, ra=ra, dec=dec, cell_sigma=np.vstack(cell_sigma))
        fn = functools.partial(_measure_tile, **fn_args)
        for band_index, map_index, rows, cols in pieces_module.iterate(fn, pieces, n_jobs, stage=stage, unit="map tiles"):
            band = BANDS[band_index]
            obsid = _tile_key(records[map_index]).encode()
            for start, stop, off in _runs(rows):
                width = stop - start
                for col, data in cols.items():
                    dsets[band_index, col][start:stop] = data[off:off + width]
                dsets[band_index, "OBSID"][start:stop] = obsid
            counts[band][ORIGIN_NOT_COVERED] -= rows.size
            for code in (1, 2, 90, 91):
                counts[band][code] += int(np.sum(cols["ORIGIN"] == code))
    return counts


def _read_cells(config, region, n_source):
    """The completeness stage's per-source cell noise and map, per band:
    `(cell_sigma, cell_obsid)` lists over BANDS. A source with `SOURCE_ROW`
    -1 has no cell (noise NaN)."""
    path = product_path(config, "sky/derived", "pacs", "completeness", "source", region=region)
    sigma = [np.full(n_source, np.nan) for _ in BANDS]
    obsid = [np.full(n_source, b"", dtype="S1") for _ in BANDS]
    if not os.path.exists(path):
        raise FileNotFoundError("pacs_photometry needs the completeness product first: %s" % path)
    with h5py.File(path, "r") as f:
        source_row = f["SOURCE_ROW"][:]
        for b, band in enumerate(BANDS):
            if "CELL_SIGMA_%s_MJY" % band not in f:
                continue
            cell_sigma = f["CELL_SIGMA_%s_MJY" % band][:]
            if cell_sigma.size == 0:
                continue
            rows = source_row[:, b]
            have = rows >= 0
            cell_obs = f["CELL_OBSID_%s" % band][:]
            sigma[b][have] = cell_sigma[rows[have]]
            obsid[b] = np.where(have, cell_obs[np.where(have, rows, 0)], b"")
    return sigma, obsid


def _build_one_region(config, region, st):
    src_path = product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(src_path, "r") as f:
        names = f["NAME"][:]
        ra = f["RA_DEG"][:].astype(np.float64)
        dec = f["DEC_DEG"][:].astype(np.float64)
    out_path = product_path(config, "sky/derived", "pacs", "photometry", "source", region=region)
    records = read_index(config, region)
    if records:
        cell_sigma, cell_obsid = _read_cells(config, region, names.size)
    else:
        cell_sigma = [np.full(names.size, np.nan) for _ in BANDS]
        cell_obsid = [np.full(names.size, b"", dtype="S1") for _ in BANDS]
    t0 = time.time()
    counts = write_region(out_path, names, ra, dec, records, cell_sigma, cell_obsid,
                          config, config.n_jobs, stage=st)
    wall = time.time() - t0
    numbers = {"sources": int(names.size), "sources_per_s": names.size / max(wall, 1e-9)}
    for band in BANDS:
        for code, label in ((1, "detected"), (2, "bound"), (90, "undetected"), (91, "uncovered")):
            numbers["%s_%s" % (band, label)] = counts[band][code]
    return out_path, numbers


def build(config, regions=None):
    """Writes each region's `photometry_pacs_source` product (all-91 where no
    PACS map covers the region). Regions run one after another; inside a
    region the map tiles are the pieces."""
    names = regions or [r.name for r in regions_module.REGIONS]
    t0 = time.time()
    for name in names:
        with progress_module.Stage("sky.derived.pacs_photometry", name) as st:
            path, numbers = _build_one_region(config, name, st)
            st.done(path, **numbers)
    progress_module.summary("sky.derived.pacs_photometry", len(names), t0)


if __name__ == "__main__":
    run(build)
