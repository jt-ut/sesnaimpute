"""One co-added PACS mosaic per band per region, built tile by tile, and the measurement engine on it.

Depends on: grid, psf. Replaces: sky/derived/pacs_mosaic.py (the co-add, zero levels, product), and the measurement half
of pacs_fit.py (Tile -> `Mosaic` + `Window`: aperture photometry, the PSF fit, bright neighbours, fitted widths,
injection patches; open_map/_load_layers/mosaic_layer_plan/mosaic_layer_strip/prepared copies -> the reader masks of
`Mosaic.window`). "Tile" in the old sense (one whole prepared map) is retired; a tile here is always the 256-pixel
storage and work unit.

Store `<data_root>/sky/derived/pacs/mosaic/<Region_safe>/<band>[__slice_<ra>_<dec>_<w>x<h>]/`:
    image/ error/ coverage/ weight/   TiledLayer float32 (fill NaN, NaN, 0, 0) on the band's PixelGrid; image/index.json
                                      "meta" holds META once (region, band, shape, window_arcsec, pixel_arcsec, resampler,
                                      maps [{path, group, smooth, smooth_step, window_px, zero_level_jy_per_pixel,
                                      zero_level_mjy_per_beam}], dropped, groups, pairs, group_keys, box, floors, units,
                                      reading, neighbours); every layer's index.json holds the WCS header
    smooth/<input map>/               each input map's smoothed error (TiledLayer float64, input pixels)
    psf/                              the mosaic PSF (psf.py)
No mosaic.hdf5, no prepared copies (the mJy scale and the error mask are the reader's, FROZEN 12).

Method (briefs/pacs_mosaic_2026-10-29/BRIEF.md, M1-M4 with amendment A1; unchanged from pacs_mosaic.py):
  M1  input set = the band's maps minus any map whose observation set is a subset of another map's.
  M2  TAN grid at the footprint centre, north up (grid.PixelGrid, FROZEN 13).
  M3a one constant zero level o_i per map from the median pairwise differences over every common output pixel
      (`solve_zero_levels`), in surface brightness (divided by the output pixel's own area factor).
  M3  each input (minus o_i) resampled onto the grid by bilinear interpolation of its surface brightness times the
      output pixel's own area; weight of map i = 1 / (smoothed error_i)^2 (FROZEN 6);
        image = sum(w_i I_i) / sum(w_i);   error = sqrt(sum(w_i^2 err_i^2)) / sum(w_i)
  M4  the PSF groups (maps with identical published-template sets and scan angles) are listed in META.
Work unit = one 256 x 256 tile: pass 1 (valid counts, pairwise differences) and pass 2 (the co-add) visit only the
tiles some input map's footprint touches; each pixel's value depends only on its own coordinates, so a tile equals the
same pixels of the old strip build bit for bit.

Measurements read a `Window` (a rectangle of the grid through the reader) and take positions in grid pixels; the
cores see positions minus the window origin. A window origin at or below the position (and >= 0) is an exact shift in
float64, so every distance the cores compute equals the whole-grid one (the sub-pixel test of the aperture weights is
a second rounding: equal unless a sample lies within ~1e-13 px of the aperture edge).

Round 2 note: psf.py is still a skeleton on main, so the input-map primitives (reading a FITS window, the smoothed
error), the PSF build and MosaicPSF are reached through private adapters to the old pacs_mosaic / pacs_psf / pacs_fit
code, each marked `# R3: swap` (round 3 replaces them by psf.InputMap / psf.build_smoothed_error /
psf.build_mosaic_psf / psf.MosaicPSF, same results).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import warnings
from collections import OrderedDict
from types import MappingProxyType
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numba
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from sesnaimpute.definitions import PACS_APERTURES as APERTURES
from sesnaimpute.definitions import PACS_PIXEL_ARCSEC as PIXEL_ARCSEC
from sesnaimpute.sky.derived.pacs import grid, psf
from sesnaimpute.sky.derived.pacs.grid import CellGrid, PixelGrid, TiledLayer
from sesnaimpute.sky.download.herschel_pacs.index import MapRecord, map_rank, maps_root, read_index
# R3: swap -- the three old modules below serve only the adapters to psf.py (a skeleton in round 2)
from sesnaimpute.sky.derived import pacs_fit as _old_fit
from sesnaimpute.sky.derived import pacs_mosaic as _old_mosaic
from sesnaimpute.sky.derived import pacs_psf as _old_psf

warnings.filterwarnings("ignore", message=".*datfix.*")

#: the layers of a mosaic, in store order
LAYERS = ("image", "error", "coverage", "weight")

# ----------------------------------------------------------------------------------------------------------------------
# constants (moved unchanged from pacs_mosaic.py / pacs_fit.py; the frozen ones live in grid.py)

#: least common valid output pixels for a pair of maps to enter the zero-level solve (pacs_mosaic.MIN_COMMON_PIXELS)
MIN_COMMON_PIXELS = 2000
#: the pair order of the zero-level solve is the old strip build's: pairs in order of the first 64-row band of the
#: grid holding a common valid pixel, then by map order (the least-squares rows, hence o_i's last bits, follow it)
PAIR_ORDER_ROWS = 64
#: fitting-box half-width in beam FWHM (spec 11.3 step 1)
FIT_BOX_FWHM = 3.0
#: bright-neighbour finding: single-PSF fit S/N against the propagated pixel noise
NEIGHBOUR_SNR = 5.0
#: a blind local maximum must reach this pixel S/N before it is fitted (only skips the noise floor)
NEIGHBOUR_PEAK_SNR = 1.0
#: a neighbour's PSF is modelled out to this many beam FWHM beyond the fitting box edge
NEIGHBOUR_REACH_FWHM = 2.0
#: at most this many nuisance sources (nearest first) enter one fit
MAX_NUISANCE = 6
#: the response table's half-width cap, arcsec (the published templates span +-100 arcsec)
PSF_TABLE_HALF_ARCSEC = 65.0
#: a fit is trusted only if this fraction of the target beam lies on valid pixels
MIN_BEAM_FRACTION = 0.9
#: sub-pixel steps per pixel of the cached outer-light patterns (enters only the annulus sky)
OUTER_LIGHT_STEPS = 16
#: outer-light patterns kept at once (one per sub-group and third x third of the grid)
OUTER_LIGHT_KEPT = 64
#: the mJy-per-Jy factor of the reader (float32, as the prepared copies were scaled)
_MJY = np.float32(1e3)

_LAYER_TEXT = {
    "image": ("Jy per output pixel", "The mosaic's flux in each pixel, the weighted mean of the input maps resampled onto "
              "this grid. Not a number where no input map has a valid pixel."),
    "error": ("Jy per output pixel", "The 1-sigma noise of the mosaic pixel, propagated from the input maps' own per-pixel "
              "noise layers and the weights. It treats neighbouring input pixels as fully correlated through the "
              "resampling (see the mosaic's READING)."),
    "coverage": ("input coverage units", "The sum over the contributing maps of their resampled coverage layer (the "
                 "number of exposure samples that fell in the pixel); 0 where no input map is valid."),
    "weight": ("(Jy per output pixel)^-2", "The total weight, the sum over the contributing maps of 1 / (smoothed "
               "noise)^2; 0 where no input map is valid."),
}

_READING = (
    "PACS {band} mosaic of {region}. Every input map of the band that is not contained in another map's observations "
    "is resampled onto this one grid (tangent projection, north up, {pix} arcsec pixels). Resampling interpolates the "
    "surface brightness (flux per unit area) bilinearly between the four nearest input pixel centres and multiplies by "
    "the output pixel's own area (a tangent-projection pixel shrinks away from the grid centre), so the flux of a "
    "source is conserved everywhere on the grid. The maps are averaged with weights "
    "1/(smoothed noise)^2, where the smoothed noise of a map is its own per-pixel noise layer median-filtered over a "
    "square window of {win} arcsec; a weight therefore follows how deep the map is, never the signal in the pixel. "
    "Pixels where a map has zero or undefined noise, zero coverage or an undefined image get weight 0 in that map. "
    "The image is sum(w*I)/sum(w); the error is sqrt(sum(w^2*e^2))/sum(w) from the maps' real per-pixel noise, with "
    "the four interpolation neighbours of a pixel taken as fully correlated, so it is not reduced by the interpolation. "
    "Each input map first has its own arbitrary zero level (PACS maps have no defined absolute zero) replaced by one "
    "constant surface brightness, solved from all overlaps between maps (least squares on the median differences of "
    "overlapping maps, each taken over every pixel the two maps share); in each output pixel that constant is "
    "multiplied by the pixel's own area, like the flux, so it changes no point-source flux measured against the local "
    "background. Units: Jy per output pixel. Pixels with no valid input are not-a-number. The store is tiled (256 x 256 "
    "pixels); a tile holding no covered pixel is not stored and reads as not-a-number (image, error) or 0 (coverage, "
    "weight).")


# ----------------------------------------------------------------------------------------------------------------------
# inputs and groups (pacs_mosaic.py:94-131)

def _label(rec) -> str:
    return f"{rec.map_id} [{os.path.basename(rec.path)}] obs {','.join(rec.obsids)}"


def input_set(records: Sequence, band: str) -> Tuple[list, list]:
    """(kept, dropped): the band's maps (each file once), minus any map whose observation set is a subset of another
    kept map's; `dropped` = [(record, host record)]. Identity: `pacs_mosaic.input_set` -- same kept list, same order
    (sorted by path): META's map order, hence FROZEN 7's sub-group numbering."""
    recs = sorted({r.path: r for r in records if r.band == band}.values(), key=lambda r: r.path)
    order = sorted(recs, key=lambda r: (-len(set(r.obsids)), map_rank(r), r.path))
    kept, dropped = [], []
    for r in order:
        host = next((k for k in kept if set(r.obsids) <= set(k.obsids)), None)
        if host is None:
            kept.append(r)
        else:
            dropped.append((r, host))
    kept.sort(key=lambda r: r.path)
    return kept, dropped


def _print_input_set(region, band, kept, dropped) -> None:
    print(f"pacs mosaic {region} {band}: {len(kept)} maps kept, {len(dropped)} dropped (observation set contained in "
          f"another map's)")
    for r in kept:
        print(f"  kept    {_label(r)}")
    for r, host in dropped:
        print(f"  dropped {_label(r)}  (inside {host.map_id} [{os.path.basename(host.path)}])")


def group_key(config, record) -> tuple:
    """A map's PSF group key: its published templates and position angles (`pacs_mosaic.group_key`)."""
    comps = _published_components(config, record)
    return tuple((c.path, round(float(c.position_angle_deg), 6), round(float(c.image.sum()), 9),
                  round(float((c.image ** 2).sum()), 9)) for c in comps)


def mosaic_dir(config, region: str, band: str, box: Optional[Sequence[float]] = None) -> str:
    """The mosaic's directory (the path of `pacs_mosaic.product_dir`: <data_root>/sky/derived/pacs/mosaic/<safe>/<band>
    [__slice_%.3f_%.3f_%.2fx%.2f])."""
    safe = re.sub(r"[^A-Za-z0-9+-]+", "_", region)
    return os.path.join(config.data_root, "sky", "derived", "pacs", "mosaic", safe,
                        band + ("" if box is None else "__slice_%.3f_%.3f_%.2fx%.2f" % tuple(box)))


def _assert_inside_derived(config, path: str) -> None:
    """`path` must really lie inside this root's own sky/derived (no symlink in between)."""
    derived = os.path.join(os.path.realpath(config.data_root), "sky", "derived")
    assert os.path.realpath(os.path.join(config.data_root, "sky", "derived")) == derived, "sky/derived is a symlink"
    real = os.path.realpath(path)
    assert real == derived or real.startswith(derived + os.sep), f"{path} resolves outside {derived}"


def _smooth_name(record) -> str:
    """The smoothed error's store folder of an input map, relative to the mosaic: smooth/<path, / -> __, no suffix>."""
    return os.path.join("smooth", os.path.splitext(record.path)[0].replace("/", "__"))


# ----------------------------------------------------------------------------------------------------------------------
# adapters to psf.py (a skeleton in round 2): the old code's results, the new store's layout.  # R3: swap

def _published_components(config, record):
    """`psf.published_components`.  # R3: swap"""
    return _old_psf.published_components_for_map(config, record)


def _beam_fwhm_arcsec(record) -> float:
    """`psf.beam_fwhm_arcsec`.  # R3: swap"""
    return _old_psf.beam_fwhm_arcsec(record)


def _aperture_eef(config, record) -> float:
    """`psf.aperture_eef`.  # R3: swap"""
    return _old_psf.aperture_eef(config, record)


def _good(img, err, cov):
    """`psf.good_pixels` (pacs_mosaic._good).  # R3: swap"""
    return np.isfinite(img) & np.isfinite(err) & (err > 0) & (cov > 0)


def _read_layers(input_map, rows: slice, cols: slice):
    """`psf.InputMap.layers` (pacs_mosaic._read_layers): float64 copies of the window rows x cols.  # R3: swap"""
    rec = input_map.record
    with fits.open(os.path.join(input_map.maps_root, rec.path), mode="denywrite", memmap=True) as hdus:
        return tuple(np.array(hdus[i].data[rows, cols], dtype=np.float64)
                     for i in (rec.image_hdu, rec.error_hdu, rec.coverage_hdu))


_SMOOTH_DENSE = {}


def _smooth_dense(directory: str) -> np.ndarray:
    """The smoothed-error samples of an input map as one dense float64 array, kept per process (they are small:
    (ny / step) x (nx / step)); keyed by the store's index time, so a rebuilt store is read again.
    # R3: swap (psf.InputMap.smoothed_error_at reads the TiledLayer)"""
    key = (directory, os.stat(os.path.join(directory, "index.json")).st_mtime_ns)
    if key not in _SMOOTH_DENSE:
        for old in [k for k in _SMOOTH_DENSE if k[0] == directory]:
            del _SMOOTH_DENSE[old]
        _SMOOTH_DENSE[key] = TiledLayer.open(directory, "r").to_dense()
    return _SMOOTH_DENSE[key]


def _smoothed_error_at(input_map, px, py):
    """`psf.InputMap.smoothed_error_at` (pacs_mosaic._lattice_at on the stored samples).  # R3: swap"""
    return _old_mosaic._lattice_at(_smooth_dense(input_map.smooth.directory), input_map.smooth_step, px, py)


def _build_smoothed_error(input_map, out_dir: str) -> TiledLayer:
    """`psf.build_smoothed_error`: pacs_mosaic._smooth_lattice's samples (FROZEN 6) written as a TiledLayer
    (float64, fill NaN) instead of _smooth/<map>.npy.  # R3: swap"""
    rec = input_map.record
    win = grid.smooth_window_px(rec.pixel_scale_arcsec)
    half, step = win // 2, grid.smooth_step_px(rec.pixel_scale_arcsec)
    ny, nx = rec.shape
    nj, ni = (ny - 1) // step + 2, (nx - 1) // step + 2
    lat = np.full((nj, ni), np.nan)
    with fits.open(os.path.join(input_map.maps_root, rec.path), mode="denywrite", memmap=True) as hdus:
        eh, ch, ih = hdus[rec.error_hdu].data, hdus[rec.coverage_hdu].data, hdus[rec.image_hdu].data
        for j in range(nj):
            r0, r1 = max(0, j * step - half), min(ny, j * step + half + 1)
            if r1 <= r0:
                continue
            err = np.array(eh[r0:r1], dtype=np.float64)
            err[~_good(np.array(ih[r0:r1], dtype=np.float64), err, np.array(ch[r0:r1], dtype=np.float64))] = np.nan
            padded = np.pad(err, ((0, 0), (half, max(0, (ni - 1) * step + win - nx - half))), constant_values=np.nan)
            view = np.lib.stride_tricks.sliding_window_view(padded, (err.shape[0], win), axis=(0, 1))
            cut = view[0, ::step][:ni].reshape(ni, -1)
            ok = np.isfinite(cut).any(axis=1)
            row = np.full(ni, np.nan)
            row[ok] = np.nanmedian(cut[ok], axis=1)
            lat[j] = row
    return TiledLayer.from_dense(lat, out_dir, fill=np.nan)


class _LegacyMosaicPSF(_old_psf.MosaicPSF):
    """`psf.MosaicPSF` on the new store: pacs_psf.MosaicPSF (every method unchanged) reading `<mosaic>/psf/psf.json`
    and each input map's smoothed error from `smooth/<map>/` (key `smooth`, was `lattice`).  # R3: swap"""

    def __init__(self, directory, maps_root, tables=True):
        self.directory = directory
        pdir = os.path.join(directory, "psf")
        with open(os.path.join(pdir, "psf.json")) as fh:
            self.meta = json.load(fh)
        m = self.meta
        self.shape = tuple(m["shape"])
        self.extent, self.subpixel_steps, self.integration_order = m["extent"], m["subpixel_steps"], m["integration_order"]
        self.n_sub = m["n_sub"]
        self.table_half = m.get("table_half")
        self.out_wcs = WCS(fits.Header.fromstring(m["out_header"])).celestial
        self._maps = []
        for q in m["maps"]:
            self._maps.append(dict(wcs=WCS(fits.Header.fromstring(q["wcs_header"])).celestial, shape=tuple(q["shape"]),
                                   lat=_smooth_dense(os.path.join(directory, q["smooth"])), step=q["smooth_step"],
                                   scale=q["scale"], sub=q["sub"], bbox=q["bbox"], path=q["path"], hdus=q["hdus"]))
        self.maps_root = str(maps_root)
        self.fbar = np.array(m["fbar"])
        self.enclosed = np.array(m["enclosed_sub"])
        self.aperture_fraction_sub = np.array(m["aperture_fraction_sub"])
        self.sub_beams = np.array(m["sub_beams"])
        if tables:
            self.subs = [np.load(os.path.join(pdir, f"sub{k}.npy"), mmap_mode="r") for k in range(self.n_sub)]
            self.table = np.load(os.path.join(pdir, "table.npy"), mmap_mode="r")


def _build_mosaic_psf(config, region, band, header_text, shape, kept, meta_maps, image, mdir, n_jobs=None):
    """`psf.build_mosaic_psf`: pacs_fit.build_mosaic_psf (pacs_fit.py:1581-1676) line for line, on the new store
    (psf/ for _psf/, map entries' `smooth`/`smooth_step` for `lattice`/`lattice_step`, the image TiledLayer's point
    reads for the dense image). Same tables, bit for bit.  # R3: swap"""
    recs = {r.path: r for r in kept}
    full_wcs = WCS(fits.Header.fromstring(header_text))          # as pacs_mosaic.Mosaic.wcs
    out_wcs = full_wcs.celestial
    ny, nx = shape
    pix = grid.pixel_scale_arcsec(out_wcs)
    out_area = pixel_area_arcsec2(out_wcs)
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    keys, entries = [], []
    for q in meta_maps:
        rec = recs[q["path"]]
        in_wcs = grid.celestial_wcs(rec.wcs_header)
        jac = _old_fit._jacobian(out_wcs, in_wcs, cx, cy)
        angle = float(np.degrees(np.arctan2(jac[1, 0], jac[0, 0])) % 90.0)
        key = grid.subgroup_key(q["group"], rec.pixel_scale_arcsec, angle)
        if key not in keys:
            keys.append(key)
        bbox = _bbox_in_output(rec, in_wcs, out_wcs, tuple(shape)) or (0, 0, 0, 0)
        entries.append(dict(path=q["path"], wcs_header=rec.wcs_header, shape=list(rec.shape), smooth=q["smooth"],
                            smooth_step=q["smooth_step"], scale=pixel_area_arcsec2(out_wcs) / pixel_area_arcsec2(in_wcs),
                            sub=keys.index(key), bbox=[int(v) for v in bbox],
                            hdus=[rec.image_hdu, rec.error_hdu, rec.coverage_hdu]))
    n_sub = len(keys)
    rep = [next(r for r, e in zip(meta_maps, entries) if e["sub"] == k) for k in range(n_sub)]
    rep_rec = [recs[q["path"]] for q in rep]
    beams = np.array([_beam_fwhm_arcsec(r) for r in rep_rec])
    pdir = os.path.join(mdir, "psf")
    os.makedirs(pdir, exist_ok=True)
    _assert_inside_derived(config, pdir)
    meta = dict(region=region, rep=[entries.index(next(e for e in entries if e["sub"] == k)) for k in range(n_sub)],
                shape=[ny, nx], n_sub=n_sub, keys=[list(k) for k in keys], maps=entries,
                out_header=full_wcs.to_header(relax=True).tostring(sep=""),
                extent=0.0, subpixel_steps=_old_fit.PSF_SUBPIXEL_STEPS, integration_order=8, fbar=[1.0 / n_sub] * n_sub,
                enclosed_sub=[1.0] * n_sub, aperture_fraction_sub=[1.0] * n_sub, sub_beams=[float(b) for b in beams])
    with open(os.path.join(pdir, "psf.json"), "w") as fh:
        json.dump(meta, fh)
    probe = _LegacyMosaicPSF(mdir, maps_root(config), tables=False)
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 40), np.linspace(0, ny - 1, 40))
    f = probe.weights(gx.ravel(), gy.ravel())
    valid = np.isfinite(image[np.rint(gy.ravel()).astype(int), np.rint(gx.ravel()).astype(int)])
    fbar = f[valid].mean(axis=0) if valid.any() else f.mean(axis=0)
    beam = float(fbar @ beams)
    beam_px = beam / pix
    half = int(np.ceil(max(FIT_BOX_FWHM * beam_px, APERTURES[band][2] / pix + 1.0)))
    table_half = int(min(half + np.ceil(NEIGHBOUR_REACH_FWHM * beam_px), PSF_TABLE_HALF_ARCSEC // pix))
    extent = table_half + 0.5
    steps = _old_fit.PSF_SUBPIXEL_STEPS
    count = int(2 * extent * steps) + 1
    n_nodes = grid.psf_node_count(nx, ny, pix)
    nodes = [(sx, sy) for sy in grid.psf_node_positions(ny, n_nodes) for sx in grid.psf_node_positions(nx, n_nodes)]
    mean_table = np.zeros((n_nodes, n_nodes, 1, count, count))
    paths = [os.path.join(pdir, f"sub{k}.npy") for k in range(n_sub)]
    comps = []
    for k in range(n_sub):
        _assert_inside_derived(config, paths[k])
        open_memmap(paths[k], mode="w+", dtype=np.float32, shape=(n_nodes, n_nodes, 1, count, count)).flush()
        comps.append(_published_components(config, rep_rec[k]))
    in_wcs = [grid.celestial_wcs(r.wcs_header) for r in rep_rec]
    centre = (n_nodes ** 2) // 2
    done = Parallel(n_jobs=n_jobs or config.n_jobs)(
        delayed(_old_fit._mosaic_node)(comps[k], out_wcs, in_wcs[k], node, extent, steps, out_area, band, pix, paths[k],
                                       i, i == centre)
        for k in range(n_sub) for i, node in enumerate(nodes))
    enclosed = np.zeros((n_sub, n_nodes, n_nodes))
    aperture = np.zeros((n_sub, n_nodes, n_nodes))
    totals = []
    for k in range(n_sub):
        res = done[k * n_nodes ** 2:(k + 1) * n_nodes ** 2]
        scale = _aperture_eef(config, rep_rec[k]) / res[centre][2]
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
    meta.update(extent=extent, fbar=[float(v) for v in fbar], enclosed_sub=enclosed.tolist(),
                aperture_fraction_sub=aperture.tolist(), beam_arcsec=beam,
                enclosed_fraction=float(fbar @ enclosed[(slice(None), *ctr)]),
                aperture_fraction=float(fbar @ aperture[(slice(None), *ctr)]), broad_over_plain_total=totals,
                table_half=table_half)
    with open(os.path.join(pdir, "psf.json"), "w") as fh:
        json.dump(meta, fh)
    return _LegacyMosaicPSF(mdir, maps_root(config))


# ----------------------------------------------------------------------------------------------------------------------
# resampling and zero levels (pacs_mosaic.py:267-458)

def pixel_area_arcsec2(wcs) -> float:
    """The pixel area at the reference pixel, arcsec^2 (`pacs_mosaic.pixel_area_arcsec2`)."""
    return float(abs(np.linalg.det(wcs.pixel_scale_matrix)) * 3600.0 ** 2)


def _intermediate(wcs, x, y):
    """The intermediate world coordinates (radians) of zero-origin pixels (x, y) of a TAN `wcs`."""
    if not all(c.endswith("-TAN") for c in wcs.wcs.ctype):
        raise ValueError("the mosaic grid is not a tangent projection")
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    m = wcs.pixel_scale_matrix
    dx, dy = x + 1.0 - wcs.wcs.crpix[0], y + 1.0 - wcs.wcs.crpix[1]
    return np.radians(m[0, 0] * dx + m[0, 1] * dy), np.radians(m[1, 0] * dx + m[1, 1] * dy)


def tan_area_factor(wcs, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The TAN pixel's area at zero-origin (x, y) over the reference pixel's, (1 + rho^2)^-1.5
    (`pacs_mosaic.tan_area_factor`)."""
    xi, eta = _intermediate(wcs, x, y)
    return (1.0 + xi ** 2 + eta ** 2) ** -1.5


def _bbox_in_output(rec, in_wcs, out_wcs, shape):
    """Output pixel bounding box (x0, x1, y0, y1) of the map's footprint, clipped to the grid; None if outside
    (pacs_mosaic._bbox_in_output)."""
    ny, nx = rec.shape
    cx = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
    cy = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    ra, dec = in_wcs.all_pix2world(cx, cy, 0)
    x, y = out_wcs.all_world2pix(ra, dec, 0)
    x0, x1 = max(int(np.floor(x.min())) - 1, 0), min(int(np.ceil(x.max())) + 2, shape[1])
    y0, y1 = max(int(np.floor(y.min())) - 1, 0), min(int(np.ceil(y.max())) + 2, shape[0])
    return None if (x1 <= x0 or y1 <= y0) else (x0, x1, y0, y1)


def map_bbox(input_map: "psf.InputMap", pixel_grid: PixelGrid) -> Optional[Tuple[int, int, int, int]]:
    """The output-pixel bounding box (x0, x1, y0, y1) of an input map's footprint, clipped to the grid; None if
    outside. Identity: `pacs_mosaic._bbox_in_output` (stored in psf.json; read by MosaicPSF.weights)."""
    return _bbox_in_output(input_map.record, input_map.wcs, pixel_grid.wcs, pixel_grid.shape)


def resample_window(input_map: "psf.InputMap", pixel_grid: PixelGrid, y0: int, y1: int, x0: int, x1: int,
                    with_smooth: bool = True) -> Optional[Dict[str, np.ndarray]]:
    """One input map resampled onto output pixels [y0, y1) x [x0, x1) (a tile): None when no output pixel falls inside
    the map, else {valid, image, error (Jy per output pixel), coverage, factor, smooth}. Identity: `pacs_mosaic.
    resample_block` (pacs_mosaic.py:290) -- each pixel's value depends only on its own coordinates, so a tile equals
    the same pixels of a strip bit for bit."""
    rec, in_wcs, out_wcs = input_map.record, input_map.wcs, pixel_grid.wcs
    out_area = pixel_area_arcsec2(out_wcs)
    ny, nx = rec.shape
    X, Y = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(y0, y1, dtype=float))
    ra, dec = out_wcs.all_pix2world(X.ravel(), Y.ravel(), 0)
    px, py = (a.reshape(X.shape) for a in in_wcs.all_world2pix(ra, dec, 0))
    inside = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (px <= nx - 1) & (py >= 0) & (py <= ny - 1)
    if not inside.any():
        return None
    ix0, ix1 = int(np.floor(px[inside].min())), int(np.floor(px[inside].max())) + 2
    iy0, iy1 = int(np.floor(py[inside].min())), int(np.floor(py[inside].max())) + 2
    ix1, iy1 = min(ix1, nx), min(iy1, ny)
    img, err, cov = _read_layers(input_map, slice(iy0, iy1), slice(ix0, ix1))           # R3: swap -> input_map.layers
    good = _good(img, err, cov)
    wx, wy = np.where(inside, px, 0.0) - ix0, np.where(inside, py, 0.0) - iy0
    xi = np.clip(np.floor(wx).astype(np.int64), 0, max(img.shape[1] - 2, 0))
    yi = np.clip(np.floor(wy).astype(np.int64), 0, max(img.shape[0] - 2, 0))
    fx, fy = np.clip(wx - xi, 0, 1), np.clip(wy - yi, 0, 1)
    scale = out_area * tan_area_factor(out_wcs, X, Y) / pixel_area_arcsec2(in_wcs)    # the output pixel's own area
    acc = {k: np.zeros(X.shape) for k in ("image", "error", "coverage")}
    valid = inside.copy()
    for dy, by in ((0, 1 - fy), (1, fy)):
        for dx, bx in ((0, 1 - fx), (1, fx)):
            jj, ii = np.minimum(yi + dy, img.shape[0] - 1), np.minimum(xi + dx, img.shape[1] - 1)
            b = bx * by
            g = good[jj, ii]
            valid &= g | (b == 0)
            for k, a in (("image", img), ("error", err), ("coverage", cov)):
                acc[k] += b * np.where(g, a[jj, ii], 0.0)
    acc["image"] *= scale
    acc["error"] *= scale
    acc["factor"] = tan_area_factor(out_wcs, X, Y)
    acc["valid"] = valid
    if with_smooth:
        s = _smoothed_error_at(input_map, np.where(inside, px, 0.0), np.where(inside, py, 0.0)) * scale  # R3: swap
        acc["smooth"] = s
        acc["valid"] = valid & np.isfinite(s) & (s > 0)
    return acc


def _tile_blocks(pixel_grid: PixelGrid, inputs, iy: int, ix: int):
    """[(input map, (x0, x1, y0, y1), resampled)] of the maps whose footprint box meets tile (iy, ix): each map
    resampled on its box clipped to the tile (pacs_mosaic._blocks, a tile for a strip), in META map order."""
    ty0, ty1, tx0, tx1 = pixel_grid.tile_bounds(iy, ix)
    out = []
    for im in inputs:
        box = map_bbox(im, pixel_grid)
        if box is None:
            continue
        bx0, bx1, by0, by1 = max(box[0], tx0), min(box[1], tx1), max(box[2], ty0), min(box[3], ty1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        r = resample_window(im, pixel_grid, by0, by1, bx0, bx1, with_smooth=True)
        if r is not None:
            out.append((im, (bx0, bx1, by0, by1), r))
    return out


def _overlap_tile(pixel_grid: PixelGrid, inputs, iy: int, ix: int):
    """`overlap_tile` plus, per pair, the first grid row holding a common valid pixel (the old pair order)."""
    blocks = _tile_blocks(pixel_grid, inputs, iy, ix)
    counts = {im.record.path: int(r["valid"].sum()) for im, _, r in blocks}
    pairs, first = {}, {}
    for a in range(len(blocks)):
        for b in range(a + 1, len(blocks)):
            (ia, (ax0, ax1, ay0, ay1), A), (ib, (bx0, bx1, by0, by1), B) = blocks[a], blocks[b]
            x0, x1, yy0, yy1 = max(ax0, bx0), min(ax1, bx1), max(ay0, by0), min(ay1, by1)
            if x1 <= x0 or yy1 <= yy0:
                continue
            sa = (slice(yy0 - ay0, yy1 - ay0), slice(x0 - ax0, x1 - ax0))
            sb = (slice(yy0 - by0, yy1 - by0), slice(x0 - bx0, x1 - bx0))
            both = A["valid"][sa] & B["valid"][sb]
            n = int(both.sum())
            if n == 0:
                continue
            d = ((A["image"][sa] - B["image"][sb]) / A["factor"][sa])[both].astype(np.float32)
            key = (ia.record.path, ib.record.path)
            pairs[key] = (n, d)
            first[key] = yy0 + int(np.flatnonzero(both.any(axis=1))[0])
    return counts, pairs, first


def overlap_tile(pixel_grid: PixelGrid, inputs: Sequence["psf.InputMap"], iy: int, ix: int
                 ) -> Tuple[Dict[str, int], Dict[Tuple[str, str], Tuple[int, np.ndarray]]]:
    """Pass 1 over one tile: each map's valid pixel count, and per pair of maps (n common valid pixels, float32
    differences R_i - R_j / factor). Identity: `pacs_mosaic._pair_strip` over the tile's pixels; the zero-level solve's
    medians take every common pixel of the grid (the concatenation order does not change a median)."""
    counts, pairs, _ = _overlap_tile(pixel_grid, inputs, iy, ix)
    return counts, pairs


def solve_zero_levels(paths: Sequence[str], npix: Dict[str, int], pair_n: dict, pair_d: dict) -> Dict[str, float]:
    """One constant o_i per map: weighted least squares on o_i - o_j = d_ij per connected set, weighted-mean zero.
    Identity: `pacs_mosaic.solve_zero_levels` (pacs_mosaic.py:392), global by design. The rows follow `pair_n`'s
    order."""
    index = {p: k for k, p in enumerate(paths)}
    off = {p: 0.0 for p in paths}
    if not pair_n:
        return off
    ii = np.array([index[a] for a, _ in pair_n])
    jj = np.array([index[b] for _, b in pair_n])
    graph = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(paths),) * 2)
    _, label = connected_components(graph, directed=False)
    for c in np.unique(label[np.unique(np.concatenate([ii, jj]))]):
        members = np.flatnonzero(label == c)
        local = {m: k for k, m in enumerate(members)}
        keys = [k for k in pair_n if label[index[k[0]]] == c]
        A = np.zeros((len(keys) + 1, len(members)))
        y = np.zeros(len(keys) + 1)
        sw = np.zeros(len(keys) + 1)
        for r, k in enumerate(keys):
            A[r, local[index[k[0]]]], A[r, local[index[k[1]]]] = 1.0, -1.0
            y[r], sw[r] = pair_d[k], np.sqrt(pair_n[k])
        A[-1] = [npix[paths[m]] for m in members]
        A[-1] /= A[-1].sum()
        sw[-1] = 1e3 * sw[:-1].max()
        o = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        for m, v in zip(members, o):
            off[paths[m]] = float(v)
    return off


def coadd_tile(pixel_grid: PixelGrid, inputs: Sequence["psf.InputMap"], offsets: Dict[str, float], iy: int, ix: int,
               layers: Dict[str, TiledLayer]) -> bool:
    """Pass 2 over one tile: the weighted co-add of every input map overlapping the tile (zero level removed), written
    into the four layers (open "r+"; the tile written only where it holds a covered pixel). Returns whether it did.
    Identity: `pacs_mosaic._strip` per pixel -- image = sum(w I)/sum(w), error = sqrt(sum(w^2 e^2))/sum(w), NaN/0 where
    sum(w) = 0; the same float64 sums in META map order, cast once to float32: bit-identical to the strip build."""
    ty0, ty1, tx0, tx1 = pixel_grid.tile_bounds(iy, ix)
    shape = (ty1 - ty0, tx1 - tx0)
    sw = np.zeros(shape)
    swi = np.zeros(shape)
    sw2e2 = np.zeros(shape)
    cov = np.zeros(shape)
    for im, (bx0, bx1, by0, by1), r in _tile_blocks(pixel_grid, inputs, iy, ix):
        ok = r["valid"]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(ok, 1.0 / r["smooth"] ** 2, 0.0)
        sl = (slice(by0 - ty0, by1 - ty0), slice(bx0 - tx0, bx1 - tx0))
        sw[sl] += w
        swi[sl] += w * np.where(ok, r["image"] - offsets[im.record.path] * r["factor"], 0.0)
        sw2e2[sl] += (w * np.where(ok, r["error"], 0.0)) ** 2
        cov[sl] += np.where(ok, r["coverage"], 0.0)
    has = sw > 0
    if not has.any():
        return False
    with np.errstate(divide="ignore", invalid="ignore"):
        image = np.where(has, swi / sw, np.nan)
        error = np.where(has, np.sqrt(sw2e2) / sw, np.nan)
    for name, a in (("image", image), ("error", error), ("coverage", cov), ("weight", sw)):
        layers[name][ty0:ty1, tx0:tx1] = a.astype(np.float32)
    return True


def _spec(pixel_grid: PixelGrid, header_text: str, inputs) -> tuple:
    """What a worker needs to rebuild the grid and the input maps: header texts, never pickled WCS objects (astropy
    pickles a WCS through a 16-digit header, so a pickled WCS is not bit-equal to the one the parent holds; the old
    strip build also re-read every WCS from its header text in the worker)."""
    return (header_text, pixel_grid.shape, pixel_grid.box,
            [(im.record, im.maps_root, im.smooth.directory, im.smooth_step) for im in inputs])


def _unspec(spec):
    header_text, shape, box, maps = spec
    pixel_grid = PixelGrid.from_header(header_text, shape, None, box)
    return pixel_grid, [psf.InputMap(r, grid.celestial_wcs(r.wcs_header), root, TiledLayer.open(d, "r"), step)
                        for r, root, d, step in maps]


def _pass1(spec, tiles):
    pixel_grid, inputs = _unspec(spec)
    return [_overlap_tile(pixel_grid, inputs, iy, ix) for iy, ix in tiles]


def _pass2(spec, offsets, tiles, layers):
    pixel_grid, inputs = _unspec(spec)
    return [coadd_tile(pixel_grid, inputs, offsets, iy, ix, layers) for iy, ix in tiles]


def _chunks(items: list, n: int) -> list:
    """`items` in at most 4 n contiguous pieces (fewer, larger joblib tasks: each carries the input list once)."""
    k = max(1, -(-len(items) // max(1, 4 * n)))
    return [items[i:i + k] for i in range(0, len(items), k)]


def _sample_rows(layer: TiledLayer, rows: np.ndarray) -> np.ndarray:
    """Whole grid rows `rows` of a layer, (len(rows), nx), each tile read once."""
    out = np.full((rows.size, layer.shape[1]), layer.fill, dtype=layer.dtype)
    t = layer.tile
    for iy, ix in layer.present():
        sel = np.flatnonzero((rows >= iy * t) & (rows < (iy + 1) * t))
        if sel.size:
            y0, y1, x0, x1 = layer.tile_bounds(iy, ix)
            out[sel, x0:x1] = layer.read_tile(iy, ix)[rows[sel] - iy * t, :x1 - x0]
    return out


def _floors(coverage: TiledLayer, pixel_grid: PixelGrid) -> "Floors":
    """FROZEN 12: the coverage floor = MIN_COVERAGE_FRACTION x the median positive coverage of whole rows at the
    grid's stride (`pacs_fit.mosaic_layer_plan`), and the error fraction."""
    cov = _sample_rows(coverage, pixel_grid.floor_sample_rows()).ravel()
    positive = cov[cov > 0]
    floor = grid.MIN_COVERAGE_FRACTION * (np.median(positive) if positive.size else 0.0)
    return Floors(float(floor), grid.ERROR_FLOOR_FRACTION)


def build_mosaic(config, region: str, band: str, box: Optional[Sequence[float]] = None,
                 n_jobs: Optional[int] = None) -> "Mosaic":
    """Builds the band's mosaic: input set, PixelGrid.build (FROZEN 13), each map's smoothed error (psf.
    build_smoothed_error), pass 1 per tile (joblib over the tiles any map's bbox touches), zero levels, pass 2 per tile,
    the layers' index.json (META in image/), then the mosaic PSF (psf.build_mosaic_psf) and the bright neighbours
    (META "neighbours"). Reads the index and the input FITS; writes the store above (realpath inside
    <data_root>/sky/derived; an existing store's layer, smooth and psf folders are replaced). Returns it opened.
    Identity: every covered pixel equal to `pacs_mosaic.build_mosaic`'s within 1e-6 relative (bit-identical on one
    platform); shape, WCS header text and META map order identical; absent tiles are exactly the tiles with no covered
    pixel."""
    root = maps_root(config)
    n_jobs = n_jobs or config.n_jobs
    kept, dropped = input_set(read_index(config, region), band)
    _print_input_set(region, band, kept, dropped)
    for r in kept:
        assert r.unit.strip().lower() == "jy/pixel", f"{r.path}: the map's unit is {r.unit!r}, not Jy/pixel"
    built = PixelGrid.build(kept, band, box)
    kept = [r for r in kept if _bbox_in_output(r, grid.celestial_wcs(r.wcs_header), built.wcs, built.shape) is not None]
    if not kept:
        raise ValueError(f"no {band} map of {region} overlaps the mosaic grid")
    header_text = built.header_text()
    # every later step uses the grid its stored header describes (the old build's strips, PSF and readers all re-read
    # the WCS from the header text, whose 16 significant digits need not hold the in-memory WCS's last bit)
    pixel_grid = PixelGrid.from_header(header_text, built.shape, None, box)
    ny, nx = pixel_grid.shape
    keys, group_of = [], {}
    for r in kept:
        k = group_key(config, r)
        if k not in keys:
            keys.append(k)
        group_of[r.path] = keys.index(k)
    groups = [[r.path for r in kept if group_of[r.path] == g] for g in range(len(keys))]
    outdir = mosaic_dir(config, region, band, box)
    os.makedirs(outdir, exist_ok=True)
    _assert_inside_derived(config, outdir)
    for name in LAYERS + ("smooth", "psf"):
        d = os.path.join(outdir, name)
        if os.path.lexists(d):
            assert not os.path.islink(d), f"{d} is a link"
            _assert_inside_derived(config, d)
            shutil.rmtree(d)
    pool = Parallel(n_jobs=n_jobs)
    plain = [psf.InputMap(r, grid.celestial_wcs(r.wcs_header), root) for r in kept]
    smooth_dirs = [os.path.join(outdir, _smooth_name(r)) for r in kept]
    for d in smooth_dirs:
        _assert_inside_derived(config, d)
    pool(delayed(_build_smoothed_error)(im, d) for im, d in zip(plain, smooth_dirs))     # R3: swap
    inputs = [psf.InputMap(r, im.wcs, root, TiledLayer.open(d, "r"), grid.smooth_step_px(r.pixel_scale_arcsec))
              for r, im, d in zip(kept, plain, smooth_dirs)]
    # the tiles any input footprint touches
    tiles = set()
    for im in inputs:
        box_ = map_bbox(im, pixel_grid)
        if box_ is None:
            continue
        bx0, bx1, by0, by1 = box_
        tiles.update((iy, ix) for iy in range(by0 // grid.TILE, (by1 - 1) // grid.TILE + 1)
                     for ix in range(bx0 // grid.TILE, (bx1 - 1) // grid.TILE + 1))
    tiles = sorted(tiles)
    # pass 1: valid pixel counts and pairwise differences, tile by tile
    spec = _spec(pixel_grid, header_text, inputs)
    res = [r for part in pool(delayed(_pass1)(spec, c) for c in _chunks(tiles, n_jobs)) for r in part]
    npix = {r.path: 0 for r in kept}
    pair_n, pair_vals, pair_first = {}, {}, {}
    for counts, pairs, first in res:
        for k, v in counts.items():
            npix[k] += v
        for k, (n, d) in pairs.items():
            pair_n[k] = pair_n.get(k, 0) + n
            pair_vals.setdefault(k, []).append(d)
            pair_first[k] = min(pair_first.get(k, first[k]), first[k])
    del res
    order = {r.path: i for i, r in enumerate(kept)}
    ordered = sorted(pair_n, key=lambda k: (pair_first[k] // PAIR_ORDER_ROWS, order[k[0]], order[k[1]]))
    pair_n = {k: pair_n[k] for k in ordered if pair_n[k] >= MIN_COMMON_PIXELS}
    pair_d = {k: float(np.median(np.concatenate(pair_vals[k]))) for k in pair_n}
    del pair_vals
    offsets = solve_zero_levels([r.path for r in kept], npix, pair_n, pair_d)
    beam = {r.path: _beam_fwhm_arcsec(r) for r in kept}
    to_mjy = {p: 1e3 * 1.1331 * (beam[p] / PIXEL_ARCSEC[band]) ** 2 for p in beam}       # Jy/pixel -> mJy/beam
    pairs_meta = [dict(a=a, b=b, d=pair_d[(a, b)], n=pair_n[(a, b)], residual=pair_d[(a, b)] - (offsets[a] - offsets[b]))
                  for a, b in pair_n]
    mean_conv = float(np.mean(list(to_mjy.values())))
    resid = np.array([q["residual"] for q in pairs_meta]) * mean_conv
    print(f"pacs mosaic {region} {band}: zero levels (mJy/beam) "
          + ", ".join(f"{os.path.basename(p)[:9]}:{offsets[p] * to_mjy[p]:.1f}" for p in offsets))
    print(f"  pair residual d_ij - (o_i - o_j): {len(pairs_meta)} pairs, rms "
          f"{np.sqrt(np.mean(resid ** 2)) if resid.size else 0:.1f} max {np.abs(resid).max() if resid.size else 0:.1f} "
          f"mJy/beam")
    # pass 2: the co-add, tile by tile, into the four layers
    layers = {}
    for name in LAYERS:
        d = os.path.join(outdir, name)
        _assert_inside_derived(config, d)
        layers[name] = TiledLayer.create(d, (ny, nx), np.float32, grid.LAYER_FILL[name], grid.TILE, header_text)
    pool(delayed(_pass2)(spec, offsets, c, layers) for c in _chunks(tiles, n_jobs))
    for name in LAYERS:
        layers[name].write_index()
    floors = _floors(TiledLayer.open(os.path.join(outdir, "coverage"), "r"), pixel_grid)
    reading = _READING.format(band=band, region=region, pix=PIXEL_ARCSEC[band], win=int(grid.SMOOTH_WINDOW_ARCSEC))
    maps_meta = [dict(path=r.path, group=group_of[r.path], smooth=_smooth_name(r),
                      smooth_step=grid.smooth_step_px(r.pixel_scale_arcsec),
                      window_px=grid.smooth_window_px(r.pixel_scale_arcsec), zero_level_jy_per_pixel=offsets[r.path],
                      zero_level_mjy_per_beam=offsets[r.path] * to_mjy[r.path]) for r in kept]
    meta = dict(region=region, band=band, shape=[ny, nx], window_arcsec=grid.SMOOTH_WINDOW_ARCSEC,
                pixel_arcsec=PIXEL_ARCSEC[band],
                resampler="bilinear interpolation of surface brightness, times the output pixel's own area",
                maps=maps_meta, dropped=[r.path for r, _ in dropped], groups=groups, pairs=pairs_meta,
                group_keys=[[list(c) for c in k] for k in keys], box=None if box is None else list(box),
                floors=dict(coverage=floors.coverage, error_fraction=floors.error_fraction),
                units={name: _LAYER_TEXT[name][0] for name in LAYERS},
                reading={name: _LAYER_TEXT[name][1] + " " + reading for name in LAYERS})
    _write_meta(config, outdir, meta)
    image = TiledLayer.open(os.path.join(outdir, "image"), "r")
    _build_mosaic_psf(config, region, band, header_text, (ny, nx), kept, maps_meta, image, outdir, n_jobs)  # R3: swap
    m = Mosaic(config, region, band, box)
    m.neighbours()                      # computed once, stored in META
    return m


def _write_meta(config, outdir: str, meta: dict) -> None:
    """META into image/index.json (atomic, `TiledLayer.write_index`)."""
    d = os.path.join(outdir, "image")
    _assert_inside_derived(config, d)
    layer = TiledLayer.open(d, "r+")
    layer.meta = meta
    layer.write_index()


# ----------------------------------------------------------------------------------------------------------------------
# the numerical cores (pacs_fit.py, moved unchanged; `_fit_core` gains the window origin and the grid shape)

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
    """`_box_templates` of the mixture sum_k fw[k] x subs[k] (the response tables of a mosaic's sub-groups);
    `tmp` is scratch of the same shape as the rows written."""
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
def _fit_core(img, err, oy, ox, map_ny, map_nx, subs, fw, extent, steps, tx, ty, tn, ux, uy, un, cx, cy, half,
              min_beam, r_in, r_out):
    """Weighted linear fits, one per row (pacs_fit._fit_core; `img`, `err` now a window of the grid whose pixel
    (0, 0) is grid pixel (oy, ox), the grid `map_ny` x `map_nx`; positions in grid pixels; a pixel outside the window
    counts as outside the grid). Row i: tn[i] target sources at (tx, ty), un[i] nuisance sources at (ux, uy), box
    centred on pixel (cx[i], cy[i]) of half-width `half`. The background is the HOPS/eHOPS convention: the clipped
    (IDL mmm) sky of the valid pixels in the annulus r_in..r_out (pixels) around the first target, subtracted from the
    data; every PSF column has its own mean over the same annulus subtracted; no free constant. The point-spread
    table is the mixture of `subs` with the row's fractions fw[i]. Returns per-member flux, gain, covariance and a
    validity flag."""
    n, mmax = tx.shape
    side = 2 * half + 1
    npix = side * side
    wny, wnx = img.shape
    kmax = ux.shape[1]
    nk = subs[0].shape[2]
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
                if 0 <= xx < map_nx and 0 <= yy < map_ny and 0 <= xx - ox < wnx and 0 <= yy - oy < wny:
                    v = img[yy - oy, xx - ox]
                    s = err[yy - oy, xx - ox]
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


# ----------------------------------------------------------------------------------------------------------------------
# the opened mosaic and its reader

@dataclass
class Window:
    """A rectangle of the mosaic as the measurements read it: rows y0 .. y0 + image.shape[0], columns x0 ..
    x0 + image.shape[1] of the grid (may reach outside the grid: those pixels are NaN). `image`, `error` float32 mJy
    per pixel, `error` NaN where unusable (FROZEN 12). A private buffer: injections paste into `image`."""
    y0: int
    x0: int
    image: np.ndarray
    error: np.ndarray


@dataclass(frozen=True)
class Floors:
    """FROZEN 12. The reader's error mask: `coverage` = MIN_COVERAGE_FRACTION x median positive coverage over
    `PixelGrid.floor_sample_rows()`; `error_fraction` = ERROR_FLOOR_FRACTION (applied to 1 / sqrt(weight))."""
    coverage: float
    error_fraction: float


@dataclass(frozen=True)
class Neighbours:
    """Blind bright sources of a mosaic: grid pixels x, y, flux (mJy), snr (`pacs_fit.Neighbours`)."""
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    snr: np.ndarray


@dataclass(frozen=True)
class PositionFit:
    """Forced PSF fits at n positions (`pacs_fit.PositionFit`): flux, gain, sigma (mJy), covered, n_nuisance."""
    flux: np.ndarray
    gain: np.ndarray
    sigma: np.ndarray
    covered: np.ndarray
    n_nuisance: np.ndarray


@dataclass(frozen=True)
class ApertureFit:
    """Forced aperture photometry at n positions (`pacs_fit.ApertureFit`): flux (mJy, total), sky (mJy/pixel),
    annulus_sigma (mJy), covered (>= 90 % aperture weight and >= half the annulus valid)."""
    flux: np.ndarray
    sky: np.ndarray
    annulus_sigma: np.ndarray
    covered: np.ndarray


def _reader(image, error, coverage, weight, floors: Floors) -> Tuple[np.ndarray, np.ndarray]:
    """FROZEN 12. (image, error) in mJy per pixel, float32, the error NaN where unusable: coverage at or below the
    floor, error <= 0, error <= error_fraction x the error its own weight expects (1e3 / sqrt(weight)) or the image
    not finite (`pacs_fit.mosaic_layer_strip`, bit for bit). Elementwise: any shape."""
    image = np.asarray(image, dtype=np.float32) * _MJY
    error = np.asarray(error, dtype=np.float32) * _MJY
    with np.errstate(divide="ignore", invalid="ignore"):
        good = (np.asarray(coverage) > floors.coverage) & (error > 0)
        expected = _MJY / np.sqrt(np.asarray(weight, dtype=np.float32))
        error[~good | ~(error > floors.error_fraction * expected) | ~np.isfinite(image)] = np.nan
    return image, error


class Mosaic:
    """An opened mosaic: `dir`, `grid` (PixelGrid), `cells` (CellGrid), `layers` {name: TiledLayer, mode "r"}, `meta`,
    `psf` (psf.MosaicPSF), `floors` (Floors), `record` (a MapRecord-like description: map_id, band, obsids, shape,
    wcs_header, pixel_scale_arcsec), `band`, `pixel_arcsec`, `beam_arcsec`, `aperture_fraction`, `enclosed_fraction`,
    `half` (fit box half-width, px), `table_half`. Also `wcs`, `shape`, `beam_px`, `ann_in_px`, `ann_out_px`.
    Replaces pacs_mosaic.Mosaic + pacs_fit.Tile/OpenMap/open_map/open_prepared/cached_tile/mosaic_record. Picklable
    by (config, region, band, box)."""

    def __init__(self, config, region: str, band: str, box: Optional[Sequence[float]] = None,
                 psf_tables: bool = True) -> None:
        """Opens the store (layers read-only, PSF tables memory-mapped when `psf_tables`) and computes `floors` once
        (or reads them from META). Reads: the mosaic store, the region's PACS index (obsids)."""
        self._args = (config, region, band, None if box is None else tuple(box), psf_tables)
        self.config, self.region, self.band = config, region, band
        self.box = None if box is None else tuple(box)
        self.dir = mosaic_dir(config, region, band, box)
        self.layers = {name: TiledLayer.open(os.path.join(self.dir, name), "r") for name in LAYERS}
        self.meta = self.layers["image"].meta or {}
        header_text = self.layers["image"].wcs_header
        self.wcs = grid.celestial_wcs(header_text)
        self.shape = tuple(int(v) for v in self.meta.get("shape", self.layers["image"].shape))
        # the pixel side as the old Tile took it (from the WCS; sets every aperture and box radius)
        self.pixel_arcsec = grid.pixel_scale_arcsec(self.wcs)
        self.grid = PixelGrid.from_header(header_text, self.shape, None, self.box)
        self.cells = CellGrid.from_wcs(self.wcs, self.shape)
        f = self.meta.get("floors")
        self.floors = (Floors(float(f["coverage"]), float(f["error_fraction"])) if f is not None
                       else _floors(self.layers["coverage"], self.grid))
        self.psf = _LegacyMosaicPSF(self.dir, maps_root(config), tables=psf_tables)          # R3: swap -> psf.MosaicPSF
        paths = {q["path"] for q in self.meta.get("maps", ())}
        obsids = tuple(sorted({str(o) for r in read_index(config, region) if r.path in paths for o in r.obsids}))
        self.record = MapRecord(map_id="mosaic_%s_%s" % (region.replace(" ", "_"), band), obsids=obsids, band=band,
                                path=os.path.relpath(self.dir, config.data_root) + "/mosaic", image_hdu=0, error_hdu=1,
                                coverage_hdu=2, shape=self.shape, unit="Jy/pixel", wcs_header=header_text, mode=1,
                                level="L25", mapper="mosaic", scan_speed=20.0, pixel_scale_arcsec=PIXEL_ARCSEC[band],
                                source="mosaic", polygon=())
        self.beam_arcsec = float(self.psf.meta["beam_arcsec"])
        self.aperture_fraction = float(self.psf.meta["aperture_fraction"])
        self.enclosed_fraction = float(self.psf.meta["enclosed_fraction"])
        self.beam_px = self.beam_arcsec / self.pixel_arcsec
        _, ann_in, ann_out = APERTURES[band]
        self.ann_in_px, self.ann_out_px = ann_in / self.pixel_arcsec, ann_out / self.pixel_arcsec
        self.half = int(np.ceil(max(FIT_BOX_FWHM * self.beam_px, self.ann_out_px + 1.0)))
        self.table_half = int(min(self.half + np.ceil(NEIGHBOUR_REACH_FWHM * self.beam_px),
                                  PSF_TABLE_HALF_ARCSEC // self.pixel_arcsec))
        self.reach = float(self.table_half)            # nuisance inclusion radius, pixels
        if self.psf.extent != self.table_half + 0.5:
            raise ValueError("the mosaic PSF table was built for another table half-width")
        self._neighbours = None
        self._tree = None
        self._outer_cache = None

    def __getstate__(self):
        config = self._args[0]
        fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
        proxies = [k for k, v in fields.items() if isinstance(v, MappingProxyType)]
        fields.update({k: dict(fields[k]) for k in proxies})        # a mappingproxy does not pickle
        return {"config": (type(config), fields, proxies), "args": self._args[1:]}

    def __setstate__(self, state):
        cls, fields, proxies = state["config"]
        fields.update({k: MappingProxyType(fields[k]) for k in proxies})
        self.__init__(cls(**fields), *state["args"])

    # the reader
    def window(self, y0: int, y1: int, x0: int, x1: int) -> Window:
        """The pixels [y0, y1) x [x0, x1) as the measurements read them: image x 1e3 (mJy, float32), error x 1e3 with
        NaN where coverage <= floors.coverage, error <= 0, error <= floors.error_fraction x 1e3 / sqrt(weight) or the
        image is not finite (FROZEN 12). Identity: equals the old prepared layers (`mosaic_layer_strip`) bit for bit on
        those pixels."""
        raw = [self.layers[n].window(y0, y1, x0, x1) for n in LAYERS]
        image, error = _reader(*raw, self.floors)
        return Window(int(y0), int(x0), image, error)

    def tile_window(self, iy: int, ix: int, halo: int) -> Window:
        """`window` of tile (iy, ix) plus `halo` pixels on every side."""
        t = grid.TILE
        return self.window(iy * t - halo, (iy + 1) * t + halo, ix * t - halo, (ix + 1) * t + halo)

    def present_tiles(self) -> List[Tuple[int, int]]:
        """The tiles holding any covered pixel (the image layer's present list)."""
        return self.layers["image"].present()

    def _groups(self, cx: np.ndarray, cy: np.ndarray, halo: int):
        """[(window, rows)]: the positions whose nearest pixel (cx, cy) lies in each tile, with the tile's window plus
        `halo` clipped to the grid (an origin >= 0 and at or below the positions: an exact shift, see the module
        docstring)."""
        ny, nx = self.shape
        t = grid.TILE
        with np.errstate(invalid="ignore"):
            gx = np.clip(np.where(np.isfinite(cx), cx, 0), 0, nx - 1).astype(np.int64) // t
            gy = np.clip(np.where(np.isfinite(cy), cy, 0), 0, ny - 1).astype(np.int64) // t
        key = gy * (nx // t + 1) + gx
        for k in np.unique(key):
            rows = np.flatnonzero(key == k)
            iy, ix = int(gy[rows[0]]), int(gx[rows[0]])
            y0, y1, x0, x1 = self.grid.tile_bounds(iy, ix)
            yield self.window(max(y0 - halo, 0), min(y1 + halo, ny), max(x0 - halo, 0), min(x1 + halo, nx)), rows

    # geometry
    def xy(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sky -> grid pixel (`grid.sky_to_pixel`)."""
        return grid.sky_to_pixel(self.wcs, ra_deg, dec_deg)

    def beam_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The local beam FWHM, arcsec (MosaicPSF.beam_at)."""
        x = np.asarray(x, float)
        return self.psf.beam_at(x.ravel(), np.asarray(y, float).ravel()).reshape(x.shape)

    def beam_px_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """`beam_at` in pixels."""
        return self.beam_at(x, y) / self.pixel_arcsec

    @property
    def beam_px_max(self) -> float:
        """The widest beam (pixels) the mixture holds."""
        return float(self.psf.sub_beams.max()) / self.pixel_arcsec

    @property
    def beam_px_min(self) -> float:
        """The narrowest beam (pixels) the mixture holds."""
        return float(self.psf.sub_beams.min()) / self.pixel_arcsec

    def injection_half_width(self) -> int:
        """Half-width (px) of an injected patch: int(min(table_half, half + ceil(beam_px))) (pacs_fit.py:1119); also the
        detection kernel's half-width h."""
        return int(min(self.table_half, self.half + np.ceil(self.beam_px)))

    def _radii(self) -> Tuple[float, float, float]:
        return tuple(r / self.pixel_arcsec for r in APERTURES[self.band])

    # measurements (each on a Window; positions in grid pixels)
    def aperture(self, x: np.ndarray, y: np.ndarray, window: Optional[Window] = None, chunk: int = 2048) -> ApertureFit:
        """FROZEN 8. Forced eHOPS aperture photometry: `_aperture_core` (5 x 5 sub-pixels, mmm-clipped annulus sky),
        then two passes of `_aperture_core_wing` with the source's own outer light (scaled to the previous pass's flux)
        removed from the annulus; flux = sum / EEF at the position (MosaicPSF mixture). `window`: the pixels to measure
        on (a pixel outside it counts as unusable; positions at or beyond its origin). Default: each position on its
        own tile's window plus the aperture reach. Identity: `pacs_fit.Tile.aperture` (pacs_fit.py:943) bit-identical
        on the same pixels; the numba cores moved here unchanged."""
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        n = x.size
        out = [np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)]
        if window is not None:
            groups = [(window, np.arange(n))]
        else:
            rb = int(np.ceil(self._radii()[2])) + 1
            groups = self._groups(np.floor(x + 0.5), np.floor(y + 0.5), rb + 1)
        for w, rows in groups:
            res = self._aperture_on(w, x[rows], y[rows], chunk)
            for a, v in zip(out, res):
                a[rows] = v
        return ApertureFit(*out)

    def _aperture_on(self, w: Window, x, y, chunk):
        """`pacs_fit.Tile.aperture` on window `w` (positions in grid pixels, shifted by the window origin)."""
        r_ap, r_in, r_out = self._radii()
        nsub = grid.APERTURE_SUBPIXELS
        n = x.size
        rb = int(np.ceil(r_out)) + 1
        flux = np.full(n, np.nan)
        sky = np.full(n, np.nan)
        sd = np.full(n, np.nan)
        area = np.full(n, np.nan)
        cov = np.zeros(n, bool)
        sd_eef = np.full(n, self.enclosed_fraction)
        for lo in range(0, n, chunk):
            sl = slice(lo, lo + chunk)
            xs, ys = x[sl], y[sl]
            xw, yw = xs - w.x0, ys - w.y0
            f, s_, d_, a_, c_ = _aperture_core(w.image, w.error, xw, yw, r_ap, r_in, r_out, nsub)
            mix = self.psf.weights(xs, ys)
            eef, fraction = self.psf.enclosed_at(xs, ys, mix), self.psf.aperture_fraction_at(xs, ys, mix)
            amp = np.where(c_, f / fraction, 0.0)
            if not c_.any():
                cov[sl] = c_
                continue
            model = self._outer_light(xs, ys, rb, mix)
            for _ in range(grid.OUTER_LIGHT_PASSES):
                f, s_, d_, a_, c_ = _aperture_core_wing(w.image, w.error, xw, yw, r_ap, r_in, r_out, nsub,
                                                        model, np.where(c_ & np.isfinite(amp), amp, 0.0))
                amp = np.where(c_, f / eef, 0.0)
            flux[sl], sky[sl], sd[sl], area[sl], cov[sl] = f / eef, s_, d_, a_, c_
            sd_eef[sl] = eef
        return flux, sky, sd * np.sqrt(area) / sd_eef, cov

    def _outer_pattern(self, k: int, node: int, rb: int) -> np.ndarray:
        """The unit-flux response on the (2 rb + 1)^2 box of the sub-pixel lattice (b, a) of third x third `node`, of
        sub-group `k` alone, kept in a least-recently-used cache (pacs_fit.Tile._outer_pattern)."""
        steps = OUTER_LIGHT_STEPS
        ny, nx = self.shape
        if self._outer_cache is None or self._outer_cache[0] != rb:
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
        one = np.zeros((sx.size, self.psf.n_sub))
        one[:, k] = 1.0
        resp = self.psf.evaluate_xy(sx, sy, px, py, one).astype(np.float32)
        cache[(k, node)] = resp.mean(axis=1).reshape(steps, steps, offs.size, offs.size)
        if len(cache) > OUTER_LIGHT_KEPT:
            cache.popitem(last=False)
        return cache[(k, node)]

    def _outer_light(self, x, y, rb, mix):
        """The unit-flux point-spread response on the (2 rb + 1)^2 pixel box around the pixel nearest each (x, y), the
        mixture of the sub-groups' cached patterns with each position's fractions `mix` (pacs_fit.Tile._outer_light)."""
        steps = OUTER_LIGHT_STEPS
        ny, nx = self.shape
        cx = np.floor(x + 0.5)
        cy = np.floor(y + 0.5)
        a = np.clip(np.floor((x - cx + 0.5) * steps).astype(np.int64), 0, steps - 1)
        b = np.clip(np.floor((y - cy + 0.5) * steps).astype(np.int64), 0, steps - 1)
        node = 3 * np.clip((3 * y / ny).astype(np.int64), 0, 2) + np.clip((3 * x / nx).astype(np.int64), 0, 2)
        out = np.zeros((x.size, 2 * rb + 1, 2 * rb + 1))
        for k in range(mix.shape[1]):
            for nd in np.unique(node[mix[:, k] > 0]):
                sel = np.flatnonzero((node == nd) & (mix[:, k] > 0))
                out[sel] += mix[sel, k, None, None] * self._outer_pattern(k, int(nd), rb)[b[sel], a[sel]]
        return out

    def neighbours(self) -> Neighbours:
        """The mosaic's blind bright sources (PSF-fit S/N >= NEIGHBOUR_SNR), found per tile with a halo of the filter
        reach: local maxima (window the narrowest beam) whose single-PSF fit at the peak, position refined by a 3 x 3
        parabola, reaches S/N >= NEIGHBOUR_SNR; where the local beam is wider than the window, a brighter peak within
        it removes the peak. Identity: `pacs_fit.Tile._find_neighbours` (pacs_fit.py:820): same list, sorted as there
        (row-major). Stored in META["neighbours"] after the first computation (replaces prepared/tile.pkl)."""
        if self._neighbours is not None:
            return self._neighbours
        q = self.meta.get("neighbours")
        if q is not None:
            nb = Neighbours(*(np.asarray(q[k], dtype=float) for k in ("x", "y", "flux", "snr")))
        else:
            nb = self._find_neighbours()
            self._store_neighbours(nb)
        self._neighbours = nb
        if nb.x.size:
            self._tree = cKDTree(np.c_[nb.x, nb.y])
        return nb

    def _store_neighbours(self, nb: Neighbours) -> None:
        """META["neighbours"] = the list (JSON floats round-trip exactly); written only when the store is writable."""
        d = os.path.join(self.dir, "image")
        if not os.access(d, os.W_OK):
            return
        _assert_inside_derived(self.config, d)
        self.meta["neighbours"] = {k: [float(v) for v in getattr(nb, k)] for k in ("x", "y", "flux", "snr")}
        layer = TiledLayer.open(d, "r+")
        layer.meta = self.meta
        layer.write_index()

    def _find_neighbours(self) -> Neighbours:
        r = max(1, int(round(self.beam_px_min)))         # the narrowest beam, wider ones thinned below
        ny, nx = self.shape
        h = r + 1                                         # the filter's reach and the 3 x 3 below
        cols = {k: [] for k in ("py", "px", "v0", "left", "right", "down", "up")}
        for iy, ix in self.present_tiles():
            ty0, ty1, tx0, tx1 = self.grid.tile_bounds(iy, ix)
            w = self.window(ty0 - h, ty1 + h, tx0 - h, tx1 + h)
            err = w.error
            img = np.where(np.isfinite(err), w.image, -np.inf).astype(np.float32)
            peak = (img == maximum_filter(img, size=2 * r + 1, mode="constant", cval=-np.inf))
            snr = np.zeros(img.shape, dtype=np.float32)
            np.divide(img, err, out=snr, where=np.isfinite(err))
            peak &= snr >= NEIGHBOUR_PEAK_SNR
            keep = np.zeros(img.shape, bool)
            keep[h:h + ty1 - ty0, h:h + tx1 - tx0] = True          # the tile's own pixels
            gy = np.arange(img.shape[0]) + ty0 - h
            gx = np.arange(img.shape[1]) + tx0 - h
            keep &= ((gy >= 1) & (gy <= ny - 2))[:, None] & ((gx >= 1) & (gx <= nx - 2))[None, :]   # not the grid's edge
            qy, qx = np.nonzero(peak & keep)
            cols["py"].append(qy + ty0 - h)
            cols["px"].append(qx + tx0 - h)
            cols["v0"].append(img[qy, qx].astype(float))
            cols["left"].append(np.maximum(img[qy, qx - 1], 0))
            cols["right"].append(np.maximum(img[qy, qx + 1], 0))
            cols["down"].append(np.maximum(img[qy - 1, qx], 0))
            cols["up"].append(np.maximum(img[qy + 1, qx], 0))
        if not cols["px"] or not sum(a.size for a in cols["px"]):
            return Neighbours(*(np.empty(0) for _ in range(4)))
        py, px, v0, left, right, down, up = (np.concatenate(cols[k]) for k in ("py", "px", "v0", "left", "right", "down",
                                                                               "up"))
        o = np.lexsort((px, py))                                   # row-major, as the old strip scan found them
        py, px, v0, left, right, down, up = (a[o] for a in (py, px, v0, left, right, down, up))
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

    def _nuisance(self, x, y, own_exclude=()):
        """Nearest bright neighbours (up to MAX_NUISANCE) within the reach of each position, except those within one
        beam of any exclusion point (pacs_fit.Tile._nuisance). Returns ux, uy (n, K) and counts (n,)."""
        n = x.size
        ux = np.zeros((n, MAX_NUISANCE))
        uy = np.zeros((n, MAX_NUISANCE))
        count = np.zeros(n, dtype=np.int64)
        self.neighbours()
        if self._tree is None or n == 0:
            return ux, uy, count
        dist, idx = self._tree.query(np.c_[x, y], k=MAX_NUISANCE, distance_upper_bound=self.reach)
        dist = dist.reshape(n, -1)
        idx = idx.reshape(n, -1)
        k = dist.shape[1]
        have = np.isfinite(dist)
        nx_, ny_ = self._neighbours.x, self._neighbours.y
        safe = np.where(have, idx, 0)
        beam = self.beam_px_at(x, y)[:, None]
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

    def _fit(self, x, y, use_nuisance=True, window: Optional[Window] = None, batch=512) -> PositionFit:
        """pacs_fit.Tile._fit: the weighted linear PSF fit at each position, on `window` or each position's tile window
        plus the fit box."""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = x.size
        out = [np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool),
               np.zeros(n, dtype=np.int64)]
        ny, nx = self.shape
        with np.errstate(invalid="ignore"):
            cx = np.rint(x).astype(np.int64)
            cy = np.rint(y).astype(np.int64)
        inside = np.isfinite(x) & np.isfinite(y) & (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        rows = np.flatnonzero(inside)
        groups = ([(window, rows)] if window is not None
                  else [(w, rows[r]) for w, r in self._groups(cx[rows].astype(float), cy[rows].astype(float),
                                                              self.half + 1)])
        subs = tuple(self.psf.subs)
        for w, sel in groups:
            for start in range(0, sel.size, batch):
                r = sel[start:start + batch]
                m = r.size
                if use_nuisance:
                    ux, uy, un = self._nuisance(x[r], y[r], own_exclude=[(x[r], y[r])])
                else:
                    ux, uy, un = np.zeros((m, 1)), np.zeros((m, 1)), np.zeros(m, dtype=np.int64)
                tx, ty = x[r, None], y[r, None]
                fw = self.psf.weights(tx[:, 0], ty[:, 0])
                flux, gain, cov, valid = _fit_core(w.image, w.error, w.y0, w.x0, ny, nx, subs, fw, self.psf.extent,
                                                   self.psf.subpixel_steps, tx, ty, np.ones(m, dtype=np.int64), ux, uy,
                                                   un, cx[r], cy[r], self.half, MIN_BEAM_FRACTION, self.ann_in_px,
                                                   self.ann_out_px)
                out[0][r] = flux[:, 0] * 1.0
                out[1][r] = gain[:, 0]
                out[2][r] = np.sqrt(np.maximum(cov[:, 0, 0], 0)) * 1.0
                out[3][r] = valid
                out[4][r] = un
        return PositionFit(*out)

    def psf_fit(self, x: np.ndarray, y: np.ndarray, window: Optional[Window] = None,
                use_nuisance: bool = True) -> PositionFit:
        """The weighted linear PSF fit with bright neighbours as nuisance columns (`Tile._fit`/`measure`)."""
        return self._fit(x, y, use_nuisance=use_nuisance, window=window)

    def fitted_fwhm(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Best-fit total FWHM (arcsec) of a detection (`Tile.fitted_fwhm`); EXTENDED where > 1.5 x beam: the beam
        broadened by a Gaussian of width chosen on a small grid by chi-square, two scan columns plus a constant
        background, no nuisance columns. NaN where the fit is not possible."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        beam_px = self.beam_px_at(x, y)
        half = self.half
        side = 2 * half + 1
        subs, fw = tuple(self.psf.subs), self.psf.weights(x, y)
        out = np.full(x.size, np.nan)
        tmp = np.zeros((subs[0].shape[2], side * side))
        ny, nx = self.shape
        bx = np.array([float(round(v)) if np.isfinite(v) else np.nan for v in x])
        by = np.array([float(round(v)) if np.isfinite(v) else np.nan for v in y])
        for w, rows in self._groups(bx, by, half + 1):
            for i in rows:
                if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                    continue
                cx, cy = int(round(x[i])), int(round(y[i]))
                y0, x0 = cy - half, cx - half
                if x0 < 0 or y0 < 0 or cx + half >= nx or cy + half >= ny:
                    continue
                d = w.image[y0 - w.y0:y0 - w.y0 + side, x0 - w.x0:x0 - w.x0 + side].astype(float).ravel()
                e = w.error[y0 - w.y0:y0 - w.y0 + side, x0 - w.x0:x0 - w.x0 + side].astype(float).ravel()
                ok = np.isfinite(d) & np.isfinite(e)
                if ok.sum() < 0.8 * d.size:
                    continue
                t = np.zeros_like(tmp)
                _mix_templates(subs, fw[i], self.psf.extent, self.psf.subpixel_steps, nx, ny, x[i], y[i], x0, y0, side,
                               t, 0, tmp)
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

    def near_bright(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Within one beam of a bright-neighbour peak (`Tile.near_bright`)."""
        x = np.asarray(x, float)
        self.neighbours()
        if self._tree is None:
            return np.zeros(x.shape, dtype=bool)
        d, _ = self._tree.query(np.c_[x.ravel(), np.asarray(y, float).ravel()], k=1,
                                distance_upper_bound=self.beam_px_max)
        return (d < self.beam_px_at(x, y).ravel()).reshape(x.shape)

    def pixel_sigma(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The masked error (mJy/px) at the nearest pixel, NaN off the data (`Tile.pixel_sigma`)."""
        with np.errstate(invalid="ignore"):
            cx = np.rint(np.asarray(x)).astype(np.int64)
            cy = np.rint(np.asarray(y)).astype(np.int64)
        ok = (cx >= 0) & (cx < self.shape[1]) & (cy >= 0) & (cy < self.shape[0])
        out = np.full(cx.shape, np.nan)
        if ok.any():
            raw = [self.layers[n][cy[ok], cx[ok]] for n in LAYERS]
            out[ok] = _reader(*raw, self.floors)[1]
        return out

    def template_patches(self, x: np.ndarray, y: np.ndarray, flux: np.ndarray,
                         mixture: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cx, cy, patches): the injection template (local MosaicPSF mixture, mean over components) times `flux` on
        (2h+1)^2 patches, lower-left pixels (cx, cy) (`Tile._template_patches`, pacs_fit.py:1138; evaluated in the
        same chunks of 256)."""
        chunk = 256
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        flux = np.broadcast_to(np.asarray(flux, float), x.shape)
        h = self.injection_half_width()
        side = 2 * h + 1
        offs = np.arange(side, dtype=float)
        cxs, cys, pats = [], [], []
        for lo in range(0, x.size, chunk):
            sl = slice(lo, lo + chunk)
            cx = np.rint(x[sl]).astype(np.int64) - h
            cy = np.rint(y[sl]).astype(np.int64) - h
            px = np.broadcast_to(cx[:, None, None] + offs[None, None, :], (cx.size, side, side))
            py = np.broadcast_to(cy[:, None, None] + offs[None, :, None], (cx.size, side, side))
            resp = (self.psf.evaluate_xy(x[sl], y[sl], px, py) if mixture is None
                    else self.psf.evaluate_xy(x[sl], y[sl], px, py, np.asarray(mixture)[sl]))
            cxs.append(cx)
            cys.append(cy)
            pats.append(resp.mean(axis=1) * flux[sl][:, None, None])
        if not pats:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, side, side))
        return np.concatenate(cxs), np.concatenate(cys), np.concatenate(pats)


def paste(window: Window, cx: np.ndarray, cy: np.ndarray, patches: np.ndarray) -> None:
    """Adds patches into `window.image` at grid lower-left pixels (cx, cy), clipped to the window
    (`pacs_fit._paste`, pacs_fit.py:1198; float32 addition order as there)."""
    image = window.image
    ny, nx = image.shape
    for i in range(len(cx)):
        patch = patches[i]
        c0, r0 = int(cx[i]) - window.x0, int(cy[i]) - window.y0
        side_y, side_x = patch.shape
        a0, b0 = max(c0, 0), max(r0, 0)
        a1, b1 = min(c0 + side_x, nx), min(r0 + side_y, ny)
        if a1 <= a0 or b1 <= b0:
            continue
        image[b0:b1, a0:a1] += patch[b0 - r0:b1 - r0, a0 - c0:a1 - c0].astype(np.float32)


def open_mosaic(config, region: str, band: str, box: Optional[Sequence[float]] = None) -> Mosaic:
    """The built mosaic opened (`Mosaic(config, region, band, box)`)."""
    return Mosaic(config, region, band, box)
