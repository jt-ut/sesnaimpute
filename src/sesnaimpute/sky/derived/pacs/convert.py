"""Convert a stored dense mosaic (image/error/coverage/weight .npy or .npy.zst + mosaic.hdf5) to the tile-store layout.

    python -m sesnaimpute.sky.derived.pacs.convert --root <data_root> --region R --band B [--delete-dense]
           [--box RA DEC W H] [--out-root OTHER_ROOT]
    python -m sesnaimpute.sky.derived.pacs.convert --directory <dense mosaic dir> [--out-directory DIR] [--delete-dense]

Reads `<root>/sky/derived/pacs/mosaic/<Region_safe>/<band>[__slice_...]/` (the old dense mosaic folder); writes, next to
the dense files (or under --out-root / --out-directory), the store `mosaic.Mosaic` opens (CONTRACT sec 5):
  image/ error/ coverage/ weight/  TiledLayer stores, each with the WCS header of `mosaic.hdf5` WCS_HEADER; `image/
                                   index.json` "meta" = META with each map's smoothed error renamed (`lattice` ->
                                   `smooth` = "smooth/<map>", `lattice_step` -> `smooth_step`) plus "units" and
                                   "reading" {layer: text} from the layer datasets' attributes (the layout build_mosaic
                                   writes). A layer missing from the folder is skipped. hpx512 per tile by write_index.
  smooth/<map>/                    each `_smooth/<map>.npy` as a TiledLayer (float64, fill NaN), as
                                   psf.build_smoothed_error writes it
  psf/                             `_psf/` (psf.json with the same renames; sub<k>.npy and table.npy copied, or
                                   decompressed from .npy.zst)
The reader floors and the bright-neighbour list are not in the old META: `mosaic.Mosaic` derives them on first open
(stage mosaic stores them).

Verification before any delete: each store is re-opened from disk and compared with the dense array over the WHOLE grid
by bit pattern (NaN payloads and -0.0 included, so every covered pixel is equal); shape, dtype and the WCS header text must
equal; every smoothed-error sample and every PSF table bit-equal. `--delete-dense` removes the dense layers (.npy and
.npy.zst), `_smooth/`, `_psf/` and `mosaic.hdf5` only after ALL of them verified. Idempotent: an existing store is
verified and kept, not rewritten. Prints bytes before (dense file read) and after (tile files) per layer.

Depends on: grid only. Replaces: the dense layout of the retired pacs_mosaic.py (stage mosaic converts a region's stored
dense mosaic with this module before detect).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from sesnaimpute.sky.derived.pacs.grid import LAYER_FILL, TiledLayer, celestial_wcs

LAYERS = ("image", "error", "coverage", "weight")
#: old META key -> new (CONTRACT sec 5)
_RENAMED = {"lattice": "smooth", "lattice_step": "smooth_step"}


def mosaic_folder(root: str, region: str, band: str, box: Optional[Sequence[float]] = None) -> str:
    """`<root>/sky/derived/pacs/mosaic/<Region_safe>/<band>[__slice_%.3f_%.3f_%.2fx%.2f]` (pacs_mosaic.product_dir)."""
    safe = re.sub(r"[^A-Za-z0-9+-]+", "_", region)
    return os.path.join(root, "sky", "derived", "pacs", "mosaic", safe,
                        band + ("" if box is None else "__slice_%.3f_%.3f_%.2fx%.2f" % tuple(box)))


def _read_hdf5(directory: str):
    import h5py
    with h5py.File(os.path.join(directory, "mosaic.hdf5"), "r") as f:
        wcs_header = f["WCS_HEADER"][()].decode()
        meta = json.loads(f["META"][()].decode())
        layers = {}
        for name in LAYERS:
            if name in f:
                layers[name] = {"units": str(f[name].attrs.get("UNITS", "")),
                                "reading": str(f[name].attrs.get("READING", ""))}
    return wcs_header, meta, layers


def _rename_keys(obj):
    """META with the old `lattice` keys renamed `smooth` (recursively through dicts and lists)."""
    if isinstance(obj, dict):
        return {_RENAMED.get(k, k): _rename_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename_keys(v) for v in obj]
    return obj


def _dense_files(directory: str, name: str) -> List[str]:
    return [p for p in (os.path.join(directory, name + ".npy"), os.path.join(directory, name + ".npy.zst"))
            if os.path.exists(p)]


def _open_dense(directory: str, name: str, scratch: str):
    """(memory-mapped dense array, the dense file's size, temporary file to remove or None). The .npy when present, else
    the .npy.zst decompressed (the `zstd` command) into `scratch`."""
    files = _dense_files(directory, name)
    if not files:
        return None, 0, None
    path = files[0]
    tmp = None
    if path.endswith(".zst"):
        os.makedirs(scratch, exist_ok=True)
        tmp = os.path.join(scratch, name + ".npy")
        subprocess.run(["zstd", "-dc", "-q", path, "-o", tmp, "-f"], check=True)
        path = tmp
    return np.load(path, mmap_mode="r"), os.path.getsize(files[0]), tmp


def _smooth_name(old: str) -> str:
    """The new smoothed-error folder of an old `lattice` path: "_smooth/<map>.npy" -> "smooth/<map>"."""
    head, name = os.path.split(old)
    assert head == "_smooth" and name.endswith(".npy"), "unexpected smoothed-error path %r" % old
    return os.path.join("smooth", name[:-len(".npy")])


def _same_bits(a: np.ndarray, b: np.ndarray) -> int:
    """Elements whose bits differ (shape and dtype must agree, else all)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.size, b.size, 1)
    return int(np.count_nonzero(np.ascontiguousarray(a).view(np.uint8) != np.ascontiguousarray(b).view(np.uint8)))


def _convert_smooth(directory: str, out_directory: str, log) -> int:
    """`_smooth/<map>.npy` -> `smooth/<map>/` (TiledLayer float64, fill NaN), verified bit-equal. Returns the number of
    differing samples."""
    src = os.path.join(directory, "_smooth")
    bad = 0
    for name in sorted(os.listdir(src)) if os.path.isdir(src) else ():
        if not name.endswith(".npy"):
            continue
        dense = np.load(os.path.join(src, name))
        final = os.path.join(out_directory, _smooth_name(os.path.join("_smooth", name)))
        if not os.path.exists(os.path.join(final, "index.json")):
            part = final + ".converting"
            if os.path.exists(part):
                shutil.rmtree(part)
            TiledLayer.from_dense(dense, part, fill=np.nan)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            os.replace(part, final)
        bad += _same_bits(TiledLayer.open(final, "r").to_dense(), dense)
    if os.path.isdir(src):
        log("smooth    %d smoothed-error stores, mismatching samples %d" % (len(os.listdir(src)), bad))
    return bad


def _convert_psf(directory: str, out_directory: str, scratch: str, log) -> int:
    """`_psf/` -> `psf/`: psf.json with each map's `lattice` -> `smooth` ("smooth/<map>") and `lattice_step` ->
    `smooth_step`; sub<k>.npy and table.npy copied (from the .npy, else the .npy.zst decompressed), verified bit-equal.
    Returns the number of differing table elements."""
    src = os.path.join(directory, "_psf")
    if not os.path.exists(os.path.join(src, "psf.json")):
        return 0
    dst = os.path.join(out_directory, "psf")
    with open(os.path.join(src, "psf.json")) as fh:
        meta = _rename_keys(json.load(fh))
    for q in meta["maps"]:
        q["smooth"] = _smooth_name(q["smooth"])
    names = ["sub%d" % k for k in range(meta["n_sub"])] + ["table"]
    bad = 0
    if not os.path.exists(os.path.join(dst, "psf.json")):
        part = dst + ".converting"
        if os.path.exists(part):
            shutil.rmtree(part)
        os.makedirs(part)
        for n in names:
            arr, _size, tmp = _open_dense(src, n, scratch)
            np.save(os.path.join(part, n + ".npy"), np.asarray(arr))
            del arr
            if tmp:
                os.remove(tmp)
        with open(os.path.join(part, "psf.json"), "w") as fh:
            json.dump(meta, fh)
        os.replace(part, dst)
    for n in names:
        arr, _size, tmp = _open_dense(src, n, scratch)
        bad += _same_bits(np.load(os.path.join(dst, n + ".npy")), np.asarray(arr))
        del arr
        if tmp:
            os.remove(tmp)
    log("psf       %d tables, mismatching elements %d" % (len(names), bad))
    return bad


def verify_layer(layer_dir: str, dense: np.ndarray, wcs_header: str) -> int:
    """Number of pixels (over the whole grid) whose bits differ between the stored tiles and `dense`; raises when shape,
    dtype or the WCS header differ. Reads the store fresh from disk, one tile row at a time."""
    layer = TiledLayer.open(layer_dir, "r")
    if tuple(layer.shape) != tuple(dense.shape) or np.dtype(layer.dtype) != dense.dtype:
        raise ValueError("%s: shape/dtype %s %s != %s %s" % (layer_dir, layer.shape, layer.dtype, dense.shape, dense.dtype))
    if layer.wcs_header != wcs_header:
        raise ValueError("%s: WCS header differs from mosaic.hdf5" % layer_dir)
    celestial_wcs(layer.wcs_header)
    want = LAYER_FILL[os.path.basename(os.path.normpath(layer_dir))]
    if not (layer.fill == want or (np.isnan(want) and np.isnan(layer.fill))):
        raise ValueError("%s: fill %r != %r" % (layer_dir, layer.fill, want))
    nbad = 0
    ny, nx = dense.shape
    uint = {4: np.uint32, 8: np.uint64}[dense.dtype.itemsize]
    t = layer.tile
    for y0 in range(0, ny, t):
        y1 = min(y0 + t, ny)
        a = np.ascontiguousarray(dense[y0:y1]).view(uint)
        b = np.ascontiguousarray(layer.window(y0, y1, 0, nx)).view(uint)
        nbad += int(np.count_nonzero(a != b))
    return nbad


def convert_directory(directory: str, out_directory: Optional[str] = None, delete_dense: bool = False,
                      log=print) -> Dict[str, dict]:
    """Convert every dense layer of the mosaic folder `directory` into tile stores under `out_directory` (default: the
    same folder). Returns {layer: {"before", "after", "bad", "seconds", "tiles"}}. Raises on any mismatch, deleting nothing."""
    out_directory = out_directory or directory
    wcs_header, meta, units = _read_hdf5(directory)
    meta = _rename_keys(meta)
    for q in meta.get("maps", ()):
        q["smooth"] = _smooth_name(q["smooth"])
    meta["units"] = {k: v["units"] for k, v in units.items()}
    meta["reading"] = {k: v["reading"] for k, v in units.items()}
    scratch = os.path.join(out_directory, ".convert_tmp")
    result: Dict[str, dict] = {}
    for name in LAYERS:
        dense, before, tmp = _open_dense(directory, name, scratch)
        if dense is None:
            continue
        t0 = time.time()
        final = os.path.join(out_directory, name)
        if not os.path.exists(os.path.join(final, "index.json")):
            # built under <out>/.converting/<name>: the folder's basename is the layer name, which picks its fill
            part = os.path.join(out_directory, ".converting", name)
            if os.path.exists(part):
                shutil.rmtree(part)
            os.makedirs(os.path.dirname(part), exist_ok=True)
            TiledLayer.from_dense(dense, part, wcs_header=wcs_header, meta=meta if name == "image" else None)
            os.replace(part, final)
            shutil.rmtree(os.path.dirname(part), ignore_errors=True)
        bad = verify_layer(final, dense, wcs_header)
        layer = TiledLayer.open(final, "r")
        after = layer.disk_bytes() + os.path.getsize(os.path.join(final, "index.json"))
        result[name] = {"before": before, "after": after, "bad": bad, "seconds": time.time() - t0,
                        "tiles": "%d/%d" % (len(layer.present()), layer.tile_rows * layer.tile_cols)}
        del dense
        if tmp:
            os.remove(tmp)
        log("%-9s dense %12d B -> tiles %12d B  ratio %.3f  tiles %s  mismatching pixels %d  %.1f s"
            % (name, before, after, after / max(before, 1), result[name]["tiles"], bad, result[name]["seconds"]))
    bad_smooth = _convert_smooth(directory, out_directory, log)
    bad_psf = _convert_psf(directory, out_directory, scratch, log)
    shutil.rmtree(scratch, ignore_errors=True)
    if any(r["bad"] for r in result.values()) or bad_smooth or bad_psf:
        raise RuntimeError("conversion mismatch in %s: %s; smooth %d, psf %d" % (
            directory, {k: v["bad"] for k, v in result.items()}, bad_smooth, bad_psf))
    if delete_dense and result:
        for name in result:
            for p in _dense_files(directory, name):
                os.remove(p)
        for d in ("_smooth", "_psf"):
            if os.path.isdir(os.path.join(directory, d)):
                shutil.rmtree(os.path.join(directory, d))
        os.remove(os.path.join(directory, "mosaic.hdf5"))
        log("deleted the dense layers (%s), _smooth/, _psf/ and mosaic.hdf5" % ", ".join(result))
    log("total dense %d B -> tiles %d B" % (sum(r["before"] for r in result.values()),
                                           sum(r["after"] for r in result.values())))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="the data root")
    ap.add_argument("--region")
    ap.add_argument("--band")
    ap.add_argument("--box", nargs=4, type=float, metavar=("RA", "DEC", "W", "H"))
    ap.add_argument("--out-root", help="write the stores under this root instead (same relative folder)")
    ap.add_argument("--directory", help="a dense mosaic folder, instead of --root/--region/--band")
    ap.add_argument("--out-directory")
    ap.add_argument("--delete-dense", action="store_true")
    a = ap.parse_args(argv)
    if a.directory:
        src, out = a.directory, a.out_directory
    else:
        if not (a.root and a.region and a.band):
            ap.error("give --root, --region and --band (or --directory)")
        src = mosaic_folder(a.root, a.region, a.band, a.box)
        out = mosaic_folder(a.out_root, a.region, a.band, a.box) if a.out_root else None
    convert_directory(src, out, a.delete_dense)
    return 0


if __name__ == "__main__":
    sys.exit(main())
