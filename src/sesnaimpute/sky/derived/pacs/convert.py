"""Convert a stored dense mosaic (image/error/coverage/weight .npy or .npy.zst + mosaic.hdf5) to the tile-store layout.

    python -m sesnaimpute.sky.derived.pacs.convert --root <data_root> --region R --band B [--delete-dense]
           [--box RA DEC W H] [--out-root OTHER_ROOT]
    python -m sesnaimpute.sky.derived.pacs.convert --directory <dense mosaic dir> [--out-directory DIR] [--delete-dense]

Reads `<root>/sky/derived/pacs/mosaic/<Region_safe>/<band>[__slice_...]/` (the folder of `pacs_mosaic.product_dir`); writes
`image/ error/ coverage/ weight/` TiledLayer stores next to the dense files (or under --out-root / --out-directory), each
with the WCS header of `mosaic.hdf5` WCS_HEADER; `image/index.json` "meta" = META (the old `lattice` keys renamed `smooth`)
plus "layers": {name: {"units", "reading"}} from the layer datasets' attributes. A layer missing from the folder is skipped.
hpx512 per tile is written by `TiledLayer.write_index`.

Verification before any delete: each store is re-opened from disk and compared with the dense array over the WHOLE grid
by bit pattern (NaN payloads and -0.0 included, so every covered pixel is equal); shape, dtype and the WCS header text must
equal. `--delete-dense` removes the layer's .npy and .npy.zst files only after ALL present layers verified. `_smooth/`,
`_psf/` and `mosaic.hdf5` are not touched. Idempotent: an existing store is verified and kept, not rewritten.
Prints bytes before (dense file read) and after (tile files) per layer. Not run on TACC.

Depends on: grid only. Replaces: the dense layer files of sky/derived/pacs_mosaic.py (round 3 switches the readers).
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
    meta["layers"] = units
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
    shutil.rmtree(scratch, ignore_errors=True)
    if any(r["bad"] for r in result.values()):
        raise RuntimeError("conversion mismatch in %s: %s" % (directory, {k: v["bad"] for k, v in result.items()}))
    if delete_dense and result:
        for name in result:
            for p in _dense_files(directory, name):
                os.remove(p)
        log("deleted the dense files of %s" % ", ".join(result))
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
