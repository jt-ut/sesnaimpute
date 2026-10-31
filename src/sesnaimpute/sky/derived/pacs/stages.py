"""The PACS build's stages and its ONE command line:

    python -m sesnaimpute.sky.derived.pacs.stages <mosaic|psf|detect|inject|pool|finish> --config C --region R
           [--band B] [--box RA DEC W H] [--round r] [--shard k] [--of K] [--n-jobs N] [--reference-product P]
    python -m sesnaimpute.sky.derived.pacs.stages all --config C --region R [--box ...]   (every stage, one process, K = 1)
    python -m sesnaimpute.sky.derived.pacs.stages table --config C --region R [--box ...] (the stage table, printed)
    python -m sesnaimpute.sky.derived.pacs.stages CONFIG [--regions R ...] [--n-jobs N]  (the RUNBOOK line: `build`)

`run` is accepted for `all`. Without --band, mosaic/psf/detect/inject/pool run every band in turn; inject without
--round runs every round in turn (K = 1 only); pool without --shard runs shards 0 .. K-1 in turn (and the assembly).

Stages, per region (per band where marked):
    mosaic  (band)  mosaic.build_mosaic (the co-add AND its PSF, as the contract's build_mosaic does) -> mosaic store
    psf     (band)  checks.zero_image_check + checks.normalisation_check on the built mosaic and PSF; a FAIL stops
                    the build                                                   -> <band>/psf_checks.hdf5 (CHECKS rows)
    detect  (band)  detect.detect_band; inject.first_batches                   -> <band>/detect/ layers, detect.hdf5
    inject  (band, round, shard of K)  inject.round_tasks[shard::K] -> inject.inject_task
                                                                                -> <band>/inject_r<r>_s<k>of<K>.hdf5
    pool    (band, shard of K)  inject.pool_tasks[shard::K] -> inject.pool_block
                                                                                -> <band>/pool_s<k>of<K>.hdf5
            and, by whichever invocation finds all K shards done, the band's assembly -> <band>/band.hdf5
    finish          photometry.merge_sources, forced_photometry, cross_match; products.write_*; the CHECKS rows (psf
                    rows, the real-map rows of the injection records, checks.chance_check, checks.region_checks) into
                    the completeness product; then, ONLY once the three products verify against their receipts, the
                    band work directories (records, detection layers) are removed.

The stage table. Every stage invocation (one band, one round, one shard) has a ROW FILE
`<work>/stages/<band>__<stage>[_r<r>][_s<k>of<K> | _assembly].json` (atomic json: stage, band, round, shard, of = K,
path, started, finished, code = the checkout's commit (+ "-dirty"), host, pid, n_jobs, outputs, note), written when
the stage starts and again when it ends. A row is DONE when its file says finished and every output verifies: a record
file against `<record>.receipt.json` (size and sha256, written after the record's atomic rename and re-read from disk
then -- the lost-write detection), a directory (mosaic store, detection layers) or a product against the manifest or
receipt kept in the row. Work is skipped where the row is done, so every stage is idempotent and a killed job resumes
at its first row not done. One file per row: K shards on K nodes never write the same file.
Mixed K: an injection round (or the pooling) whose shards were cut with another K is refused; different K in different
rounds is allowed (records and counts do not depend on K: FROZEN 2-3). After finish only the mosaic rows and the
finish row remain.

Work directory `<data_root>/sky/derived/pacs/work/<Region_safe>[__slice_...]/` (realpath inside <data_root>/sky/derived).
No pickles; no markers; no prepared copies.

Replaces: pacs_stages.py, pacs_stages_cli.py, pacs_completeness.build/preflight/finish_region/_region_inputs/_git_state,
pacs_photometry.build, and the RUNBOOK entries (`pacs_completeness`, `pacs_photometry` lines -> one line
`PY sesnaimpute.sky.derived.pacs.stages`). Removed: run_region's separate path, the preflight duplicate of prepare,
claim-finish, the plan CLI (driver.py plans).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from astropy.wcs import WCS

from sesnaimpute import build as build_module  # noqa: F401 -- registers the stage's joblib pool (build/__init__.py)
from sesnaimpute import config as config_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import regions as regions_module
from sesnaimpute.catalog import curated
from sesnaimpute.definitions import PACS_BANDS as BANDS
from sesnaimpute.definitions import PACS_MAX_EXTRA_BATCHES as MAX_EXTRA_BATCHES
from sesnaimpute.sky.derived.pacs import checks, detect, grid, inject, mosaic, photometry, products, psf
from sesnaimpute.sky.download.herschel_pacs.index import read_index

#: the stages, in order
STAGES = ("mosaic", "psf", "detect", "inject", "pool", "finish")
#: the stages run per band
BAND_STAGES = ("mosaic", "psf", "detect", "inject", "pool")
#: stages whose work is cut into K shards
SHARDED = ("inject", "pool")
#: the command-line verbs
VERBS = STAGES + ("all", "run", "table")
#: a record file's receipt is `<record>` + this
RECEIPT_SUFFIX = ".receipt.json"
#: the pooling rule of the build (None = `inject.PoolingRule.legacy()`, main 0683690's; round 3 sets the ruled one)
POOLING_RULE = None
#: the shard number of a band's assembly row (stage pool)
ASSEMBLY = -1


@dataclass(frozen=True)
class Receipt:
    """What a finished record file must be: `path`, `size` (bytes), `sha256`; written as `<path>.receipt.json` after the
    file's atomic rename."""
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class StageRow:
    """One row of the stage table: stage, band ("" for region-wide), round (-1 if none), shard (-1 if none; stage
    "pool" with shard -1 = the band's assembly), of (K), path, done (finished and every output matches its receipt),
    problem ("" or what is wrong: not run, started not finished, missing, no receipt, empty, size, checksum)."""
    stage: str
    band: str
    round: int
    shard: int
    of: int
    path: str
    done: bool
    problem: str


class StageError(RuntimeError):
    """A stage cannot run or did not complete (an earlier stage not done, mixed K, a failed check, a lost write)."""


# ----------------------------------------------------------------------------------------------------------------------
# places

def _safe(region: str, box: Optional[Sequence[float]]) -> str:
    """The region's folder name, as the mosaic store names it (`pacs_mosaic.product_dir`)."""
    return re.sub(r"[^A-Za-z0-9+-]+", "_", region) + ("" if box is None else "__slice_%.3f_%.3f_%.2fx%.2f" % tuple(box))


def _inside(path: str, root: str) -> str:
    """`path`, asserting its realpath lies inside `root`'s (scratch roots are views: never write through a link)."""
    real, top = os.path.realpath(path), os.path.realpath(root)
    if not real.startswith(top + os.sep):
        raise StageError("%s resolves to %s, outside %s" % (path, real, top))
    return path


def region_inputs(config, region: str) -> tuple:
    """(ra, dec) float64 of the region's SESNA sources in curated order (`catalog.curated.read_positions`)."""
    _name, ra, dec = curated.read_positions(config, region)
    return np.asarray(ra, np.float64), np.asarray(dec, np.float64)


def work_dir(config, region: str, box: Optional[Sequence[float]] = None, band: Optional[str] = None) -> str:
    """The region's (band's) work directory: <data_root>/sky/derived/pacs/work/<safe>[__slice_...][/<band>]
    (realpath inside <data_root>/sky/derived)."""
    d = os.path.join(config.data_root, "sky", "derived", "pacs", "work", _safe(region, box))
    if band is not None:
        d = os.path.join(d, band)
    return _inside(d, os.path.join(config.data_root, "sky", "derived"))


def _rows_dir(config, region, box):
    return os.path.join(work_dir(config, region, box), "stages")


def _record_name(stage, r, shard, of):
    if stage == "inject":
        return "inject_r%d_s%dof%d.hdf5" % (r, shard, of)
    if stage == "pool":
        return "band.hdf5" if shard == ASSEMBLY else "pool_s%dof%d.hdf5" % (shard, of)
    return {"psf": "psf_checks.hdf5", "detect": "detect.hdf5"}[stage]


def _row_name(stage, band, r, shard, of):
    name = "%s__%s" % (band or "region", stage)
    if r >= 0:
        name += "_r%d" % r
    if stage in SHARDED:
        name += "_assembly" if shard == ASSEMBLY else "_s%dof%d" % (shard, of)
    return name + ".json"


# ----------------------------------------------------------------------------------------------------------------------
# files: atomic writes, receipts, lost-write detection

def _sha256_file(path: str) -> Tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            n += len(b)
            h.update(b)
    return n, h.hexdigest()


def _fsync_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(d: str) -> None:
    try:
        _fsync_path(d)
    except OSError:          # a file system that refuses a directory fsync; the rename itself is atomic
        pass


def _place(tmp: str, path: str) -> Tuple[int, str]:
    """`tmp` made durable, renamed onto `path` and re-read from disk: returns (size, sha256); raises, removing the file,
    when what lies at `path` is not what was written or is empty (Stampede3: a 296 MB file renamed without fsync read
    back 280 MiB)."""
    _fsync_path(tmp)
    size, sha = _sha256_file(tmp)
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(os.path.abspath(path)))
    size2, sha2 = _sha256_file(path)
    if size == 0 or (size2, sha2) != (size, sha):
        try:
            os.remove(path)
        except OSError:
            pass
        raise StageError("lost write: %s is %d bytes (%s...) after its rename; %d bytes (%s...) were written"
                         % (path, size2, sha2[:12], size, sha[:12]))
    return size, sha


def _write_json(path: str, obj) -> None:
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(os.path.abspath(path)))


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_record(path: str, tables: Dict[str, np.ndarray], attrs: Optional[dict] = None) -> Receipt:
    """Writes an HDF5 record file atomically (temporary name with pid, fsync, os.replace), then its receipt (size,
    sha256; also atomic). Returns the receipt. `tables` names may hold "/" (groups); `attrs` (json-able) are stored as
    the file's RECORD_ATTRS. The file is re-read after the rename and compared (lost-write detection); a stale receipt
    is removed before anything is written, so it can never vouch for a new file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    receipt_path = path + RECEIPT_SUFFIX
    if os.path.exists(receipt_path):
        os.remove(receipt_path)
    tmp = "%s.tmp%d" % (path, os.getpid())
    with h5py.File(tmp, "w") as f:
        for name, value in tables.items():
            a = np.asarray(value)
            if a.dtype.kind == "U":
                a = np.char.encode(a, "utf-8")
            if a.dtype.kind == "O":
                raise TypeError("write_record: %s is an object array" % name)
            f.create_dataset(name, data=a)
        f.attrs["RECORD_ATTRS"] = json.dumps(attrs or {}, sort_keys=True)
    size, sha = _place(tmp, path)
    _write_json(receipt_path, dict(path=os.path.basename(path), size=size, sha256=sha))
    return Receipt(path, size, sha)


def _read_record(path: str) -> Tuple[Dict[str, np.ndarray], dict]:
    problem = verify(path)
    if problem:
        raise StageError("%s: %s" % (path, problem))
    tables = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                tables[name] = obj[()]
        f.visititems(visit)
        attrs = json.loads(f.attrs["RECORD_ATTRS"])
    return tables, attrs


def read_record(path: str) -> Dict[str, np.ndarray]:
    """Reads a record file after `verify`; raises when the record does not match its receipt."""
    return _read_record(path)[0]


def verify(path: str) -> str:
    """"" when the record exists and matches its receipt, else the problem (missing, no receipt, empty, size,
    checksum)."""
    if not os.path.exists(path):
        return "missing"
    if not os.path.exists(path + RECEIPT_SUFFIX):
        return "no receipt"
    return _verify_file(path, _read_json(path + RECEIPT_SUFFIX))


def _verify_file(path: str, receipt: dict) -> str:
    if not os.path.isfile(path):
        return "missing"
    size = os.path.getsize(path)
    if size == 0:
        return "empty"
    if size != receipt["size"]:
        return "size (%d bytes, receipt %d)" % (size, receipt["size"])
    if _sha256_file(path)[1] != receipt["sha256"]:
        return "checksum"
    return ""


def _manifest(d: str) -> dict:
    """The receipt of a directory: every file's (relative path, size, sha256), their total size, one sha256 over the
    list (temporary files excluded)."""
    files = []
    for root, dirs, names in os.walk(d):
        dirs.sort()
        rel_root = os.path.relpath(root, d)
        for n in sorted(names):
            if ".tmp" in n:
                continue
            size, sha = _sha256_file(os.path.join(root, n))
            files.append([os.path.normpath(os.path.join(rel_root, n)), size, sha])
    files.sort()
    h = hashlib.sha256("".join("%s\0%d\0%s\n" % tuple(x) for x in files).encode())
    return dict(path=d, size=int(sum(x[1] for x in files)), sha256=h.hexdigest(), files=files)


def _verify_dir(d: str, receipt: dict) -> str:
    if not os.path.isdir(d):
        return "missing"
    now = _manifest(d)
    if now["sha256"] == receipt["sha256"]:
        return ""
    have = {x[0]: x for x in now["files"]}
    for rel, size, sha in receipt["files"]:
        if rel not in have:
            return "missing %s" % rel
        if have[rel][1] != size:
            return "size of %s (%d bytes, receipt %d)" % (rel, have[rel][1], size)
        if have[rel][2] != sha:
            return "checksum of %s" % rel
    return "files added: %s" % sorted(set(have) - {x[0] for x in receipt["files"]})[:3]


def _verify_output(out: dict) -> str:
    if out["kind"] == "record":
        return verify(out["path"])
    if out["kind"] == "dir":
        return _verify_dir(out["path"], out["receipt"])
    return _verify_file(out["path"], out["receipt"])


# ----------------------------------------------------------------------------------------------------------------------
# objects <-> record tables (no pickles): the dataclasses of the modules before this one, arrays, numpy and python
# scalars, None, dicts (keys: scalars or tuples of them), lists, tuples, WCS (as its header text, refused unless exact)

def _dataclass_registry() -> dict:
    out = {}
    for module in (grid, psf, mosaic, detect, inject, photometry, products, checks):
        for value in vars(module).values():
            if isinstance(value, type) and dataclasses.is_dataclass(value):
                out["%s.%s" % (value.__module__, value.__qualname__)] = value
    return out


_DATACLASSES = _dataclass_registry()


def _wcs_header(w) -> str:
    """A WCS's header text, refused unless the WCS it parses back to is bit-equal (cell ids depend on it)."""
    text = w.to_header_string(relax=True)
    back = grid.celestial_wcs(text)
    a, b = w.wcs, back.wcs
    same = (list(a.ctype) == list(b.ctype) and np.array_equal(a.crval, b.crval) and np.array_equal(a.crpix, b.crpix)
            and np.array_equal(a.cdelt, b.cdelt) and np.array_equal(a.get_pc(), b.get_pc())
            and a.lonpole == b.lonpole and a.latpole == b.latpole)
    if not same:
        raise ValueError("a WCS does not survive its header text: it cannot be stored in a record")
    return text


def _encode(obj, name: str, tables: dict):
    """The schema node of `obj`; its arrays go to `tables` under `name`."""
    if obj is None or isinstance(obj, (bool, str)) or (isinstance(obj, (int, float)) and not isinstance(obj, np.generic)):
        return {"t": "py", "v": obj}
    if isinstance(obj, np.generic):
        tables[name] = np.asarray(obj)
        return {"t": "scalar"}
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            raise TypeError("record: %s is an object array" % name)
        if obj.dtype.kind == "U":
            tables[name] = np.char.encode(obj, "utf-8")
            return {"t": "array", "unicode": True}
        tables[name] = obj
        return {"t": "array"}
    if isinstance(obj, WCS):
        return {"t": "wcs", "header": _wcs_header(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        key = "%s.%s" % (cls.__module__, cls.__qualname__)
        if _DATACLASSES.get(key) is not cls:
            raise TypeError("record: %s is a %s, not a dataclass of the PACS package" % (name, key))
        return {"t": "dc", "cls": key,
                "f": {f.name: _encode(getattr(obj, f.name), "%s/%s" % (name, f.name), tables) for f in dataclasses.fields(obj)}}
    if isinstance(obj, dict):
        keys = list(obj)
        return {"t": "dict", "k": [_encode_key(k) for k in keys],
                "v": [_encode(obj[k], "%s/k%d" % (name, i), tables) for i, k in enumerate(keys)]}
    if isinstance(obj, (list, tuple)):
        return {"t": "tuple" if isinstance(obj, tuple) else "list",
                "v": [_encode(x, "%s/%d" % (name, i), tables) for i, x in enumerate(obj)]}
    raise TypeError("record: %s has type %s" % (name, type(obj).__name__))


def _encode_key(k):
    if isinstance(k, tuple):
        return {"tuple": [_encode_key(x) for x in k]}
    if isinstance(k, np.generic):
        k = k.item()
    if k is None or isinstance(k, (bool, int, float, str)):
        return {"v": k}
    raise TypeError("record: a dict key of type %s" % type(k).__name__)


def _decode_key(k):
    return tuple(_decode_key(x) for x in k["tuple"]) if "tuple" in k else k["v"]


def _decode(node, name: str, tables: dict):
    t = node["t"]
    if t == "py":
        return node["v"]
    if t == "scalar":
        return tables[name][()]
    if t == "array":
        a = tables[name]
        return np.char.decode(a, "utf-8") if node.get("unicode") else a
    if t == "wcs":
        return grid.celestial_wcs(node["header"])
    if t == "dc":
        cls = _DATACLASSES[node["cls"]]
        values = {f: _decode(v, "%s/%s" % (name, f), tables) for f, v in node["f"].items()}
        init = {f.name for f in dataclasses.fields(cls) if f.init}
        obj = cls(**{f: v for f, v in values.items() if f in init})
        for f, v in values.items():
            if f not in init:
                object.__setattr__(obj, f, v)
        return obj
    if t == "dict":
        return {_decode_key(k): _decode(v, "%s/k%d" % (name, i), tables) for i, (k, v) in enumerate(zip(node["k"], node["v"]))}
    items = [_decode(v, "%s/%d" % (name, i), tables) for i, v in enumerate(node["v"])]
    return tuple(items) if t == "tuple" else items


def _write_object(path: str, obj, attrs: Optional[dict] = None) -> Receipt:
    """`obj` written as a record: its arrays under "obj/...", its schema as the uint8 dataset "schema" (json)."""
    tables: dict = {}
    schema = _encode(obj, "obj", tables)
    tables["schema"] = np.frombuffer(json.dumps(schema).encode(), np.uint8)
    return write_record(path, tables, attrs)


def _read_object(path: str):
    """(object, attrs) of a record written by `_write_object`."""
    tables, attrs = _read_record(path)
    schema = json.loads(tables.pop("schema").tobytes().decode())
    return _decode(schema, "obj", tables), attrs


def _check_tables(rows: Sequence, prefix: str = "checks") -> Dict[str, np.ndarray]:
    """CHECKS rows (`products.CheckRow`) as columns."""
    text = lambda f: np.array([getattr(r, f).encode("utf-8") for r in rows], "S") if rows else np.zeros(0, "S1")
    num = lambda f: np.array([getattr(r, f) for r in rows], np.float64)
    out = {prefix + "/" + f: text(f) for f in ("name", "band", "verdict", "detail")}
    out.update({prefix + "/" + f: num(f) for f in ("value", "bar_low", "bar_high")})
    return out


def _check_rows(tables: dict, prefix: str = "checks") -> list:
    if prefix + "/name" not in tables:
        return []
    s = lambda k: [x.decode("utf-8") for x in np.atleast_1d(tables[prefix + "/" + k])]
    f = lambda k: [float(x) for x in np.atleast_1d(tables[prefix + "/" + k])]
    return [products.CheckRow(*row) for row in zip(s("name"), s("band"), f("value"), f("bar_low"), f("bar_high"),
                                                    s("verdict"), s("detail"))]


# ----------------------------------------------------------------------------------------------------------------------
# the stage table

@functools.lru_cache(maxsize=1)
def _code_state() -> str:
    """The checkout's commit (+ "-dirty" with uncommitted changes to tracked files), or "unknown". Recorded, not
    enforced."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        git = lambda *a: subprocess.run(("git", "-C", here) + a, capture_output=True, text=True, timeout=30)
        head = git("rev-parse", "HEAD")
        if head.returncode:
            return "unknown"
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip())
        return head.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass(frozen=True)
class _Key:
    stage: str
    band: str
    round: int
    shard: int
    of: int


def _key(stage, band="", round_=None, shard=-1, of=1) -> _Key:
    if stage == "inject":
        return _Key(stage, band, int(round_), int(shard), int(of))
    if stage == "pool":
        return _Key(stage, band, -1, int(shard), int(of))
    if stage == "finish":
        return _Key(stage, "", -1, -1, 1)
    return _Key(stage, band, -1, -1, 1)


_FINISH = _Key("finish", "", -1, -1, 1)


def _row_path(config, region, box, key: _Key) -> str:
    return os.path.join(_rows_dir(config, region, box), _row_name(key.stage, key.band, key.round, key.shard, key.of))


def _primary_path(config, region, box, key: _Key) -> str:
    if key.stage == "mosaic":
        return mosaic.mosaic_dir(config, region, key.band, box)
    if key.stage == "finish":
        return products.completeness_path(config, region)
    return os.path.join(work_dir(config, region, box, key.band), _record_name(key.stage, key.round, key.shard, key.of))


def _row_state(config, region, box, key: _Key) -> StageRow:
    path = _row_path(config, region, box, key)
    mk = lambda p, done, problem: StageRow(key.stage, key.band, key.round, key.shard, key.of, p, done, problem)
    if not os.path.exists(path):
        try:
            primary = _primary_path(config, region, box, key)
        except NotImplementedError:
            primary = ""
        return mk(primary, False, "not run")
    row = _read_json(path)
    key = _Key(row["stage"], row["band"], row["round"], row["shard"], row["of"])      # the assembly's K is the file's
    if not row.get("finished"):
        return mk(row["path"], False, "started %s on %s (pid %s), not finished" % (row["started"], row["host"], row["pid"]))
    for out in row["outputs"]:
        problem = _verify_output(out)
        if problem:
            return mk(out["path"], False, problem)
    return mk(row["path"], True, "")


def _row_begin(config, region, box, key: _Key, path: str, n_jobs) -> dict:
    os.makedirs(_rows_dir(config, region, box), exist_ok=True)
    row = dict(stage=key.stage, band=key.band, round=key.round, shard=key.shard, of=key.of, path=path,
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), finished=None, code=_code_state(),
               host=socket.gethostname(), pid=os.getpid(), n_jobs=n_jobs, outputs=[], note="")
    _write_json(_row_path(config, region, box, key), row)
    return row


def _row_end(config, region, box, key: _Key, row: dict, outputs: list, note: str = "") -> None:
    _write_json(_row_path(config, region, box, key),
                dict(row, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), outputs=outputs, note=note))


def _row_info(config, region, box, key: _Key) -> Optional[dict]:
    """The row file's content (started, finished, code, host, K, outputs, note), or None."""
    path = _row_path(config, region, box, key)
    return _read_json(path) if os.path.exists(path) else None


def _row_keys(config, region, box) -> List[_Key]:
    d = _rows_dir(config, region, box)
    keys = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if name.endswith(".json") and ".tmp" not in name:
                row = _read_json(os.path.join(d, name))
                keys.append(_Key(row["stage"], row["band"], row["round"], row["shard"], row["of"]))
    return keys


def _band_has_maps(config, region: str, band: str) -> bool:
    """Whether the region's map index holds a map of the band (`sky.download.herschel_pacs.index.read_index`)."""
    return any(r.band == band for r in read_index(config, region))


def stage_table(config, region: str, box: Optional[Sequence[float]] = None) -> List[StageRow]:
    """Every stage (and shard) of the region with its state, from the row files, the records and their receipts: the
    rows run so far plus, until finish is done, the rows every region has (mosaic, psf, detect and the assembly per
    band; finish). Inject and pool shard rows appear once they have started."""
    keys = set(_row_keys(config, region, box))
    if not _row_state(config, region, box, _FINISH).done:
        for b in BANDS:
            keys.update(_key(s, b) for s in ("mosaic", "psf", "detect"))
            if not any(k.stage == "pool" and k.band == b and k.shard == ASSEMBLY for k in keys):
                keys.add(_Key("pool", b, -1, ASSEMBLY, 0))
    keys.add(_FINISH)
    order = lambda k: (STAGES.index(k.stage) + (0.5 if k.stage == "pool" and k.shard == ASSEMBLY else 0),
                       BANDS.index(k.band) if k.band in BANDS else 9, k.round, k.of, k.shard)
    return [_row_state(config, region, box, k) for k in sorted(keys, key=order)]


def format_table(rows: Sequence[StageRow]) -> str:
    """The stage table as text (one line per row)."""
    out = ["%-7s %-5s %5s %9s  %-4s  %s" % ("stage", "band", "round", "shard/K", "done", "problem or path")]
    for r in rows:
        shard = "assembly" if (r.stage == "pool" and r.shard == ASSEMBLY) else ("%d/%d" % (r.shard, r.of) if r.shard >= 0 else "")
        out.append("%-7s %-5s %5s %9s  %-4s  %s" % (r.stage, r.band or "-", r.round if r.round >= 0 else "", shard,
                                                   "yes" if r.done else "no", r.problem or r.path))
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------------
# the band's state between stages

def _require(config, region, box, key: _Key, what: str) -> None:
    row = _row_state(config, region, box, key)
    if not row.done:
        raise StageError("%s %s: stage %s is not done (%s)" % (region, key.band, what, row.problem))


def _band_empty(config, region, box, band) -> bool:
    """True where stage mosaic found no map of the band."""
    info = _row_info(config, region, box, _key("mosaic", band))
    return bool(info) and not info["outputs"]


_DETECT_CACHE: dict = {}


def _detection(config, region, band, box):
    """(BandDetection or None, attrs) of the band's detect record (cached by the record's checksum)."""
    path = os.path.join(work_dir(config, region, box, band), "detect.hdf5")
    if not os.path.exists(path + RECEIPT_SUFFIX):
        raise StageError("%s: stage detect of %s %s has not run" % (path, region, band))
    k = (path, _read_json(path + RECEIPT_SUFFIX)["sha256"])
    if k not in _DETECT_CACHE:
        _DETECT_CACHE.clear()
        _DETECT_CACHE[k] = _read_object(path)
    return _DETECT_CACHE[k]


def _shard_rows(config, region, box, band, stage, r=-1) -> Dict[int, List[_Key]]:
    """{K: [row keys]} of a band's started shards of a stage (round r for inject)."""
    out: Dict[int, List[_Key]] = {}
    for k in _row_keys(config, region, box):
        if k.stage == stage and k.band == band and k.shard != ASSEMBLY and (stage != "inject" or k.round == r):
            out.setdefault(k.of, []).append(k)
    return out


def _refuse_mixed_k(config, region, box, band, stage, r, of) -> None:
    other = sorted(set(_shard_rows(config, region, box, band, stage, r)) - {of})
    if other:
        what = "injection round %d" % r if stage == "inject" else "the pooling"
        raise StageError("%s %s: %s was started with K = %s; refusing K = %d (mixed K: finish it with the same K, or "
                         "remove its rows and records)" % (region, band, what, ", ".join(map(str, other)), of))


def _round_counts(config, region, box, band, q) -> np.ndarray:
    """The (rows x pairs) injection counts of round q, summed over its K shards (every shard must be done)."""
    shards = _shard_rows(config, region, box, band, "inject", q)
    if len(shards) != 1:
        raise StageError("%s %s: injection round %d has %s" % (region, band, q, "no shard rows" if not shards else "mixed K"))
    K = next(iter(shards))
    total = None
    for s in range(K):
        key = _Key("inject", band, q, s, K)
        row = _row_state(config, region, box, key)
        if not row.done:
            raise StageError("%s %s: injection round %d shard %d of %d is not done (%s)" % (region, band, q, s, K, row.problem))
        tables, attrs = _read_record(row.path)
        if total is None:
            total = np.zeros(tuple(attrs["count_shape"]), np.int64)
        np.add.at(total, (tables["counts/row"], tables["counts/pair"]), tables["counts/n"])
    return total


def _round_tasks(config, region, band, r, box=None) -> list:
    """The ordered pieces of injection round r of the band ([] = complete): `inject.round_tasks` on the detect record
    and the counts of every earlier round's shards (all must be done)."""
    det, attrs = _detection(config, region, band, box)
    if det is None or r > MAX_EXTRA_BATCHES:
        return []
    return inject.round_tasks(det, int(attrs["first"]), r, lambda q: _round_counts(config, region, box, band, q))


def _round_task_count(config, region, band, r, box=None) -> int:
    return len(_round_tasks(config, region, band, r, box))


def _injections_complete(config, region, box, band) -> int:
    """The number of injection rounds run; raises unless every one is complete and the next has no tasks."""
    r = 0
    while _shard_rows(config, region, box, band, "inject", r):
        r += 1
    if _round_task_count(config, region, band, r, box):
        raise StageError("%s %s: injection round %d has tasks but has not run: the injections are not complete"
                         % (region, band, r))
    return r


def _pooling_rule():
    return inject.PoolingRule.legacy() if POOLING_RULE is None else POOLING_RULE


def _held_and_stats(det, band_dir: str) -> Tuple[dict, dict]:
    """The held-out half's injections and the band's statistics from its injection records (what the old
    `_write_pool_table` returned beside the table). R3: contract addition -- the record order, the half split and the
    statistics are inject's, so inject.py must provide `held_and_stats(detection, records_dir)`; until then this
    raises."""
    fn = getattr(inject, "held_and_stats", None)
    if fn is None:
        raise NotImplementedError("R3: inject.held_and_stats(detection, records_dir) is not in the contract yet")
    return fn(det, band_dir)


def _band_completeness(config, region, band, box):
    """The band's BandCompleteness from band.hdf5, its detection from detect.hdf5."""
    bc, attrs = _read_object(os.path.join(work_dir(config, region, box, band), "band.hdf5"))
    if not attrs.get("empty"):
        bc.detection = _detection(config, region, band, box)[0]
    return bc


# ----------------------------------------------------------------------------------------------------------------------
# the stages: each returns (outputs, note); an output is {"kind": record | dir | file, "path", "receipt" (dir, file)}

def _mosaic_complete(d: str) -> bool:
    return (all(os.path.exists(os.path.join(d, layer, "index.json")) for layer in mosaic.LAYERS)
            and os.path.exists(os.path.join(d, "psf", "psf.json")))


def _stage_mosaic(config, region, box, key, n_jobs, **_):
    if not _band_has_maps(config, region, key.band):
        return [], "the region has no map of the band"
    d = _inside(mosaic.mosaic_dir(config, region, key.band, box),
                os.path.join(config.data_root, "sky", "derived", "pacs", "mosaic"))
    note = ""
    if _mosaic_complete(d):
        note = "the existing mosaic store adopted"
    else:
        if os.path.exists(d):
            shutil.rmtree(d)            # a build killed before its index files is not a product
        mosaic.build_mosaic(config, region, key.band, box=box, n_jobs=n_jobs)
        if not _mosaic_complete(d):
            raise StageError("%s: build_mosaic returned without the four layers' index.json and psf/psf.json" % d)
    return [dict(kind="dir", path=d, receipt=_manifest(d))], note


def _stage_psf(config, region, box, key, n_jobs, **_):
    _require(config, region, box, _key("mosaic", key.band), "mosaic")
    if _band_empty(config, region, box, key.band):
        return [], "the region has no map of the band"
    m = mosaic.open_mosaic(config, region, key.band, box)
    rows = list(checks.zero_image_check(config, m)) + list(checks.normalisation_check(config, m, n_jobs))
    for r in rows:
        print("check %-24s %-4s %12.6g  bar [%g, %g]  %s  %s" % (r.name, r.band, r.value, r.bar_low, r.bar_high,
                                                                 r.verdict, r.detail), flush=True)
    failed = sorted({r.name for r in rows if r.verdict == "FAIL"})
    if failed:
        raise StageError("%s %s: build-time check(s) failed: %s" % (region, key.band, ", ".join(failed)))
    path = _primary_path(config, region, box, key)
    write_record(path, _check_tables(rows), dict(stage="psf", band=key.band))
    return [dict(kind="record", path=path)], ""


def _stage_detect(config, region, box, key, n_jobs, **_):
    _require(config, region, box, _key("psf", key.band), "psf")
    band_dir = work_dir(config, region, box, key.band)
    os.makedirs(band_dir, exist_ok=True)
    path = _primary_path(config, region, box, key)
    ra, dec = region_inputs(config, region)
    empty = _band_empty(config, region, box, key.band)
    det = None if empty else detect.detect_band(config, region, key.band, box, ra, dec, band_dir, n_jobs)
    if det is None:
        _write_object(path, None, dict(stage="detect", band=key.band, empty=True, n_src=int(ra.size), first=0))
        return [dict(kind="record", path=path)], "no map of the band" if empty else "the band's mosaic reaches no source"
    first = inject.first_batches(config, mosaic.open_mosaic(config, region, key.band, box), det)
    _write_object(path, det, dict(stage="detect", band=key.band, empty=False, n_src=int(det.n_src), first=int(first)))
    outputs = [dict(kind="record", path=path)]
    layers = os.path.join(band_dir, "detect")
    if os.path.isdir(layers):
        outputs.append(dict(kind="dir", path=layers, receipt=_manifest(layers)))
    return outputs, ""


def _concat_records(parts: Sequence[dict]) -> Tuple[Dict[str, np.ndarray], list, float]:
    """The pieces' outputs as one table: every array column concatenated in piece order (scalars as one-element
    columns); CHECKS rows (`checks`) collected; `seconds` summed."""
    columns: Dict[str, list] = {}
    rows, seconds = [], 0.0
    for p in parts:
        for k, v in p.items():
            if k == "seconds":
                seconds += float(sum(v.values())) if isinstance(v, dict) else float(np.sum(v))
            elif k == "checks":
                rows.extend(v or [])
            else:
                columns.setdefault(k, []).append(np.atleast_1d(np.asarray(v)))
    return {k: np.concatenate(v) for k, v in columns.items()}, rows, seconds


def _stage_inject(config, region, box, key, n_jobs, tasks=None, **_):
    det, _attrs = _detection(config, region, key.band, box)
    m = mosaic.open_mosaic(config, region, key.band, box)
    mine = tasks[key.shard::key.of]
    done = pieces_module.run(functools.partial(inject.inject_task, config, m, det), mine, n_jobs,
                             unit="injection pieces (round %d, shard %d of %d)" % (key.round, key.shard, key.of))
    table, rows, seconds = _concat_records(done)
    counts = np.asarray(inject.record_counts(table, int(np.asarray(det.cells.ids).size)))
    row, pair = np.nonzero(counts)
    tables = {"records/" + k: v for k, v in table.items()}
    tables.update({"counts/row": row.astype(np.int64), "counts/pair": pair.astype(np.int64),
                   "counts/n": counts[row, pair].astype(np.int64)})
    tables.update(_check_tables(rows))
    path = _primary_path(config, region, box, key)
    write_record(path, tables, dict(stage="inject", band=key.band, round=key.round, shard=key.shard, K=key.of,
                                    n_tasks=len(mine), seconds=seconds, count_shape=list(counts.shape)))
    return [dict(kind="record", path=path)], "%d of the round's %d pieces" % (len(mine), len(tasks))


def _stage_pool(config, region, box, key, n_jobs, **_):
    _require(config, region, box, _key("detect", key.band), "detect")
    det, _attrs = _detection(config, region, key.band, box)
    path = _primary_path(config, region, box, key)
    if det is None:
        _write_object(path, [], dict(stage="pool", band=key.band, shard=key.shard, K=key.of, n_blocks=0))
        return [dict(kind="record", path=path)], "empty band"
    _injections_complete(config, region, box, key.band)
    _mag, fgrid = inject.flux_grid()
    rule = _pooling_rule()
    mine = inject.pool_tasks(det)[key.shard::key.of]
    fn = functools.partial(inject.pool_block, det, work_dir(config, region, box, key.band), rule=rule, flux_grid_mjy=fgrid)
    done = pieces_module.run(fn, mine, n_jobs, unit="pooling blocks (shard %d of %d)" % (key.shard, key.of))
    _write_object(path, list(done), dict(stage="pool", band=key.band, shard=key.shard, K=key.of, n_blocks=len(mine),
                                         rule=repr(rule)))
    return [dict(kind="record", path=path)], ""


def _stage_assembly(config, region, box, key, n_jobs, **_):
    """The band's assembly from its K pooling shards (all done), the blocks back in `pool_tasks` order."""
    det, attrs = _detection(config, region, key.band, box)
    mag, fgrid = inject.flux_grid()
    path = _primary_path(config, region, box, key)
    if det is None:
        bc = inject.empty_band(int(attrs["n_src"]), int(np.asarray(mag).size))
        _write_object(path, bc, dict(stage="assembly", band=key.band, K=key.of, empty=True))
        return [dict(kind="record", path=path)], "empty band"
    K = key.of
    n_blocks = len(inject.pool_tasks(det))
    pooled: list = [None] * n_blocks
    for s in range(K):
        blocks, _a = _read_object(os.path.join(work_dir(config, region, box, key.band), _record_name("pool", -1, s, K)))
        idx = list(range(s, n_blocks, K))
        if len(blocks) != len(idx):
            raise StageError("%s %s: pooling shard %d of %d holds %d blocks, not %d" % (region, key.band, s, K, len(blocks), len(idx)))
        for i, b in zip(idx, blocks):
            pooled[i] = b
    held, stats = _held_and_stats(det, work_dir(config, region, box, key.band))
    bc = inject.assemble_band(det, pooled, held, stats, fgrid)
    _write_object(path, dataclasses.replace(bc, detection=None),            # the detection lives in detect.hdf5
                  dict(stage="assembly", band=key.band, K=K, empty=False, rule=repr(_pooling_rule())))
    return [dict(kind="record", path=path)], ""


def _write_product(path: str, writer) -> dict:
    """`writer(tmp)` writes a product at a temporary path beside `path`; it is made durable, renamed into place and
    re-read (lost-write detection). Returns its receipt."""
    tmp = "%s.tmp%d" % (path, os.getpid())
    if os.path.exists(tmp):
        os.remove(tmp)
    writer(tmp)
    size, sha = _place(tmp, path)
    return dict(path=path, size=size, sha256=sha)


def _assembly_k(config, region, box, band) -> int:
    """The K the band was assembled from (0 when it was not)."""
    info = _row_info(config, region, box, _Key("pool", band, -1, ASSEMBLY, 0))
    return int(info["of"]) if info else 0


def _stage_finish(config, region, box, key, n_jobs, reference_product=None, **_):
    for b in BANDS:
        _require(config, region, box, _Key("pool", b, -1, ASSEMBLY, _assembly_k(config, region, box, b)),
                 "pool (assembly) of " + b)
    bands = {b: _band_completeness(config, region, b, box) for b in BANDS}
    mosaics = {b: mosaic.open_mosaic(config, region, b, box) for b in BANDS if not _band_empty(config, region, box, b)}
    mag, fgrid = inject.flux_grid()
    paths = dict(completeness=products.completeness_path(config, region), sources=products.sources_path(config, region),
                 photometry=products.photometry_path(config, region))
    for p in paths.values():
        _inside(p, os.path.join(config.data_root, "sky", "derived"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
    ra, dec = region_inputs(config, region)
    sources = photometry.merge_sources(config, bands, mosaics, fgrid, n_jobs)
    phot = photometry.forced_photometry(config, region, mosaics, bands, ra, dec, n_jobs)
    phot, counts = photometry.cross_match(phot, sources, ra, dec, mosaics)
    receipts = {
        "completeness": _write_product(paths["completeness"], lambda t: products.write_completeness(t, region, mag, bands)),
        "sources": _write_product(paths["sources"], lambda t: products.write_sources(t, region, bands, sources)),
        "photometry": _write_product(paths["photometry"], lambda t: products.write_photometry(t, region, phot))}
    rows = []
    for b in BANDS:
        d = work_dir(config, region, box, b)
        if os.path.exists(os.path.join(d, "psf_checks.hdf5")):
            rows += _check_rows(read_record(os.path.join(d, "psf_checks.hdf5")))
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else ():
            if re.fullmatch(r"inject_r\d+_s\d+of\d+\.hdf5", name):
                rows += _check_rows(read_record(os.path.join(d, name)))
    rows += list(checks.chance_check(counts))
    rows += list(checks.region_checks(config, region, bands, box, reference_product))

    def with_checks(tmp):
        shutil.copyfile(paths["completeness"], tmp)
        products.write_checks(tmp, rows)
    receipts["completeness"] = _write_product(paths["completeness"], with_checks)
    outputs = [dict(kind="file", path=paths[k], receipt=receipts[k]) for k in ("completeness", "sources", "photometry")]
    return outputs, "%d CHECKS rows, %d FAIL" % (len(rows), sum(r.verdict == "FAIL" for r in rows))


def _clear_work(config, region, box) -> None:
    """The band work directories (records, receipts, detection layers) and the transient stages' rows removed; the
    mosaic rows and the finish row stay. Called only when the finish row is done (its products verified)."""
    root = os.path.join(config.data_root, "sky", "derived", "pacs", "work")
    for b in BANDS:
        d = work_dir(config, region, box, b)
        if os.path.exists(d):
            shutil.rmtree(_inside(d, root))
    for k in _row_keys(config, region, box):
        if k.stage not in ("mosaic", "finish"):
            os.remove(_row_path(config, region, box, k))
    _DETECT_CACHE.clear()


_RUNNERS = {"mosaic": _stage_mosaic, "psf": _stage_psf, "detect": _stage_detect, "inject": _stage_inject,
            "pool": _stage_pool, "finish": _stage_finish}


def _label(region, key: _Key) -> str:
    s = "%s %s %s" % (key.stage, region, key.band)
    if key.round >= 0:
        s += " round %d" % key.round
    if key.stage in SHARDED:
        s += " assembly" if key.shard == ASSEMBLY else " shard %d of %d" % (key.shard, key.of)
    return s


def _run(config, stage, region, band=None, box=None, round_=None, shard=0, of=1, n_jobs=None,
         reference_product=None) -> StageRow:
    if stage not in STAGES:
        raise ValueError("unknown stage %r" % stage)
    if stage in BAND_STAGES and band not in BANDS:
        raise ValueError("stage %s needs a band of %s" % (stage, BANDS))
    if stage == "inject" and round_ is None:
        raise ValueError("stage inject needs a round")
    if stage in SHARDED and not (of >= 1 and 0 <= shard < of):
        raise ValueError("shard %d is outside 0 .. %d" % (shard, of - 1))
    n_jobs = n_jobs or config.n_jobs
    key = _key(stage, band or "", round_, shard, of)
    if stage in SHARDED:
        _refuse_mixed_k(config, region, box, key.band, stage, key.round, of)
    row = _row_state(config, region, box, key)
    extra = {}
    if not row.done and stage == "inject":
        _require(config, region, box, _key("detect", key.band), "detect")
        extra["tasks"] = _round_tasks(config, region, key.band, key.round, box)     # raises unless earlier rounds are done
        if not extra["tasks"]:
            print("stage %s: no tasks, the band's injections are complete" % _label(region, key), flush=True)
            return StageRow(stage, key.band, key.round, shard, of, "", True, "")
    if not row.done:
        info = _row_begin(config, region, box, key, row.path, n_jobs)
        print("stage %s: start (%s)" % (_label(region, key), row.problem), flush=True)
        t0 = time.time()
        outputs, note = _RUNNERS[stage](config, region, box, key, n_jobs, reference_product=reference_product, **extra)
        _row_end(config, region, box, key, info, outputs, note)
        row = _row_state(config, region, box, key)
        print("stage %s: done %.0f s %s" % (_label(region, key), time.time() - t0, note), flush=True)
        if not row.done:
            raise StageError("%s: written but does not verify (%s)" % (row.path, row.problem))
    if stage == "pool":
        _assemble_if_ready(config, region, box, key.band, of, n_jobs)
    if stage == "finish":
        _clear_work(config, region, box)
    return row


def _assemble_if_ready(config, region, box, band, K, n_jobs) -> Optional[StageRow]:
    """The band's assembly, when every one of its K pooling shards is done and the assembly is not."""
    other = _assembly_k(config, region, box, band)
    if other and other != K:
        if _row_state(config, region, box, _Key("pool", band, -1, ASSEMBLY, other)).done:
            raise StageError("%s %s: the band was assembled from K = %d pooling shards; refusing K = %d" % (region, band, other, K))
    akey = _Key("pool", band, -1, ASSEMBLY, K)
    if _row_state(config, region, box, akey).done:
        return None
    if not all(_row_state(config, region, box, _Key("pool", band, -1, s, K)).done for s in range(K)):
        return None
    info = _row_begin(config, region, box, akey, _primary_path(config, region, box, akey), n_jobs)
    outputs, note = _stage_assembly(config, region, box, akey, n_jobs)
    _row_end(config, region, box, akey, info, outputs, note)
    print("stage %s: assembled from %d shards %s" % (_label(region, akey), K, note), flush=True)
    return _row_state(config, region, box, akey)


def run_stage(config, stage: str, region: str, band: Optional[str] = None, box: Optional[Sequence[float]] = None,
              round_: Optional[int] = None, shard: int = 0, of: int = 1, n_jobs: Optional[int] = None) -> StageRow:
    """Runs one stage (one shard) unless its stage row is done; returns its row. Seeds and records do not depend on
    (shard, of, n_jobs) (FROZEN 2-3). A pool shard assembles the band once all K shards are done; finish clears the
    band work directories once its products verify. An injection round without tasks writes nothing (row done)."""
    return _run(config, stage, region, band, box, round_, shard, of, n_jobs)


def run_region(config, region: str, box: Optional[Sequence[float]] = None, n_jobs: Optional[int] = None,
               reference_product: Optional[str] = None) -> None:
    """Every stage of the region in one process (K = 1): what the RUNBOOK line and a laptop build run. Identity: the
    products equal those of main 0683690's `pacs_completeness.build` + `finish_region` (except the added CHECKS).
    Every band's mosaic and its checks first (a P160 failure is minutes in, not after the P70 hour), then per band
    detect, the injection rounds and the pooling; then finish. A rerun resumes at the first row not done."""
    if _row_state(config, region, box, _FINISH).done:
        print("stages %s: finish is done and its products verify; nothing to run" % region, flush=True)
        return
    for b in BANDS:
        _run(config, "mosaic", region, b, box, n_jobs=n_jobs)
        _run(config, "psf", region, b, box, n_jobs=n_jobs)
    for b in BANDS:
        _run(config, "detect", region, b, box, n_jobs=n_jobs)
        _inject_rounds(config, region, b, box, n_jobs)
        _run(config, "pool", region, b, box, shard=0, of=1, n_jobs=n_jobs)
    _run(config, "finish", region, box=box, n_jobs=n_jobs, reference_product=reference_product)


def _inject_rounds(config, region, band, box, n_jobs) -> None:
    """Rounds 0 .. MAX_EXTRA_BATCHES of the band in this process (K = 1), to the first without tasks."""
    for r in range(MAX_EXTRA_BATCHES + 1):
        row = _run(config, "inject", region, band, box, round_=r, shard=0, of=1, n_jobs=n_jobs)
        if not row.path:
            break


def build(config, regions: Optional[Sequence[str]] = None, box: Optional[Sequence[float]] = None) -> None:
    """The RUNBOOK entry (`build.run(build)`): `run_region` for each region in turn (regions sequential)."""
    names = regions or [r.name for r in regions_module.REGIONS]
    t0 = time.time()
    for region in names:
        run_region(config, region, box)
    print("sky.derived.pacs.stages: %d region(s) %.0f s" % (len(names), time.time() - t0), flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# the command line

def _parser():
    p = argparse.ArgumentParser(prog="python -m sesnaimpute.sky.derived.pacs.stages", description=__doc__.split("\n\n")[0])
    p.add_argument("verb", choices=VERBS)
    p.add_argument("--config", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--band", choices=BANDS)
    p.add_argument("--box", type=float, nargs=4, default=None, metavar=("RA", "DEC", "W", "H"))
    p.add_argument("--round", type=int, default=None, dest="round_")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=1, help="the number of shards K (default 1)")
    p.add_argument("--n-jobs", type=int, default=None, help="pool size for this invocation, over the config's [run] n_jobs")
    p.add_argument("--reference-product", default=None, help="finish: an earlier photometry product (C4)")
    return p


def _load(path, n_jobs):
    config = config_module.load(path)
    if n_jobs is not None:
        config = dataclasses.replace(config, n_jobs=n_jobs)
        print("stages: n_jobs=%d from --n-jobs" % n_jobs, flush=True)
    return config


def _build_main(argv) -> int:
    p = argparse.ArgumentParser(prog="python -m sesnaimpute.sky.derived.pacs.stages CONFIG")
    p.add_argument("config")
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    args = p.parse_args(argv)
    build(_load(args.config, args.n_jobs), args.regions)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The command line of the module docstring. Exit 0 done (or already done), 1 failure, 2 usage."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in VERBS and not argv[0].startswith("-"):
        return _build_main(argv)
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if args.of < 1 or (args.shard is not None and not 0 <= args.shard < args.of):
        print("usage: --of must be >= 1 and --shard in 0 .. --of - 1", file=sys.stderr)
        return 2
    if args.verb == "inject" and args.round_ is None and args.of != 1:
        print("usage: inject without --round runs every round in turn, only with --of 1", file=sys.stderr)
        return 2
    config = _load(args.config, args.n_jobs)
    box = None if args.box is None else tuple(args.box)
    region, bands = args.region, ([args.band] if args.band else list(BANDS))
    try:
        if args.verb == "table":
            print(format_table(stage_table(config, region, box)))
        elif args.verb in ("all", "run"):
            run_region(config, region, box, reference_product=args.reference_product)
        elif args.verb in ("mosaic", "psf", "detect"):
            for b in bands:
                _run(config, args.verb, region, b, box)
        elif args.verb == "inject":
            for b in bands:
                if args.round_ is None:
                    _inject_rounds(config, region, b, box, None)
                else:
                    _run(config, "inject", region, b, box, round_=args.round_, shard=args.shard or 0, of=args.of)
        elif args.verb == "pool":
            for b in bands:
                for s in ([args.shard] if args.shard is not None else range(args.of)):
                    _run(config, "pool", region, b, box, shard=s, of=args.of)
        else:
            _run(config, "finish", region, box=box, reference_product=args.reference_product)
    except Exception as e:            # noqa: BLE001 -- the job's exit code carries the stage's failure
        traceback.print_exc()
        print("stages %s %s: FAILED: %s" % (args.verb, region, e), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
