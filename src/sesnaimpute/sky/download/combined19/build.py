"""Combined19 (Bovy et al. 2016, mwdust): Marshall et al. 2006 in the
Galactic plane, Bayestar19 (Green et al. 2019) elsewhere it reaches, and
Drimmel et al. 2003 only where neither exists -- one maintained 3-D
extinction product, zenodo record 3566060 (`combine19.h5`, 3,932,184,404
bytes, md5 9dc6cdbf946a79e32bdaf19813c86494).

Owner ruling 2026-10-21 ("is it easier to just use the published
product?"): the file lives in the owner's own dust-map downloads area,
`config.inputs["dustmaps_data"]` (outside this data root, the one place
C26 (briefs/round4_2026-10-20/C26_profile_adopted.md) reads besides the
root), never re-fetched by this package. Where it is already present
there with the zenodo md5, this module HARD-LINKS it into
`<root>/sky/download/combined19/combine19.h5` (`os.link`: same inode,
zero extra bytes, the same convention the owner used for Planck). Where
it is absent, this module pulls it there itself, as 16 concurrent
resumable byte-range requests (the same construction
`dustmaps_data/combined19/pull_ranges.py` already used to fetch it, byte
for byte), concatenates and md5-verifies the parts, and only then links
it into the root -- so the root's own copy is never created except by a
verified file. Every other reader in this package opens the root's own
copy, like every other download line.

`sky.derived.profile` reads this root copy with h5py directly, never
importing `mwdust` at run time (its own module docstring): the
hierarchical-HEALPix pixel lookup and the linear-in-distance-modulus
interpolation this file's layout needs are `mwdust.HierarchicalHealpixMap`'s
own published algorithm, reproduced there.
"""

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run

_URL = "https://zenodo.org/records/3566060/files/combine19.h5?download=1"
_SIZE = 3_932_184_404
_MD5 = "9dc6cdbf946a79e32bdaf19813c86494"
_N_STREAMS = 16
_FILENAME = "combine19.h5"


def _md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 24)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _pull_one_part(url, path, lo, hi):
    """Bytes `[lo, hi]` (inclusive) of `url`, resumable: a part file
    already holding `hi - lo + 1` bytes is left as it is."""
    want = hi - lo + 1
    for attempt in range(8):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == want:
            return
        if have > want:
            os.remove(path)
            have = 0
        try:
            resp = requests.get(url, headers={"Range": "bytes=%d-%d" % (lo + have, hi)},
                                 stream=True, timeout=120)
            if resp.status_code != 206:
                raise RuntimeError("status %d" % resp.status_code)
            with open(path, "ab") as f:
                for block in resp.iter_content(1 << 20):
                    f.write(block)
        except Exception as exc:
            print("sky.download.combined19: part at %r attempt %d: %s" % (path, attempt, exc))
            time.sleep(20)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have != want:
        raise RuntimeError(
            "sky.download.combined19: part at %r never reached %d bytes (got %d) "
            "after 8 attempts" % (path, want, have))


def _pull_by_byte_ranges(dest_path):
    """`_SIZE` bytes of `_URL`, as `_N_STREAMS` concurrent resumable byte
    ranges, concatenated and md5-verified against the zenodo record's own
    checksum -- the same construction `dustmaps_data/combined19/
    pull_ranges.py` used, ported here so a cold dustmaps area still
    produces a root copy without a manual step."""
    parts_dir = dest_path + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    chunk = (_SIZE + _N_STREAMS - 1) // _N_STREAMS
    bounds = []
    for i in range(_N_STREAMS):
        lo = i * chunk
        hi = min((i + 1) * chunk, _SIZE) - 1
        bounds.append((lo, hi))
    part_path = lambda i: os.path.join(parts_dir, "part_%02d" % i)
    t0 = time.time()
    with ThreadPoolExecutor(_N_STREAMS) as pool:
        list(pool.map(lambda i: _pull_one_part(_URL, part_path(i), *bounds[i]), range(_N_STREAMS)))
    print("sky.download.combined19: %d streams done in %.0f s, concatenating"
          % (_N_STREAMS, time.time() - t0))
    tmp_path = dest_path + ".tmp"
    with open(tmp_path, "wb") as out:
        for i in range(_N_STREAMS):
            with open(part_path(i), "rb") as f:
                while True:
                    block = f.read(1 << 24)
                    if not block:
                        break
                    out.write(block)
    got_size = os.path.getsize(tmp_path)
    got_md5 = _md5_of(tmp_path)
    if got_size != _SIZE or got_md5 != _MD5:
        raise ValueError(
            "sky.download.combined19: pulled file fails verification: size %d (want %d), "
            "md5 %s (want %s)" % (got_size, _SIZE, got_md5, _MD5))
    os.replace(tmp_path, dest_path)
    for i in range(_N_STREAMS):
        os.remove(part_path(i))
    os.rmdir(parts_dir)


def build(config, regions=None):
    """`regions` is accepted for interface uniformity and ignored: this
    is a survey-wide, single-file product. Hard-links the owner's
    dustmaps-area copy into the root where it is already present and
    md5-verified; pulls it into the dustmaps area first (by concurrent
    byte ranges) where it is not, then links it either way -- the root's
    own copy is always a hard link (`os.link`), never a second copy of
    the 3.9 GB file."""
    dustmaps_dir = config.inputs["dustmaps_data"]
    source_path = os.path.join(dustmaps_dir, "combined19", _FILENAME)
    dest_dir = f"{config.data_root}/sky/download/combined19"
    dest_path = os.path.join(dest_dir, _FILENAME)
    with progress_module.Stage("sky.download.combined19") as st:
        if os.path.exists(dest_path):
            print("sky.download.combined19: %s present, skipped" % dest_path)
            st.done(dest_path, hard_link=True, pulled=False)
            return
        if not (os.path.exists(source_path) and os.path.getsize(source_path) == _SIZE
                and _md5_of(source_path) == _MD5):
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            print("sky.download.combined19: %s absent or unverified, pulling %d bytes "
                  "as %d concurrent byte ranges" % (source_path, _SIZE, _N_STREAMS))
            _pull_by_byte_ranges(source_path)
        os.makedirs(dest_dir, exist_ok=True)
        os.link(source_path, dest_path)
        st.done(dest_path, hard_link=True, pulled=not os.path.exists(dest_path))


if __name__ == "__main__":
    run(build)
