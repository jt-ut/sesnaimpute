"""The SESNA completeness curves, Rob Gutermuth's own measured per-field,
per-band maps and cubes (COMPLETENESS_2026-10-20.md "The products"): a
90%-completeness magnitude map and a 29-plane differential-completeness
cube per field and band, pulled verbatim from the owner's Google Drive.
Download never computes: this module copies bytes (`rclone copy`) and then
checks the copy against the remote's own listing -- the per-region
file-count/size check the earlier Spitzer-coverage-mask download lacked
(gdown's silent 50-files-per-folder cap, DATA.md's `spitzer_coverage` row).

rclone's shared Google API client id is being retired during 2026
(https://rclone.org/drive/#making-your-own-client-id); before that happens
the owner puts the package's own client id into the `gdrive` remote by
hand, once (`rclone config update gdrive client_id <id> client_secret
<secret>`) -- this module never writes to rclone's own config.
"""

import os
import re
import subprocess

from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run

#: the read-only Drive remote the owner authorizes once by hand
#: (`rclone config create gdrive drive scope=drive.readonly`).
_REMOTE = "gdrive"
_REMOTE_ROOT = f"{_REMOTE}:SESNA_2026"
#: both the 90%-magnitude map (`*_diffcomp90.fits`) and the 29-plane
#: completeness cube (`*_diffcompcube.fits`) match this one glob.
_INCLUDE = "*diffcomp*.fits"


def _require_rclone():
    """Raises with the one-time setup commands if rclone is not installed or
    the `gdrive` remote is not configured; this module never runs either
    command itself."""
    if subprocess.run(["which", "rclone"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "sky.download.sesna_completeness: rclone is not installed. Run once: "
            "brew install rclone")
    remotes = subprocess.run(["rclone", "listremotes"], capture_output=True,
                              text=True, check=True).stdout.split()
    if f"{_REMOTE}:" not in remotes:
        raise RuntimeError(
            "sky.download.sesna_completeness: no rclone remote named "
            f"'{_REMOTE}'. Run once: "
            f"rclone config create {_REMOTE} drive scope=drive.readonly")


def _region_names(regions):
    return list(regions) if regions else [r.name for r in regions_module.REGIONS]


def _copy(dest_dir, names, st):
    """`rclone copy`, restricted per region to that region's diffcomp FITS,
    checksummed against the remote, 8 parallel transfers. rclone's own
    `--stats-one-line` progress percentage is relayed through `st.tick`."""
    os.makedirs(dest_dir, exist_ok=True)
    includes = []
    for name in names:
        includes += ["--include", f"{name}/{_INCLUDE}"]
    cmd = ["rclone", "copy", _REMOTE_ROOT, dest_dir, *includes,
           "--checksum", "--transfers", "8",
           "--stats", "10s", "--stats-one-line", "-v"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        pct = re.search(r"(\d+)%", line)
        if pct:
            st.tick(int(pct.group(1)), 100, "%")
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"sky.download.sesna_completeness: rclone copy exited {returncode}")


def _remote_listing(name):
    """{file name: size in bytes} for one region's diffcomp FITS on the
    remote, from `rclone lsf`."""
    out = subprocess.run(
        ["rclone", "lsf", "-R", "--files-only", "--format", "sp",
         f"{_REMOTE_ROOT}/{name}", "--include", _INCLUDE],
        capture_output=True, text=True, check=True).stdout
    listing = {}
    for line in out.splitlines():
        if not line:
            continue
        size, path = line.split(";", 1)
        listing[path] = int(size)
    return listing


def _local_listing(dest_dir, name):
    """The same mapping read off disk at `dest_dir/<name>/`."""
    region_dir = f"{dest_dir}/{name}"
    listing = {}
    if os.path.isdir(region_dir):
        for fname in os.listdir(region_dir):
            if "diffcomp" in fname and fname.endswith(".fits"):
                listing[fname] = os.path.getsize(f"{region_dir}/{fname}")
    return listing


def check_listing(config, regions=None):
    """Compares, per region, the remote's diffcomp file count and sizes
    against what is on disk at `<data_root>/sky/download/sesna_completeness/`;
    raises naming the region and the files on any mismatch. Read-only: lists
    the remote and reads local file sizes, writes nothing. Returns
    `{region: (n_files, total_bytes)}`."""
    dest_dir = f"{config.data_root}/sky/download/sesna_completeness"
    names = _region_names(regions)
    counts = {}
    for name in names:
        remote = _remote_listing(name)
        local = _local_listing(dest_dir, name)
        if remote != local:
            missing = sorted(set(remote) - set(local))
            extra = sorted(set(local) - set(remote))
            resized = sorted(f for f in set(remote) & set(local) if remote[f] != local[f])
            raise RuntimeError(
                f"sky.download.sesna_completeness: {name} listing mismatch -- "
                f"missing {missing}, extra {extra}, size-mismatched {resized}")
        counts[name] = (len(remote), sum(remote.values()))
    return counts


def _region_present(dest_dir, name):
    """True if `dest_dir/<name>/` holds at least one diffcomp FITS and every
    one is non-empty; decided from the local disk alone."""
    local = _local_listing(dest_dir, name)
    return bool(local) and all(size > 0 for size in local.values())


def build(config, regions=None):
    """Skips every region whose diffcomp FITS are already under
    `sky/download/sesna_completeness/` (decided from the disk alone, no
    rclone). Pulls only the missing regions, then runs `check_listing` on
    them; prints each region's file count and byte total."""
    names = _region_names(regions)
    dest_dir = f"{config.data_root}/sky/download/sesna_completeness"
    counts = {}
    missing = []
    for name in names:
        if _region_present(dest_dir, name):
            local = _local_listing(dest_dir, name)
            counts[name] = (len(local), sum(local.values()))
            print(f"sky.download.sesna_completeness: {name}: present, skipped")
        else:
            missing.append(name)
    with progress_module.Stage("sky.download.sesna_completeness") as st:
        if missing:
            _require_rclone()
            _copy(dest_dir, missing, st)
            counts.update(check_listing(config, regions=missing))
        for name in names:
            n_files, n_bytes = counts[name]
            print(f"sky.download.sesna_completeness: {name}: {n_files} files, {n_bytes} bytes")
        st.done(dest_dir, regions=len(names),
                files=sum(c[0] for c in counts.values()),
                bytes=sum(c[1] for c in counts.values()))


if __name__ == "__main__":
    run(build)
