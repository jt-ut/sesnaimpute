"""The one primitive every download module uses to pull bytes to disk."""

import os
import shutil
import urllib.request


def fetch(url, dest_path):
    """Streams `url` to `dest_path` in 4 MB pieces (a 1.4 GB member table
    never sits whole in memory), creating the directory. A file already
    at `dest_path` is left as it is and the fetch is skipped (external
    bytes are verbatim; delete the file to fetch again). Prints one line
    with the byte count. No checksum, no manifest, no retry beyond what
    urllib does on its own.
    """
    if os.path.exists(dest_path):
        print(f"fetch: {dest_path} present, skipped")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 22)
    n_bytes = os.path.getsize(dest_path)
    print(f"fetch: {url} -> {dest_path} ({n_bytes} bytes)")
