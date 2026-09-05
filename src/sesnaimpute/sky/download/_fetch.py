"""The one primitive every download module uses to pull bytes to disk."""

import os
import urllib.request


def fetch(url, dest_path):
    """Streams `url` to `dest_path`, creating the directory, overwriting
    whatever is there. Prints one line with the byte count. No checksum,
    no manifest, no retry beyond what urllib does on its own.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out:
        n_bytes = out.write(response.read())
    print(f"fetch: {url} -> {dest_path} ({n_bytes} bytes)")
