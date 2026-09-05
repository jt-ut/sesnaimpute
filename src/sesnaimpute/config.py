"""Loads the installation config: the data root and the external input paths.

Reads `root.cfg`'s `[output] data_root` and `[inputs]` section with
configparser, plus the `[run] n_jobs` worker count (CODING_RULES.md 10a's
cap, default 4 where the key is absent). A `paths` section (the internal
tree layout under `data_root`) is not built yet; only `product_path` below
encodes it.
"""

import configparser
from dataclasses import dataclass
from types import MappingProxyType

from sesnaimpute import definitions

_AREAS = ("sky/download", "sky/derived", "catalog", "granules", "bms")

#: CODING_RULES.md 10a: cap joblib worker pools at 4 where root.cfg names
#: no `[run] n_jobs`.
DEFAULT_N_JOBS = 4


@dataclass(frozen=True)
class Config:
    data_root: str
    inputs: MappingProxyType
    n_jobs: int


def load(path):
    """Reads `path` (an ini file in the shape of root.cfg) into a Config."""
    parser = configparser.ConfigParser()
    parser.read(path)
    data_root = parser["output"]["data_root"]
    inputs = MappingProxyType(dict(parser["inputs"]))
    n_jobs = parser.getint("run", "n_jobs", fallback=DEFAULT_N_JOBS)
    return Config(data_root=data_root, inputs=inputs, n_jobs=n_jobs)


def product_path(config, area, source, quantity, granule, region=None):
    """Returns the one place a product lives:

        <data_root>/<area>/<source>/<quantity>_<source>_<granule>.hdf5

    or, for a per-region product, with `__<Region>` before the extension,
    the region name verbatim. `area` is one of "sky/download", "sky/derived",
    "catalog", "granules", "bms". `granule` must be one of
    `definitions.GRANULES`. Every build writes here and every reader reads
    here; nothing else encodes the tree.
    """
    if area not in _AREAS:
        raise ValueError(f"product_path: unknown area {area!r}, must be one of {_AREAS}")
    if granule not in definitions.GRANULES:
        raise ValueError(f"product_path: unknown granule {granule!r}, must be one of {definitions.GRANULES}")
    stem = f"{quantity}_{source}_{granule}"
    if region is not None:
        stem = f"{stem}__{region}"
    return f"{config.data_root}/{area}/{source}/{stem}.hdf5"
