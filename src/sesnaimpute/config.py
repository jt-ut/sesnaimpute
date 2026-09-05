"""Loads the installation config: the data root and the external input paths.

Reads `root.cfg`'s `[output] data_root` and `[inputs]` section with
configparser. A `paths` section (the internal tree layout under `data_root`)
is not built yet; only `product_path` below encodes it.
"""

import configparser
from dataclasses import dataclass
from types import MappingProxyType

from sesnaimpute import definitions

_AREAS = ("sky/download", "sky/derived", "catalog", "bms")


@dataclass(frozen=True)
class Config:
    data_root: str
    inputs: MappingProxyType


def load(path):
    """Reads `path` (an ini file in the shape of root.cfg) into a Config."""
    parser = configparser.ConfigParser()
    parser.read(path)
    data_root = parser["output"]["data_root"]
    inputs = MappingProxyType(dict(parser["inputs"]))
    return Config(data_root=data_root, inputs=inputs)


def product_path(config, area, source, quantity, granule):
    """Returns `<data_root>/<area>/<source>/<quantity>_<source>_<granule>.hdf5`.

    `area` is one of "sky/download", "sky/derived", "catalog", "bms".
    `granule` must be one of `definitions.GRANULES`.
    """
    if area not in _AREAS:
        raise ValueError(f"product_path: unknown area {area!r}, must be one of {_AREAS}")
    if granule not in definitions.GRANULES:
        raise ValueError(f"product_path: unknown granule {granule!r}, must be one of {definitions.GRANULES}")
    filename = f"{quantity}_{source}_{granule}.hdf5"
    return f"{config.data_root}/{area}/{source}/{filename}"
