"""Canonical layout for the coherent model-register product.

Registers are derived products with one lifecycle.  They do not inherit
their storage locations from the independently curated model libraries
they describe.
"""

import os
from types import MappingProxyType

from sesnaimpute.sed_models import paths


REGISTER_FILENAMES = MappingProxyType({
    "sps": "sps_register.hdf5",
    "pahc": "pahc_register.hdf5",
    "h2shock": "h2shock_register.hdf5",
    "galz": "galz_register.hdf5",
    "agb": "agb_register.hdf5",
    "yso": "yso_register.hdf5",
})


#: The five census-YSO sub-grid keys pooled (by density) into the sole
#: "yso" register above. [Retirement, 2026-09-01] Each used to have its
#: own published leaf register (`yso_c0_register.hdf5`, etc.); those were
#: retired because every one of their `/models` columns is a byte-
#: identical slice of the pooled register's own 200000 rows -- the joint
#: per-class density computation happens before any leaf is written (see
#: `sed_models_register.density.derive`), so partitioning it into five
#: files and re-concatenating them on every build was a pure round trip
#: through disk. Subclass identity is fully recoverable from the pooled
#: file alone via its per-row `/models/SUBCLASS` column and its
#: `/members` table (`MEMBER_KEY`/`ROW_OFFSET`/`N_MODELS`). None of
#: these five keys is in `REGISTER_FILENAMES` any more -- `register_path`
#: raises `KeyError` on them -- so this tuple exists only so a consumer
#: (e.g. `sed_fit.fit._resolve_register_path`) can still recognize "this
#: model directory is a YSO member" and resolve it to the pooled "yso"
#: key, without that recognition depending on the retired keys still
#: being resolvable register keys themselves.
YSO_MEMBER_KEYS = ("yso/c0", "yso/cI", "yso/cII", "yso/cIII", "yso/td")


def register_directory():
    """Return the canonical directory holding the six register artifacts."""
    return os.path.abspath(paths.path_for("model_registers_dir"))


def register_path(key):
    """Return the flat artifact path for one exact logical library key."""
    try:
        filename = REGISTER_FILENAMES[key]
    except KeyError:
        raise KeyError("unknown model-register key %r" % (key,)) from None
    return os.path.join(register_directory(), filename)


def register_paths():
    """Return the exact ordered logical-key to flat-path mapping."""
    return {key: register_path(key) for key in REGISTER_FILENAMES}
