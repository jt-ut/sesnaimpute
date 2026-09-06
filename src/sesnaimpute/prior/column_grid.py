"""The shared column axis every class tabulates on (`IMPLEMENTATION.md`
section 2): a fixed ladder, evenly spaced in log10 column, from a floor
(`AK_FLOOR`) to a cap (`AK_CAP`, a tabulation extent, not the kernel's
support -- mass beyond it is carried analytically), at `DEX_STEP` dex per
step. The ladder is fixed by these three numbers alone -- it does not read
any region's data, does not consult the column kernel, and is not searched
for: the narrowest column-kernel width in use (0.05 dex, wider still once
the measurement term is added) is broader than the step, so linear
interpolation between adjacent nodes is faithful everywhere on the ladder.
A source's `A_s` brackets two nodes; its shape is the linear blend of the
two node tabulations, at `(NODE_LO, NODE_W)` from `bracket`.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute.build import run

#: The ladder's floor, in A_K magnitudes -- fixed, not measured.
AK_FLOOR = 0.037

#: The tabulation extent, not the kernel's support -- mass beyond it is
#: carried analytically (IMPLEMENTATION.md section 2). Fixed, not measured.
AK_CAP = 14.23

#: The ladder's spacing in log10(A_K), dex. Narrower than the narrowest
#: column-kernel width in use (0.05 dex structural, wider once the
#: measurement term is added), so linear interpolation between adjacent
#: nodes is faithful everywhere.
DEX_STEP = 0.02


def _fixed_nodes():
    """The fixed ladder itself: evenly spaced in log10(A_K) from
    `AK_FLOOR` to `AK_CAP` at `DEX_STEP` dex, endpoints pinned exactly
    (the step count is rounded to the nearest integer, so the realised
    spacing is `DEX_STEP` to within rounding, never wider)."""
    n_steps = int(round(np.log10(AK_CAP / AK_FLOOR) / DEX_STEP))
    return np.logspace(np.log10(AK_FLOOR), np.log10(AK_CAP), n_steps + 1)


def bracket(a, nodes):
    """`(NODE_LO, NODE_W)`: the node a source's column `a` brackets, and
    the interpolation weight toward `NODE_LO + 1` -- the one bracket rule
    every class shares (IMPLEMENTATION.md section 2)."""
    nodes = np.asarray(nodes, dtype=float)
    a = np.asarray(a, dtype=float)
    i = np.clip(np.searchsorted(nodes, a) - 1, 0, nodes.size - 2)
    span = nodes[i + 1] - nodes[i]
    safe_span = np.where(span == 0.0, 1.0, span)
    t = np.where(span == 0.0, 0.0, (a - nodes[i]) / safe_span)
    return i.astype(np.intp), np.clip(t, 0.0, 1.0).astype(float)


def nodes(config):
    """The fixed node ladder -- the one reader every class build uses.
    Computed directly (`AK_FLOOR`, `AK_CAP`, `DEX_STEP` are all fixed
    constants); no file read, no region loop, no dependence on the
    column kernel."""
    return _fixed_nodes()


def build(config, regions=None):
    """Writes the fixed ladder to `bms/sesna/column-grid_sesna_survey.hdf5`
    for readers that open the file directly, `A_NODES` only. `regions` is
    accepted for RUNBOOK compatibility and ignored -- the ladder is fixed
    and does not vary by region."""
    node_arr = _fixed_nodes()

    out_path = config_module.product_path(config, "bms", "sesna", "column-grid", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.create_dataset("A_NODES", data=node_arr.astype(np.float64))

    print("column_grid: floor=%.6f cap=%.6f step=%g dex -> %d nodes -> %s"
          % (node_arr[0], node_arr[-1], DEX_STEP, node_arr.size, out_path),
          flush=True)


if __name__ == "__main__":
    run(build)
