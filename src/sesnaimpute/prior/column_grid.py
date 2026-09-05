"""The shared column axis every class tabulates on (`IMPLEMENTATION.md`
section 2): nodes spaced uniformly in the cumulative column-kernel width

    tau(A) = Integral_floor^A dA' / sigma_a(A')

from a floor (the 1st percentile of `A_COL_SIG_K` survey-wide) to the cap
`AK_CAP`, with spacing `sqrt(8 * EPS_GRID)` -- the grid's fidelity bar of
`EPS_GRID` relative L1 in misplaced mass. `sigma_a(A)` is the composed
column kernel's own resolution width, `sqrt(kernel.Kernel.second_moment(A))`.
A source's `A_s` brackets two nodes; its shape is the linear blend of the
two node tabulations, at `(NODE_LO, NODE_W)` from `bracket`.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.prior import kernel as kernel_module

#: The grid's fidelity bar: relative L1 mass misplaced by the linear blend
#: between adjacent nodes (IMPLEMENTATION.md section 2).
EPS_GRID = 0.002

#: The tabulation extent, not the kernel's support -- mass beyond it is
#: carried analytically (IMPLEMENTATION.md section 2).
AK_CAP = 14.232012269510967

#: Probe count for the tau(A) quadrature. sigma_a(A) is smooth over the
#: grid's range, so this resolves the integral far below EPS_GRID's own
#: slack; fixed so the grid is reproducible.
N_PROBES = 512


def _ak_floor(config, regions):
    """The 1st percentile of `A_COL_SIG_K` survey-wide -- clamping a
    source to it moves that source by less than the smallest per-source
    measurement uncertainty anywhere in the survey."""
    parts = []
    for region in regions:
        path = config_module.product_path(config, "sky/derived", "adopted",
                                          "column", "source", region=region.name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "prior.column_grid: adopted column missing for region %r at %s "
                "-- run the 'sky.derived.column' RUNBOOK line first"
                % (region.name, path))
        with h5py.File(path, "r") as f:
            sig = np.asarray(f["A_COL_SIG_K"][:], dtype=np.float64)
        finite = sig[np.isfinite(sig)]
        if finite.size:
            parts.append(finite)
    if not parts:
        raise ValueError(
            "prior.column_grid: no finite A_COL_SIG_K found across any "
            "region's adopted column product")
    return float(np.percentile(np.concatenate(parts), 1.0))


def derive(config, regions=None, eps=EPS_GRID, n_probes=N_PROBES):
    """`(nodes, probes, sigma)`: nodes uniform in `tau(A)`, from the
    measured floor to `AK_CAP`, spacing `sqrt(8 * eps)`, endpoints pinned
    exactly. `sigma` is the composed kernel's resolution width at every
    probe, one vectorised call."""
    regions = list(regions_module.REGIONS) if regions is None else list(regions)
    floor = _ak_floor(config, regions)

    probes = np.geomspace(floor, AK_CAP, n_probes)
    kern = kernel_module.load(config)
    sigma = np.sqrt(kern.second_moment(probes))
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("prior.column_grid: sigma_a is not positive-finite "
                          "over [%.6g, %.6g]" % (floor, AK_CAP))

    inv = 1.0 / sigma
    dtau = 0.5 * (inv[1:] + inv[:-1]) * np.diff(probes)
    tau = np.concatenate([[0.0], np.cumsum(dtau)])
    tau_total = float(tau[-1])
    delta_tau = float(np.sqrt(8.0 * eps))
    n_nodes = int(np.ceil(tau_total / delta_tau))
    levels = np.linspace(0.0, tau_total, n_nodes + 1)
    nodes = np.interp(levels, tau, probes)
    nodes[0], nodes[-1] = floor, AK_CAP
    if not np.all(np.diff(nodes) > 0.0):
        raise ValueError("prior.column_grid: node grid is not strictly increasing")
    return nodes, probes, sigma


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
    """The stored node array -- the one reader every class build uses."""
    path = config_module.product_path(config, "bms", "sesna", "column-grid", "survey")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.column_grid: no column-grid product at %s -- run the "
            "'prior.column_grid' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        return f["A_NODES"][:]


def build(config, regions=None):
    """Derives the node grid over the requested regions (`regions=None`
    read as all thirty; the product is a survey total regardless) and
    writes `bms/sesna/column-grid_sesna_survey.hdf5`."""
    regions = list(regions_module.REGIONS) if regions is None else \
        [r for r in regions_module.REGIONS if r.name in set(regions)]
    node_arr, probes, sigma = derive(config, regions=regions)

    out_path = config_module.product_path(config, "bms", "sesna", "column-grid", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.create_dataset("A_NODES", data=node_arr.astype(np.float64))
        f.create_dataset("EPS_GRID", data=np.float64(EPS_GRID))
        f.create_dataset("AK_CAP", data=np.float64(AK_CAP))
        f.create_dataset("A_PROBE", data=probes.astype(np.float64))
        f.create_dataset("SIGMA_A_PROBE", data=sigma.astype(np.float64))

    print("column_grid: floor=%.6f cap=%.6f eps=%g -> %d nodes -> %s"
          % (node_arr[0], node_arr[-1], EPS_GRID, node_arr.size, out_path),
          flush=True)


if __name__ == "__main__":
    run(build)
