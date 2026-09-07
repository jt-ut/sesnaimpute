"""The GAL population sample: one survey-wide grid, no per-region grain
(SPEC_BMSTP_DRAFT.md sec. 5.4 "Marks", "Weight", "Grain";
IMPLEMENTATION_BMSTP_DRAFT.md sec. 3 row 1.4).

A galaxy carries the whole column (`x = 1` exactly; the source's own column
kernel supplies its entire extinction spread at the fitter's read, sec.
5.4 "Grain"), so the sample is one delta in `x` at every node of the
counts law's `log10 S` grid (`population/gal/counts_gal_survey.hdf5`,
`population.gal.build_counts_law`'s `LOG10_S_GRID`, `PHI_S`), weighted
`phi(S) . S` per `d log10 S` (sec. 5.4 "Weight").
"""

import h5py
import numpy as np

from sesnaimpute import config as config_module


def sample(config):
    """`(x, log10_b, w)`, one point per counts-law node: `x = 1` (the
    whole column, sec. 5.4 "Marks"), `log10_b = LOG10_S_GRID` (the node's
    `log10 S`), `w = phi(S) * S * d(log10 S)` (sec. 5.4 "Weight"; the
    node spacing is fixed by `population.gal.build_log10_s_grid`'s
    `np.linspace`, so one scalar step serves every node)."""
    path = config_module.product_path(config, "population", "gal", "counts", "survey")
    with h5py.File(path, "r") as f:
        log10_s = f["LOG10_S_GRID"][()].astype(np.float64)
        phi_s = f["PHI_S"][()].astype(np.float64)
    d_log10_s = float(log10_s[1] - log10_s[0])
    s = 10.0 ** log10_s
    w = phi_s * s * d_log10_s
    x = np.ones_like(log10_s)
    return x, log10_s, w
