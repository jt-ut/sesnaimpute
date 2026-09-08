"""The GAL population sample: one survey-wide grid, no per-region grain
(SPEC_BMSTP_DRAFT.md sec. 5.4 "Marks", "Weight", "Grain";
IMPLEMENTATION_BMSTP_DRAFT.md sec. 3 row 1.4).

A galaxy carries the whole column (`x = 1` exactly; the source's own column
kernel supplies its entire extinction spread at the fitter's read, sec.
5.4 "Grain"), so the sample is one delta in `x` at every node of the
counts law's `log10 S` grid (`population/gal/counts_gal_survey.hdf5`,
`population.gal.build_counts_law`'s `LOG10_S_GRID`, `PHI_S_POINT` --
the point-source-corrected law, `PHI_S * p(S)`, studies/swire_vs_fazio.md
sec 6: the consumer-facing quantity is the DETECTABLE galaxy density),
weighted `phi(S) . p(S) . S` per `d log10 S` (sec. 5.4 "Weight").
"""

import h5py
import numpy as np

from sesnaimpute import config as config_module


def _counts_law(config):
    """`(log10_s, phi_s_point, d_log10_s)` off the tabulated counts-law
    grid (`population.gal.build_counts_law`'s `LOG10_S_GRID`,
    `PHI_S_POINT`, the point-source-corrected law)."""
    path = config_module.product_path(config, "population", "gal", "counts", "survey")
    with h5py.File(path, "r") as f:
        log10_s = f["LOG10_S_GRID"][()].astype(np.float64)
        phi_s_point = f["PHI_S_POINT"][()].astype(np.float64)
    d_log10_s = float(log10_s[1] - log10_s[0])
    return log10_s, phi_s_point, d_log10_s


def sample(config):
    """`(x, log10_b, w)`, one point per counts-law node: `x = 1` (the
    whole column, sec. 5.4 "Marks"), `log10_b = LOG10_S_GRID` (the node's
    `log10 S`), `w = phi(S) * p(S) * S * d(log10 S)` (sec. 5.4 "Weight" --
    the SHAPE weight only, for the grid; not the sky density, `density`
    below, sec. 5.4 "Sky density": the two are not interchangeable)."""
    log10_s, phi_s_point, d_log10_s = _counts_law(config)
    s = 10.0 ** log10_s
    w = phi_s_point * s * d_log10_s
    x = np.ones_like(log10_s)
    return x, log10_s, w


def density(config):
    """`A_GAL = integral phi(S) p(S) dS`, sec. 5.4 "Sky density": `dS = S
    ln10 d(log10 S)`, so `A_GAL = Sigma_k phi(S_k) p(S_k) S_k ln10 Delta
    log10 S` over the tabulated counts-law grid -- `sample`'s shape weight
    times `ln 10`, read by `bmstp.shapes.build_gal` (P4's `DENSITY_GAL`),
    `bmstp.density` (P1's `DENSITY_GAL`) and `bmstp.atlas` (GAL's Monte
    Carlo total)."""
    log10_s, phi_s_point, d_log10_s = _counts_law(config)
    s = 10.0 ** log10_s
    return float(np.sum(phi_s_point * s * d_log10_s * np.log(10.0), dtype=np.float64))
