"""Whether each class's SED-model library samples spectral shape finely
enough to resolve the survey's own calibration precision
(SPEC_BMSTP_DRAFT.md sec 1.4: "the sum over templates is a quadrature over
the class's space of SED shapes"; sec 6.1: the fit is Gaussian in log flux,
two nuisances, extinction and gray scale; sec 3.5: the six registers). A
quadrature over templates resolves the likelihood it integrates only while
neighbouring nodes sit closer than the kernel it approximates -- here each
band's own library SAMPLING scale, `constants.LIBRARY_SAMPLING_SIGMA_LOG_DEX`
(twice the survey's absolute-calibration precision, `constants.
SIGMA_CAL_DEX`; see that constant's own derivation), not a measured survey
photometric error: the curated register's own resampling
(`sed_models_register`) already whitens by the same scale, so this check
and a library's own template spacing are in one unit by construction
(WP-PRIOR-8). Report-only (rule 12): prints the numbers and writes one
small survey product; no fit is run and no source's own fluxes are read,
only each class's own register.

The same product's `SIGMA_LIB_DEX`, one row per LIBRARY and one column per
BAND (sec 6.1), is `fittp.likelihood.prepare`'s own read of this
measurement: the per-band width the kernel sum over templates needs to
resolve a source sitting midway between a library's two nearest
neighbours, `d50 * LIBRARY_SAMPLING_SIGMA_LOG_DEX_b / (2 sqrt(6))` at the
library's own MEASURED `d50` -- the formula alone, for every class, no
thickness branch and no fallback (planner's ruling, WP-PRIOR-8 correction):
the curated registers are built by a sampler that asserts packing >= 1 and
coverage < 1 everywhere, so the spacing is uniform at the sampling scale
by construction and `d50` IS the spacing, not a locally clustered pocket's
understatement of it. A guard against that failure mode cannot fire under
the design that stands, so it is not kept as a trap for the next reader.
"""

import os

import h5py
import numpy as np
from scipy.spatial import cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.fittp.likelihood import GRAY_COLUMN
from sesnaimpute.population import selection as population_selection

#: `config.product_path`'s own stem for this module's one product
#: (`attrs_registry.REGISTRY`'s key, CODING_RULES_BMSTP.md rule 5).
_STEM = "library-resolution_check_survey"

#: The eight SESNA bands, `catalog`'s and `fittp.likelihood`'s own order.
BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: The column at which the blended extinction law (`population.selection`)
#: is evaluated for the check's extinction nuisance direction: SPEC_BMSTP
#: sec 2's blend sits fully diffuse below A_K=0.5 and fully dense above
#: A_K=1.0, so 0.5 is the blend's own midpoint -- the brief's "median
#: blended extinction law".
AK_MEDIAN = 0.5


def _register_path(config, cls):
    key = definitions.CLASS_REGISTER[cls]
    return f"{config.inputs['sed_models']}/registers/{key}_register.hdf5"


def _read_register(config, cls):
    """A library's `MODEL_NAME` and its templates' reference log-fluxes in
    the eight SESNA bands (SPEC_BMSTP_DRAFT.md sec 3.5: "reference fluxes
    in eight bands, at 1 kpc, zero extinction"), row-aligned in the
    register's own order. A minority of templates (the reddest of YSO's:
    up to 16% of one band, a genuine radiative-transfer underflow at 1 kpc
    with zero extinction, not a missing value) carry exactly zero in a
    band; floored at that band's own faintest positive template so
    `log10` stays finite without inventing a flux scale the register does
    not itself produce, and disclosed by count.
    """
    with h5py.File(_register_path(config, cls), "r") as f:
        m = f["models"]
        names = np.char.decode(m["MODEL_NAME"][:].astype("S"), "utf-8")
        raw = {b: m[f"F_REF_{b}"][:].astype(np.float64) for b in BAND_KEYS}
    log10_f_ref = np.empty((names.size, N_BANDS))
    n_floored = 0
    for j, b in enumerate(BAND_KEYS):
        v = raw[b]
        bad = v <= 0
        n_floored += int(bad.sum())
        if bad.any():
            v = np.where(bad, v[~bad].min() if (~bad).any() else np.nan, v)
        log10_f_ref[:, j] = np.log10(v)
    if n_floored:
        print("fittp.library_resolution: %s register -- %d/%d (band, template) reference "
              "fluxes are exactly zero, floored at that band's own faintest positive template"
              % (cls, n_floored, names.size * N_BANDS))
    return names, log10_f_ref


def _shape_space(log10_f_ref, kappa_prime):
    """The library's shapes in units of its own sampling scale (WP-PRIOR-8):
    reference log fluxes whitened by `constants.LIBRARY_SAMPLING_SIGMA_LOG_DEX`
    -- twice the survey's own calibration precision per band, the SAME
    scale the curation's own template resampling whitens by, so a nearest-
    neighbour spacing measured here is dimensionless, in units of that one
    sampling length -- with the two nuisance directions (the blended
    extinction law, `kappa_prime`, and the design's constant gray column,
    `fittp.likelihood.GRAY_COLUMN`), each whitened the same way, projected
    out by an orthonormal basis of their span (`np.linalg.qr`). The
    residual lies exactly in the orthogonal 6-D complement, so Euclidean
    distance among residuals kept in their ambient 8 coordinates already
    equals distance in that 6-D space.
    """
    scale = constants.LIBRARY_SAMPLING_SIGMA_LOG_DEX
    y = log10_f_ref / scale[None, :]
    gray = np.full(N_BANDS, GRAY_COLUMN)
    nuisance = np.column_stack([kappa_prime / scale, gray / scale])
    q, _ = np.linalg.qr(nuisance)
    return y - (y @ q) @ q.T


def _sigma_lib_dex(d50):
    """WP-PRIOR-8's per-band conversion, the formula alone (planner's
    correction: the thickness guard is removed): a library's own measured
    `d50` (dimensionless, sampling lengths) back to a per-band dex offset,
    `d50 * (2 sigma_b) / (2 sqrt(6))` -- distributed over the 6-D shape
    space's dimensions (dividing by `sqrt(6)`) and halved (a source midway
    between two neighbours sits at `d50/2` from each). No fallback: the
    curated registers are built by a sampler that asserts packing >= 1 and
    coverage < 1 everywhere, so a library's spacing is uniform at the
    sampling scale by construction and `d50` is trusted as that spacing
    directly, never a locally clustered pocket's understatement of it.
    """
    scale = constants.LIBRARY_SAMPLING_SIGMA_LOG_DEX
    return d50 * scale / (2.0 * np.sqrt(6.0))


def _library_sigma_lib(config, cls, kappa_prime):
    """One library's own pooled 50th/90th-percentile nearest-neighbour
    spacing (dimensionless, `_shape_space`'s whitening, `scipy.spatial.
    cKDTree`, k=2) at its full, current count, and the per-band
    `sigma_lib` the measured `d50` implies (`_sigma_lib_dex`, the formula
    alone). `d90` is reported alongside `d50` but decides nothing."""
    names, log10_f_ref = _read_register(config, cls)
    shape = _shape_space(log10_f_ref, kappa_prime)
    d, _ = cKDTree(shape).query(shape, k=2)
    d50 = float(np.percentile(d[:, 1], 50))
    d90 = float(np.percentile(d[:, 1], 90))
    return names.size, d50, d90, _sigma_lib_dex(d50)


#: `UNITS`/`READING` for every dataset here (CODING_RULES_BMSTP.md rule 5)
#: live in `attrs_registry.REGISTRY`, keyed by `(_STEM, name)`.


def _write(path, library_order, n_by_lib, d50_by_lib, d90_by_lib, sigma_lib_dex):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        for key, data in (
            ("BANDS", np.array(BAND_KEYS, dtype="S4")),
            ("LIBRARY", np.array(library_order, dtype="S6")),
            ("N_TEMPLATES", np.array([n_by_lib[c] for c in library_order], dtype=np.int64)),
            ("D50", np.array([d50_by_lib[c] for c in library_order])),
            ("D90", np.array([d90_by_lib[c] for c in library_order])),
            ("SIGMA_LIB_DEX", np.array([sigma_lib_dex[c] for c in library_order])),
        ):
            build_module.write_dataset(f, key, data, *REGISTRY[(_STEM, key)])


def build(config, regions=None):
    """Writes `fittp/check/library-resolution_check_survey.hdf5`
    (report-only; `regions` ignored, a survey check over the six
    registers, not a per-region product). Prints each library's own
    pooled nearest-neighbour spacing (d50/d90, dimensionless sampling
    lengths) and the per-band `sigma_lib` (dex) the formula gives at that
    measured `d50` -- no thickness branch (planner's ruling).
    """
    del regions
    with progress.Stage("fittp.library_resolution") as st:
        w_ramp = population_selection.law_dense_weight(np.array([AK_MEDIAN]))
        kappa_prime = population_selection.kappa_hybrid(config, w_ramp)[0]

        library_order = list(definitions.CLASS_REGISTER)
        n_by_lib, d50_by_lib, d90_by_lib, sigma_lib_dex = {}, {}, {}, {}
        for i, cls in enumerate(library_order):
            n_t, d50, d90, sigma_lib_b = _library_sigma_lib(config, cls, kappa_prime)
            n_by_lib[cls], d50_by_lib[cls], d90_by_lib[cls] = n_t, d50, d90
            sigma_lib_dex[cls] = sigma_lib_b
            print("fittp.library_resolution: %-5s n=%6d d50=%.4f d90=%.4f (sampling lengths); "
                  "sigma_lib (dex, per band %s) = %s"
                  % (cls, n_t, d50, d90, list(BAND_KEYS), np.round(sigma_lib_b, 4).tolist()))
            st.tick(i + 1, len(library_order), "libraries")

        out_path = config_module.product_path(config, "fittp", "check", "library-resolution", "survey")
        _write(out_path, library_order, n_by_lib, d50_by_lib, d90_by_lib, sigma_lib_dex)

        st.done(out_path, n_library=len(library_order))


if __name__ == "__main__":
    run(build)
