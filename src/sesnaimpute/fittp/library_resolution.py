"""Whether the YSO library has enough templates to resolve the survey's own
photometric error (SPEC_BMSTP_DRAFT.md sec 1.4: "the sum over templates is a
quadrature over the class's space of SED shapes"; sec 6.1: the fit is
Gaussian in log flux, two nuisances, extinction and gray scale; sec 3.5: the
six registers). A quadrature over templates resolves the likelihood it
integrates only while neighbouring nodes sit, in the survey's own units,
closer than the Gaussian kernel's own width -- the survey's photometric
error. Report-only (rule 12): prints the numbers and writes one small
survey product; no fit is run and no source's own fluxes are read, only
the catalogue's error columns. The same product's `SIGMA_LIB_DEX`, one
number per LIBRARY (sec 6.1), is `fittp.likelihood.prepare`'s own read of
this measurement: the width the kernel sum over templates needs to resolve
a source sitting midway between a library's two nearest neighbours.
"""

import os

import h5py
import numpy as np
from scipy.spatial import cKDTree

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.fittp.likelihood import GRAY_COLUMN
from sesnaimpute.population import selection as population_selection

#: The eight SESNA bands, `catalog`'s and `fittp.likelihood`'s own order.
BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: SPEC_BMSTP_DRAFT.md sec 6.1's calibration systematic on `sigma_log`:
#: `fittp/likelihood.py`'s `prepare` carries no such floor and
#: `constants.py` names none, so this check discloses the brief's own
#: fallback -- Reach et al. 2005 (astro-ph/0507139), the IRAC absolute
#: flux calibration's own uncertainty.
SIGMA_FLOOR_DEX = 0.02

#: `ORIGIN_FNU`'s detected code, the same value `fittp.likelihood.prepare`
#: and `fittp.cascade.build_region` compare against.
DETECTED_ORIGIN = 1

#: The column at which the blended extinction law (`population.selection`)
#: is evaluated for the check's extinction nuisance direction: SPEC_BMSTP
#: sec 2's blend sits fully diffuse below A_K=0.5 and fully dense above
#: A_K=1.0, so 0.5 is the blend's own midpoint -- the brief's "median
#: blended extinction law".
AK_MEDIAN = 0.5

#: Fixed seeds (CODING_RULES_BMSTP.md rule 4: reproducible, not a date):
#: one for the pooled source sample the resolution `sigma_i` is measured
#: on, one for the template subsets the spacing scaling curve is fit to.
SEED_SOURCE_SAMPLE = 90210
SEED_TEMPLATE_SUBSET = 31415
N_SOURCE_SAMPLE = 200_000

#: The Gaussian resolution rule (brief item 4): the template sum
#: integrates a Gaussian kernel to relative error below 1e-3 while its
#: node spacing, in kernel sigma, is at or below this.
RESOLUTION_TOL = 0.5
#: The three tolerances the trade is reported at (brief item 4).
TOLERANCES = (0.3, 0.5, 1.0)

#: SPEC_BMSTP_DRAFT.md sec 6.1 -- sigma_lib,L converts a library's own
#: median nearest-neighbour spacing d50 (measured in units of this check's
#: own per-band whitening, SIGMA_FLOOR_DEX) back to a per-band dex offset:
#: distributed over the 6-D shape space's dimensions (dividing by sqrt(6))
#: and halved (a source midway between two neighbours sits at d50/2 from
#: each).
SIGMA_LIB_SCALE = SIGMA_FLOOR_DEX / (2.0 * np.sqrt(6.0))

#: The YSO register's five sub-grid `SUBCLASS`/`SUBGRID` labels
#: (`population.yso_mass`'s own `_SUBGRIDS`).
YSO_SUBGRID_LABELS = ("C0", "CI", "CII", "CIII", "TD")
#: The four sizes the spacing scaling curve is fit from (brief item 3).
SUBSET_FRACTIONS = (1.0, 0.5, 0.25, 0.125)


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


#: Bytes per catalogue row the sample gather holds (rule 10b): FNU_MJY and
#: SIGMA_FNU_MJY, 8 bands float64 each, plus ORIGIN_FNU, 8 bands int8.
_ROW_BYTES = 2 * N_BANDS * 8 + N_BANDS


def _sigma_i(config, st):
    """The survey's own resolution per band (brief item 1): the 10th
    percentile of `sigma_log = SIGMA_FNU_MJY / (FNU_MJY ln 10)` among
    `ORIGIN_FNU`-detected measurements, pooled over a fixed-seed sample of
    `N_SOURCE_SAMPLE` sources drawn once from every region's curated
    catalogue (the same three columns `fittp.cascade` reads), floored at
    `SIGMA_FLOOR_DEX`. Each region is read in contiguous row batches
    (`sesnaimpute.batches.batches`, rule 10b) -- a handful of million-row
    regions rule out an h5py point selection at the sample's own scattered
    row indices (an O(selection size) HDF5 cost that measured minutes per
    region); a batch's own sample rows are picked out in memory instead,
    which is exact and cheap.
    """
    region_names = [r.name for r in regions_module.REGIONS]
    paths = [config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=r) for r in region_names]
    sizes = []
    for path in paths:
        with h5py.File(path, "r") as f:
            sizes.append(f["NAME"].shape[0])
    sizes = np.asarray(sizes)
    total = int(sizes.sum())
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    n_sample = min(N_SOURCE_SAMPLE, total)
    rng = np.random.default_rng(SEED_SOURCE_SAMPLE)
    global_idx = np.sort(rng.choice(total, size=n_sample, replace=False))

    sigma_log_by_band = [[] for _ in range(N_BANDS)]
    n_regions = len(region_names)
    for i, path in enumerate(paths):
        sel = global_idx[(global_idx >= offsets[i]) & (global_idx < offsets[i + 1])] - offsets[i]
        if sel.size:
            with h5py.File(path, "r") as f:
                for start, stop in batches_module.batches(sizes[i], _ROW_BYTES):
                    in_batch = sel[(sel >= start) & (sel < stop)] - start
                    if in_batch.size == 0:
                        continue
                    flux = np.asarray(f["FNU_MJY"][start:stop, :], dtype=np.float64)[in_batch]
                    sigma = np.asarray(f["SIGMA_FNU_MJY"][start:stop, :], dtype=np.float64)[in_batch]
                    origin = np.asarray(f["ORIGIN_FNU"][start:stop, :])[in_batch]
                    detected = origin == DETECTED_ORIGIN
                    safe_flux = np.where(detected & (flux > 0), flux, 1.0)
                    sigma_log = sigma / (safe_flux * np.log(10.0))
                    for b in range(N_BANDS):
                        good = detected[:, b] & (sigma_log[:, b] > 0)
                        if good.any():
                            sigma_log_by_band[b].append(sigma_log[good, b])
        st.tick(i + 1, n_regions, "regions")

    sigma_i = np.array([
        max(float(np.percentile(np.concatenate(vals), 10.0)), SIGMA_FLOOR_DEX) if vals
        else SIGMA_FLOOR_DEX
        for vals in sigma_log_by_band])
    return sigma_i, n_sample, total


def _shape_space(log10_f_ref, sigma_i, kappa_prime):
    """The library's shapes in units of the survey's own resolution
    (brief item 2): reference log fluxes whitened by `sigma_i`, with the
    two nuisance directions -- the blended extinction law and the design's
    constant gray column (`fittp.likelihood.GRAY_COLUMN`), each whitened
    the same way -- projected out by an orthonormal basis of their span
    (`np.linalg.qr`). The residual lies exactly in the orthogonal
    6-D complement, so Euclidean distance among residuals kept in their
    ambient 8 coordinates already equals distance in that 6-D space.
    """
    y = log10_f_ref / sigma_i[None, :]
    gray = np.full(N_BANDS, GRAY_COLUMN)
    nuisance = np.column_stack([kappa_prime / sigma_i, gray / sigma_i])
    q, _ = np.linalg.qr(nuisance)
    return y - (y @ q) @ q.T


def _spacing_scaling(points, seed):
    """`d_NN`'s 50th/90th percentile (`scipy.spatial.cKDTree`, k=2, brief
    item 3) at the full size and fixed-seed 1/2, 1/4, 1/8 subsets, and the
    log-log fit of the 90th percentile against size that item 4's `n*`
    reads: `log d90 = intercept + slope * log n`, `d_eff = -1/slope`.
    """
    n_full = points.shape[0]
    rng = np.random.default_rng(seed)
    sizes, p50, p90 = [], [], []
    for frac in SUBSET_FRACTIONS:
        n = max(3, int(round(n_full * frac)))
        pts = points if n >= n_full else points[rng.choice(n_full, size=n, replace=False)]
        d, _ = cKDTree(pts).query(pts, k=2)
        sizes.append(pts.shape[0])
        p50.append(float(np.percentile(d[:, 1], 50)))
        p90.append(float(np.percentile(d[:, 1], 90)))
    sizes = np.asarray(sizes, dtype=np.float64)
    p90 = np.asarray(p90)
    slope, intercept = np.polyfit(np.log(sizes), np.log(p90), 1)
    d_eff = -1.0 / slope
    return sizes.astype(np.int64), np.asarray(p50), p90, d_eff, slope, intercept


def _n_star(slope, intercept, tol):
    """The template count at which the fitted 90th-percentile-spacing line
    reaches `tol` (brief item 4): extrapolated where 200,000 already sits
    below it, interpolated where it sits above -- the same straight line
    either way.
    """
    return float(np.exp((np.log(tol) - intercept) / slope))


def _library_thickness(config, cls, sigma_i, kappa_prime):
    """One line's worth for a library other than YSO (brief item 4, "the
    question there is whether they are too THIN"): its own pooled 50th/
    90th-percentile `d_NN` at its full, current count, and whether that
    90th percentile already clears `RESOLUTION_TOL`.
    """
    names, log10_f_ref = _read_register(config, cls)
    shape = _shape_space(log10_f_ref, sigma_i, kappa_prime)
    d, _ = cKDTree(shape).query(shape, k=2)
    d50 = float(np.percentile(d[:, 1], 50))
    d90 = float(np.percentile(d[:, 1], 90))
    return names.size, d50, d90, d90 <= RESOLUTION_TOL


def _write(path, sigma_i, n_sample, n_total, groups, sizes_all, p50_all, p90_all,
           deff_all, nstar_all, other, library_order, sigma_lib_dex):
    labels = list(groups)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.attrs["N_SOURCE_SAMPLE"] = n_sample
        f.attrs["N_SOURCE_TOTAL"] = n_total
        f.attrs["TOLERANCES"] = np.asarray(TOLERANCES)
        f.attrs["RESOLUTION_TOL"] = RESOLUTION_TOL
        f.create_dataset("BANDS", data=np.array(BAND_KEYS, dtype="S4"))
        f.create_dataset("SIGMA_I", data=sigma_i)
        f.create_dataset("YSO_GROUP", data=np.array(labels, dtype="S8"))
        f.create_dataset("YSO_N_AT_SIZE", data=np.array([sizes_all[g] for g in labels]))
        f.create_dataset("YSO_D50_AT_SIZE", data=np.array([p50_all[g] for g in labels]))
        f.create_dataset("YSO_D90_AT_SIZE", data=np.array([p90_all[g] for g in labels]))
        f.create_dataset("YSO_D_EFF", data=np.array([deff_all[g] for g in labels]))
        f.create_dataset("YSO_N_STAR", data=np.array([nstar_all[g] for g in labels]))
        f.create_dataset("OTHER_LIBRARY", data=np.array([o[0] for o in other], dtype="S6"))
        f.create_dataset("OTHER_N_TEMPLATES", data=np.array([o[1] for o in other], dtype=np.int64))
        f.create_dataset("OTHER_D50", data=np.array([o[2] for o in other]))
        f.create_dataset("OTHER_D90", data=np.array([o[3] for o in other]))
        f.create_dataset("OTHER_THICK_ENOUGH", data=np.array([int(o[4]) for o in other], dtype=np.int8))
        f.create_dataset("LIBRARY", data=np.array(library_order, dtype="S6"))
        f.create_dataset("SIGMA_LIB_DEX", data=np.array(
            [sigma_lib_dex[c] for c in library_order]))


def build(config, regions=None):
    """Writes `fittp/check/library-resolution_check_survey.hdf5`
    (report-only; `regions` ignored, a survey check over the pooled
    catalogue and the six registers, not a per-region product). Prints the
    survey's own resolution per band, the YSO library's shape-space
    nearest-neighbour spacing at four sizes per sub-grid and pooled, the
    fitted effective dimension, the template count `n*` at which the
    worst sub-grid resolves the survey's error at three tolerances, and
    one line per other library on whether its own count is thick enough.
    """
    with progress.Stage("fittp.library_resolution") as st:
        sigma_i, n_sample, n_total = _sigma_i(config, st)
        print("fittp.library_resolution: sigma_i (dex, 10th-pct log-flux error, "
              "floored at %.3g, Reach+2005 IRAC calibration) = %s, from %d/%d sampled sources"
              % (SIGMA_FLOOR_DEX,
                 {k: round(float(v), 4) for k, v in zip(BAND_KEYS, sigma_i)}, n_sample, n_total))

        w_ramp = population_selection.law_dense_weight(np.array([AK_MEDIAN]))
        kappa_prime = population_selection.kappa_hybrid(config, w_ramp)[0]

        names, log10_f_ref = _read_register(config, "YSO")
        mass_path = config_module.product_path(config, "population", "yso", "mass", "survey")
        with h5py.File(mass_path, "r") as f:
            mass_names = np.char.decode(f["MODEL_NAME"][:].astype("S"), "utf-8")
            subgrid = np.char.decode(f["SUBGRID"][:].astype("S"), "utf-8")
        n_join = int(np.sum(mass_names == names)) if mass_names.size == names.size else 0
        print("fittp.library_resolution: YSO register/mass-table join n_matched=%d/%d"
              % (n_join, names.size))

        shape = _shape_space(log10_f_ref, sigma_i, kappa_prime)

        groups = ["POOLED"] + list(YSO_SUBGRID_LABELS)
        masks = {"POOLED": np.ones(names.size, dtype=bool)}
        masks.update({lab: subgrid == lab for lab in YSO_SUBGRID_LABELS})

        sizes_all, p50_all, p90_all, deff_all, nstar_all = {}, {}, {}, {}, {}
        for label in groups:
            sizes, p50, p90, d_eff, slope, intercept = _spacing_scaling(
                shape[masks[label]], SEED_TEMPLATE_SUBSET)
            sizes_all[label], p50_all[label], p90_all[label] = sizes, p50, p90
            deff_all[label] = d_eff
            nstar_all[label] = [_n_star(slope, intercept, tol) for tol in TOLERANCES]
            print("fittp.library_resolution: YSO %-6s n=%s d50=%s d90=%s d_eff=%.2f "
                  "n*(0.3,0.5,1.0)=%s"
                  % (label, sizes.tolist(), np.round(p50, 4).tolist(), np.round(p90, 4).tolist(),
                     d_eff, [round(x) for x in nstar_all[label]]))

        tol_idx = TOLERANCES.index(RESOLUTION_TOL)
        worst = max(YSO_SUBGRID_LABELS, key=lambda lab: nstar_all[lab][tol_idx])
        n_star_worst = nstar_all[worst][tol_idx]
        print("fittp.library_resolution: worst sub-grid at tol=%.1f is %s, n*=%.0f; "
              "200000 is %s it"
              % (RESOLUTION_TOL, worst, n_star_worst,
                 "ABOVE" if 200_000 >= n_star_worst else "BELOW"))

        other = []
        for cls in definitions.CLASS_REGISTER:
            if cls == "YSO":
                continue
            n_t, d50, d90, thick = _library_thickness(config, cls, sigma_i, kappa_prime)
            other.append((cls, n_t, d50, d90, thick))
            print("fittp.library_resolution: %-5s n=%6d d50=%.3f d90=%.3f -- %s"
                  % (cls, n_t, d50, d90, "thick enough" if thick else "too thin"))

        # SPEC_BMSTP_DRAFT.md sec 6.1: sigma_lib,L, one number per LIBRARY --
        # the YSO register pooled (its five sub-grids' own spacings above
        # stay a diagnostic, the register being one set, sec 1.4) -- for the
        # fitter's per-band variance floor.
        library_order = list(definitions.CLASS_REGISTER)
        d50_yso_pooled = p50_all["POOLED"][0]
        sigma_lib_dex = {"YSO": d50_yso_pooled * SIGMA_LIB_SCALE}
        sigma_lib_dex.update({cls: d50 * SIGMA_LIB_SCALE for cls, _, d50, _, _ in other})
        print("fittp.library_resolution: sigma_lib (dex, per library) = %s"
              % {c: round(sigma_lib_dex[c], 4) for c in library_order})

        out_path = config_module.product_path(config, "fittp", "check", "library-resolution", "survey")
        _write(out_path, sigma_i, n_sample, n_total, groups, sizes_all, p50_all, p90_all,
               deff_all, nstar_all, other, library_order, sigma_lib_dex)

        st.done(out_path, n_yso=names.size, worst_subgrid=worst,
                n_star_at_0p5=round(n_star_worst))


if __name__ == "__main__":
    run(build)
