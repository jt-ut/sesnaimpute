"""The classification: `P(C | D)`, the subclass posterior, the MAP class and
the imputed flux of SPEC_BMSTP_DRAFT.md section 7.1
(IMPLEMENTATION_BMSTP_DRAFT.md section 1.3 P8, section 4 row 2.5b).

Per source, the six classes' own `LN_EVIDENCE` columns (P7, one file per
class) are laid into one (n, 25) array at `definitions.CLASSMAP`'s fixed
per-class blocks (CLASSMAP groups its 25 rows by class in exactly
`CLASSES` order, so each class occupies one contiguous slice). The cascade
factor `Psi_C(s)^beta` (section 6.5) is a constant added to every subclass
column of its own class before the softmax -- `logsumexp(x_i + a) =
logsumexp(x_i) + a`, so adding `beta * ln Psi_C(s)` to a class's whole
subclass block multiplies that class's evidence by `Psi_C(s)^beta` and
leaves the subclass shares within the class untouched. A single softmax
over the full 25-column row then gives `P_SUBCLASS` directly, and
`P_CLASS` is its per-class column sum -- the two acceptance identities
(`P_CLASS` sums to 1; `P_SUBCLASS` summed within a class equals `P_CLASS`)
hold by construction, not by a separate normalisation.

`A_K_POST`, `A_K_POST_SIG` (n, 6), `CLASSES` order, carry each class's own
posterior extinction mark and its spread (P7's own columns of the same
name, SPEC_BMSTP_DRAFT.md section 6.1) straight through, unreduced by
`MAP_CLASS`: a reader divides the class column it wants by P1's own
`A_COL_K` (`bmstp/density/table_density_source__<R>.hdf5`) to form
`XI_POST` itself.
"""

import argparse
import configparser
import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.batches import batches

#: The fitter's six classes, in the order every P-product's class axis
#: uses (IMPLEMENTATION_BMSTP_DRAFT.md section 1) -- the same order
#: `definitions.CLASSMAP` groups its subclass rows in.
CLASSES = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")
YSO_INDEX = CLASSES.index("YSO")
N_BANDS = len(definitions.BANDS)

#: Each class's contiguous slice of `definitions.CLASSMAP`'s 25 rows
#: (verified against `SUBCLASSES_OF` below at import time, never assumed
#: silently).
_SUBCLASS_NAMES = tuple(definitions.CLASSMAP)
N_SUBCLASS = len(_SUBCLASS_NAMES)


def _class_slices():
    """The (start, stop) slice of the global 25-column subclass axis each
    class occupies, and the fully qualified `CLASS:subclass` label of
    every column (two classes may reuse a subclass name, e.g. AGB and
    STAR both carry "O")."""
    slices = {}
    pos = 0
    for cls in CLASSES:
        n_sub = len(definitions.SUBCLASSES_OF[cls])
        block = _SUBCLASS_NAMES[pos:pos + n_sub]
        if any(s.cls != cls for s in block):
            raise ValueError("fittp.classify: definitions.CLASSMAP is not grouped by "
                              "class in CLASSES order; the fixed-slice layout assumed "
                              "here does not hold")
        slices[cls] = (pos, pos + n_sub)
        pos += n_sub
    if pos != N_SUBCLASS:
        raise ValueError("fittp.classify: class slices %d rows, CLASSMAP has %d" % (pos, N_SUBCLASS))
    return slices


CLASS_SLICES = _class_slices()
SUBCLASS_LABELS = tuple("%s:%s" % (s.cls, s.name) for s in _SUBCLASS_NAMES)

#: Per-row working set for the batch loop (rule 10b): six classes' own
#: LN_EVIDENCE (<=9 cols), FLUX_MEAN (8) and FLUX_COV (8x8) float32 rows,
#: plus the measured flux/sigma/origin and the global (25) and (6,8,8)
#: intermediates, at a generous margin.
ROW_BYTES = 8192

#: The literature-band sensitivity runs (spec sec 7.2, sec 10; P9's own
#: RUN order). Each is a rescaling of one class's ln EV_C by ln(scale),
#: a constant added to that class's whole CLASSMAP subclass block --
#: the same logsumexp-shift trick beta uses -- so every run is
#: classification-time only, no refit.
SENSITIVITY_RUNS = ("kappa_lo", "kappa_hi", "eta_lo", "eta_hi",
                     "eps_ext_lo", "eps_ext_hi", "f_dusty_lo", "f_dusty_hi",
                     "yso_floor")

#: kappa: the young-star law's own normalisation (spec sec 5.5's N_law,
#: sec 10's kappa 14.5/18.7 pc^-2 mag^-2, exponent 2), band 0.36 dex,
#: Pokhrel+2020's cloud-to-cloud scatter -- scales YSO's whole density,
#: and H2S's with it (spec sec 5.6: `A_H2S = (N_law (x) K) eta eps`, so a
#: kappa shift of `N_law` moves H2S by exactly the same factor).
KAPPA_DEX = 0.36
#: eta_r: H2S's knots-per-young-star rate (spec sec 5.6), band 0.45 dex,
#: Froebrich+2015 (UWISH2) / Giannini+2013 -- scales H2S alone.
ETA_DEX = 0.45
#: eps_ext: H2S's star-finder cataloguing fraction (spec sec 5.6), central
#: 0.25 carried over [0.15, 0.35] from five knot-survey cross-matches --
#: scales H2S by the ratio of the band's end to its own central value.
EPS_EXT_CENTRAL, EPS_EXT_LO, EPS_EXT_HI = 0.25, 0.15, 0.35
#: F_dusty: the AGB/STAR dust-detection partition (spec sec 5.2, sec 7.2,
#: sec 10). Riebel+2012's two cited chemistry values (O-rich, C-rich) are
#: read from `bmstp.density`'s own product (F_DUSTY_O, F_DUSTY_C, F_C
#: attrs, `run_sensitivity_region`) rather than duplicated as literals
#: here, and used as the low/high ends of the band relative to the
#: density's own carbon-weighted centre `(1-F_C)*F_DUSTY_O + F_C*F_DUSTY_C`
#: -- the mixture the classification actually ran with -- not their
#: unweighted 50/50 mean (R3 D3).
#: A_min: the lowest column Pokhrel+2020's star-gas relation samples (spec
#: sec 5.5, sec 7.2) -- below it the quadratic young-star law `N_law =
#: kappa A^2` is an extrapolation, so `yso_floor` floors the law itself at
#: `max(A_s, A_MIN_YSO_LAW)`, mag A_K, applied to YSO and to H2S (sec 5.6:
#: H2S rides on the law).
A_MIN_YSO_LAW = 0.3
#: Psi floored at this fraction of its own row's maximum before
#: `beta * ln Psi` enters a class's evidence (spec sec 6.5, sec 2: "no
#: hypothesis is ever at -inf" -- the cascade may argue against a class,
#: never veto it outright).
PSI_FLOOR_FRAC = 1e-6


def _sensitivity_scale(run, f_dusty_o=None, f_dusty_c=None, f_c=None):
    """`(classes, ln_scale)` for one of the eight fixed literature-band
    runs -- the constant added to every one of `classes`'s whole CLASSMAP
    subclass blocks. `kappa_lo`/`kappa_hi` name both YSO and H2S (module
    docstring). `yso_floor` is not a fixed constant (`_yso_floor_shift`,
    per source) and is not returned here. `f_dusty_lo`/`f_dusty_hi` need
    the region's own `f_dusty_o`, `f_dusty_c`, `f_c` (module docstring).
    """
    ln10 = np.log(10.0)
    if run == "kappa_lo":
        return ("YSO", "H2S"), -KAPPA_DEX * ln10
    if run == "kappa_hi":
        return ("YSO", "H2S"), KAPPA_DEX * ln10
    if run == "eta_lo":
        return ("H2S",), -ETA_DEX * ln10
    if run == "eta_hi":
        return ("H2S",), ETA_DEX * ln10
    if run == "eps_ext_lo":
        return ("H2S",), np.log(EPS_EXT_LO / EPS_EXT_CENTRAL)
    if run == "eps_ext_hi":
        return ("H2S",), np.log(EPS_EXT_HI / EPS_EXT_CENTRAL)
    if run == "f_dusty_lo":
        nominal = (1.0 - f_c) * f_dusty_o + f_c * f_dusty_c
        return ("AGB",), np.log(f_dusty_o / nominal)
    if run == "f_dusty_hi":
        nominal = (1.0 - f_c) * f_dusty_o + f_c * f_dusty_c
        return ("AGB",), np.log(f_dusty_c / nominal)
    raise ValueError("fittp.classify: unknown fixed-scale sensitivity run %r" % run)


def _yso_floor_shift(a_col_k):
    """`ln( max(A_s, A_MIN_YSO_LAW)^2 / A_s^2 )` per source (spec sec 7.2):
    the young-star law floored at low column, not rescaled -- zero at and
    above the floor, growing only as `A_s` falls below it; applied
    identically to YSO and H2S (sec 5.6: H2S's density rides on the same
    law, never its own). `A_s` is `bmstp.density`'s own `A_COL_K` (P1) --
    the adopted column the YSO sky density itself was built from, not the
    catalogue's `AK_SESNA` (a source-derived quantity, spec sec 1.5, never
    read in `fittp`); at NGC 7129 it runs 0.076-0.62 mag with no zeros, so
    the ratio needs no numerical guard.
    """
    a_col_k = np.asarray(a_col_k, dtype=np.float64)
    a_col_k_floored = np.maximum(a_col_k, A_MIN_YSO_LAW)
    return 2.0 * (np.log(a_col_k_floored) - np.log(a_col_k))


def sensitivity_scaling_matrix(f_dusty_o, f_dusty_c, f_c):
    """`(9, 6)` `SCALING`: the factor applied to each class's density in
    each of the eight fixed literature-band runs (1.0 where a run does not
    touch that class). The ninth row (`yso_floor`) is left at 1.0 here --
    its factor is per-source and per-region, filled in at write time
    (`write_sensitivity`) from the region's own mean.
    """
    scaling = np.ones((len(SENSITIVITY_RUNS), len(CLASSES)), dtype=np.float32)
    for ri, run in enumerate(SENSITIVITY_RUNS):
        if run == "yso_floor":
            continue
        classes, ln_scale = _sensitivity_scale(run, f_dusty_o, f_dusty_c, f_c)
        for cls in classes:
            scaling[ri, CLASSES.index(cls)] = np.exp(ln_scale)
    return scaling


def _fit_path(config, region, cls):
    return config_module.product_path(config, "fittp", "fit", cls, "source", region=region)


def _require_fit_files(config, region):
    """Rule 5b: the only existence check. `classify` needs all six
    classes' P7 files; a missing one fails with the one RUNBOOK line that
    makes it, nothing more."""
    for cls in CLASSES:
        path = _fit_path(config, region, cls)
        if not os.path.exists(path):
            raise RuntimeError(
                "fittp.classify [%s]: missing %s -- run RUNBOOKtp.sh's "
                "'PY sesnaimpute.fittp.sweep --classes \"%s\"' line first" % (region, path, cls))


def _cascade_path(config, region):
    return config_module.product_path(config, "fittp", "classification", "cascade", "source", region=region)


def _batch_ln_evidence(class_files, psi_file, beta, start, stop, m):
    """One batch's global `(m, 25)` ln-evidence array: each class's own
    `LN_EVIDENCE` block, shifted by `beta * ln Psi_C(s)` when `beta != 0`
    (spec sec 6.5's logsumexp-shift, see module docstring). `Psi_C` is
    floored at `PSI_FLOOR_FRAC` of its own row's maximum before the log is
    taken (sec 6.5, sec 2) -- read here at classification time only; the
    cascade product on disk keeps its own unfloored numbers.
    """
    ln_ev = np.full((m, N_SUBCLASS), -np.inf, dtype=np.float64)
    ln_psi_by_class = None
    if beta != 0.0:
        psi_raw = np.asarray(psi_file["PSI_CLASS"][start:stop, :], dtype=np.float64)
        psi_floor = PSI_FLOOR_FRAC * psi_raw.max(axis=1, keepdims=True)
        psi_floored = np.maximum(psi_raw, psi_floor)
        with np.errstate(divide="ignore"):
            ln_psi_by_class = np.log(psi_floored)
    for ci, cls in enumerate(CLASSES):
        f = class_files[cls]
        lo, hi = CLASS_SLICES[cls]
        block = np.asarray(f["LN_EVIDENCE"][start:stop, :], dtype=np.float64)
        if beta != 0.0:
            block = block + beta * ln_psi_by_class[:, ci][:, None]
        ln_ev[:, lo:hi] = block
    return ln_ev


def _class_probs(ln_ev):
    """`(P_SUBCLASS, P_CLASS)` from a global `(m, 25)` ln-evidence array:
    one softmax, then each class's column sum (module docstring)."""
    row_max = ln_ev.max(axis=1, keepdims=True)
    weights = np.exp(ln_ev - row_max)
    p_sub = weights / weights.sum(axis=1, keepdims=True)
    p_cls = np.zeros((ln_ev.shape[0], len(CLASSES)), dtype=np.float64)
    for ci, cls in enumerate(CLASSES):
        lo, hi = CLASS_SLICES[cls]
        p_cls[:, ci] = p_sub[:, lo:hi].sum(axis=1)
    return p_sub, p_cls


def _part_path(path, bi):
    return "%s.part%d" % (path, bi)


#: The datasets every P8 part file and the joined product carry. `A_K_POST`/
#: `A_K_POST_SIG` are `(n, 6)` in `CLASSES` order (module docstring), read
#: off the six fit files' own columns of the same name exactly as
#: `LN_EVIDENCE` -- no MAP-class column: a reader picks the class it wants
#: and divides `A_K_POST` by P1's own `A_COL_K` (POSTMARK brief item 3).
_CLASSIFY_PART_KEYS = ("NAME", "P_CLASS", "P_SUBCLASS", "P_YSO", "MAP_CLASS", "N_DETECTED",
                       "CANDIDATE_FLUX", "FLUX_IMPUTED", "FLUX_IMPUTED_COV",
                       "A_K_POST", "A_K_POST_SIG",
                       "ENTROPY_CLASS", "ENTROPY_SUBCLASS")


def _classify_batch(class_files, psi_file, beta, cat_path, start, stop):
    """One ROW_BYTES batch's own P8 rows (rule 10b): a batch-sized array
    only, never a region-sized one."""
    m = stop - start
    flux_mean_stack = np.empty((len(CLASSES), m, N_BANDS), dtype=np.float64)
    flux_cov_stack = np.empty((len(CLASSES), m, N_BANDS, N_BANDS), dtype=np.float64)
    a_k_post_stack = np.empty((len(CLASSES), m), dtype=np.float32)
    a_k_post_sig_stack = np.empty((len(CLASSES), m), dtype=np.float32)
    for ci, cls in enumerate(CLASSES):
        f = class_files[cls]
        flux_mean_stack[ci] = np.asarray(f["FLUX_MEAN"][start:stop, :], dtype=np.float64)
        flux_cov_stack[ci] = np.asarray(f["FLUX_COV"][start:stop, :, :], dtype=np.float64)
        a_k_post_stack[ci] = np.asarray(f["A_K_POST"][start:stop], dtype=np.float32)
        a_k_post_sig_stack[ci] = np.asarray(f["A_K_POST_SIG"][start:stop], dtype=np.float32)

    ln_ev = _batch_ln_evidence(class_files, psi_file, beta, start, stop, m)
    # A flagged source's fit is undefined at every template of every class
    # (sweep.py sets ln_w to -inf there), so its whole (25,) evidence row
    # is -inf and the identities below must stop testing it, not turn it
    # into a fabricated STAR verdict (R3 U1).
    flagged = ~np.isfinite(ln_ev).any(axis=1)
    p_sub, p_cls = _class_probs(ln_ev)
    map_c = np.argmax(p_cls, axis=1)
    map_c = np.where(flagged, -1, map_c)

    with h5py.File(cat_path, "r") as cf:
        flux = np.asarray(cf["FNU_MJY"][start:stop], dtype=np.float64)
        origin = np.asarray(cf["ORIGIN_FNU"][start:stop])
    detected = origin == 1

    cflux = np.transpose(flux_mean_stack, (1, 0, 2)).copy()  # (m, 6, 8)
    mask = np.broadcast_to(detected[:, None, :], cflux.shape)
    cflux = np.where(mask, flux[:, None, :], cflux)
    row_idx = np.arange(m)
    # a safe (in-range) class index for the gather below; the flagged rows
    # it touches are overwritten with NaN immediately after, never read.
    map_c_safe = np.where(flagged, 0, map_c)
    imputed = cflux[row_idx, map_c_safe, :]
    imputed_cov = flux_cov_stack[map_c_safe, row_idx]
    imputed[flagged] = np.nan
    imputed_cov[flagged] = np.nan
    detected_ok = detected & ~flagged[:, None]
    imputed_identity_err = float(np.max(np.abs(imputed[detected_ok] - flux[detected_ok]))) \
        if detected_ok.any() else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ent_c = -np.sum(np.where(p_cls > 0, p_cls * np.log(p_cls), 0.0), axis=1)
        ent_s = -np.sum(np.where(p_sub > 0, p_sub * np.log(p_sub), 0.0), axis=1)

    # A_K_POST/A_K_POST_SIG, (m, 6) in CLASSES order (module docstring):
    # read off the six fit files' own columns, transposed to a row per
    # source -- no MAP-class reduction, unlike CANDIDATE_FLUX/FLUX_IMPUTED
    # above (POSTMARK brief item 3).
    a_k_post = np.ascontiguousarray(a_k_post_stack.T)
    a_k_post_sig = np.ascontiguousarray(a_k_post_sig_stack.T)

    return dict(
        p_class=p_cls.astype(np.float32), p_subclass=p_sub.astype(np.float32),
        map_class=map_c.astype(np.int8), n_detected=detected.sum(axis=1).astype(np.int8),
        candidate_flux=cflux.astype(np.float32), flux_imputed=imputed.astype(np.float32),
        flux_imputed_cov=imputed_cov.astype(np.float32),
        a_k_post=a_k_post, a_k_post_sig=a_k_post_sig,
        entropy_class=ent_c.astype(np.float32), entropy_subclass=ent_s.astype(np.float32),
        imputed_identity_err=imputed_identity_err, n_flagged=int(flagged.sum()),
    )


def _write_classify_part(part_path, batch):
    with h5py.File(part_path, "w") as f:
        f.create_dataset("NAME", data=batch["name"])
        f.create_dataset("P_CLASS", data=batch["p_class"])
        f.create_dataset("P_SUBCLASS", data=batch["p_subclass"])
        f.create_dataset("P_YSO", data=batch["p_class"][:, YSO_INDEX])
        f.create_dataset("MAP_CLASS", data=batch["map_class"])
        f.create_dataset("N_DETECTED", data=batch["n_detected"])
        f.create_dataset("CANDIDATE_FLUX", data=batch["candidate_flux"])
        f.create_dataset("FLUX_IMPUTED", data=batch["flux_imputed"])
        f.create_dataset("FLUX_IMPUTED_COV", data=batch["flux_imputed_cov"])
        f.create_dataset("A_K_POST", data=batch["a_k_post"])
        f.create_dataset("A_K_POST_SIG", data=batch["a_k_post_sig"])
        f.create_dataset("ENTROPY_CLASS", data=batch["entropy_class"])
        f.create_dataset("ENTROPY_SUBCLASS", data=batch["entropy_subclass"])


def build_region(config, region, st, beta):
    """One region's P8, written one `ROW_BYTES` batch's own part file at a
    time (rule 10b: `CANDIDATE_FLUX` and `FLUX_IMPUTED_COV` are the two
    region-sized arrays the W7 review found here); the caller joins the
    parts once every batch is done."""
    _require_fit_files(config, region)

    class_files = {}
    names = None
    for cls in CLASSES:
        f = h5py.File(_fit_path(config, region, cls), "r")
        class_files[cls] = f
        this_names = f["NAME"][:]
        if names is None:
            names = this_names
        elif not np.array_equal(names, this_names):
            raise ValueError("fittp.classify [%s]: %s's NAME does not row-align with STAR's"
                              % (region, cls))

    psi_file = None
    if beta != 0.0:
        psi_path = _cascade_path(config, region)
        if not os.path.exists(psi_path):
            raise RuntimeError(
                "fittp.classify [%s]: beta=%.3g needs %s -- run RUNBOOKtp.sh's "
                "'PY sesnaimpute.fittp.cascade' line first" % (region, beta, psi_path))
        psi_file = h5py.File(psi_path, "r")
        if not np.array_equal(psi_file["NAME"][:], names):
            raise ValueError("fittp.classify [%s]: cascade's NAME does not row-align" % region)

    cat_path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)

    n = names.shape[0]
    path = config_module.product_path(config, "fittp", "classification", "posterior",
                                       "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    part_paths = []
    imputed_identity_err = 0.0
    n_flagged = 0
    bounds = list(batches(n, ROW_BYTES))
    for bi, (start, stop) in enumerate(bounds):
        batch = _classify_batch(class_files, psi_file, beta, cat_path, start, stop)
        batch["name"] = names[start:stop]
        part_path = _part_path(path, bi)
        _write_classify_part(part_path, batch)
        part_paths.append(part_path)
        imputed_identity_err = max(imputed_identity_err, batch["imputed_identity_err"])
        n_flagged += batch["n_flagged"]
        st.tick(bi + 1, len(bounds), "batches")

    for f in class_files.values():
        f.close()
    if psi_file is not None:
        psi_file.close()

    fit_files = np.array([_fit_path(config, region, cls) for cls in CLASSES], dtype="S256")
    return dict(path=path, part_paths=part_paths, n_source=n, fit_files=fit_files,
                imputed_identity_err=imputed_identity_err, n_flagged=n_flagged)


def join_classify_parts(path, part_paths, n_source, fit_files, n_flagged):
    """Joins one region's P8 part files, one part's rows at a time,
    dataset by dataset (rule 10b: never a region-sized array); removes the
    part files once written."""
    with h5py.File(path, "w") as out:
        with h5py.File(part_paths[0], "r") as pf0:
            for key in _CLASSIFY_PART_KEYS:
                shape = (n_source,) + pf0[key].shape[1:]
                out.create_dataset(key, shape=shape, dtype=pf0[key].dtype)
        offset = 0
        for part_path in part_paths:
            with h5py.File(part_path, "r") as pf:
                m = pf["NAME"].shape[0]
                for key in _CLASSIFY_PART_KEYS:
                    out[key][offset:offset + m] = pf[key][:]
            offset += m
        out.attrs["GRANULE"] = "source"
        out.attrs["CLASSES"] = np.array(CLASSES, dtype="S8")
        out.attrs["SUBCLASSES"] = np.array(SUBCLASS_LABELS, dtype="S12")
        out.attrs["FIT_FILES"] = fit_files
        # sources flagged by the fit (n_detected < 2, or a singular design
        # matrix): MAP_CLASS is -1 for these, never STAR, and P_CLASS/
        # P_SUBCLASS/P_YSO/FLUX_IMPUTED are NaN (R3 U1, U2) -- recorded
        # once here rather than recomputed by every consumer.
        out.attrs["N_FLAGGED"] = n_flagged
    for part_path in part_paths:
        os.remove(part_path)


def run_sensitivity_region(config, region, st, beta):
    """One region's row of P9, the literature-band sensitivity (spec sec
    7.2): for each of `SENSITIVITY_RUNS`, `classify`'s own nominal
    classification (this same `beta`) re-run with one class's ln evidence
    shifted by `ln(scale)`, or (`yso_floor`) YSO's and H2S's ln evidence
    shifted per source by `_yso_floor_shift` -- classification-time only,
    no refit, batched with the classify build. Returns `n_source`,
    `frac_map_changed` (9,), `n_pyso_above_half` (10,, column 0 nominal),
    `yso_floor_mean_factor` (this region's mean over sources of
    `exp(_yso_floor_shift(a_col_k))`, for `SCALING`'s ninth row).
    """
    _require_fit_files(config, region)
    class_files = {cls: h5py.File(_fit_path(config, region, cls), "r") for cls in CLASSES}
    names = class_files["STAR"]["NAME"][:]
    psi_file = h5py.File(_cascade_path(config, region), "r") if beta != 0.0 else None

    # A_COL_K (P1, bmstp.density): the adopted column the YSO sky density
    # was itself built from, never the catalogue's own AK_SESNA (spec sec
    # 1.5: a source-derived quantity, not read anywhere in fittp).
    density_path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    density_file = h5py.File(density_path, "r")
    if not np.array_equal(density_file["NAME"][:], names):
        raise ValueError("fittp.classify [%s]: bmstp.density's NAME does not row-align "
                          "with the fit files' own" % region)
    f_dusty_o = float(density_file.attrs["F_DUSTY_O"])
    f_dusty_c = float(density_file.attrs["F_DUSTY_C"])
    f_c = float(density_file.attrs["F_C"])

    n = names.shape[0]
    n_run = len(SENSITIVITY_RUNS)
    yso_floor_ri = SENSITIVITY_RUNS.index("yso_floor")
    changed = np.zeros(n_run, dtype=np.int64)
    n_pyso = np.zeros(n_run + 1, dtype=np.int64)
    yso_floor_factor_sum = 0.0

    bounds = list(batches(n, ROW_BYTES))
    for bi, (start, stop) in enumerate(bounds):
        m = stop - start
        a_col_k_block = np.asarray(density_file["A_COL_K"][start:stop], dtype=np.float64)
        yso_floor_shift = _yso_floor_shift(a_col_k_block)
        yso_floor_factor_sum += float(np.sum(np.exp(yso_floor_shift)))

        ln_ev_nominal = _batch_ln_evidence(class_files, psi_file, beta, start, stop, m)
        _, p_cls_nom = _class_probs(ln_ev_nominal)
        map_nom = np.argmax(p_cls_nom, axis=1)
        n_pyso[0] += int((p_cls_nom[:, YSO_INDEX] > 0.5).sum())

        for ri, run in enumerate(SENSITIVITY_RUNS):
            ln_ev_run = ln_ev_nominal.copy()
            if run == "yso_floor":
                for cls in ("YSO", "H2S"):
                    lo, hi = CLASS_SLICES[cls]
                    ln_ev_run[:, lo:hi] += yso_floor_shift[:, None]
            else:
                classes, ln_scale = _sensitivity_scale(run, f_dusty_o, f_dusty_c, f_c)
                for cls in classes:
                    lo, hi = CLASS_SLICES[cls]
                    ln_ev_run[:, lo:hi] += ln_scale
            _, p_cls_run = _class_probs(ln_ev_run)
            map_run = np.argmax(p_cls_run, axis=1)
            changed[ri] += int((map_run != map_nom).sum())
            n_pyso[ri + 1] += int((p_cls_run[:, YSO_INDEX] > 0.5).sum())
        st.tick(bi + 1, len(bounds), "batches")

    for f in class_files.values():
        f.close()
    if psi_file is not None:
        psi_file.close()
    density_file.close()

    return dict(n_source=n, frac_map_changed=(changed / n if n else changed.astype(np.float64)),
                n_pyso_above_half=n_pyso,
                yso_floor_mean_factor=(yso_floor_factor_sum / n if n else float("nan")),
                f_dusty_o=f_dusty_o, f_dusty_c=f_dusty_c, f_c=f_c)


def write_sensitivity(path, region, result):
    """Updates the region's own row of the (30-region) P9 product in
    place, leaving every other region's row untouched (rule 5c). `SCALING`
    is per-region because `yso_floor`'s factor is (module docstring), and
    now so is `f_dusty_lo`/`f_dusty_hi`'s (read off `bmstp.density`'s own
    product, `run_sensitivity_region`); its other six rows repeat the same
    fixed literature-band factor in every region's slice.
    """
    region_names = tuple(r.name for r in regions_module.REGIONS)
    n_region = len(region_names)
    n_run = len(SENSITIVITY_RUNS)
    n_cls = len(CLASSES)
    yso_floor_ri = SENSITIVITY_RUNS.index("yso_floor")
    fixed = sensitivity_scaling_matrix(result["f_dusty_o"], result["f_dusty_c"], result["f_c"])
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            frac_map_changed = np.asarray(f["FRAC_MAP_CHANGED"][:])
            n_pyso_above_half = np.asarray(f["N_PYSO_ABOVE_HALF"][:])
            n_sources = np.asarray(f["N_SOURCES"][:])
            scaling_on_disk = np.asarray(f["SCALING"][:])
    else:
        frac_map_changed = np.full((n_region, n_run), np.nan, dtype=np.float32)
        n_pyso_above_half = np.full((n_region, n_run + 1), -1, dtype=np.int32)
        n_sources = np.zeros(n_region, dtype=np.int32)
        scaling_on_disk = None
    # SCALING is per-region (`yso_floor`'s factor varies by region, module
    # docstring): a product written before this fix carries the old (n_run,
    # n_cls) shape, which does not carry a per-region yso_floor factor, so
    # it is rebuilt fresh rather than reshaped.
    if scaling_on_disk is not None and scaling_on_disk.shape == (n_region, n_run, n_cls):
        scaling = scaling_on_disk
    else:
        scaling = np.broadcast_to(fixed, (n_region, n_run, n_cls)).copy()

    ridx = region_names.index(region)
    frac_map_changed[ridx] = result["frac_map_changed"]
    n_pyso_above_half[ridx] = result["n_pyso_above_half"]
    n_sources[ridx] = result["n_source"]
    scaling[ridx, yso_floor_ri, :] = 1.0
    for cls in ("YSO", "H2S"):
        scaling[ridx, yso_floor_ri, CLASSES.index(cls)] = result["yso_floor_mean_factor"]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("REGION", data=np.array(region_names, dtype="S32"))
        f.create_dataset("RUN", data=np.array(SENSITIVITY_RUNS, dtype="S16"))
        f.create_dataset("SCALING", data=scaling.astype(np.float32))
        f.create_dataset("FRAC_MAP_CHANGED", data=frac_map_changed.astype(np.float32))
        f.create_dataset("N_PYSO_ABOVE_HALF", data=n_pyso_above_half.astype(np.int32))
        f.create_dataset("N_SOURCES", data=n_sources.astype(np.int32))
        f.attrs["GRANULE"] = "region"
        f.attrs["SCALING_YSO_FLOOR_IS_PER_SOURCE"] = (
            "SCALING row %d (yso_floor) is each region's own mean over sources of "
            "exp(ln(max(A,%.2g)^2/A^2)) (A = bmstp.density's A_COL_K), not a fixed "
            "literature-band factor like the other eight rows" % (yso_floor_ri, A_MIN_YSO_LAW))


def build(config, regions=None, beta=0.0):
    """Writes `fittp/classification/posterior_classification_source__R.hdf5`
    for `regions` (default all thirty), one file per region
    (IMPLEMENTATION_BMSTP_DRAFT.md P8; SPEC_BMSTP_DRAFT.md section 7.1).
    `beta` is the cascade tempering dial of section 6.5 (`[fittp] beta`
    in root.cfg), applied at classification time with no refit.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        with progress.Stage("fittp.classify", region) as st:
            result = build_region(config, region, st, beta)
            join_classify_parts(result["path"], result["part_paths"],
                                 result["n_source"], result["fit_files"], result["n_flagged"])

            # the joined file's own small columns (n, 6) and (n, 25) --
            # not CANDIDATE_FLUX/FLUX_IMPUTED_COV, the two region-sized
            # arrays rule 10b keeps out of memory (W7 review finding 6).
            with h5py.File(result["path"], "r") as f:
                p_class = np.asarray(f["P_CLASS"][:])
                p_subclass = np.asarray(f["P_SUBCLASS"][:])
                n_detected = np.asarray(f["N_DETECTED"][:])
                map_class = np.asarray(f["MAP_CLASS"][:])

            # a flagged source (MAP_CLASS == -1) carries a NaN P_CLASS/
            # P_SUBCLASS row by construction (R3 U1); both acceptance
            # identities are tested only on the sources the fit actually
            # resolved, so one flagged source no longer stops either check.
            ok = map_class != -1
            p_class_err = float(np.max(np.abs(p_class[ok].sum(axis=1) - 1.0))) if ok.any() else 0.0
            sub_sum = np.zeros_like(p_class)
            for ci, cls in enumerate(CLASSES):
                lo, hi = CLASS_SLICES[cls]
                sub_sum[:, ci] = p_subclass[:, lo:hi].sum(axis=1)
            subclass_err = float(np.max(np.abs(sub_sum[ok] - p_class[ok]))) if ok.any() else 0.0
            n_pyso_half = int((p_class[:, YSO_INDEX] > 0.5).sum())
            two_band_frac = float((n_detected == 2).mean()) if n_detected.size else float("nan")
            st.done(result["path"], n=result["n_source"], beta=beta,
                    p_class_sum_err=p_class_err, p_subclass_sum_err=subclass_err,
                    flux_imputed_identity_err=result["imputed_identity_err"],
                    n_pyso_above_half=n_pyso_half, two_band_frac=two_band_frac,
                    n_flagged=result["n_flagged"], n_batches=len(result["part_paths"]))

        with progress.Stage("fittp.classify.sensitivity", region) as st:
            sens = run_sensitivity_region(config, region, st, beta)
            sens_path = config_module.product_path(config, "fittp", "classification", "sensitivity", "region")
            write_sensitivity(sens_path, region, sens)
            st.done(sens_path, n=sens["n_source"],
                    frac_map_changed_range="%.4g-%.4g" % (sens["frac_map_changed"].min(),
                                                           sens["frac_map_changed"].max()),
                    n_pyso_nominal=int(sens["n_pyso_above_half"][0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--beta", type=float, default=None,
                         help="default: root.cfg's [fittp] beta (0.0 if absent)")
    args = parser.parse_args()
    cfg = config_module.load(args.config)
    if args.beta is None:
        ini = configparser.ConfigParser()
        ini.read(args.config)
        beta = ini.getfloat("fittp", "beta", fallback=0.0)
    else:
        beta = args.beta
    build(cfg, regions=args.regions, beta=beta)
