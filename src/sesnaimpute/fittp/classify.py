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
hold by construction, not by a separate normalisation. At `beta != 0`,
`Psi_C(s)` reads the cascade's own `P_VERDICT_MEASURED` (P10) against the
emission table `E[k, b, v]` (`fittp.emission`, `briefs/EMISSION.md`); that
reader is not wired here -- `briefs/EMISSION.md` scopes its own consumer
(a per-template read inside the fit, not a per-class one at classification
time) as a separate, later piece of work, so this module raises rather
than approximate it. `beta = 0` is the only supported value today, and at
`beta = 0` nothing of the cascade is read.

`A_K_POST`, `A_K_POST_SIG` (n, 6), `CLASSES` order, carry each class's own
posterior extinction mark and its spread (P7's own columns of the same
name, SPEC_BMSTP_DRAFT.md section 6.1) straight through, unreduced by
`MAP_CLASS`: a reader divides the class column it wants by P1's own
`A_COL_K` (`bmstp/density/table_density_source__<R>.hdf5`) to form
`XI_POST` itself.

`LOG10_FLUX_IMPUTED` (n, 8), the one scale section 7.1 stores the imputed
photometry on: the MAP class's own `LOG10_FLUX_MEAN` on the bands it did
not measure, and `log10(FNU_MJY)` on the bands it did -- the catalogue's
own measurement, recoverable to float32 with no join back to the
catalogue. `LOG10_CANDIDATE_FLUX` (n, 6, 8) is the same construction for
every class, not only the MAP one (each class's own `LOG10_FLUX_MEAN` with
the measured bands restored).

`LOG10_FLUX_IMPUTED_COV` (n, 8, 8), section 6.1's `m`/`u` block structure
(measured bands `m`, imputed bands `u`): on `[m, m]` the catalogue's own
variance in log10 flux, `(SIGMA_FNU_MJY / (FNU_MJY ln 10))^2`, diagonal,
zero off-diagonal (two catalogue errors are independent by band); on
`[u, u']` the MAP class's own `LOG10_FLUX_COV` (P7) unchanged, between plus
within; on `[m, u]` (and its transpose `[u, m]`) the fit's own
cross-covariance between a measured and an imputed band, `sum_k P_k (D_k
Sigma_k D_k^T)[m, u]`, `P_k` the MAP class's stored `P_DENSE` (and `1 -
P_DENSE`) and `Sigma_k = (X^T W X)^-1` rebuilt here (never stored) from the
catalogue's own measured-band weights (`likelihood.SIGMA_CAL_DEX`, this
source's MAP class's own `sigma_lib,L`) and `D_k`'s design from `config`
at law `k` (`population.selection.kappa_hybrid`/`ak_per_av`, `likelihood
.GRAY_COLUMN`) -- the same two matrix products `likelihood.fit` forms per
source and design, since neither depends on which template is being
scored, so no template loop is needed to rebuild it. This block is `Sigma_k
= (X^T W X)^-1` itself, not a per-template quantity (section 6.1): the
between-template term contributes nothing to `[m, u]`, since a measured
band's own datum does not vary across templates.
"""

import argparse
import configparser
import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute.constants import GUTERMUTH_LABELS
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.batches import batches
from sesnaimpute.fittp import likelihood
from sesnaimpute.population import selection as population_selection
from sesnaimpute.readings import set_readings

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
#: LN_EVIDENCE (<=9 cols), LOG10_FLUX_MEAN (8) and LOG10_FLUX_COV (8x8)
#: float32 rows, plus the measured flux/sigma/origin and the global (25)
#: and (6,8,8) intermediates, at a generous margin.
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
    (spec sec 6.5's logsumexp-shift, see module docstring). `beta != 0`
    needs `Psi_C(s)` read from `psi_file`'s own `P_VERDICT_MEASURED` (P10)
    against the emission table, which is not wired here (module
    docstring); `beta = 0` is the only supported value.
    """
    if beta != 0.0:
        raise NotImplementedError(
            "fittp.classify: beta=%.3g needs Psi_C(s) from P_VERDICT_MEASURED and the "
            "emission table -- that reader is briefs/EMISSION.md's own scope, deferred "
            "there; beta = 0 is the only supported value today" % beta)
    ln_ev = np.full((m, N_SUBCLASS), -np.inf, dtype=np.float64)
    for ci, cls in enumerate(CLASSES):
        f = class_files[cls]
        lo, hi = CLASS_SLICES[cls]
        block = np.asarray(f["LN_EVIDENCE"][start:stop, :], dtype=np.float64)
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
#: and divides `A_K_POST` by P1's own `A_COL_K`.
_CLASSIFY_PART_KEYS = ("NAME", "CLASS_SESNA", "P_CLASS", "P_SUBCLASS", "P_YSO", "MAP_CLASS", "N_DETECTED",
                       "LOG10_CANDIDATE_FLUX", "LOG10_FLUX_IMPUTED", "LOG10_FLUX_IMPUTED_COV",
                       "A_K_POST", "A_K_POST_SIG",
                       "ENTROPY_CLASS", "ENTROPY_SUBCLASS")

#: SPEC_BMSTP_DRAFT.md section 1/6.1 -- the log10-flux Jacobian, `sigma_log
#: = sigma_f / (f ln 10)`, the same conversion `fittp.likelihood.prepare`
#: uses for the fit's own per-band variance.
_LN10 = np.log(10.0)


def _cross_cov_by_law(config, weight):
    """`[D_k Sigma_k D_k^T]` for both extinction-law designs, `k = 0`
    (diffuse) and `k = 1` (dense), vectorised over the batch's own `m`
    sources at once (module docstring): `Sigma_k = (X^T W X)^-1` is this
    design's `(2, 2)` parameter covariance, `weight` (m, 8) the per-source,
    per-band measurement weight already carrying the catalogue sigma, the
    calibration floor and each source's own MAP-class `sigma_lib,L`
    (`likelihood.prepare`'s own formula). `D_k = [ext_col_k, GRAY_COLUMN]` does not depend on the
    source (`likelihood.fit`'s own `ext_col`/`GRAY_COLUMN`), so it is built
    once per design and broadcast, never recomputed per source. Returns
    `(cross_0, cross_1)`, each `(m, 8, 8)`.
    """
    gray = np.full(N_BANDS, likelihood.GRAY_COLUMN, dtype=np.float64)
    cross = []
    for k in (0, 1):
        ext_col_k = -0.4 * population_selection.kappa_hybrid(config, float(k)) \
            * float(population_selection.ak_per_av(config, float(k)))
        design_k = np.column_stack([ext_col_k, gray])            # (8, 2)
        wx = design_k[None, :, :] * weight[:, :, None]            # (m, 8, 2)
        xtwx = np.einsum("mbi,bj->mij", wx, design_k)             # (m, 2, 2)
        det = xtwx[:, 0, 0] * xtwx[:, 1, 1] - xtwx[:, 0, 1] * xtwx[:, 1, 0]
        safe_det = np.where(det > 0, det, 1.0)
        xtwx_inv = np.empty_like(xtwx)
        xtwx_inv[:, 0, 0] = xtwx[:, 1, 1] / safe_det
        xtwx_inv[:, 1, 1] = xtwx[:, 0, 0] / safe_det
        xtwx_inv[:, 0, 1] = -xtwx[:, 0, 1] / safe_det
        xtwx_inv[:, 1, 0] = -xtwx[:, 1, 0] / safe_det
        cross.append(np.einsum("bi,mij,cj->mbc", design_k, xtwx_inv, design_k))
    return cross[0], cross[1]


def _classify_batch(config, class_files, psi_file, beta, cat_path, start, stop, sigma_lib_vals):
    """One ROW_BYTES batch's own P8 rows (rule 10b): a batch-sized array
    only, never a region-sized one."""
    m = stop - start
    log10_flux_mean_stack = np.empty((len(CLASSES), m, N_BANDS), dtype=np.float64)
    flux_cov_stack = np.empty((len(CLASSES), m, N_BANDS, N_BANDS), dtype=np.float64)
    a_k_post_stack = np.empty((len(CLASSES), m), dtype=np.float32)
    a_k_post_sig_stack = np.empty((len(CLASSES), m), dtype=np.float32)
    p_dense_stack = np.empty((len(CLASSES), m), dtype=np.float64)
    for ci, cls in enumerate(CLASSES):
        f = class_files[cls]
        log10_flux_mean_stack[ci] = np.asarray(f["LOG10_FLUX_MEAN"][start:stop, :], dtype=np.float64)
        flux_cov_stack[ci] = np.asarray(f["LOG10_FLUX_COV"][start:stop, :, :], dtype=np.float64)
        a_k_post_stack[ci] = np.asarray(f["A_K_POST"][start:stop], dtype=np.float32)
        a_k_post_sig_stack[ci] = np.asarray(f["A_K_POST_SIG"][start:stop], dtype=np.float32)
        p_dense_stack[ci] = np.asarray(f["P_DENSE"][start:stop], dtype=np.float64)

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
        sigma = np.asarray(cf["SIGMA_FNU_MJY"][start:stop], dtype=np.float64)
        origin = np.asarray(cf["ORIGIN_FNU"][start:stop])
        # the class SESNA delivered for these sources, carried through so the
        # catalogue's own label is in the product the classification is read
        # from; nothing in the pipeline reads it (rule 7).
        class_sesna = np.asarray(cf["CLASS"][start:stop], dtype=np.int64)
    unknown = set(np.unique(class_sesna).tolist()) - set(GUTERMUTH_LABELS.index.tolist())
    if unknown:
        raise ValueError("fittp.classify: catalogue CLASS codes absent from "
                          "constants.GUTERMUTH_LABELS: %s" % sorted(unknown))
    detected = (origin == 1) & (flux > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        log10_flux_meas = np.log10(np.where(detected, flux, 1.0))
    cflux = np.transpose(log10_flux_mean_stack, (1, 0, 2)).copy()  # (m, 6, 8)
    mask = np.broadcast_to(detected[:, None, :], cflux.shape)
    cflux = np.where(mask, log10_flux_meas[:, None, :], cflux)
    row_idx = np.arange(m)
    # a safe (in-range) class index for the gather below; the flagged rows
    # it touches are overwritten with NaN immediately after, never read.
    map_c_safe = np.where(flagged, 0, map_c)
    imputed = cflux[row_idx, map_c_safe, :]
    imputed_cov = flux_cov_stack[map_c_safe, row_idx]

    # LOG10_FLUX_IMPUTED_COV (module docstring, SPEC_BMSTP_DRAFT.md 6.1):
    # [u, u'] (both imputed) keeps the MAP class's own LOG10_FLUX_COV
    # untouched; [m, m'] (both measured) is zeroed, [m, m] (the diagonal)
    # is overwritten by the catalogue's own log10-flux variance below; and
    # the mixed block [m, u] carries the fit's own cross-covariance,
    # rebuilt here from the MAP class's own sigma_lib,L and its stored
    # P_DENSE split between the two extinction-law designs.
    mm_either = detected[:, :, None] | detected[:, None, :]
    cross_mask = detected[:, :, None] ^ detected[:, None, :]
    imputed_cov = np.where(mm_either, 0.0, imputed_cov)

    safe_flux = np.where(detected, flux, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_log = sigma / (safe_flux * _LN10)
    sigma_lib_map = sigma_lib_vals[map_c_safe]                        # (m,)
    sigma2 = sigma_log ** 2 + likelihood.SIGMA_CAL_DEX[None, :] ** 2 + sigma_lib_map[:, None] ** 2
    weight = np.where(detected & (sigma_log > 0), 1.0 / sigma2, 0.0)  # (m, 8)
    cross_0, cross_1 = _cross_cov_by_law(config, weight)
    p_dense_map = p_dense_stack[map_c_safe, row_idx]                  # (m,)
    cross_total = (1.0 - p_dense_map)[:, None, None] * cross_0 + p_dense_map[:, None, None] * cross_1
    imputed_cov = np.where(cross_mask, cross_total, imputed_cov)

    band_idx = np.arange(N_BANDS)
    with np.errstate(divide="ignore", invalid="ignore"):
        catalogue_var_log10 = (sigma / (flux * _LN10)) ** 2
    diag = np.where(detected, catalogue_var_log10, imputed_cov[:, band_idx, band_idx])
    imputed_cov[:, band_idx, band_idx] = diag

    imputed[flagged] = np.nan
    imputed_cov[flagged] = np.nan
    detected_ok = detected & ~flagged[:, None]
    imputed_identity_err = float(np.max(np.abs(10.0 ** imputed[detected_ok] - flux[detected_ok]))) \
        if detected_ok.any() else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ent_c = -np.sum(np.where(p_cls > 0, p_cls * np.log(p_cls), 0.0), axis=1)
        ent_s = -np.sum(np.where(p_sub > 0, p_sub * np.log(p_sub), 0.0), axis=1)

    # A_K_POST/A_K_POST_SIG, (m, 6) in CLASSES order (module docstring):
    # read off the six fit files' own columns, transposed to a row per
    # source -- no MAP-class reduction, unlike LOG10_CANDIDATE_FLUX/
    # LOG10_FLUX_IMPUTED above.
    a_k_post = np.ascontiguousarray(a_k_post_stack.T)
    a_k_post_sig = np.ascontiguousarray(a_k_post_sig_stack.T)

    return dict(
        class_sesna=class_sesna.astype(np.int16),
        p_class=p_cls.astype(np.float32), p_subclass=p_sub.astype(np.float32),
        map_class=map_c.astype(np.int8), n_detected=detected.sum(axis=1).astype(np.int8),
        log10_candidate_flux=cflux.astype(np.float32), log10_flux_imputed=imputed.astype(np.float32),
        log10_flux_imputed_cov=imputed_cov.astype(np.float32),
        a_k_post=a_k_post, a_k_post_sig=a_k_post_sig,
        entropy_class=ent_c.astype(np.float32), entropy_subclass=ent_s.astype(np.float32),
        imputed_identity_err=imputed_identity_err, n_flagged=int(flagged.sum()),
    )


def _write_classify_part(part_path, batch):
    with h5py.File(part_path, "w") as f:
        f.create_dataset("NAME", data=batch["name"])
        f.create_dataset("CLASS_SESNA", data=batch["class_sesna"])
        f.create_dataset("P_CLASS", data=batch["p_class"])
        f.create_dataset("P_SUBCLASS", data=batch["p_subclass"])
        f.create_dataset("P_YSO", data=batch["p_class"][:, YSO_INDEX])
        f.create_dataset("MAP_CLASS", data=batch["map_class"])
        f.create_dataset("N_DETECTED", data=batch["n_detected"])
        f.create_dataset("LOG10_CANDIDATE_FLUX", data=batch["log10_candidate_flux"])
        f.create_dataset("LOG10_FLUX_IMPUTED", data=batch["log10_flux_imputed"])
        f.create_dataset("LOG10_FLUX_IMPUTED_COV", data=batch["log10_flux_imputed_cov"])
        f.create_dataset("A_K_POST", data=batch["a_k_post"])
        f.create_dataset("A_K_POST_SIG", data=batch["a_k_post_sig"])
        f.create_dataset("ENTROPY_CLASS", data=batch["entropy_class"])
        f.create_dataset("ENTROPY_SUBCLASS", data=batch["entropy_subclass"])


def build_region(config, region, st, beta):
    """One region's P8, written one `ROW_BYTES` batch's own part file at a
    time (rule 10b: `LOG10_CANDIDATE_FLUX` and `LOG10_FLUX_IMPUTED_COV` are
    the two region-sized arrays the W7 review found here); the caller
    joins the parts once every batch is done."""
    _require_fit_files(config, region)

    # section 6.1's sigma_lib,L, one number per class's own library,
    # gathered once here (never per batch or per source) so `_classify_
    # batch` rebuilds the MAP class's own measurement weight exactly as
    # `likelihood.prepare` built it at fit time.
    lib_path = config_module.product_path(config, "fittp", "check", "library-resolution", "survey")
    sigma_lib_by_cls = likelihood.sigma_lib_by_class(lib_path)
    sigma_lib_vals = np.array([sigma_lib_by_cls[cls] for cls in CLASSES], dtype=np.float64)

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
        batch = _classify_batch(config, class_files, psi_file, beta, cat_path, start, stop, sigma_lib_vals)
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


#: `UNITS`/`READING` (CODING_RULES_BMSTP.md rule 5): every dataset the
#: joined P8 posterior product carries.
_READINGS = {
    "NAME": ("source name",
        "The source's name as the SESNA catalog gives it. Rows follow the catalog's "
        "own order."),
    "CLASS_SESNA": ("SESNA class code",
        "The classification SESNA delivered for this source, copied from the catalog. "
        "The codes are 0 deeply embedded protostar, 1 class I protostar, 2 class II, "
        "3 transition disk, 9 H2 shock blob, 19 PAH emitter (star-forming galaxy), 29 "
        "AGN, 39 PAH-contaminated source, 49 generic galaxy, 99 diskless star, -100 "
        "unclassified. Nothing in the pipeline reads this column, so it does not "
        "affect any other number here."),
    "N_DETECTED": ("bands",
        "How many of the source's eight bands (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 "
        "micron) hold a measured, positive flux. The remaining bands are upper limits "
        "or were never observed at this position."),
    "P_CLASS": ("probability",
        "The probability that the source belongs to each of six classes: a field "
        "star, a dusty evolved star, an aperture contaminated by nebular emission, a "
        "background galaxy, a young stellar object, or a knot of shocked gas. The "
        "column order is the CLASSES attribute of this file. The six probabilities "
        "add to 1 for every source."),
    "P_SUBCLASS": ("probability",
        "The probability of each of 25 subdivisions of the six classes, such as "
        "spectral type for a field star or evolutionary stage for a young stellar "
        "object. The column order is the SUBCLASSES attribute of this file. Adding "
        "the columns belonging to one class gives that class's probability in "
        "P_CLASS, and all 25 add to 1 for every source."),
    "P_YSO": ("probability",
        "The probability that the source is a young stellar object. This repeats the "
        "young stellar object column of P_CLASS for convenience."),
    "MAP_CLASS": ("class position",
        "Which of the six classes has the highest probability for this source, given "
        "as a position in the CLASSES attribute of this file, counting from zero. The "
        "value is -1 where the source could not be fitted, which happens when fewer "
        "than two bands hold a measured flux or when a flux error is not a finite "
        "number."),
    "A_K_POST": ("magnitudes of K-band extinction",
        "The extinction in front of the source, in magnitudes at K band (2.2 micron). "
        "There is one value per class, each the extinction the source would have if "
        "it belonged to that class, in the column order of the CLASSES attribute of "
        "this file. The value for the class in MAP_CLASS is the pipeline's estimate "
        "for the source. Each value averages over every model and extinction law the "
        "fit considered, weighted by how well each one explains the photometry."),
    "A_K_POST_SIG": ("magnitudes of K-band extinction",
        "The uncertainty on the extinction stored in A_K_POST, in magnitudes at K "
        "band, one value per class in the column order of the CLASSES attribute of "
        "this file. It is the standard deviation of the extinction over every model "
        "and extinction law the fit considered, so it reflects both how precisely the "
        "photometry fixes the extinction and how much the models that fit this source "
        "disagree about it. It is not a formal fitting error."),
    "LOG10_FLUX_IMPUTED": ("log10 of flux in mJy",
        "The source's eight-band spectrum, stored as the base-10 logarithm of flux in "
        "mJy, in the band order J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron. A band "
        "the survey measured carries the logarithm of the measured flux. A band it "
        "did not carries an estimate: the average log flux of the models that fit "
        "this source, each weighted by how well it explains the photometry. Raise 10 "
        "to the stored value to get a flux in mJy."),
    "LOG10_FLUX_IMPUTED_COV": ("squared dex",
        "The 8 by 8 covariance of the eight values in LOG10_FLUX_IMPUTED, in squared "
        "dex, where one dex is a factor of 10. On an estimated band, the square root "
        "of the diagonal is the uncertainty in log flux, so the flux lies between "
        "10**(x - s) and 10**(x + s) about 68 percent of the time, with x the stored "
        "log flux and s that square root. On a measured band, the diagonal carries "
        "the catalog's own flux error written the same way, and the error in mJy is "
        "10**x times 2.3026 times s. An entry linking a measured band to an estimated "
        "band is not zero, because the measurement constrains the estimate, and a "
        "color formed from one measured and one estimated band needs that entry. "
        "Entries linking two measured bands are zero, since the catalog's errors are "
        "independent from band to band."),
    "LOG10_CANDIDATE_FLUX": ("log10 of flux in mJy",
        "The source's eight-band spectrum as it would be under each of the six "
        "classes in turn, stored as the base-10 logarithm of flux in mJy. The class "
        "order is the CLASSES attribute of this file and the band order is J, H, Ks, "
        "3.6, 4.5, 5.8, 8.0 and 24 micron. Measured bands carry the logarithm of the "
        "measured flux; the other bands carry that class's own estimate."),
    "ENTROPY_CLASS": ("nats",
        "How uncertain the classification of this source is, computed from the six "
        "probabilities in P_CLASS. Zero means one class holds all the probability. "
        "The largest possible value, 1.79, means all six classes are equally likely."),
    "ENTROPY_SUBCLASS": ("nats",
        "How uncertain the subdivision of this source is, computed from the 25 "
        "probabilities in P_SUBCLASS. Zero means one subdivision holds all the "
        "probability. The largest possible value, 3.22, means all 25 are equally "
        "likely."),
}


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
        set_readings(out, _READINGS)
        out.attrs["GRANULE"] = "source"
        out.attrs["CLASSES"] = np.array(CLASSES, dtype="S8")
        out.attrs["SUBCLASSES"] = np.array(SUBCLASS_LABELS, dtype="S12")
        out.attrs["FIT_FILES"] = fit_files
        # sources flagged by the fit (n_detected < 2, or a singular design
        # matrix): MAP_CLASS is -1 for these, never STAR, and P_CLASS/
        # P_SUBCLASS/P_YSO/LOG10_FLUX_IMPUTED are NaN (R3 U1, U2) -- recorded
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
        set_readings(f, {
            "REGION": ("region name", "The star-forming region this row describes."),
            "RUN": ("test name",
                     "One of nine tests, each changing how many objects of a class the sky is "
                     "expected to hold, by the amount the published measurements allow, to see "
                     "how far the classification moves."),
            "SCALING": ("factor",
                         "The factor by which a test multiplied the expected number of objects "
                         "of each class, with 1.0 where a test leaves a class alone. The class "
                         "order is the CLASSES attribute of this file. The row for the young "
                         "stellar object floor test holds that region's average factor over its "
                         "own sources rather than a single published number."),
            "FRAC_MAP_CHANGED": ("fraction",
                                   "The fraction of the region's sources whose most probable "
                                   "class changed under that test, compared with the "
                                   "classification the pipeline reports."),
            "N_PYSO_ABOVE_HALF": ("sources",
                                    "How many of the region's sources have a probability above "
                                    "one half of being a young stellar object, under each test. "
                                    "The first column is the classification the pipeline "
                                    "reports, with no test applied."),
            "N_SOURCES": ("sources",
                           "How many sources the region holds, the number FRAC_MAP_CHANGED is a "
                           "fraction of."),
        })
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
            # not LOG10_CANDIDATE_FLUX/LOG10_FLUX_IMPUTED_COV, the two region-sized
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
