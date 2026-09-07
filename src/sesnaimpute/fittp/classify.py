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


def build_region(config, region, st, beta):
    """One region's P8 arrays, swept in source batches (rule 10b)."""
    _require_fit_files(config, region)

    class_files = {}
    names = None
    n_sub_by_class = {}
    for cls in CLASSES:
        f = h5py.File(_fit_path(config, region, cls), "r")
        class_files[cls] = f
        this_names = f["NAME"][:]
        if names is None:
            names = this_names
        elif not np.array_equal(names, this_names):
            raise ValueError("fittp.classify [%s]: %s's NAME does not row-align with STAR's"
                              % (region, cls))
        n_sub_by_class[cls] = f["LN_EVIDENCE"].shape[1]

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
    p_class = np.empty((n, len(CLASSES)), dtype=np.float32)
    p_subclass = np.empty((n, N_SUBCLASS), dtype=np.float32)
    map_class = np.empty(n, dtype=np.int8)
    n_detected = np.empty(n, dtype=np.int8)
    candidate_flux = np.empty((n, len(CLASSES), N_BANDS), dtype=np.float32)
    flux_imputed = np.empty((n, N_BANDS), dtype=np.float32)
    flux_imputed_cov = np.empty((n, N_BANDS, N_BANDS), dtype=np.float32)
    entropy_class = np.empty(n, dtype=np.float32)
    entropy_subclass = np.empty(n, dtype=np.float32)

    imputed_identity_err = 0.0

    bounds = list(batches(n, ROW_BYTES))
    for bi, (start, stop) in enumerate(bounds):
        m = stop - start
        ln_ev = np.full((m, N_SUBCLASS), -np.inf, dtype=np.float64)
        flux_mean_stack = np.empty((len(CLASSES), m, N_BANDS), dtype=np.float64)
        flux_cov_stack = np.empty((len(CLASSES), m, N_BANDS, N_BANDS), dtype=np.float64)
        for ci, cls in enumerate(CLASSES):
            f = class_files[cls]
            lo, hi = CLASS_SLICES[cls]
            block = np.asarray(f["LN_EVIDENCE"][start:stop, :], dtype=np.float64)
            if beta != 0.0:
                ln_psi = np.log(np.asarray(psi_file["PSI_CLASS"][start:stop, ci], dtype=np.float64))
                block = block + beta * ln_psi[:, None]
            ln_ev[:, lo:hi] = block
            flux_mean_stack[ci] = np.asarray(f["FLUX_MEAN"][start:stop, :], dtype=np.float64)
            flux_cov_stack[ci] = np.asarray(f["FLUX_COV"][start:stop, :, :], dtype=np.float64)

        row_max = ln_ev.max(axis=1, keepdims=True)
        weights = np.exp(ln_ev - row_max)
        p_sub = weights / weights.sum(axis=1, keepdims=True)

        p_cls = np.zeros((m, len(CLASSES)), dtype=np.float64)
        for ci, cls in enumerate(CLASSES):
            lo, hi = CLASS_SLICES[cls]
            p_cls[:, ci] = p_sub[:, lo:hi].sum(axis=1)

        map_c = np.argmax(p_cls, axis=1)

        with h5py.File(cat_path, "r") as cf:
            flux = np.asarray(cf["FNU_MJY"][start:stop], dtype=np.float64)
            origin = np.asarray(cf["ORIGIN_FNU"][start:stop])
        detected = origin == 1

        cflux = np.transpose(flux_mean_stack, (1, 0, 2)).copy()  # (m, 6, 8)
        mask = np.broadcast_to(detected[:, None, :], cflux.shape)
        cflux = np.where(mask, flux[:, None, :], cflux)
        row_idx = np.arange(m)
        imputed = cflux[row_idx, map_c, :]
        imputed_cov = flux_cov_stack[map_c, row_idx]
        if detected.any():
            imputed_identity_err = max(imputed_identity_err,
                                        float(np.max(np.abs(imputed[detected] - flux[detected]))))

        with np.errstate(divide="ignore"):
            ent_c = -np.sum(np.where(p_cls > 0, p_cls * np.log(p_cls), 0.0), axis=1)
            ent_s = -np.sum(np.where(p_sub > 0, p_sub * np.log(p_sub), 0.0), axis=1)

        p_class[start:stop] = p_cls.astype(np.float32)
        p_subclass[start:stop] = p_sub.astype(np.float32)
        map_class[start:stop] = map_c.astype(np.int8)
        n_detected[start:stop] = detected.sum(axis=1).astype(np.int8)
        candidate_flux[start:stop] = cflux.astype(np.float32)
        flux_imputed[start:stop] = imputed.astype(np.float32)
        flux_imputed_cov[start:stop] = imputed_cov.astype(np.float32)
        entropy_class[start:stop] = ent_c.astype(np.float32)
        entropy_subclass[start:stop] = ent_s.astype(np.float32)
        st.tick(bi + 1, len(bounds), "batches")

    for f in class_files.values():
        f.close()
    if psi_file is not None:
        psi_file.close()

    fit_files = np.array([_fit_path(config, region, cls) for cls in CLASSES], dtype="S256")
    return dict(name=names, p_class=p_class, p_subclass=p_subclass, map_class=map_class,
                n_detected=n_detected, candidate_flux=candidate_flux, flux_imputed=flux_imputed,
                flux_imputed_cov=flux_imputed_cov, entropy_class=entropy_class,
                entropy_subclass=entropy_subclass, fit_files=fit_files,
                imputed_identity_err=imputed_identity_err)


def write_region(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("NAME", data=result["name"])
        f.create_dataset("P_CLASS", data=result["p_class"])
        f.create_dataset("P_SUBCLASS", data=result["p_subclass"])
        f.create_dataset("P_YSO", data=result["p_class"][:, YSO_INDEX])
        f.create_dataset("MAP_CLASS", data=result["map_class"])
        f.create_dataset("N_DETECTED", data=result["n_detected"])
        f.create_dataset("CANDIDATE_FLUX", data=result["candidate_flux"])
        f.create_dataset("FLUX_IMPUTED", data=result["flux_imputed"])
        f.create_dataset("FLUX_IMPUTED_COV", data=result["flux_imputed_cov"])
        f.create_dataset("ENTROPY_CLASS", data=result["entropy_class"])
        f.create_dataset("ENTROPY_SUBCLASS", data=result["entropy_subclass"])
        f.attrs["GRANULE"] = "source"
        f.attrs["CLASSES"] = np.array(CLASSES, dtype="S8")
        f.attrs["SUBCLASSES"] = np.array(SUBCLASS_LABELS, dtype="S12")
        f.attrs["FIT_FILES"] = result["fit_files"]


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
            path = config_module.product_path(config, "fittp", "classification", "posterior",
                                               "source", region=region)
            write_region(path, result)

            p_class_err = float(np.max(np.abs(result["p_class"].sum(axis=1) - 1.0)))
            sub_sum = np.zeros_like(result["p_class"])
            for ci, cls in enumerate(CLASSES):
                lo, hi = CLASS_SLICES[cls]
                sub_sum[:, ci] = result["p_subclass"][:, lo:hi].sum(axis=1)
            subclass_err = float(np.max(np.abs(sub_sum - result["p_class"])))
            n_pyso_half = int((result["p_class"][:, YSO_INDEX] > 0.5).sum())
            st.done(path, n=result["name"].shape[0], beta=beta,
                    p_class_sum_err=p_class_err, p_subclass_sum_err=subclass_err,
                    flux_imputed_identity_err=result["imputed_identity_err"],
                    n_pyso_above_half=n_pyso_half)


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
