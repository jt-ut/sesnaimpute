"""The far-infrared census of the young stars: the 70 micron to 4.5 micron
flux ratio of the HOPS and eHOPS protostars, per evolutionary class
(SPEC_BMSTP_DRAFT.md sec 1.4, 5.5).

A view of the two published Herschel/Spitzer surveys alone (Furlan et al.
2016, ApJS 224, 5, the Orion HOPS tables; Pokhrel et al. 2023, ApJS 266, 32,
the Aquila eHOPS table), read from `sky/download/hops`: each source's own
Spitzer-classified evolutionary class (Class 0, Class I, flat-spectrum,
Class II), its published PACS 70 micron flux and IRAC 4.5 micron flux, and
`LOG10_R = log10(F70 / F4.5)`. A source the survey observed at 70 micron and
did not detect is kept as a CENSORED value: its ratio is known only to lie
below `LOG10_R_LIMIT`, computed from its own 4.5 micron flux and the survey's
faintest published 70 micron detection. A source the survey did not observe
at 70 micron (HOPS) or lists with no 70 micron entry (eHOPS, whose table does
not say whether the source was undetected or off the map), a source with no
measured 4.5 micron flux and a source with no class are left out; their
counts are printed.

This is HOPS and eHOPS data. The class factor of the YSO template weights
reads it, so the comparison of SESNA's classification with HOPS and eHOPS is
no longer independent of the prior on this axis.
"""

import os

import h5py
import numpy as np
from astropy.io.votable import parse

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute.attrs_registry import REGISTRY

_STEM = "farir_hops_survey"

#: The two published VizieR tables under `sky/download/hops` and their
#: survey names, in the order the rows are written.
HOPS_FILE = "J_ApJS_224_5.vot"
EHOPS_FILE = "J_ApJS_266_32.vot"

#: The class of a HOPS source (`table1`, `table7` column `Class`) and of an
#: eHOPS source (`ehops` column `Class`) as the YSO register's own subclass
#: labels (`definitions.SUBCLASSES_OF["YSO"]`).
HOPS_CLASS = {"0": "C0", "I": "CI", "flat": "FLAT", "II": "CII"}
EHOPS_CLASS = {"Class 0": "C0", "Class 1": "CI", "Flat": "FLAT", "Class 2": "CII"}

#: HOPS publishes fluxes in Jy, eHOPS in mJy.
JY_TO_MJY = 1.0e3

#: HOPS `table2`'s `f_F4.5` value for a measured (not limit, not absent) flux.
HOPS_F45_MEASURED = 1


def _plain(column):
    return np.array([str(x).strip() for x in column])


def _filled(column, scale=1.0):
    return np.ma.filled(np.ma.asarray(column, dtype=float), np.nan) * scale


def _hops(path):
    """The HOPS rows: class from `table1` (protostars) and `table7` (young
    stars without a PACS measurement), fluxes from `table2`. Returns the
    per-source arrays and the left-out counts."""
    tables = {t.ID: t.to_table() for t in parse(path).resources[0].tables}
    t1, t2, t7 = (tables["J_ApJS_224_5_table1"], tables["J_ApJS_224_5_table2"],
                  tables["J_ApJS_224_5_table7"])
    cls = dict(zip(t1["HOPS"].tolist(), _plain(t1["Class"])))
    not_observed = set()
    for hops, c, fl in zip(t7["HOPS"].tolist(), _plain(t7["Class"]), np.ma.filled(t7["Fl"], -1)):
        cls[hops] = c
        if fl == -1:
            not_observed.add(hops)
    ids = t2["HOPS"].tolist()
    f70 = _filled(t2["F70"], JY_TO_MJY)
    e70 = _filled(t2["e_F70"], JY_TO_MJY)
    f45 = _filled(t2["F4.5"], JY_TO_MJY)
    e45 = _filled(t2["e_F4.5"], JY_TO_MJY)
    f45_flag = np.ma.filled(t2["f_F4.5"], 0)
    f45 = np.where(f45_flag == HOPS_F45_MEASURED, f45, np.nan)
    rows = [i for i, h in enumerate(ids) if h in cls and cls[h] in HOPS_CLASS and h not in not_observed]
    n_left = dict(no_class=sum(1 for h in ids if h not in cls or cls[h] not in HOPS_CLASS),
                  not_observed=sum(1 for h in ids if h in not_observed))
    sel = np.array(rows, dtype=int)
    klass = np.array([HOPS_CLASS[cls[ids[i]]] for i in sel])
    return klass, f70[sel], e70[sel], f45[sel], e45[sel], n_left


def _ehops(path):
    tables = {t.ID: t.to_table() for t in parse(path).resources[0].tables}
    t = tables["J_ApJS_266_32_ehops"]
    klass = np.array([EHOPS_CLASS.get(c, "") for c in _plain(t["Class"])])
    f70, e70 = _filled(t["F70"]), _filled(t["e_F70"])
    f45, e45 = _filled(t["F4.5"]), _filled(t["e_F4.5"])
    in_class = klass != ""
    has_70 = np.isfinite(f70)
    n_left = dict(no_class=int((~in_class).sum()), not_observed=int((in_class & ~has_70).sum()))
    sel = in_class & has_70
    return klass[sel], f70[sel], e70[sel], f45[sel], e45[sel], n_left


def build(config, regions=None):
    """Writes `sky/derived/hops/farir_hops_survey.hdf5`, one row per HOPS or
    eHOPS source with a class, a 70 micron measurement or non-detection and a
    4.5 micron flux. Survey-wide: `regions` is accepted for the standard
    `build` signature and ignored (rule 5c)."""
    if regions is not None:
        print("farir_census: survey-wide product, --regions ignored")
    src = f"{config.data_root}/sky/download/hops"
    with progress_module.Stage("sky.derived.farir_census") as st:
        blocks = []
        for name, reader, fname in (("HOPS", _hops, HOPS_FILE), ("eHOPS", _ehops, EHOPS_FILE)):
            path = f"{src}/{fname}"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"farir_census: no HOPS/eHOPS table at {path!r} -- run the "
                    f"'sesnaimpute.sky.download.hops.build' RUNBOOK line")
            klass, f70, e70, f45, e45, n_left = reader(path)
            has45 = np.isfinite(f45) & (f45 > 0)
            n_no45 = int((~has45).sum())
            keep = has45
            klass, f70, e70, f45, e45 = klass[keep], f70[keep], e70[keep], f45[keep], e45[keep]
            detected = np.isfinite(f70)
            limit = float(np.nanmin(f70))
            blocks.append((name, klass, f70, e70, f45, e45, detected, limit))
            print(f"farir_census: {name}: {klass.size} sources kept ({int(detected.sum())} detected at 70 um, "
                  f"{int((~detected).sum())} censored below {limit:.1f} mJy); left out: "
                  f"{n_left['no_class']} with no class, {n_left['not_observed']} not observed at 70 um, "
                  f"{n_no45} with no measured 4.5 um flux")
        survey = np.concatenate([np.full(b[1].size, i, dtype=np.int8) for i, b in enumerate(blocks)])
        klass = np.concatenate([b[1] for b in blocks]).astype("S4")
        f70 = np.concatenate([b[2] for b in blocks])
        e70 = np.concatenate([b[3] for b in blocks])
        f45 = np.concatenate([b[4] for b in blocks])
        e45 = np.concatenate([b[5] for b in blocks])
        detected = np.concatenate([b[6] for b in blocks]).astype(np.int8)
        limit = np.concatenate([np.full(b[1].size, b[7]) for b in blocks])

        ln10 = np.log(10.0)
        log10_r = np.log10(f70 / f45)
        sigma = np.sqrt((e70 / f70) ** 2 + np.nan_to_num(e45 / f45) ** 2) / ln10
        log10_r_limit = np.log10(limit / f45)

        out_path = config_module.product_path(config, "sky/derived", "hops", "farir", "survey")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            f.attrs["SURVEYS"] = "HOPS;eHOPS"
            for name, arr in (("SURVEY", survey), ("CLASS", klass), ("FLUX_P70_MJY", f70),
                              ("SIGMA_FLUX_P70_MJY", e70), ("FLUX_I2_MJY", f45),
                              ("DETECTED_P70", detected), ("LIMIT_P70_MJY", limit),
                              ("LOG10_R", log10_r), ("SIGMA_LOG10_R", sigma),
                              ("LOG10_R_LIMIT", log10_r_limit)):
                build_module.write_dataset(f, name, arr, *REGISTRY[(_STEM, name)])
        for k in (b"C0", b"CI", b"FLAT", b"CII"):
            m = klass == k
            print(f"farir_census: class {k.decode()}: {int(m.sum())} sources, {int(detected[m].sum())} detected")
        st.done(out_path, n_source=int(klass.size), n_detected=int(detected.sum()))
    return out_path


if __name__ == "__main__":
    build_module.run(build)
