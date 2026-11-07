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


#: The three Herschel/PACS bands the census carries, micron.
BANDS = (70, 100, 160)

#: The distance each survey's published fluxes are quoted at, parsec (the
#: cloud distances the prior uses for them: Orion for HOPS, Aquila for
#: eHOPS). The templates' reference fluxes are at 1 kpc, so a source's 4.5
#: micron flux at 1 kpc is `F * (d / 1000 pc)^2`.
SURVEY_DISTANCE_PC = {"HOPS": 400.0, "eHOPS": 436.0}

#: HOPS `table2`'s flag on a band's flux: with a flux, 1 or 2 (a measurement);
#: without one, 3 means observed and not detected (the 23 classed sources
#: blank at 70 micron with flag 3 are exactly the census's censored ones) and
#: 0 means not observed. At 100 micron a blank with flag 3 cannot be told
#: from a source PACS never observed at 100 micron (26 of them have
#: F70 > 0.5 Jy, brighter than any F100 detection could be missed at), so a
#: HOPS 100 micron blank is left out, never censored.
HOPS_BLANK_CENSORED = {70: True, 100: False, 160: True}


def _hops(path):
    """The HOPS rows: class from `table1` (protostars) and `table7` (young
    stars without a PACS measurement), fluxes from `table2`. Returns the
    class, 4.5 micron flux and error, per band `(flux, error, state)` with
    state 1 detected, 0 observed and not detected, -1 left out, and the
    left-out counts."""
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
    f45 = _filled(t2["F4.5"], JY_TO_MJY)
    e45 = _filled(t2["e_F4.5"], JY_TO_MJY)
    f45_flag = np.ma.filled(t2["f_F4.5"], 0)
    f45 = np.where(f45_flag == HOPS_F45_MEASURED, f45, np.nan)
    rows = [i for i, h in enumerate(ids) if h in cls and cls[h] in HOPS_CLASS and h not in not_observed]
    n_left = dict(no_class=sum(1 for h in ids if h not in cls or cls[h] not in HOPS_CLASS),
                  not_observed=sum(1 for h in ids if h in not_observed))
    sel = np.array(rows, dtype=int)
    klass = np.array([HOPS_CLASS[cls[ids[i]]] for i in sel])
    bands = {}
    for b in BANDS:
        flux = _filled(t2[f"F{b}"], JY_TO_MJY)[sel]
        err = _filled(t2[f"e_F{b}"], JY_TO_MJY)[sel]
        flag = np.ma.filled(t2[f"f_F{b}"], -1)[sel]
        if b == 70:
            blank_state = np.zeros(sel.size, dtype=np.int8)       # as always: observed, not detected
        else:
            blank_state = np.where((flag == 3) & HOPS_BLANK_CENSORED[b], 0, -1).astype(np.int8)
        state = np.where(np.isfinite(flux), 1, blank_state).astype(np.int8)
        bands[b] = (flux, err, state)
    return klass, f45[sel], e45[sel], bands, n_left


def _ehops(path):
    """The eHOPS rows with a class, a 4.5 micron flux and a flux in at least
    one band. The table publishes no non-detection: a blank is left out."""
    tables = {t.ID: t.to_table() for t in parse(path).resources[0].tables}
    t = tables["J_ApJS_266_32_ehops"]
    klass = np.array([EHOPS_CLASS.get(c, "") for c in _plain(t["Class"])])
    f45, e45 = _filled(t["F4.5"]), _filled(t["e_F4.5"])
    flux = {b: _filled(t[f"F{b}"]) for b in BANDS}
    err = {b: _filled(t[f"e_F{b}"]) for b in BANDS}
    in_class = klass != ""
    any_band = np.zeros(klass.size, dtype=bool)
    for b in BANDS:
        any_band |= np.isfinite(flux[b])
    n_left = dict(no_class=int((~in_class).sum()), not_observed=int((in_class & ~any_band).sum()))
    sel = in_class & any_band
    bands = {b: (flux[b][sel], err[b][sel], np.where(np.isfinite(flux[b][sel]), 1, -1).astype(np.int8))
             for b in BANDS}
    return klass[sel], f45[sel], e45[sel], bands, n_left


def build(config, regions=None):
    """Writes `sky/derived/hops/farir_hops_survey.hdf5`, one row per HOPS or
    eHOPS source with a class, a 4.5 micron flux and a 70, 100 or 160 micron
    measurement or non-detection. Per band `b` in 70, 100, 160: `OBSERVED_Pb`
    (the source is in that band's census: detected, or observed and not
    detected), `DETECTED_Pb`, the flux, `LOG10_R` (70) / `LOG10_R100` /
    `LOG10_R160` and its limit. Survey-wide: `regions` is accepted for the
    standard `build` signature and ignored (rule 5c)."""
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
            klass, f45, e45, bands, n_left = reader(path)
            has45 = np.isfinite(f45) & (f45 > 0)
            n_no45 = int((~has45).sum())
            klass, f45, e45 = klass[has45], f45[has45], e45[has45]
            bands = {b: tuple(x[has45] for x in v) for b, v in bands.items()}
            limit = {b: float(np.nanmin(bands[b][0][bands[b][2] == 1])) for b in BANDS}
            blocks.append((name, klass, f45, e45, bands, limit))
            print(f"farir_census: {name}: {klass.size} sources kept; left out: {n_left['no_class']} with no class, "
                  f"{n_left['not_observed']} not observed in any band, {n_no45} with no measured 4.5 um flux")
            for b in BANDS:
                state = bands[b][2]
                print(f"farir_census: {name} {b} um: {int((state == 1).sum())} detected, {int((state == 0).sum())} "
                      f"censored below {limit[b]:.1f} mJy, {int((state == -1).sum())} blank and left out "
                      f"(of {klass.size})")
        cat = lambda get: np.concatenate([get(b) for b in blocks])
        survey = np.concatenate([np.full(b[1].size, i, dtype=np.int8) for i, b in enumerate(blocks)])
        klass = cat(lambda b: b[1]).astype("S4")
        f45 = cat(lambda b: b[2])
        distance = cat(lambda b: np.full(b[1].size, SURVEY_DISTANCE_PC[b[0]]))
        ln10 = np.log(10.0)
        columns = [("SURVEY", survey), ("CLASS", klass), ("FLUX_I2_MJY", f45),
                   ("DISTANCE_PC", distance),
                   ("LOG10_F_I2_1KPC", np.log10(f45) + 2.0 * np.log10(distance / 1000.0))]
        for b in BANDS:
            flux = cat(lambda x: x[4][b][0])
            err = cat(lambda x: x[4][b][1])
            state = cat(lambda x: x[4][b][2])
            limit = cat(lambda x: np.full(x[1].size, x[5][b]))
            detected = state == 1
            tag = "" if b == 70 else str(b)
            sigma_flux = np.where(detected, err, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                log10_r = np.where(detected, np.log10(flux / f45), np.nan)
                sigma = np.where(detected, np.sqrt((err / flux) ** 2 + np.nan_to_num(cat(lambda x: x[3]) / f45) ** 2) / ln10, np.nan)
                log10_r_limit = np.log10(limit / f45)
            columns += [(f"OBSERVED_P{b}", (state >= 0).astype(np.int8)),
                        (f"DETECTED_P{b}", detected.astype(np.int8)),
                        (f"FLUX_P{b}_MJY", np.where(detected, flux, np.nan)),
                        (f"SIGMA_FLUX_P{b}_MJY", sigma_flux),
                        (f"LIMIT_P{b}_MJY", limit),
                        (f"LOG10_R{tag}", log10_r), (f"SIGMA_LOG10_R{tag}", sigma),
                        (f"LOG10_R{tag}_LIMIT", log10_r_limit)]

        out_path = config_module.product_path(config, "sky/derived", "hops", "farir", "survey")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            f.attrs["SURVEYS"] = "HOPS;eHOPS"
            for name, arr in columns:
                build_module.write_dataset(f, name, arr, *REGISTRY[(_STEM, name)])
        for k in (b"C0", b"CI", b"FLAT", b"CII"):
            m = klass == k
            print(f"farir_census: class {k.decode()}: {int(m.sum())} sources, " + ", ".join(
                f"{int(dict(columns)[f'DETECTED_P{b}'][m].sum())} detected at {b} um" for b in BANDS))
        st.done(out_path, n_source=int(klass.size), n_detected=int(dict(columns)["DETECTED_P70"].sum()))
    return out_path


if __name__ == "__main__":
    build_module.run(build)
