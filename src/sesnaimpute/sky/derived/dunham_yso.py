"""The Dunham et al. 2015 (ApJS 220, 11) YSO census at the standard
granule: one row per catalogued YSO, `Seq` order, no selection, no class
assignment, no density (W54 forms the density from this view).

Joins `table1.dat` (18 clouds, each cloud's `Dist1`), `table2.dat` (2,966
YSOs: bolometric properties) and `table3.dat`/`table4.dat` (observed and
extinction-corrected 2MASS/Spitzer flux densities, mJy) from the fetched
`sky/download/dunham2015/` bytes (`-999`/`-999.` sentinels become `nan`,
the ReadMe's own missing-value convention). `RA_DEG`/`DEC_DEG` are parsed
from `table2.dat`'s own `ID` column, the ReadMe's `JHHMMSS.s+DDMMSS`
J2000 sexagesimal source name -- table2.dat carries no separate
coordinate columns.

`LOG10_F45_REF = log10(F45_MJY * (DIST_PC / 1000 pc)^2)` is the
extinction-corrected 4.5 micron flux scaled to 1 kpc, in the SAME units
and at the SAME reference distance as the YSO template register's own
`F_REF_I2` (`bmstp/sample_cloud.py`'s `sample_f45`, `bmstp/sample_star.py`
line 80: "mJy at the register's 1 kpc reference") -- no unit conversion,
both are already mJy at 1 kpc, so this column reads directly onto the
register's own reference-brightness axis (`bmstp/grid.py`'s
`LOG10_F45_EDGES`).
"""

import os
import re

import h5py
import numpy as np
import pandas as pd

from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute.build import run

# byte-by-byte column layouts (0-indexed half-open slices), CDS ReadMe
# for J/ApJS/220/11: table2.dat at its line 158, table3/4.dat at 180.
_TABLE2_COLSPECS = [
    (0, 4), (5, 21), (22, 38), (39, 43), (44, 51), (52, 58),
    (59, 69), (70, 77), (78, 84), (85, 95),
]
_TABLE2_NAMES = ["Seq", "Cloud", "ID", "Av", "alpha", "Tbol", "Lbol",
                  "alpha0", "Tbol0", "Lbol0"]

_TABLE34_COLSPECS = [
    (0, 4), (5, 14), (15, 26), (27, 36), (37, 46), (47, 56), (57, 66),
    (67, 76), (77, 87), (88, 96), (97, 107), (108, 117), (118, 126),
    (127, 135), (136, 144), (145, 153), (154, 162), (163, 169), (170, 177),
]
_TABLE34_NAMES = ["Seq", "F1.25", "e_F1.25", "F1.65", "e_F1.65", "F2.17",
                    "e_F2.17", "F3.6", "e_F3.6", "F4.5", "e_F4.5", "F5.8",
                    "e_F5.8", "F8.0", "e_F8.0", "F24", "e_F24", "F70", "e_F70"]

_MISSING = -999.0


def _download_path(config, name):
    path = f"{config.data_root}/sky/download/dunham2015/{name}"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "sky.derived.dunham_yso: no %r at %r -- run the "
            "'sesnaimpute.sky.download.dunham2015.build' RUNBOOK line first" % (name, path))
    return path


def _sentinel_to_nan(df, skip=("Seq", "Cloud", "ID")):
    """The ReadMe's own missing-value convention (`?=-999`/`-999.`)
    applied to every numeric column."""
    for col in df.columns:
        if col not in skip:
            df.loc[df[col] == _MISSING, col] = np.nan
    return df


def _read_table1(config):
    """`{cloud_name: Dist1_pc}` for the 18 clouds, pipe-delimited
    (`Cloud|Survey|Dist[-Dist2][flag]|r_Dist|Area|Mass|e_Mass|Ref`); the
    third field is `Dist1` alone except for Cepheus (`200-325c`, note 2),
    whose leading integer is `Dist1`."""
    path = _download_path(config, "table1.dat")
    dist1 = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("|")
            cloud = fields[0].strip()
            dist1[cloud] = int(re.match(r"\s*(\d+)", fields[2]).group(1))
    return dist1


def _read_fwf(path, colspecs, names):
    return pd.read_fwf(path, colspecs=colspecs, names=names, header=None)


#: The ReadMe's own `ID` label format, "SSTc2d or SSTgb Spitzer source
#: name (JHHMMSS.s+DDMMSS)": table2.dat carries no separate RA/Dec
#: columns, so the position is the sexagesimal name itself, parsed here.
_ID_RE = re.compile(
    r"^J(\d{2})(\d{2})(\d{2}\.\d)([+-])(\d{2})(\d{2})(\d{2})$")


def _radec_from_id(id_series):
    """`(ra_deg, dec_deg)` parsed from the `ID` column's own J2000
    sexagesimal name, `JHHMMSS.s+DDMMSS` (hours/minutes/seconds of RA,
    signed degrees/arcmin/arcsec of Dec) -- the census's only position,
    since table2.dat has no RA/Dec fields of its own."""
    m = id_series.str.strip().str.extract(_ID_RE)
    if m.isna().any(axis=None):
        bad = id_series[m.isna().any(axis=1)].tolist()
        raise ValueError(
            "sky.derived.dunham_yso: ID entries do not match the ReadMe's "
            "JHHMMSS.s+DDMMSS name format: %r" % bad[:5])
    hh = m[0].astype(np.float64)
    mm = m[1].astype(np.float64)
    ss = m[2].astype(np.float64)
    sign = np.where(m[3].to_numpy() == "-", -1.0, 1.0)
    dd = m[4].astype(np.float64)
    dm = m[5].astype(np.float64)
    ds = m[6].astype(np.float64)
    ra_deg = 15.0 * (hh + mm / 60.0 + ss / 3600.0)
    dec_deg = sign * (dd + dm / 60.0 + ds / 3600.0)
    return ra_deg.to_numpy(), dec_deg.to_numpy()


def build(config, regions=None):
    """Writes `sky/derived/dunham2015/yso_dunham2015_survey.hdf5`
    (`GRANULE = "survey"`), one row per `Seq`. `regions` is accepted for
    interface uniformity and ignored: this is a survey-wide product built
    once from the fetched CDS tables.
    """
    with progress_module.Stage("sky.derived.dunham_yso") as st:
        dist1_by_cloud = _read_table1(config)

        t2 = _sentinel_to_nan(_read_fwf(_download_path(config, "table2.dat"),
                                          _TABLE2_COLSPECS, _TABLE2_NAMES))
        t3 = _sentinel_to_nan(_read_fwf(_download_path(config, "table3.dat"),
                                          _TABLE34_COLSPECS, _TABLE34_NAMES))
        t4 = _sentinel_to_nan(_read_fwf(_download_path(config, "table4.dat"),
                                          _TABLE34_COLSPECS, _TABLE34_NAMES))
        if not (t2["Seq"].to_numpy() == t3["Seq"].to_numpy()).all() or \
           not (t2["Seq"].to_numpy() == t4["Seq"].to_numpy()).all():
            raise ValueError("sky.derived.dunham_yso: table2/3/4 Seq order disagrees")

        cloud = t2["Cloud"].str.strip()
        dist_pc = cloud.map(dist1_by_cloud).to_numpy(dtype=np.float64)
        if np.any(pd.isna(dist_pc)):
            raise ValueError("sky.derived.dunham_yso: a table2 Cloud has no table1 match")

        f45_mjy = t4["F4.5"].to_numpy(dtype=np.float64)
        log10_f45_ref = np.log10(f45_mjy * (dist_pc / 1000.0) ** 2)

        ra_deg, dec_deg = _radec_from_id(t2["ID"])

        out_path = config_module.product_path(config, "sky/derived", "dunham2015", "yso", "survey")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            f.create_dataset("SEQ", data=t2["Seq"].to_numpy(dtype=np.int64))
            f.create_dataset("CLOUD", data=cloud.to_numpy(dtype="S16"))
            f.create_dataset("ID", data=t2["ID"].str.strip().to_numpy(dtype="S16"))
            f.create_dataset("RA_DEG", data=ra_deg.astype(np.float64))
            f.create_dataset("DEC_DEG", data=dec_deg.astype(np.float64))
            f.create_dataset("DIST_PC", data=dist_pc.astype(np.float32))
            f.create_dataset("AV", data=t2["Av"].to_numpy(dtype=np.float32))
            f.create_dataset("ALPHA0", data=t2["alpha0"].to_numpy(dtype=np.float32))
            f.create_dataset("TBOL0_K", data=t2["Tbol0"].to_numpy(dtype=np.float32))
            f.create_dataset("LBOL0_LSUN", data=t2["Lbol0"].to_numpy(dtype=np.float32))
            f.create_dataset("F45_MJY", data=f45_mjy.astype(np.float32))
            f.create_dataset("E_F45_MJY", data=t4["e_F4.5"].to_numpy(dtype=np.float32))
            f.create_dataset("F45_OBS_MJY", data=t3["F4.5"].to_numpy(dtype=np.float32))
            f.create_dataset("F36_MJY", data=t4["F3.6"].to_numpy(dtype=np.float32))
            f.create_dataset("F58_MJY", data=t4["F5.8"].to_numpy(dtype=np.float32))
            f.create_dataset("F80_MJY", data=t4["F8.0"].to_numpy(dtype=np.float32))
            f.create_dataset("F24_MJY", data=t4["F24"].to_numpy(dtype=np.float32))
            f.create_dataset("LOG10_F45_REF", data=log10_f45_ref.astype(np.float32))
        st.tick(1, 1, "clouds joined")
        st.done(out_path, rows=len(t2), finite_f45=int(np.isfinite(f45_mjy).sum()))


if __name__ == "__main__":
    run(build)
