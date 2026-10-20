"""Every published excess-selected (and one protostar-only) young-star
census SESNA's own young-star law is fit against (owner's 2026-10-20
ruling, `briefs/round4_2026-10-20/C10_yso_census_table.md`): one
survey-wide product, `sky/derived/yso_census/census_yso_census_survey.hdf5`
(`config_module.product_path(config, "sky/derived", "yso_census",
"census", "survey")`), one row per catalogued object, replacing
`sky.derived.dunham_yso` (deleted; Dunham et al. 2015 is now one source
among several, ingested the same way).

Eight censuses are named by the ruling; Forbrich et al. 2009 (ApJ 704,
292, Pipe, 18 candidates) is NOT ingested -- it has no VizieR/CDS entry
(confirmed live against `vizier.cds.unistra.fr`'s own bibcode search,
2009ApJ...704..292F: "Empty result", and `cdsarc.cds.unistra.fr/ftp/J/
ApJ/704/292/` 404s) and this module writes nothing from a hand
transcription, the ruling's own instruction. Gutermuth 2009 and
membership lists (Luhman's Upper Sco) are excluded by the ruling itself.

Per-census notes (every VizieR table id verified live before coding
against it, DATA.md carries the same ids):

  Dunham2015 (J/ApJS/220/11): unchanged from the retired `dunham_yso`
      module -- `SELECTION = "excess"`, `ALPHA` = the catalogue's own
      `alpha0`, `F45_MJY` the extinction-corrected (table4) 4.5 micron
      flux, `SURVEY_FLOOR_A_K` the c2d/Gould-Belt coverage rule
      (table1.dat's own `Survey` field: c2d 0.22 mag, GB 0.33 mag).

  Megeath2012 (J/AJ/144/192, `table4.dat`, 3,479 rows): the catalogue
      carries no per-row cloud column (a single combined Orion A + B
      catalog); `CLOUD` is assigned from declination alone (Dec < -3 deg
      = "Orion A", else "Orion B" -- the paper's own field list places
      the Orion A fields, ONC/L1641/kappa Ori, at Dec -5 to -9 deg and
      the Orion B fields, NGC 2024/2068/71/L1622, at Dec -2 to +2 deg, so
      this one cut cleanly separates them; DISCLOSED approximation, no
      polygon-exact field boundary is available from the catalogue
      alone). `ALPHA` is the table's own IRAC spectral index; `Cl` in
      {D, P, FP, RP} are all dusty, excess-selected YSOs by the paper's
      own definition, so `SELECTION = "excess"` for every row (`Cl`
      itself is not carried: this product's schema has no discrete-class
      column, the ruling's own choice). `SURVEY_FLOOR_A_K = NaN` (no
      coverage/extinction completeness rule stated by the paper in a
      form comparable to Dunham's c2d/GB rule): the footprint here is
      the coverage polygon alone, the ruling's own stated fallback.

  Rebull2011 + Guieu2009 (`J/ApJS/193/25` with `J/ApJ/697/787`, North
      America Nebula): Guieu 2009 is the SAME survey's own earlier,
      IRAC-only pass (Rebull 2011's `table4.dat` `cr5` column cross-
      references it directly) -- its 1,657 objects are NOT ingested as
      additional census rows (that would double-count the same sky
      against Rebull 2011's own, later, more complete search); its bytes
      are fetched for that cross-reference provenance only. `CLOUD =
      "NAN"` for every row (one complex, the paper's own clusters --
      Gulf/Pelican/PelHat -- are sub-structure, not separate SESNA
      regions). Ingested: `table3.dat` (1,286 rows, position + IRAC/MIPS
      magnitudes) joined to `table4.dat`'s own YSO-quality/criteria rows
      by source name for completeness (no numeric slope: `table4.dat`'s
      `cl` is a discrete class string, I/flat/II/III, not carried -- see
      Megeath2012 note above -- so `ALPHA = NaN` here too) plus
      `table8.dat` (796 rows, "IRAC-selected YSOs not recovered by our
      MIPS-based search", same position columns, no class information at
      all). Total 2,082 rows -- the paper's abstract rounds this to
      "2,076" (1,286 MIPS + "806 more" from IRAC alone); the VizieR
      table's own `table8.dat` carries 796, not 806, a small published
      table-vs-text discrepancy disclosed here, not corrected by hand.
      `table9.dat` (114 "previously suggested YSOs not recovered by our
      MIPS or IRAC searches") is NOT ingested: these are not objects this
      survey's own method selected. `SURVEY_FLOOR_A_K = NaN`.

  Allen2012 (J/ApJ/750/125): `table1.dat` (56, class I), `table2.dat`
      (1,014, class II) and `table3.dat` (65, transition disk) are
      ingested -- the disk-bearing members, the ruling's own selection;
      `table4.dat` (1,440, class III) carries no infrared excess and is
      NOT ingested. `CLOUD = "Cepheus OB3b"`, `SELECTION = "excess"`,
      `ALPHA = NaN` (no per-row numeric slope in any of the three
      tables: they share one byte layout with no spectral-index column).
      `SURVEY_FLOOR_A_K = NaN`.

  Kirk2009 (J/ApJS/185/198): `table6.dat` (143 rows: position via the
      Spitzer catalog name, IRAC/MIPS fluxes) joined to `table7.dat`
      (143 rows, the SAME running `Seq`: `Cloud`, the numeric IR spectral
      index `alpha` and its class `aCl`) by `Seq` -- the paper's abstract
      rounds this to "133" candidates; the VizieR tables carry 143 rows
      each, disclosed, not corrected by hand. `CLOUD` is `table7.dat`'s
      own per-row dark-cloud name (varies row to row, e.g. "L1152"/
      "L1251"/"NGC7023" -- NOT one constant, unlike every other census
      here). `ALPHA` is `table7.dat`'s own numeric `alpha`.
      `SELECTION = "excess"`. `SURVEY_FLOOR_A_K = NaN`.

  Strafella2010 (J/ApJ/719/9, `table9.dat`, 637 rows, "VMR-D IRAC YSO
      catalog" -- the one of the paper's five tables with positions,
      fluxes AND the spectral slope `alpha`, confirmed from the ReadMe's
      own File Summary before coding against it): `CLOUD = "Vela D"`,
      `SELECTION = "excess"`, `ALPHA` the table's own `alpha` (2-20
      micron slope). `SURVEY_FLOOR_A_K = NaN`.

  Rebull2010 (J/ApJS/186/259): `table6.dat` (215, previously-identified
      Taurus members) and `table7.dat` (148, new candidate members) each
      carry the source's own position in its Spitzer name (`SST`,
      `HHMMSS.s+DDMMSS`, parsed directly, the SAME format and parser
      Dunham's `ID` column uses) and a discrete `Class` (I/II/III/flat)
      -- rows with `Class in {I, II, flat}` are ingested (members WITH
      excess, the ruling's own selection); `Class == "III"` (no excess)
      is not. `ALPHA = NaN` (no numeric slope column, only the discrete
      class). `CLOUD = "Taurus"`. `SELECTION = "excess"`.
      `SURVEY_FLOOR_A_K = NaN`.

  Kryukova2014 (J/AJ/148/11, `table3.dat`, 2,007 rows): PROTOSTARS ONLY
      (the paper's own subject, "protostar candidates in Cygnus-X" --
      no disk-bearing members are catalogued here at all).
      `SELECTION = "protostar"`, `CLOUD = "Cygnus X"`, `ALPHA` the
      table's own numeric spectral index. `SURVEY_FLOOR_A_K = NaN`. No
      Cygnus X region run reads this product in this round (the
      ruling's own no-Cygnus-X-runs instruction) -- it is ingested and
      written so the download+table exist for when a later round does.

`DIST_PC` is this project's own adopted distance for the census's cloud
(`sesnaimpute.regions.REGIONS_BY_NAME`, SAME value the region's own
geometry uses) wherever the census paper itself gives no per-source
distance (every census here except Dunham, which carries its own
`table1.dat` `Dist1` per cloud) -- `F45_MJY`/`LOG10_F45_REF` downstream
of a magnitude use the IRAC channel-2 (4.5 micron) zero-magnitude flux
density, 179.7 Jy (Reach et al. 2005, the Spitzer/IRAC Data Handbook's
own absolute calibration), `F_MJY = 179700 * 10**(-mag/2.5)`.
`LOG10_F45_REF`/`F45_MJY`/`DIST_PC` are secondary fields no region fit
reads (`population.yso_law`'s footprint and count use only `RA_DEG`,
`DEC_DEG`, `CLOUD`, `SOURCE` and the coverage polygons) -- they are best-
effort, disclosed estimates, not measurements, wherever the paper itself
gives neither a flux nor a distance.

The `coverage` group carries one polygon per `(SOURCE, CLOUD)`: the
convex hull of that group's own `(RA_DEG, DEC_DEG)` positions, computed
in a local gnomonic (tangent-plane) projection centred on the group's
own mean position (so the hull is convex on the SKY, not merely in raw
RA/Dec, which would distort a cloud away from the equator) and the
vertices projected back to RA/Dec, in hull (counter-clockwise in the
tangent plane) order. A convex hull is an approximation of a survey's
true (possibly non-convex) footprint, using only the catalogue's own
positions -- the ruling's own instruction for Megeath 2012 ("build the
polygon from the catalogue's convex hull per cloud and say so"),
generalised here to every census including Dunham 2015's 18 clouds, so
every `(SOURCE, CLOUD)` in the product gets the SAME kind of coverage
record.
"""

import os
import re

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.path import Path as _MplPath
from scipy.spatial import ConvexHull

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run

PATH_ARGS = ("sky/derived", "yso_census", "census", "survey")
_STEM = "census_yso_census_survey"
_READINGS = {name: value for (stem, name), value in REGISTRY.items() if stem == _STEM}

#: IRAC channel-2 (4.5 micron) zero-magnitude flux density (Reach et al.
#: 2005, the Spitzer/IRAC Data Handbook's own absolute calibration),
#: used wherever a census gives a magnitude but no flux.
F0_IRAC2_MJY = 179700.0

_MISSING_DUNHAM = -999.0


def _mag_to_mjy(mag):
    mag = np.asarray(mag, dtype=np.float64)
    return F0_IRAC2_MJY * 10.0 ** (-mag / 2.5)


def _download_path(config, module, name):
    path = f"{config.data_root}/sky/download/{module}/{name}"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "sky.derived.yso_census: no %r at %r -- run the "
            "'sesnaimpute.sky.download.%s.build' RUNBOOK line first" % (name, path, module))
    return path


def _read_fwf(path, colspecs, names):
    return pd.read_fwf(path, colspecs=colspecs, names=names, header=None)


def _num(series):
    """A fixed-width numeric column, blank fields and non-numeric junk
    both becoming `NaN` (`pd.to_numeric`'s own coercion) -- the CDS
    convention for a `?` (missing) field in a byte-by-byte table is
    simply blank bytes, not a sentinel, for every census here except
    Dunham's own `-999`/`-999.` (handled separately, unchanged from the
    retired module)."""
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _row(n, source, cloud, ra_deg, dec_deg, dist_pc=None, alpha=None,
         f45_mjy=None, selection="excess", survey_floor_a_k=None):
    """One census's own rows, as the common dict every `_census_*`
    function returns, `LOG10_F45_REF` filled in by the caller once
    `dist_pc` and `f45_mjy` are both known."""
    def _fill(value, dtype=np.float64):
        if value is None:
            return np.full(n, np.nan, dtype=dtype)
        arr = np.asarray(value, dtype=dtype)
        return np.broadcast_to(arr, (n,)).astype(dtype) if arr.shape == () else arr
    out = dict(
        source=np.full(n, source, dtype=object),
        cloud=np.asarray(cloud, dtype=object) if not isinstance(cloud, str)
              else np.full(n, cloud, dtype=object),
        ra_deg=_fill(ra_deg), dec_deg=_fill(dec_deg),
        dist_pc=_fill(dist_pc), alpha=_fill(alpha), f45_mjy=_fill(f45_mjy),
        selection=np.full(n, selection, dtype=object) if isinstance(selection, str)
                  else np.asarray(selection, dtype=object),
        survey_floor_a_k=_fill(survey_floor_a_k),
    )
    log10_f45_ref = np.full(n, np.nan, dtype=np.float64)
    finite = np.isfinite(out["f45_mjy"]) & np.isfinite(out["dist_pc"])
    log10_f45_ref[finite] = np.log10(out["f45_mjy"][finite] * (out["dist_pc"][finite] / 1000.0) ** 2)
    out["log10_f45_ref"] = log10_f45_ref
    return out


# ====================================================================
# Dunham et al. 2015 (J/ApJS/220/11) -- unchanged from dunham_yso
# ====================================================================

_DUNHAM_T2_COLSPECS = [(0, 4), (5, 21), (22, 38), (39, 43), (44, 51), (52, 58),
                        (59, 69), (70, 77), (78, 84), (85, 95)]
_DUNHAM_T2_NAMES = ["Seq", "Cloud", "ID", "Av", "alpha", "Tbol", "Lbol",
                     "alpha0", "Tbol0", "Lbol0"]
_DUNHAM_T34_COLSPECS = [(0, 4), (5, 14), (15, 26), (27, 36), (37, 46), (47, 56), (57, 66),
                         (67, 76), (77, 87), (88, 96), (97, 107), (108, 117), (118, 126),
                         (127, 135), (136, 144), (145, 153), (154, 162), (163, 169), (170, 177)]
_DUNHAM_T34_NAMES = ["Seq", "F1.25", "e_F1.25", "F1.65", "e_F1.65", "F2.17",
                      "e_F2.17", "F3.6", "e_F3.6", "F4.5", "e_F4.5", "F5.8",
                      "e_F5.8", "F8.0", "e_F8.0", "F24", "e_F24", "F70", "e_F70"]
_DUNHAM_ID_RE = re.compile(r"^J(\d{2})(\d{2})(\d{2}\.\d)([+-])(\d{2})(\d{2})(\d{2})$")
#: Dunham's own c2d/Gould-Belt coverage rule (`_W83_design.md`): the fit
#: footprint's column floor, by survey.
_DUNHAM_FLOOR = {"c2d": 0.22, "GB": 0.33}


def _dunham_sentinel_to_nan(df, skip=("Seq", "Cloud", "ID")):
    for col in df.columns:
        if col not in skip:
            df.loc[df[col] == _MISSING_DUNHAM, col] = np.nan
    return df


def _dunham_radec(id_series):
    m = id_series.str.strip().str.extract(_DUNHAM_ID_RE)
    if m.isna().any(axis=None):
        bad = id_series[m.isna().any(axis=1)].tolist()
        raise ValueError("sky.derived.yso_census: Dunham ID does not match "
                          "JHHMMSS.s+DDMMSS: %r" % bad[:5])
    hh, mm, ss = m[0].astype(np.float64), m[1].astype(np.float64), m[2].astype(np.float64)
    sign = np.where(m[3].to_numpy() == "-", -1.0, 1.0)
    dd, dm, ds = m[4].astype(np.float64), m[5].astype(np.float64), m[6].astype(np.float64)
    ra_deg = 15.0 * (hh + mm / 60.0 + ss / 3600.0)
    dec_deg = sign * (dd + dm / 60.0 + ds / 3600.0)
    return ra_deg.to_numpy(), dec_deg.to_numpy()


def _census_dunham2015(config):
    dist1, survey = {}, {}
    path = _download_path(config, "dunham2015", "table1.dat")
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("|")
            cloud = fields[0].strip()
            dist1[cloud] = int(re.match(r"\s*(\d+)", fields[2]).group(1))
            survey[cloud] = fields[1].strip()

    t2 = _dunham_sentinel_to_nan(_read_fwf(_download_path(config, "dunham2015", "table2.dat"),
                                             _DUNHAM_T2_COLSPECS, _DUNHAM_T2_NAMES))
    t4 = _dunham_sentinel_to_nan(_read_fwf(_download_path(config, "dunham2015", "table4.dat"),
                                             _DUNHAM_T34_COLSPECS, _DUNHAM_T34_NAMES))
    if not (t2["Seq"].to_numpy() == t4["Seq"].to_numpy()).all():
        raise ValueError("sky.derived.yso_census: Dunham table2/table4 Seq order disagrees")

    cloud = t2["Cloud"].str.strip()
    dist_pc = cloud.map(dist1).to_numpy(dtype=np.float64)
    if np.any(pd.isna(dist_pc)):
        raise ValueError("sky.derived.yso_census: a Dunham table2 Cloud has no table1 match")
    floor = cloud.map(lambda c: _DUNHAM_FLOOR[survey[c]]).to_numpy(dtype=np.float64)
    ra_deg, dec_deg = _dunham_radec(t2["ID"])
    return _row(len(t2), "Dunham2015", cloud.to_numpy(dtype=object), ra_deg, dec_deg,
                dist_pc=dist_pc, alpha=_num(t2["alpha0"]), f45_mjy=_num(t4["F4.5"]),
                selection="excess", survey_floor_a_k=floor)


# ====================================================================
# Megeath et al. 2012 (J/AJ/144/192) -- Orion A and B
# ====================================================================

_MEGEATH_COLSPECS = [(0, 4), (5, 7), (8, 10), (11, 16), (17, 18), (18, 20), (21, 23), (24, 28),
                      (29, 34), (35, 39), (40, 45), (46, 50), (51, 56), (57, 61),
                      (62, 67), (68, 72), (73, 78), (79, 83), (84, 89), (90, 94),
                      (95, 100), (101, 105), (106, 111), (112, 116), (117, 121), (122, 127), (128, 130)]
_MEGEATH_NAMES = ["Seq", "RAh", "RAm", "RAs", "DEsign", "DEd", "DEm", "DEs",
                   "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag",
                   "i1", "e_i1", "i2", "e_i2", "i3", "e_i3", "i4", "e_i4",
                   "m1", "e_m1", "AKs", "alpha", "Cl"]
#: Dec cut separating Megeath's own Orion A fields (ONC, L1641, kappa Ori
#: ext., Dec -5 to -9 deg) from the Orion B fields (NGC 2024/2068/71,
#: L1622, Dec -2 to +2 deg) -- module docstring's disclosed approximation.
_MEGEATH_DEC_SPLIT = -3.0


def _census_megeath2012(config):
    df = _read_fwf(_download_path(config, "megeath2012", "table4.dat"),
                    _MEGEATH_COLSPECS, _MEGEATH_NAMES)
    ra_deg = 15.0 * (_num(df["RAh"]) + _num(df["RAm"]) / 60.0 + _num(df["RAs"]) / 3600.0)
    sign = np.where(df["DEsign"].astype(str).str.strip() == "-", -1.0, 1.0)
    dec_deg = sign * (_num(df["DEd"]) + _num(df["DEm"]) / 60.0 + _num(df["DEs"]) / 3600.0)
    cloud = np.where(dec_deg < _MEGEATH_DEC_SPLIT, "Orion A", "Orion B")
    dist_pc = np.where(cloud == "Orion A", regions_module.REGIONS_BY_NAME["Orion A"].d_r_pc,
                        regions_module.REGIONS_BY_NAME["Orion B"].d_r_pc)
    f45_mjy = _mag_to_mjy(_num(df["i2"]))
    return _row(len(df), "Megeath2012", cloud, ra_deg, dec_deg, dist_pc=dist_pc,
                alpha=_num(df["alpha"]), f45_mjy=f45_mjy, selection="excess")


# ====================================================================
# Rebull et al. 2011 (J/ApJS/193/25) -- North America Nebula
# (Guieu et al. 2009, J/ApJ/697/787, fetched for cross-reference only --
# module docstring: not ingested as additional rows)
# ====================================================================

_REBULL2011_T389_COLSPECS = [(0, 18), (19, 29), (30, 39), (40, 56), (57, 80), (81, 93), (94, 116),
                              (117, 118), (118, 123), (124, 128), (129, 130), (130, 135), (136, 140),
                              (141, 142), (142, 147), (148, 152), (153, 154), (154, 159), (160, 164),
                              (165, 166), (166, 171), (172, 176), (177, 182), (183, 187)]
_REBULL2011_T389_NAMES = ["Name", "RAdeg", "DEdeg", "TwoMASS", "AName", "SpType", "r_SpType",
                           "l_i1", "i1", "e_i1", "l_i2", "i2", "e_i2", "l_i3", "i3", "e_i3",
                           "l_i4", "i4", "e_i4", "l_m1", "m1", "e_m1", "m2", "e_m2"]


def _census_rebull2011(config):
    t3 = _read_fwf(_download_path(config, "rebull2011", "rebull2011/table3.dat"),
                    _REBULL2011_T389_COLSPECS, _REBULL2011_T389_NAMES)
    t8 = _read_fwf(_download_path(config, "rebull2011", "rebull2011/table8.dat"),
                    _REBULL2011_T389_COLSPECS, _REBULL2011_T389_NAMES)
    both = pd.concat([t3, t8], ignore_index=True)
    ra_deg = _num(both["RAdeg"])
    dec_deg = _num(both["DEdeg"])
    f45_mjy = _mag_to_mjy(_num(both["i2"]))
    dist_pc = regions_module.REGIONS_BY_NAME["North America Nebula"].d_r_pc
    return _row(len(both), "Rebull2011", "NAN", ra_deg, dec_deg, dist_pc=dist_pc,
                alpha=None, f45_mjy=f45_mjy, selection="excess")


# ====================================================================
# Allen et al. 2012 (J/ApJ/750/125) -- Cepheus OB3b, disk-bearing only
# ====================================================================

_ALLEN_COLSPECS = [(0, 2), (3, 5), (6, 11), (12, 13), (13, 15), (16, 18), (19, 24),
                    (25, 32), (33, 38), (39, 45), (46, 51), (52, 57), (58, 62),
                    (63, 68), (69, 73), (74, 79), (80, 84), (85, 90), (91, 95),
                    (96, 101), (102, 106), (107, 112), (113, 117), (118, 123),
                    (124, 128), (129, 134), (135, 139), (140, 144), (145, 150)]
_ALLEN_NAMES = ["RAh", "RAm", "RAs", "DEsign", "DEd", "DEm", "DEs", "Vmag", "e_Vmag",
                 "VmI", "e_VmI", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Ksmag", "e_Ksmag",
                 "i1", "e_i1", "i2", "e_i2", "i3", "e_i3", "i4", "e_i4", "m1", "e_m1",
                 "AKs", "Note"]
_ALLEN_TABLES = ("table1.dat", "table2.dat", "table3.dat")  # class I, II, TD -- disk-bearing


def _census_allen2012(config):
    frames = [_read_fwf(_download_path(config, "allen2012", t), _ALLEN_COLSPECS, _ALLEN_NAMES)
              for t in _ALLEN_TABLES]
    df = pd.concat(frames, ignore_index=True)
    ra_deg = 15.0 * (_num(df["RAh"]) + _num(df["RAm"]) / 60.0 + _num(df["RAs"]) / 3600.0)
    sign = np.where(df["DEsign"].astype(str).str.strip() == "-", -1.0, 1.0)
    dec_deg = sign * (_num(df["DEd"]) + _num(df["DEm"]) / 60.0 + _num(df["DEs"]) / 3600.0)
    f45_mjy = _mag_to_mjy(_num(df["i2"]))
    dist_pc = regions_module.REGIONS_BY_NAME["Cepheus OB3"].d_r_pc
    return _row(len(df), "Allen2012", "Cepheus OB3b", ra_deg, dec_deg, dist_pc=dist_pc,
                alpha=None, f45_mjy=f45_mjy, selection="excess")


# ====================================================================
# Kirk et al. 2009 (J/ApJS/185/198) -- Cepheus Flare
# ====================================================================

_KIRK_T6_COLSPECS = [(0, 1), (2, 5), (6, 21), (22, 64), (86, 96)]
_KIRK_T6_NAMES = ["Type", "Seq", "Name", "ONames", "F4.5"]
_KIRK_T7_COLSPECS = [(0, 3), (4, 13), (14, 20), (21, 24), (25, 29), (30, 33), (34, 42), (43, 50)]
_KIRK_T7_NAMES = ["Seq", "Cloud", "alpha", "aCl", "Tbol", "Tcl", "Lbol", "LIR"]
_KIRK_NAME_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})(\d)([+-])(\d{2})(\d{2})(\d{2})(\d)$")


def _kirk_radec(name_series):
    m = name_series.str.strip().str.extract(_KIRK_NAME_RE)
    if m.isna().any(axis=None):
        bad = name_series[m.isna().any(axis=1)].tolist()
        raise ValueError("sky.derived.yso_census: Kirk2009 Name does not match "
                          "HHMMSSs+DDMMSSs: %r" % bad[:5])
    hh, mm = m[0].astype(np.float64), m[1].astype(np.float64)
    ss, s1 = m[2].astype(np.float64), m[3].astype(np.float64)
    sign = np.where(m[4].to_numpy() == "-", -1.0, 1.0)
    dd, dm = m[5].astype(np.float64), m[6].astype(np.float64)
    ds, s2 = m[7].astype(np.float64), m[8].astype(np.float64)
    ra_deg = 15.0 * (hh + mm / 60.0 + (ss + s1 / 10.0) / 3600.0)
    dec_deg = sign * (dd + dm / 60.0 + (ds + s2 / 10.0) / 3600.0)
    return ra_deg.to_numpy(), dec_deg.to_numpy()


def _census_kirk2009(config):
    t6 = _read_fwf(_download_path(config, "kirk2009", "table6.dat"), _KIRK_T6_COLSPECS, _KIRK_T6_NAMES)
    t7 = _read_fwf(_download_path(config, "kirk2009", "table7.dat"), _KIRK_T7_COLSPECS, _KIRK_T7_NAMES)
    if not (t6["Seq"].to_numpy() == t7["Seq"].to_numpy()).all():
        raise ValueError("sky.derived.yso_census: Kirk2009 table6/table7 Seq order disagrees")
    ra_deg, dec_deg = _kirk_radec(t6["Name"])
    cloud = t7["Cloud"].str.strip().to_numpy(dtype=object)
    dist_pc = regions_module.REGIONS_BY_NAME["Cepheus Flare"].d_r_pc
    return _row(len(t6), "Kirk2009", cloud, ra_deg, dec_deg, dist_pc=dist_pc,
                alpha=_num(t7["alpha"]), f45_mjy=_num(t6["F4.5"]), selection="excess")


# ====================================================================
# Strafella et al. 2010 (J/ApJ/719/9) -- Vela D
# ====================================================================

_STRAFELLA_T9_COLSPECS = [(0, 6), (7, 17), (18, 28), (29, 39), (40, 49), (50, 62), (63, 74), (75, 77),
                           (78, 90), (91, 102), (103, 105), (106, 118), (119, 130), (131, 133),
                           (134, 146), (147, 158), (159, 161), (162, 166), (167, 171), (172, 182),
                           (183, 192), (193, 194), (195, 204), (205, 214), (215, 216), (217, 222),
                           (223, 224), (225, 226), (227, 233)]
_STRAFELLA_T9_NAMES = ["IRAC", "RAdeg", "DEdeg", "GLON", "GLAT", "F36", "e_F36", "q_F36",
                        "F45", "e_F45", "q_F45", "F58", "e_F58", "q_F58", "F80", "e_F80", "q_F80",
                        "MIPS", "Dist", "F24", "e_F24", "q_F24", "F70", "e_F70", "q_F70",
                        "alpha", "CO12", "CO13", "Dust"]


def _census_strafella2010(config):
    df = _read_fwf(_download_path(config, "strafella2010", "table9.dat"),
                    _STRAFELLA_T9_COLSPECS, _STRAFELLA_T9_NAMES)
    ra_deg, dec_deg = _num(df["RAdeg"]), _num(df["DEdeg"])
    f45_jy = _num(df["F45"])
    f45_mjy = f45_jy * 1000.0
    dist_pc = regions_module.REGIONS_BY_NAME["Vela D"].d_r_pc
    return _row(len(df), "Strafella2010", "Vela D", ra_deg, dec_deg, dist_pc=dist_pc,
                alpha=_num(df["alpha"]), f45_mjy=f45_mjy, selection="excess")


# ====================================================================
# Rebull et al. 2010 (J/ApJS/186/259) -- Taurus, excess members only
# ====================================================================

_REBULL2010_COLSPECS = [(0, 15), (16, 39), (40, 47), (48, 56), (57, 61)]
_REBULL2010_NAMES = ["SST", "CName", "c2dCat", "GutCat", "Class"]
_REBULL2010_SST_RE = re.compile(r"^(\d{2})(\d{2})(\d{2}\.\d)([+-])(\d{2})(\d{2})(\d{2})$")
_REBULL2010_EXCESS_CLASSES = {"I", "II", "flat"}


def _rebull2010_radec(sst_series):
    m = sst_series.str.strip().str.extract(_REBULL2010_SST_RE)
    if m.isna().any(axis=None):
        bad = sst_series[m.isna().any(axis=1)].tolist()
        raise ValueError("sky.derived.yso_census: Rebull2010 SST does not match "
                          "HHMMSS.s+DDMMSS: %r" % bad[:5])
    hh, mm, ss = m[0].astype(np.float64), m[1].astype(np.float64), m[2].astype(np.float64)
    sign = np.where(m[3].to_numpy() == "-", -1.0, 1.0)
    dd, dm, ds = m[4].astype(np.float64), m[5].astype(np.float64), m[6].astype(np.float64)
    ra_deg = 15.0 * (hh + mm / 60.0 + ss / 3600.0)
    dec_deg = sign * (dd + dm / 60.0 + ds / 3600.0)
    return ra_deg.to_numpy(), dec_deg.to_numpy()


def _census_rebull2010(config):
    t6 = _read_fwf(_download_path(config, "rebull2010", "table6.dat"),
                    _REBULL2010_COLSPECS, _REBULL2010_NAMES)
    t7 = _read_fwf(_download_path(config, "rebull2010", "table7.dat"),
                    _REBULL2010_COLSPECS, _REBULL2010_NAMES)
    both = pd.concat([t6, t7], ignore_index=True)
    cls = both["Class"].astype(str).str.strip()
    keep = cls.isin(_REBULL2010_EXCESS_CLASSES)
    both = both[keep].reset_index(drop=True)
    ra_deg, dec_deg = _rebull2010_radec(both["SST"])
    dist_pc = regions_module.REGIONS_BY_NAME["Taurus"].d_r_pc
    return _row(len(both), "Rebull2010", "Taurus", ra_deg, dec_deg, dist_pc=dist_pc,
                alpha=None, f45_mjy=None, selection="excess")


# ====================================================================
# Kryukova et al. 2014 (J/AJ/148/11) -- Cygnus X, PROTOSTARS ONLY
# ====================================================================

_KRYUKOVA_COLSPECS = [(0, 2), (3, 5), (6, 11), (12, 14), (15, 17), (18, 23),
                       (24, 29), (30, 34), (35, 40), (41, 45), (46, 51), (52, 56),
                       (57, 62), (63, 67), (68, 73), (74, 78), (79, 84), (85, 89),
                       (90, 95), (96, 100), (101, 106), (107, 111), (112, 118),
                       (119, 127), (128, 134)]
_KRYUKOVA_NAMES = ["RAh", "RAm", "RAs", "DEd", "DEm", "DEs", "Jmag", "e_Jmag",
                    "Hmag", "e_Hmag", "Ksmag", "e_Ksmag", "i1", "e_i1", "i2", "e_i2",
                    "i3", "e_i3", "i4", "e_i4", "m1", "e_m1", "alpha", "RMEDSQ", "logL"]


def _census_kryukova2014(config):
    df = _read_fwf(_download_path(config, "kryukova2014", "table3.dat"),
                    _KRYUKOVA_COLSPECS, _KRYUKOVA_NAMES)
    ra_deg = 15.0 * (_num(df["RAh"]) + _num(df["RAm"]) / 60.0 + _num(df["RAs"]) / 3600.0)
    # Cygnus-X is entirely north of the celestial equator -- the ReadMe's
    # own byte-by-byte carries no declination-sign column at all.
    dec_deg = _num(df["DEd"]) + _num(df["DEm"]) / 60.0 + _num(df["DEs"]) / 3600.0
    f45_mjy = _mag_to_mjy(_num(df["i2"]))
    dist_pc = regions_module.REGIONS_BY_NAME["Cygnus X"].d_r_pc
    return _row(len(df), "Kryukova2014", "Cygnus X", ra_deg, dec_deg, dist_pc=dist_pc,
                alpha=_num(df["alpha"]), f45_mjy=f45_mjy, selection="protostar")


_CENSUS_BUILDERS = (
    _census_dunham2015, _census_megeath2012, _census_rebull2011, _census_allen2012,
    _census_kirk2009, _census_strafella2010, _census_rebull2010, _census_kryukova2014,
)


# ====================================================================
# coverage: one convex-hull polygon per (SOURCE, CLOUD)
# ====================================================================

def _tangent_plane(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Gnomonic (tangent-plane) projection of `(ra_deg, dec_deg)` about
    `(ra0_deg, dec0_deg)`, in degrees, small-field-accurate -- the local
    flat frame a convex hull needs to be convex ON THE SKY (module
    docstring)."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    ra0, dec0 = np.radians(ra0_deg), np.radians(dec0_deg)
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    x = np.cos(dec) * np.sin(ra - ra0) / cos_c
    y = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c
    return np.degrees(x), np.degrees(y)


def _inverse_tangent_plane(x_deg, y_deg, ra0_deg, dec0_deg):
    x, y = np.radians(x_deg), np.radians(y_deg)
    ra0, dec0 = np.radians(ra0_deg), np.radians(dec0_deg)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    c = np.where(rho > 1e-12, c, 0.0)
    sin_c, cos_c = np.sin(c), np.cos(c)
    safe_rho = np.where(rho > 1e-12, rho, 1.0)
    dec = np.arcsin(cos_c * np.sin(dec0) + (y * sin_c * np.cos(dec0)) / safe_rho)
    ra = ra0 + np.arctan2(x * sin_c, rho * np.cos(dec0) * cos_c - y * np.sin(dec0) * sin_c)
    return np.degrees(ra) % 360.0, np.degrees(dec)


def _convex_hull_polygon(ra_deg, dec_deg):
    """The convex hull of `(ra_deg, dec_deg)`, on the sky (module
    docstring): projected to a local tangent plane centred on the mean
    position, hulled there (`scipy.spatial.ConvexHull`), projected back.
    Fewer than 3 distinct positions have no hull -- the single point (or
    the two-point segment) is returned as a degenerate "polygon" so the
    caller always gets SOME coverage record, consistent with
    `ConvexHull`'s own requirement of at least 3 non-collinear points."""
    ra0, dec0 = float(np.mean(ra_deg)), float(np.mean(dec_deg))
    x, y = _tangent_plane(ra_deg, dec_deg, ra0, dec0)
    pts = np.column_stack([x, y])
    uniq = np.unique(pts, axis=0)
    if uniq.shape[0] < 3:
        hx, hy = uniq[:, 0], uniq[:, 1]
    else:
        try:
            hull = ConvexHull(pts)
            hx, hy = pts[hull.vertices, 0], pts[hull.vertices, 1]
        except Exception:
            hx, hy = uniq[:, 0], uniq[:, 1]
    return _inverse_tangent_plane(hx, hy, ra0, dec0)


def polygon_contains(ra_deg, dec_deg, poly_ra_deg, poly_dec_deg):
    """True where `(ra_deg, dec_deg)` (arrays, any shape) fall inside the
    sky polygon `(poly_ra_deg, poly_dec_deg)` (ordered vertices -- a
    `coverage` group's own `RA_DEG`/`DEC_DEG`, or any other closed
    ordered loop): both projected into the SAME local tangent plane
    used to BUILD a hull (`_convex_hull_polygon`), centred on the
    polygon's own mean vertex position, and tested there with
    `matplotlib.path.Path` (`population.yso_law`'s own fit-footprint
    test, the public reason this function has no leading underscore).
    Fewer than 3 polygon vertices (a degenerate single-point or
    two-point "polygon", `_convex_hull_polygon`'s own fallback for a
    census group with too few distinct positions to hull) contains
    nothing -- `False` everywhere, not an exception, since a region
    whose only overlapping census group is degenerate simply has no
    census-polygon footprint there.
    """
    ra0 = float(np.mean(poly_ra_deg))
    dec0 = float(np.mean(poly_dec_deg))
    px, py = _tangent_plane(np.asarray(poly_ra_deg, dtype=np.float64),
                             np.asarray(poly_dec_deg, dtype=np.float64), ra0, dec0)
    shape = np.shape(ra_deg)
    qx, qy = _tangent_plane(np.asarray(ra_deg, dtype=np.float64).ravel(),
                             np.asarray(dec_deg, dtype=np.float64).ravel(), ra0, dec0)
    if px.shape[0] < 3:
        return np.zeros(shape, dtype=bool)
    path = _MplPath(np.column_stack([px, py]))
    inside = path.contains_points(np.column_stack([qx, qy]))
    return inside.reshape(shape)


def _write_coverage(f, source_arr, cloud_arr, ra_deg, dec_deg):
    group = f.require_group("coverage")
    pairs = sorted(set(zip(source_arr.tolist(), cloud_arr.tolist())))
    for source, cloud in pairs:
        sel = (source_arr == source) & (cloud_arr == cloud)
        poly_ra, poly_dec = _convex_hull_polygon(ra_deg[sel], dec_deg[sel])
        name = f"{source}__{cloud}".replace(" ", "_").replace("/", "-")
        sub = group.require_group(name)
        reading = (
            f"The convex-hull coverage polygon of the {source} census's own "
            f"{cloud} objects ({int(sel.sum())} positions), ordered RA/Dec "
            "vertices on the sky (module docstring's tangent-plane "
            "construction): `population.yso_law`'s fit footprint for a "
            "region this polygon overlaps is admitted pixels inside it "
            "(and above the census's own SURVEY_FLOOR_A_K, where finite)."
        )
        build_module.write_dataset(sub, "RA_DEG", poly_ra.astype(np.float64), "deg", reading)
        build_module.write_dataset(sub, "DEC_DEG", poly_dec.astype(np.float64), "deg", reading)
        sub.attrs["SOURCE"] = source
        sub.attrs["CLOUD"] = cloud
        sub.attrs["N_OBJECTS"] = int(sel.sum())


# ====================================================================
# build
# ====================================================================

def build(config, regions=None):
    """Writes `sky/derived/yso_census/census_yso_census_survey.hdf5`
    (module docstring): every ingested census's rows, concatenated, plus
    the `coverage` group's per-`(SOURCE, CLOUD)` convex-hull polygons.
    `regions` is accepted for interface uniformity and ignored: this is
    a survey-wide product built from every census's own download, not
    restricted to any one SESNA region.
    """
    with progress_module.Stage("sky.derived.yso_census") as st:
        parts = []
        for i, fn in enumerate(_CENSUS_BUILDERS):
            parts.append(fn(config))
            st.tick(i + 1, len(_CENSUS_BUILDERS), fn.__name__)

        keys = ["source", "cloud", "ra_deg", "dec_deg", "dist_pc", "alpha",
                "f45_mjy", "log10_f45_ref", "selection", "survey_floor_a_k"]
        combined = {k: np.concatenate([p[k] for p in parts]) for k in keys}

        path = config_module.product_path(config, *PATH_ARGS)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            for name, data, dtype in (
                ("SOURCE", combined["source"], "S16"),
                ("CLOUD", combined["cloud"], "S24"),
                ("RA_DEG", combined["ra_deg"], np.float64),
                ("DEC_DEG", combined["dec_deg"], np.float64),
                ("DIST_PC", combined["dist_pc"], np.float32),
                ("ALPHA", combined["alpha"], np.float32),
                ("F45_MJY", combined["f45_mjy"], np.float32),
                ("LOG10_F45_REF", combined["log10_f45_ref"], np.float32),
                ("SELECTION", combined["selection"], "S12"),
                ("SURVEY_FLOOR_A_K", combined["survey_floor_a_k"], np.float32),
            ):
                if dtype in ("S16", "S24", "S12"):
                    arr = np.array([v.encode("utf-8") for v in data], dtype=dtype)
                else:
                    arr = np.asarray(data, dtype=dtype)
                build_module.write_dataset(f, name, arr, *_READINGS[name])
            _write_coverage(f, combined["source"], combined["cloud"],
                             combined["ra_deg"], combined["dec_deg"])

        n_by_source = {s: int(np.sum(combined["source"] == s)) for s in sorted(set(combined["source"]))}
        st.done(path, rows=len(combined["source"]), sources=len(n_by_source))
        print("sky.derived.yso_census rows by SOURCE: " +
              ", ".join(f"{s}={n}" for s, n in n_by_source.items()))
    return path


if __name__ == "__main__":
    run(build)
