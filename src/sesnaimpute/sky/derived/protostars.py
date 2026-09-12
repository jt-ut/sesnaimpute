"""HOPS and eHOPS in one schema: an independent, Herschel-confirmed
protostar sample for the report-only check of `atlas.protostars`
(SPEC_BMSTP_DRAFT.md sec. 5.5, sec. 9). A view of the two VOTables
`sky.download.hops` fetches -- read, never computed, so this module lives
under `sky/derived` rather than as a modelling product.

HOPS (Furlan et al. 2016, VizieR `J/ApJS/224/5`, table1) carries
`RAJ2000, DEJ2000, Class, Lbol, Tbol, Av` for 330 Orion protostars, `Class`
already one of "0"/"I"/"flat"/"II". eHOPS (Pokhrel et al. 2023, VizieR
`J/ApJS/266/32`, the `ehops` table) carries the same six quantities for
172 Aquila protostars, `Class` spelled "Class 0"/"Class 1"/"Flat"/"Class 2"
-- mapped here onto HOPS's own four labels. `Av` in both catalogues is the
SED model's fitted foreground extinction in the V band.

`REGION` is the SESNA region whose admitted footprint (the granule map's
own nside-256 source-bearing grains, `granules.access.primary_region_for_
pixel`) contains the protostar's position, `""` where it falls outside
every region's footprint -- never enters the prior or posterior, so
nothing here feeds a fit.

The 4.5 micron datum (`atlas.protostars`'s own likelihood factor, this
package's report-only check, SPEC_BMSTP_DRAFT.md sec. 5.5/9): `F45_MJY`,
`E_F45_MJY` (NaN where not measured) and `F45_MEASURED`. HOPS carries its
photometry in the VOTable's second table (`table2`, "SED data"), joined to
`table1` by the `HOPS` id, fluxes in Jy here converted to mJy; that
table's `f_F4.5` flag takes the values 1, 2 or 3 with no flag-key text in
this VOTable's own (mini-header) download, but empirically flag 3
coincides exactly with `F4.5` itself carrying no value (no photometry
obtained there), while flags 1 and 2 both carry a stored flux -- so only
flag 3 is treated as a non-measurement. eHOPS carries `F4.5`/`e_F4.5` in
its one table already in mJy; a masked value there is the non-measurement.
"""

import os

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse as parse_votable

from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.granules import access

#: (VOTable file, VizieR table name, id column, class values -> HOPS's
#: own four labels).
_CATALOGS = (
    ("HOPS", "J_ApJS_224_5.vot", "J/ApJS/224/5/table1", "HOPS",
     {"0": "0", "I": "I", "flat": "flat", "II": "II"}),
    ("eHOPS", "J_ApJS_266_32.vot", "J/ApJS/266/32/ehops", "eHOPS",
     {"Class 0": "0", "Class 1": "I", "Flat": "flat", "Class 2": "II"}),
)


#: HOPS table2's own `f_F4.5` flag (module docstring): the one value that
#: coincides with the field carrying no photometry at all -- not a
#: documented flag key (this VOTable's mini header carries none), but
#: measured directly off the download (counted 2026-09-12: every one of
#: the 14 masked `F4.5` rows among the 319 Class 0/I/flat protostars
#: carries this flag, and no other row does).
HOPS_F45_NODATA_FLAG = 3


def _table_array(path, table_name):
    """The one named VizieR table's rows, from the VOTable at `path`."""
    votable = parse_votable(path)
    for resource in votable.resources:
        for table in resource.tables:
            if table.name == table_name:
                return table.array
    raise ValueError(f"sky.derived.protostars: no table {table_name!r} in {path!r}")


def _hops_f45(path, id1):
    """HOPS's own 4.5 micron datum (module docstring): `table2`'s
    `F4.5`/`e_F4.5` (Jy, converted to mJy here) and `f_F4.5`, joined to
    `table1`'s own `HOPS` id by one `searchsorted` (no per-row loop, rule
    8). Every `table1` id is required to appear in `table2` -- HOPS's own
    photometry table, not an optional join."""
    arr = _table_array(path, "J/ApJS/224/5/table2")
    id2 = np.ma.filled(arr["HOPS"])
    order = np.argsort(id2)
    loc = np.searchsorted(id2[order], id1)
    found = id2[order][loc] == id1
    if not np.all(found):
        raise ValueError(
            f"sky.derived.protostars: {int(np.count_nonzero(~found))} HOPS table1 "
            f"id(s) missing from table2's own photometry in {path!r}")
    row = order[loc]
    flag = np.ma.filled(arr["f_F4.5"], HOPS_F45_NODATA_FLAG)[row]
    f45_jy = arr["F4.5"][row]
    e45_jy = arr["e_F4.5"][row]
    measured = ~np.ma.getmaskarray(f45_jy) & (flag != HOPS_F45_NODATA_FLAG)
    f45_mjy = np.where(measured, np.ma.filled(f45_jy, np.nan) * 1000.0, np.nan).astype(np.float32)
    e45_mjy = np.where(measured, np.ma.filled(e45_jy, np.nan) * 1000.0, np.nan).astype(np.float32)
    return f45_mjy, e45_mjy, measured


def _read_one(config, survey, file_name, table_name, id_field, class_map):
    """One catalogue's protostars in the shared schema's column order,
    `Class` mapped onto HOPS's own four labels, positions and marks read
    verbatim (`np.ma.filled` unmasks the VOTable's masked floats; neither
    catalogue's `RAJ2000/DEJ2000/Class/Lbol/Tbol/Av` carries a mask, see
    module notes). `F45_MJY`/`E_F45_MJY`/`F45_MEASURED`: HOPS joins
    `table2` (`_hops_f45`); eHOPS reads `F4.5`/`e_F4.5` straight off this
    same table, already in mJy, a masked value the non-measurement."""
    path = f"{config.data_root}/sky/download/hops/{file_name}"
    if not os.path.exists(path):
        raise ValueError(
            f"sky.derived.protostars: no {path!r} -- run RUNBOOKtp.sh's "
            f"'PY sesnaimpute.sky.download.hops.build' line")
    arr = _table_array(path, table_name)
    n = len(arr)
    class_native = np.ma.filled(arr["Class"], "")
    class_mapped = np.array([class_map[c] for c in class_native])
    if survey == "HOPS":
        f45_mjy, e45_mjy, f45_measured = _hops_f45(path, np.ma.filled(arr[id_field]))
    else:
        f45 = arr["F4.5"]
        e45 = arr["e_F4.5"]
        f45_measured = ~np.ma.getmaskarray(f45)
        f45_mjy = np.ma.filled(f45, np.nan).astype(np.float32)
        e45_mjy = np.ma.filled(e45, np.nan).astype(np.float32)
    return dict(
        survey=np.full(n, survey, dtype="S8"),
        id=np.array([str(v) for v in np.ma.filled(arr[id_field])], dtype="S16"),
        ra_deg=np.ma.filled(arr["RAJ2000"], np.nan).astype(np.float64),
        dec_deg=np.ma.filled(arr["DEJ2000"], np.nan).astype(np.float64),
        cls=class_mapped.astype("S8"),
        lbol_lsun=np.ma.filled(arr["Lbol"], np.nan).astype(np.float32),
        tbol_k=np.ma.filled(arr["Tbol"], np.nan).astype(np.float32),
        av_mag=np.ma.filled(arr["Av"], np.nan).astype(np.float32),
        f45_mjy=f45_mjy, e45_mjy=e45_mjy, f45_measured=f45_measured,
    )


def _regions(config, ra_deg, dec_deg):
    """Each protostar's SESNA region: its nside-256 grain (the granule
    map's own galactic NESTED pixelisation, `granules/build.py`) looked
    up in the region association (the admitted footprint, sec. 3's grain
    map), `""` where the grain belongs to no region."""
    gal = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    pix256 = hp.ang2pix(256, gal.l.deg, gal.b.deg, nest=True, lonlat=True)
    names = access.primary_region_for_pixel(config, pix256)
    return np.array([n.encode("utf-8") for n in names], dtype="S32")


def build(config, regions=None):
    """Writes `sky/derived/protostars/protostars_survey.hdf5`: HOPS and
    eHOPS pooled in one schema, `regions` accepted for interface
    uniformity and ignored (survey-wide: both catalogues' full row set is
    written regardless of which regions the caller names)."""
    with progress_module.Stage("sky.derived.protostars") as st:
        parts = []
        for i, (survey, file_name, table_name, id_field, class_map) in enumerate(_CATALOGS):
            parts.append(_read_one(config, survey, file_name, table_name, id_field, class_map))
            st.tick(i + 1, len(_CATALOGS), "catalogues")

        survey = np.concatenate([p["survey"] for p in parts])
        ident = np.concatenate([p["id"] for p in parts])
        ra_deg = np.concatenate([p["ra_deg"] for p in parts])
        dec_deg = np.concatenate([p["dec_deg"] for p in parts])
        cls = np.concatenate([p["cls"] for p in parts])
        lbol = np.concatenate([p["lbol_lsun"] for p in parts])
        tbol = np.concatenate([p["tbol_k"] for p in parts])
        av = np.concatenate([p["av_mag"] for p in parts])
        f45_mjy = np.concatenate([p["f45_mjy"] for p in parts])
        e45_mjy = np.concatenate([p["e45_mjy"] for p in parts])
        f45_measured = np.concatenate([p["f45_measured"] for p in parts])
        region = _regions(config, ra_deg, dec_deg)

        path = f"{config.data_root}/sky/derived/protostars/protostars_survey.hdf5"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            f.create_dataset("SURVEY", data=survey)
            f.create_dataset("ID", data=ident)
            f.create_dataset("RA_DEG", data=ra_deg)
            f.create_dataset("DEC_DEG", data=dec_deg)
            f.create_dataset("CLASS", data=cls)
            f.create_dataset("LBOL_LSUN", data=lbol)
            f.create_dataset("TBOL_K", data=tbol)
            f.create_dataset("AV_FOREGROUND_MAG", data=av)
            f.create_dataset("F45_MJY", data=f45_mjy)
            f.create_dataset("E_F45_MJY", data=e45_mjy)
            f.create_dataset("F45_MEASURED", data=f45_measured)
            f.create_dataset("REGION", data=region)

        n_by_region = {}
        for name in np.unique(region):
            n_by_region[name.decode() or "(none)"] = int(np.count_nonzero(region == name))
        n_measured_by_survey = {}
        for survey_name in ("HOPS", "eHOPS"):
            m = survey == survey_name.encode("utf-8")
            n_measured_by_survey[survey_name] = (
                int(np.count_nonzero(f45_measured[m])), int(np.count_nonzero(~f45_measured[m])))
        st.done(path, n_hops=int(np.count_nonzero(survey == b"HOPS")),
                n_ehops=int(np.count_nonzero(survey == b"eHOPS")),
                n_total=survey.size, n_regions=len(n_by_region))
        print(f"sky.derived.protostars: by region {n_by_region}")
        print(f"sky.derived.protostars: F4.5 measured/not-measured by survey (all classes, "
              f"HOPS's own {HOPS_F45_NODATA_FLAG!r} no-data flag) {n_measured_by_survey}")


if __name__ == "__main__":
    run(build)
