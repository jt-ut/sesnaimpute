"""The four H2-shock-knot survey tables SPEC_PRIORS.md section 7 (H2S) reads,
and the knot-colour ratio distribution its selection `eps_s` needs.

Parses each survey's own table from the bytes `sky.download.h2_knot_surveys`
copied, into one plain HDF5 per survey (`GRANULE="survey"`: these are
whole-survey tables, not per-region products):

    Giannini T. et al. 2013, ApJ 767, 147 (Vela-D): the 2.12 um flux and the
        four IRAC band fluxes, re-parsed from the raw VizieR fixed-width
        table (the byte ranges in its ReadMe) rather than the summed-IRAC-
        luminosity-only column the predecessor package kept (S-L4).
    Davis C.J. et al. 2009, A&A 496, 153 (Orion A): knot positions and the
        proper-motion table's own velocity/position-angle/quality columns
        -- no flux is in the raw table.
    Walawender J. et al. 2005 (Perseus, AJ 129, 2308; Barnard 1, AJ 130,
        1795): knot positions. Barnard 1's own table 3 is titled "Positions
        of H2 Shocks" -- genuinely narrowband-H2-selected. Perseus's own
        tables 2 and 3 are titled "Positions of ... HH Objects" (previously
        known and newly discovered) -- an optical Halpha/[S II] Herbig-Haro
        catalogue, not an H2 narrowband selection. This module carries both
        as the raw tables report them, with a per-row `H2_SELECTED` flag
        (True for Barnard 1 only) so a consumer can honour SPEC_PRIORS.md
        section 7's "H2-selected knots only; optical HH objects ...
        excluded by design" -- the Perseus component does not meet that
        description on its own terms; this is reported, not resolved here.
    Froebrich D. et al. 2015, MNRAS 454, 2586 (UWISH2, all-sky): surface
        brightness, size and the jet/PN/SNR/unknown classification code,
        read from the full 33,200-feature transcription (the VizieR table
        under this survey's own name is a 69-row excerpt only, its ReadMe
        says so explicitly).

Also builds the knot-colour distribution SPEC_PRIORS.md section 7's "The
knot colours" paragraph specifies: for each Giannini knot carrying both a
2.12 um flux and a given IRAC band flux, `log10(F_band / F_2.12)`, and that
ratio's median and 16-84% spread per band. Giannini+2013's own table
reports both the 2.12 um line flux and the four IRAC fluxes as flux
densities in the same unit (mJy, table2.dat Bytes-by-byte: F2.12, F3.6,
F4.5, F5.8, F8.0 are all "mJy") -- so the ratio needs no unit conversion
between a line flux and a band flux density; had Giannini instead reported
F2.12 as a line flux in erg/s/cm^2, the line would need converting to an
equivalent Ks-band flux density through the 2MASS Ks bandwidth first. Upper
limits (the raw table's `l_F*` flag columns) are excluded from the ratio:
a ratio against an upper limit is not a measured colour.

Feeds SPEC_PRIORS.md section 7 (H2S): `eps_ext` (positional cross-match
against these four surveys), `eps_s` (the knot-colour ratio distribution,
below), and the UWISH2-only jet-class surface-brightness shape.
"""

import email
import os
from html import unescape

import h5py
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.config import product_path

SURVEYS = ("giannini2013", "davis2009", "walawender2005", "uwish2")

#: Citations, one per survey table this module parses.
CITATIONS = {
    "giannini2013": "Giannini T. et al. 2013, ApJ 767, 147 (Vela-D H2 jets and knots)",
    "davis2009": "Davis C.J. et al. 2009, A&A 496, 153 (Orion A H2 outflow census)",
    "walawender2005": "Walawender J., Bally J. & Reipurth B. 2005, AJ 129, 2308 (Perseus); "
                       "Walawender J., Bally J., Di Francesco J. & Jorgensen J. 2005, AJ 130, 1795 (Barnard 1)",
    "uwish2": "Froebrich D. et al. 2015, MNRAS 454, 2586 (UWISH2 extended H2 sources)",
}

#: Giannini+2013 table2.dat fixed-width layout (0-based half-open byte
#: ranges converted from the raw ReadMe's 1-based inclusive byte ranges).
_GIANNINI_COLSPECS = [
    (0, 4), (4, 5), (6, 8), (9, 11), (12, 16),
    (17, 18), (18, 20), (21, 23), (24, 26), (27, 30),
    (31, 36), (37, 42), (43, 44), (44, 48), (49, 53),
    (54, 58), (59, 63), (64, 65), (65, 70), (71, 75),
    (76, 77), (77, 81), (82, 86), (87, 93), (94, 99),
]
_GIANNINI_NAMES = [
    "KnotID", "n_KnotID", "RAh", "RAm", "RAs",
    "DEsign", "DEd", "DEm", "DEs", "Size",
    "F2p12", "e_F2p12", "l_F3p6", "F3p6", "e_F3p6",
    "F4p5", "e_F4p5", "l_F5p8", "F5p8", "e_F5p8",
    "l_F8p0", "F8p0", "e_F8p0", "LIRAC", "L2p12",
]
#: The four IRAC bands Giannini's table carries, key -> (flux column,
#: upper-limit-flag column or None when the table reports no limits for
#: that band), wavelength label for the product's own documentation.
_GIANNINI_IRAC_BANDS = {
    "I1": ("F3p6", "l_F3p6", 3.6),
    "I2": ("F4p5", None, 4.5),
    "I3": ("F5p8", "l_F5p8", 5.8),
    "I4": ("F8p0", "l_F8p0", 8.0),
}

#: Davis+2009 tableb1.dat fixed-width layout (proper-motion table; the raw
#: table carries no flux column, S-L4).
_DAVIS_COLSPECS = [
    (0, 3), (4, 7), (7, 8), (10, 17),
    (19, 20), (21, 23), (24, 28),
    (29, 30), (30, 31), (32, 34), (35, 37),
    (38, 44), (45, 51), (52, 56), (57, 61),
    (62, 67), (68, 74), (75, 76),
]
_DAVIS_NAMES = [
    "SMZ", "SMZ2", "n_SMZ2", "Knot",
    "RAh", "RAm", "RAs",
    "DEsign", "DEd", "DEm", "DEs",
    "shiftx", "shifty", "shiftpix", "shift",
    "Vel", "Position", "Flag",
]


def _raw_dir(config):
    return f"{config.data_root}/sky/download/h2_knot_surveys"


def _require(path, runbook_line):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"sky.derived.knots.build: no file at {path} -- run "
            f"`{runbook_line}` first")


def _sexagesimal_deg(hours_or_deg, minutes, seconds, is_ra, sign=None):
    """Vectorised sexagesimal-to-degree conversion. `hours_or_deg` is hours
    for RA (`is_ra=True`, scaled by 15) or unsigned degrees for Dec
    (`is_ra=False`, combined with `sign`, +1/-1 per row)."""
    magnitude = hours_or_deg.astype(np.float64) + minutes.astype(np.float64) / 60.0 \
        + seconds.astype(np.float64) / 3600.0
    if is_ra:
        return 15.0 * magnitude
    return sign * magnitude


# ---------------------------------------------------------------- Giannini

def _parse_giannini(raw_dir):
    """Vela-D knots: KnotID, position, size, the 2.12 um flux and the four
    IRAC band fluxes, all in mJy (table2.dat's own units)."""
    path = f"{raw_dir}/vizier/J_ApJ_767_147/table2.dat"
    _require(path, "sesnaimpute.sky.download.h2_knot_surveys.build")
    df = pd.read_fwf(path, colspecs=_GIANNINI_COLSPECS, names=_GIANNINI_NAMES,
                      header=None, na_values=["---"], dtype=str)
    numeric_cols = ["RAh", "RAm", "RAs", "DEd", "DEm", "DEs", "Size",
                     "F2p12", "e_F2p12", "F3p6", "e_F3p6", "F4p5", "e_F4p5",
                     "F5p8", "e_F5p8", "F8p0", "e_F8p0", "LIRAC", "L2p12"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    sign = np.where(df["DEsign"].to_numpy() == "-", -1.0, 1.0)
    ra_deg = _sexagesimal_deg(df["RAh"], df["RAm"], df["RAs"], is_ra=True)
    dec_deg = _sexagesimal_deg(df["DEd"], df["DEm"], df["DEs"], is_ra=False, sign=sign)

    out = {
        "KNOT_ID": df["KnotID"].str.strip().to_numpy(dtype="S8"),
        "RA_DEG": ra_deg.to_numpy(dtype=np.float64),
        "DEC_DEG": dec_deg.to_numpy(dtype=np.float64),
        "SIZE_ARCSEC2": df["Size"].to_numpy(dtype=np.float64),
        "FLUX_2P12_MJY": df["F2p12"].to_numpy(dtype=np.float64),
        "FLUX_ERR_2P12_MJY": df["e_F2p12"].to_numpy(dtype=np.float64),
        "LUMINOSITY_IRAC_1E-2LSUN": df["LIRAC"].to_numpy(dtype=np.float64),
        "LUMINOSITY_2P12_1E-2LSUN": df["L2p12"].to_numpy(dtype=np.float64),
    }
    for band_key, (flux_col, limit_col, _wave) in _GIANNINI_IRAC_BANDS.items():
        is_limit = (df[limit_col].to_numpy() == "<") if limit_col else np.zeros(len(df), dtype=bool)
        out[f"FLUX_{band_key}_MJY"] = df[flux_col].to_numpy(dtype=np.float64)
        out[f"FLUX_ERR_{band_key}_MJY"] = df[f"e_{flux_col}"].to_numpy(dtype=np.float64)
        out[f"UPPER_LIMIT_{band_key}"] = is_limit
    return out


# ------------------------------------------------------------------ Davis

def _parse_davis(raw_dir):
    """Orion A proper-motion knots: position, tangential velocity, PA of
    the proper-motion vector, and the table's own quality flag (0 good,
    2 uncertain per the ReadMe) -- no flux column exists in the raw table."""
    path = f"{raw_dir}/vizier/J_A+A_496_153/tableb1.dat"
    _require(path, "sesnaimpute.sky.download.h2_knot_surveys.build")
    df = pd.read_fwf(path, colspecs=_DAVIS_COLSPECS, names=_DAVIS_NAMES,
                      header=None, dtype=str)
    numeric_cols = ["RAh", "RAm", "RAs", "DEd", "DEm", "DEs", "Vel", "Position", "Flag"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    sign = np.where(df["DEsign"].to_numpy() == "-", -1.0, 1.0)
    ra_deg = _sexagesimal_deg(df["RAh"], df["RAm"], df["RAs"], is_ra=True)
    dec_deg = _sexagesimal_deg(df["DEd"], df["DEm"], df["DEs"], is_ra=False, sign=sign)
    return dict(
        KNOT_ID=df["Knot"].str.strip().to_numpy(dtype="S16"),
        RA_DEG=ra_deg.to_numpy(dtype=np.float64),
        DEC_DEG=dec_deg.to_numpy(dtype=np.float64),
        TANGENTIAL_VELOCITY_KM_S=df["Vel"].to_numpy(dtype=np.float64),
        POSITION_ANGLE_DEG=df["Position"].to_numpy(dtype=np.float64),
        QUALITY_FLAG=df["Flag"].to_numpy(dtype=np.float64),
    )


# ------------------------------------------------------------- Walawender

def _read_html_table_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_mhtml_table_text(path):
    """Reconstructs the article's raw HTML from an owner-saved devtools
    source-view mhtml snapshot: the multipart html part's `line-content`
    cells carry the page's HTML entity-escaped, one source line per cell."""
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f)
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True).decode("utf-8", errors="replace")
            soup = BeautifulSoup(payload, "html.parser")
            cells = soup.find_all("td", class_="line-content")
            if cells:
                return "\n".join(unescape(td.get_text()) for td in cells)
            return payload
    raise ValueError(f"walawender2005 parse: no HTML part in {path}")


def _dec_deg_unsigned_north(dec_str):
    """Walawender's Dec strings carry no explicit sign; every knot in
    these two fields (Perseus, Barnard 1) is north of the celestial
    equator, so an absent sign means positive."""
    sign = -1.0 if dec_str.strip().startswith("-") else 1.0
    d, m, s = dec_str.replace("-", "").split()
    return sign * (float(d) + float(m) / 60.0 + float(s) / 3600.0)


def _ra_deg(ra_str):
    h, m, s = ra_str.split()
    return 15.0 * (float(h) + float(m) / 60.0 + float(s) / 3600.0)


def _clean_designation(text):
    """Normalises the non-breaking space the Barnard 1 mhtml source uses
    between a label and its number (e.g. "MH\\xa01") to a plain space, so
    the designation round-trips through an ascii HDF5 byte-string field."""
    return text.replace("\xa0", " ").strip()


def _perseus_previously_known(raw_dir):
    """Table 2 (Walawender+2005, AJ 129, 2308): "Positions of All
    Previously Known HH Objects", a two-column-per-row layout (two
    independent HH entries side by side)."""
    path = f"{raw_dir}/walawender/Walawender2005(AJ 129, 2308)_tbl2.html"
    soup = BeautifulSoup(_read_html_table_text(path), "html.parser")
    rows = soup.find_all("tbody")[0].find_all("tr")
    out = []
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) >= 3:
            out.append((tds[0].get_text(strip=True).rstrip("."),
                        tds[1].get_text(strip=True), tds[2].get_text(strip=True)))
        if len(tds) >= 6:
            out.append((tds[3].get_text(strip=True).rstrip("."),
                        tds[4].get_text(strip=True), tds[5].get_text(strip=True)))
    return out, "hh_known"


def _perseus_new(raw_dir):
    """Table 3 (Walawender+2005, AJ 129, 2308): "Positions of New HH
    Objects Discovered"."""
    path = f"{raw_dir}/walawender/Walawender2005(AJ 129, 2308)_tbl3.html"
    soup = BeautifulSoup(_read_html_table_text(path), "html.parser")
    rows = soup.find_all("tbody")[0].find_all("tr")
    out = []
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        out.append((tds[0].get_text(strip=True).rstrip("."),
                    tds[1].get_text(strip=True), tds[2].get_text(strip=True)))
    return out, "hh_new"


def _barnard1_h2_shocks(raw_dir):
    """Table 3 (Walawender+2005, AJ 130, 1795): "Positions of H2 Shocks in
    the Barnard 1 Region" -- the one Walawender table that is genuinely
    narrowband-H2-selected."""
    path = f"{raw_dir}/walawender/Walawender2005(AJ 130, 1795)_tbl3.mhtml"
    soup = BeautifulSoup(_read_mhtml_table_text(path), "html.parser")
    rows = soup.find_all("tbody")[0].find_all("tr")
    out = []
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        out.append((tds[0].get_text(strip=True).rstrip("."),
                    tds[1].get_text(strip=True), tds[2].get_text(strip=True)))
    return out, "h2_shocks"


def _parse_walawender(raw_dir):
    """Perseus (previously-known + newly-discovered HH objects, optical)
    and Barnard 1 (H2 shocks, narrowband) knot/object positions, combined
    with a per-row field label, table-of-origin label and an
    `H2_SELECTED` flag (True for Barnard 1's H2-shocks table only)."""
    for src, runbook_line in [
        (f"{raw_dir}/walawender/Walawender2005(AJ 129, 2308)_tbl2.html",
         "sesnaimpute.sky.download.h2_knot_surveys.build"),
        (f"{raw_dir}/walawender/Walawender2005(AJ 129, 2308)_tbl3.html",
         "sesnaimpute.sky.download.h2_knot_surveys.build"),
        (f"{raw_dir}/walawender/Walawender2005(AJ 130, 1795)_tbl3.mhtml",
         "sesnaimpute.sky.download.h2_knot_surveys.build"),
    ]:
        _require(src, runbook_line)

    known, known_origin = _perseus_previously_known(raw_dir)
    new, new_origin = _perseus_new(raw_dir)
    b1, b1_origin = _barnard1_h2_shocks(raw_dir)

    rows = []
    for designation, ra_s, dec_s in known:
        rows.append(("Perseus", known_origin, False, _clean_designation(designation), _ra_deg(ra_s), _dec_deg_unsigned_north(dec_s)))
    for designation, ra_s, dec_s in new:
        rows.append(("Perseus", new_origin, False, _clean_designation(designation), _ra_deg(ra_s), _dec_deg_unsigned_north(dec_s)))
    for designation, ra_s, dec_s in b1:
        rows.append(("Barnard 1", b1_origin, True, _clean_designation(designation), _ra_deg(ra_s), _dec_deg_unsigned_north(dec_s)))

    fields, origins, h2_selected, designations, ras, decs = zip(*rows)
    return dict(
        FIELD=np.array(fields, dtype="S16"),
        TABLE_ORIGIN=np.array(origins, dtype="S16"),
        H2_SELECTED=np.array(h2_selected, dtype=bool),
        DESIGNATION=np.array(designations, dtype="S16"),
        RA_DEG=np.array(ras, dtype=np.float64),
        DEC_DEG=np.array(decs, dtype=np.float64),
    )


# --------------------------------------------------------------- UWISH2

def _parse_uwish2(raw_dir):
    """The full 33,200-feature UWISH2 catalogue (Froebrich+2015 Table D1,
    transcribed): position, area, radius, median and maximum surface
    brightness (each knot's aperture-averaged and peak H2 1-0 S(1)
    brightness, `Fmedsb`/`Fmaxsb`), total flux, and the jet/PN/SNR/unknown
    classification code -- only `Class == 'j'` (jet) knots are the driven
    outflow population SPEC_PRIORS.md section 7 selects on."""
    path = f"{raw_dir}/uwish2/tabled1_transcribed.hdf5"
    _require(path, "sesnaimpute.sky.download.h2_knot_surveys.build")
    with h5py.File(path, "r") as f:
        ra = f["RAdeg"][()].astype(np.float64)
        dec = f["DEdeg"][()].astype(np.float64)
        area = f["Area"][()].astype(np.float64)
        rad = f["Rad"][()].astype(np.float64)
        fmedsb = f["Fmedsb"][()].astype(np.float64)
        fmaxsb = f["Fmaxsb"][()].astype(np.float64)
        ftot = f["Ftot"][()].astype(np.float64)
        cls = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in f["Class"][()]])
        uwish2_id = f["UWISH2"][()]
        group = f["Group"][()]
    return {
        "UWISH2_ID": np.asarray(uwish2_id, dtype="S32"),
        "RA_DEG": ra,
        "DEC_DEG": dec,
        "AREA_ARCSEC2": area,
        "RADIUS_ARCSEC": rad,
        "SURFACE_BRIGHTNESS_MEDIAN_1E-19_W_M2_ARCSEC2": fmedsb,
        "SURFACE_BRIGHTNESS_MAX_1E-19_W_M2_ARCSEC2": fmaxsb,
        "TOTAL_FLUX_1E-19_W_M2": ftot,
        "CLASS": np.asarray([c.encode() for c in cls], dtype="S1"),
        "JET_CLASS": (cls == "j"),
        "GROUP": np.asarray(group, dtype="S32"),
    }


# ----------------------------------------------------------------- writers

_UNITS = {
    "SIZE_ARCSEC2": "arcsec^2",
    "FLUX_2P12_MJY": "mJy", "FLUX_ERR_2P12_MJY": "mJy",
    "FLUX_I1_MJY": "mJy", "FLUX_ERR_I1_MJY": "mJy",
    "FLUX_I2_MJY": "mJy", "FLUX_ERR_I2_MJY": "mJy",
    "FLUX_I3_MJY": "mJy", "FLUX_ERR_I3_MJY": "mJy",
    "FLUX_I4_MJY": "mJy", "FLUX_ERR_I4_MJY": "mJy",
    "LUMINOSITY_IRAC_1E-2LSUN": "1e-2 Lsun", "LUMINOSITY_2P12_1E-2LSUN": "1e-2 Lsun",
    "TANGENTIAL_VELOCITY_KM_S": "km/s", "POSITION_ANGLE_DEG": "deg",
    "RA_DEG": "deg", "DEC_DEG": "deg",
    "AREA_ARCSEC2": "arcsec^2", "RADIUS_ARCSEC": "arcsec",
    "SURFACE_BRIGHTNESS_MEDIAN_1E-19_W_M2_ARCSEC2": "1e-19 W m^-2 arcsec^-2",
    "SURFACE_BRIGHTNESS_MAX_1E-19_W_M2_ARCSEC2": "1e-19 W m^-2 arcsec^-2",
    "TOTAL_FLUX_1E-19_W_M2": "1e-19 W m^-2",
}


def _write_survey(config, survey, data):
    out_path = product_path(config, "sky/derived", "knots", survey, "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        for name, arr in data.items():
            dset = f.create_dataset(name, data=arr)
            if name in _UNITS:
                dset.attrs["UNIT"] = _UNITS[name]
    return out_path


def _giannini_colours(giannini):
    """`log10(F_band / F_2.12)` per IRAC band, over knots with a real
    (non-upper-limit) detection in both the 2.12 um line and that band;
    median and 16-84% spread per band, plus N."""
    f212 = giannini["FLUX_2P12_MJY"]
    has_212 = np.isfinite(f212)
    out = {}
    for band_key in _GIANNINI_IRAC_BANDS:
        flux = giannini[f"FLUX_{band_key}_MJY"]
        is_limit = giannini[f"UPPER_LIMIT_{band_key}"]
        usable = has_212 & np.isfinite(flux) & ~is_limit
        ratio = np.log10(flux[usable] / f212[usable])
        out[band_key] = dict(
            RATIO=ratio.astype(np.float64),
            N=int(usable.sum()),
            MEDIAN=float(np.median(ratio)) if ratio.size else np.nan,
            P16=float(np.percentile(ratio, 16)) if ratio.size else np.nan,
            P84=float(np.percentile(ratio, 84)) if ratio.size else np.nan,
        )
    return out


def _write_colours(config, colours):
    out_path = product_path(config, "sky/derived", "knots", "colours", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.attrs["primary_sample"] = "Giannini et al. 2013, ApJ 767, 147, knots with a 2.12um detection"
        f.attrs["ratio_definition"] = "log10(F_IRAC_band / F_2.12um), both flux densities in mJy, no unit conversion needed"
        for band_key, stats in colours.items():
            group = f.create_group(band_key)
            group.create_dataset("LOG10_RATIO", data=stats["RATIO"])
            group.attrs["N"] = stats["N"]
            group.attrs["MEDIAN_LOG10_RATIO"] = stats["MEDIAN"]
            group.attrs["P16_LOG10_RATIO"] = stats["P16"]
            group.attrs["P84_LOG10_RATIO"] = stats["P84"]
    return out_path


def build(config, regions=None):
    """Writes `giannini2013`, `davis2009`, `walawender2005`, `uwish2` and
    `colours` -- five survey-level products. `regions` is accepted for
    interface uniformity and ignored: every product here is a whole-survey
    table, SPEC_PRIORS.md section 7's `eps_ext`/`eps_s` do not vary the
    knot catalogue itself by region.
    """
    with progress_module.Stage("sky.derived.knots") as st:
        raw_dir = _raw_dir(config)
        giannini = _parse_giannini(raw_dir)
        st.tick(1, 4, "surveys")
        davis = _parse_davis(raw_dir)
        st.tick(2, 4, "surveys")
        walawender = _parse_walawender(raw_dir)
        st.tick(3, 4, "surveys")
        uwish2 = _parse_uwish2(raw_dir)
        st.tick(4, 4, "surveys")

        _write_survey(config, "giannini2013", giannini)
        _write_survey(config, "davis2009", davis)
        _write_survey(config, "walawender2005", walawender)
        _write_survey(config, "uwish2", uwish2)

        colours = _giannini_colours(giannini)
        _write_colours(config, colours)

        print(f"sky.derived.knots build: giannini2013 {len(giannini['RA_DEG'])} knots, "
              f"davis2009 {len(davis['RA_DEG'])} knots, "
              f"walawender2005 {len(walawender['RA_DEG'])} knots "
              f"({int(walawender['H2_SELECTED'].sum())} H2-selected), "
              f"uwish2 {len(uwish2['RA_DEG'])} features "
              f"({int(uwish2['JET_CLASS'].sum())} jet-class)")
        for band_key, stats in colours.items():
            print(f"sky.derived.knots build: colours {band_key} N={stats['N']} "
                  f"median={stats['MEDIAN']:.3f} 16-84%=[{stats['P16']:.3f}, {stats['P84']:.3f}]")
        st.done(None, knots=len(giannini["RA_DEG"]) + len(davis["RA_DEG"]) + len(walawender["RA_DEG"]),
                features=len(uwish2["RA_DEG"]))


if __name__ == "__main__":
    run(build)
