"""Physical and astronomical constants, and nothing else.

Empty until a module needs a value. Add a constant here only when a module
computes with it, each with a one-line citation and its band (if band-
specific), never speculatively.

Pattern:

    # PLANCK_H = 6.62607015e-27       # erg s (CODATA 2018)
    # VEGA_ZERO_POINT_JY = {"J": 1594.0}  # Cohen, Wheaton & Megeath 2003, AJ 126, 1090
"""

import pandas as pd

from sesnaimpute import definitions

#: Gutermuth et al. 2009 (ApJS 184, 18) colour-cascade label vocabulary:
#: the eleven categories the cascade can emit, each row's SESNA-catalogue
#: `CLASS` code, population grouping and a display colour. `gutcolors`
#: (crisp.py, spec.py) reads this for its label set; it is his scheme's own
#: fixed vocabulary, not a project-invented one.
GUTERMUTH_LABELS = pd.DataFrame([
    {"code": 0,    "id": "DEEPLY_EMBEDDED", "description": "Deeply Embedded Protostar", "short": "Class 0",     "abbrev": "C0",   "pop": "YSO",          "pop_abbrev": "YSO",  "color": "#7A0C0C"},
    {"code": 1,    "id": "CLASS_I",        "description": "Class I Protostar",         "short": "Class I",     "abbrev": "CI",   "pop": "YSO",          "pop_abbrev": "YSO",  "color": "#B23A1A"},
    {"code": 2,    "id": "CLASS_II",       "description": "Class II Protostar",        "short": "Class II",    "abbrev": "CII",  "pop": "YSO",          "pop_abbrev": "YSO",  "color": "#E06D0F"},
    {"code": 3,    "id": "TRANSITION_DISK", "description": "Transition Disk",           "short": "Trans. Disk", "abbrev": "TD",   "pop": "YSO",          "pop_abbrev": "YSO",  "color": "#F4A300"},
    {"code": 9,    "id": "SHOCK_BLOB",     "description": "H2 Shock Blob",              "short": "H2 Shock",    "abbrev": "H2S",  "pop": "Contaminant",  "pop_abbrev": "CONT", "color": "#5E3AA8"},
    {"code": 39,   "id": "PAH_APERTURE",   "description": "PAH-Contaminated",          "short": "PAH Cont.",   "abbrev": "PAHC", "pop": "Contaminant",  "pop_abbrev": "CONT", "color": "#E0499C"},
    {"code": 19,   "id": "PAH_GALAXY",     "description": "PAH Emitter",                "short": "PAH Galaxy",  "abbrev": "PAHG", "pop": "Galaxy",       "pop_abbrev": "GAL",  "color": "#12966B"},
    {"code": 29,   "id": "AGN",            "description": "AGN",                        "short": "AGN",         "abbrev": "AGN",  "pop": "Galaxy",       "pop_abbrev": "GAL",  "color": "#0D93B8"},
    {"code": 49,   "id": "GENERIC_GALAXY", "description": "Generic Galaxy",             "short": "Galaxy",      "abbrev": "GGAL", "pop": "Galaxy",       "pop_abbrev": "GAL",  "color": "#4CAF32"},
    {"code": 99,   "id": "DISKLESS_STAR",  "description": "Diskless Star",              "short": "Star",        "abbrev": "DSTR", "pop": "Star",         "pop_abbrev": "STAR", "color": "#3B5BDB"},
    {"code": -100, "id": "UNCLASSIFIED",   "description": "Unclassified",               "short": "Unclass.",    "abbrev": "UNC",  "pop": "Unclassified", "pop_abbrev": "UC",   "color": "#B0B0B0"},
]).set_index("code")

#: Vega-system zero-point flux density per SESNA band, mJy. The same
#: numbers as `definitions.BANDS[*].vega_zero_point_jy`, in mJy rather
#: than Jy, keyed by band so a flux computation reads `VEGA_ZERO_POINT_MJY[key]`
#: directly. Sources (per band, as in `definitions.BANDS`): 2MASS J/H/Ks,
#: Cohen, Wheaton & Megeath 2003, AJ 126, 1090; IRAC I1-I4, Reach et al.
#: 2005 (astro-ph/0507139) / IRAC Instrument Handbook; MIPS M1, Rieke et
#: al. 2008, ApJ 135, 2245.
VEGA_ZERO_POINT_MJY = {b.key: b.vega_zero_point_jy * 1000.0 for b in definitions.BANDS}

#: Gaia DR3 G-band Vega-system zero-point flux density, mJy (3228.75 Jy).
#: Riello et al. 2021, A&A 649, A3 (Gaia EDR3 photometric calibration,
#: carried forward unchanged into DR3).
GAIA_G_VEGA_ZP_MJY = 3_228_750.0
