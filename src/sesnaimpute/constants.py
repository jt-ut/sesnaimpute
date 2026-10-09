"""Physical and astronomical constants, and nothing else.

Empty until a module needs a value. Add a constant here only when a module
computes with it, each with a one-line citation and its band (if band-
specific), never speculatively.

Pattern:

    # PLANCK_H = 6.62607015e-27       # erg s (CODATA 2018)
    # VEGA_ZERO_POINT_JY = {"J": 1594.0}  # Cohen, Wheaton & Megeath 2003, AJ 126, 1090
"""

import numpy as np
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

#: SPEC_BMSTP_DRAFT.md section 6.1 -- the absolute-calibration systematic
#: per band, dex, added in quadrature to the statistical log-flux error by
#: every consumer of the fit's per-band variance (`fittp.likelihood.prepare`,
#: `fittp.classify`). The same numbers as `definitions.BANDS[*]
#: .sigma_cal_dex`, in `definitions.BANDS`' own order (2MASS J, H, Ks;
#: IRAC I1-I4; MIPS M1). Sources (per band, as in `definitions.BANDS`):
#: 2MASS J/H/Ks, Skrutskie et al. 2006, AJ 131, 1163 (0.010 dex); IRAC
#: I1-I4, Reach et al. 2005, astro-ph/0507139 (0.013 dex); MIPS M1,
#: Engelbracht et al. 2007, PASP 119, 994 (0.017 dex).
SIGMA_CAL_DEX = np.array([b.sigma_cal_dex for b in definitions.BANDS], dtype=np.float64)

#: WP-PRIOR-8 -- the curated SED-model library's own sampling scale per
#: band, dex: twice `SIGMA_CAL_DEX`, the same whitening the curation's own
#: template resampling (`sed_models_register`) uses, so a kept library's
#: template spacing and `fittp.library_resolution`'s `sigma_lib,L` read it
#: in one unit by construction (`fittp.likelihood.prepare`'s own quadrature
#: sum, section 6.1).
#:
#: Derivation. A sum over templates spaced `d` apart in a band reproduces
#: that band's own Gaussian evidence integral (width `sigma`, a source's
#: per-band uncertainty) to relative error `2 exp(-2 pi^2 sigma^2 / d^2)`
#: (the quadrature-resolution bound for locally uniform node spacing).
#: Measured at three spacings:
#:
#:   d = 1 sigma: 5.4e-09
#:   d = 2 sigma: 1.44 %
#:   d = 3 sigma: 22.3 %
#:
#: The bound is this steep because it is an exponential in `-1/d^2`:
#: `d = 2 sigma` (this constant, with `sigma = SIGMA_CAL_DEX`) sits at the
#: EDGE of the acceptable region, not in the middle of it, and even the
#: 1.44 % figure applies only to the best-measured sources -- those sitting
#: at the calibration floor `sigma`; a typical detection's own larger
#: `sigma` makes the exponent more negative and the bound smaller still.
#: No later proposal to coarsen the library spacing further should be
#: chosen by analogy to this one without re-deriving the bound at the new
#: `d`: the knee between 1.44 % and 22.3 % is one more sigma away.
LIBRARY_SAMPLING_SIGMA_LOG_DEX = 2.0 * SIGMA_CAL_DEX
