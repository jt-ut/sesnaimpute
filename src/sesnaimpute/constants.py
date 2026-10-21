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

#: C20 brief (briefs/round4_2026-10-20/C20_detection_correlation.md): the
#: share of comonotone detection in the mixture between SESNA's
#: independent band-detection model (today's closed form, `P(>=2 of 8) =
#: 1 - Prod_i(1-p_i) - Sum_i p_i Prod_{j!=i}(1-p_j)`) and a fully
#: comonotone model (one latent uniform draw per source, shared by every
#: band: `P(>=2 of 8) = p_(2)`, the SECOND-largest of the eight p_b) --
#: `bmstp.atlas.catalogued_probability`, `fittp.likelihood._ln_nondet`,
#: `fittp.prior_reader._cell_sum`'s non-detection branch.
#:
#: Measured (`population.detection_correlation.main`, 2026-10-20) against
#: UKIDSS GPS clean-stellar stars matched to SESNA within 1'' in clean,
#: deep-covered pixels: maximum likelihood over each star's own matched/
#: unmatched outcome wants LAMBDA > 1 in BOTH regions checked -- North
#: America Nebula (uncapped 1.32-1.36, depending on whether the fit keeps
#: Ks bins where the independent model already saturates near 1) and
#: Perseus (uncapped 0.96-1.59) -- so this constant is the capped value,
#: 1.0 exactly, not a fitted interior point. The brief anticipated this
#: outcome for the fit region alone ("UKIDSS clean stars favour isolated
#: stars"); seeing it in BOTH regions is a disclosure this constant's own
#: citation carries rather than hides: North America Nebula's own
#: matched/unmatched outcomes, restricted to the four Ks bins (15.5-17.5)
#: where the independent model is not already within 2% of its own
#: ceiling, still want LAMBDA = 1.32, and Perseus's own three such bins
#: (16.0-17.5) put the empirical catalogued fraction within 1% of p_(2)
#: in every one of them -- not the "no difference" a coarser, bin-
#: averaged ratio check (C18's test1b) suggested. A later re-measurement
#: that wants an interior value should replace this constant outright,
#: not add a second one: LAMBDA is ONE survey-wide number.
LAMBDA = 1.0

#: Combined19 (Bovy et al. 2016, mwdust; `sky.download.combined19`,
#: `sky.derived.profile`'s canonical far-field column, owner ruling
#: 2026-10-21) stores E(B-V) on the SFD scale; mwdust's own
#: `aebv('2MASS Ks', sf10=True)` (Schlafly & Finkbeiner 2011, Table 6,
#: R_V = 3.1) converts it to K-band extinction: `A_Ks = AK_PER_EBV_SFD_KS
#: * E(B-V)_SFD`. Verified against mwdust 1.8's own table
#: (`mwdust.util.extCurves.aebv`) to the value stored here.
AK_PER_EBV_SFD_KS = 0.3100
