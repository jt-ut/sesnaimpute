"""Physical and astronomical constants, and nothing else.

Empty until a module needs a value. Add a constant here only when a module
computes with it, each with a one-line citation and its band (if band-
specific), never speculatively.

Pattern:

    # PLANCK_H = 6.62607015e-27       # erg s (CODATA 2018)
    # VEGA_ZERO_POINT_JY = {"J": 1594.0}  # Cohen, Wheaton & Megeath 2003, AJ 126, 1090
"""

from sesnaimpute import definitions

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
