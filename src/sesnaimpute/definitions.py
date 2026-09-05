"""What this project defines, as against what nature defines.

Physical constants live in `constants.py`. This module holds the project's
own bookkeeping vocabulary: the photometric bands SESNA measures in, the
classes and subclasses the fitter emits, and the six storage granules
(`IMPLEMENTATION.md` section 1).
"""

from dataclasses import dataclass


# --- (a) Bands ---
#
# The eight SESNA bands, transcribed from sesnacomplete.constants.BANDS.
# wvl_um is the precise in-flight-calibrated effective wavelength (matches
# the old BANDS[...].wvl_effective_um). vega_zero_point_jy is the Vega-system
# zero-point flux density in Jy.
#
# Sources, as cited in the old file:
#   2MASS J/H/Ks: Cohen, Wheaton & Megeath 2003, AJ 126, 1090.
#   IRAC I1-I4: Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook.
#   MIPS M1: Rieke et al. 2008, ApJ 135, 2245.
@dataclass(frozen=True)
class Band:
    key: str
    survey: str
    wvl_um: float
    vega_zero_point_jy: float
    source: str


BANDS = (
    Band("J", "2MASS", 1.235, 1594.0, "Cohen, Wheaton & Megeath 2003, AJ 126, 1090"),
    Band("H", "2MASS", 1.662, 1024.0, "Cohen, Wheaton & Megeath 2003, AJ 126, 1090"),
    Band("Ks", "2MASS", 2.159, 666.7, "Cohen, Wheaton & Megeath 2003, AJ 126, 1090"),
    Band("I1", "IRAC", 3.550, 280.9, "Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook"),
    Band("I2", "IRAC", 4.493, 179.7, "Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook"),
    Band("I3", "IRAC", 5.731, 115.0, "Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook"),
    Band("I4", "IRAC", 7.872, 64.13, "Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook"),
    Band("M1", "MIPS", 23.68, 7.17, "Rieke et al. 2008, ApJ 135, 2245"),
)
BANDS_BY_KEY = {b.key: b for b in BANDS}


# --- (b) Classes and subclasses ---
#
# Transcribed from sesnacomplete.constants.CLASSES and CLASSMAP. The old
# code spells the aromatic-contaminant class "PAHC" in CLASSES and "PAH" in
# some CLASSMAP comments; this project uses PAHC exactly, everywhere.
@dataclass(frozen=True)
class ModelClass:
    code: str
    description: str


@dataclass(frozen=True)
class ModelSubclass:
    cls: str
    name: str
    description: str


CLASSES = (
    ModelClass("STAR", "Bare stellar photosphere (no disk, envelope or PAH excess)"),
    ModelClass("AGB", "Dusty evolved star: AGB/RSG photosphere + circumstellar dust shell (GRAMS)"),
    ModelClass("PAHC", "Field star whose aperture contains extended PAH emission"),
    ModelClass("GAL", "Unresolved background galaxy (extragalactic contaminant)"),
    ModelClass("YSO", "Young stellar object (Richardson 2024 / Robitaille 2017 RT grid)"),
    ModelClass("H2S", "Shocked molecular hydrogen (H2) emission knot in a protostellar outflow"),
)
CLASSES_BY_CODE = {c.code: c for c in CLASSES}

CLASSMAP = (
    ModelSubclass("STAR", "O", "O-type: T_eff >= 32350 K (dwarf-sequence scale)"),
    ModelSubclass("STAR", "B", "B-type: 10200 <= T_eff < 32350 K (dwarf-sequence scale)"),
    ModelSubclass("STAR", "A", "A-type: 7310 <= T_eff < 10200 K (dwarf-sequence scale)"),
    ModelSubclass("STAR", "F", "F-type: 5990 <= T_eff < 7310 K (dwarf-sequence scale)"),
    ModelSubclass("STAR", "G", "G-type: 5325 <= T_eff < 5990 K (dwarf-sequence scale)"),
    ModelSubclass("STAR", "K", "K-type: 3890 <= T_eff < 5325 K (dwarf-sequence scale)"),
    ModelSubclass("STAR", "M", "M-type: 2325 <= T_eff < 3890 K (dwarf-sequence scale)"),
    ModelSubclass("STAR", "L", "L-type: 1312.5 <= T_eff < 2325 K (dwarf-sequence scale)"),
    ModelSubclass("STAR", "T", "T-type: T_eff < 1312.5 K (dwarf-sequence scale)"),
    ModelSubclass("AGB", "O", "Oxygen-rich: silicate dust, C/O < 1 (GRAMS O-rich, Sargent+2011)"),
    ModelSubclass("AGB", "C", "Carbon-rich: amorphous carbon + 10% SiC, C/O > 1 (GRAMS C-rich, Srinivasan+2011)"),
    ModelSubclass("PAHC", "NONE", "No subclass: one population, distinguished only by grid axes"),
    ModelSubclass("GAL", "AGN", "Power-law continuum: EW(6.2um) <= 0.2 um, inside Donley wedge"),
    ModelSubclass("GAL", "PAH", "Aromatic-dominated star formation: EW(6.2um) > 0.5 um"),
    ModelSubclass("GAL", "COMP", "Composite AGN + star formation: 0.2 < EW(6.2um) <= 0.5 um"),
    ModelSubclass("GAL", "PASS", "Passive: EW(6.2um) <= 0.2 um, outside the Donley AGN wedge"),
    ModelSubclass("YSO", "C0", "Stage 0: M_env > 0.1 Msun, T_star < 3000 K (deeply embedded)"),
    ModelSubclass("YSO", "CI", "Stage I: M_env > 0.1 Msun, T_star > 3000 K (embedded protostar)"),
    ModelSubclass("YSO", "CII", "Stage II: M_env < 0.1 Msun, disc present"),
    ModelSubclass("YSO", "CIII", "Stage III: M_env < 0.1 Msun, no disc; not a bare photosphere"),
    ModelSubclass("YSO", "TD", "Transition disc: Stage II/III carved by the Gutermuth 24um criterion"),
    ModelSubclass("H2S", "J", "J-type: discontinuous front; neutrals decelerate abruptly"),
    ModelSubclass("H2S", "C", "C-type: magnetically cushioned; ion-neutral drag decelerates the flow"),
    ModelSubclass("H2S", "Cs", "C-type solution containing a sonic point (C*)"),
    ModelSubclass("H2S", "CJ", "C-type structure with an embedded J-type discontinuity"),
)
SUBCLASSES_OF = {c.code: tuple(s.name for s in CLASSMAP if s.cls == c.code) for c in CLASSES}


# --- (c) Granules ---
#
# The six storage resolutions, from IMPLEMENTATION.md section 1.
GRANULES = ("source", "hpx512", "sightline", "tile", "region", "survey")
GRANULE_DEFINITIONS = {
    "source": "one SESNA catalogue row",
    "hpx512": "one nside-512 HEALPix pixel",
    "sightline": "one nside-256 HEALPix pixel that contains at least one catalogue source",
    "tile": "a group of hpx512 pixels, 0.5-0.7 degrees across",
    "region": "one of the 30 named star-forming regions",
    "survey": "the whole survey, one value for everything",
}
