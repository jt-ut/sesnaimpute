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
# zero-point flux density in Jy. sigma_cal_dex is the survey's own absolute-
# calibration systematic in log10 flux (SPEC_BMSTP_DRAFT.md sec 6.1), added
# in quadrature to a source's statistical log-flux error by every consumer
# of the fit's per-band variance (`fittp.likelihood.prepare`,
# `fittp.classify`); it is a property of the survey, not of the fit, so it
# lives here beside the band's other survey quantities.
#
# Sources, as cited in the old file:
#   2MASS J/H/Ks: Cohen, Wheaton & Megeath 2003, AJ 126, 1090.
#   IRAC I1-I4: Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook.
#   MIPS M1: Rieke et al. 2008, ApJ 135, 2245.
#
# sigma_cal_dex sources (distinct from the zero-point sources above):
#   2MASS J/H/Ks: Skrutskie et al. 2006, AJ 131, 1163; 0.010 dex.
#   IRAC I1-I4: Reach et al. 2005, astro-ph/0507139; 0.013 dex.
#   MIPS M1: Engelbracht et al. 2007, PASP 119, 994; 0.017 dex.
@dataclass(frozen=True)
class Band:
    key: str
    survey: str
    wvl_um: float
    vega_zero_point_jy: float
    source: str
    sigma_cal_dex: float
    sigma_cal_source: str


BANDS = (
    Band("J", "2MASS", 1.235, 1594.0, "Cohen, Wheaton & Megeath 2003, AJ 126, 1090",
         0.010, "Skrutskie et al. 2006, AJ 131, 1163"),
    Band("H", "2MASS", 1.662, 1024.0, "Cohen, Wheaton & Megeath 2003, AJ 126, 1090",
         0.010, "Skrutskie et al. 2006, AJ 131, 1163"),
    Band("Ks", "2MASS", 2.159, 666.7, "Cohen, Wheaton & Megeath 2003, AJ 126, 1090",
         0.010, "Skrutskie et al. 2006, AJ 131, 1163"),
    Band("I1", "IRAC", 3.550, 280.9, "Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook",
         0.013, "Reach et al. 2005, astro-ph/0507139"),
    Band("I2", "IRAC", 4.493, 179.7, "Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook",
         0.013, "Reach et al. 2005, astro-ph/0507139"),
    Band("I3", "IRAC", 5.731, 115.0, "Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook",
         0.013, "Reach et al. 2005, astro-ph/0507139"),
    Band("I4", "IRAC", 7.872, 64.13, "Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument Handbook",
         0.013, "Reach et al. 2005, astro-ph/0507139"),
    Band("M1", "MIPS", 23.68, 7.17, "Rieke et al. 2008, ApJ 135, 2245",
         0.017, "Engelbracht et al. 2007, PASP 119, 994"),
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
    # Placed between CI and CII (evolutionary position, not appended): a
    # flat-spectrum source's own spectral index sits between Class I's and
    # Class II's, and `SUBCLASSES_OF`/every register's `/subclass_prob`
    # column order is read by name everywhere downstream (`fittp.sweep`'s
    # `subclass_order`, `fittp.classify`'s `SUBCLASSES` attribute), never
    # by a hardcoded position -- verified by grep before choosing this
    # position (WP-REG-1).
    ModelSubclass("YSO", "FLAT", "Flat-spectrum: -0.3 <= alpha <= 0.3 (Greene et al. 1994), "
                   "relabelled from Stage/TD; grouped with the protostars in the census "
                   "(Dunham et al. 2015 count Class 0+I+Flat together)"),
    ModelSubclass("YSO", "CII", "Stage II: M_env < 0.1 Msun, disc present"),
    ModelSubclass("YSO", "CIII", "Stage III: M_env < 0.1 Msun, no disc; not a bare photosphere"),
    ModelSubclass("YSO", "TD", "Transition disc: Stage II/III carved by the Gutermuth 24um criterion"),
    ModelSubclass("H2S", "J", "J-type: discontinuous front; neutrals decelerate abruptly"),
    ModelSubclass("H2S", "C", "C-type: magnetically cushioned; ion-neutral drag decelerates the flow"),
    ModelSubclass("H2S", "Cs", "C-type solution containing a sonic point (C*)"),
    ModelSubclass("H2S", "CJ", "C-type structure with an embedded J-type discontinuity"),
)
SUBCLASSES_OF = {c.code: tuple(s.name for s in CLASSMAP if s.cls == c.code) for c in CLASSES}

#: class code -> the library register key (`sed_models/registers/<key>_
#: register.hdf5`). GAL/H2S transcribed from the old `prior.callable.
#: _LIBRARY_KEY`; STAR/PAHC/AGB/YSO from the register file names their own
#: modules already read (`prior.field_stars`/`star_population`: `sps_
#: register.hdf5`, `pahc_register.hdf5`; the on-disk `agb_register.hdf5`,
#: `yso_register.hdf5`, the latter a pooled register over YSO's five
#: sub-grids). Shared by `fit.sweep` (the fit's own model grid) and
#: `fit.terms` (the Gaia term's per-class register reads).
CLASS_REGISTER = {
    "STAR": "sps", "AGB": "agb", "PAHC": "pahc",
    "GAL": "galz", "YSO": "yso", "H2S": "h2shock",
}


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
