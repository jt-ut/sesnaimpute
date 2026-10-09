"""Package-wide constants for sesnacomplete.

Scope rule -- FOUR kinds of thing belong here:

  1. Genuine physical / unit-conversion facts, true regardless of
     methodology (speed of light, solar luminosity, mJy-to-cgs).
  2. Photometric band definitions shared across surveys/subpackages.
  3. Shared STRUCTURAL PLUGS -- values that are not physical measurements
     but must be identical everywhere they appear, because the model
     libraries are read against one another (REFERENCE_DISTANCE_CM,
     POINT_SOURCE_APERTURE_AU).
  4. Shared literature / grid CONVENTIONS used by two or more libraries
     (GUTERMUTH_LABELS, APERTURE_GRID_AU, REGIONS).

What does NOT belong here: a constant encoding a modeling choice tied to
one particular method -- e.g. the Chen et al. 1995 T_bol classification
cutpoints, or an aperture window chosen for one classification scheme.
Those stay local to the module that uses them; yso_labeling.py's own
CONSTANTS block is the example to follow.

Categories 3 and 4 were added after an audit found the file already held
GUTERMUTH_LABELS (a literature label map, category 4) while the stated
rule admitted only 1 and 2 -- i.e. the rule described less than the file
already contained. Widening it is a correction, not a loosening.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- Fundamental constants (cgs) ---
PLANCK_H = 6.62607015e-27       # erg s
BOLTZMANN_K = 1.380649e-16      # erg / K
SPEED_OF_LIGHT_CM_S = 2.99792458e10   # cm / s
GRAVITATIONAL_CONSTANT_CGS = 6.67430e-8   # cm^3 / g / s^2  (CODATA 2018)
HYDROGEN_MASS_G = 1.67353e-24             # g  (1.00782503 u, CODATA 2018)

# --- Convenience unit variant used throughout SED wavelength/frequency work ---
# (lambda in um, nu in Hz) -> nu = C_UM_S / lambda
C_UM_S = 2.99792458e14          # speed of light [um / s]

# --- Solar units ---
L_SUN_ERG_S = 3.828e33          # erg / s   (IAU nominal solar luminosity)
M_SUN_G = 1.98892e33            # g
R_SUN_CM = 6.957e10             # cm        (IAU nominal solar radius)

# --- Distance conversions ---
AU_CM = 1.495978707e13          # cm
PC_CM = 3.0856775814913673e18   # cm

# --- Flux unit conversions ---
MJY_TO_CGS = 1e-26              # mJy -> erg / s / cm^2 / Hz  (F_nu)

# --- Vega-system zero-point flux densities, mJy ---
# This project stores flux in mJy throughout (catalog FNU, convolved/*.fits
# TOTAL_FLUX, sedfitter VALUES), so these are converted to mJy here rather
# than left in the literature's native Jy -- see astro_utils.py's
# flux_to_mag/vega_color, which assume mJy on both flux and zero point.
# 2MASS J/H/Ks: Cohen, Wheaton & Megeath 2003, AJ 126, 1090 (Jy values
# 1594/1024/666.7 -- the same Jy values already cited in
# catalog/build_input.py's UB_2MASS_MJY comment).
# IRAC I1-I4: Reach et al. 2005 (astro-ph/0507139) / IRAC Instrument
# Handbook (Jy values 280.9/179.7/115.0/64.13).
# MIPS M1: Rieke et al. 2008, ApJ 135, 2245 (Jy value 7.17, quoted at an
# effective wavelength of 23.68um -- matches BANDS["M1"].wvl_effective_um).
VEGA_ZERO_POINT_MJY = {
    "J":  1_594_000.0,
    "H":  1_024_000.0,
    "Ks":   666_700.0,
    "I1":   280_900.0,
    "I2":   179_700.0,
    "I3":   115_000.0,
    "I4":    64_130.0,
    "M1":     7_170.0,
}

# --- Bolometric temperature (Myers & Ladd 1993) ---
# T_bol [K] = TBOL_K_PER_HZ * <nu> [Hz], where <nu> is a SED's flux-weighted
# mean frequency. The proportionality constant relating an SED's mean
# frequency to the temperature of a blackbody with that same mean frequency
# -- a fixed physical relation (like the other constants above), not a
# modeling choice, so it lives here rather than with any one classification
# scheme that happens to use it.
# NOTE: retained for sed_models_curate/archive/yso_soft_classify_archived.py, which
# imports it. No live consumer -- delete both together or neither.
TBOL_K_PER_HZ = 1.25e-11        # K / Hz


# --- Photometric bands ---
@dataclass(frozen=True)
class Band:
    """A single photometric band used across the SESNA archive.

    wvl_channel_um is the nominal/rounded wavelength the band is commonly
    named after (e.g. IRAC "3.6 micron", MIPS "24 micron") and is what gets
    used most often. wvl_effective_um is the precise in-flight-calibrated
    effective wavelength from the instrument handbooks. For 2MASS these two
    values are identical, since the isophotal wavelength *is* the nominal
    value. For IRAC/MIPS they differ slightly (e.g. IRAC I4: 8.0 vs
    7.872 um; MIPS M1: 24.0 vs 23.68 um).

    aperture_arcsec is the fixed photometric aperture RADIUS SESNA's
    Gutermuth-lineage pipeline used for this band -- a survey/instrument
    convention (4.0" for 2MASS, 2.4" for IRAC, 7.6" for MIPS), not a
    per-fit tunable. Radius, not diameter: sedfitter's Fitter treats the
    apertures array as radii, and Models.read converts it to a physical
    radius in AU as aperture_arcsec * d_pc.

    This value is the single authority for the aperture every model
    library is interpolated to (sedfitter Models.read) AND for any
    survey-matched hybrid SED built during curation. The two must agree;
    see sedfitter_behavior.md.
    """

    name: str
    survey: str
    wvl_channel_um: float
    wvl_effective_um: float
    aperture_arcsec: float


BANDS = {
    "J":  Band("J",  "2MASS", 1.235, 1.235, 4.0),
    "H":  Band("H",  "2MASS", 1.662, 1.662, 4.0),
    "Ks": Band("Ks", "2MASS", 2.159, 2.159, 4.0),
    "I1": Band("I1", "IRAC",  3.6,   3.550, 2.4),
    "I2": Band("I2", "IRAC",  4.5,   4.493, 2.4),
    "I3": Band("I3", "IRAC",  5.8,   5.731, 2.4),
    "I4": Band("I4", "IRAC",  8.0,   7.872, 2.4),
    "M1": Band("M1", "MIPS",  24.0,  23.68, 7.6),
}

# The project-wide convolved/{band}.fits naming, in BANDS order. Every
# library emits exactly these eight filenames.
CONVOLVED_BAND_NAMES = tuple(BANDS)

# --- Survey absolute-calibration precision, dex of log10 flux ---
# The sampling scale for every model library (see sed_models_curate: one
# noise length per band is the number below, and the whitened SED-space
# distance uses all eight).
#
# These are the surveys' ABSOLUTE-CALIBRATION systematics -- a property of
# the instruments and their calibration programmes, published by the survey
# teams, not a quantity measured from our own catalogue. They are therefore
# literature constants and live here with their citations, so that every
# procedure needing them reads one table rather than re-deriving it.
#
# They are the survey's precision FLOOR: they exceed the statistical error
# of the best-measured sources in every band, so they set the finest
# distinction any source in the survey can support. Sampling a library at
# this scale guarantees the library is never the limiting factor for the
# best data. Note this is deliberately FINER than the median detection's
# statistical error (0.0168-0.0350 dex, measured in sed_models_register's
# noise.py): a typical source cannot exploit the full resolution, but a
# bright one can, and the library must not be what stops it.
#
#   2MASS J/H/Ks  0.010 dex (2.3%)  Skrutskie et al. 2006, AJ 131, 1163
#   IRAC I1-I4    0.013 dex (3.0%)  Reach et al. 2005, PASP 117, 978
#   MIPS 24um     0.017 dex (4.0%)  Engelbracht et al. 2007, PASP 119, 994
#
# The downstream pipeline carries the same eight numbers in its own
# constants table. The two are reported as equal by each library's curation
# report; that agreement is documentary, NOT enforced by a gate.
SURVEY_CALIB_SIGMA_LOG_DEX = {
    "J":  0.010,
    "H":  0.010,
    "Ks": 0.010,
    "I1": 0.013,
    "I2": 0.013,
    "I3": 0.013,
    "I4": 0.013,
    "M1": 0.017,
}

#: The same eight values as an array in CONVOLVED_BAND_NAMES order -- the
#: form sed_models_register.density.build_quotient_space() takes.
SURVEY_CALIB_SIGMA_LOG_VECTOR = tuple(
    SURVEY_CALIB_SIGMA_LOG_DEX[_b] for _b in CONVOLVED_BAND_NAMES)


# --- The model-library sampling scale ---
# TWICE the calibration floor above, per band. This factor is recorded
# here WITH ITS DERIVATION so that it is a derived quantity rather than a
# tuning knob, and so nobody re-tunes it.
#
# DERIVATION. A class's evidence is a sum over templates of
# exp(-chi2/2). A grid of spacing d sampling that Gaussian, against a
# source of photometric error sigma, reproduces the integral to a
# relative error of 2*exp(-2*pi^2*sigma^2/d^2) (Poisson summation; the
# trapezoid rule superconverges on a Gaussian). Evaluated at d = 2*sigma:
#
#   2*exp(-2*pi^2/4) = 0.0144, i.e. 1.4% of the evidence, 0.015 nats
#
# and that applies only to the BEST-measured sources -- roughly 1% of the
# catalogue, the ones whose errors reach the calibration floor. For a
# typical detection (statistical error 0.017-0.035 dex, so d is about one
# sigma there) the same expression is below 1e-4. The library-resolution
# term in the fit covers the residual for the best sources by design.
#
# So templates closer together than this are not distinguished by ANY
# source the survey holds, and sampling finer than it buys resolution no
# datum can use. Sampling at the bare floor instead of twice it was tried
# and rejected: it implies a YSO library of order 800,000 templates, which
# is not a believable count of distinguishable SED shapes.
LIBRARY_SAMPLING_SIGMA_FACTOR = 2.0

LIBRARY_SAMPLING_SIGMA_LOG_DEX = {
    _b: LIBRARY_SAMPLING_SIGMA_FACTOR * _s
    for _b, _s in SURVEY_CALIB_SIGMA_LOG_DEX.items()
}

#: The sampling scale as an array in CONVOLVED_BAND_NAMES order -- the
#: form sed_models_register.density.build_quotient_space() takes, and the
#: vector every library's sampling uses. 2MASS 0.020, IRAC 0.026,
#: MIPS24 0.034 dex; projected through the 5-D quotient space this gives
#: sigma_eff = 0.055852 dex.
LIBRARY_SAMPLING_SIGMA_LOG_VECTOR = tuple(
    LIBRARY_SAMPLING_SIGMA_LOG_DEX[_b] for _b in CONVOLVED_BAND_NAMES)


# The 2008 Kurucz distribution's OWN convolved/ spelling, which differs from
# BANDS only for the three 2MASS bands. This is a SOURCE-SIDE fact, used
# solely by sps_curate to locate files in the upstream tree: SPS is now
# written out under CONVOLVED_BAND_NAMES like every other library, so
# nothing downstream of curation ever sees "2J" again.
#
# It used to be the OUTPUT naming as well, which forced the exception to be
# carried in three separate places -- a second band tuple here, a second one
# in model_io.CONTRACT_PROFILES, and yso_dedup.SPS_FILTERS. Standardising the
# output deleted all three. What remains is this mapping, which describes the
# upstream tree rather than anything this project ships.
SPS_SOURCE_TO_STANDARD_BAND = {
    "2J": "J", "2H": "H", "2K": "Ks",
    "I1": "I1", "I2": "I2", "I3": "I3", "I4": "I4", "M1": "M1",
}


# --- Gutermuth (2009) source classification scheme ---
# Lookup table for the SESNA catalog's CLASS column, shared across
# subpackages (sed_models_curate' galaxy/H2-shock/PAH-C tracks, the catalog
# module, and any plotting that compares against his scheme) -- the same
# reasoning BANDS above is kept here for. Indexed by `code` (the integer
# CLASS value) so downstream color/label matching against a real catalog
# column is a single vectorized call, e.g.
# `catalog_df["CLASS"].map(GUTERMUTH_LABELS["color"])`.
#
# Columns: id (the machine identifier, see below), description
# (Gutermuth's full label), short/abbrev (two levels
# of plot-label compactness), pop/pop_abbrev (the coarse super-group this
# code belongs to -- YSO / Contaminant / Galaxy / Star / Unclassified --
# deliberately a different, coarser concept than `code`/`description`
# -Gutermuth's fine-grained scheme- or the project's own 5-way
# model-fitting classes -YSO/SPS/GAL/PAH-C/H2 in the contaminant design
# spec; see sed_models_curate/docs/contaminant_overview_spec.md), color
# (hex, grouped into one hue family per population -- warm reds/oranges
# for YSO, purples/magentas for Contaminant, greens/teals for Galaxy,
# blue for Star, neutral gray for Unclassified -- so plots are visually
# groupable by population at a glance).
#
# `id` is the MACHINE identifier and the only column safe to key on. The
# others are presentation text and exist to be edited: retitling a legend
# must not break a lookup. It is also what gut_colors' cascade emits, and
# what its schedule table names when it assigns a label, so this frame --
# not a dict inside that subpackage -- is the single authority on the
# label vocabulary. Row ORDER is load-bearing for the same reason: the
# cascade stores labels as a dense integer index into this frame, because
# `code` itself is sparse and negative and cannot serve as one. Adding a
# row is fine; reordering existing rows invalidates any saved label array
# not accompanied by its own copy of the ids.
#
# NOTE on code=49 ("Generic Galaxy"): this code does NOT appear in the
# real SESNA catalog's CLASS column (confirmed against actual data --
# observed values are -100, 0, 1, 2, 3, 9, 19, 29, 39, 99 only). It is a
# project-invented placeholder for the GAL model library's many galaxy
# sub-types (spirals, ellipticals, etc.) that Gutermuth's scheme never
# labels explicitly, kept as a separate tranche within the Galaxy
# population for labeling/plotting the GAL library's provenance -- never
# expect a real source's CLASS to equal 49.
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



# --- The project's own class / subclass vocabulary ---
#
# THE CANONICAL RECORD of every CLASS and SUBCLASS name that can appear
# anywhere in this project. One authority, so that consumers validate
# against it instead of each maintaining a local vocabulary plus an alias
# table to reconcile with everyone else's.
#
# NOT the same thing as GUTERMUTH_LABELS, and the two must not be merged.
# GUTERMUTH_LABELS is HIS scheme -- the categories his colour cascade can
# emit, used to colour SESNA sources by their catalogue CLASS code.
# CLASSMAP is OURS -- what the model libraries contain, used to colour
# models and posterior results. They meet in exactly one place, the
# translation matrix T (posterior spec 5.3.1), and `gutermuth_nominal`
# below is the taxonomic correspondence used to VALIDATE T, not define it.
#
# The two groupings are also different partitions, which is easy to miss
# because they look alike: his `pop` "Contaminant" spans H2S AND PAHC,
# and his "Unclassified" has no CLASSMAP analogue (a model is always
# some class).
#
# PROVENANCE: this is a roll-up, not an authoring. Every description is
# transcribed verbatim from the CLASS_LEGEND / SUBCLASS_LEGEND of the
# library that ships it -- the curators defined these, CLASSMAP only
# collects them after the fact. A library shipping a name absent here is
# a curation bug to fix at the source, never something to alias around.
#
# ORDER IS LOAD-BEARING: the tuple order fixes T's subclass axis
# (one column per entry; len(CLASSMAP), never a hard-coded count) and the column order of any pooled subclass table. Do not sort.
@dataclass(frozen=True)
class ModelClass:
    """One of the five model-fitting classes."""
    name: str
    description: str
    color: str


@dataclass(frozen=True)
class ModelSubclass:
    """One subclass, and its nominal counterpart in Gutermuth's scheme.

    `gutermuth_nominal` is the category his scheme would nominally call
    this population, or None where his scheme has no rule for it. It is
    NOT the operational mapping: the real correspondence is many-to-one
    with gaps in both directions, and depends on brightness and detection
    pattern as well as colour -- his galaxy gates carry faintness
    requirements ([4.5] > 11.5 for PAH galaxies, > 13.5 for AGN), and
    DEEPLY_EMBEDDED is defined by IRAC dropout rather than by colour at
    all. That correspondence is T, measured by simulation. Using this
    column as T would silently reintroduce every problem T exists to
    solve.
    """
    cls: str
    name: str
    description: str
    color: str
    gutermuth_nominal: str = None


CLASSES = (
    ModelClass("YSO",  "Young stellar object (Richardson 2024 / Robitaille 2017 RT grid)", "#B23A1A"),
    ModelClass("STAR", "Bare stellar photosphere (no disk, envelope or PAH excess)",       "#3B5BDB"),
    ModelClass("GAL",  "Unresolved background galaxy (extragalactic contaminant)",         "#12966B"),
    ModelClass("H2S",  "Shocked molecular hydrogen (H2) emission knot in a protostellar outflow", "#5E3AA8"),
    ModelClass("PAHC", "Field star whose aperture contains extended PAH emission",         "#E0499C"),
    # Stage B / AGB-ext (agb_decisions.md B-10): dusty evolved-star
    # contaminant library from GRAMS. A new hue family (brown/dust-gold),
    # distinct from the warm reds/oranges of YSO, the blues of STAR, the
    # greens/teals of GAL, the purples of H2S and the magenta of PAHC --
    # chosen because the class itself is literally circumstellar dust.
    ModelClass("AGB",  "Dusty evolved star: AGB/RSG photosphere + circumstellar dust shell (GRAMS)", "#8A5A2B"),
)

# Palette: one hue family per class, matching GUTERMUTH_LABELS' own
# scheme (warm reds YSO, blue STAR, green/teal GAL, purple H2S, magenta
# PAHC), lightness varying across subclasses. Where a subclass has an
# exact Gutermuth counterpart its colour is that row's colour verbatim,
# so a predicted-vs-Gutermuth panel reads as one visual system.
#
# FLAGGED CHOICE (STAR): the spectral sequence is a blue ramp rather than
# the conventional blue-to-red, because blue-to-red would collide with
# the YSO family and make class membership unreadable at a glance. Family
# consistency was preferred over spectral convention; reverse it here if
# the plots argue otherwise.
CLASSMAP = (
    # YSO -- stage sequence, embedded (dark) to diskless (light).
    ModelSubclass("YSO", "C0",   "Stage 0: M_env > 0.1 Msun, T_star < 3000 K (deeply embedded)",         "#7A0C0C", "DEEPLY_EMBEDDED"),
    ModelSubclass("YSO", "CI",   "Stage I: M_env > 0.1 Msun, T_star > 3000 K (embedded protostar)",      "#B23A1A", "CLASS_I"),
    # FLAT sits here, between CI and CII, matching the fitter package's
    # own evolutionary order (C0, CI, FLAT, CII, CIII, TD) -- the owner's
    # library spec (2026-10-08) adds this sixth YSO subclass, relabelled
    # from Spectral Index (Greene, Wilking, Andre, Young & Lada 1994,
    # ApJ 434, 614: -0.3 <= alpha <= 0.3), overriding whatever Stage/TD a
    # row would otherwise carry -- the review's gap 2 (lib_2_YSO.md):
    # flat-spectrum templates existed in the grid all along but had no
    # subclass of their own, mislabelled CI/CII/TD by Stage argmax. No
    # Gutermuth counterpart (his four-category cascade has no flat-
    # spectrum rule), hence None. Position here is load-bearing for the
    # legend/SUBCLASS_PROB column order (module header: "ORDER IS LOAD-
    # BEARING"), not merely cosmetic -- nothing may index a subclass by
    # position, so moving it costs nothing downstream.
    ModelSubclass("YSO", "FLAT", "Flat-spectrum: -0.3 <= Spectral Index <= 0.3 (Greene et al. 1994)",  "#C9801A", None),
    ModelSubclass("YSO", "CII",  "Stage II: M_env < 0.1 Msun, disc present",                             "#E06D0F", "CLASS_II"),
    ModelSubclass("YSO", "CIII", "Stage III: M_env < 0.1 Msun, no disc; not a bare photosphere",         "#D9A05B", None),
    ModelSubclass("YSO", "TD",   "Transition disc: Stage II/III carved by the Gutermuth 24um criterion", "#F4A300", "TRANSITION_DISK"),
    # STAR -- spectral sequence, hot (dark) to cool (light). All seven
    # share DISKLESS_STAR: his cascade cannot resolve spectral type.
    ModelSubclass("STAR", "O", "O-type: T_eff >= 32350 K (dwarf-sequence scale)",           "#16276B", "DISKLESS_STAR"),
    ModelSubclass("STAR", "B", "B-type: 10200 <= T_eff < 32350 K (dwarf-sequence scale)",   "#23388F", "DISKLESS_STAR"),
    ModelSubclass("STAR", "A", "A-type: 7310 <= T_eff < 10200 K (dwarf-sequence scale)",    "#3B5BDB", "DISKLESS_STAR"),
    ModelSubclass("STAR", "F", "F-type: 5990 <= T_eff < 7310 K (dwarf-sequence scale)",     "#5C7BE8", "DISKLESS_STAR"),
    ModelSubclass("STAR", "G", "G-type: 5325 <= T_eff < 5990 K (dwarf-sequence scale)",     "#7E9BF0", "DISKLESS_STAR"),
    ModelSubclass("STAR", "K", "K-type: 3890 <= T_eff < 5325 K (dwarf-sequence scale)",     "#A3BCF7", "DISKLESS_STAR"),
    # M's lower bound is now closed (M/L cut) now that the sequence extends
    # past the dwarf/photosphere floor into the BT-Settl CIFIST L/T range
    # (D-10 extension). L and T anchors and the M/L, L/T cuts are Mamajek's
    # "Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
    # v2022.04.16 (same source as the O-K cuts): M9V 2380 K, L0V 2270 K,
    # L9V 1370 K, T0V 1255 K -- midpoints 2325 K (M/L) and 1312.5 K (L/T).
    ModelSubclass("STAR", "M", "M-type: 2325 <= T_eff < 3890 K (dwarf-sequence scale)",     "#C8D7FB", "DISKLESS_STAR"),
    ModelSubclass("STAR", "L", "L-type: 1312.5 <= T_eff < 2325 K (dwarf-sequence scale)",   "#DEE9FD", "DISKLESS_STAR"),
    ModelSubclass("STAR", "T", "T-type: T_eff < 1312.5 K (dwarf-sequence scale)",           "#F0F6FE", "DISKLESS_STAR"),
    # GAL -- COMP and PASS have no Gutermuth rule; see the note below.
    ModelSubclass("GAL", "AGN",  "Power-law continuum: EW(6.2um) <= 0.2 um, inside Donley wedge", "#0D93B8", "AGN"),
    ModelSubclass("GAL", "PAH",  "Aromatic-dominated star formation: EW(6.2um) > 0.5 um",         "#12966B", "PAH_GALAXY"),
    ModelSubclass("GAL", "COMP", "Composite AGN + star formation: 0.2 < EW(6.2um) <= 0.5 um",     "#1FA894", None),
    ModelSubclass("GAL", "PASS", "Passive: EW(6.2um) <= 0.2 um, outside the Donley AGN wedge",    "#7FB88F", None),
    # H2S -- all four share SHOCK_BLOB: his cascade cannot resolve shock
    # topology. The C-family (C, Cs, CJ) are the H2-bright,
    # IRAC-contaminating population; J-type is brighter in ionic lines.
    ModelSubclass("H2S", "J",  "J-type: discontinuous front; neutrals decelerate abruptly",             "#3E2472", "SHOCK_BLOB"),
    ModelSubclass("H2S", "C",  "C-type: magnetically cushioned; ion-neutral drag decelerates the flow", "#5E3AA8", "SHOCK_BLOB"),
    ModelSubclass("H2S", "Cs", "C-type solution containing a sonic point (C*)",                         "#8560C7", "SHOCK_BLOB"),
    ModelSubclass("H2S", "CJ", "C-type structure with an embedded J-type discontinuity",                "#A98BDD", "SHOCK_BLOB"),
    # PAHC -- deliberately unpartitioned. "NONE" rather than "all": there
    # are no subclasses, not many.
    ModelSubclass("PAHC", "NONE", "No subclass: one population, distinguished only by grid axes", "#E0499C", "PAH_APERTURE"),
    # AGB -- chemistry (agb_decisions.md B-10/B-5). SUBCLASS 'O' does NOT
    # collide with STAR:'O' (O-type star): every consumer keys on the
    # (cls, name) pair (constants.CLASSMAP_DF is indexed on ["cls","name"];
    # see sed_models_curate/agb_curate.py's module docstring for the consumer
    # sweep this entry's addition triggered). gutermuth_nominal=None for
    # both: his cascade has no AGB category, and setting it to
    # DISKLESS_STAR (99) would be wrong -- a reddened AGB star lands in a
    # YSO class under his rules, which is exactly the contamination this
    # library exists to model. The translation matrix T needs a new AGB
    # row; that is flagged to the owner, not decided here (B-10).
    ModelSubclass("AGB", "O", "Oxygen-rich: silicate dust, C/O < 1 (GRAMS O-rich, Sargent+2011)",         "#5C3A1A", None),
    ModelSubclass("AGB", "C", "Carbon-rich: amorphous carbon + 10% SiC, C/O > 1 (GRAMS C-rich, Srinivasan+2011)", "#C99456", None),
)

# THREE SUBCLASSES CARRY gutermuth_nominal = None, and that is
# information rather than missing data: his scheme has no rule for any of
# them. CIII, because diskless YSOs have photospheric colours and his
# cuts cannot isolate them -- he carries four YSO categories and no
# Class III. GAL:COMP and GAL:PASS, because his galaxy rejection tests
# for PAH emission or an AGN power law, and a composite or passive
# galaxy presents neither strongly.
#
# GENERIC_GALAXY (code 49) is deliberately NOT used as the nominal for
# COMP/PASS, even though it was added to GUTERMUTH_LABELS in
# anticipation of them. The cascade can never assign it -- it is an
# always-zero column, test-checked in gut_colors -- so a nominal pointer
# there would set up a T-validation check that can never pass, and would
# invite a T row whose COLOR contribution is permanently pinned at the
# epsilon floor.


def _classmap_frame():
    """DataFrame view, for plotting and vectorised joins.

    Derived, never edited: the tuples above are the authority, so this
    cannot drift from them. Indexed by (cls, name); call
    `.reset_index()` for flat rows.
    """
    df = pd.DataFrame([vars(s) for s in CLASSMAP])
    return df.set_index(["cls", "name"])


CLASSMAP_DF = _classmap_frame()
CLASSES_BY_NAME = {c.name: c for c in CLASSES}
SUBCLASSES_OF = {c.name: tuple(s.name for s in CLASSMAP if s.cls == c.name)
                 for c in CLASSES}
SUBCLASS_INDEX = {(s.cls, s.name): i for i, s in enumerate(CLASSMAP)}


def _validate_classmap():
    """Run at import: a typo here is a silent join failure downstream."""
    known = set(GUTERMUTH_LABELS["id"])
    seen = set()
    for s in CLASSMAP:
        if s.cls not in CLASSES_BY_NAME:
            raise ValueError(f"CLASSMAP: {s.name!r} has unknown class {s.cls!r}")
        if (s.cls, s.name) in seen:
            raise ValueError(f"CLASSMAP: duplicate entry {(s.cls, s.name)!r}")
        seen.add((s.cls, s.name))
        if s.gutermuth_nominal is not None and s.gutermuth_nominal not in known:
            raise ValueError(
                f"CLASSMAP: {s.cls}:{s.name} names gutermuth_nominal "
                f"{s.gutermuth_nominal!r}, absent from GUTERMUTH_LABELS"
            )
    empty = [c.name for c in CLASSES if not SUBCLASSES_OF[c.name]]
    if empty:
        raise ValueError(f"CLASSMAP: classes with no subclasses: {empty}")


_validate_classmap()


# --- SESNA region distances ---
# Per-region distance, the single authority for anything in the project
# that needs one -- catalog/scripts/build_all_catalogs.py imports
# `range_kpc` from here (rather than keeping its own copy) as the
# `distance_range_kpc` metadata stamped into every catalog/build_input.py
# SEDFIT_INPUT.hdf5 file.
#
# d_r_pc/sigma_pc are the literature point estimate and its 1-sigma
# uncertainty, in pc. range_kpc = (d_r_pc - sigma_pc, d_r_pc + sigma_pc),
# pc converted to kpc -- the same range every SEDFIT_INPUT.hdf5 file
# already carries in its own DISTANCE_RANGE_KPC metadata table. Where the
# source table gave sigma as a quadrature sum of two components (e.g.
# "3 (+) 24") rather than a single number, sigma_pc here is
# sqrt(a**2 + b**2), already resolved to one number.
#
# This is a v2 revision supplied directly by the user, not independently
# re-derived in this codebase -- full citations, the superseded v1 table,
# and the reasoning behind every entry (including the quadrature-sum
# convention above) are in catalog/docs/distance_table.md. Treat `notes`
# below as a compressed pointer into that document's Notes column, not the
# full rationale; "seed" in a note refers to that document's v1 table.
#
@dataclass(frozen=True)
class Region:
    name: str
    d_r_pc: float
    sigma_pc: float
    range_kpc: tuple
    basis: str
    notes: str
    # The cloud's line-of-sight depth is NOT a field here any more: it is
    # a value derived from the dust3d wave's own data products, read live
    # via `dust3d.structure_depth_pc(region_name)` rather than
    # copied into a code constant, so a rebuilt derivation moves it
    # honestly instead of drifting from a stale literal.


REGIONS = {
    'AFGL 490': Region(
        'AFGL 490', 1060, 75, (0.985, 1.135), 'S+D',
        "REVISED 2026-08-21 by owner ruling R-78 (dispatch C5), was 800+-160 "
        "(an ASSIGNED flat 20%), basis 'O (chain)'. SOURCE: the median of the "
        "seven Gaia-era determinations within 1 deg of the footprint -- "
        "Prisinzano, Damiani, Sciortino et al. 2022, A&A 664, A175, table 3 "
        "cluster 530 (its Name column is EMPTY, N=510, 0.14 deg) 1134 pc; "
        "Mullen, Mast, Kounkel, Stassun, Roman-Lopes & Tan 2025, ApJ "
        "(arXiv:2508.09393), table 1 'Cluster 1', 469 stars, 0.20 deg, "
        "1095+-43 pc; Chen B.-Q., Huang, Yuan et al. 2020, MNRAS 493, 351, "
        "cloud ID 436 (142.002, +2.174), 0.26 deg, 1065.1+-25.1 pc; Hunt & "
        "Reffert 2023, A&A 673, A114 / 2024, A&A 686, A42, 'Theia_1713' "
        "(= CWNU_1131, OC_0262), N=114, 0.96 deg, 1059.7 pc, and 'HSC_1134', "
        "N=32, 0.64 deg, 988.0 pc; Chen et al. 2020 cloud ID 385, 0.91 deg, "
        "967.4 pc; and this project's own dust map peak at 944.9 pc. Their "
        "median is 1059.7 pc, adopted as 1060. SINGLE-CITATION PRIMARY if one "
        "is wanted: Chen et al. 2020 cloud ID 436, 1065.1+-25.1 pc -- the "
        "molecular cloud distance, the physically apt quantity for an "
        "embedded protostar. AFGL 490 has NO astrometry of its own: Gaia DR3 "
        "450155767107967616 is a 2-parameter solution with a blank parallax, "
        "so the superseded row's 'no Gaia-era value' is true OF THE OBJECT "
        "and false OF ITS FIELD. sigma 75 pc is a MEASURED SPREAD -- not a "
        "published error bar and not a fraction of d_r: the seven "
        "determinations have sample sd 70.5 pc and half-range 94.6 pc, and 75 "
        "sits between. Basis 'S+D' is the composite of this table's existing "
        "'S' (Gaia cluster membership) and 'D' (Gaia 3-D dust) codes, in the "
        "grammar of the existing 'M+S'; the Notes remain authoritative over "
        "the code. REFUTED: 800 pc is a one-significant-figure RMS-survey "
        "value propagated Pokhrel et al. 2020, ApJ 896, 60 <- Obonyo et al. "
        "<- Urquhart et al., and it sits BELOW every Gaia-era determination "
        "in this field (944.9-1134 pc). Evidence: "
        "reports/C4_distance_audit.md 5.5, reports/C5_distance_revisions.md"),
    'Aquila': Region('Aquila', 436, 9, (0.427, 0.445), 'M+S', 'Ortiz-León VLBA+Gaia; Pokhrel concurs; Zucker 484±24 variant; identity = Serpens–Aquila Rift (catalog geometry)'),
    'Auriga-California': Region(
        'Auriga-California', 470, 24, (0.446, 0.494), 'D',
        "REVISED 2026-08-21 by owner ruling R-78 (dispatch C5), was 450+-23, "
        "basis 'O'. SOURCE: Zucker, Speagle, Schlafly, Green, Finkbeiner, "
        "Goodman & Alves 2019, ApJ 879, 125, table 3, row 'California' "
        "(l = 160-166, b = -10 to -7.5): 470 +- 2 (stat) +- 24 (sys), Gaia "
        "DR2 3-D dust. This restores the project's own v1 value, verified "
        "against the paper. sigma 24 pc is RULED by the dispatch; the "
        "published quadrature sqrt(2^2 + 24^2) = 24.08 pc rounds to it. "
        "DEPTH TERM -- the gradient is real and larger than either end's "
        "uncertainty: the dust sightlines through the cloud body cluster at "
        "436-473 pc (Zucker 2019/2020 recno 19/20/21; Chen B.-Q., Huang, "
        "Yuan et al. 2020, MNRAS 493, 351, clouds 187/235/189 at "
        "451.2/439.3/473.2 pc) while the LkHa 101 / NGC 1579 end at "
        "l = 165.3 is at 521-564 pc from four independent Gaia catalogues "
        "(Hunt & Reffert 'NGC_1579' 521.2 pc; Prisinzano, Damiani, Sciortino "
        "et al. 2022, A&A 664, A175, cluster 557 / NGC1579 545.0 pc "
        "recomputed from 429 member parallaxes; Kuhn, Hillenbrand et al. "
        "2019, ApJ 870, 32, 'LkHalpha 101' 564 +14/-15 pc; Dias et al. 2021, "
        "MNRAS 504, 356, 561+-102 pc) -- a ~15-20% gradient over 4 deg. "
        "Zucker's box stops at l = 166 and its E(B-V) > 1 threshold weights "
        "the dense mid-cloud, so 470 pc is a MID-CLOUD number, not an LkHa "
        "101 number. REFUTED AS A PAIRING: the 450 is Lada, Lombardi & Alves "
        "2009, ApJ 703, 52 star counts -- measured for this cloud, but "
        "pre-Gaia -- carrying a modern-style error bar (23 is 5% of 450 and "
        "24 is 5% of 470), i.e. the point estimate and its sigma came from "
        "different decades. Evidence: reports/C4_distance_audit.md 5.8, "
        "reports/C5_distance_revisions.md"),
    'BD+40o4124': Region('BD+40o4124', 980, 196, (0.784, 1.176), 'A-FLAG', 'Shevchenko 1991 via Sandell 2012; no Gaia-era primary; 1003 pc Cygnus X sightline 0.35° away recorded as corroboration-in-passing, not adopted'),
    'Cepheus Flare': Region('Cepheus Flare', 358, 32, (0.326, 0.39), 'M', "Dzib 47 YSOs; Zucker 352 concordant; seed's 300 refuted"),
    'Cepheus OB3': Region(
        'Cepheus OB3', 830, 40, (0.790, 0.870), 'M+S',
        "REVISED 2026-08-21 by owner ruling R-78 (dispatch C5), was 725+-110 "
        "(an ASSIGNED flat 15%), basis 'O (1959-era)'. THE SUPERSEDED NOTE'S "
        "CLAIM THAT 'NO MODERN ASTROMETRY EXISTS' IS FALSE AND IS STRUCK, "
        "NOT AMENDED. There is a VLBI MASER PARALLAX INSIDE THE CLOUD: Reid "
        "et al. 2019, ApJ 885, 131, table 1 recno 164, G109.87+02.11 = Cep A, "
        "1.228 +- 0.024 mas = 814.3 +- 15.9 pc. Five Gaia determinations "
        "agree: Karnath, Prchlik, Gutermuth, Allen, Megeath, Pipher, Wolk & "
        "Jeffries 2019, ApJ 871, 46 (Gaia DR2, zero-point corrected, "
        "'Cep OB3b') 819 +- 16 pc; Dias et al. 2021, MNRAS 504, 356, table "
        "12, 'LP_2249' (l 110.526, b +2.705), N=690, 819 +- 17 pc; Hunt & "
        "Reffert 2023, A&A 673, A114 / 2024, A&A 686, A42, 'ASCC_125' "
        "(= FSR_409, MWSC_3677, Theia_4), N=114, 828.2 pc; Xu et al. 2016, "
        "Sci. Adv. 2, e1600878, the same Cep A maser at 1.208 +- 0.025 mas = "
        "827.8 +- 17.1 pc; Kuhn, Hillenbrand et al. 2019, ApJ 870, 32, table "
        "1 recno 31, 863 -29/+31 pc; and Nagy, Abraham, Kospal, Park et al. "
        "2022, MNRAS 515, 1774, 816 +- 53 pc from 235 Gaia EDR3 members "
        "(paper text, not machine-verified). 830 pc is the round of their "
        "mean and is consistent with the Cep A parallax. THE IDENTIFIER IS "
        "WHY THIS WENT UNFOUND: Cep OB3b appears in Hunt & Reffert only as "
        "'ASCC_125' and in Dias 2021 only as 'LP_2249', neither "
        "name-searchable -- the same failure as IRAS 20050+2720's "
        "'65.78-2.61'. sigma 40 pc is DERIVED, not assigned: quadrature of "
        "18 pc (the dispersion among the modern determinations) and 35 pc for "
        "genuine line-of-sight depth across the footprint (the OB3a/OB3b "
        "split, plus Chen B.-Q., Huang, Yuan et al. 2020, MNRAS 493, 351, "
        "cloud 332 at l 110.566, b +2.057 -- 0.04 deg from the SESNA centroid "
        "-- reporting a dust depth of 245 +- 14 pc on exactly this "
        "sightline); sqrt(18^2 + 35^2) = 39.36, rounded to 40. "
        "SINGLE-CITATION FALLBACK: Karnath et al. 2019, ApJ 871, 46, "
        "819 +- 16 pc (the v1 table cited this as 'ApJ 890, 129' -- wrong "
        "volume and page). REFUTED: 725 pc is Blaauw, Hiltner & Johnson 1959, "
        "ApJ 130, 69 photometry of 40 early-type members, via Kun, Kiss & "
        "Balog 2008 -- the provenance claim was accurate, but the value is 66 "
        "years old and is superseded by a maser parallax inside the cloud. "
        "ALSO STRUCK: the superseded note rejected 'Pokhrel's 820' on "
        "citation-hygiene grounds. The citation IS wrong (Pokhrel et al. "
        "2020, ApJ 896, 60, table 1 credits Kun et al. 2008, which contains "
        "no 820), but 820 pc is within 1 pc of the modern astrometric answer, "
        "so rejecting a correct number for a bibliographic defect moved this "
        "row ~100 pc AWAY from the truth. Do NOT adopt Chen et al. 2020's "
        "916.8 pc as the cluster distance: that is the dust structure and it "
        "disagrees with every stellar determination by ~90 pc. CONSEQUENCE, "
        "measured: this was the survey's second-worst region by u_max "
        "(2.877, normalising on only 0.348 of its own sightline column, "
        "A_K_str/A_s 0.53); at 830 pc u_max falls to 1.360, frac to 0.735 and "
        "A_K_str/A_s rises to 1.64, both mid-distribution. Evidence: "
        "reports/C4_distance_audit.md 5.3, reports/C5_distance_revisions.md"),
    'Chameleon': Region('Chameleon', 192, 6, (0.186, 0.198), 'M', 'Cha I; Cha II (198) in depth term; SESNA spelling canonical'),
    'CrA': Region('CrA', 154, 4, (0.15, 0.158), 'M', 'Dzib'),
    'Cygnus X': Region('Cygnus X', 1400, 80, (1.32, 1.48), 'O', 'Rygl maser, combined; ±0.10 kpc sub-region spread in depth term'),
    'GGD4, CB34': Region('GGD4, CB34', 1356, 77, (1.279, 1.433), 'O', "mean of Zucker sightlines 1349/1396/1322; components CO-DISTANT; seed's 850 wrong by ~55–64% (largest error found); optional footprint check open"),
    'IC 5146': Region('IC 5146', 813, 106, (0.707, 0.919), 'M', 'Dzib 62 YSOs'),
    'IRAS 20050+2720': Region(
        'IRAS 20050+2720', 1326, 84, (1.242, 1.410), 'S',
        "REVISED 2026-08-21 by owner ruling R-77 (dispatch C4), was 700+-140. "
        "SOURCE: Prisinzano, Damiani, Sciortino et al. 2022, A&A 664, A175, "
        "table 3 group 391, listed under the literature name '65.78-2.61' "
        "(this object's own Galactic coordinates -- why name searches miss it); "
        "134 Gaia EDR3 members, published d = 1326 pc, re-derived here from "
        "their table 4 member parallaxes as 1345+-10 pc (inverse-variance "
        "weighted). sigma 84 pc as RULED, replacing an assigned 20%; it sits "
        "between the formal error on the group's mean (+-10 pc) and the full "
        "16-84 spread of the 134 members' own parallax distances (1238-1490 "
        "pc, half-spread 126 pc) -- that spread is noise-dominated, the median "
        "member parallax error 0.0745 mas exceeding the observed half-spread "
        "0.0684 mas, so the group's depth is unresolved. See C4 sec 2. "
        "CORROBORATION: Chen B.-Q., Huang, Yuan et al. 2020, MNRAS 493, 351, "
        "cloud 363 at 1107+-26 pc, 0.96 deg away, no cloud within 800 pc on "
        "this sightline. REFUTED: the seed's 700 pc was a Cygnus Rift "
        "ASSOCIATION (Dame & Thaddeus 1985 extinction jump averaged over "
        "l = 64-74 deg, via Wilking et al. 1989, ApJ 345, 257, table 1) -- "
        "0 of 134 members has a parallax consistent with it. Evidence: "
        "reports/C2_iras_and_fallback.md 1.6, reports/C4_distance_audit.md"),
    'L988': Region('L988', 620, 32, (0.588, 0.652), 'O', "Zucker d50 612/627 -- UPGRADED from assumed; A-flag retired; confirms Walawender's assumption independently"),
    'Lupus': Region('Lupus', 158.3, 0.6, (0.1577, 0.1589), 'M', 'Galli 113 members; subclouds 156–163 in depth term; Zucker 189 variant'),
    'Mon OB1': Region(
        'Mon OB1', 745, 37.12, (0.70788, 0.78212), 'D',
        "SIGMA CORRECTED 2026-08-21 by owner ruling R-79 (dispatch C5), was "
        "24.187. THIS IS AN ARITHMETIC CORRECTION, NOT A DISTANCE REVISION: "
        "d_r is UNCHANGED at 745 pc and no citation is withdrawn. The 24.187 "
        "was sqrt(3^2 + 24^2), but the 24 is not this row's systematic term. "
        "Zucker et al. quote sigma_sys = 5% of d; 24 is 3.2% of 745 and "
        "5.1% of 470, i.e. it is Auriga-California's systematic term sitting "
        "on Mon OB1's row. Zucker, Speagle, Schlafly, Green, Finkbeiner, "
        "Goodman & Alves 2019, ApJ 879, 125, table 3 publishes Mon OB1 as "
        "745 +- 3 +- 37, so the correct quadrature is sqrt(3^2 + 37^2) = "
        "37.12 pc (5.0% of d_r); the table previously UNDERSTATED this row's "
        "uncertainty by 35%. Provable from the table's own internal "
        "consistency without opening a paper: every other Zucker-based row "
        "obeys the 5% rule to within 5% (Perseus 15 vs 14.7, Taurus 7 vs "
        "7.05, L988 31 vs 31.0, S131 48 vs 45.9, Auriga-California 23 vs "
        "22.5) and only this one broke it. NGC 2264 738 concordant. Nothing "
        "downstream moves: d_norm stays 854.0 pc and u_max 1.273, and the "
        "structure flip is at 881.9 pc (+18.4%), 4.8 sigma away on the "
        "corrected value -- C2's tie-point flag at 759.0 pc was a false alarm "
        "(its two competitors at 733.7 and 784.9 pc are two clumps of ONE "
        "structure and return an identical answer). Evidence: "
        "reports/C4_distance_audit.md 5.9 and 7.1, "
        "reports/C5_distance_revisions.md"),
    'Mon R2': Region('Mon R2', 860, 31, (0.829, 0.891), 'S', "Pokhrel's own Gaia fit of >300 SESNA sources -- strongest basis in the table"),
    'Musca': Region(
        'Musca', 172, 16, (0.156, 0.188), 'D',
        "REVISED 2026-08-21 by owner ruling R-78 (dispatch C5), was 187+-30, "
        "basis 'A-FLAG'; THE A-FLAG IS RETIRED WITH IT. SOURCE: Zucker, "
        "Goodman, Alves, Bialy, Koch, Speagle et al. 2021, ApJ 919, 35, table "
        "1, row 'Musca' at (l, b) = (301.0, -9.7), 0.94 deg from the SESNA "
        "centroid: 172 pc (min/med/max 171/172/173) -- the ONLY Gaia-era "
        "determination made FOR MUSCA ITSELF AT MUSCA'S OWN COORDINATES. "
        "sigma 16 pc is DERIVED as half the spread between the two "
        "independent modern values: Zucker et al. 2021's 172 pc and the Gaia "
        "DR2 reddening jump at 140 pc of Bonne, Bontemps, Schneider et al. "
        "2020, A&A 644, A27. Edenhofer et al. 2024 places the Musca material "
        "in a 165-220 pc slab. NULLS, machine-verified: no Musca entry in "
        "Prisinzano et al. 2022, Cantat-Gaudin et al. 2020, Hunt & Reffert, "
        "Dias et al. 2021 or Kuhn et al. 2019 -- Musca has essentially no "
        "cluster. Chen B.-Q., Huang, Yuan et al. 2020, MNRAS 493, 351, cloud "
        "ID 9 is a positional match at 117.5 pc, but that catalogue's very "
        "nearby 4th-quadrant clouds run 40-60% short against Zucker on the "
        "same sightlines, so it is flagged and NOT used. REFUTED: 187 pc was "
        "an ASSOCIATION value -- the same failure mode as IRAS 20050+2720. It "
        "is arithmetically the mean of the three 'Chamaeleon' rows of Zucker "
        "et al. 2020, A&A 633, A51 ((297.5, -15.3) = 210, (303.3, -14.2) = "
        "190, (303.0, -16.7) = 161; mean 187.0 = the catalog value to the "
        "pc), sightlines 5.5-8 deg away in latitude. 'Musca' appears 0 times "
        "in the full text of Zucker et al. 2019 and is absent from Zucker et "
        "al. 2020's 95-name enumeration -- so the superseded note's 'absent "
        "from Zucker' was correct, and the response to that absence was to "
        "average three neighbours, landing the point estimate ABOVE every "
        "direct modern determination of Musca. Evidence: "
        "reports/C4_distance_audit.md 5.7, reports/C5_distance_revisions.md"),
    'NGC 7129': Region('NGC 7129', 926, 163, (0.763, 1.089), 'M', 'Dzib 40 YSOs'),
    'North America Nebula': Region('North America Nebula', 785, 16, (0.769, 0.801), 'O', "Kuhn & Hillenbrand EDR3; ~130 pc group depth in depth term; seed's 600 refuted"),
    'Ophiuchus': Region('Ophiuchus', 137.3, 1.2, (0.1361, 0.1385), 'M+S', 'VLBA'),
    'Orion A': Region('Orion A', 418, 21, (0.397, 0.439), 'S', 'Yan via Pokhrel; member gradient 388–470 in depth term'),
    'Orion B': Region('Orion B', 418, 21, (0.397, 0.439), 'S', 'as Orion A'),
    'Perseus': Region('Perseus', 294, 15.133, (0.2789, 0.3091), 'D', 'Zucker; IC 348 (321) / NGC 1333 (293) split in depth term'),
    'Pipe': Region('Pipe', 163, 5, (0.158, 0.168), 'M', "B59 members; seed's 145 (dust) superseded per R-i"),
    'S131': Region('S131', 918, 48, (0.87, 0.966), 'O', '≡ IC 1396, positionally verified from the Sharpless catalog; mean of 4 Zucker sightlines; Kun ~800 (association-era) variant per the modern-beats-pre-modern corollary'),
    'S140': Region(
        'S140', 908, 30, (0.878, 0.938), 'S',
        "REVISED 2026-08-21 by owner ruling R-78 (dispatch C5), was 764+-27, "
        "basis 'S/O'. THE SUPERSEDED BASIS IS STRUCK, NOT AMENDED: 764 +- 27 "
        "pc IS A PARALLAX TO A DIFFERENT OBJECT. Hirota et al. 2008, PASJ 60, "
        "961 measured a VERA H2O maser parallax of 1.309 +- 0.047 mas for "
        "IRAS 22198+6336, in the dark cloud L1204G (l = 107.29, b = +5.63; "
        "Reid et al. 2019's G107.29+05.63), 0.58 deg from S140, and that "
        "paper contains NO mention of S140 or Sharpless 140. The object "
        "substitution enters at Pokhrel et al. 2020, ApJ 896, 60, table 1, "
        "which lists S140 = 764 pc citing Hirota; this table inherited "
        "Pokhrel's row. The +-27 was a real uncertainty belonging to a source "
        "0.58 deg away. SOURCE: Szilagyi, Kun, Abraham & Marton 2023, MNRAS "
        "520, 1390, table 2, group 8, Name = 'SH 2-140' (l 106.598, "
        "b +5.209), 45 Gaia EDR3 members, 906.2 pc -- S140 listed BY NAME. "
        "(The v1 table cited this paper as 'MNRAS 519, 5471' -- wrong volume "
        "and page.) CORROBORATION: Hunt & Reffert 2023, A&A 673, A114 / 2024, "
        "A&A 686, A42, 'Pismis-Moreno_1', N=128, 909.1 pc; Dias et al. 2021, "
        "MNRAS 504, 356, table 12, 'Pismis_Moreno_1', N=60, 917 +- 12 pc; "
        "Reid et al. 2019, ApJ 885, 131, table 1, G108.18+05.51, a VLBI CH3OH "
        "maser in L1206 in the same complex 1.39 deg away, 915.8 +- 15.1 pc; "
        "Zucker et al. 2020, A&A 633, A51, 901 and 891 pc. Kun, Kiss & Balog "
        "2008 -- the same chapter the Cepheus OB3 row leant on -- already put "
        "S140 'at a distance of about 900 pc from the Sun'. This project's "
        "own dust map peaks at 923.9 pc with prominence 0.991, the MAXIMUM on "
        "the sightline. sigma 30 pc is DERIVED, and is deliberately NOT "
        "Szilagyi's 5.4 pc, which is a standard error of the mean of 45 "
        "members and not a distance uncertainty: quadrature of 10 pc (the "
        "scatter among the modern determinations), ~16 pc (the Gaia parallax "
        "systematic floor at 900 pc) and ~20 pc (the exciting-cluster vs "
        "embedded-cluster ambiguity); sqrt(10^2 + 16^2 + 20^2) = 27.5, "
        "rounded to 30. There really are TWO structures on this sightline -- "
        "IRAS 22198+6336 / L1204G at ~764-776 pc, and S140 / L1204 / L1206 / "
        "Pismis-Moreno 1 / the dominant dust wall at ~905-920 pc -- and the "
        "canon was pinned to the foreground one, which is why this was the "
        "one region of 30 that FAILED the project's own peak-concordance "
        "check (+20.9%; +1.8% at 908 pc). Evidence: "
        "reports/C4_distance_audit.md 5.1, reports/C5_distance_revisions.md"),
    'S171': Region(
        'S171', 1010, 40, (0.970, 1.050), 'S',
        "REVISED 2026-08-21 by owner ruling R-78 (dispatch C5), was 1100+-110 "
        "(an ASSIGNED flat 10%), basis 'O'. SOURCE: Wiesneth, Muzic & "
        "Almendros-Abad 2025, A&A 703, A193, table 2, column d2, "
        "'Berkeley 59' (S171's cluster): 1009 +- 12 pc, Gaia DR3 maximum "
        "likelihood WITH the Lindegren et al. 2021 parallax-bias correction "
        "applied (1037 +- 12 pc uncorrected) -- the only determination on "
        "this sightline with an explicit zero-point correction. "
        "CORROBORATION: Hunt & Reffert 2023, A&A 673, A114, 'Berkeley_59' ID "
        "2856 (= Theia_670), 992 Gaia DR3 members, 1015.4 pc; Dias et al. "
        "2021, MNRAS 504, 356, table 12 recno 144, 'Berkeley_59', N=223, "
        "1026 +- 26 pc; Panwar, Sharma, Ojha, Samal, Singh & Yadav 2024, AJ "
        "168, 89, 1000 +- 60 pc. Median of the four modern determinations "
        "1012 pc, full range 26 pc. sigma 40 pc is RULED by the dispatch -- "
        "NOT derived and NOT a fraction of d_r; it is bracketed by Wiesneth's "
        "formal +-12 pc, Dias's +-26 pc, Kuhn's +-50 pc and Panwar's +-60 pc, "
        "and it is 4.0% of d_r. REFUTED IN PART: Kuhn, Hillenbrand et al. "
        "2019, ApJ 870, 32, table 1 recno 1, Reg = 'Berkeley 59', "
        "1100 +- 50 pc is a Gaia DR2 parallax INVERSION with no zero-point "
        "correction (0.91 +- 0.04 mas, 225 members). The drift is traceable "
        "and is NOT a disagreement: DR2 0.91 mas -> DR3 0.9605 mas (Hunt & "
        "Reffert, 992 members) -> zero-point corrected ~0.99 mas (Wiesneth). "
        "ALSO CORRECTED: the superseded note's 'no published sigma' was "
        "accurate about Gahm, Wilhelm, Persson, Djupvik & Portegies Zwart "
        "2022, A&A 663, A111 sect. 4.1, whose '0.9 +- 0.1 mas' is the SCATTER "
        "of ~21 individual O/B member parallaxes with no DR3 zero-point "
        "correction and whose '1.1 kpc' is a rounded 1/0.9 -- but the "
        "response to a missing sigma was to assign 10%, about 3x the modern "
        "spread. The revision also removes this row's knife edge: at 1010 pc "
        "d_r sits 4.6% below the chosen 1057.4 pc structure and 11.0% from "
        "its competitor, and the 4.0% sigma never reaches the flip (2.8 sigma "
        "away), where the assigned 10% straddled it. Evidence: "
        "reports/C4_distance_audit.md 5.6 and 7.2, "
        "reports/C5_distance_revisions.md"),
    'Scorpius': Region('Scorpius', 146, 6.708, (0.1393, 0.1527), 'M', 'Galli kinematic; de Zeeuw transcription discharged'),
    'Taurus': Region('Taurus', 141, 7.28, (0.1337, 0.1483), 'D', 'Zucker; L1495 (129.5) in depth term'),
    'Vela D': Region(
        'Vela D', 930, 80, (0.850, 1.010), 'D',
        "REVISED 2026-08-21 by owner ruling R-78 (dispatch C5), was 700+-200 "
        "(28.6%, the widest fractional uncertainty in the table), basis 'O'. "
        "FOOTPRINT SETTLED FIRST, which discharges the row's PENDING check: "
        "Strafella et al. 2010 define the Spitzer VMR-D field (PID 30335) as "
        "263d00' <= l <= 264d29', -0d50' <= b <= 0d42', centre "
        "(l, b) = (263.74, -0.067) -- 0.02 deg from the SESNA centroid, so "
        "the SESNA 'Vela D' footprint IS that field. Zucker et al. 2020, A&A "
        "633, A51 has five Vela_C rows and NO Vela_D row, and its nearest "
        "Vela_C sightline is 1.755 deg OUTSIDE a 1.2 deg^2 footprint: NO "
        "OVERLAP -- Vela C is a neighbour, not this cloud. SOURCE: "
        "Dharmawardena, Bailer-Jones, Fouesneau et al. 2023, MNRAS 519, 228, "
        "table 3, region 'Vela D', leaf L70, centroid (264.6, -0.43, 972 pc), "
        "leaf extent 956-988 pc -- the only published Gaia 3-D-dust structure "
        "LABELLED Vela D whose box overlaps the footprint. CORROBORATION: "
        "Chen B.-Q., Huang, Yuan et al. 2020, MNRAS 493, 351, table 1 cloud "
        "ID 382 (265.747, +0.457), 2.10 deg, 934.4 +- 22.1 pc; Zucker et al. "
        "2020 Vela_C, 931 pc (866-965), 1.76-2.77 deg; an ON-FOOTPRINT, "
        "model-free StarHorse (Anders et al. 2022, A&A 658, A91) "
        "A_V-vs-distance wall -- median A_V is FLAT at 0.6-0.8 mag from 400 "
        "to 800 pc, so THERE IS NO EXTINCTION STEP AT 700 pc, while the "
        "single +1.1 mag step runs 800-975 pc with its half-rise at ~880 pc, "
        "a control field 3.5 deg away showing only a weak smooth gradient; "
        "and this project's own dust map peak at 898.3 pc, prominence 0.474, "
        "the sightline MAXIMUM. sigma 80 pc is a MEASURED METHOD SCATTER "
        "across those determinations (~890-972 pc) -- the honest "
        "characterisation, since no published measurement is centred exactly "
        "on the box; it is NOT a published error bar and NOT a fraction of "
        "d_r, and it drops the fractional uncertainty from 28.6% to 8.6%. "
        "REFUTED: 700 +- 200 pc is Liseau et al. 1992, A&A 265, 577 IR "
        "photometry of Class I sources measured for 'VMR A, C and D' JOINTLY "
        "-- there is no per-cloud value and no measurement of this box -- "
        "carried via Massi et al. 2007. The v1 table's citation was "
        "mis-scoped in the other direction: Massi et al. 2019, A&A 628, A110 "
        "is a LABOCA survey of Vela C whose appendix derives a Vela C "
        "distance. DO NOT USE the 'Pettersson & Reipurth 1007 +- 30 pc' that "
        "the v1 range traces to: it is explicitly IN PREPARATION and "
        "unciteable. SEPARATE FLAG, NOT A DISTANCE QUESTION: the Gaia-visible "
        "YSO candidates inside the Spitzer box are predominantly Carina-arm "
        "background -- of 504 SPICY candidates, the 210 with Gaia DR3 "
        "astrometry peak sharply at 0.44 mas (~2.29 kpc) with a comoving core "
        "of 75 stars -- which is a contamination question for N_YSO, on the "
        "record and out of scope here. Evidence: "
        "reports/C4_distance_audit.md 5.2, reports/C5_distance_revisions.md"),
}


# --- Shared structural plugs and grid conventions (scope categories 3 & 4) ---
#
# These are not measurements. They are values that MUST be identical across
# libraries, because the model directories are read against one another and a
# divergence in the 7th significant digit silently shifts an aperture frame.

# The DISTANCE header every library writes into flux.fits. NOT a physical
# parsec conversion and deliberately NOT derived from PC_CM: galaxy, h2shock,
# pahc, all five YSO strata AND the upstream Robitaille release all carry this
# exact literal. PC_CM * 1000 = 3.0856775814913673e21 differs by 8.6e-7 and is
# used by nothing. For pahc it is load-bearing rather than a plug -- its
# aperture reference frame divides by this value -- so "improving" the
# precision would silently move pahc's apertures off the YSO grid it copies.
REFERENCE_DISTANCE_CM = 3.08568025e21     # cm; matches the upstream release header

# Point-source libraries (galaxy, h2shock, sps) carry a single aperture whose
# value is a structural sentinel meaning "no aperture dependence", not a size.
# Distinct in meaning from any flux floor that happens to share the magnitude.
POINT_SOURCE_APERTURE_AU = 1e-30          # AU

# The 20-value aperture axis shared by the aperture-dependent libraries (YSO,
# and pahc which copies YSO's so the two can be compared). Verified
# byte-identical to the upstream release's APERTURES column (tobytes() equal,
# max relative difference 0.0), so generating it here rather than reading a
# neighbouring library's flux.fits at build time changes not one output byte.
APERTURE_GRID_AU = np.logspace(2.0, 6.0, 20)   # AU, 100 -> 1e6


# --- The extinction domain [R-87] ---
#
# THE PROJECT-WIDE EXTINCTION BOUND. One quantity, one place. The fitter
# needs a viable extinction range and the census needs a viable one too;
# before R-87 the fitter's was a per-run key in every `.cfg` file and the
# census's was a re-typed float literal in four modules, so nothing made the
# two agree or even compared them. This is the project-wide reference:
# anyone who needs a viable extinction range references it here rather than
# specifying one independently.
#
# DECLARED IN A_K, NOT A_V. This is the point, not a formatting choice.
# Since R-84 the extinction law is gated PER SOURCE, so one nominal A_V
# number is two different physical bounds depending on which law the source
# drew: A_V 500 is A_K 71.2 on Whitney r550 and A_K 58.7 on Draine R_V=3.1.
# That is exactly the class- and law-dependent structure R-84 removed from
# the extinction law, reintroduced through the choice of CURRENCY, and it
# cannot be compared against the census's A_K domain at all. Declared in
# A_K it is ONE physical bound; converted at the point of use through the
# curve-internal ratio of whichever law the source drew, AK_MAX becomes
# A_V 527.0 on Whitney and A_V 638.7 on Draine -- the same physical limit
# expressed twice. That is the K-CURRENCY DOCTRINE (R-84): everything stays
# in native A_K and conversion happens once, at the fitter boundary, with
# `ExtinctionLaw.av_over_ak_curve` (owner ruling 2026-08-18) -- never with a
# law file's published `.info` `AV_over_AK` scalar, which is provenance only.
#
# WHAT IT IS. A PHYSICS GUARD on the fitter's fitted extinction, not a
# search bound and not a prior. `sedfitter` solves A_V in CLOSED FORM
# (`linear_regression`) and then merely CLAMPS the solution into this
# interval (`sedfitter/models.py:369-372` and `:392-393`), so a narrower
# interval cannot make the estimate more precise -- given the model and the
# photometry it is already exact -- it can only OVERRIDE the data. Widening
# it costs nothing: there is no grid to enlarge. Truncating the plausible
# range of extinction is the CENSUS PRIOR's job, where it is explicit,
# normalised and carries a recorded out-of-support rule [R-70].
#
# WHY THESE TWO NUMBERS.
#   AK_MIN = 0     negative extinction is impossible. Nothing else.
#   AK_MAX = 75    beyond anything in the data, with room to spare. The
#                  largest column in the survey is AK_IRDUST = 47.131
#                  (Orion A) over all 8,660,483 sources (p50 0.418, p99
#                  1.304, p99.99 4.402; next highest regions Aquila 17.75,
#                  Orion B 17.47, Perseus 16.34). Because the bound is
#                  declared in the same currency the column is measured in,
#                  that comparison is direct and law-independent: 75 > 47.131
#                  clamps nothing, on any law. For scale, the equivalent
#                  A_V ceilings measured earlier clamped (Whitney / Draine)
#                  78 / 132 sources at A_V 100, 17 / 23 at 200, 4 / 9 at
#                  300, 0 / 1 at 400 and 0 / 0 at 500. Since every clamped
#                  fit is mis-priced (ROOT-07, not ruled), the guard is
#                  chosen so it never fires. It is NOT a claim that a source
#                  at A_K 47 is a galaxy -- at A_K = 47 a background galaxy
#                  is dimmed 47 mag in K and nothing survives.
#
# `AK_MAX` IS NO LONGER COUPLED TO THE PRIOR AT ALL [R-95 RULING 1].
# It was, until 2026-08-22, tied to `AK_CAP` by the invariant
# `AK_MAX >= AK_CAP`, on the reasoning that `AK_CAP` was the DOMAIN the
# census prior's `a` axis was truncated and renormalised on [R-70] and the
# clamp must never truncate that support. **THE PRIOR NO LONGER HAS A
# SUPPORT FOR THE CLAMP TO TRUNCATE.** [R-95] ruling 1 made the `a` axis
# `[0, inf)` on all five cells, so there is nothing for `AK_MAX` to be
# greater than: whatever the fitter returns, the prior is defined there and
# prices it. `AK_CAP` survives as a TABULATION EXTENT -- where the density
# is cached -- and a clamp has no business being compared against a cache.
#
# THE INEQUALITY STILL HOLDS ARITHMETICALLY (75 > 14.232) and
# `tests/sed_fit/test_config.py` still asserts it over every configured law.
# It is retained as a CONSISTENCY CHECK ON THE TABULATION -- a clamp inside
# the cached extent would mean the fitter could never reach the part of the
# axis that has to be evaluated analytically, which would hide a bug rather
# than cause one -- and NOT as the support statement it used to be. Do not
# re-derive one number from the other in either direction.
#
# `AK_CAP` AND `AK_MAX` ARE NOT ONE CONSTANT, and this file must never be
# read as implying they are. That was already [R-88] ruling 1's corollary --
# a clamp that must never fire and a normalisation domain are two quantities
# with two jobs -- and [R-95] ruling 1, which SUPERSEDES [R-88] ruling 1,
# leaves the conclusion standing on a different footing: they are now a
# clamp that must never fire and a CACHE EXTENT, which is further apart
# still. What [R-89] ruling 5 changes is only that `AK_CAP` has exactly
# ONE definition instead of five -- it was re-typed as a bare literal at four
# further sites -- and that definition is below. The column-grid product
# (`class_densities.column_grid.build()`) derives its node ladder from
# this value at build time, and this file is the leaf every other module
# already imports `AK_CAP` through. `census/shape_yso.py`'s successor is
# `bms_prior/class_densities/class_density_yso_h2s.py`.
AK_MIN = 0.0
AK_MAX = 75.0

#: `(AK_MIN, AK_MAX)`, for callers that want the pair.
AK_RANGE = (AK_MIN, AK_MAX)

# --------------------------------------------------------------------
# THE CENSUS PRIOR'S `a` DOMAIN, AND THE A-NODE LADDER ON IT  [R-89]
# --------------------------------------------------------------------
#
# `AK_CAP` -- A_V 100 mag converted CURVE-INTERNALLY on whitney.r550
# (14.232012269510967 / 0.14232012269510967 = 100.0 exactly) [R-62].
#
# **IT IS A TABULATION EXTENT, NOT A SUPPORT [R-95 RULING 1].** It is
# WHERE THE DENSITY IS CACHED, not where it is defined. Every census cell's
# `a` axis is a proper density on `SUPPORT_A = [0, inf)` and normalises
# ONCE, over everything; the tail above `AK_CAP` is EVALUATED, not cut.
#
# WHAT THIS SUPERSEDES, EXPLICITLY. Until 2026-08-22 this block read "It is
# the domain the census prior's `a` axis is truncated and renormalised over
# [R-70], and `support_a = [0, AK_CAP]` on all five cells [R-88 ruling 1]".
# **[R-95] ruling 1 SUPERSEDES [R-88] ruling 1 and its corollary**, and
# RETIRES truncate-and-renormalise [R-70] on the `a` axis (it stands
# unchanged on every other axis and for the `a < 0` boundary, which is
# `AK_MIN` and is physics).
#
# THE OWNER'S REASONING, VERBATIM: "Extinction is a positive quantity with
# no inherent upper bound. Our priors should respect that, but should be
# the bulk of their mass at expected ranges of AK (each model class will
# have a diff distribution of this mass on the support [0, inf])."
#
# THE DEFECT IT FIXES. The census was separately ruled to PRICE whatever
# extinction the fitter returns, with implausibility SCORED BY THE PRIOR. A
# truncated prior cannot do that: outside its support it returns ZERO,
# which ANNIHILATES the hypothesis instead of penalising it. Positivity
# everywhere the fitter can land is what that ruling requires.
#
# THE COST, MEASURED BEFORE THE CHANGE WAS MADE (per-class ratio of the
# `[0, AK_CAP]` integral to the `[0, inf)` integral -- every density moves
# by exactly this factor):
#
#     lambda~_STAR   1.000000 EXACTLY, at every A-node
#     lambda~_PAH    1.000000 EXACTLY, at every A-node
#     lambda~_YSO    0.999982 source-weighted; worst node 0.90074
#     lambda~_H2S    same as YSO (one shared a factor)
#     lambda~_GAL    0.999911 source-weighted; worst node 0.51078
#
# STAR and PAH are EXACTLY unchanged and that is a measurement, not an
# approximation: their rows sit at `a = A*U <= A <= AK_CAP` and the
# conditional measurement kernel's maximum offset is +1.185 at `A = 1` and
# NEGATIVE everywhere above `A = 4` (max -0.887 for the whole `A > 4.837`
# window), so no mass has ever crossed the cap on those two cells. GAL is
# the one the cost note predicted: its operator is the identity, so at
# `A = AK_CAP` the cap sits at the MEDIAN of `p(T | column)` and the
# truncation was throwing away half the density and scaling the remainder
# up by 1.958. 78 of 8,660,483 sources (0.0009 %) carry a column above
# `AK_CAP` at all.
#
# See the `AK_MAX` block above for why the clamp and this number are
# separate; the arithmetic relation `AK_MAX > AK_CAP` is asserted in
# `tests/sed_fit/test_config.py` over every configured law, and in
# `tests/bms_prior/test_class_densities_wave.py` directly, as a check on
# the tabulation.
#
# ONE DEFINITION [R-89 ruling 5]. It was previously declared in the
# census package's `shape_yso.py` and re-typed as a bare literal at four
# more sites, including `bmselect/scripts/s4_shape_lib.py` -- none of
# those old paths exist in the current tree; the successor of the last,
# `bms_posterior/studies/s4_shape_lib.py`, now imports `AK_CAP` from here
# rather than re-typing it. Five copies of one number, now one.
AK_CAP = 14.232012269510967

#: **THE `a` AXIS'S SUPPORT [R-95 RULING 1]**: `[0, inf)`, on ALL FIVE
#: census cells. Not `[0, AK_CAP]` -- see the block above `AK_CAP` for what
#: that number is now and for the measured per-class cost of the change.
#: `np.inf` is the honest upper limit and is written as such so that no
#: consumer can mistake a large finite number for a bound it may truncate on.
SUPPORT_A = (AK_MIN, float("inf"))

#: The ratified grid precision [Q49, RULED 2026-08-25]: the maximum
#: fraction of prior probability mass the A tabulation may misplace
#: (relative L1 between the exact conditional and its grid interpolant,
#: worst mid-interval column). Anchors: equals the TIGHTEST measured
#: cell noise floor (tabulation subordinate to every cell's own
#: sampling error) and puts <= ~2*eps = 0.004 nats on any class
#: log-odds, 25x under Q36's 0.1-nat bar. Full justification: Q49.
A_GRID_EPS = 0.002


# --- Gaia detectability sigmoid (sed_fit's Gamma term) ---
#
# The model-side detectability sigmoid H(Gmag) = sigmoid((G_LIM - Gmag) /
# TAU_G) that gates the Gaia-congruence term (`sed_fit.term_hooks.
# gaia_hook`) -- PINNED literature values, not a fitted or hedged
# quantity, so they live here as a plain physical/instrumental fact
# rather than as a per-module tunable.
#
# G_LIM: Gaia DR3's 50%-completeness G magnitude for a moderately dense
# field -- the empirical DR3 selection function, Cantat-Gaudin et al.
# 2023, A&A 669, A55. NOT the onboard detection threshold G ~ 20.7
# quoted in Prusti et al. 2016, A&A 595, A1 (the mission overview) --
# that is where the instrument can trigger a detection, not where the
# catalog is complete.
G_LIM = 20.5   # mag

# TAU_G: the sigmoid's softness, in mag. Built from two components: the
# instrument completeness rolloff itself (~0.34 mag, Cantat-Gaudin et
# al. 2023) broadened by the model-side G-band prediction error
# (~0.65 mag), which is dominated by the fixed G-band extinction
# coefficient's colour/A0 dependence (Danielski et al. 2018, A&A 614,
# A19) rather than by the rolloff. Derivation: studies/
# S-D9_gaia_detection_limit.md.
TAU_G = 0.50   # mag


# --- SED-library distance grid ---
#
# LOGD_STEP: the log10-distance grid spacing shared by every SESNA SED
# library's `models.conf` (`logd_step`) -- the single value every
# `write_models_conf` call site writes and every fitter-side
# trial-distance grid (`sed_fit.nodes`) derives its own node count
# from. One authority, not a re-typed literal per site.
LOGD_STEP = 0.02   # dex, models.conf logd_step, every SESNA library


# --- H2-shock knot occurrence and extraction fractions ---
#
# ETA_KNOTS_PER_YSO: H2 knots present per intrinsic young star, entering
# the H2S amplitude directly (`class_density_yso_h2s.h2s_amplitude_
# deg2`) in place of the retired knot-to-catalogue folded rate -- a
# per-field value where a direct count against the served law measures
# it, and one value for every other (far) region otherwise
# (`eta_knots_per_yso`). Every caller reads this dict and its default
# by name, never a hardcoded number.
ETA_KNOTS_PER_YSO = {
    "Cygnus X": 0.0105,
    "North America Nebula": 0.0572,
    "Vela D": 0.0668,
}
#: Every SESNA region without its own direct count -- the near-field
#: value (Cygnus X/North America Nebula's own counts agree with it to
#: 0.07 dex).
ETA_KNOTS_PER_YSO_POOLED = 0.06
#: Single band, dex, the three-field spread -- shared by every field
#: and the default value alike.
ETA_KNOTS_PER_YSO_BAND_DEX = 0.45
ETA_KNOTS_PER_YSO_CITATION = (
    "direct count against the served law over the UWISH2 and Giannini "
    "footprints (study S-D37b)")


def eta_knots_per_yso(region):
    """Knots per intrinsic young star for `region`: `ETA_KNOTS_PER_YSO`'s
    own per-field value where measured, `ETA_KNOTS_PER_YSO_POOLED`
    everywhere else -- the one place this lookup happens, so a future
    re-measurement only has to edit the dict/pooled value above."""
    return float(ETA_KNOTS_PER_YSO.get(str(region), ETA_KNOTS_PER_YSO_POOLED))


#: EPS_EXT: the fraction of knots clearing a region's own detection
#: limits that go on to receive a SESNA catalogue row -- a positional
#: cross-match against five external H2-knot surveys, applied as a flat
#: multiplicative factor on the H2S amplitude alongside the per-source
#: pass fraction. `EPS_EXT_CI` is the cross-match's own 95% statistical
#: interval; `EPS_EXT_BAND` is the wider band carried until the
#: measured flux dependence (a systematic beyond the statistical CI) is
#: priced in.
EPS_EXT = 0.27
EPS_EXT_CI = (0.230, 0.313)
EPS_EXT_BAND = (0.15, 0.35)
EPS_EXT_CITATION = (
    "positional cross-match against five H2 knot surveys (study S-D44)")
