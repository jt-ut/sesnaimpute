"""Curate a yso-conforming copy of the sps (Kurucz/ATLAS9 stellar photosphere)
model set into a new output directory.

Stages -- the canonical numbering. The curation plan's Sec. 2.1b and the
driver's per-stage logging use these same five names, plus stage 0's raw
fetch(es) (Extension A / D-18 and Extension A' / D-25, neither one of the
five and neither part of curate_sps_model_set -- see the driver):

  0. fetch_btsettl_cifist  -- (Extension A / D-18) pin + fetch the raw
                              BT-Settl CIFIST ascii grid; run once, ahead of
                              time, NOT invoked by curate_sps_model_set
     fetch_btsettl_agss    -- (Extension A' / D-25) same pattern, the raw
                              BT-Settl AGSS2009 ascii grid; also run once,
                              ahead of time, and optional -- omitting its
                              output dir (`agss_download_dir=None`) makes
                              curate_sps_model_set reproduce the pre-A'
                              CK03+CIFIST-only build exactly
  1. copy_static_metadata  -- models.conf written; parameters.fits rebuilt
                              for the MERGED CK03+CIFIST(+AGSS) row set with
                              a new SOURCE column (D-16, D-24)
  2. build_flux_cube       -- the common grid (D-13); CK03 rows log-log
                              interpolated onto it, CIFIST rows flux-
                              conservingly rebinned with a Planck tail
                              (D-14/D-15); spliced together (D-12); AGSS
                              rows (when given) ride the CIFIST path (D-21)
                              and are appended strictly additively (D-19/D-20)
  3. write_flux_fits       -- one combined flux.fits via model_io.write_flux_cube
  4. build_convolved_bands -- convolve the 8 SESNA bands from the flux.fits
                              just written, through this project's filter curves
  5. write_classmap_fits   -- CLASS/SUBCLASS labels, now spanning O..T (D-17);
                              unchanged by Extension A' (D-A'2/D-A'5)

Until Extension A (2026-08-22, D-11..D-18), there was no real curation here:
no new physics, no rebuilt grid, no new source data -- stage 1 authored
models.conf and rewrote only the padding of one parameters.fits column;
stage 2/3 repacked 3,808 per-model files into the one combined cube the
source distribution never shipped; stage 4 convolved that cube through this
project's own filter curves. Extension A changes that: the library is
extended downward from 3,500 K to 1,200 K with BT-Settl CIFIST2011_2015
models (D-11), on a new common log-uniform R=300 wavelength grid replacing
the inherited 1,321-point one (D-13), with CIFIST fluxes diluted and
renormalised to exact bolometric closure (D-14), a Planck-tail red-end
extrapolation and closure/coverage keep-rule (D-15), a new `SOURCE` column
and `_cifist` name suffix (D-16), and `SUBCLASS` extended to `L`/`T` (D-17).
Extension A' (2026-08-24, D-19..D-25) is strictly additive on top: 79 more
rows, BT-Settl AGSS2009 cool giants (T_eff 2300-3400 K x log g -0.5..+2.0,
plus an additive 3500-3600 K subset), filling the log g < 2.5 gravity hole
Extension A left below 3,500 K; NMODELS 3,987 -> 4,066 once a production
build is run with `agss_download_dir` given. No existing row is culled,
renamed or revalued (D-20), and CLASSMAP/SUBCLASS_LEGEND do not change.
See `docs/sps_decisions.md` D-11 through D-25 for the full reasoning; this
docstring only summarises where each decision is implemented.


DESIGN DECISIONS
----------------

Aperture axis (D-4). sps is aperture-independent (models.conf:
`aperture_dependent = no`), so VALUES/UNCERTAINTIES keep a size-1 aperture axis
-- matching yso's own stellar.fits precedent for aperture-independent data --
rather than dropping the axis entirely. The single aperture value is
`constants.POINT_SOURCE_APERTURE_AU`, a structural sentinel meaning "no
aperture dependence", not a size.

DISTANCE (D-3). A DISTANCE header is written, but it is a plug value, not a
real model distance assumption. sedfitter's own docs classify this Kurucz
photosphere package as a "distance-independent / unscaled" model set: fitting
uses a free scale factor (combined with Av) rather than a physical distance, so
the underlying flux values were never meant to represent a specific distance in
the first place. SEDCube.read() (sedfitter/sed/cube.py) unconditionally
requires a DISTANCE keyword to open the file at all, so we write one purely for
structural/loadability compatibility -- it carries no scientific meaning for
this model set and should never be used for distance scaling. The value is
`constants.REFERENCE_DISTANCE_CM`, the exact literal every library in this
project and the upstream Robitaille release carry.

Validity mask. PRIMARY also carries a (n_models,) int array that SEDCube.read()
interprets as a per-model validity mask (`hdulist[0].data.astype(bool)`). All
3808 Kurucz models are treated as valid here -- unlike yso, sps has no known
invalid/placeholder rows to flag.

Wavelength order (D-2, project-level -- see `model_directory_format.md` Sec. 6.1).
WAVELENGTH/FREQUENCY are stored DESCENDING, not the per-model seds/*.fits.gz
files' native ascending order. `sedfitter.sed.cube.SEDCube.read(order='nu')`
has a real bug for ascending-stored files: it reverses the 1D wavelength array
during reordering but not the flux cube's wavelength axis, silently mismatching
wavelength and flux (caught with a discriminating test: the coolest Kurucz
model's real zero-flux transition wavelength came out 0.0091um instead of the
correct 0.0885um). Storing descending means frequency is already ascending, so
the buggy reorder branch never fires. Independently confirmed against
Robitaille's own yso/s---s-i/flux.fits, which uses this same convention.

models.conf is WRITTEN via `model_io.write_models_conf`, not copied. It used to
be copied verbatim, on the argument that it was Robitaille's own file and D-1's
untouched-copy provenance claim covered it. The cost of that was a conf with no
`version` key, which has two consequences, both now resolved by writing it:

  * `Models.read` took `_read_version_1`, which never opens flux.fits. The
    descending-wavelength fix below therefore protected a file sedfitter never
    read during fitting. Under version 2 the cube IS opened -- via
    `SEDCube.read(order='nu')`, whose default order is exactly the buggy path
    the convention exists to sidestep -- so the fix finally does something.
  * `convolve_model_dir` took `_convolve_model_dir_1`, which globs
    `seds/*.fits.gz`; the curated root has none, so convolved/ was not
    regenerable in place at all. Version 2 routes it to
    `_convolve_model_dir_2`, which reads flux.fits.

Measured before switching: v1 and v2 build bit-identical model flux arrays
(3808x8) because `_read_version_2` still reads convolved/{band}.fits for named
filters -- the cube is only used for its model count. Fitted chi2 agrees to
1.2e-05, from v2 storing fluxes in a float32 memmap where v1 uses a float64
array, and the best-fit model was unchanged across every source tested. So
version 2 changes which files sedfitter opens, not what it computes.

parameters.fits is carried through with MODEL_NAME's trailing padding stripped
(see write_stripped_parameters_fits): the 2008 file space-pads its 30A field,
astropy NUL-pads ours, and `convolve.py:142` compares the two RAW. Without the
strip, convolve_model_dir refuses the directory and convolved/ can never be
regenerated from the cube -- which matters because the curated root has no seds/
and the upstream tree is slated for retirement. Only the padding differs; every
parameter value is bit-identical.

The provenance claim narrows accordingly: convolved/ contents are still
byte-for-byte upstream artifacts; models.conf is now this project's file,
carrying the upstream `name` string but not its comment banner; parameters.fits
is upstream DATA, no longer an upstream FILE.

convolved/ is CONVOLVED HERE, not copied (D-9). It used to be byte-for-byte
copies of the 2008 distribution's own band files, renamed 2J/2H/2K -> J/H/Ks.
That left SPS as the only library not convolved on this project's filter
curves, and the difference is not benign: regenerating gives -1.95% (H) to
+3.42% (M1) against the shipped 2008 values, with ~0 per-model scatter. A
uniform offset would be absorbed by sedfitter's free scale factor; this one is
differential, i.e. a COLOUR error of roughly 0.06 mag in H-M1, and the 5-class
scheme compares P(c|F) ACROSS libraries. So a source's chi2 against SPS versus
against GAL would have differed partly because of whose filter curves were
used. Convolving here puts all five libraries on one photometric footing.

FILTWAV is therefore BANDS[b].wvl_effective_um via build_sedfitter_filter --
load-bearing rather than cosmetic, since FILTWAV propagates to
models.wavelengths and thence to av_law, setting each band's extinction
coefficient. It must be identical per band across all five libraries.

Consequence: NOTHING in the curated root is a byte-for-byte upstream artifact
any more. models.conf became ours at D-6, parameters.fits at D-7, convolved/
here. D-1 is fully superseded. The underlying DATA is still entirely upstream
-- the SEDs, the grid and the parameter values are untouched Kurucz.

Per-model I/O in the model loop (Sec. 3.5 exemption). `build_sps_flux_fits`
deliberately opens 3,808 gzipped FITS files sequentially inside the model loop
-- the shape Sec. 3.6 otherwise prohibits -- because the source distribution
stores exactly one gzip per model and offers no combined form to read instead.
The loop pre-allocates the cube at its final size on the first model and does a
single write at the end, so the cost is ~60 s once per rebuild, not repeated
allocation. Deliberately not batched or parallelised because the run is
one-shot and the wall time is already dominated by gzip decompression.

Internal guard. The loop asserts that every model shares the first model's
wavelength grid and raises ValueError otherwise. This is the library's only
internal consistency check; a source distribution with a heterogeneous grid
would make the (n_models, 1, n_wav) cube meaningless.
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

from sesnaimpute.sed_models import paths
from sesnaimpute.sed_models.constants import (
    CONVOLVED_BAND_NAMES, C_UM_S, LOGD_STEP, POINT_SOURCE_APERTURE_AU,
    REFERENCE_DISTANCE_CM, SPEED_OF_LIGHT_CM_S,
    LIBRARY_SAMPLING_SIGMA_LOG_VECTOR as _LIBRARY_SAMPLING_SIGMA_LOG_VECTOR,
)
from sesnaimpute.sed_models.curate import curve_of_growth as cog
from sesnaimpute.sed_models.curate import model_io
from sesnaimpute.sed_models.curate.model_convolution import build_convolved_bands
from sesnaimpute.sed_models.curate.model_io import write_flux_cube, write_models_conf

# READ-ONLY use of sed_models_register (owner's library specification,
# 2026-10-10): `density.build_quotient_space` is the project-wide,
# library-independent machinery that turns a sigma_log vector into the
# fitter's noise length SIGEFF -- it depends only on sigma_log and the
# two extinction laws, never on a library's own grid -- so the 1-sigma
# class-library sampling below reproduces exactly the space (and SIGEFF)
# any register would use, WITHOUT reading, writing or rebuilding a
# register. `density.py` imports nothing from `sed_models_curate`, so
# this import has no cycle back to this module.
#
# `sed_models_register.noise` (the catalogue-derived sigma_log) is
# DELIBERATELY NOT imported here any more: the owner's 2026-10-10 ruling
# retired it as this sampling's input in favour of
# `constants.SURVEY_CALIB_SIGMA_LOG_VECTOR` (see
# `build_sps_quotient_space`) -- its raw files have been deleted, so it
# is a stale artefact.
from sesnaimpute.sed_models.register import density as _register_density

SED_FILENAME_SUFFIX = "_sed.fits.gz"

# MODEL_NAME cube width. 30A, NOT the project-wide 34A -- and this is a
# DECISION, not drift (docs/model_library_standard.md §1: a deviation with a
# written reason is conformant).
#
# SPS is a copy-repack library: parameters.fits is copied verbatim from the 2008
# Robitaille distribution and carries 30A. Writing flux.fits at 34A would break
# MODEL_NAME width uniformity *within this directory* -- the shared validator's
# C10 check -- unless parameters.fits were also rewritten, which would mean
# regenerating an upstream artifact purely to widen a column. Upstream names are
# at most 17 characters, so nothing truncates either way, and no consumer reads
# MODEL_NAME across libraries (the five classes are fit separately, never
# co-fit). The uniformity that matters is within a model directory, and 30A
# delivers it.
#
# Tried the alternative and measured it: setting 34A here makes the validator
# report `C10: MODEL_NAME width differs across cubes: flux.fits 34A,
# parameters.fits 30A`.
#
# convolved/*.fits is 30A for an unrelated upstream reason: sedfitter writes
# those names itself as astype('S30').
MODEL_NAME_COLUMN_FORMAT = "30A"

DISTANCE_COMMENT = "FAKE plug value, not real -- see docstring"

# The upstream `name` value, carried into the models.conf this module now
# WRITES (see module docstring's "models.conf is WRITTEN" section) rather
# than copies.
MODELS_CONF_NAME = "Kurucz stellar photospheres"

# CONVMETH header note stamped onto convolved/*.fits (D-9): sps is convolved
# HERE, not copied, same as galaxy_curate.CONVMETH_NOTE's pattern.
CONVMETH_NOTE = "sedfitter convolve_model_dir, Kurucz photospheres"

# The 8 SESNA bands to convolve, in constants.BANDS/CONVOLVED_BAND_NAMES
# order -- sps emits the project-wide band set like every other library
# (D-9), not the upstream distribution's 46-band set.
CONVOLVED_BANDS_TO_BUILD = CONVOLVED_BAND_NAMES


# ====================================================================
# Extension A (D-11..D-18): common wavelength grid, CK03 interpolation,
# CIFIST dilution/renormalisation/rebin, red-end Planck tail, splice.
# ====================================================================
# The machinery below is a clean port of
# `SESNA_Complete/claude code/sps_model/splice_study/splice_lib.py`, the
# reference implementation the design study (sps_decisions.md D-11..D-15)
# was measured against -- ported rather than imported so this module has no
# dependency on a scratch study directory.

# D-13: log-uniform, R = lambda/Delta-lambda = 300, over 0.009-1e6 um --
# 5,559 points. Chosen because it is the first candidate (of R in
# {100,200,300,500,1000} plus a hybrid grid) with a >=3x margin on the 0.2%
# band-error target (worst measured 0.061%) and is at or above the
# INHERITED 1,321-point grid's resolution everywhere, so no CK03 information
# is lost anywhere in the cube. See D-13 for the full candidate table.
GRID_R = 300
GRID_WAVELENGTH_MIN_UM = 0.009
GRID_WAVELENGTH_MAX_UM = 1e6

# D-14: CIFIST surface fluxes are diluted to 1 Rsun at 1 kpc, the same
# convention CK03 rows already carry (measured there, not assumed --
# R/Rsun = 1.0003 +/- 0.003 across all 3,808 CK03 rows). R_SUN_CM is the
# literal IAU nominal value; REFERENCE_DISTANCE_CM is the project-wide
# ~1 kpc constant every library's DISTANCE header carries.
SOLAR_RADIUS_CM = 6.957e10
CIFIST_DILUTION_MJY_PER_CGS_FNU = (SOLAR_RADIUS_CM / REFERENCE_DISTANCE_CM) ** 2 * 1e26

# D-15: keep-rule gate. A CIFIST model is kept only if its NATIVE spectrum
# (before any renormalisation/dilution/tail) reaches lambda >= 18.0 um (the
# M1 blue edge -- below this the entire M1 band response would be
# extrapolated, which measures the extrapolation policy rather than the
# model) AND its native bolometric closure int F_lambda dlambda / (sigma
# Teff^4) lies in [0.8, 1.2] (the addendum's closure gate -- the coolest,
# lowest-gravity corner of the grid did not converge, a 20-48% deficit that
# is not a normalisation convention). Measured on the full fetched grid:
# 195 - 1 (lambda_max) - 9 (closure) = 185 kept.
CIFIST_KEEP_LAMBDA_MAX_MIN_UM = 18.0
CIFIST_CLOSURE_MIN = 0.8
CIFIST_CLOSURE_MAX = 1.2

# D-15: the red-end Planck tail's anchor window (the mean of the last 1 um
# of real data is what the scaled B_nu(Teff) is normalised to match) and the
# internal sampling resolution used to generate the tail's own synthetic
# wavelength points before it is handed to the SAME flux-conserving rebin
# used for the real data. The tail resolution is deliberately much finer
# than GRID_R=300 -- B_nu is smooth, but D-13 already measured that
# rebinning a too-coarsely-sampled SOURCE reconstructs a piecewise-constant
# spectrum on the destination grid (the -15% CK03-rebin error), and the tail
# is cheap to oversample (tens of thousands of points, not millions).
CIFIST_PLANCK_TAIL_WINDOW_UM = 1.0
CIFIST_PLANCK_TAIL_GRID_R = 5000

# CGS Planck-function constants, local to this module (matches
# STEFAN_BOLTZMANN_CGS's precedent above: used only inside the tail
# extrapolation's B_nu shape function, not a project-wide quantity).
_PLANCK_H_ERG_S = 6.62607015e-27
_PLANCK_K_ERG_K = 1.380649e-16

# Speed of light, Angstrom/s -- C_UM_S (um/s) * 1e4, kept as its own literal
# so the F_lambda -> F_nu conversion reads as the textbook identity
# F_nu = F_lambda * lambda_A^2 / c_A rather than requiring the reader to
# mentally rescale a um/s constant.
_C_ANGSTROM_PER_S = C_UM_S * 1e4


# ====================================================================
# classmap.fits -- class / subclass vocabulary
# ====================================================================
# CLASS id, coordinated across the five libraries rather than chosen here:
# GUTERMUTH_LABELS.abbrev where a library maps to exactly one label (H2S,
# PAHC), pop_abbrev where it spans several (STAR, GAL, YSO).
MODEL_CLASS = "STAR"

CLASS_LEGEND = (
    ("STAR", "Bare stellar photosphere (no disk, envelope or PAH excess)"),
)

# --- the Teff -> spectral letter scale ------------------------------
# Anchors transcribed from Mamajek's "Modern Mean Dwarf Stellar Color and
# Effective Temperature Sequence", version 2022.04.16, whose own header asks
# that Pecaut & Mamajek (2013, ApJS 208, 9) be cited. Values are the Teff of
# each letter's hottest (X0V) and coolest (X9V) subtype.
#
# The published sequence leaves a GAP between letters -- B9V is 10700 K and
# A0V is 9700 K -- so a boundary has to be placed inside it. We take the
# MIDPOINT of the two adjacent anchors. That is a small declared convention
# on top of published numbers, and the provenance header says so.
MAMAJEK_ANCHORS_K = {
    "O9V": 33300, "B0V": 31400, "B9V": 10700, "A0V": 9700,
    "A9V": 7400, "F0V": 7220, "F9V": 6050, "G0V": 5930,
    "G9V": 5380, "K0V": 5270, "K9V": 3930, "M0V": 3850,
    # D-17 (Extension A): four more anchors from the SAME table/version,
    # extending the sequence down through the CIFIST grid's cool end
    # (1,200-3,900 K). M0V above is unchanged; these four are new.
    "M9V": 2380, "L0V": 2270, "L9V": 1370, "T0V": 1255,
}

# (letter, lower Teff bound inclusive, upper Teff bound exclusive)
# Built from the anchors so the arithmetic is visible rather than a table of
# magic numbers: a reader can check 10200 = (10700 + 9700) / 2.
def _letter_edges():
    a = MAMAJEK_ANCHORS_K
    return {
        "O/B": (a["O9V"] + a["B0V"]) / 2,   # 32350
        "B/A": (a["B9V"] + a["A0V"]) / 2,   # 10200
        "A/F": (a["A9V"] + a["F0V"]) / 2,   #  7310
        "F/G": (a["F9V"] + a["G0V"]) / 2,   #  5990
        "G/K": (a["G9V"] + a["K0V"]) / 2,   #  5325
        "K/M": (a["K9V"] + a["M0V"]) / 2,   #  3890
        "M/L": (a["M9V"] + a["L0V"]) / 2,   #  2325  -- D-17
        "L/T": (a["L9V"] + a["T0V"]) / 2,   #  1312.5 -- D-17
    }


SPECTRAL_LETTER_EDGES_K = _letter_edges()

# Coolest-first. D-17 extends this with L and T; T is now the open-ended
# terminal (coolest) bucket, taking over that role from M.
SPECTRAL_LETTERS = ("T", "L", "M", "K", "G", "F", "A", "B", "O")

# THE AUTHORITY for what subclasses exist here. Subclasses are authored, not
# discovered: Pecaut & Mamajek supply the Teff scale, but deciding that the
# NINE MK letters -- rather than subtypes, or letter+luminosity class -- are
# this library's subclasses is this project's decision. Changing this tuple
# is a definitional change, not a cosmetic one. D-17 (Extension A) adds L
# and T, and closes M's lower bound now that it is no longer the terminal
# bucket.
#
# The numeric ranges live in these DESCRIPTIONS rather than in a header card:
# a HIERARCH card has ~51 characters and a table row has no limit, so this is
# where the cuts travel with the file.
SUBCLASS_LEGEND = (
    ("O", "O-type: T_eff >= 32350 K (dwarf-sequence scale)"),
    ("B", "B-type: 10200 <= T_eff < 32350 K (dwarf-sequence scale)"),
    ("A", "A-type: 7310 <= T_eff < 10200 K (dwarf-sequence scale)"),
    ("F", "F-type: 5990 <= T_eff < 7310 K (dwarf-sequence scale)"),
    ("G", "G-type: 5325 <= T_eff < 5990 K (dwarf-sequence scale)"),
    ("K", "K-type: 3890 <= T_eff < 5325 K (dwarf-sequence scale)"),
    ("M", "M-type: 2325 <= T_eff < 3890 K (dwarf-sequence scale)"),
    ("L", "L-type: 1312.5 <= T_eff < 2325 K (dwarf-sequence scale)"),
    ("T", "T-type: T_eff < 1312.5 K (dwarf-sequence scale)"),
)


def assign_spectral_letter(t_eff):
    """MK spectral letter for each T_EFF, on the DWARF sequence regardless of
    log g. Returns a string array.

    The dwarf-scale approximation is stated, and it is cheap at LETTER
    granularity in a way it would not be at subtype granularity: K0V is
    5270 K against K0III's ~4660 K, a 600 K error, yet both still land in K.
    The approximation can slip a model by one letter, and only near a
    boundary -- never further. Typing giants properly would need a second
    Teff table for luminosity classes III/I, and the log g -> luminosity
    class cut that would select between them has no published source (D-10).

    D-17 (Extension A) extends the walk with L and T: T is now the
    open-ended terminal (coolest) bucket, and M's range is closed at its
    bottom (2325 K) rather than running to T_eff=0.

    Raises on a non-finite T_EFF: having committed to subclassing this
    library, every model carries one.
    """
    t = np.asarray(t_eff, dtype=float)
    if not np.isfinite(t).all():
        raise ValueError(
            f"{int((~np.isfinite(t)).sum())} model(s) have a non-finite T_EFF and "
            "cannot be typed; every model in this library must carry a SUBCLASS.")
    edges = SPECTRAL_LETTER_EDGES_K
    # HOTTEST-FIRST is load-bearing. Each test is `t < upper`, so a cooler
    # letter's condition is also satisfied by every hotter one; walking cool
    # to hot would let each later test overwrite the correct cooler label and
    # collapse the whole grid onto one letter. O is the default because it is
    # the only letter with no upper bound; T (D-17) is now the analogous
    # terminal case at the cool end -- the last test applied, so nothing
    # overwrites it.
    out = np.full(t.shape, "O", dtype=object)
    for letter, upper in (("B", edges["O/B"]), ("A", edges["B/A"]), ("F", edges["A/F"]),
                          ("G", edges["F/G"]), ("K", edges["G/K"]), ("M", edges["K/M"]),
                          ("L", edges["M/L"]), ("T", edges["L/T"])):
        out[t < upper] = letter
    return out.astype(str)



def _model_names_in_order(sps_root):
    """Model names in parameters.fits row order."""
    with fits.open(sps_root / "parameters.fits") as hdul:
        return [name.strip() for name in hdul[1].data["MODEL_NAME"]]


def _read_model_sed(sps_root, model_name, sed_suffix=SED_FILENAME_SUFFIX):
    """Return (wavelength_um, frequency_hz, flux_mjy, flux_err_mjy) for one model."""
    path = sps_root / "seds" / f"{model_name}{sed_suffix}"
    with fits.open(path) as hdul:
        wavelengths = hdul["WAVELENGTHS"].data
        seds = hdul["SEDS"].data
        return (
            wavelengths["WAVELENGTH"],
            wavelengths["FREQUENCY"],
            seds["TOTAL_FLUX"][0],
            seds["TOTAL_FLUX_ERR"][0],
        )


def build_sps_flux_fits(sps_root, output_path, overwrite=True,
                        sed_suffix=SED_FILENAME_SUFFIX,
                        aperture_au=POINT_SOURCE_APERTURE_AU,
                        distance_cm=REFERENCE_DISTANCE_CM,
                        name_format=MODEL_NAME_COLUMN_FORMAT):
    """Pre-Extension-A stages 2-3: combine sps/seds/*.fits.gz into one
    flux.fits on the INHERITED 1,321-point native grid (yso schema), with no
    CIFIST rows and no splice.

    Superseded as the production path by `build_sps_merged_rows` /
    `build_sps_flux_fits_from_rows` (D-13: the common R=300 grid; D-11/D-12:
    the CIFIST splice), which `curate_sps_model_set` now calls. Kept here,
    still exercised by nothing else in this module, as a plain CK03-only
    repack -- e.g. for reproducing the pre-Extension-A cube, or as a
    reference to diff the new pipeline's CK03 branch against on the
    overlapping rows.
    """
    sps_root = Path(sps_root)
    model_names = _model_names_in_order(sps_root)
    n_models = len(model_names)

    ref_wavelength = None
    ref_frequency = None
    flux = None
    flux_err = None

    for i, model_name in enumerate(model_names):
        wavelength, frequency, model_flux, model_flux_err = _read_model_sed(
            sps_root, model_name, sed_suffix=sed_suffix
        )

        if ref_wavelength is None:
            ref_wavelength = wavelength
            ref_frequency = frequency
            n_wav = len(ref_wavelength)
            # (n_models, n_ap, n_wav), matching yso flux.fits/stellar.fits convention
            flux = np.empty((n_models, 1, n_wav), dtype=np.float32)
            flux_err = np.empty((n_models, 1, n_wav), dtype=np.float32)
        elif not np.array_equal(wavelength, ref_wavelength):
            raise ValueError(f"{model_name}: wavelength grid does not match reference")

        flux[i, 0, :] = model_flux
        flux_err[i, 0, :] = model_flux_err

    # reverse to descending wavelength / ascending frequency -- avoids a real
    # SEDCube.read(order='nu') reorder bug for ascending-stored files; see
    # module docstring
    ref_wavelength = ref_wavelength[::-1]
    ref_frequency = ref_frequency[::-1]
    flux = flux[:, :, ::-1]
    flux_err = flux_err[:, :, ::-1]

    return write_flux_cube(
        output_path,
        names=np.array(model_names),
        wave_um_desc=ref_wavelength,
        freq_hz_desc=ref_frequency,
        values=flux,
        uncertainties=flux_err,
        distance_cm=distance_cm,
        distance_comment=DISTANCE_COMMENT,
        apertures_au=np.array([aperture_au]),
        # valid=None -> all ones: sps has no invalid/placeholder rows to flag
        valid=None,
        name_format=name_format,
        overwrite=overwrite,
    )


# --------------------------------------------------------------------
# D-13: the common grid, and the two per-source ways of putting a row on it
# --------------------------------------------------------------------

def _loggrid(wave_min_um, wave_max_um, r):
    """Log-uniform wavelength grid, ASCENDING, R = lambda/Delta-lambda = r,
    from wave_min_um to wave_max_um inclusive. Ported from
    splice_lib.loggrid; shared by the common grid (D-13) and the CIFIST
    red-end tail's internal sampling grid (D-15)."""
    n = int(np.ceil(np.log(wave_max_um / wave_min_um) * r)) + 1
    return np.exp(np.linspace(np.log(wave_min_um), np.log(wave_max_um), n))


def build_common_wavelength_grid(r=GRID_R, wave_min_um=GRID_WAVELENGTH_MIN_UM,
                                 wave_max_um=GRID_WAVELENGTH_MAX_UM):
    """D-13: the one common wavelength grid every row of the merged
    CK03+CIFIST cube is placed on -- log-uniform, R = lambda/Delta-lambda =
    `r`, from `wave_min_um` to `wave_max_um`. Defaults give 5,559 points.

    Returns (wave_um_desc, freq_hz_desc): `wave_um_desc` is STRICTLY
    DESCENDING (D-2 / model_directory_format.md Sec.5); `freq_hz_desc` is
    the frequency array in the SAME row order (so it is numerically
    ascending) -- matching `write_flux_cube`'s own naming convention for
    these two arrays.
    """
    wave_asc = _loggrid(wave_min_um, wave_max_um, r)
    wave_desc = wave_asc[::-1]
    freq_hz_desc = C_UM_S / wave_desc
    return wave_desc, freq_hz_desc


def interp_loglog_zero_preserving(nu_out_asc, nu_in_asc, f_in):
    """Log-log interpolation of F_nu(nu) from (nu_in_asc, f_in) onto
    nu_out_asc -- both ASCENDING in frequency. Used for the CK03 branch of
    D-13 (rebinning a source at CK03's native R~4 above 10 um reconstructs a
    piecewise-CONSTANT spectrum and costs -15% in M1; interpolation is the
    correct operation when the target is finer than the source, which holds
    for CK03 everywhere -- see D-13's "why interpolate CK03 rather than
    rebin").

    Outside `nu_in_asc`'s coverage -> 0.0 (model_directory_format.md
    Sec.10's wavelength-axis convention). Where EITHER of the two
    bracketing input samples is exactly zero -- the CK03 blue-cutoff, a
    real UV opacity edge in cool models -- the output is 0.0 rather than
    log-interpolated, so a hard cutoff in the source stays a hard cutoff
    on the new grid instead of smearing across one destination bin.
    """
    nu_out = np.asarray(nu_out_asc, dtype=float)
    nu_in = np.asarray(nu_in_asc, dtype=float)
    f_in = np.asarray(f_in, dtype=float)
    out = np.zeros_like(nu_out)
    inside = (nu_out >= nu_in[0]) & (nu_out <= nu_in[-1])
    if not np.any(inside):
        return out

    xo = nu_out[inside]
    idx = np.clip(np.searchsorted(nu_in, xo, side="right") - 1, 0, len(nu_in) - 2)
    lo, hi = idx, idx + 1
    f_lo, f_hi = f_in[lo], f_in[hi]
    both_pos = (f_lo > 0) & (f_hi > 0)

    result = np.zeros(xo.shape)
    if np.any(both_pos):
        x0 = np.log(nu_in[lo[both_pos]])
        x1 = np.log(nu_in[hi[both_pos]])
        y0 = np.log(f_lo[both_pos])
        y1 = np.log(f_hi[both_pos])
        t = (np.log(xo[both_pos]) - x0) / (x1 - x0)
        result[both_pos] = np.exp(y0 + t * (y1 - y0))

    out[inside] = result
    return out


def build_ck03_row_on_grid(native_freq_hz, native_flux_mjy, common_freq_hz_desc):
    """One CK03 row via log-log interpolation onto the common grid (D-13).

    `native_freq_hz`/`native_flux_mjy` are `_read_model_sed`'s native-order
    arrays (ascending wavelength / DESCENDING frequency). `common_freq_hz_desc`
    is `build_common_wavelength_grid`'s frequency array -- ascending in VALUE,
    in flux.fits's descending-wavelength ROW order. Returns values in that
    same row order (i.e. ready to drop straight into a flux.fits row).
    """
    nu_in_asc = np.asarray(native_freq_hz, dtype=float)[::-1]
    f_in_asc = np.asarray(native_flux_mjy, dtype=float)[::-1]
    return interp_loglog_zero_preserving(
        np.asarray(common_freq_hz_desc, dtype=float), nu_in_asc, f_in_asc)


def _read_ck03_parameters(source_root):
    """(names, teff, logg, z) straight from source_root/parameters.fits,
    MODEL_NAME stripped, in the source file's own row order.

    `source_root` is `downloads/ck03`, the manual-acquisition archive --
    not a file this package writes, so its column names are the upstream
    distribution's own, `LOG[G]` and `[Z/H]`, read as such. The project's
    own parameters.fits (`LOGG`/`Z_H`, `write_stripped_parameters_fits`)
    is authored fresh from these values.
    """
    with fits.open(Path(source_root) / "parameters.fits") as hdul:
        table = hdul[1].data
        names = np.char.strip(np.asarray(table["MODEL_NAME"]).astype(str))
        teff = np.asarray(table["T_EFF"], dtype=float)
        logg = np.asarray(table["LOG[G]"], dtype=float)
        z = np.asarray(table["[Z/H]"], dtype=float)
    return names, teff, logg, z


def build_ck03_grid_rows(source_root, common_freq_hz_desc,
                         sed_suffix=SED_FILENAME_SUFFIX, verbose=True):
    """Stage 2 (CK03 half): every source_root/seds/*.fits.gz row, log-log
    interpolated onto the common grid (D-13). Returns a list of dicts
    {name, teff, logg, z, source='CK03', values_desc}, in
    source_root/parameters.fits row order (the splice step sorts the
    merged set; this function's own order does not matter downstream)."""
    source_root = Path(source_root)
    names, teff, logg, z = _read_ck03_parameters(source_root)
    rows = []
    for i, name in enumerate(names):
        _, freq, flux, _ = _read_model_sed(source_root, name, sed_suffix=sed_suffix)
        values_desc = build_ck03_row_on_grid(freq, flux, common_freq_hz_desc)
        rows.append(dict(name=name, teff=float(teff[i]), logg=float(logg[i]),
                         z=float(z[i]), source="CK03", values_desc=values_desc))
    if verbose:
        print(f"    CK03: {len(rows)} rows log-log interpolated onto the common grid")
    return rows


# --------------------------------------------------------------------
# D-14/D-15: CIFIST rows -- air->vac, dilution+renormalisation, Planck
# tail, flux-conserving rebin
# --------------------------------------------------------------------

def air_to_vacuum_angstrom(lam_air_angstrom):
    """Ciddor (1996) air->vacuum wavelength conversion (D-11/D-18's noted
    SVO air convention), applied only where the air convention is defined
    (>2000 A); identity elsewhere. A ~2.7e-4 relative shift, moving no band
    flux at the 0.01% level at R=300 (D-18) -- applied anyway because it is
    free and the task this module implements asks for it explicitly.
    Ported from splice_study/splice_lib.py:air_to_vac."""
    lam = np.asarray(lam_air_angstrom, dtype=float)
    s2 = (1e4 / np.maximum(lam, 2000.0)) ** 2
    n = 1 + 0.05792105 / (238.0185 - s2) + 0.00167917 / (57.362 - s2)
    return np.where(lam > 2000.0, lam * n, lam)


def flambda_to_fnu_cgs(f_lambda_cgs_per_angstrom, wave_angstrom):
    """F_nu [erg/s/cm2/Hz] = F_lambda [erg/s/cm2/A] * lambda[A]^2 / c[A/s] --
    the standard flux-per-unit-frequency identity, with c in Angstrom/s so
    it applies directly to CIFIST's native A/erg-s-cm2-A units."""
    wave_a = np.asarray(wave_angstrom, dtype=float)
    return np.asarray(f_lambda_cgs_per_angstrom, dtype=float) * wave_a ** 2 / _C_ANGSTROM_PER_S


def _planck_bnu_shape_cgs(nu_hz, teff_k):
    """B_nu(nu, T) shape, cgs Planck function -- 2 h nu^3/c^2 / (exp(h nu/kT)-1).
    Used ONLY as a fitted SHAPE for the D-15 red-end tail: the tail's
    amplitude `s` is fit to match real data in whatever units the caller's
    F_nu array carries (mJy, post-dilution), so this function's own
    overall normalisation is irrelevant and cancels out of the fit."""
    x = np.clip(_PLANCK_H_ERG_S * np.asarray(nu_hz, dtype=float)
               / (_PLANCK_K_ERG_K * teff_k), 1e-12, 700.0)
    return (2 * _PLANCK_H_ERG_S * np.asarray(nu_hz, dtype=float) ** 3
           / SPEED_OF_LIGHT_CM_S ** 2) / np.expm1(x)


def _append_planck_tail(wave_um_vac_asc, flux_mjy_asc, teff_k,
                        window_um=CIFIST_PLANCK_TAIL_WINDOW_UM,
                        tail_grid_r=CIFIST_PLANCK_TAIL_GRID_R,
                        wave_max_um=GRID_WAVELENGTH_MAX_UM):
    """D-15: append a scaled Planck tail beyond a CIFIST spectrum's last
    native sample, out to wave_max_um. `wave_um_vac_asc`/`flux_mjy_asc` are
    ascending, post-dilution (mJy). `s` is fixed so the mean of
    B_nu(T_eff) over the last `window_um` of real data equals the data's
    own mean F_nu there -- the "scaled Planck B_nu(Teff)" policy measured
    to be accurate to <=0.1% in M1 for any model reaching 23 um (D-15),
    beating both flavours of Rayleigh-Jeans extrapolation tested in the
    design study.

    Returns (wave_um_asc, flux_mjy_asc) -- native data plus the tail,
    strictly increasing in wavelength, ready for `flux_conserving_rebin_wavelength`.
    """
    wave = np.asarray(wave_um_vac_asc, dtype=float)
    flux = np.asarray(flux_mjy_asc, dtype=float)
    wave_max_native = wave[-1]

    window_mask = wave >= (wave_max_native - window_um)
    nu_window = C_UM_S / wave[window_mask]
    mean_data = float(np.mean(flux[window_mask]))
    mean_bnu = float(np.mean(_planck_bnu_shape_cgs(nu_window, teff_k)))
    s = mean_data / mean_bnu

    if wave_max_native >= wave_max_um:
        return wave, flux

    # [1:] drops the grid's own first point, which equals wave_max_native
    # exactly -- keeps the combined array STRICTLY increasing.
    tail_wave = _loggrid(wave_max_native, wave_max_um, tail_grid_r)[1:]
    tail_nu = C_UM_S / tail_wave
    tail_flux = s * _planck_bnu_shape_cgs(tail_nu, teff_k)
    return np.concatenate([wave, tail_wave]), np.concatenate([flux, tail_flux])


def _bin_edges_nu_asc(nu_asc):
    """Bin edges matching sedfitter's Filter.rebin convention: arithmetic
    midpoints in nu, with the first/last point's own value as the outer
    edge. Ported from splice_lib.bin_edges_nu."""
    e = np.empty(len(nu_asc) + 1)
    e[1:-1] = 0.5 * (nu_asc[:-1] + nu_asc[1:])
    e[0] = nu_asc[0]
    e[-1] = nu_asc[-1]
    return e


def flux_conserving_rebin_wavelength(wave_src_asc_um, fnu_src, wave_dst_asc_um):
    """Rebin F_nu from a fine source grid onto wave_dst_asc_um (both
    ASCENDING in wavelength), conserving int F_nu dnu inside each
    destination bin. Bins are the arithmetic-midpoint-in-nu convention
    sedfitter's own `Filter.rebin` attributes to each destination point
    (D-13). Destination bins entirely outside the source's coverage ->
    exact 0.0 (model_directory_format.md Sec.10).

    Ported from splice_study/splice_lib.py:flux_conserving_rebin. Used for
    the CIFIST branch of D-13: CIFIST is natively far finer (R up to
    ~5e5) than the common grid, so rebinning -- not interpolating -- is the
    correct operation (the reverse of the CK03 case; see
    `interp_loglog_zero_preserving`'s docstring).
    """
    nu_s = C_UM_S / np.asarray(wave_src_asc_um, dtype=float)[::-1]
    f_s = np.asarray(fnu_src, dtype=float)[::-1]
    nu_d = C_UM_S / np.asarray(wave_dst_asc_um, dtype=float)[::-1]
    edges = _bin_edges_nu_asc(nu_d)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f_s[1:] + f_s[:-1]) * np.diff(nu_s))])
    ec = np.clip(edges, nu_s[0], nu_s[-1])
    c = np.interp(ec, nu_s, cum)
    dnu = np.diff(ec)
    out = np.zeros(len(nu_d))
    good = dnu > 0
    out[good] = np.diff(c)[good] / dnu[good]
    return out[::-1]


def _parse_cifist_ascii_fast(path):
    """Same contract as `_parse_cifist_ascii` (header teff/logg; the
    lambda=0 sentinel row dropped; wavelength grid asserted strictly
    increasing; returns (wavelength_um AIR, f_lambda_cgs, teff, logg)), but
    parsed with pandas' C engine (falling back to `np.loadtxt` if pandas is
    not importable) instead of a pure-Python per-line loop.

    Why this exists (item 9 of the task this module implements): the
    fetched CIFIST files run to ~1.2M lines each; the pure-Python loop in
    `_parse_cifist_ascii` costs seconds per file there, which is fine for
    the handful of files the existing Stage-0 tests touch but would put a
    195-file build well past a "few minutes" target. Measured: ~0.4 s/file
    with pandas' C engine on these files, vs several times that with the
    per-line loop.
    """
    import gzip
    import itertools
    import re

    with gzip.open(path, "rt") as f:
        header_lines = list(itertools.islice(f, 8))

    teff = logg = None
    for line in header_lines:
        m = re.search(r"teff\s*=\s*([\d.]+)", line)
        if m:
            teff = float(m.group(1))
        # D-21 (Extension A'): [-+\d.]+, not [\d.]+ -- AGSS carries
        # `# logg = -0.5 log(cm/s2)` for the entire log g = -0.5 row (17 of
        # the 79 D-19 nodes, exactly the leak corner). The unsigned class
        # cannot match a leading '-', so the search failed outright and this
        # function raised "could not parse teff/logg from header comments"
        # on every one of those files. teff's regex is unchanged -- every
        # CIFIST/AGSS teff is positive -- so this fix cannot affect teff
        # parsing or any existing (non-negative-logg) file's logg parsing.
        m = re.search(r"logg\s*=\s*([-+\d.]+)", line)
        if m:
            logg = float(m.group(1))
    if teff is None or logg is None:
        raise ValueError(f"{path}: could not parse teff/logg from header comments")

    try:
        import pandas as pd
        frame = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                            dtype=float, engine="c")
        wave_a = frame[0].to_numpy(dtype=float)
        flux = frame[1].to_numpy(dtype=float)
    except ImportError:
        data = np.loadtxt(path, comments="#")
        wave_a, flux = data[:, 0].astype(float), data[:, 1].astype(float)

    if wave_a.size and wave_a[0] == 0.0:
        wave_a = wave_a[1:]
        flux = flux[1:]
    if not np.all(np.diff(wave_a) > 0):
        raise ValueError(f"{path}: wavelength grid is not strictly increasing "
                         "after dropping the lambda=0 sentinel row")

    return wave_a * _CIFIST_ANGSTROM_TO_UM, flux, teff, logg


def cifist_model_name(teff, logg, z=0.0):
    """D-16: CIFIST MODEL_NAME -- the CK03 formatter plus a `_cifist`
    suffix (e.g. 'kt01200g+4.5z+0.0_cifist', 24 characters, within the 30A
    cube width). A SUFFIX rather than a prefix so the four census scripts
    that parse names with `re.match(r"kt(\\d+)g([+-][\\d.]+)z([+-][\\d.]+)")`
    (anchored at the start only) still match unchanged."""
    return "kt%05dg%+.1fz%+.1f_cifist" % (int(round(teff)), logg, z)


def _iter_cifist_files(download_dir):
    """*.dat.gz files in download_dir, sorted -- excludes the sibling
    README.md and manifest bookkeeping fetch_btsettl_cifist writes there."""
    return sorted(Path(download_dir).glob("*.dat.gz"))


def build_cifist_grid_rows(download_dir, common_wave_desc_um, verbose=True):
    """Stage 2 (CIFIST half, D-11/D-14/D-15): every download_dir/*.dat.gz
    file -> the D-15 keep-rule gate -> (for kept models) air->vac, F_lambda
    -> F_nu, exact-closure renormalisation, dilution to 1 Rsun @ 1 kpc, a
    Planck red-end tail, and a flux-conserving rebin onto the common grid.

    Returns (kept, skipped):
      kept    -- list of dicts {name, teff, logg, z=0.0, source='CIFIST',
                 values_desc, renorm_factor, closure_native}
      skipped -- list of dicts {file, teff, logg, closure_native,
                 lambda_max_native, reason}
    """
    common_wave_asc = np.asarray(common_wave_desc_um, dtype=float)[::-1]
    kept, skipped = [], []
    files = _iter_cifist_files(download_dir)

    for path in files:
        wave_air_um, f_lambda_native, teff, logg = _parse_cifist_ascii_fast(str(path))
        closure_native = cifist_bolometric_ratio(wave_air_um, f_lambda_native, teff)
        lambda_max_native = float(wave_air_um[-1])

        reasons = []
        if lambda_max_native < CIFIST_KEEP_LAMBDA_MAX_MIN_UM:
            reasons.append(f"lambda_max={lambda_max_native:.2f}um < "
                           f"{CIFIST_KEEP_LAMBDA_MAX_MIN_UM}um")
        if not (CIFIST_CLOSURE_MIN <= closure_native <= CIFIST_CLOSURE_MAX):
            reasons.append(f"native closure {closure_native:.3f} outside "
                           f"[{CIFIST_CLOSURE_MIN},{CIFIST_CLOSURE_MAX}]")
        if reasons:
            skipped.append(dict(file=str(path), teff=teff, logg=logg,
                                closure_native=closure_native,
                                lambda_max_native=lambda_max_native,
                                reason="; ".join(reasons)))
            if verbose:
                print(f"    SKIP {path.name}: {'; '.join(reasons)}")
            continue

        wave_a_air = wave_air_um / _CIFIST_ANGSTROM_TO_UM
        wave_a_vac = air_to_vacuum_angstrom(wave_a_air)
        wave_um_vac = wave_a_vac * _CIFIST_ANGSTROM_TO_UM

        f_nu_surface = flambda_to_fnu_cgs(f_lambda_native, wave_a_vac)

        # D-14 addendum: renormalise to EXACT bolometric closure before
        # dilution, so every row means "a 1 Rsun star of the labelled
        # T_EFF" regardless of how close the native model came to closing.
        renorm_ratio = cifist_bolometric_ratio(wave_um_vac, f_lambda_native, teff)
        renorm_factor = 1.0 / renorm_ratio
        f_nu_surface = f_nu_surface * renorm_factor

        f_nu_mjy_native = f_nu_surface * CIFIST_DILUTION_MJY_PER_CGS_FNU

        wave_full, flux_full = _append_planck_tail(wave_um_vac, f_nu_mjy_native, teff)

        values_asc = flux_conserving_rebin_wavelength(wave_full, flux_full, common_wave_asc)
        values_desc = values_asc[::-1]

        name = cifist_model_name(teff, logg)
        kept.append(dict(name=name, teff=float(teff), logg=float(logg), z=0.0,
                         source="CIFIST", values_desc=values_desc,
                         renorm_factor=renorm_factor, closure_native=closure_native))
        if verbose:
            print(f"    KEEP {path.name}: native closure={closure_native:.3f} "
                 f"renorm factor={renorm_factor:.4f}")

    if verbose:
        print(f"    CIFIST: {len(kept)} kept, {len(skipped)} skipped "
             f"(of {len(files)} fetched)")
    return kept, skipped


# --------------------------------------------------------------------
# D-12: splice -- cull the CK03 rows that exactly duplicate a kept CIFIST row
# --------------------------------------------------------------------

def _splice_key(teff, logg, z):
    """(T_EFF, LOGG, Z_H) rounded to the grids' own native precision
    (whole K; 0.1 dex; 0.01 dex) so a CK03 float32 value and a CIFIST value
    parsed from ascii text compare equal at an exact grid coincidence
    without float-representation noise deciding the answer."""
    return (round(teff), round(logg, 1), round(z, 2))


def splice_ck03_and_cifist(ck03_rows, cifist_rows):
    """D-12: cull the CK03 rows whose (T_EFF, LOGG, Z_H) exactly match a
    KEPT CIFIST row's (T_EFF, LOGG, 0.0) -- CIFIST wins at the six
    coincident nodes (3,500 K, Z 0, log g 2.5...5.0). CK03 is otherwise
    retained through the whole 3,500-3,900 K overlap (D-12: the extension's
    purpose is the 1,200-3,500 K coverage CK03 lacks, not replacement of
    what it already has).

    Returns (merged_rows, culled_ck03_rows). `merged_rows` is sorted
    ASCII-lexicographically by MODEL_NAME -- the one row order every
    emitted file (flux.fits, parameters.fits, classmap.fits, convolved/)
    shares (D-12/D-16).
    """
    cifist_keys = {_splice_key(r["teff"], r["logg"], r["z"]) for r in cifist_rows}
    kept_ck03 = [r for r in ck03_rows
                if _splice_key(r["teff"], r["logg"], r["z"]) not in cifist_keys]
    culled_ck03 = [r for r in ck03_rows
                  if _splice_key(r["teff"], r["logg"], r["z"]) in cifist_keys]
    merged = kept_ck03 + list(cifist_rows)
    merged.sort(key=lambda r: r["name"])
    return merged, culled_ck03


# --------------------------------------------------------------------
# D-19..D-25 (Extension A'): AGSS2009 giant rows -- same CIFIST pipeline
# (D-21), a smoothness closure gate replacing D-15's absolute band (D-22),
# and a strictly-additive three-way merge on top of the CK03+CIFIST splice
# above (D-20).
# --------------------------------------------------------------------

def agss_model_name(teff, logg, z=0.0):
    """D-24: AGSS MODEL_NAME -- the SAME formatter as cifist_model_name,
    with an `_agss` suffix instead of `_cifist` (e.g.
    'kt02300g-0.5z+0.0_agss', 22 characters, well within the 30A cube
    width). A suffix, not a prefix, for the same reason D-16 chose one for
    CIFIST: the census scripts' `re.match(r"kt(\\d+)g([+-][\\d.]+)z([+-][\\d.]+)")`
    anchors only at the start."""
    return "kt%05dg%+.1fz%+.1f_agss" % (int(round(teff)), logg, z)


def _iter_agss_files(download_dir):
    """*.dat.gz files in download_dir, sorted -- mirrors _iter_cifist_files;
    excludes any sibling README/manifest bookkeeping fetch_btsettl_agss
    writes there."""
    return sorted(Path(download_dir).glob("*.dat.gz"))


# D-22: smoothness-gate tuning. T1 measured closure(T_eff) at fixed log g to
# be smooth and nearly flat (spread 0.002 across 2,600-3,500 K at log g
# +2.0) everywhere on the D-19 box, so a quadratic-or-lower fit with a 3
# sigma residual flag is generous -- it exists to catch a genuinely bad file
# (like the (2500,+2.5) 0.34-low outlier T1 found just outside the box), not
# to second-guess the expected low-g flux-radius trend.
AGSS_CLOSURE_SMOOTHNESS_MAX_POLY_ORDER = 2
AGSS_CLOSURE_SMOOTHNESS_SIGMA = 3.0


# Absolute floor on the residual scale used to judge "> 3 sigma" (closure
# units). Two reasons this exists, not just a magic-number safety net:
# (1) T1's measured closure precision never approaches this floor -- the box
# has real point-to-point scatter of 0.002-0.13, so a floor at 1e-4 cannot
# hide a genuine departure; (2) without it, a near-exactly-polynomial
# surface (the box IS nearly flat/linear in T_eff at fixed log g per T1, and
# any degree<=2 SYNTHETIC test fixture built to be smooth is exactly
# polynomial) drives the fit residual std down toward floating-point noise
# (~1e-16), and dividing by that turns an utterly negligible residual into a
# spurious multi-sigma "outlier" -- a real failure mode caught while writing
# this gate's tests.
AGSS_CLOSURE_SMOOTHNESS_RESIDUAL_FLOOR = 1e-4


def _agss_closure_smoothness_gate(entries, max_poly_order=AGSS_CLOSURE_SMOOTHNESS_MAX_POLY_ORDER,
                                  sigma=AGSS_CLOSURE_SMOOTHNESS_SIGMA):
    """D-22: replace D-15's absolute closure band [0.8, 1.2] -- which would
    reject the entire log g <= 0.0 block for a normalisation CONVENTION
    (extended low-gravity models quote flux at an outer radius, not a
    defect; T1) -- with a smoothness check on closure(T_eff) at FIXED log g.

    `entries` -- list of dicts with (at least) 'teff', 'logg',
    'closure_native'. Grouped by log g (rounded to 0.1 dex); within each
    group of n >= 3 points, fits a polynomial of order
    min(max_poly_order, n-1) to closure vs T_eff and flags any point whose
    residual exceeds `sigma` times the group's residual std. A group with
    fewer than 3 points (nothing to fit a curve AND judge a residual
    against) passes through unflagged -- not a case D-19's box ever
    presents (every log g row has 12 or 13 T_eff points).

    Returns the list of FLAGGED entries (each with the original keys plus
    'residual' and 'resid_sigma'), empty when the surface is smooth
    everywhere -- the expected, and only tested-for-shipping, outcome.
    """
    flagged = []
    by_logg = {}
    for e in entries:
        by_logg.setdefault(round(e["logg"], 1), []).append(e)

    for logg, group in sorted(by_logg.items()):
        n = len(group)
        if n < 3:
            continue
        order = min(max_poly_order, n - 1)
        idx = np.argsort([g["teff"] for g in group])
        ordered = [group[i] for i in idx]
        teff = np.array([g["teff"] for g in ordered], dtype=float)
        closure = np.array([g["closure_native"] for g in ordered], dtype=float)
        coeffs = np.polyfit(teff, closure, order)
        resid = closure - np.polyval(coeffs, teff)
        # ROBUST scale estimate (median absolute deviation, not std): a
        # single genuine outlier should not get to inflate the very yardstick
        # used to judge it -- MAD's ~50% breakdown point keeps that from
        # happening the way a plain std would (measured while writing this
        # gate's outlier test: a plain std let one bad node roughly double
        # the scale estimate, nearly hiding itself).
        mad = float(np.median(np.abs(resid)))
        scale = max(1.4826 * mad, AGSS_CLOSURE_SMOOTHNESS_RESIDUAL_FLOOR)
        for entry, r in zip(ordered, resid):
            if abs(r) > sigma * scale:
                flagged.append(dict(entry, residual=float(r), resid_sigma=float(abs(r) / scale)))
    return flagged


def build_agss_grid_rows(download_dir, common_wave_desc_um, verbose=True):
    """Stage 2 (AGSS half, D-19/D-21/D-22): every download_dir/*.dat.gz AGSS
    file, through the SAME pipeline build_cifist_grid_rows uses (D-21) --
    air->vac, F_lambda -> F_nu, exact-closure renormalisation, dilution to
    1 Rsun @ 1 kpc, a Planck red-end tail, and a flux-conserving rebin onto
    the common grid -- with two differences from the CIFIST branch:

      * the D-15 closure gate [0.8, 1.2] is REPLACED by the D-22 smoothness
        gate (`_agss_closure_smoothness_gate`), because it would reject the
        entire log g <= 0.0 block for a low-gravity flux-radius CONVENTION,
        not non-convergence (T1);
      * the D-15 lambda_max >= 18um keep-rule is KEPT (T1 measured every
        D-19 box node reaches natively to ~999 um, so this never actually
        rejects anything here) but RAISES rather than silently skipping --
        an AGSS file failing it would be a genuinely unexpected event on
        this box, not an expected grid feature the way 10/195 CIFIST files
        were.

    A smoothness-gate flag, like a lambda_max failure, RAISES rather than
    skips: the box was measured (T1) to have no such row, so a flag means a
    bad file, not an expected exclusion -- D-15's CIFIST skip-and-continue
    pattern does not apply here.

    Returns (kept, closure_log):
      kept        -- list of dicts {name, teff, logg, z=0.0,
                     source='AGSS2009', values_desc, renorm_factor,
                     closure_native} -- the same row shape
                     build_cifist_grid_rows emits, with SOURCE='AGSS2009'
                     (D-24) instead of 'CIFIST'
      closure_log -- the PRE-renormalisation native closure per row (T6
                     wants this table), list of dicts {teff, logg,
                     closure_native}, sorted by (logg, teff)
    """
    common_wave_asc = np.asarray(common_wave_desc_um, dtype=float)[::-1]
    files = _iter_agss_files(download_dir)

    parsed = []
    for path in files:
        wave_air_um, f_lambda_native, teff, logg = _parse_cifist_ascii_fast(str(path))
        closure_native = cifist_bolometric_ratio(wave_air_um, f_lambda_native, teff)
        lambda_max_native = float(wave_air_um[-1])
        if lambda_max_native < CIFIST_KEEP_LAMBDA_MAX_MIN_UM:
            raise ValueError(
                f"{path}: AGSS native lambda_max={lambda_max_native:.2f}um < "
                f"{CIFIST_KEEP_LAMBDA_MAX_MIN_UM}um -- unexpected on the D-19 box "
                "(T1 measured native coverage to ~999um at every node); "
                "investigate rather than silently drop.")
        parsed.append(dict(path=path, wave_air_um=wave_air_um,
                           f_lambda_native=f_lambda_native, teff=teff, logg=logg,
                           closure_native=closure_native))

    flagged = _agss_closure_smoothness_gate(parsed)
    if flagged:
        detail = "; ".join(
            f"T_eff={f['teff']:.0f} log g={f['logg']:+.1f} closure={f['closure_native']:.4f} "
            f"({f['resid_sigma']:.1f} sigma from the fit)" for f in flagged)
        raise ValueError(
            f"D-22 smoothness gate: {len(flagged)} AGSS row(s) depart more than "
            f"{AGSS_CLOSURE_SMOOTHNESS_SIGMA} sigma from a smooth closure(T_eff) fit "
            f"at fixed log g -- none expected on the D-19 box (T1): {detail}")

    kept = []
    closure_log = []
    for entry in parsed:
        wave_air_um = entry["wave_air_um"]
        f_lambda_native = entry["f_lambda_native"]
        teff, logg = entry["teff"], entry["logg"]
        closure_native = entry["closure_native"]

        wave_a_air = wave_air_um / _CIFIST_ANGSTROM_TO_UM
        wave_a_vac = air_to_vacuum_angstrom(wave_a_air)
        wave_um_vac = wave_a_vac * _CIFIST_ANGSTROM_TO_UM

        f_nu_surface = flambda_to_fnu_cgs(f_lambda_native, wave_a_vac)

        # D-22: renormalise to EXACT bolometric closure before dilution,
        # same as CIFIST (D-14 addendum) -- unchanged even though the GATE
        # that decides whether to keep the row is different.
        renorm_ratio = cifist_bolometric_ratio(wave_um_vac, f_lambda_native, teff)
        renorm_factor = 1.0 / renorm_ratio
        f_nu_surface = f_nu_surface * renorm_factor

        f_nu_mjy_native = f_nu_surface * CIFIST_DILUTION_MJY_PER_CGS_FNU

        wave_full, flux_full = _append_planck_tail(wave_um_vac, f_nu_mjy_native, teff)

        values_asc = flux_conserving_rebin_wavelength(wave_full, flux_full, common_wave_asc)
        values_desc = values_asc[::-1]

        name = agss_model_name(teff, logg)
        kept.append(dict(name=name, teff=float(teff), logg=float(logg), z=0.0,
                         source="AGSS2009", values_desc=values_desc,
                         renorm_factor=renorm_factor, closure_native=closure_native))
        closure_log.append(dict(teff=float(teff), logg=float(logg),
                                closure_native=float(closure_native)))
        if verbose:
            print(f"    KEEP {entry['path'].name}: native closure={closure_native:.3f} "
                 f"renorm factor={renorm_factor:.4f}")

    closure_log.sort(key=lambda r: (r["logg"], r["teff"]))
    if verbose:
        print(f"    AGSS: {len(kept)} kept (of {len(files)} fetched)")
    return kept, closure_log


def merge_agss_rows(existing_rows, agss_rows):
    """D-20/D-24: append AGSS rows to the already-spliced CK03+CIFIST row
    set, strictly additive. ASSERTS zero (T_EFF, LOGG, Z_H) collisions
    between `agss_rows` and `existing_rows` -- D-19 excludes the five
    would-be duplicate (3500K, 0.0..+2.0) nodes by construction, so this is
    the regression check on that claim, not the mechanism that enforces it.

    Returns the combined list, RE-SORTED ASCII-lexicographically by
    MODEL_NAME (the one row order every emitted file shares, D-12/D-16,
    now also D-24).
    """
    existing_keys = {}
    for r in existing_rows:
        existing_keys[_splice_key(r["teff"], r["logg"], r["z"])] = r["name"]

    collisions = [(r["name"], existing_keys[_splice_key(r["teff"], r["logg"], r["z"])])
                 for r in agss_rows
                 if _splice_key(r["teff"], r["logg"], r["z"]) in existing_keys]
    if collisions:
        raise AssertionError(
            f"AGSS rows collide with existing (T_EFF, LOGG, Z_H) nodes -- "
            f"D-19/D-20 assume zero collisions by construction: {collisions}")

    merged = list(existing_rows) + list(agss_rows)
    merged.sort(key=lambda r: r["name"])
    return merged


def build_sps_merged_rows(source_root, cifist_download_dir, agss_download_dir=None,
                          verbose=True):
    """Stage 2 (D-11..D-15, plus D-19..D-25 when `agss_download_dir` is
    given): the merged model set on the common log-uniform R=300 grid.
    CK03+CIFIST splice per D-12; AGSS (Extension A') appended additively on
    top per D-20, iff `agss_download_dir` is not None -- omitting it
    reproduces the pre-Extension-A' 3,987-row build exactly (backward
    compatible default; callers/tests written against the two-source build
    are unaffected).

    Returns (wave_um_desc, freq_hz_desc, merged_rows, report):
      wave_um_desc, freq_hz_desc -- build_common_wavelength_grid()'s pair
      merged_rows -- list of dicts (see build_ck03_grid_rows /
                     build_cifist_grid_rows / build_agss_grid_rows), sorted
                     ASCII-lexicographically by MODEL_NAME
      report -- counts, the CIFIST skip log, and (when AGSS is included)
                the AGSS closure log, for the driver to print and assert
                against (D-12/D-15: 6 culled, 185 kept CIFIST, 3,987
                CK03+CIFIST rows; D-19: 79 kept AGSS, 4,066 total merged)
    """
    wave_um_desc, freq_hz_desc = build_common_wavelength_grid()
    if verbose:
        print(f"    common grid: {len(wave_um_desc)} points, R={GRID_R}, "
             f"{GRID_WAVELENGTH_MIN_UM}-{GRID_WAVELENGTH_MAX_UM} um")

    ck03_rows = build_ck03_grid_rows(source_root, freq_hz_desc, verbose=verbose)
    cifist_rows, cifist_skipped = build_cifist_grid_rows(
        cifist_download_dir, wave_um_desc, verbose=verbose)

    merged_rows, culled_ck03 = splice_ck03_and_cifist(ck03_rows, cifist_rows)

    agss_rows, agss_closure_log = [], []
    if agss_download_dir is not None:
        if verbose:
            print("    AGSS (Extension A', D-19..D-25):")
        agss_rows, agss_closure_log = build_agss_grid_rows(
            agss_download_dir, wave_um_desc, verbose=verbose)
        merged_rows = merge_agss_rows(merged_rows, agss_rows)

    report = dict(
        n_ck03_total=len(ck03_rows),
        n_ck03_culled=len(culled_ck03),
        culled_names=[r["name"] for r in culled_ck03],
        n_cifist_fetched=len(cifist_rows) + len(cifist_skipped),
        n_cifist_kept=len(cifist_rows),
        n_cifist_skipped=len(cifist_skipped),
        skipped=cifist_skipped,
        n_agss_kept=len(agss_rows),
        agss_closure_log=agss_closure_log,
        n_merged=len(merged_rows),
    )
    if verbose:
        msg = (f"    splice: {report['n_ck03_total']} CK03 - "
              f"{report['n_ck03_culled']} culled + {report['n_cifist_kept']} CIFIST")
        if agss_download_dir is not None:
            msg += f" + {report['n_agss_kept']} AGSS"
        msg += f" = {report['n_merged']} rows"
        print(msg)
    return wave_um_desc, freq_hz_desc, merged_rows, report


def build_sps_flux_fits_from_rows(merged_rows, wave_um_desc, freq_hz_desc, output_path,
                                  overwrite=True, aperture_au=POINT_SOURCE_APERTURE_AU,
                                  distance_cm=REFERENCE_DISTANCE_CM,
                                  name_format=MODEL_NAME_COLUMN_FORMAT):
    """Stages 2-3: write flux.fits from `build_sps_merged_rows`' output.

    UNCERTAINTIES = float32(0.01 * VALUES) for every row, CK03 and CIFIST
    alike -- CK03's native seds/*.fits.gz TOTAL_FLUX_ERR is already exactly
    1% of TOTAL_FLUX (verified), so this is not a new convention, only a
    uniform way to write it once the rows are regridded and merged.
    """
    names = np.array([r["name"] for r in merged_rows])
    n_models, n_wav = len(merged_rows), len(wave_um_desc)
    values = np.empty((n_models, 1, n_wav), dtype=np.float32)
    for i, row in enumerate(merged_rows):
        values[i, 0, :] = row["values_desc"]
    uncertainties = (0.01 * values).astype(np.float32)

    return write_flux_cube(
        output_path,
        names=names,
        wave_um_desc=wave_um_desc,
        freq_hz_desc=freq_hz_desc,
        values=values,
        uncertainties=uncertainties,
        distance_cm=distance_cm,
        distance_comment=DISTANCE_COMMENT,
        apertures_au=np.array([aperture_au]),
        valid=None,
        name_format=name_format,
        overwrite=overwrite,
    )


def write_stripped_parameters_fits(merged_rows, output_path, overwrite=True):
    """Stage 1b: write parameters.fits for the MERGED CK03+CIFIST(+AGSS)
    row set (D-11..D-16, D-19..D-25), in `merged_rows`' order
    (ASCII-lexicographic by MODEL_NAME -- D-12/D-24).

    Columns: MODEL_NAME (30A, stripped), T_EFF (E, K), LOGG (E), Z_H (E)
    -- same formats/units/VALUES the upstream CK03 file carries, but
    HDF5-safe names rather than its literal "LOG[G]"/"[Z/H]" (a bracket
    and, worse, a "/" -- HDF5 treats "/" as a path separator, so a
    column named "[Z/H]" written verbatim into an HDF5 register becomes
    a subgroup "[Z" holding a dataset "H]"; see library.write_library/
    write_register's HDF5-safe-identifier assertion) -- plus a FOURTH
    column, SOURCE (8A: 'CK03'/'CIFIST', D-16, plus 'AGSS2009' when
    `merged_rows` includes the Extension A' rows, D-24).

    CAVEAT superseding this function's pre-Extension-A docstring: it no
    longer copies parameters.fits through from a single source table with
    only MODEL_NAME's padding stripped (that was D-7's whole claim). It now
    WRITES the table from `merged_rows`: for CK03-derived rows, T_EFF/
    LOGG/Z_H are the bit-identical VALUES `build_ck03_grid_rows` read
    from the upstream file (renamed on the way in, see
    `_read_ck03_parameters`); CIFIST-derived rows are new data (D-11), and
    SOURCE is authored here from each row's own provenance, not carried
    through from any source table -- neither CK03's upstream file nor
    CIFIST (which has no parameters.fits of its own) has such a column.
    """
    names = np.array([row["name"] for row in merged_rows])
    teff = np.array([row["teff"] for row in merged_rows], dtype=np.float32)
    logg = np.array([row["logg"] for row in merged_rows], dtype=np.float32)
    z = np.array([row["z"] for row in merged_rows], dtype=np.float32)
    source = np.array([row["source"] for row in merged_rows])

    columns = [
        fits.Column(name="MODEL_NAME", format=MODEL_NAME_COLUMN_FORMAT, array=names),
        fits.Column(name="T_EFF", format="E", unit="K", array=teff),
        fits.Column(name="LOGG", format="E", array=logg),
        fits.Column(name="Z_H", format="E", array=z),
        fits.Column(name="SOURCE", format="8A", array=source),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="PARAMETERS")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(output_path, overwrite=overwrite)
    return SpsParameters(model_names=names, t_eff=np.asarray(teff, dtype=float))


def curate_sps_model_set(source_root, cifist_download_dir, output_root,
                         agss_download_dir=None, overwrite=True,
                         convolved_bands=None, verbose=True):
    """Build the CK03+CIFIST(+AGSS)-spliced, yso-conforming copy of the sps
    model set at output_root (D-11..D-25). See the module docstring's
    Stages list for the 0-5 numbering; stage 0 (the raw CIFIST/AGSS fetch)
    is NOT invoked here -- it is a separate, one-time, network-touching
    step (the driver calls `fetch_btsettl_cifist`/`fetch_btsettl_agss`
    itself, ahead of time).

    `agss_download_dir`, when given, adds the 79 D-19 AGSS2009 giant rows
    (Extension A') on top of the CK03+CIFIST splice, additively (D-20);
    omitting it (the default) reproduces the pre-Extension-A' 3,987-row
    build exactly.

    `convolved_bands` is the band-name tuple to convolve, defaulting to
    CONVOLVED_BANDS_TO_BUILD.
    """
    if convolved_bands is None:
        convolved_bands = CONVOLVED_BANDS_TO_BUILD
    source_root = Path(source_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # stage 1a -- models.conf WRITTEN (D-6).
    write_models_conf(str(output_root / "models.conf"), name=MODELS_CONF_NAME,
                      aperture_dependent=False, length_subdir=0,
                      logd_step=LOGD_STEP, version=2)

    # stage 2 -- the merged, gridded row set: common R=300 grid (D-13), CK03
    # log-log interpolated, CIFIST rebinned with a Planck tail (D-14/D-15),
    # spliced (D-12); AGSS appended additively (D-19..D-25) iff
    # agss_download_dir is given.
    if verbose:
        print("  stage 2  build_flux_cube")
    wave_um_desc, freq_hz_desc, merged_rows, report = build_sps_merged_rows(
        source_root, cifist_download_dir, agss_download_dir=agss_download_dir,
        verbose=verbose)

    if report["n_cifist_kept"] != 185:
        raise AssertionError(
            f"expected 185 kept CIFIST rows (D-15), got {report['n_cifist_kept']}")
    if report["n_ck03_culled"] != 6:
        raise AssertionError(
            f"expected 6 culled CK03 duplicates (D-12), got {report['n_ck03_culled']}")
    if agss_download_dir is not None:
        if report["n_agss_kept"] != 79:
            raise AssertionError(
                f"expected 79 kept AGSS rows (D-19), got {report['n_agss_kept']}")
        if report["n_merged"] != 4066:
            raise AssertionError(
                f"expected 4066 merged rows (D-19), got {report['n_merged']}")
        if verbose:
            print("    AGSS closure (pre-renormalisation, T6 wants this table), "
                 "sorted by (log g, T_eff):")
            for row in report["agss_closure_log"]:
                print(f"      T_eff={row['teff']:.0f} log g={row['logg']:+.1f} "
                     f"closure={row['closure_native']:.4f}")
    elif report["n_merged"] != 3987:
        raise AssertionError(
            f"expected 3987 merged rows (D-12), got {report['n_merged']}")

    # stage 1b -- parameters.fits, from the SAME merged_rows (D-16), so row
    # order can never drift from stage 2/3/4/5's.
    if verbose:
        print("  stage 1  copy_static_metadata (parameters.fits, SOURCE column)")
    parameters = write_stripped_parameters_fits(
        merged_rows, output_root / "parameters.fits", overwrite=overwrite)

    # stage 3 -- flux.fits
    if verbose:
        print("  stage 3  write_flux_fits")
    build_sps_flux_fits_from_rows(merged_rows, wave_um_desc, freq_hz_desc,
                                  output_root / "flux.fits", overwrite=overwrite)

    # stage 4 -- convolve OUR OWN band fluxes from the flux.fits just
    # written, through this project's filter curves (D-9), with MODEL_NAME
    # in the SAME merged/sorted order as every other file (D-12/D-16) --
    # NOT source_root's order, which no longer matches after the splice.
    if verbose:
        print("  stage 4  build_convolved_bands")
    model_names = np.array([row["name"] for row in merged_rows])
    build_convolved_bands(str(output_root), tuple(convolved_bands),
                          model_names, CONVMETH_NOTE)

    # stage 5 -- classmap.fits, from stage 1's in-memory arrays rather than
    # by re-reading parameters.fits, so the two cannot drift.
    if verbose:
        print("  stage 5  write_classmap_fits")
    write_classmap_fits(parameters, output_root / "classmap.fits",
                        model_dir=str(output_root))

    return output_root, report


@dataclass
class SpsParameters:
    """The parameter arrays stage 1 already holds, returned so classmap.fits
    is built from the SAME in-memory values that went into parameters.fits.

    CLASSMAP duplicates what parameters.fits carries -- SUBCLASS is a pure
    function of T_EFF -- and is therefore capable of disagreeing with it.
    Computing once and writing twice is what makes that impossible; re-reading
    the library to build CLASSMAP would reintroduce exactly that gap.
    """
    model_names: np.ndarray
    t_eff: np.ndarray


def write_classmap_fits(parameters, out_path, model_dir=None):
    """classmap.fits, via the shared writer in model_io.

    This library supplies only what is its own -- the class id, the two
    legends and the provenance text -- because the shared writer deliberately
    holds no vocabulary.

    Built from stage 1's in-memory MODEL_NAME/T_EFF rather than by re-reading
    parameters.fits: SUBCLASS is a pure function of T_EFF, so the two files
    could otherwise disagree. Passing `model_dir` turns on the assertion that
    MODEL_NAME matches flux.fits IN ORDER.
    """
    return model_io.write_classmap_fits(
        out_path,
        names=parameters.model_names,
        class_id=MODEL_CLASS,
        subclass=assign_spectral_letter(parameters.t_eff),
        class_legend=CLASS_LEGEND,
        subclass_legend=SUBCLASS_LEGEND,
        provenance=(
            ("CLASS_SOURCE", "library declaration; all models here are this class"),
            ("SUBCLASS_SOURCE", "MK letter from T_EFF; dwarf sequence, all log g"),
            ("SUBCLASS_REF", "Pecaut & Mamajek 2013 ApJS 208,9; tab v2022.04.16"),
            ("SUBCLASS_NOTE", "cuts = midpoints of adjacent X9V/X0V anchors"),
            ("LEGEND_SOURCE", "Teff cuts from the table; descriptions authored"),
            # D-17 (Extension A): the four anchors that extend the sequence
            # through L and T, same table/version as the seven above.
            ("SUBCLASS_EXT", "D-17 anchors: M9V2380 L0V2270 L9V1370 T0V1255K"),
        ),
        model_dir=model_dir,
        name_format=MODEL_NAME_COLUMN_FORMAT,
    )


# ====================================================================
# Owner's library specification (2026-10-10): the 1-sigma-sampled SPS
# CLASS LIBRARY, built on top of the full raw splice above.
# ====================================================================
# SPS has two products from here on, in TWO DIFFERENT DIRECTORIES:
#
#   output_root()                      -- UNCHANGED: the full raw splice
#                                          (today 4,066 rows) this module
#                                          already builds above. Stays
#                                          exactly where it is because
#                                          build/pahc.py's `sps_dir()` and
#                                          build/yso.py's `build_paths().
#                                          sps_lib_dir` both call
#                                          `paths.path_for("sed_models_sps")`
#                                          DIRECTLY and are not edited by
#                                          this task -- so whatever sits at
#                                          that literal path must keep
#                                          being a complete sedfitter model
#                                          directory holding the FULL
#                                          photosphere family (pahc draws
#                                          hosts from it; yso_dedup fits
#                                          candidates against it; neither
#                                          can use a 1-sigma-coarsened
#                                          grid -- see this module's and
#                                          build/sps.py's own docstrings).
#   output_root()/SAMPLED_SUBDIR        -- NEW: the 1-sigma-sampled class
#                                          library -- the one that answers
#                                          "what can the fitter actually
#                                          distinguish" -- plus the MEMBERS
#                                          group (members.fits) recording
#                                          the degeneracy the sampling
#                                          collapses. This is a complete,
#                                          independent model directory in
#                                          its own right (own models.conf/
#                                          parameters.fits/flux.fits/
#                                          convolved//classmap.fits), not a
#                                          view onto the full splice.
#
# This is the opposite nesting from the spec text's own example ("the
# full set under a subdirectory"), chosen because it is the only layout
# that satisfies "keep the existing path working for the two consumers"
# LITERALLY given the hard constraint that build/pahc.py and
# build/yso.py (via yso_dedup.py) are not to be edited -- see
# CURATION_SPS.md sec. "the two consumers" for the full reasoning.
SAMPLED_SUBDIR = "sampled_1sigma"
MEMBERS_FILENAME = "members.fits"

#: this library's three physical parameters -- exactly parameters.fits'
#: non-MODEL_NAME, non-SOURCE columns -- in the order the MEMBERS group's
#: range columns are emitted.
SPS_PARAMETER_COLUMNS = (("T_EFF", "T_EFF"), ("LOGG", "LOGG"), ("Z_H", "ZH"))

#: SUBCLASS_LEGEND's codes, in that tuple's own order -- the fraction
#: columns the MEMBERS group's represented-set bookkeeping carries.
SUBCLASS_CODES = tuple(code for code, _ in SUBCLASS_LEGEND)


def _read_full_sps_library(full_root, convolved_bands=CONVOLVED_BANDS_TO_BUILD):
    """Everything the 1-sigma sampling needs from an ALREADY-BUILT full
    SPS splice at `full_root`: the canonical MODEL_NAME row order (flux.
    fits' own -- every other file in the directory shares it, by D-12/
    D-16/D-24's own sort-and-write discipline), T_EFF/LOGG/Z_H/SOURCE
    from parameters.fits, SUBCLASS from classmap.fits, and the 8-band
    reference flux f_ref (mJy) every raw model's SED-space coordinate is
    built from. No physics is re-derived here -- this only reads files
    `curate_sps_model_set` already wrote."""
    full_root = Path(full_root)
    with fits.open(full_root / "flux.fits", memmap=True) as hdul:
        names = np.char.strip(np.asarray(hdul["MODEL_NAMES"].data.field(0)).astype(str))
        wave_um_desc = np.asarray(hdul["SPECTRAL_INFO"].data["WAVELENGTH"], dtype=float)
        freq_hz_desc = np.asarray(hdul["SPECTRAL_INFO"].data["FREQUENCY"], dtype=float)
        values = np.asarray(hdul["VALUES"].data, dtype=np.float32)
        uncertainties = np.asarray(hdul["UNCERTAINTIES"].data, dtype=np.float32)
        distance_cm = float(hdul["PRIMARY"].header["DISTANCE"])

    with fits.open(full_root / "parameters.fits", memmap=True) as hdul:
        ptable = hdul[1].data
        pnames = np.char.strip(np.asarray(ptable["MODEL_NAME"]).astype(str))
        order = model_io.align_by_name(names, pnames)
        teff = np.asarray(ptable["T_EFF"], dtype=float)[order]
        logg = np.asarray(ptable["LOGG"], dtype=float)[order]
        zh = np.asarray(ptable["Z_H"], dtype=float)[order]
        source = np.char.strip(np.asarray(ptable["SOURCE"]).astype(str))[order]

    with fits.open(full_root / "classmap.fits", memmap=True) as hdul:
        ctable = hdul["CLASSMAP"].data
        cnames = np.char.strip(np.asarray(ctable["MODEL_NAME"]).astype(str))
        corder = model_io.align_by_name(names, cnames)
        subclass = np.char.strip(np.asarray(ctable["SUBCLASS"]).astype(str))[corder]

    f_ref = np.empty((names.size, len(convolved_bands)), dtype=float)
    for j, band in enumerate(convolved_bands):
        bnames, bflux = model_io.load_convolved_total_flux_mjy(str(full_root), band)
        border = model_io.align_by_name(names, bnames)
        f_ref[:, j] = np.asarray(bflux, dtype=float)[border]

    return dict(names=names, wave_um_desc=wave_um_desc, freq_hz_desc=freq_hz_desc,
                values=values, uncertainties=uncertainties, distance_cm=distance_cm,
                teff=teff, logg=logg, zh=zh, source=source, subclass=subclass,
                f_ref=f_ref)


def build_sps_quotient_space(verbose=True):
    """The SED-space quotient projector the sampling runs in.

    Owner's ruling (2026-10-10, two revisions, both superseding this
    function's original catalogue-derived design): the sampling scale is
    `constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR` -- TWICE the surveys'
    own published ABSOLUTE-CALIBRATION floor (2MASS J/H/Ks 0.010 dex,
    Skrutskie et al. 2006 AJ 131,1163; IRAC I1-I4 0.013 dex, Reach et al.
    2005 PASP 117,978; MIPS M1 0.017 dex, Engelbracht et al. 2007 PASP
    119,994 -- i.e. 0.020/0.026/0.034 dex per band at this factor), NOT
    the catalogue-derived `sigma_log` this function used to read via
    `sed_models_register.noise.read`. That product is RETIRED as an
    input here: its own raw files have been deleted, so it is a stale
    artefact, and this module no longer opens it.

    THE FACTOR OF 2 IS DERIVED, NOT A KNOB (`constants.py`'s own
    derivation, repeated here because it is the reason this library's
    count is what it is): a grid of spacing d sampling a class's
    exp(-chi2/2) evidence against a source of photometric error sigma
    reproduces the integral to 2*exp(-2*pi^2*sigma^2/d^2) (Poisson
    summation). At d = 2*sigma that is 0.0144 -- 1.4% of the evidence,
    0.015 nats -- and only for the best-measured ~1% of the catalogue
    (the sources whose error reaches the calibration floor); for a
    typical detection (statistical error 0.017-0.035 dex, so d is about
    one sigma there) the same expression is below 1e-4. Sampling at the
    bare floor instead of twice it was tried and rejected (constants.py):
    it implies an unbelievable template count. `constants.
    SURVEY_CALIB_SIGMA_LOG_VECTOR` (the bare floor, no factor) was this
    function's intermediate design, briefly in place and superseded
    before any product shipped at that scale.

    `constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR` is a survey PROPERTY
    (instrument calibration, published by the survey teams, times a
    derived evidence-resolution factor) -- not a population/prior/Gaia
    quantity, so reading it does not violate spec item 4 (independence).

    `sed_models_register.density.build_quotient_space` is still the
    machinery that turns this vector into the 5-D quotient projector and
    `sigma_eff` -- shared, library-independent (it depends only on
    `sigma_log` and the two extinction laws, never on a library), and
    now DEFAULTS to this same vector -- so this still reproduces exactly
    the space and SIGEFF any of the six registers would use, without
    this module reading, writing or rebuilding one.
    """
    sigma_log = np.asarray(_LIBRARY_SAMPLING_SIGMA_LOG_VECTOR, dtype=float)
    space = _register_density.build_quotient_space(
        sigma_log=sigma_log,
        sigma_source="constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR (2x the literature "
                     "survey calibration floor; 2MASS/IRAC/MIPS, see constants.py)")
    if verbose:
        print(f"    quotient space: d={space.d}, SIGEFF=sigma_eff="
              f"{space.sigma_eff:.6f} dex (sigma_log source: "
              "constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR, 2x literature calibration floor)")
    return space


def greedy_max_coverage_r_net(coords, names, radius):
    """Deterministic greedy r-net at `radius` (the coordinator's
    2026-10-10 correction, replacing an earlier "pick any uncovered
    model" / medoid-refinement design): at each step, among every raw
    model still UNCOVERED, keep the one with the most UNCOVERED
    neighbours within `radius` -- the standard greedy maximum-coverage
    set-cover rule -- mark every uncovered neighbour of the kept model
    covered, and repeat until nothing is uncovered. Ties are broken on
    MODEL_NAME, smallest wins, so the whole procedure is fully
    deterministic: NO RANDOM SEED is used or needed anywhere here.

    Both properties the specification asks for fall out of the rule
    itself, not from any later repair pass:

      COVERING  (every raw model within `radius` of a kept template) --
                the loop only terminates once nothing is uncovered.
      PACKING   (no two kept templates closer than `radius`) -- a model
                is only ever kept while it is still uncovered, i.e.
                while every already-kept template is more than `radius`
                away from it (ball queries are symmetric: if some kept
                template r were within `radius` of candidate i, i would
                already have been marked covered when r was kept).

    Picking the locally densest uncovered point FIRST also means each
    representative sits centrally in its own eventual Voronoi cell by
    construction -- no post-hoc medoid move is needed or performed.

    `coords` -- (n, d) quotient-space coordinates. `names` -- (n,)
    MODEL_NAME, for the tie-break only. Returns the kept RAW INDEX array,
    in the order chosen (not sorted).
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    tree = cKDTree(coords)
    # One ball query per point -- n is a few thousand here, so a plain
    # Python loop (rather than the vectorized multi-point form, whose
    # return shape varies across scipy versions) is both fast enough and
    # unambiguous.
    neighbor_lists = [np.asarray(tree.query_ball_point(coords[i], r=radius), dtype=int)
                      for i in range(n)]
    uncovered_count = np.array([nb.size for nb in neighbor_lists], dtype=np.int64)
    covered = np.zeros(n, dtype=bool)
    reps = []

    while not np.all(covered):
        uncovered_idx = np.flatnonzero(~covered)
        counts = uncovered_count[uncovered_idx]
        best_count = counts.max()
        candidates = uncovered_idx[counts == best_count]
        if candidates.size > 1:
            best = int(candidates[np.argmin(names[candidates])])
        else:
            best = int(candidates[0])
        reps.append(best)

        nb = neighbor_lists[best]
        newly = nb[~covered[nb]]
        covered[newly] = True
        # Maintain uncovered_count by DECREMENTING rather than
        # recomputing from scratch: every point k that counted a
        # newly-covered j as an uncovered neighbour loses that neighbour.
        for j in newly:
            for k in neighbor_lists[j]:
                if not covered[k]:
                    uncovered_count[k] -= 1

    return np.array(reps, dtype=int)


def voronoi_assign(coords, rep_idx):
    """Assign every row of `coords` to its NEAREST representative (by
    index into `coords`, i.e. a raw-model index), once and at the end --
    the represented sets are DERIVED from the kept representatives, never
    the other way round. Returns (assigned_raw_idx, distance)."""
    coords = np.asarray(coords, dtype=float)
    rep_idx = np.asarray(rep_idx, dtype=int)
    tree = cKDTree(coords[rep_idx])
    dist, nearest = tree.query(coords, k=1, workers=-1)
    return rep_idx[nearest], np.asarray(dist, dtype=float)


def min_pairwise_representative_distance(coords, rep_idx):
    """Minimum pairwise distance among the kept representatives -- the
    packing-property check (must be >= the sampling radius). `inf` for
    fewer than two representatives."""
    rep_idx = np.asarray(rep_idx, dtype=int)
    if rep_idx.size < 2:
        return float("inf")
    rep_coords = np.asarray(coords, dtype=float)[rep_idx]
    tree = cKDTree(rep_coords)
    dist, _ = tree.query(rep_coords, k=2, workers=-1)
    return float(dist[:, 1].min())


def write_sps_members_fits(output_path, *, rep_names, member_counts,
                           subclass_fractions, param_ranges,
                           dist_sigma_median, dist_sigma_max,
                           raw_names, raw_rep_names, raw_dist_sigma,
                           provenance, overwrite=True):
    """The MEMBERS group for the 1-sigma-sampled SPS class library (spec
    item 3). No register is built by this module, so this file -- not a
    register -- is where the degeneracy bookkeeping lives; the generic
    register pass due after all six libraries land is expected to
    promote it the way it promotes parameters.fits columns.

    Two extensions:
      MEMBERS     one row per KEPT TEMPLATE: its represented set's count,
                  SUBCLASS_LEGEND fractions, and (T_EFF, LOGG, Z_H)
                  min/median/max, plus the within-set member-to-
                  representative distance (SIGEFF units) median/max, so
                  a reader sees how centred each set is without
                  re-deriving it from MEMBERSHIP.
      MEMBERSHIP  one row per RAW MODEL: which kept template's
                  represented set it falls into (Voronoi assignment) and
                  its distance to that template in SIGEFF units. The
                  union over MEMBERS' represented sets is the full raw
                  set -- nothing the raw library knew is lost.

    Point values are NOT duplicated here (schema parity, spec item 3):
    a kept template's own T_EFF/LOGG/Z_H/SOURCE stay in
    parameters.fits, where every other library's point values live.
    """
    n_kept = len(rep_names)
    cols_members = [
        fits.Column(name="MODEL_NAME", format=MODEL_NAME_COLUMN_FORMAT,
                    array=np.array(rep_names)),
        fits.Column(name="N_MEMBERS", format="J",
                    array=np.array(member_counts, dtype=np.int32)),
    ]
    for code in SUBCLASS_CODES:
        cols_members.append(fits.Column(
            name=f"FRAC_{code}", format="D",
            array=np.array([subclass_fractions[i].get(code, 0.0)
                            for i in range(n_kept)], dtype=float)))
    for pname, short in SPS_PARAMETER_COLUMNS:
        for stat in ("MIN", "MEDIAN", "MAX"):
            cols_members.append(fits.Column(
                name=f"{short}_{stat}", format="D",
                array=np.array([param_ranges[i][pname][stat] for i in range(n_kept)],
                               dtype=float)))
    cols_members.append(fits.Column(name="DIST_SIGMA_MEDIAN", format="D",
                                    array=np.array(dist_sigma_median, dtype=float)))
    cols_members.append(fits.Column(name="DIST_SIGMA_MAX", format="D",
                                    array=np.array(dist_sigma_max, dtype=float)))
    members_hdu = fits.BinTableHDU.from_columns(cols_members, name="MEMBERS")

    membership_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="MODEL_NAME", format=MODEL_NAME_COLUMN_FORMAT,
                    array=np.array(raw_names)),
        fits.Column(name="REP_NAME", format=MODEL_NAME_COLUMN_FORMAT,
                    array=np.array(raw_rep_names)),
        fits.Column(name="DIST_SIGMA", format="D",
                    array=np.array(raw_dist_sigma, dtype=float)),
    ], name="MEMBERSHIP")

    primary = fits.PrimaryHDU()
    for key, value in provenance:
        primary.header[key] = model_io.provenance_card(key, str(value))

    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fits.HDUList([primary, members_hdu, membership_hdu]).writeto(
        output_path, overwrite=overwrite)
    return output_path


def build_sps_sampled_library(full_root, sampled_root,
                              convolved_bands=None, overwrite=True, verbose=True):
    """Owner's library specification (2026-10-10): build the 1-sigma-
    sampled SPS CLASS LIBRARY at `sampled_root` from the already-built
    full raw splice at `full_root` (today 4,066 rows). `sampled_root`
    must be a DIFFERENT directory from `full_root` -- see this section's
    header comment on the two-directory layout and why the full splice
    is never moved out of `full_root`.

    Pipeline:
      1. Read flux.fits/parameters.fits/classmap.fits/convolved/*.fits
         from `full_root` -- no physics is re-derived.
      2. The project-wide quotient space (5-D; removes gray/scale and
         both Av laws) from the library sampling scale
         (`constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR`, 2x the literature
         survey calibration floor), fed to
         `sed_models_register.density.build_quotient_space`
         (`build_sps_quotient_space`) -- no register is read, written or
         rebuilt.
      3. `greedy_max_coverage_r_net` at SIGEFF (deterministic, no seed),
         then `voronoi_assign` once at the end to derive every kept
         template's represented set (coordinator's correction,
         2026-10-10: the partition is INDUCED by the representatives,
         not decided before them).
      4. Per kept template: the represented set's count, SUBCLASS
         fractions and (T_EFF, LOGG, Z_H) ranges -- the MEMBERS
         group (spec item 3).
      5. Write the sampled library: models.conf, parameters.fits (point
         values only -- schema parity with the other five libraries),
         flux.fits, convolved/*.fits (RE-CONVOLVED from the sliced
         flux.fits, not sliced convolved arrays, so the two can never
         drift apart), classmap.fits, members.fits.

    Spec item 4 (independence): the only inputs anywhere in this
    pipeline are the library's own convolved fluxes, the project's
    noise/extinction machinery, and the raw grid's own parameters/
    subclass labels -- no population weight, prior quantity or Gaia
    quantity enters at any step.

    Returns (sampled_root, report) -- `report` is a dict with the
    acceptance numbers the task asks for (kept count, covering radius
    and minimum inter-representative distance as fractions of SIGEFF,
    the member-to-representative distance distribution, etc).
    """
    if convolved_bands is None:
        convolved_bands = CONVOLVED_BANDS_TO_BUILD
    full_root = Path(full_root)
    sampled_root = Path(sampled_root)
    if sampled_root.resolve() == full_root.resolve():
        raise ValueError("sampled_root must differ from full_root -- the full "
                         "raw splice is never overwritten by the sampled library")
    sampled_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  sampled-library stage 1  read full splice at {full_root}")
    lib = _read_full_sps_library(full_root, convolved_bands=convolved_bands)
    names = lib["names"]
    n_raw = int(names.size)

    if verbose:
        print("  sampled-library stage 2  quotient space (library sampling scale "
             "= 2x literature calibration floor; READ-ONLY density.build_quotient_space)")
    space = build_sps_quotient_space(verbose=verbose)
    sigma_eff = float(space.sigma_eff)

    # B0 floor, via the SAME shared function reference.derive uses, so the
    # two cannot drift apart (cog.peak_and_floor's own docstring).
    flux_cubes = {b: lib["f_ref"][:, [j]] for j, b in enumerate(convolved_bands)}
    peak_row, floor_dex_row, floor_linear_row, all_zero_flux = cog.peak_and_floor(flux_cubes)
    if all_zero_flux.any():
        raise ValueError(
            f"{int(all_zero_flux.sum())} SPS model(s) have zero flux in every "
            "band -- no peak to anchor the B0 floor to")
    x = np.log10(np.maximum(lib["f_ref"], floor_linear_row[:, None]))
    coords = space.project(x)

    if verbose:
        print("  sampled-library stage 3  greedy max-coverage r-net + Voronoi assignment")
    reps = greedy_max_coverage_r_net(coords, names, sigma_eff)
    assigned, dist = voronoi_assign(coords, reps)
    dist_sigma = dist / sigma_eff
    covering_radius_fraction = float(dist.max() / sigma_eff)
    min_pw = min_pairwise_representative_distance(coords, reps)
    min_pairwise_fraction = float(min_pw / sigma_eff) if np.isfinite(min_pw) else float("inf")
    coverage_fraction = float(np.mean(dist <= sigma_eff * (1 + 1e-9)))
    if verbose:
        print(f"    kept={len(reps)} of {n_raw} raw models; "
             f"covering radius / SIGEFF = {covering_radius_fraction:.6f} (<= 1.0); "
             f"min inter-representative distance / SIGEFF = {min_pairwise_fraction:.6f} "
             f"(>= 1.0); coverage fraction = {coverage_fraction:.6f}")

    order_sort = np.argsort(names[reps])
    reps_sorted = reps[order_sort]
    kept_names = names[reps_sorted]

    if verbose:
        print(f"  sampled-library stage 4  members group ({len(reps_sorted)} kept templates)")
    member_counts, subclass_fractions, param_ranges = [], [], []
    dist_sigma_median_per_rep, dist_sigma_max_per_rep = [], []
    raw_rep_names = names[assigned]
    for r in reps_sorted:
        member_idx = np.flatnonzero(assigned == r)
        member_counts.append(int(member_idx.size))
        sc = lib["subclass"][member_idx]
        subclass_fractions.append({code: float(np.mean(sc == code)) for code in SUBCLASS_CODES})
        ranges = {}
        for pname, _short in SPS_PARAMETER_COLUMNS:
            arr = {"T_EFF": lib["teff"], "LOGG": lib["logg"], "Z_H": lib["zh"]}[pname]
            vals = arr[member_idx]
            ranges[pname] = dict(MIN=float(np.min(vals)), MEDIAN=float(np.median(vals)),
                                 MAX=float(np.max(vals)))
        param_ranges.append(ranges)
        dsig = dist_sigma[member_idx]
        dist_sigma_median_per_rep.append(float(np.median(dsig)))
        dist_sigma_max_per_rep.append(float(np.max(dsig)))

    if verbose:
        print("  sampled-library stage 5  write models.conf/parameters.fits/flux.fits/"
             "convolved/classmap.fits/members.fits")

    write_models_conf(str(sampled_root / "models.conf"),
                      name=MODELS_CONF_NAME + " -- 1-sigma-sampled class library",
                      aperture_dependent=False, length_subdir=0,
                      logd_step=LOGD_STEP, version=2)

    rows = [dict(name=str(nm), teff=float(lib["teff"][r]), logg=float(lib["logg"][r]),
                z=float(lib["zh"][r]), source=str(lib["source"][r]))
           for nm, r in zip(kept_names, reps_sorted)]
    parameters = write_stripped_parameters_fits(
        rows, sampled_root / "parameters.fits", overwrite=overwrite)

    write_flux_cube(
        str(sampled_root / "flux.fits"),
        names=kept_names,
        wave_um_desc=lib["wave_um_desc"], freq_hz_desc=lib["freq_hz_desc"],
        values=lib["values"][reps_sorted], uncertainties=lib["uncertainties"][reps_sorted],
        distance_cm=lib["distance_cm"], distance_comment=DISTANCE_COMMENT,
        apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
        valid=None, name_format=MODEL_NAME_COLUMN_FORMAT, overwrite=overwrite)

    build_convolved_bands(str(sampled_root), tuple(convolved_bands), kept_names, CONVMETH_NOTE)

    write_classmap_fits(parameters, sampled_root / "classmap.fits", model_dir=str(sampled_root))

    write_sps_members_fits(
        str(sampled_root / MEMBERS_FILENAME),
        rep_names=kept_names, member_counts=member_counts,
        subclass_fractions=subclass_fractions, param_ranges=param_ranges,
        dist_sigma_median=dist_sigma_median_per_rep, dist_sigma_max=dist_sigma_max_per_rep,
        raw_names=names, raw_rep_names=raw_rep_names, raw_dist_sigma=dist_sigma,
        provenance=(
            # Full method description and HDU widths are in the report /
            # this module's docstrings -- a HIERARCH card's value is
            # capped at ~51-55 chars (model_io.provenance_card), too
            # short for either, so these cards carry a short tag rather
            # than the full text.
            ("SAMPLING_METHOD", "greedy max-cov r-net; Voronoi sets; no seed"),
            ("SIGMA_EFF_DEX", f"{sigma_eff:.6f}"),
            ("N_RAW_MODELS", str(n_raw)),
            ("N_KEPT_TEMPLATES", str(len(reps_sorted))),
            ("COVER_RAD_FRAC", f"{covering_radius_fraction:.6f}"),
            ("MIN_PW_FRAC", f"{min_pairwise_fraction:.6f}"),
            ("COVERAGE_FRAC", f"{coverage_fraction:.6f}"),
            # Owner's 2026-10-10 ruling: 2x the literature survey
            # calibration floor, NOT the retired catalogue-derived
            # sigma_log.
            ("SIGMA_LOG_SRC", "constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR"),
        ),
        overwrite=overwrite)

    problems = model_io.validate_model_directory(str(sampled_root), profile="sps")
    if problems:
        raise SystemExit("\n".join(problems))

    report = dict(
        n_raw=n_raw, n_kept=int(len(reps_sorted)), sigma_eff=sigma_eff,
        covering_radius_fraction=covering_radius_fraction,
        min_pairwise_fraction=min_pairwise_fraction,
        coverage_fraction=coverage_fraction,
        dist_sigma_median_overall=float(np.median(dist_sigma)),
        dist_sigma_max_overall=float(np.max(dist_sigma)),
        kept_names=kept_names.tolist(),
        member_counts=member_counts,
        subclass_fractions=subclass_fractions,
        param_ranges=param_ranges,
        dist_sigma_median_per_rep=dist_sigma_median_per_rep,
        dist_sigma_max_per_rep=dist_sigma_max_per_rep,
    )
    return sampled_root, report


# ====================================================================
# Stage 0 (D-18): raw-data acquisition, BT-Settl CIFIST grid
# ====================================================================
# Stage 0 in the module docstring's numbering above, but still NOT wired
# into curate_sps_model_set -- it is a separate, one-time, network-touching
# step the driver runs ahead of the build (mirrors galaxy_curate.py's own
# Stage 0 pattern: pinned + sha256-verified fetches through
# model_io.fetch_pinned), extending the SPS grid to cooler (sub-3500 K)
# photospheres via the BT-Settl CIFIST2011_2015 grid (Allard, Homeier &
# Freytag 2012; Allard+2013; Baraffe+2015; Caffau+2011), the substellar/
# late-M continuation the Kurucz/ATLAS9 grid does not cover. One adaptation
# from the galaxy_curate precedent: nobody has published fixed checksums
# for these 446 SVO-generated ASCII exports (unlike Polletta/Berta/Brown's
# archived files), so the manifest is BOOTSTRAPPED on first fetch rather
# than hard-coded, the same pattern pahc_curate.fetch_pahc_excess_templates
# already uses for its 145 unchecksummed DL21 files.
#
# Stage 2 (`build_sps_merged_rows` / `build_cifist_grid_rows`, above) is the
# CIFIST branch this section's raw spectra feed into -- D-11 through D-15.

# SVO Theoretical Spectra SSAP listing for this collection: one VOTable row
# per (Teff, logg, [M/H], alpha) grid point (446 total for CIFIST2011_2015),
# each row carrying an `fid` inside its Access.Reference URL. Appending
# "&fid=<N>&format=ascii" to the base URL is the address of one spectrum.
CIFIST_SSAP_URL = "https://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=bt-settl-cifist"
CIFIST_MODEL_URL_FMT = CIFIST_SSAP_URL + "&fid={fid}&format=ascii"

# Selection window: cool edge of Teff=1200 K (well below the Kurucz/ATLAS9
# SPS grid's 3500 K floor) up to 3900 K (a few hundred K of overlap with the
# bottom of that grid), on the grid's 100 K steps only. CIFIST also carries
# 50 K half-steps in part of this range (1550, 1650, ..., 2350 K) that are
# deliberately SKIPPED, not missed -- mixing 50 K and 100 K spacing into one
# Teff axis would make it non-uniform for anything that resamples along it
# later, and 100 K is plenty fine for a photosphere grid. [M/H] and alpha
# are not separate selection constants: this collection carries only
# [M/H]=0, alpha=0 (verified against the full 446-row listing), so
# fetch_btsettl_cifist filters on that literal rather than a named window.
CIFIST_TEFF_MIN_K = 1200
CIFIST_TEFF_MAX_K = 3900
CIFIST_TEFF_STEP_K = 100

# Stefan-Boltzmann constant, cgs (erg s^-1 cm^-2 K^-4; CODATA 2018, exact via
# the SI-defined h, k_B, c). Used only by load_cifist_spectrum's bolometric
# sanity check -- not a project-wide quantity, so it stays local here rather
# than joining constants.py's genuine-physical-constant list.
STEFAN_BOLTZMANN_CGS = 5.670374419e-5

_CIFIST_ANGSTROM_TO_UM = 1e-4

# Bootstrapped sha256 pin: {filename: {"fid", "teff", "logg", "sha256",
# "nbytes_raw"}}. "nbytes_raw" and the hash both describe the RAW
# (uncompressed) ASCII bytes -- the file kept on disk is gzipped (see
# fetch_btsettl_cifist), so this is the one place that size/hash pair is
# recorded before compression throws the byte count away.
_CIFIST_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sps", "btsettl_cifist_manifest.json",
)


def _load_cifist_manifest():
    if os.path.exists(_CIFIST_MANIFEST_PATH):
        with open(_CIFIST_MANIFEST_PATH) as f:
            import json
            return json.load(f)
    return {}


def _save_cifist_manifest(manifest):
    import json
    directory = os.path.dirname(_CIFIST_MANIFEST_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(_CIFIST_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _cifist_filename(teff, logg):
    """Lyon-style model filename, Teff/100 as a zero-padded 3-digit TTT
    (e.g. Teff=1200 -> 'lte012...', Teff=3900 -> 'lte039...')."""
    ttt = f"{int(round(teff / 100.0)):03d}"
    return f"lte{ttt}-{logg:.1f}-0.0a+0.0.BT-Settl.CIFIST2011_2015.dat.gz"


def _select_cifist_models(timeout=120):
    """Parse the CIFIST SSAP VOTable listing and return the rows to fetch:
    [M/H]=0, alpha=0 (this collection's only values), CIFIST_TEFF_MIN_K <=
    Teff <= CIFIST_TEFF_MAX_K, Teff on the CIFIST_TEFF_STEP_K grid. Every
    logg the listing has for an accepted Teff is kept -- the grid has holes
    (e.g. Teff=1400 K has no logg=2.5) and this function does not try to
    fill them. Returns a list of {"teff": int, "logg": float, "fid": int}
    dicts, sorted by (teff, logg). Expect 195 rows."""
    import io
    import re
    import urllib.request

    from astropy.io.votable import parse_single_table

    with urllib.request.urlopen(CIFIST_SSAP_URL, timeout=timeout) as resp:
        votable_bytes = resp.read()
    table = parse_single_table(io.BytesIO(votable_bytes)).to_table()

    rows = []
    for r in table:
        teff = float(r["teff"])
        logg = float(r["logg"])
        meta = float(r["meta"])
        alpha = float(r["alpha"])
        if meta != 0.0 or alpha != 0.0:
            continue
        if not (CIFIST_TEFF_MIN_K <= teff <= CIFIST_TEFF_MAX_K):
            continue
        if int(round(teff)) % CIFIST_TEFF_STEP_K != 0:
            continue
        access_ref = str(r["Access.Reference"])
        m = re.search(r"[?&]fid=(\d+)", access_ref)
        if not m:
            raise ValueError(f"could not parse fid from Access.Reference={access_ref!r}")
        rows.append({"teff": int(round(teff)), "logg": logg, "fid": int(m.group(1))})

    rows.sort(key=lambda d: (d["teff"], d["logg"]))
    return rows


def _gzip_postprocess(partial_path, dest_path):
    """fetch_pinned postprocess hook: gzip-compress the sha256-VERIFIED raw
    ASCII at partial_path into dest_path (~5x smaller), then remove
    partial_path. Used because the pinned hash is of the raw content, not
    of whatever ends up on disk."""
    import gzip
    import shutil
    directory = os.path.dirname(dest_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(partial_path, "rb") as f_in, gzip.open(dest_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(partial_path)


def _gzip_verify_dest(dest_path):
    """fetch_pinned verify_dest hook: sha256 of the UNCOMPRESSED content
    inside an existing gzip file -- the idempotency check has to decompress
    first, since the pin is on the raw ASCII, not on the gzip bytes (which
    are not even deterministic across gzip runs/versions)."""
    import gzip
    import hashlib
    h = hashlib.sha256()
    with gzip.open(dest_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_gzip_bytes(raw_bytes, dest_path):
    import gzip
    directory = os.path.dirname(dest_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with gzip.open(dest_path, "wb") as f:
        f.write(raw_bytes)


def _with_retry(func, *, max_retries=4, base_delay_s=5.0, verbose=True):
    """Call func() with exponential backoff on a NETWORK failure (URLError,
    socket.timeout, ConnectionError/TimeoutError). A sha256 mismatch
    (ValueError, raised by model_io.fetch_pinned) is NOT retried here -- a
    changed or corrupted artifact is a real event to investigate, not a
    transient condition worth another attempt."""
    import socket
    import time
    import urllib.error

    retry_exceptions = (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError)
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except retry_exceptions as exc:
            if attempt == max_retries:
                raise
            delay = base_delay_s * (2 ** (attempt - 1))
            if verbose:
                print(f"    fetch failed (attempt {attempt}/{max_retries}): {exc!r}; "
                      f"retrying in {delay:.0f}s")
            time.sleep(delay)


def fetch_btsettl_cifist(download_dir, *, timeout=180, sleep_between_fetches_s=1.5,
                         max_retries=4, retry_base_delay_s=5.0, verbose=True,
                         max_workers=6):
    """Stage 0 (CIFIST extension): fetch the BT-Settl CIFIST2011_2015 grid's
    Teff=1200-3900 K, 100 K-step selection from the SVO Theoretical Spectra
    service into download_dir/btsettl_cifist/, one gzipped ASCII file per
    (Teff, logg) model (~195 files, ~8.4 GB uncompressed / ~1.7 GB gzipped).

    SELECTION. See _select_cifist_models: [M/H]=0, alpha=0 (this
    collection's only values), Teff in [CIFIST_TEFF_MIN_K,
    CIFIST_TEFF_MAX_K] on the CIFIST_TEFF_STEP_K grid, every logg the
    listing has for an accepted Teff (the grid has holes, e.g. Teff=1400 K
    has no logg=2.5 -- not filled in here).

    NAMING. download_dir/btsettl_cifist/<_cifist_filename(teff, logg)>,
    Lyon-style: 'lte<TTT>-<logg:.1f>-0.0a+0.0.BT-Settl.CIFIST2011_2015.dat.gz',
    TTT = Teff/100 zero-padded to 3 digits.

    PINNING -- "pin on first fetch", not a pre-known hash. Unlike
    galaxy_curate's fetchers, whose sha256 manifests are hard-coded because
    upstream ships fixed, already-published files, nobody has published
    checksums for these SVO-generated ASCII exports. The manifest at
    data/sps/btsettl_cifist_manifest.json is therefore BOOTSTRAPPED, the
    same way pahc_curate.fetch_pahc_excess_templates bootstraps its DL21
    manifest: the first run that meets a filename with no manifest entry
    downloads it, hashes the RAW (uncompressed) bytes, and records {fid,
    teff, logg, sha256, nbytes_raw} -- loudly logged as a first-fetch pin,
    never a silent default. THE MANIFEST IS FROZEN FROM THAT POINT ON:
    every later run treats the recorded hash as the pin and verifies
    against it (via model_io.fetch_pinned, see STORAGE below); a mismatch
    RAISES. That mismatch is a signal to investigate an upstream change --
    it must never be resolved by just updating the recorded hash.

    STORAGE. The cache key (the pinned sha256) is the RAW ascii bytes, but
    the file kept on disk is GZIPPED. model_io.fetch_pinned's contract is
    "verify, then rename exactly the bytes downloaded", which does not fit
    "keep something other than what was hashed", so fetch_pinned gained two
    small, backward-compatible hooks for this (see its docstring):
    `postprocess` (here, `_gzip_postprocess`) gzips the verified raw file
    into `dest` instead of the plain rename; `verify_dest` (here,
    `_gzip_verify_dest`) decompresses an existing `dest` to recompute the
    raw-content hash for the idempotency check. Both are no-ops for every
    other fetch_pinned caller in this project (they default to the
    previous behavior when omitted).

    POLITENESS. Owner ruling 2026-08-24 supersedes this fetcher's original
    serial-only stance: embarrassingly parallel bulk downloads run in
    parallel, so PINNED entries (manifest already has a hash for this
    filename -- the normal steady state once the manifest is frozen) are
    fetched through a modest ThreadPoolExecutor, `max_workers` wide
    (default 6; a 6-worker re-fetch of the full 195-file CIFIST cache
    completed in 19.5 min on 2026-08-24, every file sha256-verified).
    `max_workers=1` reproduces the pre-parallel behaviour exactly, including
    the sleep_between_fetches_s pause after each actual NETWORK fetch
    (never after a cache hit -- an already-verified file costs nothing to
    skip); with `max_workers>1` that inter-fetch sleep is skipped, since the
    pool itself already caps how many requests are in flight. Per-file
    retry/backoff is unchanged either way: up to max_retries attempts with
    exponential backoff (retry_base_delay_s * 2**attempt) on a network
    failure. UNPINNED entries -- the BOOTSTRAP path, first-ever fetch of a
    filename with no manifest entry -- are always processed SERIALLY,
    regardless of max_workers: manifest writes are not thread-safe and pins
    must stay deterministic, so a first-ever bootstrap run stays serial for
    that part even though the pinned part above still parallelises.

    Idempotent: a cached file whose DECOMPRESSED sha256 already matches its
    manifest entry is skipped with no network request at all.

    Returns download_dir/btsettl_cifist.
    """
    import hashlib
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_dir = os.path.join(download_dir, "btsettl_cifist")
    os.makedirs(out_dir, exist_ok=True)
    manifest = _load_cifist_manifest()
    selection = _select_cifist_models(timeout=timeout)

    # Partition first: PINNED rows (manifest already has this filename) are
    # read-only against `manifest` and safe to fan out across the pool;
    # UNPINNED rows mutate `manifest` and must stay serial (see POLITENESS).
    pinned_rows = []
    unpinned_rows = []
    for row in selection:
        filename = _cifist_filename(row["teff"], row["logg"])
        (pinned_rows if manifest.get(filename) is not None else unpinned_rows).append(row)

    def _process_pinned_row(row):
        filename = _cifist_filename(row["teff"], row["logg"])
        dest = os.path.join(out_dir, filename)
        entry = manifest[filename]

        already_ok = os.path.exists(dest) and _gzip_verify_dest(dest) == entry["sha256"]
        if already_ok:
            return "cached"

        def _do_fetch():
            return model_io.fetch_pinned(
                CIFIST_MODEL_URL_FMT.format(fid=row["fid"]), entry["sha256"], dest,
                timeout=timeout, postprocess=_gzip_postprocess,
                verify_dest=_gzip_verify_dest,
            )

        _with_retry(_do_fetch, max_retries=max_retries,
                   base_delay_s=retry_base_delay_s, verbose=verbose)
        if max_workers <= 1:
            time.sleep(sleep_between_fetches_s)
        return "fetched"

    if max_workers <= 1:
        results = [_process_pinned_row(row) for row in pinned_rows]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_pinned_row, row) for row in pinned_rows]
            results = [fut.result() for fut in as_completed(futures)]

    n_cached = results.count("cached")
    n_fetched = results.count("fetched")

    # Bootstrap: no pin exists yet for these files -- download, hash, and
    # record (mirrors pahc_curate.fetch_pahc_excess_templates's DL21
    # bootstrap; see docstring above). Always serial -- see POLITENESS.
    n_pinned = 0
    for row in unpinned_rows:
        filename = _cifist_filename(row["teff"], row["logg"])
        dest = os.path.join(out_dir, filename)

        def _do_bootstrap(row=row):
            url = CIFIST_MODEL_URL_FMT.format(fid=row["fid"])
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()

        raw = _with_retry(_do_bootstrap, max_retries=max_retries,
                          base_delay_s=retry_base_delay_s, verbose=verbose)
        sha256 = hashlib.sha256(raw).hexdigest()
        if verbose:
            print(f"    PIN ON FIRST FETCH: {filename} sha256={sha256} "
                  f"({len(raw)} bytes raw)")
        manifest[filename] = {"fid": row["fid"], "teff": row["teff"], "logg": row["logg"],
                              "sha256": sha256, "nbytes_raw": len(raw)}
        _write_gzip_bytes(raw, dest)
        _save_cifist_manifest(manifest)
        n_pinned += 1
        if max_workers <= 1:
            time.sleep(sleep_between_fetches_s)

    if verbose:
        print(f"    btsettl_cifist: {n_cached} cached, {n_fetched} re-verified/fetched, "
              f"{n_pinned} newly pinned, {len(selection)} total")
    return out_dir


# ====================================================================
# Stage 0 (D-19/D-25, Extension A'): raw-data acquisition, BT-Settl
# AGSS2009 grid
# ====================================================================
# Mirrors the CIFIST Stage 0 section immediately above, byte-for-byte in
# pattern (D-18): model_io.fetch_pinned, gzip storage, a bootstrapped FROZEN
# sha256 manifest (nobody has published checksums for these SVO-generated
# ASCII exports either). Unlike CIFIST's selection (a Teff/logg RANGE with
# whatever logg values the listing happens to carry), AGSS's selection is an
# EXACT 79-node set (D-19) -- the T1 study found the box has no holes, so
# there is nothing to enumerate from the index beyond resolving each of the
# 79 (Teff, logg) pairs to its fid.

AGSS_SSAP_URL = "https://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=bt-settl-agss"
AGSS_MODEL_URL_FMT = AGSS_SSAP_URL + "&fid={fid}&format=ascii"

# D-19: the box -- T_eff 2300-3400 K (100 K steps) x log g -0.5..+2.0
# (0.5 dex steps), solar Z, alpha=0, no holes (72 nodes) -- plus the seven
# nodes that collide with nothing shipped above it: (3500, -0.5) (CK03 has
# no log g < 0 anywhere) and (3600, -0.5..+2.0) (CK03 has nothing at all at
# 3600 K). 72 + 7 = 79.
AGSS_TEFF_MIN_K = 2300
AGSS_TEFF_MAX_K = 3400
AGSS_TEFF_STEP_K = 100
AGSS_LOGG_VALUES = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
AGSS_EXTRA_NODES_TEFF_LOGG = ((3500, -0.5),) + tuple((3600, g) for g in AGSS_LOGG_VALUES)

_AGSS_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sps", "btsettl_agss_manifest.json",
)


def _agss_wanted_nodes():
    """D-19: the exact 79 (T_EFF:int, LOGG:float rounded to 0.1 dex)
    nodes Extension A' fetches -- the 72-node box plus the 7 strictly-
    additive nodes. Sorted by (teff, logg). Raises if the count is ever not
    79 (a guard against a future edit to the constants above silently
    changing the fetch set)."""
    box = [(t, g) for t in range(AGSS_TEFF_MIN_K, AGSS_TEFF_MAX_K + 1, AGSS_TEFF_STEP_K)
          for g in AGSS_LOGG_VALUES]
    nodes = sorted(set(box) | set(AGSS_EXTRA_NODES_TEFF_LOGG))
    if len(nodes) != 79:
        raise AssertionError(f"D-19 fetch set should be 79 nodes, computed {len(nodes)}")
    return nodes


def _agss_filename(teff, logg):
    """Lyon-style model filename, mirroring _cifist_filename's convention
    (Teff/100 as a zero-padded 3-digit TTT) but tagged AGSS2009 so the two
    vintages never collide on disk even if ever fetched into the same
    directory.

    Unlike _cifist_filename (whose grid never carries a negative log g, so
    `-{logg:.1f}` never doubles a minus sign), this uses an EXPLICIT sign
    (`{logg:+.1f}`, same convention as cifist_model_name/agss_model_name's
    `%+.1f`) with no separate literal hyphen -- `lte023-0.5-...` for
    log g=-0.5, `lte036+2.0-...` for log g=+2.0 -- so the sign character is
    unambiguous and a negative log g never collides with `--` doubling."""
    ttt = f"{int(round(teff / 100.0)):03d}"
    return f"lte{ttt}{logg:+.1f}-0.0a+0.0.BT-Settl.AGSS2009.dat.gz"


def _select_agss_models(timeout=120):
    """Resolve the D-19 79-node fetch set against the live AGSS SSAP index.
    Returns [{"teff": int, "logg": float, "fid": int}, ...] sorted by
    (teff, logg). RAISES if the index is missing any of the 79 (solar-Z,
    alpha=0) nodes -- T1 verified the box is complete (101/102 nodes in the
    wider 2000-3600 K box; the single hole, (3500, +1.5), is outside D-19's
    box and never requested here)."""
    import io
    import re
    import urllib.request

    from astropy.io.votable import parse_single_table

    with urllib.request.urlopen(AGSS_SSAP_URL, timeout=timeout) as resp:
        votable_bytes = resp.read()
    table = parse_single_table(io.BytesIO(votable_bytes)).to_table()

    wanted = set(_agss_wanted_nodes())
    by_key = {}
    for r in table:
        teff = float(r["teff"])
        logg = float(r["logg"])
        meta = float(r["meta"])
        alpha = float(r["alpha"])
        if meta != 0.0 or alpha != 0.0:
            continue
        key = (int(round(teff)), round(logg, 1))
        if key not in wanted:
            continue
        access_ref = str(r["Access.Reference"])
        m = re.search(r"[?&]fid=(\d+)", access_ref)
        if not m:
            raise ValueError(f"could not parse fid from Access.Reference={access_ref!r}")
        by_key[key] = {"teff": key[0], "logg": key[1], "fid": int(m.group(1))}

    missing = wanted - set(by_key)
    if missing:
        raise ValueError(
            f"AGSS SSAP index is missing {len(missing)} of the 79 D-19 nodes: "
            f"{sorted(missing)}")

    return [by_key[k] for k in sorted(wanted)]


def _load_agss_manifest():
    if os.path.exists(_AGSS_MANIFEST_PATH):
        with open(_AGSS_MANIFEST_PATH) as f:
            import json
            return json.load(f)
    return {}


def _save_agss_manifest(manifest):
    import json
    directory = os.path.dirname(_AGSS_MANIFEST_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(_AGSS_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def fetch_btsettl_agss(download_dir, *, timeout=180, sleep_between_fetches_s=1.5,
                       max_retries=4, retry_base_delay_s=5.0, verbose=True,
                       max_workers=6):
    """Stage 0 (Extension A', D-19/D-25): fetch the BT-Settl AGSS2009
    grid's exact 79-node D-19 selection from the SVO Theoretical Spectra
    service into download_dir/btsettl_agss/, one gzipped ASCII file per
    (Teff, logg) model. T1 measured ~0.69 GB raw / ~0.14 GB gzipped for all
    79 (files are 1.75 MB at T_eff <= 2500 K, 12.1 MB at T_eff >= 2600 K --
    far smaller than CIFIST's ~43 MB files).

    Mirrors fetch_btsettl_cifist's contract exactly (D-18 pattern -- see
    that function's docstring for the full pin/storage/politeness
    rationale, unchanged here): PINNING is bootstrapped-on-first-fetch (no
    published checksums exist for these SVO exports either) into
    data/sps/btsettl_agss_manifest.json, frozen from that point on; STORAGE
    gzips the sha256-verified raw ASCII via the same `_gzip_postprocess`/
    `_gzip_verify_dest` hooks; POLITENESS is a modest pool (default
    max_workers=6) over PINNED entries per owner ruling 2026-08-24,
    superseding the original serial-only stance -- see
    fetch_btsettl_cifist's docstring for the full max_workers=1 vs
    max_workers>1 sleep-pacing/bootstrap-stays-serial rationale, unchanged
    here -- with per-file retry/backoff on transient failures unaffected.

    SELECTION. See _agss_wanted_nodes/_select_agss_models: the exact 79
    D-19 nodes (T_eff 2300-3400 K x log g -0.5..+2.0, PLUS (3500,-0.5) and
    (3600,-0.5..+2.0)) -- not a range with holes filled in, since T1
    verified this box has none.

    NAMING. download_dir/btsettl_agss/<_agss_filename(teff, logg)>.

    Idempotent: a cached file whose DECOMPRESSED sha256 already matches its
    manifest entry is skipped with no network request at all.

    Returns download_dir/btsettl_agss.
    """
    import hashlib
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_dir = os.path.join(download_dir, "btsettl_agss")
    os.makedirs(out_dir, exist_ok=True)
    manifest = _load_agss_manifest()
    selection = _select_agss_models(timeout=timeout)

    # Partition first: PINNED rows (manifest already has this filename) are
    # read-only against `manifest` and safe to fan out across the pool;
    # UNPINNED rows mutate `manifest` and must stay serial (see POLITENESS).
    pinned_rows = []
    unpinned_rows = []
    for row in selection:
        filename = _agss_filename(row["teff"], row["logg"])
        (pinned_rows if manifest.get(filename) is not None else unpinned_rows).append(row)

    def _process_pinned_row(row):
        filename = _agss_filename(row["teff"], row["logg"])
        dest = os.path.join(out_dir, filename)
        entry = manifest[filename]

        already_ok = os.path.exists(dest) and _gzip_verify_dest(dest) == entry["sha256"]
        if already_ok:
            return "cached"

        def _do_fetch():
            return model_io.fetch_pinned(
                AGSS_MODEL_URL_FMT.format(fid=row["fid"]), entry["sha256"], dest,
                timeout=timeout, postprocess=_gzip_postprocess,
                verify_dest=_gzip_verify_dest,
            )

        _with_retry(_do_fetch, max_retries=max_retries,
                   base_delay_s=retry_base_delay_s, verbose=verbose)
        if max_workers <= 1:
            time.sleep(sleep_between_fetches_s)
        return "fetched"

    if max_workers <= 1:
        results = [_process_pinned_row(row) for row in pinned_rows]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_pinned_row, row) for row in pinned_rows]
            results = [fut.result() for fut in as_completed(futures)]

    n_cached = results.count("cached")
    n_fetched = results.count("fetched")

    # Bootstrap: no pin exists yet for these files -- download, hash, and
    # record (mirrors fetch_btsettl_cifist's own bootstrap). Always serial
    # -- see POLITENESS.
    n_pinned = 0
    for row in unpinned_rows:
        filename = _agss_filename(row["teff"], row["logg"])
        dest = os.path.join(out_dir, filename)

        def _do_bootstrap(row=row):
            url = AGSS_MODEL_URL_FMT.format(fid=row["fid"])
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()

        raw = _with_retry(_do_bootstrap, max_retries=max_retries,
                          base_delay_s=retry_base_delay_s, verbose=verbose)
        sha256 = hashlib.sha256(raw).hexdigest()
        if verbose:
            print(f"    PIN ON FIRST FETCH: {filename} sha256={sha256} "
                  f"({len(raw)} bytes raw)")
        manifest[filename] = {"fid": row["fid"], "teff": row["teff"], "logg": row["logg"],
                              "sha256": sha256, "nbytes_raw": len(raw)}
        _write_gzip_bytes(raw, dest)
        _save_agss_manifest(manifest)
        n_pinned += 1
        if max_workers <= 1:
            time.sleep(sleep_between_fetches_s)

    if verbose:
        print(f"    btsettl_agss: {n_cached} cached, {n_fetched} re-verified/fetched, "
              f"{n_pinned} newly pinned, {len(selection)} total")
    return out_dir


# ====================================================================
# Stage 0 (D-26 amendment, 2026-08-24): raw-data acquisition, CK03 source
# library
# ====================================================================
# D-26 originally moved the CK03 source library (models.conf,
# parameters.fits, seds/, convolved/ -- 3,856 files, 98.4 MB) into
# download_dir/ck03/ as a hand-copied, NON-purgeable archive, on the
# premise that "it has no fetcher and no live upstream". The owner
# subsequently identified a live upstream -- Tom Robitaille's sedfitter
# model package on UW-Madison's institutional FTP -- and verified on
# 2026-08-24 that its extracted contents are byte-identical (3,856/3,856
# sha256 matches) to both the committed manifest and the on-disk archive.
# `fetch_ck03` below makes the archive PROGRAMMATICALLY reproducible; it
# does NOT change the purge exemption (see purge_downloads, unchanged, and
# the D-26 amendment in docs/sps_decisions.md for why the exemption stands
# regardless).
#
# UNLIKE fetch_btsettl_cifist/fetch_btsettl_agss (D-18/D-25), whose
# per-file manifests are BOOTSTRAPPED because nobody published checksums
# for those SVO-generated exports, this is a single, already-published,
# frozen 2011 tarball -- so its package-level sha256 is HARD-CODED here,
# the same way galaxy_curate.py's POLLETTA_FILE_SHA256 / BERTA_FILE_SHA256
# / BROWN_*_SHA256 pins are (D-... galaxy source pins). The per-file
# manifest it unpacks into (data/sps/ck03_manifest.json) was ALREADY
# committed by D-26's original hand-copy and is not touched here.

# ftp:// works with model_io.fetch_pinned as-is: it downloads via
# urllib.request.urlopen, whose default opener has an FTPHandler built in.
CK03_PACKAGE_URL = (
    "ftp://ftp.astro.wisc.edu/outgoing/tom/model_packages/models_kurucz_05sep11.tgz"
)
CK03_PACKAGE_SHA256 = "c8013a3f1193852cc844a5951aa2e1408837b5e871e2e401b5bc9221e676b1e0"
CK03_PACKAGE_NBYTES = 87108714

# The tgz's single top-level directory; its CONTENTS (not the directory
# itself) are what land in download_dir/ck03/ -- fetch_ck03 flattens one
# level so the result is download_dir/ck03/models.conf, not
# download_dir/ck03/models_kurucz/models.conf.
CK03_PACKAGE_ROOT_DIRNAME = "models_kurucz"

_CK03_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sps", "ck03_manifest.json",
)


def _load_ck03_manifest():
    """Load the committed {relative_path: {"sha256", "nbytes"}} manifest
    (see _CK03_MANIFEST_PATH), mirroring _load_cifist_manifest's path
    construction. Unlike that function this manifest is not optional --
    it is a byte-for-byte record of an already-copied archive and is
    expected to always be present."""
    import json
    with open(_CK03_MANIFEST_PATH) as f:
        return json.load(f)


def _sha256_of_file(path, chunk_bytes=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_ck03_tree(root_dir, manifest):
    """Verify every file `manifest` records exists under root_dir with a
    matching sha256. Returns a list of (relative_path, reason) problems
    ("missing" or a mismatch description); an empty list means root_dir
    fully satisfies manifest."""
    problems = []
    for rel_path, entry in manifest.items():
        full_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(full_path):
            problems.append((rel_path, "missing"))
            continue
        got = _sha256_of_file(full_path)
        if got != entry["sha256"]:
            problems.append(
                (rel_path, f"sha256 mismatch (expected {entry['sha256']}, got {got})"))
    return problems


def _safe_extractall(tar, dest_dir):
    """tarfile.extractall without a `filter=` argument (Python 3.9 has none)
    would follow a member path outside dest_dir if the tarball were
    malicious; CK03_PACKAGE_SHA256 already pins this specific, trusted
    tarball, but this check is cheap insurance against a corrupted or
    substituted download slipping past to the filesystem."""
    dest_abs = os.path.abspath(dest_dir)
    for member in tar.getmembers():
        member_abs = os.path.abspath(os.path.join(dest_dir, member.name))
        if member_abs != dest_abs and not member_abs.startswith(dest_abs + os.sep):
            raise ValueError(f"ck03: tar member escapes extraction dir: {member.name!r}")
    tar.extractall(dest_dir)


def fetch_ck03(download_dir, *, timeout=600, max_retries=4, retry_base_delay_s=5.0,
               verbose=True):
    """Stage 0 (D-26 amendment): fetch the CK03 (Castelli & Kurucz 2003
    ATLAS9) source library -- Tom Robitaille's sedfitter "models_kurucz"
    package -- into download_dir/ck03/ (models.conf, parameters.fits,
    seds/, convolved/; 3,856 files).

    CACHE HIT (idempotent, no network). If download_dir/ck03/ already
    exists, every file data/sps/ck03_manifest.json records is checked
    against it by sha256; if all 3,856 match, this returns immediately with
    no network request. Any missing/mismatched files are reported (up to 20,
    then a count) before falling through to a full re-fetch below -- this
    function never trusts a partially-correct cache silently.

    FETCH. download_dir/ck03/ missing or failing verification triggers a
    fresh fetch:
      1. The tgz is downloaded via model_io.fetch_pinned against the
         HARD-CODED CK03_PACKAGE_SHA256 (a single already-published, frozen
         2011 tarball -- unlike the CIFIST/AGSS per-file manifests, there is
         nothing here to bootstrap). fetch_pinned's own dest-exists-and-
         verifies short-circuit makes a partially-completed prior tgz
         download resumable-in-effect (re-verified, not re-downloaded, if it
         already matches). A network failure during download is retried
         (_with_retry, up to max_retries with exponential backoff); a
         sha256 MISMATCH is not retried -- it is raised immediately, before
         any extraction, and nothing on disk is touched (fetch_pinned only
         ever removes its own `.part` scratch file).
      2. The verified tgz is extracted into a scratch directory, its
         CK03_PACKAGE_ROOT_DIRNAME root located, and EVERY file the manifest
         records is sha256-verified against that extracted root -- a second,
         independent check on top of the package-level hash, since the
         package hash alone would not catch e.g. a re-packaged tgz with the
         same overall bytes-on-the-wire but a shuffled/renamed file inside.
         Any mismatch (there should never be one, given step 1's package
         hash) raises loudly and DELETES NOTHING -- the tgz and the scratch
         extraction are left on disk for inspection.
      3. Only once every file verifies does this replace download_dir/ck03/
         (flattening the root dir's contents up one level) and delete the
         tgz -- the extracted, manifest-verified tree is the cache, not the
         tgz.

    Returns download_dir/ck03. Does NOT touch purge_downloads or the
    driver's build path -- see the module-level comment above this
    function and docs/sps_decisions.md's D-26 amendment.
    """
    import shutil
    import tarfile

    out_dir = os.path.join(download_dir, "ck03")
    manifest = _load_ck03_manifest()

    if os.path.isdir(out_dir):
        problems = _verify_ck03_tree(out_dir, manifest)
        if not problems:
            if verbose:
                print(f"    ck03: cache hit, {len(manifest)}/{len(manifest)} files "
                      f"verified, no network")
            return out_dir
        if verbose:
            print(f"    ck03: existing {out_dir} failed verification "
                  f"({len(problems)} problem(s)) -- re-fetching:")
            for rel_path, reason in problems[:20]:
                print(f"      {rel_path}: {reason}")
            if len(problems) > 20:
                print(f"      ... and {len(problems) - 20} more")

    os.makedirs(download_dir, exist_ok=True)
    tgz_path = os.path.join(download_dir, "ck03_models_kurucz_05sep11.tgz")

    def _do_fetch_tgz():
        return model_io.fetch_pinned(CK03_PACKAGE_URL, CK03_PACKAGE_SHA256, tgz_path,
                                     timeout=timeout)

    _with_retry(_do_fetch_tgz, max_retries=max_retries,
               base_delay_s=retry_base_delay_s, verbose=verbose)

    got_nbytes = os.path.getsize(tgz_path)
    if got_nbytes != CK03_PACKAGE_NBYTES:
        raise ValueError(
            f"{CK03_PACKAGE_URL}: size mismatch after a sha256-verified download -- "
            f"expected {CK03_PACKAGE_NBYTES} bytes, got {got_nbytes}. The size and "
            "sha256 pins disagree about the same published artifact; investigate "
            "before touching either.")

    if verbose:
        print(f"    ck03: package sha256 verified ({got_nbytes} bytes), extracting...")

    extract_scratch = os.path.join(download_dir, "ck03_extract_scratch")
    if os.path.isdir(extract_scratch):
        shutil.rmtree(extract_scratch)
    os.makedirs(extract_scratch)
    with tarfile.open(tgz_path, "r:gz") as tf:
        _safe_extractall(tf, extract_scratch)

    extracted_root = os.path.join(extract_scratch, CK03_PACKAGE_ROOT_DIRNAME)
    if not os.path.isdir(extracted_root):
        raise ValueError(
            f"{CK03_PACKAGE_URL}: expected a '{CK03_PACKAGE_ROOT_DIRNAME}/' root "
            f"directory after extraction, found: {sorted(os.listdir(extract_scratch))}")

    problems = _verify_ck03_tree(extracted_root, manifest)
    if problems:
        detail = "\n".join(f"  {rel}: {reason}" for rel, reason in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ValueError(
            f"ck03: {len(problems)}/{len(manifest)} file(s) failed manifest "
            f"verification after extraction. Nothing deleted -- {tgz_path!r} and "
            f"{extract_scratch!r} are left on disk for inspection:\n{detail}{more}")

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    shutil.move(extracted_root, out_dir)
    shutil.rmtree(extract_scratch, ignore_errors=True)
    os.remove(tgz_path)

    if verbose:
        print(f"    ck03: {len(manifest)}/{len(manifest)} files verified, fetched fresh")
    return out_dir


def purge_downloads(download_dir):
    """Delete the btsettl_cifist AND btsettl_agss raw-download caches under
    download_dir/ (D-18, D-25).

    NOT wired into any driver -- callers invoke this explicitly (e.g. after
    a build that has consumed the caches) to reclaim disk space. Safe to
    call: both caches are fully reproducible from their sha256 manifests
    (data/sps/btsettl_cifist_manifest.json, data/sps/btsettl_agss_manifest.json
    -- every file's sha256, of the raw content, is pinned there), so nothing
    deleted here is a source of truth. Idempotent: a missing directory is
    not an error. Returns (cifist_dir, agss_dir), the two directory paths
    that were (or would have been) removed.
    """
    import shutil
    cifist_dir = os.path.join(download_dir, "btsettl_cifist")
    agss_dir = os.path.join(download_dir, "btsettl_agss")
    for out_dir in (cifist_dir, agss_dir):
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
    return cifist_dir, agss_dir


def _parse_cifist_ascii(path):
    """Raw parse of one gzipped btsettl_cifist file, no bolometric check:
    header teff/logg, the lambda=0 sentinel row dropped, wavelength grid
    asserted strictly increasing. Split out of load_cifist_spectrum so a
    caller auditing the WHOLE fetched grid (e.g. a verification report) can
    still get every file's (wavelength_um, flux, teff, logg, ratio) even for
    the handful that fail load_cifist_spectrum's stricter bolometric gate --
    see that function's docstring for why those failures are real grid
    physics, not parse errors, and are worth reporting rather than hiding
    behind an exception.

    Returns (wavelength_um, f_lambda_cgs, teff, logg). Raises ValueError on
    a missing header or a non-monotonic wavelength grid -- both genuine
    parse failures, unlike the bolometric check.
    """
    import gzip
    import re

    teff = logg = None
    wave_a, flux = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"teff\s*=\s*([\d.]+)", line)
                if m:
                    teff = float(m.group(1))
                # D-21: same sign fix as _parse_cifist_ascii_fast -- see that
                # function's comment for the full rationale.
                m = re.search(r"logg\s*=\s*([-+\d.]+)", line)
                if m:
                    logg = float(m.group(1))
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            wave_a.append(float(parts[0]))
            flux.append(float(parts[1]))

    if teff is None or logg is None:
        raise ValueError(f"{path}: could not parse teff/logg from header comments")

    wave_a = np.asarray(wave_a, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if wave_a.size and wave_a[0] == 0.0:
        wave_a = wave_a[1:]
        flux = flux[1:]

    if not np.all(np.diff(wave_a) > 0):
        raise ValueError(f"{path}: wavelength grid is not strictly increasing "
                         "after dropping the lambda=0 sentinel row")

    return wave_a * _CIFIST_ANGSTROM_TO_UM, flux, teff, logg


def cifist_bolometric_ratio(wavelength_um, f_lambda_cgs, teff):
    """int F_lambda dlambda / (sigma * Teff^4), the identity that DEFINES
    Teff for a self-consistent model atmosphere -- 1.0 for an exact
    blackbody, and the quantity load_cifist_spectrum's sanity check is
    built on. Split out so both that check and a caller reporting on files
    it rejects (see _parse_cifist_ascii) compute it identically."""
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    wave_a = np.asarray(wavelength_um, dtype=float) / _CIFIST_ANGSTROM_TO_UM
    f_bol = trapz(f_lambda_cgs, wave_a)  # erg/s/cm^2: int F_lambda[erg/s/cm2/A] dlambda[A]
    return f_bol / (STEFAN_BOLTZMANN_CGS * teff ** 4)


def load_cifist_spectrum(path):
    """Load one gzipped BT-Settl CIFIST ASCII spectrum written by
    fetch_btsettl_cifist. Returns (wavelength_um, f_lambda_cgs, teff, logg).

    wavelength_um is AIR wavelength (SVO's convention for this collection --
    the 2025-05-27 SVO wavelength-rounding fix predates this pull, so no
    correction for that is needed or applied here). f_lambda_cgs is F_lambda
    [erg/s/cm2/Angstrom] AT THE STELLAR SURFACE, not diluted by distance --
    directly comparable to sigma*Teff^4 (see the sanity check below).

    The file's first data row is a lambda=0 SENTINEL, not a real wavelength
    point, and is dropped (in _parse_cifist_ascii). The remaining wavelength
    grid is asserted STRICTLY INCREASING; a violation raises rather than
    being silently sorted, since an out-of-order native grid would indicate
    something wrong with the file rather than something to paper over.

    SANITY CHECK. Integrating f_lambda_cgs over wavelength should reproduce
    the bolometric blackbody flux sigma*Teff^4 (cifist_bolometric_ratio) to
    within a physically loose 0.8-1.2 -- BT-Settl photospheres are not
    blackbodies, and Teff itself is only DEFINED via this identity for a
    self-consistent atmosphere model, so some departure is expected, not a
    defect. RAISES outside that band; a tighter 0.98-1.02 logs a warning
    rather than failing.

    MEASURED ON THE FULL FETCHED GRID (all 195 files, not assumed): 186/195
    land inside [0.8, 1.2] (median ratio 0.990); 74 of those 186 land
    outside the tighter [0.98, 1.02] warn band and log a warning. The other
    9 RAISE here, and they are not corrupt or truncated -- every one has
    full ~1000 um wavelength coverage (checked directly) -- they are a real,
    physical corner of the grid: Teff in {1300,1500,1600,1700,1800,1900} K
    at logg in {2.5, 3.0} ONLY, i.e. the coolest + lowest-gravity corner of
    this selection, where the ratio falls as low as 0.518 (Teff=1900 K,
    logg=2.5). This is a genuine BT-Settl characteristic at that combination
    (extended, low-gravity atmospheres with strong non-blackbody structure),
    not a defect in this fetcher or loader -- but it does mean
    load_cifist_spectrum does not load every file in the grid, by design,
    and a caller that needs to inspect one of those 9 anyway should use
    _parse_cifist_ascii + cifist_bolometric_ratio directly rather than
    working around this function's raise.
    """
    wavelength_um, flux, teff, logg = _parse_cifist_ascii(path)
    ratio = cifist_bolometric_ratio(wavelength_um, flux, teff)

    if not (0.8 <= ratio <= 1.2):
        raise ValueError(
            f"{path}: bolometric sanity check failed -- "
            f"int F_lambda dlambda / (sigma Teff^4) = {ratio:.3f}, outside [0.8, 1.2] "
            "(see docstring: expected for the coolest/lowest-logg corner of "
            "this grid, not necessarily a parse bug or truncation)")
    if not (0.98 <= ratio <= 1.02):
        warnings.warn(
            f"{path}: bolometric ratio {ratio:.4f} outside [0.98, 1.02] "
            "(still within the [0.8, 1.2] pass band -- see docstring)")

    return wavelength_um, flux, teff, logg


# ====================================================================
# Verification-only: a reference convolution independent of sedfitter's own
# ====================================================================
# NOT used by curate_sps_model_set / build_convolved_bands -- the pipeline
# convolves through sedfitter's own convolve_model_dir (D-9), unchanged by
# Extension A. These two functions exist purely so a build's convolved/
# band fluxes can be spot-checked against an INDEPENDENT integration: fine
# log-log interpolation of F_nu on a grid spanning the filter's support,
# rather than sedfitter's own approach of rebinning the FILTER onto the
# MODEL's (coarser) grid. Ported from splice_study/splice_lib.py's
# loglog_interp_fnu / reference_convolve, which the design study (D-13)
# validated reproduces convolved/*.fits to ratio=1.000000 for a
# fully-resolved CK03 model.

def loglog_interp_fnu(nu_out_asc, nu_in_asc, f_in):
    """Log-log interpolation of F_nu(nu), both arrays ASCENDING in
    frequency. Outside nu_in_asc's coverage -> 0. Unlike
    `interp_loglog_zero_preserving` (the production CK03 branch), this
    falls back to LINEAR interpolation across any non-positive stretch --
    matching splice_lib's reference-integration convention, which needs a
    smooth interpolant on the fine grid rather than the CK03 pipeline's
    hard zero-preservation rule. Verification-only; used by
    `reference_convolve`'s fine integration grid."""
    nu_out_asc = np.asarray(nu_out_asc, dtype=float)
    nu_in_asc = np.asarray(nu_in_asc, dtype=float)
    f_in = np.asarray(f_in, dtype=float)
    out = np.zeros_like(nu_out_asc)
    inside = (nu_out_asc >= nu_in_asc[0]) & (nu_out_asc <= nu_in_asc[-1])
    if not np.any(inside):
        return out
    pos = f_in > 0
    if np.all(pos):
        out[inside] = np.exp(np.interp(np.log(nu_out_asc[inside]),
                                       np.log(nu_in_asc), np.log(f_in)))
    else:
        lin = np.interp(nu_out_asc[inside], nu_in_asc, f_in)
        out[inside] = lin
        if np.any(pos):
            sub = np.exp(np.interp(np.log(nu_out_asc[inside]),
                                   np.log(nu_in_asc[pos]), np.log(f_in[pos])))
            near = np.interp(nu_out_asc[inside], nu_in_asc, pos.astype(float)) > 0.999
            out[inside] = np.where(near, sub, lin)
    return out


def reference_convolve(wave_um_asc, fnu, band_name, npt=6000):
    """int(F_nu R dnu) / int(R dnu) on a fine grid spanning band_name's
    filter support -- the reference sedfitter's own grid-based convolution
    (rebin the FILTER onto the MODEL grid, then sum) is spot-checked
    against. `wave_um_asc`/`fnu` must be ASCENDING in wavelength.

    The integration grid is the union of the filter's own nodes, the
    model's own nodes inside the filter, and a dense log grid -- the union
    with the model's nodes is load-bearing (splice_lib's own finding):
    without it, a structured source spectrum is silently resampled and the
    integral stops converging (measured at 0.6% on CIFIST). Ported from
    splice_study/splice_lib.py:reference_convolve.
    """
    from sesnaimpute.sed_models.data_loader import load_filter_curve

    fc = load_filter_curve(band_name)
    order = np.argsort(fc.wave_um)
    nu_f = C_UM_S / fc.wave_um[order][::-1]
    r_f = fc.response[order][::-1]

    wave_um_asc = np.asarray(wave_um_asc, dtype=float)
    nu_m = C_UM_S / wave_um_asc[::-1]
    f_m = np.asarray(fnu, dtype=float)[::-1]

    i0, i1 = np.searchsorted(nu_m, [nu_f[0], nu_f[-1]])
    grid = np.unique(np.concatenate([
        nu_f, nu_m[max(i0 - 1, 0):min(i1 + 1, len(nu_m))],
        np.logspace(np.log10(nu_f[0]), np.log10(nu_f[-1]), npt)]))
    grid = grid[(grid >= nu_f[0]) & (grid <= nu_f[-1])]
    rg = np.interp(grid, nu_f, r_f)
    fg = loglog_interp_fnu(grid, nu_m, f_m)
    return float(np.trapz(fg * rg, grid) / np.trapz(rg, grid))

