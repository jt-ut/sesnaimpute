"""
pahc_curate.py
====================================================================
Curation pipeline for the PAH-contaminant (PAH-C) model set: a Kurucz
photosphere plus a Draine et al. (2021) PAH excess, producing a
sedfitter-ready, APERTURE-INDEPENDENT model directory (models.conf,
parameters.fits, flux.fits, convolved/{band}.fits). See
sed_models_curate/docs/contaminant_pahc_spec.md for the design rationale
("the why"); this module is "the how", written entirely from that spec.

A PAH-C model is a dim field star whose photometric aperture caught
resolved structured PAH cloud emission, adding a spurious 5.8/8.0 um
excess. A star must be present: the excess-to-star ratio R is the
primary axis, and the pure-knot regime is a different population.

Pipeline, in sequence (these stage numbers are the ones the driver's
banners and the spec's "Building the library" list use):

1. DL21 PAH-excess templates (fetch_pahc_excess_templates /
   load_pahc_excess_shape). Fetches the summed emission spectra for one
   illuminating spectrum against a per-file checksum manifest, and
   reads the excess SHAPE for one PAH size distribution. DL21 ships
   nu*P_nu, so the loader divides by nu once -- a factor of frequency
   does not cancel in the shape ratio P_hat(lambda)/P_hat(8um) the
   composite uses. The shape is normalised at 8um on DL21's own grid
   before any resampling, which is required rather than tidy: DL21's
   native P_nu lies entirely below FLUX_FLOOR and would otherwise clip
   to a flat line without raising. Native blue edge is 1.0056 um; the
   PAH-C grid starts at 0.2 um, so the excess is zero-padded there.

2. Stellar photosphere (load_sps_stellar_sed): reads a chosen
   (T_eff, log_g, [Z/H]) row directly from the existing SPS flux.fits.
   No per-model rescaling -- the absolute normalisation is arbitrary and
   cancels into the fit's free scale (spec section 2.1).

3. Wavelength grid (build_pahc_wavelength_grid) and resampling
   (resample_sed_log_log): 0.2-45 um, 403 points, log-spaced ascending
   internally, flipped to descending only when writing flux.fits. Both
   SPS and DL21 are resampled onto it by log-log interpolation, zero
   outside each source's native coverage -- never extrapolated.

4. Composite physics (compute_composite_sed, band_aperture_scaling):

       M(lambda) = s(lambda)
                   + R * s(8um) * [P_hat(lambda)/P_hat(8um)] * a(lambda)

   The star is a point source and carries no aperture factor. The excess
   is extended, so it carries a(lambda) = (theta(lambda)/theta_IRAC)^alpha,
   the ratio of that band's survey aperture to IRAC's, from
   constants.BANDS. R is anchored in the IRAC beam, where the ratio is 1,
   so M(8um)/s(8um) = 1 + R exactly.

5. Raw parameter grid (build_pahc_raw_grid, OWNER LIBRARY-SPEC REBUILD,
   2026-10-08): host (T_eff, log g) x alpha x PAH size x R. alpha, PAH
   size and R are UNCHANGED module constants (8-band leverage/degeneracy
   measurement, spec section 7). The HOST axis is no longer a
   hand-picked anchor list -- see load_pahc_host_grid: hosts are every
   (T_eff, log g) row the SPS RAW photosphere splice carries inside the
   field population's own window (PAHC_HOST_TEFF_MIN_K/MAX_K,
   PAHC_HOST_LOGG_MIN/MAX), fixing the defect that a closed 6000 K /
   log g in {1.0, 3.0} anchor list pinned real hotter/other-gravity
   hosts to the grid edge (measured: 46% of confident Orion A PAHC
   sources on the two hottest anchors). This raw grid (321 hosts x 11
   alpha x 3 size x 32 R = 338,976 models, 2026-10-08 build) is NEVER
   written whole -- it exists only to be sampled, next.

5.5 SED-space sampling (build_pahc_raw_flux_and_coords, greedy_r_net,
   pahc_member_statistics -- owner library spec, 2026-10-08): one global
   greedy r-net at radius SIGEFF (one photometric noise length,
   sed_models_register.density.build_quotient_space, READ-ONLY -- NOT
   sed_models_register.noise, retired, not read here) over the 8
   SESNA-band log10 fluxes of every raw model in stage 5, projected onto
   the 5-D quotient space (gray/scale + both Av laws removed). Convolved
   band flux is computed through the EXACT linear separability of
   compute_composite_sed in R (band convolution is a linear operator;
   see build_pahc_raw_flux_and_coords's docstring), not through 338,976
   individual SED convolutions. greedy_r_net itself is MEMORY-BOUNDED --
   a lazy-heap greedy that holds only O(n) per-model counts, never a
   materialised neighbour adjacency (see its own docstring for why that
   distinction is load-bearing at this point count). The kept templates
   are the raw models that SURVIVE the r-net; every other raw model is
   Voronoi-assigned to its nearest kept template, forming that
   template's MEMBERS set (count, subclass fractions,
   T_EFF/LOGG/ALPHA/PAH_SIZE/R ranges -- written to members.fits). No
   interpolation, no population weight, no prior or Gaia quantity enters
   this step; the kept COUNT is whatever the r-net yields, not a target.

6. Assemble flux.fits/parameters.fits/models.conf/classmap.fits through
   the shared writers in model_io, now for the KEPT templates only
   (point values -- schema parity with every other library). VALUES is
   (n_models, 1, n_wav) -- the point-source aperture axis of length 1
   that every aperture-independent library in the release carries.
   UNCERTAINTIES are all zero. classmap.fits is built from the SAME
   in-memory model set, never by re-reading parameters.fits, since the
   two carry overlapping information and could otherwise disagree. This
   library declares CLASS = PAHC and defines no subclass (spec section
   4.1). members.fits (stage 5.5's output) is additional, not part of
   the release contract profile.

7. convolved/{band}.fits: reuses model_convolution.build_convolved_band_fits
   directly; PAH-C SEDs are smooth continua, so no library-specific
   convolution code is needed.

8. Construction-correctness validation, spec section 8 items 1-4:
   validate_anchor_identity (1), validate_band_aperture_behavior (2),
   validate_unit_scale_sanity (3), validate_convolution_consistency (4).
   Item 4 matters because nothing else in the build reads convolved/.
   Item 6 (grid-density normalisation) is a fit-time weighting concern,
   out of scope for a curation module.

   OWNER LIBRARY-SPEC REBUILD NOTE: items 2 and 3 test properties of
   compute_composite_sed itself (the alpha/aperture-ratio law; hotter
   is brighter) and NEED a full factorial sub-grid to find comparable
   pairs (same host varying alpha; same alpha/size/R varying host) --
   structure the sparse, r-net-sampled KEPT set no longer carries by
   construction. build_pahc_validation_grid builds a small, dedicated
   full-factorial "cross" (one host x every alpha x size x R, union
   every host x one fixed alpha/size/R) purely for this in-memory,
   pre-write check; it is never written to disk. Items 1 and 4 are
   per-model / grid-structure-independent and still run against the
   final, written, KEPT library exactly as before.

9. Scientific validation against Gutermuth's PAH-aperture cut
   (pahc_locus_statistics / plot_pahc_locus /
   validate_pahc_locus_vs_gutermuth), spec section 8 items 5 and 7. Run
   from scripts/validate_pahc_vs_gutermuth.py, not the build driver.

WHY APERTURE-INDEPENDENT, given the excess is extended: sedfitter fits a
free scale only for aperture-independent models (models.py Models.fit:
the ndim == 3 branch solves A_V alone). The aperture physics is kept as
the band-ratio factor above; what is dropped is the stored aperture axis
and the fitted distance, because the region distance factorises out of
the growth law and is absorbed into R. See spec section 2.3.
"""

import glob
import gzip
import os
import tarfile
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from astropy.io import fits

from sesnaimpute.sed_models.astro_utils import vega_color, color_error, reddening_vector_per_ak
from sesnaimpute.sed_models.constants import (APERTURE_GRID_AU, BANDS, C_UM_S, GUTERMUTH_LABELS,
                                     POINT_SOURCE_APERTURE_AU,
                                     PC_CM, REFERENCE_DISTANCE_CM)
from sesnaimpute.sed_models.data_loader import load_filter_curve
from sesnaimpute.sed_models.curate import model_io
# The shared SEDFIT_INPUT reader. It resolves the CLASS filter to row
# coordinates and reads only the matching rows, which on the 1.5 GB
# Cygnus X file is the difference between reading ~1000 rows and reading
# the whole table, and it returns all 8 bands.


# ====================================================================
# CONSTANTS
# ====================================================================
#
# Every number a reader might otherwise find as a bare literal in a
# function body, with its spec backlink. The five grid axes below are
# FIXED BY EMPIRICAL CALIBRATION (spec section 7): a colour-step sweep
# against the 0.05 mag per-colour error floor, half-floor target
# ~0.025 mag. They live here to be VISIBLE, not to be tunable -- the
# builders take no axis arguments, deliberately. Re-deriving an axis
# means re-running that calibration and rewriting spec section 7, not
# passing a keyword.

# --- output contract (spec sections 3, 4) ---
MODEL_NAME_FORMAT = model_io.MODEL_NAME_FORMAT   # "34A", the release-wide cube width
FLUX_FLOOR = 1e-30            # mJy; clip before log10 in the resamplers. NOT the
                              # point-source aperture sentinel of the same magnitude
                              # in constants.py -- semantically unrelated, do not alias.
ANCHOR_WAVELENGTH_UM = 8.0    # the 8um excess/star anchor, spec section 2.2
DEFAULT_MODEL_SET_NAME = ("PAH-aperture contaminant composites "
                          "(Kurucz photosphere + DL21 PAH excess)")

# --- classmap.fits vocabulary (spec section 4.1) --------------------
# The CLASS id, settled across the five libraries rather than chosen
# here: abbrev where a library maps to exactly one GUTERMUTH_LABELS row,
# pop_abbrev where it spans several. PAH-C maps to exactly one (code 39,
# PAH_APERTURE), so the model class and the source label share one token.
# `pop_abbrev` would be CONT, which H2-shock also carries -- it does not
# identify a library.
MODEL_CLASS = "PAHC"

# Emitted in full even though this library has one class, so a
# multi-library aggregation can concatenate CLASS_LEGEND tables and
# recover the whole vocabulary without a lookup held somewhere else.
CLASS_LEGEND = (
    ("PAHC", "Field star whose aperture contains extended PAH emission"),
)

# THIS LIBRARY DEFINES NO SUBCLASS, and that is a decision, not an
# omission. Subclasses are authored, not discovered: the question is not
# whether the grid has discrete axes but whether any of them names a
# distinct POPULATION a consumer should aggregate on.
#
# None does. Every model here is the same physical object -- a field star
# seen through aperture PAH emission -- and the four axes are knobs on
# that one construction. T_EFF and R are continuous, so any split would
# freeze an arbitrary threshold into the authoritative label file when R
# already ships as a column anyone can cut where they like. PAH_SIZE is
# the tempting one, being discrete and three-valued, but it is a model
# INPUT: the same contaminated star is fit by all three, so subclassing
# on it would report three PAH-C populations that do not exist. That is
# exactly what H2-shock's SUBCLASS_NOTE warns against when it records its
# own subclass as "an OUTCOME of the shock solution, not a grid axis".
#
# The token is a COORDINATED choice, shared with any other library
# reaching the same conclusion (SPS is the likely one), because
# aggregation code should need no per-library special case. NONE over an
# empty string, which FITS cannot express as a zero-width column and
# which readers turn into null inconsistently; and over repeating the
# class, which makes a groupby report a subclass that was never defined.
PAHC_SUBCLASS = "NONE"
SUBCLASS_LEGEND = (
    ("NONE", "No subclass: one population, distinguished only by grid axes"),
)

# --- wavelength grid (spec section 3.1) ---
# 0.2 um, not the 0.8 the eight SESNA bands alone would need: Gaia G
# spans 0.327-1.050 um, and 84% of its response weight lies blueward of
# 0.8. A grid that stops at 0.8 does not error on a G convolution -- the
# resampler returns exactly zero outside the native coverage while the
# band denominator still integrates the full response, so it returns a
# plausible flux 2.1 mag too faint. 0.2 matches H2-shock (galaxy uses
# 0.3); the extra margin over G's blue edge is nearly free and this is a
# limit that has to be right for filters nobody has asked for yet.
#
# The extension costs no physics: SPS spans 0.0091-1e6 um so the
# photosphere is fully sampled there, and the DL21 excess is exactly
# zero below 1.0056 um. In the G band a PAH-C source IS its star.
PAHC_WAVE_MIN_UM = 0.2
PAHC_WAVE_MAX_UM = 45.0
# 403 points holds the log step at the calibrated 0.0135 in ln(lambda)
# across the wider range -- the same sampling density as the previous
# 300 points over 0.8-45 um, not a coarser grid stretched over it.
PAHC_N_WAVE = 403

# --- aperture-growth curve (spec section 2.3) ---
# ---------------------------------------------------------------------
# APERTURE TREATMENT -- READ THIS BEFORE CHANGING models.conf
#
# This library is aperture-INDEPENDENT, and that is load-bearing rather
# than a simplification. sedfitter offers exactly two packages
# (models.py Models.fit):
#
#   ndim == 2  aperture-independent -> linear_regression solves BOTH
#              A_V and a free scale. Brightness is free.
#   ndim == 3  aperture-dependent   -> optimal_scaling solves A_V ONLY.
#              There is no scale term; the only brightness freedom is
#              which distance slice wins, i.e. (d_max/d_min)^2.
#
# PAH-C needs the free scale. Its brightness is however much PAH surface
# brightness lands in the beam -- a property of position in the cloud,
# not of distance -- and the emission sits at the region's known
# distance anyway, so there is no distance to infer. SPS, GAL and
# H2-shock are all aperture-independent; PAH-C matches them.
#
# THE APERTURE PHYSICS IS KEPT, because the region distance factorises
# out of the growth law exactly:
#
#   g(theta_b * d) = (theta_b * d / ap_ref)^alpha
#                  = (d/d_ref)^alpha * (theta_b/theta_IRAC)^alpha
#
# The first factor is a constant multiplying the whole excess term,
# which is what R already parameterises. Only the BAND RATIO survives,
# and it needs no distance. One library serves every region: no
# per-region builds, no distance grid.
#
# R therefore means: the 8um excess-to-star flux ratio within the IRAC
# (2.4") beam. It is an angular observable, not a flux within some
# physical radius at an assumed distance.
# ---------------------------------------------------------------------

# --- the host-free grid axes (spec section 7; FIXED, see the block header) ---
# alpha x PAH_SIZE x R = 11 x 3 x 32 = 1,056 non-host combinations, now
# crossed against the HOST grid below (load_pahc_host_grid) rather than
# against a fixed anchor list.
#
# Every choice below is the output of an 8-band leverage and degeneracy
# measurement on COMPOSITE colours, not a configuration preference.
# 8 bands, not the two Gutermuth's plane uses: lg U reads as 0.018 mag in
# the IRAC pair and 2.641 mag in 8 bands, essentially all of it MIPS24,
# so a 2-colour measurement calls live axes dead.
PAHC_Z = 0.0          # fixed (solar metallicity), not a gridded axis -- no host, warm or cool, varies it

# ---------------------------------------------------------------------
# SUPERSEDED, 2026-10-08 (owner library-spec rebuild, PART 1: host
# coverage defect). PAHC_T_EFF_VALUES_K / PAHC_LOG_G_BY_T_EFF /
# PAHC_LOG_G_WARM / PAHC_LOG_G_COOL and build_pahc_parameter_grid (the
# functions that read them) are REMOVED, not merely unused: a hand-picked
# 11-anchor host list, closed at T_eff 6000 K with log g in {1.0, 3.0}
# only, is the defect being fixed -- real hosts hotter than 6000 K (and
# at other gravities) had nowhere to go and pinned to the grid edge
# (measured: 46% of confident Orion A PAHC sources on the two hottest
# anchors). Hosts are now load_pahc_host_grid's SPS-raw-splice window
# (PAHC_HOST_TEFF_MIN_K/MAX_K, PAHC_HOST_LOGG_MIN/MAX, below).
#
# The two measurements that justified the OLD anchors' log g remain true
# and are the PROVENANCE for the new window's log g span, so they are
# kept here as the historical record rather than deleted outright:
#
# WARM (D-11): log g FIXED at 3.0 was justified by measurement, not a
# population claim -- the full dwarf(4.5)->giant(2.0) span moves 8-band
# composite colour by 0.043 mag, of which only 0.009 is unreachable by
# the other axes, under the 0.05 mag colour floor either way. The
# literature would not support a single luminosity class in any case:
# the faint background along Galactic-plane cloud sightlines is
# genuinely mixed (Besancon/Robin+2003 gives K/M/G dwarfs at 0.8-3.6 kpc
# and giants/subgiants OF ALL TYPES at 5.4-5.7 kpc; c2d measured 25-90%
# background AGB/giant contamination in the Cha II Class III sample).
# "Of all types" is the operative phrase the OLD anchor list's 6000 K
# ceiling violated by implicitly admitting only G-type-or-cooler giants
# -- this Besancon citation is one of the two sources the new host
# window below is drawn from.
#
# COOL (D-15): log g FIXED at +1.0 -- NOT immaterial the way the warm
# case is (+0.5-vs-+1.0 gravity sensitivity at cool T_eff is 0.069-0.095
# mag, ABOVE the 0.05 mag colour floor; studies/pahc/cool_giant_anchors/
# 02_gravity_micro_study.py). This is the OLD anchors' log g FLOOR
# (1.0), carried forward unchanged into PAHC_HOST_LOGG_MIN below.
# ---------------------------------------------------------------------

# --- host grid (PART 1 of the owner library-spec rebuild, 2026-10-08) ---
# Hosts are drawn from the SPS RAW photosphere splice (the full
# CK03 + CIFIST + AGSS2009 set that load_pahc_host_grid reads and
# verifies -- NOT the sampled SPS class library a concurrent rebuild
# produces), every (T_EFF, LOGG) row inside this window, at the fixed
# PAHC_Z -- no hand-picked anchor list, no interpolation beyond what
# the SPS grid actually carries.
#
# WINDOW SOURCES (both already the OLD anchors' own provenance, above):
#  * Besancon (Robin+2003): K/M/G dwarfs at 0.8-3.6 kpc, giants/
#    subgiants OF ALL TYPES at 5.4-5.7 kpc.
#  * The SPS library's own raw coverage (T_eff 1200-50000 K, log g
#    -0.5 to 5.5 at PAHC_Z=0 -- measured directly off the splice; see
#    load_pahc_host_grid). The window below is a SUBSET of this: it is
#    never extrapolated past what SPS actually ships.
#
# ADOPTED WINDOW -- a judgement call, stated so it can be revisited,
# not a re-derived physical boundary:
#   T_EFF 2800-10000 K. Floor UNCHANGED (D-15's cool-giant anchors,
#   2800/3000/3300 K, sit inside it by construction). Ceiling at the
#   conventional top of ordinary dwarf/giant/subgiant spectral types
#   (~A0) -- short of the rare, luminous, early-type (O/B) population
#   that Gutermuth's "DIM field star" definition (Appendix A: "a star
#   must be present", the aperture catching only RESOLVED PAH structure,
#   not light from the star itself) argues against admitting in the
#   first place.
#   LOG G 1.0-5.0. Floor matches the OLD cool-giant anchors' log g
#   +1.0 (D-15, above). Ceiling excludes only the SPS grid's extreme
#   log g 5.5 edge (beyond ordinary dwarf gravities).
# Gaps inside this window are SPS's own (e.g. a missing log g rung at
# one T_eff) and are reported as gaps, never filled.
PAHC_HOST_TEFF_MIN_K = 2800.0
PAHC_HOST_TEFF_MAX_K = 10000.0
PAHC_HOST_LOGG_MIN = 1.0
PAHC_HOST_LOGG_MAX = 5.0

# PAH size distribution, DL21's three published size distributions. Kept
# as a gridded axis on 2.4-3.7 half-floor steps of UNIQUE 8-band reach
# (the component no combination of the other axes reproduces).
PAHC_PAH_SIZE_VALUES = ("sma", "std", "lrg")

# Even in colour (Delta[4.5]-[5.8], the axis R drives) rather than in
# log R -- the colour response to R saturates, so log-spacing overshoots
# at high R. Derived from a dense 4000-pt colour sweep at the grid
# reference point (T_eff 4500 K, log g 3.0, st/std, lg U 2.00), spanning
# x = 0.041 -> 2.016 in uniform 0.0637 mag steps.
#
# CEILING = 100, and it is a physics bound, not a round number: it is
# where the aperture excess overtakes the photosphere at 3.6 um (measured
# 103-113 across T_eff 3500-6000 K, near-invariant). Gutermuth (2009)
# Appendix A defines this population as "some dim field stars" whose
# photometric apertures caught resolved structured PAH emission -- a star
# must be present -- so the ceiling is set where the star stops
# dominating the band the colour cut anchors on -- 3.6 um, not 5.8,
# because 3.6 is where Gutermuth's cut is anchored.
PAHC_R_VALUES = (
    0.1000, 0.7518, 1.4504, 2.1999, 3.0045, 3.8689,
    4.7985, 5.7992, 6.8774, 8.0404, 9.2965, 10.6548,
    12.1257, 13.7208, 15.4537, 17.3393, 19.3953, 21.6418,
    24.1020, 26.8030, 29.7768, 33.0607, 36.6994, 40.7464,
    45.2661, 50.3373, 56.0568, 62.5457, 69.9566, 78.4854,
    88.3869, 100.0000,
)

# alpha is now a GRIDDED axis, reversing the shipped library's collapse.
# Bounds are exact rather than assumed: 0 = point-like (no growth),
# 2 = uniform extended emission (flux proportional to area). This is the
# axis carrying the one piece of physics that makes PAH-C
# aperture-dependent at all, and its value is genuinely uncertain --
# the effective growth depends on each cloud's structure on aperture
# scales AFTER background-annulus subtraction, which is why spec
# section 2.3 made the law cloud-agnostic in the first place.
#
# alpha is DEGENERATE with lg U (0.4 deg apart in 8-band colour) and with
# the PAH abundance f_pah (1.0 deg): all three are near-pure MIPS24
# levers with ~100 half-floor steps of leverage and ~zero unique reach.
# Exactly one of the three is gridded. alpha is chosen over lg U because
# its bounds are physical limits rather than an assumed range, and
# because it shapes the g(ap) curve sedfitter interpolates rather than
# being a purely spectral knob.
PAHC_ALPHA_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)

# --- fixed DL21 excess selectors (not gridded axes) ---
# lg U: fixed, as the non-gridded member of the degenerate MIPS24 trio.
# 2.00 = U 100, representative of PAH-bright cloud material illuminated
# by nearby early-type stars -- between the diffuse ISM (lg U 0) and a
# dense PDR edge (lg U 4).
PAHC_LG_U_TOKEN = "2.00"
# ionization: fixed at DL21's standard charge distribution. Measured
# 0.4 half-floor steps of unique 8-band reach, the weakest of the
# candidates, and it sits only 12.6-13.5 deg from the alpha/lg U/f_pah
# direction, so gridding alpha already covers most of it.
PAHC_IONIZATION = "st"
# f_pah: fixed at DL21's fiducial PAH abundance (40.85 ppm PAH carbon per
# H, measured from DL21's own dnda tables). Third member of the
# degenerate trio.
PAHC_F_PAH = 1.0
# radiation field: fixed at the diffuse ISRF. Measured 0.0 half-floor
# steps of unique 8-band reach against a 3 Myr starburst field.
PAHC_RADIATION_FIELD = "mMMP"

# --- accepted resolution limits (recorded, not chased) ---
# alpha: 0.25 mag steps at 24 um against a 0.045 mag median MIPS24
# magnitude error, i.e. ~5.5 sigma per step. Full resolution would need
# ~55 alpha points and a model cube several GB in size. Mitigating
# context, measured on the labelled sample: only 229 of 5,819 CLASS=39
# sources (3.9%) have a MIPS24 detection at all, so for 96% of the class
# this axis is unconstrained by data and the grid merely marginalises it.
# R: 0.0637 mag steps in [4.5]-[5.8], ~2.5x the 0.025 mag half-floor
# target. The R ceiling moving 10 -> 100 tripled the span the ladder must
# cover; meeting the half-floor rule exactly would need ~84 R points.

# --- validation tolerances (spec section 8) ---
# item 1 is a plain ratio of float32-stored values -> machine-precision tight.
PAHC_ANCHOR_TOLERANCE = 1e-6
# item 2 takes a DIFFERENCE of float32-stored values, which amplifies float32
# rounding far more than a ratio does -- 1e-4 is still tight against real
# physics, just realistic for float32 storage.
PAHC_APERTURE_TOLERANCE = 1e-4
# item 4 compares two independent convolutions of the same SED; 0.01 mag is
# a fifth of the colour-error floor and a few times the measured agreement.
PAHC_CONVOLUTION_TOLERANCE_MAG = 0.01


# ====================================================================
# Stage 1: DL21 PAH-excess templates
# ====================================================================
#
# Draine, Li, Hensley, Hunt, Sandstrom & Smith 2021, ApJ 917, 3.
# Archive: https://www.astro.princeton.edu/~draine/PAHspec/
#
# Replaces DL07, which stops at U_min = 25, fixes the illuminating
# spectrum to MMP83 by construction, and exposes neither PAH charge
# state nor PAH size distribution -- the two parameters that set the
# 3.3 vs 6.2/7.7/8.6 balance, i.e. IRAC1 against IRAC3/4.

DL21_BASE_URL = "https://www.astro.princeton.edu/~draine/PAHspec"
DL21_FETCH_TIMEOUT_S = 120

# One illuminating spectrum. mMMP is the modified Mathis-Mezger-Panagia
# field -- the diffuse ISRF. The archive ships 14; this one is fixed
# because it measured 0.0 half-floor steps of unique 8-band reach
# against a 3 Myr starburst field (spec section 7).
DL21_RADFIELD_DIR = "mMMP"
DL21_RADFIELD_TOKEN = "mmpisrf"

# The archive's own axes, as literal filename tokens. The build READS
# only PAHC_LG_U_TOKEN x PAHC_IONIZATION x the three PAH sizes -- three
# files -- but FETCHES the whole per-field slice, as the DL07 build did
# before it: the count is a meaningful completeness check, the axes stay
# re-derivable without a re-fetch, and it is 4.2 MB either way.
DL21_LGU_TOKENS = ("0.00", "0.50", "1.00", "1.50", "2.00", "2.50", "3.00", "3.50",
                   "4.00", "4.50", "5.00", "5.50", "6.00", "6.50", "7.00")
DL21_IONIZATION_TOKENS = ("lo", "st", "hi")
DL21_SIZE_TOKENS = ("sma", "std", "lrg")
DL21_SPECTRUM_FILE_COUNT = (len(DL21_LGU_TOKENS) * len(DL21_IONIZATION_TOKENS)
                            * len(DL21_SIZE_TOKENS))

# NOT fetched: the archive's per-grain-size `iout_*` files, 258 MB of the
# 262 MB subdirectory. They exist to let a user synthesize a custom
# size/ionization mix, and DL21's nine pre-summed (size x ionization)
# combinations already are that.

DL21_MANIFEST_NAME = "dl21_manifest.json"

# Locate the data block by matching the column-header STRING, never by a
# row count -- the same discipline the DL07 parser used. Column order on
# that line: wave, total, Astrodust, PAH^+, PAH^0.
_DL21_COLUMN_HEADER_MARKER = ("(um)", "Astrodust")


def _dl21_spectrum_filename(lgu_token, ionization, size):
    return f"pahspec.out_{DL21_RADFIELD_TOKEN}_{lgu_token}_{ionization}_{size}.gz"


def _dl21_all_filenames():
    """Every file the fetch pulls, in a fixed order.

    The size/ionization distributions carry no radiation-field token --
    they are a property of the dust model, not of what illuminates it --
    and the archive ships identical copies in every field subdirectory.
    """
    names = [_dl21_spectrum_filename(u, ion, size)
             for u in DL21_LGU_TOKENS
             for ion in DL21_IONIZATION_TOKENS
             for size in DL21_SIZE_TOKENS]
    names += [f"pahspec_dnda.out_{ion}_{size}"
              for ion in DL21_IONIZATION_TOKENS
              for size in DL21_SIZE_TOKENS]
    names.append(f"isrf_{DL21_RADFIELD_TOKEN}_0.00.gz")
    return names


def dl21_spectrum_path(download_dir, lgu_token, ionization, size):
    """Path to one summed DL21 spectrum under download_dir."""
    return os.path.join(download_dir, DL21_RADFIELD_DIR,
                        _dl21_spectrum_filename(lgu_token, ionization, size))


def count_dl21_spectrum_files(download_dir):
    """How many of the summed spectra are present on disk."""
    return sum(os.path.exists(dl21_spectrum_path(download_dir, u, ion, size))
               for u in DL21_LGU_TOKENS
               for ion in DL21_IONIZATION_TOKENS
               for size in DL21_SIZE_TOKENS)


def _sha256(path, chunk_bytes=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_pahc_excess_templates(download_dir, timeout=DL21_FETCH_TIMEOUT_S, verbose=True):
    """Stage 1: fetch + verify the DL21 summed emission spectra.

    MANIFEST-PINNED, not single-SHA-pinned. model_io.fetch_pinned takes a
    known sha256 up front, which suits a single published tarball; DL21
    is 145 files with no published checksums, so the pin is BOOTSTRAPPED
    -- the first run downloads and records every file's sha256, and every
    later run verifies against that record through fetch_pinned, which
    then also serves as the cache (it returns early when the file already
    hashes correctly). The discipline spec section 1.1 protects -- a
    mismatch is an event to investigate, never a hash to update -- holds
    from run 2 onward. Run 1 is honestly unpinned and says so.
    """
    import json
    import urllib.request

    subdir = os.path.join(download_dir, DL21_RADFIELD_DIR)
    os.makedirs(subdir, exist_ok=True)
    manifest_path = os.path.join(download_dir, DL21_MANIFEST_NAME)

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    if not manifest and verbose:
        print(f"    NO MANIFEST -- bootstrapping the pin. This run is UNPINNED; "
              f"it writes {DL21_MANIFEST_NAME}, and every later run verifies "
              f"against it.")

    n_fetched = n_cached = 0
    for name in _dl21_all_filenames():
        dest = os.path.join(subdir, name)
        url = f"{DL21_BASE_URL}/{DL21_RADFIELD_DIR}/{name}"
        expected = manifest.get(name)

        if expected is not None:
            before = os.path.exists(dest) and _sha256(dest) == expected
            model_io.fetch_pinned(url, expected, dest, timeout=timeout)
            n_cached += before
            n_fetched += not before
            continue

        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
        partial = dest + ".part"
        with open(partial, "wb") as out:
            out.write(payload)
        os.replace(partial, dest)
        manifest[name] = _sha256(dest)
        n_fetched += 1

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    n_present = count_dl21_spectrum_files(download_dir)
    if n_present != DL21_SPECTRUM_FILE_COUNT:
        raise AssertionError(
            f"DL21 fetch incomplete: {n_present} of {DL21_SPECTRUM_FILE_COUNT} "
            f"summed spectra present under {subdir}")
    if verbose:
        print(f"    {n_fetched} fetched, {n_cached} cached, "
              f"{n_present}/{DL21_SPECTRUM_FILE_COUNT} spectra verified")
    return download_dir


def load_pahc_excess_shape(download_dir, pah_size):
    """P_hat(lambda) for one PAH size distribution, normalised so
    P_hat(8um) = 1, on DL21's native ASCENDING grid (DL07's descended).

    Reads the `total` column at the fixed (lg U, ionization, radiation
    field) of spec section 7.

    TWO THINGS THIS FUNCTION MUST DO, both of which fail quietly if
    skipped:

    1. DIVIDE BY nu. DL21 ships nu*P_nu; the composite is built in F_nu
       and uses only P_hat(lambda)/P_hat(8um), and a factor of frequency
       does NOT cancel in that ratio -- it would tilt the excess across
       exactly the bands carrying the signal.
    2. NORMALISE BEFORE ANY RESAMPLING. DL21's native P_nu spans ~2e-40
       to ~4e-34, entirely below FLUX_FLOOR. Fed unnormalised through
       resample_sed_log_log every point clips to the floor and the excess
       returns FLAT -- no exception, no NaN, every template identical,
       which reads as a null result rather than a bug.
    """
    path = dl21_spectrum_path(download_dir, PAHC_LG_U_TOKEN, PAHC_IONIZATION, pah_size)
    with gzip.open(path, "rt") as fh:
        lines = fh.readlines()

    header_idx = next(i for i, line in enumerate(lines)
                      if all(tok in line for tok in _DL21_COLUMN_HEADER_MARKER))

    wave_um, nu_p_nu = [], []
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            wave, total = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        wave_um.append(wave)
        nu_p_nu.append(total)

    wave_um = np.array(wave_um)
    if wave_um.size == 0:
        raise ValueError(f"{path}: no spectrum rows parsed after the column header")

    p_nu = np.array(nu_p_nu) * wave_um / C_UM_S          # nu*P_nu -> P_nu
    p_8 = np.interp(ANCHOR_WAVELENGTH_UM, wave_um, p_nu)
    if not np.isfinite(p_8) or p_8 <= 0.0:
        raise ValueError(f"{path}: non-positive 8um anchor {p_8!r}")
    return wave_um, p_nu / p_8


# ====================================================================
# Stage 2: stellar photosphere (existing SPS library)
# ====================================================================

def load_sps_stellar_sed(sps_dir, t_eff, log_g, z):
    """(wave_um, flux_mjy) for the SPS model matching (t_eff, log_g, z)
    exactly (within float tolerance), read straight from SPS flux.fits --
    no per-model rescaling (spec section 1.2). wave_um is SPS's native
    descending order."""
    with fits.open(os.path.join(sps_dir, "parameters.fits")) as h:
        sps_t_eff = h[1].data["T_EFF"]
        sps_log_g = h[1].data["LOGG"]
        sps_z = h[1].data["Z_H"]

    matches = np.where(
        np.isclose(sps_t_eff, t_eff) & np.isclose(sps_log_g, log_g) & np.isclose(sps_z, z)
    )[0]
    if len(matches) != 1:
        raise ValueError(f"expected exactly 1 SPS model at (T_eff={t_eff}, log_g={log_g}, "
                         f"Z_H={z}), found {len(matches)}")
    idx = matches[0]

    with fits.open(os.path.join(sps_dir, "flux.fits")) as h:
        wave_um = h["SPECTRAL_INFO"].data["WAVELENGTH"]
        flux_mjy = h["VALUES"].data[idx, 0, :]
    return wave_um, flux_mjy


# ====================================================================
# Stage 3: PAH-C wavelength grid + resampling
# ====================================================================

def build_pahc_wavelength_grid():
    """Full-resolution model wavelength grid, log-spaced ASCENDING (only
    flipped to descending when writing flux.fits -- see module
    docstring). 0.2-45um brackets the 8 SESNA filters AND Gaia G with
    margin; 403 points amply resolve a smooth continuum + broad PAH
    features (spec section 3.1) -- no fine spike-sampling needed, unlike
    H2-shock's line spectra. Takes no arguments: the grid is fixed by
    calibration (measured: rebuilding at four times the sampling density
    moves the IRAC composite colours by at most 0.006 mag), not a knob."""
    return np.logspace(np.log10(PAHC_WAVE_MIN_UM), np.log10(PAHC_WAVE_MAX_UM),
                       PAHC_N_WAVE)


def resample_sed_log_log(wave_native_um, flux_native, wave_target_um, flux_floor=FLUX_FLOOR):
    """Resample an SED onto wave_target_um by log-log interpolation.

    Out-of-coverage convention, stated rather than implied: on the
    WAVELENGTH axis the result is EXACTLY ZERO outside the native
    source's own coverage -- never extrapolated, never clamped (spec
    section 2.4). That is what zero-pads DL21's 1.0056um blue edge down
    to the grid's 0.2um start -- correct here, since dust emission really
    is nil in the optical, but note it is also what makes a filter
    extending past the grid fail SILENTLY rather than loudly.

    wave_native_um may be ascending or descending (sorted internally);
    non-positive flux values are floored before the log transform."""
    order = np.argsort(wave_native_um)
    wave_sorted = wave_native_um[order]
    flux_safe = np.clip(flux_native[order], flux_floor, None)

    log_target = np.log10(wave_target_um)
    log_native_wave = np.log10(wave_sorted)
    log_native_flux = np.log10(flux_safe)

    interp_log_flux = np.interp(log_target, log_native_wave, log_native_flux, left=np.nan, right=np.nan)

    result = 10.0 ** interp_log_flux
    result[np.isnan(interp_log_flux)] = 0.0
    return result


# ====================================================================
# Stage 4: composite physics
# ====================================================================

# Wavelengths (um) at which the survey aperture changes group. Both sit
# in genuine gaps in the filter set -- 2MASS response ends at 2.375 and
# IRAC begins at 3.081; IRAC ends at 9.953 and MIPS24 begins at 19.494
# (measured at 1e-3 of peak response) -- so no filter integrates across a
# discontinuity and the step is exact rather than approximate. Re-measure
# these if the filter set ever changes.
BAND_APERTURE_EDGES_UM = (2.728, 14.723)


def band_aperture_scaling(wave_um, alpha):
    """(theta(lambda)/theta_IRAC)^alpha -- the aperture factor on the
    excess term, as a function of wavelength.

    The survey aperture is a property of the BAND, not of the model, and
    sedfitter does not apply it for aperture-independent libraries: in
    Model.read the `aperture_arcsec` passed per filter is consumed only
    when n_distances is not None, and the ndim == 2 branch just takes
    conv.flux[:, 0]. So the band ratios have to be baked in HERE -- this
    is required, not optional, and there is no double-counting risk.

    Only the RATIO to the IRAC aperture appears. The absolute aperture,
    and hence the region distance, cancels into R (see the APERTURE
    TREATMENT note in the CONSTANTS block).

    Apertures come from constants.BANDS -- 2MASS 4", IRAC 2.4",
    MIPS24 7.6" -- not from a local table."""
    wave_um = np.asarray(wave_um, dtype=float)
    theta_ratio = np.full(wave_um.shape, 1.0)          # IRAC, by definition
    theta_ratio[wave_um < BAND_APERTURE_EDGES_UM[0]] = (
        BANDS["J"].aperture_arcsec / BANDS["I1"].aperture_arcsec)
    theta_ratio[wave_um > BAND_APERTURE_EDGES_UM[1]] = (
        BANDS["M1"].aperture_arcsec / BANDS["I1"].aperture_arcsec)
    return theta_ratio ** alpha


def compute_composite_sed(s_resampled_mjy, p_hat_resampled, wave_um_ascending,
                          r_ratio, alpha):
    """One composite SED, (n_wave,) in mJy:

        M(lambda) = s(lambda)
                    + R * s(8um) * [P_hat(lambda)/P_hat(8um)]
                        * (theta(lambda)/theta_IRAC)^alpha

    The star term is a point source and carries no aperture factor; only
    the excess does, and only through the BAND RATIO -- the region
    distance has already cancelled into R (see the APERTURE TREATMENT
    note in the CONSTANTS block).

    R is anchored in the IRAC beam, where the ratio is 1 by definition,
    so M(8um)/s(8um) = 1 + R exactly. That identity is now unconditional:
    the aperture-dependent version could only state it at a continuous
    ap_ref that fell between stored apertures."""
    s_8 = np.interp(ANCHOR_WAVELENGTH_UM, wave_um_ascending, s_resampled_mjy)
    p_8 = np.interp(ANCHOR_WAVELENGTH_UM, wave_um_ascending, p_hat_resampled)

    p_ratio = p_hat_resampled / p_8   # dimensionless, P_hat(lambda)/P_hat(8um)
    aperture_factor = band_aperture_scaling(wave_um_ascending, alpha)
    return s_resampled_mjy + r_ratio * s_8 * p_ratio * aperture_factor


# ====================================================================
# Stage 5: the host grid (owner library-spec rebuild, PART 1) + the raw
# parameter grid crossing it against the host-free axes above.
# ====================================================================

# The 8 SESNA bands, in sed_models_register.density's own order
# (constants.BANDS' key order -- density.py projects an 8-vector
# indexed this way, so the convolved fluxes fed to it MUST agree).
PAHC_BAND_ORDER = ("J", "H", "Ks", "I1", "I2", "I3", "I4", "M1")

# Minimum distinct SPS sources required to accept sps_dir as the RAW
# splice rather than a down-sampled class library -- the splice is
# CK03 + CIFIST (+ AGSS2009 for the cool-giant extension); a sampled
# library built FOR SHIPPING would plausibly collapse to one source or
# a token few rows. This is a floor, not an exact count: it exists so
# load_pahc_host_grid fails loudly instead of silently pre-coarsening
# PAHC's own host coverage (spec instruction: confirm the path holds
# the full set before building, report rather than guess).
PAHC_HOST_GRID_MIN_SOURCES = 3
PAHC_HOST_GRID_MIN_ROWS = 1000


def load_pahc_host_grid(sps_dir, teff_min=PAHC_HOST_TEFF_MIN_K, teff_max=PAHC_HOST_TEFF_MAX_K,
                        logg_min=PAHC_HOST_LOGG_MIN, logg_max=PAHC_HOST_LOGG_MAX,
                        z=PAHC_Z):
    """Every (T_EFF, LOGG) row the SPS RAW photosphere splice carries
    inside [teff_min, teff_max] x [logg_min, logg_max] at the fixed
    PAHC_Z -- PART 1 of the owner library-spec rebuild. Returns a sorted
    list of (t_eff, log_g) tuples, ascending.

    CONFIRMS sps_dir HOLDS THE FULL RAW SPLICE, not the sampled SPS
    class library a concurrent rebuild produces: asserts at least
    PAHC_HOST_GRID_MIN_SOURCES distinct SOURCE values and at least
    PAHC_HOST_GRID_MIN_ROWS rows total. A sampled shipping library
    (~25 templates, one row per kept template) would fail this loudly
    rather than silently pre-coarsening PAHC's host coverage.

    NO POPULATION WEIGHT: every row inside the window is included once,
    regardless of how densely SPS's own grid samples that corner --
    weighting by density would be a prior quantity, forbidden by the
    library spec's independence item."""
    with fits.open(os.path.join(sps_dir, "parameters.fits")) as h:
        sps_t_eff = h[1].data["T_EFF"].astype(float)
        sps_log_g = h[1].data["LOGG"].astype(float)
        sps_z = h[1].data["Z_H"].astype(float)
        sps_source = np.char.strip(h[1].data["SOURCE"].astype(str))

    n_rows = sps_t_eff.size
    n_sources = len(set(sps_source.tolist()))
    if n_sources < PAHC_HOST_GRID_MIN_SOURCES or n_rows < PAHC_HOST_GRID_MIN_ROWS:
        raise AssertionError(
            f"{sps_dir}/parameters.fits looks like the SAMPLED SPS class "
            f"library, not the raw CK03+CIFIST+AGSS splice: {n_rows} rows, "
            f"{n_sources} distinct SOURCE value(s) {sorted(set(sps_source.tolist()))} "
            f"(need >= {PAHC_HOST_GRID_MIN_ROWS} rows, >= {PAHC_HOST_GRID_MIN_SOURCES} "
            f"sources). PAHC's host grid must be drawn from the full raw splice -- "
            f"building from a down-sampled library would pre-coarsen PAHC's own "
            f"host coverage before this module's own sampling. Point sps_dir at "
            f"the raw splice, or report this rather than proceeding.")

    m = ((sps_z == z) & (sps_t_eff >= teff_min) & (sps_t_eff <= teff_max)
         & (sps_log_g >= logg_min) & (sps_log_g <= logg_max))
    if not m.any():
        raise AssertionError(
            f"{sps_dir}/parameters.fits: no rows at Z_H={z} inside "
            f"T_eff=[{teff_min},{teff_max}] x log g=[{logg_min},{logg_max}]")
    hosts = sorted(set(zip(sps_t_eff[m].tolist(), sps_log_g[m].tolist())))
    # host -> source, for callers that need a single-grid subset (e.g.
    # build_pahc_validation_grid, which must not chain a monotonicity
    # check across the CK03/CIFIST splice handoff -- see its docstring).
    # Takes the FIRST matching row's source; splice_ck03_and_cifist
    # already removes exact (T_eff, log g) duplicates between CK03 and
    # CIFIST upstream, so collisions here are not expected.
    host_source = {}
    for tt, gg, ss in zip(sps_t_eff[m], sps_log_g[m], sps_source[m]):
        host_source.setdefault((float(tt), float(gg)), ss)
    return hosts, dict(n_raw_rows=n_rows, n_raw_sources=n_sources,
                       sources=sorted(set(sps_source.tolist())),
                       host_source=host_source)


def build_pahc_raw_grid(hosts, alphas=PAHC_ALPHA_VALUES, pah_sizes=PAHC_PAH_SIZE_VALUES,
                        r_values=PAHC_R_VALUES):
    """The FULL raw grid: every host (T_eff, log g) x alpha x PAH size x
    R, as parallel int index arrays in a FIXED, deterministic nested
    order -- host outer, then alpha, then PAH size, then R (the same
    order np.meshgrid(..., indexing='ij').ravel() produces, and the
    order greedy_r_net's tie-break and format_pahc_model_name's index
    both rely on). This raw grid is for SAMPLING ONLY (Stage 5.5); it is
    never written whole to disk -- see the module docstring.

    Returns (host_idx, alpha_idx, pah_idx, r_idx), each (n_raw,) int,
    indexing into `hosts`, `alphas`, `pah_sizes`, `r_values`
    respectively."""
    n_host, n_alpha, n_pah, n_r = len(hosts), len(alphas), len(pah_sizes), len(r_values)
    host_idx, alpha_idx, pah_idx, r_idx = np.meshgrid(
        np.arange(n_host), np.arange(n_alpha), np.arange(n_pah), np.arange(n_r),
        indexing="ij")
    return (host_idx.ravel(), alpha_idx.ravel(), pah_idx.ravel(), r_idx.ravel())


def build_pahc_raw_band_flux(hosts, host_idx, alpha_idx, pah_idx, r_idx,
                             alphas, pah_sizes, r_values,
                             sps_dir, dl21_dir, wave_um, bands=PAHC_BAND_ORDER):
    """The 8-band convolved flux (n_raw, len(bands)) of EVERY raw grid
    model, computed WITHOUT building or convolving 338,976 individual
    403-point SEDs.

    EXACT LINEAR SEPARABILITY, not an approximation. compute_composite_sed
    is affine in R at fixed (host, alpha, pah_size):

        M(lambda) = s(lambda) + R * [s(8um) * P_hat(lambda)/P_hat(8um)
                                      * aperture_factor(lambda, alpha)]

    and `_independent_band_flux` (band convolution) is a LINEAR operator
    in the flux array it is handed (interp at fixed x-grid, then trapz,
    both linear in the y-values) -- both source arrays (star, excess
    shape) live on the SAME wave_um grid they were resampled onto, so
    there is no edge/coverage subtlety for the outer convolution to get
    wrong. So, writing F_band for one band's convolved flux,

        F_band(host, alpha, pah, R) = F_star_band(host)
                                       + R * s_8(host) * G(pah, alpha, band)

    where G(pah, alpha, band) is band b's convolved flux of the UNIT
    excess shape P_hat(lambda)/P_hat(8um) * aperture_factor(lambda,
    alpha) (no star, R=1). This lets the whole 338,976-model raw grid
    be built from only len(hosts) + len(pah_sizes)*len(alphas) convolved
    SEDs (354 for the 2026-10-08 build) via broadcasting. Cross-checked
    against compute_composite_sed + `_independent_band_flux` run the slow
    way on random raw-grid rows: machine-precision agreement (<1e-15
    relative), see studies/pahc/ for the check script.
    """
    n_host = len(hosts)
    star_band = np.empty((n_host, len(bands)))
    s8_arr = np.empty(n_host)
    for i, (t, g) in enumerate(hosts):
        wave_native, flux_native = load_sps_stellar_sed(sps_dir, t, g, PAHC_Z)
        s_resampled = resample_sed_log_log(wave_native, flux_native, wave_um)
        s8_arr[i] = np.interp(ANCHOR_WAVELENGTH_UM, wave_um, s_resampled)
        for b_i, b in enumerate(bands):
            star_band[i, b_i] = _independent_band_flux(wave_um, s_resampled, b)

    n_pah, n_alpha = len(pah_sizes), len(alphas)
    excess_unit_band = np.empty((n_pah, n_alpha, len(bands)))
    for p_i, pah in enumerate(pah_sizes):
        wave_native, p_hat_native = load_pahc_excess_shape(dl21_dir, pah)
        p_hat = resample_sed_log_log(wave_native, p_hat_native, wave_um)
        p_hat = p_hat / np.interp(ANCHOR_WAVELENGTH_UM, wave_um, p_hat)
        for a_i, a in enumerate(alphas):
            unit_excess = p_hat * band_aperture_scaling(wave_um, a)
            for b_i, b in enumerate(bands):
                excess_unit_band[p_i, a_i, b_i] = _independent_band_flux(wave_um, unit_excess, b)

    r_val = np.asarray(r_values, dtype=float)[r_idx]
    flux = (star_band[host_idx]
            + (r_val * s8_arr[host_idx])[:, None] * excess_unit_band[pah_idx, alpha_idx])
    if not (np.all(np.isfinite(flux)) and np.all(flux > 0)):
        raise ValueError("raw-grid convolved band flux is non-finite or non-positive")
    return flux


#: Chunk size for the one-time, memory-bounded initial-count pass over
#: every raw-grid point (query_ball_point(..., return_length=True)) --
#: see greedy_r_net. Bounds the SciPy-internal temporary allocation to
#: one chunk's worth of query output at a time, never the whole raw grid.
R_NET_COUNT_CHUNK = 20000


def greedy_r_net(coords, radius, chunk_size=R_NET_COUNT_CHUNK):
    """Deterministic greedy maximum-uncovered-neighbour r-net -- the
    standard greedy set-cover rule, over `coords` (n, d) at `radius`
    (one SIGEFF, the fit's one noise length) -- MEMORY-BOUNDED, LAZY-HEAP
    form.

    MEMORY DISCIPLINE (why this replaces an adjacency-matrix
    implementation). The raw grid is 338,976 points with a mean degree
    of ~2,350 within one SIGEFF: a materialised adjacency -- whether a
    Python list-of-lists from query_ball_point/query_ball_tree, or a
    scipy.sparse CSR built from cKDTree.query_pairs's (M, 2) pair array
    -- is tens of GB either way (M ~ 4e8 pairs). This implementation
    NEVER materialises a neighbour list for more than one point at a
    time, and holds only O(n) per-model state (one int64 count array,
    one bool covered array, one heap of (count, index) pairs) -- a few
    tens of MB total, independent of the mean degree.

    ALGORITHM. Standard lazy-deletion priority-queue greedy for maximum
    coverage:
      1. One cKDTree over `coords`.
      2. INITIAL counts only (no neighbour lists): chunked
         `tree.query_ball_point(coords[chunk], r=radius,
         return_length=True)`, an int64 ndarray of COUNTS, at most
         `chunk_size` points per call.
      3. A max-heap of (-count, index) -- heapq is a min-heap, so the
         negated count surfaces the largest first; ties break on the
         smaller `index`, i.e. the raw grid's own fixed flat index
         (host > alpha > PAH size > R nested order), exactly the
         tie-break an `np.argmax`-based scan would give. No random seed
         anywhere.
      4. Pop the top. If already covered, discard (lazy deletion). Else
         recompute its LIVE uncovered count with ONE single-point query
         `tree.query_ball_point(coords[idx], r=radius)` (returns a plain
         list of neighbour indices for that ONE point only; converted to
         an array, counted against `covered`, and discarded immediately
         -- never retained). If `live < stored`, the heap entry was
         stale (something covered one of its neighbours since it was
         pushed): push `(-live, idx)` back and continue. If `live ==
         stored`, SELECT idx -- mark it and every one of those live
         neighbours covered -- and move on. `live` can never exceed
         `stored`, because covered counts only ever shrink over time, so
         this is the full lazy-greedy correctness argument: a point is
         selected only when its freshly-measured count matches the
         largest count anywhere in the heap, i.e. it is certainly the
         current global maximum.
      5. Terminate when the heap is empty -- every point has been
         popped at least once, and every popped point is either chosen
         or covered by construction (6 below), so nothing uncovered can
         remain.

    COVERING AND PACKING HOLD BY CONSTRUCTION, not merely by later
    measurement:
      * COVERING -- a point leaves the heap only by being discarded
        (already covered) or selected (and then marked covered itself);
        the loop does not end until the heap is empty, so every point
        is covered by the time it terminates.
      * PACKING -- a point is only ever selected while its own
        `covered` flag is still False. Any point within `radius` of an
        already-selected representative is in that representative's
        live-neighbour list and was marked covered at that earlier
        step, so it can never be selected afterwards -- no two selected
        points are ever closer than `radius`.

    Returns the chosen raw-grid indices, in SELECTION order (not
    sorted; the caller may re-sort for a readable shipped order)."""
    import heapq

    n = coords.shape[0]
    tree = cKDTree(coords)

    # Step 2: initial COUNTS ONLY, chunked, one int64 per model -- no
    # neighbour lists held at any point, and no more than `chunk_size`
    # points' worth of query_ball_point output materialised at once.
    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        counts[start:stop] = tree.query_ball_point(
            coords[start:stop], r=radius, return_length=True)

    # Step 3: max-heap via negated counts; index is the deterministic
    # tie-break (smallest flat index wins among equal counts).
    heap = [(-int(counts[i]), i) for i in range(n)]
    heapq.heapify(heap)
    del counts

    covered = np.zeros(n, dtype=bool)
    chosen = []
    while heap:
        neg_stored, idx = heapq.heappop(heap)
        if covered[idx]:
            continue
        # Step 4: LIVE count via ONE single-point query; the returned
        # neighbour list is for this one point only and is discarded as
        # soon as this iteration marks it covered.
        nbrs = np.asarray(tree.query_ball_point(coords[idx], r=radius),
                          dtype=np.int64)
        uncovered_nbrs = nbrs[~covered[nbrs]]
        live = int(uncovered_nbrs.size)
        stored = -neg_stored
        if live < stored:
            heapq.heappush(heap, (-live, idx))
            continue
        # live == stored (live > stored is impossible -- see docstring):
        # idx is the current global maximum. Select it.
        chosen.append(idx)
        covered[uncovered_nbrs] = True
        covered[idx] = True
    return np.array(chosen, dtype=np.int64)


def pahc_member_statistics(hosts, alphas, pah_sizes, r_values,
                           host_idx, alpha_idx, pah_idx, r_idx,
                           coords, chosen, sigma_eff):
    """Voronoi-assigns EVERY raw-grid model to its NEAREST chosen
    template (never to whichever ball covered it during the greedy
    traversal -- the represented sets are derived from the final
    representatives, not the other way round), then summarises each
    kept template's represented set: its count, its subclass fractions
    (trivially {'NONE': 1.0} -- PAHC_SUBCLASS, this library defines
    none), and the min/median/max of T_EFF, LOGG, ALPHA, PAH_SIZE
    (ordinal sma=0 < std=1 < lrg=2, DL21's own grain-size ordering) and
    R over that set.

    Also returns the coverage/packing diagnostics for the report:
    coverage identity (fraction of raw models within one SIGEFF of
    their nearest kept template -- must be 1.0), the covering radius
    and the member-to-representative distance distribution in SIGEFF
    units, and the minimum inter-representative distance in SIGEFF
    units (packing; must be >= 1.0)."""
    rep_coords = coords[chosen]
    rep_tree = cKDTree(rep_coords)
    dist, nearest_rep_local = rep_tree.query(coords, k=1, workers=-1)

    diagnostics = {
        "coverage_identity": float(np.mean(dist <= sigma_eff * (1.0 + 1e-9))),
        "covering_radius_over_sigma_eff": float(dist.max() / sigma_eff),
        "member_dist_sigma_eff_median": float(np.median(dist) / sigma_eff),
        "member_dist_sigma_eff_max": float(dist.max() / sigma_eff),
    }
    if len(chosen) > 1:
        d2, _ = rep_tree.query(rep_coords, k=2, workers=-1)
        diagnostics["min_inter_rep_distance_over_sigma_eff"] = float(d2[:, 1].min() / sigma_eff)
    else:
        diagnostics["min_inter_rep_distance_over_sigma_eff"] = float("inf")

    pah_ordinal = {p: i for i, p in enumerate(pah_sizes)}
    t_eff_of_host = np.array([hosts[h][0] for h in host_idx])
    logg_of_host = np.array([hosts[h][1] for h in host_idx])
    alpha_of_model = np.asarray(alphas, dtype=float)[alpha_idx]
    pah_ord_of_model = np.array([pah_ordinal[pah_sizes[p]] for p in pah_idx])
    r_of_model = np.asarray(r_values, dtype=float)[r_idx]

    members = []
    for k in range(len(chosen)):
        sel = nearest_rep_local == k
        n_sel = int(sel.sum())
        if n_sel == 0:
            raise AssertionError(
                f"kept template {k} (raw index {chosen[k]}) represents zero raw "
                f"models -- it should always represent at least itself")
        members.append({
            "n_members": n_sel,
            "subclass_frac_none": 1.0,
            "t_eff_min": float(t_eff_of_host[sel].min()),
            "t_eff_median": float(np.median(t_eff_of_host[sel])),
            "t_eff_max": float(t_eff_of_host[sel].max()),
            "logg_min": float(logg_of_host[sel].min()),
            "logg_median": float(np.median(logg_of_host[sel])),
            "logg_max": float(logg_of_host[sel].max()),
            "alpha_min": float(alpha_of_model[sel].min()),
            "alpha_median": float(np.median(alpha_of_model[sel])),
            "alpha_max": float(alpha_of_model[sel].max()),
            "pah_size_min": int(pah_ord_of_model[sel].min()),
            "pah_size_median": float(np.median(pah_ord_of_model[sel])),
            "pah_size_max": int(pah_ord_of_model[sel].max()),
            "r_min": float(r_of_model[sel].min()),
            "r_median": float(np.median(r_of_model[sel])),
            "r_max": float(r_of_model[sel].max()),
        })
    return members, diagnostics


PAHC_PAH_SIZE_ORDINAL_LEGEND = tuple(enumerate(PAHC_PAH_SIZE_VALUES))  # (0,'sma'), (1,'std'), (2,'lrg')


def write_pahc_members_fits(members, model_names, out_path, n_raw, diagnostics):
    """members.fits: one row per KEPT template, the represented-set
    metadata item 3 of the library spec asks for (count, subclass
    fractions, T_EFF/LOGG/ALPHA/PAH_SIZE/R ranges). Point values stay in
    parameters.fits (schema parity with every other library); this file
    is additional and carries no release-contract requirement -- not
    checked by model_io.validate_model_directory, no PRODUCT.json, no
    hash, no verification gate.

    PAH_SIZE_MIN/MEDIAN/MAX are DL21's own grain-size ordinal (sma=0 <
    std=1 < lrg=2, PAHC_PAH_SIZE_ORDINAL_LEGEND) -- PAH_SIZE is
    categorical, so there is no other way to state a "range" for it.
    """
    if len(members) != len(model_names):
        raise ValueError(f"{len(members)} member records != {len(model_names)} model names")
    cols = [fits.Column(name="MODEL_NAME", format=MODEL_NAME_FORMAT, array=np.asarray(model_names))]
    int_fields = ("n_members",)
    float_fields = ("subclass_frac_none",
                    "t_eff_min", "t_eff_median", "t_eff_max",
                    "logg_min", "logg_median", "logg_max",
                    "alpha_min", "alpha_median", "alpha_max",
                    "pah_size_min", "pah_size_median", "pah_size_max",
                    "r_min", "r_median", "r_max")
    for f in int_fields:
        cols.append(fits.Column(name=f.upper(), format="K",
                                array=np.array([m[f] for m in members], dtype=np.int64)))
    for f in float_fields:
        cols.append(fits.Column(name=f.upper(), format="D",
                                array=np.array([m[f] for m in members], dtype=float)))
    members_hdu = fits.BinTableHDU.from_columns(cols, name="MEMBERS")

    primary_hdu = fits.PrimaryHDU()
    h = primary_hdu.header
    h["NRAW"] = (int(n_raw), "raw (host x alpha x PAH size x R) models sampled")
    h["NKEPT"] = (len(members), "kept templates = rows of this table")
    h["SIGEFF"] = (diagnostics.get("sigma_eff", np.nan), "one noise length (dex), r-net radius")
    h["COVID"] = (diagnostics["coverage_identity"], "frac. raw models within 1 SIGEFF of nearest kept (=1.0)")
    h["COVRAD"] = (diagnostics["covering_radius_over_sigma_eff"], "max covering radius / SIGEFF (<=1.0)")
    h["PACKMIN"] = (diagnostics["min_inter_rep_distance_over_sigma_eff"],
                   "min inter-representative distance / SIGEFF (>=1.0)")
    h["MDISTMED"] = (diagnostics["member_dist_sigma_eff_median"], "median member-to-rep distance / SIGEFF")
    h["MDISTMAX"] = (diagnostics["member_dist_sigma_eff_max"], "max member-to-rep distance / SIGEFF")
    h["ALGO"] = ("greedy max-uncovered-neighbour r-net, no seed", "selection rule")
    h["TIEBRK"] = ("smallest flat index, host>alpha>PAH_size>R order", "deterministic tie-break")
    h["PAHORD0"] = (PAHC_PAH_SIZE_VALUES[0], "PAH_SIZE ordinal 0")
    h["PAHORD1"] = (PAHC_PAH_SIZE_VALUES[1], "PAH_SIZE ordinal 1")
    h["PAHORD2"] = (PAHC_PAH_SIZE_VALUES[2], "PAH_SIZE ordinal 2")
    fits.HDUList([primary_hdu, members_hdu]).writeto(out_path, overwrite=True)
    return out_path


def build_pahc_validation_grid(hosts, host_source, alphas=PAHC_ALPHA_VALUES,
                               pah_sizes=PAHC_PAH_SIZE_VALUES, r_values=PAHC_R_VALUES):
    """A small, dedicated full-factorial sub-grid for the IN-MEMORY,
    PRE-WRITE construction-correctness checks (spec section 8 items 2
    and 3) -- never written to disk.

    Items 2 (alpha/aperture-ratio law) and 3 (hotter-is-brighter) each
    need COMPARABLE PAIRS that differ in exactly one axis -- item 2
    needs one host with every (alpha, pah_size, R); item 3 needs several
    hosts AT A SINGLE log g, varying only T_eff, at one fixed (alpha,
    pah_size, R) -- validate_unit_scale_sanity sorts by T_eff and
    demands strict monotonicity, which only item 3's OWN claim
    ("hotter is brighter at fixed structure") predicts.

    ONE FURTHER SPLIT, found empirically widening the host span past the
    OLD anchors' range: at a fixed log g, the SPS splice hands off
    between source grids (CIFIST below ~3700-3900 K, CK03 above) by
    INTERLEAVING T_eff, not by a clean cut -- e.g. at log g 3.0,
    (3700 CIFIST, 3750 CK03, 3800 CIFIST, 3900 CIFIST, 4000 CK03).
    Deep in this probe's flux-starved blue tail (0.2-0.21 um, many
    orders of magnitude below the SED's peak), the two INDEPENDENTLY
    COMPUTED atmosphere grids disagree enough to break strict T_eff
    monotonicity locally if chained together -- a real splice-boundary
    artefact, confirmed source-by-source (CK03 alone, or CIFIST alone,
    at fixed log g, IS strictly monotonic there; the failure appears
    only when the two are chained in one sorted T_eff sequence). So this
    grid runs item 3 as ONE INDEPENDENT monotonicity group PER SOURCE
    GRID present at the reference log g (a distinct alpha value per
    group keeps them as separate validate_unit_scale_sanity keys, never
    chained together), rather than one chain across the whole span.

    A SECOND, related contamination was found the same way and is fixed
    the same way it is described, not worked around: item 2's cross
    sweeps the FULL alpha range for `ref_host`, which necessarily
    includes the `(alpha, pah_size, R)` triple item 3 reserves as EACH
    source's own monotonicity key (every `source_alpha[source]` value is
    itself one of `alphas`, and `ref_pah`/`ref_r` are themselves swept by
    item 2). So `ref_host` -- which belongs to exactly ONE source -- was
    landing, via that shared key alone, inside every OTHER source's
    item-3 chain too, splice-boundary artefact and all. Each row below
    therefore carries an explicit "_item" tag (2 or the owning source's
    name); `validate_unit_scale_sanity` groups item-3's monotonicity
    check by `(alpha, pah_size, R, _item)`, so item 2's single
    `ref_host` row always sits alone in its own singleton group (nothing
    to compare it against) and never contaminates a source's chain, no
    matter which (alpha, pah_size, R) triple the two crosses happen to
    share.

    Returns one reference host (drawn from the largest such source
    group) crossed with every alpha x PAH size x R, UNION every T_eff
    at the reference log g within EACH source grid at its own
    (alpha, PAH size, R) key. Both checks are properties of
    compute_composite_sed itself, not of which raw points happen to
    survive sampling, so checking them against this cross is exactly as
    informative as checking the full raw grid, at a fraction of the
    cost.

    `host_source` is `load_pahc_host_grid`'s provenance dict entry
    mapping each (T_eff, log g) host to its SPS SOURCE column value."""
    ref_pah, ref_r = pah_sizes[0], r_values[0]

    counts = {}
    for _, g in hosts:
        counts[g] = counts.get(g, 0) + 1
    ref_logg = min(g for g, n in counts.items() if n == max(counts.values()))
    teff_scan_hosts = [h for h in hosts if h[1] == ref_logg]

    by_source = {}
    for h in teff_scan_hosts:
        by_source.setdefault(host_source[h], []).append(h)
    # Largest source group's reference host anchors the item-2 cross
    # (avoids any chance of it coincidentally sharing a T_eff with a
    # host from a DIFFERENT source in a group it is not part of).
    ref_source = max(by_source, key=lambda s: len(by_source[s]))
    ref_host = by_source[ref_source][0]
    if len(alphas) < len(by_source):
        raise AssertionError(
            f"build_pahc_validation_grid: {len(by_source)} source grids "
            f"{sorted(by_source)} at log g={ref_logg} but only {len(alphas)} "
            f"alpha values to key them independently -- widen PAHC_ALPHA_VALUES "
            f"or revisit this grid's design")
    source_alpha = {s: alphas[i] for i, s in enumerate(sorted(by_source))}

    seen = set()
    grid = []
    for a in alphas:
        for z in pah_sizes:
            for r in r_values:
                key = (ref_host, a, z, r)
                if key not in seen:
                    seen.add(key)
                    grid.append({"t_eff": ref_host[0], "log_g": ref_host[1],
                                "alpha": a, "pah_size": z, "r": r,
                                "_item": 2})
    for source, group_hosts in by_source.items():
        a = source_alpha[source]
        for host in group_hosts:
            key = (host, a, ref_pah, ref_r)
            if key not in seen:
                seen.add(key)
                grid.append({"t_eff": host[0], "log_g": host[1],
                            "alpha": a, "pah_size": ref_pah, "r": ref_r,
                            "_item": source})
    return grid


def format_pahc_model_name(index):
    """Index-based name ('pahc_00001', spec section 4) -- avoids the
    MODEL_NAME width limits a 5-axis descriptive name would risk."""
    return f"pahc_{index + 1:05d}"


# ====================================================================
# Stage 6: assemble flux.fits / parameters.fits / models.conf
# ====================================================================

@dataclass
class PahcModelSet:
    """Combined, ready-to-write arrays across all (t_eff, alpha,
    pah_size, r) models. Return type of prepare_pahc_model_arrays."""
    model_names: np.ndarray
    t_eff: np.ndarray
    log_g: np.ndarray               # per-model, from the SPS host grid (load_pahc_host_grid)
    alpha: np.ndarray
    pah_size: np.ndarray
    r: np.ndarray
    wave_um_desc: np.ndarray        # descending -- matches the SEDCube ordering fix
    freq_hz_desc: np.ndarray        # ascending
    values_mjy: np.ndarray          # (n_models, 1, n_wave) -- point-source shape
    uncertainties_mjy: np.ndarray   # same shape, all zero (theoretical models, spec section 3)
    item_tag: np.ndarray            # grid's "_item" (2, source name, or "" outside the
                                     # validation grid) -- validate_unit_scale_sanity's
                                     # extra groupby key, see build_pahc_validation_grid


def pahc_distance_placeholder():
    """The DISTANCE value written into flux.fits.

    SEDCube.read requires the keyword, but nothing in this library uses
    it: the models are aperture-independent and carry no reference
    distance. It is the same structural placeholder galaxy and H2-shock
    write, taken from constants so all three agree."""
    return REFERENCE_DISTANCE_CM


def prepare_pahc_model_arrays(grid, wave_um, sps_dir, dl21_dir):
    """Build every (t_eff, alpha, pah_size, r) composite on wave_um
    (ascending in), then reverse to descending wavelength / ascending
    frequency to avoid the SEDCube.read() reorder bug (see
    galaxy_curate.py's module docstring for the original discovery).

    No aperture arguments: the library is aperture-independent and the
    band-aperture ratios are inside compute_composite_sed. Output carries
    the point-source aperture axis of length 1 that SPS, GAL and
    H2-shock all use."""
    n_models = len(grid)
    values = np.empty((n_models, wave_um.size), dtype=np.float64)
    model_names = []

    star_cache = {}
    excess_cache = {}

    for i, p in enumerate(grid):
        t_key = (p["t_eff"], p["log_g"])
        if t_key not in star_cache:
            wave_native, flux_native = load_sps_stellar_sed(
                sps_dir, p["t_eff"], p["log_g"], PAHC_Z)
            star_cache[t_key] = resample_sed_log_log(wave_native, flux_native, wave_um)
        s_resampled = star_cache[t_key]

        e_key = p["pah_size"]
        if e_key not in excess_cache:
            wave_native, p_hat_native = load_pahc_excess_shape(dl21_dir, p["pah_size"])
            p_hat = resample_sed_log_log(wave_native, p_hat_native, wave_um)
            # Re-anchor ON THE PAH-C GRID. load_pahc_excess_shape anchors
            # at 8um on DL21's native grid (it must, to clear FLUX_FLOOR
            # before resampling); compute_composite_sed then forms
            # P_hat/P_hat(8um) from the RESAMPLED array. Without this the
            # two differ by a constant ~1.001, which would break the
            # exact anchor identity validate_anchor_identity checks.
            excess_cache[e_key] = p_hat / np.interp(ANCHOR_WAVELENGTH_UM, wave_um, p_hat)
        p_hat_resampled = excess_cache[e_key]

        values[i] = compute_composite_sed(
            s_resampled, p_hat_resampled, wave_um, p["r"], p["alpha"])
        model_names.append(format_pahc_model_name(i))

    wave_um_desc = wave_um[::-1]
    freq_hz_desc = C_UM_S / wave_um_desc
    # (n_models, 1, n_wave): the point-source aperture axis every
    # aperture-independent library in the release carries.
    values_desc = values[:, ::-1].astype(np.float32)[:, np.newaxis, :]
    uncertainties_desc = np.zeros_like(values_desc)

    return PahcModelSet(
        model_names=np.array(model_names),
        t_eff=np.array([p["t_eff"] for p in grid]),
        log_g=np.array([p["log_g"] for p in grid]),
        alpha=np.array([p["alpha"] for p in grid]),
        pah_size=np.array([p["pah_size"] for p in grid]),
        r=np.array([p["r"] for p in grid]),
        wave_um_desc=wave_um_desc, freq_hz_desc=freq_hz_desc,
        values_mjy=values_desc, uncertainties_mjy=uncertainties_desc,
        item_tag=np.array([str(p.get("_item", "")) for p in grid]),
    )


def write_flux_fits(model_set, out_path, distance_cm):
    """Write flux.fits through the shared release writer: PRIMARY
    (all-valid mask + DISTANCE, the aperture reference frame -- NOT a
    plug, spec section 3), MODEL_NAMES, SPECTRAL_INFO (descending
    wavelength / ascending frequency), APERTURES (the shared 20 values),
    VALUES/UNCERTAINTIES."""
    return model_io.write_flux_cube(
        out_path,
        names=model_set.model_names,
        wave_um_desc=model_set.wave_um_desc,
        freq_hz_desc=model_set.freq_hz_desc,
        values=model_set.values_mjy,
        distance_cm=distance_cm,
        # The single point-source sentinel row, as GAL and H2-shock write.
        # POINT_SOURCE_APERTURE_AU, not a bare 1e-30: it happens to equal
        # FLUX_FLOOR, and model_directory_format.md section 6 warns
        # explicitly that the two must not be aliased -- one is a
        # structural "no aperture dependence" marker, the other a mJy
        # clip.
        # sedfitter's aperture-independent branch reads conv.flux[:, 0]
        # and ignores the aperture entirely, so a multi-aperture axis
        # here would NOT error -- it would silently take column 0, which
        # under the old 20-value axis is the 100 AU slice, i.e. almost no
        # excess. Length 1 makes that unreachable.
        apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
        uncertainties=model_set.uncertainties_mjy,
        name_format=MODEL_NAME_FORMAT,
        distance_comment="structural placeholder (aperture-independent)",
    )


def write_parameters_fits(model_set, out_path):
    """Write parameters.fits: MODEL_NAME (primary key, row order must
    match flux.fits exactly) + T_EFF, LOGG, ALPHA, PAH_SIZE, R. LOGG is a
    PER-MODEL column (D-15): since each model's host now carries its own
    (T_eff, log g) off the SPS host grid (load_pahc_host_grid) rather
    than one global value, it can no longer be asserted in the PRIMARY
    header the way a single fixed quantity can -- a per-model quantity
    may not also be asserted globally. Everything
    still fixed (not gridded, not per-model) is recorded in the PRIMARY
    header (spec section 4) so a reader can reconstruct the full model
    definition from the file alone.

    Float columns are `D`: the release convention is one float format
    across every library."""
    params_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="MODEL_NAME", format=MODEL_NAME_FORMAT, array=model_set.model_names),
            fits.Column(name="T_EFF", format="D", unit="K", array=model_set.t_eff),
            # Per-model, not a PRIMARY-header keyword (D-15): retains the
            # keyword name the old fixed global LOGG card carried, so a
            # reader who knew the old file finds continuity, not a rename.
            fits.Column(name="LOGG", format="D", array=model_set.log_g),
            fits.Column(name="ALPHA", format="D", array=model_set.alpha),
            # PAH_SIZE is DL21's published size-distribution label
            # ("sma"/"std"/"lrg"), not a number -- stored as text rather
            # than encoded to an integer so the file needs no legend.
            fits.Column(name="PAH_SIZE", format="3A", array=model_set.pah_size),
            fits.Column(name="R", format="D", array=model_set.r),
        ],
        name="PARAMETERS",
    )
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["ZH"] = (PAHC_Z, "fixed photosphere [Z/H]")
    primary_hdu.header["LG_U"] = (float(PAHC_LG_U_TOKEN), "fixed DL21 log10 starlight intensity")
    primary_hdu.header["PAHION"] = (PAHC_IONIZATION, "fixed DL21 PAH charge distribution")
    primary_hdu.header["F_PAH"] = (PAHC_F_PAH, "fixed PAH abundance scaling (1 = DL21 fiducial)")
    primary_hdu.header["RADFIELD"] = (PAHC_RADIATION_FIELD, "fixed DL21 illuminating spectrum")
    primary_hdu.header["EXCESSRC"] = ("DL21", "Draine et al. 2021 ApJ 917 3 (was DL07)")
    fits.HDUList([primary_hdu, params_hdu]).writeto(out_path, overwrite=True)
    return out_path


# version 3 (D-15): the cool-giant anchor extension changes parameters.fits'
# schema (a per-model LOGG column replaces the fixed PRIMARY-header keyword)
# and the T_EFF axis grows from 8 to 11 anchors -- a real content change to
# flag to any reader who cached version 2's shape.
PAHC_MODELS_CONF_VERSION = 3


def write_models_conf(out_path, name=DEFAULT_MODEL_SET_NAME):
    """Write models.conf through the shared writer.

    aperture_dependent = NO, matching SPS, GAL and H2-shock: sedfitter
    gives its ndim == 3 branch no free scale, and PAH-C needs one. See
    the APERTURE TREATMENT note in the CONSTANTS block."""
    return model_io.write_models_conf(out_path, name=name, aperture_dependent=False,
                                      version=PAHC_MODELS_CONF_VERSION)


def write_classmap_fits(model_set, out_path, model_dir=None):
    """Write classmap.fits through the shared release writer.

    Nothing downstream recomputes these labels, so this file's header is
    the only record of how they were derived -- which for PAH-C means
    recording that the class is a library declaration and that no
    subclass is defined.

    Emitted from `model_set`, the SAME in-memory arrays that went into
    flux.fits, never by re-reading parameters.fits: CLASSMAP duplicates
    what that file already knows and so is capable of disagreeing with
    it. Building both in one run from one array is what prevents drift.
    Passing `model_dir` additionally asserts the row order matches
    flux.fits on disk.
    """
    return model_io.write_classmap_fits(
        out_path,
        names=model_set.model_names,
        class_id=MODEL_CLASS,
        subclass=np.full(len(model_set.model_names), PAHC_SUBCLASS),
        class_legend=CLASS_LEGEND,
        subclass_legend=SUBCLASS_LEGEND,
        model_dir=model_dir,
        provenance=(
            ("CLASS_SOURCE", "library declaration; all models here are this class"),
            # Not "this library has no subclasses" -- it has one, whose
            # meaning is that the library defines none. The distinction
            # matters to a consumer deciding whether to expect more later.
            ("SUBCLASS_SOURCE", "none defined; every model carries NONE"),
            ("SUBCLASS_REF", "n/a -- no subclass axis"),
            ("SUBCLASS_NOTE", "grid axes are in parameters.fits, not subclasses"),
            ("LEGEND_SOURCE", "class from GUTERMUTH_LABELS 39; text authored here"),
        ),
    )


# ====================================================================
# Stage 8: construction-correctness validation (spec section 8, items
# 1-4). Items 5 and 7 are scientific validation and live in Stage 9;
# item 6 (grid-density normalisation) is a fit-time weighting concern,
# out of scope for a curation module.
# ====================================================================

def validate_anchor_identity(model_dir, grid, sps_dir, wave_um,
                             tolerance=PAHC_ANCHOR_TOLERANCE):
    """Spec section 8 item 1: M(8um)/s(8um) == 1 + R, for every model.

    Exact and unconditional: the band-aperture ratio is 1 in the IRAC
    beam by definition, so the identity is a property of every stored
    SED rather than of one favoured aperture.

    s(8um) is recomputed independently here (fresh load_sps_stellar_sed
    + resample per t_eff, matching construction) rather than reused from
    build-time state, so this is a genuine cross-check of the on-disk
    flux.fits, not a tautology. Returns the max relative deviation;
    raises AssertionError past `tolerance`."""
    meta = model_io.read_flux_metadata(os.path.join(model_dir, "flux.fits"))
    with fits.open(os.path.join(model_dir, "flux.fits")) as h:
        values = h["VALUES"].data          # (n_models, 1, n_wave), descending wavelength

    wave_um_asc = meta.lam_um[::-1]
    star_cache = {}
    max_rel_dev = 0.0
    for i, p in enumerate(grid):
        t_key = (p["t_eff"], p["log_g"])
        if t_key not in star_cache:
            wave_native, flux_native = load_sps_stellar_sed(
                sps_dir, p["t_eff"], p["log_g"], PAHC_Z)
            star_cache[t_key] = resample_sed_log_log(wave_native, flux_native, wave_um)
        s_8 = np.interp(ANCHOR_WAVELENGTH_UM, wave_um, star_cache[t_key])

        m_8 = np.interp(ANCHOR_WAVELENGTH_UM, wave_um_asc, values[i, 0, ::-1])
        expected = 1.0 + p["r"]
        rel_dev = abs(m_8 / s_8 - expected) / expected
        max_rel_dev = max(max_rel_dev, rel_dev)

    if max_rel_dev > tolerance:
        raise AssertionError(
            f"anchor identity M(8um)/s(8um) != 1+R: max relative deviation "
            f"{max_rel_dev:.2e} exceeds tolerance {tolerance:.2e}")
    return max_rel_dev


def validate_band_aperture_behavior(model_set, grid, tolerance=PAHC_APERTURE_TOLERANCE):
    """Spec section 8 item 2, rewritten for the aperture-independent
    build: the excess must carry the band-aperture ratio
    (theta_b/theta_IRAC)^alpha, and the star must not.

    Uses two differences, so that everything except the aperture factor
    cancels ALGEBRAICALLY rather than being supplied by the test:

      * difference two models that share (T_eff, PAH size, alpha) and
        differ only in R. The star term cancels -- which is itself the
        confirmation that the star carries no aperture factor, since any
        residual would survive the subtraction:

            D_alpha(lambda) = dR * s(8um) * P_hat(lambda)/P_hat(8um)
                                 * (theta(lambda)/theta_IRAC)^alpha

      * then ratio two such differences that share everything EXCEPT
        alpha. s(8um), dR and the whole excess shape cancel:

            D_a2(lambda) / D_a1(lambda)
                == (theta(lambda)/theta_IRAC)^(a2 - a1)

    The right-hand side involves no model quantity at all -- only the
    published band apertures and the two alphas -- so the test cannot be
    satisfied by construction. Deriving the expected ratio from the
    models themselves would make it a tautology; do not.

    Probed at one wavelength inside each aperture group so all three
    ratios are exercised, including the 2MASS and MIPS24 groups where
    the factor differs from 1. Checked in memory, pre-write."""
    wave_um_asc = model_set.wave_um_desc[::-1]
    values_asc = model_set.values_mjy[:, 0, ::-1]
    probes = (2.2, 5.0, 24.0)          # one per aperture group

    # index by (T_eff, pah_size, R) -> {alpha: row}
    by_model = {}
    for i, p in enumerate(grid):
        by_model.setdefault((p["t_eff"], p["pah_size"], p["r"]), {})[p["alpha"]] = i

    r_values = sorted({p["r"] for p in grid})
    r_lo, r_hi = r_values[0], r_values[-1]

    max_rel_dev = 0.0
    n_checked = 0
    for (t_eff, size, r), _rows in by_model.items():
        if r != r_hi:
            continue
        lo_rows = by_model.get((t_eff, size, r_lo))
        hi_rows = by_model.get((t_eff, size, r_hi))
        if not lo_rows or not hi_rows:
            continue
        alphas = sorted(set(lo_rows) & set(hi_rows))
        if len(alphas) < 2:
            continue
        a1, a2 = alphas[0], alphas[-1]

        def difference(alpha, wave):
            hi = np.interp(wave, wave_um_asc, values_asc[hi_rows[alpha]])
            lo = np.interp(wave, wave_um_asc, values_asc[lo_rows[alpha]])
            return hi - lo

        for wave in probes:
            d1, d2 = difference(a1, wave), difference(a2, wave)
            if d1 == 0:
                continue
            observed = d2 / d1
            expected = band_aperture_scaling(np.array([wave]), a2 - a1)[0]
            max_rel_dev = max(max_rel_dev, abs(observed - expected) / abs(expected))
            n_checked += 1

    if n_checked == 0:
        raise AssertionError("band-aperture behaviour: no comparable model pairs found")
    if max_rel_dev > tolerance:
        raise AssertionError(
            f"band-aperture behaviour inconsistent: max relative deviation "
            f"{max_rel_dev:.2e} exceeds tolerance {tolerance:.2e}")
    return max_rel_dev, n_checked


def find_unit_scale_sanity_probe_index(model_set):
    """The DYNAMIC probe index for validate_unit_scale_sanity (D-15's gate
    generalization): the bluest grid wavelength index at which EVERY
    T_eff anchor's photosphere flux exceeds FLUX_FLOOR.

    WHY DYNAMIC, NOT THE FIXED 0.2um GRID BLUE EDGE (the pre-D-15 gate).
    The gate's INTENT is a T_eff-ordering check where the photosphere
    dominates absolutely -- a pure statement about the star. The fixed
    0.2um probe served that intent for the original 8 warm CK03/CIFIST
    anchors (3500-6000K), all of which have real, distinguishable flux
    there. It stopped serving that intent once the cool-giant extension
    (D-15) added 2800K and 3000K anchors (log g +1.0, AGSS2009): both
    carry GENUINELY ZERO native flux at every wavelength up to ~0.207um
    -- deep molecular-UV (TiO/VO/H2O) blanketing, confirmed directly
    against SPS's own flux.fits, not a resampling artifact or a bug.
    0.2um falls inside that zero-flux region for BOTH, so after
    resample_sed_log_log's FLUX_FLOOR clip they tie EXACTLY, and a
    strict ">" comparison can never be satisfied there regardless of the
    real physics -- the probe point itself had become unphysical for the
    extended grid, not the comparison.

    Two alternatives were considered and rejected. Relaxing the
    comparison to ">=" would silently accept a genuine tie ANYWHERE, for
    every library that uses this gate, not just this one known case --
    far too broad a weakening for a narrow, understood cause. Special-
    casing "equal at FLUX_FLOOR is not a violation" would bake the
    FLUX_FLOOR constant into the gate's pass/fail logic, coupling a
    curation-module implementation detail (the clip value) to a
    validation concern that should not need to know it. Moving the probe
    wavelength is the fix that stays true to the gate's actual intent:
    find where the star ITSELF is resolvable for every anchor, and
    compare there.

    D-12's 0.2um GRID blue edge is unaffected by any of this -- it exists
    so a Gaia G convolution does not silently lose 84% of G's response
    weight (G spans 0.327-1.050um), and every anchor has real flux from
    0.327um onward. Only THIS gate's probe point moves; the grid itself
    does not.

    Uses one representative row per distinct T_eff (the first grid row
    for that T_eff): blueward of DL21's 1.0056um excess turn-on the
    excess is exactly zero-padded, so composite VALUES equal the bare
    photosphere for ANY (alpha, pah_size, R) at that T_eff, and the probe
    found here is always well blueward of that turn-on for every anchor
    currently in the grid (confirmed: the two coolest anchors first clear
    FLUX_FLOOR by ~0.207um, two decades of wavelength blueward of
    1.0056um). If a future anchor were cool enough that no grid
    wavelength cleared FLUX_FLOOR for it at all, this function raises
    rather than silently returning a probe past the excess turn-on.

    Returns the descending-array index (matching model_set.wave_um_desc /
    model_set.values_mjy's wavelength axis)."""
    values = model_set.values_mjy
    t_eff_values = sorted(set(float(t) for t in model_set.t_eff))
    row_by_t_eff = {t: int(np.argmax(model_set.t_eff == t)) for t in t_eff_values}

    n_wave = model_set.wave_um_desc.size
    clears_floor = np.ones(n_wave, dtype=bool)
    for row in row_by_t_eff.values():
        clears_floor &= values[row, 0, :] > FLUX_FLOOR

    if not clears_floor.any():
        raise AssertionError(
            "unit/scale sanity: no grid wavelength exists at which every T_eff "
            "anchor's photosphere clears FLUX_FLOOR -- the gate cannot find a "
            "probe point where the star is resolvable for every anchor")

    # wave_um_desc is DESCENDING (index 0 = reddest, index n_wave-1 =
    # bluest), so among indices that clear the floor for every anchor,
    # the BLUEST one is the one with the LARGEST index.
    return int(np.where(clears_floor)[0].max())


def validate_unit_scale_sanity(model_set):
    """Spec section 8 item 3: composite fluxes are mJy at D_ref, finite
    and positive everywhere, and a hotter-T_eff composite really is
    brighter -- i.e. SPS's Stefan-Boltzmann-consistent surface flux
    carries through the resample and the composite sum rather than being
    silently renormalised.

    Checked at the bluest grid wavelength at which EVERY anchor's
    photosphere clears FLUX_FLOOR (find_unit_scale_sanity_probe_index,
    D-15's gate generalization -- see its docstring for why this replaced
    a fixed 0.2um probe), where the photosphere dominates absolutely and
    the PAH excess is zero-padded to nothing, so the ordering is a pure
    statement about the star. Returns the number of (alpha, pah_size, R)
    groups checked; raises AssertionError on a violation."""
    values = model_set.values_mjy
    if not np.isfinite(values).all():
        raise AssertionError("composite VALUES contain non-finite fluxes")
    if not (values > 0).all():
        raise AssertionError("composite VALUES contain non-positive fluxes")

    blue_idx = find_unit_scale_sanity_probe_index(model_set)
    blue_flux = values[:, 0, blue_idx]

    keys = [(float(a), str(z), float(r), str(it)) for a, z, r, it in
            zip(model_set.alpha, model_set.pah_size, model_set.r, model_set.item_tag)]
    n_groups = 0
    for key in sorted(set(keys)):
        rows = np.array([i for i, k in enumerate(keys) if k == key])
        ordered = blue_flux[rows[np.argsort(model_set.t_eff[rows])]]
        if not np.all(np.diff(ordered) > 0):
            raise AssertionError(
                f"unit/scale sanity: probe-wavelength flux is not monotonically "
                f"increasing with T_eff at (alpha, pah_size, R)={key}")
        n_groups += 1
    return n_groups


def _independent_band_flux(wave_um_asc, flux_mjy_asc, band):
    """One band flux by an INDEPENDENT trapezoidal convolution.

    Deliberately NOT model_convolution's machinery, even though that is
    the blessed path for building convolved/. This function exists so
    validate_convolution_consistency can compare the pipeline's output
    against something that shares no code with it; reusing the pipeline
    here would make that gate circular. It is the one place in this
    module where not reusing a shared helper is the point.

    F = int F_nu R dnu / int R dnu, on the filter's own grid."""
    fc = load_filter_curve(band)
    nu_hz = C_UM_S / fc.wave_um
    order = np.argsort(nu_hz)
    nu_hz, response = nu_hz[order], fc.response[order]
    f_nu = np.interp(C_UM_S / nu_hz, wave_um_asc, flux_mjy_asc, left=0.0, right=0.0)
    return np.trapz(f_nu * response, nu_hz) / np.trapz(response, nu_hz)


def validate_convolution_consistency(model_dir, bands=("I1", "I2", "I3"), n_sample=24,
                                     tolerance=PAHC_CONVOLUTION_TOLERANCE_MAG):
    """Spec section 8 item 4: the pipeline-convolved band fluxes at ap_ref
    must reproduce the colours of an INDEPENDENT convolution of the same
    stored SEDs, to `tolerance` mag.

    Without it, convolved/ ships unverified: every other check in this
    module reads flux.fits and passes whether or not the convolution step
    did the right thing.

    Samples n_sample models evenly across the grid -- the check is about
    the convolution path, not about any one model. Returns the max colour
    discrepancy in mag; raises AssertionError past `tolerance`."""
    meta = model_io.read_flux_metadata(os.path.join(model_dir, "flux.fits"))
    rows = np.unique(np.linspace(0, len(meta.names) - 1, n_sample).astype(int))
    with fits.open(os.path.join(model_dir, "flux.fits")) as h:
        seds = np.asarray(h["VALUES"].data[rows, 0, ::-1], dtype=float)   # ascending

    wave_um_asc = meta.lam_um[::-1]
    analytic = {b: np.array([_independent_band_flux(wave_um_asc, sed, b) for sed in seds])
                for b in bands}
    pipeline = {b: model_io.load_convolved_total_flux_mjy(model_dir, b)[1][rows]
                for b in bands}

    max_dev = 0.0
    for b_a, b_b in zip(bands[:-1], bands[1:]):
        c_analytic = vega_color(analytic[b_a], analytic[b_b], b_a, b_b)
        c_pipeline = vega_color(pipeline[b_a], pipeline[b_b], b_a, b_b)
        max_dev = max(max_dev, float(np.max(np.abs(c_analytic - c_pipeline))))

    if max_dev > tolerance:
        raise AssertionError(
            f"convolved/ disagrees with an independent convolution of flux.fits: "
            f"max colour deviation {max_dev:.3e} mag exceeds tolerance {tolerance:.3e} mag")
    return max_dev

