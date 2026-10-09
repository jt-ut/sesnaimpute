"""
galaxy_curate.py
====================================================================
Curation pipeline for the combined galaxy-contaminant model set (Polletta
2007 SWIRE, Berta 2013, Brown 2014 GALSEDATLAS), producing a sedfitter-ready
model directory (models.conf, parameters.fits, flux.fits, info.fits,
members.fits, convolved/{band}.fits) from the three sources' raw downloads.
This is the point of record for how that model set is built -- see
build/galz.py for the driver that calls these stages in order.

REBUILT to the owner's six-library specification (2026-10-08). The spec has
three parts, and the pipeline below is organised around them:

  1. COVERAGE -- each source set stays within its own PUBLISHED redshift
     reach, never extrapolated: Polletta 2007 z < 2, Berta 2013 z < 3,
     Brown 2014 z < 0.06 (GALAXY_SOURCE_REDSHIFT_CAP). Carried over
     unchanged from the earlier 2026-10-09 curation session.

  2. SAMPLING AT THE FITTER'S RESOLUTION -- one global r-net, at one noise
     length (SIGEFF -- twice the surveys' published absolute-calibration
     floor, a literature constant; see read_sampling_quotient_space),
     over the WHOLE capped raw set, pooled across all three sources: every raw (template,
     z) row within SIGEFF of a kept representative, no two kept
     representatives closer than SIGEFF. The kept count is whatever this
     yields -- no per-source budget, no strata, no interpolation. The raw
     set itself is now generated on a DENSE per-source redshift grid
     (RAW_GRID_N_Z, same density for every source) within each source's own
     cap, dense enough that the r-net, not the grid, sets the final
     spacing (build/galz.py runs and reports the doubling test this rests
     on) -- replacing the single shared z=0-6 grid the 2026-10-09 session
     cut down per source after the fact, which starved Brown's narrow
     (z<0.06) cap of resolution relative to Polletta/Berta's wider ones.

  3. MEMBERS -- every kept representative's represented set (a Voronoi
     partition of the raw set over the fixed representatives, never the
     other way round) is summarised in members.fits: its count, its
     SUBCLASS (AGN/COMP/PAH/PASS) fractions, its TEMPLATE_CLASS/DONLEY_AGN
     composition, and its REDSHIFT/PAH_EW_6_2 ranges. The representative's
     OWN point values stay in parameters.fits/info.fits (this library's
     `models` group analogue), for schema parity with the other five
     libraries.

No population weight, prior quantity, or Gaia quantity enters anywhere in
this module (independence, item 4 of the spec).

Pipeline, in sequence (mirrors the stage order in build/galz.py):

0. Acquisition (fetch_polletta2007_swire_templates /
   fetch_berta2013_templates / fetch_brown2014_galsedatlas). Each source
   tree is pulled into download_dir from a PINNED reference -- Polletta
   from a git commit SHA rather than a moving branch, Berta and Brown from
   an explicit file manifest rather than a live directory crawl -- and every
   file is checked against a recorded sha256 before the cache counts as
   complete. An upstream edit or a truncated download therefore fails the
   build loudly instead of silently changing the model set.

1. Dense per-source redshift grid + rest-frame template loading. Each
   source has its own loader (load_polletta_rest_frame_templates,
   load_berta_rest_frame_templates, load_brown_rest_frame_templates) --
   deliberately NOT one generic loader with an if/elif on source name,
   since each source's native wavelength unit and file layout is genuinely
   different (see each loader's docstring). "Redshifting" here is a
   trivial wavelength stretch (observed_wavelength_um), not yet resampled
   onto a common grid -- flux is left unchanged (no 1/(1+z) dimming; a
   single wavelength-independent factor is degenerate with sedfitter's
   free per-source scale factor).

   build_redshift_grid is called ONCE PER SOURCE now, with that source's
   OWN GALAXY_SOURCE_REDSHIFT_CAP as z_max and the shared RAW_GRID_N_Z as
   n_z -- not once, globally, at z_max=6.0 (see item 2 above and build_
   redshift_grid's own docstring for why that mattered for Brown
   specifically).

2. Master wavelength grid (build_master_wavelength_grid): observed-frame,
   log-uniform, spanning the 8 SESNA bands' full bandpass footprint with
   margin, extended blue to 0.3um to also cover a future Gaia G-band need
   (true blue edge 0.32um) without under-resolving PAH/silicate features.

3. TEMPLATE_CLASS (5-way superclass: Elliptical/Spiral/Starburst/AGN/
   Composite) and PAH_STRENGTH/PAH_EW_6_2 (local-continuum equivalent
   width, um; see measure_pah_strength/measure_pah_ew_6_2). Classification
   is per-source (classify_polletta_templates/classify_berta_templates/
   classify_brown_templates), same no-shared-dispatcher reasoning as the
   loaders. Both quantities are intrinsic REST-FRAME properties of each
   template -- independent of redshift and of the sampling step below --
   so this runs over the 185 base templates, reading the rest-frame
   archives from Stage 1 directly.

4. classify_parent_subclasses: SUBCLASS + DONLEY_AGN, also per PARENT
   TEMPLATE (185), via a dedicated throwaway z=0-only convolution
   (work_dir/parents_z0/), through the OFFICIAL sedfitter path
   (model_convolution.build_convolved_bands) -- not the fast internal
   convolution Stage 5 below uses for the sampling decision. This used to
   run AFTER Stage 6, reading the FINAL shipped set's own ".z0.000" rows;
   that assumption breaks once the r-net (Stage 6) is free to choose any
   redshift as a template's representative, so classification is computed
   here, first, decoupled from whatever the sampling later picks.
   donley2012_agn and assign_galaxy_subclass are reused unmodified.

5. compute_raw_grid_photometry: the dense raw set's 8-band SESNA
   photometry (fast internal convolution, build_fast_band_filters/
   fast_band_fluxes_mjy -- not sedfitter's convolve_model_dir, which would
   need an on-disk model directory per raw row), built and discarded one
   template at a time so the full (n_templates, n_z, n_wave) cube this
   density would otherwise need is never materialized.

6. sample_raw_grid: project to the 5-D quotient space (read_sampling_
   quotient_space, density.build_quotient_space -- read-only, no register
   written) and run greedy_max_cover_r_net at radius SIGEFF (item 2
   above); then assign_nearest_representative partitions the whole raw
   set into each kept representative's MEMBERS Voronoi cell.

7. prepare_combined_model_arrays: resample ONLY the kept representatives'
   full master-grid SEDs (directly from the Stage 1 rest-frame archives --
   the kept count is small enough that no intermediate full-resolution
   cube is needed), apply the two correctness fixes every prior session
   already established:

   a. F_lambda -> F_nu conversion. All three sources store flux as F_lambda
      (confirmed from each source's own documentation). sedfitter's
      VALUES/BUNIT convention requires F_nu (mJy). Converted via
      F_nu = F_lambda * lambda^2 / c, lambda in Angstrom. For Polletta/
      Berta (arbitrarily-normalized, not real physical flux) this doesn't
      yield a "real" mJy value, but it applies the correct wavelength-
      dependent reshaping of relative color, which is what matters for
      filter convolution -- absolute scale is irrelevant since sedfitter
      fits a free per-source scale factor.

   b. Wavelength stored DESCENDING (not ascending). Confirmed via direct
      testing that sedfitter.sed.cube.SEDCube.read(order='nu') -- what
      convolve_model_dir uses -- has a real bug for ascending-stored files:
      it reverses the 1D wavelength array during reordering but not the
      flux cube's wavelength axis, silently mismatching wavelength and
      flux.

   UNCERTAINTIES are written all-zero: none of the three sources provide
   per-wavelength flux uncertainties, so this is a documented absence, not
   a fabricated placeholder.

   flux.fits and models.conf go through model_io.write_flux_cube /
   model_io.write_models_conf, the release-wide writers shared by every
   library. parameters.fits/info.fits/members.fits stay local: their
   columns are this library's own schema. TEMPLATE_CLASS, PAH_EW_6_2 and
   DONLEY_AGN (Stage 4's lookup) and SUBCLASS land in parameters.fits/
   info.fits in ONE pass now (write_parameters_fits/write_info_fits) --
   there is no more Stage-ordering reason to defer DONLEY_AGN/SUBCLASS to
   after Stage 8, since classify_parent_subclasses (Stage 4) no longer
   depends on the kept set's own convolution.

8. build_members_summary + write_members_fits: the MEMBERS group (item 3
   above), built from Stage 6's Voronoi partition plus Stage 4's per-
   template lookup.

9. build_convolved_bands: runs sedfitter's own convolve_model_dir against
   the assembled flux.fits, for the 8 SESNA bands only (2MASS J/H/Ks, IRAC
   I1-I4, MIPS M1), then stamps CONVMETH and verifies the written files.
   This is the SHIPPED convolution -- the only one whose output is read by
   anything downstream of this build -- and runs on the small kept set
   only, exactly as before.

Shared infrastructure, NOT a numbered stage: filter construction
(build_sedfitter_filter -- an explicit-normalization wrapper around
sedfitter's Filter class) and the convolution itself both live in
model_convolution.py; this module reaches them through build_convolved_
bands (the shipped path) and, separately, build_fast_band_filters/
fast_band_fluxes_mjy (the sampling-decision path, Stage 5) -- the two are
deliberately different code paths for different purposes (see each
function's own docstring).

Model naming: "{tag}.{template_name}.z{redshift:.4f}", tag in
{pol07, ber13, bro14}. MODEL_NAME is the primary key in every table
(flux.fits, parameters.fits, info.fits, members.fits) and row order must
match exactly across all four -- sedfitter's own convolve step enforces
this with a direct equality check for the first three; members.fits is
built directly from the same in-memory `combined` object, so it cannot
drift from them.
====================================================================
"""

import csv
import glob
import heapq
import os
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

from sesnaimpute.sed_models.constants import (
    BANDS, C_UM_S, LIBRARY_SAMPLING_SIGMA_LOG_VECTOR, LOGD_STEP, MJY_TO_CGS,
    POINT_SOURCE_APERTURE_AU, REFERENCE_DISTANCE_CM,
)
from sesnaimpute.sed_models.curate import model_io
from sesnaimpute.sed_models.curate.model_convolution import (
    build_convolved_bands, build_sedfitter_filter,
)
# READ-ONLY: the quotient-space PROJECTION (5-D, removing the gray/scale
# direction and both Av laws) the r-net sampling (item 2 of the owner's
# library specification) uses. sigma_log itself is NOT read from
# sed_models_register any more (see read_sampling_quotient_space's
# docstring) -- only this one projection function, which depends only on
# sigma_log (now a literature constant) and the two Av laws, never on a
# library or a register. Neither sed_models_register/ nor sed_models/
# registers/ is ever written from this module.
from sesnaimpute.sed_models.register import density as register_density

ANGSTROM_TO_UM = 1e-4
MODEL_NAME_FORMAT = model_io.MODEL_NAME_FORMAT  # "34A", the release-wide cube width
FLUX_FLOOR = 1e-30  # flux clip before any log transform -- NOT the aperture plug

SOURCE_TAGS = ("pol07", "ber13", "bro14")

# The speed of light, the mJy scale factor, the DISTANCE plug and the
# point-source APERTURE plug used to be re-declared here. They are shared
# facts, so they now come from sesnacomplete.constants:
#   C_UM_S                    -> Angstrom/s as C_UM_S * 1e4 (bit-exact)
#   MJY_TO_CGS                -> the erg/s/cm^2/Hz per mJy factor
#   REFERENCE_DISTANCE_CM     -> flux.fits DISTANCE
#   POINT_SOURCE_APERTURE_AU  -> the single APERTURES row
# REFERENCE_DISTANCE_CM carries the exact literal 3.08568025e21, which is
# what the upstream Robitaille release's DISTANCE header carries and what
# every library here already shipped. It is deliberately NOT derived from
# constants.PC_CM * 1000 (which differs in the 7th significant digit); the
# point is to match the release, not to state a better parsec.


# ====================================================================
# Stage 0: acquisition -- pinned, checksummed fetches of the three sources
# ====================================================================
# Each fetcher is a stage function called by the driver, replacing the three
# download_*.sh shell scripts this pipeline was developed with. Two things
# changed in the move, and both are the point of the move:
#
#   1. The references are PINNED. The Polletta script pulled a GitHub
#      `master` branch and the Berta script CRAWLED a live CDS directory
#      autoindex, so in both cases an upstream edit would have silently
#      changed the model set with nothing in the build to notice.
#   2. Every file carries a sha256 and the CACHE KEY IS THE VERIFIED FILE
#      (model_io.fetch_pinned), so a killed download cannot be mistaken for
#      a complete one and a changed upstream artifact raises rather than
#      being absorbed.
#
# Per-source parsing deliberately stays in the three loaders below; only the
# download/verify/cache mechanism is shared.

# --- Polletta 2007 (SWIRE) -------------------------------------------
# Polletta et al. 2007, ApJ 663, 81. 25 templates (3 elliptical, 7 spiral,
# 6 starburst, 7 AGN, 2 composite), ~1000 Angstrom to 1000 um; col 1
# wavelength [Angstrom], col 2 F_lambda normalized at 5500 A.
# Original host: M. Polletta, IASF Milano (INAF),
#   http://www.iasf-milano.inaf.it/~polletta/templates/swire_templates.html
# Retrieved instead from the version-controlled qsfit mirror (identical
# files), pinned to the single commit that has ever touched that directory
# -- the tree was added in "First commit" (2016-12-04) and never modified,
# so this pin is the whole history of the files, not a snapshot of a moving
# branch. Terms: free for research use; cite the paper.
POLLETTA_COMMIT = "d0fbd279f5338d8e4544354ccb1bfb28b13c53ea"
POLLETTA_BASE_URL = (
    "https://raw.githubusercontent.com/gcalderone/qsfit/"
    f"{POLLETTA_COMMIT}/IDL/qsfit/components/swire"
)
POLLETTA_FILE_SHA256 = {
    "Readme":                   "7de63bd811d99aad07e8852013c624f1e42ef15cb1df69c9ffed7ecd86b24f90",
    "Ell2_template_norm.sed":   "a8f278c7a4a8689507dc203c1947fe11f2c27877a58cf41d3d0a3fecab4caa0e",
    "Ell5_template_norm.sed":   "f984b8d9cd9d679fc2043dfa084662904792ca7ddf81f24403b9f234635984d3",
    "Ell13_template_norm.sed":  "fb0f39665cef012cc5a9a763132c42358b19a18e69a9b40a3b6a5f83bdb72ba3",
    "S0_template_norm.sed":     "3388dc5d26d7621a0547f476491871296734cfe8a3de8e8b45c40512f9795456",
    "Sa_template_norm.sed":     "d90e3e72a3c5c6f16fab112e0c7926fb3aaae14890da4a38a9bb325d1ac6b763",
    "Sb_template_norm.sed":     "baa0e942b4152bba48d8479ccb5f8d6bbbe9964d7b1a640a587e9e0d3835299e",
    "Sc_template_norm.sed":     "78fe39e899d9a9c66f3b0656d803196867854bbb390c05add11b7793839758ef",
    "Sd_template_norm.sed":     "7ec1fae3410124aab410f88079a7913fe5cc32acd56120b53db0aa22e80a87fb",
    "Sdm_template_norm.sed":    "c81a920540071adcc5e138a0189fdb8f1fe83608905af0a0f6db02f078b25028",
    "Spi4_template_norm.sed":   "887e6370fdbec031401247399ca83680ee6c967217a254102574c3422a729559",
    "M82_template_norm.sed":    "42baf673f262a4133c297b88c21fd5e248acbcf30f5afe52e4e6b928452d8b16",
    "N6090_template_norm.sed":  "5bd80ed367680d21a849d24a5ea1a4256bb4c92fefcf7225fd20e7118f259b85",
    "N6240_template_norm.sed":  "1470e62ea8c1a0ddb3622c59c734c5aff18c6c4e90f40620b25f77ca0ac0235d",
    "Arp220_template_norm.sed": "45408ba08722ecbcf4078ae9681a4bc3b35daf1e7d2d62d1f2ee003ec6ec08cf",
    "I22491_template_norm.sed": "345a4e19cdad9f04ed0611092b21553cc342d798d566810c539daddd793b2e7a",
    "I20551_template_norm.sed": "0cf8be66aec10fa07c1bdaedea94bc462bf8252167d71e7a13fc728156fb0f1f",
    "Sey18_template_norm.sed":  "a22245ed3361ae4e3955b8084cca400eae87f35cb3352641caf58b7e36f335a0",
    "Sey2_template_norm.sed":   "86c168f848b202dc83ffb209b7d99377f288d563e568785dac6fc08a0f12220f",
    "QSO1_template_norm.sed":   "66236fdd5ed76526b8ea461cbfdfc3a8ed1ac37c16d6e95da4626e7b537125fd",
    "TQSO1_template_norm.sed":  "36bbce81dd0b434201b7d17ace55a563fc8c555d1e1f488716a624545f0b7bcb",
    "BQSO1_template_norm.sed":  "0f149de1a0dfba60c1a6cdcddf122b104887ae762d2e7e6ec154a074a41dc3f1",
    "QSO2_template_norm.sed":   "3ff720093985037a94e9c9a7fcf64fc8dc9fa7bea69f38e22488f140056ee836",
    "Torus_template_norm.sed":  "7af3b1c66a1f8c45538e44d202b57dc573bc0741cc96db8f5a032a46a51677f1",
    "Mrk231_template_norm.sed": "5923ac9843b74375bdda319a8ab747c9cf90de2e82e3af3cc9dda1914513d959",
    "I19254_template_norm.sed": "419713cf5ca969476589003b49363118a2246526d8177d01fe6a545c4a346f14",
}


def fetch_polletta2007_swire_templates(download_dir):
    """Stage 0, Polletta: Readme + the 25 *_template_norm.sed files into
    download_dir/polletta2007_swire/, each verified against
    POLLETTA_FILE_SHA256. Idempotent -- a file that already hashes correctly
    is not re-downloaded. Returns the source subdirectory."""
    out_dir = os.path.join(download_dir, "polletta2007_swire")
    for name, sha256 in POLLETTA_FILE_SHA256.items():
        model_io.fetch_pinned(f"{POLLETTA_BASE_URL}/{name}", sha256,
                              os.path.join(out_dir, name))
    return out_dir


# --- Berta 2013 ------------------------------------------------------
# Berta et al. 2013, A&A 551, A100, "Panchromatic spectral energy
# distributions of Herschel sources". CDS/VizieR catalog J/A+A/551/A100;
# freely available via CDS for scientific use, cite the paper.
# The shell script crawled the catalog's Apache autoindex and mirrored
# everything. This manifest is deliberately NARROWER: the 32 temp1p6
# templates the loader reads, plus the two files that are the authority for
# how to read them -- ReadMe (units: "0.1 nm", i.e. exactly 1 Angstrom) and
# list.dat (the descriptive names the class keyword rules key off). The
# tempLIR/tempM normalizations are the same shapes up to a per-template
# constant and are not fetched. download_dir therefore no longer mirrors
# the whole CDS catalog -- see galaxy_curation_plan.md section 2.2.
BERTA_BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/551/A100"
BERTA_FILE_SHA256 = {
    "ReadMe":                           "57fb37603097717ffa57fb9d7182e68d7cead820081461ed7f52fa79a41a51f6",
    "list.dat":                         "7326b8b288317a5b3558f571b30931cc1af7b99f581d8ecedeb8b8ac8adfeb95",
    "temp1p6/Blue_SF_glx.norm_1p6":     "c70ea253edc6b669e28a311e090752af1d0454eefc2f429ce7adad0073c4d630",
    "temp1p6/BroadFIR_SF_glx.norm_1p6": "1da324b6a3b80f89095362cd62a0d0e500f4aedf6810d9e8753500bef1201ce4",
    "temp1p6/Cold_glx.norm_1p6":        "53ab0e7d19392a4a1c37e80768f8b536c75f752933df7d50ce68aada38e06a22",
    "temp1p6/Elliptical.norm_1p6":      "b8cfbe0b88dbeed4ad17d61bddb5e395bf04198dbc1a495893732333f5a52cdb",
    "temp1p6/Ly_break.norm_1p6":        "40c341f7f2f2ecbacd9bac4f82cf93e7d1062debb7bc9b357128136fd2deb3b3",
    "temp1p6/MIR_powlaw_SF_glx.norm_1p6": "9572d251271dea12a73ea874ab4e87a5fc2420f85e88e51f7e299f26498a75dd",
    "temp1p6/MIRex_SF_glx.norm_1p6":    "67083c425b4c3c4086f8b1010ee1a500b748959b003e1c45399a87d9a3fab109",
    "temp1p6/Mod_SF_glx.norm_1p6":      "8fbdf47d3d4ce32f02e6611a27e5bc4290c69cd37bde3e00b6da3eb274204d58",
    "temp1p6/Obs_SF_glx.norm_1p6":      "9d6d06300368b4b058a251ccab3019fadb47d2af1125991d489a671f6baae312",
    "temp1p6/PAH_SF_glx.norm_1p6":      "61af6b394f1657961d8da0a7855c4453b88329ef425571cbd5d66a106403c0d7",
    "temp1p6/Red_SF_glx_1.norm_1p6":    "42560ed71c14dd1b9c0c714fa5a4bddfa59c2a3634347c0c126d8535a1ebea7f",
    "temp1p6/Red_SF_glx_2.norm_1p6":    "66be9004c1a820cf8ed5d65fd19a95f7c6acff6f2b7374f41f9b6aff0e0a5efa",
    "temp1p6/SF_Type1_AGN_1.norm_1p6":  "b80895a40470ddbad377a728d24d68035a40ddacf11b6889ab5ccbb5bca5eb7f",
    "temp1p6/SF_Type1_AGN_2.norm_1p6":  "b93517ab130efba0f38ebf2404c9ece09b1d9392e75bd7c02e0bd5b22930e404",
    "temp1p6/SF_Type1_AGN_3.norm_1p6":  "416ad36a6f5f8fd950aba70be5fefaa2b17cf937bfd612f6ea28f2d389a37870",
    "temp1p6/SF_Type1_AGN_4.norm_1p6":  "c824af21ad3122f1d57f68ebb57dd220482ce0ccf5b9d6d74e9caa805728ea54",
    "temp1p6/SF_Type2_AGN_1.norm_1p6":  "9975633d67ab846ab69e6ded92c4ae69016aa0911d1673ef884b0b55555308aa",
    "temp1p6/SF_Type2_AGN_2.norm_1p6":  "67987190da5d922166e82eb331b39bacca2d16ca22d111f2564e25739a5ee1c1",
    "temp1p6/SF_Type2_AGN_3.norm_1p6":  "b564e372c978ce9424c1075d5b4abf158e182a30431d80d0156f484552dcf75a",
    "temp1p6/SF_glx_1.norm_1p6":        "f856a1e2e7eecf0f2950a3dd29e5b9742db92de76480afdeea4ee4a0ed113622",
    "temp1p6/SF_glx_2.norm_1p6":        "afa04a042c0db9f318df825b2c3e33a1fe395c6e48d853fcae1bcf1e648528cd",
    "temp1p6/Secular_glx.norm_1p6":     "a52ee7783fcca556c9b40d3d9f8ec0ee248954d4add80113222a619afb6a90c7",
    "temp1p6/Si_break.norm_1p6":        "82f2133b8dc2a709eb20611140002b8a7b60383a3c72ca766620387cd7641d39",
    "temp1p6/Spiral.norm_1p6":          "9c5591c5e1b1f31b80ede237df053207b1bb5f94a93499d24afbb690931e5bc2",
    "temp1p6/Torus.norm_1p6":           "0e0d19be1e68b240a7b22e2284dd2fe4a7750a32811e22bf4e24d98933f5a6c8",
    "temp1p6/Type1_AGN_1.norm_1p6":     "00fd9bc0ece7ca134ff7d9d1e8a46ff56cc554bb303762a03422f55ebf7d68c7",
    "temp1p6/Type2_AGN_1.norm_1p6":     "500b99dfc59c8bca826eb2c032df4fdad5c6584dc1d87f3912ea638787fd1926",
    "temp1p6/Type2_AGN_2.norm_1p6":     "febc12c7693f4c012722f72d638a65873ea803e39ea6fad263b51dd7cde6b854",
    "temp1p6/Warm_SF_glx.norm_1p6":     "5917c8638dad16c49f33284b379d3cfa31c5698a3a4427f01e9dd70460799816",
    "temp1p6/WeakPAH_SF_glx_1.norm_1p6": "d4759377d87a32463f319debe67307703360b94d8432a65adf2cb538eaf71306",
    "temp1p6/WeakPAH_SF_glx_2.norm_1p6": "d77035d3c4b2d88835d29e69d0b1b66235c660ce2de65f23dd238da60a6288b9",
    "temp1p6/Young_SF_glx.norm_1p6":    "5c69aa615a0b0ac5775f7608320705dbd8ae6770f80b59a5adf0fd46931f1db2",
}


def fetch_berta2013_templates(download_dir):
    """Stage 0, Berta: ReadMe + list.dat + the 32 temp1p6 templates into
    download_dir/berta2013_templates/, each verified against
    BERTA_FILE_SHA256. An explicit manifest, not a directory crawl, so the
    file set is a build input rather than whatever the CDS index says
    today. Returns the source subdirectory."""
    out_dir = os.path.join(download_dir, "berta2013_templates")
    for rel_path, sha256 in BERTA_FILE_SHA256.items():
        model_io.fetch_pinned(f"{BERTA_BASE_URL}/{rel_path}", sha256,
                              os.path.join(out_dir, rel_path))
    return out_dir


# --- Brown 2014 (GALSEDATLAS) ----------------------------------------
# Brown et al. 2014, ApJS 212, 18. MAST HLSP, DOI 10.17909/t9-5bxk-dh29,
# licence CC BY 4.0, host https://archive.stsci.edu/hlsp/galsedatlas.
# COUNT NOTE, carried over from the shell script: MAST states 129 galaxies;
# the per-galaxy table lists 128 individual *_spec.dat links, which is what
# the manifest below holds and what the loader reads. The combined catalog
# FITS is the provider's completeness reference and is fetched alongside
# the four summary CSVs, though only the summary CSV is parsed (Stage 4).
BROWN_BASE_URL = "https://archive.stsci.edu/hlsps/galsedatlas"
BROWN_SPECTRUM_TEMPLATE = "hlsp_galsedatlas_multi_multi_{slug}_multi_v1_spec.dat"
# The combined catalog sits at the HLSP root; everything else under atlas/.
BROWN_ROOT_LEVEL_CATALOG = "hlsp_galsedatlas_multi_multi_all_multi_v1_cat.fits"
BROWN_CATALOG_SHA256 = {
    "hlsp_galsedatlas_multi_multi_all_multi_v1_cat.fits":
        "f36201f293d97cd0e780bc42491868ac7151a5ba50f7feb2aa1bdee4be4e05e8",
    "hlsp_galsedatlas_multi_multi_summary_multi_v1_cat.csv":
        "56525acb9da0eef1d3dc9cc38cdf9b8b495bb4ef8c848bb91a1037b398acf01f",
    "hlsp_galsedatlas_multi_multi_photometry_multi_v1_cat.csv":
        "624fa7f2238a7002a6fbf73c403998366e2729b8506259c218afcc183204eb94",
    "hlsp_galsedatlas_multi_multi_photometry-minus-a_multi_v1_cat.csv":
        "1a03cd1a3c19806b3c4075472c75311516748942c99d2178e00a05ece7ff04cb",
    "hlsp_galsedatlas_multi_multi_extinction_multi_v1_cat.csv":
        "6f06314ba088232a3b5c504b2e970a6a2bea41d62797d8ebf266509b1dd8b073",
}
BROWN_SPECTRUM_SHA256 = {
    "arp-118":      "03cd4b27ff85dca281d3d7f85321dcac0e6a82c2686a015b62494953305e475d",
    "arp-256-n":    "c44e17638abb962a38554471bfbbf921bd07bfc45e360bde21efddf5bfa0fec0",
    "arp-256-s":    "47928b20689e429d562e45abb417f02e2d63a9e69cc58d42fabf81d941b2e665",
    "cgcg-049-057": "2421d551c8533176964f7eab09a49625494368746d925a8b40105be7336902a5",
    "cgcg-436-030": "82c26866f67f8baff471b92d5a6d0f59dad62a6ed7212c302d3f5e86257d3efd",
    "cgcg-453-062": "3154cfa018441d3aa0588d8d7fb776a94c57e29a5c036c14cecfe601838f9a40",
    "haro-06":      "d65ab8c518b858a8a21a83c6ea9559d5804dbf45b49a509624e3c7a4aba2d301",
    "ic-0691":      "6009740c5126a38d80e303d6ceffed77723155cc316594858483e055cede0c21",
    "ic-0860":      "9a7b8139ad77ba512c6317d58963550a45de03b3f8c895f8457dc15006d9b5c1",
    "ic-0883":      "78fd8346cb8d10ee06c4a2c7a1b8d60c9d5a4204d24ee3ac888adffe9c80f862",
    "ic-4051":      "ab32cf8411ed2ac2a6a27b17bfa6638f97e066e4565aeb3f19861420f6a54696",
    "ic-4553":      "a5908ef107acba6187c47d87525e630f0075d8027324ee42bc492bc5c5d957c3",
    "ii-zw-096":    "02d4139657a6315e68aafc01a09ce4030e7674ff35bb061085586ad04823b201",
    "iii-zw-035":   "dc5e294768afe5f1ebb379619fbbf15165498df36790c214be8ac6db4d9a39c4",
    "iras-08572+3915": "ecb95276b0795f9b73dec48999882117b289b49ea6d6180d61688a99954e0fc1",
    "iras-17208-0014": "08d0dcb1e42e77e13c0f6b320b0148f4e0110876e3fe93319e6e3fcae08be05c",
    "mrk-0331":     "0d794950485406f583eb596a48cb20f3fb352bdd750836b53f9160ee72eb2408",
    "mrk-0475":     "080f513e65c5c04c75fa366f484a927c73f7ab2214f942d62394e1016be5cea0",
    "mrk-0930":     "36d3c066738fde1f42d93cab810af675c578f19aa2bdf840111293b57067c9c2",
    "mrk-1450":     "ba2f463d38ea88a5b11aebc95d66272cc519d99903f92890c6c5d7095b652c58",
    "mrk-1490":     "52c63288f74fbe756d3d3cabad27ce4b38f0b18e97404483bd04de543e6f2b1c",
    "mrk-33":       "8a8d4751e804831cea06cccb05c6bc255aa2afa5d7b2639d95ce3144900d3755",
    "ngc-0337":     "00c10ce45a4417418e71909adc16755edb6c1593b1b451c8602f7a2d92fdebc8",
    "ngc-0474":     "19841a899185a66bec87e918204ca9deab53d6937cd9e0c57cb8a51ab0ef6b52",
    "ngc-0520":     "490504c4fbc3fc84978ce43bc5759be444d10fe0cfbeae850088b47bc764592a",
    "ngc-0584":     "88197ae6373a8cbc6946ed376b897bd71e3b5a2d39d443568f5ef18813586482",
    "ngc-0628":     "e2facdc2aad0414932f3f5224c0fcd0e9b34d9b346e3e9389472556a6c1152ee",
    "ngc-0660":     "9bfa39f04157ffbdb2d799ce8e82f55c8e0f8b0cbe2e619f0a5f54b71880d851",
    "ngc-0695":     "75ff1ecd21dc90bd179039c7085c4a4226efa63566cc35987ce0ebca3320f529",
    "ngc-0750":     "27d5b65466f13c62f8ca66b3504a4b32e3b68cd377c1f8ea208261078cbc99d4",
    "ngc-0855":     "8faeb8cc888e157d8b0fa00c61d190832f95acd71e8a6a0b78151069729fae7c",
    "ngc-1068":     "922dc99167e38e5623d07726057b5198001f81e4208ef22f27a0ae7978b4ab36",
    "ngc-1144":     "d9cfa0ce93755da2999abe31364efc74558e560374cca67dbf2e05b9c334048e",
    "ngc-1275":     "75d342448ad16aa4dcb436274a7971115ee1cf5882543092c5567725e53e989e",
    "ngc-1614":     "010e61ecbab94ee0631aa5e8ce8008aa000524f881bb4b91816594e65d5b2230",
    "ngc-2388":     "1f8ad8175893058fd709a2327fd2c882ebbd58a8bd20de5027a44106bea90057",
    "ngc-2403":     "a5acd1ebe990f7f95806a1b4ab759e8011dc7cd586112265150596114de110e0",
    "ngc-2537":     "29f008ed2abc88c818f1f1bf8139910714bc0096bc7c02c45147ad5f69522db1",
    "ngc-2623":     "75f1d31cd0eb937223e8d84b2e3b4b2e4d058afae94924fef986198e2a81da94",
    "ngc-2798":     "646d397e0dac416030db3499f4704051cd064e87d900b37f7ac4fb7711bfa1eb",
    "ngc-3049":     "ae7d160d4d763b5a504e711a82520c3920f5ee1770dc40879704c1d300283317",
    "ngc-3079":     "8b38a9e22675c4bfc756314f15f47446645918eb4a80e73de8a2fcd96e44e8c8",
    "ngc-3190":     "c6f8032debf1a196224a9209dd99590bb983e7151be81894502b7322473a8223",
    "ngc-3198":     "0445324e95f953c29729a0d881fb2b767948c3292ddfd92e95f182826f286096",
    "ngc-3265":     "57b5ea4b104f33fbd8f51416d83a4db9c7de682609b4373666c88eb4d2afcd17",
    "ngc-3310":     "6953ab29bd7f229c097d19f89e057f846979dfe7bbbb678eb7731d230234505c",
    "ngc-3351":     "bdceead0c4bb66a65d1c5793a3cf7401023e22b42bf31f823699f92c200df3e0",
    "ngc-3379":     "80edad7bc9cd41eea8d878cf15496c3e13f198ca7e0a57806ca771df157cd6a1",
    "ngc-3521":     "7225e4140e5ce5e6c073dfe5dfa8f1eeed63dfbec96ce049f1123f92d4085a3c",
    "ngc-3627":     "a9183cd04ea65f2a16e7c32595c7cc832018e110441da7a814a5f3403db24643",
    "ngc-3690":     "d09450467d628c7fd347d99e9138b3b63e60171a56cc4a1146ec4c4ba939877d",
    "ngc-3773":     "01a3bab33d2fd81457b722c17580300e069783a282c45c98862d1702c67262b5",
    "ngc-3870":     "59c8787e1d796fd0e5a7482ea63016e53ab306fd7d84ab81687abf8dde68fdd1",
    "ngc-3938":     "57d1c91588d8c00e3c55fa6c20866d3aa927ca977b72cda9e22aeddcacd1b153",
    "ngc-4088":     "f111881985020b4e828193f4d9c24bb837d7c4d52b1076044f0799a16050e280",
    "ngc-4125":     "caf443f5c12210c756106d63cf9fb379392142cf77f5b80c6287ac66ca571261",
    "ngc-4138":     "dc3f5ee3fdc03e0e79c3f31f366bfe57d1a77293a6696633ecdfe257b95135f4",
    "ngc-4168":     "a6942a66b5d44d6bb106dba005770b4a7359f998ec566f1e8570f388b24c4cf5",
    "ngc-4194":     "762643d6c30d526052bdccf869ac5dcf0b864d29424775a08f2179f8645c9826",
    "ngc-4254":     "ec861180fd3ac3e727397c564e7ac860fab4cc47e6dbdd791020051cf017d70a",
    "ngc-4321":     "608b485b6408851c5035b6173ad94e57c05133331d331c4cb0f7c20d1f7df01d",
    "ngc-4365":     "0403fe4da79272e2ba57da8cad7a68d8fdf299cc82277f09005ba48047bd85b7",
    "ngc-4385":     "32287bc7bb2c5d3711f9800d627408f75a5db0041cfb7d9cba29776250805f01",
    "ngc-4387":     "c3bf4d333f82d48904fe92d72d05034f52fd25c14f2741b41f2fa40911621f82",
    "ngc-4450":     "3e70d1ead20cb0b06100b006071002b03090227234a946d2b64654148b1f1f46",
    "ngc-4458":     "861b2fd2b31bf05ca90673d945bf2609f10a2bf818157da8c3578ca055b6f9db",
    "ngc-4473":     "06d46a4530d31aa08e1baaab8572f0d805c326e7b71fc93c3c4312868cb2ed0c",
    "ngc-4486":     "e724614dda3fc3ffc1e416940e1495661b3d6365a12adc89c2743d17e486d900",
    "ngc-4536":     "b9c5bcc47e2901ff4377fdd4c8a35169481701bf4ce35f7f2fd21a87f3894b43",
    "ngc-4550":     "07424462881483a99d4c877b6bb998e2159557311050f0723a9bd015022c4caf",
    "ngc-4551":     "526447335c9ba3feabbf98c41de3b43e52107cee13439c4b0dc1a477d11bf33b",
    "ngc-4552":     "99058c2e1f2c238440ea3383dfea89748a481a51580a686b7d611bb5f41a0bb0",
    "ngc-4559":     "998f8043e9d355d573939e192259bc1bbd155aaa1e59a97b3b30f5ee8f2f312d",
    "ngc-4569":     "96140e924a960af7aa1b1abad4190a2a284c7f62292515d98b7b95d94499e2c5",
    "ngc-4579":     "b015ce130adaf43b754142d318ad9340a9a321f12308debe0d2bae2d8cf1f3ff",
    "ngc-4594":     "4ee817a9e9e483cec7fc879dcf8931a7cbd16cc3c3ab68cab52bca3d639e6c5b",
    "ngc-4621":     "0d20e0a0e16fb4f7a6910035f534d8c4a7eed51c533eb9316f2ab80adc9f3e2c",
    "ngc-4625":     "524e7ed146f9912bc8e7284e2cfd805583cf85093182d9d0f7e4f96032748587",
    "ngc-4631":     "ff32fce48dd40919e9b35af1fdf3c557c098c583d5d7068490d64056c5ee4109",
    "ngc-4660":     "0459f2504ff001f59e7877a895d0034560fc4f8b5e2e744b70cb1fcea9ce5de6",
    "ngc-4670":     "1e4a2a1f7e8caa243565871e01a1564d8e74f7a78594ae6daa3ffdcb060a642f",
    "ngc-4676-a":   "7e295d105d292153298f1507a80730f91f965a0cf616bdd7746f3bc196336b3d",
    "ngc-4725":     "d2f52f642aad6ab9b0d41a09370f5aa9def809c3a242194a9013f4268f4f0f81",
    "ngc-4826":     "72e86a572e3292d804684ce3bb65ae76b769cbcc1aa861714593f7726bd11718",
    "ngc-4860":     "2ff08d3d2f8c4583e25afb8a2fc7e0f4de2315a6a29793ad2da5c9109b329451",
    "ngc-4889":     "d289a5c4ae1d9f70ddd6c0c6c2f39adaf6947eaa8587d468c4654a9f9445a077",
    "ngc-4926":     "c771816f4375dab0b0e7f8bf6ee213178d2a456cec66409ae8c639864baeedce",
    "ngc-5033":     "a378d3e3aa84debee3c3729d4d8e194553dc25f3b3c865905ca94c5a8e8caaee",
    "ngc-5055":     "5de1978a3b8ace6a60a74be47c51335ebebe7607ac25a06fdce866d734c2e642",
    "ngc-5104":     "e5b462e454ff7d874d68b2f9480709ba8e17d44991e2f4c257d8f5a5726cf710",
    "ngc-5194":     "5a6be0504821c0e23bd1143e688627b99e476016708e338a0e06aa0b8efed7fa",
    "ngc-5195":     "0169780f17796d9b5052c96e14c433204af3f8425f648727115b754fc7c016db",
    "ngc-5256":     "306273baa4303f65abe44a3bb2d2061159599bb5a202ee5adbb3061cb8d3ef93",
    "ngc-5257":     "6494272c132f379409f854fe0f25d3fda4b952c8f47a9cdb97938ffee0d79aa4",
    "ngc-5258":     "455f42e76b9c451248ad6f2fb89846a8dc615861f1b5d5423de9531c26386050",
    "ngc-5653":     "a85327887a3976681937baabe6cd98401b1a2b05e9bd61c31153caf86c157aec",
    "ngc-5713":     "8e152053a137ea33dedd13bc3be1172d546a9d7366a8f120ff2d32d94e8ee67e",
    "ngc-5866":     "921bed73d6f4e6fe87a110fa7d3c2432bec93e6ba0dc58bfb65cb00fcb688175",
    "ngc-5953":     "4b29ce9b8bb810706518b57cf60c048fc121fba0b748f903a177b889397e5914",
    "ngc-5992":     "df15cc5803c0061fac35522610ff0ccb0a5522d6e51bf9004e277a8af9aec69e",
    "ngc-6052":     "68d08917e8ee3bafb8217a4480f2551ea7114972fcbdd719fc0b28a4937a0d2d",
    "ngc-6090":     "d7341ecff653eb20cf3489169f554469548d487165f74f62fd84878e3b094fd9",
    "ngc-6240":     "e98380ab65f600d6d4fc76ad24136385813411fb7fba9e78061adba53444e783",
    "ngc-7331":     "1d704e751e89eb33e9cd2bc07e229c8daa0f477f2177dd645c1b096078b3c3e2",
    "ngc-7585":     "d260493c5e3637bf5eb61dbebf3fb386a3e9b7a9cf67212e9299b8141304b900",
    "ngc-7591":     "ff117792ad4c5e394ec7a8e6362d0d5695939456a2becdebb5e992f310f3d553",
    "ngc-7592":     "dc197ac026d59e352243d40a914f9df55730a6b34b66168b2b4f8d90e24958c7",
    "ngc-7673":     "64845a60aed9c95bf9a10f96cd00d27ec60466412640084a8690ed68e6e8b222",
    "ngc-7674":     "dd401893b75b2f2bd0694b2528f2c94f83e784980b70938a3bed4185335c0996",
    "ngc-7679":     "2cf1639bdeb75150ea60676317adf780ecba47ad52d6aa5ef182cfa31048fa8c",
    "ngc-7714":     "501a20825fd7111a84a8c2af5251aa3ddb3a0950adf1a34d817485407d6249ee",
    "ngc-7771":     "df967950c3262a0273f74f32f3b85d3e57f1e10ca04a1b983c0e70698cf58a7e",
    "ugc-04881":    "94cc147730a65f35eb0ca8b9ba9410f8701f52bc92e01f8b2a2fdea77713760f",
    "ugc-05101":    "031344512ae6d48f6466dd2500edef9ca01783515e433d7d3eb02691d3e42668",
    "ugc-06665":    "de17d8400f8ba8eff051976ba5641be59ab503600cf3857c7571e08638c7e34e",
    "ugc-06850":    "669fd3f995ce63b0d64b8204a7a40e5bd33a60a1dbc66e3f242e1f9ede46252a",
    "ugc-08335-nw": "1bf2fc5f207ce0004ff185e3aa6db41af2b0360c188c82dcfd0abdb47e632adc",
    "ugc-08335-se": "69c26ec2fca228dab086c98da88ac9665722fde3b05c261dbaae52febe2d468d",
    "ugc-08696":    "6553f8d0d9bcb49a145b5efc4e14e9460d0d5dc95b4210ecc556fe26e3aa0a14",
    "ugc-09618":    "d2ac815f0675f8bf3da9262ec7f66206edd463c25ef4f69f94d87bb453fc239a",
    "ugc-09618-n":  "1cd81eeafba2bae9d41df2dc0b07586e6da14b148d653ad71288eb3cb04c9243",
    "ugc-09618-s":  "a726b074ab03325450700750edf5be8e96cf4d3f0d34dba86ffe1fd38011c621",
    "ugc-12150":    "da4249ad07c7f85cec0ec32040f9176775650560e74352ce2e6ff544a0a14487",
    "ugca-166":     "4882cd8476cfef1ea16698313290d6baffaedd02e628c2b2b4c9f1f9c5ea32f4",
    "ugca-208":     "7083a610b80b6e6d8a0aeefeec57a3fe6d8cccc76a7072d9485dcb7136258902",
    "ugca-219":     "2e3e0f21008213bb98cd7f610bb3ff43ed211bd3f2977fce8b3e1dcda99d01c0",
    "ugca-410":     "43ce52c24ffd242515b61a7e62ef592ba2d422c7460e0f4248ee20d77970923b",
    "um-461":       "77fa665d96b771875950f3292caba5236d34b1e5a7324656e783a9db2c1997f1",
}


def fetch_brown2014_galsedatlas(download_dir):
    """Stage 0, Brown: the 5 catalog files into
    download_dir/brown2014_galsedatlas/catalogs/ and the 128 per-galaxy
    *_spec.dat into .../spectra/, each verified against
    BROWN_CATALOG_SHA256 / BROWN_SPECTRUM_SHA256. Returns the source
    subdirectory."""
    out_dir = os.path.join(download_dir, "brown2014_galsedatlas")
    for name, sha256 in BROWN_CATALOG_SHA256.items():
        remote = name if name == BROWN_ROOT_LEVEL_CATALOG else f"atlas/{name}"
        model_io.fetch_pinned(f"{BROWN_BASE_URL}/{remote}", sha256,
                              os.path.join(out_dir, "catalogs", name))
    for slug, sha256 in BROWN_SPECTRUM_SHA256.items():
        name = BROWN_SPECTRUM_TEMPLATE.format(slug=slug)
        model_io.fetch_pinned(f"{BROWN_BASE_URL}/atlas/{name}", sha256,
                              os.path.join(out_dir, "spectra", name))
    return out_dir


def fetch_all_galaxy_sources(download_dir):
    """Stage 0: all three source trees, pinned and checksummed. Returns
    {source_tag: source_directory}."""
    return {
        "pol07": fetch_polletta2007_swire_templates(download_dir),
        "ber13": fetch_berta2013_templates(download_dir),
        "bro14": fetch_brown2014_galsedatlas(download_dir),
    }


# ====================================================================
# Stage 1: redshift grid + per-source rest-frame template loading
# ====================================================================

def build_redshift_grid(z_max=6.0, n_z=101):
    """Uniform in log10(1+z) from z=0 to z_max -- denser at low z (where most
    realistic contaminants fall) than a linear-in-z grid would be, matching
    standard photo-z-code convention (constant fractional resolution).

    CALLED ONCE PER SOURCE, with THAT source's own
    GALAXY_SOURCE_REDSHIFT_CAP as z_max and the shared RAW_GRID_N_Z as
    n_z (build/galz.py Stage 1) -- not once, globally, at a fixed
    z_max=6.0. A single shared grid cut down to each source's cap after
    the fact spends its resolution on whichever source's cap is
    largest (Polletta/Berta) and starves the narrowest one: at the old
    z=0-6, 101-point grid, Brown's z<0.06 cap kept only 3 nodes (0,
    0.02, 0.04), nowhere near its own reddest member (IRAS 08572+3915,
    z=0.0583). Calling this function separately per source, each with
    its OWN z_max, gives every source the SAME fractional resolution
    (same n_z, same functional form -- no per-set step is hand-tuned)
    over its OWN full reach, so a narrow-reach source is not penalised
    for another source's wider one."""
    z_grid = 10 ** np.linspace(0, np.log10(1 + z_max), n_z) - 1
    z_grid[0] = 0.0
    z_grid[-1] = z_max
    return z_grid


# --- per-source-tag redshift reach (CURATION_2026-10-09.md sec.3; review
# lib_3_GALZ.md concern 4) ---------------------------------------------
#
# The library used to redshift all three sources' templates out to the one
# shared z_max=6.0 grid above, uniformly -- "coverage insurance" for
# templates with no published support out there. Each set's PUBLISHED
# redshift reach is narrower, and the library should stop offering a
# (template, z) row the fitter cannot trust past that reach.
#
# These are declared as REACH CAPS applied to the one shared
# build_redshift_grid(6.0, 101) grid at assembly time
# (prepare_combined_model_arrays keeps only each source's rows with
# REDSHIFT < its cap) -- not as a smaller z_max passed back into
# build_redshift_grid -- so a cap always lands strictly inside the
# published reach (the nearest grid node below it), never past it, and
# the per-step resolution stays identical to the one grid every row was
# already generated on.
POLLETTA2007_Z_MAX = 2.0
# Polletta et al. 2007 (ApJ 663, 81) built/validated these SWIRE archetype
# SEDs against SWIRE/SDSS sources out to z ~ 2 (lib_3_GALZ.md sec.2 item 2;
# the curation brief's own "Polletta 2007 z < 2").

BERTA2013_Z_MAX = 3.0
# Berta et al. 2013 (A&A 551, A100) fitted the 32 temp1p6 template shapes to
# Herschel PEP/HerMES sources out to z ~ 3 (lib_3_GALZ.md sec.2 item 2; the
# curation brief's own "Berta 2013 z < 3").

BROWN2014_Z_MAX = 0.06
# Brown et al. 2014 (ApJS 212, 18) is a NEARBY-galaxy atlas: every one of
# its 128 shipped spectra is a real galaxy's own SED, already
# de-redshifted to rest frame by the data provider
# (load_brown_rest_frame_templates), then redshifted by THIS pipeline a
# second time with no change to PAH strength, dust temperature or
# metallicity (lib_3_GALZ.md sec.2 item 2: "no evolution"). "Local" is not
# a free choice of number: the atlas's own summary catalog
# (hlsp_galsedatlas_multi_multi_summary_multi_v1_cat.csv, cz column) gives
# each of the 129 galaxies' OWN observed recession redshift, measured
# cz = -235 to +17477 km/s, i.e. -0.0008 <= z <= 0.0583 (the reddest member
# is IRAS 08572+3915, itself a nearby ULIRG, not a cosmological source).
# 0.06 rounds that measured maximum up by about one part in 20 so the cap
# does not clip the atlas's own reddest galaxy -- the templates are never
# offered to the fitter past the redshift of the real nearby galaxies they
# were built from.

GALAXY_SOURCE_REDSHIFT_CAP = {
    "pol07": POLLETTA2007_Z_MAX,
    "ber13": BERTA2013_Z_MAX,
    "bro14": BROWN2014_Z_MAX,
}

SOURCE_REST_ARCHIVE = {
    "pol07": "polletta2007_rest.npz",
    "ber13": "berta2013_rest.npz",
    "bro14": "brown2014_rest.npz",
}

# MODEL_NAME-ONLY alias for one Berta 2013 template whose name is too long
# for sedfitter's own convolved-flux writer. This project's MODEL_NAME_
# FORMAT is 34 characters, but sedfitter.convolved_fluxes.ConvolvedFluxes.
# write() hard-codes `self.model_names.astype('S30')` -- a FIXED 30-
# character ceiling, narrower than this library's own format, that
# TRUNCATES silently (no error) rather than raising. "MIR_powlaw_SF_glx"
# (17 characters) makes "{tag}.{name}.z{redshift:.4f}" 31 characters at
# EVERY redshift in this library (the redshift suffix is always exactly 7
# characters within this library's z caps, so this is not a redshift-
# dependent edge case) -- one over sedfitter's ceiling, where
# build_convolved_bands's own post-write order check (verify_convolved_
# band_fits) catches the resulting corruption as a MODEL_NAME mismatch
# rather than letting it ship silently truncated.
#
# Applied ONLY at MODEL_NAME-string construction, in
# prepare_combined_model_arrays, below. TEMPLATE_NAME (info.fits),
# the classification-CSV lookup key and parent_lookup's key all keep the
# template's real, full name -- MODEL_NAME is never reverse-parsed
# anywhere in this module, so this alias cannot silently corrupt any
# other column. No other template name (across all 185, every source) is
# within one character of the 30-character ceiling.
MODEL_NAME_TEMPLATE_ALIAS = {
    ("ber13", "MIR_powlaw_SF_glx"): "MIR_powlaw_SFglx",  # 17 -> 16 chars
}

# RAW_GRID_N_Z: the dense per-source redshift-grid point count (item 2 of
# the owner's library specification). ONE constant shared by all three
# sources -- only each source's own z_max (GALAXY_SOURCE_REDSHIFT_CAP)
# differs, so no per-set step is hand-tuned.
#
# OWNER'S RULING (supersedes an earlier densification-ladder attempt at
# this session, which doubled RAW_GRID_N_Z repeatedly up to 320 looking
# for a kept-count plateau): the grid does not need to be denser than
# build_redshift_grid's own default (101 points, log-uniform in
# log10(1+z)) for the r-net to be the binding constraint, and this is
# measurable DIRECTLY from the existing grid's own photometry, with no
# extra photometry run at all -- see adjacent_redshift_step_diagnostics,
# called from build/galz.py Stage 6, which measures the quotient-space
# distance between ADJACENT raw-grid redshift nodes within each template,
# in SIGEFF units, and reports what fraction of those adjacent steps are
# >= 1 SIGEFF apart (the fraction where the GRID, not the sampler, is
# the binding constraint). At RAW_GRID_N_Z=101: median adjacent-step
# distance and the grid-limited fraction are reported in CURATION_GALZ.md
# -- both small, confirming the sampler decides almost everywhere. A
# previous comment here recorded specific kept counts (905/953 at
# n_z=320/640) that were never actually produced by a completed run
# under the (then-broken) memory-unbounded algorithm; they were wrong and
# have been removed along with the ladder they purported to summarize.
RAW_GRID_N_Z = 101

# Relative (per-row) flux floor applied before log10 in the sampling
# projection, as a fraction of THAT ROW's own largest band flux -- not an
# absolute number, because Polletta/Berta carry arbitrary per-template
# normalizations (module docstring) with no common physical scale against
# Brown's real mJy fluxes, while the quotient-space projection itself is
# invariant to any one row's overall scale (removing the gray/brightness
# direction is exactly removing the row's own mean). The floor therefore
# only ever matters for a row with a literal 0.0 in some band (redshifted
# past a template's native coverage edge -- resample_one_sed_onto_master_
# grid's sentinel), where it sets how extreme that band's log-ratio reads
# relative to the row's own other bands, rather than for any row with full
# 8-band coverage.
SAMPLING_FLOOR_REL = 1e-10


def load_polletta_rest_frame_templates(download_dir):
    """Polletta 2007 SWIRE templates. Native units: Angstrom (col 1).
    Returns {name: (wave_rest_um, flux)}."""
    templates = {}
    pattern = os.path.join(download_dir, "polletta2007_swire", "*_template_norm.sed")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path).replace("_template_norm.sed", "")
        wave_a, flux = np.loadtxt(path, unpack=True)
        templates[name] = (wave_a * ANGSTROM_TO_UM, flux)
    return templates


def load_berta_rest_frame_templates(download_dir):
    """
    Berta 2013 templates. Only temp1p6 (one of three equivalent normalizations
    -- confirmed same shape as tempLIR/tempM, just a per-template constant
    scale factor apart) since amplitude normalization doesn't matter here.

    Native units per the ReadMe: "0.1nm". This is NOT a typo needing a x0.1
    correction -- 0.1 nm equals exactly 1 Angstrom, so the raw column value
    IS the wavelength in Angstrom already. Verified against the BC03 91 A
    blue cutoff (raw value ~91.0) during earlier inspection.

    Returns {name: (wave_rest_um, flux)}.
    """
    templates = {}
    pattern = os.path.join(download_dir, "berta2013_templates", "temp1p6", "*.norm_1p6")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path).replace(".norm_1p6", "")
        wave_a, flux = np.loadtxt(path, unpack=True)  # raw value already == Angstrom
        templates[name] = (wave_a * ANGSTROM_TO_UM, flux)
    return templates


def load_brown_rest_frame_templates(download_dir):
    """
    Brown 2014 GALSEDATLAS. Column 1 is explicitly "Restframe Wavelength
    (Angstrom)" in the file header -- de-redshifting from each galaxy's true
    (small, verified via cz) recession redshift is already done for us by
    the data provider. Column 4 (source flag: 0=model/1=optical/2=Akari/
    3=Spitzer IRS) is preserved as provenance metadata for later filtering.

    Returns {name: (wave_rest_um, flux, source_flag)}.
    """
    templates = {}
    pattern = os.path.join(
        download_dir, "brown2014_galsedatlas", "spectra",
        "hlsp_galsedatlas_multi_multi_*_multi_v1_spec.dat",
    )
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        name = name.replace("hlsp_galsedatlas_multi_multi_", "").replace("_multi_v1_spec.dat", "")
        wave_a, flux, _obs_wave_a, source_flag = np.loadtxt(path, unpack=True)
        templates[name] = (wave_a * ANGSTROM_TO_UM, flux, source_flag)
    return templates


def observed_wavelength_um(wave_rest_um, z):
    """Wavelength stretch only -- no flux dimming applied (see module docstring)."""
    return wave_rest_um * (1.0 + z)


def save_rest_frame_template_archive(out_dir, out_name, templates, z_grid, has_source_flag=False):
    """Write one source's rest-frame templates + the shared z_grid to a
    compressed .npz. Source-agnostic -- callers pass already-loaded
    per-source templates, so no source-specific logic lives here."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {"z_grid": z_grid, "names": np.array(sorted(templates.keys()))}
    for name, vals in templates.items():
        if has_source_flag:
            wave_um, flux, source_flag = vals
            payload[f"{name}__source_flag"] = source_flag
        else:
            wave_um, flux = vals
        payload[f"{name}__wave_rest_um"] = wave_um
        payload[f"{name}__flux"] = flux
    out_path = os.path.join(out_dir, out_name)
    np.savez_compressed(out_path, **payload)
    return out_path


# ====================================================================
# Stage 2: master wavelength grid
# ====================================================================

def build_master_wavelength_grid(wave_min_um=0.3, wave_max_um=35.0, n_wave=3360):
    """Shared master wavelength grid definition, observed-frame, for the
    combined galaxy template library. The red edge of 35um clears MIPS-24's
    full response with margin: the measured 8-band bandpass footprint is
    1.066-32.207um, and 10.1% of MIPS-24's integrated response lies beyond
    26.5um, so a red edge chosen from a nominal band-centre span would have
    truncated it. No coverage is needed past MIPS-24's response. The blue
    edge is extended down to 0.3um to also cover a future Gaia G-band need
    (true blue edge 0.32um; a 0.5um floor would have clipped ~20% of its
    total integrated response).
    Resolution: ~1626 points/decade, close to Berta's native ~2075/decade,
    so extending the blue edge doesn't wash out PAH/silicate feature
    resolution."""
    return np.logspace(np.log10(wave_min_um), np.log10(wave_max_um), n_wave)


# ====================================================================
# Stage 3: resample redshifted SEDs onto the master grid
# ====================================================================

def resample_one_sed_onto_master_grid(wave_rest_um, flux, z, master_wave_um, flux_floor=FLUX_FLOOR):
    """
    Return flux on master_wave_um for this template redshifted to z.

    Method: linear interpolation in log10(wavelength) vs log10(flux) -- the
    standard choice for broadband SEDs spanning many decades in both axes,
    since it assumes a local power law between native points rather than a
    straight line in linear space (which would badly misrepresent the huge
    dynamic range). Master-grid points outside this (template, z)'s actual
    redshifted wavelength coverage are set to exactly 0.0, never
    extrapolated -- 0 is the correct "no data here" sentinel for
    sedfitter's convolution (a plain sum with no NaN-handling; NaN would
    silently poison it).

    Non-positive flux values (Brown 2014 has a small number of noise-driven
    negative points; Polletta/Berta have none) are floored to flux_floor
    before the log transform.
    """
    wave_obs_um = observed_wavelength_um(wave_rest_um, z)
    flux_safe = np.clip(flux, flux_floor, None)

    log_master_wave = np.log10(master_wave_um)
    log_wave_obs = np.log10(wave_obs_um)
    log_flux = np.log10(flux_safe)

    interp_log_flux = np.interp(log_master_wave, log_wave_obs, log_flux, left=np.nan, right=np.nan)

    result = 10.0 ** interp_log_flux
    result[np.isnan(interp_log_flux)] = 0.0
    return result.astype(np.float32)


# There used to be a resample_seds_onto_master_grid/save_resampled_sed_
# archive pair here, materializing EVERY (template, z) SED of a source's
# full redshift grid onto the master wavelength grid and saving that whole
# cube to disk. That fit a world where the kept set WAS the raw set minus
# a redshift-cap mask. Item 2 of the owner's library specification makes
# the raw set a dense, disposable intermediate the r-net below samples
# from -- at RAW_GRID_N_Z density that full cube would be tens of GB
# across three sources for no downstream reader -- so the function most
# of this module now needs is the opposite shape: compute only the 8
# SESNA-band flux of every raw row (compute_raw_grid_photometry, below),
# and resample the full master-grid SED only for the much smaller KEPT
# set, directly from the rest-frame archive (prepare_combined_model_
# arrays), never saving an intermediate full-resolution raw cube at all.


# ====================================================================
# Sampling at the fitter's resolution (owner's library specification,
# item 2): one global r-net at SIGEFF over the whole capped raw set.
# ====================================================================
#
# ALGORITHM (owner's final ruling, superseding two earlier drafts of this
# section): greedy r-net by MAXIMUM UNCOVERED COVERAGE, not by traversal
# order and not by a medoid-refinement pass on a frozen partition.
#
#   while some raw model is uncovered:
#       pick the uncovered model with the most uncovered neighbours
#       within SIGEFF (deterministic tie-break: (source_tag,
#       template_name, redshift) ascending -- a fixed total order, no
#       random seed anywhere in this algorithm);
#       keep it; mark every model within SIGEFF of it covered.
#
# This gives both required properties BY CONSTRUCTION, not by a later
# check-and-patch pass:
#   COVERING -- the loop only stops when nothing is uncovered, so every
#       raw model ends within SIGEFF of some kept template.
#   PACKING -- a model is only ever picked while uncovered, i.e. while it
#       is NOT within SIGEFF of any already-kept template, so no two kept
#       templates are closer than SIGEFF.
#   CENTRING -- picking the uncovered point with the most uncovered
#       neighbours, at the moment it is picked, puts each representative
#       in the middle of a dense region of what is still left to cover,
#       rather than at an arbitrary traversal-order survivor sitting at
#       the edge of its own ball.
#
# The REPRESENTED SET for each kept template (what the MEMBERS group
# reports) is a SEPARATE, later step: every raw model is assigned to its
# NEAREST kept template (a Voronoi partition over the already-fixed
# representatives), never the other way round -- the sets are derived
# from the representatives, not the representatives chosen to fit
# pre-drawn sets.

#: MEMBERS group subclasses, in the fixed column order every members.fits
#: fraction column follows. Identical to (SUBCLASS_AGN, SUBCLASS_COMPOSITE,
#: SUBCLASS_PAH, SUBCLASS_PASSIVE), declared later in this module next to
#: assign_galaxy_subclass; written as a literal here purely so this
#: section can sit next to build_redshift_grid rather than after the
#: subclass machinery.
MEMBERS_SUBCLASSES = ("AGN", "COMP", "PAH", "PASS")

#: TEMPLATE_CLASS values (Stage 4's 5-way superclass), in the fixed column
#: order every members.fits FRAC_TEMPLATE_CLASS_* column follows.
MEMBERS_TEMPLATE_CLASSES = ("Elliptical", "Spiral", "Starburst", "AGN", "Composite")


def build_fast_band_filters(bands, master_wave_um):
    """Precompute, once, everything compute_raw_grid_photometry needs to
    convolve a (n_rows, n_wave) F_nu block against `bands` without ever
    calling sedfitter's own convolve_model_dir (which requires an on-disk
    model directory per call -- far too slow to run once per raw-grid row
    at RAW_GRID_N_Z density).

    Uses build_sedfitter_filter (model_convolution.py, imported, not
    modified) for the SAME filter curves and SAME central wavelengths
    every official convolved/{band}.fits uses, interpolated onto the
    master wavelength grid's own frequency axis and explicitly
    renormalized there (numerator/denominator, not relying on the
    filter's native-grid integral=1 normalization carrying over exactly
    through a second interpolation) -- the same safeguard model_
    convolution.analytic_band_flux_for_line_spectrum uses for the same
    reason.

    This is a SAMPLING-DECISION convolution, not a shipped one: it feeds
    only the r-net's distance computation. The library's actual shipped
    convolved/{band}.fits (Stage 6, build_convolved_bands) is produced
    exactly as before, by the official sedfitter path, on the much
    smaller KEPT set only.

    Returns {band: (freq_hz_asc, response_on_grid, denom)}.
    """
    # master_wave_um is ascending; frequency is then descending, so flip.
    freq_hz_asc = (C_UM_S / master_wave_um)[::-1]
    out = {}
    for band in bands:
        f = build_sedfitter_filter(band)
        filt_freq = np.asarray(f.nu.to("Hz").value, dtype=float)  # ascending
        filt_resp = np.asarray(f.response, dtype=float)
        response_on_grid = np.interp(freq_hz_asc, filt_freq, filt_resp, left=0.0, right=0.0)
        denom = np.trapz(response_on_grid, freq_hz_asc)
        out[band] = (freq_hz_asc, response_on_grid, denom)
    return out


def fast_band_fluxes_mjy(flux_lambda_asc_block, master_wave_um, band_filters, bands):
    """Convolve a (n_rows, n_wave) F_lambda block (ascending wavelength,
    per-Angstrom, as resample_one_sed_onto_master_grid returns) against
    every filter in `band_filters` (build_fast_band_filters), in `bands`
    order. Converts to F_nu (mJy) internally via flambda_to_fnu_mjy --
    the SAME conversion the shipped flux.fits uses -- then integrates in
    frequency space.

    Returns (n_rows, len(bands)) float64, band flux in mJy.
    """
    flux_nu_asc_wave = flambda_to_fnu_mjy(flux_lambda_asc_block, master_wave_um[np.newaxis, :])
    flux_nu_asc_freq = flux_nu_asc_wave[:, ::-1]  # ascending wavelength -> ascending frequency
    out = np.empty((flux_lambda_asc_block.shape[0], len(bands)), dtype=np.float64)
    for i, band in enumerate(bands):
        freq_hz_asc, response_on_grid, denom = band_filters[band]
        numerator = np.trapz(flux_nu_asc_freq * response_on_grid[np.newaxis, :], freq_hz_asc, axis=1)
        out[:, i] = numerator / denom
    return out


@dataclass(frozen=True)
class RawGridPhotometry:
    """One row per (source_tag, template_name, redshift) in the dense,
    per-source-capped raw grid -- the whole population the r-net samples
    from. `band_flux_mjy` is in BAND_ORDER column order."""

    source_tag: np.ndarray
    template_name: np.ndarray
    redshift: np.ndarray
    band_flux_mjy: np.ndarray    # (n, n_band)
    band_order: tuple
    n_zero_band_rows: int         # diagnostic: rows with >=1 exactly-zero band


def compute_raw_grid_photometry(redshifted_dir, master_wave_um, band_filters,
                                band_order=tuple(BANDS)):
    """The dense raw set's 8-band photometry, computed WITHOUT ever
    materializing a (n_templates, n_z, n_wave) cube for a whole source:
    one template's (n_z_kept, n_wave) block is built, convolved to 8
    numbers per row, and discarded before the next template starts. Peak
    memory stays at one template's block, not the whole raw set's.

    Reads each source's dense per-source archive (galaxy_curate.
    SOURCE_REST_ARCHIVE, written by Stage 1 with that source's own
    GALAXY_SOURCE_REDSHIFT_CAP as z_max) and keeps only the rows with
    REDSHIFT < that source's own cap -- the same strict inequality
    prepare_combined_model_arrays always applied, now applied here
    instead, since this is the first place a (template, z) pair exists
    at all.
    """
    tags, names, zs = [], [], []
    blocks = []
    n_zero_band_rows = 0

    for tag, fname in SOURCE_REST_ARCHIVE.items():
        d = np.load(os.path.join(redshifted_dir, fname))
        z_grid_full = d["z_grid"]
        z_cap = GALAXY_SOURCE_REDSHIFT_CAP[tag]
        z_kept = z_grid_full[z_grid_full < z_cap]
        if z_kept.size == 0:
            raise ValueError(f"{tag}: redshift cap {z_cap} leaves zero raw grid nodes")

        for name in d["names"]:
            name = str(name)
            wave_rest_um = d[f"{name}__wave_rest_um"]
            flux_rest = d[f"{name}__flux"]

            flux_block = np.empty((z_kept.size, master_wave_um.size), dtype=np.float32)
            for j, z in enumerate(z_kept):
                flux_block[j] = resample_one_sed_onto_master_grid(
                    wave_rest_um, flux_rest, z, master_wave_um)

            band_flux = fast_band_fluxes_mjy(flux_block, master_wave_um, band_filters, band_order)
            n_zero_band_rows += int(np.any(band_flux <= 0.0, axis=1).sum())

            tags.extend([tag] * z_kept.size)
            names.extend([name] * z_kept.size)
            zs.extend(z_kept.tolist())
            blocks.append(band_flux)

    return RawGridPhotometry(
        source_tag=np.array(tags), template_name=np.array(names),
        redshift=np.array(zs, dtype=np.float64),
        band_flux_mjy=np.concatenate(blocks, axis=0),
        band_order=tuple(band_order),
        n_zero_band_rows=n_zero_band_rows,
    )


def project_raw_grid_to_quotient_space(photometry, space, floor_rel=SAMPLING_FLOOR_REL):
    """log10(flux) -> quotient-space coordinates for every raw row.

    Floored PER ROW at floor_rel * that row's own largest band flux, not
    at a shared absolute number -- see SAMPLING_FLOOR_REL's docstring for
    why a shared floor is the wrong concept across Polletta/Berta's
    arbitrary normalization and Brown's real mJy scale. The projection
    itself (space.project) removes the all-ones/gray direction, which is
    exactly per-row mean subtraction, so it does not care what a row's
    overall scale was -- the floor only has any effect on a row with a
    literal 0.0 in some band.
    """
    flux = np.asarray(photometry.band_flux_mjy, dtype=float)
    row_floor = np.maximum(flux.max(axis=1), 0.0) * floor_rel
    row_floor = np.where(row_floor > 0, row_floor, 1e-300)  # all-zero row guard, never expected
    x = np.log10(np.maximum(flux, row_floor[:, None]))
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite log10 flux survived the per-row sampling floor")
    return np.ascontiguousarray(space.project(x))


def greedy_max_cover_r_net(coords, radius, tie_break_keys, chunk_size=20000):
    """Greedy r-net by maximum uncovered coverage (module header above),
    in MEMORY-BOUNDED lazy-heap form. Fully deterministic: no random seed
    anywhere. Ties in the uncovered-neighbour count are broken by
    `tie_break_keys[i]`, which the caller must make a total order (no two
    rows equal) -- here, (source_tag, template_name, redshift).

    MEMORY. The earlier version of this function called
    `tree.query_ball_point(coords, radius)` (no `return_length`), which
    materialises the FULL neighbour adjacency as an (n,) object array of
    Python lists -- O(n * mean_degree) list cells, tens of GB at this
    library's raw-grid density. That is NEVER done here. Only two
    primitives touch the tree after construction:
      - `query_ball_point(chunk, radius, return_length=True)`, chunked at
        `chunk_size` points, for the INITIAL counts only -- one int64 per
        point, no neighbour lists at all.
      - `query_ball_point(one_point, radius)`, called only for the point
        about to be tested/selected, and discarded immediately after use
        -- memory for at most one neighbour list at a time.
    So working memory is O(n) ints (counts, covered, the heap) plus one
    transient neighbour list, never O(n * mean_degree).

    ALGORITHM (lazy/accelerated greedy for monotone submodular
    maximization). The heap key is an UPPER BOUND on a point's current
    uncovered-neighbour count: exact when pushed, but only ever
    non-increasing afterwards as other points get covered. Popping the
    largest bound and recomputing its LIVE count either confirms it (no
    other selection has touched its neighbourhood since -- it is still
    the true global maximum, so select it) or lowers it (push the
    corrected value back and keep popping). Because true counts are
    monotone non-increasing over the whole run, a bound that recomputes
    unchanged is guaranteed still the maximum over ALL points, including
    ones not yet repopped -- so this selects exactly the same sequence as
    a version that recomputed every point's exact count before every
    choice, just without ever holding more than one neighbour list.

    Returns kept_idx: (K,) int64, ORIGINAL row indices into `coords`, in
    SELECTION order.
    """
    n = coords.shape[0]
    tree = cKDTree(coords)

    # Step 2: initial counts only, chunked -- one int64 per point, never a
    # neighbour list.
    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        counts[start:stop] = tree.query_ball_point(
            coords[start:stop], radius, return_length=True, workers=-1)

    covered = np.zeros(n, dtype=bool)
    heap = [(-int(counts[i]), tie_break_keys[i], i) for i in range(n)]
    heapq.heapify(heap)

    kept = []
    while heap:
        neg_count, _, i = heapq.heappop(heap)
        if covered[i]:
            continue  # already covered by an earlier selection -- discard
        stored = -neg_count
        neighbors = tree.query_ball_point(coords[i], radius)  # ONE point
        live = sum(1 for j in neighbors if not covered[j])
        if live < stored:
            # stale upper bound -- push the revalidated count back and
            # keep going; `neighbors` is discarded, never retained.
            heapq.heappush(heap, (-live, tie_break_keys[i], i))
            continue
        # live == stored: confirmed the current global maximum -- select.
        kept.append(i)
        for j in neighbors:
            covered[j] = True

    if not covered.all():
        # every row must end covered (by itself at least, once kept, or by
        # a kept neighbour) -- this would mean the loop exited with the
        # heap empty but uncovered rows remaining, which greedy max-cover
        # cannot do by construction; kept only as a hard assertion.
        raise RuntimeError("greedy_max_cover_r_net: heap exhausted with uncovered rows left")
    return np.array(kept, dtype=np.int64)


def assign_nearest_representative(coords, kept_idx):
    """Voronoi membership: every raw row -> the position (0..K-1) of its
    NEAREST kept representative in `kept_idx`, plus that distance.

    This is deliberately a SEPARATE step from greedy_max_cover_r_net's
    own covered-by-ball bookkeeping (module header): the represented sets
    the MEMBERS group reports are nearest-assignment Voronoi cells over
    the already-fixed representatives, not an artifact of selection
    order."""
    rep_tree = cKDTree(coords[kept_idx])
    dist, owner_pos = rep_tree.query(coords, k=1, workers=-1)
    return owner_pos, dist


@dataclass(frozen=True)
class SamplingResult:
    """Everything build/galz.py and the report need from one r-net run."""

    kept_idx: np.ndarray          # (K,) into the raw photometry arrays
    owner_pos: np.ndarray         # (n_raw,) -> position into kept_idx
    member_dist: np.ndarray       # (n_raw,) distance to nearest representative
    sigma_eff: float
    diagnostics: dict = field(default_factory=dict)


def sample_raw_grid(photometry, space):
    """Run the full sampling step (project -> greedy r-net -> Voronoi
    membership -> verification measurements) over one RawGridPhotometry.
    """
    coords = project_raw_grid_to_quotient_space(photometry, space)
    tie_break_keys = list(zip(photometry.source_tag.tolist(),
                              photometry.template_name.tolist(),
                              photometry.redshift.tolist()))
    if len(set(tie_break_keys)) != len(tie_break_keys):
        raise ValueError("raw grid has duplicate (source_tag, template_name, redshift) rows; "
                         "the tie-break order requires a total order")

    kept_idx = greedy_max_cover_r_net(coords, space.sigma_eff, tie_break_keys)
    owner_pos, member_dist = assign_nearest_representative(coords, kept_idx)

    covering_radius_over_sigeff = float(member_dist.max() / space.sigma_eff)
    rep_tree = cKDTree(coords[kept_idx])
    rep_nn_dist, _ = rep_tree.query(coords[kept_idx], k=2, workers=-1)
    min_rep_separation_over_sigeff = float(rep_nn_dist[:, 1].min() / space.sigma_eff)

    diagnostics = {
        "n_raw": int(coords.shape[0]),
        "n_kept": int(kept_idx.size),
        "sigma_eff": space.sigma_eff,
        "covering_radius_over_sigeff": covering_radius_over_sigeff,
        "min_rep_separation_over_sigeff": min_rep_separation_over_sigeff,
        "member_dist_over_sigeff_median": float(np.median(member_dist) / space.sigma_eff),
        "member_dist_over_sigeff_max": float(member_dist.max() / space.sigma_eff),
        "coverage_fraction_within_sigeff": float(np.mean(member_dist <= space.sigma_eff)),
        "n_zero_band_rows": photometry.n_zero_band_rows,
    }
    return SamplingResult(kept_idx=kept_idx, owner_pos=owner_pos, member_dist=member_dist,
                          sigma_eff=space.sigma_eff, diagnostics=diagnostics)


def adjacent_redshift_step_diagnostics(photometry, space):
    """Is the raw redshift grid finer or coarser than the r-net's own
    sampling radius? Answered DIRECTLY from the already-computed raw
    photometry, with no new photometry and no re-run at a different grid
    density (owner's ruling, superseding an earlier densification-ladder
    approach -- see RAW_GRID_N_Z's docstring).

    For each (source_tag, template_name) family's raw-grid redshift
    sequence (ascending), measures the quotient-space distance between
    EVERY ADJACENT pair of grid nodes, in SIGEFF units:

      < 1 SIGEFF -- the two nodes are closer together than the radius
          the r-net discriminates at, so the SAMPLER decides which (if
          either) survives; the grid step there is not what sets the
          kept count.
      >= 1 SIGEFF -- the two nodes are already far enough apart that the
          r-net cannot treat them as one representative's neighbourhood
          regardless of what (if anything) lies between them; the GRID
          step there is itself a binding constraint on resolution, not
          the sampler.

    Returns a dict: n_pairs, median/p25/p75 of the pooled (adjacent-pair
    distance / SIGEFF) distribution, grid_limited_fraction (fraction of
    pairs >= 1 SIGEFF), and the same four numbers broken out per
    source_tag (key f"{tag}_median" etc.), since the three sources'
    redshift caps differ by two orders of magnitude and are expected to
    sit very differently on this scale (Brown's dz per grid step at
    z<0.06 is tiny in SIGEFF units purely because its own cap is tiny).
    """
    coords = project_raw_grid_to_quotient_space(photometry, space)
    order = np.lexsort((photometry.redshift, photometry.template_name, photometry.source_tag))
    tag_o = photometry.source_tag[order]
    name_o = photometry.template_name[order]
    coords_o = coords[order]

    same_family = (tag_o[1:] == tag_o[:-1]) & (name_o[1:] == name_o[:-1])
    step_dist = np.linalg.norm(coords_o[1:] - coords_o[:-1], axis=1)
    pair_tag = tag_o[1:]  # the family tag each pair belongs to

    step_over_sigeff = step_dist[same_family] / space.sigma_eff
    pair_tag = pair_tag[same_family]

    def _stats(x):
        return {
            "n_pairs": int(x.size),
            "median": float(np.median(x)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "grid_limited_fraction": float(np.mean(x >= 1.0)),
        }

    out = _stats(step_over_sigeff)
    for tag in SOURCE_TAGS:
        tag_mask = pair_tag == tag
        if tag_mask.any():
            out[tag] = _stats(step_over_sigeff[tag_mask])
    return out


def read_sampling_quotient_space(laws=None):
    """The project-wide quotient space (SIGEFF + the 5-D projection).

    OWNER'S RULING (final; supersedes both the catalogue-derived
    SIGMA_LOG this used to read via sed_models_register.noise.read, and
    an intermediate draft that sampled at the bare survey calibration
    floor): sigma_log is sesnacomplete.constants.
    LIBRARY_SAMPLING_SIGMA_LOG_VECTOR -- TWICE the survey teams' own
    published absolute-calibration floor (2MASS J/H/Ks 0.010 dex,
    Skrutskie et al. 2006 AJ 131,1163; IRAC I1-I4 0.013 dex, Reach et al.
    2005 PASP 117,978; MIPS 24um 0.017 dex, Engelbracht et al. 2007 PASP
    119,994 -- so 0.020/0.026/0.034 dex respectively).

    The factor of 2 (constants.LIBRARY_SAMPLING_SIGMA_FACTOR) is derived,
    not tuned: a grid of spacing d sampling a chi2 Gaussian of width sigma
    reproduces the evidence integral (sum of exp(-chi2/2) over templates)
    to 2*exp(-2*pi^2*sigma^2/d^2); at d=2*sigma that error is 0.0144
    (1.4%, 0.015 nat), and it applies only to the best-measured ~1% of
    the catalogue (sources whose error reaches the calibration floor
    itself) -- for a typical detection (0.017-0.035 dex statistical
    error, so d~1*sigma there) the same expression is below 1e-4.
    Sampling at the bare floor (d=1*sigma) was tried and rejected: it
    implies template counts not a believable count of distinguishable
    SED shapes (constants.LIBRARY_SAMPLING_SIGMA_FACTOR's own docstring
    carries the full derivation).

    SIGEFF at this scale is 0.055853 dex -- sed_models_register.density.
    build_quotient_space's own new DEFAULT (confirmed by calling it with
    no sigma_log argument at all), passed explicitly here anyway so this
    function, and the report it feeds, name the vector without relying
    on reaching into the register's default.

    The projection itself (5-D, removing the gray/scale direction and
    both Av laws) is unchanged and still read-only -- no register is
    built or read here, and sed_models_register.noise is no longer
    imported at all: its SIGMA_LOG.hdf5 product was built from catalogue
    files that have since been deleted, so it is a stale artefact, not an
    input."""
    kwargs = {} if laws is None else {"laws": laws}
    return register_density.build_quotient_space(
        sigma_log=np.asarray(LIBRARY_SAMPLING_SIGMA_LOG_VECTOR, dtype=float),
        sigma_source="sesnacomplete.constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR "
                     "(2x survey absolute-calibration floor, not catalogue-measured)",
        **kwargs)


# ====================================================================
# Stage 4: TEMPLATE_CLASS + PAH_STRENGTH
# ====================================================================

_POLLETTA_CLASS = {}
for _n in ["Ell2", "Ell5", "Ell13"]:
    _POLLETTA_CLASS[_n] = "Elliptical"
for _n in ["S0", "Sa", "Sb", "Sc", "Sd", "Sdm", "Spi4"]:
    _POLLETTA_CLASS[_n] = "Spiral"
for _n in ["M82", "Arp220", "N6090", "N6240", "I22491", "I20551"]:
    _POLLETTA_CLASS[_n] = "Starburst"
for _n in ["Sey18", "Sey2", "QSO1", "QSO2", "BQSO1", "TQSO1", "Torus"]:
    _POLLETTA_CLASS[_n] = "AGN"
for _n in ["Mrk231", "I19254"]:
    _POLLETTA_CLASS[_n] = "Composite"


def _classify_polletta_template_class(name):
    """Polletta's own documented 25-template split (3 elliptical, 7 spiral,
    6 starburst, 7 AGN, 2 composite)."""
    return _POLLETTA_CLASS[name]


def _classify_berta_template_class(name):
    """Assigned from list.dat's descriptive names via keyword rules."""
    if name in ("Elliptical", "Secular_glx", "Cold_glx"):
        return "Elliptical"
    if name in ("Spiral", "Ly_break"):
        return "Spiral"
    if name in ("Torus", "Si_break"):
        return "AGN"
    if name.startswith("Type1_AGN") or name.startswith("Type2_AGN"):
        return "AGN"
    if name.startswith("SF_Type1_AGN") or name.startswith("SF_Type2_AGN") or name == "MIR_powlaw_SF_glx":
        return "Composite"
    if "SF_glx" in name:
        return "Spiral"
    raise ValueError(f"unclassified Berta template: {name!r}")


_NGC1275_OVERRIDE = "ngc-1275"  # T-type=99 placeholder, Pec, BPT unmeasured -- known Seyfert (Perseus A/3C 84)


def _classify_brown_template_class(slug, csv_row):
    """
    Assigned from the summary CSV's BPT emission-line diagnostic (AGN/SF/
    SF-AGN/-) plus morphology (peculiar/merger tag as a Starburst proxy,
    T-type for the structural E-vs-spiral split when BPT is unmeasured).
    One manual override: NGC 1275 (T-type=99 placeholder, BPT unmeasured,
    Pec morphology) is a well-documented Seyfert nucleus (Perseus A/3C 84),
    so it's assigned AGN rather than left to the generic fallback.
    """
    if slug == _NGC1275_OVERRIDE:
        return "AGN"

    bpt = csv_row["BPT"]
    morph = csv_row["Morphology"].lower()
    is_pec = "pec" in morph

    if bpt == "AGN":
        return "AGN"
    if bpt == "SF/AGN":
        return "Composite"
    if bpt == "SF":
        return "Starburst" if is_pec else "Spiral"

    # bpt == '-': no emission-line measurement: fall back to structural T-type
    ttype_raw = csv_row["T-type"]
    ttype = float(ttype_raw) if ttype_raw not in ("-", "99.0", "90.0") else None
    if ttype is None:
        raise ValueError(f"unresolvable Brown template (no BPT, no T-type): {slug!r}")
    return "Elliptical" if ttype <= 0 else "Spiral"


def _normalize_brown_csv_name(raw_name):
    """CSV 'Name' -> our file-slug convention (lowercase, spaces -> hyphens)."""
    return raw_name.strip().lower().replace(" ", "-")


def load_brown_summary_csv(csv_path):
    """Brown 2014 summary CSV, keyed by our file-slug name convention."""
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            row = {k: v.strip() for k, v in row.items()}
            slug = _normalize_brown_csv_name(row["Name"])
            rows[slug] = row
    return rows


_PAH_FEATURES = [
    # (feature_lo, feature_hi, cont_blue_lo, cont_blue_hi, cont_red_lo, cont_red_hi), um
    (6.0, 6.5, 5.85, 5.95, 6.60, 6.70),
    (10.95, 11.6, 10.15, 10.35, 11.65, 11.80),
]


def _local_log_interp(wave_rest_um, flux, targets_um):
    """Log-log interpolation of one template onto a few continuum/feature
    target wavelengths. NOT a resampler: this is the local continuum
    estimate behind _feature_equivalent_width.

    Out-of-span targets CLAMP to the nearest edge flux, stated explicitly
    via left=/right= rather than inherited from np.interp's default. The two
    axis conventions in this pipeline are deliberately different and both
    are now written down where they apply: continuum-side interpolation
    clamps to the edge value (here), wavelength-axis resampling returns
    exactly 0.0 outside coverage (resample_one_sed_onto_master_grid).
    In practice measure_pah_strength's span guard means the clamp never
    fires today; it is made explicit so the convention cannot drift."""
    log_w = np.log10(wave_rest_um)
    log_f = np.log10(np.clip(flux, FLUX_FLOOR, None))
    return 10.0 ** np.interp(np.log10(targets_um), log_w, log_f,
                             left=log_f[0], right=log_f[-1])


def _feature_equivalent_width(wave_rest_um, flux, feat_lo, feat_hi, cb_lo, cb_hi, cr_lo, cr_hi):
    cont_blue = np.mean(_local_log_interp(wave_rest_um, flux, np.linspace(cb_lo, cb_hi, 5)))
    cont_red = np.mean(_local_log_interp(wave_rest_um, flux, np.linspace(cr_lo, cr_hi, 5)))
    cont_blue_wave = 0.5 * (cb_lo + cb_hi)
    cont_red_wave = 0.5 * (cr_lo + cr_hi)

    grid = np.linspace(feat_lo, feat_hi, 100)
    flux_on_grid = _local_log_interp(wave_rest_um, flux, grid)
    # linear local continuum spanning the two anchor points
    continuum = cont_blue + (cont_red - cont_blue) * (grid - cont_blue_wave) / (cont_red_wave - cont_blue_wave)

    return np.trapz((flux_on_grid - continuum) / continuum, grid)


def measure_pah_ew_6_2(wave_rest_um, flux):
    """Local-continuum equivalent width (um) of the 6.2um PAH complex ALONE.

    Deliberately separate from measure_pah_strength, which returns the SUM of
    the 6.2 and 11.3um EWs. The sum is a project-local quantity with no
    published thresholds; EW(6.2) on its own is the standard mid-IR
    AGN/starburst discriminant, and the literature boundaries below apply to
    it and only to it. Mixing the two would attach published numbers to a
    quantity they were never measured against.

    Same span guard as measure_pah_strength: NaN if the template does not
    cover the feature's own continuum anchors.
    """
    feat = _PAH_FEATURES[0]
    if wave_rest_um.min() > feat[2] or wave_rest_um.max() < feat[5]:
        return np.nan
    return _feature_equivalent_width(wave_rest_um, flux, *feat)


# EW(6.2um PAH) class boundaries, um. Armus et al. 2007 (ApJ 656, 148):
# EW <= 0.2 AGN-dominated, 0.2 < EW <= 0.5 AGN+starburst COMPOSITE,
# EW > 0.5 starburst-dominated. Concordant independent determinations:
# Brandl et al. 2006 and Petric et al. 2011 both place AGN-dominated below
# ~0.2; Stierwalt et al. 2013 (GOALS) uses 0.27 / 0.54 for the same split on
# nuclear spectra.
#
# Armus is adopted over Stierwalt because it is the one that explicitly names
# the intermediate region as a COMPOSITE CLASS rather than as the gap between
# two dominance regimes -- and composite is a class this library needs. The
# choice is not free: on the 185 parents it moves ~17 templates between
# COMPOSITE and PAH (starburst bin 59 vs 42). Recorded in the info.fits header
# so which boundary set produced a shipped label never has to be re-derived.
PAH_EW_AGN_MAX = 0.2
PAH_EW_STARBURST_MIN = 0.5
PAH_EW_BOUNDARY_SOURCE = "Armus+2007 ApJ 656,148 (0.2/0.5 um)"


def measure_pah_strength(wave_rest_um, flux):
    """
    Sum of local-continuum equivalent widths (um) of the 6.2 and 11.3um PAH
    complexes -- chosen as the two bands most robust to the 9.7um silicate
    trough that otherwise contaminates simple (non-PAHFIT) continuum
    placement around 7.7/8.6um. Simple linear local continuum, not a full
    PAHFIT-style deblend -- proportionate to use as a coarse label, not
    publication-grade dust physics.
    """
    if wave_rest_um.min() > _PAH_FEATURES[0][2] or wave_rest_um.max() < _PAH_FEATURES[-1][5]:
        return np.nan  # template doesn't cover the PAH region at all
    total = 0.0
    for feat in _PAH_FEATURES:
        total += _feature_equivalent_width(wave_rest_um, flux, *feat)
    return total


def _classification_row(tag, name, template_class, pah, pah_ew_6_2, csv_row=None):
    row = {
        "source_tag": tag, "template_name": name,
        "template_class": template_class, "pah_strength": pah,
        "pah_ew_6_2": pah_ew_6_2,
        "source_cz": "", "morphology": "", "bpt_class": "", "d_l": "",
    }
    if csv_row is not None:
        row["source_cz"] = csv_row["cz"]
        row["morphology"] = csv_row["Morphology"]
        row["bpt_class"] = csv_row["BPT"]
        row["d_l"] = csv_row["D_L"]
    return row


def classify_polletta_templates(redshifted_dir):
    """Per-template TEMPLATE_CLASS + PAH_STRENGTH + PAH_EW_6_2 rows for Polletta, from its
    rest-frame archive. Returns a list of row-dicts."""
    d = np.load(os.path.join(redshifted_dir, "polletta2007_rest.npz"))
    rows = []
    for name in d["names"]:
        name = str(name)
        cls = _classify_polletta_template_class(name)
        wave, flux = d[f"{name}__wave_rest_um"], d[f"{name}__flux"]
        pah = measure_pah_strength(wave, flux)
        ew62 = measure_pah_ew_6_2(wave, flux)
        rows.append(_classification_row("pol07", name, cls, pah, ew62))
    return rows


def classify_berta_templates(redshifted_dir):
    """Per-template TEMPLATE_CLASS + PAH_STRENGTH + PAH_EW_6_2 rows for Berta, from its
    rest-frame archive. Returns a list of row-dicts."""
    d = np.load(os.path.join(redshifted_dir, "berta2013_rest.npz"))
    rows = []
    for name in d["names"]:
        name = str(name)
        cls = _classify_berta_template_class(name)
        wave, flux = d[f"{name}__wave_rest_um"], d[f"{name}__flux"]
        pah = measure_pah_strength(wave, flux)
        ew62 = measure_pah_ew_6_2(wave, flux)
        rows.append(_classification_row("ber13", name, cls, pah, ew62))
    return rows


def classify_brown_templates(redshifted_dir, brown_csv_path):
    """Per-template TEMPLATE_CLASS + PAH_STRENGTH + PAH_EW_6_2 rows for Brown, from its
    rest-frame archive plus the summary CSV's morphology/BPT/cz/D_L
    provenance. Returns a list of row-dicts."""
    d = np.load(os.path.join(redshifted_dir, "brown2014_rest.npz"))
    csv_rows = load_brown_summary_csv(brown_csv_path)
    rows = []
    for name in d["names"]:
        name = str(name)
        csv_row = csv_rows[name]
        cls = _classify_brown_template_class(name, csv_row)
        wave, flux = d[f"{name}__wave_rest_um"], d[f"{name}__flux"]
        pah = measure_pah_strength(wave, flux)
        ew62 = measure_pah_ew_6_2(wave, flux)
        rows.append(_classification_row("bro14", name, cls, pah, ew62, csv_row=csv_row))
    return rows


def write_template_classification_csv(out_path, *row_lists):
    """Combine per-source classification row-lists and write the single
    template_classification.csv all three sources share."""
    all_rows = [row for rows in row_lists for row in rows]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    return out_path


# ====================================================================
# Stage 5: assemble flux.fits / parameters.fits / info.fits / models.conf
# ====================================================================
# (Filter construction -- build_sedfitter_filter / pivot_wavelength_um --
# lives in model_convolution.py: shared infrastructure invoked within
# Stages 5-6, not a numbered stage of its own. See the module docstring.
# Not imported here; nothing in this module calls it directly.)

@dataclass
class CombinedGalaxyModelSet:
    """Combined, ready-to-write arrays across every KEPT (template, z) row
    -- the r-net's chosen representatives, one per members.fits row.
    Return type of prepare_combined_model_arrays.

    subclass/donley_agn are carried here too (not appended in a later
    pass, as the pre-sampling pipeline did): classify_parent_subclasses
    computes both up front, per PARENT TEMPLATE, decoupled from which
    redshift ends up as that template's representative, so there is no
    Stage-ordering reason left to defer them."""
    model_names: np.ndarray
    redshift: np.ndarray
    source_library: np.ndarray
    template_name: np.ndarray
    template_class: np.ndarray
    pah_strength: np.ndarray
    pah_ew_6_2: np.ndarray
    source_cz: np.ndarray
    morphology: np.ndarray
    bpt_class: np.ndarray
    d_l: np.ndarray
    subclass: np.ndarray
    donley_agn: np.ndarray
    wave_um_desc: np.ndarray      # descending -- matches the SEDCube ordering fix
    freq_hz_desc: np.ndarray      # ascending
    values_mjy: np.ndarray        # (n_models, 1, n_wave), F_nu, wavelength axis matches wave_um_desc
    uncertainties_mjy: np.ndarray  # same shape, all zero (no source provides per-point errors)


def flambda_to_fnu_mjy(flux_per_angstrom, wave_um):
    """F_nu[mJy] = F_lambda[per Angstrom] * lambda[Angstrom]^2 / c[Angstrom/s]
    / MJY_TO_CGS. c and the mJy scale factor come from constants.py rather
    than being re-declared here; C_UM_S * 1e4 is bit-exactly the old
    2.99792458e18 literal, and dividing by MJY_TO_CGS was checked to give
    float32 output bit-identical to the old * 1e26 across all 6.3e7 values
    of the shipped cube."""
    wave_angstrom = wave_um * 1e4
    f_nu_cgs = flux_per_angstrom * wave_angstrom ** 2 / (C_UM_S * 1e4)
    return f_nu_cgs / MJY_TO_CGS


def load_template_classification_csv(csv_path):
    """template_classification.csv -> dict keyed by (source_tag, template_name)."""
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows[(row["source_tag"], row["template_name"])] = row
    return rows


_SOURCE_LIBRARY_LABEL = {"pol07": "Polletta2007", "ber13": "Berta2013", "bro14": "Brown2014"}


def prepare_combined_model_arrays(redshifted_dir, master_wave_um, kept_rows,
                                  classification_csv_path, parent_lookup):
    """
    Build one CombinedGalaxyModelSet from an explicit list of KEPT
    (source_tag, template_name, redshift) rows -- the r-net's chosen
    representatives (galaxy_curate.sample_raw_grid), one row per
    members.fits entry -- rather than looping over a source's whole
    per-source redshift grid, as the pre-sampling pipeline did.

    The kept count is small (hundreds to a few thousand, see the build's
    own report), so each row's full master-grid SED is resampled directly
    from the source's rest-frame archive (REDSHIFTED_DIR) here, with no
    intermediate full-resolution cube ever written for it -- unlike the
    dense raw grid the sampling step itself runs over
    (compute_raw_grid_photometry), which never resamples to full
    resolution at all.

    TEMPLATE_CLASS/PAH_EW_6_2/PAH_STRENGTH come from
    classification_csv_path (Stage 4, rest-frame, per template);
    SUBCLASS/DONLEY_AGN come from `parent_lookup`
    (classify_parent_subclasses's return), both broadcast per (source_tag,
    template_name) exactly as before -- these four quantities are
    intrinsic rest-frame template properties, independent of which
    redshift a given kept row happens to carry.
    """
    classification = load_template_classification_csv(classification_csv_path)
    rest_archives = {}

    all_model_names, all_redshift = [], []
    all_source_library, all_template_name, all_template_class = [], [], []
    all_pah_strength, all_source_cz, all_morphology, all_bpt_class, all_d_l = [], [], [], [], []
    all_pah_ew_6_2, all_subclass, all_donley_agn = [], [], []
    flux_rows = []

    seen_names = set()
    for tag, name, z in kept_rows:
        if tag not in rest_archives:
            rest_archives[tag] = np.load(os.path.join(redshifted_dir, SOURCE_REST_ARCHIVE[tag]))
        d = rest_archives[tag]
        wave_rest_um = d[f"{name}__wave_rest_um"]
        flux_rest = d[f"{name}__flux"]
        flux_rows.append(resample_one_sed_onto_master_grid(wave_rest_um, flux_rest, z, master_wave_um))

        model_name_part = MODEL_NAME_TEMPLATE_ALIAS.get((tag, name), name)
        model_name = f"{tag}.{model_name_part}.z{z:.4f}"
        if model_name in seen_names:
            raise ValueError(
                f"duplicate MODEL_NAME {model_name!r} -- two kept rows for the same "
                "template rounded to the same 4-decimal redshift")
        seen_names.add(model_name)

        row = classification[(tag, name)]
        parent = parent_lookup[(tag, name)]
        all_model_names.append(model_name)
        all_redshift.append(z)
        all_source_library.append(_SOURCE_LIBRARY_LABEL[tag])
        all_template_name.append(name)
        all_template_class.append(parent["template_class"])
        all_pah_strength.append(float(row["pah_strength"]))
        all_pah_ew_6_2.append(parent["pah_ew_6_2"])
        all_source_cz.append(float(row["source_cz"]) if row["source_cz"] else np.nan)
        all_morphology.append(row["morphology"])
        all_bpt_class.append(row["bpt_class"])
        all_d_l.append(float(row["d_l"]) if row["d_l"] else np.nan)
        all_subclass.append(parent["subclass"])
        all_donley_agn.append(parent["donley_agn"])

    model_names = np.array(all_model_names)
    redshift = np.array(all_redshift, dtype=np.float64)
    flux_all = np.asarray(flux_rows, dtype=np.float32)  # (n_models, n_wave), F_lambda, ascending wave

    # -- fix a: F_lambda -> F_nu (mJy), using the (ascending) master grid as-is --
    flux_nu = flambda_to_fnu_mjy(flux_all, master_wave_um[np.newaxis, :])

    # -- fix b: reverse to descending wavelength / ascending frequency --
    wave_um_desc = master_wave_um[::-1]
    freq_hz_desc = C_UM_S / wave_um_desc  # c[um/s] / wave[um] -> Hz, ascending
    flux_nu_desc = flux_nu[:, ::-1]

    values = flux_nu_desc[:, np.newaxis, :].astype(np.float32)  # (n_models, 1, n_wave)
    uncertainties = np.zeros_like(values)

    return CombinedGalaxyModelSet(
        model_names=model_names, redshift=redshift,
        source_library=np.array(all_source_library), template_name=np.array(all_template_name),
        template_class=np.array(all_template_class), pah_strength=np.array(all_pah_strength),
        pah_ew_6_2=np.array(all_pah_ew_6_2),
        source_cz=np.array(all_source_cz), morphology=np.array(all_morphology),
        bpt_class=np.array(all_bpt_class), d_l=np.array(all_d_l),
        subclass=np.array(all_subclass), donley_agn=np.array(all_donley_agn, dtype=bool),
        wave_um_desc=wave_um_desc, freq_hz_desc=freq_hz_desc,
        values_mjy=values, uncertainties_mjy=uncertainties,
    )


GALAXY_DISTANCE_COMMENT = "FAKE plug value, not real -- see module docstring"


def write_flux_fits(combined, out_path):
    """Write flux.fits through the release-wide writer
    (model_io.write_flux_cube): PRIMARY (all-valid mask + the DISTANCE
    plug), MODEL_NAMES, SPECTRAL_INFO (descending wavelength / ascending
    frequency), APERTURES (a single point-source row -- the point-source
    shape, as opposed to stellar.fits, which omits the HDU), and
    VALUES/UNCERTAINTIES in mJy.

    The five libraries each used to carry their own copy of this function;
    they had drifted in the PRIMARY contents, the column units and the
    MODEL_NAME width. This call reproduces galaxy's shipped flux.fits
    byte-for-byte, including the DISTANCE card's comment, which is kept
    verbatim for exactly that reason.
    """
    return model_io.write_flux_cube(
        out_path,
        names=combined.model_names,
        wave_um_desc=combined.wave_um_desc,
        freq_hz_desc=combined.freq_hz_desc,
        values=combined.values_mjy,
        uncertainties=combined.uncertainties_mjy,
        apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
        distance_cm=REFERENCE_DISTANCE_CM,
        distance_comment=GALAXY_DISTANCE_COMMENT,
        name_format=MODEL_NAME_FORMAT,
    )


def write_parameters_fits(combined, out_path):
    """Write parameters.fits: MODEL_NAME (primary key, row order must match
    flux.fits exactly) + REDSHIFT (the one real fitting-grid quantitative
    axis) + TEMPLATE_CLASS, PAH_EW_6_2 and DONLEY_AGN -- all three
    intrinsic REST-FRAME per-template properties (Stage 4 for the first
    two; classify_parent_subclasses for DONLEY_AGN), broadcast across
    each template's kept representative(s) exactly as info.fits does.

    All four non-key columns are written in this ONE pass. The pre-
    sampling pipeline wrote DONLEY_AGN in a separate later pass
    (add_donley_agn_to_parameters_fits) because it needed
    convolved/{band}.fits, which did not exist until Stage 6 -- but that
    convolution ran on the FINAL kept set's own z=0 row, which no longer
    reliably exists once the r-net is free to choose any redshift as a
    template's representative. classify_parent_subclasses computes
    DONLEY_AGN earlier, from a dedicated z=0-only convolution independent
    of the sampling's choice of representative, so there is no longer a
    Stage-ordering reason to defer it."""
    params_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="MODEL_NAME", format=MODEL_NAME_FORMAT, array=combined.model_names),
            fits.Column(name="REDSHIFT", format="D", array=combined.redshift),
            fits.Column(name="TEMPLATE_CLASS", format="20A", array=combined.template_class),
            fits.Column(name="PAH_EW_6_2", format="D", unit="um", array=combined.pah_ew_6_2),
            fits.Column(name="DONLEY_AGN", format="L", array=combined.donley_agn),
        ],
        name="PARAMETERS",
    )
    fits.HDUList([fits.PrimaryHDU(), params_hdu]).writeto(out_path, overwrite=True)
    return out_path


def write_info_fits(combined, out_path):
    """Write info.fits: descriptive/provenance columns, not read by
    sedfitter's own fitting machinery. MODEL_NAME carried explicitly here
    too (not just positional alignment), matching the YSO package's own
    info.fits precedent.

    SUBCLASS is written here directly, in the same pass as every other
    column -- the pre-sampling pipeline appended it in a separate later
    rewrite (add_subclass_to_info_fits) because it needed Stage 6's
    convolved/{band}.fits; classify_parent_subclasses now computes it
    earlier (see write_parameters_fits's DONLEY_AGN docstring for the
    same reasoning), so info.fits never exists without it."""
    info_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="MODEL_NAME", format=MODEL_NAME_FORMAT, array=combined.model_names),
            fits.Column(name="SOURCE_LIBRARY", format="14A", array=combined.source_library),
            fits.Column(name="TEMPLATE_NAME", format="20A", array=combined.template_name),
            fits.Column(name="TEMPLATE_CLASS", format="20A", array=combined.template_class),
            fits.Column(name="PAH_STRENGTH", format="D", unit="um", array=combined.pah_strength),
            fits.Column(name="PAH_EW_6_2", format="D", unit="um", array=combined.pah_ew_6_2),
            fits.Column(name="SOURCE_CZ", format="D", unit="km/s", array=combined.source_cz),
            fits.Column(name="MORPHOLOGY", format="20A", array=combined.morphology),
            fits.Column(name="BPT_CLASS", format="10A", array=combined.bpt_class),
            fits.Column(name="D_L", format="D", unit="Mpc", array=combined.d_l),
            fits.Column(name="SUBCLASS", format=SUBCLASS_FORMAT, array=combined.subclass),
        ],
        name="INFO",
    )
    header = info_hdu.header
    header["SUBCLS"] = ("AGN|PAH|PASS|COMP", "galz mid-IR SUBCLASS values")
    header["SUBCLSRC"] = (PAH_EW_BOUNDARY_SOURCE, "PAH EW class boundaries")
    header["SUBCLSAG"] = (DONLEY_SOURCE, "AGN wedge, DONLEY_AGN column")
    header["SUBCLSEW"] = (f"AGN<={PAH_EW_AGN_MAX}, COMP<={PAH_EW_STARBURST_MIN}, PAH>",
                          "EW(6.2um PAH) boundaries, um")
    header["SUBCLSZ"] = ("rest-frame z=0, independent of kept rep. z",
                         "both axes are intrinsic properties")
    header["SUBCLSWN"] = ("Donley cut is observed-frame, used rest-frame",
                          "documented repurposing; see curation plan")
    header["SUBCLSTC"] = ("TEMPLATE_CLASS=AGN also sets SUBCLASS=AGN",
                          "curation AGN call; CURATION_2026-10-09 sec.3")
    fits.HDUList([fits.PrimaryHDU(), info_hdu]).writeto(out_path, overwrite=True)
    return out_path


DEFAULT_MODEL_SET_NAME = "Combined galaxy contaminant templates (Polletta 2007 / Berta 2013 / Brown 2014)"


def write_models_conf(out_path, name=DEFAULT_MODEL_SET_NAME):
    """Write models.conf through the release-wide writer
    (model_io.write_models_conf): version=2 since this is one monolithic
    flux.fits cube (no per-model seds/ files), aperture_dependent=no
    (point-source templates for contaminant-ID against unresolved
    photometry -- see project decision on aperture dependence),
    length_subdir=0, logd_step=LOGD_STEP (constants.py's shared value).
    Byte-identical to the shipped file."""
    return model_io.write_models_conf(
        out_path, name=name, aperture_dependent=False,
        length_subdir=0, logd_step=LOGD_STEP, version=2,
    )


# ====================================================================
# Stage 6: convolved/{band}.fits -- see model_convolution.py
# ====================================================================
# build_convolved_bands lives in model_convolution.py -- it is the plain
# build -> stamp -> verify path every CONTINUUM library needs, and sps uses
# the same function. Imported at module top only so Stage 7 below sits next
# to it in the file; the driver calls it from model_convolution directly.

CONVMETH_NOTE = "sedfitter convolve_model_dir, continuum SED"

SUBCLASS_AGN = "AGN"          # low PAH + power-law continuum
SUBCLASS_PAH = "PAH"          # aromatic-dominated (star-forming / starburst)
SUBCLASS_PASSIVE = "PASS"     # low PAH and no power law: old stellar population
SUBCLASS_COMPOSITE = "COMP"   # intermediate PAH: AGN + starburst
SUBCLASS_FORMAT = "8A"
DONLEY_SOURCE = "Donley+2012 ApJ 748,142 Eq.2"

# --- classmap.fits vocabulary --------------------------------------
# The CLASS id for this library. Settled across the five libraries rather
# than chosen here: abbrev where a library maps to exactly one
# GUTERMUTH_LABELS row, pop_abbrev where it spans several. GAL is the
# pop_abbrev covering PAH_GALAXY / AGN / GENERIC_GALAXY.
MODEL_CLASS = "GAL"

# Emitted in full even though this library has one class, so a multi-library
# aggregation can concatenate CLASS_LEGEND tables and recover the whole
# vocabulary without a lookup held somewhere else.
CLASS_LEGEND = (
    ("GAL", "Unresolved background galaxy (extragalactic contaminant)"),
)

# THE AUTHORITY for what subclasses exist here. Subclasses are authored, not
# discovered: the Armus and Donley papers supply boundaries, but deciding
# that those boundaries constitute these four classes -- rather than three,
# or none -- is this project's decision. Changing this tuple is a
# definitional change, not a cosmetic one.
#
# The numeric boundaries live in these DESCRIPTIONS rather than in a header
# card, because a HIERARCH card cannot hold them: a table row has no length
# limit and a card has ~51 characters.
#
# Deliberately NOT Gutermuth's PAHG/AGN/GGAL galaxy vocabulary -- the plan's
# Sec.1.4 Gutermuth-independence requirement means GAL's own subclasses must
# not be derived from it.
# AGN's description below covers TWO ways in: the observed-frame Donley
# wedge (donley2012_agn) OR the curation's own rest-frame TEMPLATE_CLASS
# call of "AGN" (Stage 4; galaxy_curate.classify_*_templates). Review
# lib_3_GALZ.md concern 2: of the 27 templates TEMPLATE_CLASS calls AGN,
# only 14 used to carry SUBCLASS AGN -- the Donley wedge alone recovers a
# minority of known AGN (Donley+2012's own stated completeness), so 5 of
# the 13 that leaked out were sitting in PASS, which is not a passive
# population by construction. Folding TEMPLATE_CLASS == "AGN" into the
# AGN signal passed to assign_galaxy_subclass (classify_parent_subclasses)
# recovers those 5; TEMPLATE_CLASS AGN templates with EW(6.2) > 0.2 um
# stay PAH/COMP, unchanged -- PAH dominance already wins over any
# continuum signal (assign_galaxy_subclass's own ordering), and that
# ordering is not changed here. DONLEY_AGN the COLUMN still records only
# the pure geometric wedge test, never the OR -- see donley2012_agn.
SUBCLASS_LEGEND = (
    ("AGN",  "Power-law continuum: EW(6.2um) <= 0.2 um, and inside the "
             "Donley wedge or TEMPLATE_CLASS=AGN"),
    ("PAH",  "Aromatic-dominated star formation: EW(6.2um) > 0.5 um"),
    ("COMP", "Composite AGN + star formation: 0.2 < EW(6.2um) <= 0.5 um"),
    ("PASS", "Passive: EW(6.2um) <= 0.2 um, outside the Donley AGN wedge "
             "and not TEMPLATE_CLASS=AGN"),
)


def donley2012_agn(f36, f45, f58, f80):
    """Donley et al. 2012 (ApJ 748, 142) Eq. 2 -- the revised IRAC AGN wedge.

    Returns a boolean array. Inputs are IRAC fluxes in any consistent unit:
    only RATIOS are used, which is what makes this usable on a scale-free
    template library where an apparent magnitude would be meaningless.
    Transcribed from the paper:

        x = log10(f_5.8 / f_3.6),   y = log10(f_8.0 / f_4.5)

        x >= 0.08  AND  y >= 0.15
          AND  y >= 1.21*x - 0.27  AND  y <= 1.21*x + 0.27
          AND  f_4.5 > f_3.6  AND  f_5.8 > f_4.5  AND  f_8.0 > f_5.8

    The last three enforce a monotonically rising IRAC SED -- a red power
    law, rather than a stellar bump with a red tail.
    """
    f36, f45 = np.asarray(f36, float), np.asarray(f45, float)
    f58, f80 = np.asarray(f58, float), np.asarray(f80, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(f58 / f36)
        y = np.log10(f80 / f45)
    return ((x >= 0.08) & (y >= 0.15)
            & (y >= 1.21 * x - 0.27) & (y <= 1.21 * x + 0.27)
            & (f45 > f36) & (f58 > f45) & (f80 > f58))


def assign_galaxy_subclass(pah_ew_6_2, is_donley_agn):
    """Combine the two axes into one label per template.

    PAH strength is tested FIRST; the continuum axis only breaks the low-PAH
    tie. An object with strong aromatic emission is aromatic-dominated
    whatever its continuum is doing -- that is what "dominated" means in
    Armus+2007's own framing -- so a Donley detection does not override it.
    Consequence, intended: a template can satisfy Donley and still be labelled
    PAH or COMP (4 of the 18 Donley detections do). DONLEY_AGN is kept as its
    own column so that is visible rather than buried inside SUBCLASS.

    RAISES on a template with no PAH coverage rather than emitting a blank.
    Subclassing is all-or-nothing here: having committed to subclassing this
    library, every model carries one, so an unclassifiable template is a build
    failure to be looked at, not a silently empty cell for a consumer to
    discover. No template hits this today -- all 185 cover the 6.2um anchors --
    so the guard exists to keep it that way.
    """
    ew = np.asarray(pah_ew_6_2, float)
    agn = np.asarray(is_donley_agn, bool)
    missing = ~np.isfinite(ew)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} template(s) have no finite PAH_EW_6_2, so they "
            "cannot be subclassed; every model in this library must carry a "
            "SUBCLASS. Check the 6.2um continuum-anchor coverage of those templates.")
    out = np.full(ew.shape, SUBCLASS_PASSIVE, dtype=object)
    out[ew > PAH_EW_STARBURST_MIN] = SUBCLASS_PAH
    out[(ew > PAH_EW_AGN_MAX) & (ew <= PAH_EW_STARBURST_MIN)] = SUBCLASS_COMPOSITE
    out[(ew <= PAH_EW_AGN_MAX) & agn] = SUBCLASS_AGN
    return out.astype(str)




IRAC_SUBCLASS_BANDS = ("I1", "I2", "I3", "I4")   # 3.6, 4.5, 5.8, 8.0 um


@dataclass
class GalaxySubclasses:
    """Stage 7's labels, per (template, z) row, in flux.fits order.

    Returned rather than only written, so classmap.fits is built from the
    SAME in-memory arrays that went into info.fits. CLASSMAP duplicates what
    info.fits carries and is therefore capable of disagreeing with it;
    computing once and writing twice is what makes that impossible.
    """
    model_names: np.ndarray
    subclass: np.ndarray
    donley_agn: np.ndarray
    parent_counts: dict



def write_classmap_fits(subclasses, out_path, model_dir=None):
    """classmap.fits, via the shared writer in model_io.

    This library supplies only what is its own -- the class id, the two
    legends and the provenance text -- because the shared writer deliberately
    holds no vocabulary; the five libraries have genuinely different notions
    of a subclass and there is no shared enumeration to derive them from.

    Built from Stage 7's in-memory labels, never by re-reading info.fits:
    CLASSMAP duplicates what info.fits carries and could otherwise drift from
    it. Passing `model_dir` turns on the assertion that MODEL_NAME matches
    flux.fits IN ORDER.
    """
    return model_io.write_classmap_fits(
        out_path,
        names=subclasses.model_names,
        class_id=MODEL_CLASS,
        subclass=subclasses.subclass,
        class_legend=CLASS_LEGEND,
        subclass_legend=SUBCLASS_LEGEND,
        provenance=(
            ("CLASS_SOURCE", "library declaration; all models here are this class"),
            ("SUBCLASS_SOURCE", "EW(6.2um) x (Donley wedge or TEMPLATE_CLASS=AGN)"),
            ("SUBCLASS_REF", "Armus+2007 ApJ 656,148; Donley+2012 ApJ 748,142"),
            ("SUBCLASS_NOTE", "assigned at z=0, broadcast to all kept redshifts"),
            ("LEGEND_SOURCE", "boundaries published; descriptions authored here"),
        ),
        model_dir=model_dir,
        name_format=MODEL_NAME_FORMAT,
    )


def classify_parent_subclasses(redshifted_dir, classification_csv_path, work_dir,
                                master_wave_um, bands=IRAC_SUBCLASS_BANDS):
    """SUBCLASS + DONLEY_AGN, per PARENT TEMPLATE (185), computed BEFORE
    sampling and independent of it.

    Both axes assign_galaxy_subclass needs -- the Donley IRAC wedge and
    the curation's own TEMPLATE_CLASS -- are intrinsic REST-FRAME
    properties of a template (module docstring, Stage 4), not of any one
    redshift. The pre-sampling pipeline computed them AFTER Stage 6, by
    reading convolved/{band}.fits for whichever rows of the FINAL shipped
    set happened to end in ".z0.000". That assumption breaks once the
    r-net is free to choose any redshift as a template's representative
    (item 2 of the owner's library specification): a template's kept row
    is now usually NOT at z=0. So this classifies the 185 parents on
    their own z=0 SED FIRST, through a dedicated throwaway model
    directory (work_dir/parents_z0/: flux.fits + models.conf only --
    sedfitter's convolve_model_dir needs nothing else), convolved via the
    OFFICIAL path (model_convolution.build_convolved_bands -- the same
    filters and method every shipped convolved/{band}.fits uses, not the
    fast internal convolution the sampling step uses for its distance
    computation), and the result is a per-(source_tag, template_name)
    lookup that prepare_combined_model_arrays broadcasts onto whichever
    row the r-net actually kept.

    donley2012_agn and assign_galaxy_subclass are called exactly as
    before, unmodified; only the TIMING and the INPUT (a z=0-only
    directory, not the final kept set) changed.

    Returns (lookup, parent_counts):
      lookup: {(source_tag, template_name): {"template_class",
               "pah_ew_6_2", "donley_agn" (bool), "subclass"}}
      parent_counts: {subclass: n_templates}, for the Stage printout
      (template-level, matching the pre-sampling report's convention).
    """
    if fits is None:
        raise RuntimeError("astropy is required to write FITS files.")
    classification = load_template_classification_csv(classification_csv_path)

    tags, names, flux_rows = [], [], []
    for tag, fname in SOURCE_REST_ARCHIVE.items():
        d = np.load(os.path.join(redshifted_dir, fname))
        for name in d["names"]:
            name = str(name)
            wave_rest_um = d[f"{name}__wave_rest_um"]
            flux_rest = d[f"{name}__flux"]
            # rest frame IS the z=0 observed frame (1+z=1): no redshifting
            flux_rows.append(resample_one_sed_onto_master_grid(
                wave_rest_um, flux_rest, 0.0, master_wave_um))
            tags.append(tag)
            names.append(name)

    flux_block = np.asarray(flux_rows, dtype=np.float32)
    flux_nu = flambda_to_fnu_mjy(flux_block, master_wave_um[np.newaxis, :])
    wave_um_desc = master_wave_um[::-1]
    freq_hz_desc = C_UM_S / wave_um_desc
    flux_nu_desc = flux_nu[:, ::-1]
    model_names = np.array([f"{t}.{n}.z0.000" for t, n in zip(tags, names)])

    parent_dir = os.path.join(work_dir, "parents_z0")
    os.makedirs(parent_dir, exist_ok=True)
    model_io.write_flux_cube(
        os.path.join(parent_dir, "flux.fits"),
        names=model_names, wave_um_desc=wave_um_desc, freq_hz_desc=freq_hz_desc,
        values=flux_nu_desc[:, np.newaxis, :].astype(np.float32),
        uncertainties=np.zeros((len(model_names), 1, master_wave_um.size), dtype=np.float32),
        apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
        distance_cm=REFERENCE_DISTANCE_CM, distance_comment=GALAXY_DISTANCE_COMMENT,
        name_format=MODEL_NAME_FORMAT,
    )
    model_io.write_models_conf(
        os.path.join(parent_dir, "models.conf"),
        name="GALZ parent-template (z=0) classification set, not shipped",
        aperture_dependent=False, length_subdir=0, logd_step=LOGD_STEP, version=2,
    )
    # convolve_model_dir (version-2 path) asserts parameters.fits'
    # MODEL_NAME against the flux cube's own names -- a minimal MODEL_NAME-
    # only table satisfies that; no other parameter is needed for this
    # throwaway classification directory.
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="MODEL_NAME", format=MODEL_NAME_FORMAT, array=model_names)],
            name="PARAMETERS"),
    ]).writeto(os.path.join(parent_dir, "parameters.fits"), overwrite=True)
    build_convolved_bands(parent_dir, bands, model_names,
                          "sedfitter convolve_model_dir, z=0 classification set")

    flux = {}
    for band in bands:
        with fits.open(os.path.join(parent_dir, "convolved", f"{band}.fits")) as hdul:
            band_names = np.array([str(n).strip() for n in hdul[1].data["MODEL_NAME"]])
            if not np.array_equal(band_names, model_names):
                raise ValueError(f"parents_z0/convolved/{band}.fits MODEL_NAME order mismatch")
            values = np.asarray(hdul[1].data["TOTAL_FLUX"], dtype=float)
            flux[band] = values[:, 0] if values.ndim > 1 else values

    agn = donley2012_agn(*(flux[b] for b in bands))
    template_class = np.array([classification[(t, n)]["template_class"] for t, n in zip(tags, names)])
    pah_ew_6_2 = np.array([float(classification[(t, n)]["pah_ew_6_2"]) for t, n in zip(tags, names)])
    agn_signal = agn | (template_class == "AGN")
    subclass = assign_galaxy_subclass(pah_ew_6_2, agn_signal)

    lookup = {
        (t, n): {
            "template_class": template_class[i], "pah_ew_6_2": float(pah_ew_6_2[i]),
            "donley_agn": bool(agn[i]), "subclass": subclass[i],
        }
        for i, (t, n) in enumerate(zip(tags, names))
    }
    labels, counts = np.unique(subclass, return_counts=True)
    return lookup, dict(zip(labels.tolist(), counts.tolist()))


# ====================================================================
# MEMBERS group (owner's library specification, item 3): per kept
# template, its represented set's subclass fractions, parameter ranges,
# TEMPLATE_CLASS/DONLEY_AGN composition, and count.
# ====================================================================
#
# The represented sets are the Voronoi partition assign_nearest_
# representative already computed (sample_raw_grid): every raw row
# belongs to the position (0..K-1) of its nearest kept representative,
# which is also the row order of `combined` (prepare_combined_model_
# arrays preserves kept_idx's selection order end to end). Point values
# for the representative ITSELF stay in parameters.fits/info.fits (the
# `models` group's analogue, for schema parity with the other five
# libraries); members.fits carries the represented SET's own ranges and
# compositions, never the other way round.

@dataclass(frozen=True)
class MembersSummary:
    """One row per kept template, aligned to `combined.model_names` (same
    order, same length)."""

    model_name: np.ndarray
    n_members: np.ndarray
    subclass_fraction: dict        # {subclass: (K,) float}
    template_class_fraction: dict  # {template_class: (K,) float}
    donley_agn_fraction: np.ndarray
    redshift_min: np.ndarray
    redshift_median: np.ndarray
    redshift_max: np.ndarray
    pah_ew_6_2_min: np.ndarray
    pah_ew_6_2_median: np.ndarray
    pah_ew_6_2_max: np.ndarray
    dist_to_rep_sigeff_median: np.ndarray
    dist_to_rep_sigeff_max: np.ndarray


def build_members_summary(photometry, sampling_result, combined, parent_lookup):
    """Build the MEMBERS group content from the raw photometry, the
    r-net's sampling result (owner_pos: Voronoi assignment; member_dist),
    and the already-assembled `combined` (whose row order IS kept_idx's
    selection order -- see module header).
    """
    n = photometry.source_tag.size
    k_total = combined.model_names.size
    if sampling_result.kept_idx.size != k_total:
        raise ValueError(
            f"sampling kept {sampling_result.kept_idx.size} rows but combined carries "
            f"{k_total} -- kept_rows must be built from sampling_result.kept_idx in order")

    # per-raw-row lookup arrays, built once per unique (source_tag,
    # template_name) group rather than per row (185 groups, not n rows).
    subclass_arr = np.empty(n, dtype=object)
    template_class_arr = np.empty(n, dtype=object)
    pah_ew_arr = np.empty(n, dtype=float)
    donley_arr = np.empty(n, dtype=bool)
    for (tag, name), p in parent_lookup.items():
        mask = (photometry.source_tag == tag) & (photometry.template_name == name)
        subclass_arr[mask] = p["subclass"]
        template_class_arr[mask] = p["template_class"]
        pah_ew_arr[mask] = p["pah_ew_6_2"]
        donley_arr[mask] = p["donley_agn"]
    if any(v is None for v in subclass_arr):
        raise ValueError("some raw rows matched no parent_lookup (source_tag, template_name) key")

    owner_pos = sampling_result.owner_pos
    order = np.argsort(owner_pos, kind="stable")
    sorted_owner = owner_pos[order]
    boundaries = np.searchsorted(sorted_owner, np.arange(k_total + 1))

    n_members = np.empty(k_total, dtype=np.int64)
    subclass_fraction = {s: np.zeros(k_total) for s in MEMBERS_SUBCLASSES}
    template_class_fraction = {c: np.zeros(k_total) for c in MEMBERS_TEMPLATE_CLASSES}
    donley_agn_fraction = np.empty(k_total)
    redshift_min = np.empty(k_total)
    redshift_median = np.empty(k_total)
    redshift_max = np.empty(k_total)
    pah_ew_6_2_min = np.empty(k_total)
    pah_ew_6_2_median = np.empty(k_total)
    pah_ew_6_2_max = np.empty(k_total)
    dist_med = np.empty(k_total)
    dist_max = np.empty(k_total)

    for k in range(k_total):
        idx = order[boundaries[k]:boundaries[k + 1]]
        if idx.size == 0:
            raise ValueError(
                f"kept template at position {k} ({combined.model_names[k]!r}) has an "
                "empty represented set -- every representative must cover at least itself")
        n_members[k] = idx.size

        sub = subclass_arr[idx]
        for s in MEMBERS_SUBCLASSES:
            subclass_fraction[s][k] = float(np.mean(sub == s))

        tc = template_class_arr[idx]
        for c in MEMBERS_TEMPLATE_CLASSES:
            template_class_fraction[c][k] = float(np.mean(tc == c))

        donley_agn_fraction[k] = float(np.mean(donley_arr[idx]))

        z = photometry.redshift[idx]
        redshift_min[k], redshift_median[k], redshift_max[k] = z.min(), np.median(z), z.max()

        ew = pah_ew_arr[idx]
        pah_ew_6_2_min[k], pah_ew_6_2_median[k], pah_ew_6_2_max[k] = ew.min(), np.median(ew), ew.max()

        d_sigeff = sampling_result.member_dist[idx] / sampling_result.sigma_eff
        dist_med[k], dist_max[k] = float(np.median(d_sigeff)), float(d_sigeff.max())

    return MembersSummary(
        model_name=combined.model_names, n_members=n_members,
        subclass_fraction=subclass_fraction, template_class_fraction=template_class_fraction,
        donley_agn_fraction=donley_agn_fraction,
        redshift_min=redshift_min, redshift_median=redshift_median, redshift_max=redshift_max,
        pah_ew_6_2_min=pah_ew_6_2_min, pah_ew_6_2_median=pah_ew_6_2_median,
        pah_ew_6_2_max=pah_ew_6_2_max,
        dist_to_rep_sigeff_median=dist_med, dist_to_rep_sigeff_max=dist_max,
    )


def write_members_fits(members, out_path):
    """Write members.fits: one row per kept template (MODEL_NAME joins to
    flux.fits/parameters.fits/info.fits positionally, same row order), the
    represented set's own count, subclass/TEMPLATE_CLASS/DONLEY_AGN
    composition, and REDSHIFT/PAH_EW_6_2 ranges -- the degeneracy
    bookkeeping (owner's library specification, item 3). Point values for
    the representative itself stay in parameters.fits/info.fits (the
    `models` group's analogue); this file is the represented SET's own
    statistics, never duplicated there."""
    columns = [
        fits.Column(name="MODEL_NAME", format=MODEL_NAME_FORMAT, array=members.model_name),
        fits.Column(name="N_MEMBERS", format="J", array=members.n_members),
    ]
    for s in MEMBERS_SUBCLASSES:
        columns.append(fits.Column(name=f"FRAC_SUBCLASS_{s}", format="D",
                                   array=members.subclass_fraction[s]))
    columns += [
        fits.Column(name="REDSHIFT_MIN", format="D", array=members.redshift_min),
        fits.Column(name="REDSHIFT_MEDIAN", format="D", array=members.redshift_median),
        fits.Column(name="REDSHIFT_MAX", format="D", array=members.redshift_max),
        fits.Column(name="PAH_EW_6_2_MIN", format="D", unit="um", array=members.pah_ew_6_2_min),
        fits.Column(name="PAH_EW_6_2_MEDIAN", format="D", unit="um", array=members.pah_ew_6_2_median),
        fits.Column(name="PAH_EW_6_2_MAX", format="D", unit="um", array=members.pah_ew_6_2_max),
    ]
    for c in MEMBERS_TEMPLATE_CLASSES:
        columns.append(fits.Column(name=f"FRAC_TEMPLATE_CLASS_{c.upper()}", format="D",
                                   array=members.template_class_fraction[c]))
    columns += [
        fits.Column(name="FRAC_DONLEY_AGN", format="D", array=members.donley_agn_fraction),
        fits.Column(name="DIST_TO_REP_SIGEFF_MEDIAN", format="D",
                    array=members.dist_to_rep_sigeff_median),
        fits.Column(name="DIST_TO_REP_SIGEFF_MAX", format="D",
                    array=members.dist_to_rep_sigeff_max),
    ]
    members_hdu = fits.BinTableHDU.from_columns(columns, name="MEMBERS")
    members_hdu.header["MEMBSRC"] = ("nearest-rep. Voronoi assignment over r-net kept set",
                                      "represented-set definition")
    fits.HDUList([fits.PrimaryHDU(), members_hdu]).writeto(out_path, overwrite=True)
    return out_path
