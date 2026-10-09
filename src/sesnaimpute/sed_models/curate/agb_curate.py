"""Curate the GRAMS dusty-AGB model grids into a yso-conforming CLASS AGB
model set.

Stage B (`SESNA_Complete/claude code/agb_model/STAGE_B_PLAN.md`, 2026-08-23)
adds a sixth model library so bright, red, warm-dust evolved stars have a
competitor to the YSO class in the fitter, rather than being silently
absorbed by it. This module is the curation code; see
`agb_model/grams_characterisation.md` (Opus, 2026-08-22) for the source
characterisation every number below was measured against, and
`docs/agb_decisions.md` (T3, planner) for the numbered B-1..B-15 decision
record this module implements piece by piece.

Stages -- mirrors sps_curate.py's numbering convention (0 = raw fetch, not
part of the curation proper; 1-5 = the five build stages every library in
this project shares):

  0. fetch_grams        -- (T2, implemented) pin + fetch the two raw GRAMS
                            FITS grids from figshare; run once, ahead of
                            time, NOT invoked by any later stage
  1. select              -- (T4, implemented) B-2/B-3/B-4 subset: O-rich
                            collapsed to its 1225 (Teff,Rin,tau10) shapes,
                            one L per shape (nearest 5000 Lsun); C-rich
                            greedily thinned at 0.05 mag Chebyshev in the 7
                            adjacent colours (reusing locus_study's
                            algorithm verbatim); both cut at the B-4 tau
                            floor (O tau10>=0.0128, C tau11_3>=0.02)
  2. convert              -- (T4, implemented) F_mJy = F_Jy * 2.5e6 (50kpc
                            -> 1kpc plug, Jy->mJy); B-7 closure gate
                            [0.95,1.05] on the native spectrum vs
                            L/(4 pi d^2); blue zero-fill, red nu^(2+beta)
                            tail (B-8); log-log interpolation onto the
                            common R=300 grid (reusing sps_curate's own
                            build_common_wavelength_grid /
                            interp_loglog_zero_preserving)
  2.5 population cut      -- (CURATION_2026-10-09.md sec 6 item 2,
                            2026-10-09) `apply_evolved_teff_cut` drops the
                            88 rows (67 O + 21 C) above the population's
                            own T_eff<3981K evolved-star selector
                            (provenance: bms_prior/class_densities/
                            class_density_star_family.py
                            EVOLVED_LOGTE_MAX=3.6). Leaves 1,892 rows.
                            [2026-10-09 THIN-TAU INTERPOLATION, ADDED AND
                            REVERTED SAME DAY: a log-tau interpolation of
                            the O-rich thin end was added under this
                            stage and then removed -- the register's own
                            O-rich nearest-neighbour spacing BEFORE that
                            interpolation was already median 0.339
                            sigma_eff / p90 0.792 (3x finer than the
                            noise), and Section 6's "adjacent nodes under
                            sigma" criterion was already satisfied;
                            interpolation produced redundancy (median
                            0.129 / p90 0.395 sigma_eff after 1,667 new
                            rows) and the library spec forbids
                            interpolation beyond the raw grid regardless.
                            See `sample_agb_templates` below for the
                            spec-compliant replacement: a greedy r-net
                            SELECTION over these same 1,892 raw rows, no
                            new rows synthesized.]
  3. parameters           -- (T4, implemented) the widened B-9 schema
                            (MODEL_NAME, T_EFF, LOGG, Z_H, TAU,
                            TAU_WAVE_UM, R_IN, L_SUN, T_IN, MLR_DUST, CHEM,
                            SOURCE); B-9 names, `_pNNN` only where the
                            C-rich (Teff,Rin,tau) tuple is degenerate.
                            Applied to ALL 1,892 curated rows, THEN
                            sorted ASCII-lexicographically, BEFORE Stage
                            2.75 runs -- so every native name stays
                            byte-identical regardless of which rows that
                            stage keeps, and so Stage 2.75's tie-break
                            (lowest row index) is that fixed name order.
  2.75 sample              -- (owner library spec, 2026-10-08, with the
                            2026-10-08 greedy-max-coverage correction and
                            the 2026-10-08 sampling-SCALE ruling, settled
                            same day at `LIBRARY_SAMPLING_SIGMA_LOG_VECTOR`
                            below) deterministic greedy set-cover r-net
                            (NO seed) at radius sigma_eff (one noise
                            length) over the 1,892 named, name-sorted
                            curated rows, projected into
                            sed_models_register.density's 5-D quotient
                            space (gray/scale + both Av laws removed).
                            sigma_eff is built from
                            constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR --
                            TWICE the surveys' own published
                            absolute-calibration systematics floor (2MASS
                            0.010->0.020 dex, IRAC 0.013->0.026 dex,
                            MIPS24 0.017->0.034 dex; factor of 2 derived,
                            not tuned: a d=2*sigma grid reproduces a
                            class's evidence integral to 1.4%/0.015 nats
                            for the best ~1% of sources, below 1e-4 for a
                            typical detection -- see constants.py). This
                            IS `build_quotient_space`'s own default now.
                            Supersedes an earlier catalogue-derived
                            SIGMA_LOG scale that stage used at first (0.6%
                            different numerically, since it happened to be
                            close); that product is retired, its source
                            files deleted, and `noise` is not imported
                            anywhere in this module or its driver.
                            Selects the KEPT templates that become the
                            shipped models group (centrally placed in
                            their eventual represented sets by
                            construction -- see `greedy_rnet`); every raw
                            row is then assigned to its nearest kept
                            template (Voronoi, derived from the
                            representatives) for the MEMBERS group
                            (represented set, subclass fractions,
                            parameter ranges, count). No interpolation,
                            no population weight, no prior or Gaia
                            quantity enters this step.
  4. write                -- (T4, implemented) models.conf, flux.fits
                            (B-6 per-band uncertainty floor), parameters.fits,
                            classmap.fits (CLASS='AGB', SUBCLASS in
                            {'O','C'}, B-10) -- for the KEPT templates
                            only (point values; schema parity with every
                            other library); members.fits records the
                            represented-set metadata for the full 1,892
  5. convolve             -- (T4, implemented) the 8 SESNA bands (B-11 --
                            no G, matching every other library), plus an
                            in-stage validation against reference_convolve
                            on the native spectrum and against GRAMS'
                            own Fphot

Stage 0 (T2) and stages 1-5 (T4, gated on the T1 locus study and T3's
decisions doc) are all implemented in this module. `curate_agb_model_set`
below is the stage 1-5 orchestrator, mirroring
`sps_curate.curate_sps_model_set`.


DESIGN DECISIONS
-----------------

Source (B-1, STAGE_B_PLAN.md / grams_characterisation.md §1.2). Both
chemistries come from figshare 10.6084/m9.figshare.5572834.v2 (Srinivasan),
the only source that is complete (both chemistries), ships full spectra
(not just synthetic photometry), and is CC BY 4.0. VizieR carries C-rich
only; the DESK/Zenodo re-packaging drops `Fstar` (the bare-photosphere
spectrum, wanted for the naked-vs-dusty B-4 cut) and all 82 bands of
synthetic photometry (wanted for the B-11 convolved/ validation, since
GRAMS's own `Fphot` gives an independent cross-check of this project's own
convolution). `Fstar` and `Fphot` are therefore fetched and retained in the
raw download for later validation use even though they are not carried
into `flux.fits` (B-1's explicit scope).

  GRAMS_FILES = {"grams_o.fits": <figshare files/9684331>,
                 "grams_c.fits": <figshare files/9684328>}

Citation correction (grams_characterisation.md §1.1): the O-rich grid paper
is Sargent, Srinivasan & Meixner 2011, ApJ 728, 93. The C-rich grid paper is
Srinivasan, Sargent & Meixner 2011, **A&A 532, A54** -- NOT "ApJ 734, 24",
which is an unrelated paper and sometimes seen misquoted for this grid. The
FITS headers themselves name Sargent+2011 (O) and Srinivasan+2011 (C), so
any provenance string this project writes for the C-rich grid must cite
A&A 532, A54.

Manifest pinning, BOOTSTRAPPED not hard-coded (mirrors
sps_curate._load_cifist_manifest / fetch_btsettl_cifist). figshare's DOI is
versioned and the two files are individually addressed by permanent
`ndownloader.figshare.com/files/<id>` URLs, but no third party has
published an independent checksum for them, so the pin at
data/agb/grams_manifest.json is written on first fetch and VERIFIED, never
silently updated, on every later call -- same discipline as the CIFIST
extension's data/sps/btsettl_cifist_manifest.json. Unlike CIFIST, there is
no gzip postprocess step here: GRAMS ships as plain FITS binary tables
(230.5 MB + 46.7 MB = 277 MB total, already reasonably sized), so the bytes
kept on disk ARE the pinned content and `model_io.fetch_pinned`'s plain
rename path is used as-is.

`verify_dest` structural check. Beyond the byte-for-byte sha256 pin,
`_grams_verify_dest` opens each cached file with astropy and asserts HDU1
is a BinTableHDU with the row count grams_characterisation.md §1.4 measured
on the actual files (68,600 O-rich / 12,243 C-rich) and that it carries the
`Teff, Lspec, Fspec, Fstar, Fphot` columns every later stage (and B-1's own
retained-for-validation claim) depends on. A byte hash alone would catch a
truncated download or a changed upstream artifact but would not catch, say,
a hand-placed wrong file that happens to already have the right sha256
entry removed from the manifest -- cheap insurance given the file is opened
with astropy exactly once per fetch call regardless.

`purge_downloads` mirrors sps_curate.purge_downloads's contract (safe,
idempotent, reproducible from the manifest) but is scoped to the two named
files rather than an rmtree of the whole directory -- download_dir here
IS the GRAMS-specific directory the driver names (`downloads/grams/`),
analogous to sps_curate's own out_dir (`downloads/btsettl_cifist/`), so
removing the two known files is the exact analogue of that rmtree without
assuming the directory is used for nothing else.


STAGE 1-5 DESIGN DECISIONS (T4)
--------------------------------

Stage 1 -- select (B-2/B-3/B-4). O-rich: `select_orich_shapes` collapses
the 68,600 rows to the 1225 distinct (Teff, R_in, tau10) shapes (B-3: L is
an exact multiplicative scale in this grid -- verified in
grams_characterisation.md §1.5 item 1 -- so one row per shape loses no
shape information), keeping the L node nearest 5000 Lsun (inside the
TRILEGAL evolved-star range). C-rich: `select_crich_thin` greedily thins
all 12,243 rows at 0.05 mag Chebyshev distance in the 7 ADJACENT Vega
colours (J-H, H-Ks, Ks-I1, I1-I2, I2-I3, I3-I4, I4-M1), computed from the
grid's own `mphot` synthetic photometry (columns 4..11 of the 82-filter
block -- verified against `Fphot` in locus_study/scripts/grams_lib.py: `Fphot
* 10**(mphot/2.5)` reproduces the project's own Vega zero points to within
the I4 filter-convention difference noted there). `greedy_thin_fast` below
is a line-for-line port of `locus_study/scripts/grams_lib.py`'s function of
the same name (T1's reference implementation) -- walking rows in FILE
ORDER and discarding a row iff it is within `tol` of an already-kept row in
EVERY one of the 7 colours simultaneously -- so this module's thinning
reproduces T1's count exactly (1551), not the earlier order-dependent 1567
quoted in grams_characterisation.md §1.6 (see locus_study/README.md §1.1:
"greedy thinning is order-dependent... pin the code, not the number").
Finally `apply_b4_tau_cut` drops O-rich tau10 < 0.0128 and C-rich
tau11_3 < 0.02 (B-4, from the T1 locus study), expected to leave ~925 O +
~1055 C = ~1980 rows.

Stage 2 -- convert (B-7/B-8). `native_closure_ratio` reproduces
grams_characterisation.md §1.4's exact check (integral of F_nu dnu on the
NATIVE 50 kpc spectrum vs L/(4 pi d^2) at d=50 kpc) so the B-7 gate
operates on the same quantity the characterisation measured (range
0.9988-1.0255 O / 0.9987-1.0125 C, i.e. inside [0.95,1.05] for every row
seen so far -- the gate is still enforced, and any row outside it is
dropped and logged, per B-7). Unit conversion is the literal B-2/§3.2 item
1+2 formula: F_mJy = F_Jy * (50000/1000)^2 * 1000 = F_Jy * 2.5e6. The red
end (B-8) is extended with a power-law tail F_nu ~ nu^(2+beta) fit from the
spectrum's own last two native nodes (beta clipped to [0,2]) out to the
grid's own 1e6 um edge -- `_agb_red_tail_beta` / `_append_agb_red_tail` --
then the EXTENDED spectrum (native + tail) is log-log interpolated onto
the common grid with sps_curate's own `interp_loglog_zero_preserving`,
which zero-fills everything blueward of the native coverage automatically
(the CK03 pipeline's own zero-preservation rule, reused verbatim -- no new
blue-fill code needed). This is bookkeeping only: no SESNA filter reaches
either the native blue edge (0.20/0.234 um) or the grid edges (0.009/1e6
um).

Stage 3 -- parameters (B-9). `agb_model_name` builds
`grams{O|C}_t{Teff}_r{Rin}_tau{tau:.2e}` (28 chars for the two worked
examples in B-9), and `assign_agb_names` appends `_pN` (shrinking tau's
exponential precision as needed to stay <=30 chars) ONLY to rows sharing an
exact (chem, Teff, R_in, tau) tuple with another row -- which can only
happen in the C-rich grid, where multiple photospheres (logg, Mass, C2O)
sit at the same dust-shell axes; O-rich rows are unique on
(Teff, R_in, tau10) by construction (that IS the definition of a "shape",
B-3). Every name is asserted <=30 chars and globally unique before writing;
rows are then sorted ASCII-lexicographically by name, matching every other
library's row-order convention (D-12/D-16).

Stage 4 -- write (B-6/B-10). `models.conf` via model_io.write_models_conf
(aperture_dependent=no per B-15's measured shell-size check, length_subdir=0,
logd_step=0.02, version=2). `flux.fits` via a thin B-9-aware wrapper around
model_io.write_flux_cube, same DISTANCE plug as SPS
(`constants.REFERENCE_DISTANCE_CM`, i.e. the SAME 1 kpc frame, not GRAMS'
native 50 kpc -- the flux values themselves already carry the B-2/§3.2
distance-squared conversion). UNCERTAINTIES (B-6) is the project's 1%
fractional floor, raised over each affected band's OWN filter support
(measured from the shipped filter curves: G 0.32-1.05 um, J 1.066-1.442 um,
H 1.44-1.85 um, Ks 1.934-2.384 um, I3 4.74421-6.62251 um) to the
per-chemistry sampling error measured directly against the point-sampling
error on the project's own R=300 photospheres, at the maximum over the
library's T_eff range (agb_model/validation/README.md §3.3/§3.5; O: G 18%,
J 3.5%, H 4.6%, Ks 1.8%; C: G 35%, J 1.5%, H 10%, Ks 4.0%, I3 1.8%).
`classmap.fits` via model_io.write_classmap_fits: CLASS='AGB', SUBCLASS is
each row's chemistry ('O'/'C'), PRIMARY provenance citing Sargent+2011 ApJ
728,93 (O) and Srinivasan+2011 A&A 532,A54 (C), CC BY 4.0 (B-1).

Stage 5 -- convolve (B-11/D-9). `convolved/{J,H,Ks,I1,I2,I3,I4,M1}.fits`
(the project's usual 8 bands, no Gaia G -- B-11) via
model_convolution.build_convolved_bands, run against the flux.fits just
written (D-9: convolved/ is always DERIVED, never copied or substituted
from a source table's own photometry). `validate_agb_convolution` then
draws a stratified sample of rows and checks the grid-convolved values two
ways, logged but NOT gating the build: (a) against `reference_convolve` on
each row's own NATIVE spectrum (expected agreement <1%, since the grid is
log-log interpolated from those same native nodes -- this is the same
comparison Stage-A's own R=300 grid standard is built on); (b) against
GRAMS' own `Fphot` for the 8 bands it carries (expected 0.05-3% typically,
excursions to ~8% at extreme tau -- see grams_characterisation.md §1.4 and
locus_study/README.md §4 item 4 for why GRAMS' own photometry is not
expected to agree exactly: different filter set/convention, coarser native
sampling).
"""

import functools
import hashlib
import json
import os
import socket
import time
import urllib.error
import urllib.request
from collections import defaultdict

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

from sesnaimpute.sed_models.constants import (
    C_UM_S, CONVOLVED_BAND_NAMES, LOGD_STEP, L_SUN_ERG_S,
    POINT_SOURCE_APERTURE_AU, REFERENCE_DISTANCE_CM,
)
from sesnaimpute.population.star_population import (
    EVOLVED_LOGTE_MAX as _EVOLVED_LOGTE_MAX,
)
from sesnaimpute.sed_models.curate import model_io
from sesnaimpute.sed_models.curate.model_convolution import build_convolved_bands
from sesnaimpute.sed_models.curate.sps_curate import (
    MODEL_NAME_COLUMN_FORMAT,
    build_common_wavelength_grid,
    interp_loglog_zero_preserving,
    reference_convolve,
)

# ====================================================================
# Stage 0 -- fetch
# ====================================================================

# figshare 10.6084/m9.figshare.5572834.v2 (Srinivasan), CC BY 4.0.
# O-rich: Sargent, Srinivasan & Meixner 2011, ApJ 728, 93.
# C-rich: Srinivasan, Sargent & Meixner 2011, A&A 532, A54 -- NOT
# "ApJ 734, 24" (a citation error that appears in some downstream task
# briefs; see the module docstring's "Citation correction" note and
# grams_characterisation.md §1.1).
GRAMS_DOI = "10.6084/m9.figshare.5572834.v2"
GRAMS_LICENSE = "CC BY 4.0"
GRAMS_CITATION_O = "Sargent, B. A., Srinivasan, S., & Meixner, M. 2011, ApJ, 728, 93"
GRAMS_CITATION_C = "Srinivasan, S., Sargent, B. A., & Meixner, M. 2011, A&A, 532, A54"

GRAMS_FILES = {
    "grams_o.fits": "https://ndownloader.figshare.com/files/9684331",
    "grams_c.fits": "https://ndownloader.figshare.com/files/9684328",
}

# Rows measured directly on the fetched files (grams_characterisation.md
# §1.4), not quoted from either paper: O-rich = 1225 (Teff,R_in,tau)
# combinations x 56 L nodes = 68600; C-rich = 12243 (L welded to the
# photosphere, not a free axis -- see §1.5).
GRAMS_EXPECTED_ROWS = {
    "grams_o.fits": 68600,
    "grams_c.fits": 12243,
}

# Columns _grams_verify_dest requires HDU1 to carry (grams_characterisation.md
# §1.4's measured column list is longer; these five are the ones B-1's
# validation-only retention of Fstar/Fphot and later stages' flux
# conversion (Teff, Lspec, Fspec) actually depend on).
GRAMS_REQUIRED_COLUMNS = ("Teff", "Lspec", "Fspec", "Fstar", "Fphot")

# Bootstrapped sha256 pin: {filename: {"sha256", "nbytes", "url",
# "expected_rows"}}. Mirrors sps_curate._CIFIST_MANIFEST_PATH's placement
# one level up from sed_models_curate/, under data/<library>/.
_GRAMS_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "agb", "grams_manifest.json",
)


def _load_grams_manifest():
    if os.path.exists(_GRAMS_MANIFEST_PATH):
        with open(_GRAMS_MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def _save_grams_manifest(manifest):
    directory = os.path.dirname(_GRAMS_MANIFEST_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(_GRAMS_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _validate_grams_fits_structure(path, expected_rows,
                                   required_columns=GRAMS_REQUIRED_COLUMNS):
    """Open `path` with astropy and assert HDU1 is a BinTableHDU with
    exactly `expected_rows` rows and every column in `required_columns`.
    Raises ValueError on any mismatch -- a real event (truncated download,
    upstream schema change, wrong file) to investigate, never something to
    silently accept as a valid cache hit."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        if len(hdul) < 2:
            raise ValueError(
                f"{path}: expected at least 2 HDUs (PRIMARY + BinTable), got {len(hdul)}")
        hdu1 = hdul[1]
        if not isinstance(hdu1, fits.BinTableHDU):
            raise ValueError(f"{path}: HDU1 is {type(hdu1).__name__}, not BinTableHDU")
        n_rows = int(hdu1.data.shape[0])
        if n_rows != expected_rows:
            raise ValueError(f"{path}: HDU1 has {n_rows} rows, expected {expected_rows}")
        missing = [c for c in required_columns if c not in hdu1.columns.names]
        if missing:
            raise ValueError(f"{path}: HDU1 missing required column(s): {missing}")


def _grams_verify_dest(path, expected_rows, chunk_bytes=1 << 20):
    """`model_io.fetch_pinned`'s `verify_dest` hook for GRAMS files: first
    runs `_validate_grams_fits_structure` (raises on a structurally bad
    file), then returns the plain sha256 hex digest of `path`'s raw bytes.
    No gzip postprocess for GRAMS (module docstring), so the dest bytes ARE
    the pinned content -- this differs from sps_curate's CIFIST hook, which
    must decompress before hashing."""
    _validate_grams_fits_structure(path, expected_rows)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _with_retry(func, *, max_retries=4, base_delay_s=5.0, verbose=True):
    """Call func() with exponential backoff on a NETWORK failure. Ported
    from sps_curate._with_retry (same contract, same exception set): a
    sha256 mismatch or a structural-validation ValueError is NOT retried
    here -- a changed or corrupted artifact is a real event to investigate,
    not a transient condition worth another attempt."""
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


def _bootstrap_fetch(url, dest, expected_rows, timeout, verbose):
    """First-ever fetch of one GRAMS file: download to dest+'.part',
    validate its FITS structure, hash it, pin the hash into the manifest,
    then rename into place. Mirrors sps_curate.fetch_btsettl_cifist's
    bootstrap branch, adapted for a single large file with no gzip step."""
    directory = os.path.dirname(dest)
    if directory:
        os.makedirs(directory, exist_ok=True)
    partial = dest + ".part"

    def _do_download():
        with urllib.request.urlopen(url, timeout=timeout) as response, \
                open(partial, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        return partial

    _with_retry(_do_download, verbose=verbose)

    # Validate BEFORE pinning -- a structurally bad first download must
    # never be recorded as the trusted hash.
    _validate_grams_fits_structure(partial, expected_rows)

    h = hashlib.sha256()
    with open(partial, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha256 = h.hexdigest()
    nbytes = os.path.getsize(partial)

    os.replace(partial, dest)
    if verbose:
        print(f"    PIN ON FIRST FETCH: {os.path.basename(dest)} sha256={sha256} "
              f"({nbytes} bytes)")
    return sha256, nbytes


def fetch_grams(download_dir, *, timeout=600, verbose=True):
    """Stage 0: fetch grams_o.fits and grams_c.fits from figshare (B-1)
    into download_dir, sha256-pinned at data/agb/grams_manifest.json.

    Pin-on-first-fetch, verify-on-every-later-call -- the same semantics as
    sps_curate.fetch_btsettl_cifist / data/sps/btsettl_cifist_manifest.json
    (see the module docstring's "Manifest pinning" section). No gzip
    postprocess: GRAMS is plain FITS, so the file kept at
    download_dir/<filename> is exactly what was hashed.

    Idempotent: a cached file whose sha256 already matches its manifest
    entry AND passes `_validate_grams_fits_structure` is left untouched
    with no network request (model_io.fetch_pinned's own cache-hit path).
    Network failures are retried with exponential backoff (_with_retry); a
    sha256 mismatch or a structural-validation failure is NOT retried --
    both are real events to investigate, never resolved by silently
    re-pinning.

    Returns {"grams_o.fits": path, "grams_c.fits": path}.
    """
    download_dir = str(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    manifest = _load_grams_manifest()
    paths = {}

    for filename, url in GRAMS_FILES.items():
        dest = os.path.join(download_dir, filename)
        expected_rows = GRAMS_EXPECTED_ROWS[filename]
        entry = manifest.get(filename)

        if entry is not None:
            verify_dest = functools.partial(_grams_verify_dest,
                                            expected_rows=expected_rows)

            def _do_fetch(url=url, entry=entry, dest=dest, verify_dest=verify_dest):
                return model_io.fetch_pinned(
                    url, entry["sha256"], dest, timeout=timeout, verify_dest=verify_dest)

            _with_retry(_do_fetch, verbose=verbose)
            if verbose:
                print(f"    {filename}: verified against pinned sha256={entry['sha256']}")
        else:
            if verbose:
                print(f"    {filename}: NO MANIFEST ENTRY -- bootstrapping the pin "
                      f"(unpinned this run; data/agb/grams_manifest.json will record "
                      f"the sha256 from here on).")
            sha256, nbytes = _bootstrap_fetch(url, dest, expected_rows, timeout, verbose)
            manifest[filename] = {
                "sha256": sha256, "nbytes": nbytes,
                "expected_rows": expected_rows, "url": url,
            }
            _save_grams_manifest(manifest)

        paths[filename] = dest

    return paths


def purge_downloads(download_dir):
    """Delete the fetched GRAMS FITS files (grams_o.fits, grams_c.fits)
    under download_dir/. Mirrors sps_curate.purge_downloads's contract --
    safe (nothing deleted here is a source of truth; both files are
    reproducible from data/agb/grams_manifest.json's sha256 pins) and
    idempotent (a missing directory or missing files are not errors).
    Scoped to the two named files rather than an rmtree of the whole
    directory; see the module docstring for why. Returns download_dir
    (str).
    """
    download_dir = str(download_dir)
    for filename in GRAMS_FILES:
        path = os.path.join(download_dir, filename)
        if os.path.exists(path):
            os.remove(path)
    return download_dir


def load_grams_table(path):
    """Astropy Table of one GRAMS FITS file's HDU1 (the model table) -- a
    thin wrapper so every later stage (subset selection, unit conversion,
    interpolation -- T4) reads GRAMS through one entry point rather than
    each re-opening the file itself."""
    from astropy.table import Table
    return Table.read(str(path), hdu=1)


# ====================================================================
# Stage 1 -- select (B-2/B-3/B-4)
# ====================================================================

BANDS8 = ("J", "H", "Ks", "I1", "I2", "I3", "I4", "M1")

# GRAMS' 82-filter synthetic-photometry block carries, at columns 4..11
# (0-indexed), exactly 2MASSJ, 2MASSH, 2MASSK, IRAC36, IRAC45, IRAC58,
# IRAC80, MIPS24 -- contiguous, and in BANDS8 order. Verified against the
# FITS header's own filter-order COMMENT block and against
# Fphot*10**(mphot/2.5), which reproduces the project's own Vega zero
# points (see locus_study/scripts/grams_lib.py's module docstring). This
# index array is CONSUMED both by Stage 1's thinning (on `mphot`) and by
# Stage 5's Fphot validation (on `Fphot`), so the two must agree with each
# other -- they do, by construction, since both read this one constant.
GRAMS_PHOT_IDX = np.arange(4, 12)

# B-2: greedy Chebyshev thinning tolerance for the C-rich grid, in the 7
# adjacent Vega colours.
AGB_THIN_TOL_MAG = 0.05

# B-3: the O-rich L node nearest this luminosity is kept (one L per shape).
AGB_ORICH_TARGET_L_SUN = 5000.0

# B-4 (T1 locus study, locus_study/README.md §1): the lowest tau node at
# which the dust excess exceeds 0.30 mag in EVERY (Teff, R_in) combination
# present at that node and above.
AGB_TAU10_MIN_ORICH = 0.0128
AGB_TAU11_3_MIN_CRICH = 0.02

# CURATION_2026-10-09.md sec 6 / lib_6_AGB.md concern 4: "the population's
# own cut" -- the STAR family's TRILEGAL evolved-star selector
# (bms_prior/class_densities/class_density_star_family.py EVOLVED_LOGTE_MAX
# = 3.6, i.e. log10 T_eff < 3.6) is also the spec's own definition of the
# population this library describes (agb_curation_plan.md line 194 /
# studies/agb/grams_characterisation/README.md Sec.2.1: "evolved =
# logg<1.0 AND logTe<3.6 (T_eff<3981K) AND logL>3.0"). GRAMS ships four
# O-rich nodes (4100/4300/4500/4700 K) and one C-rich node (4000 K) past
# this cut -- 88 of 1980 rows -- templates the population this library
# targets never produces. Removed here, not relabelled. Read from the one
# place (`10 ** EVOLVED_LOGTE_MAX`), not a hand-rounded literal: a rounded
# 3981.0 is STRICTER than the population's own 3981.0717..., so a row at
# T_eff=3981.07 would pass the prior's cut and fail this one.
AGB_EVOLVED_TEFF_MAX_K = 10 ** _EVOLVED_LOGTE_MAX


def greedy_thin_fast(cols, tol):
    """Greedy Chebyshev thinning: walk rows in FILE ORDER, discard row i
    iff it lies within `tol` of an already-kept row in EVERY column
    simultaneously. Line-for-line port of
    locus_study/scripts/grams_lib.py:greedy_thin_fast (T1's reference
    implementation, B-2) -- verified there to give bit-identical results
    to the naive O(N^2) python-loop form (`greedy_thin`) on the first 3000
    C-rich rows. Reproduces T1's count exactly (C-rich, tol=0.05 -> 1551),
    not the earlier order-dependent 1567 quoted in
    grams_characterisation.md §1.6 -- see locus_study/README.md §1.1
    ("greedy thinning is order-dependent... pin the code, not the number").

    Returns the indices (into `cols`' row axis) that are KEPT, ascending.
    """
    n = len(cols)
    kept_idx = []
    alive = np.ones(n, dtype=bool)
    for i in range(n):
        if not alive[i]:
            continue
        kept_idx.append(i)
        keptv = cols[i]
        rest = np.arange(i + 1, n)
        rest = rest[alive[rest]]
        if len(rest):
            d = np.max(np.abs(cols[rest] - keptv), axis=1)
            alive[rest[d <= tol]] = False
    return np.array(kept_idx, dtype=int)


def select_orich_shapes(table, target_l_sun=AGB_ORICH_TARGET_L_SUN):
    """B-3: collapse the O-rich grid's 68,600 rows to its 1225 distinct
    (Teff, R_in, tau10) SHAPES, keeping the L node nearest `target_l_sun`
    for each -- L is an exact multiplicative scale in this grid
    (grams_characterisation.md §1.5 item 1: F_nu ratio between two rows
    differing only in L equals the L ratio to 1 part in 1e15), so any L
    node preserves the shape's colour exactly. Returns a sorted int array
    of row indices into `table`, length == number of distinct shapes
    (expected 1225)."""
    teff = np.asarray(table["Teff"], dtype=float)
    rin = np.asarray(table["Rin"], dtype=float)
    tau = np.asarray(table["tau10"], dtype=float)
    lum = np.asarray(table["Lum"], dtype=float)

    key = np.stack([teff, rin, tau], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)

    sel = np.full(len(uniq), -1, dtype=int)
    order = np.argsort(np.abs(np.log(lum / target_l_sun)))
    for i in order:                     # first hit per shape wins -> nearest L
        if sel[inv[i]] < 0:
            sel[inv[i]] = i
    if (sel < 0).any():
        raise AssertionError("select_orich_shapes: some shape has no L node assigned")
    return np.sort(sel)


def select_crich_thin(table, tol=AGB_THIN_TOL_MAG):
    """B-2: greedily thin every C-rich row (all 12,243) at `tol` mag
    Chebyshev distance in the 7 adjacent Vega colours, computed from the
    grid's own `mphot`. Returns a sorted int array of kept row indices
    (expected 1551 at tol=0.05)."""
    mphot = np.asarray(table["mphot"], dtype=float)[:, GRAMS_PHOT_IDX]
    adjacent = mphot[:, :-1] - mphot[:, 1:]     # J-H, H-Ks, ..., I4-M1
    kept = greedy_thin_fast(adjacent, tol)
    return np.sort(kept)


def apply_b4_tau_cut(table, row_idx, tau_column, tau_min, rtol=1e-5):
    """B-4: keep only the `row_idx` rows whose `tau_column` >= tau_min.

    `tau_min` is a grid NODE value (e.g. 0.0128), but the column is stored
    float32 in the source FITS -- measured: the 0.0128 node reads back as
    0.012799999676644802 (3.2e-10 below the float64 literal), which a bare
    `>=` would silently exclude, dropping an entire extra tau node's worth
    of shapes (42, for O-rich). `rtol` guards against exactly this
    representation gap without loosening the cut to the next grid node
    down (the next O-rich node is 0.0064, 2x smaller -- far outside any
    plausible float32 rounding).

    Returns (kept_idx, n_dropped)."""
    row_idx = np.asarray(row_idx)
    tau = np.asarray(table[tau_column], dtype=float)[row_idx]
    keep_mask = tau >= tau_min * (1.0 - rtol)
    return row_idx[keep_mask], int((~keep_mask).sum())


def select_agb_rows(o_table, c_table, verbose=True):
    """Stage 1: the full B-2/B-3/B-4 selection for both chemistries.

    Returns {"o_idx", "c_idx", "report"}: `o_idx`/`c_idx` are row indices
    into `o_table`/`c_table` respectively, AFTER the B-4 cut (i.e. ready
    for Stage 2). `report` carries the counts at each step (logged here
    and returned for the driver to print again in its own summary).
    """
    o_shapes = select_orich_shapes(o_table)
    n_o_shapes = len(o_shapes)
    o_idx, n_o_dropped = apply_b4_tau_cut(o_table, o_shapes, "tau10", AGB_TAU10_MIN_ORICH)

    c_thin = select_crich_thin(c_table)
    n_c_thin = len(c_thin)
    c_idx, n_c_dropped = apply_b4_tau_cut(c_table, c_thin, "tau11_3", AGB_TAU11_3_MIN_CRICH)

    report = dict(
        n_o_native=len(o_table), n_o_shapes=n_o_shapes,
        n_o_b4_dropped=n_o_dropped, n_o_selected=len(o_idx),
        n_c_native=len(c_table), n_c_thinned=n_c_thin,
        n_c_b4_dropped=n_c_dropped, n_c_selected=len(c_idx),
    )
    if verbose:
        print(f"    stage 1  select: O-rich {report['n_o_native']} rows -> "
              f"{n_o_shapes} shapes (B-3) -> B-4 cut drops {n_o_dropped} "
              f"(tau10<{AGB_TAU10_MIN_ORICH}) -> {len(o_idx)} kept")
        print(f"    stage 1  select: C-rich {report['n_c_native']} rows -> "
              f"{n_c_thin} thinned at {AGB_THIN_TOL_MAG} mag (B-2) -> B-4 cut "
              f"drops {n_c_dropped} (tau11_3<{AGB_TAU11_3_MIN_CRICH}) -> "
              f"{len(c_idx)} kept")
    return dict(o_idx=o_idx, c_idx=c_idx, report=report)


# ====================================================================
# Stage 2 -- convert (B-7/B-8)
# ====================================================================

GRAMS_NATIVE_DISTANCE_KPC = 50.0
# B-2/§3.2 items 1+2: F_mJy = F_Jy * (50 kpc / 1 kpc)^2 * 1000 (Jy->mJy).
AGB_FLUX_SCALE_JY_TO_MJY_AT_1KPC = (GRAMS_NATIVE_DISTANCE_KPC / 1.0) ** 2 * 1000.0  # 2.5e6

# B-7: closure gate on the NATIVE (50 kpc, Jy) spectrum. Measured range
# (grams_characterisation.md §1.4): 0.9988-1.0255 (O), 0.9987-1.0125 (C) --
# inside this gate for every row characterised so far, but the gate is
# still enforced and any row outside it is dropped and logged.
AGB_CLOSURE_MIN = 0.95
AGB_CLOSURE_MAX = 1.05

# B-8: red-tail internal sampling resolution -- same order as sps_curate's
# own CIFIST tail (CIFIST_PLANCK_TAIL_GRID_R=5000): the tail is smooth (a
# pure power law), so it is deliberately oversampled relative to
# GRID_R=300 before being handed to the same log-log interpolation.
AGB_RED_TAIL_GRID_R = 5000

# Matches sps_curate.GRID_WAVELENGTH_MAX_UM -- the common grid's red edge.
GRID_WAVELENGTH_MAX_UM = 1e6


def _agb_loggrid(wave_min_um, wave_max_um, r):
    """Log-uniform ASCENDING grid, R = lambda/Delta-lambda = r. Ported
    from sps_curate._loggrid (leading-underscore, module-private there);
    duplicated rather than imported so this module has no dependency on
    another module's private internals."""
    n = int(np.ceil(np.log(wave_max_um / wave_min_um) * r)) + 1
    return np.exp(np.linspace(np.log(wave_min_um), np.log(wave_max_um), n))


def native_closure_ratio(wave_um_asc, fspec_jy_asc, lum_lsun,
                         distance_kpc=GRAMS_NATIVE_DISTANCE_KPC):
    """B-7: integral(F_nu dnu) on the NATIVE 50 kpc spectrum (Jy), over
    L/(4 pi d^2) -- reproduces grams_characterisation.md §1.4's exact
    check. `wave_um_asc`/`fspec_jy_asc` ascending in wavelength."""
    nu_asc = C_UM_S / np.asarray(wave_um_asc, dtype=float)[::-1]
    f_cgs_asc = np.asarray(fspec_jy_asc, dtype=float)[::-1] * 1e-23   # Jy -> erg/s/cm2/Hz
    f_bol = np.trapz(f_cgs_asc, nu_asc)
    d_cm = distance_kpc * REFERENCE_DISTANCE_CM
    predicted = lum_lsun * L_SUN_ERG_S / (4.0 * np.pi * d_cm ** 2)
    return float(f_bol / predicted)


def _agb_red_tail_beta(wave_um_asc, fnu_asc):
    """B-8: beta from the spectrum's last two NATIVE nodes, for
    F_nu ~ nu^(2+beta), i.e. F_nu ~ lambda^-(2+beta). Clipped to [0,2]."""
    lam1, lam2 = wave_um_asc[-2], wave_um_asc[-1]
    f1, f2 = fnu_asc[-2], fnu_asc[-1]
    if not (f1 > 0 and f2 > 0):
        return 0.0
    slope = (np.log(f2) - np.log(f1)) / (np.log(lam2) - np.log(lam1))  # dlnF/dlnlambda
    beta = -slope - 2.0
    return float(np.clip(beta, 0.0, 2.0))


def _append_agb_red_tail(wave_um_asc, fnu_asc, beta,
                         wave_max_um=GRID_WAVELENGTH_MAX_UM,
                         tail_grid_r=AGB_RED_TAIL_GRID_R):
    """B-8: append an F_nu ~ lambda^-(2+beta) tail beyond the native red
    edge, out to wave_max_um. Bookkeeping only -- no SESNA filter reaches
    either grid edge (native blue edge 0.20/0.234 um or grid edges
    0.009/1e6 um)."""
    wave_max_native = wave_um_asc[-1]
    if wave_max_native >= wave_max_um:
        return wave_um_asc, fnu_asc
    tail_wave = _agb_loggrid(wave_max_native, wave_max_um, tail_grid_r)[1:]
    f_edge = fnu_asc[-1]
    tail_flux = f_edge * (tail_wave / wave_max_native) ** (-(2.0 + beta))
    return (np.concatenate([wave_um_asc, tail_wave]),
            np.concatenate([fnu_asc, tail_flux]))


def build_agb_row_on_grid(wave_um_native_asc, fnu_mjy_native_asc, common_freq_hz_desc):
    """One AGB row on the common R=300 grid: append the B-8 red tail, then
    log-log interpolate (sps_curate.interp_loglog_zero_preserving, which
    zero-fills everything blueward of native coverage -- the CK03
    zero-preservation rule, reused verbatim, IS the B-8 blue-fill policy;
    no separate blue-fill code is needed). Returns (values_desc, beta)."""
    beta = _agb_red_tail_beta(wave_um_native_asc, fnu_mjy_native_asc)
    wave_ext, flux_ext = _append_agb_red_tail(wave_um_native_asc, fnu_mjy_native_asc, beta)
    nu_in_asc = C_UM_S / wave_ext[::-1]
    f_in_asc = flux_ext[::-1]
    values_desc = interp_loglog_zero_preserving(common_freq_hz_desc, nu_in_asc, f_in_asc)
    return values_desc, beta


def build_agb_rows(o_table, c_table, o_idx, c_idx, common_freq_hz_desc, verbose=True):
    """Stage 2: convert + B-7 gate + B-8 grid every selected row of both
    chemistries.

    Returns (rows, closure_dropped, report):
      rows            -- list of dicts, one per SURVIVING row: {chem,
                         source_row, teff, logg, tau, tau_wave_um, rin,
                         l_sun, t_in, mlr_dust, closure_native, beta,
                         values_desc}
      closure_dropped -- list of dicts describing every row the B-7 gate
                         dropped: {chem, source_row, teff, rin, tau, ratio}
      report          -- counts, for the driver to log
    """
    rows = []
    closure_dropped = []

    specs = (
        ("O", o_table, o_idx, "tau10", 10.0),
        ("C", c_table, c_idx, "tau11_3", 11.3),
    )
    for chem, table, idx, tau_col, tau_wave_um in specs:
        idx = np.asarray(idx)
        lam_native = np.asarray(table["Lspec"][0], dtype=float)   # identical every row (B-1)
        teff = np.asarray(table["Teff"][idx], dtype=float)
        logg = np.asarray(table["logg"][idx], dtype=float)
        rin = np.asarray(table["Rin"][idx], dtype=float)
        tau = np.asarray(table[tau_col][idx], dtype=float)
        lum = np.asarray(table["Lum"][idx], dtype=float)
        tin = np.asarray(table["Tin"][idx], dtype=float)
        mlr = np.asarray(table["MLR"][idx], dtype=float)
        fspec_jy = np.asarray(table["Fspec"][idx], dtype=float)   # (len(idx), nlam)

        for k, i in enumerate(idx):
            ratio = native_closure_ratio(lam_native, fspec_jy[k], lum[k])
            if not (AGB_CLOSURE_MIN <= ratio <= AGB_CLOSURE_MAX):
                closure_dropped.append(dict(
                    chem=chem, source_row=int(i), teff=float(teff[k]),
                    rin=float(rin[k]), tau=float(tau[k]), ratio=ratio))
                continue

            fnu_mjy_native = fspec_jy[k] * AGB_FLUX_SCALE_JY_TO_MJY_AT_1KPC
            values_desc, beta = build_agb_row_on_grid(
                lam_native, fnu_mjy_native, common_freq_hz_desc)

            rows.append(dict(
                chem=chem, source_row=int(i), teff=float(teff[k]), logg=float(logg[k]),
                tau=float(tau[k]), tau_wave_um=tau_wave_um, rin=float(rin[k]),
                l_sun=float(lum[k]), t_in=float(tin[k]), mlr_dust=float(mlr[k]),
                closure_native=ratio, beta=beta,
                values_desc=values_desc.astype(np.float32),
            ))

    report = dict(n_closure_dropped=len(closure_dropped), n_converted=len(rows))
    if verbose:
        print(f"    stage 2  convert: {len(rows)} rows converted + gridded; "
              f"B-7 closure gate dropped {len(closure_dropped)} "
              f"(outside [{AGB_CLOSURE_MIN},{AGB_CLOSURE_MAX}])")
        for d in closure_dropped:
            print(f"        DROPPED {d['chem']} teff={d['teff']:.0f} rin={d['rin']:.2f} "
                 f"tau={d['tau']:.4g} closure={d['ratio']:.4f}")
    return rows, closure_dropped, report


# ====================================================================
# Stage 2.5 -- population cut
# (CURATION_2026-10-09.md sec 6 item 2; lib_6_AGB.md concern 4)
# ====================================================================

def apply_evolved_teff_cut(rows, teff_max=AGB_EVOLVED_TEFF_MAX_K, verbose=True):
    """Drop every row with T_EFF > `teff_max` -- the population's own
    evolved-star selector (`AGB_EVOLVED_TEFF_MAX_K`'s docstring), not a
    library-side choice. Expected to drop 67 O-rich (4100/4300/4500/4700 K)
    + 21 C-rich (4000 K) = 88 of the then-1980 native rows.

    Returns (kept_rows, report): `report` carries the before/after counts
    and the per-chemistry drop breakdown, for the driver to log.
    """
    kept, dropped = [], []
    for r in rows:
        (kept if r["teff"] <= teff_max else dropped).append(r)

    by_chem_teff = defaultdict(int)
    for r in dropped:
        by_chem_teff[(r["chem"], r["teff"])] += 1

    report = dict(
        n_before=len(rows), n_dropped=len(dropped), n_after=len(kept),
        teff_max=teff_max,
        dropped_by_chem_teff={k: v for k, v in sorted(by_chem_teff.items())},
    )
    if verbose:
        print(f"    stage 2.5 teff cut: {report['n_before']} -> drop "
              f"{report['n_dropped']} (T_EFF>{teff_max:g}K, the population's own "
              f"evolved-star selector) -> {report['n_after']} kept")
        for (chem, teff), n in report["dropped_by_chem_teff"].items():
            print(f"        DROPPED {n} {chem} rows at T_EFF={teff:.0f}K")
    return kept, report


# ====================================================================
# Stage 2.75 -- sampling at the fitter's resolution (owner library spec,
# 2026-10-08, with the 2026-10-08 greedy-max-coverage correction below).
# One global deterministic greedy r-net over the 1,892 curated rows,
# radius = one noise length (sigma_eff of sed_models_register.density's
# quotient space), no interpolation, no strata, no budget -- the kept
# count is whatever the covering yields.
# ====================================================================


def agb_band8_log10_fluxes(rows, wave_um_desc, bands=BANDS8):
    """(n, 8) log10 band flux array, one row per element of `rows`, in
    `bands` order -- `reference_convolve` applied to each row's own
    common-grid spectrum (`values_desc`), the SAME convolution Stage 5's
    validator and the register's own reference pipeline use. This is the
    "8 log10 band fluxes" the owner's sampling spec projects into the
    quotient space; it is computed directly from the curated spectra, not
    from any already-written convolved/*.fits (none exist yet at this
    point in the pipeline -- Stage 2.75 runs before Stage 4/5)."""
    bands = tuple(bands)
    wave_um_asc = np.asarray(wave_um_desc, dtype=float)[::-1]
    out = np.empty((len(rows), len(bands)), dtype=float)
    for i, row in enumerate(rows):
        fnu_asc = row["values_desc"][::-1]
        out[i] = [reference_convolve(wave_um_asc, fnu_asc, b) for b in bands]
    bad = ~np.all(np.isfinite(out) & (out > 0), axis=1)
    if bad.any():
        bad_names = [rows[i].get("name", rows[i]["source_row"]) for i in np.flatnonzero(bad)[:5]]
        raise ValueError(
            f"agb_band8_log10_fluxes: {int(bad.sum())} row(s) have a non-finite or "
            f"non-positive band flux, e.g. {bad_names}; cannot take log10")
    return np.log10(out)


def greedy_rnet(coords, radius):
    """Deterministic greedy MAXIMUM-COVERAGE r-net at `radius` (owner
    correction, 2026-10-08, superseding an earlier random-order /
    medoid-repair draft). At each step, among all currently UNCOVERED
    points, pick the one with the MOST UNCOVERED NEIGHBOURS within
    `radius` (a point counts as its own neighbour, so an isolated point
    still has count >= 1); keep it; mark every point within `radius` of
    it covered; repeat until nothing remains uncovered. Ties are broken
    by the LOWEST row index -- the caller passes `coords` already in the
    project's own ASCII-lexicographic MODEL_NAME order (D-12/D-16), so
    "lowest row index" IS "earliest MODEL_NAME": a fixed total order,
    with NO RANDOM SEED. (A seed would add nothing: the rule plus this
    tie-break is already fully deterministic, and dropping the seed is
    itself more reproducible than carrying one that does no work.)

    This single rule gives BOTH of the owner spec's item-2 conditions BY
    CONSTRUCTION, chosen centrally from the start rather than repaired
    afterward:
      covering -- the loop only terminates once every point is covered;
      packing  -- a point is only ever picked while still uncovered, i.e.
                  strictly more than `radius` from every already-kept
                  point (a nearer point would already have been marked
                  covered when that earlier point was kept), so no two
                  kept points can end up closer than `radius`.

    Implementation: one cKDTree built once gives every point's within-
    `radius` neighbour list; an uncovered-neighbour-count array is
    decremented in place as points become covered, rather than
    recomputed from scratch each step, keeping this near-linear at this
    library's scale (~1,900 rows).

    Returns an ascending int array of kept positions into `coords`'s row
    axis.
    """
    coords = np.ascontiguousarray(coords, dtype=float)
    n = coords.shape[0]
    tree = cKDTree(coords)
    neighbor_lists = [np.asarray(idx, dtype=np.int64)
                      for idx in tree.query_ball_point(coords, radius)]
    uncovered_count = np.array([len(nb) for nb in neighbor_lists], dtype=np.int64)
    covered = np.zeros(n, dtype=bool)

    kept = []
    n_remaining = n
    while n_remaining > 0:
        active_idx = np.flatnonzero(~covered)
        counts = uncovered_count[active_idx]
        best_count = counts.max()
        pick = int(active_idx[counts == best_count].min())   # tie-break: lowest row index
        kept.append(pick)
        nb = neighbor_lists[pick]
        newly = nb[~covered[nb]]
        covered[newly] = True
        n_remaining -= len(newly)
        for p in newly:
            for q in neighbor_lists[p]:
                uncovered_count[q] -= 1
    return np.sort(np.asarray(kept, dtype=int))


def sample_agb_templates(rows, wave_um_desc, sigma_log_by_band, bands=BANDS8, verbose=True):
    """Stage 2.75: the owner library spec's sampling step (2026-10-08
    greedy-maximum-coverage correction -- see `greedy_rnet`'s docstring
    for why there is no medoid-repair pass and no seed).

    `rows` MUST already be sorted into ASCII-lexicographic MODEL_NAME
    order (the orchestrator assigns names and sorts before calling this)
    so that `greedy_rnet`'s tie-break -- lowest row index -- is that
    fixed total order rather than an arbitrary one.

    `sigma_log_by_band` is the SAMPLING-SCALE vector the caller supplies
    -- `constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR` in production (owner
    ruling, 2026-10-08): TWICE the surveys' own published
    absolute-calibration systematics floor, a derived literature constant
    (see that constant's docstring for the factor-of-2 derivation), NOT
    read from any sed_models_register product (an earlier catalogue-
    derived SIGMA_LOG read was retired by the same ruling -- its source
    files no longer exist). This function is agnostic to where the vector
    came from; it only requires BANDS8 order and shape.

    Builds the 5-D quotient space from `sigma_log_by_band`
    (`sed_models_register.density.build_quotient_space` -- READ-ONLY use
    of the register's own projector/basis machinery; no register product
    is read or written here), projects every one of the 1,892 curated
    rows' 8 log10 band fluxes into it, runs `greedy_rnet` at radius
    `space.sigma_eff` -- one noise length -- to get the KEPT templates
    (centrally placed in their eventual represented sets by construction,
    since each is chosen while it still has the most uncovered
    neighbours), and ONLY THEN assigns every raw row to its NEAREST kept
    template (Voronoi assignment) to form the represented sets -- derived
    from the representatives, never the other way round.

    Returns (kept_idx, member_of, nearest_dist, report):
      kept_idx     -- ascending int array into `rows`, the KEPT templates
      member_of    -- (len(rows),) int array, each row's assigned
                      template expressed as an index into `rows`
                      (member_of[kept_idx] == kept_idx, each kept
                      template is its own member)
      nearest_dist -- (len(rows),) float array, quotient-space distance
                      to the assigned template
      report       -- the kept count, the measured covering radius (max
                      member-to-representative distance, as a fraction
                      of sigma_eff -- verified <=1.0), the measured
                      minimum inter-representative distance (verified
                      >=1.0 sigma_eff), and the member-to-representative
                      distance distribution in sigma_eff units
    """
    from sesnaimpute.sed_models.register import density as _density

    bands = tuple(bands)
    sigma_log_by_band = np.asarray(sigma_log_by_band, dtype=float)
    if sigma_log_by_band.shape != (len(bands),):
        raise ValueError(
            f"sigma_log_by_band must be shape ({len(bands)},) in {bands} order, "
            f"got {sigma_log_by_band.shape}")

    log10_flux8 = agb_band8_log10_fluxes(rows, wave_um_desc, bands=bands)

    space = _density.build_quotient_space(
        laws=_density.DEFAULT_LAWS, sigma_log=sigma_log_by_band,
        sigma_source="constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR (2x survey "
                     "absolute-calibration systematics, literature + derivation) -- "
                     "sed_models_curate.build.agb's sigma_log_vector()")
    coords = space.project(log10_flux8)
    radius = space.sigma_eff

    kept_idx = greedy_rnet(coords, radius)

    # Voronoi assignment: every raw row to its nearest kept template.
    # Derived from the representatives, never the other way round.
    tree_kept = cKDTree(coords[kept_idx])
    nearest_dist, nearest_j = tree_kept.query(coords, k=1)
    member_of = kept_idx[nearest_j]

    covering_radius_sigeff = float(np.max(nearest_dist) / radius)
    coverage_fraction = float(np.mean(nearest_dist <= radius * (1.0 + 1e-9)))
    if covering_radius_sigeff > 1.0 + 1e-6:
        raise AssertionError(
            f"sample_agb_templates: covering radius {covering_radius_sigeff:.6f} "
            "sigma_eff exceeds 1.0 -- greedy_rnet's covering guarantee failed")

    if len(kept_idx) > 1:
        kept_coords = coords[kept_idx]
        ktree = cKDTree(kept_coords)
        dd, _ = ktree.query(kept_coords, k=2)
        min_pairwise_kept_distance = float(np.min(dd[:, 1]))
    else:
        min_pairwise_kept_distance = float("inf")
    min_pairwise_sigeff = min_pairwise_kept_distance / radius
    if min_pairwise_sigeff < 1.0 - 1e-6:
        raise AssertionError(
            f"sample_agb_templates: min pairwise kept-template distance "
            f"{min_pairwise_sigeff:.6f} sigma_eff is below 1.0 -- greedy_rnet's "
            "packing guarantee failed")

    dist_sigeff = nearest_dist / radius

    report = dict(
        n_raw=len(rows), n_kept=int(len(kept_idx)),
        quotient_dim=int(space.d), radius_sigeff=float(radius),
        laws=tuple(space.laws),
        sigma_log_by_band={b: float(s) for b, s in zip(bands, sigma_log_by_band)},
        coverage_fraction=coverage_fraction,
        covering_radius_sigeff=covering_radius_sigeff,
        min_pairwise_kept_distance=min_pairwise_kept_distance,
        min_pairwise_kept_distance_sigeff=min_pairwise_sigeff,
        nearest_dist_median=float(np.median(nearest_dist)),
        nearest_dist_p90=float(np.percentile(nearest_dist, 90)),
        member_dist_sigeff_median=float(np.median(dist_sigeff)),
        member_dist_sigeff_max=float(np.max(dist_sigeff)),
        selection_rule="greedy maximum-uncovered-coverage set cover over the quotient "
                      "space; tie-break = lowest row index in ASCII-lexicographic "
                      "MODEL_NAME order; deterministic, no seed",
    )
    if verbose:
        print(f"    stage 2.75 sample: quotient space d={space.d}, "
              f"sigma_eff={radius:.6f} dex (laws={space.laws})")
        print(f"        greedy max-coverage r-net over {len(rows)} raw rows -> "
              f"{len(kept_idx)} kept templates ({report['selection_rule']})")
        print(f"        covering radius: {covering_radius_sigeff:.6f} sigma_eff "
              f"(must be <=1.0); min inter-representative distance: "
              f"{min_pairwise_sigeff:.6f} sigma_eff (must be >=1.0)")
        print(f"        coverage identity (fraction of raw rows within sigma_eff of "
              f"their assigned template): {coverage_fraction:.6f}")
        print(f"        member-to-representative distance: median "
              f"{report['member_dist_sigeff_median']:.3f} sigma_eff, max "
              f"{report['member_dist_sigeff_max']:.3f} sigma_eff")
    return kept_idx, member_of, nearest_dist, report


# ====================================================================
# Stage 3 -- parameters (B-9)
# ====================================================================

# B-9: [Z/H] is fixed PER CHEMISTRY (the photosphere grid's own
# metallicity, not a per-row axis): O-rich PHOENIX log(Z/Zsun)=-0.5
# (frozen, grams_characterisation.md §1.4); C-rich COMARCS Z/Zsun=0.33 ->
# log10(0.33) = -0.4815 (quoted -0.48 in agb_decisions.md B-9).
AGB_ZH = {"O": -0.5, "C": float(np.log10(0.33))}

AGB_SOURCE = {"O": "GRAMS-O", "C": "GRAMS-C"}


def _agb_rin_code(rin):
    """Compact R_in code: `r` + the decimal point stripped from `%g` --
    injective on both grids' actual R_in sets (O: {3,7,11,15} ->
    {'3','7','11','15'}; C: {1.5,3,4.5,7,12} -> {'15','3','45','7','12'};
    no collisions within either set), zero-padded to 2 digits. Matches the
    `r11`/`r03` codes in B-9's worked examples."""
    return "r" + ("%g" % rin).replace(".", "").zfill(2)


def agb_model_name(chem, teff, rin, tau, p_index=None, tau_decimals=2):
    """B-9: `grams{O|C}_t{Teff}_r{Rin}_tau{tau:.<tau_decimals>e}`
    (`gramsO_t3300_r11_tau1.00e+00` / `gramsC_t3000_r03_tau5.00e-01` for
    the worked examples in the decisions doc), with an optional
    `_p{p_index}` disambiguator appended for a degenerate C-rich
    (Teff, R_in, tau) tuple. If the name would exceed 30 characters with
    the suffix attached, tau's exponential precision is shrunk (down to 0
    decimals) until it fits; raises if it still does not."""
    def _base(decimals):
        return f"grams{chem}_t{int(round(teff))}_{_agb_rin_code(rin)}_tau{tau:.{decimals}e}"

    name = _base(tau_decimals)
    if p_index is None:
        if len(name) > 30:
            raise ValueError(f"AGB model name {name!r} exceeds 30 chars")
        return name

    suffix = f"_p{p_index}"
    decimals = tau_decimals
    while len(name) + len(suffix) > 30 and decimals > 0:
        decimals -= 1
        name = _base(decimals)
    name = name + suffix
    if len(name) > 30:
        raise ValueError(f"AGB model name {name!r} exceeds 30 chars "
                         "even at minimum tau precision")
    return name


def assign_agb_names(rows):
    """B-9: assign a MODEL_NAME to every row of `rows` (in place). `_pN` is
    appended ONLY to rows sharing an exact (chem, Teff, R_in, tau) tuple
    with another row -- degenerate only in the C-rich grid, where multiple
    photospheres (logg, Mass, C2O) sit at the same dust-shell axes;
    O-rich rows are unique on (Teff, R_in, tau10) by construction (that IS
    the definition of a "shape", B-3). The p-index is assigned in a
    deterministic order (by each row's native table row index) so a
    rebuild reproduces the same names. Asserts every name is <=30 chars
    and globally unique. Returns `rows`."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        key = (r["chem"], r["teff"], r["rin"], r["tau"])
        groups[key].append(i)

    for key, idxs in groups.items():
        chem, teff, rin, tau = key
        if len(idxs) == 1:
            rows[idxs[0]]["name"] = agb_model_name(chem, teff, rin, tau)
        else:
            idxs_sorted = sorted(idxs, key=lambda i: rows[i]["source_row"])
            for p, i in enumerate(idxs_sorted, start=1):
                rows[i]["name"] = agb_model_name(chem, teff, rin, tau, p_index=p)

    names = [r["name"] for r in rows]
    too_long = [n for n in names if len(n) > 30]
    if too_long:
        raise ValueError(f"AGB: MODEL_NAME exceeds 30 characters: {too_long[:5]}")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"AGB: MODEL_NAME is not unique: {dupes[:5]}")
    return rows


def write_agb_parameters_fits(rows, output_path, overwrite=True):
    """Stage 3: write parameters.fits with the B-9 widened schema
    (MODEL_NAME 30A, T_EFF, LOGG, Z_H, TAU, TAU_WAVE_UM, R_IN, L_SUN,
    T_IN, MLR_DUST, CHEM 1A, SOURCE 8A -- LOGG/Z_H, not the bracket/
    slash "LOG[G]"/"[Z/H]" this used to carry: HDF5 treats "/" as a
    path separator, so "[Z/H]" written verbatim into the register's
    /models group became a subgroup "[Z" holding a dataset "H]"), in
    `rows`' order (caller sorts by
    name first -- see `curate_agb_model_set`). `rows` here is the KEPT
    template set from Stage 2.75 -- point values only, for schema parity
    with every other library (owner library spec item 3); the full
    represented-set metadata (subclass fractions, parameter ranges,
    counts) lives in members.fits, not here."""
    names = np.array([r["name"] for r in rows])
    teff = np.array([r["teff"] for r in rows], dtype=np.float32)
    logg = np.array([r["logg"] for r in rows], dtype=np.float32)
    zh = np.array([AGB_ZH[r["chem"]] for r in rows], dtype=np.float32)
    tau = np.array([r["tau"] for r in rows], dtype=np.float32)
    tau_wave = np.array([r["tau_wave_um"] for r in rows], dtype=np.float32)
    rin = np.array([r["rin"] for r in rows], dtype=np.float32)
    l_sun = np.array([r["l_sun"] for r in rows], dtype=np.float32)
    t_in = np.array([r["t_in"] for r in rows], dtype=np.float32)
    mlr = np.array([r["mlr_dust"] for r in rows], dtype=np.float32)
    chem = np.array([r["chem"] for r in rows])
    source = np.array([AGB_SOURCE[r["chem"]] for r in rows])

    columns = [
        fits.Column(name="MODEL_NAME", format=MODEL_NAME_COLUMN_FORMAT, array=names),
        fits.Column(name="T_EFF", format="E", unit="K", array=teff),
        fits.Column(name="LOGG", format="E", array=logg),
        fits.Column(name="Z_H", format="E", array=zh),
        fits.Column(name="TAU", format="E", array=tau),
        fits.Column(name="TAU_WAVE_UM", format="E", unit="um", array=tau_wave),
        fits.Column(name="R_IN", format="E", array=rin),
        fits.Column(name="L_SUN", format="E", unit="Lsun", array=l_sun),
        fits.Column(name="T_IN", format="E", unit="K", array=t_in),
        fits.Column(name="MLR_DUST", format="E", unit="Msun/yr", array=mlr),
        fits.Column(name="CHEM", format="1A", array=chem),
        fits.Column(name="SOURCE", format="8A", array=source),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="PARAMETERS")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(output_path, overwrite=overwrite)
    return names


# ====================================================================
# Stage 4 -- write (B-6/B-10)
# ====================================================================

MODELS_CONF_NAME = "GRAMS dusty evolved stars (AGB)"

# B-6: 1% fractional floor everywhere, raised over each affected band's
# OWN filter support (measured from the shipped filter curves -- the
# nonzero-response extent, not the nominal band centre) to the
# per-chemistry sampling error. Values are the validator's measured-maximum
# recommendation (agb_model/validation/README.md F1/§3.3/§3.5): the earlier
# numbers came from a flux-conserving rebin of a single 3000 K spectrum, but
# a coarse RT output grid is a *point sampling*, and the true error is a
# strong function of T_eff (e.g. O-rich H: 4.6% at 2500 K -> 0.2% at 4500 K;
# C-rich H: 10.2% at 2500 K -> 0.5% at 4500 K). These per-chemistry
# constants are pinned at the measured maximum over the library's full
# T_eff range as a conservative floor, not a T_eff-dependent model. G is
# declared even though no G.fits ships (B-11) -- it is a statement about
# what the cube itself would mean if convolved with a G-like filter.
AGB_UNCERTAINTY_FLOOR = 0.01
AGB_BAND_SUPPORT_UM = {
    "G": (0.32, 1.05),
    "J": (1.066, 1.442),
    "H": (1.44, 1.85),
    "Ks": (1.934, 2.384),
    "I3": (4.74421, 6.62251),
}
AGB_BAND_UNCERTAINTY_FRAC = {
    "O": {"G": 0.18, "J": 0.035, "H": 0.046, "Ks": 0.018},
    "C": {"G": 0.35, "J": 0.015, "H": 0.10, "Ks": 0.040, "I3": 0.018},
}


def build_agb_uncertainty_row(values_desc, wave_um_desc, chem):
    """B-6: per-row UNCERTAINTIES -- 1% of |VALUES| everywhere, raised
    over each band's own filter support to this chemistry's measured
    sampling error (still expressed as a fraction of VALUES, so it scales
    with the row's own flux, matching sps's own flat-1%-of-VALUES
    convention). A band with no per-chemistry entry (e.g. I3 for O-rich)
    falls back to the base floor -- the 1% base is already an adequate
    bound there (README §3.5). Overlapping band supports combine via
    np.maximum, not last-writer-wins (F2), so a point inside two windows
    always carries the larger of the two floors."""
    wave_um_desc = np.asarray(wave_um_desc, dtype=float)
    frac = np.full(wave_um_desc.shape, AGB_UNCERTAINTY_FLOOR, dtype=np.float64)
    for band, (lo, hi) in AGB_BAND_SUPPORT_UM.items():
        band_frac = AGB_BAND_UNCERTAINTY_FRAC[chem].get(band, AGB_UNCERTAINTY_FLOOR)
        mask = (wave_um_desc >= lo) & (wave_um_desc <= hi)
        frac[mask] = np.maximum(frac[mask], band_frac)
    return (frac * np.abs(np.asarray(values_desc, dtype=np.float64))).astype(np.float32)


def build_agb_flux_fits(rows, wave_um_desc, freq_hz_desc, output_path, overwrite=True,
                        aperture_au=POINT_SOURCE_APERTURE_AU,
                        distance_cm=REFERENCE_DISTANCE_CM,
                        name_format=MODEL_NAME_COLUMN_FORMAT):
    """Stage 4: flux.fits -- VALUES from each row's Stage-2 `values_desc`,
    UNCERTAINTIES per B-6 (`build_agb_uncertainty_row`). DISTANCE is the
    SAME 1 kpc plug value SPS carries (`constants.REFERENCE_DISTANCE_CM`)
    -- not GRAMS' own 50 kpc, which has already been folded into VALUES by
    Stage 2's unit conversion (B-2/§3.2)."""
    names = np.array([r["name"] for r in rows])
    n_models, n_wav = len(rows), len(wave_um_desc)
    values = np.empty((n_models, 1, n_wav), dtype=np.float32)
    uncertainties = np.empty((n_models, 1, n_wav), dtype=np.float32)
    for i, row in enumerate(rows):
        values[i, 0, :] = row["values_desc"]
        uncertainties[i, 0, :] = build_agb_uncertainty_row(
            row["values_desc"], wave_um_desc, row["chem"])

    return model_io.write_flux_cube(
        output_path,
        names=names,
        wave_um_desc=wave_um_desc,
        freq_hz_desc=freq_hz_desc,
        values=values,
        uncertainties=uncertainties,
        distance_cm=distance_cm,
        distance_comment="FAKE plug value, not real -- same 1 kpc frame as sps (D-3 "
                         "precedent); GRAMS' native 50 kpc is already folded into "
                         "VALUES (Stage 2 / B-2 unit conversion)",
        apertures_au=np.array([aperture_au]),
        valid=None,
        name_format=name_format,
        overwrite=overwrite,
    )


AGB_CLASS_LEGEND = (
    ("AGB", "Dusty evolved star: AGB/RSG photosphere + circumstellar dust shell (GRAMS)"),
)
AGB_SUBCLASS_LEGEND = (
    ("O", "Oxygen-rich: silicate dust, C/O < 1 (GRAMS O-rich, Sargent+2011)"),
    ("C", "Carbon-rich: amorphous carbon + 10% SiC, C/O > 1 (GRAMS C-rich, Srinivasan+2011)"),
)


# CURATION_2026-10-09.md sec 6 item 3 / lib_6_AGB.md concern 9: "no filter
# curve identity, response convention, zero point, grid version... is
# recorded on the register, on parameters.fits, on models.conf or on the
# eight convolved/*.fits" -- verified true of the product on disk before
# this build (no FILT_/ZP_/GRAMS_VER-shaped keyword anywhere under
# sed_models/agb/). Recorded here, on classmap.fits's PRIMARY provenance
# header, the one place this package already writes per-library citation
# strings (sps_curate.py's own SUBCLASS_REF/LEGEND_SOURCE cards are the
# precedent this follows) -- no other library in this package records
# filter/zero-point/grid-version provenance either, so this establishes
# the convention rather than deviating from one.
#
# Filter curves: data/filter_curves/{J,H,Ks,I1,I2,I3,I4,M1}.dat, the SVO
# Filter Profile Service transmission curves `model_convolution.py`'s
# `load_filter_curve` reads for every library's convolution (no AGB-
# specific filter set; same 8 curves sps/h2shock/pahc/galz/yso convolve
# through). Zero points: `constants.VEGA_ZERO_POINT_MJY` -- 2MASS J/H/Ks
# from Cohen, Wheaton & Megeath 2003 AJ 126,1090 (1594/1024/666.7 Jy, the
# same Cohen zero points this package's five other libraries use -- the
# brief's "libraries stay on Cohen's zero points" instruction, not
# revisited here); IRAC I1-I4 from Reach et al. 2005 / the IRAC
# Instrument Handbook (280.9/179.7/115.0/64.13 Jy); MIPS M1 from Rieke
# et al. 2008, AJ 135,2245 (7.17 Jy). Grid version: the figshare DOI's own
# version suffix, `GRAMS_DOI` (".v2"), already pinned byte-for-byte in
# `data/agb/grams_manifest.json` (Stage 0) -- there is no other version
# string GRAMS itself publishes.
AGB_PROVENANCE_FILT_ZP = (
    ("FILT_SOURCE", "SVO Filter Profile Service (8 SESNA bands)"),
    ("FILT_2MASS", "2MASS.J,2MASS.H,2MASS.Ks (J,H,Ks)"),
    ("FILT_IRAC", "IRAC.I1,IRAC.I2,IRAC.I3,IRAC.I4 (I1-I4)"),
    ("FILT_MIPS", "MIPS.24mu (M1)"),
    ("ZP_2MASS_JY", "J=1594.0 H=1024.0 Ks=666.7 (Cohen+2003 AJ126,1090)"),
    ("ZP_IRAC_JY", "I1=280.9 I2=179.7 I3=115.0 I4=64.13 (Reach+2005)"),
    ("ZP_MIPS_JY", "M1=7.17 (Rieke+2008 AJ135,2245)"),
    ("GRAMS_VER", f"figshare {GRAMS_DOI} (Srinivasan)"),
)


def write_agb_classmap_fits(rows, out_path, model_dir=None,
                            name_format=MODEL_NAME_COLUMN_FORMAT):
    """Stage 4: classmap.fits -- CLASS='AGB', SUBCLASS = each row's
    chemistry (B-10), PRIMARY provenance citing the two grid papers (B-1,
    with the ApJ 734,24 citation error corrected -- see the module
    docstring's "Citation correction" note), the CC BY 4.0 licence, and
    (CURATION_2026-10-09.md sec 6 item 3) the filter-curve identity, the
    zero point per band and the GRAMS grid version -- see
    `AGB_PROVENANCE_FILT_ZP`'s docstring for why these live here."""
    names = np.array([r["name"] for r in rows])
    subclass = np.array([r["chem"] for r in rows])
    return model_io.write_classmap_fits(
        out_path,
        names=names,
        class_id="AGB",
        subclass=subclass,
        class_legend=AGB_CLASS_LEGEND,
        subclass_legend=AGB_SUBCLASS_LEGEND,
        provenance=(
            ("CLASS_SOURCE", "library declaration; all models here are this class"),
            ("SUBCLASS_SOURCE", "GRAMS grid of origin: O-rich or C-rich (B-10)"),
            ("SUBCLASS_REF", "Sargent+2011 ApJ 728,93; Srinivasan+2011 A&A 532,A54"),
            ("SUBCLASS_NOTE", "O=oxygen-rich silicate dust; C=carbon-rich AmC+SiC"),
            ("LEGEND_SOURCE", "descriptions authored; see SUBCLASS_REF"),
            ("LICENSE", GRAMS_LICENSE),
        ) + AGB_PROVENANCE_FILT_ZP,
        model_dir=model_dir,
        name_format=name_format,
    )


# Owner library spec item 3 (MEMBERS). Point values (T_EFF, LOGG, TAU,
# R_IN, L_SUN, T_IN, MLR_DUST) stay in parameters.fits' per-model schema
# for parity with every other library; this is the set of columns whose
# per-represented-set MIN_/MED_/MAX_ members.fits carries instead. HDF5-
# safe names throughout (LOGG, not "LOG[G]") -- see write_agb_parameters_
# fits's docstring on why a bracket/slash name cannot survive a write
# into the register's /models group unchanged.
AGB_MEMBER_PARAM_COLUMNS = ("T_EFF", "LOGG", "TAU", "R_IN", "L_SUN", "T_IN", "MLR_DUST")
_AGB_MEMBER_PARAM_ROW_KEY = {
    "T_EFF": "teff", "LOGG": "logg", "TAU": "tau", "R_IN": "rin",
    "L_SUN": "l_sun", "T_IN": "t_in", "MLR_DUST": "mlr_dust",
}


def write_agb_members_fits(all_rows, kept_idx, member_of, nearest_dist, radius_sigeff,
                           output_path, overwrite=True, name_format=MODEL_NAME_COLUMN_FORMAT):
    """Stage 2.75 product: members.fits -- the owner library spec's
    MEMBERS group. `all_rows` is the FULL named, name-sorted set of 1,892
    curated rows (post-revert, pre-/post-sampling alike -- naming does
    not change); `kept_idx`/`member_of`/`nearest_dist` are
    `sample_agb_templates`'s outputs over that same array.

    Two HDUs:

      MEMBERS -- one row per KEPT template (the models-group row count):
        MODEL_NAME, N_MEMBERS (represented-set size, self included),
        FRAC_O / FRAC_C (the represented set's O/C chemistry fractions),
        then MIN_/MED_/MAX_ for each of `AGB_MEMBER_PARAM_COLUMNS` over
        the represented set. Point values themselves are NOT repeated
        here -- they live in parameters.fits, in the kept template's own
        row -- so this table carries only the aggregate the models group
        schema has no room for.

      MEMBERS_DETAIL -- one row per RAW curated row (all 1,892):
        MODEL_NAME (that row's own native name), TEMPLATE_NAME (the kept
        template representing it, joinable to MEMBERS and to
        parameters.fits), CHEM, and DIST_SIGEFF (quotient-space distance
        to that template, in units of sigma_eff -- the sampling radius).
    """
    names_all = np.array([r["name"] for r in all_rows])
    kept_names = names_all[kept_idx]
    template_name_of = names_all[member_of]

    n_kept = len(kept_idx)
    n_members = np.zeros(n_kept, dtype=np.int32)
    frac_o = np.zeros(n_kept, dtype=np.float32)
    frac_c = np.zeros(n_kept, dtype=np.float32)
    stats = {col: dict(min=np.zeros(n_kept, dtype=np.float32),
                       med=np.zeros(n_kept, dtype=np.float32),
                       max=np.zeros(n_kept, dtype=np.float32))
            for col in AGB_MEMBER_PARAM_COLUMNS}

    for g, kept_row_idx in enumerate(kept_idx):
        member_pos = np.flatnonzero(member_of == kept_row_idx)
        members = [all_rows[i] for i in member_pos]
        n_members[g] = len(members)
        chems = np.array([m["chem"] for m in members])
        frac_o[g] = float(np.mean(chems == "O"))
        frac_c[g] = float(np.mean(chems == "C"))
        for col in AGB_MEMBER_PARAM_COLUMNS:
            key = _AGB_MEMBER_PARAM_ROW_KEY[col]
            vals = np.array([m[key] for m in members], dtype=float)
            stats[col]["min"][g] = np.min(vals)
            stats[col]["med"][g] = np.median(vals)
            stats[col]["max"][g] = np.max(vals)

    members_columns = [
        fits.Column(name="MODEL_NAME", format=name_format, array=kept_names),
        fits.Column(name="N_MEMBERS", format="J", array=n_members),
        fits.Column(name="FRAC_O", format="E", array=frac_o),
        fits.Column(name="FRAC_C", format="E", array=frac_c),
    ]
    for col in AGB_MEMBER_PARAM_COLUMNS:
        # col is already an HDF5-safe identifier (AGB_MEMBER_PARAM_COLUMNS);
        # no bracket/slash sanitising needed any more.
        members_columns.append(fits.Column(name=f"MIN_{col}", format="E", array=stats[col]["min"]))
        members_columns.append(fits.Column(name=f"MED_{col}", format="E", array=stats[col]["med"]))
        members_columns.append(fits.Column(name=f"MAX_{col}", format="E", array=stats[col]["max"]))
    members_hdu = fits.BinTableHDU.from_columns(members_columns, name="MEMBERS")

    detail_columns = [
        fits.Column(name="MODEL_NAME", format=name_format, array=names_all),
        fits.Column(name="TEMPLATE_NAME", format=name_format, array=template_name_of),
        fits.Column(name="CHEM", format="1A", array=np.array([r["chem"] for r in all_rows])),
        fits.Column(name="DIST_SIGEFF", format="E",
                    array=(np.asarray(nearest_dist, dtype=np.float64) /
                           float(radius_sigeff)).astype(np.float32)),
    ]
    detail_hdu = fits.BinTableHDU.from_columns(detail_columns, name="MEMBERS_DETAIL")

    primary = fits.PrimaryHDU()
    primary.header["NRAW"] = (len(all_rows),
                              "raw curated rows sampled over (post-revert, 1892)")
    primary.header["NKEPT"] = (n_kept, "kept templates = models-group row count")
    primary.header["SIGEFF"] = (float(radius_sigeff),
                                "sampling radius, one noise length, dex "
                                "(sed_models_register.density sigma_eff)")
    primary.header["SELMETH"] = ("greedy-max-coverage",
                                 "deterministic set-cover r-net, no interpolation")
    fits.HDUList([primary, members_hdu, detail_hdu]).writeto(output_path, overwrite=overwrite)
    return kept_names


# ====================================================================
# Stage 5 -- convolve (B-11/D-9), plus an in-stage validation
# ====================================================================

CONVMETH_NOTE = "sedfitter convolve_model_dir, GRAMS dusty AGB (O+C)"


def _agb_native_spectrum_mjy(table, source_row):
    """(wave_um_asc, fnu_mjy_asc) for one native GRAMS row, in the SAME
    mJy@1kpc convention flux.fits carries (B-2/§3.2 conversion applied),
    for use as the Stage-5 validation reference."""
    lam = np.asarray(table["Lspec"][0], dtype=float)
    fspec_jy = np.asarray(table["Fspec"][source_row], dtype=float)
    return lam, fspec_jy * AGB_FLUX_SCALE_JY_TO_MJY_AT_1KPC


def validate_agb_convolution(model_dir, rows, o_table, c_table, bands=BANDS8,
                             n_sample_per_chem=40, seed=0, verbose=True):
    """Stage 5 validation -- logged, does NOT gate the build (per the task
    spec):

      (a) grid-convolved convolved/{band}.fits values vs `reference_convolve`
          on each sampled row's own NATIVE spectrum (expect <1%, since the
          grid is log-log interpolated from those same native nodes --
          the same comparison Stage A's own R=300 grid standard rests on).
      (b) the same grid-convolved values vs GRAMS' own `Fphot`, for the 8
          bands GRAMS carries at GRAMS_PHOT_IDX (expect 0.05-3% typically,
          excursions to ~8% at extreme tau -- grams_characterisation.md
          §1.4).

    `bands` MUST be in BANDS8 order (the default): that order is what
    GRAMS_PHOT_IDX's Fphot column selection assumes.

    Returns {"reference": {band: array of |rel err|}, "fphot": {band: ...}}.
    """
    bands = tuple(bands)
    if bands != BANDS8:
        raise ValueError(f"validate_agb_convolution: bands must be BANDS8 order, got {bands}")

    rng = np.random.default_rng(seed)
    tables = {"O": o_table, "C": c_table}

    by_chem = defaultdict(list)
    for i, r in enumerate(rows):
        by_chem[r["chem"]].append(i)

    sample_row_idx = []
    for chem, idxs in by_chem.items():
        n = min(n_sample_per_chem, len(idxs))
        sample_row_idx.extend(rng.choice(idxs, size=n, replace=False).tolist())

    grid_flux = {band: dict(zip(*model_io.load_convolved_total_flux_mjy(model_dir, band)))
                for band in bands}

    ref_err = {b: [] for b in bands}
    fphot_err = {b: [] for b in bands}

    for i in sample_row_idx:
        row = rows[i]
        table = tables[row["chem"]]
        source_row = row["source_row"]
        wave_native, fnu_native_mjy = _agb_native_spectrum_mjy(table, source_row)
        fphot_jy = np.asarray(table["Fphot"][source_row], dtype=float)[GRAMS_PHOT_IDX]
        fphot_mjy = fphot_jy * AGB_FLUX_SCALE_JY_TO_MJY_AT_1KPC

        for j, band in enumerate(bands):
            grid_value = grid_flux[band][row["name"]]
            ref_value = reference_convolve(wave_native, fnu_native_mjy, band)
            if ref_value != 0:
                ref_err[band].append(abs(grid_value / ref_value - 1.0))
            fphot_value = fphot_mjy[j]
            if fphot_value > 0:
                fphot_err[band].append(abs(grid_value / fphot_value - 1.0))

    if verbose:
        n_avail = ", ".join(f"{c}={len(v)}" for c, v in sorted(by_chem.items()))
        print(f"    stage 5  validate: {len(sample_row_idx)} sampled rows "
              f"(of {n_avail} available)")
        for band in bands:
            r = np.asarray(ref_err[band])
            f = np.asarray(fphot_err[band])
            r_msg = (f"median={np.median(r):.3%} p95={np.percentile(r, 95):.3%} "
                    f"max={r.max():.3%}") if r.size else "n/a"
            f_msg = (f"median={np.median(f):.3%} p95={np.percentile(f, 95):.3%} "
                    f"max={f.max():.3%}") if f.size else "n/a"
            print(f"        {band:>3s}  vs reference_convolve: {r_msg}   "
                 f"vs GRAMS Fphot: {f_msg}")

    return dict(reference={b: np.asarray(v) for b, v in ref_err.items()},
               fphot={b: np.asarray(v) for b, v in fphot_err.items()})


# ====================================================================
# Orchestrator -- stages 1-5, mirrors sps_curate.curate_sps_model_set
# ====================================================================

def curate_agb_model_set(grams_o_path, grams_c_path, output_root, overwrite=True,
                         convolved_bands=None, verbose=True, sigma_log_by_band=None):
    """Stages 1-5: build the yso-conforming CLASS AGB model set at
    `output_root` from the two GRAMS FITS grids Stage 0 (`fetch_grams`)
    already fetched. Mirrors `sps_curate.curate_sps_model_set`'s shape,
    with Stage 2.75 (`sample_agb_templates`) inserted between the
    population cut and naming-order fixing on one side and the
    point-value writers on the other: the MODELS group (parameters.fits,
    flux.fits, classmap.fits, convolved/) is written for the KEPT
    templates only; members.fits carries the full represented-set
    metadata for all 1,892 curated rows.

    `sigma_log_by_band` : the per-band sampling-scale sigma_log in dex, in
    BANDS8 order -- `constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR` in
    production (owner ruling, 2026-10-08): TWICE the surveys' own
    published absolute-calibration systematics floor (a derived literature
    constant -- see that constant's docstring for the factor-of-2
    derivation), supplied by the caller. This is also
    `sed_models_register.density.build_quotient_space`'s own default as
    of the same ruling. REQUIRED here -- no placeholder is invented if
    omitted. [Supersedes an earlier catalogue-derived
    `sesnacomplete.sed_models_register.noise` SIGMA_LOG read
    (CURATION_2026-10-09.md sec 6 item 1); that product's source files
    have since been deleted, making it a stale artefact -- see
    `sample_agb_templates`'s docstring.] The SAME vector builds Stage
    2.75's quotient space (READ-ONLY use of
    `sed_models_register.density`'s projector machinery; no register
    product is read or written here).

    Returns (output_root, report): `report` merges the Stage-1 selection
    counts, the Stage-2 conversion counts and closure-gate drop list, the
    Stage-2.5 population-cut report, the Stage-2.75 sampling report, the
    raw and kept row counts, and the Stage-5 validation distributions
    (measured on the kept templates).
    """
    if sigma_log_by_band is None:
        raise ValueError(
            "curate_agb_model_set requires sigma_log_by_band -- the sampling-scale "
            "per-band sigma_log vector (constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR "
            "in production), in BANDS8 J,H,Ks,I1,I2,I3,I4,M1 order. No placeholder "
            "is invented here.")
    if convolved_bands is None:
        convolved_bands = CONVOLVED_BAND_NAMES
    output_root = str(output_root)
    os.makedirs(output_root, exist_ok=True)

    if verbose:
        print("  stage 1  select")
    o_table = load_grams_table(grams_o_path)
    c_table = load_grams_table(grams_c_path)
    selection = select_agb_rows(o_table, c_table, verbose=verbose)

    if verbose:
        print("  stage 2  convert")
    wave_um_desc, freq_hz_desc = build_common_wavelength_grid()
    rows, closure_dropped, convert_report = build_agb_rows(
        o_table, c_table, selection["o_idx"], selection["c_idx"],
        freq_hz_desc, verbose=verbose)

    if verbose:
        print("  stage 2.5 population cut")
    rows, teff_cut_report = apply_evolved_teff_cut(rows, verbose=verbose)

    if verbose:
        print("  stage 3  parameters (naming, all raw rows)")
    rows = assign_agb_names(rows)
    rows.sort(key=lambda r: r["name"])   # ASCII-lexicographic row order,
                                         # matching every other library (D-12/D-16);
                                         # also fixes Stage 2.75's tie-break order.

    if verbose:
        print("  stage 2.75 sample")
    kept_idx, member_of, nearest_dist, sample_report = sample_agb_templates(
        rows, wave_um_desc, sigma_log_by_band, verbose=verbose)
    kept_rows = [rows[i] for i in kept_idx]   # already name-sorted (kept_idx ascending)

    write_agb_parameters_fits(kept_rows, os.path.join(output_root, "parameters.fits"),
                              overwrite=overwrite)
    write_agb_members_fits(rows, kept_idx, member_of, nearest_dist,
                           sample_report["radius_sigeff"],
                           os.path.join(output_root, "members.fits"), overwrite=overwrite)

    if verbose:
        print("  stage 4  write")
    model_io.write_models_conf(
        os.path.join(output_root, "models.conf"),
        name=MODELS_CONF_NAME, aperture_dependent=False,
        length_subdir=0, logd_step=LOGD_STEP, version=2)
    build_agb_flux_fits(kept_rows, wave_um_desc, freq_hz_desc,
                        os.path.join(output_root, "flux.fits"), overwrite=overwrite)
    write_agb_classmap_fits(kept_rows, os.path.join(output_root, "classmap.fits"),
                            model_dir=output_root)

    if verbose:
        print("  stage 5  convolve")
    model_names = np.array([r["name"] for r in kept_rows])
    build_convolved_bands(output_root, tuple(convolved_bands), model_names, CONVMETH_NOTE)
    validation = validate_agb_convolution(output_root, kept_rows, o_table, c_table,
                                          verbose=verbose)

    report = dict(
        select=selection["report"], convert=convert_report,
        closure_dropped=closure_dropped,
        teff_cut=teff_cut_report, sample=sample_report,
        n_raw=len(rows), n_final=len(kept_rows),
        validation=validation,
    )
    return output_root, report
