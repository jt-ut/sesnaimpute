"""
yso_aperture.py
====================================================================
The aperture gate: for every model in the grid, the minimum source
distance at which the survey's photometric apertures fall inside the
range the radiative transfer actually resolved.

**The problem this exists to solve.** R17/Richardson's `flux.fits` carries
**NaN** at small apertures for essentially every medium-bearing geometry
-- the RT did not resolve them. The released `convolved/{band}.fits` files
encode that NaN as `0`, which is indistinguishable from a real measurement
of "no flux". Downstream, labeling interpolates its survey-matched mu-SED
between the two grid apertures bracketing the survey aperture, clamps any
zero to the per-model flux floor, and produces a finite, plausible-looking
number with no data behind it.

Measured on the shipped library before this gate existed: **26.3% of the
199,999 selected models carried at least one fabricated band**, and where
one bracket was undefined but the other real, the fabricated value sat a
median **1168x** below the true flux -- ~3 dex against a 0.015-0.034 dex
metric. Those models therefore carry meaningless labels of their own: within
`spubhmi` alone the fabrication rate ran 59.9% for cIII against 0.9% for cII,
and where clean models' labels reproduce Richardson's Stage almost exactly,
fabricated ones do not.

**Scope of the harm, measured rather than assumed.** Fabricated models cluster
in SED space (floor-dominated SEDs are all alike), so they become each other's
nearest neighbours -- but in the fixed-sigma metric they sit astronomically far
from genuine models and carry ~zero kernel weight. Gating before vs. after
labeling changes the surviving models' labels not at all: **99.965% identical
argmax** over the 1,642,685 labeled in both runs, median change in max(p) of
0.0000. So the damage is to the fabricated models' own labels, not to their
neighbours'.

This still runs **before labeling**, for three ordinary reasons: `K = ceil(sqrt(N))`
then tracks the real pool, the ANN index is not padded with 246k junk vectors,
and the labeling pass is not spent on models about to be discarded.

**What "undefined" means, and how it is detected (as of 2026-10-27).**
Before this date, this module could only infer "undefined" indirectly:
the release's own `convolved/{band}.fits` files zero-filled a NaN raw
sample, so an aperture was called undefined iff all 8 convolved bands
read exactly `0` there (validated against the raw `flux.fits` NaN mask
on 400 models x 20 apertures: precision 1.0000, recall 1.0000; a model
with a central star always emits *something* within 100 AU, so
all-8-bands-zero never occurred as a real measurement -- checked
directly on 1,885 models whose smallest aperture IS defined, of which
exactly zero were all-band zero).

Owner's ruling 2026-10-27 (bms_review/briefs/yso_reconvolve_2026-10-27/
BRIEF.md): this package now convolves all eleven bands itself, directly
from each geometry's raw `flux.fits`, storing an undefined (model,
aperture, band) cell as NaN -- a real missing-data marker, never a
fabricated 0 (see `convolve_yso.py`). That makes the old all-8-bands-0
heuristic unnecessary and, for the three PACS bands specifically, wrong
in the OTHER direction too: PACS-only noise-flagged SAMPLES at an
otherwise-defined aperture (interleaved with finite ones, ~0.7% MC
uncertainty) now read as a real but UNDEFINED band at that one aperture,
which the old all-8-zero rule could never detect (it only caught an
aperture undefined in literally every band at once). `undefined_apertures`
below is therefore PER BAND now (`~isfinite(flux_cubes[band])`
directly), and the gate (`d_min_from_undefined`) requires each band's
OWN two brackets finite at the survey's own aperture for that band --
not one shared all-bands test.

This module computes and returns; `yso_curate.run_aperture` decides the
threshold and writes the registry.
====================================================================
"""

import os

import numpy as np
import pandas as pd
from astropy.io import fits

from sesnaimpute import definitions
from sesnaimpute.sed_models.constants import APERTURE_GRID_AU, BANDS, REGIONS
from sesnaimpute.sed_models.curate.model_io import align_by_name
# The canonical convolved-band reader: opens convolved/{band}.fits (this
# package's OWN `yso_convolved` family -- see `_read_band_flux`'s own
# docstring) and aligns its rows to parameters.fits order by MODEL_NAME
# (those files carry their own row order). Reused rather than
# reimplemented so "which flux is this model's" means one thing across
# the pipeline. Deliberately the raw reader, NOT read_geometry_flux_cubes,
# which clamps negatives to 0 -- that clamp would turn a band sitting at
# float noise (~-1e-11) into an exact zero and could mark a defined
# aperture undefined.
from sesnaimpute.sed_models.curate.yso_labeling import (
    BAND_ORDER, ASSEMBLY_BANDS, _CONVOLVED_FILENAME, _CONVOLVED_SUBDIR, _read_band_flux,
)

# Fraction below the nearest region distance at which the gate is set.
#
# The aperture grid is log-spaced (ratio 1.615 per step), so a buffer costs
# nothing until it crosses a bracket boundary. IRAC binds (smallest
# aperture); its boundaries sit at 0.1099 and 0.1784 kpc, so any D_NEAR in
# that window needs the same ap2/ap3 brackets. The free ceiling is ~17.5%;
# 20% drops to ap1/ap2 and costs ~47,000 models. 10% sits mid-plateau,
# tolerates the nearest region (Taurus) being revised down to ~120 pc, and
# is ~1.8 sigma beyond a REGIONS.range_kpc that is already d +/- 1 sigma.
D_BUFFER_FRAC = 0.10

# Warn when D_NEAR lands within this fraction of a bracket boundary: a
# later distance revision would then silently change the gate's strictness.
BOUNDARY_WARN_FRAC = 0.10


def region_distance_range():
    """(min, max) of every region's `range_kpc`, from `constants.REGIONS`.

    Read at run time, never hardcoded -- a distance revision propagates to
    the gate automatically. `range_kpc` is already `d_r +/- 1 sigma`, so
    the minimum is a 1-sigma-low bound before D_BUFFER_FRAC is applied.
    """
    lo = min(r.range_kpc[0] for r in REGIONS.values())
    hi = max(r.range_kpc[1] for r in REGIONS.values())
    return float(lo), float(hi)


def compute_d_near(buffer_frac=D_BUFFER_FRAC):
    """The gate threshold: `buffer_frac` below the nearest region distance."""
    lo, _ = region_distance_range()
    return lo * (1.0 - buffer_frac)


def _band_brackets(d_near, apertures_au):
    """Per band, the grid aperture indices bracketing the survey aperture at
    `d_near`, plus how close `d_near` sits to a bracket boundary.

    A boundary lies at the distance where the survey aperture equals a grid
    aperture, i.e. `ap[j] / (arcsec * 1000)` kpc.
    """
    out = {}
    for band in ASSEMBLY_BANDS:
        arcsec = BANDS[band].aperture_arcsec
        au = arcsec * d_near * 1000.0
        k = int(np.searchsorted(apertures_au, au))
        k = min(max(k, 1), len(apertures_au) - 1)
        boundaries = apertures_au / (arcsec * 1000.0)
        margin = float(np.min(np.abs(boundaries - d_near)) / d_near)
        out[band] = {
            "aperture_arcsec": float(arcsec),
            "survey_aperture_au_at_d_near": float(au),
            "bracket_lo_index": k - 1,
            "bracket_hi_index": k,
            "bracket_lo_au": float(apertures_au[k - 1]),
            "bracket_hi_au": float(apertures_au[k]),
            "fractional_margin_to_boundary": margin,
        }
    return out


def undefined_apertures(flux_cubes, bands=ASSEMBLY_BANDS):
    """{band: (n_models, n_apertures) bool}: True where that band's OWN
    raw-reconvolved flux is NaN (owner's ruling 2026-10-27) -- a direct,
    per-band missing-data mask, replacing the retired all-eleven-bands-
    exactly-0 heuristic the release's fabricated zeros used to require
    (see module docstring).

    `flux_cubes` is {band: (n, n_ap)} as returned per geometry (the raw
    `_read_band_flux` reader, NaN preserved, never clamped to 0)."""
    return {band: ~np.isfinite(flux_cubes[band]) for band in bands}


def d_min_from_undefined(undefined_by_band, apertures_au, d_near, bands=ASSEMBLY_BANDS):
    """(n_models,) float: minimum distance in kpc at which EVERY band's
    survey aperture is bracketed by defined grid apertures, PER BAND
    (owner's ruling 2026-10-27 -- `undefined_by_band` is now {band:
    (n,n_ap) bool}, not one ANDed array).

    Per band, let `f` be 1 + the LAST undefined index (the defensive
    logic this module has always used: the undefined block was a
    contiguous prefix for every one of the 2.2M models under the old
    all-bands-0 detection, measured `n_noncontiguous_undefined == 0`; `f`
    keyed on the last undefined index rather than the first defined one
    still refuses to interpolate across a hole if that ever changes).
    Converting to distance, `d_min_band = ap[f] / (aperture_arcsec *
    1000)`, and the model's raw `d_min` is the max over bands (the
    binding band) -- exactly the prior formula, just computed from each
    band's OWN mask instead of one shared one.

    NEW under per-band NaN detection, which the old heuristic never had
    to handle: a genuinely undefined cell can now also sit ISOLATED above
    the prefix (a single noise-flagged PACS sample at an otherwise-
    defined aperture -- see module docstring), which the prefix-only `f`
    does not see. So this function additionally requires, per band, that
    the TWO GRID APERTURES BRACKETING this band's own survey aperture AT
    D_NEAR are both finite (`_band_brackets`'s own bracket indices); a
    model failing that check is forced to `d_min = inf` regardless of
    what the prefix alone would say -- the same "computed, never usable"
    convention already used below for a model with no defined aperture at
    all. This keeps the function's contract (one `d_min_kpc` per model,
    compared against `d_near` by the SAME `d_min_kpc < d_near` test
    everywhere this is consumed, e.g. `build_yso.run_gates`) unchanged,
    so nothing downstream of this module needs to know the rule changed.
    """
    n_models, n_ap = undefined_by_band[bands[0]].shape
    idx = np.arange(n_ap)[None, :]
    brackets = _band_brackets(d_near, apertures_au)

    d_min = np.zeros(n_models, dtype=float)
    bracket_ok = np.ones(n_models, dtype=bool)
    for band in bands:
        u = undefined_by_band[band]
        last_undefined = np.where(u.any(axis=1), (idx * u).max(axis=1), -1)
        f = last_undefined + 1
        all_undefined = f >= n_ap
        f_safe = np.minimum(f, n_ap - 1)

        scale = BANDS[band].aperture_arcsec * 1000.0
        d_min_band = np.where(all_undefined, np.inf, apertures_au[f_safe] / scale)
        d_min = np.maximum(d_min, d_min_band)

        lo_idx = brackets[band]["bracket_lo_index"]
        hi_idx = brackets[band]["bracket_hi_index"]
        bracket_ok &= ~u[:, lo_idx] & ~u[:, hi_idx]

    d_min = np.where(bracket_ok, d_min, np.inf)
    return d_min


def _geometry_apertures(geometry_dir):
    """The 20-value aperture grid, read from this geometry's own
    (reconvolved) convolved file rather than assumed. Cross-checked
    against `constants.APERTURE_GRID_AU`, which the release is documented
    to match -- a mismatch means the grid changed and every distance
    derived from it is wrong, so it raises rather than proceeding."""
    from sesnaimpute.sed_models import paths

    geometry = os.path.basename(os.path.normpath(geometry_dir))
    band = BAND_ORDER[0]
    path = os.path.join(paths.path_for("yso_convolved", geometry=geometry), f"{band}.fits")
    with fits.open(path, memmap=True) as hdul:
        apertures = np.asarray(hdul["APERTURES"].data["APERTURE"], dtype=float)

    expected = np.asarray(APERTURE_GRID_AU, dtype=float)
    if apertures.shape != expected.shape or not np.allclose(apertures, expected, rtol=1e-9):
        raise ValueError(
            f"{geometry_dir}'s aperture grid does not match constants."
            f"APERTURE_GRID_AU -- every d_min derived from it would be wrong")
    return apertures


def _old_rule_d_min_kpc(geometry_dir, ref_names, apertures_au):
    """DIAGNOSTIC ONLY (brief Part 2's required old-vs-new comparison
    print): reproduces the RETIRED all-eight-SESNA-bands-exactly-0 gate
    from the release's OWN convolved/ files (never read by anything else
    in this package any more -- owner's ruling 2026-10-27), in the SAME
    single-ANDed-mask, prefix-only form this module used before today."""
    bands = definitions.SESNA_BANDS
    undefined = None
    for band in bands:
        path = os.path.join(geometry_dir, "convolved",
                            _CONVOLVED_SUBDIR[band], _CONVOLVED_FILENAME[band])
        with fits.open(path, memmap=True) as h:
            names = np.asarray(h[1].data["MODEL_NAME"]).astype(str)
            flux = np.asarray(h[1].data["TOTAL_FLUX"], dtype=np.float64)
        order = align_by_name(ref_names, names)
        is_zero = (flux[order] == 0)
        undefined = is_zero if undefined is None else (undefined & is_zero)

    n_models, n_ap = undefined.shape
    idx = np.arange(n_ap)[None, :]
    last_undefined = np.where(undefined.any(axis=1), (idx * undefined).max(axis=1), -1)
    f = last_undefined + 1
    all_undefined = f >= n_ap
    f_safe = np.minimum(f, n_ap - 1)

    d_min = np.zeros(n_models, dtype=float)
    for band in bands:
        scale = BANDS[band].aperture_arcsec * 1000.0
        d_min = np.maximum(d_min, apertures_au[f_safe] / scale)
    d_min[all_undefined] = np.inf
    return d_min


def scan_geometry(geometry_dir, geometry, d_near, apertures_au=None, verbose=True):
    """`d_min_kpc` for one geometry under the NEW per-band rule, plus that
    geometry's diagnostics -- and the brief's required old-vs-new gate
    comparison: the pass rate of the RETIRED all-eight-SESNA-bands-0 rule
    side by side with the new eleven-band NaN-aware rule, the loss by
    Richardson Stage, and how many of that loss is attributable ONLY to a
    PACS band (new rule minus a would-be eight-band-only version of the
    new rule)."""
    if apertures_au is None:
        apertures_au = _geometry_apertures(geometry_dir)

    ref_names = np.char.strip(np.asarray(
        fits.getdata(os.path.join(geometry_dir, "parameters.fits"), 1)["MODEL_NAME"]).astype(str))
    flux_cubes = {band: _read_band_flux(geometry_dir, band, ref_names) for band in ASSEMBLY_BANDS}

    undefined = undefined_apertures(flux_cubes)
    d_min = d_min_from_undefined(undefined, apertures_au, d_near)

    # How often the undefined block is non-contiguous, per band -- the
    # reason d_min_from_undefined keys on the LAST undefined index rather
    # than the first defined one. A row that is entirely undefined IS
    # contiguous; `argmin` returns 0 for an all-True row, so it must be
    # special-cased or every all-undefined model is miscounted as
    # non-contiguous.
    n_ap = apertures_au.size
    idx = np.arange(n_ap)[None, :]
    n_noncontiguous = 0
    for band in ASSEMBLY_BANDS:
        u = undefined[band]
        first_defined = np.where(u.all(axis=1), n_ap, u.argmin(axis=1))
        is_prefix_block = (idx < first_defined[:, None]) == u
        n_noncontiguous += int((~is_prefix_block).any(axis=1).sum())

    info = {
        "n_models": int(ref_names.size),
        "n_noncontiguous": n_noncontiguous,
        "n_all_undefined": int(np.isinf(d_min).sum()),
    }

    # --- old-vs-new gate comparison (brief Part 2 / Verification item 2) ---
    d_min_old = _old_rule_d_min_kpc(geometry_dir, ref_names, apertures_au)
    pass_old = d_min_old < d_near
    pass_new = d_min < d_near
    # The new rule restricted to the eight SESNA bands only (no PACS at
    # all): a model that would pass on that alone but fails the full
    # eleven-band rule was lost ONLY because of a PACS band.
    undefined_no_pacs = {b: undefined[b] for b in definitions.SESNA_BANDS}
    d_min_no_pacs = d_min_from_undefined(undefined_no_pacs, apertures_au, d_near,
                                         bands=definitions.SESNA_BANDS)
    pass_no_pacs = d_min_no_pacs < d_near
    n_lost_to_pacs = int((pass_no_pacs & ~pass_new).sum())

    info_names = np.asarray(fits.getdata(os.path.join(geometry_dir, "info.fits"), 1)["Model Name"]).astype(str)
    stage_raw = np.asarray(fits.getdata(os.path.join(geometry_dir, "info.fits"), 1)["Stage"], dtype=int)
    stage = stage_raw[align_by_name(ref_names, info_names)]

    if verbose:
        print(f"  [{geometry}] gate: old(8-band-zero) pass={int(pass_old.sum())} "
             f"({100 * pass_old.mean():.2f}%)  new(11-band-NaN) pass={int(pass_new.sum())} "
             f"({100 * pass_new.mean():.2f}%)  lost_only_to_a_PACS_band={n_lost_to_pacs}", flush=True)
        for s in sorted(set(stage.tolist())):
            m = stage == s
            if not m.any():
                continue
            lost = int((m & pass_old & ~pass_new).sum())
            print(f"    stage={s:3d}  n={int(m.sum()):7d}  old_pass={int((m & pass_old).sum()):7d}  "
                 f"new_pass={int((m & pass_new).sum()):7d}  lost(old_pass&~new_pass)={lost:7d}", flush=True)

    return pd.DataFrame({
        "model_id": np.char.add(f"{geometry}:", np.arange(len(d_min)).astype(str)),
        "d_min_kpc": d_min,
    }), info


def scan_geometries(yso_root, geometries, buffer_frac=D_BUFFER_FRAC, verbose=True):
    """
    `d_min_kpc` for every model across all geometries.

    Returns `(DataFrame[model_id, d_min_kpc], run_info)`. `run_info` is the
    `/metadata/aperture` payload: the region distance range and the
    resulting `D_NEAR`, per-band brackets and boundary margins, per-geometry
    pass rates, and the non-contiguity tally.

    Reads only `convolved/{band}.fits` and `parameters.fits` -- never
    `flux.fits`, consistent with the rest of the pipeline. (`info.fits`
    is also read now, for the old-vs-new gate comparison's loss-by-Stage
    print -- `info.fits` is column-only and cheap, the same file `build_yso`'s
    own ingest stage already reads.)
    """
    d_lo, d_hi = region_distance_range()
    d_near = compute_d_near(buffer_frac)
    apertures_au = None

    frames, per_geometry = [], {}
    n_noncontiguous = n_all_undefined = 0
    for i, geometry in enumerate(geometries, start=1):
        geometry_dir = os.path.join(yso_root, geometry)
        if apertures_au is None:
            apertures_au = _geometry_apertures(geometry_dir)
        frame, info = scan_geometry(geometry_dir, geometry, d_near, apertures_au=apertures_au,
                                    verbose=verbose)
        frames.append(frame)
        n_pass = int((frame["d_min_kpc"] < d_near).sum())
        per_geometry[geometry] = {
            "n_models": info["n_models"],
            "n_valid_aperture": n_pass,
            "pass_rate": n_pass / info["n_models"] if info["n_models"] else 0.0,
        }
        n_noncontiguous += info["n_noncontiguous"]
        n_all_undefined += info["n_all_undefined"]
        if verbose:
            print(f"  [{i:2d}/{len(geometries)}] {geometry:9s} n={info['n_models']:7d}  "
                  f"aperture-defined {100 * per_geometry[geometry]['pass_rate']:5.1f}%", flush=True)

    d_min = pd.concat(frames, axis=0, ignore_index=True)
    brackets = _band_brackets(d_near, apertures_au)

    tight = {b: v["fractional_margin_to_boundary"] for b, v in brackets.items()
             if v["fractional_margin_to_boundary"] < BOUNDARY_WARN_FRAC}
    if tight and verbose:
        print(f"  WARNING: D_NEAR={d_near:.4f} kpc sits within "
              f"{BOUNDARY_WARN_FRAC:.0%} of an aperture-bracket boundary for "
              f"{sorted(tight)} -- a small distance revision would change the "
              f"gate's strictness. Margins: "
              f"{ {b: round(m, 4) for b, m in sorted(tight.items())} }", flush=True)

    run_info = {
        "d_lo_region_kpc": d_lo,
        "d_hi_region_kpc": d_hi,
        "d_buffer_frac": buffer_frac,
        "d_near_kpc": d_near,
        "n_regions": len(REGIONS),
        "bands": brackets,
        "boundary_warning_bands": sorted(tight),
        "per_geometry": per_geometry,
        "n_models": int(len(d_min)),
        "n_valid_aperture": int((d_min["d_min_kpc"] < d_near).sum()),
        "n_noncontiguous_undefined": n_noncontiguous,
        "n_all_apertures_undefined": n_all_undefined,
        "undefined_signature": "per band, this package's own reconvolved flux is NaN "
                              "(owner's ruling 2026-10-27; was: all 8 convolved bands "
                              "exactly 0 at that aperture)",
    }
    return d_min, run_info
