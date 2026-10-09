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

**What "undefined" means, and how it is detected.** An aperture is
undefined iff all 8 convolved bands are exactly `0` there. Validated
against the raw `flux.fits` NaN mask on 400 models x 20 apertures:
precision 1.0000, recall 1.0000. A model with a central star always emits
*something* within 100 AU, so all-8-bands-zero never occurs as a real
measurement -- checked directly on 1,885 models whose smallest aperture IS
defined, of which exactly zero are all-band zero. Per-*band* zeros at a
defined aperture are a different thing and are genuine (120/120 sampled had
fully finite raw data across the filter), which is why the test is all-8
rather than per-band.

This module computes and returns; `yso_curate.run_aperture` decides the
threshold and writes the registry.
====================================================================
"""

import os

import numpy as np
import pandas as pd

from sesnaimpute.sed_models.constants import APERTURE_GRID_AU, BANDS, REGIONS
# The canonical convolved-band reader: opens convolved/{band}.fits and
# aligns its rows to parameters.fits order by MODEL_NAME (those files carry
# their own row order). Reused rather than reimplemented so "which flux is
# this model's" means one thing across the pipeline. Deliberately the raw
# reader, NOT read_geometry_flux_cubes, which clamps negatives to 0 -- that
# clamp would turn a band sitting at float noise (~-1e-11) into an exact
# zero and could mark a defined aperture undefined.
from sesnaimpute.sed_models.curate.yso_labeling import BAND_ORDER, _read_band_flux

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
    for band in BAND_ORDER:
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


def undefined_apertures(flux_cubes):
    """(n_models, n_apertures) bool: True where all 8 bands are exactly 0.

    `flux_cubes` is {band: (n, n_ap)} as returned per geometry. ANDed band
    by band so only one band's array is resident at a time -- the largest
    geometry is 720k x 20 per band.
    """
    undefined = None
    for band in BAND_ORDER:
        is_zero = (flux_cubes[band] == 0)
        undefined = is_zero if undefined is None else (undefined & is_zero)
    return undefined


def d_min_from_undefined(undefined, apertures_au):
    """(n_models,) float: minimum distance in kpc at which EVERY band's
    survey aperture is bracketed by defined grid apertures.

    Per band, let `f` be the first index from which everything upward is
    defined. Interpolation needs both brackets defined, so the target
    aperture must exceed `ap[f]`; converting to distance,

        d_min_band = ap[f] / (aperture_arcsec * 1000)

    and the model's `d_min` is the max over bands -- the binding band.

    `f` is taken as **1 + the LAST undefined index** rather than the first
    defined one. On the current grid the two agree exactly -- the undefined
    block is a contiguous prefix for every one of the 2.2M models, measured
    (`n_noncontiguous_undefined == 0`). This is therefore defensive, not
    required by the data: if a later release ever leaves a defined aperture
    below an undefined one, keying on the last undefined index still
    refuses to interpolate across the hole, whereas keying on the first
    defined one would silently bracket across it.

    A model with no defined aperture at all gets `inf` -- computed, and
    never usable at any distance. That is distinct from `NaN`, which the
    registry reserves for "not yet computed".
    """
    n_models, n_ap = undefined.shape
    idx = np.arange(n_ap)[None, :]
    last_undefined = np.where(undefined.any(axis=1), (idx * undefined).max(axis=1), -1)
    f = last_undefined + 1

    all_undefined = f >= n_ap
    f_safe = np.minimum(f, n_ap - 1)

    d_min = np.zeros(n_models, dtype=float)
    for band in BAND_ORDER:
        scale = BANDS[band].aperture_arcsec * 1000.0
        d_min = np.maximum(d_min, apertures_au[f_safe] / scale)
    d_min[all_undefined] = np.inf
    return d_min


def _geometry_apertures(geometry_dir):
    """The 20-value aperture grid, read from this geometry's own convolved
    file rather than assumed. Cross-checked against
    `constants.APERTURE_GRID_AU`, which the release is documented to match
    -- a mismatch means the grid changed and every distance derived from it
    is wrong, so it raises rather than proceeding."""
    from astropy.io import fits
    from sesnaimpute.sed_models.curate.yso_labeling import _CONVOLVED_FILENAME, _CONVOLVED_SUBDIR

    band = BAND_ORDER[0]
    path = os.path.join(geometry_dir, "convolved",
                        _CONVOLVED_SUBDIR[band], _CONVOLVED_FILENAME[band])
    with fits.open(path, memmap=True) as hdul:
        apertures = np.asarray(hdul["APERTURES"].data["APERTURE"], dtype=float)

    expected = np.asarray(APERTURE_GRID_AU, dtype=float)
    if apertures.shape != expected.shape or not np.allclose(apertures, expected, rtol=1e-9):
        raise ValueError(
            f"{geometry_dir}'s aperture grid does not match constants."
            f"APERTURE_GRID_AU -- every d_min derived from it would be wrong")
    return apertures


def scan_geometry(geometry_dir, geometry, apertures_au=None):
    """`d_min_kpc` for one geometry, plus that geometry's diagnostics."""
    from astropy.io import fits

    if apertures_au is None:
        apertures_au = _geometry_apertures(geometry_dir)

    ref_names = np.char.strip(np.asarray(
        fits.getdata(os.path.join(geometry_dir, "parameters.fits"), 1)["MODEL_NAME"]).astype(str))
    flux_cubes = {band: _read_band_flux(geometry_dir, band, ref_names) for band in BAND_ORDER}

    undefined = undefined_apertures(flux_cubes)
    d_min = d_min_from_undefined(undefined, apertures_au)

    # How often the undefined block is non-contiguous -- the reason
    # d_min_from_undefined keys on the LAST undefined index rather than the
    # first defined one. A row that is entirely undefined IS contiguous;
    # `argmin` returns 0 for an all-True row, so it must be special-cased or
    # every all-undefined model is miscounted as non-contiguous.
    n_ap = undefined.shape[1]
    idx = np.arange(n_ap)[None, :]
    first_defined = np.where(undefined.all(axis=1), n_ap, undefined.argmin(axis=1))
    is_prefix_block = (idx < first_defined[:, None]) == undefined
    n_noncontiguous = int((~is_prefix_block).any(axis=1).sum())

    info = {
        "n_models": int(undefined.shape[0]),
        "n_noncontiguous": n_noncontiguous,
        "n_all_undefined": int(np.isinf(d_min).sum()),
    }
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
    `flux.fits`, consistent with the rest of the pipeline.
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
        frame, info = scan_geometry(geometry_dir, geometry, apertures_au=apertures_au)
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
        "undefined_signature": "all 8 convolved bands exactly 0 at that aperture",
    }
    return d_min, run_info
