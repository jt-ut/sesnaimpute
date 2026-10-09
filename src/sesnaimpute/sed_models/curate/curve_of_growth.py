"""
curve_of_growth.py
====================================================================
Reducing an aperture-dependent model library's curve of growth to one
flux per band, at the aperture this survey actually measures in.

A model set with `aperture_dependent = yes` ships flux at N aperture
radii, because an extended source's measured flux depends on the beam.
A survey measures at a fixed ANGULAR aperture, which at the library's
own distance is a specific radius in AU. This module interpolates the
one to the other.

Library-agnostic despite living in `sed_models_curate/`: nothing here is YSO
specific, and `interp_at_aperture` handles a single-aperture (NAP == 1)
library by returning its flux unchanged, so the four
`aperture_dependent = no` libraries (sps, galz, h2shock, pahc) can call
the same entry point as a no-op. It lives here rather than at package
top level because the quantity is only meaningful for a model library,
and `sed_models_curate` is what owns those; `sed_models_register` consumes libraries
and imports downward, which is the natural direction (nothing in
`sed_models_curate` imports `sed_models_register`).

Named for the physical object rather than "aperture" to stay clear of
`yso_aperture.py`, which is the eligibility gate -- an unrelated job
that happens to share the word.

WHY LINEAR-IN-AU, AND NOT LOG-LOG
---------------------------------
`sedfitter`'s `ConvolvedFluxes.interpolate` (convolved_fluxes.py:321)
is `scipy.interp1d` at its default kind: **linear in aperture, on raw
linear flux**. That is the array `Models.read` hands the fitter, so it
is what every fitted scale factor multiplies. Anything derived from a
library in order to be COMPARED WITH A FIT -- a survey-matched SED, a
reference flux, a colour -- has to use the fitter's convention or it
describes a template the fitter never presents.

Log-log is arguably the better interpolation of a curve of growth (it
is closer to a power law than a straight line on a log-spaced aperture
grid), so this is a deliberate accuracy sacrifice to honour a contract,
not a bug fix. Measured on the R17 grid, the two agree to 4-5 decimals
for the ~86% of models whose bracket is flat, and diverge sharply for
the rest: median 0.024 sigma in the fixed-sigma metric, but p99 = 45.9
sigma, and 10.8% of models shift by more than 1 sigma.

THE FLOOR GOES ON AFTER, NOT BEFORE
-----------------------------------
Under log-log, clamping the bracket endpoints and clamping the result
are the same operation. Under linear they are NOT, and the difference
is the entire reason the old and new mu-SEDs disagree in the tail:
clamping first makes a synthetic `peak_i * 10**-D` an interpolation
ENDPOINT, and a power law fitted through it lands ~4 dex low (worst
case measured: 15 dex). Interpolating raw makes `f_lo == 0` harmless
-- the result is just `frac * f_hi` -- and leaves the floor doing only
its intended job of keeping a subsequent log finite.
====================================================================
"""

import numpy as np

from sesnaimpute.sed_models.constants import BANDS

# Per-model flux floor depth, in dex below that model's own peak.
#
# The ANCHOR (per-model, over the model's own full cube) is the load-
# bearing choice and is argued in yso_decisions.md D-9: a grid-wide
# per-band floor coupled every model's floor to whichever model was
# brightest elsewhere, and pushed 25-28% of all values below it.
#
# The DEPTH is a convention inside a measured safe range, and is
# recorded as such rather than dressed up as derived. 10 dex is 25
# magnitudes of dynamic range -- more than any photometric measurement
# or scale-marginalised fit can use. Measured on the R17 grid:
# physically meaningful values sit within 8-9 dex of peak, while below
# ~10 dex absolute fluxes fall to ~1e-11 mJy and keep going (11.7% of
# raw interpolated values are already below float32's smallest normal,
# 11.2% are exactly zero) -- i.e. radiative-transfer Monte-Carlo noise
# and denormals, not signal. Any depth in roughly [9, 40] behaves
# identically downstream, because both the labeling kernel and the
# dedup chi-square saturate far above it; 10 is the shallowest value
# that clips no meaningful signal.
FLOOR_DEX_BELOW_PEAK = 10.0

PC_CM = 3.0856775814913673e18


def survey_aperture_au(distance_cm):
    """{band: aperture radius in AU} at `distance_cm`, from
    `BANDS[*].aperture_arcsec`.

    `distance_cm` comes from the library's own flux.fits DISTANCE
    header, never a hardcoded value -- a release at another distance
    then needs no code change, and a survey aperture revision
    propagates from `constants.BANDS` alone.
    """
    distance_pc = distance_cm / PC_CM
    return {band: spec.aperture_arcsec * distance_pc for band, spec in BANDS.items()}


def peak_and_floor(flux_cubes, dex_below_peak=FLOOR_DEX_BELOW_PEAK):
    """Per-model flux floor from an ALREADY-READ {band: (n, n_ap)} dict
    -- no I/O.

    `peak_i` is the max over model i's OWN full cube (every band, every
    aperture), so the floor is scale-invariant and independent of the
    rest of the grid. Consumers that need "dark band" to mean the same
    thing must share this function rather than re-deriving it.

    Returns (peak_row, floor_dex_row, floor_linear_row, all_zero_flux),
    all (n,) with `all_zero_flux` bool -- True where `peak_i == 0`, i.e.
    no flux anywhere, so there is no peak to anchor to. Those rows are
    excluded upstream on other grounds; the flag is returned so a caller
    can assert rather than silently propagate a meaningless floor.
    """
    bands = list(flux_cubes)
    n = flux_cubes[bands[0]].shape[0]
    peak_row = np.zeros(n)
    for band in bands:
        peak_row = np.maximum(peak_row, np.asarray(flux_cubes[band]).max(axis=1))
    all_zero_flux = peak_row <= 0
    # log10(0) = -inf on all-zero rows: guarded rather than left to warn.
    with np.errstate(divide="ignore"):
        floor_dex_row = np.log10(peak_row) - dex_below_peak
    return peak_row, floor_dex_row, 10.0 ** floor_dex_row, all_zero_flux


def interp_at_aperture(flux_n_ap, aperture_au, target_au):
    """Flux at `target_au`, linear in aperture on raw linear flux --
    numerically what `sedfitter`'s `interp1d(apertures, flux)(target)`
    returns (verified to 1e-15 across the survey apertures).

    NO floor is applied. Clamp the RESULT if you need a positive value
    for a log; clamping the inputs instead reintroduces the synthetic-
    endpoint problem described in this module's header.

    `flux_n_ap` is (n, n_ap), `aperture_au` (n_ap,) ascending, and the
    return is (n,). A single-aperture library (n_ap == 1) returns that
    column unchanged, so aperture-independent model sets can share this
    call site.

    `target_au` outside the grid is clamped to the nearest interior
    bracket, matching sedfitter's own edge behaviour. Callers that
    expect interiority should assert it (see
    `assert_target_interior`) rather than rely on the clamp, which
    would otherwise read as a real interpolated value.
    """
    flux_n_ap = np.asarray(flux_n_ap)
    aperture_au = np.asarray(aperture_au, dtype=float)
    if aperture_au.size == 1:
        return flux_n_ap[:, 0]

    if target_au <= aperture_au[0]:
        idx_hi = 1
    elif target_au >= aperture_au[-1]:
        idx_hi = aperture_au.size - 1
    else:
        idx_hi = int(np.searchsorted(aperture_au, target_au))
    idx_lo = idx_hi - 1

    x_lo, x_hi = aperture_au[idx_lo], aperture_au[idx_hi]
    frac = (target_au - x_lo) / (x_hi - x_lo)
    f_lo = flux_n_ap[:, idx_lo]
    return f_lo + frac * (flux_n_ap[:, idx_hi] - f_lo)


def assert_target_interior(aperture_au, targets_au, context=""):
    """Raise unless every target aperture is strictly interior to the
    grid, so a miscalibrated survey aperture or an unusually narrow
    aperture grid fails loudly instead of silently clamping to an edge
    bracket and reading as a real measurement.
    """
    aperture_au = np.asarray(aperture_au, dtype=float)
    if aperture_au.size == 1:
        return
    for name, target in dict(targets_au).items():
        if not (aperture_au[0] < target < aperture_au[-1]):
            raise ValueError(
                f"{name}'s survey aperture ({target:.1f} AU) is not strictly interior to "
                f"the aperture grid ({aperture_au[0]:.1f}-{aperture_au[-1]:.1f} AU)"
                f"{' -- ' + context if context else ''}; interpolation would clamp to the "
                "nearest bracket and return a value with no data behind it."
            )


def survey_matched_sed(flux_cubes, aperture_au, distance_cm, band_order,
                       dex_below_peak=FLOOR_DEX_BELOW_PEAK):
    """The shared entry point: one flux per band, at each band's own
    survey aperture, floored -- `mu_sed` in the YSO curation pipeline
    and `f_ref` in `sed_models_register`, which are the same quantity (verified
    equal to 6e-15 on the shipped library).

    HYBRID BY CONSTRUCTION: each band is evaluated at ITS OWN aperture,
    so this is not a slice of the cube at any single aperture. That
    mismatch is the survey's physical reality -- a 24um measurement
    integrates a larger region than a 3.6um one -- and is reproduced
    rather than averaged away.

    Returns (sed_linear, floor_linear_row, all_zero_flux): `sed_linear`
    is (n, len(band_order)) in the cubes' own flux unit (mJy
    throughout this project), strictly positive wherever `peak_i > 0`.
    """
    targets = survey_aperture_au(distance_cm)
    assert_target_interior(aperture_au, {b: targets[b] for b in band_order})
    _, _, floor_linear_row, all_zero_flux = peak_and_floor(flux_cubes, dex_below_peak)

    n = flux_cubes[band_order[0]].shape[0]
    sed = np.empty((n, len(band_order)), dtype=float)
    for i, band in enumerate(band_order):
        raw = interp_at_aperture(flux_cubes[band], aperture_au, targets[band])
        sed[:, i] = np.maximum(raw, floor_linear_row)   # floor AFTER, see header
    return sed, floor_linear_row, all_zero_flux
