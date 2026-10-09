"""
synth.py
====================================================================
Gaia G synthesis and its extinction coefficient (curation
instructions B1/B2).

WHY THIS CANNOT REUSE f_ref. G0 is not a survey-aperture quantity.
Gaia's effective windowed-PSF collection is of order 1 arcsec, i.e.
~1000 AU in the 1 kpc reference frame, against the survey's
4000 / 2400 / 7600 AU. So G0 is synthesized from the SED cube
interpolated to ITS OWN aperture; f_ref and mu_sed are both evaluated
somewhere else and cannot supply it. B1's companion column G0(100 AU)
exists to measure how much that ruling matters per model.

TWO QUANTITIES, ONE WEIGHTING. B2 defines

    kG(law) = -2.5 log10 [ <f * 10^(-0.4 A_lambda/A_V)>_G / <f>_G ]

per unit A_V. The two integrals must use the SAME band average as G0
itself, because the pair is consumed as

    G_observed = G0 * 10^(-0.4 * A_V * kG)

and that identity only holds if kG was defined against the average
that produced G0. So `_band_average` is used for all three, and the
weighting is recorded rather than assumed. It is the frequency-space,
response-weighted mean of F_nu -- the same convention sedfitter uses
for the eight photometric bands, which keeps G0 commensurable with
f_ref if anyone ever forms a G-to-IR colour.

DARKNESS, per B0/B1. A dark G band means UNDETECTABLE, and the stored
value is 0 (the true flux), not a floor and not NaN -- Gmag =
-2.5log10(0) = +inf already fails any detection threshold, and 0 does
not propagate through sums. kG for a dark model is 0 as well, NOT NaN:
with G0 = 0, a NaN kG would give 0 * 10^nan = nan and reintroduce the
NaN one column over. NaN is reserved for the one situation that is
genuinely no-answer -- a library whose SED does not span the G
bandpass -- which is CHECKED here rather than assumed, and which no
shipped library currently trips.

THE DARKNESS REFERENCE, and a deviation worth knowing about. B1 says
"below 1e-6 of the model's peak band flux" without saying at which
aperture the peak is measured. Taken literally as max(f_ref), the test
degenerates for the ~36 c0 models whose survey-aperture flux
underflows: their peak is a denormal, so almost nothing compares below
it and they would be recorded as detectable. This module instead
measures the peak over the eight bands synthesized AT THE G APERTURE,
so numerator and reference share a beam. It is one extra convolution
of an array already in memory, and it makes the test mean "is G dark
relative to this model's own brightness, as Gaia would see it".
====================================================================
"""

from dataclasses import dataclass, field

import numpy as np

from sesnaimpute.sed_models.constants import BANDS
from sesnaimpute.sed_models.data_loader import load_extinction_law, load_filter_curve

from .darkness import DARK_FRACTION

__all__ = ["SynthTables", "derive", "CoverageError", "DEFAULT_LAWS"]

_BAND_KEYS = list(BANDS)
_C_UM_S = 2.99792458e14
_V_BAND_UM = 0.55

#: B1's ruling: Gaia's ~1 arcsec window at the 1 kpc frame.
G_APERTURE_AU = 1000.0
#: B1's companion diagnostic: the compact-star floor.
G_APERTURE_COMPANION_AU = 100.0

DEFAULT_LAWS = ("draine_rv3.1", "whitney.r550")


class CoverageError(RuntimeError):
    """A library's SED does not span a filter it is being convolved with."""


@dataclass(frozen=True)
class SynthTables:
    """B1/B2 products for one library."""

    g0_mjy: np.ndarray               # (n_models,) at G_APERTURE_AU
    g0_companion_mjy: np.ndarray     # (n_models,) at 100 AU; None if aperture-free
    kg: np.ndarray                   # (n_models, n_law) mag per unit A_V
    laws: tuple
    dark: np.ndarray                 # (n_models,) bool -- G band dark
    aperture_au: float
    diagnostics: dict = field(default_factory=dict)


def _response_on(nu, curve):
    """Filter response resampled onto a frequency grid, zero outside."""
    fn = _C_UM_S / curve.wave_um
    order = np.argsort(fn)
    return np.interp(nu, fn[order], curve.response[order], left=0.0, right=0.0)


def _band_average(flux_nu, nu_sorted, response):
    """Response-weighted mean of F_nu, in frequency. Returns (avg, bad).

    `flux_nu` is (..., n_wav) already ordered to match `nu_sorted`.

    OUT-OF-SUPPORT WAVELENGTHS ARE MASKED, NOT MULTIPLIED BY ZERO. The
    R17 cubes carry NaN from submm Monte-Carlo noise far outside every
    filter's support -- 16.0% of cells in yso/c0, 6.6% in cII, with
    92.5% of sampled c0 models affected somewhere. Because NaN * 0 =
    NaN, a naive `flux * response` sum lets a NaN at 3000 um poison a
    J-band result computed entirely from clean data. sedfitter's own
    convolve_model_dir has this defect (it rebins onto the full grid and
    sums over all wavelengths); the YSO curation session measured
    445,417 good J-band cells destroyed that way on one geometry.

    A NaN INSIDE the support is different -- genuinely missing data --
    and is returned in `bad` rather than silently zeroed.
    """
    denom = np.trapz(response, nu_sorted)
    if denom <= 0:
        raise CoverageError("filter has no overlap with the SED grid")
    in_support = response > 0
    masked = np.where(in_support, flux_nu, 0.0)
    bad = np.isnan(masked).any(axis=-1)
    if np.any(bad):
        masked = np.nan_to_num(masked, nan=0.0)
    return np.trapz(masked * response, nu_sorted, axis=-1) / denom, bad


def _check_coverage(wav_um, curve, name, tol=0.999):
    """Assert the SED grid spans the filter's support.

    B1 keeps NaN for the no-coverage case; this turns "never fires on a
    shipped library" from an assumption into a checked precondition.
    Returns the fraction of the filter's response that the grid covers.
    """
    resp = curve.response
    w = curve.wave_um
    total = np.trapz(resp, w)
    inside = (w >= wav_um.min()) & (w <= wav_um.max())
    covered = np.trapz(np.where(inside, resp, 0.0), w)
    frac = float(covered / total) if total > 0 else 0.0
    return frac, frac >= tol


def _slice_cube_at_aperture(hdul, apertures_au, target_au):
    """Cube slice interpolated to `target_au`, linear in AU.

    Matches sedfitter's ConvolvedFluxes.interpolate convention. Reads
    only the two bracketing aperture planes from the memmap rather than
    the whole cube -- VALUES is ~450 MB for a single yso sub-grid.
    """
    data = hdul["VALUES"].data
    if data.shape[1] == 1:
        return np.asarray(data[:, 0, :], dtype=float)
    grid = np.asarray(apertures_au, dtype=float)
    if target_au < grid[0] or target_au > grid[-1]:
        raise CoverageError(
            f"aperture {target_au:g} AU outside stored grid "
            f"[{grid[0]:g}, {grid[-1]:g}]"
        )
    k = int(np.searchsorted(grid, target_au) - 1)
    k = min(max(k, 0), len(grid) - 2)
    span = grid[k + 1] - grid[k]
    w = 0.0 if span == 0 else (target_au - grid[k]) / span
    lo = np.asarray(data[:, k, :], dtype=float)
    hi = np.asarray(data[:, k + 1, :], dtype=float)
    return lo * (1.0 - w) + hi * w


def _k_lambda(law_name, wav_um):
    """A_lambda / A_V on the SED's wavelength grid.

    Same normalization sedfitter uses (chi at 0.55 um), so kG composes
    with the per-band k(i) the fitter applies.
    """
    law = load_extinction_law(law_name)
    w, chi = law.wave_um, law.opacity_cm2_per_g
    order = np.argsort(w)
    chi_v = np.interp(_V_BAND_UM, w[order], chi[order])
    if chi_v <= 0:
        raise ValueError(f"{law_name}: opacity at {_V_BAND_UM} um is not positive")
    return np.interp(wav_um, w[order], chi[order], left=0.0, right=0.0) / chi_v


def derive(lib, laws=DEFAULT_LAWS, aperture_au=G_APERTURE_AU,
           companion_au=G_APERTURE_COMPANION_AU):
    """Derive G0 and kG for one library."""
    g_curve = load_filter_curve("G")

    hdul, wav, apertures = lib.open_cube()
    try:
        frac, ok = _check_coverage(wav, g_curve, "G")
        aperture_dependent = lib.meta.aperture_dependent

        sed = _slice_cube_at_aperture(
            hdul, apertures, aperture_au if aperture_dependent else 0.0
        ) if aperture_dependent else np.asarray(hdul["VALUES"].data[:, 0, :], dtype=float)

        sed_companion = (
            _slice_cube_at_aperture(hdul, apertures, companion_au)
            if aperture_dependent else None
        )
    finally:
        hdul.close()

    order = np.argsort(wav)
    wav_s = wav[order]
    nu = _C_UM_S / wav_s
    nu_order = np.argsort(nu)
    nu_s = nu[nu_order]

    bad_in_support = np.zeros(lib.meta.n_models, dtype=bool)

    def _avg(flux, response):
        val, bad = _band_average(flux[:, order][:, nu_order], nu_s, response)
        np.logical_or(bad_in_support, bad, out=bad_in_support)
        return val

    resp_g = _response_on(nu_s, g_curve)

    # The aperture gate (YSO curation D-19/D-20). R17 stores NaN at small
    # apertures for medium-bearing geometries; an aperture is UNDEFINED
    # there, not empty. The assembled strata are gated so the survey and
    # G apertures always have data behind them, but the 100 AU companion
    # sits in slice 0 -- exactly where undefined apertures live (2.1% of
    # c0, 10.3% of cII). Reading those as flux would report "compact and
    # faint" for a model that simply was not computed there.
    companion_defined = None
    if aperture_dependent and sed_companion is not None:
        companion_defined = ~np.all(np.isnan(sed_companion), axis=1)

    diag = {
        "n_models": lib.meta.n_models,
        "aperture_au": float(aperture_au) if aperture_dependent else None,
        "companion_au": float(companion_au) if aperture_dependent else None,
        "g_coverage_fraction": frac,
        "weighting": "response-weighted mean of F_nu in frequency "
                     "(matches sedfitter's band convention)",
        "darkness_reference": "peak of the 8 bands synthesized at the G aperture",
        "dark_fraction_threshold": DARK_FRACTION,
    }

    if not ok:
        # B1's genuine no-answer case. Never fires on a shipped library;
        # if it ever does, it is a real gap and must surface as one.
        n = lib.meta.n_models
        diag["coverage_ok"] = False
        return SynthTables(
            g0_mjy=np.full(n, np.nan),
            g0_companion_mjy=np.full(n, np.nan) if aperture_dependent else None,
            kg=np.full((n, len(laws)), np.nan), laws=tuple(laws),
            dark=np.zeros(n, dtype=bool),
            aperture_au=float(aperture_au), diagnostics=diag,
        )
    diag["coverage_ok"] = True

    g0 = _avg(sed, resp_g)
    g0_comp = _avg(sed_companion, resp_g) if sed_companion is not None else None

    # Darkness reference: the eight bands at the SAME aperture, so the
    # numerator and its reference share a beam (see module docstring).
    band_peak = np.zeros_like(g0)
    for b in _BAND_KEYS:
        r = _response_on(nu_s, load_filter_curve(b))
        if np.trapz(r, nu_s) <= 0:
            continue
        band_peak = np.maximum(band_peak, _avg(sed, r))

    with np.errstate(invalid="ignore"):
        dark = ~(g0 >= DARK_FRACTION * band_peak)
    dark |= ~np.isfinite(g0)

    # kG, defined against the same average that produced G0.
    #
    # `atten` MUST be evaluated on `wav` -- the cube's own wavelength
    # order -- not on the sorted `wav_s`. sedfitter writes flux.fits with
    # order='nu', so wavelength runs DESCENDING (4839 -> 0.010 um for
    # yso), and `_avg` re-sorts internally. Building atten on the
    # sorted grid and multiplying it into an unsorted `sed` pairs each
    # wavelength's attenuation with the flux at its mirror wavelength.
    # That is silent -- it yields plausible small numbers rather than an
    # error -- and produced kG ~ 0.002 against the ~0.87 that A_G/A_V
    # ~ 0.86 requires.
    kg = np.zeros((len(g0), len(laws)))
    for j, law_name in enumerate(laws):
        atten = 10.0 ** (-0.4 * _k_lambda(law_name, wav))
        num = _avg(sed * atten[None, :], resp_g)
        with np.errstate(divide="ignore", invalid="ignore"):
            kg[:, j] = -2.5 * np.log10(num / g0)

    # Self-check that would have caught the above: for a FLAT F_nu
    # spectrum, kG reduces to the response-weighted mean of k(lambda)
    # over G, which for these laws must land near 0.85-0.95. Recorded
    # per law so a future ordering or normalization slip shows up as a
    # number rather than as plausible-looking output.
    flat = np.ones((1, len(wav)))
    kg_flat = []
    for law_name in laws:
        atten = 10.0 ** (-0.4 * _k_lambda(law_name, wav))
        num, _ = _band_average((flat * atten)[:, order][:, nu_order], nu_s, resp_g)
        den, _ = _band_average(flat[:, order][:, nu_order], nu_s, resp_g)
        kg_flat.append(float(-2.5 * np.log10(num[0] / den[0])))
    diag["kg_flat_spectrum"] = kg_flat

    # B0.1: finite everywhere, with substitutes chosen to be
    # arithmetically correct. Dark -> G0 = 0 (true flux) and kG = 0
    # (extinction on zero flux is zero flux, so the product is 0).
    g0 = np.where(dark, 0.0, g0)
    kg = np.where(dark[:, None] | ~np.isfinite(kg), 0.0, kg)
    if g0_comp is not None:
        g0_comp = np.where(np.isfinite(g0_comp), g0_comp, 0.0)

    diag["n_dark"] = int(dark.sum())
    diag["dark_fraction"] = float(dark.mean())
    diag["kg_median"] = [float(np.median(kg[~dark, j])) if np.any(~dark) else 0.0
                         for j in range(len(laws))]
    diag["n_nan_inside_support"] = int(bad_in_support.sum())
    if g0_comp is not None:
        if companion_defined is not None:
            diag["n_companion_undefined"] = int((~companion_defined).sum())
            g0_comp = np.where(companion_defined, g0_comp, 0.0)
        live = (~dark) & (g0 > 0) & (g0_comp > 0)
        if companion_defined is not None:
            live &= companion_defined
        diag["g0_1000_over_100_median"] = (
            float(np.median(g0[live] / g0_comp[live])) if np.any(live) else None
        )
        diag["n_companion_comparable"] = int(live.sum())
    diag["all_finite"] = bool(np.all(np.isfinite(g0)) and np.all(np.isfinite(kg)))

    return SynthTables(
        g0_mjy=g0, g0_companion_mjy=g0_comp, kg=kg, laws=tuple(laws),
        dark=dark, aperture_au=float(aperture_au), diagnostics=diag,
    )
