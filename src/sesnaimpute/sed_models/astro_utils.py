"""
astro_utils.py
====================================================================
Package-wide flux <-> magnitude <-> color conversions. Add general
astronomy utilities here (not tied to one subpackage/track) rather than
starting a parallel convention elsewhere -- same reasoning constants.py
and plot_style.py give for their own scope.

UNIT CONVENTION -- READ BEFORE USING: every function below assumes flux
(and any zero point) is in **mJy**, this project's flux unit throughout
(catalog FNU, convolved/*.fits TOTAL_FLUX, sedfitter VALUES). There is no
unit checking -- these are plain floats/arrays, matching constants.py's
own convention (e.g. SPEED_OF_LIGHT_CM_S, C_UM_S). Passing a flux or zero
point in Jy (or any other unit) will silently produce a wrong magnitude:
flux_to_mag/mag_to_flux depend on flux and zero_point sharing units, so a
1000x (mJy vs Jy) mismatch there is a real 7.5 mag error. color/vega_color
are more forgiving -- a color is a ratio of same-band-pair magnitudes, so
it comes out identical whether the shared flux unit is mJy or Jy, AS LONG
AS both fluxes share one unit and both zero points share one unit -- but
stick to mJy throughout anyway, to match every other flux value in this
project and avoid relying on that cancellation.
====================================================================
"""

import numpy as np

from .constants import VEGA_ZERO_POINT_MJY


def flux_to_mag(flux_mjy, zero_point_mjy):
    """Magnitude from a flux and its reference zero-point flux, both in
    mJy (see module docstring). zero_point_mjy sets the magnitude SYSTEM
    (Vega, AB, ...) -- pass VEGA_ZERO_POINT_MJY[band] for the Vega system
    already tabulated in constants.py, or any other zero point in mJy for
    a different system."""
    return -2.5 * np.log10(flux_mjy / zero_point_mjy)


def mag_to_flux(mag, zero_point_mjy):
    """Inverse of flux_to_mag: flux in mJy (same unit as zero_point_mjy)
    from a magnitude in zero_point_mjy's system."""
    return zero_point_mjy * 10 ** (-mag / 2.5)


def color(flux_a_mjy, flux_b_mjy, zero_point_a_mjy, zero_point_b_mjy):
    """mag_a - mag_b, from two fluxes and their respective zero points
    (all mJy, see module docstring). Both bands should come from the
    same zero-point system (e.g. both Vega) for the color to be
    physically meaningful."""
    return flux_to_mag(flux_a_mjy, zero_point_a_mjy) - flux_to_mag(flux_b_mjy, zero_point_b_mjy)


def vega_color(flux_a_mjy, flux_b_mjy, band_a, band_b):
    """color() using constants.VEGA_ZERO_POINT_MJY, looked up by
    BANDS-style band key (e.g. "I1", "I2"). flux_a_mjy/flux_b_mjy are
    assumed to be in mJy (see module docstring)."""
    return color(
        flux_a_mjy, flux_b_mjy,
        VEGA_ZERO_POINT_MJY[band_a], VEGA_ZERO_POINT_MJY[band_b],
    )


def color_error(flux_a_mjy, flux_b_mjy, sigma_a_mjy, sigma_b_mjy):
    """Propagated 1-sigma uncertainty on a color (mag_a - mag_b), from
    the two fluxes' own 1-sigma flux uncertainties (all mJy, see module
    docstring): sigma_color = (2.5/ln10) * sqrt((sigma_a/flux_a)^2 +
    (sigma_b/flux_b)^2). Independent of which zero-point system the
    color itself uses -- the zero point is a constant offset and drops
    out of the error propagation."""
    return (2.5 / np.log(10.0)) * np.sqrt(
        (sigma_a_mjy / flux_a_mjy) ** 2 + (sigma_b_mjy / flux_b_mjy) ** 2
    )


def reddening_vector_per_ak(law_name="whitney.r550"):
    """(d([4.5]-[5.8])/dA_K, d([3.6]-[4.5])/dA_K) for `law_name`.

    Extinction is a pure magnitude-space additive shift, exactly linear in
    A_K for a fixed-shape opacity law, so the reddening track in an IRAC
    colour-colour plane is a straight line of fixed slope.

    Shared by the H2-shock and PAH-C contaminant tracks, which each held a
    character-for-character copy. One of the copies justified itself as
    avoiding "an odd cross-track dependency" -- but that module already
    imported from the other, so the stated reason did not hold.

    NOTE the slope is NOT law-invariant: whitney.r550 gives 1.5699 while
    draine_rv3.1 gives 1.7339. Any claim about how this track sits relative
    to a published colour cut must be stated per law, or as a property that
    survives across laws -- not as a single number.

    Returns (dx, dy) in that order: x = [4.5]-[5.8], y = [3.6]-[4.5].
    """
    import numpy as np

    from .constants import BANDS
    from .data_loader import load_extinction_law

    law = load_extinction_law(law_name)

    def kappa(wave_um):
        return np.interp(wave_um, law.wave_um, law.opacity_cm2_per_g)

    kappa_k = kappa(BANDS["Ks"].wvl_effective_um)
    a_36 = kappa(BANDS["I1"].wvl_effective_um) / kappa_k
    a_45 = kappa(BANDS["I2"].wvl_effective_um) / kappa_k
    a_58 = kappa(BANDS["I3"].wvl_effective_um) / kappa_k

    return (a_45 - a_58), (a_36 - a_45)      # (dx, dy)


def reddening_vector_per_ak(band_blue, band_mid, band_red,
                            law_name="whitney.r550"):
    """Colour shift per unit A_K along a reddening ray, as
    (d(mid-red)/dA_K, d(blue-mid)/dA_K).

    Reddening is a pure magnitude-space additive shift, exactly linear in
    A_K for a fixed-shape opacity law, so a locus reddens along a straight
    ray of constant slope in any colour-colour plane. That makes this the
    natural companion to a colour-space region test: a model's reachability
    under extinction is decided by this direction, not by its position
    alone.

    `law_name` must be the law the FIT applies, not a dereddening law
    adopted by a classification scheme -- the two answer different
    questions and are not interchangeable.
    """
    from .constants import BANDS
    from .data_loader import load_extinction_law

    law = load_extinction_law(law_name)

    def a_over_ak(band):
        return (np.interp(BANDS[band].wvl_effective_um, law.wave_um, law.opacity_cm2_per_g)
                / np.interp(BANDS["Ks"].wvl_effective_um, law.wave_um, law.opacity_cm2_per_g))

    return (a_over_ak(band_mid) - a_over_ak(band_red),
            a_over_ak(band_blue) - a_over_ak(band_mid))
