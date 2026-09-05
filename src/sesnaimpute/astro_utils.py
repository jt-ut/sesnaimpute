"""Common conversions: magnitudes, fluxes, extinction ratios.

Empty until a module needs one. Add a function here only when a module
computes with it, each citing the zero point or law it uses.
"""

import numpy as np


def mag_to_flux_mjy(mag, zero_point_mjy):
    """`f = zero_point * 10**(-0.4*mag)`, mJy -- the standard magnitude-
    flux relation, in whichever photometric system `zero_point_mjy` is
    itself quoted in (Vega or AB): the zero point is the flux of a
    zero-magnitude source in that system.
    """
    return np.asarray(zero_point_mjy, dtype=float) * 10.0 ** (
        -0.4 * np.asarray(mag, dtype=float))


def distance_modulus(d_pc):
    """`5*log10(d_pc) - 5`: the standard distance modulus, `d_pc` in
    parsecs.
    """
    return 5.0 * np.log10(np.asarray(d_pc, dtype=float)) - 5.0
