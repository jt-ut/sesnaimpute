"""
featurize.py
====================================================================
flux (mJy) -> magnitudes -> colours, propagated colour sigmas, and the
Phase-2 dereddened quantities.

All nonlinearity in the scheme lives here.  Phase 1 and Phase 3 are
affine in the band magnitudes directly; Phase 2 is affine only WITHIN a
dereddening branch, and the branch selection is a discrete switch (with
a small discontinuity at the breakpoint -- errata E8/E9).
====================================================================
"""

from dataclasses import dataclass

import numpy as np

from sesnaimpute.constants import VEGA_ZERO_POINT_MJY

from sesnaimpute.gutcolors.spec import BAND_ORDER

# --- extinction ratios, Flaherty+07 as quoted in the appendix -------------
E_JH_OVER_E_HK = 1.73
E_HK_OVER_E_K36 = 1.49
E_HK_OVER_E_K45 = 1.17

# E_[3.6]-[4.5] / E_H-K.  The appendix prints this with a spurious outer
# inverse, giving 5.447 -- the reciprocal of what the closed forms in the
# same subsection require.  Erratum E1.
C_3645_CORRECTED = 1.0 / E_HK_OVER_E_K45 - 1.0 / E_HK_OVER_E_K36   # 0.183560
C_3645_PRINTED = 1.0 / C_3645_CORRECTED                            # 5.447

# A_3.6 / E_H-K.  Never supplied by the appendix; derived from Flaherty's
# A_3.6/A_K together with E_H-K/E_K-3.6 = 1.49.  Erratum E2.
#   E_K-3.6 = A_K (1 - A_3.6/A_K)  ->  A_K/E_H-K = 1/(1.49 (1 - 0.632))
A36_OVER_AK = 0.632
AK_OVER_E_HK = 1.0 / (E_HK_OVER_E_K36 * (1.0 - A36_OVER_AK))       # 1.8238
A36_OVER_E_HK = A36_OVER_AK * AK_OVER_E_HK                         # 1.1526

# --- intrinsic-colour loci ------------------------------------------------
CTTS_SLOPE, CTTS_INTERCEPT = 0.58, 0.52      # [J-H]_0 = 0.58 [H-K]_0 + 0.52
JH0_FLOOR = 0.6                              # for [H-K]_0 <= 0.14
HK0_BREAK_J = 0.14

GT05_SLOPE, GT05_INTERCEPT = 1.33, 0.133     # [H-K]_0 = 1.33 X0 + 0.133
HK0_FLOOR = 0.2                              # for X0 <= 0.06
X0_BREAK = 0.06


@dataclass(frozen=True)
class SchemeConfig:
    c_mode: str = "corrected"          # erratum E1: "corrected" | "printed"
    clamp_ehk: bool = True             # erratum E4
    sigma_binding: str = "per_colour"  # erratum E3; see spec.SIGMA_BINDINGS

    @property
    def C(self) -> float:
        return C_3645_CORRECTED if self.c_mode == "corrected" else C_3645_PRINTED


def flux_to_mag(flux_mjy):
    """Vega magnitudes from an (N, 8) mJy flux array.

    Non-positive flux yields NaN -- a physically meaningful outcome for a
    low-S/N draw, and one that must not silently become a magnitude.
    """
    flux = np.asarray(flux_mjy, dtype=float)
    zp = np.array([VEGA_ZERO_POINT_MJY[b] for b in BAND_ORDER])
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(np.where(flux > 0, flux, np.nan) / zp)
    return mag


def sigma_to_mag(flux_mjy, sigma_mjy):
    """Per-band 1-sigma magnitude uncertainty from flux and flux error."""
    flux = np.asarray(flux_mjy, dtype=float)
    sig = np.asarray(sigma_mjy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (2.5 / np.log(10.0)) * np.abs(sig / np.where(flux > 0, flux, np.nan))
    return np.where(np.isfinite(out), out, np.inf)


def colours(mag):
    """The nine colours and three magnitudes used by the boundary table.

    D = A+B and E = B+C are computed from the magnitudes rather than from
    A/B/C, so this stays correct if the redundancy is ever removed.
    """
    m = {b: mag[..., i] for i, b in enumerate(BAND_ORDER)}
    return {
        "A": m["I1"] - m["I2"],
        "B": m["I2"] - m["I3"],
        "C": m["I3"] - m["I4"],
        "D": m["I1"] - m["I3"],
        "E": m["I2"] - m["I4"],
        "F": m["I1"] - m["M1"],
        "G": m["I2"] - m["M1"],
        "H": m["I3"] - m["M1"],
        "I": m["I4"] - m["M1"],
        "m36": m["I1"],
        "m45": m["I2"],
        "m24": m["M1"],
    }


def colour_sigmas(sig_mag):
    """Quadrature sums over the constituent bands."""
    s = {b: sig_mag[..., i] for i, b in enumerate(BAND_ORDER)}
    q = lambda a, b: np.sqrt(s[a] ** 2 + s[b] ** 2)  # noqa: E731
    return {
        "sA": q("I1", "I2"),
        "sB": q("I2", "I3"),
        "sC": q("I3", "I4"),
        "sD": q("I1", "I3"),
        "sE": q("I2", "I4"),
        "sY": q("Ks", "I1"),
    }


def deredden(mag, has_J, config=None):
    """Phase-2 dereddening.

    Solves for the intrinsic [H-Ks] colour by intersecting the reddening
    vector with an intrinsic-colour locus, then dereddens Ks-[3.6],
    [3.6]-[4.5] and [3.6] by the recovered E(H-K).

    Two locus branches per case, selected by where the solution lands.
    The J-present branch uses the CTTS locus (Meyer+97) floored at
    [J-H]_0 >= 0.6; the J-absent branch uses the GT05 YSO locus floored
    at [H-K]_0 >= 0.2.

    Returns (X0, Y0, m36_0, E_HK, branch) where branch is 0 for the
    sloped locus and 1 for the floor.
    """
    cfg = config or SchemeConfig()
    C = cfg.C

    m = {b: mag[..., i] for i, b in enumerate(BAND_ORDER)}
    JH = m["J"] - m["H"]
    HK = m["H"] - m["Ks"]
    A = m["I1"] - m["I2"]

    # --- J present: intersect with the CTTS locus ---------------------
    with np.errstate(invalid="ignore"):
        hk0_j = (JH - E_JH_OVER_E_HK * HK - CTTS_INTERCEPT) / (
            CTTS_SLOPE - E_JH_OVER_E_HK
        )
        flat_j = hk0_j <= HK0_BREAK_J
        hk0_j_flat = HK - (JH - JH0_FLOOR) / E_JH_OVER_E_HK
        hk0_j = np.where(flat_j, hk0_j_flat, hk0_j)

    # --- J absent: intersect with the GT05 locus -----------------------
    with np.errstate(invalid="ignore"):
        hk0_n = (GT05_SLOPE * (C * HK - A) - GT05_INTERCEPT) / (GT05_SLOPE * C - 1.0)
        # on the sloped branch [H-K]_0 = 1.33 X0 + 0.133, so testing
        # X0 <= 0.06 is the same as testing [H-K]_0 <= 0.2128
        flat_n = hk0_n <= (GT05_SLOPE * X0_BREAK + GT05_INTERCEPT)
        hk0_n = np.where(flat_n, HK0_FLOOR, hk0_n)

    hk0 = np.where(has_J, hk0_j, hk0_n)
    branch = np.where(has_J, flat_j, flat_n).astype(np.int8)

    E_HK = HK - hk0
    if cfg.clamp_ehk:
        E_HK = np.maximum(E_HK, 0.0)   # erratum E4

    Y0 = (m["Ks"] - m["I1"]) - E_HK / E_HK_OVER_E_K36
    X0 = A - E_HK * C
    m36_0 = m["I1"] - E_HK * A36_OVER_E_HK

    return X0, Y0, m36_0, E_HK, branch


def featurize(flux_mjy, sigma_mjy, has_J=None, config=None):
    """Everything the boundary table needs, as a (features, sigmas) pair."""
    mag = flux_to_mag(flux_mjy)
    sig_mag = sigma_to_mag(flux_mjy, sigma_mjy)

    feat = colours(mag)
    sig = colour_sigmas(sig_mag)

    if has_J is None:
        # Fallback only.  Callers should pass the J DETECTION flag: a
        # catalogue that fills non-detections with an upper-limit flux
        # yields a finite magnitude, and this test would then send the
        # source down the J-present dereddening branch on a fabricated
        # J magnitude.  See crisp.classify_crisp, which passes the mask.
        has_J = np.isfinite(mag[..., BAND_ORDER.index("J")])
    X0, Y0, m36_0, E_HK, branch = deredden(mag, has_J, config)
    feat.update(X0=X0, Y0=Y0, m36_0=m36_0)

    return feat, sig, dict(mag=mag, sigma_mag=sig_mag, E_HK=E_HK, branch=branch)
