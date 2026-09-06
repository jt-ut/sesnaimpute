"""
route.py
====================================================================
Admission: which phase (if any) a source enters, from band availability
and photometric uncertainty.

Routing is upstream of all geometry and is entirely discrete.  It is
also an independent source of label variance: two SEDs with identical
colours can receive different labels purely because one was admitted to
Phase 1 and the other to Phase 2, which uses different gates in a
different order after dereddening.
====================================================================
"""

import numpy as np

from sesnaimpute.gutcolors.spec import BAND_ORDER

# Gutermuth's admission thresholds, in magnitudes.
SIGMA_MAX_IRAC = 0.2
SIGMA_MAX_2MASS = 0.1
SIGMA_MAX_MIPS = 0.2

IRAC = ("I1", "I2", "I3", "I4")

# route codes: phase-1/2 admission in the low bits, 24 um overlay in bit 2
ROUTE_NONE, ROUTE_P1, ROUTE_P2 = 0, 1, 2
ROUTE_NAMES = {
    0: "none", 1: "P1", 2: "P2",
    4: "none_24", 5: "P1_24", 6: "P2_24",
}


def detection_mask(sigma_mag, valid=None):
    """Per-band boolean: may the cascade use this value?

    `valid` is a plain boolean mask -- True means the band carries a
    value the cascade may consider.  This package deliberately does not
    understand any catalogue's provenance encoding; callers translate
    their own codes.  Passing None derives the mask from sigma alone.

    A band with a measurement but sigma above the scheme's threshold is
    treated as a NON-detection, which is what makes the thresholds
    admission criteria rather than mere quality flags.
    """
    sigma_mag = np.asarray(sigma_mag, dtype=float)
    thresh = np.array(
        [
            SIGMA_MAX_2MASS if b in ("J", "H", "Ks") else
            SIGMA_MAX_MIPS if b == "M1" else
            SIGMA_MAX_IRAC
            for b in BAND_ORDER
        ]
    )
    ok = np.isfinite(sigma_mag) & (sigma_mag < thresh)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    return ok


def route(detected):
    """Admission route per source.

    Phase 1: all four IRAC bands detected.
    Phase 2: lacks 5.8 or 8.0, and has H and Ks.  Disjoint from Phase 1
             by construction.
    Phase 3: 24 um detected -- an overlay on either, or on neither.
    """
    det = {b: detected[..., i] for i, b in enumerate(BAND_ORDER)}

    p1 = det["I1"] & det["I2"] & det["I3"] & det["I4"]
    p2 = (~(det["I3"] & det["I4"])) & det["H"] & det["Ks"]

    code = np.where(p1, ROUTE_P1, np.where(p2, ROUTE_P2, ROUTE_NONE))
    code = code + 4 * det["M1"].astype(int)
    return code


def has_24(route_code):
    return (np.asarray(route_code) & 4) != 0


def phase12(route_code):
    return np.asarray(route_code) & 3


def irac_incomplete(route_code):
    """Erratum E7: 'lacks detections in some IRAC bands' is defined as
    FAILING PHASE-1 ADMISSION, not as literally missing bands.  A source
    with all four IRAC measurements but sigma > 0.2 in one of them is
    incomplete for the scheme's purposes, because Phase 1 never saw it.
    """
    return phase12(route_code) != ROUTE_P1


def longest_irac(detected):
    """Index into IRAC of the longest-wavelength IRAC band detected,
    or -1 if none.  Used by the deeply-embedded criterion, whose operand
    is chosen by the detection mask rather than by value.
    """
    idx = np.full(detected.shape[:-1], -1, dtype=np.int8)
    for k, band in enumerate(IRAC):
        col = detected[..., BAND_ORDER.index(band)]
        idx = np.where(col, k, idx)
    return idx


def guard_values(usable, route_code, seen=None, route_seen=None):
    """Evaluate every guard named in the tables.

    Two masks, because the guards ask two different questions and only
    an IMPUTED SED can make them disagree.

    `usable` is what the flux vector currently holds -- after imputation,
    every band.  It answers "is this a value I may read?", which is what
    the E13 row guards (has_I1 ... has_M1) need.

    `seen` is what the survey actually DETECTED.  It answers "was this
    source undetectable in IRAC?" -- a fact about the source, unchanged
    by fitting a model to it.  Only `irac_incomplete` and `longest_irac`
    read it: the deeply-embedded criterion, whose whole content is that
    the dropout IS the evidence.

    Defaults to `usable`, so for observed photometry the distinction does
    not exist and nothing changes.
    """
    seen = usable if seen is None else seen
    route_seen = route_code if route_seen is None else route_seen

    li = longest_irac(seen)
    has_I3 = usable[..., BAND_ORDER.index("I3")]
    out = {f"has_{b}": usable[..., i] for i, b in enumerate(BAND_ORDER)}
    out.update({
        "longest_irac == I1": li == 0,
        "longest_irac == I2": li == 1,
        "longest_irac == I3": li == 2,
        "longest_irac == I4": li == 3,
        "has_I3": has_I3,
        "not has_I3": ~has_I3,
        "irac_incomplete": irac_incomplete(route_seen),
        # the protostar audit needs 5.8 OR 4.5 to be evaluable at all;
        # with neither, it must not fire and must not demote either.
        "auditable": usable[..., BAND_ORDER.index("I3")]
                     | usable[..., BAND_ORDER.index("I2")],
    })
    return out
