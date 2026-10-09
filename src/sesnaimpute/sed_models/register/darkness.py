"""
darkness.py
====================================================================
The dark-band convention (curation instructions B0), in one place.

The model grids contain bands with no flux -- 25% of yso/c0 is
exactly zero in J, 32% of cI -- and every logarithmic quantity built on
them has to decide what log10(0) means. This module owns that decision
so it is stated once rather than re-invented per consumer.

THE THRESHOLD. A band is DARK when

    f(i) < DARK_FRACTION * max_j f(j)

i.e. below one part in a million of that model's own brightest band.
Measured justification, on the shipped grids: the zero/positive boundary
is NOT the real boundary -- the smallest strictly positive J flux in
yso/c0 is 7.73e-46 mJy, a denormal rather than a measurement, and the
1st percentile of "each model's faintest band over its own peak" is
1.1e-41. The MEDIAN model's faintest band sits at 6.5e-6 of its peak. So
1e-6 falls just below the faintest genuinely-computed band and far above
the underflow. Selecting on "strictly positive" instead would retain
7.73e-46 and reject 0, which are the same statement.

SCOPE -- THIS IS THE SEMANTIC TEST, NOT THE NUMERICAL FLOOR. B0
specifies two thresholds and they are not interchangeable:

    NUMERICAL FLOOR   1e-10 x the model's FULL-CUBE peak, via
                      sed_models_curate.curve_of_growth.peak_and_floor.
                      Job: stop log10 ever seeing a zero. Used by
                      reference.py (f_ref) and density.py (B3's
                      quotient space). NOT this module.
    SEMANTIC TEST     1e-6 x the peak over the 8 BANDS, this module.
                      Job: decide whether a band carries usable
                      signal. Used by synth.py for G0's
                      detectability rule (B1).

The anchors differ because the questions do: 1e-10 of cube peak is
not "undetectable", and 1e-6 of a survey-aperture peak is not
numerically safe -- for the ~36 c0 models whose beam flux underflows,
a survey-anchored floor lands at 1.76e-37 mJy, which is not a floor.

DIFFERENT ACTIONS, and the action stays at the call site,
deliberately:

    B1 G0        -> store 0, the true value; no log is taken at storage
                    time and Gmag = -2.5log10(0) = +inf is already the
                    "undetectable" answer
    B2 kG        -> store 0, so that 0 * 10^(-0.4 Av kG) evaluates to 0
                    rather than 0 * nan
    slopes       -> store 0, because the slope multiplies a flux that
                    is already ~0

This module therefore exports the TEST and one log-safe helper, not a
general "brighten this SED" routine. A helper that floored silently and
returned only numbers would be an unannounced flooring convention --
precisely the failure B0 exists to prevent, reintroduced as a
convenience. Every function that substitutes returns its report
alongside, so a caller has to at least name it to discard it.

See also B0.1: every library-register quantity is finite, the
substitute is chosen to be arithmetically correct rather than
convenient, and DARK_BANDS records per model which bands were affected.
====================================================================
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DARK_FRACTION", "DarkReport", "dark_mask", "log10_floored",
    "floor_dark", "pack_dark_bands", "unpack_dark_bands",
]

#: B0. One number, one place. Changing it changes G0's detectability
#: rule and B3's quotient space together, which is the intent.
DARK_FRACTION = 1e-6


@dataclass(frozen=True)
class DarkReport:
    """What a dark-band operation did, so it can be reported.

    Returned by every substituting function rather than logged
    internally: a caller must name it to ignore it.
    """

    threshold_fraction: float
    n_models: int
    bands: tuple                 # band labels, if the caller supplied them
    n_dark: np.ndarray           # (n_band,) count of dark entries
    n_all_dark_models: int       # models dark in EVERY band -- pathological
    n_any_dark_models: int       # models dark in at least one band

    @property
    def fraction(self):
        return self.n_dark / max(self.n_models, 1)

    @property
    def any_dark(self):
        return bool(np.any(self.n_dark))

    def describe(self, indent="  "):
        """One block per report, for the build printout."""
        if not self.any_dark:
            return f"{indent}no dark bands (threshold {self.threshold_fraction:.0e})"
        width = max((len(b) for b in self.bands), default=4)
        lines = [
            f"{indent}dark-band floor applied "
            f"(threshold {self.threshold_fraction:.0e} of model peak):"
        ]
        for b, n, f in zip(self.bands, self.n_dark, self.fraction):
            if n:
                lines.append(f"{indent}  {b:<{width}}  {n:>8d}  ({100 * f:5.2f}%)")
        lines.append(
            f"{indent}  models with >=1 dark band: {self.n_any_dark_models}"
            f" / {self.n_models}"
        )
        if self.n_all_dark_models:
            lines.append(
                f"{indent}  WARNING: {self.n_all_dark_models} model(s) dark in "
                f"EVERY band -- peak flux is itself ~0"
            )
        return "\n".join(lines)


def _peak(flux, axis):
    peak = np.nanmax(flux, axis=axis, keepdims=True)
    # A model whose peak is <=0 has no scale to be relative to. Treat every
    # band as dark; the caller sees it via n_all_dark_models.
    return np.where(peak > 0, peak, np.inf)


def dark_mask(flux, axis=-1):
    """Boolean mask, True where a band is dark.

    `flux` is (..., n_band) by default. Non-finite entries count as dark:
    they carry no usable value either.
    """
    flux = np.asarray(flux, dtype=float)
    with np.errstate(invalid="ignore"):
        mask = ~(flux >= DARK_FRACTION * _peak(flux, axis))
    return mask


def _report(flux, mask, bands, axis):
    flux = np.asarray(flux)
    n_band = flux.shape[axis]
    if bands is None:
        bands = tuple(str(i) for i in range(n_band))
    flat = mask.reshape(-1, n_band) if mask.ndim > 1 else mask.reshape(1, n_band)
    return DarkReport(
        threshold_fraction=DARK_FRACTION,
        n_models=flat.shape[0],
        bands=tuple(bands),
        n_dark=flat.sum(axis=0),
        n_all_dark_models=int(np.count_nonzero(flat.all(axis=1))),
        n_any_dark_models=int(np.count_nonzero(flat.any(axis=1))),
    )


def floor_dark(flux, axis=-1, bands=None):
    """Raise dark bands to the threshold value. Returns (floored, report).

    Named `floor`, not `brighten`: the caller can see what happened.
    """
    flux = np.asarray(flux, dtype=float)
    mask = dark_mask(flux, axis=axis)
    floor = DARK_FRACTION * _peak(flux, axis)
    floored = np.where(mask, np.broadcast_to(floor, flux.shape), flux)
    # A model with no positive peak has no meaningful floor; leave it at 0
    # and let n_all_dark_models carry the signal.
    floored = np.where(np.isfinite(floored), floored, 0.0)
    return floored, _report(flux, mask, bands, axis)


def log10_floored(flux, axis=-1, bands=None):
    """log10 of `flux` with the B0 floor applied first.

    The B3 path. Guarantees a finite result for any model with at least
    one positive band; models with no positive band anywhere are reported
    via `n_all_dark_models` and returned as the log of the floor, which
    is finite but meaningless -- callers should drop those models on the
    report rather than on the values.
    """
    floored, report = floor_dark(flux, axis=axis, bands=bands)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(floored)
    out = np.where(np.isfinite(out), out, np.log10(DARK_FRACTION))
    return out, report


def pack_dark_bands(mask):
    """Pack an (n_models, n_band<=8) dark mask into one uint8 per model.

    Bit i is band i, in the caller's band order. This is the B0.1
    per-model record: one byte per model makes every substituted entry
    exactly recoverable instead of merely inferable from an aggregate.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] > 8:
        raise ValueError(
            f"expected (n_models, n_band<=8), got {mask.shape}"
        )
    weights = (1 << np.arange(mask.shape[1], dtype=np.uint16)).astype(np.uint16)
    return (mask * weights).sum(axis=1).astype(np.uint8)


def unpack_dark_bands(packed, n_band=8):
    """Inverse of `pack_dark_bands`."""
    packed = np.asarray(packed, dtype=np.uint8)[:, None]
    bits = (1 << np.arange(n_band, dtype=np.uint16)).astype(np.uint16)
    return (packed & bits) != 0
