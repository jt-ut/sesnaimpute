"""
reference.py
====================================================================
The reference state C0: each model's band fluxes at the survey
apertures, in the library's stored frame.

f_ref is the denominator every fitted brightness is priced against --
the fitter reports B_hat as a multiplier on exactly these numbers, so
getting the convention wrong displaces every B in the census by a
constant. Two cases:

APERTURE-INDEPENDENT libraries (sps, pahc, h2shock, galz): f_ref IS the
stored convolved flux. There is one aperture slice, it carries a
sentinel value (1e-30) that nothing reads, and no interpolation happens.
Note this is not the same as "the aperture does not matter": pahc's
contamination amplitude is a free grid axis (R), which is why one SED
can serve three different survey apertures without contradiction.

APERTURE-DEPENDENT libraries (yso): f_ref is the 20-slice curve of
growth INTERPOLATED to that band's survey aperture, converted to AU at
the reference distance. Three conventions have to match sedfitter
exactly or the census is comparing against a different template than the
fit used:

  - aperture_AU(band) = BANDS[band].aperture_arcsec * d_ref[pc], with
    aperture_arcsec a RADIUS (Models.read converts it exactly this way);
  - interpolation LINEAR IN AU on raw linear flux, matching scipy
    interp1d's default as used by ConvolvedFluxes.interpolate -- not
    log-log, which differs by up to 0.5%;
  - d_ref = 1 kpc, because sedfitter hardcodes a 1 kpc frame
    (`conv.flux * (u.kpc / m.distances)**2`) regardless of what the
    library's DISTANCE header declares. At d_ref = 1 kpc that factor is
    exactly 1, so f_ref needs no distance scaling at all. Any other
    d_ref would require applying it by hand.

DLOGF_DLOGAP exists because f_ref is a family indexed by distance, not a
single SED: the fitter evaluates the template at 10^SC, not at d_ref.
The local log-log slope lets a consumer correct to a nearby distance to
first order without re-reading the cube. Measured colour drift across a
0.7-1.0 kpc range is a median 0.0147 dex, 90th percentile 0.093 -- small
but not negligible in the tail.

R_HALF_SB is the outermost radius where the annular surface brightness
still exceeds half its peak -- sedfitter's own resolvedness measure,
computed here from the RAW 20-slice grid as intended. sedfitter's
in-fitter version computes it from the already-interpolated,
already-distance-dimmed array, which degenerates into "veto the nearest
distance" regardless of model size; it is also inert under memmap. This
is the correct quantity, stored for the pending remove_resolved ruling
and any Tier-2 completeness term.
====================================================================
"""

from dataclasses import dataclass, field

import numpy as np

from sesnaimpute.sed_models.constants import BANDS
from sesnaimpute.sed_models.curate import curve_of_growth as cog

__all__ = ["ReferenceState", "derive", "ApertureRangeError"]

_BAND_KEYS = list(BANDS)
_D_REF_KPC = 1.0
_PC_CM = 3.0856775814913673e18
#: sedfitter hardcodes a 1 kpc frame; a library declaring otherwise is
#: silently misscaled by the fitter, so we check rather than adapt.
_EXPECTED_DISTANCE_CM = 1000.0 * _PC_CM


class ApertureRangeError(RuntimeError):
    """A requested aperture falls outside the library's stored grid."""


@dataclass(frozen=True)
class ReferenceState:
    """C0 for one library."""

    bands: tuple
    aperture_arcsec: np.ndarray        # (8,) radii, from constants.BANDS
    d_ref_kpc: float
    aperture_dependent: bool

    f_ref: np.ndarray                  # (n_models, 8) mJy

    # Aperture-dependent libraries only; None otherwise.
    aperture_au: np.ndarray = None     # (8,) at d_ref
    dlogf_dlogap: np.ndarray = None    # (n_models, 8)
    r_half_sb: np.ndarray = None       # (n_models, 8) AU
    encfrac: np.ndarray = None         # (n_models, 8) fraction of widest slice
    floor_linear: np.ndarray = None    # (n_models,) shared B0 floor, NOT applied

    diagnostics: dict = field(default_factory=dict)


def _bracket_index(ap_grid, xq):
    """Index of the lower bracketing knot, for the local log-log slope."""
    k = int(np.searchsorted(ap_grid, xq) - 1)
    return min(max(k, 0), len(ap_grid) - 2)


def _loglog_slope(ap_grid, flux, k):
    """Local d log10 f / d log10 aperture across the bracketing interval.

    NaN where either bracketing flux is non-positive -- a real occurrence
    in the YSO grids, where a fraction of models have zero flux in the
    near-IR bands.
    """
    f0, f1 = flux[:, k], flux[:, k + 1]
    a0, a1 = ap_grid[k], ap_grid[k + 1]
    out = np.full(f0.shape, np.nan)
    ok = (f0 > 0) & (f1 > 0) & (a1 > a0)
    out[ok] = (np.log10(f1[ok]) - np.log10(f0[ok])) / (np.log10(a1) - np.log10(a0))
    return out


def _radius_half_peak_sb(ap_grid, flux, fraction=0.5):
    """sedfitter's find_radius_sigma, on the raw curve of growth.

    Annular surface brightness between consecutive apertures; return the
    OUTERMOST radius where it still exceeds `fraction` of its peak.
    Faithful to sedfitter's own implementation (ConvolvedFluxes.
    find_radius_sigma) so the number means what the fitter means by it.
    """
    n_models, n_ap = flux.shape
    sigma = np.zeros_like(flux, dtype=float)
    sigma[:, 0] = flux[:, 0] / ap_grid[0] ** 2
    sigma[:, 1:] = np.diff(flux, axis=1) / (ap_grid[1:] ** 2 - ap_grid[:-1] ** 2)
    peak = np.nanmax(sigma, axis=1)
    radius = np.zeros(n_models)
    thresh = fraction * peak
    for ia in range(n_ap - 2, -1, -1):
        calc = (sigma[:, ia] > thresh) & (radius == 0.0)
        if not np.any(calc):
            continue
        denom = sigma[calc, ia] - sigma[calc, ia + 1]
        step = np.where(
            denom != 0,
            (sigma[calc, ia] - thresh[calc]) / np.where(denom != 0, denom, 1.0),
            0.0,
        )
        radius[calc] = ap_grid[ia] + step * (ap_grid[ia + 1] - ap_grid[ia])
    radius[sigma[:, -1] > thresh] = ap_grid[-1]
    return radius


def derive(lib, d_ref_kpc=_D_REF_KPC):
    """Build the reference state for one library.

    d_ref_kpc defaults to 1.0 and should stay there: sedfitter's 1 kpc
    frame is hardcoded, so any other value silently needs a (1/d)^2
    correction applied by hand.
    """
    ap_arcsec = np.array([BANDS[b].aperture_arcsec for b in _BAND_KEYS], dtype=float)
    bands = tuple(_BAND_KEYS)
    conv = [lib.convolved(b) for b in _BAND_KEYS]
    n_models = lib.meta.n_models

    diag = {
        "d_ref_kpc": float(d_ref_kpc),
        "aperture_arcsec": ap_arcsec.tolist(),
        "n_models": n_models,
    }

    if not lib.meta.aperture_dependent:
        f_ref = np.column_stack([c.flux_mjy[:, 0] for c in conv])
        # The B0 floor is computed for EVERY library, not only the
        # aperture-gridded ones: density.py takes log10(f_ref) for all of
        # them, so all of them need a floor defined. Here the "full cube"
        # is a single aperture slice, so the cube peak and the 8-band peak
        # coincide and cog's anchor reduces to the same number -- but it
        # is computed through the shared function rather than reproduced,
        # so the two branches cannot drift apart.
        flux_cubes = {b: np.asarray(c.flux_mjy, dtype=float)
                      for b, c in zip(bands, conv)}
        _, _, floor_linear, all_zero = cog.peak_and_floor(flux_cubes)
        diag["mode"] = "aperture-independent: stored slice 0, no interpolation"
        diag["zero_flux_fraction"] = [
            float(np.mean(f_ref[:, i] <= 0)) for i in range(len(bands))
        ]
        diag["floor_dex_below_peak"] = float(cog.FLOOR_DEX_BELOW_PEAK)
        diag["n_below_floor"] = [
            int(np.sum(f_ref[:, i] < floor_linear)) for i in range(len(bands))
        ]
        diag["n_all_zero_flux"] = int(all_zero.sum())
        return ReferenceState(
            bands=bands, aperture_arcsec=ap_arcsec, d_ref_kpc=float(d_ref_kpc),
            aperture_dependent=False, f_ref=f_ref, floor_linear=floor_linear,
            diagnostics=diag,
        )

    # ---------------------------------------------------- aperture-gridded
    # The library's declared frame is CHECKED, not adapted to: sedfitter
    # hardcodes 1 kpc, so a library saying otherwise is misscaled by the
    # fitter and must fail here rather than diverge quietly.
    if not np.isclose(lib.distance_cm, _EXPECTED_DISTANCE_CM, rtol=1e-6):
        raise ApertureRangeError(
            f"{lib.key}: flux.fits declares DISTANCE = {lib.distance_cm:.6g} cm "
            f"({lib.distance_cm / _PC_CM / 1000:.4f} kpc), but sedfitter hardcodes "
            f"a 1 kpc frame (models.py:334) and never reads this header. f_ref "
            f"would not be the template the fitter compares against."
        )

    grid = np.asarray(conv[0].apertures_au, dtype=float)
    flux_cubes = {b: np.asarray(c.flux_mjy, dtype=float)
                  for b, c in zip(bands, conv)}
    targets = cog.survey_aperture_au(lib.distance_cm)
    cog.assert_target_interior(grid, {b: targets[b] for b in bands},
                               context=f"{lib.key}")
    _, _, floor_linear, all_zero = cog.peak_and_floor(flux_cubes)

    ap_au_req = np.array([targets[b] for b in bands], dtype=float)
    f_ref = np.empty((n_models, len(bands)))
    slope = np.empty((n_models, len(bands)))
    rhalf = np.empty((n_models, len(bands)))
    encf = np.empty((n_models, len(bands)))
    grid_span = []

    for i, b in enumerate(bands):
        fl = flux_cubes[b]
        g = np.asarray(conv[i].apertures_au, dtype=float)
        if g.ndim != 1 or fl.shape[1] != len(g):
            raise ApertureRangeError(
                f"{lib.key}/{b}: flux shape {fl.shape} does not "
                f"match {len(g)} apertures"
            )
        # Shared implementation -- linear in AU, raw, NO floor applied.
        # Clamping inputs would make the synthetic floor an interpolation
        # ENDPOINT, which is the mechanism behind the heavy tail the YSO
        # curation pass measured (median f_hi/f_lo = 60 among affected
        # models, worst case 15 dex low).
        f_ref[:, i] = cog.interp_at_aperture(fl, g, targets[b])
        k = _bracket_index(g, targets[b])
        slope[:, i] = _loglog_slope(g, fl, k)
        rhalf[:, i] = _radius_half_peak_sb(g, fl)
        with np.errstate(invalid="ignore", divide="ignore"):
            encf[:, i] = np.where(fl[:, -1] > 0, f_ref[:, i] / fl[:, -1], np.nan)
        grid_span.append((float(g[0]), float(g[-1]), int(k)))

    diag["mode"] = ("aperture-dependent: 20-slice curve of growth, "
                    "linear-in-AU interpolation at d_ref")
    diag["aperture_au_at_d_ref"] = ap_au_req.tolist()
    diag["stored_grid_span_au"] = grid_span
    diag["zero_flux_fraction"] = [
        float(np.mean(f_ref[:, i] <= 0)) for i in range(len(bands))
    ]
    diag["encfrac_median"] = [
        float(np.nanmedian(encf[:, i])) for i in range(len(bands))
    ]
    diag["dlogf_dlogap_median"] = [
        float(np.nanmedian(slope[:, i])) for i in range(len(bands))
    ]

    # f_ref is stored RAW. The floor travels beside it rather than baked
    # in, because the C0 identity f_model = f_ref * B_hat * 10^(-0.4 Av k)
    # has to hold on true values -- a floored f_ref makes it false for
    # exactly the dark bands, and I3 would then flag a healthy library.
    # Consumers that take a log apply `np.maximum(f_ref, floor_linear)`.
    diag["floor_dex_below_peak"] = float(cog.FLOOR_DEX_BELOW_PEAK)
    diag["n_below_floor"] = [
        int(np.sum(f_ref[:, i] < floor_linear)) for i in range(len(bands))
    ]
    diag["n_all_zero_flux"] = int(all_zero.sum())

    return ReferenceState(
        bands=bands, aperture_arcsec=ap_arcsec, d_ref_kpc=float(d_ref_kpc),
        aperture_dependent=True, f_ref=f_ref, aperture_au=ap_au_req,
        dlogf_dlogap=slope, r_half_sb=rhalf, encfrac=encf,
        floor_linear=floor_linear, diagnostics=diag,
    )
