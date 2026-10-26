"""Empirical PACS recovery probabilities, without invented grid endpoints.

These are the numerical building blocks used by the regional products stage. Accepted
positions are covered and outside one beam of the frozen first-pass PACS
detections. Undetected SESNA neighbours remain part of the sky. Rejected
positions are removed from the denominator, not encoded as failed detections.
FIR-only sources and knots remain in the data: their threshold crossings count
as detections, including the measured zero-flux false-positive floor. Neither
endpoint is rescaled. The existing reader's faint clamp remains unchanged;
the production grid must extend to negligible injected signal and a levelled
bright response. The fixed band grid is audited on every appended curve row.
"""

from dataclasses import dataclass

import numba
import numpy as np

from .constants import (DETECTION_SNR, STORAGE_SCALE, PSF_INTEGRATION_ORDER, N_POS,
                        MAG_MIN, MAG_MAX, MAG_STEP, FAINT_MAX_SIGNAL_SNR, ZERO_POINT_JY)


def canonical_grid():
    """The fixed, region-independent magnitude and decreasing-mJy band grid."""
    n=int(round((MAG_MAX-MAG_MIN)/MAG_STEP))+1
    magnitude=MAG_MIN+MAG_STEP*np.arange(n,dtype=np.float64)
    return magnitude,1000*ZERO_POINT_JY*10**(-magnitude/2.5)


@dataclass(frozen=True)
class RecoverySamples:
    """Measured trial statistics, with explicit accepted-position denominators.

    ``thresholds`` and ``accepted`` are (n_cell,n_trial). Other arrays are
    (n_cell,). Rejected thresholds are NaN. A cell with zero accepted trials
    has floor_fraction=NaN and has_curve=False, never a fabricated zero curve.
    floor_counts is measured directly from b/sigma>=5 on accepted positions.
    """

    thresholds: np.ndarray
    accepted: np.ndarray
    n_positions: np.ndarray
    floor_counts: np.ndarray
    floor_fraction: np.ndarray

    @property
    def has_curve(self):
        return self.n_positions > 0


def threshold_fluxes(flux_mjy, sigma_mjy, gain, covered, pacs_excluded):
    """Return measured thresholds and zero-flux diagnostics for accepted trials.

    Every input has shape (n_cell,n_trial). Only ``covered & ~pacs_excluded``
    positions enter counts or denominators. The exclusion mask comes solely
    from the frozen first-pass PACS detections within one beam; undetected SESNA
    neighbours are not excluded. It must not mask arbitrary FIR-only emission.
    Invalid flux, uncertainty, or gain at an accepted position raises: a real
    extraction failure is not silently removed or changed into a non-detection.
    The sampler should refill to 200 accepted positions; this low-level routine
    keeps actual accepted denominators explicit, including zero. The separate
    excluded-area diagnostic must use uniform samples BEFORE such refilling.
    """
    flux = np.asarray(flux_mjy, dtype=np.float64)
    sigma = np.asarray(sigma_mjy, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    covered = np.asarray(covered, dtype=bool)
    excluded = np.asarray(pacs_excluded, dtype=bool)
    if flux.ndim != 2 or flux.shape[1] > 32767:
        raise ValueError("extractor arrays must be (n_cell,n_trial), n_trial<=32767")
    if any(a.shape != flux.shape for a in (sigma, gain, covered, excluded)):
        raise ValueError("extractor arrays must have identical shapes")
    accepted = covered & ~excluded
    valid = np.isfinite(flux) & np.isfinite(sigma) & (sigma > 0)
    valid &= np.isfinite(gain) & (gain > 0)
    if np.any(accepted & ~valid):
        raise ValueError("accepted samples require finite flux and positive finite sigma/gain")
    result = np.full(flux.shape, np.nan, dtype=np.float64)
    statistic = np.full(flux.shape, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        statistic[accepted] = flux[accepted] / sigma[accepted]
        # Algebraically (T*sigma-b)/g, evaluated around the same b/sigma
        # statistic as detection so a threshold tie has the same sign at F=0.
        result[accepted] = ((DETECTION_SNR - statistic[accepted])
                            * (sigma[accepted] / gain[accepted]))
    if not np.all(np.isfinite(result[accepted])):
        raise ValueError("accepted threshold arithmetic overflow")
    n_positions = accepted.sum(axis=1).astype(np.int16)
    floor_counts = (accepted & (statistic >= DETECTION_SNR)).sum(axis=1).astype(np.int16)
    threshold_floor = (accepted & (result <= 0)).sum(axis=1)
    if not np.array_equal(floor_counts, threshold_floor):
        raise ArithmeticError("zero-flux threshold counts disagree with the measured S/N floor")
    floor_fraction = np.full(flux.shape[0], np.nan, dtype=np.float64)
    np.divide(floor_counts, n_positions, out=floor_fraction, where=n_positions > 0)
    values = (result, accepted, n_positions, floor_counts, floor_fraction)
    for value in values:
        value.setflags(write=False)
    return RecoverySamples(*values)


@dataclass(frozen=True)
class CellBackground:
    """Frozen extracted-flux calibration; all flux quantities are mJy."""

    mean_mjy: np.ndarray
    scatter_mjy: np.ndarray
    pixel_sigma_rms_mjy: np.ndarray
    n_positions: np.ndarray


def calibrate_cell_scatter(flux_mjy, pixel_sigma_mjy, covered):
    """Raw sample SD of paired physical-source flux on a predeclared sky pool.

    The caller declares the calibration stream and accepted calibration mask,
    recording its noise/detection sequencing separately; this statistics helper
    does not choose exclusions or iterate detections. Keep calibration positions
    distinct from held-out completeness positions. No clipping, mean subtraction from measurements,
    pixel-variance subtraction, or tuning to a desired floor is performed.
    The sample mean is reported separately. The returned scatter is frozen and
    added in quadrature to propagated pixel sigma in measure_sources, both for
    real sources and for injected trials. Invalid covered extraction raises.
    """
    flux = np.asarray(flux_mjy, dtype=float)
    pixel = np.asarray(pixel_sigma_mjy, dtype=float)
    keep = np.asarray(covered, dtype=bool)
    if flux.ndim != 2 or pixel.shape != flux.shape or keep.shape != flux.shape:
        raise ValueError("calibration arrays must share (cell,position) shape")
    if np.any(keep & (~np.isfinite(flux) | ~np.isfinite(pixel) | (pixel <= 0))):
        raise ValueError("covered calibration positions require valid extracted flux and pixel sigma")
    count = keep.sum(axis=1)
    if np.any(count < 2):
        raise ValueError("at least two covered calibration positions per cell are required")
    mean = np.array([np.mean(f[k]) for f,k in zip(flux,keep)])
    scatter = np.array([np.std(f[k],ddof=1) for f,k in zip(flux,keep)])
    pixel_rms = np.array([np.sqrt(np.mean(p[k]**2)) for p,k in zip(pixel,keep)])
    if not np.all(np.isfinite(mean) & np.isfinite(scatter) & np.isfinite(pixel_rms)):
        raise ValueError("calibration arithmetic overflow")
    for value in (mean,scatter,pixel_rms,count):
        value.flags.writeable = False
    return CellBackground(mean,scatter,pixel_rms,count)


@numba.njit(cache=True)
def _recovery_counts(thresholds, accepted, flux_grid):
    counts = np.full((thresholds.shape[0], flux_grid.size), -1, dtype=np.int16)
    for cell in range(thresholds.shape[0]):
        ordered = np.sort(thresholds[cell][accepted[cell]])
        if ordered.size == 0:
            continue
        for plane in range(flux_grid.size):
            counts[cell, plane] = np.searchsorted(ordered, flux_grid[plane], side="right")
    return counts


def recovery_counts(samples, flux_grid_mjy):
    """Count individual detections per cell on an explicit bright-to-faint grid.

    ``samples`` comes from threshold_fluxes. At equality a source is detected
    (S/N>=5). No FIR-only contaminants are rejected. Zero-accepted cells return
    a row of -1 counts, explicitly meaning NO CURVE; use samples.has_curve to
    select rows before encoding, or refill them. Empty bands return (0,n_plane).
    No faint-end zero or bright-end unity is imposed.
    """
    if not isinstance(samples, RecoverySamples):
        raise TypeError("samples must be a RecoverySamples result from threshold_fluxes")
    grid = _flux_grid(flux_grid_mjy)
    return _recovery_counts(samples.thresholds, samples.accepted, grid)


def _flux_grid(flux_grid_mjy):
    grid = np.asarray(flux_grid_mjy, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("flux grid must be a vector with at least two planes")
    if not np.all(np.isfinite(grid) & (grid > 0)) or not np.all(np.diff(grid) < 0):
        raise ValueError("flux grid must be finite, positive, and strictly decreasing")
    return grid


def verify_zero_flux_floor(samples, zero_flux_statistic):
    """Check measured zero-injection S/N against the CDF floor, cell by cell.

    ``zero_flux_statistic`` must be independently measured by the extractor at
    the SAME candidate positions, in their original order. Only samples.accepted
    enters the comparison. A mismatch raises, including a position-level mismatch
    that happens to cancel in the per-cell total. Return the measured floor counts;
    zero-accepted cells still have no probability (samples.floor_fraction is NaN).
    """
    statistic = np.asarray(zero_flux_statistic, dtype=np.float64)
    if statistic.shape != samples.accepted.shape:
        raise ValueError("zero-flux statistic must match the original candidate positions")
    if not np.all(np.isfinite(statistic[samples.accepted])):
        raise ValueError("accepted zero-flux measurements must have finite detection statistics")
    measured = samples.accepted & (statistic >= DETECTION_SNR)
    predicted = samples.accepted & (samples.thresholds <= 0)
    if not np.array_equal(measured, predicted):
        raise AssertionError("measured zero-flux detections disagree with CDF floor positions")
    counts = measured.sum(axis=1).astype(np.int16)
    if not np.array_equal(counts, samples.floor_counts):
        raise AssertionError("measured zero-flux detections disagree with stored floor counts")
    return counts


def grid_diagnostics(samples, flux_grid_mjy, sigma_mjy, gain):
    """Audit an explicit grid without choosing a scientific endpoint tolerance.

    Return per-cell ``faint_max_signal_to_noise`` = max(g*F_faint/sigma) over
    accepted positions and ``bright_unrecovered_count`` = number of accepted
    thresholds above F_bright. Zero-accepted cells use NaN and -1 respectively.
    These report whether the faint injection is negligible and whether the bright
    response has levelled off; they never alter counts, endpoints, or the grid.
    """
    grid = _flux_grid(flux_grid_mjy)
    sigma = np.asarray(sigma_mjy, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    if sigma.shape != samples.accepted.shape or gain.shape != sigma.shape:
        raise ValueError("grid-audit noise/gain must match original candidate positions")
    valid = np.isfinite(sigma) & (sigma > 0) & np.isfinite(gain) & (gain > 0)
    if np.any(samples.accepted & ~valid):
        raise ValueError("accepted grid-audit noise/gain must be positive and finite")
    signal = np.full(sigma.shape, -np.inf, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        signal[samples.accepted] = gain[samples.accepted] * grid[-1] / sigma[samples.accepted]
    if not np.all(np.isfinite(signal[samples.accepted])):
        raise ValueError("grid-audit signal-to-noise overflow")
    faint = np.max(signal, axis=1, initial=-np.inf)
    faint[~samples.has_curve] = np.nan
    bright = (samples.accepted & (samples.thresholds > grid[0])).sum(axis=1).astype(np.int16)
    bright[~samples.has_curve] = -1
    return {"faint_max_signal_to_noise": faint,
            "bright_unrecovered_count": bright}


class CurveTable:
    """Measured singleton or group-target rows on an explicit fixed band grid.

    Supply accepted positions and matching frozen extraction noise. Group
    targets use their full joint functional, never singleton statistics.
    Append stable native map/cell rows first, then stable group/target rows.
    Only encoded tables and small diagnostics accumulate, not pixel cutouts.
    """

    def __init__(self, flux_grid_mjy, *, faint_signal_limit=FAINT_MAX_SIGNAL_SNR):
        self.flux_grid=_flux_grid(flux_grid_mjy).copy()
        if not np.isfinite(faint_signal_limit) or faint_signal_limit<=0:
            raise ValueError("positive finite faint signal audit limit required")
        self.faint_signal_limit=float(faint_signal_limit)
        self._tables=[]
        self._floors=[]
        self.n_rows=0
        self.max_faint_signal_to_noise=0.

    def append(self,flux_mjy,sigma_mjy,gain,*,zero_flux_statistic=None):
        flux=np.asarray(flux_mjy,float)
        if flux.ndim!=2 or flux.shape[1]!=N_POS:
            raise ValueError("published curves require exactly N_POS accepted positions per row")
        accepted=np.ones(flux.shape,bool)
        samples=threshold_fluxes(flux,sigma_mjy,gain,accepted,~accepted)
        if zero_flux_statistic is not None:
            verify_zero_flux_floor(samples,zero_flux_statistic)
        counts=recovery_counts(samples,self.flux_grid)
        audit=grid_diagnostics(samples,self.flux_grid,sigma_mjy,gain)
        if np.any(audit['bright_unrecovered_count']!=0):
            raise ValueError("canonical bright endpoint has not reached measured plateau; extend band constant")
        if np.any(audit['faint_max_signal_to_noise']>self.faint_signal_limit):
            raise ValueError("canonical faint endpoint is not negligible relative to frozen noise")
        if not np.array_equal(counts[:,-1],samples.floor_counts):
            raise ValueError("canonical faint count differs from measured zero-injection floor")
        first=self.n_rows
        self.n_rows+=flux.shape[0]
        if self.n_rows>np.iinfo(np.int32).max:
            raise ValueError("curve row identifiers exceed int32")
        self._tables.append(encode_curves(counts,samples.n_positions))
        self._floors.append(samples.floor_counts)
        self.max_faint_signal_to_noise=max(self.max_faint_signal_to_noise,
            float(np.max(audit['faint_max_signal_to_noise'],initial=0.)))
        return np.arange(first,self.n_rows,dtype=np.int32)

    def arrays(self):
        table=np.concatenate(self._tables,axis=0) if self._tables else np.empty((0,self.flux_grid.size),np.uint16)
        return {'curves':table,'n_positions':np.full(self.n_rows,N_POS,np.int16),
                'completed_from':np.full(self.n_rows,np.nan,np.float32),
                'envelope':table.max(axis=0) if self.n_rows else np.zeros(self.flux_grid.size,np.uint16),
                'floor_counts':np.concatenate(self._floors) if self._floors else np.empty(0,np.int16)}


def verify_injection_batch(image, error, coverage, templates, annulus_mask, *,
                           min_coverage, cell_shape, flux_grid_mjy,
                           component_source, cell_scatter_mjy, injection_template,
                           target_source=0, rtol=1e-9, atol=1e-10):
    """Rebuild and verify actual cutouts at zero and every explicit grid flux.

    Inputs follow extractor.prepare_fit: image/error are mJy/pixel and templates
    are pixel response per mJy TOTAL source flux. Flattened batch order must be
    cell-major for ``cell_shape=(n_cell,n_accepted)``. All supplied positions
    must already be covered and outside the detected-PACS beam exclusion; the separate
    fixed-pool area estimate belongs to geometry.sample_cells. Invalid accepted
    extraction raises. Real FIR-only background emission is retained unchanged.

    This diagnostic writes no products, selects no PSF/beam/grid calibration,
    and verifies one template hypothesis at a time. Tolerances below are solely
    numerical comparisons, not faint/bright scientific acceptance thresholds.
    Memory is bounded by the caller's cutout batch; injections run serially.
    ``cell_scatter_mjy`` must be a frozen calibration, broadcastable to
    (n_fit,n_physical_source); two template columns map to each physical source
    via component_source. injection_template is the explicit unit-flux physical
    source profile, independent of the two unconstrained fitting coefficients.
    """
    from .extractor import prepare_fit, measure_sources

    data = np.asarray(image, dtype=np.float64)
    psf = np.asarray(templates, dtype=np.float64)
    shape = tuple(cell_shape)
    if (len(shape) != 2 or any(int(v) != v or v <= 0 for v in shape)
            or data.ndim != 3 or data.shape[0] != np.prod(shape)):
        raise ValueError("positive cell_shape must match the flattened cutout batch")
    shape = tuple(int(v) for v in shape)
    if (psf.ndim != 4 or psf.shape[0] != data.shape[0]
            or psf.shape[2:] != data.shape[1:]
            or int(target_source) != target_source):
        raise ValueError("templates and target source must match cutouts")
    source = np.asarray(component_source)
    if source.shape != (psf.shape[1],) or not np.issubdtype(source.dtype,np.integer):
        raise ValueError("component_source must label every template column")
    target_source = int(target_source)
    if target_source not in source:
        raise ValueError("target source has no template columns")
    injection = np.asarray(injection_template,dtype=float)
    if injection.shape != data.shape or not np.all(np.isfinite(injection)):
        raise ValueError("explicit physical-source injection template must match cutouts")
    grid = _flux_grid(flux_grid_mjy)

    def extract(pixels):
        context = prepare_fit(pixels, error, coverage, psf, annulus_mask,
                              min_coverage=min_coverage)
        measured = measure_sources(pixels,context,psf,source,
                                   cell_scatter_mjy=cell_scatter_mjy)
        if not np.all(measured.valid & measured.identifiable[:, target_source]):
            raise ValueError("accepted position failed extraction; do not remove it from denominator")
        operator = context.operator[:,np.flatnonzero(source == target_source)].sum(axis=1)
        gain = np.einsum('bp,bp->b',operator,injection.reshape(data.shape[0],-1))
        return (measured.flux[:,target_source].reshape(shape),
                measured.sigma[:,target_source].reshape(shape),gain.reshape(shape))

    flux, sigma, gain = extract(data)
    accepted = np.ones(shape, dtype=bool)
    samples = threshold_fluxes(flux, sigma, gain, accepted, ~accepted)
    predicted = recovery_counts(samples, grid)
    # Independent full preparation/measurement on an explicit zero injection.
    zero_flux, zero_sigma, _ = extract(data + 0.0 * injection)
    verify_zero_flux_floor(samples, zero_flux / zero_sigma)
    direct = np.empty_like(predicted)
    max_sigma_relative = 0.0
    max_flux_absolute = 0.0
    for plane, injected in enumerate(grid):
        injected_flux, injected_sigma, injected_gain = extract(
            data + injected * injection)
        np.testing.assert_allclose(injected_sigma, sigma, rtol=rtol, atol=atol)
        np.testing.assert_allclose(injected_gain, gain, rtol=rtol, atol=atol)
        expected_flux = flux + gain * injected
        np.testing.assert_allclose(injected_flux, expected_flux, rtol=rtol, atol=atol)
        direct[:, plane] = (injected_flux / injected_sigma >= DETECTION_SNR).sum(axis=1)
        max_sigma_relative = max(max_sigma_relative,
                                 float(np.max(np.abs(injected_sigma / sigma - 1))))
        max_flux_absolute = max(max_flux_absolute,
                                float(np.max(np.abs(injected_flux - expected_flux))))
    if not np.array_equal(direct, predicted):
        raise AssertionError("full-rebuild injection counts differ from threshold CDF")
    return {"samples": samples, "analytic_counts": predicted,
            "injected_counts": direct, "grid": grid_diagnostics(samples, grid, sigma, gain),
            "max_sigma_relative_error": max_sigma_relative,
            "max_flux_absolute_error_mjy": max_flux_absolute}


def run_map_injection_pilot(open_map, cell_ids, neighbours, psf, *, beam_arcsec,
                            half_size, annulus_arcsec, min_coverage, flux_grid_mjy,
                            seed, area_draws, max_draws, cell_scatter_mjy,
                            injection_weights, n_positions=200,
                            integration_order=PSF_INTEGRATION_ORDER):
    """Print bounded per-cell real-map checks for an explicit PSF hypothesis.

    This is a verification runner, not a production curve builder. Caller
    selects representative interior cells, supplies a published/calibrated PSF
    hypothesis and exclusion beam, and supplies the explicit diagnostic grid.
    Each cell is sampled/fitted independently to bound cutout memory. It returns
    small count/area summaries; never writes products or claims full-map coverage.
    ``neighbours`` is the frozen first-pass detected-PACS set, also used for
    grouping. ``cell_scatter_mjy`` is a supplied frozen calibration consistent
    with that set; this runner does not choose calibration/detection sequencing.
    The fit uses two unconstrained scan columns;
    ``injection_weights`` describes only the explicitly tested injected source.
    """
    from .geometry import MapGrid, DetectedPacsNeighbours, SamplingBudgetError, sample_cells
    from .photometry import prepare_positions, positions_covered

    grid = MapGrid.from_wcs(open_map.wcs, open_map.image.shape)
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or not ids.size or np.unique(ids).size != ids.size:
        raise ValueError("pilot requires a nonempty unique cell vector")
    grid.bounds(ids)
    _flux_grid(flux_grid_mjy)
    if not isinstance(neighbours,DetectedPacsNeighbours):
        raise TypeError("current completeness requires the frozen detected-PACS set")
    scatter = np.broadcast_to(np.asarray(cell_scatter_mjy,float),(ids.size,))
    if np.any(~np.isfinite(scatter) | (scatter<0)):
        raise ValueError("supply one frozen extracted-flux scatter per cell")
    weights = np.asarray(injection_weights,float)
    if weights.shape != (2,) or np.any(~np.isfinite(weights) | (weights<0)) or not np.isclose(weights.sum(),1):
        raise ValueError("supply two normalized injection weights, not fitting weights")

    def usable(x, y):
        sky = grid.wcs.pixel_to_world(x, y).icrs
        return positions_covered(open_map, sky.ra.deg.ravel(), sky.dec.deg.ravel(),
                                 half_size=half_size,
                                 min_coverage=min_coverage).reshape(x.shape)

    def raw(x, y):
        ix = np.clip(np.floor(x + 0.5).astype(int), 0, grid.shape[1] - 1)
        iy = np.clip(np.floor(y + 0.5).astype(int), 0, grid.shape[0] - 1)
        values = open_map.coverage[iy, ix]
        return np.isfinite(values) & (values > 0)

    results = []
    for cell_index,cell_id in enumerate(ids):
        try:
            selected = sample_cells(grid, np.asarray([cell_id]), neighbours, beam_arcsec,
                                    usable, seed=seed, area_draws=area_draws,
                                    max_draws=max_draws, n_positions=n_positions,
                                    raw_covered_at_pixels=raw)
        except SamplingBudgetError as exc:
            result = {"cell_id": int(cell_id), "status": "sampling_budget_exhausted",
                      "n_available": exc.n_available, "area": exc.area,
                      "raw_covered_fraction": exc.raw_covered_fraction}
            results.append(result)
            print("PACS pilot cell=%d NO CURVE accepted_in_budget=%d "
                  "raw_area=%g usable_area=%g pacs_excluded=%g" % (
                      cell_id, exc.n_available[0], exc.raw_covered_fraction[0],
                      exc.area.covered_fraction[0], exc.area.excluded_fraction[0]), flush=True)
            continue
        sky = grid.wcs.pixel_to_world(selected.x.ravel(), selected.y.ravel()).icrs
        cutouts = prepare_positions(open_map, sky.ra.deg, sky.dec.deg, psf,
                                    half_size=half_size, annulus_arcsec=annulus_arcsec,
                                    min_coverage=min_coverage,
                                    integration_order=integration_order)
        if not np.all(cutouts.covered):
            raise AssertionError("accepted sample coverage changed during cutout preparation")
        result = verify_injection_batch(
            cutouts.image, cutouts.error, cutouts.coverage, cutouts.templates,
            cutouts.annulus_mask, min_coverage=min_coverage,
            cell_shape=selected.x.shape, flux_grid_mjy=flux_grid_mjy,
            component_source=np.array([0,0]),cell_scatter_mjy=scatter[cell_index],
            injection_template=np.einsum('k,bkyx->byx',weights,cutouts.templates))
        result.update(cell_id=int(cell_id), status="verified", area=selected.area,
                      raw_covered_fraction=selected.raw_covered_fraction,
                      draws_used=selected.draws_used)
        results.append(result)
        print("PACS pilot cell=%d n=%d floor=%g raw_area=%g usable_area=%g "
              "pacs_excluded=%g sigma_rel=%g faint_snr=%g bright_remaining=%d" % (
                  cell_id, result["samples"].n_positions[0],
                  result["samples"].floor_fraction[0], selected.raw_covered_fraction[0],
                  selected.area.covered_fraction[0], selected.area.excluded_fraction[0],
                  result["max_sigma_relative_error"],
                  result["grid"]["faint_max_signal_to_noise"][0],
                  result["grid"]["bright_unrecovered_count"][0]), flush=True)
    return results


def encode_curves(counts, n_positions):
    """Encode recovery counts as uint16 fractions x600; 200 trials are exact.

    ``n_positions`` is one positive integer or a vector with one denominator
    per cell, counting covered trials outside the detected-PACS exclusion. Cells
    with zero accepted positions cannot be encoded as curves; filter explicitly
    with RecoverySamples.has_curve or refill before calling this function.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be a two-dimensional integer array")
    n = np.asarray(n_positions)
    if not np.issubdtype(n.dtype, np.integer) or n.ndim > 1:
        raise ValueError("n_positions must be an integer or integer vector")
    if n.ndim == 1 and n.shape != (counts.shape[0],):
        raise ValueError("one denominator is required per cell")
    if np.any(n <= 0) or np.any(n > 32767):
        raise ValueError("position counts must be in 1..32767")
    denominator = n if n.ndim == 0 else n[:, None]
    if np.any(counts < 0) or np.any(counts > denominator):
        raise ValueError("recovery counts must lie between zero and the denominator")
    return np.rint(counts.astype(np.float64) * STORAGE_SCALE / denominator).astype(np.uint16)


@dataclass(frozen=True)
class PixelCurves:
    """Conditional pixel curves, with coverage and exclusion areas kept separate."""

    curves: np.ndarray
    covered_fraction: np.ndarray
    excluded_fraction: np.ndarray
    has_curve: np.ndarray


@numba.njit(cache=True)
def _pixel_curves(curves, sample_rows, covered, excluded):
    result = np.zeros((sample_rows.shape[0], curves.shape[1]), dtype=np.float32)
    coverage_fraction = np.zeros(sample_rows.shape[0], dtype=np.float32)
    excluded_fraction = np.full(sample_rows.shape[0], np.nan, dtype=np.float32)
    has_curve = np.zeros(sample_rows.shape[0], dtype=np.bool_)
    for pixel in range(sample_rows.shape[0]):
        count = 0
        n_covered = 0
        n_excluded = 0
        sums = np.zeros(curves.shape[1], dtype=np.float64)
        for sample in range(sample_rows.shape[1]):
            if covered[pixel, sample]:
                n_covered += 1
                if excluded[pixel, sample]:
                    n_excluded += 1
                    continue
                row = sample_rows[pixel, sample]
                count += 1
                for plane in range(curves.shape[1]):
                    sums[plane] += curves[row, plane]
        coverage_fraction[pixel] = n_covered / sample_rows.shape[1]
        if n_covered:
            excluded_fraction[pixel] = n_excluded / n_covered
        if count:
            has_curve[pixel] = True
            for plane in range(curves.shape[1]):
                result[pixel, plane] = sums[plane] / (STORAGE_SCALE * count)
        elif n_covered:
            result[pixel, :] = np.nan
    return result, coverage_fraction, excluded_fraction, has_curve


def pixel_curves(curves, sample_rows, covered, pacs_excluded):
    """Average accepted-area curves; report coverage and exclusion separately.

    ``curves`` is uint16 x600; ``sample_rows`` is (n_pixel, 256), one selected
    tile/cell row per equal-area nested child. Explicit ``covered`` and
    ``pacs_excluded`` masks have the same shape; -1 rows never infer coverage.
    Only covered nonexcluded samples enter the mean, and every such sample must
    have a curve row. Excluded samples can lack rows. Each sample occurs once
    even where tiles overlap. Return PixelCurves: covered_fraction is covered
    area/whole pixel, excluded_fraction is excluded covered area/covered area
    (NaN if no coverage). Neither fraction scales the conditional mean curve.
    Covered pixels with no accepted area have NaN curves and has_curve=False;
    truly uncovered pixels retain zero curves with covered_fraction=0.
    """
    curves = np.asarray(curves)
    rows = np.asarray(sample_rows)
    covered = np.asarray(covered, dtype=bool)
    excluded = np.asarray(pacs_excluded, dtype=bool)
    if curves.ndim != 2 or curves.dtype != np.uint16 or np.any(curves > STORAGE_SCALE):
        raise ValueError("curves must be a uint16 (n_cell,n_plane) table in 0..600")
    if rows.ndim != 2 or rows.shape[1] != 256 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("sample_rows must be an integer (n_pixel,256) table")
    if covered.shape != rows.shape or excluded.shape != rows.shape:
        raise ValueError("pixel coverage and detected-PACS exclusion masks must match sample rows")
    accepted = covered & ~excluded
    if np.any(accepted & ((rows < 0) | (rows >= curves.shape[0]))):
        raise ValueError("accepted covered sample has no valid curve row")
    return PixelCurves(*_pixel_curves(curves, rows, covered, excluded))
