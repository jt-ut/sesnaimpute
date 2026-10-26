"""Fixed-position linear cutout fits shared by PACS photometry and injection.

This numerical core accepts calibrated, pixel-integrated PSF columns; it does
not manufacture a PACS PSF or choose scientific coverage/separability limits.
Inputs are bounded batches with the same number of members and cutout shape.
Every member, including a nuisance neighbour, remains in the design matrix.

``prepare_fit`` propagates error-layer pixel noise only. Annulus residuals are
diagnostics, never an additional pixel-noise variance. ``measure_sources`` sums
the paired columns of each physical source and adds frozen cell-level scatter
of empty-sky EXTRACTED FLUX in quadrature. Both terms are always present in that
physical-source API. All flux units must be consistent (normally mJy).
"""

from dataclasses import dataclass

import numba
import numpy as np


@dataclass(frozen=True)
class FitContext:
    shape: tuple
    operator: np.ndarray
    covariance: np.ndarray
    gain: np.ndarray
    identifiable: np.ndarray
    group_identifiable: np.ndarray
    valid: np.ndarray
    fit_mask: np.ndarray
    annulus_scatter: np.ndarray
    annulus_count: np.ndarray
    rank: np.ndarray
    condition: np.ndarray
    annulus_identifiable: np.ndarray
    column_scale: np.ndarray
    scaled_projection: np.ndarray


@dataclass(frozen=True)
class Measurements:
    flux: np.ndarray
    sigma: np.ndarray
    gain: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    identifiable: np.ndarray
    group_total: np.ndarray
    group_sigma: np.ndarray
    group_identifiable: np.ndarray
    background: np.ndarray
    valid: np.ndarray


@numba.njit(cache=True)
def _annulus_statistics(image, error, coverage, annulus, min_coverage):
    """Unclipped sample scatter; clipping would be another data-dependent rule."""
    n_fit, n_pixel = image.shape
    scatter = np.empty(n_fit, dtype=np.float64)
    count = np.zeros(n_fit, dtype=np.int64)
    for i in range(n_fit):
        mean = 0.0
        m2 = 0.0
        for j in range(n_pixel):
            if (annulus[i, j] and np.isfinite(image[i, j])
                    and np.isfinite(error[i, j]) and error[i, j] > 0.0
                    and np.isfinite(coverage[i, j])
                    and coverage[i, j] >= min_coverage):
                count[i] += 1
                delta = image[i, j] - mean
                mean += delta / count[i]
                m2 += delta * (image[i, j] - mean)
        scatter[i] = np.sqrt(m2 / (count[i] - 1)) if count[i] > 1 else np.nan
    return scatter, count


def _readonly(array):
    array.setflags(write=False)
    return array


@numba.njit(cache=True)
def _prepare_singleton(data, errors, cover, psf, annulus, min_coverage):
    """Centered two-column WLS; same model and covariance as the SVD path.

    Normalize weights by the smallest valid error to avoid unnecessarily
    large weights. This scalar cancels from the estimator and condition.
    Numerically delicate cases are returned to the general SVD implementation.
    """
    n_fit, n_pixel = data.shape
    operator = np.zeros((n_fit, 2, n_pixel), dtype=np.float64)
    covariance = np.zeros((n_fit, 2, 2), dtype=np.float64)
    gain = np.zeros((n_fit, 1), dtype=np.float64)
    identifiable = np.ones((n_fit, 1), dtype=np.bool_)
    group_identifiable = np.ones(n_fit, dtype=np.bool_)
    valid = np.zeros(n_fit, dtype=np.bool_)
    mask = np.zeros((n_fit, n_pixel), dtype=np.bool_)
    scatter = np.full(n_fit, np.nan)
    count = np.zeros(n_fit, dtype=np.int64)
    rank = np.full(n_fit, 2, dtype=np.int64)
    condition = np.full(n_fit, np.inf)
    annulus_identifiable = np.ones(n_fit, dtype=np.bool_)
    column_scale = np.ones((n_fit, 2), dtype=np.float64)
    scaled_projection = np.zeros((n_fit, 2, 2), dtype=np.float64)
    fallback = np.zeros(n_fit, dtype=np.bool_)
    eps_sqrt = np.sqrt(np.finfo(np.float64).eps)
    for i in range(n_fit):
        min_error = np.inf
        n_good = 0
        psf_reference = 0.0
        for j in range(n_pixel):
            good = (np.isfinite(data[i, j]) and np.isfinite(errors[i, j])
                    and errors[i, j] > 0 and np.isfinite(cover[i, j])
                    and cover[i, j] >= min_coverage)
            mask[i, j] = good and not annulus[i, j]
            if mask[i, j]:
                if n_good == 0:
                    psf_reference = psf[i, j]
                n_good += 1
                min_error = min(min_error, errors[i, j])
        if n_good < 2:
            fallback[i] = True
            continue
        sum_weight = 0.0
        centered_mean = 0.0
        for j in range(n_pixel):
            if mask[i, j]:
                weight = (min_error / errors[i, j])**2
                if weight == 0.0:
                    continue
                new_sum = sum_weight + weight
                centered_mean += (weight / new_sum) * ((psf[i, j] - psf_reference) - centered_mean)
                sum_weight = new_sum
        centered_square = 0.0
        for j in range(n_pixel):
            if mask[i, j]:
                weight = (min_error / errors[i, j])**2
                centered_square += weight * ((psf[i, j] - psf_reference) - centered_mean)**2
        psf_mean = psf_reference + centered_mean
        gram00 = centered_square + sum_weight * psf_mean**2
        gram01 = sum_weight * psf_mean
        determinant_sqrt = np.sqrt(sum_weight * centered_square)
        scale_product = np.sqrt(gram00) * np.sqrt(sum_weight)
        if determinant_sqrt > 0 and scale_product > 0:
            # Condition of the column-equilibrated weighted design. The
            # determinant expression avoids cancellation when columns align.
            condition[i] = (scale_product + abs(gram01)) / determinant_sqrt
            column_scale[i, 0] = np.sqrt(gram00) / min_error
            column_scale[i, 1] = np.sqrt(sum_weight) / min_error
        # Keep the existing numerical condition guard. Degenerate designs need
        # the SVD's rank and null-space/annulus identifiability diagnostics.
        if not np.isfinite(condition[i]) or condition[i] >= 1.0 / eps_sqrt:
            fallback[i] = True
            continue
        scaled_projection[i, 0, 0] = 1.0
        scaled_projection[i, 1, 1] = 1.0
        flux = 0.0
        background = 0.0
        for j in range(n_pixel):
            if mask[i, j]:
                weight = (min_error / errors[i, j])**2
                lf = weight * ((psf[i, j] - psf_reference) - centered_mean) / centered_square
                lb = weight / sum_weight - psf_mean * lf
                operator[i, 0, j] = lf
                operator[i, 1, j] = lb
                flux += lf * data[i, j]
                background += lb * data[i, j]
                gain[i, 0] += lf * psf[i, j]
        mean = 0.0
        m2 = 0.0
        for j in range(n_pixel):
            if (annulus[i, j] and np.isfinite(data[i, j])
                    and np.isfinite(errors[i, j]) and errors[i, j] > 0
                    and np.isfinite(cover[i, j]) and cover[i, j] >= min_coverage):
                residual = data[i, j] - (flux * psf[i, j] + background)
                count[i] += 1
                delta = residual - mean
                mean += delta / count[i]
                m2 += delta * (residual - mean)
        if count[i] > 1:
            scatter[i] = np.sqrt(m2 / (count[i] - 1))
        valid[i] = True
        for j in range(n_pixel):
            if mask[i, j]:
                variance = errors[i, j]**2
                lf = operator[i, 0, j]
                lb = operator[i, 1, j]
                covariance[i, 0, 0] += lf * lf * variance
                covariance[i, 0, 1] += lf * lb * variance
                covariance[i, 1, 1] += lb * lb * variance
        covariance[i, 1, 0] = covariance[i, 0, 1]
    return ((operator, covariance, gain, identifiable, group_identifiable,
             valid, mask, scatter, count, rank, condition, annulus_identifiable,
             column_scale, scaled_projection), fallback)


def prepare_fit(image, error, coverage, templates, annulus_mask, *, min_coverage,
                max_work_bytes=512 * 1024**2):
    """Prepare a batch of simultaneous PSF + constant-background fits.

    ``image``, ``error`` (1-sigma per pixel), and ``coverage`` have shape
    (n_fit, ny, nx); ``templates`` has shape (n_fit, n_member, ny, nx).
    Each template is the pixel response to one unit of TOTAL source flux.
    ``annulus_mask`` is either (ny, nx) or (n_fit, ny, nx). The science
    columns use every valid pixel outside the annulus, weighted by inverse
    error-layer variance only. Covariance is L diag(error**2) L.T. Annulus
    residual scatter is diagnostic only, with no effect on validity, weights
    or covariance. Use measure_sources with a frozen cell scatter for final
    source uncertainties; this function supplies only their pixel-noise term.
    The diagonal error layer does not encode pixel correlations. No clipping,
    positivity constraints, or recentering is performed.

    Columns of the weighted design are equilibrated before the SVD; operators
    and covariance are returned in the original physical units. ``condition``
    describes that equilibrated design. Rank-deficient designs use its
    Moore-Penrose solution; only identifiable physical functionals are unique.
    Unidentifiable coefficients are explicitly flagged and have NaN sigma.
    Numerical identifiability is distinct from a scientific blend-correlation
    limit. This routine does not authorize an origin code or an upper bound.
    Batches are the caller's responsibility; no group is split or truncated.
    A conservative working-array estimate is checked before allocation. The
    512 MiB default leaves room below the 1 GiB worker target for a tile and
    interpreter; this estimate does not include caller-held arrays or BLAS.
    """
    image = np.asarray(image)
    error = np.asarray(error)
    coverage = np.asarray(coverage)
    templates = np.asarray(templates)
    if image.ndim != 3 or error.shape != image.shape or coverage.shape != image.shape:
        raise ValueError("image, error and coverage must have matching (n_fit, ny, nx) shapes")
    n_fit, ny, nx = image.shape
    if (templates.ndim != 4 or templates.shape[0] != n_fit
            or templates.shape[2:] != (ny, nx) or templates.shape[1] < 1):
        raise ValueError("templates must have shape (n_fit, n_member >= 1, ny, nx)")
    if n_fit < 1 or not np.isfinite(min_coverage) or min_coverage < 0:
        raise ValueError("require a nonempty batch and finite nonnegative min_coverage")
    n_member = templates.shape[1]
    estimated = 8 * n_fit * (12 * ny * nx * (n_member + 1) + 8 * (n_member + 1)**2)
    if max_work_bytes <= 0 or estimated > max_work_bytes:
        raise MemoryError("fit batch working-array estimate %d bytes exceeds %d; reduce batch size, "
                          "do not split a physical group" % (estimated, max_work_bytes))
    image = np.asarray(image, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    templates = np.asarray(templates, dtype=np.float64)
    annulus = np.asarray(annulus_mask, dtype=bool)
    if annulus.shape not in ((ny, nx), image.shape):
        raise ValueError("annulus_mask must match image or its last two dimensions")
    annulus = np.broadcast_to(annulus, image.shape).reshape(n_fit, -1)
    data = image.reshape(n_fit, -1)
    errors = error.reshape(n_fit, -1)
    cover = coverage.reshape(n_fit, -1)
    psf = templates.reshape(n_fit, n_member, -1).transpose(0, 2, 1)
    if not np.all(np.isfinite(psf)):
        raise ValueError("PSF template values must be finite, including outside the fit mask")
    if n_member != 1:
        return _prepare_general(data, errors, cover, psf, annulus, min_coverage, image.shape)
    values, fallback = _prepare_singleton(data, errors, cover, psf[:, :, 0], annulus, min_coverage)
    if np.any(fallback):
        # Degenerate or numerically delicate designs use exactly the same SVD
        # implementation as larger groups. This never changes group membership.
        rows = np.flatnonzero(fallback)
        reference = _prepare_general(data[rows], errors[rows], cover[rows], psf[rows],
                                     annulus[rows], min_coverage, (len(rows), ny, nx))
        for array, name in zip(values, list(FitContext.__dataclass_fields__)[1:]):
            array[rows] = getattr(reference, name)
    return FitContext(image.shape, *(_readonly(a) for a in values))


def _prepare_general(data, errors, cover, psf, annulus, min_coverage, shape):
    """General simultaneous SVD path, also the private numerical reference."""
    n_fit = data.shape[0]
    n_member = psf.shape[2]
    mask = (~annulus & np.isfinite(data) & np.isfinite(errors) & (errors > 0)
            & np.isfinite(cover) & (cover >= min_coverage))
    valid = mask.sum(axis=1) >= n_member + 1
    sqrt_weight = np.zeros_like(data)
    usable = mask & valid[:, None]
    np.divide(1.0, errors, out=sqrt_weight, where=usable)
    design = np.concatenate((psf, np.ones((*psf.shape[:2], 1))), axis=2)
    whitened = design * sqrt_weight[:, :, None]
    column_scale = np.linalg.norm(whitened, axis=1)
    column_scale = np.where(column_scale > 0, column_scale, 1.0)
    equilibrated = whitened / column_scale[:, None, :]
    u, singular, vt = np.linalg.svd(equilibrated, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(whitened.shape[1:])
    keep = singular > tolerance * singular[:, :1]
    inverse = np.zeros_like(singular)
    np.divide(1.0, singular, out=inverse, where=keep)
    v = vt.transpose(0, 2, 1)
    operator = (((v * inverse[:, None, :]) @ u.transpose(0, 2, 1))
                / column_scale[:, :, None]) * sqrt_weight[:, None, :]
    projection = (v * keep[:, None, :]) @ vt
    # Projection onto the identifiable coefficient subspace. sqrt(eps) is a
    # numerical tolerance only, not the unresolved scientific blend criterion.
    identity = np.eye(n_member + 1)
    numerical_tolerance = np.sqrt(np.finfo(float).eps)
    identifiable = np.linalg.norm(projection - identity, axis=1) < numerical_tolerance
    group_vector = np.r_[np.ones(n_member), 0.0][None, :] / column_scale
    group_identifiable = np.linalg.norm(np.einsum('bi,bij->bj',group_vector,projection)-group_vector, axis=1) < (
        numerical_tolerance * np.linalg.norm(group_vector, axis=1))
    # A singular fit may still predict the annulus uniquely (coincident PSFs
    # are the important case). Reject extrapolation along an unidentified
    # coefficient direction instead of treating its arbitrary value as noise.
    annulus_valid = (annulus & np.isfinite(data) & np.isfinite(errors) & (errors > 0)
                     & np.isfinite(cover) & (cover >= min_coverage))
    annulus_design = np.where(annulus_valid[:, :, None], design / column_scale[:, None, :], 0.0)
    annulus_identifiable = np.linalg.norm(annulus_design @ (identity - projection), axis=(1, 2)) <= (
        numerical_tolerance * np.linalg.norm(annulus_design, axis=(1, 2)))
    smallest = np.min(np.where(keep, singular, np.inf), axis=1)
    condition = np.full(n_fit, np.inf)
    np.divide(singular[:, 0], smallest, out=condition, where=np.isfinite(smallest))
    # Numerical safeguard against catastrophic cancellation, not a scientific
    # separability threshold: retained condition * eps must stay below sqrt(eps).
    # Report this condition and reject unstable fits, never silently split them.
    valid &= (condition < 1.0 / numerical_tolerance)
    clean = np.where(mask, data, 0.0)
    coeff = np.einsum("bkp,bp->bk", operator, clean)
    model = np.einsum("bpk,bk->bp", design, coeff)
    residual = data - model
    scatter, count = _annulus_statistics(residual, errors, cover, annulus, min_coverage)
    variance = np.where(mask & valid[:, None], errors**2, 0.0)
    covariance = (operator * variance[:, None, :]) @ operator.transpose(0, 2, 1)
    gain = np.diagonal(operator[:, :n_member] @ psf, axis1=1, axis2=2).copy()
    values = (operator, covariance, gain, identifiable[:, :n_member].copy(),
              group_identifiable, valid, mask, scatter, count, keep.sum(axis=1),
              condition, annulus_identifiable, column_scale, projection)
    return FitContext(shape, *(_readonly(a) for a in values))


def measure(image, context):
    """Apply the same frozen linear operator to original or injected cutouts.

    Only pixels excluded during preparation may change finite/missing status.
    The returned ``valid`` concerns cutout fit support, not center coverage;
    callers must separately apply the specified coverage test at each target.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.shape != context.shape:
        raise ValueError("image shape differs from prepared context")
    flat = image.reshape(image.shape[0], -1)
    if np.any(context.fit_mask & ~np.isfinite(flat)):
        raise ValueError("a previously usable fit pixel became nonfinite")
    clean = np.where(context.fit_mask, flat, 0.0)
    coeff = np.einsum("bkp,bp->bk", context.operator, clean)
    covariance = context.covariance[:, :-1, :-1]
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    sigma = np.sqrt(np.maximum(diagonal, 0.0))
    denominator = sigma[:, :, None] * sigma[:, None, :]
    correlation = np.full_like(covariance, np.nan)
    np.divide(covariance, denominator, out=correlation, where=denominator > 0)
    sigma = np.where(context.identifiable & context.valid[:, None], sigma, np.nan)
    flux = np.where(context.valid[:, None], coeff[:, :-1], np.nan)
    group_total = flux.sum(axis=1)
    group_sigma = np.sqrt(np.maximum(covariance.sum(axis=(1, 2)), 0.0))
    group_sigma = np.where(context.group_identifiable & context.valid, group_sigma, np.nan)
    background = np.where(context.valid, coeff[:, -1], np.nan)
    return Measurements(flux, sigma, context.gain, covariance, correlation,
                        context.identifiable, group_total, group_sigma,
                        context.group_identifiable, background, context.valid)


@dataclass(frozen=True)
class SourceMeasurements:
    flux: np.ndarray
    pixel_covariance: np.ndarray
    covariance: np.ndarray
    pixel_sigma: np.ndarray
    cell_scatter: np.ndarray
    sigma: np.ndarray
    identifiable: np.ndarray
    valid: np.ndarray
    group_total: np.ndarray
    group_pixel_sigma: np.ndarray
    group_sigma: np.ndarray
    group_identifiable: np.ndarray
    condition: np.ndarray


def measure_sources(image, context, templates, component_source, *, cell_scatter_mjy,
                    cell_covariance_mjy2=None):
    """Sum two scan columns per physical source, preserving full covariance.

    component_source maps every template column onto contiguous source IDs.
    cell_scatter_mjy is mandatory, shape (n_fit,n_source) or broadcastable;
    calibrate with the SAME extracted-flux functional on a declared empty-sky
    pool. For a multi-source group's upper-bound sigma supply the cell covariance
    of these same extracted-source fluxes: sky correlations are not invented.
    A singleton's group sigma equals its source sigma. Negative scan amplitudes
    remain allowed; they are nuisance coefficients, not physical scan weights.
    """
    templates = np.asarray(templates,dtype=float)
    assignment = np.asarray(component_source)
    if (templates.ndim != 4 or templates.shape[0] != image.shape[0] or
            templates.shape[2:] != image.shape[1:] or assignment.shape != (templates.shape[1],) or
            not np.issubdtype(assignment.dtype,np.integer) or np.any(assignment < 0)):
        raise ValueError('Invalid component-to-source mapping or template geometry')
    nsource = int(assignment.max())+1
    if not np.all(np.bincount(assignment,minlength=nsource)==2):
        raise ValueError('Exactly two scan components are required per physical source')
    transform = (np.arange(nsource)[:,None]==assignment[None,:]).astype(float)
    raw = measure(image,context)
    flux = raw.flux @ transform.T
    covariance = np.einsum('si,bij,tj->bst',transform,raw.covariance,transform)
    # A physical sum may be identifiable even when its two components are not.
    functional = np.pad(transform,((0,0),(0,1)))
    scaled_functional = functional[None,:,:] / context.column_scale[:,None,:]
    difference = np.einsum('bsk,bkj->bsj',scaled_functional,context.scaled_projection)-scaled_functional
    identifiable = np.linalg.norm(difference,axis=2) < np.sqrt(np.finfo(float).eps)*np.linalg.norm(scaled_functional,axis=2)
    scatter = np.broadcast_to(np.asarray(cell_scatter_mjy,float),flux.shape)
    if np.any(~np.isfinite(scatter)) or np.any(scatter < 0):
        raise ValueError('A finite nonnegative frozen cell scatter is required')
    pixel_sigma = np.sqrt(np.maximum(np.diagonal(covariance,axis1=1,axis2=2),0))
    usable = identifiable & context.valid[:,None]
    sigma = np.where(usable,np.hypot(pixel_sigma,scatter),np.nan)
    pixel_sigma = np.where(usable,pixel_sigma,np.nan)
    if cell_covariance_mjy2 is None and nsource == 1:
        sky_covariance = scatter[:,:,None]**2
    elif cell_covariance_mjy2 is None:
        sky_covariance = np.full_like(covariance,np.nan)
        index = np.arange(nsource)
        sky_covariance[:,index,index] = scatter**2
    else:
        sky_covariance = np.broadcast_to(np.asarray(cell_covariance_mjy2,float),covariance.shape)
        if (np.any(~np.isfinite(sky_covariance)) or
                not np.allclose(sky_covariance,sky_covariance.transpose(0,2,1)) or
                not np.allclose(np.diagonal(sky_covariance,axis1=1,axis2=2),scatter**2) or
                np.any(np.linalg.eigvalsh(sky_covariance) < -1e-10*np.maximum(1,scatter.max()**2))):
            raise ValueError('Cell covariance must be symmetric PSD with diagonal scatter squared')
    total_covariance = covariance+sky_covariance
    group_sigma = np.where(raw.group_identifiable & raw.valid,
                           np.sqrt(np.maximum(total_covariance.sum(axis=(1,2)),0)),np.nan)
    return SourceMeasurements(flux,covariance,total_covariance,pixel_sigma,scatter,sigma,identifiable,
                              context.valid,raw.group_total,raw.group_sigma,group_sigma,
                              raw.group_identifiable,context.condition)


def verify():
    """Print mathematical checks using synthetic Gaussians, NOT PACS calibration."""
    from time import perf_counter
    start = perf_counter()
    rng = np.random.default_rng(8423)
    y, x = np.mgrid[:31, :31]
    positions = np.array([[[14.2, 15.1], [17.7, 14.6]],
                          [[15.1, 15.1], [15.1, 15.1]]])
    dx = x[None, None] - positions[:, :, 0, None, None]
    dy = y[None, None] - positions[:, :, 1, None, None]
    templates = np.exp(-(dx**2 + dy**2) / (2 * 2.5**2))
    templates /= templates.sum(axis=(2, 3), keepdims=True)
    error = np.broadcast_to(0.08 + 0.002 * x + 0.001 * y, (2, 31, 31)).copy()
    image = rng.normal(size=error.shape) * error + 2.0 + 0.01 * x
    coverage = np.ones_like(image)
    # Missing science and coverage pixels exercise a geometry that is fixed
    # across injections, including heterogeneous error-layer weights.
    image[:, 14, 14] = np.nan
    coverage[:, 16, 17] = 0
    annulus = (x - 15)**2 + (y - 15)**2 >= 8**2
    max_flux_error = 0.0
    max_scatter_error = 0.0
    max_sigma_relative = 0.0
    for n_member in (1, 2):
        psf = templates[:, :n_member]
        context = prepare_fit(image, error, coverage, psf, annulus, min_coverage=1)
        before = measure(image, context)
        assert np.all(context.valid)
        assert np.all(context.group_identifiable)
        if n_member == 2:
            assert np.all(context.identifiable[0]) and not np.any(context.identifiable[1])
        for injected_flux in (0.1, 500.0, 100000.0):
            injected = image + injected_flux * psf[:, 0]
            after = measure(injected, context)
            rebuilt = prepare_fit(injected, error, coverage, psf, annulus, min_coverage=1)
            repeated = measure(injected, rebuilt)
            expected = injected_flux * context.gain[:, 0]
            discrepancy = np.max(np.abs(after.flux[:, 0] - before.flux[:, 0] - expected))
            max_flux_error = max(max_flux_error, discrepancy)
            assert np.all(rebuilt.valid)
            assert np.allclose(repeated.flux, after.flux, rtol=1e-12, atol=1e-9)
            assert np.allclose(repeated.sigma, before.sigma, rtol=1e-9, atol=1e-10, equal_nan=True)
            assert np.allclose(repeated.group_sigma, before.group_sigma, rtol=1e-9)
            assert np.allclose(after.group_total - before.group_total, injected_flux, rtol=1e-10, atol=1e-8)
            max_scatter_error = max(max_scatter_error,
                                    np.max(np.abs(rebuilt.annulus_scatter - context.annulus_scatter)))
            good = np.isfinite(before.sigma)
            max_sigma_relative = max(max_sigma_relative,
                                     np.max(np.abs(repeated.sigma[good] / before.sigma[good] - 1)))
        # Independent sandwich covariance identity, including excluded pixels.
        pixel_variance = np.where(context.fit_mask,
                                  error.reshape(2, -1)**2, 0)
        covariance = np.einsum("bip,bp,bjp->bij", context.operator, pixel_variance, context.operator)
        assert np.allclose(covariance, context.covariance, rtol=1e-12, atol=1e-12)
    assert max_flux_error < 1e-8
    invalid = prepare_fit(image, error, np.zeros_like(coverage), templates, annulus, min_coverage=1)
    assert not np.any(measure(image, invalid).valid)
    print("Synthetic mathematical checks: frozen/rebuilt flux max absolute error", max_flux_error)
    print("Full rebuild: annulus scatter max absolute change", max_scatter_error)
    print("Full rebuild: sigma max relative change", max_sigma_relative)
    print("Singleton, separable/coincident blends, varying errors, masks and covariance checks passed")
    print("Extractor verification wall seconds", perf_counter() - start)
    verify_singleton()
    benchmark_singleton()
    return max_flux_error


def verify_singleton():
    """Compare optimized singletons with the unchanged general SVD estimator."""
    rng = np.random.default_rng(10031)
    y, x = np.mgrid[:31, :31]
    base = np.exp(-((x - 14.31)**2 + (y - 15.23)**2) / 12.5)
    base /= base.sum()
    scales = np.array([1e-12, 1e-6, 1.0, 1e6, 1e12])
    profiles = [base * scale for scale in scales]
    profiles.extend([np.ones_like(base), np.zeros_like(base),
                     1 + 1e-3 * base, 1 + 1e-6 * base, 1 + 1e-12 * base])
    templates = np.asarray(profiles)[:, None]
    n = len(profiles)
    image = rng.normal(size=(n, 31, 31)) * 0.2 + 3 + x * 0.01
    error = np.exp(rng.normal(size=image.shape)) * 0.1
    coverage = np.ones_like(image)
    image[:, 12, 11] = np.nan
    coverage[:, 13, 17] = 0
    annulus = (x - 15)**2 + (y - 15)**2 > 9**2
    context = prepare_fit(image, error, coverage, templates, annulus, min_coverage=1)
    reference = _prepare_general(image.reshape(n, -1), error.reshape(n, -1),
                                 coverage.reshape(n, -1), templates[:, 0].reshape(n, -1, 1),
                                 np.broadcast_to(annulus, image.shape).reshape(n, -1),
                                 1, image.shape)
    for name in ("valid", "rank", "identifiable", "group_identifiable", "annulus_identifiable", "fit_mask"):
        assert np.array_equal(getattr(context, name), getattr(reference, name)), name
    for name in ("operator", "covariance", "gain", "annulus_scatter", "condition"):
        assert np.allclose(getattr(context, name), getattr(reference, name),
                           rtol=2e-7, atol=1e-10, equal_nan=True), name
    before = measure(image, context)
    after_image = image + templates[:, 0] * 10
    after_context = prepare_fit(after_image, error, coverage, templates, annulus, min_coverage=1)
    after = measure(after_image, after_context)
    # Extremely rescaled or nearly constant columns are numerical-status
    # probes; use the well-conditioned profiles for an injection identity.
    good = (context.valid & context.identifiable[:, 0]
            & (context.condition < 1e5))
    assert np.allclose(after.sigma[good], before.sigma[good], rtol=1e-8, atol=1e-10)
    # Equilibrated conditioning intentionally no longer rejects a tiny PSF
    # solely because of its units. Its enormous fitted coefficient makes a
    # subtraction of before/after fluxes precision-limited in those same units.
    cancellation = 64*np.finfo(float).eps*(np.abs(after.flux[good,0])+np.abs(before.flux[good,0]))
    assert np.allclose(after.flux[good, 0] - before.flux[good, 0],
                       10 * context.gain[good, 0], rtol=1e-8, atol=1e-8+cancellation)
    print("Singleton/SVD scaled, heteroscedastic, masked and degenerate comparisons passed")


def benchmark_singleton():
    """Print a warmed core comparison; this is not a regional performance claim."""
    from time import perf_counter
    y, x = np.mgrid[:31, :31]
    psf = np.exp(-((x - 14.3)**2 + (y - 15.1)**2) / 12.5)
    psf /= psf.sum()
    n = 500
    image = np.broadcast_to(2 + 0.01 * x + psf, (n, 31, 31)).copy()
    error = np.broadcast_to(0.08 + 0.002 * x, image.shape).copy()
    coverage = np.ones_like(image)
    templates = np.broadcast_to(psf, (n, 1, 31, 31))
    annulus = (x - 15)**2 + (y - 15)**2 > 9**2
    prepare_fit(image[:1], error[:1], coverage[:1], templates[:1], annulus, min_coverage=1)
    optimized = []
    reference = []
    for repeat in range(3):
        start = perf_counter()
        context = prepare_fit(image, error, coverage, templates, annulus, min_coverage=1)
        measure(image, context)
        optimized.append(perf_counter() - start)
        del context
        start = perf_counter()
        context = _prepare_general(image.reshape(n, -1), error.reshape(n, -1),
                                   coverage.reshape(n, -1), templates[:, 0].reshape(n, -1, 1),
                                   np.broadcast_to(annulus, image.shape).reshape(n, -1),
                                   1, image.shape)
        measure(image, context)
        reference.append(perf_counter() - start)
        del context
    fast = float(np.median(optimized))
    general = float(np.median(reference))
    print("Warmed 500 singleton 31x31 prepare+measure seconds: optimized", fast,
          "general SVD", general, "speedup", general / fast)
    return fast, general
