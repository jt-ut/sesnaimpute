"""
model_convolution.py
====================================================================
Shared convolution infrastructure for the sed_models_curate curate modules:
everything that turns a model SED into a band flux, and the single
writer for convolved/{band}.fits. Its counterpart is model_io.py, which
owns flux.fits / parameters.fits / models.conf.

Nothing here is library-specific, and nothing assumes aperture-dependence
-- sedfitter's convolve_model_dir loops over however many apertures a
model set's SEDCube declares, so aperture-independent and
aperture-dependent sets are handled identically.

This is model-directory construction plumbing, which is why it is not in
astro_utils.py (package-wide flux/mag/colour math) and is used only by
the curate modules.

The LINE SPECTRA section below applies to any model set whose SED is a
set of delta functions rather than a sampled continuum. Such a set needs
both a deposition width to be storable on a wavelength grid at all, and
an exact analytic band flux that a continuum does not admit.
====================================================================
"""

import os

import numpy as np
from astropy import units as u
from astropy.io import fits

from sedfitter.filter import Filter
from sedfitter.convolve import convolve_model_dir

from sesnaimpute.sed_models.data_loader import load_filter_curve
from sesnaimpute.sed_models.constants import BANDS, C_UM_S, SPEED_OF_LIGHT_CM_S


def pivot_wavelength_um(wave_um, response):
    """Standard photometric pivot wavelength: sqrt(int R*lam dlam / int R/lam dlam)."""
    return np.sqrt(np.trapz(response * wave_um, wave_um) / np.trapz(response / wave_um, wave_um))


def build_sedfitter_filter(name):
    """
    name -> a ready-to-use, normalized sedfitter Filter object.

    Explicit-normalization wrapper around sedfitter's Filter class:
    normalization (integral over nu = 1) is required by sedfitter's
    convolution but is easy to forget since Filter.read() doesn't do it
    automatically -- so it's always called out loud here, never buried.

    central_wavelength: BANDS[name].wvl_effective_um for the 8 SESNA bands
    (already-documented instrument-handbook values), computed pivot
    wavelength for anything not in BANDS (currently just Gaia G).
    """
    fc = load_filter_curve(name)

    if name in BANDS:
        central_um = BANDS[name].wvl_effective_um
    else:
        central_um = pivot_wavelength_um(fc.wave_um, fc.response)

    # sedfitter requires ascending frequency == descending wavelength
    order = np.argsort(-fc.wave_um)
    wave_sorted = fc.wave_um[order]
    resp_sorted = fc.response[order]

    f = Filter()
    f.name = name
    f.central_wavelength = central_um * u.micron
    f.nu = (C_UM_S / wave_sorted) * u.Hz
    f.response = resp_sorted
    f.normalize()  # explicit -- required by sedfitter's convolution, never implicit
    return f


def build_convolved_band_fits(model_dir, bands):
    """Run sedfitter's own convolve_model_dir against model_dir's assembled
    flux.fits/parameters.fits/models.conf, writing convolved/{band}.fits for
    each name in bands. Aperture-independent and aperture-dependent model
    sets both work unchanged -- convolve_model_dir loops over however many
    apertures the SEDCube declares."""
    filters = [build_sedfitter_filter(name) for name in bands]
    convolve_model_dir(model_dir, filters, overwrite=True)


def build_convolved_bands(model_dir, bands, model_names, convmeth_note=""):
    """Write convolved/{band}.fits, stamp which path produced them, and
    verify the result -- the whole convolution step for a CONTINUUM library.

    Shared by galaxy and sps, which need exactly this and nothing more. It is
    the same build -> stamp -> verify shape h2shock uses, minus its
    conservation gate: that gate exists because a LINE spectrum can only be
    stored on a wavelength grid by smearing it, so sedfitter's
    interpolate-then-integrate convolution is inexact for one and an analytic
    reference is needed to prove it. A sampled continuum is what
    convolve_model_dir is built for, so there is no second path to fall back
    to or check against, and CONVMETH is unconditionally 'sedfitter'.

    Stamping anyway is not bookkeeping for its own sake: every library's
    convolved/ then declares its own provenance in the shipped file, so which
    path produced a given band never has to be re-derived by inspecting the
    code that wrote it.

    verify_convolved_band_fits is the part that earns its runtime. It checks
    each file exists, carries the expected model count, holds finite fluxes,
    and -- the invariant nothing else covers -- that MODEL_NAME appears in the
    SAME ORDER as flux.fits. The output-contract check compares those name
    sets SORTED (C9), which a row-order permutation would pass; sedfitter
    indexes convolved/ rows positionally against the cube, so order is what
    actually has to hold.

    FILTWAV comes from build_sedfitter_filter, i.e. BANDS[b].wvl_effective_um
    for the survey bands. That is load-bearing downstream, not cosmetic:
    FILTWAV propagates to models.wavelengths and thence to av_law, so it sets
    each band's EXTINCTION coefficient. It must therefore be identical for a
    given band across all five libraries -- see filter_central_wavelength_um.
    """
    build_convolved_band_fits(model_dir, bands)
    stamp_convmeth_header(model_dir, bands, "sedfitter", convmeth_note)
    return verify_convolved_band_fits(model_dir, bands, model_names)


# ====================================================================
# LINE SPECTRA
# ====================================================================
#
# A line-emission model set is a set of delta functions, not a sampled
# continuum, which changes two things:
#
#   * Depositing it onto a wavelength grid needs an explicit, chosen
#     profile width -- a NUMERICAL representation width, not a physical
#     one (see LINE_DEPOSITION_WIDTH_KMS).
#   * Its band flux can be computed EXACTLY, by evaluating the filter
#     response at each line's own wavelength, with no interpolation of
#     spikes. That exactness is what makes the analytic path a valid
#     reference to check sedfitter's grid-based convolution against --
#     a check continuum libraries never needed.

# Numerical representation width for depositing a line onto a wavelength
# grid: broad enough that both a model grid and a filter-curve grid
# resolve each line, yet <=1.5% of the ~1um filter widths so band fluxes
# are unaffected. Not physical, and nothing downstream should read it as
# a linewidth. The analytic band flux below does not depend on it at all.
LINE_DEPOSITION_WIDTH_KMS = 3000.0

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def deposit_line_fluxes_onto_wavelength_grid(lines, line_fluxes, wave_um,
                                             line_width_kms=LINE_DEPOSITION_WIDTH_KMS):
    """Deposit each line as a Gaussian in ln(wavelength), area-normalized
    in frequency so its integral equals the line's energy flux (i.e. an
    F_nu profile with integral(F_nu dnu) = line_fluxes[i]). wave_um must
    be ascending. `lines` needs only a `.wave_um` attribute.

    This is the readable per-line reference form. For production use
    build_line_deposition_matrix, which is the same computation as a
    single matmul; this loop is what that form is checked against."""
    freq_hz = C_UM_S / wave_um
    ln_wave = np.log(wave_um)
    d_ln_wave = np.abs(np.gradient(ln_wave))

    fractional_width_fwhm = (line_width_kms * 1e5) / SPEED_OF_LIGHT_CM_S  # km/s -> cm/s, over c in cm/s
    sigma_ln_wave = max(fractional_width_fwhm * FWHM_TO_SIGMA, 1.5 * np.median(d_ln_wave))

    flux_nu = np.zeros_like(wave_um)
    in_range = (lines.wave_um >= wave_um[0]) & (lines.wave_um <= wave_um[-1]) & (line_fluxes > 0)
    for wave_line_um, flux_line in zip(lines.wave_um[in_range], line_fluxes[in_range]):
        profile = np.exp(-0.5 * ((ln_wave - np.log(wave_line_um)) / sigma_ln_wave) ** 2)
        profile = profile / np.trapz(profile, freq_hz)
        flux_nu += flux_line * np.abs(profile)
    return flux_nu


def build_line_deposition_matrix(lines, wave_um, line_width_kms=LINE_DEPOSITION_WIDTH_KMS):
    """Vectorized form of deposit_line_fluxes_onto_wavelength_grid's
    per-line Python loop: builds the (n_lines_in_range, n_wave) Gaussian
    profile matrix P once, so depositing any model's line fluxes becomes a
    single `line_fluxes[in_range] @ P` matmul instead of looping per line,
    per model. wave_um must be ascending, matching
    deposit_line_fluxes_onto_wavelength_grid. Returns (in_range, P):
    in_range is a boolean mask into lines' arrays, wavelength-coverage
    only -- deliberately NOT gated on line_fluxes > 0 like the loop
    version's mask, since a model's zero-flux line contributes exactly
    0.0 * (finite profile) = 0.0 to the matmul regardless, so dropping
    that flux-dependent half of the gate changes nothing (the profile
    itself never depends on flux, only on wavelength/width, so there is
    no NaN/Inf risk from including a zero-flux line's finite row here).
    P's rows already carry the same normalize-then-abs (not
    abs-then-normalize) order as the loop version, to stay numerically
    identical."""
    freq_hz = C_UM_S / wave_um
    ln_wave = np.log(wave_um)
    d_ln_wave = np.abs(np.gradient(ln_wave))

    fractional_width_fwhm = (line_width_kms * 1e5) / SPEED_OF_LIGHT_CM_S
    sigma_ln_wave = max(fractional_width_fwhm * FWHM_TO_SIGMA, 1.5 * np.median(d_ln_wave))

    in_range = (lines.wave_um >= wave_um[0]) & (lines.wave_um <= wave_um[-1])
    ln_wave_line = np.log(lines.wave_um[in_range])

    # (n_lines_in_range, n_wave), same math as the loop body, broadcast over lines
    delta = ln_wave[None, :] - ln_wave_line[:, None]
    profile = np.exp(-0.5 * (delta / sigma_ln_wave) ** 2)
    denom = np.trapz(profile, freq_hz, axis=1)
    P = np.abs(profile / denom[:, None])

    return in_range, P


def analytic_band_flux_for_line_spectrum(lines, line_fluxes, filter_wave_um, filter_response,
                                         convention="flat"):
    """Exact convolved band flux for a line spectrum -- no interpolation-
    of-spikes risk, since each line's filter response is evaluated
    directly at its own wavelength. convention="flat" matches how
    sedfitter's own Filter.normalize()/convolve_model_dir actually
    convolve, and is what the conservation check uses;
    convention="nuFnu" (the IRAC isophotal Fnu*nu=const convention) is
    provided only because reference generators default to it, and would
    silently disagree with what convolved/{band}.fits actually
    contains."""
    order = np.argsort(filter_wave_um)
    filt_wave = filter_wave_um[order]
    filt_resp = filter_response[order]
    filt_freq = C_UM_S / filt_wave  # descending, since filt_wave ascending

    response_at_line = np.interp(lines.wave_um, filt_wave, filt_resp, left=0.0, right=0.0)
    numerator = np.sum(line_fluxes * response_at_line)

    if convention == "flat":
        denominator = abs(np.trapz(filt_resp, filt_freq))
    elif convention == "nuFnu":
        freq0 = np.trapz(filt_resp * filt_freq, filt_freq) / np.trapz(filt_resp, filt_freq)
        denominator = abs(np.trapz(filt_resp * (freq0 / filt_freq), filt_freq))
    else:
        raise ValueError(f"unknown convention {convention!r}")

    return numerator / denominator


def build_band_response_weights(bands, lines):
    """Precompute, per band, the (response_at_line, denominator) pair that
    analytic_band_flux_for_line_spectrum's convention="flat" path derives
    from (band, lines.wave_um) alone, never from line_fluxes. Computing
    this once here -- rather than once per (model, band) inside a nested
    loop -- turns per-model band flux into
    `np.sum(line_fluxes * response_at_line, axis=1) / denominator`.

    Returned as a (response_at_line, denominator) pair rather than a
    single pre-divided weight vector, so callers sum over lines and then
    divide. That order matters: floating-point division does not
    distribute exactly over a sum, so pre-dividing would disagree with
    analytic_band_flux_for_line_spectrum at the ~1e-16 level. Matmul has
    the same issue one level up, since BLAS reorders the summation, so
    prefer an explicit np.sum here. Returns {band: (response_at_line,
    denominator)}."""
    weights = {}
    for band in bands:
        fc = load_filter_curve(band)
        order = np.argsort(fc.wave_um)
        filt_wave = fc.wave_um[order]
        filt_resp = fc.response[order]
        filt_freq = C_UM_S / filt_wave
        response_at_line = np.interp(lines.wave_um, filt_wave, filt_resp, left=0.0, right=0.0)
        denominator = abs(np.trapz(filt_resp, filt_freq))
        weights[band] = (response_at_line, denominator)
    return weights


def write_convolved_band_fits(model_dir, band, model_names, total_flux_mjy, apertures_au,
                              filter_wavelength_um, total_flux_err_mjy=None,
                              convmeth=None, convmeth_note="", convmeth_error=None,
                              model_name_format="30A"):
    """Write one convolved/{band}.fits in the project's model-directory
    convention -- THE single writer for that file.

    Two things produce convolved fluxes in this project: sedfitter's own
    convolve_model_dir (via build_convolved_band_fits, the primary path
    for every library) and the exact analytic path for line spectra
    (write_convolved_band_fits_from_analytic). Both must emit a
    structurally identical file. Route any new producer through here
    rather than building its own HDUList.

    The layout is taken from the shipped libraries, not from a spec:
      PRIMARY            FILTWAV, NMODELS, NAP  (+ CONVMETH/CONVERR)
      CONVOLVED FLUXES   MODEL_NAME (30A), TOTAL_FLUX (nD, dim '(n)',
                         mJy, n=len(apertures_au)), TOTAL_FLUX_ERR (nD,
                         dim '(n)', mJy)
      APERTURES          APERTURE (D, AU)

    `n` was hardcoded to 1 in the column FORMAT (always "D") regardless
    of `apertures_au`'s real length until this fix -- every caller
    before YSO was a point-source/`aperture_dependent=no` library with
    n_ap=1, where "D" (repeat count 1) happens to equal n_ap, so the
    bug was invisible. YSO's real 20-aperture curve of growth is the
    first n_ap>1 caller, and astropy's own `Column` validator catches
    the mismatch (`VerifyError`, repeat count 1 against TDIM '(20)')
    rather than writing a silently wrong file -- fixed at the source,
    not special-cased for YSO.

    TOTAL_FLUX_ERR defaults to exact zeros. That matches an analytic
    library honestly (no stochastic noise) and is accepted by sedfitter --
    Robitaille's own YSO flux.fits carries millions of exact zeros. It is
    also never read by the fit: chi_squared() takes its errors from the
    source, not the model."""
    total_flux_mjy = np.atleast_1d(np.asarray(total_flux_mjy, dtype=np.float64))
    if total_flux_err_mjy is None:
        total_flux_err_mjy = np.zeros_like(total_flux_mjy)
    apertures_au = np.atleast_1d(np.asarray(apertures_au, dtype=np.float64))
    n_models, n_ap = len(model_names), apertures_au.size

    primary = fits.PrimaryHDU()
    primary.header["FILTWAV"] = (float(filter_wavelength_um), "filter central wavelength, um")
    primary.header["NMODELS"] = n_models
    primary.header["NAP"] = n_ap
    if convmeth is not None:
        primary.header["CONVMETH"] = (convmeth, convmeth_note[:47])
        if convmeth_error is not None:
            primary.header["CONVERR"] = (float(convmeth_error),
                                         "max rel. error vs analytic reference")

    fluxes = fits.BinTableHDU.from_columns([
        fits.Column(name="MODEL_NAME", format=model_name_format, array=np.asarray(model_names)),
        fits.Column(name="TOTAL_FLUX", format=f"{n_ap}D", dim=f"({n_ap})", unit="mJy",
                    array=total_flux_mjy.reshape(n_models, n_ap)),
        fits.Column(name="TOTAL_FLUX_ERR", format=f"{n_ap}D", dim=f"({n_ap})", unit="mJy",
                    array=np.asarray(total_flux_err_mjy, dtype=np.float64).reshape(n_models, n_ap)),
    ], name="CONVOLVED FLUXES")

    apertures = fits.BinTableHDU.from_columns([
        fits.Column(name="APERTURE", format="D", unit="AU", array=apertures_au),
    ], name="APERTURES")

    out_dir = os.path.join(model_dir, "convolved")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{band}.fits")
    fits.HDUList([primary, fluxes, apertures]).writeto(out_path, overwrite=True)
    return out_path


def analytic_band_fluxes_for_all_models(lines, line_fluxes_by_model, bands):
    """Exact band flux for every model of a line-spectrum library:
    {band: (n_models,) array}.

    Generic over libraries -- it takes the per-model line fluxes directly
    ((n_models, n_lines), in whatever flux unit the caller intends the
    output to carry) rather than a library-specific parameter grid plus a
    physics callback. Uses build_band_response_weights so the per-model
    reduction is one vectorised sum, and keeps that function's
    sum-then-divide order so results match
    analytic_band_flux_for_line_spectrum bit-for-bit."""
    weights = build_band_response_weights(bands, lines)
    line_fluxes_by_model = np.atleast_2d(line_fluxes_by_model)
    out = {}
    for band in bands:
        response_at_line, denominator = weights[band]
        out[band] = np.sum(line_fluxes_by_model * response_at_line, axis=1) / denominator
    return out


def write_convolved_band_fits_from_analytic(model_dir, lines, line_fluxes_by_model, bands,
                                            model_names, apertures_au,
                                            sedfitter_max_rel_error=None,
                                            model_name_format="30A"):
    """Write convolved/{band}.fits for every band directly from the exact
    analytic line-spectrum formula, bypassing flux.fits entirely.

    Applied uniformly across all `bands`, including any that individually
    passed the conservation gate, so a library ships one consistent method
    rather than a per-band patchwork. CONVMETH records that choice in each
    file, and CONVERR carries the sedfitter path's measured error where
    known, so which path shipped never has to be re-derived by
    inspection."""
    reference = analytic_band_fluxes_for_all_models(lines, line_fluxes_by_model, bands)
    written = {}
    for band in bands:
        err = None if sedfitter_max_rel_error is None else sedfitter_max_rel_error.get(band)
        written[band] = write_convolved_band_fits(
            model_dir, band,
            model_names=model_names,
            total_flux_mjy=reference[band],
            apertures_au=apertures_au,
            filter_wavelength_um=filter_central_wavelength_um(band),
            convmeth="analytic",
            convmeth_note="exact line-spectrum convolution",
            convmeth_error=err,
            model_name_format=model_name_format,
        )
    return written


def verify_convolved_band_fits(model_dir, bands, model_names):
    """Read convolved/{band}.fits back off disk and confirm each is
    structurally sound and consistent with the model list it claims to
    describe. Raises AssertionError on the first failure.

    This does not compare flux VALUES against an independent reference --
    for a line spectrum, the shipped fluxes are the analytic value by
    construction, so checking them against that same analytic value would
    be circular and would trivially pass. What is worth checking after any
    write is that the files exist, carry the expected model count and
    order, and hold finite fluxes. Returns {band: n_models}."""
    counts = {}
    for band in bands:
        path = os.path.join(model_dir, "convolved", f"{band}.fits")
        assert os.path.exists(path), f"{path} was not written"
        with fits.open(path) as hdul:
            header, data = hdul[0].header, hdul["CONVOLVED FLUXES"].data
            written = np.array([str(n).strip() for n in data["MODEL_NAME"]])
            assert len(written) == len(model_names), (
                f"{band}: {len(written)} rows written, expected {len(model_names)}")
            assert header["NMODELS"] == len(model_names), (
                f"{band}: NMODELS={header['NMODELS']} disagrees with the table")
            assert np.array_equal(written, np.asarray(model_names)), (
                f"{band}: MODEL_NAME order does not match the model list")
            flux = data["TOTAL_FLUX"]
            assert np.all(np.isfinite(flux)), f"{band}: non-finite TOTAL_FLUX"
            counts[band] = len(written)
    return counts


def filter_central_wavelength_um(name):
    """The wavelength the extinction law is evaluated at for this band, and
    what goes in FILTWAV.

    BANDS[name].wvl_effective_um for the survey bands -- the published
    in-flight-calibrated effective wavelengths. Effective wavelength is
    the right choice rather than pivot: FILTWAV propagates to
    models.wavelengths and thence to av_law, so it sets the band's
    EXTINCTION coefficient, and the band-integrated extinction is the
    source-weighted mean of A_lambda, whose first-order estimator is the
    effective wavelength. (Pivot wavelength answers a different question
    -- where F_lambda and F_nu convert exactly -- and has no relationship
    to extinction weighting.)

    This value must be the SAME for a given band across all five
    libraries: the classes are compared against each other, so a band
    whose A_lambda differed by class would bias the comparison. That is
    why it is fixed here rather than computed per library, even though
    "effective wavelength" is genuinely source-dependent. Anything not in
    BANDS falls back to the computed pivot."""
    if name in BANDS:
        return BANDS[name].wvl_effective_um
    fc = load_filter_curve(name)
    return pivot_wavelength_um(fc.wave_um, fc.response)


def stamp_convmeth_header(model_dir, bands, method, note="", max_rel_error=None):
    """Record which convolution path actually produced
    convolved/{band}.fits.

    A library that validates the pipeline convolution and falls back to
    the analytic path on failure otherwise leaves no trace in the shipped
    output of which path won. A writer that produces the file itself can
    set CONVMETH directly; this helper covers the other branch --
    stamping the header onto files already written by build_convolved_band_fits.
    Opens each file in update mode rather than rewriting it, so only the
    header changes. max_rel_error, if given (a {band: rel_error} dict), is
    recorded per band as CONVERR."""
    for band in bands:
        path = os.path.join(model_dir, "convolved", f"{band}.fits")
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["CONVMETH"] = (method, note)
            if max_rel_error is not None and band in max_rel_error:
                hdul[0].header["CONVERR"] = (max_rel_error[band],
                                             "max relative error vs the analytic reference, this band")
            hdul.flush()
