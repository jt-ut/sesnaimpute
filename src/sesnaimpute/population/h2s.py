"""H2S: shocked H2 knots (SPEC_PRIORS.md section 7).

A knot is where a young star's jet strikes the cloud; it exists whether or
not its driver is catalogued, so the count rides on the YSO **law** count
(`population.yso.law_count`), never on the selected YSO count:

    N_H2S(s) = (N_law (x) K)(s) . eta_r . eps_ext . eps_s(a, Sigma)

Three pieces, two of them built here per source or per region, the third
(the extinction shape `p(a | A_s)`) shared unchanged with YSO -- it is not
stored again (IMPLEMENTATION.md section 3, H2S row).

1. `N_law (x) K`: the law count (section 6.1, before any selection),
   spatially blurred by the knot-driver displacement kernel `K(r) =
   exp(-r/lambda) / (2 pi lambda r)`, on a 1-arcmin gnomonic lattice per
   region (`lambda` converted to an angle at the region's own distance).
   Written per source, `bms/h2s/law-blurred_h2s_source`.
2. `eta_r`, `eps_ext`: two amplitude-only scalars, cited below, that never
   vary per source. `eta_r` is knots present per intrinsic law-predicted
   young star (S-D37b); `eps_ext` is the fraction of knots clearing the
   survey's limits that SESNA's own extraction actually rows (S-D44).
3. `eps_s(a, Sigma)`: the selection, per source and exact (SPEC_PRIORS.md
   1.3, 7): at surface brightness Sigma, Ks (`knot_ks_log10_flux`) and
   the dimming are single exact numbers, so only the four IRAC bands
   carry any randomness, each an independent draw from its own tiny
   Giannini colour-ratio table (13-52 measured knots; J, H and M1 carry
   no Giannini ratio and never clear). With that few distinct values,
   the two-of-eight test's own pass probability is worked out in closed
   form (`exact_source_selection`) rather than approximated by drawing
   combinations: each band's own exact clearing fraction against its
   whole table (`band_clear_fraction`, a `searchsorted`, no sampling
   error), combined algebraically over the 2^4 ways the four can clear.
   `EPS[n, n_x, n_sigma]` is an ON-THE-FLY lookup (owner, 2026-09-06,
   the same change `population.gal` made for its own per-source galaxy
   selection: no `bms/h2s/selection_h2s_source` product any more) on
   the shared scaled-extinction ladder `X_LADDER`, at every source's own
   eight limits (`catalog.limits.limits`) and own column -- no depth
   groups, no common-mode shift. `source_selection` is the public entry
   point; `population.counts_cloud.build_region` calls it per source batch.
   The tiny ratio tables carry no shared per-knot index back to
   Giannini's table, so the four IRAC bands are treated as independent
   populations here, disclosed rather than assumed away; pairing them
   by knot, were the index available, would be the better form (it
   preserves the measured colour correlations).

The region product (`bms/h2s/prior_h2s_region`) keeps everything else:
the law-blurred field, `eta_r`, `eps_ext`, and the region's own
brightness lognormal (including the Sigma grid `source_selection` reads
at call time).

No library enters a count or a shape (C3): the h2shock template register
supplies SED templates to the fitter only, never a selection average.
"""

import math
import os

import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import healpy as hp
from scipy.signal import fftconvolve

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.population import selection as selection_module
from sesnaimpute.population import yso as yso_module

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)
KS_IDX = BAND_KEYS.index("Ks")
IRAC_RATIO_BAND_KEYS = ("I1", "I2", "I3", "I4")
IRAC_RATIO_BAND_IDX = np.array([BAND_KEYS.index(b) for b in IRAC_RATIO_BAND_KEYS])

#: Knots present per intrinsic law-predicted young star, per field
#: (SPEC_PRIORS.md section 7 decision 3): the external knot survey's own
#: knot surface density, depth-corrected, over the CURRENT law's own area
#: integral (`prior.yso.law_area_integral`: Pokhrel's kappa, the whole
#: adopted column, no pedestal) on the same footprint -- re-measured
#: 2026-09-06 against the law as it now stands, replacing S-D37b's numbers,
#: which were measured against the retired dust-temperature/kappa_r
#: denominator. Old value beside each new one, and the old/new ratio:
#:   Cygnus X               0.0105 -> 0.0065  (old/new 1.62x)
#:   North America Nebula   0.0572 -> 0.0394  (old/new 1.45x)
#:   Vela D                 0.0668 -> 0.0408  (old/new 1.64x)
#: Measured only where an external H2 knot survey covers the field; every
#: other region is nearer than Cygnus X, so it takes the elsewhere value.
#: The old elsewhere value (0.06) is not the knot-count-weighted mean of
#: the three fields (that would have been 0.0141) nor the added pooled
#: reading (0.0082) -- it matches, to rounding, the plain mean of the two
#: NEARER fields' own eta (North America Nebula, Vela D), excluding
#: Cygnus X, which is the one field "elsewhere" is not nearer than: (0.0572
#: + 0.0668)/2 = 0.0620 -> 0.06. That is the rule applied here too.
ETA = {
    "Cygnus X": 0.0065,
    "North America Nebula": 0.0394,
    "Vela D": 0.0408,
}
ETA_ELSEWHERE = 0.0401

#: The three-field spread of `eta`, dex (SPEC_PRIORS.md section 7 table;
#: S-D37b measures 0.447, restated to two figures). Reported, never
#: marginalised.
ETA_BAND_DEX = 0.45

#: The fraction of knots clearing the survey's limits for which SESNA's
#: own point-source finder produces a catalogue row: the pooled positional
#: cross-match against five H2 knot surveys, excess over two independent
#: chance controls (S-D44, the study record for this number; its own
#: 95% CI is the reported band).
EPS_EXT = 0.25
EPS_EXT_LO, EPS_EXT_HI = 0.21, 0.29

#: The knot-driver radial-offset scale, pc: a maximum-likelihood fit to
#: knot-to-protostar separations in the Orion A (Davis et al. 2009) and
#: Perseus (Walawender et al. 2005) knot catalogues, source-coincident
#: knots excluded (SPEC_PRIORS.md section 7).
LAMBDA_PC = 0.32

#: The offset kernel's own binning pixel, arcmin -- also the resolution
#: floor below which a region's own angular `lambda` does not resolve
#: against the lattice (the honest-skip branch of `blur_law_field`).
KERNEL_PIXEL_ARCMIN = 1.0

#: The kernel lattice's truncation radius, in units of lambda, renormalised
#: to sum to exactly 1.0 after truncation -- the truncated mass is disclosed
#: by construction (`build_offset_kernel`), never separately corrected.
KERNEL_TRUNCATE_LAMBDA = 8.0

#: A convolved-count floor, in raw (unblurred) source-weight units, below
#: which a lattice pixel is masked rather than sampled: below the weight of
#: half a raw source the kernel is reporting a value interpolated across
#: mostly-empty footprint, not a measurement. Sources landing in a masked
#: pixel fall back to their own pointwise law count (`blur_law_field`).
KERNEL_COUNT_FLOOR = 0.5

#: The two UWISH2 fields whose own jet-class ('j') coverage is dense
#: enough to serve as the reference population every region's knot
#: brightness is transported from (SPEC_PRIORS.md section 7: "the 1,349
#: UWISH2 jet-class knots in Cygnus X and the North America Nebula").
UWISH2_REFERENCE_REGIONS = ("Cygnus X", "North America Nebula")

#: SESNA's own fixed photometric aperture radius for the IRAC bands (the
#: Gutermuth-lineage extraction pipeline's survey/instrument convention;
#: Gutermuth et al. 2009, ApJS 184, 18) -- the aperture a shocked knot's
#: flux is collected through, and the one the H2S shock-template library's
#: own beam-filled convention is keyed to.
SESNA_APERTURE_RADIUS_ARCSEC = 2.4

ARCSEC2_PER_SR = (180.0 * 3600.0 / np.pi) ** 2

#: The SESNA aperture's own solid angle, sr.
OMEGA_AP_SR = np.pi * SESNA_APERTURE_RADIUS_ARCSEC ** 2 / ARCSEC2_PER_SR

#: UWISH2 Table D1's native flux/surface-brightness unit (Froebrich et al.
#: 2015, MNRAS 454, 2586, ReadMe: Ftot/Fmedsb, 1e-19 W/m^2[/arcsec^2]).
UWISH2_FLUX_UNIT_W_M2 = 1.0e-19
#: 1 W/m^2 = (1e7 erg/s) / (1e4 cm^2) = 1e3 erg/s/cm^2.
W_M2_TO_ERG_S_CM2 = 1.0e3

#: The speed of light, m/s (SI-exact, CODATA).
C_M_S = 2.99792458e8

#: The 2MASS Ks isophotal bandwidth, um (Cohen, Wheaton & Megeath 2003,
#: AJ 126, 1090, Table 1) -- the flat-filter equivalent width a narrow
#: line landing in the Ks passband is spread across to make an in-band
#: flux density, the same "flat filter" convention sedfitter's own filter
#: normalisation uses.
KS_BANDWIDTH_UM = 0.262

#: The flat-filter conversion, mJy per (erg/s/cm^2) of in-band 2.12um line
#: flux: `F_nu[mJy] = F_line[erg/s/cm^2] / delta_nu_Ks * 1e26` (1 mJy =
#: 1e-26 erg/s/cm^2/Hz), `delta_nu_Ks = c * delta_lambda_Ks / lambda_Ks^2`.
_KS_WAVELENGTH_UM = definitions.BANDS_BY_KEY["Ks"].wvl_um
_DELTA_NU_KS_HZ = (C_M_S * (KS_BANDWIDTH_UM * 1e-6)
                   / (_KS_WAVELENGTH_UM * 1e-6) ** 2)
KS_INBAND_MJY_PER_ERG_S_CM2 = 1.0e26 / _DELTA_NU_KS_HZ

#: The brightness axis's own tabulation: 41 points spanning the region's
#: fitted lognormal +/- this many standard deviations in log10 Sigma
#: (SPEC_PRIORS.md section 7).
N_SIGMA_GRID = 41
SIGMA_GRID_NSIGMA = 4.0


def eta_for_region(region):
    """`eta_r`: the field's own measured value where an external knot
    survey covers it, else the survey-wide elsewhere value."""
    return ETA.get(region, ETA_ELSEWHERE)


# ---------------------------------------------------------------------------
# 1. N_law (x) K -- the per-source blurred law field
# ---------------------------------------------------------------------------

def _adopted_columns(config, region):
    """`(a_col, provenance)`, every source of `region`, in catalogue row
    order -- the same adopted-column product `population.yso.law_count` reads
    its own copy of."""
    path = config_module.product_path(config, "sky/derived", "adopted",
                                       "column", "source", region=region)
    cols = access.per_source(config, region, path,
                              ["A_COL_K", "A_COL_PROVENANCE"])
    return (np.asarray(cols["A_COL_K"], dtype=float),
            np.asarray(cols["A_COL_PROVENANCE"]))


def _source_positions(config, region):
    """`(l_deg, b_deg)`, every source of `region`, in catalogue row order,
    from the curated catalogue's own galactic coordinates."""
    path = config_module.product_path(config, "catalog", "sesna",
                                       "sources", "source", region=region)
    cols = access.per_source(config, region, path,
                              ["GAL_L_DEG", "GAL_B_DEG"])
    return (np.asarray(cols["GAL_L_DEG"], dtype=float),
            np.asarray(cols["GAL_B_DEG"], dtype=float))


def gnomonic_project(l_deg, b_deg, l0_deg, b0_deg):
    """Tangent-plane gnomonic projection about `(l0, b0)`, arcmin -- flat
    enough over one region's own few-degree footprint for a 1-arcmin
    binning lattice."""
    l, b = np.radians(l_deg), np.radians(b_deg)
    l0, b0 = np.radians(l0_deg), np.radians(b0_deg)
    dl = l - l0
    cosc = np.sin(b0) * np.sin(b) + np.cos(b0) * np.cos(b) * np.cos(dl)
    x = np.cos(b) * np.sin(dl) / cosc
    y = (np.cos(b0) * np.sin(b) - np.sin(b0) * np.cos(b) * np.cos(dl)) / cosc
    to_arcmin = 180.0 / np.pi * 60.0
    return x * to_arcmin, y * to_arcmin


def centroid_lb(l_deg, b_deg):
    """Circular mean of `l_deg` (wrap-safe), plain mean of `b_deg` -- the
    projection's own tangent point."""
    l = np.radians(l_deg)
    l0 = np.degrees(np.arctan2(np.mean(np.sin(l)), np.mean(np.cos(l)))) % 360.0
    return l0, float(np.mean(b_deg))


def build_offset_kernel(lambda_arcmin, pixel_arcmin=KERNEL_PIXEL_ARCMIN):
    """The offset kernel `K(r) = p(r)/(2 pi r)`, `p(r) = exp(-r/lambda)/
    lambda` (SPEC_PRIORS.md section 7), discretised onto the pixel lattice
    out to `KERNEL_TRUNCATE_LAMBDA * lambda_arcmin` and renormalised to sum
    to exactly 1.0 (the truncated mass is disclosed by construction, not
    separately corrected). The centre pixel avoids the `1/r` cusp with the
    analytic mass enclosed within a half-pixel radius, `1 -
    exp(-r/lambda)`, rather than the divergent `K(0)`.
    """
    half_extent = int(np.ceil(
        KERNEL_TRUNCATE_LAMBDA * lambda_arcmin / pixel_arcmin))
    offsets = np.arange(-half_extent, half_extent + 1) * pixel_arcmin
    dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
    r = np.sqrt(dx ** 2 + dy ** 2)
    pixel_area = pixel_arcmin ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = (np.exp(-r / lambda_arcmin)
                  / (2.0 * np.pi * lambda_arcmin * r)) * pixel_area
    centre = r == 0.0
    kernel[centre] = 1.0 - np.exp(-(0.5 * pixel_arcmin) / lambda_arcmin)
    kernel[~np.isfinite(kernel)] = 0.0
    total = float(np.sum(kernel))
    if total <= 0.0:
        raise ValueError(
            "h2s.build_offset_kernel: lattice sums to <= 0 for "
            "lambda_arcmin=%r" % lambda_arcmin)
    return kernel / total


def blur_law_field(l_deg, b_deg, n_law_source, lambda_arcmin):
    """The offset-blurred law field, sampled back at each source's own
    pixel: a source-count-weighted mean of `n_law_source` on a 1-arcmin
    gnomonic grid, convolved with the physical offset kernel by FFT
    (`scipy.signal.fftconvolve` -- direct convolution does not scale to a
    region's full lattice), with an honest per-source fallback to its own
    pointwise value where the convolved pixel occupancy does not clear
    `KERNEL_COUNT_FLOOR` (edge pixels averaging across mostly-empty
    footprint, never silently zeroed). Returns `(blurred_source,
    n_fallback)`.
    """
    l0, b0 = centroid_lb(l_deg, b_deg)
    x, y = gnomonic_project(l_deg, b_deg, l0, b0)
    xedges = np.arange(np.min(x), np.max(x) + KERNEL_PIXEL_ARCMIN, KERNEL_PIXEL_ARCMIN)
    yedges = np.arange(np.min(y), np.max(y) + KERNEL_PIXEL_ARCMIN, KERNEL_PIXEL_ARCMIN)

    values = np.asarray(n_law_source, dtype=float)
    sum_grid, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=values)
    cnt_grid, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])

    kernel = build_offset_kernel(lambda_arcmin)
    conv_sum = fftconvolve(sum_grid, kernel, mode="same")
    conv_cnt = fftconvolve(cnt_grid, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        blurred_grid = conv_sum / conv_cnt
    blurred_grid = np.where(conv_cnt >= KERNEL_COUNT_FLOOR, blurred_grid, np.nan)

    ix = np.clip(np.digitize(x, xedges) - 1, 0, blurred_grid.shape[0] - 1)
    iy = np.clip(np.digitize(y, yedges) - 1, 0, blurred_grid.shape[1] - 1)
    sampled = blurred_grid[ix, iy]
    is_fallback = ~np.isfinite(sampled)
    blurred_source = np.where(is_fallback, values, sampled)
    return blurred_source.astype(float), int(np.sum(is_fallback))


def build_law_blurred(config, region):
    """Writes `bms/h2s/law-blurred_h2s_source`: the region's law count,
    per source, spatially blurred by the knot-driver displacement kernel
    at this region's own angular `lambda` -- an honest skip (the pointwise
    law count, unchanged) where `lambda` does not resolve against the
    1-arcmin binning pixel. Returns `(path, n_law_source,
    n_law_blurred_source, n_fallback, blur_applied)`.
    """
    a_col, provenance = _adopted_columns(config, region)
    n_law_source = yso_module.law_count(config, region, a_col, provenance)
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    lambda_arcmin = float(np.degrees(LAMBDA_PC / d_r_pc) * 60.0)

    blur_applied = lambda_arcmin >= KERNEL_PIXEL_ARCMIN
    if blur_applied:
        l_deg, b_deg = _source_positions(config, region)
        n_law_blurred, n_fallback = blur_law_field(l_deg, b_deg, n_law_source, lambda_arcmin)
    else:
        # honest skip: this region's own angular offset scale does not
        # resolve against the 1-arcmin lattice, so the pointwise law
        # count is used unchanged.
        n_law_blurred, n_fallback = n_law_source.copy(), 0

    path = config_module.product_path(config, "population", "h2s", "law-blurred",
                                       "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.attrs["LAMBDA_ARCMIN"] = lambda_arcmin
        f.attrs["BLUR_APPLIED"] = int(blur_applied)
        f.create_dataset("N_LAW_BLURRED_DEG2", data=n_law_blurred.astype(np.float64))
    return path, n_law_source, n_law_blurred, n_fallback, blur_applied


# ---------------------------------------------------------------------------
# 2. The region brightness lognormal -- UWISH2 jet knots, transported
# ---------------------------------------------------------------------------

def uwish2_reference_knots(config):
    """`(log10_sigma_native_w_m2_sr, area_pc2)`, one row per UWISH2
    jet-class knot in the two reference fields: each knot's own SURFACE
    BRIGHTNESS (`Ftot/Area`, S-D32's "mean_iso" estimator, the one its
    region-invariance was measured on -- distance-invariant, read
    directly off UWISH2's own photometry, native 2.12um line-flux units,
    W/m^2/sr) and its PHYSICAL area (its own field's adopted distance),
    everything `region_sigma_lognormal` needs to transport this population
    to any target region's own distance.
    """
    path = config_module.product_path(config, "sky/derived", "knots", "uwish2", "survey")
    with h5py.File(path, "r") as f:
        jet = f["JET_CLASS"][:].astype(bool)
        ra = np.asarray(f["RA_DEG"][:], dtype=float)[jet]
        dec = np.asarray(f["DEC_DEG"][:], dtype=float)[jet]
        area_arcsec2 = np.asarray(f["AREA_ARCSEC2"][:], dtype=float)[jet]
        ftot = np.asarray(f["TOTAL_FLUX_1E-19_W_M2"][:], dtype=float)[jet]

    gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    pix256 = hp.ang2pix(256, gal.l.deg, gal.b.deg, nest=True, lonlat=True).astype(np.int64)
    knot_region = access.primary_region_for_pixel(config, pix256)

    good = np.isfinite(ftot) & np.isfinite(area_arcsec2) & (ftot > 0) & (area_arcsec2 > 0)
    keep = np.isin(knot_region, UWISH2_REFERENCE_REGIONS) & good
    if not np.any(keep):
        raise ValueError(
            "h2s.uwish2_reference_knots: no UWISH2 jet-class knots matched "
            "the reference regions %r" % (UWISH2_REFERENCE_REGIONS,))

    d_home_by_region = {r: regions_module.REGIONS_BY_NAME[r].d_r_pc
                        for r in UWISH2_REFERENCE_REGIONS}
    d_home_pc = np.array([d_home_by_region[r] for r in knot_region[keep]], dtype=float)

    sb_native_w_m2_sr = (ftot[keep] * UWISH2_FLUX_UNIT_W_M2
                         / area_arcsec2[keep] * ARCSEC2_PER_SR)
    omega_native_sr = area_arcsec2[keep] / ARCSEC2_PER_SR
    area_pc2 = omega_native_sr * d_home_pc ** 2
    return np.log10(sb_native_w_m2_sr), area_pc2


def transport_log10_sigma(log10_sb_native, area_pc2, d_target_pc):
    """`log10 Sigma` at `d_target_pc` (SPEC_PRIORS.md section 7): `Sigma =
    Sigma_knot . min(Omega_knot/Omega_ap, 1)`. A resolved knot
    (`Omega_knot > Omega_ap`) keeps its own observed surface brightness
    unchanged, independent of distance; an unresolved knot reduces to
    inverse-square placement of its own total flux from its home distance
    to `d_target_pc` -- both branches join continuously at `Omega_knot ==
    Omega_ap`. Vectorised over knots, one row per row of the input.
    """
    omega_knot_sr = np.asarray(area_pc2, dtype=float) / float(d_target_pc) ** 2
    aperture_term = np.minimum(0.0, np.log10(omega_knot_sr / OMEGA_AP_SR))
    return np.asarray(log10_sb_native, dtype=float) + aperture_term


def region_sigma_lognormal(log10_sigma):
    """`(mean, std)` of `log10 Sigma` over the transported reference
    population -- the region's own brightness lognormal (SPEC_PRIORS.md
    section 7: 0.19 dex where every knot fills the aperture, 0.43 dex in
    Cygnus X; reported here per region)."""
    return float(np.mean(log10_sigma)), float(np.std(log10_sigma, ddof=1))


# ---------------------------------------------------------------------------
# 3. Knot colours -- Ks-equivalent flux plus Giannini's per-band ratios
# ---------------------------------------------------------------------------

def _load_giannini_ratios(config):
    """`{band_key: log10_ratio_array}` for the four IRAC bands Giannini's
    Vela D knots carry a measured colour for (SPEC_PRIORS.md section 7,
    "the knot colours")."""
    path = config_module.product_path(config, "sky/derived", "knots", "colours", "survey")
    out = {}
    with h5py.File(path, "r") as f:
        for band in IRAC_RATIO_BAND_KEYS:
            out[band] = np.asarray(f[band]["LOG10_RATIO"][:], dtype=float)
    return out


def knot_ks_log10_flux(log10_sigma):
    """`(n_knot,)` log10 mJy: each knot's aperture-collected 2.12um line
    flux (`Sigma . Omega_ap`), converted to a Ks-band-equivalent flux
    density through the 2MASS Ks bandwidth's flat-filter convention
    (`KS_INBAND_MJY_PER_ERG_S_CM2`, lifted from the quarry's
    `_h2s_knot_log10_i_mjy_per_sr` -- a narrow line spread evenly across
    the filter's own effective width) -- the one band every knot carries
    a single, exact flux for. The four IRAC bands carry no single flux
    per knot (SPEC_PRIORS.md section 7, "the knot colours"): their own
    empirical `F_band/F_2.12` ratio populations enter the selection
    directly, at the population's own equal weight, in
    `build_region_selection`. J, H and M1 carry no Giannini ratio and so
    never clear a limit, which that function's reduced two-of-eight form
    encodes directly.
    """
    log10_sigma = np.asarray(log10_sigma, dtype=float)
    f_line_w_m2 = 10.0 ** log10_sigma * OMEGA_AP_SR
    f_line_erg_s_cm2 = f_line_w_m2 * W_M2_TO_ERG_S_CM2
    f_ks_mjy = f_line_erg_s_cm2 * KS_INBAND_MJY_PER_ERG_S_CM2
    return np.log10(f_ks_mjy)


# ---------------------------------------------------------------------------
# 4. The per-source selection EPS[n, n_x, n_sigma]
# ---------------------------------------------------------------------------

#: The per-batch working-array budget (`sesnaimpute.batches.batches`).
BATCH_BUDGET_BYTES = 512 << 20

#: Re-exported from `prior.selection`, the one place they are defined.
X_LADDER = selection_module.X_LADDER


def _sorted_giannini_ratios(config):
    """`{band_key: ascending log10_ratio_array}`, cached: Giannini's own
    per-band tables are tiny (13-52 measured knots), so a source's exact
    clearing probability against one is a `searchsorted` on the sorted
    table, not a Monte Carlo draw from it."""
    return {band: np.sort(arr) for band, arr in _load_giannini_ratios(config).items()}


def band_clear_fraction(threshold, sorted_ratio_vals):
    """The EXACT fraction of one IRAC band's own Giannini colour-ratio
    table at or above `threshold` (any shape): every measured ratio is
    an equally-weighted population member (module docstring, point 3),
    so this is the table's own empirical survival fraction, read off by
    `searchsorted` -- no draw, so no Monte Carlo error at all.
    """
    n = sorted_ratio_vals.size
    idx = np.searchsorted(sorted_ratio_vals, threshold, side="left")
    return 1.0 - idx / n


def exact_source_selection(log10_lim, a_query, kappa, log10_sigma_grid, sorted_ratios):
    """`(n_src, n_x, n_sigma)` f8: the EXACT two-of-eight pass fraction
    (module docstring, point 3), computed in closed form rather than by
    drawing colour-ratio combinations. Ks and the dimming are exact
    per (source, ladder point, Sigma point); J, H and M1 never clear.
    Only the four IRAC bands are random, each an independent draw from
    its own tiny Giannini table (disclosed in the module docstring), so
    the two-of-eight test's probability is exact algebra on the four
    bands' own per-point clearing probabilities `p_i` (`band_clear_
    fraction`), not a sampling average: with Ks failing, >=2 of the 4
    IRAC bands must clear; with Ks clearing, >=1 must. Both reduce to a
    direct sum over the 2^4 = 16 ways the four bands can clear, taking
    each band's own probability or its complement.
    """
    log10_ks = knot_ks_log10_flux(log10_sigma_grid)  # (n_sigma,)
    dimming = 0.4 * a_query[:, :, None] * kappa[:, :, :]  # (n_src, n_x, 8)

    ks_clears = (log10_ks[None, None, :]
                 >= (log10_lim[:, None, None, KS_IDX] + dimming[:, :, None, KS_IDX]))

    p = []
    for band, b_idx in zip(IRAC_RATIO_BAND_KEYS, IRAC_RATIO_BAND_IDX):
        threshold = (log10_lim[:, None, None, b_idx] + dimming[:, :, None, b_idx]
                     - log10_ks[None, None, :])  # (n_src, n_x, n_sigma)
        p.append(band_clear_fraction(threshold, sorted_ratios[band]))
    p1, p2, p3, p4 = p
    q1, q2, q3, q4 = (1.0 - x for x in p)

    p0 = q1 * q2 * q3 * q4
    p1_exact = (p1 * q2 * q3 * q4 + q1 * p2 * q3 * q4
                + q1 * q2 * p3 * q4 + q1 * q2 * q3 * p4)
    return np.where(ks_clears, 1.0 - p0, 1.0 - p0 - p1_exact)


def _sigma_at_target(worst_lim, sorted_ratios, target, lo, hi, tol=1.0e-3, max_iter=60):
    """Bisection for the single `log10 Sigma` at which the closed-form
    selection (`exact_source_selection`, monotone non-decreasing in
    Sigma) crosses `target`, evaluated at zero dimming (`x = 0`, so
    `kappa` plays no part) against one fixed per-band limit row. No
    grid: one source, one ladder point, one trial Sigma per iteration.
    """
    log10_lim_row = worst_lim[None, :]
    a_query = np.zeros((1, 1))
    kappa = np.ones((1, 1, worst_lim.size))

    def eps_at(sigma):
        eps = exact_source_selection(log10_lim_row, a_query, kappa, np.array([sigma]), sorted_ratios)
        return float(eps[0, 0, 0])

    if eps_at(hi) < target:
        return hi
    if eps_at(lo) >= target:
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if eps_at(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi


def region_sigma_grid(config, region, logsig_mean, logsig_std):
    """The region's `log10 Sigma` grid for the H2S selection
    (SPEC_PRIORS.md section 7). Runs from `mean - 4 sigma` to the LARGER
    of `mean + 4 sigma` and the Sigma at which the exact selection
    (`exact_source_selection`) reaches 0.999 at `x = 0` (no dimming) and
    the region's own brightest -- i.e. hardest-to-clear, the maximum over
    sources -- per-band limits: the worst-case source's own saturation
    point, found once per region by bisection on the closed-form
    selection, no grid needed for the search. Spacing is the smaller of
    `sigma / 8` and a quarter of the selection's own 0.001-to-0.999
    transition width, so the step where `eps_s` jumps from 0 to order 1
    (the row-1222 defect) is resolved by several cells, never one.
    Returns `(grid, sigma_999, transition_width)`.
    """
    log10_lim = np.log10(limits_module.limits(config, region))
    worst_lim = np.max(log10_lim, axis=0)
    sorted_ratios = _sorted_giannini_ratios(config)

    search_lo = logsig_mean - 2.0 * SIGMA_GRID_NSIGMA * logsig_std
    search_hi = logsig_mean + 8.0 * SIGMA_GRID_NSIGMA * logsig_std
    sigma_999 = _sigma_at_target(worst_lim, sorted_ratios, 0.999, search_lo, search_hi)
    sigma_001 = _sigma_at_target(worst_lim, sorted_ratios, 0.001, search_lo, sigma_999)
    transition_width = max(sigma_999 - sigma_001, 1.0e-6)

    lo = logsig_mean - SIGMA_GRID_NSIGMA * logsig_std
    hi = max(logsig_mean + SIGMA_GRID_NSIGMA * logsig_std, sigma_999)
    spacing = min(logsig_std / 8.0, transition_width / 4.0)
    n_points = max(N_SIGMA_GRID, int(np.ceil((hi - lo) / spacing)) + 1)
    grid = np.linspace(lo, hi, n_points)
    return grid, sigma_999, transition_width


def _row_bytes(n_x, n_sigma):
    """The per-source working-array footprint one batch holds: the
    `(n_x, n_sigma)` per-band clearing-probability arrays, four of them
    plus a few working copies -- no population array at all (`exact_
    source_selection` reads each Giannini table directly)."""
    return 12 * n_x * n_sigma * 8 + N_BANDS * 8 + n_x * N_BANDS * 8


def _region_sigma_grid_only(config, region):
    """`LOG10_SIGMA_GRID` alone, from the region product `write_region`
    already carries (module docstring) -- the one region-level input
    `source_selection` needs beyond the shared, tiny Giannini tables."""
    path = config_module.product_path(config, "population", "h2s", "prior", "region", region=region)
    with h5py.File(path, "r") as f:
        return np.asarray(f["LOG10_SIGMA_GRID"][:], dtype=np.float64)


def source_selection(config, region, log10_lim_8, a_query):
    """`(n, n_x, n_sigma)` f8: the exact H2S knot selection, an ON-THE-FLY
    lookup (owner, 2026-09-06: no more `bms/h2s/selection_h2s_source`
    product, the same change `population.gal` made for its own per-source
    galaxy selection) -- the same closed-form `exact_source_selection`
    the old per-source product batched to disk, now evaluated directly
    from the caller's own per-source limits (`log10_lim_8`, (n, 8)) and
    query extinctions (`a_query`, (n, n_x), already `X_LADDER . A_s`)
    against the region's own Sigma grid (`prior_h2s_region`'s
    `LOG10_SIGMA_GRID`) and the shared Giannini colour-ratio tables.
    `population.counts_cloud.build_region` is the one caller, one already-
    loaded source batch at a time -- this function does no batching of
    its own, only the small region-product and ratio-table reads (both
    cheap enough to repeat every call).
    """
    sorted_ratios = _sorted_giannini_ratios(config)
    log10_sigma_grid = _region_sigma_grid_only(config, region)
    w_dense = selection_module.law_dense_weight(a_query)
    kappa = selection_module.kappa_hybrid(config, w_dense)
    return exact_source_selection(log10_lim_8, a_query, kappa, log10_sigma_grid, sorted_ratios)


def report_source_selection(config, region, log10_sigma_grid, logsig_mean, x1_idx,
                             keep_eps_x1_column=False):
    """Batched report-only pass over the whole region (SPEC_PRIORS.md
    section 7 "Checks"), calling the same `source_selection` on-the-fly
    lookup `counts_cloud` uses, never holding more than one batch's `(n_x,
    n_sigma)` rows at once: the median `eps_s` at zero extinction and at
    the source's own column, and the two monotonicity violations
    (non-increasing in the ladder, non-decreasing in Sigma). Returns
    `(n_source, eps_a0_median, eps_x1_median, max_x_violation,
    max_sigma_violation, eps_x1_column)` -- `eps_x1_column` is `(n,
    n_sigma)` at the ladder's `x=1` point, joined across batches only
    when `keep_eps_x1_column` (the caller passes this for Vela D alone,
    the one region the literature check below reads it for): an
    `(n_source, n_sigma)` array is never joined for a million-source
    region (CODING_RULES_BMSTP.md 10a), so elsewhere this returns an
    empty array and the per-batch slice is never taken.
    """
    log10_lim = np.log10(limits_module.limits(config, region))
    n_source = log10_lim.shape[0]
    adopted_path = config_module.product_path(
        config, "sky/derived", "adopted", "column", "source", region=region)
    a_col = np.asarray(
        access.per_source(config, region, adopted_path, ["A_COL_K"])["A_COL_K"], dtype=np.float64)

    x_ladder = X_LADDER
    n_x, n_sigma = x_ladder.size, log10_sigma_grid.size
    j_mean = int(np.argmin(np.abs(log10_sigma_grid - logsig_mean)))

    max_x_violation, max_sigma_violation = 0.0, 0.0
    eps_a0_parts, eps_x1_parts, eps_x1_col_parts = [], [], []
    row_bytes = _row_bytes(n_x, n_sigma)
    for start, stop in batches_module.batches(n_source, row_bytes, budget_bytes=BATCH_BUDGET_BYTES):
        lim_b = np.ascontiguousarray(log10_lim[start:stop])
        a_query_b = np.ascontiguousarray(x_ladder[None, :] * a_col[start:stop, None])
        eps = source_selection(config, region, lim_b, a_query_b)

        # Subtracting eps's own two shifted, OVERLAPPING views directly
        # (`np.diff`, or `eps[:, 1:, :] - eps[:, :-1, :]`) measured, batch
        # after batch on Aquila, as tens of GB of resident memory that
        # never comes back, regardless of whether the difference lands in
        # a fresh array or a reused `out=` buffer -- the overlap itself,
        # not the output allocation, is what a batched loop cannot carry
        # 300 times. Forcing each shifted slice to its own plain
        # (non-overlapping) copy before subtracting removes the growth
        # entirely (measured flat at every batch count).
        if n_x > 1:
            d_x = np.array(eps[:, 1:, :]) - np.array(eps[:, :-1, :])
            max_x_violation = max(max_x_violation, max(0.0, float(np.max(d_x))))
        if n_sigma > 1:
            d_sigma = np.array(eps[:, :, 1:]) - np.array(eps[:, :, :-1])
            max_sigma_violation = max(max_sigma_violation, max(0.0, float(-np.min(d_sigma))))
        eps_a0_parts.append(eps[:, 0, j_mean])
        eps_x1_parts.append(eps[:, x1_idx, j_mean])
        if keep_eps_x1_column:
            eps_x1_col_parts.append(eps[:, x1_idx, :])

    eps_a0 = np.concatenate(eps_a0_parts) if eps_a0_parts else np.empty(0)
    eps_x1 = np.concatenate(eps_x1_parts) if eps_x1_parts else np.empty(0)
    eps_x1_column = (np.concatenate(eps_x1_col_parts, axis=0) if eps_x1_col_parts
                     else np.empty((0, n_sigma)))
    return (n_source, float(np.median(eps_a0)), float(np.median(eps_x1)),
            max_x_violation, max_sigma_violation, eps_x1_column)


def write_region(path, eta, eps_ext, lambda_pc, d_r_pc, logsig_mean, logsig_std,
                  n_knots_ref, log10_sigma_grid):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "region"
        f.create_dataset("ETA", data=np.float64(eta))
        f.create_dataset("ETA_BAND_DEX", data=np.float64(ETA_BAND_DEX))
        f.create_dataset("EPS_EXT", data=np.float64(eps_ext))
        f.create_dataset("EPS_EXT_LO", data=np.float64(EPS_EXT_LO))
        f.create_dataset("EPS_EXT_HI", data=np.float64(EPS_EXT_HI))
        f.create_dataset("LAMBDA_PC", data=np.float64(lambda_pc))
        f.create_dataset("D_R_PC", data=np.float64(d_r_pc))
        f.create_dataset("LOGSIG_MEAN", data=np.float64(logsig_mean))
        f.create_dataset("LOGSIG_STD", data=np.float64(logsig_std))
        f.create_dataset("N_KNOTS_REF", data=np.int64(n_knots_ref))
        f.create_dataset("LOG10_SIGMA_GRID", data=log10_sigma_grid.astype(np.float64))


# ---------------------------------------------------------------------------
# report-only checks (SPEC_PRIORS.md section 7, "Checks")
# ---------------------------------------------------------------------------

def source_pass_fraction(eps_own_column, log10_sigma_grid, logsig_mean, logsig_std):
    """`(n,)`: each source's own `eps_s`, the region's Sigma lognormal
    integrated against `eps_own_column` (n, n_sigma) -- the source's own
    exact selection curve at `x = 1` (its own column, no interpolation:
    every source's EPS is already tabulated at its own `A_s`) --
    vectorised over sources, no per-source Python loop."""
    density = np.exp(-0.5 * ((log10_sigma_grid - logsig_mean) / logsig_std) ** 2)
    density /= np.trapz(density, log10_sigma_grid)
    return np.trapz(eps_own_column * density[None, :], log10_sigma_grid, axis=1)


def kernel_mass_conservation(n_law_source, n_law_blurred):
    """Relative difference between the blurred field's own total and the
    unblurred law field's total (SPEC_PRIORS.md section 7 acceptance: the
    kernel is normalised to sum to 1, so blurring redistributes weight
    without creating or destroying it, within the fallback pixels' own
    honest pass-through)."""
    total_in = float(np.sum(n_law_source))
    total_out = float(np.sum(n_law_blurred))
    return abs(total_out - total_in) / total_in if total_in > 0 else 0.0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Writes, per region, the blurred law field and the exact per-source
    selection (both source granule), and the region product (eta,
    eps_ext, the brightness lognormal). `regions` default: all thirty."""
    import numba
    numba.set_num_threads(max(1, int(config.n_jobs)))
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    log10_sb_native, area_pc2 = uwish2_reference_knots(config)
    n_knots_ref = log10_sb_native.size
    print(f"h2s: {n_knots_ref} UWISH2 jet-class reference knots "
          f"({', '.join(UWISH2_REFERENCE_REGIONS)})")

    x1_idx = int(np.argmin(np.abs(X_LADDER - 1.0)))

    for region in region_names:
        st = progress.Stage("prior.h2s", region)
        d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
        eta = eta_for_region(region)

        path_law, n_law_source, n_law_blurred, n_fallback, blur_applied = build_law_blurred(
            config, region)
        mass_violation = kernel_mass_conservation(n_law_source, n_law_blurred)
        median_ratio = (float(np.median(n_law_blurred)) / float(np.median(n_law_source))
                        if np.median(n_law_source) > 0 else float("nan"))
        print(f"h2s: {region}: law-blurred: median N_LAW={np.median(n_law_source):.4g} "
              f"median N_LAW_BLURRED={np.median(n_law_blurred):.4g} "
              f"(ratio={median_ratio:.4f}) blur_applied={blur_applied} "
              f"n_fallback={n_fallback} mass_violation={mass_violation:.3e} -> {path_law}")

        log10_sigma = transport_log10_sigma(log10_sb_native, area_pc2, d_r_pc)
        logsig_mean, logsig_std = region_sigma_lognormal(log10_sigma)
        log10_sigma_grid, sigma_999, transition_width = region_sigma_grid(
            config, region, logsig_mean, logsig_std)
        old_edge = logsig_mean + SIGMA_GRID_NSIGMA * logsig_std
        tail_mass_beyond_old_edge = 0.5 * math.erfc(SIGMA_GRID_NSIGMA / math.sqrt(2.0))
        print(f"h2s: {region}: sigma grid: [{log10_sigma_grid[0]:.4f}, "
              f"{log10_sigma_grid[-1]:.4f}] n={log10_sigma_grid.size} "
              f"(old fixed edge={old_edge:.4f}, saturation(0.999)={sigma_999:.4f}, "
              f"transition_width={transition_width:.4f} dex) lognormal one-sided tail "
              f"beyond old edge={tail_mass_beyond_old_edge:.3e}")

        path_region = config_module.product_path(config, "population", "h2s", "prior",
                                                  "region", region=region)
        write_region(path_region, eta, EPS_EXT, LAMBDA_PC, d_r_pc, logsig_mean, logsig_std,
                    n_knots_ref, log10_sigma_grid)

        (n_source, eps_a0, eps_x1, max_x_violation, max_sigma_violation,
         eps_x1_column) = report_source_selection(
            config, region, log10_sigma_grid, logsig_mean, x1_idx,
            keep_eps_x1_column=(region == "Vela D"))
        st.done(path_region, n_source=n_source, logsig_mean=logsig_mean, eta=float(eta))
        print(f"h2s: {region}: n_source={n_source} LOGSIG_MEAN={logsig_mean:.4f} "
              f"LOGSIG_STD={logsig_std:.4f} median_eps(x=0,Sigma=mean)={eps_a0:.4f} "
              f"median_eps(x=1,Sigma=mean)={eps_x1:.4f} max_x_violation={max_x_violation:.3e} "
              f"max_sigma_violation={max_sigma_violation:.3e} (on-the-fly, no per-source "
              f"product) -> {path_region}")

        if region == "Vela D":
            eps_own_column = eps_x1_column
            eps_s_source = source_pass_fraction(eps_own_column, log10_sigma_grid, logsig_mean, logsig_std)
            a_col_vela, prov_vela = _adopted_columns(config, region)
            n_law_vela = yso_module.law_count(config, region, a_col_vela, prov_vela)
            knots_per_deg2 = float(np.mean(n_law_vela * eta * eps_s_source))
            print(f"h2s: Vela D literature check: mean(N_law . eta . eps_s) = "
                  f"{knots_per_deg2:.2f} deg^-2, against Giannini+2013's 44-57 deg^-2 "
                  f"(eps_ext left out, per SPEC_PRIORS.md section 7 Checks)")


if __name__ == "__main__":
    run(build)
