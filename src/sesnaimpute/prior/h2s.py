"""H2S: shocked H2 knots (SPEC_PRIORS.md section 7).

A knot is where a young star's jet strikes the cloud; it exists whether or
not its driver is catalogued, so the count rides on the YSO **law** count
(`prior.yso.law_count`), never on the selected YSO count:

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
3. `eps_s(a, Sigma)`: the selection tables, per region, on the knots'
   own aperture-limited surface brightness `Sigma` and the column grid's
   `a` nodes -- the same exact-selection-per-depth-group machinery every
   other class uses (`prior.selection.split_common_mode`,
   `prior.depth_groups.DepthGroups`), but with the measured UWISH2
   knot population standing in for a template library (C3): every
   jet-class UWISH2 knot in Cygnus X and the North America Nebula,
   transported to this region's distance, its aperture-limited surface
   brightness converted to a Ks-band-equivalent flux density and thence,
   through Giannini+2013's own measured knot colours, to the other seven
   bands. Written per region, `bms/h2s/prior_h2s_region`, alongside
   `eta_r`/`eps_ext` and the region's own brightness lognormal.

No library enters a count or a shape (C3): the h2shock template register
supplies SED templates to the fitter only, never a selection average.
"""

import os

import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import healpy as hp
from scipy.signal import fftconvolve

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid as column_grid_module
from sesnaimpute.prior import depth_groups as depth_groups_module
from sesnaimpute.prior import selection as selection_module
from sesnaimpute.prior import yso as yso_module

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
KS_IDX = BAND_KEYS.index("Ks")
IRAC_RATIO_BAND_KEYS = ("I1", "I2", "I3", "I4")
IRAC_RATIO_BAND_IDX = np.array([BAND_KEYS.index(b) for b in IRAC_RATIO_BAND_KEYS])

#: Knots present per intrinsic law-predicted young star, per field
#: (SPEC_PRIORS.md section 7; S-D37b, "eta against the law as served
#: today"): the external knot survey's own knot surface density, depth-
#: corrected, over the law count's integral on the same footprint.
#: Measured only where an external H2 knot survey covers the field; every
#: other region is nearer than Cygnus X, so it takes the elsewhere value.
ETA = {
    "Cygnus X": 0.0105,
    "North America Nebula": 0.0572,
    "Vela D": 0.0668,
}
ETA_ELSEWHERE = 0.06

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

#: The knot-colour ratio draw's fixed seed (SPEC_PRIORS.md section 7,
#: "draw one ratio per knot per band from the empirical arrays, fixed
#: seed").
RATIO_SEED = 0

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
    order -- the same adopted-column product `prior.yso.law_count` reads
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

    path = config_module.product_path(config, "bms", "h2s", "law-blurred",
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


def knot_band_log10_flux(log10_sigma, ratios, seed=RATIO_SEED):
    """`(n_knot, 8)` log10 mJy, `definitions.BANDS` order: each knot's
    aperture-collected 2.12um line flux (`Sigma . Omega_ap`), converted to
    a Ks-band-equivalent flux density through the 2MASS Ks bandwidth's
    flat-filter convention (`KS_INBAND_MJY_PER_ERG_S_CM2`, lifted from the
    quarry's `_h2s_knot_log10_i_mjy_per_sr` -- a narrow line spread evenly
    across the filter's own effective width), then, for the four IRAC
    bands, multiplied by one ratio per knot per band drawn (fixed seed,
    with replacement, vectorised) from Giannini+2013's own measured
    `F_band/F_2.12` distribution. J, H and M1 carry no Giannini ratio and
    so no knot flux (`-inf`, never clears a limit) -- SPEC_PRIORS.md
    section 7 discloses this rather than substituting a library value.
    """
    log10_sigma = np.asarray(log10_sigma, dtype=float)
    n_knot = log10_sigma.size
    f_line_w_m2 = 10.0 ** log10_sigma * OMEGA_AP_SR
    f_line_erg_s_cm2 = f_line_w_m2 * W_M2_TO_ERG_S_CM2
    f_ks_mjy = f_line_erg_s_cm2 * KS_INBAND_MJY_PER_ERG_S_CM2

    out = np.full((n_knot, len(BAND_KEYS)), -np.inf, dtype=float)
    out[:, KS_IDX] = np.log10(f_ks_mjy)

    rng = np.random.default_rng(seed)
    for band in IRAC_RATIO_BAND_KEYS:
        draw = rng.choice(ratios[band], size=n_knot, replace=True)
        out[:, BAND_KEYS.index(band)] = np.log10(f_ks_mjy) + draw
    return out


# ---------------------------------------------------------------------------
# 4. The per-region selection tables EPS[k, node, sigma]
# ---------------------------------------------------------------------------

def _depth_groups_path(config):
    return config_module.product_path(config, "bms", "sesna", "depth-groups", "region")


def region_limit_log10_8(config, region):
    """`(knots, limit_log10)`: the region's depth-group knots and each
    group's own 8-band dimmed-limit origin -- the reference limit with the
    group centre's own Delta added over the five Spitzer bands only (the
    three 2MASS bands, Ks included, carry no per-source depth map and stay
    at the region's reference value, SPEC_PRIORS.md section 1.3)."""
    knots = depth_groups_module.DepthGroups.read(_depth_groups_path(config), region)
    depth_idx = np.array([BAND_KEYS.index(b) for b in selection_module.BANDS_DEPTH])
    limit_log10 = np.broadcast_to(knots.ref_log10_flim, (knots.n_groups, len(BAND_KEYS))).copy()
    limit_log10[:, depth_idx] += knots.group_centres
    return knots, limit_log10


def build_region_selection(a_nodes, config, log10_flux8, log10_sigma_grid, bin_idx, limit_log10):
    """`(eps, bin_counts)`: `eps` is `(K, n_node, n_sigma)`, the fraction
    of the transported knot population that, dimmed through column
    `a_nodes[i]` by `10**(-0.4*a*kappa_i(a))` in every band, clears any two
    of eight bands (SPEC_PRIORS.md section 1.3) at depth group `k`'s own
    limit, binned onto `log10_sigma_grid`.

    Vectorised over knots and bands within each node (one matmul against a
    knot-to-bin indicator); looped only over the shared column-grid nodes,
    the same structure `prior.gal.build_region_selection` uses for its own
    four-band population. The dimming vector `kappa_i(a)` is computed once
    for every node ahead of the loop (`selection.kappa_hybrid` has no
    batched per-node form of its own only in the sense that each node's
    ramp weight differs; the call itself is already vectorised over nodes).
    """
    n_knot = log10_flux8.shape[0]
    n_node = a_nodes.size
    n_sigma = log10_sigma_grid.size
    K = limit_log10.shape[0]
    finite8 = np.isfinite(log10_flux8)
    kappa8_by_node = selection_module.kappa_hybrid(
        config, selection_module.law_dense_weight(a_nodes))          # (n_node, 8)

    indicator = np.zeros((n_knot, n_sigma), dtype=np.float32)
    indicator[np.arange(n_knot), bin_idx] = 1.0
    bin_counts = indicator.sum(axis=0)
    safe_counts = np.where(bin_counts > 0, bin_counts, 1.0)

    eps = np.zeros((K, n_node, n_sigma), dtype=np.float32)
    for i in range(n_node):
        dimmed = log10_flux8 - 0.4 * a_nodes[i] * kappa8_by_node[i][None, :]   # (n_knot, 8)
        clears = finite8 & (dimmed[None, :, :] >= limit_log10[:, None, :])     # (K, n_knot, 8)
        passed = (clears.sum(axis=2) >= selection_module.MIN_BANDS).astype(np.float32)
        num = passed @ indicator                                              # (K, n_sigma)
        eps[:, i, :] = np.where(bin_counts > 0, num / safe_counts, 0.0)
    return eps, bin_counts


def write_region(path, eta, eps_ext, lambda_pc, d_r_pc, logsig_mean, logsig_std,
                  n_knots_ref, log10_sigma_grid, a_nodes, eps, group_centres):
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
        f.create_dataset("A_NODES", data=a_nodes.astype(np.float64))
        f.create_dataset("EPS", data=eps.astype(np.float32))
        f.create_dataset("GROUP_CENTRES", data=group_centres.astype(np.float64))


# ---------------------------------------------------------------------------
# report-only checks (SPEC_PRIORS.md section 7, "Checks")
# ---------------------------------------------------------------------------

def source_pass_fraction(a_col, nodes, group_idx, eps_table, log10_sigma_grid,
                          logsig_mean, logsig_std):
    """`(n,)`: each source's own `eps_s`, the region's Sigma lognormal
    integrated against its depth group's selection curve, node-blended --
    vectorised over sources, no per-source Python loop."""
    node_lo, node_w = column_grid_module.bracket(a_col, nodes)
    curve_lo = eps_table[group_idx, node_lo, :]
    curve_hi = eps_table[group_idx, np.clip(node_lo + 1, 0, nodes.size - 1), :]
    curve = (1.0 - node_w)[:, None] * curve_lo + node_w[:, None] * curve_hi
    density = np.exp(-0.5 * ((log10_sigma_grid - logsig_mean) / logsig_std) ** 2)
    density /= np.trapz(density, log10_sigma_grid)
    return np.trapz(curve * density[None, :], log10_sigma_grid, axis=1)


def kernel_mass_conservation(n_law_source, n_law_blurred):
    """Relative difference between the blurred field's own total and the
    unblurred law field's total (SPEC_PRIORS.md section 7 acceptance: the
    kernel is normalised to sum to 1, so blurring redistributes weight
    without creating or destroying it, within the fallback pixels' own
    honest pass-through)."""
    total_in = float(np.sum(n_law_source))
    total_out = float(np.sum(n_law_blurred))
    return abs(total_out - total_in) / total_in if total_in > 0 else 0.0


def eps_monotonicity_violations(eps, bin_counts):
    """`(max_a_violation, max_sigma_violation)`: EPS must be non-increasing
    in `a` (more extinction never helps) and non-decreasing in `log10
    Sigma` (a brighter knot never clears fewer bands); both expected 0.
    The Sigma direction is graded only between adjacent bins that both
    hold at least one transported knot -- an empty bin's EPS is a
    construction default (`build_region_selection`'s own `bin_counts > 0`
    guard), not a measurement, and a boundary against it is not a real
    test of the population's own monotonicity."""
    d_a = np.diff(eps.astype(np.float64), axis=1)
    both_populated = (bin_counts[:-1] > 0) & (bin_counts[1:] > 0)
    d_sigma = np.diff(eps.astype(np.float64), axis=2)[:, :, both_populated]
    sigma_violation = (float(np.max(np.clip(-d_sigma, 0.0, None)))
                       if d_sigma.size else 0.0)
    return float(np.max(np.clip(d_a, 0.0, None))), sigma_violation


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Writes, per region, the blurred law field (source granule) and the
    region product (eta, eps_ext, the brightness lognormal, the selection
    tables). `regions` default: all thirty."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    a_nodes = column_grid_module.nodes(config)
    ratios = _load_giannini_ratios(config)
    log10_sb_native, area_pc2 = uwish2_reference_knots(config)
    n_knots_ref = log10_sb_native.size
    print(f"h2s: {n_knots_ref} UWISH2 jet-class reference knots "
          f"({', '.join(UWISH2_REFERENCE_REGIONS)})")

    for region in region_names:
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
        log10_sigma_grid = np.linspace(
            logsig_mean - SIGMA_GRID_NSIGMA * logsig_std,
            logsig_mean + SIGMA_GRID_NSIGMA * logsig_std, N_SIGMA_GRID)

        log10_flux8 = knot_band_log10_flux(log10_sigma, ratios)
        edges = 0.5 * (log10_sigma_grid[1:] + log10_sigma_grid[:-1])
        bin_idx = np.clip(np.searchsorted(edges, log10_sigma), 0, N_SIGMA_GRID - 1)

        knots, limit_log10 = region_limit_log10_8(config, region)
        eps, bin_counts = build_region_selection(
            a_nodes, config, log10_flux8, log10_sigma_grid, bin_idx, limit_log10)

        path_region = config_module.product_path(config, "bms", "h2s", "prior",
                                                  "region", region=region)
        write_region(path_region, eta, EPS_EXT, LAMBDA_PC, d_r_pc, logsig_mean, logsig_std,
                    n_knots_ref, log10_sigma_grid, a_nodes, eps, knots.group_centres)

        max_a_violation, max_sigma_violation = eps_monotonicity_violations(eps, bin_counts)
        k_median = knots.n_groups // 2
        i_a0 = int(np.argmin(np.abs(a_nodes - 0.0)))
        i_a2 = int(np.argmin(np.abs(a_nodes - 2.0)))
        j_mean = int(np.argmin(np.abs(log10_sigma_grid - logsig_mean)))
        eps_a0 = float(eps[k_median, i_a0, j_mean])
        eps_a2 = float(eps[k_median, i_a2, j_mean])
        print(f"h2s: {region}: LOGSIG_MEAN={logsig_mean:.4f} LOGSIG_STD={logsig_std:.4f} "
              f"K={knots.n_groups} eps(a=0,Sigma=mean)={eps_a0:.4f} "
              f"eps(a=2,Sigma=mean)={eps_a2:.4f} max_a_violation={max_a_violation:.3e} "
              f"max_sigma_violation={max_sigma_violation:.3e} -> {path_region}")

        if region == "Vela D":
            f_lim8 = limits_module.limits(config, region)
            log10_flim8_source = np.log10(f_lim8)
            _s, delta5_source = selection_module.split_common_mode(
                log10_flim8_source, knots.ref_log10_flim)
            group_idx = knots.assign_group(delta5_source)
            a_col_vela, prov_vela = _adopted_columns(config, region)
            eps_s_source = source_pass_fraction(
                a_col_vela, a_nodes, group_idx, eps, log10_sigma_grid, logsig_mean, logsig_std)
            n_law_vela = yso_module.law_count(config, region, a_col_vela, prov_vela)
            knots_per_deg2 = float(np.mean(n_law_vela * eta * eps_s_source))
            print(f"h2s: Vela D literature check: mean(N_law . eta . eps_s) = "
                  f"{knots_per_deg2:.2f} deg^-2, against Giannini+2013's 44-57 deg^-2 "
                  f"(eps_ext left out, per SPEC_PRIORS.md section 7 Checks)")


if __name__ == "__main__":
    run(build)
