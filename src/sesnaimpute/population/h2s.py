"""H2S: shocked H2 knots (SPEC_PRIORS.md section 7).

The pure functions here compute the knot brightness lognormal at a
region's own distance from the UWISH2 reference population, and the
Ks-band-equivalent flux a knot's surface brightness gives on that
lognormal's axis. `bmstp.shapes` and `bmstp.atlas` read them directly.

1. `uwish2_reference_knots`: each UWISH2 jet-class knot's surface
   brightness and physical area in the two reference fields
   (SPEC_PRIORS.md section 7: "the 1,349 UWISH2 jet-class knots in
   Cygnus X and the North America Nebula").
2. `transport_log10_sigma`: that population's surface brightness placed
   at a target region's own distance (a resolved knot keeps its own
   observed surface brightness; an unresolved knot scales by
   inverse-square placement of its own total flux) -- `region_sigma_
   lognormal` then takes its `(mean, std)` in log10 Sigma, the region's
   own knot brightness lognormal.
3. `knot_ks_log10_flux`: a knot's aperture-collected 2.12um line flux at
   a given surface brightness, converted to a Ks-band-equivalent flux
   density through the 2MASS Ks bandwidth's flat-filter convention -- the
   one band every knot carries a single, exact flux for. The four IRAC
   bands carry no single flux per knot (SPEC_PRIORS.md section 7, "the
   knot colours"): their own empirical `F_band/F_2.12` colour-ratio
   tables (`_load_giannini_ratios`) are drawn from directly wherever a
   knot's IRAC flux is needed. J, H and M1 carry no Giannini ratio.
"""

import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import healpy as hp

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: The four IRAC bands Giannini's Vela D knots carry a measured colour for.
IRAC_RATIO_BAND_KEYS = ("I1", "I2", "I3", "I4")

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
    (`KS_INBAND_MJY_PER_ERG_S_CM2` -- a narrow line spread evenly across
    the filter's own effective width) -- the one band every knot carries
    a single, exact flux for.
    """
    log10_sigma = np.asarray(log10_sigma, dtype=float)
    f_line_w_m2 = 10.0 ** log10_sigma * OMEGA_AP_SR
    f_line_erg_s_cm2 = f_line_w_m2 * W_M2_TO_ERG_S_CM2
    f_ks_mjy = f_line_erg_s_cm2 * KS_INBAND_MJY_PER_ERG_S_CM2
    return np.log10(f_ks_mjy)
