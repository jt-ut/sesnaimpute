"""The H2S knot rate `eta_r`, knots per law-predicted young star, formed at
build time against the fitted young-star law (`_W83_design.md`, `population.
yso.law_count`'s `KAPPA_USED`) -- the owner's 2026-09-14 ruling that nothing
here may carry a coefficient-dependent number as a copied literal.

Three fields carry an external H2 knot survey (SPEC_PRIORS.md section 7,
S-D37b): `KNOT_COMPLETENESS` below is each field's own survey fact,
independent of the law -- completeness (Cygnus X, North America Nebula) or
`eps_lim` depth factor (Vela D) from Froebrich et al. 2015 (UWISH2) and
Giannini et al. 2013 (Vela D). The footprint itself is no longer a fixed
literal area: it is carried as nside-512 pixels (`footprint_pixels`), the
region's own admitted pixels (`catalog/sesna/depth-grid_sesna_hpx512__<R>.
hdf5`'s `HPX_PIX_512`, reused from `population.yso_law._admitted_pix512`)
weighted by their `catalog/sesna/coverage_sesna_hpx512__<R>.hdf5` `FRAC`,
restricted for Cygnus X and North America Nebula to pixels whose centre
falls inside a UWISH2 image (`sky/derived/knots/uwish2_images_knots_
survey.hdf5`, S-D37's own image-square test). Both the raw knot count
(`knots_in_footprint`, from `sky/derived/knots/uwish2_knots_survey.hdf5`
or `sky/derived/knots/giannini2013_knots_survey.hdf5`) and the law's
predicted young-star count (`region_predicted_yso`) are formed on those
SAME pixels: the denominator is `population.yso_law._fit_region`'s own
per-pixel area-integral expression, `population.yso.law_area_integral`
(genuinely area-averaged over the column map) times each pixel's solid
angle and `cloud_frac**2` and `FRAC`, summed -- not the region's
catalogued sources' mean law density times a fixed area, which under-runs
the integral because sources avoid the densest gas (a convex, squared law).

`fitted` turns those into a depth-corrected knot rate per field, against
whichever `KAPPA_USED` the law product currently carries. `pooled` is the
rate reported to every region without its own knot survey: the geometric
mean of the fitted rates restricted to fields under 1 kpc (Cygnus X, at
1.4 kpc, is excluded -- at that distance the knot survey's completeness
correction is the largest of the three and the region's own predicted
young-star count is dominated by Planck-arm columns, so its fitted rate is
the least reliable of the three to carry to every other region). `eta_for_
region` is what every reader calls: a field's own fitted rate where it has
a survey, else the pooled rate.
"""

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.population import yso as yso_module
from sesnaimpute.population import yso_law as yso_law_module
from sesnaimpute.population import young_stars as young_stars_module

#: Each field's own knot-survey depth fact (SPEC_PRIORS.md section 7,
#: S-D37b): the completeness (Cygnus X, North America Nebula) or
#: `eps_lim` depth factor (Vela D) the survey's raw footprint count
#: clears -- a purely geometric/survey fact, independent of the YSO law.
KNOT_COMPLETENESS = {
    "Cygnus X": 0.885,
    "North America Nebula": 0.865,
    "Vela D": 0.675887,
}

#: UWISH2's own image geometry (S-D37, Froebrich et al. 2015 sect. 2 and
#: the WFCAM 2048x2048 pixel array at 0.4 arcsec): each image is this
#: side, in arcmin, a square with sides along RA and Dec.
UWISH2_IMAGE_SIDE_ARCMIN = 13.653

#: The two UWISH2-surveyed fields; Vela D runs against Giannini's own
#: table instead (module docstring).
_UWISH2_REGIONS = ("Cygnus X", "North America Nebula")

#: `{region: (survey product name, JET_CLASS filter)}` -- which
#: `sky/derived/knots/<survey>_knots_survey.hdf5` each field's raw knot
#: count is read from, and whether it needs the UWISH2 jet-class cut.
_KNOT_SURVEY = {
    "Cygnus X": ("uwish2", True),
    "North America Nebula": ("uwish2", True),
    "Vela D": ("giannini2013", False),
}

#: The pooled band's own floor, dex: the fitted fields' scatter about the
#: pooled log-rate is reported at least this wide even where the three
#: fields' own fitted rates happen to agree more closely than this.
ETA_BAND_DEX_FLOOR = 0.45

#: Distance below which a field's own fitted rate enters the pooled
#: average (`pooled`) -- Cygnus X, at 1.4 kpc, is excluded; see the module
#: docstring.
POOLED_D_R_PC_MAX = 1000.0


# ====================================================================
# the footprint: nside-512 pixels
# ====================================================================

def _unit_vectors(ra_deg, dec_deg):
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _uwish2_image_positions(config):
    """UWISH2's own image centres (RA, Dec), `sky/derived/knots/
    uwish2_images_knots_survey.hdf5`, `sky.derived.knots`'s own product
    off `sky.download.h2_knot_surveys.build`'s Table C1 CSV."""
    path = config_module.product_path(config, "sky/derived", "knots", "uwish2_images", "survey")
    with h5py.File(path, "r") as f:
        ra = np.asarray(f["RA_DEG"][:], dtype=np.float64)
        dec = np.asarray(f["DEC_DEG"][:], dtype=np.float64)
    return ra, dec


def _in_uwish2_footprint(ra_deg, dec_deg, img_ra, img_dec, k=12):
    """S-D37's own UWISH2 image-square membership test: each image is a
    `UWISH2_IMAGE_SIDE_ARCMIN` square, sides along RA and Dec (verified
    fact). Only the k nearest image centres can contain a given point, so
    a kd-tree query over unit vectors is exact for this test and needs no
    all-pairs pass."""
    half_side_deg = UWISH2_IMAGE_SIDE_ARCMIN / 120.0
    tree = cKDTree(_unit_vectors(img_ra, img_dec))
    kk = min(k, img_ra.size)
    _, idx = tree.query(_unit_vectors(ra_deg, dec_deg), k=kk)
    idx = np.atleast_2d(idx)
    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    dra = (ra_deg[:, None] - img_ra[idx] + 180.0) % 360.0 - 180.0
    ddec = dec_deg[:, None] - img_dec[idx]
    inside = ((np.abs(ddec) <= half_side_deg)
              & (np.abs(dra * np.cos(np.radians(dec_deg[:, None]))) <= half_side_deg))
    return inside.any(axis=1)


def _coverage_frac(config, region, pix512):
    """Each `pix512` pixel's own `catalog.coverage` `FRAC` (join on
    `HPX_PIX`, `catalog/sesna/coverage_sesna_hpx512__<R>.hdf5`); a pixel
    absent from the coverage product has `FRAC = 0`. Duplicated from
    `bmstp.atlas._coverage`'s own join rather than imported: population
    may not import bmstp (bmstp imports population)."""
    path = config_module.product_path(config, "catalog", "sesna", "coverage", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)
    order = np.argsort(cov_pix)
    cov_pix_sorted = cov_pix[order]
    loc = np.searchsorted(cov_pix_sorted, pix512)
    loc = np.minimum(loc, max(cov_pix_sorted.size - 1, 0))
    hit = order[loc]
    found = (cov_pix_sorted.size > 0) & (cov_pix_sorted[loc] == pix512)
    out = np.zeros(pix512.shape, dtype=np.float64)
    out[found] = frac[hit[found]]
    return out


def footprint_pixels(config, region):
    """`(pix512, frac)`: `region`'s knot-survey footprint as nside-512
    pixels -- the region's own admitted pixels (`population.yso_law.
    _admitted_pix512`, `catalog/sesna/depth-grid_sesna_hpx512__<R>.
    hdf5`'s `HPX_PIX_512`), each with its own `catalog.coverage` `FRAC`
    (`_coverage_frac`). Cygnus X and North America Nebula are further
    restricted to pixels whose centre falls inside at least one UWISH2
    image square (`_in_uwish2_footprint`); Vela D keeps every admitted
    pixel (Giannini's search covers the same Spitzer mosaic)."""
    pix512 = yso_law_module._admitted_pix512(config, region)
    frac = _coverage_frac(config, region, pix512)
    if region in _UWISH2_REGIONS:
        lon_deg, lat_deg = hp.pix2ang(512, pix512, nest=True, lonlat=True)
        icrs = SkyCoord(l=lon_deg * u.deg, b=lat_deg * u.deg, frame="galactic").icrs
        img_ra, img_dec = _uwish2_image_positions(config)
        keep = _in_uwish2_footprint(icrs.ra.deg, icrs.dec.deg, img_ra, img_dec)
        pix512 = pix512[keep]
        frac = frac[keep]
    return pix512, frac


def footprint_area_deg2(config, region):
    """`region`'s own footprint area, deg^2: `Sigma FRAC * OMEGA_PIX_512_
    DEG2` over `footprint_pixels`."""
    _pix512, frac = footprint_pixels(config, region)
    return float(np.sum(frac) * yso_law_module.OMEGA_PIX_512_DEG2)


# ====================================================================
# the numerator: raw knots on the footprint
# ====================================================================

def knots_in_footprint(config, region, pix512=None):
    """The raw knot count on `region`'s footprint pixels. Cygnus X and
    North America Nebula: rows of `sky/derived/knots/uwish2_knots_
    survey.hdf5` with `JET_CLASS == True`; Vela D: every row of
    `sky/derived/knots/giannini2013_knots_survey.hdf5`. Each knot's own
    RA/Dec is mapped to its nside-512 NESTED Galactic pixel and counted
    where that pixel is one of the footprint's own (`footprint_pixels`,
    reused via `pix512` if the caller already has it)."""
    if pix512 is None:
        pix512, _frac = footprint_pixels(config, region)
    survey, want_jet_class = _KNOT_SURVEY[region]
    path = config_module.product_path(config, "sky/derived", "knots", survey, "survey")
    with h5py.File(path, "r") as f:
        ra = np.asarray(f["RA_DEG"][:], dtype=np.float64)
        dec = np.asarray(f["DEC_DEG"][:], dtype=np.float64)
        if want_jet_class:
            keep = np.asarray(f["JET_CLASS"][:], dtype=bool)
            ra, dec = ra[keep], dec[keep]
    gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    knot_pix = hp.ang2pix(512, gal.l.deg, gal.b.deg, nest=True, lonlat=True).astype(np.int64)
    footprint_sorted = np.sort(pix512)
    loc = np.searchsorted(footprint_sorted, knot_pix)
    loc = np.minimum(loc, max(footprint_sorted.size - 1, 0))
    in_footprint = (footprint_sorted.size > 0) & (footprint_sorted[loc] == knot_pix)
    return int(np.count_nonzero(in_footprint))


def knots_corrected(config, region):
    """`region`'s own depth-corrected knot count on its own footprint --
    a knot-survey fact, not a law quantity: `knots_in_footprint /
    KNOT_COMPLETENESS`."""
    return knots_in_footprint(config, region) / KNOT_COMPLETENESS[region]


# ====================================================================
# the denominator: the law's area integral on the footprint
# ====================================================================

def region_predicted_yso(config, region):
    """The law's own predicted young-star count on `region`'s knot-survey
    footprint, EXACTLY `population.yso_law._fit_region`'s own per-pixel
    expression (module docstring), with the region's own coefficient
    (`kappa=None` makes `population.yso.law_area_integral` read the law
    product's own `KAPPA_USED`), summed over the footprint's own pixels
    (`footprint_pixels`) instead of the fit footprint."""
    pix512, frac = footprint_pixels(config, region)
    d_front, d_back = yso_module.cloud_interval_pc(config, region)
    _xi, _mass, cloud_frac, _removed = young_stars_module.sightline_lookup(
        config, region, pix512, d_front, d_back)
    n_per_pix = (yso_module.law_area_integral(config, region, pix512, kappa=None)
                 * yso_law_module.OMEGA_PIX_512_DEG2 * cloud_frac ** 2 * frac)
    return float(n_per_pix.sum())


# ====================================================================
# fitted / pooled / eta_for_region
# ====================================================================

def fitted(config):
    """`{region: eta_r}` for every field carrying a knot survey: each
    field's own depth-corrected knot count over the law's own predicted
    young-star count on that field's own footprint -- formed here against
    whatever `KAPPA_USED` the law product currently carries, never a
    copied ratio."""
    return {region: knots_corrected(config, region) / region_predicted_yso(config, region)
            for region in KNOT_COMPLETENESS}


def pooled(config, fitted_rates=None):
    """`(eta_pooled, band_dex)`: `eta_pooled` is the geometric mean of the
    fitted rates restricted to fields under `POOLED_D_R_PC_MAX` (Cygnus X
    excluded; see the module docstring), reported to every region without
    its own knot survey; `band_dex` is the larger of `ETA_BAND_DEX_FLOOR`
    and the rms of those fields' own fitted rates' log10 about the pooled
    log."""
    fitted_rates = fitted(config) if fitted_rates is None else fitted_rates
    pooled_regions = [r for r in fitted_rates
                       if regions_module.REGIONS_BY_NAME[r].d_r_pc < POOLED_D_R_PC_MAX]
    log_rates = np.log10(np.array([fitted_rates[r] for r in pooled_regions]))
    eta_pooled = float(10.0 ** np.mean(log_rates))
    scatter = float(np.sqrt(np.mean((log_rates - np.mean(log_rates)) ** 2)))
    band_dex = max(ETA_BAND_DEX_FLOOR, scatter)
    return eta_pooled, band_dex


def eta_for_region(config, region):
    """`(eta_r, band_dex)`: the field's own fitted rate (band 0.0, exact
    for its own footprint) where `region` carries a knot survey, else the
    pooled rate and its band."""
    if region in KNOT_COMPLETENESS:
        return knots_corrected(config, region) / region_predicted_yso(config, region), 0.0
    return pooled(config)
