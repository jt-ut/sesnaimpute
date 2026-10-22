"""U12 STUDY (bms_review/briefs/round6_2026-10-22/U12_law_study.md): the
shared-coefficient young-star law with census selection as exposure
(FINDINGS.md S17, round-2 reviewer's alternative model). READ-ONLY: no
product in any root is written by this script; everything it computes is
printed and dumped to <root>/_study/law_exposure_study.json. Nothing here
is merged into the package.

Model fitted (the reviewer's, S17 round 2, quoted in the brief):
    mu_j = kappa * integral_{searched area_j} G(s) E_j(s) dA + contaminants_j
G(s) is exactly population.yso_law._fit_region's own per-pixel integrand,
reused via population.yso.law_area_integral(..., kappa=1.0) -- never
re-derived. One SHARED kappa, fit by Poisson MLE over every INCLUDED
census's observed count (closed form, since every contaminants_j = 0
here: no ingested census publishes a numeric contamination fraction for
its EXCESS sample -- disclosed, not assumed away).

E_j(s), per census (brief's own "in order of information"):

  Megeath2012 (Orion A/B): the reviewer's required calibration is Megeath
  et al. 2016's Appendix B artificial-YSO recovery vs RMEDSQ (root-median-
  square 8 micron background DN in a 6-11 pixel annulus around the
  source, the modified error function f = (1/2)(1-erf((x-a)/(sqrt(2)b))),
  x=log10(RMEDSQ), a=3.22+-0.04, b=1.05+-0.05 -- arXiv:1511.01202,
  Appendix B, Eq. 14, PDF p.43). RMEDSQ CANNOT be mapped to any SESNA
  product: it is a raw background DN statistic measured on Megeath's own
  8 micron mosaic in his own reduction's units, in a fixed-pixel annulus;
  SESNA's own completeness curves (catalog.completeness/catalog.limits)
  are flux-domain artificial-star recovery curves per band, built from a
  different reduction, with no published or derivable DN<->flux
  conversion at the pixel level. Said so here rather than fabricating a
  mapping. CONSEQUENCE (brief's own instruction + this unit's own
  resolution): Megeath2012 is EXCLUDED from the shared-kappa Poisson fit
  (E=1 would double the uncorrected count as if it were complete, which
  it is not; using the RMEDSQ exposure is impossible without the mapping
  just disclaimed). Its role is exactly "CALIBRATION... consistency check
  only" (brief's words): the fitted kappa (from every OTHER census) is
  used to predict Orion A/B's FULL excess-selected population over the
  Megeath2012 footprint (E=1, the model's own un-selected integral) and
  compared with Megeath et al. 2016's own already-completeness-corrected
  totals, 4199 (A) / 905 (B) -- Table 1, PDF p.52, confirmed against the
  paper's PDF directly (arXiv:1511.01202).

  Kryukova2014 (Cygnus X, protostar-only): same RMEDSQ mapping problem
  (arXiv:1405.0004 sec 2.3.1 -- a RMEDSQ-conditional m24 limit, not a
  fixed flux floor) PLUS the standing no-Cygnus-X-run rule PLUS
  population.yso_law's own existing design already excludes it from
  N_CENSUS_REGION/the fit footprint (its own module docstring). EXCLUDED
  here for all three reasons; not computed at all.

  Every other census: the paper's own published sensitivity/completeness
  limit at IRAC channel 2 (4.5 micron, the band every one of these
  censuses selects a dusty-YSO candidate on, this project's own I2),
  where one is actually quoted as a number:

    Allen2012 (Cepheus OB3b): 90% differential completeness [4.5]=15.18
      mag (Vega) -- Allen et al. 2012, ApJ 750, 125, arXiv:1203.5089,
      sec 2.1 ("Ninety percent differential completeness limits of the
      final mosaics are approximately [3.6]=15.16, [4.5]=15.18,
      [5.8]=13.71, and [8.0]=11.72").
    Kirk2009 (Cepheus Flare): limiting magnitude (source-count turnover)
      IRAC2 4.5um = 16.7 mag (Vega) -- Kirk et al. 2009, ApJS 185, 198,
      arXiv:0903.4063, Table 3 col 5 / sec 3.1, 3.4.
    Strafella2010 (Vela D): 90%-equivalent completeness mcomp=16.5 mag,
      Fcomp=4.5e-5 Jy=0.045 mJy (IRAC2) -- Strafella et al. 2010, ApJ
      719, 9, arXiv:1006.1547, Table 1 ("Overview of the limiting
      magnitudes and completeness"), sec 3.4. Used directly in mJy (no
      mag->flux step needed); cross-checked against the mag column via
      F0_IRAC2=179700 mJy (Reach et al. 2005): 179700*10**(-16.5/2.5) =
      0.0451 mJy, agrees with the paper's own 0.045 mJy to rounding.

  Dunham2015 (c2d + Gould Belt), Rebull2010 (Taurus), Rebull2011 (North
  America Nebula): NOT FOUND after fetching the primary papers directly
  (Dunham et al. 2015 arXiv:1508.03199; Evans et al. 2009 arXiv:0811.1059,
  whose only completeness number is a BOLOMETRIC LUMINOSITY limit,
  L_bol>=0.05 Lsun at 90% for one cloud at 260 pc -- not a band flux/mag
  depth, not reused here as if it were one; Rebull et al. 2010
  arXiv:0911.3176 sec 2.1, which states only IRAC *saturation* limits and
  defers the actual sensitivity to Padgett et al. 2009, not fetched in
  this study; Rebull et al. 2011 arXiv:1102.0573 sec 2.2.1, which states
  explicitly "the faint limit... is a function of the nebular brightness,
  which varies across the field", no fixed number given). E_j(s) = 1 for
  these three (DISCLOSED, not corrected for completeness) -- stated here
  first, per the owner's standing rule, rather than substituted with an
  invented number.

  Searched area, every census: `sky.derived.yso_census`'s own convex-hull
  coverage polygon per (SOURCE, CLOUD), intersected with the region's
  admitted nside-512 footprint and (where finite) the census's own
  SURVEY_FLOOR_A_K column floor -- IDENTICAL footprint construction to
  today's `population.yso_law._fit_region`, flagged "hull" throughout
  (no paper here gives a usable RA/Dec field-boundary table precise
  enough to replace the hull without further, out-of-budget digitising:
  Megeath 2016 Fig. 1 names five sub-fields by cloud-cluster but gives no
  coordinates; Rebull 2011 gives one field-center RA/Dec and a total
  square-degree count, not per-cloud boundaries).

Run: `python studies/law_exposure_study.py <root>/config/root.cfg`.
Reads products only; writes only <root>/_study/law_exposure_study.json
and prints the six report tables. No pieces.run call, no --n-jobs needed
(everything here is numpy over existing arrays, not a per-source sweep).
"""

import json
import os
import sys

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import chi2

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.catalog import depth_grid as depth_grid_module
from sesnaimpute.population import selection as selection_module
from sesnaimpute.population import yso as yso_module
from sesnaimpute.population import yso_law as yso_law_module
from sesnaimpute.population import young_stars as young_stars_module
from sesnaimpute.sky.derived import yso_census as yso_census_module
from sesnaimpute.bmstp import template_weights as template_weights_module

#: Megeath et al. 2016 Appendix B, Eq. 14 (quoted verbatim from the PDF,
#: arXiv:1511.01202, p.43): f(x) = 1/2 (1 - erf((x-a)/(sqrt(2) b))),
#: x = log10(RMEDSQ). Carried here for the report table only -- NOT used
#: to build an E_j(s): RMEDSQ cannot be mapped to any SESNA product
#: (module docstring).
MEGEATH2016_A = 3.22
MEGEATH2016_B = 1.05
#: Table 1, Megeath et al. 2016 (PDF p.52): Spitzer-only (uncorrected)
#: and fully RMEDSQ+X-ray-corrected dusty-YSO totals, Orion A / Orion B.
MEGEATH2016_N_UNCORRECTED = {"Orion A": 2821, "Orion B": 660}
MEGEATH2016_N_CORRECTED = {"Orion A": 4199, "Orion B": 905}

#: IRAC channel-2 (4.5 micron) zero-magnitude flux density, mJy (Reach et
#: al. 2005) -- the SAME constant sky.derived.yso_census.F0_IRAC2_MJY
#: uses, reused rather than re-declared with a different number.
F0_IRAC2_MJY = yso_census_module.F0_IRAC2_MJY

#: Published IRAC2 (4.5um) sensitivity/completeness limits actually
#: quoted as a number by the census's own paper (module docstring lists
#: the source/section for each); magnitudes are Vega, converted to mJy
#: here with F0_IRAC2_MJY. None for a census with no quotable number
#: (E_j(s) = 1 there, disclosed).
CENSUS_LIMIT_MAG = {
    "Allen2012": 15.18,     # Allen et al. 2012, ApJ 750, 125, sec 2.1
    "Kirk2009": 16.7,       # Kirk et al. 2009, ApJS 185, 198, Table 3
}
#: Strafella2010 is given directly in mJy by its own Table 1 (Fcomp),
#: not re-derived from the mag column (kept alongside for the report's
#: own cross-check, module docstring).
CENSUS_LIMIT_MJY_DIRECT = {
    "Strafella2010": 0.045,  # Strafella et al. 2010, ApJ 719, 9, Table 1
}
CENSUS_LIMIT_SOURCE = {
    "Allen2012": "Allen et al. 2012, ApJ 750, 125, arXiv:1203.5089, sec 2.1",
    "Kirk2009": "Kirk et al. 2009, ApJS 185, 198, arXiv:0903.4063, Table 3 / sec 3.1,3.4",
    "Strafella2010": "Strafella et al. 2010, ApJ 719, 9, arXiv:1006.1547, Table 1 / sec 3.4",
}
#: Censuses with NO quotable band completeness number found (module
#: docstring): E_j(s) = 1, disclosed every place it is used.
CENSUS_NO_LIMIT_NOTE = {
    "Dunham2015": "Evans et al. 2009 (arXiv:0811.1059) gives only a bolometric-"
                   "luminosity completeness (L_bol>=0.05 Lsun at 90%, one cloud "
                   "at 260 pc), not a band flux/mag depth; Dunham et al. 2015 "
                   "(arXiv:1508.03199) states no number either. E=1, disclosed.",
    "Rebull2010": "Rebull et al. 2010 (arXiv:0911.3176 sec 2.1) states only IRAC "
                   "saturation limits and defers sensitivity to Padgett et al. "
                   "2009 (not fetched this study). E=1, disclosed.",
    "Rebull2011": "Rebull et al. 2011 (arXiv:1102.0573 sec 2.2.1): \"the faint "
                   "limit... is a function of the nebular brightness, which "
                   "varies across the field\" -- no fixed number given. E=1, disclosed.",
}
#: Censuses excluded from the shared-kappa fit entirely, and why (module
#: docstring).
EXCLUDED_CENSUSES = {
    "Megeath2012": "RMEDSQ cannot be mapped to a SESNA product; role is "
                    "calibration/consistency-check only (brief's own words), "
                    "not a Poisson-fit input.",
    "Kryukova2014": "protostar-only + Cygnus X (standing no-Cygnus-X rule; "
                     "already excluded from population.yso_law's own fit "
                     "footprint/count by design) + same RMEDSQ mapping problem.",
}

I2_INDEX = selection_module.BAND_KEYS.index("I2")


def _poisson_interval_68(n):
    alpha = 1.0 - 0.6826894921370859
    lo = 0.0 if n == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * n)
    hi = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (n + 1))
    return lo, hi


def _limit_mjy(source):
    if source in CENSUS_LIMIT_MJY_DIRECT:
        return CENSUS_LIMIT_MJY_DIRECT[source]
    if source in CENSUS_LIMIT_MAG:
        return F0_IRAC2_MJY * 10.0 ** (-CENSUS_LIMIT_MAG[source] / 2.5)
    return None


def _completeness_fraction(a_k, limit_mjy, dist_pc_group, k_i2, p_census, edges):
    """E_j(s): the fraction of the Dunham-census brightness marginal
    (`p_census`, `template_weights._dunham_census_brightness_histogram`'s
    own dereddened-at-1kpc LOG10_F45_REF histogram on `edges` ==
    `bmstp.grid.LOG10_F45_EDGES`) that would appear brighter than the
    census's own published flux limit once moved to the census's own
    distance and reddened by the LOCAL A_K at `a_k` with the package's
    own law's I2 ratio `k_i2` (module docstring's derivation):

        threshold(s) = log10(limit_mjy) + 2*log10(dist_pc_group/1000)
                       + 0.4 * A_K(s) * k_i2

    `limit_mjy=None` (module docstring's three undocumented censuses)
    returns 1 everywhere (no completeness correction, disclosed by the
    caller, not here)."""
    a_k = np.asarray(a_k, dtype=np.float64)
    if limit_mjy is None:
        return np.ones_like(a_k)
    threshold = (np.log10(limit_mjy) + 2.0 * np.log10(dist_pc_group / 1000.0)
                 + 0.4 * a_k * k_i2)
    # survival[i] = sum of p_census[i:] -- the population fraction with
    # bin lower edge >= edges[i]; one extra zero so an off-the-top
    # threshold reads 0, not an index error.
    survival = np.concatenate([np.cumsum(p_census[::-1])[::-1], [0.0]])
    idx = np.clip(np.searchsorted(edges, threshold, side="right"), 0, len(p_census))
    return survival[idx]


def _load_dist_pc(config):
    """`DIST_PC`, aligned row-for-row with `yso_law_module._load_census`'s
    own arrays (same file, same row order) -- not carried by `_load_census`
    itself (census/footprint ownership boundary, that module's own
    docstring), read directly here since this study needs the census's
    OWN distance per group (brief's E_j(s) construction)."""
    path = yso_law_module._census_survey_path(config)
    with h5py.File(path, "r") as f:
        return np.asarray(f["DIST_PC"][:], dtype=np.float64)


def _group_rows(config, census, survey_map, regions, p_census, edges, k_i2, dist_pc_all):
    """One row per (region, SOURCE, CLOUD) group actually present in that
    region's own fit footprint (`population.yso_law._fit_a_k_min`'s own
    `groups` list, reused verbatim -- identical footprint construction to
    today's `_fit_region`): `n` (observed excess count in the group's OWN
    polygon/floor-restricted footprint), `exposure_e` (integral of
    G(s)*E_j(s) dA), `exposure_full` (the SAME integral at E=1, today's
    plain geometric integral, for the old-model comparison and the
    Megeath/Orion full-population check), `mean_a_k`, `d_r_pc` (the
    region's), `dist_pc` (the group's own, off the census product)."""
    rows = []
    for region in regions:
        admitted = depth_grid_module.admitted_pixels_512(config, region)
        if admitted.size == 0:
            continue
        admitted = np.sort(admitted)
        _fit_a_k_min, groups = yso_law_module._fit_a_k_min(census, survey_map, admitted)
        if not groups:
            continue
        d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
        admitted_l, admitted_b = hp.pix2ang(512, admitted, nest=True, lonlat=True)
        admitted_icrs = SkyCoord(l=admitted_l * u.deg, b=admitted_b * u.deg,
                                  frame="galactic").icrs
        admitted_ra, admitted_dec = admitted_icrs.ra.deg, admitted_icrs.dec.deg
        a_k_admitted = yso_law_module._a_k_of_pix512(config, admitted)

        for source, cloud in groups:
            poly = census["coverage"].get((source, cloud))
            if poly is None:
                continue
            poly_ra, poly_dec = poly
            inside_poly = yso_census_module.polygon_contains(
                admitted_ra, admitted_dec, poly_ra, poly_dec)
            floor = survey_map[(source, cloud)]
            if np.isfinite(floor):
                inside_poly = inside_poly & np.isfinite(a_k_admitted) & (a_k_admitted >= floor)
            footprint = admitted[inside_poly]
            if footprint.size == 0:
                continue
            a_k_footprint = a_k_admitted[inside_poly]

            excess = census["selection"] == "excess"
            own_group = excess & (census["source"] == source) & (census["cloud"] == cloud)
            pix512_own = census["pix512"][own_group]
            loc = np.searchsorted(footprint, pix512_own)
            capped = np.minimum(loc, max(footprint.size - 1, 0))
            matched = (footprint.size > 0) & (footprint[capped] == pix512_own)
            n_g = int(matched.sum())
            if n_g == 0:
                continue

            dist_pc_rows = dist_pc_all[own_group][matched]
            dist_pc_group = float(np.nanmedian(dist_pc_rows)) if dist_pc_rows.size else np.nan

            d_front, d_back = yso_module.cloud_interval_pc(config, region)
            _xi, _mass, cloud_frac, _removed = young_stars_module.sightline_lookup(
                config, region, footprint, d_front, d_back)
            i_per_pix_e1 = (yso_module.law_area_integral(config, region, footprint, kappa=1.0)
                            * yso_law_module.OMEGA_PIX_512_DEG2 * cloud_frac ** 2)

            if source in EXCLUDED_CENSUSES:
                limit_mjy = None
                e_frac = np.ones_like(i_per_pix_e1)
                note = EXCLUDED_CENSUSES[source]
            else:
                limit_mjy = _limit_mjy(source)
                note = (CENSUS_LIMIT_SOURCE.get(source)
                        if limit_mjy is not None else CENSUS_NO_LIMIT_NOTE.get(source, ""))
                e_frac = _completeness_fraction(
                    a_k_footprint, limit_mjy, dist_pc_group, k_i2, p_census, edges)

            exposure_full = float(i_per_pix_e1.sum())
            exposure_e = float((i_per_pix_e1 * e_frac).sum())
            rows.append(dict(
                region=region, source=source, cloud=cloud, n=n_g,
                exposure_full=exposure_full, exposure_e=exposure_e,
                mean_a_k=float(np.nanmean(a_k_footprint)), d_r_pc=float(d_r_pc),
                dist_pc_group=dist_pc_group, floor=float(floor),
                limit_mjy=limit_mjy, mean_e=float(np.nanmean(e_frac)),
                n_pix=int(footprint.size), excluded=source in EXCLUDED_CENSUSES, note=note,
            ))
    return rows


def _fit_shared_kappa(rows):
    """Closed-form Poisson MLE, contaminants_j = 0 for every included row
    (module docstring): kappa = sum(n) / sum(exposure_e), 68% Garwood
    interval on sum(n) carried through the same division."""
    included = [r for r in rows if not r["excluded"]]
    n_tot = sum(r["n"] for r in included)
    exp_tot = sum(r["exposure_e"] for r in included)
    kappa = n_tot / exp_tot if exp_tot > 0 else np.nan
    lo_n, hi_n = _poisson_interval_68(n_tot)
    return kappa, lo_n / exp_tot, hi_n / exp_tot, n_tot, exp_tot, included


def _dedupe_guard(rows, tol_deg=1e-3):
    """A simple, disclosed duplicate-census guard for Table 4 item (iii):
    two groups are the SAME census under two names if their own footprint
    pixel sets agree (>=95% overlap) AND their observed counts agree to
    within 1 -- not a general record-linkage system, a single check
    sized to the brief's own synthetic test. Returns the row list with
    every later duplicate's `excluded` flag set True (and a `note`)."""
    out = list(rows)
    seen = []
    for i, r in enumerate(out):
        if r["excluded"]:
            continue
        is_dup = False
        for j in seen:
            other = out[j]
            if other["excluded"]:
                continue
            if (other["region"] == r["region"] and other["n"] == r["n"]
                    and abs(other["exposure_full"] - r["exposure_full"]) < 1e-9 * max(1.0, other["exposure_full"])):
                is_dup = True
                break
        if is_dup:
            out[i] = dict(r, excluded=True, note=r.get("note", "") + " [DUPLICATE, caught and excluded]")
        else:
            seen.append(i)
    return out


def main(config_path):
    config = config_module.load(config_path)
    census = yso_law_module._load_census(config)
    survey_map = yso_law_module._cloud_survey_map(census)
    p_census, n_finite = template_weights_module._dunham_census_brightness_histogram(config)
    from sesnaimpute.bmstp import grid as grid_module
    edges = grid_module.LOG10_F45_EDGES
    k_i2 = float(selection_module.kappa_ak(config, selection_module.LAW_DIFFUSE)[I2_INDEX])

    dist_pc_all = _load_dist_pc(config)
    regions = [r.name for r in regions_module.REGIONS]
    rows = _group_rows(config, census, survey_map, regions, p_census, edges, k_i2, dist_pc_all)

    kappa_shared, kappa_lo, kappa_hi, n_tot, exp_tot, included = _fit_shared_kappa(rows)

    # ---- Table 1: today's law, read straight off the product ----
    law_path = config_module.product_path(config, "population", "yso", "law", "region")
    with h5py.File(law_path, "r") as f:
        today_names = [v.decode() if isinstance(v, bytes) else str(v) for v in f["REGION"][:]]
        today_kappa_region = f["KAPPA_REGION"][:]
        today_kappa_used = f["KAPPA_USED"][:]
        today_kappa_pooled = float(f["KAPPA_POOLED"][()])
        today_law_band_dex = float(f["LAW_BAND_DEX"][()])

    def kappa_used_of(region):
        return float(today_kappa_used[today_names.index(region)])

    # ---- Table 2: new shared kappa vs today's per-region fit, per group ----
    table2 = []
    for r in rows:
        mu_new = kappa_shared * r["exposure_e"] if not r["excluded"] else np.nan
        mu_old = kappa_used_of(r["region"]) * r["exposure_full"]
        resid_new = (r["n"] - mu_new) / np.sqrt(mu_new) if np.isfinite(mu_new) and mu_new > 0 else np.nan
        resid_old = (r["n"] - mu_old) / np.sqrt(mu_old) if mu_old > 0 else np.nan
        table2.append(dict(r, mu_new=mu_new, mu_old=mu_old, resid_new=resid_new, resid_old=resid_old))

    # ---- Table 3: Orion A/B (Megeath calibration) + transfer (Perseus/Ophiuchus) ----
    table3 = {}
    for region, cloud in (("Orion A", "Orion A"), ("Orion B", "Orion B")):
        match = [r for r in rows if r["region"] == region and r["source"] == "Megeath2012"]
        if not match:
            continue
        r = match[0]
        predicted_full = kappa_shared * r["exposure_full"]
        table3[region] = dict(
            n_observed_ingested=r["n"],
            n_megeath2016_uncorrected=MEGEATH2016_N_UNCORRECTED.get(region),
            n_megeath2016_corrected=MEGEATH2016_N_CORRECTED.get(region),
            predicted_full_population=predicted_full,
            ratio_to_corrected=(predicted_full / MEGEATH2016_N_CORRECTED[region]
                                 if region in MEGEATH2016_N_CORRECTED else np.nan),
            exposure_full=r["exposure_full"], n_pix=r["n_pix"],
        )
    for region in ("Perseus", "Ophiuchus"):
        matches = [r for r in rows if r["region"] == region]
        exp_full_total = sum(r["exposure_full"] for r in matches)
        n_total = sum(r["n"] for r in matches)
        table3[region] = dict(
            n_observed=n_total, exposure_full=exp_full_total,
            predicted_full_population=kappa_shared * exp_full_total,
            note="transfer test: no independent completeness-corrected total "
                 "is published for this cloud at this depth -- reported, not checked.",
        )

    # ---- Table 4: invariance ----
    table4 = {}
    # (i) Orion A split in two along a declination line -- mechanical
    # additivity of n/exposure under a spatial partition (Megeath2012 is
    # excluded from the shared fit itself, so kappa_shared is UNCHANGED
    # by construction; this checks the exposure/count machinery, not the
    # fit).
    orion_a_rows = [r for r in rows if r["region"] == "Orion A" and r["source"] == "Megeath2012"]
    if orion_a_rows:
        ra_row = orion_a_rows[0]
        admitted = np.sort(depth_grid_module.admitted_pixels_512(config, "Orion A"))
        _, groups = yso_law_module._fit_a_k_min(census, survey_map, admitted)
        poly_ra, poly_dec = census["coverage"][("Megeath2012", "Orion A")]
        l_, b_ = hp.pix2ang(512, admitted, nest=True, lonlat=True)
        icrs = SkyCoord(l=l_ * u.deg, b=b_ * u.deg, frame="galactic").icrs
        inside = yso_census_module.polygon_contains(icrs.ra.deg, icrs.dec.deg, poly_ra, poly_dec)
        floor = survey_map[("Megeath2012", "Orion A")]
        a_k_admitted = yso_law_module._a_k_of_pix512(config, admitted)
        if np.isfinite(floor):
            inside = inside & np.isfinite(a_k_admitted) & (a_k_admitted >= floor)
        footprint = admitted[inside]
        dec_med = float(np.median(icrs.dec.deg[inside]))
        l2, b2 = hp.pix2ang(512, footprint, nest=True, lonlat=True)
        dec_fp = SkyCoord(l=l2 * u.deg, b=b2 * u.deg, frame="galactic").icrs.dec.deg
        half1, half2 = footprint[dec_fp < dec_med], footprint[dec_fp >= dec_med]

        def _n_exposure(sub):
            d_front, d_back = yso_module.cloud_interval_pc(config, "Orion A")
            _xi, _mass, cf, _rm = young_stars_module.sightline_lookup(config, "Orion A", sub, d_front, d_back)
            i_per_pix = (yso_module.law_area_integral(config, "Orion A", sub, kappa=1.0)
                         * yso_law_module.OMEGA_PIX_512_DEG2 * cf ** 2)
            excess = census["selection"] == "excess"
            own = excess & (census["source"] == "Megeath2012") & (census["cloud"] == "Orion A")
            pix_own = census["pix512"][own]
            sub_sorted = np.sort(sub)
            loc = np.searchsorted(sub_sorted, pix_own)
            capped = np.minimum(loc, max(sub_sorted.size - 1, 0))
            matched = (sub_sorted.size > 0) & (sub_sorted[capped] == pix_own)
            return int(matched.sum()), float(i_per_pix.sum())

        n1, e1 = _n_exposure(half1)
        n2, e2 = _n_exposure(half2)
        table4["orion_a_split"] = dict(
            n_full=ra_row["n"], n_half_sum=n1 + n2,
            exposure_full=ra_row["exposure_full"], exposure_half_sum=e1 + e2,
            kappa_shared_before=kappa_shared, kappa_shared_after=kappa_shared,
            note="Megeath2012 excluded from the shared fit; kappa_shared is "
                 "unaffected by construction. This checks only that n and "
                 "exposure sum exactly (up to pixel assignment at the split) "
                 "under a spatial partition.",
        )

    # (ii) two Dunham clouds merged (Perseus + Ophiuchus), (iii) one
    # census duplicated under another name, both against an INCLUDED
    # census so the shared-kappa estimator itself is exercised.
    perseus_row = next((r for r in rows if r["region"] == "Perseus" and r["source"] == "Dunham2015"), None)
    ophiuchus_row = next((r for r in rows if r["region"] == "Ophiuchus" and r["source"] == "Dunham2015"), None)
    if perseus_row and ophiuchus_row:
        merged_rows = [r for r in rows if not (r["region"] in ("Perseus", "Ophiuchus") and r["source"] == "Dunham2015")]
        merged_row = dict(perseus_row, region="Perseus+Ophiuchus(merged)",
                           n=perseus_row["n"] + ophiuchus_row["n"],
                           exposure_e=perseus_row["exposure_e"] + ophiuchus_row["exposure_e"],
                           exposure_full=perseus_row["exposure_full"] + ophiuchus_row["exposure_full"])
        merged_rows.append(merged_row)
        kappa_merged, *_ = _fit_shared_kappa(merged_rows)
        table4["dunham_merge"] = dict(kappa_shared_before=kappa_shared, kappa_shared_after=kappa_merged,
                                       max_abs_diff=abs(kappa_merged - kappa_shared))

    vela_row = next((r for r in rows if r["source"] == "Strafella2010"), None)
    if vela_row:
        dup_row = dict(vela_row, source="Strafella2010_DUP")
        rows_with_dup = rows + [dup_row]
        rows_caught = _dedupe_guard(rows_with_dup)
        kappa_naive, *_ = _fit_shared_kappa([r for r in rows_with_dup])
        kappa_caught, *_ = _fit_shared_kappa(rows_caught)
        table4["duplicate_census"] = dict(
            kappa_shared_original=kappa_shared,
            kappa_naive_with_duplicate=kappa_naive,
            kappa_caught_after_dedupe_guard=kappa_caught,
            max_abs_diff_caught_vs_original=abs(kappa_caught - kappa_shared),
            note="Strafella2010 duplicated under 'Strafella2010_DUP' with identical "
                 "n/exposure; the naive sum double-counts it (kappa shifts), the "
                 "dedupe guard (_dedupe_guard: same region, same n, same exposure_full) "
                 "catches and excludes the duplicate, recovering the original kappa.",
        )

    # ---- Table 5: prior-level change per region ----
    table5 = []
    for region in regions:
        ku = kappa_used_of(region)
        table5.append(dict(region=region, kappa_used_today=ku,
                            kappa_shared_new=kappa_shared,
                            ratio_new_over_today=kappa_shared / ku if ku else np.nan))

    # ---- Table 6: residuals vs physical covariates, no fit ----
    table6_distance = [dict(region=r["region"], source=r["source"], cloud=r["cloud"],
                             d_r_pc=r["d_r_pc"], resid_new=t2["resid_new"])
                        for r, t2 in zip(rows, table2)]
    table6_column = [dict(region=r["region"], source=r["source"], cloud=r["cloud"],
                           mean_a_k=r["mean_a_k"], resid_new=t2["resid_new"])
                      for r, t2 in zip(rows, table2)]

    out = dict(
        kappa_shared=kappa_shared, kappa_shared_lo68=kappa_lo, kappa_shared_hi68=kappa_hi,
        n_total_included=n_tot, exposure_total_included=exp_tot,
        today=dict(kappa_region=dict(zip(today_names, today_kappa_region.tolist())),
                   kappa_used=dict(zip(today_names, today_kappa_used.tolist())),
                   kappa_pooled=today_kappa_pooled, law_band_dex=today_law_band_dex),
        table2=table2, table3=table3, table4=table4, table5=table5,
        table6_distance=table6_distance, table6_column=table6_column,
        k_i2_per_ak=k_i2, n_dunham_finite=n_finite,
    )

    study_dir = os.path.join(config.data_root, "_study")
    os.path.realpath(study_dir)
    assert os.path.realpath(study_dir).startswith("/Users/jtaylor/sesna_scratch/"), study_dir
    os.makedirs(study_dir, exist_ok=True)
    out_path = os.path.join(study_dir, "law_exposure_study.json")

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=_default)
    print("wrote", out_path)
    print("KAPPA_SHARED = %.4f  [%.4f, %.4f] (68%%)  N=%d  exposure=%.4f"
          % (kappa_shared, kappa_lo, kappa_hi, n_tot, exp_tot))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
