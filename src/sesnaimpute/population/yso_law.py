"""The young-star law's coefficient, fitted per SESNA region on a
published census of excess-selected young stars (Class 0, I, flat, II,
transition disk, weak excess), region by region, the pooled value only
where no census exists -- the owner's 2026-09-14 ruling (`_W83_design.md`),
extended by the owner's 2026-10-20 ruling
(`briefs/round4_2026-10-20/C10_yso_census_table.md`) to every published
census of the same kind (`sky.derived.yso_census`'s eight sources, not
Dunham et al. 2015 alone): the young-star level is no longer Pokhrel et
al. 2020's coefficient, it is fitted here, on our own column maps,
nothing about SESNA read except positions and the footprint.

Method, per region: the region's ADMITTED nside-512 pixels (`catalog.
depth_grid`'s own product) restricted to a FIT FOOTPRINT -- those pixels
inside the UNION of the census coverage polygons (`sky.derived.yso_census`'s
own `coverage` group, one convex hull per `(SOURCE, CLOUD)`) overlapping
the region AND, for each such polygon, at or above that census's own
`SURVEY_FLOOR_A_K` column floor wherever it is finite (Dunham et al.
2015's own c2d/Gould-Belt rule, read off the census product's own column
rather than re-derived; every other census here states no comparable
rule, so its own polygon alone is the floor, `SURVEY_FLOOR_A_K = NaN`,
module docstring of `sky.derived.yso_census`) -- `_fit_a_k_min` finds the
`(SOURCE, CLOUD)` group(s) the region's admitted (unrestricted) footprint
holds census objects from (every ingested EXCESS census alike; a
protostar-only census, Kryukova2014/Cygnus X, is counted separately,
`N_CENSUS_PROTOSTAR_ONLY`, and plays no part in the footprint or
`N_CENSUS_REGION` -- the law unit's own choice how it enters), and
`FIT_A_K_MIN` (a report value, the majority-by-count group's own floor,
ties to the stricter -- higher -- finite floor) is reported alongside;
a region with two excess censuses of the SAME kind overlapping it pools
both groups' objects into one footprint (union of their polygons) and
one `N_CENSUS_REGION` (Orion: Megeath2012 is the only one that overlaps,
so this case is not exercised by this round's own three fitted regions).
`CENSUS_SOURCES` records the overlapping census(es)' own `SOURCE` names,
";"-joined. `N_CENSUS_REGION` is the count of EXCESS census objects
(every slope) whose position falls in the fit footprint. The model's own
predicted count over
that same footprint, with the coefficient normalised OUT (so what remains
is a pure geometric/column integral, coefficient 1): `population.yso.
law_area_integral(config, region, pix)` (already `KAPPA_HERSCHEL` times the
geometric integral) times the pixel's solid angle and its `cloud_frac**2`
-- `population.young_stars.build_region` forms exactly this per pixel, so
its own `sightline_lookup` is reused here for `cloud_frac`, not re-derived
-- divided by `KAPPA_HERSCHEL` to strip the old coefficient back out,
summed over the footprint: `I`. `KAPPA_REGION = N / I`, the Poisson
maximum-likelihood coefficient, fitted only where `N >= 50`; its 68%
interval is the standard Poisson (Garwood) interval on `N`, converted by
the same `/ I`. `KAPPA_POOLED` is the geometric mean of the fitted regions'
kappa (the centre of a lognormal cloud-to-cloud scatter, each cloud weighing
the same); `LAW_BAND_DEX` is the rms of the fitted regions' own log10 kappa
about log10 `KAPPA_POOLED`. `KAPPA_USED` is `KAPPA_REGION` where fitted, else
`KAPPA_POOLED`. The class shares (`CLASS_SHARE_PROTO/DISK/WEAK`) are the
whole census's own `ALPHA0` slope fractions, independent of region.

Writes `population/yso/law_yso_region.hdf5` (unchanged path/name); no
`KAPPA_HERSCHEL`/`KAPPA_PLANCK` dataset is written any more -- a reader
that still expects one fails loudly, which is the point.

The diskless Class III members (owner's ruling 2026-10-20, `bms_review/
briefs/round4_2026-10-20/C11_yso_class3_population.md`, with two later
corrections on the same thread): `KAPPA_USED` and `CLASS_SHARE_PROTO/
DISK/WEAK` above count only the EXCESS-SELECTED census (what every
Spitzer census can see); the published disk fraction at the region's
age (`regions.Region.disk_fraction`, `.age_myr`, C12's own literature
table -- a measured sum where an excess-independent census covers the
field, the Richert et al. 2018 relation otherwise, the owner's ruling)
gives the diskless share the excess count implies: `DISK_FRACTION` (per
region, 1.0 -- the identity case -- for a region with no published
value applied), `N_LAW_TOTAL_FACTOR` (the factor `population.yso.
law_count`/`law_area_integral` multiply the excess law by to read the
TOTAL young-star level -- every reader of the level, `bmstp.density`'s
`DENSITY_YSO`, `population.young_stars`'s anchor subtraction and
`bmstp.atlas`'s own YSO member draw, follows through that one pair of
functions, no change of their own; NOT `population.knot_rate`'s
`region_predicted_yso`, which is calibrated per EXCESS-SELECTED young
star and reads an explicit `kappa` instead, deliberately bypassing this
factor), and `CLASS_SHARE_DISKLESS = 1 - 1/N_LAW_TOTAL_FACTOR` (this
region's diskless share of the TOTAL population, read by `bmstp.
template_weights`'s own fourth {CIII} census group).

`N_LAW_TOTAL_FACTOR`'s own formula depends on `regions.Region.disk_
fraction_definition` (owner's ruling 2026-10-20, second correction;
C12's own RULED VALUES section states each region's own definition):
`1 / DISK_FRACTION` where the published fraction counts protostars in
BOTH its excess numerator and its total denominator ("all" -- Richert
et al. 2018, Esplin & Luhman 2020, the bare relation), or
`CLASS_SHARE_PROTO + (1 - CLASS_SHARE_PROTO) / DISK_FRACTION` where
protostars are excluded from both ("disks" -- Ribas et al. 2014, Luhman
et al. 2016, Stelzer & Scholz 2009): the excess law's own protostar
share then stays a share of the excess-selected count, and only the
remaining disk-or-diskless share is inflated by `1 / DISK_FRACTION`.
Both reduce to `N_LAW_TOTAL_FACTOR = 1` at `DISK_FRACTION = 1`.

`disk_fraction`/`n_law_total_factor`/`class_share_diskless` below are
the one place every reader of these three reads from -- `n_law_total_
factor` reads the STORED dataset (not `1 / disk_fraction`), since its
own formula is definition-dependent. CLASS_SHARE_PROTO/DISK/WEAK
themselves are left as fitted -- "the three census shares become
shares of the total" is a report-time identity (each divided by
`N_LAW_TOTAL_FACTOR`, the fourth being CLASS_SHARE_DISKLESS, so the
four sum to 1 per region), not a second stored copy.
"""

import os

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.stats import chi2

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog import depth_grid as depth_grid_module
from sesnaimpute.population import yso as yso_module
from sesnaimpute.population import young_stars as young_stars_module
from sesnaimpute.sky.derived import yso_census as yso_census_module

_STEM = "law_yso_region"
_READINGS = {name: value for (stem, name), value in REGISTRY.items() if stem == _STEM}

#: The fit footprint's own column floor (SPEC ruling, `_W83_design.md`):
#: the c2d coverage rule (A_V >= 2) and the Gould Belt rule (A_V >= 3),
#: at A_K = A_V / 9.something folded into these two mag values directly
#: as the ruling states them.
FIT_A_K_MIN_C2D = 0.22
FIT_A_K_MIN_GB = 0.33

#: A region is fitted only with at least this many census objects in its
#: own fit footprint; fewer leaves `KAPPA_REGION` (and its interval) NaN.
MIN_N_FOR_FIT = 50

#: Report-only column bins (mag of A_K) so the exponent 2 can be eyed
#: per region (the design's acceptance bar).
_REPORT_BIN_EDGES = (0.22, 0.5, 1.0, np.inf)

#: nside-512's own fixed pixel solid angle (deg^2) -- every HEALPix pixel
#: at a given nside has the identical area, so this is one constant, not
#: a per-pixel read.
OMEGA_PIX_512_DEG2 = float(hp.nside2pixarea(512, degrees=True))


# ====================================================================
# the census: positions, clouds, slopes
# ====================================================================

def _census_survey_path(config):
    return config_module.product_path(config, *yso_census_module.PATH_ARGS)


def _decode_strs(arr):
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr])


def _load_census(config):
    """Every ingested census (`sky.derived.yso_census`, owner's
    2026-10-20 ruling, replacing the Dunham-only product this used to
    read): `ra_deg`, `dec_deg`, `source` (the census name, e.g.
    "Dunham2015"/"Megeath2012"), `cloud`, `alpha` (every ingested
    object's own `ALPHA` column, NaN where its census gives no numeric
    slope), `selection` ("excess"/"protostar"), `survey_floor_a_k`,
    `pix512` (the object's own nside-512 galactic NESTED pixel, the
    survey's standard pixelisation -- the SAME frame the region
    admission and depth-grid pixels are in), and `coverage`: `{(source,
    cloud): (poly_ra_deg, poly_dec_deg)}`, every `(SOURCE, CLOUD)`
    group's own convex-hull coverage polygon.

    `alpha0` is ALSO carried, but -- CENSUS INPUT/FOOTPRINT ownership
    boundary (this brief, C11 owns SHARES/LEVEL and `_class_shares`
    below, not touched by this unit) -- restricted to Dunham2015's own
    rows (NaN everywhere else), so `_class_shares`'s statistics are
    numerically UNCHANGED by this ruling's seven added censuses unless
    C11 itself chooses to read the new, unrestricted `alpha`/`source`/
    `selection` instead: pooling every census's own ALPHA into one
    class-share statistic without being asked materially changes that
    statistic (measured while developing this change: Dunham-only
    CLASS_SHARE_PROTO = 0.1796, pooled across all eight censuses'
    ALPHA = 0.3866 -- Megeath2012 alone carries far more high-alpha
    protostar candidates than Dunham's own census does), so this unit
    does not make that choice for C11."""
    path = _census_survey_path(config)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "population.yso_law: no census at %s -- run the "
            "'sesnaimpute.sky.derived.yso_census' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        ra_deg = np.asarray(f["RA_DEG"][:], dtype=np.float64)
        dec_deg = np.asarray(f["DEC_DEG"][:], dtype=np.float64)
        source = _decode_strs(f["SOURCE"][:])
        cloud = _decode_strs(f["CLOUD"][:])
        alpha = np.asarray(f["ALPHA"][:], dtype=np.float64)
        selection = _decode_strs(f["SELECTION"][:])
        survey_floor_a_k = np.asarray(f["SURVEY_FLOOR_A_K"][:], dtype=np.float64)
        coverage = {}
        for name in f["coverage"]:
            g = f["coverage"][name]
            key = (str(g.attrs["SOURCE"]), str(g.attrs["CLOUD"]))
            coverage[key] = (np.asarray(g["RA_DEG"][:], dtype=np.float64),
                              np.asarray(g["DEC_DEG"][:], dtype=np.float64))
    gal = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    pix512 = hp.ang2pix(512, gal.l.deg, gal.b.deg, nest=True, lonlat=True).astype(np.int64)
    alpha0 = np.where(source == "Dunham2015", alpha, np.nan)
    return dict(ra_deg=ra_deg, dec_deg=dec_deg, source=source, cloud=cloud,
                alpha=alpha, alpha0=alpha0, selection=selection, survey_floor_a_k=survey_floor_a_k,
                pix512=pix512, coverage=coverage)


def _cloud_survey_map(census):
    """`{(source, cloud): survey_floor_a_k}`, read directly off the
    loaded census's own `SURVEY_FLOOR_A_K` column (constant within a
    `(SOURCE, CLOUD)` group by construction, `sky.derived.yso_census`'s
    own build): generalises today's Dunham-only `table1.dat` `Survey`
    read (0.22 mag c2d / 0.33 mag Gould Belt) to every census here, most
    of which state no comparable coverage rule (`NaN` -- the fit
    footprint for such a census is its coverage polygon alone)."""
    out = {}
    for source, cloud, floor in zip(census["source"], census["cloud"], census["survey_floor_a_k"]):
        key = (source, cloud)
        if key not in out:
            out[key] = float(floor)
    return out


# ====================================================================
# per-region fit footprint
# ====================================================================

_COLUMN_SIGHTLINE_CACHE = {}


def _column_sightline(config):
    """`(pix256_sorted, a_k_sorted)`, the adopted sightline column's own
    nside-256 pixel and `A_K`, cached (every region's fit reads it)."""
    key = id(config)
    if key not in _COLUMN_SIGHTLINE_CACHE:
        path = config_module.product_path(config, "sky/derived", "adopted",
                                           "column", "sightline")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "population.yso_law: no adopted sightline column product at %s -- run the "
                "'sesnaimpute.sky.derived.column' RUNBOOK line first" % path)
        with h5py.File(path, "r") as f:
            pix256 = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
            a_k = np.asarray(f["A_K"][:], dtype=np.float64)
        order = np.argsort(pix256)
        _COLUMN_SIGHTLINE_CACHE[key] = (pix256[order], a_k[order])
    return _COLUMN_SIGHTLINE_CACHE[key]


def _a_k_of_pix512(config, pix512):
    """Each `pix512` pixel's own gas column, `A_K` of its nside-256
    PARENT sightline (`pix512 >> 2`) -- `NaN` where that parent carries
    no sightline column row."""
    pix256_sorted, a_k_sorted = _column_sightline(config)
    parent = pix512 >> 2
    loc = np.searchsorted(pix256_sorted, parent)
    capped = np.minimum(loc, max(pix256_sorted.size - 1, 0))
    matched = (pix256_sorted.size > 0) & (pix256_sorted[capped] == parent)
    out = np.full(pix512.shape, np.nan, dtype=np.float64)
    out[matched] = a_k_sorted[capped[matched]]
    return out


def _fit_a_k_min(census, survey_map, admitted_pix512):
    """The `(SOURCE, CLOUD)` census group(s) the region's own
    (unrestricted) admitted footprint holds EXCESS census objects from
    (a protostar-only census, e.g. Kryukova2014/Cygnus X, never enters
    the footprint or this report value -- `_fit_region`'s own
    `N_CENSUS_PROTOSTAR_ONLY` counts it separately, the law unit's own
    choice how it then enters), and the majority-by-count group's own
    `SURVEY_FLOOR_A_K` as the report value `FIT_A_K_MIN` (ties go to the
    stricter -- higher -- finite floor; any finite floor outranks NaN in
    this tie-break, generalising today's c2d/Gould-Belt rule to every
    census here). Returns `(fit_a_k_min, groups)`: `groups` is the
    sorted list of EVERY `(SOURCE, CLOUD)` pair present, not only the
    majority one -- `_fit_region` unions all of their coverage polygons
    into one footprint (the 2026-10-20 ruling's own "a region with two
    censuses of the same kind pools their objects"). Returns `(None,
    [])` where the region holds no excess census object at all."""
    excess = census["selection"] == "excess"
    pix512_excess = census["pix512"][excess]
    admitted_sorted = admitted_pix512  # already sorted by caller
    loc = np.searchsorted(admitted_sorted, pix512_excess)
    capped = np.minimum(loc, max(admitted_sorted.size - 1, 0))
    matched = (admitted_sorted.size > 0) & (admitted_sorted[capped] == pix512_excess)
    if not matched.any():
        return None, []
    source_here = census["source"][excess][matched]
    cloud_here = census["cloud"][excess][matched]
    pairs = np.array([f"{s}\x1f{c}" for s, c in zip(source_here, cloud_here)])
    uniq, counts = np.unique(pairs, return_counts=True)
    groups = sorted(tuple(u.split("\x1f")) for u in uniq)
    unknown = [g for g in groups if g not in survey_map]
    if unknown:
        raise KeyError(
            "population.yso_law: census group(s) %r have no SURVEY_FLOOR_A_K "
            "entry -- sky.derived.yso_census's own build is incomplete for "
            "them" % unknown)
    max_count = counts.max()
    top_groups = [tuple(u.split("\x1f")) for u, c in zip(uniq, counts) if c == max_count]
    top_floors = [survey_map[g] if np.isfinite(survey_map[g]) else -np.inf for g in top_groups]
    fit_a_k_min = survey_map[top_groups[int(np.argmax(top_floors))]]
    return fit_a_k_min, groups


def _poisson_interval_68(n):
    """The standard (Garwood) 68% Poisson confidence interval on an
    observed count `n` -- the exact Gamma-function interval (Gehrels
    1986's own construction), not a Gaussian approximation, since
    several regions fit near the `MIN_N_FOR_FIT = 50` floor."""
    alpha = 1.0 - 0.6826894921370859
    lo = 0.0 if n == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * n)
    hi = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (n + 1))
    return lo, hi


def _fit_region(config, region, census, survey_map):
    """One region's fit row: `dict(d_r_pc, pc2_per_deg2, fit_a_k_min,
    n_census, census_sources, n_census_protostar_only, kappa, kappa_lo,
    kappa_hi, i_integral, bin_report)`. `kappa`/`kappa_lo`/`kappa_hi` are
    `NaN` where `n_census < MIN_N_FOR_FIT` or no census group overlaps
    the region at all.

    The fit footprint (owner's 2026-10-20 ruling): the region's admitted
    pixels inside the UNION of the overlapping EXCESS census group(s)'
    own coverage polygon (`sky.derived.yso_census.polygon_contains`),
    each restricted to its own `SURVEY_FLOOR_A_K` where that is finite
    (a census stating no coverage rule contributes its polygon alone, no
    column restriction)."""
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    pc2 = float(yso_module.pc2_per_deg2(d_r_pc))

    admitted = depth_grid_module.admitted_pixels_512(config, region)
    fit_a_k_min, groups = _fit_a_k_min(census, survey_map, admitted)

    row = dict(d_r_pc=float(d_r_pc), pc2_per_deg2=pc2,
               fit_a_k_min=np.nan, n_census=0, census_sources="",
               n_census_protostar_only=0,
               kappa=np.nan, kappa_lo=np.nan, kappa_hi=np.nan,
               i_integral=np.nan, bin_report=None)
    if fit_a_k_min is None or admitted.size == 0:
        return row
    row["fit_a_k_min"] = fit_a_k_min
    row["census_sources"] = ";".join(sorted(set(g[0] for g in groups)))

    admitted_l, admitted_b = hp.pix2ang(512, admitted, nest=True, lonlat=True)
    admitted_icrs = SkyCoord(l=admitted_l * u.deg, b=admitted_b * u.deg, frame="galactic").icrs
    admitted_ra, admitted_dec = admitted_icrs.ra.deg, admitted_icrs.dec.deg
    a_k_admitted = _a_k_of_pix512(config, admitted)

    in_footprint = np.zeros(admitted.shape, dtype=bool)
    for group in groups:
        poly = census["coverage"].get(group)
        if poly is None:
            raise KeyError(
                "population.yso_law: census group %r has no coverage polygon -- "
                "sky.derived.yso_census's own build is incomplete for it" % (group,))
        poly_ra, poly_dec = poly
        inside_poly = yso_census_module.polygon_contains(admitted_ra, admitted_dec, poly_ra, poly_dec)
        floor = survey_map[group]
        if np.isfinite(floor):
            inside_poly = inside_poly & np.isfinite(a_k_admitted) & (a_k_admitted >= floor)
        in_footprint |= inside_poly
    footprint = admitted[in_footprint]
    a_k_footprint = a_k_admitted[in_footprint]
    if footprint.size == 0:
        return row

    excess = census["selection"] == "excess"
    pix512_excess = census["pix512"][excess]
    loc = np.searchsorted(footprint, pix512_excess)
    capped = np.minimum(loc, max(footprint.size - 1, 0))
    matched = (footprint.size > 0) & (footprint[capped] == pix512_excess)
    n_per_pix = np.bincount(capped[matched], minlength=footprint.size).astype(np.float64)
    n_census = int(n_per_pix.sum())
    row["n_census"] = n_census

    protostar = census["selection"] == "protostar"
    pix512_protostar = census["pix512"][protostar]
    loc_p = np.searchsorted(footprint, pix512_protostar)
    capped_p = np.minimum(loc_p, max(footprint.size - 1, 0))
    matched_p = (footprint.size > 0) & (footprint[capped_p] == pix512_protostar)
    row["n_census_protostar_only"] = int(matched_p.sum())

    d_front, d_back = yso_module.cloud_interval_pc(config, region)
    _xi, _mass, cloud_frac, _removed = young_stars_module.sightline_lookup(
        config, region, footprint, d_front, d_back)

    # `population.young_stars.build_region` forms exactly this per
    # pixel (`law_area_integral * omega_pix * cloud_frac**2`), reused
    # rather than re-derived, at coefficient 1: the pure geometric/column
    # integral this fit solves for.
    i_per_pix = (yso_module.law_area_integral(config, region, footprint, kappa=1.0)
                 * OMEGA_PIX_512_DEG2 * cloud_frac ** 2)
    i_integral = float(i_per_pix.sum())
    row["i_integral"] = i_integral

    bins = []
    for lo, hi in zip(_REPORT_BIN_EDGES[:-1], _REPORT_BIN_EDGES[1:]):
        sel = (a_k_footprint >= lo) & (a_k_footprint < hi)
        bins.append((lo, hi, float(n_per_pix[sel].sum()), float(i_per_pix[sel].sum())))
    row["bin_report"] = bins

    if n_census >= MIN_N_FOR_FIT and i_integral > 0:
        kappa = n_census / i_integral
        lo_n, hi_n = _poisson_interval_68(n_census)
        row["kappa"] = kappa
        row["kappa_lo"] = lo_n / i_integral
        row["kappa_hi"] = hi_n / i_integral
    return row


# ====================================================================
# class shares (whole census, region-independent)
# ====================================================================

def _class_shares(census):
    alpha0 = census["alpha0"]
    finite = np.isfinite(alpha0)
    a = alpha0[finite]
    n = a.size
    proto = float(np.count_nonzero(a >= -0.3)) / n
    disk = float(np.count_nonzero((a >= -1.6) & (a < -0.3))) / n
    weak = float(np.count_nonzero(a < -1.6)) / n
    return proto, disk, weak


# ====================================================================
# build
# ====================================================================

def build(config, regions=None):
    """Writes `population/yso/law_yso_region.hdf5` (module docstring).
    `regions` (default: all thirty) selects which regions' own rows are
    fitted and written this call (CODING_RULES.md 5c, other rows left
    untouched); `KAPPA_POOLED`/`LAW_BAND_DEX` are pooled over EVERY
    fitted row the product then holds (this call's and the rows an
    earlier run left in place), so a region-restricted call leaves the
    other regions' `KAPPA_USED` intact; the class shares are the whole
    census's."""
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    census = _load_census(config)
    survey_map = _cloud_survey_map(census)

    rows = {}
    with progress.Stage("population.yso_law") as st:
        for i, region in enumerate(names):
            rows[region] = _fit_region(config, region, census, survey_map)
            st.tick(i + 1, len(names), region)

        proto, disk, weak = _class_shares(census)

        path = config_module.product_path(config, "population", "yso", "law", "region")
        tables_module.update_rows(
            path, names,
            {
                "D_R_PC": np.array([rows[r]["d_r_pc"] for r in names], dtype=np.float64),
                "PC2_PER_DEG2": np.array([rows[r]["pc2_per_deg2"] for r in names], dtype=np.float64),
                "KAPPA_REGION": np.array([rows[r]["kappa"] for r in names], dtype=np.float64),
                "KAPPA_REGION_LO": np.array([rows[r]["kappa_lo"] for r in names], dtype=np.float64),
                "KAPPA_REGION_HI": np.array([rows[r]["kappa_hi"] for r in names], dtype=np.float64),
                "N_CENSUS_REGION": np.array([rows[r]["n_census"] for r in names], dtype=np.int64),
                "FIT_A_K_MIN": np.array([rows[r]["fit_a_k_min"] for r in names], dtype=np.float64),
                "CENSUS_SOURCES": np.array([rows[r]["census_sources"] for r in names], dtype="S64"),
                "N_CENSUS_PROTOSTAR_ONLY": np.array(
                    [rows[r]["n_census_protostar_only"] for r in names], dtype=np.int64),
            },
            granule="region", readings=_READINGS)

        with tables_module.open_product(path, granule="region") as f:
            for stale in ("KAPPA_HERSCHEL", "KAPPA_PLANCK"):
                if stale in f:
                    del f[stale]
            kappa_region = np.asarray(f["KAPPA_REGION"][:], dtype=np.float64)
            # the pooled coefficient is the geometric mean over EVERY
            # fitted cloud in the product (the rows this call fitted and
            # the rows an earlier full run left in place), so a
            # region-restricted call never re-pools over its own rows
            # alone and blanks `KAPPA_USED` for the other regions
            # (2026-09-15: a one-region run of `population.yso` did
            # exactly that). Owner's ruling 2026-09-14: the coefficient
            # scatters lognormally from cloud to cloud, so the centre is
            # the mean of the logarithm and the band the rms about it;
            # each cloud weighs the same.
            fitted_all = np.isfinite(kappa_region)
            if fitted_all.any():
                log_k = np.log10(kappa_region[fitted_all])
                kappa_pooled = float(10.0 ** np.mean(log_k))
                law_band_dex = float(np.sqrt(np.mean((log_k - np.log10(kappa_pooled)) ** 2)))
            else:
                kappa_pooled, law_band_dex = np.nan, np.nan
            fitted = [names[i] for i in range(len(names)) if np.isfinite(rows[names[i]]["kappa"])]
            kappa_used = np.where(fitted_all, kappa_region, kappa_pooled)

            # the young-star law's diskless Class III share (owner's
            # ruling 2026-10-20): DISK_FRACTION is `regions.py`'s own
            # per-region value (1.0, the identity case, for every region
            # with no published age/disk-fraction applied there),
            # carried over the WHOLE 30-region canonical order on every
            # call -- unlike KAPPA_REGION/N_CENSUS_REGION (this call's
            # own `names` only), there is nothing here to FIT, so there
            # is no "untouched region" to leave alone: every row is
            # simply read straight off `regions.py` again.
            #
            # N_LAW_TOTAL_FACTOR depends on `regions.Region.disk_fraction_
            # definition` (owner's ruling 2026-10-20, second correction):
            # "all" -- the published fraction counts protostars in BOTH
            # its excess numerator and its total denominator (Richert et
            # al. 2018, Esplin & Luhman 2020, and the bare relation) --
            # `1 / DISK_FRACTION`, as before; "disks" -- the published
            # fraction is Class II(+TD) over Class II+III, PROTOSTARS
            # EXCLUDED from both (Ribas et al. 2014, Luhman et al. 2016,
            # Stelzer & Scholz 2009) -- `CLASS_SHARE_PROTO + (1 -
            # CLASS_SHARE_PROTO) / DISK_FRACTION`: the excess law's own
            # protostar share stays a share of the excess-selected
            # count (protostars are never in the "disks" fraction's own
            # bookkeeping at all), and only the REMAINING, disk-or-
            # diskless share is inflated by `1 / DISK_FRACTION`.
            # CLASS_SHARE_DISKLESS = 1 - 1/N_LAW_TOTAL_FACTOR in BOTH
            # cases (the diskless members' own share of the resulting
            # TOTAL population; not `1 - DISK_FRACTION`, which only
            # equals that in the "all" case). Where DISK_FRACTION = 1,
            # N_LAW_TOTAL_FACTOR = 1 regardless of definition and
            # CLASS_SHARE_DISKLESS = 0, so `population.yso.law_count`/
            # `law_area_integral` (the one place every reader of the
            # level follows through) and `bmstp.template_weights`'s
            # fourth {CIII} group change nothing for that region (the
            # Identities section's own bit-identical check). CLASS_
            # SHARE_PROTO/DISK/WEAK themselves are left exactly as
            # fitted above -- shares of the EXCESS-SELECTED census
            # alone, read as-is by `bmstp.template_weights.
            # yso_population_weight` (survey-wide, unchanged); "shares
            # of the total" is a report-time identity, printed below,
            # not a second stored copy of these three numbers.
            all_names = [r.name for r in regions_module.REGIONS]
            disk_fraction = np.array(
                [regions_module.REGIONS_BY_NAME[r].disk_fraction for r in all_names],
                dtype=np.float64)
            is_disks_definition = np.array(
                [regions_module.REGIONS_BY_NAME[r].disk_fraction_definition == "disks"
                 for r in all_names])
            n_law_total_factor = np.where(
                is_disks_definition,
                proto + (1.0 - proto) / disk_fraction,
                1.0 / disk_fraction)
            class_share_diskless = 1.0 - 1.0 / n_law_total_factor

            for name, value in (
                ("KAPPA_USED", kappa_used),
                ("KAPPA_POOLED", np.float64(kappa_pooled)),
                ("LAW_BAND_DEX", np.float64(law_band_dex)),
                ("CLASS_SHARE_PROTO", np.float64(proto)),
                ("CLASS_SHARE_DISK", np.float64(disk)),
                ("CLASS_SHARE_WEAK", np.float64(weak)),
                ("DISK_FRACTION", disk_fraction),
                ("N_LAW_TOTAL_FACTOR", n_law_total_factor),
                ("CLASS_SHARE_DISKLESS", class_share_diskless),
            ):
                if name in f:
                    del f[name]
                build_module.write_dataset(f, name, value, *_READINGS[name])
            f.attrs["KAPPA_EXPONENT"] = 2.0
            f.attrs["CENSUS"] = "Dunham et al. 2015"

        st.done(path, n_fitted=len(fitted), kappa_pooled=float(kappa_pooled),
                 law_band_dex=float(law_band_dex))

    print("region                    N        I           kappa            68%% interval  A_K_min")
    for region in names:
        r = rows[region]
        k = r["kappa"]
        k_str = ("%8.3f" % k) if np.isfinite(k) else "     NaN"
        lo_str = ("%.3f" % r["kappa_lo"]) if np.isfinite(r["kappa_lo"]) else "NaN"
        hi_str = ("%.3f" % r["kappa_hi"]) if np.isfinite(r["kappa_hi"]) else "NaN"
        amin = ("%.2f" % r["fit_a_k_min"]) if np.isfinite(r["fit_a_k_min"]) else " -- "
        print("%-25s %5d %11.4f  %s  [%s, %s]   %s"
              % (region, r["n_census"], r["i_integral"] if np.isfinite(r["i_integral"]) else float("nan"),
                 k_str, lo_str, hi_str, amin))
        if r["bin_report"]:
            for lo, hi, n_bin, i_bin in r["bin_report"]:
                hi_disp = "inf" if not np.isfinite(hi) else ("%.2f" % hi)
                print("    A_K [%.2f, %s): N=%.0f I=%.4f" % (lo, hi_disp, n_bin, i_bin))
    print("KAPPA_POOLED=%.3f  LAW_BAND_DEX=%.3f  n_fitted=%d"
          % (kappa_pooled, law_band_dex, len(fitted)))
    print("CLASS_SHARE_PROTO=%.4f CLASS_SHARE_DISK=%.4f CLASS_SHARE_WEAK=%.4f"
          % (proto, disk, weak))

    # the diskless Class III share (owner's ruling 2026-10-20): report-
    # time only -- "shares of the total" is never a second stored copy
    # of CLASS_SHARE_PROTO/DISK/WEAK, printed here per processed region
    # as the identity check the brief's own Identities section names
    # ("the four shares sum to 1 for every region"). Total = excess x
    # factor, so a share of the TOTAL is a share of the EXCESS divided
    # by the SAME factor, in both the "all" and "disks" cases alike
    # (`1 / factor` replaces the "all"-only `DISK_FRACTION` this report
    # used before the second correction; the two agree exactly when
    # `definition == "all"`, since then `factor = 1 / DISK_FRACTION`).
    print("region                    DISK_FRACTION  definition  N_LAW_TOTAL_FACTOR  "
          "PROTO_of_total  DISK_of_total  WEAK_of_total  DISKLESS  sum")
    for region in names:
        r_obj = regions_module.REGIONS_BY_NAME[region]
        df = float(r_obj.disk_fraction)
        definition = r_obj.disk_fraction_definition
        factor = (proto + (1.0 - proto) / df) if definition == "disks" else (1.0 / df)
        inv_factor = 1.0 / factor
        proto_tot, disk_tot, weak_tot = proto * inv_factor, disk * inv_factor, weak * inv_factor
        diskless_tot = 1.0 - inv_factor
        total = proto_tot + disk_tot + weak_tot + diskless_tot
        print("%-25s %13.4f  %10s  %18.4f  %14.4f  %13.4f  %13.4f  %8.4f  %.10f"
              % (region, df, definition, factor, proto_tot, disk_tot, weak_tot,
                 diskless_tot, total))
    return path


# ====================================================================
# accessors -- the one place every reader of DISK_FRACTION/N_LAW_TOTAL_
# FACTOR/CLASS_SHARE_DISKLESS reads from, never regions.py directly for
# the two derived quantities (CODING_RULES.md: one stored number, one
# reader convention) -- cached per (config, region) the same way `yso.
# _kappa_used` caches KAPPA_USED. A NaN on disk (a region this file's
# own last build never touched -- a fresh product with no build yet, or
# a build that restricted `regions` to a subset not including this one)
# reads back as the identity value: DISK_FRACTION=1.0, N_LAW_TOTAL_
# FACTOR=1.0, CLASS_SHARE_DISKLESS=0.0, never a propagated NaN.
# ====================================================================

_LAW_ROW_CACHE = {}


def _law_row(config, region, name, identity_value):
    key = (id(config), region, name)
    if key not in _LAW_ROW_CACHE:
        path = config_module.product_path(config, "population", "yso", "law", "region")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "population.yso_law: law product missing at %s -- run the "
                "'sesnaimpute.population.yso_law' RUNBOOK line first" % path)
        with h5py.File(path, "r") as f:
            if name not in f:
                raise KeyError(
                    "population.yso_law: %s lacks %s -- run the "
                    "'sesnaimpute.population.yso_law' RUNBOOK line first" % (path, name))
            region_names = [v.decode("utf-8") if isinstance(v, bytes) else str(v)
                            for v in f["REGION"][:]]
            if region not in region_names:
                raise ValueError("population.yso_law: region %r has no row in %s"
                                  % (region, path))
            value = float(f[name][region_names.index(region)])
        _LAW_ROW_CACHE[key] = identity_value if not np.isfinite(value) else value
    return _LAW_ROW_CACHE[key]


def disk_fraction(config, region):
    """This region's own DISK_FRACTION (module docstring, owner's ruling
    2026-10-20) off `law_yso_region.hdf5` -- 1.0 (identity) where the row
    is NaN (no build has touched this region's row yet)."""
    return _law_row(config, region, "DISK_FRACTION", 1.0)


def n_law_total_factor(config, region):
    """This region's own N_LAW_TOTAL_FACTOR off `law_yso_region.hdf5`
    (module docstring): `1 / disk_fraction(config, region)` where this
    region's own `regions.Region.disk_fraction_definition` is "all", or
    `CLASS_SHARE_PROTO + (1 - CLASS_SHARE_PROTO) / disk_fraction(config,
    region)` where it is "disks" (owner's ruling 2026-10-20, second
    correction) -- read off the stored dataset rather than recomputed
    here, since the formula depends on `definition`, not on `disk_
    fraction` alone. 1.0 (no change) at the identity case."""
    return _law_row(config, region, "N_LAW_TOTAL_FACTOR", 1.0)


def class_share_diskless(config, region):
    """This region's own CLASS_SHARE_DISKLESS = 1 - disk_fraction(config,
    region) off `law_yso_region.hdf5` -- 0.0 (no diskless members) at the
    identity case."""
    return _law_row(config, region, "CLASS_SHARE_DISKLESS", 0.0)


if __name__ == "__main__":
    run(build)
