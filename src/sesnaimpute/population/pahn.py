"""PAHN: the star-less PAH nebular source population's sky density
(briefs/round4_2026-10-20/C14_pahn_population.md; owner's 2026-10-20
ruling, "H2S is the pattern" -- PAHN carries its own level, shape and
weights like H2S, a nebular population on the cloud). This module is
PAHN's LEVEL; the SHAPE lives in `bmstp.sample_cloud`/`bmstp.shapes`, the
WEIGHTS in `bmstp.template_weights`, built the same architecture as
`population.pahc_curve`'s own contamination curve (module docstring
there): measured directly from SESNA photometry, written per region on
one shared axis, read back as a held-at-the-ends callable.

THE PATTERN: a catalogued source with a measured (`ORIGIN_FNU == 1`) I3
(5.8um) and I4 (8.0um) flux and an unmeasured (`ORIGIN_FNU != 1`) I1
(3.6um) and I2 (4.5um) flux -- any 2MASS, any M1, neither constrains the
pattern -- a patch of PAH-bright nebulosity extracted as a point source,
bright enough in the two reddest IRAC bands to clear the source finder
but carrying no point-source-like near/mid-infrared counterpart. THIS IS
AN UPPER BOUND on the true star-less PAHN population (disclosed at every
reader, `READING` below): the pattern count also contains any true
protostar too faint at 3.6 and 4.5 micron to detect, and artefacts.

THE MEASUREMENT, per region, per admitted nside-512 pixel (the same
pixel axis `catalog.limits.completeness`'s own `HPX_PIX_512` carries,
`bmstp.atlas._depth_grid`'s "admitted pixel axis"): the pattern count on
that pixel, over the pixel's own COVERED solid angle (the any-IRAC union
coverage fraction times the nside-512 pixel area -- the same area
convention `population.knot_rate.footprint_area_deg2` and `bmstp.atlas`'s
own `surveyed_area_deg2` use, so a partially-mosaicked pixel is not
counted as fully surveyed), binned by the pixel's own I4 50%-completeness
limit (`catalog.limits.Completeness.fifty_percent_pixel`, log10 mJy, 0.1
dex bins, ONE shared axis across every region this build call covers --
`population.pahc_curve`'s own `REGION` convention). The per-bin density
is the TOTAL pattern count over the TOTAL covered area of every admitted
pixel in that bin, objects per square degree, with the standard 68%
Garwood Poisson interval (Gehrels 1986) on the bin's own raw count,
carried onto the density scale by the same division. `DENSITY_PAHN`,
`read()`'s own callable, is that binned curve, linear between the
bins a region actually populates and held flat beyond the first/last
one -- `population.pahc_curve.read`'s own convention, restated here
rather than imported (a population module duplicates another's small
read-time convenience rather than reaching into it, the same rule
`population.knot_rate`'s own docstring states for `bmstp.atlas`'s
`_coverage`/this module's `_coverage_frac`).

`bmstp.density` reads this curve at each source's own pixel limit, times
`bmstp.shapes`' own `ON_GRID_PAHN` (the shape's own retained fraction on
the common depth axis) -- no further factor: PAHN carries no external
survey calibration of its own (unlike H2S's `population.knot_rate.eta_
for_region`), so its density is the measured curve and the shape's own
retention alone.

Writes `population/pahn/density_pahn_region.hdf5`. No register is read
here (the PAHN SED-template library does not exist on this branch): this
module reads only the curated catalogue and the measured completeness
product, both already built.
"""

import os

import h5py
import healpy as hp
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import chi2

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module

_STEM = "density_pahn_region"

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
IDX_I1, IDX_I2, IDX_I3, IDX_I4 = (BAND_KEYS.index(k) for k in ("I1", "I2", "I3", "I4"))

#: The pattern's own bin width in log10(I4 50%-completeness limit, mJy)
#: (the brief's own "0.1 dex bins").
LOG10_LIM_BIN_WIDTH_DEX = 0.1

#: HEALPix nside=512 pixel area, deg^2 (the same literal `bmstp.density.
#: _HPX512_PIXEL_DEG2`/`bmstp.atlas._HPX512_PIXEL_DEG2` use: population
#: may not import bmstp, `population.knot_rate`'s own docstring rule, so
#: this one-line constant is the module's own copy, not a cross-layer
#: import).
_HPX512_PIXEL_DEG2 = 41252.96 / (12 * 512 ** 2)

#: The upper-bound disclosure every DENSITY_PAHN/N_PER_BIN reader carries
#: (brief: "Readings").
_UPPER_BOUND_READING = ("an upper bound: the pattern count also contains any true "
                         "protostar too faint at 3.6 and 4.5 to detect, and artefacts")


# ---------------------------------------------------------------------------
# 1. the pattern
# ---------------------------------------------------------------------------

def pattern_mask(fnu, origin):
    """`(n,)` bool: THE PATTERN (module docstring) -- a measured
    (`ORIGIN_FNU == 1`) I3 and I4 flux, an unmeasured (`ORIGIN_FNU != 1`)
    I1 and I2 flux. `fnu` is accepted but unused beyond its own shape
    (the pattern is an `ORIGIN_FNU` test alone, not a flux cut); kept as
    an argument so a caller already holding both arrays passes them the
    same way `population.pahc_curve.region_measurement` does."""
    origin = np.asarray(origin)
    measured_34 = (origin[:, IDX_I3] == 1) & (origin[:, IDX_I4] == 1)
    unmeasured_12 = (origin[:, IDX_I1] != 1) & (origin[:, IDX_I2] != 1)
    return measured_34 & unmeasured_12


# ---------------------------------------------------------------------------
# 2. the covered area per admitted pixel -- duplicated, not imported
# (`population.knot_rate._coverage_frac`'s own docstring: "Duplicated
# from `bmstp.atlas._coverage`'s own join rather than imported: population
# may not import bmstp (bmstp imports population)." -- this is the fourth
# independent copy of the same small join, after `catalog.depth_grid.
# _pixel_coverage`, `bmstp.atlas._coverage` and `population.knot_rate.
# _coverage_frac`, by that same convention.)
# ---------------------------------------------------------------------------

def _coverage_frac(config, region, pix512):
    """Each `pix512` pixel's own any-IRAC union coverage fraction, from
    `sky/derived/spitzer/coverage_spitzer_hpx512__<R>.hdf5`'s own
    per-band `FRAC` (I1 I2 I3 I4 M1 column order): the four IRAC bands'
    `max`, a LOWER BOUND on the true union, exact where the four IRAC
    footprints coincide (nearly everywhere). A pixel absent from the
    product (never touched by a mosaic) reads 0."""
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "population.pahn: no Spitzer mosaic coverage for %r at %s -- run the "
            "'sesnaimpute.sky.derived.coverage' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        cov_pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)
    union = frac[:, :4].max(axis=1) if frac.size else np.empty(0, dtype=np.float64)
    order = np.argsort(cov_pix)
    cov_pix_sorted = cov_pix[order]
    loc = np.searchsorted(cov_pix_sorted, pix512)
    loc = np.minimum(loc, max(cov_pix_sorted.size - 1, 0))
    hit = order[loc]
    found = (cov_pix_sorted.size > 0) & (cov_pix_sorted[loc] == pix512)
    out = np.zeros(pix512.shape, dtype=np.float64)
    out[found] = union[hit[found]]
    return out


# ---------------------------------------------------------------------------
# 3. one region's own per-pixel measurement
# ---------------------------------------------------------------------------

def region_measurement(config, region):
    """`(i4_limit_mjy, area_deg2, pattern_count)`, one row per admitted
    pixel (`catalog.limits.completeness`'s own `HPX_PIX_512` axis, the
    same "admitted pixel axis" `bmstp.atlas._depth_grid` reads): each
    pixel's own I4 50%-completeness limit, its own covered solid angle
    (`_coverage_frac` times `_HPX512_PIXEL_DEG2`), and the pattern count
    whose own nside-512 NESTED pixel (`catalog.sesna`'s `GAL_L_DEG`/
    `GAL_B_DEG`) is that pixel. Every pattern source lands on an admitted
    pixel by construction (the admitted set is every pixel carrying a
    catalogued source, `catalog.depth_grid.admitted_pixels_512`'s own
    docstring) -- checked here, not assumed, rule 6's "no silent
    fallback"."""
    reader = limits_module.completeness(config, region)
    pix512 = reader.hpx_pix_512
    rows = np.arange(pix512.shape[0])
    i4_limit_mjy = reader.fifty_percent_pixel("I4", rows)
    coverage = _coverage_frac(config, region, pix512)
    area_deg2 = coverage * _HPX512_PIXEL_DEG2

    cat_path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(cat_path):
        raise FileNotFoundError(
            "population.pahn: no curated catalogue for %r at %s -- run the "
            "'sesnaimpute.catalog.curated' RUNBOOK line first" % (region, cat_path))
    with h5py.File(cat_path, "r") as f:
        fnu = f["FNU_MJY"][:]
        origin = f["ORIGIN_FNU"][:]
        gl = np.asarray(f["GAL_L_DEG"][:], dtype=np.float64)
        gb = np.asarray(f["GAL_B_DEG"][:], dtype=np.float64)

    pattern = pattern_mask(fnu, origin)
    src_pix = hp.ang2pix(512, gl, gb, nest=True, lonlat=True).astype(np.int64)

    order = np.argsort(pix512)
    pix_sorted = pix512[order]
    src_pix_pattern = src_pix[pattern]
    loc = np.searchsorted(pix_sorted, src_pix_pattern)
    loc = np.minimum(loc, max(pix_sorted.size - 1, 0))
    hit = order[loc]
    found = (pix_sorted.size > 0) & (pix_sorted[loc] == src_pix_pattern)
    if not np.all(found):
        bad = int(src_pix_pattern[~found][0]) if np.any(~found) else -1
        raise ValueError(
            "population.pahn [%s]: pattern source's own nside-512 pixel %d is absent "
            "from the admitted pixel axis -- the admitted set should carry every "
            "catalogued source's pixel" % (region, bad))
    pattern_count = np.bincount(hit, minlength=pix512.size).astype(np.int64)

    return i4_limit_mjy, area_deg2, pattern_count, int(pattern.sum())


# ---------------------------------------------------------------------------
# 4. the shared bin axis, the per-bin density, the Poisson interval
# ---------------------------------------------------------------------------

def shared_edges(log10_limits_by_region, bin_width=LOG10_LIM_BIN_WIDTH_DEX):
    """`(n_bin+1,)`: ONE log10(I4 50% limit, mJy) bin axis for every
    region this build call covers (brief: "bins on a shared axis"), fixed
    -width bins spanning the pooled finite range across every region's
    own admitted pixels."""
    pooled = np.concatenate([np.asarray(x, dtype=np.float64) for x in log10_limits_by_region])
    finite = np.isfinite(pooled)
    if not np.any(finite):
        raise ValueError("population.pahn.shared_edges: no finite I4 50% limit anywhere")
    lo, hi = float(np.min(pooled[finite])), float(np.max(pooled[finite]))
    n_bin = max(1, int(np.ceil((hi - lo) / bin_width)))
    return np.linspace(lo, lo + n_bin * bin_width, n_bin + 1)


def _poisson_interval_68(n):
    """The standard 68% Poisson confidence interval on an observed count
    `n` (Gehrels 1986's own exact Gamma-function construction) -- the
    same formula `population.yso_law._poisson_interval_68` uses, rewritten
    here rather than imported (a population module does not reach into
    another's underscore-named internals; each of this project's own
    Poisson-interval call sites carries its own copy of this four-line
    formula)."""
    alpha = 1.0 - 0.6826894921370859
    lo = 0.0 if n == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * n)
    hi = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (n + 1))
    return lo, hi


#: The PAHC register's own fixed DL21 token (`sed_models.curate.
#: pahc_curate`'s own fiducial choice, read as module constants there:
#: `PAHC_LG_U_TOKEN`, `PAHC_IONIZATION`) and the three published
#: grain-size distributions `load_pahc_excess_shape` reads over: PAHN has
#: no SED register of its own on this branch, so its own SHAPE brightness
#: conversion (I4 observed -> I2/4.5um reference) falls back to the PURE
#: DL21 PAH-excess shape at that SAME fiducial, "the DL21 shape's own
#: ratio computed the same way" the 33 PAHN templates' own median would
#: be (brief).
DL21_SIZE_TOKENS = ("sma", "std", "lrg")


def dl21_i2_i4_log10_ratio(config):
    """`log10(F_I2/F_I4)`, SHAPE's own fallback I2/I4 conversion (module
    docstring, DL21_SIZE_TOKENS): the median, over the three published
    grain-size distributions, of the pure DL21 PAH-excess shape's own
    ratio at the two IRAC bands' own effective wavelength (`definitions.
    BANDS_BY_KEY`) -- `sed_models.curate.pahc_curate.load_pahc_excess_
    shape`'s own `P_hat(lambda)`, normalised so `P_hat(8um) = 1`, POINT-
    SAMPLED (linear interpolation on its native wavelength grid, not
    filter-convolved against the IRAC response curves) at I2's 4.493 um
    and I4's 7.872 um -- a disclosed simplification (this build's own
    report), since a true per-template register entry would be the
    filter-convolved flux (`sed_models.build_pahc`'s own convolution),
    not a single point sample. The DL21 spectra this reads
    (`config.inputs['sed_models']/downloads/PAHspec`) are already on
    disk (`sed_models.curate.pahc_curate.fetch_pahc_excess_templates`'s
    own earlier fetch for PAHC): no network fetch here."""
    from sesnaimpute.sed_models.curate import pahc_curate
    download_dir = f"{config.inputs['sed_models']}/downloads/PAHspec"
    wvl_i2 = definitions.BANDS_BY_KEY["I2"].wvl_um
    wvl_i4 = definitions.BANDS_BY_KEY["I4"].wvl_um
    ratios = []
    for size in DL21_SIZE_TOKENS:
        wave_um, p_hat = pahc_curate.load_pahc_excess_shape(download_dir, size)
        p_i2 = float(np.interp(wvl_i2, wave_um, p_hat))
        p_i4 = float(np.interp(wvl_i4, wave_um, p_hat))
        ratios.append(np.log10(p_i2 / p_i4))
    return float(np.median(ratios))


def pattern_f45_1kpc(config, region, log10_ratio_i2_i4):
    """`(log10_f_ref_i2_1kpc, log10_f_ref_i4_1kpc)`, one row per pattern
    source in `region` (module docstring's pattern): each source's own
    measured I4 flux (mJy), placed at the register's own 1 kpc reference
    convention every other class's `C_THETA` uses (`bmstp.sample_cloud`
    module docstring's `delta = -2 log10(d_r / 1 kpc)`, applied here
    directly to each source's own observed flux at the region's one
    distance `d_r`, since this is the measured SOURCE's own flux, not a
    template placed at an arbitrary distance), and the I2 (4.5um)
    reference flux the SAME brightness conversion `log10_ratio_i2_i4`
    implies (the PAHN templates' own I2/I4 ratio, or its DL21 fallback,
    `dl21_i2_i4_log10_ratio`)."""
    reg = regions_module.REGIONS_BY_NAME[region]
    d_r_pc = float(reg.d_r_pc)
    scale_1kpc = (d_r_pc / 1000.0) ** 2

    cat_path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(cat_path):
        raise FileNotFoundError(
            "population.pahn: no curated catalogue for %r at %s -- run the "
            "'sesnaimpute.catalog.curated' RUNBOOK line first" % (region, cat_path))
    with h5py.File(cat_path, "r") as f:
        fnu = f["FNU_MJY"][:]
        origin = f["ORIGIN_FNU"][:]
    pattern = pattern_mask(fnu, origin)
    f_i4_obs_mjy = fnu[pattern, IDX_I4]
    log10_f_ref_i4_1kpc = np.log10(f_i4_obs_mjy * scale_1kpc)
    log10_f_ref_i2_1kpc = log10_f_ref_i4_1kpc + log10_ratio_i2_i4
    return log10_f_ref_i2_1kpc, log10_f_ref_i4_1kpc


def binned_density(i4_limit_mjy, area_deg2, pattern_count, edges):
    """`(density, poisson_lo, poisson_hi, n_per_bin, area_per_bin)`, each
    `(n_bin,)`: per bin of `edges` (log10 I4 50% limit, mJy), the TOTAL
    pattern count over the TOTAL covered area of every admitted pixel
    whose own limit lands there (module docstring) -- objects per square
    degree -- with the 68% Garwood Poisson interval on the bin's own raw
    count, carried onto the density scale by the same division. A bin no
    admitted pixel's own limit reaches (`area_per_bin == 0`) is NaN, read
    by nothing (`read`'s own knots are the populated bins alone)."""
    finite = np.isfinite(i4_limit_mjy) & np.isfinite(area_deg2) & (area_deg2 > 0)
    log10_lim = np.log10(i4_limit_mjy[finite])
    area = area_deg2[finite]
    count = pattern_count[finite].astype(np.float64)
    n_bin = edges.size - 1
    bin_idx = np.clip(np.searchsorted(edges, log10_lim, side="right") - 1, 0, n_bin - 1)

    n_per_bin_f = np.bincount(bin_idx, weights=count, minlength=n_bin)
    area_per_bin = np.bincount(bin_idx, weights=area, minlength=n_bin)
    n_per_bin = np.round(n_per_bin_f).astype(np.int64)

    density = np.full(n_bin, np.nan, dtype=np.float64)
    poisson_lo = np.full(n_bin, np.nan, dtype=np.float64)
    poisson_hi = np.full(n_bin, np.nan, dtype=np.float64)
    populated = area_per_bin > 0
    density[populated] = n_per_bin_f[populated] / area_per_bin[populated]
    for b in np.flatnonzero(populated):
        lo_n, hi_n = _poisson_interval_68(int(n_per_bin[b]))
        poisson_lo[b] = lo_n / area_per_bin[b]
        poisson_hi[b] = hi_n / area_per_bin[b]
    return density, poisson_lo, poisson_hi, n_per_bin, area_per_bin


# ---------------------------------------------------------------------------
# 5. write, read
# ---------------------------------------------------------------------------

def write_region_table(path, edges, region_names, density, n_per_bin, poisson_lo, poisson_hi,
                        area_per_bin, pattern_count_total):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "region"
        for name, data, extra in (
            ("LOG10_LIM_EDGES", edges.astype(np.float64), {}),
            ("REGION", np.array(region_names, dtype=object),
             dict(dtype=h5py.string_dtype(encoding="utf-8"))),
            ("DENSITY_PAHN", density.astype(np.float64), {}),
            ("N_PER_BIN", n_per_bin.astype(np.int64), {}),
            ("POISSON_LO", poisson_lo.astype(np.float64), {}),
            ("POISSON_HI", poisson_hi.astype(np.float64), {}),
            ("AREA_DEG2_PER_BIN", area_per_bin.astype(np.float64), {}),
            ("PATTERN_COUNT_REGION", np.asarray(pattern_count_total, dtype=np.int64), {}),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)], **extra)


def _region_row(config, region, caller):
    """`(centers, density, n_per_bin)`, this region's own row of the
    stored curve, straight off disk -- the one place that opens the
    product."""
    path = config_module.product_path(config, "population", "pahn", "density", "region")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "population.pahn: no PAHN density product at %r -- run the "
            "'sesnaimpute.population.pahn' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        edges = f["LOG10_LIM_EDGES"][:]
        region_names = [r.decode("utf-8") if isinstance(r, bytes) else r for r in f["REGION"][:]]
        if region not in region_names:
            raise ValueError(
                f"population.pahn.{caller}: region {region!r} is not one of the "
                f"{len(region_names)} regions this density was built on {region_names!r} "
                f"-- rebuild 'population.pahn' with that region included")
        r = region_names.index(region)
        density = f["DENSITY_PAHN"][r]
        n_per_bin = f["N_PER_BIN"][r]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density, n_per_bin


def read(config, region):
    """The shipped `DENSITY_PAHN(limit)` for one region, as a callable,
    linear in log10(I4 50% limit, mJy) between the bins this region
    actually populates and held flat beyond the first/last one
    (`population.pahc_curve.read`'s own held-at-the-ends convention,
    module docstring). `limit` is the I4 50%-completeness flux in mJy
    (not its log10): the callable itself takes `log10(limit)`, the same
    argument convention every other per-source density reader in this
    package uses on its own axis."""
    centers, density, n_per_bin = _region_row(config, region, "read")
    populated = np.isfinite(density) & (n_per_bin > 0)
    if not np.any(populated):
        raise ValueError(f"population.pahn.read: region {region!r} has no populated bin "
                          "at all -- its own pattern count was zero everywhere")
    c = centers[populated]
    d = density[populated]
    return interp1d(c, d, kind="linear", bounds_error=False, fill_value=(float(d[0]), float(d[-1])))


def source_pixel_i4_limit_mjy(config, region, hpx_pix_512):
    """`(n,)` mJy: each `hpx_pix_512` source's own I4 50%-completeness
    limit AT ITS OWN ADMITTED PIXEL (`catalog.limits.Completeness.
    fifty_percent_pixel`) -- the SAME per-pixel limit `region_measurement`
    bins `DENSITY_PAHN` by, not the per-source curve `fifty_percent`
    (`bmstp.density`'s own `D_PAHC`, a different quantity): `bmstp.
    density`'s own `DENSITY_PAHN` reads `read(config, region)` at this
    value, so the curve is read back on the exact axis it was built on.
    `+inf` (never `NaN`) for a source whose own pixel carries no finite
    limit, the convention every other reader in this package uses."""
    reader = limits_module.completeness(config, region)
    pix512 = reader.hpx_pix_512
    order = np.argsort(pix512)
    pix_sorted = pix512[order]
    hpx_pix_512 = np.asarray(hpx_pix_512, dtype=np.int64)
    loc = np.searchsorted(pix_sorted, hpx_pix_512)
    loc = np.minimum(loc, max(pix_sorted.size - 1, 0))
    hit = order[loc]
    found = (pix_sorted.size > 0) & (pix_sorted[loc] == hpx_pix_512)
    if not np.all(found):
        bad = int(hpx_pix_512[~found][0])
        raise ValueError(
            "population.pahn.source_pixel_i4_limit_mjy [%s]: source pixel %d is absent "
            "from the admitted pixel axis" % (region, bad))
    limit = reader.fifty_percent_pixel("I4", hit)
    return np.where(np.isfinite(limit), limit, np.inf)


# ---------------------------------------------------------------------------
# 6. build
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Measures the pattern's own sky density per region over `regions`
    (default: all thirty), writes `population/pahn/density_pahn_region.
    hdf5` with one row per region, bins on one shared axis (module
    docstring)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    st = progress.Stage("population.pahn")

    per_region = {}
    for i_region, region in enumerate(region_names, start=1):
        i4_limit_mjy, area_deg2, pattern_count, n_pattern_total = region_measurement(config, region)
        per_region[region] = (i4_limit_mjy, area_deg2, pattern_count, n_pattern_total)
        st.tick(i_region, len(region_names), "regions (measurement)")

    log10_limits_by_region = [np.log10(v[0][np.isfinite(v[0])]) for v in per_region.values()]
    edges = shared_edges(log10_limits_by_region)
    n_bin = edges.size - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    n_region = len(region_names)
    density = np.full((n_region, n_bin), np.nan, dtype=np.float64)
    n_per_bin = np.zeros((n_region, n_bin), dtype=np.int64)
    poisson_lo = np.full((n_region, n_bin), np.nan, dtype=np.float64)
    poisson_hi = np.full((n_region, n_bin), np.nan, dtype=np.float64)
    area_per_bin = np.zeros((n_region, n_bin), dtype=np.float64)
    pattern_count_total = np.zeros(n_region, dtype=np.int64)

    for i, region in enumerate(region_names):
        i4_limit_mjy, area_deg2, pattern_count, n_pattern_total = per_region[region]
        d_r, lo_r, hi_r, n_r, a_r = binned_density(i4_limit_mjy, area_deg2, pattern_count, edges)
        density[i], n_per_bin[i], poisson_lo[i], poisson_hi[i], area_per_bin[i] = d_r, n_r, lo_r, hi_r, a_r
        pattern_count_total[i] = n_pattern_total

        populated = np.isfinite(d_r) & (n_r > 0)
        median_density = float(np.median(d_r[populated])) if np.any(populated) else float("nan")
        print(f"population.pahn {region}: pattern_count={n_pattern_total} "
              f"n_bin_populated={int(populated.sum())}/{n_bin} "
              f"DENSITY_PAHN median={median_density:.6g} deg^-2", flush=True)
        bin_report = ", ".join(
            f"[{centers[b]:.2f}: N={n_r[b]} rho={d_r[b]:.4g} ({poisson_lo[i,b]:.4g}-{poisson_hi[i,b]:.4g})]"
            for b in np.flatnonzero(populated))
        print(f"population.pahn {region}: bins (log10 I4-lim-mJy center: N, density deg^-2, "
              f"68% Poisson interval) {bin_report}", flush=True)

    out_path = config_module.product_path(config, "population", "pahn", "density", "region")
    write_region_table(out_path, edges, region_names, density, n_per_bin, poisson_lo, poisson_hi,
                        area_per_bin, pattern_count_total)
    st.done(out_path, n_region=n_region, n_bin=n_bin,
            pattern_count_total=int(pattern_count_total.sum()))


if __name__ == "__main__":
    run(build)
