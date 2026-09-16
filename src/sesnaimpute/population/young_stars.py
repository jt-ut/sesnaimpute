"""The model's own expectation of young stars in each STAR anchor pixel
and magnitude bin, so the anchor-weight stage can subtract them from the
observed Gaia/2MASS histograms before fitting `W` (SPEC_PRIORS.md
section 2.1, the "young stars in the anchors" row; IMPLEMENTATION.md
section 6, build order stage 7).

Two pieces, per occupied nside-512 anchor pixel (the same pixels and
edges `population.anchor_tiles.write_histograms` already wrote).

COUNT. The anchor subtraction is an AREA count (SPEC_PRIORS.md section
2.1's "young stars in the anchors" row: "the law count integrated over
the tile"), so the pixel's expectation is the law integrated over the
column MAP inside the pixel, `population.yso.law_area_integral` (section 6.1's
area form: Herschel-covered pixels integrate the HGBS map's own cells at
their native beam scale; elsewhere the Planck sightline column carries
the kernel's sub-beam variance), times the pixel's own solid angle
(`OMEGA_PIX_DEG2`), evaluated at the CLOUD's own share of the column
(SPEC_BMSTP_DRAFT.md section 5.5 "Sky density"): the law is quadratic in
column, so the pixel's parent sightline `cloud_frac = 1 - u(d_front)`
(`population.yso.cloud_interval_pc`, `sightline_lookup`) enters squared,
exactly as `bmstp.density` forms `DENSITY_YSO` at `A_cloud = A_gas *
cloud_frac` -- the young stars this module deducts from the field-star
population are members of the region's cloud, the same class the YSO
prior and the fitter carry, not the whole sightline's worth:

    N_YOUNG_TOTAL(pix) = law_area_integral(pix) * Omega_pix * cloud_frac(pix)**2

`N_law` is convex (squared) in column, so the MEAN of the pixel's own
SOURCE-level law counts (`population.yso.law_count` at each source's own
adopted column and arm) under-runs this integral: SESNA's own sources
avoid the densest gas, so they are a biased-low sample of the pixel's own
column field (Jensen's inequality on the unsampled structure). That
source-sampled quantity is carried alongside, unused past this module, so
the gap is visible:

    N_YOUNG_SOURCE_MEAN(pix) = mean_i[ N_law(A_col_i, arm_i) ] * Omega_pix

MAGNITUDES. Masses are drawn from the Chabrier system IMF over
0.1-1.4 Msun (this module's own closed-form IMF: `chabrier_integral`,
`_imf_density_unnormalized`, `_IMF_NORM`). The installed BHAC15 grid --
the Spitzer, 2MASS AND Gaia tables alike -- stops at 1.4 Msun, so the
IMF's own fraction above it, `IMF_FRAC_ABOVE_1P4` (about 0.1), is
carried as a separate bright count `N_BRIGHT`, disclosed rather than
binned: a >=1.4 Msun, 1 Myr photosphere at a region's own distance is
brighter than every anchor bin's own bright edge (`G_EDGES[0]`,
`KS_EDGES[0]`), so it can never land in a bin -- and no isochrone row
exists above 1.4 Msun to place it at anyway.

Each mass's photosphere is BHAC15's 1 Myr row: Ks from `BHAC15_iso.
2mass` (Vega system, `constants.VEGA_ZERO_POINT_MJY`, this module's own
`abs_mag_grid`), G from this module's own read of `BHAC15_iso.GAIA` (the
same block structure, parsed by `_parse_bhac15_table`; the file's own
header states "Type of calibration used: Vega", the same CALSPEC Alpha
Lyrae standard as the 2MASS table, so its zero point is `constants.
GAIA_G_VEGA_ZP_MJY`, Riello et al. 2021 -- named here though never
multiplied in: every quantity below stays in magnitudes and the zero
point only matters for a flux conversion this module does not do). Both
bands read the region's own distance `d_r_pc` (section 2.1: "at the
region distance"), never a per-source distance.

Extinction is drawn from the pixel's own embedding density: `a = A_pix *
u`, `u` on the parent nside-256 sightline's own full-resolution
`XI_EDGES`/`P_XI` (`population.yso.embedding_and_ridge`), RESTRICTED to
the region's cloud interval `[d_front, d_back]` and renormalised
(`population.yso.restrict_and_renormalize`, `sightline_lookup`) -- a
young star in the anchor's own count is a member of the region's cloud,
the SAME restricted population the YSO prior and `bmstp.sample_cloud`
place, never the whole sightline's foreground-to-background span.
`(G_obs, Ks_obs)` come from `population.anchor_tiles.magnitudes_at_
extinction` -- the ONE function that turns a local column into the two
anchor magnitudes, `field_stars` and `anchor_tiles` unchanged, at each
mass point's own `KG_DRAINE`/`KG_WHITNEY` (`_young_star_kg_by_mass`): a
young star's G magnitude is read straight off the BHAC15 isochrone
(`gaia_abs_mag_grid`), not synthesized against any sps atmosphere, so
there is no per-star [M/H] to match the way `field_stars.match_templates`
matches a TRILEGAL field star -- the isochrone's own T_eff and log g
place each mass jointly on the register's template grid, nearest in
(log10(T_eff), log g), at [M/H] = 0. Gaia's own detection sigmoid (`population.anchor_tiles.gaia_detection_
weight`, reused) weights `N_G_YOUNG`; `N_KS_YOUNG` carries none (2MASS
is treated complete to `KS_CUT_MAG`, the anchor histograms' own
convention).

DISCLOSED (section 2.1, once): the photosphere carries no disk excess.
Real Class I/II young stars are brighter than a bare photosphere by a
few tenths of a magnitude in Ks, so this subtraction is conservative
(undercounts) by about that much, worst in the youngest, most embedded
clusters.

Products, per region, `bms/anchors/young-stars_anchors_hpx512__<Region>
.hdf5`: `HPX_PIX_512`; `N_G_YOUNG` (n_pix, n_G_bins), Gaia-weighted;
`N_KS_YOUNG` (n_pix, n_Ks_bins), the 1 Myr isochrone; `N_YOUNG_TOTAL`
(n_pix), the area-integrated count above; root attr `GRANULE="hpx512"`.
`N_YOUNG_SOURCE_MEAN` (the source-sampled count beside `N_YOUNG_TOTAL`
so the area-integration gap is visible) and `N_BRIGHT` (the disclosed
>1.4 Msun share of `N_YOUNG_TOTAL`, never binned) are computed and
reported at build, not stored.

Algebraic acceptance (reported at build, per pixel, to 1e-9): the 1 Myr
Ks histogram's own bins, plus whatever of the binned (0.1-1.4 Msun)
population falls beyond `KS_EDGES`'s faint edge (`Ks_obs >= 14.3`) or
its bright edge (`Ks_obs < KS_EDGES[0]`, expected negligible -- "bin
edges cover the bright side" -- and reported), plus `N_BRIGHT`, sum to
exactly `N_YOUNG_TOTAL`: the mass quadrature's own weight is forced to
sum to the IMF's exact in-range fraction and the embedding density's own
`u`-cell weights already sum to exactly 1 (`population.yso.embedding_and_
ridge`), so nothing but the three named pieces can carry the total.
"""

import os
import re

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.special import erf

from sesnaimpute import astro_utils
from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.population import anchor_tiles
from sesnaimpute.population import field_stars
from sesnaimpute.population import selection
from sesnaimpute.population import yso

# ---------------------------------------------------------------------
# constants -- every number cited
# ---------------------------------------------------------------------

#: BHAC15_iso.GAIA's own per-row columns (READ_INFO; the same block
#: structure `_parse_bhac15_table` already reads for the Spitzer and
#: 2MASS tables), in the file's own header order.
_GAIA_ROW_COLUMNS = (
    "MASS_MSUN", "TEFF_K", "LOGL", "LOGG", "R_RSUN", "LI_LI0",
    "F33", "F33B", "F41", "F45B", "F47", "F51", "FHA", "F57", "F63B",
    "F67", "F75", "F78", "F82", "F82B", "F89",
    "G_RSV", "G", "G_BP", "G_RP",
)

#: This module's own mass-quadrature resolution (module docstring's IMF
#: x u vectorisation).
N_MASS_QUADRATURE = 200

#: Anchor pixels per joblib block: bounds one block's `(n_pix, n_mass,
#: n_u)` array instead of holding a whole region's at once (rule 8/9).
PIXEL_BLOCK = 32

_GAIA_TABLE_CACHE = {}


# ---------------------------------------------------------------------
# the BHAC15 isochrone tables (Baraffe et al. 2015, A&A 577, A42)
# ---------------------------------------------------------------------

ISOCHRONE_MASS_MAX_MSUN = 1.4
#: Exact tabulated age rows in every BHAC15 file; no age interpolation.
AGE_1MYR_GYR = 0.0010
AGE_3MYR_GYR = 0.0030

#: The AB system's own defining zero point (Oke & Gunn 1983, ApJS 43,
#: 481): `m_AB = -2.5*log10(f_Jy / 3631)`, independent of band -- what
#: `BHAC15_iso.SPITZER`'s own "Type of calibration used: AB" header
#: means its magnitudes are measured against.
AB_ZERO_POINT_MJY = 3631.0e3

_SPITZER_ROW_COLUMNS = (
    "MASS_MSUN", "TEFF_K", "LOGL", "LOGG", "R_RSUN", "LI_LI0",
    "IRAC1", "IRAC2", "IRAC3", "IRAC4", "IRSBLUE", "IRSRED",
    "MIPS24", "MIPS70", "MIPS160",
)
_TWOMASS_ROW_COLUMNS = (
    "MASS_MSUN", "TEFF_K", "LOGL", "LOGG", "R_RSUN", "LI_LI0",
    "MJ", "MH", "MK",
)
#: `BAND_KEYS` -> (isochrone file, that file's own column name); the
#: survey's own eight bands, J/H/Ks Vega off `.2mass`, I1-I4/M1 AB off
#: `.SPITZER` (the two files are calibrated differently and are
#: converted to flux through the zero point that matches each one's own
#: system).
BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)
_BAND_SOURCE = {
    "J": ("2mass", "MJ"), "H": ("2mass", "MH"), "Ks": ("2mass", "MK"),
    "I1": ("spitzer", "IRAC1"), "I2": ("spitzer", "IRAC2"),
    "I3": ("spitzer", "IRAC3"), "I4": ("spitzer", "IRAC4"),
    "M1": ("spitzer", "MIPS24"),
}
_BAND_ZERO_POINT_MJY = {
    key: (constants.VEGA_ZERO_POINT_MJY[key] if src == "2mass" else AB_ZERO_POINT_MJY)
    for key, (src, _col) in _BAND_SOURCE.items()
}

_AGE_HEADER_RE = re.compile(r"^!\s*t\s*\(Gyr\)\s*=\s*([0-9.]+)\s*$")
_TABLE_CACHE = {}


def _parse_bhac15_table(path, row_columns):
    """`(n, 1 + len(row_columns))`: `AGE_GYR` (the block's own `! t
    (Gyr) = ...` header) followed by the file's own per-row columns, one
    row per tabulated mass. Shared block structure between
    `BHAC15_iso.SPITZER` and `BHAC15_iso.2mass` (READ_INFO).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"population.young_stars: no BHAC15 table at {path!r} -- run "
            f"the 'sesnaimpute.sky.download.baraffe2015_bhac15' RUNBOOK line first")
    with open(path) as f:
        text = f.read()
    rows = []
    current_age = None
    for line in text.splitlines():
        header = _AGE_HEADER_RE.match(line.strip())
        if header:
            current_age = float(header.group(1))
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("!") or current_age is None:
            continue
        tokens = stripped.split()
        if len(tokens) != len(row_columns):
            continue
        rows.append([current_age] + [float(t) for t in tokens])
    if not rows:
        raise ValueError(f"population.young_stars: no data rows parsed from {path!r}")
    return np.array(rows, dtype=np.float64)


def _load_tables(config):
    """`{"spitzer": table, "2mass": table}`, cached for the life of the
    process.
    """
    key = config.data_root
    if key not in _TABLE_CACHE:
        base = f"{config.data_root}/sky/download/baraffe2015_bhac15"
        _TABLE_CACHE[key] = {
            "spitzer": _parse_bhac15_table(f"{base}/BHAC15_iso.SPITZER", _SPITZER_ROW_COLUMNS),
            "2mass": _parse_bhac15_table(f"{base}/BHAC15_iso.2mass", _TWOMASS_ROW_COLUMNS),
        }
    return _TABLE_CACHE[key]


def abs_mag_grid(config, age_gyr, mass_grid):
    """`(n_mass, 8)`: absolute magnitude at every mass in `mass_grid`,
    `BAND_KEYS` order, at the isochrone's exact `age_gyr` row, linear
    interpolation in log10(mass) against that file's own tabulated
    masses (module docstring: J/H/Ks Vega off `.2mass`, I1-I4/M1 AB off
    `.SPITZER`).
    """
    tables = _load_tables(config)
    row_columns = {"spitzer": _SPITZER_ROW_COLUMNS, "2mass": _TWOMASS_ROW_COLUMNS}
    log_mass = np.log10(np.asarray(mass_grid, dtype=float))
    out = np.empty((log_mass.size, N_BANDS), dtype=np.float64)
    for key, (src, column) in _BAND_SOURCE.items():
        table = tables[src]
        col_idx = 1 + row_columns[src].index(column)
        mask = np.isclose(table[:, 0], age_gyr, atol=1e-6)
        if not np.any(mask):
            raise ValueError(
                f"population.young_stars: no t={age_gyr:.4f} Gyr block in "
                f"BHAC15_iso.{src}")
        masses = table[mask, 1]
        mags = table[mask, col_idx]
        order = np.argsort(masses)
        out[:, BAND_KEYS.index(key)] = np.interp(
            log_mass, np.log10(masses[order]), mags[order])
    return out


# ---------------------------------------------------------------------
# the population: closed-form Chabrier (2003) system IMF
# ---------------------------------------------------------------------

CHABRIER_MC_MSUN = 0.2
CHABRIER_SIGMA_DEX = 0.55
CHABRIER_SLOPE_ABOVE_1MSUN = 1.35
#: Integration bounds (Pokhrel et al. 2020, ApJ 896, 60, sec. 2.2.3).
IMF_MASS_MIN_MSUN = 0.1
IMF_MASS_MAX_MSUN = 150.0
_IMF_LN10 = np.log(10.0)


def _lognormal_antiderivative(log10_mass):
    """Antiderivative in log10(M) of the lognormal piece below 1 Msun."""
    log_mc = np.log10(CHABRIER_MC_MSUN)
    return (CHABRIER_SIGMA_DEX * np.sqrt(np.pi / 2.0)
            * erf((log10_mass - log_mc) / (CHABRIER_SIGMA_DEX * np.sqrt(2.0))))


def _power_law_value_at_1msun():
    """The lognormal piece's own value at M = 1 Msun: the power-law
    piece's amplitude, by continuity at the join.
    """
    log_mc = np.log10(CHABRIER_MC_MSUN)
    return np.exp(-(log_mc ** 2) / (2.0 * CHABRIER_SIGMA_DEX ** 2))


def _power_law_antiderivative(log10_mass):
    """Antiderivative in log10(M) of the power-law piece above 1 Msun."""
    return (_power_law_value_at_1msun()
            * 10.0 ** (-CHABRIER_SLOPE_ABOVE_1MSUN * log10_mass)
            / (CHABRIER_SLOPE_ABOVE_1MSUN * _IMF_LN10))


def chabrier_integral(mass_lo, mass_hi):
    """Closed-form integral of the Chabrier system IMF's dN/dlog10(M)
    from `mass_lo` to `mass_hi` (Msun, `mass_lo <= mass_hi`), split at
    the lognormal/power-law join (1 Msun) as needed. Vectorised.
    """
    mass_lo = np.asarray(mass_lo, dtype=np.float64)
    mass_hi = np.asarray(mass_hi, dtype=np.float64)
    log_lo = np.log10(mass_lo)
    log_hi = np.log10(mass_hi)

    below_lo = np.minimum(log_hi, 0.0)
    lognormal_part = np.where(
        log_lo < 0.0,
        _lognormal_antiderivative(below_lo) - _lognormal_antiderivative(log_lo),
        0.0)

    above_lo = np.maximum(log_lo, 0.0)
    power_law_part = np.where(
        log_hi > 0.0,
        _power_law_antiderivative(above_lo) - _power_law_antiderivative(log_hi),
        0.0)
    return lognormal_part + power_law_part


#: The full normalising integral, 0.1-150 Msun.
_IMF_NORM = chabrier_integral(IMF_MASS_MIN_MSUN, IMF_MASS_MAX_MSUN)


def imf_fraction_above(m_lim):
    """`f_IMF(M > m_lim)`, the Chabrier system IMF's own mass fraction
    above `m_lim` (Msun), normalised over `[IMF_MASS_MIN_MSUN,
    IMF_MASS_MAX_MSUN]`. Vectorised.
    """
    m_lim = np.clip(np.asarray(m_lim, dtype=np.float64),
                     IMF_MASS_MIN_MSUN, IMF_MASS_MAX_MSUN)
    return chabrier_integral(m_lim, IMF_MASS_MAX_MSUN) / _IMF_NORM


#: The IMF mass fraction beyond the installed isochrone's own ceiling: a
#: 1 Myr star up there is treated as detected outright (module
#: docstring). About 0.1 (reported in the timed run, never tuned).
IMF_FRAC_ABOVE_1P4 = float(imf_fraction_above(ISOCHRONE_MASS_MAX_MSUN))

#: The Chabrier IMF's own mass fraction inside the installed isochrone's
#: range -- partition of the whole IMF, exact by construction.
MASS_FRAC_IN_RANGE = 1.0 - IMF_FRAC_ABOVE_1P4


def _imf_density_unnormalized(mass):
    """`xi(log10 M) = dN/dlog10(M)`, unnormalised (the same lognormal /
    power-law form `chabrier_integral` integrates, continuous at 1 Msun
    by construction): the mass-quadrature weight before division by
    `_IMF_NORM`.
    """
    mass = np.asarray(mass, dtype=np.float64)
    log10_mass = np.log10(mass)
    log_mc = np.log10(CHABRIER_MC_MSUN)
    lognormal = np.exp(-(log10_mass - log_mc) ** 2 / (2.0 * CHABRIER_SIGMA_DEX ** 2))
    power_law = _power_law_value_at_1msun() * mass ** (-CHABRIER_SLOPE_ABOVE_1MSUN)
    return np.where(mass <= 1.0, lognormal, power_law)


# ---------------------------------------------------------------------
# the Gaia photosphere -- this module's own read of BHAC15_iso.GAIA
# ---------------------------------------------------------------------

def _load_gaia_table(config):
    key = config.data_root
    if key not in _GAIA_TABLE_CACHE:
        path = f"{config.data_root}/sky/download/baraffe2015_bhac15/BHAC15_iso.GAIA"
        _GAIA_TABLE_CACHE[key] = _parse_bhac15_table(path, _GAIA_ROW_COLUMNS)
    return _GAIA_TABLE_CACHE[key]


def gaia_abs_mag_grid(config, age_gyr, mass_grid):
    """Abs G at the isochrone's exact `age_gyr` row, linear
    interpolation in log10(mass) against the table's own tabulated
    masses -- the same convention `abs_mag_grid` uses for the other
    eight bands (module docstring: Vega system).
    """
    table = _load_gaia_table(config)
    mask = np.isclose(table[:, 0], age_gyr, atol=1e-6)
    if not np.any(mask):
        raise ValueError(
            f"prior.young_stars: no t={age_gyr:.4f} Gyr block in BHAC15_iso.GAIA")
    col_idx = 1 + _GAIA_ROW_COLUMNS.index("G")
    masses = table[mask, 1]
    g_mag = table[mask, col_idx]
    order = np.argsort(masses)
    return np.interp(np.log10(np.asarray(mass_grid, dtype=float)),
                      np.log10(masses[order]), g_mag[order])


def _young_star_kg_by_mass(config, mass_grid):
    """`(kg_diffuse, kg_dense)`, each `(n_mass,)`: the sps atmosphere
    register's `KG_DRAINE`/`KG_WHITNEY` (Danielski et al. 2018's
    `A_G/A_V` per template, `sed_models/registers/sps_register.hdf5`,
    `field_stars.load_atmosphere_grid`), one per mass point. A young
    star's G magnitude is placed straight off the BHAC15 1 Myr isochrone
    (`gaia_abs_mag_grid`), not synthesized against any sps atmosphere,
    so it carries no per-star [M/H]; its T_eff and log g (both
    tabulated in the same isochrone block, each interpolated in
    log10(mass) exactly as `gaia_abs_mag_grid` interpolates G) place
    each mass point on the register's template grid through
    `field_stars.match_templates`, the same nearest-neighbour match a
    TRILEGAL field star's atmosphere uses, jointly in (log10(T_eff),
    log g), at [M/H] = 0.
    """
    table = _load_gaia_table(config)
    mask = np.isclose(table[:, 0], AGE_1MYR_GYR, atol=1e-6)
    if not np.any(mask):
        raise ValueError(
            f"prior.young_stars: no t={AGE_1MYR_GYR:.4f} Gyr block in BHAC15_iso.GAIA")
    masses = table[mask, 1]
    teff = table[mask, 1 + _GAIA_ROW_COLUMNS.index("TEFF_K")]
    logg = table[mask, 1 + _GAIA_ROW_COLUMNS.index("LOGG")]
    order = np.argsort(masses)
    log10_mass = np.log10(np.asarray(mass_grid, dtype=np.float64))
    teff_at_mass = 10.0 ** np.interp(
        log10_mass, np.log10(masses[order]), np.log10(teff[order]))
    logg_at_mass = np.interp(log10_mass, np.log10(masses[order]), logg[order])

    register_path = f"{config.data_root}/sed_models/registers/sps_register.hdf5"
    atmosphere = field_stars.load_atmosphere_grid(register_path)
    idx, _ = field_stars.match_templates(
        teff_at_mass, logg_at_mass, np.zeros_like(teff_at_mass), atmosphere["grid"])
    return atmosphere["kg_diffuse"][idx], atmosphere["kg_dense"][idx]


# ---------------------------------------------------------------------
# the population: this module's own quadrature resolution
# ---------------------------------------------------------------------

def mass_grid_and_weight(n=N_MASS_QUADRATURE):
    """`(mass_grid, weight)`: `n` masses log-spaced over 0.1-1.4 Msun,
    and each point's own trapezoidal probability mass in log10(mass) --
    the Chabrier density above (`_imf_density_unnormalized`/`_IMF_NORM`),
    at this module's own point count. `weight.sum()` is forced to
    exactly `MASS_FRAC_IN_RANGE` so the module docstring's algebraic
    acceptance is exact to float64, not left to quadrature error.
    """
    mass_grid = np.geomspace(IMF_MASS_MIN_MSUN, ISOCHRONE_MASS_MAX_MSUN, n)
    density = _imf_density_unnormalized(mass_grid) / _IMF_NORM
    log10_mass = np.log10(mass_grid)
    dw = np.empty_like(log10_mass)
    dw[0] = 0.5 * (log10_mass[1] - log10_mass[0])
    dw[-1] = 0.5 * (log10_mass[-1] - log10_mass[-2])
    dw[1:-1] = 0.5 * (log10_mass[2:] - log10_mass[:-2])
    weight = density * dw
    weight *= MASS_FRAC_IN_RANGE / weight.sum()
    return mass_grid, weight


# ---------------------------------------------------------------------
# per-pixel inputs: the law count and the parent sightline's placement
# ---------------------------------------------------------------------

def law_count_per_pixel(config, region, pixels):
    """`N_law` averaged per pixel (deg^-2), aligned to `pixels` -- the
    SOURCE-sampled mean of each pixel's own sources' `population.yso.law_count`
    (module docstring's `N_YOUNG_SOURCE_MEAN`), kept only as the reported
    comparison against the area-integrated `population.yso.law_area_integral`
    this module now uses for `N_YOUNG_TOTAL`.
    """
    rs = access.region_slice(config, region)
    a_col, _, provenance = yso._adopted_columns(config, region)
    src_pix = rs["hpx_pix_512"]
    law_src = yso.law_count(config, region, a_col, provenance)

    uniq_pix, inverse = np.unique(src_pix, return_inverse=True)
    if not np.array_equal(uniq_pix, pixels):
        raise ValueError(
            "prior.young_stars: %r's granule-map pixel set disagrees with "
            "bms.anchors.histograms's own pixel set -- rerun the "
            "'prior.anchor_tiles' RUNBOOK line for this region" % region)
    n_src = np.bincount(inverse, minlength=pixels.size).astype(np.float64)
    sum_law = np.bincount(inverse, weights=law_src, minlength=pixels.size)
    return sum_law / n_src


def sightline_lookup(config, region, pixels, d_front, d_back):
    """Per anchor pixel, its own parent nside-256 sightline's embedding
    density AT THE PROFILE'S OWN FULL RESOLUTION (`population.yso.
    embedding_and_ridge`), restricted to the region's cloud interval
    `[d_front, d_back]` and renormalised (`population.yso.restrict_and_
    renormalize`) -- never the stored 32-cell coarsened product (which
    carries no distance per cell to restrict by) and never the whole
    sightline: a deducted young star sits on the SAME cloud population
    `bmstp.sample_cloud` places (sec. 5.5), recomputed in memory at the
    same full resolution `bmstp` uses rather than read off the coarsened
    disk form. The parent is the HEALPix NESTED ancestor, `pixels >> 2`.

    Returns `(xi_edges, mass, cloud_frac, removed_frac_report)`: `xi_edges`
    (n_pix, n_cell+1) and `mass` (n_pix, n_cell, summing to 1 per row)
    are the restricted, renormalised placement; `cloud_frac` (n_pix,) is
    `1 - u(d_front)`, the cloud's own share of each pixel's column (sec.
    5.5 "Sky density", the SAME quantity `bmstp.density._cloud_column_
    fraction` forms, duplicated here for the same import-direction
    reason `cloud_interval_pc` is); `removed_frac_report` is the median
    fraction of pre-restriction mass the interval removed, across the
    region's occupied sightlines.
    """
    profile = yso._load_profile_arrays(config, region)
    embed = yso.embedding_and_ridge(profile)
    dist_pc = profile["dist_pc"]
    n_d = dist_pc.size
    sl_pix = profile["hpx_pix_256"]
    n_sl = sl_pix.size
    d_edges = np.empty((n_sl, n_d + 1), dtype=np.float64)
    d_edges[:, :n_d] = dist_pc[None, :]
    d_edges[:, n_d] = dist_pc[-1] + 2.0 * profile["tail_efold_pc"]
    xi_edges = embed["xi_edges"]
    p_u = embed["p_u"]

    u_lo, u_hi = xi_edges[:, :-1], xi_edges[:, 1:]
    d_lo, d_hi = d_edges[:, :-1], d_edges[:, 1:]
    mass, inside_frac, removed_frac = yso.restrict_and_renormalize(
        p_u, u_lo, u_hi, d_lo, d_hi, d_front, d_back)
    mass_restricted = (mass * inside_frac) / np.maximum(1.0 - removed_frac, 1e-300)[:, None]

    # the cloud's own share of the column, `1 - u(d_front)` (sec. 5.5
    # "Sky density"): the map's own DIST_PC knots are common to every
    # sightline, so the bracketing edge is found once, not per row.
    j = int(np.clip(np.searchsorted(dist_pc, d_front), 1, n_d - 1))
    d0, d1 = dist_pc[j - 1], dist_pc[j]
    frac = (d_front - d0) / (d1 - d0) if d1 > d0 else 0.0
    u_front = xi_edges[:, j - 1] + frac * (xi_edges[:, j] - xi_edges[:, j - 1])
    cloud_frac = 1.0 - u_front

    order = np.argsort(sl_pix)
    sl_pix_sorted = sl_pix[order]
    parent256 = pixels >> 2
    loc = np.searchsorted(sl_pix_sorted, parent256)
    capped = np.minimum(loc, max(sl_pix_sorted.size - 1, 0))
    matched = (sl_pix_sorted.size > 0) & (sl_pix_sorted[capped] == parent256)
    if not np.all(matched):
        raise ValueError(
            "prior.young_stars: %r has an anchor pixel whose parent "
            "nside-256 sightline is absent from its own sky.derived.profile "
            "product" % region)
    idx = order[capped]
    return xi_edges[idx], mass_restricted[idx], cloud_frac[idx], float(np.median(removed_frac))


# ---------------------------------------------------------------------
# the vectorised block: IMF x u-cells for one group of pixels
# ---------------------------------------------------------------------

def _weighted_hist(values, weights, edges):
    """`(counts, faint_overflow, bright_overflow)`: `counts` (n_blk,
    n_bins) is the weighted histogram of `values`/`weights` (n_blk,
    n_mass, n_u) into `edges`, one `bincount` over a flattened
    pixel*bin index (no Python loop below the block, rule 8);
    `faint_overflow`/`bright_overflow` (n_blk,) are the weight landing
    at or past `edges[-1]` / below `edges[0]`, so `counts.sum(axis=1) +
    faint_overflow + bright_overflow` recovers the block's own total
    weight exactly (the module docstring's acceptance identity).
    """
    n_blk = values.shape[0]
    n_bins = edges.size - 1
    pix_idx = np.broadcast_to(np.arange(n_blk)[:, None, None], values.shape)
    bin_idx = np.searchsorted(edges, values, side="right") - 1
    in_range = (values >= edges[0]) & (values < edges[-1])
    counts = np.bincount(pix_idx[in_range] * n_bins + bin_idx[in_range],
                          weights=weights[in_range],
                          minlength=n_blk * n_bins).reshape(n_blk, n_bins)
    faint_mask = values >= edges[-1]
    bright_mask = values < edges[0]
    faint_overflow = np.bincount(pix_idx[faint_mask], weights=weights[faint_mask],
                                  minlength=n_blk)
    bright_overflow = np.bincount(pix_idx[bright_mask], weights=weights[bright_mask],
                                   minlength=n_blk)
    return counts, faint_overflow, bright_overflow


def _pixel_block(config, a_pix_blk, xi_edges_blk, mass_blk, n_young_blk,
                  mass_grid, mass_weight, ks_abs_1myr, g_abs_1myr,
                  mu, g_edges, ks_edges, k_g_diffuse, k_g_dense, r_diffuse, r_dense):
    """One block's `(N_G_YOUNG, N_KS_YOUNG)` and the 1 Myr Ks acceptance
    pieces, vectorised over every mass and every `u`-cell of every pixel
    in the block at once. `k_g_diffuse`/`k_g_dense` are `(n_mass,)`,
    each mass point's own register-template coefficient
    (`_young_star_kg_by_mass`).
    """
    u_mid = 0.5 * (xi_edges_blk[:, :-1] + xi_edges_blk[:, 1:])      # (n_blk, n_u)
    u_mass = mass_blk                                             # sums to 1 per row (cloud-restricted)

    a_rep = a_pix_blk[:, None] * u_mid                            # (n_blk, n_u)

    weight = (n_young_blk[:, None, None] * mass_weight[None, :, None]
              * u_mass[:, None, :])                                # (n_blk, n_mass, n_u)

    ks_app_1myr = ks_abs_1myr + mu
    g_app_1myr = g_abs_1myr + mu

    # the ONE Gaia dimming law (module docstring): `anchor_tiles.
    # magnitudes_at_extinction`, the same function `field_stars`'s own
    # per-matched-template anchors use, at each mass point's own
    # template coefficient (`k_g_diffuse`/`k_g_dense`, `(n_mass,)`).
    g_obs, ks_obs_1myr = anchor_tiles.magnitudes_at_extinction(
        a_rep[:, None, :], g_app_1myr[None, :, None], ks_app_1myr[None, :, None],
        k_g_diffuse[None, :, None], k_g_dense[None, :, None], r_diffuse, r_dense)

    p_g = anchor_tiles.gaia_detection_weight(g_obs)
    n_g, _, _ = _weighted_hist(g_obs, weight * p_g, g_edges)
    n_ks, ks_faint, ks_bright = _weighted_hist(ks_obs_1myr, weight, ks_edges)

    return n_g, n_ks, ks_faint, ks_bright


# ---------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------

def build_region(config, region):
    """Computes and writes one region's young-star anchor-pixel product
    (module docstring)."""
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    mu = float(astro_utils.distance_modulus(d_r_pc))

    hist_path = config_module.product_path(config, "population", "anchors",
                                            "histograms", "hpx512", region=region)
    if not os.path.exists(hist_path):
        raise FileNotFoundError(
            "prior.young_stars: anchor histograms missing for region %r at "
            "%s -- run the 'prior.anchor_tiles' RUNBOOK line first" % (region, hist_path))
    with h5py.File(hist_path, "r") as f:
        pixels = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        a_pix = np.asarray(f["A_PIX_K"][:], dtype=np.float64)
        omega_pix_deg2 = np.asarray(f["OMEGA_PIX_DEG2"][:], dtype=np.float64)
        g_edges = np.asarray(f["G_EDGES"][:], dtype=np.float64)
        ks_edges = np.asarray(f["KS_EDGES"][:], dtype=np.float64)

    d_front, d_back = yso.cloud_interval_pc(config, region)
    n_young_source_mean = law_count_per_pixel(config, region, pixels) * omega_pix_deg2
    xi_edges, mass, cloud_frac, removed_frac_report = sightline_lookup(
        config, region, pixels, d_front, d_back)
    # the model's own young-star density (sec. 5.5 "Sky density"): the
    # law is quadratic in column, so evaluating it at the CLOUD's own
    # share of the column, `A_cloud = A_gas * cloud_frac`, is the whole-
    # column integral times `cloud_frac**2` -- exactly how `bmstp.
    # density` forms `DENSITY_YSO` (`a_cloud = a_col_gas * cloud_frac`,
    # then squared inside the law), imported rather than re-derived: the
    # class this module deducts is the SAME young-star population by
    # construction, foreground-only deducted, never the whole sightline.
    n_young_total_whole_column = yso.law_area_integral(config, region, pixels) * omega_pix_deg2
    n_young_total = n_young_total_whole_column * cloud_frac ** 2

    mass_grid, mass_weight = mass_grid_and_weight()
    ks_abs_1myr = abs_mag_grid(
        config, AGE_1MYR_GYR, mass_grid)[:, BAND_KEYS.index("Ks")]
    g_abs_1myr = gaia_abs_mag_grid(config, AGE_1MYR_GYR, mass_grid)

    # the one Gaia dimming law (module docstring): each mass point's own
    # nearest-template register coefficient, the SAME `r_diffuse`/
    # `r_dense` (A_K/A_V) construction `anchor_tiles.build` uses.
    k_g_diffuse, k_g_dense = _young_star_kg_by_mass(config, mass_grid)
    r_diffuse = float(selection.ak_per_av(config, 0.0))
    r_dense = float(selection.ak_per_av(config, 1.0))

    n_pix = pixels.size
    starts = list(range(0, n_pix, PIXEL_BLOCK))
    blocks = Parallel(n_jobs=config.n_jobs)(
        delayed(_pixel_block)(
            config, a_pix[s:s + PIXEL_BLOCK], xi_edges[s:s + PIXEL_BLOCK],
            mass[s:s + PIXEL_BLOCK], n_young_total[s:s + PIXEL_BLOCK],
            mass_grid, mass_weight, ks_abs_1myr, g_abs_1myr,
            mu, g_edges, ks_edges, k_g_diffuse, k_g_dense, r_diffuse, r_dense)
        for s in starts)

    n_g_young = np.concatenate([b[0] for b in blocks], axis=0) if blocks else \
        np.empty((0, g_edges.size - 1))
    n_ks_young = np.concatenate([b[1] for b in blocks], axis=0) if blocks else \
        np.empty((0, ks_edges.size - 1))
    ks_faint = np.concatenate([b[2] for b in blocks]) if blocks else np.empty(0)
    ks_bright = np.concatenate([b[3] for b in blocks]) if blocks else np.empty(0)

    n_bright = n_young_total * IMF_FRAC_ABOVE_1P4

    # algebraic acceptance (module docstring): every named piece sums
    # back to N_YOUNG_TOTAL, per pixel, to float64 precision.
    closure = n_ks_young.sum(axis=1) + ks_faint + ks_bright + n_bright
    max_rel_dev = float(np.max(np.abs(closure - n_young_total)
                                / np.maximum(n_young_total, 1e-300)))
    if max_rel_dev >= 1e-9:
        raise ValueError(
            "prior.young_stars: %r's N_KS_YOUNG + overflow + N_BRIGHT closure "
            "against N_YOUNG_TOTAL misses 1e-9 (worst %.3e)" % (region, max_rel_dev))

    return dict(
        pixels=pixels, a_pix=a_pix, n_g_young=n_g_young, n_ks_young=n_ks_young,
        n_young_total=n_young_total,
        n_young_total_whole_column=n_young_total_whole_column,
        n_young_source_mean=n_young_source_mean,
        n_bright=n_bright, g_edges=g_edges, ks_edges=ks_edges,
        ks_faint_overflow=ks_faint, ks_bright_overflow=ks_bright,
        max_rel_dev=max_rel_dev, removed_frac_report=removed_frac_report,
    )


def _write_product(config, region, result):
    path = config_module.product_path(config, "population", "anchors", "young-stars",
                                       "hpx512", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.create_dataset("HPX_PIX_512", data=result["pixels"].astype(np.int64))
        f.create_dataset("N_G_YOUNG", data=result["n_g_young"].astype(np.float32))
        f.create_dataset("N_KS_YOUNG", data=result["n_ks_young"].astype(np.float32))
        f.create_dataset("N_YOUNG_TOTAL", data=result["n_young_total"].astype(np.float64))
    return path


def build(config, regions=None):
    """Writes, per region (default: all thirty), `bms/anchors/young-
    stars_anchors_hpx512__<Region>.hdf5` (module docstring). Prints,
    per region, the total expected young-star count, the observed 2MASS
    count (`Ks < 14.3`) in the same pixels and their ratio, and the
    algebraic acceptance's worst-pixel relative deviation.
    """
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in names:
        with progress.Stage("prior.young_stars", region) as st:
            result = build_region(config, region)
            path = _write_product(config, region, result)

            hist_path = config_module.product_path(config, "population", "anchors",
                                                    "histograms", "hpx512", region=region)
            with h5py.File(hist_path, "r") as f:
                n_ks_obs = np.asarray(f["N_KS_OBS"][:], dtype=np.float64).sum(axis=1)

            total_young = float(result["n_young_total"].sum())
            total_young_whole_column = float(result["n_young_total_whole_column"].sum())
            total_source_mean = float(result["n_young_source_mean"].sum())
            total_obs = float(n_ks_obs.sum())
            ratio = total_young / total_obs if total_obs > 0 else float("nan")
            area_over_source = (total_young / total_source_mean
                                 if total_source_mean > 0 else float("nan"))
            share = np.where(n_ks_obs > 0, result["n_young_total"] / n_ks_obs, 0.0)
            i_worst = int(np.argmax(share))
            st.done(path, n_young_total=total_young, ratio_to_2mass=ratio)
        print(
            "prior.young_stars: %s N_YOUNG_TOTAL=%.2f (whole-column, pre-cloud-"
            "restriction, was %.2f, x%.4f) N_YOUNG_SOURCE_MEAN=%.2f "
            "(area/source-mean=%.3fx) 2MASS(Ks<14.3)=%.2f ratio=%.4f "
            "worst-pixel young/2MASS share=%.4f at A_PIX_K=%.3f "
            "median cloud-interval-restriction removed_frac=%.4f "
            "max_rel_dev=%.3e -> %s"
            % (region, total_young, total_young_whole_column,
               total_young / total_young_whole_column if total_young_whole_column > 0 else float("nan"),
               total_source_mean, area_over_source,
               total_obs, ratio, share[i_worst],
               result["a_pix"][i_worst], result["removed_frac_report"],
               result["max_rel_dev"], path))


if __name__ == "__main__":
    run(build)
