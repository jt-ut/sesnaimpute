"""The YSO mass-based selection (SPEC_PRIORS.md section 6.2): the inner
integral of `epsilon_YSO`, per region and depth group, on the shared
extinction grid --

    g[k, j] = Integral dM f_IMF(M) * 1[ photosphere(M) at d_r,
              dimmed by A_GRID[j], clears >= 2 of 8 at group k's limits ]

for the 1 Myr and 3 Myr BHAC15 (Baraffe et al. 2015, A&A 577, A42)
isochrones. `epsilon_YSO(s)` itself (the extinction integral over a
sightline's own embedding density, `Integral da p(a | A_s) * g(a)`) is
built elsewhere, per sightline; this module ships only `g`, the
population- and photosphere-side factor, tabulated once per region so
that stage is a lookup, not a per-source isochrone evaluation.

THE POPULATION. The Chabrier (2003, PASP 115, 763) system IMF: a
lognormal below 1 Msun (M_c = 0.2 Msun, sigma = 0.55 dex), a power law
dN/dlog10(M) ~ M^-1.35 above it, continuous at the join, normalised over
0.1-150 Msun (SPEC_PRIORS.md 6.2). The installed BHAC15 grid covers only
0.01-1.4 Msun, so a star above 1.4 Msun is treated as detected outright
(a 1 Myr photosphere above 1.4 Msun clears any SESNA limit inside 2 kpc)
and `g` is the mass integral over [0.1, 1.4] plus the IMF's own mass
fraction above 1.4 Msun, `IMF_FRAC_ABOVE_1P4` (about 0.1, computed and
reported below, never tuned). Below 0.1 Msun the IMF carries no weight
(the integration floor), so nothing there needs a photosphere at all.
The closed-form Chabrier integral (`chabrier_integral`, an error-function
antiderivative below 1 Msun, a power law above) and the isochrone loader
below serve the survey's own two-of-eight test on all eight bands; the one-band
form (the literature check below) is the same machinery on 8 um alone.

THE PHOTOSPHERE. J, H, Ks come from `BHAC15_iso.2mass`, absolute
magnitudes in the Vega system (its own header: "apparent Vega mag is
assumed NULL in each bandpass"); I1-I4 and M1 (MIPS 24 um) come from
`BHAC15_iso.SPITZER`, absolute magnitudes in the AB system (its own
header: "Type of calibration used: AB") -- the two files are calibrated
differently and are converted to flux through the zero point that
matches each one's own system: `constants.VEGA_ZERO_POINT_MJY` for the
2MASS bands, this module's own `AB_ZERO_POINT_MJY` for the Spitzer
bands. MIPS 24um is tabulated in the SPITZER file and used like every
other band; a bare 1 Myr photosphere there is far below any SESNA M1
limit, so band M1 in practice never clears (measured, not assumed;
reported in the timed run). Both 1 Myr and 3 Myr are exact tabulated age
rows (0.0010 and 0.0030 Gyr); no age interpolation. A star's absolute
magnitude at an off-grid mass is linearly interpolated in log10(mass)
against the isochrone's own tabulated masses.

THE TEST. Apparent flux at the region's own `d_r_pc` (`constants.
REGIONS`), dimmed by `10**(-0.4 * a * kappa_i(a))` with the blended
diffuse/dense law (`prior.selection.kappa_hybrid`), compared to a depth
group's own 8-band limit vector: the region's reference limit
(`depth_groups.DepthGroups.ref_log10_flim`) with the five Spitzer bands
shifted by the group's own `Delta` centre (`prior.depth_groups`,
`prior.selection.BANDS_DEPTH`). `>= 2` of 8 clearing is a catalogued
detection (`prior.selection.MIN_BANDS`).

THE LITERATURE CHECK (reported, not shipped; SPEC_PRIORS.md 6.2, 6.5 item
4). Gutermuth et al. (2009) sec. 7.1's one-band form: the mass whose I4
(8.0 um) photospheric magnitude, undimmed, at a region's own distance
equals the region's own reference I4 limit, and the IMF fraction above
it. Written to the 30-row summary alongside `D_R_PC` and `g` at zero
extinction for both ages at the region's median depth group.
"""

import os
import re

import h5py
import numpy as np
from scipy.special import erf

from sesnaimpute import astro_utils
from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.build import run
from sesnaimpute.prior import column_grid as column_grid_module
from sesnaimpute.prior import depth_groups as depth_groups_module
from sesnaimpute.prior import selection

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: Index, in `BAND_KEYS` order, of the five bands a depth group's own
#: `Delta` shifts (`prior.selection.BANDS_DEPTH`); the three 2MASS bands
#: keep the region's reference limit unchanged (SPEC_PRIORS.md 1.3).
_DEPTH_IDX = np.array([BAND_KEYS.index(b) for b in selection.BANDS_DEPTH])

# --- the population: Chabrier 2003 system IMF (SPEC_PRIORS.md 6.2) ---
CHABRIER_MC_MSUN = 0.2
CHABRIER_SIGMA_DEX = 0.55
CHABRIER_SLOPE_ABOVE_1MSUN = 1.35
#: Integration bounds (Pokhrel et al. 2020, ApJ 896, 60, sec. 2.2.3).
IMF_MASS_MIN_MSUN = 0.1
IMF_MASS_MAX_MSUN = 150.0
_LN10 = np.log(10.0)

# --- the photosphere: BHAC15, 1 and 3 Myr (SPEC_PRIORS.md 6.2, 6.5) ---
ISOCHRONE_MASS_MIN_MSUN = 0.01
ISOCHRONE_MASS_MAX_MSUN = 1.4
#: Exact tabulated age rows in both BHAC15 files; no age interpolation.
AGE_1MYR_GYR = 0.0010
AGE_3MYR_GYR = 0.0030
#: The mass quadrature: a fine log-M grid over the IMF's own overlap
#: with the isochrone, 0.1-1.4 Msun (SPEC_PRIORS.md 6.2).
N_MASS_QUADRATURE = 500
#: A finer grid for the one-band literature check's own mass inversion
#: (a single evaluation per region, not a per-node tabulation).
_N_MASS_INVERT = 4000

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
#: `BAND_KEYS` -> (isochrone file, that file's own column name).
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


# ---------------------------------------------------------------------
# the BHAC15 tables
# ---------------------------------------------------------------------

def _parse_bhac15_table(path, row_columns):
    """`(n, 1 + len(row_columns))`: `AGE_GYR` (the block's own `! t
    (Gyr) = ...` header) followed by the file's own per-row columns, one
    row per tabulated mass. Shared block structure between
    `BHAC15_iso.SPITZER` and `BHAC15_iso.2mass` (READ_INFO).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"prior.yso_selection: no BHAC15 table at {path!r} -- run the "
            f"'sesnaimpute.sky.download.baraffe2015_bhac15' RUNBOOK line first")
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
        raise ValueError(f"prior.yso_selection: no data rows parsed from {path!r}")
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
                f"prior.yso_selection: no t={age_gyr:.4f} Gyr block in "
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
            / (CHABRIER_SLOPE_ABOVE_1MSUN * _LN10))


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


def mass_quadrature():
    """`(mass_grid, imf_weight)`: `N_MASS_QUADRATURE` points log-spaced
    over `[IMF_MASS_MIN_MSUN, ISOCHRONE_MASS_MAX_MSUN]`, and the
    IMF-normalised density at each -- `trapz(imf_weight, x=log10(mass_
    grid))` is `chabrier_integral(0.1, 1.4) / _IMF_NORM` (SPEC_PRIORS.md
    6.2's mass integral, quadrature form).
    """
    mass_grid = np.geomspace(IMF_MASS_MIN_MSUN, ISOCHRONE_MASS_MAX_MSUN, N_MASS_QUADRATURE)
    return mass_grid, _imf_density_unnormalized(mass_grid) / _IMF_NORM


# ---------------------------------------------------------------------
# the inner selection integral, per region
# ---------------------------------------------------------------------

def group_limit_vectors(depth_groups):
    """`(K, 8)`: each depth group's own 8-band log10 limit vector, in
    `BAND_KEYS` order -- the region's reference limit
    (`depth_groups.ref_log10_flim`) with the five Spitzer bands shifted
    by the group's own `Delta` centre (SPEC_PRIORS.md 1.3).
    """
    n_groups = depth_groups.group_centres.shape[0]
    limits = np.tile(depth_groups.ref_log10_flim[None, :], (n_groups, 1))
    limits[:, _DEPTH_IDX] = (
        depth_groups.ref_log10_flim[_DEPTH_IDX][None, :] + depth_groups.group_centres)
    return limits


def selection_table(config, age_gyr, d_r_pc, a_grid, group_limits, mass_grid, imf_weight):
    """`(K, n_a)`: `g[k, j]`, the IMF-weighted two-of-eight selection at
    `age_gyr`, every depth group and every extinction node -- the module
    docstring's inner integral, fully vectorised (no loop over sources,
    masses, groups or nodes).
    """
    abs_mag = abs_mag_grid(config, age_gyr, mass_grid)  # (n_mass, 8)
    mu = astro_utils.distance_modulus(d_r_pc)
    zero_points = np.array([_BAND_ZERO_POINT_MJY[k] for k in BAND_KEYS])
    apparent_mag = abs_mag + mu  # undimmed, (n_mass, 8)
    log10_f0 = np.log10(zero_points)[None, :] - 0.4 * apparent_mag  # (n_mass, 8)

    with np.errstate(divide="ignore"):
        w = selection.law_dense_weight(a_grid)  # a_grid[0] == 0 -> log(0), clipped to weight 0
    kappa_at_a = selection.kappa_hybrid(config, w)  # (n_a, 8)
    dimming = 0.4 * np.asarray(a_grid, dtype=float)[:, None] * kappa_at_a  # (n_a, 8)
    log10_f = log10_f0[:, None, :] - dimming[None, :, :]  # (n_mass, n_a, 8)

    n_clear = np.sum(
        log10_f[None, :, :, :] >= group_limits[:, None, None, :], axis=-1)  # (K, n_mass, n_a)
    cleared = (n_clear >= selection.MIN_BANDS).astype(np.float64)

    log10_mass = np.log10(mass_grid)
    inner = np.trapz(cleared * imf_weight[None, :, None], x=log10_mass, axis=1)  # (K, n_a)
    return inner + IMF_FRAC_ABOVE_1P4


# ---------------------------------------------------------------------
# the literature check: Gutermuth et al. 2009 sec. 7.1, one band
# ---------------------------------------------------------------------

def mass_limit_one_band(config, d_r_pc, log10_flim_i4, age_gyr=AGE_1MYR_GYR):
    """`M_lim`: the mass whose I4 (8.0 um) photospheric magnitude at
    `d_r_pc`, undimmed, equals `log10_flim_i4` (mJy) -- Gutermuth et al.
    2009 sec. 7.1's mass-sensitivity construction, the reported check
    (SPEC_PRIORS.md 6.2, 6.5 item 4). Beyond the tabulated magnitude
    range, clamped to the isochrone's own mass edge, never extrapolated.
    """
    fine_masses = np.geomspace(ISOCHRONE_MASS_MIN_MSUN, ISOCHRONE_MASS_MAX_MSUN, _N_MASS_INVERT)
    abs_mag_i4 = abs_mag_grid(config, age_gyr, fine_masses)[:, BAND_KEYS.index("I4")]
    mu = astro_utils.distance_modulus(d_r_pc)
    zp = _BAND_ZERO_POINT_MJY["I4"]
    apparent_mag_target = -2.5 * (log10_flim_i4 - np.log10(zp))
    abs_mag_target = apparent_mag_target - mu
    order = np.argsort(abs_mag_i4)
    return float(np.interp(abs_mag_target, abs_mag_i4[order], fine_masses[order]))


# ---------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------

def build_region(config, region):
    """Computes the selection tables and the literature-check summary
    row for one region.
    """
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    depth_groups_path = config_module.product_path(
        config, "bms", "sesna", "depth-groups", "region")
    depth_groups = depth_groups_module.DepthGroups.read(depth_groups_path, region)
    group_limits = group_limit_vectors(depth_groups)

    nodes = column_grid_module.nodes(config)
    a_grid = np.concatenate(([0.0], np.asarray(nodes, dtype=float)))

    mass_grid, imf_weight = mass_quadrature()
    g_1myr = selection_table(config, AGE_1MYR_GYR, d_r_pc, a_grid, group_limits,
                             mass_grid, imf_weight)
    g_3myr = selection_table(config, AGE_3MYR_GYR, d_r_pc, a_grid, group_limits,
                             mass_grid, imf_weight)

    log10_flim_i4 = float(depth_groups.ref_log10_flim[BAND_KEYS.index("I4")])
    m_lim = mass_limit_one_band(config, d_r_pc, log10_flim_i4, AGE_1MYR_GYR)

    median_group = int(depth_groups.assign_group(np.zeros((1, len(selection.BANDS_DEPTH))))[0])

    return {
        "A_GRID": a_grid, "G_1MYR": g_1myr, "G_3MYR": g_3myr,
        "GROUP_CENTRES": depth_groups.group_centres, "D_R_PC": d_r_pc,
        "M_LIM_8UM_1MYR": m_lim, "IMF_FRAC_ABOVE_MLIM": float(imf_fraction_above(m_lim)),
        "MEDIAN_GROUP": median_group,
    }


def _write_selection(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "region"
        f.attrs["D_R_PC"] = float(result["D_R_PC"])
        f.attrs["IMF_FRAC_ABOVE_1P4"] = IMF_FRAC_ABOVE_1P4
        f.create_dataset("A_GRID", data=result["A_GRID"].astype(np.float64))
        f.create_dataset("G_1MYR", data=result["G_1MYR"].astype(np.float32))
        f.create_dataset("G_3MYR", data=result["G_3MYR"].astype(np.float32))
        f.create_dataset("GROUP_CENTRES", data=result["GROUP_CENTRES"].astype(np.float64))


def build(config, regions=None):
    """Writes, per region (default: all thirty), `bms/yso/selection_yso_
    region__<Region>.hdf5` (`A_GRID`, `G_1MYR`, `G_3MYR`,
    `GROUP_CENTRES`), and the 30-row literature-check summary
    `bms/yso/summary_yso_region.hdf5` (SPEC_PRIORS.md 6.2, 6.4).
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    region_rows, d_r_pc_row, m_lim_row, f_above_row = [], [], [], []
    g1_a0_row, g3_a0_row = [], []
    for region in region_names:
        result = build_region(config, region)
        sel_path = config_module.product_path(
            config, "bms", "yso", "selection", "region", region=region)
        _write_selection(sel_path, result)
        print(f"yso_selection: {region}: K={result['GROUP_CENTRES'].shape[0]} "
              f"M_LIM_8UM_1MYR={result['M_LIM_8UM_1MYR']:.4f} Msun "
              f"f_IMF(>M_lim)={result['IMF_FRAC_ABOVE_MLIM']:.4f} -> {sel_path}")

        region_rows.append(region)
        d_r_pc_row.append(result["D_R_PC"])
        m_lim_row.append(result["M_LIM_8UM_1MYR"])
        f_above_row.append(result["IMF_FRAC_ABOVE_MLIM"])
        g1_a0_row.append(float(result["G_1MYR"][result["MEDIAN_GROUP"], 0]))
        g3_a0_row.append(float(result["G_3MYR"][result["MEDIAN_GROUP"], 0]))

    summary_path = config_module.product_path(config, "bms", "yso", "summary", "region")
    tables_module.update_rows(
        summary_path, region_names,
        {
            "D_R_PC": np.asarray(d_r_pc_row, dtype=np.float64),
            "M_LIM_8UM_1MYR": np.asarray(m_lim_row, dtype=np.float64),
            "IMF_FRAC_ABOVE_MLIM": np.asarray(f_above_row, dtype=np.float64),
            "G_1MYR_A0_MEDIAN_GROUP": np.asarray(g1_a0_row, dtype=np.float64),
            "G_3MYR_A0_MEDIAN_GROUP": np.asarray(g3_a0_row, dtype=np.float64),
        },
        granule="region", attrs={"IMF_FRAC_ABOVE_1P4": IMF_FRAC_ABOVE_1P4})
    print(f"yso_selection: summary -> {summary_path}")


if __name__ == "__main__":
    run(build)
