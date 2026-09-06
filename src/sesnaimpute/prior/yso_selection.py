"""The YSO mass-based selection (SPEC_PRIORS.md section 6.2): the inner
integral of `epsilon_YSO`, per source, at the source's own eight limits
and its own scaled-extinction ladder --

    g[n, k] = Integral dM f_IMF(M) * 1[ photosphere(M) at d_r,
              dimmed by X_LADDER[k] * A_s[n], clears >= 2 of 8 at
              source n's own limits ]

for the 1 Myr and 3 Myr BHAC15 (Baraffe et al. 2015, A&A 577, A42)
isochrones. `epsilon_YSO(s)` itself (the extinction integral over a
sightline's own embedding density, `Integral da p(a | A_s) * g(a)`) is
built elsewhere, per sightline; this module ships only `g`, the
population- and photosphere-side factor, tabulated once per source on
the shared ladder so that stage is a lookup, not a per-source isochrone
evaluation. Selection is now exact per source (SPEC_PRIORS.md 1.3): no
depth groups, no common-mode shift, every source's own eight limits and
own column enter directly.

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
diffuse/dense law (`prior.selection.kappa_hybrid`), compared to the
source's own 8-band limit vector (`catalog.limits.limits`) at query
extinction `a = X_LADDER * A_s`. `>= 2` of 8 clearing is a catalogued
detection (`prior.selection.MIN_BANDS`).

THE LITERATURE CHECK (reported, not shipped; SPEC_PRIORS.md 6.2, 6.5 item
4). Gutermuth et al. (2009) sec. 7.1's one-band form: the mass whose I4
(8.0 um) photospheric magnitude, undimmed, at a region's own distance
equals a limit equals the region's own reference I4 limit, and the IMF
fraction above it. `mass_limit_one_band` is vectorised over sources and
written per source alongside `g`.
"""

import os
import re

import h5py
import numpy as np
from scipy.special import erf

from sesnaimpute import astro_utils
from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.prior import selection

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: The shared scaled-extinction ladder (`prior.selection`, the one place
#: it is defined).
X_LADDER = selection.X_LADDER

#: The per-batch working-array budget (`sesnaimpute.batches.batches`).
BATCH_BUDGET_BYTES = 512 << 20

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
#: (one evaluation per source, not a per-node tabulation).
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
# the inner selection integral, per source
# ---------------------------------------------------------------------

def selection_table(config, age_gyr, d_r_pc, a_query, source_limits, mass_grid, imf_weight):
    """`(n_src, n_x)`: `g[n, k]`, the IMF-weighted two-of-eight selection
    at `age_gyr` for every source in the batch and every ladder point --
    the module docstring's inner integral, fully vectorised (no loop
    over sources, masses or ladder points). `a_query` (n_src, n_x) is
    `X_LADDER * A_s`; `source_limits` (n_src, 8) is that source's own
    log10 detection limits (`catalog.limits.limits`).
    """
    abs_mag = abs_mag_grid(config, age_gyr, mass_grid)  # (n_mass, 8)
    mu = astro_utils.distance_modulus(d_r_pc)
    zero_points = np.array([_BAND_ZERO_POINT_MJY[k] for k in BAND_KEYS])
    apparent_mag = abs_mag + mu  # undimmed, (n_mass, 8)
    log10_f0 = np.log10(zero_points)[None, :] - 0.4 * apparent_mag  # (n_mass, 8)

    w = selection.law_dense_weight(a_query)  # (n_src, n_x)
    kappa_at_a = selection.kappa_hybrid(config, w)  # (n_src, n_x, 8)
    dimming = 0.4 * a_query[:, :, None] * kappa_at_a  # (n_src, n_x, 8)
    log10_f = log10_f0[None, None, :, :] - dimming[:, :, None, :]  # (n_src, n_x, n_mass, 8)

    n_clear = np.sum(
        log10_f >= source_limits[:, None, None, :], axis=-1)  # (n_src, n_x, n_mass)
    cleared = (n_clear >= selection.MIN_BANDS).astype(np.float64)

    log10_mass = np.log10(mass_grid)
    inner = np.trapz(cleared * imf_weight[None, None, :], x=log10_mass, axis=2)  # (n_src, n_x)
    return inner + IMF_FRAC_ABOVE_1P4


def _row_bytes(n_x, n_mass):
    """The per-source working-array footprint one batch holds: the
    dominant term is the `(n_x, n_mass, 8)` dimmed-flux array built
    twice (1 Myr and 3 Myr) per source."""
    return 2 * n_x * n_mass * N_BANDS * 8 + n_x * N_BANDS * 8 * 2 + N_BANDS * 8


# ---------------------------------------------------------------------
# the literature check: Gutermuth et al. 2009 sec. 7.1, one band
# ---------------------------------------------------------------------

def mass_limit_one_band(config, d_r_pc, log10_flim_i4, age_gyr=AGE_1MYR_GYR):
    """`M_lim`: the mass whose I4 (8.0 um) photospheric magnitude at
    `d_r_pc`, undimmed, equals `log10_flim_i4` (mJy) -- Gutermuth et al.
    2009 sec. 7.1's mass-sensitivity construction, the reported check
    (SPEC_PRIORS.md 6.2, 6.5 item 4). Beyond the tabulated magnitude
    range, clamped to the isochrone's own mass edge, never extrapolated.
    `log10_flim_i4` may be a scalar or an `(n,)` array of per-source
    limits; vectorised throughout.
    """
    fine_masses = np.geomspace(ISOCHRONE_MASS_MIN_MSUN, ISOCHRONE_MASS_MAX_MSUN, _N_MASS_INVERT)
    abs_mag_i4 = abs_mag_grid(config, age_gyr, fine_masses)[:, BAND_KEYS.index("I4")]
    mu = astro_utils.distance_modulus(d_r_pc)
    zp = _BAND_ZERO_POINT_MJY["I4"]
    apparent_mag_target = -2.5 * (np.asarray(log10_flim_i4, dtype=float) - np.log10(zp))
    abs_mag_target = apparent_mag_target - mu
    order = np.argsort(abs_mag_i4)
    return np.interp(abs_mag_target, abs_mag_i4[order], fine_masses[order])


# ---------------------------------------------------------------------
# per-region build: one batched pass over sources
# ---------------------------------------------------------------------

def build_and_write_region(config, region):
    """Computes and writes the region's exact per-source YSO mass
    selection, one batch of sources at a time so no batch's working
    arrays exceed `BATCH_BUDGET_BYTES`.
    """
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    log10_lim = np.log10(limits_module.limits(config, region))
    n_source = log10_lim.shape[0]

    adopted_path = config_module.product_path(
        config, "sky/derived", "adopted", "column", "source", region=region)
    a_col = np.asarray(
        access.per_source(config, region, adopted_path, ["A_COL_K"])["A_COL_K"], dtype=np.float64)
    if a_col.shape[0] != n_source:
        raise ValueError(
            "prior.yso_selection: %r's column count (%d) does not match "
            "the region's %d sources" % (adopted_path, a_col.shape[0], n_source))

    mass_grid, imf_weight = mass_quadrature()
    x_ladder = X_LADDER
    n_x = x_ladder.size

    path = config_module.product_path(config, "bms", "yso", "selection", "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.attrs["D_R_PC"] = float(d_r_pc)
        f.attrs["IMF_FRAC_ABOVE_1P4"] = IMF_FRAC_ABOVE_1P4
        f.create_dataset("X_LADDER", data=x_ladder.astype("f8"))
        ds_g1 = f.create_dataset("G_1MYR", shape=(n_source, n_x), dtype="f4")
        ds_g3 = f.create_dataset("G_3MYR", shape=(n_source, n_x), dtype="f4")
        ds_mlim = f.create_dataset("M_LIM_8UM_1MYR", shape=(n_source,), dtype="f4")
        ds_flim = f.create_dataset("IMF_FRAC_ABOVE_MLIM", shape=(n_source,), dtype="f4")

        row_bytes = _row_bytes(n_x, mass_grid.size)
        for start, stop in batches_module.batches(n_source, row_bytes, budget_bytes=BATCH_BUDGET_BYTES):
            lim_b = np.ascontiguousarray(log10_lim[start:stop])
            a_b = a_col[start:stop]
            a_query_b = np.ascontiguousarray(x_ladder[None, :] * a_b[:, None])

            ds_g1[start:stop] = selection_table(
                config, AGE_1MYR_GYR, d_r_pc, a_query_b, lim_b, mass_grid, imf_weight).astype("f4")
            ds_g3[start:stop] = selection_table(
                config, AGE_3MYR_GYR, d_r_pc, a_query_b, lim_b, mass_grid, imf_weight).astype("f4")

            m_lim_b = mass_limit_one_band(config, d_r_pc, lim_b[:, BAND_KEYS.index("I4")], AGE_1MYR_GYR)
            ds_mlim[start:stop] = m_lim_b.astype("f4")
            ds_flim[start:stop] = imf_fraction_above(m_lim_b).astype("f4")

    return path, n_source


def _build_one(config, region):
    path, n_source = build_and_write_region(config, region)
    with h5py.File(path, "r") as f:
        m_lim_med = float(np.median(f["M_LIM_8UM_1MYR"][:]))
        f_above_med = float(np.median(f["IMF_FRAC_ABOVE_MLIM"][:]))
    print(f"yso_selection: {region}: n_source={n_source} "
          f"median(M_LIM_8UM_1MYR)={m_lim_med:.4f} Msun "
          f"median(f_IMF(>M_lim))={f_above_med:.4f} -> {path}")
    return path


def build(config, regions=None):
    """Writes the exact per-source YSO mass selection for `regions`
    (default: all thirty), one product per region (module docstring)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        _build_one(config, region)


if __name__ == "__main__":
    run(build)
