"""The synthetic field-star population, per region (SPEC_PRIORS.md section
1.5; shared by STAR, AGB and PAHC).

A TRILEGAL simulation along the region's own pointing predicts, for every
synthetic star, a true distance, an atmosphere (T_eff, log g, [M/H]) and
an intrinsic eight-band flux -- but TRILEGAL's 2MASS+Spitzer output carries
no Gaia G band and no per-star dust-law information of its own. Both are
supplied by the project's stellar-atmosphere library, and only as a change
of units (SPEC_PRIORS.md section 0.3, C3): the library never enters a
count or a shape, it only lets a TRILEGAL star's own colour be read off
against a template's reference fluxes.

Each star is matched to its nearest atmosphere template in scaled
(log T_eff, log g, [M/H]) (`sed_models/registers/sps_register.hdf5`,
`MODEL_NAME`'s `kt<Teff>g<logg>z<[M/H]>` grammar). The match supplies two
things, both already computed at the register's own build time from the
matched template's own SED:

  - a Gaia G proxy, `G = Ks + (G - Ks)_template`, since the frozen
    TRILEGAL runs were never queried with a Gaia band;
  - the per-star Gaia extinction coefficient `k_G = A_G / A_V`, under
    both the diffuse-ISM and the dense-cloud law (Danielski et al. 2018,
    A&A 614, A19, their polynomial in intrinsic colour and A_V), so a
    later stage can blend the two by SPEC_PRIORS.md section 1.3's ramp
    per star, per column.

Retention keeps every star clearing SESNA's own two-of-eight-band cut
undimmed, at the region's DEEPEST limits (the 1st percentile, per band,
of the real catalogue's own per-source 50%-completeness limits): the
stored sample must serve every source's own limits, including the
region's deepest ones, or a source deeper than some coarser reference
would be missing stars its own depth admits (reading note 04, section A).

What this module does NOT do (IMPLEMENTATION.md section 6, stage 5;
reading note 04, section A). No placement on the region's extinction
profile, no per-node column scan, no depth-profile-cell grid: a star's
actual line-of-sight extinction is assigned per tile in a later stage
(stage 10), against a tile's own mean profile, not here. Every flux and
colour this module writes is TRILEGAL's own intrinsic value; nothing here
is dimmed. The shard-queue twin build path, `ram_guard`, `TaskReport` and
every manifest/register read beyond the two files named above (the
region's TRILEGAL parts and the sps atmosphere register) are dropped as
bookkeeping the science does not need.

Writes one product per region: `field-stars_trilegal_region.hdf5`, root
datasets the retained sample (one row per star clearing retention), plus
a `RAW` group carrying the pre-retention population's own four columns
-- what SPEC_PRIORS.md section 2.1's anchor prediction needs, since
retention itself biases that prediction (reading note 04, section A,
`raw_anchor_columns`).
"""

import os
import re

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.prior.selection import MIN_BANDS
from sesnaimpute.sky.download.trilegal import build as trilegal_download

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: TRILEGAL 1.6's own column header (Girardi et al. 2005, A&A 436, 895),
#: verbatim; a header that does not match this is a service format
#: change, not a silent misread.
TRILEGAL_COLUMNS = (
    "Gc", "logAge", "[M/H]", "m_ini", "logL", "logTe", "logg",
    "m-M0", "Av", "m2/m1", "mbol", "J", "H", "Ks",
    "[3.6]", "[4.5]", "[5.8]", "[8.0]", "[24]", "[70]", "[160]", "Mact",
)

#: TRILEGAL's own band columns, in `definitions.BANDS` order (2MASS J, H,
#: Ks then IRAC I1-I4 then MIPS M1).
BAND_KEYS = tuple(b.key for b in definitions.BANDS)
TRILEGAL_BAND_COLUMNS = ("J", "H", "Ks", "[3.6]", "[4.5]", "[5.8]", "[8.0]", "[24]")

#: The atmosphere register's `MODEL_NAME` grammar: `kt<Teff_K>g<logg>z<[M/H]>`,
#: an optional grid-family suffix ignored. `sed_models_register`'s own
#: build convention; unchanged since the register carries no separate
#: Teff/logg/[M/H] columns of its own.
MODEL_NAME_PATTERN = re.compile(r"kt(\d+)g([+-][\d.]+)z([+-][\d.]+)")

#: The scaled-distance weights the nearest-template match uses to make
#: (log T_eff, log g, [M/H]) commensurate in one k-d tree -- the same
#: metric `match_sps` used in the quarry module
#: (`sesnacomplete.bms_prior.staging.synthetic_field_stars.match_sps`),
#: carried forward unchanged so a star matches the template it always has.
SCALE_LOG_TEFF = 0.05
SCALE_LOG_G = 0.5
SCALE_MH = 0.5

#: SESNA's own catalogue rule (SPEC_PRIORS.md section 1.3): at least this
#: many of the eight bands must clear a source's detection limit.
RETENTION_MIN_BANDS = MIN_BANDS

#: The percentile of the real catalogue's per-source 50%-completeness
#: limits that stands in for "the region's deepest limits"
#: (SPEC_PRIORS.md section 1.5).
DEEPEST_LIMIT_PERCENTILE = 1.0


# ---------------------------------------------------------------------------
# the atmosphere register: nearest-template match, the two unit changes
# ---------------------------------------------------------------------------

def _parse_model_grid(model_names):
    """`(n_model, 3)` (Teff [K], log g, [M/H]), parsed from `MODEL_NAME`."""
    parsed = np.empty((len(model_names), 3), dtype=np.float64)
    for i, name in enumerate(model_names):
        m = MODEL_NAME_PATTERN.match(name)
        if m is None:
            raise ValueError(f"sps_register MODEL_NAME {name!r} does not match "
                              f"the kt<Teff>g<logg>z<[M/H]> grammar")
        parsed[i, 0] = float(m.group(1))
        parsed[i, 1] = float(m.group(2))
        parsed[i, 2] = float(m.group(3))
    return parsed


def load_atmosphere_grid(register_path):
    """The sps atmosphere register's per-template grid and its two library
    unit changes: `grid` (n_model, 3) Teff/logg/[M/H], `g_minus_ks` (n_model,)
    the Gaia G proxy colour, `kg_diffuse`/`kg_dense` (n_model,) the Danielski
    et al. 2018 A_G/A_V coefficient under the diffuse and dense laws.

    `G0_FLUX` and `F_REF_Ks` are the matched template's own raw reference
    fluxes (register attr `FREFRAW`); their ratio needs no floor
    correction (the floor guards a single band's log against a near-zero
    flux, not a same-model ratio of two positive fluxes).
    """
    with h5py.File(register_path, "r") as f:
        models = f["models"]
        names = np.asarray(models["MODEL_NAME"]).astype(str)
        grid = _parse_model_grid(names)
        g0_flux = np.asarray(models["G0_FLUX"], dtype=np.float64)
        f_ref_ks = np.asarray(models["F_REF_Ks"], dtype=np.float64)
        kg_diffuse = np.asarray(models["KG_DRAINE"], dtype=np.float64)
        kg_dense = np.asarray(models["KG_WHITNEY"], dtype=np.float64)
    g_minus_ks = (
        -2.5 * np.log10(g0_flux / f_ref_ks)
        + 2.5 * np.log10(constants.GAIA_G_VEGA_ZP_MJY / constants.VEGA_ZERO_POINT_MJY["Ks"])
    )
    return dict(grid=grid, g_minus_ks=g_minus_ks, kg_diffuse=kg_diffuse, kg_dense=kg_dense)


def match_templates(teff_k, logg, mh, grid):
    """Nearest atmosphere-register template per star, in scaled
    (log T_eff, log g, [M/H]): `(template_index, match_distance)`, both
    `(n,)`.
    """
    x = np.column_stack([
        np.log10(teff_k) / SCALE_LOG_TEFF,
        np.asarray(logg, dtype=np.float64) / SCALE_LOG_G,
        np.asarray(mh, dtype=np.float64) / SCALE_MH,
    ])
    g = np.column_stack([
        np.log10(grid[:, 0]) / SCALE_LOG_TEFF,
        grid[:, 1] / SCALE_LOG_G,
        grid[:, 2] / SCALE_MH,
    ])
    dist, idx = cKDTree(g).query(x)
    return idx, dist


# ---------------------------------------------------------------------------
# the region's TRILEGAL population
# ---------------------------------------------------------------------------

def _read_trilegal_part(path):
    """One TRILEGAL ascii part as a DataFrame, its header checked against
    `TRILEGAL_COLUMNS` so a service format change fails loudly rather than
    silently misreading columns.
    """
    with open(path, "r") as fh:
        header = fh.readline()
    names = tuple(header.lstrip("#").split())
    if names != TRILEGAL_COLUMNS:
        raise ValueError(f"{path}: unexpected TRILEGAL columns {names}")
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                      names=names, engine="c")
    df = df.dropna(subset=["m-M0"])
    if len(df) == 0:
        raise ValueError(f"{path}: no rows")
    return df


def read_region_trilegal(config, region):
    """The region's whole TRILEGAL population (every part concatenated)
    and its pointing: `(df, area_deg2)`. The file names, the pointing's
    galactic (l, b) and its simulated solid angle come from
    `sky.download.trilegal.build.REGION_POINTINGS`, the one place this
    project's TRILEGAL acquisition is described.
    """
    if region not in trilegal_download.REGION_POINTINGS:
        raise ValueError(f"field_stars: {region!r} is not in "
                          f"sky.download.trilegal.build.REGION_POINTINGS")
    info = trilegal_download.REGION_POINTINGS[region]
    trilegal_dir = f"{config.data_root}/sky/download/trilegal"
    file_names = trilegal_download._file_names(region, info["n_parts"])
    parts = [_read_trilegal_part(f"{trilegal_dir}/{name}") for name in file_names]
    df = pd.concat(parts, ignore_index=True)
    return df, float(info["area_deg2"])


def intrinsic_fluxes_mjy(df):
    """`(n, 8)` intrinsic apparent fluxes in mJy from TRILEGAL's Vega
    magnitudes, on this project's own Vega zero points (`constants.VEGA_ZERO_POINT_MJY`) --
    the same zero points SESNA's own fluxes are measured on.
    """
    out = np.empty((len(df), len(BAND_KEYS)), dtype=np.float64)
    for j, (band, column) in enumerate(zip(BAND_KEYS, TRILEGAL_BAND_COLUMNS)):
        out[:, j] = constants.VEGA_ZERO_POINT_MJY[band] * 10.0 ** (-0.4 * df[column].to_numpy())
    return out


def distance_pc(df):
    """`d = 10**(0.2*(m-M0) + 1)` pc, from TRILEGAL's true distance
    modulus (`m-M0`, dust-free by the query's own `internal_extinction_kind
    = 0`, SPEC_PRIORS.md section 1.5)."""
    return 10.0 ** (0.2 * df["m-M0"].to_numpy() + 1.0)


# ---------------------------------------------------------------------------
# retention: the region's deepest limits, two of eight undimmed
# ---------------------------------------------------------------------------

def deepest_limits(config, region):
    """`(8,)` mJy: the region's deepest per-source 50%-completeness limit
    in each band, the `DEEPEST_LIMIT_PERCENTILE`-th percentile over the
    real catalogue's own sources (SPEC_PRIORS.md section 1.5) -- so the
    retained sample serves every source's own limits, not only a typical
    one.
    """
    f_lim = limits_module.limits(config, region)
    return np.percentile(f_lim, DEEPEST_LIMIT_PERCENTILE, axis=0)


def passes_two_of_eight(flux, f_lim, min_bands=RETENTION_MIN_BANDS):
    """Which rows of `flux` (n, 8) clear at least `min_bands` of `f_lim`
    (8,) -- undimmed, by construction: this module carries no per-star
    extinction (see module docstring)."""
    flux = np.asarray(flux, dtype=np.float64)
    n_clear = np.sum(flux >= np.asarray(f_lim, dtype=np.float64)[None, :], axis=1)
    return n_clear >= min_bands


# ---------------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------------

def build_region(config, region, atmosphere):
    """The region's matched, retained TRILEGAL table plus the raw group,
    as a dict of arrays ready for `write_region`.
    """
    df, area_deg2 = read_region_trilegal(config, region)
    n_raw = len(df)

    flux = intrinsic_fluxes_mjy(df)
    dist_pc = distance_pc(df)
    log_teff = df["logTe"].to_numpy(dtype=np.float64)
    logg = df["logg"].to_numpy(dtype=np.float64)
    mh = df["[M/H]"].to_numpy(dtype=np.float64)
    ks_mag = df["Ks"].to_numpy(dtype=np.float64)

    idx, match_dist = match_templates(10.0 ** log_teff, logg, mh, atmosphere["grid"])
    g_proxy = ks_mag + atmosphere["g_minus_ks"][idx]
    kg_diffuse = atmosphere["kg_diffuse"][idx]
    kg_dense = atmosphere["kg_dense"][idx]

    # this population carries one TRILEGAL query pointing per region
    # (`sky.download.trilegal.build.REGION_POINTINGS`): every row shares
    # pointing index 0.
    pointing_index = np.zeros(n_raw, dtype=np.int16)

    f_lim_deep = deepest_limits(config, region)
    keep = passes_two_of_eight(flux, f_lim_deep)

    return dict(
        n_raw=n_raw,
        area_deg2=area_deg2,
        raw=dict(
            g_proxy=g_proxy,
            ks_mag=ks_mag,
            dist_pc=dist_pc,
            pointing_index=pointing_index,
        ),
        retained=dict(
            dist_pc=dist_pc[keep],
            log_teff=log_teff[keep],
            log_g=logg[keep],
            mh=mh[keep],
            fnu_mjy=flux[keep],
            g_proxy=g_proxy[keep],
            ks_mag=ks_mag[keep],
            k_g_diffuse=kg_diffuse[keep],
            k_g_dense=kg_dense[keep],
            template_index=idx[keep],
            match_dist=match_dist[keep],
            pointing_index=pointing_index[keep],
        ),
    )


def write_region(path, region, result):
    """Writes one region's field-star product: the retained sample at
    root, the pre-retention population's own four anchor-prediction
    columns under `RAW`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ret = result["retained"]
    raw = result["raw"]
    with h5py.File(path, "w") as f:
        f.create_dataset("DIST_PC", data=ret["dist_pc"].astype(np.float32))
        f.create_dataset("LOG_TEFF", data=ret["log_teff"].astype(np.float32))
        f.create_dataset("LOG_G", data=ret["log_g"].astype(np.float32))
        f.create_dataset("MH", data=ret["mh"].astype(np.float32))
        f.create_dataset("FNU_MJY", data=ret["fnu_mjy"].astype(np.float32))
        f.create_dataset("G_PROXY", data=ret["g_proxy"].astype(np.float32))
        f.create_dataset("KS_MAG", data=ret["ks_mag"].astype(np.float32))
        f.create_dataset("K_G_DIFFUSE", data=ret["k_g_diffuse"].astype(np.float32))
        f.create_dataset("K_G_DENSE", data=ret["k_g_dense"].astype(np.float32))
        f.create_dataset("TEMPLATE_INDEX", data=ret["template_index"].astype(np.int32))
        f.create_dataset("MATCH_DIST", data=ret["match_dist"].astype(np.float32))
        f.create_dataset("POINTING_INDEX", data=ret["pointing_index"].astype(np.int16))

        raw_group = f.create_group("RAW")
        raw_group.create_dataset("G_PROXY", data=raw["g_proxy"].astype(np.float32))
        raw_group.create_dataset("KS_MAG", data=raw["ks_mag"].astype(np.float32))
        raw_group.create_dataset("DIST_PC", data=raw["dist_pc"].astype(np.float32))
        raw_group.create_dataset("POINTING_INDEX", data=raw["pointing_index"].astype(np.int16))

        f.attrs["GRANULE"] = "region"
        f.attrs["OMEGA_SIM_DEG2"] = float(result["area_deg2"])
        f.attrs["N_RAW"] = int(result["n_raw"])


def build(config, regions=None):
    """Writes the synthetic field-star product for `regions` (default:
    all thirty), one file per region.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    register_path = f"{config.data_root}/sed_models/registers/sps_register.hdf5"
    atmosphere = load_atmosphere_grid(register_path)
    for region in region_names:
        result = build_region(config, region, atmosphere)
        path = config_module.product_path(config, "bms", "trilegal", "field-stars",
                                            "region", region=region)
        write_region(path, region, result)
        print(f"field_stars: {region}: N_RAW={result['n_raw']} "
              f"N_RETAINED={len(result['retained']['dist_pc'])} -> {path}")


if __name__ == "__main__":
    run(build)
