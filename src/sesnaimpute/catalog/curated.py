"""The curated SESNA source catalogue, per region (SPEC_PRIORS.md section 1.3).

Turns the raw SESNA delivery (IPAC1 text, `config.inputs["raw_catalogs"]`)
into one plain-dataset HDF5 per region, holding the per-band flux
assembly that the detection-limit map (`depths.py`, `limits.py`) and every
other consumer read. Two substitutions fill a non-detection, chosen by
band:

- 2MASS (J/H/Ks): the flux and its `DCOMP90` entry both become the fixed
  survey bound `UB_2MASS_MJY` -- the mJy conversion of the 2MASS Point
  Source Catalog's documented 10-sigma/~99%-completeness limiting
  magnitudes (Skrutskie et al. 2006, AJ 131, 1163; Cutri et al. 2003,
  2MASS Explanatory Supplement section 1.6), via F = F0 * 10**(-mag/2.5)
  with the zero points in `definitions.BANDS` (Cohen, Wheaton & Megeath
  2003, AJ 126, 1090).
- IRAC/MIPS (I1-I4, M1): the flux becomes the source's own `DCOMP90`
  completeness-limit flux, or, when the source has none of its own, the
  median `DCOMP90` of its `DEFAULT_NN_K` nearest sky neighbours that do.

Every substitution is recorded per band in `ORIGIN_FNU` (1 = measured
detection, 2 = 2MASS survey bound, 90 = own DCOMP90, 91 = neighbour
DCOMP90). `SPEC_PRIORS.md` section 1.3 reads the resulting `DCOMP90_MJY`
map, rescaled per region and band, as the per-source 50%-completeness
limit for the five Spitzer bands.
"""

import os
import re

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sesnaimpute import batches as batches_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run

#: Row-batch memory budget for the raw-delivery read (rule 10b): the raw
#: delivery carries far more columns than this module selects (measured
#: 18.8 GB resident reading Cygnus X's full un-selected column set), so
#: `_read_raw_delivery` never asks `pandas.read_fwf` for a column it does
#: not keep, and reads even that reduced set one row-batch at a time.
ROW_BATCH_BUDGET_BYTES = 512 << 20

_RAW_SUFFIX = {
    "J": "J", "H": "H", "Ks": "KS",
    "I1": "3_6", "I2": "4_5", "I3": "5_8", "I4": "8_0", "M1": "24",
}

# 2MASS Point Source Catalog global upper bounds (mJy): J=15.8, H=15.1,
# Ks=14.3 mag (Skrutskie et al. 2006, AJ 131, 1163; Cutri et al. 2003,
# 2MASS Explanatory Supplement section 1.6) converted with the zero
# points in definitions.BANDS.
UB_2MASS_MJY = {"J": 0.763, "H": 0.934, "Ks": 1.27}

# A filled flux is not a Gaussian measurement: SIGMA_FNU carries the
# fitter's fixed violation-penalty scale for an upper limit (sedfitter's
# chi_squared, valid=3 branch): error=0.99 gives a large but finite
# penalty short of the infinite cutoff at error=1.0.
UPPER_LIMIT_SIGMA = 0.99

# Sky neighbours averaged (by median) for a source with no DCOMP90 of
# its own, in an IRAC/MIPS band.
DEFAULT_NN_K = 5

# The delivery's own 2MASS zero points (Vega system, Jy), measured from
# its FNU_J/H/KS columns against its MAG_J/H/KS columns: J 1669, H 980,
# Ks 620 (FNU / 10**(-0.4*MAG) is this constant per band to 1 Jy scatter
# over 18,000-77,000 rows in Perseus and Lupus, `studies/star_colour_
# offset.md`). They differ from `definitions.BANDS`'s Cohen, Wheaton &
# Megeath 2003 zero points (J 1594, H 1024, Ks 666.7 Jy) that every model
# flux in the package is on, so the curated J/H/Ks flux is formed here
# from the delivered magnitude with `definitions.BANDS`'s zero point
# directly, never from the delivered FNU_J/H/KS.

_DTYPE_MAPPER = {"CHAR": str, "DOUBLE": float, "FLOAT": float, "INT": "Int64"}


def _read_raw_delivery(path, select_columns, batch_budget_bytes=ROW_BATCH_BUDGET_BYTES):
    """Reads one region's IPAC1 raw-delivery text file: 5 header lines
    (a `\\created` stamp, pipe-delimited fixed-width column names, dtypes,
    units, null-value sentinels), then one fixed-width row per source.
    `pandas.read_fwf` is asked for `select_columns`'s own byte ranges
    only -- the raw delivery carries many more columns than this module
    keeps, and parsing the ones it discards is the module's dominant
    memory cost -- and is called once per row batch (rule 10b), each
    batch's rows concatenated in file order. Returns the selected columns
    as a `pandas.DataFrame` (missing values left as the file's own
    sentinel, not NaN) and a `{column: sentinel}` map.
    """
    with open(path, "r") as f:
        head = [next(f) for _ in range(5)]
    pipelocs = np.array([m.start() for m in re.finditer(r"\|", head[1])])
    lo_all, hi_all = pipelocs[:-1] + 1, pipelocs[1:]
    names_all = [head[1][i:j].strip() for i, j in zip(lo_all, hi_all)]
    dtype_names_all = [head[2][i:j].strip() for i, j in zip(lo_all, hi_all)]
    naval_strs_all = [head[4][i:j].strip() for i, j in zip(lo_all, hi_all)]

    select = list(select_columns)
    order_of = {n: k for k, n in enumerate(select)}
    keep = sorted((i for i, n in enumerate(names_all) if n in order_of),
                  key=lambda i: order_of[names_all[i]])
    colspecs = [(int(lo_all[i]), int(hi_all[i])) for i in keep]
    names = [names_all[i] for i in keep]

    navals = {}
    for i in keep:
        n, v, dt = names_all[i], naval_strs_all[i], dtype_names_all[i]
        if v == "null":
            navals[n] = None
        elif dt == "INT":
            navals[n] = int(float(v))
        else:
            navals[n] = float(v)

    with open(path, "r") as f:
        n_rows = sum(1 for _ in f) - 5
    row_bytes = int(sum(hi - lo for lo, hi in colspecs)) or 1

    parts = [
        pd.read_fwf(path, colspecs=colspecs, skiprows=5 + start, nrows=stop - start,
                    header=None, names=names)
        for start, stop in batches_module.batches(n_rows, row_bytes, budget_bytes=batch_budget_bytes)
    ]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    return df, navals


def _raw_catalog_path(config, region):
    return os.path.join(config.inputs["raw_catalogs"], f"{region}_catalog_ipac1.txt")


def _median_nn_impute(ra_deg, dec_deg, source_mask, query_mask, values, k=DEFAULT_NN_K):
    """The median DCOMP90 of a source's `k` nearest sky neighbours (3-D
    unit-sphere Euclidean distance, distortion-free and wraparound-free,
    searched with a k-d tree), used when the source has no DCOMP90 of its
    own.
    """
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    vectors = np.column_stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    n_source = int(source_mask.sum())
    tree = cKDTree(vectors[source_mask])
    k_eff = min(k, n_source)
    _, neighbor_idx = tree.query(vectors[query_mask], k=k_eff, workers=-1)
    if k_eff == 1:
        neighbor_idx = neighbor_idx.reshape(-1, 1)
    source_values = values[source_mask]
    return np.median(source_values[neighbor_idx], axis=1)


def _assemble_region(config, region):
    bands = definitions.BANDS
    select_columns = ["SESNA_NAME", "ra", "dec", "L", "B", "CLASS", "AK"]
    for b in bands:
        if b.survey == "2MASS":
            select_columns += [f"MAG_{_RAW_SUFFIX[b.key]}", f"SIGMA_MAG_{_RAW_SUFFIX[b.key]}"]
        else:
            select_columns += [f"FNU_{_RAW_SUFFIX[b.key]}", f"SIGMA_FNU_{_RAW_SUFFIX[b.key]}"]
    for b in bands:
        if b.survey in ("IRAC", "MIPS"):
            select_columns.append(f"DCOMP90_FNU_{_RAW_SUFFIX[b.key]}")

    df, navals = _read_raw_delivery(_raw_catalog_path(config, region), select_columns)

    dup = df["SESNA_NAME"].duplicated()
    if dup.any():
        df = df[~dup]

    n = len(df)
    name = df["SESNA_NAME"].str.replace("SESNA ", "", regex=False).str.strip().to_numpy()
    ra_deg = df["ra"].to_numpy(dtype=np.float64)
    dec_deg = df["dec"].to_numpy(dtype=np.float64)
    gal_l_deg = df["L"].to_numpy(dtype=np.float64)
    gal_b_deg = df["B"].to_numpy(dtype=np.float64)
    class_ = df["CLASS"].to_numpy()
    ak_sesna = df["AK"].to_numpy(dtype=np.float64)

    n_bands = len(bands)
    fnu = np.empty((n, n_bands), dtype=np.float64)
    sigma_fnu = np.empty((n, n_bands), dtype=np.float64)
    dcomp90 = np.empty((n, n_bands), dtype=np.float64)
    origin = np.ones((n, n_bands), dtype=np.int8)

    for j, b in enumerate(bands):
        suffix = _RAW_SUFFIX[b.key]

        if b.survey == "2MASS":
            # F = F0 * 10**(-0.4*m), sigma_F = 0.4*ln(10)*F*sigma_m, with
            # F0 = definitions.BANDS's own zero point (mJy), not the
            # delivery's (constants-block comment above).
            raw_mag = df[f"MAG_{suffix}"].to_numpy(dtype=np.float64)
            raw_sigma_mag = df[f"SIGMA_MAG_{suffix}"].to_numpy(dtype=np.float64)
            mag_naval = navals[f"MAG_{suffix}"]
            isna_flux = raw_mag == mag_naval
            f0_mjy = b.vega_zero_point_jy * 1000.0
            conv_fnu = f0_mjy * 10.0 ** (-0.4 * raw_mag)
            conv_sigma = 0.4 * np.log(10.0) * conv_fnu * raw_sigma_mag

            bound = UB_2MASS_MJY[b.key]
            fnu[:, j] = np.where(isna_flux, bound, conv_fnu)
            sigma_fnu[:, j] = np.where(isna_flux, UPPER_LIMIT_SIGMA, conv_sigma)
            origin[isna_flux, j] = 2
            dcomp90[:, j] = bound
            continue

        raw_fnu = df[f"FNU_{suffix}"].to_numpy(dtype=np.float64)
        raw_sigma = df[f"SIGMA_FNU_{suffix}"].to_numpy(dtype=np.float64)
        fnu_naval = navals[f"FNU_{suffix}"]
        isna_flux = raw_fnu == fnu_naval

        dcomp_col = f"DCOMP90_FNU_{suffix}"
        raw_dcomp = df[dcomp_col].to_numpy(dtype=np.float64)
        dcomp_naval = navals[dcomp_col]
        isna_dcomp = raw_dcomp == dcomp_naval
        has_dcomp = ~isna_dcomp

        dcomp_j = np.empty(n, dtype=np.float64)
        dcomp_j[has_dcomp] = raw_dcomp[has_dcomp]
        if isna_dcomp.any():
            dcomp_j[isna_dcomp] = _median_nn_impute(
                ra_deg, dec_deg, has_dcomp, isna_dcomp, raw_dcomp
            )
            case_flux_no_dcomp = isna_dcomp & ~isna_flux
            dcomp_j[case_flux_no_dcomp] = np.maximum(
                dcomp_j[case_flux_no_dcomp], raw_fnu[case_flux_no_dcomp]
            )
        dcomp90[:, j] = dcomp_j

        fnu_j = raw_fnu.copy()
        sigma_j = raw_sigma.copy()
        origin_j = origin[:, j]

        case_no_flux_has_dcomp = isna_flux & has_dcomp
        fnu_j[case_no_flux_has_dcomp] = dcomp_j[case_no_flux_has_dcomp]
        sigma_j[case_no_flux_has_dcomp] = UPPER_LIMIT_SIGMA
        origin_j[case_no_flux_has_dcomp] = 90

        case_no_flux_no_dcomp = isna_flux & isna_dcomp
        fnu_j[case_no_flux_no_dcomp] = dcomp_j[case_no_flux_no_dcomp]
        sigma_j[case_no_flux_no_dcomp] = UPPER_LIMIT_SIGMA
        origin_j[case_no_flux_no_dcomp] = 91

        fnu[:, j] = fnu_j
        sigma_fnu[:, j] = sigma_j

    return {
        "n": n, "name": name, "ra_deg": ra_deg, "dec_deg": dec_deg,
        "gal_l_deg": gal_l_deg, "gal_b_deg": gal_b_deg, "class_": class_,
        "ak_sesna": ak_sesna, "fnu": fnu, "sigma_fnu": sigma_fnu,
        "dcomp90": dcomp90, "origin": origin,
    }


def build(config, regions=None):
    """Builds the curated `source`-granule catalogue for each region in
    `regions` (default: every region in `regions.REGIONS`).
    """
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    band_keys = [b.key for b in definitions.BANDS]

    for region in regions:
        with progress_module.Stage("catalog.curated", region) as st:
            raw_path = _raw_catalog_path(config, region)
            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"catalog.curated.build: raw delivery missing for region {region!r} "
                    f"at {raw_path!r} -- run the '--- catalog ---' RUNBOOK line for it"
                )
            assembled = _assemble_region(config, region)

            out_path = config_module.product_path(
                config, "catalog", "sesna", "sources", "source", region=region
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            name_bytes = assembled["name"].astype("S")
            with h5py.File(out_path, "w") as f:
                f.attrs["GRANULE"] = "source"
                f.attrs["REGION"] = region
                f.attrs["N_SOURCES"] = assembled["n"]
                f.attrs["BANDS"] = band_keys
                f.create_dataset("NAME", data=name_bytes)
                f.create_dataset("RA_DEG", data=assembled["ra_deg"])
                f.create_dataset("DEC_DEG", data=assembled["dec_deg"])
                f.create_dataset("GAL_L_DEG", data=assembled["gal_l_deg"])
                f.create_dataset("GAL_B_DEG", data=assembled["gal_b_deg"])
                f.create_dataset("CLASS", data=assembled["class_"])
                f.create_dataset("AK_SESNA", data=assembled["ak_sesna"])
                f.create_dataset("FNU_MJY", data=assembled["fnu"])
                f.create_dataset("SIGMA_FNU_MJY", data=assembled["sigma_fnu"])
                f.create_dataset("DCOMP90_MJY", data=assembled["dcomp90"])
                f.create_dataset("ORIGIN_FNU", data=assembled["origin"])
            st.done(out_path, sources=int(assembled["n"]))


if __name__ == "__main__":
    run(build)
