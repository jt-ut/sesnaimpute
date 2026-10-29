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
- IRAC/MIPS (I1-I4, M1): the flux becomes the source's own 90%-completeness
  flux read off Rob Gutermuth's own measured map at the source's nearest
  pixel (`catalog.completeness.lookup`, COMPLETENESS_2026-10-20.md rule 1),
  never the catalogue's own `DCOMP90_FNU_*` column (no longer read) and
  never a neighbour average (the old nearest-neighbour substitution is
  gone: a source with no map value of its own is off-map, not missing).

Every substitution is recorded per band in `ORIGIN_FNU` (1 = measured
detection, 2 = the survey's own upper limit on an undetected source (2MASS bound),
3 = PACS bound -- a detection within a beam whose flux cannot be assigned to this source
alone, FNU the detection's flux, a ceiling on this source's flux; neither detected nor
undetected, passed through as written by the PACS stages, 90 = covered, undetected -- the map's
own 90%-completeness flux there, 91 = off-map, no completeness value and
no non-detection term -- COMPLETENESS_2026-10-20.md rule 2: "sources with
a catalogue DCOMP90 are exactly the on-map sources", so a band with no
map value is a band not covered there, same meaning Pokhrel et al. 2023
(eHOPS I, ApJS 266, 32) gave the old catalogue column). `DCOMP90_MJY` is
the map's 90%-completeness flux at the source's own pixel for every
source in the five Spitzer bands (detected or not), `NaN` where off-map;
`catalog.limits.completeness` reads the map's own curves directly for
every consumer's non-detection term, not this column.

This table is the SED: eleven bands in `definitions.BANDS` order, the eight
SESNA bands from the delivery and the three Herschel PACS bands (P70, P100,
P160) from the PACS photometry input
`sky/derived/pacs/photometry_pacs_source__<Region>.hdf5`, taken as written
(total flux and ORIGIN codes 1, 3, 90, 91 are
the PACS stages' own, copied without interpretation). Catalogue INCLUSION is a separate rule,
`definitions.SELECTION`, applied downstream, never here. The PACS input is
joined row for row on NAME after this stage's "SESNA "-strip and
duplicate-drop; a mismatch is an error naming the first differing row. A
region with no PACS input file is an error, not a fallback: the PACS stages
write an all-91 file for every region, so the file always exists.
`read_positions` and `n_sources` give the PACS stages this stage's rows
without reading this product (which depends on them).
"""

import functools
import os
import re

import h5py
import numpy as np
import pandas as pd

from sesnaimpute import batches as batches_module
from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog import completeness as completeness_module

#: This module's own key into `attrs_registry.REGISTRY`
#: (`config.product_path`'s file stem for what it writes).
_STEM = "sources_sesna_source"

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


def _read_fwf_batch(path, colspecs, names, bounds):
    """One row-batch's worth of fixed-width parsing (the pool's unit of
    work): `pandas.read_fwf` reads only `bounds`'s own rows at
    `colspecs`'s byte ranges, independent of every other batch.
    """
    start, stop = bounds
    return pd.read_fwf(path, colspecs=colspecs, skiprows=5 + start, nrows=stop - start,
                        header=None, names=names)


def _read_raw_delivery(path, select_columns, n_jobs=1, batch_budget_bytes=ROW_BATCH_BUDGET_BYTES):
    """Reads one region's IPAC1 raw-delivery text file: 5 header lines
    (a `\\created` stamp, pipe-delimited fixed-width column names, dtypes,
    units, null-value sentinels), then one fixed-width row per source.
    `pandas.read_fwf` is asked for `select_columns`'s own byte ranges
    only -- the raw delivery carries many more columns than this module
    keeps, and parsing the ones it discards is the module's dominant
    memory cost. Each row batch (rule 10b) is a row-local, independent
    parse, so the batches are the pool's pieces (`pieces.run`,
    `n_jobs` workers) rather than a serial list comprehension; results
    come back in file order and are concatenated in that order. Returns
    the selected columns as a `pandas.DataFrame` (missing values left as
    the file's own sentinel, not NaN) and a `{column: sentinel}` map.
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

    bounds = list(batches_module.batches(n_rows, row_bytes, budget_bytes=batch_budget_bytes))
    fn = functools.partial(_read_fwf_batch, path, colspecs, names)
    parts = pieces_module.run(fn, bounds, n_jobs, unit="row batches")
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    return df, navals


def _raw_catalog_path(config, region):
    return os.path.join(config.inputs["raw_catalogs"], f"{region}_catalog_ipac1.txt")


_DELIVERY_BANDS = tuple(b for b in definitions.BANDS if b.survey != "PACS")
_PACS_BANDS = tuple(b for b in definitions.BANDS if b.survey == "PACS")


def _strip_and_dedup(df):
    """The delivery's rows as this product has them: duplicate SESNA_NAME rows
    dropped (first kept), then the names with the "SESNA " prefix stripped."""
    dup = df["SESNA_NAME"].duplicated()
    if dup.any():
        df = df[~dup]
    name = df["SESNA_NAME"].str.replace("SESNA ", "", regex=False).str.strip().to_numpy()
    return df, name


def read_positions(config, region):
    """Returns (name, ra_deg, dec_deg) for the region's curated rows, in the
    curated product's own row order: the delivery read at these three
    columns only, through the same fixed-width reader and the same
    "SESNA "-strip and duplicate-drop this stage applies."""
    df, _ = _read_raw_delivery(_raw_catalog_path(config, region), ["SESNA_NAME", "ra", "dec"],
                               n_jobs=config.n_jobs)
    df, name = _strip_and_dedup(df)
    return name, df["ra"].to_numpy(dtype=np.float64), df["dec"].to_numpy(dtype=np.float64)


def n_sources(config, region):
    """The number of rows of the region's curated product."""
    return len(read_positions(config, region)[0])


def _pacs_path(config, region):
    return config_module.product_path(config, "sky/derived", "pacs", "photometry", "source", region=region)


def _assemble_region(config, region):
    bands = _DELIVERY_BANDS
    select_columns = ["SESNA_NAME", "ra", "dec", "L", "B", "CLASS", "AK"]
    for b in bands:
        if b.survey == "2MASS":
            select_columns += [f"MAG_{_RAW_SUFFIX[b.key]}", f"SIGMA_MAG_{_RAW_SUFFIX[b.key]}"]
        else:
            select_columns += [f"FNU_{_RAW_SUFFIX[b.key]}", f"SIGMA_FNU_{_RAW_SUFFIX[b.key]}"]
    # No DCOMP90_FNU_* column: the completeness map (catalog.completeness),
    # not the catalogue's own column, is this file's one source of a
    # source's 90%-completeness flux (module docstring).

    df, navals = _read_raw_delivery(_raw_catalog_path(config, region), select_columns, n_jobs=config.n_jobs)

    df, name = _strip_and_dedup(df)

    n = len(df)
    ra_deg = df["ra"].to_numpy(dtype=np.float64)
    dec_deg = df["dec"].to_numpy(dtype=np.float64)
    gal_l_deg = df["L"].to_numpy(dtype=np.float64)
    gal_b_deg = df["B"].to_numpy(dtype=np.float64)
    class_ = df["CLASS"].to_numpy()
    ak_sesna = df["AK"].to_numpy(dtype=np.float64)

    n_bands = len(definitions.BANDS)
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

        # Every source's own 90%-completeness flux, read off Rob
        # Gutermuth's map at its nearest pixel (COMPLETENESS_2026-10-20.md
        # rule 1); NaN where off-map (rule 2 -- no non-detection term).
        maps = completeness_module.field_maps(config, region, b.key)
        field_idx, iy, ix = completeness_module.lookup(
            config, region, b.key, ra_deg, dec_deg, n_jobs=config.n_jobs, maps=maps)
        on_map = field_idx != -1
        mag90 = np.full(n, np.nan, dtype=np.float64)
        for fidx, m in enumerate(maps):
            sel = field_idx == fidx
            if sel.any():
                mag90[sel] = m.map90[iy[sel], ix[sel]]
        f0_mjy = b.vega_zero_point_jy * 1000.0
        dcomp_j = np.where(on_map, f0_mjy * 10.0 ** (-0.4 * mag90), np.nan)
        dcomp90[:, j] = dcomp_j

        fnu_j = raw_fnu.copy()
        sigma_j = raw_sigma.copy()
        origin_j = origin[:, j]

        case_undetected_on_map = isna_flux & on_map
        fnu_j[case_undetected_on_map] = dcomp_j[case_undetected_on_map]
        sigma_j[case_undetected_on_map] = UPPER_LIMIT_SIGMA
        origin_j[case_undetected_on_map] = 90

        case_undetected_off_map = isna_flux & ~on_map
        fnu_j[case_undetected_off_map] = np.nan
        sigma_j[case_undetected_off_map] = np.nan
        origin_j[case_undetected_off_map] = 91

        fnu[:, j] = fnu_j
        sigma_fnu[:, j] = sigma_j

    # The three PACS bands, taken as the PACS stages wrote them.
    pacs_path = _pacs_path(config, region)
    if not os.path.exists(pacs_path):
        raise FileNotFoundError(
            f"catalog.curated: PACS photometry input missing for region {region!r} at {pacs_path!r} "
            f"-- the PACS stages write one (all-91 where nothing is covered) for every region")
    keys = [x.key for x in definitions.BANDS]
    with h5py.File(pacs_path, "r") as f:
        pacs_name = np.asarray(f["NAME"][()]).astype(str)
        cur_name = name.astype(str)
        if len(pacs_name) != n or not np.array_equal(pacs_name, cur_name):
            k = min(len(pacs_name), n)
            bad = np.nonzero(pacs_name[:k] != cur_name[:k])[0]
            row = int(bad[0]) if len(bad) else k
            raise ValueError(
                f"catalog.curated: PACS NAME does not match the delivery for {region!r}: "
                f"{len(pacs_name)} PACS rows vs {n} curated rows; first differing row {row}: "
                f"PACS {pacs_name[row] if row < len(pacs_name) else None!r} vs "
                f"curated {cur_name[row] if row < n else None!r}")
        for b in _PACS_BANDS:
            j = keys.index(b.key)
            fnu[:, j] = f[f"FNU_{b.key}_MJY"][()]
            sigma_fnu[:, j] = f[f"SIGMA_{b.key}_MJY"][()]
            origin[:, j] = f[f"ORIGIN_{b.key}"][()]
            dcomp90[:, j] = np.nan

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
                for name, data in (
                    ("NAME", name_bytes),
                    ("RA_DEG", assembled["ra_deg"]),
                    ("DEC_DEG", assembled["dec_deg"]),
                    ("GAL_L_DEG", assembled["gal_l_deg"]),
                    ("GAL_B_DEG", assembled["gal_b_deg"]),
                    ("CLASS", assembled["class_"]),
                    ("AK_SESNA", assembled["ak_sesna"]),
                    ("FNU_MJY", assembled["fnu"]),
                    ("SIGMA_FNU_MJY", assembled["sigma_fnu"]),
                    ("DCOMP90_MJY", assembled["dcomp90"]),
                    ("ORIGIN_FNU", assembled["origin"]),
                ):
                    build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])
            st.done(out_path, sources=int(assembled["n"]))


if __name__ == "__main__":
    run(build)
