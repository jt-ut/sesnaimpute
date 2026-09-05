"""The adopted source column: the finer arm wherever it reaches
(SPEC_PRIORS.md section 1.1).

`A_COL_K = A_HERSCHEL_K` (36.3 arcsec beam) where the region has a
Herschel product and the source is covered, finite and positive;
`A_COL_K = A_K` from the Planck arm (`planck_source_column.py`, PLANCK_FWHM_ARCMIN
beam) elsewhere. Sigma, the beam and the Herschel map id all follow the
same arm as the value -- one arm per source, never blended. A merged
value that is anywhere non-finite or non-positive raises rather than
ships, since both arms guarantee 100% finite, positive coverage on their
own.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.sky.derived.planck_source_column import _load_planck_calibration

PROV_HERSCHEL = np.uint8(0)
PROV_PLANCK = np.uint8(1)
HERSCHEL_STATED_FWHM_ARCSEC = 36.3


def _dec(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def _load_herschel_arm(config, region, n_sources):
    """`{covered, a_k, sigma_a_k, map_id, map_names}` for `region`, or
    `None` where the region has no Herschel coverage at all.

    Repoint when sky/derived/herschel/column_herschel_source lands: this
    reads the OLD stand-in per-source Herschel column product instead,
    reordered by its own ROW into catalogue-row order."""
    new_path = config_module.product_path(config, "sky/derived", "herschel", "column", "source", region=region)
    if os.path.exists(new_path):
        with h5py.File(new_path, "r") as f:
            return dict(
                covered=np.asarray(f["COVERED"][:], dtype=bool),
                a_k=np.asarray(f["A_K"][:], dtype=np.float64),
                sigma_a_k=np.asarray(f["SIGMA_A_K"][:], dtype=np.float64),
                map_id=np.asarray(f["MAP_ID"][:], dtype=np.int32),
                map_names=np.asarray(f["MAP_NAME"][:]),
            )

    old_path = os.path.join(config.data_root, "sky/derived/source/herschel_dust-columns",
                            "dust_column_herschel_arm.hdf5")
    if not os.path.exists(old_path):
        return None
    with h5py.File(old_path, "r") as f:
        pr = f["per_region"]
        if region not in pr:
            return None
        g = pr[region]
        row = np.asarray(g["ROW"][:], dtype=np.int64)
        if row.size != n_sources or not np.array_equal(np.sort(row), np.arange(n_sources)):
            raise ValueError(
                f"sky.derived.column.build: Herschel stand-in ROW for {region!r} is not a "
                f"permutation of range({n_sources}) -- cannot align to the curated catalogue"
            )
        order = np.argsort(row)
        return dict(
            covered=np.asarray(g["COVERED"][:], dtype=bool)[order],
            a_k=np.asarray(g["A_HERSCHEL_K"][:], dtype=np.float64)[order],
            sigma_a_k=np.asarray(g["SIG_A_HERSCHEL_K"][:], dtype=np.float64)[order],
            map_id=np.asarray(g["MAP_ID"][:], dtype=np.int32)[order],
            map_names=np.asarray(g["MAP_NAMES"][:]),
        )


def merge_region(config, region, cal):
    """Reads this region's two arms and merges them: Herschel where
    covered, finite and positive; else Planck. Returns the merged arrays."""
    planck_path = config_module.product_path(config, "sky/derived", "planck", "column", "source", region=region)
    if not os.path.exists(planck_path):
        raise FileNotFoundError(
            f"sky.derived.column.build: Planck arm missing for region {region!r} at "
            f"{planck_path!r} -- run the Planck source-column RUNBOOK line for it"
        )
    with h5py.File(planck_path, "r") as f:
        a_col = np.asarray(f["A_K"][:], dtype=np.float64)
        sig_col = np.asarray(f["SIGMA_A_K"][:], dtype=np.float64)

    n = a_col.size
    fwhm = np.full(n, cal["fwhm_arcmin"] * 60.0, dtype=np.float32)
    prov = np.full(n, PROV_PLANCK, dtype=np.uint8)
    map_id = np.full(n, -1, dtype=np.int32)
    map_names = np.array([], dtype=object)

    herschel = _load_herschel_arm(config, region, n)
    if herschel is not None:
        covered, a_h, sig_h = herschel["covered"], herschel["a_k"], herschel["sigma_a_k"]
        use_h = covered & np.isfinite(a_h) & (a_h > 0)
        a_col[use_h] = a_h[use_h]
        sig_col[use_h] = sig_h[use_h]
        fwhm[use_h] = HERSCHEL_STATED_FWHM_ARCSEC
        prov[use_h] = PROV_HERSCHEL
        map_id[use_h] = herschel["map_id"][use_h]
        map_names = herschel["map_names"]

    bad = ~np.isfinite(a_col) | (a_col <= 0)
    if np.any(bad):
        raise ValueError(
            f"sky.derived.column.build: {region!r} has {int(np.count_nonzero(bad))} source(s) "
            "with a non-finite or non-positive merged column"
        )

    return dict(a_col=a_col, sig_col=sig_col, prov=prov, fwhm=fwhm, map_id=map_id, map_names=map_names,
               n_herschel=int(np.count_nonzero(prov == PROV_HERSCHEL)), n=n)


def _build_one_region(config, region, cal):
    d = merge_region(config, region, cal)
    out_path = config_module.product_path(config, "sky/derived", "adopted", "column", "source", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    name_bytes = np.array([(x if isinstance(x, bytes) else str(x).encode("utf-8")) for x in d["map_names"]])
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.create_dataset("A_COL_K", data=d["a_col"].astype(np.float32))
        f.create_dataset("A_COL_SIG_K", data=d["sig_col"].astype(np.float32))
        f.create_dataset("A_COL_PROVENANCE", data=d["prov"])
        f.create_dataset("A_COL_FWHM_ARCSEC", data=d["fwhm"])
        f.create_dataset("HERSCHEL_MAP_ID", data=d["map_id"])
        f.create_dataset("MAP_NAME", data=name_bytes)
    return region, d["n"], d["n_herschel"]


def build(config, regions=None):
    """Builds the adopted source-column product for each region in
    `regions` (default: every region in `regions.REGIONS`), merging the
    Planck and Herschel arms one region at a time, parallelised with
    joblib."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    cal = _load_planck_calibration(config)
    Parallel(n_jobs=-1)(delayed(_build_one_region)(config, region, cal) for region in regions)


if __name__ == "__main__":
    run(build)
