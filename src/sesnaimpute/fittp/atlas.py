"""The posterior atlas: per admitted nside-512 pixel, the mean `P(C | D)`
over the pixel's sources and the count of `P(YSO | D) > 0.5` sources
(SPEC_BMSTP_DRAFT.md section 8, "The posterior atlas"; IMPLEMENTATION_
BMSTP_DRAFT.md section 1.3 P11, section 4 row 2.6).

The admitted pixel list is `catalog.depth_grid`'s own rows (P6) -- every
nside-512 child of a region's source-bearing nside-256 pixels, not merely
an occupied one. Each source's pixel is `bmstp.density`'s own `HPX_512`
column (P1); the per-source `P_CLASS`/`P_YSO` come from `fittp.classify`
(P8). A pandas groupby folds the source table to one row per occupied
pixel (no loop over pixels); an admitted pixel with no sources of its own
keeps `N_SOURCES = 0` and a NaN mean, not a borrowed neighbour's -- unlike
the depth grid's `F_LIM`, a class probability has no meaning to borrow
from a pixel with different sources.
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.fittp.classify import CLASSES, YSO_INDEX


def _require(path, region, runbook_line):
    if not os.path.exists(path):
        raise RuntimeError("fittp.atlas [%s]: missing %s -- run RUNBOOKtp.sh's "
                            "'%s' line first" % (region, path, runbook_line))
    return path


def build_region(config, region, st):
    depth_path = _require(
        config_module.product_path(config, "catalog", "sesna", "depth-grid", "hpx512", region=region),
        region, "PY sesnaimpute.catalog.depth_grid")
    with h5py.File(depth_path, "r") as f:
        admitted = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)

    density_path = _require(
        config_module.product_path(config, "bmstp", "density", "table", "source", region=region),
        region, "PY sesnaimpute.bmstp.density")
    with h5py.File(density_path, "r") as f:
        hpx = np.asarray(f["HPX_512"][:], dtype=np.int64)

    post_path = _require(
        config_module.product_path(config, "fittp", "classification", "posterior", "source", region=region),
        region, "PY sesnaimpute.fittp.classify")
    with h5py.File(post_path, "r") as f:
        p_class = np.asarray(f["P_CLASS"][:], dtype=np.float64)
        p_yso = np.asarray(f["P_YSO"][:], dtype=np.float64)

    if hpx.shape[0] != p_class.shape[0]:
        raise ValueError("fittp.atlas [%s]: bmstp.density's HPX_512 (%d rows) does not "
                          "row-align with fittp.classify's P_CLASS (%d rows)"
                          % (region, hpx.shape[0], p_class.shape[0]))

    df = pd.DataFrame(p_class, columns=CLASSES)
    df["HPX_PIX_512"] = hpx
    df["YSO_HALF"] = p_yso > 0.5
    grouped = df.groupby("HPX_PIX_512", sort=True)
    occ_pix = grouped.size().index.to_numpy(dtype=np.int64)
    occ_n = grouped.size().to_numpy(dtype=np.int32)
    occ_mean_p = grouped[list(CLASSES)].mean().to_numpy(dtype=np.float32)
    occ_n_yso = grouped["YSO_HALF"].sum().to_numpy(dtype=np.int32)
    st.tick(1, 1, "pixels (grouped)")

    pix = np.sort(admitted)
    is_occ = np.isin(pix, occ_pix)
    occ_order = np.argsort(occ_pix)
    occ_row = occ_order[np.searchsorted(occ_pix[occ_order], pix[is_occ])]

    n_sources = np.zeros(pix.size, dtype=np.int32)
    mean_p = np.full((pix.size, len(CLASSES)), np.nan, dtype=np.float32)
    n_yso_half = np.zeros(pix.size, dtype=np.int32)
    n_sources[is_occ] = occ_n[occ_row]
    mean_p[is_occ] = occ_mean_p[occ_row]
    n_yso_half[is_occ] = occ_n_yso[occ_row]

    return dict(pix=pix, n_sources=n_sources, mean_p=mean_p, n_yso_half=n_yso_half,
                n_total_sources=p_class.shape[0])


def write_region(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("HPX_PIX_512", data=result["pix"])
        f.create_dataset("N_SOURCES", data=result["n_sources"])
        for ci, cls in enumerate(CLASSES):
            f.create_dataset("MEAN_P_%s" % cls, data=result["mean_p"][:, ci])
        f.create_dataset("N_YSO_ABOVE_HALF", data=result["n_yso_half"])
        f.attrs["GRANULE"] = "hpx512"
        f.attrs["CLASSES"] = np.array(CLASSES, dtype="S8")


def build(config, regions=None):
    """Writes `fittp/atlas/posterior_atlas_hpx512__R.hdf5` for `regions`
    (default all thirty), one file per region (IMPLEMENTATION_BMSTP_
    DRAFT.md P11; SPEC_BMSTP_DRAFT.md section 8's posterior atlas).
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        with progress.Stage("fittp.atlas", region) as st:
            result = build_region(config, region, st)
            path = config_module.product_path(config, "fittp", "atlas", "posterior", "hpx512", region=region)
            write_region(path, result)
            total_check = int(result["n_sources"].sum())
            st.done(path, n_pix=result["pix"].shape[0], n_sources_summed=total_check,
                    n_sources_region=result["n_total_sources"],
                    total_count_ratio=total_check / result["n_total_sources"]
                    if result["n_total_sources"] else float("nan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()
    cfg = config_module.load(args.config)
    build(cfg, regions=args.regions)
