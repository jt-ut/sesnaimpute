"""The posterior atlas: per admitted nside-512 pixel, the mean `P(C | D)`
over the pixel's sources and the count of `P(YSO | D) > 0.5` sources
(SPEC_BMSTP_DRAFT.md section 8, "The posterior atlas"; IMPLEMENTATION_
BMSTP_DRAFT.md section 1.3 P11, section 4 row 2.6).

The admitted pixel list is `catalog.depth_grid`'s own rows (P6) -- every
nside-512 child of a region's source-bearing nside-256 pixels, not merely
an occupied one. Each source's pixel is `bmstp.density`'s own `HPX_512`
column (P1); the per-source `P_CLASS`/`P_YSO` come from `fittp.classify`
(P8), read in source batches (rule 10b) and folded straight into each
admitted pixel's running sum and count -- P8's `CANDIDATE_FLUX` and
`FLUX_IMPUTED_COV` are what make a region-sized read of it expensive
(W7 review finding 6), so `P_CLASS`/`P_YSO` alone are read here, never
the whole file. An admitted pixel with no sources of its own keeps
`N_SOURCES = 0` and a NaN mean, not a borrowed neighbour's -- unlike the
depth grid's `F_LIM`, a class probability has no meaning to borrow from a
pixel with different sources.
"""

import argparse
import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.batches import batches
from sesnaimpute.fittp.classify import CLASSES, YSO_INDEX

#: Per-row working set for the batch loop (rule 10b): HPX_512 (int64) and
#: P_CLASS (6 float64) / P_YSO (float64) reads, at a generous margin.
ROW_BYTES = 256


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
        pix = np.sort(np.asarray(f["HPX_PIX_512"][:], dtype=np.int64))
    n_pix = pix.size

    density_path = _require(
        config_module.product_path(config, "bmstp", "density", "table", "source", region=region),
        region, "PY sesnaimpute.bmstp.density")
    post_path = _require(
        config_module.product_path(config, "fittp", "classification", "posterior", "source", region=region),
        region, "PY sesnaimpute.fittp.classify")

    sum_p = np.zeros((n_pix, len(CLASSES)), dtype=np.float64)
    count = np.zeros(n_pix, dtype=np.int64)
    n_yso_half = np.zeros(n_pix, dtype=np.int64)

    with h5py.File(density_path, "r") as fd, h5py.File(post_path, "r") as fp:
        n_total = fd["HPX_512"].shape[0]
        if fp["P_CLASS"].shape[0] != n_total:
            raise ValueError("fittp.atlas [%s]: bmstp.density's HPX_512 (%d rows) does not "
                              "row-align with fittp.classify's P_CLASS (%d rows)"
                              % (region, n_total, fp["P_CLASS"].shape[0]))
        bounds = list(batches(n_total, ROW_BYTES))
        for bi, (start, stop) in enumerate(bounds):
            hpx_b = np.asarray(fd["HPX_512"][start:stop], dtype=np.int64)
            p_class_b = np.asarray(fp["P_CLASS"][start:stop], dtype=np.float64)
            p_yso_b = np.asarray(fp["P_YSO"][start:stop], dtype=np.float64)

            idx = np.clip(np.searchsorted(pix, hpx_b), 0, n_pix - 1)
            valid = pix[idx] == hpx_b
            idx_v = idx[valid]
            np.add.at(count, idx_v, 1)
            np.add.at(sum_p, idx_v, p_class_b[valid])
            np.add.at(n_yso_half, idx_v, (p_yso_b[valid] > 0.5).astype(np.int64))
            st.tick(bi + 1, len(bounds), "batches")

    mean_p = np.full((n_pix, len(CLASSES)), np.nan, dtype=np.float32)
    has = count > 0
    mean_p[has] = (sum_p[has] / count[has, None]).astype(np.float32)

    return dict(pix=pix, n_sources=count.astype(np.int32), mean_p=mean_p,
                n_yso_half=n_yso_half.astype(np.int32), n_total_sources=n_total)


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
