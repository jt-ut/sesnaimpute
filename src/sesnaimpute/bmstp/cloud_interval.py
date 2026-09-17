"""The sightline's own dense fraction (SPEC_BMSTP_DRAFT.md section 2, the
diffuse/dense law): per sightline, how much of the region's cloud interval
`[d_front, d_back]` (`sample_cloud.cloud_interval_pc`) sits in front of each
`log10 xi` cell, from the SIGHTLINE'S OWN cumulative extinction profile
`xi(d)` (`sample_cloud._region_profile`, the same native cumulative profile
`bmstp.shapes.build_cloud` reads) -- sky data alone (the 3-D dust map and
the cloud interval), never the prior. An additive stage: it reads what
`bmstp.shapes` already loads and writes a new product, nothing that exists.

Per sightline row, `XI_FRONT`/`XI_BACK` are the profile's own `xi(d)`
(interpolated in `d`, clipped to `[0, 1]`) at the cloud interval's front and
back edge; `W_CLOUD` is the diffuse/dense ramp weight
(`population.selection.law_dense_weight`) evaluated at the column of dust
BETWEEN those two depths, `(XI_BACK - XI_FRONT) * A_COL_K` with `A_COL_K`
the sightline's own extinction column (`sky/derived/adopted/extinction`'s
`sightline` granule -- the same column P1 (`bmstp.density`) assigns each of
the row's sources, section 3.2's "extinction column"). `fittp.prior_reader`
reads this product through P1's `SIGHTLINE_ROW` to form the per-cell dense
fraction `w_i` every class's prior read blends the two extinction-law
designs by (section 2, section 4.2).
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.bmstp import sample_cloud
from sesnaimpute.population import selection as population_selection


def _sightline_a_col_k(config, region, hpx_pix_256):
    """This region's own sightlines' `A_COL_K`, the extinction column
    (survey-wide `sky/derived/adopted/extinction/sightline` product,
    keyed by `HPX_PIX_256`) -- the same column `bmstp.density` (P1)
    assigns each row's sources, read here at the sightline granule
    directly rather than re-derived."""
    path = config_module.product_path(config, "sky/derived", "adopted", "extinction", "sightline")
    with h5py.File(path, "r") as f:
        pix_all = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        a_k_all = np.asarray(f["A_K"][:], dtype=np.float64)
    order = np.argsort(pix_all)
    pix_sorted = pix_all[order]
    a_k_sorted = a_k_all[order]
    loc = np.searchsorted(pix_sorted, hpx_pix_256)
    capped = np.minimum(loc, pix_sorted.size - 1) if pix_sorted.size else loc
    ok = pix_sorted.size > 0 and np.all(pix_sorted[capped] == hpx_pix_256)
    if not ok:
        missing = hpx_pix_256[pix_sorted[capped] != hpx_pix_256] if pix_sorted.size else hpx_pix_256
        raise ValueError("bmstp.cloud_interval [%s]: sightline pixel(s) %r missing from %s"
                          % (region, missing[:5].tolist(), path))
    return a_k_sorted[capped]


def build_region(config, region, st):
    """One region's `XI_FRONT`, `XI_BACK`, `W_CLOUD`, one row per sightline,
    in `bmstp.shapes.build_cloud`'s own sightline order (so a source's
    `SIGHTLINE_ROW`, P1, indexes this product exactly as it indexes P3)."""
    loaded = sample_cloud._region_profile(config, region)
    xi_edges = loaded["xi_edges"]      # (n_sl, n_d+1), monotone non-decreasing in d
    d_edges = loaded["d_edges"]        # (n_sl, n_d+1), the SAME distance grid every row shares
    hpx_pix_256 = loaded["hpx_pix_256"]
    n_sl = hpx_pix_256.size

    d_front, d_back = sample_cloud.cloud_interval_pc(config, region)

    xi_front = np.zeros(n_sl, dtype=np.float64)
    xi_back = np.zeros(n_sl, dtype=np.float64)
    for row in range(n_sl):
        # xi(d) at this sightline's own native cumulative profile: d_edges[row]
        # spans at least [dist_pc[0], dist_pc[-1] + 2*tail_efold_pc[row]], which
        # `sample_cloud.cloud_interval_pc` already caps d_front/d_back inside of
        # (its own floor/cap against dist_first and dist_last + 2*min(tail_efold)),
        # so both lookups sit inside every row's own domain -- no extrapolation.
        xi_front[row] = np.clip(np.interp(d_front, d_edges[row], xi_edges[row]), 0.0, 1.0)
        xi_back[row] = np.clip(np.interp(d_back, d_edges[row], xi_edges[row]), 0.0, 1.0)
    # XI_FRONT <= XI_BACK by construction (xi(d) non-decreasing in d,
    # d_front <= d_back); guarded against float round-off alone.
    xi_back = np.maximum(xi_back, xi_front)

    a_col_k = _sightline_a_col_k(config, region, hpx_pix_256)
    # a sightline with no dust between the two depths (XI_BACK == XI_FRONT,
    # including a sightline whose own profile never reaches past d_front --
    # no cloud interval) reads a column of dust of exactly zero there, and
    # `law_dense_weight(0) == 0`: XI_FRONT = XI_BACK = 0 and W_CLOUD = 0
    # both fall out of this one formula, no special case.
    w_cloud = population_selection.law_dense_weight((xi_back - xi_front) * a_col_k)

    path = config_module.product_path(config, "bmstp", "shape", "cloud_interval", "sightline", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        f.attrs["D_FRONT_PC"] = float(d_front)
        f.attrs["D_BACK_PC"] = float(d_back)
        f.create_dataset("HPX_PIX_256", data=hpx_pix_256)
        f.create_dataset("XI_FRONT", data=xi_front.astype(np.float32))
        f.create_dataset("XI_BACK", data=xi_back.astype(np.float32))
        f.create_dataset("W_CLOUD", data=w_cloud.astype(np.float32))
    return path, n_sl, xi_front, xi_back, w_cloud


def build(config, regions=None):
    """Writes `bmstp/shape/cloud_interval_shape_sightline__R.hdf5` for
    `regions` (default all thirty), one file per region, directly after
    `bmstp.shapes` in the RUNBOOK (this module reads only what
    `bmstp.shapes`/`sample_cloud` already load; it writes nothing any other
    stage reads until `fittp.prior_reader` does)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        with progress.Stage("bmstp.cloud_interval", region) as st:
            path, n_sl, xi_front, xi_back, w_cloud = build_region(config, region, st)
            removed_frac = 1.0 - (xi_back - xi_front)
            st.done(path, n_sightline=n_sl,
                    xi_front_range=(float(xi_front.min()), float(xi_front.max())) if n_sl else (0.0, 0.0),
                    xi_back_range=(float(xi_back.min()), float(xi_back.max())) if n_sl else (0.0, 0.0),
                    w_cloud_range=(float(w_cloud.min()), float(w_cloud.max())) if n_sl else (0.0, 0.0),
                    removed_frac_median=float(np.median(removed_frac)) if n_sl else float("nan"))


if __name__ == "__main__":
    run(build)
