"""The region depth-group grid (SPEC_PRIORS.md section 1.3,
IMPLEMENTATION.md section 4): a region's real per-source detection-limit
residual `Delta`, clustered by k-means into `K` groups, so a class's
exact selection (`prior.selection`) is tabulated once per group centre
instead of once per source.

Per source, `catalog.limits.limits` supplies the 8-band log10 detection
limit; `prior.selection.split_common_mode` divides it, against the
region's own per-band median (`REF_LOG10_FLIM`), into a common-mode
depth shift `s` and the four-degree-of-freedom residual `Delta` over the
five Spitzer bands. `fit_depth_groups` clusters the region's real
`Delta` vectors (`scipy.cluster.vq.kmeans2`, k-means++ init, fixed seed)
into `k_groups` centres and keeps a fixed-seed subsample of the real
`Delta` distribution for `selection.table_residual` to grade a fitted
table against.
"""

import os

import h5py
import numpy as np
from scipy.cluster.vq import kmeans2

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.prior import selection

#: `fit_depth_groups`'s default group count when a caller does not
#: search for a residual-clearing `K` -- also `selection.fit_by_
#: residual`'s own starting point before it doubles (owner ruling
#: 2026-09-04, SPEC_PRIORS.md 1.3).
K_GROUPS_DEFAULT = 16

#: The largest `delta_sample` `fit_depth_groups` stores -- the region's
#: real per-source `Delta` distribution `selection.table_residual` draws
#: from, capped so the product stays small.
_DELTA_SAMPLE_CAP = 4096


class DepthGroups:
    """One region's reference detection-limit vector and its per-band
    `Delta` group centres. `ref_log10_flim` is `(8,)`, `group_centres`
    is `(K, 5)` over `selection.BANDS_DEPTH`, `delta_sample` is a
    fixed-seed subsample (at most `_DELTA_SAMPLE_CAP` rows) of the
    region's own real per-source `Delta`.
    """

    def __init__(self, region, ref_log10_flim, group_centres, delta_sample, n_groups):
        self.region = region
        self.ref_log10_flim = np.asarray(ref_log10_flim, dtype=float)
        self.group_centres = np.asarray(group_centres, dtype=float)
        self.delta_sample = np.asarray(delta_sample, dtype=float)
        self.n_groups = int(n_groups)

    def assign_group(self, delta_5):
        """`(n,)`: the nearest group centre's index for each row of
        `delta_5` -- vectorised, no per-row Python.
        """
        delta_5 = np.atleast_2d(np.asarray(delta_5, dtype=float))
        centres = self.group_centres
        d2 = np.sum((delta_5[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
        return np.argmin(d2, axis=1)

    def write(self, path):
        """Upserts this region's group into the shared depth-groups
        file at `path`, creating it if absent.
        """
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        mode = "a" if os.path.exists(path) else "w"
        with h5py.File(path, mode) as f:
            f.attrs["GRANULE"] = "region"
            if self.region in f:
                del f[self.region]
            g = f.create_group(self.region)
            g.attrs["N_GROUPS"] = self.n_groups
            g.create_dataset("REF_LOG10_FLIM", data=self.ref_log10_flim)
            g.create_dataset("GROUP_CENTRES", data=self.group_centres)
            g.create_dataset(
                "DELTA_SAMPLE",
                data=self.delta_sample.reshape(-1, len(selection.BANDS_DEPTH)))

    @classmethod
    def read(cls, path, region):
        if not os.path.exists(path):
            raise ValueError(
                f"the depth-groups product is missing: {path!r} -- build it "
                f"with prior.depth_groups.build")
        with h5py.File(path, "r") as f:
            if region not in f:
                raise ValueError(f"{path!r} carries no depth-groups group for {region!r}")
            g = f[region]
            return cls(
                region=region,
                ref_log10_flim=g["REF_LOG10_FLIM"][:],
                group_centres=g["GROUP_CENTRES"][:],
                delta_sample=g["DELTA_SAMPLE"][:],
                n_groups=int(g.attrs["N_GROUPS"]),
            )


def fit_depth_groups(config, region, k_groups=K_GROUPS_DEFAULT, seed=0):
    """Measures `region`'s reference detection-limit vector and clusters
    its own catalogued sources' `Delta` into `k_groups` groups
    (`scipy.cluster.vq.kmeans2`, k-means++ init, fixed `seed`).
    """
    k_groups = int(k_groups)
    if k_groups < 1:
        raise ValueError(f"fit_depth_groups: k_groups must be >= 1, got {k_groups}")

    f_lim = limits_module.limits(config, region)
    log10_flim_8 = np.log10(f_lim)
    ref_log10_flim = np.median(log10_flim_8, axis=0)
    _s, delta5 = selection.split_common_mode(log10_flim_8, ref_log10_flim)

    n_sources = delta5.shape[0]
    k_eff = max(1, min(k_groups, n_sources))
    group_centres, labels = kmeans2(delta5, k_eff, minit="++", seed=seed)

    rng_sample = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    cap = min(n_sources, _DELTA_SAMPLE_CAP)
    sample_idx = rng_sample.choice(n_sources, size=cap, replace=False)
    delta_sample = delta5[sample_idx]

    return DepthGroups(
        region=region,
        ref_log10_flim=ref_log10_flim,
        group_centres=group_centres,
        delta_sample=delta_sample,
        n_groups=k_eff,
    )


def build(config, regions=None):
    """Writes the depth-group grid for `regions` (default: all thirty)
    into the shared `bms/sesna/depth-groups_sesna_region.hdf5` product,
    one h5 group per region.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    path = config_module.product_path(config, "bms", "sesna", "depth-groups", "region")
    for region in region_names:
        knots = fit_depth_groups(config, region)
        knots.write(path)


if __name__ == "__main__":
    run(build)
