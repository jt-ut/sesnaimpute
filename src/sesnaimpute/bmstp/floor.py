"""The common floor `Lambda_floor(s)`, once per region (SPEC_BMSTP_DRAFT.md
section 4.1's common-floor rule, `fittp.prior_reader.common_floor`):
`Lambda_floor(s) = grid.FLOOR * max` over the six classes' own
`peak_density(s)`, so an empty prior cell reads the SAME density for
every class and the likelihood alone decides between them. It depends
only on the prior chain's own products (P1's density table, P2-P4's shape
grids, P5's template weights) and the source list -- nothing in it
depends on the fit -- so it belongs here, in the prior chain, rather than
being recomputed as `fittp.sweep`'s own six-class first pass on every one
of the 180 independent {class, region} cluster jobs. The floor's own
value is unchanged; only where it is computed moves.
"""

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.bmstp import grid
from sesnaimpute.fittp import prior_reader

#: the six class codes, the order `common_floor` maxes over
CLASSES = tuple(c.code for c in definitions.CLASSES)


def build_region(config, region):
    """`LAMBDA_FLOOR` `(n_source,)` float64 for `region`: each of the six
    classes' own `Prior` (`fittp.prior_reader.load`), that class's peak
    cell density over every source (`prior_reader.peak_density`, the
    unblurred grain peak times the class's own retained density), maxed
    over classes and scaled by `grid.FLOOR` (`prior_reader.common_floor`).
    Always all six classes, regardless of which classes a later fit run
    sweeps: the floor is common by construction only if every class's own
    P2-P5 products exist and are read here."""
    peaks = []
    for cls in CLASSES:
        reader = prior_reader.load(config, region, cls)
        peaks.append(prior_reader.peak_density(reader, np.arange(reader.density.size)))
    return prior_reader.common_floor(peaks)


def build(config, regions=None):
    """Writes `bmstp/density/floor_density_source[__R].hdf5` for `regions`
    (default all thirty regions, rule 5c), one file per region, rows
    aligned by source with `bmstp/density/table_density_source__R.hdf5`
    (P1)."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        with progress.Stage("bmstp.floor", region) as st:
            lambda_floor = build_region(config, region)
            path = config_module.product_path(
                config, "bmstp", "density", "floor", "source", region=region)
            with h5py.File(path, "w") as f:
                f.attrs["GRANULE"] = "source"
                f.attrs["CLASSES"] = np.array(CLASSES, dtype="S8")
                f.attrs["FLOOR"] = grid.FLOOR
                f.create_dataset("LAMBDA_FLOOR", data=lambda_floor.astype(np.float64))
            st.done(path, n=lambda_floor.size,
                    lambda_floor_median=float(np.median(lambda_floor)),
                    lambda_floor_max=float(np.max(lambda_floor)))


if __name__ == "__main__":
    run(build)
