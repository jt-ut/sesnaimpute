"""The SWIRE galaxy sample, survey-wide (SPEC_BMSTP_DRAFT.md sec 3.2, sec 5.4;
SPEC_PRIORS.md sec 5.1's star-galaxy split).

A view of SWIRE alone: the six blank-field catalogues (Lonsdale et al. 2003,
PASP 115, 897; Surace et al. 2005 DR2 release), star-galaxy separated, kept
as one row per surviving galaxy with its 4.5um flux, its three IRAC colours
and their errors, and its bin on the flux grid the counts law
(`bms/gal/counts_gal_survey.hdf5`) is tabulated on. No density and no
colour grid are formed here: SPEC_BMSTP sec 5.4's GAL template weights form
a kernel density from these rows, per flux node, at read time.

The star-galaxy split reproduces the one `prior.gal.select_star_galaxy_split`
adopted for `bms/gal/counts_gal_survey.hdf5`: none of that module's split
candidates reproduces Fazio et al. 2004's star-subtracted counts within the
fitted cosmic-variance band, so the smallest-max-deviation candidate is the
one actually adopted -- IRAC 3.6um stellarity (SExtractor CLASS_STAR, Bertin
& Arnouts 1996, A&AS 117, 393) at or above `STELLARITY_STAR_MIN`, not the
per-band extended-flag rule (`prior.gal.classify_galaxy_extended_flag`,
which scored worse against Fazio's counts). Verified directly against
`bms/gal/counts_gal_survey.hdf5`'s own fit and cosmic-variance band before
adoption here.

Each IRAC colour is stored as `prior.gal`'s own internal convention,
log10(flux_a) - log10(flux_b), dex -- not a magnitude scaled by -2.5 --
because this is the exact quantity `prior.gal.build_colour_cdf_tables`
tabulates `bms/gal/counts_gal_survey.hdf5`'s CDF axes on; only this
convention lets this module's acceptance identity (the two CDFs agreeing
exactly for I1, up to the axis sign for I3/I4 -- see below) hold. The sign
convention is uniform, `COLOUR_AB = log10(F_A) - log10(F_B)`, for all three
pairs; `prior.gal`'s own I3/I4 axes are built on the opposite sign
(`log10(F_I3) - log10(F_I2)`, `log10(F_I4) - log10(F_I2)`), so this
module's acceptance check evaluates those two up to that sign flip, not a
literal equality of the raw arrays. `UNIT`/`DEFINITION` attrs on every
COLOUR_*/SIGMA_COLOUR_* dataset say so for a reader of the file alone.

A band is missing when its flux or its uncertainty is absent or a sentinel
(flux <= 0 or uncertainty <= 0, the SWIRE -99 sentinel included; no
bandwidth is ever formed from a sentinel): both the colour and the sigma
built from it are `NaN`, not +-inf. `N_FINITE_I1I2`/`I2I3`/`I2I4`/`ALL`
(file attrs) and `N_NODE_I1I2`/`N_NODE_ALL` ((61,) datasets, per S node)
report how many rows actually carry a usable value -- what `population`'s
GAL template-weight kernel (SPEC_BMSTP_DRAFT sec 5.4) can build from.
"""

import os

import h5py
import numpy as np
import pandas as pd

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: The six SWIRE fields' catalogue files, sky/download/swire/ (this
#: project's own `sesnaimpute.sky.download.swire.build`), in the same order
#: as `prior.gal.SWIRE_FIELD_FILES`.
SWIRE_FIELD_FILES = (
    "swire_elaisn1.csv", "swire_elaisn2.csv", "swire_elaiss1.csv",
    "swire_lockman.csv", "swire_xmmlss.csv", "swire_cdfs.csv",
)

#: SWIRE's own aperture-2 flux and uncertainty columns, 3.6/4.5/5.8/8.0um
#: (`sky.download.swire.build.COLUMNS`), uJy as distributed.
SWIRE_FLUX_COLUMNS = ("flux_ap2_36", "flux_ap2_45", "flux_ap2_58", "flux_ap2_80")
SWIRE_FLUX_ERR_COLUMNS = ("uncf_ap2_36", "uncf_ap2_45", "uncf_ap2_58", "uncf_ap2_80")

#: The adopted star-galaxy split (module docstring): IRAC 3.6um stellarity
#: (`prior.gal`'s `stell_36` column) at or above this threshold is a star.
#: An unmeasured stellarity is kept as a galaxy, `prior.gal.
#: classify_galaxy_stellarity`'s own conservative default.
STELLARITY_STAR_MIN = 0.95

#: The 61-node flux grid every GAL product is tabulated on (SPEC_BMSTP_DRAFT
#: sec 3.2; `prior.gal.build_log10_s_grid`): SWIRE's own I2 5-sigma depth
#: (log10 of 6.0 uJy in mJy) to Fazio et al. 2004 Table 1's brightest
#: tabulated 4.5um row -- the counts law's own tabulated range, held fixed
#: here so it matches `bms/gal/counts_gal_survey.hdf5`'s LOG10_S_GRID
#: exactly rather than being recomputed from the Fazio table again.
N_S_GRID = 61
LOG10_S_GRID_LO = -2.221848749616356
LOG10_S_GRID_HI = 1.2540644529143379

#: Per-dataset `UNIT`/`DEFINITION` attrs (owner ruling): a reader of the
#: file alone gets the convention without reading this module.
_COLOUR_DEFINITIONS = {
    "COLOUR_I1I2": "log10(F_I1) - log10(F_I2), IRAC fluxes in mJy",
    "COLOUR_I2I3": "log10(F_I2) - log10(F_I3), IRAC fluxes in mJy",
    "COLOUR_I2I4": "log10(F_I2) - log10(F_I4), IRAC fluxes in mJy",
    "SIGMA_COLOUR_I1I2": "sigma_f / (f ln 10), summed in quadrature over the two bands (I1, I2)",
    "SIGMA_COLOUR_I2I3": "sigma_f / (f ln 10), summed in quadrature over the two bands (I2, I3)",
    "SIGMA_COLOUR_I2I4": "sigma_f / (f ln 10), summed in quadrature over the two bands (I2, I4)",
}


def _read_field(path, field_index):
    """One SWIRE field CSV -> the surviving galaxies' rows (rule 10b: the
    catalogue is read one field at a time, never all six loaded together).
    A row survives if it is classed a galaxy by the adopted split and its
    4.5um (I2) flux is measured and positive -- the same population
    `prior.gal.build_colour_cdf_tables` tabulates the CDF axes on.
    """
    cols = list(SWIRE_FLUX_COLUMNS) + list(SWIRE_FLUX_ERR_COLUMNS) + ["stell_36"]
    df = pd.read_csv(path, usecols=cols)
    flux_mjy = df[list(SWIRE_FLUX_COLUMNS)].to_numpy(dtype=float) / 1000.0
    err_mjy = df[list(SWIRE_FLUX_ERR_COLUMNS)].to_numpy(dtype=float) / 1000.0
    stell36 = df["stell_36"].to_numpy(dtype=float)

    is_star = np.isfinite(stell36) & (stell36 >= STELLARITY_STAR_MIN)
    f1, f2, f3, f4 = flux_mjy[:, 0], flux_mjy[:, 1], flux_mjy[:, 2], flux_mjy[:, 3]
    valid_i2 = np.isfinite(f2) & (f2 > 0)
    keep = (~is_star) & valid_i2
    idx = np.flatnonzero(keep)

    f1k, f2k, f3k, f4k = f1[idx], f2[idx], f3[idx], f4[idx]
    e1, e2, e3, e4 = err_mjy[idx, 0], err_mjy[idx, 1], err_mjy[idx, 2], err_mjy[idx, 3]

    # A band is missing -- flux or its own uncertainty absent or a
    # sentinel (SWIRE's -99), never usable for a colour or a bandwidth --
    # so both go NaN for that band, not +-inf (owner ruling): a reader
    # forming a kernel bandwidth downstream must never see a sentinel.
    def band_valid(f, e):
        return np.isfinite(f) & (f > 0) & np.isfinite(e) & (e > 0)

    v1, v2, v3, v4 = band_valid(f1k, e1), band_valid(f2k, e2), band_valid(f3k, e3), band_valid(f4k, e4)

    def log10_or_nan(f, v):
        return np.where(v, np.log10(np.where(v, f, 1.0)), np.nan)

    log10_f1, log10_f2 = log10_or_nan(f1k, v1), log10_or_nan(f2k, v2)
    log10_f3, log10_f4 = log10_or_nan(f3k, v3), log10_or_nan(f4k, v4)

    ln10 = np.log(10.0)
    sigma1 = np.where(v1, e1 / (f1k * ln10), np.nan)
    sigma2 = np.where(v2, e2 / (f2k * ln10), np.nan)
    sigma3 = np.where(v3, e3 / (f3k * ln10), np.nan)
    sigma4 = np.where(v4, e4 / (f4k * ln10), np.nan)

    # NODE reads I2's flux alone (the split's own gate, `valid_i2`, already
    # guarantees it is finite and positive), independent of whether I2's
    # own uncertainty is a sentinel -- a bad error never removes a galaxy
    # from the flux grid, only from a colour or sigma that uses that band.
    log10_s_grid = np.linspace(LOG10_S_GRID_LO, LOG10_S_GRID_HI, N_S_GRID)
    edges = 0.5 * (log10_s_grid[1:] + log10_s_grid[:-1])
    log10_f2_flux = np.log10(f2k)
    in_range = (log10_f2_flux >= LOG10_S_GRID_LO) & (log10_f2_flux <= LOG10_S_GRID_HI)
    node = np.clip(np.searchsorted(edges, log10_f2_flux), 0, N_S_GRID - 1)
    node = np.where(in_range, node, -1).astype(np.int16)

    colour_i1i2 = (log10_f1 - log10_f2).astype(np.float32)
    colour_i2i3 = (log10_f2 - log10_f3).astype(np.float32)
    colour_i2i4 = (log10_f2 - log10_f4).astype(np.float32)

    return dict(
        LOG10_S=log10_f2_flux.astype(np.float32),
        NODE=node,
        COLOUR_I1I2=colour_i1i2,
        COLOUR_I2I3=colour_i2i3,
        COLOUR_I2I4=colour_i2i4,
        SIGMA_COLOUR_I1I2=np.sqrt(sigma1 ** 2 + sigma2 ** 2).astype(np.float32),
        SIGMA_COLOUR_I2I3=np.sqrt(sigma2 ** 2 + sigma3 ** 2).astype(np.float32),
        SIGMA_COLOUR_I2I4=np.sqrt(sigma2 ** 2 + sigma4 ** 2).astype(np.float32),
        FIELD=np.full(idx.size, field_index, dtype=np.int8),
    ), int(is_star.sum()), int(df.shape[0])


def build(config, regions=None):
    """Writes `sky/derived/swire/galaxies_swire_survey.hdf5`, one row per
    surviving galaxy across all six SWIRE fields. Survey-wide: `regions` is
    accepted for the standard `build` signature and ignored (rule 5c)."""
    if regions is not None:
        print("swire_galaxies: survey-wide product, --regions ignored")

    dest_dir = f"{config.data_root}/sky/download/swire"
    with progress_module.Stage("sky.derived.swire_galaxies") as st:
        field_blocks, n_stars_removed, n_fields = [], 0, len(SWIRE_FIELD_FILES)
        for i, name in enumerate(SWIRE_FIELD_FILES):
            path = f"{dest_dir}/{name}"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"swire_galaxies: no SWIRE field catalogue at {path!r} -- run the "
                    f"'sesnaimpute.sky.download.swire.build' RUNBOOK line")
            block, n_star, n_row = _read_field(path, i)
            field_blocks.append(block)
            n_stars_removed += n_star
            print(f"swire_galaxies: {name}: {n_row} rows, {n_star} stars removed, "
                  f"{block['LOG10_S'].size} galaxies kept")
            st.tick(i + 1, n_fields, "fields")

        columns = {k: np.concatenate([b[k] for b in field_blocks]) for k in field_blocks[0]}
        n_galaxies = int(columns["LOG10_S"].size)

        finite_i1i2 = np.isfinite(columns["COLOUR_I1I2"])
        finite_i2i3 = np.isfinite(columns["COLOUR_I2I3"])
        finite_i2i4 = np.isfinite(columns["COLOUR_I2I4"])
        finite_all = finite_i1i2 & finite_i2i3 & finite_i2i4
        node = columns["NODE"]
        n_node_i1i2 = np.zeros(N_S_GRID, dtype=np.int64)
        n_node_all = np.zeros(N_S_GRID, dtype=np.int64)
        in_grid = node >= 0
        np.add.at(n_node_i1i2, node[in_grid & finite_i1i2], 1)
        np.add.at(n_node_all, node[in_grid & finite_all], 1)

        out_path = config_module.product_path(config, "sky/derived", "swire", "galaxies", "survey")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            f.attrs["STELLARITY_STAR_MIN"] = STELLARITY_STAR_MIN
            f.attrs["FIELDS"] = ";".join(SWIRE_FIELD_FILES)
            f.attrs["N_STARS_REMOVED"] = n_stars_removed
            f.attrs["N_GALAXIES"] = n_galaxies
            f.attrs["N_FINITE_I1I2"] = int(finite_i1i2.sum())
            f.attrs["N_FINITE_I2I3"] = int(finite_i2i3.sum())
            f.attrs["N_FINITE_I2I4"] = int(finite_i2i4.sum())
            f.attrs["N_FINITE_ALL"] = int(finite_all.sum())
            f.create_dataset("LOG10_S_GRID",
                              data=np.linspace(LOG10_S_GRID_LO, LOG10_S_GRID_HI, N_S_GRID).astype(np.float64))
            f.create_dataset("N_NODE_I1I2", data=n_node_i1i2)
            f.create_dataset("N_NODE_ALL", data=n_node_all)
            for name, arr in columns.items():
                dset = f.create_dataset(name, data=arr)
                if name in _COLOUR_DEFINITIONS:
                    dset.attrs["UNIT"] = "dex"
                    dset.attrs["DEFINITION"] = _COLOUR_DEFINITIONS[name]

        st.done(out_path, n_galaxies=n_galaxies, n_stars_removed=n_stars_removed)
    return dict(n_galaxies=n_galaxies, n_stars_removed=n_stars_removed, path=out_path)


if __name__ == "__main__":
    build_module.run(build)
