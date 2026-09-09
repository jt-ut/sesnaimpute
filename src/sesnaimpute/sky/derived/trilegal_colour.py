"""TRILEGAL's own Gaia G - 2MASS Ks colour, as a function of atmosphere
alone (`log T_eff`, `log g`, `[M/H]`), from the one query
`sky.download.trilegal.colour` ran in its Gaia+Tycho2+2MASS system
(module docstring there). A star's colour at fixed atmosphere is the same
synthesis anywhere on the sky, so this one field's table stands in for
every region's own TRILEGAL run, which carries no Gaia band of its own
(W42): `population.field_stars.G_PROXY` reads it in place of the
fitter's atmosphere-library `G - Ks`.

A view of external data, no model: the median `G - Ks` (and its 16-84 %
half-width and star count) in cells of `log T_eff` (0.02 dex), `log g`
(0.25 dex) and `[M/H]` (0.25 dex), from the download file's own columns.
Cells with no star are NaN. Written once, survey-wide
(`GRANULE = "survey"`) -- nothing here is per-region.
"""

import os

import h5py
import numpy as np
import pandas as pd

from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute.build import run

#: TRILEGAL's own column header for the Gaia+Tycho2+2MASS system
#: (`sky.download.trilegal.colour`), verbatim; a header that does not
#: match this is a service format change, not a silent misread.
TRILEGAL_COLOUR_COLUMNS = (
    "Gc", "logAge", "[M/H]", "m_ini", "logL", "logTe", "logg",
    "m-M0", "Av", "m2/m1", "mbol", "G", "G_BP", "G_RP", "B_T", "V_T",
    "J", "H", "Ks", "Mact",
)

#: The three axes' bin widths (dex), the brief's own grid.
BIN_LOG_TEFF = 0.02
BIN_LOG_G = 0.25
BIN_MH = 0.25


def _download_path(config):
    return f"{config.data_root}/sky/download/trilegal/colour_gaiaDR2_2mass_perseus.dat"


def read_download(path):
    """The one query's table as a DataFrame, header-checked."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"trilegal_colour derive: no download at {path} -- run the "
            "'sesnaimpute.sky.download.trilegal.colour' RUNBOOK line first")
    with open(path, "r") as fh:
        header = fh.readline()
    names = tuple(header.lstrip("#").split())
    if names != TRILEGAL_COLOUR_COLUMNS:
        raise ValueError(f"{path}: unexpected TRILEGAL columns {names}")
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=names, engine="c")
    df = df.dropna(subset=["m-M0"])
    if len(df) == 0:
        raise ValueError(f"{path}: no rows")
    return df


def _edges(values, step):
    """Bin edges of width `step`, aligned to multiples of `step`, spanning
    `values`' own range (with a small pad so the maximum value falls
    strictly inside the last bin)."""
    lo = np.floor(values.min() / step) * step
    hi = np.ceil(values.max() / step) * step + step
    return np.arange(lo, hi + step * 0.5, step)


def colour_table(df):
    """The median `G - Ks`, its 16-84 % half-width and star count per
    populated `(log T_eff, log g, [M/H])` cell -- three arrays, all
    `(n_teff, n_g, n_mh)`, from the download file's own columns; an
    empty cell is NaN in all three. Fully vectorised: one bin index per
    axis per star, one `np.add.at`/percentile pass per populated cell.
    """
    log_teff = df["logTe"].to_numpy(dtype=np.float64)
    log_g = df["logg"].to_numpy(dtype=np.float64)
    mh = df["[M/H]"].to_numpy(dtype=np.float64)
    g_minus_ks = (df["G"] - df["Ks"]).to_numpy(dtype=np.float64)

    e_teff = _edges(log_teff, BIN_LOG_TEFF)
    e_g = _edges(log_g, BIN_LOG_G)
    e_mh = _edges(mh, BIN_MH)
    shape = (e_teff.size - 1, e_g.size - 1, e_mh.size - 1)

    i_teff = np.clip(np.searchsorted(e_teff, log_teff, side="right") - 1, 0, shape[0] - 1)
    i_g = np.clip(np.searchsorted(e_g, log_g, side="right") - 1, 0, shape[1] - 1)
    i_mh = np.clip(np.searchsorted(e_mh, mh, side="right") - 1, 0, shape[2] - 1)
    flat = np.ravel_multi_index((i_teff, i_g, i_mh), shape)

    median = np.full(np.prod(shape), np.nan, dtype=np.float64)
    halfwidth = np.full(np.prod(shape), np.nan, dtype=np.float64)
    count = np.zeros(np.prod(shape), dtype=np.int64)
    order = np.argsort(flat)
    flat_sorted = flat[order]
    gk_sorted = g_minus_ks[order]
    # one boundary-aware pass over the populated cells only -- no Python
    # loop over stars, one iteration per OCCUPIED cell (rule 8, 9).
    boundaries = np.flatnonzero(np.diff(flat_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [flat_sorted.size]))
    for start, stop in zip(starts, stops):
        cell = flat_sorted[start]
        vals = gk_sorted[start:stop]
        median[cell] = np.median(vals)
        p16, p84 = np.percentile(vals, [16.0, 84.0])
        halfwidth[cell] = 0.5 * (p84 - p16)
        count[cell] = vals.size

    return dict(
        log_teff_edges=e_teff, log_g_edges=e_g, mh_edges=e_mh,
        median=median.reshape(shape), halfwidth=halfwidth.reshape(shape),
        count=count.reshape(shape),
    )


def write_table(path, table, n_stars):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.attrs["N_STARS"] = int(n_stars)
        f.create_dataset("LOG_TEFF_EDGES", data=table["log_teff_edges"].astype(np.float64))
        f.create_dataset("LOG_G_EDGES", data=table["log_g_edges"].astype(np.float64))
        f.create_dataset("MH_EDGES", data=table["mh_edges"].astype(np.float64))
        f.create_dataset("G_MINUS_KS_MEDIAN", data=table["median"].astype(np.float32))
        f.create_dataset("G_MINUS_KS_HALFWIDTH", data=table["halfwidth"].astype(np.float32))
        f.create_dataset("COUNT", data=table["count"].astype(np.int32))


def build(config, regions=None):
    """Writes `sky/derived/trilegal/colour_trilegal_survey.hdf5`.
    `regions` is accepted for the common build entry-point convention but
    ignored -- this product is survey-wide, not per region.
    """
    out_path = config_module.product_path(config, "sky/derived", "trilegal", "colour", "survey")
    with progress_module.Stage("sky.derived.trilegal_colour") as st:
        if os.path.exists(out_path):
            print(f"trilegal_colour derive: {out_path} present, skipped")
            st.done(out_path, skipped=1)
            return
        df = read_download(_download_path(config))
        table = colour_table(df)
        write_table(out_path, table, len(df))
        n_cells = table["median"].size
        n_populated = int(np.isfinite(table["median"]).sum())
        st.done(out_path, n_stars=len(df), n_cells=n_cells, n_populated=n_populated,
                fill_frac=n_populated / n_cells)


if __name__ == "__main__":
    run(build)
