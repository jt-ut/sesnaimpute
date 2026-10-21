"""Marshall et al. 2006's 3-D extinction curve, per admitted sightline
(briefs/round4_2026-10-20/C25_far_column.md).

C24 (bms_review/briefs/round4_2026-10-20/C24_locate_red_tail.md) located the
missing red tail of distant background stars: beyond the Edenhofer et al.
2023 cube's own edge, `sky.derived.profile.a_of_d` caps a star's column at
the pixel's own cloud column `A_PIX_K` -- in the Galactic plane, where the
cube's 2 kpc reach is about half the full line of sight, that undercounts
what a star at 4-8 kpc really sees. Marshall et al. 2006 (2MASS red giants,
VizieR J/A+A/453/635, `sky.download.marshall2006`) measured `A_Ks(d)` to
~10 kpc on a 15-arcmin grid covering `|l| <= 100`, `|b| <= 10`; this module
reads that table once and, for every region's own admitted nside-256
sightline (`sky.derived.profile`'s own product, read for its `DIST_PC`
grid and sky position so this module never re-derives admission), carries
the nearest cell's curve onto the SAME distance grid `sky.derived.profile`
is tabulated on -- linear in distance between Marshall's own bins (plus
the implicit origin `A_Ks(0) = 0`), held at the last bin's own value past
it, so `sky.derived.profile._RegionProfile.a_of_d` can read `A_KS` and
`DIST_PC` off this file row for row with `A_CUM_K`, no re-interpolation
there.

`COVERED` is 0 (and `A_KS` all zero) wherever the sightline sits outside
Marshall's own box (`|l| <= 100`, `|b| <= 10`) OR its nearest cell is
farther than one cell width (15 arcmin) away -- a sparse patch of the box
Marshall's own grid does not actually reach. Every region gets this
product, uncovered regions all-zero/`COVERED` all False, so a consumer
never has to branch on whether the file exists (module docstring
precedent: every `sky.derived` product this package writes, written for
every region in the survey).
"""

import gzip
import os

import h5py
import healpy as hp
import numpy as np
from scipy.spatial import cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.config import product_path

_STEM = "extinction_marshall_sightline"

#: Marshall et al. 2006's own survey box (module docstring, their sec. 2).
L_MAX_DEG = 100.0
B_MAX_DEG = 10.0

#: Marshall et al. 2006's own cell size, 15 arcmin, in degrees -- "farther
#: than one cell" (module docstring) is this separation.
CELL_DEG = 15.0 / 60.0

#: The table's own maximum distance-bin count (VizieR J/A+A/453/635
#: `table1.dat`'s own `nb` column, 1-33 -- `c18_diag/c24_tests.py`'s T3/T4
#: construction, reused exactly).
_N_BIN_MAX = 33


def _input_dir(config):
    return f"{config.data_root}/sky/download/marshall2006"


_TABLE_CACHE = {}


def load_table(config):
    """`(l_deg, b_deg, r_kpc, a_ks_mag)`, each `(n_cell,)` or `(n_cell,
    33)`: Marshall et al. 2006's own per-cell distance bins and their
    `A_Ks`, NaN past a cell's own `nb` (`c18_diag/c24_tests.py`'s T3/T4
    byte layout, reused exactly -- see the ReadMe's byte-by-byte
    description). Cached per process: a pure function of the download,
    like `population.anchor_tiles.gaia_selection`."""
    path = _input_dir(config)
    if path in _TABLE_CACHE:
        return _TABLE_CACHE[path]
    readme = f"{path}/ReadMe"
    table = f"{path}/table1.dat.gz"
    for p in (readme, table):
        if not os.path.exists(p):
            raise FileNotFoundError(
                "sky.derived.marshall_extinction: Marshall et al. 2006 input missing at %s "
                "-- run the sesnaimpute.sky.download.marshall2006.build RUNBOOK line for it" % p)
    gl, gb, r_kpc, a_ks = [], [], [], []
    with gzip.open(table, "rt") as fh:
        for line in fh:
            l = float(line[1:7])
            b = float(line[8:14])
            nb = int(line[26:29])
            r = np.full(_N_BIN_MAX, np.nan)
            a = np.full(_N_BIN_MAX, np.nan)
            for i in range(nb):
                s = 29 + 24 * i
                try:
                    r[i] = float(line[s:s + 6])
                    a[i] = float(line[s + 6:s + 12])
                except ValueError:
                    pass
            gl.append(l)
            gb.append(b)
            r_kpc.append(r)
            a_ks.append(a)
    gl, gb, r_kpc, a_ks = (np.asarray(x, dtype=np.float64) for x in (gl, gb, r_kpc, a_ks))
    _TABLE_CACHE[path] = (gl, gb, r_kpc, a_ks)
    return gl, gb, r_kpc, a_ks


def nearest_cell(cell_l, cell_b, l_deg, b_deg):
    """`(cell_index, separation_deg)`: for each `(l_deg, b_deg)`, the
    nearest Marshall cell by plain angular separation (`c18_diag/
    c24_tests.py`'s `marshall_cells`, reused exactly)."""
    tree = cKDTree(hp.ang2vec(cell_l, cell_b, lonlat=True))
    chord, idx = tree.query(hp.ang2vec(l_deg, b_deg, lonlat=True))
    sep_deg = np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))
    return idx, sep_deg


def curve_on_grid(r_kpc, a_ks, cell_idx, dist_pc_grid):
    """`A_Ks(d)` on `dist_pc_grid` (pc), one row per `cell_idx` entry:
    piecewise-linear in distance through `(0, 0)` and the cell's own
    finite bins (sorted by distance), held at the last bin's own value
    past it (`c18_diag/c24_tests.py`'s `marshall_a`, reused exactly)."""
    d_kpc = np.asarray(dist_pc_grid, dtype=np.float64) / 1000.0
    out = np.empty((cell_idx.size, d_kpc.size), dtype=np.float64)
    for row, c in enumerate(cell_idx):
        r = r_kpc[c]
        a = a_ks[c]
        ok = np.isfinite(r) & np.isfinite(a)
        rr = np.concatenate([[0.0], r[ok]])
        aa = np.concatenate([[0.0], a[ok]])
        order = np.argsort(rr)
        out[row] = np.interp(d_kpc, rr[order], aa[order])
    return out


def _in_box(l_deg, b_deg):
    """Per-sightline test of Marshall's own survey box, `|l| <= 100`
    (wrapped: `l in [0, 100]` or `l in [260, 360]`), `|b| <= 10`."""
    l_deg = np.asarray(l_deg, dtype=np.float64)
    b_deg = np.asarray(b_deg, dtype=np.float64)
    return ((l_deg <= L_MAX_DEG) | (l_deg >= 360.0 - L_MAX_DEG)) & (np.abs(b_deg) <= B_MAX_DEG)


def _build_one_region(config, region, cell_l, cell_b, r_kpc, a_ks):
    profile_path = product_path(config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            "sky.derived.marshall_extinction: %s missing -- run the "
            "sesnaimpute.sky.derived.profile RUNBOOK line for this region first "
            "(this module reads its own DIST_PC grid and admitted sightlines)" % profile_path)
    with h5py.File(profile_path, "r") as f:
        hpx = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        dist_pc = np.asarray(f["DIST_PC"][:], dtype=np.float64)
        gl = np.asarray(f["GAL_L_DEG"][:], dtype=np.float64)
        gb = np.asarray(f["GAL_B_DEG"][:], dtype=np.float64)

    in_box = _in_box(gl, gb)
    cell_idx = np.full(hpx.size, -1, dtype=np.int64)
    sep_deg = np.full(hpx.size, np.nan, dtype=np.float64)
    a_ks_grid = np.zeros((hpx.size, dist_pc.size), dtype=np.float64)
    if np.any(in_box):
        idx, sep = nearest_cell(cell_l, cell_b, gl[in_box], gb[in_box])
        cell_idx[in_box] = idx
        sep_deg[in_box] = sep
    covered = in_box & (sep_deg <= CELL_DEG)
    if np.any(covered):
        a_ks_grid[covered] = curve_on_grid(r_kpc, a_ks, cell_idx[covered], dist_pc)

    out_path = product_path(config, "sky/derived", "marshall", "extinction", "sightline", region=region)
    real_root = os.path.realpath(config.data_root)
    if not os.path.realpath(out_path).startswith(real_root + os.sep):
        raise ValueError("sky.derived.marshall_extinction: refusing to write outside the data "
                          "root %s: %s" % (real_root, out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        for name, data in (
            ("HPX_PIX_256", hpx),
            ("DIST_PC", dist_pc),
            ("A_K", a_ks_grid),
            ("COVERED", covered),
            ("SEPARATION_DEG", sep_deg),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])
    return dict(region=region, n_sightline=int(hpx.size), n_covered=int(np.count_nonzero(covered)),
                median_sep_deg=float(np.nanmedian(sep_deg[covered])) if np.any(covered) else float("nan"))


def build(config, regions=None):
    """Writes each region's `extinction_marshall_sightline` product for
    `regions` (default: all thirty) -- every region, covered or not
    (module docstring), one `Stage` per region."""
    import time
    names = regions or [r.name for r in regions_module.REGIONS]
    cell_l, cell_b, r_kpc, a_ks = load_table(config)
    t0 = time.time()
    for region in names:
        with progress_module.Stage("sky.derived.marshall_extinction", region) as st:
            row = _build_one_region(config, region, cell_l, cell_b, r_kpc, a_ks)
            path = product_path(config, "sky/derived", "marshall", "extinction", "sightline", region=region)
            st.done(path, n_sightline=row["n_sightline"], n_covered=row["n_covered"],
                    median_sep_deg=row["median_sep_deg"])
        print("marshall_extinction: %s: %d/%d sightlines covered, median separation %.3f deg"
              % (region, row["n_covered"], row["n_sightline"], row["median_sep_deg"]))
    progress_module.summary("sky.derived.marshall_extinction", len(names), t0)


if __name__ == "__main__":
    from sesnaimpute.build import run
    run(build)
