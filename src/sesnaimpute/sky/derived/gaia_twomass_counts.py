"""The Gaia archive's own Gaia-2MASS crossmatch, reshaped onto the
region's exact nside-512 pixel set (SPEC_BMSTP_DRAFT.md section 5.1,
"joint and marginal bins" row): the joint `(G, Ks)` anchor count, an
external, whole-pixel view like the two marginals it sits beside
(`sky.derived.gaia_counts`, `sky.derived.twomass_counts`), in place of
`population.anchor_observed`'s earlier joint histogram built from
SESNA's own partial-pixel sources (`studies/star_count_chain.md`).

Derive never fetches: this reads `sky/download/gaia_twomass_counts/
joint-counts_gaia-twomass_hpx512__<Region>.csv` (already a per-pixel,
per-G-bin, per-Ks-bin count table -- the TAP service's own GROUP BY, see
that module's docstring) and the granule map, and does exactly one
computation -- restricting the download's bounding-box superset of
pixels to the region's own occupied nside-512 set and depositing the
counts into a dense `(n_pix, n_G_bin, n_Ks_bin)` array, zero where the
download carried no row for a (pixel, G bin, Ks bin) triple. A download
row repeated across two Galactic-latitude strips (that module's own
throttle fallback) is summed here by the same `np.add.at` accumulation
`sky.derived.gaia_counts`/`twomass_counts` already use for a download's
own duplicate rows.

`G_EDGES` is the download's own 1-mag Gaia grid, `10..19`
(`sky.download.gaia_twomass_counts.G_LIMIT`), NOT `sky.derived.gaia_
counts.MAG_EDGES` (`10..21`, the wider Gaia marginal grid): the joint
table's own spec cut is `G < 19` (SPEC_BMSTP_DRAFT.md 5.1). `KS_EDGES`
is `sky.derived.twomass_counts.MAG_EDGES` verbatim -- the same half-
magnitude grid to 14.3, so a `population.anchor_observed` join against
either marginal lands on shared bin edges by construction, not by a
separate check.
"""

import os

import h5py
import numpy as np

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access
from sesnaimpute.sky.derived import twomass_counts as twomass_derived

G_EDGES = np.arange(10.0, 20.0, 1.0)  # SPEC_BMSTP_DRAFT.md 5.1 "joint and marginal bins": G < 19
KS_EDGES = twomass_derived.MAG_EDGES  # the 2MASS marginal's own grid, 9.0-14.3 (module docstring)


def _download_path(config, region):
    return f"{config.data_root}/sky/download/gaia_twomass_counts/joint-counts_gaia-twomass_hpx512__{region}.csv"


def _region_pixels(config, region):
    """The region's own occupied nside-512 pixels, sorted, from the granule map."""
    return np.unique(access.region_slice(config, region)["hpx_pix_512"])


def _counts_cube(pixels, csv_path):
    """`(n_pix, n_G_bin, n_Ks_bin)` int64: the download's per-pixel-per-
    G-bin-per-Ks-bin rows restricted to `pixels` (the region's own set),
    a row's Ks bin index read directly from its own half-magnitude
    `ks_bin_lo` (the same arithmetic the download's own `FLOOR(2*ks_m)/2`
    inverts: `round((ks_bin_lo - 9.0) * 2)`, landing on `KS_EDGES`' index
    exactly since both share the same 0.5-mag grid from 9.0)."""
    n_pix, n_g, n_ks = pixels.size, G_EDGES.size - 1, KS_EDGES.size - 1
    n = np.zeros((n_pix, n_g, n_ks), dtype=np.int64)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"sky.derived.gaia_twomass_counts: no download CSV at {csv_path} -- run the "
            "'sesnaimpute.sky.download.gaia_twomass_counts.build' RUNBOOK line first")
    table = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
    if table.size == 0:
        return n
    row_pix = table["hpx_pix_512"].astype(np.int64)
    g_idx = table["g_bin_lo"].astype(np.int64) - 10
    ks_idx = np.round((table["ks_bin_lo"].astype(np.float64) - 9.0) * 2.0).astype(np.int64)
    row_n = table["n"].astype(np.int64)
    loc = np.searchsorted(pixels, row_pix)
    in_set = (loc < n_pix) & (pixels[np.minimum(loc, n_pix - 1)] == row_pix)
    in_g = (g_idx >= 0) & (g_idx < n_g)
    in_ks = (ks_idx >= 0) & (ks_idx < n_ks)
    keep = in_set & in_g & in_ks
    np.add.at(n, (loc[keep], g_idx[keep], ks_idx[keep]), row_n[keep])
    return n


def _write_region(config, region, pixels, n):
    out_path = f"{config.data_root}/sky/derived/gaia/joint-counts_gaia-twomass_hpx512__{region}.hdf5"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.create_dataset("HPX_PIX_512", data=pixels.astype(np.int64))
        f.create_dataset("G_EDGES", data=G_EDGES.astype(np.float64))
        f.create_dataset("KS_EDGES", data=KS_EDGES.astype(np.float64))
        f.create_dataset("N_GK", data=n.astype(np.int64))


def build(config, regions=None):
    """Writes `sky/derived/gaia/joint-counts_gaia-twomass_hpx512__
    <Region>.hdf5` for each requested region (default: all thirty)."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    written = []
    with progress_module.Stage("sky.derived.gaia_twomass_counts") as st:
        n_regions = len(regions)
        total_counts = 0
        for i, region in enumerate(regions):
            pixels = _region_pixels(config, region)
            n = _counts_cube(pixels, _download_path(config, region))
            _write_region(config, region, pixels, n)
            written.append(region)
            total_counts += int(n.sum())
            print(f"sky.derived.gaia_twomass_counts: {region}: {pixels.size} pixels, {int(n.sum())} joint counts")
            st.tick(i + 1, n_regions, "regions")
        st.done(None, regions=n_regions, total_counts=total_counts)
    return dict(regions_written=written)


if __name__ == "__main__":
    build_module.run(build)
