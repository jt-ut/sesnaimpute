"""The region's sightline page (deck figure P1): the median source's own
cumulative extinction against distance, the depth fraction it implies,
and where the simulated field stars along that sightline sit.

Three panels share one distance axis. Top: `A_CUM_K`, the median
source's sightline row of the 3-D dust map's cumulative extinction
profile (`sky.derived.profile`/`profile_edenhofer_sightline`),
against `DIST_PC`. Middle: the depth fraction ``xi(d) = A_CUM_K /
A_INF_K``, the fraction of the sightline's whole column (the profile's
own far end, `A_INF_K`) that lies in front of distance `d` -- the same
normalisation `population.yso.embedding_and_ridge` gives the cloud
embedding, so `xi` reaches exactly 1 once every layer of the sightline
is behind the point. Both panels shade the region's cloud interval
(`sky.derived.profile`/`depth_edenhofer_region`'s `D_LO_PC`-
`D_HI_PC`) grey. Bottom: a histogram of the simulated field stars'
distances (`population.field_stars`/`field-stars_trilegal_region`,
`DIST_PC`) along the same sightline direction -- how many stars a model
of the Milky Way places at each distance, so a flat, dust-free stretch
of ``xi(d)`` in the middle panel can be read against how many stars
share it.

The median source is `atlas.shapes._select_source`'s own pick: the
catalogued source whose own column `A_COL_K` is closest to the region's
median.
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MaxNLocator

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import regions as regions_module

#: Grid conventions, page conventions (`_figure_conventions.md`).
_TITLE_FONTSIZE = 18
_LABEL_FONTSIZE = 13
_TICK_FONTSIZE = 10
_STATEMENT_FONTSIZE = 9

#: The distance axis every panel shares, pc.
_D_MIN, _D_MAX = 0.0, 2000.0

#: The bottom panel's bin width, pc.
_HIST_BIN_PC = 40.0

#: The middle panel's log axis, and the depth-fraction floor a value
#: below it plots as (matplotlib masks non-positive values on a log
#: axis, so a source's own zero-extinction start is simply not drawn --
#: the same as clipping to the floor).
_XI_MIN, _XI_MAX = 1e-3, 1.0

#: A dust-free stretch (the brief's own threshold): `dA/dd` below this,
#: mag/pc, and longer than `_FLAT_MIN_PC`, is reported.
_FLAT_DADD_MAG_PER_PC = 1e-4
_FLAT_MIN_PC = 40.0

#: One hue for the line/histogram (viridis, package convention: a
#: sequential colour map is viridis, a line is one hue).
_HUE = plt.get_cmap("viridis")(0.25)
_BAND_COLOR = "0.85"
_EDGE_COLOR = "0.35"


def _select_source(a_col):
    """The region's median source (`atlas.shapes._select_source`,
    unmodified): the catalogued source whose own `A_COL_K` is closest to
    the region's median."""
    return int(np.argmin(np.abs(a_col - np.median(a_col))))


def _read_source_table(config, region):
    path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    with h5py.File(path, "r") as f:
        return dict(
            name=f["NAME"][:],
            a_col=f["A_COL_K"][:].astype(np.float64),
            sightline_row=f["SIGHTLINE_ROW"][:].astype(np.int64),
        )


def _read_sightline_profile(config, region, row):
    path = config_module.product_path(config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    with h5py.File(path, "r") as f:
        dist_pc = f["DIST_PC"][:].astype(np.float64)
        a_cum_k = f["A_CUM_K"][row, :].astype(np.float64)
        a_inf_k = float(f["A_INF_K"][row])
    return dist_pc, a_cum_k, a_inf_k


def _read_cloud_interval(config, region):
    path = config_module.product_path(config, "sky/derived", "edenhofer", "depth", "region")
    with h5py.File(path, "r") as f:
        names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in f["REGION"][:]]
        if region not in names:
            raise ValueError("figure_sightline: region %r has no row in %s" % (region, path))
        i = names.index(region)
        d_lo = float(f["D_LO_PC"][i])
        d_hi = float(f["D_HI_PC"][i])
    return d_lo, d_hi


def _read_field_star_distances(config, region):
    path = config_module.product_path(config, "population", "trilegal", "field-stars", "region", region=region)
    with h5py.File(path, "r") as f:
        return f["DIST_PC"][:].astype(np.float64)


def _flat_stretches(dist_pc, a_cum_k, xi):
    """`(d_lo, d_hi, xi_mean)` for every dust-free stretch (module
    docstring): `|dA/dd| < _FLAT_DADD_MAG_PER_PC` over a run of grid
    cells spanning more than `_FLAT_MIN_PC`, `dA/dd` by `np.gradient`
    (the profile's own distance grid is not uniformly spaced)."""
    dadd = np.gradient(a_cum_k, dist_pc)
    flat = np.abs(dadd) < _FLAT_DADD_MAG_PER_PC
    idx = np.nonzero(flat)[0]
    stretches = []
    if idx.size == 0:
        return stretches
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        stretches.append((start, prev))
        start = prev = i
    stretches.append((start, prev))
    out = []
    for s, e in stretches:
        length = dist_pc[e] - dist_pc[s]
        if length > _FLAT_MIN_PC:
            out.append((float(dist_pc[s]), float(dist_pc[e]), float(xi[s:e + 1].mean())))
    return out


def build_region(config, region):
    src = _read_source_table(config, region)
    idx_median = _select_source(src["a_col"])
    name = src["name"][idx_median]
    name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
    a_s = float(src["a_col"][idx_median])
    row = int(src["sightline_row"][idx_median])

    dist_pc, a_cum_k, a_inf_k = _read_sightline_profile(config, region, row)
    xi = a_cum_k / a_inf_k
    d_lo, d_hi = _read_cloud_interval(config, region)
    field_dist_pc = _read_field_star_distances(config, region)
    stretches = _flat_stretches(dist_pc, a_cum_k, xi)

    plot_style.apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(16, 7), sharex=True)
    fig.subplots_adjust(top=0.91, bottom=0.20, hspace=0.08, left=0.07, right=0.98)

    for ax in (ax1, ax2):
        ax.axvspan(d_lo, d_hi, color=_BAND_COLOR, zorder=0)

    ax1.plot(dist_pc, a_cum_k, color=_HUE, lw=1.6)
    ax1.set_ylabel(r"$\mathbf{A_K}$ [mag]", fontsize=_LABEL_FONTSIZE)
    ax1.set_ylim(bottom=0.0)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))

    with np.errstate(divide="ignore", invalid="ignore"):
        ax2.plot(dist_pc, xi, color=_HUE, lw=1.6)
    ax2.axhline(1.0, color=_EDGE_COLOR, lw=0.9, linestyle=":", alpha=0.9)
    ax2.set_yscale("log")
    ax2.set_ylim(_XI_MIN, _XI_MAX)
    ax2.set_ylabel(r"$\mathbf{\xi}$", fontsize=_LABEL_FONTSIZE)
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))

    bins = np.arange(_D_MIN, _D_MAX + _HIST_BIN_PC, _HIST_BIN_PC)
    ax3.hist(field_dist_pc, bins=bins, color=_HUE, edgecolor="white", linewidth=0.3)
    ax3.set_ylabel("simulated stars per bin", fontsize=_LABEL_FONTSIZE)
    ax3.set_xlabel("Distance [pc]", fontsize=_LABEL_FONTSIZE)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=4))

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(_D_MIN, _D_MAX)
        ax.tick_params(labelsize=_TICK_FONTSIZE)

    fig.suptitle("%s Sightline" % region, fontsize=_TITLE_FONTSIZE, y=0.98)

    line1 = ("Extinction along the region's median sightline from the 3-D dust map of "
             "Edenhofer et al. 2023 (top), the depth fraction ξ = A(d) / A(∞) it")
    line2 = ("implies (middle), and the distances of the field stars a model of the "
             "Milky Way (TRILEGAL) places along it (bottom); the grey band is the cloud.")
    fig.text(0.5, 0.085, line1, fontsize=_STATEMENT_FONTSIZE, ha="center", va="bottom")
    fig.text(0.5, 0.03, line2, fontsize=_STATEMENT_FONTSIZE, ha="center", va="bottom")

    out_dir = f"{config.data_root}/sky/derived/edenhofer/figures"
    os.makedirs(out_dir, exist_ok=True)
    paths = [f"{out_dir}/sightline_edenhofer_region__{region}.png",
             f"{out_dir}/sightline_edenhofer_region__{region}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=150)
    plt.close(fig)

    print("figure_sightline %s: median source %s, A_s = %.4f mag, sightline A(inf) = %.4f mag" % (region, name, a_s, a_inf_k))
    print("figure_sightline %s: cloud interval D_LO_PC = %.1f, D_HI_PC = %.1f pc" % (region, d_lo, d_hi))
    if stretches:
        print("figure_sightline %s: dust-free stretches (> %d pc, dA/dd < %.0e mag/pc):"
              % (region, int(_FLAT_MIN_PC), _FLAT_DADD_MAG_PER_PC))
        for s_lo, s_hi, xi_mean in stretches:
            print("  d = %.1f-%.1f pc, xi ~= %.4f" % (s_lo, s_hi, xi_mean))
    else:
        print("figure_sightline %s: no dust-free stretch longer than %d pc" % (region, int(_FLAT_MIN_PC)))

    return paths


def build(config, regions=None):
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    paths = []
    for region in region_names:
        paths.extend(build_region(config, region))
    return paths


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()
    build(config_module.load(args.config), regions=args.regions)


if __name__ == "__main__":
    _main()
