"""The count check, one point per region, survey-wide (SPEC_BMSTP_DRAFT.md
sec. 8's "the prior's own predicted-over-observed check"): for each of the
thirty `bmstp.atlas` regions, the prior's `TOTAL_PREDICTED` over the
survey's own `TOTAL_OBSERVED` selected count -- the whole selected
population (filled marker) and the same ratio restricted to sources bright
enough that completeness is one on both sides of the check, `RATIO_BRIGHT3`
and `RATIO_BRIGHT10` (three and ten times the pixel's 4.5 um
half-detection flux; hollow markers).

`bmstp.atlas.build_region` writes, per region,
`bmstp/atlas/prior_atlas_hpx512__<R>.hdf5` with root attributes only:
`TOTAL_PREDICTED`, `TOTAL_OBSERVED`, `RATIO_BRIGHT3`, `RATIO_BRIGHT10`
and `SURVEYED_AREA_DEG2` (this page's four numbers plus the area each
marker's size encodes); no per-pixel arrays are read here.
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.transforms import blended_transform_factory

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute import regions as regions_module

PAGE_W_IN = 16.0
PAGE_H_IN = 6.0

_TITLE_FONTSIZE = 18
_LABEL_FONTSIZE = 13
_TICK_FONTSIZE = 10
_STATEMENT_FONTSIZE = 9

_Y_MIN = 0.3
_Y_MAX = 3.0
_Y_TICKS = (0.3, 0.5, 1.0, 2.0, 3.0)

#: Marker-size floor/ceiling in points^2 (`s` in `ax.scatter`), scaled by
#: the square root of `SURVEYED_AREA_DEG2` so the ~240x span in area across
#: the thirty regions (0.12 to 30 deg^2) reads as a mild size difference
#: rather than swamping the panel -- the smallest region's marker is still
#: >= 6 pt across (`sqrt(36) = 6`), matching the convention's marker floor.
_SIZE_MIN = 36.0
_SIZE_MAX = 200.0

_STATEMENT = (
    "The prior's expected number of selected sources over the survey's own count, region by "
    "region, for all sources (filled) and for sources bright enough that completeness is one "
    "(hollow, brighter than 3x and 10x the 4.5 um half-detection flux); the dashed line is the "
    "survey-wide ratio."
)


def _read_region(config, region):
    """One region's four attributes off `bmstp/atlas/prior_atlas_hpx512__
    <region>.hdf5` (module docstring), read-only."""
    path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        total_predicted = float(f.attrs["TOTAL_PREDICTED"])
        total_observed = float(f.attrs["TOTAL_OBSERVED"])
        ratio_bright3 = float(f.attrs["RATIO_BRIGHT3"])
        ratio_bright10 = float(f.attrs["RATIO_BRIGHT10"])
        area_deg2 = float(f.attrs["SURVEYED_AREA_DEG2"])
    return total_predicted, total_observed, ratio_bright3, ratio_bright10, area_deg2


def _read_all(config, region_names):
    """Every named region's four attributes, as parallel arrays plus the
    region names, unordered (the caller sorts)."""
    names, predicted, observed, bright3, bright10, area = [], [], [], [], [], []
    for region in region_names:
        tp, to, r3, r10, a = _read_region(config, region)
        names.append(region)
        predicted.append(tp)
        observed.append(to)
        bright3.append(r3)
        bright10.append(r10)
        area.append(a)
    return (np.array(names, dtype=object), np.array(predicted, dtype=np.float64),
            np.array(observed, dtype=np.float64), np.array(bright3, dtype=np.float64),
            np.array(bright10, dtype=np.float64), np.array(area, dtype=np.float64))


def _marker_sizes(area_deg2):
    """`_SIZE_MIN` to `_SIZE_MAX`, scaled by the square root of area
    (module docstring: a mild size difference across the 0.12-30 deg^2
    span, not a size proportional to area itself, which would let the
    largest region's marker swallow the panel)."""
    span = float(np.sqrt(area_deg2).max())
    frac = np.sqrt(area_deg2) / span if span > 0 else np.zeros_like(area_deg2)
    return _SIZE_MIN + (_SIZE_MAX - _SIZE_MIN) * frac


def build(config, regions=None, formats=("png", "pdf")):
    """Writes `bmstp/atlas/figures/count-check_atlas_survey.png/.pdf`: one
    panel, the ratio of the prior's predicted to the survey's observed
    selected count, region by region, ordered by `TOTAL_OBSERVED`
    descending (module docstring). `regions` defaults to every region
    `regions.REGIONS` names."""
    with progress.Stage("bmstp.figure_count_check") as st:
        plot_style.apply_style()

        region_names = list(regions) if regions is not None else [r.name for r in regions_module.REGIONS]
        names, predicted, observed, bright3, bright10, area = _read_all(config, region_names)

        order = np.argsort(-observed)
        names = names[order]
        predicted = predicted[order]
        observed = observed[order]
        bright3 = bright3[order]
        bright10 = bright10[order]
        area = area[order]

        ratio_all = predicted / observed
        survey_ratio = float(predicted.sum() / observed.sum())
        median_ratio = float(np.median(ratio_all))
        lo_idx = int(np.argmin(ratio_all))
        hi_idx = int(np.argmax(ratio_all))

        print("bmstp.figure_count_check: survey-wide ratio = %.4f (sum predicted / sum observed, "
              "%d regions)" % (survey_ratio, names.size))
        print("bmstp.figure_count_check: median region ratio = %.4f" % median_ratio)
        print("bmstp.figure_count_check: lowest = %s (%.4f); highest = %s (%.4f)"
              % (names[lo_idx], ratio_all[lo_idx], names[hi_idx], ratio_all[hi_idx]))
        for name, r_all, r3, r10 in zip(names, ratio_all, bright3, bright10):
            print("bmstp.figure_count_check: %-20s  all=%.4f  bright3=%.4f  bright10=%.4f"
                  % (name, r_all, r3, r10))

        fig = plot_style.new_sized_figure(PAGE_W_IN, PAGE_H_IN)
        ax = fig.add_axes([0.055, 0.30, 0.87, 0.56])

        x = np.arange(names.size)
        sizes = _marker_sizes(area)
        marker_color = "#31688e"  # viridis mid-blue, the one hue this page's markers share

        ax.axhline(1.0, color="black", lw=1.2, zorder=1)
        ax.axhline(survey_ratio, color="grey", lw=1.4, linestyle="--", zorder=1)
        blend = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.01, survey_ratio, "%.3f" % survey_ratio, color="grey", transform=blend,
                fontsize=_TICK_FONTSIZE, va="center", ha="left", fontweight="bold", clip_on=False)

        ax.scatter(x, ratio_all, s=sizes, marker="o", facecolor=marker_color,
                   edgecolor=marker_color, linewidth=1.0, zorder=3, label="all selected sources")
        ax.scatter(x, bright3, s=sizes, marker="^", facecolor="none",
                   edgecolor=marker_color, linewidth=1.3, zorder=3, label=r"bright $>3\times$")
        ax.scatter(x, bright10, s=sizes, marker="s", facecolor="none",
                   edgecolor=marker_color, linewidth=1.3, zorder=3, label=r"bright $>10\times$")
        ax.scatter([], [], s=_SIZE_MIN, marker="o", facecolor=marker_color, edgecolor=marker_color,
                   label="marker size $\\propto$ surveyed area (small to large)")
        ax.scatter([], [], s=_SIZE_MAX, marker="o", facecolor=marker_color, edgecolor=marker_color,
                   label="_nolegend_")

        ax.set_yscale("log")
        ax.set_ylim(_Y_MIN, _Y_MAX)
        ax.set_yticks(_Y_TICKS)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: ("%g" % v)))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_ylabel(plot_style.label(r"$\mathbf{predicted\,/\,observed}$"), fontsize=_LABEL_FONTSIZE)

        ax.set_xlim(-0.6, names.size - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(list(names), rotation=60, ha="right", fontsize=_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)
        ax.set_title("Count Check by Region", fontsize=_TITLE_FONTSIZE, pad=8)

        legend = ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9, ncol=1)
        plot_style.style_legend(legend)

        fig.text(0.5, 0.045, _STATEMENT, fontsize=_STATEMENT_FONTSIZE, ha="center", va="top", wrap=True)

        out_dir = os.path.join(config.data_root, "bmstp", "atlas", "figures")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for fmt in formats:
            path = os.path.join(out_dir, "count-check_atlas_survey.%s" % fmt)
            fig.savefig(path, dpi=150)
            paths.append(path)
        plt.close(fig)
        st.done(paths[0], survey_ratio=survey_ratio, n_regions=names.size)
    return paths


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()
    config = config_module.load(args.config)
    build(config, regions=args.regions)


if __name__ == "__main__":
    _main()
