"""The STAR-anchor calibration figure: a region's observed-versus-
predicted star counts, per magnitude bin, on Gaia G (stars in front of
the cloud) and 2MASS Ks (stars seen through it), with the adopted
reweighting factor `W` drawn beneath each panel (P2 brief;
`_figure_conventions.md`).

Reads three products `population.anchor_tiles`, `population.young_stars`
and `population.anchor_weights` already wrote, per region:

  `histograms_anchors_hpx512__<Region>.hdf5` -- `G_EDGES`, `KS_EDGES`,
  `N_G_OBS`/`N_G_PRED` (n_pix, n_G), `N_KS_OBS`/`N_KS_PRED` (n_pix, n_Ks).
  A region's deep-survey (UKIDSS) Ks bins and the pixels the deep survey
  does not cover are already zeroed upstream (`anchor_tiles.
  _combine_ks_axis`/`build_region`), so summing every pixel's own row
  already respects `DEEP_COVERED` -- no further masking is done here.

  `young-stars_anchors_hpx512__<Region>.hdf5` -- `N_G_YOUNG`/`N_KS_YOUNG`
  (n_pix, n_G/n_Ks), the model's own expectation of young stars in each
  pixel and bin, subtracted (elsewhere, before a weight is ever fit) from
  the observed counts above.

  `weights_anchors_tile__<Region>.hdf5` -- `W_REGION_G`/`W_REGION_KS`
  (n_G/n_Ks), the region-pooled adopted weight per bin
  (`anchor_weights.fit_tile_weights`'s `w_region`) the calibration
  actually uses.

Each panel sums its histogram's own per-pixel counts over the region's
whole pixel set to one region-total step histogram per bin, plotted with
observed, predicted (TRILEGAL, before calibration) and young-star counts
together; a strip below shares the same magnitude axis and shows
`W_REGION_G`/`W_REGION_KS` as points against a line at 1.
"""

import argparse
import os

import h5py
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import regions as regions_module

#: page size the figure conventions name for a product figure
#: (`_figure_conventions.md`: "16 x 6 in unless the content needs
#: another shape" -- this one does not).
_PAGE_WIDTH_IN = 16.0
_PAGE_HEIGHT_IN = 6.0

#: the one hue used for the young-star line and the weight-strip points
#: (`_figure_conventions.md`: "a line is one hue"), a mid viridis so it
#: reads as a derived quantity distinct from the black observed/grey
#: predicted pair.
_HUE_COLOR = "#375A8C"
_OBS_COLOR = "black"
_PRED_COLOR = "0.55"

_DPI = 150

_STATEMENT = (
    "Star counts per magnitude toward Orion A: observed by Gaia (stars in front of the cloud) "
    "and 2MASS (stars through it), against a model of the Milky Way (TRILEGAL) before "
    "calibration; the strips show the factor the model is scaled by, bin by bin, after "
    "subtracting the known young stars.")


def _read_histograms(config, region):
    path = config_module.product_path(config, "population", "anchors", "histograms", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        return dict(
            g_edges=np.asarray(f["G_EDGES"][:], dtype=np.float64),
            ks_edges=np.asarray(f["KS_EDGES"][:], dtype=np.float64),
            n_g_obs=np.asarray(f["N_G_OBS"][:], dtype=np.float64).sum(axis=0),
            n_g_pred=np.asarray(f["N_G_PRED"][:], dtype=np.float64).sum(axis=0),
            n_ks_obs=np.asarray(f["N_KS_OBS"][:], dtype=np.float64).sum(axis=0),
            n_ks_pred=np.asarray(f["N_KS_PRED"][:], dtype=np.float64).sum(axis=0),
        )


def _read_young_stars(config, region):
    path = config_module.product_path(config, "population", "anchors", "young-stars", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        return dict(
            n_g_young=np.asarray(f["N_G_YOUNG"][:], dtype=np.float64).sum(axis=0),
            n_ks_young=np.asarray(f["N_KS_YOUNG"][:], dtype=np.float64).sum(axis=0),
        )


def _read_weights(config, region):
    path = config_module.product_path(config, "population", "anchors", "weights", "tile", region=region)
    with h5py.File(path, "r") as f:
        return dict(
            w_region_g=np.asarray(f["W_REGION_G"][:], dtype=np.float64),
            w_region_ks=np.asarray(f["W_REGION_KS"][:], dtype=np.float64),
        )


def _step_values(values):
    """`values` (n_bin,) repeated onto `edges` (n_bin+1,) for
    `ax.step(edges, ..., where="post")`: the last edge repeats the last
    bin's own value rather than dropping to zero."""
    return np.append(values, values[-1])


def _draw_panel(ax_hist, ax_w, edges, n_obs, n_pred, n_young, w_region, xlabel):
    floor = 0.5 * np.min(np.concatenate([n_obs[n_obs > 0], n_pred[n_pred > 0]])) \
        if np.any(n_obs > 0) or np.any(n_pred > 0) else 0.1
    ax_hist.step(edges, _step_values(n_obs), where="post", color=_OBS_COLOR, lw=1.8, label="observed")
    ax_hist.step(edges, _step_values(n_pred), where="post", color=_PRED_COLOR, lw=1.8, label="model (TRILEGAL)")
    ax_hist.step(edges, _step_values(n_young), where="post", color=_HUE_COLOR, lw=1.0, label="young stars")
    ax_hist.set_yscale("log")
    ax_hist.set_ylim(bottom=max(floor, 1e-1))
    ax_hist.yaxis.set_major_formatter(ScalarFormatter())
    ax_hist.yaxis.set_minor_formatter(NullFormatter())
    ax_hist.tick_params(labelsize=10)
    ax_hist.set_ylabel(r"$\mathbf{N}$", fontsize=13, fontweight="bold")
    ax_hist.legend(fontsize=9, frameon=False)
    ax_hist.tick_params(labelbottom=False)

    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_w.axhline(1.0, color="0.6", lw=1.0, linestyle="--")
    ax_w.plot(centers, w_region, marker="o", ms=6, linestyle="none", color=_HUE_COLOR)
    ax_w.tick_params(labelsize=10)
    ax_w.set_ylabel(r"$\mathbf{W}$", fontsize=13, fontweight="bold")
    ax_w.set_xlabel(xlabel, fontsize=13, fontweight="bold")


def build_region_figure(config, region):
    hist = _read_histograms(config, region)
    young = _read_young_stars(config, region)
    weights = _read_weights(config, region)

    plot_style.apply_style()
    fig = plot_style.new_sized_figure(_PAGE_WIDTH_IN, _PAGE_HEIGHT_IN)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[3.2, 1.0], hspace=0.06, wspace=0.22,
                           left=0.06, right=0.98, top=0.86, bottom=0.20)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_gw = fig.add_subplot(gs[1, 0], sharex=ax_g)
    ax_k = fig.add_subplot(gs[0, 1])
    ax_kw = fig.add_subplot(gs[1, 1], sharex=ax_k)

    _draw_panel(ax_g, ax_gw, hist["g_edges"], hist["n_g_obs"], hist["n_g_pred"], young["n_g_young"],
                weights["w_region_g"], r"$\mathbf{G}$ [mag]")
    _draw_panel(ax_k, ax_kw, hist["ks_edges"], hist["n_ks_obs"], hist["n_ks_pred"], young["n_ks_young"],
                weights["w_region_ks"], r"$\mathbf{K_s}$ [mag]")

    fig.suptitle(f"{region} Star Count Calibration", fontsize=18, fontweight="bold")
    statement = _STATEMENT if region == "Orion A" else _STATEMENT.replace("Orion A", region, 1)
    fig.text(0.5, 0.02, statement, ha="center", va="bottom", fontsize=9, wrap=True)

    stats = dict(
        region=region,
        total_g_obs=float(hist["n_g_obs"].sum()), total_g_pred=float(hist["n_g_pred"].sum()),
        total_g_young=float(young["n_g_young"].sum()),
        total_ks_obs=float(hist["n_ks_obs"].sum()), total_ks_pred=float(hist["n_ks_pred"].sum()),
        total_ks_young=float(young["n_ks_young"].sum()),
        mean_w_g=float(np.nanmean(weights["w_region_g"])),
        mean_w_ks=float(np.nanmean(weights["w_region_ks"])),
    )
    return fig, stats


def _write_figure(config, region, fig):
    fig_dir = f"{config.data_root}/population/anchors/figures"
    os.makedirs(fig_dir, exist_ok=True)
    stem = f"anchors_anchors_region__{region}"
    png_path = f"{fig_dir}/{stem}.png"
    pdf_path = f"{fig_dir}/{stem}.pdf"
    fig.savefig(png_path, dpi=_DPI)
    fig.savefig(pdf_path)
    return png_path, pdf_path


def build(config, regions=None):
    """Writes, per region (default: all thirty), `population/anchors/
    figures/anchors_anchors_region__<Region>.{png,pdf}` -- the star-count
    calibration figure (module docstring). Prints, per region, the
    region-total observed/predicted/young counts on both anchors and the
    mean adopted weight per band.
    """
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in names:
        fig, stats = build_region_figure(config, region)
        try:
            png_path, pdf_path = _write_figure(config, region, fig)
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)
        print(
            "population.figure_anchors: %s G obs=%.1f pred=%.1f young=%.1f mean_W_G=%.4f "
            "Ks obs=%.1f pred=%.1f young=%.1f mean_W_Ks=%.4f -> %s"
            % (region, stats["total_g_obs"], stats["total_g_pred"], stats["total_g_young"],
               stats["mean_w_g"], stats["total_ks_obs"], stats["total_ks_pred"],
               stats["total_ks_young"], stats["mean_w_ks"], png_path))


def _main():
    parser = argparse.ArgumentParser(description="Build the STAR-anchor calibration figure.")
    parser.add_argument("config", help="path to root.cfg (or an overlay pointing at it)")
    parser.add_argument("--regions", nargs="+", default=None, help="region names (default: all thirty)")
    args = parser.parse_args()
    config = config_module.load(args.config)
    build(config, regions=args.regions)


if __name__ == "__main__":
    _main()
