"""The galaxy counts figure (P3): `population/gal/counts_gal_survey.hdf5`'s
own `PHI_S`/`PHI_S_POINT` broken power law against `LOG10_S_GRID`, with the
`P_POINT` point-like fraction in a strip beneath -- one survey-wide figure,
region-independent, matching `gal.py`'s own build.

`PHI_S` is `gal.py`'s intrinsic 4.5 um galaxy number-counts law, phi(S) =
-dN(>S)/dS, in that module's own units: galaxies deg^-2 mJy^-1 (`gal.
BrokenPowerLaw.differential`'s own docstring), fitted once, survey-wide, to
Fazio et al. (2004)'s own three Spitzer fields (Bootes, EGS, QSO1700 --
`gal.FIELD_COLUMNS`/`gal.fit_all_variants`); this is the counts law itself,
never SWIRE. `PHI_S_POINT = PHI_S * P_POINT` is that same law times the
IRAC point-source retention p(S) (`gal.point_source_retention`), measured
against the wide-field SWIRE survey (Surace et al. 2005) by comparing its
own extended-source flags to Fazio's star counts (`gal.py`'s module
docstring, `studies/swire_vs_fazio.md`) -- SWIRE is the point-source
fraction's own source, not the counts law's.
"""

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import progress

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 10
STATEMENT_FONTSIZE = 9

#: This figure's own fixed page size (the brief's own acceptance line),
#: not a `FIGURE_SIZES` entry -- one panel with a strip beneath, wide
#: enough for the strip's own axis and the statement line below it.
PAGE_WIDTH_IN, PAGE_HEIGHT_IN = 10.0, 7.0


def _plain_number(value):
    """A log-axis tick value with no exponent notation ('10000', not
    '1e+04'; '0.03', not '3e-02') -- the convention's own "plain numbers
    on the ticks", the same rule `atlas.render._format_log_tick` applies."""
    if value == 0:
        return "0"
    rounded = round(value)
    if abs(value - rounded) <= 1e-6 * max(abs(value), 1.0) and rounded != 0:
        return "%d" % rounded
    return ("%f" % value).rstrip("0").rstrip(".")


def _plain_formatter():
    return FuncFormatter(lambda v, _pos: _plain_number(v))


def build(config, regions=None):
    """Reads `population/gal/counts_gal_survey.hdf5` and writes the one
    survey-wide galaxy-counts figure. Survey-wide; `regions` is accepted
    and ignored, matching `gal.build`'s own convention."""
    st = progress.Stage("population.gal.figure_gal_counts")

    in_path = config_module.product_path(config, "population", "gal", "counts", "survey")
    with h5py.File(in_path, "r") as f:
        log10_s_grid = np.asarray(f["LOG10_S_GRID"][:], dtype=np.float64)
        phi_s = np.asarray(f["PHI_S"][:], dtype=np.float64)
        phi_s_point = np.asarray(f["PHI_S_POINT"][:], dtype=np.float64)
        p_point = np.asarray(f["P_POINT"][:], dtype=np.float64)
        alpha_bright = float(f["ALPHA_BRIGHT"][()])
        alpha_faint = float(f["ALPHA_FAINT"][()])
        log10_s_break = float(f["LOG10_S_BREAK"][()])
        log10_a = float(f["LOG10_A"][()])
        fit_rms_dex = float(f["FIT_RMS_DEX"][()])
        cosmic_variance_dex = float(f["COSMIC_VARIANCE_DEX"][()])
        fit_log10_s_min = float(f["FIT_LOG10_S_MIN"][()])
        fit_log10_s_max = float(f["FIT_LOG10_S_MAX"][()])

    print(f"figure_gal_counts: read {in_path}: log10_A={log10_a:.4f} "
          f"log10_S_break={log10_s_break:.4f} alpha_faint={alpha_faint:.4f} "
          f"alpha_bright={alpha_bright:.4f} fit_rms_dex={fit_rms_dex:.4f} "
          f"cosmic_variance_dex={cosmic_variance_dex:.4f} "
          f"fit_range=[{fit_log10_s_min:.4f}, {fit_log10_s_max:.4f}]")

    s_mjy = 10.0 ** log10_s_grid
    s_lo, s_hi = 10.0 ** fit_log10_s_min, 10.0 ** fit_log10_s_max
    s_break = 10.0 ** log10_s_break
    xlim = (min(s_mjy.min(), s_break) / 1.4, s_mjy.max() * 1.4)

    plot_style.apply_style()
    fig = plot_style.new_sized_figure(PAGE_WIDTH_IN, PAGE_HEIGHT_IN)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.08,
                           left=0.11, right=0.97, top=0.90, bottom=0.20)
    ax = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax)

    for a in (ax, ax_strip):
        a.axvspan(s_lo, s_hi, color="0.90", zorder=0)
        a.axvline(s_break, color="0.35", linestyle=":", linewidth=1.3, zorder=1)
        a.set_xscale("log")
        a.set_xlim(*xlim)

    ax.plot(s_mjy, phi_s, color="tab:blue", linewidth=2.0, label="all galaxies")
    ax.plot(s_mjy, phi_s_point, color="tab:orange", linewidth=2.0, label="point-like (Spitzer)")
    ax.set_yscale("log")
    ax.set_ylabel(plot_style.label(r"$\mathbf{\phi(S)}$", "galaxies deg$^{-2}$ mJy$^{-1}$"),
                  fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_formatter(_plain_formatter())
    ax.legend(fontsize=TICK_FONTSIZE, loc="upper right", frameon=False)
    plt.setp(ax.get_xticklabels(), visible=False)

    trans = ax.get_xaxis_transform()
    ax.text(s_break / (10.0 ** 0.15), 0.92, f"slope {alpha_faint:.2f}", transform=trans,
            ha="right", va="top", fontsize=TICK_FONTSIZE, color="0.25")
    ax.text(s_break * (10.0 ** 0.15), 0.92, f"slope {alpha_bright:.2f}", transform=trans,
            ha="left", va="top", fontsize=TICK_FONTSIZE, color="0.25")

    ax_strip.plot(s_mjy, p_point, color="tab:green", linewidth=2.0)
    ax_strip.set_ylim(0.0, 1.0)
    ax_strip.set_ylabel("point-like fraction", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax_strip.set_xlabel(plot_style.label(r"$\mathbf{S_{4.5}}$", "mJy"),
                         fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax_strip.tick_params(labelsize=TICK_FONTSIZE)
    ax_strip.xaxis.set_major_formatter(_plain_formatter())

    fig.suptitle("Galaxy Counts", fontsize=TITLE_FONTSIZE, fontweight="bold", y=0.98)

    statement = (
        "Galaxies per square degree against 4.5 µm flux, a broken power law fitted to "
        "Fazio et al. (2004)'s own wide-field Spitzer galaxy counts, and the fraction of them "
        "Spitzer sees as points rather than extended, measured against the wide-field SWIRE "
        f"survey; the fit's scatter is {fit_rms_dex:.2f} dex and the field-to-field variance "
        f"is {cosmic_variance_dex:.2f} dex."
    )
    fig.text(0.5, 0.06, statement, fontsize=STATEMENT_FONTSIZE, ha="center", va="top", wrap=True)

    out_dir = os.path.join(config.data_root, "population", "gal", "figures")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in ("png", "pdf"):
        path = os.path.join(out_dir, f"counts_gal_survey.{fmt}")
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)

    st.done(paths[0], log10_a=log10_a, log10_s_break=log10_s_break, alpha_faint=alpha_faint,
            alpha_bright=alpha_bright, fit_rms_dex=fit_rms_dex,
            cosmic_variance_dex=cosmic_variance_dex)
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
