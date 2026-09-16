r"""The young-star law across the regions, and the census against the
library (P4, `bms_review/briefs/P4.md`): one survey-wide figure, two
panels.

Left panel -- the star-gas law itself (`population.yso.law_count`,
`KAPPA_POOLED * (d_r*pi/180)^2 * A_K^2`, fitted on the Dunham census): the
region product `population/yso/law_yso_region.hdf5` carries only
`D_R_PC` and `PC2_PER_DEG2` per region (no per-region column), so the
one axis the 30 regions can vary along is distance -- each region's
young-star count per square degree AT A REFERENCE COLUMN (A_K = 1 mag,
so `N_law = KAPPA_POOLED * PC2_PER_DEG2`) against `D_R_PC`. Because
`KAPPA_POOLED` is the pooled coefficient drawn here, a
region's point falls exactly on the curve `N_law(d) = KAPPA_POOLED *
(d*pi/180)^2`; the `LAW_BAND_DEX` band around it is Pokhrel+2020's own
reported cloud-to-cloud scatter of the normalisation, not a fit
residual. Drawn against distance rather than column since the region
product carries no per-region column to plot against -- see P4.md's
"choose what yso.py's form makes honest" and the report.

Right panel -- the template weight's numerator and denominator (spec
sec 5.5 "Template weights", `bmstp.template_weights.
yso_population_weight`'s statement (i)): the census's own `log10 F_4.5`
density (Dunham et al. 2015's `LOG10_F45_REF`) against the model
library's (`C_THETA`, the register's flux at the same 1 kpc reference
distance -- `bmstp.template_weights._dunham_census_brightness_
histogram`'s own docstring: "the view's own LOG10_F45_REF (already the
dereddened 4.5 micron flux at 1 kpc, the register's own convention, no
unit conversion against C_THETA)"), both binned on the shared axis
`LOG10_F45_CENTERS`. The ratio strip beneath is the PLAIN ratio of the
two normalised histograms, not `factor_0/W` collapsed over templates:
`factor_0/W` in `bmstp/weights/yso_weights_survey.hdf5` is the
FINISHED per-template weight `w_theta` (statement (i) x (ii) x (iii),
inclination and evolutionary-class census already folded in) broadcast
across every brightness cell -- its column sums are the same constant
at every one of the 110 cells (confirmed by inspection: 0.9998385
everywhere), so it carries no brightness-dependent shape to show in a
ratio-vs-`log10 F_4.5` strip. The plain census/library histogram ratio
is statement (i) alone, the piece of the template weight that actually
varies with brightness, and is what the panel's own words describe:
"whose ratio weights the library's templates."
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, NullFormatter

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute.build import run

#: The reference column the law's left panel is evaluated at (P4.md:
#: "at a reference column (say A_K = 1 mag)").
A_K_REFERENCE = 1.0


def _decode(arr):
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr])


def _read_law(config):
    path = config_module.product_path(config, "population", "yso", "law", "region")
    with h5py.File(path, "r") as f:
        region = _decode(f["REGION"][:])
        d_r_pc = np.asarray(f["D_R_PC"][:], dtype=np.float64)
        pc2_per_deg2 = np.asarray(f["PC2_PER_DEG2"][:], dtype=np.float64)
        kappa_used = np.asarray(f["KAPPA_USED"][:], dtype=np.float64)
        kappa_herschel = float(f["KAPPA_POOLED"][()])
        law_band_dex = float(f["LAW_BAND_DEX"][()])
    return dict(region=region, d_r_pc=d_r_pc, pc2_per_deg2=pc2_per_deg2,
                kappa_herschel=kappa_herschel, kappa_used=kappa_used, law_band_dex=law_band_dex, path=path)


def _read_census(config):
    path = config_module.product_path(config, "sky/derived", "dunham2015", "yso", "survey")
    with h5py.File(path, "r") as f:
        log10_f45_ref = np.asarray(f["LOG10_F45_REF"][:], dtype=np.float64)
    return dict(log10_f45_ref=log10_f45_ref, path=path)


def _read_library(config):
    path = config_module.product_path(config, "bmstp", "weights", "yso", "survey")
    with h5py.File(path, "r") as f:
        c_theta = np.asarray(f["C_THETA"][:], dtype=np.float64)
        log10_f45_centers = np.asarray(f["LOG10_F45_CENTERS"][:], dtype=np.float64)
    return dict(c_theta=c_theta, log10_f45_centers=log10_f45_centers, path=path)


def _plain_number_formatter():
    """Log-scale tick labels as plain numbers (no `1e+04`, house
    convention): integers at or above 1, one decimal below it."""
    def fmt(value, _pos):
        if value <= 0:
            return ""
        if value >= 1.0:
            return f"{value:,.0f}"
        return f"{value:g}"
    return FuncFormatter(fmt)


def _figure_path(config):
    law_path = config_module.product_path(config, "population", "yso", "law", "region")
    figures_dir = os.path.join(os.path.dirname(law_path), "figures")
    return os.path.join(figures_dir, "law_yso_survey")


def build(config, regions=None):
    """The one survey-wide figure (P4.md): reads the three fixed
    products with h5py in read-only mode and writes
    `population/yso/figures/law_yso_survey.{png,pdf}`. `regions` is
    accepted for the package's module convention but unused -- this
    figure is survey-wide, not per-region."""
    plot_style.apply_style()

    law = _read_law(config)
    census = _read_census(config)
    library = _read_library(config)

    # ---- left panel: the law itself, young stars per pc^2 against the dust column ----
    kappa = law["kappa_herschel"]
    band_dex = law["law_band_dex"]
    a_line = np.logspace(np.log10(0.05), np.log10(5.0), 200)
    n_line = kappa * a_line ** 2
    band_lo = n_line * 10.0 ** (-band_dex)
    band_hi = n_line * 10.0 ** (band_dex)
    # Orion A's own columns: the 16th to 84th percentile of its sources' sightline columns
    region_mark = (regions[0] if regions else "Orion A")
    orion_idx = np.where(law["region"] == region_mark)[0]
    dens_path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region_mark)
    with h5py.File(dens_path, "r") as f:
        a_col = np.asarray(f["A_COL_K"][:], dtype=np.float64)
    a_lo, a_hi = np.percentile(a_col[np.isfinite(a_col) & (a_col > 0)], [16, 84])
    pc2_per_deg2_mark = float(law["pc2_per_deg2"][orion_idx[0]]) if orion_idx.size else float("nan")

    # ---- right panel: census vs. library, and their ratio ----
    centers = library["log10_f45_centers"]
    d_center = float(np.diff(centers).mean())
    edges = np.concatenate([centers - d_center / 2.0, [centers[-1] + d_center / 2.0]])

    f45_ref = census["log10_f45_ref"]
    finite = np.isfinite(f45_ref)
    census_counts, _ = np.histogram(f45_ref[finite], bins=edges)
    census_hist = census_counts.astype(np.float64) / max(1, int(finite.sum()))

    c_theta = library["c_theta"]
    library_counts, _ = np.histogram(c_theta, bins=edges)
    library_hist = library_counts.astype(np.float64) / c_theta.size

    ratio = np.full_like(census_hist, np.nan)
    have_both = (library_hist > 0) & (census_hist > 0)
    ratio[have_both] = census_hist[have_both] / library_hist[have_both]

    census_median = float(np.median(f45_ref[finite]))
    library_median = float(np.median(c_theta))

    # ---- the figure ----
    fig = plt.figure(figsize=(16.0, 6.0))
    gs = GridSpec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[3.0, 1.0],
                  hspace=0.08, wspace=0.24,
                  left=0.06, right=0.98, top=0.84, bottom=0.24)

    ax_law = fig.add_subplot(gs[:, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_ratio = fig.add_subplot(gs[1, 1], sharex=ax_hist)

    # left: the law
    ax_law.fill_between(a_line, band_lo, band_hi, color="0.55", alpha=0.25, lw=0,
                         zorder=1, label=f"cloud-to-cloud band (±{band_dex:.2f} dex)")
    ax_law.plot(a_line, n_line, color="0.35", lw=2.2, zorder=2, label=r"the law, $N = \kappa A_K^2$")
    ax_law.axvspan(a_lo, a_hi, color="#1b9e77", alpha=0.18, lw=0, zorder=0,
                   label=f"{region_mark}'s sightline columns (16th to 84th percentile)")
    ax_law.set_xscale("log")
    ax_law.set_yscale("log")
    ax_law.set_xlim(0.05, 5.0)
    ax_law.xaxis.set_major_formatter(_plain_number_formatter())
    ax_law.yaxis.set_major_formatter(_plain_number_formatter())
    ax_law.xaxis.set_minor_formatter(NullFormatter())
    ax_law.yaxis.set_minor_formatter(NullFormatter())
    ax_law.set_xlabel(plot_style.label(r"dust column $\mathbf{A_K}$", "mag"))
    ax_law.set_ylabel(plot_style.label("young stars", r"pc$^{-2}$"))
    ax_law.set_title("Star-Gas Law", fontsize=13)
    ax_law.text(0.03, 0.03, f"at {region_mark}'s distance, 1 deg² = {pc2_per_deg2_mark:.0f} pc²",
                transform=ax_law.transAxes, fontsize=9, ha="left", va="bottom", color="0.25")
    leg = ax_law.legend(loc="upper left", fontsize=9, frameon=False)

    # right, top: the two histograms
    ax_hist.bar(centers, census_hist, width=d_center, align="center",
                color="black", alpha=0.55, label="census (Dunham et al. 2015)")
    ax_hist.bar(centers, library_hist, width=d_center, align="center",
                color="0.65", alpha=0.55, label="model library")
    ax_hist.set_ylabel("Fraction of sample")
    ax_hist.set_title("Census and Library Brightness", fontsize=13)
    ax_hist.tick_params(labelbottom=False)
    ax_hist.legend(loc="upper right", fontsize=9, frameon=False)

    # right, bottom: the ratio strip
    ax_ratio.axhline(1.0, color="0.6", lw=1.0, ls="--", zorder=1)
    ax_ratio.plot(centers, ratio, color="#d95f02", lw=1.6, marker="o",
                  markersize=3.5, zorder=2)
    ax_ratio.set_yscale("log")
    ax_ratio.yaxis.set_major_formatter(_plain_number_formatter())
    ax_ratio.yaxis.set_minor_formatter(NullFormatter())
    ax_ratio.set_ylabel("census / library", fontsize=11)
    ax_ratio.set_xlabel(plot_style.label(r"$\mathbf{log_{10}\,F_{4.5}}$", "mJy"))

    fig.suptitle("Young-Star Law and Census", fontsize=18, fontweight="bold")

    statement = ("Left: the star–gas law of Pokhrel et al. 2020, young stars per unit "
                 "area rising as the square of the dust column, with the range of columns Orion A's sources sit under; "
                 "right: the 4.5 µm brightness of the young stars in a Spitzer census "
                 "of nearby clouds (Dunham et al. 2015) against the model library's, whose "
                 "ratio weights the library's templates.")
    fig.text(0.5, 0.03, statement, ha="center", va="bottom", fontsize=9, wrap=True)

    out_stem = _figure_path(config)
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    png_path = out_stem + ".png"
    pdf_path = out_stem + ".pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)

    print(f"figure_yso_law: KAPPA_POOLED={kappa:.3f} young stars pc^-2 mag^-2, "
          f"LAW_BAND_DEX={band_dex:.3f} dex, "
          f"census median log10 F_4.5={census_median:.3f} (n={int(finite.sum())}), "
          f"library median log10 F_4.5={library_median:.3f} (n={c_theta.size})")
    print(f"figure_yso_law: wrote {png_path}")
    print(f"figure_yso_law: wrote {pdf_path}")
    return png_path, pdf_path


def _main():
    run(build)


if __name__ == "__main__":
    _main()
