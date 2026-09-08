"""The prior at a source: the six shapes and levels, one page per region
(SPEC_BMSTP_DRAFT.md sec. 1.1's factorisation, sec. 4.1-4.2, sec. 5.1-5.6;
W18). Nothing shows what the prior actually says at one source on the
axes the fitter multiplies -- this is that page's replacement for the new
design, after the earlier `bms_backup/prior/validation/atlas/atlas_*.pdf`.
Report-only: nothing written here is read by the fitter or by any other
`bmstp`/`fittp` stage.

Per region, two sources from the density table: the one at the region's
median `A_COL_K` and the one at its 99th percentile. For each, six panels
of the shape `h_C(x, log10 B)` the fitter reads for that source --
`fittp.prior_reader.load`/`prepare`, which blurs the class's stored grain
shape by the source's own column kernel (sec. 4.2) -- in the old page's
order (GAL, YSO, H2S, STAR, PAHC, AGB), plus a bar of the six `DENSITY_<C>`
at the source (`A_C(s)`, sec. 1.1). The page keeps shape and level apart
because that is how the spec states the intensity and how a defect (a
collapsed class, an off-grid peak, a wrong-unit brightness axis) is
located.
"""

import argparse

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.bmstp import grid as bmstp_grid
from sesnaimpute.bmstp import sample_gal
from sesnaimpute.fittp import prior_reader

PAGE_W_IN, PAGE_H_IN = 16.0, 9.0

#: The old page's panel order (W18's brief).
CLASS_ORDER = ("GAL", "YSO", "H2S", "STAR", "PAHC", "AGB")

#: STAR/AGB/PAHC/YSO share the template-unit brightness axis (sec. 2):
#: `log10 B = 0` is the template at 1 kpc, `B = 1`.
TEMPLATE_UNIT_CLASSES = ("STAR", "AGB", "PAHC", "YSO")

_Y_LABEL = {
    "STAR": "log10 B  [template units]", "AGB": "log10 B  [template units]",
    "PAHC": "log10 B  [template units]", "YSO": "log10 B  [template units]",
    "GAL": "log10 S  [mJy]", "H2S": "log10 Σ  [erg s⁻¹ cm⁻² sr⁻¹]",
}

_ARM_NAME = {0: "Herschel", 1: "Planck"}


def _select_sources(a_col):
    """`(idx_median, idx_p99)`: the catalogued source closest to the
    region's median `A_COL_K` and closest to its 99th percentile."""
    idx_median = int(np.argmin(np.abs(a_col - np.median(a_col))))
    idx_p99 = int(np.argmin(np.abs(a_col - np.percentile(a_col, 99.0))))
    return idx_median, idx_p99


def _read_density_table(config, region):
    """P1's per-source rows this page needs: `NAME`, `A_COL_K`, `ARM`,
    the star-family/cloud grain indices, the six `DENSITY_<C>` (sec.
    4.1), and the file's `KAPPA_HERSCHEL`/`KAPPA_PLANCK` attributes
    (sec. 5.5's law, the caption line)."""
    path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    with h5py.File(path, "r") as f:
        d = dict(
            name=f["NAME"][:], a_col=f["A_COL_K"][:].astype(np.float64),
            arm=f["ARM"][:].astype(np.int64), tile=f["TILE"][:].astype(np.int64),
            sightline=f["SIGHTLINE_ROW"][:].astype(np.int64),
            kappa_herschel=float(f.attrs["KAPPA_HERSCHEL"]),
            kappa_planck=float(f.attrs["KAPPA_PLANCK"]),
        )
        for c in ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S"):
            d[c] = f["DENSITY_%s" % c][:].astype(np.float64)
    return d


def _mass_outside_star(config, region):
    """`(MASS_OUTSIDE_STAR, MASS_OUTSIDE_AGB)` per tile (P2), the grid's
    own build-time identity (sec. 2's 0.1% bar) -- PAHC reads STAR's
    (sec. 5.3 "Grain": the shape IS STAR's)."""
    path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
    with h5py.File(path, "r") as f:
        return f["MASS_OUTSIDE_STAR"][:].astype(np.float64), f["MASS_OUTSIDE_AGB"][:].astype(np.float64)


def _mass_outside_yso(config, region):
    """`MASS_OUTSIDE_YSO` per sightline (P3) -- H2S reads it too (its own
    grid is YSO's `x` marginal times an exact, always-normalised Gaussian
    on `log10 Sigma`, sec. 5.6 "Marks": no mass is lost on that axis)."""
    path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    with h5py.File(path, "r") as f:
        return f["MASS_OUTSIDE_YSO"][:].astype(np.float64)


def _gal_mass_outside(config):
    """GAL's survey-wide `mass_outside` (P4), recomputed by the same
    `grid.bin` call `bmstp.shapes.build_gal` makes (not stored on the
    product itself, sec. 4.1 P4) -- cheap: one 61-node sample."""
    path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
    with h5py.File(path, "r") as f:
        origin = float(f.attrs["LOG10_B_ORIGIN"])
    x, log10_b, w = sample_gal.sample(config)
    _, mass_outside = bmstp_grid.bin(x, log10_b, w, origin, 0.0)
    return mass_outside


def _class_mass_outside(cls, dtab, mass_star, mass_agb, mass_yso, mass_gal, src_idx):
    if cls in ("STAR", "PAHC"):
        return float(mass_star[dtab["tile"][src_idx]])
    if cls == "AGB":
        return float(mass_agb[dtab["tile"][src_idx]])
    if cls in ("YSO", "H2S"):
        return float(mass_yso[dtab["sightline"][src_idx]])
    return float(mass_gal)  # GAL


def _panel_arrays(config, region, cls, rows):
    """`(density, x_edges, b_edges)`: the shape `h_C` the fitter reads
    for each of `rows`' sources -- `fittp.prior_reader.load`/`prepare`,
    which blurs the class's stored grain shape by the source's own
    column kernel (sec. 4.2) and floors/renormalises it -- converted from
    a per-cell mass to a density per mag·dex (`/ (dlx * dlb)`, sec. 4.1's
    "the page plots h_C, a pdf ... per mag·dex")."""
    reader = prior_reader.load(config, region, cls)
    h = prior_reader.prepare(reader, np.asarray(rows))
    density = h.astype(np.float64) / (reader.dlx * reader.dlb)
    return density, reader.x_edges, reader.b_edges


def _build_region_data(config, region):
    dtab = _read_density_table(config, region)
    idx_median, idx_p99 = _select_sources(dtab["a_col"])
    rows = np.array([idx_median, idx_p99])
    mass_star, mass_agb = _mass_outside_star(config, region)
    mass_yso = _mass_outside_yso(config, region)
    mass_gal = _gal_mass_outside(config)

    panels = {}
    for cls in CLASS_ORDER:
        density2, x_edges, b_edges = _panel_arrays(config, region, cls, rows)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
        for i, src_idx in enumerate(rows):
            d = density2[i]
            pi, pj = np.unravel_index(int(np.argmax(d)), d.shape)
            mass = 1.0 - _class_mass_outside(cls, dtab, mass_star, mass_agb, mass_yso, mass_gal, src_idx)
            panels[(i, cls)] = dict(
                density=d, x_edges=x_edges, b_edges=b_edges,
                x_centers=x_centers, b_centers=b_centers, mass=mass,
                peak_x=float(x_centers[pi]), peak_b=float(b_centers[pj]),
                b_at_1=0.0 if cls in TEMPLATE_UNIT_CLASSES else None)
    return dtab, rows, panels


def _print_numbers(region, dtab, rows, panels):
    for i, src_idx in enumerate(rows):
        label = "median" if i == 0 else "p99"
        name = dtab["name"][src_idx].decode("utf-8")
        print("atlas.shapes [%s] %s source %s: A_COL_K=%.4g mag arm=%s"
              % (region, label, name, dtab["a_col"][src_idx], _ARM_NAME[int(dtab["arm"][src_idx])]))
        for cls in CLASS_ORDER:
            p = panels[(i, cls)]
            b1 = ("%.4g" % p["b_at_1"]) if p["b_at_1"] is not None else "n/a"
            print("atlas.shapes [%s] %s/%s: mass=%.6f peak_log10x=%.4g peak_log10B=%.4g log10B_at_B1=%s"
                  % (region, label, cls, p["mass"], p["peak_x"], p["peak_b"], b1))


def _draw_figure(config, region, dtab, rows, panels):
    plot_style.apply_style()
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))

    margin_l, margin_r, margin_t, margin_b = 0.55, 0.95, 0.85, 0.55
    row_gap, col_gap, bar_ratio = 0.90, 0.14, 1.5
    usable_w = PAGE_W_IN - margin_l - margin_r
    unit_w = (usable_w - 6 * col_gap) / (6.0 + bar_ratio)
    shape_w, bar_w = unit_w, unit_w * bar_ratio
    row_h = (PAGE_H_IN - margin_t - margin_b - row_gap) / 2.0

    all_log = np.concatenate([np.log10(p["density"]).ravel() for p in panels.values()])
    norm = Normalize(vmin=float(all_log.min()), vmax=float(all_log.max()))
    cmap = plt.get_cmap("viridis")
    im = None

    for i in range(2):
        y0 = margin_b + (1 - i) * (row_h + row_gap)
        src_idx = rows[i]
        a_col_src = dtab["a_col"][src_idx]
        for c, cls in enumerate(CLASS_ORDER):
            x0 = margin_l + c * (shape_w + col_gap)
            ax = fig.add_axes([x0 / PAGE_W_IN, y0 / PAGE_H_IN, shape_w / PAGE_W_IN, row_h / PAGE_H_IN])
            p = panels[(i, cls)]
            im = ax.imshow(np.log10(p["density"]).T, origin="lower", aspect="auto",
                            extent=[p["x_edges"][0], p["x_edges"][-1], p["b_edges"][0], p["b_edges"][-1]],
                            cmap=cmap, norm=norm)
            if p["b_at_1"] is not None:
                ax.axhline(0.0, color="white", lw=0.6, ls="--", alpha=0.8)
            ax.text(0.02, 0.03, "mass=%.4f\npeak=(%.2f, %.2f)" % (p["mass"], p["peak_x"], p["peak_b"]),
                    transform=ax.transAxes, fontsize=5.5, color="white", va="bottom")
            ax.set_ylabel(_Y_LABEL[cls], fontsize=6.5)
            ax.set_xlabel("log10 x", fontsize=6.5)
            ax.tick_params(labelsize=6)
            xt = np.array([-3.0, -1.0, 1.0])
            ax.set_xticks(xt)
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xt)
            ax2.set_xticklabels(["%.3g" % (a_col_src * 10.0 ** v) for v in xt], fontsize=5)
            ax2.tick_params(length=2, pad=1, labelsize=5)
            if c == 0:
                ax2.set_xlabel("a  [A_K mag]", fontsize=5.5, labelpad=1)
            ax.set_title(cls, fontsize=9, weight="bold", pad=16)

        # the bar: the six DENSITY_<C> at this row's source, per deg^2.
        x_bar = margin_l + 6 * (shape_w + col_gap)
        ax_bar = fig.add_axes([x_bar / PAGE_W_IN, y0 / PAGE_H_IN, bar_w / PAGE_W_IN, row_h / PAGE_H_IN])
        values = np.array([dtab[cls][src_idx] for cls in CLASS_ORDER])
        bars = ax_bar.bar(range(6), values, color="steelblue")
        ax_bar.set_yscale("log")
        ax_bar.set_xticks(range(6))
        ax_bar.set_xticklabels(CLASS_ORDER, fontsize=6.5, rotation=30)
        ax_bar.set_ylabel("A_C(s)  [deg$^{-2}$]", fontsize=7)
        ax_bar.tick_params(labelsize=6)
        for x, v in zip(range(6), values):
            ax_bar.text(x, v * 1.15, "%.3g" % v, ha="center", va="bottom", fontsize=5.5)
        label = "median A_COL_K" if i == 0 else "99th pct. A_COL_K"
        ax_bar.set_title(label, fontsize=7.5)

    bar_rect = [(margin_l + 6 * (shape_w + col_gap) + bar_w + 0.15) / PAGE_W_IN,
                margin_b / PAGE_H_IN, 0.22 / PAGE_W_IN, (2 * row_h + row_gap) / PAGE_H_IN]
    cax = fig.add_axes(bar_rect)
    fig.colorbar(im, cax=cax, label=plot_style.label("log10 h", "mag$^{-1}$ dex$^{-1}$"))

    r = regions_module.REGIONS_BY_NAME[region]
    name_med = dtab["name"][rows[0]].decode("utf-8")
    name_p99 = dtab["name"][rows[1]].decode("utf-8")
    arm_med = _ARM_NAME[int(dtab["arm"][rows[0]])]
    arm_p99 = _ARM_NAME[int(dtab["arm"][rows[1]])]
    caption = (
        "%s -- median source %s (A_COL_K=%.3g mag, %s arm); "
        "p99 source %s (A_COL_K=%.3g mag, %s arm); "
        "d_r=%.0f±%.0f pc; KAPPA_HERSCHEL=%.1f, KAPPA_PLANCK=%.1f"
        % (region, name_med, dtab["a_col"][rows[0]], arm_med, name_p99, dtab["a_col"][rows[1]], arm_p99,
           r.d_r_pc, r.sigma_pc, dtab["kappa_herschel"], dtab["kappa_planck"]))
    fig.suptitle(caption, fontsize=9.5, y=1.0 - 0.15 / PAGE_H_IN)

    out_dir = f"{config.data_root}/bmstp/atlas/figures"
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in ("png", "pdf"):
        path = f"{out_dir}/prior-shapes_{region}.{fmt}"
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths


def build_region(config, region):
    with progress.Stage("atlas.shapes", region) as st:
        dtab, rows, panels = _build_region_data(config, region)
        _print_numbers(region, dtab, rows, panels)
        paths = _draw_figure(config, region, dtab, rows, panels)
        mass_min = min(p["mass"] for p in panels.values())
        mass_max = max(p["mass"] for p in panels.values())
        st.done(paths[0], n_panel=len(panels), mass_min=mass_min, mass_max=mass_max)
    return paths


def build(config, regions=None):
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        build_region(config, region)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()
    config = config_module.load(args.config)
    build(config, regions=args.regions)


if __name__ == "__main__":
    _main()
