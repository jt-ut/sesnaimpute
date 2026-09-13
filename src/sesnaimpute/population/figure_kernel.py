"""The survey-wide figure for the cloud-class structural width and its
within-beam tilt exponent, `sesnaimpute.population.kernel`'s own joint fit
on the HOPS (Orion A) and eHOPS (Aquila) Class 0/I/flat protostars.

Left panel: the fitted protostars' own measured `log10 xi_hat` (their
fitted foreground extinction over their matched SESNA source's own beam
column), a histogram, with the fitted spread's predicted distribution
drawn over it as a line -- the adopted one-component cloud-class mixture
(`kernel._cloud_single_component`, tilted by the adopted `gamma`, exactly
`Kernel.mixture`'s own arithmetic for a cloud class), evaluated at each
protostar's own sightline cloud-interval column distribution
(`kernel._cloud_cells_for_pixels`) and averaged over the sample -- the
same pooling `atlas.protostars._kernel_validation` performs, but on the
sample's own per-protostar columns rather than one region's shared
sightline.

Right panel: the joint profile-likelihood surface the fit is read off,
`CLOUD_GAMMA_SIGMA_LOGLIKE` over the spread and tilt-exponent grid, as a
drop from its own maximum, with the two intervals the fit itself reports
(`_CLOUD_JOINT_DLOGLIKE` = 1.15, and 3.0) contoured and the adopted point
marked.

Reads only `population/sesna/kernel_sesna_survey.hdf5` and the same
catalogues `kernel._match_protostars_to_beam` reads; writes only this
figure's own png/pdf, beside that product.
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute.population import kernel as kernel_module
from sesnaimpute.population import selection as selection_module

_LN10 = float(np.log(10.0))
_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))

#: the profile-likelihood drop the fit's own joint 68% and one-sided-3.0
#: intervals are drawn at (`kernel._CLOUD_JOINT_DLOGLIKE`, and the
#: brief's own second contour).
_DLOGLIKE_LEVELS = (1.15, 3.0)

#: the step (dex) the predicted density line and the histogram's own
#: domain are drawn at -- fine enough to trace the curve, coarse enough
#: that the average over protostars at each point is cheap.
_Z_STEP_DEX = 0.02

_PAGE_W_IN = 16.0
_PAGE_H_IN = 6.0

_TITLE_FONTSIZE = 18
_LABEL_FONTSIZE = 13
_STATEMENT_FONTSIZE = 9

_STATEMENT_TEMPLATE = (
    "Left: the measured depth fraction of {n} Herschel-confirmed protostars, whose "
    "fitted foreground can exceed their sightline's column because a protostar sits "
    "in a core the column map's beam does not resolve, against the spread fitted to "
    "them;\nright: the fit's likelihood over the spread and the tilt exponent, with "
    "the adopted values."
)


def _norm_cdf(z):
    """`Phi(z)`, the standard normal CDF, off the exact `erf`."""
    return 0.5 * (1.0 + erf(z / _SQRT2))


def _read_fit(config):
    """The adopted `(gamma, sigma_cloud)` and the joint likelihood grid
    the fit reports, read straight from the product -- no refit here."""
    path = config_module.product_path(config, "population", "sesna", "kernel", "survey")
    with h5py.File(path, "r") as f:
        return dict(
            sigma_cloud=float(f["CLOUD_SIGMA_HERSCHEL_DEX"][()]),
            sigma_p16=float(f["CLOUD_SIGMA_HERSCHEL_P16"][()]),
            sigma_p84=float(f["CLOUD_SIGMA_HERSCHEL_P84"][()]),
            gamma=float(f["CLOUD_GAMMA_HERSCHEL"][()]),
            gamma_p16=float(f["CLOUD_GAMMA_HERSCHEL_P16"][()]),
            gamma_p84=float(f["CLOUD_GAMMA_HERSCHEL_P84"][()]),
            sigma_grid=f["CLOUD_SIGMA_GRID_DEX"][:],
            gamma_grid=f["CLOUD_GAMMA_GRID"][:],
            loglike2d=f["CLOUD_GAMMA_SIGMA_LOGLIKE"][:],
            n_protostars_fit=int(f["N_PROTOSTARS_FIT"][()]),
            zp_herschel_k=float(f["ZP_HERSCHEL_K"][()]),
        )


def _sample_and_mixture(config, fit):
    """The fitted protostars' own `log10 xi_hat` and, for each of them,
    its own sightline cloud-interval column distribution (`log10x`,
    `mass`) and the adopted one-component cloud-class mixture
    (`w_tilt`, `mu_tilt`, `sigma_tot`) -- the same objects
    `kernel._fit_cloud_gamma_sigma_herschel` forms at its own trial grid
    points, formed here once at the adopted `(gamma, sigma_cloud)`
    alone."""
    from sesnaimpute.population import yso as yso_module

    match = kernel_module._match_protostars_to_beam(config)
    a_beam = match["a_beam"]
    n_proto = a_beam.size

    a_p = match["av_mag"] * selection_module.ak_per_av(
        config, selection_module.law_dense_weight(a_beam))
    log10_xi_hat = np.log10(a_p / a_beam)

    extra_var = ((match["sigma_beam"] / (a_beam * _LN10)) ** 2
                 + (fit["zp_herschel_k"] / (a_beam * _LN10)) ** 2)

    log10x_parts = []
    for region in kernel_module._PROTOSTAR_REGIONS:
        sel = match["region"] == region.encode("utf-8")
        if not np.any(sel):
            continue
        log10x_r, mass_r = kernel_module._cloud_cells_for_pixels(
            config, yso_module, region, match["pix256"][sel])
        log10x_parts.append((sel, log10x_r, mass_r))
    n_u_max = max(p[1].shape[1] for p in log10x_parts)
    log10x = np.zeros((n_proto, n_u_max))
    mass = np.zeros((n_proto, n_u_max))
    for sel, log10x_r, mass_r in log10x_parts:
        n_u = log10x_r.shape[1]
        rows = np.where(sel)[0]
        log10x[rows[:, None], np.arange(n_u)[None, :]] = log10x_r
        mass[rows[:, None], np.arange(n_u)[None, :]] = mass_r

    w0, mu0, sigma0 = kernel_module._cloud_single_component(fit["sigma_cloud"], n_proto)
    sigma_tot = np.sqrt(sigma0 ** 2 + extra_var[:, None])
    c = fit["gamma"] * _LN10
    mu_tilt = mu0 + c * sigma_tot * sigma_tot
    log_wt = c * mu0 + (c * sigma_tot) ** 2 / 2.0
    log_wt -= log_wt.max(axis=1, keepdims=True)
    wt = np.stack([w0, 1.0 - w0], axis=1) * np.exp(log_wt)
    w_tilt = wt[:, 0] / wt.sum(axis=1)

    return dict(log10_xi_hat=log10_xi_hat, log10x=log10x, mass=mass,
                w_tilt=w_tilt, mu_tilt=mu_tilt, sigma_tot=sigma_tot, n_proto=n_proto)


def _pooled_cdf(z, sample):
    """`P(log10 xi_hat <= z)`, the adopted mixture pooled over the
    sample's own per-protostar sightline columns -- one shared number,
    averaged over protostars, exactly `kernel._fit_cloud_gamma_sigma_
    herschel`'s own `_pooled_cdf` at the adopted grid point."""
    off0 = (z - sample["log10x"] - sample["mu_tilt"][:, 0:1]) / sample["sigma_tot"][:, 0:1]
    off1 = (z - sample["log10x"] - sample["mu_tilt"][:, 1:2]) / sample["sigma_tot"][:, 1:2]
    cdf_cell = (sample["w_tilt"][:, None] * _norm_cdf(off0)
                + (1.0 - sample["w_tilt"][:, None]) * _norm_cdf(off1))
    return float(np.mean(np.sum(sample["mass"] * cdf_cell, axis=1)))


def _solve_quantile(target, sample, lo=-6.0, hi=3.0, iters=60):
    """The `target` quantile of `_pooled_cdf`, by bisection (rule 8: one
    array pass over the sample per iteration, not a python loop over
    protostars)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _pooled_cdf(mid, sample) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _predicted_density(z_grid, sample):
    """The predicted density of `log10 xi_hat` at each point of `z_grid`:
    the adopted mixture, pooled over the sample's own per-protostar
    sightline columns the same way `_pooled_cdf` pools the CDF. One pass
    over the sample per grid point (rule 8)."""
    dens = np.empty(z_grid.size, dtype=np.float64)
    for iz, z in enumerate(z_grid):
        off0 = (z - sample["log10x"] - sample["mu_tilt"][:, 0:1]) / sample["sigma_tot"][:, 0:1]
        off1 = (z - sample["log10x"] - sample["mu_tilt"][:, 1:2]) / sample["sigma_tot"][:, 1:2]
        pdf0 = np.exp(-0.5 * off0 * off0) / (sample["sigma_tot"][:, 0:1] * _SQRT2PI)
        pdf1 = np.exp(-0.5 * off1 * off1) / (sample["sigma_tot"][:, 1:2] * _SQRT2PI)
        pdf_cell = sample["w_tilt"][:, None] * pdf0 + (1.0 - sample["w_tilt"][:, None]) * pdf1
        dens[iz] = np.mean(np.sum(sample["mass"] * pdf_cell, axis=1))
    return dens


def _figures_dir(config):
    kernel_path = config_module.product_path(config, "population", "sesna", "kernel", "survey")
    return os.path.join(os.path.dirname(kernel_path), "figures")


def _draw(config, fit, sample, emp_median, emp_p84, pred_median, pred_p84):
    plot_style.apply_style()

    log10_xi_hat = sample["log10_xi_hat"]
    xlo = min(float(np.min(log10_xi_hat)), -0.5) - 0.1
    xhi = max(float(np.max(log10_xi_hat)), 0.1) + 0.1
    bins = np.linspace(xlo, xhi, 31)
    bin_width = bins[1] - bins[0]
    z_grid = np.arange(xlo, xhi + 1e-9, _Z_STEP_DEX)
    density = _predicted_density(z_grid, sample)
    n = sample["n_proto"]
    expected_counts = density * n * bin_width

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_PAGE_W_IN, _PAGE_H_IN))

    ax1.hist(log10_xi_hat, bins=bins, color="steelblue", edgecolor="white",
              linewidth=0.4, label="protostars")
    ax1.plot(z_grid, expected_counts, color="0.35", lw=2.2, label="predicted (fitted spread)")
    ax1.axvline(0.0, color="black", lw=1.2, linestyle=":", label=r"edge $\xi = 1$")
    ax1.set_xlim(xlo, xhi)
    ax1.set_xlabel(plot_style.label(r"$\mathbf{log_{10}\,\hat{\xi}}$"), fontsize=_LABEL_FONTSIZE)
    ax1.set_ylabel("protostars per bin", fontsize=_LABEL_FONTSIZE, fontweight="bold")
    ax1.tick_params(labelsize=10)
    ax1.legend(fontsize=9, loc="upper left")

    sigma_grid = fit["sigma_grid"]
    gamma_grid = fit["gamma_grid"]
    loglike2d = fit["loglike2d"]
    dloglike = float(np.max(loglike2d)) - loglike2d
    # The grid spans a wide sigma/gamma range (this brief's own domain,
    # 0.03-0.80 dex, 0-3.0), so most of it sits thousands of log-
    # likelihood units below the maximum -- filled contours at levels
    # spaced by factors of about three keep the valley near the adopted
    # point readable and the far plateau a single pale tone.
    fill_levels = [0.0, 1.15, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    mesh = ax2.contourf(sigma_grid, gamma_grid, np.minimum(dloglike, fill_levels[-1] - 1.0),
                         levels=fill_levels, cmap="viridis", extend="neither")
    cbar = fig.colorbar(mesh, ax=ax2, ticks=fill_levels[1:])
    cbar.set_label(plot_style.label(r"$\mathbf{\Delta\,log\,L}$"), fontsize=_LABEL_FONTSIZE)
    cbar.set_ticklabels(["%g" % v for v in fill_levels[1:]])
    cbar.ax.tick_params(labelsize=10)
    contours = ax2.contour(sigma_grid, gamma_grid, dloglike, levels=list(_DLOGLIKE_LEVELS),
                            colors="white", linewidths=1.3)
    ax2.clabel(contours, fmt={lvl: f"{lvl:g}" for lvl in _DLOGLIKE_LEVELS}, fontsize=8)
    ax2.plot(fit["sigma_cloud"], fit["gamma"], marker="*", markersize=16, color="white",
              markeredgecolor="black", markeredgewidth=1.0, linestyle="none", label="adopted")
    ax2.set_xlabel(plot_style.label("spread", "dex"), fontsize=_LABEL_FONTSIZE)
    ax2.set_ylabel("tilt exponent", fontsize=_LABEL_FONTSIZE, fontweight="bold")
    ax2.tick_params(labelsize=10)
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle("Measured Depth Fraction Spread", fontsize=_TITLE_FONTSIZE, fontweight="bold", y=0.99)
    statement = _STATEMENT_TEMPLATE.format(n=n)
    fig.text(0.5, 0.015, statement, fontsize=_STATEMENT_FONTSIZE, ha="center", va="bottom",
              linespacing=1.3)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.20, wspace=0.25)

    out_dir = _figures_dir(config)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "spread_sesna_survey.png")
    pdf_path = os.path.join(out_dir, "spread_sesna_survey.pdf")
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    return png_path, pdf_path


def build(config, regions=None):
    """Builds the one survey-wide figure. `regions` is accepted (the
    package-wide `build(config, regions=None)` convention) and ignored --
    the fit itself is already survey-pooled over the two protostar
    regions."""
    st = progress.Stage("population.figure_kernel")
    fit = _read_fit(config)
    sample = _sample_and_mixture(config, fit)

    emp_median = float(np.median(sample["log10_xi_hat"])) if sample["n_proto"] else float("nan")
    emp_p84 = float(np.percentile(sample["log10_xi_hat"], 84.0)) if sample["n_proto"] else float("nan")
    pred_median = _solve_quantile(0.5, sample) if sample["n_proto"] else float("nan")
    pred_p84 = _solve_quantile(0.84, sample) if sample["n_proto"] else float("nan")

    png_path, pdf_path = _draw(config, fit, sample, emp_median, emp_p84, pred_median, pred_p84)

    print("figure_kernel: N=%d protostars (product's own N_PROTOSTARS_FIT=%d); "
          "adopted spread sigma=%.3f dex (68%% interval %.3f-%.3f), "
          "tilt exponent gamma=%.2f (68%% interval %.2f-%.2f)"
          % (sample["n_proto"], fit["n_protostars_fit"], fit["sigma_cloud"],
             fit["sigma_p16"], fit["sigma_p84"], fit["gamma"], fit["gamma_p16"],
             fit["gamma_p84"]), flush=True)
    print("figure_kernel: log10 xi_hat -- empirical median=%.3f, predicted median=%.3f; "
          "empirical p84=%.3f, predicted p84=%.3f"
          % (emp_median, pred_median, emp_p84, pred_p84), flush=True)
    st.done(png_path, n_protostars=sample["n_proto"], sigma_cloud=fit["sigma_cloud"],
            gamma=fit["gamma"])
    print("figure_kernel: wrote %s and %s" % (png_path, pdf_path), flush=True)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()
    config = config_module.load(args.config)
    build(config, regions=args.regions)


if __name__ == "__main__":
    _main()
