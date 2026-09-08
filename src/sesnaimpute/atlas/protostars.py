"""The protostar check: HOPS (Orion A) and eHOPS (Aquila) against the
prior (SPEC_BMSTP_DRAFT.md sec. 5.5's "Check (report only)"; sec. 9's
protostar-fraction row). Report-only: nothing here feeds the prior or the
posterior. Per region, three panels against `sky.derived.protostars`'s
Herschel-confirmed protostar sample -- position (the prior's cataloged
YSO share at a protostar's pixel versus at every cataloged source's own
pixel), extinction (the protostar's own foreground column against the
prior's `x = a/A_s` marginal), and count (the prior's intrinsic YSO
density map with the protostars overplotted, against Dunham et al. 2014's
protostellar fraction). `CLASS` is read only for this report -- rule 7
("never read a classification label in anything that feeds a prior")
does not apply, since nothing computed here is a prior input.
"""

import argparse
import os

import astropy.units as u
import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.stats import ks_2samp

from sesnaimpute import config as config_module
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute.atlas.render import _add_panel, _align, _colorbar, _footprint_geometry, _log_norm, _reproject
from sesnaimpute.population import selection as population_selection

NSIDE_512 = 512
PAGE_W_IN, PAGE_H_IN = 16.0, 9.0

#: The three catalogue classes the check uses (SPEC_BMSTP_DRAFT.md sec.
#: 5.5's Class 0/I/flat protostellar population); the catalogue's own
#: Class II rows are excluded and only counted.
CLASS_USED = (b"0", b"I", b"flat")

#: `x_proto > X_PROTO_REPORT` is reported by count, disclosed, and kept
#: in the histogram (sec. 5.5's check, this brief's Panel B).
X_PROTO_REPORT = 3.0

#: Dunham et al. 2014, ApJ 783, 29, Table 1: Class 0+I+flat over all
#: young stellar objects, pooled over the c2d+Gould Belt clouds -- the
#: literature protostellar fraction SPEC_BMSTP_DRAFT.md's H2S section
#: (sec. 5.6) treats as a constant.
DUNHAM2014_PROTOSTELLAR_FRACTION = 0.27

#: The fixed-seed resample size-matching draw from the prior's own
#: binned `x` marginal, for a genuine two-sample KS test against the
#: protostars' empirical `log10 x_proto` sample (Panel B).
_KS_RESAMPLE_SEED = 0


def _pix512_galactic(ra_deg, dec_deg):
    """Each position's nside-512 galactic NESTED pixel -- the granule
    map's own pixelisation (`granules/build.py`), reached here by one
    ICRS -> galactic rotation (never a local approximation, sec. 8)."""
    gal = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    return hp.ang2pix(NSIDE_512, gal.l.deg, gal.b.deg, nest=True, lonlat=True)


def _read_protostars(config, region):
    """This region's rows of `sky.derived.protostars`'s pooled HOPS/eHOPS
    sample (`REGION` already the admitted-footprint match, sec. 3)."""
    path = f"{config.data_root}/sky/derived/protostars/protostars_survey.hdf5"
    if not os.path.exists(path):
        raise ValueError(
            f"atlas.protostars: no {path!r} -- run RUNBOOKtp.sh's "
            f"'PY sesnaimpute.sky.derived.protostars' line")
    with h5py.File(path, "r") as f:
        region_col = f["REGION"][:]
        mask = region_col == region.encode("utf-8")
        return dict(
            ra_deg=f["RA_DEG"][:][mask], dec_deg=f["DEC_DEG"][:][mask],
            cls=f["CLASS"][:][mask], av_mag=f["AV_FOREGROUND_MAG"][:][mask])


def _read_prior_atlas(config, region):
    """P6's admitted nside-512 pixels, own column and YSO share, sorted
    by pixel for the `searchsorted` joins below."""
    path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    if not os.path.exists(path):
        raise ValueError(
            f"atlas.protostars: no {path!r} -- run RUNBOOKtp.sh's "
            f"'PY sesnaimpute.bmstp.atlas' line")
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        a_col = np.asarray(f["A_COL_K"][:], dtype=np.float64)
        share_yso = np.asarray(f["SHARE_YSO"][:], dtype=np.float64)
    order = np.argsort(pix)
    return pix[order], a_col[order], share_yso[order]


def _read_cloud_shape(config, region):
    """P3's `X_MARGINAL` per sightline, sec. 5.5's `x` marginal, sorted
    by nside-256 pixel."""
    path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    if not os.path.exists(path):
        raise ValueError(
            f"atlas.protostars: no {path!r} -- run RUNBOOKtp.sh's "
            f"'PY sesnaimpute.bmstp.shapes' line")
    with h5py.File(path, "r") as f:
        pix256 = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        x_marginal = np.asarray(f["X_MARGINAL"][:], dtype=np.float64)
        log10_x_edges = np.asarray(f["LOG10_X_EDGES"][:], dtype=np.float64)
    order = np.argsort(pix256)
    return pix256[order], x_marginal[order], log10_x_edges


def _read_density_table(config, region):
    """P1's per-source `HPX_512` and `DENSITY_YSO` (sec. 5.5's quadratic
    column law, evaluated at each catalogued source's own column)."""
    path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    if not os.path.exists(path):
        raise ValueError(
            f"atlas.protostars: no {path!r} -- run RUNBOOKtp.sh's "
            f"'PY sesnaimpute.bmstp.density' line")
    with h5py.File(path, "r") as f:
        hpx512 = np.asarray(f["HPX_512"][:], dtype=np.int64)
        density_yso = np.asarray(f["DENSITY_YSO"][:], dtype=np.float64)
    return hpx512, density_yso


def _join(atlas_pix, atlas_values, query_pix):
    """`(values, found)`: `atlas_values` at each `query_pix`'s own atlas
    row (`atlas_pix` sorted), `found` False where `query_pix` is not one
    of the atlas's admitted pixels -- one `searchsorted`, no per-item
    loop."""
    loc = np.minimum(np.searchsorted(atlas_pix, query_pix), atlas_pix.size - 1)
    found = atlas_pix[loc] == query_pix
    values = np.full(query_pix.shape, np.nan, dtype=atlas_values.dtype)
    values[found] = atlas_values[loc[found]]
    return values, found


def _panel_a(protostars_pix, atlas_pix, share_yso, density_hpx512):
    """The cumulative distribution of the prior's cataloged YSO share at
    the protostars' pixels versus at every cataloged source's own pixel
    (SPEC_BMSTP_DRAFT.md sec. 5.5 Panel A). Returns the two share arrays
    (protostars, sources) and the printed numbers."""
    share_proto, found = _join(atlas_pix, share_yso, protostars_pix)
    n_dropped = int(np.count_nonzero(~found))
    share_proto = share_proto[found]
    share_source, found_src = _join(atlas_pix, share_yso, density_hpx512)
    share_source = share_source[found_src]
    median_proto = float(np.median(share_proto)) if share_proto.size else float("nan")
    median_source = float(np.median(share_source)) if share_source.size else float("nan")
    frac_above = (float(np.mean(share_proto > median_source))
                  if share_proto.size else float("nan"))
    return dict(share_proto=share_proto, share_source=share_source,
                n_dropped=n_dropped, median_proto=median_proto,
                median_source=median_source, frac_above=frac_above)


def _panel_b(config, protostars_pix, av_mag, atlas_pix, a_col, cloud_pix256, x_marginal, log10_x_edges):
    """The protostars' own scaled extinction `x_proto = a_proto/A_s`
    against the prior's `x` marginal at their own sightlines (sec. 5.5
    Panel B)."""
    a_col_proto, found = _join(atlas_pix, a_col, protostars_pix)
    av_mag = av_mag[found]
    a_col_proto = a_col_proto[found]
    # the blended diffuse/dense-cloud law's own A_K/A_V ratio at the
    # pixel's column (sec. 2: "the blended law's A_K/A_V ratio at the
    # source's column converting Av_hat to a_hat").
    w_ramp = population_selection.law_dense_weight(a_col_proto)
    ak_per_av = population_selection.ak_per_av(config, w_ramp)
    a_proto = av_mag * ak_per_av
    x_proto = a_proto / a_col_proto

    n_av_missing = int(np.count_nonzero(np.isnan(av_mag)))
    n_av_zero = int(np.count_nonzero(av_mag == 0.0))
    finite = np.isfinite(x_proto) & (x_proto > 0.0)
    n_x_exceeds = int(np.count_nonzero(finite & (x_proto > X_PROTO_REPORT)))
    log10_x_proto = np.log10(x_proto[finite])

    # the prior's own prediction: the mean of X_MARGINAL over the
    # protostars' own nside-256 sightlines (pixel -> sightline is the
    # nested parent, pix512 // 4).
    proto_pix256 = protostars_pix[found] // 4
    loc256 = np.minimum(np.searchsorted(cloud_pix256, proto_pix256), cloud_pix256.size - 1)
    found_sl = cloud_pix256[loc256] == proto_pix256
    x_marginal_proto = x_marginal[loc256[found_sl]]
    mean_x_marginal = (x_marginal_proto.mean(axis=0) if x_marginal_proto.shape[0]
                        else np.zeros(log10_x_edges.size - 1))
    probs = mean_x_marginal / mean_x_marginal.sum() if mean_x_marginal.sum() > 0 else mean_x_marginal
    centers = 0.5 * (log10_x_edges[:-1] + log10_x_edges[1:])
    cell_width = float(log10_x_edges[1] - log10_x_edges[0])
    cum = np.concatenate([[0.0], np.cumsum(probs)])
    prior_median = float(np.interp(0.5, cum, np.concatenate([[log10_x_edges[0]], log10_x_edges[1:]])))

    # a genuine two-sample KS test: a fixed-seed resample of the prior's
    # own binned prediction, size-matched to the protostar sample.
    n_used = log10_x_proto.size
    if n_used and probs.sum() > 0:
        synth = np.random.RandomState(_KS_RESAMPLE_SEED).choice(centers, size=n_used, p=probs)
        ks_stat, _ = ks_2samp(log10_x_proto, synth)
    else:
        ks_stat = float("nan")

    return dict(log10_x_proto=log10_x_proto, centers=centers, probs=probs,
                log10_x_edges=log10_x_edges,
                cell_width=cell_width, median_log10_x_proto=(
                    float(np.median(log10_x_proto)) if n_used else float("nan")),
                prior_median=prior_median, ks_stat=float(ks_stat),
                n_av_missing=n_av_missing, n_av_zero=n_av_zero,
                n_sightline_missing=int(np.count_nonzero(~found_sl)),
                n_x_exceeds=n_x_exceeds)


def _panel_c(footprint_pix, density_hpx512, density_yso, n_proto_footprint):
    """The map of the prior's intrinsic YSO density per pixel, the
    protostars overplotted by class, and the implied protostellar
    fraction against Dunham et al. 2014 (sec. 5.5 Panel C)."""
    uniq_pix, inverse = np.unique(density_hpx512, return_inverse=True)
    sum_density = np.bincount(inverse, weights=density_yso)
    count_per_pix = np.bincount(inverse)
    mean_density = sum_density / count_per_pix

    aligned = _align(footprint_pix, uniq_pix, mean_density, np.nan)
    pixel_area_deg2 = float(hp.nside2pixarea(NSIDE_512, degrees=True))
    n_yso_prior = float(np.nansum(aligned) * pixel_area_deg2)
    ratio = n_proto_footprint / n_yso_prior if n_yso_prior > 0 else float("nan")

    geom = _footprint_geometry(footprint_pix)
    grid = _reproject(footprint_pix, aligned, geom["grid_pix"], geom["shape"])
    return dict(geom=geom, grid=grid, n_yso_prior=n_yso_prior, ratio=ratio)


def build_region(config, region, formats=("png", "pdf")):
    """Writes `bmstp/atlas/figures/protostar-check_<region>.png/.pdf`
    and prints the check's numbers (SPEC_BMSTP_DRAFT.md sec. 5.5, sec.
    9's protostellar-fraction row)."""
    with progress.Stage("atlas.protostars", region) as st:
        protostars = _read_protostars(config, region)
        n_class_ii = int(np.count_nonzero(protostars["cls"] == b"II"))
        used = np.isin(protostars["cls"], CLASS_USED)
        ra_u = protostars["ra_deg"][used]
        dec_u = protostars["dec_deg"][used]
        av_u = protostars["av_mag"][used]
        pix_u = _pix512_galactic(ra_u, dec_u)

        atlas_pix, a_col, share_yso = _read_prior_atlas(config, region)
        cloud_pix256, x_marginal, log10_x_edges = _read_cloud_shape(config, region)
        density_hpx512, density_yso = _read_density_table(config, region)

        # Panel A: position -- the prior's cataloged YSO share at a
        # protostar's own pixel versus at every cataloged source's own
        # pixel (a protostar outside the admitted footprint is dropped
        # and counted).
        a = _panel_a(pix_u, atlas_pix, share_yso, density_hpx512)
        n_proto_footprint = pix_u.size - a["n_dropped"]

        # Panel B: extinction -- x_proto = a_proto/A_s against the
        # prior's own x marginal at the protostars' sightlines.
        b = _panel_b(config, pix_u, av_u, atlas_pix, a_col, cloud_pix256, x_marginal, log10_x_edges)

        # Panel C: count -- the intrinsic YSO density map, the
        # protostars overplotted, and the implied protostellar fraction.
        c = _panel_c(atlas_pix, density_hpx512, density_yso, n_proto_footprint)

        print(f"atlas.protostars [{region}]: n_class_ii_excluded={n_class_ii} "
              f"n_dropped_outside_footprint={a['n_dropped']} n_proto_used={n_proto_footprint}")
        print(f"atlas.protostars [{region}] panel A (position): "
              f"median_share_protostars={a['median_proto']:.4g} "
              f"median_share_sources={a['median_source']:.4g} "
              f"frac_protostars_above_source_median={a['frac_above']:.4g}")
        print(f"atlas.protostars [{region}] panel B (extinction): "
              f"n_av_missing={b['n_av_missing']} n_av_zero={b['n_av_zero']} "
              f"n_sightline_missing={b['n_sightline_missing']} "
              f"n_x_proto_gt_{X_PROTO_REPORT:g}={b['n_x_exceeds']} "
              f"median_log10_x_proto={b['median_log10_x_proto']:.4g} "
              f"prior_median_log10_x={b['prior_median']:.4g} ks_stat={b['ks_stat']:.4g}")
        print(f"atlas.protostars [{region}] panel C (count): "
              f"N_proto={n_proto_footprint} N_YSO_prior={c['n_yso_prior']:.4g} "
              f"ratio={c['ratio']:.4g} "
              f"Dunham+2014 Class0+I+flat/all={DUNHAM2014_PROTOSTELLAR_FRACTION:g}")

        paths = _draw_figure(config, region, protostars, n_proto_footprint, a, b, c, formats)
        st.done(paths[0], n_proto_used=n_proto_footprint, n_class_ii=n_class_ii,
                median_share_protostars=a["median_proto"], ks_stat_panel_b=b["ks_stat"],
                n_yso_prior=c["n_yso_prior"], protostellar_fraction=c["ratio"])
    return paths


def _draw_figure(config, region, protostars, n_proto_footprint, a, b, c, formats):
    plot_style.apply_style()
    fig = plot_style.new_sized_figure(PAGE_W_IN, PAGE_H_IN)
    margin_l, margin_r, margin_t, margin_b, gap = 0.7, 0.3, 1.1, 0.6, 0.5
    usable_w = PAGE_W_IN - margin_l - margin_r
    slot_w = (usable_w - 2 * gap) / 3.0
    panel_h = PAGE_H_IN - margin_t - margin_b

    # Panel A -- the two CDFs of the prior's YSO share.
    ax_a = fig.add_axes([margin_l / PAGE_W_IN, margin_b / PAGE_H_IN, slot_w / PAGE_W_IN, panel_h / PAGE_H_IN])
    floor = 1e-6
    for label, values in (("protostars", a["share_proto"]), ("all cataloged sources", a["share_source"])):
        vals = np.sort(np.clip(values, floor, None))
        if vals.size:
            ax_a.plot(vals, np.arange(1, vals.size + 1) / vals.size, label=label)
    ax_a.set_xscale("log")
    ax_a.set_xlabel(plot_style.label("prior cataloged YSO share"))
    ax_a.set_ylabel("cumulative fraction")
    ax_a.set_title("A. position", fontsize=10)
    ax_a.legend(fontsize=7, loc="upper left")

    # Panel B -- the histogram of log10 x_proto against the prior's own
    # x marginal, both normalized to unit area on the common axis.
    ax_b = fig.add_axes([(margin_l + slot_w + gap) / PAGE_W_IN, margin_b / PAGE_H_IN,
                          slot_w / PAGE_W_IN, panel_h / PAGE_H_IN])
    if b["log10_x_proto"].size:
        ax_b.hist(b["log10_x_proto"], bins=b["log10_x_edges"], density=True, alpha=0.5, label="protostars")
    ax_b.plot(b["centers"], b["probs"] / b["cell_width"], label="prior x marginal", color="C1")
    ax_b.set_xlabel(plot_style.label("log10 x_proto"))
    ax_b.set_ylabel("density")
    ax_b.set_title("B. extinction", fontsize=10)
    ax_b.legend(fontsize=7, loc="upper right")

    # Panel C -- the prior's intrinsic YSO density map with the
    # protostars overplotted by class, on the atlas's own footprint.
    wcs = c["geom"]["wcs"]
    rect_c = (margin_l + 2 * (slot_w + gap), margin_b, slot_w - 0.75, panel_h)
    ax_c, im_c = _add_panel(fig, rect_c, PAGE_W_IN, PAGE_H_IN, wcs, c["grid"], "viridis",
                             norm=_log_norm(c["grid"]), title="C. count (deg$^{-2}$)")
    class_colors = {b"0": "white", b"I": "gold", b"flat": "orange", b"II": "red"}
    for cls, color in class_colors.items():
        m = protostars["cls"] == cls
        if np.any(m):
            ax_c.scatter(protostars["ra_deg"][m], protostars["dec_deg"][m],
                         transform=ax_c.get_transform("world"), s=6, color=color,
                         edgecolor="black", linewidths=0.2, label=cls.decode(), zorder=5)
    ax_c.legend(fontsize=6, loc="upper right", markerscale=1.5)
    bar_rect = (rect_c[0] + rect_c[2] + 0.12, rect_c[1] + 0.05 * rect_c[3], 0.35, 0.9 * rect_c[3])
    _colorbar(fig, im_c, bar_rect, PAGE_W_IN, PAGE_H_IN)

    title = (f"{region} -- protostar check: N_proto={n_proto_footprint}, "
             f"N_YSO_prior={c['n_yso_prior']:.4g}, ratio={c['ratio']:.3f} "
             f"(Dunham+2014 {DUNHAM2014_PROTOSTELLAR_FRACTION:g})")
    fig.suptitle(title, fontsize=11, y=1.0 - 0.12 / PAGE_H_IN)

    out_dir = os.path.join(config.data_root, "bmstp", "atlas", "figures")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"protostar-check_{region}.{fmt}")
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths


def _regions_with_protostars(config):
    """Every region with at least one protostar in `REGION` (sec. 5.5's
    default), the union of `sky.derived.protostars`'s own admitted-
    footprint assignment."""
    path = f"{config.data_root}/sky/derived/protostars/protostars_survey.hdf5"
    with h5py.File(path, "r") as f:
        region_col = f["REGION"][:]
    names = np.unique(region_col)
    return [n.decode("utf-8") for n in names if n != b""]


def build(config, regions=None):
    """`build(config, regions=None)`: per region, `build_region`
    (default: every region with at least one protostar in `REGION`, sec.
    5.5 -- in practice Orion A and Aquila)."""
    region_names = regions if regions is not None else _regions_with_protostars(config)
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
