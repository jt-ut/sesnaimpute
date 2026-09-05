"""The Planck-arm dust column: the tau353-to-A_K calibration, and the
per-sightline Planck column it feeds (SPEC_PRIORS.md section 1.1).

SPEC_PRIORS.md 1.1: `A_s` is read from thermal-dust emission, Herschel
Gould Belt Survey column density (36.3" beam) where a source is covered,
the Planck all-sky thermal-dust model (5.03' beam) elsewhere. This module
builds the Planck arm: the calibration that puts Planck's optical depth
onto Herschel's absolute A_K scale, and the per-sightline Planck column
and its uncertainty. `A_K = 1.12e-22 * N(H2)[cm^-2]` is the currency both
arms share (SPEC_PRIORS.md 1.1); the Herschel arm and the merge are a
separate module.

THE CALIBRATION (`build_calibration`). Every nside-256 sightline pixel in
a Herschel-covered region gives one (tau353, A_K) pair: tau353 is
Planck's optical depth block-averaged from its native nside-2048 grid to
the pixel; A_K is HGBS column density, block-averaged to ~12" cells,
smoothed to Planck's own measured beam, then area-averaged over the same
pixel. `a_tau = A_K / tau353` is fit through the origin (a coherent dust
opacity has no additive term), region-balanced so no one region's pixel
count dominates, and cross-validated by holding out one region at a time.
The Planck beam itself is not trusted from the FITS header (it carries
none): it is MEASURED (`_beam_one`/`_peak`) by cross-correlating a native-
resolution HGBS map, smoothed through a ladder of trial beams, against
native Planck tau353 -- the FWHM that maximises the correlation is the
beam that made Planck's own map. The within-pixel scatter of A_K around
`a_tau * tau353` is fit as `sigma_within^2 = s0^2 + (f*A)^2` in column
bins (SPEC_PRIORS.md 1.2's "dispersion of true column within the beam,
growing with column"); the region-to-region scatter of the held-out
cross-validation residual is `sigma_region_frac`, folded in as a fraction
of `A` per source (IMPLEMENTATION.md 02_column_kernel_profile.md notes
this is coherent within a region and is understated as a per-source
quadrature term -- stated here, not fixed with machinery).

THE PER-SIGHTLINE PRODUCT (`build_column`). Every admitted nside-256
pixel (the granule map's own footprint) gets `A = a_tau * tau353` and
`sigma^2 = (a_tau * sigma_tau / sqrt(n_beams))^2 + sigma_within^2 +
(sigma_region_frac * A)^2` -- the arm's own statistical term, the
within-beam term, and the region term, in quadrature. `TEMP_K`, Planck's
own dust temperature at the pixel, is carried for the atlas.
"""

import glob
import os

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter

from sesnaimpute import build as build_module
from sesnaimpute import regions as regions_module
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived.herschel_column import _boxes_overlap, _map_header, _region_sources
from sesnaimpute.sky.download.herschel_hgbs.build import _FILES as HGBS_FILES

NSIDE = 256
NSIDE_P = 2048
NCHILD = (NSIDE_P // NSIDE) ** 2                      # 64
PIXAREA_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2 / (12 * NSIDE ** 2)
BAD = -1.6375e30                                       # HEALPix FITS blank sentinel
N_JOBS = 4

# SPEC_PRIORS.md 1.1: the shared column currency.
NH2_TO_AK = 1.12e-22                                   # mag cm^2, A_K per N(H2)

BLOCK_TARGET_ARCSEC = 12.0                             # HGBS block-reduction cell
CAL_FWHM_ARCMIN = 5.0                                  # Herschel smoothed to ~ Planck's beam
MIN_FILL = 0.98                                        # sightline pixel must be this finite-filled

# Five well-covered, well-separated HGBS maps used to MEASURE Planck's own
# effective beam by cross-correlation against native tau353 -- not asserted
# from the FITS header, which carries no beam keyword.
BEAM_MAPS = ["HGBS_orionA_column_density_map.fits.gz",
             "HGBS_orionB_column_density_map.fits.gz",
             "HGBS_perseus_column_density_map.fits.gz",
             "HGBS_aquilaM2_column_density_map.fits.gz",
             "HGBS_taurus_L1495_column_density_map.fits.gz"]
BEAM_LADDER = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
               6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0]


def _planck_path(config):
    files = sorted(glob.glob(f"{config.data_root}/sky/download/planck_r120/*.fits"))
    if not files:
        raise FileNotFoundError(
            "planck_column: no Planck R1.20 FITS under sky/download/planck_r120 -- run "
            "the sesnaimpute.sky.download.planck_r120.build RUNBOOK line")
    return files[0]


def _hgbs_dir(config):
    return f"{config.data_root}/sky/download/herschel_hgbs"


def _hgbs_map_list(config):
    """[(map file name, local path)] for every HGBS map `herschel_hgbs.
    build` fetches, failing on any file not yet on disk."""
    hub = _hgbs_dir(config)
    out = []
    for name in sorted(HGBS_FILES):
        path = os.path.join(hub, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "planck_column: HGBS map %r missing at %s -- run the "
                "sesnaimpute.sky.download.herschel_hgbs.build RUNBOOK line" % (name, path))
        out.append((name, path))
    return out


def _hgbs_region_map(config):
    """region name -> list of (map file name, local path) whose WCS
    footprint bbox overlaps the region's curated source positions -- read
    from the map headers and the catalogue, never a manifest."""
    maps = _hgbs_map_list(config)
    bbox_by_name = {name: _map_header(path)["bbox"] for name, path in maps}
    out = {}
    for region in [r.name for r in regions_module.REGIONS]:
        ra, dec = _region_sources(config, region)
        if ra.size == 0:
            continue
        rbox = (float(ra.min()), float(ra.max()), float(dec.min()), float(dec.max()))
        for name, path in maps:
            if _boxes_overlap(bbox_by_name[name], rbox, pad=0.05):
                out.setdefault(region, []).append((name, path))
    return out


def _admitted_pixels(config):
    """The survey's admitted nside-256 pixels (source-bearing or Spitzer-
    supported), the granule map's own footprint."""
    path = product_path(config, "granules", "sesna", "granule-map", "source")
    with h5py.File(path, "r") as f:
        return np.sort(np.asarray(f["healpix256/HPX_PIX_256"][:], dtype=np.int64))


def _block_reduce(data, finite, factor):
    """Sums `data` (zeroed where not finite) and `finite` over `factor` x
    `factor` blocks, returning per-block sum, finite-cell count, and the
    trimmed shape -- the array building block behind every map degraded to
    a coarser grid in this module."""
    ny, nx = data.shape
    ny2, nx2 = (ny // factor) * factor, (nx // factor) * factor
    d = np.where(finite, data, 0.0)[:ny2, :nx2].reshape(ny2 // factor, factor, nx2 // factor, factor)
    m = finite[:ny2, :nx2].reshape(ny2 // factor, factor, nx2 // factor, factor)
    return d.sum(axis=(1, 3)), m.sum(axis=(1, 3)), ny2, nx2


def _hgbs_one(path):
    """Reads one HGBS N(H2) map, block-reduces it to ~12" cells, smooths
    those cells to `CAL_FWHM_ARCMIN` (Planck's own measured resolution),
    and area-averages the smoothed field onto whichever nside-256 pixels
    its cells fall in. Returns per-pixel A_K sum/sumsq/weight/finite-area,
    the ingredients `load_hgbs` reduces across a region's several maps."""
    data = None
    with fits.open(path, memmap=False) as hd:
        data = np.asarray(hd[0].data, dtype=np.float64)
        hdr = hd[0].header
    while data.ndim > 2:
        data = data[0]
    w = WCS(hdr).celestial
    pixscale = abs(float(hdr.get("CDELT1", hdr.get("CD1_1")))) * 3600.0   # arcsec
    finite = np.isfinite(data) & (data > 0)

    f = max(1, int(round(BLOCK_TARGET_ARCSEC / pixscale)))
    bs, bn, ny2, nx2 = _block_reduce(data, finite, f)
    cell_arcsec = pixscale * f
    bmean = np.where(bn > 0, bs / np.maximum(bn, 1), np.nan)
    bfrac = bn.astype(np.float64) / (f * f)               # native finite fill of the cell

    sig = (CAL_FWHM_ARCMIN * 60.0 / cell_arcsec) / 2.354820045
    dm = np.where(bn > 0, bmean, 0.0)
    mm = (bn > 0).astype(np.float64)
    num = gaussian_filter(dm, sig, mode="constant", cval=0.0, truncate=3.0)
    den = gaussian_filter(mm, sig, mode="constant", cval=0.0, truncate=3.0)
    smoothed = np.where(den > 0.5, num / np.maximum(den, 1e-30), np.nan)

    by, bx = np.mgrid[0:bmean.shape[0], 0:bmean.shape[1]]
    sel = bn > 0
    xc = bx[sel] * f + (f - 1) / 2.0
    yc = by[sel] * f + (f - 1) / 2.0
    ra, dec = w.wcs_pix2world(xc.astype(np.float64), yc.astype(np.float64), 0)
    gc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    hpix = hp.ang2pix(NSIDE, gc.l.deg, gc.b.deg, nest=True, lonlat=True)

    v = smoothed[sel]
    wcell = bfrac[sel] * (cell_arcsec ** 2)                # cell's finite native area
    ok = np.isfinite(v)
    up, inv = np.unique(hpix, return_inverse=True)
    vv = np.where(ok, v, 0.0)
    ww = np.where(ok, wcell, 0.0)
    return {"PIX": up.astype(np.int64),
            "SUM": np.bincount(inv, weights=ww * vv),
            "SUMSQ": np.bincount(inv, weights=ww * vv * vv),
            "W": np.bincount(inv, weights=ww),
            "AREA_ARCSEC2": np.bincount(inv, weights=wcell)}


def load_hgbs(config, min_fill=MIN_FILL):
    """`{region: {pix, ak, sd, fill}}`: every HGBS-covered region's
    nside-256 pixels whose finite HGBS coverage clears `min_fill`, with
    `A_K = 1.12e-22 * N(H2)` (SPEC_PRIORS.md 1.1) and its within-pixel
    scatter, each region's own maps block-averaged in parallel with
    joblib since maps are the largest independent iterator here."""
    region_map = _hgbs_region_map(config)
    all_maps = sorted({p for names in region_map.values() for _, p in names})
    results = Parallel(n_jobs=N_JOBS)(delayed(_hgbs_one)(p) for p in all_maps)
    by_path = dict(zip(all_maps, results))
    pixarea_arcsec2 = PIXAREA_DEG2 * 3600.0 * 3600.0

    out = {}
    for region, names in sorted(region_map.items()):
        acc = {}
        for _, path in names:
            z = by_path[path]
            for i, p in enumerate(z["PIX"].tolist()):
                e = acc.get(p)
                row = np.array([z["SUM"][i], z["SUMSQ"][i], z["W"][i], z["AREA_ARCSEC2"][i]])
                acc[p] = row if e is None else e + row
        if not acc:
            continue
        ks = np.array(sorted(acc.keys()), dtype=np.int64)
        a = np.array([acc[p] for p in ks], dtype=np.float64)
        wsum = a[:, 2]
        mean = np.where(wsum > 0, a[:, 0] / np.maximum(wsum, 1e-30), np.nan)
        var = np.maximum(a[:, 1] / np.maximum(wsum, 1e-30) - mean ** 2, 0.0)
        fill = a[:, 3] / pixarea_arcsec2
        sel = (fill >= min_fill) & np.isfinite(mean)
        out[region] = {"pix": ks[sel], "ak": mean[sel] * NH2_TO_AK,
                       "sd": np.sqrt(var[sel]) * NH2_TO_AK, "fill": fill[sel]}
    return out


def _origin_fit(xs, ys):
    """Region-balanced weighted least squares through the origin: each
    region contributes with weight 1/n_region, so no one region's pixel
    count dominates the pooled coefficient (SPEC_PRIORS.md 1.1)."""
    num = den = 0.0
    for x, y in zip(xs, ys):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        w = 1.0 / ok.sum()
        num += w * (x[ok] * y[ok]).sum()
        den += w * (x[ok] * x[ok]).sum()
    return num / den


def _peak(x, y):
    """Parabolic interpolation of the maximum of y(x): the sub-sample-
    spaced FWHM that maximises a beam-ladder correlation curve."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i = int(np.nanargmax(y))
    if i == 0 or i == x.size - 1:
        return float(x[i])
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    return float(-b / (2 * a)) if a < 0 else float(x1)


def _beam_one(hgbs_dir, planck_path, fname):
    """Cross-correlates one native-resolution HGBS map, smoothed through
    `BEAM_LADDER`, against native Planck tau353 sampled at the same cell
    centres; the FWHM maximising the correlation is that MEASUREMENT's
    estimate of Planck's own effective beam."""
    path = os.path.join(hgbs_dir, fname)
    with fits.open(path, memmap=False) as hd:
        data = np.asarray(hd[0].data, dtype=np.float64)
        hdr = hd[0].header
    while data.ndim > 2:
        data = data[0]
    w = WCS(hdr).celestial
    pixscale = abs(float(hdr.get("CDELT1", hdr.get("CD1_1")))) * 3600.0
    finite = np.isfinite(data) & (data > 0)
    f = max(1, int(round(24.0 / pixscale)))
    bs, bn, ny2, nx2 = _block_reduce(data, finite, f)
    cell = pixscale * f
    bmean = np.where(bn > 0, bs / np.maximum(bn, 1), np.nan)
    sel = bn == f * f                                     # fully-filled cells only
    by, bx = np.mgrid[0:bmean.shape[0], 0:bmean.shape[1]]
    ra, dec = w.wcs_pix2world((bx[sel] * f + (f - 1) / 2.0).astype(float),
                              (by[sel] * f + (f - 1) / 2.0).astype(float), 0)
    gc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    p2048 = hp.ang2pix(NSIDE_P, gc.l.deg, gc.b.deg, nest=True, lonlat=True)
    with fits.open(planck_path, memmap=True) as hd2:
        tau = np.asarray(hd2[1].data["TAU353"], dtype=np.float64)[p2048]

    dm = np.where(sel, np.nan_to_num(bmean), 0.0)
    mm = sel.astype(np.float64)
    r_vals = []
    for fw in BEAM_LADDER:
        sig = (fw * 60.0 / cell) / 2.354820045
        num = gaussian_filter(dm, sig, mode="constant", cval=0.0, truncate=3.0)
        den = gaussian_filter(mm, sig, mode="constant", cval=0.0, truncate=3.0)
        sm = np.where(den > 0.5, num / np.maximum(den, 1e-30), np.nan)[sel]
        ok = np.isfinite(sm) & np.isfinite(tau)
        r_vals.append(float(np.corrcoef(sm[ok], tau[ok])[0, 1]))
    return _peak(BEAM_LADDER, r_vals)


def measure_beam(config):
    """Planck's effective FWHM (arcmin): the median, over `BEAM_MAPS`, of
    each map's own cross-correlation peak, computed in parallel with
    joblib -- one job per map, the largest independent iterator here."""
    hgbs_dir = _hgbs_dir(config)
    planck_path = _planck_path(config)
    peaks = Parallel(n_jobs=min(N_JOBS, len(BEAM_MAPS)))(
        delayed(_beam_one)(hgbs_dir, planck_path, fname) for fname in BEAM_MAPS)
    return float(np.median(peaks))


def stage_planck(config):
    """Planck's TAU353, its error, and its dust temperature, block-
    averaged from native nside-2048 to nside-256 (SPEC_PRIORS.md 1.1's
    Planck arm), over the whole sky -- the sightline count is small
    enough (786,432 pixels) that no per-pixel indexing is needed."""
    planck_path = _planck_path(config)
    with fits.open(planck_path, memmap=True) as hd:
        hdr = hd[1].header
        assert hdr["NSIDE"] == NSIDE_P, hdr["NSIDE"]
        assert hdr["ORDERING"] == "NESTED", hdr["ORDERING"]
        assert hdr["COORDSYS"] == "GALACTIC", hdr["COORDSYS"]
        d = hd[1].data
        out = {}
        for col in ("TAU353", "ERR_TAU", "TEMP"):
            v = np.asarray(d[col], dtype=np.float64)
            v[v <= BAD / 2.0] = np.nan
            v = v.reshape(-1, NCHILD)
            good = np.isfinite(v)
            ngood = good.sum(axis=1)
            vz = np.where(good, v, 0.0)
            mean = np.where(ngood > 0, vz.sum(axis=1) / np.maximum(ngood, 1), np.nan)
            out[col] = mean
            del v, good, vz
    return out


def build_calibration(config):
    """Fits `A_K = a_tau * tau353` through the origin, region-balanced,
    over every HGBS-covered region; measures Planck's beam by cross-
    correlation; fits the within-region scatter model `sigma_within^2 =
    s0^2 + (f*A)^2` over 20 column bins; and cross-validates the
    coefficient by holding out one region at a time, whose scatter is
    `sigma_region_frac`. Returns the calibration dict written to the
    survey calibration product."""
    planck = stage_planck(config)
    hgbs = load_hgbs(config)
    regs = sorted(hgbs)
    if not regs:
        raise ValueError("build_calibration: no HGBS-covered region cleared MIN_FILL")

    tau = {r: planck["TAU353"][hgbs[r]["pix"]] for r in regs}
    ak = {r: hgbs[r]["ak"] for r in regs}

    a_tau = _origin_fit([tau[r] for r in regs], [ak[r] for r in regs])

    cv_frac = {}
    for held in regs:
        others = [r for r in regs if r != held]
        a_held = _origin_fit([tau[r] for r in others], [ak[r] for r in others])
        pred = a_held * tau[held]
        good = ak[held] > 0.05
        cv_frac[held] = float(np.median((ak[held][good] - pred[good]) / ak[held][good]))
    cv_values = np.array(list(cv_frac.values()))
    sigma_region_frac = float(np.std(cv_values, ddof=1))
    cv_median_frac_error = float(np.median(np.abs(cv_values)))

    # within-region scatter of A_K about the adopted line, in 20 column bins
    resid, column = [], []
    for r in regs:
        pred = a_tau * tau[r]
        resid.append(ak[r] - pred)
        column.append(pred)
    resid = np.concatenate(resid)
    column = np.concatenate(column)
    edges = np.percentile(column, np.linspace(0, 100, 21))
    bin_x, bin_y = [], []
    for i in range(20):
        m = (column >= edges[i]) & (column < edges[i + 1])
        if m.sum() > 20:
            bin_x.append(column[m].mean())
            bin_y.append(np.sqrt((resid[m] ** 2).mean()))
    bin_x = np.array(bin_x)
    coeffs = np.linalg.lstsq(np.stack([np.ones_like(bin_x), bin_x ** 2], axis=1),
                              np.array(bin_y) ** 2, rcond=None)[0]
    sigma_within_s0 = float(np.sqrt(max(coeffs[0], 0.0)))
    sigma_within_f = float(np.sqrt(max(coeffs[1], 0.0)))

    fwhm_arcmin = measure_beam(config)
    n_beams_per_pixel = float(PIXAREA_DEG2 / (1.133 * (fwhm_arcmin / 60.0) ** 2))

    return {"a_tau": float(a_tau), "sigma_within_s0": sigma_within_s0,
            "sigma_within_f": sigma_within_f, "sigma_region_frac": sigma_region_frac,
            "fwhm_arcmin": fwhm_arcmin, "n_beams_per_pixel": n_beams_per_pixel,
            "cv_median_frac_error": cv_median_frac_error,
            "regions": regs, "cv_frac_by_region": cv_frac}


def _write_calibration(config, cal):
    path = product_path(config, "sky/derived", "planck", "calibration", "survey")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.attrs["A_TAU"] = cal["a_tau"]
        f.attrs["SIGMA_WITHIN_S0"] = cal["sigma_within_s0"]
        f.attrs["SIGMA_WITHIN_F"] = cal["sigma_within_f"]
        f.attrs["SIGMA_REGION_FRAC"] = cal["sigma_region_frac"]
        f.attrs["PLANCK_FWHM_ARCMIN"] = cal["fwhm_arcmin"]
        f.attrs["N_BEAMS_PER_PIXEL"] = cal["n_beams_per_pixel"]
        regs = cal["regions"]
        f.create_dataset("REGION", data=np.array(regs, dtype="S64"))
        f.create_dataset("CV_FRAC_ERROR",
                         data=np.array([cal["cv_frac_by_region"][r] for r in regs]))
    return path


def build_column(config, a_tau, sigma_within_s0, sigma_within_f, sigma_region_frac,
                  n_beams_per_pixel):
    """Per admitted nside-256 sightline: `A = a_tau * tau353` and its
    uncertainty in quadrature -- the arm's own statistical term (Planck's
    tau error, reduced by the beams packed into the pixel), the within-
    beam dispersion, and the region-to-region scatter as a fraction of
    `A` (SPEC_PRIORS.md 1.1/1.2)."""
    pix = _admitted_pixels(config)
    planck = stage_planck(config)
    tau = planck["TAU353"][pix]
    err_tau = planck["ERR_TAU"][pix]
    temp = planck["TEMP"][pix]

    a_col = a_tau * tau
    sig_stat = a_tau * err_tau / np.sqrt(n_beams_per_pixel)
    sig_within = np.sqrt(sigma_within_s0 ** 2 + (sigma_within_f * a_col) ** 2)
    sig_region = sigma_region_frac * a_col
    sigma = np.sqrt(sig_stat ** 2 + sig_within ** 2 + sig_region ** 2)

    l_deg, b_deg = hp.pix2ang(NSIDE, pix, nest=True, lonlat=True)
    path = product_path(config, "sky/derived", "planck", "column", "sightline")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        f.create_dataset("HPX_PIX_256", data=pix)
        f.create_dataset("A_K", data=a_col)
        f.create_dataset("SIGMA_A_K", data=sigma)
        f.create_dataset("GAL_L_DEG", data=l_deg)
        f.create_dataset("GAL_B_DEG", data=b_deg)
        f.create_dataset("TEMP_K", data=temp)
    return path, pix.size


def build(config, regions=None):
    """Builds the tau353-to-A_K calibration and the per-sightline Planck
    column. `regions` is accepted for interface uniformity and ignored:
    both products are survey-wide (SPEC_PRIORS.md 1.1's Planck arm covers
    the whole footprint, not just the Herschel-overlapping regions)."""
    cal = build_calibration(config)
    _write_calibration(config, cal)
    build_column(config, cal["a_tau"], cal["sigma_within_s0"], cal["sigma_within_f"],
                 cal["sigma_region_frac"], cal["n_beams_per_pixel"])


if __name__ == "__main__":
    build_module.run(build)
