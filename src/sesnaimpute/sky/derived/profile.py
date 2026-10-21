"""The sightline profile A(d) and region cloud depth (spec SPEC_PRIORS.md 1.4).

`build(config, regions=None)` writes, per region, the cumulative extinction
`A(d)` on the region's occupied nside-256 sightlines: Edenhofer et al. 2023's
mean differential-extinction map, converted E -> A_K through the ZGR23 R(Ks)
ratio, spliced to the deeper 2 kpc release using only its increment past the
517-boundary splice point, then completed to infinity by an exponential
dust-disc tail anchored so `A(inf)` equals the sightline's own total column
(from the adopted sightline column product, SPEC_PRIORS.md 1.1: Herschel
where covered, Planck elsewhere). It also writes one
survey-wide file of each region's cloud distance and line-of-sight depth,
read off the same profiles: the half-width between the 16th/84th percentile
of extinction-weighted distance inside the structure nearest the region's
canon distance.

`read(config, region)` returns the evaluator (`_RegionProfile`, C26
below): `a_of_d(d_pc, hpx_pix, *, total_column_ak)` is the cumulative
extinction to distance `d_pc`; `u(d_pc, ...)` is `A(d)/total_column_ak`.

C25 (briefs/round4_2026-10-20/C25_far_column.md) located the missing red
tail of distant background stars (beyond the Edenhofer cube's edge,
capping a star's column at the pixel's own cloud column undercounts a
plane sightline's real total line-of-sight column by a factor of ~2) and
repaired it with Marshall et al. 2006's own measured far column on
covered sightlines; C25's own report found the repair still left the
atlas's STAR/AGB members and `population.detection_correlation` reading
a CLIPPED placement fraction, never seeing the far reddening a star
beyond the map's edge really carries.

C26 (owner ruling 2026-10-21, briefs/round4_2026-10-20/
C26_profile_adopted.md): ONE canonical 3-D extinction curve per admitted
sightline, built once here and read everywhere, replaces C25's per-read
residual/tail/scale algebra outright. `profile_adopted_sightline`
(`build`'s second product, `_build_adopted_one_region`) stores, per
sightline, Edenhofer's own in-cube shape rescaled so its edge value
equals the sightline's own adopted extinction column (the same `min(1,
total/a_edge)` rule the retired algebra applied), and beyond the cube's
edge Combined19's (Bovy et al. 2016, mwdust) own INCREMENT past that
edge -- Marshall 2006 where its own box covers the sightline, Bayestar19
(Green et al. 2019) elsewhere it reaches, Drimmel 2003 only where
neither exists, `FAR_MAP` recording which, computed from Combined19's
own build rule, never guessed. A reader (`_RegionProfile.a_of_d`) does
one thing: rescale the in-cube part by its own total column over the
sightline's, add the far part unscaled -- `u = A(d) / total` can exceed
1 beyond the edge, same as C25, now on EVERY sightline rather than only
a Marshall-covered one. `profile_edenhofer_sightline` (`build`'s first
product, `_build_one_region`, UNCHANGED by C26) stays exactly as it
was: the raw Edenhofer map's own extraction, the input view this
module's own depth measurement (and a handful of consumers that only
need the map's own in-cube shape, `bmstp.shapes`/`sample_cloud`/
`population.yso`'s cloud-interval construction -- the cloud interval
sits well inside the cube, so they read nothing new) still use;
`profile_adopted_sightline` is the canonical product every OTHER
consumer (population, bmstp's atlas/density/shapes member placement)
reads now. `sky.download.marshall2006` and `sky.derived.
marshall_extinction` are RETIRED (Combined19 already contains
Marshall's own table).
"""

import os

import h5py
import healpy as hp
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.interpolate import InterpolatedUnivariateSpline

from sesnaimpute import build as build_module
from sesnaimpute import constants
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.config import product_path

_PROFILE_STEM = "profile_edenhofer_sightline"
_DEPTH_STEM = "depth_edenhofer_region"
_DEPTH_READINGS = {name: value for (stem, name), value in REGISTRY.items() if stem == _DEPTH_STEM}

#: C26 (owner ruling 2026-10-21): the canonical adopted-profile product's
#: own file stem (`sky/derived/adopted/profile_adopted_sightline__<R>.hdf5`).
_ADOPTED_STEM = "profile_adopted_sightline"

# --- the unit chain (spec 1.4) -------------------------------------------
#
# Edenhofer et al. 2023's map stores unitless E of Zhang, Green & Rix
# (2023) per parsec; ZGR23's own extinction curve gives A_Ks / E at the
# curve's 2159 nm (2MASS Ks) row. No A_V, no colour transform, anywhere.
ZGR23_CURVE_WAVELENGTH_NM_KS = 2159.0
ZGR23_R_KS = 0.2695426046848297

#: The incumbent map's own outer boundary, pc -- the splice point past
#: which only the auxiliary (2 kpc) release's INCREMENT is read.
SPLICE_PC = 1248.10009765625

MAP_INNER_NAME = "mean_and_std_healpix.fits"
MAP_OUTER_NAME = "validation_with_less_data_but_2kpc_mean_and_std_healpix.fits"
CURVE_NAME = "zgr23_extinction_curve.txt"

NSIDE = 256

# --- the far-field tail geometry (spec 1.4) -------------------------------
#
# Drimmel & Spergel (2001), ApJ 556, 181: exponential dust disc, scale
# height 134.4 pc, scale length 2.26 kpc. R_0 is the GRAVITY Collaboration
# (2019, A&A 625, L10) geometric distance to Sgr A*, 8178 pc.
H_Z_PC = 134.4
H_R_PC = 2260.0
R_0_PC = 8178.0
TAIL_LVERT_MAX_PC = 5000.0
DISC_QUAD_MAX_PC = 1.0e5
DISC_QUAD_N = 8001
MODE_VERTICAL = 0
MODE_DISC = 1


def _input_dir(config):
    return f"{config.data_root}/sky/download/edenhofer2023"


def read_zgr23_extinction_curve(path):
    """(wavelength_nm, R) read from the ZGR23 `extinction_curve.txt` --
    the file `ZGR23_R_KS` above was transcribed from."""
    table = np.loadtxt(path, skiprows=1)
    return table[:, 0], table[:, 1]


# --- reading the Edenhofer HEALPix-times-distance map ---------------------


class EdenhoferHealpixMap:
    """The mean differential-extinction map: `density` (n_shell, n_pix) in
    E per pc, `radii`/`bounds` the shell centres/boundaries in pc, `inner`
    the extinction already integrated to `bounds[0]`."""

    def __init__(self, density, radii, bounds, inner, hdul):
        self.density = density
        self.radii = radii
        self.bounds = bounds
        self.inner = inner
        self._hdul = hdul

    @property
    def n_shell(self):
        return self.density.shape[0]

    def close(self):
        if self._hdul is not None:
            self._hdul.close()
            self._hdul = None


def load_healpix_map(path):
    """Open one Edenhofer release FITS file and return its MEAN layer as
    an `EdenhoferHealpixMap`. Only the mean is read; the posterior std.
    is read separately by `read_std_columns` where sigma is needed."""
    hdul = fits.open(path, mode="readonly", memmap=True)
    density = radii = bounds = inner = None
    for hdu in hdul:
        name = hdu.name.lower()
        if isinstance(hdu, fits.PrimaryHDU) and hdu.data is None:
            continue
        if name in ("primary", "mean", "healpix times distance"):
            density = hdu.data
        elif isinstance(hdu, fits.BinTableHDU):
            names = list(hdu.data.names)
            if names == ["radial pixel centers"]:
                radii = np.asarray(hdu.data["radial pixel centers"], dtype=float)
            elif names == ["radial pixel boundaries"]:
                bounds = np.asarray(hdu.data["radial pixel boundaries"], dtype=float)
        elif isinstance(hdu, fits.ImageHDU) and name.startswith(
            ("mean of integrated inner", "integrated inner")
        ):
            inner = hdu.data
    if density is None or radii is None or bounds is None or inner is None:
        hdul.close()
        raise ValueError("map file %s is missing one of density/radii/bounds/inner" % path)
    return EdenhoferHealpixMap(density=density, radii=radii, bounds=bounds, inner=inner, hdul=hdul)


def extract_columns(dust_map, pixels, row_chunk=64):
    """Density columns for `pixels`, (n_shell, n_pix_sel), chunked over
    shells so a handful of angular pixels never pulls the full memmap."""
    pixels = np.asarray(pixels)
    out = np.empty((dust_map.n_shell, pixels.size), dtype=np.float32)
    for start in range(0, dust_map.n_shell, row_chunk):
        stop = min(start + row_chunk, dust_map.n_shell)
        block = np.asarray(dust_map.density[start:stop, :])
        out[start:stop, :] = block[:, pixels]
    return out


def sightline_profiles(dust_map, pixels, to_k_band=True, row_chunk=64):
    """`(distances, a_cum, radii, rho)` for `pixels`, from one pass over
    the map: `distances` the shell boundaries, `a_cum` cumulative A_K
    (n_shell+1, n_pix_sel), `radii` shell centres, `rho` differential
    A_K/pc (n_shell, n_pix_sel)."""
    rho = extract_columns(dust_map, pixels, row_chunk=row_chunk).astype(np.float64)
    if to_k_band:
        rho *= ZGR23_R_KS
    dvol = np.diff(dust_map.bounds)[:, np.newaxis]
    a_cum = np.empty((dust_map.n_shell + 1, rho.shape[1]), dtype=np.float64)
    inner = np.asarray(dust_map.inner)[np.asarray(pixels)].astype(np.float64)
    a_cum[0, :] = inner * ZGR23_R_KS if to_k_band else inner
    np.cumsum(rho * dvol, axis=0, out=a_cum[1:, :])
    a_cum[1:, :] += a_cum[0, :]
    return dust_map.bounds.copy(), a_cum.astype(np.float32), dust_map.radii.copy(), rho


def profile_quantile_distances(distances, a_cum, fractions=(0.16, 0.5, 0.84)):
    """Distances at which each column profile reaches given fractions of
    its own total column; NaN for a profile with non-positive total."""
    a_cum = np.atleast_2d(a_cum)
    total = a_cum[-1, :]
    out = np.full((len(fractions), a_cum.shape[1]), np.nan, dtype=np.float64)
    usable = total > 0
    if not np.any(usable):
        return out
    cols = np.where(usable)[0]
    for i, frac in enumerate(fractions):
        target = frac * total[usable]
        for j, col in enumerate(cols):
            out[i, col] = np.interp(target[j], a_cum[:, col], distances)
    return out


def _smooth_log(values, n_bins):
    """Boxcar smoothing over `n_bins` samples of the map's log-spaced
    radial grid, so the smoothing width is a fixed fraction of distance."""
    if n_bins <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(int(n_bins), dtype=float) / float(int(n_bins))
    padded = np.pad(np.asarray(values, dtype=float), (int(n_bins) // 2, int(n_bins) // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    trim = int(n_bins) // 2
    return smoothed[trim:trim + len(values)]


def structure_depth(radii, rho, d_r, smooth_bins=9, min_prominence=0.05,
                     base_frac=0.10, saddle_frac=0.50, max_frac_depth=0.50):
    """Radial extent of the dust structure containing the cloud at `d_r`
    (spec 1.4): smooth `rho` in log distance, find the prominent local
    maximum nearest `d_r`, walk outward to 10% of the peak or a saddle
    below 50% of it (whichever comes first), and report the 16-84
    percentile half-width of extinction-weighted distance inside that
    bracket. Returns a dict; `ok` is False (numeric fields NaN) when no
    structure separable from the diffuse background is found."""
    radii = np.asarray(radii, dtype=float)
    rho = np.asarray(rho, dtype=float)
    blank = {"ok": False, "reason": "", "d_peak_pc": np.nan, "d_lo_pc": np.nan,
             "d_hi_pc": np.nan, "sigma_depth_pc": np.nan, "fwhm_pc": np.nan}
    if not np.isfinite(d_r) or d_r < radii[0] or d_r > radii[-1] or not np.any(rho > 0):
        blank["reason"] = "region distance outside the map or no positive density"
        return blank

    smooth = _smooth_log(rho, smooth_bins)
    rise = np.diff(smooth)
    is_max = np.zeros(smooth.size, dtype=bool)
    is_min = np.zeros(smooth.size, dtype=bool)
    is_max[1:-1] = (rise[:-1] > 0) & (rise[1:] <= 0)
    is_min[1:-1] = (rise[:-1] <= 0) & (rise[1:] > 0)
    is_min[0] = is_min[-1] = True
    max_idx = np.where(is_max)[0]
    if max_idx.size == 0:
        blank["reason"] = "no local maximum on the sightline"
        return blank

    min_idx = np.where(is_min)[0]
    scale = float(smooth.max())
    keep = []
    for m in max_idx:
        left_c = min_idx[min_idx < m]
        right_c = min_idx[min_idx > m]
        if left_c.size == 0 or right_c.size == 0:
            continue
        prominence = smooth[m] - max(smooth[left_c[-1]], smooth[right_c[0]])
        if scale > 0 and prominence >= min_prominence * scale:
            keep.append(m)
    if not keep:
        blank["reason"] = "no peak clears the prominence floor"
        return blank

    log_r = np.log(radii)
    target = np.log(d_r)
    peak = min(keep, key=lambda m: abs(log_r[m] - target))

    base_level = base_frac * smooth[peak]
    saddle_level = saddle_frac * smooth[peak]
    lo_idx = 0
    for i in range(peak, 0, -1):
        if smooth[i] <= base_level or (is_min[i] and smooth[i] <= saddle_level):
            lo_idx = i
            break
    hi_idx = smooth.size - 1
    for i in range(peak, smooth.size - 1):
        if smooth[i] <= base_level or (is_min[i] and smooth[i] <= saddle_level):
            hi_idx = i
            break

    seg_r = radii[lo_idx:hi_idx + 1]
    if seg_r.size < 3:
        blank["reason"] = "structure spans fewer than three radial shells"
        return blank
    seg_rho = np.clip(rho[lo_idx:hi_idx + 1], 0.0, None)
    weight = seg_rho * np.gradient(seg_r)
    total = weight.sum()
    if not (total > 0):
        blank["reason"] = "structure carries no extinction"
        return blank

    cdf = np.cumsum(weight) / total
    d16, d84 = np.interp([0.16, 0.84], cdf, seg_r)
    sigma = 0.5 * float(d84 - d16)
    d50 = float(np.interp(0.5, cdf, seg_r))

    half = 0.5 * float(smooth[peak])
    left_half = radii[lo_idx]
    for i in range(peak, lo_idx - 1, -1):
        if smooth[i] <= half:
            left_half = radii[i]
            break
    right_half = radii[hi_idx]
    for i in range(peak, hi_idx + 1):
        if smooth[i] <= half:
            right_half = radii[i]
            break

    frac = sigma / d50 if d50 > 0 else np.inf
    ok = frac <= max_frac_depth
    return {"ok": ok,
            "reason": "" if ok else "fractional depth exceeds %.2f" % max_frac_depth,
            "d_peak_pc": float(radii[peak]), "d_lo_pc": float(radii[lo_idx]),
            "d_hi_pc": float(radii[hi_idx]), "sigma_depth_pc": sigma,
            "fwhm_pc": float(right_half - left_half)}


# --- reading the map's sigma layer and splicing the two releases ---------


def read_std_columns(path, pixels, shell_slice=None, row_chunk=64, want_inner=True):
    """STD. columns for `pixels`, native units (E per pc, E for the inner
    layer): `(std[n_shell_sel, n_pix], inner_std[n_pix] or None)`."""
    pixels = np.asarray(pixels)
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        sh = hdul["STD."]
        n_shell = sh.data.shape[0]
        lo, hi = (0, n_shell) if shell_slice is None else shell_slice
        out = np.empty((hi - lo, pixels.size), dtype=np.float32)
        for start in range(lo, hi, row_chunk):
            stop = min(start + row_chunk, hi)
            block = np.asarray(sh.data[start:stop, :])
            out[start - lo:stop - lo, :] = block[:, pixels]
        inner_std = None
        if want_inner:
            for hdu in hdul:
                if hdu.name.lower().startswith("std. of integrated inner"):
                    inner_std = np.asarray(hdu.data, dtype=np.float64)[pixels]
                    break
            if inner_std is None:
                raise ValueError("no integrated-inner STD. HDU in %s" % path)
    return out.astype(np.float64), inner_std


def read_mean_columns(path, pixels, shell_slice, row_chunk=64):
    """MEAN columns for `pixels` over shells `[lo, hi)`, native unit."""
    pixels = np.asarray(pixels)
    lo, hi = shell_slice
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        mh = hdul["MEAN"]
        out = np.empty((hi - lo, pixels.size), dtype=np.float32)
        for start in range(lo, hi, row_chunk):
            stop = min(start + row_chunk, hi)
            block = np.asarray(mh.data[start:stop, :])
            out[start - lo:stop - lo, :] = block[:, pixels]
    return out.astype(np.float64)


def read_radial_axes(path):
    """`(boundaries, centres)` of a map's radial axis, pc."""
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        bounds = np.asarray(hdul["RADIAL PIXEL BOUNDARIES"].data.field(0), dtype=float)
        centers = np.asarray(hdul["RADIAL PIXEL CENTERS"].data.field(0), dtype=float)
    return bounds, centers


def propagate(std_ak, widths, base_unc=None, base_cor=None):
    """Cumulative sigma from per-shell A_K contributions: `(unc, cor)`,
    each (n_pix, n_shell+1), the uncorrelated and fully-correlated sums."""
    s = std_ak.T * widths
    n_pix = s.shape[0]
    b_u = np.zeros(n_pix) if base_unc is None else np.asarray(base_unc)
    b_c = np.zeros(n_pix) if base_cor is None else np.asarray(base_cor)
    unc = np.concatenate([b_u[:, None] ** 2, b_u[:, None] ** 2 + np.cumsum(s ** 2, axis=1)], axis=1)
    cor = np.concatenate([b_c[:, None], b_c[:, None] + np.cumsum(s, axis=1)], axis=1)
    return np.sqrt(unc), cor


def read_inner_map(map_inner, pixels, r_ks):
    """The incumbent (1248.1 pc) map's contribution for `pixels`: grid,
    A(d), rho, and sigma."""
    dust_map = load_healpix_map(map_inner)
    try:
        dist1, a_cum1, radii1, rho1 = sightline_profiles(dust_map, pixels)
    finally:
        dust_map.close()
    if dist1.size != 517 or abs(dist1[-1] - SPLICE_PC) >= 1e-9:
        raise ValueError("%s: expected the 517-boundary grid ending at %.5f pc, got %d ending at %.5f"
                         % (map_inner, SPLICE_PC, dist1.size, dist1[-1]))
    a_cum1_T = np.ascontiguousarray(a_cum1.T)
    std1_E, inner_std1_E = read_std_columns(map_inner, pixels)
    sig1_unc, sig1_cor = propagate(std1_E * r_ks, np.diff(dist1),
                                   base_unc=inner_std1_E * r_ks, base_cor=inner_std1_E * r_ks)
    return {"dist1": dist1, "a_cum1_T": a_cum1_T, "radii1": radii1, "rho1": rho1,
            "sig1_unc": sig1_unc, "sig1_cor": sig1_cor}


def read_outer_increment(map_outer, pixels, r_ks):
    """The auxiliary (1995 pc) map's INCREMENT past `SPLICE_PC` only, for
    `pixels`: the two reconstructions' offset cancels out of a difference
    and never enters A(d)."""
    bnd2, cen2 = read_radial_axes(map_outer)
    k = int(np.searchsorted(bnd2, SPLICE_PC, side="right"))
    n_out = bnd2.size - k
    if not (bnd2[k - 1] <= SPLICE_PC < bnd2[k]):
        raise ValueError("%s: the splice point does not lie in shell %d [%.5f, %.5f)"
                         % (map_outer, k - 1, bnd2[k - 1], bnd2[k]))
    rho2_E = read_mean_columns(map_outer, pixels, (k - 1, bnd2.size - 1))
    std2_E, _ = read_std_columns(map_outer, pixels, (k - 1, bnd2.size - 1), want_inner=False)
    rho2 = rho2_E * r_ks
    std2 = std2_E * r_ks
    widths2 = np.diff(np.concatenate([[SPLICE_PC], bnd2[k:]]))
    dA2 = np.cumsum(rho2.T * widths2, axis=1)
    s2_terms = (std2.T * widths2).T
    return {"bnd2": bnd2, "cen2": cen2, "k": k, "dA2": dA2, "s2_terms": s2_terms, "rho2": rho2}


def splice_axes(dist1, radii1, bnd2, cen2, k):
    """`(dist_new, rho_dist_new)`: the inner map's boundaries followed by
    the auxiliary map's boundaries past the splice, and the matching
    shell-centre grid (the joining shell's centre is the geometric mean of
    the splice point and the first auxiliary boundary)."""
    dist_new = np.concatenate([dist1, bnd2[k:]])
    rho_dist_new = np.concatenate([radii1, [np.sqrt(SPLICE_PC * bnd2[k])], cen2[k:bnd2.size - 1]])
    return dist_new, rho_dist_new


# --- the far-field tail (spec 1.4) ----------------------------------------


def tail_shape_vertical(d_pc, d_edge_pc, scale_pc):
    """`1 - exp(-(d - d_edge)/L)`: the out-of-plane tail shape."""
    x = np.clip(np.asarray(d_pc, float) - float(d_edge_pc), 0.0, None)
    return -np.expm1(-x / float(scale_pc))


def disc_quadrature(l_deg, b_deg, d_edge_pc, h_z_pc=H_Z_PC, h_R_pc=H_R_PC, r_0_pc=R_0_PC,
                    dmax=DISC_QUAD_MAX_PC, n=DISC_QUAD_N):
    """`(s, F)`: the in-plane branch's normalised cumulative integral
    through the Drimmel & Spergel exponential disc, rising from 0 at
    `d_edge` to 1 at `dmax`."""
    lr, br = np.radians(float(l_deg)), np.radians(float(b_deg))
    s = float(d_edge_pc) + np.geomspace(1.0e-2, dmax - float(d_edge_pc), n)
    z = s * np.sin(br)
    rp = s * np.cos(br)
    r = np.sqrt(r_0_pc ** 2 + rp ** 2 - 2.0 * r_0_pc * rp * np.cos(lr))
    rho = np.exp(-np.abs(z) / h_z_pc - (r - r_0_pc) / h_R_pc)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(s))])
    return s, cum / cum[-1]


def efold_pc(s, f, d_edge_pc):
    """Distance past `d_edge` at which the tail shape reaches `1 - 1/e`."""
    return float(np.interp(1.0 - 1.0 / np.e, f, s) - d_edge_pc)


def _weighted_quantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(int(np.searchsorted(cw, q)), x.size - 1)])


def _tail_geometry(gl, gb, nsrc, d_edge):
    """Per-sightline tail geometry and the region's fallback sightline
    (the source-weighted mean direction): geometry only (mode, scale,
    e-fold length), independent of any column value. The vertical branch
    is `1-exp(-(d-d_edge)/L)` with `L = h_z/|sin b|`; the in-plane branch
    (where `L` would exceed `TAIL_LVERT_MAX_PC`) is the disc quadrature's
    own e-fold. Returns `(mode, scale, efold, ref)`."""
    sinb = np.abs(np.sin(np.radians(gb)))
    with np.errstate(divide="ignore"):
        lvert = np.where(sinb > 0, H_Z_PC / np.maximum(sinb, 1e-300), np.inf)
    mode = np.where(lvert > TAIL_LVERT_MAX_PC, MODE_DISC, MODE_VERTICAL).astype(np.int8)
    scale = np.where(mode == MODE_VERTICAL, lvert, np.nan)
    efold = np.array(scale, float)
    for i in np.nonzero(mode == MODE_DISC)[0]:
        s, f = disc_quadrature(gl[i], gb[i], d_edge)
        efold[i] = efold_pc(s, f, d_edge)

    w = nsrc / nsrc.sum()
    lr, br = np.radians(gl), np.radians(gb)
    vx = np.sum(w * np.cos(br) * np.cos(lr))
    vy = np.sum(w * np.cos(br) * np.sin(lr))
    vz = np.sum(w * np.sin(br))
    l_ref = float(np.degrees(np.arctan2(vy, vx)) % 360.0)
    b_ref = float(np.degrees(np.arctan2(vz, np.hypot(vx, vy))))
    lv_ref = H_Z_PC / max(abs(np.sin(np.radians(b_ref))), 1e-300)
    if lv_ref > TAIL_LVERT_MAX_PC:
        mode_ref = MODE_DISC
        s, f = disc_quadrature(l_ref, b_ref, d_edge)
        ef_ref, sc_ref = efold_pc(s, f, d_edge), np.nan
    else:
        mode_ref, sc_ref, ef_ref = MODE_VERTICAL, lv_ref, lv_ref

    ref = dict(l_ref=l_ref, b_ref=b_ref, mode_ref=mode_ref, scale_ref=sc_ref, efold_ref=ef_ref)
    return mode, scale, efold, ref


def _edge_density(a_out, dist_new, h_z_pc):
    """The map's own mean density, A_K/pc, over the last `h_z_pc` of its
    distance axis (spec 1.4): `(A_edge - A(d_edge - h_z_pc)) / h_z_pc`,
    read off the map's own (unscaled) cumulative profile `a_out`."""
    d_edge = float(dist_new[-1])
    d_ref = d_edge - h_z_pc
    idx = int(np.clip(np.searchsorted(dist_new, d_ref), 1, dist_new.size - 1))
    d0, d1 = dist_new[idx - 1], dist_new[idx]
    frac = (d_ref - d0) / (d1 - d0)
    a_ref = a_out[:, idx - 1] + frac * (a_out[:, idx] - a_out[:, idx - 1])
    return (a_out[:, -1] - a_ref) / h_z_pc


def _far_field_residual(a_edge_raw, a_inf, rho_edge, efold):
    """The column beyond the map's own edge, and the scale that spreads
    the remainder inside the map (spec 1.4, last paragraph). The excess
    `A_inf - A_edge` is capped at what the map's own edge density
    implies through the sightline's own tail shape, `A_tail,map =
    rho_edge * efold`: at high latitude the map under-resolves dense
    cloud that the star-colour column already sees (median share of the
    column that `max(0, A_inf - A_edge)` alone puts beyond the map's 2 kpc edge: 0.43
    Ophiuchus, 0.37 Aquila, 0.29 Pipe, 0.14 Orion A, 0.08 Perseus,
    against the map's own edge-density share 0.000 / 0.014 / 0.21 /
    0.013 / 0.000 -- so that column belongs inside the cloud instead.
    `residual = min(max(0, A_inf - A_edge), A_tail,map)`: never more
    than the raw excess, never more than the map itself implies. The
    remainder of the excess is spread inside the map in the map's own
    shape by scaling the in-map profile (and its density) by `s =
    (A_inf - residual)/A_edge >= 1`. The existing rescale-DOWN case (the
    map's own edge already exceeds the total column, residual 0) is the
    same formula with `s <= 1`; both are one scaling on `s` here."""
    raw_excess = np.maximum(0.0, a_inf - a_edge_raw)
    a_tail_map = rho_edge * efold
    capped = raw_excess > a_tail_map
    residual = np.where(capped, a_tail_map, raw_excess)
    down = a_edge_raw > a_inf
    denom = np.where(a_edge_raw > 0, a_edge_raw, 1.0)
    s = np.where(down, a_inf / denom, np.where(capped, (a_inf - residual) / denom, 1.0))
    return residual, s, capped


def region_weight(n_src):
    """The region's fallback (no-`hpx_pix`) mean profile is weighted by
    each admitted sightline's share of the region's catalogued sources."""
    weight = n_src.astype(float)
    return weight / weight.sum()


# --- per-region measurement (spliced grid only, float64 throughout) ------


def measure_region(d_r_pc, inner, splice, dist_new, rho_dist_new, weight):
    """One region's spliced-grid profile and depth. `inner`/`splice` are
    this region's own `read_inner_map`/`read_outer_increment` results;
    `weight` is `region_weight`'s per-sightline weight. Computed once, in
    float64, on the spliced grid -- no inner-grid duplicate, no float32
    round-trip."""
    a_cum1_T = inner["a_cum1_T"].astype(np.float64)
    dA2 = splice["dA2"]
    a_out = np.concatenate([a_cum1_T, a_cum1_T[:, -1:] + dA2], axis=1)

    mean_rho_inner = (inner["rho1"] * weight[np.newaxis, :]).sum(axis=1)
    rho_outer = (splice["rho2"] * weight[np.newaxis, :]).sum(axis=1)
    rho_mean_full = np.concatenate([mean_rho_inner, rho_outer])

    depth = structure_depth(rho_dist_new, rho_mean_full, d_r_pc)
    return {"a_out": a_out, "depth": depth}


# --- build ------------------------------------------------------------


def _admitted_sightlines(config, region):
    """The region's admitted sightlines (spec IMPLEMENTATION.md 1): every
    nside-256 galactic-NESTED pixel the granule map associates with
    `region` -- source-bearing, full stop (every row's N_SOURCE_ROWS > 0)
    -- and each row's source count."""
    path = product_path(config, "granules", "sesna", "granule-map", "source")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "profile.build: granule map missing at %r -- run the "
            "sesnaimpute.granules.build RUNBOOK line for it" % path)
    with h5py.File(path, "r") as f:
        names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in f["region/REGION"][:]]
        if region not in names:
            raise ValueError("profile.build: region %r is absent from the granule map %s" % (region, path))
        code = int(f["region/REGION_CODE"][names.index(region)])
        assoc = f["association/region_healpix256"]
        assoc_code = np.asarray(assoc["REGION_CODE"][:], dtype=np.int64)
        sel = assoc_code == code
        pix = np.asarray(assoc["HPX_PIX_256"][:], dtype=np.int64)[sel]
        nsrc = np.asarray(assoc["N_SOURCE_ROWS"][:], dtype=np.float64)[sel]
    order = np.argsort(pix)
    return pix[order], nsrc[order]


def _join_total_column(config, region, admitted_pix):
    """The sightline's total column to infinity, A_K, from the adopted
    extinction sightline product(the emission column
    scaled to the NICEST star-colour map's own beam-scale factor, so
    `u(d)` runs to the extinction total, not the bare gas column)."""
    path = product_path(config, "sky/derived", "adopted", "extinction", "sightline")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "profile.build: adopted extinction sightline product missing at %s -- run the "
            "sesnaimpute.sky.derived.column RUNBOOK line for it" % path)
    # column.py imports this module, so its region-code reader is imported here
    from sesnaimpute.sky.derived.column import _region_codes
    code = _region_codes(config, [region])[region]
    with h5py.File(path, "r") as f:
        own = np.asarray(f["REGION_CODE"][:], dtype=np.int64) == code
        hpx = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)[own]
        a_inf = np.asarray(f["A_K"][:], dtype=float)[own]
    order = np.argsort(hpx)
    hpx_sorted = hpx[order]
    pos = np.searchsorted(hpx_sorted, admitted_pix)
    capped = np.minimum(pos, hpx_sorted.size - 1) if hpx_sorted.size else pos
    ok = hpx_sorted.size and (hpx_sorted[capped] == admitted_pix)
    if not np.all(ok):
        missing = admitted_pix[~np.asarray(ok)][:5]
        raise ValueError("profile.build: %s has no total column for sightline(s) %s of region %r"
                         % (path, missing.tolist(), region))
    return a_inf[order][capped]


def _build_one_region(config, region, canon, input_dir):
    admitted_pix, nsrc = _admitted_sightlines(config, region)
    a_inf = _join_total_column(config, region, admitted_pix)
    gl, gb = hp.pix2ang(NSIDE, admitted_pix, nest=True, lonlat=True)

    inner_path = f"{input_dir}/{MAP_INNER_NAME}"
    outer_path = f"{input_dir}/{MAP_OUTER_NAME}"
    for path in (inner_path, outer_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "profile.build: Edenhofer input missing at %s -- run the "
                "sesnaimpute.sky.download.edenhofer2023.build RUNBOOK line for it" % path)
    inner = read_inner_map(inner_path, admitted_pix, ZGR23_R_KS)
    splice = read_outer_increment(outer_path, admitted_pix, ZGR23_R_KS)
    dist_new, rho_dist_new = splice_axes(inner["dist1"], inner["radii1"], splice["bnd2"], splice["cen2"], splice["k"])

    weight = region_weight(nsrc)
    measured = measure_region(canon.d_r_pc, inner, splice, dist_new, rho_dist_new, weight)
    a_out, depth = measured["a_out"], measured["depth"]
    # C26 (owner ruling 2026-10-21): the RAW, unscaled splice-grid profile
    # -- `_build_adopted_one_region` rescales it its own way, by the
    # sightline's own adopted column alone, never through the residual/
    # tail-shape algebra below (which stays, unchanged, for this file's
    # own profile_edenhofer_sightline output). A plain copy, captured
    # before that algebra mutates `a_out` in place two lines down.
    a_out_raw_for_adopted = a_out.copy()

    d_edge = dist_new[-1]
    a_edge_raw = a_out[:, -1]
    mode, scale_geom, efold, ref = _tail_geometry(gl, gb, nsrc, d_edge)

    # spec 1.4, last paragraph: the column beyond the map's own edge is
    # bounded by what the map's own edge density implies through the
    # sightline's own tail shape (_far_field_residual); the remainder of
    # the excess is spread inside the map by scaling the in-map profile
    # (and its density) by s -- see that docstring for the rule and why.
    rho_edge = _edge_density(a_out, dist_new, H_Z_PC)
    residual, s, capped = _far_field_residual(a_edge_raw, a_inf, rho_edge, efold)
    rescaled = s != 1.0
    a_out = a_out * s[:, np.newaxis]
    a_edge = a_out[:, -1]
    rescaled_frac = float(np.mean(rescaled))
    print("profile.build: %s: %d/%d sightlines rescaled (%.1f%%), %d/%d far-field capped to the map's edge density"
          % (region, int(np.count_nonzero(rescaled)), rescaled.size, 100.0 * rescaled_frac,
             int(np.count_nonzero(capped)), capped.size))

    w = nsrc / nsrc.sum()
    fallback = dict(a_edge_ref_k=float(np.sum(w * a_edge)), a_col_sightline_ref_k=float(np.sum(w * a_inf)),
                    residual_ref_k=float(np.sum(w * residual)), gal_l_ref_deg=ref["l_ref"], gal_b_ref_deg=ref["b_ref"],
                    tail_mode_ref=ref["mode_ref"], tail_scale_ref_pc=ref["scale_ref"], tail_efold_ref_pc=ref["efold_ref"])
    tail = dict(residual=residual, mode=mode, scale=scale_geom, efold=efold)

    dist_pc = np.concatenate([[0.0], dist_new])
    a_cum_k = np.concatenate([np.zeros((admitted_pix.size, 1)), a_out], axis=1)
    rho_k_per_pc = np.diff(a_cum_k, axis=1) / np.diff(dist_pc)[np.newaxis, :]
    fallback_a_cum = weight @ a_cum_k

    out_path = product_path(config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        for name, data in (
            ("HPX_PIX_256", admitted_pix),
            ("DIST_PC", dist_pc),
            ("A_CUM_K", a_cum_k),
            ("RHO_K_PER_PC", rho_k_per_pc),
            ("GAL_L_DEG", gl),
            ("GAL_B_DEG", gb),
            ("TAIL_RESIDUAL_K", tail["residual"]),
            ("TAIL_MODE", tail["mode"]),
            ("TAIL_SCALE_PC", tail["scale"]),
            ("TAIL_EFOLD_PC", tail["efold"]),
            ("A_INF_K", a_inf),
            ("RESCALED", rescaled),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_PROFILE_STEM, name)])
        fb = f.create_group("fallback")
        for name, data in (
            ("DIST_PC", dist_pc),
            ("A_CUM_K", fallback_a_cum),
            ("A_EDGE_K", fallback["a_edge_ref_k"]),
            ("A_COL_SIGHTLINE_K", fallback["a_col_sightline_ref_k"]),
            ("RESIDUAL_K", fallback["residual_ref_k"]),
            ("GAL_L_DEG", fallback["gal_l_ref_deg"]),
            ("GAL_B_DEG", fallback["gal_b_ref_deg"]),
            ("TAIL_MODE", fallback["tail_mode_ref"]),
            ("TAIL_SCALE_PC", fallback["tail_scale_ref_pc"]),
            ("TAIL_EFOLD_PC", fallback["tail_efold_ref_pc"]),
        ):
            build_module.write_dataset(fb, name, data, *REGISTRY[(_PROFILE_STEM, f"fallback/{name}")])

    return dict(region=region, d_r_pc=canon.d_r_pc, sigma_d_pc=canon.sigma_pc,
                d_peak_pc=depth["d_peak_pc"], d_lo_pc=depth["d_lo_pc"], d_hi_pc=depth["d_hi_pc"],
                sigma_depth_pc=depth["sigma_depth_pc"], fwhm_pc=depth["fwhm_pc"], depth_ok=bool(depth["ok"]),
                rescaled_frac=rescaled_frac,
                # C26: the pieces `_build_adopted_one_region` needs, so it
                # builds the canonical profile from the SAME one pass over
                # the Edenhofer maps instead of re-reading them.
                admitted_pix=admitted_pix, gl=gl, gb=gb, dist_new=dist_new,
                a_out_raw=a_out_raw_for_adopted, a_inf=a_inf)



# ---------------------------------------------------------------------------
# C26 (owner ruling 2026-10-21): the canonical adopted-profile product --
# Combined19's hierarchical-HEALPix pixel lookup and linear-in-distance-
# modulus interpolation, reproduced with h5py+healpy alone (no mwdust
# import at run time, this module's own docstring).
# ---------------------------------------------------------------------------

#: Bovy et al. 2016's own 120-node distance-modulus grid (mwdust's
#: `Combined19.__init__`'s own `self._distmods`): the SAME grid
#: Bayestar19's `bayestar2019.h5` names in its `DM_bin_edges` attribute --
#: one shared axis for both files this module reads.
_DISTMODS_120 = np.linspace(4.0, 18.875, 120)

#: The far-field tail's own log-spaced grid past the Edenhofer cube's
#: edge, out to Combined19/Marshall's own ~10 kpc reach -- C25's
#: `sky.derived.marshall_extinction`'s own `TAIL_MAX_PC`/`TAIL_N_NODES`,
#: and (C26) `population.star_population.shared_distance_grid`'s own
#: former tail, now built ONCE here: both read `profile_adopted_
#: sightline`'s own `DIST_PC` instead of reconstructing it.
ADOPTED_TAIL_MAX_PC = 10_000.0
ADOPTED_TAIL_N_NODES = 64

#: Combined19's own build rule (mwdust/util/combine_dustmaps19.py:
#: `inMar`), reproduced here so `FAR_MAP` is COMPUTED, never guessed: the
#: box Marshall et al. 2006's own survey covers (their sec. 2, widened by
#: half a Combined19 nside-512 cell to the exact box the build script
#: tests on that grid).
FAR_MAP_L_MAX_DEG = 100.125
FAR_MAP_B_MAX_DEG = 10.125

#: `FAR_MAP` codes (module docstring, C26 "The product"): which of
#: Combined19's own three constituents supplies a sightline's far column.
FAR_MAP_MARSHALL = 1
FAR_MAP_BAYESTAR = 2
FAR_MAP_DRIMMEL = 3


def _hierarchical_lookup_rows(pixel_info, l_deg, b_deg):
    """Row index into `pixel_info` (a `(nside, healpix_index, ...)`
    recarray, Bovy's own hierarchical-HEALPix layout) for each `(l_deg,
    b_deg)`, NESTED ordering, finest `nside` tried first --
    `mwdust.HierarchicalHealpixMap._lbIndx`'s own algorithm, reproduced
    with h5py+healpy alone (module docstring: no mwdust import at run
    time). `-1` where no level of the hierarchy carries a pixel there."""
    l_deg = np.atleast_1d(np.asarray(l_deg, dtype=np.float64))
    b_deg = np.atleast_1d(np.asarray(b_deg, dtype=np.float64))
    nsides = np.unique(np.asarray(pixel_info["nside"]))[::-1]
    row = np.full(l_deg.size, -1, dtype=np.int64)
    found = np.zeros(l_deg.size, dtype=bool)
    for nside in nsides:
        sel = pixel_info["nside"] == nside
        hidx = np.asarray(pixel_info["healpix_index"])[sel].astype(np.int64)
        rows = np.flatnonzero(sel)
        if hidx.size == 0:
            continue
        order = np.argsort(hidx)
        hidx_sorted = hidx[order]
        rows_sorted = rows[order]
        pidx = hp.ang2pix(int(nside), l_deg, b_deg, nest=True, lonlat=True)
        j = np.searchsorted(hidx_sorted, pidx)
        j_clamped = np.minimum(j, hidx_sorted.size - 1)
        ok = (~found) & (hidx_sorted[j_clamped] == pidx)
        row[ok] = rows_sorted[j_clamped[ok]]
        found |= ok
    return row


def _dist_kpc_to_distmod(d_kpc):
    """Bovy's own distance-modulus convention (mwdust `_evaluate`):
    `distmod = 5 log10(d_kpc) + 10`."""
    return 5.0 * np.log10(np.asarray(d_kpc, dtype=np.float64)) + 10.0


def _spline_curve_ak(ebv_row, dm_query):
    """mwdust's own `interpk=1` reconstruction
    (`scipy.interpolate.InterpolatedUnivariateSpline(distmods, best_fit[i],
    k=1)`, `mwdust.HierarchicalHealpixMap._evaluate`) at `dm_query`,
    converted from Combined19's own E(B-V) (SFD scale) to K-band
    extinction (`constants.AK_PER_EBV_SFD_KS`): the SAME scipy spline
    mwdust itself calls, so this reproduces its interpolation exactly;
    only the pixel LOOKUP above is this module's own code."""
    spline = InterpolatedUnivariateSpline(_DISTMODS_120, ebv_row, k=1)
    return spline(dm_query) * constants.AK_PER_EBV_SFD_KS


def combined19_curve_ak(combined19_path, l_deg, b_deg, dm_query):
    """`(n_sightline, n_query)` A_Ks, Combined19's own value at each
    `(l_deg[i], b_deg[i])` and each distance modulus in `dm_query`,
    read from the root's own `combine19.h5` (`sky.download.combined19`)
    by h5py alone: `pixel_info` (small, read whole) locates each
    sightline's own row by `_hierarchical_lookup_rows`; `best_fit` (the
    3.9 GB array) is read only at the DISTINCT rows this region's own
    sightlines actually need (module docstring: never the whole array)."""
    l_deg = np.atleast_1d(np.asarray(l_deg, dtype=np.float64))
    b_deg = np.atleast_1d(np.asarray(b_deg, dtype=np.float64))
    with h5py.File(combined19_path, "r") as f:
        pixel_info = f["pixel_info"][:]
        rows = _hierarchical_lookup_rows(pixel_info, l_deg, b_deg)
        if np.any(rows < 0):
            bad = np.flatnonzero(rows < 0)[:5]
            raise ValueError(
                "sky.derived.profile: %d sightline(s) have no Combined19 pixel at any "
                "hierarchy level, e.g. (l=%r, b=%r)"
                % (int(np.count_nonzero(rows < 0)), l_deg[bad].tolist(), b_deg[bad].tolist()))
        uniq_rows, inverse = np.unique(rows, return_inverse=True)
        best_fit = f["best_fit"]
        curves = np.empty((uniq_rows.size, dm_query.size), dtype=np.float64)
        for i, r in enumerate(uniq_rows):
            ebv_row = np.asarray(best_fit[int(r), :], dtype=np.float64)
            curves[i] = _spline_curve_ak(ebv_row, dm_query)
    return curves[inverse]


def bayestar_lookup(bayestar_path, l_deg, b_deg):
    """`(row, far_reliable_pc)`: each `(l_deg, b_deg)`'s own row into
    Bayestar19's `bayestar2019.h5` pixel_info (same hierarchical lookup,
    its own native pixelization -- a DIFFERENT file from Combined19's),
    `-1` where it carries none, and that row's own `DM_reliable_max`
    converted to pc (`10**(DM/5 + 1)`, the ordinary distance-modulus
    inverse) -- NaN where the row is `-1`."""
    l_deg = np.atleast_1d(np.asarray(l_deg, dtype=np.float64))
    b_deg = np.atleast_1d(np.asarray(b_deg, dtype=np.float64))
    with h5py.File(bayestar_path, "r") as f:
        pixel_info = f["pixel_info"][:]
        rows = _hierarchical_lookup_rows(pixel_info, l_deg, b_deg)
        reliable_pc = np.full(l_deg.size, np.nan, dtype=np.float64)
        ok = rows >= 0
        dm_reliable_max = np.asarray(pixel_info["DM_reliable_max"])[rows[ok]].astype(np.float64)
        reliable_pc[ok] = 10.0 ** (dm_reliable_max / 5.0 + 1.0)
    return rows, reliable_pc


def a_local_from_pieces(inside, far, a_edge, total_column_ak):
    """The one expression `_RegionProfile.a_of_d` and `population.
    anchor_tiles._magnitudes_from_pieces` both call (C26 item 3: "no
    second derivation"): `inside` is `A_CUM_K` interpolated at
    `min(d, D_EDGE)`, `far` is `max(0, A_CUM_K(d) - A_EDGE_K)` (already
    0 for `d <= D_EDGE`) -- both cached per sightline by the caller,
    since neither depends on the pixel's own `total_column_ak`;
    `a_edge` is `A_EDGE_K` for that sightline. Returns `inside *
    min(1, total/a_edge) + far`."""
    total = np.asarray(total_column_ak, dtype=float)
    scale = np.minimum(1.0, total / a_edge) if a_edge > 0 else np.ones_like(total)
    return inside * scale + far


def far_map_codes(l_deg, b_deg, bayestar_rows):
    """`FAR_MAP` (module docstring): Combined19's own build rule --
    Marshall's box first, else Bayestar where `bayestar_rows >= 0`
    (`bayestar_lookup`'s own row), else Drimmel."""
    l_deg = np.asarray(l_deg, dtype=np.float64)
    b_deg = np.asarray(b_deg, dtype=np.float64)
    in_marshall = ((np.abs(b_deg) < FAR_MAP_B_MAX_DEG)
                   & ((l_deg < FAR_MAP_L_MAX_DEG) | (l_deg > 360.0 - FAR_MAP_L_MAX_DEG)))
    return np.where(in_marshall, FAR_MAP_MARSHALL,
                     np.where(np.asarray(bayestar_rows) >= 0, FAR_MAP_BAYESTAR,
                              FAR_MAP_DRIMMEL)).astype(np.int8)


def _build_adopted_one_region(config, region, admitted_pix, gl, gb, dist_new, a_out_raw, a_inf,
                               dustmaps_dir):
    """Writes `profile_adopted_sightline` for `region` (C26 owner ruling
    2026-10-21, module docstring): `a_out_raw`/`dist_new`/`a_inf` are
    `_build_one_region`'s own raw splice-grid profile, distance grid and
    sightline total column -- the SAME one pass over the Edenhofer maps,
    never re-read.

    In-cube: `A_CUM_K = a_out_raw * min(1, a_inf / a_edge_raw)` -- today's
    `a_of_d`'s own scale rule (identity 2), applied once here instead of
    once per read. `A_EDGE_K` is that scaled profile's own edge value.

    Beyond the cube's edge: Combined19's own A_Ks, read at the cube's edge
    and at each of `ADOPTED_TAIL_N_NODES` log-spaced tail nodes out to
    `ADOPTED_TAIL_MAX_PC` (the SAME two endpoints for every sightline of
    the region, since the cube's edge distance is one number shared by
    every sightline in the survey) -- `A_CUM_K` there is `A_EDGE_K +
    max(0, C19(d) - C19(d_edge))`, Combined19's own INCREMENT, never its
    absolute value (owner ruling: "beyond the edge Combined19's own
    increment (never its absolute value)")."""
    d_edge = float(dist_new[-1])
    a_edge_raw = a_out_raw[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_build = np.where(a_edge_raw > 0.0, np.minimum(1.0, a_inf / a_edge_raw), 1.0)
    dist_incube = np.concatenate([[0.0], dist_new])
    a_cum_incube = np.concatenate(
        [np.zeros((admitted_pix.size, 1)), a_out_raw * scale_build[:, np.newaxis]], axis=1)
    a_edge_k = a_cum_incube[:, -1]

    tail = np.geomspace(d_edge, ADOPTED_TAIL_MAX_PC, ADOPTED_TAIL_N_NODES)[1:]
    dist_pc = np.concatenate([dist_incube, tail])

    dm_query = _dist_kpc_to_distmod(np.concatenate([[d_edge], tail]) / 1000.0)
    combined19_path = os.path.join(dustmaps_dir, "combined19", "combine19.h5")
    bayestar_path = os.path.join(dustmaps_dir, "bayestar", "bayestar2019.h5")
    curves_ak = combined19_curve_ak(combined19_path, gl, gb, dm_query)
    far_increment = np.maximum(0.0, curves_ak[:, 1:] - curves_ak[:, :1])
    a_cum_tail = a_edge_k[:, np.newaxis] + far_increment
    a_cum_k = np.concatenate([a_cum_incube, a_cum_tail], axis=1)

    bayestar_rows, far_reliable_all = bayestar_lookup(bayestar_path, gl, gb)
    far_map = far_map_codes(gl, gb, bayestar_rows)
    far_reliable_pc = np.where(far_map == FAR_MAP_BAYESTAR, far_reliable_all, np.nan)

    out_path = product_path(config, "sky/derived", "adopted", "profile", "sightline", region=region)
    real_root = os.path.realpath(config.data_root)
    if not os.path.realpath(out_path).startswith(real_root + os.sep):
        raise ValueError("profile.build: refusing to write outside the data root %s: %s"
                          % (real_root, out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        for name, data in (
            ("HPX_PIX_256", admitted_pix),
            ("GAL_L_DEG", gl),
            ("GAL_B_DEG", gb),
            ("DIST_PC", dist_pc),
            ("A_CUM_K", a_cum_k),
            ("A_EDGE_K", a_edge_k),
            ("D_EDGE_PC", d_edge),
            ("FAR_MAP", far_map),
            ("FAR_RELIABLE_PC", far_reliable_pc),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_ADOPTED_STEM, name)])

    n_sl = admitted_pix.size
    frac_marshall = float(np.mean(far_map == FAR_MAP_MARSHALL))
    frac_bayestar = float(np.mean(far_map == FAR_MAP_BAYESTAR))
    frac_drimmel = float(np.mean(far_map == FAR_MAP_DRIMMEL))
    print("profile.build_adopted: %s: %d sightlines, far map Marshall %.1f%% / Bayestar %.1f%% / "
          "Drimmel %.1f%%, FAR_RELIABLE_PC present for %d/%d Bayestar sightline(s)"
          % (region, n_sl, 100.0 * frac_marshall, 100.0 * frac_bayestar, 100.0 * frac_drimmel,
             int(np.count_nonzero(np.isfinite(far_reliable_pc))),
             int(np.count_nonzero(far_map == FAR_MAP_BAYESTAR))))
    return dict(region=region, frac_marshall=frac_marshall, frac_bayestar=frac_bayestar,
                frac_drimmel=frac_drimmel)


def build(config, regions=None):
    """Writes each region's sightline profile (`profile_edenhofer_
    sightline`, unchanged by C26), its canonical adopted profile
    (`profile_adopted_sightline`, C26 owner ruling 2026-10-21 -- see the
    module docstring), and one survey-wide depth file (spec 1.4). Regions
    run in a joblib pool; each worker streams the two Edenhofer maps once,
    for its own region's occupied pixels, and builds BOTH products from
    that one pass."""
    names = regions or [r.name for r in regions_module.REGIONS]
    canon_by_name = regions_module.REGIONS_BY_NAME
    input_dir = _input_dir(config)
    dustmaps_dir = config.inputs["dustmaps_data"]
    with progress_module.Stage("sky.derived.profile") as st:
        n_regions = len(names)
        n_done = [0]

        def _one(name):
            r = _build_one_region(config, name, canon_by_name[name], input_dir)
            _build_adopted_one_region(
                config, name, r["admitted_pix"], r["gl"], r["gb"], r["dist_new"],
                r["a_out_raw"], r["a_inf"], dustmaps_dir)
            n_done[0] += 1
            st.tick(n_done[0], n_regions, "regions")
            return r

        rows = Parallel(n_jobs=config.n_jobs)(delayed(_one)(name) for name in names)

        depth_path = product_path(config, "sky/derived", "edenhofer", "depth", "region")
        depth_rows = {
            key: np.array([r[col] for r in rows], dtype=np.float64)
            for key, col in (("D_R_PC", "d_r_pc"), ("SIGMA_D_PC", "sigma_d_pc"), ("D_PEAK_PC", "d_peak_pc"),
                              ("D_LO_PC", "d_lo_pc"), ("D_HI_PC", "d_hi_pc"), ("SIGMA_DEPTH_PC", "sigma_depth_pc"),
                              ("FWHM_PC", "fwhm_pc"))
        }
        depth_rows["DEPTH_OK"] = np.array([r["depth_ok"] for r in rows], dtype=bool)
        tables_module.update_rows(depth_path, names, depth_rows, granule="region", readings=_DEPTH_READINGS)
        st.done(depth_path, regions=n_regions, depth_ok=int(depth_rows["DEPTH_OK"].sum()),
                median_d_peak_pc=float(np.nanmedian(depth_rows["D_PEAK_PC"])))


# --- the reader -------------------------------------------------------


class _RegionProfile:
    """One region's evaluator (C26 owner ruling 2026-10-21, briefs/
    round4_2026-10-20/C26_profile_adopted.md), loaded from its canonical
    `profile_adopted_sightline` and `depth_edenhofer_region` files. The
    retired per-read residual/tail-shape/scale algebra (C25 and earlier)
    is gone: the adopted product already carries the sightline's own
    in-cube scale and far-field increment baked in; a read rescales the
    in-cube part by its own caller's total column and adds the far part
    unscaled -- see `a_of_d`'s own docstring for the one expression."""

    def __init__(self, profile_path, back_edge_pc, front_edge_pc):
        with h5py.File(profile_path, "r") as f:
            self.dist = f["DIST_PC"][:].astype(float)
            self.a_cum = f["A_CUM_K"][:].astype(float)
            self.a_edge = f["A_EDGE_K"][:].astype(float)
            self.hpx = f["HPX_PIX_256"][:].astype(np.int64)
            self.gl = f["GAL_L_DEG"][:].astype(float)
            self.gb = f["GAL_B_DEG"][:].astype(float)
            self.far_map = f["FAR_MAP"][:].astype(np.int8)
            self.far_reliable_pc = f["FAR_RELIABLE_PC"][:].astype(float)
            self.d_edge = float(f["D_EDGE_PC"][()])
        self.order = np.argsort(self.hpx)
        self.hpx_sorted = self.hpx[self.order]
        self.back_edge_pc = back_edge_pc
        self.front_edge_pc = front_edge_pc

    def sightline_row(self, hpx_pix):
        """Row index/indices into this region's profile for `hpx_pix`
        (nside-256 galactic NESTED); raises on an unoccupied pixel."""
        p = np.asarray(hpx_pix, dtype=np.int64)
        j = np.searchsorted(self.hpx_sorted, p)
        bad = (j >= self.hpx_sorted.size) | (self.hpx_sorted[np.minimum(j, self.hpx_sorted.size - 1)] != p)
        if np.any(bad):
            raise ValueError("hpx_pix %s is not an occupied sightline of this region"
                             % np.unique(p[bad])[:5].tolist())
        return self.order[j]

    def a_of_d(self, d_pc, hpx_pix, *, total_column_ak):
        """Cumulative extinction A_K from 0 to `d_pc` (C26 owner ruling
        2026-10-21): `A_CUM_K(min(d, D_EDGE)) * min(1, total/A_EDGE_K) +
        max(0, A_CUM_K(d) - A_EDGE_K)` -- one expression, interpolated on
        `DIST_PC`, valid for `d_pc` in or beyond the cube without a
        separate branch.

        The in-cube term rescales the stored, already-edge-normalised
        profile by the CALLER's own total column over the stored
        `A_EDGE_K` -- the same sightline value at the common call (a
        source or pixel reading its own sightline's column back), a
        DIFFERENT value where a caller's own column differs from the
        sightline's (a pixel's own mean column, C26 identity 2). The
        `min(1, ...)` keeps the in-cube part at or below what the
        caller's own total column implies -- the same "so u <= 1 inside
        the cube" invariant every design through C25 kept, now against
        `A_EDGE_K` rather than a second, raw edge value.

        The second term is Combined19's own INCREMENT past the cube's
        edge (already baked into `A_CUM_K` beyond `D_EDGE_PC` at build
        time, `_build_adopted_one_region`): `A_CUM_K(d) - A_EDGE_K` is
        exactly 0 for `d <= D_EDGE_PC` (the in-cube profile is
        non-decreasing and reaches `A_EDGE_K` exactly at the edge), so
        `max(0, ...)` contributes nothing there and the far increment
        unscaled beyond it -- `u = A(d)/total` CAN exceed 1 beyond the
        edge (owner ruling: Combined19's own measurement there is not
        capped at any pixel's or source's own column), on every
        sightline now, not only a Marshall-covered one (C25)."""
        row = int(np.asarray(self.sightline_row(hpx_pix)))
        d = np.asarray(d_pc, dtype=float)
        a_edge = float(self.a_edge[row])
        inside = np.interp(np.minimum(d, self.d_edge), self.dist, self.a_cum[row])
        far = np.maximum(0.0, np.interp(d, self.dist, self.a_cum[row]) - a_edge)
        return a_local_from_pieces(inside, far, a_edge, total_column_ak)

    def u(self, d_pc, hpx_pix, *, total_column_ak):
        """`u(d) = A(d) / total_column_ak`: divides by the CALLER's own
        total column, never by the stored profile's own edge value."""
        a = self.a_of_d(d_pc, hpx_pix, total_column_ak=total_column_ak)
        return a / np.asarray(total_column_ak, dtype=float)

    def row_of_lb(self, l_deg, b_deg):
        """`(row, hpx_pix)`: the admitted sightline nearest `(l_deg,
        b_deg)`, plain angular-separation argmax over this
        profile's own admitted sightlines -- stands in for a population
        member's own sky position where it has none (a TRILEGAL field
        star, `population.star_population`'s module docstring: "a field
        star carries a distance but no sky position of its own")."""
        l1, b1 = np.radians(float(l_deg)), np.radians(float(b_deg))
        l2, b2 = np.radians(self.gl), np.radians(self.gb)
        cos_sep = np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2)
        row = int(np.argmax(cos_sep))
        return row, int(self.hpx[row])

    def cloud_back_edge_pc(self):
        """The region's measured cloud back edge, pc (`D_HI_PC` of the
        structure `structure_depth` found nearest the canon distance)."""
        return self.back_edge_pc

    def cloud_front_edge_pc(self):
        """The region's measured cloud NEAR edge, pc (`D_LO_PC` of the
        same structure bracket `structure_depth` found nearest the canon
        distance, stored beside `D_HI_PC` in the depth product; owner
        ruling 2026-09-06, item 1: `star_population`'s front/behind
        placement threshold is `u` at this distance, not a fixed 0.5)."""
        return self.front_edge_pc


def read(config, region):
    """The region's profile evaluator (see `_RegionProfile`): C26 owner
    ruling 2026-10-21 reads `profile_adopted_sightline`, the canonical
    curve, and `depth_edenhofer_region`'s own cloud front/back edges --
    `sky.derived.marshall_extinction` is retired and read no more."""
    profile_path = product_path(config, "sky/derived", "adopted", "profile", "sightline", region=region)
    depth_path = product_path(config, "sky/derived", "edenhofer", "depth", "region")
    for path in (profile_path, depth_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "profile.read: product missing at %r -- run the "
                "sesnaimpute.sky.derived.profile RUNBOOK line for it" % path)
    with h5py.File(depth_path, "r") as f:
        names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in f["REGION"][:]]
        if region not in names:
            raise ValueError("profile.read: region %r has no row in %s" % (region, depth_path))
        back_edge_pc = float(f["D_HI_PC"][names.index(region)])
        front_edge_pc = float(f["D_LO_PC"][names.index(region)])
    return _RegionProfile(profile_path, back_edge_pc, front_edge_pc)


if __name__ == "__main__":
    from sesnaimpute import build as build_module
    build_module.run(build)
