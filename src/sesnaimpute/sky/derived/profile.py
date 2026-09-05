"""The sightline profile A(d) and region cloud depth (spec SPEC_PRIORS.md 1.4).

`build(config, regions=None)` writes, per region, the cumulative extinction
`A(d)` on the region's occupied nside-256 sightlines: Edenhofer et al. 2023's
mean differential-extinction map, converted E -> A_K through the ZGR23 R(Ks)
ratio, spliced to the deeper 2 kpc release using only its increment past the
517-boundary splice point, then completed to infinity by an exponential
dust-disc tail anchored so `A(inf)` equals the sightline's own total column
(from the Planck-anchored emission column product). It also writes one
survey-wide file of each region's cloud distance and line-of-sight depth,
read off the same profiles: the half-width between the 16th/84th percentile
of extinction-weighted distance inside the structure nearest the region's
canon distance.

`read(config, region)` returns the evaluator: `a_of_d(d_pc, hpx_pix=None,
*, total_column_ak)` is the cumulative extinction to distance `d_pc`, with
the tail's amplitude set by the caller's own total column (never a value
read out of the file); `u(d_pc, ...)` is `A(d)/A(inf)` per spec 1.4 -- it
divides by the sightline's total column, never by the profile's measured
back edge. `tail_shape`, `sightline_row`, `distance_knots_pc` and
`cloud_back_edge_pc` are the supporting pieces a class shape needs.
"""

import os

import h5py
import healpy as hp
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed

from sesnaimpute import regions as regions_module
from sesnaimpute.config import product_path

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
    return f"{config.data_root}/sky/download/edenhofer-profile-inputs"


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


def _far_field_for_region(hpx, gl, gb, a_edge, a_inf, nsrc, d_edge):
    """Per-sightline tail parameters and the region's fallback sightline
    (the source-weighted mean direction): `residual = max(0, A_inf -
    A_edge)`, the vertical branch `1-exp(-(d-d_edge)/L)` with `L =
    h_z/|sin b|`, or the in-plane disc quadrature where `L` would exceed
    `TAIL_LVERT_MAX_PC`. Returns `(tail, fallback)`."""
    res = np.maximum(0.0, a_inf - a_edge)
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
    a_edge_ref = float(np.sum(w * a_edge))
    a_inf_ref = float(np.sum(w * a_inf))
    res_ref = float(np.sum(w * res))
    lv_ref = H_Z_PC / max(abs(np.sin(np.radians(b_ref))), 1e-300)
    if lv_ref > TAIL_LVERT_MAX_PC:
        mode_ref = MODE_DISC
        s, f = disc_quadrature(l_ref, b_ref, d_edge)
        ef_ref, sc_ref = efold_pc(s, f, d_edge), np.nan
    else:
        mode_ref, sc_ref, ef_ref = MODE_VERTICAL, lv_ref, lv_ref

    tail = dict(residual=res, mode=mode, scale=scale, efold=efold)
    fallback = dict(a_edge_ref_k=a_edge_ref, a_col_sightline_ref_k=a_inf_ref,
                    residual_ref_k=res_ref, gal_l_ref_deg=l_ref, gal_b_ref_deg=b_ref,
                    tail_mode_ref=mode_ref, tail_scale_ref_pc=sc_ref, tail_efold_ref_pc=ef_ref)
    return tail, fallback


def region_weight(a_s_mean, n_src):
    """The A_s-weighted mean-profile weight; falls back to source counts
    where no pixel carries a positive column (a region with no per-source
    column input still has a mean profile, weighted by where its sources
    are)."""
    weight = a_s_mean.copy()
    if not np.any(weight > 0):
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

    base_u, base_c = inner["sig1_unc"][:, -1], inner["sig1_cor"][:, -1]
    s2 = splice["s2_terms"].T
    sig_u = np.concatenate([inner["sig1_unc"], np.sqrt(base_u[:, None] ** 2 + np.cumsum(s2 ** 2, axis=1))], axis=1)
    sig_c = np.concatenate([inner["sig1_cor"], base_c[:, None] + np.cumsum(s2, axis=1)], axis=1)

    mean_rho_inner = (inner["rho1"] * weight[np.newaxis, :]).sum(axis=1)
    rho_outer = (splice["rho2"] * weight[np.newaxis, :]).sum(axis=1)
    rho_mean_full = np.concatenate([mean_rho_inner, rho_outer])

    depth = structure_depth(rho_dist_new, rho_mean_full, d_r_pc)
    return {"a_out": a_out, "sig_u": sig_u, "sig_c": sig_c, "depth": depth}


# --- build ------------------------------------------------------------


def _admitted_sightlines(config, region):
    """The region's admitted sightlines (spec IMPLEMENTATION.md 1): every
    occupied nside-256 galactic-NESTED pixel the granule map associates
    with `region`, and its source count."""
    path = product_path(config, "granules", "sesna", "granule-map", "source")
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
    """The sightline's total column to infinity, A_K: the Planck sightline
    column product if it has landed, else (repoint when planck column
    lands) the old per-region emission-dust-column file."""
    new_path = product_path(config, "sky/derived", "planck", "column", "sightline")
    if os.path.exists(new_path):
        with h5py.File(new_path, "r") as f:
            hpx = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
            a_inf = np.asarray(f["A_K"][:], dtype=float)
        source_desc = new_path
    else:
        old_path = f"{config.data_root}/sky/derived/healpix256/planck-r120_emission-dust-column/emission_dust_column_ainf.hdf5"
        if not os.path.exists(old_path):
            raise FileNotFoundError(
                "profile.build: no total-column product at %s or %s -- run the "
                "planck sightline column RUNBOOK line first" % (new_path, old_path))
        with h5py.File(old_path, "r") as f:
            g = f["per_region"][region]
            hpx = np.asarray(g["HPX_PIX"][:], dtype=np.int64)
            a_inf = np.asarray(g["A_COL_SIGHTLINE_PLANCK_K"][:], dtype=float)
        source_desc = old_path
    order = np.argsort(hpx)
    hpx_sorted = hpx[order]
    pos = np.searchsorted(hpx_sorted, admitted_pix)
    capped = np.minimum(pos, hpx_sorted.size - 1) if hpx_sorted.size else pos
    ok = hpx_sorted.size and (hpx_sorted[capped] == admitted_pix)
    if not np.all(ok):
        missing = admitted_pix[~np.asarray(ok)][:5]
        raise ValueError("profile.build: %s has no total column for sightline(s) %s of region %r"
                         % (source_desc, missing.tolist(), region))
    return a_inf[order][capped]


def _build_one_region(config, region, canon, input_dir):
    admitted_pix, nsrc = _admitted_sightlines(config, region)
    a_inf = _join_total_column(config, region, admitted_pix)
    gl, gb = hp.pix2ang(NSIDE, admitted_pix, nest=True, lonlat=True)

    inner = read_inner_map(f"{input_dir}/{MAP_INNER_NAME}", admitted_pix, ZGR23_R_KS)
    splice = read_outer_increment(f"{input_dir}/{MAP_OUTER_NAME}", admitted_pix, ZGR23_R_KS)
    dist_new, rho_dist_new = splice_axes(inner["dist1"], inner["radii1"], splice["bnd2"], splice["cen2"], splice["k"])

    weight = region_weight(np.zeros(admitted_pix.size), nsrc)
    measured = measure_region(canon.d_r_pc, inner, splice, dist_new, rho_dist_new, weight)
    a_out, sig_u, sig_c, depth = measured["a_out"], measured["sig_u"], measured["sig_c"], measured["depth"]

    a_edge = a_out[:, -1]
    tail, fallback = _far_field_for_region(admitted_pix, gl, gb, a_edge, a_inf, nsrc, dist_new[-1])

    dist_pc = np.concatenate([[0.0], dist_new])
    a_cum_k = np.concatenate([np.zeros((admitted_pix.size, 1)), a_out], axis=1)
    sigma_unc_k = np.concatenate([np.zeros((admitted_pix.size, 1)), sig_u], axis=1)
    sigma_cor_k = np.concatenate([np.zeros((admitted_pix.size, 1)), sig_c], axis=1)
    rho_k_per_pc = np.diff(a_cum_k, axis=1) / np.diff(dist_pc)[np.newaxis, :]
    fallback_a_cum = weight @ a_cum_k

    out_path = product_path(config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        f.create_dataset("HPX_PIX_256", data=admitted_pix)
        f.create_dataset("DIST_PC", data=dist_pc)
        f.create_dataset("A_CUM_K", data=a_cum_k)
        f.create_dataset("RHO_K_PER_PC", data=rho_k_per_pc)
        f.create_dataset("SIGMA_UNC_K", data=sigma_unc_k)
        f.create_dataset("SIGMA_COR_K", data=sigma_cor_k)
        f.create_dataset("GAL_L_DEG", data=gl)
        f.create_dataset("GAL_B_DEG", data=gb)
        f.create_dataset("TAIL_RESIDUAL_K", data=tail["residual"])
        f.create_dataset("TAIL_MODE", data=tail["mode"])
        f.create_dataset("TAIL_SCALE_PC", data=tail["scale"])
        f.create_dataset("TAIL_EFOLD_PC", data=tail["efold"])
        f.create_dataset("A_INF_K", data=a_inf)
        fb = f.create_group("fallback")
        fb.create_dataset("DIST_PC", data=dist_pc)
        fb.create_dataset("A_CUM_K", data=fallback_a_cum)
        fb.create_dataset("A_EDGE_K", data=fallback["a_edge_ref_k"])
        fb.create_dataset("A_COL_SIGHTLINE_K", data=fallback["a_col_sightline_ref_k"])
        fb.create_dataset("RESIDUAL_K", data=fallback["residual_ref_k"])
        fb.create_dataset("GAL_L_DEG", data=fallback["gal_l_ref_deg"])
        fb.create_dataset("GAL_B_DEG", data=fallback["gal_b_ref_deg"])
        fb.create_dataset("TAIL_MODE", data=fallback["tail_mode_ref"])
        fb.create_dataset("TAIL_SCALE_PC", data=fallback["tail_scale_ref_pc"])
        fb.create_dataset("TAIL_EFOLD_PC", data=fallback["tail_efold_ref_pc"])

    return dict(region=region, d_r_pc=canon.d_r_pc, sigma_d_pc=canon.sigma_pc,
                d_peak_pc=depth["d_peak_pc"], d_lo_pc=depth["d_lo_pc"], d_hi_pc=depth["d_hi_pc"],
                sigma_depth_pc=depth["sigma_depth_pc"], fwhm_pc=depth["fwhm_pc"], depth_ok=bool(depth["ok"]))


def build(config, regions=None):
    """Writes each region's sightline profile and one survey-wide depth
    file (spec 1.4). Regions run in a joblib pool; each worker streams
    the two Edenhofer maps once, for its own region's occupied pixels."""
    names = regions or [r.name for r in regions_module.REGIONS]
    canon_by_name = regions_module.REGIONS_BY_NAME
    input_dir = _input_dir(config)
    rows = Parallel(n_jobs=-1)(
        delayed(_build_one_region)(config, name, canon_by_name[name], input_dir) for name in names)

    depth_path = product_path(config, "sky/derived", "edenhofer", "depth", "region")
    os.makedirs(os.path.dirname(depth_path), exist_ok=True)
    with h5py.File(depth_path, "w") as f:
        f.attrs["GRANULE"] = "region"
        f.create_dataset("REGION", data=np.array([r["region"] for r in rows], dtype="S64"))
        for key, col in (("D_R_PC", "d_r_pc"), ("SIGMA_D_PC", "sigma_d_pc"), ("D_PEAK_PC", "d_peak_pc"),
                         ("D_LO_PC", "d_lo_pc"), ("D_HI_PC", "d_hi_pc"), ("SIGMA_DEPTH_PC", "sigma_depth_pc"),
                         ("FWHM_PC", "fwhm_pc")):
            f.create_dataset(key, data=np.array([r[col] for r in rows], dtype=np.float64))
        f.create_dataset("DEPTH_OK", data=np.array([r["depth_ok"] for r in rows], dtype=bool))


# --- the reader -------------------------------------------------------


class _RegionProfile:
    """One region's evaluator, loaded from its `profile_edenhofer_sightline`
    and `depth_edenhofer_region` files."""

    def __init__(self, profile_path, back_edge_pc):
        with h5py.File(profile_path, "r") as f:
            self.dist = f["DIST_PC"][:].astype(float)
            self.a_cum = f["A_CUM_K"][:].astype(float)
            self.hpx = f["HPX_PIX_256"][:].astype(np.int64)
            self.gl = f["GAL_L_DEG"][:].astype(float)
            self.gb = f["GAL_B_DEG"][:].astype(float)
            self.tail_mode = f["TAIL_MODE"][:].astype(int)
            self.tail_scale = f["TAIL_SCALE_PC"][:].astype(float)
            fb = f["fallback"]
            self.fb_a_cum = fb["A_CUM_K"][:].astype(float)
            self.fb_mode = int(fb["TAIL_MODE"][()])
            self.fb_scale = float(fb["TAIL_SCALE_PC"][()])
            self.fb_l = float(fb["GAL_L_DEG"][()])
            self.fb_b = float(fb["GAL_B_DEG"][()])
        self.order = np.argsort(self.hpx)
        self.hpx_sorted = self.hpx[self.order]
        self.d_edge = float(self.dist[-1])
        self.back_edge_pc = back_edge_pc
        self._quad_cache = {}

    def _quad(self, key, l_deg, b_deg):
        if key not in self._quad_cache:
            self._quad_cache[key] = disc_quadrature(l_deg, b_deg, self.d_edge)
        return self._quad_cache[key]

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

    def tail_shape(self, d_pc, hpx_pix=None):
        """The dimensionless tail shape: 0 at the profile edge, 1 at
        infinity, geometry only (no amplitude)."""
        d = np.asarray(d_pc, dtype=float)
        if hpx_pix is None:
            if self.fb_mode == MODE_VERTICAL:
                return tail_shape_vertical(d, self.d_edge, self.fb_scale)
            s, f = self._quad("fallback", self.fb_l, self.fb_b)
            return np.clip(np.interp(d, s, f, left=0.0, right=1.0), 0.0, 1.0)
        row = int(np.asarray(self.sightline_row(hpx_pix)))
        if int(self.tail_mode[row]) == MODE_VERTICAL:
            return tail_shape_vertical(d, self.d_edge, float(self.tail_scale[row]))
        s, f = self._quad(row, self.gl[row], self.gb[row])
        return np.clip(np.interp(d, s, f, left=0.0, right=1.0), 0.0, 1.0)

    def a_of_d(self, d_pc, hpx_pix=None, *, total_column_ak):
        """Cumulative extinction A_K from 0 to `d_pc`: the measured
        profile inside the map's edge, `A_edge + residual * tail_shape(d)`
        beyond it, with `residual = max(0, total_column_ak - A_edge)`."""
        d = np.asarray(d_pc, dtype=float)
        if hpx_pix is None:
            inside = np.interp(d, self.dist, self.fb_a_cum)
            a_edge = float(self.fb_a_cum[-1])
        else:
            row = int(np.asarray(self.sightline_row(hpx_pix)))
            inside = np.interp(d, self.dist, self.a_cum[row])
            a_edge = float(self.a_cum[row, -1])
        residual = max(0.0, float(total_column_ak) - a_edge) if np.ndim(total_column_ak) == 0 \
            else np.maximum(0.0, np.asarray(total_column_ak, dtype=float) - a_edge)
        return inside + residual * self.tail_shape(d, hpx_pix)

    def u(self, d_pc, hpx_pix=None, *, total_column_ak):
        """`u(d) = A(d) / A(inf)` (spec 1.4): divides by the sightline's
        total column, never by the measured profile's back edge."""
        a = self.a_of_d(d_pc, hpx_pix, total_column_ak=total_column_ak)
        return a / np.asarray(total_column_ak, dtype=float)

    def distance_knots_pc(self):
        """The reconstruction's own spliced radial shell boundaries, pc."""
        return np.array(self.dist[1:], dtype=float)

    def cloud_back_edge_pc(self):
        """The region's measured cloud back edge, pc (`D_HI_PC` of the
        structure `structure_depth` found nearest the canon distance)."""
        return self.back_edge_pc


def read(config, region):
    """The region's profile evaluator (see `_RegionProfile`)."""
    profile_path = product_path(config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    depth_path = product_path(config, "sky/derived", "edenhofer", "depth", "region")
    with h5py.File(depth_path, "r") as f:
        names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in f["REGION"][:]]
        if region not in names:
            raise ValueError("profile.read: region %r has no row in %s" % (region, depth_path))
        back_edge_pc = float(f["D_HI_PC"][names.index(region)])
    return _RegionProfile(profile_path, back_edge_pc)


if __name__ == "__main__":
    from sesnaimpute import build as build_module
    build_module.run(build)
