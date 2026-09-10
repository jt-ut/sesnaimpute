"""Per-sightline width from the released posterior samples, not the
correlated-sum bound (repair-list row 8, spec 1.4): Edenhofer et al. 2024
state no correlation length for the map's kernel
(`bms_review/studies/edenhofer_kernel.md`) and advise using the 12
released posterior samples for any quantitative use; the released
per-voxel standard deviation, summed fully correlated along the line of
sight (`sky.derived.profile`'s `SIGMA_COR_K`), is only an upper bound in
the kernel's absence, and the uncorrelated sum (`SIGMA_UNC_K`) is the
matching lower bound. This stage recomputes each admitted sightline's
cumulative extinction `A_s(d)` from each of the 12 samples -- reusing
`sky.derived.profile`'s `extract_columns` (one sample's layer at a time,
the same `(n_shell, n_pix)` orientation as the mean map) and
`splice_axes` (the inner/outer join geometry) -- and stores the standard
deviation across samples, `SIGMA_SAMPLES_K`, per region. The samples
files (19.5/24.7 GB) are streamed a few shell-rows at a time, for the
region's own sightline pixels, never a whole layer or a whole sample;
the coordinator deletes them once every region's product exists here, so
nothing here may read them twice. The map's own pre-69-pc "integrated
inner" contribution is not released per sample (only its mean and std
are); since it is one additive constant per sightline shared by every
sample, it is taken from the mean map for all 12 samples alike and drops
out of the across-sample standard deviation by construction.
"""

import os

import h5py
import healpy as hp
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed

from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived import profile as profile_module

SAMPLES_INNER_NAME = "samples_healpix.fits"
SAMPLES_OUTER_NAME = "validation_with_less_data_but_2kpc_samples_healpix.fits"

N_PIX = hp.nside2npix(profile_module.NSIDE)


# --- opening a samples cube: axes located from the header/shape alone ----


def _locate_sample_axes(shape, n_pix=N_PIX):
    """`(pix_axis, sample_axis, shell_axis)` of a samples cube's shape,
    read from the HDU's shape (equivalently its header `NAXISn`
    keywords, since no data is touched to get it): the pixel axis is the
    one of length `n_pix`; of the remaining two, the shorter is the
    sample axis (12 samples versus hundreds of radial shells)."""
    if len(shape) != 3:
        raise ValueError("edenhofer_samples: expected a 3-D samples cube, got shape %r" % (shape,))
    pix_candidates = [i for i, n in enumerate(shape) if n == n_pix]
    if len(pix_candidates) != 1:
        raise ValueError("edenhofer_samples: cannot find the %d-pixel axis in shape %r" % (n_pix, shape))
    pix_axis = pix_candidates[0]
    other = [i for i in range(3) if i != pix_axis]
    if shape[other[0]] < shape[other[1]]:
        sample_axis, shell_axis = other
    else:
        shell_axis, sample_axis = other
    return pix_axis, sample_axis, shell_axis


class SamplesCube:
    """One samples FITS file's density cube, axis roles and radial axis.
    `layer(s)` is sample `s`'s `(n_shell, n_pix)` view -- basic indexing
    only (an integer on the sample axis, slices elsewhere), so it is a
    lazy view over the memmap, never a copy of the whole cube."""

    def __init__(self, hdul, data, pix_axis, sample_axis, shell_axis, bounds, centers):
        self.hdul = hdul
        self.data = data
        self.pix_axis = pix_axis
        self.sample_axis = sample_axis
        self.shell_axis = shell_axis
        self.bounds = bounds
        self.centers = centers

    @property
    def n_shell(self):
        return self.data.shape[self.shell_axis]

    @property
    def n_samples(self):
        return self.data.shape[self.sample_axis]

    def layer(self, s):
        idx = [slice(None)] * 3
        idx[self.sample_axis] = int(s)
        view = self.data[tuple(idx)]
        remaining = [a - (1 if a > self.sample_axis else 0) for a in (self.shell_axis, self.pix_axis)]
        if remaining != [0, 1]:
            view = np.moveaxis(view, remaining, [0, 1])
        return view

    def close(self):
        self.hdul.close()


class _LayerProxy:
    """Adapts one sample's `(n_shell, n_pix)` view to the interface
    `sky.derived.profile.extract_columns` reads off a mean map."""

    def __init__(self, density):
        self.density = density

    @property
    def n_shell(self):
        return self.density.shape[0]


def open_samples_cube(path):
    """Opens `path`, locates its samples cube's axes from the HDU's own
    shape (never its data), and reads its radial-boundary table with
    `sky.derived.profile.read_radial_axes` -- the same `BinTableHDU`s the
    mean/std release carries."""
    hdul = fits.open(path, mode="readonly", memmap=True)
    cube_hdu = None
    for hdu in hdul:
        if isinstance(hdu, fits.PrimaryHDU) and hdu.data is None:
            continue
        if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) and hdu.data is not None and len(hdu.shape) == 3:
            cube_hdu = hdu
            break
    if cube_hdu is None:
        hdul.close()
        raise ValueError("edenhofer_samples: no 3-D samples cube in %s" % path)
    pix_axis, sample_axis, shell_axis = _locate_sample_axes(cube_hdu.shape)
    bounds, centers = profile_module.read_radial_axes(path)
    return SamplesCube(hdul, cube_hdu.data, pix_axis, sample_axis, shell_axis, bounds, centers)


# --- per-sample cumulative extinction, inner and outer maps ---------------


def sample_a_cum_inner(cube, pixels, inner_baseline_ak, row_chunk=64):
    """`(n_samples, n_shell+1, n_pix_sel)` per-sample cumulative A_K on
    the inner map's own grid, sharing the mean map's own pre-69-pc
    baseline (module docstring) for every sample."""
    widths = np.diff(cube.bounds)[:, None]
    out = np.empty((cube.n_samples, cube.n_shell + 1, pixels.size), dtype=np.float64)
    for s in range(cube.n_samples):
        rho = profile_module.extract_columns(_LayerProxy(cube.layer(s)), pixels, row_chunk=row_chunk)
        rho = rho.astype(np.float64) * profile_module.ZGR23_R_KS
        out[s, 0, :] = inner_baseline_ak
        np.cumsum(rho * widths, axis=0, out=out[s, 1:, :])
        out[s, 1:, :] += inner_baseline_ak[None, :]
    return out


def sample_a_increment_outer(cube, pixels, k, row_chunk=64):
    """`(n_samples, n_shell-k+1, n_pix_sel)` per-sample cumulative
    increment past the splice point, the outer map's own contribution
    only (`sky.derived.profile.read_outer_increment`'s rule: the two
    reconstructions' offset cancels out of a difference)."""
    widths2 = np.diff(np.concatenate([[profile_module.SPLICE_PC], cube.bounds[k:]]))[:, None]
    out = np.empty((cube.n_samples, cube.bounds.size - k, pixels.size), dtype=np.float64)
    for s in range(cube.n_samples):
        rho_full = profile_module.extract_columns(_LayerProxy(cube.layer(s)), pixels, row_chunk=row_chunk)
        rho2 = rho_full.astype(np.float64)[k - 1:, :] * profile_module.ZGR23_R_KS
        out[s] = np.cumsum(rho2 * widths2, axis=0)
    return out


def sample_a_cum(inner_cube, outer_cube, pixels, inner_baseline_ak):
    """`(dist_pc, a_cum_samples)`: the spliced per-sample cumulative
    A_K, `(n_samples, n_dist, n_pix_sel)`, on the same distance grid
    `sky.derived.profile._build_one_region` writes (a leading `0.0` at
    `A=0`, then the inner map's boundaries, then the outer map's past
    the splice point)."""
    k = int(np.searchsorted(outer_cube.bounds, profile_module.SPLICE_PC, side="right"))
    if not (outer_cube.bounds[k - 1] <= profile_module.SPLICE_PC < outer_cube.bounds[k]):
        raise ValueError("edenhofer_samples: the splice point does not lie in the outer samples' shell %d" % (k - 1))
    dist_new, _ = profile_module.splice_axes(inner_cube.bounds, inner_cube.centers, outer_cube.bounds,
                                              outer_cube.centers, k)
    a_inner = sample_a_cum_inner(inner_cube, pixels, inner_baseline_ak)
    increment = sample_a_increment_outer(outer_cube, pixels, k)
    a_outer = a_inner[:, -1:, :] + increment
    a_full = np.concatenate([a_inner, a_outer], axis=1)
    dist_pc = np.concatenate([[0.0], dist_new])
    zeros = np.zeros((a_full.shape[0], 1, a_full.shape[2]))
    a_full = np.concatenate([zeros, a_full], axis=1)
    return dist_pc, a_full


# --- the outer-samples-absent fallback (spec: repair-list row 8) ---------


def fallback_sigma_spliced(sigma_cor_outer, ratio_inner_median):
    """Where the outer samples file is absent: the outer map's own
    `SIGMA_COR_K` scaled by the region's median
    `SIGMA_SAMPLES_K / SIGMA_COR_K` over its inner-map cells behind the
    cloud -- the inner map's own ratio stands in for the outer map's,
    since neither the samples nor a stated correlation length are
    available there."""
    return sigma_cor_outer * ratio_inner_median


# --- build -----------------------------------------------------------


def _region_ratio_medians(sigma_samples, sigma_cor, sigma_unc, dist_pc, d_hi_pc, splice_mask=None):
    """Per-region median ratios behind the cloud (`dist_pc > d_hi_pc`),
    `SIGMA_SAMPLES_K / SIGMA_COR_K` and `/ SIGMA_UNC_K`, restricted to
    cells where the two bounds actually bracket the samples estimate
    (`SIGMA_UNC_K <= SIGMA_SAMPLES_K <= SIGMA_COR_K`)."""
    behind = dist_pc > d_hi_pc
    if splice_mask is not None:
        behind = behind & splice_mask
    if not np.any(behind):
        return np.nan, np.nan
    s = sigma_samples[:, behind]
    c = sigma_cor[:, behind]
    u = sigma_unc[:, behind]
    bracket = (s >= u) & (s <= c) & (c > 0) & (u > 0)
    if not np.any(bracket):
        return np.nan, np.nan
    return float(np.median(s[bracket] / c[bracket])), float(np.median(s[bracket] / u[bracket]))


def _build_one_region(config, region, input_dir):
    pixels, _ = profile_module._admitted_sightlines(config, region)

    inner_mean_path = f"{input_dir}/{profile_module.MAP_INNER_NAME}"
    inner_baseline_e = None
    with fits.open(inner_mean_path, memmap=True) as hdul:
        for hdu in hdul:
            if hdu.name.lower().startswith(("mean of integrated inner", "integrated inner")):
                inner_baseline_e = np.asarray(hdu.data, dtype=np.float64)[pixels]
                break
    if inner_baseline_e is None:
        raise ValueError("edenhofer_samples.build: no integrated-inner MEAN HDU in %s" % inner_mean_path)
    inner_baseline_ak = inner_baseline_e * profile_module.ZGR23_R_KS

    inner_path = f"{input_dir}/{SAMPLES_INNER_NAME}"
    outer_path = f"{input_dir}/{SAMPLES_OUTER_NAME}"
    if not os.path.exists(inner_path):
        raise FileNotFoundError(
            "edenhofer_samples.build: samples input missing at %s -- run the "
            "sesnaimpute.sky.download.edenhofer2023.build RUNBOOK line for it" % inner_path)
    inner_cube = open_samples_cube(inner_path)
    ratio_splice = None
    try:
        if os.path.exists(outer_path):
            outer_cube = open_samples_cube(outer_path)
            try:
                dist_pc, a_full = sample_a_cum(inner_cube, outer_cube, pixels, inner_baseline_ak)
            finally:
                outer_cube.close()
        else:
            print("edenhofer_samples.build: %s: outer samples absent, splicing "
                  "with the inner ratio (RATIO_SPLICED)" % region)
            a_inner = sample_a_cum_inner(inner_cube, pixels, inner_baseline_ak)
            dist_pc = np.concatenate([[0.0], inner_cube.bounds])
            zeros = np.zeros((a_inner.shape[0], 1, a_inner.shape[2]))
            a_full = np.concatenate([zeros, a_inner], axis=1)
    finally:
        inner_cube.close()

    mean_a = a_full.mean(axis=0)
    sigma_samples = a_full.std(axis=0, ddof=0).T  # (n_pix_sel, n_dist)

    profile_path = product_path(config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    depth_path = product_path(config, "sky/derived", "edenhofer", "depth", "region")
    with h5py.File(profile_path, "r") as f:
        prof_dist = f["DIST_PC"][:]
        prof_hpx = f["HPX_PIX_256"][:]
        sigma_cor = f["SIGMA_COR_K"][:]
        sigma_unc = f["SIGMA_UNC_K"][:]
        a_cum_mean_file = f["A_CUM_K"][:]
    order = np.argsort(prof_hpx)
    pos = order[np.searchsorted(prof_hpx[order], pixels)]
    sigma_cor = sigma_cor[pos]
    sigma_unc = sigma_unc[pos]
    a_cum_mean_file = a_cum_mean_file[pos]

    with h5py.File(depth_path, "r") as f:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in f["REGION"][:]]
        d_hi_pc = float(f["D_HI_PC"][names.index(region)])

    if len(dist_pc) != len(prof_dist) or not np.allclose(dist_pc, prof_dist, rtol=0, atol=1e-6):
        # the outer samples were absent: sigma_samples only spans the
        # inner grid so far -- extend it with the fallback rule (module
        # docstring / spec repair-list row 8), on the profile's own
        # (spliced) distance axis.
        n_inner = dist_pc.size
        behind_inner = prof_dist[:n_inner] > d_hi_pc
        ratio = sigma_samples[:, behind_inner] / np.maximum(sigma_cor[:, :n_inner][:, behind_inner], 1e-300)
        ratio_splice = float(np.nanmedian(ratio)) if ratio.size else 1.0
        pad = sigma_cor[:, n_inner:] * ratio_splice
        sigma_samples = np.concatenate([sigma_samples, pad], axis=1)
        dist_pc = prof_dist
        print("edenhofer_samples.build: %s: RATIO_SPLICED=%.3g applied to %d cells past the inner edge"
              % (region, ratio_splice, pad.shape[1]))

    med_cor, med_unc = _region_ratio_medians(sigma_samples, sigma_cor, sigma_unc, dist_pc, d_hi_pc)
    max_rel_diff = float(np.nanmax(np.abs(mean_a.T - a_cum_mean_file[:, :mean_a.shape[0]])
                                    / np.maximum(np.abs(a_cum_mean_file[:, :mean_a.shape[0]]), 1e-12)))
    print("edenhofer_samples.build: %s: median SIGMA_SAMPLES_K/SIGMA_COR_K=%.3g, /SIGMA_UNC_K=%.3g behind the cloud, "
          "max rel. diff of the sample mean against A_CUM_K=%.3g" % (region, med_cor, med_unc, max_rel_diff))

    out_path = product_path(config, "sky/derived", "edenhofer", "profile-sigma-samples", "sightline", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        if ratio_splice is not None:
            f.attrs["RATIO_SPLICED"] = ratio_splice
        f.create_dataset("HPX_PIX_256", data=pixels)
        f.create_dataset("DIST_PC", data=dist_pc)
        f.create_dataset("SIGMA_SAMPLES_K", data=sigma_samples)

    return dict(region=region, max_rel_diff=max_rel_diff, med_cor=med_cor, med_unc=med_unc)


def build(config, regions=None):
    """Writes each region's `SIGMA_SAMPLES_K` product (module docstring).
    Regions run in a joblib pool; each worker streams the two samples
    files once, for its own region's occupied pixels."""
    names = regions or [r.name for r in regions_module.REGIONS]
    input_dir = profile_module._input_dir(config)
    with progress_module.Stage("sky.derived.edenhofer_samples") as st:
        n_regions = len(names)
        n_done = [0]

        def _one(name):
            r = _build_one_region(config, name, input_dir)
            n_done[0] += 1
            st.tick(n_done[0], n_regions, "regions")
            return r

        rows = Parallel(n_jobs=config.n_jobs)(delayed(_one)(name) for name in names)
        st.done(None, regions=n_regions,
                median_max_rel_diff=float(np.nanmedian([r["max_rel_diff"] for r in rows])))


if __name__ == "__main__":
    from sesnaimpute import build as build_module
    build_module.run(build)
