"""Per-sightline width from the released posterior samples, not the
correlated-sum bound: Edenhofer et al. 2024 state no correlation length
for the map's kernel (`bms_review/studies/edenhofer_kernel.md`) and
advise using the 12 released posterior samples for any quantitative use;
the released per-voxel standard deviation, summed fully correlated along
the line of sight (`sky.derived.profile`'s `SIGMA_COR_K`), is only an
upper bound in the kernel's absence, and the uncorrelated sum
(`SIGMA_UNC_K`) is the matching lower bound.

Each samples file is one fixed-layout cube, HDU `SAMPLES`, numpy shape
`(12, n_shell, 786432)` -- read in one sequential pass, shell-row
chunked, gathering every admitted region's sightlines at once (their
union, far fewer bytes than one region's worth of re-reads per region);
the outer file's pass starts at the shell holding the inner map's own
edge, never earlier. From the union columns, each region's cumulative
extinction `A_s(d)` is formed per sample -- the same conversion and
splice `sky.derived.profile` uses for the mean -- and `SIGMA_SAMPLES_K`
is their standard deviation, every term in it sampled: the map's
pre-69-pc "integrated inner" contribution is itself released per sample
(the `INTEGRATED INNER 68.8 PC` HDU, one column per sample), gathered
for the union pixels in the same pass and used as each sample's own
starting value, never a value shared across samples. Where the outer
samples file is absent from disk, the spliced part
is the outer map's `SIGMA_COR_K` scaled by the region's own median
`SIGMA_SAMPLES_K / SIGMA_COR_K` behind the cloud on the inner map
(`RATIO_SPLICED`, one line printed); any other disagreement between a
samples cube and its mean map (shell count, missing HDU, distance axis)
raises. The samples files are deleted by the coordinator once every
region's product exists here.
"""

import os

import h5py
import healpy as hp
import numpy as np
from astropy.io import fits

from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived import profile as profile_module

SAMPLES_INNER_NAME = "samples_healpix.fits"
SAMPLES_OUTER_NAME = "validation_with_less_data_but_2kpc_samples_healpix.fits"
SAMPLES_HDU_NAME = "SAMPLES"
#: The pre-69-pc integrated layer's own per-sample HDU (inner map only --
#: the outer map's analogous baseline is never read, since its increment
#: is anchored to the inner map's own edge, not to its own baseline).
INNER_BASELINE_HDU_NAME = "INTEGRATED INNER 68.8 PC"
N_SAMPLES = 12
N_PIX = hp.nside2npix(profile_module.NSIDE)

#: Shell rows read per chunk of a samples cube: `N_SAMPLES * row_chunk *
#: N_PIX * 4 bytes` stays under ~0.3 GB, comfortably inside the run's
#: memory ceiling however many sightlines the union covers.
SHELL_ROW_CHUNK = 8


# --- the union pass: one sequential read per samples file -----------------


def _admitted_by_region(config, names):
    """`(union_pixels, pix_by_region)`: the sorted union of the named
    regions' admitted sightlines, and each region's own pixel array."""
    pix_by_region = {name: profile_module._admitted_sightlines(config, name)[0] for name in names}
    union_pixels = np.unique(np.concatenate(list(pix_by_region.values())))
    return union_pixels, pix_by_region


def _local_index(union_pixels, region_pixels):
    idx = np.searchsorted(union_pixels, region_pixels)
    if not np.array_equal(union_pixels[idx], region_pixels):
        raise ValueError("edenhofer_samples: a region's admitted pixel is missing from the union read")
    return idx


def _read_samples_union(path, n_shell_expected, union_pixels, shell_lo=0, shell_hi=None,
                         row_chunk=SHELL_ROW_CHUNK):
    """One sequential pass over `path`'s `SAMPLES` cube, shell-row
    chunked, gathering the union pixels for every sample at once:
    `(N_SAMPLES, shell_hi-shell_lo, n_union_pix)` float32. The cube's
    fixed layout (numpy shape `(12, n_shell, 786432)`) is asserted, never
    discovered; a shell count that disagrees with the mean map's own, or
    a missing `SAMPLES` HDU, fails with one sentence."""
    shell_hi = n_shell_expected if shell_hi is None else shell_hi
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        if SAMPLES_HDU_NAME not in hdul:
            raise ValueError("edenhofer_samples: no %r HDU in %s" % (SAMPLES_HDU_NAME, path))
        hdu = hdul[SAMPLES_HDU_NAME]
        if hdu.shape != (N_SAMPLES, n_shell_expected, N_PIX):
            raise ValueError(
                "edenhofer_samples: %s's %r HDU has shape %r, expected (%d, %d, %d) from the mean map's own "
                "shell count" % (path, SAMPLES_HDU_NAME, hdu.shape, N_SAMPLES, n_shell_expected, N_PIX))
        data = hdu.data
        out = np.empty((N_SAMPLES, shell_hi - shell_lo, union_pixels.size), dtype=np.float32)
        for start in range(shell_lo, shell_hi, row_chunk):
            stop = min(start + row_chunk, shell_hi)
            block = np.asarray(data[:, start:stop, :])
            out[:, start - shell_lo:stop - shell_lo, :] = block[:, :, union_pixels]
    return out


def _read_baseline_union(path, hdu_name, union_pixels):
    """The samples file's own per-sample pre-69-pc integrated layer,
    gathered for the union pixels: `(N_SAMPLES, n_union_pix)`, native
    (E) units. Fixed layout (numpy shape `(12, 786432)`), asserted, never
    discovered; a missing HDU or a shape disagreement fails with one
    sentence."""
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        if hdu_name not in hdul:
            raise ValueError("edenhofer_samples: no %r HDU in %s" % (hdu_name, path))
        hdu = hdul[hdu_name]
        if hdu.shape != (N_SAMPLES, N_PIX):
            raise ValueError("edenhofer_samples: %s's %r HDU has shape %r, expected (%d, %d)"
                              % (path, hdu_name, hdu.shape, N_SAMPLES, N_PIX))
        block = np.asarray(hdu.data[:, union_pixels], dtype=np.float64)
    return block


# --- per-region cumulative extinction and width, from the union arrays ----


def cumulative_inner(union_inner, bounds1, baseline_per_sample_e):
    """`(N_SAMPLES, n_shell+1, n_pix)` per-sample cumulative A_K on the
    inner map's own grid, from raw (native-unit) density columns and
    each sample's own pre-69-pc integrated value, `baseline_per_sample_e`
    `(N_SAMPLES, n_pix)` (module docstring: every term here is sampled)."""
    widths = np.diff(bounds1)[None, :, None]
    rho = union_inner.astype(np.float64) * profile_module.ZGR23_R_KS
    baseline_ak = baseline_per_sample_e * profile_module.ZGR23_R_KS
    out = np.empty((N_SAMPLES, bounds1.size, union_inner.shape[-1]), dtype=np.float64)
    out[:, 0, :] = baseline_ak
    np.cumsum(rho * widths, axis=1, out=out[:, 1:, :])
    out[:, 1:, :] += baseline_ak[:, None, :]
    return out


def cumulative_outer_increment(union_outer, bnd2, k):
    """`(N_SAMPLES, bnd2.size-k, n_pix)` per-sample cumulative increment
    past the splice point, from raw density columns for shells
    `[k-1, bnd2.size-1)` only (the outer samples file is never read
    below the splice)."""
    widths2 = np.diff(np.concatenate([[profile_module.SPLICE_PC], bnd2[k:]]))[None, :, None]
    rho2 = union_outer.astype(np.float64) * profile_module.ZGR23_R_KS
    return np.cumsum(rho2 * widths2, axis=1)


def sigma_samples_of(a_full):
    """`SIGMA_SAMPLES_K`, `(n_pix, n_dist)`: the across-sample std of
    `A(d)` -- every term, including the pre-69-pc baseline, comes from
    the samples themselves (module docstring), so no correction term is
    added here; zero at the leading `d=0` point, where every sample is 0
    by construction."""
    return a_full.std(axis=0, ddof=0).T


def region_a_full(union_inner_region, bounds1, centers1, baseline_per_sample_e_region,
                   union_outer_region=None, bnd2=None, cen2=None, k=None):
    """`(dist_pc, a_full)`: one region's per-sample cumulative A_K,
    `sky.derived.profile`'s own distance grid (a leading `0.0`, then the
    inner map's boundaries, then -- if the outer samples are present --
    the outer map's past the splice point, via `profile.splice_axes`)."""
    a_inner = cumulative_inner(union_inner_region, bounds1, baseline_per_sample_e_region)
    if union_outer_region is not None:
        increment = cumulative_outer_increment(union_outer_region, bnd2, k)
        a_outer = a_inner[:, -1:, :] + increment
        a_full = np.concatenate([a_inner, a_outer], axis=1)
        dist_new, _ = profile_module.splice_axes(bounds1, centers1, bnd2, cen2, k)
    else:
        a_full = a_inner
        dist_new = bounds1
    dist_pc = np.concatenate([[0.0], dist_new])
    zeros = np.zeros((a_full.shape[0], 1, a_full.shape[2]))
    a_full = np.concatenate([zeros, a_full], axis=1)
    return dist_pc, a_full


def region_ratio_medians(sigma_samples, sigma_cor, sigma_unc, dist_pc, d_hi_pc):
    """Per-region median ratios behind the cloud (`dist_pc > d_hi_pc`),
    `SIGMA_SAMPLES_K / SIGMA_COR_K` and `/ SIGMA_UNC_K`, restricted to
    cells where the two bounds actually bracket the samples estimate."""
    behind = dist_pc > d_hi_pc
    if not np.any(behind):
        return np.nan, np.nan
    s, c, u = sigma_samples[:, behind], sigma_cor[:, behind], sigma_unc[:, behind]
    bracket = (s >= u) & (s <= c) & (c > 0) & (u > 0)
    if not np.any(bracket):
        return np.nan, np.nan
    return float(np.median(s[bracket] / c[bracket])), float(np.median(s[bracket] / u[bracket]))


# --- build -----------------------------------------------------------


def _build_one_region(config, region, region_pixels, union_pixels, union_inner, bounds1, centers1,
                       baseline_per_sample_e, union_outer, bnd2, cen2, k, outer_present,
                       inner_mean_path, outer_mean_path):
    idx = _local_index(union_pixels, region_pixels)
    union_outer_region = union_outer[:, :, idx] if outer_present else None
    dist_pc, a_full = region_a_full(union_inner[:, :, idx], bounds1, centers1, baseline_per_sample_e[:, idx],
                                     union_outer_region, bnd2, cen2, k)
    sigma_samples = sigma_samples_of(a_full)
    mean_a = a_full.mean(axis=0)

    profile_path = product_path(config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    depth_path = product_path(config, "sky/derived", "edenhofer", "depth", "region")
    with h5py.File(profile_path, "r") as f:
        prof_hpx = f["HPX_PIX_256"][:]
        prof_dist = f["DIST_PC"][:]
        sigma_cor = f["SIGMA_COR_K"][:]
        sigma_unc = f["SIGMA_UNC_K"][:]
    order = np.argsort(prof_hpx)
    pos = order[np.searchsorted(prof_hpx[order], region_pixels)]
    sigma_cor, sigma_unc = sigma_cor[pos], sigma_unc[pos]
    with h5py.File(depth_path, "r") as f:
        names_all = [v.decode() if isinstance(v, bytes) else str(v) for v in f["REGION"][:]]
        d_hi_pc = float(f["D_HI_PC"][names_all.index(region)])

    ratio_splice = None
    if outer_present:
        if dist_pc.size != prof_dist.size or not np.allclose(dist_pc, prof_dist, atol=1e-6):
            raise ValueError(
                "edenhofer_samples.build: %s: the samples' spliced distance axis disagrees with the built "
                "profile's -- radial-axis mismatch between the samples cubes and the mean maps" % region)
    else:
        n_inner = dist_pc.size
        if n_inner > prof_dist.size or not np.allclose(dist_pc, prof_dist[:n_inner], atol=1e-6):
            raise ValueError(
                "edenhofer_samples.build: %s: the inner samples' distance axis disagrees with the built "
                "profile's -- radial-axis mismatch between the samples cube and the mean map" % region)
        behind_inner = prof_dist[:n_inner] > d_hi_pc
        ratio = sigma_samples[:, behind_inner] / np.maximum(sigma_cor[:, :n_inner][:, behind_inner], 1e-300)
        ratio_splice = float(np.nanmedian(ratio)) if ratio.size else 1.0
        pad = sigma_cor[:, n_inner:] * ratio_splice
        sigma_samples = np.concatenate([sigma_samples, pad], axis=1)
        dist_pc = prof_dist
        print("edenhofer_samples.build: %s: outer samples absent -- RATIO_SPLICED=%.3g applied to %d cells "
              "past the inner edge" % (region, ratio_splice, pad.shape[1]))

    # identity: mean-over-samples A(d) against the UNSCALED in-map+splice
    # profile (not A_CUM_K, which carries `profile.py`'s per-sightline
    # far-field rescale -- an unrelated correction this stage does not
    # reproduce).
    inner_ref = profile_module.read_inner_map(inner_mean_path, region_pixels, profile_module.ZGR23_R_KS)
    a_ref = inner_ref["a_cum1_T"].astype(np.float64)
    if outer_present:
        splice_ref = profile_module.read_outer_increment(outer_mean_path, region_pixels, profile_module.ZGR23_R_KS)
        a_ref = np.concatenate([a_ref, a_ref[:, -1:] + splice_ref["dA2"]], axis=1)
    a_ref = np.concatenate([np.zeros((region_pixels.size, 1)), a_ref], axis=1)
    max_rel_diff = float(np.nanmax(np.abs(mean_a.T - a_ref) / np.maximum(np.abs(a_ref), 1e-12)))

    med_cor, med_unc = region_ratio_medians(sigma_samples, sigma_cor, sigma_unc, dist_pc, d_hi_pc)
    print("edenhofer_samples.build: %s: max rel. diff of the sample mean against the unscaled profile=%.3g, "
          "median SIGMA_SAMPLES_K/SIGMA_COR_K=%.3g, /SIGMA_UNC_K=%.3g behind the cloud"
          % (region, max_rel_diff, med_cor, med_unc))

    out_path = product_path(config, "sky/derived", "edenhofer", "profile-sigma-samples", "sightline", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        if ratio_splice is not None:
            f.attrs["RATIO_SPLICED"] = ratio_splice
        f.create_dataset("HPX_PIX_256", data=region_pixels)
        f.create_dataset("DIST_PC", data=dist_pc)
        f.create_dataset("SIGMA_SAMPLES_K", data=sigma_samples)

    return dict(region=region, max_rel_diff=max_rel_diff, med_cor=med_cor, med_unc=med_unc)


def build(config, regions=None):
    """One sequential pass per samples file over the union of the named
    regions' admitted sightlines (module docstring), then each region's
    own `SIGMA_SAMPLES_K` product from the union columns."""
    names = regions or [r.name for r in regions_module.REGIONS]
    input_dir = profile_module._input_dir(config)
    inner_samples_path = f"{input_dir}/{SAMPLES_INNER_NAME}"
    outer_samples_path = f"{input_dir}/{SAMPLES_OUTER_NAME}"
    if not os.path.exists(inner_samples_path):
        raise FileNotFoundError(
            "edenhofer_samples.build: samples input missing at %s -- run the "
            "sesnaimpute.sky.download.edenhofer2023.build RUNBOOK line for it" % inner_samples_path)
    outer_present = os.path.exists(outer_samples_path)

    inner_mean_path = f"{input_dir}/{profile_module.MAP_INNER_NAME}"
    outer_mean_path = f"{input_dir}/{profile_module.MAP_OUTER_NAME}"
    bounds1, centers1 = profile_module.read_radial_axes(inner_mean_path)
    bnd2, cen2 = profile_module.read_radial_axes(outer_mean_path)
    k = int(np.searchsorted(bnd2, profile_module.SPLICE_PC, side="right"))
    if not (bnd2[k - 1] <= profile_module.SPLICE_PC < bnd2[k]):
        raise ValueError("edenhofer_samples.build: the splice point does not lie in the outer map's shell %d"
                          % (k - 1))

    with progress_module.Stage("sky.derived.edenhofer_samples") as st:
        union_pixels, pix_by_region = _admitted_by_region(config, names)
        baseline_per_sample_e = _read_baseline_union(inner_samples_path, INNER_BASELINE_HDU_NAME, union_pixels)

        union_inner = _read_samples_union(inner_samples_path, bounds1.size - 1, union_pixels)
        st.tick(1, 3, "inner samples read (%d union pixels)" % union_pixels.size)
        if outer_present:
            union_outer = _read_samples_union(outer_samples_path, bnd2.size - 1, union_pixels,
                                               shell_lo=k - 1, shell_hi=bnd2.size - 1)
        else:
            union_outer = None
            print("edenhofer_samples.build: %s absent from disk -- RATIO_SPLICED applies past the inner edge"
                  % outer_samples_path)
        st.tick(2, 3, "outer samples read" if outer_present else "outer samples absent")

        rows = [_build_one_region(config, name, pix_by_region[name], union_pixels, union_inner, bounds1, centers1,
                                   baseline_per_sample_e, union_outer, bnd2, cen2, k, outer_present,
                                   inner_mean_path, outer_mean_path)
                for name in names]
        st.tick(3, 3, "regions written")
        st.done(None, regions=len(names),
                median_max_rel_diff=float(np.nanmedian([r["max_rel_diff"] for r in rows])))


if __name__ == "__main__":
    from sesnaimpute import build as build_module
    build_module.run(build)
