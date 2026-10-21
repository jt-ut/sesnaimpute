"""VISTA Hemisphere Survey (VHS) point-source counts per nside-512
HEALPix pixel (C19 Part B: `sky.derived.ukidss_counts`'s own sibling
and product shape, read where UKIDSS does not cover a pixel --
`population.anchor_tiles._combine_ks_axis` takes the deep Ks axis from
UKIDSS where UKIDSS covers a pixel, else from this product where it
does).

Derive never fetches: `sky.download.vhs.build` already pulled the open
class set with no error-bit cut (that module's own docstring, the same
query shape C19 Part A gives the UKIDSS pull); this module does the one
real selection itself -- point-like sources only (`POINT_LIKE_CLASSES`,
the archive's own `mergedClass` -1/-2), with NO error-bit cut at all --
then the nside-512, half-magnitude histogram, exactly `ukidss_counts`'
own binning: `healpy.ang2pix(512, l, b, nest=True, lonlat=True)` (no
frame conversion -- `vhsSource`'s `l`/`b` are already Galactic), then a
half-mag histogram per pixel, restricted to the region's own occupied
pixel set from the granule map.

CALIBRATION, the same product and construction `ukidss_counts` writes
(C19 Part B item 5): the ratio (2MASS's own count / this module's own
count) over the Ks `OVERLAP_LOWER_MAG` to 2MASS's own faint-edge
overlap window, on the pixels this region's VHS pull covers at all,
with the combined-Poisson log interval `population.anchor_tiles.
pooled_ratio_pass1` already uses elsewhere in this project. 2MASS is
the reference, VHS the extension (owner ruling): `population.
anchor_tiles` reads this ratio and multiplies this product's own deep
counts by it before anything else sees them -- no SESNA count enters
this calibration on either side, only the two external surveys.

Photometric system. VIRCAM Ks (VHS) is calibrated to the VISTA system
(Gonzalez-Fernandez et al. 2018, MNRAS 474, 5459's own VISTA-to-2MASS
transformation carries a J-or-H colour term of the same order as
UKIDSS's own WFCAM-to-2MASS term, `ukidss_counts`'s own docstring);
this download carries Ks alone, not J or H, so the term cannot be
evaluated per source and is left unapplied for the same reason
`ukidss_counts` leaves its own term unapplied: a few percent of one
0.5-mag bin width, not a correction the counts feel.

C28 (2026-10-21): `MAG_EDGES` extends to 18.5 against the download's
own new `Ks < 18.5` cut, and this module measures its own completeness
turnover the same way `ukidss_counts.differential_turnover_mag` does --
see that module's own docstring for the construction and
`KS_COMPLETE_TO`'s own reading below.
"""

import os

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import batches as batches_module
from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.granules import access

_STEM = "counts_vhs_hpx512"

NSIDE = 512

#: Row-batch memory budget (rule 10b): one row per VHS detection, as
#: `ukidss_counts.ROW_BATCH_BUDGET_BYTES` is for its own download.
ROW_BATCH_BUDGET_BYTES = 512 << 20
#: Nineteen half-magnitude bins, 9.0-18.5 (C28, 2026-10-21: raised from
#: 17.0) -- the same grid `sky.derived.ukidss_counts.MAG_EDGES` uses,
#: against this download's own `Ks < 18.5` cut
#: (`sky.download.vhs.build.K_CEILING`).
MAG_EDGES = np.arange(9.0, 18.5001, 0.5)

#: C28 item 3: the same turnover-test constants
#: `sky.derived.ukidss_counts` uses.
TURNOVER_LOWER_MAG = 15.0
TURNOVER_SIGMA = 1.0

#: C28 item 3's own escape clause: VHS's published 5-sigma depth for the
#: main/wide survey component (McMahon et al. 2013, The Messenger 154,
#: 35, sec. 2 survey design table) -- used in place of a measured
#: turnover the ambiguity check (`differential_turnover_mag`'s own
#: `ambiguous` flag) rejects, the same construction
#: `sky.derived.ukidss_counts.PUBLISHED_DEPTH_MAG` uses for GPS.
PUBLISHED_DEPTH_MAG = 18.0

#: The archive's own point-like `mergedClass` codes this module counts
#: (module docstring) -- the same two `sky.derived.ukidss_counts.
#: POINT_LIKE_CLASSES` uses; `-3` (probable galaxy) is on the
#: download's own CSV but never counted here.
POINT_LIKE_CLASSES = (-1, -2)

#: The overlap window's own bright edge, the same value
#: `sky.derived.ukidss_counts.OVERLAP_LOWER_MAG` uses -- both anchors
#: are taken as complete from here to the 2MASS anchor's faint cut.
OVERLAP_LOWER_MAG = 13.0


def _download_path(config, region):
    return f"{config.data_root}/sky/download/vhs/sources_vhs_hpx512__{region}.csv"


def _region_pixels(config, region):
    """The region's own occupied nside-512 pixels, sorted, from the
    granule map -- the same call `twomass_counts`/`ukidss_counts` make,
    so this product's `HPX_PIX_512` is theirs by construction."""
    return np.unique(access.region_slice(config, region)["hpx_pix_512"])


def _csv_row_batch_size(csv_path, budget_bytes=ROW_BATCH_BUDGET_BYTES):
    """Rows per batch (rule 10b) for `csv_path`: its own first data row's
    byte width sets `batches.batches`' per-row footprint."""
    with open(csv_path, "r") as f:
        f.readline()  # header
        first = f.readline()
    if not first:
        return 1
    with open(csv_path, "r") as f:
        n_rows = sum(1 for _ in f) - 1
    row_bytes = len(first.encode("utf-8"))
    _, stop = next(batches_module.batches(max(n_rows, 1), row_bytes, budget_bytes=budget_bytes))
    return stop


def _twomass_overlap_edges(config, region):
    """The sibling 2MASS counts product's own `MAG_EDGES` (and matching
    `HPX_PIX_512`, `N`), restricted to the overlap window
    `OVERLAP_LOWER_MAG` to its own faint edge -- the exact bins both
    anchors are taken as complete on, read from the product itself
    rather than re-typed (the same read `sky.derived.ukidss_counts`
    makes)."""
    path = config_module.product_path(config, "sky/derived", "twomass", "counts", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"vhs_counts derive: no 2MASS counts product for region {region!r} at "
            f"{path} -- run `sesnaimpute.sky.derived.twomass_counts` first")
    with h5py.File(path, "r") as f:
        pixels = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        edges = np.asarray(f["MAG_EDGES"][:], dtype=np.float64)
        n = np.asarray(f["N"][:], dtype=np.float64)
    lo = int(np.searchsorted(edges, OVERLAP_LOWER_MAG + 1e-9)) - 1
    lo = max(lo, 0)
    return pixels, edges[lo:], n[:, lo:]


def _counts_matrix(pixels, csv_path, overlap_edges):
    """`(n_pix, n_bins)` int64 main histogram (`MAG_EDGES`) and
    `(n_pix, n_overlap_bins)` int64 overlap histogram (`overlap_edges`,
    the sibling 2MASS product's own overlap-window edges): each
    downloaded row placed at its own Galactic nside-512 pixel, kept only
    if its `mergedClass` is point-like (`POINT_LIKE_CLASSES`) with NO
    error-bit cut, restricted to `pixels` (the region's own occupied
    set). Read in row batches (rule 10b)."""
    n_pix, n_bins = pixels.size, MAG_EDGES.size - 1
    n_ov_bins = overlap_edges.size - 1
    n = np.zeros((n_pix, n_bins), dtype=np.int64)
    n_overlap = np.zeros((n_pix, n_ov_bins), dtype=np.int64)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"vhs_counts derive: no download CSV at {csv_path} -- run the "
            "'sesnaimpute.sky.download.vhs.build' RUNBOOK line first")
    chunksize = _csv_row_batch_size(csv_path)
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if chunk.empty:
            continue
        point_like = chunk["mergedClass"].isin(POINT_LIKE_CLASSES).to_numpy()
        if not point_like.any():
            continue
        chunk = chunk[point_like]
        l_deg = chunk["l"].to_numpy(dtype=np.float64)
        b_deg = chunk["b"].to_numpy(dtype=np.float64)
        k_mag = chunk["ksAperMag3"].to_numpy(dtype=np.float64)
        row_pix = hp.ang2pix(NSIDE, l_deg, b_deg, nest=True, lonlat=True).astype(np.int64)
        loc = np.searchsorted(pixels, row_pix)
        in_set = (loc < n_pix) & (pixels[np.minimum(loc, n_pix - 1)] == row_pix)

        bin_idx = np.searchsorted(MAG_EDGES, k_mag, side="right") - 1
        in_bin = (bin_idx >= 0) & (bin_idx < n_bins)
        keep = in_set & in_bin
        np.add.at(n, (loc[keep], bin_idx[keep]), 1)

        ov_idx = np.searchsorted(overlap_edges, k_mag, side="right") - 1
        in_ov = (ov_idx >= 0) & (ov_idx < n_ov_bins)
        keep_ov = in_set & in_ov
        np.add.at(n_overlap, (loc[keep_ov], ov_idx[keep_ov]), 1)
    return n, n_overlap


def calibration_ratio(n_2mass_overlap, n_vhs_overlap, deep_covered):
    """`(ratio, lo, hi)`: the overlap-window ratio (summed 2MASS count /
    summed VHS count, over the pixels VHS covers at all) and its
    combined-Poisson log interval -- the same construction
    `sky.derived.ukidss_counts.calibration_ratio` and `population.
    anchor_tiles.pooled_ratio_pass1` use for a ratio of two counts
    elsewhere in this project. `nan` for all three where VHS carries no
    overlap-window coverage at all."""
    n2 = float(n_2mass_overlap[deep_covered].sum())
    nv = float(n_vhs_overlap[deep_covered].sum())
    if nv <= 0.0 or n2 <= 0.0:
        return float("nan"), float("nan"), float("nan")
    ratio = n2 / nv
    sigma_ln = np.sqrt(1.0 / n2 + 1.0 / nv)
    return ratio, ratio * np.exp(-sigma_ln), ratio * np.exp(sigma_ln)


def _floor_half_mag(mag):
    """`mag` rounded DOWN to the nearest half-magnitude grid point --
    `sky.derived.ukidss_counts._floor_half_mag`, lifted verbatim."""
    return float(np.floor(mag * 2.0 + 1e-9) / 2.0)


def _wls_predict(x, y, sigma_y, x0):
    """`sky.derived.ukidss_counts._wls_predict`, lifted verbatim (that
    module's own docstring)."""
    w = 1.0 / sigma_y ** 2
    s = w.sum()
    sx = (w * x).sum()
    sy = (w * y).sum()
    sxx = (w * x * x).sum()
    sxy = (w * x * y).sum()
    delta = s * sxx - sx * sx
    slope = (s * sxy - sx * sy) / delta
    intercept = (sxx * sy - sx * sxy) / delta
    predicted = slope * x0 + intercept
    var_pred = (sxx - 2.0 * x0 * sx + x0 * x0 * s) / delta
    return predicted, np.sqrt(max(var_pred, 0.0))


def differential_turnover_mag(n, deep_covered, mag_edges=None,
                               lower_mag=TURNOVER_LOWER_MAG, sigma=TURNOVER_SIGMA):
    """`sky.derived.ukidss_counts.differential_turnover_mag`'s own
    construction (that module's own docstring: tracks the pooled count's
    own log10 SLOPE bin to bin, never the level, since even a fully
    complete luminosity function decelerates smoothly well before any
    hard depth limit; the ambiguity check below, same construction),
    against this module's `MAG_EDGES`. Returns `(complete_to, centres,
    counts, turned, ambiguous)`."""
    edges = MAG_EDGES if mag_edges is None else np.asarray(mag_edges, dtype=np.float64)
    if not np.any(deep_covered):
        return float("nan"), np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), False
    lo_bin = max(int(np.searchsorted(edges, lower_mag + 1e-9)) - 1, 0)
    edges_sub = edges[lo_bin:]
    counts = n[deep_covered][:, lo_bin:].sum(axis=0).astype(np.float64)
    centres = 0.5 * (edges_sub[:-1] + edges_sub[1:])
    n_bins = counts.size
    turned = np.zeros(n_bins, dtype=bool)
    if n_bins < 4:
        return float(edges_sub[-1]), centres, counts, turned, False
    empty = np.flatnonzero(counts <= 0.0)
    if empty.size:
        first_empty = int(empty[0])
        turned[first_empty:] = True
        return float(edges_sub[first_empty]), centres, counts, turned, False
    log_counts = np.log10(counts)
    sigma_log = 1.0 / (np.sqrt(counts) * np.log(10.0))
    dx = centres[1:] - centres[:-1]
    slopes = (log_counts[1:] - log_counts[:-1]) / dx
    sigma_slope = np.sqrt(sigma_log[1:] ** 2 + sigma_log[:-1] ** 2) / dx
    x_slope = centres[1:]
    good = [0, 1]
    first_turn = None
    for k in range(2, slopes.size):
        predicted, sigma_fit = _wls_predict(x_slope[good], slopes[good], sigma_slope[good], x_slope[k])
        sigma_total = np.sqrt(sigma_fit ** 2 + sigma_slope[k] ** 2)
        if (predicted - slopes[k]) > sigma * sigma_total:
            turned[k + 1] = True
            first_turn = (k + 1) if first_turn is None else first_turn
        else:
            good.append(k)
    if first_turn is None:
        return float(edges_sub[-1]), centres, counts, turned, False
    ambiguous = first_turn + 1 < n_bins and bool(np.max(counts[first_turn + 1:]) > counts[first_turn])
    complete_to = float(edges_sub[first_turn])
    return complete_to, centres, counts, turned, ambiguous


def quartile_turnover_mags(n, deep_covered, mag_edges=None,
                           lower_mag=TURNOVER_LOWER_MAG, sigma=TURNOVER_SIGMA):
    """`sky.derived.ukidss_counts.quartile_turnover_mags`'s own
    construction, against this module's `MAG_EDGES`."""
    idx_covered = np.flatnonzero(deep_covered)
    if idx_covered.size < 8:
        return []
    density = n[idx_covered].sum(axis=1).astype(np.float64)
    q_edges = np.percentile(density, [25.0, 50.0, 75.0])
    q_of = np.searchsorted(q_edges, density, side="right")
    out = []
    for q in range(4):
        mask = np.zeros(deep_covered.shape, dtype=bool)
        mask[idx_covered[q_of == q]] = True
        if mask.sum() < 4:
            out.append(float("nan"))
            continue
        complete_to, _, _, _, _ = differential_turnover_mag(n, mask, mag_edges, lower_mag, sigma)
        out.append(complete_to)
    return out


def _write_region(config, region, pixels, n, calibration, ks_complete_to):
    out_path = config_module.product_path(config, "sky/derived", "vhs", "counts", "hpx512", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.attrs["N_SOURCES"] = int(n.sum())
        for name, data in (
            ("HPX_PIX_512", pixels.astype(np.int64)),
            ("MAG_EDGES", MAG_EDGES.astype(np.float64)),
            ("N", n.astype(np.int64)),
            ("CALIBRATION", np.asarray(calibration, dtype=np.float64)),
            ("KS_COMPLETE_TO", np.float64(ks_complete_to)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])


def build(config, regions=None):
    """Writes `sky/derived/vhs/counts_vhs_hpx512__<Region>.hdf5` for
    each requested region (default: all thirty). A region whose
    download is header-only (outside VHS's footprint) writes `N` all
    zero, with `N_SOURCES = 0` and `CALIBRATION` all NaN -- the
    coverage record `sky.download.vhs.build`'s own acceptance check
    reads."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    written = []
    with progress_module.Stage("sky.derived.vhs_counts") as st:
        n_regions = len(regions)
        total_counts = 0
        for i, region in enumerate(regions):
            pixels = _region_pixels(config, region)
            tm_pixels, overlap_edges, n_2mass_overlap = _twomass_overlap_edges(config, region)
            if not np.array_equal(pixels, tm_pixels):
                raise ValueError(
                    f"vhs_counts derive: {region!r}'s granule-map pixel set disagrees "
                    "with the 2MASS counts product's own pixel set -- rerun "
                    "`sesnaimpute.sky.derived.twomass_counts` for this region")
            n, n_overlap = _counts_matrix(pixels, _download_path(config, region), overlap_edges)
            deep_covered = n.sum(axis=1) > 0
            calibration = calibration_ratio(n_2mass_overlap, n_overlap, deep_covered)
            complete_to_raw, centres, counts, turned, ambiguous = differential_turnover_mag(n, deep_covered)
            if not np.isfinite(complete_to_raw):
                ks_complete_to = float("nan")
            elif ambiguous:
                ks_complete_to = PUBLISHED_DEPTH_MAG
            else:
                ks_complete_to = _floor_half_mag(complete_to_raw)
            quartiles = quartile_turnover_mags(n, deep_covered)
            _write_region(config, region, pixels, n, calibration, ks_complete_to)
            written.append(region)
            total_counts += int(n.sum())
            ratio, lo, hi = calibration
            print(f"vhs_counts derive: {region}: {pixels.size} pixels, {int(n.sum())} total counts, "
                  f"{int(deep_covered.sum())} deep-covered, calibration (2MASS/VHS, Ks "
                  f"{OVERLAP_LOWER_MAG:.1f}-{overlap_edges[-1]:.1f}) = {ratio:.4f} "
                  f"[{lo:.4f}, {hi:.4f}]")
            if np.isfinite(complete_to_raw):
                bin_report = ", ".join(
                    f"{centres[j]:.2f}:{counts[j]:.0f}{'*' if turned[j] else ''}"
                    for j in range(centres.size))
                amb_note = (f" -- AMBIGUOUS (a later bin's count exceeds the flagged bin's own "
                            f"count): using the published depth {PUBLISHED_DEPTH_MAG:.1f} instead "
                            f"(McMahon et al. 2013, The Messenger 154, 35)" if ambiguous else "")
                print(f"vhs_counts derive: {region}: turnover measured at {complete_to_raw:.2f}{amb_note} "
                      f"(KS_COMPLETE_TO {ks_complete_to:.1f}, anchor edge min(18.0, this) applies "
                      f"downstream); bins (centre:count, * turned over): {bin_report}; "
                      f"pixel-density-quartile turnovers: {quartiles}")
            else:
                print(f"vhs_counts derive: {region}: no deep coverage at all -- KS_COMPLETE_TO nan")
            st.tick(i + 1, n_regions, "regions")
        st.done(None, regions=n_regions, total_counts=total_counts)
    return dict(regions_written=written)


if __name__ == "__main__":
    build_module.run(build)
