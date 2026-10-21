"""UKIDSS GPS point-source counts per nside-512 HEALPix pixel
(SPEC_BMSTP_DRAFT.md 5.1's STAR anchor: this is the 2MASS anchor
counts product's own shape, `sky.derived.twomass_counts`, carried past
its 14.3 cut to K = 17.0 -- W38 reads this to extend the STAR weight's
faint 2MASS bins with GPS's deeper reach where GPS covers the region
(Lucas et al. 2008, MNRAS 391, 136: K approx 18 5-sigma, |b| lesssim
5 deg, l 15-107 deg and 141-230 deg)).

C19 (2026-10-21). `sky.download.ukidss_gps.build` no longer applies
W34's own post-processing error-bit cut or narrow class cut at query
time (that module's own docstring) -- C18's diagnosis found the old
cut holding only 0.77-0.79 of the stars SESNA's own 2MASS photometry
shows at Ks 12-14. This module now does the one real selection itself:
point-like sources only (`POINT_LIKE_CLASSES`, the archive's own
`mergedClass` -1/-2, star/probable star -- never -3, probable galaxy,
which the download keeps on hand but this module does not count), with
NO error-bit cut at all (`k_1ppErrBits` is carried by the download
verbatim but read by nothing downstream of this module). Derive still
never fetches: the nside-512, half-magnitude histogram is exactly
`twomass_counts`'s own binning, `healpy.ang2pix(512, l, b, nest=True,
lonlat=True)` (no frame conversion -- GPS's `l`/`b` are already
Galactic) then a half-mag histogram per pixel, restricted to the
region's own occupied pixel set from the granule map -- the same set
`twomass_counts` restricts to, since both read `access.region_slice`
for it, W37's pixel-identity check.

The CALIBRATION this module now also computes and writes (C19 Part A
item 2): even with the open class cut above, this module's own point-
source counts in the magnitude range both anchors measure (Ks
`OVERLAP_LOWER_MAG` to the 2MASS anchor's own faint edge,
`anchor_tiles.KS_CUT_MAG` = 14.3 -- read from the sibling 2MASS counts
product's own `MAG_EDGES`, never re-typed here) can still disagree with
2MASS's: the ratio (2MASS's own count / this module's own count) over
those bins, on the pixels this region's deep survey actually covers
(`N.sum(axis=1) > 0`, the same "deep-covered" test `population.
anchor_tiles` runs on this product), with the combined-Poisson log
interval `population.anchor_tiles.pooled_ratio_pass1` already uses
elsewhere in this project for the same kind of ratio (`sigma_ln =
sqrt(1/n_2mass + 1/n_ukidss)`). 2MASS is the reference, UKIDSS the
extension (owner ruling): `population.anchor_tiles` reads this ratio
and multiplies this product's own deep counts by it before anything
else sees them -- no SESNA count enters this calibration on either
side, only the two external surveys.

Photometric system. WFCAM K (UKIDSS/GPS) is calibrated to the UKIRT
system (Hewett, Warren, Leggett & Hodgkin 2006, MNRAS 367, 454), and its
transformation to 2MASS Ks (Hodgkin, Irwin, Hewett & Warren 2009, MNRAS
394, 675) carries a J-or-H colour term of order 0.01-0.03 mag (Hewett
et al. 2006 table 5's synthetic-photometry coefficients on a typical
GPS-field J-K); this download carries K alone, not J or H, so the term
cannot be evaluated per source. Left unapplied: at most 0.03 mag against
this histogram's 0.5-mag bins is a few percent of one bin width, not a
correction the counts feel.
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

_STEM = "counts_ukidss_hpx512"

NSIDE = 512

#: Row-batch memory budget (rule 10b): like `twomass_counts`'s download,
#: this is one row per GPS detection, and a dense region's file can carry
#: millions of rows (Cygnus X measured ~9M, W37 report) -- never read
#: whole. The histogram accumulation (`np.add.at`) is exact and
#: order-free, so batching changes nothing it computes.
ROW_BATCH_BUDGET_BYTES = 512 << 20
#: Sixteen half-magnitude bins, 9.0-17.0: the download's own K < 17.5
#: cut (`sky.download.ukidss_gps.build.K_CEILING`) gives this top edge a
#: half-bin of headroom, as `twomass_counts.MAG_EDGES` does against its
#: 14.3 cut.
MAG_EDGES = np.arange(9.0, 17.0001, 0.5)

#: C19: the archive's own point-like `mergedClass` codes this module
#: counts (module docstring) -- `-3` (probable galaxy) is on the
#: download's own CSV but never counted here.
POINT_LIKE_CLASSES = (-1, -2)

#: C19 Part A item 2: the overlap window's own bright edge -- both
#: anchors are taken as complete from here to the 2MASS anchor's faint
#: cut (read from the sibling 2MASS product's own `MAG_EDGES`, not
#: re-typed as a constant here).
OVERLAP_LOWER_MAG = 13.0


def _download_path(config, region):
    return f"{config.data_root}/sky/download/ukidss_gps/sources_ukidss_hpx512__{region}.csv"


def _region_pixels(config, region):
    """The region's own occupied nside-512 pixels, sorted, from the
    granule map -- the same call `twomass_counts` makes, so the two
    products' `HPX_PIX_512` are identical by construction (W37's
    pixel-identity check)."""
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
    rather than re-typed."""
    path = config_module.product_path(config, "sky/derived", "twomass", "counts", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ukidss_counts derive: no 2MASS counts product for region {region!r} at "
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
            f"ukidss_counts derive: no download CSV at {csv_path} -- run the "
            "'sesnaimpute.sky.download.ukidss_gps.build' RUNBOOK line first")
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
        k_mag = chunk["k_1AperMag3"].to_numpy(dtype=np.float64)
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


def calibration_ratio(n_2mass_overlap, n_ukidss_overlap, deep_covered):
    """`(ratio, lo, hi)`: the overlap-window ratio (summed 2MASS count /
    summed UKIDSS count, over the pixels UKIDSS covers at all) and its
    combined-Poisson log interval -- the same construction
    `population.anchor_tiles.pooled_ratio_pass1` uses for a ratio of two
    counts elsewhere in this project. `nan` for all three where UKIDSS
    carries no overlap-window coverage at all (nothing to calibrate
    against; `population.anchor_tiles` must not multiply by a fabricated
    1.0 in that case -- it reads the absence and skips the multiply)."""
    n2 = float(n_2mass_overlap[deep_covered].sum())
    nu = float(n_ukidss_overlap[deep_covered].sum())
    if nu <= 0.0 or n2 <= 0.0:
        return float("nan"), float("nan"), float("nan")
    ratio = n2 / nu
    sigma_ln = np.sqrt(1.0 / n2 + 1.0 / nu)
    return ratio, ratio * np.exp(-sigma_ln), ratio * np.exp(sigma_ln)


def _write_region(config, region, pixels, n, calibration):
    out_path = config_module.product_path(config, "sky/derived", "ukidss", "counts", "hpx512", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.attrs["N_SOURCES"] = int(n.sum())
        for name, data in (
            ("HPX_PIX_512", pixels.astype(np.int64)),
            ("MAG_EDGES", MAG_EDGES.astype(np.float64)),
            ("N", n.astype(np.int64)),
            ("CALIBRATION", np.asarray(calibration, dtype=np.float64)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])


def build(config, regions=None):
    """Writes `sky/derived/ukidss/counts_ukidss_hpx512__<Region>.hdf5` for
    each requested region (default: all thirty). A region whose download
    is header-only (outside GPS's footprint) writes `N` all zero, with
    `N_SOURCES = 0` and `CALIBRATION` all NaN -- the coverage record
    W37's acceptance check reads."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    written = []
    with progress_module.Stage("sky.derived.ukidss_counts") as st:
        n_regions = len(regions)
        total_counts = 0
        for i, region in enumerate(regions):
            pixels = _region_pixels(config, region)
            tm_pixels, overlap_edges, n_2mass_overlap = _twomass_overlap_edges(config, region)
            if not np.array_equal(pixels, tm_pixels):
                raise ValueError(
                    f"ukidss_counts derive: {region!r}'s granule-map pixel set disagrees "
                    "with the 2MASS counts product's own pixel set -- rerun "
                    "`sesnaimpute.sky.derived.twomass_counts` for this region")
            n, n_overlap = _counts_matrix(pixels, _download_path(config, region), overlap_edges)
            deep_covered = n.sum(axis=1) > 0
            calibration = calibration_ratio(n_2mass_overlap, n_overlap, deep_covered)
            _write_region(config, region, pixels, n, calibration)
            written.append(region)
            total_counts += int(n.sum())
            ratio, lo, hi = calibration
            print(f"ukidss_counts derive: {region}: {pixels.size} pixels, {int(n.sum())} total counts, "
                  f"{int(deep_covered.sum())} deep-covered, calibration (2MASS/UKIDSS, Ks "
                  f"{OVERLAP_LOWER_MAG:.1f}-{overlap_edges[-1]:.1f}) = {ratio:.4f} "
                  f"[{lo:.4f}, {hi:.4f}]")
            st.tick(i + 1, n_regions, "regions")
        st.done(None, regions=n_regions, total_counts=total_counts)
    return dict(regions_written=written)


if __name__ == "__main__":
    build_module.run(build)
