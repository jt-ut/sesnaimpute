"""UKIDSS GPS point-source counts per nside-512 HEALPix pixel
(SPEC_BMSTP_DRAFT.md 5.1's STAR anchor: this is the 2MASS anchor
counts product's own shape, `sky.derived.twomass_counts`, carried past
its 14.3 cut to K = 18.0 at most -- W38/C28 read this to extend the
STAR weight's faint 2MASS bins with GPS's deeper reach where GPS covers
the region (Lucas et al. 2008, MNRAS 391, 136: K approx 18 5-sigma,
|b| lesssim 5 deg, l 15-107 deg and 141-230 deg)).

C28 (2026-10-21, "the deep Ks anchor reaches the faint bin's own
stars"): the 18-56 uJy STAR members sit at Ks 17-18.3 (C24f/g), beyond
the old 17.0 anchor edge. The download now reaches K < 18.5
(`sky.download.ukidss_gps.build.K_CEILING`) and `MAG_EDGES` extends to
18.5 to match (half a bin of headroom past the new 18.0 anchor
ceiling, same convention as the old 17.0/17.5 pair). This module also
now measures, rather than assumes, where ITS OWN counts turn over past
`TURNOVER_LOWER_MAG`: `differential_turnover_mag` tracks the pooled,
deep-covered-pixel log10 differential count's own SLOPE (its bin-to-bin
rate, not the count itself -- a real, still-complete luminosity
function decelerates smoothly well before any hard depth limit, so the
level itself is never log-linear; it is the slope's own trend that a
genuine turnover breaks) one half-mag bin at a time, and calls the
first bin whose own slope falls below the brighter bins' fitted slope
trend by more than its own Poisson scatter the turnover; `KS_COMPLETE_TO`
is the bright edge of the last bin still consistent with that slope
trend (this product's own faint edge, 18.5, if no bin's slope ever
falls below it).
`population.anchor_tiles` takes the region's own deep anchor edge as
`min(18.0, floor_half_mag(KS_COMPLETE_TO))` -- never past the
published depth even where this measurement is ambiguous.
`quartile_turnover_mags` repeats the same test within each quartile of
the deep-covered pixels' own total detection count (a crowding proxy),
reported but not stored: a pooled turnover driven by the crowded
quartile alone shows up as a spread across the four numbers.

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
#: Nineteen half-magnitude bins, 9.0-18.5 (C28, 2026-10-21: raised from
#: 17.0): the download's own K < 18.5 cut
#: (`sky.download.ukidss_gps.build.K_CEILING`) gives this top edge a
#: half-bin of headroom, as `twomass_counts.MAG_EDGES` does against its
#: 14.3 cut.
MAG_EDGES = np.arange(9.0, 18.5001, 0.5)

#: C28 item 3: the turnover test's own bright edge -- the differential
#: count's log slope is measured from here to this product's own faint
#: edge (MAG_EDGES[-1]), pooled over deep-covered pixels and per
#: pixel-density quartile.
TURNOVER_LOWER_MAG = 15.0
#: C28 item 3: a bin counts as turned over when its own pooled log10
#: differential count falls below the brighter, already-accepted bins'
#: fitted trend by more than this many Poisson sigma on its own count
#: (the brief's "more than the Poisson scatter").
TURNOVER_SIGMA = 1.0

#: C28 item 3's own escape clause: GPS's published 5-sigma depth (module
#: docstring, Lucas et al. 2008, MNRAS 391, 136) -- used in place of a
#: measured turnover the ambiguity check (`differential_turnover_mag`'s
#: own `ambiguous` flag) rejects: a later bin's count exceeding the
#: flagged bin's own count means the flagged deceleration did not hold,
#: so the brief's own fallback is this published depth, never a
#: silently patched measurement.
PUBLISHED_DEPTH_MAG = 18.0

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


def _floor_half_mag(mag):
    """`mag` rounded DOWN to the nearest half-magnitude grid point (the
    grid `MAG_EDGES` itself sits on) -- never up, so a measured turnover
    never grants the anchor a bin it did not actually clear."""
    return float(np.floor(mag * 2.0 + 1e-9) / 2.0)


def _wls_predict(x, y, sigma_y, x0):
    """Weighted least squares `y = a*x + b` on `(x, y)` with per-point
    sigma `sigma_y` (standard heteroscedastic-error propagation, valid
    down to two points since the weights come from the DATA's own known
    measurement scatter, not the fit's residuals): returns
    `(predicted_y0, sigma_y0)`, the fit's own predictive value and
    uncertainty at `x0` (excluding `x0`'s own measurement scatter, which
    the caller adds in quadrature separately)."""
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
    """C28 item 3, MEASURED not assumed: the deep survey's own
    completeness turnover. Pools `n`'s counts over `deep_covered` pixels
    into one differential count per half-mag bin from `lower_mag` to
    `mag_edges`'s own faint edge, then tracks that count's own log10
    SLOPE bin to bin -- never the level itself, which even a fully
    complete luminosity function does not hold log-linear (it
    decelerates smoothly well before any hard depth limit; testing the
    LEVEL against a constant-slope trend reads that ordinary
    deceleration as a false turnover at the very first tested bin,
    measured against real GPS/VHS data). The first two slopes (between
    bins 0-1 and 1-2) seed a trend in the slope itself (weighted least
    squares, slope vs the bin centre it leads into, each point weighted
    by its own propagated Poisson variance), and each next bin's own
    slope is tested against that trend extrapolated from every slope
    accepted so far: a slope that falls below the prediction by more
    than `sigma` combined sigma -- the fit's own predictive uncertainty
    at that point (`_wls_predict`) and the slope's own propagated
    Poisson scatter, added in quadrature -- marks its bin turned over,
    and its own slope is never added to the trend (once the survey has
    turned over, a brighter trend fit through it is no longer
    meaningful, though later bins are still tested against the last
    good trend). An empty bin cannot be log10'd and marks its own bin,
    and every bin after it, turned over outright -- no detections is
    the strongest possible falloff.

    AMBIGUITY (the brief's own escape clause, "if the slope test is
    ambiguous say so"): a slope deceleration can be flagged significant
    at one bin and still be followed by the count rising again past
    that very bin's own value -- a real, sustained turnover never does
    that. Where some LATER bin's own count exceeds the first
    flagged bin's count, the flag is ambiguous: `KS_COMPLETE_TO` is
    measured here exactly as flagged (never silently patched), but
    `ambiguous=True` tells the caller to report it and fall back to the
    survey's own published depth instead, per the brief.

    Returns `(complete_to, centres, counts, turned, ambiguous)`:
    `complete_to` is the bright edge of the first turned-over bin, or
    `mag_edges[-1]` (this product's own faint edge) if no bin's slope
    ever falls below the trend -- the survey is then read as complete
    to its own available depth, never beyond it. `centres`/`counts`/
    `turned` are the window's own per-bin arrays, for the crowding
    check and the report (`turned`'s first two entries are always
    False: a bin's slope is only defined, and only tested, from the
    third bin on). `ambiguous` is always False where no bin is flagged
    at all, or where the flagged bin is the window's own last bin (no
    later bin to contradict it). `nan` (and empty `centres`/`counts`/
    `turned`, `ambiguous` False) where `deep_covered` holds no pixel at
    all -- nothing to measure, the same convention `calibration_ratio`
    uses for no coverage."""
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
    x_slope = centres[1:]  # slope k (bin k -> bin k+1) is keyed to the bin it leads into, centres[k+1]
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
    """C28 item 3's own crowding check: `differential_turnover_mag`
    repeated within each quartile of the deep-covered pixels' own total
    detection count (`n.sum(axis=1)`, a crowding proxy -- every nside-512
    pixel has the same solid angle, so a higher total count is a denser
    field). Four independent turnover reads instead of one pooled read:
    a turnover driven by the crowded quartile alone, rather than the
    whole deep-covered set, shows up as a spread across the four
    numbers (reported, never stored -- `KS_COMPLETE_TO` is the pooled
    read alone)."""
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
            ("KS_COMPLETE_TO", np.float64(ks_complete_to)),
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
            complete_to_raw, centres, counts, turned, ambiguous = differential_turnover_mag(n, deep_covered)
            if not np.isfinite(complete_to_raw):
                ks_complete_to = float("nan")
            elif ambiguous:
                # the brief's own escape clause: a flagged deceleration a
                # later bin's count goes on to exceed is not a sustained
                # turnover -- fall back to the survey's own published
                # depth (cited above), never silently patch the measurement.
                ks_complete_to = PUBLISHED_DEPTH_MAG
            else:
                ks_complete_to = _floor_half_mag(complete_to_raw)
            quartiles = quartile_turnover_mags(n, deep_covered)
            _write_region(config, region, pixels, n, calibration, ks_complete_to)
            written.append(region)
            total_counts += int(n.sum())
            ratio, lo, hi = calibration
            print(f"ukidss_counts derive: {region}: {pixels.size} pixels, {int(n.sum())} total counts, "
                  f"{int(deep_covered.sum())} deep-covered, calibration (2MASS/UKIDSS, Ks "
                  f"{OVERLAP_LOWER_MAG:.1f}-{overlap_edges[-1]:.1f}) = {ratio:.4f} "
                  f"[{lo:.4f}, {hi:.4f}]")
            if np.isfinite(complete_to_raw):
                bin_report = ", ".join(
                    f"{centres[j]:.2f}:{counts[j]:.0f}{'*' if turned[j] else ''}"
                    for j in range(centres.size))
                amb_note = (f" -- AMBIGUOUS (a later bin's count exceeds the flagged bin's own "
                            f"count): using the published depth {PUBLISHED_DEPTH_MAG:.1f} instead "
                            f"(Lucas et al. 2008, MNRAS 391, 136)" if ambiguous else "")
                print(f"ukidss_counts derive: {region}: turnover measured at {complete_to_raw:.2f}{amb_note} "
                      f"(KS_COMPLETE_TO {ks_complete_to:.1f}, anchor edge min(18.0, this) applies "
                      f"downstream); bins (centre:count, * turned over): {bin_report}; "
                      f"pixel-density-quartile turnovers: {quartiles}")
            else:
                print(f"ukidss_counts derive: {region}: no deep coverage at all -- KS_COMPLETE_TO nan")
            st.tick(i + 1, n_regions, "regions")
        st.done(None, regions=n_regions, total_counts=total_counts)
    return dict(regions_written=written)


if __name__ == "__main__":
    build_module.run(build)
