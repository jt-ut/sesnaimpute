"""Every acceptance check of the PACS build, each returning `products.CheckRow`s that the stage stores in the product's
CHECKS table: the build-time checks that stop a build (zero image, normalisation), the information-only real-map check,
and the region checks after FINISH (C1, C2, the detection report, C4/C5 against HOPS, the HPPSC checks, the chance
coincidences). A check prints nothing; what it has to say is in its rows (value, bar, verdict, and "n=<cases>; ..." in the
detail).

Depends on: grid, psf, mosaic, detect, inject, photometry, products. Replaces: pacs_completeness.self_check/_real_check/
_acceptance_c1/_acceptance_c2/_report_detection/acceptance/_binomial_error/_verdict/_robust/_at_injected_flux,
pacs_checks.py (all: representative_maps, _aperture_sum, _template_in_aperture, _unit_source, expected_enclosed,
check_positions, _real_sources, normalisation_check), pacs_photometry.acceptance/_c4_ratio/_read_hops/_match_hops/
_read_hppsc/hppsc_checks, and the print-only reporting of finish_region. The two normalisation paths (self_check and
normalisation_check) stay two checks with distinct names (they test different things: the zero-image sky term vs the PSF
divisor), run from ONE place (stages, stage "psf"). Bars are today's, named constants here, except two the science track
changed (FIX_SPEC sec 8-12): the C1 FLUX90 half-split share has no flat 90 %, its bar is derived from the map's own counts
(`_flux90_halves_bar`, science/M5_c1_bar.md sec 3), and the HPPSC flux ratio is information.

Row names (value; bar):
  ZERO_IMAGE_WORST (worst |aperture sum / (injected x EEF) - 1|; <= 0.005), ZERO_IMAGE_TEMPLATE (worst |A_direct/EEF - 1|;
  <= 0.015), ZERO_IMAGE_SKY_SHARE (info)
  NORM_DIVISOR / NORM_INJECTED (worst |dev|; <= 0.003), NORM_PATCH_SUMS (<= 0.01), NORM_TEMPLATE_VS_EEF (<= 0.015)
  REAL_MAP_MEAN_RATIO (info)
  C1_WITHIN_1SIGMA (>= 60), C1_WITHIN_2SIGMA (>= 93), C1_WITHIN_0.03 (info), C1_WORST (info), C1_FLUX90_HALVES_SHARE (>=
  the derived bar), C1_FLUX90_HALVES_MEDIAN (info)
  C2_APERTURE_<class>_BIN<j>, C2_DETECTED_<class>_BIN<j> (robust scatter of the normalised held-out residuals; 0.90 - 1.10
  where >= 200 residuals), C2_APERTURE_BIAS_ABOVE_FLUX90, C2_DETECTED_BIAS_ABOVE_FLUX90 (per cent; |.| <= 1), C2_NAN_SIGMA
  (info: the reads whose SIGMA is NaN, which are not dropped silently); class = quiet, noisy and, where the completeness
  carries the aperture-box/cell coverage ratio r, gradient (|ln r| > 0.03)
  DETECTION_RECOVERY_SNR20 (>= 0.99), DETECTION_R_MATCH (info), DETECTION_FLOOR_SKYBIN<j> (info)
  C4_FLUX_RATIO_70 (info), C4_VS_REFERENCE_PERCENT (+-2), C4_ORIGIN3_<band> (info), C4_HOPS_BRIGHT_ORIGIN<code> (info),
  C5_DETECTED_<lo>_<hi> / C5_PREDICTED_<lo>_<hi> (info)
  HPPSC_COVERAGE, HPPSC_OFFSET_MEDIAN_SEP, HPPSC_FLUX_RATIO, HPPSC_RECOVERY_ALL, HPPSC_RECOVERY_POINT (info),
  HPPSC_RECOVERY_POINT_SNR (>= 0.90 within the binomial error at the bar)
  CHANCE_COINCIDENCES (info)

What the checks read of the carriers (contract changes needed are listed in REPORT_R2_E.md): Mosaic: .psf (MosaicPSF: .meta
= psf.json, .weights, .map_weights, .enclosed_at, .n_sub, .shape), .grid (PixelGrid), .window, .template_patches,
.injection_half_width, .pixel_arcsec, .band; BandCompleteness.held: row, plane, near, recovered, dflux, flux, det_flux and
build_errors = {"scatter": (n_row, n_plane)} (the error built from the build half alone); optionally build_row,
build_plane, build_recovered (the build half's records, for the derived C1 bar).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from sesnaimpute.catalog import curated
from sesnaimpute.definitions import PACS_APERTURES as APERTURES
from sesnaimpute.definitions import PACS_BANDS as BANDS
from sesnaimpute.definitions import PACS_BLEND_RADIUS_BEAMS as BLEND_RADIUS_BEAMS
from sesnaimpute.definitions import PACS_MIN_INJECTIONS as MIN_INJECTIONS
from sesnaimpute.definitions import PACS_POOL_WIDENINGS as POOL_WIDENINGS
from sesnaimpute.definitions import PACS_SNR_BIN_CENTRES as SNR_BIN_CENTRES
from sesnaimpute.definitions import COMPLETENESS_STORAGE_SCALE as STORAGE_SCALE
from sesnaimpute.sky.derived.pacs import grid, inject, mosaic, photometry, products, psf
from sesnaimpute.sky.derived.pacs.inject import BandCompleteness
from sesnaimpute.sky.derived.pacs.mosaic import Mosaic, Window
from sesnaimpute.sky.derived.pacs.products import CheckRow

NAN = float("nan")

# ----------------------------------------------------------------------------------------------------------------------
# bars and constants (moved here, values unchanged)

#: normalisation (pacs_checks.py): real (taken at the release EEF) against the divisor, and injected against real; the
#: release template against the release EEF; pixel sums of the same patch (owner 2026-10-30: 1 %)
TIE_TOLERANCE = 0.003
RELEASE_TOLERANCE = 0.015
SUM_TOLERANCE = 0.01
SUBSAMPLES = 6                   # per input pixel side (the box average)
RIM_ARCSEC = (10.0, 20.0, 30.0)  # inside and outside the footprint edge
FLUX_MJY = 1000.0
PHASES = tuple((a / 3.0, b / 3.0) for b in range(3) for a in range(3))              # input-grid shifts averaged over
SOURCE_PHASES = tuple((a / 3.0, b / 3.0) for b in (-1, 0, 1) for a in (-1, 0, 1))    # source shifts (output pixels)
#: the zero-image check (pacs_completeness.py:1061)
ZERO_CHECK_FLUX_MJY = 500.0
ZERO_CHECK_POSITIONS = 20
ZERO_CHECK_TOLERANCE = 0.005
#: C1 (pacs_completeness._acceptance_c1): the held-out curve against the stored one on the transition planes
C1_TRANSITION = (0.02, 0.98)
C1_SIGMA1_BAR = 60.0
C1_SIGMA2_BAR = 93.0
C1_ABSOLUTE = 0.03
#: FLUX90 of the two halves agrees within this fraction (the share of cells is C1's second statistic)
C1_FLUX90_AGREE = 0.10
#: the derived bar for that share (science/M5_c1_bar.md sec 3): simulated halves from the map's own pooled counts, this many
#: repetitions on at most BAR_MAX_CELLS cells (a regular subsample of rows), a block bootstrap over BAR_BLOCK_ARCSEC blocks
BAR_REPS = 20
BAR_MAX_CELLS = 20000
BAR_BLOCK_ARCSEC = 240.0
BAR_BOOTSTRAP = 400
BAR_MIN_BLOCKS = 3
BAR_MIN_UNIT_CELLS = 8          # a (block, repetition) unit with fewer cells is left out of the bootstrap (M5)
BAR_SEED = 1
#: C2 (RULINGS_3 D3): expected scatter 1.02 +- 0.05, bar 1.00 +- 0.10 where at least C2_MIN_RESIDUALS residuals; bias bar 1 %
C2_BAR = (0.90, 1.10)
C2_MIN_RESIDUALS = 200
C2_BIAS_BAR_PERCENT = 1.0
#: a cell is noisy where its CELL_SIGMA exceeds this multiple of its map's median
C2_NOISY_FACTOR = 2.0
#: the gradient class: |ln r| above this, r = aperture-box coverage / cell coverage (FIX_SPEC sec 12)
GRADIENT_LN_R = 0.03
#: detection report (RULINGS_5): recovery of held-out injections at S/N >= 20 is at least 0.99; sky bins of the floor
DETECTION_SNR = 20.0
DETECTION_RECOVERY_BAR = 0.99
SKY_BINS = 5
MAD_TO_SIGMA = 1.4826
#: HOPS (Fischer et al. 2016, ApJS 224, 5) 70 micron flux of the sources, in mJy, with the target's position, from the
#: VizieR tables of the HOPS download
HOPS_FILE = "J_ApJS_224_5.vot"
HOPS_MATCH_ARCSEC = 2.0
#: the bright-source comparison of the flux convention (C4), and its agreement with the earlier product
C4_MIN_F70_MJY = 200.0
C4_REFERENCE_AGREE_PERCENT = 2.0
#: the bins of HOPS 70 micron flux, mJy, of the detection comparison (C5)
C5_EDGES_MJY = (40.0, 70.0, 100.0, 200.0, 500.0, 1000.0, float("inf"))
#: HPPSC (Marton et al. 2017; `sky.download.hppsc`): band lists, the flux comparison's S/N, and the recovery bar (planner
#: ruling, owner-approved 2026-10-29); the recovery check counts HPPSC's point-like entries (fitted width at most
#: HPPSC_POINT_BEAMS of our beams) of HPPSC S/N at least HPPSC_RECOVERY_SNR (owner ruling 2026-10-29)
HPPSC_FILES = {"P70": ("ppsc_70.csv", "blue"), "P100": ("ppsc_100.csv", "green"), "P160": ("ppsc_160.csv", "red")}
HPPSC_FLUX_SNR = 10.0
HPPSC_RECOVERY_BAR = 0.90
HPPSC_POINT_BEAMS = 1.5
HPPSC_RECOVERY_SNR = 10.0


def _row(name: str, band: str, value: float, low: float = NAN, high: float = NAN, detail: str = "", n: Optional[int] = None,
         info: bool = False) -> CheckRow:
    """A CheckRow: PASS where the value is inside [low, high] (NaN = unbounded), FAIL outside, INFO for `info` or no bar.
    The detail starts with n=<cases> when given."""
    value = float(value)
    bounded = not (np.isnan(low) and np.isnan(high))
    if info or not bounded:
        verdict = "INFO"
    else:
        ok = np.isfinite(value) and (np.isnan(low) or value >= low) and (np.isnan(high) or value <= high)
        verdict = "PASS" if ok else "FAIL"
    return CheckRow(name, band, value, float(low), float(high), verdict, ("n=%d; " % n if n is not None else "") + detail)


def _binomial_error(fraction, n):
    """One-sigma binomial error of a fraction from n trials, the fraction held off 0 and 1 by half a trial."""
    p = np.clip(fraction, 0.5 / np.maximum(n, 1), 1 - 0.5 / np.maximum(n, 1))
    return np.sqrt(p * (1 - p) / np.maximum(n, 1))


def _robust(v):
    return MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))) if v.size else np.nan


# ----------------------------------------------------------------------------------------------------------------------
# the S/N-plane machinery the completeness checks read (inject.py's, until it is filled)

def _pool_matrices(centres: np.ndarray) -> list:
    """The pooling matrices (radius PACS_POOL_ARCSEC x each widening, no class) of cells at `centres` (arcsec)."""
    return inject.centre_matrices(centres, inject.PoolingRule.legacy())


def _snr_brackets(sigma_rows, flux_grid):
    """For every (row, flux plane): the two S/N planes bracketing F / CELL_SIGMA of the row and the weight of the upper one
    (linear in log S/N; held at the grid's ends)."""
    snr = inject.snr_grid()
    lo, n_x = np.log10(snr[0]), snr.size
    step = (np.log10(snr[-1]) - lo) / (n_x - 1)
    with np.errstate(all="ignore"):
        lx = np.log10(flux_grid[None, :] / np.asarray(sigma_rows, float)[:, None])
    t = np.clip((lx - lo) / step, 0.0, n_x - 1.0)
    t = np.where(np.isfinite(t), t, 0.0)
    k0 = np.minimum(np.floor(t).astype(np.int64), n_x - 2)
    return k0, t - k0


def _at_flux(values_x, k0, w):
    """`values_x` (n_row, n_snr) read at the bracketing S/N planes of every (row, flux plane)."""
    r = np.arange(values_x.shape[0])[:, None]
    lo, hi = values_x[r, k0], values_x[r, k0 + 1]
    with np.errstate(invalid="ignore"):
        mid = (1.0 - w) * lo + w * hi
    return np.where(w <= 0.0, lo, np.where(w >= 1.0, hi, mid))


def _at_injected_flux(values, flux_grid, row, flux):
    """values (n_row, n_plane) on the flux grid read for each injection at its own row and injected flux, linearly in log
    flux between the bracketing planes (held at the grid's ends)."""
    lf = np.log10(flux_grid[::-1])
    t = np.interp(np.log10(flux), lf, np.arange(lf.size, dtype=float))
    k0 = np.minimum(np.floor(t).astype(np.int64), lf.size - 2)
    w = t - k0
    v = values[:, ::-1]
    return (1 - w) * v[row, k0] + w * v[row, k0 + 1]


def _counts(row, plane, recovered, n_row, n_x):
    """Injections and recovered injections at every (row, S/N plane)."""
    key = row.astype(np.int64) * n_x + plane
    tot = np.bincount(key, minlength=n_row * n_x).reshape(n_row, n_x).astype(np.float64)
    got = np.bincount(key[recovered], minlength=n_row * n_x).reshape(n_row, n_x).astype(np.float64)
    return tot, got


# ----------------------------------------------------------------------------------------------------------------------
# C1

def _flux90_halves_bar(completeness: dict, held: dict, reps: int = BAR_REPS, seed: int = BAR_SEED) -> Tuple[float, float, float, int]:
    """The derived bar of the C1 FLUX90 share (science/M5_c1_bar.md sec 3): bar = S_hat - 2 sigma, from the map's OWN
    stored pooled counts and curves. Per (cell, S/N plane) the truth p is the pooled recovered fraction of both halves on the
    widening the held half chose (the build half's records, `held` carrying build_row/build_plane/build_recovered; without them the held half alone
    roughens the curves, S_hat falls and a bar from it would be lenient, so there is no bar: only S_hat), the two halves are drawn Binomial(n, p)/n
    (n the held half's pooled count), read at F / CELL_SIGMA and FLUX90 taken as the code takes it; S_hat = 100 x (1 - mean
    over cells of P[|FLUX90_A - FLUX90_B| / FLUX90_B > 10 %]) over `reps` draws; sigma = the standard deviation of that share
    over a bootstrap of the map's (block, repetition) units, BAR_BLOCK_ARCSEC blocks, drawn until the map's cell count is reached
    (the realised share of one split, not the expected one, is what the measured share is). Returns (bar, S_hat,
    sigma, blocks); NaN bar where fewer than BAR_MIN_BLOCKS blocks (the map is too small to have a bar)."""
    snr = inject.snr_grid()
    n_x = snr.size
    flux90 = inject.flux90
    n_row = completeness["curve"].shape[0]
    Ws = _pool_matrices(completeness["centres"])
    stride = max(1, int(np.ceil(n_row / BAR_MAX_CELLS)))
    E = np.arange(0, n_row, stride)
    WE = [W[E] for W in Ws]
    th, gh = _counts(held["row"], held["plane"], held["recovered"].astype(bool), n_row, n_x)
    has_build = "build_row" in held
    if has_build:
        tb, gb = _counts(held["build_row"], held["build_plane"], held["build_recovered"].astype(bool), n_row, n_x)
    chosen = np.full((E.size, n_x), len(WE) - 1)
    for i in reversed(range(len(WE) - 1)):
        chosen = np.where(WE[i] @ th >= MIN_INJECTIONS, i, chosen)

    def pick(M):
        arr = [W @ M for W in WE]
        out = arr[-1].copy()
        for i in reversed(range(len(WE) - 1)):
            out = np.where(chosen == i, arr[i], out)
        return out

    n = pick(th)
    if has_build:
        nb = pick(tb)
        p = np.where(n + nb > 0, (pick(gh) + pick(gb)) / np.maximum(n + nb, 1), 0.0)
    else:                      # the held half alone roughens the truth, S_hat falls and the bar would be lenient: S_hat only
        p = np.where(n > 0, pick(gh) / np.maximum(n, 1), 0.0)
    sigma = completeness["sigma"][E].astype(np.float64)
    flux = completeness["flux"]
    k0, w = _snr_brackets(sigma, flux)
    n_pos = completeness["n_pos"][E]

    def f90_of(g, nn):
        with np.errstate(all="ignore"):
            fx = np.where(nn > 0, g / nn, np.nan)
        return flux90(np.nan_to_num(_at_flux(fx, k0, w), nan=0.0), flux)

    rng = np.random.default_rng(seed)
    c = completeness["centres"][E]
    block = (np.floor(c[:, 0] / BAR_BLOCK_ARCSEC).astype(np.int64) * 100003 + np.floor(c[:, 1] / BAR_BLOCK_ARCSEC).astype(np.int64))
    _, inv = np.unique(block, return_inverse=True)
    n_block = int(inv.max()) + 1 if inv.size else 0
    shares, units_cells, units_bad = [], [], []
    for _ in range(reps):
        a = f90_of(rng.binomial(n.astype(int), p), n)
        b = f90_of(rng.binomial(n.astype(int), p), n)
        ok = np.isfinite(a) & np.isfinite(b) & (n_pos >= MIN_INJECTIONS)
        with np.errstate(all="ignore"):
            rel = np.abs(a - b) / b
        bad = ok & (rel > C1_FLUX90_AGREE)
        shares.append(100.0 * np.mean(rel[ok] <= C1_FLUX90_AGREE) if ok.any() else np.nan)
        units_cells.append(np.bincount(inv, weights=ok, minlength=n_block))        # one unit = a block in one repetition
        units_bad.append(np.bincount(inv, weights=bad, minlength=n_block))
    s_hat = float(np.nanmean(shares))
    cells_u, bad_u = np.concatenate(units_cells) * stride, np.concatenate(units_bad) * stride     # a unit stands for `stride` times its cells
    keep = cells_u >= BAR_MIN_UNIT_CELLS * stride
    cells_u, bad_u = cells_u[keep], bad_u[keep]
    if not has_build or n_block < BAR_MIN_BLOCKS or cells_u.size == 0:
        return NAN, s_hat, NAN, n_block
    total = float(np.mean([u.sum() for u in units_cells]) * stride)      # the map's cells with a FLUX90 in both halves
    m = int(total / cells_u.mean() * 1.5) + 3
    boot = np.empty(BAR_BOOTSTRAP)
    for i in range(BAR_BOOTSTRAP):
        j = rng.integers(0, cells_u.size, m)
        k = min(int(np.searchsorted(np.cumsum(cells_u[j]), total)) + 1, m)
        boot[i] = 100.0 * (1.0 - bad_u[j[:k]].sum() / cells_u[j[:k]].sum())
    sig = float(boot.std())
    return s_hat - 2.0 * sig, s_hat, sig, n_block


def completeness_c1(band: str, completeness: dict, held: dict) -> List[CheckRow]:
    """C1: the held-out half's recovered fraction against the build half's CURVE, per flux bin (binomial error);
    `completeness` = products.read_completeness_band. Identity: `_acceptance_c1`'s values.

    Of the (cell, flux plane) on the curve's transition (either half between 0.02 and 0.98) with at least MIN_INJECTIONS
    held-out injections behind them, the share whose difference lies within the combined 1-sigma binomial error of the two
    halves (>= 60 %), within 2 sigma (>= 93 %), within 0.03 absolute (information); and the share of cells whose FLUX90 of the
    two halves agree within 10 %, against the bar derived from the map's own counts (`_flux90_halves_bar`)."""
    d = completeness
    curve_on_flux_grid = inject.curve_on_flux_grid
    flux90 = inject.flux90
    n_row, n_plane = d["curve"].shape
    Ws = _pool_matrices(d["centres"])
    fh, tot = curve_on_flux_grid(Ws, held["row"].astype(np.int64), held["plane"].astype(np.int64), held["recovered"].astype(bool),
                                 d["sigma"], d["flux"], n_row)
    fb = d["curve"].astype(float) / STORAGE_SCALE
    transition = (((fh > C1_TRANSITION[0]) & (fh < C1_TRANSITION[1])) | ((fb > C1_TRANSITION[0]) & (fb < C1_TRANSITION[1])))
    use = (tot >= MIN_INJECTIONS) & (d["n_pos"][:, None] >= MIN_INJECTIONS) & transition
    diff = np.abs(fb - fh)
    sig = np.hypot(_binomial_error(fh, tot), _binomial_error(fb, tot))
    n = int(use.sum())
    rows: List[CheckRow] = []
    if n == 0:
        rows.append(_row("C1_WITHIN_1SIGMA", band, NAN, detail="no (cell, plane) has %d held-out injections" % MIN_INJECTIONS, n=0, info=True))
    else:
        p1, p2, pa = (100.0 * (use & c).sum() / n for c in (diff <= sig, diff <= 2 * sig, diff <= C1_ABSOLUTE))
        where = "%.1f %% of all (cell, plane)" % (100.0 * n / diff.size)
        rows.append(_row("C1_WITHIN_1SIGMA", band, p1, low=C1_SIGMA1_BAR, detail="share of the transition planes within 1 sigma, " + where, n=n))
        rows.append(_row("C1_WITHIN_2SIGMA", band, p2, low=C1_SIGMA2_BAR, detail="share within 2 sigma, " + where, n=n))
        rows.append(_row("C1_WITHIN_0.03", band, pa, detail="share within 0.03 absolute (information)", n=n, info=True))
        worst = int(np.argmax(np.where(use, diff / np.maximum(sig, 1e-9), -1).ravel()))
        r, p = divmod(worst, n_plane)
        rows.append(_row("C1_WORST", band, diff[r, p] / max(sig[r, p], 1e-9),
                         detail="worst (cell, plane) in sigma: map %s cell %d plane m=%.2f: curve %.3f held-out %.3f (%d injections)"
                         % (d["map_id"][d["cell_map"][r]].decode(), d["cell_id"][r], d["mag"][p], fb[r, p], fh[r, p], tot[r, p]),
                         n=n, info=True))
    f90 = flux90(fh, d["flux"])
    ok = np.isfinite(f90) & np.isfinite(d["flux90"]) & (d["n_pos"] >= MIN_INJECTIONS)
    rel = np.abs(f90 - d["flux90"]) / d["flux90"]
    if ok.any():
        share = 100.0 * float(np.mean(rel[ok] <= C1_FLUX90_AGREE))
        bar, s_hat, sig_s, blocks = _flux90_halves_bar(d, held)
        if np.isfinite(bar):
            detail = ("share of cells whose FLUX90 of the two halves agree within 10 %%; bar = S_hat - 2 sigma = %.1f - 2 x %.1f "
                      "from the map's own counts (%d independent blocks)" % (s_hat, sig_s, blocks))
            rows.append(_row("C1_FLUX90_HALVES_SHARE", band, share, low=bar, detail=detail, n=int(ok.sum())))
        else:
            why = ("the held record carries no build-half records" if "build_row" not in held
                   else "only %d independent blocks" % blocks)
            rows.append(_row("C1_FLUX90_HALVES_SHARE", band, share, detail="no bar: %s (S_hat from the held half alone %.1f)" % (why, s_hat),
                             n=int(ok.sum()), info=True))
        rows.append(_row("C1_FLUX90_HALVES_MEDIAN", band, 100.0 * float(np.median(rel[ok])),
                         detail="median difference of FLUX90 between the halves, per cent", n=int(ok.sum()), info=True))
    else:
        rows.append(_row("C1_FLUX90_HALVES_SHARE", band, NAN, detail="no cells with FLUX90 in both halves", n=0, info=True))
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# C2 and the detection report

def completeness_c2(band: str, completeness: dict, held: dict) -> List[CheckRow]:
    """C2: the held-out (dflux / SIGMA) scatter per S/N bin (definitions.PACS_SNR_BIN_EDGES); no read is dropped (a NaN
    SIGMA is counted and reported as its own row). Identity: `_acceptance_c2`'s values on the non-NaN reads.

    The held-out half against the error built from the BUILD half alone (`held["build_errors"]`). Aperture path, per noise
    class and S/N bin, inside the flux grid: the robust scatter of held-out dflux / SIGMA_build (SCATTER_build read at the
    injection's own cell and flux, with the build's flux-proportional term held["eps_flux"] where fitted) (expected 1.02 +/- 0.05, bar 1.00 +/- 0.10 where >= 200 residuals), over the injections farther than
    BLEND_RADIUS_BEAMS from every detection (a source within it is an upper bound). Classes: quiet and noisy (a cell is noisy
    where its CELL_SIGMA exceeds twice its map's median) and, where the records carry `lnr`, gradient (the injections with
    |ln r| > 0.03). Bias above FLUX90 inside the flux grid: |median dflux / injected| <= 1 %. The same on the detected
    sources' fluxes (the flux of the PACS source the blind detection found)."""
    d = completeness
    e = held["build_errors"]
    snr_bin = inject.snr_bin
    bin_of_plane = snr_bin(inject.snr_grid())
    row = held["row"].astype(np.int64)
    plane = held["plane"].astype(np.int64)
    k = bin_of_plane[plane]
    flux = held["flux"].astype(np.float64)
    median_sigma = np.array([np.median(d["sigma"][d["cell_map"] == m]) if (d["cell_map"] == m).any() else np.nan
                             for m in range(d["map_id"].size)])
    noisy = d["sigma"] > C2_NOISY_FACTOR * median_sigma[d["cell_map"]]
    classes = [("quiet", ~noisy[row]), ("noisy", noisy[row])]
    if "lnr" in held:
        with np.errstate(invalid="ignore"):
            classes.append(("gradient", np.abs(np.asarray(held["lnr"], float)) > GRADIENT_LN_R))
    in_grid = lambda f: (f >= d["flux"].min()) & (f <= d["flux"].max())
    ap = ~held["near"].astype(bool) & in_grid(flux)
    s_build = np.full(row.size, np.nan)
    s_build[ap] = _at_injected_flux(e["scatter"], d["flux"], row[ap], flux[ap])
    # the same SIGMA the products give a detection: with the flux-proportional term where the build fitted one
    eps = inject.eps_rows(held.get("eps_flux"), d["sigma"])
    s_build[ap] = inject.flux_sigma(s_build[ap], eps[row[ap]], flux[ap])
    rows: List[CheckRow] = []
    n_nan = int(np.sum(ap & ~np.isfinite(s_build)))
    rows.append(_row("C2_NAN_SIGMA", band, n_nan, detail="held-out reads inside the flux grid whose SCATTER is NaN (not in any scatter above)",
                     n=int(ap.sum()), info=True))

    def scatter_rows(prefix, residual, mask):
        for label, cls in classes:
            for j in range(len(SNR_BIN_CENTRES)):
                sel = mask & cls & (k == j) & np.isfinite(s_build) & (s_build > 0)
                z = residual[sel] / s_build[sel]
                value = _robust(z)
                name = "C2_%s_%s_BIN%d" % (prefix, label.upper(), j)
                detail = "normalised scatter of the held-out residuals; expected 1.02 +/- 0.05"
                if sel.sum() >= C2_MIN_RESIDUALS:
                    rows.append(_row(name, band, value, C2_BAR[0], C2_BAR[1], detail, n=int(sel.sum())))
                else:
                    rows.append(_row(name, band, value, detail="fewer than %d residuals" % C2_MIN_RESIDUALS, n=int(sel.sum()), info=True))

    scatter_rows("APERTURE", held["dflux"].astype(np.float64), ap)
    f90 = d["flux90"][row]
    above = ap & np.isfinite(f90) & (flux > f90) & in_grid(flux)
    if above.sum():
        bias = 100.0 * np.median(held["dflux"][above] / flux[above])
        rows.append(_row("C2_APERTURE_BIAS_ABOVE_FLUX90", band, bias, -C2_BIAS_BAR_PERCENT, C2_BIAS_BAR_PERCENT,
                         "median residual / injected flux above FLUX90, per cent", n=int(above.sum())))
    det = held["recovered"].astype(bool) & np.isfinite(held["det_flux"])
    ddet = held["det_flux"].astype(np.float64) - flux
    scatter_rows("DETECTED", ddet, ap & det)
    above_d = above & det
    if above_d.sum():
        bias = 100.0 * np.median(ddet[above_d] / flux[above_d])
        rows.append(_row("C2_DETECTED_BIAS_ABOVE_FLUX90", band, bias, -C2_BIAS_BAR_PERCENT, C2_BIAS_BAR_PERCENT,
                         "median detected-flux residual / injected flux above FLUX90, per cent", n=int(above_d.sum())))
    return rows


def detection_report(band: str, completeness: dict, held: dict) -> List[CheckRow]:
    """DETECTIONS: recovery of the held-out injections at S/N >= 20 (bar >= 99 %), the matching radius, and the curve's
    floor (its value at the faintest flux plane, the chance of lying within the radius of an existing PACS source) by sky bin
    (`_report_detection`)."""
    d = completeness
    snr = inject.snr_grid()[held["plane"].astype(np.int64)]
    bright = snr >= DETECTION_SNR
    frac = float(held["recovered"][bright].mean()) if bright.any() else np.nan
    rows = [_row("DETECTION_RECOVERY_SNR20", band, frac, low=DETECTION_RECOVERY_BAR,
                 detail="recovered fraction of the held-out injections at S/N >= 20", n=int(bright.sum())),
            _row("DETECTION_R_MATCH", band, d["r_match"], detail="matching radius, arcsec (median over maps)", info=True)]
    floor = d["curve"][:, -1].astype(np.float64) / STORAGE_SCALE
    for j in range(SKY_BINS):
        sel = d["sky_bin"] == j
        rows.append(_row("DETECTION_FLOOR_SKYBIN%d" % j, band, np.median(floor[sel]) if sel.any() else np.nan,
                         detail="curve at the faintest plane, median over the sky bin (0 faintest sky)", n=int(sel.sum()), info=True))
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# C4, C5 against HOPS; the HPPSC checks; the chance coincidences

def _read_hops(config):
    """`(ra deg, dec deg, F70 mJy)` of the HOPS targets with a measured 70 micron flux (flag 1 or 2) and a position in the
    HOPS tables."""
    from astropy.io.votable import parse
    path = os.path.join(config.data_root, "sky", "download", "hops", HOPS_FILE)
    tables = {t.name: t.array for t in parse(path).resources[0].tables}
    pos = {}
    for name in ("table1", "table7", "table8", "table9"):
        a = tables["J/ApJS/224/5/" + name]
        for h, r, d in zip(a["HOPS"], a["RAJ2000"], a["DEJ2000"]):
            pos[int(h)] = (float(r), float(d))
    t2 = tables["J/ApJS/224/5/table2"]
    f70 = np.asarray(t2["F70"].filled(np.nan), float) * 1000.0
    keep = np.isfinite(f70) & np.array([int(h) in pos for h in t2["HOPS"]])
    return (np.array([pos[int(h)][0] for h in t2["HOPS"][keep]]), np.array([pos[int(h)][1] for h in t2["HOPS"][keep]]), f70[keep])


def _match_hops(config, region):
    """The HOPS targets matched to the region's SESNA sources within HOPS_MATCH_ARCSEC (nearest source): `(hops index,
    source row)` arrays, plus the HOPS fluxes."""
    ra, dec, f70 = _read_hops(config)
    _name, s_ra, s_dec = curated.read_positions(config, region)
    tree = cKDTree(photometry.unit_vectors(s_ra.astype(np.float64), s_dec.astype(np.float64)))
    dist, row = tree.query(photometry.unit_vectors(ra, dec), distance_upper_bound=photometry.chord(HOPS_MATCH_ARCSEC))
    found = np.isfinite(dist)
    return np.flatnonzero(found), row[found], f70


def _c4_ratio(config, region, path):
    """Median (our 70 micron flux / HOPS's) over the HOPS targets of at least C4_MIN_F70_MJY that the product at `path`
    detects (ORIGIN 1), and how many. The flux is put on HOPS's convention (no empty-sky offset subtracted): FNU plus the
    product's CELL_OFFSET where it stores one."""
    import h5py
    h, row, f70 = _match_hops(config, region)
    with h5py.File(path, "r") as f:
        flux, origin = f["FNU_P70_MJY"][:][row], f["ORIGIN_P70"][:][row]
        if "CELL_OFFSET_P70_MJY" in f:
            flux = flux + f["CELL_OFFSET_P70_MJY"][:][row]
    use = (f70[h] >= C4_MIN_F70_MJY) & (origin == photometry.ORIGIN_DETECTED) & np.isfinite(flux)
    return (float(np.median(flux[use] / f70[h][use])) if use.any() else float("nan"), int(use.sum()),
            int((f70[h] >= C4_MIN_F70_MJY).sum()))


def hops_checks(config, region: str, photometry_path: str, reference_product: Optional[str] = None) -> List[CheckRow]:
    """C4 (ours/HOPS 70 um ratio above C4_MIN_F70_MJY, and against `reference_product`) and C5 (recovery per flux bin
    C5_EDGES_MJY). Identity: `pacs_photometry.acceptance`'s values.

    C4: the median of our 70 micron flux (FNU plus CELL_OFFSET: HOPS's convention) over HOPS's for HOPS targets of at least
    200 mJy that the photometry product detects; against the earlier product's the bar is agreement within 2 %. Also the
    ORIGIN 3 counts per band (blend / confusion) and the final 70 micron ORIGIN of the bright targets. C5: the HOPS targets
    binned by their 70 micron flux: the fraction detected (ORIGIN 1) among those the band covers (one row), and the mean of
    the completeness curve at their SESNA source's cell and their flux (another), with binomial errors in the detail."""
    import h5py
    rows: List[CheckRow] = []
    ratio, n, n_all = _c4_ratio(config, region, photometry_path)
    rows.append(_row("C4_FLUX_RATIO_70", "P70", ratio, detail="median (ours + CELL_OFFSET) / HOPS at 70 micron, HOPS targets >= %g mJy detected by us (of %d)"
                     % (C4_MIN_F70_MJY, n_all), n=n, info=True))
    if reference_product is not None:
        old, n_old, _ = _c4_ratio(config, region, reference_product)
        pct = 100.0 * (ratio / old - 1.0)
        rows.append(_row("C4_VS_REFERENCE_PERCENT", "P70", pct, -C4_REFERENCE_AGREE_PERCENT, C4_REFERENCE_AGREE_PERCENT,
                         "difference of C4 from the reference product's %.4f" % old, n=n_old))
    with h5py.File(photometry_path, "r") as f:
        for b in BANDS:
            o3 = f["ORIGIN_" + b][:] == photometry.ORIGIN_BOUND
            conf = int(np.sum(o3 & f["CONFUSED_" + b][:]))
            rows.append(_row("C4_ORIGIN3_" + b, b, int(o3.sum()), detail="ORIGIN 3 sources: blend %d, confusion %d" % (int(o3.sum()) - conf, conf),
                             n=int(o3.sum()), info=True))
    h, row, f70 = _match_hops(config, region)
    comp = products.completeness_path(config, region)
    with h5py.File(photometry_path, "r") as f:
        origin = f["ORIGIN_P70"][:][row]
    bright = f70[h] >= C4_MIN_F70_MJY
    for code in (1, 3, 90, 91):
        rows.append(_row("C4_HOPS_BRIGHT_ORIGIN%d" % code, "P70", int((origin[bright] == code).sum()),
                         detail="HOPS targets >= %g mJy by final 70 micron ORIGIN" % C4_MIN_F70_MJY, n=int(bright.sum()), info=True))
    with h5py.File(comp, "r") as f:
        cell = f["SOURCE_ROW"][:][row, 0]
        curve = f["CURVE_P70"][:]
        grid_mjy = 1000.0 * f["CURVE_P70"].attrs["ZERO_POINT_JY"] * 10.0 ** (-f["MAG_GRID_P70"][:].astype(float) / 2.5)
    flux = f70[h]
    covered = (origin == photometry.ORIGIN_DETECTED) | (origin == photometry.ORIGIN_UNDETECTED)
    expected = np.full(h.size, np.nan)
    for i in np.flatnonzero(covered & (cell >= 0)):
        expected[i] = np.interp(np.log10(flux[i]), np.log10(grid_mjy[::-1]), curve[cell[i], ::-1] / float(STORAGE_SCALE))
    for lo, hi in zip(C5_EDGES_MJY[:-1], C5_EDGES_MJY[1:]):
        tag = "%g_%g" % (lo, hi)
        k = covered & np.isfinite(expected) & (flux >= lo) & (flux < hi)
        if not k.any():
            rows.append(_row("C5_DETECTED_" + tag, "P70", NAN, detail="no HOPS target covered", n=0, info=True))
            continue
        obs = float(np.mean(origin[k] == photometry.ORIGIN_DETECTED))
        pred = float(np.mean(expected[k]))
        err = float(np.sqrt(np.sum(expected[k] * (1 - expected[k]))) / k.sum())
        rows.append(_row("C5_DETECTED_" + tag, "P70", obs, detail="detected fraction (ORIGIN 1) of HOPS targets %g-%g mJy, +- %.3f" % (
            lo, hi, np.sqrt(max(obs * (1 - obs), 1e-9) / k.sum())), n=int(k.sum()), info=True))
        rows.append(_row("C5_PREDICTED_" + tag, "P70", pred, detail="the completeness curve's mean at their cells and fluxes, +- %.3f" % err,
                         n=int(k.sum()), info=True))
    return rows


def _read_hppsc(config, band):
    """HPPSC's band list as arrays: ra, dec (deg), flux (mJy), snr (structure-noise S/N), edge flag, fwhm (arcsec: the
    geometric mean of its fitted widths, NaN where it gives none); and the observation table's obsids for the band."""
    import csv
    from sesnaimpute.sky.download.hppsc.build import download_dir
    name, colour = HPPSC_FILES[band]
    cols = {k: [] for k in ("ra", "dec", "flux", "snr", "edgeflag")}
    widths = {"fwhmxfit": [], "fwhmyfit": []}
    with open(os.path.join(download_dir(config), name)) as fh:
        for r in csv.DictReader(fh):
            for k in cols:
                cols[k].append(r[k])
            for k in ("fwhmxfit", "fwhmyfit"):
                widths[k].append(float(r[k]) if r[k] not in ("", None) else np.nan)
    out = {k: np.asarray(v, float) for k, v in cols.items() if k != "edgeflag"}
    out["edge"] = np.asarray(cols["edgeflag"]) == "t"
    out["fwhm"] = np.sqrt(np.asarray(widths["fwhmxfit"]) * np.asarray(widths["fwhmyfit"]))
    with open(os.path.join(download_dir(config), "ppsc_obstbl.csv")) as fh:
        out["obsids"] = {int(r["obsid"]) for r in csv.DictReader(fh) if r["band"] == colour}
    return out


def input_checks(config, region: str, band: str) -> List[CheckRow]:
    """What a build would otherwise find out at FINISH, checked at stage psf (the retired preflight's input checks): the
    band's HPPSC list and observation table read and non-empty (INPUT_HPPSC), the HOPS table read (INPUT_HOPS: the
    targets matched in the region, none outside Orion), the region's SESNA positions (INPUT_POSITIONS > 0), and the
    completeness writer on empty bands (INPUT_WRITERS: every dataset name registered). A failure is a FAIL row (the
    stage stops). Writes only a temporary file."""
    import tempfile
    rows = []

    def guarded(name, fn):
        try:
            value, detail = fn()
            rows.append(_row(name, band, value, low=1.0, detail=detail))
        except Exception as e:                    # noqa: BLE001 -- the failure is the row
            rows.append(_row(name, band, 0.0, low=1.0, detail="FAILED: %r" % (e,)))

    def hppsc():
        h = _read_hppsc(config, band)
        return float(min(h["ra"].size, len(h["obsids"]))), "%d HPPSC sources, %d observations" % (h["ra"].size, len(h["obsids"]))

    def hops():
        n = _match_hops(config, region)[0].size
        return 1.0, "HOPS table read; %d targets matched in the region" % n

    def positions():
        _name, ra, _dec = curated.read_positions(config, region)
        return float(len(ra)), "%d SESNA positions" % len(ra)

    def writers():
        mag, _fgrid = inject.flux_grid()
        bands = {b: inject.empty_band(5, mag.size) for b in BANDS}
        with tempfile.TemporaryDirectory(dir=os.path.join(config.data_root, "sky", "derived")) as tmp:
            products.write_completeness(os.path.join(tmp, "c.hdf5"), region, mag, bands)
        return 1.0, "write_completeness on empty bands"

    for name, fn in (("INPUT_HPPSC", hppsc), ("INPUT_HOPS", hops), ("INPUT_POSITIONS", positions),
                     ("INPUT_WRITERS", writers)):
        guarded(name, fn)
    return rows


def _mosaic_psf(config, region, band, box):
    """The band mosaic's PSF without its tables (weights and beams only)."""
    return Mosaic(config, region, band, box, psf_tables=False).psf


def hppsc_checks(config, region: str, sources_path: str, photometry_path: str,
                 box: Optional[Sequence[float]] = None) -> List[CheckRow]:
    """HPPSC recovery (>= HPPSC_RECOVERY_BAR above HPPSC_RECOVERY_SNR), flux ratio (information), offsets, coverage
    (`pacs_photometry.hppsc_checks`).

    The external check of the region's PACS source list against HPPSC, per band (HPPSC enters no product): (1) how many of
    our maps HPPSC was built from; (2) the offsets of common sources, HPPSC to our nearest detection in the band within one
    beam, at that band's own detection position, as information; (3) the median flux ratio ours / HPPSC over common sources
    (within one beam, HPPSC S/N >= 10), ours on HPPSC's convention (FNU plus the cell offset), as information; (4) of the HPPSC
    sources inside our footprint with flux at least our FLUX90 at their position, fitted width at most 1.5 beams and HPPSC
    S/N >= 10, the fraction our detection finds in the band within one beam, bar 0.90 within the binomial error at the bar
    (the same over all of them and over the point-like ones as information). `box`: a development slice's mosaics."""
    import h5py
    from sesnaimpute.catalog.completeness import _PacsBand
    src = sources_path
    with h5py.File(src, "r") as f:
        mine = {b: dict(det=f["DETECTED_" + b][:], fnu=f["FNU_%s_MJY" % b][:], off=f["CELL_OFFSET_%s_MJY" % b][:],
                        ra=f["RA_%s_DEG" % b][:], dec=f["DEC_%s_DEG" % b][:]) for b in BANDS}
    rows: List[CheckRow] = []
    comp = products.completeness_path(config, region)
    with h5py.File(comp, "r") as fc:
        for b in BANDS:
            if "CELL_ID_" + b not in fc or fc["CELL_ID_" + b].shape[0] == 0:
                rows.append(_row("HPPSC_COVERAGE", b, NAN, detail="the band has no cells", n=0, info=True))
                continue
            h = _read_hppsc(config, b)
            ours = {int(o) for m in fc["MAP_OBSID_" + b][:] for o in m.decode().split(",") if o}
            rows.append(_row("HPPSC_COVERAGE", b, len(ours & h["obsids"]),
                             detail="observations of our mosaic that are among HPPSC's (its Table 8), of %d" % len(ours), n=len(ours), info=True))
            cra, cde = fc["CELL_RA_" + b][:], fc["CELL_DEC_" + b][:]
            inbox = ((h["ra"] > cra.min() - 0.1) & (h["ra"] < cra.max() + 0.1) & (h["dec"] > cde.min() - 0.1)
                     & (h["dec"] < cde.max() + 0.1))
            hra, hde, hflux, hsnr = h["ra"][inbox], h["dec"][inbox], h["flux"][inbox], h["snr"][inbox]
            row = _PacsBand(fc, b).rows(hra, hde)
            inside = row >= 0
            flux90 = np.where(inside, fc["FLUX90_%s_MJY" % b][:][np.maximum(row, 0)], np.nan)
            det = np.flatnonzero(mine[b]["det"])
            pra, pde = mine[b]["ra"], mine[b]["dec"]                  # each band's own detection position
            pu = photometry.unit_vectors(np.nan_to_num(pra), np.nan_to_num(pde))
            if det.size == 0 or not inside.any():
                rows.append(_row("HPPSC_RECOVERY_POINT_SNR", b, NAN, detail="%d HPPSC sources inside our footprint, %d detections of ours: no comparison"
                                 % (int(inside.sum()), det.size), n=int(inside.sum()), info=True))
                continue
            tree = cKDTree(pu[det])
            hu = photometry.unit_vectors(hra, hde)
            psf = _mosaic_psf(config, region, b, box)
            beam = psf.beam_at_sky(hra, hde)                          # the band's local beam at each HPPSC source
            dist, j = tree.query(hu, distance_upper_bound=photometry.chord(psf.sub_beams.max()))
            near = np.isfinite(dist) & inside & (np.nan_to_num(dist, posinf=9.0) < photometry.chord(beam))
            k = det[np.minimum(j, det.size - 1)]
            sep = np.where(near, np.degrees(2.0 * np.arcsin(np.clip(np.nan_to_num(dist, posinf=0.0) / 2.0, 0, 1))) * 3600.0, np.nan)
            cosd = np.cos(np.radians(hde))
            east = (pra[k] - hra) * cosd * 3600.0
            north = (pde[k] - hde) * 3600.0
            if near.any():
                rows.append(_row("HPPSC_OFFSET_MEDIAN_SEP", b, np.median(sep[near]),
                                 detail="median separation ours - HPPSC of common sources within one local beam (median %.1f arcsec), arcsec; 68th %.2f, 90th %.2f; "
                                 "median east %+.2f north %+.2f; within %.1f arcsec %.3f"
                                 % (np.median(beam[near]), np.percentile(sep[near], 68), np.percentile(sep[near], 90), np.median(east[near]),
                                    np.median(north[near]), photometry.MATCH_RADIUS_ARCSEC,
                                    float(np.mean(sep[near] <= photometry.MATCH_RADIUS_ARCSEC))), n=int(near.sum()), info=True))
            common = near
            ours_flux = mine[b]["fnu"][k] + np.nan_to_num(mine[b]["off"][k])
            use = common & (hsnr >= HPPSC_FLUX_SNR) & np.isfinite(ours_flux) & (hflux > 0)
            if use.any():
                rows.append(_row("HPPSC_FLUX_RATIO", b, np.median(ours_flux[use] / hflux[use]),
                                 detail="median ours (FNU + cell offset) / HPPSC over common sources with HPPSC S/N >= %g (information)" % HPPSC_FLUX_SNR,
                                 n=int(use.sum()), info=True))
            else:
                rows.append(_row("HPPSC_FLUX_RATIO", b, NAN, detail="no common sources with HPPSC S/N >= %g" % HPPSC_FLUX_SNR, n=0, info=True))
            above = inside & np.isfinite(flux90) & (hflux >= flux90)
            point = h["fwhm"][inbox] <= HPPSC_POINT_BEAMS * beam
            for name, pool, ruled in (("HPPSC_RECOVERY_ALL", above, False), ("HPPSC_RECOVERY_POINT", above & point, False),
                                      ("HPPSC_RECOVERY_POINT_SNR", above & point & (hsnr >= HPPSC_RECOVERY_SNR), True)):
                if not pool.any():
                    rows.append(_row(name, b, NAN, detail="no HPPSC source inside our footprint at or above our FLUX90", n=0, info=True))
                    continue
                n_pool = int(pool.sum())
                frac = float(np.mean(common[pool]))
                err = float(np.sqrt(HPPSC_RECOVERY_BAR * (1 - HPPSC_RECOVERY_BAR) / n_pool))
                detail = ("recovered fraction of the HPPSC sources inside our footprint at or above our FLUX90 there; binomial error at the bar %.3f; "
                          "HPPSC edge-flagged among them %d" % (err, int(np.sum(h["edge"][inbox][pool]))))
                if ruled:
                    rows.append(_row(name, b, frac, low=HPPSC_RECOVERY_BAR - err, detail=detail, n=n_pool))
                else:
                    rows.append(_row(name, b, frac, detail=detail + " (information)", n=n_pool, info=True))
    return rows


def chance_check(counts: dict) -> List[CheckRow]:
    """CHANCE (information): expected chance coincidences of the association (was the EXPECTED_CHANCE_COINCIDENCES
    attribute and a print): the local SESNA density x pi (2 arcsec)^2, summed over the region's PACS sources
    (`counts["chance"]`, from photometry.cross_match)."""
    return [_row("CHANCE_COINCIDENCES", "", counts["chance"], detail="expected chance coincidences of the SESNA-PACS association over the region",
                 info=True)]


def real_map_rows(band: str, stats: dict) -> List[CheckRow]:
    """REAL_MAP_MEAN_RATIO (information, one row per half): the mean of (aperture flux after injecting F minus before) / F
    at inject.REAL_CHECK_POSITIONS empty-sky positions of the real map, F = 10 x DETECTION_SNR x the median CELL_SIGMA
    (inject._real_map_ratios, run by the designated injection piece; the band's statistics carry it as `check`).
    Identity: `pacs_completeness._real_check`'s printed means."""
    check = (stats or {}).get("check")
    if check is None:
        return []
    return [_row("REAL_MAP_MEAN_RATIO", band, float(v), detail="half %d: injected-minus-before / F on the real map" % h,
                 info=True) for h, v in enumerate(check)]


def region_checks(config, region: str, bands: Dict[str, BandCompleteness], box: Optional[Sequence[float]] = None,
                  reference_product: Optional[str] = None) -> List[CheckRow]:
    """Every check after FINISH, in one list (C1, C2, detections, C4/C5, HPPSC, chance), for products.write_checks.
    Reads the three written products of the region; `bands[b].held` is the held-out half and its build-only error."""
    import h5py
    rows: List[CheckRow] = []
    comp = products.completeness_path(config, region)
    for b in BANDS:
        held = bands[b].held
        with h5py.File(comp, "r") as f:
            empty = f["CURVE_" + b].shape[0] == 0
        if held is None or empty:
            continue
        c = products.read_completeness_band(comp, b)
        rows += completeness_c1(b, c, held) + completeness_c2(b, c, held) + detection_report(b, c, held)
        rows += real_map_rows(b, bands[b].stats)
    phot = products.photometry_path(config, region)
    rows += hops_checks(config, region, phot, reference_product)
    rows += hppsc_checks(config, region, products.sources_path(config, region), phot, box)
    with h5py.File(phot, "r") as f:
        if "EXPECTED_CHANCE_COINCIDENCES" in f["ORIGIN_P70"].attrs:
            rows += chance_check({"chance": float(f["ORIGIN_P70"].attrs["EXPECTED_CHANCE_COINCIDENCES"])})
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# the build-time checks: the real unit source, the zero image, the normalisation

def _representative_maps(config, meta: dict) -> list:
    """Per sub-group: (index record, psf.json map entry, input WCS, components) of the map the PSF build used."""
    from sesnaimpute.sky.download.herschel_pacs.index import read_index
    recs = {r.path: r for r in read_index(config, meta["region"])}
    out = []
    for i in meta["rep"]:
        q = meta["maps"][i]
        rec = recs[q["path"]]
        out.append((rec, q, grid.celestial_wcs(rec.wcs_header), psf.published_components(config, rec)))
    return out


def _aperture_sum(img, x, y, radii_px):
    """The aperture sum before sky (the aperture sum the divisor is the fraction of) at pixel (x, y)."""
    r_ap, r_in, r_out = radii_px
    f, s, _sd, area, cov = mosaic._aperture_core(np.ascontiguousarray(img, np.float32), np.ones(img.shape, np.float32),
                                                 np.array([x]), np.array([y]), r_ap, r_in, r_out, grid.APERTURE_SUBPIXELS)
    if not cov[0]:
        raise ValueError("the aperture of the synthetic source is not covered")
    return float(f[0] + s[0] * area[0])


_DISC: dict = {}


def _template_in_aperture(comps, in_wcs, r_ap, step=0.4, nbox=7):
    """The release template, smoothed by the input pixel's box, integrated inside the aperture radius (arcsec), unit flux:
    on a fine grid of cells `step` arcsec wide with an anti-aliased edge, the box average over nbox x nbox offsets across the
    pixel (along its own, rotated axes). The PSF build holds this (the box-only table) to the release EEF; the value here is
    position-free, so the input grid's sampling of a smooth beam does not enter it."""
    import hashlib
    key = (r_ap, tuple(np.round(in_wcs.pixel_scale_matrix.ravel() * 3600.0, 9)),
           tuple((float(c.position_angle_deg), hashlib.sha1(np.ascontiguousarray(c.image).tobytes()).hexdigest()) for c in comps))
    if key not in _DISC:
        m = in_wcs.pixel_scale_matrix * 3600.0                                  # sky arcsec (east, north) per input pixel
        o = np.arange(-int(r_ap / step) - 2, int(r_ap / step) + 3) * step
        xx, yy = np.meshgrid(o, o)
        w = np.clip((r_ap - np.hypot(xx, yy)) / step + 0.5, 0.0, 1.0)
        c = (np.arange(nbox) + 0.5) / nbox - 0.5
        total = 0.0
        for u in c:
            for v in c:
                de, dn = m @ np.array([u, v])
                total += psf.template_density(comps, xx.ravel() + de, yy.ravel() + dn).reshape(xx.shape)
        _DISC[key] = float((total / nbox ** 2 * w).sum() * step ** 2)
    return _DISC[key]


def _unit_source(rep, out_wcs, band, xo, yo, half_out, phases=PHASES):
    """The real unit source at output pixel (xo, yo) (zero-origin), averaged over the input grid's sub-pixel `phases` (the
    PSF build's tables are the phase average; one phase of a 3.2 arcsec input grid sampling an 11 arcsec beam scatters by
    +-0.2 % in an aperture sum): returns (A_real, A_template, block, x0, y0), the block (2 half_out + 1)^2 of the resampled
    image (flux fraction per output pixel) with lower-left pixel (x0, y0)."""
    rec, _q, in_wcs, comps = rep
    in_area = mosaic.pixel_area_arcsec2(in_wcs)
    out_area = mosaic.pixel_area_arcsec2(out_wcs)
    pix_in, pix_out = np.sqrt(in_area), np.sqrt(out_area)
    r_ap, r_in, r_out = APERTURES[band]
    half_in = int(np.ceil(1.3 * half_out * pix_out / pix_in)) + 4
    ra0, dec0 = out_wcs.all_pix2world(xo, yo, 0)
    px0, py0 = (float(v) for v in in_wcs.all_world2pix(ra0, dec0, 0))
    ix0, iy0 = int(round(px0)) - half_in, int(round(py0)) - half_in
    n = 2 * half_in + 1
    S = SUBSAMPLES
    # one lattice of sub-samples, 1/S input pixel apart, serves every phase: a phase of k/3 pixel is a shift of k S/3 cells
    steps = [(int(round(fx * S)), int(round(fy * S))) for fx, fy in phases]
    if any(abs(a - fx * S) > 1e-9 or abs(b - fy * S) > 1e-9 for (a, b), (fx, fy) in zip(steps, phases)):
        raise ValueError("the phases must be whole multiples of 1/SUBSAMPLES input pixel")
    m = (n + 1) * S
    lat = (np.arange(m) + 0.5) / S - 0.5
    gx, gy = np.meshgrid(ix0 + lat, iy0 + lat)
    ra, dec = in_wcs.all_pix2world(gx.ravel(), gy.ravel(), 0)
    # offsets on the sky from the source (arcsec; the flat form is exact to 1e-7 inside the template's +-100 arcsec)
    east = (((ra - ra0 + 180.0) % 360.0) - 180.0) * np.cos(np.radians((dec + dec0) / 2)) * 3600.0
    north = (dec - dec0) * 3600.0
    density = psf.template_density(comps, east, north).reshape(m, m)
    cx, cy = int(np.rint(xo)), int(np.rint(yo))
    x0, y0 = cx - half_out, cy - half_out
    block = 0.0
    for (a, b), (fx, fy) in zip(steps, phases):
        img = density[b:b + n * S, a:a + n * S].reshape(n, S, n, S).mean(axis=(1, 3)) * in_area   # flux fraction per input pixel
        win = in_wcs.deepcopy()
        win.wcs.crpix = win.wcs.crpix - np.array([ix0 + fx, iy0 + fy])
        blk = _resample(img, win, out_wcs, out_area, x0, x0 + 2 * half_out + 1, y0, y0 + 2 * half_out + 1)
        if blk is None or not blk["valid"].all():
            raise ValueError("the synthetic input does not cover the block")
        block = block + blk["image"] / len(phases)
    a_real = _aperture_sum(block, xo - x0, yo - y0, (r_ap / pix_out, r_in / pix_out, r_out / pix_out))
    return a_real, _template_in_aperture(comps, in_wcs, r_ap), block, x0, y0


def _expected_enclosed(config, m, x, y, mix):
    """The mosaic's expected pre-sky aperture sum of a unit injected source at the output positions (x, y) with sub-group
    fractions `mix` (n, n_sub): sum_k f_k EEF_k A_real_k / A_direct_k, every term recomputed on the synthetic real source (no
    number from the PSF build). Returns (expected (n,), worst |A_direct/EEF - 1|)."""
    meta = m.psf.meta
    reps = _representative_maps(config, meta)
    eef = [psf.aperture_eef(config, r[0]) for r in reps]
    half_out = int(np.ceil(APERTURES[m.band][2] / m.pixel_arcsec)) + 4
    out = np.zeros(len(x))
    worst = 0.0
    for i in range(len(x)):
        for k in np.flatnonzero(mix[i] > 0):
            a_real, a_direct, *_ = _unit_source(reps[k], m.grid.wcs, m.band, x[i], y[i], half_out)
            out[i] += mix[i, k] * eef[k] * a_real / a_direct
            worst = max(worst, abs(a_direct / eef[k] - 1))
    return out, worst


def _check_positions(m, meta, rep_entry, margin):
    """The check positions of one sub-group (output pixels, zero-origin, inside the mosaic by `margin`): 3 x 3 nodes, 4
    mid-cells between them, 4 at 80 % of the half-extent, and 10/20/30 arcsec inside and outside the footprint edge of the
    representative map (the first of its four edges that lies inside the mosaic), at a fixed sub-pixel phase."""
    ny, nx = m.psf.shape
    n = m.grid.psf_node_count()
    sx = np.linspace(-.5, nx - .5, n)
    sy = np.linspace(-.5, ny - .5, n)
    pts = [(a, b, "node") for b in sy for a in sx]
    mx, my = (sx[:-1] + sx[1:]) / 2, (sy[:-1] + sy[1:]) / 2
    pts += [(a, b, "mid") for b in (my[0], my[-1]) for a in (mx[0], mx[-1])]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    pts += [(cx + s * 0.8 * nx / 2, cy + t * 0.8 * ny / 2, "outer") for s in (-1, 1) for t in (-1, 1)]
    x0, x1, y0, y1 = meta["maps"][rep_entry]["bbox"]
    pix = float(np.mean(np.abs(np.diag(m.grid.wcs.pixel_scale_matrix)))) * 3600.0
    for edge in ("left", "right", "bottom", "top"):
        c = {"left": x0, "right": x1, "bottom": y0, "top": y1}[edge]
        horizontal = edge in ("left", "right")
        lim = nx if horizontal else ny
        if margin < c < lim - 1 - margin:
            mid = ((y0 + y1) / 2 if horizontal else (x0 + x1) / 2)
            for d in RIM_ARCSEC:
                for sgn in (-1, 1):
                    along = c + sgn * d / pix
                    pts.append((along, mid, "rim") if horizontal else (mid, along, "rim"))
            break
    out = []
    for a, b, kind in pts:
        a, b = float(np.clip(a + 0.31, margin, nx - 1 - margin)), float(np.clip(b - 0.22, margin, ny - 1 - margin))
        out.append((a, b, kind))
    return out


def _real_sources(rep, out_wcs, band, positions, half_out, h):
    """`_unit_source` at each of one sub-group's check positions: (A_real, A_real averaged over SOURCE_PHASES, A_direct, the
    (2h+1)^2 patch around the source's nearest pixel) per position (a worker's unit of `normalisation_check`)."""
    out = []
    for x, y, _kind in positions:
        a_real, a_direct, block, _bx, _by = _unit_source(rep, out_wcs, band, x, y, half_out)
        a_mean = float(np.mean([_unit_source(rep, out_wcs, band, x + fx, y + fy, half_out)[0] for fx, fy in SOURCE_PHASES]))
        out.append((a_real, a_mean, a_direct, block[half_out - h:half_out + h + 1, half_out - h:half_out + h + 1]))
    return out


def _injected_window(m: Mosaic, x: float, y: float, flux: float, reach: int, mixture=None, error=None) -> Window:
    """A private window of (2 reach + 1)^2 pixels around the pixel nearest (x, y): a zero image (error: ones, or `error`)
    holding an injected source of `flux` mJy at (x, y) (the mixture `mixture` of sub-group fractions in place of the local
    one)."""
    cx, cy = int(np.rint(x)), int(np.rint(y))
    side = 2 * reach + 1
    win = Window(cy - reach, cx - reach, np.zeros((side, side), np.float32),
                 np.ones((side, side), np.float32) if error is None else np.array(error, np.float32))
    px, py, patches = m.template_patches(np.array([x]), np.array([y]), np.full(1, flux), mixture)
    mosaic.paste(win, px, py, patches)
    return win


def zero_image_check(config, m: Mosaic) -> List[CheckRow]:
    """ZERO_IMAGE: the aperture sum of injected unit sources on a zero image over (injected x EEF) at
    ZERO_CHECK_POSITIONS positions; bar ZERO_CHECK_TOLERANCE; FAIL stops the build. Works on a private zero Window per
    position (no whole-grid array, D sec 1). Identity: the numbers `pacs_completeness.self_check` prints.

    ZERO_CHECK_FLUX_MJY is injected at empty-sky positions (the whole aperture and annulus on valid data, no two closer than
    the injected patch's half-width plus the annulus's outer radius), the aperture SUM with no sky subtraction divided by the
    injected flux times the mosaic's encircled-energy fraction at the aperture radius (total-flux normalisation; the local
    mixture of the input maps' REAL unit sources, resampled: absolute, independent of the PSF build's own fractions): every
    position within 0.5 %. Also the template against the release EEF (1.5 %) and, as information, the share of the aperture
    flux the sky term of the catalogue's own measurement removes on a zero image."""
    rng = np.random.default_rng(0)
    ny, nx = tuple(m.grid.shape)
    h = m.injection_half_width()
    margin = h + 2
    r_ap, r_in, r_out = (r / m.pixel_arcsec for r in APERTURES[m.band])
    reach = int(np.ceil(r_out)) + 2
    separation = h + r_out
    xs, ys = [], []
    for _ in range(100000):
        x, y = rng.uniform(margin, nx - margin), rng.uniform(margin, ny - margin)
        iy, ix = int(round(y)), int(round(x))
        err = m.window(iy - reach, iy + reach + 1, ix - reach, ix + reach + 1).error
        if np.isfinite(err).all() and all(np.hypot(x - a, y - b) > separation for a, b in zip(xs, ys)):
            xs.append(x)
            ys.append(y)
            if len(xs) == ZERO_CHECK_POSITIONS:
                break
    x, y = np.array(xs), np.array(ys)
    R = max(reach, h)
    f = np.zeros(x.size)
    sky = np.zeros(x.size)
    area = np.zeros(x.size)
    cov = np.zeros(x.size, bool)
    for i in range(x.size):
        cx, cy = int(np.rint(x[i])), int(np.rint(y[i]))
        err = m.window(cy - R, cy + R + 1, cx - R, cx + R + 1).error
        win = _injected_window(m, x[i], y[i], ZERO_CHECK_FLUX_MJY, R, error=err)
        fi, si, _sd, ai, ci = mosaic._aperture_core(win.image, win.error, np.array([x[i] - win.x0]), np.array([y[i] - win.y0]), r_ap, r_in, r_out, grid.APERTURE_SUBPIXELS)
        f[i], sky[i], area[i], cov[i] = fi[0], si[0], ai[0], ci[0]
    eef, tie = _expected_enclosed(config, m, x, y, m.psf.weights(x, y))
    expected = ZERO_CHECK_FLUX_MJY * eef
    ratio = (f + sky * area)[cov] / expected[cov]
    sky_share = (sky * area)[cov] / expected[cov]
    worst = float(np.abs(ratio - 1).max()) if ratio.size else NAN
    return [_row("ZERO_IMAGE_WORST", m.band, worst, high=ZERO_CHECK_TOLERANCE,
                 detail="worst |aperture sum / (injected x EEF_total(r_ap)) - 1| on a zero image; min %.4f max %.4f; EEF %.4f on average"
                 % (ratio.min() if ratio.size else NAN, ratio.max() if ratio.size else NAN, float(np.mean(eef))), n=int(ratio.size)),
            _row("ZERO_IMAGE_TEMPLATE", m.band, tie, high=RELEASE_TOLERANCE,
                 detail="worst |A_direct / release EEF - 1| of the synthetic real source (template, pixel area or EEF file)", n=int(x.size)),
            _row("ZERO_IMAGE_SKY_SHARE", m.band, 100.0 * float(sky_share.mean()) if ratio.size else NAN,
                 detail="share (per cent) of the aperture flux the catalogue's sky term removes on a zero image", n=int(ratio.size), info=True)]


def normalisation_check(config, m: Mosaic, n_jobs: Optional[int] = None) -> List[CheckRow]:
    """NORMALISATION (real/divisor, injected/real, patch sums, template/release EEF) per sub-group at the check
    positions; bars TIE 0.003, SUM 0.01 (rims only where the sub-group contributes), RELEASE 0.015. Identity: the values
    `pacs_checks.normalisation_check` computes (its `.rows`).

    Every sub-group of the band's mosaic, at its check positions: (a) the real unit source's aperture sum, averaged over
    SOURCE_PHASES (the divisor is a phase average), taken at the release EEF, over the production divisor
    (`MosaicPSF.enclosed_at` with the weights one-hot on the sub-group) within TIE_TOLERANCE of 1; (b) the same through the
    production injection (one-hot) against the real one within TIE_TOLERANCE; (c) the pixel sums of the same patch within
    SUM_TOLERANCE; (d) A_direct / EEF within RELEASE_TOLERANCE. A sub-group is tested where it contributes to the mosaic
    (owner ruling 2026-10-30): every position, rims included, only where its own maps carry weight (`map_weights`). One row per
    statistic with the worst |deviation|; a failing statistic lists its worst positions in the detail."""
    from joblib import Parallel, delayed
    psf = m.psf
    meta = psf.meta
    reps = _representative_maps(config, meta)
    h = m.injection_half_width()
    half_out = max(h, int(np.ceil(APERTURES[m.band][2] / m.pixel_arcsec)) + 4)
    r_px = tuple(r / m.pixel_arcsec for r in APERTURES[m.band])
    worst = dict(divisor=0.0, injected=0.0, sums=0.0, tie=0.0)
    where: dict = {}
    bad: dict = {}
    n = 0
    positions = [_check_positions(m, meta, meta["rep"][k], h + 2) for k in range(len(reps))]
    for k in range(len(reps)):
        pts = positions[k]
        x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        w = psf.map_weights(x, y)[:, k] if pts else np.zeros(0)
        positions[k] = [p for p, wk in zip(pts, w) if wk > 0]
    out_wcs = m.grid.wcs
    real = Parallel(n_jobs=n_jobs or 1)(delayed(_real_sources)(rep, out_wcs, m.band, positions[k], half_out, h)
                                        for k, rep in enumerate(reps))
    reach = int(np.ceil(r_px[2])) + 3
    R = max(reach, h)
    tol = dict(divisor=TIE_TOLERANCE, injected=TIE_TOLERANCE, sums=SUM_TOLERANCE, tie=RELEASE_TOLERANCE)
    for k, rep in enumerate(reps):
        eef = psf.aperture_eef(config, rep[0])
        one = np.zeros((1, psf.n_sub))
        one[0, k] = 1.0
        for (x, y, kind), (a_real, a_mean, a_direct, real_patch) in zip(positions[k], real[k]):
            norm = eef / a_direct                    # the real source at the release EEF (the template's own shortfall is `tie`)
            a_real, a_mean, real_patch = a_real * norm, a_mean * norm, real_patch * norm
            divisor = float(psf.enclosed_at(np.array([x]), np.array([y]), one)[0])
            win = _injected_window(m, x, y, FLUX_MJY, R, mixture=one)
            cx, cy = int(np.rint(x)), int(np.rint(y))
            a_inj = _aperture_sum(win.image[R - reach:R + reach + 1, R - reach:R + reach + 1], x - cx + reach, y - cy + reach, r_px) / FLUX_MJY
            patch = win.image[R - h:R + h + 1, R - h:R + h + 1] / FLUX_MJY
            dev = dict(divisor=a_mean / divisor - 1, injected=a_inj / a_real - 1, sums=patch.sum() / real_patch.sum() - 1,
                       tie=a_direct / eef - 1)
            n += 1
            for key, v in dev.items():
                if abs(v) >= worst[key]:
                    worst[key] = abs(v)
                    where[key] = "sub-group %d %s (%.1f, %.1f) %+.4f" % (k, kind, x, y, v)
                if abs(v) > tol[key]:
                    bad.setdefault(key, []).append("sub-group %d %s (%.1f, %.1f) %+.4f" % (k, kind, x, y, v))
    names = dict(divisor="NORM_DIVISOR", injected="NORM_INJECTED", sums="NORM_PATCH_SUMS", tie="NORM_TEMPLATE_VS_EEF")
    what = dict(divisor="real unit source (at the release EEF) / production divisor - 1", injected="production injection / real unit source - 1",
                sums="pixel sums of the injected patch / the real unit source's - 1", tie="release template / release EEF - 1")
    rows = []
    for key in ("divisor", "injected", "sums", "tie"):
        detail = "worst |%s| over %d sub-groups; at %s" % (what[key], len(reps), where.get(key, "-"))
        if key in bad:
            detail += "; %d positions outside the bar, e.g. %s" % (len(bad[key]), "; ".join(bad[key][:4]))
        rows.append(_row(names[key], m.band, worst[key], high=tol[key], detail=detail, n=n))
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# a synthetic input map through the real resampler

class _ArrayMap(psf.InputMap):
    """An input map whose layers are caller arrays (image, error 1e-3, coverage 1) on `wcs`: the zero-image check's
    synthetic unit source through `mosaic.resample_window`."""

    def __init__(self, image: np.ndarray, wcs):
        super().__init__(record=SimpleNamespace(shape=image.shape), wcs=wcs, maps_root="")
        self._layers = (image, np.full(image.shape, 1e-3), np.ones(image.shape))

    def layers(self, rows: slice, cols: slice):
        return tuple(np.array(a[rows, cols], dtype=np.float64) for a in self._layers)


def _resample(img, in_wcs, out_wcs, out_area, x0, x1, y0, y1):
    """The real resampler applied to a synthetic input image (`out_area` is the output grid's reference pixel area,
    which resample_window takes from the grid)."""
    return mosaic.resample_window(_ArrayMap(img, in_wcs), SimpleNamespace(wcs=out_wcs), y0, y1, x0, x1, with_smooth=False)
