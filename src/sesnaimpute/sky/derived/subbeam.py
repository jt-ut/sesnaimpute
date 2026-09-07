"""The sub-beam term of the column kernel (SPEC_PRIORS.md section 1.2):
how much the true column moves within one Herschel beam.

The kernel needs the dispersion of true column within the beam, growing
with column, at the arm's stated beam. An HGBS map only measures a
TWO-SCALE increment directly -- the width of `ln A_L1 - ln A_L2` between
two ladder points -- since the finest thing on the map is its own 36.3
arcsec beam. The ABSOLUTE pencil-to-beam width is a MODEL EXTRAPOLATION
below that: each map is decimated to a common 12 arcsec grid and smoothed
through a 16-rung beam ladder (NaN/footprint-normalised Gaussian
convolution, so the map edge shows up as low weight rather than leaking
column into the measurement); the two-scale widths fit a `P(k) ~ k^-beta`
power-law model per region, and the absolute width follows from the fit.
Validated by refitting on coarse scales only (>=108 arcsec) and
predicting the held-out fine pairs; a block bootstrap over 512 arcsec
tiles gives the fit's sampling uncertainty.

Also derives the three scalars `prior/kernel.py`'s evaluator needs to
place the sub-beam width at an arbitrary beam: the completion factor
`W_abs(L)/sd_two_scale(L)` at the beams the conditional tables are
tabulated at (108, 301.8, 821 arcsec -- Planck's beam and the extinction-
profile grid's nside-256 pixel), the beam-rescaling exponent `(beta-2)/2`,
and an offset exponent `p` through the two measured median offsets at 302
and 821 arcsec via `offset(L) = -c*L**p` (only `p` is stored; `c` cancels).

One map job's peak resident memory is a fixed per-worker baseline (the
interpreter and its imported libraries) plus a measured cost per native
pixel -- calibrated by profiling the largest map (orionB, 84,722,900
pixels) alone, with the map read via a bounded-memory streaming
decompression (never holding the compressed and fully-decompressed bytes
at once) rather than astropy's whole-file gzip read, and the beam-ladder
convolution kept in float32 throughout rather than upcast to float64. The
map-level joblib pool is capped to
`floor(6 GiB / (300 MiB + largest_map_pixels * 17 B))` workers -- orionB
bounds this to 3 workers by formula.
"""

import gzip
import os
import tempfile

import h5py
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from scipy import fft as sfft
from scipy import optimize
from scipy.special import ndtr

from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.sky.derived.herschel_column import _map_header
from sesnaimpute.sky.download.herschel_hgbs.build import _FILES as HGBS_FILES

#: N(H2) -> A_K, ZGR23 (matches sesnaimpute.sky.derived.column's own use).
AK_PER_NH2 = 1.12e-22
#: Below this a map pixel is padding or noise, not measured sky.
AK_FLOOR = 1.0e-3
#: The HGBS column maps' own stated beam FWHM.
HGBS_BEAM_ARCSEC = 36.3

#: Planck's own beam at 353 GHz, arcsec, as it actually acts on the R1.20
#: all-sky thermal-dust product this pipeline reads (`sky.derived.
#: planck_column`'s `measure_beam`): 5.03', found by cross-correlating a
#: native-resolution HGBS map, smoothed through a trial-beam ladder,
#: against native Planck tau353, since Planck's own FITS header carries
#: no beam keyword for this product. NOT the nominal 2015-release 353 GHz
#: scan-beam FWHM (4.94' = 296") -- that is the instrument's beam on the
#: sky; this is the map's own effective resolution after the R1.20
#: pipeline's own smoothing and pixelisation, the beam this project's
#: column actually carries. The one Planck beam every consumer reads:
#: the sub-beam ladder's own nearest rung (`SCALES`), the completion
#: factor's own tabulation point (`L302` below), and `prior.kernel`'s
#: `STATED_BEAM_ARCSEC["planck"]` (imported from here, never redefined).
PLANCK_BEAM_ARCSEC = 301.8

#: Common working grid every map is decimated onto before smoothing.
TARGET_PIX_ARCSEC = 12.0
#: Gaussian FWHM with the same variance as a working-cell top hat, sqrt(8 ln2 / 12).
FWHM_PER_TOPHAT = 0.6797

#: The beam ladder, arcsec. 302 sits at Planck's measured beam (5.03');
#: 821 at the nside-256 pixel the extinction-profile grid uses (13.7').
SCALES = np.array([45.0, 54.0, 72.0, 90.0, 108.0, 130.0, 160.0, 190.0, 230.0,
                    302.0, 360.0, 432.0, 540.0, 660.0, 821.0, 1000.0])

#: A cell needs at least this much real (non-padding, non-edge) weight in
#: its normalised convolution before it counts as measured sky.
MIN_WEIGHT = 0.90
#: Block-bootstrap tile size.
BOOT_TILE_ARCSEC = 512.0

#: The joblib memory-cap formula (module docstring): one worker's fixed
#: interpreter/library overhead, the per-pixel cost of one map job above
#: that baseline (measured on the orionB map, the largest, with a margin
#: over the observed peak), and the ceiling the pool must fit
#: workers-times-per-job under.
PROCESS_BASELINE_BYTES = 300 * (1024 ** 2)
PEAK_BYTES_PER_PIXEL = 17
POOL_MEM_CEILING_BYTES = 6 * (1024 ** 3)

# d = ln A_ref - ln A_L, the two-scale increment histogram grid.
D_LO, D_HI, N_D = -2.0, 2.0, 4000
D_EDGES = np.linspace(D_LO, D_HI, N_D + 1)
D_CENT = 0.5 * (D_EDGES[1:] + D_EDGES[:-1])
D_DX = D_EDGES[1] - D_EDGES[0]

# Coarse d grid for the per-tile block bootstrap.
DT_LO, DT_HI, N_DT = -1.0, 1.0, 200
_DT = (DT_HI - DT_LO) / N_DT
DT_CENT = np.concatenate(([DT_LO - 0.5 * _DT],
                          DT_LO + _DT * (np.arange(N_DT) + 0.5),
                          [DT_HI + 0.5 * _DT]))

# 2-D conditional histogram: ln A_beam x d -- p(true column | beam value).
K_A_EDGES = np.linspace(np.log(0.01), np.log(30.0), 61)
K_D_EDGES = np.linspace(-1.5, 1.5, 301)

#: sqrt(8 ln2): converts a beam FWHM (arcsec) to sigma units for the
#: power-spectrum model below.
FWHM = 2.3548200450309493
#: The three beams the completion factor is tabulated at, and the target
#: added at the map's own reference beam for the beam-rescaling law.
L_REF, L108, L302, L821 = HGBS_BEAM_ARCSEC, 108.0, PLANCK_BEAM_ARCSEC, 821.0
TARGETS = np.array([L_REF, L108, L302, L821])
BETA_GRID = np.arange(2.05, 5.001, 0.005)
U = np.exp(np.linspace(np.log(1e-4), np.log(60.0), 4000))
QS = [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99]
#: The held-out-scale validation boundary: fit on pairs with L1 >= this,
#: predict the finer, held-out pairs.
VALID_SPLIT = 108.0
#: A pair enters the fit only if its two beams differ by at least this
#: factor; nearly-equal beams give a near-cancelling difference whose
#: width is noise.
RHO_MIN = 1.5
N_BOOT = 200
#: Ladder rungs must sit within this many arcsec of a tabulated target.
LADDER_TOL_ARCSEC = 0.5

_UPOW = U[None, :] ** (1.0 - BETA_GRID[:, None])     # (n_beta, n_u)
_G1 = np.exp(-U ** 2 / 2.0)
_JCACHE = {}

def _Jcol(rho):
    """`ln J(beta, rho)` over the whole beta grid, for one beam ratio `rho`."""
    key = round(float(rho), 6)
    v = _JCACHE.get(key)
    if v is None:
        f = (_G1 - np.exp(-(U * rho) ** 2 / 2.0)) ** 2
        v = np.log(np.maximum(np.trapz(_UPOW * f[None, :], U, axis=1), 1e-300))
        _JCACHE[key] = v
    return v

_LNJINF = np.log(np.trapz(_UPOW * ((1.0 - _G1) ** 2)[None, :], U, axis=1))

def _Jinf_at(beta):
    """`J(beta, rho -> inf)`: the pencil-to-beam integral at the fitted beta."""
    return float(np.exp(np.interp(beta, BETA_GRID, _LNJINF)))

def quant(counts, cent, qs=QS):
    """Quantiles `qs` of a histogram `counts` on centres `cent`; NaN below
    200 counts, too few to resolve a quantile curve."""
    c = np.asarray(counts, dtype=np.float64)
    t = c.sum()
    if t < 200:
        return np.full(len(qs), np.nan)
    return np.interp(qs, np.cumsum(c) / t, cent)

def sd_of(counts, cent):
    """Standard deviation of a histogram; the measured two-scale width."""
    c = np.asarray(counts, dtype=np.float64)
    t = c.sum()
    if t < 200:
        return np.nan
    mu = float((c * cent).sum() / t)
    return float(np.sqrt((c * (cent - mu) ** 2).sum() / t))

def w68_of(counts, cent):
    """Half the 16-84 percentile spread: a robust alternative width."""
    q = quant(counts, cent, [0.16, 0.84])
    return 0.5 * (q[1] - q[0])

def fit_ps(pairs, sel=None, rho_min=RHO_MIN):
    """Fits `Var(L1->L2) = A * sigma_1**(beta-2) * J(beta, sigma_2/sigma_1)`
    for a 2-D power-law power spectrum `P(k) ~ k**-beta` seen through
    Gaussian beams -- the model that turns two-scale increments into an
    absolute pencil-to-beam width. `pairs` are `(L1, L2, width)` rows.
    Returns `(A, beta, rms of the ln-width residual)`.
    """
    P = pairs if sel is None else pairs[sel]
    P = P[np.isfinite(P[:, 2]) & (P[:, 2] > 0) & (P[:, 1] >= rho_min * P[:, 0])]
    if len(P) < 4:
        return np.nan, np.nan, np.nan
    ls1 = np.log(P[:, 0] / FWHM)
    lJ = np.column_stack([_Jcol(r) for r in P[:, 1] / P[:, 0]])   # (n_beta, n_pair)
    lv = lJ + (BETA_GRID[:, None] - 2.0) * ls1[None, :]
    lw2 = 2.0 * np.log(P[:, 2])
    la = np.mean(lw2[None, :] - lv, axis=1)
    ss = np.mean((lw2[None, :] - lv - la[:, None]) ** 2, axis=1)
    i = int(np.argmin(ss))
    return float(np.exp(la[i])), float(BETA_GRID[i]), float(np.sqrt(ss[i]) / 2.0)

def frac_measured(beta, L, L0):
    """Fraction of `Var(0->L)` contributed by wavenumbers inside the map's
    own resolved range (`k` with `u = k*sigma < L/L0`); the remainder is
    the model extrapolation below the map's native beam."""
    if not np.isfinite(beta):
        return np.nan
    umax = float(L) / float(L0)
    f = U ** (1.0 - beta) * (1.0 - _G1) ** 2
    tot = np.trapz(f, U)
    m = U <= umax
    return float(np.trapz(f[m], U[m]) / tot) if tot > 0 else np.nan

def w_abs(A, beta, L):
    """The absolute pencil-to-beam width at beam `L`, from the fitted
    power spectrum."""
    if not np.isfinite(A) or not np.isfinite(beta):
        return np.nan * np.asarray(L, dtype=float)
    return np.sqrt(A * (np.asarray(L, dtype=float) / FWHM) ** (beta - 2.0)
                   * _Jinf_at(beta))

def predict(A, beta, L1, L2):
    """The two-scale width the fit predicts for `L1 -> L2`, used by the
    held-out-scale validation below."""
    i = int(np.argmin(np.abs(BETA_GRID - beta)))
    lJ = np.array([_Jcol(r)[i] for r in np.atleast_1d(L2 / L1)])
    return np.sqrt(A * (np.atleast_1d(L1) / FWHM) ** (beta - 2.0) * np.exp(lJ))

def build_pairs(hist_d, ladder, scales, L0, cent, stat):
    """Every measured `(L1, L2, width)` pair: the map's native resolution
    against each ladder rung, plus each ladder rung already visited as a
    reference (`ladder`, keyed `"L<rung>"`) against every coarser rung."""
    rows = []
    for i, L in enumerate(scales):
        v = stat(hist_d[i], cent)
        if np.isfinite(v) and v > 0:
            rows.append((L0, L, v))
    for key, hh in sorted(ladder.items()):
        Lref = float(key[1:])
        for i, L in enumerate(scales):
            if L <= Lref:
                continue
            v = stat(hh[i], cent)
            if np.isfinite(v) and v > 0:
                rows.append((Lref, L, v))
    return np.array(rows)

# ------------------------------------------------------------------ helpers

def _load_map(path):
    """Loads one HGBS map as float32. A `.gz` map is streamed to a plain
    temporary FITS file in bounded chunks rather than decompressed
    wholesale into a Python buffer, so the load never holds the
    compressed stream and a full in-memory decompressed copy at the same
    time; the temporary file is then memory-mapped and copied into the
    one float32 array this function returns, and removed."""
    if path.endswith(".gz"):
        fd, tmp_path = tempfile.mkstemp(suffix=".fits")
        os.close(fd)
        try:
            with gzip.open(path, "rb") as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    dst.write(chunk)
            raw = fits.getdata(tmp_path, memmap=True)
            out = np.squeeze(np.asarray(raw)).astype(np.float32, copy=True)
        finally:
            os.remove(tmp_path)
    else:
        raw = fits.getdata(path, memmap=True)
        out = np.squeeze(np.asarray(raw)).astype(np.float32, copy=False)
    return out

def _decimate_mean(ak, valid, f):
    """Block-mean by integer factor `f`, NaN-aware; `w` is the real-sky
    fraction per cell. `ak` is zeroed at its invalid pixels in place (the
    caller's copy is not needed again), so no full native-resolution
    temporary is made; each block sum accumulates in float64 -- the only
    float64 use here -- and is cast back to float32 right away."""
    h, w = ak.shape
    h2, w2 = (h // f) * f, (w // f) * f
    ak = ak[:h2, :w2]
    valid = valid[:h2, :w2]
    ak[~valid] = 0.0
    a = ak.reshape(h2 // f, f, w2 // f, f).sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
    m = valid.reshape(h2 // f, f, w2 // f, f).sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
    out = np.zeros_like(a)
    ok = m > 0
    out[ok] = a[ok] / m[ok]
    return out, m / (f * f)

def _gauss_stack(A, W, pix, sig_list):
    """Normalised Gaussian convolution of `A` (weighted by `W`) at each sigma:
    one rfft2 of `(A*W)` and `W`, each ladder rung an analytic multiply and an
    inverse transform. Zero-padded, so the footprint edge is low weight, not
    leaked column (the caller masks on that weight)."""
    ny, nx = A.shape
    smax = max(sig_list) / pix
    pad = int(np.ceil(4.0 * smax)) + 2
    py = sfft.next_fast_len(ny + 2 * pad)
    px_ = sfft.next_fast_len(nx + 2 * pad)
    num = np.zeros((py, px_), dtype=np.float32)
    den = np.zeros((py, px_), dtype=np.float32)
    num[:ny, :nx] = (A * W).astype(np.float32)
    den[:ny, :nx] = W.astype(np.float32)
    FN = sfft.rfft2(num, workers=1)
    FD = sfft.rfft2(den, workers=1)
    del num, den
    ky = np.fft.fftfreq(py)[:, None]
    kx = np.fft.rfftfreq(px_)[None, :]
    k2 = (ky * ky + kx * kx).astype(np.float32)
    del ky, kx
    for sg in sig_list:
        s_px = sg / pix
        tf = np.exp(-2.0 * (np.pi ** 2) * (s_px ** 2) * k2)
        n = sfft.irfft2(FN * tf, s=(py, px_), workers=1)[:ny, :nx]
        d = sfft.irfft2(FD * tf, s=(py, px_), workers=1)[:ny, :nx]
        yield n, d

def _hist_pair(idx, nbins):
    return np.bincount(idx, minlength=nbins)[:nbins].astype(np.int64)

# ------------------------------------------------------------- map processing

def process_map(job):
    """One HGBS map: decimate to 12 arcsec, smooth through the beam ladder,
    histogram the two-scale increment `d` against the native resolution and
    each ladder rung already visited (`ladder`), the 2-D conditional
    histogram of `d` against beam value (`kern`), and a per-tile histogram
    for the block bootstrap (`hist_tile`). `job` is `(cloud, path, pix_native)`."""
    cloud, path, pix_native = job
    f = max(1, int(round(TARGET_PIX_ARCSEC / pix_native)))
    pix = pix_native * f

    ak = _load_map(path)
    ak *= np.float32(AK_PER_NH2)
    valid = np.isfinite(ak) & (ak > AK_FLOOR)
    if int(valid.sum()) < 10000:
        return cloud, None
    A0, W0 = _decimate_mean(ak, valid, f)
    del ak, valid
    V0 = W0 >= MIN_WEIGHT
    if int(V0.sum()) < 5000:
        return cloud, None
    A0 = np.where(V0, A0, 0.0)

    # Effective reference resolution: the 36.3" beam convolved with the
    # working-cell top hat.
    L0 = float(np.sqrt(HGBS_BEAM_ARCSEC ** 2 + (FWHM_PER_TOPHAT * pix) ** 2))
    scales = SCALES[SCALES > L0 * 1.05]
    sig_list = [np.sqrt(L * L - L0 * L0) / FWHM for L in scales]
    ny, nx = A0.shape
    ns = len(scales)

    tb = max(2, int(round(BOOT_TILE_ARCSEC / pix)))
    ti = (np.arange(ny) // tb)[:, None] * (nx // tb + 1) + (np.arange(nx) // tb)[None, :]
    n_tiles = int(ti.max()) + 1

    lnA0 = np.zeros_like(A0)
    np.log(A0, out=lnA0, where=V0)

    hist_d = np.zeros((ns, N_D), dtype=np.int64)
    hist_tile = np.zeros((n_tiles, ns, N_DT + 2), dtype=np.int32)
    kern = np.zeros((ns, len(K_A_EDGES) - 1, len(K_D_EDGES) - 1), dtype=np.int32)

    # Extra reference rungs the ladder re-visits as coarser-pair references.
    ref_idx = [i for i, L in enumerate(scales) if L in (72.0, 108.0, 160.0, 230.0)]
    ref_keep = {}
    ladder = {ri: np.zeros((ns, N_D), dtype=np.int64) for ri in ref_idx}

    for i, (n, d) in enumerate(_gauss_stack(A0, W0, pix, sig_list)):
        ok = (d >= MIN_WEIGHT) & V0
        AL = np.zeros_like(n)
        np.divide(n, d, out=AL, where=ok)
        lnAL = np.zeros_like(AL)
        np.log(AL, out=lnAL, where=ok & (AL > AK_FLOOR))
        sel = ok & (AL > AK_FLOOR)
        if int(sel.sum()) < 1000:
            continue
        s0 = lnA0[sel]
        sl = lnAL[sel]
        dd = s0 - sl

        j = np.clip(((dd - D_LO) / D_DX).astype(np.int32), 0, N_D - 1)
        hist_d[i] = _hist_pair(j, N_D)

        jt = np.clip(np.floor((dd - DT_LO) / (DT_HI - DT_LO) * N_DT).astype(np.int32) + 1,
                     0, N_DT + 1)
        comb = ti[sel].astype(np.int64) * (N_DT + 2) + jt
        bc = np.bincount(comb, minlength=n_tiles * (N_DT + 2))
        hist_tile[:, i, :] = bc[:n_tiles * (N_DT + 2)].reshape(n_tiles, N_DT + 2)

        ja = np.searchsorted(K_A_EDGES, sl, side="right") - 1
        jd = np.searchsorted(K_D_EDGES, dd, side="right") - 1
        m = (ja >= 0) & (ja < len(K_A_EDGES) - 1) & (jd >= 0) & (jd < len(K_D_EDGES) - 1)
        if m.any():
            comb = ja[m].astype(np.int64) * (len(K_D_EDGES) - 1) + jd[m]
            bc = np.bincount(comb, minlength=(len(K_A_EDGES) - 1) * (len(K_D_EDGES) - 1))
            kern[i] = bc.reshape(len(K_A_EDGES) - 1, len(K_D_EDGES) - 1)

        for ri, (rln, rok) in ref_keep.items():
            if ri >= i:
                continue
            m = rok & sel
            if int(m.sum()) < 1000:
                continue
            dd2 = (rln[m] - lnAL[m]).astype(np.float64)
            j2 = np.clip(((dd2 - D_LO) / D_DX).astype(np.int32), 0, N_D - 1)
            ladder[ri][i] = _hist_pair(j2, N_D)

        if i in ref_idx:
            ref_keep[i] = (lnAL.astype(np.float32), sel.copy())

    ladder = {"L%.0f" % scales[ri]: h for ri, h in ladder.items()}
    return cloud, dict(L0=L0, scales=scales, hist_d=hist_d, hist_tile=hist_tile,
                        kern=kern, ladder=ladder, n_tiles=n_tiles)

# ------------------------------------------------------------ region join

def _ladder_index(scales, target, tol=LADDER_TOL_ARCSEC):
    i = int(np.argmin(np.abs(scales - target)))
    if abs(scales[i] - target) > tol:
        raise RuntimeError(
            "sub-beam ladder has no rung within %.2f arcsec of %.1f (nearest %.2f)"
            % (tol, target, scales[i]))
    return i

def _region_join(per_map, cloud_regions, herschel_regions):
    """Pools per-map payloads into one power-spectrum fit, held-out-scale
    validation, block bootstrap and quantile curve per Herschel-covered region."""
    scales = None
    for r in per_map.values():
        if scales is None:
            scales = r["scales"]
        else:
            assert np.allclose(r["scales"], scales)
    ns = len(scales)
    i108, i302, i821 = (_ladder_index(scales, L108), _ladder_index(scales, L302),
                         _ladder_index(scales, L821))

    per_region = {}
    for reg in herschel_regions:
        clouds = [c for c in sorted(per_map) if reg in cloud_regions[c]]
        if not clouds:
            continue
        hd = np.zeros((ns, N_D))
        kern_sum = np.zeros((ns, len(K_A_EDGES) - 1, len(K_D_EDGES) - 1), dtype=np.int64)
        ladder, tiles, L0s = {}, [], []
        for c in clouds:
            r = per_map[c]
            hd += r["hist_d"]
            kern_sum += r["kern"]
            tiles.append(r["hist_tile"].reshape(r["hist_tile"].shape[0], -1))
            L0s.append(r["L0"])
            for kk, vv in r["ladder"].items():
                ladder[kk] = ladder.get(kk, 0) + vv
        L0 = float(np.mean(L0s))
        tile = np.concatenate(tiles, axis=0).astype(np.float32)
        tile = tile[tile.sum(axis=1) > 0]
        n_tiles = tile.shape[0]

        Q = np.array([quant(hd[i], D_CENT) for i in range(ns)])
        sd = np.array([sd_of(hd[i], D_CENT) for i in range(ns)])

        pairs_sd = build_pairs(hd, ladder, scales, L0, D_CENT, sd_of)
        A, beta, rms = fit_ps(pairs_sd)

        # Held-out-scale validation: fit on coarse pairs only, predict the
        # finer, held-out pairs, report the log-error in dex.
        coarse = pairs_sd[:, 0] >= VALID_SPLIT
        Ac, bc, _ = fit_ps(pairs_sd, coarse)
        Pf = pairs_sd[(~coarse) & (pairs_sd[:, 1] >= RHO_MIN * pairs_sd[:, 0])]
        n_fit, n_pred, rms_pred_dex = int(coarse.sum()), int(len(Pf)), np.nan
        if len(Pf):
            pr = predict(Ac, bc, Pf[:, 0], Pf[:, 1])
            lerr = np.log(pr / Pf[:, 2])
            rms_pred_dex = float(np.sqrt(np.mean(lerr ** 2)) / np.log(10.0))

        # Block bootstrap over BOOT_TILE_ARCSEC tiles for the fit's own
        # sampling uncertainty.
        rng = np.random.default_rng(abs(hash(reg)) % (2 ** 31))
        Wd = rng.multinomial(n_tiles, np.full(n_tiles, 1.0 / n_tiles),
                             size=N_BOOT).astype(np.float32)
        acc = (Wd @ tile).reshape(N_BOOT, ns, N_DT + 2)
        jcoarse = np.clip(np.floor((D_CENT - DT_LO) / _DT).astype(int) + 1, 0, N_DT + 1)
        sd_c = np.array([sd_of(np.bincount(jcoarse, weights=hd[i], minlength=N_DT + 2), DT_CENT)
                         for i in range(ns)])
        corr = np.where(np.isfinite(sd_c) & (sd_c > 0), sd / sd_c, 1.0)
        boot = np.full((N_BOOT, 2 + len(TARGETS)), np.nan)
        for b in range(N_BOOT):
            wb = np.array([sd_of(acc[b, i], DT_CENT) for i in range(ns)]) * corr
            pb = np.column_stack([np.full(ns, L0), scales, wb])
            ab, bb, _ = fit_ps(pb)
            boot[b, 0], boot[b, 1] = ab, bb
            boot[b, 2:] = w_abs(ab, bb, TARGETS)
        boot_beta = boot[:, 1]
        w_boot_med = np.nanmedian(boot[:, 2:], axis=0)

        off302, off821 = float(Q[i302, 3]), float(Q[i821, 3])
        if off302 >= 0 or off821 >= 0:
            raise RuntimeError(
                "%s: median offsets at 302/821 arcsec not both negative "
                "(%.4f, %.4f) -- the two-point power-law anchor requires it"
                % (reg, off302, off821))
        offset_exponent = float(np.log(off821 / off302) / np.log(L821 / L302))

        per_region[reg] = dict(
            beta=beta, beta_p16=float(np.nanpercentile(boot_beta, 16)),
            beta_p84=float(np.nanpercentile(boot_beta, 84)),
            w_abs_ref=w_boot_med[0], w_abs_108=w_boot_med[1],
            w_abs_302=w_boot_med[2], w_abs_821=w_boot_med[3],
            completion_108=w_boot_med[1] / sd[i108],
            completion_302=w_boot_med[2] / sd[i302],
            completion_821=w_boot_med[3] / sd[i821],
            rescale_exponent=(beta - 2.0) / 2.0, offset_exponent=offset_exponent,
            n_fit=n_fit, n_pred=n_pred, rms_pred_dex=rms_pred_dex, quantiles=Q,
            kern_L108=kern_sum[i108], kern_L302=kern_sum[i302],
            kern_L821=kern_sum[i821])

        print("  subbeam REGION %-16s maps=%d tiles=%4d beta=%.2f "
              "completion(108/302/821)=%.3f/%.3f/%.3f rms_pred=%.3f dex"
              % (reg, len(clouds), n_tiles, beta, per_region[reg]["completion_108"],
                 per_region[reg]["completion_302"], per_region[reg]["completion_821"],
                 rms_pred_dex), flush=True)

    return scales, per_region

def _map_inventory(config, regions):
    """`(jobs, cloud_regions)`: one job per HGBS map (`sky/download/
    herschel_hgbs.build`'s `_FILES`, files at `<data_root>/sky/download/
    herschel_hgbs/<name>`), largest file first, each map's pixel scale read
    from its own FITS header. Which regions a map serves comes from data,
    never a manifest: `cloud_regions[name]` is every requested region whose
    Herschel `column`/`source` product (SPEC_PRIORS.md section 1.1) has a
    `COVERED` source whose `MAP_ID` points to that map's name in the
    product's own `MAP_NAME` list."""
    hub = os.path.join(config.data_root, "sky", "download", "herschel_hgbs")
    cloud_regions = {name: [] for name in HGBS_FILES}
    for region in sorted(regions):
        col_path = config_module.product_path(config, "sky/derived", "herschel", "column",
                                               "source", region=region)
        if not os.path.exists(col_path):
            raise FileNotFoundError(
                "subbeam.build: Herschel column missing for region %r at %s -- run "
                "the 'sky.derived.herschel_column' RUNBOOK line first" % (region, col_path))
        with h5py.File(col_path, "r") as f:
            covered = np.asarray(f["COVERED"][:], dtype=bool)
            map_id = np.asarray(f["MAP_ID"][:])
            map_name = np.asarray(f["MAP_NAME"][:])
        served_ids = np.unique(map_id[covered])
        for raw in map_name[served_ids[served_ids >= 0]]:
            name = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if name in cloud_regions:
                cloud_regions[name].append(region)

    jobs = []
    for name, regs in cloud_regions.items():
        if not regs:
            continue
        path = os.path.join(hub, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "subbeam.build: HGBS map %r missing at %s -- run the "
                "'sky.download.herschel_hgbs' RUNBOOK line first" % (name, path))
        pixscale = _map_header(path)["pixscale_arcsec"]
        jobs.append((os.path.getsize(path), (name, path, pixscale)))
    jobs.sort(key=lambda j: -j[0])
    return [j for _, j in jobs], cloud_regions

def _pixel_count(path):
    """A map's pixel count from its header alone (no pixel data read)."""
    with fits.open(path, memmap=False) as hd:
        chosen = next(c for c in hd if c.header.get("NAXIS", 0) >= 2)
        ny, nx = chosen.shape[-2:]
    return int(ny) * int(nx)

def _pool_n_jobs(config, jobs):
    """Caps the map-level pool so `workers * per_map_peak` stays under
    `POOL_MEM_CEILING_BYTES`, `per_map_peak` being one worker's fixed
    baseline plus the largest map's pixel count times `PEAK_BYTES_PER_PIXEL`
    (module docstring)."""
    if not jobs:
        return config.n_jobs
    max_pixels = max(_pixel_count(path) for _, path, _ in jobs)
    per_map_peak = PROCESS_BASELINE_BYTES + max_pixels * PEAK_BYTES_PER_PIXEL
    cap = max(1, POOL_MEM_CEILING_BYTES // max(per_map_peak, 1))
    return int(min(config.n_jobs, cap))

def build(config, regions=None):
    """Builds the sub-beam region product, parallelised over HGBS maps with
    joblib (largest map first, pool size capped by `_pool_n_jobs`); the map
    inventory and which regions each map serves come from `_map_inventory`
    (the fetched file set and the Herschel column products), never a
    manifest."""
    with progress_module.Stage("sky.derived.subbeam") as st:
        if regions is None:
            regions = [r.name for r in regions_module.REGIONS]
        regions = set(regions) & {r.name for r in regions_module.REGIONS}

        jobs, cloud_regions = _map_inventory(config, regions)
        print("subbeam: %d HGBS maps overlapping the requested regions" % len(jobs),
              flush=True)
        n_jobs = _pool_n_jobs(config, jobs)
        n_maps = len(jobs)
        n_done = [0]

        def _one(j):
            r = process_map(j)
            n_done[0] += 1
            st.tick(n_done[0], n_maps, "maps")
            return r

        results = Parallel(n_jobs=n_jobs)(delayed(_one)(j) for j in jobs)
        per_map = {cloud: payload for cloud, payload in results if payload is not None}
        print("subbeam: %d/%d maps processed" % (len(per_map), len(jobs)), flush=True)

        herschel_regions = sorted(regions)
        scales, per_region = _region_join(per_map, cloud_regions, herschel_regions)
        regs = list(per_region)
        print("subbeam: %d/%d requested regions with Herschel coverage"
              % (len(regs), len(regions)), flush=True)

        out_path = config_module.product_path(config, "sky/derived", "herschel", "subbeam",
                                              "region")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        scalar_cols = dict(BETA="beta", BETA_BOOT_P16="beta_p16", BETA_BOOT_P84="beta_p84",
                           W_ABS_36P3="w_abs_ref", W_ABS_L108="w_abs_108",
                           W_ABS_L302="w_abs_302", W_ABS_L821="w_abs_821",
                           COMPLETION_L108="completion_108", COMPLETION_L302="completion_302",
                           COMPLETION_L821="completion_821",
                           RESCALE_EXPONENT="rescale_exponent",
                           OFFSET_EXPONENT="offset_exponent", N_FIT="n_fit",
                           N_PRED="n_pred", RMS_PRED_DEX="rms_pred_dex")
        with h5py.File(out_path, "w") as fh:
            fh.attrs["GRANULE"] = "region"
            fh.create_dataset("REGION", data=np.array([r.encode("utf-8") for r in regs]))
            fh.create_dataset("SCALES", data=scales)
            fh.create_dataset("QS", data=np.array(QS))
            for name, key in scalar_cols.items():
                fh.create_dataset(name, data=np.array([per_region[r][key] for r in regs]))
            fh.create_dataset("COND_QUANTILES",
                              data=np.array([per_region[r]["quantiles"] for r in regs]))
            # The column-conditional kernel `prior/kernel.py` evaluates: the
            # 2-D KA (log column) x KD (log ratio) count histogram `kern`,
            # summed over this region's maps, at each of the three tabulated
            # beams. KA_EDGES/KD_EDGES are the shared axes (K_A_EDGES/K_D_EDGES
            # above); K_A_EDGES is already in ln(column).
            fh.create_dataset("KA_EDGES", data=K_A_EDGES)
            fh.create_dataset("KD_EDGES", data=K_D_EDGES)
            for lab in ("L108", "L302", "L821"):
                fh.create_dataset("COND_KERNEL_%s" % lab,
                                  data=np.array([per_region[r]["kern_%s" % lab]
                                                for r in regs]))
        print("subbeam: wrote %s (%.2f MB)"
              % (out_path, os.path.getsize(out_path) / 1e6), flush=True)
        _write_mixture_datasets(out_path)
        st.done(out_path, regions=len(regs), maps=len(per_map))

#: A (region, beam, KA) cell needs this many counts before a mixture is fit.
MIX_MIN_COUNTS = 200
#: Gauss-Hermite nodes used to average the noise-convolved model over the
#: fine map's Gaussian noise `eps ~ Normal(0, sigma_map)`.
N_GH_NOISE = 31
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(N_GH_NOISE)
_GH_WN = _GH_W / np.sqrt(np.pi)
#: A structural mixture is reported as reducible to one Gaussian when the
#: two CDFs differ by no more than this everywhere.
SINGLE_GAUSS_CDF_TOL = 0.02


def _estimate_fine_noise(kern_L108_region, ka_cent, kd_cent):
    """Fine-map (36.3 arcsec) noise level in A_K units for one region, from
    its own 108 arcsec conditional histogram alone (no per-map error plane
    or header noise value is read here). At the lowest populated column
    bin the true column is close to the map floor, so the OBSERVED
    `s = ln(T/A)` distribution's upper half is driven almost entirely by
    positive additive noise excursions pushing `T` up from a near-floor
    value -- unlike the lower half, which is compressed against the floor
    by the log. Half the 50-84 percentile spread of `s` in that bin,
    converted back to linear units by the bin's own `A`, estimates the
    noise's standard deviation."""
    counts_per_bin = kern_L108_region.sum(axis=1)
    order = np.argsort(ka_cent)
    for j in order:
        if counts_per_bin[j] >= MIX_MIN_COUNTS:
            c = kern_L108_region[j].astype(np.float64)
            tot = c.sum()
            cdf = np.cumsum(c) / tot
            q50, q84 = np.interp([0.50, 0.84], cdf, kd_cent)
            A = float(np.exp(ka_cent[j]))
            return A * (q84 - q50)
    return np.nan


def _mixture_cdf(x, w, mu1, s1, mu2, s2):
    """CDF of the two-Gaussian structural mixture at `x`."""
    return w * ndtr((x - mu1) / s1) + (1.0 - w) * ndtr((x - mu2) / s2)


def mixture_moments(w, mu1, s1, mu2, s2):
    """Mean, standard deviation, skewness and excess kurtosis of the
    two-Gaussian mixture, in closed form from its component parameters."""
    mean = w * mu1 + (1.0 - w) * mu2
    d1, d2 = mu1 - mean, mu2 - mean
    var = w * (s1 ** 2 + d1 ** 2) + (1.0 - w) * (s2 ** 2 + d2 ** 2)
    sd = np.sqrt(np.maximum(var, 0.0))
    m3 = w * d1 * (d1 ** 2 + 3.0 * s1 ** 2) + (1.0 - w) * d2 * (d2 ** 2 + 3.0 * s2 ** 2)
    m4 = (w * (3.0 * s1 ** 4 + 6.0 * s1 ** 2 * d1 ** 2 + d1 ** 4)
          + (1.0 - w) * (3.0 * s2 ** 4 + 6.0 * s2 ** 2 * d2 ** 2 + d2 ** 4))
    skew = m3 / sd ** 3 if sd > 0 else np.nan
    kurt = m4 / sd ** 4 - 3.0 if sd > 0 else np.nan
    return mean, sd, skew, kurt


def forward_obs_cdf(edges, A, w, mu1, s1, mu2, s2, sigma_map):
    """CDF of the OBSERVED `s_obs = ln(max(A*exp(s)+eps, floor)/A)` at each
    of `edges`, `s` drawn from the structural mixture and
    `eps ~ Normal(0, sigma_map)` the fine map's own noise. Because
    `T_obs = max(A*exp(s)+eps, floor)` is monotonic non-decreasing in `s`
    for fixed `eps`, `P(s_obs<=edge | eps) = mixture_cdf(ln((A*exp(edge)-eps)/A))`
    whenever `A*exp(edge)-eps` is positive (zero otherwise, since `T` itself
    is always positive), and the noise is integrated out by Gauss-Hermite
    quadrature over `eps`."""
    Tobs_edge = A * np.exp(np.asarray(edges, dtype=np.float64))
    eps = sigma_map * np.sqrt(2.0) * _GH_X
    thresh = Tobs_edge[:, None] - eps[None, :]
    pos = thresh > 0.0
    safe = np.where(pos, thresh, 1.0)
    s_true_thr = np.log(safe / A)
    cdf_k = np.where(pos, _mixture_cdf(s_true_thr, w, mu1, s1, mu2, s2), 0.0)
    cdf_edge = (cdf_k * _GH_WN[None, :]).sum(axis=1)
    return np.where(Tobs_edge >= AK_FLOOR, cdf_edge, 0.0)


def fit_one_mixture(counts, A, sigma_map, kd_edges, kd_cent):
    """Weighted-least-squares fit of the noise-convolved forward model
    (`forward_obs_cdf`) to one observed histogram `counts` on `kd_edges`,
    from a moment-matched two-component start. Returns
    `(w, mu1, s1, mu2, s2, max_cdf_err)`, `max_cdf_err` the largest
    absolute difference between the fitted model's CDF and the raw
    empirical CDF at the bin edges."""
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    p = counts / tot
    mean = float(np.sum(p * kd_cent))
    sd = float(np.sqrt(max(np.sum(p * (kd_cent - mean) ** 2), 1e-8)))

    def resid(theta):
        wl, mu1, ls1, dmu, ls2 = theta
        w = 1.0 / (1.0 + np.exp(-wl))
        s1, s2 = np.exp(ls1), np.exp(ls2)
        mu2 = mu1 + dmu
        cdf = forward_obs_cdf(kd_edges, A, w, mu1, s1, mu2, s2, sigma_map)
        pred = np.diff(cdf) * tot
        return (counts - pred) / np.sqrt(pred + 1.0)

    theta0 = [0.0, mean - 0.3 * sd, np.log(max(sd * 0.7, 1e-3)),
              0.6 * sd, np.log(max(sd * 1.5, 1e-3))]
    res = optimize.least_squares(resid, theta0, method="lm", max_nfev=400)
    wl, mu1, ls1, dmu, ls2 = res.x
    w = 1.0 / (1.0 + np.exp(-wl))
    s1, s2 = np.exp(ls1), np.exp(ls2)
    mu2 = mu1 + dmu
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        s1, s2 = s2, s1
        w = 1.0 - w
    cdf_fit = forward_obs_cdf(kd_edges, A, w, mu1, s1, mu2, s2, sigma_map)
    cdf_emp = np.concatenate(([0.0], np.cumsum(counts) / tot))
    max_err = float(np.max(np.abs(cdf_fit - cdf_emp)))
    return w, float(mu1), float(s1), float(mu2), float(s2), max_err


def _fit_pooled_mixture(kern, fine_noise, ka_cent, kd_edges, kd_cent, n_beam, n_ka, beams):
    """Fits the noise-separated structural mixture on the SURVEY-POOLED
    histogram at every (beam, KA bin): the raw per-region histograms are
    summed count-for-count (a count-weighted pool of histograms is just
    their sum), and the noise level fed to the same forward model is the
    count-weighted root-mean-square of the contributing regions' own
    `FINE_MAP_NOISE_K`, weighted by each region's own counts in that bin
    -- the pooled distribution is the count-weighted mixture of the
    regions' distributions, not an average of their fitted parameters, so
    it needs its own fit. Returns `(W, MU1, MU2, SIG1, SIG2, ERR)`, each
    `(n_beam, n_ka)`."""
    shp = (n_beam, n_ka)
    W = np.full(shp, np.nan)
    MU1 = np.full(shp, np.nan)
    MU2 = np.full(shp, np.nan)
    S1 = np.full(shp, np.nan)
    S2 = np.full(shp, np.nan)
    ERR = np.full(shp, np.nan)
    for b, lab in enumerate(beams):
        K = kern[lab]                              # (n_region, n_ka, n_kd)
        counts_per_bin = K.sum(axis=2)              # (n_region, n_ka)
        for j in range(n_ka):
            pooled_counts = K[:, j, :].sum(axis=0)  # (n_kd,)
            if pooled_counts.sum() < MIX_MIN_COUNTS:
                continue
            wt = counts_per_bin[:, j]
            ok = np.isfinite(fine_noise) & (wt > 0)
            if not ok.any():
                continue
            sigma_pool = float(np.sqrt(np.sum(wt[ok] * fine_noise[ok] ** 2)
                                       / np.sum(wt[ok])))
            A = float(np.exp(ka_cent[j]))
            w, mu1, s1, mu2, s2, err = fit_one_mixture(
                pooled_counts, A, sigma_pool, kd_edges, kd_cent)
            W[b, j], MU1[b, j], MU2[b, j] = w, mu1, mu2
            S1[b, j], S2[b, j], ERR[b, j] = s1, s2, err
    return W, MU1, MU2, S1, S2, ERR


def _write_mixture_datasets(out_path):
    """Fits the noise-separated structural mixture at every (region, beam,
    KA bin) with at least `MIX_MIN_COUNTS` counts, from the histograms
    already stored in the sub-beam product, plus the survey-pooled fit at
    every (beam, KA bin) (`_fit_pooled_mixture`), and adds both as new
    datasets (nothing already in the file is touched)."""
    beams = ("L108", "L302", "L821")
    with h5py.File(out_path, "r") as fh:
        ka_edges = fh["KA_EDGES"][:]
        kd_edges = fh["KD_EDGES"][:]
        kern = {lab: fh["COND_KERNEL_%s" % lab][:] for lab in beams}
    ka_cent = 0.5 * (ka_edges[:-1] + ka_edges[1:])
    kd_cent = 0.5 * (kd_edges[:-1] + kd_edges[1:])
    n_region, n_ka = kern["L108"].shape[0], len(ka_cent)
    n_beam = len(beams)

    fine_noise = np.array([_estimate_fine_noise(kern["L108"][r], ka_cent, kd_cent)
                           for r in range(n_region)])

    shp = (n_region, n_beam, n_ka)
    W = np.full(shp, np.nan)
    MU1 = np.full(shp, np.nan)
    MU2 = np.full(shp, np.nan)
    S1 = np.full(shp, np.nan)
    S2 = np.full(shp, np.nan)
    ERR = np.full(shp, np.nan)

    for r in range(n_region):
        sigma_map = fine_noise[r]
        for b, lab in enumerate(beams):
            K = kern[lab][r]
            counts_per_bin = K.sum(axis=1)
            for j in range(n_ka):
                if counts_per_bin[j] < MIX_MIN_COUNTS or not np.isfinite(sigma_map):
                    continue
                A = float(np.exp(ka_cent[j]))
                w, mu1, s1, mu2, s2, err = fit_one_mixture(
                    K[j], A, sigma_map, kd_edges, kd_cent)
                W[r, b, j], MU1[r, b, j], MU2[r, b, j] = w, mu1, mu2
                S1[r, b, j], S2[r, b, j], ERR[r, b, j] = s1, s2, err

    (PW, PMU1, PMU2, PS1, PS2, PERR) = _fit_pooled_mixture(
        kern, fine_noise, ka_cent, kd_edges, kd_cent, n_beam, n_ka, beams)

    with h5py.File(out_path, "a") as fh:
        for name in ("MIX_W", "MIX_MU1", "MIX_MU2", "MIX_SIG1", "MIX_SIG2",
                     "MIX_MAX_CDF_ERR", "MIX_KA_CENTRES", "FINE_MAP_NOISE_K",
                     "MIX_POOLED_W", "MIX_POOLED_MU1", "MIX_POOLED_MU2",
                     "MIX_POOLED_SIG1", "MIX_POOLED_SIG2", "MIX_POOLED_MAX_CDF_ERR"):
            if name in fh:
                del fh[name]
        fh.create_dataset("MIX_W", data=W)
        fh.create_dataset("MIX_MU1", data=MU1)
        fh.create_dataset("MIX_MU2", data=MU2)
        fh.create_dataset("MIX_SIG1", data=S1)
        fh.create_dataset("MIX_SIG2", data=S2)
        fh.create_dataset("MIX_MAX_CDF_ERR", data=ERR)
        fh.create_dataset("MIX_KA_CENTRES", data=ka_cent)
        fh.create_dataset("FINE_MAP_NOISE_K", data=fine_noise)
        fh.create_dataset("MIX_POOLED_W", data=PW)
        fh.create_dataset("MIX_POOLED_MU1", data=PMU1)
        fh.create_dataset("MIX_POOLED_MU2", data=PMU2)
        fh.create_dataset("MIX_POOLED_SIG1", data=PS1)
        fh.create_dataset("MIX_POOLED_SIG2", data=PS2)
        fh.create_dataset("MIX_POOLED_MAX_CDF_ERR", data=PERR)
    print("subbeam: wrote mixture fit datasets (per-region and pooled) to %s"
          % out_path, flush=True)


def fit_only(config, regions=None):
    """Fits and stores the noise-separated mixture datasets from the
    sub-beam product already on disk, without touching any map."""
    out_path = config_module.product_path(config, "sky/derived", "herschel",
                                          "subbeam", "region")
    _write_mixture_datasets(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--fit-only", action="store_true",
                        help="fit the noise-separated mixture from the "
                             "persisted histograms only; no map is read")
    args = parser.parse_args()
    cfg = config_module.load(args.config)
    if args.fit_only:
        fit_only(cfg, regions=args.regions)
    else:
        build(cfg, regions=args.regions)
