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
"""

import json
import os

import h5py
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from scipy import fft as sfft

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run

#: N(H2) -> A_K, ZGR23 (matches sesnaimpute.sky.derived.column's own use).
AK_PER_NH2 = 1.12e-22
#: Below this a map pixel is padding or noise, not measured sky.
AK_FLOOR = 1.0e-3
#: The HGBS column maps' own stated beam FWHM.
HGBS_BEAM_ARCSEC = 36.3

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
L_REF, L108, L302, L821 = HGBS_BEAM_ARCSEC, 108.0, 301.8, 821.0
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
    d = np.squeeze(np.asarray(fits.getdata(path)))
    return d.astype(np.float32)

def _decimate_mean(ak, valid, f):
    """Block-mean by integer factor `f`, NaN-aware; `w` is the real-sky fraction per cell."""
    h, w = ak.shape
    h2, w2 = (h // f) * f, (w // f) * f
    a = np.where(valid[:h2, :w2], ak[:h2, :w2], 0.0).astype(np.float64)
    m = valid[:h2, :w2].astype(np.float64)
    a = a.reshape(h2 // f, f, w2 // f, f).sum(axis=(1, 3))
    m = m.reshape(h2 // f, f, w2 // f, f).sum(axis=(1, 3))
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
        yield n.astype(np.float64), d.astype(np.float64)

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

    dat = _load_map(path)
    ak = dat.astype(np.float64) * AK_PER_NH2
    del dat
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
    """`(jobs, cloud_regions)` from the HGBS maps' own `PRODUCT.json`: one job
    per fetched map `(cloud, local_path, pixel_scale_arcsec)`, largest first;
    `cloud_regions[cloud]` is its `overlap_regions` restricted to `regions`.
    Nothing else is read from `PRODUCT.json`."""
    inv_path = os.path.join(config.data_root, "sky", "download", "herschel-hgbs",
                            "PRODUCT.json")
    with open(inv_path) as fh:
        products = json.load(fh)["products"]
    jobs, cloud_regions = [], {}
    for name, meta in products.items():
        regs = [r for r in meta.get("overlap_regions", []) if r in regions]
        if not meta.get("fetched") or not regs:
            continue
        cloud = meta["cloud"]
        cloud_regions[cloud] = regs
        jobs.append((meta["bytes"], (cloud, meta["local_path"],
                                     float(meta["pixel_scale_arcsec"]))))
    jobs.sort(key=lambda j: -j[0])
    return [j for _, j in jobs], cloud_regions

def build(config, regions=None):
    """Builds the sub-beam region product, parallelised over HGBS maps with
    joblib (largest map first); reads the map inventory from
    `sky/download/herschel-hgbs/PRODUCT.json`."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    regions = set(regions) & {r.name for r in regions_module.REGIONS}

    jobs, cloud_regions = _map_inventory(config, regions)
    print("subbeam: %d HGBS maps overlapping the requested regions" % len(jobs),
          flush=True)
    results = Parallel(n_jobs=-1)(delayed(process_map)(j) for j in jobs)
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

if __name__ == "__main__":
    run(build)
