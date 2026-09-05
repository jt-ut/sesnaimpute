"""Per-source Herschel dust column, A_K, from the HGBS N(H2) maps.

SPEC_PRIORS.md section 1.1: `A_s`, the total dust column to infinity on a
source's sightline, in A_K magnitudes, read from thermal-dust emission --
Herschel Gould Belt Survey column density (36.3" beam) wherever a map
covers the source. Currency: A_K = 1.12e-22 * N(H2) (AV_PER_NH2: Bohlin,
Savage & Drake 1978's gas-to-dust ratio; AK_PER_AV: Rieke & Lebofsky
1985, ApJ 288, 618).

Three measurements, in order:

  * COLUMN. Every requested region's own catalogued sources are sampled,
    nearest pixel, against every HGBS map whose manifest entry declares
    it overlaps that region. Where two maps could serve the same source,
    the larger map (by file size, `_size_ordered_desc`) wins -- it is
    taken as the deeper, more complete reduction of that cloud.
  * BEAM. Each map's own absolute beam FWHM, fit from the map's
    azimuthally averaged power spectrum (`_beam_powerspec_array`): a
    map-to-map replicate comparison needs the field's own resolution,
    not the survey's stated 36.3" alone.
  * SIGMA. The zero point (a per-field, fully correlated systematic) and
    the random per-source scatter, both measured from the half-
    difference of independent HGBS reductions of the same sky at every
    pair of overlapping maps (`_pair_native`, `_sig_model`):
    sigma_rand^2 = c0^2 + c1^2 A^2 (fit over every pair's column-binned
    half-difference scatter), sigma_zp the RMS of the pairs' own median
    half-differences. The manifest's overlapping pairs are what the
    survey has; the zero point this measures is nearer a lower bound
    than a converged estimate.

BEAM and SIGMA are survey-wide -- one measurement per map / per
overlapping pair, independent of which regions are requested -- and are
always fit over every map the HGBS manifest lists, written once to the
`beams`/`survey` product (a survey-granule product ignores the region
list, IMPLEMENTATION.md section 1b). COLUMN is per region: `build`
writes one `column`/`source` file per requested region that at least one
map serves; a region no map covers writes nothing.

Lifted from sesna-complete's sky/derived/dust_columns.py Herschel arm
(`_open_hgbs_map`, `_bbox_of`, `_boxes_overlap`, `_size_ordered_desc`,
`_sample_map_job`, `_largest_finite_window`, `_ps_profile`, `_ps_fit`,
`_beam_powerspec_array`, `_pair_native`, `_sig_model`). Dropped: the
manifest-validation and checkpointed-staging bookkeeping, every attrs
stamp beyond `GRANULE`, the beam fit's known-smoothing control check
(a diagnostic that fed no written product), and the old code's
SIG_GAIN/SIG_POS terms -- the two-pair replicate measurement supports
only a zero point and a random term, so SIGMA_A_K is narrowed to those
two in quadrature.
"""

import json
import os
import time

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run

#: N(H2) [cm^-2] -> A_K [mag]. AV_PER_NH2: Bohlin, Savage & Drake 1978
#: gas-to-dust ratio, A_V/N(H2). AK_PER_AV: Rieke & Lebofsky 1985,
#: ApJ 288, 618, A_K/A_V.
AV_PER_NH2 = 1.000e-21
AK_PER_AV = 0.112
NH2_TO_AK = AV_PER_NH2 * AK_PER_AV  # 1.12e-22 mag cm^2

HERSCHEL_STATED_FWHM_ARCSEC = 36.3  # the HGBS survey's own quoted beam.

#: The largest HGBS map is tens of megapixels of float32 plus FFT work
#: arrays at the beam stage; capped rather than one-worker-per-core so a
#: full-survey run does not hold the whole manifest in memory at once.
MAP_WORKERS = 8


# ---------------------------------------------------------------------------
# The manifest and the map geometry it carries.
# ---------------------------------------------------------------------------

def _hgbs_dir(config):
    return os.path.join(config.data_root, "sky", "download", "herschel-hgbs")


def _map_list(config):
    """[{name, path, bbox, regions}] for every fetched HGBS map, sorted
    by name for a deterministic MAP_ID order -- the download hub's own
    PRODUCT.json `products` table, whose `overlap_regions` names which of
    the 30 regions each map serves."""
    hub = _hgbs_dir(config)
    manifest_path = os.path.join(hub, "PRODUCT.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            "herschel_column.build: no HGBS download manifest at %s -- "
            "the Herschel maps are an external input, fetched outside this "
            "package's own RUNBOOK" % manifest_path)
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    out = []
    for name in sorted(manifest["products"]):
        v = manifest["products"][name]
        if not v.get("fetched"):
            continue
        out.append(dict(name=name, path=v["local_path"], bbox=tuple(v["bbox"]),
                         regions=tuple(v["overlap_regions"])))
    if not out:
        raise FileNotFoundError("herschel_column.build: manifest at %s lists no fetched map" % manifest_path)
    return out


def _size_ordered_desc(maps):
    """`maps` sorted by on-disk file size, descending, so the heaviest
    job in a pool of equal-sized workers is dispatched first -- and, for
    the per-source merge, so the largest, most complete reduction of a
    cloud is offered to each source before any smaller one."""
    def _fsize(m):
        try:
            return os.path.getsize(m["path"])
        except OSError:
            return 0
    return sorted(maps, key=_fsize, reverse=True)


def _open_hgbs_map(path):
    """One HGBS FITS map: 2-D float32 N(H2) data, its celestial WCS, and
    its pixel scale in arcsec."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(path, memmap=False) as hd:
        chosen = next(c for c in hd if c.data is not None and np.ndim(c.data) >= 2)
        data = np.squeeze(np.asarray(chosen.data))
        hdr = chosen.header.copy()
    while data.ndim > 2:
        data = data[0]
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    cd = hdr.get("CDELT1", hdr.get("CD1_1"))
    pixscale = abs(float(cd)) * 3600.0
    return data, WCS(hdr).celestial, pixscale


def _bbox_of(wcs, shape):
    """(ra_min, ra_max, dec_min, dec_max) of a map's own footprint, from
    its four corners and edge midpoints, wrap-safe about the map's own
    median RA."""
    ny, nx = shape
    xs = np.array([0, nx - 1, 0, nx - 1, (nx - 1) / 2., (nx - 1) / 2., 0, nx - 1])
    ys = np.array([0, 0, ny - 1, ny - 1, 0, ny - 1, (ny - 1) / 2., (ny - 1) / 2.])
    sky = wcs.pixel_to_world(xs, ys)
    ra = sky.icrs.ra.deg
    dec = sky.icrs.dec.deg
    c = float(np.median(ra))
    ra = c + ((ra - c + 180.0) % 360.0) - 180.0
    return float(ra.min()), float(ra.max()), float(dec.min()), float(dec.max())


def _boxes_overlap(a, b, pad=0.0):
    """Whether two (ra_min, ra_max, dec_min, dec_max) boxes intersect,
    shifting `b`'s RA by whichever multiple of 360 deg brings it nearest
    `a` first, so a box straddling the 0/360 seam is not falsely ruled
    out."""
    if a[3] + pad < b[2] - pad or b[3] + pad < a[2] - pad:
        return False
    sh = round((0.5 * (a[0] + a[1]) - 0.5 * (b[0] + b[1])) / 360.0) * 360.0
    b0, b1 = b[0] + sh, b[1] + sh
    return not (a[1] + pad < b0 - pad or b1 + pad < a[0] - pad)


# ---------------------------------------------------------------------------
# BEAM: each map's own absolute FWHM from its azimuthally averaged P(k).
# ---------------------------------------------------------------------------

def _largest_finite_window(finite, want):
    """(y0, x0, S) of a fully-finite square window, S <= want, or None --
    a summed-area table over the map's finite-pixel mask lets every
    candidate window's finite-pixel count be read in one lookup rather
    than rescanned."""
    ny, nx = finite.shape
    S = min(want, ny, nx)
    while S >= 128:
        ii = np.zeros((ny + 1, nx + 1), dtype=np.int64)
        np.cumsum(np.cumsum(finite.astype(np.int64), axis=0), axis=1, out=ii[1:, 1:])
        step = max(1, S // 8)
        ys = np.arange(0, ny - S + 1, step)
        xs = np.arange(0, nx - S + 1, step)
        if ys.size and xs.size:
            block = (ii[np.ix_(ys + S, xs + S)] - ii[np.ix_(ys, xs + S)]
                     - ii[np.ix_(ys + S, xs)] + ii[np.ix_(ys, xs)])
            j = int(np.argmax(block))
            iy, ix = np.unravel_index(j, block.shape)
            if block[iy, ix] == S * S:
                return int(ys[iy]), int(xs[ix]), int(S)
        S //= 2
    return None


def _ps_profile(sub, px):
    """Azimuthally averaged power spectrum of a square, mean-removed,
    Hann-windowed image."""
    S = sub.shape[0]
    w1 = np.hanning(S)
    a = (sub - sub.mean()) * (w1[:, None] * w1[None, :])
    F = np.fft.rfft2(a)
    P = (F.real ** 2 + F.imag ** 2) / float(S * S)
    ky = np.fft.fftfreq(S, d=px)[:, None]
    kx = np.fft.rfftfreq(S, d=px)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    knyq = 0.5 / px
    kmin = 4.0 / (S * px)
    edges = np.logspace(np.log10(kmin), np.log10(knyq), 45)
    idx = np.digitize(k.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < edges.size - 1)
    nb = np.bincount(idx[ok], minlength=edges.size - 1)
    sp = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=edges.size - 1)
    sk = np.bincount(idx[ok], weights=k.ravel()[ok], minlength=edges.size - 1)
    g = nb > 8
    return sk[g] / nb[g], sp[g] / nb[g]


def _ps_fit(kb, pb):
    """Fit P(k) = A k^-b exp(-4 pi^2 sigma^2 k^2) + N: a power-law sky
    spectrum, a Gaussian beam roll-off, and a white noise floor, over a
    grid of starting beam widths so the fit is not stuck at a local
    minimum."""
    from scipy.optimize import least_squares
    pnorm = float(pb[0])
    pn = pb / pnorm
    y = np.log(pn)

    def model(theta):
        lA, bb, sig, lN = theta
        m = (np.exp(lA) * kb ** (-bb)
             * np.exp(-4.0 * np.pi ** 2 * sig ** 2 * kb ** 2) + np.exp(lN))
        return np.log(m) - y

    lA0 = float(np.log(pn[0]) + 2.5 * np.log(kb[0]))
    lN0 = float(np.log(pn[-1]))
    best = None
    for sig0 in (3.0, 8.0, 15.0, 25.0, 40.0):
        try:
            r = least_squares(model, [lA0, 2.5, sig0, lN0],
                               bounds=([lA0 - 60.0, 0.0, 0.5, lN0 - 40.0],
                                       [lA0 + 60.0, 8.0, 200.0, lN0 + 40.0]),
                               max_nfev=8000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    lA, bb, sig, lN = best.x
    return dict(power_index=float(bb), sigma_arcsec=float(sig),
                fwhm_arcsec=float(2.3548200450309493 * sig),
                noise_floor_rel=float(np.exp(lN)), resid_rms=float(np.sqrt(np.mean(best.fun ** 2))))


def _beam_powerspec_array(data, pixscale_arcsec):
    """Absolute beam FWHM from a map's own azimuthally-averaged P(k),
    fit inside the largest fully-finite window the map offers."""
    t0 = time.time()
    px = pixscale_arcsec
    finite = np.isfinite(data) & (data > 0)
    win = _largest_finite_window(finite, 2048)
    if win is None:
        return dict(ok=False, reason="no fully-finite 128+ window", pixscale_arcsec=px, wall_s=time.time() - t0)
    y0, x0, S = win
    sub = np.asarray(data[y0:y0 + S, x0:x0 + S], dtype=np.float64)
    kb, pb = _ps_profile(sub, px)
    fit = _ps_fit(kb, pb)
    del sub
    if fit is None:
        return dict(ok=False, reason="fit failed", pixscale_arcsec=px, wall_s=time.time() - t0)
    out = dict(ok=True, pixscale_arcsec=px, window=[y0, x0, S], wall_s=time.time() - t0)
    out.update(fit)
    return out


def _beam_job(map_entry):
    data, _wcs, px = _open_hgbs_map(map_entry["path"])
    out = _beam_powerspec_array(data, px)
    del data
    out["map"] = map_entry["name"]
    return out


# ---------------------------------------------------------------------------
# SIGMA: zero point and random scatter from overlapping-map replicates.
# ---------------------------------------------------------------------------

def _pair_native(name_a, path_a, name_b, path_b):
    """The map-to-map SIG_ZP/SIG_RAND replicate: matched NATIVE-
    resolution samples of two overlapping HGBS reductions of the same
    sky, each eroded away from its own map edge by twice the stated beam
    so only their common, fully-sampled interior is compared."""
    from astropy.coordinates import SkyCoord
    from scipy.ndimage import binary_erosion
    t0 = time.time()
    A, wa, pxa = _open_hgbs_map(path_a)
    B, wb, pxb = _open_hgbs_map(path_b)
    erode_arcsec = 2.0 * HERSCHEL_STATED_FWHM_ARCSEC

    def eroded(M, px):
        # A flat square structuring element is the Minkowski sum of a
        # horizontal and a vertical run, so eroding with a (1, 2r+1) then
        # a (2r+1, 1) segment matches eroding once with the full square,
        # far cheaper at a large radius.
        good = np.isfinite(M) & (M > 0)
        r = max(1, int(round(erode_arcsec / px)))
        good = binary_erosion(good, structure=np.ones((1, 2 * r + 1), bool), border_value=0)
        good = binary_erosion(good, structure=np.ones((2 * r + 1, 1), bool), border_value=0)
        return good

    gb = eroded(B, pxb)
    ga = eroded(A, pxa)
    step = max(1, int(round(6.0 / pxb)))
    ny, nx = B.shape
    yy, xx = np.mgrid[0:ny:step, 0:nx:step]
    m = gb[yy, xx]
    fail = lambda reason: dict(pair=(name_a, name_b), ok=False, reason=reason, n=0, wall_s=time.time() - t0)
    if m.sum() < 300:
        return fail("B interior empty")
    yy, xx = yy[m], xx[m]
    vb = B[yy, xx].astype(np.float64)
    del B, gb
    sky = wb.pixel_to_world(xx.astype(float), yy.astype(float))
    xa, ya = wa.world_to_pixel(SkyCoord(sky.icrs.ra, sky.icrs.dec, frame="icrs"))
    xai = np.rint(xa).astype(np.int64)
    yai = np.rint(ya).astype(np.int64)
    nya, nxa = A.shape
    ins = (xai >= 0) & (xai < nxa) & (yai >= 0) & (yai < nya)
    if ins.sum() < 300:
        return fail("no A overlap")
    xai, yai, vb = xai[ins], yai[ins], vb[ins]
    keep = ga[yai, xai]
    if keep.sum() < 300:
        return fail("no common 2-beam interior")
    xai, yai, vb = xai[keep], yai[keep], vb[keep]
    va0 = A[yai, xai].astype(np.float64)
    del A, ga

    d = 0.5 * (vb - va0) * NH2_TO_AK
    mn = 0.5 * (vb + va0) * NH2_TO_AK
    qs = np.percentile(mn, np.linspace(0, 100, 9))
    bins = []
    for i in range(8):
        sel = (mn >= qs[i]) & (mn <= qs[i + 1])
        if sel.sum() > 30:
            dd = d[sel]
            bins.append((float(np.median(mn[sel])),
                         float(1.482602218505602 * np.median(np.abs(dd - np.median(dd))))))
    return dict(pair=(name_a, name_b), ok=True, n=int(vb.size),
                halfdiff_ak_pct=[float(v) for v in np.percentile(d, [1, 16, 50, 84, 99])],
                sig_bins=bins, wall_s=time.time() - t0)


def _sig_model(pairs):
    """SIG_ZP (per-field, fully correlated) and the coefficients of
    SIG_RAND^2 = c0^2 + c1^2 A^2 (per source), both measured from the
    overlapping-map replicates `_pair_native` returns: SIG_ZP is the RMS
    of each pair's own median half-difference; c0, c1 are a least-squares
    fit of every pair's column-binned half-difference scatter (MAD^2)
    against the bin's mean column, pooled over every pair."""
    zp, rnd = [], []
    for r in pairs:
        if not r.get("ok"):
            continue
        zp.append(abs(r["halfdiff_ak_pct"][2]))
        rnd.extend(r["sig_bins"])
    rnd = np.asarray(rnd, dtype=np.float64)
    if rnd.size:
        a2 = rnd[:, 0] ** 2
        s2 = rnd[:, 1] ** 2
        design = np.column_stack([np.ones_like(a2), a2])
        coef, *_ = np.linalg.lstsq(design, s2, rcond=None)
        c0 = float(np.sqrt(max(coef[0], 0.0)))
        c1 = float(np.sqrt(max(coef[1], 0.0)))
    else:
        c0 = c1 = 0.0
    sig_zp = float(np.sqrt(np.mean(np.square(zp)))) if zp else 0.0
    return dict(sig_zp_ak=sig_zp, rand_c0=c0, rand_c1=c1, n_pairs=len(zp))


# ---------------------------------------------------------------------------
# COLUMN: per-source, per-map sampling and the per-region merge.
# ---------------------------------------------------------------------------

def _region_sources(config, region):
    """(RA_DEG, DEC_DEG) for one region's curated catalogue, in
    catalogue-row order -- the order every written column matches."""
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "herschel_column.build: curated catalogue missing for region %r at %s -- "
            "run the '--- catalog ---' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        ra = np.asarray(f["RA_DEG"][:], dtype=np.float64)
        dec = np.asarray(f["DEC_DEG"][:], dtype=np.float64)
    return ra, dec


def _sample_map_job(map_entry, region_sources):
    """One HGBS map, one joblib worker: nearest-pixel N(H2) at every
    source of every requested region the map's own footprint reaches.
    `region_sources` is {region: (ra_deg, dec_deg)}. Vectorised over
    sources within each region; a region's own bounding box is screened
    against the map's own footprint (fresh from its WCS, not the
    manifest) before any `world_to_pixel` call is made."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    data, wcs, _px = _open_hgbs_map(map_entry["path"])
    ny, nx = data.shape
    box = _bbox_of(wcs, data.shape)

    out = {}
    for region, (ra, dec) in region_sources.items():
        if ra.size == 0:
            continue
        rbox = (float(ra.min()), float(ra.max()), float(dec.min()), float(dec.max()))
        if not _boxes_overlap(box, rbox, pad=0.05):
            continue
        cand = np.where((dec >= box[2] - 0.05) & (dec <= box[3] + 0.05))[0]
        if cand.size == 0:
            continue
        x, y = wcs.world_to_pixel(SkyCoord(ra[cand] * u.deg, dec[cand] * u.deg, frame="icrs"))
        xi = np.rint(x).astype(np.int64)
        yi = np.rint(y).astype(np.int64)
        inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        if not inside.any():
            continue
        cand, xi, yi = cand[inside], xi[inside], yi[inside]
        vals = data[yi, xi]
        on = np.isfinite(vals) & (vals > 0)
        if on.any():
            out[region] = (cand[on], vals[on].astype(np.float64))
    del data
    return map_entry["name"], out


def _write_beams_survey(config, maps, beam_by_name, sig):
    """The survey-wide product: every HGBS map's own beam FWHM and the
    zero-point/random sigma model the per-region column files read by
    name."""
    out_path = config_module.product_path(config, "sky/derived", "herschel", "beams", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = [m["name"] for m in maps]
    fwhm = np.array([beam_by_name[n]["fwhm_arcsec"] if beam_by_name.get(n, {}).get("ok") else np.nan
                      for n in names], dtype=np.float32)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        f.create_dataset("MAP_NAME", data=np.array([n.encode("utf-8") for n in names]))
        f.create_dataset("BEAM_FWHM_ARCSEC", data=fwhm)
        f.create_dataset("SIGMA_ZP_K", data=np.float64(sig["sig_zp_ak"]))
        f.create_dataset("C0", data=np.float64(sig["rand_c0"]))
        f.create_dataset("C1", data=np.float64(sig["rand_c1"]))
        f.create_dataset("N_PAIRS", data=np.int64(sig["n_pairs"]))


def _write_region(config, region, a_k, sig_a_k, sig_rand, sig_zp, covered, map_id, map_names):
    """One region's `column`/`source` product, catalogue-row order."""
    out_path = config_module.product_path(config, "sky/derived", "herschel", "column", "source", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.create_dataset("A_K", data=a_k.astype(np.float32))
        f.create_dataset("SIGMA_A_K", data=sig_a_k.astype(np.float32))
        f.create_dataset("SIGMA_RAND_K", data=sig_rand.astype(np.float32))
        f.create_dataset("SIGMA_ZP_K", data=sig_zp.astype(np.float32))
        f.create_dataset("COVERED", data=covered)
        f.create_dataset("MAP_ID", data=map_id.astype(np.int32))
        f.create_dataset("MAP_NAME", data=np.array([n.encode("utf-8") for n in map_names]))


def build(config, regions=None):
    """The Herschel arm (SPEC_PRIORS.md section 1.1): `beams`/`survey`
    (every HGBS map's own beam FWHM and the SIG_ZP/SIG_RAND replicate
    model, fit over the whole manifest regardless of `regions`) and
    `column`/`source`, one file per requested region that at least one
    map serves. Maps are parallelised with joblib; sampling within a map
    is vectorised over that region's sources.
    """
    maps = _map_list(config)

    t_beam0 = time.time()
    ordered = _size_ordered_desc(maps)
    beam_results = Parallel(n_jobs=MAP_WORKERS)(delayed(_beam_job)(m) for m in ordered)
    beam_by_name = {r["map"]: r for r in beam_results}

    pair_jobs = [(maps[i]["name"], maps[i]["path"], maps[j]["name"], maps[j]["path"])
                 for i in range(len(maps)) for j in range(i + 1, len(maps))
                 if _boxes_overlap(maps[i]["bbox"], maps[j]["bbox"])]
    pair_results = Parallel(n_jobs=MAP_WORKERS)(delayed(_pair_native)(*p) for p in pair_jobs)
    sig = _sig_model(pair_results)
    beam_wall_s = time.time() - t_beam0
    _write_beams_survey(config, maps, beam_by_name, sig)

    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]

    t_sample0 = time.time()
    region_ra_dec = {r: _region_sources(config, r) for r in regions
                      if any(r in m["regions"] for m in maps)}
    active_maps = _size_ordered_desc(
        [m for m in maps if set(m["regions"]) & set(region_ra_dec)])
    sample_results = Parallel(n_jobs=MAP_WORKERS)(
        delayed(_sample_map_job)(m, {r: region_ra_dec[r] for r in m["regions"] if r in region_ra_dec})
        for m in active_maps)
    per_map = dict(sample_results)

    written = []
    for region in regions:
        if region not in region_ra_dec:
            continue
        serving = _size_ordered_desc([m for m in maps if region in m["regions"]])
        n = region_ra_dec[region][0].size
        a_k = np.full(n, np.nan, dtype=np.float64)
        map_id = np.full(n, -1, dtype=np.int32)
        for mi, m in enumerate(serving):
            hits = per_map.get(m["name"], {}).get(region)
            if hits is None:
                continue
            idx, nh2 = hits
            fill_mask = map_id[idx] < 0
            if not fill_mask.any():
                continue
            sel = idx[fill_mask]
            a_k[sel] = nh2[fill_mask] * NH2_TO_AK
            map_id[sel] = mi
        covered = map_id >= 0
        sig_rand = np.where(covered, np.sqrt(sig["rand_c0"] ** 2 + (sig["rand_c1"] * a_k) ** 2), np.nan)
        sig_zp = np.where(covered, sig["sig_zp_ak"], np.nan)
        sig_a_k = np.sqrt(sig_rand ** 2 + sig_zp ** 2)
        _write_region(config, region, a_k, sig_a_k, sig_rand, sig_zp, covered, map_id,
                       [m["name"] for m in serving])
        written.append(region)
    sample_wall_s = time.time() - t_sample0

    return dict(beam_wall_s=beam_wall_s, sample_wall_s=sample_wall_s,
                regions_written=written, sig_model=sig)


if __name__ == "__main__":
    run(build)
