"""Absolute normalisation checks of the PACS mosaic (a wrong scale, EEF file, unit or kernel cannot cancel in them).

A REAL unit source is synthesised in a representative input map's own pixels, with no table and no kernel algebra:
the release template (`pacs_fit._template_density`, the published components at the map's position angles) as
surface brightness at the true sky offsets of 6 x 6 sub-samples of each input pixel (the box average), times the
input pixel's nominal area; that image is resampled onto the mosaic grid by `pacs_mosaic.resample_block` (the real
resampler), and measured with the production aperture centred on the source (`pacs_fit._aperture_core`, 5 x 5
sub-pixel edge weights, no sky: the aperture sum). Three numbers follow per (sub-group, position):

  A_real    the aperture sum of the resampled unit source (what the mosaic holds where only that map covers),
  A_direct  the template's own integral (smoothed by the input pixel's box) inside the aperture radius
            (`template_in_aperture`, fine grid, no pixel sampling),
  EEF       the release encircled-energy fraction of the total flux inside the aperture radius (`aperture_eef`).

A_direct / EEF = 1 ties the template, the pixel areas and the EEF file to each other (absolute); A_real / A_direct
is the resampling's own broadening, the independent counterpart of the PSF build's e_broad / e_plain. The mosaic's
expected aperture sum of an injected unit source at a position with sub-group fractions f is sum_k f_k EEF_k
A_real_k / A_direct_k (`expected_enclosed`, used by `pacs_completeness.self_check`); `normalisation_check` compares
it with the production divisor `MosaicPSF.enclosed_at` and with the production injection (`Tile.injected`) on
every sub-group alone, at nodes, mid-cells, outer positions and the representative map's footprint rim.
"""
import hashlib
import time
from types import SimpleNamespace

import numpy as np

from sesnaimpute.sky.derived import pacs_mosaic
from sesnaimpute.sky.derived.pacs_fit import (_aperture_core, _template_density, read_index)
from sesnaimpute.sky.derived.pacs_psf import APERTURES, aperture_eef, published_components_for_map

TIE_TOLERANCE = 0.003            # real (normalised to the release EEF) against the divisor; injected against real
RELEASE_TOLERANCE = 0.015        # A_direct / EEF: the release template against the release EEF (60 arcsec/s: -0.7 %)
SUM_TOLERANCE = 0.001            # pixel sums of the same patch, real against injected
SUBSAMPLES = 6                   # per input pixel side (the box average; 2 x 2 left a 3.2 arcsec pixel's box too narrow: +0.21 %)
RIM_ARCSEC = (10.0, 20.0, 30.0)  # inside and outside the footprint edge
FLUX_MJY = 1000.0


def representative_maps(config, mosaic_dir, meta):
    """Per sub-group: (index record, psf.json map entry, input WCS, components) of the map the PSF build used."""
    recs = {r.path: r for r in read_index(config, meta["region"])}
    out = []
    for k, i in enumerate(meta["rep"]):
        q = meta["maps"][i]
        rec = recs[q["path"]]
        out.append((rec, q, pacs_mosaic._celestial(rec), published_components_for_map(config, rec)))
    return out


def _aperture_sum(img, x, y, radii_px):
    """The aperture sum before sky (the aperture sum the divisor is the fraction of) at pixel (x, y)."""
    r_ap, r_in, r_out = radii_px
    f, s, _sd, area, cov = _aperture_core(np.ascontiguousarray(img, np.float32), np.ones(img.shape, np.float32),
                                          np.array([x]), np.array([y]), r_ap, r_in, r_out, 5)
    if not cov[0]:
        raise ValueError("the aperture of the synthetic source is not covered")
    return float(f[0] + s[0] * area[0])


_DISC = {}


def template_in_aperture(comps, in_wcs, r_ap, step=0.4, nbox=7):
    """The release template, smoothed by the input pixel's box, integrated inside the aperture radius (arcsec), unit
    flux: on a fine grid of cells `step` arcsec wide with an anti-aliased edge, the box average over nbox x nbox offsets
    across the pixel (along its own, rotated axes). The PSF build holds this (the box-only table) to the release EEF; the
    value here is position-free, so the input grid's sampling of a smooth beam does not enter it."""
    # keyed on the content (an object id is reused once a worker's earlier sub-group is freed)
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
                total += _template_density(comps, xx.ravel() + de, yy.ravel() + dn).reshape(xx.shape)
        _DISC[key] = float((total / nbox ** 2 * w).sum() * step ** 2)
    return _DISC[key]


PHASES = tuple((a / 3.0, b / 3.0) for b in range(3) for a in range(3))      # input-grid shifts (input pixels) the source is averaged over
# source shifts (output pixels) test (a) averages the real source over: the divisor is the mean over sub-pixel phases
# (`pacs_fit._table_aperture`); one phase of a 3.2 arcsec output grid under a 4 pixel aperture reads up to +0.2 % off it
SOURCE_PHASES = tuple((a / 3.0, b / 3.0) for b in (-1, 0, 1) for a in (-1, 0, 1))


def unit_source(rep, out_wcs, band, xo, yo, half_out, phases=PHASES):
    """The real unit source at output pixel (xo, yo) (zero-origin), averaged over the input grid's sub-pixel
    `phases` (the PSF build's tables are the phase average; one phase of a 3.2 arcsec input grid sampling a 11 arcsec
    beam scatters by +-0.2 % in an aperture sum): returns (A_real, A_template, block, x0, y0), the block
    (2 half_out + 1)^2 of the resampled image (flux fraction per output pixel) with lower-left pixel (x0, y0)."""
    rec, _q, in_wcs, comps = rep
    in_area = pacs_mosaic.pixel_area_arcsec2(in_wcs)
    out_area = pacs_mosaic.pixel_area_arcsec2(out_wcs)
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
    density = _template_density(comps, east, north).reshape(m, m)
    cx, cy = int(np.rint(xo)), int(np.rint(yo))
    x0, y0 = cx - half_out, cy - half_out
    block = 0.0
    for (a, b), (fx, fy) in zip(steps, phases):
        img = density[b:b + n * S, a:a + n * S].reshape(n, S, n, S).mean(axis=(1, 3)) * in_area   # flux fraction per input pixel
        win = in_wcs.deepcopy()
        win.wcs.crpix = win.wcs.crpix - np.array([ix0 + fx, iy0 + fy])
        blk = pacs_mosaic.resample_block(None, SimpleNamespace(shape=img.shape), win, out_wcs, out_area, x0, x0 + 2 * half_out + 1,
                                         y0, y0 + 2 * half_out + 1, layers=(img, np.full(img.shape, 1e-3), np.ones(img.shape)))
        if blk is None or not blk["valid"].all():
            raise ValueError("the synthetic input does not cover the block")
        block = block + blk["image"] / len(phases)
    a_real = _aperture_sum(block, xo - x0, yo - y0, (r_ap / pix_out, r_in / pix_out, r_out / pix_out))
    return a_real, template_in_aperture(comps, in_wcs, r_ap), block, x0, y0


def expected_enclosed(config, record, tile, x, y, mix):
    """The mosaic's expected pre-sky aperture sum of a unit injected source at the output positions (x, y) with
    sub-group fractions `mix` (n, n_sub): sum_k f_k EEF_k A_real_k / A_direct_k, every term recomputed on the
    synthetic real source (no number from the PSF build). Returns (expected (n,), worst |A_direct/EEF - 1|)."""
    psf = tile._psf
    reps = representative_maps(config, record.mosaic_dir, psf.meta)
    eef = [aperture_eef(config, r[0]) for r in reps]
    half_out = int(np.ceil(APERTURES[record.band][2] / tile.pixel_arcsec)) + 4
    out = np.zeros(len(x))
    worst = 0.0
    for i in range(len(x)):
        for k in np.flatnonzero(mix[i] > 0):
            a_real, a_direct, *_ = unit_source(reps[k], psf.out_wcs, record.band, x[i], y[i], half_out)
            out[i] += mix[i, k] * eef[k] * a_real / a_direct
            worst = max(worst, abs(a_direct / eef[k] - 1))
    return out, worst


def check_positions(psf, meta, rep_entry, margin):
    """The check positions of one sub-group (output pixels, zero-origin, inside the mosaic by `margin`): 3 x 3 nodes,
    4 mid-cells between them, 4 at 80 % of the half-extent, and 10/20/30 arcsec inside and outside the footprint edge
    of the representative map (the first of its four edges that lies inside the mosaic), at a fixed sub-pixel phase."""
    ny, nx = psf.shape
    n = psf.enclosed.shape[-1]
    sx = np.linspace(-.5, nx - .5, n)
    sy = np.linspace(-.5, ny - .5, n)
    pts = [(a, b, "node") for b in sy for a in sx]
    mx, my = (sx[:-1] + sx[1:]) / 2, (sy[:-1] + sy[1:]) / 2
    pts += [(a, b, "mid") for b in (my[0], my[-1]) for a in (mx[0], mx[-1])]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    pts += [(cx + s * 0.8 * nx / 2, cy + t * 0.8 * ny / 2, "outer") for s in (-1, 1) for t in (-1, 1)]
    x0, x1, y0, y1 = meta["maps"][rep_entry]["bbox"]
    pix = float(np.mean(np.abs(np.diag(psf.out_wcs.pixel_scale_matrix)))) * 3600.0
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
    """`unit_source` at each of one sub-group's check positions: (A_real, A_real averaged over SOURCE_PHASES, A_direct,
    the (2h+1)^2 patch around the source's nearest pixel) per position (a worker's unit of `normalisation_check`)."""
    out = []
    for x, y, _kind in positions:
        a_real, a_direct, block, _bx, _by = unit_source(rep, out_wcs, band, x, y, half_out)
        a_mean = float(np.mean([unit_source(rep, out_wcs, band, x + fx, y + fy, half_out)[0] for fx, fy in SOURCE_PHASES]))
        out.append((a_real, a_mean, a_direct, block[half_out - h:half_out + h + 1, half_out - h:half_out + h + 1]))
    return out


def normalisation_check(config, record, verbose=True):
    """C2: every sub-group of the band's mosaic, at its check positions (`check_positions`): (a) the real unit source's
    aperture sum, averaged over SOURCE_PHASES (the divisor is a phase average), over the production divisor (`MosaicPSF.enclosed_at` with the weights one-hot on the sub-group)
    within TIE_TOLERANCE of 1 (the real source taken at the release EEF: its aperture sum times EEF / A_direct, so a
    template that integrates a little differently from the EEF file, `tie`, does not enter this test; (d) tests it); (b) the same through `Tile.injected` (the production injection, one-hot) against the
    real one within TIE_TOLERANCE; (c) the pixel sums of the same patch within SUM_TOLERANCE; (d) A_direct / EEF within
    RELEASE_TOLERANCE. Raises ValueError listing the worst failures. Returns the worst deviations."""
    from joblib import Parallel, delayed
    from sesnaimpute.sky.derived import pacs_fit
    tile = pacs_fit.cached_tile(config, record)
    psf = tile._psf
    reps = representative_maps(config, record.mosaic_dir, psf.meta)
    h = tile.injection_half_width()
    half_out = max(h, int(np.ceil(APERTURES[record.band][2] / tile.pixel_arcsec)) + 4)
    r_px = tuple(r / tile.pixel_arcsec for r in APERTURES[record.band])
    worst = dict(divisor=0.0, injected=0.0, sums=0.0, tie=0.0)
    bad = []
    rows = []
    n = 0
    t0 = time.time()
    positions = [check_positions(psf, psf.meta, psf.meta["rep"][k], h + 2) for k in range(len(reps))]
    # a sub-group is tested where it contributes to the mosaic (owner ruling 2026-10-30): its nodes, mid-cells and
    # outer positions only where its own maps carry weight (`map_weights`, before the mean-mixture fallback of an
    # uncovered position); its rim positions, 10-30 arcsec either side of its
    # map's edge, always (Aquila's 12.8 deg P160 mosaic failed the patch sums only at nodes degrees from a sub-group's
    # maps, where its weight is 0 and its table is never used)
    for k in range(len(reps)):
        pts = positions[k]
        x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        w = psf.map_weights(x, y)[:, k] if pts else np.zeros(0)
        positions[k] = [p for p, wk in zip(pts, w) if p[2] == "rim" or wk > 0]
    # the real sources (the costly part) in parallel, one sub-group per task; the injections below on one blank image
    real = Parallel(n_jobs=config.n_jobs)(delayed(_real_sources)(rep, psf.out_wcs, record.band, positions[k], half_out, h)
                                          for k, rep in enumerate(reps))
    reach = int(np.ceil(r_px[2])) + 3
    with tile.blank_image():
        for k, rep in enumerate(reps):
            eef = aperture_eef(config, rep[0])
            one = np.zeros((1, psf.n_sub))
            one[0, k] = 1.0
            for (x, y, kind), (a_real, a_mean, a_direct, real_patch) in zip(positions[k], real[k]):
                norm = eef / a_direct                    # the real source at the release EEF (the template's own shortfall is `tie`)
                a_real, a_mean, real_patch = a_real * norm, a_mean * norm, real_patch * norm
                divisor = float(psf.enclosed_at(np.array([x]), np.array([y]), one)[0])
                cx, cy = int(np.rint(x)), int(np.rint(y))
                with tile.injected(np.array([x]), np.array([y]), FLUX_MJY, mixture=one):
                    win = np.array(tile.image[cy - reach:cy + reach + 1, cx - reach:cx + reach + 1])
                    if win.shape != (2 * reach + 1, 2 * reach + 1):
                        raise ValueError("check position (%.1f, %.1f) too near the mosaic edge" % (x, y))
                    a_inj = _aperture_sum(win, x - cx + reach, y - cy + reach, r_px) / FLUX_MJY
                    patch = np.array(tile.image[cy - h:cy + h + 1, cx - h:cx + h + 1]) / FLUX_MJY
                dev = dict(divisor=a_mean / divisor - 1, injected=a_inj / a_real - 1,
                           sums=patch.sum() / real_patch.sum() - 1, tie=a_direct / eef - 1)
                tol = dict(divisor=TIE_TOLERANCE, injected=TIE_TOLERANCE, sums=SUM_TOLERANCE, tie=RELEASE_TOLERANCE)
                n += 1
                rows.append((k, kind, x, y, dev))
                for key, v in dev.items():
                    worst[key] = max(worst[key], abs(v))
                    if abs(v) > tol[key]:
                        bad.append("sub-group %d %s (%.1f, %.1f) %s %+.4f" % (k, kind, x, y, key, v))
    if verbose:
        print("  normalisation check (real unit source) %s: %d sub-groups, %d positions, %.0f s; worst |deviation|: real / "
              "divisor %.4f, injected / real %.4f, patch sums %.4f, template / release EEF %.4f (bars %.3f, %.3f, %.3f, %.3f)"
              % (record.band, psf.n_sub, n, time.time() - t0, worst["divisor"], worst["injected"], worst["sums"], worst["tie"],
                 TIE_TOLERANCE, TIE_TOLERANCE, SUM_TOLERANCE, RELEASE_TOLERANCE), flush=True)
    if bad:
        raise ValueError("%s: the mosaic's normalisation fails the real-source check at %d of %d tests, e.g. %s"
                         % (record.band, len(bad), 4 * n, "; ".join(bad[:4])))
    normalisation_check.rows = rows
    return worst
