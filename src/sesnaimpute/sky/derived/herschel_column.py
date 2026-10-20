"""Per-source Herschel dust column, A_K, from the HGBS N(H2) maps
(SPEC_PRIORS.md section 1.1): Herschel Gould Belt Survey column density
(36.3" beam) wherever a map covers the source. A_K = 1.12e-22 * N(H2)
(AV_PER_NH2: Bohlin, Savage & Drake 1978; AK_PER_AV: Rieke & Lebofsky
1985, ApJ 288, 618).

The map set is the 23 FITS files `herschel_hgbs.build` fetches. Which
regions a map serves is read from the data -- curated sources sampled
against each map, screened first by its WCS bbox -- never a manifest.
`build` writes one `column`/`source` file per requested region, including
one no map reaches, so a consumer fails only on a missing file.
SIGMA_RAND is survey-wide, from overlapping map pairs' half-difference
(sigma_rand^2 = c0^2 + c1^2 A^2), written to `sigma`/`survey` with each
map's beam FWHM from its FITS header where stated, else 36.3".

The zero point is one systematic per field, measured per field, not one
survey constant from two overlapping map pairs (owner, 2026-09-06). A
"field" here is one HGBS-covered SESNA region. The measurement and the
write into this module's own `sigma`/`survey` product (`FIELD_NAME`/
`ZP_FIELD`/`ZP_SIGMA_FIELD`/`ZP_N_FIELD`; `SIGMA_ZP_K` kept as the RMS of
the per-field values, for a reader that still wants one number) both live
in `sky.derived.column` (`_measure_field_zeropoints`/
`_write_field_zeropoints`), the stage that already holds both arms at
sightline granule where it merges them -- not here, so the measurement
runs in the same pass that applies it, rather than needing a second
product of this same stage first. `column.py`'s merge (`merge_region`)
then subtracts the field's own offset from the Herschel arm's `A_K`: a
measured systematic left unapplied is an error of its own size (owner,
2026-09-06).
"""

import functools
import os
import re
import time

import astropy.units as u
import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_erosion

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.sky.download.herschel_hgbs.build import _FILES as HGBS_FILES

_SIGMA_STEM = "sigma_herschel_survey"
_COLUMN_STEM = "column_herschel_source"

#: `_pair_native` holds two full-resolution HGBS maps at once (up to
#: 22,500x13,000 float32, ~1.17 GB each) plus their eroded masks --
#: about 2.9 GB at the largest pair -- so `config.n_jobs` concurrent
#: pair jobs is bounded here, not left at the general worker count, to
#: keep the pool's total under the 6 GB stage budget (CODING_RULES.md 10a).
PAIR_JOBS_MAX_CONCURRENT = 2

#: N(H2) [cm^-2] -> A_K [mag]. AV_PER_NH2: Bohlin, Savage & Drake 1978
#: gas-to-dust ratio, A_V/N(H2). AK_PER_AV: Rieke & Lebofsky 1985,
#: ApJ 288, 618, A_K/A_V.
AV_PER_NH2 = 1.000e-21
AK_PER_AV = 0.112
NH2_TO_AK = AV_PER_NH2 * AK_PER_AV  # 1.12e-22 mag cm^2

#: The HGBS survey's own quoted beam (Andre et al. 2010, A&A 518, L102).
HERSCHEL_STATED_FWHM_ARCSEC = 36.3

#: Standard header keywords carrying a beam FWHM, degrees or arcsec.
_BEAM_KEYWORDS_DEG = ("BMAJ",)
_BEAM_KEYWORDS_ARCSEC = ("BEAM", "BEAMFWHM", "FWHM", "RESOLUTN", "RESOLUTION")
#: The HGBS maps carry no such keyword; the beam appears instead as free
#: text in a COMMENT card, e.g. "resolution of 36.3 arcsec".
_BEAM_COMMENT_RE = re.compile(r"resolution of\s+([\d.]+)\s*arcsec", re.IGNORECASE)

# The map list and its geometry -- read from the fetched FITS files.

def _map_list(config):
    """[{name, path}] for every map `sesnaimpute.sky.download.herschel_hgbs.
    build` fetches, sorted by name for a deterministic MAP_ID order."""
    hub = os.path.join(config.data_root, "sky", "download", "herschel_hgbs")
    out = []
    for name in sorted(HGBS_FILES):
        path = os.path.join(hub, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "herschel_column.build: HGBS map %r missing at %s -- run "
                "'python -m sesnaimpute.sky.download.herschel_hgbs.build'" % (name, path))
        out.append(dict(name=name, path=path))
    return out

def _map_header(path):
    """The map's WCS footprint bbox, pixel scale and beam FWHM (arcsec,
    `None` if unstated), without loading pixel data -- shape and WCS come
    from header keywords alone."""
    with fits.open(path, memmap=False) as hd:
        chosen = next(c for c in hd if c.header.get("NAXIS", 0) >= 2)
        hdr = chosen.header.copy()
        shape = chosen.shape[-2:]  # trailing two axes: some maps carry
        # degenerate leading Stokes/frequency axes of size 1 (NAXIS3/4).
    wcs = WCS(hdr).celestial
    bbox = _bbox_of(wcs, shape)
    cd = hdr.get("CDELT1", hdr.get("CD1_1"))
    pixscale = abs(float(cd)) * 3600.0
    return dict(bbox=bbox, pixscale_arcsec=pixscale, beam_arcsec=_beam_from_header(hdr))

def _beam_from_header(hdr):
    """A beam FWHM in arcsec from a standard keyword, else the HGBS
    'resolution of NN.N arcsec' COMMENT text; `None` if neither exists."""
    for kw in _BEAM_KEYWORDS_DEG:
        if kw in hdr:
            return float(hdr[kw]) * 3600.0
    for kw in _BEAM_KEYWORDS_ARCSEC:
        if kw in hdr:
            return float(hdr[kw])
    for c in hdr.get("COMMENT", []):
        m = _BEAM_COMMENT_RE.search(str(c))
        if m:
            return float(m.group(1))
    return None

def _open_hgbs_map(path):
    """One HGBS FITS map: 2-D float32 N(H2) data, its celestial WCS, and
    its pixel scale in arcsec."""
    with fits.open(path, memmap=False) as hd:
        chosen = next(c for c in hd if c.header.get("NAXIS", 0) >= 2)
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
    shifting `b`'s RA by the multiple of 360 deg nearest `a` first, so a
    box straddling the 0/360 seam is not falsely ruled out."""
    if a[3] + pad < b[2] - pad or b[3] + pad < a[2] - pad:
        return False
    sh = round((0.5 * (a[0] + a[1]) - 0.5 * (b[0] + b[1])) / 360.0) * 360.0
    b0, b1 = b[0] + sh, b[1] + sh
    return not (a[1] + pad < b0 - pad or b1 + pad < a[0] - pad)

def _size_ordered_desc(maps):
    """`maps` sorted by file size, descending: the largest, most complete
    reduction of a cloud is offered to each source first."""
    def _fsize(m):
        try:
            return os.path.getsize(m["path"])
        except OSError:
            return 0
    return sorted(maps, key=_fsize, reverse=True)

# SIGMA: zero point and random scatter from overlapping-map replicates.

def _pair_native(name_a, path_a, name_b, path_b):
    """The map-to-map SIG_ZP/SIG_RAND replicate: matched native-resolution
    samples of two overlapping HGBS reductions, eroded from each map's
    edge by twice the stated beam to their common, fully-sampled interior."""
    t0 = time.time()
    A, wa, pxa = _open_hgbs_map(path_a)
    B, wb, pxb = _open_hgbs_map(path_b)
    erode_arcsec = 2.0 * HERSCHEL_STATED_FWHM_ARCSEC

    def eroded(M, px):
        # A square structuring element is a horizontal run's Minkowski
        # sum with a vertical run, far cheaper at a large radius.
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
    """SIG_ZP (per-field, fully correlated) and the SIG_RAND^2 = c0^2 +
    c1^2 A^2 coefficients, from the overlapping-map replicates
    `_pair_native` returns: SIG_ZP is the RMS of each pair's own median
    half-difference; c0, c1 fit every pair's column-binned half-difference
    scatter (MAD^2) against the bin's mean column, pooled over all pairs."""
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

def _write_sigma_survey(config, maps, header_by_name, sig):
    """The survey-wide product: every map's own beam FWHM (header value,
    else the survey's 36.3") and the random-term model (C0/C1) the
    per-region column files are computed with. The zero point itself,
    `SIGMA_ZP_K`, is NOT written here -- `sig["sig_zp_ak"]` (the RMS of
    the overlapping-map-pair half-differences) is a diagnostic printed by
    `build` only; the one `SIGMA_ZP_K` this file carries is
    `sky.derived.column._write_field_zeropoints`'s per-field measurement,
    appended into this same file in the pass that follows (module
    docstring). A reader before that pass has run finds no `SIGMA_ZP_K`
    at all, rather than this stage's own superseded number."""
    out_path = config_module.product_path(config, "sky/derived", "herschel", "sigma", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = [m["name"] for m in maps]
    fwhm = np.array([header_by_name[n]["beam_arcsec"] or HERSCHEL_STATED_FWHM_ARCSEC
                      for n in names], dtype=np.float32)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        for name, data in (
            ("MAP_NAME", np.array([n.encode("utf-8") for n in names])),
            ("BEAM_FWHM_ARCSEC", fwhm),
            ("C0", np.float64(sig["rand_c0"])),
            ("C1", np.float64(sig["rand_c1"])),
            ("N_PAIRS", np.int64(sig["n_pairs"])),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_SIGMA_STEM, name)])


# COLUMN: per-source, per-map sampling and the per-region merge.

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
    """One HGBS map: nearest-pixel N(H2) at every source of every
    requested region the map reaches -- the coverage test that decides
    which regions a map serves. `region_sources` is {region: (ra_deg,
    dec_deg)}, bbox-screened per region first. The column build itself
    samples per source block (`_sample_region_block`); this whole-map
    form is what `sky.derived.planck_column._hgbs_region_map` reads for
    the calibration's region labels."""
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


def _sample_region_block(map_entries, ra, dec, bounds):
    """One region, one source block (owner ruling 2026-10-09: pieces over
    source blocks, far more pieces than workers): the nearest-pixel N(H2)
    lookup, converted to A_K, against every candidate map overlapping this
    region, in size-descending order, first match wins. `map_entries` is
    `[(candidate_index, data, wcs, bbox), ...]` for this region, bound into
    this function once (joblib memory-maps the large map arrays to the
    worker); `ra`/`dec` are the region's full coordinate arrays, also
    region-wide read-only inputs; `bounds` is the piece's own `(start,
    stop)` source-index range. Returns `(start, stop, a_k, candidate_id)`,
    `candidate_id` -1 where no map covers the source."""
    start, stop = bounds
    sub_ra = ra[start:stop]
    sub_dec = dec[start:stop]
    nloc = stop - start
    a_k = np.full(nloc, np.nan, dtype=np.float64)
    cand_id = np.full(nloc, -1, dtype=np.int64)
    for ci, data, wcs, box in map_entries:
        fill = cand_id < 0
        if not fill.any():
            break
        loc = np.where(fill & (sub_dec >= box[2] - 0.05) & (sub_dec <= box[3] + 0.05))[0]
        if loc.size == 0:
            continue
        x, y = wcs.world_to_pixel(SkyCoord(sub_ra[loc] * u.deg, sub_dec[loc] * u.deg, frame="icrs"))
        xi = np.rint(x).astype(np.int64)
        yi = np.rint(y).astype(np.int64)
        ny, nx = data.shape
        inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        if not inside.any():
            continue
        loc2 = loc[inside]
        vals = data[yi[inside], xi[inside]]
        on = np.isfinite(vals) & (vals > 0)
        if on.any():
            sel = loc2[on]
            a_k[sel] = vals[on].astype(np.float64) * NH2_TO_AK
            cand_id[sel] = ci
    return start, stop, a_k, cand_id

def _write_region(config, region, a_k, sig_a_k, sig_rand, sig_zp, covered, map_id, map_names):
    """One region's `column`/`source` product, catalogue-row order --
    even a region no map reaches gets one (COVERED all False, MAP_ID -1)."""
    out_path = config_module.product_path(config, "sky/derived", "herschel", "column", "source", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        for name, data in (
            ("A_K", a_k.astype(np.float32)),
            ("SIGMA_A_K", sig_a_k.astype(np.float32)),
            ("SIGMA_RAND_K", sig_rand.astype(np.float32)),
            ("SIGMA_ZP_K", sig_zp.astype(np.float32)),
            ("COVERED", covered),
            ("MAP_ID", map_id.astype(np.int32)),
            ("MAP_NAME", np.array([n.encode("utf-8") for n in map_names])),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_COLUMN_STEM, name)])

def build(config, regions=None):
    """`sigma`/`survey` (beam FWHM and the SIG_ZP/SIG_RAND model, fit over
    the whole map set regardless of `regions`) and `column`/`source`, one
    file per requested region. Maps parallelised with joblib."""
    with progress_module.Stage("sky.derived.herschel_column") as st:
        maps = _map_list(config)
        header_by_name = {m["name"]: _map_header(m["path"]) for m in maps}
        for m in maps:
            m["bbox"] = header_by_name[m["name"]]["bbox"]

        pair_jobs = [(maps[i]["name"], maps[i]["path"], maps[j]["name"], maps[j]["path"])
                     for i in range(len(maps)) for j in range(i + 1, len(maps))
                     if _boxes_overlap(maps[i]["bbox"], maps[j]["bbox"])]
        t_sig0 = time.time()
        pair_results = pieces_module.run(
            lambda p: _pair_native(*p), pair_jobs,
            min(config.n_jobs, PAIR_JOBS_MAX_CONCURRENT))
        sig = _sig_model(pair_results)
        sigma_wall_s = time.time() - t_sig0
        _write_sigma_survey(config, maps, header_by_name, sig)

        if regions is None:
            regions = [r.name for r in regions_module.REGIONS]

        t_sample0 = time.time()
        region_ra_dec = {r: _region_sources(config, r) for r in regions}
        candidate_maps = _size_ordered_desc(
            [m for m in maps if any(
                ra.size and _boxes_overlap(
                    m["bbox"], (float(ra.min()), float(ra.max()), float(dec.min()), float(dec.max())), pad=0.05)
                for ra, dec in region_ra_dec.values())])

        # Regions run one after another (owner ruling 2026-10-09); inside a
        # region the source axis -- the piece whose cost scales with the
        # region (Cygnus X's catalogue is 1000x NGC 7129's) -- is cut into
        # pieces and run through the one shared pool. Each candidate map is
        # opened once per region, in this parent process, and bound into the
        # piece function as a region-wide read-only input (joblib memory-maps
        # it to the workers); a piece carries only its source-index bounds.
        written = []
        n_regions = len(regions)
        for i, region in enumerate(regions):
            ra, dec = region_ra_dec[region]
            n = ra.size
            a_k = np.full(n, np.nan, dtype=np.float64)
            map_id = np.full(n, -1, dtype=np.int32)
            serving_names = []
            if n:
                rbox = (float(ra.min()), float(ra.max()), float(dec.min()), float(dec.max()))
                region_map_idx = [ci for ci, m in enumerate(candidate_maps)
                                   if _boxes_overlap(m["bbox"], rbox, pad=0.05)]
                map_entries = []
                for ci in region_map_idx:
                    data, wcs, _px = _open_hgbs_map(candidate_maps[ci]["path"])
                    map_entries.append((ci, data, wcs, candidate_maps[ci]["bbox"]))
                fn = functools.partial(_sample_region_block, map_entries, ra, dec)
                blocks = pieces_module.run(fn, pieces_module.slices(n, config.n_jobs), config.n_jobs)
                cand_id = np.full(n, -1, dtype=np.int64)
                for start, stop, a_k_blk, cand_id_blk in blocks:
                    a_k[start:stop] = a_k_blk
                    cand_id[start:stop] = cand_id_blk
                # Compact to the maps actually used (size-descending, first
                # match wins -- identical to the per-source candidate order
                # above), so MAP_ID indexes only the serving maps, same as
                # before the within-region piece split.
                used = sorted(set(int(c) for c in cand_id[cand_id >= 0]))
                for k, c in enumerate(used):
                    map_id[cand_id == c] = k
                serving_names = [candidate_maps[c]["name"] for c in used]
            covered = map_id >= 0
            sig_rand = np.where(covered, np.sqrt(sig["rand_c0"] ** 2 + (sig["rand_c1"] * a_k) ** 2), np.nan)
            sig_zp = np.where(covered, sig["sig_zp_ak"], np.nan)
            sig_a_k = np.sqrt(sig_rand ** 2 + sig_zp ** 2)
            _write_region(config, region, a_k, sig_a_k, sig_rand, sig_zp, covered, map_id,
                           serving_names)
            frac = float(covered.mean()) if n else float("nan")
            med = float(np.median(a_k[covered])) if covered.any() else float("nan")
            print("herschel_column REGION %-16s n=%6d covered=%.3f median_A_K=%.3f maps=%d"
                  % (region, n, frac, med, len(serving_names)), flush=True)
            written.append(region)
            st.tick(i + 1, n_regions, "regions")
        sample_wall_s = time.time() - t_sample0

        print("herschel_column SIGMA sig_zp_ak=%.4f c0=%.4f c1=%.4f n_pairs=%d"
              % (sig["sig_zp_ak"], sig["rand_c0"], sig["rand_c1"], sig["n_pairs"]), flush=True)
        for m in maps:
            h = header_by_name[m["name"]]
            found = "%.2f" % h["beam_arcsec"] if h["beam_arcsec"] else "none"
            print("herschel_column BEAM %-45s header=%s used=%.2f"
                  % (m["name"], found, h["beam_arcsec"] or HERSCHEL_STATED_FWHM_ARCSEC), flush=True)

        st.done(None, regions=n_regions, sig_zp_ak=float(sig["sig_zp_ak"]))
        return dict(sigma_wall_s=sigma_wall_s, sample_wall_s=sample_wall_s,
                    regions_written=written, sig_model=sig)

if __name__ == "__main__":
    run(build)
