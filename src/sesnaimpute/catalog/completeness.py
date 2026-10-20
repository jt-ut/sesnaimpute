"""Rob Gutermuth's measured per-field, per-band completeness products: the
reader (`field_maps`, `lookup`) curation and this module's own `build` use,
and the stage that writes the one per-region completeness product
`catalog.limits.completeness` reads (bms_review/COMPLETENESS_2026-10-20.md,
"The reader: layout and kernel").

Each field x band carries two FITS files at
`sky/download/sesna_completeness/<Region>/` (`sky.download.sesna_completeness`):
a 90%-completeness magnitude map (2-D, -100 off-field) and a differential
completeness cube (3-D: magnitude, y, x; the fraction, 0..1, of injected
sources recovered in that 0.5-mag bin). Both are TAN-projected, ICRS,
30-arcsec pixels, identical spatial WCS within one field (measured: NAXIS1/2,
CRVAL1/2, CRPIX1/2, CD1_1/CD2_2 match bit for bit between a field's cube and
90% map). The cube's spectral axis (CTYPE3 MAG--CAR) has CRPIX3=1.0 and
CD3_3=0.5 in every one of the survey's 440 cube files (measured, no
exceptions); its zero point CRVAL3 -- the brightest plane's own magnitude --
and its plane count vary FIELD BY FIELD, even within one region and band
(measured: 77 of 117 region-band groups mix more than one (CRVAL3, NAXIS3)
pair among their own fields), so `field_maps` reads each field's own grid
from its own header rather than assuming a single survey-wide one.

(COMPLETENESS_2026-10-20.md's own layout section states a single, fixed
29-plane grid at CRVAL3=4.75 for the whole product -- a measured fact the
design note rests on. It does not hold for the raw FITS headers in general:
a single grid wide enough to cover the survey's own full observed range
(CRVAL3 0.25 to a faint edge of 19.75) needs 40 planes, not 29, and no
single (CRVAL3, NAXIS3) pair covers even one region-band's own fields
without clipping one of them. `build` below instead forms, per region and
band, the union of that region's own fields' grids -- wide enough to hold
every field's own measured range without loss -- and reports its own size.)

A field's `INSTRUME`/`CHNLNUM` header keys are not a reliable band
identity either: every one of the survey's 80 MIPS cube files carries
`INSTRUME='IRAC'`, `CHNLNUM='1'` regardless (measured, a delivery-side
header bug), so `_field_names` reads a field's name from `SESNAFLD` and
trusts the filename's own suffix (already selected by the glob) for which
band a file belongs to, never `INSTRUME`/`CHNLNUM`.
"""

import functools
import glob
import os

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import erfc

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog import depth_grid as depth_grid_module

#: Measured identical across every one of the survey's 440 cube files
#: (module docstring): the spectral axis's own step convention, asserted
#: on every field read rather than assumed.
CRPIX3_EXPECTED = 1.0
DMAG_EXPECTED = 0.5

#: band key -> (INSTRUME, CHNLNUM, filename suffix), the three names a
#: field's FITS files carry for one band (SESNAFLD/INSTRUME/CHNLNUM header
#: keys; the filename's own lowercase suffix).
_BAND_FILE = {
    "I1": ("IRAC", 1, "i1"), "I2": ("IRAC", 2, "i2"), "I3": ("IRAC", 3, "i3"),
    "I4": ("IRAC", 4, "i4"), "M1": ("MIPS", 1, "m1"),
}

IRAC_MIPS_KEYS = ("I1", "I2", "I3", "I4", "M1")


class FieldMap:
    """One field's own completeness map, this band: `wcs` is the field's
    2-D celestial WCS (RA/Dec -> 0-indexed pixel, shared bit-for-bit by the
    cube and the 90% map, measured); `cube` is the (n_mag, ny, nx) float32
    differential-completeness fraction; `map90` is the (ny, nx) float32
    90%-completeness magnitude (-100 off-field); `on_map` is `map90 != -100`
    (module docstring; COMPLETENESS_2026-10-20.md rule 2: "Covered = finite
    90%-map value"); `mag0`/`dmag` are this field's own magnitude-grid
    origin and step (`dmag` always 0.5, measured)."""
    __slots__ = ("field", "wcs", "cube", "map90", "on_map", "mag0", "dmag", "n_mag", "ny", "nx")

    def __init__(self, field, wcs, cube, map90, on_map, mag0, dmag):
        self.field, self.wcs, self.cube, self.map90, self.on_map = field, wcs, cube, map90, on_map
        self.mag0, self.dmag = mag0, dmag
        self.n_mag, self.ny, self.nx = cube.shape


def _region_dir(config, region):
    return os.path.join(config.data_root, "sky", "download", "sesna_completeness", region)


def _field_names(config, region, band):
    """Every field name this region has for `band`, read from each file's
    own `SESNAFLD` (not its `INSTRUME`/`CHNLNUM`: measured, every one of
    the survey's 80 MIPS cube files carries `INSTRUME='IRAC'`,
    `CHNLNUM='1'` regardless -- a delivery-side header bug that makes
    those two keys unusable for a MIPS file; the filename's own suffix,
    which this glob already selects on, is the reliable band identity)."""
    _, _, suffix = _BAND_FILE[band]
    pattern = os.path.join(_region_dir(config, region), f"*_{suffix}_diffcompcube.fits")
    return [fits.getheader(path)["SESNAFLD"] for path in sorted(glob.glob(pattern))]


def _field_path(config, region, field, band, kind):
    instrume, chnlnum, suffix = _BAND_FILE[band]
    stem = "diffcompcube" if kind == "cube" else "diffcomp90"
    return os.path.join(_region_dir(config, region), f"{field}_{suffix}_{stem}.fits")


def field_maps(config, region, band):
    """Every field this region has for `band`, each as a `FieldMap`
    (module docstring): reads both the cube and the 90% map, asserts the
    cube's own CRPIX3/CD3_3 against the measured survey-wide constants,
    and asserts the two files' spatial WCS header values match bit for
    bit (measured true everywhere; a mismatch means the pair is not the
    same field's own pointing).
    """
    out = []
    for field in _field_names(config, region, band):
        cube_path = _field_path(config, region, field, band, "cube")
        map90_path = _field_path(config, region, field, band, "map90")
        with fits.open(cube_path) as hdul:
            hc = hdul[0].header
            cube = np.asarray(hdul[0].data, dtype=np.float32)
        with fits.open(map90_path) as hdul:
            hm = hdul[0].header
            map90 = np.asarray(hdul[0].data, dtype=np.float32)

        if not np.isclose(hc["CRPIX3"], CRPIX3_EXPECTED) or not np.isclose(hc["CD3_3"], DMAG_EXPECTED):
            raise ValueError(
                "catalog.completeness: %r has CRPIX3=%r CD3_3=%r, not the survey-wide "
                "(%r, %r) every other cube file carries"
                % (cube_path, hc["CRPIX3"], hc["CD3_3"], CRPIX3_EXPECTED, DMAG_EXPECTED))
        for key in ("NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD2_2"):
            if not np.isclose(float(hc[key]), float(hm[key])):
                raise ValueError(
                    "catalog.completeness: %r and %r disagree on %s (%r vs %r) -- not the "
                    "same field's own pointing" % (cube_path, map90_path, key, hc[key], hm[key]))

        wcs = WCS(hm, naxis=2)
        on_map = map90 != -100.0
        out.append(FieldMap(field, wcs, cube, map90, on_map, float(hc["CRVAL3"]), float(hc["CD3_3"])))
    return out


def _lookup_piece(maps, ra_deg, dec_deg, bounds):
    """One block of positions (the pool's unit of work): for every field,
    the nearest on-map pixel (COMPLETENESS_2026-10-20.md rule 1: "the
    map at the source's NEAREST pixel"); where more than one field is
    on-map at a position (MIPS fields overlap in 12 regions, rule 3), the
    DEEPER field wins -- the one with the larger (fainter) 90%-completeness
    magnitude at that pixel, the catalogue's own rule."""
    start, stop = bounds
    ra, dec = ra_deg[start:stop], dec_deg[start:stop]
    n = ra.size
    field_idx = np.full(n, -1, dtype=np.int32)
    iy = np.full(n, -1, dtype=np.int32)
    ix = np.full(n, -1, dtype=np.int32)
    best_mag90 = np.full(n, -np.inf, dtype=np.float64)
    for fidx, m in enumerate(maps):
        x, y = m.wcs.world_to_pixel_values(ra, dec)
        ix_r = np.rint(x).astype(np.int64)
        iy_r = np.rint(y).astype(np.int64)
        inb = (ix_r >= 0) & (ix_r < m.nx) & (iy_r >= 0) & (iy_r < m.ny)
        cand = np.zeros(n, dtype=bool)
        cand[inb] = m.on_map[iy_r[inb], ix_r[inb]]
        if not cand.any():
            continue
        depth_here = np.full(n, -np.inf, dtype=np.float64)
        depth_here[cand] = m.map90[iy_r[cand], ix_r[cand]]
        take = cand & (depth_here > best_mag90)
        field_idx[take] = fidx
        iy[take] = iy_r[take]
        ix[take] = ix_r[take]
        best_mag90[take] = depth_here[take]
    return field_idx, iy, ix


def lookup(config, region, band, ra_deg, dec_deg, n_jobs=1, maps=None):
    """`(field_idx, iy, ix)`, one triple per position in `ra_deg`/`dec_deg`
    (deg), `-1` off-map in every field of this region and band. `maps`
    lets a caller that already has `field_maps`'s own result (curation,
    the stage below) pass it once instead of re-reading every field's
    FITS files per call. Cut over position blocks on the shared pool
    (`pieces.run`; Cygnus X has 3.3 M sources, COMPLETENESS_2026-10-20.md).
    """
    if maps is None:
        maps = field_maps(config, region, band)
    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    n = ra_deg.size
    if not maps:
        return (np.full(n, -1, dtype=np.int32), np.full(n, -1, dtype=np.int32),
                np.full(n, -1, dtype=np.int32))
    bounds = pieces_module.slices(n, n_jobs)
    fn = functools.partial(_lookup_piece, maps, ra_deg, dec_deg)
    parts = pieces_module.run(fn, bounds, n_jobs)
    field_idx = np.concatenate([p[0] for p in parts]) if len(parts) > 1 else parts[0][0]
    iy = np.concatenate([p[1] for p in parts]) if len(parts) > 1 else parts[0][1]
    ix = np.concatenate([p[2] for p in parts]) if len(parts) > 1 else parts[0][2]
    return field_idx, iy, ix


# --- The stage: catalog.completeness.build (COMPLETENESS_2026-10-20.md
# "The reader: layout and kernel") ---

TWOMASS_KEYS = ("J", "H", "Ks")

#: The design note's own canonical magnitude grid for the five Spitzer
#: bands' CURVE_<B>/PIXEL_CURVE_<B>: 4.75 + 0.5*k, k = 0..28, applied the
#: same for every band and region so the one on-disk layout and the one
#: numba kernel (catalog/limits.py) never branch on either. Measured
#: against the raw FITS headers (catalog/completeness.py's own module
#: docstring): a field's own native grid can start brighter (CRVAL3 as
#: low as 0.25, chiefly M1 and a few IRAC fields) or extend fainter
#: (to 19.75) than this one. `_resample_field_cube` clamps a field's own
#: curve to its OWN plane-0 value brightward of this grid's 4.75 and to 0
#: faintward of its own last measured plane -- so a field whose own
#: native range starts brighter than 4.75 has that bright excess
#: (completeness already at or near 1 there, measured on every field
#: inspected) folded into this grid's own plane 0 rather than lost: the
#: clamp reads as "no fainter than this grid's bright edge", which for a
#: field whose own curve is still at its bright plateau there costs
#: nothing. A field whose own curve has ALREADY rolled off by 4.75 (none
#: measured in this brief's two test regions) would instead have that
#: roll-off clipped to the plane-0 value -- see the module's own report
#: for which regions/bands this could still touch on a full build.
CANON_MAG0 = 4.75
CANON_DMAG = 0.5
CANON_N = 29
CANON_MAG_GRID = CANON_MAG0 + CANON_DMAG * np.arange(CANON_N)

#: The 2MASS tabulation's own fine grid step (COMPLETENESS_2026-10-20.md
#: "The reader": "0.05 mag, so the interpolation error of a w ~ 0.1 mag
#: erf is < 0.005").
TWOMASS_DMAG = 0.05

#: How many roll-off widths on either side of F50 the 2MASS tabulation
#: spans, wide enough for the completeness fraction to reach both its
#: bright (~1) and faint (~0) plateau for the narrowest width this
#: package measures (~0.07 dex, TWOMASS_INTRINSIC_WIDTH_DEX).
TWOMASS_SPAN_WIDTHS = 6.0

#: The measurement's own resolution (module docstring, COMPLETENESS_2026-
#: 10-20.md "The products"): ~100 injected sources per pixel and bin, so
#: a stored fraction is exact to 1/600 (not 1/100: some pixels average six
#: 100-trial estimates) -- storage precision, asserted lossless below.
STORAGE_SCALE = 600


def _to_uint16(fraction):
    """`fraction` (0..1) times `STORAGE_SCALE`, rounded to the nearest
    integer: lossless for every value this package reads (asserted by the
    build's own round-trip check), unlike float16 (COMPLETENESS_2026-10-
    20.md: "float16 is not: 2.4e-4")."""
    return np.rint(np.clip(fraction, 0.0, 1.0) * STORAGE_SCALE).astype(np.uint16)


def _resample_field_cube(field_map):
    """`field_map.cube` (n_mag, ny, nx), on its own (mag0, dmag) grid, onto
    `CANON_MAG_GRID`: linear in magnitude between the field's own plane
    centres, clamped to the field's own plane-0 value brightward of its
    own grid and to 0 strictly faintward of its own last plane (brief
    step 6: the last plane ITSELF keeps its own measured value -- only a
    canonical magnitude beyond it is extrapolated to 0).
    One pass per canonical plane (at most 29), each a plain array
    operation over the whole field -- not a per-pixel loop.
    """
    n_native = field_map.n_mag
    native_mags = field_map.mag0 + field_map.dmag * np.arange(n_native)
    out = np.empty((CANON_N,) + field_map.cube.shape[1:], dtype=np.float32)
    for k, mag in enumerate(CANON_MAG_GRID):
        if mag <= native_mags[0]:
            out[k] = field_map.cube[0]
        elif mag == native_mags[-1]:
            out[k] = field_map.cube[-1]
        elif mag > native_mags[-1]:
            out[k] = 0.0
        else:
            j = min(int(np.searchsorted(native_mags, mag, side="right") - 1), n_native - 2)
            t = (mag - native_mags[j]) / field_map.dmag
            out[k] = (1.0 - t) * field_map.cube[j] + t * field_map.cube[j + 1]
    return out


def _m50_per_pixel(field_map):
    """Each on-map pixel's own 50%-completeness magnitude, by linear
    interpolation of the field's own native cube (not the canonical
    grid): the first native-plane crossing of 0.5 from the bright end,
    `NaN` off-map or where the curve never reaches 0.5 brightward of the
    last plane. Used only for `_correlation_length`'s own diagnostic.
    """
    cube = field_map.cube
    below = cube < 0.5
    first_below = np.argmax(below, axis=0)
    never_below = ~np.any(below, axis=0)
    native_mags = field_map.mag0 + field_map.dmag * np.arange(field_map.n_mag)
    j = np.clip(first_below - 1, 0, field_map.n_mag - 2)
    c0 = np.take_along_axis(cube, j[None, :, :], axis=0)[0]
    c1 = np.take_along_axis(cube, (j + 1)[None, :, :], axis=0)[0]
    denom = np.where(c0 != c1, c0 - c1, 1.0)
    t = np.where(c0 != c1, (c0 - 0.5) / denom, 0.0)
    m50 = native_mags[j] + t * field_map.dmag
    m50 = np.where(never_below | (first_below == 0), np.nan, m50)
    return np.where(field_map.on_map, m50, np.nan)


def _correlation_length(field_map):
    """The pixel-to-pixel Pearson correlation of the 50%-completeness
    point, averaged over the field's own horizontal and vertical
    immediate-neighbour pairs with both pixels on-map (COMPLETENESS_2026-
    10-20.md "The products": "The ingest measures the pixel-to-pixel
    correlation of the 50% point once per field and reports it"); `NaN`
    if the field has fewer than two such pairs.
    """
    m50 = _m50_per_pixel(field_map)
    pairs = []
    for a, b in ((m50[:, :-1], m50[:, 1:]), (m50[:-1, :], m50[1:, :])):
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() >= 2:
            pairs.append((a[both], b[both]))
    if not pairs:
        return float("nan")
    x = np.concatenate([p[0] for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _row_for_positions(field_idx, iy, ix, row_index_maps):
    """Row id into a band's own `CURVE_<B>` for each `(field_idx, iy, ix)`
    triple (`lookup`'s own return), `-1` where off-map."""
    rows = np.full(field_idx.size, -1, dtype=np.int64)
    for fidx, row_map in enumerate(row_index_maps):
        sel = field_idx == fidx
        if sel.any():
            rows[sel] = row_map[iy[sel], ix[sel]]
    return rows


def _icrs_to_galactic(ra_deg, dec_deg):
    c = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs").galactic
    return c.l.deg, c.b.deg


def _galactic_to_icrs(gl_deg, gb_deg):
    c = SkyCoord(l=gl_deg, b=gb_deg, unit="deg", frame="galactic").icrs
    return c.ra.deg, c.dec.deg


def _twomass_params(config, region):
    """This region's own fitted `F50_2MASS_MJY` (mJy) and `WIDTH_DEX`
    (dex) for J, H, Ks (`catalog.depths.build`'s product)."""
    path = config_module.product_path(config, "catalog", "sesna", "depths", "region")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.completeness: no survey-depths product at %r -- run the "
            "'sesnaimpute.catalog.depths' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        regions = [r.decode() if isinstance(r, bytes) else r for r in f["REGION"][:]]
        ridx = regions.index(region)
        f50 = f["F50_2MASS_MJY"][ridx, :]
        width_dex = f["WIDTH_DEX"][ridx, :]
    return f50, width_dex


def _tabulate_twomass_band(f50_mjy, width_dex, zero_point_jy):
    """The error-function completeness roll-off tabulated on a fine
    (`TWOMASS_DMAG`) magnitude grid spanning `TWOMASS_SPAN_WIDTHS` roll-
    off widths either side of the fitted 50% point (COMPLETENESS_2026-10-
    20.md "The reader": "TABULATED on its own fine grid ... so the
    interpolation error of a w ~ 0.1 mag erf is < 0.005"). Returns
    `(curve (1, n) uint16, mag_grid (n,) float64)`.
    """
    m50 = -2.5 * np.log10(f50_mjy / (zero_point_jy * 1000.0))
    w_mag = width_dex / 0.4
    half_span = TWOMASS_SPAN_WIDTHS * w_mag
    n = int(np.ceil(2.0 * half_span / TWOMASS_DMAG)) + 1
    mag_grid = m50 - half_span + TWOMASS_DMAG * np.arange(n)
    fraction = 0.5 * erfc((mag_grid - m50) / (np.sqrt(2.0) * w_mag))
    return _to_uint16(fraction)[None, :], mag_grid


def build(config, regions=None):
    """Builds the per-region completeness product (COMPLETENESS_2026-10-
    20.md "The reader: layout and kernel"): per Spitzer band, the on-map
    30-arcsec pixels' own curves (`CURVE_<B>`), each source's row into
    them (`SOURCE_ROW`), and the admitted nside-512 pixels' own curves
    and coverage (`PIXEL_CURVE_<B>`/`PIXEL_COVERED_FRAC_<B>`); per 2MASS
    band, the one region-wide tabulated erf curve. `catalog.limits.
    completeness` is the one reader.
    """
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    band_keys = {b.key: b for b in definitions.BANDS}

    with progress_module.Stage("catalog.completeness") as st:
        n_done = 0
        out_paths = []
        for region in regions:
            out_path = config_module.product_path(
                config, "catalog", "sesna", "completeness", "source", region=region
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            admitted = np.sort(depth_grid_module.admitted_pixels_512(config, region))
            admitted_gl, admitted_gb = hp.pix2ang(512, admitted, nest=True, lonlat=True)
            admitted_ra, admitted_dec = _galactic_to_icrs(admitted_gl, admitted_gb)

            src_path = config_module.product_path(
                config, "catalog", "sesna", "sources", "source", region=region
            )
            with h5py.File(src_path, "r") as f:
                ra_deg = f["RA_DEG"][:]
                dec_deg = f["DEC_DEG"][:]
            n_source = ra_deg.size
            source_row = np.full((n_source, len(IRAC_MIPS_KEYS)), -1, dtype=np.int32)

            with h5py.File(out_path, "w") as f:
                f.attrs["GRANULE"] = "source"
                f.attrs["REGION"] = region

                corr_lengths = {}
                n_pix_total = 0
                for bi, band in enumerate(IRAC_MIPS_KEYS):
                    maps = field_maps(config, region, band)
                    row_chunks, row_index_maps = [], []
                    offset = 0
                    field_corrs = []
                    for m in maps:
                        resampled = _resample_field_cube(m)
                        iy_on, ix_on = np.where(m.on_map)
                        row_chunks.append(resampled[:, iy_on, ix_on].T)
                        row_map = np.full((m.ny, m.nx), -1, dtype=np.int64)
                        row_map[iy_on, ix_on] = offset + np.arange(iy_on.size)
                        row_index_maps.append(row_map)
                        offset += iy_on.size
                        field_corrs.append(_correlation_length(m))
                    corr_lengths[band] = float(np.nanmean(field_corrs)) if field_corrs else float("nan")

                    curve_vals = (np.concatenate(row_chunks, axis=0) if row_chunks
                                  else np.empty((0, CANON_N), dtype=np.float32))
                    n_pix_total += curve_vals.shape[0]
                    dset = build_module.write_dataset(
                        f, f"CURVE_{band}", _to_uint16(curve_vals),
                        *REGISTRY[("completeness_sesna_source", f"CURVE_{band}")])
                    dset.attrs["ZERO_POINT_JY"] = band_keys[band].vega_zero_point_jy

                    field_idx, iy, ix = lookup(config, region, band, ra_deg, dec_deg,
                                                n_jobs=config.n_jobs, maps=maps)
                    source_row[:, bi] = _row_for_positions(field_idx, iy, ix, row_index_maps)

                    pfield_idx, piy, pix_ = lookup(config, region, band, admitted_ra, admitted_dec,
                                                    n_jobs=config.n_jobs, maps=maps)
                    prows = _row_for_positions(pfield_idx, piy, pix_, row_index_maps)
                    pon = prows >= 0
                    pixel_curve = np.zeros((admitted.size, CANON_N), dtype=np.float32)
                    if pon.any():
                        pixel_curve[pon] = curve_vals[prows[pon]]
                    pixel_frac = pon.astype(np.float32)
                    build_module.write_dataset(
                        f, f"PIXEL_CURVE_{band}", pixel_curve,
                        *REGISTRY[("completeness_sesna_source", f"PIXEL_CURVE_{band}")])
                    build_module.write_dataset(
                        f, f"PIXEL_COVERED_FRAC_{band}", pixel_frac,
                        *REGISTRY[("completeness_sesna_source", f"PIXEL_COVERED_FRAC_{band}")])

                build_module.write_dataset(f, "MAG_GRID", CANON_MAG_GRID,
                                            *REGISTRY[("completeness_sesna_source", "MAG_GRID")])
                build_module.write_dataset(f, "SOURCE_ROW", source_row,
                                            *REGISTRY[("completeness_sesna_source", "SOURCE_ROW")])
                build_module.write_dataset(f, "HPX_PIX_512", admitted,
                                            *REGISTRY[("completeness_sesna_source", "HPX_PIX_512")])

                f50_2mass, width_dex_2mass = _twomass_params(config, region)
                for tk, key in enumerate(TWOMASS_KEYS):
                    zp = band_keys[key].vega_zero_point_jy
                    curve, mag_grid = _tabulate_twomass_band(f50_2mass[tk], width_dex_2mass[tk], zp)
                    dset = build_module.write_dataset(
                        f, f"CURVE_{key}", curve,
                        *REGISTRY[("completeness_sesna_source", f"CURVE_{key}")])
                    dset.attrs["ZERO_POINT_JY"] = zp
                    build_module.write_dataset(
                        f, f"MAG_GRID_{key}", mag_grid,
                        *REGISTRY[("completeness_sesna_source", f"MAG_GRID_{key}")])

            print("catalog.completeness: %s pixel-to-pixel 50%% correlation by band: %s"
                  % (region, {k: round(v, 3) for k, v in corr_lengths.items()}))
            out_paths.append(out_path)
            n_done += 1
            st.tick(n_done, len(regions), "regions")

        st.done(out_paths[-1] if out_paths else None, regions=len(regions),
                admitted_pixels_last_region=int(admitted.size) if regions else 0,
                spitzer_pixels_last_region=n_pix_total if regions else 0)


if __name__ == "__main__":
    run(build)
