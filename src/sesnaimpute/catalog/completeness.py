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

(COMPLETENESS_2026-10-20.md's own layout section originally stated a
single, fixed 29-plane grid at CRVAL3=4.75 for the whole product -- read
off four files, not measured survey-wide, a mistake this module's own
first pass caught and that file's own "Corrections 2026-10-20" section
replaced: ONE canonical grid PER BAND, spanning that band's own
survey-wide range (`BAND_MAG_RANGE` below, confirmed independently here
over all 440 cube files), each field placed onto it by EXACT plane
alignment -- CRPIX3=1.0/CD3_3=0.5 everywhere means a field's own grid
always shares the band grid's phase, so no interpolation is needed or
done. A field whose own cube is cut off before reaching 0 (97 of 440,
Rob cannot regenerate them) is completed along the band's own measured
roll-off template, `_measure_band_template`/`_complete_field` below.)

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
import numba
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import erfc
# healpy's own import, AFTER scipy.special (catalog/depths.py's own
# module docstring: importing healpy before `scipy.special.erfc` corrupts
# loky's pickling of that ufunc across worker processes, measured, import
# order the only variable reproducing it).
import healpy as hp

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

IRAC_KEYS = ("I1", "I2", "I3", "I4")
IRAC_MIPS_KEYS = IRAC_KEYS + ("M1",)   # M1 last: the build masks it to the IRAC union the four have established

#: All eight bands, `definitions.BANDS`' own fixed order (J, H, Ks, I1-I4,
#: M1): the one column order `completeness_sesna_source`'s own SOURCE_ROW
#: shares with `fittp.sweep`'s `BAND_KEYS` (same tuple, same order), so a
#: band's position in `definitions.BANDS` is directly its own column --
#: no per-band index lookup needed at either end.
ALL_BAND_KEYS = tuple(b.key for b in definitions.BANDS)


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

#: Each admitted nside-512 pixel's own `PIXEL_CURVE_<B>` and
#: `PIXEL_COVERED_FRAC_<B>` are the AREA AVERAGE over the pixel's own
#: footprint (COMPLETENESS_2026-10-20.md "The reader": "per admitted
#: nside-512 pixel the area-averaged curve, from the 30'' pixels inside
#: it", and `PIXEL_COVERED_FRAC_<B>` its on-map share): the pixel is
#: supersampled at this many nested children per side (16: 256 points,
#: the SAME density `sky.derived.coverage` measures the atlas's COVERAGE
#: with, so this product's covered fraction and that product's FRAC are
#: one number), every point looked up at its nearest 30-arcsec pixel
#: (rule 1), the curve averaged over the on-map points, the fraction the
#: on-map share. The centre sample the 2026-10-21 code took instead (one
#: lookup at the pixel's centre) zeroed every partly covered pixel whose
#: centre fell off-map, and with it N_CAT_GAL/N_CAT_H2S/N_CAT_PAHN in
#: about half of every region's pixels (DEFECT_ATLAS_NCAT_ZEROS_2026-10-22).
PIXEL_SUPERSAMPLE_PER_SIDE = 16


def _admitted_sample_points(admitted):
    """`(ra_deg, dec_deg, n_sub)`: the nested nside-(512 x
    `PIXEL_SUPERSAMPLE_PER_SIDE`) children of every admitted pixel, pixel-
    major with `n_sub` points per pixel, in ICRS -- the one sample set
    `build` evaluates each band's own footprint and curves on."""
    n_sub = PIXEL_SUPERSAMPLE_PER_SIDE ** 2
    hi = (np.asarray(admitted, dtype=np.int64)[:, None] * n_sub
          + np.arange(n_sub, dtype=np.int64)[None, :]).ravel()
    gl, gb = hp.pix2ang(512 * PIXEL_SUPERSAMPLE_PER_SIDE, hi, nest=True, lonlat=True)
    ra, dec = _galactic_to_icrs(gl, gb)
    return ra, dec, n_sub


TWOMASS_KEYS = ("J", "H", "Ks")

#: Corrections 2026-10-20 (COMPLETENESS_2026-10-20.md "Corrections", item
#: 1): a single fixed 29-plane, CRVAL3=4.75 grid for every band was read
#: off four files, not measured survey-wide; the real grid is ONE PER
#: BAND, spanning that band's own survey-wide range -- independently
#: confirmed here over every one of the survey's 440 cube files (first
#: plane / last plane / plane count): I1 4.75..19.75 (31), I2 3.75..18.75
#: (31), I3 1.25..16.75 (32), I4 1.75..16.25 (30), M1 0.25..12.25 (25).
#: CRPIX3=1.0 and CD3_3=0.5 hold everywhere (measured, no exceptions), so
#: every field's own native grid shares this grid's exact phase: a
#: field's own plane k lands at this grid's own index `offset + k` for an
#: EXACT integer `offset`, never a fraction -- `_place_field_cube` below
#: places it there directly, no interpolation, ever.
BAND_MAG_RANGE = {
    "I1": (4.75, 19.75), "I2": (3.75, 18.75), "I3": (1.25, 16.75),
    "I4": (1.75, 16.25), "M1": (0.25, 12.25),
}
CANON_DMAG = 0.5


def _band_mag_grid(band):
    """This band's own canonical magnitude grid (`BAND_MAG_RANGE`,
    `CANON_DMAG`-spaced)."""
    lo, hi = BAND_MAG_RANGE[band]
    n = int(round((hi - lo) / CANON_DMAG)) + 1
    return lo + CANON_DMAG * np.arange(n)

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


# --- The kernel (COMPLETENESS_2026-10-20.md "The reader: layout and
# kernel"): used by every consumer through `catalog.limits.completeness`'s
# reader object, never called directly outside this module and that one. ---

#: OWNER DECISION 2026-10-20 (COMPLETENESS_2026-10-20.md "The reader"):
#: the measurement's own resolution -- about 100 injected-source trials
#: per pixel and magnitude bin -- means a stored completeness of 1.00
#: (every trial recovered) still carries a non-detection probability
#: below ~1/200, not 0. `ln_one_minus_completeness` floors its result at
#: `LN_ONE_MINUS_C_FLOOR` so a template that predicts a bright flux in an
#: empty but COVERED band is heavily disfavoured by the fitter, never
#: given an infinite (impossible) penalty: a single empty band must
#: never veto an otherwise-good fit by itself.
C_CAP = 1.0 - 1.0 / 200.0
LN_ONE_MINUS_C_FLOOR = float(np.log(1.0 - C_CAP))  # ln(0.005) = -5.2983...


@numba.njit(cache=True)
def completeness(curve_row, m0, dm, n, log10_f, a, c):
    """`idx = a * log10_f + c` (a fractional plane index into `curve_row`,
    uint16, value x `STORAGE_SCALE`); `idx <= 0` clamps to `curve_row[0]`
    (brightward of the grid); `idx >= n - 1` returns 0 (faintward of the
    last plane: the measured curve has already reached 0 there);
    otherwise linear between the two bracketing planes, divided by
    `STORAGE_SCALE` once. `m0`/`dm` are not read in the hot path (the
    caller's own `a`/`c` already fold them in); kept in the signature so
    a caller can assert them against the band's own grid. Two gathers,
    four flops per evaluation.
    """
    out = np.empty(log10_f.shape[0], dtype=np.float64)
    for i in range(log10_f.shape[0]):
        idx = a * log10_f[i] + c
        if idx <= 0.0:
            out[i] = curve_row[0] / STORAGE_SCALE
        elif idx >= n - 1:
            out[i] = 0.0
        else:
            j = int(idx)
            t = idx - j
            out[i] = (np.float64(curve_row[j]) + t * (np.float64(curve_row[j + 1]) - np.float64(curve_row[j]))) / STORAGE_SCALE
    return out


@numba.njit(cache=True)
def ln_one_minus_completeness(curve_row, m0, dm, n, log10_f, a, c):
    """The fitter's own non-detection form: `ln(1 - C)`, `C` the same
    piecewise-linear lookup as `completeness`, floored at `C_CAP` so the
    result never falls below `LN_ONE_MINUS_C_FLOOR` (the `OWNER DECISION`
    above).
    """
    out = np.empty(log10_f.shape[0], dtype=np.float64)
    for i in range(log10_f.shape[0]):
        idx = a * log10_f[i] + c
        if idx <= 0.0:
            cval = curve_row[0] / STORAGE_SCALE
        elif idx >= n - 1:
            cval = 0.0
        else:
            j = int(idx)
            t = idx - j
            cval = (np.float64(curve_row[j]) + t * (np.float64(curve_row[j + 1]) - np.float64(curve_row[j]))) / STORAGE_SCALE
        if cval > C_CAP:
            cval = C_CAP
        out[i] = np.log(1.0 - cval)
    return out


@numba.njit(cache=True, parallel=True)
def completeness_many(curves, rows, m0, dm, n, log10_f, a, c, scale):
    """The vectorised twin of `completeness` over `(row, log10_f)` pairs
    (`rows`/`log10_f` the same length): one band's own `curves` table,
    `rows[i]` the source's or pixel's own row for pair `i`. `rows[i] < 0`
    (off-map, no curve) returns 0 -- the design's own "no non-detection
    term" for an off-map band is a caller-side skip (`ORIGIN_FNU == 91`
    or `PIXEL_COVERED_FRAC == 0`), not a value this kernel fabricates; 0
    is a safe default for a caller that reduces by summing this
    function's own output. `scale` is `STORAGE_SCALE` for a uint16,
    value-x-`STORAGE_SCALE` table (`CURVE_<band>`) and 1.0 for a float32,
    already-a-fraction table (`PIXEL_CURVE_<band>`: the curve of the ONE
    30-arcsec pixels' curves area-averaged over the nside-512 pixel's own
    on-map footprint, as `build` forms it on `_admitted_sample_points`
    (`PIXEL_SUPERSAMPLE_PER_SIDE`) -- not itself an exact
    multiple of `1 / STORAGE_SCALE`, so it is stored as the plain
    fraction, not quantized); the design note's own signature omits this
    (it covers only the uint16 case), added here since this one kernel
    reads both of this module's own curve tables. `numba.prange` over
    blocks, each the same four flops as the scalar kernel -- used inside
    `pieces.run` pieces.
    """
    out = np.empty(rows.shape[0], dtype=np.float64)
    for i in numba.prange(rows.shape[0]):
        row = rows[i]
        if row < 0:
            out[i] = 0.0
            continue
        idx = a * log10_f[i] + c
        if idx <= 0.0:
            out[i] = curves[row, 0] / scale
        elif idx >= n - 1:
            out[i] = 0.0
        else:
            j = int(idx)
            t = idx - j
            out[i] = (np.float64(curves[row, j]) + t * (np.float64(curves[row, j + 1]) - np.float64(curves[row, j]))) / scale
    return out


def _place_field_cube(field_map, band):
    """Places `field_map.cube` (n_mag, ny, nx), on its own (mag0, dmag)
    grid, onto `band`'s own canonical grid (`_band_mag_grid`) by EXACT
    plane alignment -- Corrections 2026-10-20.md item 1: "each field
    placed by exact plane alignment (no resampling)". Every field's own
    `mag0` differs from the band grid's own first plane by an exact
    integer number of `CANON_DMAG` steps (both share the same phase,
    CRPIX3=1.0/CD3_3=0.5 everywhere, measured): `offset` below is always
    an exact integer, never rounded. Brightward of the field's own first
    plane: the field's own first value (the design's own rule). Faintward
    of the field's own last plane: 0 here, as a placeholder -- `build`
    calls `_complete_curve` afterward to replace it wherever the field's
    own curve had not reached 0 on its own (Corrections 2026-10-20.md
    item 2).
    """
    grid = _band_mag_grid(band)
    n_band = grid.size
    offset = int(round((field_map.mag0 - grid[0]) / CANON_DMAG))
    if offset < 0 or offset + field_map.n_mag > n_band:
        raise ValueError(
            "catalog.completeness: field %r's own grid (mag0=%r, n=%d) does not fit inside "
            "band %r's canonical grid (%r..%r, n=%d) -- BAND_MAG_RANGE is stale"
            % (field_map.field, field_map.mag0, field_map.n_mag, band, grid[0], grid[-1], n_band))
    out = np.zeros((n_band,) + field_map.cube.shape[1:], dtype=np.float32)
    if offset > 0:
        out[:offset] = field_map.cube[0]
    out[offset:offset + field_map.n_mag] = field_map.cube
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


def _pix128_row(gl_deg, gb_deg, hpx_pix_128):
    """Row into `hpx_pix_128` (ascending) for each `(gl_deg, gb_deg)`
    position's own nside-128 NESTED pixel (`catalog.depths`' own (lon,
    lat) pair, its module docstring). Never -1: 2MASS carries no map
    footprint to fall off (COMPLETENESS_2026-10-20.md: "2MASS is
    all-sky"), and every source's own pixel is one `hpx_pix_128` was
    built from (`catalog.depths._build_pixel_product`'s "every pixel
    that holds at least one catalogue source"); a position whose own
    pixel is absent means the caller passed the wrong region's catalogue
    or an admitted pixel whose own nside-128 ancestor is not actually
    source-bearing, both a build-time bug, so this asserts rather than
    substituting -1 silently."""
    pix = hp.ang2pix(128, np.asarray(gl_deg, dtype=np.float64),
                      np.asarray(gb_deg, dtype=np.float64), nest=True, lonlat=True).astype(np.int64)
    row = np.searchsorted(hpx_pix_128, pix)
    row = np.clip(row, 0, hpx_pix_128.size - 1)
    found = hpx_pix_128[row] == pix
    if not np.all(found):
        raise ValueError(
            "catalog.completeness: %d position(s) have no nside-128 row in this region's "
            "depths_sesna_hpx128 product -- 2MASS is all-sky, this should never happen"
            % int(np.sum(~found)))
    return row


def _twomass_pixel_params(config, region):
    """This region's own per-nside-128-pixel 2MASS fit (`catalog.depths`'
    own `depths_sesna_hpx128__<Region>.hdf5`, briefs/round4_2026-10-20/
    C6_twomass_pixel_depths.md): `(hpx_pix_128 (n_pix,) ascending,
    f50_mjy (n_pix, 3), width_dex (n_pix, 3), from_regional (n_pix, 3)
    bool)`, `TWOMASS_KEYS` column order. A pixel's own row already
    carries the region's own depths_sesna_region value where
    `from_regional` is true (`catalog.depths.build` copies it in), so
    this reader needs no separate region-wide fallback of its own."""
    path = config_module.product_path(config, "catalog", "sesna", "depths", "hpx128", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.completeness: no per-pixel survey-depths product at %r -- run the "
            "'sesnaimpute.catalog.depths' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        hpx_pix_128 = np.asarray(f["HPX_PIX_128"][:], dtype=np.int64)
        f50_mjy = np.asarray(f["F50_2MASS_MJY"][:], dtype=np.float64)
        width_dex = np.asarray(f["WIDTH_DEX"][:], dtype=np.float64)
        from_regional = np.asarray(f["FROM_REGIONAL"][:], dtype=bool)
    return hpx_pix_128, f50_mjy, width_dex, from_regional


def _tabulate_twomass_pixel_table(f50_mjy, width_dex, zero_point_jy):
    """Every nside-128 pixel's own error-function completeness roll-off,
    tabulated on ONE shared fine (`TWOMASS_DMAG`) magnitude grid -- the
    SAME layout the five Spitzer bands' own `CURVE_<band>` tables use,
    one row per pixel (briefs/round4_2026-10-20/C6_twomass_pixel_depths.md
    item 2: "the union span"). The grid spans every row's own
    `TWOMASS_SPAN_WIDTHS` roll-off widths either side of its own 50%
    point (COMPLETENESS_2026-10-20.md "The reader": "so the interpolation
    error of a w ~ 0.1 mag erf is < 0.005"): its first plane is the
    BRIGHTEST row's own `m50 - half_span`, its last the FAINTEST row's
    own `m50 + half_span`. A row narrower than that union is clamped to
    exactly 1 brightward and 0 faintward of its OWN `+-TWOMASS_SPAN_
    WIDTHS` (the brief's own "rows outside their own +-6 widths are 1
    brightward and 0 faintward") rather than left to the erf's own
    (already negligible there) tail. `f50_mjy`/`width_dex` are `(n_pix,)`.
    Returns `(curve (n_pix, n) uint16, mag_grid (n,) float64)`.
    """
    m50 = -2.5 * np.log10(f50_mjy / (zero_point_jy * 1000.0))  # (n_pix,)
    w_mag = width_dex / 0.4
    half_span = TWOMASS_SPAN_WIDTHS * w_mag
    lo = float(np.min(m50 - half_span))
    hi = float(np.max(m50 + half_span))
    n = int(np.ceil((hi - lo) / TWOMASS_DMAG)) + 1
    mag_grid = lo + TWOMASS_DMAG * np.arange(n)
    z = (mag_grid[None, :] - m50[:, None]) / (np.sqrt(2.0) * w_mag[:, None])
    fraction = 0.5 * erfc(z)
    bright = mag_grid[None, :] < (m50[:, None] - half_span[:, None])
    faint = mag_grid[None, :] > (m50[:, None] + half_span[:, None])
    fraction = np.where(bright, 1.0, np.where(faint, 0.0, fraction))
    return _to_uint16(fraction), mag_grid


#: Rule 9: a fixed-seed subsample of each field's own COMPLETE pixels
#: (its own curve already at 0 by its own last native plane) is all the
#: band template's own median needs -- the template is one shape for the
#: whole survey, not a per-field statistic, so no field should dominate
#: the sample just because it has more on-map pixels than another.
TEMPLATE_MAX_PIXELS_PER_FIELD = 2000
TEMPLATE_SEED = 20261020


def _measure_band_template(config, band, regions):
    """The band's own survey-wide median roll-off shape (Corrections
    2026-10-20.md item 2): completeness `C` versus magnitude offset from
    each COMPLETE pixel's own 50% point (`_m50_per_pixel`), measured from
    every COMPLETE on-map pixel (one whose own last native plane has
    already reached exactly 0) in every one of `regions`' own fields for
    `band` -- the whole survey, not merely the regions a given `build`
    call is writing a product for, since the template is one shape
    shared by every region. A fixed-seed subsample of each field's own
    complete pixels (`TEMPLATE_MAX_PIXELS_PER_FIELD`, rule 9) keeps the
    scan's own cost bounded; each sampled pixel contributes every one of
    its own native planes (not a second subsample), so the median at an
    offset near the 50% point rests on many more samples than one near
    either plateau. Returns `(offsets (k,) float64, ascending, median_c
    (k,) float64)`.
    """
    rng = np.random.default_rng(TEMPLATE_SEED)
    offset_chunks, c_chunks = [], []
    for region in regions:
        for m in field_maps(config, region, band):
            iy_on, ix_on = np.where(m.on_map)
            if iy_on.size == 0:
                continue
            last_vals = m.cube[-1, iy_on, ix_on]
            complete_idx = np.flatnonzero(last_vals == 0.0)
            if complete_idx.size == 0:
                continue
            if complete_idx.size > TEMPLATE_MAX_PIXELS_PER_FIELD:
                complete_idx = rng.choice(complete_idx, size=TEMPLATE_MAX_PIXELS_PER_FIELD, replace=False)
            y, x = iy_on[complete_idx], ix_on[complete_idx]
            m50 = _m50_per_pixel(m)
            this_m50 = m50[y, x]
            ok = np.isfinite(this_m50)
            if not ok.any():
                continue
            y, x, this_m50 = y[ok], x[ok], this_m50[ok]
            native_mags = m.mag0 + m.dmag * np.arange(m.n_mag)
            offs = native_mags[:, None] - this_m50[None, :]  # (n_native, n_sel)
            cvals = m.cube[:, y, x]  # (n_native, n_sel)
            offset_chunks.append(offs.ravel())
            c_chunks.append(cvals.ravel())
    if not offset_chunks:
        return np.array([0.0]), np.array([0.0])
    offs = np.concatenate(offset_chunks)
    cs = np.concatenate(c_chunks)
    lo = CANON_DMAG * np.floor(offs.min() / CANON_DMAG)
    hi = CANON_DMAG * np.ceil(offs.max() / CANON_DMAG)
    edges = np.arange(lo - CANON_DMAG / 2, hi + CANON_DMAG, CANON_DMAG)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(offs, edges) - 1, 0, centers.size - 1)
    median_c = np.full(centers.size, np.nan)
    counts = np.bincount(bin_idx, minlength=centers.size)
    for b in np.flatnonzero(counts > 0):
        median_c[b] = np.median(cs[bin_idx == b])
    valid = np.isfinite(median_c)
    first, last = int(np.argmax(valid)), valid.size - 1 - int(np.argmax(valid[::-1]))
    return centers[first:last + 1], median_c[first:last + 1]


def _complete_field(placed, field_map, band, grid, template_offsets, template_c):
    """Replaces `placed`'s own faintward-of-the-field's-own-last-plane
    zeros (left there by `_place_field_cube`) with `band`'s own template,
    continued from each needing pixel's own last measured value, aligned
    so the template passes through it (Corrections 2026-10-20.md item 2);
    a pixel whose own curve already reached 0 is untouched. Returns
    `(placed, completed_from (ny, nx) float32, NaN where nothing was
    completed)`.
    """
    n_band = grid.size
    offset = int(round((field_map.mag0 - grid[0]) / CANON_DMAG))
    n_native = field_map.n_mag
    n_synth = n_band - offset - n_native
    completed_from = np.full(field_map.cube.shape[1:], np.nan, dtype=np.float32)
    if n_synth <= 0:
        return placed, completed_from
    last_val = field_map.cube[-1]
    needs = field_map.on_map & (last_val > 0.0)
    if not needs.any():
        return placed, completed_from
    iy, ix = np.where(needs)
    c_last = last_val[iy, ix].astype(np.float64)

    # Invert the template (ascending offset, roughly descending C): np.interp
    # needs its own xp ascending, so query on C with both arrays reversed
    # together (same permutation keeps the (offset, C) pairing correct).
    c_asc, off_corresponding = template_c[::-1], template_offsets[::-1]
    offset_star = np.interp(c_last, c_asc, off_corresponding)

    steps = CANON_DMAG * np.arange(1, n_synth + 1)
    query_offsets = offset_star[None, :] + steps[:, None]  # (n_synth, n_needing)
    tmpl_vals = np.interp(query_offsets.ravel(), template_offsets, template_c,
                           left=1.0, right=0.0).reshape(query_offsets.shape)
    tmpl_vals = np.clip(tmpl_vals, 0.0, 1.0).astype(np.float32)

    placed[offset + n_native:offset + n_native + n_synth, iy, ix] = tmpl_vals
    completed_from[iy, ix] = grid[offset + n_native]
    return placed, completed_from


def build(config, regions=None):
    """Builds the per-region completeness product (COMPLETENESS_2026-10-
    20.md "The reader: layout and kernel"): per Spitzer band, the on-map
    30-arcsec pixels' own curves (`CURVE_<B>`), each source's row into
    them, and the admitted nside-512 pixels' own curves and coverage
    (`PIXEL_CURVE_<B>`/`PIXEL_COVERED_FRAC_<B>`); per 2MASS band, the
    SAME layout at nside-128 pixel resolution instead of 30-arcsec
    (briefs/round4_2026-10-20/C6_twomass_pixel_depths.md): `catalog.
    depths`' own per-pixel fit (`_twomass_pixel_params`), tabulated on
    one shared grid (`_tabulate_twomass_pixel_table`). `SOURCE_ROW` is
    one (n_source, 8) table, `ALL_BAND_KEYS` column order, shared by
    both band families -- 2MASS's own column is never -1 (all-sky).
    `catalog.limits.completeness` is the one reader.
    """
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    band_keys = {b.key: b for b in definitions.BANDS}

    # The band template is one shape for the whole survey (Corrections
    # 2026-10-20.md item 2), so it is measured once here, over every
    # region in regions_module.REGIONS -- not merely the regions this
    # call is writing a product for -- and reused for all of them.
    all_regions = [r.name for r in regions_module.REGIONS]
    templates = {}
    for band in IRAC_MIPS_KEYS:
        offs, med_c = _measure_band_template(config, band, all_regions)
        templates[band] = (offs, med_c)
        print("catalog.completeness: %s band template (C at magnitude offset from 50%% point, "
              "%.1f..%.1f mag in %.1f steps): %s"
              % (band, offs[0], offs[-1], CANON_DMAG, np.round(med_c, 3).tolist()))

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
            sample_ra, sample_dec, n_sub = _admitted_sample_points(admitted)

            src_path = config_module.product_path(
                config, "catalog", "sesna", "sources", "source", region=region
            )
            with h5py.File(src_path, "r") as f:
                ra_deg = f["RA_DEG"][:]
                dec_deg = f["DEC_DEG"][:]
                gal_l_deg = f["GAL_L_DEG"][:]
                gal_b_deg = f["GAL_B_DEG"][:]
            n_source = ra_deg.size
            source_row = np.full((n_source, len(ALL_BAND_KEYS)), -1, dtype=np.int32)

            with h5py.File(out_path, "w") as f:
                f.attrs["GRANULE"] = "source"
                f.attrs["REGION"] = region

                corr_lengths = {}
                n_pix_total = 0
                n_completed_total = {}
                pixel_on, pixel_sum = {}, {}
                for band in IRAC_MIPS_KEYS:
                    grid = _band_mag_grid(band)
                    n_band = grid.size
                    template_offs, template_c = templates[band]
                    maps = field_maps(config, region, band)
                    row_chunks, completed_chunks, row_index_maps = [], [], []
                    offset = 0
                    field_corrs = []
                    n_completed = 0
                    for m in maps:
                        placed = _place_field_cube(m, band)
                        placed, completed_from = _complete_field(
                            placed, m, band, grid, template_offs, template_c)
                        iy_on, ix_on = np.where(m.on_map)
                        row_chunks.append(placed[:, iy_on, ix_on].T)
                        completed_chunks.append(completed_from[iy_on, ix_on])
                        n_completed += int(np.isfinite(completed_from[iy_on, ix_on]).sum())
                        row_map = np.full((m.ny, m.nx), -1, dtype=np.int64)
                        row_map[iy_on, ix_on] = offset + np.arange(iy_on.size)
                        row_index_maps.append(row_map)
                        offset += iy_on.size
                        field_corrs.append(_correlation_length(m))
                    corr_lengths[band] = float(np.nanmean(field_corrs)) if field_corrs else float("nan")
                    n_completed_total[band] = n_completed

                    curve_vals = (np.concatenate(row_chunks, axis=0) if row_chunks
                                  else np.empty((0, n_band), dtype=np.float32))
                    completed_from_vals = (np.concatenate(completed_chunks) if completed_chunks
                                            else np.empty(0, dtype=np.float32))
                    n_pix_total += curve_vals.shape[0]
                    dset = build_module.write_dataset(
                        f, f"CURVE_{band}", _to_uint16(curve_vals),
                        *REGISTRY[("completeness_sesna_source", f"CURVE_{band}")])
                    dset.attrs["ZERO_POINT_JY"] = band_keys[band].vega_zero_point_jy
                    build_module.write_dataset(f, f"MAG_GRID_{band}", grid,
                                                *REGISTRY[("completeness_sesna_source", f"MAG_GRID_{band}")])
                    build_module.write_dataset(
                        f, f"COMPLETED_FROM_{band}", completed_from_vals,
                        *REGISTRY[("completeness_sesna_source", f"COMPLETED_FROM_{band}")])

                    # the region's own ENVELOPE curve (C4_coverage_readers's
                    # fix, population.field_stars's retention test): the
                    # deepest completeness ANYWHERE in the region's own
                    # footprint for this band, the max over every on-map
                    # 30-arcsec pixel's own CURVE_<band> row at each
                    # magnitude -- a synthetic field star carries no sky
                    # position of its own (SPEC_PRIORS 1.5), so retention
                    # reads this position-independent curve, never one
                    # pointing's own.
                    envelope_vals = (curve_vals.max(axis=0) if curve_vals.shape[0]
                                      else np.zeros(n_band, dtype=np.float32))
                    build_module.write_dataset(
                        f, f"ENVELOPE_{band}", _to_uint16(envelope_vals),
                        *REGISTRY[("completeness_sesna_source", f"ENVELOPE_{band}")])

                    field_idx, iy, ix = lookup(config, region, band, ra_deg, dec_deg,
                                                n_jobs=config.n_jobs, maps=maps)
                    source_row[:, ALL_BAND_KEYS.index(band)] = _row_for_positions(
                        field_idx, iy, ix, row_index_maps)

                    # each admitted pixel's own sample points
                    # (`_admitted_sample_points`), looked up on this band's
                    # own maps: which points are on-map in this band, and
                    # the sum of this band's curve over them (an off-map
                    # point contributes zero). The per-pixel curve is
                    # formed after the loop, over the IRAC union (see
                    # there), so M1 -- the last key -- masks its sum to the
                    # union the four IRAC bands have established by then.
                    pfield_idx, piy, pix_ = lookup(config, region, band, sample_ra, sample_dec,
                                                    n_jobs=config.n_jobs, maps=maps)
                    prows = _row_for_positions(pfield_idx, piy, pix_, row_index_maps)
                    prows = prows.reshape(admitted.size, n_sub)
                    pon = prows >= 0
                    pixel_on[band] = pon
                    if band == "M1":
                        irac_union = np.logical_or.reduce([pixel_on[b] for b in IRAC_KEYS])
                        pon = pon & irac_union
                    psum = np.zeros((admitted.size, n_band), dtype=np.float64)
                    for i in np.flatnonzero(pon.any(axis=1)):
                        psum[i] = curve_vals[prows[i, pon[i]]].sum(axis=0)
                    pixel_sum[band] = psum

                # The per-pixel curve and covered fraction, over the IRAC
                # union: the atlas weights every admitted pixel by its
                # COVERAGE, the any-IRAC fraction (`bmstp.atlas._coverage`,
                # max over I1-I4 of `sky.derived.coverage`'s FRAC, within
                # 1 % of the union's own area on every region measured),
                # so a band's own curve here is its mean over the union's
                # points with the band's off-map points at zero -- then
                # `PIXEL_CURVE_<B>` x COVERAGE x area is that band's
                # expected detections over the pixel's covered area, for
                # every band including one (I2 at a mosaic edge, M1) whose
                # own footprint is only part of the union. The fraction is
                # the band's own on-map share of the whole pixel (the
                # design's PIXEL_COVERED_FRAC, equal to the coverage
                # product's FRAC). A pixel with no IRAC point on any map
                # keeps all-zero curves.
                irac_union = np.logical_or.reduce([pixel_on[b] for b in IRAC_KEYS])
                n_union = irac_union.sum(axis=1).astype(np.float64)
                for band in IRAC_MIPS_KEYS:
                    pixel_curve = np.zeros_like(pixel_sum[band], dtype=np.float32)
                    has = n_union > 0
                    pixel_curve[has] = (pixel_sum[band][has] / n_union[has, None]).astype(np.float32)
                    pixel_frac = (pixel_on[band].sum(axis=1) / float(n_sub)).astype(np.float32)
                    build_module.write_dataset(
                        f, f"PIXEL_CURVE_{band}", pixel_curve,
                        *REGISTRY[("completeness_sesna_source", f"PIXEL_CURVE_{band}")])
                    build_module.write_dataset(
                        f, f"PIXEL_COVERED_FRAC_{band}", pixel_frac,
                        *REGISTRY[("completeness_sesna_source", f"PIXEL_COVERED_FRAC_{band}")])

                # 2MASS, the SAME layout as the five Spitzer bands above,
                # at nside-128 pixel resolution instead of 30-arcsec
                # (briefs/round4_2026-10-20/C6_twomass_pixel_depths.md
                # item 2): catalog.depths' own per-pixel fit (or the
                # region's, where that fit failed a gate), tabulated on
                # one shared grid per band.
                hpx_pix_128, f50_2mass_pix, width_dex_2mass_pix, from_regional_2mass = \
                    _twomass_pixel_params(config, region)
                source_pix128_row = _pix128_row(gal_l_deg, gal_b_deg, hpx_pix_128)
                admitted_pix128_row = _pix128_row(admitted_gl, admitted_gb, hpx_pix_128)
                for tk, key in enumerate(TWOMASS_KEYS):
                    zp = band_keys[key].vega_zero_point_jy
                    curve, mag_grid = _tabulate_twomass_pixel_table(
                        f50_2mass_pix[:, tk], width_dex_2mass_pix[:, tk], zp)
                    dset = build_module.write_dataset(
                        f, f"CURVE_{key}", curve,
                        *REGISTRY[("completeness_sesna_source", f"CURVE_{key}")])
                    dset.attrs["ZERO_POINT_JY"] = zp
                    build_module.write_dataset(
                        f, f"MAG_GRID_{key}", mag_grid,
                        *REGISTRY[("completeness_sesna_source", f"MAG_GRID_{key}")])
                    build_module.write_dataset(
                        f, f"COMPLETED_FROM_{key}", np.full(curve.shape[0], np.nan, dtype=np.float32),
                        *REGISTRY[("completeness_sesna_source", f"COMPLETED_FROM_{key}")])
                    build_module.write_dataset(
                        f, f"FROM_REGIONAL_{key}", from_regional_2mass[:, tk],
                        *REGISTRY[("completeness_sesna_source", f"FROM_REGIONAL_{key}")])
                    envelope_vals = curve.max(axis=0) if curve.shape[0] else np.zeros(curve.shape[1], dtype=np.uint16)
                    build_module.write_dataset(
                        f, f"ENVELOPE_{key}", envelope_vals.astype(np.uint16),
                        *REGISTRY[("completeness_sesna_source", f"ENVELOPE_{key}")])

                    source_row[:, ALL_BAND_KEYS.index(key)] = source_pix128_row
                    pixel_curve_2mass = (curve[admitted_pix128_row].astype(np.float32)
                                         / STORAGE_SCALE)
                    build_module.write_dataset(
                        f, f"PIXEL_CURVE_{key}", pixel_curve_2mass,
                        *REGISTRY[("completeness_sesna_source", f"PIXEL_CURVE_{key}")])
                    build_module.write_dataset(
                        f, f"PIXEL_COVERED_FRAC_{key}", np.ones(admitted.size, dtype=np.float32),
                        *REGISTRY[("completeness_sesna_source", f"PIXEL_COVERED_FRAC_{key}")])

                build_module.write_dataset(f, "SOURCE_ROW", source_row,
                                            *REGISTRY[("completeness_sesna_source", "SOURCE_ROW")])
                build_module.write_dataset(f, "HPX_PIX_512", admitted,
                                            *REGISTRY[("completeness_sesna_source", "HPX_PIX_512")])

                n_fallback_2mass = {key: int(from_regional_2mass[:, tk].sum())
                                    for tk, key in enumerate(TWOMASS_KEYS)}

            print("catalog.completeness: %s pixel-to-pixel 50%% correlation by band: %s"
                  % (region, {k: round(v, 3) for k, v in corr_lengths.items()}))
            print("catalog.completeness: %s on-map pixels completed (cut off before reaching 0) by band: %s"
                  % (region, n_completed_total))
            print("catalog.completeness: %s 2MASS nside-128 pixels on the regional fallback "
                  "(of %d), by band: %s" % (region, hpx_pix_128.size, n_fallback_2mass))
            out_paths.append(out_path)
            n_done += 1
            st.tick(n_done, len(regions), "regions")

        st.done(out_paths[-1] if out_paths else None, regions=len(regions),
                admitted_pixels_last_region=int(admitted.size) if regions else 0,
                spitzer_pixels_last_region=n_pix_total if regions else 0)


if __name__ == "__main__":
    run(build)
