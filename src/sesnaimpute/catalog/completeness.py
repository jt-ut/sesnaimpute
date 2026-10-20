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

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from sesnaimpute import pieces as pieces_module

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
