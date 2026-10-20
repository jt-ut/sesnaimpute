"""The per-source completeness reader (SPEC_PRIORS.md section 1.3;
SPEC_BMSTP_DRAFT.md section 3.3; COMPLETENESS_2026-10-20.md "The reader:
layout and kernel").

`completeness(config, region)` is the only completeness reader in the
package: every consumer of a source's or a pixel's detection completeness
-- the likelihood's non-detection terms, the field-star retention, the
atlas, every class build -- reads this object, never `catalog.
completeness.build`'s product file directly, and never the kernel
functions in `catalog.completeness` (`completeness`, `ln_one_minus_
completeness`, `completeness_many`) directly either. It memory-maps that
file's `CURVE_<band>`/`PIXEL_CURVE_<band>` tables (never copied per
worker) and exposes the design note's own kernel through plain methods:
`fraction`, `ln_one_minus_fraction`, `fraction_pixel`, `covered`.

This replaces the old `limits(config, region)` function outright (deleted,
not kept beside it): that interface returned one scalar 50%-completeness
flux per source and band, from a region-wide counts fit (`catalog.
depth_grid`'s `F_LIM_50_MJY`) shifted by the source's own `DCOMP90`. Rob
Gutermuth's measured maps (`catalog.completeness`) make that shift-and-fit
unnecessary: every consumer now reads the MEASURED curve at its own
source's or pixel's own position directly, so the five Spitzer bands'
`limits()` callers all need rewiring to this object (this unit's own
report lists every call site the change breaks).
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute.catalog import completeness as completeness_module

IRAC_MIPS_KEYS = completeness_module.IRAC_MIPS_KEYS
TWOMASS_KEYS = completeness_module.TWOMASS_KEYS


def _band_ac(mag0, dmag, zero_point_jy):
    """`(a, c)` converting a queried log10 flux (mJy) straight to a
    fractional plane index on the band's own `(mag0, dmag)` grid:
    `mag = -2.5 log10(f) + 2.5 log10(f0)`, `idx = (mag - mag0) / dmag`,
    so `a = -2.5 / dmag`, `c = (2.5 log10(f0_mJy) - mag0) / dmag`
    (COMPLETENESS_2026-10-20.md "The reader"'s own `A_B`, `C_B`; this
    module's own general form for a grid whose `dmag` is not 0.5 or
    whose `mag0` is not 4.75 -- the three 2MASS bands' own fine grids).
    """
    f0_mjy = zero_point_jy * 1000.0
    a = -2.5 / dmag
    c = (2.5 * np.log10(f0_mjy) - mag0) / dmag
    return a, c


class Completeness:
    """The reader object `completeness(config, region)` returns: the
    memory-mapped `catalog.completeness` tables for `region`, plus one
    `fraction`/`fraction_pixel`/`ln_one_minus_fraction`/`covered` call per
    band, each resolving to `catalog.completeness`'s own kernels, and the
    single-number methods (`fifty_percent*`, `local_width_dex*`) a pixel-
    or source-level consumer that still wants one flux and one width per
    band gets from the SAME stored curve, by the same linear
    interpolation `fraction` uses, never a second stored statistic
    (briefs/round4_2026-10-20/C4_coverage_readers.md: "A reader that needs
    a single number per pixel or source (the 50% crossing) gets it from
    one reader method, computed from the curve, never stored twice")."""

    def __init__(self, h5file):
        self._f = h5file
        self.source_row = np.asarray(h5file["SOURCE_ROW"][:], dtype=np.int64)
        self.hpx_pix_512 = np.asarray(h5file["HPX_PIX_512"][:], dtype=np.int64)
        self._ac = {}
        self._curve = {}
        self._pixel_curve = {}
        self._mag_grid_band = {}
        self._completed_from = {}
        self._zero_point_jy = {}
        self._envelope = {}
        # Corrections 2026-10-20.md item 1: ONE canonical grid PER BAND
        # (no grid shared across bands any more), read from that band's
        # own MAG_GRID_<band>.
        for band in IRAC_MIPS_KEYS:
            zp = h5file[f"CURVE_{band}"].attrs["ZERO_POINT_JY"]
            grid = np.asarray(h5file[f"MAG_GRID_{band}"][:], dtype=np.float64)
            self._mag_grid_band[band] = grid
            self._zero_point_jy[band] = float(zp)
            self._ac[band] = _band_ac(float(grid[0]), float(grid[1] - grid[0]), zp)
            self._curve[band] = h5file[f"CURVE_{band}"]
            self._pixel_curve[band] = h5file[f"PIXEL_CURVE_{band}"]
            self._completed_from[band] = np.asarray(h5file[f"COMPLETED_FROM_{band}"][:], dtype=np.float32)
            self._envelope[band] = np.asarray(h5file[f"ENVELOPE_{band}"][:])
        for band in TWOMASS_KEYS:
            zp = h5file[f"CURVE_{band}"].attrs["ZERO_POINT_JY"]
            grid = np.asarray(h5file[f"MAG_GRID_{band}"][:], dtype=np.float64)
            self._mag_grid_band[band] = grid
            self._zero_point_jy[band] = float(zp)
            self._ac[band] = _band_ac(float(grid[0]), float(grid[1] - grid[0]), zp)
            self._curve[band] = h5file[f"CURVE_{band}"]

    def _grid_params(self, band):
        grid = self._mag_grid_band[band]
        return float(grid[0]), float(grid[1] - grid[0])

    def _row_for(self, band, source_index):
        if band in TWOMASS_KEYS:
            return np.zeros(np.atleast_1d(source_index).shape, dtype=np.int64)
        return np.atleast_1d(self.source_row[source_index, IRAC_MIPS_KEYS.index(band)])

    def covered(self, band):
        """This region's own per-source boolean: `True` where the source
        has a completeness curve in `band` (Spitzer: `SOURCE_ROW >= 0`;
        2MASS: always `True`, no off-map concept for a survey-wide
        bound)."""
        if band in TWOMASS_KEYS:
            return np.ones(self.source_row.shape[0], dtype=bool)
        return self.source_row[:, IRAC_MIPS_KEYS.index(band)] >= 0

    def completed_from(self, band, row):
        """`band`'s own `COMPLETED_FROM_<band>` at `row` (a row into
        `CURVE_<band>`, e.g. from `source_row`): the magnitude of the
        first plane that is the band's own template continuation rather
        than Rob's own measurement, `NaN` where nothing was completed
        (Corrections 2026-10-20.md item 2). 2MASS carries no such
        column (its one curve is tabulated whole, never cut off)."""
        if band in TWOMASS_KEYS:
            return np.full(np.atleast_1d(row).shape, np.nan)
        return self._completed_from[band][np.atleast_1d(row)]

    def fraction(self, band, flux_mjy, source_index):
        """The completeness fraction at one source's own row (`band`,
        `source_index`), evaluated at every one of `flux_mjy`'s own
        values (mJy) -- the scalar kernel, one call per source."""
        log10_f = np.log10(np.asarray(flux_mjy, dtype=np.float64))
        row = int(self._row_for(band, source_index)[0])
        g0, dm = self._grid_params(band)
        n = self._curve[band].shape[1]
        a, c = self._ac[band]
        if row < 0:
            return np.zeros(log10_f.shape[0], dtype=np.float64)
        return completeness_module.completeness(self._curve[band][row], g0, dm, n, log10_f, a, c)

    def ln_one_minus_fraction(self, band, flux_mjy, source_index):
        """The fitter's non-detection form (`ln_one_minus_completeness`)
        at one source's own row, `band`, over an array of predicted
        fluxes (mJy, e.g. every template in the sweep's library)."""
        log10_f = np.log10(np.asarray(flux_mjy, dtype=np.float64))
        row = int(self._row_for(band, source_index)[0])
        g0, dm = self._grid_params(band)
        n = self._curve[band].shape[1]
        a, c = self._ac[band]
        if row < 0:
            return np.full(log10_f.shape[0], completeness_module.LN_ONE_MINUS_C_FLOOR)
        return completeness_module.ln_one_minus_completeness(
            self._curve[band][row], g0, dm, n, log10_f, a, c)

    def envelope_fraction(self, band, flux_mjy):
        """The completeness fraction at `flux_mjy`'s own values (mJy), read
        from this region's own ENVELOPE_<band> curve (`catalog.
        completeness.build`'s max, over every on-map pixel, of CURVE_<band>
        at each magnitude): the deepest completeness ANYWHERE in the
        region's own footprint for `band`, position-independent --
        `population.field_stars`'s retention test, since no synthetic
        field star carries a sky position of its own (SPEC_PRIORS.md
        section 1.5). 2MASS has no per-pixel map at all (module docstring),
        so its own region-wide tabulated curve (`fraction` at row 0) IS
        already position-independent and is read the same way here, never
        a second curve."""
        if band in TWOMASS_KEYS:
            return self.fraction(band, flux_mjy, 0)
        log10_f = np.log10(np.asarray(flux_mjy, dtype=np.float64))
        g0, dm = self._grid_params(band)
        n = self._envelope[band].shape[0]
        a, c = self._ac[band]
        return completeness_module.completeness(self._envelope[band], g0, dm, n, log10_f, a, c)

    def pixel_fraction_params(self, band):
        """`(curve, g0, dm, n, a, c, scale)`: this band's own completeness
        table as a plain in-memory array (`PIXEL_CURVE_<band>`, or row 0 of
        `CURVE_<band>` for 2MASS -- the SAME table and row convention
        `fraction_pixel` reads) plus the grid parameters `catalog.
        completeness.completeness_many` needs, for a caller that evaluates
        many `(pixel_row, flux)` pairs across many separate calls in one
        worker's lifetime (`bmstp.atlas`'s per-tile/per-sightline member
        quadrature) and would otherwise pay `fraction_pixel`'s own
        per-call table read (`curves[:]`) again on every one of them: read
        ONCE here and cached by the caller (`pieces.py`'s per-process
        cache convention, `bmstp.sample_star`'s own pattern) for its own
        worker process's lifetime."""
        g0, dm = self._grid_params(band)
        a, c = self._ac[band]
        if band in TWOMASS_KEYS:
            curves = self._curve[band]
            return (np.asarray(curves[:]), g0, dm, curves.shape[1], a, c,
                    float(completeness_module.STORAGE_SCALE))
        curves = self._pixel_curve[band]
        return np.asarray(curves[:]), g0, dm, curves.shape[1], a, c, 1.0

    def fraction_pixel(self, band, flux_mjy, pixel_row):
        """The completeness fraction at each of `pixel_row`'s own rows (a
        row index into `PIXEL_CURVE_<band>`, not a HEALPix pixel number;
        2MASS ignores it, one shared curve), `band`, evaluated at
        `flux_mjy`'s own flux (mJy, broadcast to `pixel_row`'s shape) --
        the vectorised kernel, over arrays of `(row, magnitude)` pairs.
        """
        if band in TWOMASS_KEYS:
            rows = np.zeros(np.atleast_1d(pixel_row).shape, dtype=np.int64)
            curves = self._curve[band]
            scale = completeness_module.STORAGE_SCALE
        else:
            rows = np.ascontiguousarray(np.atleast_1d(pixel_row).astype(np.int64))
            curves = self._pixel_curve[band]
            scale = 1.0  # PIXEL_CURVE_<band> is float32, already a plain fraction
        log10_f = np.ascontiguousarray(
            np.broadcast_to(np.log10(np.asarray(flux_mjy, dtype=np.float64)), rows.shape))
        g0, dm = self._grid_params(band)
        n = curves.shape[1]
        a, c = self._ac[band]
        return completeness_module.completeness_many(curves[:], rows, g0, dm, n, log10_f, a, c, scale)

    def _crossing_mag(self, band, rows, target_c):
        """`(n,)` magnitude where `band`'s own measured curve, at each of
        `rows`'s own rows, crosses `target_c` (the FIRST crossing from
        the bright end, `catalog.completeness._m50_per_pixel`'s own
        convention): linear interpolation between the two bracketing
        planes, `NaN` where `rows < 0` (off-map) or the curve never
        reaches `target_c` brightward of its own last plane."""
        g0, dm = self._grid_params(band)
        curve_table = self._curve[band]
        rows = np.atleast_1d(rows)
        out = np.full(rows.shape, np.nan, dtype=np.float64)
        valid = rows >= 0
        if not valid.any():
            return out
        sub = curve_table[:][rows[valid]].astype(np.float64) / completeness_module.STORAGE_SCALE
        below = sub < target_c
        first_below = np.argmax(below, axis=1)
        never_below = ~below.any(axis=1)
        j = np.clip(first_below - 1, 0, sub.shape[1] - 2)
        c0 = np.take_along_axis(sub, j[:, None], axis=1)[:, 0]
        c1 = np.take_along_axis(sub, (j + 1)[:, None], axis=1)[:, 0]
        denom = np.where(c0 != c1, c0 - c1, 1.0)
        t = np.where(c0 != c1, (c0 - target_c) / denom, 0.0)
        mag = g0 + dm * (j.astype(np.float64) + t)
        mag = np.where(never_below | (first_below == 0), np.nan, mag)
        out[valid] = mag
        return out

    def _mag_to_flux_mjy(self, band, mag):
        return 10.0 ** (np.log10(self._zero_point_jy[band] * 1000.0) - mag / 2.5)

    def fifty_percent(self, band, source_index):
        """`(n,)` mJy: the flux where this source's own curve (`band`)
        crosses completeness = 0.5 -- the one number a per-source
        consumer that used to read the retired `F_LIM_50`/`F_LIM_50_MJY`
        gets instead, from the SAME stored curve `fraction` reads."""
        rows = self._row_for(band, source_index)
        mag = self._crossing_mag(band, rows, 0.5)
        return self._mag_to_flux_mjy(band, mag)

    def _crossing_mag_pixel(self, band, pixel_row, target_c):
        """`_crossing_mag`'s own twin for `PIXEL_CURVE_<band>`: a plain
        float32 fraction (an area average, module docstring), not a
        value-x-`STORAGE_SCALE` table, so the crossing search runs on
        the raw fraction directly rather than sharing `_crossing_mag`'s
        own `/STORAGE_SCALE` step. 2MASS has one region-wide row (index
        0, `_row_for`'s own convention) -- `local_width_dex_pixel` and
        `fifty_percent_pixel` both resolve there to `_local_width_dex`/
        `_crossing_mag`, the SAME per-row search `fraction` itself uses."""
        rows = np.ascontiguousarray(np.atleast_1d(pixel_row).astype(np.int64))
        curve_table = self._pixel_curve[band]
        g0, dm = self._grid_params(band)
        out = np.full(rows.shape, np.nan, dtype=np.float64)
        valid = rows >= 0
        if valid.any():
            sub = curve_table[:][rows[valid]].astype(np.float64)
            below = sub < target_c
            first_below = np.argmax(below, axis=1)
            never_below = ~below.any(axis=1)
            j = np.clip(first_below - 1, 0, sub.shape[1] - 2)
            c0 = np.take_along_axis(sub, j[:, None], axis=1)[:, 0]
            c1 = np.take_along_axis(sub, (j + 1)[:, None], axis=1)[:, 0]
            denom = np.where(c0 != c1, c0 - c1, 1.0)
            t = np.where(c0 != c1, (c0 - target_c) / denom, 0.0)
            mag = g0 + dm * (j.astype(np.float64) + t)
            out[valid] = np.where(never_below | (first_below == 0), np.nan, mag)
        return out

    def fifty_percent_pixel(self, band, pixel_row):
        """`(n,)` mJy: the flux where `PIXEL_CURVE_<band>` at each of
        `pixel_row`'s own rows crosses completeness = 0.5 -- the one
        number a per-pixel consumer that used to read the retired
        `F_LIM_50_PIX_MJY` gets instead."""
        if band in TWOMASS_KEYS:
            rows = np.zeros(np.atleast_1d(pixel_row).shape, dtype=np.int64)
            mag = self._crossing_mag(band, rows, 0.5)
        else:
            mag = self._crossing_mag_pixel(band, pixel_row, 0.5)
        return self._mag_to_flux_mjy(band, mag)

    def _local_width_dex(self, band, rows):
        """`(n,)` dex: this row's own measured local roll-off width, the
        per-row analogue of `catalog.depths`' region-wide fitted
        `WIDTH_DEX` (COMPLETENESS_2026-10-20.md rule 5: the Gaussian
        model is not used for the completeness term itself -- every
        probability-computing consumer reads the measured curve directly
        (`fraction`/`fraction_pixel`/`completeness_many`), never this
        width -- but `catalog.figure_depth`'s own depth figure still
        draws one disclosed Gaussian APPROXIMATION per pixel and needs
        one width number for it; this is the SAME erf-width definition
        depths.py uses (`WIDTH_DEX = 0.4 * w_mag`), with `w_mag` now the
        row's own measured half-distance between its C=0.8413 and
        C=0.1587 crossings (one sigma either side of the row's own
        measured 50% point, the convention `catalog.depths.build`'s own
        fit used) -- a per-row MEASUREMENT, not a second region-wide
        fit."""
        mag_bright = self._crossing_mag(band, rows, 0.8413)  # one sigma brighter than 50%
        mag_faint = self._crossing_mag(band, rows, 0.1587)   # one sigma fainter than 50%
        w_mag = (mag_faint - mag_bright) / 2.0
        return 0.4 * w_mag

    def local_width_dex_pixel(self, band, pixel_row):
        """`(n,)` dex: `_local_width_dex`'s own pixel-curve form, read only
        by `catalog.figure_depth`'s disclosed Gaussian approximation
        (this class's own docstring: never by a probability computation)."""
        if band in TWOMASS_KEYS:
            rows = np.zeros(np.atleast_1d(pixel_row).shape, dtype=np.int64)
            return self._local_width_dex(band, rows)
        w_mag = (self._crossing_mag_pixel(band, pixel_row, 0.1587)
                 - self._crossing_mag_pixel(band, pixel_row, 0.8413)) / 2.0
        return 0.4 * w_mag


def completeness(config, region):
    """Returns the `Completeness` reader for `region`: the memory-mapped
    `catalog.completeness` product plus the kernels in `catalog.
    completeness`. The returned object owns an open, read-only h5py file
    (memory-mapped datasets, never copied per worker, module docstring);
    a caller that forks workers should open one reader per worker, not
    share this one across a process boundary.
    """
    path = config_module.product_path(config, "catalog", "sesna", "completeness", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "catalog.limits.completeness: no completeness product at %r -- run the "
            "'sesnaimpute.catalog.completeness' RUNBOOK line first" % path)
    h5file = h5py.File(path, "r")
    return Completeness(h5file)
