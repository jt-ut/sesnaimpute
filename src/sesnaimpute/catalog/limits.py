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
    band, each resolving to `catalog.completeness`'s own kernels."""

    def __init__(self, h5file):
        self._f = h5file
        self.source_row = np.asarray(h5file["SOURCE_ROW"][:], dtype=np.int64)
        self.hpx_pix_512 = np.asarray(h5file["HPX_PIX_512"][:], dtype=np.int64)
        self._ac = {}
        self._curve = {}
        self._pixel_curve = {}
        self._mag_grid_band = {}
        # Corrections 2026-10-20.md item 1: ONE canonical grid PER BAND
        # (no grid shared across bands any more), read from that band's
        # own MAG_GRID_<band>.
        for band in IRAC_MIPS_KEYS:
            zp = h5file[f"CURVE_{band}"].attrs["ZERO_POINT_JY"]
            grid = np.asarray(h5file[f"MAG_GRID_{band}"][:], dtype=np.float64)
            self._mag_grid_band[band] = grid
            self._ac[band] = _band_ac(float(grid[0]), float(grid[1] - grid[0]), zp)
            self._curve[band] = h5file[f"CURVE_{band}"]
            self._pixel_curve[band] = h5file[f"PIXEL_CURVE_{band}"]
        for band in TWOMASS_KEYS:
            zp = h5file[f"CURVE_{band}"].attrs["ZERO_POINT_JY"]
            grid = np.asarray(h5file[f"MAG_GRID_{band}"][:], dtype=np.float64)
            self._mag_grid_band[band] = grid
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
