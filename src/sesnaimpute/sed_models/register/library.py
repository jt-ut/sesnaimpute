"""
library.py
====================================================================
A uniform, read-only view over one sedfitter model library directory.

Every library on disk is the same five things -- `models.conf`,
`flux.fits`, `parameters.fits`, optionally `info.fits`, and
`convolved/{band}.fits`, plus `classmap.fits` -- and the one shape
difference that remains is real: aperture-dependent libraries carry a
20-point aperture axis that aperture-independent ones collapse to a
single sentinel slice.

Three OTHER differences this module once absorbed have been retired at
the source, and the workarounds are gone with them: `sps` no longer
ships `2J/2H/2K` band filenames, `yso`' info.fits no longer spells
the join key `Model Name`, and the subclass label no longer lives in a
different file per library. Each is now one spelling everywhere, so a
file that disagrees is a curation bug to fix rather than a case to
handle. That is the standing rule here: absorb genuine shape
differences, never vocabulary ones.

What it deliberately does NOT do:

- It does not re-derive anything a library owns. CLASS and SUBCLASS are
  read from `classmap.fits` and passed through verbatim; only the
  CENSUS class assignment is `sed_models_register`'s, and that lives in the
  driver. Structural checks only -- row alignment, completeness,
  vocabulary closure -- never a recomputation of a curator's physics.
- It does not load the SED cube eagerly. `flux.fits`'s VALUES array is
  ~900 MB for a single yso sub-grid; it is memory-mapped and only
  the slices a caller actually indexes are paged in.

The model-name array is the join key throughout the project, and three
files carry their own copy of it. `Models.read` takes the names it fits
against from the CONVOLVED files (`conv.model_names`, sedfitter
models.py:344-347), not from `flux.fits`, so `check_alignment` compares
all three rather than trusting the cube alone -- a guard that checked
only `flux.fits` would miss the case it exists to catch.
====================================================================
"""

import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from astropy.io import fits

from sedfitter.utils import parfile

from sesnaimpute.sed_models.constants import BANDS

__all__ = ["ModelLibrary", "LibraryMeta", "ConvolvedBand", "AlignmentError"]

_BAND_KEYS = list(BANDS)

#: The join key. One spelling, required everywhere -- R17's info.fits
#: once used "Model Name" and YSO assembly now renames it at the read
#: boundary, so a file that disagrees is a curation bug to fix at the
#: source rather than an alias to absorb here.
_NAME_KEY = "MODEL_NAME"


class AlignmentError(RuntimeError):
    """Raised when a library's model-name arrays disagree across files."""


@dataclass(frozen=True)
class LibraryMeta:
    """The `models.conf` contents plus what the file layout implies."""

    path: str
    name: str
    aperture_dependent: bool
    logd_step: float
    version: int
    n_models: int
    n_apertures: int
    has_info: bool

    @property
    def key(self) -> str:
        """Short identifier: the directory name, or 'parent/child' for a
        sub-grid such as yso/cII whose bare name ('cII') is ambiguous."""
        path = os.path.normpath(self.path)
        parent, child = os.path.split(path)
        grandparent = os.path.basename(parent)
        return f"{grandparent}/{child}" if grandparent else child


@dataclass(frozen=True)
class ConvolvedBand:
    """One band's convolved fluxes, as stored."""

    band: str            # BANDS key, e.g. "Ks" -- not the on-disk filename
    filename: str
    filtwav_um: float
    apertures_au: np.ndarray   # (n_ap,)
    flux_mjy: np.ndarray       # (n_models, n_ap)
    error_mjy: np.ndarray      # (n_models, n_ap)


def _read_table(path, hdu=1):
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[hdu].data)


def _find_name_column(rec):
    if _NAME_KEY in (rec.dtype.names or ()):
        return _NAME_KEY
    raise AlignmentError(
        f"no {_NAME_KEY!r} column; file has {list(rec.dtype.names or ())}"
    )


def _as_str(arr):
    """Normalize a FITS string column to stripped unicode."""
    return np.char.strip(np.asarray(arr).astype(str))


class ModelLibrary:
    """Read-only accessor for one model library directory.

    Parameters
    ----------
    path : str
        Directory containing `models.conf`. For yso this is the
        sub-grid directory (e.g. `.../yso/cII`), not the parent.
    """

    def __init__(self, path):
        self.path = os.path.normpath(path)
        conf_path = os.path.join(self.path, "models.conf")
        if not os.path.exists(conf_path):
            raise FileNotFoundError(f"no models.conf in {self.path}")
        self._conf = parfile.read(conf_path, "conf")

    # ---------------------------------------------------------------- paths

    def _path(self, *parts):
        return os.path.join(self.path, *parts)

    def _optional(self, *parts):
        p = self._path(*parts)
        return p if os.path.exists(p) else None

    @cached_property
    def _band_files(self):
        """Map BANDS key -> convolved filename.

        All five libraries now name these files by the BANDS key; sps's
        historical `2J/2H/2K` spelling was retired when the libraries were
        standardized, so a missing file is a real error rather than an
        alias to try.
        """
        out = {}
        for band in _BAND_KEYS:
            for suffix in (".fits", ".fits.gz"):
                candidate = self._path("convolved", band + suffix)
                if os.path.exists(candidate):
                    out[band] = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"{self.key}: no convolved file for band {band!r} "
                    f"in {self._path('convolved')}"
                )
        return out

    @property
    def key(self):
        """Short identifier, safe to use before `meta` is built."""
        parent, child = os.path.split(self.path)
        grandparent = os.path.basename(parent)
        return f"{grandparent}/{child}" if grandparent else child

    # ---------------------------------------------------------------- meta

    @cached_property
    def meta(self):
        names = self.model_names
        with fits.open(self._band_files[_BAND_KEYS[0]], memmap=False) as hdul:
            n_ap = int(hdul[0].header.get("NAP", 1))
        return LibraryMeta(
            path=self.path,
            name=str(self._conf.get("name", "")),
            aperture_dependent=bool(self._conf["aperture_dependent"]),
            logd_step=float(self._conf.get("logd_step", 0.02)),
            version=int(self._conf.get("version", 1)),
            n_models=len(names),
            n_apertures=n_ap,
            has_info=self._optional("info.fits") is not None,
        )

    # ---------------------------------------------------------------- names

    @cached_property
    def model_names(self):
        """The canonical model-name array, taken from `parameters.fits`.

        `parameters.fits` is the canonical order because sedfitter's own
        `filter_table` sorts against it and `_convolve_model_dir_2` asserts
        the cube matches it. `check_alignment` verifies the other two files
        agree.
        """
        rec = _read_table(self._path("parameters.fits"))
        return _as_str(rec[_find_name_column(rec)])

    def check_alignment(self):
        """Assert model-name arrays agree across parameters/flux/convolved.

        Returns the list of sources checked. Raises `AlignmentError` on the
        first disagreement, reporting the first offending row.
        """
        canonical = self.model_names
        checked = ["parameters.fits"]

        def _compare(other, label):
            if len(other) != len(canonical):
                raise AlignmentError(
                    f"{self.meta.key}: {label} has {len(other)} models, "
                    f"parameters.fits has {len(canonical)}"
                )
            bad = np.flatnonzero(other != canonical)
            if bad.size:
                i = int(bad[0])
                raise AlignmentError(
                    f"{self.meta.key}: {label} disagrees with parameters.fits "
                    f"at row {i} ({other[i]!r} vs {canonical[i]!r}); "
                    f"{bad.size} of {len(canonical)} rows differ"
                )
            checked.append(label)

        with fits.open(self._path("flux.fits"), memmap=True) as hdul:
            if "MODEL_NAMES" in [h.name for h in hdul]:
                _compare(_as_str(hdul["MODEL_NAMES"].data["MODEL_NAME"]),
                         "flux.fits")

        # The array the fitter actually uses (sedfitter models.py:344-347).
        first = _BAND_KEYS[0]
        _compare(self.convolved(first).model_names_checked, f"convolved/{first}")
        return checked

    # ---------------------------------------------------------------- tables

    @cached_property
    def parameters(self):
        return _read_table(self._path("parameters.fits"))

    @cached_property
    def info(self):
        """`info.fits`'s table, or None where the library has no info file."""
        path = self._optional("info.fits")
        return None if path is None else _read_table(path)

    # ---------------------------------------------------------------- fluxes

    def convolved(self, band):
        """Read one band's convolved fluxes. `band` is a BANDS key."""
        path = self._band_files[band]
        with fits.open(path, memmap=False) as hdul:
            filtwav = hdul[0].header.get("FILTWAV")
            tc = hdul["CONVOLVED FLUXES"].data
            names = _as_str(tc["MODEL_NAME"])
            flux = np.atleast_2d(np.asarray(tc["TOTAL_FLUX"], dtype=float))
            err = np.atleast_2d(np.asarray(tc["TOTAL_FLUX_ERR"], dtype=float))
            if flux.shape[0] == 1 and flux.shape[1] == len(names):
                flux = flux.T          # (n_models, 1) for NAP=1 layouts
                err = err.T
            try:
                ta = hdul["APERTURES"].data
                apertures = np.asarray(ta["APERTURE"], dtype=float)
            except KeyError:
                apertures = np.array([np.nan])

        out = ConvolvedBand(
            band=band,
            filename=path,
            filtwav_um=None if filtwav is None else float(filtwav),
            apertures_au=apertures,
            flux_mjy=flux,
            error_mjy=err,
        )
        # Attach for check_alignment without widening the public dataclass.
        object.__setattr__(out, "model_names_checked", names)
        return out

    def convolved_all(self):
        """All 8 bands, in BANDS order."""
        return [self.convolved(b) for b in _BAND_KEYS]

    def filtwav(self):
        """The 8 FILTWAV values in BANDS order (microns)."""
        return np.array([self.convolved(b).filtwav_um for b in _BAND_KEYS],
                        dtype=float)

    # ---------------------------------------------------------------- cube

    @cached_property
    def distance_cm(self):
        """The library's declared reference distance, from flux.fits.

        Read rather than assumed, but note sedfitter does NOT consult it:
        `Models._read_version_2` hardcodes a 1 kpc frame
        (`conv.flux * (u.kpc / m.distances)**2`, models.py:334). So this
        value is used to CHECK that assumption, not to act on it -- a
        library declaring anything else would be silently misscaled by
        the fitter, and that should fail loudly here instead.
        """
        with fits.open(self._path("flux.fits"), memmap=True) as hdul:
            d = hdul[0].header.get("DISTANCE")
        if d is None:
            raise AlignmentError(f"{self.key}: flux.fits has no DISTANCE header")
        return float(d)

    @cached_property
    def classmap(self):
        """The library's own CLASS / SUBCLASS declaration, plus legends.

        `classmap.fits` is the CURATOR's product and the authority on what
        the models are; `sed_models_register` reads it and never re-derives it.
        Returns a dict with `subclass`, `class_declared`, the two legend
        mappings, `prob` (an (n, n_sub) array with `prob_columns`, or None
        where the assignment is hard), and `header` for provenance.

        Structural checks only -- row alignment and vocabulary closure.
        The labels themselves are taken as given.
        """
        path = self._optional("classmap.fits")
        if path is None:
            raise AlignmentError(
                f"{self.key}: no classmap.fits. Every library must declare "
                f"its own CLASS/SUBCLASS; sed_models_register does not infer them."
            )
        with fits.open(path, memmap=False) as hdul:
            hdr = dict(hdul[0].header)
            cm = hdul["CLASSMAP"].data
            names = _as_str(cm["MODEL_NAME"])
            subclass = _as_str(cm["SUBCLASS"])
            cls_col = _as_str(cm["CLASS"])
            legends = {}
            for ext, key in (("CLASS_LEGEND", "CLASS"),
                             ("SUBCLASS_LEGEND", "SUBCLASS")):
                try:
                    t = hdul[ext].data
                except KeyError:
                    legends[ext] = {}
                    continue
                legends[ext] = dict(zip(_as_str(t[key]),
                                        _as_str(t["DESCRIPTION"])))
            prob = prob_cols = None
            if "SUBCLASS_PROB" in [h.name for h in hdul]:
                t = hdul["SUBCLASS_PROB"].data
                prob_cols = [c for c in t.dtype.names if c != "MODEL_NAME"]
                prob = np.column_stack([np.asarray(t[c], float)
                                        for c in prob_cols])
                if not np.array_equal(_as_str(t["MODEL_NAME"]), names):
                    raise AlignmentError(
                        f"{self.key}: SUBCLASS_PROB rows do not match CLASSMAP"
                    )

        if len(names) != len(self.model_names) or np.any(names != self.model_names):
            raise AlignmentError(
                f"{self.key}: classmap.fits model names do not match "
                f"parameters.fits"
            )
        if np.any(subclass == ""):
            raise AlignmentError(
                f"{self.key}: {int(np.sum(subclass == ''))} models have a "
                f"blank SUBCLASS"
            )
        vocab = set(legends.get("SUBCLASS_LEGEND", {}))
        unknown = sorted(set(subclass.tolist()) - vocab) if vocab else []
        if unknown:
            raise AlignmentError(
                f"{self.key}: SUBCLASS values {unknown} are absent from "
                f"SUBCLASS_LEGEND -- the file disagrees with itself"
            )
        if prob is not None:
            if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-5):
                raise AlignmentError(f"{self.key}: SUBCLASS_PROB rows do not sum to 1")
            if not np.all(np.isfinite(prob)):
                raise AlignmentError(f"{self.key}: SUBCLASS_PROB has non-finite values")
            am = np.asarray(prob_cols)[prob.argmax(axis=1)]
            if np.any(am != subclass):
                raise AlignmentError(
                    f"{self.key}: SUBCLASS is not the argmax of SUBCLASS_PROB "
                    f"({int(np.sum(am != subclass))} rows differ)"
                )
        return {
            "subclass": subclass,
            "class_declared": cls_col,
            "class_legend": legends.get("CLASS_LEGEND", {}),
            "subclass_legend": legends.get("SUBCLASS_LEGEND", {}),
            "prob": prob,
            "prob_columns": prob_cols,
            "header": hdr,
        }

    def open_cube(self):
        """Open `flux.fits` memory-mapped. Caller closes.

        Returns (hdulist, wav_um, apertures_au). The VALUES array is
        `hdulist['VALUES'].data` with shape (n_models, n_ap, n_wav); it is
        NOT read into memory until indexed.
        """
        hdul = fits.open(self._path("flux.fits"), memmap=True)
        wav = np.asarray(hdul["SPECTRAL_INFO"].data["WAVELENGTH"], dtype=float)
        try:
            ap = np.asarray(hdul["APERTURES"].data["APERTURE"], dtype=float)
        except KeyError:
            ap = np.array([np.nan])
        return hdul, wav, ap

    def __repr__(self):
        m = self.meta
        kind = "aperture-dependent" if m.aperture_dependent else "aperture-independent"
        return (f"<ModelLibrary {m.key}: {m.n_models} models, {kind}, "
                f"nap={m.n_apertures}, info={'yes' if m.has_info else 'no'}>")
