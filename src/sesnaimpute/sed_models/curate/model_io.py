"""
model_io.py
====================================================================
Shared, streaming-only FITS reader for model SED releases (YSO, SPS,
galaxy, ...) that follow the MODEL_NAMES / SPECTRAL_INFO / APERTURES /
VALUES release convention. Lives at the `sed_models_curate` package level, not
inside any one model-type subpackage, because every subpackage
(yso_soft_classify, yso curation, and future SPS/galaxy curation work) needs
to read model flux files the same way.

Deliberately generic: no scheme-specific constants or defaults live here
(e.g. no aperture-window default -- callers pass their own). The one
YSO-specific accommodation is that `inclination` is treated as an OPTIONAL
column (SPS/galaxy libraries have no viewing-angle concept and won't have
one) -- same graceful-degradation treatment already used for the optional
'Source Luminosity' cross-check column.

Named `model_io` rather than `io` to avoid any confusion with the stdlib
`io` module when imported from sibling subpackages.
====================================================================
"""

import configparser
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from astropy.io import fits
except ImportError:
    fits = None

from sesnaimpute.sed_models.constants import C_UM_S, LOGD_STEP, L_SUN_ERG_S, MJY_TO_CGS


@dataclass
class ModelSet:
    """In-memory handle to a block of models from one geometry/library."""
    lam_um: np.ndarray               # (n_wav,)
    nu_hz: np.ndarray                 # (n_wav,)  from SPECTRAL_INFO FREQUENCY col
    apertures_au: np.ndarray           # (n_ap,)
    model_names: np.ndarray             # (n_models,) str
    group_key: np.ndarray                # (n_models,) str, geometry-prefixed
    values: np.ndarray                    # (n_models, n_ap, n_wav)  F_nu [mJy]
    inclination: Optional[np.ndarray] = None   # (n_models,) deg, if the release has one


@dataclass
class FluxMetadata:
    """Everything about a flux.fits file except the (large) VALUES array
    itself. Return type of read_flux_metadata."""
    names: np.ndarray
    lam_um: np.ndarray
    nu_hz: np.ndarray
    apertures_au: np.ndarray
    order: tuple            # VALUES axis order -> (models, ap, wav)
    vshape: tuple
    distance_cm: Optional[float]
    group_key_raw: np.ndarray   # name.rsplit('_',1)[0], NOT geometry-prefixed


def axes_to_models_ap_wav(shape, n_models, n_ap, n_wav):
    """
    Return the numpy transpose order that maps a VALUES array of `shape` to
    (models, ap, wav), identified by matching axis LENGTHS rather than
    assuming a FITS/numpy convention. Robust whether astropy returns
    (models,ap,wav) directly or the reversed (wav,ap,models), as long as
    n_models/n_ap/n_wav are pairwise distinct for the file in question.
    """
    want = (n_models, n_ap, n_wav)
    if tuple(shape) == want:
        return (0, 1, 2)
    order = []
    for target in want:
        matches = [ax for ax, n in enumerate(shape) if n == target and ax not in order]
        if not matches:
            raise ValueError(f"VALUES shape {shape} has no axis of length {target} "
                             f"(n_models={n_models}, n_ap={n_ap}, n_wav={n_wav}).")
        order.append(matches[0])
    return tuple(order)


def read_flux_metadata(flux_path: str) -> FluxMetadata:
    """
    Low-level reader for flux.fits metadata: names, wavelength/frequency
    grid, apertures, VALUES shape/axis-order, distance. Reads headers and
    small tables only -- never the VALUES array itself.

    Runs generic sanity checks unconditionally (BUNIT, wavelength range);
    scheme-specific checks (e.g. a luminosity cross-check) are the caller's
    responsibility, since not every release has the columns needed for them.
    """
    if fits is None:
        raise RuntimeError("astropy is required to read FITS files.")

    with fits.open(flux_path, memmap=True) as hdul:
        names = np.asarray(hdul["MODEL_NAMES"].data.field(0)).astype(str)
        spec = hdul["SPECTRAL_INFO"].data
        lam = np.asarray(spec.field(0), dtype=float)     # WAVELENGTH [um]
        try:
            nu = np.asarray(spec["FREQUENCY"], dtype=float)
        except (KeyError, AttributeError):
            nu = C_UM_S / lam
        aps = np.asarray(hdul["APERTURES"].data.field(0), dtype=float)
        vshape = tuple(hdul["VALUES"].shape)
        order = axes_to_models_ap_wav(vshape, names.size, aps.size, lam.size)
        distance_cm = hdul["PRIMARY"].header.get("DISTANCE", None)
        bunit = hdul["VALUES"].header.get("BUNIT", "").strip()

    if bunit and bunit.lower() not in ("mjy",):
        warnings.warn(f"{flux_path}: VALUES BUNIT='{bunit}', expected 'mJy' "
                      "(F_nu). Downstream feature scaling assumes F_nu in mJy.")
    if not (1e-2 < np.nanmedian(lam) < 1e5):
        warnings.warn(f"{flux_path}: WAVELENGTH median={np.nanmedian(lam):.3g}; "
                      "expected um.")

    gk_raw = np.array([n.rsplit("_", 1)[0] for n in names])
    return FluxMetadata(names=names, lam_um=lam, nu_hz=nu, apertures_au=aps,
                        order=order, vshape=vshape,
                        distance_cm=(float(distance_cm) if distance_cm else None),
                        group_key_raw=gk_raw)


def make_model_id(geometry_folder, row_index):
    """
    The canonical row-identity key for this model grid:
    `<geometry_folder>:<row_index>`, NOT `MODEL_NAME`. `MODEL_NAME` is not
    reliably unique (1,258 values repeat within s-pbhmi/s-pbsmi alone, all
    on already-invalid placeholder rows -- an upstream release artifact),
    while `row_index` (a row's position within that geometry's
    parameters.fits/info.fits/flux.fits, verified to share row order) is
    collision-free by construction and needs no lookup: any consumer
    streaming these files in native order (FluxReader.iter_blocks /
    iter_group_blocks) already knows both `geometry_folder` and the row's
    index, so it can reconstruct this exact id independently. Lives here
    (not in yso_curation_state.py, where it originated) since it is a
    property of the model grid itself, not of the curation state
    specifically -- every consumer that streams a geometry's files
    (labeling, dedup, fps) needs the same id, not just the curation state.
    """
    return f"{geometry_folder}:{row_index}"


def model_ids_for_names(geometry_folder, model_names, flux_path):
    """
    Vectorized model_id lookup for one geometry: given that geometry's
    `flux.fits` path and an array of `model_name` values (e.g. a labeling/
    dedup/fps phase's own output), returns the corresponding model_id array
    by resolving each name to its row position in flux.fits' MODEL_NAMES
    table -- the same canonical row-order reference `make_model_id` itself
    is built on. Cheap: reads flux.fits metadata only (headers/small
    tables via `read_flux_metadata`), never the VALUES array, so re-doing
    this lookup in a downstream phase is not a repeat of the expensive
    convolved-flux read.
    """
    meta = read_flux_metadata(flux_path)
    name_to_idx = {name: i for i, name in enumerate(meta.names)}
    return np.array([make_model_id(geometry_folder, name_to_idx[name]) for name in model_names])


def align_by_name(reference_names, other_names):
    """
    Index array `idx` such that other_names[idx] lines up with
    reference_names elementwise -- used to align parameters.fits rows
    (arbitrary order) to flux.fits MODEL_NAMES order. Fast path when
    already aligned (the common case) avoids building the lookup dict.
    """
    reference_names = np.asarray(reference_names)
    other_names = np.asarray(other_names)
    if np.array_equal(reference_names, other_names):
        return np.arange(reference_names.size)
    idx = {nm: i for i, nm in enumerate(other_names)}
    return np.array([idx[nm] for nm in reference_names])


def load_convolved_total_flux_mjy(model_dir: str, band: str):
    """MODEL_NAME (stripped) and TOTAL_FLUX (mJy) from
    convolved/{band}.fits, for any library. Returns (names, flux_mjy).

    The first aperture only -- correct as-is for aperture-independent
    libraries (which carry exactly one), and the reference/first aperture
    for aperture-dependent ones; callers needing a specific aperture
    should interpolate the ConvolvedFluxes themselves rather than widen
    this."""
    with fits.open(os.path.join(model_dir, "convolved", f"{band}.fits")) as hdul:
        data = hdul["CONVOLVED FLUXES"].data
        names = np.array([str(n).strip() for n in data["MODEL_NAME"]])
        flux_mjy = data["TOTAL_FLUX"][:, 0]
    return names, flux_mjy


def read_params_columns(params_path: str, names, extra_columns=()):
    """
    Read optional columns from parameters.fits, aligned to `names` (flux.fits
    MODEL_NAMES order) via align_by_name. Returns a dict with keys
    'inclination', 'Source Luminosity', and any names listed in
    `extra_columns` -- each value is None if that column isn't present in
    this release (e.g. SPS/galaxy libraries have no 'inclination' column at
    all, since there's no viewing-angle concept for a static photosphere;
    this is treated the same as any other optional column, not a special
    case or a required one).
    """
    if fits is None:
        raise RuntimeError("astropy is required to read FITS files.")
    wanted = ("inclination", "Source Luminosity") + tuple(extra_columns)
    with fits.open(params_path, memmap=True) as ph:
        pdata = ph[1].data
        pnames = np.asarray(pdata["MODEL_NAME"]).astype(str)
        order = align_by_name(names, pnames)
        cols = pdata.columns.names
        out = {}
        for col in wanted:
            if col in cols:
                out[col] = np.asarray(pdata[col], dtype=float)[order]
            else:
                out[col] = None
    return out


def luminosity_crosscheck(ms: ModelSet, source_lum, distance_cm, n=5):
    """
    Independent check that VALUES is F_nu (not nu*F_nu). L_bol from the SED =
    4 pi d^2 * int F_nu dnu (largest aperture, total flux). Compare to a
    known-luminosity column (e.g. 'Source Luminosity'). A ~unity ratio
    confirms F_nu; a systematic offset by <nu> would flag a wrong
    convention. Warns only (never raises). Caller decides whether to run
    this at all -- not every release has a comparable luminosity column.
    """
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    ap_last = ms.apertures_au.size - 1
    rows = np.linspace(0, ms.values.shape[0] - 1, n).astype(int)
    ratios = []
    for r in rows:
        fnu = ms.values[r, ap_last, :].astype(float)
        m = np.isfinite(ms.nu_hz) & np.isfinite(fnu) & (fnu > 0.0)
        if m.sum() < 3 or not np.isfinite(source_lum[r]) or source_lum[r] <= 0:
            continue
        order = np.argsort(ms.nu_hz[m])
        nn, ff = ms.nu_hz[m][order], fnu[m][order]
        f_bol = trapz(ff * MJY_TO_CGS, nn)             # erg/s/cm^2
        L = 4 * np.pi * distance_cm**2 * f_bol / L_SUN_ERG_S
        ratios.append(L / source_lum[r])
    if ratios:
        med = float(np.nanmedian(ratios))
        if not (0.3 < med < 3.0):
            warnings.warn(
                f"Luminosity cross-check: median L_SED/L_source = {med:.3g} "
                f"(expected ~1 for F_nu). A large offset may indicate a wrong "
                f"flux convention or aperture/distance mismatch.")
        return med
    return None


class FluxReader:
    """
    Memory-light reader for one model library/geometry's flux.fits. Loads
    metadata only (names, wavelengths, apertures, optional columns); SED
    values are read per block on demand via .model_set(rows), so the full
    VALUES cube is never resident.

    check_luminosity requires an explicit `luminosity_column` name (e.g.
    'Source Luminosity' for the YSO release) since not every model library
    has a comparable column -- no default is assumed here.
    """

    def __init__(self, flux_path, params_path=None, geometry=None,
                check_luminosity=False, luminosity_column="Source Luminosity",
                extra_param_columns=()):
        self.flux_path = flux_path
        self.geometry = geometry or flux_path

        meta = read_flux_metadata(flux_path)
        self.lam_um = meta.lam_um
        self.nu_hz = meta.nu_hz
        self.apertures_au = meta.apertures_au
        self.names = meta.names
        self._vshape = meta.vshape
        self._order = meta.order
        pfx = f"{self.geometry}:"
        self.group_key = np.array([pfx + g for g in meta.group_key_raw])
        self.n_models = self.names.size

        self.inclination = None
        self.params = {}
        if params_path is not None:
            self.params = read_params_columns(params_path, self.names,
                                              extra_columns=extra_param_columns)
            self.inclination = self.params.get("inclination")

        if check_luminosity and meta.distance_cm:
            source_lum = self.params.get(luminosity_column)
            if source_lum is not None:
                n_check = min(5, self.n_models)
                rows = np.linspace(0, self.n_models - 1, n_check).astype(int)
                ms_check = self.model_set(rows)
                luminosity_crosscheck(ms_check, source_lum[rows], meta.distance_cm, n=n_check)

    def model_set(self, rows) -> ModelSet:
        """Return a ModelSet holding only `rows` (read from disk on demand)."""
        rows = np.asarray(rows)
        with fits.open(self.flux_path, memmap=True) as h:
            raw = h["VALUES"].data
            model_axis = self._order.index(0) if 0 in self._order else 0
            sl = [slice(None)] * raw.ndim
            sl[model_axis] = rows
            block = np.asarray(raw[tuple(sl)], dtype=np.float32)
            block = np.transpose(block, self._order)   # -> (len(rows), n_ap, n_wav)
        incl = self.inclination[rows] if self.inclination is not None else None
        return ModelSet(lam_um=self.lam_um, nu_hz=self.nu_hz,
                        apertures_au=self.apertures_au, model_names=self.names[rows],
                        group_key=self.group_key[rows], values=block, inclination=incl)

    def iter_group_blocks(self, target_block=4000):
        """
        Yield ModelSets covering whole groups, ~`target_block` models each, so
        every physical group stays intact within a block (needed for spread
        calibration). Groups are contiguous runs of the same group_key.
        """
        gk = self.group_key
        starts = [0] + [i for i in range(1, self.n_models) if gk[i] != gk[i - 1]]
        starts.append(self.n_models)
        i = 0
        while i < len(starts) - 1:
            lo = starts[i]
            j = i
            while j < len(starts) - 1 and (starts[j + 1] - lo) < target_block:
                j += 1
            hi = starts[j + 1] if j + 1 < len(starts) else self.n_models
            yield self.model_set(np.arange(lo, hi))
            i = j + 1

    def iter_blocks(self, block=4000):
        """Yield (lo, hi, ModelSet) for up to `block` consecutive models."""
        for lo in range(0, self.n_models, block):
            hi = min(lo + block, self.n_models)
            yield lo, hi, self.model_set(np.arange(lo, hi))


def aperture_indices(ms: ModelSet, au_range):
    """
    Indices of the model's aperture grid whose radius falls within the
    physical AU window `au_range = (lo_au, hi_au)` (inclusive). Resolved per
    loaded grid, so the same physical window selects the correct indices
    for any geometry/library's aperture grid. No default -- callers supply
    their own scheme-specific window (see e.g.
    yso_soft_classify.constants.APERTURE_AU_RANGE). Returns a possibly-empty
    integer array (empty -> that model's SED will be all-NaN, by design).
    """
    lo, hi = au_range
    return np.where((ms.apertures_au >= lo) & (ms.apertures_au <= hi))[0]


def aperture_window_aggregate(flux_by_aperture, ap_idx, axis):
    """
    Masked-median aggregate over an aperture axis, given already-resolved
    in-window aperture indices (`aperture_indices`) and which axis of
    `flux_by_aperture` is the aperture axis. Non-positive values are
    masked before the median (never treated as real flux, matching
    `bracket_sed`'s original behavior).

    Shared low-level piece behind `bracket_sed`/`bracket_sed_block`
    (full SED cubes, aperture axis in the middle) and any single-band,
    no-wavelength-axis aggregation (e.g. a convolved `{band}.fits`
    TOTAL_FLUX array, aperture axis last) -- the aperture-window-
    selection + median-aggregation logic is identical in spirit in both
    cases (curation plan Sec 3.1a), so it lives here once rather than
    being reimplemented per caller with two chances to handle the AU
    window boundary inconsistently.

    Caller's responsibility: `ap_idx` non-empty (an empty window is a
    degenerate case each caller special-cases in its own all-NaN shape).
    """
    window = np.take(flux_by_aperture, ap_idx, axis=axis).astype(float)
    window = np.where(window > 0.0, window, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(window, axis=axis)


def bracket_sed(ms: ModelSet, model_row: int, au_range):
    """
    Aggregate the SED over the physical aperture window `au_range` for one
    model row: per-wavelength MEDIAN over the in-window apertures (robust to
    small-aperture dropouts), with non-positive values masked first. Models
    are never dropped; if no aperture falls in the window, or all in-window
    fluxes are non-positive, the returned fnu is all-NaN. Returns
    (lam, nu, fnu[mJy]).
    """
    ap = aperture_indices(ms, au_range)
    if ap.size == 0:
        return ms.lam_um, ms.nu_hz, np.full(ms.lam_um.size, np.nan)
    fnu = aperture_window_aggregate(ms.values[model_row], ap, axis=0)   # (n_wav,)  F_nu [mJy]
    return ms.lam_um, ms.nu_hz, fnu


def bracket_sed_block(ms: ModelSet, au_range):
    """
    Vectorized `bracket_sed` for every row in `ms` at once: the aperture
    window (`au_range`) is the same for every row in a block, so the
    per-wavelength median only needs computing once for the whole block
    rather than once per row -- verified to give bit-identical results to
    looping `bracket_sed` over each row, ~3.7x faster for the aggregation
    step alone (which is ~half the per-row cost of the full feature
    pipeline; see sed_models_curate/docs/curation_workflow.md). Returns
    (lam, nu, fnu[mJy]) where fnu has shape (n_rows, n_wav), rows in the
    same order as `ms`.
    """
    ap = aperture_indices(ms, au_range)
    n_rows = ms.values.shape[0]
    if ap.size == 0:
        return ms.lam_um, ms.nu_hz, np.full((n_rows, ms.lam_um.size), np.nan)
    fnu = aperture_window_aggregate(ms.values, ap, axis=1)   # (n_rows, n_wav)  F_nu [mJy]
    return ms.lam_um, ms.nu_hz, fnu


# ====================================================================
# Output-contract validation
# ====================================================================
#
# One check that every library's driver can run as its last step, so a
# model directory cannot ship in a state `sedfitter` refuses to load.
#
# It exists because five independent `flux.fits` writers drifted apart:
# the YSO strata shipped without column units (so `SEDCube.read` raised
# on the production fitting path), without `DISTANCE` on stellar.fits,
# and with `np.arange` where the PRIMARY validity mask belongs. Each was
# individually invisible; all of them are caught below in ~1.5 s per
# directory.
#
# Deliberately NOT wired into the writers: this validates the artifact on
# disk, so it stays honest even if a writer is bypassed or hand-edited.

CONVOLVED_BANDS = ("J", "H", "Ks", "I1", "I2", "I3", "I4", "M1")
MODELS_CONF_REQUIRED_KEYS = ("name", "length_subdir", "aperture_dependent", "logd_step")

# Per-library deviations from the contract. Every entry is a DECISION with
# a reason, not an oversight -- `docs/model_library_standard.md` §1: divergence with a
# written reason is conformant; silent divergence is the finding.
CONTRACT_PROFILES = {
    #                    info.fits  stellar.fits  version key   band names
    "galaxy":  dict(info=True,  stellar=False, require_version=True,  bands=CONVOLVED_BANDS),
    "h2shock": dict(info=False, stellar=False, require_version=True,  bands=CONVOLVED_BANDS),
    # pahc bumped models.conf to version 3 at the cool-giant anchor rebuild
    # (pahc_decisions.md D-15): parameters.fits' schema changed (a per-model
    # LOGG column replaces the old fixed PRIMARY-header keyword) and T_EFF
    # grew from 8 to 11 anchors -- real content changes, not a relabel, so
    # this profile expects "3" where every other profile still expects "2".
    "pahc":    dict(info=False, stellar=False, require_version=True,  bands=CONVOLVED_BANDS,
                    version="3"),
    # SPS was the one profile needing exemptions here: its models.conf was
    # copied verbatim (no `version` key) and its convolved/ kept the 2008
    # distribution's own 2J/2H/2K spelling. Both are gone -- sps_curate now
    # writes the conf with version = 2 and emits the standard band names -- so
    # SPS takes the same row as everything else. Its remaining deviations
    # (30A MODEL_NAME, no info.fits) are expressed by the fields below.
    "sps":     dict(info=False, stellar=False, require_version=True,  bands=CONVOLVED_BANDS),
    "yso":     dict(info=True,  stellar=True,  require_version=True,  bands=CONVOLVED_BANDS),
    # Stage B / AGB-ext (agb_decisions.md B-9/B-11): 30A MODEL_NAME (same
    # cube-width choice as sps -- see sps_curate.MODEL_NAME_COLUMN_FORMAT's
    # docstring for why 30A rather than the project-wide 34A), no
    # info.fits/stellar.fits (no per-model provenance columns beyond
    # parameters.fits, no photosphere companion cube), the standard 8-band
    # convolved/ set (B-11, no Gaia G).
    "agb":     dict(info=False, stellar=False, require_version=True,  bands=CONVOLVED_BANDS),
}

# h2shock and pahc omit info.fits (no per-model provenance columns to carry);
# only YSO emits stellar.fits (the photosphere-only companion cube).


def validate_model_directory(path, profile="galaxy"):
    """Check an emitted model directory against the release contract.

    Returns a list of problem strings, empty if the directory is
    conformant -- so a driver ends with

        problems = validate_model_directory(out_dir, profile="h2shock")
        if problems:
            raise SystemExit("\\n".join(problems))

    `profile` selects the per-library exemptions in CONTRACT_PROFILES, or
    may be a dict with the same keys.

    Checks, by code:
      C1  required files present
      C2  HDU names and order
      C3  PRIMARY is an all-ones validity mask (not arange, not absent)
      C4  PRIMARY carries DISTANCE
      C5  column units: WAVELENGTH um, FREQUENCY Hz, APERTURE AU, TOTAL_FLUX mJy
      C6  WAVELENGTH strictly descending (sidesteps SEDCube.read(order='nu'))
      C7  VALUES BUNIT
      C8  VALUES shape agrees with n_models / n_apertures / n_wav
      C9  MODEL_NAME sets agree across every file, compared STRIPPED
      C10 MODEL_NAME width uniform across the cubes, and wide enough.
          convolved/ is checked only for truncation: sedfitter itself
          writes those names as S30 (convolved_fluxes.py), so their width
          is fixed upstream and is NOT expected to match the cubes'.
      C11 EXTNAME on parameters.fits / info.fits
      C12 parameter column formats
      C13 models.conf keys
      C14 aperture_dependent agrees with the aperture count
      C15 length_subdir implies a seds/ directory
      C16 convolved/ present and complete for the band set
      C17 convolved/ aperture count agrees with flux.fits
      C18 convolved/ fluxes are finite
      C19 classmap.fits HDU names and order (SUBCLASS_PROB optional)
      C20 CLASSMAP MODEL_NAME matches flux.fits ELEMENTWISE AND IN ORDER
      C21 every SUBCLASS is non-blank and appears in SUBCLASS_LEGEND
      C22 CLASS is a single value, matching the PRIMARY CLASS card and
          present in CLASS_LEGEND
      C23 provenance cards present, and NMODELS agrees with the row count
    """
    if fits is None:
        raise ImportError("astropy is required for validate_model_directory")
    prof = CONTRACT_PROFILES[profile] if isinstance(profile, str) else profile
    problems = []

    def chk(condition, code, message):
        if not condition:
            problems.append(f"{code}: {message}")

    # ---- C1 ------------------------------------------------------------
    for filename in ("flux.fits", "parameters.fits", "models.conf", "classmap.fits"):
        chk(os.path.exists(os.path.join(path, filename)), "C1", f"missing {filename}")
    if prof["info"]:
        chk(os.path.exists(os.path.join(path, "info.fits")), "C1", "missing info.fits")
    if prof["stellar"]:
        chk(os.path.exists(os.path.join(path, "stellar.fits")), "C1", "missing stellar.fits")

    def _check_cube(filename, has_apertures, has_uncertainties):
        """C2-C8 for one flux-cube file. Returns {names, name_format, n_ap}."""
        out = {}
        with fits.open(os.path.join(path, filename), memmap=True) as hdul:
            got = [h.name for h in hdul]
            want = (["PRIMARY", "MODEL_NAMES", "SPECTRAL_INFO"]
                    + (["APERTURES"] if has_apertures else [])
                    + ["VALUES"]
                    + (["UNCERTAINTIES"] if has_uncertainties else []))
            chk(got == want, "C2", f"{filename} HDU order {got} != {want}")

            primary = hdul[0].data
            is_arange = primary is not None and np.array_equal(primary, np.arange(primary.size))
            chk(primary is not None and np.all(primary == 1), "C3",
                f"{filename} PRIMARY is not an all-ones validity mask"
                + (" (it is np.arange -- row 0 reads as invalid)" if is_arange else ""))
            chk("DISTANCE" in hdul[0].header, "C4", f"{filename} PRIMARY lacks DISTANCE")

            spectral = hdul["SPECTRAL_INFO"]
            units = {c.name: c.unit for c in spectral.columns}
            chk(units.get("WAVELENGTH") == "um", "C5",
                f"{filename} WAVELENGTH unit={units.get('WAVELENGTH')!r}")
            chk(units.get("FREQUENCY") == "Hz", "C5",
                f"{filename} FREQUENCY unit={units.get('FREQUENCY')!r}")
            if has_apertures:
                aperture_unit = hdul["APERTURES"].columns[0].unit
                chk(aperture_unit == "AU", "C5", f"{filename} APERTURE unit={aperture_unit!r}")
                out["n_ap"] = len(hdul["APERTURES"].data)

            lam = np.asarray(spectral.data["WAVELENGTH"], dtype=float)
            chk(np.all(np.diff(lam) < 0), "C6", f"{filename} WAVELENGTH is not strictly descending")
            bunit = hdul["VALUES"].header.get("BUNIT")
            chk(bunit == "mJy", "C7", f"{filename} VALUES BUNIT={bunit!r}")

            names = np.char.strip(np.asarray(hdul["MODEL_NAMES"].data.field(0)).astype(str))
            out["names"] = names
            out["name_format"] = hdul["MODEL_NAMES"].columns[0].format
            shape = hdul["VALUES"].shape
            chk(shape[0] == names.size and shape[-1] == lam.size, "C8",
                f"{filename} VALUES shape {shape} vs n_models={names.size} n_wav={lam.size}")
            if has_apertures:
                chk(shape[1] == out["n_ap"], "C8",
                    f"{filename} VALUES aperture axis {shape[1]} != {out['n_ap']}")
        return out

    flux = _check_cube("flux.fits", has_apertures=True, has_uncertainties=True)
    if prof["stellar"]:
        stellar = _check_cube("stellar.fits", has_apertures=False, has_uncertainties=False)
        chk(np.array_equal(flux["names"], stellar["names"]), "C9",
            "stellar.fits MODEL_NAMES differ from flux.fits")

    # ---- C9-C12 tables -------------------------------------------------
    cube_name_formats = {"flux.fits": flux["name_format"]}
    for filename, extname, required in (("parameters.fits", "PARAMETERS", True),
                                        ("info.fits", "INFO", prof["info"])):
        if not required:
            continue
        with fits.open(os.path.join(path, filename), memmap=True) as hdul:
            table = hdul[1]
            cube_name_formats[filename] = table.columns[0].format
            chk(table.name == extname, "C11", f"{filename} EXTNAME={table.name!r} != {extname!r}")
            names = np.char.strip(np.asarray(table.data.field(0)).astype(str))
            chk(sorted(names) == sorted(flux["names"]), "C9",
                f"{filename} MODEL_NAME set differs from flux.fits "
                f"(n={names.size} vs {flux['names'].size})")
            for column in table.columns[1:]:
                chk(column.format[-1] in "ADEJKIL", "C12",
                    f"{filename} column {column.name} has format {column.format}")

    chk(len(set(cube_name_formats.values())) == 1, "C10",
        f"MODEL_NAME width differs across cubes: {cube_name_formats}")
    longest = max(len(n) for n in flux["names"])
    chk(longest <= int(flux["name_format"][:-1]), "C10",
        f"longest MODEL_NAME is {longest} chars but format is {flux['name_format']}")

    # ---- C13-C15 models.conf -------------------------------------------
    parser = configparser.ConfigParser()
    with open(os.path.join(path, "models.conf")) as fh:
        parser.read_string("[conf]\n" + fh.read())
    conf = dict(parser["conf"])
    for key in MODELS_CONF_REQUIRED_KEYS:
        chk(key in conf, "C13", f"models.conf missing {key}")
    if "logd_step" in conf:
        chk(float(conf["logd_step"]) == LOGD_STEP, "C13",
            f"models.conf logd_step={conf['logd_step']!r} != "
            f"constants.LOGD_STEP={LOGD_STEP!r}")
    if prof["require_version"]:
        expected_version = prof.get("version", "2")
        chk(conf.get("version") == expected_version, "C13",
            f"models.conf version={conf.get('version')!r} != expected {expected_version!r}")
    n_ap = flux.get("n_ap", 1)
    chk((conf.get("aperture_dependent") == "yes") == (n_ap > 1), "C14",
        f"models.conf aperture_dependent={conf.get('aperture_dependent')!r} but NAP={n_ap}")
    length_subdir = int(conf.get("length_subdir", 0))
    chk(length_subdir == 0 or os.path.isdir(os.path.join(path, "seds")), "C15",
        f"models.conf length_subdir={length_subdir} but there is no seds/ directory")

    # ---- C16-C18 convolved/ --------------------------------------------
    convolved_dir = os.path.join(path, "convolved")
    chk(os.path.isdir(convolved_dir), "C16", "no convolved/ directory")
    if os.path.isdir(convolved_dir):
        found = tuple(sorted(f[:-5] for f in os.listdir(convolved_dir) if f.endswith(".fits")))
        chk(found == tuple(sorted(prof["bands"])), "C16",
            f"convolved/ bands {found} != {tuple(sorted(prof['bands']))}")
        for band in found:
            with fits.open(os.path.join(convolved_dir, f"{band}.fits"), memmap=True) as hdul:
                table = hdul[1]
                names = np.char.strip(np.asarray(table.data["MODEL_NAME"]).astype(str))
                chk(sorted(names) == sorted(flux["names"]), "C9",
                    f"convolved/{band}.fits MODEL_NAME set differs from flux.fits")
                width = int(table.columns[0].format[:-1])
                chk(max(len(n) for n in names) <= width, "C10",
                    f"convolved/{band}.fits names risk truncation at {table.columns[0].format}")
                total_flux = np.asarray(table.data["TOTAL_FLUX"])
                band_n_ap = total_flux.shape[1] if total_flux.ndim > 1 else 1
                chk(band_n_ap == n_ap, "C17",
                    f"convolved/{band}.fits NAP={band_n_ap} != flux.fits NAP={n_ap}")
                chk(table.columns["TOTAL_FLUX"].unit == "mJy", "C5",
                    f"convolved/{band}.fits TOTAL_FLUX unit="
                    f"{table.columns['TOTAL_FLUX'].unit!r}")
                chk(np.isfinite(total_flux).all(), "C18",
                    f"convolved/{band}.fits contains non-finite fluxes")

    # ---- C19-C23 classmap.fits ------------------------------------------
    #
    # Nothing downstream recomputes these labels, so classmap.fits is the only
    # record of them -- which also means nothing else would catch it going
    # wrong. The checks are deliberately WITHIN one library: a library's CLASS
    # is whatever its own classmap says it is, so there is no cross-library
    # collision check here and none is wanted.
    classmap_path = os.path.join(path, "classmap.fits")
    if os.path.exists(classmap_path):
        with fits.open(classmap_path, memmap=True) as hdul:
            got = [h.name for h in hdul]
            want = ["PRIMARY", "CLASSMAP", "CLASS_LEGEND", "SUBCLASS_LEGEND"]
            # SUBCLASS_PROB is optional -- only libraries whose SUBCLASS is the
            # argmax of a soft distribution emit it.
            chk(got == want or got == want + ["SUBCLASS_PROB"], "C19",
                f"classmap.fits HDU order {got} != {want} (+ optional SUBCLASS_PROB)")

            if "CLASSMAP" in got:
                table = hdul["CLASSMAP"]
                cm_names = np.char.strip(np.asarray(table.data["MODEL_NAME"]).astype(str))
                # ORDER, not just set: consumers join classmap to the cube
                # positionally, and C9-style sorted comparison would pass a
                # permutation.
                chk(np.array_equal(cm_names, flux["names"]), "C20",
                    "classmap.fits MODEL_NAME does not match flux.fits elementwise and in order")

                subclass = np.char.strip(np.asarray(table.data["SUBCLASS"]).astype(str))
                chk(not (subclass == "").any(), "C21",
                    f"classmap.fits has {int((subclass == '').sum())} blank SUBCLASS value(s); "
                    "a library that subclasses at all must subclass every model")
                if "SUBCLASS_LEGEND" in got:
                    legend = np.char.strip(
                        np.asarray(hdul["SUBCLASS_LEGEND"].data.field(0)).astype(str))
                    missing = sorted(set(np.unique(subclass)) - set(legend.tolist()) - {""})
                    chk(not missing, "C21",
                        f"SUBCLASS values {missing} are absent from SUBCLASS_LEGEND, "
                        "which is the authority for what subclasses exist")

                classes = np.unique(np.char.strip(np.asarray(table.data["CLASS"]).astype(str)))
                chk(classes.size == 1, "C22",
                    f"classmap.fits carries {classes.size} distinct CLASS values {list(classes)}; "
                    "one library declares one class")
                header = hdul[0].header
                if classes.size == 1:
                    chk(str(header.get("CLASS", "")).strip() == classes[0], "C22",
                        f"PRIMARY CLASS={header.get('CLASS')!r} disagrees with the "
                        f"CLASSMAP column value {classes[0]!r}")
                    if "CLASS_LEGEND" in got:
                        cl = np.char.strip(
                            np.asarray(hdul["CLASS_LEGEND"].data.field(0)).astype(str))
                        chk(classes[0] in set(cl.tolist()), "C22",
                            f"CLASS {classes[0]!r} is absent from CLASS_LEGEND")

                for key in ("CLASS", "CLASS_SOURCE", "SUBCLASS_SOURCE",
                            "SUBCLASS_REF", "SUBCLASS_NOTE", "LEGEND_SOURCE"):
                    chk(key in header, "C23", f"classmap.fits PRIMARY lacks {key}")
                chk(header.get("NMODELS") == cm_names.size, "C23",
                    f"classmap.fits NMODELS={header.get('NMODELS')} but the table has "
                    f"{cm_names.size} rows")

    return problems


# ====================================================================
# Output-contract writer
# ====================================================================
#
# One writer for every library's flux cube, replacing five independent
# copies that had drifted apart in six ways (PRIMARY contents, three
# column-unit fields, MODEL_NAME width, `overwrite` default).
#
# The drift was not cosmetic: the YSO strata shipped with `np.arange`
# where the validity mask belongs, with no column units, and with no
# DISTANCE on stellar.fits -- so `SEDCube.read` raised on the production
# fitting path. Units and DISTANCE are therefore REQUIRED ARGUMENTS
# here, not optional keywords: the defect class becomes unexpressible
# rather than merely fixed.
#
# `apertures_au=None` omits the APERTURES HDU entirely (the stellar.fits
# shape). A single-element array is the point-source shape (galaxy,
# h2shock, sps). These are different files, not two readings of one flag.

MODEL_NAME_FORMAT = "34A"        # cubes: flux/stellar/parameters/info
# convolved/*.fits use 30A instead, and that is NOT a divergence to fix:
# sedfitter writes those names itself as astype('S30')
# (convolved_fluxes/convolved_fluxes.py), so the width is fixed upstream.


def write_flux_cube(path, *, names, wave_um_desc, freq_hz_desc, values,
                    distance_cm, apertures_au=None, uncertainties=None,
                    valid=None, name_format=MODEL_NAME_FORMAT,
                    distance_comment="reference distance for the aperture frame",
                    header_extras=None, values_header_extras=None,
                    uncertainties_header_extras=None, overwrite=True):
    """Write one model flux cube in the release convention.

    Parameters
    ----------
    path : str
    names : (n_models,) str array          MODEL_NAME column
    wave_um_desc : (n_wav,) float          wavelength [um], STRICTLY DESCENDING
    freq_hz_desc : (n_wav,) float          frequency [Hz], ascending
    values : (n_models, n_ap, n_wav) float F_nu [mJy]
    distance_cm : float                    REQUIRED -- SEDCube.read reads this
                                           unconditionally; stellar.fits shipped
                                           without it and raised KeyError.
    apertures_au : (n_ap,) float or None   None omits the APERTURES HDU
                                           (stellar.fits shape). One element is
                                           the point-source shape.
    uncertainties : same shape as `values`, or None to omit the HDU
    valid : (n_models,) int or None        None -> all ones. Anything else is
                                           almost certainly a mistake: sedfitter
                                           reads this as a boolean mask, so a 0
                                           silently drops that model.
    name_format : str                      cube width; see MODEL_NAME_FORMAT
    distance_comment : str                 preserved per library so existing
                                           files reproduce byte-for-byte
    header_extras : dict or None           extra PRIMARY header cards
    values_header_extras : dict or None    extra cards for VALUES.
    uncertainties_header_extras : dict      extra cards for UNCERTAINTIES;
                                           defaults to `values_header_extras`.
                                           Values may be `v` or `(v, comment)`,
                                           astropy's own convention. Needed by libraries
                                           whose stored values are scale-free:
                                           `BUNIT='mJy'` is required by
                                           sedfitter's unit conversion, so the
                                           REAL normalisation has to be declared
                                           in a second keyword, e.g.
                                           `{"FLUXNORM": "unit total line flux"}`.
                                           Without this the shared writer cannot
                                           express what h2shock already ships.

    Wavelength order is asserted, not assumed: every library stores
    descending to sidestep an upstream SEDCube.read(order='nu') reorder
    bug, and that convention is only safe if it is enforced at the one
    place all five libraries now pass through.
    """
    if fits is None:
        raise RuntimeError("astropy is required to write FITS files.")

    names = np.asarray(names)
    wave_um_desc = np.asarray(wave_um_desc, dtype=float)
    freq_hz_desc = np.asarray(freq_hz_desc, dtype=float)
    n_models, n_wav = names.size, wave_um_desc.size

    if not np.all(np.diff(wave_um_desc) < 0):
        raise ValueError(f"{path}: wave_um_desc must be strictly descending "
                         "(the release convention; see module notes).")
    longest = max((len(str(n)) for n in names), default=0)
    if longest > int(name_format[:-1]):
        raise ValueError(f"{path}: longest MODEL_NAME is {longest} chars, "
                         f"which does not fit format {name_format!r}.")
    if values.shape[0] != n_models or values.shape[-1] != n_wav:
        raise ValueError(f"{path}: values shape {values.shape} disagrees with "
                         f"n_models={n_models}, n_wav={n_wav}.")

    n_ap = 1 if apertures_au is None else np.asarray(apertures_au).size
    primary = fits.PrimaryHDU(data=(np.ones(n_models, dtype=np.int64)
                                    if valid is None else np.asarray(valid)))
    primary.header["DISTANCE"] = (distance_cm, distance_comment)
    primary.header["NWAV"] = n_wav
    primary.header["NAP"] = n_ap
    for key, value in (header_extras or {}).items():
        primary.header[key] = value

    hdus = [primary,
            fits.BinTableHDU.from_columns(
                [fits.Column(name="MODEL_NAME", format=name_format, array=names)],
                name="MODEL_NAMES"),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="WAVELENGTH", format="D", unit="um", array=wave_um_desc),
                 fits.Column(name="FREQUENCY", format="D", unit="Hz", array=freq_hz_desc)],
                name="SPECTRAL_INFO")]

    if apertures_au is not None:
        hdus.append(fits.BinTableHDU.from_columns(
            [fits.Column(name="APERTURE", format="D", unit="AU",
                         array=np.asarray(apertures_au, dtype=float))],
            name="APERTURES"))

    values_hdu = fits.ImageHDU(data=values, name="VALUES")
    values_hdu.header["BUNIT"] = "mJy"
    for key, value in (values_header_extras or {}).items():
        values_hdu.header[key] = value
    hdus.append(values_hdu)

    if uncertainties is not None:
        unc_extras = (uncertainties_header_extras if uncertainties_header_extras is not None
                      else values_header_extras)
        unc_hdu = fits.ImageHDU(data=uncertainties, name="UNCERTAINTIES")
        unc_hdu.header["BUNIT"] = "mJy"
        for key, value in (unc_extras or {}).items():
            unc_hdu.header[key] = value
        hdus.append(unc_hdu)

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path


def write_models_conf(path, *, name, aperture_dependent, length_subdir=0,
                      logd_step=LOGD_STEP, version=2):
    """Write a `models.conf`, replacing three byte-identical copies.

    The three differed only in `name` and `aperture_dependent`, and each
    carried the same latent bug: `os.makedirs(os.path.dirname(path))`
    raises FileNotFoundError when handed a bare relative filename, because
    `dirname` is then `""`. Fixed here.

    `logd_step`'s default is `constants.LOGD_STEP`, the one project-wide
    grid spacing every SESNA library's `models.conf` shares -- not a
    scheme-specific default (a caller building an off-grid library still
    passes its own).

    NOT used by SPS, deliberately: its `models.conf` is copied verbatim
    from Robitaille's distribution and carries no `version` key, so
    sedfitter takes its v1 path. That is a documented exemption, not an
    oversight -- do not "fix" SPS by routing it through this function
    without re-reading why (it changes which files the fitter reads).
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    text = (f"name = {name}\n"
            f"length_subdir = {length_subdir}\n"
            f"aperture_dependent = {'yes' if aperture_dependent else 'no'}\n"
            f"logd_step = {logd_step}\n"
            f"version = {version}\n")
    with open(path, "w") as fh:
        fh.write(text)
    return path


def provenance_card(key, value):
    """Value for a full-word provenance keyword, checked to fit its card.

    Full-word keys become HIERARCH cards, and HIERARCH cannot use
    CONTINUE -- so a value that overflows 80 characters aborts the write
    with a generic astropy error, after the file is partly built.
    Failing here instead names the offending key and its exact budget.
    Anything that will not fit belongs in a table row, which has no
    length limit.
    """
    budget = 80 - len("HIERARCH ") - len(key) - len(" = ") - 2
    if len(value) > budget:
        raise ValueError(
            f"provenance value for {key!r} is {len(value)} chars, {budget} available; "
            f"shorten it or move the detail to a table")
    return value


def write_classmap_fits(path, *, names, class_id, subclass, class_legend,
                        subclass_legend, provenance, subclass_prob=None,
                        model_dir=None, name_format=MODEL_NAME_FORMAT,
                        overwrite=True):
    """Write one library's `classmap.fits` -- the authoritative per-model
    class/subclass label -- in the shape every library shares.

        PRIMARY           provenance
        CLASSMAP          MODEL_NAME, CLASS, SUBCLASS
        CLASS_LEGEND      CLASS, DESCRIPTION      -- full vocabulary (schema)
        SUBCLASS_LEGEND   SUBCLASS, DESCRIPTION   -- what THIS file describes
        SUBCLASS_PROB     MODEL_NAME, <one column per subclass>  -- OPTIONAL

    That asymmetry is deliberate: the class legend is a schema a consumer
    can rely on across libraries, while the subclass legend describes this
    particular file. With a hard label that means the subclasses actually
    shipped; with `subclass_prob` it means the full vocabulary, because the
    probability columns span it and a legend that did not would fail to
    describe columns in its own file.

    SUBCLASS_PROB exists for libraries whose SUBCLASS is the argmax of a
    soft distribution, where the hard label is lossy and the distribution
    is the real product. It is a SEPARATE HDU rather than extra CLASSMAP
    columns so that CLASSMAP keeps its three-column shape in every
    library and stays concatenable; a library with no soft labels simply
    omits it, and a consumer that does not want probabilities does not
    pay for them.

    Only the FITS assembly is shared. Each library keeps its own class id,
    legends and provenance text in its own module, because the five have
    genuinely different notions of a subclass and there is deliberately no
    shared enumeration to derive them from.

    Parameters
    ----------
    names : (n_models,) str      MODEL_NAME, in flux.fits row order
    class_id : str               this library's CLASS, written to every row
    subclass : (n_models,) str   per-model SUBCLASS; every value must appear
                                 in `subclass_legend`, which is the authority
                                 for what subclasses exist
    class_legend : sequence of (code, description)
    subclass_legend : sequence of (code, description)
    provenance : mapping or sequence of (key, value), in card order.
                 CLASS and NMODELS are written from `class_id` and `names`
                 and must NOT appear here.
    subclass_prob : (n_models, len(subclass_legend)) or None
                 per-model probability over the subclass vocabulary, in
                 `subclass_legend` order. Columns are named for their
                 subclass code, so a consumer maps by NAME and never has to
                 rely on column order.

                 The columns span the FULL `subclass_legend`, not the
                 filtered set SUBCLASS_LEGEND emits. A library shipped as
                 per-subclass folders has one subclass present per folder
                 while every model still carries a full distribution, so
                 filtering these columns would discard most of the content.

                 Validated on write: finite, rows sum to 1, and
                 `argmax` reproduces `subclass` exactly. That last check is
                 what stops this HDU and CLASSMAP disagreeing -- the same
                 hazard that makes CLASSMAP itself worth emitting from the
                 model set rather than by re-reading the library.
    model_dir : str or None      if given, MODEL_NAME is additionally asserted
                                 equal to flux.fits IN ORDER -- the invariant a
                                 sorted-set comparison would miss

    No build timestamp is written. A library is reproducible byte-for-byte
    from its driver, and a clock reading would destroy that for no gain:
    provenance describes the derivation, which does not change between runs.
    """
    names = np.array([str(n).strip() for n in np.asarray(names)])
    subclass = np.asarray(subclass).astype(str)
    if names.size == 0:
        raise ValueError("classmap needs at least one model")
    if subclass.size != names.size:
        raise ValueError(f"subclass has {subclass.size} entries for {names.size} models")
    if not subclass_legend:
        raise ValueError(
            "subclass_legend is empty; a library with no meaningful subclass "
            "still declares one code so the SUBCLASS_LEGEND HDU is never absent")

    if model_dir is not None:
        with fits.open(os.path.join(model_dir, "flux.fits")) as hdul:
            flux_names = np.array([str(n).strip()
                                   for n in hdul["MODEL_NAMES"].data.field(0)])
        if not np.array_equal(flux_names, names):
            raise ValueError("classmap MODEL_NAME does not match flux.fits in order")

    unknown = set(np.unique(subclass)) - {code for code, _ in subclass_legend}
    if unknown:
        raise ValueError(f"SUBCLASS values {sorted(unknown)} are not in subclass_legend, "
                         "which is the authority for what subclasses exist")

    items = (list(provenance.items()) if hasattr(provenance, "items")
             else list(provenance))
    reserved = {"CLASS", "NMODELS"} & {k for k, _ in items}
    if reserved:
        raise ValueError(f"{sorted(reserved)} are written from class_id/names; "
                         "remove them from provenance")

    primary = fits.PrimaryHDU()
    header = primary.header
    header["CLASS"] = provenance_card("CLASS", class_id)
    for key, value in items:
        header[key] = provenance_card(key, value)
    header["NMODELS"] = names.size

    width = max(len(code) for code, _ in subclass_legend)
    classmap = fits.BinTableHDU.from_columns([
        fits.Column(name="MODEL_NAME", format=name_format, array=names),
        fits.Column(name="CLASS", format=f"{len(class_id)}A",
                    array=np.full(names.size, class_id)),
        fits.Column(name="SUBCLASS", format=f"{width}A", array=subclass),
    ], name="CLASSMAP")

    def _legend(rows, code_column, code_width):
        return fits.BinTableHDU.from_columns([
            fits.Column(name=code_column, format=f"{code_width}A",
                        array=np.array([c for c, _ in rows])),
            fits.Column(name="DESCRIPTION",
                        format=f"{max(len(d) for _, d in rows)}A",
                        array=np.array([d for _, d in rows])),
        ], name=f"{code_column}_LEGEND")

    # SUBCLASS_LEGEND normally describes THIS file, so it is filtered to the
    # subclasses actually shipped. When subclass_prob is present that filter
    # would leave the file self-inconsistent: the probability columns span
    # the full vocabulary, so a legend cut to the shipped subclasses fails to
    # describe columns the same file carries. Shipping the distribution IS an
    # assertion about the whole vocabulary, so the legend follows it and stays
    # a true description of this file rather than becoming a schema.
    present = ([(c, d) for c, d in subclass_legend]
               if subclass_prob is not None
               else [(c, d) for c, d in subclass_legend if c in set(subclass)])
    hdus = [primary, classmap,
            _legend(class_legend, "CLASS",
                    max(len(c) for c, _ in class_legend)),
            _legend(present, "SUBCLASS", width)]

    if subclass_prob is not None:
        codes = [code for code, _ in subclass_legend]
        prob = np.asarray(subclass_prob, dtype=float)
        if prob.shape != (names.size, len(codes)):
            raise ValueError(
                f"subclass_prob is {prob.shape}, expected "
                f"{(names.size, len(codes))} -- one column per subclass_legend "
                "entry, in that order, spanning the FULL vocabulary")
        if not np.isfinite(prob).all():
            raise ValueError("subclass_prob contains non-finite values")
        row_sum = prob.sum(axis=1)
        if not np.allclose(row_sum, 1.0, rtol=0.0, atol=1e-9):
            worst = float(np.abs(row_sum - 1.0).max())
            raise ValueError(f"subclass_prob rows do not sum to 1 (worst |1-sum| = {worst:.3e})")
        implied = np.asarray(codes)[prob.argmax(axis=1)]
        if not np.array_equal(implied, subclass):
            n_bad = int((implied != subclass).sum())
            raise ValueError(
                f"subclass_prob argmax disagrees with SUBCLASS for {n_bad} model(s) -- "
                "the two would ship contradicting labels; emit both from the same "
                "arrays in the same run")
        hdus.append(fits.BinTableHDU.from_columns(
            [fits.Column(name="MODEL_NAME", format=name_format, array=names)]
            + [fits.Column(name=code, format="D", array=prob[:, i])
               for i, code in enumerate(codes)],
            name="SUBCLASS_PROB"))

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path


def fetch_pinned(url, sha256, dest, *, timeout=120, chunk_bytes=1 << 20,
                 postprocess=None, verify_dest=None):
    """Download `url` to `dest`, verifying it against `sha256` first.

    Replaces two unpinned fetches. Both had the same shape of defect:
    an unversioned URL plus a cache keyed on something other than the
    artifact itself, so a silent upstream change altered the build and
    an interrupted download could read as a complete cache.

    The cache key here is the VERIFIED artifact: if `dest` exists and
    already hashes to `sha256`, nothing is downloaded. A partial file
    fails the hash and is re-fetched, so there is no state in which a
    truncated download is mistaken for a good one.

    Downloads to `dest + ".part"` and renames only after the hash
    matches, so `dest` is never observed in a half-written state.

    `postprocess` and `verify_dest` are two small, OPTIONAL hooks added for
    callers that want to keep something other than the raw downloaded bytes
    at `dest` (e.g. sps_curate.fetch_btsettl_cifist gzip-compresses the
    verified ASCII before storing it, since the cache key is the sha256 of
    the RAW content but the artifact on disk is ~5x smaller gzipped).
    Both default to the previous behavior, so every existing caller is
    unaffected:

    postprocess(partial_path, dest_path) : callable or None
        If given, called after `partial`'s sha256 has been verified against
        `sha256`, IN PLACE OF the plain `os.replace(partial, dest)` rename.
        Responsible for producing `dest` from `partial` and for removing
        `partial` itself.
    verify_dest(dest_path) -> sha256 hex string : callable or None
        If given, used instead of hashing `dest` directly to decide whether
        an EXISTING `dest` already satisfies `sha256` (the idempotency
        check) -- needed when `postprocess` means `dest`'s bytes are not
        themselves the pinned content (e.g. they're gzipped, and gzip
        output is not even deterministic across runs/versions).
    """
    import hashlib
    import urllib.request

    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_bytes), b""):
                h.update(chunk)
        return h.hexdigest()

    dest_digest = verify_dest if verify_dest is not None else digest
    if os.path.exists(dest) and dest_digest(dest) == sha256:
        return dest

    directory = os.path.dirname(dest)
    if directory:
        os.makedirs(directory, exist_ok=True)
    partial = dest + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as response, open(partial, "wb") as out:
        while True:
            chunk = response.read(chunk_bytes)
            if not chunk:
                break
            out.write(chunk)

    got = digest(partial)
    if got != sha256:
        os.remove(partial)
        raise ValueError(f"{url}: sha256 mismatch\n  expected {sha256}\n  got      {got}\n"
                         "The upstream artifact changed, or the download was corrupted. "
                         "Do NOT update the expected hash without establishing which.")
    if postprocess is not None:
        postprocess(partial, dest)
    else:
        os.replace(partial, dest)
    return dest
