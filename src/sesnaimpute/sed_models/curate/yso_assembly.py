"""
yso_assembly.py
====================================================================
Assembly phase (curation plan `docs/yso_curation_plan.md` §4;
full mechanism spec `docs/yso_assembly_curation_plan.md`). Turns the curation
state's `fps.selected` output into five new,
self-contained pseudo-geometry folders (`STRATUM_OUTPUT_FOLDER`:
`c0`/`cI`/`cII`/`cIII`/`td`), each schema-matched to a native geometry
folder (`models.conf`, `info.fits`, `parameters.fits`, `flux.fits`,
`stellar.fits`, `convolved/{band}.fits`), but populated by pooling and
subsetting rows across all 18 native geometries.

**Scope of this module: assembly only.** Reads `fps.selected`;
performs no additional filtering of its own. Does not touch the
curation state (this phase's "output" is the five folders on disk, not a
new curation-state dataset).

**Efficiency design (spec §6):** each of the (up to 18) contributing
geometries is opened exactly once, across every file type, and its
selected rows are extracted in one combined read covering every stratum
it feeds -- not once per (geometry, stratum) pair. The five stage
output files are pre-created at their true final row count (already
known from FPS's own `n_k` allocation) and opened once in
`mode='update'` (memmapped) for the whole geometry sweep, so memory is
bounded by one geometry's selected-row chunk, never by the total output
size. Pre-allocating at final size is what FITS's fixed-shape format
requires (no extendable/chunked dataset concept, unlike HDF5), not an
arbitrary implementation choice.
====================================================================
"""

import glob
import os
import shutil
import tempfile
import time
import uuid

import numpy as np
import pandas as pd
from astropy.io import fits

from sesnaimpute.sed_models.constants import BANDS
from sesnaimpute.sed_models.curate import yso_curation_state as reg
from sesnaimpute.sed_models.curate.model_io import (
    align_by_name, axes_to_models_ap_wav, validate_model_directory, write_classmap_fits,
    write_flux_cube, write_models_conf,
)
from sesnaimpute.sed_models.curate.yso_fps import STRATUM_LABELS, STRATUM_OUTPUT_FOLDER
from sesnaimpute.sed_models.curate.yso_labeling import BAND_ORDER, _CONVOLVED_FILENAME, _CONVOLVED_SUBDIR
from sesnaimpute.sed_models.curate import yso_pms as pms

# ==========================================================================
# CONSTANTS (spec SS2)
# ==========================================================================

# Two genuinely different contracts, not one column unified two ways
# (model_directory_format.md §6): the four main files (info/parameters/
# flux/stellar) are OUR OWN format, widened to fit the longest name across
# all 18 native geometries -- MODEL_NAME_WIDTH = 34. convolved/*.fits is a
# different contract, fixed upstream by sedfitter's own
# convolved_fluxes.py:253 (`astype('S30')`) -- CONVOLVED_MODEL_NAME_WIDTH
# stays 30 and must never be unified with the first.
MODEL_NAME_WIDTH = 34
CONVOLVED_MODEL_NAME_WIDTH = 30

# qc_value_spotcheck's re-verification tolerance (named per model_directory_
# format.md's requirement that spot-check tolerances be stated, not implicit).
SPOTCHECK_RTOL = 1e-5
SPOTCHECK_ATOL = 0.0

# Full native parameters.fits union across all 18 geometries (spec SS2.3) --
# confirmed by inspecting every geometry's column set, not derived at
# runtime. 30 physical columns + MODEL_NAME (handled separately, since it
# is read alongside the union rather than being part of it).
PARAM_UNION_COLUMNS = (
    "star.radius", "star.temperature", "disk.mass", "disk.rmax", "disk.beta",
    "disk.p", "disk.h100", "disk.rmin", "envelope.rho_0", "envelope.rc",
    "envelope.power", "envelope.rmin", "cavity.power", "cavity.theta_0",
    "cavity.rho_0", "ambient.density", "ambient.temperature", "scattering",
    "inclination", "Source Luminosity", "Av", "Inner Radius", "Outer Radius",
    "Disk Minimum Q", "Line-of-Sight Masses", "Sphere Masses",
    "Line-of-Sight Mass-Weighted Temperatures",
    "Line-of-Sight Photon-Weighted Temperatures",
    "Sphere Mass-Weighted Temperatures", "Spectral Index",
)
# The union's aperture-indexed (20D) columns -- everything else is a plain
# scalar (D), except `scattering` (K, int64), which is present in ALL 18
# geometries (confirmed) and therefore never actually needs a missing-fill.
PARAM_APERTURE_ARRAY_COLUMNS = frozenset({
    "Spectral Index", "Line-of-Sight Masses", "Sphere Masses",
    "Line-of-Sight Mass-Weighted Temperatures",
    "Line-of-Sight Photon-Weighted Temperatures",
    "Sphere Mass-Weighted Temperatures",
})

N_APERTURES = 20      # confirmed universal across all 18 geometries (spec SS2.4)


# ==========================================================================
# CLASS / SUBCLASS VOCABULARY (classmap.fits)
# ==========================================================================
#
# This library's own vocabulary. There is deliberately no shared
# enumeration across the five libraries -- agreement is documentary, so
# these codes are settled with the other library owners rather than
# derived. Both levels come from `constants.GUTERMUTH_LABELS`, the
# project-wide label authority: YSO is its `pop_abbrev`, and C0/CI/CII/TD
# are its `abbrev` values for the same populations.

MODEL_CLASS = "YSO"

CLASS_LEGEND = (
    ("YSO", "Young stellar object (Richardson 2024 / Robitaille 2017 RT grid)"),
)

# THE AUTHORITY for what subclasses exist here. Subclasses are authored,
# not discovered: Richardson supplies the Stage column and Gutermuth the
# 24um criterion, but deciding that those constitute these five
# populations -- rather than four, or Stage alone -- is this project's
# decision. Changing this tuple is a definitional change.
#
# Order is the vocabulary order, and SUBCLASS_PROB's columns follow it.
#
# CIII is the one code with no GUTERMUTH_LABELS counterpart, and the reason
# is structural rather than an oversight: Gutermuth's scheme carries four
# YSO-population categories (DEEPLY_EMBEDDED, CLASS_I, CLASS_II,
# TRANSITION_DISK) and no Class III, because diskless YSOs have
# photospheric colours and his colour cuts cannot isolate them. So C0/CI/
# CII/TD map one-to-one onto his categories and CIII has nowhere to sit.
#
# It is still NOT the diskless-star class. Stage III is envelope- and
# disc-free, hence physically adjacent to SPS's bare photospheres -- which
# is exactly why dedup exists, and why the models that genuinely ARE
# indistinguishable from SPS get flagged and excluded. What ships as CIII
# is the remainder: young stars whose SED a bare photosphere does not
# reproduce. Do not merge the two vocabularies on the strength of the name.
#
# Consequence for consumers, not for this library: any method leaning on
# Gutermuth's cascade has less leverage on CIII than on the other four,
# because his scheme has no category to map it onto.
SUBCLASS_LEGEND = (
    ("C0",   "Stage 0: M_env > 0.1 Msun, T_star < 3000 K (deeply embedded)"),
    ("CI",   "Stage I: M_env > 0.1 Msun, T_star > 3000 K (embedded protostar)"),
    ("CII",  "Stage II: M_env < 0.1 Msun, disc present"),
    ("CIII", "Stage III: M_env < 0.1 Msun, no disc; not a bare photosphere"),
    ("TD",   "Transition disc: Stage II/III carved by the Gutermuth 24um criterion"),
)

# label (the curation state's argmax spelling) -> SUBCLASS code. Built from
# STRATUM_LABELS rather than written out, so a change to the stratum
# vocabulary cannot silently desynchronise the two.
SUBCLASS_BY_LABEL = dict(zip(STRATUM_LABELS, [c for c, _ in SUBCLASS_LEGEND]))
N_WAVELENGTHS = 200   # confirmed universal across all 18 geometries (spec SS2.4)

# Flattened output layout (spec SS2.5): convolved/{band}.fits, no survey
# subdirectory -- {band} is yso_labeling.BAND_ORDER's short code, matching
# galz's convolved/{band}.fits naming (sed_fit/docs/config_format.md:85-86)
# -- NOT SPS, which uses its own 2J/2H/2K-style band codes.
OUTPUT_BAND_FILENAME = {b: f"{b}.fits" for b in BAND_ORDER}

# ==========================================================================
# MEMBERS GROUP VOCABULARY (owner library spec item 3, 2026-10-08)
# ==========================================================================
#
# "mass" is DERIVED (yso_pms.PMSTrack, the same 1 Myr BHAC15+MIST track
# the PMS gate reads) -- every other quantity here is a native
# PARAM_UNION_COLUMNS column, read straight off parameters.fits. Av and
# inclination are the two scalar extras the spec names; the envelope and
# disc blocks are every PARAM_UNION_COLUMNS column under those two
# prefixes. A column absent from a given geometry's parameters.fits
# (e.g. a pure-envelope geometry's disk.* columns) is 0.0 for that row --
# "no such structure", the same convention `yso_fps.read_harmonized_
# params` used before this module superseded it.
MEMBER_PARAM_COLUMNS = (
    "Source Luminosity", "Av", "inclination",
    "envelope.rho_0", "envelope.rmin", "envelope.power", "envelope.rc",
    "disk.mass", "disk.rmax", "disk.beta", "disk.p", "disk.h100", "disk.rmin",
)

#: FITS column stems (<=8 chars where practical) for MIN_/MED_/MAX_ prefixing.
MEMBER_PARAM_FITS_NAMES = {
    "mass": "MASS", "Source Luminosity": "LSUN", "Av": "AV", "inclination": "INCL",
    "envelope.rho_0": "ENVRHO0", "envelope.rmin": "ENVRMIN",
    "envelope.power": "ENVPOW", "envelope.rc": "ENVRC",
    "disk.mass": "DISKMASS", "disk.rmax": "DISKRMAX", "disk.beta": "DISKBETA",
    "disk.p": "DISKP", "disk.h100": "DISKH100", "disk.rmin": "DISKRMIN",
}

MEMBER_QUANTITIES = ("mass",) + MEMBER_PARAM_COLUMNS


# ==========================================================================
# ROW-PROVENANCE TABLE (spec SS5)
# ==========================================================================

def represented_set_fractions(curation_state_path, verbose=True):
    """
    Per kept template (`fps.selected == 1`), its represented set's STAGE
    FRACTIONS (owner library spec item 3) -- the fraction of its
    Voronoi-assigned members (`fps.owner_model_id == this template's own
    model_id`, over every row the sampling stage determined, i.e.
    `fps.selected != FLAG_NA`) carrying each `labeling.label` value.
    This is what makes `subclass_prob` "a real distribution instead of
    today's delta function": a kept template's represented set can span
    more than one Stage/TD label once sampling is global rather than
    per-stratum.

    A kept template's SUBCLASS (and therefore its stratum folder) is the
    ARGMAX of its own represented set's fractions -- not necessarily its
    own native label -- since `model_io.write_classmap_fits` validates
    that SUBCLASS reproduces `subclass_prob`'s argmax exactly.

    Returns a DataFrame indexed by kept-template model_id: one
    `frac_<STRATUM_LABEL>` column per label (sums to 1 per row), `n_members`,
    and `majority_label` (STRATUM_LABELS value, the argmax).
    """
    fps_ds = reg.read_dataset(curation_state_path, "fps",
                              columns=["selected", "owner_model_id"])
    labeling = reg.read_dataset(curation_state_path, "labeling", columns=["label"])

    determined = fps_ds[fps_ds["selected"] != reg.FLAG_NA].copy()
    determined = determined.merge(labeling, on="model_id", how="left")

    counts = determined.pivot_table(index="owner_model_id", columns="label",
                                    values="model_id", aggfunc="count", fill_value=0)
    for k in STRATUM_LABELS:
        if k not in counts.columns:
            counts[k] = 0
    counts = counts[list(STRATUM_LABELS)]
    n_members = counts.sum(axis=1)
    fractions = counts.div(n_members, axis=0)
    fractions.columns = [f"frac_{k}" for k in STRATUM_LABELS]
    fractions["n_members"] = n_members.astype(np.int64)
    fractions["majority_label"] = counts.idxmax(axis=1)
    fractions.index.name = "model_id"

    if verbose:
        n_mixed = int((fractions[[f"frac_{k}" for k in STRATUM_LABELS]].max(axis=1) < 1.0).sum())
        print(f"[assembly] represented-set fractions: {len(fractions)} kept templates, "
              f"{n_mixed} with a represented set spanning more than one label", flush=True)
    return fractions


def build_row_provenance_table(curation_state_path, verbose=True):
    """
    One flat table driving the entire extraction/write process (spec SS5):
    every `fps.selected == 1` model_id (a KEPT TEMPLATE -- the owner
    library spec's global r-net, not a per-stratum budget), decoded to
    its source `(geometry, geometry_row_index)`, mapped to its
    destination `stratum_folder` by its REPRESENTED SET's majority label
    (`represented_set_fractions`), with a precomputed `output_row_index`
    -- this row's position within its destination stratum's final
    output files.

    `output_row_index` is assigned by sorting on
    `(stratum_folder, geometry, geometry_row_index)` and taking a
    cumulative count within each `stratum_folder` group (SS5.2) --
    computed here, once, before any file I/O.

    Returns a DataFrame: model_id, geometry, geometry_row_index, label
    (the kept template's OWN native label, point-value provenance only),
    stratum_folder (= majority_label's folder), output_row_index,
    frac_<STRATUM_LABEL> x5, n_members.

    The `frac_*` columns ride along so `classmap.fits` can be emitted
    from the same in-memory arrays as the rest of the run rather than by
    re-reading the curation state afterwards -- one read, no opportunity
    for the label file and the library to disagree.
    """
    labeling = reg.read_dataset(curation_state_path, "labeling", columns=["label"])
    fps = reg.read_dataset(curation_state_path, "fps", columns=["selected"])
    fractions = represented_set_fractions(curation_state_path, verbose=verbose)

    df = fps.merge(labeling, on="model_id", how="left")
    df = df[df["selected"] == 1].copy()
    df = df.merge(fractions, left_on="model_id", right_index=True, how="left")

    parts = df["model_id"].str.split(":", n=1, expand=True)
    df["geometry"] = parts[0]
    df["geometry_row_index"] = parts[1].astype(int)
    df["stratum_folder"] = df["majority_label"].astype(str).map(STRATUM_OUTPUT_FOLDER)

    if df["stratum_folder"].isna().any():
        bad = sorted(df.loc[df["stratum_folder"].isna(), "majority_label"].unique())
        raise ValueError(f"majority_label value(s) {bad} have no STRATUM_OUTPUT_FOLDER "
                         "mapping -- every fps.selected row must have a valid represented-set "
                         "majority label.")

    df = df.sort_values(["stratum_folder", "geometry", "geometry_row_index"]).reset_index(drop=True)
    df["output_row_index"] = df.groupby("stratum_folder").cumcount()

    frac_columns = [f"frac_{k}" for k in STRATUM_LABELS]
    df = df[["model_id", "geometry", "geometry_row_index", "label", "majority_label",
            "stratum_folder", "output_row_index", "n_members"] + frac_columns]

    if verbose:
        print(f"[assembly] row-provenance table: {len(df)} kept templates", flush=True)
        print(f"[assembly] per-stratum counts: {df.groupby('stratum_folder').size().to_dict()}",
             flush=True)
        print(f"[assembly] contributing geometries: {df['geometry'].nunique()}", flush=True)
    return df


def stratum_n_k(table):
    """{stratum_folder: N_k} from the row-provenance table -- the exact
    final row count each stratum's output files are pre-allocated to."""
    return table.groupby("stratum_folder").size().to_dict()


def reference_geometry_dir(table, yso_root):
    """The one geometry universal constants (APERTURES/SPECTRAL_INFO/
    DISTANCE/FILTWAV) are copied from -- any contributing geometry works
    (spec SS2.4), so this is just `table`'s first row's geometry, named
    once here rather than as two independently-typed-out `table
    ["geometry"].iloc[0]` expressions (`build_and_write_assembly` and
    `qc_report` each rebuild their own `table` -- this does not couple
    them, it just makes the shared "which geometry" logic a single
    definition instead of a coincidence)."""
    return os.path.join(yso_root, table["geometry"].iloc[0])


# ==========================================================================
# SOURCE-SIDE EXTRACTION (one call per geometry, spec SS6.1)
# ==========================================================================

def _extract_info_rows(info_path, row_index_sorted):
    """`info.fits` row subset, by row position (spec SS6.2).

    The join key is READ as R17's own `Model Name` -- that is the upstream
    file's column and not ours to rename -- and returned as `MODEL_NAME`, the
    spelling every other file in every library uses and the one the assembled
    output writes. The rename happens here, once, at the boundary.
    """
    with fits.open(info_path, memmap=True) as h:
        data = h[1].data
        return {
            "MODEL_NAME": np.asarray(data["Model Name"])[row_index_sorted].astype(str),
            "Above MS": np.asarray(data["Above MS"], dtype=np.int64)[row_index_sorted],
            "Stage": np.asarray(data["Stage"], dtype=np.int64)[row_index_sorted],
            "Class": np.asarray(data["Class"], dtype=np.int64)[row_index_sorted],
        }


def _extract_parameters_rows(params_path, row_index_sorted):
    """
    `parameters.fits` row subset, by row position (spec SS6.2), for
    whichever of `PARAM_UNION_COLUMNS` this geometry's file actually has
    (fancy-indexing on axis 0 handles both scalar `D` and aperture-array
    `20D` columns identically). Also returns MODEL_NAME for this row
    subset -- the `ref_names` the convolved-band reads (SS6.1) align to.

    **Release quirk, confirmed across all 18 geometries (a full shape
    audit, not just the columns that happened to error first):** five
    union columns carry an extra spurious size-1 axis relative to their
    declared FITS format -- `Inner Radius`/`Outer Radius`/
    `Disk Minimum Q` (format `D`, but `(N, 1)` not `(N,)`) and
    `Sphere Masses`/`Sphere Mass-Weighted Temperatures` (format `20D`,
    but `(N, 1, 20)` not `(N, 20)`) -- a TDIM artifact of how the release
    wrote these five specific columns; every other scalar/aperture-array
    column round-trips at its declared shape with no anomaly. Confirmed
    geometry-independent: every geometry that has any of these five shows
    the same extra axis, every geometry that has any other column does
    not. Normalized here via an unconditional `reshape` to the KNOWN
    target shape (`(n,)` scalar / `(n, N_APERTURES)` aperture-array)
    whenever the raw shape doesn't already match -- robust regardless of
    exactly where the spurious axis sits, since a spurious size-1 axis
    never changes the total element count a reshape depends on. Every
    non-aperture-array column ends up one consistent shape regardless of
    which geometry it came from -- otherwise the skeleton write (SS6.3,
    plain 1D/2D columns) fails with a broadcast error.

    Returns (model_names, {column: array}) -- only present columns are
    keyed; absent columns are the caller's responsibility to leave as the
    output skeleton's NaN placeholder (spec SS4).
    """
    n = len(row_index_sorted)
    with fits.open(params_path, memmap=True) as h:
        data = h[1].data
        cols = data.columns.names
        model_names = np.asarray(data["MODEL_NAME"])[row_index_sorted].astype(str)
        out = {}
        for col in PARAM_UNION_COLUMNS:
            if col not in cols:
                continue
            arr = np.asarray(data[col], dtype=np.float64)[row_index_sorted]
            target_shape = (n, N_APERTURES) if col in PARAM_APERTURE_ARRAY_COLUMNS else (n,)
            if arr.shape != target_shape:
                arr = arr.reshape(target_shape)
            out[col] = arr
    return model_names, out


def _extract_values_cube_rows(path, row_index_sorted, n_ap, want_uncertainties):
    """
    Row-subset reader for the shared MODEL_NAMES/SPECTRAL_INFO/VALUES(
    /UNCERTAINTIES) convention `flux.fits` and `stellar.fits` both use
    (spec SS2.4) -- one function for both, since `stellar.fits` is a
    strict subset of `flux.fits`'s structure (no APERTURES, no
    UNCERTAINTIES, a singleton aperture axis instead of 20).

    Reuses `model_io.axes_to_models_ap_wav`'s axis-detection exactly as
    `model_io.FluxReader.model_set` does, rather than re-deriving it --
    `model_io.read_flux_metadata` itself isn't used here because it hard-
    requires an `APERTURES` HDU, which `stellar.fits` doesn't have.

    Returns (model_names, values, uncertainties_or_None), `values`/
    `uncertainties` shaped `(len(rows), n_ap, N_WAVELENGTHS)`.
    """
    with fits.open(path, memmap=True) as h:
        names_full = np.asarray(h["MODEL_NAMES"].data["MODEL_NAME"]).astype(str)
        n_models = names_full.size
        vshape = tuple(h["VALUES"].shape)
        order = axes_to_models_ap_wav(vshape, n_models, n_ap, N_WAVELENGTHS)
        model_axis = order.index(0) if 0 in order else 0

        def _read(hdu_name):
            raw = h[hdu_name].data
            sl = [slice(None)] * raw.ndim
            sl[model_axis] = row_index_sorted
            block = np.asarray(raw[tuple(sl)], dtype=np.float32)
            return np.transpose(block, order)

        names = names_full[row_index_sorted]
        values = _read("VALUES")
        uncertainties = _read("UNCERTAINTIES") if want_uncertainties else None
    return names, values, uncertainties


def _extract_convolved_band_rows(geometry_dir, band, ref_names):
    """
    TOTAL_FLUX + TOTAL_FLUX_ERR for one band, aligned to `ref_names` BY
    MODEL_NAME (spec SS6.2) -- convolved files can carry their own
    internal row order, the same reason `yso_labeling._read_band_flux`
    already aligns by name rather than position. Mirrors `_read_band_flux`
    exactly, plus TOTAL_FLUX_ERR
    (which that function doesn't need for its own purpose).
    """
    path = os.path.join(geometry_dir, "convolved", _CONVOLVED_SUBDIR[band],
                        _CONVOLVED_FILENAME[band])
    with fits.open(path, memmap=True) as h:
        names = np.asarray(h[1].data["MODEL_NAME"]).astype(str)
        flux = np.asarray(h[1].data["TOTAL_FLUX"], dtype=np.float64)
        flux_err = np.asarray(h[1].data["TOTAL_FLUX_ERR"], dtype=np.float64)
    order = align_by_name(ref_names, names)
    return flux[order], flux_err[order]


def _read_reference_constants(reference_dir):
    """
    The universal, geometry-independent constants confirmed in spec
    SS2.4 -- read ONCE from any single geometry (which one doesn't
    matter) and copied verbatim into every stage's output: APERTURES,
    SPECTRAL_INFO (both HDUs' full column data), and DISTANCE.

    FILTWAV is the one exception: it is NOT copied from the release. It
    sets the wavelength at which the fitter evaluates the extinction law
    for that band (`Models._read_version_*` -> `m.wavelengths` ->
    `av_law`), so it must be identical for a given band across all five
    libraries or a band's A_lambda would differ by library and bias every
    cross-library comparison. The project-wide value is
    `BANDS[band].wvl_effective_um`; the release's own values diverge from
    it by up to 1.5% in wavelength, which is 11% in k at I4 (the band
    sits on the rising edge of the 9.7 um silicate feature).

    Stamping rather than reconvolving is exact, not an approximation: our
    convolution of the release's flux.fits reproduces the released
    convolved fluxes BITWISE on every defined cell, so the transmission
    curves already agree and FILTWAV is the only real difference.
    """
    with fits.open(os.path.join(reference_dir, "flux.fits"), memmap=True) as h:
        aperture_au = np.asarray(h["APERTURES"].data["APERTURE"], dtype=np.float64)
        wavelength = np.asarray(h["SPECTRAL_INFO"].data["WAVELENGTH"], dtype=np.float64)
        frequency = np.asarray(h["SPECTRAL_INFO"].data["FREQUENCY"], dtype=np.float64)
        distance_cm = float(h["PRIMARY"].header["DISTANCE"])

    filtwav = {band: float(BANDS[band].wvl_effective_um) for band in BAND_ORDER}

    return {
        "aperture_au": aperture_au, "wavelength": wavelength, "frequency": frequency,
        "distance_cm": distance_cm, "filtwav": filtwav,
    }


# ==========================================================================
# DESTINATION-SIDE SKELETONS (pre-allocated at final N_k, spec SS6.3)
# ==========================================================================

def _create_info_skeleton(path, n_k):
    cols = [
        fits.Column(name="MODEL_NAME", format=f"{MODEL_NAME_WIDTH}A",
                   array=np.full(n_k, "", dtype=f"U{MODEL_NAME_WIDTH}")),
        fits.Column(name="Above MS", format="K", array=np.zeros(n_k, dtype=np.int64)),
        fits.Column(name="Stage", format="K", array=np.full(n_k, -1, dtype=np.int64)),
        fits.Column(name="Class", format="K", array=np.full(n_k, -1, dtype=np.int64)),
        fits.Column(name="geometry", format="20A", array=np.full(n_k, "", dtype="U20")),
        fits.Column(name="geometry_row_index", format="K", array=np.full(n_k, -1, dtype=np.int64)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="INFO")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _create_parameters_skeleton(path, n_k):
    cols = [fits.Column(name="MODEL_NAME", format=f"{MODEL_NAME_WIDTH}A",
                        array=np.full(n_k, "", dtype=f"U{MODEL_NAME_WIDTH}"))]
    for col in PARAM_UNION_COLUMNS:
        if col in PARAM_APERTURE_ARRAY_COLUMNS:
            cols.append(fits.Column(name=col, format=f"{N_APERTURES}D",
                                    array=np.full((n_k, N_APERTURES), np.nan)))
        else:
            cols.append(fits.Column(name=col, format="D", array=np.full(n_k, np.nan)))
    hdu = fits.BinTableHDU.from_columns(cols, name="PARAMETERS")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _create_flux_skeleton(path, n_k, ref, apertures_au):
    """One flux-cube skeleton via the shared writer (model_io.write_flux_cube)
    -- units, DISTANCE, and the all-ones PRIMARY validity mask are the
    writer's own responsibility, not re-implemented here. `apertures_au`
    is `ref["aperture_au"]` for flux.fits, `None` for stellar.fits (the
    writer's own point for "omit the APERTURES HDU / no aperture axis") --
    the two files are one writer, one shape argument, not two forks.

    Model axis FIRST -- confirmed the native on-disk layout for both
    flux.fits and stellar.fits is (n_models, n_ap, n_wav) directly (not
    the reversed (n_wav, n_ap, n_models) order model_io.py's generic
    axes_to_models_ap_wav also anticipates for other releases), verified
    across s---s-i/spubhmi/s-p-hmi.
    """
    n_ap = N_APERTURES if apertures_au is not None else 1
    values = np.zeros((n_k, n_ap, N_WAVELENGTHS), dtype=np.float32)
    uncertainties = np.zeros((n_k, n_ap, N_WAVELENGTHS), dtype=np.float32) if apertures_au is not None else None
    write_flux_cube(
        path, names=np.full(n_k, "", dtype=f"U{MODEL_NAME_WIDTH}"),
        wave_um_desc=ref["wavelength"], freq_hz_desc=ref["frequency"],
        values=values, distance_cm=ref["distance_cm"], apertures_au=apertures_au,
        uncertainties=uncertainties, name_format=f"{MODEL_NAME_WIDTH}A")


def _create_band_skeleton(path, n_k, band, ref):
    primary = fits.PrimaryHDU()
    primary.header["FILTWAV"] = ref["filtwav"][band]
    primary.header["NMODELS"] = n_k
    primary.header["NAP"] = N_APERTURES

    flux_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="MODEL_NAME", format=f"{CONVOLVED_MODEL_NAME_WIDTH}A",
                   array=np.full(n_k, "", dtype=f"U{CONVOLVED_MODEL_NAME_WIDTH}")),
        fits.Column(name="TOTAL_FLUX", format=f"{N_APERTURES}D", unit="mJy",
                   array=np.full((n_k, N_APERTURES), np.nan)),
        fits.Column(name="TOTAL_FLUX_ERR", format=f"{N_APERTURES}D", unit="mJy",
                   array=np.full((n_k, N_APERTURES), np.nan)),
    ], name="CONVOLVED FLUXES")
    ap_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="APERTURE", format="D", unit="AU", array=ref["aperture_au"])],
        name="APERTURES")
    fits.HDUList([primary, flux_hdu, ap_hdu]).writeto(path, overwrite=True)


def _write_models_conf(out_path, stratum_folder):
    """Write `models.conf` via the shared writer (model_io.write_models_conf)
    rather than copying the source geometry's file verbatim (spec SS2.1).
    The copied file inherited `length_subdir = 2` from the native release,
    while no stratum ships a `seds/` directory -- the shared writer's own
    default (`length_subdir=0`) is the one that matches what's actually on
    disk. `aperture_dependent=True` since every stratum's flux.fits carries
    a real (20-element) aperture axis."""
    write_models_conf(out_path, name=stratum_folder, aperture_dependent=True)


def _stratum_file_paths(base_dir, stratum_folder):
    """{file_key: path} for one stratum folder under `base_dir` -- the one
    place that names this layout, so the skeleton writer and the
    post-move path remap (`build_and_write_assembly`) cannot disagree
    about it."""
    stratum_dir = os.path.join(base_dir, stratum_folder)
    conv_dir = os.path.join(stratum_dir, "convolved")
    return {
        "info": os.path.join(stratum_dir, "info.fits"),
        "parameters": os.path.join(stratum_dir, "parameters.fits"),
        "flux": os.path.join(stratum_dir, "flux.fits"),
        "stellar": os.path.join(stratum_dir, "stellar.fits"),
        "classmap": os.path.join(stratum_dir, "classmap.fits"),
        "members": os.path.join(stratum_dir, "members.fits"),
        "bands": {b: os.path.join(conv_dir, OUTPUT_BAND_FILENAME[b]) for b in BAND_ORDER},
    }


def create_output_skeletons(out_dir, n_k_by_stratum, reference_dir, verbose=True):
    """
    Pre-create all 5 stratum output folders at their true final row count
    (spec SS6.3) -- must happen before the geometry sweep starts. Returns
    {stratum_folder: {file_key: path}} for `sweep_and_write` to open.
    """
    ref = _read_reference_constants(reference_dir)
    paths = {}
    for stratum_folder in STRATUM_OUTPUT_FOLDER.values():
        n_k = n_k_by_stratum.get(stratum_folder, 0)
        stratum_dir = os.path.join(out_dir, stratum_folder)
        stratum_paths = _stratum_file_paths(out_dir, stratum_folder)
        os.makedirs(os.path.dirname(stratum_paths["bands"][BAND_ORDER[0]]), exist_ok=True)

        _write_models_conf(os.path.join(stratum_dir, "models.conf"), stratum_folder)
        _create_info_skeleton(stratum_paths["info"], n_k)
        _create_parameters_skeleton(stratum_paths["parameters"], n_k)
        _create_flux_skeleton(stratum_paths["flux"], n_k, ref, apertures_au=ref["aperture_au"])
        _create_flux_skeleton(stratum_paths["stellar"], n_k, ref, apertures_au=None)
        for band in BAND_ORDER:
            _create_band_skeleton(stratum_paths["bands"][band], n_k, band, ref)

        paths[stratum_folder] = stratum_paths
        if verbose:
            print(f"[assembly] created skeleton {stratum_folder}/ (N_k={n_k})", flush=True)
    return paths


# ==========================================================================
# GEOMETRY SWEEP: read once per geometry, scatter-write to every stage it feeds
# ==========================================================================

def _assert_contiguous(output_rows):
    """Sanity check (spec SS5.2): one geometry's contribution to one stage
    must be a single contiguous output-row block, by construction of how
    output_row_index was assigned."""
    if output_rows.size == 0:
        return
    lo, hi = int(output_rows.min()), int(output_rows.max())
    if not np.array_equal(np.sort(output_rows), np.arange(lo, hi + 1)):
        raise ValueError("output_row_index is not contiguous for this geometry/stage "
                         "block -- build_row_provenance_table's offset assignment is broken.")


def sweep_and_write(table, yso_root, stratum_handles, verbose=True):
    """
    The single geometry sweep (spec SS6.1/SS6.3): for each geometry in
    `table`, open its native files once, extract the union of rows needed
    across every stratum it feeds, and scatter-write into each stratum's
    already-open, pre-allocated output files at the precomputed
    `output_row_index` slice. `stratum_handles` is
    `{stratum_folder: {file_key: open HDUList in mode='update'}}` (§ below).
    """
    n_geoms = table["geometry"].nunique()
    for gi, (geometry, geom_table) in enumerate(table.groupby("geometry", sort=True)):
        geom_table = geom_table.sort_values("geometry_row_index").reset_index(drop=True)
        row_index_sorted = geom_table["geometry_row_index"].to_numpy()
        geometry_dir = os.path.join(yso_root, geometry)

        info_block = _extract_info_rows(os.path.join(geometry_dir, "info.fits"), row_index_sorted)
        model_names, param_block = _extract_parameters_rows(
            os.path.join(geometry_dir, "parameters.fits"), row_index_sorted)
        flux_names, flux_values, flux_unc = _extract_values_cube_rows(
            os.path.join(geometry_dir, "flux.fits"), row_index_sorted, N_APERTURES, True)
        stellar_names, stellar_values, _ = _extract_values_cube_rows(
            os.path.join(geometry_dir, "stellar.fits"), row_index_sorted, 1, False)

        band_blocks = {band: _extract_convolved_band_rows(geometry_dir, band, model_names)
                      for band in BAND_ORDER}

        for stratum_folder, sub in geom_table.groupby("stratum_folder"):
            local_pos = sub.index.to_numpy()
            output_rows = sub["output_row_index"].to_numpy()
            _assert_contiguous(output_rows)
            lo, hi = int(output_rows.min()), int(output_rows.max()) + 1
            handles = stratum_handles[stratum_folder]

            info_hdu = handles["info"][1].data
            info_hdu["MODEL_NAME"][lo:hi] = info_block["MODEL_NAME"][local_pos]
            info_hdu["Above MS"][lo:hi] = info_block["Above MS"][local_pos]
            info_hdu["Stage"][lo:hi] = info_block["Stage"][local_pos]
            info_hdu["Class"][lo:hi] = info_block["Class"][local_pos]
            info_hdu["geometry"][lo:hi] = geometry
            info_hdu["geometry_row_index"][lo:hi] = row_index_sorted[local_pos]

            param_hdu = handles["parameters"][1].data
            param_hdu["MODEL_NAME"][lo:hi] = model_names[local_pos]
            for col, arr in param_block.items():
                param_hdu[col][lo:hi] = arr[local_pos]

            # flux_values/uncertainties/stellar_values are already
            # (n_rows, n_ap, n_wav) (_extract_values_cube_rows), matching
            # the skeleton's model-axis-first layout directly -- no
            # transpose needed (confirmed empirically: the native on-disk
            # order is (n_models, n_ap, n_wav), not the reversed order
            # model_io.py's axis-detection also anticipates for other
            # releases).
            flux_hdu = handles["flux"]
            flux_hdu["MODEL_NAMES"].data["MODEL_NAME"][lo:hi] = flux_names[local_pos]
            flux_hdu["VALUES"].data[lo:hi, :, :] = flux_values[local_pos]
            flux_hdu["UNCERTAINTIES"].data[lo:hi, :, :] = flux_unc[local_pos]

            stellar_hdu = handles["stellar"]
            stellar_hdu["MODEL_NAMES"].data["MODEL_NAME"][lo:hi] = stellar_names[local_pos]
            stellar_hdu["VALUES"].data[lo:hi, :, :] = stellar_values[local_pos]

            for band in BAND_ORDER:
                flux_b, err_b = band_blocks[band]
                band_hdu = handles["bands"][band][1].data
                band_hdu["MODEL_NAME"][lo:hi] = model_names[local_pos]
                band_hdu["TOTAL_FLUX"][lo:hi] = flux_b[local_pos]
                band_hdu["TOTAL_FLUX_ERR"][lo:hi] = err_b[local_pos]

        if verbose:
            print(f"[assembly] geometry {geometry} ({gi + 1}/{n_geoms}): "
                 f"{len(geom_table)} rows written across "
                 f"{geom_table['stratum_folder'].nunique()} stratum(strata)", flush=True)


def _open_stratum_handles(stratum_paths):
    """Open every file of every stratum in mode='update' (memmapped) -- kept
    open for the whole geometry sweep (spec SS6.3)."""
    handles = {}
    for stratum_folder, paths in stratum_paths.items():
        handles[stratum_folder] = {
            "info": fits.open(paths["info"], mode="update", memmap=True),
            "parameters": fits.open(paths["parameters"], mode="update", memmap=True),
            "flux": fits.open(paths["flux"], mode="update", memmap=True),
            "stellar": fits.open(paths["stellar"], mode="update", memmap=True),
            "bands": {b: fits.open(paths["bands"][b], mode="update", memmap=True)
                     for b in BAND_ORDER},
        }
    return handles


def _close_stratum_handles(handles):
    for stratum_folder, files in handles.items():
        for key, hdul in files.items():
            if key == "bands":
                for band_hdul in hdul.values():
                    band_hdul.flush()
                    band_hdul.close()
            else:
                hdul.flush()
                hdul.close()


# ==========================================================================
# TOP-LEVEL ORCHESTRATION
# ==========================================================================

def write_stratum_classmaps(out_dir, table, verbose=True):
    """`classmap.fits` for each stratum folder -- the authoritative
    per-model class/subclass label, sibling to flux.fits.

    One file per stratum rather than one for the library, because each
    stratum folder is a drop-in substitute for a native geometry folder and
    has to be self-describing on its own. A consequence is that SUBCLASS is
    constant within a file and SUBCLASS_LEGEND therefore has a single row --
    correct under the shared writer's rule that the subclass legend
    describes THIS file while the class legend is the cross-library schema.

    SUBCLASS_PROB is the reason this is worth shipping at all, now a REAL
    distribution rather than a delta function (owner library spec item
    3): `frac_<STRATUM_LABEL>` is the kept template's REPRESENTED SET's
    own label fractions (`represented_set_fractions`), not a per-model
    KNN-soft label -- a kept template whose r-net cell straddles two
    Stage/TD labels carries a genuinely mixed distribution here. SUBCLASS
    is that distribution's argmax (`majority_label`), which is why a kept
    template can land in a stratum folder other than its own native
    label. The distribution lives nowhere else a library consumer can
    reach -- the curation state is a working artefact, not a product.

    Labels and fractions come from the in-memory row-provenance table;
    MODEL_NAME comes from the info.fits this run just wrote, and
    `model_dir` makes the shared writer assert it against flux.fits in
    order. A disagreement anywhere raises rather than shipping.
    """
    frac_columns = [f"frac_{k}" for k in STRATUM_LABELS]
    written = {}
    for stratum_folder, sub in table.groupby("stratum_folder"):
        sub = sub.sort_values("output_row_index")
        stratum_dir = os.path.join(out_dir, stratum_folder)

        with fits.open(os.path.join(stratum_dir, "info.fits"), memmap=False) as h:
            names = np.asarray(h[1].data["MODEL_NAME"]).astype(str)

        majority = sub["majority_label"].to_numpy().astype(str)
        unknown = sorted(set(majority) - set(SUBCLASS_BY_LABEL))
        if unknown:
            raise ValueError(f"majority_label(s) {unknown} have no SUBCLASS_BY_LABEL mapping")
        subclass = np.array([SUBCLASS_BY_LABEL[v] for v in majority])

        written[stratum_folder] = write_classmap_fits(
            os.path.join(stratum_dir, "classmap.fits"),
            names=names,
            class_id=MODEL_CLASS,
            subclass=subclass,
            class_legend=CLASS_LEGEND,
            subclass_legend=SUBCLASS_LEGEND,
            subclass_prob=sub[frac_columns].to_numpy(dtype=float),
            provenance=(
                ("CLASS_SOURCE", "library declaration; every model here is a YSO"),
                ("SUBCLASS_SOURCE", "argmax of the represented set's own label fractions "
                                    "(owner library spec item 3, 2026-10-08)"),
                ("SUBCLASS_REF", "Richardson 2024 Stage; Gutermuth 2009 24um carve"),
                ("SUBCLASS_NOTE", "stored argmax is authoritative; ties broken by order"),
                ("LEGEND_SOURCE", "codes and descriptions authored here"),
                ("PROB_HDU", "SUBCLASS_PROB holds the represented set's real label "
                             "fractions, not a per-model soft label"),
            ),
            model_dir=stratum_dir,
            name_format=f"{MODEL_NAME_WIDTH}A",
        )
        if verbose:
            print(f"[assembly] classmap {stratum_folder}/ ({len(names)} models, "
                  f"SUBCLASS={subclass[0]})", flush=True)
    return written


# ==========================================================================
# MEMBERS GROUP (owner library spec item 3): per-kept-template represented-
# set parameter ranges (min/median/max) + count, over the full GATED
# (eligibility.eligible == 1) population, keyed by `fps.owner_model_id`
# (the Voronoi assignment `yso_fps.run_global_sampling` already computed --
# derived FROM the kept templates, never re-derived here).
# ==========================================================================

def compute_member_parameter_ranges(curation_state_path, yso_root, data_root, verbose=True):
    """Stream every geometry's contribution to the gated pool ONCE
    (mirrors `yso_fps.read_harmonized_params`'s streaming pattern) and
    reduce to per-kept-template min/median/max over `MEMBER_QUANTITIES`,
    plus the member-to-template distance (already stored in `fps.
    dist_sigeff` -- not recomputed).

    `mass` is derived via `yso_pms.PMSTrack` (the same 1 Myr BHAC15+MIST
    track the PMS gate reads) from each row's own `Source Luminosity` --
    not a native grid column.

    Returns a DataFrame indexed by kept-template model_id (same index
    space as `represented_set_fractions`'s return).
    """
    fps_ds = reg.read_dataset(curation_state_path, "fps",
                              columns=["selected", "owner_model_id", "dist_sigeff"])
    ingest = reg.read_dataset(curation_state_path, "ingest", columns=["geometry_folder"])

    determined = fps_ds[fps_ds["selected"] != reg.FLAG_NA].copy()
    determined = determined.merge(ingest[["model_id", "geometry_folder"]],
                                  on="model_id", how="left")

    track = pms.PMSTrack(data_root)

    frames = []
    for geom, sub in determined.groupby("geometry_folder", sort=False):
        row_index = sub["model_id"].str.slice(len(geom) + 1).astype(np.int64).to_numpy()
        params_path = os.path.join(yso_root, geom, "parameters.fits")
        _names, cols = _extract_parameters_rows(params_path, row_index)

        lum = cols.get("Source Luminosity", np.zeros(len(sub)))
        with np.errstate(divide="ignore", invalid="ignore"):
            finite_pos = np.isfinite(lum) & (lum > 0)
            log10_l = np.where(finite_pos, lum, 1.0)
            log10_l = np.log10(log10_l)
        mass = track.mass_from_log10_l(log10_l)
        mass = np.where(finite_pos, mass, np.nan)

        block = {
            "model_id": sub["model_id"].to_numpy(),
            "owner_model_id": sub["owner_model_id"].to_numpy(),
            "dist_sigeff": sub["dist_sigeff"].to_numpy(dtype=float),
            "mass": mass,
        }
        for col in MEMBER_PARAM_COLUMNS:
            block[col] = cols.get(col, np.zeros(len(sub)))
        frames.append(pd.DataFrame(block))
        if verbose:
            print(f"[assembly] members params streamed {geom}: {len(sub)} rows", flush=True)

    long_df = pd.concat(frames, axis=0, ignore_index=True)
    grouped = long_df.groupby("owner_model_id")

    stats = {}
    for q in MEMBER_QUANTITIES:
        g = grouped[q]
        stats[f"min_{q}"] = g.min()
        stats[f"median_{q}"] = g.median()
        stats[f"max_{q}"] = g.max()
    out = pd.DataFrame(stats)
    out["dist_sigeff_median"] = grouped["dist_sigeff"].median()
    out["dist_sigeff_max"] = grouped["dist_sigeff"].max()
    out["n_members"] = grouped.size().astype(np.int64)
    out.index.name = "model_id"

    if verbose:
        print(f"[assembly] member parameter ranges: {len(out)} kept templates, "
              f"{len(long_df)} total represented-set rows", flush=True)
    return out


def write_stratum_members_fits(out_dir, table, member_stats, *, sigma_eff, n_eligible,
                               algorithm_note="", verbose=True):
    """`members.fits` for each stratum folder: one row per KEPT TEMPLATE
    in that folder, carrying its represented set's label fractions
    (already in `table`), its `MEMBER_QUANTITIES` min/median/max
    (`member_stats`), its member count, and its member-to-template
    distance (median/max, SIGEFF units). Point values stay in
    parameters.fits -- this is schema parity with every other library's
    `models` group (owner library spec item 3's last sentence).
    """
    written = {}
    for stratum_folder, sub in table.groupby("stratum_folder"):
        sub = sub.sort_values("output_row_index")
        stratum_dir = os.path.join(out_dir, stratum_folder)

        with fits.open(os.path.join(stratum_dir, "info.fits"), memmap=False) as h:
            names = np.asarray(h[1].data["MODEL_NAME"]).astype(str)

        stats_sub = member_stats.loc[sub["model_id"].to_numpy()]

        cols = [
            fits.Column(name="MODEL_NAME", format=f"{MODEL_NAME_WIDTH}A", array=names),
            fits.Column(name="N_MEMBERS", format="K",
                       array=stats_sub["n_members"].to_numpy(dtype=np.int64)),
        ]
        for k in STRATUM_LABELS:
            cols.append(fits.Column(name=f"FRAC_{SUBCLASS_BY_LABEL[k]}", format="D",
                                    array=sub[f"frac_{k}"].to_numpy(dtype=float)))
        for q in MEMBER_QUANTITIES:
            stem = MEMBER_PARAM_FITS_NAMES.get(q, q)
            for stat in ("min", "median", "max"):
                cols.append(fits.Column(name=f"{stat.upper()[:3]}_{stem}", format="D",
                                        array=stats_sub[f"{stat}_{q}"].to_numpy(dtype=float)))
        cols.append(fits.Column(name="DIST_MED_SIGEFF", format="D",
                                array=stats_sub["dist_sigeff_median"].to_numpy(dtype=float)))
        cols.append(fits.Column(name="DIST_MAX_SIGEFF", format="D",
                                array=stats_sub["dist_sigeff_max"].to_numpy(dtype=float)))

        members_table = fits.BinTableHDU.from_columns(cols, name="MEMBERS")
        members_table.header["COMMENT"] = (
            "one row per KEPT TEMPLATE; fractions/ranges are over its represented "
            "set (Voronoi-assigned), not the template's own point value (see "
            "parameters.fits for that)")

        primary = fits.PrimaryHDU()
        primary.header["SIGEFF"] = (float(sigma_eff), "noise length [dex], the sampling radius")
        primary.header["NELIG"] = (int(n_eligible), "eligible (gated) models, whole library")
        primary.header["NTEMPL"] = (len(names), "kept templates in this stratum folder")
        primary.header["ALGO"] = "greedy maximum-uncovered-neighbour r-net, deterministic, no seed"
        if algorithm_note:
            primary.header["COMMENT"] = algorithm_note[:70]
        primary.header["COMMENT"] = (
            "covering: every gated model is within SIGEFF of its owner_model_id "
            "representative, verified in /metadata/fps")

        path = os.path.join(stratum_dir, "members.fits")
        fits.HDUList([primary, members_table]).writeto(path, overwrite=True)
        written[stratum_folder] = path
        if verbose:
            print(f"[assembly] members {stratum_folder}/ ({len(names)} templates)", flush=True)
    return written


#: Staging directories live under the user's home with this prefix. The
#: `try/finally` in `build_and_write_assembly` removes the one a run owns
#: on every CATCHABLE exit -- success, exception, KeyboardInterrupt -- but
#: SIGKILL is not catchable, so a hard kill strands ~8 GB there. That is
#: not hypothetical: it happened three times while this change was being
#: developed. `_sweep_stale_staging` is the other half of the guarantee.
_STAGING_PREFIX = ".sesna_yso_assembly_"

#: A staging directory younger than this is assumed to belong to a run
#: that is still going and is left alone. Concurrent assembly runs are not
#: expected, but deleting a live run's output would be far worse than
#: leaving a dead run's for one more hour.
_STAGING_STALE_AFTER_S = 3600.0


def _sweep_stale_staging(verbose=True):
    """Delete staging directories stranded by a previously killed run.

    Returns the paths removed. Safe by construction: the prefix makes
    these unambiguously ours, and a staging directory that outlived its
    run holds nothing a caller can use -- the move into place is the last
    thing a successful run does.
    """
    now = time.time()
    removed = []
    for path in sorted(glob.glob(os.path.join(os.path.expanduser("~"),
                                              _STAGING_PREFIX + "*"))):
        if not os.path.isdir(path):
            continue
        try:
            if now - os.stat(path).st_mtime < _STAGING_STALE_AFTER_S:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    if removed and verbose:
        print(f"[assembly] swept {len(removed)} stale staging director"
              f"{'y' if len(removed) == 1 else 'ies'} from a previously "
              f"killed run: {', '.join(os.path.basename(p) for p in removed)}",
              flush=True)
    return removed


def _move_stratum_into_place(src_dir, dst_dir, verbose=True):
    """Move one finished stratum directory from staging into its final
    location.

    Existing-destination policy (a rebuild overwrites a shipped stratum
    directory, and an interrupted move must never leave a half-old/
    half-new mix inside it): rename the current `dst_dir` aside to a
    throwaway sibling name, rename the fully-written staging directory
    into `dst_dir`, then delete the old one. Each of those two renames
    is a single atomic directory-rename syscall on one volume (`dst_dir`
    always is; `src_dir`/staging is too, per the same-volume check in
    `build_and_write_assembly`, or `shutil.move` degrades to copy+delete
    if not) -- so at every instant `dst_dir` either does not exist yet,
    holds the complete old directory, or holds the complete new one.
    Never a partial mix of the two.
    """
    aside = None
    if os.path.exists(dst_dir):
        aside = f"{dst_dir}.stale-{uuid.uuid4().hex}"
        os.rename(dst_dir, aside)
    try:
        shutil.move(src_dir, dst_dir)
    except Exception:
        # Best-effort restore of the pre-existing directory so a failed
        # move doesn't also delete the old, still-good product.
        if aside is not None and not os.path.exists(dst_dir):
            os.rename(aside, dst_dir)
        raise
    if aside is not None:
        shutil.rmtree(aside, ignore_errors=True)
    if verbose:
        print(f"[assembly] moved {os.path.basename(dst_dir)}/ into place at {dst_dir}",
             flush=True)


def build_and_write_assembly(curation_state_path, yso_root, out_dir, data_root, verbose=True):
    """
    Full assembly run (spec SS6): build the row-provenance table,
    pre-create the 5 stratum output skeletons at their true final size,
    sweep every contributing geometry exactly once (reading and scatter-
    writing), then close everything. Returns the row-provenance table and
    the {stratum_folder: {file_key: path}} mapping.

    `data_root` resolves the two 1 Myr PMS track files `members.fits`'s
    derived `mass` column needs (`yso_pms.PMSTrack`) -- same convention
    as `yso_curate.run_pms`.

    **I/O note (not a scientific change):** the actual writing happens in
    a staging directory outside the (Dropbox-synced) `out_dir` tree, not
    in `out_dir` itself. The scatter-write sweep (SS6.1/SS6.3) keeps
    ~60 files memory-mapped in mode='update' and re-dirties pages across
    them for the run's whole duration; inside a live sync daemon's watched
    tree that repeated re-dirtying is what is slow (measured 72 min
    in-tree vs 6.3 min out-of-tree for the identical bytes -- the sweep
    itself, not the I/O volume, since the one purely sequential 8.2 GB
    write in this phase, skeleton pre-allocation, is unaffected: ~29s vs
    ~35s). Staging under `os.path.expanduser("~")` -- same APFS volume as
    `out_dir` in the deployed layout -- means the final move is an
    `os.rename` per stratum folder, not a second 8.2 GB copy. Every
    numerical value, row order, column, and on-disk file structure is
    identical to writing straight into `out_dir`; only the path the bytes
    are written to first, and how they get to their final location,
    changed.

    The staging directory is always removed before returning, on every
    exit path (success, an exception from any stage below, or
    KeyboardInterrupt) -- an aborted run must never leave ~8.2 GB behind
    in the user's home directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    _sweep_stale_staging(verbose=verbose)
    staging_dir = tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=os.path.expanduser("~"))
    try:
        same_volume = os.stat(staging_dir).st_dev == os.stat(out_dir).st_dev
        if verbose:
            if same_volume:
                print(f"[assembly] staging output in {staging_dir} (same filesystem "
                     f"volume as {out_dir} -- the final move will be an instant "
                     "rename, not a copy)", flush=True)
            else:
                print(f"[assembly] WARNING: staging directory {staging_dir} is on a "
                     f"different filesystem volume than {out_dir} (st_dev "
                     f"{os.stat(staging_dir).st_dev} vs {os.stat(out_dir).st_dev}) -- "
                     "the final move will copy the ~8.2 GB output instead of an "
                     "instant rename. Still correct, just slower.", flush=True)

        table = build_row_provenance_table(curation_state_path, verbose=verbose)
        n_k = stratum_n_k(table)

        reference_dir = reference_geometry_dir(table, yso_root)
        if verbose:
            print(f"[assembly] reference geometry for universal constants: "
                 f"{os.path.basename(reference_dir)}", flush=True)

        staged_paths = create_output_skeletons(staging_dir, n_k, reference_dir, verbose=verbose)
        handles = _open_stratum_handles(staged_paths)
        try:
            sweep_and_write(table, yso_root, handles, verbose=verbose)
        finally:
            _close_stratum_handles(handles)

        write_stratum_classmaps(staging_dir, table, verbose=verbose)

        fps_metadata = reg.read_metadata(curation_state_path, "fps")
        member_stats = compute_member_parameter_ranges(
            curation_state_path, yso_root, data_root, verbose=verbose)
        write_stratum_members_fits(
            staging_dir, table, member_stats,
            sigma_eff=fps_metadata["sigma_eff"], n_eligible=fps_metadata["n_eligible"],
            algorithm_note=fps_metadata.get("algorithm", ""), verbose=verbose)

        if verbose:
            print(f"[assembly] staged output complete -- moving "
                 f"{len(staged_paths)} stratum folder(s) from staging into {out_dir}",
                 flush=True)

        stratum_paths = {}
        for stratum_folder in staged_paths:
            src_dir = os.path.join(staging_dir, stratum_folder)
            dst_dir = os.path.join(out_dir, stratum_folder)
            _move_stratum_into_place(src_dir, dst_dir, verbose=verbose)
            stratum_paths[stratum_folder] = _stratum_file_paths(out_dir, stratum_folder)

        if verbose:
            print(f"[assembly] done -- wrote {sum(n_k.values())} models across "
                 f"{len(stratum_paths)} stratum folders", flush=True)
        return table, stratum_paths
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


# ==========================================================================
# QUALITY / ACCURACY REPORT (post-hoc, run after build_and_write_assembly)
# ==========================================================================

def _band_output_path(out_dir, stratum_folder, band):
    return os.path.join(out_dir, stratum_folder, "convolved", OUTPUT_BAND_FILENAME[band])


def qc_structural_completeness(out_dir, expected_n_k, verbose=True):
    """
    Every stratum's files agree on row count with the curation state's own
    `n_k` (from a freshly-rebuilt row-provenance table, not assumed from
    the run that already happened), and no output-skeleton placeholder
    survives (an empty `MODEL_NAME`/`Model Name`, `geometry_row_index ==
    -1`, or `Stage == -1` means some row was never actually written to).
    """
    issues = []
    for stratum_folder, n_k in expected_n_k.items():
        stratum_dir = os.path.join(out_dir, stratum_folder)
        with fits.open(os.path.join(stratum_dir, "info.fits"), memmap=True) as h:
            info = h[1].data
            if len(info) != n_k:
                issues.append(f"{stratum_folder}: info.fits has {len(info)} rows, expected {n_k}")
            if (np.asarray(info["MODEL_NAME"]) == "").any():
                issues.append(f"{stratum_folder}: info.fits has empty Model Name placeholder(s)")
            if (np.asarray(info["geometry_row_index"]) == -1).any():
                issues.append(f"{stratum_folder}: info.fits has geometry_row_index==-1 placeholder(s)")
            if (np.asarray(info["Stage"]) == -1).any():
                issues.append(f"{stratum_folder}: info.fits has Stage==-1 placeholder(s) -- an "
                             "assembled model must have passed every eligibility gate, "
                             "valid_stage among them, so this should never happen")

        with fits.open(os.path.join(stratum_dir, "parameters.fits"), memmap=True) as h:
            n_params = len(h[1].data)
        with fits.open(os.path.join(stratum_dir, "flux.fits"), memmap=True) as h:
            n_flux = np.asarray(h["MODEL_NAMES"].data["MODEL_NAME"]).size
        with fits.open(os.path.join(stratum_dir, "stellar.fits"), memmap=True) as h:
            n_stellar = np.asarray(h["MODEL_NAMES"].data["MODEL_NAME"]).size
        for n_check, label in ((n_params, "parameters.fits"), (n_flux, "flux.fits"),
                               (n_stellar, "stellar.fits")):
            if n_check != n_k:
                issues.append(f"{stratum_folder}: {label} has {n_check} rows, expected {n_k}")
        for band in BAND_ORDER:
            with fits.open(_band_output_path(out_dir, stratum_folder, band), memmap=True) as h:
                n_band = len(h[1].data)
            if n_band != n_k:
                issues.append(f"{stratum_folder}: convolved/{band}.fits has {n_band} rows, "
                             f"expected {n_k}")

    if verbose:
        print(f"[qc] structural completeness: {'OK' if not issues else str(len(issues)) + ' issue(s)'}",
             flush=True)
        for i in issues:
            print(f"[qc]   {i}", flush=True)
    return issues


def qc_full_population_accounting(out_dir, curation_state_path, verbose=True):
    """
    The union of every stratum's `(geometry, geometry_row_index)` pairs
    (read back from each stratum's own `info.fits` provenance columns) must
    exactly equal the curation state's `fps.selected == 1` model_id set -- no
    model omitted, none duplicated, none invented.
    """
    expected = set(build_row_provenance_table(curation_state_path, verbose=False)["model_id"])

    # Chain check: every assembled model must ALSO be marked eligible.
    # True by construction (build_fps_outputs writes selected=-1 for
    # anything not eligible==1), which is exactly why it is worth
    # asserting -- it fails only if the phases were run out of order or a
    # dataset was written by hand, and that failure is otherwise silent.
    eligibility = reg.read_dataset(curation_state_path, "eligibility", columns=["eligible"])
    eligible_ids = set(eligibility.loc[reg.is_true(eligibility["eligible"]), "model_id"])
    selected_not_eligible = expected - eligible_ids

    found = []
    for stratum_folder in STRATUM_OUTPUT_FOLDER.values():
        with fits.open(os.path.join(out_dir, stratum_folder, "info.fits"), memmap=True) as h:
            info = h[1].data
            geoms = np.asarray(info["geometry"]).astype(str)
            idxs = np.asarray(info["geometry_row_index"])
        found.extend(f"{g}:{i}" for g, i in zip(geoms, idxs))

    found_set = set(found)
    n_dupes = len(found) - len(found_set)
    missing = expected - found_set
    extra = found_set - expected

    issues = []
    if selected_not_eligible:
        issues.append(f"{len(selected_not_eligible)} fps.selected model(s) are not "
                      f"eligibility.eligible==1 (e.g. {sorted(selected_not_eligible)[:5]}) -- "
                      "the curation state's phases disagree about who was in scope")
    if n_dupes:
        issues.append(f"{n_dupes} duplicate model_id(s) across stratum outputs")
    if missing:
        issues.append(f"{len(missing)} fps.selected model_id(s) missing from every stratum output "
                      f"(e.g. {sorted(missing)[:5]})")
    if extra:
        issues.append(f"{len(extra)} model_id(s) in stratum outputs not in fps.selected "
                      f"(e.g. {sorted(extra)[:5]})")

    if verbose:
        print(f"[qc] full-population accounting: {len(found)} rows written, "
             f"{len(expected)} expected -- {'OK' if not issues else str(len(issues)) + ' issue(s)'}",
             flush=True)
        for i in issues:
            print(f"[qc]   {i}", flush=True)
    return issues


def qc_universal_constants(out_dir, reference_dir, verbose=True):
    """APERTURES/SPECTRAL_INFO/DISTANCE must match the reference geometry
    EXACTLY in every stratum's output (spec SS2.4/SS6.4 -- copied verbatim,
    never recomputed). FILTWAV must instead match `BANDS[band].wvl_effective_um`,
    the project-wide standard every library is stamped against, NOT the
    release's own value -- see `_read_reference_constants`."""
    ref = _read_reference_constants(reference_dir)
    issues = []
    for stratum_folder in STRATUM_OUTPUT_FOLDER.values():
        stratum_dir = os.path.join(out_dir, stratum_folder)
        with fits.open(os.path.join(stratum_dir, "flux.fits"), memmap=True) as h:
            if not np.array_equal(h["APERTURES"].data["APERTURE"], ref["aperture_au"]):
                issues.append(f"{stratum_folder}: flux.fits APERTURES mismatch")
            if not np.array_equal(h["SPECTRAL_INFO"].data["WAVELENGTH"], ref["wavelength"]):
                issues.append(f"{stratum_folder}: flux.fits SPECTRAL_INFO WAVELENGTH mismatch")
            if float(h["PRIMARY"].header["DISTANCE"]) != ref["distance_cm"]:
                issues.append(f"{stratum_folder}: flux.fits DISTANCE mismatch")
        for band in BAND_ORDER:
            with fits.open(_band_output_path(out_dir, stratum_folder, band), memmap=True) as h:
                if not np.array_equal(h["APERTURES"].data["APERTURE"], ref["aperture_au"]):
                    issues.append(f"{stratum_folder}: convolved/{band}.fits APERTURES mismatch")
                if float(h["PRIMARY"].header["FILTWAV"]) != ref["filtwav"][band]:
                    issues.append(f"{stratum_folder}: convolved/{band}.fits FILTWAV is "
                                  f"{float(h['PRIMARY'].header['FILTWAV'])}, expected the "
                                  f"project-wide {ref['filtwav'][band]}")

    if verbose:
        print(f"[qc] universal constants: {'OK' if not issues else str(len(issues)) + ' issue(s)'}",
             flush=True)
        for i in issues:
            print(f"[qc]   {i}", flush=True)
    return issues


def qc_contract_validation(out_dir, verbose=True):
    """model_io.validate_model_directory(profile='yso') against every
    stratum -- the release-contract checks (EXTNAME, column units, PRIMARY
    validity mask, models.conf keys, MODEL_NAME width/truncation, etc.)
    that qc_universal_constants does not cover: that check is a targeted
    bit-identity comparison against the reference geometry (kept as-is),
    not a contract check, which is exactly why a passing QC run once
    shipped FITS files sedfitter couldn't load."""
    issues = []
    for stratum_folder in STRATUM_OUTPUT_FOLDER.values():
        problems = validate_model_directory(os.path.join(out_dir, stratum_folder), profile="yso")
        issues.extend(f"{stratum_folder}: {p}" for p in problems)

    if verbose:
        print(f"[qc] contract validation: {'OK' if not issues else str(len(issues)) + ' issue(s)'}",
             flush=True)
        for i in issues:
            print(f"[qc]   {i}", flush=True)
    return issues


def qc_value_spotcheck(out_dir, yso_root, n_sample_per_stratum=300, seed=0, verbose=True):
    """
    For a random sample of rows in each stratum's output, re-read the
    recorded source geometry directly (via the row's own `info.fits`
    `geometry`/`geometry_row_index` provenance, not a re-derived mapping)
    and compare against the output value -- every `PARAM_UNION_COLUMNS`
    entry present in the source geometry, `flux.fits` VALUES, and one
    convolved band. This is the direct check that the actual bytes
    written match the actual bytes in the release, at real production
    scale, not just on the earlier smoke-test's small sample.

    Indexes `parameters.fits`/`flux.fits` by `geometry_row_index` directly
    -- both share the same positional row-order convention as MODEL_NAME
    (spec SS2.1; `make_model_id`'s own convention), so there is no need to
    materialize and linearly search a geometry's full MODEL_NAME column
    (up to 720k strings) per sampled row.
    Per-geometry files are also opened once, not once per row: samples are
    grouped by source geometry first.

    Tolerances are `SPOTCHECK_RTOL`/`SPOTCHECK_ATOL` (named, not an
    implicit `np.isclose` default).
    """
    rng = np.random.default_rng(seed)
    mismatches = []
    n_checked = 0

    for stratum_folder in STRATUM_OUTPUT_FOLDER.values():
        stratum_dir = os.path.join(out_dir, stratum_folder)
        with fits.open(os.path.join(stratum_dir, "info.fits"), memmap=True) as h:
            info = h[1].data
            n_k = len(info)
            if n_k == 0:
                continue
            sample = rng.choice(n_k, size=min(n_sample_per_stratum, n_k), replace=False)
            sample_geom = np.asarray(info["geometry"]).astype(str)[sample]
            sample_gidx = np.asarray(info["geometry_row_index"])[sample]

        with fits.open(os.path.join(stratum_dir, "parameters.fits"), memmap=True) as h:
            out_params = h[1].data
            out_model_name = np.asarray(out_params["MODEL_NAME"]).astype(str)[sample]
            out_param_cols = {col: np.asarray(out_params[col])[sample] for col in PARAM_UNION_COLUMNS}
        with fits.open(os.path.join(stratum_dir, "flux.fits"), memmap=True) as h:
            out_flux = h["VALUES"].data[sample, :, :]

        # Per-geometry opens hoisted out of the row loop: one parameters.fits
        # + flux.fits handle per DISTINCT source geometry in this sample,
        # not one per sampled row.
        for geom in np.unique(sample_geom):
            rows = np.where(sample_geom == geom)[0]
            gidx = sample_gidx[rows]

            params_path = os.path.join(yso_root, geom, "parameters.fits")
            with fits.open(params_path, memmap=True) as h:
                src = h[1].data
                src_name = np.char.strip(np.asarray(src["MODEL_NAME"][gidx]).astype(str))
                src_cols = {}
                for col in PARAM_UNION_COLUMNS:
                    if col in src.columns.names:
                        src_cols[col] = np.asarray(src[col][gidx])

            name_ok = src_name == out_model_name[rows]
            for j, r in enumerate(rows):
                if not name_ok[j]:
                    mismatches.append(f"{stratum_folder} sample {r} ({geom}:{gidx[j]}): "
                                     "parameters.fits MODEL_NAME mismatch")
                    continue
                for col, src_vals in src_cols.items():
                    if not np.allclose(src_vals[j], out_param_cols[col][r],
                                       rtol=SPOTCHECK_RTOL, atol=SPOTCHECK_ATOL, equal_nan=True):
                        mismatches.append(f"{stratum_folder} sample {r} ({geom}:{gidx[j]}): "
                                         f"parameters.fits column {col!r} mismatch")

            flux_path = os.path.join(yso_root, geom, "flux.fits")
            with fits.open(flux_path, memmap=True) as h:
                src_flux = h["VALUES"].data[gidx, :, :]
            for j, r in enumerate(rows):
                if not np.allclose(out_flux[r], src_flux[j],
                                   rtol=SPOTCHECK_RTOL, atol=SPOTCHECK_ATOL, equal_nan=True):
                    mismatches.append(f"{stratum_folder} sample {r} ({geom}:{gidx[j]}): "
                                     "flux.fits VALUES mismatch")
                n_checked += 1

    if verbose:
        print(f"[qc] value spot-check: {n_checked} rows verified against source -- "
             f"{'OK' if not mismatches else str(len(mismatches)) + ' mismatch(es)'}", flush=True)
        for m in mismatches[:20]:
            print(f"[qc]   {m}", flush=True)
    return mismatches


def qc_distribution_summary(out_dir, verbose=True):
    """Per-stratum Richardson Stage/Class breakdown and geometry composition
    -- not a pass/fail check, a sanity read: e.g. c0 should skew toward
    Stage 0, cIII/td toward bare-star/near-diskless geometries."""
    summary = {}
    for stratum_folder in STRATUM_OUTPUT_FOLDER.values():
        with fits.open(os.path.join(out_dir, stratum_folder, "info.fits"), memmap=True) as h:
            info = h[1].data
            if len(info) == 0:
                summary[stratum_folder] = {"stage_counts": {}, "geometry_counts": {}}
                continue
            stage_vals, stage_counts = np.unique(np.asarray(info["Stage"]), return_counts=True)
            geom_vals, geom_counts = np.unique(np.asarray(info["geometry"]).astype(str),
                                               return_counts=True)
        summary[stratum_folder] = {
            "stage_counts": dict(zip(stage_vals.tolist(), stage_counts.tolist())),
            "geometry_counts": dict(sorted(zip(geom_vals.tolist(), geom_counts.tolist()),
                                          key=lambda kv: -kv[1])[:5]),
        }
        if verbose:
            print(f"[qc] {stratum_folder}: Richardson Stage counts = "
                 f"{summary[stratum_folder]['stage_counts']}", flush=True)
            print(f"[qc] {stratum_folder}: top geometries = "
                 f"{summary[stratum_folder]['geometry_counts']}", flush=True)
    return summary


def qc_report(out_dir, curation_state_path, yso_root, n_sample_per_stratum=300, seed=0, verbose=True):
    """
    Full post-hoc quality/accuracy report for a completed assembly run
    -- independent of `build_and_write_assembly` (re-reads everything from
    disk/curation-state fresh, no shared state with whatever run already
    happened). Six checks:

    1. `qc_structural_completeness` -- row counts match the curation state's
       own n_k, no placeholder values survive.
    2. `qc_full_population_accounting` -- every fps.selected model_id
       appears in exactly one stratum output, no duplicates, nothing extra.
    3. `qc_universal_constants` -- APERTURES/SPECTRAL_INFO/DISTANCE/
       FILTWAV match the reference geometry exactly everywhere.
    4. `qc_contract_validation` -- model_io.validate_model_directory
       (profile='yso') against every stratum: EXTNAME, column units,
       PRIMARY validity mask, models.conf keys, MODEL_NAME width, etc.
       This is the check that would have caught the release contract
       being broken (missing units/DISTANCE) despite QC passing before.
    5. `qc_value_spotcheck` -- a random sample per stratum (default 300)
       re-verified against the source release directly, at production
       scale (not just the earlier smoke test's handful of rows).
    6. `qc_distribution_summary` -- per-stratum Stage/geometry composition,
       for a human sanity read, not a pass/fail gate.

    Returns {check_name: result}; prints a PASS/FAIL summary line per
    check if verbose.
    """
    table = build_row_provenance_table(curation_state_path, verbose=False)
    n_k = stratum_n_k(table)
    reference_dir = reference_geometry_dir(table, yso_root)

    results = {
        "structural_completeness": qc_structural_completeness(out_dir, n_k, verbose=verbose),
        "full_population_accounting": qc_full_population_accounting(out_dir, curation_state_path,
                                                                    verbose=verbose),
        "universal_constants": qc_universal_constants(out_dir, reference_dir, verbose=verbose),
        "contract_validation": qc_contract_validation(out_dir, verbose=verbose),
        "value_spotcheck": qc_value_spotcheck(out_dir, yso_root,
                                              n_sample_per_stratum=n_sample_per_stratum,
                                              seed=seed, verbose=verbose),
    }
    distribution = qc_distribution_summary(out_dir, verbose=verbose)
    results["distribution_summary"] = distribution

    if verbose:
        n_issues = sum(len(v) for k, v in results.items() if k != "distribution_summary")
        print(f"\n[qc] TOTAL: {'ALL CHECKS PASSED' if n_issues == 0 else f'{n_issues} issue(s) found'}",
             flush=True)
    return results
