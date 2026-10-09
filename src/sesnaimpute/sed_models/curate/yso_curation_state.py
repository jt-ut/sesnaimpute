"""
yso_curation_state.py
====================================================================
The YSO model library curation state: schema, seeding, and the read/write
contract every curation phase (labeling, dedup, fps, assembly) shares.
Implements Section 1 of `docs/yso_curation_plan.md`.

The curation state is one HDF5 file holding **six datasets**, each keyed on
`model_id`, with every dataset containing exactly one row for every
`model_id` in the same order (Section 1.1) -- a phase that hasn't run, or
doesn't apply to a given model, leaves its columns null for that row,
never drops it:

    /registry/ingest       source columns read from the release
    /registry/eligibility  EVERY gate, one polarity, in pipeline order
    /registry/labeling     p_stage_* + label
    /features/mu_sed       the clamped hybrid survey-matched mu-SED, mJy
    /registry/dedup        dup / chi2 / matched_sps_id
    /registry/fps          selected

**Gates live in exactly one place.** `eligibility` holds every
inclusion/exclusion decision -- `valid_star`, `valid_stage`,
`valid_aperture`, `valid_nondup` -- as same-polarity int8 flags in the
order the phases determine them, plus the derived `eligible` rollup and
`d_min_kpc` (the measurement the aperture gate thresholds). So "why is
model X not in the subsample" is answered by reading one row left to
right, and no phase re-derives eligibility for itself. `ingest` carries
only what the release says; `labeling` only its products.

`mu_sed` is a features store, not a flag/label dataset -- hence
`features/` rather than `registry/`. Dedup reads it directly as its
SPS-matching query SED.

This module defines the schema and the read/write contract; the phase
modules compute, and `yso_curate` sequences and writes.

**Missing-value encoding (Section 1.4) -- there is no universal HDF5
"null".** Verified empirically against this project's actual pandas/
pytables versions: plain `bool`/`int64`/`float64` round-trip cleanly
through `HDFStore(format="table")`; pandas' *nullable* extension dtypes
(`"boolean"`, `"Int64"`) do not -- writing one raises
`AttributeError: module 'tables' has no attribute 'BooleanCol'` outright,
not a silent degradation. So every column uses a type-appropriate
sentinel instead of a nullable dtype:
    - **flags** (`valid_star`, `valid_stage`, `valid_aperture`,
      `valid_nondup`, `eligible`, `dup`, `selected`) -- `int8`, `1`/`0`
      for True/False, `FLAG_NA = -1` for "not yet determined".
      Never pandas `bool`/`"boolean"`. Read via `is_true()`/
      `is_determined()`, not ambient truthiness -- `-1` is a nonzero int
      and therefore Python-truthy, so `if row['valid_star']:` would treat
      "not yet determined" as "true."
    - **Stage-like ints** (`stage`, `alpha_class`) -- plain `int64`, `-1`
      is also the domain's own "unclassified"/"not a real classification"
      sentinel (never actually missing for any row).
    - **floats** (`spectral_index`, `p_stage_*`) -- `float64`, `NaN` for
      missing (floats round-trip NaN through HDF5 natively, no sentinel
      needed).
    - **`label`** -- pandas `category` dtype; confirmed
      empirically to round-trip a real missing value (as NaN) through
      `format="table"` cleanly, unlike the nullable "boolean"/"Int64"
      extension types -- no sentinel needed here either.
    - **strings** (`model_name`, `matched_sps_id`) -- plain `str`, empty
      string `""` for missing.

**`model_id` is a synthetic `<geometry>:<row_index>` key, not
`MODEL_NAME`.** Confirmed
against the real release: 1,258 distinct `MODEL_NAME` values repeat within
a single geometry (11,322 rows, confined to `s-pbhmi`/`s-pbsmi`), always on
already-invalid placeholder rows (`star.radius == 0`, Richardson
`Class == -1`) -- an upstream artifact of those two geometries' files, not
a reader bug. Row position is used instead because `parameters.fits`,
`info.fits`, and `flux.fits` were verified to share byte-identical row
order for both affected geometries, so it's a real, collision-free key
every file already agrees on -- any later phase streaming `flux.fits` in
native order can reproduce the same `model_id` with no curation-state lookup.
The original `MODEL_NAME` is kept as an ordinary `model_name` column.
`make_model_id` itself lives in `model_io.py`, not here, since it is a
property of the model grid, not of this curation state specifically.
====================================================================
"""

import json
import os
import shutil

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except ImportError:
    fits = None

from sesnaimpute.sed_models.curate.model_io import align_by_name, make_model_id
from sesnaimpute.sed_models.curate.yso_labeling import BAND_ORDER, STAGE_UNCLASSIFIED

# The 18 YSO geometry subfolders (Richardson+ 2024 / Robitaille 2017),
# matching the order tallied in docs/yso_upstream_release_format.md.
GEOMETRIES = (
    "s---s-i", "s---smi", "s-p-hmi", "s-p-smi", "s-pbhmi", "s-pbsmi",
    "s-u-hmi", "s-u-smi", "s-ubhmi", "s-ubsmi", "sp--h-i", "sp--hmi",
    "sp--s-i", "sp--smi", "spu-hmi", "spu-smi", "spubhmi", "spubsmi",
)

# Not-yet-determined sentinel for int8 flag columns (Section 1.4). A
# nonzero int, so NEVER test a flag with ambient truthiness -- use
# is_true()/is_determined() below.
FLAG_NA = -1

# Per-dataset column name/dtype schema (Section 1.1/1.4) -- the single
# source of truth every phase's reader/writer imports rather than
# hardcoding strings independently (Section 1.4). All five datasets
# (ingest, labeling, mu_sed, dedup, fps) are defined here.
SCHEMA = {
    "ingest": {
        # SOURCE COLUMNS ONLY -- no gates. Every eligibility flag lives in
        # the `eligibility` dataset below, so "what the release says" and
        # "what we decided about it" are never interleaved in one table.
        "model_id": "string",           # synthetic "<geometry_folder>:<row_index>" key --
                                         # see module docstring; NOT MODEL_NAME
        "model_name": "string",         # original MODEL_NAME, reference/debugging only
        "geometry_folder": "string",
        "inclination": "float64",       # nullable (NaN): not every geometry row need carry one
        "star_radius": "float64",       # raw star.radius from parameters.fits
        "stage": "int64",               # Richardson's structural Stage (0/I/II/III as 0/1/2/3,
                                         # or -1 unclassified -- STAGE_UNCLASSIFIED, always
                                         # defined, never actually missing)
        "alpha_class": "int64",         # Richardson's own alpha-based Class code, -1/0/1/2/3/4,
                                         # always defined -- reference/cross-check only (Section 1.2)
        "spectral_index": "float64",    # Spectral Index at the largest aperture
    },

    "eligibility": {
        # EVERY eligibility gate, one polarity (1=pass / 0=fail /
        # FLAG_NA=not yet determined), in pipeline order -- so "how does
        # eligibility evolve" is answered by reading one row left to right.
        # `eligible` is the Kleene AND over GATE_COLUMNS (see
        # compute_eligible); it is a materialised convenience that provably
        # cannot disagree with the gates, never an independent fact.
        "model_id": "string",
        "valid_star": "int8",           # gate 1, ingest:   star_radius > 0
        "valid_stage": "int8",          # gate 2, ingest:   stage != STAGE_UNCLASSIFIED
        "valid_aperture": "int8",       # gate 3, aperture: d_min_kpc < D_NEAR
        "d_min_kpc": "float64",         # NOT a gate -- the measurement gate 3 thresholds.
                                         # Minimum distance at which all 8 bands' survey
                                         # apertures are bracketed by DEFINED grid apertures.
                                         # Written for every model, independent of any other
                                         # gate, so a future distance range can be evaluated
                                         # without rescanning the 18 geometries. NaN = not
                                         # yet computed.
        "valid_pms": "int8",            # gate 4, pms (yso_pms.py): the model's own
                                         # (star.temperature, Source Luminosity) matches the
                                         # 1 Myr BHAC15+MIST pre-main-sequence track at SOME
                                         # mass, within yso_pms.PMS_TOL_DEX.
        "pms_delta_dex": "float64",     # NOT a gate -- the measurement gate 4 thresholds:
                                         # log10(T_eff_obs) - log10(T_eff_track) at the
                                         # L-implied mass. NaN = not yet computed.
        "valid_nondup": "int8",         # gate 5, dedup:    dedup.dup == 0. Polarity-corrected
                                         # view of `dup` (whose 1 is the PROBLEM outcome) so
                                         # the rollup is a plain same-polarity AND.
        "eligible": "int8",             # derived: Kleene AND over GATE_COLUMNS, written by
                                         # the finalize stage once every gate is determined
    },

    "labeling": {
        # Names match yso_labeling.py's label output (labeling spec §6):
        # (p_stage_0, p_stage_I, p_stage_II, p_stage_III, p_stage_TD) sum to 1;
        # label is their argmax.
        "model_id": "string",
        "p_stage_0": "float64",
        "p_stage_I": "float64",
        "p_stage_II": "float64",
        "p_stage_III": "float64",
        "p_stage_TD": "float64",
        "label": "category",
        # No valid_flux: `peak_i > 0` is implied by eligibility.valid_aperture
        # (a model with no flux anywhere has every aperture undefined, so it
        # fails the aperture gate first). Verified on the current grid --
        # after the aperture gate, a peak_i>0 test removes 0 further rows.
        # The 4 all-zero-flux rows are accounted for under valid_aperture.
    },

    "mu_sed": {
        # The clamped, hybrid survey-matched mu-SED (labeling spec §4.1/§6),
        # in LINEAR mJy, one column per band (BAND_ORDER). A features store,
        # not a flag/label dataset -- dedup reads this directly as its query
        # SED against the SPS library rather than recomputing it. NaN for
        # any model_id never labeled -- no SED was computed for those, so
        # there is nothing to persist.
        #
        # LINEAR HERE, LOG EVERYWHERE ELSE. This is the only place the
        # pipeline holds linear flux, because that is the unit sedfitter's
        # `error` array and the SPS convolved fluxes use, so dedup can hand
        # it straight to fit_sources. Every distance and neighbour search is
        # in log space: labeling never round-trips through storage (it keeps
        # mu in dex and packs mu/sigma vectors for FAISS), and fps converts
        # to Vega magnitudes on read. Do not "simplify" this to dex -- dedup
        # is the one call site that must not undo it.
        #
        # Built by curve_of_growth.survey_matched_sed: linear-in-AU
        # interpolation matching the fitter (D-23), floor applied after
        # (D-9). sed_models_register's f_ref is the same quantity from the same
        # function; the names differ deliberately (f_ref carries the
        # census's B_hat * f_ref identity).
        "model_id": "string",
        **{f"mu_{band}": "float64" for band in BAND_ORDER},
    },

    "dedup": {
        # Filled by yso_dedup.compute_dedup (curation plan §2; dedup spec).
        # dup=1 duplicate / 0 tested&unique / FLAG_NA(-1) not tested (no
        # label). chi2/matched_sps_id recorded for every TESTED
        # model (dup in {0,1}), not only duplicates -- full audit trail
        # (dedup spec §2.3 step 3/§5). NaN/"" for dup=-1 rows via the
        # standard align_to_ingest fill.
        "model_id": "string",
        "dup": "int8",
        "chi2": "float64",
        "matched_sps_id": "string",
    },

    "fps": {
        # Filled by yso_fps (owner library spec, 2026-10-08): ONE global
        # greedy-coverage r-net over the whole gated (eligible==1) set, no
        # strata, no per-stratum budget -- superseding the former
        # per-stratum FPS this dataset was originally named for. selected=1
        # KEPT TEMPLATE (a representative the r-net chose) / 0 eligible,
        # covered by some kept template, but not itself kept / FLAG_NA(-1)
        # not eligible (eligibility.eligible != 1).
        "model_id": "string",
        "selected": "int8",
        "owner_model_id": "string",     # the kept template this row's represented-set
                                         # membership belongs to (itself, for selected==1
                                         # rows) -- "" for FLAG_NA(-1) rows, never computed.
        "dist_sigeff": "float64",       # quotient-space distance to owner_model_id, in units
                                         # of SIGEFF (so <= 1.0 for every determined row,
                                         # by the r-net's own covering guarantee). NaN for
                                         # FLAG_NA(-1) rows.
    },
}

# int8-flag columns needing the FLAG_NA fill/cast treatment in
# _coerce_dtypes -- kept in sync with SCHEMA above, not re-derived from it,
# so a future "int8"-typed non-flag column (unlikely, but not impossible)
# can't silently get swept into flag handling.
FLAG_COLUMNS = frozenset({"valid_star", "valid_stage", "valid_aperture",
                          "valid_nondup", "eligible", "dup", "selected"})

# The eligibility gates, IN PIPELINE ORDER. This tuple is the contract:
#   * `eligible` is exactly the Kleene AND over these columns
#     (compute_eligible), so the gates alone reconstruct it;
#   * a future gate is added by appending here -- the rollup and every
#     consumer pick it up with no other edit;
#   * the order is the order the phases determine them, so reading an
#     eligibility row left to right IS the pipeline's decision history.
# Deliberately a tuple, not the FLAG_COLUMNS frozenset: order is meaning.
GATE_COLUMNS = ("valid_star", "valid_stage", "valid_aperture", "valid_pms", "valid_nondup")

# HDF5 dataset key per phase (Section 1.1's "/registry/{phase}" paths,
# without the leading slash -- pandas.HDFStore keys are looked up either
# way, but store.keys() reports them with one, see write_dataset). "mu_sed"
# lives under "features/", not "registry/", matching the labeling spec's
# own distinction between the curation state's flag/label datasets and the
# separate mu-SED features store (spec §6) -- same HDF5 file, different
# top-level group.
DATASET_KEYS = {
    "ingest": "registry/ingest",
    "eligibility": "registry/eligibility",
    "labeling": "registry/labeling",
    "mu_sed": "features/mu_sed",
    "dedup": "registry/dedup",
    "fps": "registry/fps",
}

# Run-level metadata group paths (Section 1.3) -- written/read via
# write_metadata/read_metadata below. `ingest` records the upstream
# source root/row count/build date -- the only place that identity is
# recorded, so it is not optional. The other three phase drivers each write
# their own group here too (curate_yso_labeling.py/curate_yso_dedup.py/
# curate_yso_fps.py).
GROUP_PATHS = {
    "ingest": "/metadata/ingest",
    "aperture": "/metadata/aperture",
    "pms": "/metadata/pms",
    "labeling": "/metadata/labeling",
    "dedup": "/metadata/dedup",
    "fps": "/metadata/fps",
}

DEFAULT_CURATION_STATE_FILENAME = "yso_curation_state.h5"


def is_true(flag_col):
    """True where an int8 flag column (Section 1.4) equals 1. Do not use
    ambient truthiness (`if row['valid_flux']:`) -- FLAG_NA (-1) is a
    nonzero int and therefore Python-truthy."""
    return flag_col == 1


def is_determined(flag_col):
    """True where an int8 flag column (Section 1.4) has been resolved,
    i.e. is not the FLAG_NA (-1) not-determined sentinel."""
    return flag_col != FLAG_NA


def compute_eligible(elig):
    """
    The `eligible` rollup: a **Kleene (three-valued) AND** over
    GATE_COLUMNS, returned as an int8 array aligned to `elig`'s rows.

        any gate == 0          -> 0    (definitely ineligible)
        all gates == 1         -> 1    (definitely eligible)
        otherwise              -> -1   (some gate not yet determined)

    Kleene rather than "any -1 makes the whole thing -1": a model that has
    already FAILED a determined gate is ineligible no matter what the
    outstanding phases decide, so reporting 0 is both correct and final.
    The safety property the curation plan cares about is preserved either
    way -- a not-yet-run phase can never make a model look eligible,
    because `1` still requires every gate to be determined AND passing.

    This is the ONLY definition of eligibility in the codebase. No phase
    re-derives it (fps previously open-coded `is_determined(dup) &
    ~is_true(dup)` in two places); they read the stored column, which the
    finalize stage writes from exactly this function.
    """
    missing = [c for c in GATE_COLUMNS if c not in elig.columns]
    if missing:
        raise ValueError(f"eligibility frame is missing gate column(s) {missing}")

    gates = np.stack([np.asarray(elig[c], dtype=np.int8) for c in GATE_COLUMNS])
    out = np.full(gates.shape[1], FLAG_NA, dtype=np.int8)
    out[(gates == 1).all(axis=0)] = 1
    out[(gates == 0).any(axis=0)] = 0      # after the all-pass write: 0 wins over -1
    return out


def valid_nondup_from_dup(dup):
    """The `valid_nondup` gate from dedup's own `dup` column -- the one
    place this polarity flip is written.

        dup == 0  (tested, not a duplicate)  -> 1   pass
        dup == 1  (confirmed SPS duplicate)  -> 0   fail
        dup == -1 (not tested)               -> -1  not determined

    `dup`'s `1` is the PROBLEM outcome, inverted relative to every other
    flag in the curation state -- which is exactly why eligibility stores the
    corrected view rather than making every reader remember the exception.
    """
    dup = np.asarray(dup)
    out = np.full(dup.shape[0], FLAG_NA, dtype="int8")
    out[dup == 0] = 1
    out[dup == 1] = 0
    return out


def seed_eligibility(ingest):
    """
    Build the `eligibility` dataset's initial state from `ingest`
    (Section 1.1): the two ingest-time gates resolved, every later gate at
    FLAG_NA and `d_min_kpc` at NaN.

    A pure function of `ingest` -- it derives `valid_star`/`valid_stage`
    from the source columns rather than having `seed_registry` emit them,
    so ingest stays "what the release says" and eligibility stays "what we
    decided about it."
    """
    n = len(ingest)
    return _coerce_dtypes(pd.DataFrame({
        "model_id": ingest["model_id"].to_numpy(),
        "valid_star": (np.asarray(ingest["star_radius"], dtype=float) > 0).astype("int8"),
        "valid_stage": (np.asarray(ingest["stage"], dtype="int64")
                        != STAGE_UNCLASSIFIED).astype("int8"),
        "valid_aperture": np.full(n, FLAG_NA, dtype="int8"),
        "d_min_kpc": np.full(n, np.nan),
        "valid_pms": np.full(n, FLAG_NA, dtype="int8"),
        "pms_delta_dex": np.full(n, np.nan),
        "valid_nondup": np.full(n, FLAG_NA, dtype="int8"),
        "eligible": np.full(n, FLAG_NA, dtype="int8"),
    }), "eligibility")


def _coerce_dtypes(df, dataset):
    """Cast every column present in `df` to SCHEMA[dataset]'s declared
    dtype, filling the type-appropriate missing-value sentinel first where
    the target dtype can't hold NaN natively (Section 1.4): int8 flags ->
    FLAG_NA, int64 Stage-like columns -> -1, strings -> ''. Floats and
    category dtypes round-trip NaN natively, so no fill is applied there."""
    df = df.copy()
    for col, dtype in SCHEMA[dataset].items():
        if col not in df.columns:
            continue
        if dtype == "int8":
            df[col] = df[col].fillna(FLAG_NA).astype("int8")
        elif dtype == "int64":
            df[col] = df[col].fillna(-1).astype("int64")
        elif dtype == "float64":
            df[col] = df[col].astype("float64")
        elif dtype == "category":
            df[col] = df[col].astype("category")
        elif dtype == "string":
            df[col] = df[col].fillna("").astype(str)
        else:
            raise ValueError(f"unhandled dtype {dtype!r} for column {col!r}")
    return df


def align_to_ingest(df, ingest_model_ids, dataset):
    """Reindex `df` (already containing a `model_id` column) to contain
    exactly one row per model_id in `ingest_model_ids`, in that exact
    order -- model_ids in the target not present in `df` get every column
    filled with the dataset's missing-value convention (Section 1.4),
    never dropped. This enforces the curation state's core invariant (Section
    1.1): every dataset has one row for every model_id, aligned, whether
    or not that phase actually touched a given row. `dataset` selects
    which SCHEMA entry (and therefore fill/cast convention) to apply."""
    df = df.set_index("model_id").reindex(ingest_model_ids).reset_index()
    return _coerce_dtypes(df, dataset)


def _seed_one_geometry(geom, geom_dir, verbose=True):
    """Read one geometry's seed columns from its parameters.fits/info.fits."""
    if fits is None:
        raise RuntimeError("astropy is required to read FITS files.")

    params_path = os.path.join(geom_dir, "parameters.fits")
    info_path = os.path.join(geom_dir, "info.fits")

    with fits.open(params_path, memmap=True) as ph:
        pdata = ph[1].data
        cols = pdata.columns.names
        model_name = np.asarray(pdata["MODEL_NAME"]).astype(str)
        star_radius = np.asarray(pdata["star.radius"], dtype=float)
        inclination = (np.asarray(pdata["inclination"], dtype=float)
                       if "inclination" in cols else np.full(model_name.size, np.nan))
        if "Spectral Index" in cols:
            # (n_models, 20); largest aperture is the last column -- apertures
            # are log-spaced ascending (100 AU -> 1e6 AU) per
            # docs/yso_upstream_release_format.md.
            spectral_index = np.asarray(pdata["Spectral Index"], dtype=float)[:, -1]
        else:
            spectral_index = np.full(model_name.size, np.nan)

    with fits.open(info_path, memmap=True) as ih:
        idata = ih[1].data
        info_names = np.asarray(idata["Model Name"]).astype(str)
        info_class = np.asarray(idata["Class"], dtype=float)
        info_stage = np.asarray(idata["Stage"], dtype=int)

    # Rows are documented as already in matching order, but align explicitly
    # rather than assume -- align_by_name's fast path is a no-op cost when so.
    # (Note: with duplicated names this dict-based fallback would be
    # ambiguous -- but the fast path applies here since arrays match exactly,
    # which is exactly why model_id uses row position, not MODEL_NAME.)
    order = align_by_name(model_name, info_names)
    alpha_class = info_class[order].astype("int64")
    stage = info_stage[order].astype("int64")

    row_index = np.arange(model_name.size)
    # Vectorised equivalent of [make_model_id(geom, i) for i in row_index]:
    # make_model_id is exactly f"{geometry}:{row_index}", and this builds the
    # same strings without a 720k-iteration Python loop per geometry. The
    # single-element call below pins that equivalence so a change to
    # make_model_id's format cannot silently desync this fast path.
    ids = np.char.add(f"{geom}:", row_index.astype(str))
    if model_name.size and ids[0] != make_model_id(geom, 0):
        raise AssertionError(
            f"vectorised model_id build ({ids[0]!r}) disagrees with "
            f"make_model_id ({make_model_id(geom, 0)!r}) -- the id format changed")

    df = pd.DataFrame({
        "model_id": ids,
        "model_name": model_name,
        "geometry_folder": geom,
        "inclination": inclination,
        "star_radius": star_radius,
        "stage": stage,
        "alpha_class": alpha_class,
        "spectral_index": spectral_index,
    })
    if verbose:
        print(f"[seed] {geom}: {len(df)} rows", flush=True)
    return df


def seed_registry(yso_root, geometries=GEOMETRIES, verbose=True):
    """
    Build the curation state's `ingest` dataset (Section 1.1) from the full,
    un-deduplicated YSO grid: model_id, model_name, geometry_folder,
    inclination, star_radius, stage, alpha_class, spectral_index -- source
    columns only. The two ingest-time GATES derived from these
    (valid_star, valid_stage) belong to the `eligibility` dataset; call
    seed_eligibility(ingest) to build it.

    Cheap, column-only reads (parameters.fits + info.fits per geometry) --
    never touches flux.fits (Sequencing step 1). model_id is synthetic
    (geometry + row position, see make_model_id) and therefore
    collision-free by construction; the uniqueness check below is a cheap
    sanity assertion, not an expected failure mode.
    """
    frames = [_seed_one_geometry(geom, os.path.join(yso_root, geom), verbose=verbose)
              for geom in geometries]
    df = pd.concat(frames, axis=0, ignore_index=True)

    n_dup = int(df["model_id"].duplicated().sum())
    if n_dup:
        raise ValueError(
            f"{n_dup} duplicate model_id values across the concatenated grid -- "
            "make_model_id should be collision-free by construction; this "
            "indicates a bug in the row-position keying, not a data issue.")
    return _coerce_dtypes(df, "ingest")


def read_dataset(path, dataset, columns=None):
    """Read one dataset (any SCHEMA key) from the curation-state file.

    `columns` reads just those columns off disk rather than the whole
    frame -- every dataset is the full 2.2M-row aligned set, so a caller
    that wants `model_id` + one flag should say so. `model_id` is always
    included, since it is the key every consumer joins on.
    """
    if dataset not in SCHEMA:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {list(SCHEMA)}")
    if columns is None:
        return pd.read_hdf(path, key=DATASET_KEYS[dataset])

    unknown = [c for c in columns if c not in SCHEMA[dataset]]
    if unknown:
        raise ValueError(f"{dataset!r} has no column(s) {unknown}; "
                         f"expected from {list(SCHEMA[dataset])}")
    wanted = ["model_id"] + [c for c in columns if c != "model_id"]
    return pd.read_hdf(path, key=DATASET_KEYS[dataset], columns=wanted)


def _atomic_update(out_dir, filename, puts):
    """
    Apply `puts` -- a {hdf_key: DataFrame} mapping -- to the curation-state file,
    atomically and WITHOUT touching any other key (Section 1.5).

    The file is copied at the filesystem level, the new keys are `put` into
    the copy, and the copy replaces the original. So:

      * **atomic** -- a crash leaves the original file untouched, which is
        the property Section 1.5 requires (HDF5 handles partial writes
        badly);
      * **scoped** -- only the keys in `puts` are rewritten. Every other
        dataset and `/metadata/*` group is carried through as opaque bytes,
        never deserialised into pandas and re-serialised.

    That second property is the point. The previous implementation read
    every dataset into memory and wrote them all back on every call, which
    made a one-dataset update cost the whole curation state: measured at **37 s
    per write** on the 668 MB production file versus **4.4 s** here, an
    8.4x saving that scales with the number of datasets. Bloat from HDF5
    not reclaiming replaced-node space is negligible in practice -- ~20 KB
    per write when the replacement is the same shape, which it always is
    here (every dataset is the full aligned row set).
    """
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, filename)
    tmp_path = final_path + ".tmp"

    if os.path.exists(final_path):
        shutil.copy2(final_path, tmp_path)
        mode = "a"
    else:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        mode = "w"

    try:
        with pd.HDFStore(tmp_path, mode=mode) as store:
            for key, frame in puts.items():
                store.put(key, frame, format="table")
        os.replace(tmp_path, final_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return final_path


def _existing_ingest_ids(path):
    """The existing file's ingest model_id column, or None if absent.
    Reads that ONE column rather than the whole 2.2M-row dataset -- the
    alignment guard needs nothing else."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_hdf(path, key=DATASET_KEYS["ingest"],
                           columns=["model_id"])["model_id"].to_numpy()
    except KeyError:
        return None


def write_datasets(frames, out_dir, filename=DEFAULT_CURATION_STATE_FILENAME):
    """
    Write/replace one or more datasets (any SCHEMA keys) in the curation-state
    file in a single atomic update -- `frames` is a {dataset: DataFrame}
    mapping. Every dataset and `/metadata/*` group NOT named in `frames` is
    left exactly as it was on disk, untouched and unread.

    Prefer this over repeated `write_dataset` calls when a phase produces
    several datasets (labeling writes `labeling` + `mu_sed`): one call is
    one file copy instead of two.

    For any dataset other than 'ingest', if an 'ingest' dataset already
    exists in the file, the frame's model_id column must match it exactly
    (same set, same order) -- call align_to_ingest() first; a mismatch
    raises rather than silently breaking the curation state's per-model row
    alignment (Section 1.1).
    """
    unknown = [d for d in frames if d not in SCHEMA]
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; expected from {list(SCHEMA)}")

    coerced = {d: _coerce_dtypes(df, d) for d, df in frames.items()}

    final_path = os.path.join(out_dir, filename)
    needs_guard = [d for d in coerced if d != "ingest"]
    if needs_guard:
        ingest_ids = _existing_ingest_ids(final_path)
        if ingest_ids is not None:
            for d in needs_guard:
                if not np.array_equal(coerced[d]["model_id"].to_numpy(), ingest_ids):
                    raise ValueError(
                        f"{d!r} dataset's model_id column does not match the "
                        f"existing 'ingest' dataset's model_id set/order -- call "
                        f"align_to_ingest() first (Section 1.1's alignment invariant).")

    return _atomic_update(out_dir, filename,
                          {DATASET_KEYS[d]: df for d, df in coerced.items()})


def write_dataset(df, dataset, out_dir, filename=DEFAULT_CURATION_STATE_FILENAME):
    """Write/replace a single dataset -- a one-entry `write_datasets`."""
    return write_datasets({dataset: df}, out_dir, filename=filename)


def write_metadata(metadata, phase, out_dir, filename=DEFAULT_CURATION_STATE_FILENAME):
    """
    Write run-level metadata (Section 1.3) for one phase (e.g. 'fps') to
    `/metadata/{phase}`, atomically (temp file + os.replace, same as
    write_dataset) -- every other dataset/metadata group already in the
    file is carried through unchanged.

    `metadata` must be JSON-serializable (plain dict/list/str/int/float/
    bool -- cast numpy scalars to native Python types before calling
    this). Stored as a single-row, single-column DataFrame holding the
    JSON-encoded string, so it round-trips through the same
    `pd.HDFStore(format="table")` machinery every other dataset uses --
    no new pytables API, no new dependency for a handful of run-level
    scalars/dicts.
    """
    if phase not in GROUP_PATHS:
        raise ValueError(f"unknown metadata phase {phase!r}; expected one of {list(GROUP_PATHS)}")

    payload = json.dumps(metadata, default=str)
    meta_df = pd.DataFrame({"json": [payload]})
    return _atomic_update(out_dir, filename, {GROUP_PATHS[phase]: meta_df})


def read_metadata(path, phase):
    """Read one phase's run-level metadata dict back from
    `/metadata/{phase}` (written by write_metadata)."""
    if phase not in GROUP_PATHS:
        raise ValueError(f"unknown metadata phase {phase!r}; expected one of {list(GROUP_PATHS)}")
    df = pd.read_hdf(path, key=GROUP_PATHS[phase])
    return json.loads(df["json"].iloc[0])
