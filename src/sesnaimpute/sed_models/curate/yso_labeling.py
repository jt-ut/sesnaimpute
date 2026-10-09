"""
yso_labeling.py
====================================================================
Labeling for the Robitaille (2017) / Richardson et al. (2024) YSO
model release: assigns every model row a soft probability distribution over
`{0, I, II, III, TD}`. Full method: see the companion specification document
`docs/yso_labeling_curation_plan.md`.

Naming, strictly (see spec §0):
    Stage  -- Richardson's physical label, read from info.fits, never
              recomputed here.
    Label  -- this module's derived output: `label_probs` (the 5-tuple
              distribution) and `label` (its argmax).

Metric: a FIXED per-band sigma (`FIXED_SIGMA_BY_BAND`), derived from the
survey's own per-band fractional flux error (dex = log10(e) * fracerr) --
the same normalization philosophy as the sedfitter's own chi^2
(observational error, not model variability). Consequences:
    - D^2(i,j) = sum_band (mu_i,band - mu_j,band)^2 / sigma_band^2, the
      SAME sigma_band on both sides -- a plain global standardization, not
      a per-pair or per-model term.
    - So the distance is a genuine metric, exactly Euclidean L2 after
      pre-scaling each band by 1/sigma_band. FAISS (IndexIVFFlat,
      approximate) indexes it directly: no packed sigma vector, no custom
      njit distance function.
    - Nothing is ever imputed -- every row, every band, uses the same
      constant.
(Rejected alternatives: docs/yso_decisions.md D-8.)

Pipeline: Stage (read) -> carve TD per-aperture, marginalize (level 1)
          -> soften over the SED neighborhood (level 2, fixed-sigma metric)
          -> label.

Typical usage
-------------
    from sesnaimpute.sed_models.curate.yso_labeling import label_geometries

    result, run_info = label_geometries(
        geometries=["s---s-i", "sp--hmi", ...],
        yso_root="/path/to/sed_models/yso",
        K=1400,
        curation_state_path="/path/to/yso_curation_state.h5",
    )

`result` is a plain DataFrame (one row per model x inclination); `run_info`
is the per-run diagnostics dict (§6 of the spec). Neither is written to the
curation state -- that is a driver's responsibility, not this module's.

The curation state is READ, for one thing only: `/registry/eligibility`, which
decides which models are labeled at all. Those gates are determined by
earlier phases (ingest, aperture) and this module trusts them rather than
re-deriving them from the FITS files -- one definition, one place.

Internal organization (dependency order):
    CONSTANTS -> IO -> FEATURES (mu only) -> TD CARVE -> LEVEL 1
    -> METRIC/ANN (fixed sigma, FAISS) -> LEVEL 2 (LABEL) -> DIAGNOSTICS
    -> DRIVER
====================================================================
"""

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except ImportError:
    fits = None

try:
    import faiss
except ImportError:
    faiss = None

from sesnaimpute.sed_models.constants import BANDS, PC_CM, VEGA_ZERO_POINT_MJY
from sesnaimpute.sed_models.curate import curve_of_growth as cog
from sesnaimpute.sed_models.curate.model_io import align_by_name, read_flux_metadata

# ==========================================================================
# CONSTANTS
# ==========================================================================

# Band order fixed throughout this module -- must match the index convention
# used to pack feature vectors (indices 0-7 = mu, pre-scaled by
# 1/FIXED_SIGMA_BY_BAND; see _pack_vectors) and FIXED_SIGMA_ARRAY.
# Derived from constants.BANDS (the single source of truth for band order)
# rather than hardcoded, so the YSO side can never silently drift out of
# sync with the SPS/fitter side.
BAND_ORDER = tuple(BANDS)

# Where each band's convolved photometry file lives, relative to a
# geometry's directory (geometry_dir/convolved/<survey_subdir>/<filename>).
_CONVOLVED_SUBDIR = {"J": "2MASS", "H": "2MASS", "Ks": "2MASS",
                     "I1": "Spitzer", "I2": "Spitzer", "I3": "Spitzer",
                     "I4": "Spitzer", "M1": "Spitzer"}
_CONVOLVED_FILENAME = {"J": "2MASS.J.fits", "H": "2MASS.H.fits", "Ks": "2MASS.Ks.fits",
                       "I1": "IRAC.I1.fits", "I2": "IRAC.I2.fits", "I3": "IRAC.I3.fits",
                       "I4": "IRAC.I4.fits", "M1": "MIPS.24mu.fits"}

# Vega zero points, Jy -- read from constants.VEGA_ZERO_POINT_MJY (their
# single home) and converted from this project's mJy convention. Needed
# only for the Gutermuth TD color cuts (I1-I4, M1).
VEGA_F0_JY = {b: VEGA_ZERO_POINT_MJY[b] / 1000.0 for b in ("I1", "I2", "I3", "I4", "M1")}

# Richardson's own sentinel for "no classifiable Stage" in info.fits' Stage
# column -- confirmed empirically (spec §3.2): distinct values across the
# grid are exactly {-1, 0, 1, 2, 3}, with -1 concentrated in the eight
# pure-envelope (no-disk-column) geometries, ~9.3% of valid rows.
STAGE_UNCLASSIFIED = -1
STAGE_VALUES = (0, 1, 2, 3)

# Per-MODEL flux floor (clamp), spec §4.1: floor_i = peak_i / 10**FLOOR_DEX_
# BELOW_PEAK, where peak_i is the max raw flux row i reaches over its OWN
# full cube -- all 8 bands x all 20 apertures (not just the survey-matched
# aperture: sigma uses all 20, so all 20 need protecting from underflow, or
# sigma gets corrupted by whichever apertures the floor missed). Every one
# of row i's flux values is clamped to max(flux, 10**floor_dex_i) before
# log10 -- this replaces masking entirely: a band is never excluded from
# the distance anymore, a catastrophically-extincted band just reads as
# "at this model's own floor," a bounded, consistent value, rather than
# NaN (excluded) or a numerically-meaningless extreme (some raw values
# underflow to the float32-subnormal limit, ~1.4e-45, or land anywhere in
# a continuous chain up to that -- confirmed empirically, no natural gap
# between "artifact" and "real flux" exists, so this is a deliberate
# clamp, not a detected threshold).
#
# PER-MODEL, not grid-wide: each row's floor depends only on its own SED,
# so it is never coupled to grid composition. This is also the physically
# correct notion of "dark" -- a band is negligible if it contributes
# nothing to THIS object's own SED, not relative to whatever the single
# brightest model elsewhere in the grid happens to be. (A grid-wide
# per-band floor was rejected: docs/yso_decisions.md D-9.)
# Owned by curve_of_growth, which also carries the justification for its
# depth. Re-exported here because yso_fps and yso_curate already read it off
# this module and both record it in run metadata.
FLOOR_DEX_BELOW_PEAK = cog.FLOOR_DEX_BELOW_PEAK

# Gutermuth (2009) Appendix A.3 TD condition, Vega colors, contamination
# gates dropped (meaningless for noise-free models; spec §3.3).
TD_PHOT_I2_MINUS_I4_MAX = 0.5
TD_PHOT_I1_MINUS_I3_MAX = 0.35
TD_EXCESS_I3_MINUS_M1_MIN = 2.5
TD_EXCESS_I2_MINUS_M1_MIN = 2.5

# Per-band survey photometric precision -- the MEASURED form: the median
# fractional flux uncertainty over the SESNA catalog's stellar/YSO-labeled
# sources, one value per band (labeling spec 4.2). This is a YSO-curation
# calibration input, not a property of the band, so it lives here rather
# than in constants.BANDS: nothing outside this pipeline consumes it, and
# its provenance is a labeled subset of the catalog rather than the survey
# at large.
#
# Two consumers, two representations, ONE stored number: labeling wants
# dex (below), dedup wants the fractional form for the sedfitter's `error`
# array (yso_dedup.FRACERR imports FRACERR_ARRAY directly). Storing the
# measurement and deriving the dex form keeps the two exact reciprocals of
# each other (docs/yso_decisions.md D-7).
FRACERR_BY_BAND = {
    "J": 0.0347, "H": 0.0329, "Ks": 0.0388,
    "I1": 0.0326, "I2": 0.0368, "I3": 0.0571, "I4": 0.0785,
    "M1": 0.0644,
}
FRACERR_ARRAY = np.array([FRACERR_BY_BAND[b] for b in BAND_ORDER])

# FIXED per-band sigma (log-flux/dex units). A fractional flux error is
# ~additive in log10, so dex = log10(e) * fractional error -- the same
# normalization philosophy as the sedfitter's own chi^2 (observational
# error, not model variability). Nothing is ever imputed: every row, every
# band, uses this same constant. (docs/yso_decisions.md D-8.)
LOG10_E = float(np.log10(np.e))
FIXED_SIGMA_BY_BAND = {b: LOG10_E * f for b, f in FRACERR_BY_BAND.items()}
FIXED_SIGMA_ARRAY = LOG10_E * FRACERR_ARRAY


def _log(msg, verbose=True):
    """Progress print (module has no logging framework dependency; the
    driver's runs are long enough at production scale -- potentially
    ~5 min at grid scale -- progress output still makes a stall visible, see the
    build_grid/build_ann_index/compute_label_chunked call sites)."""
    if verbose:
        print(f"[yso_labeling] {msg}", flush=True)


# ==========================================================================
# IO
# ==========================================================================

def _geometry_paths(geometry_dir):
    return dict(
        info=os.path.join(geometry_dir, "info.fits"),
        parameters=os.path.join(geometry_dir, "parameters.fits"),
        flux=os.path.join(geometry_dir, "flux.fits"),
    )


def _read_stage_and_validity(geometry_dir, ref_names):
    """Stage (info.fits, aligned to ref_names) and star.radius>0 validity
    (parameters.fits, aligned to ref_names)."""
    if fits is None:
        raise RuntimeError("astropy is required to read FITS files.")
    with fits.open(_geometry_paths(geometry_dir)["info"], memmap=True) as h:
        info_names = np.asarray(h[1].data["Model Name"]).astype(str)
        stage = np.asarray(h[1].data["Stage"], dtype=int)
    order = align_by_name(ref_names, info_names)
    stage = stage[order]

    with fits.open(_geometry_paths(geometry_dir)["parameters"], memmap=True) as h:
        p_names = np.asarray(h[1].data["MODEL_NAME"]).astype(str)
        radius = np.asarray(h[1].data["star.radius"], dtype=float)
    order = align_by_name(ref_names, p_names)
    valid = radius[order] > 0
    return stage, valid


def _read_band_flux(geometry_dir, band, ref_names):
    """(n_rows, 20) mJy TOTAL_FLUX for one band, aligned to ref_names."""
    if fits is None:
        raise RuntimeError("astropy is required to read FITS files.")
    path = os.path.join(geometry_dir, "convolved", _CONVOLVED_SUBDIR[band],
                        _CONVOLVED_FILENAME[band])
    with fits.open(path, memmap=True) as h:
        names = np.asarray(h[1].data["MODEL_NAME"]).astype(str)
        flux = np.asarray(h[1].data["TOTAL_FLUX"], dtype=np.float64)
    order = align_by_name(ref_names, names)
    return flux[order]


# ==========================================================================
# FEATURES (mu only -- sigma is FIXED_SIGMA_BY_BAND, a constant) -- spec 4.1
# ==========================================================================

def read_geometry_flux_cubes(geometry_dir, ref_names):
    """Read all 8 bands' raw (n,20) flux cubes for one geometry -- the
    single disk-read pass per geometry. Flux is physically non-negative;
    a handful of raw values across the grid are tiny negative floating-
    point artifacts (confirmed: ~1e-11 to 1e-10 mJy magnitude, ~40 orders
    below any real flux -- clearly noise, not signal). Clamped to 0 here,
    the single point flux enters this module, so peak_row (see
    compute_row_features_from_cubes) is never spuriously dragged negative
    by one noisy aperture and no downstream step needs to special-case it.
    Returns {band: flux_20ap}."""
    return {band: np.maximum(_read_band_flux(geometry_dir, band, ref_names), 0.0)
           for band in BAND_ORDER}


def compute_row_features_from_cubes(flux_cubes, aperture_au, distance_cm):
    """mu (n,8) in log10(mJy) ONLY -- this module uses FIXED_SIGMA_BY_BAND (a
    constant), not a per-model aperture-variance sigma, so there is no sigma
    to compute, no degeneracy to detect, and no imputation of any kind. From
    an ALREADY-READ flux_cubes dict (see read_geometry_flux_cubes -- this
    function does no I/O).

    The SED itself is `curve_of_growth.survey_matched_sed`: linear-in-AU
    interpolation matching what the fitter will do, with the per-model flux
    floor applied AFTER interpolating. This module owns what mu is FOR;
    `curve_of_growth` owns how it is built, and `sed_models_register` derives the
    same quantity (`f_ref`) from the same function.

    Returns (mu, clamp_report, floor_dex_row, all_zero_flux). `clamp_report`
    = {band: {n_clamped, n_total}} counts raw cube values below the floor --
    the labeling spec's per-band clamped-fraction diagnostic.
    `all_zero_flux` is (n,) bool where peak_i == 0 (build_grid drops those at
    seed time).
    """
    sed_linear, floor_linear_row, all_zero_flux = cog.survey_matched_sed(
        flux_cubes, aperture_au, distance_cm, BAND_ORDER)

    clamp_report = {}
    for band in BAND_ORDER:
        flux_n_ap = flux_cubes[band]
        clamp_report[band] = dict(
            n_clamped=int((flux_n_ap < floor_linear_row[:, None]).sum()),
            n_total=int(flux_n_ap.size))

    # log10(0) = -inf only on all-zero rows, which are dropped at seed time.
    with np.errstate(divide="ignore"):
        mu = np.log10(sed_linear)
        floor_dex_row = np.log10(floor_linear_row)
    return mu, clamp_report, floor_dex_row, all_zero_flux


# ==========================================================================
# TD CARVE (Gutermuth, per-aperture, marginalized) -- spec 3.3
# ==========================================================================

def _vega_mag(flux_mjy, f0_jy):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux_mjy > 0, -2.5 * np.log10(flux_mjy / (f0_jy * 1000.0)), np.nan)


def compute_td_fraction(flux_cubes):
    """Per-row TD fraction: Gutermuth flag evaluated per (row, aperture),
    marginalized (equal weight) over the row's USABLE apertures only.
    Apertures missing a required band are excluded from BOTH the
    numerator and the denominator (spec 3.3: "excluded from that row's
    marginalization"; docs/yso_decisions.md D-5).
    Rows with zero usable apertures get td_frac=0: no TD carve can ever
    be evaluated for them, so the full level-1 mass correctly stays on
    the row's own Richardson Stage (level1_labels needs no special-casing
    as a result, since td_frac is always finite here).

    Returns (td_frac, n_usable): n_usable is (n,) int, the per-row usable-
    aperture count (0..20) -- a diagnostic persisted by load_geometry_table
    for build_grid's audit report (how many rows hit the zero-usable
    path)."""
    mag = {b: _vega_mag(flux_cubes[b], VEGA_F0_JY[b]) for b in ("I1", "I2", "I3", "I4", "M1")}
    photospheric = ((mag["I2"] - mag["I4"]) < TD_PHOT_I2_MINUS_I4_MAX) & \
                   ((mag["I1"] - mag["I3"]) < TD_PHOT_I1_MINUS_I3_MAX)
    excess = ((mag["I3"] - mag["M1"]) > TD_EXCESS_I3_MINUS_M1_MIN) | \
             ((mag["I2"] - mag["M1"]) > TD_EXCESS_I2_MINUS_M1_MIN)
    td_flag_ap = photospheric & excess  # (n, 20) bool, NaN-safe (comparisons -> False)

    usable_ap = ~(np.isnan(mag["I1"]) | np.isnan(mag["I2"]) | np.isnan(mag["I3"]) |
                 np.isnan(mag["I4"]) | np.isnan(mag["M1"]))          # (n, 20) bool
    n_usable = usable_ap.sum(axis=1)                                  # (n,) int, 0..20

    with np.errstate(invalid="ignore", divide="ignore"):
        td_frac = td_flag_ap.sum(axis=1) / n_usable
    td_frac = np.where(n_usable > 0, td_frac, 0.0)
    return td_frac, n_usable


# ==========================================================================
# LEVEL 1 (Stage + TD carve -> 5-tuple per row) -- spec 3.3
# ==========================================================================

CLASS_NAMES = ("0", "I", "II", "III", "TD")


def level1_labels(stage, td_frac):
    """(n,5) level-1 distribution per row: one-hot on Stage for 0/I; Stage
    II AND Stage III both split into {Stage: 1-f, TD: f} (spec 3.3 -- TD is
    carved from either Stage, a single shared TD column; Stage III is
    itself the diskless Stage, not "disk-bearing")."""
    n = len(stage)
    L = np.zeros((n, 5))
    L[stage == 0, 0] = 1.0
    L[stage == 1, 1] = 1.0
    m2 = stage == 2
    L[m2, 2] = 1.0 - td_frac[m2]
    L[m2, 4] = td_frac[m2]
    m3 = stage == 3
    L[m3, 3] = 1.0 - td_frac[m3]
    L[m3, 4] = td_frac[m3]
    return L


# ==========================================================================
# METRIC / ANN -- FIXED-SIGMA experiment (see module docstring)
# ==========================================================================
# Packing: vector[0:8] = mu / FIXED_SIGMA_BY_BAND, pre-scaled per band so a
# PLAIN L2 distance on these vectors equals exactly
# sum_band (mu_i,band - mu_j,band)^2 / sigma_band^2 -- the SAME sigma_band
# on both sides (a fixed constant, not a per-pair, per-model term). No
# sigma half packed anymore (8-dim, not 16) -- there is no per-model sigma
# to carry alongside mu. This is exactly why FAISS (a real metric) can
# replace pynndescent's custom-metric NNDescent here: with the per-pair
# denominator gone, the distance is a genuine Euclidean metric after
# pre-scaling, not a non-metric custom function -- so a standard
# approximate FAISS index (IndexIVFFlat, below) is a valid drop-in, not
# just an exact one.


def _pack_vectors(mu):
    """(n,8) mu pre-scaled by 1/FIXED_SIGMA_BY_BAND -- plain L2 on these
    vectors equals the fixed-sigma D^2 exactly, applied identically to
    both the indexed data and the query vectors (see build_ann_index /
    query_neighbors)."""
    return np.ascontiguousarray(mu / FIXED_SIGMA_ARRAY[None, :], dtype=np.float32)


# Default n_jobs: all cores but one, so the machine stays responsive during
# a long run. Applied to FAISS via omp_set_num_threads.
DEFAULT_N_JOBS = max(1, (os.cpu_count() or 2) - 1)

# IndexIVFFlat parameters. This K (~sqrt(N), ~1,400 at production scale) is
# unusually large for an ANN index -- most IVF/HNSW tuning guidance targets
# K of 10-100, where the whole point is to touch only a tiny fraction of
# the data. At K~1,400 you need enough real candidates gathered from
# whichever cells get probed to fill K well, so nprobe has to be a
# non-trivial fraction of nlist regardless of index type; IVF's cost model
# stays linear and predictable as nprobe grows (unlike HNSW, whose
# efSearch would need to be pushed to several times K to keep recall up at
# this scale, eroding most of its usual advantage) -- see the fixed-sigma
# handoff for the reasoning that led here. IVF's own coarse quantizer
# (k-means) also naturally puts more, smaller cells where the grid is
# locally dense, which lines up with this pipeline's own confirmed finding
# that local SED-space density varies enormously (the ESS/degenerate-
# region behavior all the earlier diagnostics were built around).
#
# nlist ~ 4*sqrt(N) is the standard FAISS rule-of-thumb starting point.
# nprobe is deliberately generous (not a minimal/fastest-possible value)
# given this step doesn't need to be exact, just close -- cheap to lower
# later if a production run shows it's still far faster than needed.
IVF_NLIST_PER_SQRT_N = 4
DEFAULT_NPROBE = 32


def build_ann_index(vectors, n_jobs=DEFAULT_N_JOBS, nlist=None, nprobe=DEFAULT_NPROBE, verbose=True):
    """FAISS IndexIVFFlat (approximate: exact L2 distance to whichever
    candidates a probed cell contains, but not every cell is probed, so
    recall -- not distance accuracy -- is what's approximate) over the
    pre-scaled 8-dim vectors. Replaces IndexFlatL2 (exact brute force):
    with sigma now fixed (not per-pair), the distance is a genuine
    Euclidean metric after pre-scaling, so this is a valid drop-in without
    needing pynndescent's custom-metric/RP-tree machinery either way --
    IVF is chosen over HNSW specifically because K is unusually large here
    (see the module notes above `IVF_NLIST_PER_SQRT_N`).

    `nlist` defaults to IVF_NLIST_PER_SQRT_N * sqrt(n_rows) (FAISS's own
    rule-of-thumb starting point) if not given. The coarse quantizer is a
    plain IndexFlatL2 in the same 8-dim space; training runs on the full
    `vectors` array (n_rows is already grid-scale, well within FAISS's
    "no more than ~200*nlist training points needed" guidance, so no
    separate subsampling step is added). `index.nprobe` is set once here,
    not per-query -- see query_neighbors."""
    if faiss is None:
        raise RuntimeError("faiss is required for the fixed-sigma label neighborhood search.")
    faiss.omp_set_num_threads(n_jobs)
    n_rows, dim = vectors.shape
    if nlist is None:
        nlist = int(IVF_NLIST_PER_SQRT_N * np.sqrt(n_rows))
    _log(f"building FAISS IndexIVFFlat: {n_rows} rows, dim={dim}, nlist={nlist}, nprobe={nprobe} ...",
        verbose)
    t0 = time.time()
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    _log(f"index built in {time.time()-t0:.1f}s", verbose)
    return index


def query_neighbors(index, vectors, K):
    """Retrieve K candidates per row (self included, excluded downstream by
    exact identity). Returns (idx_graph, dist_graph) -- FAISS's own
    .search() returns (distances, indices); swapped here so downstream code
    (compute_label_chunked, _ensure_self_included, etc) has one
    consistent convention regardless of which ANN library is underneath.
    dist_graph is SQUARED L2, which equals D^2 exactly given the
    pre-scaling in _pack_vectors, for whichever candidates were found --
    IndexIVFFlat's approximation is in which candidates get considered
    (only cells within `index.nprobe` of the query, set once in
    build_ann_index), not in the distance values themselves."""
    dist_graph, idx_graph = index.search(np.ascontiguousarray(vectors, dtype=np.float32), K)
    return idx_graph, dist_graph.astype(np.float64)


# ==========================================================================
# LEVEL 2 (LABEL) -- spec 4.4
# ==========================================================================

@dataclass
class LabelResult:
    label_probs: np.ndarray        # (n,5), columns = CLASS_NAMES order
    label: np.ndarray              # (n,) str, argmax of label_probs
    ess: np.ndarray                # (n,) effective sample size = 1/sum(p_j^2)
    n_eff_neighbors: np.ndarray    # (n,) count of neighbors with w>0 (self excluded)
    total_weight: np.ndarray       # (n,) sum of raw (unnormalized) weight


KERNEL_HALF_WIDTH_FACTOR = 2.0   # the Gaussian kernel's own factor (exp(-D^2/2)), not a tunable knob


def _kernel_weights(dist_sq):
    """The Gaussian neighborhood kernel, `exp(-D^2/2)` (spec §4.4) -- one
    shared definition, used by both the production path
    (compute_label/compute_label_chunked) and localization_check, the
    diagnostic that validates it. A diagnostic silently drifting from the
    code it validates is the failure mode a single definition removes."""
    return np.exp(-dist_sq / KERNEL_HALF_WIDTH_FACTOR)


def _ensure_self_included(idx_graph, dist_graph, row_ids):
    """Guarantee each row's own index appears in its candidate set exactly
    once, with D^2=0 (weight=1, the global max since D^2>=0 always) -- the
    model's own level-1 label is meant to speak loudest (spec 4.3, updated:
    self is now INCLUDED, not excluded). The approximate ANN search finds
    self in its own top-K most but not all of the time (~95% measured on a
    spot check) -- rather than rely on that, neutralize any self entry the
    search happened to return (avoids double-counting) and append one clean
    guaranteed self column. row_ids: (m,) global row indices for this batch
    (idx_graph/dist_graph are (m,K))."""
    row_ids = np.asarray(row_ids)
    self_mask = idx_graph == row_ids[:, None]
    dist_graph = np.where(self_mask, np.inf, dist_graph)
    idx_graph = np.concatenate([idx_graph, row_ids[:, None]], axis=1)
    dist_graph = np.concatenate([dist_graph, np.zeros((len(row_ids), 1), dtype=dist_graph.dtype)], axis=1)
    return idx_graph, dist_graph


def compute_label(idx_graph, dist_graph, level1, row_ids=None):
    """Level-2 softening from an ALREADY-RETRIEVED (n,K) neighbor graph:
    each row's label_probs is the weight-averaged level-1 distribution of its
    neighbors, self INCLUDED (guaranteed via _ensure_self_included -- see
    its docstring). row_ids defaults to arange(n), i.e. idx_graph's own row
    order; pass explicitly if idx_graph came from querying a subset (e.g.
    localization_check). Only safe to call directly on graphs already known
    to be small (diagnostics on a handful of rows) -- the full-grid driver
    uses compute_label_chunked instead (see its docstring for why a
    single (N,K) graph does not fit in memory at production scale)."""
    n, K = idx_graph.shape
    if row_ids is None:
        row_ids = np.arange(n)
    idx_graph, dist_graph = _ensure_self_included(idx_graph, dist_graph, row_ids)

    weight = _kernel_weights(dist_graph)

    wsum = weight.sum(axis=1)
    n_eff = (weight > 0).sum(axis=1)
    safe_wsum = np.where(wsum > 0, wsum, 1.0)
    w_norm = weight / safe_wsum[:, None]
    ess = np.where(wsum > 0, 1.0 / np.sum(w_norm ** 2, axis=1), 0.0)

    L_neighbors = level1[idx_graph]                        # (n,K,5)
    label_probs = (w_norm[:, :, None] * L_neighbors).sum(axis=1)   # (n,5)

    argmax = label_probs.argmax(axis=1)
    labels = np.array(CLASS_NAMES)[argmax]
    return LabelResult(label_probs=label_probs, label=labels,
                        ess=ess, n_eff_neighbors=n_eff, total_weight=wsum)


# Rows per query/aggregate batch for the full-grid driver. Bounds memory to
# O(chunk_size*K) rather than O(N*K) (the idx/dist graphs) or O(N*K*5) (the
# level1-gather tensor inside compute_label) -- both infeasible at
# production scale. At N~2e6, K~1400: a single whole-grid query would need
# ~11GB each for idx_graph/dist_graph, and compute_label's L_neighbors =
# level1[idx_graph] would be (2e6, 1400, 5) -- ~56-112GB depending on dtype.
# At chunk_size=5000, K=1400 the per-chunk arrays are ~100-300MB, matching
# the chunked pattern already verified safe earlier in this project (the
# FAISS-based prototype's near-OOM was exactly this class of mistake, a
# (batch,N)-scale array instead of a bounded chunk).
DEFAULT_CHUNK_SIZE = 5000


def compute_label_chunked(index, vectors, level1, K, chunk_size=DEFAULT_CHUNK_SIZE,
                              verbose=True, progress_every=20, query_row_ids=None):
    """Query + level-2 aggregate in row-chunks (see DEFAULT_CHUNK_SIZE for
    why). Sequential over chunks -- each chunk's own query() call is
    already internally multi-threaded (parallel_batch_queries=True on the
    index), so an additional outer thread pool here would oversubscribe
    cores rather than help. Returns the same LabelResult
    compute_label would from one big (infeasible) query.

    vectors is the QUERY set -- it need not be the same data the index was
    built from. When it's a SUBSET of the index's data (e.g. a stratified
    sample queried against a full-grid index), pass query_row_ids: the true
    global row index (matching level1's/the index's own row order) for each
    row of `vectors`, so self-inclusion and the level1 gather reference the
    right rows. Defaults to arange(n) (vectors IS the index's own full data,
    in its own order) when not given.

    No mid-run checkpointing: the loop always starts this pass at row 0
    (`range(0, n, chunk_size)`) and there is no resume branch, because the
    pipeline is re-run wholesale when needed rather than
    built out into real resumability nothing here currently needs."""
    n = vectors.shape[0]
    if query_row_ids is None:
        query_row_ids = np.arange(n)
    label_probs = np.zeros((n, 5), dtype=np.float64)
    ess = np.zeros(n, dtype=np.float64)
    n_eff = np.zeros(n, dtype=np.int64)
    total_weight = np.zeros(n, dtype=np.float64)

    n_chunks = (n + chunk_size - 1) // chunk_size
    t_start = time.time()
    for ci, lo in enumerate(range(0, n, chunk_size)):
        hi = min(lo + chunk_size, n)
        idx_graph, dist_graph = query_neighbors(index, vectors[lo:hi], K)
        row_ids = query_row_ids[lo:hi]
        idx_graph, dist_graph = _ensure_self_included(idx_graph, dist_graph, row_ids)

        weight = _kernel_weights(dist_graph)

        wsum = weight.sum(axis=1)
        n_eff[lo:hi] = (weight > 0).sum(axis=1)
        total_weight[lo:hi] = wsum
        safe_wsum = np.where(wsum > 0, wsum, 1.0)
        w_norm = weight / safe_wsum[:, None]
        ess[lo:hi] = np.where(wsum > 0, 1.0 / np.sum(w_norm ** 2, axis=1), 0.0)

        if verbose and ((ci + 1) % progress_every == 0 or ci + 1 == n_chunks):
            elapsed = time.time() - t_start
            rate = hi / max(elapsed, 1e-6)
            eta_s = (n - hi) / max(rate, 1e-6)
            _log(f"labeling: {hi}/{n} rows ({ci+1}/{n_chunks} chunks), "
                f"{rate:.0f} rows/s, elapsed {elapsed/60:.1f} min, ETA {eta_s/60:.1f} min",
                verbose)

        L_neighbors = level1[idx_graph]                     # (chunk,K,5) -- bounded
        label_probs[lo:hi] = (w_norm[:, :, None] * L_neighbors).sum(axis=1)

        del idx_graph, dist_graph, weight, w_norm, L_neighbors

    argmax = label_probs.argmax(axis=1)
    labels = np.array(CLASS_NAMES)[argmax]
    return LabelResult(label_probs=label_probs, label=labels,
                        ess=ess, n_eff_neighbors=n_eff, total_weight=total_weight)


# ==========================================================================
# DIAGNOSTICS -- spec 5
# ==========================================================================

def localization_check(index, vectors, K, row_indices, labels=None):
    """ESS + retained-weight check (spec §5) on a handful of representative
    rows, via query_neighbors on the already-built index (only the
    rows-of-interest need dense inspection, though the query covers the
    full grid so real neighbors are found)."""
    if labels is None:
        labels = [str(i) for i in row_indices]
    row_ids = np.asarray(list(row_indices))
    query_idx, query_dist = query_neighbors(index, vectors[row_ids], K)
    query_idx, query_dist = _ensure_self_included(query_idx, query_dist, row_ids)
    out = {}
    for row_pos, (qi, label) in enumerate(zip(row_indices, labels)):
        row_idx = query_idx[row_pos]
        row_dist = query_dist[row_pos]
        w = _kernel_weights(row_dist)
        wsum = w.sum()
        p = w / wsum if wsum > 0 else w
        ess = 1.0 / np.sum(p ** 2) if wsum > 0 else np.nan
        w_sorted = np.sort(w)[::-1]
        cumw = np.cumsum(w_sorted)
        tail_share = (cumw[-1] - cumw[int(0.9 * len(cumw))]) / cumw[-1] if cumw[-1] > 0 else np.nan
        out[label] = dict(ess=ess, total_weight=wsum, tail_share_last10pct=tail_share)
    return out


def _default_localization_rows(df, n_per_group=2, seed=0):
    """Stratified-by-(geometry, Stage) sample of representative row
    indices for `localization_check` -- the production driver used to
    call `label_geometries` with `localization_rows=None`, which skips
    the check entirely (spec §5 shipped with a `null` entry). Labels are
    "{geometry}:stage={stage}:{i}" so the run_info dict stays
    human-readable. Kept small (n_per_group per combination, ~a few
    hundred rows total across the grid) -- this is a spot-check, not a
    census."""
    rng = np.random.default_rng(seed)
    row_indices, labels = {}, []
    for (geometry, stage), group in df.groupby(["geometry", "stage"], sort=False):
        idx = group.index.to_numpy()
        chosen = rng.choice(idx, size=min(n_per_group, len(idx)), replace=False)
        for i, row_idx in enumerate(chosen):
            row_indices[f"{geometry}:stage={stage}:{i}"] = int(row_idx)
    return row_indices


def td_locus(df):
    """Per-geometry mean p_stage_TD (level-2, post-softening label TD
    mass), plus the level-1 III->TD carve rate (spec §3.3's validation
    note): the mean aperture-marginalized td_frac among HARD Stage III
    rows specifically -- if ~0, Stage III is effectively diskless (no 24um
    excess -> never flagged) and the II+III carve reduces to a II-only
    carve in practice; a materially non-zero value is the expected/handled
    case, a surprisingly large one suggests II/III mislabel leakage. This
    is a LEVEL-1 (pre-neighborhood) rate, computed directly from `stage`
    and `td_frac` -- deliberately not the level-2 p_stage_TD, so it
    isolates the carve itself from the neighborhood softening. Also
    reports the Stage-II rate for comparison."""
    stage3 = df["stage"] == 3
    stage2 = df["stage"] == 2
    return {
        "p_stage_TD_by_geometry": df.groupby("geometry")["p_stage_TD"].mean()
                                    .sort_values(ascending=False).to_dict(),
        "stage_iii_td_rate": float(df.loc[stage3, "td_frac"].mean()) if stage3.any() else float("nan"),
        "stage_ii_td_rate": float(df.loc[stage2, "td_frac"].mean()) if stage2.any() else float("nan"),
    }


def argmax_vs_stage(df):
    hard_name = np.array(CLASS_NAMES)[np.clip(df["stage"].to_numpy(), 0, 3)]
    match = df["label"].to_numpy() == hard_name
    match_or_td = match | ((df["label"] == "TD") & (df["stage"] == 2)) | \
                  ((df["label"] == "TD") & (df["stage"] == 3))
    return dict(n=len(df), exact_match_rate=float(match.mean()),
               td_as_origin_stage_match_rate=float(match_or_td.mean()))


# ==========================================================================
# DRIVER
# ==========================================================================

def load_geometry_table(geometry, yso_root):
    """One geometry's full row table: Stage, validity, mu feature array
    (no sigma -- this module uses FIXED_SIGMA_BY_BAND, a constant), TD
    fraction, model_name, per-row floor_dex (diagnostic). Floor is computed
    entirely from this geometry's own rows (spec §4.1 -- see
    compute_row_features_from_cubes), so this is a single disk-read pass.
    No filtering applied here (unclassified-Stage filtering is a grid-wide
    step, driver-level)."""
    geometry_dir = os.path.join(yso_root, geometry)
    meta = read_flux_metadata(_geometry_paths(geometry_dir)["flux"])
    if meta.distance_cm is None:
        raise ValueError(f"{geometry}: flux.fits PRIMARY header has no DISTANCE keyword.")

    stage, valid_row = _read_stage_and_validity(geometry_dir, meta.names)
    flux_cubes = read_geometry_flux_cubes(geometry_dir, meta.names)
    mu, clamp_report, floor_dex_row, all_zero_flux = \
        compute_row_features_from_cubes(flux_cubes, meta.apertures_au, meta.distance_cm)
    td_frac, td_n_usable_ap = compute_td_fraction(flux_cubes)

    # model_id is the curation state's key -- <geometry>:<row_index>, row_index
    # being the position in this geometry's own files, which is the order
    # `meta.names` is read in. Carried here so the eligibility filter can
    # be applied by key rather than by re-deriving gates from these columns.
    model_id = np.char.add(f"{geometry}:", np.arange(len(meta.names)).astype(str))

    df = pd.DataFrame({
        "model_id": model_id,
        "geometry": geometry, "model_name": meta.names,
        "valid": valid_row, "stage": stage, "td_frac": td_frac,
        "td_n_usable_ap": td_n_usable_ap,
        "floor_dex": floor_dex_row, "all_zero_flux": all_zero_flux,
    })
    return df, mu, clamp_report


def eligible_model_ids(curation_state_path):
    """The model_ids labeling may process: everything not ALREADY ruled out
    by a determined gate, read from `/registry/eligibility`.

    Uses `reg.compute_eligible` -- the curation state's single definition of
    eligibility -- and keeps `!= 0`, i.e. "not definitively ineligible".
    At labeling time gate 4 (`valid_nondup`) is still undetermined, since
    dedup's candidate pool is defined by labeling's own output; Kleene AND
    reports those rows as `-1`, not `0`, so they are correctly kept. This
    works unchanged whichever gates happen to have run.

    Imported inside the function because `yso_curation_state` imports BAND_ORDER
    and STAGE_UNCLASSIFIED from this module -- a module-level import here
    would close the cycle. The clean fix is to move those two constants to
    `constants.py`, which is deliberately left alone for now.
    """
    from sesnaimpute.sed_models.curate import yso_curation_state as reg

    eligibility = reg.read_dataset(str(curation_state_path), "eligibility")
    keep = reg.compute_eligible(eligibility) != 0
    return eligibility.loc[keep, "model_id"].to_numpy()


def build_grid(geometries, yso_root, eligible_ids, verbose=True):
    """Single-pass grid build: per geometry, read raw flux once and compute
    mu/TD-fraction using that geometry's own rows' per-model floor (spec
    §4.1, unchanged -- floor_i depends only on row i's own flux cube; see
    FLOOR_DEX_BELOW_PEAK). No sigma computation, no imputation of any kind
    -- this variant's normalizer (FIXED_SIGMA_BY_BAND) is a constant, not
    derived from the grid. Then concatenate and keep the rows in
    `eligible_ids`. Returns (df, mu, seed_report).

    **The filter is membership in `eligible_ids`, not a locally recomputed
    one.** This function used to re-derive `star.radius > 0`, `Stage != -1`
    and `peak_i > 0` from the raw files -- a second implementation of rules
    the curation state already stores as `valid_star`/`valid_stage`/
    `valid_aperture`, with nothing forcing the two to agree. The gates now
    have exactly one home (`yso_curation_state.GATE_COLUMNS`), and this reads it.

    That also means the aperture gate is honoured HERE, before the ANN
    index is built. A model whose survey aperture the radiative transfer
    never resolved has a floor-dominated, fabricated SED, and labeling it
    yields a meaningless label -- so it is excluded rather than labeled
    and discarded later. Note this is about the fabricated model's OWN
    label: measurement shows such models do not detectably shift their
    neighbours' labels (see yso_aperture's module docstring).
    """
    eligible_ids = set(np.asarray(eligible_ids).tolist())
    dfs, mus = [], []
    clamp_totals = {b: dict(n_clamped=0, n_total=0) for b in BAND_ORDER}
    for gi, geom in enumerate(geometries):
        t0 = time.time()
        df, mu, clamp_report = load_geometry_table(geom, yso_root)
        _log(f"loaded {geom} ({gi+1}/{len(geometries)}): {len(df)} rows in {time.time()-t0:.1f}s", verbose)
        dfs.append(df)
        mus.append(mu)
        for b in BAND_ORDER:
            clamp_totals[b]["n_clamped"] += clamp_report[b]["n_clamped"]
            clamp_totals[b]["n_total"] += clamp_report[b]["n_total"]

    df = pd.concat(dfs, ignore_index=True)
    mu = np.concatenate(mus, axis=0)

    keep = df["model_id"].isin(eligible_ids).to_numpy()

    # Cross-check, not a filter: the aperture gate should already have
    # excluded every all-zero-flux row (peak_i <= 0 means every aperture is
    # undefined, so d_min is inf). If one survives into the kept set the
    # gates disagree with the flux files and the run must stop -- silently
    # labeling a model with no SED is exactly what this pipeline exists to
    # prevent.
    leaked = int((keep & df["all_zero_flux"].to_numpy()).sum())
    if leaked:
        raise AssertionError(
            f"{leaked} all-zero-flux model(s) passed the eligibility gates -- "
            "valid_aperture should have excluded them (yso_aperture); the "
            "curation state and the flux files disagree.")

    clamped_fraction = {b: (clamp_totals[b]["n_clamped"] / clamp_totals[b]["n_total"]
                            if clamp_totals[b]["n_total"] else 0.0)
                       for b in BAND_ORDER}

    # floor_dex is per-ROW (spec §4.1), not one grid-wide number -- report
    # its distribution, over the RETAINED population (post seed-time
    # filter). A wide spread is expected; a spike concentrated at one value
    # across unrelated geometries would suggest a bug.
    floor_dex_kept = df.loc[keep, "floor_dex"]
    floor_dex_distribution = {
        "min": float(floor_dex_kept.min()),
        "median": float(floor_dex_kept.median()),
        "max": float(floor_dex_kept.max()),
    }
    floor_dex_distribution_by_geometry = {
        geom: {stat: float(val) for stat, val in stats.items()}
        for geom, stats in df.loc[keep].groupby("geometry")["floor_dex"]
                            .agg(["min", "median", "max"]).to_dict("index").items()
    }

    # TD zero-usable-aperture audit (spec 3.3's own "audit for aperture-
    # systematic dropout"), over the RETAINED population -- how many rows
    # hit compute_td_fraction's zero-usable path (td_frac forced to 0
    # rather than left NaN). Only Stage II/III rows actually feed this
    # into level1_labels' TD carve; 0/I rows never read td_frac at all.
    kept_df = df.loc[keep]
    td_zero_usable = kept_df["td_n_usable_ap"] == 0
    td_zero_usable_stage_ii_or_iii = td_zero_usable & kept_df["stage"].isin([2, 3])

    # The per-gate drop counts (unclassified, invalid star, all-zero flux)
    # are no longer reported here: those gates are determined upstream and
    # their tallies live in /metadata/ingest and /metadata/aperture. What
    # this pass owns is how many rows it was handed and what it did with
    # the flux.
    seed_report = {
        "n_grid_rows": int(len(df)),
        "n_kept": int(keep.sum()),
        "n_dropped_ineligible": int((~keep).sum()),
        "dropped_by_geometry": df.loc[~keep, "geometry"].value_counts().to_dict(),
        "clamped_fraction_by_band": clamped_fraction,
        "floor_dex_distribution": floor_dex_distribution,
        "floor_dex_distribution_by_geometry": floor_dex_distribution_by_geometry,
        "n_td_zero_usable_apertures": int(td_zero_usable.sum()),
        "n_td_zero_usable_apertures_stage_ii_or_iii": int(td_zero_usable_stage_ii_or_iii.sum()),
        "td_zero_usable_by_geometry": kept_df.loc[td_zero_usable, "geometry"]
                                       .value_counts().to_dict(),
    }

    df = df.loc[keep].reset_index(drop=True)
    mu = mu[keep]
    return df, mu, seed_report


def label_geometries(geometries, yso_root, K, curation_state_path, n_jobs=DEFAULT_N_JOBS,
                     localization_rows=None, verbose=True, output_dir=None, nlist=None,
                     nprobe=DEFAULT_NPROBE, chunk_size=DEFAULT_CHUNK_SIZE):
    """Full driver: build the gated grid, level-1 labels, ANN index, level-2
    labeling, and the spec-5 sanity checks. Returns (result_df, run_info).

    `curation_state_path` is READ (for `/registry/eligibility`, via
    eligible_model_ids) and never written -- this module returns its
    products and a driver persists them (spec 6). Reading the gates rather
    than re-deriving them is what keeps one definition of who gets labeled;
    see build_grid. verbose=True prints per-geometry
    load progress, index-build timing, and chunk-by-chunk rate/ETA for the
    labeling pass. Measured on the gated 1.64M-row pool: ~5,500 rows/s,
    so the pass is ~5 min and the FAISS index build ~4 s -- but progress
    output is still worth having, since the per-chunk rate is what makes a
    stall visible (see _log).

    `nlist`/`nprobe` (FAISS IndexIVFFlat) and `chunk_size` (the labeling
    pass's row-chunk size) are exposed here so a caller can actually
    override them without editing the module.

    result_df also carries the clamped hybrid survey-matched mu-SED, one
    column per band (`mu_J`..`mu_M1`, BAND_ORDER), in LINEAR mJy (spec
    §4.1/§6) -- the same mu the neighborhood metric operated on (already
    floored per the per-model flux floor), converted once here from the
    metric's internal log-flux units. This is a required output, not a
    diagnostic: dedup uses it directly as its query SED against the SPS
    library, so it must be the exact SED labeling used, not a separately
    recomputed raw flux. The driver persists it to its own features store
    (e.g. yso_curation_state's "mu_sed" dataset), not the curation state's
    flag datasets -- this module reads the eligibility gates but writes nothing.

    If output_dir is given, the final result + run_info are written to
    <output_dir>/labeling_result.parquet and
    <output_dir>/labeling_run_info.json."""
    eligible_ids = eligible_model_ids(curation_state_path)
    _log(f"eligibility gate: {len(eligible_ids)} model_ids admitted to labeling", verbose)
    df, mu, seed_report = build_grid(geometries, yso_root, eligible_ids, verbose=verbose)

    level1 = level1_labels(df["stage"].to_numpy(), df["td_frac"].to_numpy())
    vectors = _pack_vectors(mu)

    index = build_ann_index(vectors, n_jobs=n_jobs, nlist=nlist, nprobe=nprobe, verbose=verbose)

    # localization_rows=None (the default) wires up a stratified-by-
    # (geometry, Stage) sample; pass {} explicitly to skip, or an explicit
    # dict to override. Only an EMPTY dict/falsy-but-not-None skips.
    if localization_rows is None:
        localization_rows = _default_localization_rows(df)
    loc_report = None
    if localization_rows:
        loc_report = localization_check(index, vectors, K,
                                        list(localization_rows.values()),
                                        list(localization_rows.keys()))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result = compute_label_chunked(index, vectors, level1, K, chunk_size=chunk_size,
                                       verbose=verbose)

    # Top-ESS localization check (spec §5's "retention plateau"): the
    # rows most likely to saturate the K retrieval cap are only knowable
    # AFTER the full-grid ESS is computed, so this is a second,
    # post-hoc localization_check call on the actual top-ESS rows, not a
    # priori guessed ones. This is the check that would have caught the
    # ESS-saturation finding (44,739 rows at ESS > 0.99*K) directly.
    top_ess_idx = np.argsort(result.ess)[::-1][:10]
    top_ess_labels = {f"top_ess_{i}(row={int(idx)})": int(idx)
                      for i, idx in enumerate(top_ess_idx)}
    top_ess_report = localization_check(index, vectors, K,
                                        list(top_ess_labels.values()),
                                        list(top_ess_labels.keys()))

    out = df[["geometry", "model_name", "stage", "td_frac"]].copy()
    for i, name in enumerate(CLASS_NAMES):
        out[f"p_stage_{name}"] = result.label_probs[:, i]
    out["label"] = result.label
    out["ess"] = result.ess
    out["n_eff_neighbors"] = result.n_eff_neighbors
    out["total_weight"] = result.total_weight

    # Persisted feature output (spec §4.1/§6): the clamped hybrid
    # survey-matched mu-SED, converted from the metric's log-flux units
    # back to linear mJy -- the units the sedfitter (and dedup's SPS
    # query) actually consume. This is the SAME mu the neighborhood metric
    # operated on (already floored per §4.1), not a separately recomputed
    # raw flux -- one source of truth.
    mu_linear = 10.0 ** mu
    for bi, band in enumerate(BAND_ORDER):
        out[f"mu_{band}"] = mu_linear[:, bi]

    run_info = {
        "K": K,
        "survey_apertures_arcsec": {b: BANDS[b].aperture_arcsec for b in BAND_ORDER},
        "seed_filter": seed_report,
        "localization_check": loc_report,
        "localization_check_top_ess": top_ess_report,
        "td_locus": td_locus(out),
        "argmax_vs_stage": argmax_vs_stage(out),
        "ess_distribution": {
            "median": float(np.median(result.ess)),
            "p10": float(np.percentile(result.ess, 10)),
            "p90": float(np.percentile(result.ess, 90)),
            "max": float(result.ess.max()),
        },
        "label_sum_check": {
            "min": float(result.label_probs.sum(axis=1).min()),
            "max": float(result.label_probs.sum(axis=1).max()),
        },
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, "labeling_result.parquet")
        run_info_path = os.path.join(output_dir, "labeling_run_info.json")
        out.to_parquet(result_path)
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=2, default=str)
        _log(f"saved result to {result_path} and run_info to {run_info_path}", verbose)

    return out, run_info
