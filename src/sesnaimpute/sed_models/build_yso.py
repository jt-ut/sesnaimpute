"""
build_yso.py
====================================================================
ONE command, the owner's global-sampling ruling (CURATION_2026-10-09.md
section 2), entirely in memory: no `yso_curation_state.h5`, no
`labeling_run/`, `fps_run/`, `dedup_batches/`, no five stratum
directories -- ONE library directory `yso/` and ONE register.

    python3.9 -m sesnaimpute.sed_models.build_yso <sed_models data root>

INPUTS READ (never written):
    <data root's sibling>/... the raw Robitaille (2017) / Richardson
        (2024) v1.2 release, Zenodo 10.5281/zenodo.8114592, 18 geometry
        directories, 2.2 M raw models -- resolved by `resolve_yso_root`
        below; this run's resolved path is printed at start-up (the
        release is NOT staged under this project's own `downloads/`
        tree today; see that function for the exact fallback chain and
        which path was actually used).
    <data root>/sps/_raw/ -- the RAW stellar-photosphere splice (4,066
        rows) for in-memory dedup, now that Unit SPS has landed it.
        REQUIRED: this line fails loudly, naming the path, if
        `sps/_raw/` is absent -- it never falls back to `sps/` (the
        KEPT/sampled set, the wrong population to dedup against).
    <data root's parent>/sky/download/baraffe2015_bhac15/BHAC15_iso.2mass
    <data root's parent>/sky/download/mist2016/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_basic.iso
        -- the 1 Myr pre-main-sequence track, same files
        `sesnaimpute.population.yso_mass` reads, for the PMS gate.

OUTPUTS WRITTEN (nothing else):
    <data root>/yso/                   ONE library directory (replaces
                                        whatever was there -- the five
                                        stratum directories and the old
                                        `yso_curation_state.h5` go)
    <data root>/registers/yso_register.hdf5

SCIENCE RANGE THIS LIBRARY MUST COVER (owner library spec / review
lib_2_YSO.md): Robitaille's 18-geometry grid itself imposes no
pre-main-sequence evolutionary track (there is no native stellar-mass
axis -- `star.radius`/`star.temperature` are independent free
parameters). "Mass" below is therefore the 1 Myr BHAC15+MIST
track-IMPLIED mass at each row's own luminosity (`yso_pms.PMSTrack`,
the same track the PMS gate itself reads), floored at this project's
IMF floor 0.1 Msun, not a native grid column -- disclosed, not a grid
property. The PMS gate below removes every row whose (T_eff, L) pair
is off that track at any mass; the raw grid's coverage of 0.1-50 Msun
and the gate's own removals are printed at run time, not asserted.

STAGES, each printing its removals or counts:
    1. ingest            Stage/alpha_class/spectral_index from every
                          geometry's info.fits/parameters.fits (cheap,
                          column-only; `yso_curation_state.seed_registry`)
    2. gates              aperture gate (`yso_aperture`), 1 Myr PMS gate
                          (`yso_pms`), each printing removals per stage
    3. labeling + dedup   hard label per gated row (Stage from
                          info.fits; TD carved by the Gutermuth 2009
                          24um criterion as `yso_labeling.
                          compute_td_fraction` already defines it;
                          FLAT relabelled from Spectral Index, Greene
                          et al. 1994's -0.3/+0.3 band, overriding
                          Stage/TD); dedup against `sps/_raw/` by the
                          SAME metric the r-net below uses -- nearest-
                          neighbour distance in the whitened quotient
                          space, `yso_dedup.duplicate_mask`, ONE
                          cKDTree over the 4,066 SPS points and one
                          batched query, not a sedfitter chi^2 fit (the
                          coordinator's 2026-10-08 ruling: the retired
                          `sed_fit`-based metric's register lookup has
                          no meaning for an unregistered raw splice)
    4. r-net              ONE global sequential r-net at radius 1 in
                          the whitened quotient space
                          (`sed_models.sampling`, Unit S), no strata,
                          no budget, over every gated+non-duplicate
                          model
    5. members + assembly FRAC_<subclass> and parameter ranges per kept
                          template (`sampling.members_table`); library
                          files for the kept templates only
    6. write              `library.write_library` + `library.
                          write_register`, ONE library directory, ONE
                          register

MEMORY. The only step whose input scales to ~1.6 M points is the r-net
(stage 4), and it is bounded by construction: `sampling.r_net` never
builds a persistent neighbour-adjacency list (the 50 GB pattern that
killed the prior attempt, `sed_models_curate.yso_fps.
greedy_coverage_r_net`'s `tree.query_ball_point(coords, radius)` over
every point at once -- NOT called anywhere in this module). Every
other step processes one geometry's rows at a time and discards them
before the next geometry opens. Peak RSS is measured with
`resource.getrusage` and printed at the end.

STAGING. Everything is built into `<data root>/_yso_build_work/` -- a
SIBLING of `<data root>/yso/`, never nested under it (nesting the
scratch dir under the output dir means the final swap's
`rmtree(out_lib_dir)` deletes the staged build it is about to move;
this bit the SPS unit's line twice) -- and the real `<data root>/yso/`
is replaced only once, by `shutil.move` after every acceptance assert
below has passed, with an explicit check that the staged path is not
inside the output path before the swap. `_yso_build_work/` is removed
on entry (stale scratch from a killed prior run) and on exit (success
or failure) -- no state file outlives this process.

ACCEPTANCE (asserted here, numbers printed at the end): coverage max
distance under 1.0 with zero uncovered gated models; packing >= 1.0;
members' N_MEMBERS sums to the gated+non-dup count; every kept template
has its own parameters row; FRAC columns sum to one per kept template;
the register opens with h5py and its n_model equals the kept count; the
package imports.
====================================================================
"""

import argparse
import os
import platform
import resource
import shutil
import sys
import time

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from sesnaimpute.sed_models import paths
from sesnaimpute.sed_models.constants import BANDS
from sesnaimpute.sed_models import library, sampling
from sesnaimpute.sed_models.curate import model_io
from sesnaimpute.sed_models.curate import yso_aperture as apert
from sesnaimpute.sed_models.curate import yso_curation_state as regstate
from sesnaimpute.sed_models.curate import yso_dedup as dedup
from sesnaimpute.sed_models.curate import yso_pms as pms
from sesnaimpute.sed_models.curate.model_convolution import write_convolved_band_fits
from sesnaimpute.sed_models.curate.yso_assembly import (
    MEMBER_PARAM_COLUMNS,
    MEMBER_PARAM_FITS_NAMES,
    N_APERTURES,
    N_WAVELENGTHS,
    _extract_convolved_band_rows,
    _extract_parameters_rows,
    _extract_values_cube_rows,
    _read_reference_constants,
)
from sesnaimpute.sed_models.curate.yso_labeling import (
    BAND_ORDER,
    compute_row_features_from_cubes,
    compute_td_fraction,
    read_geometry_flux_cubes,
)
# Dedup's disclosure diagnostic (run_dedup, below) needs the raw
# extinction-law vector, exactly as sampling.py's own quotient-space
# construction reads it -- the one other `sed_models_register` call
# this line makes, matching sampling.py's own single such import.
from sesnaimpute.sed_models.register import density

GEOMETRIES = regstate.GEOMETRIES

MODEL_CLASS = "YSO"
CLASS_LEGEND = (("YSO", "Young stellar object (Richardson 2024 / Robitaille 2017 RT grid)"),)
SUBCLASS_LEGEND = (
    ("C0", "Stage 0: deeply embedded (Richardson/Robitaille Stage column)"),
    ("CI", "Stage I: embedded protostar"),
    # FLAT between CI and CII: the evolutionary order the fitter
    # package's own copy of the canon now holds (coordinator, 2026-10-
    # 08), and the source classmap.fits's SUBCLASS_LEGEND is where the
    # register's /legends/subclass order actually comes from -- moving
    # constants.CLASSMAP alone (done separately) does not reorder this
    # library's own legend; both must agree.
    ("FLAT", "Flat-spectrum: -0.3 <= Spectral Index <= 0.3 (Greene et al. 1994), overrides Stage/TD"),
    ("CII", "Stage II: disc-bearing"),
    ("CIII", "Stage III: diskless; not a bare photosphere (see dedup)"),
    ("TD", "Transition disc: Stage II/III carved by the Gutermuth (2009) 24um criterion"),
)
STAGE_TO_SUBCLASS = {0: "C0", 1: "CI", 2: "CII", 3: "CIII"}
FLAT_ALPHA_LO, FLAT_ALPHA_HI = -0.3, 0.3
TD_FRAC_HARD = 0.5
MODEL_NAME_WIDTH = 34
CONVOLVED_MODEL_NAME_WIDTH = 30
#: One radius for both dedup (distance to the nearest sps/_raw/ model)
#: and the global r-net -- one whitened sampling sigma in every
#: direction at once (sampling.py's own convention).
R_NET_RADIUS = 1.0

#: Native parameters.fits columns the A12 promotion (MEMBER_PARAM_COLUMNS,
#: via yso_assembly.MEMBER_PARAM_FITS_NAMES) does not already carry, but
#: the owner's spec names beside the mass -- added here rather than by
#: editing yso_assembly's own constant, which the retired disk pipeline
#: also reads. HDF5-safe (letters/digits only), same convention.
EXTRA_PARAM_FITS_NAMES = {"star.temperature": "TEFF", "star.radius": "RSUN"}


def _log(msg):
    print(msg, flush=True)


def peak_rss_gb():
    kb_or_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_or_b / 1e9 if platform.system() == "Darwin" else kb_or_b / 1e6


# ==========================================================================
# PATH RESOLUTION
# ==========================================================================

def resolve_yso_root(data_root):
    """Where the raw Robitaille/Richardson release actually lives.

    NOT under `<data_root>/downloads/` -- confirmed empirically, that
    tree holds only btsettl/ck03/galaxy/grams/PAHspec. The real location
    recorded in the retired `yso_curation_state.h5`'s own `/metadata/
    ingest` provenance (this run's own fallback, named here so the
    docstring's "say which path" instruction is honoured) is a sibling
    project tree. Checked in order; the first that has all 18
    geometries wins, and the chosen path is printed.
    """
    candidates = [
        os.path.join(data_root, "downloads", "yso"),
        os.environ.get("YSO_RAW_ROOT", ""),
        "/Users/jtaylor/Dropbox/Research/SESNA_SEDFit_v2/sed_models/yso",
    ]
    for cand in candidates:
        if cand and all(os.path.isdir(os.path.join(cand, g)) for g in GEOMETRIES):
            return cand
    raise FileNotFoundError(
        f"no candidate yso raw root has all {len(GEOMETRIES)} geometries; tried {candidates}")


def _wire_paths_module(root_arg):
    """Several project modules this line calls into (the register
    derivation `sed_models_register.density` reaches for, `constants`'
    own lazily-resolved areas) resolve paths through `sesnacomplete.
    paths`, which needs a real ingested config -- `paths.load_config
    (<root.cfg>)` or `paths.set_data_root`/`set_input_area` wired
    directly. The brief's CLI accepts either a root.cfg path or a bare
    data root; this resolves both, finding the real on-disk `config/
    root.cfg` when there is one (the canonical, already-correct
    `[inputs] sed_models` / `yso_models_download` entries) and only
    wiring paths.py by hand when there is not.

    Returns (sed_models_dir, yso_root), both absolute.
    """
    root_arg = os.path.abspath(root_arg)
    if os.path.isfile(root_arg):
        paths.load_config(root_arg)
    else:
        candidates = [os.path.join(root_arg, "config", "root.cfg"),
                     os.path.join(os.path.dirname(root_arg.rstrip("/")), "config", "root.cfg")]
        cfg = next((c for c in candidates if os.path.isfile(c)), None)
        if cfg:
            paths.load_config(cfg)
        else:
            data_root_guess = (root_arg if os.path.basename(root_arg.rstrip("/")) != "sed_models"
                              else os.path.dirname(root_arg.rstrip("/")))
            paths.set_data_root(data_root_guess)
            sed_models_dir = os.path.join(data_root_guess, "sed_models")
            paths.set_input_area("sed_models", sed_models_dir)
            paths.set_input_area("yso_models_download", resolve_yso_root(sed_models_dir))
    return (os.path.abspath(paths.input_dir("sed_models")),
           os.path.abspath(paths.input_dir("yso_models_download")))


def resolve_paths(root_arg):
    data_root, yso_root = _wire_paths_module(root_arg)
    sps_lib_dir = os.path.join(data_root, "sps")
    sps_raw_dir = os.path.join(sps_lib_dir, "_raw")
    if not os.path.isdir(sps_raw_dir):
        raise FileNotFoundError(
            f"build_yso: {sps_raw_dir} does not exist. This line requires the raw "
            "stellar splice at sps/_raw/ for deduplication (the owner's brief; "
            "sps/ alone is the KEPT/sampled set, not the raw splice to dedup "
            "against) -- it does not fall back to sps/ silently.")
    dedup_source = sps_raw_dir
    pms_data_root = os.path.dirname(data_root.rstrip("/"))
    out_lib_dir = os.path.join(data_root, "yso")
    registers_dir = os.path.join(data_root, "registers")
    # Sibling of yso/, NOT nested under it: `build`'s swap does
    # `rmtree(out_lib_dir)` before moving the staged build into place, so a
    # work_dir nested under out_lib_dir would delete the very staged build
    # it is about to move (the bug the SPS unit's line hit twice).
    work_dir = os.path.join(data_root, "_yso_build_work")
    return dict(data_root=data_root, yso_root=yso_root, sps_lib_dir=sps_lib_dir,
               dedup_source=dedup_source, pms_data_root=pms_data_root,
               out_lib_dir=out_lib_dir, registers_dir=registers_dir, work_dir=work_dir)


# ==========================================================================
# STAGE 1 -- INGEST (cheap, column-only; every row)
# ==========================================================================

def run_ingest(yso_root, verbose=True):
    ingest = regstate.seed_registry(yso_root, geometries=GEOMETRIES, verbose=verbose)
    n_raw = len(ingest)
    n_valid_star = int((ingest["star_radius"] > 0).sum())
    n_valid_stage = int((ingest["stage"] != regstate.STAGE_UNCLASSIFIED).sum())
    _log(f"[ingest] {n_raw} raw models across {len(GEOMETRIES)} geometries; "
        f"star_radius>0: {n_valid_star}; stage classified: {n_valid_stage}")
    return ingest


# ==========================================================================
# STAGE 2 -- GATES (aperture, PMS), each printing removals per stage
# ==========================================================================

def run_gates(ingest, yso_root, pms_data_root, verbose=True):
    n_raw = len(ingest)
    valid_star = (ingest["star_radius"].to_numpy() > 0)
    valid_stage = (ingest["stage"].to_numpy() != regstate.STAGE_UNCLASSIFIED)

    d_min, aperture_info = apert.scan_geometries(yso_root, GEOMETRIES, verbose=verbose)
    d_min = d_min.set_index("model_id").loc[ingest["model_id"]]["d_min_kpc"].to_numpy()
    valid_aperture = d_min < aperture_info["d_near_kpc"]

    track = pms.PMSTrack(pms_data_root)
    teff_all = np.full(n_raw, np.nan)
    lum_all = np.full(n_raw, np.nan)
    for geom in GEOMETRIES:
        sub = ingest[ingest["geometry_folder"] == geom]
        if sub.empty:
            continue
        row_index = sub["model_id"].str.slice(len(geom) + 1).astype(np.int64).to_numpy()
        teff, lum = pms.read_stellar_columns(os.path.join(yso_root, geom), row_index)
        teff_all[sub.index.to_numpy()] = teff
        lum_all[sub.index.to_numpy()] = lum
    valid_pms, pms_delta_dex, pms_mass = pms.off_track_flag(teff_all, lum_all, track)
    valid_pms = valid_pms.astype(bool)

    stage = ingest["stage"].to_numpy()
    _log("\n[gates] removals per stage (raw Stage column; -1 = unclassified):")
    for s in (-1, 0, 1, 2, 3):
        mask_s = stage == s
        n_s = int(mask_s.sum())
        if n_s == 0:
            continue
        n_fail_star = int((mask_s & ~valid_star).sum())
        n_fail_ap = int((mask_s & valid_star & valid_stage & ~valid_aperture).sum())
        n_fail_pms = int((mask_s & valid_star & valid_stage & valid_aperture & ~valid_pms).sum())
        n_pass = int((mask_s & valid_star & valid_stage & valid_aperture & valid_pms).sum())
        _log(f"    stage={s:2d}  n={n_s:8d}  fail_star={n_fail_star:7d}  "
            f"fail_aperture={n_fail_ap:7d}  fail_pms={n_fail_pms:7d}  pass_all={n_pass:7d}")

    eligible_pregate = valid_star & valid_stage & valid_aperture & valid_pms
    _log(f"[gates] D_NEAR={aperture_info['d_near_kpc']:.4f} kpc; "
        f"PMS tol={pms.PMS_TOL_DEX:.4f} dex, m_top_bhac15={track.m_top:.3f}, "
        f"m_top_mist={track.m_top_high:.1f}")
    _log(f"[gates] pre-dedup eligible: {int(eligible_pregate.sum())} of {n_raw}")
    return dict(eligible_pregate=eligible_pregate, pms_mass=pms_mass, track=track,
               aperture_info=aperture_info)


# ==========================================================================
# STAGE 3a -- LABELING (hard label per gated row; one geometry pass)
# ==========================================================================

def _hard_label(stage, td_frac, spectral_index):
    label = np.array([STAGE_TO_SUBCLASS.get(int(s), "CII") for s in stage], dtype=object)
    td_eligible = np.isin(stage, (2, 3))
    label = np.where(td_eligible & (td_frac > TD_FRAC_HARD), "TD", label)
    flat = np.isfinite(spectral_index) & (spectral_index >= FLAT_ALPHA_LO) & (spectral_index <= FLAT_ALPHA_HI)
    label = np.where(flat, "FLAT", label)
    return label.astype(str)


def run_labeling_and_features(ingest, eligible_pregate, pms_mass, yso_root, verbose=True):
    """One pass per geometry over the pre-dedup-eligible rows only: mu
    (clamped log10 flux at survey aperture), dark_mask, hard label, and
    the member-parameter columns (MEMBER_PARAM_COLUMNS + mass).

    ALSO the dark-band gate (coordinator finding, 2026-10-08): a model
    dark in ALL EIGHT bands at the survey aperture (mu at its own floor
    in every band, i.e. dark_mask.all(axis=1)) whitens to the origin of
    the quotient space -- gray, no SED shape at all. One such model
    became the nearest representative of 23,317 others (kept template
    wXx3OKex_03, N_MEMBERS next-largest 1,434), giving a 2.6 Msun
    passive-disc star FRAC_C0=0.197/FRAC_CI=0.794 from members that are
    not hypotheses for any catalogued source. Removed HERE, before
    dedup and before the r-net, in the same per-geometry pass that
    already computes dark_mask -- no second flux read. Counted per
    Richardson Stage and returned (printed by the caller alongside the
    star/aperture/PMS gate table, `fail_dark`)."""
    ref = _read_reference_constants(os.path.join(yso_root, GEOMETRIES[0]))
    aperture_au, distance_cm = ref["aperture_au"], ref["distance_cm"]

    mu_blocks, dark_blocks, label_blocks, id_blocks, name_blocks, param_blocks = [], [], [], [], [], []
    fail_dark_by_stage = {}
    n_dark_total = n_kept_rows_total = 0
    for geom in GEOMETRIES:
        sub = ingest[(ingest["geometry_folder"] == geom) & eligible_pregate]
        if sub.empty:
            continue
        local_idx = sub.index.to_numpy()
        row_index = sub["model_id"].str.slice(len(geom) + 1).astype(np.int64).to_numpy()
        geometry_dir = os.path.join(yso_root, geom)

        ref_names = sub["model_name"].to_numpy()
        flux_cubes = read_geometry_flux_cubes(geometry_dir, ref_names)
        mu, _clamp, floor_dex_row, all_zero = compute_row_features_from_cubes(
            flux_cubes, aperture_au, distance_cm)
        td_frac, _n_usable = compute_td_fraction(flux_cubes)
        del flux_cubes

        floor_linear_row = 10.0 ** floor_dex_row
        sed_linear = 10.0 ** mu
        dark_mask = np.isclose(sed_linear, floor_linear_row[:, None], rtol=1e-6)

        # Dark-band gate: drop every row dark in all 8 bands, BEFORE it
        # ever reaches dedup or the r-net. stage recorded here (on the
        # full sub, pre-filter) for the per-stage fail_dark tally.
        stage_arr = sub["stage"].to_numpy()
        all_dark = dark_mask.all(axis=1)
        n_dark_total += int(all_dark.sum())
        n_kept_rows_total += int((~all_dark).sum())
        for s in np.unique(stage_arr[all_dark]):
            fail_dark_by_stage[int(s)] = fail_dark_by_stage.get(int(s), 0) + int(
                np.sum(all_dark & (stage_arr == s)))

        keep = ~all_dark
        sub = sub.loc[keep]
        local_idx = local_idx[keep]
        row_index = row_index[keep]
        mu, dark_mask, td_frac = mu[keep], dark_mask[keep], td_frac[keep]

        label = _hard_label(sub["stage"].to_numpy(), td_frac, sub["spectral_index"].to_numpy())

        _, param_cols = _extract_parameters_rows(
            os.path.join(geometry_dir, "parameters.fits"), row_index)
        mass_here = pms_mass[local_idx]
        # HDF5-safe names at the source (letters/digits/underscore only --
        # "Source Luminosity"/"envelope.rho_0" etc. are not): reuse
        # yso_assembly's own MEMBER_PARAM_FITS_NAMES mapping rather than
        # inventing a second one, so this library's parameters.fits and
        # the owner library spec's existing member-column stems agree.
        prow = {MEMBER_PARAM_FITS_NAMES["mass"]: mass_here}
        for col in MEMBER_PARAM_COLUMNS:
            prow[MEMBER_PARAM_FITS_NAMES[col]] = param_cols.get(col, np.zeros(len(sub)))
        # Coordinator finding: the A12 promotion dropped Robitaille's own
        # star.temperature/star.radius -- the spec's YSO row names
        # temperature beside the mass. Added alongside, same native-
        # column read (_extract_parameters_rows already has them in
        # param_cols; PARAM_UNION_COLUMNS carries both), same HDF5-safe
        # naming convention as MEMBER_PARAM_FITS_NAMES.
        for col, safe in EXTRA_PARAM_FITS_NAMES.items():
            prow[safe] = param_cols.get(col, np.zeros(len(sub)))

        mu_blocks.append(mu)
        dark_blocks.append(dark_mask)
        label_blocks.append(label)
        id_blocks.append(sub["model_id"].to_numpy())
        name_blocks.append(sub["model_name"].to_numpy())
        param_blocks.append(pd.DataFrame(prow))
        if verbose:
            _log(f"[labeling] {geom}: {len(sub)} gated rows (dropped "
                f"{int(all_dark.sum())} all-8-dark), all_zero_flux={int(np.sum(all_zero))}")

    model_id = np.concatenate(id_blocks)
    model_name = np.concatenate(name_blocks)
    mu = np.concatenate(mu_blocks, axis=0)
    dark_mask = np.concatenate(dark_blocks, axis=0)
    label = np.concatenate(label_blocks)
    params = pd.concat(param_blocks, axis=0, ignore_index=True)

    _log("\n[gates] dark-band gate (all 8 bands at floor at the survey "
        "aperture), removals per stage:")
    for s in sorted(fail_dark_by_stage):
        _log(f"    stage={s:2d}  fail_dark={fail_dark_by_stage[s]:7d}")
    _log(f"[gates] fail_dark total={n_dark_total}; pre-dedup eligible after "
        f"dark gate: {n_kept_rows_total}")

    counts = pd.Series(label).value_counts()
    _log(f"\n[labeling] hard-label counts over {len(label)} gated rows:\n{counts}")
    return model_id, model_name, mu, dark_mask, label, params, fail_dark_by_stage


# ==========================================================================
# STAGE 3b -- DEDUP: the r-net's own metric against sps/_raw/ (coordinator
# ruling 2026-10-08 -- NOT sed_fit; see yso_dedup.py's module docstring)
# ==========================================================================

def run_dedup(model_id, mu, dark_mask, dedup_source, radius=R_NET_RADIUS, verbose=True):
    """Nearest-neighbour distance, in the SAME whitened quotient space
    the global r-net uses, from every gated YSO row to the nearest
    `sps/_raw/` model. A YSO model within `radius` of its nearest SPS
    model is a photosphere duplicate -- one `cKDTree` over the 4,066
    SPS points, one batched query over the whole gated pool."""
    t0 = time.time()
    coords = sampling.whiten(mu, dark_mask)
    dup, dist, nearest, sps_names, sps_mu = dedup.duplicate_mask(coords, dedup_source, radius)
    elapsed = time.time() - t0
    _log(f"[dedup] nearest-sps/_raw/ distance in the whitened quotient space, "
        f"radius={radius}: {int(dup.sum())} of {len(model_id)} duplicates "
        f"({100*dup.mean():.2f}%) in {elapsed:.2f}s")

    # Disclosure (coordinator's ruling item 2), not a correction: the
    # quotient allows negative extinction, so a duplicate whose nearest
    # sps/_raw/ match implies A_V < 0 is one a one-sided A_V>=0 fit would
    # not have called a duplicate. av_law is -0.4*chi(lambda)/chi(V) <= 0
    # at every band (density._av_law), so projecting the whitened
    # (YSO - matched SPS) difference onto that direction's own unit
    # vector is negative exactly when the implied A_V is negative.
    sigma_vec, _ = sampling.quotient_space()
    av_law_whitened = density._av_law(sampling._SAMPLING_LAW) / sigma_vec
    av_dir = av_law_whitened / np.linalg.norm(av_law_whitened)
    diff = (mu[dup] / sigma_vec) - (sps_mu[nearest[dup]] / sigma_vec)
    n_bluer = int(np.sum((diff @ av_dir) < 0)) if dup.any() else 0
    _log(f"[dedup] disclosure: {n_bluer} of {int(dup.sum())} duplicates "
        f"({100*n_bluer/max(int(dup.sum()),1):.1f}%) have their nearest sps/_raw/ "
        f"match implying negative extinction (bluer than that photosphere) -- a "
        f"one-sided A_V>=0 fit would not have matched these; not removed from "
        f"the duplicate set, per the ruling.")
    return dup, dist, nearest, sps_names, sps_mu, n_bluer


# ==========================================================================
# STAGE 4 -- ONE GLOBAL R-NET (sampling.py, Unit S; memory-bounded)
# ==========================================================================

def run_rnet(mu, dark_mask, radius=1.0, verbose=True):
    sigma_vec, projector = sampling.quotient_space()
    coords = sampling.whiten(mu, dark_mask)
    _log(f"[r-net] {coords.shape[0]} gated+non-dup models in {coords.shape[1]}-D "
        f"whitened quotient space, radius={radius}")
    t0 = time.time()
    kept_index = sampling.r_net(coords, radius=radius)
    _log(f"[r-net] kept {len(kept_index)} templates in {(time.time()-t0)/60.0:.2f} min; "
        f"peak RSS so far {peak_rss_gb():.3f} GB")
    rep_of, dist_to_rep = sampling.assign(coords, kept_index)
    max_dist, n_uncovered = sampling.coverage(coords, kept_index, radius=radius)
    min_kept_dist = sampling.packing(coords, kept_index)
    _log(f"[r-net] coverage max_dist={max_dist:.6f} (<{radius}), n_uncovered={n_uncovered}; "
        f"packing min_kept_dist={min_kept_dist:.6f} (>={radius})")
    return coords, kept_index, rep_of, dist_to_rep, max_dist, n_uncovered, min_kept_dist


def effective_dimension(coords, radii=(0.8, 1.2), frac=0.10, seed=0):
    """log(kept-count ratio) / log(radius ratio) on a `frac` subsample at
    two radii -- the brief's own diagnostic, not an acceptance number."""
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    idx = rng.choice(n, size=max(1, int(round(n * frac))), replace=False)
    sub = coords[idx]
    counts = []
    for r in radii:
        kept = sampling.r_net(sub, radius=r)
        counts.append(len(kept))
    d_eff = (np.log(counts[0] / counts[1]) / np.log(radii[1] / radii[0]))
    return dict(radii=radii, counts=counts, n_sub=sub.shape[0], d_eff=float(d_eff))


# ==========================================================================
# STAGE 5 -- MEMBERS + LIBRARY FILES FOR KEPT TEMPLATES ONLY
# ==========================================================================

def build_members_and_library(model_id, model_name, label, params, coords,
                               kept_index, rep_of, dist_to_rep, ingest, yso_root, verbose=True):
    members = sampling.members_table(model_name, rep_of, dist_to_rep, label, Table.from_pandas(params))

    kept_model_id = model_id[kept_index]
    kept_model_name = model_name[kept_index]
    kept_label = label[kept_index]
    kept_geom = ingest.set_index("model_id").loc[kept_model_id, "geometry_folder"].to_numpy()
    kept_row_index = np.array([
        int(mid.split(":")[-1]) for mid in kept_model_id], dtype=np.int64)

    ref = _read_reference_constants(os.path.join(yso_root, GEOMETRIES[0]))

    n_k = len(kept_index)
    values = np.zeros((n_k, N_APERTURES, N_WAVELENGTHS), dtype=np.float32)
    uncertainties = np.zeros((n_k, N_APERTURES, N_WAVELENGTHS), dtype=np.float32)
    convolved_flux = {b: np.zeros((n_k, N_APERTURES), dtype=np.float64) for b in BAND_ORDER}
    convolved_err = {b: np.zeros((n_k, N_APERTURES), dtype=np.float64) for b in BAND_ORDER}

    order_df = pd.DataFrame({"pos": np.arange(n_k), "geom": kept_geom, "row": kept_row_index,
                             "name": kept_model_name})
    for geom, sub in order_df.groupby("geom"):
        positions = sub["pos"].to_numpy()
        row_index_sorted_order = np.argsort(sub["row"].to_numpy())
        row_index_sorted = sub["row"].to_numpy()[row_index_sorted_order]
        positions_sorted = positions[row_index_sorted_order]

        geometry_dir = os.path.join(yso_root, geom)
        flux_names, flux_values, flux_unc = _extract_values_cube_rows(
            os.path.join(geometry_dir, "flux.fits"), row_index_sorted, N_APERTURES, True)
        values[positions_sorted] = flux_values
        uncertainties[positions_sorted] = flux_unc

        model_names_here = sub["name"].to_numpy()[row_index_sorted_order]
        for band in BAND_ORDER:
            flux_b, err_b = _extract_convolved_band_rows(geometry_dir, band, model_names_here)
            convolved_flux[band][positions_sorted] = flux_b
            convolved_err[band][positions_sorted] = err_b
        if verbose:
            _log(f"[assembly] {geom}: {len(sub)} kept templates read back")

    flux_kwargs = dict(wave_um_desc=ref["wavelength"], freq_hz_desc=ref["frequency"],
                       values=values, distance_cm=ref["distance_cm"],
                       apertures_au=ref["aperture_au"], uncertainties=uncertainties)
    classmap_kwargs = dict(class_id=MODEL_CLASS, subclass=kept_label,
                           class_legend=CLASS_LEGEND, subclass_legend=SUBCLASS_LEGEND,
                           provenance={"ORIGIN": "Richardson 2024 / Robitaille 2017 RT grid",
                                       "DOI": "10.5281/zenodo.8114592",
                                       "SAMPLING": "global r-net, radius=1.0, whitened quotient space"})
    convolved_kwargs = {
        band: dict(total_flux_mjy=convolved_flux[band], apertures_au=ref["aperture_au"],
                  filter_wavelength_um=float(BANDS[band].wvl_effective_um),
                  total_flux_err_mjy=convolved_err[band])
        for band in BAND_ORDER
    }
    models_conf_kwargs = dict(name="yso", aperture_dependent=True)

    kept_params = params.iloc[kept_index].reset_index(drop=True)
    kept_params.insert(0, "MODEL_NAME", kept_model_name)
    parameters_table = Table.from_pandas(kept_params)

    return dict(flux=flux_kwargs, parameters=parameters_table, classmap=classmap_kwargs,
               model_names=kept_model_name, members=members, convolved=convolved_kwargs,
               models_conf=models_conf_kwargs)


# ==========================================================================
# MAIN
# ==========================================================================

def build(data_root, radius=R_NET_RADIUS, verbose=True):
    t_start = time.time()
    p = resolve_paths(data_root)
    _log(f"[paths] data_root={p['data_root']}")
    _log(f"[paths] yso_root (raw release)={p['yso_root']}")
    _log(f"[paths] dedup splice={p['dedup_source']}")
    _log(f"[paths] pms_data_root={p['pms_data_root']}")

    if os.path.isdir(p["work_dir"]):
        shutil.rmtree(p["work_dir"])

    ingest = run_ingest(p["yso_root"], verbose=verbose)
    gates = run_gates(ingest, p["yso_root"], p["pms_data_root"], verbose=verbose)
    eligible_pregate = gates["eligible_pregate"]

    model_id, model_name, mu, dark_mask, label, params, fail_dark_by_stage = run_labeling_and_features(
        ingest, eligible_pregate, gates["pms_mass"], p["yso_root"], verbose=verbose)

    t_dedup0 = time.time()
    dup, dist, nearest, sps_names, sps_mu, n_bluer = run_dedup(
        model_id, mu, dark_mask, p["dedup_source"], radius=radius, verbose=verbose)
    dedup_min = (time.time() - t_dedup0) / 60.0
    keep_nondup = ~dup
    model_id, model_name, mu, dark_mask, label, params = (
        model_id[keep_nondup], model_name[keep_nondup], mu[keep_nondup],
        dark_mask[keep_nondup], label[keep_nondup], params.iloc[keep_nondup].reset_index(drop=True))
    n_gated = len(model_id)
    _log(f"\n[gated] final gated+non-dup pool: {n_gated}")

    t_rnet0 = time.time()
    coords, kept_index, rep_of, dist_to_rep, max_dist, n_uncovered, min_kept_dist = run_rnet(
        mu, dark_mask, radius=radius, verbose=verbose)
    rnet_min = (time.time() - t_rnet0) / 60.0

    eff_dim = effective_dimension(coords)
    _log(f"[r-net] effective dimension (10% subsample, radii {eff_dim['radii']}): "
        f"{eff_dim['d_eff']:.3f} (counts {eff_dim['counts']})")

    # For the record (coordinator's instruction): how many gated-AND-KEPT
    # templates are dark in exactly 7 of 8 bands -- one band of shape
    # only, no action taken on them.
    n_dark_7of8 = int(np.sum(dark_mask[kept_index].sum(axis=1) == 7))
    _log(f"[report] kept templates dark in 7 of 8 bands (one band of shape "
        f"only; no action): {n_dark_7of8} of {len(kept_index)}")

    lib_payload = build_members_and_library(
        model_id, model_name, label, params, coords, kept_index, rep_of, dist_to_rep,
        ingest, p["yso_root"], verbose=verbose)

    # ---- acceptance asserts ----
    members = lib_payload["members"]
    assert max_dist < 1.0 + 1e-9, f"coverage max_dist {max_dist} >= 1.0"
    assert n_uncovered == 0, f"{n_uncovered} uncovered gated models"
    assert min_kept_dist >= radius - 1e-9, f"packing {min_kept_dist} < {radius}"
    assert int(np.sum(members["N_MEMBERS"])) == n_gated, "N_MEMBERS does not sum to gated count"
    frac_cols = [c for c in members.colnames if c.startswith("FRAC_")]
    frac_sum = np.sum([np.asarray(members[c]) for c in frac_cols], axis=0)
    assert np.allclose(frac_sum, 1.0), "FRAC columns do not sum to one"
    assert len(lib_payload["parameters"]) == len(kept_index), "parameters row count != kept count"
    _log("[accept] coverage/packing/FRAC/N_MEMBERS/parameters asserts: PASS")

    # Two targeted acceptance checks for the dark-band gate itself
    # (coordinator's instruction): wXx3OKex_03 is a LEGITIMATE bright
    # gray model (F_nu 35-45 mJy at every aperture, 1.2-24um) -- gray,
    # not dark -- so the gate must NOT have removed it, and it should
    # keep the single-digit N_MEMBERS its gray neighbours K8lVjk47_03
    # and fTAkps5d_04 (N_MEMBERS=7) have, now that the 23,317 all-dark
    # rows that whitened to the same origin-adjacent point are gone.
    member_names = np.char.strip(np.asarray(members["MODEL_NAME"]).astype(str))
    sentinel_row = np.flatnonzero(member_names == "wXx3OKex_03")
    assert sentinel_row.size == 1, (
        "wXx3OKex_03 is not a kept template after the dark-band gate -- "
        "the gate removed a legitimate bright gray model, not only dark ones")
    sentinel_n = int(members["N_MEMBERS"][sentinel_row[0]])
    assert sentinel_n < 10, (
        f"wXx3OKex_03 still has N_MEMBERS={sentinel_n} (expected single-digit, "
        f"like K8lVjk47_03/fTAkps5d_04) -- the dark-band gate did not remove "
        f"the all-dark rows that were inflating it")
    sentinel_frac = {c[len('FRAC_'):]: float(members[c][sentinel_row[0]]) for c in frac_cols}
    protostellar_like = sentinel_frac.get("C0", 0.0) + sentinel_frac.get("CI", 0.0) > 0.5
    _log(f"[accept] wXx3OKex_03 kept, N_MEMBERS={sentinel_n} (was 23,317); "
        f"FRAC={sentinel_frac}; still reads protostellar (C0+CI>0.5)? "
        f"{protostellar_like} -- {'SECOND PROBLEM, reported not fixed' if protostellar_like else 'reads as a passive-disc star, as expected'}")

    kept_subclass = lib_payload["classmap"]["subclass"]
    n_c0 = int(np.sum(kept_subclass == "C0"))
    n_ci = int(np.sum(kept_subclass == "CI"))
    _log(f"[report] kept templates: {len(kept_index)} (vs 200,000 shipped); "
        f"C0={n_c0} CI={n_ci} C0:CI ratio={n_c0/max(n_ci,1):.3f}")

    # ---- stage, then replace the real library directory atomically ----
    stage_dir = os.path.join(p["work_dir"], "yso")
    os.makedirs(p["work_dir"], exist_ok=True)
    library.write_library(stage_dir, "yso", **lib_payload)
    register_path = library.write_register(p["registers_dir"], "yso", stage_dir)

    # verify register before touching the real directory
    import h5py
    with h5py.File(register_path, "r") as h:
        n_model_reg = int(h["models"]["MODEL_NAME"].shape[0])
        models_cols = set(h["models"].keys())
        required_physical = {MEMBER_PARAM_FITS_NAMES[c] for c in
                             ("mass", "Source Luminosity", "Av", "inclination",
                              "envelope.rho_0", "envelope.rc", "disk.mass")}
        required_physical |= set(EXTRA_PARAM_FITS_NAMES.values())
        missing_physical = required_physical - models_cols
        assert not missing_physical, f"models group missing A12 columns {missing_physical}"

        assert "subclass_prob" in h, "no /subclass_prob group (members-sourced FRAC join)"
        sp = h["subclass_prob"]
        sp_codes = set(sp.keys()) - {"MODEL_NAME"}
        assert sp_codes == {"C0", "CI", "CII", "CIII", "TD", "FLAT"}, (
            f"/subclass_prob codes {sp_codes} != the six strata")
        sp_sum = np.sum([np.asarray(sp[c]) for c in sorted(sp_codes)], axis=0)
        assert np.allclose(sp_sum, 1.0), "/subclass_prob rows do not sum to one"

        assert "bands" in h and "VEGA_ZP_MJY" in h["bands"], "no /bands VEGA_ZP_MJY"
        assert h["bands"]["VEGA_ZP_MJY"].shape[0] == 8, "/bands does not carry 8 zero points"
    assert n_model_reg == len(kept_index), (
        f"register n_model {n_model_reg} != kept count {len(kept_index)}")
    _log("[accept] register /models A12 physical columns, /subclass_prob (6, sum=1), "
        "/bands 8 zero points: PASS")

    stage_real = os.path.realpath(stage_dir)
    out_real = os.path.realpath(p["out_lib_dir"])
    assert not stage_real.startswith(out_real + os.sep) and stage_real != out_real, (
        f"staged build {stage_real} is inside the output directory {out_real} -- "
        "removing out_lib_dir before the move would delete the staged build")
    if os.path.isdir(p["out_lib_dir"]):
        shutil.rmtree(p["out_lib_dir"])
    shutil.move(stage_dir, p["out_lib_dir"])
    shutil.rmtree(p["work_dir"], ignore_errors=True)

    wall_min = (time.time() - t_start) / 60.0
    _log(f"\n[done] wall time {wall_min:.1f} min (dedup {dedup_min:.2f} min, "
        f"r-net {rnet_min:.2f} min); peak RSS {peak_rss_gb():.3f} GB")
    return dict(n_raw=len(ingest), n_gated=n_gated, n_kept=len(kept_index),
               max_dist=max_dist, n_uncovered=n_uncovered, min_kept_dist=min_kept_dist,
               eff_dim=eff_dim, wall_min=wall_min, dedup_min=dedup_min, rnet_min=rnet_min,
               peak_rss_gb=peak_rss_gb(), register_path=register_path,
               lib_dir=p["out_lib_dir"], n_c0=n_c0, n_ci=n_ci, n_bluer_dedup=n_bluer)


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python3.9 -m sesnaimpute.sed_models.build_yso",
        description="Build the YSO model library, in memory, one global r-net, "
                    "one library directory, one register.")
    parser.add_argument("data_root", help="sed_models data root "
                        "(e.g. .../SESNA_Complete/sed_models)")
    parser.add_argument("--radius", type=float, default=R_NET_RADIUS)
    args = parser.parse_args(raw_args)
    build(args.data_root, radius=args.radius)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
