"""
build_galz.py
====================================================================
ONE command: the galaxy contaminant library (Polletta 2007 SWIRE,
Berta 2013, Brown 2014 GALSEDATLAS), redshifted on a grid this line
states, sampled once in the shared whitened quotient space, and
written with its register in one act.

    python3.9 -m sesnaimpute.sed_models.build_galz <root.cfg path or sed_models dir>

Paths resolve through `sesnacomplete.paths` (`ingest`/`downloads_dir`/
`lib_dir`/`registers_dir` below), the same way `build_h2shock.py` does
-- not by joining subpaths onto the project root by hand, which once
wrote `<project_root>/galz/` and `<project_root>/registers/` instead of
`<project_root>/sed_models/galz/` and `.../sed_models/registers/` when
given root.cfg (the project root, not the sed_models dir).

Reads only:
    <sed_models>/downloads/galaxy/{polletta2007_swire,berta2013_templates,
    brown2014_galsedatlas}  (195 files)
    sesnacomplete.constants (BANDS, LIBRARY_SAMPLING_SIGMA_LOG_VECTOR,
    REFERENCE_DISTANCE_CM, POINT_SOURCE_APERTURE_AU, LOGD_STEP)
Writes only:
    <sed_models>/galz/            (the library directory -- replaced whole)
    <sed_models>/registers/galz_register.hdf5

STAGES 0-4 are `sesnacomplete.sed_models_curate.galaxy_curate`'s own
rest-frame pipeline, called here and not rewritten: acquisition
(`fetch_all_galaxy_sources`), per-source rest-frame template loading
(`load_{polletta,berta,brown}_rest_frame_templates`), the master
wavelength grid (`build_master_wavelength_grid`), TEMPLATE_CLASS /
PAH_EW_6_2 classification (`classify_{polletta,berta,brown}_templates`)
and SUBCLASS/DONLEY_AGN (`classify_parent_subclasses`, its own
throwaway z=0 convolution through the official sedfitter path). The
fast raw-grid convolution primitives (`resample_one_sed_onto_master_
grid`, `build_fast_band_filters`, `fast_band_fluxes_mjy`) are reused
the same way; only the GRID those primitives are driven over is new
here.

THE RAW SET IS DEFINED IN THIS LINE. `galaxy_curate`'s own
`build_redshift_grid`/`RAW_GRID_N_Z`/`GALAXY_SOURCE_REDSHIFT_CAP` (a
log-uniform-in-log10(1+z), 101-point-per-source grid) is NOT called:
each of the 185 parent templates is instead redshifted on a LINEAR
grid in z, step 1/17 (0.058823..., stated as 0.059), from z=0 to that
SOURCE's own published redshift reach:

    Polletta 2007 (pol07)  z_max = 2.0   (ApJ 663, 81: built/validated
                                          against SWIRE/SDSS sources to
                                          z ~ 2)
    Berta 2013   (ber13)   z_max = 3.0   (A&A 551, A100: fitted to
                                          Herschel PEP/HerMES sources
                                          to z ~ 3)
    Brown 2014   (bro14)   z_max = 1.0   (the brief's stated reach for
                                          this build; the atlas's own
                                          galaxies are nearby, z<0.06 --
                                          this is coverage insurance,
                                          same role the old z=0-6 tail
                                          played, now bounded at the
                                          population's own z~3 edge
                                          rather than left at z=6)

The step is the MEASURED "17 noise lengths per unit z" the brief
states: SIGEFF (the whitened quotient-space sampling radius,
`sampling.quotient_space`) is 1.0 in whitened units, and one "noise
length in redshift" is the z-step whose own adjacent-row quotient-
space displacement is of that order -- 1/17 of a unit in z, both
numbers (17 and 0.059) stated here so the grid is reproducible from
this line alone with no external measurement run. Each cap is an
EXACT multiple of 1/17 (2*17=34, 3*17=51, 1*17=17), so the grid lands
exactly on every source's cap with no partial last step and the raw
set is enumerated by name, "<tag>.<parent>.z<zzzz>", deterministically.

SAMPLING is `sesnacomplete.sed_models.sampling`, the ONE shared
quotient-space/whiten/r-net/assign/coverage/packing/members_table
(Unit S, 2026-10-08) -- not `galaxy_curate.greedy_max_cover_r_net` /
`assign_nearest_representative` / `project_raw_grid_to_quotient_space`
/ `read_sampling_quotient_space`, which `sampling.py` replaces
project-wide (one of the six near-duplicate r-nets its own module
header names). The per-row flux floor and dark-band flag reuse
`sed_models_curate.curve_of_growth.peak_and_floor` -- the SAME
per-model, 10-dex-below-its-own-peak floor `reference.derive` applies
to `f_ref` -- rather than a second floor convention invented here.

MEMORY. The raw grid here is a few thousand rows (not the tens of
thousands a global z=0-6 grid gives), so each parent template's
(n_z, n_wave) resampled block is built and discarded one template at a
time (never a (n_template, n_z, n_wave) cube), and `sampling.r_net`'s
own bounded KD-tree bookkeeping is the only large structure at any
instant. No subprocess is started by this line (sedfitter's
convolution runs in-process); none therefore needs killing.

STAGING. The library is assembled in a scratch directory
(`tempfile.mkdtemp`) and `<sed_models>/galz/` is replaced WHOLE only
after every write and every acceptance assert below has passed; any
`_work/` under `<sed_models>/galz/` left by a previous run is removed
on entry, and the scratch directory (library build AND the throwaway
z=0 classification directory) is removed on exit, success or failure,
so no state file outlives this run.
====================================================================
"""

import argparse
import os
import resource
import shutil
import sys
import tempfile
import time

import numpy as np
from astropy.table import Table

from sesnaimpute.sed_models import paths
from sesnaimpute.sed_models.constants import (
    CONVOLVED_BAND_NAMES, LOGD_STEP, POINT_SOURCE_APERTURE_AU,
    REFERENCE_DISTANCE_CM,
)
from sesnaimpute.sed_models import library, sampling
from sesnaimpute.sed_models.curate import curve_of_growth as cog
from sesnaimpute.sed_models.curate import galaxy_curate as gc
from sesnaimpute.sed_models.curate.model_convolution import build_convolved_bands

NOISE_LENGTHS_PER_UNIT_Z = 17
REDSHIFT_STEP = 1.0 / NOISE_LENGTHS_PER_UNIT_Z  # 0.058823...

#: Published redshift reach, this line's own grid cap per source --
#: see module header. NOT galaxy_curate.GALAXY_SOURCE_REDSHIFT_CAP
#: (that dict's bro14 entry, 0.06, is the atlas's own measured member
#: redshift; the brief's cap for THIS build is 1.0, stated here).
SOURCE_REDSHIFT_CAP = {"pol07": 2.0, "ber13": 3.0, "bro14": 1.0}

SOURCE_CAP_NOTE = {
    "pol07": "Polletta+2007 ApJ 663,81, validated to z~2",
    "ber13": "Berta+2013 A&A 551,A100, fitted to Herschel PEP/HerMES to z~3",
    "bro14": "brief's stated reach for this build (atlas members z<0.06)",
}

#: The Fazio et al. 2004 (ApJS 154, 39) galaxy counts the population
#: feeds this library through; "to z about 3" is the brief's own
#: statement of the science range this library must cover.
SCIENCE_RANGE_Z_MAX = 3.0


def redshift_grid(z_max):
    """Linear grid in z, step `REDSHIFT_STEP`, 0 to `z_max` inclusive.
    `z_max` must be an exact multiple of `REDSHIFT_STEP` (true for all
    three `SOURCE_REDSHIFT_CAP` values by construction), so the grid
    lands exactly on the cap with no partial last step."""
    n = int(round(z_max * NOISE_LENGTHS_PER_UNIT_Z)) + 1
    grid = np.arange(n, dtype=np.float64) / NOISE_LENGTHS_PER_UNIT_Z
    if abs(grid[-1] - z_max) > 1e-9:
        raise ValueError(
            f"redshift_grid: z_max={z_max} is not an exact multiple of "
            f"1/{NOISE_LENGTHS_PER_UNIT_Z}; grid's last node is {grid[-1]}")
    return grid


def ingest(config_or_root):
    """`config_or_root` is either a plain data-root directory (the
    `sed_models` directory itself, e.g. `.../SESNA_Complete/sed_models`)
    or a `root.cfg`-style config (`sesnacomplete.paths.load_config`) --
    the production convention. Mirrors `build_h2shock.ingest`: a bare
    directory is ingested via `paths.set_data_root` with `sed_models`
    (if present under it) registered as the `sed_models` input area, so
    `downloads_dir`/`lib_dir`/`registers_dir` resolve the same way
    either path is given. Returns the ingested data root."""
    if os.path.isdir(config_or_root):
        root = paths.set_data_root(config_or_root)
        sed_models_dir = os.path.join(root, "sed_models")
        if os.path.isdir(sed_models_dir):
            paths.set_input_area("sed_models", sed_models_dir)
        return root
    return paths.load_config(config_or_root)


def downloads_dir():
    return os.path.join(paths.input_dir("sed_models"), "downloads", "galaxy")


def lib_dir():
    return paths.path_for("sed_models_galz")


def registers_dir():
    return paths.path_for("model_registers_dir")


def _rest_templates(download_dir):
    """Stage 1 loaders, unmodified: {tag: {name: (wave_rest_um, flux[,
    source_flag])}}."""
    return {
        "pol07": gc.load_polletta_rest_frame_templates(download_dir),
        "ber13": gc.load_berta_rest_frame_templates(download_dir),
        "bro14": gc.load_brown_rest_frame_templates(download_dir),
    }


def _save_rest_archives(scratch_dir, templates_by_tag, z_grid_by_tag):
    """galaxy_curate.save_rest_frame_template_archive, unmodified, so
    classify_{polletta,berta,brown}_templates and classify_parent_
    subclasses (which read these archives) can be called unchanged.
    The archive's own stored z_grid is not read by either of those
    functions (they look only at `names` and each name's wave/flux);
    this line's grid is passed anyway so the archive is self-
    describing."""
    out_dir = os.path.join(scratch_dir, "rest_archives")
    for tag, templates in templates_by_tag.items():
        gc.save_rest_frame_template_archive(
            out_dir, gc.SOURCE_REST_ARCHIVE[tag], templates,
            z_grid_by_tag[tag], has_source_flag=(tag == "bro14"))
    return out_dir


def _raw_model_name(tag, name, z):
    alias = gc.MODEL_NAME_TEMPLATE_ALIAS.get((tag, name), name)
    return f"{tag}.{alias}.z{z:.4f}"


def _raw_grid_photometry(templates_by_tag, z_grid_by_tag, master_wave_um):
    """The dense raw set's 8-band photometry -- THIS line's grid, driven
    through galaxy_curate's fast internal convolution primitives
    (reused, not reimplemented). One parent template's (n_z, n_wave)
    block is built and discarded before the next starts.

    Returns (tags, names, redshifts, band_flux_mjy[n, 8], band_order).
    """
    band_order = CONVOLVED_BAND_NAMES
    band_filters = gc.build_fast_band_filters(band_order, master_wave_um)
    tags, names, zs, blocks = [], [], [], []
    for tag in gc.SOURCE_TAGS:
        z_grid = z_grid_by_tag[tag]
        for name, vals in templates_by_tag[tag].items():
            wave_rest_um, flux_rest = vals[0], vals[1]
            block = np.empty((z_grid.size, master_wave_um.size), dtype=np.float32)
            for j, z in enumerate(z_grid):
                block[j] = gc.resample_one_sed_onto_master_grid(
                    wave_rest_um, flux_rest, z, master_wave_um)
            band_flux = gc.fast_band_fluxes_mjy(block, master_wave_um, band_filters, band_order)
            n = z_grid.size
            tags.extend([tag] * n)
            names.extend([name] * n)
            zs.extend(z_grid.tolist())
            blocks.append(band_flux)
    return (np.array(tags), np.array(names), np.array(zs, dtype=np.float64),
            np.concatenate(blocks, axis=0), band_order)


def _floor_and_dark(band_flux, band_order):
    """The register's own B0/B0.1 convention (curve_of_growth.
    peak_and_floor: per-MODEL floor, 10 dex below that model's own peak
    flux over every band) -- reused, not a second floor invented for
    sampling. Returns (log10_flux, dark_mask), both (n, 8), ready for
    `sampling.whiten`."""
    flux_cubes = {b: band_flux[:, i:i + 1] for i, b in enumerate(band_order)}
    _peak, _floor_dex, floor_linear, all_zero = cog.peak_and_floor(flux_cubes)
    if np.any(all_zero):
        raise ValueError(
            f"{int(all_zero.sum())} raw rows have exactly zero flux in "
            f"every band -- no peak to floor against")
    floored = np.maximum(band_flux, floor_linear[:, None])
    dark_mask = band_flux <= floor_linear[:, None]
    return np.log10(floored), dark_mask


def _broadcast(parent_lookup, tags, names, key):
    return np.array([parent_lookup[(t, n)][key] for t, n in zip(tags, names)])


def build():
    t0 = time.perf_counter()
    download_dir = downloads_dir()
    final_lib_dir = lib_dir()
    registers_dir_path = registers_dir()

    stale_work = os.path.join(final_lib_dir, "_work")
    if os.path.isdir(stale_work):
        shutil.rmtree(stale_work)
        print(f"removed stale state: {stale_work}")

    scratch = tempfile.mkdtemp(prefix="galz_build_")
    try:
        scratch_lib = os.path.join(scratch, "galz")
        scratch_classify = os.path.join(scratch, "classify")

        # ---- Stage 0: acquisition (pinned, checksummed; idempotent -- the
        # 195 files already on disk are verified, not re-fetched) ----
        gc.fetch_all_galaxy_sources(download_dir)

        # ---- Stage 1: rest-frame loading ----
        templates_by_tag = _rest_templates(download_dir)
        n_templates = {tag: len(t) for tag, t in templates_by_tag.items()}
        print(f"sources   : pol07={n_templates['pol07']} ber13={n_templates['ber13']} "
              f"bro14={n_templates['bro14']} (185 total expected)")

        # ---- THE RAW SET, defined in this line ----
        z_grid_by_tag = {tag: redshift_grid(cap) for tag, cap in SOURCE_REDSHIFT_CAP.items()}
        print(f"raw grid  : step 1/{NOISE_LENGTHS_PER_UNIT_Z} = {REDSHIFT_STEP:.6f} "
              f"({REDSHIFT_STEP:.3f} stated)")
        for tag, cap in SOURCE_REDSHIFT_CAP.items():
            print(f"            {tag}: z=0..{cap} ({z_grid_by_tag[tag].size} nodes) "
                  f"-- {SOURCE_CAP_NOTE[tag]}")

        archives_dir = _save_rest_archives(scratch_classify, templates_by_tag, z_grid_by_tag)

        # ---- Stage 2: master wavelength grid ----
        master_wave_um = gc.build_master_wavelength_grid()

        # ---- Stage 3: TEMPLATE_CLASS / PAH_EW_6_2 (rest-frame, per parent) ----
        brown_csv_path = os.path.join(
            download_dir, "brown2014_galsedatlas", "catalogs",
            "hlsp_galsedatlas_multi_multi_summary_multi_v1_cat.csv")
        classification_csv = os.path.join(scratch_classify, "template_classification.csv")
        gc.write_template_classification_csv(
            classification_csv,
            gc.classify_polletta_templates(archives_dir),
            gc.classify_berta_templates(archives_dir),
            gc.classify_brown_templates(archives_dir, brown_csv_path),
        )

        # ---- Stage 4: SUBCLASS + DONLEY_AGN, per parent, z=0-only throwaway ----
        parent_lookup, parent_counts = gc.classify_parent_subclasses(
            archives_dir, classification_csv, scratch_classify, master_wave_um,
            bands=gc.IRAC_SUBCLASS_BANDS)
        print(f"subclass  : parents by SUBCLASS {parent_counts} (185 templates)")

        # ---- raw-grid photometry (fast internal convolution, reused) ----
        tags, names, zs, band_flux, band_order = _raw_grid_photometry(
            templates_by_tag, z_grid_by_tag, master_wave_um)
        n_raw = tags.size
        raw_model_names_arr = np.array(
            [_raw_model_name(t, n, z) for t, n, z in zip(tags, names, zs)])
        if np.unique(raw_model_names_arr).size != n_raw:
            raise ValueError(
                "raw MODEL_NAME collision -- the grid step is too coarse for "
                "the 4-decimal name format")
        print(f"raw set   : {n_raw} rows (185 parents x per-source grid); "
              f"old shipped raw set 59,015")

        # ---- the ONE shared sampler (sampling.py) ----
        log10_flux, dark_mask = _floor_and_dark(band_flux, band_order)
        sigma_vec, projector = sampling.quotient_space()
        coords = sampling.whiten(log10_flux, dark_mask)
        kept_index = sampling.r_net(coords, radius=1.0)
        rep_of, dist_to_rep = sampling.assign(coords, kept_index)
        max_dist, n_uncovered = sampling.coverage(coords, kept_index, radius=1.0)
        min_kept_dist = sampling.packing(coords, kept_index)
        n_kept = kept_index.size
        print(f"sampling  : whitened quotient space d={np.linalg.matrix_rank(projector)}, "
              f"radius=1.0 -> kept {n_kept} of {n_raw}")
        print(f"          : coverage max_dist={max_dist:.6f} n_uncovered={n_uncovered}")
        print(f"          : packing min_kept_dist={min_kept_dist:.6f}")

        subclass_raw = _broadcast(parent_lookup, tags, names, "subclass")
        pah_ew_raw = _broadcast(parent_lookup, tags, names, "pah_ew_6_2")
        params_raw = Table()
        params_raw["REDSHIFT"] = zs
        params_raw["PAH_EW_6_2"] = pah_ew_raw.astype(np.float64)
        members = sampling.members_table(
            names=raw_model_names_arr, rep_of=rep_of, dist_to_rep=dist_to_rep,
            subclass=subclass_raw, params=params_raw)

        # ---- assemble the kept set's write arrays (reuses galaxy_curate's
        # F_lambda->F_nu / wavelength-reversal / MODEL_NAME-alias logic) ----
        kept_index_sorted = np.sort(kept_index)
        kept_rows = list(zip(tags[kept_index_sorted], names[kept_index_sorted],
                             zs[kept_index_sorted]))
        combined = gc.prepare_combined_model_arrays(
            archives_dir, master_wave_um, kept_rows, classification_csv, parent_lookup)

        parameters = Table()
        parameters["MODEL_NAME"] = combined.model_names
        parameters["REDSHIFT"] = combined.redshift
        parameters["TEMPLATE_CLASS"] = combined.template_class
        parameters["PAH_EW_6_2"] = combined.pah_ew_6_2
        parameters["DONLEY_AGN"] = combined.donley_agn
        parameters["SOURCE"] = combined.source_library

        classmap = dict(
            class_id=gc.MODEL_CLASS,
            subclass=combined.subclass,
            class_legend=gc.CLASS_LEGEND,
            subclass_legend=gc.SUBCLASS_LEGEND,
            provenance=(
                ("CLASS_SOURCE", "library declaration; all models here are this class"),
                ("SUBCLASS_SOURCE", "EW(6.2um) x (Donley wedge or TEMPLATE_CLASS=AGN)"),
                ("SUBCLASS_REF", "Armus+2007 ApJ 656,148; Donley+2012 ApJ 748,142"),
                ("SUBCLASS_NOTE", "assigned at z=0, broadcast to all kept redshifts"),
                ("RAWGRID_NOTE", "z grid linear step 1/17; caps 2/3/1"),
            ),
        )
        flux = dict(
            wave_um_desc=combined.wave_um_desc,
            freq_hz_desc=combined.freq_hz_desc,
            values=combined.values_mjy,
            uncertainties=combined.uncertainties_mjy,
            apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
            distance_cm=REFERENCE_DISTANCE_CM,
            distance_comment=gc.GALAXY_DISTANCE_COMMENT,
            name_format=gc.MODEL_NAME_FORMAT,
        )
        models_conf = dict(
            name=gc.DEFAULT_MODEL_SET_NAME, aperture_dependent=False,
            length_subdir=0, logd_step=LOGD_STEP, version=2,
        )

        library.write_library(scratch_lib, "galz", flux, parameters, classmap,
                              combined.model_names, members, convolved={},
                              models_conf=models_conf)
        build_convolved_bands(scratch_lib, CONVOLVED_BAND_NAMES, combined.model_names,
                              "sedfitter convolve_model_dir, continuum SED")
        print(f"library   : wrote {scratch_lib} ({n_kept} models)")

        # ---- acceptance, asserted here ----
        assert max_dist < 1.0, f"coverage max_dist {max_dist} >= 1.0"
        assert n_uncovered == 0, f"{n_uncovered} raw models left uncovered"
        assert min_kept_dist >= 1.0, f"packing min_kept_dist {min_kept_dist} < 1.0"
        assert int(np.sum(members["N_MEMBERS"])) == n_raw, "members N_MEMBERS does not sum to the raw count"
        assert len(parameters) == n_kept, "parameters row count != kept count"
        frac_cols = [c for c in members.colnames if c.startswith("FRAC_")]
        frac_sum = np.sum([np.asarray(members[c], dtype=float) for c in frac_cols], axis=0)
        assert np.allclose(frac_sum, 1.0, atol=1e-9), "FRAC_* columns do not sum to one"
        print("acceptance: coverage/packing/members-sum/parameters-rows/FRAC-sum all hold")

        # ---- register, same scratch, same act ----
        scratch_registers = os.path.join(scratch, "registers")
        library.write_register(scratch_registers, "galz", scratch_lib)
        reg_path_scratch = os.path.join(scratch_registers, "galz_register.hdf5")

        import h5py
        with h5py.File(reg_path_scratch, "r") as f:
            n_model_reg = f["models"]["MODEL_NAME"].shape[0]
            reg_cols = set(f["models"].keys())
            band_cols = set(f["bands"].keys())
            vega_zp = np.asarray(f["bands"]["VEGA_ZP_MJY"])
            has_subclass_prob = "subclass_prob" in f
            if has_subclass_prob:
                sp_cols = [c for c in f["subclass_prob"].keys() if c != "MODEL_NAME"]
                sp_sum = np.sum([np.asarray(f["subclass_prob"][c], dtype=float)
                                 for c in sp_cols], axis=0)
        assert n_model_reg == n_kept, f"register n_model {n_model_reg} != kept {n_kept}"

        # ---- Unit S round 2 (29228b4): the A12 parameters join, the
        # bands group's per-band Vega zero point, and members-sourced
        # subclass_prob are now asserted here, not just reported ----
        required_param_cols = ("REDSHIFT", "TEMPLATE_CLASS", "PAH_EW_6_2",
                               "DONLEY_AGN", "SOURCE")
        missing_param_cols = [c for c in required_param_cols if c not in reg_cols]
        assert not missing_param_cols, (
            f"register models group missing A12 parameter columns: {missing_param_cols}")
        assert "SUBCLASS" in reg_cols, "register models group missing SUBCLASS"
        assert "VEGA_ZP_MJY" in band_cols, "register bands group missing VEGA_ZP_MJY"
        assert vega_zp.shape[0] == 8, f"bands group has {vega_zp.shape[0]} VEGA_ZP_MJY rows, not 8"
        assert np.all(np.isfinite(vega_zp)), "VEGA_ZP_MJY has non-finite entries"
        assert has_subclass_prob, "register has no subclass_prob group"
        assert np.allclose(sp_sum, 1.0, atol=1e-9), "subclass_prob rows do not sum to one"
        print(f"register  : opens with h5py, n_model={n_model_reg}; models group carries "
              f"{sorted(required_param_cols)} and SUBCLASS; bands group carries "
              f"{vega_zp.shape[0]} VEGA_ZP_MJY values; subclass_prob "
              f"({','.join(sp_cols)}) rows sum to one")

        # ---- science-range coverage and gaps ----
        print(f"science range: Fazio 2004 counts' population to z~{SCIENCE_RANGE_Z_MAX}")
        for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)):
            tags_in_band = [tag for tag, cap in SOURCE_REDSHIFT_CAP.items() if cap > lo]
            n_templ = sum(n_templates[t] for t in tags_in_band)
            print(f"            z in ({lo},{hi}]: {n_templ} templates reach "
                  f"({','.join(sorted(tags_in_band))})")
        print(f"            z > {SCIENCE_RANGE_Z_MAX}: 0 templates (no source's "
              f"cap extends past the stated science range)")
        kept_z = combined.redshift
        for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)):
            frac = float(np.mean((kept_z > lo) & (kept_z <= hi))) if lo > 0 else \
                float(np.mean(kept_z <= hi))
            print(f"            kept representatives with z in ({lo},{hi}]: "
                  f"{frac:.1%}")

        # ---- replace the library directory and register, whole, at the end ----
        if os.path.isdir(final_lib_dir):
            shutil.rmtree(final_lib_dir)
        shutil.move(scratch_lib, final_lib_dir)
        os.makedirs(registers_dir_path, exist_ok=True)
        shutil.move(reg_path_scratch, os.path.join(registers_dir_path, "galz_register.hdf5"))
        print(f"wrote     : {final_lib_dir}")
        print(f"wrote     : {os.path.join(registers_dir_path, 'galz_register.hdf5')}")

    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    wall = time.perf_counter() - t0
    peak_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS (darwin), KB everywhere else (Linux).
    peak_mb = peak_raw / 1024.0 / 1024.0 if sys.platform == "darwin" else peak_raw / 1024.0
    print(f"\nGALZ complete: raw={n_raw} kept={n_kept} wall={wall:.1f}s "
          f"peak_rss={peak_mb:.1f}MB")
    return {
        "raw": n_raw, "kept": n_kept, "max_dist": max_dist, "n_uncovered": n_uncovered,
        "min_kept_dist": min_kept_dist, "wall_s": wall, "peak_mb": peak_mb,
    }


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python3.9 -m sesnaimpute.sed_models.build_galz",
        description="Build the GALZ (galaxy contaminant) model library and register.")
    parser.add_argument("root", help="root.cfg path, or the sed_models data-root directory")
    args = parser.parse_args(raw_args)
    ingest(args.root)
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
