"""
build_h2shock.py -- the one-command H2-shock (Paris-Durham) library line.

    python3.9 -m sesnaimpute.sed_models.build_h2shock CONFIG
    python3.9 -m sesnaimpute.sed_models.build_h2shock CONFIG --check

CONFIG is either a `paths.load_config`-style config/overlay file (e.g.
`config/root.cfg`) or a bare data-root directory (in which case
`<root>/sed_models` is used directly as the `sed_models` input area).

DOWNLOADS THIS LINE READS (and nothing else upstream of them):
  * Kristensen et al. 2023, A&A 675, A86 (VizieR J/A+A/675/A86) -- the
    Paris-Durham steady-state shock grid (h2exc/physgrid/coldens/h2width/
    h2opr tables), cached at
    `<sed_models>/downloads/h2shock/kristensen2023_pd_grid.npz`.
  * Roueff et al. 2019, A&A 630, A58 (VizieR J/A+A/630/A58, table2) -- the
    H2 rovibrational line list (wavelength, upper level, Einstein A),
    cached at `<sed_models>/downloads/h2shock/roueff2019_h2_lines.csv`.
Both are fetched once (idempotent: fetched only if the cache file is
absent) and then live under `downloads/`, never inside the product
directory -- so "rebuild from the downloads" is no longer vacuous for
this library (the audit's `code.md` sec. "H2SHOCK" defect).

PUBLISHED SCIENCE RANGE this library must cover, printed against the
raw (physically-cut) population at build time: the grid's own full node
ranges in log10 n_H [cm-3], v_s [km/s], b (B = b*sqrt(n_H) uG), G0 (Mathis
ISRF scaling), beside the measured protostellar-outflow-knot range
log10 n_H = 4-5 cm-3, v_s ~ 10-40 km/s (Nisini et al. 2010, A&A 518, L120;
Giannini et al. 2011, ApJ 738, 80) -- the POPULATION's range, not a cut:
the grid is shipped whole and uniformly weighted (contaminant_overview_
spec.md), so this is disclosure, not a gate.

STAGES (h2shock_curate.py's science, unchanged -- see that module for
derivations):
  1. fetch/cache the two downloads above; load; the h2exc-vs-coldens
     ingest check (`validate_level_columns_against_coldens`)
  2. match the Roueff lines to the grid's 150 tabulated levels
  3. the two physical cuts (supersonic; Crutcher B_max) -> the RAW
     (physically-cut) population
  4. SED-space sampling: `sesnacomplete.sed_models.sampling` (Unit S's
     ONE shared whitened quotient-space r-net), replacing this driver's
     former private `quotient_space()`/`greedy_coverage_r_net()` call --
     8-band log10 flux, divided per band by the project's sampling sigma
     (`constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR`), radius 1.0 in that
     whitened space. DARK_BANDS is 0 for every template in this library
     (a pure line spectrum has positive flux in all 8 bands at every
     raw model, verified below), so `dark_mask` is all-False and no
     floor branch is exercised.
  5. assemble flux/parameters/classmap/members/models.conf for the KEPT
     templates and write them, plus the register, in one act through
     `sesnacomplete.sed_models.library.write_library` /
     `.write_register` -- staged in `<lib_dir>/_work/staged/` and swapped
     into place only after the output-contract check passes; `_work/` is
     deleted on entry (in case a previous run died mid-build) and on
     exit, so no state file outlives this run.
  6. convolved/{band}.fits, written directly from the exact analytic
     line-spectrum value (`h2shock_curate.build_convolved_bands`), run
     against the staged directory before the swap. A line spectrum's
     band flux is the analytic one by construction, so sedfitter's own
     interpolate-then-integrate convolution is never run and there is no
     conservation gate to fall back from -- see that function's own
     docstring.

No subprocess is started by this line (no multiprocessing/joblib pool),
so there is nothing to kill; noted here rather than silently assumed.
"""

import argparse
import os
import resource
import shutil
import sys
import time

import numpy as np
from astropy.table import Table

from sesnaimpute.sed_models import paths
from sesnaimpute.sed_models.constants import POINT_SOURCE_APERTURE_AU, REFERENCE_DISTANCE_CM
from sesnaimpute.sed_models import library, sampling
from sesnaimpute.sed_models.curate import h2shock_curate as pdc
from sesnaimpute.sed_models.curate import model_io

CONTRACT_PROFILE = "h2shock"
CONVOLVED_BANDS = ("J", "H", "Ks", "I1", "I2", "I3", "I4", "M1")
RADIUS = 1.0
REGISTER_KEY = "h2shock"

# Measured protostellar-outflow-knot range, printed beside the grid's
# own range -- not a cut. Nisini et al. 2010 (A&A 518, L120); Giannini
# et al. 2011 (ApJ 738, 80).
MEASURED_LOG_NH_RANGE = (4.0, 5.0)
MEASURED_VS_RANGE_KMS = (10.0, 40.0)


def ingest(config_or_root):
    if os.path.isdir(config_or_root):
        root = paths.set_data_root(config_or_root)
        sed_models_dir = os.path.join(root, "sed_models")
        if os.path.isdir(sed_models_dir):
            paths.set_input_area("sed_models", sed_models_dir)
        return root
    return paths.load_config(config_or_root)


def downloads_dir():
    return os.path.join(paths.input_dir("sed_models"), "downloads", "h2shock")


def line_list_csv():
    return os.path.join(downloads_dir(), "roueff2019_h2_lines.csv")


def grid_npz():
    return os.path.join(downloads_dir(), "kristensen2023_pd_grid.npz")


def lib_dir():
    return paths.path_for("sed_models_h2shock")


def registers_dir():
    return paths.path_for("model_registers_dir")


def _flux_conv_header():
    conv = (f"I_nu x Omega({pdc.IRAC_BEAM_RADIUS_ARCSEC} arcsec)",
            "surface brightness through the IRAC beam")
    return conv


def check():
    out = lib_dir()
    problems = model_io.validate_model_directory(out, profile=CONTRACT_PROFILE)
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"h2shock: validate_model_directory clean at {out}")
    return {"library": "h2shock", "output_root": out, "status": "clean"}


def build():
    t0 = time.time()
    peak_kb_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ---- Stage 1: downloads (cached under downloads/h2shock/, not the
    # product directory -- "rebuild from the downloads" is no longer
    # vacuous for this library) ----
    dl_dir = downloads_dir()
    os.makedirs(dl_dir, exist_ok=True)
    line_list_path = line_list_csv()
    grid_path = grid_npz()
    if not os.path.exists(line_list_path):
        pdc.fetch_roueff_h2_line_list(line_list_path)
    if not os.path.exists(grid_path):
        pdc.fetch_pd_shock_grid(grid_path)
    lines = pdc.load_roueff_h2_line_list(line_list_path)
    grid = pdc.load_pd_shock_grid(grid_path)
    print(f"downloads  : {line_list_path}")
    print(f"           : {grid_path}")
    print(f"line list  : {len(lines)} transitions")
    print(f"shock grid : {len(grid)} models x {grid.level_v.size} levels")

    # ---- Stage 1b: the load-bearing ingest check ----
    stats = pdc.validate_level_columns_against_coldens(grid)
    print(f"ingest     : sum(g*exp(col))/N(H2) median {stats['ratio_median']:.4f} "
          f"(p5 {stats['ratio_p5']:.4f}, p95 {stats['ratio_p95']:.4f}), "
          f"within 5%: {stats['within_tolerance_frac']:.1%}")
    assert abs(stats["ratio_median"] - 1.0) < 0.05, (
        "h2exc reading is wrong -- refusing to build on it")

    # ---- Stage 2: matched lines ----
    subset, level_index = pdc.select_matched_lines(lines, grid)
    print(f"lines used : {len(subset)} (upper level present in the 150-level solution)")

    # ---- Stage 3: the two physical cuts -> the raw (physically-cut) population ----
    keep, report = pdc.physical_cut_mask(grid)
    idx_keep = np.flatnonzero(keep)
    n_raw = idx_keep.size
    print(f"cuts       : {report['n_total']} published"
          f" - {report['n_no_solution']} unconverged"
          f" - {report['n_subsonic']} subsonic"
          f" - {report['n_overfield']} over Crutcher B_max"
          f" = {report['n_kept']} shipped (raw population)")

    # ---- Published science range vs. the raw population's coverage ----
    log_nh_grid = np.log10(grid.column("nH"))
    v_s_grid = grid.column("Vs")
    b_grid = grid.column("b")
    g0_grid = grid.column("G0")
    log_nh_raw = log_nh_grid[idx_keep]
    v_s_raw = v_s_grid[idx_keep]
    in_nh = (log_nh_raw >= MEASURED_LOG_NH_RANGE[0]) & (log_nh_raw <= MEASURED_LOG_NH_RANGE[1])
    in_vs = (v_s_raw >= MEASURED_VS_RANGE_KMS[0]) & (v_s_raw <= MEASURED_VS_RANGE_KMS[1])
    print("science range (grid's own node range, raw population N={}):".format(n_raw))
    print(f"  log10 n_H [cm-3] : grid {log_nh_grid.min():.0f}-{log_nh_grid.max():.0f}"
          f"  vs measured {MEASURED_LOG_NH_RANGE[0]:.0f}-{MEASURED_LOG_NH_RANGE[1]:.0f}"
          f" (Nisini+2010, Giannini+2011) -- raw pop. inside: {in_nh.mean():.1%},"
          f" gap outside: {(~in_nh).mean():.1%}")
    print(f"  v_s [km/s]       : grid {v_s_grid.min():.0f}-{v_s_grid.max():.0f}"
          f"  vs measured {MEASURED_VS_RANGE_KMS[0]:.0f}-{MEASURED_VS_RANGE_KMS[1]:.0f}"
          f" -- raw pop. inside: {in_vs.mean():.1%}, gap outside: {(~in_vs).mean():.1%}")
    print(f"  b (B=b*sqrt(nH) uG): grid {b_grid.min():.2f}-{b_grid.max():.2f}"
          " (field already Crutcher-cut above)")
    print(f"  G0 (Mathis ISRF) : grid {g0_grid.min():.0f}-{g0_grid.max():.0f}"
          " (unbounded by design -- SESNA spans quiescent and OB-irradiated clouds alike)")

    # ---- Stage 4: SED-space sampling via the ONE shared sampler ----
    band_fluxes = pdc.compute_band_fluxes(subset, level_index, grid, pdc.SAMPLING_BANDS)
    stacked = np.column_stack([band_fluxes[b][idx_keep] for b in pdc.SAMPLING_BANDS])
    bad = ~np.isfinite(stacked) | (stacked <= 0)
    assert not np.any(bad), (
        f"{int(bad.sum())} of {stacked.size} raw-population band-flux entries are "
        f"non-positive or non-finite -- cannot place them in log10 SED space")
    log10_flux = np.log10(stacked)
    dark_mask = np.zeros_like(log10_flux, dtype=bool)  # DARK_BANDS==0 for all H2S templates

    coords = sampling.whiten(log10_flux, dark_mask)
    kept_index = sampling.r_net(coords, radius=RADIUS)
    rep_of, dist_to_rep = sampling.assign(coords, kept_index)
    max_dist, n_uncovered = sampling.coverage(coords, kept_index, radius=RADIUS)
    min_kept_dist = sampling.packing(coords, kept_index)
    print(f"sampling   : per-band whitened r-net (sampling.py) at radius {RADIUS} over "
          f"{n_raw} raw models -> {kept_index.size} kept templates")
    print(f"           : coverage max_dist={max_dist:.6f} (< {RADIUS}), "
          f"n_uncovered={n_uncovered}")
    print(f"           : packing min_kept_dist={min_kept_dist:.6f} (>= {RADIUS})")
    assert max_dist < RADIUS, "sampling.coverage: max_dist is not below the radius"
    assert n_uncovered == 0, "sampling.coverage: raw models left uncovered"
    assert min_kept_dist >= RADIUS, "sampling.packing: kept templates closer than the radius"

    kept_sorted = np.unique(rep_of)          # ascending positions into idx_keep/coords
    final_idx = idx_keep[kept_sorted]        # global grid rows of the kept templates
    n_kept = final_idx.size

    # ---- members table, from the raw population, via sampling.members_table ----
    names_raw = pdc.format_pd_model_names(grid, idx_keep)
    params_raw = grid.params[idx_keep]
    derived_raw = {k: v[idx_keep] for k, v in grid.derived.items()}
    i_h2_raw = pdc.compute_h2_1_0_s1_intensity(subset, level_index, grid, idx_keep)
    subclass_raw = np.array([pdc.SHOCK_TYPE_LABELS[float(v)] for v in params_raw[:, 6]])

    members_params = Table()
    for ax in pdc.SAMPLING_PARAM_AXES:
        members_params[pdc.AXIS_FITS_NAMES[ax]] = params_raw[:, pdc.PARAM_COLUMNS.index(ax)]
    for k, v in derived_raw.items():
        members_params[k] = v
    members_params["I_H2_1_0_S1"] = i_h2_raw

    members = sampling.members_table(names_raw, rep_of, dist_to_rep, subclass_raw, members_params)
    assert int(np.sum(members["N_MEMBERS"])) == n_raw, (
        "members.N_MEMBERS does not sum to the raw count")
    frac_cols = [c for c in members.colnames if c.startswith("FRAC_")]
    frac_sum = np.sum([np.asarray(members[c]) for c in frac_cols], axis=0)
    assert np.allclose(frac_sum, 1.0), "FRAC_* columns do not sum to one per row"
    assert len(members) == n_kept, "members table row count disagrees with the kept count"
    print(f"members    : {n_kept} kept templates, {n_raw} raw members, "
          f"FRAC columns {frac_cols}")

    # ---- Stage 5: assemble flux/parameters/classmap for the kept templates ----
    model_names_kept = pdc.format_pd_model_names(grid, final_idx)
    assert np.array_equal(np.asarray(members["MODEL_NAME"]).astype(str),
                          model_names_kept.astype(str)), (
        "members.MODEL_NAME disagrees with the kept-template name derivation")

    line_fluxes_mjy_kept = pdc.compute_line_fluxes_mjy(subset, level_index, grid, final_idx)
    model_set = pdc.prepare_pd_model_arrays(subset, level_index, grid, final_idx,
                                            line_fluxes_mjy_kept)
    finite = np.isfinite(model_set.values_mjy)
    assert np.all(finite), "non-finite values in the flux cube"

    params_table = Table()
    # Fixed-width 34-char MODEL_NAME, matching flux.fits's MODEL_NAME_FORMAT
    # (C10: every cube's MODEL_NAME column must share one width) -- the
    # dtype itemsize, not the string content, drives the FITS column
    # format astropy.table.Table.write chooses.
    params_table["MODEL_NAME"] = np.asarray(model_names_kept, dtype=f"<U{int(pdc.MODEL_NAME_FORMAT[:-1])}")
    params_table.meta["EXTNAME"] = "PARAMETERS"  # C11
    for ax in pdc.PARAM_COLUMNS[:6]:
        params_table[pdc.AXIS_FITS_NAMES[ax]] = model_set.params[:, pdc.PARAM_COLUMNS.index(ax)]
    params_table["B_MICROGAUSS"] = model_set.b_microgauss
    params_table["SHOCK_TYPE"] = np.array(
        [pdc.SHOCK_TYPE_LABELS[float(v)] for v in model_set.params[:, 6]])
    for fits_col, _table_name, _viz_col, _unit, _comment in pdc.DERIVED_SPECS:
        params_table[fits_col] = model_set.derived[fits_col]
    params_table["I_H2_1_0_S1"] = model_set.i_h2_1_0_s1
    assert len(params_table) == n_kept, "every kept template must have its own parameters row"

    subclass_kept = np.array(params_table["SHOCK_TYPE"])
    conv = _flux_conv_header()
    flux_kwargs = dict(
        wave_um_desc=model_set.wave_um_desc,
        freq_hz_desc=model_set.freq_hz_desc,
        values=model_set.values_mjy,
        distance_cm=REFERENCE_DISTANCE_CM,
        apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
        uncertainties=model_set.uncertainties_mjy,
        distance_comment="structural plug, not load-bearing here",
        values_header_extras={"FLUXCONV": conv},
        uncertainties_header_extras={
            "FLUXCONV": conv,
            "COMMENT": "exact zeros: analytic models, no stochastic noise",
        },
    )
    classmap_kwargs = dict(
        class_id=pdc.MODEL_CLASS,
        subclass=subclass_kept,
        class_legend=pdc.CLASS_LEGEND,
        subclass_legend=pdc.SUBCLASS_LEGEND,
        provenance=(
            ("CLASS_SOURCE", "library declaration; all models here are this class"),
            ("SUBCLASS_SOURCE", "Paris-Durham physgrid Type column, unmodified"),
            ("SUBCLASS_REF", "VizieR J/A+A/675/A86 (Kristensen+ 2023)"),
            ("SUBCLASS_NOTE", "an OUTCOME of the shock solution, not a grid axis"),
            ("LEGEND_SOURCE", "codes from the catalogue; descriptions authored here"),
            ("INPUT_GRID", "downloads/h2shock/kristensen2023_pd_grid.npz"),
            ("INPUT_LINES", "downloads/h2shock/roueff2019_h2_lines.csv"),
        ),
    )
    models_conf_kwargs = dict(name=pdc.DEFAULT_MODEL_SET_NAME, aperture_dependent=False)

    # ---- stage the build in scratch, under the library dir's _work/,
    # deleted on entry (a dead prior run) and on exit ----
    out = lib_dir()
    work_root = os.path.join(out, "_work")
    if os.path.isdir(work_root):
        shutil.rmtree(work_root)
    os.makedirs(work_root)
    staged = os.path.join(work_root, "staged")
    os.makedirs(staged)

    try:
        library.write_library(
            staged, REGISTER_KEY, flux=flux_kwargs, parameters=params_table,
            classmap=classmap_kwargs, model_names=model_names_kept, members=members,
            convolved={}, models_conf=models_conf_kwargs)
        print(f"wrote      : flux.fits, parameters.fits, classmap.fits, models.conf, "
              f"members.fits -> {staged} (staged)")

        # ---- Stage 6: convolved/{band}.fits -- the exact analytic line-
        # spectrum convolution, unconditional (the only path for a line
        # spectrum; see h2shock_curate.build_convolved_bands) ----
        pdc.build_convolved_bands(
            staged, model_names_kept, subset, line_fluxes_mjy_kept,
            CONVOLVED_BANDS, POINT_SOURCE_APERTURE_AU)
        print("convolved  : shipped via the analytic path, structurally verified.")

        problems = model_io.validate_model_directory(staged, profile=CONTRACT_PROFILE)
        if problems:
            raise SystemExit("\n".join(problems))
        print("output contract: validate_model_directory clean (staged)")

        # ---- replace the library directory's content with the staged build ----
        for entry in os.listdir(out):
            if entry == "_work":
                continue
            path = os.path.join(out, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        for entry in os.listdir(staged):
            shutil.move(os.path.join(staged, entry), os.path.join(out, entry))
        print(f"replaced   : {out} with the staged build")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    # ---- register, in one act, from the now-live library directory ----
    reg_path = library.write_register(registers_dir(), REGISTER_KEY, out)
    print(f"register   : {reg_path}")

    # ---- register identities (Unit S round 2, curation-s 29228b4: A12
    # join, per-band Vega zero points, members-sourced subclass_prob) ----
    import h5py
    with h5py.File(reg_path, "r") as f:
        model_cols = list(f["models"].keys())
        n_model_reg = f["models"]["MODEL_NAME"].shape[0]
        band_names = [b.decode() if isinstance(b, bytes) else b
                     for b in f["bands"]["BAND"][()]]
        vega_zp = np.asarray(f["bands"]["VEGA_ZP_MJY"])
        prob_cols = [c for c in f["subclass_prob"].keys() if c != "MODEL_NAME"]
        prob_sum = np.sum([np.asarray(f["subclass_prob"][c]) for c in prob_cols], axis=0)

    assert n_model_reg == n_kept, (
        f"register models count {n_model_reg} != kept count {n_kept}")
    print(f"register check: opens with h5py, n_model={n_model_reg} (== kept count {n_kept})")

    param_cols_expected = set(params_table.colnames) - {"MODEL_NAME"}
    param_cols_present = param_cols_expected & set(model_cols)
    assert param_cols_present == param_cols_expected, (
        f"register /models is missing parameters.fits columns: "
        f"{sorted(param_cols_expected - param_cols_present)}")
    assert "SUBCLASS" in model_cols, "register /models lacks SUBCLASS"
    assert "I_H2_1_0_S1" in model_cols, "register /models lacks I_H2_1_0_S1"
    print(f"register check: /models carries all {len(param_cols_expected)} "
          f"parameters.fits columns (incl. I_H2_1_0_S1) plus SUBCLASS")

    assert len(band_names) == 8 and vega_zp.shape == (8,), (
        f"register /bands does not carry 8 VEGA_ZP_MJY entries: {vega_zp.shape}")
    assert np.all(np.isfinite(vega_zp)) and np.all(vega_zp > 0), (
        "register /bands VEGA_ZP_MJY has a non-finite or non-positive entry")
    print(f"register check: /bands carries {vega_zp.size} VEGA_ZP_MJY zero points "
          f"({dict(zip(band_names, vega_zp.tolist()))})")

    assert len(prob_cols) == len(frac_cols), (
        f"register /subclass_prob column count {len(prob_cols)} != "
        f"members FRAC_ column count {len(frac_cols)}")
    assert np.allclose(prob_sum, 1.0), (
        "register /subclass_prob rows do not sum to one")
    print(f"register check: /subclass_prob ({sorted(prob_cols)}) is float64, "
          f"all {prob_sum.size} rows sum to 1.0 (min={prob_sum.min():.10f}, "
          f"max={prob_sum.max():.10f})")

    import sesnaimpute  # noqa: F401 -- the package imports
    print("package import: sesnaimpute imports cleanly")

    wall_s = time.time() - t0
    peak_kb_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "n_raw": int(n_raw),
        "n_kept": int(n_kept),
        "coverage_max_dist": float(max_dist),
        "n_uncovered": int(n_uncovered),
        "packing_min_dist": float(min_kept_dist),
        "wall_s": wall_s,
        "peak_rss_bytes": int(peak_kb_end),
    }
    print(f"\nH2-shock (Paris-Durham) library complete at {out}: "
          f"{n_raw} raw (physically-cut) models sampled to {n_kept} kept templates")
    print(f"wall time  : {wall_s:.1f} s")
    print(f"peak RSS   : {peak_kb_end / 1e6:.1f} MB (ru_maxrss; bytes on macOS)")
    return out, report


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m sesnaimpute.sed_models.build_h2shock",
        description="Build the H2-shock (Paris-Durham) contaminant model library "
                    "and its register, from the downloads, in one command.")
    parser.add_argument("config", help="project configuration file, or a bare data-root dir")
    parser.add_argument("--check", action="store_true",
                        help="read-only validation of the installed product")
    args = parser.parse_args(raw_args)
    ingest(args.config)
    if args.check:
        check()
        return 0
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
