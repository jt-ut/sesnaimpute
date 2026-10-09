"""
build_agb.py
====================================================================
ONE command: downloads + constants -> the AGB (GRAMS dusty-shell)
library and its register, in one act (curation consolidation,
2026-10-08, Unit AGB).

    python3.9 -m sesnaimpute.sed_models.build_agb <root.cfg path or data root>

READS: `<root>/downloads/grams/{grams_o,grams_c}.fits` (fetched if
absent; sha256-pinned, cache-hit skips the network -- `agb_curate.
fetch_grams`), and this package's own constants
(`LIBRARY_SAMPLING_SIGMA_LOG_VECTOR`, via `sed_models.sampling`).

WRITES ONLY: `<root>/agb/` (the library directory, replaced whole) and
`<root>/registers/agb_register.hdf5`. Nothing else on disk outlives
this process -- every intermediate (the convolution scratch directory
sedfitter's own `convolve_model_dir` needs on-disk flux.fits/
parameters.fits/models.conf to read, and the staged library
`sed_models.library.write_library` writes to before the swap) lives
under one `tempfile.mkdtemp()` directory that is removed in a `finally`
block, win or lose.

SCIENCE RANGE (GRAMS's own grids -- Sargent, Srinivasan & Meixner 2011,
ApJ 728, 93 for O-rich; Srinivasan, Sargent & Meixner 2011, A&A 532,
A54 for C-rich): TAU 1.28e-2 to 26 (O-rich), 2e-2 to 4 (C-rich); T_EFF
2100 to 3981 K (the upper bound is this library's own population cut,
`10 ** EVOLVED_LOGTE_MAX`, read from
`bms_prior.class_densities.class_density_star_family`, not GRAMS' own
grid edge, which runs four O-rich nodes and one C-rich node past it --
removed here, not relabelled; see `agb_curate.AGB_EVOLVED_TEFF_MAX_K`'s
docstring). Printed at run time below, with the raw curated set's own
coverage of it and the gaps.

STAGES, mirroring `agb_curate.curate_agb_model_set`'s shape but routing
the sampling step through the SHARED sampler (`sed_models.sampling`,
Unit S) instead of `agb_curate.greedy_rnet`/`sample_agb_templates` (the
old isotropic-metric r-net the curation audit found under-covers three
carbon-star models under the registers' own dark-band convention), and
the final write through the SHARED library/register writer
(`sed_models.library`) instead of six separate per-file calls:

  0  fetch_grams                       (idempotent; network only on a cache miss)
  1  select_agb_rows                   (B-2/B-3/B-4, agb_curate.py, unchanged)
  2  build_agb_rows                    (B-7/B-8 convert+gate+grid, unchanged)
  2.5 apply_evolved_teff_cut            (population's own cut, unchanged)
  3  assign_agb_names + sort           (B-9 naming, unchanged)
  4  sample in the shared whitened quotient space (sampling.py, radius=1.0)
  5  assemble the kept-template library (parameters/flux/classmap/convolved)
     and the represented-set members table, via EXISTING per-row science
     functions (`agb_curate.build_agb_uncertainty_row`,
     `write_agb_parameters_fits`, the published filter/zero-point
     provenance already in `agb_curate.AGB_PROVENANCE_FILT_ZP`) --
     nothing here recomputes a number `agb_curate.py` already computes
  6  library.write_library + library.write_register, staged then swapped
     in for the real `<root>/agb/` directory

ACCEPTANCE, asserted inline: coverage max_dist < 1.0 with zero
uncovered raw rows and packing >= 1.0 (`sampling.coverage`/`.packing`,
the registers' own whitened quotient-space radius); N_MEMBERS sums to
the raw curated count; every kept template has its own parameters.fits
row; FRAC_<subclass> columns sum to 1.0 per kept template; the register
opens with h5py and its NMODELS attribute equals the kept count; every
`parameters.fits` column (not just SUBCLASS) is on the register's
models group, by MODEL_NAME (`sed_models.library.write_register`'s A12
join, `curation-s` round 2, `29228b4`); TAU is on the models group and
groupable by SUBCLASS, with min(TAU) per SUBCLASS (the dusty-floor
number the pipeline reads) finite and positive for both O and C;
the bands group carries exactly 8 finite, positive VEGA_ZP_MJY zero
points; every `subclass_prob` row (O/C) sums to 1.0.
====================================================================
"""

import os
import shutil
import sys
import tempfile
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table

from sesnaimpute.sed_models import paths
from sesnaimpute.sed_models.constants import (
    CONVOLVED_BAND_NAMES, LOGD_STEP, POINT_SOURCE_APERTURE_AU, REFERENCE_DISTANCE_CM,
)
from sesnaimpute.sed_models import library, sampling
from sesnaimpute.sed_models.curate import model_io
from sesnaimpute.sed_models.curate.curve_of_growth import peak_and_floor
from sesnaimpute.sed_models.curate.model_convolution import build_convolved_bands
from sesnaimpute.sed_models.curate.agb_curate import (
    AGB_CLASS_LEGEND,
    AGB_EVOLVED_TEFF_MAX_K,
    AGB_PROVENANCE_FILT_ZP,
    AGB_SUBCLASS_LEGEND,
    AGB_ZH,
    BANDS8,
    CONVMETH_NOTE,
    GRAMS_LICENSE,
    MODELS_CONF_NAME,
    agb_band8_log10_fluxes,
    apply_evolved_teff_cut,
    assign_agb_names,
    build_agb_rows,
    build_agb_uncertainty_row,
    fetch_grams,
    load_grams_table,
    select_agb_rows,
    write_agb_parameters_fits,
)
from sesnaimpute.sed_models.curate.sps_curate import (
    MODEL_NAME_COLUMN_FORMAT,
    build_common_wavelength_grid,
)

#: The three carbon-star models the curation audit found uncovered under
#: the OLD isotropic metric/dark-band convention (curation_audit_2026-10-08);
#: the brief's "specific check" -- report their distance to their kept
#: representative under the registers' dark-band convention now shared
#: by this sampler.
_AUDIT_CARBON_CHECK = (
    "gramsC_t2600_r12_tau3.00e+00",
    "gramsC_t2600_r12_tau3.50e+00",
    "gramsC_t2800_r12_tau3.50e+00",
)

#: Static classmap provenance cards `agb_curate.write_agb_classmap_fits`
#: writes -- duplicated here (not imported, since that function writes
#: straight to a file rather than returning kwargs) rather than editing
#: agb_curate.py, which this unit was not asked to touch. The filter/
#: zero-point/grid-version cards ARE imported (`AGB_PROVENANCE_FILT_ZP`),
#: so nothing about the published provenance is retyped.
_AGB_CLASSMAP_PROVENANCE = (
    ("CLASS_SOURCE", "library declaration; all models here are this class"),
    ("SUBCLASS_SOURCE", "GRAMS grid of origin: O-rich or C-rich (B-10)"),
    ("SUBCLASS_REF", "Sargent+2011 ApJ 728,93; Srinivasan+2011 A&A 532,A54"),
    ("SUBCLASS_NOTE", "O=oxygen-rich silicate dust; C=carbon-rich AmC+SiC"),
    ("LEGEND_SOURCE", "descriptions authored; see SUBCLASS_REF"),
    ("LICENSE", GRAMS_LICENSE),
) + AGB_PROVENANCE_FILT_ZP

_RAW_PARAM_COLUMNS = ("T_EFF", "LOG[G]", "[Z/H]", "TAU", "TAU_WAVE_UM",
                     "R_IN", "L_SUN", "T_IN", "MLR_DUST")


def _resolve_sed_models_root(arg):
    """`arg` is either a config/overlay file (ingested via
    `paths.load_config`, then resolved against the `[inputs] sed_models`
    area -- the same area `agb_curate`'s old driver reads) or a bare data
    root directory, used directly."""
    if os.path.isfile(arg):
        paths.load_config(arg)
        return os.path.abspath(paths.input_dir("sed_models"))
    return os.path.abspath(arg)


def _peak_rss_mb():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1024.0


def _raw_params_table(rows):
    t = Table()
    t["T_EFF"] = np.array([r["teff"] for r in rows], dtype=np.float64)
    t["LOGG"] = np.array([r["logg"] for r in rows], dtype=np.float64)
    t["Z_H"] = np.array([AGB_ZH[r["chem"]] for r in rows], dtype=np.float64)
    t["TAU"] = np.array([r["tau"] for r in rows], dtype=np.float64)
    t["TAU_WAVE_UM"] = np.array([r["tau_wave_um"] for r in rows], dtype=np.float64)
    t["R_IN"] = np.array([r["rin"] for r in rows], dtype=np.float64)
    t["L_SUN"] = np.array([r["l_sun"] for r in rows], dtype=np.float64)
    t["T_IN"] = np.array([r["t_in"] for r in rows], dtype=np.float64)
    t["MLR_DUST"] = np.array([r["mlr_dust"] for r in rows], dtype=np.float64)
    return t


def build(root):
    t0 = time.perf_counter()
    grams_dir = os.path.join(root, "downloads", "grams")
    lib_dir = os.path.join(root, "agb")
    registers_dir = os.path.join(root, "registers")

    print(f"build_agb: downloads={grams_dir}  constants=sesnaimpute.sed_models.constants  "
          f"-> {lib_dir}, {registers_dir}/agb_register.hdf5")
    print("science range (GRAMS, Sargent+2011/Srinivasan+2011): "
          "TAU O-rich 1.28e-2..26, C-rich 2e-2..4; "
          f"T_EFF 2100..{AGB_EVOLVED_TEFF_MAX_K:.4f} K (population's own evolved cut, "
          "4 GRAMS nodes past it removed, not relabelled)")

    print("stage 0  fetch_grams")
    fetched = fetch_grams(grams_dir, timeout=600)
    for filename, path in fetched.items():
        print(f"    {filename}: {path}")
    o_table = load_grams_table(str(fetched["grams_o.fits"]))
    c_table = load_grams_table(str(fetched["grams_c.fits"]))

    print("stage 1  select")
    selection = select_agb_rows(o_table, c_table)

    print("stage 2  convert")
    wave_um_desc, freq_hz_desc = build_common_wavelength_grid()
    rows, closure_dropped, convert_report = build_agb_rows(
        o_table, c_table, selection["o_idx"], selection["c_idx"], freq_hz_desc)

    print("stage 2.5 population cut")
    rows, teff_cut_report = apply_evolved_teff_cut(rows)

    print("stage 3  name + sort")
    rows = assign_agb_names(rows)
    rows.sort(key=lambda r: r["name"])

    n_raw = len(rows)
    names_all = np.array([r["name"] for r in rows])
    chem_all = np.array([r["chem"] for r in rows])

    for chem in ("O", "C"):
        taus = np.array([r["tau"] for r in rows if r["chem"] == chem])
        teffs = np.array([r["teff"] for r in rows if r["chem"] == chem])
        print(f"    raw {chem}-rich: n={taus.size}  TAU [{taus.min():.4g},{taus.max():.4g}]  "
              f"T_EFF [{teffs.min():.0f},{teffs.max():.0f}] K")
    teff_all = np.array([r["teff"] for r in rows])
    print(f"    raw all: n={n_raw}  T_EFF [{teff_all.min():.0f},{teff_all.max():.0f}] K "
          f"(science range upper edge {AGB_EVOLVED_TEFF_MAX_K:.4f} K; "
          f"gap = {AGB_EVOLVED_TEFF_MAX_K - teff_all.max():.4f} K unreached, O-rich grid node "
          "spacing is 200 K so this is sub-node)")

    print("stage 4  sample (shared whitened quotient space, radius=1.0)")
    log10_raw = agb_band8_log10_fluxes(rows, wave_um_desc, bands=BANDS8)
    linear_raw = 10.0 ** log10_raw
    flux_cubes = {b: linear_raw[:, i:i + 1] for i, b in enumerate(BANDS8)}
    _, _, floor_linear_row, all_zero = peak_and_floor(flux_cubes)
    if all_zero.any():
        raise AssertionError(
            f"build_agb: {int(all_zero.sum())} raw row(s) have zero flux in every band")
    dark_mask = linear_raw < floor_linear_row[:, None]
    log10_flux = np.log10(np.maximum(linear_raw, floor_linear_row[:, None]))

    sigma_vec, projector = sampling.quotient_space()
    coords = sampling.whiten(log10_flux, dark_mask)
    kept_idx = sampling.r_net(coords, radius=1.0)
    rep_of, dist_to_rep = sampling.assign(coords, kept_idx)
    max_dist, n_uncovered = sampling.coverage(coords, kept_idx, radius=1.0)
    min_kept_dist = sampling.packing(coords, kept_idx)
    kept_sorted = np.unique(rep_of)
    assert set(kept_sorted.tolist()) == set(np.asarray(kept_idx).tolist()), (
        "build_agb: r_net's kept set and assign()'s representative set disagree")
    n_kept = int(kept_sorted.size)

    assert max_dist < 1.0, f"build_agb: coverage max_dist={max_dist} >= 1.0"
    assert n_uncovered == 0, f"build_agb: {n_uncovered} raw rows uncovered"
    assert min_kept_dist >= 1.0, f"build_agb: packing min_kept_dist={min_kept_dist} < 1.0"
    print(f"    {n_raw} raw -> {n_kept} kept; coverage max_dist={max_dist:.5f} (<1.0), "
          f"n_uncovered=0; packing min_kept_dist={min_kept_dist:.5f} (>=1.0)")

    print("    specific check: the three audit carbon-star models")
    for name in _AUDIT_CARBON_CHECK:
        hit = np.flatnonzero(names_all == name)
        if hit.size == 0:
            print(f"        {name}: not present in the curated raw set")
            continue
        i = int(hit[0])
        own_rep = i in set(kept_sorted.tolist())
        print(f"        {name}: dist_to_rep={dist_to_rep[i]:.6f} "
              f"(radius=1.0; own kept template: {own_rep})")

    raw_params = _raw_params_table(rows)
    members = sampling.members_table(names_all, rep_of, dist_to_rep, chem_all, raw_params)
    assert int(np.sum(members["N_MEMBERS"])) == n_raw, (
        "build_agb: members N_MEMBERS does not sum to the raw count")
    frac_sum = np.zeros(n_kept, dtype=np.float64)
    for code in ("C", "O"):
        frac_sum += np.asarray(members[f"FRAC_{code}"], dtype=np.float64)
    assert np.allclose(frac_sum, 1.0), "build_agb: FRAC columns do not sum to one"

    kept_rows = [rows[i] for i in kept_sorted]
    model_names = names_all[kept_sorted]

    print("stage 5  assemble kept-template library")
    scratch_dir = tempfile.mkdtemp(prefix="agb_build_")
    try:
        params_scratch_path = os.path.join(scratch_dir, "parameters_scratch.fits")
        write_agb_parameters_fits(kept_rows, params_scratch_path, overwrite=True)
        parameters_table = Table.read(params_scratch_path, hdu="PARAMETERS")
        assert len(parameters_table) == n_kept, (
            "build_agb: parameters table row count != kept template count")

        n_wav = len(wave_um_desc)
        values = np.empty((n_kept, 1, n_wav), dtype=np.float32)
        uncertainties = np.empty((n_kept, 1, n_wav), dtype=np.float32)
        for i, row in enumerate(kept_rows):
            values[i, 0, :] = row["values_desc"]
            uncertainties[i, 0, :] = build_agb_uncertainty_row(
                row["values_desc"], wave_um_desc, row["chem"])

        flux_kwargs = dict(
            wave_um_desc=wave_um_desc, freq_hz_desc=freq_hz_desc, values=values,
            uncertainties=uncertainties, distance_cm=REFERENCE_DISTANCE_CM,
            apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
            distance_comment="FAKE plug value, not real -- same 1 kpc frame as sps; "
                             "GRAMS' native 50 kpc already folded into VALUES",
            name_format=MODEL_NAME_COLUMN_FORMAT,
        )
        classmap_kwargs = dict(
            class_id="AGB",
            subclass=np.array([r["chem"] for r in kept_rows]),
            class_legend=AGB_CLASS_LEGEND,
            subclass_legend=AGB_SUBCLASS_LEGEND,
            provenance=_AGB_CLASSMAP_PROVENANCE,
            name_format=MODEL_NAME_COLUMN_FORMAT,
        )
        models_conf_kwargs = dict(
            name=MODELS_CONF_NAME, aperture_dependent=False, length_subdir=0,
            logd_step=LOGD_STEP, version=2,
        )

        # sedfitter's convolve_model_dir needs an on-disk model directory;
        # this is the ONLY thing staged outside library.write_library's own
        # write, and it lives entirely under scratch_dir.
        conv_scratch = os.path.join(scratch_dir, "conv")
        os.makedirs(conv_scratch, exist_ok=True)
        model_io.write_flux_cube(os.path.join(conv_scratch, "flux.fits"),
                                 names=model_names, **flux_kwargs)
        parameters_table.write(os.path.join(conv_scratch, "parameters.fits"), overwrite=True)
        model_io.write_models_conf(os.path.join(conv_scratch, "models.conf"),
                                   **models_conf_kwargs)
        build_convolved_bands(conv_scratch, tuple(CONVOLVED_BAND_NAMES), model_names,
                              CONVMETH_NOTE)

        convolved_kwargs = {}
        for band in CONVOLVED_BAND_NAMES:
            band_path = os.path.join(conv_scratch, "convolved", f"{band}.fits")
            with fits.open(band_path) as hdul:
                prim = hdul[0].header
                flux_tbl = hdul["CONVOLVED FLUXES"].data
                ap_tbl = hdul["APERTURES"].data
                convolved_kwargs[band] = dict(
                    total_flux_mjy=np.asarray(flux_tbl["TOTAL_FLUX"], dtype=np.float64),
                    total_flux_err_mjy=np.asarray(flux_tbl["TOTAL_FLUX_ERR"], dtype=np.float64),
                    apertures_au=np.asarray(ap_tbl["APERTURE"], dtype=np.float64),
                    filter_wavelength_um=float(prim["FILTWAV"]),
                    convmeth=prim.get("CONVMETH"),
                    convmeth_note=CONVMETH_NOTE,
                )
        print(f"    convolved {len(convolved_kwargs)} bands via sedfitter convolve_model_dir "
              "(scratch, discarded)")

        staged_lib_dir = os.path.join(scratch_dir, "agb_staged")
        library.write_library(
            staged_lib_dir, "agb", flux_kwargs, parameters_table, classmap_kwargs,
            model_names, members, convolved_kwargs, models_conf_kwargs)

        problems = model_io.validate_model_directory(staged_lib_dir, profile="agb")
        if problems:
            raise SystemExit("\n".join(problems))
        print("    output contract: validate_model_directory clean (profile='agb')")

        print(f"stage 6  swap in {lib_dir}")
        if os.path.isdir(lib_dir):
            shutil.rmtree(lib_dir)
        os.makedirs(os.path.dirname(lib_dir) or ".", exist_ok=True)
        shutil.move(staged_lib_dir, lib_dir)

        print("stage 7  write_register")
        register_path = library.write_register(registers_dir, "agb", lib_dir)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    def _decode(cols):
        return [c.decode() if isinstance(c, bytes) else str(c) for c in cols]

    import h5py
    with h5py.File(register_path, "r") as f:
        register_n_model = int(f.attrs["NMODELS"])
        model_cols = set(_decode(f["models"].attrs["columns"]))
        tau = np.asarray(f["models"]["TAU"], dtype=float)
        subclass = np.asarray(f["models"]["SUBCLASS"]).astype(str)
        band_cols = set(_decode(f["bands"].attrs["columns"]))
        vega_zp = np.asarray(f["bands"]["VEGA_ZP_MJY"], dtype=float)
        sp_cols = _decode(f["subclass_prob"].attrs["columns"])
        sp_subclass_cols = [c for c in sp_cols if c != "MODEL_NAME"]
        sp_sum = np.zeros(register_n_model, dtype=np.float64)
        for c in sp_subclass_cols:
            sp_sum += np.asarray(f["subclass_prob"][c], dtype=np.float64)

    assert register_n_model == n_kept, (
        f"build_agb: register NMODELS={register_n_model} != kept count {n_kept}")
    assert "SUBCLASS" in model_cols, "build_agb: register models group lacks SUBCLASS"
    parameters_cols_joined = set(parameters_table.colnames) - {"MODEL_NAME"}
    joined = parameters_cols_joined & model_cols
    assert joined == parameters_cols_joined, (
        f"build_agb: register models group is missing parameters.fits columns: "
        f"{sorted(parameters_cols_joined - joined)}")

    # coordinator's three checks (2026-10-08, after curation-s round 2):
    assert "TAU" in model_cols, "build_agb: register models group lacks TAU"
    min_tau_by_subclass = {}
    for code in ("O", "C"):
        vals = tau[subclass == code]
        assert vals.size > 0, f"build_agb: no {code}-rich rows in the register's models group"
        assert np.all(np.isfinite(vals)) and np.all(vals > 0), (
            f"build_agb: {code}-rich TAU has a non-finite or non-positive value")
        min_tau_by_subclass[code] = float(vals.min())

    assert "VEGA_ZP_MJY" in band_cols, "build_agb: register bands group lacks VEGA_ZP_MJY"
    assert vega_zp.size == 8, f"build_agb: bands group has {vega_zp.size} zero points, not 8"
    assert np.all(np.isfinite(vega_zp)) and np.all(vega_zp > 0), (
        "build_agb: a band zero point is non-finite or non-positive")

    assert np.allclose(sp_sum, 1.0), (
        "build_agb: subclass_prob rows do not sum to one "
        f"(min={sp_sum.min():.6f}, max={sp_sum.max():.6f})")

    print(f"register: {register_path}  NMODELS={register_n_model}  "
          f"models-group columns={sorted(model_cols)}")
    print(f"    models group: TAU grouped by SUBCLASS, min(TAU) per SUBCLASS (dusty floor) = "
          f"{min_tau_by_subclass}")
    print(f"    bands group: {vega_zp.size} VEGA_ZP_MJY zero points = {vega_zp.tolist()}")
    print(f"    subclass_prob: {len(sp_subclass_cols)} subclass columns "
          f"{sp_subclass_cols}, row sums in [{sp_sum.min():.6f}, {sp_sum.max():.6f}]")

    wall_s = time.perf_counter() - t0
    peak_mb = _peak_rss_mb()
    print(f"wall time: {wall_s:.1f} s   peak RSS: {peak_mb:.1f} MB")
    return dict(
        n_raw=n_raw, n_kept=n_kept, max_dist=max_dist, n_uncovered=n_uncovered,
        min_kept_dist=min_kept_dist, register_path=register_path, lib_dir=lib_dir,
        wall_s=wall_s, peak_mb=peak_mb,
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit(
            "usage: python3.9 -m sesnaimpute.sed_models.build_agb "
            "<root.cfg path or data root>")
    root = _resolve_sed_models_root(args[0])
    build(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
