"""
sesnaimpute.sed_models.build_sps
====================================================================
ONE command: downloads -> sps/ (the kept, 1-sigma-sampled stellar-
photosphere class library) + sps/_raw/ (the full raw splice) +
registers/sps_register.hdf5.

    python3.9 -m sesnaimpute.sed_models.build_sps <root.cfg path or data root>

Reads, read-only:
    downloads/ck03               -- CK03/ATLAS9 (Castelli & Kurucz 2003)
    downloads/btsettl_cifist     -- BT-Settl CIFIST2011_2015 (Allard,
                                     Homeier & Freytag 2012)
    downloads/btsettl_agss       -- BT-Settl AGSS2009 (same code paper,
                                     Asplund et al. 2009 abundances)
All three already sha256-pinned on disk (`sps_curate.py`'s own fetchers
are NOT invoked here -- this line assumes the caches exist).

Writes, nothing else:
    sps/            the KEPT templates (flux.fits, parameters.fits,
                     classmap.fits, convolved/{band}.fits, members.fits,
                     models.conf) -- the library a fitter loads.
    sps/_raw/       the full raw splice (flux.fits, parameters.fits,
                     classmap.fits, convolved/{band}.fits, models.conf)
                     -- NOT a state file: `build/pahc.py` and
                     `build/yso.py` read it as their photosphere host
                     grid. Part of this line's product.
    registers/sps_register.hdf5

SCIENCE RANGE this library is required to cover (the splice's own,
Castelli & Kurucz 2003 for CK03; Allard et al. 2012 for the BT-Settl
grids): T_eff 1,200-50,000 K, log g -0.5 to +5.5, [M/H] -2.5 to +0.5.
Printed against the raw grid's actual extent and internal gaps below.

STAGES
  1. the raw splice (`sed_models_curate.sps_curate.curate_sps_model_set`
     -- the science module, unchanged: stage 0 fetches are assumed
     already run, stages 1-5 build the CK03+CIFIST+AGSS splice on the
     common R=300 grid, convolve the 8 SESNA bands through
     `model_convolution` -- sedfitter's own `convolve_model_dir`, which
     is the "flat" normalisation convention (`Filter.normalize()`,
     integral over nu = 1; see `model_convolution.
     analytic_band_flux_for_line_spectrum`'s docstring for why that name
     is used project-wide), the correct path for a sampled CONTINUUM
     library -- and write classmap.fits (SUBCLASS = MK letter from
     T_eff, dwarf sequence, all log g).
  2. read the raw splice back (`sps_curate._read_full_sps_library`; no
     physics re-derived) for T_EFF/LOGG/Z_H/SOURCE/SUBCLASS and the
     8-band reference flux f_ref every raw model's SED-space coordinate
     needs. (`sps_curate.py` now writes and reads these two columns as
     `LOGG`/`Z_H` everywhere, including `sps/_raw/parameters.fits` --
     not `LOG[G]`/`[Z/H]` -- HDF5-safe at the source, per
     `sesnaimpute.sed_models.library._assert_hdf5_safe_names`.)
  3. the shared sampler (`sesnaimpute.sed_models.sampling`): the B0
     floor per model (`curve_of_growth.peak_and_floor`, the same
     floor-before-log convention `density.derive` applies to f_ref),
     whiten, r-net at radius 1.0, nearest-kept assignment, coverage,
     packing, members table (FRAC by SUBCLASS letter, ranges over
     T_EFF/LOGG/Z_H).
  4. write the kept library and its register in one act
     (`sesnaimpute.sed_models.library.write_library` /
     `.write_register`).

Everything is staged under `sps/_work/` and that directory is removed
on entry AND on exit -- no state file outlives the run. `sps/_raw/` is
a product, not state, and is written directly (not staged then moved)
since it is itself `curate_sps_model_set`'s normal output location.
"""

import argparse
import os
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

from sesnaimpute.sed_models import paths
from sesnaimpute.sed_models.constants import POINT_SOURCE_APERTURE_AU, BANDS
from sesnaimpute.sed_models import sampling, library
from sesnaimpute.sed_models.curate import curve_of_growth as cog
from sesnaimpute.sed_models.curate.sps_curate import (
    CLASS_LEGEND,
    CONVMETH_NOTE,
    CONVOLVED_BANDS_TO_BUILD,
    DISTANCE_COMMENT,
    LOGD_STEP,
    MODEL_CLASS,
    MODEL_NAME_COLUMN_FORMAT,
    MODELS_CONF_NAME,
    SUBCLASS_LEGEND,
    _read_full_sps_library,
    curate_sps_model_set,
)

#: the splice's own published science range (module docstring; Castelli &
#: Kurucz 2003, Allard et al. 2012) -- design intent, not a benchmark.
SCIENCE_RANGE = {
    "T_EFF": (1200.0, 50000.0),
    "LOGG": (-0.5, 5.5),
    "Z_H": (-2.5, 0.5),
}

SAMPLING_RADIUS = 1.0


# ---------------------------------------------------------------------------
# paths -- this unit writes ONLY sps/ (+ sps/_raw/) and registers/sps_register.hdf5
# ---------------------------------------------------------------------------

def output_root():
    """The KEPT library -- what a fitter loads."""
    return Path(paths.path_for("sed_models_sps"))


def raw_root():
    """The full raw splice -- `build/pahc.py` and `build/yso.py`'s
    photosphere host grid. A product, not a state file."""
    return output_root() / "_raw"


def work_root():
    """Build scratch. Deleted on entry and on exit -- no state file
    outlives the run.

    Deliberately a SIBLING of `output_root()`, not nested under it:
    this unit replaces the whole library directory wholesale (the
    SPS-directory-inversion fix -- the kept set moves to the top level,
    the raw splice moves to `_raw/`), so the scratch area staging that
    replacement cannot live inside the directory being removed, or the
    removal step destroys the very build it is about to swap in."""
    return output_root().parent / "_sps_build_work"


def registers_dir():
    return Path(paths.path_for("model_registers_dir"))


def downloads_root():
    return Path(paths.input_dir("sed_models")) / "downloads"


def source_root():
    return downloads_root() / "ck03"


def cifist_download_dir():
    return downloads_root() / "btsettl_cifist"


def agss_download_dir():
    return downloads_root() / "btsettl_agss"


def _ingest(root_arg):
    """`root_arg` is either a root.cfg-style INI file (ingested via
    `paths.load_config`) or a bare data-root directory (ingested
    directly via `paths.set_data_root` + `paths.set_input_area`, with
    `sed_models` -- the one input area this driver reads -- assumed at
    `<data_root>/sed_models`, the project's own convention, matching
    root.cfg)."""
    p = Path(root_arg)
    if p.is_file():
        return paths.load_config(str(p))
    data_root = paths.set_data_root(str(p))
    paths.set_input_area("sed_models", str(Path(data_root) / "sed_models"))
    return data_root


def _peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    return rss / 1024.0 if platform.system() == "Linux" else rss / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# science-range coverage of the raw grid -- printed, not asserted (the
# brief's acceptance asserts are about covering/packing/members, not the
# published range: edges and interior holes are a disclosure).
# ---------------------------------------------------------------------------

def _print_science_range_coverage(teff, logg, zh):
    print("science range coverage (raw grid against the splice's own "
          "published range):")
    for label, arr, (lo, hi) in (
            ("T_EFF (K)", teff, SCIENCE_RANGE["T_EFF"]),
            ("LOGG", logg, SCIENCE_RANGE["LOGG"]),
            ("Z_H", zh, SCIENCE_RANGE["Z_H"])):
        below = int(np.sum(arr < lo))
        above = int(np.sum(arr > hi))
        print(f"  {label}: target [{lo:g}, {hi:g}]  raw grid spans "
              f"[{arr.min():g}, {arr.max():g}]  "
              f"{below} rows below lo, {above} rows above hi "
              f"(of {arr.size})")

    # Interior gaps -- the grid is not rectangular (D-11..D-25): report
    # the T_eff range actually populated at each distinct log g, and the
    # Z_H values actually populated below 3500 K, as a MEASUREMENT of
    # the installed grid rather than a restatement of the design docs.
    print("  interior shape (measured, not asserted):")
    for g in sorted(set(np.round(logg, 2).tolist())):
        mask = np.isclose(logg, g)
        print(f"    log g={g:+.1f}: T_eff [{teff[mask].min():.0f}, "
              f"{teff[mask].max():.0f}] K, n={int(mask.sum())}")
    cool = teff < 3500.0
    zh_cool = sorted(set(np.round(zh[cool], 3).tolist()))
    print(f"  Z_H values present at T_eff < 3500 K: {zh_cool} "
          f"({int(cool.sum())} of {teff.size} rows)")


def build(root_arg):
    t0 = time.time()
    _ingest(root_arg)

    out = output_root()
    raw = raw_root()
    regs_dir = registers_dir()
    work = work_root()

    # No state file outlives the run -- clear any leftover scratch from
    # a previous, interrupted run before building anything.
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    print(f"build_sps: downloads={downloads_root()} -> {out}  "
          f"(+ {raw})  + {regs_dir / 'sps_register.hdf5'}")

    try:
        # ---- Stage 1: the raw splice, in a scratch subdirectory, via the
        # unmodified science module (curate_sps_model_set) -----------------
        raw_scratch = work / "raw"
        _, report = curate_sps_model_set(
            source_root(), cifist_download_dir(), raw_scratch,
            agss_download_dir=agss_download_dir(), overwrite=True,
            convolved_bands=CONVOLVED_BANDS_TO_BUILD, verbose=True)
        n_raw = report["n_merged"]
        print(f"raw splice: {n_raw} models "
              f"({report['n_ck03_total']} CK03 - {report['n_ck03_culled']} "
              f"culled + {report['n_cifist_kept']} CIFIST + "
              f"{report['n_agss_kept']} AGSS)")

        # ---- Stage 2: read the raw splice back -- no physics re-derived --
        raw_data = _read_full_sps_library(raw_scratch,
                                          convolved_bands=CONVOLVED_BANDS_TO_BUILD)
        names = raw_data["names"]
        teff, logg, zh = raw_data["teff"], raw_data["logg"], raw_data["zh"]
        source = raw_data["source"]
        subclass = raw_data["subclass"]
        f_ref = raw_data["f_ref"]                       # (n_raw, 8) mJy
        wave_um_desc = raw_data["wave_um_desc"]
        freq_hz_desc = raw_data["freq_hz_desc"]
        values = raw_data["values"]                     # (n_raw, 1, n_wav)
        uncertainties = raw_data["uncertainties"]
        distance_cm = raw_data["distance_cm"]

        assert names.size == n_raw, (
            f"raw splice readback disagrees with the build report: "
            f"{names.size} rows read vs {n_raw} reported")

        _print_science_range_coverage(teff, logg, zh)

        # ---- Stage 3: the shared sampler ----------------------------------
        sigma_vec, projector = sampling.quotient_space()
        flux_cubes = {band: f_ref[:, j:j + 1]
                      for j, band in enumerate(CONVOLVED_BANDS_TO_BUILD)}
        peak, floor_dex, floor_linear, all_zero = cog.peak_and_floor(flux_cubes)
        assert not np.any(all_zero), (
            "build_sps: a raw model has zero flux in every band -- no "
            "B0 floor to anchor to; refusing to sample")
        dark_mask = f_ref < floor_linear[:, None]
        log10_flux = np.log10(np.maximum(f_ref, floor_linear[:, None]))
        n_dark_rows = int(np.any(dark_mask, axis=1).sum())
        print(f"B0 floor: {n_dark_rows} of {n_raw} raw rows have >=1 band "
              f"at the per-model floor (expect 0 -- no SPS photosphere is "
              f"dark in any band)")

        coords = sampling.whiten(log10_flux, dark_mask)
        kept_index = sampling.r_net(coords, radius=SAMPLING_RADIUS)
        rep_of, dist_to_rep = sampling.assign(coords, kept_index)
        max_dist, n_uncovered = sampling.coverage(coords, kept_index,
                                                  radius=SAMPLING_RADIUS)
        min_kept_dist = sampling.packing(coords, kept_index)

        n_kept = int(kept_index.size)
        print(f"sampling: whitened quotient space d=6, radius=1.0 -- "
              f"kept {n_kept} of {n_raw} raw models")
        print(f"coverage: max_dist={max_dist:.6f} (< 1.0 required), "
              f"n_uncovered={n_uncovered} (0 required)")
        print(f"packing : min kept-kept distance={min_kept_dist:.6f} "
              f"(>= 1.0 required)")

        assert max_dist < 1.0, f"covering failed: max_dist={max_dist} >= 1.0"
        assert n_uncovered == 0, (
            f"covering failed: {n_uncovered} raw models uncovered")
        assert min_kept_dist >= 1.0, (
            f"packing failed: min_kept_dist={min_kept_dist} < 1.0")

        # kept rows, in ascending raw-index order -- the SAME order
        # members_table derives internally (np.unique(rep_of)), so the
        # library's files and members.fits agree row-for-row.
        kept_sorted = np.unique(rep_of)
        assert kept_sorted.size == n_kept, (
            f"kept_index ({n_kept}) and np.unique(rep_of) "
            f"({kept_sorted.size}) disagree on the kept set")

        kept_names = np.array(names[kept_sorted], dtype="<U30")
        kept_teff = teff[kept_sorted]
        kept_logg = logg[kept_sorted]
        kept_zh = zh[kept_sorted]
        kept_source = source[kept_sorted]
        kept_subclass = subclass[kept_sorted]
        kept_values = values[kept_sorted]
        kept_uncertainties = uncertainties[kept_sorted]
        kept_f_ref = f_ref[kept_sorted]

        # Rayleigh-Jeans collapse: physics, not a defect -- a Planck
        # function at 20,000+ K is featureless over an 8-band NIR/MIR
        # baseline, so many hot raw rows share one kept representative.
        params_raw = Table()
        params_raw["T_EFF"] = teff
        params_raw["LOGG"] = logg
        params_raw["Z_H"] = zh
        members = sampling.members_table(names, rep_of, dist_to_rep,
                                         subclass, params_raw)

        assert int(np.sum(members["N_MEMBERS"])) == n_raw, (
            f"members N_MEMBERS sums to {int(np.sum(members['N_MEMBERS']))}, "
            f"not the raw count {n_raw}")
        frac_cols = [c for c in members.colnames if c.startswith("FRAC_")]
        frac_sums = np.sum([np.asarray(members[c], dtype=float)
                            for c in frac_cols], axis=0)
        assert np.allclose(frac_sums, 1.0), (
            f"FRAC_* columns do not sum to 1.0 on every row "
            f"(min={frac_sums.min()}, max={frac_sums.max()})")

        biggest = int(np.argmax(members["N_MEMBERS"]))
        print(f"Rayleigh-Jeans collapse: the largest kept template, "
              f"{members['MODEL_NAME'][biggest]!r}, represents "
              f"{int(members['N_MEMBERS'][biggest])} of {n_raw} raw rows "
              f"(T_EFF_MIN={members['T_EFF_MIN'][biggest]:.0f} K, "
              f"T_EFF_MAX={members['T_EFF_MAX'][biggest]:.0f} K) -- physics "
              f"(a hot Planck tail is featureless across 8 NIR/MIR bands), "
              f"not a sampling defect.")

        # ---- Stage 4: write the kept library + register in one act -------
        flux_kwargs = dict(
            wave_um_desc=wave_um_desc, freq_hz_desc=freq_hz_desc,
            values=kept_values, distance_cm=distance_cm,
            distance_comment=DISTANCE_COMMENT,
            apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
            uncertainties=kept_uncertainties,
            name_format=MODEL_NAME_COLUMN_FORMAT,
        )
        classmap_kwargs = dict(
            class_id=MODEL_CLASS,
            subclass=kept_subclass,
            class_legend=CLASS_LEGEND,
            subclass_legend=SUBCLASS_LEGEND,
            provenance=(
                ("CLASS_SOURCE", "library declaration; all models here are this class"),
                ("SUBCLASS_SOURCE", "MK letter from T_EFF; dwarf sequence, all log g"),
                ("SUBCLASS_REF", "Pecaut & Mamajek 2013 ApJS 208,9; tab v2022.04.16"),
                ("SUBCLASS_NOTE", "cuts = midpoints of adjacent X9V/X0V anchors"),
                ("LEGEND_SOURCE", "Teff cuts from the table; descriptions authored"),
                ("SUBCLASS_EXT", "D-17 anchors: M9V2380 L0V2270 L9V1370 T0V1255K"),
                ("SAMPLING", "1-sigma whitened r-net, radius=1.0, sampling.py"),
            ),
            name_format=MODEL_NAME_COLUMN_FORMAT,
        )
        convolved_kwargs = {
            band: dict(
                total_flux_mjy=kept_f_ref[:, j],
                apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
                filter_wavelength_um=BANDS[band].wvl_effective_um,
                convmeth="sedfitter", convmeth_note=CONVMETH_NOTE,
            )
            for j, band in enumerate(CONVOLVED_BANDS_TO_BUILD)
        }
        models_conf_kwargs = dict(
            name=MODELS_CONF_NAME, aperture_dependent=False,
            length_subdir=0, logd_step=LOGD_STEP, version=2,
        )

        parameters_kept = Table()
        parameters_kept["MODEL_NAME"] = kept_names
        parameters_kept["T_EFF"] = kept_teff.astype(np.float32)
        # LOGG/Z_H: the shared convention `sps_curate.py` now writes and
        # reads everywhere (including sps/_raw/parameters.fits) -- see
        # `sesnaimpute.sed_models.library._assert_hdf5_safe_names`,
        # which the shared writer now enforces on both parameters and
        # members column names before any write.
        parameters_kept["LOGG"] = kept_logg.astype(np.float32)
        parameters_kept["Z_H"] = kept_zh.astype(np.float32)
        parameters_kept["SOURCE"] = kept_source
        for col in parameters_kept.colnames:
            assert len(parameters_kept[col]) == n_kept, (
                f"parameters.fits column {col} length mismatch")
        assert set(kept_names) <= set(names), (
            "every kept template must be one of the raw rows")

        kept_scratch = work / "kept"
        library.write_library(
            str(kept_scratch), "sps", flux_kwargs, parameters_kept, classmap_kwargs,
            kept_names, members, convolved_kwargs, models_conf_kwargs)
        print(f"staged kept library -> {kept_scratch} ({n_kept} models)")

        # ---- replace the library directory, staged-then-swapped: never
        # edit `out` in place. `out` today holds the OLD layout (raw
        # splice at top level, the old isotropic-metric sample in a
        # `sampled_1sigma/` subdirectory) -- both retired by this line,
        # so the whole directory is removed and rebuilt from scratch. ----
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(kept_scratch), str(out))
        # raw_scratch (the full splice) becomes the permanent sps/_raw/
        # product -- not state, see module docstring.
        shutil.move(str(raw_scratch), str(raw))
        print(f"swapped in: {out} (kept) and {raw} (raw splice)")

        register_path = library.write_register(str(regs_dir), "sps", str(out))
        print(f"wrote register -> {register_path}")

        import h5py
        with h5py.File(register_path, "r") as f:
            models_cols = set(f["models"].keys())
            n_model_register = f["models"]["MODEL_NAME"].shape[0]

            # A12: every parameters.fits column joined into /models.
            for param_col in ("T_EFF", "LOGG", "Z_H"):
                assert param_col in models_cols, (
                    f"register /models is missing the A12 join column "
                    f"{param_col!r}; have {sorted(models_cols)}")

            # /bands carries one Vega zero point per band.
            vega_zp = np.asarray(f["bands"]["VEGA_ZP_MJY"])
            assert vega_zp.shape[0] == len(CONVOLVED_BANDS_TO_BUILD), (
                f"register /bands VEGA_ZP_MJY has {vega_zp.shape[0]} rows, "
                f"not {len(CONVOLVED_BANDS_TO_BUILD)} (one per band)")
            assert np.all(np.isfinite(vega_zp)), (
                "register /bands VEGA_ZP_MJY has non-finite zero points")

            # subclass_prob (members-sourced, FRAC_<subclass>) sums to 1.
            assert "subclass_prob" in f, (
                "register has no /subclass_prob group -- members.fits "
                "FRAC_<subclass> columns did not propagate")
            prob_group = f["subclass_prob"]
            prob_cols = [c for c in prob_group.keys() if c != "MODEL_NAME"]
            prob_sums = np.sum(
                [np.asarray(prob_group[c], dtype=float) for c in prob_cols], axis=0)
            assert np.allclose(prob_sums, 1.0), (
                f"register /subclass_prob rows do not sum to 1.0 "
                f"(min={prob_sums.min()}, max={prob_sums.max()})")

        assert n_model_register == n_kept, (
            f"register n_model={n_model_register} != kept count {n_kept}")
        print(f"register verified: h5py opens {register_path}, "
              f"n_model={n_model_register} == kept count {n_kept}; "
              f"/models carries T_EFF/LOGG/Z_H (A12); /bands carries "
              f"{vega_zp.shape[0]} VEGA_ZP_MJY values; /subclass_prob "
              f"({len(prob_cols)} codes) sums to 1.0 on every row")

    finally:
        # No state file outlives the run -- _work/ never survives, success
        # or failure (the kept/raw scratch subdirectories are already
        # moved OUT of _work by this point on the success path, so this
        # only ever removes leftovers from a failed run).
        if work.exists():
            shutil.rmtree(work)

    wall_s = time.time() - t0
    peak_mb = _peak_rss_mb()
    print(f"\nbuild_sps complete: {n_raw} raw, {n_kept} kept "
          f"(coverage max_dist={max_dist:.6f}, n_uncovered={n_uncovered}, "
          f"packing={min_kept_dist:.6f})")
    print(f"wall time: {wall_s:.1f} s   peak RSS: {peak_mb:.1f} MB")

    return dict(
        n_raw=n_raw, n_kept=n_kept, max_dist=max_dist,
        n_uncovered=n_uncovered, min_kept_dist=min_kept_dist,
        wall_s=wall_s, peak_mb=peak_mb, output_root=str(out),
        raw_root=str(raw), register_path=register_path,
    )


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m sesnaimpute.sed_models.build_sps",
        description="Build the stellar-photosphere (SPS) kept class "
                     "library + raw splice + register, in one command.")
    parser.add_argument("root", help="root.cfg path, or a bare data-root directory")
    args = parser.parse_args(raw_args)
    build(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
