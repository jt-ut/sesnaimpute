"""
convolve_yso.py (renamed from convolve_pacs.py: the YSO path now dominates
this module)
====================================================================
Owner's ruling 2026-10-27 (bms_review/briefs/yso_reconvolve_2026-10-27/
BRIEF.md; do not reopen): the YSO library's eleven band fluxes (J H Ks
I1 I2 I3 I4 M1 P70 P100 P160) are convolved by THIS package from each
geometry's raw `flux.fits`, with undefined (NaN) SED samples treated as
MISSING, never as zero. The Robitaille/Richardson release's OWN
`convolved/{survey}/{file}.fits` files are no longer read as a data
source by anything in this package -- they carried the release's own
zero-fill for the exact same NaN samples, so at the apertures SESNA
uses, 32% of spubsmi models and 37% of s-p-hmi models carried a PACS
160 flux of exactly zero (13-25% P70/P100, 1.4-18% MIPS 24, 0.2-13%
IRAC 8) -- a fabricated measurement, not a real zero.

THE CONVOLUTION (`convolve_cube`): per model, aperture and band,
    flux = sum_finite(F * R) / sum_finite(R) * sum_all(R)
with R the sedfitter-normalised filter rebinned onto the cube's own
frequency grid (exactly as the old PACS-only `convolve_cube` did), the
sums over wavelength samples. A cell is UNDEFINED (NaN) when
sum_finite(R) < 0.5 * sum_all(R) (less than half the filter's weight
sits on finite samples) -- which also covers "no sample finite at all"
as a special case. Where every in-band sample is finite this reduces
exactly to sum_finite(R) == sum_all(R), so flux == sum(F*R): today's
plain sedfitter sum. TOTAL_FLUX_ERR mirrors the SAME arithmetic with
UNCERTAINTIES' sigma in place of F (summed, not quadrature -- that is
what the release itself did), using F's own finite mask, NaN wherever
flux is NaN. Negative raw samples (float noise, ~1e-11 mJy) are kept
as-is; no clamp lives here -- the per-model flux floor clamp downstream
(`curve_of_growth`) is unchanged.

Read-only input: each geometry's own raw `flux.fits` (VALUES +
UNCERTAINTIES), nothing else from the release. Output: one file per
geometry and band at the `yso_convolved` family path
(`paths_families.py`; `<sed_models area>/yso_convolved/<geometry>/
<band>.fits`), in the SAME layout `model_convolution.
write_convolved_band_fits` has always written (MODEL_NAME, TOTAL_FLUX,
TOTAL_FLUX_ERR, APERTURES), rows in the geometry's OWN parameters.fits
order (the join key is MODEL_NAME, never position).

A DIAGNOSTIC-ONLY comparison against the release's own convolved files
is available (`--verify-against-release`, off by default): per
geometry and band, it prints models/seconds, the count of UNDEFINED
cells, how many cells the release zero-filled where we now have a
finite value, and the median ratio ours/release restricted to cells
where the raw band was FULLY finite (no missing sample at all) -- that
ratio must be 1 to 1e-4, since the two arithmetics agree exactly there.
This is a read of the release's files for a printed number ONLY; it is
never written to any output file and nothing downstream consumes it.

Entry points (keeping the four continuum libraries' PACS-add entry
point working, unchanged, alongside the new YSO one):
    python -m sesnaimpute.sed_models.convolve_yso <root.cfg> sps agb galz pahn [--no-register]
    python -m sesnaimpute.sed_models.convolve_yso <root.cfg> yso
        [--geometries ...] [--out <dir>] [--limit N] [--verify-against-release] [--no-register]
`--out` writes directly to that directory tree (per-geometry
subdirectories, `<out>/<geometry>/<band>.fits`) instead of the
`yso_convolved` family path -- for a scratch/verification run that must
never touch the real data root. `--limit N` convolves only the first N
models of each geometry (by raw flux.fits row order) -- a timing/
verification slice, never a scientific subset.
Not h2shock (a line
spectrum: its bands are the exact analytic convolution, rebuilt by build_h2shock)
and not pahc (its register also needs the host-photosphere table only build_pahc has).
====================================================================
"""

import argparse
import os
import resource
import sys
import time

import numpy as np
from astropy import units as u
from astropy.io import fits
from sedfitter.sed import SEDCube

from sesnaimpute.sed_models.constants import BANDS
from sesnaimpute.sed_models.curate import model_convolution as mc
from sesnaimpute.sed_models.curate.model_io import align_by_name
from sesnaimpute.sed_models.curate.yso_labeling import _CONVOLVED_FILENAME, _CONVOLVED_SUBDIR

PACS_BANDS = tuple(b for b in BANDS if BANDS[b].survey == "PACS")

#: All eleven bands, `constants.BANDS` order (J H Ks I1 I2 I3 I4 M1 P70
#: P100 P160) -- every band the YSO reconvolution writes, matching
#: `yso_labeling.ASSEMBLY_BANDS`.
YSO_BANDS = tuple(BANDS)

#: Models per chunk: 20000 x 200 wavelengths x 4 bytes = 16 MB per read
#: (doubled when UNCERTAINTIES is also read).
CHUNK_MODELS = 20000

#: Fraction of the filter's total response weight that must sit on
#: finite samples for a (model, aperture) cell to be DEFINED; below
#: this (including zero, i.e. no finite sample at all) the cell is
#: UNDEFINED, stored as NaN (brief Part 1).
MIN_FINITE_WEIGHT_FRAC = 0.5


# ==========================================================================
# PART 1 -- THE CONVOLUTION (YSO)
# ==========================================================================

def convolve_cube(flux_fits, filt, chunk=CHUNK_MODELS):
    """(n_model, n_ap) band flux and flux_err in mJy of every model of a
    flux.fits cube through sedfitter Filter `filt` (already normalised),
    undefined (model, aperture) cells NaN rather than 0 -- see module
    docstring for the exact formula and the UNDEFINED rule.

    Returns (flux, flux_err, undefined, fully_finite, names, ap_au).
    flux_err is None if the cube carries no UNCERTAINTIES extension.
    fully_finite (n, n_ap) bool: True where every in-band (R>0) sample
    was finite -- a strict subset of ~undefined, used only to restrict
    the release ratio-check to cells with no missing sample at all.
    """
    cube = SEDCube.read(flux_fits, order="nu", memmap=True)
    binned = filt.rebin(cube.nu)
    resp = np.asarray(binned.response, dtype=np.float64)
    sum_all_r = float(resp.sum())
    val_factor = cube.val.unit.to(u.mJy)
    has_unc = cube.unc is not None
    unc_factor = cube.unc.unit.to(u.mJy) if has_unc else None

    n, n_ap = cube.n_models, cube.n_ap
    flux = np.full((n, n_ap), np.nan, dtype=np.float64)
    flux_err = np.full((n, n_ap), np.nan, dtype=np.float64) if has_unc else None
    undefined = np.zeros((n, n_ap), dtype=bool)
    fully_finite = np.zeros((n, n_ap), dtype=bool)

    tol = 1e-9 * max(sum_all_r, 1.0)
    for i_ap in range(n_ap):
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            v = np.asarray(cube.val[lo:hi, i_ap, :], dtype=np.float64) * val_factor
            fin = np.isfinite(v)
            sum_finite_r = fin.astype(np.float64) @ resp
            bad = sum_finite_r < MIN_FINITE_WEIGHT_FRAC * sum_all_r
            weighted = np.where(fin, v, 0.0) * resp[None, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                f = weighted.sum(axis=1) / sum_finite_r * sum_all_r
            flux[lo:hi, i_ap] = np.where(bad, np.nan, f)
            undefined[lo:hi, i_ap] = bad
            fully_finite[lo:hi, i_ap] = sum_finite_r >= (sum_all_r - tol)

            if has_unc:
                s = np.asarray(cube.unc[lo:hi, i_ap, :], dtype=np.float64) * unc_factor
                weighted_s = np.where(fin, s, 0.0) * resp[None, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    fe = weighted_s.sum(axis=1) / sum_finite_r * sum_all_r
                flux_err[lo:hi, i_ap] = np.where(bad, np.nan, fe)

    names = np.asarray(cube.names)
    ap_au = np.asarray(cube.apertures.to(u.au).value
                       if hasattr(cube.apertures, "unit") else cube.apertures, dtype=float)
    return flux, flux_err, undefined, fully_finite, names, ap_au


def _release_band_flux(geometry_dir, band, ref_names):
    """DIAGNOSTIC ONLY: the release's OWN convolved/{subdir}/{filename}.fits
    for `band`, aligned to `ref_names` -- never a data source for any
    product of this package, read here only for the printed comparison
    (see module docstring)."""
    path = os.path.join(geometry_dir, "convolved", _CONVOLVED_SUBDIR[band], _CONVOLVED_FILENAME[band])
    with fits.open(path, memmap=True) as h:
        names = np.asarray(h[1].data["MODEL_NAME"]).astype(str)
        flux = np.asarray(h[1].data["TOTAL_FLUX"], dtype=np.float64)
    order = align_by_name(ref_names, names)
    return flux[order]


def convolve_geometry(geometry_dir, out_dir, bands=YSO_BANDS, chunk=CHUNK_MODELS,
                      limit=None, verify_against_release=False, log=print):
    """Convolve one YSO geometry's raw flux.fits into `<out_dir>/<band>.fits`
    for every band in `bands` -- rows written in the geometry's OWN
    parameters.fits order (the join key is MODEL_NAME, never position).
    `limit`, if given, convolves only the first `limit` models (by raw
    flux.fits row order) -- a timing/verification slice, never a
    scientific subset. Returns {band: path}."""
    flux_fits = os.path.join(geometry_dir, "flux.fits")
    assert os.path.exists(flux_fits), f"{flux_fits} missing"
    ref_names = np.char.strip(np.asarray(
        fits.getdata(os.path.join(geometry_dir, "parameters.fits"), 1)["MODEL_NAME"]).astype(str))
    if limit is not None:
        ref_names = ref_names[:limit]

    # Filters first: a missing curve stops the run before any work.
    filters = {b: mc.build_sedfitter_filter(b) for b in bands}
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for b in bands:
        t0 = time.perf_counter()
        flux, flux_err, undefined, fully_finite, names, ap_au = convolve_cube(flux_fits, filters[b], chunk=chunk)
        if limit is not None:
            flux = flux[:limit]
            flux_err = flux_err[:limit] if flux_err is not None else None
            undefined, fully_finite, names = undefined[:limit], fully_finite[:limit], names[:limit]
        order = align_by_name(ref_names, names)
        flux, undefined, fully_finite = flux[order], undefined[order], fully_finite[order]
        flux_err = flux_err[order] if flux_err is not None else None

        written[b] = mc.write_convolved_band_fits(
            None, b, ref_names, flux, ap_au,
            filter_wavelength_um=mc.filter_central_wavelength_um(b),
            total_flux_err_mjy=flux_err, convmeth="sedfitter",
            convmeth_note="YSO raw flux.fits, NaN-aware (owner 2026-10-27)",
            out_dir=out_dir)

        n_models = flux.shape[0]
        n_undefined = int(undefined.sum())
        elapsed = time.perf_counter() - t0
        msg = (f"  {b}: {n_models} models x {ap_au.size} apertures "
               f"[{elapsed:.1f}s]  undefined(model,ap) cells={n_undefined}")

        if verify_against_release:
            release_flux = _release_band_flux(geometry_dir, b, ref_names)
            zero_mismatch = int(((release_flux == 0) & np.isfinite(flux)).sum())
            ratio_mask = fully_finite & (release_flux != 0) & np.isfinite(flux)
            if ratio_mask.any():
                ratio = flux[ratio_mask] / release_flux[ratio_mask]
                median_ratio = float(np.median(ratio))
            else:
                median_ratio = float("nan")
            msg += (f"  release_zero_ours_finite={zero_mismatch}  "
                   f"median_ratio_ours/release(fully-finite,n={int(ratio_mask.sum())})={median_ratio:.6f}")
        log(msg)
    return written


# ==========================================================================
# PACS-ADD ENTRY POINT, unchanged, for the four continuum libraries
# (sps/agb/galz/pahn -- not yso any more, see main())
# ==========================================================================

def _pacs_convolve_cube(flux_fits, filt, chunk=CHUNK_MODELS):
    """The original (pre-2026-10-27), PACS-only, no-NaN-handling
    convolution: `sum(F * R) * val_factor`, exactly sedfitter's own sum.
    Kept unchanged for sps/agb/galz/pahn, whose cubes carry no NaN -- the
    YSO path now uses `convolve_cube` above instead."""
    cube = SEDCube.read(flux_fits, order="nu", memmap=True)
    binned = filt.rebin(cube.nu)
    val_factor = cube.val.unit.to(u.mJy)
    n, n_ap = cube.n_models, cube.n_ap
    out = np.zeros((n, n_ap), dtype=np.float64)
    for i_ap in range(n_ap):
        for lo in range(0, n, chunk):
            v = cube.val[lo:lo + chunk, i_ap, :]
            resp = binned.response.astype(v.dtype)
            out[lo:lo + chunk, i_ap] = np.sum(v * resp, axis=1) * val_factor
    return out, np.asarray(cube.names), np.asarray(cube.apertures.to(u.au).value
                                                   if hasattr(cube.apertures, "unit")
                                                   else cube.apertures, dtype=float)


def convolve_library(lib_dir, bands=PACS_BANDS, chunk=CHUNK_MODELS, log=print):
    """Write convolved/<band>.fits for each PACS band of a continuum
    library (sps/agb/galz/pahn) at `lib_dir` from its flux.fits, then
    verify the files. Returns {band: path}."""
    flux_fits = os.path.join(lib_dir, "flux.fits")
    assert os.path.exists(flux_fits), f"{flux_fits} missing"
    filters = {b: mc.build_sedfitter_filter(b) for b in bands}
    written = {}
    names = None
    for b in bands:
        t0 = time.perf_counter()
        flux, names, ap_au = _pacs_convolve_cube(flux_fits, filters[b], chunk=chunk)
        written[b] = mc.write_convolved_band_fits(
            lib_dir, b, names, flux, ap_au,
            filter_wavelength_um=mc.filter_central_wavelength_um(b),
            convmeth="sedfitter", convmeth_note="convolved onto the built library")
        log(f"  {b}: {flux.shape[0]} models x {flux.shape[1]} apertures "
            f"[{time.perf_counter() - t0:.1f}s]  flux range "
            f"[{flux.min():.4g}, {flux.max():.4g}] mJy")
    mc.verify_convolved_band_fits(lib_dir, bands, names)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sesnaimpute.sed_models.convolve_yso",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="root.cfg path")
    ap.add_argument("keys", nargs="+", help="library keys: sps agb galz pahn yso")
    ap.add_argument("--no-register", action="store_true",
                    help="write the convolved files only; leave the register alone")
    ap.add_argument("--geometries", nargs="+", default=None,
                    help="yso only: convolve only these geometry folders")
    ap.add_argument("--out", default=None,
                    help="yso only: write <out>/<geometry>/<band>.fits directly, "
                         "bypassing the yso_convolved family path (scratch runs)")
    ap.add_argument("--limit", type=int, default=None,
                    help="yso only: convolve only the first N models of each geometry")
    ap.add_argument("--verify-against-release", action="store_true",
                    help="yso only: also print the diagnostic comparison against the "
                         "release's own convolved files (never used as a data source)")
    args = ap.parse_args(argv)

    from sesnaimpute.sed_models import library, paths
    from sesnaimpute.sed_models.curate import yso_curation_state as regstate
    paths.load_config(args.root)
    regs_dir = paths.path_for("model_registers_dir")
    for bad in ("h2shock", "pahc"):
        if bad in args.keys:
            ap.error(f"{bad}: rebuilt by its own build_{bad}, not by this module")

    for key in args.keys:
        t0 = time.perf_counter()
        if key == "yso":
            yso_root = paths.input_dir("yso_models_download")
            geometries = args.geometries or list(regstate.GEOMETRIES)
            unknown = [g for g in geometries if g not in regstate.GEOMETRIES]
            if unknown:
                ap.error(f"unknown geometries {unknown}; choose from {list(regstate.GEOMETRIES)}")
            print(f"[yso] raw release root={yso_root}", flush=True)
            for g in geometries:
                geometry_dir = os.path.join(yso_root, g)
                out_dir = (os.path.join(args.out, g) if args.out
                          else paths.path_for("yso_convolved", geometry=g))
                print(f"[yso] {g} -> {out_dir}", flush=True)
                convolve_geometry(geometry_dir, out_dir, limit=args.limit,
                                 verify_against_release=args.verify_against_release)
        else:
            lib_dir = paths.path_for(f"sed_models_{key}")
            print(f"[{key}] {lib_dir}", flush=True)
            convolve_library(lib_dir)
            if not args.no_register:
                out = library.write_register(regs_dir, key, lib_dir)
                print(f"  register -> {out}", flush=True)
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        gb = kb / 1e9 if sys.platform == "darwin" else kb / 1e6
        print(f"[{key}] done in {time.perf_counter() - t0:.1f}s, peak RSS {gb:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
