"""
convolve_pacs.py
====================================================================
Add the three Herschel PACS bands (P70, P100, P160) to a library that is
already built: read its flux.fits, write convolved/P70.fits, P100.fits and
P160.fits exactly as the other eight (TOTAL flux, the same sedfitter
arithmetic, the same file layout, no normalisation factor), and rebuild its
register. Nothing else is touched: flux.fits, members, classmap, parameters
and models.conf stay as they are, and the eight old convolved files are not
rewritten. Works for any library (aperture-dependent or not).

The convolved file stores the band flux at every aperture of the library's
cube (20 for the YSO grid); the PACS aperture (BANDS[band].aperture_arcsec x
distance, 9.6 / 8.0 / 12.8 arcsec) is applied later by
register.reference.derive, by the same linear-in-AU interpolation
(curate.curve_of_growth.interp_at_aperture) that serves the other bands.

The sum over wavelength is sedfitter's own (convolve_model_dir, version-2
cubes): per aperture, sum(F * response) over the cube's frequency grid with
the filter rebinned onto it. Here the cube is memory-mapped and read in
chunks of models, so the 387k-model YSO cube never has to fit in memory;
rows are independent, so chunking does not change a value.

FAILS LOUDLY if a filter curve file is absent ("filter curve P70.dat
missing"): the curves are inputs under data/filter_curves/, placed by hand.

    python -m sesnaimpute.sed_models.convolve_pacs <root.cfg> <key>... [--no-register]

<key> is a library family key: sps, agb, galz, pahn or yso. For yso the three
PACS files are taken from the release's own Pacs.blue/green/red convolutions by
model name (no filter curve needed); every other key convolves from its spectra.
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

PACS_BANDS = tuple(b for b in BANDS if BANDS[b].survey == "PACS")

#: Models per chunk: 20000 x 200 wavelengths x 4 bytes = 16 MB per read.
CHUNK_MODELS = 20000


def convolve_cube(flux_fits, filt, chunk=CHUNK_MODELS):
    """(n_model, n_ap) band flux in mJy of every model of a flux.fits cube
    through sedfitter Filter `filt` (already normalised), plus the names and
    apertures (AU)."""
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
    """Write convolved/<band>.fits for each PACS band of the library at
    `lib_dir` from its flux.fits, then verify the files. Returns {band: path}."""
    flux_fits = os.path.join(lib_dir, "flux.fits")
    assert os.path.exists(flux_fits), f"{flux_fits} missing"
    # Filters first: a missing curve stops the run before any work.
    filters = {b: mc.build_sedfitter_filter(b) for b in bands}
    written = {}
    names = None
    for b in bands:
        t0 = time.perf_counter()
        flux, names, ap_au = convolve_cube(flux_fits, filters[b], chunk=chunk)
        written[b] = mc.write_convolved_band_fits(
            lib_dir, b, names, flux, ap_au,
            filter_wavelength_um=mc.filter_central_wavelength_um(b),
            convmeth="sedfitter", convmeth_note="convolved onto the built library")
        log(f"  {b}: {flux.shape[0]} models x {flux.shape[1]} apertures "
            f"[{time.perf_counter() - t0:.1f}s]  flux range "
            f"[{flux.min():.4g}, {flux.max():.4g}] mJy")
    mc.verify_convolved_band_fits(lib_dir, bands, names)
    return written


def add_release_pacs_to_yso(lib_dir, yso_root, bands=PACS_BANDS, log=print):
    """YSO only: add convolved/P70|P100|P160.fits to the built YSO library from
    the Robitaille/Richardson release's OWN PACS convolutions
    (<geometry>/convolved/Herschel/Pacs.blue|green|red.fits), picked by
    MODEL_NAME for the library's models. This is the same read the library
    assembly does for its eight bands (yso_assembly._extract_convolved_band_rows,
    yso_labeling._CONVOLVED_SUBDIR/_CONVOLVED_FILENAME); it needs no filter
    curve of ours and no re-assembly. FILTWAV is the project value
    (BANDS[band].wvl_effective_um), as for the other bands."""
    from sesnaimpute.sed_models.curate.yso_labeling import _CONVOLVED_FILENAME, _CONVOLVED_SUBDIR
    from sesnaimpute.sed_models.curate.model_io import align_by_name
    with fits.open(os.path.join(lib_dir, "convolved", "J.fits"), memmap=True) as h:
        names = np.asarray([str(n).strip() for n in h["CONVOLVED FLUXES"].data["MODEL_NAME"]])
        ap_au = np.asarray(h["APERTURES"].data["APERTURE"], dtype=float)
    geoms = sorted(g for g in os.listdir(yso_root)
                   if os.path.isdir(os.path.join(yso_root, g, "convolved")))
    written = {}
    for b in bands:
        t0 = time.perf_counter()
        flux = np.full((names.size, ap_au.size), np.nan)
        err = np.zeros_like(flux)
        index = {n: i for i, n in enumerate(names)}
        found = 0
        for g in geoms:
            path = os.path.join(yso_root, g, "convolved", _CONVOLVED_SUBDIR[b], _CONVOLVED_FILENAME[b])
            with fits.open(path, memmap=True) as h:
                rn = np.asarray([str(n).strip() for n in h[1].data["MODEL_NAME"]])
                pos = np.array([index.get(n, -1) for n in rn])
                keep = pos >= 0
                if not keep.any():
                    continue
                flux[pos[keep]] = np.asarray(h[1].data["TOTAL_FLUX"], dtype=np.float64)[keep]
                err[pos[keep]] = np.asarray(h[1].data["TOTAL_FLUX_ERR"], dtype=np.float64)[keep]
                found += int(keep.sum())
        assert found == names.size and np.all(np.isfinite(flux)), (
            f"{b}: release supplied {found} of {names.size} models")
        written[b] = mc.write_convolved_band_fits(
            lib_dir, b, names, flux, ap_au,
            filter_wavelength_um=mc.filter_central_wavelength_um(b),
            total_flux_err_mjy=err, convmeth="release",
            convmeth_note="Richardson 2024 release convolution")
        log(f"  {b}: {found} models x {ap_au.size} apertures [{time.perf_counter() - t0:.1f}s]  "
            f"flux range [{flux.min():.4g}, {flux.max():.4g}] mJy")
    mc.verify_convolved_band_fits(lib_dir, bands, names)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sesnaimpute.sed_models.convolve_pacs",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="root.cfg path")
    ap.add_argument("keys", nargs="+", help="library keys: sps agb galz h2shock pahc pahn yso")
    ap.add_argument("--no-register", action="store_true",
                    help="write the convolved files only; leave the register alone")
    args = ap.parse_args(argv)

    from sesnaimpute.sed_models import library, paths
    paths.load_config(args.root)
    regs_dir = paths.path_for("model_registers_dir")
    for bad in ("h2shock", "pahc"):
        if bad in args.keys:
            ap.error(f"{bad}: rebuilt by its own build_{bad}, not by this module")
    for key in args.keys:
        lib_dir = paths.path_for(f"sed_models_{key}")
        print(f"[{key}] {lib_dir}", flush=True)
        t0 = time.perf_counter()
        if key == "yso":
            add_release_pacs_to_yso(lib_dir, paths.input_dir("yso_models_download"))
        else:
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
