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

THE FILL, THEN THE CONVOLUTION (amendment 2026-10-27, owner YES;
AMENDMENT_fill_rule.md). An undefined aperture is a resolution limit of
the model's own aperture set, not absent flux (Richardson et al. 2024,
ApJ 961, 2, sec. III.7; the original release filled it with the closest
valid flux, which artificially brightened inner apertures). Per model and
wavelength sample, apertures ABOVE the model's all-wavelength-undefined
prefix that are undefined are filled by inward power-law extrapolation of
the model's own curve of growth from the two nearest defined apertures
above (slope clamped to [0, 3]; fewer than two -> stays undefined) --
`fill_apertures`. The completed SED is then convolved over ALL samples,
flux = sum(F * R), no renormalisation; a (model, aperture, band) cell is
NaN only if a sample the filter covers is still undefined after the fill.
TOTAL_FLUX_ERR is the same sum of the UNCERTAINTIES (summed, as the
release did), NaN where flux is NaN or a covered sample was filled.
Negative raw samples (float noise, ~1e-11 mJy) are kept as-is; the clamp
lives downstream (`curve_of_growth`). Where every covered raw sample is
finite the value equals the release's sedfitter sum.

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

#: Models per chunk: 5000 x 20 apertures x 200 wavelengths x 8 bytes
#: (float64) = 160 MB per working array; the fill and the eleven-band
#: convolution run on the whole (model, aperture, wavelength) block at once.
CHUNK_MODELS = 5000

#: Clamp on the inward power-law slope of a model's curve of growth: a
#: curve of growth cannot fall outward (0); 2 is uniform surface
#: brightness, 3 the ceiling against a noisy pair (amendment, 2026-10-27).
SLOPE_MIN, SLOPE_MAX = 0.0, 3.0

#: Fill reach (amendment 2, 2026-10-27): a fill from the nearest defined
#: aperture more than MAX_FILL_STEPS grid steps out is validated only for
#: far-infrared samples (> FAR_IR_UM) at an aperture where the model's own
#: 24 um sample is defined (the far-IR-faint case, 1-1.5 dex below 24 um).
MAX_FILL_STEPS = 2
FAR_IR_UM = 30.0
ANCHOR_UM = 24.0

FILL_RULE_SENTENCE = (
    "A model's aperture that its radiative transfer could not resolve at one wavelength, "
    "but resolved at others, is filled at that wavelength by extending the model's own "
    "curve of growth inward as a power law through its two nearest defined larger apertures "
    "(slope limited to 0 to 3), as in Richardson et al. 2024, ApJ 961, 2, sec. III.7, "
    "without the original release's brightening; the nearest defined aperture must lie within "
    "two grid steps, except beyond 30 um where the model's 24 um flux is defined at that "
    "aperture; apertures undefined at every wavelength "
    "stay undefined, and a band is NaN only if a sample it covers is still undefined.")


# ==========================================================================
# PART 1 -- THE FILL AND THE CONVOLUTION (YSO)
# ==========================================================================

def fill_apertures(v, log_ap, wav_um):
    """Fill, IN PLACE, the undefined cells of `v` (n, n_ap, n_wav), float64,
    that lie above each model's all-wavelength-undefined PREFIX, by inward
    power-law extrapolation of the model's own curve of growth at that
    wavelength (amendment 2026-10-27; Richardson et al. 2024 sec. III.7).

    Defined = finite and > 0; sources are ORIGINALLY defined cells only
    (a filled cell is never a source). For an undefined cell (k, w) with
    the two nearest defined apertures above it j1 < j2:
        s = log10(F(j2)/F(j1)) / log10(a_j2/a_j1),  clamped to [0, 3]
        log10 F(k) = log10 F(j1) - s * log10(a_j1 / a_k)
    Fewer than two defined above -> the cell stays undefined. Reach
    (amendment 2): the fill needs j1 - k <= MAX_FILL_STEPS, except for
    samples beyond FAR_IR_UM where the raw sample nearest ANCHOR_UM is
    finite at aperture k; otherwise the cell stays undefined. The prefix
    is the leading run of apertures with NO finite sample at any
    wavelength; it is never filled.

    Returns (filled (n,n_ap,n_wav) bool, steps (n,n_ap,n_wav) int8: the
    aperture-index distance k -> j1 of every filled cell (0 elsewhere),
    left (n,n_ap,n_wav) bool: undefined above the prefix and not filled).
    Vectorised over models and wavelengths; one backward pass over the
    apertures."""
    n, n_ap, n_wav = v.shape
    has_fin = np.isfinite(v).any(axis=2)
    first_fin = np.where(has_fin.any(axis=1), has_fin.argmax(axis=1), n_ap)
    prefix = np.arange(n_ap)[None, :] < first_fin[:, None]          # (n, n_ap)

    filled = np.zeros(v.shape, dtype=bool)
    steps = np.zeros(v.shape, dtype=np.int8)
    left = np.zeros(v.shape, dtype=bool)
    l1 = np.zeros((n, n_wav)); l2 = np.zeros((n, n_wav))
    a1 = np.zeros((n, n_wav)); a2 = np.zeros((n, n_wav))
    j1 = np.zeros((n, n_wav), dtype=np.int16)
    have1 = np.zeros((n, n_wav), dtype=bool)
    have2 = np.zeros((n, n_wav), dtype=bool)
    wav_um = np.asarray(wav_um, dtype=float)
    far_ir = (wav_um > FAR_IR_UM)[None, :]
    i_anchor = int(np.argmin(np.abs(wav_um - ANCHOR_UM)))
    for k in range(n_ap - 1, -1, -1):
        vk = v[:, k, :]
        defined = np.isfinite(vk) & (vk > 0)
        undef = ~np.isfinite(vk) & ~prefix[:, k][:, None]
        anchor_defined = np.isfinite(vk[:, i_anchor])[:, None]          # raw, before any fill at k
        in_reach = ((j1 - k) <= MAX_FILL_STEPS) | (far_ir & anchor_defined)
        tgt = undef & have2 & in_reach
        left[:, k, :] = undef & ~tgt
        if tgt.any():
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.clip((l2 - l1) / (a2 - a1), SLOPE_MIN, SLOPE_MAX)
            lf = l1 - s * (a1 - log_ap[k])
            vk[tgt] = 10.0 ** lf[tgt]
            filled[:, k, :] = tgt
            steps[:, k, :] = np.where(tgt, np.minimum(j1 - k, 127), 0)
        # push this aperture into the "nearest defined above" state
        with np.errstate(invalid="ignore", divide="ignore"):
            lk = np.log10(np.where(defined, vk, 1.0))
        l2 = np.where(defined, l1, l2); a2 = np.where(defined, a1, a2)
        have2 = np.where(defined, have1, have2)
        l1 = np.where(defined, lk, l1); a1 = np.where(defined, log_ap[k], a1)
        j1 = np.where(defined, k, j1); have1 = have1 | defined
    return filled, steps, left


def convolve_cube(flux_fits, filters, chunk=CHUNK_MODELS, limit=None):
    """Fill the model's undefined apertures (`fill_apertures`), then convolve
    the completed SED through every sedfitter Filter in `filters` ({band:
    Filter}, already normalised), reading each chunk of the raw cube ONCE.

        flux = sum_all(F * R)             (no renormalisation)

    with R the filter rebinned onto the cube's own frequency grid. A
    (model, aperture, band) cell is UNDEFINED (NaN) iff some sample the
    filter covers (R > 0) is still undefined after the fill. TOTAL_FLUX_ERR
    is the same sum over the UNCERTAINTIES cube (summed, as the release
    did), NaN where flux is NaN or a covered sample was filled (a filled
    sample has no Monte Carlo uncertainty of its own).

    `limit` bounds the compute to the first `limit` models of the raw cube.

    Returns (per_band, names, ap_au, stats): per_band[band] = (flux,
    flux_err or None, undefined, fully_finite) with fully_finite True where
    every covered RAW sample was finite (no fill involved); stats counts the
    fill (cells filled by 1, 2, >=3 steps; cells left undefined above the
    prefix)."""
    cube = SEDCube.read(flux_fits, order="nu", memmap=True)
    bands = list(filters)
    resp = np.stack([np.asarray(filters[b].rebin(cube.nu).response, dtype=np.float64)
                     for b in bands], axis=1)                       # (n_wav, n_band)
    inband = (resp > 0).astype(np.float64)
    val_factor = cube.val.unit.to(u.mJy)
    has_unc = cube.unc is not None
    unc_factor = cube.unc.unit.to(u.mJy) if has_unc else None

    ap_au = np.asarray(cube.apertures.to(u.au).value
                       if hasattr(cube.apertures, "unit") else cube.apertures, dtype=float)
    log_ap = np.log10(ap_au)
    wav_um = np.asarray(cube.wav.to(u.micron).value if hasattr(cube.wav, "unit") else cube.wav, dtype=float)
    n, n_ap = cube.n_models, cube.n_ap
    if limit is not None:
        n = min(n, limit)
    nb = len(bands)
    flux = np.full((n, n_ap, nb), np.nan)
    flux_err = np.full((n, n_ap, nb), np.nan) if has_unc else None
    fully_finite = np.zeros((n, n_ap, nb), dtype=bool)
    stats = {"filled_1": 0, "filled_2": 0, "filled_3plus": 0, "left_undefined": 0}

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        m = hi - lo
        v = np.array(cube.val[lo:hi], dtype=np.float64) * val_factor     # (m, n_ap, n_wav)
        raw_nonfinite = (~np.isfinite(v)).reshape(m * n_ap, -1).astype(np.float64)
        fully_finite[lo:hi] = ((raw_nonfinite @ inband) == 0).reshape(m, n_ap, nb)

        filled, steps, left = fill_apertures(v, log_ap, wav_um)
        stats["filled_1"] += int((steps == 1).sum())
        stats["filled_2"] += int((steps == 2).sum())
        stats["filled_3plus"] += int((steps >= 3).sum())
        stats["left_undefined"] += int(left.sum())

        still_bad = (~np.isfinite(v)).reshape(m * n_ap, -1).astype(np.float64) @ inband > 0
        f = np.where(np.isfinite(v), v, 0.0).reshape(m * n_ap, -1) @ resp
        flux[lo:hi] = np.where(still_bad, np.nan, f).reshape(m, n_ap, nb)
        if has_unc:
            s = np.array(cube.unc[lo:hi], dtype=np.float64) * unc_factor
            s_bad = (~np.isfinite(s) | filled).reshape(m * n_ap, -1).astype(np.float64) @ inband > 0
            fe = np.where(np.isfinite(s), s, 0.0).reshape(m * n_ap, -1) @ resp
            flux_err[lo:hi] = np.where(still_bad | s_bad, np.nan, fe).reshape(m, n_ap, nb)

    names = np.asarray(cube.names)[:n]
    per_band = {b: (flux[:, :, i], None if flux_err is None else flux_err[:, :, i],
                    ~np.isfinite(flux[:, :, i]), fully_finite[:, :, i])
                for i, b in enumerate(bands)}
    return per_band, names, ap_au, stats


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
    """Fill and convolve one YSO geometry's raw flux.fits into
    `<out_dir>/<band>.fits` for every band in `bands`, prints the fill
    counts and the per-band NaN fraction. Returns {band: path}.

    With `limit=None`: rows are written in the geometry's OWN
    parameters.fits order (join key MODEL_NAME, never position). With
    `limit` (a timing/verification slice, never a scientific subset): the
    first `limit` models in the raw cube's OWN row order, no alignment."""
    flux_fits = os.path.join(geometry_dir, "flux.fits")
    assert os.path.exists(flux_fits), f"{flux_fits} missing"
    ref_names = None
    if limit is None:
        ref_names = np.char.strip(np.asarray(
            fits.getdata(os.path.join(geometry_dir, "parameters.fits"), 1)["MODEL_NAME"]).astype(str))

    filters = {b: mc.build_sedfitter_filter(b) for b in bands}
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    per_band, names, ap_au, stats = convolve_cube(flux_fits, filters, chunk=chunk, limit=limit)
    t_conv = time.perf_counter() - t0
    order = None
    if ref_names is None:
        out_names = names
    else:
        order = align_by_name(ref_names, names)
        out_names = ref_names
    n_models = len(out_names)
    log(f"  fill: models={n_models} filled_1step={stats['filled_1']} filled_2steps={stats['filled_2']} "
        f"filled_3plus={stats['filled_3plus']} left_undefined_above_prefix={stats['left_undefined']}  "
        f"fill+convolve all {len(bands)} bands [{t_conv:.1f}s]")

    written = {}
    for b in bands:
        flux, flux_err, undefined, fully_finite = per_band[b]
        if order is not None:
            flux, undefined, fully_finite = flux[order], undefined[order], fully_finite[order]
            flux_err = flux_err[order] if flux_err is not None else None
        written[b] = mc.write_convolved_band_fits(
            None, b, out_names, flux, ap_au,
            filter_wavelength_um=mc.filter_central_wavelength_um(b),
            total_flux_err_mjy=flux_err, convmeth="sedfitter",
            convmeth_note="YSO raw flux.fits, filled (owner 2026-10-27)",
            out_dir=out_dir)
        with fits.open(written[b], mode="update") as h:
            h[0].header["READING"] = FILL_RULE_SENTENCE
        n_undefined = int(undefined.sum())
        msg = (f"  {b}: undefined(model,ap) cells={n_undefined} "
               f"({100 * n_undefined / undefined.size:.2f}% NaN)")
        if verify_against_release:
            release_flux = _release_band_flux(geometry_dir, b, out_names)
            zero_mismatch = int(((release_flux == 0) & np.isfinite(flux)).sum())
            ratio_mask = fully_finite & (release_flux != 0) & np.isfinite(flux)
            if ratio_mask.any():
                ratio = flux[ratio_mask] / release_flux[ratio_mask]
                msg += (f"  release_zero_ours_finite={zero_mismatch}  fully-finite n={int(ratio_mask.sum())} "
                        f"median_ratio={float(np.median(ratio)):.6f} max|ratio-1|={float(np.abs(ratio - 1).max()):.2e}")
        log(msg)

    mc.verify_convolved_band_fits(None, bands, out_names, out_dir=out_dir, allow_nan=True)
    return written


# ==========================================================================
# PACS-ADD ENTRY POINT, unchanged, for the four continuum libraries
# (sps/agb/galz/pahn -- not yso any more, see main())
# ==========================================================================

def _pacs_convolve_cube(flux_fits, filt, chunk=20000):
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


def convolve_library(lib_dir, bands=PACS_BANDS, chunk=20000, log=print):
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
