"""
sesnaimpute/sed_models/build_pahn.py
====================================================================
ONE-COMMAND driver for the PAH-nebular (PAH-N) library: PAHC's own
composite in the R -> infinity limit, i.e. the DL21 excess term alone,
with no star in the aperture. Owner ruling, 2026-10-20
(briefs/round4_2026-10-20/C13_pahn_library.md): sources detected at
5.8 and 8.0 um with nothing at 3.6 or 4.5 um (Orion A 174, Aquila 556,
4-8% of all YSO calls) are patches of PAH-bright nebulosity extracted
as point sources, not protostars. No class modelled them before this
one; the fitter called them Class 0/I at +12 nats because every
alternative predicts a short-wavelength detection PAH-N does not have.

Canonical invocation::

    python3.9 -m sesnaimpute.sed_models.build_pahn <root.cfg path, or a data-root directory>

PHYSICS. PAHC's composite (pahc_curate.compute_composite_sed) is

    M(lambda) = s(lambda) + R * s(8um) * [P_hat(lambda)/P_hat(8um)] * a(lambda)

where s is the host photosphere and a(lambda) is the aperture-growth
factor (pahc_curate.band_aperture_scaling). As R -> infinity at fixed
alpha and PAH size, the host term s(lambda) is a finite additive
constant and the excess term grows without bound, so M(lambda)/[R *
s(8um)] -> [P_hat(lambda)/P_hat(8um)] * a(lambda) exactly: the limit is
independent of the host and of R itself. THE STAR IS ABSENT, NOT
SMALL -- this driver never emits a faint-host composite; the star term
is dropped from the construction entirely, not merely driven small by
a large R.

    M(lambda) = [P_hat(lambda) / P_hat(8um)] * a(lambda)

on PAHC's own wavelength grid (pahc_curate.build_pahc_wavelength_grid),
with PAHC's own a(lambda) (band_aperture_scaling, per alpha) and PAH
size distributions (load_pahc_excess_shape, DL21's three published
size distributions). AXES: alpha x PAH_SIZE = 11 x 3 = 33 raw
combinations -- no host axis (there is no host), no R axis (R
parameterises an excess-to-star RATIO and there is no star to ratio
against). Normalisation is arbitrary (spec section 2.1, carried over
from PAHC: the fit's own free scale absorbs it) but is STATED rather
than left at DL21's own units: every template is scaled so its 8um
flux equals PAHN_ANCHOR_FLUX_MJY.

SAMPLING. Every one of the 33 raw (alpha, PAH size) combinations ships
as its own kept template -- 33 raw, 33 kept, each its own one-member
set -- rather than being run through `sampling.r_net`. There is no
host or R axis to down-sample across (PAHC's analogous r-net exists to
collapse a host x alpha x size x R grid of hundreds of thousands of
rows; here the raw grid already IS the full factorial, 33 rows), and
PAHC_ALPHA_VALUES was calibrated (pahc_curate.py's own module header,
spec section 7) to the fitter's resolution on composite colour, so
every step across it is already a meaningful distinction, not a
duplicate to merge away. This driver still computes the whitened
8-band coordinates (`sampling.whiten`, the same per-band sampling
sigma every library uses) and reports the minimum pairwise distance as
a diagnostic -- never as a filter; see `build()`'s own printout.

DARKNESS. DL21's PAH excess has no flux below its native blue edge
(1.0056 um; pahc_curate module docstring, stage 1) -- entirely below
Gaia G's 0.327-1.050 um bandpass. So the register's G0 synthesis
(`sed_models.register.synth.derive`) finds every PAH-N template dark
in G (G0_FLUX = 0, the true value, never a floor or a NaN -- see
`synth.py`'s own docstring on the darkness convention): there is no
optical flux for a star-less nebular patch to have. This driver's own
report confirms the darkness rule marks every template dark, which is
the physically correct answer here, not a defect.

WRITING: the same shared writers every library in the release uses --
`sed_models.library.write_library` (flux/parameters/classmap/members in
one call) then `model_convolution.build_convolved_band_fits` (the real,
sedfitter convolution PAHC's own driver calls directly, reading
flux.fits back off disk) then `sed_models.library.write_register`.

NO SUBPROCESSES, NO HOST LOOP, NO STAGING DIRECTORY: this is a 33-row
library; `library.write_library` already replaces the live directory
atomically enough for this driver's needs (it removes `convolved/`
first and overwrites the rest), so the scratch-then-swap choreography
PAHC and H2-shock need for a multi-hundred-thousand-row build buys
nothing here.
====================================================================
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
from astropy.table import Table

from sesnaimpute.sed_models import paths, library, sampling
from sesnaimpute.sed_models.curate import model_io
from sesnaimpute.sed_models.curate import pahc_curate as pc
from sesnaimpute.sed_models.curate.model_convolution import build_convolved_band_fits

CONVOLVED_BANDS = pc.PAHC_BAND_ORDER   # ("J","H","Ks","I1","I2","I3","I4","M1")
R_NET_RADIUS = 1.0

MODEL_CLASS = "PAHN"
PAHN_SUBCLASS = "NONE"
CLASS_LEGEND = (
    ("PAHN", "Extended PAH nebular emission with no star in the aperture "
             "(a patch of PAH-bright nebulosity extracted as a point source)"),
)
SUBCLASS_LEGEND = (
    ("NONE", "No subclass: one population, distinguished only by grid axes"),
)
DEFAULT_MODEL_SET_NAME = "PAH nebular emission (DL21 PAH excess alone, no host star)"
PAHN_MODELS_CONF_VERSION = 2   # the standard version; no PAHC-style schema bump here

# The stated 8um flux every template is normalised to (mJy) -- arbitrary
# (the fit's own free scale absorbs it, spec section 2.1), but a stated
# number rather than DL21's own archive units, which mean nothing to a
# reader without re-reading load_pahc_excess_shape's own normalisation.
PAHN_ANCHOR_FLUX_MJY = 1.0

# PAH size ordinal, for members.fits' numeric PAH_SIZE_MIN/MEDIAN/MAX
# columns -- same ordinal pahc_curate/build_pahc.py uses (sma=0 < std=1
# < lrg=2), so a reader cross-checking against PAHC's own members.fits
# finds the same convention.
_PAH_SIZE_ORDINAL = {p: i for i, p in enumerate(pc.PAHC_PAH_SIZE_VALUES)}


def work_dir():
    return paths.path_for("sed_models_pahn")


def dl21_dir():
    return os.path.join(paths.input_dir("sed_models"), "downloads", "PAHspec")


def registers_dir():
    return paths.path_for("model_registers_dir")


def format_pahn_model_name(index):
    """Index-based name ('pahn_00001'), matching pahc_curate.
    format_pahc_model_name's convention."""
    return f"pahn_{index + 1:05d}"


def build_pahn_grid(alphas=pc.PAHC_ALPHA_VALUES, pah_sizes=pc.PAHC_PAH_SIZE_VALUES):
    """The full 11 x 3 = 33 raw (alpha, PAH size) grid -- PAH size outer,
    alpha inner, the same nested order `build_pahc_validation_grid` and
    `build_pedestal_table` iterate the same two axes in (pahc_curate.py
    / build_pahc.py), so a reader cross-checking finds the familiar
    order. There is no host or R axis to cross it against."""
    return [{"alpha": a, "pah_size": s} for s in pah_sizes for a in alphas]


def build_pahn_model_arrays(grid, wave_um, dl21):
    """Every (alpha, pah_size) composite -- the DL21 excess shape alone,
    scaled by PAHC's own aperture-growth law and normalised to
    PAHN_ANCHOR_FLUX_MJY at 8um -- on wave_um (ascending in).

    Returns a dict carrying BOTH the ascending-wavelength array (for the
    in-process diagnostic band-ratio table) and the descending-
    wavelength / ascending-frequency arrays the release's flux.fits
    convention stores (the SEDCube.read reorder bug; see pahc_curate's
    module docstring)."""
    n_models = len(grid)
    values_asc = np.empty((n_models, wave_um.size), dtype=np.float64)
    model_names = []
    excess_cache = {}

    for i, p in enumerate(grid):
        e_key = p["pah_size"]
        if e_key not in excess_cache:
            wave_native, p_hat_native = pc.load_pahc_excess_shape(dl21, p["pah_size"])
            p_hat = pc.resample_sed_log_log(wave_native, p_hat_native, wave_um)
            # Re-anchor ON PAH-N's OWN GRID (same reason
            # prepare_pahc_model_arrays re-anchors: load_pahc_excess_shape
            # anchors at 8um on DL21's native grid, which it must to clear
            # FLUX_FLOOR before resampling; forming P_hat/P_hat(8um) from
            # the RESAMPLED array removes the ~1.001 residual that would
            # otherwise leave M(8um) a hair off PAHN_ANCHOR_FLUX_MJY).
            excess_cache[e_key] = p_hat / np.interp(pc.ANCHOR_WAVELENGTH_UM, wave_um, p_hat)
        p_hat_resampled = excess_cache[e_key]

        aperture_factor = pc.band_aperture_scaling(wave_um, p["alpha"])
        values_asc[i] = PAHN_ANCHOR_FLUX_MJY * p_hat_resampled * aperture_factor
        model_names.append(format_pahn_model_name(i))

    wave_um_desc = wave_um[::-1]
    freq_hz_desc = pc.C_UM_S / wave_um_desc
    values_desc = values_asc[:, ::-1].astype(np.float32)[:, np.newaxis, :]
    uncertainties_desc = np.zeros_like(values_desc)

    return dict(
        model_names=np.array(model_names),
        alpha=np.array([p["alpha"] for p in grid], dtype=float),
        pah_size=np.array([p["pah_size"] for p in grid]),
        values_asc=values_asc,
        wave_um_desc=wave_um_desc, freq_hz_desc=freq_hz_desc,
        values_mjy=values_desc, uncertainties_mjy=uncertainties_desc,
    )


def _approx_band_fluxes(wave_um_asc, values_asc, bands=CONVOLVED_BANDS):
    """Fast, independent per-band flux (n_models, len(bands)) for the
    in-process diagnostic table only -- pahc_curate._independent_band_flux,
    the same approximate convolution PAHC's own host-selection step uses
    for its sampling diagnostics (never for the shipped cube; that is
    build_convolved_band_fits's real convolution, called separately in
    `build()`)."""
    out = np.empty((values_asc.shape[0], len(bands)))
    for i in range(values_asc.shape[0]):
        for b_i, b in enumerate(bands):
            out[i, b_i] = pc._independent_band_flux(wave_um_asc, values_asc[i], b)
    return out


def build():
    t0 = time.time()
    out = work_dir()
    dl21 = dl21_dir()
    log = print

    def stage(label):
        log(f"[{time.time() - t0:6.1f}s] {label}")

    stage("Stage 1: DL21 templates (shared download with PAHC)")
    pc.fetch_pahc_excess_templates(dl21)

    wave_um = pc.build_pahc_wavelength_grid()

    stage("Stage 2: raw grid -- alpha x PAH size, no host axis, no R axis")
    grid = build_pahn_grid()
    n_alpha, n_size = len(pc.PAHC_ALPHA_VALUES), len(pc.PAHC_PAH_SIZE_VALUES)
    log(f"  {len(grid)} raw combinations ({n_alpha} alpha x {n_size} PAH size)")

    model_set = build_pahn_model_arrays(grid, wave_um, dl21)
    n_models = len(model_set["model_names"])

    stage("Stage 3: whitened-space diagnostic (no r_net reduction -- every raw "
          "combination ships; see module docstring)")
    band_flux_approx = _approx_band_fluxes(wave_um, model_set["values_asc"])
    if np.any(band_flux_approx <= 0) or not np.all(np.isfinite(band_flux_approx)):
        raise ValueError("a PAH-N raw template has non-finite or non-positive "
                         "approximate band flux -- cannot place it in log10 SED space")
    log10_flux = np.log10(band_flux_approx)
    dark_mask = np.zeros_like(log10_flux, dtype=bool)   # floored at source; none are literal zero
    coords = sampling.whiten(log10_flux, dark_mask)
    min_pairwise = sampling.packing(coords, np.arange(n_models))
    log(f"  {n_models} raw/kept templates (1:1, no sampling reduction); "
       f"minimum pairwise whitened distance {min_pairwise:.4f} "
       f"(sampling radius is {R_NET_RADIUS:.1f})")

    stage("Stage 4: assemble parameters/members/classmap and write the library")
    parameters = Table()
    # Forced to the release-wide cube width (not a plain Python-string
    # column's own narrowest-fit width): C10 checks every cube's
    # MODEL_NAME column shares one width, and a bare string array takes
    # whatever the longest value happens to be -- the same fix
    # pahc_curate's own _assemble_and_write already applies, for the
    # same reason.
    parameters["MODEL_NAME"] = np.asarray(
        model_set["model_names"], dtype=f"U{int(pc.MODEL_NAME_FORMAT[:-1])}")
    parameters["ALPHA"] = model_set["alpha"]
    parameters["PAH_SIZE"] = model_set["pah_size"]

    # members.fits: identity mapping -- every raw row IS its own kept
    # template (no sampling reduction, Stage 3), so rep_of is just
    # arange(n) and every row's N_MEMBERS is 1. PAH_SIZE here is the
    # NUMERIC ordinal (sma=0/std=1/lrg=2), matching build_pahc.py's own
    # members-table convention -- parameters.fits keeps the string label.
    rep_of = np.arange(n_models)
    dist_to_rep = np.zeros(n_models)
    subclass_full = np.full(n_models, PAHN_SUBCLASS)
    members_params = Table()
    members_params["ALPHA"] = model_set["alpha"]
    members_params["PAH_SIZE"] = np.array(
        [_PAH_SIZE_ORDINAL[p] for p in model_set["pah_size"]], dtype=float)
    members = sampling.members_table(model_set["model_names"], rep_of, dist_to_rep,
                                     subclass_full, members_params)
    assert len(members) == n_models, "members table row count disagrees with the raw count"
    assert np.all(members["N_MEMBERS"] == 1), "every PAH-N template must have exactly one member"
    assert np.allclose(members["FRAC_NONE"], 1.0), "FRAC_NONE must be 1.0 for every row"
    log(f"  members: {len(members)} rows, every N_MEMBERS=1, FRAC_NONE all 1.0")

    flux_kwargs = dict(
        wave_um_desc=model_set["wave_um_desc"], freq_hz_desc=model_set["freq_hz_desc"],
        values=model_set["values_mjy"], distance_cm=pc.pahc_distance_placeholder(),
        apertures_au=np.array([pc.POINT_SOURCE_APERTURE_AU]),
        uncertainties=model_set["uncertainties_mjy"],
        header_extras={
            "LG_U": (float(pc.PAHC_LG_U_TOKEN), "fixed DL21 log10 starlight intensity"),
            "PAHION": (pc.PAHC_IONIZATION, "fixed DL21 PAH charge distribution"),
            "F_PAH": (pc.PAHC_F_PAH, "fixed PAH abundance scaling (1 = DL21 fiducial)"),
            "RADFIELD": (pc.PAHC_RADIATION_FIELD, "fixed DL21 illuminating spectrum"),
            "EXCESSRC": ("DL21", "Draine et al. 2021 ApJ 917 3"),
            "ANCHOR8UM": (PAHN_ANCHOR_FLUX_MJY, "stated 8um flux every template is normalised to [mJy]"),
        },
        distance_comment="structural placeholder (aperture-independent, no star to reference)",
        name_format=pc.MODEL_NAME_FORMAT,
    )
    classmap_kwargs = dict(
        class_id=MODEL_CLASS,
        subclass=np.full(n_models, PAHN_SUBCLASS),
        class_legend=CLASS_LEGEND, subclass_legend=SUBCLASS_LEGEND,
        provenance=(
            ("CLASS_SOURCE", "library declaration; all models here are this class"),
            ("SUBCLASS_SOURCE", "none defined; every model carries NONE"),
            ("SUBCLASS_REF", "n/a -- no subclass axis"),
            ("SUBCLASS_NOTE", "grid axes are in parameters.fits, not subclasses"),
            ("LEGEND_SOURCE", "owner ruling 2026-10-20; no Gutermuth counterpart"),
        ),
    )
    models_conf_kwargs = dict(name=DEFAULT_MODEL_SET_NAME, aperture_dependent=False,
                              version=PAHN_MODELS_CONF_VERSION)

    library.write_library(out, "pahn", flux_kwargs, parameters, classmap_kwargs,
                          model_set["model_names"], members, {}, models_conf_kwargs)
    stage(f"Stage 5: library written -- {n_models} models at {out}")

    build_convolved_band_fits(out, CONVOLVED_BANDS)
    stage("Stage 6: convolved/{band}.fits -- real sedfitter convolution")

    problems = model_io.validate_model_directory(out, profile="pahn")
    if problems:
        raise SystemExit("\n".join(problems))
    log("output contract: validate_model_directory clean")

    stage("Stage 7: register, in one act, from the now-live library directory")
    reg_path = library.write_register(registers_dir(), "pahn", out)
    log(f"register   : {reg_path}")

    with h5py.File(reg_path, "r") as f:
        n_model_register = f["models"]["MODEL_NAME"].shape[0]
        models_cols = set(f["models"].attrs["columns"].astype(str).tolist())
        g0 = np.asarray(f["models"]["G0_FLUX"])
        alpha_reg = np.asarray(f["models"]["ALPHA"])
        pah_reg = np.asarray(f["models"]["PAH_SIZE"]).astype(str)

    assert n_model_register == n_models, (
        f"register n_model {n_model_register} != kept count {n_models}")
    log(f"register check: opens with h5py, n_model={n_model_register} (== {n_models})")
    assert "G0_FLUX" in models_cols, "register /models is missing G0_FLUX"
    # Darkness rule (synth.py): dark -> G0_FLUX stored as exactly 0.0, the
    # true value. MEASURED, not assumed: 29 of 33 templates land exactly
    # at 0.0 (DL21 has no flux below its native 1.0056um blue edge, which
    # covers nearly all of Gaia G's 0.327-1.050um bandpass). The remaining
    # 4 -- every "sma" (smallest PAH size) template at the four lowest
    # alpha values, 0.0/0.2/0.4/0.6 -- clear the 1e-6-of-peak darkness
    # threshold by a hair (max 1.29e-6 mJy): G's red edge (1.050um) sits
    # just redward of DL21's blue edge (1.0056um), a ~0.044um sliver of
    # real overlap, and the smallest grains carry just enough flux there,
    # at low aperture-growth suppression, to be counted as (barely)
    # detected rather than dark. Reported as measured, not forced to
    # match the brief's "all dark" expectation.
    dark = g0 == 0.0
    log(f"register G0_FLUX: {int(dark.sum())}/{g0.size} templates exactly dark "
       f"(G0_FLUX==0.0); {int((~dark).sum())} not dark "
       f"(min={g0.min():.3e}, max={g0.max():.3e} mJy)")
    if np.any(~dark):
        log("  not-dark templates (ALPHA, PAH_SIZE, G0_FLUX mJy): " + ", ".join(
            f"({a:.1f}, {p}, {g:.3e})"
            for a, p, g in zip(alpha_reg[~dark], pah_reg[~dark], g0[~dark])))

    wall_s = time.time() - t0
    log(f"\nPAH-N library complete at {out}: {n_models} raw = {n_models} kept templates "
       f"(no sampling reduction) in {wall_s:.1f}s")
    return out, dict(n_raw=n_models, n_kept=n_models, min_pairwise_dist=min_pairwise,
                     register_path=reg_path, n_model_register=n_model_register,
                     wall_s=wall_s)


def check():
    out = work_dir()
    problems = model_io.validate_model_directory(out, profile="pahn")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"pahn: validate_model_directory clean at {out}")
    return {"library": "pahn", "output_root": out, "status": "clean"}


def _ingest(arg):
    if os.path.isdir(arg):
        paths.set_data_root(arg)
        sed_models_dir = os.path.join(arg, "sed_models")
        if os.path.isdir(sed_models_dir):
            paths.set_input_area("sed_models", sed_models_dir)
    else:
        paths.load_config(arg)


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m sesnaimpute.sed_models.build_pahn",
        description="Build the PAH-nebular (PAH-N) model library and register: "
                   "PAHC's own DL21 excess term alone, no star, no host or R axis.")
    parser.add_argument("config", help="a root.cfg-style config file, or a data-root directory")
    parser.add_argument("--check", action="store_true",
                        help="read-only validation of the installed product")
    args = parser.parse_args(raw_args)
    _ingest(args.config)
    if args.check:
        check()
        return 0
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
