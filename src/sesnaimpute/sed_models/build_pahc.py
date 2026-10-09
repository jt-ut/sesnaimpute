"""
sesnaimpute/sed_models/build_pahc.py
====================================================================
ONE-COMMAND driver for the PAH-contaminant (PAH-C) library, coded
against the shared `sed_models.sampling`/`sed_models.library` API
(Unit S, `curation-s`) rather than `pahc_curate.py`'s own
greedy_r_net/pahc_member_statistics/write_*_fits copies, per the
2026-10-08 curation plan (`bms_review/CURATION_PLAN_2026-10-08.md`,
Unit P) and `bms_review/briefs/curation/UNIT_PAHC.md`.

Canonical invocation::

    python3.9 -m sesnaimpute.sed_models.build_pahc <root.cfg path, or a data-root directory>

INPUTS AND VERSIONS
    `downloads/PAHspec` (Draine, Li, Hensley, Hendrix, Smith 2021, ApJ
    917, 3) -- fetched once by `pahc_curate.fetch_pahc_excess_templates`
    against a locally bootstrapped per-file checksum manifest, cached.

    `sps/_raw/` -- Unit SPS's raw CK03+CIFIST+AGSS2009 photosphere
    splice, relocated there in parallel with this unit. AT THE TIME OF
    THIS BUILD `sps/_raw/` DOES NOT YET EXIST, so this driver reads the
    splice from `sed_models/sps/` AS IT STANDS TODAY -- 4,066 rows at
    the top level, confirmed by this driver's own sanity check
    (>= 1,000 rows, >= 3 distinct SOURCE values) before it is trusted as
    the raw splice rather than a down-sampled shipping library -- and
    says so in its own report. Once Unit SPS lands the relocation, this
    docstring's input path should be the only line that needs to change
    (`sps_dir()` below).

PUBLISHED SCIENCE RANGE THIS LINE MUST COVER (printed at the end): the
whole field-star population PAHC partitions against STAR (spec 5.3) --
spectral types O through T, i.e. every host SPS's raw splice carries,
not a closed anchor list -- crossed against the catalogue's own
measured aperture-contamination range, q (the 8um excess-to-star flux
ratio within the IRAC beam) 0.2 to 30.

HOST SPECIFICATION (owner library spec, this brief). Hosts are the
UNION of three sets, each independently justified, drawn from the
Z=0 (fixed PAHC metallicity) subset of the raw SPS splice:

  (a) SHAPE-DISTINCT hosts: `sampling.r_net` (radius 1, the project's
      one SED-sampling scale) run directly over the whole Z=0 splice's
      8-band whitened log flux. This replaces a hand-picked anchor list
      with "every photospheric SHAPE the splice actually contains, at
      the fitter's own resolution" -- the handoff that preceded this
      build measured 27 such hosts; this run reports its own count,
      not a target to match.
  (b) HOT anchors at T_eff = 8,000 / 12,000 / 20,000 / 50,000 K, at
      every log g the splice carries there (inside the splice's own
      log g range, -0.5 to +5.5) -- SHAPE-DEGENERATE BY DESIGN: hot
      photospheres collapse together under the sampling scale (they
      differ from each other far less than one SIGEFF), so admitting
      them costs almost nothing in kept-template count while reaching
      the population's hot end, which is what cures the 6,000 K edge
      the 2026-10-08 library review (`bms_review/studies/
      review_2026-10-08/lib_5_PAHC.md`, concern 1) measured 45.8% of
      confident Orion A PAHC sources piling onto.
  (c) COOL-GIANT anchors at T_eff = 2,800 / 3,000 / 3,300 K, log g 1.0
      (D-15's own floor, carried forward unchanged -- gravity
      sensitivity there is ABOVE the colour floor, so these three are
      not shape-degenerate the way the hot anchors are and must be
      named explicitly rather than relying on the r-net to find them).

This host set is crossed against the PAH component grid AS
`pahc_curate` BUILDS IT -- `PAHC_ALPHA_VALUES` x `PAHC_PAH_SIZE_VALUES`
x `PAHC_R_VALUES` (11 x 3 x 32), unchanged module constants -- via the
existing, already-generic `pahc_curate.build_pahc_raw_grid` /
`build_pahc_raw_band_flux` (linear-separable 8-band convolution, no
per-raw-model SED ever built). The composite physics
(`compute_composite_sed`, DL21 loading, SPS loading) is UNCHANGED;
only the HOST SET and the SAMPLING/WRITER are new.

SAMPLING AND MEMBERS: `sed_models.sampling.whiten`/`r_net`/`assign`/
`coverage`/`packing`/`members_table`, at radius 1.0, replacing
`pahc_curate.greedy_r_net`/`pahc_member_statistics` for this build.
PAHC defines no subclass (`pahc_curate.PAHC_SUBCLASS = "NONE"`), so
`members_table` emits exactly one FRAC column (`FRAC_NONE`, every
value 1.0) and ranges over T_EFF, LOGG, ALPHA, PAH_SIZE (DL21's own
ordinal: sma=0 < std=1 < lrg=2) and R, per the brief.

WRITING: `sed_models.library.write_library` (flux/parameters/classmap/
members in one call) then `model_convolution.build_convolved_band_fits`
(the already-shared higher-level writer PAHC's old driver also called
directly, reading flux.fits back off disk -- `write_library`'s own
`convolved` argument expects pre-convolved per-band kwargs this driver
does not otherwise need to compute by hand) then
`sed_models.library.write_register`, all inside this one process, in
that order -- "one act" in the sense the brief means (one command, no
separate invocation), not a literal single Python call.

STAGING: the whole library is assembled in a scratch directory
(`tempfile.mkdtemp`, outside the data tree) and only `shutil.move`d
into place over the live `pahc/` directory at the very end, on
success -- the live directory is never edited in place, and the
scratch directory is removed on any exit path (including before it is
created, in case a previous run's temp directory is still named by a
stale local variable -- there is no persistent state file naming it
between runs, so there is nothing to clean up from a PRIOR run; this
run's own scratch directory is simply never left behind).

MEMORY. No step here builds a full neighbour-adjacency list:
`sampling.r_net` is the memory-bounded implementation documented in
its own module (rebuilds a KD-tree over the shrinking uncovered subset
each outer iteration; `query_ball_point(..., return_length=True)` for
counts, never a stored index list, plus one single-point indexed query
per selection). `pahc_curate.build_pahc_raw_band_flux` never builds
more than `len(hosts) + len(PAH_SIZE)*len(ALPHA)` individual SEDs
(linear separability in R), regardless of the raw grid's row count.

NO SUBPROCESSES. This driver starts no pool and no subprocess; nothing
to kill on exit.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table

from sesnaimpute.sed_models import paths
from sesnaimpute.sed_models.constants import POINT_SOURCE_APERTURE_AU
from sesnaimpute.sed_models import library, sampling
from sesnaimpute.sed_models.curate import model_io
from sesnaimpute.sed_models.curate import pahc_curate as pc
from sesnaimpute.sed_models.curate.curve_of_growth import peak_and_floor
from sesnaimpute.sed_models.curate.model_convolution import build_convolved_band_fits

CONVOLVED_BANDS = pc.PAHC_BAND_ORDER   # ("J","H","Ks","I1","I2","I3","I4","M1")
R_NET_RADIUS = 1.0

# --- host specification (owner library spec, this brief) ---
HOT_ANCHOR_TEFF_K = (8000.0, 12000.0, 20000.0, 50000.0)
COOL_ANCHOR_TEFF_K = (2800.0, 3000.0, 3300.0)
COOL_ANCHOR_LOGG = 1.0

# --- published science range this line must cover (printed, not asserted:
# no register column carries spectral type, and q is a property of the
# PAH_SIZE x ALPHA x R grid this line already ships whole) ---
SCIENCE_RANGE_SPECTRAL_TYPES = "O through T (the field population STAR sums over)"
SCIENCE_RANGE_Q_MIN = 0.2
SCIENCE_RANGE_Q_MAX = 30.0

# "TIME A 500-MODEL SLICE FIRST" (brief): a complete, small end-to-end
# dry run (host selection already done; a one-host, full-PAH-grid slice
# close to 500 raw rows) through sampling AND the writer, in a throwaway
# scratch directory, before the real build commits to the full raw grid.
SLICE_N_HOSTS = 1
TIME_BUDGET_S = 20 * 60


def work_dir():
    return paths.path_for("sed_models_pahc")


def sps_dir():
    """The SPS RAW splice. `sps/_raw/` landed mid-build (Unit SPS's
    relocation, 2026-10-08) -- confirmed on disk: `sed_models/sps/`
    (top level) is now the SAMPLED/KEPT SPS library (118 rows), and
    `sed_models/sps/_raw/` is the raw CK03+CIFIST+AGSS2009 splice
    (4,066 rows). This driver reads the latter, as its module docstring
    always said it would once `_raw/` existed -- the top-level fallback
    this docstring described earlier in the day is no longer live and
    is not used."""
    raw = os.path.join(paths.path_for("sed_models_sps"), "_raw")
    return raw if os.path.isdir(raw) else paths.path_for("sed_models_sps")


def dl21_dir():
    return os.path.join(paths.input_dir("sed_models"), "downloads", "PAHspec")


def registers_dir():
    return paths.path_for("model_registers_dir")


def pahc_backup_dir():
    return os.path.join(os.path.dirname(work_dir()),
                        "pahc_backup_pre_rebuild_20261008T110641")


# ====================================================================
# Host specification
# ====================================================================

def _load_sps_raw_splice_z0(sps):
    """Every (T_EFF, LOGG) row of the SPS RAW splice at the fixed PAHC
    metallicity (`pc.PAHC_Z` = 0.0), plus a sanity report confirming the
    splice -- not a down-sampled shipping library -- is what was read
    (>= `pc.PAHC_HOST_GRID_MIN_ROWS` rows, >=
    `pc.PAHC_HOST_GRID_MIN_SOURCES` distinct SOURCE values, exactly the
    check `pc.load_pahc_host_grid` already makes, reused here directly
    rather than re-implemented).

    Column names LOGG/Z_H (curation-s 527dc57: HDF5-safe names fixed at
    the source, renamed from LOG[G]/[Z/H] in every writer/reader this
    package owns, including sps_curate's own parameters.fits writer and
    pahc_curate.load_sps_stellar_sed/load_pahc_host_grid -- matched here
    since this is the one place in this driver that reads sps_dir's
    parameters.fits directly rather than through those two functions).

    Returns (hosts, host_name, report) -- `hosts` a sorted list of
    (t_eff, log_g) tuples, each a DISTINCT splice row at Z=0 (the splice
    is already deduplicated upstream of this driver); `host_name` a
    dict (t_eff, log_g) -> that row's own MODEL_NAME in `sps/_raw/`,
    the register's `library/pahc_fstar`'s HOST column (coordinator's
    ruling, this round)."""
    with fits.open(os.path.join(sps, "parameters.fits")) as h:
        t_eff = h[1].data["T_EFF"].astype(float)
        log_g = h[1].data["LOGG"].astype(float)
        z = h[1].data["Z_H"].astype(float)
        source = np.char.strip(h[1].data["SOURCE"].astype(str))
        name = np.char.strip(h[1].data["MODEL_NAME"].astype(str))

    n_rows, n_sources = t_eff.size, len(set(source.tolist()))
    if n_rows < pc.PAHC_HOST_GRID_MIN_ROWS or n_sources < pc.PAHC_HOST_GRID_MIN_SOURCES:
        raise AssertionError(
            f"{sps}/parameters.fits looks like a SAMPLED PAHC/SPS class "
            f"library, not the raw splice: {n_rows} rows, {n_sources} "
            f"distinct SOURCE value(s). sps/_raw/ has not landed yet and "
            f"sed_models/sps/ no longer carries the raw splice either -- "
            f"refusing to pre-coarsen PAHC's host coverage.")

    m = z == pc.PAHC_Z
    hosts = sorted(set(zip(t_eff[m].tolist(), log_g[m].tolist())))
    host_name = {}
    for tt, gg, nn in zip(t_eff[m], log_g[m], name[m]):
        host_name.setdefault((float(tt), float(gg)), nn)
    report = dict(n_raw_rows=n_rows, n_raw_sources=n_sources,
                 sources=sorted(set(source.tolist())), n_z0_rows=int(m.sum()))
    return hosts, host_name, report


def _host_band_flux(hosts, sps, wave_um, bands=CONVOLVED_BANDS):
    """(n_host, len(bands)) convolved flux of the BARE photosphere at
    each host -- the same per-host loop `pc.build_pahc_raw_band_flux`
    already runs internally (`load_sps_stellar_sed` + `resample_sed_log_log`
    + `_independent_band_flux`, the module's own approximate-but-adequate
    convolution it already uses for sampling, not shipping -- see that
    function's docstring), factored out here because host SELECTION
    (this function) needs it on the bare splice, before any PAH
    component or R/alpha axis exists."""
    out = np.empty((len(hosts), len(bands)))
    for i, (t, g) in enumerate(hosts):
        wave_native, flux_native = pc.load_sps_stellar_sed(sps, t, g, pc.PAHC_Z)
        s = pc.resample_sed_log_log(wave_native, flux_native, wave_um)
        for b_i, b in enumerate(bands):
            out[i, b_i] = pc._independent_band_flux(wave_um, s, b)
    return out


def _whiten_flux(flux, bands=CONVOLVED_BANDS):
    """log10(floored flux) + dark mask -> `sampling.whiten` coords, via
    the shared per-model floor (`curve_of_growth.peak_and_floor`, 10 dex
    below each row's OWN peak across these bands) -- the same B0/B0.1
    floor convention `density.derive` applies, which `sampling.py`'s
    own docstring requires the caller to have already applied."""
    cubes = {b: flux[:, i:i + 1] for i, b in enumerate(bands)}
    peak, _, floor_linear, all_zero = peak_and_floor(cubes)
    if np.any(all_zero):
        raise ValueError("a host/raw-grid row has zero flux in every band")
    floored = np.maximum(flux, floor_linear[:, None])
    dark = flux < floor_linear[:, None]
    return sampling.whiten(np.log10(floored), dark)


def select_hosts(sps, wave_um, log=print):
    """The union host set: (a) shape-distinct r-net hosts, (b) hot
    anchors, (c) cool-giant anchors. Returns (hosts, host_name, counts)
    where `hosts` is a sorted list of (t_eff, log_g), `host_name` maps
    each to its `sps/_raw/` MODEL_NAME, and `counts` names the three
    contributions and the union, for the report."""
    z0_hosts, host_name, splice_report = _load_sps_raw_splice_z0(sps)
    log(f"host splice: {splice_report['n_raw_rows']} raw rows, "
       f"{splice_report['n_raw_sources']} source(s) {splice_report['sources']} "
       f"-- {splice_report['n_z0_rows']} at [Z/H]=0")

    # (a) shape-distinct: r-net directly over the whole Z=0 splice
    t0 = time.time()
    flux_z0 = _host_band_flux(z0_hosts, sps, wave_um)
    coords_z0 = _whiten_flux(flux_z0)
    shape_idx = sampling.r_net(coords_z0, radius=R_NET_RADIUS)
    host_r_net_s = time.time() - t0
    shape_hosts = sorted(z0_hosts[i] for i in shape_idx)
    log(f"(a) shape-distinct hosts: r-net of {len(z0_hosts)} Z=0 splice rows "
       f"at radius 1 -> {len(shape_hosts)} kept ({host_r_net_s:.1f}s)")

    # (b) hot anchors: every log g the splice carries at these T_eff, Z=0
    t_eff_arr = np.array([t for t, _ in z0_hosts])
    logg_arr = np.array([g for _, g in z0_hosts])
    hot_hosts = []
    for t_hot in HOT_ANCHOR_TEFF_K:
        sel = np.isclose(t_eff_arr, t_hot)
        hot_hosts.extend((t_hot, g) for g in sorted(logg_arr[sel].tolist()))
    log(f"(b) hot anchors: {HOT_ANCHOR_TEFF_K} K -> {len(hot_hosts)} host rows "
       f"(log g {min(g for _, g in hot_hosts):.1f} to "
       f"{max(g for _, g in hot_hosts):.1f})")

    # (c) cool-giant anchors
    cool_hosts = [(t, COOL_ANCHOR_LOGG) for t in COOL_ANCHOR_TEFF_K]
    missing = [h for h in cool_hosts if h not in set(z0_hosts)]
    if missing:
        raise AssertionError(f"cool-giant anchors not in the splice: {missing}")
    log(f"(c) cool-giant anchors: {cool_hosts}")

    union = sorted(set(shape_hosts) | set(hot_hosts) | set(cool_hosts))
    log(f"host union: {len(shape_hosts)} + {len(hot_hosts)} + {len(cool_hosts)} "
       f"-> {len(union)} distinct hosts after dedup")
    counts = dict(n_shape=len(shape_hosts), n_hot=len(hot_hosts),
                 n_cool=len(cool_hosts), n_union=len(union), r_net_s=host_r_net_s)
    return union, host_name, counts


# ====================================================================
# library/pahc_fstar (coordinator's ruling, replacing sed_models_register
# /pahc.py's per-cell least-squares decomposition): the bare-photosphere
# reference, constructed exactly -- not fitted -- through the SAME real
# convolution path the composites use (sedfitter's convolve_model_dir,
# via model_convolution.build_convolved_band_fits), never the fast
# approximate `_independent_band_flux` sampling uses. A tiny scratch
# "library" (flux.fits/parameters.fits/models.conf only -- everything
# convolve_model_dir needs, nothing convolved/'s own reader doesn't)
# is written, convolved, read back, and discarded -- "call the same
# function on them separately" (the ruling's own second option).
# ====================================================================

def _write_minimal_convolvable(scratch_dir, names, wave_um_desc, freq_hz_desc, values_desc):
    """flux.fits + parameters.fits + models.conf only -- the minimum
    `model_convolution.build_convolved_band_fits` (sedfitter's
    `convolve_model_dir`) needs. No classmap.fits, no members.fits:
    nothing downstream of the convolve call reads them here."""
    os.makedirs(scratch_dir, exist_ok=True)
    model_io.write_flux_cube(
        os.path.join(scratch_dir, "flux.fits"), names=names,
        wave_um_desc=wave_um_desc, freq_hz_desc=freq_hz_desc, values=values_desc,
        distance_cm=pc.pahc_distance_placeholder(),
        apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
        uncertainties=np.zeros_like(values_desc), name_format=pc.MODEL_NAME_FORMAT)
    params = Table()
    params["MODEL_NAME"] = np.asarray(names, dtype=f"U{int(pc.MODEL_NAME_FORMAT[:-1])}")
    params.write(os.path.join(scratch_dir, "parameters.fits"), overwrite=True)
    # version=2 (not the default 1): selects sedfitter's CUBE-based
    # convolution path (_convolve_model_dir_2, reads flux.fits directly).
    # version=1 globs model_dir/seds/*.fits[.gz] -- individual per-model
    # SED files this scratch helper never writes -- and raises "No SEDs
    # found" otherwise.
    model_io.write_models_conf(os.path.join(scratch_dir, "models.conf"),
                              name="scratch (pahc_fstar/pedestal helper)",
                              aperture_dependent=False, version=2)


def _read_convolved_band(lib_dir, band):
    """`(names, flux_mjy)` straight off `convolved/{band}.fits`'s
    `CONVOLVED FLUXES` HDU -- the same file/HDU
    `sed_models_register.library.ModelLibrary.convolved` reads, read
    directly here to avoid instantiating a full `ModelLibrary` (which
    wants a richer directory) for a throwaway scratch library."""
    path = os.path.join(lib_dir, "convolved", f"{band}.fits")
    with fits.open(path, memmap=False) as h:
        tc = h["CONVOLVED FLUXES"].data
        names = np.char.strip(np.asarray(tc["MODEL_NAME"]).astype(str))
        flux = np.atleast_1d(np.asarray(tc["TOTAL_FLUX"], dtype=float)).reshape(-1)
    return names, flux


def _host_sed_values(hosts, sps, wave_um):
    """(n_host, n_wave) resampled bare-photosphere SEDs, ASCENDING
    wave_um -- the same `load_sps_stellar_sed` + `resample_sed_log_log`
    every other step in this driver uses, kept in memory once so the
    fstar table and the 8-um anchor (`host_s8`) are built from the
    SAME arrays rather than two separate re-reads of sps_dir."""
    out = np.empty((len(hosts), wave_um.size), dtype=np.float64)
    for i, (t, g) in enumerate(hosts):
        wave_native, flux_native = pc.load_sps_stellar_sed(sps, t, g, pc.PAHC_Z)
        out[i] = pc.resample_sed_log_log(wave_native, flux_native, wave_um)
    return out


def build_host_fstar(hosts, host_name, sps, wave_um, bands=CONVOLVED_BANDS, log=print):
    """`library/pahc_fstar`: one row per host, HOST (sps/_raw/
    MODEL_NAME) + T_EFF + LOGG + Z_H + the eight REAL convolved band
    fluxes of the bare photosphere -- the exact construction the
    coordinator's ruling specifies, replacing the deleted
    `sed_models_register/pahc.py` fit.

    Also returns `host_s8` (dict (t_eff, log_g) -> F_nu at 8.000um,
    interpolated on this SAME resampled SED, never the nearest grid
    sample) and `host_band` (dict (t_eff, log_g) -> (8,) band flux),
    both needed only by the Stage-9 linear identity check, not by the
    register.
    """
    sed = _host_sed_values(hosts, sps, wave_um)
    s8 = np.array([np.interp(pc.ANCHOR_WAVELENGTH_UM, wave_um, row) for row in sed])

    names = [host_name[h] for h in hosts]
    wave_desc = wave_um[::-1]
    freq_desc = pc.C_UM_S / wave_desc
    values_desc = sed[:, ::-1].astype(np.float32)[:, np.newaxis, :]

    scratch = tempfile.mkdtemp(prefix="pahc_hostfstar_")
    try:
        _write_minimal_convolvable(scratch, names, wave_desc, freq_desc, values_desc)
        build_convolved_band_fits(scratch, bands)
        band_flux = np.empty((len(hosts), len(bands)))
        for b_i, b in enumerate(bands):
            conv_names, conv_flux = _read_convolved_band(scratch, b)
            order = np.array([np.flatnonzero(conv_names == n)[0] for n in names])
            band_flux[:, b_i] = conv_flux[order]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    table = Table()
    table["HOST"] = np.asarray(names, dtype=f"U{int(pc.MODEL_NAME_FORMAT[:-1])}")
    table["T_EFF"] = np.array([h[0] for h in hosts], dtype=float)
    table["LOGG"] = np.array([h[1] for h in hosts], dtype=float)
    table["Z_H"] = np.full(len(hosts), pc.PAHC_Z, dtype=float)
    for b_i, b in enumerate(bands):
        table[b] = band_flux[:, b_i]

    host_s8 = {h: float(s8[i]) for i, h in enumerate(hosts)}
    host_band = {h: band_flux[i] for i, h in enumerate(hosts)}
    log(f"  pahc_fstar: {len(hosts)} hosts, bands {bands}")
    return table, host_s8, host_band


def build_pedestal_table(dl21, wave_um, bands=CONVOLVED_BANDS,
                         alphas=None, pah_sizes=None, log=print):
    """The 33 (ALPHA x PAH_SIZE) unit-pedestal spectra -- `P(lambda;
    alpha, size)` in `compute_composite_sed`'s own notation, i.e.
    `P_hat(lambda)/P_hat(8um) * aperture_factor(lambda, alpha)` -- REAL-
    convolved through the same `build_convolved_band_fits` path.
    Returns `pedestal_band` : dict (alpha, pah_size) -> (8,) band flux.
    """
    alphas = pc.PAHC_ALPHA_VALUES if alphas is None else alphas
    pah_sizes = pc.PAHC_PAH_SIZE_VALUES if pah_sizes is None else pah_sizes

    names, seds = [], []
    for size in pah_sizes:
        wave_native, p_hat_native = pc.load_pahc_excess_shape(dl21, size)
        p_hat = pc.resample_sed_log_log(wave_native, p_hat_native, wave_um)
        p_hat = p_hat / np.interp(pc.ANCHOR_WAVELENGTH_UM, wave_um, p_hat)
        for alpha in alphas:
            seds.append(p_hat * pc.band_aperture_scaling(wave_um, alpha))
            names.append(f"ped_{alpha:.1f}_{size}")

    seds = np.asarray(seds)
    wave_desc = wave_um[::-1]
    freq_desc = pc.C_UM_S / wave_desc
    values_desc = seds[:, ::-1].astype(np.float32)[:, np.newaxis, :]

    scratch = tempfile.mkdtemp(prefix="pahc_pedestal_")
    try:
        _write_minimal_convolvable(scratch, names, wave_desc, freq_desc, values_desc)
        build_convolved_band_fits(scratch, bands)
        band_flux = np.empty((len(names), len(bands)))
        for b_i, b in enumerate(bands):
            conv_names, conv_flux = _read_convolved_band(scratch, b)
            order = np.array([np.flatnonzero(conv_names == n)[0] for n in names])
            band_flux[:, b_i] = conv_flux[order]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    pedestal_band = {}
    i = 0
    for size in pah_sizes:
        for alpha in alphas:
            pedestal_band[(float(alpha), size)] = band_flux[i]
            i += 1
    log(f"  pedestal table: {len(names)} (alpha, size) spectra, bands {bands}")
    return pedestal_band


def check_linear_identity(model_set, host_band, host_s8, pedestal_band, lib_dir,
                          bands=CONVOLVED_BANDS, n_sample=200, tol=1e-4, log=print):
    """Stage 9, replacing the deleted `sed_models_register/pahc.py` fit:
    on a FIXED sample of `n_sample` kept templates, convolution is
    linear, so

        composite_band = host_band + R * host_s8 * pedestal_band

    must hold to `tol` relative, in every band, against the REAL
    convolved composite (read back from `lib_dir/convolved/`, the same
    files the build just wrote). Deterministic sample (evenly strided
    over the kept set, not random). Raises naming the worst (template,
    band) on failure; otherwise returns the worst relative deviation
    and the template/band it occurred at.
    """
    n_kept = len(model_set.model_names)
    idx = np.unique(np.linspace(0, n_kept - 1, min(n_sample, n_kept)).astype(int))
    sample_names = model_set.model_names[idx]

    predicted = np.empty((idx.size, len(bands)))
    for row, k in enumerate(idx):
        host_key = (float(model_set.t_eff[k]), float(model_set.log_g[k]))
        ped_key = (float(model_set.alpha[k]), str(model_set.pah_size[k]).strip())
        predicted[row] = (host_band[host_key]
                         + float(model_set.r[k]) * host_s8[host_key] * pedestal_band[ped_key])

    actual = np.empty((idx.size, len(bands)))
    for b_i, b in enumerate(bands):
        conv_names, conv_flux = _read_convolved_band(lib_dir, b)
        pos = {n: i for i, n in enumerate(conv_names)}
        actual[:, b_i] = [conv_flux[pos[n]] for n in sample_names]

    rel = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-300)
    flat = np.argmax(rel)
    row_w, b_w = np.unravel_index(flat, rel.shape)
    worst = float(rel[row_w, b_w])
    worst_name, worst_band = str(sample_names[row_w]), bands[b_w]
    log(f"  linear identity: {idx.size} sampled kept templates x {len(bands)} bands, "
       f"worst relative deviation {worst:.3e} ({worst_name}, band {worst_band})")
    if worst > tol:
        raise AssertionError(
            f"composite = host + R*host(8um)*pedestal identity failed: worst relative "
            f"deviation {worst:.3e} > {tol:.0e} at template {worst_name!r}, band {worst_band!r}")
    return worst, worst_name, worst_band


# ====================================================================
# One raw-grid build + sample + write, parameterised over the host list
# so the 500-model timing slice and the real build share one code path.
# ====================================================================

def _build_grid_and_sample(hosts, sps, dl21, wave_um, log=print):
    host_idx, alpha_idx, pah_idx, r_idx = pc.build_pahc_raw_grid(hosts)
    n_raw = int(host_idx.size)
    flux = pc.build_pahc_raw_band_flux(
        hosts, host_idx, alpha_idx, pah_idx, r_idx,
        pc.PAHC_ALPHA_VALUES, pc.PAHC_PAH_SIZE_VALUES, pc.PAHC_R_VALUES,
        sps, dl21, wave_um, bands=CONVOLVED_BANDS)
    coords = _whiten_flux(flux)

    t0 = time.time()
    kept = sampling.r_net(coords, radius=R_NET_RADIUS)
    r_net_s = time.time() - t0

    rep_of, dist_to_rep = sampling.assign(coords, kept)
    max_dist, n_uncovered = sampling.coverage(coords, kept, radius=R_NET_RADIUS)
    packing = sampling.packing(coords, kept)
    log(f"  n_raw={n_raw} n_kept={kept.size} r_net={r_net_s:.1f}s "
       f"max_dist={max_dist:.4f} n_uncovered={n_uncovered} packing={packing:.4f}")
    return dict(host_idx=host_idx, alpha_idx=alpha_idx, pah_idx=pah_idx, r_idx=r_idx,
               n_raw=n_raw, flux=flux, coords=coords, kept=kept, rep_of=rep_of,
               dist_to_rep=dist_to_rep, max_dist=max_dist, n_uncovered=n_uncovered,
               packing=packing, r_net_s=r_net_s)


def _assemble_and_write(hosts, grid_result, wave_um, sps, dl21, lib_dir, log=print):
    host_idx, alpha_idx, pah_idx, r_idx = (
        grid_result["host_idx"], grid_result["alpha_idx"],
        grid_result["pah_idx"], grid_result["r_idx"])
    kept_sorted = np.sort(grid_result["kept"])
    n_raw = grid_result["n_raw"]
    rep_of, dist_to_rep = grid_result["rep_of"], grid_result["dist_to_rep"]

    # The kept templates' own grid rows, in ASCENDING raw-index order --
    # the same order sampling.members_table derives internally from
    # np.unique(rep_of), so model_names[k] here IS the k-th kept
    # template members_table will also see.
    kept_grid = [
        {"t_eff": hosts[host_idx[i]][0], "log_g": hosts[host_idx[i]][1],
         "alpha": pc.PAHC_ALPHA_VALUES[alpha_idx[i]],
         "pah_size": pc.PAHC_PAH_SIZE_VALUES[pah_idx[i]],
         "r": pc.PAHC_R_VALUES[r_idx[i]]}
        for i in kept_sorted
    ]
    model_set = pc.prepare_pahc_model_arrays(kept_grid, wave_um, sps, dl21)

    # names_full: only entries at a KEPT raw index are ever read by
    # members_table (it reads names[rep] for rep in np.unique(rep_of)).
    names_full = np.full(n_raw, "", dtype=object)
    for k, raw_i in enumerate(kept_sorted):
        names_full[raw_i] = model_set.model_names[k]

    subclass_full = np.full(n_raw, pc.PAHC_SUBCLASS)
    pah_ordinal = {p: i for i, p in enumerate(pc.PAHC_PAH_SIZE_VALUES)}
    raw_params = Table()
    raw_params["T_EFF"] = np.array([hosts[h][0] for h in host_idx], dtype=float)
    raw_params["LOGG"] = np.array([hosts[h][1] for h in host_idx], dtype=float)
    raw_params["ALPHA"] = np.asarray(pc.PAHC_ALPHA_VALUES, dtype=float)[alpha_idx]
    raw_params["PAH_SIZE"] = np.array(
        [pah_ordinal[pc.PAHC_PAH_SIZE_VALUES[p]] for p in pah_idx], dtype=float)
    raw_params["R"] = np.asarray(pc.PAHC_R_VALUES, dtype=float)[r_idx]

    members = sampling.members_table(names_full, rep_of, dist_to_rep,
                                     subclass_full, raw_params)
    n_members_sum = int(np.sum(members["N_MEMBERS"]))
    frac_sum_ok = bool(np.allclose(members["FRAC_NONE"], 1.0))
    log(f"  members: {len(members)} rows, N_MEMBERS sum={n_members_sum} "
       f"(raw={n_raw}), FRAC_NONE all 1.0: {frac_sum_ok}")
    assert n_members_sum == n_raw, "members N_MEMBERS does not sum to the raw count"
    assert frac_sum_ok, "FRAC_NONE does not sum to one on every row"

    # curation-s 527dc57 fixed write_library's parameters.fits EXTNAME
    # at the source (fits.table_to_hdu + explicit .name = "PARAMETERS").
    # Still forced here: the MODEL_NAME column's WIDTH, so it matches
    # flux.fits' MODEL_NAME_FORMAT (model_io's C10 check) -- a plain
    # Python-string column takes whatever width the longest string
    # happens to be, which is narrower than the release-wide cube width.
    parameters = Table()
    parameters["MODEL_NAME"] = np.asarray(
        model_set.model_names, dtype=f"U{int(pc.MODEL_NAME_FORMAT[:-1])}")
    parameters["T_EFF"] = model_set.t_eff
    parameters["LOGG"] = model_set.log_g
    parameters["ALPHA"] = model_set.alpha
    parameters["PAH_SIZE"] = model_set.pah_size
    parameters["R"] = model_set.r
    assert len(parameters) == len(model_set.model_names) == kept_sorted.size

    flux_kwargs = dict(
        wave_um_desc=model_set.wave_um_desc, freq_hz_desc=model_set.freq_hz_desc,
        values=model_set.values_mjy, distance_cm=pc.pahc_distance_placeholder(),
        apertures_au=np.array([POINT_SOURCE_APERTURE_AU]),
        uncertainties=model_set.uncertainties_mjy,
        header_extras={
            "ZH": (pc.PAHC_Z, "fixed photosphere [Z/H]"),
            "LG_U": (float(pc.PAHC_LG_U_TOKEN), "fixed DL21 log10 starlight intensity"),
            "PAHION": (pc.PAHC_IONIZATION, "fixed DL21 PAH charge distribution"),
            "F_PAH": (pc.PAHC_F_PAH, "fixed PAH abundance scaling (1 = DL21 fiducial)"),
            "RADFIELD": (pc.PAHC_RADIATION_FIELD, "fixed DL21 illuminating spectrum"),
        },
        distance_comment="structural placeholder (aperture-independent)",
        name_format=pc.MODEL_NAME_FORMAT,
    )
    classmap_kwargs = dict(
        class_id=pc.MODEL_CLASS,
        subclass=np.full(len(model_set.model_names), pc.PAHC_SUBCLASS),
        class_legend=pc.CLASS_LEGEND, subclass_legend=pc.SUBCLASS_LEGEND,
        provenance=(
            ("CLASS_SOURCE", "library declaration; all models here are this class"),
            ("SUBCLASS_SOURCE", "none defined; every model carries NONE"),
            ("SUBCLASS_REF", "n/a -- no subclass axis"),
            ("SUBCLASS_NOTE", "grid axes are in parameters.fits, not subclasses"),
            ("LEGEND_SOURCE", "class from GUTERMUTH_LABELS 39; text authored here"),
        ),
    )
    models_conf_kwargs = dict(name=pc.DEFAULT_MODEL_SET_NAME, aperture_dependent=False,
                              version=pc.PAHC_MODELS_CONF_VERSION)

    library.write_library(lib_dir, "pahc", flux_kwargs, parameters, classmap_kwargs,
                          model_set.model_names, members, {}, models_conf_kwargs)
    build_convolved_band_fits(lib_dir, CONVOLVED_BANDS)

    problems = model_io.validate_model_directory(lib_dir, profile="pahc")
    if problems:
        raise SystemExit("\n".join(problems))

    return dict(n_kept=len(model_set.model_names), members=members,
               n_members_sum=n_members_sum, frac_sum_ok=frac_sum_ok,
               model_set=model_set)


# ====================================================================
# Timing slice, then the real build
# ====================================================================

def _time_slice(hosts, sps, dl21, wave_um, log=print):
    """Brief: TIME A 500-MODEL SLICE FIRST. A complete, small end-to-end
    build (one host's full ALPHA x PAH_SIZE x R grid, 1,056 raw rows --
    the smallest unit that still exercises every stage once) through
    sampling AND the writer, in a throwaway scratch directory, timed and
    then discarded. Catches an integration defect before the real
    ~50,000-row build, and gives a per-raw-model rate to project the
    full run's wall time against the brief's 20-minute flag."""
    slice_hosts = hosts[:SLICE_N_HOSTS]
    t0 = time.time()
    gr = _build_grid_and_sample(slice_hosts, sps, dl21, wave_um, log=log)
    scratch = tempfile.mkdtemp(prefix="pahc_slice_")
    try:
        _assemble_and_write(slice_hosts, gr, wave_um, sps, dl21, scratch, log=log)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    elapsed = time.time() - t0
    rate = elapsed / gr["n_raw"]
    log(f"500-model slice (actually {gr['n_raw']} raw rows, {SLICE_N_HOSTS} host(s)): "
       f"{elapsed:.1f}s end-to-end ({rate * 1000:.3f} ms/raw-model)")
    return elapsed, rate


def build():
    import resource

    t_start = time.time()
    out = work_dir()
    sps = sps_dir()
    dl21 = dl21_dir()
    log = print

    def stage(label):
        log(f"[{time.time() - t_start:6.1f}s] {label}")

    stage("Stage 1: DL21 templates")
    pc.fetch_pahc_excess_templates(dl21)

    wave_um = pc.build_pahc_wavelength_grid()

    stage("Stage 2: host specification (a) shape-distinct r-net (b) hot anchors (c) cool-giant anchors")
    hosts, host_name, host_counts = select_hosts(sps, wave_um, log=log)
    host_r_net_s = host_counts.pop("r_net_s")

    stage("Stage 3: TIME A 500-MODEL SLICE FIRST")
    slice_elapsed, rate = _time_slice(hosts, sps, dl21, wave_um, log=log)
    full_n_raw_est = len(hosts) * len(pc.PAHC_ALPHA_VALUES) * len(pc.PAHC_PAH_SIZE_VALUES) * len(pc.PAHC_R_VALUES)
    projected = rate * full_n_raw_est
    log(f"  projected full build ({full_n_raw_est} raw rows): {projected:.0f}s "
       f"({'OK' if projected < TIME_BUDGET_S else 'OVER the 20-minute flag'})")

    stage(f"Stage 4: full raw grid, {len(hosts)} hosts x "
         f"{len(pc.PAHC_ALPHA_VALUES)} alpha x {len(pc.PAHC_PAH_SIZE_VALUES)} size x "
         f"{len(pc.PAHC_R_VALUES)} R")
    t_full_build = time.time()
    gr = _build_grid_and_sample(hosts, sps, dl21, wave_um, log=log)
    full_build_s = time.time() - t_full_build
    stage(f"Stage 5: sampled -- {gr['n_raw']} raw -> {gr['kept'].size} kept "
         f"(r_net {gr['r_net_s']:.1f}s; host r-net was {host_r_net_s:.1f}s)")

    assert gr["max_dist"] < 1.0, f"coverage max_dist {gr['max_dist']} not < 1.0"
    assert gr["n_uncovered"] == 0, f"{gr['n_uncovered']} raw models uncovered"
    assert gr["packing"] >= R_NET_RADIUS, f"packing {gr['packing']} < {R_NET_RADIUS}"

    # Staging (coordinator's ruling, item 4): BOTH the library and the
    # register are assembled fully in scratch -- the register against
    # the SCRATCH library, before either touches the live data tree --
    # and only swapped into place once both exist and every assert
    # below has passed. This is the fix for the earlier run, where the
    # library was moved into place and the register write failed
    # afterward, leaving a library with no matching register.
    scratch_lib = tempfile.mkdtemp(prefix="pahc_build_lib_")
    scratch_reg_dir = tempfile.mkdtemp(prefix="pahc_build_reg_")
    try:
        result = _assemble_and_write(hosts, gr, wave_um, sps, dl21, scratch_lib, log=log)
        stage(f"Stage 6: library assembled in scratch ({result['n_kept']} kept templates)")

        stage("Stage 7: library/pahc_fstar (exact construction, real convolution)")
        pahc_fstar, host_s8, host_band = build_host_fstar(
            hosts, host_name, sps, wave_um, log=log)
        pedestal_band = build_pedestal_table(dl21, wave_um, log=log)

        stage("Stage 8: linear identity check (replaces the deleted pahc.py fit)")
        worst_dev, worst_name, worst_band = check_linear_identity(
            result["model_set"], host_band, host_s8, pedestal_band, scratch_lib, log=log)

        reg_path_scratch = library.write_register(
            scratch_reg_dir, "pahc", scratch_lib, pahc_fstar=pahc_fstar)
        stage(f"Stage 9: register assembled in scratch -> {reg_path_scratch}")

        # ---- every assert against the SCRATCH register, before the swap ----
        required_param_cols = {"T_EFF", "LOGG", "ALPHA", "PAH_SIZE", "R"}
        with h5py.File(reg_path_scratch, "r") as f:
            n_model_register = f["models"]["MODEL_NAME"].shape[0]
            models_cols = set(f["models"].attrs["columns"].astype(str).tolist())
            bands_cols = set(f["bands"].attrs["columns"].astype(str).tolist())
            vega_zp = np.asarray(f["bands"]["VEGA_ZP_MJY"])
            has_subclass_prob = "subclass_prob" in f
            if has_subclass_prob:
                sp_all_cols = f["subclass_prob"].attrs["columns"].astype(str).tolist()
                sp_cols = [c for c in sp_all_cols if c != "MODEL_NAME"]
                sp_values = np.column_stack(
                    [np.asarray(f["subclass_prob"][c]) for c in sp_cols])
            has_pahc_fstar = "pahc_fstar" in f["library"]
            if has_pahc_fstar:
                fstar_cols = set(f["library"]["pahc_fstar"].attrs["columns"]
                                 .astype(str).tolist())
                n_fstar_rows = f["library"]["pahc_fstar"]["HOST"].shape[0]
            no_deleted_groups = not any(
                g in f["library"] for g in ("pahc_f", "pahc_g", "pahc_q_span"))
            no_bracket_subgroup = "[Z" not in f["models"]

        assert n_model_register == result["n_kept"], (
            f"register n_model {n_model_register} != kept count {result['n_kept']}")
        missing_params = required_param_cols - models_cols
        assert not missing_params, f"models group missing parameter columns: {missing_params}"
        assert "SUBCLASS" in models_cols, "models group missing SUBCLASS"
        assert no_bracket_subgroup, "models group has a '[Z' subgroup -- an unsafe column name leaked through"
        assert "VEGA_ZP_MJY" in bands_cols, "bands group missing VEGA_ZP_MJY"
        assert vega_zp.shape == (8,), f"VEGA_ZP_MJY has {vega_zp.shape} values, expected 8"
        assert has_subclass_prob, "register carries no subclass_prob group"
        assert sp_cols == ["NONE"], f"subclass_prob columns {sp_cols} != ['NONE'] (PAHC has no subclass)"
        assert np.allclose(sp_values.sum(axis=1), 1.0), "subclass_prob rows do not sum to one"
        assert has_pahc_fstar, "register carries no library/pahc_fstar table"
        assert n_fstar_rows == len(hosts), (
            f"library/pahc_fstar has {n_fstar_rows} rows, expected {len(hosts)} hosts")
        expected_fstar_cols = {"HOST", "T_EFF", "LOGG", "Z_H"} | set(CONVOLVED_BANDS)
        assert fstar_cols == expected_fstar_cols, (
            f"library/pahc_fstar columns {sorted(fstar_cols)} != "
            f"{sorted(expected_fstar_cols)}")
        assert no_deleted_groups, "a deleted pahc_f/pahc_g/pahc_q_span group reappeared"

        log(f"\nmodels group columns ({len(models_cols)}): {sorted(models_cols)}")
        log(f"models group carries {sorted(required_param_cols)} and SUBCLASS, "
           f"no '[Z' subgroup: confirmed")
        log(f"bands group VEGA_ZP_MJY (8 per-band Vega zero points, mJy): "
           f"{dict(zip(CONVOLVED_BANDS, vega_zp.tolist()))}")
        log(f"subclass_prob: one column {sp_cols}, every row sums to "
           f"{float(sp_values.sum(axis=1).mean()):.6f}")
        log(f"library/pahc_fstar: {n_fstar_rows} rows, columns {sorted(fstar_cols)}; "
           f"no pahc_f/pahc_g/pahc_q_span: confirmed")

        # ---- both scratch artifacts exist and validate -- NOW swap both ----
        if os.path.isdir(out):
            shutil.rmtree(out)
        shutil.move(scratch_lib, out)
        final_reg_path = os.path.join(registers_dir(), "pahc_register.hdf5")
        if os.path.exists(final_reg_path):
            os.remove(final_reg_path)
        os.makedirs(registers_dir(), exist_ok=True)
        shutil.move(reg_path_scratch, final_reg_path)
        reg_path = final_reg_path
        stage(f"Stage 10: swapped into place -- library at {out}, register at {reg_path}")
    finally:
        if os.path.isdir(scratch_lib):
            shutil.rmtree(scratch_lib, ignore_errors=True)
        if os.path.isdir(scratch_reg_dir):
            shutil.rmtree(scratch_reg_dir, ignore_errors=True)

    # Coordinator's word (ruling item 6): delete the distrusted backup
    # ONLY after every assert above has passed -- which, by this point
    # in the function, it has (an AssertionError above would have
    # raised before this line is ever reached, and the `finally` above
    # would already have cleaned the scratch dirs without touching the
    # live tree).
    if os.path.isdir(pahc_backup_dir()):
        shutil.rmtree(pahc_backup_dir())
        log(f"deleted {pahc_backup_dir()}")
    else:
        log(f"{pahc_backup_dir()} not present -- nothing to delete")

    log(f"\nscience range: hosts {SCIENCE_RANGE_SPECTRAL_TYPES}; "
       f"q (8um excess/star ratio) {SCIENCE_RANGE_Q_MIN}-{SCIENCE_RANGE_Q_MAX} "
       f"-- library R axis covers {pc.PAHC_R_VALUES[0]:.3f}-{pc.PAHC_R_VALUES[-1]:.1f}")
    log(f"host counts: shape-distinct={host_counts['n_shape']} "
       f"hot-anchor={host_counts['n_hot']} cool-anchor={host_counts['n_cool']} "
       f"union={host_counts['n_union']}")

    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS (Darwin) reports ru_maxrss in BYTES; Linux reports KB.
    peak_rss_mb = peak_rss_bytes / (1024.0 ** 2 if sys.platform == "darwin" else 1024.0)

    log(f"\nPAHC library complete at {out}: raw={gr['n_raw']} kept={result['n_kept']} "
       f"coverage(max_dist={gr['max_dist']:.4f}, n_uncovered={gr['n_uncovered']}) "
       f"packing={gr['packing']:.4f} full-build={full_build_s:.1f}s "
       f"host-r-net={host_r_net_s:.1f}s full-r-net={gr['r_net_s']:.1f}s "
       f"linear-identity-worst={worst_dev:.3e} "
       f"peak-RSS={peak_rss_mb:.1f}MB total={time.time() - t_start:.1f}s")

    return out, dict(
        n_raw=gr["n_raw"], n_kept=result["n_kept"], max_dist=gr["max_dist"],
        n_uncovered=gr["n_uncovered"], packing=gr["packing"],
        slice_elapsed=slice_elapsed, full_build_s=full_build_s,
        host_r_net_s=host_r_net_s, full_r_net_s=gr["r_net_s"],
        linear_identity_worst=worst_dev, linear_identity_worst_at=(worst_name, worst_band),
        total_s=time.time() - t_start, host_counts=host_counts,
        register_path=reg_path, n_model_register=n_model_register,
        peak_rss_mb=peak_rss_mb,
    )


def check():
    out = work_dir()
    problems = model_io.validate_model_directory(out, profile="pahc")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"pahc: validate_model_directory clean at {out}")
    return {"library": "pahc", "output_root": out, "status": "clean"}


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
        prog="python -m sesnaimpute.sed_models.build_pahc",
        description="Build the PAH-contaminant (PAH-C) model library and register "
                   "from downloads, the raw SPS splice, the composite construction "
                   "and the project's sampling constants. Needs the SPS raw splice "
                   "built/present first.")
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
