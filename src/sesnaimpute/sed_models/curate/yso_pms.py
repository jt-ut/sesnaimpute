"""
yso_pms.py
====================================================================
The 1 Myr pre-main-sequence (PMS) gate (owner's library spec, 2026-10-08):
"a template whose stellar parameters are off the track at any mass is
removed."

WHY. Robitaille's radiative-transfer grid samples `star.temperature` and
`Source Luminosity` as independent free parameters -- it imposes no
stellar-evolution constraint at all (`lib_2_YSO.md` review: "the grid
imposes no pre-main-sequence track"). A real 1 Myr pre-main-sequence
star's (T_eff, L) pair is NOT free: it sits on the isochrone, one point
per mass. A template whose (T_eff, L) combination matches no mass on
that isochrone is not a model of anything a 1 Myr YSO population
produces, however plausible either coordinate looks on its own.

THE TRACK, both already on disk and already used by
`sesnaimpute.population.yso_mass` for exactly this isochrone (same
files, same age points; read directly here rather than imported, since
`sesnaimpute` is a different package) -- this module does not re-derive
the isochrone, it re-reads the same two files:

    BHAC15 1 Myr (Baraffe et al. 2015): 0.1 Msun (this project's own
    IMF floor) up to ~1.4 Msun, from the "t (Gyr) = 0.0010" block of
    <data_root>/sky/download/baraffe2015_bhac15/BHAC15_iso.2mass
    (columns M/Ms, Teff[K], L/Ls, in that order).

    MIST v1.2 1 Myr (Choi et al. 2016), joined at BHAC15's own top mass
    (M_TOP) and extending to ~298.9 Msun, from the log10(age/yr)=6.0
    block of <data_root>/sky/download/mist2016/
    MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_basic.iso (columns
    initial_mass, log_L, log_Teff at positions 2/7/10).

Mass -> log10(L) is monotone by construction along a 1 Myr PMS track
(the same property `yso_mass.py` relies on): inverting it gives a
unique mass for every luminosity. Mass -> T_eff is NOT assumed
monotone; it is read directly off the same, mass-ordered track rows,
with a BHAC15/MIST junction exactly at M_TOP (continuous by
construction, mirroring `yso_mass.py`'s own join).

THE TEST. For a template with its own (T_eff_obs, L_obs):
    1. m = mass such that the track's log10(L) at mass m equals
       log10(L_obs)  (monotone inversion, floored at the IMF floor
       0.1 Msun and capped at the MIST top, exactly as `yso_mass.py`).
    2. T_eff_track = the track's own T_eff at that same mass m
       (direct interpolation against mass, no inversion).
    3. valid_pms = |log10(T_eff_obs) - log10(T_eff_track)| <= PMS_TOL_DEX.

PMS_TOL_DEX is a disclosed, round-number choice (log10(2) = a factor of
two in T_eff), not a value measured or specified anywhere -- there is no
existing number in this project's record to reuse for a joint (T,L)
agreement tolerance, and the task does not hand one down. It is applied
uniformly, with no per-geometry or per-stage exception, and the measured
removal rate is reported rather than tuned to match any particular
number.
====================================================================
"""

import os

import numpy as np

try:
    from astropy.io import fits
except ImportError:
    fits = None

#: BHAC15 age block used (Gyr) -- "t (Gyr) = 0.0010" in the file's own header lines.
AGE_1MYR_GYR = 0.0010

#: MIST age block used -- log10(age/yr) = 6.0, i.e. 1 Myr.
MIST_LOG10_AGE_YR = 6.0

#: This project's own IMF/track floor (matches `yso_mass.M_FLOOR_MSUN`):
#: a template whose L implies a lower mass than this takes the floor's
#: own track point instead of extrapolating below it.
M_FLOOR_MSUN = 0.1

#: Disclosed, round-number T_eff agreement tolerance (see module
#: docstring) -- a factor of two, i.e. log10(2).
PMS_TOL_DEX = float(np.log10(2.0))


def _bhac15_path(data_root):
    return os.path.join(data_root, "sky", "download", "baraffe2015_bhac15",
                        "BHAC15_iso.2mass")


def _mist_path(data_root):
    return os.path.join(data_root, "sky", "download", "mist2016",
                        "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_basic.iso")


def read_bhac15_1myr_track(data_root):
    """BHAC15 1 Myr track, mass-sorted: (mass, teff_K, log10_L).

    Same file, same age selection as `sesnaimpute.population.yso_mass.
    _read_bhac15_1myr_track` -- read directly here rather than imported
    (different package)."""
    path = _bhac15_path(data_root)
    rows = []
    age = None
    with open(path) as f:
        for line in f:
            if "t (Gyr)" in line:
                age = float(line.split("=")[1])
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            if age is not None and abs(age - AGE_1MYR_GYR) < 1.0e-6:
                mass, teff, log_l = (float(x) for x in stripped.split()[:3])
                rows.append((mass, teff, log_l))
    if not rows:
        raise ValueError(f"yso_pms: no {AGE_1MYR_GYR} Gyr block found in {path}")
    arr = np.array(sorted(rows), dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]   # mass, teff_K, log10_L


def read_mist_1myr_track(data_root):
    """MIST v1.2 1 Myr isochrone, mass-sorted: (mass, teff_K, log10_L).

    `log_Teff` (file column 10) is converted to linear Kelvin here so
    both tracks share one T_eff convention -- `yso_mass.py` never needed
    this conversion because it discards MIST's T_eff column entirely."""
    path = _mist_path(data_root)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"yso_pms: missing {path}")
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if abs(float(parts[1]) - MIST_LOG10_AGE_YR) < 1.0e-6:
                mass, log_l, log_teff = float(parts[2]), float(parts[7]), float(parts[10])
                rows.append((mass, 10.0 ** log_teff, log_l))
    if not rows:
        raise ValueError(f"yso_pms: no log10 age {MIST_LOG10_AGE_YR} isochrone in {path}")
    arr = np.array(sorted(rows), dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]   # mass, teff_K, log10_L


def _monotone_envelope(track_mass, track_log_l):
    """The track's own increasing envelope in log10 L (mass-sorted input)
    -- identical rule to `yso_mass._monotone_envelope`, duplicated here
    (different package) rather than imported."""
    running_max = np.maximum.accumulate(track_log_l)
    is_new_max = np.concatenate(([True], track_log_l[1:] > running_max[:-1]))
    return track_mass[is_new_max], track_log_l[is_new_max]


class PMSTrack:
    """The joined BHAC15 (<= M_TOP) + MIST (> M_TOP) 1 Myr track, built
    once and reused for every template -- `mass_from_log10_l` (monotone
    inversion) and `teff_from_mass` (direct interpolation against mass,
    continuous at the junction)."""

    def __init__(self, data_root):
        b_mass, b_teff, b_logl = read_bhac15_1myr_track(data_root)
        mono_mass, mono_logl = _monotone_envelope(b_mass, b_logl)
        self.m_top = float(mono_mass[-1])

        m_mass, m_teff, m_logl = read_mist_1myr_track(data_root)
        junction_logl = float(np.interp(self.m_top, m_mass, m_logl))
        above = m_mass > self.m_top
        hi_mass = np.concatenate(([self.m_top], m_mass[above]))
        hi_logl = np.concatenate(([junction_logl], m_logl[above]))
        mono_mass_hi, mono_logl_hi = _monotone_envelope(hi_mass, hi_logl)
        self.m_top_high = float(mono_mass_hi[-1])

        self._mono_mass, self._mono_logl = mono_mass, mono_logl
        self._mono_mass_hi, self._mono_logl_hi = mono_mass_hi, mono_logl_hi
        self.junction_discontinuity_dex = junction_logl - float(mono_logl[-1])

        # Direct (mass, Teff) interpolants, continuous at M_TOP by
        # construction: the low branch is BHAC15's own raw, mass-sorted
        # rows up to and including M_TOP (= mono_mass[-1], which is
        # b_mass[-1] whenever the monotone-in-L envelope keeps every
        # BHAC15 row, the normal case); the high branch starts from that
        # same (M_TOP, Teff) point and continues with every MIST row
        # above M_TOP -- not the L-monotone envelope, which can drop
        # rows but never moves the mass or Teff of the ones it keeps.
        self._lo_mass_teff, self._lo_teff = b_mass, b_teff
        above_raw = m_mass > self.m_top
        teff_at_m_top = float(np.interp(self.m_top, b_mass, b_teff))
        self._hi_mass_teff = np.concatenate(([self.m_top], m_mass[above_raw]))
        self._hi_teff = np.concatenate(([teff_at_m_top], m_teff[above_raw]))

    def mass_from_log10_l(self, log10_l):
        """Monotone inversion of the joined track, floored at
        `M_FLOOR_MSUN` and capped at the MIST top -- same rule as
        `yso_mass.build`."""
        log10_l = np.asarray(log10_l, dtype=float)
        flag_high = log10_l > self._mono_logl[-1]
        m = np.where(
            flag_high,
            np.interp(log10_l, self._mono_logl_hi, self._mono_mass_hi),
            np.interp(log10_l, self._mono_logl, self._mono_mass))
        m = np.where(flag_high & (log10_l > self._mono_logl_hi[-1]), self.m_top_high, m)
        m = np.where((~flag_high) & (m < M_FLOOR_MSUN), M_FLOOR_MSUN, m)
        return m

    def teff_from_mass(self, mass):
        """T_eff at `mass`, by direct interpolation of the track's own
        mass-ordered rows (BHAC15 below M_TOP, MIST above, continuous at
        the junction). No inversion, no monotonicity assumed in T_eff."""
        mass = np.asarray(mass, dtype=float)
        lo = np.interp(mass, self._lo_mass_teff, self._lo_teff)
        hi = np.interp(mass, self._hi_mass_teff, self._hi_teff)
        return np.where(mass <= self.m_top, lo, hi)

    def predicted_teff(self, log10_l):
        """T_eff a 1 Myr PMS star of this luminosity would have."""
        return self.teff_from_mass(self.mass_from_log10_l(log10_l))


def read_stellar_columns(geometry_dir, row_index):
    """`star.temperature` [K] and `Source Luminosity` [Lsun] at `row_index`
    positions into this geometry's own `parameters.fits` -- the same
    positional-row convention `yso_fps.read_harmonized_params` and
    `make_model_id` use (parameters.fits/info.fits/flux.fits share
    byte-identical row order per geometry, see `yso_curation_state.py`'s
    module docstring), so no name-based alignment is needed here."""
    if fits is None:
        raise RuntimeError("astropy is required to read FITS files.")
    with fits.open(os.path.join(geometry_dir, "parameters.fits"), memmap=True) as h:
        teff = np.asarray(h[1].data["star.temperature"], dtype=np.float64)[row_index]
        lum = np.asarray(h[1].data["Source Luminosity"], dtype=np.float64)[row_index]
    return teff, lum


def off_track_flag(teff_obs, lum_obs, track, tol_dex=PMS_TOL_DEX):
    """valid_pms (1 pass / 0 fail) plus the diagnostic deviation, for
    arrays of observed `star.temperature` [K] / `Source Luminosity`
    [Lsun]. A non-positive or non-finite luminosity/temperature (the
    small already-invalid placeholder rows other gates also catch)
    fails here too, rather than raising -- `valid_star`/`valid_stage`
    already exclude those rows from the pipeline's pool; this gate does
    not need to special-case them to stay well-defined.

    Returns (valid_pms: int8 array, delta_dex: float array, mass: float
    array) -- `delta_dex` and `mass` are kept for the curation-state
    metadata/report, not written per-row."""
    teff_obs = np.asarray(teff_obs, dtype=float)
    lum_obs = np.asarray(lum_obs, dtype=float)
    ok = np.isfinite(teff_obs) & (teff_obs > 0) & np.isfinite(lum_obs) & (lum_obs > 0)

    delta_dex = np.full(teff_obs.shape, np.nan)
    mass = np.full(teff_obs.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_l = np.where(ok, np.log10(np.where(lum_obs > 0, lum_obs, 1.0)), np.nan)
        m = track.mass_from_log10_l(np.where(ok, log10_l, 0.0))
        teff_pred = track.teff_from_mass(m)
        delta = np.where(ok, np.log10(np.where(teff_obs > 0, teff_obs, 1.0))
                        - np.log10(np.where(teff_pred > 0, teff_pred, 1.0)), np.nan)
    delta_dex[ok] = delta[ok]
    mass[ok] = m[ok]

    valid = np.zeros(teff_obs.shape, dtype="int8")
    valid[ok] = (np.abs(delta_dex[ok]) <= tol_dex).astype("int8")
    return valid, delta_dex, mass
