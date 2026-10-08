"""The YSO templates' stellar mass (SPEC_BMSTP_DRAFT.md sec 3.5, sec 5.5
"Template weights", sec 10): the one physical quantity the pooled YSO
register lacks that the Chabrier IMF factor needs. Each of the register's
398,458 kept templates carries a luminosity from its own radiative-
transfer grid, sampled independently of any stellar-evolution track; the
template's mass is read off a pre-main-sequence track at the template's
own luminosity, by monotone interpolation of mass against log10 L.

The BHAC15 1 Myr track only reaches 1.4 M⊙ (log10 L = 0.52), well below
most of the register's luminosities; above that top, mass comes from the
MIST v1.2 1 Myr isochrone (Choi et al. 2016) instead, which reaches
298.9 M⊙. The two tracks disagree at the join (MIST's own log10 L at
1.4 M⊙, by its own interpolation, is 0.538, not BHAC15's 0.52): this
discontinuity is disclosed, not patched, and a template just above
BHAC15's top with log10 L below MIST's 1.4 M⊙ value takes 1.4 M⊙ flat.

YSO is one library directory, not five sub-grids (WP-REG-1, coordinator
addendum 2026-10-08): the curated register's own `/models` carries the
A12 physical-parameter join already (`LSUN`, among the Robitaille grid's
own HDF5-safe column names), one row per kept template, so this reads
`LSUN` straight off the SAME register file this module already opens for
`MODEL_NAME`/`SUBCLASS` -- no second file, no join to verify. The old
sub-grid `parameters.fits` files' `star.temperature` column has no
counterpart in the rebuilt register or in `yso/parameters.fits` (checked:
neither carries any temperature column); since nothing downstream ever
read this module's own `T_EFF` output (confirmed by grep: it was "stored
for reference but not used"), that dataset is dropped rather than
written from a guessed name, and the discrepancy is reported upstream.
"""

import os

import h5py
import numpy as np

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run

_STEM = "mass_yso_survey"

#: The BHAC15 age this design derives mass at (SPEC_BMSTP_DRAFT.md sec 10:
#: "isochrone BHAC15, 1 Myr"), in the track file's own age unit (Gyr).
AGE_1MYR_GYR = 0.0010

#: The Chabrier 2003 system IMF's own low-mass floor (SPEC_BMSTP_DRAFT.md
#: sec 5.5, sec 10): a template whose luminosity implies a lower mass
#: than this takes the floor instead.
M_FLOOR_MSUN = 0.1

#: Twenty templates checked per build, chosen by a fixed draw -- arbitrary,
#: fixed so the check is reproducible run to run (CODING_RULES_BMSTP.md
#: rule 4: not a date).
N_CHECK_TEMPLATES = 20
CHECK_SEED = 20135

#: The MIST v1.2 age this design extends the mass-luminosity relation at
#: above BHAC15's top (SPEC_BMSTP_DRAFT.md sec 10's own 1 Myr choice,
#: matched to the MIST grid's own age column, log10(age/yr)).
MIST_LOG10_AGE_YR = 6.0


def _read_register(config):
    """The pooled YSO register's own `MODEL_NAME`, `SUBCLASS` and `LSUN`
    (SPEC_BMSTP_DRAFT.md sec 3.5), in its row order: the curated register's
    `/models` now carries the A12 physical-parameter join (WP-REG-1), so
    the template's own luminosity is read off the SAME file as the join
    key and the class partition -- one open, no sub-grid join to verify."""
    path = f"{config.inputs['sed_models']}/registers/yso_register.hdf5"
    with h5py.File(path, "r") as f:
        names = np.char.decode(f["models"]["MODEL_NAME"][:].astype("S"), "utf-8")
        subclass = np.char.decode(f["models"]["SUBCLASS"][:].astype("S"), "utf-8")
        lsun = f["models"]["LSUN"][:].astype(np.float64)
    return names, subclass, lsun


def _read_bhac15_1myr_track(config):
    """The BHAC15 1 Myr row set (mass, log10 L, T_eff), from the 2MASS
    filter table fetched by `sky.download.baraffe2015_bhac15` (its
    docstring: the author's own age blocks, `! M/Ms Teff L/Ls g R/Rs
    Li/Li0 Mj Mh Mk` columns, `!  t (Gyr) =` headers). Only the age block
    this design uses is kept; the file's other ages and photometry
    columns are not needed here."""
    path = f"{config.data_root}/sky/download/baraffe2015_bhac15/BHAC15_iso.2mass"
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
        raise ValueError(f"yso_mass: no {AGE_1MYR_GYR} Gyr block found in {path}")
    arr = np.array(sorted(rows), dtype=np.float64)
    return arr[:, 0], arr[:, 2], arr[:, 1]  # mass, log10 L, T_eff, mass-sorted


def _read_mist_1myr_track(config):
    """The MIST v1.2 1 Myr isochrone's `initial_mass`, `log_L`, `log_Teff`
    (mass-sorted), from the solar-metallicity, non-rotating basic
    isochrone file fetched by `sky.download.mist2016` (its own EEP-block
    ascii format: `#`-commented headers giving the 25 column names, data
    rows with `log10_isochrone_age_yr` in column 2, `initial_mass` in
    column 3, `log_L` in column 8, `log_Teff` in column 11)."""
    path = (f"{config.data_root}/sky/download/mist2016/"
            "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_basic.iso")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"yso_mass: missing {path}; run RUNBOOKtp.sh's sky.download.mist2016 stage first")
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if abs(float(parts[1]) - MIST_LOG10_AGE_YR) < 1.0e-6:
                rows.append((float(parts[2]), float(parts[7]), float(parts[10])))
    if not rows:
        raise ValueError(f"yso_mass: no log10 age {MIST_LOG10_AGE_YR} isochrone in {path}")
    arr = np.array(sorted(rows), dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]  # mass, log10 L, T_eff, mass-sorted


def _monotone_envelope(track_mass, track_log_l, label):
    """The track's own increasing envelope in log10 L: the mass-sorted
    points at which log10 L exceeds every point below it. Used both ways
    (mass<->log10 L are then a strictly increasing pair), and, where the
    raw track is not monotone in L, this keeps the lowest-mass crossing
    for every L value (the brief's disclosed rule) by construction --
    disclosed here if it ever drops a point."""
    running_max = np.maximum.accumulate(track_log_l)
    is_new_max = np.concatenate(([True], track_log_l[1:] > running_max[:-1]))
    if not np.all(is_new_max):
        bad = np.where(~is_new_max)[0]
        print("yso_mass: %s is not monotone in log10 L at mass index %d "
              "(M=%.4f Msun); using the lowest-mass crossing"
              % (label, bad[0], track_mass[bad[0]]), flush=True)
    return track_mass[is_new_max], track_log_l[is_new_max]


def build(config, regions=None):
    """Writes `population/yso/mass_yso_survey.hdf5`, one row per pooled
    YSO register template, in the register's own order. `regions` is
    accepted for RUNBOOK compatibility and ignored (rule 5c): the register
    and the isochrone are both survey-wide."""
    with progress.Stage("population.yso_mass") as st:
        # WP-REG-1: the register's own LSUN (its `/models`' A12
        # parameter join) is the template's luminosity directly -- no
        # sub-grid join, no row-order check, since `_read_register`'s
        # three arrays come from the SAME file read, row for row.
        reg_names, reg_subclass, reg_lsun = _read_register(config)
        n_register = reg_names.size
        join_log_l = np.log10(reg_lsun)
        join_label = reg_subclass
        print("yso_mass: register n_model=%d, SUBCLASS counts=%s"
              % (n_register, dict(zip(*np.unique(join_label, return_counts=True)))),
              flush=True)

        track_mass, track_log_l, _track_teff = _read_bhac15_1myr_track(config)
        mono_mass, mono_log_l = _monotone_envelope(track_mass, track_log_l, "BHAC15 1 Myr track")
        m_top = float(mono_mass[-1])

        # The high branch (module docstring): MIST v1.2 1 Myr above
        # BHAC15's own top, joined at exactly M_TOP by MIST's own
        # interpolated log10 L there -- the junction discontinuity
        # against BHAC15's 0.52 dex is reported, not removed.
        mist_mass, mist_log_l, _mist_teff = _read_mist_1myr_track(config)
        junction_log_l = float(np.interp(m_top, mist_mass, mist_log_l))
        above = mist_mass > m_top
        hi_mass = np.concatenate(([m_top], mist_mass[above]))
        hi_log_l = np.concatenate(([junction_log_l], mist_log_l[above]))
        mono_mass_hi, mono_log_l_hi = _monotone_envelope(
            hi_mass, hi_log_l, "MIST v1.2 1 Myr isochrone above 1.4 Msun")
        m_top_high = float(mono_mass_hi[-1])
        print("yso_mass: junction at M_TOP=%.4f Msun -- BHAC15 log10 L=%.4f, "
              "MIST log10 L=%.4f (discontinuity %.4f dex)"
              % (m_top, mono_log_l[-1], junction_log_l, junction_log_l - mono_log_l[-1]),
              flush=True)

        # The derivation (sec 3.5, extended): below BHAC15's own top, the
        # mass at which the BHAC15 1 Myr track has the template's own
        # luminosity; above it, the same monotone interpolation against
        # the MIST 1 Myr track instead.
        flag_high = join_log_l > mono_log_l[-1]
        m_star = np.where(
            flag_high,
            np.interp(join_log_l, mono_log_l_hi, mono_mass_hi),
            np.interp(join_log_l, mono_log_l, mono_mass))
        flag_above_top = flag_high & (join_log_l > mono_log_l_hi[-1])
        m_star = np.where(flag_above_top, m_top_high, m_star)
        flag_below_floor = (~flag_high) & (m_star < M_FLOOR_MSUN)
        m_star = np.where(flag_below_floor, M_FLOOR_MSUN, m_star)

        n_above_top_high = int(flag_above_top.sum())
        n_below_floor = int(flag_below_floor.sum())
        n_high = int(flag_high.sum())

        # Check (brief's acceptance #2): twenty templates by a fixed
        # draw, the (BHAC15 or MIST, whichever the template used) track's
        # log10 L at the stored mass reproduces the template's own
        # log10 L for the unflagged, and the relevant track's boundary
        # value for the flagged.
        rng = np.random.default_rng(CHECK_SEED)
        check_idx = rng.choice(n_register, size=N_CHECK_TEMPLATES, replace=False)
        log_l_back = np.where(
            flag_high[check_idx],
            np.interp(m_star[check_idx], mono_mass_hi, mono_log_l_hi),
            np.interp(m_star[check_idx], mono_mass, mono_log_l))
        flagged_check = flag_above_top[check_idx] | flag_below_floor[check_idx]
        floor_log_l = float(np.interp(M_FLOOR_MSUN, mono_mass, mono_log_l))
        boundary_log_l = np.where(flag_above_top[check_idx], mono_log_l_hi[-1], floor_log_l)
        target = np.where(flagged_check, boundary_log_l, join_log_l[check_idx])
        diffs = np.abs(log_l_back.astype(np.float32) - target.astype(np.float32))
        print("yso_mass: check n=%d flagged=%d max|dlog10L| unflagged=%.3g max|dlog10L| flagged=%.3g"
              % (N_CHECK_TEMPLATES, int(flagged_check.sum()),
                 float(diffs[~flagged_check].max()) if (~flagged_check).any() else 0.0,
                 float(diffs[flagged_check].max()) if flagged_check.any() else 0.0),
              flush=True)

        out_path = config_module.product_path(config, "population", "yso", "mass", "survey")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            f.attrs["ISOCHRONE"] = "BHAC15 1 Myr"
            f.attrs["ISOCHRONE_HIGH"] = "MIST v1.2 1 Myr, [Fe/H] 0, v/vcrit 0"
            f.attrs["M_TOP"] = m_top
            f.attrs["M_FLOOR"] = M_FLOOR_MSUN
            f.attrs["M_TOP_HIGH"] = m_top_high
            f.attrs["N_ABOVE_TOP"] = n_above_top_high
            f.attrs["N_ABOVE_TOP_HIGH"] = n_above_top_high
            f.attrs["N_BELOW_FLOOR"] = n_below_floor
            f.attrs["N_HIGH"] = n_high
            for name, data in (
                ("MODEL_NAME", np.char.encode(reg_names, "utf-8")),
                ("M_STAR", m_star.astype(np.float32)),
                ("LOG10_L", join_log_l.astype(np.float32)),
                ("SUBGRID", np.char.encode(join_label, "utf-8")),
                ("FLAG_ABOVE_TOP", flag_above_top.astype(np.int8)),
                ("FLAG_BELOW_FLOOR", flag_below_floor.astype(np.int8)),
                ("FLAG_HIGH", flag_high.astype(np.int8)),
            ):
                build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])

        q = np.percentile(m_star, [10, 50, 90, 99])
        st.done(out_path, n_register=n_register, m_top=m_top, m_top_high=m_top_high,
                 n_high=n_high, n_above_top_high=n_above_top_high, n_below_floor=n_below_floor)
    print("yso_mass: M_STAR range [%.4f, %.4f] Msun, quantiles p10=%.4f p50=%.4f "
          "p90=%.4f p99=%.4f -> %s"
          % (float(m_star.min()), float(m_star.max()), q[0], q[1], q[2], q[3], out_path),
          flush=True)


if __name__ == "__main__":
    run(build)
