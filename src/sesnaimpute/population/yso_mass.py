"""The YSO templates' stellar mass (SPEC_BMSTP_DRAFT.md sec 3.5, sec 5.5
"Template weights", sec 10): the one physical quantity the pooled YSO
register lacks that the Chabrier IMF factor needs. Each of the register's
200,000 templates carries a luminosity and a temperature from its own
radiative-transfer grid, sampled independently of any stellar-evolution
track; the template's mass is read off the BHAC15 1 Myr pre-main-sequence
track at the template's own luminosity, by monotone interpolation of mass
against log10 L. Temperature is stored beside the mass but never used: it
is the grid's own freely-sampled `star.temperature`, not a track quantity.

The five sub-grid `parameters.fits` files (`c0`, `cI`, `cII`, `cIII`,
`td`), read in that order and concatenated, reproduce the pooled
register's own row order exactly (`yso_register.hdf5`'s
`members/MEMBER_KEY`, `N_MODELS`, `ROW_OFFSET`); the join is verified
against the register's own `MODEL_NAME`, not assumed.
"""

import os

import h5py
import numpy as np
from astropy.io import fits

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute.build import run

#: The five YSO sub-grids, in the pooled register's own row order
#: (`yso_register.hdf5`'s `members/MEMBER_KEY`), with the `SUBCLASS`
#: label each carries in that register.
_SUBGRIDS = (("c0", "C0"), ("cI", "CI"), ("cII", "CII"),
             ("cIII", "CIII"), ("td", "TD"))

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


def _read_register(config):
    """The pooled YSO register's own `MODEL_NAME` and `SUBCLASS`, in its
    row order (SPEC_BMSTP_DRAFT.md sec 3.5: the register carries no
    physical parameters, only the join key and the class partition)."""
    path = f"{config.inputs['sed_models']}/registers/yso_register.hdf5"
    with h5py.File(path, "r") as f:
        names = np.char.decode(f["models"]["MODEL_NAME"][:].astype("S"), "utf-8")
        subclass = np.char.decode(f["models"]["SUBCLASS"][:].astype("S"), "utf-8")
    return names, subclass


def _read_subgrid(config, subdir):
    """One sub-grid's `MODEL_NAME`, `log10(Source Luminosity)` and
    `star.temperature` (SPEC_BMSTP_DRAFT.md sec 3.5: the three physical
    columns on disk, `Source Luminosity` in L_sun, linear)."""
    path = f"{config.inputs['sed_models']}/yso/{subdir}/parameters.fits"
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"yso_mass: missing {path}; run RUNBOOK.sh's YSO curation stage first")
    with fits.open(path) as hdul:
        d = hdul[1].data
        names = np.char.strip(d["MODEL_NAME"].astype(str))
        log_l = np.log10(d["Source Luminosity"].astype(np.float64))
        t_eff = d["star.temperature"].astype(np.float64)
    return names, log_l, t_eff


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


def _monotone_envelope(track_mass, track_log_l):
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
        print("yso_mass: BHAC15 1 Myr track is not monotone in log10 L at mass "
              "index %d (M=%.4f Msun); using the lowest-mass crossing"
              % (bad[0], track_mass[bad[0]]), flush=True)
    return track_mass[is_new_max], track_log_l[is_new_max]


def build(config, regions=None):
    """Writes `population/yso/mass_yso_survey.hdf5`, one row per pooled
    YSO register template, in the register's own order. `regions` is
    accepted for RUNBOOK compatibility and ignored (rule 5c): the register
    and the isochrone are both survey-wide."""
    with progress.Stage("population.yso_mass") as st:
        reg_names, reg_subclass = _read_register(config)
        n_register = reg_names.size

        # The join: each sub-grid's own MODEL_NAME set, concatenated in
        # the register's own sub-grid order, checked against the
        # register's MODEL_NAME row for row (SPEC_BMSTP_DRAFT.md sec 3.5).
        sub_names, sub_log_l, sub_teff, sub_label, sub_counts = [], [], [], [], {}
        for subdir, label in _SUBGRIDS:
            names, log_l, t_eff = _read_subgrid(config, subdir)
            sub_names.append(names)
            sub_log_l.append(log_l)
            sub_teff.append(t_eff)
            sub_label.append(np.full(names.size, label))
            sub_counts[label] = names.size
        join_names = np.concatenate(sub_names)
        join_log_l = np.concatenate(sub_log_l)
        join_teff = np.concatenate(sub_teff)
        join_label = np.concatenate(sub_label)

        n_matched = int(np.sum(join_names == reg_names)) if join_names.size == n_register else 0
        subgrid_sum = sum(sub_counts.values())
        print("yso_mass: join n_matched=%d n_register=%d subgrid_sum=%d subgrid_counts=%s"
              % (n_matched, n_register, subgrid_sum, sub_counts), flush=True)
        if n_matched != n_register or subgrid_sum != n_register:
            raise ValueError(
                "yso_mass: the pooled register and the five sub-grid parameters.fits "
                "files do not join row for row (n_matched=%d, n_register=%d, "
                "subgrid_sum=%d)" % (n_matched, n_register, subgrid_sum))
        if not np.array_equal(join_label, reg_subclass):
            raise ValueError("yso_mass: sub-grid SUBCLASS labels disagree with the register")

        track_mass, track_log_l, _track_teff = _read_bhac15_1myr_track(config)
        mono_mass, mono_log_l = _monotone_envelope(track_mass, track_log_l)
        m_top = float(mono_mass[-1])

        # The derivation (sec 3.5): the mass at which the 1 Myr track has
        # the template's own luminosity, by monotone interpolation of
        # mass against log10 L.
        m_star = np.interp(join_log_l, mono_log_l, mono_mass)
        flag_above_top = join_log_l > mono_log_l[-1]
        m_star = np.where(flag_above_top, m_top, m_star)
        flag_below_floor = m_star < M_FLOOR_MSUN
        m_star = np.where(flag_below_floor, M_FLOOR_MSUN, m_star)

        n_above_top = int(flag_above_top.sum())
        n_below_floor = int(flag_below_floor.sum())

        # Check (brief's acceptance #2): twenty templates by a fixed
        # draw, the track's log10 L at the stored mass reproduces the
        # template's own log10 L for the unflagged, and the track's
        # boundary value for the flagged.
        rng = np.random.default_rng(CHECK_SEED)
        check_idx = rng.choice(n_register, size=N_CHECK_TEMPLATES, replace=False)
        log_l_back = np.interp(m_star[check_idx], mono_mass, mono_log_l)
        flagged_check = flag_above_top[check_idx] | flag_below_floor[check_idx]
        # For the unflagged, the round trip should reproduce the
        # template's own log10 L; for the flagged, the stored mass is a
        # boundary mass (M_TOP or M_FLOOR), so the round trip should
        # reproduce the track's own value there, not the template's own
        # (out-of-range) luminosity.
        boundary_log_l = np.where(flag_above_top[check_idx], mono_log_l[-1],
                                   np.interp(M_FLOOR_MSUN, mono_mass, mono_log_l))
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
            f.attrs["M_TOP"] = m_top
            f.attrs["M_FLOOR"] = M_FLOOR_MSUN
            f.attrs["N_ABOVE_TOP"] = n_above_top
            f.attrs["N_BELOW_FLOOR"] = n_below_floor
            f.create_dataset("MODEL_NAME", data=np.char.encode(reg_names, "utf-8"))
            f.create_dataset("M_STAR", data=m_star.astype(np.float32))
            f.create_dataset("LOG10_L", data=join_log_l.astype(np.float32))
            f.create_dataset("T_EFF", data=join_teff.astype(np.float32))
            f.create_dataset("SUBGRID", data=np.char.encode(join_label, "utf-8"))
            f.create_dataset("FLAG_ABOVE_TOP", data=flag_above_top.astype(np.int8))
            f.create_dataset("FLAG_BELOW_FLOOR", data=flag_below_floor.astype(np.int8))

        st.done(out_path, n_register=n_register, m_top=m_top, n_above_top=n_above_top,
                 n_below_floor=n_below_floor)
    print("yso_mass: M_STAR range [%.4f, %.4f] Msun -> %s"
          % (float(m_star.min()), float(m_star.max()), out_path), flush=True)


if __name__ == "__main__":
    run(build)
