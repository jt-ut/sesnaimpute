"""The measured emission table E[k, b, v] (SPEC_BMSTP_DRAFT.md sec 6.5): for
each library's own templates, the fraction of subclass k's templates that
Gutermuth's colour cascade (`gutcolors.crisp.classify_crisp`) calls verdict
v when scaled to sit at apparent 4.5 micron flux bin b
(`bmstp.grid.LOG10_F45_EDGES`). It is the measured object sec 6.5's Psi
will read once the fitter's own comparison (`fittp.cascade`) turns the
cascade on the survey's photometry; this module never touches a source.

Run at sigma = 0 (the crisp limit) with `valid = detected` = every band
True (the model SED is complete): `gutcolors.route.route` admits Phase 1
whenever all four IRAC bands are usable and Phase 2 only when I3 or I4 is
NOT usable (`route.py`, disjoint by construction), so an all-bands-true
template always routes to Phase 1 and never to Phase 2. Phase 2 is where
the cascade's own extinction correction lives -- `gutcolors.featurize.
deredden` solves for E(H-K) by intersecting the observed colour with an
intrinsic-colour locus and subtracts it from Ks-[3.6], [3.6]-[4.5] and
[3.6] before those rows are tested. `featurize.featurize` still computes
those dereddened features for every row (`crisp.evaluate_rows` evaluates
all 45 rows unconditionally), but `crisp.run_schedule` only acts on a
Phase-2 schedule entry when the source's route is Phase 2, which an
all-bands-true template never is, so the dereddened rows are computed and
then never consulted. `valid = detected` = all bands True is therefore
the argument (passed to `classify_crisp`) that turns the cascade's
extinction correction off: the library's templates are intrinsic SEDs
with no reddening applied, and the template's own reddening -- if any is
ever added -- is the fit's business (EMISSION.md), not this table's.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute.build import run
from sesnaimpute.bmstp import grid
from sesnaimpute.gutcolors import crisp

#: The six libraries, `fittp.prior_reader._LIB`'s own values (EMISSION.md
#: "Module and runbook").
LIBRARIES = ("sps", "agb", "pahc", "galz", "yso", "h2shock")

#: `definitions.BANDS`' order (J H Ks I1 I2 I3 I4 M1) is `gutcolors.crisp`'s
#: own BAND_ORDER (crisp.py module docstring); I2 is 4.5 micron
#: (`definitions.BANDS[4].wvl_um == 4.493`).
_BAND_KEYS = tuple(b.key for b in definitions.BANDS)
_I2 = _BAND_KEYS.index("I2")
N_LABELS = len(crisp.LABELS)
_UNCLASSIFIED = crisp.LABEL_INDEX["UNCLASSIFIED"]

#: Per-library working-set: one (n_model, 8) float64 flux array per bin,
#: rebuilt from the register's own F_REF each bin (rule 10b's per-batch
#: estimate) -- 200,000 templates x 8 bands x 8 bytes is 12.8 MB, well
#: under the ceiling, and the register itself (loaded once, not per bin)
#: is the same size again.
HAND_CHECK_N = 20
HAND_CHECK_SEED = 0


def _read_register(config, key):
    """This library's own reference SED: `F_REF` (n_model, 8) mJy in
    `definitions.BANDS` order, `FLOOR_LINEAR` (the register's own raw-flux
    floor, `fittp.sweep._register`'s FREFRAW convention) and `SUBCLASS`,
    all in the register's own row order."""
    path = os.path.join(config.inputs["sed_models"], "registers", "%s_register.hdf5" % key)
    if not os.path.exists(path):
        raise ValueError("fittp.emission: no register at %r -- build the %s SED model "
                          "library register first" % (path, key))
    with h5py.File(path, "r") as f:
        m = f["models"]
        n_model = m["MODEL_NAME"].shape[0]
        f_ref = np.empty((n_model, len(_BAND_KEYS)), dtype=np.float64)
        for j, bkey in enumerate(_BAND_KEYS):
            f_ref[:, j] = np.asarray(m["F_REF_%s" % bkey][:], dtype=np.float64)
        floor_linear = np.asarray(m["FLOOR_LINEAR"][:], dtype=np.float64)
        subclass_raw = m["SUBCLASS"][:]
    subclass = np.array([s.decode() if isinstance(s, bytes) else s for s in subclass_raw])
    return f_ref, floor_linear, subclass


def _subclass_order(subclass):
    """The register's own subclass set, in first-appearance row order
    (EMISSION.md identity 1: "SUBCLASSES equals the register's subclass
    set in the register's order")."""
    _, first = np.unique(subclass, return_index=True)
    return subclass[np.sort(first)]


def _scale_to_bin(f_ref, f45_floored, log10_target):
    """The (n_model, 8) SED with every template's I2 (4.5 micron) flux
    scaled to sit at `10**log10_target` mJy (EMISSION.md "Computation"):
    linear in the register's own F_REF, so every band moves by the same
    per-template factor and the SED's shape -- its subclass's colours --
    is unchanged."""
    scale = (10.0 ** log10_target) / f45_floored
    return f_ref * scale[:, None]


def _hand_check_pairs(n_model, n_bin, rng):
    """20 random (template, bin) pairs (EMISSION.md identity 4)."""
    theta = rng.integers(0, n_model, size=HAND_CHECK_N)
    b = rng.integers(0, n_bin, size=HAND_CHECK_N)
    return theta, b


def _emission_table(f_ref, floor_linear, subclass, st):
    """`E[k, b, v]`, `SUBCLASSES`, and the identity-4 hand check, for one
    library (EMISSION.md "Computation"). One `classify_crisp` call per
    brightness bin, vectorised over every template in the library; no
    population weight (a library property, not the luminosity function,
    which stays in P5)."""
    n_model = f_ref.shape[0]
    edges = grid.LOG10_F45_EDGES
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bin = centers.size

    subclasses = _subclass_order(subclass)
    n_sub = subclasses.size
    sub_onehot = (subclass[:, None] == subclasses[None, :]).astype(np.float64)  # (n_model, n_sub)

    f45_floored = np.maximum(f_ref[:, _I2], floor_linear)
    valid = np.ones((n_model, len(_BAND_KEYS)), dtype=bool)
    sigma = np.zeros((n_model, len(_BAND_KEYS)), dtype=np.float64)

    rng = np.random.default_rng(HAND_CHECK_SEED)
    hand_theta, hand_bin = _hand_check_pairs(n_model, n_bin, rng)
    hand_stored = np.full(HAND_CHECK_N, -1, dtype=np.int64)

    counts = np.zeros((n_sub, n_bin, N_LABELS), dtype=np.float64)
    for b in range(n_bin):
        flux = _scale_to_bin(f_ref, f45_floored, centers[b])
        label = crisp.classify_crisp(flux, sigma, valid=valid, detected=valid).label
        onehot_v = (label[:, None] == np.arange(N_LABELS)[None, :]).astype(np.float64)
        counts[:, b, :] = sub_onehot.T @ onehot_v

        here = hand_bin == b
        if here.any():
            hand_stored[here] = label[hand_theta[here]]
        st.tick(b + 1, n_bin, "brightness bins")

    totals = counts.sum(axis=2, keepdims=True)
    E = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0).astype(np.float32)

    # identity 4: a direct, single-row crisp call on the same scaled SED,
    # independent of the batched per-bin loop above.
    hand_direct = np.full(HAND_CHECK_N, -1, dtype=np.int64)
    for i in range(HAND_CHECK_N):
        theta, b = int(hand_theta[i]), int(hand_bin[i])
        flux_row = _scale_to_bin(f_ref[theta:theta + 1], f45_floored[theta:theta + 1], centers[b])
        hand_direct[i] = crisp.classify_crisp(
            flux_row, sigma[theta:theta + 1],
            valid=valid[theta:theta + 1], detected=valid[theta:theta + 1]).label[0]
    hand_ok = bool(np.array_equal(hand_stored, hand_direct))

    return E, subclasses, edges, hand_ok


def _write(path, E, subclasses, edges):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("E", data=E)
        f.create_dataset("SUBCLASSES", data=np.array(subclasses, dtype="S16"))
        f.create_dataset("LOG10_F45_EDGES", data=edges.astype(np.float64))
        f.create_dataset("LABELS", data=np.array(crisp.LABELS, dtype="S20"))
        f.attrs["GRANULE"] = "survey"
        f.attrs["LIBRARY"] = "gutcolors"


def build(config, regions=None):
    """Writes `fittp/emission/<key>_emission_survey.hdf5` for each of the
    six libraries (`config.product_path(config, "fittp", "emission", key,
    "survey")`, EMISSION.md "Module and runbook"). Survey-wide and
    region-independent: `regions` is accepted, per rule 5c, and ignored.
    """
    del regions
    for key in LIBRARIES:
        with progress.Stage("fittp.emission", key) as st:
            f_ref, floor_linear, subclass = _read_register(config, key)
            E, subclasses, edges, hand_ok = _emission_table(f_ref, floor_linear, subclass, st)

            row_sum_err = float(np.max(np.abs(E.sum(axis=2) - 1.0)))
            path = config_module.product_path(config, "fittp", "emission", key, "survey")
            _write(path, E, subclasses, edges)

            st.done(path, n_model=f_ref.shape[0], n_subclass=subclasses.size,
                     row_sum_err=row_sum_err, hand_check_20_ok=hand_ok)

            centers = 0.5 * (edges[:-1] + edges[1:])
            for target_mjy in (0.01, 0.1, 10.0):
                b = int(np.argmin(np.abs(centers - np.log10(target_mjy))))
                for k, sub in enumerate(subclasses):
                    row = E[k, b, :]
                    top2 = np.argsort(row)[::-1][:2]
                    print("fittp.emission %s subclass=%s bin~%.3g mJy: "
                          "top1=%s(%.3f) top2=%s(%.3f)"
                          % (key, sub, 10.0 ** centers[b],
                             crisp.LABELS[top2[0]], row[top2[0]],
                             crisp.LABELS[top2[1]], row[top2[1]]))

            for k, sub in enumerate(subclasses):
                below = np.flatnonzero(E[k, :, _UNCLASSIFIED] < 0.5)
                faintest = "%.3g mJy" % (10.0 ** centers[below[0]]) if below.size else "never"
                print("fittp.emission %s subclass=%s: UNCLASSIFIED < 0.5 from %s"
                      % (key, sub, faintest))


if __name__ == "__main__":
    run(build)
