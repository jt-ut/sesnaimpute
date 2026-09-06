"""
crisp.py
====================================================================
The hard cascade: rows -> terms -> gates -> schedule -> one label.

Fully vectorised.  Row evaluation is a single (N, 45) boolean array;
term evaluation is a matrix product against the ternary incidence, so
it stays O(N x 45 x n_terms) in flops and O(N x 45) in memory rather
than materialising an (N, n_terms, 45) intermediate.
====================================================================
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from sesnaimpute.constants import GUTERMUTH_LABELS

from sesnaimpute.gutcolors import featurize as fz
from sesnaimpute.gutcolors import route as rt
from sesnaimpute.gutcolors import spec

# The label vocabulary is NOT declared here.  GUTERMUTH_LABELS is the one
# authority -- it already carries the SESNA CLASS codes, the plot labels
# and the colours, and a second list in this module could only drift from
# it.  We take the `id` column, which is the machine identifier; the row
# order supplies the dense index, since CLASS codes are sparse and
# negative and cannot serve as one.
#
# LABELS therefore includes GENERIC_GALAXY (code 49), which this cascade
# can never assign -- Gutermuth's scheme has no rule for it.  That is
# deliberate: label histograms and confusion matrices then line up with
# GUTERMUTH_LABELS row-for-row, so colours and legends need no reindexing,
# and the always-zero column is itself checked by the tests.
LABELS = tuple(GUTERMUTH_LABELS["id"])
LABEL_INDEX = {name: i for i, name in enumerate(LABELS)}
CLASS_CODE = np.asarray(GUTERMUTH_LABELS.index, dtype=np.int64)   # index -> CLASS
UNASSIGNED = -1


def labels_from_class(class_codes):
    """SESNA CLASS column -> label index.  Raises on an unknown code
    rather than silently mapping it to UNCLASSIFIED."""
    codes = np.asarray(class_codes, dtype=np.int64)
    lut = {int(c): i for i, c in enumerate(CLASS_CODE)}
    unknown = set(np.unique(codes).tolist()) - set(lut)
    if unknown:
        raise ValueError(f"CLASS codes absent from GUTERMUTH_LABELS: {sorted(unknown)}")
    return np.array([lut[int(c)] for c in codes.ravel()]).reshape(codes.shape)

def _valid_arg(valid, origin):
    """`origin` was the former name of `valid` -- it named a SESNA column
    (ORIGIN_FNU) rather than anything in the scheme, and it read as the
    provenance of the value rather than as permission to use it."""
    if origin is None:
        return valid
    if valid is not None:
        raise TypeError("pass `valid` or `origin`, not both (origin is the old name)")
    import warnings
    warnings.warn("`origin=` is now `valid=`", DeprecationWarning, stacklevel=3)
    return origin


_ROUTE_CODE = {"P1": rt.ROUTE_P1, "P2": rt.ROUTE_P2}

_TABLES = {}


def table(sigma_binding="per_colour"):
    """Cached table, one per sigma-binding variant (erratum E3)."""
    if sigma_binding not in _TABLES:
        _TABLES[sigma_binding] = spec.load(sigma_binding=sigma_binding)
    return _TABLES[sigma_binding]


@dataclass
class Trace:
    """Why each source got the label it got.

    `entry` is the schedule index that last wrote the label -- the gate
    that fired, or the residual.  `binding_row` is the constraint within
    that gate's winning term that came closest to failing, i.e. the one
    actually deciding the outcome; -1 where there is no winning term
    (the residual, or UNCLASSIFIED).  `phase3` records which Phase-3
    transitions applied, since a source can be reinstated and then
    demoted in the same pass.
    """

    entry: np.ndarray          # (N,) index into tab.schedule, -1 if none
    binding_row: np.ndarray    # (N,) index into tab.row_ids, -1 if none
    phase3: np.ndarray         # (N, n_phase3) bool


@dataclass
class Result:
    label: np.ndarray          # (N,) int index into LABELS
    route: np.ndarray          # (N,) int, see route.ROUTE_NAMES
    row_true: np.ndarray       # (N, 45) bool
    # (N, 45) signed perpendicular distance to each boundary, magnitudes.
    # POSITIVE IS INSIDE -- positive means the row is satisfied, matching
    # row_true, and the sign does not depend on whether the published
    # inequality was written `>` or `<`.  NaN for the Phase-2 rows, whose
    # normal depends on the dereddening branch.
    #
    # It is pure geometry: row_true additionally applies each row's
    # detection guard, so on the guarded Phase-3 rows a source can have
    # a positive distance and row_true False.  Never the reverse.
    distance: np.ndarray
    gate_fired: np.ndarray     # (N, n_gates) bool
    trace: Trace = None

    def label_names(self):
        return np.array(LABELS, dtype=object)[self.label]

    def route_names(self):
        return np.array([rt.ROUTE_NAMES[int(c)] for c in np.ravel(self.route)])


def evaluate_rows(tab, feat, sig, guards):
    """(N, 45) truth and signed distance.

    Distance is the margin divided by ||w|| in 8-magnitude space, so it
    is a perpendicular distance in magnitudes.  Phase-2 rows depend on
    the dereddening branch and therefore have no fixed magnitude-space
    normal; their distance is NaN, and supplying it is Phase-2 work.
    """
    n = np.broadcast(*[np.asarray(v) for v in feat.values()]).shape
    n = n[0] if n else 1
    n_rows = len(tab.rows)

    truth = np.empty((n, n_rows), dtype=bool)
    dist = np.full((n, n_rows), np.nan)

    for j, row in enumerate(tab.rows):
        lhs = np.zeros(n)
        for f, c in row.coef.items():
            lhs = lhs + c * np.asarray(feat[f])
        for s, k in row.ksigma.items():
            lhs = lhs + k * np.asarray(sig[s])
        g = row.direction * (lhs - row.t)

        with np.errstate(invalid="ignore"):
            ok = g > 0.0 if row.strict else g >= 0.0
        ok &= np.isfinite(g)          # a NaN feature can never satisfy a cut

        if row.guard is not None:
            for gname in row.guard.split(","):
                ok &= guards[gname.strip()]

        truth[:, j] = ok
        if row.id in tab.w_mag:
            dist[:, j] = g / np.linalg.norm(tab.w_mag[row.id])

    return truth, dist


def evaluate_terms(tab, row_true):
    """(N, n_terms) bool, via violation counting against the incidence."""
    M = tab.incidence()
    pos = (M > 0).astype(np.int8)
    neg = (M < 0).astype(np.int8)
    rt_ = row_true.astype(np.int8)
    violations = (1 - rt_) @ pos.T + rt_ @ neg.T
    return violations == 0


def evaluate_gates(tab, term_true):
    """OR over the terms carrying each gate."""
    out = {}
    for gi, gate in enumerate(tab.gates):
        cols = [i for i, t in enumerate(tab.terms) if t.gate == gate]
        out[gate] = term_true[:, cols].any(axis=1)
    return out


def run_schedule(tab, gates, route_code, guards):
    """Ordered pass: Phase 1-2 cascade, then Phase 3 transitions."""
    n = len(route_code)
    label = np.full(n, UNASSIGNED, dtype=np.int8)

    p12 = rt.phase12(route_code)
    has24 = rt.has_24(route_code)

    entry_of = np.full(n, -1, dtype=np.int16)
    p3_cols = {ei: i for i, ei in enumerate(
        [i for i, e in enumerate(tab.schedule) if e.phase == 3])}
    p3 = np.zeros((n, len(p3_cols)), dtype=bool)

    for ei, e in enumerate(tab.schedule):
        # -- does this entry apply to this source at all?
        if e.phase in (1, 2):
            applies = np.zeros(n, dtype=bool)
            for r in (e.route or ()):
                applies |= p12 == _ROUTE_CODE[r]
        else:
            applies = has24.copy()

        if e.guard is not None:
            applies &= guards[e.guard]

        # -- precondition on current state
        if e.when == "unassigned":
            applies &= label == UNASSIGNED
        elif e.when == "any":
            pass
        else:
            holds = np.zeros(n, dtype=bool)
            for name in e.when:
                holds |= label == LABEL_INDEX[name]
            applies &= holds

        # -- does the gate fire?
        if e.gate is not None:
            fired = gates[e.gate]
            if e.on == "not_fire":
                fired = ~fired
            applies &= fired

        label = np.where(applies, LABEL_INDEX[e.assign], label)
        entry_of = np.where(applies, ei, entry_of)
        if e.phase == 3:
            p3[:, p3_cols[ei]] = applies

    return (np.where(label == UNASSIGNED, LABEL_INDEX["UNCLASSIFIED"], label),
            entry_of, p3)


def classify_crisp(flux_mjy, sigma_mjy, valid=None, detected=None,
                   config=None, tab=None, origin=None):
    """Gutermuth+2009 hard label for an (N, 8) SED array.

    flux_mjy, sigma_mjy : (N, 8) in mJy, band order J H Ks I1 I2 I3 I4 M1.

    Two per-band boolean masks, both (N, 8) and both optional:

    valid     may the cascade USE this value?  True for anything the flux
              vector really holds -- observed or imputed.  None derives it
              from sigma against the scheme's own thresholds.
    detected  did the SURVEY SEE this band?  Defaults to `valid`.

    They differ only for an IMPUTED SED, where a band the survey never
    detected now carries a model's prediction: `valid` says yes (use the
    number), `detected` says no (the sky never gave it).  Keeping them
    apart is not pedantry -- the deeply-embedded class is DEFINED by the
    dropout, so a fully imputed SED can never reach it if the two are
    conflated.  See docs/gut09_imputed.md.

    `origin` is the former name of `valid`, accepted for compatibility.

    A model SED is simply a row with sigma = 0; nothing here special-cases
    it.  Note that sigma = 0 passes every admission threshold, so noiseless
    SEDs always route to Phase 1 (plus Phase 3 if 24 um is present) unless
    an explicit `valid` mask says otherwise.
    """
    valid = _valid_arg(valid, origin)
    cfg = config or fz.SchemeConfig()
    tab = tab or table(cfg.sigma_binding)

    flux = np.atleast_2d(np.asarray(flux_mjy, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma_mjy, dtype=float))
    if valid is not None:
        valid = np.atleast_2d(np.asarray(valid, dtype=bool))

    # The usable mask must be settled BEFORE dereddening: whether J is
    # a real value selects the Phase-2 locus branch, and a catalogue
    # that fills non-detections with an upper-limit flux would otherwise
    # send those sources down the J-present branch on a fabricated J.
    sigma_mag = fz.sigma_to_mag(flux, sigma)
    usable = rt.detection_mask(sigma_mag, valid)
    has_J = usable[..., 0]                        # BAND_ORDER[0] == "J"

    feat, sig, aux = fz.featurize(flux, sigma, has_J=has_J, config=cfg)
    route_code = rt.route(usable)

    if detected is None:
        seen, route_seen = usable, route_code
    else:
        # sigma is deliberately NOT applied here: this mask records what
        # the survey saw, and an imputed band has no survey sigma to
        # threshold.  Callers whose detection criterion includes a sigma
        # cut should fold it in before passing the mask.
        seen = np.atleast_2d(np.asarray(detected, dtype=bool))
        route_seen = rt.route(seen)

    guards = rt.guard_values(usable, route_code, seen, route_seen)

    row_true, dist = evaluate_rows(tab, feat, sig, guards)
    term_true = evaluate_terms(tab, row_true)
    gates = evaluate_gates(tab, term_true)
    label, entry_of, p3 = run_schedule(tab, gates, route_code, guards)

    return Result(
        label=label,
        route=route_code,
        row_true=row_true,
        distance=dist,
        gate_fired=np.column_stack([gates[g] for g in tab.gates]),
        trace=Trace(entry=entry_of,
                    binding_row=binding_rows(tab, entry_of, term_true, dist),
                    phase3=p3),
    )


def binding_rows(tab, entry_of, term_true, dist):
    """For each source, the row inside the winning term that came closest
    to failing.  That is the constraint actually deciding the label, and
    it is what makes a disagreement diagnosable rather than mysterious."""
    idx = {rid: i for i, rid in enumerate(tab.row_ids)}
    out = np.full(len(entry_of), -1, dtype=np.int16)

    for ei, e in enumerate(tab.schedule):
        if e.gate is None:
            continue
        here = entry_of == ei
        if not here.any():
            continue
        cols = [i for i, tm in enumerate(tab.terms) if tm.gate == e.gate]
        for ti in cols:
            rows = [idx[r] for r in tab.terms[ti].rows]
            sel = here & term_true[:, ti] & (out < 0)
            if not sel.any():
                continue
            d = np.where(np.isnan(dist[np.ix_(sel, rows)]), np.inf,
                         dist[np.ix_(sel, rows)])
            out[sel] = np.array(rows)[d.argmin(axis=1)]
    return out
