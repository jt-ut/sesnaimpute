"""
spec.py
====================================================================
Load, validate and compile the Gutermuth+2009 boundary table.

The table (tables/boundaries.toml) is authored in COLOUR space so it
diffs line-by-line against the appendix; this module compiles it to
8-MAGNITUDE space, where the norm ||w|| is unambiguous.  The redundant
colours (D = A+B, E = B+C) give different norms in the two spaces --
PAP1 is 1.720 in the (A,B) plane and 2.955 in magnitude space -- so the
compiled form is the one geometry should ever be done in.

Every row reduces to

        ( w . f  +  k . sigma )   SENSE   t

with SENSE one of  >  <  >=  <= .  Strictness is per-row and load
bearing (SHK2, PAP1, PAP2, C2_3 are the non-strict ones).
====================================================================
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from sesnaimpute import definitions
from sesnaimpute.constants import GUTERMUTH_LABELS

try:
    import tomllib as _toml
except ImportError:  # Python < 3.11
    import tomli as _toml


TABLE_DIR = Path(__file__).parent / "tables"

# --- sigma binding (erratum E3) ------------------------------------------
# The Class II constraints carry a bare, unsubscripted `sigma` in the arXiv
# source.  Two readings are defensible and the choice is empirical, so it is
# a flag rather than a decision baked into the table.
#
#   per_colour  sigma of the colour appearing in that term.  The arXiv
#               reading, and the default -- it beat every alternative
#               against SESNA (0.9901 vs 0.9896 / 0.9893 / 0.9884 / 0.9879).
#   published   the ApJS version substitutes fixed symbols bound to specific
#               colours, in one place applying sigma{[4.5]-[5.8]} to a
#               constraint on [4.5]-[8.0].  Almost certainly a copy-editing
#               casualty, but reproducible on request.
#   all_sA / all_sE   one colour's sigma used throughout; diagnostics.
#   none        sigma terms dropped entirely.
SIGMA_BINDINGS = {
    "per_colour": {},          # as authored in boundaries.toml
    "published": {
        "C2_1": {"sB": -1.0},
        "C2_3": {"sD": 1.0, "sB": 3.5},
    },
    "all_sA": {
        "C2_1": {"sA": -1.0}, "C2_2": {"sA": -1.0},
        "C2_3": {"sA": 4.5},  "C2_4": {"sA": -1.0},
    },
    "all_sE": {
        "C2_1": {"sE": -1.0}, "C2_2": {"sE": -1.0},
        "C2_3": {"sE": 4.5},  "C2_4": {"sE": -1.0},
    },
    "none": {"C2_1": {}, "C2_2": {}, "C2_3": {}, "C2_4": {}},
}

# The catalogue's own eight-band order (sesnaimpute.definitions.BANDS),
# not a copy: this scheme reads whichever bands its host project measures
# in that same order.
BAND_ORDER = tuple(b.key for b in definitions.BANDS)

# Features that are piecewise-linear in the band magnitudes (dereddening
# branch chosen per source), so they carry no fixed `mags` expansion.
DERIVED_FEATURES = ("X0", "Y0", "m36_0")

_SAFE_EXPR = re.compile(r"^[0-9+\-*/(). eE]+$")


def _evaluate(expr) -> float:
    """Evaluate a stored expression string.

    Coefficients and thresholds are kept as expressions rather than
    decimals on purpose: 1.05/1.2 and 1.4*(-0.7)+0.15 do not land on
    exact binary, and pre-rounding them introduces a permanent silent
    offset in every comparison against the published scheme.
    """
    if isinstance(expr, (int, float)):
        return float(expr)
    if not _SAFE_EXPR.match(expr):
        raise ValueError(f"unsafe expression in table: {expr!r}")
    return float(eval(expr, {"__builtins__": {}}, {}))


@dataclass(frozen=True)
class Row:
    """One published constraint, normalised."""

    id: str
    gate: str
    published: str
    coef: Dict[str, float]            # over features
    ksigma: Dict[str, float]          # over propagated colour sigmas
    sense: str                        # ">" "<" ">=" "<="
    t: float
    term: Optional[str] = None
    panel: Optional[str] = None
    guard: Optional[str] = None

    @property
    def strict(self) -> bool:
        return "=" not in self.sense

    @property
    def direction(self) -> int:
        """+1 if the predicate is 'greater', -1 if 'less'."""
        return 1 if self.sense.startswith(">") else -1

    @property
    def is_derived(self) -> bool:
        """True if the row touches a dereddened feature (Phase 2)."""
        return any(f in DERIVED_FEATURES for f in self.coef)

    def margin(self, feat: Dict[str, float], sigma: Dict[str, float]) -> float:
        """Signed margin.  Positive means the constraint is satisfied
        (strictly); zero means exactly on the boundary."""
        lhs = sum(c * feat[f] for f, c in self.coef.items())
        lhs += sum(k * sigma[s] for s, k in self.ksigma.items())
        return self.direction * (lhs - self.t)

    def holds(self, feat: Dict[str, float], sigma: Dict[str, float]) -> bool:
        g = self.margin(feat, sigma)
        return g > 0.0 if self.strict else g >= 0.0


@dataclass(frozen=True)
class Term:
    """A conjunction over rows: +1 require true, -1 require false."""

    id: str
    gate: str
    rows: Dict[str, int]
    panel: Optional[str] = None


@dataclass(frozen=True)
class ScheduleEntry:
    """One ordered action.  Phases 1-2 are a cascade; Phase 3 entries are
    state transitions on the label already assigned."""

    phase: int
    assign: str
    when: object                     # "unassigned" | "any" | [labels]
    gate: Optional[str] = None       # None = unconditional (Phase-1 residual)
    on: str = "fire"                 # "fire" | "not_fire"
    route: Optional[Tuple[str, ...]] = None
    guard: Optional[str] = None


@dataclass
class Table:
    rows: Tuple[Row, ...]
    features: Dict[str, dict]
    sigmas: Dict[str, dict]
    panels: Dict[str, dict]
    terms: Tuple[Term, ...] = ()
    schedule: Tuple[ScheduleEntry, ...] = ()

    # magnitude-space compilation, one entry per non-derived row
    w_mag: Dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, row_id: str) -> Row:
        return self._by_id[row_id]

    def __post_init__(self):
        self._by_id = {r.id: r for r in self.rows}

    @property
    def row_ids(self) -> Tuple[str, ...]:
        return tuple(r.id for r in self.rows)

    @property
    def gates(self) -> Tuple[str, ...]:
        """Gate ids, in first-appearance order over the term table."""
        seen = []
        for t in self.terms:
            if t.gate not in seen:
                seen.append(t.gate)
        return tuple(seen)

    def incidence(self) -> np.ndarray:
        """Dense ternary term x row matrix, (n_terms, 45)."""
        idx = {rid: i for i, rid in enumerate(self.row_ids)}
        M = np.zeros((len(self.terms), len(self.rows)), dtype=np.int8)
        for ti, term in enumerate(self.terms):
            for rid, v in term.rows.items():
                M[ti, idx[rid]] = v
        return M

    def norm(self, row_id: str) -> float:
        """||w|| in 8-magnitude space.  Undefined for Phase-2 rows,
        whose normal depends on the dereddening branch."""
        return float(np.linalg.norm(self.w_mag[row_id]))


def _feature_to_mags(features: Dict[str, dict], name: str) -> np.ndarray:
    """Expand one feature into the 8-band magnitude basis."""
    spec = features[name]
    if spec.get("kind") == "derived":
        raise KeyError(f"{name} is branch-dependent; no fixed expansion")
    w = np.zeros(len(BAND_ORDER))
    for band, c in spec["mags"].items():
        w[BAND_ORDER.index(band)] += float(c)
    return w


def load(path: Optional[Path] = None, sigma_binding: str = "per_colour") -> Table:
    """Load, validate and compile the boundary table.

    `sigma_binding` selects the reading of erratum E3; see
    SIGMA_BINDINGS.  It rebinds only the Class II rows and leaves
    everything else untouched.
    """
    if sigma_binding not in SIGMA_BINDINGS:
        raise ValueError(f"unknown sigma_binding {sigma_binding!r}")
    override = SIGMA_BINDINGS[sigma_binding]
    path = Path(path) if path else TABLE_DIR / "boundaries.toml"
    with open(path, "rb") as fh:
        raw = _toml.load(fh)

    rows = []
    for entry in raw["row"]:
        rows.append(
            Row(
                id=entry["id"],
                gate=entry["gate"],
                published=entry["published"],
                coef={k: _evaluate(v) for k, v in entry["coef"].items()},
                ksigma=(
                    override[entry["id"]] if entry["id"] in override
                    else {k: _evaluate(v) for k, v in entry.get("ksigma", {}).items()}
                ),
                sense=entry["sense"],
                t=_evaluate(entry["t"]),
                term=entry.get("term"),
                panel=entry.get("panel"),
                guard=entry.get("guard"),
            )
        )

    with open(path.parent / "terms.toml", "rb") as fh:
        raw_terms = _toml.load(fh)
    terms = tuple(
        Term(id=e["id"], gate=e["gate"], rows=dict(e["rows"]), panel=e.get("panel"))
        for e in raw_terms["term"]
    )

    with open(path.parent / "schedule.toml", "rb") as fh:
        raw_sched = _toml.load(fh)
    schedule = tuple(
        ScheduleEntry(
            phase=e["phase"],
            assign=e["assign"],
            when=e["when"],
            gate=e.get("gate"),
            on=e.get("on", "fire"),
            route=tuple(e["route"]) if "route" in e else None,
            guard=e.get("guard"),
        )
        for e in raw_sched["entry"]
    )

    table = Table(
        rows=tuple(rows),
        features=raw["features"],
        sigmas=raw["sigmas"],
        panels=raw["panels"],
        terms=terms,
        schedule=schedule,
    )

    # --- compile colour space -> magnitude space --------------------
    for row in table.rows:
        if row.is_derived:
            continue
        w = np.zeros(len(BAND_ORDER))
        for fname, c in row.coef.items():
            w += c * _feature_to_mags(table.features, fname)
        table.w_mag[row.id] = w

    _validate(table)
    return table


def _validate(table: Table) -> None:
    ids = [r.id for r in table.rows]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate row ids: {sorted(dupes)}")

    for row in table.rows:
        if row.sense not in (">", "<", ">=", "<="):
            raise ValueError(f"{row.id}: bad sense {row.sense!r}")
        for f in row.coef:
            if f not in table.features:
                raise ValueError(f"{row.id}: unknown feature {f!r}")
        for s in row.ksigma:
            if s not in table.sigmas:
                raise ValueError(f"{row.id}: unknown sigma {s!r}")
        if row.panel and row.panel not in table.panels:
            raise ValueError(f"{row.id}: unknown panel {row.panel!r}")
        if not row.coef:
            raise ValueError(f"{row.id}: empty coefficient vector")

    # P2_YSO3 and P2_PROTO are parallel by construction (scheme doc 4.4).
    a, b = table["P2_YSO3"], table["P2_PROTO"]
    if a.coef != b.coef or a.ksigma != b.ksigma:
        raise ValueError("P2_YSO3 / P2_PROTO no longer share a normal vector")

    known_rows = set(table.row_ids)
    term_ids = [t.id for t in table.terms]
    if len(term_ids) != len(set(term_ids)):
        raise ValueError("duplicate term ids")
    for term in table.terms:
        unknown = set(term.rows) - known_rows
        if unknown:
            raise ValueError(f"{term.id}: unknown rows {sorted(unknown)}")
        if not term.rows:
            raise ValueError(f"{term.id}: empty term")
        for rid, v in term.rows.items():
            if v not in (1, -1):
                raise ValueError(f"{term.id}: incidence for {rid} must be +/-1")

    # Every row must participate somewhere, or it is dead weight that will
    # silently stop being tested.
    used = set().union(*(set(t.rows) for t in table.terms)) if table.terms else set()
    orphans = known_rows - used
    if orphans:
        raise ValueError(f"rows in no term: {sorted(orphans)}")

    gates = set(table.gates)
    vocab = set(GUTERMUTH_LABELS["id"])
    for e in table.schedule:
        if e.gate is not None and e.gate not in gates:
            raise ValueError(f"schedule references unknown gate {e.gate!r}")
        if e.on not in ("fire", "not_fire"):
            raise ValueError(f"bad `on` value {e.on!r}")
        if e.guard is not None and e.gate is None:
            raise ValueError("guard on an unconditional entry")
        # Label names are owned by constants.GUTERMUTH_LABELS, not by this
        # table.  Catching a typo here beats discovering it as a KeyError
        # part-way through a survey run.
        named = {e.assign} | (set(e.when) if isinstance(e.when, list) else set())
        unknown = named - vocab
        if unknown:
            raise ValueError(
                f"schedule names labels absent from GUTERMUTH_LABELS: "
                f"{sorted(unknown)}")
