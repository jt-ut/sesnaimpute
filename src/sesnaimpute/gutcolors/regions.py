"""
regions.py
====================================================================
A geometric view of the boundary table, for callers who want ONE cut
rather than the whole cascade.

The cascade (crisp.py, prob.py) answers "what is this source?".  This
module answers "is this colour inside Gutermuth's shock wedge, and
where do I draw it?" -- which is a different question, and the one you
have when you are studying a single class in isolation.

Vocabulary, all of it already in the tables:

    row     one published inequality
    term    a conjunction of rows.  Geometrically an INTERSECTION OF
            HALF-PLANES, hence convex -- here called a Wedge.
    gate    a disjunction of terms (PAH galaxies have two rule sets in
            two different colour planes)
    label   what the schedule assigns when a gate fires

A Wedge lives in a `panel` -- a named 2-D colour/magnitude plane, the
same ones the paper plots.  Rows whose features fit in that plane become
polygon edges; rows that do not are kept separately as `offplane`.  That
split is not a convenience, it is the truth about the cut: the PAH
galaxy wedge is a triangle in ([5.8]-[8.0], [4.5]-[5.8]) AND a
brightness condition [4.5] > 11.5 that no colour-colour diagram can
show.  Silently dropping it would make `contains` wrong; silently
including it would make `polygon` impossible.

PRECEDENCE IS NOT APPLIED HERE.  `contains` asks only whether this
region's own conditions hold.  A source can be inside the shock wedge
and still be labelled PAH_GALAXY by the cascade, because PAH is tested
first.  If you want the label, call classify_crisp.

Typical use
-----------
    from sesnaimpute.gutcolors import regions

    w = regions.get("SHOCK_BLOB").wedge()      # the one shock term
    w.panel.xlabel, w.panel.ylabel             # '[4.5]-[5.8]', '[3.6]-[4.5]'
    w.contains(x=B, y=A)                       # bool array
    w.polygon()                                # (M, 2) vertices, closed
    w.plot(ax)                                 # matplotlib, optional

    # straight from SEDs, no colour bookkeeping:
    x, y = regions.panel_coords(flux_mjy, w.panel)
    inside = regions.get("SHOCK_BLOB").contains_sed(flux_mjy, sigma_mjy)
====================================================================
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from sesnaimpute.gutcolors import crisp
from sesnaimpute.gutcolors import featurize as fz
from sesnaimpute.gutcolors import route as rt

# Complement of each sense, used when a term requires a row FALSE.
_NEGATE = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}


# ==========================================================================
# Panels
# ==========================================================================

@dataclass(frozen=True)
class Panel:
    """A named 2-D plane: which feature on each axis, and how to draw it."""

    id: str
    x: str                      # feature id, e.g. "B"
    y: str
    xlabel: str                 # human form, e.g. "[4.5]-[5.8]"
    ylabel: str
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    y_invert: bool = False      # magnitude axes run bright-to-faint

    @property
    def features(self) -> Tuple[str, str]:
        return (self.x, self.y)

    def box(self) -> Tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax), ordered regardless of axis inversion."""
        return (min(self.xlim), max(self.xlim), min(self.ylim), max(self.ylim))


def _panels(tab) -> Dict[str, Panel]:
    out = {}
    for pid, p in tab.panels.items():
        out[pid] = Panel(
            id=pid,
            x=p["x"], y=p["y"],
            xlabel=tab.features[p["x"]]["label"],
            ylabel=tab.features[p["y"]]["label"],
            xlim=tuple(p["xlim"]), ylim=tuple(p["ylim"]),
            y_invert=bool(p.get("y_invert", False)),
        )
    return out


def panels(tab=None) -> Dict[str, Panel]:
    """Every named plane in the table."""
    return _panels(tab or crisp.table())


def make_panel(x: str, y: str, xlim=(-1.0, 3.0), ylim=(-1.0, 3.0), tab=None) -> Panel:
    """An ad-hoc plane, for re-projecting a wedge into axes the paper
    never plotted.  `x` and `y` are feature ids from the table."""
    tab = tab or crisp.table()
    for f in (x, y):
        if f not in tab.features:
            raise KeyError(f"unknown feature {f!r}; known: {sorted(tab.features)}")
    return Panel(id=f"{x}_{y}", x=x, y=y,
                 xlabel=tab.features[x]["label"], ylabel=tab.features[y]["label"],
                 xlim=tuple(xlim), ylim=tuple(ylim))


# ==========================================================================
# Half-planes
# ==========================================================================

@dataclass(frozen=True)
class Line:
    """One in-plane inequality:  a*x + b*y  SENSE  c.

    `c` is the threshold with any sigma passed to `get()` already folded
    in -- a single scalar shift, which is what a drawn boundary needs.
    Per-SOURCE sigma is different: it moves the boundary by a different
    amount for every source, so it cannot live in `c`.  `ksigma` keeps
    the row's sigma coefficients for that case, and `margin`/`holds`
    accept a sigma mapping of arrays.
    """

    row_id: str
    a: float
    b: float
    c: float
    sense: str
    published: str
    negated: bool = False       # the term requires this row FALSE
    ksigma: Tuple[Tuple[str, float], ...] = ()   # (sigma_id, coefficient)
    sigma_baked: bool = False   # a scalar sigma was folded into `c`

    @property
    def strict(self) -> bool:
        return "=" not in self.sense

    def threshold(self, sigma=None):
        """`c` shifted by a per-source sigma mapping.

        The row reads  w.f + k.sigma  SENSE  t,  so the boundary sits at
        t - k.sigma.  Signs are the published ones: for PAP1/PAP2 they
        make the cut easier to trip as uncertainty grows.
        """
        if not sigma:
            return self.c
        if self.sigma_baked:
            raise ValueError(
                f"{self.row_id}: sigma was already folded in by get(sigma=...); "
                f"pass it in one place or the other, not both")
        c = self.c
        for name, k in self.ksigma:
            if name not in sigma:
                raise KeyError(
                    f"{self.row_id} needs sigma {name!r}; got {sorted(sigma)}. "
                    f"regions.colour_sigmas(flux, sigma) builds the full set.")
            c = c - k * np.asarray(sigma[name], dtype=float)
        return c

    def margin(self, x, y, sigma=None):
        """Signed slack.  POSITIVE means the constraint is satisfied,
        i.e. positive is INSIDE; zero is exactly on the boundary.

        The sign is normalised per row, so it does not depend on whether
        the published inequality was written `>` or `<`.  Units are
        those of the row's own left-hand side (magnitudes for every row
        in the table), which is not a perpendicular distance unless the
        normal happens to be a unit vector -- see crisp.Result.distance
        for the normalised form.
        """
        lhs = self.a * np.asarray(x, dtype=float) + self.b * np.asarray(y, dtype=float)
        d = 1.0 if self.sense.startswith(">") else -1.0
        return d * (lhs - self.threshold(sigma))

    def holds(self, x, y, sigma=None):
        g = self.margin(x, y, sigma)
        with np.errstate(invalid="ignore"):
            ok = g > 0.0 if self.strict else g >= 0.0
        return ok & np.isfinite(g)

    def endpoints(self, box):
        """Where the boundary crosses the box, as ((x0,y0),(x1,y1)), or
        None if it misses.  For drawing the edge itself."""
        xmin, xmax, ymin, ymax = box
        pts = []
        if self.b != 0.0:
            for xv in (xmin, xmax):
                yv = (self.c - self.a * xv) / self.b
                if ymin - 1e-9 <= yv <= ymax + 1e-9:
                    pts.append((xv, yv))
        if self.a != 0.0:
            for yv in (ymin, ymax):
                xv = (self.c - self.b * yv) / self.a
                if xmin - 1e-9 <= xv <= xmax + 1e-9:
                    pts.append((xv, yv))
        uniq = []
        for p in pts:
            if not any(abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9 for q in uniq):
                uniq.append(p)
        return (uniq[0], uniq[1]) if len(uniq) >= 2 else None

    def _le_form(self):
        """(a, b, c) rewritten as a*x + b*y <= c, for polygon clipping."""
        if self.sense.startswith(">"):
            return (-self.a, -self.b, -self.c)
        return (self.a, self.b, self.c)

    def __str__(self):
        return f"{self.a:+.4g}*x {self.b:+.4g}*y {self.sense} {self.c:.4g}"


# ==========================================================================
# Wedges
# ==========================================================================

@dataclass(frozen=True)
class Wedge:
    """One term, viewed in one panel.

    `lines` are the rows expressible in the panel; `offplane` are the rows
    that are not, with their incidence sign.  A point satisfying every
    line is inside the drawn region; a SOURCE satisfying the term must
    also satisfy the offplane rows (and any detection guards), which is
    what `contains_sed` does.
    """

    id: str                     # term id, e.g. "T_SHOCK"
    gate: str
    label: str                  # what the schedule assigns, "" if none
    panel: Panel
    lines: Tuple[Line, ...]
    offplane: Tuple[Tuple[str, int], ...]   # (row_id, +1 require / -1 forbid)
    _tab: object = None

    # -- membership -------------------------------------------------------

    @property
    def sigma_names(self) -> Tuple[str, ...]:
        """Propagated colour sigmas this wedge's in-plane rows read.
        Empty for a wedge whose boundaries do not move with uncertainty.
        `regions.colour_sigmas(flux, sigma)` supplies all of them."""
        seen = []
        for ln in self.lines:
            for name, _ in ln.ksigma:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)

    @property
    def sigma_of(self) -> Dict[str, str]:
        """{sigma_id: the colour it is the uncertainty OF}.

        Keys of a sigma mapping are FEATURE names, never axis names, and
        the two are easy to transpose: T_PAHAP is drawn with x = B and
        y = A, so `sA` is the sigma of the Y axis.  Worse, the Phase-2
        wedges read sigmas on the MEASURED colours (`sA`, `sY`) while
        being drawn in DEREDDENED ones (`X0`, `Y0`) -- there the sigmas
        do not correspond to the axes at all.  Use `sigma_from_axes`
        rather than pairing them by hand.
        """
        tab = self._tab or crisp.table()
        return {n: tab.sigmas[n]["of"] for n in self.sigma_names}

    def sigma_from_axes(self, x=None, y=None) -> Dict[str, object]:
        """Build a sigma mapping from per-AXIS uncertainties.

        Transposing x and y silently yields a plausible-looking wrong
        answer -- there is nothing in the numbers to catch it -- so this
        does the pairing from the panel definition instead, and refuses
        when the wedge's sigmas are not axis quantities.

            s = w.sigma_from_axes(x=sigma_x, y=sigma_y)
            w.contains(px, py, sigma=s)
        """
        of = self.sigma_of
        by_feature = {}
        if x is not None:
            by_feature[self.panel.x] = x
        if y is not None:
            by_feature[self.panel.y] = y

        out, unmapped = {}, {}
        for name, feature in of.items():
            if feature in by_feature:
                out[name] = by_feature[feature]
            else:
                unmapped[name] = feature
        if unmapped:
            raise ValueError(
                f"{self.id}: {sorted(unmapped)} are sigmas of "
                f"{sorted(set(unmapped.values()))}, which are not this panel's "
                f"axes ({self.panel.x}, {self.panel.y}). Supply them by name "
                f"instead -- regions.colour_sigmas(flux, sigma) builds the "
                f"full set from SEDs.")
        return out

    @property
    def guards(self) -> Tuple[Tuple[str, str], ...]:
        """(row_id, guard) for rows carrying a detection guard.  Empty
        means `contains_sed` applies no guard and is exactly the rows."""
        tab = self._tab or crisp.table()
        out = []
        for rid in [ln.row_id for ln in self.lines] + [r for r, _ in self.offplane]:
            if tab[rid].guard:
                out.append((rid, tab[rid].guard))
        return tuple(out)

    def contains(self, x, y, sigma=None):
        """Inside the drawn region?  In-plane rows only.

        `x`, `y` are values of the panel's two features (broadcastable).

        `sigma` is an optional mapping of propagated COLOUR sigmas --
        {'sA': array, ...} as returned by `regions.colour_sigmas` -- and
        shifts each boundary per source, exactly as the cascade does.
        Omit it for the nominal published cut.  Every name in
        `self.sigma_names` must be present; a missing one raises rather
        than silently defaulting to zero.

        If `offplane` is non-empty this is a NECESSARY but not sufficient
        condition for the term -- see `contains_sed`.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if not self.lines:
            raise ValueError(
                f"{self.id} has no rows expressible in panel {self.panel.id} "
                f"(features {self.panel.features}); pass panel= to re-project"
            )
        ok = np.ones(np.broadcast(x, y).shape, dtype=bool)
        for ln in self.lines:
            ok = ok & ln.holds(x, y, sigma)
        return ok

    def contains_sed(self, flux_mjy, sigma_mjy=None, valid=None,
                     detected=None, config=None):
        """Exact term evaluation from SEDs.

        Applies every row (in-plane and off), the propagated sigma terms,
        and any per-row detection guard.  It does NOT apply admission
        routing (P1/P2) or cascade precedence -- it answers only whether
        this term holds.

        `valid` is what the flux vector holds; `detected` is what the
        survey saw (defaults to `valid`, and differs only for an imputed
        SED).  Pass your own rather than letting them be derived from
        sigma if your selection differs.  `self.guards` is empty when no row here
        reads the mask at all, in which case this is exactly the rows.
        """
        return _term_truth(self._tab or crisp.table(), (self.id,),
                           flux_mjy, sigma_mjy, valid, detected, config)

    def margins(self, x, y, sigma=None):
        """(n_lines,) or (N, n_lines) signed slack per in-plane row,
        column order matching `self.lines`.

        POSITIVE IS INSIDE: a point satisfies the wedge iff every margin
        is positive, so `margins(...).min(axis=-1) > 0` reproduces
        `contains` for strict rows.  The smallest margin identifies the
        binding constraint -- the row that came closest to failing.
        """
        return np.stack([ln.margin(x, y, sigma) for ln in self.lines], axis=-1)

    # -- geometry ---------------------------------------------------------

    def polygon(self, xlim=None, ylim=None):
        """Closed vertex list, (M, 2), of the region clipped to the panel
        box.  Returns an empty (0, 2) array if the region does not meet
        the box.  Gutermuth's wedges are open at one or more edges; the
        box is what closes them, exactly as in the published figures."""
        if not self.lines:
            raise ValueError(
                f"{self.id} has no rows expressible in panel {self.panel.id}; "
                f"nothing to draw"
            )
        xmin, xmax, ymin, ymax = self.panel.box()
        if xlim is not None:
            xmin, xmax = min(xlim), max(xlim)
        if ylim is not None:
            ymin, ymax = min(ylim), max(ylim)

        poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        for ln in self.lines:
            poly = _clip(poly, *ln._le_form())
            if not poly:
                return np.zeros((0, 2))
        return np.asarray(poly, dtype=float)

    def edges(self, xlim=None, ylim=None):
        """Each in-plane boundary as a segment across the box, keyed by
        row id.  Use when you want the paper's lines rather than a filled
        patch."""
        xmin, xmax, ymin, ymax = self.panel.box()
        if xlim is not None:
            xmin, xmax = min(xlim), max(xlim)
        if ylim is not None:
            ymin, ymax = min(ylim), max(ylim)
        out = {}
        for ln in self.lines:
            seg = ln.endpoints((xmin, xmax, ymin, ymax))
            if seg is not None:
                out[ln.row_id] = np.asarray(seg, dtype=float)
        return out

    # -- presentation -----------------------------------------------------

    def plot(self, ax=None, xlim=None, ylim=None, fill=True, outline=True,
             label=None, set_limits=False, set_labels=True,
             fill_kw=None, line_kw=None, **kw):
        """Draw the region.  matplotlib is imported here, not at module
        import, so the package stays plot-free for batch use.

        The clip box comes from `xlim`/`ylim`, falling back to the axes'
        current limits if they have been set, and only then to the
        panel's.  A wedge is open at one or more edges, so SOMETHING has
        to close it -- but it should be the caller's data range, not a
        default in this table.  `set_limits=True` if you do want the
        panel's limits applied to the axes.

        `**kw` styles both artists; `fill_kw` and `line_kw` override it
        per artist, so face and edge can differ and neither collides
        with the other's defaults.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        if xlim is None and ax.dataLim.width > 0:
            xlim = ax.get_xlim()
        if ylim is None and ax.dataLim.height > 0:
            ylim = ax.get_ylim()

        poly = self.polygon(xlim=xlim, ylim=ylim)
        if len(poly):
            face = _defaults({**kw, **(fill_kw or {})}, alpha=0.18)
            line = _defaults({**kw, **(line_kw or {})}, lw=1.2)
            if fill:
                face.setdefault("label", label)
                ax.fill(poly[:, 0], poly[:, 1], **face)
                label = None                      # one legend entry, not two
            if outline:
                line.setdefault("label", label)
                closed = np.vstack([poly, poly[:1]])
                ax.plot(closed[:, 0], closed[:, 1], **line)

        if set_labels:
            ax.set_xlabel(self.panel.xlabel)
            ax.set_ylabel(self.panel.ylabel)
        if set_limits:
            ax.set_xlim(*self.panel.xlim)
            ax.set_ylim(*self.panel.ylim)
        return ax

    def describe(self) -> str:
        tab = self._tab or crisp.table()
        out = [f"{self.id}  (gate {self.gate}"
               + (f" -> {self.label}" if self.label else "") + ")",
               f"  panel {self.panel.id}: x = {self.panel.xlabel}, "
               f"y = {self.panel.ylabel}"]
        for ln in self.lines:
            neg = "  [NEGATED]" if ln.negated else ""
            out.append(f"  {ln.row_id:<10s} {tab[ln.row_id].published}{neg}")
            sig = ("  [shifts with " + ", ".join(
                f"{n} = sigma of {tab.sigmas[n]['of']}" for n, _ in ln.ksigma) + "]"
                if ln.ksigma else "")
            out.append(f"  {'':<10s}   -> {ln}{sig}")
        for rid, sign in self.offplane:
            neg = "  [NEGATED]" if sign < 0 else ""
            out.append(f"  {rid:<10s} {tab[rid].published}"
                       f"   (off-panel){neg}")
        return "\n".join(out)

    def __str__(self):
        return self.describe()


# matplotlib accepts several spellings of the same property, so a default
# of ours must stand down if the caller supplied ANY spelling of it --
# passing both raises rather than picking one.
_MPL_ALIASES = {
    "alpha": ("alpha",),
    "lw": ("lw", "linewidth"),
    "ls": ("ls", "linestyle"),
    "color": ("color", "c"),
}


def _defaults(kw, **defs):
    out = dict(kw)
    for key, val in defs.items():
        if not any(a in out for a in _MPL_ALIASES.get(key, (key,))):
            out[key] = val
    return out


def _clip(poly, a, b, c):
    """Sutherland-Hodgman clip of a convex polygon by a*x + b*y <= c."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp = a * p[0] + b * p[1] - c
        dq = a * q[0] + b * q[1] - c
        if dp <= 0.0:
            out.append(p)
        if (dp <= 0.0) != (dq <= 0.0):
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


# ==========================================================================
# Regions (a gate, or a label: the OR over its wedges)
# ==========================================================================

@dataclass(frozen=True)
class Region:
    """One or more wedges, unioned.  Non-convex in general, and its parts
    may live in different panels -- P1_PAH does."""

    id: str
    kind: str                   # "label" | "gate" | "term"
    wedges: Tuple[Wedge, ...]
    _tab: object = None

    def wedge(self, panel=None) -> Wedge:
        """The single wedge, when there is only one.  Raises otherwise,
        so a caller never silently gets half of a disjunction."""
        cands = self.wedges if panel is None else [
            w for w in self.wedges if w.panel.id == panel]
        if len(cands) != 1:
            raise ValueError(
                f"{self.id} has {len(cands)} wedges "
                f"({[w.id for w in cands]}); pick one from .wedges")
        return cands[0]

    @property
    def panel_ids(self) -> Tuple[str, ...]:
        seen = []
        for w in self.wedges:
            if w.panel.id not in seen:
                seen.append(w.panel.id)
        return tuple(seen)

    def contains(self, x, y, panel=None, sigma=None):
        """Union over the wedges sharing a panel.  `panel` is required
        when the region spans more than one.  `sigma` is a per-source
        mapping keyed by FEATURE name; see Wedge.contains."""
        if panel is None:
            if len(self.panel_ids) != 1:
                raise ValueError(
                    f"{self.id} spans panels {self.panel_ids}; pass panel=")
            panel = self.panel_ids[0]
        ws = [w for w in self.wedges if w.panel.id == panel]
        if not ws:
            raise ValueError(f"{self.id} has no wedge in panel {panel!r}")
        out = None
        for w in ws:
            hit = w.contains(x, y, sigma=sigma)
            out = hit if out is None else (out | hit)
        return out

    def contains_sed(self, flux_mjy, sigma_mjy=None, valid=None,
                     detected=None, config=None):
        """Exact: does any of this region's terms hold for these SEDs?
        All rows, sigma terms and detection guards; no precedence."""
        return _term_truth(self._tab or crisp.table(),
                           tuple(w.id for w in self.wedges),
                           flux_mjy, sigma_mjy, valid, detected, config)

    def plot(self, ax=None, panel=None, label=None, **kw):
        """Draw every wedge sharing a panel.  See Wedge.plot for the
        styling and clip-box arguments, which are passed straight
        through; `label` is applied once, not once per wedge."""
        if panel is None and len(self.panel_ids) != 1:
            raise ValueError(f"{self.id} spans panels {self.panel_ids}; pass panel=")
        panel = panel or self.panel_ids[0]
        for w in self.wedges:
            if w.panel.id == panel:
                ax = w.plot(ax=ax, label=label, **kw)
                label = None
        return ax

    def describe(self) -> str:
        head = f"=== {self.id}  ({self.kind}, {len(self.wedges)} term"
        head += "s ===" if len(self.wedges) != 1 else " ==="
        return "\n\n".join([head] + [w.describe() for w in self.wedges])

    def __str__(self):
        return self.describe()


# ==========================================================================
# Construction
# ==========================================================================

def _label_of_gate(tab) -> Dict[str, str]:
    out = {}
    for e in tab.schedule:
        if e.gate is not None:
            out.setdefault(e.gate, e.assign)
    return out


def _build_wedge(tab, term, panel, sigma, labels) -> Wedge:
    lines, offplane = [], []
    for rid, sign in term.rows.items():
        row = tab[rid]
        if set(row.coef) <= set(panel.features):
            c = row.t - sum(k * float(sigma.get(s, 0.0))
                            for s, k in row.ksigma.items())
            a = row.coef.get(panel.x, 0.0)
            b = row.coef.get(panel.y, 0.0)
            sense = row.sense if sign > 0 else _NEGATE[row.sense]
            lines.append(Line(row_id=rid, a=a, b=b, c=c, sense=sense,
                              published=row.published, negated=sign < 0,
                              ksigma=tuple(sorted(row.ksigma.items())),
                              sigma_baked=bool(row.ksigma) and bool(sigma)))
        else:
            offplane.append((rid, int(sign)))
    return Wedge(
        id=term.id,
        gate=term.gate,
        label=labels.get(term.gate, ""),
        panel=panel,
        lines=tuple(lines),
        offplane=tuple(offplane),
        _tab=tab,
    )


def get(name: str, panel=None, sigma: Optional[Dict[str, float]] = None,
        tab=None) -> Region:
    """Look up a label, gate or term id and return its Region.

    panel : Panel, panel id, or None to use each term's declared panel.
            Pass one to re-project every wedge into the same plane --
            rows that do not fit simply move to `offplane`.
    sigma : {sigma_id: value} to shift the sigma-bearing boundaries
            (PAP1/PAP2, the Class II rows, the Phase-2 rows).  Omitted
            means zero, i.e. the published nominal boundary.
    """
    tab = tab or crisp.table()
    sigma = sigma or {}
    P = _panels(tab)
    labels = _label_of_gate(tab)

    if isinstance(panel, str):
        if panel not in P:
            raise KeyError(f"unknown panel {panel!r}; known: {sorted(P)}")
        panel = P[panel]

    by_term = {t.id: t for t in tab.terms}
    if name in by_term:
        kind, terms = "term", [by_term[name]]
    elif name in set(tab.gates):
        kind, terms = "gate", [t for t in tab.terms if t.gate == name]
    elif name in crisp.LABEL_INDEX:
        gates = [g for g, lab in labels.items() if lab == name]
        kind, terms = "label", [t for t in tab.terms if t.gate in gates]
        if not terms:
            why = ("no Gutermuth rule assigns it -- it is a project "
                   "placeholder for GAL sub-types his scheme never labels"
                   if name == "GENERIC_GALAXY" else
                   "it is a cascade residual: what is left when nothing "
                   "fires, so there is no region to return")
            raise KeyError(f"{name!r} is a label but not a region -- {why}")
    else:
        raise KeyError(
            f"unknown region {name!r}.  Labels: {sorted(crisp.LABEL_INDEX)}.  "
            f"Gates: {sorted(tab.gates)}.  Terms: {sorted(by_term)}.")

    wedges = tuple(
        _build_wedge(tab, t, panel or P[t.panel], sigma, labels) for t in terms)
    return Region(id=name, kind=kind, wedges=wedges, _tab=tab)


def names(tab=None) -> Dict[str, Tuple[str, ...]]:
    """Everything `get` accepts, grouped."""
    tab = tab or crisp.table()
    labels = _label_of_gate(tab)
    return {
        "labels": tuple(sorted(set(labels.values()))),
        "gates": tuple(tab.gates),
        "terms": tuple(t.id for t in tab.terms),
        "panels": tuple(sorted(tab.panels)),
    }


# ==========================================================================
# SED -> panel coordinates
# ==========================================================================

def features_of(flux_mjy, sigma_mjy=None, valid=None, config=None):
    """Every feature the table uses, from an (N, 8) mJy flux array.

    Returns (features, colour_sigmas).  Band order J H Ks I1 I2 I3 I4 M1.
    A model SED is sigma = 0.  The J detection flag that selects the
    Phase-2 dereddening branch is taken from `valid` when given, exactly
    as classify_crisp does -- catalogues that fill non-detections with an
    upper limit yield a finite J magnitude and would otherwise take the
    wrong branch.
    """
    cfg = config or fz.SchemeConfig()
    flux = np.atleast_2d(np.asarray(flux_mjy, dtype=float))
    sigma = (np.zeros_like(flux) if sigma_mjy is None
             else np.atleast_2d(np.asarray(sigma_mjy, dtype=float)))
    if valid is not None:
        valid = np.atleast_2d(np.asarray(valid, dtype=bool))

    sigma_mag = fz.sigma_to_mag(flux, sigma)
    usable = rt.detection_mask(sigma_mag, valid)
    feat, sig, _ = fz.featurize(flux, sigma, has_J=usable[..., 0], config=cfg)
    return feat, sig


def colour_sigmas(flux_mjy, sigma_mjy):
    """Propagated colour sigmas, {'sA': (N,), ...}, from an (N, 8) SED.

    This is what `Wedge.contains(x, y, sigma=...)` wants -- the same
    quadrature sums the cascade uses, so a colour-space membership test
    with sigma needs no reconstruction from the line coefficients.

        x, y  = regions.panel_coords(flux, w.panel)
        s     = regions.colour_sigmas(flux, sigma)
        inside = w.contains(x, y, sigma=s)
    """
    _, sig = features_of(flux_mjy, sigma_mjy)
    return sig


def panel_coords(flux_mjy, panel, sigma_mjy=None, valid=None, config=None):
    """(x, y) for a panel, straight from SEDs."""
    if isinstance(panel, Wedge):
        panel = panel.panel
    if isinstance(panel, str):
        panel = _panels(crisp.table())[panel]
    feat, _ = features_of(flux_mjy, sigma_mjy, valid, config)
    return feat[panel.x], feat[panel.y]


def _term_truth(tab, term_ids, flux_mjy, sigma_mjy, valid, detected, config):
    """Exact truth of the OR over `term_ids`, via the cascade's own row
    and term evaluation.  Precedence is not applied."""
    cfg = config or fz.SchemeConfig()
    flux = np.atleast_2d(np.asarray(flux_mjy, dtype=float))
    sigma = (np.zeros_like(flux) if sigma_mjy is None
             else np.atleast_2d(np.asarray(sigma_mjy, dtype=float)))
    if valid is not None:
        valid = np.atleast_2d(np.asarray(valid, dtype=bool))

    sigma_mag = fz.sigma_to_mag(flux, sigma)
    usable = rt.detection_mask(sigma_mag, valid)
    feat, sig, _ = fz.featurize(flux, sigma, has_J=usable[..., 0], config=cfg)
    route_code = rt.route(usable)
    if detected is None:
        seen, route_seen = usable, route_code
    else:
        seen = np.atleast_2d(np.asarray(detected, dtype=bool))
        route_seen = rt.route(seen)
    guards = rt.guard_values(usable, route_code, seen, route_seen)

    row_true, _ = crisp.evaluate_rows(tab, feat, sig, guards)
    term_true = crisp.evaluate_terms(tab, row_true)
    cols = [i for i, t in enumerate(tab.terms) if t.id in set(term_ids)]
    return term_true[:, cols].any(axis=1)


def summary(tab=None) -> str:
    """One line per term: where it lives and how many edges it has."""
    tab = tab or crisp.table()
    P = _panels(tab)
    labels = _label_of_gate(tab)
    out = [f"{'term':<12s} {'gate':<12s} {'label':<16s} {'panel':<12s} "
           f"{'edges':>5s} {'off':>4s}"]
    for t in tab.terms:
        w = _build_wedge(tab, t, P[t.panel], {}, labels)
        out.append(f"{t.id:<12s} {t.gate:<12s} {w.label:<16s} "
                   f"{t.panel:<12s} {len(w.lines):>5d} {len(w.offplane):>4d}")
    return "\n".join(out)
