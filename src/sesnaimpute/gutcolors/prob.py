"""
prob.py
====================================================================
The probabilistic cascade.

Each of Gutermuth's boundaries is evaluated as the probability that the
source's photometry satisfies it, given the propagated per-band flux
errors; these are composed through his gating order as a chain rule
over the sequence.

No free parameters -- the smoothing scale is the photometry.  Each
boundary is softened by ITS OWN propagated uncertainty, which differs
per row because the rows stack bands differently: with 0.05 mag errors
everywhere, C1_2 (m36 - m45) has nu = 0.071 while PAP1
(m36 - 2.4 m45 + 1.4 m58) has nu = 0.148.

Rows fall into three cases, decided structurally at load:

  flux1 (8 rows)   a magnitude threshold IS a flux threshold.  Exact.
  flux2 (24 rows)  a colour threshold IS a flux-ratio threshold, and a
                   ratio comparison is linear in flux.  Exact.
  mag  (13 rows)   non-integer weights, or dereddened features.  Phi in
                   magnitude space, uncorrected in this version (see
                   docs/gut09_probabilistic_cascade.md sections 4.3, 10).

--- SHAPE OF THE IMPLEMENTATION (ruling R-56) -----------------------

The cascade is called once per (source, model) pair inside an
allocation-limited fitting loop, so it is written as a THREE-STAGE
pipeline rather than as one per-call function:

  compiled(tab)               once per boundary table.  Flux-space
                              thresholds as scalars, the four
                              dereddening branches as a (4, 6, 8)
                              constant, the incidence and gate->column
                              maps, and a live-row mask per route code.

  prepare_source(...) -> ctx  once per SOURCE.  sigma_fill is a per-band
                              FRACTIONAL error (R-52) and the source's
                              own photometry is spliced in unchanged, so
                              r = sigma/flux -- and therefore the
                              detection mask, the route, every guard,
                              the propagated colour sigmas, the whole
                              flux->magnitude bias/skew correction, the
                              effective thresholds, the Cornish-Fisher
                              (nu, gamma) pairs and the Phase-2 nu^2
                              table -- is a CONSTANT down the model
                              batch.  All of it is hoisted here.

  classify_prob_batched(ctx, flux, sigma)   once per model batch.  A
                              handful of small matmuls, one ndtr per
                              family over a row-contiguous (45, N)
                              buffer, then compose and schedule.

`classify_prob(flux, sigma, ...)` is a thin wrapper over the two: it
builds a context and calls the batched core.  When the batch's
per-band fractional errors and masks are constant down the rows -- the
pair-completed case -- it collapses the context to a single row, so the
wrapper pays the per-source cost once rather than N times.

See docs/gut09_probabilistic_cascade.md.
====================================================================
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorised

from sesnaimpute.constants import VEGA_ZERO_POINT_MJY

from sesnaimpute.gutcolors import corrections as co
from sesnaimpute.gutcolors import crisp
from sesnaimpute.gutcolors import featurize as fz
from sesnaimpute.gutcolors import route as rt
from sesnaimpute.gutcolors.spec import BAND_ORDER

ZP = np.array([VEGA_ZERO_POINT_MJY[b] for b in BAND_ORDER])
_K_MAG = 2.5 / np.log(10.0)      # mag per unit fractional flux error

# --- Phase-2 dereddening normals -----------------------------------------
# E(H-K) is affine in the band magnitudes WITHIN a branch, so each of the
# four (has_J, flat) combinations has a fixed 8-vector.  Derived from the
# closed forms in featurize.deredden; verified by the reddening identity
# (redden with A_H - A_K = 1 and dE_HK must come out to 1.000).
_J = BAND_ORDER.index("J")
_H = BAND_ORDER.index("H")
_K = BAND_ORDER.index("Ks")
_1 = BAND_ORDER.index("I1")
_2 = BAND_ORDER.index("I2")


def _ehk_normals():
    """w such that E_HK = w . m + const, one per dereddening branch."""
    out = {}

    # J present, sloped:  E_HK = (JH - 0.58 HK - 0.52) / 1.15
    w = np.zeros(8)
    d = fz.E_JH_OVER_E_HK - fz.CTTS_SLOPE          # 1.15
    w[_J] = 1.0 / d
    w[_H] = -(1.0 + fz.CTTS_SLOPE) / d
    w[_K] = fz.CTTS_SLOPE / d
    out[(True, 0)] = w

    # J present, flat:  E_HK = (JH - 0.6) / 1.73
    w = np.zeros(8)
    w[_J] = 1.0 / fz.E_JH_OVER_E_HK
    w[_H] = -1.0 / fz.E_JH_OVER_E_HK
    out[(True, 1)] = w

    # J absent, sloped:  E_HK = (HK - 1.33 A - 0.133) / 0.75581
    w = np.zeros(8)
    d = 1.0 - fz.GT05_SLOPE * fz.C_3645_CORRECTED   # 0.75581
    w[_H] = 1.0 / d
    w[_K] = -1.0 / d
    w[_1] = -fz.GT05_SLOPE / d
    w[_2] = fz.GT05_SLOPE / d
    out[(False, 0)] = w

    # J absent, flat:  E_HK = HK - 0.2
    w = np.zeros(8)
    w[_H] = 1.0
    w[_K] = -1.0
    out[(False, 1)] = w
    return out


EHK_NORMAL = _ehk_normals()

# Branch index used by the compiled Phase-2 table: 2 * has_J + flat.  It
# is the sort order of EHK_NORMAL's keys, and is what featurize.deredden
# reports as (has_J, branch).
BRANCH_KEYS = ((False, 0), (False, 1), (True, 0), (True, 1))

# E_HK = w . m + b, one additive constant per branch (erratum E4 fix).
# Each is read straight off the closed form quoted in _ehk_normals above
# by isolating the term with no magnitude factor.
EHK_CONST = {
    (True, 0): -fz.CTTS_INTERCEPT / (fz.E_JH_OVER_E_HK - fz.CTTS_SLOPE),
    (True, 1): -fz.JH0_FLOOR / fz.E_JH_OVER_E_HK,
    (False, 0): -fz.GT05_INTERCEPT / (1.0 - fz.GT05_SLOPE * fz.C_3645_CORRECTED),
    (False, 1): -fz.HK0_FLOOR,
}

# Same weights/constants as arrays, indexed by branch code (2*has_J + flat),
# for the vectorised E_HK mean used by the clamp-mixture fix (E4).
EHK_NORMAL_MAT = np.stack([EHK_NORMAL[k] for k in BRANCH_KEYS])    # (4, 8)
EHK_CONST_ARR = np.array([EHK_CONST[k] for k in BRANCH_KEYS])      # (4,)
EHK_WSUM = EHK_NORMAL_MAT.sum(axis=1)                               # (4,)


def _derived_normals(w_ehk):
    """w for X0, Y0, m36_0 given a branch's E_HK normal.

    w_ehk is (8, N) -- one column per source, since the branch varies.
    The unit vectors broadcast against it as (8, 1).
    """
    e1, e2, ek = np.zeros((8, 1)), np.zeros((8, 1)), np.zeros((8, 1))
    e1[_1] = 1.0
    e2[_2] = 1.0
    ek[_K] = 1.0
    return {
        "X0": e1 - e2 - fz.C_3645_CORRECTED * w_ehk,
        "Y0": ek - e1 - w_ehk / fz.E_HK_OVER_E_K36,
        "m36_0": e1 - fz.A36_OVER_E_HK * w_ehk,
    }


# --- structural classification of rows -----------------------------------

@dataclass(frozen=True)
class RowCase:
    kind: str                 # "flux1" | "flux2" | "mag"
    band_a: int = -1          # flux1: the band;  flux2: the +1 band
    band_b: int = -1          # flux2: the -1 band
    coef: float = 1.0         # flux1: the coefficient on band_a


def classify_row_cases(tab) -> Dict[str, RowCase]:
    """Which evaluation route each row takes.  A property of the scheme,
    computed once -- never of a source."""
    cases = {}
    for row in tab.rows:
        if row.is_derived:
            cases[row.id] = RowCase("mag")
            continue
        w = tab.w_mag[row.id]
        nz = np.flatnonzero(w)
        if len(nz) == 1 and abs(abs(w[nz[0]]) - 1.0) < 1e-12:
            cases[row.id] = RowCase("flux1", band_a=int(nz[0]), coef=float(w[nz[0]]))
        elif len(nz) == 2 and np.allclose(np.abs(w[nz]), 1.0):
            pos = nz[w[nz] > 0]
            neg = nz[w[nz] < 0]
            if len(pos) == 1 and len(neg) == 1:
                cases[row.id] = RowCase("flux2", band_a=int(pos[0]), band_b=int(neg[0]))
            else:
                cases[row.id] = RowCase("mag")
        else:
            cases[row.id] = RowCase("mag")
    return cases


# --- the softened step ----------------------------------------------------

def _positions(js):
    """A family's destination rows, as a slice when they are a contiguous
    run (then the family writes straight into the row buffer) and as an
    index array otherwise."""
    a = np.array(js, dtype=np.intp)
    if a.size and a[-1] - a[0] == a.size - 1:
        return slice(int(a[0]), int(a[-1]) + 1)
    return a


def _ndtr_into(num, den, out):
    """Phi(num/den) written into `out`, with den = 0 handled as the limit:
    a zero-width distribution puts all its mass on the sign of num.

    `num` and `out` are (k, N); `den` may be (k, N) or (k, 1) -- the
    per-source case, where the width is a constant of the source.
    `num` must not alias `out`.
    """
    wide = den > 0
    np.divide(num, np.where(wide, den, 1.0), out=out)
    ndtr(out, out=out)
    if not wide.all():
        np.copyto(out, (num > 0).astype(float), where=~wide)
    return out


# --- compiled scheme ------------------------------------------------------
#
# Everything below depends only on the TABLE (and, for a route-specialised
# variant, on the route code).  It is built once and reused by every
# source and every model batch.

class _Flux1Family:
    """flux1 rows: a magnitude threshold IS a flux threshold."""
    __slots__ = ("pos", "band", "sgn", "t", "coef", "ksigma", "has_k", "k")

    def __init__(self, tab, cases, js):
        rows = [tab.rows[j] for j in js]
        cs = [cases[r.id] for r in rows]
        self.k = len(js)
        self.pos = _positions(js)
        self.band = np.array([c.band_a for c in cs], dtype=np.intp)
        self.coef = np.array([c.coef for c in cs], dtype=float)
        self.sgn = np.array([r.direction * (1 if c.coef > 0 else -1)
                             for r, c in zip(rows, cs)], dtype=float)[:, None]
        self.t = np.array([r.t for r in rows], dtype=float)
        self.ksigma = [dict(r.ksigma) for r in rows]
        self.has_k = any(self.ksigma)


class _Flux2Family:
    """flux2 rows: a colour threshold IS a flux-ratio threshold."""
    __slots__ = ("pos", "a", "b", "d", "t", "zpr", "ksigma", "has_k", "k")

    def __init__(self, tab, cases, js):
        rows = [tab.rows[j] for j in js]
        cs = [cases[r.id] for r in rows]
        self.k = len(js)
        self.pos = _positions(js)
        self.a = np.array([c.band_a for c in cs], dtype=np.intp)
        self.b = np.array([c.band_b for c in cs], dtype=np.intp)
        self.d = np.array([r.direction for r in rows], dtype=float)[:, None]
        self.t = np.array([r.t for r in rows], dtype=float)
        self.zpr = np.array([ZP[c.band_b] / ZP[c.band_a] for c in cs])[:, None]
        self.ksigma = [dict(r.ksigma) for r in rows]
        self.has_k = any(self.ksigma)


class _MagFamily:
    """Phase-1 rows with non-integer weights: Phi in magnitude space,
    Cornish-Fisher corrected.  `W` is the 8-magnitude expansion, so the
    bias-corrected colours never have to be re-derived (F9)."""
    __slots__ = ("pos", "W", "W2", "W3", "Wnz", "d", "t", "wsum", "ksigma", "k")

    def __init__(self, tab, js):
        rows = [tab.rows[j] for j in js]
        self.k = len(js)
        self.pos = _positions(js)
        W = (np.column_stack([tab.w_mag[r.id] for r in rows])
             if rows else np.zeros((8, 0)))
        self.W = np.ascontiguousarray(W)
        self.W2 = np.ascontiguousarray(W ** 2)
        self.W3 = np.ascontiguousarray(W ** 3)
        self.Wnz = np.ascontiguousarray((W != 0.0).astype(float))
        self.d = np.array([r.direction for r in rows], dtype=float)[:, None]
        self.t = np.array([r.t for r in rows], dtype=float)
        self.wsum = W.sum(axis=0)
        self.ksigma = [dict(r.ksigma) for r in rows]


class _DerivedFamily:
    """Phase-2 rows.  Affine only within a dereddening branch, so their
    normals are a (4, k, 8) constant of the scheme (F3) rather than an
    (8, N) array rebuilt per row per call.

    The max(E_HK, 0) clamp (erratum E4) makes each row's TRUE
    distribution a two-component mixture rather than one smooth
    Gaussian -- see docs/gut09_probabilistic_cascade.md 8.1.  Alongside
    the existing unclamped normal (`WD2`, `WDSUM`), this stores what the
    mixture needs: `WDxEHK` for Cov(E_HK, row) (the truncated-moment
    correlation term), and `WC2`/`WCSUM` for the clamped branch's own
    variance, evaluated at the SINGLE (branch-independent) normal where
    X0, Y0, m36_0 reduce to their undereddened forms.
    """
    __slots__ = ("pos", "coefs", "ksigma", "d", "t", "WD2", "WDSUM",
                 "WDxEHK", "WC2", "WCSUM", "needs_colours", "k")

    def __init__(self, tab, js):
        rows = [tab.rows[j] for j in js]
        self.k = len(js)
        self.pos = _positions(js)
        self.coefs = [dict(r.coef) for r in rows]
        self.ksigma = [dict(r.ksigma) for r in rows]
        self.d = np.array([r.direction for r in rows], dtype=float)
        self.t = np.array([r.t for r in rows], dtype=float)
        names = set().union(*(set(c) for c in self.coefs)) if rows else set()
        self.needs_colours = bool(names - {"X0", "Y0", "m36_0"})

        # the clamped branch (E_HK == 0): a single normal, since the
        # w_ehk term drops out identically regardless of which of the
        # four dereddening branches would otherwise have applied.
        dn0 = _derived_normals(np.zeros((8, 1)))
        WC = np.zeros((self.k, 8))
        for col, row in enumerate(rows):
            wc = np.zeros(8)
            for fname, c in row.coef.items():
                wc = wc + c * dn0[fname][:, 0]
            WC[col] = wc
        self.WC2 = np.ascontiguousarray(WC ** 2)
        self.WCSUM = WC.sum(axis=1)

        WD2 = np.zeros((len(BRANCH_KEYS), self.k, 8))
        WDxEHK = np.zeros((len(BRANCH_KEYS), self.k, 8))
        WSUM = np.zeros((len(BRANCH_KEYS), self.k))
        for bi, key in enumerate(BRANCH_KEYS):
            w_ehk = EHK_NORMAL[key]
            dn = _derived_normals(w_ehk[:, None])
            for col, row in enumerate(rows):
                w = np.zeros(8)
                for fname, c in row.coef.items():
                    w = w + c * dn[fname][:, 0]
                WD2[bi, col] = w ** 2
                WDxEHK[bi, col] = w * w_ehk
                WSUM[bi, col] = w.sum()
        self.WD2 = np.ascontiguousarray(WD2.reshape(-1, 8))
        self.WDxEHK = np.ascontiguousarray(WDxEHK.reshape(-1, 8))
        self.WDSUM = WSUM.reshape(-1)


class CompiledScheme:
    """The boundary table, compiled for evaluation.

    `live` selects the rows to evaluate; a route-specialised variant
    drops the rows that feed gates the route can never reach (F8).  Rows
    outside `live` are left at zero in the row buffer, and the gates
    they feed are reported as zero -- neither is read by the schedule
    for that route, so the label distribution is unchanged.
    """

    def __init__(self, tab, live=None, route_code=None):
        self.tab = tab
        self.route_code = route_code
        self.n_rows = len(tab.rows)
        self.n_terms = len(tab.terms)
        self.cases = classify_row_cases(tab)
        self.sig_names = tuple(tab.sigmas)
        live = (np.ones(self.n_rows, dtype=bool) if live is None
                else np.asarray(live, dtype=bool))
        self.live = live
        self.subset = not live.all()

        f1, f1v, f2, f2v, mg, dv = [], [], [], [], [], []
        for j, row in enumerate(tab.rows):
            if not live[j]:
                continue
            kind = self.cases[row.id].kind
            if kind == "flux1":
                (f1v if row.ksigma else f1).append(j)
            elif kind == "flux2":
                (f2v if row.ksigma else f2).append(j)
            elif row.is_derived:
                dv.append(j)
            else:
                mg.append(j)

        self.flux1 = [_Flux1Family(tab, self.cases, js) for js in (f1, f1v) if js]
        self.flux2 = [_Flux2Family(tab, self.cases, js) for js in (f2, f2v) if js]
        self.mag = _MagFamily(tab, mg) if mg else None
        self.derived = _DerivedFamily(tab, dv) if dv else None

        self.guarded = [(j, [g.strip() for g in tab.rows[j].guard.split(",")])
                        for j in range(self.n_rows)
                        if live[j] and tab.rows[j].guard]

        # incidence and gate->column maps, compiled once (F7).  The pair
        # order reproduces compose's multiplication order exactly.
        M = tab.incidence()
        idx = {rid: i for i, rid in enumerate(tab.row_ids)}
        dead_terms = {ti for ti in range(M.shape[0])
                      if not all(live[idx[rid]] for rid in tab.terms[ti].rows)}
        self.pairs = [(ti, int(ri), int(M[ti, ri]))
                      for ti in range(M.shape[0]) if ti not in dead_terms
                      for ri in np.flatnonzero(M[ti])]
        self.gate_cols = {}
        for g in tab.gates:
            cols = [i for i, t in enumerate(tab.terms)
                    if t.gate == g and i not in dead_terms]
            if cols:
                self.gate_cols[g] = cols
        self._variants = {}

    # -- route specialisation ---------------------------------------
    def live_rows_for(self, code):
        """Boolean row mask: which rows can affect the label of a source
        on route `code`.  Rows feeding gates no schedule entry can reach
        for that route are dead."""
        tab = self.tab
        p12 = int(code) & 3
        has24 = bool(int(code) & 4)
        gates = set()
        for e in tab.schedule:
            if e.gate is None:
                continue
            if e.phase in (1, 2):
                for r in (e.route or ()):
                    if crisp._ROUTE_CODE[r] == p12:
                        gates.add(e.gate)
                        break
            elif has24:
                gates.add(e.gate)
        idx = {rid: i for i, rid in enumerate(tab.row_ids)}
        live = np.zeros(self.n_rows, dtype=bool)
        for t in tab.terms:
            if t.gate in gates:
                for rid in t.rows:
                    live[idx[rid]] = True
        return live

    def for_route(self, code):
        code = int(code)
        hit = self._variants.get(code)
        if hit is None:
            hit = CompiledScheme(self.tab, self.live_rows_for(code), code)
            self._variants[code] = hit
        return hit


_COMPILED = {}


def compiled(tab):
    """The compiled form of `tab`, cached per table identity.

    Keyed on the table object, not on nothing: two tables built with
    different sigma bindings must not share a compilation.
    """
    key = id(tab)
    hit = _COMPILED.get(key)
    if hit is None or hit[0] is not tab:
        hit = (tab, CompiledScheme(tab))
        _COMPILED[key] = hit
    return hit[1]


def row_cases(tab):
    """Structural row classification for `tab`, cached with its
    compilation."""
    return compiled(tab).cases


# --- per-source context ---------------------------------------------------

class SourceContext:
    """Everything the cascade can settle before it sees a model batch.

    Built by `prepare_source`.  Every array here has a trailing axis of
    length 1 (the pair-completed case, where sigma/flux is a constant of
    the source) or N (a heterogeneous batch); both broadcast against the
    (rows, N) buffers of the batched core.
    """

    __slots__ = ("scheme", "tab", "cfg", "s", "sigma_cal", "corrections",
                 "route", "guards", "has_J", "n_src", "mu",
                 "f1_thresh", "f2_ratio", "mg_off", "mg_nu_safe", "mg_wide",
                 "mg_all_wide", "mg_gam", "dv_nu", "dv_off",
                 "dv_cov", "dv_ehk_nu2", "dv_nuC2")

    @property
    def n_rows(self):
        return self.scheme.n_rows

    @property
    def live_rows(self):
        """(45,) bool -- which rows this context actually evaluates."""
        return self.scheme.live


def _build_context(scheme, cfg, sig, guards, route, has_J, sig_mag,
                   mu, sd, gamma, s, sigma_cal, corrections):
    """Assemble a SourceContext from already-propagated per-source
    quantities.  `sig` is the colour-sigma dict, `sig_mag` the per-band
    magnitude sigma (inf on an undetected band), and (mu, sd, gamma) the
    flux->magnitude moment tables, or None when corrections are off."""
    ctx = SourceContext()
    ctx.scheme = scheme
    ctx.tab = scheme.tab
    ctx.cfg = cfg
    ctx.s = float(s)
    ctx.sigma_cal = float(sigma_cal)
    ctx.corrections = bool(corrections)
    ctx.route = route
    ctx.guards = guards
    ctx.has_J = has_J
    ctx.mu = mu

    # per-band sd used by the magnitude-space rows: the tabulated sd(r)
    # when correcting, else the first-order sigma_mag.  Undetected bands
    # carry sigma_mag = inf; zero them for the variance sum, since any
    # row that reads such a band already has a NaN margin.
    sm = np.where(np.isfinite(sig_mag), sig_mag, 0.0)
    ctx.n_src = sm.shape[0]

    # -- effective thresholds.  Gut's own sigma penalties move the
    #    threshold, and sigma is a constant of the source, so t_eff is
    #    too: 28 of 45 rows then exponentiate a SCALAR (F1).
    def _t_eff(fam):
        t = np.repeat(fam.t[:, None], ctx.n_src if fam.has_k else 1, axis=1)
        if fam.has_k:
            for col, ks in enumerate(fam.ksigma):
                for name, k in ks.items():
                    t[col] = t[col] - k * np.asarray(sig[name])
        return t

    ctx.f1_thresh = [ZP[f.band][:, None] * 10.0 ** (-(_t_eff(f) / f.coef[:, None]) / 2.5)
                     for f in scheme.flux1]
    ctx.f2_ratio = [10.0 ** (_t_eff(f) / 2.5) * f.zpr for f in scheme.flux2]

    mg = scheme.mag
    if mg is not None:
        off = np.zeros((mg.k, ctx.n_src))
        for col, ks in enumerate(mg.ksigma):
            for name, k in ks.items():
                off[col] = off[col] + k * np.asarray(sig[name])
        ctx.mg_off = off - mg.t[:, None]

        sd2 = np.ascontiguousarray(((sd ** 2) if corrections else (sm ** 2)).T)
        nu2 = mg.W2.T @ sd2
        if sigma_cal > 0.0:
            # (sum_b w_b)^2 is 0 for any pure colour and 1 for a
            # single-band cut -- the cancellation is automatic
            nu2 = nu2 + (mg.wsum[:, None] * sigma_cal) ** 2
        nu = np.sqrt(nu2)
        wide = nu > 0
        ctx.mg_wide = wide
        ctx.mg_all_wide = bool(wide.all())
        ctx.mg_nu_safe = np.where(wide, nu, 1.0)
        if corrections:
            # third cumulant of w.m, then Cornish-Fisher on the
            # standardised margin.  The odd cumulant flips sign with the
            # row's sense, hence the `direction` factor.
            k3 = mg.W3.T @ np.ascontiguousarray((gamma * sd ** 3).T)
            ctx.mg_gam = np.where(wide, mg.d * k3 / ctx.mg_nu_safe ** 3, 0.0)
        else:
            ctx.mg_gam = None
    else:
        ctx.mg_off = ctx.mg_nu_safe = ctx.mg_wide = ctx.mg_gam = None
        ctx.mg_all_wide = True

    dv = scheme.derived
    if dv is not None:
        # Phase-2 rows are never bias-corrected: the log-transform bias
        # and skew is not the dominant error there -- the max(E_HK, 0)
        # clamp is (erratum E4).  Their width uses sigma_mag directly.
        # Fixed by treating the row as a two-component mixture: the
        # clamped branch at E_HK = 0, and the unclamped branch through
        # its truncated (E_HK > 0) conditional moments -- see the dv
        # block in _row_block and docs/gut09_probabilistic_cascade.md
        # 8.1.  Everything precomputable from sigma alone (source-level,
        # not model-level) is hoisted here, same as the rest of the file.
        sm2T = np.ascontiguousarray((sm ** 2).T)          # (8, n_src)
        nu2 = dv.WD2 @ sm2T
        cov = dv.WDxEHK @ sm2T
        ehk_nu2 = (EHK_NORMAL_MAT ** 2) @ sm2T             # (4, n_src)
        nuC2 = dv.WC2 @ sm2T                                # (k, n_src)
        if sigma_cal > 0.0:
            nu2 = nu2 + (dv.WDSUM[:, None] * sigma_cal) ** 2
            ehk_wsum_rep = np.repeat(EHK_WSUM, dv.k)         # (4k,)
            cov = cov + (dv.WDSUM * ehk_wsum_rep)[:, None] * sigma_cal ** 2
            ehk_nu2 = ehk_nu2 + (EHK_WSUM[:, None] * sigma_cal) ** 2
            nuC2 = nuC2 + (dv.WCSUM[:, None] * sigma_cal) ** 2
        ctx.dv_nu = np.sqrt(nu2)
        ctx.dv_cov = cov
        ctx.dv_ehk_nu2 = ehk_nu2
        ctx.dv_nuC2 = nuC2
        off = np.zeros((dv.k, ctx.n_src))
        for col, ks in enumerate(dv.ksigma):
            for name, k in ks.items():
                off[col] = off[col] + k * np.asarray(sig[name])
        ctx.dv_off = off - dv.t[:, None]
    else:
        ctx.dv_nu = ctx.dv_off = None
        ctx.dv_cov = ctx.dv_ehk_nu2 = ctx.dv_nuC2 = None

    return ctx


def prepare_source(flux_mjy, sigma_mjy, valid=None, detected=None, config=None,
                   tab=None, s=0.0, sigma_cal=0.0, corrections=True, origin=None,
                   live_rows=False):
    """Hoist every sigma-derived constant of one source out of the model
    batch, returning a context for `classify_prob_batched`.

    In the fitting loop the per-band FRACTIONAL error is fixed down the
    batch -- sigma_s/f_s in the detected bands (the source's own
    photometry, spliced in unchanged) and the per-band median fractional
    sigma_fill elsewhere (R-52).  So r = sigma/flux, and with it the
    detection mask, the admission route, all 19 guards, the propagated
    colour sigmas, the flux->magnitude bias/skew tables, the effective
    thresholds, the Cornish-Fisher (nu, gamma) pairs and the Phase-2
    nu^2 table, are constants of the SOURCE.  Pass a single (1, 8) row
    of flux and sigma and they are computed once.

    An (N, 8) pair may also be passed, in which case the same quantities
    are carried per row; the batched core broadcasts either way.

    `live_rows=True` additionally specialises the compiled scheme to the
    source's route, skipping the rows that feed gates the route cannot
    reach.  The label distribution is unchanged; the skipped entries of
    `row_prob` / `gate_prob` are reported as zero.  It applies only when
    every row of the context shares one route code.
    """
    valid = crisp._valid_arg(valid, origin)
    cfg = config or fz.SchemeConfig()
    tab = tab or crisp.table(cfg.sigma_binding)

    flux = np.atleast_2d(np.asarray(flux_mjy, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma_mjy, dtype=float))
    if valid is not None:
        valid = np.atleast_2d(np.asarray(valid, dtype=bool))

    sigma_mag = fz.sigma_to_mag(flux, sigma)
    usable = rt.detection_mask(sigma_mag, valid)
    has_J = usable[..., _J]

    if detected is None:
        seen, route_seen = usable, None
    else:
        seen = np.atleast_2d(np.asarray(detected, dtype=bool))
        route_seen = rt.route(seen)

    route_code = rt.route(usable)
    guards = rt.guard_values(usable, route_code, seen,
                             route_code if route_seen is None else route_seen)

    # `s` is an isotropic floor in MAGNITUDES -- boundary uncertainty, not
    # measurement error -- so it must not affect the detection mask or the
    # admission route, which are settled above on the reported sigma.
    if s > 0.0:
        sigma = np.sqrt(sigma ** 2 + (np.abs(flux) * s / _K_MAG) ** 2)
        sigma_mag = fz.sigma_to_mag(flux, sigma)

    sig = fz.colour_sigmas(sigma_mag)

    mu = sd = gamma = None
    if corrections:
        # bias and skew of the flux -> magnitude transform, for the rows
        # that cannot be evaluated in flux space.  Vanishes at sigma = 0,
        # and depends on flux and sigma only through r = sigma/flux.
        _, mu, sd, gamma = co.band_corrections(flux, sigma)

    scheme = compiled(tab)
    if live_rows:
        codes = np.unique(route_code)
        if codes.size == 1:
            scheme = scheme.for_route(int(codes[0]))

    return _build_context(scheme, cfg, sig, guards, route_code, has_J,
                          sigma_mag, mu, sd, gamma, s, sigma_cal, corrections)


# --- the batched core -----------------------------------------------------

_ZERO = np.zeros(1)


def _row_block(ctx, fluxT, sigmaT, mag, feat, branch, n):
    """(n_rows, N) probability that each boundary holds.

    Row-contiguous throughout (F4): each family is evaluated as one
    matrix of margins and one `ndtr` (F5), then scattered into the
    buffer.  `feat` and `branch` supply the Phase-2 features; they are
    computed here unless a caller already holds them.
    """
    C = ctx.scheme
    P = np.zeros((C.n_rows, n)) if C.subset else np.empty((C.n_rows, n))

    def dest(fam):
        """The family's slot in the buffer: a view when its rows are a
        contiguous run, else a scratch matrix to scatter afterwards."""
        return P[fam.pos] if type(fam.pos) is slice else np.empty((fam.k, n))

    def put(fam, buf):
        if type(fam.pos) is not slice:
            P[fam.pos] = buf

    # -- flux1: a magnitude threshold IS a flux threshold ------------
    for fam, F_t in zip(C.flux1, ctx.f1_thresh):
        Fb, sb = fluxT[fam.band], sigmaT[fam.band]
        if ctx.sigma_cal > 0.0:
            # a single-band cut has sum(w) = 1, so the common-mode
            # calibration term survives here at full weight
            sb = np.sqrt(sb ** 2 + (np.abs(Fb) * ctx.sigma_cal / _K_MAG) ** 2)
        put(fam, _ndtr_into(fam.sgn * (F_t - Fb), sb, dest(fam)))

    # -- flux2: a colour threshold IS a flux-ratio threshold ---------
    for fam, k in zip(C.flux2, ctx.f2_ratio):
        fa, fb = fluxT[fam.a], fluxT[fam.b]
        sa, sb = sigmaT[fam.a], sigmaT[fam.b]
        put(fam, _ndtr_into(fam.d * (fb - k * fa),
                            np.sqrt(sb ** 2 + (k * sa) ** 2), dest(fam)))

    # -- magnitude space, Cornish-Fisher corrected -------------------
    mg = C.mag
    if mg is not None:
        # w_mag IS the colour expansion, so the bias-corrected margin is
        # one (8, k) matmul against the corrected magnitudes -- the
        # second colours()/deredden() pass is never needed (F9).
        mag_c = (mag + ctx.mu) if ctx.corrections else mag
        nanm = ~np.isfinite(mag_c)
        any_nan = bool(nanm.any())
        mcT = np.ascontiguousarray((np.where(nanm, 0.0, mag_c) if any_nan
                                    else mag_c).T)
        g = dest(mg)
        np.matmul(mg.W.T, mcT, out=g)
        g += ctx.mg_off
        g *= mg.d
        gpos = None if ctx.mg_all_wide else (g > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g /= ctx.mg_nu_safe
            if ctx.corrections:
                shift = (ctx.mg_gam / 6.0) * (g ** 2 - 1.0)
                # the expansion is asymptotic; the clip exists so a
                # pathological input degrades to Phi(z), not to nonsense
                az = np.abs(g)
                np.clip(shift, -0.5 * az - 0.5, 0.5 * az + 0.5, out=shift)
                g += shift
            ndtr(g, out=g)
        if gpos is not None:
            np.copyto(g, gpos.astype(float), where=~ctx.mg_wide)
        if any_nan:
            # a row may only be poisoned by a NaN magnitude in a band it
            # actually reads -- the matmul would spread it to all rows
            np.copyto(g, 0.0, where=(mg.Wnz.T @ nanm.T.astype(float)) > 0)
        put(mg, g)

    # -- Phase 2: affine within a dereddening branch, mixed across the
    #    max(E_HK, 0) clamp (erratum E4 fix) --------------------------
    dv = C.derived
    if dv is not None:
        bidx = np.where(ctx.has_J, 2, 0) + branch                    # (n,)
        w_ehk = EHK_NORMAL_MAT[bidx]                                   # (n, 8)
        # a band this branch doesn't read (weight 0) may still be NaN
        # (non-positive flux, or simply undetected) -- 0 * NaN is NaN, so
        # fill before the dot and poison only rows a NaN band actually
        # feeds, same convention as the mag family below.
        nan_mag = ~np.isfinite(mag)
        mag_safe = np.where(nan_mag, 0.0, mag)
        poison_ehk = np.any((w_ehk != 0.0) & nan_mag, axis=1)
        mu_ehk = np.einsum("ij,ij->i", mag_safe, w_ehk) + EHK_CONST_ARR[bidx]
        mu_ehk = np.where(poison_ehk, np.nan, mu_ehk)

        take = bidx[None, :] * dv.k + np.arange(dv.k)[:, None]         # (k, n)
        cov = (ctx.dv_cov[take, 0] if ctx.dv_cov.shape[1] == 1
              else np.take_along_axis(ctx.dv_cov, take, axis=0))
        nu_ru2 = (ctx.dv_nu[take, 0] ** 2 if ctx.dv_nu.shape[1] == 1
                 else np.take_along_axis(ctx.dv_nu, take, axis=0) ** 2)
        ehk_nu2 = (ctx.dv_ehk_nu2[bidx, 0] if ctx.dv_ehk_nu2.shape[1] == 1
                  else ctx.dv_ehk_nu2[bidx, np.arange(n)])
        nu_ehk = np.sqrt(ehk_nu2)
        wide_ehk = nu_ehk > 0.0

        # P(E_HK <= 0) -- exact given the affine-in-mag / Gaussian-in-flux
        # model, via the same zero-width limit every other row uses.
        w_clamp = _ndtr_into((-mu_ehk)[None, :], nu_ehk[None, :],
                             np.empty((1, n)))[0]

        # Truncated-normal moments of R_u | E_HK > 0, by moment matching
        # (R_u is affine in the same Gaussian mag vector as E_HK, so the
        # bivariate truncation moments are closed-form -- no quadrature
        # needed).  c is the standardised distance of E_HK below zero;
        # lam is the inverse Mills ratio E[Z | Z > c].
        with np.errstate(divide="ignore", invalid="ignore"):
            nu_ehk_safe = np.where(wide_ehk, nu_ehk, 1.0)
            c = np.where(wide_ehk, -mu_ehk / nu_ehk_safe, 0.0)
            tail = ndtr(-c)
            phi_c = np.exp(-0.5 * c * c) / np.sqrt(2.0 * np.pi)
            lam = phi_c / np.where(tail > 1e-300, tail, 1e-300)
            beta = np.where(wide_ehk, cov / nu_ehk_safe[None, :], 0.0)

        A = mag[:, _1] - mag[:, _2]
        feat_c = {"X0": A, "Y0": mag[:, _K] - mag[:, _1], "m36_0": mag[:, _1]}
        feat_u = {"X0": A - mu_ehk * fz.C_3645_CORRECTED,
                 "Y0": (mag[:, _K] - mag[:, _1]) - mu_ehk / fz.E_HK_OVER_E_K36,
                 "m36_0": mag[:, _1] - mu_ehk * fz.A36_OVER_E_HK}

        gd_c = np.empty((dv.k, n))
        gd_u = np.empty((dv.k, n))
        var_u = np.empty((dv.k, n))
        for i, coef in enumerate(dv.coefs):
            acc_c = np.zeros(n)
            acc_u = np.zeros(n)
            for fname, c_coef in coef.items():
                if fname in feat_c:
                    acc_c = acc_c + c_coef * feat_c[fname]
                    acc_u = acc_u + c_coef * feat_u[fname]
                else:
                    v = np.asarray(feat[fname])
                    acc_c = acc_c + c_coef * v
                    acc_u = acc_u + c_coef * v
            gd_c[i] = dv.d[i] * (acc_c + ctx.dv_off[i])
            mu_u_trunc = acc_u + beta[i] * lam
            var_u[i] = np.where(
                wide_ehk,
                np.clip(nu_ru2[i] - beta[i] ** 2 * lam * (lam - c), 0.0, None),
                nu_ru2[i])
            gd_u[i] = dv.d[i] * (mu_u_trunc + ctx.dv_off[i])

        Pc = _ndtr_into(gd_c, np.sqrt(ctx.dv_nuC2), np.empty((dv.k, n)))
        Pu = _ndtr_into(gd_u, np.sqrt(var_u), np.empty((dv.k, n)))
        out = dest(dv)
        out[:] = w_clamp[None, :] * Pc + (1.0 - w_clamp[None, :]) * Pu
        put(dv, out)

    # -- one batched finite/guard pass over the whole buffer (F5) ----
    np.copyto(P, 0.0, where=~np.isfinite(P))
    for j, names in C.guarded:
        for gname in names:
            P[j] *= ctx.guards[gname]
    return P


def _compose(ctx, P, n):
    """P(T) = prod over the term's rows with the ternary sign, then
    P(G) = 1 - prod(1 - P(T)) over the gate's terms."""
    C = ctx.scheme
    T = np.ones((C.n_terms, n))
    for ti, ri, s in C.pairs:
        T[ti] *= P[ri] if s > 0 else (1.0 - P[ri])
    gate_cols = C.gate_cols
    return {g: (1.0 - np.prod(1.0 - T[gate_cols[g]], axis=0)
                if g in gate_cols else _ZERO)
            for g in C.tab.gates}


def run_schedule_prob(tab, gates, route_code, guards, n=None):
    """Chain rule over the cascade, then Phase 3 as a stochastic matrix.

    `route_code` and the guards may be length-1 (a per-source constant);
    pass `n` for the batch size in that case.
    """
    n = len(route_code) if n is None else int(n)
    K = len(crisp.LABELS)
    P = np.zeros((n, K))
    unassigned = np.ones(n)

    p12 = rt.phase12(route_code)
    has24 = rt.has_24(route_code)

    for e in tab.schedule:
        target = crisp.LABEL_INDEX[e.assign]

        if e.phase in (1, 2):
            applies = np.zeros(n, dtype=bool)
            for r in (e.route or ()):
                applies |= p12 == crisp._ROUTE_CODE[r]
            fire = np.ones(n) if e.gate is None else gates[e.gate]
            if e.on == "not_fire":
                fire = 1.0 - fire
            if e.guard is not None:
                applies &= guards[e.guard]

            moved = unassigned * fire * applies
            P[:, target] += moved
            unassigned -= moved

        else:  # phase 3 -- a transition on mass already assigned
            applies = np.array(has24, copy=True)
            if e.guard is not None:
                applies &= guards[e.guard]
            fire = gates[e.gate]
            if e.on == "not_fire":
                fire = 1.0 - fire
            rate = fire * applies

            if e.when == "any":
                sources = list(range(K))
                moved_un = unassigned * rate
                P[:, target] += moved_un
                unassigned -= moved_un
            else:
                sources = [crisp.LABEL_INDEX[name] for name in e.when]

            take = np.zeros(n)
            for k in sources:
                if k == target:
                    continue
                m = P[:, k] * rate
                P[:, k] -= m
                take += m
            P[:, target] += take

    P[:, crisp.LABEL_INDEX["UNCLASSIFIED"]] += unassigned
    return P


@dataclass
class ProbResult:
    prob: np.ndarray           # (N, 11), rows sum to 1
    route: np.ndarray          # (N,)
    row_prob: np.ndarray       # (N, 45)
    gate_prob: np.ndarray      # (N, n_gates)


def classify_prob_batched(ctx, flux_mjy, sigma_mjy):
    """The cascade for one model batch against a prepared source.

    `ctx` comes from `prepare_source` and carries the route, the guards
    and every sigma-derived constant; only the fluxes change down the
    batch.  Returns the same ProbResult as `classify_prob`.
    """
    C = ctx.scheme
    tab = C.tab
    flux = np.atleast_2d(np.asarray(flux_mjy, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma_mjy, dtype=float))
    n = flux.shape[0]

    if ctx.s > 0.0:
        sigma = np.sqrt(sigma ** 2 + (np.abs(flux) * ctx.s / _K_MAG) ** 2)

    fluxT = np.ascontiguousarray(flux.T)
    sigmaT = np.ascontiguousarray(sigma.T)
    mag = fz.flux_to_mag(flux)

    feat, branch = None, None
    dv = C.derived
    if dv is not None:
        X0, Y0, m36_0, _, branch = fz.deredden(mag, ctx.has_J, ctx.cfg)
        feat = fz.colours(mag) if dv.needs_colours else {}
        feat.update(X0=X0, Y0=Y0, m36_0=m36_0)

    # a zero-width boundary and a non-positive flux are both ordinary
    # outcomes here, and both are handled explicitly downstream
    with np.errstate(divide="ignore", invalid="ignore"):
        P = _row_block(ctx, fluxT, sigmaT, mag, feat, branch, n)
    gates = _compose(ctx, P, n)
    out = run_schedule_prob(tab, gates, ctx.route, ctx.guards, n=n)

    route = ctx.route
    if route.shape[0] != n:
        route = np.repeat(route, n)
    return ProbResult(
        prob=out,
        route=route,
        row_prob=P.T,
        gate_prob=np.column_stack([np.broadcast_to(gates[g], (n,))
                                   for g in tab.gates]),
    )


def row_probabilities(tab, flux, sigma, feat, sig, guards, aux, cases=None,
                      sigma_cal=0.0, corr=None):
    """(N, 45) probability that each boundary holds, from already
    propagated features.

    A thin adapter over the batched core, for callers that already hold
    `feat` / `sig` / `guards` / `aux`.  `cases` is accepted and ignored:
    the row classification now lives on the compiled scheme.
    """
    del cases
    scheme = compiled(tab)
    mu = corr["mu"] if corr is not None else None
    sd = corr["sd"] if corr is not None else None
    gamma = corr["gamma"] if corr is not None else None
    ctx = _build_context(scheme, fz.SchemeConfig(), sig, guards, None,
                         aux["has_J"], aux["sigma_mag"], mu, sd, gamma,
                         0.0, sigma_cal, corr is not None)
    flux = np.atleast_2d(np.asarray(flux, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        P = _row_block(ctx, np.ascontiguousarray(flux.T),
                       np.ascontiguousarray(sigma.T), aux["mag"], feat,
                       aux["branch"], flux.shape[0])
    return P.T


def classify_prob(flux_mjy, sigma_mjy, valid=None, detected=None, config=None,
                  tab=None, s=0.0, sigma_cal=0.0, corrections=True, origin=None):
    """Probabilistic Gutermuth label for an (N, 8) SED array.

    Returns a distribution over crisp.LABELS whose rows sum to 1.  At
    sigma = 0 it is one-hot and equals classify_crisp exactly.

    `valid` (may the cascade use this value?) and `detected` (did the
    survey see this band?) are as in classify_crisp, and differ only for
    an imputed SED.  `origin` is the former name of `valid`.

    This is a thin wrapper over `prepare_source` + `classify_prob_batched`
    and is the right entry point for a one-off call.  A fitting loop that
    evaluates many model SEDs against ONE source should call those two
    directly, so the per-source constants are hoisted out of the batch.

    When the batch itself is one source -- identical masks, and per-band
    fractional errors equal to SOURCE_RTOL down the rows, which is what a
    pair-completed batch looks like -- the wrapper detects it and builds
    the context from the first row alone, so it pays the per-source cost
    once rather than N times.  See `_one_source`.
    """
    valid = crisp._valid_arg(valid, origin)
    flux = np.atleast_2d(np.asarray(flux_mjy, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma_mjy, dtype=float))
    if valid is not None:
        valid = np.atleast_2d(np.asarray(valid, dtype=bool))
    if detected is not None:
        detected = np.atleast_2d(np.asarray(detected, dtype=bool))

    src = (slice(0, 1) if _one_source(flux, sigma, valid, detected)
           else slice(None))
    ctx = prepare_source(flux[src], sigma[src],
                         valid=None if valid is None else valid[src],
                         detected=None if detected is None else detected[src],
                         config=config, tab=tab, s=s, sigma_cal=sigma_cal,
                         corrections=corrections)
    return classify_prob_batched(ctx, flux, sigma)


# How closely the per-band fractional errors must agree down a batch for
# the wrapper to treat it as ONE source.  sigma_mag IS 2.5/ln10 times
# r = sigma/flux, so this is a relative tolerance on r.  It is not a
# free dial: a batch built as sigma = frac * flux reproduces `frac` only
# to a rounding error, so exact equality would miss the very case the
# hoist exists for, while 1e-12 bounds any induced change in a boundary
# probability far below the 1e-12 fixture-parity budget.
SOURCE_RTOL = 1e-12


def _one_source(flux, sigma, valid, detected):
    """Is this batch one source's photometry repeated -- same masks, same
    fractional errors -- so that every sigma-derived quantity is a
    constant down the rows?"""
    if flux.shape[0] < 2:
        return True
    if not (_rows_identical(valid) and _rows_identical(detected)):
        return False
    sm = fz.sigma_to_mag(flux, sigma)
    sm0 = sm[:1]
    with np.errstate(invalid="ignore"):
        # an undetected band carries inf, where `==` settles it; the
        # difference is only consulted where both entries are finite
        close = (sm == sm0) | (np.abs(sm - sm0) <= SOURCE_RTOL * np.abs(sm0))
    if not close.all():
        return False
    # the admission thresholds are step functions of sigma_mag, so the
    # detection mask must agree exactly, not merely to SOURCE_RTOL
    return _rows_identical(rt.detection_mask(sm, valid))


def _rows_identical(a):
    """True if every row of `a` equals the first.  None counts as
    identical; inf compares equal, NaN never does."""
    if a is None:
        return True
    return bool(np.all(a == a[:1]))
