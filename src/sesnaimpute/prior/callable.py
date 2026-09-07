"""The per-source posterior-facing prior callable (`10_POSTERIOR.md`
section 3's three properties; `SPEC_PRIORS.md` section 0.2; the prior
table's own join, `IMPLEMENTATION.md` sections 3 and 5).

`SourcePrior(config, region, cls=None)` loads one region's upstream
products: the prior table (`prior.table.read`) always, plus whichever of
the three field-family tile shapes (`prior.star_shapes`), the YSO
sightline shape and its shared column kernel (`prior.yso.YsoShape`,
`prior.kernel.Kernel`), the galaxy counts law (`bms/gal/counts/survey`),
and the H2S region lognormal (`bms/h2s/prior_h2s_region`) that `cls`
actually needs -- `cls=None` loads all six, for the identity check below;
a fitter worker scoped to one class loads only that class's own
materials and `log_density` refuses every other class. STAR/AGB/PAHC and
H2S read their selection directly per source from their own exact
per-source selection products (`prior.star_selection`, `prior.h2s`):
`EPS_<cls>[s]`, an `(n_x, n_b)` curve on the shared scaled-extinction
ladder `X_LADDER` (`x = a / A_s`) by the class's own second axis.
`SourcePrior.prepare(rows)` gathers, once per batch, only these two
products' rows (float16 on disk, converted to float32 for the batch) --
the one quantity too large to hold for a whole survey in memory at once.
GAL has no per-source product (owner, 2026-09-06): its selection is
looked up on the fly, per query point, in the survey-wide colour-CDF
tables loaded once in `__init__`, from the source's own four IRAC
limits (already resident in the region table) -- no `prepare` gather.
`log_density(cls, rows, a, log10_b, model_index=None)` returns `ln
lambda~_cls(a, log10 B | I_s)`, vectorised over the batch of sources and
query points (rule 8): no Python loop over either.

**The five formulas** (`SPEC_PRIORS.md` section 0.2's identity `N_C(s) =
Integral p_C(a,b|I_s) eps_s(a,b) da db`, `lambda~_C = p_C eps_s / N_C`):

STAR, AGB, PAHC
    `lambda~_C(a, log10 B) = p_C(a, log10 B | I_s) . eps_s(a, log10 B) / Z_C`

    `p_C` is the source's own tile shape, read through `star_shapes.
    ClassShape.density(a, log10_b, tile_id, A_s, sigma_col, map_class[,
    f_lim8])` itself: the shape's own two-component kernel mixture
    (`Kernel.mixture`), each component read at its own shift and width
    (bracketed in the width ladder, bicubic in `(log10 x, log10 B)`,
    PAHC additionally blended over its own 8 micron limit grid) and the
    two combined by the mixture weight `w`. Its MASS-per-cell output is
    converted to a density in `(a, log10 B)` here by the grid's fixed
    cell area and the `d(log10 x)/da = 1/(a ln10)` Jacobian.

    `eps_s(a, log10 B)` is this source's own `EPS_STAR`/`EPS_AGB`/
    `EPS_PAHC[s]`, an `(n_x, n_b)` curve on `X_LADDER` x
    `LOG10_B_GRID_<cls>` (`prior.star_selection`), read by one bilinear
    interpolation at `x = a / A_s` and the query's `log10 B`: held at the
    nearest edge value outside the tabulated box on either axis. No
    common-mode shift, no depth group: the population's exact selection
    was evaluated at this source's own eight limits when the product was
    built.

    `Z_C` is the table's own stored normaliser (`Z_STAR`/`Z_AGB`/
    `Z_PAHC`).

GAL
    `lambda~_GAL(a, log10 B) = p_a(a | A_s) . p(log10 S(log10 B), a) / Z_GAL`

    A galaxy's `a` is the column, spread only by the source's own column
    kernel (`SPEC_PRIORS.md` section 1.2): `p_a` is `Kernel.pdf` evaluated
    directly at this source's `(A_s, sigma_col, map_class)` -- a plain
    log-normal density in `T`, no tabulation, no quadrature.

    `p(log10 S, a)` is the counts law's own normalised density
    (`gal_phi_total = Integral phi(S) . S ln10 dlog10 S`, a survey-wide
    constant) times this source's own selection, evaluated at `log10 S =
    log10 B + log10 f_ref,h` (the query's library model's own 4.5 micron
    reference flux, `galz_register`'s `F_REF_I2`) and `a`: there is no
    per-source galaxy product (owner, 2026-09-06); the selection is
    looked up directly, inside the compiled kernel, in the survey-wide
    colour-CDF tables (`prior.gal.read_cdf_tables`, loaded once in
    `__init__`) from this source's own four IRAC limits (`F_LIM_50_MJY`,
    already in the region table) and the hybrid law's dimming at `a`
    (`selection.law_dense_weight`/`kappa_hybrid`, inlined as scalar
    arithmetic), linearly interpolated only across the S-grid's bins --
    `prior.gal.source_selection_from_cdf`'s own arithmetic, not a
    bilinear read of a stored array.

    `Z_GAL` is the table's own stored normaliser (`counts_star_family.
    gal_counts`'s per-source integral), read directly rather than
    recomputed here at the nominal column alone.

YSO
    `lambda~_YSO(a, log10 B) = p(a | A_s) . N(log10 B; mu(a), sigma)`,
    `mu(a) = RIDGE_INTERCEPT + RIDGE_SLOPE . a`

    `p(a | A_s)` is `YsoShape.marginal_exact`, this source's own exact
    extinction marginal at its own adopted column AND measurement
    uncertainty (`A_COL_K`, `A_COL_SIG_K`) -- closed form, a finite sum
    over the sightline's embedding cells, no quadrature. The conditional
    brightness is the closed-form Gaussian it always was. No selection in
    the YSO shape; `Z_YSO = 1` by construction, not read from the table.

H2S
    `lambda~_H2S(a, log10 B) = p(a | A_s) . p_r(log10 Sigma) . eps_s(a,
    Sigma) / Z_H2S`, `log10 Sigma = log10 B + log10 Sigma_ref,h`

    `p(a | A_s)` is YSO's own `marginal_exact`, shared not copied. `p_r`
    is the region's own lognormal in `log10 Sigma`
    (`LOGSIG_MEAN`/`LOGSIG_STD`), equally a Gaussian in `log10 B` for a
    fixed template. `eps_s` is this source's own `EPS[s]` (`prior.h2s`'s
    per-source selection, `(n_x, n_sigma)` on `X_LADDER` x
    `LOG10_SIGMA_GRID`), bilinearly interpolated at `x = a / A_s` and
    `log10 Sigma`. `Sigma_ref,h`: the h2shock register carries no
    dedicated H2 1-0 S(1) reference dataset, so this callable uses its
    `F_REF_Ks` as the brightness unit (disclosed fallback, unchanged from
    the prior design). `Z_H2S` is the table's stored `Z_H2S`.

Every class maps `a <= 0` to `-inf`. Outside a tabulated box, STAR/AGB/
PAHC apply the shape's own declared analytic tail; every class's `eps`
read holds the nearest edge value past `X_LADDER`'s ends or its own
second-axis grid ends (the same "end bins held" convention `SPEC_PRIORS.
md` section 1.3 already states for PAHC's own limit interpolation);
YSO/H2S are analytic in `a` throughout their support and Gaussian in
`log10 B`.

**Per-batch tabulation (`prepare`).** The only quantity too large to hold
in memory for a whole survey at once is the STAR-family and H2S
per-source selection products' `EPS` arrays (float16 on disk, `(n_source,
n_x, n_b-or-sigma)`): `prepare(rows)` reads and float32-converts only the
rows of the current batch (about ten thousand sources, `CODING_RULES.md`
10b) from each of the two files. GAL needs no gather -- there is no
per-source galaxy product (owner, 2026-09-06). Every other per-source
quantity (`A_COL_K`, `A_COL_SIG_K`, the ridge, the normalisers, GAL's own
four IRAC limits) is already resident in the region table loaded once in
`__init__`, and every extinction marginal (GAL's kernel `pdf`, YSO/H2S's
`marginal_exact`) is now closed-form, so `log_density` needs no further
per-batch tabulation for them.
"""

import math
import os
import time

import h5py
import numba
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute.prior import gal as gal_module
from sesnaimpute.prior import h2s as h2s_module
from sesnaimpute.prior import selection as selection_module
from sesnaimpute.prior import star_shapes
from sesnaimpute.prior import table as table_module
from sesnaimpute.prior import yso as yso_module

#: `ln(10)`: the `d(log10 x)/da = 1/(a ln10)` and `d(log10 S)/dS = 1/(S
#: ln10)` Jacobians every class but YSO/H2S needs once (module docstring).
LN10 = float(np.log(10.0))

#: `1/sqrt(2 pi)`: the Gaussian normalisation GAL's kernel density,
#: YSO's conditional brightness and H2S's region lognormal all need.
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

BAND_KEYS = tuple(b.key for b in definitions.BANDS)

FAMILY_CLASSES = ("star", "agb", "pahc")
CLASSES = ("star", "agb", "pahc", "gal", "yso", "h2s")

#: The two library keys `IMPLEMENTATION.md` section 3's GAL/H2S rows need
#: a per-model reference flux from (`sed_models/registers/<key>_register.
#: hdf5`, module docstring: H2S falls back to `F_REF_Ks`, the register
#: carrying no dedicated H2 1-0 S(1) dataset).
_LIBRARY_KEY = {"gal": "galz", "h2s": "h2shock"}
_LIBRARY_BAND = {"gal": "F_REF_I2", "h2s": "F_REF_Ks"}

#: `_interp_eps_2d`'s own query-batch block size (`CODING_RULES.md` 10a):
#: its `(chunk, n_x, n_v)` fancy-index intermediate never scales with the
#: caller's own, possibly source-times-model-sized, batch.
_INTERP_CHUNK = 20000


#: `_marginal_exact_chunked`'s own query-batch block size
#: (`CODING_RULES.md` 10a): `YsoShape.marginal_exact`'s own internal gather
#: is `(chunk, n_embedding_cell)` (hundreds of cells per sightline), so a
#: caller passing this file's own source-times-query-point batch straight
#: through would blow past the 8 GB ceiling on a batch of any size; this
#: chunk keeps that gather bounded regardless of how many query points the
#: caller passes.
_MARGINAL_CHUNK = 20000


def _marginal_exact_chunked(yso_shape, a, sl_rows, a_col, sigma_col, map_class, zp_sigma_k=None):
    """`(n,)`: `YsoShape.marginal_exact`, called in `_MARGINAL_CHUNK`
    blocks so its own per-call `(chunk, n_embedding_cell)` gather never
    scales with the caller's own batch (module docstring). `map_class` is
    the SOURCE's own arm (`A_COL_PROVENANCE`), required -- never the
    sightline's block-averaged one (fixed defect, owner 2026-09-06).
    `zp_sigma_k`, one per source (mag, 0 for Planck-arm), is the Herschel
    field zero point's own uncertainty; omitting it falls back to the
    survey-wide RMS, as before."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, _MARGINAL_CHUNK):
        stop = min(start + _MARGINAL_CHUNK, n)
        zp_chunk = zp_sigma_k[start:stop] if zp_sigma_k is not None else None
        out[start:stop] = yso_shape.marginal_exact(
            a[start:stop], sl_rows[start:stop], a_col[start:stop], sigma_col[start:stop],
            map_class[start:stop], zp_sigma_k=zp_chunk)
    return out


#: `sqrt(2)`, the standard-normal CDF's own argument scale (`erf(z /
#: _SQRT2_MARGINAL)`), needed before `_SQRT2PI` is defined further down.
_SQRT2_MARGINAL = float(np.sqrt(2.0))


@numba.njit(cache=True, fastmath=True, error_model="numpy")
def _lognormal_inv_moment_upto_scalar(t, m, s):
    """`E[1/T ; T <= t]` for `ln T ~ Normal(m, s)`, `yso._lognormal_
    inv_moment_upto`'s own scalar arithmetic (module docstring there):
    `exp(-m + s^2/2) . Phi((ln t - m)/s + s)`. `t <= 0` gives 0 (matches
    the numpy version's `errstate(divide="ignore")` -> `log(0) = -inf`
    -> `Phi(-inf) = 0`, without numba raising on `log` of a
    non-positive number)."""
    if t <= 0.0:
        return 0.0
    z = (math.log(t) - m) / s + s
    return math.exp(-m + 0.5 * s * s) * 0.5 * (1.0 + math.erf(z / _SQRT2_MARGINAL))


@numba.njit(cache=True, fastmath=True, error_model="numpy")
def _marginal_component_scalar(a, edges, p_u, n_cell, m, s):
    """One mixture component's `_marginal_rows` sum over the sightline's
    `n_cell` embedding cells (`yso.YsoShape._marginal_rows`'s own
    arithmetic): `p(a|A_s) = sum_k p_u[k] . (E[1/T; a/edges[k]] -
    E[1/T; a/edges[k+1]])`. `edges`/`p_u` are this ONE point's own
    sightline row, already gathered by the caller."""
    total = 0.0
    for k in range(n_cell):
        hi_k = a / edges[k]
        lo_k = a / edges[k + 1]
        total += p_u[k] * (_lognormal_inv_moment_upto_scalar(hi_k, m, s)
                           - _lognormal_inv_moment_upto_scalar(lo_k, m, s))
    return total


@numba.njit(cache=True, fastmath=True, error_model="numpy")
def _marginal_exact_numba(a, edges2d, p_u2d, a_col, w, mu0, sigma0, mu1, sigma1):
    """`(n,)`: `YsoShape.marginal_exact`'s own arithmetic, one compiled
    loop over the query batch -- `Kernel.mixture`'s `(w, mu, sigma)` is
    still computed in numpy by the caller (already cheap, vectorised,
    not the bottleneck); this loop is the 32-cell closed-form sum
    itself, the part that was scipy's `erf` over a `(chunk, n_cell)`
    broadcast array."""
    n = a.shape[0]
    n_cell = p_u2d.shape[1]
    out = np.empty(n, dtype=np.float64)
    ln10 = math.log(10.0)
    for i in range(n):
        ak = a[i]
        if ak <= 0.0:
            out[i] = 0.0
            continue
        log_a_col = math.log(a_col[i])
        m0 = log_a_col + mu0[i] * ln10
        s0 = sigma0[i] * ln10
        m1v = log_a_col + mu1[i] * ln10
        s1 = sigma1[i] * ln10
        t0 = _marginal_component_scalar(ak, edges2d[i], p_u2d[i], n_cell, m0, s0)
        t1 = _marginal_component_scalar(ak, edges2d[i], p_u2d[i], n_cell, m1v, s1)
        val = w[i] * t0 + (1.0 - w[i]) * t1
        out[i] = val if val > 0.0 else 0.0
    return out


def _marginal_exact_compiled(yso_shape, a, sl_rows, a_col, sigma_col, map_class, zp_sigma_k=None):
    """`(n,)`: the compiled replacement for `_marginal_exact_chunked`,
    same `_MARGINAL_CHUNK` chunking discipline (module docstring above
    it, `CODING_RULES.md` 10a): `Kernel.mixture`'s own structural
    interpolation stays in numpy per chunk (cheap, unchanged), but the
    32-cell closed-form sum is `_marginal_exact_numba`'s compiled loop
    instead of `YsoShape._marginal_rows`'s scipy-`erf` broadcast."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, _MARGINAL_CHUNK):
        stop = min(start + _MARGINAL_CHUNK, n)
        zp_chunk = zp_sigma_k[start:stop] if zp_sigma_k is not None else None
        a_c = a[start:stop]
        acol_c = a_col[start:stop]
        w, mu, sigma = yso_shape.kernel.mixture(
            acol_c, sigma_col[start:stop], map_class[start:stop], zp_sigma_k=zp_chunk)
        edges2d = np.ascontiguousarray(yso_shape.u_edges[sl_rows[start:stop]])
        p_u2d = np.ascontiguousarray(yso_shape.p_u[sl_rows[start:stop]])
        out[start:stop] = _marginal_exact_numba(
            a_c, edges2d, p_u2d, acol_c, w,
            np.ascontiguousarray(mu[:, 0]), np.ascontiguousarray(sigma[:, 0]),
            np.ascontiguousarray(mu[:, 1]), np.ascontiguousarray(sigma[:, 1]))
    return out


#: The per-source extinction-marginal curve `prepare` builds once per
#: batch (owner ruling, 2026-09-06, step C2c): the diagnosis was the
#: erf count, and the fix follows from the fitter's own design ("the
#: prior is read once before the loop") -- `marginal_exact` is a
#: function of `a` ALONE per source (`YsoShape.marginal_exact` never
#: depends on `log10 B`), so it need only be evaluated once per source,
#: not once per (source, template) pair. `_MARGINAL_CURVE_N_LOG`
#: log-spaced points from `1e-3 A_s` to `A_s` itself, then
#: `_MARGINAL_CURVE_N_LIN` MORE points linear from `A_s` to the
#: kernel's own `_MARGINAL_CURVE_SIGMAS`-sigma tail (the same generous
#: bound `_a_cell_mass`/`_extinction_grid` use elsewhere) -- 400 points,
#: `64` erf calls each (32 cells x 2 components) = ~26k erf calls per
#: source, against ~200,000 templates x 64 calls direct: ~500x fewer.
#: Every per-template read is then a linear interpolation on this
#: curve, not a fresh closed-form evaluation.
_MARGINAL_CURVE_N_LOG = 200
_MARGINAL_CURVE_N_LIN = 200
_MARGINAL_CURVE_N = _MARGINAL_CURVE_N_LOG + _MARGINAL_CURVE_N_LIN
_MARGINAL_CURVE_A_FLOOR_FRAC = 1.0e-3
_MARGINAL_CURVE_SIGMAS = 6.0


def _marginal_curve_grid(a_col, a_hi):
    """`(n, _MARGINAL_CURVE_N)`: one source's own `a` grid for the
    extinction-marginal curve (module note above `_MARGINAL_CURVE_N`)."""
    a_floor = _MARGINAL_CURVE_A_FLOOR_FRAC * a_col
    log_part = _quad_grid_log(a_floor, a_col, _MARGINAL_CURVE_N_LOG)
    t = np.linspace(0.0, 1.0, _MARGINAL_CURVE_N_LIN + 1)[1:]
    lin_part = a_col[:, None] + t[None, :] * (a_hi - a_col)[:, None]
    return np.concatenate([log_part, lin_part], axis=1)


def _interp_marginal_curve(grid, curve, a_query):
    """`(n,)`: linear interpolation of `curve[i]` (already gathered to
    one row per query point, `(n, _MARGINAL_CURVE_N)`) at `a_query[i]`
    on `grid[i]` -- held at the nearest edge value outside the grid's
    own range. Plain numpy; NOT called in production any more (owner
    ruling, 2026-09-06, step C2d: `_yso_shape_numba`/`_h2s_shape_numba`
    replaced it once measured at the real fitter block size -- step
    C2c's 36k-point probe made a compiled loop over this arithmetic
    look slower, but numba's own per-call dispatch overhead was
    amortised away at 520k points/call). Kept as the numpy reference
    the compiled kernels were verified against, chunked at
    `_INTERP_CHUNK` so the per-block gather never scales with the
    caller's own batch."""
    n = a_query.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_g = grid.shape[1]
    for start in range(0, n, _INTERP_CHUNK):
        stop = min(start + _INTERP_CHUNK, n)
        g = grid[start:stop]
        c = curve[start:stop]
        a = np.clip(a_query[start:stop], g[:, 0], g[:, -1])
        idx = np.clip(np.sum(a[:, None] >= g, axis=1) - 1, 0, n_g - 2)
        rows_i = np.arange(g.shape[0])
        g_lo, g_hi = g[rows_i, idx], g[rows_i, idx + 1]
        c_lo, c_hi = c[rows_i, idx], c[rows_i, idx + 1]
        t = np.where(g_hi > g_lo, (a - g_lo) / (g_hi - g_lo), 0.0)
        out[start:stop] = c_lo + t * (c_hi - c_lo)
    return out


_INV_SQRT_2PI_NUMBA = 1.0 / math.sqrt(2.0 * math.pi)


@numba.njit(cache=True, fastmath=True, error_model="numpy")
def _yso_shape_numba(grid, curve, a, mean_b, width, b):
    """`(n,)`: YSO's `shape` -- the per-source curve's linear
    interpolation (module note above `_interp_marginal_curve`) times
    the closed-form conditional Gaussian on the ridge -- as ONE
    compiled loop over the `(n, m)` block (owner ruling, 2026-09-06,
    step C2d: measured at the REAL fitter block size, 128 sources x
    4,066 models = 520k points/call, not the 36k-point probe that made
    a compiled bracket-and-blend look slower in step C2c -- numba's own
    per-call dispatch overhead is amortised away at the real scale)."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_g = grid.shape[1]
    for i in range(n):
        ai = a[i]
        if ai <= 0.0:
            out[i] = 0.0
            continue
        ac = ai
        if ac < grid[i, 0]:
            ac = grid[i, 0]
        elif ac > grid[i, n_g - 1]:
            ac = grid[i, n_g - 1]
        lo = 0
        hi = n_g - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if grid[i, mid] <= ac:
                lo = mid
            else:
                hi = mid
        span = grid[i, lo + 1] - grid[i, lo]
        t = (ac - grid[i, lo]) / span if span > 0.0 else 0.0
        p_a = curve[i, lo] + t * (curve[i, lo + 1] - curve[i, lo])

        z = (b[i] - mean_b[i]) / width[i]
        p_b = _INV_SQRT_2PI_NUMBA / width[i] * math.exp(-0.5 * z * z)
        out[i] = p_a * p_b
    return out


@numba.njit(cache=True, fastmath=True, error_model="numpy")
def _h2s_shape_numba(grid, curve, a, log10_sigma, logsig_mean, logsig_std):
    """`(n,)`: H2S's `shape` -- the per-source curve's linear
    interpolation times the region's `log10 Sigma` lognormal -- as ONE
    compiled loop (owner ruling, 2026-09-06, step C2d; measured at the
    real 520k-point/call fitter block size, not the 36k-point probe
    that made a compiled bracket-and-blend look slower in step C2c)."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_g = grid.shape[1]
    for i in range(n):
        ai = a[i]
        if ai <= 0.0:
            out[i] = 0.0
            continue
        ac = ai
        if ac < grid[i, 0]:
            ac = grid[i, 0]
        elif ac > grid[i, n_g - 1]:
            ac = grid[i, n_g - 1]
        lo = 0
        hi = n_g - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if grid[i, mid] <= ac:
                lo = mid
            else:
                hi = mid
        span = grid[i, lo + 1] - grid[i, lo]
        t = (ac - grid[i, lo]) / span if span > 0.0 else 0.0
        p_a = curve[i, lo] + t * (curve[i, lo + 1] - curve[i, lo])

        zsig = (log10_sigma[i] - logsig_mean) / logsig_std
        p_sigma = _INV_SQRT_2PI_NUMBA / logsig_std * math.exp(-0.5 * zsig * zsig)
        out[i] = p_a * p_sigma
    return out


@numba.njit(cache=True, fastmath=True, error_model="numpy")
def _h2s_selection_numba(x_ladder, sigma_grid, eps_batch, x_query, log10_sigma):
    """`(n,)`: H2S's `selection` -- the bilinear read of the stored
    `(x, Sigma)` selection array, one compiled loop (owner ruling,
    2026-09-06, step C2d; same real-block-size measurement as `_h2s_
    shape_numba`)."""
    n = x_query.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        ix, tx = _bracket(x_ladder, x_query[i])
        iv, tv = _bracket(sigma_grid, log10_sigma[i])
        e00 = eps_batch[i, ix, iv]
        e01 = eps_batch[i, ix, iv + 1]
        e10 = eps_batch[i, ix + 1, iv]
        e11 = eps_batch[i, ix + 1, iv + 1]
        e_lo = e00 + tv * (e01 - e00)
        e_hi = e10 + tv * (e11 - e10)
        out[i] = e_lo + tx * (e_hi - e_lo)
    return out


def _read_zp_sigma_k(path, n):
    """`ZP_SIGMA_K` (mag) per source -- the Herschel field zero point's
    own uncertainty, 0 for a Planck-arm source (owner, 2026-09-06) -- if
    the adopted column product has it, else zeros: a region not yet
    rebuilt with the per-field offset runs exactly as before (matches
    `prior.counts_star_family._read_zp_sigma_k`)."""
    with h5py.File(path, "r") as f:
        if "ZP_SIGMA_K" in f:
            return np.asarray(f["ZP_SIGMA_K"][:], dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def _library_reference_flux(config, cls):
    """`(n_model,)`: `_LIBRARY_BAND[cls]` off `_LIBRARY_KEY[cls]`'s own
    register (module docstring's GAL/H2S "a change of units" reads)."""
    path = os.path.join(config.inputs["sed_models"], "registers",
                         "%s_register.hdf5" % _LIBRARY_KEY[cls])
    with h5py.File(path, "r") as f:
        return np.asarray(f["models"][_LIBRARY_BAND[cls]][:], dtype=np.float64)


def _interp_eps_2d(eps_batch, x_ladder, grid, x_query, val_query):
    """`(n,)`: bilinear interpolation of `eps_batch[i]` (`(n, n_x, n_v)`,
    already gathered to one row per query point) at `(x_query[i],
    val_query[i])` on `x_ladder` and `grid` -- both axes held at the
    nearest edge value outside their own range (module docstring's "end
    bins held" convention). Processed in `_INTERP_CHUNK` blocks
    (`CODING_RULES.md` 10a) so the per-block fancy-index gather never
    scales with the caller's own batch. Plain numpy (owner ruling,
    2026-09-06, step C2c): a compiled per-point loop over this same
    arithmetic was measured SLOWER than this vectorised version (0.76
    vs 0.36 us/point) -- bilinear interpolation is simple enough that
    numpy's own vectorised clip/searchsorted/fancy-index already beats
    a numba dispatch per point at realistic batch sizes."""
    n = x_query.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_x, n_v = x_ladder.size, grid.size
    for start in range(0, n, _INTERP_CHUNK):
        stop = min(start + _INTERP_CHUNK, n)
        eps = eps_batch[start:stop]
        x = np.clip(x_query[start:stop], x_ladder[0], x_ladder[-1])
        ix = np.clip(np.searchsorted(x_ladder, x) - 1, 0, n_x - 2)
        x_lo, x_hi = x_ladder[ix], x_ladder[ix + 1]
        tx = np.where(x_hi > x_lo, (x - x_lo) / (x_hi - x_lo), 0.0)

        v = np.clip(val_query[start:stop], grid[0], grid[-1])
        iv = np.clip(np.searchsorted(grid, v) - 1, 0, n_v - 2)
        v_lo, v_hi = grid[iv], grid[iv + 1]
        tv = np.where(v_hi > v_lo, (v - v_lo) / (v_hi - v_lo), 0.0)

        rows_i = np.arange(eps.shape[0])
        e00 = eps[rows_i, ix, iv]
        e01 = eps[rows_i, ix, iv + 1]
        e10 = eps[rows_i, ix + 1, iv]
        e11 = eps[rows_i, ix + 1, iv + 1]
        e_lo = e00 + tv * (e01 - e00)
        e_hi = e10 + tv * (e11 - e10)
        out[start:stop] = e_lo + tx * (e_hi - e_lo)
    return out


_SQRT2PI = float(np.sqrt(2.0 * np.pi))
_HERSCHEL_ARM_CODE = 0


@numba.njit(cache=True, fastmath=True)
def _bracket(grid, x):
    """`(i, t)`: the bracket index and fractional position of `x` on the
    increasing `grid`, clamped at either end -- the compiled kernels' own
    scalar version of `column_grid.bracket`/`np.searchsorted`."""
    n = grid.shape[0]
    xc = x
    if xc < grid[0]:
        xc = grid[0]
    elif xc > grid[n - 1]:
        xc = grid[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] <= xc:
            lo = mid
        else:
            hi = mid
    span = grid[lo + 1] - grid[lo]
    t = (xc - grid[lo]) / span if span > 0.0 else 0.0
    return lo, t


#: The hybrid extinction law's own ramp domain (`selection.LAW_RAMP_LO`/
#: `LAW_RAMP_HI`), inlined as compile-time constants so the compiled GAL
#: kernel needs no Python call back into `selection.law_dense_weight`
#: per point (owner, 2026-09-06: no per-source galaxy product -- the
#: kernel now dims the source's own four IRAC limits itself).
_LAW_RAMP_LO = selection_module.LAW_RAMP_LO
_LAW_RAMP_HI = selection_module.LAW_RAMP_HI


@numba.njit(cache=True, fastmath=True)
def _interp1_scalar(row, axis, x):
    i, t = _bracket(axis, x)
    return row[i] + t * (row[i + 1] - row[i])


@numba.njit(cache=True, fastmath=True)
def _interp2_scalar(table, axis_a, axis_b, xa, xb):
    ia, ta = _bracket(axis_a, xa)
    ib, tb = _bracket(axis_b, xb)
    v_lo = table[ia, ib] + tb * (table[ia, ib + 1] - table[ia, ib])
    v_hi = table[ia + 1, ib] + tb * (table[ia + 1, ib + 1] - table[ia + 1, ib])
    return v_lo + ta * (v_hi - v_lo)


@numba.njit(cache=True, fastmath=True)
def _interp3_scalar(table, axis_a, axis_b, axis_c, xa, xb, xc):
    ia, ta = _bracket(axis_a, xa)
    ib, tb = _bracket(axis_b, xb)
    ic, tc = _bracket(axis_c, xc)
    v00 = table[ia, ib, ic] + tc * (table[ia, ib, ic + 1] - table[ia, ib, ic])
    v01 = table[ia, ib + 1, ic] + tc * (table[ia, ib + 1, ic + 1] - table[ia, ib + 1, ic])
    v10 = table[ia + 1, ib, ic] + tc * (table[ia + 1, ib, ic + 1] - table[ia + 1, ib, ic])
    v11 = table[ia + 1, ib + 1, ic] + tc * (table[ia + 1, ib + 1, ic + 1] - table[ia + 1, ib + 1, ic])
    v_lo = v00 + tb * (v01 - v00)
    v_hi = v10 + tb * (v11 - v10)
    return v_lo + ta * (v_hi - v_lo)


@numba.njit(cache=True, fastmath=True)
def _gal_eps_at_sgrid(j, lim1, lim2, lim3, lim4, dim1, dim2, dim3, dim4, log10_s_grid,
                       g1, g3, g4, joint, pair13, pair14, pair34, marg1, marg3, marg4):
    """One S-grid point's exact two-of-four pass fraction, the scalar
    version of `prior.gal.source_selection_from_cdf`'s own inner loop:
    the orthant probability of the survey-wide colour-CDF tables at this
    source's own four thresholds, no galaxy touched."""
    s = log10_s_grid[j]
    tau1 = lim1 - s + dim1
    tau2 = lim2 - s + dim2
    tau3 = lim3 - s + dim3
    tau4 = lim4 - s + dim4

    f3d = _interp3_scalar(joint[j], g1, g3, g4, tau1, tau3, tau4)
    at_least1 = 1.0 - f3d

    f1 = _interp1_scalar(marg1[j], g1, tau1)
    f3 = _interp1_scalar(marg3[j], g3, tau3)
    f4 = _interp1_scalar(marg4[j], g4, tau4)
    f13 = _interp2_scalar(pair13[j], g1, g3, tau1, tau3)
    f14 = _interp2_scalar(pair14[j], g1, g4, tau1, tau4)
    f34 = _interp2_scalar(pair34[j], g3, g4, tau3, tau4)
    pair_ab = 1.0 - f1 - f3 + f13
    pair_ac = 1.0 - f1 - f4 + f14
    pair_bc = 1.0 - f3 - f4 + f34
    triple = 1.0 - f1 - f3 - f4 + f13 + f14 + f34 - f3d
    at_least2 = pair_ab + pair_ac + pair_bc - 2.0 * triple

    eps = at_least1 if tau2 <= 0.0 else at_least2
    if eps < 0.0:
        eps = 0.0
    elif eps > 1.0:
        eps = 1.0
    return eps


@numba.njit(cache=True, fastmath=True)
def _gal_shape_numba(a, b, mi, a_col, sigma_col, arm_idx, zp_sigma_k, gal_fref,
                     log10_s_grid, phi_density_grid,
                     kernel_ln_nodes, kernel_w, kernel_mu, kernel_sigma):
    """`(n,)`: GAL's SHAPE half, `p_a . p(log10 S, a)`'s first factor --
    `Kernel.pdf`'s own node bracket and two-component sum at this
    SOURCE's own arm (`arm_idx`, fixed defect, owner 2026-09-06: was the
    sightline's block-averaged majority) times the counts law's
    normalised density at `log10 S = log10 B + log10 f_ref`. Zero at `a
    <= 0` (the class's own support). The same arithmetic the fused
    reader always did, split out so the class's `shape`/`selection` pair
    (owner ruling, 2026-09-06) can be read, and timed, apart."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    ln10 = np.log(10.0)
    for k in range(n):
        ak = a[k]
        if ak <= 0.0:
            out[k] = 0.0
            continue
        acol = a_col[k]
        scol = sigma_col[k]
        arm = arm_idx[k]

        i, t = _bracket(kernel_ln_nodes, np.log(acol))
        w = kernel_w[arm, i] + t * (kernel_w[arm, i + 1] - kernel_w[arm, i])
        mu0 = kernel_mu[arm, i, 0] + t * (kernel_mu[arm, i + 1, 0] - kernel_mu[arm, i, 0])
        mu1 = kernel_mu[arm, i, 1] + t * (kernel_mu[arm, i + 1, 1] - kernel_mu[arm, i, 1])
        sg0 = kernel_sigma[arm, i, 0] + t * (kernel_sigma[arm, i + 1, 0] - kernel_sigma[arm, i, 0])
        sg1 = kernel_sigma[arm, i, 1] + t * (kernel_sigma[arm, i + 1, 1] - kernel_sigma[arm, i, 1])

        sigma_col_dex = scol / (acol * ln10)
        zp_dex = zp_sigma_k[k] / (acol * ln10) if arm == _HERSCHEL_ARM_CODE else 0.0
        extra_var = sigma_col_dex * sigma_col_dex + zp_dex * zp_dex
        sigma0 = np.sqrt(sg0 * sg0 + extra_var)
        sigma1 = np.sqrt(sg1 * sg1 + extra_var)

        log10t = np.log10(ak)
        z0 = (log10t - (np.log10(acol) + mu0)) / sigma0
        z1 = (log10t - (np.log10(acol) + mu1)) / sigma1
        dens = (w * np.exp(-0.5 * z0 * z0) / (sigma0 * _SQRT2PI)
               + (1.0 - w) * np.exp(-0.5 * z1 * z1) / (sigma1 * _SQRT2PI))
        p_a = dens / (ak * ln10)

        log10_s = b[k] + np.log10(gal_fref[mi[k]])
        iv, tv = _bracket(log10_s_grid, log10_s)
        pd_lo = phi_density_grid[iv]
        pd_hi = phi_density_grid[iv + 1]
        phi_density = pd_lo + tv * (pd_hi - pd_lo)

        out[k] = p_a * phi_density
    return out


@numba.njit(cache=True, fastmath=True)
def _gal_selection_numba(a, b, mi, gal_fref, log10_lim_irac, kd_irac, kw_irac,
                         log10_s_grid,
                         cdf_g1, cdf_g3, cdf_g4, cdf_joint, cdf_pair13, cdf_pair14, cdf_pair34,
                         cdf_marg1, cdf_marg3, cdf_marg4):
    """`(n,)`: GAL's SELECTION half -- the hybrid law's dimming at this
    query's own `a` (`selection.law_dense_weight`/`kappa_hybrid`,
    inlined) and the exact two-of-four colour-CDF lookup
    (`_gal_eps_at_sgrid`) at this SOURCE's own four IRAC limits
    (`log10_lim_irac`, already resident on the region table -- there is
    no per-source galaxy product, owner 2026-09-06). Zero at `a <= 0`;
    the value there is discarded by the composition's zero `shape`
    anyway. The same arithmetic the fused reader always did."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    ln_ramp_ratio = np.log(_LAW_RAMP_HI / _LAW_RAMP_LO)
    for k in range(n):
        ak = a[k]
        if ak <= 0.0:
            out[k] = 0.0
            continue
        # the hybrid law's dimming at this query's own column `ak`
        # (`selection.law_dense_weight`/`kappa_hybrid`, module docstring).
        xw = np.log(ak / _LAW_RAMP_LO) / ln_ramp_ratio
        if xw < 0.0:
            xw = 0.0
        elif xw > 1.0:
            xw = 1.0
        wd = xw * xw * (3.0 - 2.0 * xw)
        dim1 = 0.4 * ak * ((1.0 - wd) * kd_irac[0] + wd * kw_irac[0])
        dim2 = 0.4 * ak * ((1.0 - wd) * kd_irac[1] + wd * kw_irac[1])
        dim3 = 0.4 * ak * ((1.0 - wd) * kd_irac[2] + wd * kw_irac[2])
        dim4 = 0.4 * ak * ((1.0 - wd) * kd_irac[3] + wd * kw_irac[3])

        log10_s = b[k] + np.log10(gal_fref[mi[k]])
        iv, tv = _bracket(log10_s_grid, log10_s)

        lim1 = log10_lim_irac[k, 0]
        lim2 = log10_lim_irac[k, 1]
        lim3 = log10_lim_irac[k, 2]
        lim4 = log10_lim_irac[k, 3]
        e_lo = _gal_eps_at_sgrid(iv, lim1, lim2, lim3, lim4, dim1, dim2, dim3, dim4, log10_s_grid,
                                 cdf_g1, cdf_g3, cdf_g4, cdf_joint, cdf_pair13, cdf_pair14, cdf_pair34,
                                 cdf_marg1, cdf_marg3, cdf_marg4)
        e_hi = _gal_eps_at_sgrid(iv + 1, lim1, lim2, lim3, lim4, dim1, dim2, dim3, dim4, log10_s_grid,
                                 cdf_g1, cdf_g3, cdf_g4, cdf_joint, cdf_pair13, cdf_pair14, cdf_pair34,
                                 cdf_marg1, cdf_marg3, cdf_marg4)
        out[k] = e_lo + tv * (e_hi - e_lo)
    return out


class _FamilyClass(object):
    """STAR/AGB/PAHC's `shape`/`selection` pair (owner ruling, 2026-09-06):
    `shape` is `ClassShape.density` converted to a density in `(a, log10
    B)` by the fixed cell area and the `d(log10 x)/da` Jacobian;
    `selection` is this source's own tabulated `EPS_<cls>`, bilinear on
    `(x = a/A_s, log10 B)`. Both take the SAME `(rows, a, log10_b)`
    block every class's pair takes; `Z_%s` is read by the shared
    composition, not here."""

    def __init__(self, prior, cls):
        self._prior = prior
        self.cls = cls
        self.z_column = "Z_%s" % cls.upper()

    def shape(self, rows, a, log10_b, model_index=None):
        p = self._prior
        shape_obj = p.shapes[self.cls]
        fg = p.family_grids[self.cls]
        shp = a.shape
        rows_f, a_f, b_f = rows.ravel(), a.ravel(), log10_b.ravel()
        valid = a_f > 0.0
        a_safe = np.where(valid, a_f, 1.0)

        tile_id = p.table["TILE_ID"][rows_f]
        a_col = p.table["A_COL_K"][rows_f]
        sigma_col = p.table["A_COL_SIG_K"][rows_f]
        map_class = np.where(
            p.table["A_COL_PROVENANCE"][rows_f] == star_shapes._PLANCK_PROVENANCE_CODE,
            "planck", "herschel")
        f_lim8 = (p.table["F_LIM_50_MJY"][rows_f, p._idx_i4] if self.cls == "pahc" else None)
        zp_sigma_k = p._zp_sigma_k[rows_f]

        mass = shape_obj.density(a_f, b_f, tile_id, a_col, sigma_col, map_class, f_lim8=f_lim8,
                                 zp_sigma_k=zp_sigma_k)
        p_ab = np.where(valid, mass / (a_safe * LN10 * fg["dx"] * fg["db"]), 0.0)
        return p_ab.reshape(shp)

    def selection(self, rows, a, log10_b, model_index=None):
        p = self._prior
        fg = p.family_grids[self.cls]
        shp = a.shape
        rows_f, a_f, b_f = rows.ravel(), a.ravel(), log10_b.ravel()
        valid = a_f > 0.0
        a_safe = np.where(valid, a_f, 1.0)
        a_col = p.table["A_COL_K"][rows_f]
        x_query = np.where(valid, a_safe / a_col, 0.0)

        local = p._prep_local_index(rows_f)
        eps_batch = p._prep_star_eps[self.cls][local]
        eps_s = _interp_eps_2d(eps_batch, fg["x_ladder"], fg["b_grid"], x_query, b_f)
        return eps_s.reshape(shp)


class _GalClass(object):
    """GAL's `shape`/`selection` pair: `shape` is `Kernel.pdf`'s
    two-component mixture times the counts law's own normalised density
    at `log10 S = log10 B + log10 f_ref`; `selection` is the colour-CDF
    lookup at this source's own four IRAC limits (already compiled --
    owner, 2026-09-06: no per-source galaxy product any more). `Z_GAL`
    is read by the shared composition."""

    z_column = "Z_GAL"

    def __init__(self, prior):
        self._prior = prior

    def shape(self, rows, a, log10_b, model_index=None):
        p = self._prior
        shp = a.shape
        rows_f = rows.ravel()
        a_f, b_f = a.ravel(), log10_b.ravel()
        mi_f = model_index.ravel().astype(np.int64)
        a_col = p.table["A_COL_K"][rows_f]
        sigma_col = p.table["A_COL_SIG_K"][rows_f]
        arm_idx = p.table["A_COL_PROVENANCE"][rows_f].astype(np.int64)
        zp_sigma_k = p._zp_sigma_k[rows_f]
        kern = p.yso_shape.kernel
        out = _gal_shape_numba(a_f, b_f, mi_f, a_col, sigma_col, arm_idx, zp_sigma_k,
                               p.gal_fref, p.gal_log10_s_grid, p._gal_phi_density_grid,
                               kern._ln_nodes, kern._w, kern._mu, kern._sigma)
        return out.reshape(shp)

    def selection(self, rows, a, log10_b, model_index=None):
        p = self._prior
        shp = a.shape
        rows_f = rows.ravel()
        a_f, b_f = a.ravel(), log10_b.ravel()
        mi_f = model_index.ravel().astype(np.int64)
        log10_lim_irac = np.ascontiguousarray(
            np.log10(p.table["F_LIM_50_MJY"][rows_f][:, p._gal_irac_idx]))
        cdf = p._gal_cdf
        out = _gal_selection_numba(a_f, b_f, mi_f, p.gal_fref, log10_lim_irac,
                                   p.gal_kd_irac, p.gal_kw_irac, p.gal_log10_s_grid,
                                   cdf["g1"], cdf["g3"], cdf["g4"], cdf["joint"],
                                   cdf["pair13"], cdf["pair14"], cdf["pair34"],
                                   cdf["marg1"], cdf["marg3"], cdf["marg4"])
        return out.reshape(shp)


class _YsoClass(object):
    """YSO's `shape`/`selection` pair: `shape` is `YsoShape.
    marginal_exact` (this source's own exact extinction marginal) times
    the closed-form conditional Gaussian on the ridge; `selection` is
    ONE everywhere -- the YSO shape carries no selection at all, `Z_YSO
    = 1` by construction (module docstring, disclosed asymmetry, owner
    2026-09-06's single question)."""

    z_column = None

    def __init__(self, prior):
        self._prior = prior

    def shape(self, rows, a, log10_b, model_index=None):
        p = self._prior
        shp = a.shape
        rows_f, a_f, b_f = rows.ravel(), a.ravel(), log10_b.ravel()

        local = p._prep_local_index(rows_f)
        grid_l = np.ascontiguousarray(p._prep_marginal_grid[local])
        curve_l = np.ascontiguousarray(p._prep_marginal_curve[local])
        mean_b = p.table["RIDGE_INTERCEPT"][rows_f] + p.table["RIDGE_SLOPE"][rows_f] * a_f
        width = p.table["RIDGE_WIDTH"][rows_f]
        out = _yso_shape_numba(grid_l, curve_l, a_f, mean_b, width, b_f)
        return out.reshape(shp)

    def selection(self, rows, a, log10_b, model_index=None):
        return np.ones_like(a, dtype=np.float64)


class _H2sClass(object):
    """H2S's `shape`/`selection` pair: `shape` is `YsoShape.
    marginal_exact` (shared with YSO, not copied) times the region's own
    `log10 Sigma` lognormal; `selection` is this source's own tabulated
    `EPS`, bilinear on `(x = a/A_s, log10 Sigma)`. `Z_H2S` is read by the
    shared composition."""

    z_column = "Z_H2S"

    def __init__(self, prior):
        self._prior = prior

    def shape(self, rows, a, log10_b, model_index=None):
        p = self._prior
        shp = a.shape
        rows_f, a_f, b_f = rows.ravel(), a.ravel(), log10_b.ravel()

        local = p._prep_local_index(rows_f)
        grid_l = np.ascontiguousarray(p._prep_marginal_grid[local])
        curve_l = np.ascontiguousarray(p._prep_marginal_curve[local])
        mi_f = model_index.ravel()
        log10_sigma = b_f + np.log10(p.h2s_fref[mi_f])
        out = _h2s_shape_numba(grid_l, curve_l, a_f, log10_sigma,
                              p.h2s_logsig_mean, p.h2s_logsig_std)
        return out.reshape(shp)

    def selection(self, rows, a, log10_b, model_index=None):
        p = self._prior
        shp = a.shape
        rows_f, a_f, b_f = rows.ravel(), a.ravel(), log10_b.ravel()
        mi_f = model_index.ravel()
        valid = a_f > 0.0
        a_safe = np.where(valid, a_f, 1.0)

        a_col = p.table["A_COL_K"][rows_f]
        log10_sigma = b_f + np.log10(p.h2s_fref[mi_f])
        x_query = np.where(valid, a_safe / a_col, 0.0)

        local = p._prep_local_index(rows_f)
        eps_batch = np.ascontiguousarray(p._prep_h2s_eps[local])
        out = _h2s_selection_numba(p.h2s_x_ladder, p.h2s_log10_sigma_grid,
                                   eps_batch, x_query, log10_sigma)
        return out.reshape(shp)


#: `Z`'s own quadrature -- GAL ONLY (owner ruling, 2026-09-06, step C2d,
#: narrowed twice: step C1 tried the kernel's exact cell mass for every
#: tabulated class and step C2a moved all six into this routine; both
#: withdrawn except for GAL). `Z` is the normaliser of the density the
#: fitter actually reads. GAL's shape factors PERFECTLY into `kernel_
#: pdf(a) . phi_density(b)`, so the kernel's own EXACT per-cell mass
#: (`Kernel.cdf` differences, `_a_cell_mass`) on `_Z_QUAD_NA` log-spaced
#: cells carries the whole `a` integral exactly, and the measured
#: identity is excellent (~0.05-0.1%) -- worth the live recompute. Every
#: other class's `Z` is simpler and cheaper read straight off the
#: table, computed once upstream by the stage that already owns the
#: dot product: STAR/AGB/PAHC from `prior.counts_star_family.
#: family_counts` (reconstruction error against the bicubic read
#: ~1.5%, inside the 2% bar); H2S from `prior.counts_cloud`'s own exact
#: marginal-ladder-cell-mass dot product (`Z_H2S = EPS_H2S`); YSO is 1
#: by construction. `log10 B` on GAL's own STORED grid (`gal_log10_s_
#: grid`, not resampled), since that is where the selection read is
#: itself exact only at those points.
_Z_QUAD_NA = 48
_Z_QUAD_SIGMAS = 6.0
_Z_QUAD_A_FLOOR_FRAC = 1.0e-3


def _quad_grid_log(lo, hi, num):
    """`(n, num)`: log-spaced grid per source between per-source `lo`
    and `hi` (both `(n,)`, `hi` floored at ten times `lo` so a
    degenerate source still gets a real span)."""
    t = np.linspace(0.0, 1.0, num)
    log_lo = np.log10(lo)
    log_hi = np.log10(np.maximum(hi, lo * 10.0))
    return 10.0 ** (log_lo[:, None] + t[None, :] * (log_hi - log_lo)[:, None])


def _a_cell_mass(kernel_obj, a_col, sigma_col, map_class_code, a_floor, a_hi, n_cells):
    """`(centers, mass, pdf_center)`, each `(n, n_cells)`: `n_cells`
    log-spaced cells in `a` from `a_floor` to `a_hi`, `mass` the
    kernel mixture's EXACT probability in each cell (`Kernel.cdf` at the
    cell edges, differenced -- the mixture's own closed form, not a
    sampled density), `centers` each cell's geometric-mean point and
    `pdf_center` the kernel's OWN density there (`Kernel.pdf`) -- the
    reference a caller divides its own `shape` value by before
    multiplying by `mass`, so the kernel's exact per-cell probability
    carries the `a` integral and only the REMAINING factor (which the
    kernel does not already model) is sampled at the cell centre."""
    edges = _quad_grid_log(a_floor, a_hi, n_cells + 1)
    centers = np.sqrt(edges[:, :-1] * edges[:, 1:])
    cdf_edges = kernel_obj.cdf(edges, a_col, sigma_col, map_class_code)
    mass = np.maximum(cdf_edges[:, 1:] - cdf_edges[:, :-1], 0.0)
    pdf_center = kernel_obj.pdf(centers, a_col, sigma_col, map_class_code)
    return centers, mass, pdf_center


#: `_z_by_quadrature`'s own source-batch chunk (`CODING_RULES.md` 10a).
#: Unchunked, a batch's own working set through the class's `shape`/
#: `selection` read (STAR/AGB/PAHC's bicubic spline fit and evaluate,
#: called once per unique (tile, node[, limit]) group but still holding
#: `n_batch . _Z_QUAD_NA . nb` points' worth of query and intermediate
#: arrays at once) scales with the WHOLE prepared batch, not the query
#: axis alone -- the STAR/AGB blowup this fixes (measured: STAR peaked
#: 7.7 GB, AGB killed at 8-10 GB, for one 2,643-source NGC 7129 batch;
#: was 3.9 GB before this stage's `Z` quadrature existed). Bounded here
#: at `_Z_QUAD_CHUNK_POINTS` = `_Z_QUAD_MEM_BUDGET_BYTES / (_Z_QUAD_
#: ARRAYS_ALIVE * 8)` points (source chunk size x `_Z_QUAD_NA` x `nb`)
#: per call into the read -- `_Z_QUAD_ARRAYS_ALIVE` is a round number
#: calibrated to the measured STAR blowup (~950 bytes/point observed;
#: 128 float64-array-equivalents/point, with margin for AGB, is the
#: nearest round cover), the same "budget / bytes-per-point" discipline
#: `_INTERP_CHUNK`/`_MARGINAL_CHUNK` already use elsewhere in this file.
_Z_QUAD_MEM_BUDGET_BYTES = 512 * 1024 * 1024
_Z_QUAD_ARRAYS_ALIVE = 128
_Z_QUAD_CHUNK_POINTS = _Z_QUAD_MEM_BUDGET_BYTES // (_Z_QUAD_ARRAYS_ALIVE * 8)


def _z_by_quadrature(prior, cls, rows, stage=None):
    """`(n,)`: `Z[s]` for GAL only (module note above `_Z_QUAD_NA`;
    every other class's caller reads `Z` off the table instead -- see
    `prior.table._recompute_z_columns`), chunked over SOURCES at
    `_Z_QUAD_CHUNK_POINTS` points/chunk (module note above `_Z_QUAD_
    MEM_BUDGET_BYTES`) so the read's own working set stays bounded
    regardless of the batch's own size. Reads `prior`'s own `shape`/
    `selection` objects -- the exact arithmetic `log_density` composes
    -- so a source whose selection is genuinely zero everywhere (no
    galaxy could be catalogued at its own limits) returns `Z = 0`, the
    correct statement that the class is impossible on this sightline,
    not a defect (owner ruling, 2026-09-06). `stage`, a `progress.Stage`
    the caller already opened, gets one `.tick()` per source chunk;
    `None` (the default) prints nothing."""
    assert cls == "gal", "_z_by_quadrature: GAL only (owner ruling, 2026-09-06, step C2d)"
    n = rows.shape[0]
    nb = prior.gal_log10_s_grid.size
    chunk_n = max(1, _Z_QUAD_CHUNK_POINTS // (_Z_QUAD_NA * nb))
    if n > chunk_n:
        out = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk_n):
            stop = min(start + chunk_n, n)
            out[start:stop] = _z_by_quadrature_chunk(prior, rows[start:stop])
            if stage is not None:
                stage.tick(stop, n, "sources")
        return out
    out = _z_by_quadrature_chunk(prior, rows)
    if stage is not None:
        stage.tick(n, n, "sources")
    return out


def _z_by_quadrature_chunk(prior, rows):
    """`(n,)`: one chunk's worth of GAL's own exact-cell-mass `Z`
    quadrature -- see `_z_by_quadrature` for the method."""
    n = rows.shape[0]
    obj = prior._classes["gal"]
    a_col = prior.table["A_COL_K"][rows]
    sigma_col = prior.table["A_COL_SIG_K"][rows]
    map_class_code = prior.table["A_COL_PROVENANCE"][rows]
    a_floor = _Z_QUAD_A_FLOOR_FRAC * a_col

    kernel_obj = prior.yso_shape.kernel
    mu, sigma = kernel_obj.params(a_col, sigma_col, map_class_code)
    a_hi = a_col * 10.0 ** (mu + _Z_QUAD_SIGMAS * sigma)
    b_row = prior.gal_log10_s_grid - np.log10(prior.gal_fref[0])  # the STORED grid
    nb = b_row.size
    mi_grid = np.zeros((n, _Z_QUAD_NA * nb), dtype=np.intp)
    b_grid = np.broadcast_to(b_row, (n, nb))

    # the kernel's exact per-cell mass (module note above `_Z_QUAD_NA`)
    # -- shape factors perfectly into kernel_pdf(a) . phi_density(b), so
    # the kernel's own cell mass carries the `a` integral exactly and
    # only the remaining factor is sampled at the cell centre.
    a_grid, mass, pdf_center = _a_cell_mass(
        kernel_obj, a_col, sigma_col, map_class_code, a_floor, a_hi, _Z_QUAD_NA)

    a_full = np.repeat(a_grid, nb, axis=1)
    b_full = np.tile(b_grid, (1, _Z_QUAD_NA))
    rows2d = np.broadcast_to(rows[:, None], a_full.shape)

    shape_val = obj.shape(rows2d, a_full, b_full, model_index=mi_grid)
    sel_val = obj.selection(rows2d, a_full, b_full, model_index=mi_grid)
    dens = (shape_val * sel_val).reshape(n, _Z_QUAD_NA, nb)

    b_grid_3d = np.broadcast_to(b_grid[:, None, :], dens.shape)
    inner = np.trapz(dens, x=b_grid_3d, axis=2)  # (n, NA): trapz over log10 B only

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pdf_center > 0.0, inner / pdf_center, 0.0)
    return np.sum(mass * ratio, axis=1)


class SourcePrior(object):
    """`log_density(cls, rows, a, log10_b, model_index=None)`, one region's
    upstream products loaded once (module docstring). `cls=None` (the
    default) loads all six classes' materials, for the all-class
    identity check; a fitter worker that only ever fits one class passes
    that class's name and this loads ONLY that class's own shape/kernel/
    selection/table materials -- the other five classes then raise if
    called. GAL and H2S/YSO all share the one sightline shape
    (`YsoShape`): GAL for its column kernel and per-sightline arm,
    YSO/H2S for `marginal_exact` itself."""

    def __init__(self, config, region, cls=None):
        if cls is not None and cls not in CLASSES:
            raise ValueError(
                "SourcePrior: unknown class %r, must be one of %r or None" % (cls, CLASSES))
        self.config = config
        self.region = region
        self.cls = cls
        self.table = table_module.read(config, region)
        self.n_source = self.table["A_COL_K"].shape[0]
        self._idx_i4 = BAND_KEYS.index("I4")
        # `ZP_SIGMA_K` is read from the adopted column product itself, not
        # `self.table` (`prior.table` does not carry it) -- the Herschel
        # field zero point's own per-source uncertainty (owner,
        # 2026-09-06), 0 for a Planck-arm source.
        adopted_path = config_module.product_path(
            config, "sky/derived", "adopted", "column", "source", region=region)
        self._zp_sigma_k = _read_zp_sigma_k(adopted_path, self.n_source)

        family_wanted = FAMILY_CLASSES if cls is None else (
            (cls,) if cls in FAMILY_CLASSES else ())
        gal_wanted = cls is None or cls == "gal"
        h2s_wanted = cls is None or cls == "h2s"
        yso_shape_wanted = cls is None or cls in ("gal", "yso", "h2s")

        # -- STAR, AGB, PAHC: the tile shape (density only) and the
        # per-source selection product's own grids (its `EPS` arrays are
        # read per batch, `prepare`) -- only the classes this instance
        # was asked for.
        self.shapes = {}
        self.family_grids = {}
        self._star_selection_path = None
        if family_wanted:
            self._star_selection_path = config_module.product_path(
                config, "bms", "star", "selection", "source", region=region)
            with h5py.File(self._star_selection_path, "r") as f:
                x_ladder = f["X_LADDER"][:].astype(np.float64)
                for c in family_wanted:
                    shape = star_shapes.read(config, region, c)
                    dx = float(np.mean(np.diff(shape.x_edges)))
                    db = float(np.mean(np.diff(shape.b_edges)))
                    self.shapes[c] = shape
                    self.family_grids[c] = dict(
                        dx=dx, db=db, x_ladder=x_ladder,
                        b_grid=f["LOG10_B_GRID_%s" % c.upper()][:].astype(np.float64))

        # -- GAL: the survey-wide counts law and the survey-wide colour-CDF
        # tables (`prior.gal.read_cdf_tables`) -- there is no per-source
        # galaxy product (owner, 2026-09-06); the selection is looked up
        # per query point, inside the compiled kernel, from this source's
        # own four IRAC limits instead.
        self.gal_log10_s_grid = self.gal_fref = None
        self.gal_phi_total = self._gal_phi_density_grid = None
        self._gal_cdf = None
        self.gal_kd_irac = self.gal_kw_irac = None
        self._gal_irac_idx = None
        if gal_wanted:
            counts_path = config_module.product_path(config, "bms", "gal", "counts", "survey")
            with h5py.File(counts_path, "r") as f:
                self.gal_log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
                gal_phi_s = f["PHI_S"][:].astype(np.float64)
            self._gal_cdf = gal_module.read_cdf_tables(counts_path)
            self.gal_fref = _library_reference_flux(config, "gal")

            # the hybrid law's two fixed curves at the four IRAC bands
            # only (module docstring): the compiled kernel blends them by
            # each query's own ramp weight, no per-source table.
            self.gal_kd_irac = selection_module.kappa_ak(
                config, selection_module.LAW_DIFFUSE)[gal_module.IRAC_BAND_IDX]
            self.gal_kw_irac = selection_module.kappa_ak(
                config, selection_module.LAW_DENSE)[gal_module.IRAC_BAND_IDX]
            self._gal_irac_idx = gal_module.IRAC_BAND_IDX

            # `phi(S).S` is only PROPORTIONAL to a density (`SPEC_PRIORS.md`
            # section 5.2's own "prop"); `gal_phi_total` (a survey-wide
            # constant) makes it one. The normaliser `Z_GAL` itself is read
            # from the table (`Z_GAL`, `counts_star_family.gal_counts`'s own
            # full per-source integral of the counts law against this
            # source's own selection over ALL of extinction and flux, fixed
            # defect: this used to be recomputed here at the nominal column
            # `x = 1` alone).
            self.gal_phi_total = float(np.trapz(
                gal_phi_s * (10.0 ** self.gal_log10_s_grid) * LN10, self.gal_log10_s_grid))
            self._gal_phi_density_grid = (
                gal_phi_s * (10.0 ** self.gal_log10_s_grid) * LN10) / self.gal_phi_total

        # -- YSO/H2S/GAL: the shared sightline shape (`marginal_exact`
        # for YSO/H2S; the column kernel and per-sightline arm for GAL).
        self.yso_shape = yso_module.YsoShape.read(config, region) if yso_shape_wanted else None

        # -- H2S: the region's Sigma lognormal and the shared Sigma grid.
        # There is no per-source H2S selection product any more (owner,
        # 2026-09-06, the cleanup unit's own change): `prepare` calls
        # `h2s.source_selection` on the fly instead, on the SAME shared
        # ladder `h2s.X_LADDER` every source uses (not a per-region file).
        self.h2s_x_ladder = self.h2s_log10_sigma_grid = None
        self.h2s_logsig_mean = self.h2s_logsig_std = self.h2s_fref = None
        if h2s_wanted:
            self.h2s_x_ladder = np.asarray(h2s_module.X_LADDER, dtype=np.float64)
            h2s_region_path = config_module.product_path(
                config, "bms", "h2s", "prior", "region", region=region)
            with h5py.File(h2s_region_path, "r") as f:
                self.h2s_logsig_mean = float(f["LOGSIG_MEAN"][()])
                self.h2s_logsig_std = float(f["LOGSIG_STD"][()])
                self.h2s_log10_sigma_grid = np.asarray(f["LOG10_SIGMA_GRID"][:], dtype=np.float64)
            self.h2s_fref = _library_reference_flux(config, "h2s")

        # -- per-batch tabulation (`prepare`, module docstring): unset
        # until a batch is prepared; `log_density` refuses every class
        # until then (rule 6: fail on the impossible).
        self._prep_rows = None
        self._prep_star_eps = None
        self._prep_h2s_eps = None
        self._prep_marginal_grid = None
        self._prep_marginal_curve = None

        # -- one object per class, the shape/selection protocol (owner
        # ruling, 2026-09-06): built only for the classes this instance
        # was scoped to load, same as `self.shapes` etc above.
        self._classes = {}
        for c in family_wanted:
            self._classes[c] = _FamilyClass(self, c)
        if gal_wanted:
            self._classes["gal"] = _GalClass(self)
        if cls is None or cls == "yso":
            self._classes["yso"] = _YsoClass(self)
        if h2s_wanted:
            self._classes["h2s"] = _H2sClass(self)

    def prepare(self, rows):
        """Gathers this batch's own rows (about ten thousand sources,
        `CODING_RULES.md` 10b): STAR family's `EPS_<cls>` from its own
        per-source product (float16 -> float32, still a file); H2S's own
        `(n, n_x, n_sigma)` selection is an ON-THE-FLY call to `h2s.
        source_selection` instead (owner, 2026-09-06: no per-source H2S
        product any more, same change `prior.gal` already made) -- this
        batch's own eight limits (`F_LIM_50_MJY`, already resident on
        the table) and query extinctions (`h2s_x_ladder . A_s`). GAL
        needs no gather at all: its compiled kernel reads the source's
        own four IRAC limits straight off the region table.

        Also builds YSO/H2S's own extinction-marginal CURVE, once per
        source (module note above `_MARGINAL_CURVE_N`, owner ruling,
        2026-09-06, step C2c): `marginal_exact` depends on `a` alone, so
        it is evaluated once per source on a fixed 400-point grid here,
        and every per-template `log_density` read (`_YsoClass`/
        `_H2sClass`) is a linear interpolation on it instead of a fresh
        closed-form evaluation -- ~500x fewer erf calls than evaluating
        directly at every template."""
        rows = np.asarray(rows, dtype=np.intp)
        uniq_rows = np.unique(rows)

        star_eps = {}
        if self._star_selection_path is not None:
            with h5py.File(self._star_selection_path, "r") as f:
                for c in self.shapes:
                    star_eps[c] = f["EPS_%s" % c.upper()][uniq_rows, :, :].astype(np.float32)
        h2s_eps = None
        if self.h2s_x_ladder is not None:
            a_col_batch = self.table["A_COL_K"][uniq_rows]
            a_query = self.h2s_x_ladder[None, :] * a_col_batch[:, None]
            log10_lim_8 = np.log10(self.table["F_LIM_50_MJY"][uniq_rows])
            h2s_eps = h2s_module.source_selection(
                self.config, self.region, log10_lim_8, a_query).astype(np.float32)

        marginal_grid = marginal_curve = None
        if "yso" in self._classes or "h2s" in self._classes:
            a_col_u = self.table["A_COL_K"][uniq_rows]
            sigma_col_u = self.table["A_COL_SIG_K"][uniq_rows]
            sl_rows_u = self.table["HPX256_ROW"][uniq_rows]
            map_class_u = self.table["A_COL_PROVENANCE"][uniq_rows]
            zp_u = self._zp_sigma_k[uniq_rows]
            mu_b, sigma_b = self.yso_shape.kernel.params(
                a_col_u, sigma_col_u, map_class_u, zp_sigma_k=zp_u)
            a_hi_u = a_col_u * 10.0 ** (mu_b + _MARGINAL_CURVE_SIGMAS * sigma_b)
            marginal_grid = _marginal_curve_grid(a_col_u, a_hi_u)
            n_u = uniq_rows.size

            def _rep(arr):
                return np.repeat(arr, _MARGINAL_CURVE_N)

            curve_flat = _marginal_exact_compiled(
                self.yso_shape, marginal_grid.ravel(), _rep(sl_rows_u), _rep(a_col_u),
                _rep(sigma_col_u), _rep(map_class_u), zp_sigma_k=_rep(zp_u))
            marginal_curve = curve_flat.reshape(n_u, _MARGINAL_CURVE_N)

        self._prep_rows = uniq_rows
        self._prep_star_eps = star_eps
        self._prep_h2s_eps = h2s_eps
        self._prep_marginal_grid = marginal_grid
        self._prep_marginal_curve = marginal_curve

    def _prep_local_index(self, rows):
        """`(n,)`: `rows`'s own position in the last `prepare`d batch --
        every class's one gather into that batch's `EPS` arrays (rule 6:
        raise, do not silently recompute, when a row was never
        prepared)."""
        if self._prep_rows is None:
            raise ValueError(
                "SourcePrior.log_density: call prepare(rows) once per batch first")
        loc = np.searchsorted(self._prep_rows, rows)
        capped = np.minimum(loc, max(self._prep_rows.size - 1, 0))
        if self._prep_rows.size == 0 or not np.all(self._prep_rows[capped] == rows):
            raise ValueError(
                "SourcePrior.log_density: rows are not a subset of the last "
                "prepare(rows) batch")
        return capped

    # -----------------------------------------------------------------
    # broadcasting: rows (n,), a/log10_b (n,) or (n,m), model_index (m,)
    # or (n,m) -- one common (n,m) shape, no Python loop (rule 8).
    # -----------------------------------------------------------------

    def _prepare(self, rows, a, log10_b, model_index):
        rows = np.asarray(rows, dtype=np.intp)
        n = rows.shape[0]
        a1 = np.atleast_1d(np.asarray(a, dtype=np.float64))
        b1 = np.atleast_1d(np.asarray(log10_b, dtype=np.float64))
        if a1.ndim == 1:
            a1 = a1[:, None]
        if b1.ndim == 1:
            b1 = b1[:, None]
        pieces = [a1, b1]
        mi1 = None
        if model_index is not None:
            mi1 = np.atleast_1d(np.asarray(model_index, dtype=np.intp))
            if mi1.ndim == 1:
                mi1 = mi1[None, :]
            pieces.append(mi1)
        shp = np.broadcast_shapes(*[p.shape for p in pieces])
        if shp[0] == 1 and n > 1:
            shp = (n,) + shp[1:]
        if shp[0] != n:
            raise ValueError(
                "SourcePrior.log_density: query batch's leading dimension %r "
                "does not match rows (%d)" % (shp, n))
        a2 = np.broadcast_to(a1, shp)
        b2 = np.broadcast_to(b1, shp)
        rows2d = np.broadcast_to(rows[:, None], shp)
        mi2 = np.broadcast_to(mi1, shp).astype(np.intp) if mi1 is not None else None
        return rows2d, a2, b2, mi2, shp

    def log_density(self, cls, rows, a, log10_b, model_index=None):
        """`ln lambda~_cls(a, log10 B | I_s)` for catalogue `rows` (n,)
        and query points `a`/`log10_b` ((n,) or (n,m)); `model_index`
        ((m,) or (n,m)) is required for GAL and H2S, ignored otherwise
        (module docstring). ONE composition for all six classes (owner
        ruling, 2026-09-06): `ln shape + ln selection - ln Z[rows]`,
        where `shape`/`selection` are the class's own object (below) and
        `Z` is `Z_<CLS>` off the table -- `prior.table` now WRITES that
        column from this same `_z_by_quadrature` (called once per class
        per source at table-build time, not per batch here: owner
        ruling, 2026-09-06's step C2a), so this is still the normaliser
        of the density the fitter actually reads, just computed
        upstream once for the whole survey instead of per `prepare`
        batch. `Z = 0` (every object of this class is genuinely
        impossible at this source's own limits) correctly returns
        `-inf` for every template -- that is not a defect, `check`
        below counts such sources separately rather than folding them
        into the identity's own worst/typical deviation."""
        if cls not in CLASSES:
            raise ValueError("SourcePrior.log_density: unknown class %r, must be one of %r"
                             % (cls, CLASSES))
        if self.cls is not None and cls != self.cls:
            raise ValueError(
                "SourcePrior.log_density: this instance was scoped to class %r at "
                "construction, cannot read class %r" % (self.cls, cls))
        if cls in ("gal", "h2s") and model_index is None:
            raise ValueError("SourcePrior.log_density: class %r needs model_index" % (cls,))
        rows2d, a2, b2, mi2, shp = self._prepare(rows, a, log10_b, model_index)

        obj = self._classes[cls]
        shape_val = obj.shape(rows2d, a2, b2, model_index=mi2)
        sel_val = obj.selection(rows2d, a2, b2, model_index=mi2)
        numerator = shape_val * sel_val
        z = self.table["Z_%s" % cls.upper()][rows2d]
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_val = np.log(numerator) - np.log(z)
        return np.where((numerator > 0.0) & (z > 0.0), ln_val, -np.inf)


# ---------------------------------------------------------------------------
# report (rules 10, 11, 13): the normalisation identity and the read cost
# ---------------------------------------------------------------------------

#: The number of query points per source per class the timed read-cost
#: probe uses.
_N_QUERY_TIMING = 9

#: The library model count the timed read-cost probe uses for every class
#: (the STAR/`sps` library's own count, used as one shared probe size
#: across classes so the six numbers are comparable).
_READ_COST_N_MODEL = 4066

#: `CODING_RULES.md` 10a: the read-cost probe never evaluates more than
#: this many models in one `log_density` call.
_READ_COST_BLOCK = 512

#: `CODING_RULES.md` 10a: the normalisation check's own `(a, log10 B)`
#: mesh is capped at this many points per axis, one source and one class
#: at a time.
_CHECK_GRID_N = 400

#: How many kernel sigmas past the source's own column the check's own
#: extinction grid reaches for GAL/YSO/H2S: generous enough that the
#: log-normal's own tail past this bound is negligible at the check's own
#: 0.02/1e-3 bars.
_A_GRID_SIGMAS = 6.0
_A_GRID_FLOOR = 1.0e-6

_SURVEY_N_SOURCES = 8.66e6


def _family_grid(shape, a_col):
    """`(a_grid, b_grid)`: a fine grid covering STAR/AGB/PAHC's tabulated
    box plus six cells of the declared analytic tail on every edge --
    `check`'s own normalisation grid."""
    x_lo, x_hi = shape.x_edges[0], shape.x_edges[-1]
    b_lo, b_hi = shape.b_edges[0], shape.b_edges[-1]
    x_cell = float(np.mean(np.diff(shape.x_edges)))
    b_cell = float(np.mean(np.diff(shape.b_edges)))
    log_x = np.linspace(x_lo - 6 * x_cell, x_hi + 6 * x_cell, _CHECK_GRID_N)
    b_grid = np.linspace(b_lo - 6 * b_cell, b_hi + 6 * b_cell, _CHECK_GRID_N)
    a_grid = a_col * (10.0 ** log_x)
    return a_grid, b_grid


def _integrate(prior, cls, row, a_grid, b_grid, model_index=None):
    """The trapezoid-rule integral of `exp(log_density)` over the
    rectangular `(a_grid, b_grid)` mesh for one source -- `check`'s own
    numerical acceptance test."""
    n_a, n_b = a_grid.size, b_grid.size
    rows = np.full(n_a * n_b, row, dtype=np.intp)
    a_flat = np.repeat(a_grid, n_b)
    b_flat = np.tile(b_grid, n_a)
    mi = None if model_index is None else np.array([model_index], dtype=np.intp)
    ln_val = prior.log_density(cls, rows, a_flat, b_flat, model_index=mi)
    density = np.exp(ln_val).reshape(n_a, n_b)
    integral = float(np.trapz(np.trapz(density, b_grid, axis=1), a_grid, axis=0))
    return integral


def _integrate_yso(prior, row, a_grid):
    """`integral`: YSO's own `(a, log10 B)` integral over a SHEARED mesh,
    `log10 B` centred on the ridge's own `mean(a)` at every `a_grid`
    point."""
    ridge_b = float(prior.table["RIDGE_INTERCEPT"][row])
    ridge_m = float(prior.table["RIDGE_SLOPE"][row])
    width_b = float(prior.table["RIDGE_WIDTH"][row])
    n_a, n_b = a_grid.size, _CHECK_GRID_N
    mean_b = ridge_b + ridge_m * a_grid
    b_lo, b_hi = mean_b - 6 * width_b, mean_b + 6 * width_b
    t = np.linspace(0.0, 1.0, n_b)
    b_grid_2d = b_lo[:, None] + t[None, :] * (b_hi - b_lo)[:, None]

    rows = np.full(n_a * n_b, row, dtype=np.intp)
    a_flat = np.repeat(a_grid, n_b)
    b_flat = b_grid_2d.ravel()
    ln_val = prior.log_density("yso", rows, a_flat, b_flat)
    density = np.exp(ln_val).reshape(n_a, n_b)
    inner = np.trapz(density, x=b_grid_2d, axis=1)
    return float(np.trapz(inner, a_grid, axis=0))


def _extinction_grid(prior, row, a_col):
    """Log-spaced `a` grid from a tiny floor to `_A_GRID_SIGMAS` kernel
    sigmas past `a_col`, at this source's own arm: the exact extinction
    marginal every class but the family shapes reads is sharply peaked
    toward `a_col`, so log-spacing (the same fix `star_shapes` uses for
    its own `log10 x` axis) is needed to resolve the peak. `map_class` is
    the source's own `A_COL_PROVENANCE` (fixed defect, owner 2026-09-06:
    was `YsoShape._map_class`'s sightline majority)."""
    sigma_col = float(prior.table["A_COL_SIG_K"][row])
    map_class = prior.table["A_COL_PROVENANCE"][row:row + 1]
    mu, sigma = prior.yso_shape.kernel.params(
        np.array([a_col]), np.array([sigma_col]), map_class)
    a_hi = a_col * 10.0 ** (mu[0] + _A_GRID_SIGMAS * sigma[0])
    return np.geomspace(_A_GRID_FLOOR, max(a_hi, 1.0e-5), _CHECK_GRID_N)


def check(config, region, n_sources=50, seed=0):
    """For `n_sources` random catalogue rows and each of the six classes,
    the numerical integral of `exp(log_density)` over a fine `(a, log10
    B)` grid against 1: within 0.02 for STAR/AGB/PAHC/GAL/H2S
    (`star_shapes.EPS_SHAPE`), within 1e-3 for YSO (analytic). A source
    whose `Z_<CLS>` (`prior.table`, `_z_by_quadrature` at build time) is
    zero has no object of this class catalogueable at its own limits --
    every template correctly reads `-inf` there, so its integral is
    correctly zero, and it is counted separately (`n_zero`) rather than
    folded into the identity's own typical/worst deviation (owner
    ruling, 2026-09-06). Reports typical (median) and worst deviation
    per class over the sources where `Z > 0`, and the wall time of one
    source's `log_density` call for one class at 4,066 models times 9
    query points, extrapolated linearly to the survey's 8.66e6
    sources."""
    prior = SourcePrior(config, region)
    rng = np.random.default_rng(seed)
    n = min(int(n_sources), prior.n_source)
    rows = rng.choice(prior.n_source, size=n, replace=False)
    prior.prepare(rows)

    # `Z`'s own cost: GAL ONLY (owner ruling, 2026-09-06, step C2d) is
    # still computed live, by `prior.table`'s build (`_recompute_z_
    # columns`), not here -- this re-times `_z_by_quadrature` directly,
    # on this run's own sources, purely to report the number the table
    # build pays once per survey source. Every other class's `Z` is a
    # plain table column now (`prior.counts_star_family`/`prior.
    # counts_cloud`'s own dot product), nothing to time here.
    z_cost_ms = {}
    if "gal" in prior._classes:
        t0 = time.time()
        _z_by_quadrature(prior, "gal", rows)
        z_cost_ms["gal"] = (time.time() - t0) / rows.size * 1000.0

    devs = {cls: [] for cls in CLASSES}
    n_zero = {cls: 0 for cls in CLASSES}
    for row in rows:
        a_col = float(prior.table["A_COL_K"][row])

        for cls in FAMILY_CLASSES:
            if prior.table["Z_%s" % cls.upper()][row] <= 0.0:
                n_zero[cls] += 1
                continue
            shape = prior.shapes[cls]
            a_grid, b_grid = _family_grid(shape, a_col)
            a_grid = a_grid[a_grid > 0.0]
            integral = _integrate(prior, cls, row, a_grid, b_grid)
            devs[cls].append(abs(integral - 1.0))

        a_grid_cloud = _extinction_grid(prior, row, a_col)

        if prior.table["Z_GAL"][row] <= 0.0:
            n_zero["gal"] += 1
        else:
            b_grid_gal = prior.gal_log10_s_grid - np.log10(prior.gal_fref[0])
            integral = _integrate(prior, "gal", row, a_grid_cloud, b_grid_gal, model_index=0)
            devs["gal"].append(abs(integral - 1.0))

        if prior.table["Z_YSO"][row] <= 0.0:
            n_zero["yso"] += 1
        else:
            integral = _integrate_yso(prior, row, a_grid_cloud)
            devs["yso"].append(abs(integral - 1.0))

        # The check's own Sigma domain must be the same grid `_interp_eps_2d`
        # interpolates on (module docstring's "end bins held" convention) --
        # a wider independent range double-counts the held edge value past
        # the real grid, the row-1222 callable/product mismatch this fixes.
        if prior.table["Z_H2S"][row] <= 0.0:
            n_zero["h2s"] += 1
        else:
            log10_sigma_lo = prior.h2s_log10_sigma_grid[0]
            log10_sigma_hi = prior.h2s_log10_sigma_grid[-1]
            b_grid_h2s = (np.linspace(log10_sigma_lo, log10_sigma_hi, _CHECK_GRID_N)
                         - np.log10(prior.h2s_fref[0]))
            integral = _integrate(prior, "h2s", row, a_grid_cloud, b_grid_h2s, model_index=0)
            devs["h2s"].append(abs(integral - 1.0))

    worst = {cls: (max(devs[cls]) if devs[cls] else 0.0) for cls in CLASSES}
    typical = {cls: (float(np.median(devs[cls])) if devs[cls] else 0.0) for cls in CLASSES}

    read_cost = {}
    probe_row = int(rows[0])
    rows_probe = np.array([probe_row], dtype=np.intp)
    a_probe_val = max(float(prior.table["A_COL_K"][probe_row]), 1.0e-3)
    b_probe_block = np.linspace(-2.0, 2.0, _N_QUERY_TIMING)
    for cls in CLASSES:
        n_gal_h2s = prior.gal_fref.size if cls == "gal" else (
            prior.h2s_fref.size if cls == "h2s" else None)
        wall_s = 0.0
        for start in range(0, _READ_COST_N_MODEL, _READ_COST_BLOCK):
            n_block = min(_READ_COST_BLOCK, _READ_COST_N_MODEL - start)
            m = n_block * _N_QUERY_TIMING
            a_probe = np.full((1, m), a_probe_val)
            b_probe = np.tile(b_probe_block, n_block)[None, :]
            mi_probe = None
            if cls in ("gal", "h2s"):
                model_ids = np.arange(start, start + n_block) % n_gal_h2s
                mi_probe = np.repeat(model_ids, _N_QUERY_TIMING)
            t0 = time.time()
            prior.log_density(cls, rows_probe, a_probe, b_probe, model_index=mi_probe)
            wall_s += time.time() - t0
        read_cost[cls] = dict(n_model=_READ_COST_N_MODEL, wall_s=wall_s, per_source_s=wall_s,
                              survey_hours=wall_s * _SURVEY_N_SOURCES / 3600.0)

    return worst, typical, n_zero, read_cost, z_cost_ms


def report(region, worst, typical, n_zero, read_cost, z_cost_ms):
    lines = ["prior.callable: %s: normalisation check, sources with Z > 0 only "
            "(bar 0.02 all but YSO 1e-3)" % region]
    for cls in CLASSES:
        bar = 1.0e-3 if cls == "yso" else 0.02
        lines.append("prior.callable: %s: %s: typical |integral - 1| = %.4g, "
                     "worst = %.4g (bar %.4g)%s, Z=0 sources: %d"
                     % (region, cls, typical[cls], worst[cls], bar,
                        "" if worst[cls] <= bar else "  ** EXCEEDS BAR **", n_zero[cls]))
    lines.append("prior.callable: %s: read cost (one source x N models x %d query points)"
                 % (region, _N_QUERY_TIMING))
    for cls in CLASSES:
        rc = read_cost[cls]
        lines.append("prior.callable: %s: %s: n_model=%d wall=%.4gs -> "
                     "survey extrapolation (8.66e6 sources) = %.3g hours"
                     % (region, cls, rc["n_model"], rc["wall_s"], rc["survey_hours"]))
    lines.append("prior.callable: %s: Z quadrature cost (prepare, ms/source, one class)"
                 % (region,))
    for cls in CLASSES:
        if cls in z_cost_ms:
            lines.append("prior.callable: %s: %s: %.4g ms/source"
                         % (region, cls, z_cost_ms[cls]))
    return lines


if __name__ == "__main__":
    import sys

    from sesnaimpute import config as _config_module

    cfg = _config_module.load(sys.argv[1])
    for _region in sys.argv[2:]:
        _worst, _typical, _n_zero, _read_cost, _z_cost_ms = check(cfg, _region)
        for _line in report(_region, _worst, _typical, _n_zero, _read_cost, _z_cost_ms):
            print(_line, flush=True)
