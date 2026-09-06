"""The per-source posterior-facing prior callable (`10_POSTERIOR.md`
section 3's three properties; `SPEC_PRIORS.md` section 0.2; the prior
table's own join, `IMPLEMENTATION.md` sections 3 and 5; the quarry
`bms_prior/prior_table/assemble.py`'s `SourcePriorCallable` -- the convex
blend of two bracketing node evaluators, times the source's own selection
over its normaliser, without its Gaussian-shortcut hooks or "clause"
verifiers, which are the fitter's own concern).

`SourcePrior(config, region)` loads every upstream product for one region
once: the prior table (`prior.table.read`), the shared column grid
(`prior.column_grid.nodes`), the three field-family shapes and their exact
selection models (`prior.star_shapes`, `prior.counts_star_family.
read_class_model`), the GAL selection table and counts law (`prior.gal`),
the YSO sightline shape (`prior.yso.YsoShape`, shared with H2S's own
extinction marginal), and the H2S region product (`prior.h2s`). One method,
`log_density(cls, rows, a, log10_b, model_index=None)`, then returns
`ln lambda~_cls(a, log10 B | I_s)` for a batch of catalogue rows and a
batch of query points per row, vectorised throughout (rule 8): no Python
loop over sources or query points, only over the handful of DISTINCT
(node, group, map-class) combinations a batch touches, matching the same
grouping idiom `prior.counts_star_family.grouped_eval` already uses.

**The five formulas** (`IMPLEMENTATION.md` section 3's "evaluated per
source by" column; `SPEC_PRIORS.md` section 0.2's identity
`N_C(s) = Integral p_C(a,b|I_s) eps_s(a,b) da db`, `lambda~_C = p_C eps_s
/ N_C`):

STAR, AGB, PAHC
    `lambda~_C(a, log10 B) = p_C(a, log10 B | I_s) . eps_s(a, log10 B) / Z_C`

    `p_C` is the source's own tile shape (`star_shapes.ClassShape.
    density`, blend of the two shape nodes bracketing `A_s` in `log A`,
    bicubic in `(log10 x, log10 B)` with `x = a / A_s`, PAHC additionally
    blended over its own 8 micron limit grid) converted from the stored
    array's own units to a density in `(a, log10 B)`. The stored array is
    a discretised probability MASS on the `(log10 x, log10 B)` grid (its
    own algebraic acceptance, `star_shapes._report`, is `DENSITY.sum() ==
    1 - MASS_OUTSIDE`, a bin-mass identity, not a density integral), so a
    pointwise read must be divided by the grid's own fixed cell area
    (`dx . db`, the same constant for every node and tile of one class and
    region) to become `f(log10 x, log10 B)`, the density with respect to
    `d(log10 x) d(log10 B)`; the further change of variables
    `log10 x = log10 a - log10 A_s` (`A_s` fixed per source) carries a
    Jacobian `d(log10 x)/da = 1/(a ln 10)`:

        p_C(a, log10 B) = ClassShape.density(...) / (a . ln10 . dx . db)

    `eps_s` is the class's exact selection (`prior.selection.
    PassFractionModel.evaluate`) at the QUERY point's own extinction `a`
    (bracketed against the shared column grid, `model.a_nodes`, exactly
    as `counts_star_family.algebraic_check`'s own "direct" reference
    evaluation does -- the more precise of the two forms that module's own
    docstring names, since a pointwise callable has no reason to take the
    coarser (tile, node)-grid shortcut Z's own build-time integral does
    for speed), the query's `log10 B`, the source's own common-mode shift
    `s` and four-degree-of-freedom residual `Delta` (recomputed here from
    `catalog.limits` against the class's OWN reference limit vector,
    `prior.selection.split_common_mode` -- the prior table's stored
    `GROUP` is the SHARED depth-groups product's assignment and is not a
    valid index into a family class's own, independently K-searched
    selection table, whose `K` can differ by an order of magnitude
    (Perseus: STAR/PAHC K=128, AGB K=16, the shared product's own default
    K=16); recomputing `Delta` and letting `PassFractionModel.evaluate`'s
    own `assign_group` resolve it is the one path guaranteed to match
    whichever `K` that class's build settled on).

    `Z_C` is the table's own stored normaliser (`Z_STAR`/`Z_AGB`/
    `Z_PAHC`): the identity this callable owes the fitter is
    `Integral p_C(a,b) eps_s(a,b) da db = Z_C`, so dividing the pointwise
    numerator above by the SAME stored number is what makes `lambda~_C`
    integrate to 1 (up to the two disclosed approximations already
    budgeted into the shape's own 0.02 fidelity bar, `EPS_SHAPE`: `Z_C`'s
    own build-time integral omits the shape's analytic tail contribution,
    and evaluates selection on the node's own `a = a_node . x` rather
    than the query's exact `a` -- both stated as within-bar
    approximations by `counts_star_family`'s own module docstring).

GAL
    `lambda~_GAL(a, log10 B) = p_a(a | A_s) . p(log10 S(log10 B)) / Z_GAL`

    A galaxy's `a` **is** the column, spread only by the column kernel
    (`SPEC_PRIORS.md` section 5.2); `p_a` is that kernel's own marginal
    `p(T = a | A_s)`, a scale mixture of the kernel's quadrature nodes at
    embedding density `u = 1` (a point mass, `SPEC_PRIORS.md` section
    1.4's `u(d) -> 1` far behind the cloud): reused from `prior.yso.
    YsoShape`'s own PRE-TABULATED per-(column-node, map-class) quadrature
    (`KERNEL_T`/`KERNEL_W`, blended at the source's own node bracket,
    `NODE_LO`/`NODE_W`) rather than a fresh Gauss-Hermite computation per
    source, for the same reason `YsoShape.marginal_at` reads it that way:
    survey-wide reuse is the point of tabulating it once. The quadrature
    pair `(T_q, w_q)` is, like the shape arrays above, a discretised MASS
    (`w_q` a trapezoid-rule bin mass, `kernel._trapezoid_weights`), so the
    density at an arbitrary query `a` is recovered by dividing each
    `w_q` by its own local `T` spacing and interpolating in `ln T`
    (`_kernel_density_rows`) -- the same mass-to-density conversion the
    family classes need above, on a different tabulation.

    `p(log10 S)` is the counts law's own NORMALISED density
    (`SPEC_PRIORS.md` section 5.2's own "prop": `phi(S) . S` is only
    proportional to a density, so this callable divides by its own
    `gal_phi_total = Integral phi(S) . S ln10 dlog10 S`, a survey-wide
    constant computed once in `__init__`) times the exact selection
    `eps_k(log10 S | node blend)` (`prior.gal`'s own `r(a, log10 S)`
    table, the SAME `PassFractionModel` machinery as the family classes,
    with `log10 S` standing in for `log10 B`), evaluated at `log10 S =
    log10 B + log10 f_ref,h`, the query's library model own 4.5 micron
    reference flux (`galz_register`'s `F_REF_I2`, a change of units, C3).
    `Z_GAL` here is NOT the prior table's stored `Z_GAL`
    (`prior.counts_star_family.gal_counts`'s own `Integral phi(S) S eps
    dS / Integral phi(S) S dS`, an eps-weighted AVERAGE with no relation
    to this identity): it is `Integral (normalised p(log10 S)) . eps
    dlog10 S`, this callable's own `gal_z`, computed once per source in
    `__init__` from the same normalised density the numerator above uses,
    so `Integral shape . eps = Z` holds by construction.

YSO
    `lambda~_YSO(a, log10 B) = p(a | A_s) . N(log10 B; mu(a), sigma)`,
    `mu(a) = RIDGE_INTERCEPT + RIDGE_SLOPE . a`

    `p(a | A_s)` is the sightline's own exact extinction marginal
    (`YsoShape.marginal_at`, `SPEC_PRIORS.md` section 6.3): no tabulation,
    a finite sum over the kernel's quadrature of the embedding density's
    own step function, node-blended. The conditional brightness is
    Gaussian in `log10 B` with the table's own per-source ridge (ridge
    parameters vary only by sightline, but are carried per source in the
    table already). No selection lives in the YSO shape (spec section
    6.2); `Z_YSO = 1` by construction and is not read from the table.

H2S
    `lambda~_H2S(a, log10 B) = p(a | A_s) . p_r(log10 Sigma) . eps_k(a,
    Sigma | group) / Z_H2S`, `log10 Sigma = log10 B + log10 Sigma_ref,h`

    `p(a | A_s)` is YSO's own marginal, shared not copied (spec section
    7: "identical to YSO's embedding density on this sightline"). `p_r`
    is the region's own lognormal in `log10 Sigma` (Gaussian in
    `log10 Sigma`, `LOGSIG_MEAN`/`LOGSIG_STD`); since `log10 Sigma =
    log10 B + const` for a fixed template, this is equally a Gaussian in
    `log10 B`, no extra Jacobian. `eps_k` is `prior.h2s`'s own `(K, n_node,
    n_sigma)` table, node-blended in `a` and linearly interpolated in
    `log10 Sigma`, read at the source's own `GROUP` -- H2S's own build
    (`prior.h2s.region_limit_log10_8`) clusters directly off the SHARED
    `prior.depth_groups` product (unlike the family classes above), so the
    table's stored `GROUP` is exactly the row `prior.h2s`'s own
    `GROUP_CENTRES` axis was built at, no recomputation needed.
    `Sigma_ref,h`: the H2S library register (`h2shock_register.hdf5`)
    carries no dedicated H2 1-0 S(1) line-alone surface-brightness
    reference dataset (only the ordinary per-band `F_REF_<band>` fluxes
    every register carries) -- disclosed, not silently substituted: this
    callable uses the register's `F_REF_Ks` as the brightness unit, per
    the brief's own fallback instruction, so H2S's `log10 B` origin here
    is the shock library's Ks-band reference flux, not the true S(1)
    line-alone reference the spec names. `Z_H2S` is the table's stored
    `Z_H2S` (`= EPS_H2S`, `prior.counts_cloud`'s own identity).

Every class maps `a < 0` to `-inf` (the shapes' own `a < 0 -> 0` density,
logged); outside a tabulated box every class but YSO/H2S applies the
shape's own declared analytic tail (`ClassShape._eval_node`) or the
selection table's own boundary hold (`column_grid.bracket`'s clip, the
same "end bins held" convention `SPEC_PRIORS.md` section 1.3 already
states for PAHC's own limit interpolation); YSO and H2S are analytic in
`a` throughout their support by construction (`YsoShape.marginal_at`'s own
finite quadrature sum) and Gaussian in `log10 B`, so no tail declaration
is needed for either.

**Per-batch tabulation (`prepare`).** The fitter reads every model of a
class once per source (the paragraph above, `10_POSTERIOR.md`'s adopted
reading), so GAL/YSO/H2S's exact extinction machinery -- a sum over the
column kernel's own hundreds of quadrature points, `YsoShape.
marginal_at`/`_kernel_density_rows` -- would otherwise be recomputed once
per (source, model) pair: for YSO's 200,000 models that is 200,000 exact
sums per source. `SourcePrior.prepare(rows)` tabulates, once per batch of
about ten thousand sources (`CODING_RULES.md` 10b), every quantity those
three classes' reads need, on a fixed 256-point `log10 a` grid (`N_A_GRID`)
sized from the shared column grid's own floor to four times the largest
kernel quadrature node the batch's sources touch (the check's own
normalisation grid reaches three times that same bound, `_extinction_grid`,
so the margin keeps every query the check makes inside the interpolation
range):

  - YSO's `p(a | A_s)` (shared with H2S, `YsoShape.marginal_at`'s own
    formula) and GAL's `p_a` (`_kernel_density_rows`'s own formula, the
    embedding-density-free kernel reconstruction -- a DIFFERENT quantity
    from YSO's, despite sharing the same underlying kernel quadrature),
    each `(n_batch, 256)`, built once per DISTINCT `(sightline row, node
    index)` pair the batch touches (`prior.counts_cloud.bin_mass_exact`'s
    own grouping idiom: a region's sources touch orders of magnitude fewer
    such pairs than sources) and then node-blended per source, never
    evaluated per source directly.
  - GAL's `p(log10 S)`, `(n_batch, 61)`, already divided by the source's
    own `Z_GAL` -- built at the source's OWN adopted-column node bracket
    (the same bracket `gal_z`'s build-time integral already uses in
    `__init__`, so this tabulation's numerator and that denominator agree
    by construction, tightening rather than loosening the query-a bracket
    the un-tabulated read used).
  - H2S's `eps(a, Sigma)`, `(K depth groups, 256, 41)`, node-blended in
    `a` at every one of the 256 grid points for every depth group at once
    -- independent of which particular sources the batch holds beyond the
    shared grid, so one small table serves the whole batch and every
    source reads it by its own `GROUP` index.

`log_density` then reads only these tables (a `log10 a`, and for GAL/H2S a
further `log10 S`/`Sigma`, linear interpolation, held at the nearest edge
value outside the tabulated range, the same convention
`_kernel_density_rows` already used) for GAL/YSO/H2S; STAR/AGB/PAHC are
unchanged (their own per-read cost was already at the STAR floor, module
docstring's measured numbers, so tabulating their selection factor too
would add `counts_star_family.grouped_eval`'s own machinery for no
measured speed gain -- not done, see `readcost.py`'s own report).
"""

import os
import time

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.prior import column_grid, counts_star_family, depth_groups
from sesnaimpute.prior import selection, star_shapes
from sesnaimpute.prior import table as table_module
from sesnaimpute.prior import yso as yso_module

#: `ln(10)`: the `d(log10 x)/da = 1/(a ln10)` and `d(log10 S)/dS = 1/(S
#: ln10)` Jacobians every class but YSO/H2S needs once (module docstring).
LN10 = float(np.log(10.0))

#: `1/sqrt(2 pi)`: the Gaussian normalisation YSO's conditional brightness
#: and H2S's region lognormal both need.
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

FAMILY_CLASSES = ("star", "agb", "pahc")
CLASSES = ("star", "agb", "pahc", "gal", "yso", "h2s")

#: `CODING_RULES.md` 10a: `_interp_rows`/`_interp_h2s_eps` (GAL/YSO/H2S's
#: table reads) process their query batch in blocks of this many rows, so
#: their `(chunk, grid.size)` intermediates never scale with the caller's
#: own `n . m`.
_KERNEL_CHUNK = 20000

#: The two library keys `IMPLEMENTATION.md` section 3's GAL/H2S rows need
#: a per-model reference flux from (`sed_models/registers/<key>_register.
#: hdf5`, module docstring: H2S falls back to `F_REF_Ks`, the register
#: carrying no dedicated H2 1-0 S(1) dataset).
_LIBRARY_KEY = {"gal": "galz", "h2s": "h2shock"}
_LIBRARY_BAND = {"gal": "F_REF_I2", "h2s": "F_REF_Ks"}

#: `prepare`'s own `log10 a` tabulation width (module docstring): fine
#: enough that the linear interpolation between adjacent grid points is
#: far below the classes' own 0.02 shape fidelity bar over the smooth
#: kernel-quadrature marginals it replaces.
N_A_GRID = 256

#: `prepare`'s and `_extinction_grid`'s shared floor: GAL/YSO/H2S's exact
#: marginals do not vanish as `a -> 0` (the embedding density's near-field
#: cell, or the kernel's own near-`T=0` mass, is generically nonzero
#: there), so both the tabulation and the check's own integration grid
#: must reach this close to zero rather than the shared column grid's own
#: measured floor (`column_grid.nodes`'s first node), which is calibrated
#: to a source's own measurement uncertainty, not to this tail.
_A_GRID_FLOOR = 1.0e-6

#: `prepare`'s grid ceiling is this many times the largest kernel
#: quadrature node (`T`) the batch's sources touch; the check's own
#: normalisation grid (`_extinction_grid`) reaches three times that same
#: bound, so this margin keeps every query the check makes inside the
#: interpolation range.
_A_GRID_SAFETY = 4.0

#: Distinct `(sightline row, node index)` pairs per joblib chunk
#: (CODING_RULES.md 10a): each chunk's own transient arrays are `(chunk
#: pairs * N_A_GRID, n_quadrature)`, several of them alive at once inside
#: `_marginal_rows`/`_kernel_density_rows`/`_row_bin`, and up to
#: `config.n_jobs` chunks run at once -- a smaller bound than
#: `prior.counts_cloud.PAIR_CHUNK` uses, because this module's own grid
#: (`N_A_GRID` = 256) is wider than that module's bin-edge count and the
#: kernel's own quadrature order (measured ~257) multiplies it again; a
#: region with many sightlines and nodes (Perseus: 297 sightlines x 183
#: nodes) touches thousands of distinct pairs, so the bound must hold per
#: chunk, not just in total.
_PREP_PAIR_CHUNK = 40


def _library_reference_flux(config, cls):
    """`(n_model,)`: `_LIBRARY_BAND[cls]` off `_LIBRARY_KEY[cls]`'s own
    register (module docstring's GAL/H2S "a change of units" reads)."""
    path = os.path.join(config.inputs["sed_models"], "registers",
                         "%s_register.hdf5" % _LIBRARY_KEY[cls])
    with h5py.File(path, "r") as f:
        return np.asarray(f["models"][_LIBRARY_BAND[cls]][:], dtype=np.float64)


def _kernel_density_rows(t, w, a_query):
    """`(n,)`: the continuous kernel density `p(T = a_query)` per row,
    recovered from `(t, w)` (each `(n, n_q)`, a trapezoid-rule bin MASS
    on `T`, `kernel.Kernel.nodes`'s own construction) by dividing each
    quadrature weight by its own local `T` spacing
    (`kernel._trapezoid_weights`'s own half-width formula, inverted) and
    interpolating linearly in `ln T` -- the same mass-to-density
    conversion `star_shapes.ClassShape` needs on its own grid (module
    docstring). Held at the nearest edge value outside the tabulated `T`
    range (`SPEC_PRIORS.md` section 1.3's "end bins held" convention)."""
    spacing = np.empty_like(t)
    spacing[:, 0] = 0.5 * (t[:, 1] - t[:, 0])
    spacing[:, -1] = 0.5 * (t[:, -1] - t[:, -2])
    spacing[:, 1:-1] = 0.5 * (t[:, 2:] - t[:, :-2])
    dens = w / spacing

    log_t = np.log(t)
    log_a = np.log(np.clip(a_query, 1.0e-300, None))
    n, n_q = t.shape
    idx = np.clip(np.sum(log_t <= log_a[:, None], axis=1) - 1, 0, n_q - 2)
    rows = np.arange(n)
    lo, hi = log_t[rows, idx], log_t[rows, idx + 1]
    d_lo, d_hi = dens[rows, idx], dens[rows, idx + 1]
    span = np.where(hi > lo, hi - lo, 1.0)
    frac = np.clip((log_a - lo) / span, 0.0, 1.0)
    return d_lo + frac * (d_hi - d_lo)


def _unique_row_node_pairs(rows, node_lo, node_hi, n_node):
    """Every distinct `(sightline row, node index)` pair a batch's own
    sources touch through either side of their own node bracket (the same
    helper `prior.counts_cloud._unique_row_node_pairs` uses): `(uniq_row,
    uniq_node)`, and each source's own index into them for its `NODE_LO`
    side and its `NODE_HI` side."""
    rows = rows.astype(np.int64)
    key_lo = rows * n_node + node_lo.astype(np.int64)
    key_hi = rows * n_node + node_hi.astype(np.int64)
    uniq_keys, inverse = np.unique(np.concatenate([key_lo, key_hi]), return_inverse=True)
    uniq_row = (uniq_keys // n_node).astype(np.intp)
    uniq_node = (uniq_keys % n_node).astype(np.intp)
    n_src = rows.size
    return uniq_row, uniq_node, inverse[:n_src], inverse[n_src:]


def _pair_grid_values(shape, a_grid, row_chunk, node_chunk):
    """`(marginal, kernel_density)`, each `(n_pairs, N_A_GRID)`: YSO's
    exact `p(a | A_s)` (`YsoShape._marginal_rows`) and GAL's exact kernel
    reconstruction at embedding density `u = 1` (`_kernel_density_rows`,
    module docstring -- a DIFFERENT quantity sharing the same underlying
    kernel quadrature), both evaluated at every grid point of `a_grid` for
    one chunk of distinct `(sightline row, node index)` pairs -- one
    batched call into `YsoShape`'s own per-row machinery, the same
    primitives `marginal_at`'s own per-source formula uses, evaluated here
    once per pair rather than once per source."""
    n_pairs = row_chunk.size
    n_grid = a_grid.size
    rows_rep = np.repeat(row_chunk, n_grid)
    node_rep = np.repeat(node_chunk, n_grid)
    a_rep = np.tile(a_grid, n_pairs)
    t, w = shape._gather_quadrature(rows_rep, node_rep)
    marginal = shape._marginal_rows(a_rep, rows_rep, t, w)
    kernel_density = _kernel_density_rows(t, w, a_rep)
    return marginal.reshape(n_pairs, n_grid), kernel_density.reshape(n_pairs, n_grid)


def _build_pair_tables(config, shape, a_grid, rows, node_lo, node_w):
    """`(p_a_yso, p_a_gal)`, each `(n_batch, N_A_GRID)` float32: YSO/H2S's
    shared extinction marginal and GAL's own kernel reconstruction, tiled
    on `a_grid` and node-blended per source, from one deduplicated pass over
    the batch's own distinct `(sightline row, node index)` pairs
    (`_unique_row_node_pairs`; module docstring's `prepare`). Chunked over
    pairs under `joblib` threads (`CODING_RULES.md` 10a), the same
    thread-pool idiom `prior.counts_cloud.bin_mass_exact` uses so `shape`
    (a whole region's sightline arrays) is never pickled into a worker."""
    n_node = shape.n_node
    node_hi = np.clip(node_lo + 1, 0, n_node - 1)
    uniq_row, uniq_node, inv_lo, inv_hi = _unique_row_node_pairs(rows, node_lo, node_hi, n_node)

    n_pairs = uniq_row.size
    n_chunks = max(1, int(np.ceil(n_pairs / _PREP_PAIR_CHUNK)))
    idx_chunks = np.array_split(np.arange(n_pairs), n_chunks)
    results = Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(_pair_grid_values)(shape, a_grid, uniq_row[idx], uniq_node[idx])
        for idx in idx_chunks)
    if n_pairs:
        m_pairs = np.concatenate([r[0] for r in results], axis=0)
        k_pairs = np.concatenate([r[1] for r in results], axis=0)
    else:
        m_pairs = np.empty((0, a_grid.size))
        k_pairs = np.empty((0, a_grid.size))

    w = node_w[:, None]
    p_a_yso = (1.0 - w) * m_pairs[inv_lo] + w * m_pairs[inv_hi]
    p_a_gal = (1.0 - w) * k_pairs[inv_lo] + w * k_pairs[inv_hi]
    return p_a_yso.astype(np.float32), p_a_gal.astype(np.float32)


def _interp_rows(table, grid, local_idx, query):
    """`(n,)`: linear interpolation of `table[local_idx[i], :]` at
    `query[i]` on the shared ascending 1-D `grid`, held at the nearest
    edge value outside the tabulated range (`_kernel_density_rows`'s own
    "end bins held" convention). Processed in `_KERNEL_CHUNK` blocks
    (`CODING_RULES.md` 10a) so the `(chunk, grid.size)` gather this needs
    never scales with the caller's own, possibly source-times-model-sized,
    batch."""
    n = query.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_grid = grid.size
    for start in range(0, n, _KERNEL_CHUNK):
        stop = min(start + _KERNEL_CHUNK, n)
        li = local_idx[start:stop]
        clamped = np.clip(query[start:stop], grid[0], grid[-1])
        idx = np.clip(np.searchsorted(grid, clamped) - 1, 0, n_grid - 2)
        lo, hi = grid[idx], grid[idx + 1]
        span = np.where(hi > lo, hi - lo, 1.0)
        frac = np.where(hi > lo, (clamped - lo) / span, 0.0)
        rows_i = np.arange(li.size)
        v_lo = table[li, idx]
        v_hi = table[li, idx + 1]
        out[start:stop] = v_lo + frac * (v_hi - v_lo)
    return out


def _interp_h2s_eps(eps_grid, log10_a_grid, log10_sigma_grid, group_idx, log10_a, log10_sigma):
    """`(n,)`: H2S's tabulated `eps(a, Sigma)` (`prepare`'s `(K, N_A_GRID,
    41)` table) read by one linear interpolation in `a`, then one in
    `Sigma`, at each row's own depth group -- both edges held (the same
    convention `_interp_rows` uses). Processed in `_KERNEL_CHUNK` blocks
    (`CODING_RULES.md` 10a): the intermediate `(chunk, n_sigma)` curve this
    needs never scales with the caller's own batch."""
    n = group_idx.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_a_grid = log10_a_grid.size
    n_sigma_grid = log10_sigma_grid.size
    for start in range(0, n, _KERNEL_CHUNK):
        stop = min(start + _KERNEL_CHUNK, n)
        g = group_idx[start:stop]
        la = np.clip(log10_a[start:stop], log10_a_grid[0], log10_a_grid[-1])
        ls = np.clip(log10_sigma[start:stop], log10_sigma_grid[0], log10_sigma_grid[-1])

        idx_a = np.clip(np.searchsorted(log10_a_grid, la) - 1, 0, n_a_grid - 2)
        lo_a, hi_a = log10_a_grid[idx_a], log10_a_grid[idx_a + 1]
        span_a = np.where(hi_a > lo_a, hi_a - lo_a, 1.0)
        frac_a = np.where(hi_a > lo_a, (la - lo_a) / span_a, 0.0)
        curve = ((1.0 - frac_a)[:, None] * eps_grid[g, idx_a, :]
                 + frac_a[:, None] * eps_grid[g, idx_a + 1, :])          # (chunk, n_sigma)

        idx_s = np.clip(np.searchsorted(log10_sigma_grid, ls) - 1, 0, n_sigma_grid - 2)
        lo_s, hi_s = log10_sigma_grid[idx_s], log10_sigma_grid[idx_s + 1]
        span_s = np.where(hi_s > lo_s, hi_s - lo_s, 1.0)
        frac_s = np.where(hi_s > lo_s, (ls - lo_s) / span_s, 0.0)
        rows_i = np.arange(curve.shape[0])
        out[start:stop] = (curve[rows_i, idx_s]
                           + frac_s * (curve[rows_i, idx_s + 1] - curve[rows_i, idx_s]))
    return out


class SourcePrior(object):
    """`log_density(cls, rows, a, log10_b, model_index=None)`, one region's
    every upstream product loaded once (module docstring)."""

    def __init__(self, config, region):
        self.config = config
        self.region = region
        self.table = table_module.read(config, region)
        self.n_source = self.table["A_COL_K"].shape[0]
        self.a_nodes_full = column_grid.nodes(config)

        f_lim8 = limits_module.limits(config, region)
        self.log10_flim8 = np.log10(f_lim8)
        self._idx_i4 = selection.BAND_KEYS.index("I4")

        # -- STAR, AGB, PAHC: the tile shape and the exact selection model,
        # each class's own Delta/s recomputed against ITS OWN reference
        # limit vector (module docstring: the table's shared GROUP does
        # not index a family class's own, independently sized table).
        self.shapes = {}
        self.family_models = {}
        for cls in FAMILY_CLASSES:
            shape = star_shapes.read(config, region, cls)
            model = counts_star_family.read_class_model(config, region, cls)
            s, delta5 = selection.split_common_mode(self.log10_flim8, model.knots.ref_log10_flim)
            dx = float(np.mean(np.diff(shape.x_edges)))
            db = float(np.mean(np.diff(shape.b_edges)))
            self.shapes[cls] = shape
            self.family_models[cls] = dict(model=model, s=s, delta5=delta5, dx=dx, db=db)

        # -- GAL: the region's r(a, log10 S) selection table and the
        # survey-wide counts law (`prior.counts_star_family.gal_counts`'s
        # own reads, reused verbatim).
        gal_path = config_module.product_path(config, "bms", "gal", "prior", "region", region=region)
        with h5py.File(gal_path, "r") as f:
            gal_a_nodes = f["A_NODES"][:].astype(np.float64)
            gal_log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
            gal_eps = f["EPS"][:].astype(np.float64)
            gal_group_centres = f["GROUP_CENTRES"][:].astype(np.float64)
            gal_ref_log10_flim = f["REF_LOG10_FLIM"][:].astype(np.float64)
        counts_path = config_module.product_path(config, "bms", "gal", "counts", "survey")
        with h5py.File(counts_path, "r") as f:
            law_log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
            self.gal_phi_s = f["PHI_S"][:].astype(np.float64)
        if not np.allclose(law_log10_s_grid, gal_log10_s_grid):
            raise ValueError("prior.callable: GAL's region prior and the survey counts "
                             "law disagree on LOG10_S_GRID for region %r" % region)
        self.gal_log10_s_grid = gal_log10_s_grid
        gal_knots = depth_groups.DepthGroups(
            region, gal_ref_log10_flim, gal_group_centres,
            np.zeros((0, gal_group_centres.shape[1])), gal_group_centres.shape[0])
        self.gal_model = selection.PassFractionModel(gal_a_nodes, gal_log10_s_grid, gal_knots, gal_eps)
        self.gal_s, self.gal_delta5 = selection.split_common_mode(self.log10_flim8, gal_ref_log10_flim)
        self.gal_fref = _library_reference_flux(config, "gal")

        # `prior.gal`'s own stored Z_GAL (`Integral phi(S) S eps dS /
        # Integral phi(S) S dS`, `counts_star_family.gal_counts`) is an
        # eps-weighted AVERAGE, not this callable's needed `Integral
        # (normalised shape) . eps` -- the two are different quantities
        # (module docstring's GAL row correction). `phi(S) . S` is only
        # PROPORTIONAL to a density (`SPEC_PRIORS.md` section 5.2's own
        # "prop"); its own normaliser `gal_phi_total = Integral phi(S) . S
        # . ln10 dlog10 S` (a survey-wide constant) makes it one, and this
        # callable's own `gal_z[s] = Integral p_logS_intrinsic . eps
        # dlog10 S`, computed once per source here (vectorised over the
        # whole region, `PassFractionModel.integrate_count`, rule 8), is
        # the normaliser the identity `Integral shape . eps = Z` actually
        # needs.
        self.gal_phi_total = float(np.trapz(
            self.gal_phi_s * (10.0 ** self.gal_log10_s_grid) * LN10, self.gal_log10_s_grid))
        w1_norm = (self.gal_phi_s * (10.0 ** self.gal_log10_s_grid) * LN10) / self.gal_phi_total
        node_lo_all, node_w_all = column_grid.bracket(self.table["A_COL_K"], self.gal_model.a_nodes)
        self.gal_z = np.asarray(self.gal_model.integrate_count(
            w1_norm[None, :], node_lo_all, node_w_all, self.gal_s, self.gal_delta5))
        # `w1_norm` IS `phi(S).S.ln10 / gal_phi_total` on `gal_log10_s_grid`
        # itself (the interpolated form `_log_density_gal` reads at a query
        # point is the same array evaluated off-grid) -- `prepare` reuses
        # it verbatim for the GAL `p(log10 S)` tabulation (module docstring).
        self._gal_phi_density_grid = w1_norm

        # -- YSO: the sightline shape (marginal_at, shared with H2S).
        self.yso_shape = yso_module.YsoShape.read(config, region)

        # -- H2S: the region's Sigma lognormal and (K, n_node, n_sigma)
        # selection table, keyed by the SAME shared GROUP the table stores
        # (module docstring).
        h2s_path = config_module.product_path(config, "bms", "h2s", "prior", "region", region=region)
        with h5py.File(h2s_path, "r") as f:
            self.h2s_a_nodes = f["A_NODES"][:].astype(np.float64)
            self.h2s_log10_sigma_grid = f["LOG10_SIGMA_GRID"][:].astype(np.float64)
            self.h2s_eps = f["EPS"][:].astype(np.float64)
            self.h2s_logsig_mean = float(f["LOGSIG_MEAN"][()])
            self.h2s_logsig_std = float(f["LOGSIG_STD"][()])
        self.h2s_fref = _library_reference_flux(config, "h2s")

        # -- per-batch tabulation (`prepare`, module docstring): unset
        # until a batch is prepared; `log_density` refuses GAL/YSO/H2S
        # until then (rule 6: fail on the impossible, not silently
        # recompute the batch's own exact quadrature per read).
        self._prep_rows = None
        self._prep_log10_a_grid = None
        self._prep_p_a_yso = None
        self._prep_p_a_gal = None
        self._prep_gal_eps_curve = None
        self._prep_h2s_eps_grid = None

    def prepare(self, rows):
        """Tabulates every quantity GAL/YSO/H2S's reads need for exactly
        the sources in `rows` (about ten thousand, `CODING_RULES.md` 10b) --
        called once per batch, before that batch's `log_density` calls
        (module docstring). Distinct `(sightline row, node index)` pairs are
        evaluated once and shared by every source touching them
        (`_build_pair_tables`), never once per source; H2S's selection
        table is built once for every depth group at once, independent of
        which particular sources the batch holds beyond the shared grid."""
        rows = np.asarray(rows, dtype=np.intp)
        uniq_rows = np.unique(rows)

        sl_rows = self.table["HPX256_ROW"][uniq_rows].astype(np.intp)
        node_lo = self.table["NODE_LO"][uniq_rows].astype(np.intp)
        node_w = self.table["NODE_W"][uniq_rows]

        # the tabulation's `log10 a` grid (module docstring): the shared
        # column grid's own measured floor to `_A_GRID_SAFETY` times the
        # largest kernel quadrature node this batch's sources touch, on
        # either side of their own node bracket.
        node_hi = np.clip(node_lo + 1, 0, self.yso_shape.n_node - 1)
        touched_nodes = np.unique(np.concatenate([node_lo, node_hi]))
        t_max = max(float(self.yso_shape.kernel_t["herschel"][touched_nodes].max()),
                    float(self.yso_shape.kernel_t["planck"][touched_nodes].max()))
        log10_a_grid = np.linspace(np.log10(_A_GRID_FLOOR),
                                    np.log10(t_max * _A_GRID_SAFETY), N_A_GRID)
        a_grid = 10.0 ** log10_a_grid

        p_a_yso, p_a_gal = _build_pair_tables(self.config, self.yso_shape, a_grid,
                                              sl_rows, node_lo, node_w)

        # GAL: the exact selection curve `eps(log10 S)`, already divided by
        # Z_GAL, at the source's own adopted-column node bracket (module
        # docstring: the same bracket `gal_z`'s own build-time integral
        # already uses) -- NOT the counts law's own `phi` density: `phi`
        # multiplies at the query's own RAW `log10 S` (`_gal_phi_density_
        # grid`, survey-wide, applied at read time by a plain shared
        # interpolation) while only `eps` is read after the source's own
        # common-mode shift `s` (`PassFractionModel.evaluate`'s own split);
        # folding `phi` into this per-source, shift-indexed table would
        # apply that shift to `phi` too, which the exact formula never does.
        a_col_batch = self.table["A_COL_K"][uniq_rows]
        node_lo_g, node_w_g = column_grid.bracket(a_col_batch, self.gal_model.a_nodes)
        curve = self.gal_model._eps_curve(node_lo_g, node_w_g, self.gal_delta5[uniq_rows])
        z_batch = self.gal_z[uniq_rows]
        safe_z = np.where(z_batch > 0.0, z_batch, 1.0)[:, None]
        p_logs = np.where(z_batch[:, None] > 0.0, curve / safe_z, 0.0)

        # H2S: eps(a, Sigma) node-blended in a at every grid point, for
        # every depth group at once (module docstring).
        node_lo_h, node_w_h = column_grid.bracket(a_grid, self.h2s_a_nodes)
        node_hi_h = np.minimum(node_lo_h + 1, self.h2s_a_nodes.size - 1)
        curve_lo = self.h2s_eps[:, node_lo_h, :]
        curve_hi = self.h2s_eps[:, node_hi_h, :]
        eps_grid_h2s = ((1.0 - node_w_h)[None, :, None] * curve_lo
                        + node_w_h[None, :, None] * curve_hi)

        self._prep_rows = uniq_rows
        self._prep_log10_a_grid = log10_a_grid
        self._prep_p_a_yso = p_a_yso
        self._prep_p_a_gal = p_a_gal
        self._prep_gal_eps_curve = p_logs.astype(np.float32)
        self._prep_h2s_eps_grid = eps_grid_h2s.astype(np.float32)

    def _prep_local_index(self, rows):
        """`(n,)`: `rows`'s own position in the last `prepare`d batch --
        every GAL/YSO/H2S read's one gather into that batch's tables (rule
        6: raise, do not silently recompute, when a row was never
        prepared)."""
        if self._prep_rows is None:
            raise ValueError(
                "SourcePrior.log_density: call prepare(rows) once per batch "
                "before reading class 'gal', 'yso' or 'h2s'")
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
        (module docstring)."""
        if cls not in CLASSES:
            raise ValueError("SourcePrior.log_density: unknown class %r, must be one of %r"
                             % (cls, CLASSES))
        if cls in ("gal", "h2s") and model_index is None:
            raise ValueError("SourcePrior.log_density: class %r needs model_index" % (cls,))
        rows2d, a2, b2, mi2, shp = self._prepare(rows, a, log10_b, model_index)
        if cls in FAMILY_CLASSES:
            out = self._log_density_family(cls, rows2d, a2, b2)
        elif cls == "gal":
            out = self._log_density_gal(rows2d, a2, b2, mi2)
        elif cls == "yso":
            out = self._log_density_yso(rows2d, a2, b2)
        else:
            out = self._log_density_h2s(rows2d, a2, b2, mi2)
        return out.reshape(shp)

    # -----------------------------------------------------------------
    # STAR, AGB, PAHC
    # -----------------------------------------------------------------

    def _log_density_family(self, cls, rows2d, a2, b2):
        shape = self.shapes[cls]
        fm = self.family_models[cls]
        model = fm["model"]
        rows = rows2d.ravel()
        a = a2.ravel()
        b = b2.ravel()
        valid = a > 0.0
        a_safe = np.where(valid, a, 1.0)

        tile_id = self.table["TILE_ID"][rows]
        a_col = self.table["A_COL_K"][rows]
        f_lim8 = (self.table["F_LIM_50_MJY"][rows, self._idx_i4] if cls == "pahc" else None)
        raw = shape.density(a_safe, b, tile_id, a_col, f_lim8=f_lim8)
        p_ab = raw / (a_safe * LN10 * fm["dx"] * fm["db"])
        p_ab = np.where(valid, p_ab, 0.0)

        node_lo, node_w = column_grid.bracket(a_safe, model.a_nodes)
        eps = model.evaluate(node_lo=node_lo, node_w=node_w, log10_b=b,
                             s=fm["s"][rows], delta_5=fm["delta5"][rows])

        z = self.table["Z_%s" % cls.upper()][rows]
        numerator = p_ab * eps
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_val = np.log(numerator) - np.log(z)
        return np.where((numerator > 0.0) & (z > 0.0) & valid, ln_val, -np.inf)

    # -----------------------------------------------------------------
    # GAL
    # -----------------------------------------------------------------

    def _log_density_gal(self, rows2d, a2, b2, mi2):
        """Reads only `prepare`'s tables (module docstring): GAL's own
        `p_a` at the query's `a`; `p(log10 S)` splits into the counts
        law's own density `phi` at the query's RAW `log10 S` (a plain
        shared interpolation, `_gal_phi_density_grid` -- survey-wide, no
        per-source table) and the exact selection `eps` (already divided
        by `Z_GAL`) at `log10 S` shifted by the source's own common-mode
        `s` (`prepare`'s own `_prep_gal_eps_curve`) -- no
        `PassFractionModel` call and no kernel quadrature sum."""
        rows = rows2d.ravel()
        a = a2.ravel()
        b = b2.ravel()
        mi = mi2.ravel()
        valid = a > 0.0
        local = self._prep_local_index(rows)

        log10_s = b + np.log10(self.gal_fref[mi])
        phi_density = np.interp(log10_s, self.gal_log10_s_grid, self._gal_phi_density_grid)
        shifted = log10_s - self.gal_s[rows]
        eps_over_z = _interp_rows(self._prep_gal_eps_curve, self.gal_log10_s_grid, local, shifted)
        p_logs = phi_density * eps_over_z

        log10_a = np.log10(np.clip(a, 1.0e-300, None))
        p_a = _interp_rows(self._prep_p_a_gal, self._prep_log10_a_grid, local, log10_a)
        p_a = np.where(valid, p_a, 0.0)

        numerator = p_a * p_logs
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_val = np.log(numerator)
        return np.where((numerator > 0.0) & valid, ln_val, -np.inf)

    # -----------------------------------------------------------------
    # YSO
    # -----------------------------------------------------------------

    def _log_density_yso(self, rows2d, a2, b2):
        """Reads only `prepare`'s tables (module docstring): the shared
        `p(a | A_s)` at the query's `a`, one `log10 a` interpolation, no
        kernel quadrature sum; the conditional brightness stays the
        closed-form Gaussian it always was."""
        rows = rows2d.ravel()
        a = a2.ravel()
        b = b2.ravel()
        local = self._prep_local_index(rows)

        log10_a = np.log10(np.clip(a, 1.0e-300, None))
        p_a = _interp_rows(self._prep_p_a_yso, self._prep_log10_a_grid, local, log10_a)
        p_a = np.where(a > 0.0, p_a, 0.0)

        mean_b = self.table["RIDGE_INTERCEPT"][rows] + self.table["RIDGE_SLOPE"][rows] * a
        width = self.table["RIDGE_WIDTH"][rows]
        z_score = (b - mean_b) / width
        p_b = _INV_SQRT_2PI / width * np.exp(-0.5 * z_score ** 2)

        numerator = p_a * p_b
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_val = np.log(numerator)
        return np.where((numerator > 0.0) & (a >= 0.0), ln_val, -np.inf)

    # -----------------------------------------------------------------
    # H2S
    # -----------------------------------------------------------------

    def _log_density_h2s(self, rows2d, a2, b2, mi2):
        """Reads only `prepare`'s tables (module docstring): YSO's shared
        `p(a | A_s)` and the `(K, N_A_GRID, 41)` selection table at the
        source's own depth group, one `log10 a` interpolation then one
        `log10 Sigma` interpolation (`_interp_h2s_eps`); the region
        lognormal stays the closed-form Gaussian it always was."""
        rows = rows2d.ravel()
        a = a2.ravel()
        b = b2.ravel()
        mi = mi2.ravel()
        local = self._prep_local_index(rows)

        log10_a = np.log10(np.clip(a, 1.0e-300, None))
        p_a = _interp_rows(self._prep_p_a_yso, self._prep_log10_a_grid, local, log10_a)
        p_a = np.where(a > 0.0, p_a, 0.0)

        log10_sigma = b + np.log10(self.h2s_fref[mi])
        z_score = (log10_sigma - self.h2s_logsig_mean) / self.h2s_logsig_std
        p_sigma = _INV_SQRT_2PI / self.h2s_logsig_std * np.exp(-0.5 * z_score ** 2)

        group_idx = self.table["GROUP"][rows].astype(np.intp)
        eps_val = _interp_h2s_eps(self._prep_h2s_eps_grid, self._prep_log10_a_grid,
                                  self.h2s_log10_sigma_grid, group_idx, log10_a, log10_sigma)

        z = self.table["Z_H2S"][rows]
        numerator = p_a * p_sigma * eps_val
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_val = np.log(numerator) - np.log(z)
        return np.where((numerator > 0.0) & (z > 0.0) & (a >= 0.0), ln_val, -np.inf)


# ---------------------------------------------------------------------------
# report (rules 10, 11, 13): the normalisation identity and the read cost
# ---------------------------------------------------------------------------

#: The number of query points per source per class the timed read-cost
#: probe uses (brief: "9 query points per class").
_N_QUERY_TIMING = 9

#: The library model count the timed read-cost probe uses for every class
#: (brief: "4,066 models" -- the STAR/`sps` library's own count, used as
#: one shared probe size across classes so the six numbers are
#: comparable).
_READ_COST_N_MODEL = 4066

#: `CODING_RULES.md` 10a / the planner's capped-run mandate: the read-cost
#: probe never evaluates more than this many models in one `log_density`
#: call, so `PassFractionModel`'s own `(block, n_a, n_b)`-shaped
#: intermediates (and this callable's own broadcast arrays) stay small
#: regardless of how many models the full probe covers.
_READ_COST_BLOCK = 512

#: `CODING_RULES.md` 10a: the normalisation check's own `(a, log10 B)`
#: mesh is capped at this many points per axis, one source and one class
#: at a time (never a batch over sources or classes at once).
_CHECK_GRID_N = 400

_SURVEY_N_SOURCES = 8.66e6


def _family_grid(shape, a_col):
    """`(a_grid, b_grid)`: a fine grid covering STAR/AGB/PAHC's tabulated
    box plus six cells of the declared analytic tail on every edge (module
    docstring's tail formula decays geometrically past the edge, so six
    cells already carries negligible remaining mass at the shape's own
    cell width) -- `check`'s own normalisation grid."""
    x_lo, x_hi = shape.x_edges[0], shape.x_edges[-1]
    b_lo, b_hi = shape.b_edges[0], shape.b_edges[-1]
    x_cell = float(np.mean(np.diff(shape.x_edges)))
    b_cell = float(np.mean(np.diff(shape.b_edges)))
    log_x = np.linspace(x_lo - 6 * x_cell, x_hi + 6 * x_cell, _CHECK_GRID_N)
    b_grid = np.linspace(b_lo - 6 * b_cell, b_hi + 6 * b_cell, _CHECK_GRID_N)
    a_grid = a_col * (10.0 ** log_x)
    return a_grid, b_grid


def _integrate(prior, cls, row, a_grid, b_grid, model_index=None):
    """`(integral, da, db)`: the trapezoid-rule integral of
    `exp(log_density)` over the rectangular `(a_grid, b_grid)` mesh for
    one source -- `check`'s own numerical acceptance test. `model_index`
    (one fixed model for the whole grid) is passed at shape `(1,)`, never
    repeated to the grid's own length: `log_density`'s own broadcasting
    treats a 1-D `model_index` as "shared across every row" (the brief's
    `(m,)` case), so a length-1 array is the correct, O(1) way to hold one
    model fixed -- repeating it to `n_a . n_b` would instead be read as
    `n_a . n_b` DISTINCT shared models and broadcast against the grid's
    own `n_a . n_b` rows outer-product style (`CODING_RULES.md` 10a: this
    was the callable's own `(n_a n_b)^2` blow-up, not `PassFractionModel`'s)."""
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
    `log10 B` centred on the ridge's own `mean(a) = RIDGE_INTERCEPT +
    RIDGE_SLOPE . a` at every `a_grid` point rather than one fixed window
    (`_integrate`'s plain rectangular mesh mislocates almost all of a
    steep ridge's own conditional mass once `a` ranges far from the
    single `a` a fixed window was centred at -- the shape's own
    correlation between the two axes, `SPEC_PRIORS.md` section 6.3, is
    exactly what a rectangular grid in a FIXED window cannot follow).
    `np.trapz`'s own `x` argument takes the per-row `log10 B` array
    directly, so the sheared mesh integrates exactly like a regular one."""
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


def _extinction_grid(a_col, a_hi):
    """Log-spaced `a` grid from a tiny floor to `a_hi` (`_CHECK_GRID_N`
    points): the extinction marginal every class but the family shapes
    reads (`YsoShape.marginal_at`, GAL's kernel-at-`u=1` reconstruction)
    is sharply peaked toward small `a` for a column this far from the
    source's own `A_s` scale (a foreground-heavy embedding density, or
    the kernel's own near-`A_s` core) -- a LINEAR grid spends almost all
    of its `_CHECK_GRID_N` points where the density is smooth and starves
    the peak; log-spacing is the same fix `star_shapes` uses for its own
    `log10 x` axis. Shares `prepare`'s own `_A_GRID_FLOOR` so the check's
    query never falls below the tabulation's own interpolation range."""
    return np.geomspace(_A_GRID_FLOOR, max(a_hi, 1.0e-5), _CHECK_GRID_N)


def check(config, region, n_sources=50, seed=0):
    """For `n_sources` random catalogue rows and each of the six classes,
    the numerical integral of `exp(log_density)` over a fine `(a, log10
    B)` grid (`_family_grid` for STAR/AGB/PAHC; the kernel's own tabulated
    `T` range for GAL; six ridge/lognormal widths either side of the mean
    for YSO/H2S) against 1 (`10_POSTERIOR.md` section 3's normalisation
    requirement): within 0.02 for STAR/AGB/PAHC/GAL/H2S (`star_shapes.
    EPS_SHAPE`, the same bar their own build-time approximations are
    budgeted against), within 1e-3 for YSO (analytic, no such
    approximation). Reports the worst deviation per class, and the wall
    time of one source's `log_density` call for one class at 4,066 models
    (the STAR/`sps` library's own count) times 9 query points, extrapolated
    linearly to the survey's 8.66e6 sources."""
    prior = SourcePrior(config, region)
    rng = np.random.default_rng(seed)
    n = min(int(n_sources), prior.n_source)
    rows = rng.choice(prior.n_source, size=n, replace=False)
    # GAL/YSO/H2S now read only `prepare`'s tables (module docstring): this
    # check's own batch is exactly the `n` rows it queries.
    prior.prepare(rows)

    worst = {cls: 0.0 for cls in CLASSES}
    for row in rows:
        a_col = float(prior.table["A_COL_K"][row])

        for cls in FAMILY_CLASSES:
            shape = prior.shapes[cls]
            a_grid, b_grid = _family_grid(shape, a_col)
            a_grid = a_grid[a_grid > 0.0]
            integral = _integrate(prior, cls, row, a_grid, b_grid)
            worst[cls] = max(worst[cls], abs(integral - 1.0))

        # GAL: the kernel's own tabulated T range at the source's node
        # bracket sets the a-grid; the counts law's own LOG10_S_GRID,
        # mapped through model 0's reference flux, sets log10 B.
        sl_rows = prior.table["HPX256_ROW"][row:row + 1].astype(np.intp)
        node_lo = prior.table["NODE_LO"][row:row + 1].astype(np.intp)
        node_hi = np.minimum(node_lo + 1, prior.yso_shape.n_node - 1)
        t_lo, _ = prior.yso_shape._gather_quadrature(sl_rows, node_lo)
        t_hi, _ = prior.yso_shape._gather_quadrature(sl_rows, node_hi)
        a_grid_gal = _extinction_grid(a_col, max(t_lo.max(), t_hi.max()) * 3.0)
        b_grid_gal = prior.gal_log10_s_grid - np.log10(prior.gal_fref[0])
        integral = _integrate(prior, "gal", row, a_grid_gal, b_grid_gal, model_index=0)
        worst["gal"] = max(worst["gal"], abs(integral - 1.0))

        # YSO: the sheared mesh (`_integrate_yso`) follows the ridge's own
        # a-dependent mean; H2S's lognormal in log10 Sigma does not
        # depend on a, so its own window is a fixed six-sigma band.
        a_grid_cloud = _extinction_grid(a_col, max(t_lo.max(), t_hi.max()) * 3.0)
        integral = _integrate_yso(prior, row, a_grid_cloud)
        worst["yso"] = max(worst["yso"], abs(integral - 1.0))

        log10_sigma_lo = prior.h2s_logsig_mean - 6 * prior.h2s_logsig_std
        log10_sigma_hi = prior.h2s_logsig_mean + 6 * prior.h2s_logsig_std
        b_grid_h2s = (np.linspace(log10_sigma_lo, log10_sigma_hi, _CHECK_GRID_N)
                     - np.log10(prior.h2s_fref[0]))
        integral = _integrate(prior, "h2s", row, a_grid_cloud, b_grid_h2s, model_index=0)
        worst["h2s"] = max(worst["h2s"], abs(integral - 1.0))

    # -- the read cost: one source x 4,066 models x 9 query points per
    # class, extrapolated to the survey (rule 10: "time it on real data"),
    # the 4,066 covered in blocks of `_READ_COST_BLOCK` (CODING_RULES.md
    # 10a / the planner's capped-run mandate) so no single `log_density`
    # call this probe makes ever queries more than one block's worth of
    # models at once.
    read_cost = {}
    probe_row = int(rows[0])
    # `rows_probe` is ONE source (n=1): `a_probe`/`b_probe` are given as
    # explicit (1, m) arrays (the brief's "(n, m)" form) so `log_density`'s
    # own broadcasting never mistakes the m query points for m sources
    # (`_integrate`'s own docstring names the same outer-product trap for
    # `model_index`; here it is `a`/`log10_b` that must stay 2-D, one row,
    # not a length-m array `_prepare` would otherwise read as m sources).
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

    return worst, read_cost


def report(region, worst, read_cost):
    lines = ["prior.callable: %s: normalisation check (bar 0.02 all but YSO 1e-3)" % region]
    for cls in CLASSES:
        bar = 1.0e-3 if cls == "yso" else 0.02
        lines.append("prior.callable: %s: %s: worst |integral - 1| = %.4g (bar %.4g)%s"
                     % (region, cls, worst[cls], bar,
                        "" if worst[cls] <= bar else "  ** EXCEEDS BAR **"))
    lines.append("prior.callable: %s: read cost (one source x N models x %d query points)"
                 % (region, _N_QUERY_TIMING))
    for cls in CLASSES:
        rc = read_cost[cls]
        lines.append("prior.callable: %s: %s: n_model=%d wall=%.4gs -> "
                     "survey extrapolation (8.66e6 sources) = %.3g hours"
                     % (region, cls, rc["n_model"], rc["wall_s"], rc["survey_hours"]))
    return lines


if __name__ == "__main__":
    import sys

    from sesnaimpute import config as _config_module

    cfg = _config_module.load(sys.argv[1])
    for _region in sys.argv[2:]:
        _worst, _read_cost = check(cfg, _region)
        for _line in report(_region, _worst, _read_cost):
            print(_line, flush=True)
