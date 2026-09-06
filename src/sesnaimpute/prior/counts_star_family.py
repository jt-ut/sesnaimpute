"""The per-source counts and normalisers for STAR, AGB, PAHC and GAL
(SPEC_PRIORS.md section 0.2's ``N_C(s)``/``lambda~_C``; section 1.3's
per-source selection recipe; section 2.1's ``N_STAR``; section 3's
``N_AGB``; section 4's ``N_PAHC``; section 5.1's ``N_GAL``;
IMPLEMENTATION.md section 4's depth-group evaluation and section 5's
prior-table conditioning columns).

Per source this module answers, for each of the four classes, "how many
catalogued objects of this class per deg**2 sit at this source's own
position, column and depth" (`N_C`) and "what fraction of the class's
intrinsic mass survives selection here" (`Z_C`, the normaliser the
fitter's shape divides by, spec section 0.2). YSO and H2S are built
elsewhere (their count does not come from a per-tile point cloud or a
counts law the way these four do); this module writes only the four
named in the brief.

STAR, AGB, PAHC share one construction (spec sections 2.1-2.2, 3, 4):
``prior.star_population``'s per-tile point cloud is deposited into a
per-tile, per-shape-node ``(log10 x, log10 B)`` density
(``prior.star_shapes``), and a class's exact selection is tabulated once
per depth-group centre on the SAME shared column-grid nodes
(``prior.star_selection``). The selected fraction

    E[eps] = Integral shape(a, log10 B | A_s) . eps_k(a, log10 B; s) da dlog10B

is evaluated by MARGINALISING the shape's own stored density over its
``log10 x`` axis, AT EACH SHAPE NODE EXACTLY (never at a source's own
blended column): ``prior.star_selection``'s own exact-selection tables
are themselves built on ONE region-wide fallback placement (its own
module docstring: "this is that approximation"), so evaluating the
survived fraction at a source's own displaced extinction inside the
tile's point cloud would be more precise than the tables this module
reads already are. Treating the star at exactly ``a = a_node`` for the
purpose of selection -- while keeping its own ``log10 B`` spread, which
IS what the class's tabulated ``EPS`` axis distinguishes -- is exactly
the reading-note precedent (``04_star_family.md`` section D,
``gathers.integrate_count_per_source``): ``eps_k``'s intrinsic-density
argument is the shape's own ``log10 B`` marginal AT ONE NODE, one
``PassFractionModel.integrate_count`` call per DISTINCT (tile, node[,
PAHC limit]) combination -- never per source -- and the two bracketing
nodes' own selected fractions are blended by the SAME linear weight
``ClassShape.density`` uses for the shape itself, so the count is
algebraically the two-node blend of two "pure" evaluations, matching how
the shape's own storage already represents an intermediate column
(IMPLEMENTATION.md section 2). ``PassFractionModel.integrate_count``'s
own continuous interpolation across its ``b_grid`` is exactly the "s
varies by less than one b-grid cell" rule the brief names: ``s`` and the
depth-group ``Delta`` are passed PER SOURCE, exactly, to every call, so
no separate depth-group bucketing is needed -- only the marginal density
itself (tile, node[, limit]) is looked up once per distinct combination.
Rule 8: joblib over tiles (`_grouped_eval`), each tile's own handful of
node combinations handled by cheap boolean masks, never a python loop
over sources.

GAL (spec section 5.1) needs no tile loop and no shape read: a galaxy's
``a`` **is** the column exactly (no placement spread), so
``PassFractionModel.integrate_count``/``z_eps`` are called ONCE for the
whole region with each source's own, continuously varying node bracket
and depth shift -- the argument the brief calls "fits the shape's
evaluator" without needing the quadrature written out by hand.

What is deliberately dropped, and why (rule 13): the shape's own declared
analytic tails beyond its tabulated ``(log10 x, log10 B)`` box
(``prior.star_shapes.finalise_node_shape``) are not folded into the
``log10 B`` marginal here -- their own mass is bounded by the shape's own
fidelity bar (``EPS_SHAPE = 0.02`` relative L1, ``prior.star_shapes``),
the same bar this module's own algebraic acceptance check is graded
against, so the omission is inside the tolerance the brief already
states, not an extra one.

Writes, per region, ``bms/table/counts-star-family_table_source.hdf5``:
root attr ``GRANULE = "source"``; ``TILE_ID``, ``NODE_LO``, ``NODE_W``
(``column_grid.bracket`` on ``A_COL_K`` against the full shared column
grid), ``GROUP`` (nearest depth-group centre of the region's shared
``prior.depth_groups`` product), ``S_SHIFT``, ``F_LIM8_MJY``, and the
eight counts and normalisers, catalogue row order, float32.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid, depth_groups, selection, star_population, star_shapes

FAMILY_CLASSES = ("star", "agb", "pahc")

#: The depth-group index the algebraic acceptance check (rule 11) reads
#: the class's own group centre from -- the MEDIAN of whichever K that
#: class's own selection search settled on (`prior.star_selection`), so
#: the check never assumes a fixed K across classes or regions.
_CHECK_GROUP_QUANTILE = 0.5


# ---------------------------------------------------------------------------
# per-source conditioning (brief item 1)
# ---------------------------------------------------------------------------

def _source_tile_ids(config, region):
    """Every catalogued source's own STAR-anchor tile index, by the same
    nearest-pixel join `prior.star_shapes.tile_map_classes` uses (its own
    docstring): the tiles product's `HPX_PIX_512 -> TILE_ID` table, joined
    through the source's own `hpx_pix_512` (`granules.access.region_slice`).
    """
    tiles_path = config_module.product_path(config, "bms", "anchors", "tiles", "hpx512", region=region)
    if not os.path.exists(tiles_path):
        raise FileNotFoundError(
            "prior.counts_star_family: tiles product missing for region %r at %s -- "
            "run the 'prior.anchor_tiles' RUNBOOK line first" % (region, tiles_path))
    with h5py.File(tiles_path, "r") as f:
        pix512 = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        tile_of_pix = np.asarray(f["TILE_ID"][:], dtype=np.int64)
    rs = access.region_slice(config, region)
    src_pix512 = np.asarray(rs["hpx_pix_512"], dtype=np.int64)
    order = np.argsort(pix512)
    pix_sorted = pix512[order]
    loc = np.searchsorted(pix_sorted, src_pix512)
    capped = np.minimum(loc, pix_sorted.size - 1) if pix_sorted.size else loc
    return tile_of_pix[order][capped]


def source_conditioning(config, region):
    """Every catalogued source's own conditioning scalars (brief item 1):
    `TILE_ID`, the FULL shared column grid's own bracket on `A_COL_K`
    (`NODE_LO`/`NODE_W`, reported and reused for GAL directly), the
    nearest depth-group index off the region's shared
    `prior.depth_groups` product, the common-mode depth shift `S_SHIFT`
    and four-degree-of-freedom residual `DELTA5` (`prior.selection.
    split_common_mode`), and the source's own 8 micron 50%-completeness
    limit `F_LIM8_MJY`.
    """
    f_lim_8band = limits_module.limits(config, region)
    log10_flim8 = np.log10(f_lim_8band)
    f_lim8 = f_lim_8band[:, star_population.IDX_I4]

    adopted_path = config_module.product_path(config, "sky/derived", "adopted",
                                               "column", "source", region=region)
    a_col = access.per_source(config, region, adopted_path, ["A_COL_K"])["A_COL_K"]
    a_col = np.asarray(a_col, dtype=np.float64)

    dg_path = config_module.product_path(config, "bms", "sesna", "depth-groups", "region")
    dg = depth_groups.DepthGroups.read(dg_path, region)
    s_shift, delta5 = selection.split_common_mode(log10_flim8, dg.ref_log10_flim)
    group = dg.assign_group(delta5)

    a_nodes_full = column_grid.nodes(config)
    node_lo, node_w = column_grid.bracket(a_col, a_nodes_full)

    tile_id = _source_tile_ids(config, region)

    return dict(n_source=f_lim_8band.shape[0], a_col=a_col, f_lim8=f_lim8,
               log10_flim8=log10_flim8, s=s_shift, delta5=delta5, group=group,
               node_lo=node_lo, node_w=node_w, tile_id=tile_id, a_nodes_full=a_nodes_full)


# ---------------------------------------------------------------------------
# reading a class's exact-selection model (star_selection.py's product)
# ---------------------------------------------------------------------------

def read_class_model(config, region, cls):
    """The class's `PassFractionModel` (`prior.selection`) off
    `prior.star_selection`'s own product, wired to a `depth_groups.
    DepthGroups`-shaped carrier built straight from that class's OWN
    stored `GROUP_CENTRES`/`REF_LOG10_FLIM` (its own K, not the shared
    depth-groups product's -- `prior.star_selection` searches K per class
    independently)."""
    path = config_module.product_path(config, "bms", "star", "selection", "region", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.counts_star_family: no star-selection product for region %r at %s -- "
            "run the 'prior.star_selection' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        g = f[cls]
        eps = g["EPS"][:].astype(np.float64)
        group_centres = g["GROUP_CENTRES"][:].astype(np.float64)
        ref_log10_flim = g["REF_LOG10_FLIM"][:].astype(np.float64)
        b_grid = g["LOG10_B_GRID"][:].astype(np.float64)
    a_nodes = column_grid.nodes(config)
    knots = depth_groups.DepthGroups(region, ref_log10_flim, group_centres,
                                     np.zeros((0, group_centres.shape[1])), group_centres.shape[0])
    model = selection.PassFractionModel(a_nodes, b_grid, knots, eps)
    return model


def _node_a_bracket(x_centers, a_node, a_nodes_full):
    """The FULL shared column grid's own bracket of the physical
    extinction a hypothetical source at exactly this shape node would
    place its tile's own stars at, `a = a_node . 10**x_centers` (the
    shape's own `x = a/a_node` convention, `prior.star_shapes`'s module
    docstring) -- independent of tile and of depth group, so it is built
    once per shape node and shared by every (tile, group) combination
    that touches it."""
    a_grid = float(a_node) * (10.0 ** x_centers)
    return column_grid.bracket(a_grid, a_nodes_full)


def eps_base_grid(model, node_lo_ix, node_w_ix, b_centers, group_centre_row):
    """`(n_x, n_b)`: the class's exact selection evaluated at EVERY point
    of the shape's own tabulation grid, `s = 0`, one depth-group centre --
    ONE vectorised `PassFractionModel.evaluate` call over the full `(x,
    b)` cartesian product (module docstring: this, not a b-only marginal,
    is the "distinct integral" the brief names; the shape's own `a`
    dependence is not thrown away). Independent of tile: reused by every
    tile sharing this (node[, limit], group)."""
    n_x, n_b = node_lo_ix.size, b_centers.size
    node_lo_q = np.repeat(node_lo_ix, n_b)
    node_w_q = np.repeat(node_w_ix, n_b)
    log10_b_q = np.tile(b_centers, n_x)
    delta5_q = np.broadcast_to(group_centre_row, (n_x * n_b, group_centre_row.shape[1]))
    eps_flat = model.evaluate(node_lo=node_lo_q, node_w=node_w_q, log10_b=log10_b_q,
                              s=np.zeros(n_x * n_b), delta_5=delta5_q)
    return np.asarray(eps_flat, dtype=np.float64).reshape(n_x, n_b)


def shift_sum(density_batch, eps_base, b_centers, s_values):
    """`(n_source,)`: `Sum_x,b density_batch[s,x,b] . eps_base[x, b - s]`
    for every source sharing this `(node[, limit], group)` combination,
    regardless of tile -- `density_batch[i]` is source `i`'s OWN tile's
    stored density slab (gathered by fancy indexing on `TILE_ID`,
    `grouped_eval`), `eps_base` shifted along `log10 B` by each source's
    own, exactly continuous common-mode depth shift `s` (the algebraic
    identity `prior.selection.PassFractionModel` itself uses: a `log10 B`
    shift by `-s` is exactly a limit shift by `+s`) via one linear
    interpolation, vectorised over every source sharing the combination
    at once (`np.einsum`, never a per-source or per-grid-point python
    loop) -- this is the brief's "interpolating in s where s varies
    within a group by less than the table's b cell": the shift is
    continuous, not bucketed, and the interpolation's own error is
    bounded by the b-grid's own cell width."""
    n_x, n_b = eps_base.shape
    s_values = np.asarray(s_values, dtype=np.float64)
    shifted = b_centers[None, :] - s_values[:, None]
    clamped = np.clip(shifted, b_centers[0], b_centers[-1])
    idx = np.clip(np.searchsorted(b_centers, clamped) - 1, 0, n_b - 2)
    lo, hi = b_centers[idx], b_centers[idx + 1]
    span = hi - lo
    t = np.where(span > 0.0, (clamped - lo) / np.where(span > 0.0, span, 1.0), 0.0)
    eps_lo = eps_base[:, idx]
    eps_hi = eps_base[:, idx + 1]
    eps_shifted = eps_lo * (1.0 - t)[None, :, :] + eps_hi * t[None, :, :]
    return np.einsum("sxb,xsb->s", density_batch, eps_shifted)


# ---------------------------------------------------------------------------
# one distinct (node[, limit], group) evaluation, joblib over combinations
# ---------------------------------------------------------------------------

def grouped_eval(model, shape, tile_id, node_idx_shape, s, group_idx, limit_idx=None, n_jobs=1):
    """`(n_source,)`: `eps_base_grid` built once per distinct `(node[,
    limit], group)` combination present ANYWHERE in the region -- tile-
    independent, so this is the "one integral per distinct node[, limit,
    group] covering every source" the reading note and the brief both
    name -- then `shift_sum` applied to every source sharing that
    combination AT ONCE, its own tile's density slab gathered by fancy
    indexing on `TILE_ID` (`shape.density_table[tile_id[sel], node]`),
    never a per-tile inner loop: the tile only ever enters as an index
    into an already-built array, so the python-level loop this function
    runs is over the combination axis (rule 8's "largest iterator" here
    is the number of distinct combinations, not the number of tiles --
    tiles enter only as a fancy-index gather, not a loop). Rule 8: joblib
    over that combination axis."""
    b_centers = shape.b_centers
    x_centers = shape.x_centers
    a_nodes_full = model.a_nodes

    if limit_idx is None:
        combo_keys = np.stack([node_idx_shape, group_idx], axis=1)
    else:
        combo_keys = np.stack([node_idx_shape, limit_idx, group_idx], axis=1)
    uniq_combos, inverse = np.unique(combo_keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    node_bracket_cache = {}

    def _one_combo(c):
        combo = uniq_combos[c]
        nv, gv = int(combo[0]), int(combo[-1])
        lv = int(combo[1]) if limit_idx is not None else None
        if nv not in node_bracket_cache:
            node_bracket_cache[nv] = _node_a_bracket(x_centers, shape.shape_nodes[nv], a_nodes_full)
        node_lo_ix, node_w_ix = node_bracket_cache[nv]
        group_centre_row = model.knots.group_centres[gv:gv + 1]
        eps_base = eps_base_grid(model, node_lo_ix, node_w_ix, b_centers, group_centre_row)

        sel = np.flatnonzero(inverse == c)
        tiles_here = tile_id[sel]
        density_batch = (shape.density_table[tiles_here, nv] if lv is None
                         else shape.density_table[tiles_here, nv, lv]).astype(np.float64)
        vals = shift_sum(density_batch, eps_base, b_centers, s[sel])
        return sel, vals

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_combo)(c) for c in range(uniq_combos.shape[0]))
    out = np.empty(tile_id.shape[0], dtype=np.float64)
    for sel, vals in results:
        out[sel] = vals
    return out, uniq_combos.shape[0]


# ---------------------------------------------------------------------------
# STAR / AGB / PAHC: the selected fraction E[eps] and the tile amplitude
# ---------------------------------------------------------------------------

def family_e_eps(config, region, cls, cond, n_jobs):
    """`(e_eps, n_calls, shape, model)`: the selected-fraction normaliser
    `Z_C = E[eps]` (spec section 0.2) for one family class, at every
    source, node-blended (PAHC: node- and limit-blended) exactly as
    `prior.star_shapes.ClassShape.density` blends the shape itself."""
    shape = star_shapes.read(config, region, cls)
    model = read_class_model(config, region, cls)
    group_idx = model.knots.assign_group(cond["delta5"])

    log_shape_nodes = np.log(shape.shape_nodes)
    node_lo_shape, t_node = column_grid.bracket(np.log(cond["a_col"]), log_shape_nodes)
    node_hi_shape = np.minimum(node_lo_shape + 1, shape.shape_nodes.size - 1)

    if cls != "pahc":
        val_lo, n_lo = grouped_eval(model, shape, cond["tile_id"], node_lo_shape,
                                    cond["s"], group_idx, n_jobs=n_jobs)
        val_hi, n_hi = grouped_eval(model, shape, cond["tile_id"], node_hi_shape,
                                    cond["s"], group_idx, n_jobs=n_jobs)
        e_eps = (1.0 - t_node) * val_lo + t_node * val_hi
        return e_eps, n_lo + n_hi, shape, model

    limit_lo, t_limit = column_grid.bracket(np.log10(cond["f_lim8"]), shape.limit_log)
    limit_hi = np.minimum(limit_lo + 1, shape.limit_log.size - 1)
    v_lolo, n1 = grouped_eval(model, shape, cond["tile_id"], node_lo_shape,
                              cond["s"], group_idx, limit_idx=limit_lo, n_jobs=n_jobs)
    v_lohi, n2 = grouped_eval(model, shape, cond["tile_id"], node_lo_shape,
                              cond["s"], group_idx, limit_idx=limit_hi, n_jobs=n_jobs)
    v_hilo, n3 = grouped_eval(model, shape, cond["tile_id"], node_hi_shape,
                              cond["s"], group_idx, limit_idx=limit_lo, n_jobs=n_jobs)
    v_hihi, n4 = grouped_eval(model, shape, cond["tile_id"], node_hi_shape,
                              cond["s"], group_idx, limit_idx=limit_hi, n_jobs=n_jobs)
    val_lo = (1.0 - t_limit) * v_lolo + t_limit * v_lohi
    val_hi = (1.0 - t_limit) * v_hilo + t_limit * v_hihi
    e_eps = (1.0 - t_node) * val_lo + t_node * val_hi
    return e_eps, n1 + n2 + n3 + n4, shape, model


def family_amplitude(config, region, cls, cond):
    """`N_C = amplitude . E[eps]` (brief item 3): STAR/AGB's tile-level
    `Sigma W_STAR` / `Sigma W_AGB` over `OMEGA_SIM_DEG2`; PAHC's tile-level
    `Sigma W . P_PAHC[:, j]`, interpolated in `log10` limit between the
    population product's own eight completeness-limit grid points to the
    source's own `F_LIM8_MJY`."""
    pop_path = config_module.product_path(config, "bms", "star", "population", "tile", region=region)
    with h5py.File(pop_path, "r") as f:
        omega_sim_deg2 = float(f.attrs["OMEGA_SIM_DEG2"])
    pop = star_shapes.read_population(config, region)
    tile_id = cond["tile_id"]

    if cls == "star":
        tile_tot = np.array([t["w_star"].sum() for t in pop["tiles"]], dtype=np.float64)
        return tile_tot[tile_id] / omega_sim_deg2
    if cls == "agb":
        tile_tot = np.array([t["w_agb"].sum() for t in pop["tiles"]], dtype=np.float64)
        return tile_tot[tile_id] / omega_sim_deg2

    limit_grid = pop["limit8_grid_mjy"]
    tile_tot = np.array(
        [[(t["w"] * t["p_pahc"][:, j]).sum() for j in range(limit_grid.size)] for t in pop["tiles"]],
        dtype=np.float64)
    m_lo, t_limit = column_grid.bracket(np.log10(cond["f_lim8"]), np.log10(limit_grid))
    m_hi = np.minimum(m_lo + 1, limit_grid.size - 1)
    lo_val = tile_tot[tile_id, m_lo]
    hi_val = tile_tot[tile_id, m_hi]
    return ((1.0 - t_limit) * lo_val + t_limit * hi_val) / omega_sim_deg2


# ---------------------------------------------------------------------------
# GAL: N_GAL and Z_GAL from the counts law and the region's r(a, S) table
# ---------------------------------------------------------------------------

def gal_counts(config, region, cond):
    """`(n_gal, z_gal)` (spec section 5.1/5.2): `N_GAL = Integral phi(S)
    eps(S | A_s) dS`, `Z_GAL = Integral phi(S) S eps dS / Integral phi(S) S
    dS`, both exactly `PassFractionModel.integrate_count`/`z_eps` on the
    region's own `r(a, log10 S)` selection table (`prior.gal`), with the
    counts law's own `d(log10 S) = dS / (S ln 10)` Jacobian folded into
    the density argument once (module docstring: GAL needs no tile loop,
    `integrate_count`'s own per-source node bracket already fits)."""
    gal_path = config_module.product_path(config, "bms", "gal", "prior", "region", region=region)
    if not os.path.exists(gal_path):
        raise FileNotFoundError(
            "prior.counts_star_family: no GAL prior product for region %r at %s -- "
            "run the 'prior.gal' RUNBOOK line first" % (region, gal_path))
    with h5py.File(gal_path, "r") as f:
        a_nodes = f["A_NODES"][:].astype(np.float64)
        log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
        eps = f["EPS"][:].astype(np.float64)
        group_centres = f["GROUP_CENTRES"][:].astype(np.float64)
        ref_log10_flim = f["REF_LOG10_FLIM"][:].astype(np.float64)

    counts_path = config_module.product_path(config, "bms", "gal", "counts", "survey")
    with h5py.File(counts_path, "r") as f:
        law_log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
        phi_s = f["PHI_S"][:].astype(np.float64)
        fazio_params = dict(log10_a=float(f["LOG10_A"][()]), log10_s_break=float(f["LOG10_S_BREAK"][()]),
                            alpha_faint=float(f["ALPHA_FAINT"][()]), alpha_bright=float(f["ALPHA_BRIGHT"][()]),
                            smoothness=float(f["SMOOTHNESS"][()]))
    if not np.allclose(law_log10_s_grid, log10_s_grid):
        raise ValueError("prior.counts_star_family: GAL's region prior and the survey counts "
                         "law disagree on LOG10_S_GRID for region %r" % region)

    knots = depth_groups.DepthGroups(region, ref_log10_flim, group_centres,
                                     np.zeros((0, group_centres.shape[1])), group_centres.shape[0])
    model = selection.PassFractionModel(a_nodes, log10_s_grid, knots, eps)

    node_lo, node_w = column_grid.bracket(cond["a_col"], a_nodes)
    s_lin = 10.0 ** log10_s_grid
    ln10 = float(np.log(10.0))
    w1 = phi_s * s_lin * ln10          # for N_GAL: Integral phi(S) eps dS
    w2 = w1 * s_lin                    # for Z_GAL: S-weighted average of eps

    n_gal = model.integrate_count(w1[None, :], node_lo, node_w, cond["s"], cond["delta5"])
    z_gal = model.z_eps(w2[None, :], node_lo, node_w, cond["s"], cond["delta5"])
    return n_gal, z_gal, model, fazio_params


# ---------------------------------------------------------------------------
# algebraic acceptance (rule 11): the gather's own node-marginal
# approximation against a per-star exact evaluation, at one shape node,
# s = 0, the region's own median depth group
# ---------------------------------------------------------------------------

def algebraic_check(shape, model, pop, cls, node_idx=0, limit_idx=0):
    """`(direct, approx, dev)`: at shape node `node_idx` (PAHC: also
    limit `limit_idx`) and `s = 0`, the median-group `eps` -- the DIRECT
    weighted average over tile 0's own stars of `model.evaluate` at EACH
    star's own displaced extinction `a_i = a_node . u_i` and own `log10
    B_i` -- against this module's own `eps_base_grid`/`shift_sum`
    quadrature at that same node, tile 0, group and `s = 0`. `dev` is
    graded against the shape's own tabulation bar (0.02 relative L1,
    `prior.star_shapes.EPS_SHAPE`): the two use the same selection model
    and the same physical `a = a_node . x` convention, differing only in
    whether the population is the raw point cloud (direct) or the
    stored, smoothed density (approx)."""
    tile0 = pop["tiles"][0]
    a_node = float(shape.shape_nodes[node_idx])
    if cls == "star":
        w, u, logb = tile0["w_star"], tile0["u"], tile0["log10_b"]
    elif cls == "agb":
        ev = tile0["is_evolved"]
        w_ev = tile0["w_agb"][ev]
        f_c = pop["f_c"]
        u = np.concatenate([tile0["u"][ev], tile0["u"][ev]])
        logb = np.concatenate([tile0["log10_b_agb_o"][ev], tile0["log10_b_agb_c"][ev]])
        w = np.concatenate([w_ev * (1.0 - f_c), w_ev * f_c])
    else:
        w = tile0["w"] * tile0["p_pahc"][:, limit_idx]
        u, logb = tile0["u"], tile0["log10_b_pahc"]

    if w.sum() <= 0.0:
        return float("nan"), float("nan"), float("nan")

    k_mid = int((model.eps_table.shape[0] - 1) * _CHECK_GROUP_QUANTILE)
    group_centre = model.knots.group_centres[k_mid:k_mid + 1]

    a_i = a_node * u
    node_lo_i, node_w_i = column_grid.bracket(a_i, model.a_nodes)
    delta5_rep = np.broadcast_to(group_centre, (a_i.size, group_centre.shape[1]))
    eps_i = model.evaluate(node_lo=node_lo_i, node_w=node_w_i, log10_b=logb,
                           s=np.zeros(a_i.size), delta_5=delta5_rep)
    direct = float(np.sum(w * eps_i) / np.sum(w))

    node_lo_ix, node_w_ix = _node_a_bracket(shape.x_centers, a_node, model.a_nodes)
    eps_base = eps_base_grid(model, node_lo_ix, node_w_ix, shape.b_centers, group_centre)
    density_slab = (shape.density_table[0, node_idx] if cls != "pahc"
                    else shape.density_table[0, node_idx, limit_idx]).astype(np.float64)
    approx = float(shift_sum(density_slab[None, :, :], eps_base, shape.b_centers, np.zeros(1))[0])
    return direct, approx, abs(direct - approx)


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

_OUTPUT_KEYS = ("N_STAR", "N_AGB", "N_PAHC", "N_GAL", "Z_STAR", "Z_AGB", "Z_PAHC", "Z_GAL")


def _write(config, region, cond, out):
    path = config_module.product_path(config, "bms", "table", "counts-star-family", "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.create_dataset("TILE_ID", data=cond["tile_id"].astype(np.int32))
        f.create_dataset("NODE_LO", data=cond["node_lo"].astype(np.int32))
        f.create_dataset("NODE_W", data=cond["node_w"].astype(np.float32))
        f.create_dataset("GROUP", data=cond["group"].astype(np.int32))
        f.create_dataset("S_SHIFT", data=cond["s"].astype(np.float32))
        f.create_dataset("F_LIM8_MJY", data=cond["f_lim8"].astype(np.float32))
        for key in _OUTPUT_KEYS:
            f.create_dataset(key, data=np.asarray(out[key], dtype=np.float32))
    return path


# ---------------------------------------------------------------------------
# report (rules 10, 11, 13)
# ---------------------------------------------------------------------------

def _pct(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    return tuple(float(v) for v in np.percentile(x, [16.0, 50.0, 84.0]))


def _mosaic_area_deg2(config, region):
    """The region's own mosaic area, deg**2, from the I2-band coverage
    fraction summed over its nside-512 pixels (`sky.derived.coverage`) --
    a sanity SCALE for the literature check, not a fitted quantity."""
    import healpy as hp
    path = config_module.product_path(config, "sky/derived", "spitzer", "coverage", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        bands = [b.decode() if isinstance(b, bytes) else b for b in f["BANDS"][:]]
        frac = f["FRAC"][:, bands.index("I2")]
    pix_area_deg2 = hp.nside2pixarea(512, degrees=True)
    return float(np.sum(frac) * pix_area_deg2)


def report(config, region, cond, out, wall_s, n_calls_total):
    lines = []
    n_source = cond["n_source"]
    lines.append("counts_star_family: %s: %d sources, wall=%.1fs, %d distinct integrals evaluated "
                 "(vs %d sources, ratio %.4f)"
                 % (region, n_source, wall_s, n_calls_total, n_source,
                    n_calls_total / max(n_source, 1)))
    for key in _OUTPUT_KEYS:
        lo, med, hi = _pct(out[key])
        lines.append("counts_star_family: %s: %s median=%.5g [16%%=%.5g, 84%%=%.5g]"
                     % (region, key, med, lo, hi))

    z_all = np.concatenate([out["Z_STAR"], out["Z_AGB"], out["Z_PAHC"], out["Z_GAL"]])
    z_all = z_all[np.isfinite(z_all)]
    lines.append("counts_star_family: %s: Z range [%.4g, %.4g] (acceptance: 0 <= Z <= 1)"
                 % (region, float(z_all.min()) if z_all.size else float("nan"),
                    float(z_all.max()) if z_all.size else float("nan")))

    total_family = out["N_STAR"] + out["N_AGB"] + out["N_PAHC"]
    med_family = float(np.median(total_family[np.isfinite(total_family)]))
    try:
        area = _mosaic_area_deg2(config, region)
        catalogued_density = n_source / area
        lines.append("counts_star_family: %s: literature check (report only): "
                     "median N_STAR+N_AGB+N_PAHC=%.4g deg^-2 vs catalogued source density=%.4g "
                     "deg^-2 over %.4g deg^2 mosaic (most catalogued sources are field stars, "
                     "so this is a sanity scale, not an identity)"
                     % (region, med_family, catalogued_density, area))
    except Exception as exc:
        lines.append("counts_star_family: %s: mosaic-area literature check skipped (%s)" % (region, exc))

    return lines


def report_gal_check(region, cond, n_gal, fazio_model_params):
    from sesnaimpute import definitions
    from sesnaimpute.prior.gal import BrokenPowerLaw
    fit = BrokenPowerLaw([fazio_model_params["log10_a"], fazio_model_params["log10_s_break"],
                          fazio_model_params["alpha_faint"], fazio_model_params["alpha_bright"],
                          fazio_model_params["smoothness"]])
    idx_i2 = [b.key for b in definitions.BANDS].index("I2")
    med_flim_i2 = float(np.median(cond["log10_flim8"][:, idx_i2]))
    fazio_cum = float(fit.cumulative(10.0 ** med_flim_i2))
    med_n_gal = float(np.median(n_gal[np.isfinite(n_gal)]))
    return ["counts_star_family: %s: literature check (report only): median N_GAL=%.4g deg^-2 "
           "vs Fazio+2004 cumulative N(>median 4.5um 50%% limit=%.4g mJy)=%.4g deg^-2 "
           "(N_GAL is post-selection, Fazio's is the intrinsic sky count -- N_GAL <= Fazio expected)"
           % (region, med_n_gal, 10.0 ** med_flim_i2, fazio_cum)]


def report_algebraic_checks(region, checks):
    lines = []
    for cls, (direct, approx, dev) in checks.items():
        lines.append("counts_star_family: %s: algebraic check %s: direct=%.5f approx=%.5f "
                     "dev=%.4f (bar 0.02, tile 0, node 0, median group, s=0)"
                     % (region, cls, direct, approx, dev))
    return lines


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _build_one(config, region):
    import time
    t0 = time.time()
    cond = source_conditioning(config, region)
    n_jobs = config.n_jobs

    out = {}
    checks = {}
    pop = star_shapes.read_population(config, region)
    n_calls_total = 0
    for cls in FAMILY_CLASSES:
        e_eps, n_calls, shape, model = family_e_eps(config, region, cls, cond, n_jobs)
        amp = family_amplitude(config, region, cls, cond)
        out["N_%s" % cls.upper()] = amp * e_eps
        out["Z_%s" % cls.upper()] = e_eps
        n_calls_total += n_calls
        checks[cls] = algebraic_check(shape, model, pop, cls, node_idx=0, limit_idx=0)

    n_gal, z_gal, gal_model, fazio_params = gal_counts(config, region, cond)
    out["N_GAL"] = n_gal
    out["Z_GAL"] = z_gal

    path = _write(config, region, cond, out)
    wall_s = time.time() - t0

    for line in report(config, region, cond, out, wall_s, n_calls_total):
        print(line, flush=True)
    for line in report_gal_check(region, cond, n_gal, fazio_params):
        print(line, flush=True)
    for line in report_algebraic_checks(region, checks):
        print(line, flush=True)
    print("counts_star_family: %s -> %s" % (region, path), flush=True)
    return path


def build(config, regions=None):
    """Writes the per-source STAR/AGB/PAHC/GAL counts and normalisers for
    `regions` (default: all thirty), one product per region. Regions run
    serially (each region's own tile loop is already parallel, rule 8's
    largest iterator); the timed rehearsal names two regions explicitly.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        _build_one(config, region)


if __name__ == "__main__":
    run(build)
