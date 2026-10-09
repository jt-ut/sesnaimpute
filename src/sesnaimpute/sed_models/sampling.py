"""
sampling.py
====================================================================
The ONE sampler every library driver codes against (curation
consolidation, 2026-10-08). Replaces six near-duplicate greedy r-nets
(`sps_curate.greedy_max_coverage_r_net`, `agb_curate.greedy_rnet`,
`galaxy_curate.greedy_max_cover_r_net`, `h2shock_curate.
greedy_coverage_r_net`, `pahc_curate.greedy_r_net`, `yso_fps.
greedy_coverage_r_net`), six members writers, and three copies of the
sampling-scale quotient-space plumbing (`sps_curate.
build_sps_quotient_space`, `build/agb.py.sigma_log_vector`, `build/
h2shock.py.quotient_space`) with one implementation, one members
schema, one scale.

THE SPACE. `sed_models_register.density.build_quotient_space` builds
the FITTER's 5-D quotient projector: SVD of the un-whitened nuisance
columns {gray, draine_rv3.1, whitney.r550}, giving the complement of
BOTH extinction laws (the fit marginalises over which law is right, so
both reddening directions are nuisance to it). That is the wrong space
for SAMPLING: the r-net's job is to decide how finely the SED shape
needs to be gridded given the SURVEY's per-band calibration floor, and
mixing eight bands of different floor (2MASS 0.010 dex, IRAC 0.013,
MIPS24 0.017, doubled -- `constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR`)
through an UN-whitened projector under one isotropic radius (what the
duplicated r-nets above did, radius = SIGEFF, a single scalar) measures
distance in units that are too fine in the best-calibrated bands and
too coarse in the worst ones.

So this module WHITENS first -- divides every band's log10 flux by its
own sampling sigma -- and only then removes the nuisance directions,
in the WHITENED space, so a radius of 1.0 is one whitened unit (one
per-band sampling sigma) in every direction at once. Only TWO
directions are nuisance here, not three: the gray column (overall
brightness/scale) and ONE extinction column -- a single reddening law
is enough to state "two SEDs differing only by scale and reddening
along this law are not what the r-net should distinguish"; the
SECOND law in `density.DEFAULT_LAWS` is the FITTER's law-choice
uncertainty, not a sampling nuisance, so it stays IN this module's
6-D space (d = 8 - 2 = 6, not density.py's d = 5). The construction
method is reused verbatim from `density.build_quotient_space` (SVD of
the nuisance columns, not Gram-Schmidt -- the two laws are nearly
collinear, cos 0.999337, so Gram-Schmidt loses orthogonality there;
SVD does not), applied to the nuisance columns AFTER dividing by
`sigma_vec` rather than before.

DARK BANDS. `whiten()` takes `log10_flux` already floored by the
caller under the register's B0/B0.1 convention (`reference.py`'s
FLOOR_LINEAR floor applied before any log, the same floor `density.
derive` applies to `f_ref`) and `dark_mask` flagging which entries sit
at that floor. This module does not re-floor or zero those entries: it
projects the floored log value exactly as `density.derive` projects
`f_ref` after its own floor, with no extra branch for darkness, so
curation's sampling distance and the register's fit-space density are
one arithmetic operating on two different quotient spaces, not two
different conventions for what a dark band IS. `dark_mask` is carried
through for the caller's own diagnostics (a model dark in most bands
is intrinsically closer to its neighbours in the dark bands than real
SED structure would produce, which is exactly `density.py`'s own
floor-dominated-cluster note); `whiten` does not act on it beyond
validating its shape.

SCALE. `r_net` is the SEQUENTIAL r-net [coordinator's scaling
correction], not the greedy maximum-coverage rule this module shipped
with first. That rule (rebuild a KD-tree over the shrinking uncovered
set every outer iteration, keep the point with the most uncovered
neighbours) measured at ~3 ms per surviving point on the gated YSO
population with a cKDTree rebuild per iteration -- O(k n log n), hours
over 956,513 models -- to buy a maximum-coverage GUARANTEE that
covering and packing never needed: visiting every point once, in a
FIXED deterministic order, and keeping it whenever no already-kept
point covers it gives the same two guarantees by construction, with no
per-step argmax over a shrinking set. See `r_net`'s own docstring for
the order and the rebuild-batching that makes it O(n log k).

MEMORY. No step here builds a full neighbour-adjacency list. `r_net`
holds the points, one cKDTree over the kept set as of its last rebuild,
and a small buffer (bounded by the rebuild batch size) of kept points
not yet folded into that tree -- never an O(n) persistent per-point
neighbour list and never O(n^2).
====================================================================
"""

import numpy as np
from astropy.table import Column, Table
from scipy.spatial import cKDTree

from sesnaimpute.sed_models.constants import LIBRARY_SAMPLING_SIGMA_LOG_VECTOR
from sesnaimpute.sed_models.register import density

__all__ = ["quotient_space", "whiten", "r_net", "assign", "coverage",
           "packing", "members_table"]

#: The single extinction law whose column is removed as a sampling
#: nuisance, alongside gray. `density.DEFAULT_LAWS[0]` -- the same law
#: `density.build_quotient_space` removes first; its SECOND law is the
#: fitter's own law-uncertainty direction and is deliberately kept IN
#: this module's space (see the module header).
_SAMPLING_LAW = density.DEFAULT_LAWS[0]


def quotient_space():
    """`(sigma_vec, projector)`.

    `sigma_vec` : `constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR` as an
    (8,) float array -- the per-band sampling sigma (twice the surveys'
    published absolute-calibration floor; see that constant).

    `projector` : the (8, 8) orthogonal projector onto the complement of
    {gray, `_SAMPLING_LAW`} IN THE WHITENED space, i.e. built from those
    two nuisance columns divided by `sigma_vec` -- `density.
    build_quotient_space`'s SVD construction, reused verbatim, applied
    to the whitened columns rather than the raw ones. Rank-checked
    rather than assumed: a projector onto fewer than 6 dimensions would
    mean the two whitened nuisance columns had become linearly
    dependent, which should not happen for two different directions.
    """
    sigma_vec = np.asarray(LIBRARY_SAMPLING_SIGMA_LOG_VECTOR, dtype=float)
    gray = np.ones(sigma_vec.shape[0])
    av_law = density._av_law(_SAMPLING_LAW)
    nuisance = np.column_stack([gray, av_law]) / sigma_vec[:, None]

    # SVD, not Gram-Schmidt: see module header and density.py's own note
    # on near-collinear extinction directions.
    U, s, _ = np.linalg.svd(nuisance, full_matrices=True)
    rank = int(np.sum(s > s[0] * 1e-12))
    if rank != 2:
        raise ValueError(
            f"sampling.quotient_space: expected the two whitened nuisance "
            f"columns (gray, {_SAMPLING_LAW}) to have rank 2, got {rank} "
            f"(singular values {s.tolist()})"
        )
    basis = U[:, rank:]          # (8, 6) orthonormal columns, the kept space
    projector = basis @ basis.T  # (8, 8), onto the 6-D complement
    return sigma_vec, projector


def whiten(log10_flux, dark_mask):
    """`log10_flux` (n, 8), `dark_mask` (n, 8) -> `coords` (n, 8).

    Divides by `sigma_vec` per band, then applies the projector from
    `quotient_space()`. The result stays (n, 8): the projector removes
    two dimensions' worth of VARIANCE, not two columns, so two models
    differing only in scale/reddening along `_SAMPLING_LAW` land at the
    same point rather than at a shorter vector -- distances computed on
    the full 8 columns equal distances computed on a 6-D basis for the
    kept subspace, with no reduction step needed before `r_net`.

    `dark_mask` is validated for shape parity and otherwise unused here
    (see the module header on why darkness needs no extra branch once
    the caller has floored `log10_flux`).
    """
    log10_flux = np.asarray(log10_flux, dtype=float)
    dark_mask = np.asarray(dark_mask, dtype=bool)
    if log10_flux.ndim != 2 or log10_flux.shape[1] != 8:
        raise ValueError(f"whiten: log10_flux must be (n, 8), got {log10_flux.shape}")
    if dark_mask.shape != log10_flux.shape:
        raise ValueError(
            f"whiten: dark_mask shape {dark_mask.shape} != "
            f"log10_flux shape {log10_flux.shape}"
        )
    sigma_vec, projector = quotient_space()
    x = log10_flux / sigma_vec[None, :]
    return x @ projector.T


#: How many newly-kept points accumulate in the brute-force buffer
#: before they are folded into one rebuilt cKDTree. The one knob in
#: `r_net`'s complexity: each point costs one O(log k) tree query plus
#: an O(_RNET_REBUILD_EVERY) brute-force scan (O(1) in n), and a tree
#: of size k is rebuilt once every `_RNET_REBUILD_EVERY` keeps rather
#: than once per keep.
_RNET_REBUILD_EVERY = 2048


def r_net(coords, radius=1.0):
    """Deterministic SEQUENTIAL r-net at `radius`.

    [Coordinator's scaling correction, replacing the greedy maximum-
    coverage rule this function shipped with first: that rule rebuilt a
    cKDTree over the shrinking UNCOVERED set every outer iteration --
    O(k) rebuilds, each over up to n points, i.e. O(k n log n) overall,
    measured at ~3 ms per surviving point on the gated YSO population
    (956,513 models) -- hours, for a maximum-coverage GUARANTEE that
    covering and packing never actually needed.]

    ORDER: index order after a stable sort on `coords[:, 0]` (the first
    whitened coordinate) -- deterministic, no random seed, and one
    O(n log n) sort rather than a per-step argmax over a shrinking set.

    RULE: visit points in that order. A point is KEPT if no
    already-kept point lies within `radius` of it; otherwise it is left
    for `assign()` to resolve against its nearest kept point after
    `r_net` returns (this function only needs the yes/no distance test,
    not which kept point is nearest).

    COVERING holds because a point that is not kept was checked, at the
    moment it was visited, against every point kept so far -- and kept
    points are never later removed, so that same kept point still
    covers it at the end. PACKING holds because a point is only kept
    after that identical check found every existing kept point farther
    than `radius` away. Neither guarantee depends on visit order beyond
    "fixed and exhaustive", which is exactly why the maximum-coverage
    rule's extra bookkeeping (always keeping the locally densest
    uncovered point first) was never load-bearing for them.

    BOUNDED MEMORY, O(n log k) TIME: the "already-kept" test is one
    `cKDTree.query(point, k=1)` against a tree built over the kept set
    AS OF ITS LAST REBUILD, plus a brute-force scan of the (at most
    `_RNET_REBUILD_EVERY`) points kept since that rebuild. Once that
    buffer reaches `_RNET_REBUILD_EVERY`, it is folded into one rebuilt
    tree over every kept point so far and the buffer empties. No
    per-point neighbour list, bounded or otherwise, and no O(n) or
    O(n^2) structure at any point.

    Returns the kept RAW INDEX array, in VISIT order (i.e. sorted by
    `coords[:, 0]`, stably) -- not coverage-count order, since nothing
    here counts coverage any more.
    """
    coords = np.ascontiguousarray(coords, dtype=float)
    n, d = coords.shape
    if n == 0:
        return np.array([], dtype=int)

    order = np.argsort(coords[:, 0], kind="stable")

    kept = []
    tree = None
    # Preallocated buffer for the points kept since the last rebuild --
    # a Python list here would force an O(len(recent)) list-to-array
    # conversion on EVERY visited point (not just every keep), which
    # dominates the real cost far more than the distance arithmetic
    # itself. A fixed (_RNET_REBUILD_EVERY, d) array and a row count is
    # O(1) to extend and O(1) to slice (a view, no copy).
    recent_buf = np.empty((_RNET_REBUILD_EVERY, d))
    n_recent = 0

    for idx in order:
        p = coords[idx]
        min_dist = np.inf
        if tree is not None:
            d_tree, _ = tree.query(p, k=1)
            min_dist = float(d_tree)
        if n_recent:
            diff = recent_buf[:n_recent] - p
            m = float(np.sqrt(np.einsum("ij,ij->i", diff, diff)).min())
            if m < min_dist:
                min_dist = m

        if min_dist <= radius:
            continue   # covered by an already-kept point

        kept.append(int(idx))
        recent_buf[n_recent] = p
        n_recent += 1
        if n_recent >= _RNET_REBUILD_EVERY:
            tree = cKDTree(coords[kept])
            n_recent = 0

    return np.array(kept, dtype=int)


def assign(coords, kept_index):
    """Nearest-kept-point assignment for every row of `coords`.

    Returns `(rep_of, dist_to_rep)`, both (n,). One batched nearest-
    neighbour query against the kept subset (`cKDTree.query(..., k=1)`)
    -- a single-point-per-row nearest query, not a stored adjacency
    list; scipy does not materialise an (n, n_kept) distance matrix.
    """
    coords = np.ascontiguousarray(coords, dtype=float)
    kept_index = np.asarray(kept_index, dtype=int)
    tree = cKDTree(coords[kept_index])
    dist, nearest = tree.query(coords, k=1, workers=-1)
    dist = np.atleast_1d(dist)
    nearest = np.atleast_1d(nearest)
    return kept_index[nearest], np.asarray(dist, dtype=float)


def coverage(coords, kept_index, radius=1.0):
    """`(max_dist, n_uncovered)`: the covering check on `kept_index`.

    `max_dist` is the largest nearest-kept-point distance over every
    raw point (derived from `assign`, not a second query); `n_uncovered`
    counts points further than `radius` from their nearest kept point.
    """
    _, dist = assign(coords, kept_index)
    max_dist = float(dist.max()) if dist.size else 0.0
    n_uncovered = int(np.sum(dist > radius))
    return max_dist, n_uncovered


def packing(coords, kept_index):
    """Minimum pairwise distance among the kept points (the packing
    check: must be >= the sampling radius). `inf` for fewer than two."""
    coords = np.ascontiguousarray(coords, dtype=float)
    kept_index = np.asarray(kept_index, dtype=int)
    if kept_index.size < 2:
        return float("inf")
    rep_coords = coords[kept_index]
    tree = cKDTree(rep_coords)
    dist, _ = tree.query(rep_coords, k=2, workers=-1)
    return float(np.asarray(dist)[:, 1].min())


def members_table(names, rep_of, dist_to_rep, subclass, params):
    """One row per KEPT template -- the represented-set table every
    library's `write_library` call passes to the shared register writer.

    `names` (n,), `rep_of` (n,) raw index of each row's nearest kept
    point, `dist_to_rep` (n,), `subclass` (n,) and `params` an astropy
    `Table` (n, p) of named physical-parameter columns. The kept set is
    `np.unique(rep_of)` (ascending raw-index order, since `rep_of` only
    ever holds values from `r_net`'s kept index).

    Columns: `MODEL_NAME`, `N_MEMBERS`, `MEMBER_INDEX` (variable-length,
    the raw row indices of the represented set), `DIST_MAX`,
    `FRAC_<subclass>` per distinct subclass value (normalised to sum to
    exactly 1.0 in float64 across a row), and `<param>_MIN`/`_MEDIAN`/
    `_MAX` per column of `params`.
    """
    names = np.asarray(names)
    rep_of = np.asarray(rep_of, dtype=int)
    dist_to_rep = np.asarray(dist_to_rep, dtype=float)
    subclass = np.asarray(subclass)
    kept = np.unique(rep_of)
    n_kept = kept.size

    subclass_codes = sorted(set(subclass.tolist()))

    model_name, n_members, member_index, dist_max = [], [], [], []
    frac_cols = {code: np.empty(n_kept, dtype=np.float64) for code in subclass_codes}
    param_names = list(params.colnames)
    param_cols = {
        p: {stat: np.empty(n_kept, dtype=np.float64) for stat in ("MIN", "MEDIAN", "MAX")}
        for p in param_names
    }

    for row, rep in enumerate(kept):
        member_idx = np.flatnonzero(rep_of == rep)
        model_name.append(names[rep])
        n_members.append(member_idx.size)
        member_index.append(member_idx.astype(np.int32))
        dist_max.append(float(dist_to_rep[member_idx].max()))

        sub_here = subclass[member_idx]
        total = member_idx.size
        raw_frac = {code: float(np.sum(sub_here == code)) / total for code in subclass_codes}
        norm = sum(raw_frac.values())
        for code in subclass_codes:
            frac_cols[code][row] = (raw_frac[code] / norm) if norm > 0 else 0.0

        for p in param_names:
            vals = np.asarray(params[p])[member_idx].astype(float)
            param_cols[p]["MIN"][row] = np.min(vals)
            param_cols[p]["MEDIAN"][row] = np.median(vals)
            param_cols[p]["MAX"][row] = np.max(vals)

    table = Table()
    table["MODEL_NAME"] = np.array(model_name)
    table["N_MEMBERS"] = np.array(n_members, dtype=np.int32)
    table["MEMBER_INDEX"] = Column(member_index, dtype=object)
    table["DIST_MAX"] = np.array(dist_max, dtype=np.float64)
    for code in subclass_codes:
        table[f"FRAC_{code}"] = frac_cols[code]
    for p in param_names:
        for stat in ("MIN", "MEDIAN", "MAX"):
            table[f"{p}_{stat}"] = param_cols[p][stat]
    return table
