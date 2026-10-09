"""
yso_fps.py
====================================================================
SED-space sampling at the fitter's resolution (owner library spec,
2026-10-08). Replaces the former per-stratum Farthest-Point-Sampling
(FPS) coverage machinery ENTIRELY -- there is no per-stratum budget any
more, and no strata enter the sampling at all.

ONE GLOBAL GREEDY-COVERAGE R-NET, over the whole gated (eligibility.
eligible == 1) pool, in the project-wide 5-D quotient space
(`sed_models_register.density.build_quotient_space` -- gray/scale and
both Av laws projected out), at radius SIGEFF (one photometric noise
length, `sed_models_register.noise`'s measured SIGMA_LOG). Module
docstring, not a path: this module never imports `sed_models_register`
itself (the project rule is that only the driver reads that package's
products -- see `build.yso.quotient_space()`); every function here
takes an already-built `space` object exposing `.project(x)` and
`.sigma_eff`, matching `h2shock_curate.py`'s identical pattern for the
same spec item.

THE ALGORITHM is a deterministic GREEDY MAXIMUM-COVERAGE r-net
(`greedy_coverage_r_net`): at each step, among the eligible models not
yet covered, keep the one with the most uncovered neighbours within
SIGEFF, mark everything within SIGEFF of it covered, and repeat until
nothing is uncovered. This gives both properties the specification asks
for BY CONSTRUCTION, with no repair pass and no RNG seed:

  covering -- every eligible model is marked covered only by a kept
              template within SIGEFF, so the loop cannot terminate
              until every model has one;
  packing  -- a model already covered is never eligible to be chosen
              (the loop only ever pops UNCOVERED models), so any two
              kept templates are necessarily more than SIGEFF apart.

Each kept template is therefore the centre of the densest UNCLAIMED
region at the moment it is picked -- not an arbitrary traversal hit and
not a medoid bolted on after the fact. The represented sets
(`assign_to_nearest_representative`) are derived FROM the kept
templates by nearest-neighbour (Voronoi) assignment, strictly after the
representatives are fixed -- never the other way round.

FULLY DETERMINISTIC, NO RNG SEED: ties in "most uncovered neighbours"
break on ascending row index into the eligible-model coordinate array,
which is itself built in ascending `model_id` order (a total order that
exists before this stage runs -- see `eligible_sed_coordinates`).
Re-running on the same inputs reproduces the same kept set exactly.

WHAT THE SAMPLING SPACE IS: the 8 log10 band fluxes already stored in
`/features/mu_sed` (labeling's own clamped, hybrid survey-matched SED,
linear mJy), log10'd and projected through `space`. NOT the 20
harmonized physical parameters and NOT the 7 SED colors the retired
per-stratum machinery built (CANDIDATE_FEATURE_COLUMNS, transform_
parameter_block, compute_block_balance_alpha, etc. -- all removed,
along with N_SUB/FLOOR_PCT/compute_entropy_k/allocate_n_k/
_fps_kmeans_plus_plus): the owner's library spec samples in SED space
alone, at one noise length, with no strata and no per-stratum budget.

STRATUM_LABELS / STRATUM_OUTPUT_FOLDER are KEPT, unchanged -- they still
name the five on-disk directories and the stage vocabulary assembly
maps a kept template's REPRESENTED-SET MAJORITY onto (see
`yso_assembly.py`); they no longer drive any sampling decision.
====================================================================
"""

import heapq

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sesnaimpute.sed_models.curate import yso_curation_state as reg
from sesnaimpute.sed_models.curate.yso_labeling import BAND_ORDER

STRATUM_LABELS = ("0", "I", "II", "III", "TD")
STRATUM_OUTPUT_FOLDER = {"0": "c0", "I": "cI", "II": "cII", "III": "cIII", "TD": "td"}


# ==========================================================================
# ELIGIBLE-POOL SED COORDINATES
# ==========================================================================

def eligible_sed_coordinates(curation_state_path, space, verbose=True):
    """Every `eligibility.eligible == 1` model's 8 log10 band fluxes
    (from `/features/mu_sed`, already the clamped survey-matched SED
    labeling built), projected into `space`'s 5-D quotient coordinates.

    Returns (model_ids, coords), `model_ids` sorted ascending (the fixed
    total order `greedy_coverage_r_net`'s tie-break relies on; see this
    module's docstring).
    """
    eligibility = reg.read_dataset(curation_state_path, "eligibility", columns=["eligible"])
    mu_sed = reg.read_dataset(curation_state_path, "mu_sed")

    elig_ids = eligibility.loc[reg.is_true(eligibility["eligible"]), "model_id"].to_numpy()
    elig_ids = np.sort(elig_ids)

    mu = mu_sed.set_index("model_id").loc[elig_ids]
    flux_mjy = np.column_stack([mu[f"mu_{band}"].to_numpy(dtype=float) for band in BAND_ORDER])

    bad = ~np.isfinite(flux_mjy) | (flux_mjy <= 0)
    if np.any(bad):
        raise ValueError(
            f"{int(bad.any(axis=1).sum())} of {len(elig_ids)} eligible model(s) have a "
            f"non-positive or non-finite mu_sed band flux -- cannot take log10")

    coords = space.project(np.log10(flux_mjy))
    if verbose:
        print(f"[fps] {len(elig_ids)} eligible models -> {coords.shape[1]}-D "
              f"quotient-space coordinates", flush=True)
    return elig_ids, coords


# ==========================================================================
# GREEDY MAXIMUM-COVERAGE R-NET (owner library spec item 2)
# ==========================================================================

def greedy_coverage_r_net(coords, radius):
    """Deterministic greedy MAXIMUM-COVERAGE r-net over `coords` (n, d).

    See the module header for why this gives both covering and packing
    at `radius` by construction, with no seed and no repair pass.
    Returns the chosen representatives' row positions into `coords`,
    ascending.

    Implementation: one `cKDTree.query_ball_point` call builds every
    point's within-`radius` neighbour list once; a lazily-updated max-
    heap (Python's `heapq`, keyed on negated uncovered-neighbour count)
    then picks the next representative in amortised O(log n) per
    update, so the whole run is close to O(n log n) rather than the
    O(n^2) a naive recompute-every-step version would cost. Ties in the
    heap key break on ascending index automatically, via tuple
    comparison on `(-count, index)` -- no `key=` needed and no RNG.
    """
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, radius, workers=-1)
    counts = np.array([len(nb) for nb in neighbor_lists], dtype=np.int64)
    uncovered = np.ones(n, dtype=bool)
    heap = [(-int(counts[i]), i) for i in range(n)]
    heapq.heapify(heap)

    reps = []
    n_uncovered = n
    while n_uncovered > 0:
        neg_count, i = heapq.heappop(heap)
        if not uncovered[i]:
            continue                      # already covered: stale entry
        if -neg_count != counts[i]:
            heapq.heappush(heap, (-int(counts[i]), i))
            continue                      # count changed since pushed: refresh
        reps.append(i)
        newly_covered = [j for j in neighbor_lists[i] if uncovered[j]]
        for j in newly_covered:
            uncovered[j] = False
        n_uncovered -= len(newly_covered)
        touched = set()
        for j in newly_covered:
            for k in neighbor_lists[j]:
                if uncovered[k]:
                    counts[k] -= 1
                    touched.add(k)
        for k in touched:
            heapq.heappush(heap, (-int(counts[k]), k))

    return np.array(sorted(reps), dtype=np.int64)


def packing_radius(coords, rep_positions):
    """Minimum pairwise distance among the chosen representatives --
    the packing identity's measured value, which `greedy_coverage_r_net`
    guarantees exceeds its `radius` argument."""
    if rep_positions.size < 2:
        return np.inf
    rep_coords = coords[rep_positions]
    dist, _ = cKDTree(rep_coords).query(rep_coords, k=2, workers=-1)
    return float(dist[:, 1].min())


def assign_to_nearest_representative(coords, rep_positions):
    """Voronoi assignment of every row of `coords` to its nearest entry
    of `coords[rep_positions]`. Run strictly AFTER `greedy_coverage_
    r_net` -- the represented sets are derived from the representatives,
    never used to choose them.

    Returns `(owner, dist)`: `owner` indexes into `rep_positions`
    (0..len(rep_positions)-1); `dist` is the Euclidean distance in
    `coords`' units (dex of quotient-space log-flux; divide by
    `sigma_eff` for multiples of the noise length)."""
    dist, owner = cKDTree(coords[rep_positions]).query(coords, k=1, workers=-1)
    return owner, dist


# ==========================================================================
# DRIVER-FACING ENTRY POINTS (called by yso_curate.run_fps)
# ==========================================================================

def run_global_sampling(curation_state_path, space, verbose=True):
    """Full Stage-6 run: coordinates -> greedy r-net -> Voronoi
    assignment -> the two curation-state products (`fps` dataset rows,
    `/metadata/fps` report). No seed, no strata, no budget.

    Returns (model_ids, coords, rep_positions, owner, dist, report).
    """
    model_ids, coords = eligible_sed_coordinates(curation_state_path, space, verbose=verbose)
    radius = space.sigma_eff

    if verbose:
        print(f"[fps] greedy max-uncovered-neighbour r-net at SIGEFF={radius:.6f} dex "
              f"over {len(model_ids)} eligible models", flush=True)
    rep_positions = greedy_coverage_r_net(coords, radius)
    owner, dist = assign_to_nearest_representative(coords, rep_positions)

    coverage_fraction = float(np.mean(dist <= radius * (1.0 + 1e-9)))
    packing = packing_radius(coords, rep_positions)
    dist_sigeff = dist / radius

    report = dict(
        n_eligible=int(len(model_ids)), n_kept=int(rep_positions.size),
        sigma_eff=float(radius), quotient_dim=int(space.d),
        laws=tuple(space.laws), sigma_log=[float(s) for s in space.sigma_log],
        sigma_is_placeholder=bool(space.sigma_is_placeholder),
        coverage_fraction=coverage_fraction,
        packing_radius_dex=float(packing),
        packing_radius_sigeff=float(packing / radius) if np.isfinite(packing) else None,
        dist_sigeff_median=float(np.median(dist_sigeff)),
        dist_sigeff_max=float(np.max(dist_sigeff)),
        algorithm="greedy maximum-uncovered-neighbour r-net, deterministic, no seed; "
                  "tie-break ascending model_id (see eligible_sed_coordinates)",
    )
    if verbose:
        print(f"[fps] kept {rep_positions.size} templates; coverage identity "
              f"{coverage_fraction:.6f} (must be 1.0); packing radius "
              f"{packing:.6f} dex = {report['packing_radius_sigeff']} x SIGEFF "
              f"(must be >= 1.0)", flush=True)
        print(f"[fps] member-to-template distance, SIGEFF units: median "
              f"{report['dist_sigeff_median']:.4f}, max {report['dist_sigeff_max']:.4f}",
              flush=True)
    return model_ids, coords, rep_positions, owner, dist, report


def build_fps_outputs(curation_state_path, model_ids, rep_positions, owner, dist, space):
    """Build this phase's `/registry/fps` frame -- `selected` (1 kept
    template / 0 eligible member / FLAG_NA(-1) ineligible),
    `owner_model_id` (the kept template this row's membership belongs
    to) and `dist_sigeff` (quotient-space distance to it, in SIGEFF
    units) -- aligned to the full ingest row set via `reg.align_to_ingest`.
    """
    ingest = reg.read_dataset(curation_state_path, "ingest", columns=["geometry_folder"])

    rep_ids = model_ids[rep_positions]
    owner_ids = rep_ids[owner]
    selected = np.zeros(len(model_ids), dtype="int8")
    selected[rep_positions] = 1

    df = pd.DataFrame({
        "model_id": model_ids, "selected": selected,
        "owner_model_id": owner_ids, "dist_sigeff": dist / space.sigma_eff,
    })
    df = reg.align_to_ingest(df, ingest["model_id"].to_numpy(), "fps")
    return df
