"""One {region, class}: the class evidence, subclass evidence and top-K
posterior record of SPEC_BMSTP_DRAFT.md section 1.3
(IMPLEMENTATION_BMSTP_DRAFT.md section 1.3 P7, section 4 row 2.4).

The unit of work is ONE SOURCE (PARALLEL brief): for source `s`, section 2's
two extinction-law designs (diffuse `k=d`, `w=0`; dense `k=D`, `w=1`) and
every template `theta` of the class's library,

    ln w_theta,k = ln <Lambda_C>_s,k(theta)   -- prior_reader.ln_prior, section 4.2,
                                                  the DESIGN's own cell weight
                 + ln L_hat_s,k(theta)         -- likelihood.fit(batch, ..., w)'s
                                                  chi2 and non-detection term, this design
                 + ln Gamma_s,k(theta)         -- gaia.GaiaTerm.ln_gamma, section 6.4,
                                                  this design's own marks

    ln w_theta   = logsumexp_k(ln w_theta,k)      -- the mixture; the sightline's own
                                                      per-cell dense fraction (section 2)
                                                      already makes the two designs'
                                                      prior terms sum to the single-law
                                                      one, so no further weight is
                                                      applied here
    p_k(theta)   = exp(ln w_theta,k - ln w_theta) -- each design's own posterior share
                                                      of that template's weight

(the library-sampling weight is already folded into the prior's own
template-weight factors, section 6.3, so no separate library weight is
added here). The class evidence is `logsumexp` over every template's mixture
`ln w_theta`; the subclass evidence restricts that sum to one subclass's
templates (`-inf` where a subclass has none). `TOPK_LAW` records, per
top-K template, which design carried the larger `ln w_theta,k`; the record's
other TOPK_* fields, and `TOPK_LN_GAMMA`, are that design's own numbers.
`P_DENSE` is the source's own posterior weight on the dense design,
`sum_theta p_theta * p_D(theta)`.

Region and class are read once, in the PARENT process (`build_region_class`):
the register (`_register`), the region's non-detection width (`_width_dex`),
the class's own library-resolution number (`likelihood.sigma_lib_by_class`),
the Gaia term (`GaiaTerm`), the class's own `Prior` reader
(`prior_reader.load`, this class's only -- the prior read applies no
floor, so no other class's grid is read)
and the region's whole catalogue (`_load_region_catalog`: fluxes, uncertainties,
`ORIGIN_FNU`, `NAME`, `F_LIM_50`). These are stashed in `_WORKER` and a
`multiprocessing.get_context("fork").Pool` is created AFTER that load, and
after one warm source has compiled every numba kernel in the parent so the
workers inherit the compiled code rather than each paying for it -- the
workers inherit `_WORKER` and that code by fork by
copy-on-write, never through `Pool`'s `initializer`/`initargs` (which
pickles) and never by opening a register, grid or product themselves.
`_init_worker` (the Pool's own, data-free initializer) pins numba and BLAS
to one thread each per worker, since the source axis is now the pool's own
parallel axis and the per-template numba kernels threading too would
oversubscribe.

`pool.imap(_source_task, ..., chunksize=_IMAP_CHUNKSIZE)` dispatches one
task per source and returns results in catalogue order; the parent groups
them into batches of `[fit] batch_size` sources purely as the part-file
write granularity (`<product>.partN`, rule 10b, `sed_fit/batch.py`'s own
pattern) -- no block, and no per-block memory budget, exists anywhere in
this module or in `fittp.likelihood` any more. A source whose task raises
is caught inside `_source_task` and returned as a NaN row with `FAILED`
set (rule: one bad source must not lose a multi-hour job); `join_parts`
sums `FAILED` across every part file it finds on disk into `FAILED_ROWS`,
and `build`'s done line reports `n_failed`.

Three of section 1.3's own numbers are pinned to one reading each, since a
second reading of the same physical quantity is a second, silently
different answer: the non-detection roll-off width is the per-source
limits product's own region-band `W_DEX` (the same fit the retention floor
and the atlas's acceptance fraction already read, not `catalog.depths`'
differently-fit `WIDTH_DEX`); Gamma and the prior read a design's own
unclamped optimum `(a_hat, log10_b_hat)`, never a mix of clamped and
unclamped marks for the two factors of the same evidence; and section 9's
Occam gap subtracts `max_theta ln(mixture_k(Lambda_k L_hat_k))` (Gamma
excluded from both the mixture and the maximum). A
flagged source's `FLUX_MEAN`/`LOG10_FLUX_COV` are written `NaN`, not the
zero a `p_theta` of zero everywhere would otherwise silently accumulate,
since a flagged fit has no posterior to report a flux moment of.

`A_K_POST`/`A_K_POST_SIG` (section 6.1's reported extinction mark, not
`TOPK_A_K`'s maximum-likelihood one) fold `prior_reader.ln_prior`'s own
per-template, per-design posterior first/second moment of `a` over
templates and designs with the joint weight `p_theta * p_k(theta)`; a
flagged source writes `NaN` for both, the same convention as the flux
moments.

`LOG10_FLUX_COV` (section 1, replacing the old `FLUX_COV`) is the total
variance of the imputed flux in log10 flux, `between + within`: `between`
is the `p_theta * p_k`-weighted covariance of the `(template, design)`
pairs' own log10 flux at their clamped marks (the spread across templates
AND designs); `within` is `sum_k P_k * D_k Sigma_k D_k^T`, `P_k` each
design's own total posterior weight (`sum_theta p_theta * p_k(theta)`),
`D_k = [ext_col_k, GRAY_COLUMN]` that design's own `(8, 2)` design and
`Sigma_k = xtwx_inv_k` its own `(2, 2)` parameter covariance (`likelihood
.fit`'s own return) -- one `(8, 8)` matrix per design, not per template,
since `D_k Sigma_k D_k^T` does not depend on the template.
"""

import glob
import multiprocessing as mp
import os
import re

import h5py
import numba

# numba's own workqueue layer, "forksafe everywhere" (numba/np/ufunc/parallel.py):
# this module warms every kernel in the parent and then forks a pool, so the layer
# has to survive the fork. Set before numba initialises one; each worker pins
# itself to a single thread anyway (`_init_worker`), so the choice costs nothing.
numba.config.THREADING_LAYER = "forksafe"
import numpy as np
import threadpoolctl
from scipy.special import logsumexp

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.catalog import limits as catalog_limits
from sesnaimpute.fittp import likelihood
from sesnaimpute.fittp import prior_reader
from sesnaimpute.fittp.gaia import GaiaTerm

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: the six classes, in the fitter's own class-axis order
#: (IMPLEMENTATION_BMSTP_DRAFT.md section 1).
CLASSES = tuple(c.code for c in definitions.CLASSES)

#: one `imap` task per source, but many small tasks paid for one at a time
#: costs more in IPC than it saves; a modest chunksize amortises that
#: without holding back result consumption (results are still delivered,
#: and written, in catalogue order) for long.
_IMAP_CHUNKSIZE = 16

#: Per-worker memory floor, MB: what a forked worker costs beyond its own
#: source's arrays -- the interpreter, the imported stack, and the
#: copy-on-write pages the parent's inherited objects dirty as CPython
#: refcounts them. Measured with `vmmap --summary` (physical footprint, which
#: excludes the clean pages workers share, unlike RSS) on the YSO library, the
#: largest: ~310 MB steady, ~430 MB peak per worker. Rounded to the peak, since
#: the node has to hold it. `build_region_class`'s warm-up is what keeps the
#: JIT out of this figure: without it every worker compiles its own kernels and
#: the floor is ~180 MB higher.
WORKER_PROCESS_FLOOR_MB = 430

#: The datasets every P7 part file and the joined product carry, in write
#: order (one row per source; `FAILED`, below, is a part-file-only column
#: consumed at join time into `FAILED_ROWS`, never copied into the joined
#: product itself). `N_LAW_ITER` is gone: the self-consistent
#: extinction-law solve it counted no longer exists; `TOPK_LAW`,
#: `TOPK_LN_GAMMA` and `P_DENSE` are new (section 2, section 6.4).
_PART_KEYS = ("NAME", "LN_EVIDENCE", "FLUX_MEAN", "LOG10_FLUX_COV", "TOPK_MODEL", "TOPK_A_K",
              "TOPK_LOG10_B", "TOPK_CHI2", "TOPK_LN_L", "TOPK_LN_PRIOR", "TOPK_FLUX", "TOPK_LAW",
              "TOPK_LN_GAMMA", "OCCAM_GAP", "A_K_POST", "A_K_POST_SIG", "P_DENSE",
              "FRAC_CLAMPED", "N_DETECTED")

_FIELD_OF_KEY = {
    "NAME": "name", "LN_EVIDENCE": "ln_evidence", "FLUX_MEAN": "flux_mean",
    "LOG10_FLUX_COV": "log10_flux_cov", "TOPK_MODEL": "topk_model", "TOPK_A_K": "topk_a_k",
    "TOPK_LOG10_B": "topk_log10_b", "TOPK_CHI2": "topk_chi2", "TOPK_LN_L": "topk_ln_l",
    "TOPK_LN_PRIOR": "topk_ln_prior", "TOPK_FLUX": "topk_flux", "TOPK_LAW": "topk_law",
    "TOPK_LN_GAMMA": "topk_ln_gamma", "OCCAM_GAP": "occam_gap",
    "A_K_POST": "a_k_post", "A_K_POST_SIG": "a_k_post_sig", "P_DENSE": "p_dense",
    "FRAC_CLAMPED": "frac_clamped", "N_DETECTED": "n_detected",
}


def _register(config, cls):
    """`(template_log, subclass_idx, n_sub)` for `cls`'s own library
    register (`definitions.CLASS_REGISTER`): `template_log` is `(n_model,
    8)` float32 `log10 F_REF` (floored at the register's own
    `FLOOR_LINEAR`, its FREFRAW convention -- a genuinely dark band is not
    `-inf`), in `definitions.BANDS` order; `subclass_idx` locates each
    model in `definitions.SUBCLASSES_OF[cls]`'s own order (P7's
    `LN_EVIDENCE` column order).
    """
    key = definitions.CLASS_REGISTER[cls]
    path = os.path.join(config.inputs["sed_models"], "registers", "%s_register.hdf5" % key)
    with h5py.File(path, "r") as f:
        n_model = f["models"]["MODEL_NAME"].shape[0]
        f_ref = np.empty((n_model, N_BANDS), dtype=np.float64)
        for j, bkey in enumerate(BAND_KEYS):
            f_ref[:, j] = np.asarray(f["models"]["F_REF_%s" % bkey][:], dtype=np.float64)
        floor_linear = np.asarray(f["models"]["FLOOR_LINEAR"][:], dtype=np.float64)
        subclass_raw = f["models"]["SUBCLASS"][:]
    subclass_names = [s.decode() if isinstance(s, bytes) else s for s in subclass_raw]
    subclass_order = definitions.SUBCLASSES_OF[cls]
    sub_to_idx = {s: i for i, s in enumerate(subclass_order)}
    unknown = set(subclass_names) - set(sub_to_idx)
    if unknown:
        raise ValueError("fittp.sweep: register %r carries SUBCLASS %r not in "
                          "definitions.SUBCLASSES_OF[%r] = %r"
                          % (path, sorted(unknown), cls, subclass_order))
    subclass_idx = np.array([sub_to_idx[s] for s in subclass_names], dtype=np.intp)
    template_log = np.log10(np.maximum(f_ref, floor_linear[:, None])).astype(np.float32)
    return template_log, subclass_idx, len(subclass_order)


def _width_dex(config, region):
    """The region's own non-detection roll-off width, all eight bands: the
    per-source limits product's own `W_DEX` (`catalog.depth_grid`'s region
    counts fit for the five Spitzer bands, `catalog.depths`' fitted width
    for the three 2MASS bands, `catalog/depth_grid.py`'s module docstring)
    -- the same width the retention floor and the atlas's acceptance
    fraction already read (`population.field_stars._region_width_dex`,
    `bmstp.atlas._depth_grid`), so the likelihood's non-detection term
    prices the same roll-off as the rest of the completeness chain instead
    of a second, differently-fit width off `catalog.depths` (audit A row 1)."""
    path = config_module.product_path(config, "catalog", "sesna", "limits", "source", region=region)
    with h5py.File(path, "r") as f:
        return np.asarray(f["W_DEX"][:], dtype=np.float64)


def _load_region_catalog(config, region):
    """This region's whole catalogue, read ONCE (item 4): fluxes,
    uncertainties and `ORIGIN_FNU` (the same read `fittp.cascade` uses),
    `NAME`, and `catalog.limits.limits`'s own per-source `F_LIM_50` -- so
    every worker's own per-source task slices these already-loaded, fork-
    inherited arrays instead of opening either product itself."""
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        flux = np.asarray(f["FNU_MJY"][:], dtype=np.float64)
        sigma = np.asarray(f["SIGMA_FNU_MJY"][:], dtype=np.float64)
        origin = f["ORIGIN_FNU"][:]
        name = f["NAME"][:]
    f_lim50 = catalog_limits.limits(config, region)
    return dict(flux=flux, sigma=sigma, origin=origin, name=name, f_lim50=f_lim50)


def _n_sources(config, region):
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        return f["NAME"].shape[0]


@numba.njit(parallel=True, cache=True)
def _flux_moments_topk_kernel(log10_flux, p_theta, sorted_order, rank_position,
                               good, out_mean, out_m2, out_topk_flux):
    """The evidence-weighted flux moments (section 1.3's posterior mean and
    second moment in flux, `E[F]` and `E[F F^T]`) and the top-K flux record,
    one pass per source (`prange`, item 5's own worker-side thread cap:
    called here with a leading axis of 1, one source at a time -- the
    kernel's own generality over many sources is unused, not a block): for
    each template `t` the linear flux `f = 10**log10_flux[s, t, :]` (8
    float64 values, held in a small per-source buffer, never written to a
    `(n, n_model, 8)` array) is folded into `mean += p_theta*f` and `m2 +=
    p_theta*f f^T`, and, for the templates already chosen by the
    argpartition/argsort on `ln_w` before this kernel runs (`_source_task`),
    copied into `out_topk_flux` at that template's rank. `sorted_order` is
    the source's own top-K template indices sorted ascending, and
    `rank_position` maps each ascending slot back to its position in the
    ln_w-descending top-K list, so the O(1) pointer walk below (advancing
    only when `t` reaches the next sorted index) lands each flux in the
    same slot `topk_model`/`topk_a_k`/etc. use for that template. `good[s]`
    false (a flagged source) skips the top-K write only -- `out_topk_flux`
    was pre-filled with NaN by the caller -- while the moments still
    accumulate p_theta=0 everywhere, matching the unmasked
    `flux_mean`/`flux_cov` the caller overwrites with NaN afterward.
    """
    n_block = log10_flux.shape[0]
    n_model = log10_flux.shape[1]
    n_bands = log10_flux.shape[2]
    k_keep = sorted_order.shape[1]
    for s in numba.prange(n_block):
        mean = np.zeros(n_bands)
        m2 = np.zeros((n_bands, n_bands))
        f = np.empty(n_bands)
        ptr = 0
        row_good = good[s]
        for t in range(n_model):
            for b in range(n_bands):
                f[b] = 10.0 ** log10_flux[s, t, b]
            p = p_theta[s, t]
            for a in range(n_bands):
                pa = p * f[a]
                mean[a] += pa
                for b in range(n_bands):
                    m2[a, b] += pa * f[b]
            if row_good and ptr < k_keep and t == sorted_order[s, ptr]:
                r = rank_position[s, ptr]
                for b in range(n_bands):
                    out_topk_flux[s, r, b] = f[b]
                ptr += 1
        for a in range(n_bands):
            out_mean[s, a] = mean[a]
            for b in range(n_bands):
                out_m2[s, a, b] = m2[a, b]


# ---------------------------------------------------------------------------
# the worker side: one task per source (PARALLEL brief items 3, 4, 5, 7)
# ---------------------------------------------------------------------------

#: Populated by `_set_worker_state` in the PARENT, before the pool forks;
#: every worker then reads it by copy-on-write inheritance -- never sent
#: through `Pool`'s own `initializer`/`initargs`, which pickles. Read-only
#: after the pool is created.
_WORKER = {}


def _set_worker_state(**kw):
    global _WORKER
    _WORKER = kw


def _init_worker():
    """The Pool's own initializer, run once per forked worker before its
    first task -- deliberately carries no data (item 4: an initializer
    that DID take the register/reader/catalogue would pickle them once per
    worker, exactly what fork is here to avoid). Caps numba and BLAS to one
    thread each (item 5): the source axis is now the parallel axis (the
    pool itself), so the per-template numba kernels
    (`fittp.likelihood._ln_one_minus_c_kernel`, `fittp.prior_reader.
    _cell_sum`, `fittp.gaia._gaia_h_kernel`, `_flux_moments_topk_kernel`
    above) and BLAS's own small gemms threading too would oversubscribe
    the machine. Must run before any of those kernels executes in this
    worker for the first time, which it does: nothing calls one before a
    task does."""
    numba.set_num_threads(1)
    threadpoolctl.threadpool_limits(1, user_api="blas")


def _empty_row(topk, n_sub):
    return dict(
        ln_evidence=np.full(n_sub, -np.inf, dtype=np.float32),
        flux_mean=np.full(N_BANDS, np.nan, dtype=np.float32),
        log10_flux_cov=np.full((N_BANDS, N_BANDS), np.nan, dtype=np.float32),
        topk_model=np.full(topk, -1, dtype=np.int32),
        topk_a_k=np.full(topk, np.nan, dtype=np.float32),
        topk_log10_b=np.full(topk, np.nan, dtype=np.float32),
        topk_chi2=np.full(topk, np.nan, dtype=np.float32),
        topk_ln_l=np.full(topk, np.nan, dtype=np.float32),
        topk_ln_prior=np.full(topk, np.nan, dtype=np.float32),
        topk_flux=np.full((topk, N_BANDS), np.nan, dtype=np.float32),
        topk_law=np.full(topk, -1, dtype=np.int8),
        topk_ln_gamma=np.full(topk, np.nan, dtype=np.float32),
        occam_gap=np.float32(np.nan),
        a_k_post=np.float32(np.nan), a_k_post_sig=np.float32(np.nan),
        p_dense=np.float32(np.nan),
        frac_clamped=np.float32(np.nan),
        n_detected=np.int8(-1),
    )


def _source_task(i):
    """One task, one source: fit `cls`'s whole library to catalogue row
    `i` (SPEC_BMSTP_DRAFT.md section 1.3; the PARALLEL brief's "one
    iteration = fit this class's models to this one source"), reading
    nothing from disk -- every product this touches came from `_WORKER`,
    loaded once in the parent and reached this process by fork. A source
    whose fit raises is caught here and returned as a NaN row with
    `FAILED` set (item 7): one bad source must not lose the whole sweep,
    and the failure is recorded, never silent.
    """
    w = _WORKER
    topk = w["topk"]
    n_sub = w["n_sub"]
    try:
        flux = w["flux"][i]
        sigma = w["sigma"][i]
        origin = w["origin"][i]
        f_lim50 = w["f_lim50"][i]

        batch = likelihood.prepare(w["config"], flux, sigma, origin,
                                    w["sigma_lib_l"], f_lim50, w["width_dex"])

        n_model = w["n_model"]
        rows = np.array([i])
        model_index = np.arange(n_model)
        h = prior_reader.prepare(w["reader"], rows)

        # section 2's two-design mixture: the sightline's own per-cell dense
        # fraction w_i (fittp.prior_reader.load, from bmstp.cloud_interval
        # through P1's SIGHTLINE_ROW), 1 - w_i owned by the diffuse design,
        # w_i by the dense one -- never a per-hypothesis extinction mark.
        w_dense_cell = w["reader"].w_dense[i][None, :]           # (1, n_x)
        cell_weight_by_law = (1.0 - w_dense_cell, w_dense_cell)

        # Where the source's own w_i is 0 in every cell, the dense design
        # cannot contribute: its cell sum is exactly zero at every template
        # (`prior_reader._cell_sum`'s cell-weight guard, both the exact/table
        # path and the two edge fallbacks), so its ln_lambda is -inf for
        # every template regardless of what its own fit or Gamma read would
        # say, and that -inf alone drives ln_w_k[1] to -inf too (a finite or
        # -inf ln_l/ln_gamma added to -inf is still -inf). The fit, prior
        # read and Gamma call for that design are skipped; the mixture reads
        # the -inf directly instead of computing it.
        dense_possible = bool(np.any(w_dense_cell > 0.0))
        active_laws = (0, 1) if dense_possible else (0,)

        fits = [None, None]
        ln_lambda = [None, None]       # float32 (m,) per design
        a_post_t = [None, None]        # float64 (m,) per design
        a2_post_t = [None, None]
        ln_l = [None, None]             # float64 (m,) per design
        ln_gamma = [None, None]         # float64 (m,) per design
        for k in active_laws:
            blend_w = float(k)
            fit_k = likelihood.fit(batch, w["template_log"], blend_w)
            fits[k] = fit_k

            # d(log10 B)/d(a_K) from this design's own d(SC)/d(A_V)
            # (fit_k.slope_sc_av): log10_B = -2*SC, a_K = ak_per_av * A_V,
            # so d(log10_B)/d(a_K) = -2 * slope_sc_av / ak_per_av
            # (likelihood.fit's docstring, section 1.3).
            slope_log10b_per_ak = np.array([-2.0 * fit_k.slope_sc_av / fit_k.ak_per_av])
            ln_lambda_k, a_post_k, a2_post_k = prior_reader.ln_prior(
                w["reader"], rows, h, fit_k.a_hat[None, :], fit_k.log10_b_hat[None, :],
                slope_log10b_per_ak, np.array([fit_k.sigma_a_ak]), model_index,
                cell_weight_by_law[k])
            ln_lambda[k] = ln_lambda_k[0]
            a_post_t[k] = a_post_k[0]
            a2_post_t[k] = a2_post_k[0]

            ln_l[k] = -0.5 * fit_k.chi2_min.astype(np.float64) + fit_k.ln_nondet.astype(np.float64)

            # Gamma reads this design's own unclamped optimum a_hat/log10_b_hat
            # (one set of marks per design for the two class-evidence factors,
            # section 6.4; the clamped marks stay for the reported marks, the
            # top-K record and the flux prediction only, section 6.1).
            ln_gamma[k] = w["gaia_term"].ln_gamma(
                rows, model_index, fit_k.a_hat[None, :], fit_k.log10_b_hat[None, :],
                w["cls"].lower())[0]

        if not dense_possible:
            ln_lambda[1] = np.full(n_model, -np.inf, dtype=np.float32)
            ln_l[1] = np.full(n_model, -np.inf, dtype=np.float64)
            ln_gamma[1] = np.zeros(n_model, dtype=np.float64)
            a_post_t[1] = np.full(n_model, np.nan, dtype=np.float64)
            a2_post_t[1] = np.full(n_model, np.nan, dtype=np.float64)

        combined_flagged = fits[0].flagged or (fits[1].flagged if fits[1] is not None else False)

        ln_lambda64 = [x.astype(np.float64) for x in ln_lambda]
        ln_w_k = np.stack([ln_lambda64[k] + ln_l[k] + ln_gamma[k] for k in (0, 1)], axis=0)  # (2, m)
        if combined_flagged:
            ln_w_k = np.full((2, n_model), -np.inf, dtype=np.float64)

        # the mixture (section 2): logsumexp over the two designs at each
        # template; the sightline's own w_i already makes the two designs'
        # PRIOR terms sum to the single-law read, so no further weight is
        # applied here (module docstring).
        ln_w = logsumexp(ln_w_k, axis=0)                        # (m,)
        with np.errstate(invalid="ignore"):
            p_k = np.exp(ln_w_k - ln_w[None, :])                # (2, m), each design's own share
        p_k = np.where(np.isfinite(p_k), p_k, 0.0)

        subclass_idx = w["subclass_idx"]
        ln_evidence64 = np.full(n_sub, -np.inf, dtype=np.float64)
        for kk in range(n_sub):
            mask = subclass_idx == kk
            if mask.any():
                ln_evidence64[kk] = logsumexp(ln_w[mask])
        ev_total = logsumexp(ln_evidence64)

        with np.errstate(invalid="ignore"):
            p_theta = np.exp(ln_w - ev_total)
        p_theta = np.where(np.isfinite(p_theta), p_theta, 0.0)

        k_keep = min(topk, n_model)
        order = np.argpartition(-ln_w, k_keep - 1)[:k_keep]
        order = order[np.argsort(-ln_w[order])]

        # section 9's Occam gap is `ln EV_C - max_theta ln(mixture)`: the
        # mixture here is the two designs' own (Lambda, L_hat) alone, Gamma
        # excluded from both the mixture and the subtracted maximum.
        ln_ll_k = np.stack([ln_lambda64[k] + ln_l[k] for k in (0, 1)], axis=0)   # (2, m)
        mixture_ll = logsumexp(ln_ll_k, axis=0)
        occam_gap = float(ev_total - mixture_ll.max())

        good = not combined_flagged

        # TOPK_LAW: which design carried the larger ln_w_k for each of the
        # k_keep selected templates (section 2); TOPK_LN_GAMMA (section 6.4)
        # rides the same branch.
        topk_law_sel = (ln_w_k[1, order] > ln_w_k[0, order]).astype(np.int8)     # (k_keep,)

        # the posterior extinction mark (section 6.1): the p_theta * p_k
        # -weighted mean over templates AND designs of prior_reader.ln_prior's
        # own per-template, per-design posterior first/second moment of a,
        # nansum applied per design then summed: a NaN moment carries
        # p_theta * p_k = 0 by construction wherever ln_w_k is -inf there.
        with np.errstate(invalid="ignore"):
            a_post = 0.0
            a2_post = 0.0
            for k in (0, 1):
                wk = p_theta * p_k[k]
                a_post += float(np.nansum(wk * a_post_t[k]))
                a2_post += float(np.nansum(wk * a2_post_t[k]))
            a_post_sig = float(np.sqrt(max(a2_post - a_post ** 2, 0.0)))
        if not good or not np.isfinite(ev_total):
            a_post, a_post_sig = float("nan"), float("nan")

        # P_DENSE: the source's own posterior weight on the dense
        # design, 0 wherever the sightline's own w_i is 0 in every cell
        # (the dense design's prior term is -inf everywhere then, so
        # p_k[1] is 0 for every template).
        p_dense = float(np.sum(p_theta * p_k[1])) if good else float("nan")

        # each design's own fitted flux at its own clamped marks: recovering
        # the A_V-unit extinction that design's own column (fit_k.ext_col)
        # was built in from its reported a_K mark (fit_k.a_hat_clamped =
        # A_V_clamped * ak_per_av, likelihood.fit's own docstring).
        log10_flux_by_law = [None, None]
        for k in active_laws:
            fit_k = fits[k]
            av_clamped_k = fit_k.a_hat_clamped.astype(np.float64) / fit_k.ak_per_av
            log10_flux_by_law[k] = (
                w["template_log"].astype(np.float64)
                + fit_k.ext_col.astype(np.float64)[None, :] * av_clamped_k[:, None]
                + fit_k.log10_b_hat_clamped.astype(np.float64)[:, None])              # (m, 8)

        sort_idx = np.argsort(order)
        sorted_order = order[sort_idx].astype(np.int64)
        rank_position = sort_idx.astype(np.int64)

        # FLUX_MEAN (mJy) and LOG10_FLUX_COV's between term, from the 2m
        # (template, design) points weighted by p_theta * p_k (section 6.1):
        # the moments kernel is unchanged, called once per design (its own
        # (m, 8) log10 flux and (m,) weight), the two LINEAR weighted sums
        # added -- correct since p_theta*p_k[0] + p_theta*p_k[1] = p_theta,
        # which already sums to 1 over templates.
        out_mean_sum = np.zeros(N_BANDS, dtype=np.float64)
        topk_flux_by_law = [None, None]
        mean_log10 = np.zeros(N_BANDS, dtype=np.float64)
        m2_log10 = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)
        for k in active_laws:
            wk = (p_theta * p_k[k]).astype(np.float64)
            out_mean = np.empty((1, N_BANDS), dtype=np.float64)
            out_m2 = np.empty((1, N_BANDS, N_BANDS), dtype=np.float64)
            out_topk_flux = np.full((1, topk, N_BANDS), np.nan, dtype=np.float32)
            _flux_moments_topk_kernel(
                log10_flux_by_law[k][None, :, :], wk[None, :], sorted_order[None, :],
                rank_position[None, :], np.array([good]), out_mean, out_m2, out_topk_flux)
            out_mean_sum += out_mean[0]
            topk_flux_by_law[k] = out_topk_flux[0]
            # the between term: the weighted mean/second moment of LOG10 flux
            # itself (not the kernel's own linear moments), summed over the
            # two designs the same way the mean above is.
            weighted_log10 = log10_flux_by_law[k] * wk[:, None]
            mean_log10 += weighted_log10.sum(axis=0)
            m2_log10 += weighted_log10.T @ log10_flux_by_law[k]
        flux_mean = out_mean_sum
        between = m2_log10 - np.outer(mean_log10, mean_log10)

        # the within term: sum_k P_k * D_k Sigma_k D_k^T, P_k each design's
        # own total posterior weight, one (8, 8) matrix per design (D_k
        # Sigma_k D_k^T does not depend on the template). A design left out
        # of `active_laws` carries P_k = 0 exactly (its own ln_w_k is -inf
        # everywhere, so p_k is 0 everywhere), so it contributes nothing and
        # is skipped rather than computed and multiplied by zero.
        within = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)
        p_total = [0.0, 0.0]
        for k in active_laws:
            fit_k = fits[k]
            p_total[k] = float(np.sum(p_theta * p_k[k]))
            d_k = np.column_stack([fit_k.ext_col.astype(np.float64),
                                    np.full(N_BANDS, likelihood.GRAY_COLUMN, dtype=np.float64)])
            within += p_total[k] * (d_k @ fit_k.xtwx_inv @ d_k.T)
        log10_flux_cov = between + within

        if not good:
            flux_mean = np.full(N_BANDS, np.nan)
            log10_flux_cov = np.full((N_BANDS, N_BANDS), np.nan)

        # FRAC_CLAMPED: the p_k-weighted mean of the two designs' own clamp
        # fractions, the same P_k total weight `within` used; a skipped
        # design's own P_k is 0, so its (unread) clamp fraction contributes
        # nothing either way.
        frac_clamped_1 = fits[1].frac_clamped if fits[1] is not None else np.float32(0.0)
        frac_clamped = np.float32(p_total[0] * fits[0].frac_clamped + p_total[1] * frac_clamped_1)

        topk_model = np.full(topk, -1, dtype=np.int32)
        topk_a_k = np.full(topk, np.nan, dtype=np.float32)
        topk_log10_b = np.full(topk, np.nan, dtype=np.float32)
        topk_chi2 = np.full(topk, np.nan, dtype=np.float32)
        topk_ln_l = np.full(topk, np.nan, dtype=np.float32)
        topk_ln_prior = np.full(topk, np.nan, dtype=np.float32)
        topk_law = np.full(topk, -1, dtype=np.int8)
        topk_ln_gamma = np.full(topk, np.nan, dtype=np.float32)
        topk_flux = np.full((topk, N_BANDS), np.nan, dtype=np.float32)
        if good:
            topk_model[:k_keep] = order.astype(np.int32)
            topk_law[:k_keep] = topk_law_sel
            for k in (0, 1):
                sel = topk_law_sel == k
                if not sel.any():
                    continue
                idx_sel = order[sel]
                fit_k = fits[k]
                topk_a_k[:k_keep][sel] = fit_k.a_hat_clamped[idx_sel]
                topk_log10_b[:k_keep][sel] = fit_k.log10_b_hat_clamped[idx_sel]
                topk_chi2[:k_keep][sel] = fit_k.chi2_min[idx_sel]
                topk_ln_l[:k_keep][sel] = ln_l[k][idx_sel].astype(np.float32)
                topk_ln_prior[:k_keep][sel] = ln_lambda[k][idx_sel]
                topk_ln_gamma[:k_keep][sel] = ln_gamma[k][idx_sel].astype(np.float32)
                topk_flux[:k_keep][sel] = topk_flux_by_law[k][:k_keep][sel]
        else:
            occam_gap = float("nan")

        zero_ext_count = int(sum(int((fits[k].a_hat < 0.0).sum()) for k in active_laws)) if good else 0
        n_templates_checked = len(active_laws) * n_model if good else 0

        row = dict(
            ln_evidence=ln_evidence64.astype(np.float32),
            flux_mean=flux_mean.astype(np.float32), log10_flux_cov=log10_flux_cov.astype(np.float32),
            topk_model=topk_model, topk_a_k=topk_a_k, topk_log10_b=topk_log10_b,
            topk_chi2=topk_chi2, topk_ln_l=topk_ln_l, topk_ln_prior=topk_ln_prior,
            topk_flux=topk_flux, topk_law=topk_law, topk_ln_gamma=topk_ln_gamma,
            occam_gap=np.float32(occam_gap),
            a_k_post=np.float32(a_post), a_k_post_sig=np.float32(a_post_sig),
            p_dense=np.float32(p_dense),
            frac_clamped=frac_clamped, n_detected=np.int8(batch.n_detected),
        )
        return dict(i=i, failed=False, zero_ext_count=zero_ext_count,
                     n_templates_checked=n_templates_checked, **row)
    except Exception as exc:  # rule: one bad source must not lose the job
        row = _empty_row(topk, n_sub)
        return dict(i=i, failed=True, zero_ext_count=0, n_templates_checked=0,
                     error=str(exc), **row)


# ---------------------------------------------------------------------------
# the parent side: load once, dispatch the pool, write parts, join
# ---------------------------------------------------------------------------

def _part_path(path, bi):
    return "%s.part%d" % (path, bi)


def _discover_parts(path):
    """Every `<path>.partN` file already on disk, sorted by `N` (item 9): a
    restart's own re-run (`--batches`) writes only the batches it names,
    and `join_parts` re-discovers every part present -- from this run or an
    earlier one -- rather than trusting an in-memory list a crash would
    have lost.
    """
    pattern = re.compile(re.escape(path) + r"\.part(\d+)$")
    found = {}
    for p in glob.glob(glob.escape(path) + ".part*"):
        m = pattern.match(p)
        if m:
            found[int(m.group(1))] = p
    return [found[k] for k in sorted(found)]


def _assemble_batch(results, name_slice, n_sub, topk):
    """Folds one batch's own per-source task results (already in catalogue
    order, `build_region_class`'s own `imap` consumption) into the batch-
    sized arrays `_write_part` writes -- a batch-sized array, never a
    region-sized one (rule 10b)."""
    m = len(results)
    ln_evidence = np.empty((m, n_sub), dtype=np.float32)
    flux_mean = np.empty((m, N_BANDS), dtype=np.float32)
    log10_flux_cov = np.empty((m, N_BANDS, N_BANDS), dtype=np.float32)
    topk_model = np.empty((m, topk), dtype=np.int32)
    topk_a_k = np.empty((m, topk), dtype=np.float32)
    topk_log10_b = np.empty((m, topk), dtype=np.float32)
    topk_chi2 = np.empty((m, topk), dtype=np.float32)
    topk_ln_l = np.empty((m, topk), dtype=np.float32)
    topk_ln_prior = np.empty((m, topk), dtype=np.float32)
    topk_flux = np.empty((m, topk, N_BANDS), dtype=np.float32)
    topk_law = np.empty((m, topk), dtype=np.int8)
    topk_ln_gamma = np.empty((m, topk), dtype=np.float32)
    occam_gap = np.empty(m, dtype=np.float32)
    a_k_post = np.empty(m, dtype=np.float32)
    a_k_post_sig = np.empty(m, dtype=np.float32)
    p_dense = np.empty(m, dtype=np.float32)
    frac_clamped = np.empty(m, dtype=np.float32)
    n_detected = np.empty(m, dtype=np.int8)
    failed = np.zeros(m, dtype=bool)
    zero_ext_count = 0
    n_templates_checked = 0
    for k, r in enumerate(results):
        ln_evidence[k] = r["ln_evidence"]
        flux_mean[k] = r["flux_mean"]
        log10_flux_cov[k] = r["log10_flux_cov"]
        topk_model[k] = r["topk_model"]
        topk_a_k[k] = r["topk_a_k"]
        topk_log10_b[k] = r["topk_log10_b"]
        topk_chi2[k] = r["topk_chi2"]
        topk_ln_l[k] = r["topk_ln_l"]
        topk_ln_prior[k] = r["topk_ln_prior"]
        topk_flux[k] = r["topk_flux"]
        topk_law[k] = r["topk_law"]
        topk_ln_gamma[k] = r["topk_ln_gamma"]
        occam_gap[k] = r["occam_gap"]
        a_k_post[k] = r["a_k_post"]
        a_k_post_sig[k] = r["a_k_post_sig"]
        p_dense[k] = r["p_dense"]
        frac_clamped[k] = r["frac_clamped"]
        n_detected[k] = r["n_detected"]
        failed[k] = r["failed"]
        zero_ext_count += r["zero_ext_count"]
        n_templates_checked += r["n_templates_checked"]
    return dict(name=name_slice, ln_evidence=ln_evidence, flux_mean=flux_mean, log10_flux_cov=log10_flux_cov,
                topk_model=topk_model, topk_a_k=topk_a_k, topk_log10_b=topk_log10_b,
                topk_chi2=topk_chi2, topk_ln_l=topk_ln_l, topk_ln_prior=topk_ln_prior,
                topk_flux=topk_flux, topk_law=topk_law, topk_ln_gamma=topk_ln_gamma, occam_gap=occam_gap,
                a_k_post=a_k_post, a_k_post_sig=a_k_post_sig, p_dense=p_dense,
                frac_clamped=frac_clamped,
                n_detected=n_detected, failed=failed,
                zero_ext_count=zero_ext_count, n_templates_checked=n_templates_checked)


def _write_part(part_path, batch):
    with h5py.File(part_path, "w") as f:
        for key in _PART_KEYS:
            f.create_dataset(key, data=batch[_FIELD_OF_KEY[key]])
        f.create_dataset("FAILED", data=batch["failed"])
        f.attrs["ZERO_EXT_COUNT"] = batch["zero_ext_count"]
        f.attrs["N_TEMPLATES_CHECKED"] = batch["n_templates_checked"]


def build_region_class(config, region, cls, st, reader, catalog,
                        n_workers=1, limit=None, batches=None):
    """Sweeps one {region, class}'s whole region (or, with `limit`, its
    first `limit` catalogue rows only -- a timing/acceptance device, never
    a default) one source at a time, over a `multiprocessing` pool of
    `n_workers` (section 1.3; IMPLEMENTATION_BMSTP_DRAFT.md section 4 row
    2.4; PARALLEL brief). Sources are grouped into batches of `[fit]
    batch_size` only as the part-file write granularity; each batch's own
    rows are written straight to their own part file (rule 10b). `batches`,
    given, restricts this call to those batch indices only (item 9: a
    restart writes just the missing parts); the caller (`build`) always
    joins whatever part files are on disk after. Returns the summary
    numbers `join_parts` and the report need. `reader` is this class's own
    `Prior` (`build`'s own `prior_reader.load`, this class's only);
    `catalog` is `_load_region_catalog`'s one whole-region read,
    also loaded once by `build` and shared across classes.
    """
    template_log, subclass_idx, n_sub = _register(config, cls)
    n_model = template_log.shape[0]
    gaia_term = GaiaTerm(config, region)
    # gaia.GaiaTerm.warm's own docstring (item 4): its per-class register
    # and field-star marginal are lazily cached on first `ln_gamma` call by
    # design; left lazy, each of this call's own forked workers would
    # independently open that product on its own first task. Warmed here,
    # in the parent, before the pool below forks.
    gaia_term.warm(cls.lower())
    width_dex = _width_dex(config, region)
    topk = config.fit_topk

    lib_path = config_module.product_path(config, "fittp", "check", "library-resolution", "survey")
    sigma_lib_l = likelihood.sigma_lib_by_class(lib_path)[cls]

    n_source = catalog["flux"].shape[0]
    if limit is not None:
        n_source = min(n_source, limit)

    # item 8's memory disclosure: printed once, at the top of this stage, from
    # closed-form cost plus one documented constant -- no RAM measurement, no
    # auto-capping. The user reads this line and sets --workers themselves.
    # Two terms, because the array arithmetic alone understates a worker several
    # times over and would have the user oversubscribe a node: this source's own
    # arrays, and WORKER_PROCESS_FLOOR_MB, the floor every forked worker carries
    # whatever the library size. Two extinction-law designs fit every source
    # (section 2's mixture) whenever the dense one can contribute, so the
    # per-source working set the equivalents count prices is doubled; this
    # is the worst case (`active_laws` skips the dense design, and its own
    # fit/prior-read/Gamma cost, wherever the sightline's own w_i is 0 in
    # every cell).
    arrays_mb = 2 * n_model * N_BANDS * 4 * likelihood.WORKER_WORKING_SET_EQUIV / 1e6
    per_worker_mb = arrays_mb + WORKER_PROCESS_FLOOR_MB
    total_gb = per_worker_mb * n_workers / 1024.0
    print("fittp.sweep.%s [%s]: n_model=%d, per worker %.0f MB (%.0f MB arrays + %d MB "
          "process floor), --workers %d -> %.1f GB" % (cls, region, n_model, per_worker_mb,
                                                        arrays_mb, WORKER_PROCESS_FLOOR_MB,
                                                        n_workers, total_gb), flush=True)

    batch_size = config.fit_batch_size
    batch_bounds = [(s, min(s + batch_size, n_source)) for s in range(0, n_source, batch_size)]
    selected = set(batches) if batches is not None else None

    path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    _set_worker_state(config=config, cls=cls, reader=reader, gaia_term=gaia_term,
                       template_log=template_log, subclass_idx=subclass_idx, n_sub=n_sub,
                       width_dex=width_dex, topk=topk, n_model=n_model,
                       sigma_lib_l=sigma_lib_l,
                       flux=catalog["flux"], sigma=catalog["sigma"], origin=catalog["origin"],
                       f_lim50=catalog["f_lim50"])

    # numba JIT-compiles once per PROCESS, so a worker meeting a kernel cold
    # pays its compile memory privately -- measured at ~82 MB per kernel, which
    # 100 workers would carry 100 times over. Warming here, in the parent,
    # leaves the compiled code in pages every worker inherits copy-on-write:
    # measured private growth per worker falls from ~82 MB a kernel to ~1.6 MB.
    # The warm is one real source through the real task, so it compiles exactly
    # the signatures the workers go on to call (`_cell_sum` alone takes 21
    # arguments; an explicit signature list would be a second thing to keep
    # right). `_source_task` catches its own failures, so this cannot raise.
    numba.set_num_threads(1)
    _source_task(batch_bounds[0][0])

    # Forked below every load above, and below the warm -- the threading layer
    # pinned at import is fork-safe by numba's own contract.
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_init_worker) as pool:
        for bi, (bstart, bstop) in enumerate(batch_bounds):
            if selected is not None and bi not in selected:
                continue
            m = bstop - bstart
            results = [None] * m
            for r in pool.imap(_source_task, range(bstart, bstop), chunksize=_IMAP_CHUNKSIZE):
                results[r["i"] - bstart] = r
            batch = _assemble_batch(results, catalog["name"][bstart:bstop], n_sub, topk)
            _write_part(_part_path(path, bi), batch)
            st.tick(bi + 1, len(batch_bounds), "batches")

    density_file = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    lib, granule = ("sps", "region") if cls == "STAR" else \
        {"AGB": ("agb", "region"), "PAHC": ("pahc", "region"), "YSO": ("yso", "survey"),
         "H2S": ("h2shock", "survey"), "GAL": ("galz", "survey")}[cls]
    weights_file = config_module.product_path(
        config, "bmstp", "weights", lib, granule, region=(region if granule == "region" else None))
    return dict(
        path=path, n_source=n_source, n_model=n_model, n_batches=len(batch_bounds),
        subclasses=definitions.SUBCLASSES_OF[cls], library=definitions.CLASS_REGISTER[cls],
        density_file=density_file, weights_file=weights_file,
    )


def join_parts(summary, topk):
    """Joins one {region, class}'s part files into the final P7 product,
    one part's rows at a time, dataset by dataset (rule 10b: never a
    region-sized array); removes the part files once written. Item 9:
    the part list comes from `_discover_parts`, not an in-memory record
    of what THIS call wrote, so a join after a restart picks up parts an
    earlier, crashed run already left on disk. Raises if any batch's part
    file is still missing -- rule 6, fail on the impossible, naming what
    to rerun rather than joining a silently incomplete product. Also folds
    every part's own `ZERO_EXT_COUNT`/`N_TEMPLATES_CHECKED` (for the
    report's `zero_ext_frac`) and `FAILED` column (item 7's `FAILED_ROWS`,
    the count reported as `n_failed`) across every part, present or
    reproduced.
    """
    path = summary["path"]
    n_source = summary["n_source"]
    n_batches = summary["n_batches"]
    part_paths = _discover_parts(path)
    if len(part_paths) != n_batches:
        raise RuntimeError(
            "fittp.sweep.join_parts [%s]: %d of %d batch part files present for %s -- "
            "rerun with --batches naming the missing indices before joining"
            % (summary.get("cls"), len(part_paths), n_batches, path))
    zero_ext_count = 0
    n_templates_checked = 0
    failed_chunks = []
    with h5py.File(path, "w") as out:
        for key in _PART_KEYS:
            with h5py.File(part_paths[0], "r") as pf0:
                shape = (n_source,) + pf0[key].shape[1:]
                dtype = pf0[key].dtype
            out.create_dataset(key, shape=shape, dtype=dtype)
        offset = 0
        for part_path in part_paths:
            with h5py.File(part_path, "r") as pf:
                m = pf["NAME"].shape[0]
                for key in _PART_KEYS:
                    out[key][offset:offset + m] = pf[key][:]
                failed_here = np.nonzero(pf["FAILED"][:])[0]
                failed_chunks.append(failed_here + offset)
                zero_ext_count += int(pf.attrs["ZERO_EXT_COUNT"])
                n_templates_checked += int(pf.attrs["N_TEMPLATES_CHECKED"])
            offset += m
        failed_rows = (np.concatenate(failed_chunks) if failed_chunks
                        else np.array([], dtype=np.int64)).astype(np.int64)
        out.create_dataset("FAILED_ROWS", data=failed_rows)
        out.attrs["GRANULE"] = "source"
        out.attrs["CLASS"] = summary.get("cls")
        out.attrs["SUBCLASSES"] = np.array(summary["subclasses"], dtype="S8")
        out.attrs["LIBRARY"] = summary["library"]
        out.attrs["N_MODEL"] = summary["n_model"]
        out.attrs["K"] = topk
        out.attrs["WEIGHTS_FILE"] = summary["weights_file"]
        out.attrs["DENSITY_FILE"] = summary["density_file"]
    for part_path in part_paths:
        os.remove(part_path)
    zero_ext_frac = zero_ext_count / n_templates_checked if n_templates_checked else float("nan")
    return dict(zero_ext_frac=zero_ext_frac, n_failed=int(failed_rows.size))


def build(config, regions=None, classes=None, limit=None, n_workers=1, batches=None):
    """Writes `fittp/fit/<CLS>_fit_source__R.hdf5` for every {region,
    class} pair (default all thirty regions, all six classes; section
    1.3, IMPLEMENTATION_BMSTP_DRAFT.md P7). `limit` restricts every
    region to its first `limit` catalogue rows -- a timing/acceptance
    device for a class whose prior read does not fit the run budget on
    the whole region, never a default. `n_workers` sizes the per-{region,
    class} multiprocessing pool (item 6: `--workers` on the CLI, falling
    back to `[fittp] workers`, default 1 -- never auto-detected, never
    capped by the code). `batches`, given, restricts every {region, class}
    this call sweeps to those batch indices only (item 9). Per region,
    the whole catalogue (`_load_region_catalog`) is loaded once here and
    shared across every class this call sweeps; each class in `classes`
    then loads only its OWN `Prior` (`prior_reader.load`), never the
    other five's shape grids or template-weight tables.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    class_codes = classes if classes is not None else list(CLASSES)
    for region in region_names:
        catalog = _load_region_catalog(config, region)
        for cls in class_codes:
            with progress.Stage("fittp.sweep.%s" % cls, region) as st:
                reader = prior_reader.load(config, region, cls)
                summary = build_region_class(config, region, cls, st, reader,
                                              catalog, n_workers=n_workers, limit=limit,
                                              batches=batches)
                summary["cls"] = cls
                joined = join_parts(summary, config.fit_topk)
                with h5py.File(summary["path"], "r") as f:
                    occam = np.asarray(f["OCCAM_GAP"][:])
                occam_finite = occam[np.isfinite(occam)]
                occam_median = float(np.median(occam_finite)) if occam_finite.size else float("nan")
                st.done(summary["path"], n=summary["n_source"], n_model=summary["n_model"],
                        zero_ext_frac=joined["zero_ext_frac"], occam_gap_median=occam_median,
                        n_batches=summary["n_batches"], n_failed=joined["n_failed"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="default: all six -- every {region, class} pair is swept")
    parser.add_argument("--limit", type=int, default=None,
                         help="sweep only the first LIMIT catalogue rows of each region "
                              "(a timing/acceptance device, never a default)")
    parser.add_argument("--workers", type=int, default=None,
                         help="the per-{region, class} multiprocessing pool size, one task "
                              "per source; falls back to [fittp] workers (default 1). Never "
                              "auto-detected, never capped by the code -- read this stage's "
                              "own printed per-worker cost and set the number that fits")
    parser.add_argument("--batches", type=int, nargs="+", default=None,
                         help="rerun only these batch indices (0-based, by position in the "
                              "region's own source order at [fit] batch_size) -- e.g. after a "
                              "crash, regenerate just the missing part files; the join always "
                              "re-discovers every part file on disk")
    args = parser.parse_args()
    cfg = config_module.load(args.config)
    n_workers = args.workers if args.workers is not None else cfg.fittp_workers
    build(cfg, regions=args.regions, classes=args.classes, limit=args.limit,
          n_workers=n_workers, batches=args.batches)
