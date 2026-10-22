"""One {region, class}: the class evidence, subclass evidence and top-K
posterior record of SPEC_BMSTP_DRAFT.md section 1.3
(IMPLEMENTATION_BMSTP_DRAFT.md section 1.3 P7, section 4 row 2.4).

The unit of work is ONE SOURCE (PARALLEL brief): for source `s`, section 2's
two extinction-law designs (diffuse `k=d`, `w=0`; dense `k=D`, `w=1`) and
every template `theta` of the class's library,

    ln w_theta,k = ln <Lambda_C>_s,k(theta)   -- prior_reader.ln_prior, section 4.2,
                                                  the DESIGN's own cell weight, carrying
                                                  the non-detection factor (NONDET brief
                                                  section 1) AND the Gaia factor (NONDET2
                                                  brief section 1) at each cell's own a*
                 + ln L_hat_s,k(theta)         -- likelihood.fit(batch, ..., w)'s chi2
                                                  plus the per-source normalisation only,
                                                  this design (neither the non-detection
                                                  nor the Gaia term rides here any more)

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
`ln w_theta`; the subclass evidence is a `logsumexp` of `ln w_theta + ln
p_theta,k` over every template, `p_theta,k` the register's own per-template
stage fraction (`_register`, one-hot where a register carries none) --
a proper marginal, not a hard mask, so a kept template that represents a
mixed-stage set (WP-PRIOR-1's "ALSO" finding, ledger C1) is shared across
the subclasses it represents; the class evidence is unchanged by this
either way, since every template's own fractions sum to 1. `TOPK_LAW`
records, per top-K template, which design carried the larger `ln
w_theta,k`; the record's
per-design TOPK_* fields (`TOPK_A_K`, `TOPK_CHI2`, `TOPK_LN_L`, `TOPK_LN_L`,
`TOPK_LN_PRIOR_ML`, `TOPK_LN_GAMMA_ML`, `TOPK_FLUX`) are that design's own
numbers; the separable record's three additive fields (NONDET2 brief
section 2: `TOPK_LN_PRIOR_FIT`, `TOPK_LN_UNSEEN`, `TOPK_LN_GAIA`) are the
LAW-MIXED S0, S1 - S0, S2 - S1 instead, so their sum is `ln w_theta` itself
at every top-K template. `P_DENSE` is the source's own posterior weight on
the dense design, `sum_theta p_theta * p_D(theta)`.

Region and class are read once, in the PARENT process (`build_region_class`):
the register (`_register`), the class's own library-resolution number
(`likelihood.sigma_lib_by_class`), the Gaia term (`GaiaTerm`), the class's
own `Prior` reader (`prior_reader.load`, this class's only -- the prior
read applies no floor, so no other class's grid is read) and the region's
whole catalogue (`_load_region_catalog`: fluxes, uncertainties,
`ORIGIN_FNU`, `NAME`, and COMPLETENESS_2026-10-20.md's own measured
completeness curves -- `curve_tables`, `source_row8` (one row per source
per one of the eight bands, `BAND_KEYS` order, C6_twomass_pixel_depths.md:
2MASS's own column is never -1), `band_a`/`band_c`/`band_n`, read whole
off `catalog.limits.completeness`'s product into plain arrays rather than
kept as that reader's own h5py handle, unsafe to share across this
stage's fork). These are stashed in `_WORKER` and a
`multiprocessing.get_context("fork").Pool` is created AFTER that load, and
after one warm source has compiled every numba kernel in the parent so the
workers inherit the compiled code rather than each paying for it -- the
workers inherit `_WORKER` and that code by fork by
copy-on-write, never through `Pool`'s `initializer`/`initargs` (which
pickles) and never by opening a register, grid or product themselves.
Read once, shared pages, never per worker: every one of `_WORKER`'s own
arrays that `prior_reader.load`/`_load_region_catalog` read off disk is,
wherever the dataset allows it, a `numpy.memmap` (`prior_reader.
mmap_dataset`, MAPARRAYS section 1) opened once here, in the parent,
before the fork -- its pages already sit in the OS page cache, and every
forked worker's own mapping reads those SAME physical pages rather than a
private copy, which is what keeps this stage's own parent small at fork
(the small Python objects in `_WORKER` -- `config`, `topk`, the register's
own compressed arrays -- still reach a worker by fork's ordinary
copy-on-write, unchanged).
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
different answer: the non-detection term reads SESNA's own measured
differential completeness curve at the source's position (COMPLETENESS_
2026-10-20.md, `catalog.limits.completeness`'s product -- the same curve
the retention floor and the atlas's acceptance fraction read, not a
region-wide fit of any kind); Gamma and the prior read a design's own
unclamped optimum `(a_hat, log10_b_hat)`, never a mix of clamped and
unclamped marks for the two factors of the same evidence; and section 9's
Occam gap (NONDET2 brief section 2, restating NONDET.md section 8's ruling)
is `logsumexp_theta(S1) - max_theta(S1)`, the SAME vector `S1(theta)` (the
law-mixed prior-and-fit-and-non-detection total, `s1_arr` below) in both
terms -- Gamma in NEITHER term, unlike `ev_total` -- so it is non-negative
by construction (`logsumexp(v) >= max(v)` for any real `v`), a
library-volume-and-non-detection read, no longer a read of how well any one
template's photometric fit alone, or Gaia, favours it. A
flagged source's `LOG10_FLUX_MEAN`/`LOG10_FLUX_COV` are written `NaN`, not
the zero a `p_theta` of zero everywhere would otherwise silently
accumulate, since a flagged fit has no posterior to report a flux moment
of.

`A_K_POST`/`A_K_POST_SIG` (section 6.1's reported extinction mark, not
`TOPK_A_K`'s maximum-likelihood one) fold `prior_reader.ln_prior`'s own
per-template, per-design posterior first/second moment of `a` over
templates and designs with the joint weight `p_theta * p_k(theta)`; a
flagged source writes `NaN` for both, the same convention as the flux
moments.

`LOG10_FLUX_MEAN` is the `p_theta * p_k`-weighted mean, in log10 flux, of
the `(template, design)` pairs' own log10 flux at their clamped marks --
the imputed photometry's own first moment, on the one scale section 6.1
stores the mean and covariance on (`FLUX_MEAN`, the same mean in linear
mJy, is gone: a second moment of the same distribution on a second scale
is a second, silently different number). `LOG10_FLUX_COV` (section 1, replacing the old
`FLUX_COV`) is the total variance of the imputed flux in log10 flux,
`between + within`: `between` is the `p_theta * p_k`-weighted covariance
of the same `(template, design)` pairs' log10 flux about `LOG10_FLUX_MEAN`
(the spread across templates AND designs); `within` is `sum_k P_k * D_k
Sigma_k D_k^T`, `P_k` each design's own total posterior weight
(`sum_theta p_theta * p_k(theta)`), `D_k = [ext_col_k, GRAY_COLUMN]` that
design's own `(8, 2)` design and `Sigma_k = xtwx_inv_k` its own `(2, 2)`
parameter covariance (`likelihood.fit`'s own return) -- one `(8, 8)`
matrix per design, not per template, since `D_k Sigma_k D_k^T` does not
depend on the template. `TOPK_FLUX` stays linear mJy, the top-K record's
own per-template value, unaffected by either rename.
"""

import errno
import glob
import multiprocessing as mp
import os
import re
import time

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

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.catalog import limits as catalog_limits
from sesnaimpute.fittp import likelihood
from sesnaimpute.fittp import prior_reader
from sesnaimpute.fittp.gaia import GaiaTerm
from sesnaimpute.granules import access as granule_access

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: the seven classes, in the fitter's own class-axis order
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
#: refcounts them. MEASURED ON THE CLUSTER, 2026-10-08:
#: `/proc/<pid>/smaps_rollup` read from inside the running Aquila YSO job
#: on an spr node at 112 workers (the largest library, 200,000 templates)
#: gives a mean `Pss` of 72.8 MB and a mean `Private_Dirty` of 69.0 MB per
#: worker -- `Pss` being the right measure, since it apportions a shared
#: mapping's pages fairly across the workers sharing it. Rounded to 70.
#: The previous value, 430, was a macOS `vmmap` peak from the laptop and
#: overstated this by six times.
#:
#: The TOTAL the stage prints from this (arrays + floor, times the worker
#: count) is still an UPPER BOUND, because the arrays term counts pages
#: that are now memory-mapped and therefore shared rather than private to
#: each worker. The same measurement put that whole job's real footprint
#: at 7.96 GB of summed `Pss` across its 112 workers, against the 61 GB
#: the formula printed with the old floor. Treat the printed number as a
#: ceiling when capping `--workers`, not as the cost.
#:
#: `build_region_class`'s warm-up is what keeps the JIT out of this
#: figure: without it every worker compiles its own kernels and the floor
#: is ~180 MB higher.
WORKER_PROCESS_FLOOR_MB = 70

#: The datasets every P7 part file and the joined product carry, in write
#: order (one row per source; `FAILED`, below, is a part-file-only column
#: consumed at join time into `FAILED_ROWS`, never copied into the joined
#: product itself). `N_LAW_ITER` is gone: the self-consistent
#: extinction-law solve it counted no longer exists; `TOPK_LAW` and
#: `P_DENSE` are new since section 2/6.4. NONDET2 brief section 2: `
#: TOPK_LN_PRIOR` (which silently carried prior x non-detection after
#: NONDET) and `TOPK_LN_GAMMA` (the Gaia term at one point, used inside
#: the evidence) are gone; the separable record in their place is `
#: TOPK_LN_PRIOR_FIT` (S0), `TOPK_LN_UNSEEN` (S1 - S0), `TOPK_LN_GAIA`
#: (S2 - S1) -- the three EXACT additive marginal factors summing to `
#: ln_w` -- plus two report-only guides, `TOPK_LN_PRIOR_ML` (the prior
#: alone at the best-fit point, today's `TOPK_LN_PRIOR` before NONDET)
#: and `TOPK_LN_GAMMA_ML` (the Gaia term at the best-fit point, today's
#: `TOPK_LN_GAMMA`).
_PART_KEYS = ("NAME", "LN_EVIDENCE", "LOG10_FLUX_MEAN", "LOG10_FLUX_COV", "TOPK_MODEL", "TOPK_A_K",
              "TOPK_LOG10_B", "TOPK_CHI2", "TOPK_LN_L",
              "TOPK_LN_PRIOR_FIT", "TOPK_LN_UNSEEN", "TOPK_LN_GAIA",
              "TOPK_LN_PRIOR_ML", "TOPK_LN_GAMMA_ML",
              "TOPK_FLUX", "TOPK_LAW", "OCCAM_GAP", "A_K_POST", "A_K_POST_SIG", "P_DENSE",
              "FRAC_CLAMPED", "N_DETECTED",
              "LN_EVIDENCE_MEASURED", "LN_EVIDENCE_NONDET", "LN_PRIOR_MASS")

#: Every dataset the joined fit file carries, including `FAILED_ROWS`
#: (join-time only, not a `_PART_KEYS` member), has its `UNITS`/`READING`
#: (CODING_RULES_BMSTP.md rule 5) in `attrs_registry.REGISTRY`, keyed by
#: `(cls + "_fit_source", name)` -- `_STEM`, below -- the same text
#: repeated for every one of the seven classes' own files.
def _STEM(cls):
    return "%s_fit_source" % cls


_FIELD_OF_KEY = {
    "NAME": "name", "LN_EVIDENCE": "ln_evidence", "LOG10_FLUX_MEAN": "log10_flux_mean",
    "LOG10_FLUX_COV": "log10_flux_cov", "TOPK_MODEL": "topk_model", "TOPK_A_K": "topk_a_k",
    "TOPK_LOG10_B": "topk_log10_b", "TOPK_CHI2": "topk_chi2", "TOPK_LN_L": "topk_ln_l",
    "TOPK_LN_PRIOR_FIT": "topk_ln_prior_fit", "TOPK_LN_UNSEEN": "topk_ln_unseen",
    "TOPK_LN_GAIA": "topk_ln_gaia", "TOPK_LN_PRIOR_ML": "topk_ln_prior_ml",
    "TOPK_LN_GAMMA_ML": "topk_ln_gamma_ml",
    "TOPK_FLUX": "topk_flux", "TOPK_LAW": "topk_law", "OCCAM_GAP": "occam_gap",
    "A_K_POST": "a_k_post", "A_K_POST_SIG": "a_k_post_sig", "P_DENSE": "p_dense",
    "FRAC_CLAMPED": "frac_clamped", "N_DETECTED": "n_detected",
    "LN_EVIDENCE_MEASURED": "ln_evidence_measured", "LN_EVIDENCE_NONDET": "ln_evidence_nondet",
    "LN_PRIOR_MASS": "ln_prior_mass",
}


def _register(config, cls):
    """`(template_log, n_sub, subclass_prob)` for `cls`'s own library
    register (`definitions.CLASS_REGISTER`): `template_log` is `(n_model,
    8)` float32 `log10 F_REF` (floored at the register's own
    `FLOOR_LINEAR`, its FREFRAW convention -- a genuinely dark band is not
    `-inf`), in `definitions.BANDS` order. `subclass_prob` is `(n_model,
    n_sub)` float64, every row summing to 1, in `definitions.
    SUBCLASSES_OF[cls]`'s own order (P7's `LN_EVIDENCE` column order):
    the register's own `subclass_prob` group's per-template stage
    fractions where the register carries one (today, YSO only --
    WP-PRIOR-1's "ALSO" finding, ledger C1: the register's own evidence
    that a kept template represents a mixed-stage set, non-one-hot on
    about 46% of rows); a one-hot row at the register's hard `SUBCLASS`
    label otherwise, so the subclass evidence below reduces exactly to
    today's hard-label partition for every class without a
    `subclass_prob` group.
    """
    key = definitions.CLASS_REGISTER[cls]
    path = os.path.join(config.inputs["sed_models"], "registers", "%s_register.hdf5" % key)
    subclass_order = definitions.SUBCLASSES_OF[cls]
    with h5py.File(path, "r") as f:
        n_model = f["models"]["MODEL_NAME"].shape[0]
        f_ref = np.empty((n_model, N_BANDS), dtype=np.float64)
        for j, bkey in enumerate(BAND_KEYS):
            f_ref[:, j] = np.asarray(f["models"]["F_REF_%s" % bkey][:], dtype=np.float64)
        floor_linear = np.asarray(f["models"]["FLOOR_LINEAR"][:], dtype=np.float64)
        subclass_raw = f["models"]["SUBCLASS"][:]
        subclass_names = [s.decode() if isinstance(s, bytes) else s for s in subclass_raw]
        sub_to_idx = {s: i for i, s in enumerate(subclass_order)}
        unknown = set(subclass_names) - set(sub_to_idx)
        if unknown:
            raise ValueError("fittp.sweep: register %r carries SUBCLASS %r not in "
                              "definitions.SUBCLASSES_OF[%r] = %r"
                              % (path, sorted(unknown), cls, subclass_order))
        subclass_idx = np.array([sub_to_idx[s] for s in subclass_names], dtype=np.intp)

        if "subclass_prob" in f:
            missing = [s for s in subclass_order if s not in f["subclass_prob"]]
            if missing:
                raise ValueError(
                    "fittp.sweep: register %r carries 'subclass_prob' but is missing "
                    "column(s) %r of definitions.SUBCLASSES_OF[%r] = %r"
                    % (path, missing, cls, subclass_order))
            subclass_prob = np.stack(
                [np.asarray(f["subclass_prob"][s][:], dtype=np.float64) for s in subclass_order],
                axis=1)
        else:
            subclass_prob = None
    template_log = np.log10(np.maximum(f_ref, floor_linear[:, None])).astype(np.float32)

    if subclass_prob is None:
        # no register carries a per-template stage-fraction table for
        # this class: a one-hot row at the hard label, exactly today's
        # partition (built this way so the fitter's own formula, not a
        # special case, produces the identical result).
        subclass_prob = np.zeros((n_model, len(subclass_order)), dtype=np.float64)
        subclass_prob[np.arange(n_model), subclass_idx] = 1.0
    else:
        row_sum = subclass_prob.sum(axis=1)
        bad = ~np.isclose(row_sum, 1.0, atol=1e-6)
        if bad.any():
            raise ValueError(
                "fittp.sweep: register %r 'subclass_prob' rows do not sum to 1 (max "
                "|sum-1|=%.3g on %d/%d rows) -- not a proper per-template stage fraction"
                % (path, float(np.max(np.abs(row_sum[bad] - 1.0))), int(bad.sum()), n_model))

    return template_log, len(subclass_order), subclass_prob


def _prior_mass_by_source(config, region, cls, n_source):
    """C7 rework (2026-10-20): `(n_source,) float32, (n_source,) bool` --
    `LN_PRIOR_MASS` and whether the source's own pixel matched -- read
    once per {region, class}, in the parent, never per source.

    The owner's ruling: the brief's first try at this column
    (`fittp.prior_reader._cell_sum`'s own `out_cat`/`_p_cat`, since
    deleted) integrated the catalogue-admission probability over the
    FIT's own +/-5 sigma window, so it depended on the data through the
    window's position, and read -inf for most of a class's own sources
    wherever the only templates that could fit needed extinction outside
    the window's own catalogued range (measured: AGB 300 -> 4 finite on
    the C7 slice). The quantity actually wanted -- the prior's own
    catalogued mass over its FULL extinction support, no likelihood
    anywhere -- is already priced once per admitted nside-512 pixel by
    `bmstp.atlas`'s own prior atlas (`N_CAT_<cls>`, deg^-2: the class's
    whole population at the pixel's own MEAN column, every member's own
    probability of clearing SESNA's two-of-eight rule at the pixel's own
    measured completeness curves -- `bmstp.atlas.build_region`'s own
    module docstring). Re-deriving it per source and per cell (the
    honest, data-independent version of the Part B attempt: every
    template's FULL support, no window) would cost ~8 curve reads x 96
    cells x every template per source -- measured ~2 s/source for YSO,
    unaffordable over a whole region. This read is the affordable,
    EXACT quantity instead: one pixel lookup per source.

    Each source's own sky pixel comes from the granule map's own
    `HPX_PIX_512` (`granules.sesna.granule-map.source`, the same
    per-source join `sky.derived.gaia_match` and C7 Part A's own
    `fittp.gaia._load_source_xyz` already read), looked up in the atlas's
    own `HPX_PIX_512` (verified sorted ascending below -- `np.searchsorted`
    assumes it). A source whose own pixel is not one of the atlas's
    admitted pixels is NOT every source's: `matched` is False there and
    `LN_PRIOR_MASS` is NaN, counted by the caller (`st.done`'s
    `n_prior_mass_unmapped`), never silently zero. An admitted pixel
    whose own `N_CAT_<cls>` is exactly zero reads `-inf` -- correct: no
    such objects there, not a defect.
    """
    granule_map_path = config_module.product_path(config, "granules", "sesna", "granule-map", "source")
    pix512 = np.asarray(
        granule_access.per_source(config, region, granule_map_path, ["HPX_PIX_512"])["HPX_PIX_512"],
        dtype=np.int64)
    if pix512.shape[0] != n_source:
        raise ValueError(
            "fittp.sweep._prior_mass_by_source [%s]: granule map has %d source rows, "
            "the region's own catalogue has %d -- row counts cannot be reconciled"
            % (region, pix512.shape[0], n_source))
    atlas_path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    with h5py.File(atlas_path, "r") as f:
        atlas_pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        n_cat = np.asarray(f["N_CAT_%s" % cls][:], dtype=np.float64)
    if atlas_pix.size and np.any(np.diff(atlas_pix) <= 0):
        raise ValueError(
            "fittp.sweep._prior_mass_by_source [%s/%s]: %s's own HPX_PIX_512 is not "
            "sorted ascending -- the searchsorted lookup below assumes it is"
            % (region, cls, atlas_path))
    loc = np.searchsorted(atlas_pix, pix512)
    capped = np.clip(loc, 0, max(atlas_pix.size - 1, 0))
    matched = (atlas_pix.size > 0) & (atlas_pix[capped] == pix512)
    ln_prior_mass = np.full(n_source, np.nan, dtype=np.float32)
    # bmstp.atlas's own Monte Carlo quadrature can leave a true zero as a
    # tiny negative float (measured: YSO as low as -4.8e-18 at an admitted
    # pixel) -- clipped at 0 before the log so that reads -inf (a zero
    # count, correctly), never NaN from log of a negative number.
    with np.errstate(divide="ignore"):
        ln_n_cat = np.log(np.maximum(n_cat[capped], 0.0))
    ln_prior_mass[matched] = ln_n_cat[matched].astype(np.float32)
    return ln_prior_mass, matched


def _load_region_catalog(config, region):
    """This region's whole catalogue, read ONCE (item 4): fluxes,
    uncertainties and `ORIGIN_FNU` (the same read `fittp.cascade` uses),
    `NAME`, and the region's own measured completeness curves
    (COMPLETENESS_2026-10-20.md, `catalog.limits.completeness`'s own
    product) -- so every worker's own per-source task slices these
    already-loaded, fork-inherited arrays instead of opening either
    product itself.

    The reader's `CURVE_<B>` tables are h5py datasets tied to one open
    file handle, unsafe to share across the pool's fork (`catalog.limits.
    Completeness`'s own docstring: "a caller that forks workers should
    open one reader per worker, not share this one across a process
    boundary"); mapped here instead with `prior_reader.mmap_dataset`
    (the same technique `FNU_MJY`/`SIGMA_FNU_MJY` already use) -- a
    `numpy.memmap` opened straight on the completeness product's own
    file and byte offset, so it carries no live HDF5 handle at all once
    created and is exactly as fork-safe as every other array this stage
    reads once in the parent (MAPARRAYS section 1): its pages live once
    in the OS page cache, read by every worker, never copied or pickled
    per piece. The reader's own h5py file handle is closed immediately
    after, since nothing here still needs it.
    """
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        # memory-mapped (MAPARRAYS section 1): FNU_MJY/SIGMA_FNU_MJY are
        # already float64 on disk, so this is the same values as the old
        # `.astype(np.float64)`, not held a second time in process memory.
        flux = prior_reader.mmap_dataset(f, "FNU_MJY")
        sigma = prior_reader.mmap_dataset(f, "SIGMA_FNU_MJY")
        origin = prior_reader.mmap_dataset(f, "ORIGIN_FNU")
        name = prior_reader.mmap_dataset(f, "NAME")

    comp = catalog_limits.completeness(config, region)
    band_a = np.empty(N_BANDS, dtype=np.float64)
    band_c = np.empty(N_BANDS, dtype=np.float64)
    band_n = np.empty(N_BANDS, dtype=np.int64)
    curve_tables = {}
    for b, band in enumerate(BAND_KEYS):
        a, c = comp._ac[band]
        band_a[b], band_c[b] = a, c
        band_n[b] = comp._mag_grid_band[band].size
        curve_tables[band] = prior_reader.mmap_dataset(comp._f, "CURVE_%s" % band)
    source_row8 = np.asarray(comp.source_row, dtype=np.int64)
    max_n = int(band_n.max())
    comp._f.close()

    return dict(flux=flux, sigma=sigma, origin=origin, name=name,
                curve_tables=curve_tables, source_row8=source_row8,
                band_a=band_a, band_c=band_c, band_n=band_n, max_n=max_n)


def _n_sources(config, region):
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        return f["NAME"].shape[0]


@numba.njit(parallel=True, cache=True)
def _topk_flux_kernel(log10_flux, sorted_order, rank_position, good, out_topk_flux):
    """The top-K flux record (`TOPK_FLUX`, linear mJy), one pass per
    source (`prange`, item 5's own worker-side thread cap: called here
    with a leading axis of 1, one source at a time -- the kernel's own
    generality over many sources is unused, not a block): for each
    template `t` already chosen by the argpartition/argsort on `ln_w`
    before this kernel runs (`_source_task`), the linear flux `f = 10**
    log10_flux[s, t, :]` is copied into `out_topk_flux` at that template's
    rank. `sorted_order` is the source's own top-K template indices sorted
    ascending, and `rank_position` maps each ascending slot back to its
    position in the ln_w-descending top-K list, so the O(1) pointer walk
    below (advancing only when `t` reaches the next sorted index) lands
    each flux in the same slot `topk_model`/`topk_a_k`/etc. use for that
    template. `good[s]` false (a flagged source) skips the write --
    `out_topk_flux` was pre-filled with NaN by the caller."""
    n_block = log10_flux.shape[0]
    n_model = log10_flux.shape[1]
    n_bands = log10_flux.shape[2]
    k_keep = sorted_order.shape[1]
    for s in numba.prange(n_block):
        f = np.empty(n_bands)
        ptr = 0
        row_good = good[s]
        if not row_good:
            continue
        for t in range(n_model):
            if ptr >= k_keep:
                break
            if t == sorted_order[s, ptr]:
                for b in range(n_bands):
                    f[b] = 10.0 ** log10_flux[s, t, b]
                r = rank_position[s, ptr]
                for b in range(n_bands):
                    out_topk_flux[s, r, b] = f[b]
                ptr += 1


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
    _cell_sum`, `fittp.gaia._gaia_h_kernel`, `_topk_flux_kernel`
    above) and BLAS's own small gemms threading too would oversubscribe
    the machine. Must run before any of those kernels executes in this
    worker for the first time, which it does: nothing calls one before a
    task does."""
    numba.set_num_threads(1)
    threadpoolctl.threadpool_limits(1, user_api="blas")


def _empty_row(topk, n_sub):
    return dict(
        ln_evidence=np.full(n_sub, -np.inf, dtype=np.float32),
        log10_flux_mean=np.full(N_BANDS, np.nan, dtype=np.float32),
        log10_flux_cov=np.full((N_BANDS, N_BANDS), np.nan, dtype=np.float32),
        topk_model=np.full(topk, -1, dtype=np.int32),
        topk_a_k=np.full(topk, np.nan, dtype=np.float32),
        topk_log10_b=np.full(topk, np.nan, dtype=np.float32),
        topk_chi2=np.full(topk, np.nan, dtype=np.float32),
        topk_ln_l=np.full(topk, np.nan, dtype=np.float32),
        topk_ln_prior_fit=np.full(topk, np.nan, dtype=np.float32),
        topk_ln_unseen=np.full(topk, np.nan, dtype=np.float32),
        topk_ln_gaia=np.full(topk, np.nan, dtype=np.float32),
        topk_ln_prior_ml=np.full(topk, np.nan, dtype=np.float32),
        topk_ln_gamma_ml=np.full(topk, np.nan, dtype=np.float32),
        topk_flux=np.full((topk, N_BANDS), np.nan, dtype=np.float32),
        topk_law=np.full(topk, -1, dtype=np.int8),
        occam_gap=np.float32(np.nan),
        ln_evidence_measured=np.float32(np.nan), ln_evidence_nondet=np.float32(np.nan),
        ln_prior_mass=np.float32(np.nan),
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

        # COMPLETENESS_2026-10-20.md "Per-source access in the fitter",
        # amended by C6_twomass_pixel_depths.md: this source's own eight
        # rows (one per band, BAND_KEYS order; 2MASS's own column is
        # never -1, C6_twomass_pixel_depths.md) into `catalog.limits.
        # completeness`'s own CURVE_<B> tables (`w["curve_tables"]`,
        # fork-inherited plain arrays, never re-read from disk), gathered
        # ONCE into one (8, max_n) array before the template loop
        # (`likelihood.prepare`/`_ln_nondet`, `prior_reader._cell_sum`).
        row8 = w["source_row8"][i]
        curve_rows = np.zeros((N_BANDS, w["max_n"]), dtype=np.float64)
        covered = np.empty(N_BANDS, dtype=bool)
        for b, band in enumerate(BAND_KEYS):
            n_b = w["band_n"][b]
            r = row8[b]
            covered[b] = r >= 0
            if r >= 0:
                curve_rows[b, :n_b] = w["curve_tables"][band][r]

        batch = likelihood.prepare(w["config"], flux, sigma, origin,
                                    w["sigma_lib_l"], curve_rows, covered,
                                    w["band_a"], w["band_c"], w["band_n"])

        n_model = w["n_model"]
        rows = np.array([i])
        model_index = np.arange(n_model)
        h = prior_reader.prepare(w["reader"], rows)

        # section 2's two-design mixture: the sightline's own per-cell dense
        # fraction w_i (fittp.prior_reader.load, from bmstp.cloud_interval
        # through P1's SIGHTLINE_ROW), 1 - w_i owned by the diffuse design,
        # w_i by the dense one -- never a per-hypothesis extinction mark.
        w_dense_cell = w["reader"].w_dense_row(i)[None, :]        # (1, n_x)
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

        # NONDET2 brief section 1: G_S, A_X and the register's per-template
        # G0MAG/KG_DRAINE/KG_WHITNEY are law-independent (the per-cell ramp
        # blend that varies with the extinction `a` is formed inside
        # `_cell_sum` itself, at each cell's own `a*`), so this reads once
        # per source, not once per design below.
        gaia_inputs = w["gaia_term"].cell_inputs(rows, model_index, w["cls"].lower())

        fits = [None, None]
        # NONDET2 section 2: the separable record's three totals, per
        # design, each WITHOUT ln_l (chi2 + normalisation, added below, per
        # law, before the two designs mix): ln_lambda0 carries neither the
        # non-detection nor the Gaia factor, ln_lambda1 the non-detection
        # factor only, ln_lambda2 both -- `ln_lambda`, the one array this
        # call returned before NONDET2.
        ln_lambda0 = [None, None]      # float32 (m,) per design -- S0, no ln_l
        ln_lambda1 = [None, None]      # float32 (m,) per design -- S1, no ln_l
        ln_lambda2 = [None, None]      # float32 (m,) per design -- S2, no ln_l
        a_post_t = [None, None]        # float64 (m,) per design
        a2_post_t = [None, None]
        # ln_l: chi2 plus the per-source normalisation (NONDET brief
        # section 1 -- the non-detection band charge no longer rides here,
        # it is already folded into ln_lambda, inside the prior's own cell
        # sum), PLUS, since S01 (review_2026-10-22/FINDINGS.md), the
        # evidence's own nuisance-integration volume (added below, once
        # per design k, never per template). ln_l_report keeps the OLD
        # full quantity (chi2 plus the non-detection charge AT THE CLAMPED
        # MARK, likelihood.fit's own `ln_nondet`), with NO volume term,
        # for TOPK_LN_L alone, which stays the maximum-likelihood-point
        # diagnostic it has always been (brief section 1; S01's reviewer
        # reply keeps this diagnostic untouched).
        ln_l = [None, None]             # float64 (m,) per design
        ln_l_report = [None, None]      # float64 (m,) per design
        # NONDET2 section 1: the Gaia term now rides inside ln_lambda2 (per
        # cell, at that cell's own a*); `ln_gamma_ml` is kept only for the
        # two report-only ML-point uses, TOPK_LN_GAMMA_ML and PSI_VOTES'
        # Gaia reading (`gaia.GaiaTerm.ln_gamma`, unchanged).
        ln_gamma_ml = [None, None]      # float64 (m,) per design
        for k in active_laws:
            blend_w = float(k)
            fit_k = likelihood.fit(batch, w["template_log"], blend_w)
            fits[k] = fit_k

            # d(log10 B)/d(a_K) from this design's own d(SC)/d(A_V)
            # (fit_k.slope_sc_av): log10_B = -2*SC, a_K = ak_per_av * A_V,
            # so d(log10_B)/d(a_K) = -2 * slope_sc_av / ak_per_av
            # (likelihood.fit's docstring, section 1.3).
            slope_log10b_per_ak = np.array([-2.0 * fit_k.slope_sc_av / fit_k.ak_per_av])
            # NONDET brief section 1: this design's own per-band dimming in
            # A_K (`-0.4 kappa_b`, likelihood.fit's own `ext_col` divided by
            # its `ak_per_av` undoes the A_V->A_K conversion built into
            # `ext_col`, EXACT_CODE_SITES), fed to the cell sum so it can
            # place the non-detection factor at each cell's own a*.
            ext_col_ak_k = fit_k.ext_col.astype(np.float64) / fit_k.ak_per_av
            # FINDINGS.md S03 (briefs/round6_2026-10-22/
            # U02_nondet_conditional.md): the conditional comonotone term
            # needs the detected bands' own p_d beside the undetected
            # bands' p_u, so `batch.detected_mask` rides beside
            # `batch.nondet_mask` into the cell sum, argument-list only.
            ln_lambda2_k, ln_lambda0_k, ln_lambda1_k, a_post_k, a2_post_k = prior_reader.ln_prior(
                w["reader"], rows, h, fit_k.a_hat[None, :], fit_k.log10_b_hat[None, :],
                slope_log10b_per_ak, np.array([fit_k.sigma_a_ak]), model_index,
                cell_weight_by_law[k],
                w["template_log"], ext_col_ak_k,
                batch.curve_rows[None, :, :],
                batch.band_a, batch.band_c, batch.band_n,
                batch.nondet_mask[None, :], batch.detected_mask[None, :], gaia_inputs)
            ln_lambda2[k] = ln_lambda2_k[0]
            ln_lambda0[k] = ln_lambda0_k[0]
            ln_lambda1[k] = ln_lambda1_k[0]
            a_post_t[k] = a_post_k[0]
            a2_post_t[k] = a2_post_k[0]

            # `likelihood._ln_nondet` has always added `batch.ln_norm_term`
            # in float32 (its own return dtype); reading it through the same
            # float32 cast here, rather than full float64, keeps a source
            # with nothing to charge bit-identical to `main` (NONDET brief
            # section 4 identity 3) -- the two are equal to float64's own
            # ~1e-7 relative precision either way, so nothing scientific
            # rides on which one this reads.
            ln_l[k] = -0.5 * fit_k.chi2_min.astype(np.float64) + np.float64(np.float32(batch.ln_norm_term))
            ln_l_report[k] = -0.5 * fit_k.chi2_min.astype(np.float64) + fit_k.ln_nondet.astype(np.float64)

            # S01 (review_2026-10-22/FINDINGS.md; reviewer reply narrowed
            # this to the EVIDENCE alone, not the peak-likelihood
            # diagnostic -- see TOPK_LN_L's own reading, unchanged): the
            # evidence's own nuisance-integration volume, `2 pi sqrt(det
            # Sigma_ab)`, Sigma_ab the likelihood's covariance in the
            # fitter's own (a, log10 B) coordinates, not its (A_V, SC)
            # ones. T = diag(ak_per_av, GRAY_COLUMN) maps (A_V, SC) ->
            # (a, log10_B) = (ak_per_av*A_V, GRAY_COLUMN*SC) (likelihood.
            # fit's own docstring, SPEC_BMSTP_DRAFT.md section 1.3), so
            # Sigma_ab = T xtwx_inv T^T and det Sigma_ab =
            # (ak_per_av*GRAY_COLUMN)^2 * det(xtwx_inv). This is one
            # scalar per (source, class, law k): `design` (and so
            # `xtwx_inv`/`ak_per_av`) carries no template axis in
            # likelihood.fit, so it is added once here to the whole (m,)
            # array, never per template. A flagged design (fewer than two
            # detected bands, or the eye(2) det<=0 substitute, `fit_k.
            # flagged`) gets 0 added -- chi2_min is already NaN there, as
            # today's code treats it. Added to `ln_l` (and so to every S0/
            # S1/S2 that sums it in) ONLY, never to `ln_l_report`: that
            # field stays TOPK_LN_L, the maximum-likelihood-point
            # diagnostic its own reading already describes, which this
            # finding does not touch.
            if fit_k.flagged:
                ln_vol_k = 0.0
            else:
                det_xtwx_inv = (fit_k.xtwx_inv[0, 0] * fit_k.xtwx_inv[1, 1]
                                 - fit_k.xtwx_inv[0, 1] * fit_k.xtwx_inv[1, 0])
                det_sigma_ab = (fit_k.ak_per_av * likelihood.GRAY_COLUMN) ** 2 * det_xtwx_inv
                ln_vol_k = float(np.log(2.0 * np.pi) + 0.5 * np.log(det_sigma_ab))
            ln_l[k] = ln_l[k] + ln_vol_k

            # Gamma reads this design's own unclamped optimum a_hat/log10_b_hat;
            # report-only now (NONDET2 section 1) -- the clamped marks stay
            # for the reported marks, the top-K record and the flux
            # prediction only, section 6.1.
            ln_gamma_ml[k] = w["gaia_term"].ln_gamma(
                rows, model_index, fit_k.a_hat[None, :], fit_k.log10_b_hat[None, :],
                w["cls"].lower())[0]

        if not dense_possible:
            ln_lambda0[1] = np.full(n_model, -np.inf, dtype=np.float32)
            ln_lambda1[1] = np.full(n_model, -np.inf, dtype=np.float32)
            ln_lambda2[1] = np.full(n_model, -np.inf, dtype=np.float32)
            ln_l[1] = np.full(n_model, -np.inf, dtype=np.float64)
            ln_l_report[1] = np.full(n_model, -np.inf, dtype=np.float64)
            ln_gamma_ml[1] = np.zeros(n_model, dtype=np.float64)
            a_post_t[1] = np.full(n_model, np.nan, dtype=np.float64)
            a2_post_t[1] = np.full(n_model, np.nan, dtype=np.float64)

        combined_flagged = fits[0].flagged or (fits[1].flagged if fits[1] is not None else False)

        # NONDET2 section 2: S0, S1, S2 are each a logsumexp over the two
        # designs of that design's own (cell-sum total + ln_l) -- the law
        # mixture is already applied per cell inside `_cell_sum`, so no
        # further weight is applied here. S2 is `ln_w` exactly: the Gaia
        # term no longer rides as a separate addend at this level.
        ln_lambda0_64 = [x.astype(np.float64) for x in ln_lambda0]
        ln_lambda1_64 = [x.astype(np.float64) for x in ln_lambda1]
        ln_lambda2_64 = [x.astype(np.float64) for x in ln_lambda2]
        s0_k = np.stack([ln_lambda0_64[k] + ln_l[k] for k in (0, 1)], axis=0)    # (2, m)
        s1_k = np.stack([ln_lambda1_64[k] + ln_l[k] for k in (0, 1)], axis=0)    # (2, m)
        ln_w_k = np.stack([ln_lambda2_64[k] + ln_l[k] for k in (0, 1)], axis=0)  # (2, m) == S2 per law
        if combined_flagged:
            s0_k = np.full((2, n_model), -np.inf, dtype=np.float64)
            s1_k = np.full((2, n_model), -np.inf, dtype=np.float64)
            ln_w_k = np.full((2, n_model), -np.inf, dtype=np.float64)

        with np.errstate(invalid="ignore"):
            s0_arr = logsumexp(s0_k, axis=0)                    # (m,) -- TOPK_LN_PRIOR_FIT
            s1_arr = logsumexp(s1_k, axis=0)                    # (m,)
        # the mixture (section 2): logsumexp over the two designs at each
        # template; the sightline's own w_i already makes the two designs'
        # PRIOR terms sum to the single-law read, so no further weight is
        # applied here (module docstring).
        ln_w = logsumexp(ln_w_k, axis=0)                        # (m,) == S2
        with np.errstate(invalid="ignore"):
            p_k = np.exp(ln_w_k - ln_w[None, :])                # (2, m), each design's own share
        p_k = np.where(np.isfinite(p_k), p_k, 0.0)

        # WP-PRIOR-1's "ALSO" finding / WP-HOUSE wiring (ledger C1): the
        # subclass evidence is a proper marginal over the register's own
        # per-template stage fractions `p_theta,k` (`_register`, one-hot
        # where a register carries none), not a hard mask --
        # `ln EV_k = logsumexp_theta(ln w_theta + ln p_theta,k)`.
        # `P_CLASS` (`ev_total` below) is unchanged by construction: since
        # every row of `ln_subclass_prob` sums (in linear space) to 1,
        # `logsumexp_k(EV_k) = logsumexp_theta(ln w_theta + ln
        # sum_k p_theta,k) = logsumexp_theta(ln w_theta)` exactly, the same
        # total a hard one-hot partition always gave.
        ln_subclass_prob = w["ln_subclass_prob"]            # (n_model, n_sub)
        with np.errstate(invalid="ignore"):
            ln_evidence64 = logsumexp(ln_w[:, None] + ln_subclass_prob, axis=0)
        ev_total = logsumexp(ln_evidence64)

        with np.errstate(invalid="ignore"):
            p_theta = np.exp(ln_w - ev_total)
        p_theta = np.where(np.isfinite(p_theta), p_theta, 0.0)

        k_keep = min(topk, n_model)
        order = np.argpartition(-ln_w, k_keep - 1)[:k_keep]
        order = order[np.argsort(-ln_w[order])]

        # section 9's Occam gap, NONDET2 brief section 2 (NONDET.md section
        # 8's ruling, restated): `logsumexp_theta(S1) - max_theta(S1)`, the
        # SAME vector (`s1_arr`) in both terms -- Gamma in NEITHER term, so
        # this is library-volume-and-non-detection spread alone, non-negative
        # by construction (logsumexp(v) >= max(v) for any real v). The old
        # Gamma-and-L_hat-free `cell_sum_mix` reading survives nowhere. A
        # source whose S1 is -inf at every template (good=True here, but
        # every class template vetoed by the prior -- row 12's fix) subtracts
        # -inf from -inf; the resulting NaN is overwritten below ("else:
        # occam_gap = nan") for a flagged source, and is itself the correct,
        # disclosed reading for a prior-vetoed one, not a warning-worthy one.
        with np.errstate(invalid="ignore"):
            occam_gap = float(logsumexp(s1_arr) - s1_arr.max())
            ln_ev_measured = float(logsumexp(s0_arr))
            ln_ev_nondet = float(logsumexp(s1_arr))

        # C7 rework (2026-10-20): LN_PRIOR_MASS is a pure lookup now, not a
        # per-source computation -- `w["ln_prior_mass"]` is this {region,
        # class}'s own (n_source,) array, read once in the parent off
        # bmstp.atlas's prior atlas (`_prior_mass_by_source`, called from
        # `build_region_class`), one value per source's own sky pixel.
        # NaN where the source's pixel is not one of the atlas's admitted
        # pixels (counted separately, `n_prior_mass_unmapped`).
        ln_prior_mass = float(w["ln_prior_mass"][i])

        good = not combined_flagged

        # TOPK_LAW: which design carried the larger ln_w_k for each of the
        # k_keep selected templates (section 2).
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

        # LOG10_FLUX_MEAN and LOG10_FLUX_COV's between term, from the 2m
        # (template, design) points weighted by p_theta * p_k (section 6.1):
        # the between term is the weighted mean/second moment of LOG10 flux
        # itself, summed over the two designs; TOPK_FLUX (linear mJy) is
        # the one thing `_topk_flux_kernel` still extracts per design.
        topk_flux_by_law = [None, None]
        mean_log10 = np.zeros(N_BANDS, dtype=np.float64)
        m2_log10 = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)
        for k in active_laws:
            wk = (p_theta * p_k[k]).astype(np.float64)
            out_topk_flux = np.full((1, topk, N_BANDS), np.nan, dtype=np.float32)
            _topk_flux_kernel(
                log10_flux_by_law[k][None, :, :], sorted_order[None, :],
                rank_position[None, :], np.array([good]), out_topk_flux)
            topk_flux_by_law[k] = out_topk_flux[0]
            weighted_log10 = log10_flux_by_law[k] * wk[:, None]
            mean_log10 += weighted_log10.sum(axis=0)
            m2_log10 += weighted_log10.T @ log10_flux_by_law[k]
        log10_flux_mean = mean_log10
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
            log10_flux_mean = np.full(N_BANDS, np.nan)
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
        topk_ln_prior_fit = np.full(topk, np.nan, dtype=np.float32)
        topk_ln_unseen = np.full(topk, np.nan, dtype=np.float32)
        topk_ln_gaia = np.full(topk, np.nan, dtype=np.float32)
        topk_ln_prior_ml = np.full(topk, np.nan, dtype=np.float32)
        topk_ln_gamma_ml = np.full(topk, np.nan, dtype=np.float32)
        topk_law = np.full(topk, -1, dtype=np.int8)
        topk_flux = np.full((topk, N_BANDS), np.nan, dtype=np.float32)
        if good:
            topk_model[:k_keep] = order.astype(np.int32)
            topk_law[:k_keep] = topk_law_sel
            # NONDET2 section 2: the separable record's three additive
            # fields are the LAW-MIXED S0, S1 - S0, S2 - S1 at the selected
            # templates -- not that template's winning design alone, unlike
            # every other TOPK_* field below (identity 1: their sum is
            # `ln_w[order]` exactly, to float32).
            topk_ln_prior_fit[:k_keep] = s0_arr[order].astype(np.float32)
            # a top-K template whose S1 (or S0) is -inf -- vetoed by the
            # prior at every cell (row 12's fix) -- subtracts -inf from
            # -inf; the NaN that results is the correct, disclosed reading
            # (identity 1 still holds: NaN + anything is NaN, consistent
            # with ln_w itself being -inf there), not a warning-worthy one.
            with np.errstate(invalid="ignore"):
                topk_ln_unseen[:k_keep] = (s1_arr[order] - s0_arr[order]).astype(np.float32)
                topk_ln_gaia[:k_keep] = (ln_w[order] - s1_arr[order]).astype(np.float32)
            for k in (0, 1):
                sel = topk_law_sel == k
                if not sel.any():
                    continue
                idx_sel = order[sel]
                fit_k = fits[k]
                topk_a_k[:k_keep][sel] = fit_k.a_hat_clamped[idx_sel]
                topk_log10_b[:k_keep][sel] = fit_k.log10_b_hat_clamped[idx_sel]
                topk_chi2[:k_keep][sel] = fit_k.chi2_min[idx_sel]
                topk_ln_l[:k_keep][sel] = ln_l_report[k][idx_sel].astype(np.float32)
                # TOPK_LN_PRIOR_ML: the prior alone (no non-detection, no
                # Gaia), the winning design's own number -- today's
                # TOPK_LN_PRIOR before NONDET (module docstring, section 2).
                topk_ln_prior_ml[:k_keep][sel] = ln_lambda0[k][idx_sel]
                topk_ln_gamma_ml[:k_keep][sel] = ln_gamma_ml[k][idx_sel].astype(np.float32)
                topk_flux[:k_keep][sel] = topk_flux_by_law[k][:k_keep][sel]
        else:
            occam_gap = float("nan")

        zero_ext_count = int(sum(int((fits[k].a_hat < 0.0).sum()) for k in active_laws)) if good else 0
        n_templates_checked = len(active_laws) * n_model if good else 0

        # NONDET2 brief identity 1: TOPK_LN_PRIOR_FIT + TOPK_LN_UNSEEN +
        # TOPK_LN_GAIA against ln_w[order] (float64, before any of the
        # three parts' own float32 cast), at every top-K entry this source
        # kept. Checked only where both sides are finite, or both are
        # exactly -inf (the two totals agree the template is vetoed); a
        # template whose window carries no mass at all makes S0 = S1 = -inf
        # together (row 12's fix), so TOPK_LN_UNSEEN is NaN there -- a
        # disclosed 0/0, not a defect -- and is excluded from the error
        # statistic, counted separately.
        id1_max_err, id1_sum_err, id1_n_checked, id1_n_undefined = 0.0, 0.0, 0, 0
        if good and k_keep > 0:
            true_sum = ln_w[order].astype(np.float64)
            stored_sum = (topk_ln_prior_fit[:k_keep].astype(np.float64)
                          + topk_ln_unseen[:k_keep].astype(np.float64)
                          + topk_ln_gaia[:k_keep].astype(np.float64))
            both_neginf = np.isneginf(true_sum) & np.isneginf(stored_sum)
            with np.errstate(invalid="ignore"):
                diff = np.abs(stored_sum - true_sum)
            checked = np.isfinite(diff) | both_neginf
            diff_checked = np.where(both_neginf, 0.0, diff)[checked]
            id1_n_checked = int(checked.sum())
            id1_n_undefined = int(k_keep - id1_n_checked)
            if id1_n_checked:
                id1_max_err = float(diff_checked.max())
                id1_sum_err = float(diff_checked.sum())

        row = dict(
            ln_evidence=ln_evidence64.astype(np.float32),
            log10_flux_mean=log10_flux_mean.astype(np.float32), log10_flux_cov=log10_flux_cov.astype(np.float32),
            topk_model=topk_model, topk_a_k=topk_a_k, topk_log10_b=topk_log10_b,
            topk_chi2=topk_chi2, topk_ln_l=topk_ln_l,
            topk_ln_prior_fit=topk_ln_prior_fit, topk_ln_unseen=topk_ln_unseen,
            topk_ln_gaia=topk_ln_gaia, topk_ln_prior_ml=topk_ln_prior_ml,
            topk_ln_gamma_ml=topk_ln_gamma_ml,
            topk_flux=topk_flux, topk_law=topk_law,
            occam_gap=np.float32(occam_gap),
            ln_evidence_measured=np.float32(ln_ev_measured if good else np.nan),
            ln_evidence_nondet=np.float32(ln_ev_nondet if good else np.nan),
            ln_prior_mass=np.float32(ln_prior_mass),
            a_k_post=np.float32(a_post), a_k_post_sig=np.float32(a_post_sig),
            p_dense=np.float32(p_dense),
            frac_clamped=frac_clamped, n_detected=np.int8(batch.n_detected),
        )
        return dict(i=i, failed=False, zero_ext_count=zero_ext_count,
                     n_templates_checked=n_templates_checked,
                     id1_max_err=id1_max_err, id1_sum_err=id1_sum_err,
                     id1_n_checked=id1_n_checked, id1_n_undefined=id1_n_undefined, **row)
    except Exception as exc:  # rule: one bad source must not lose the job
        row = _empty_row(topk, n_sub)
        return dict(i=i, failed=True, zero_ext_count=0, n_templates_checked=0,
                     id1_max_err=0.0, id1_sum_err=0.0, id1_n_checked=0, id1_n_undefined=0,
                     error=str(exc), **row)


# ---------------------------------------------------------------------------
# the parent side: load once, dispatch the pool, write parts, join
# ---------------------------------------------------------------------------

#: A02's coordinator revision (2026-10-22): NAME/index agreement alone
#: cannot tell a stale part of an EARLIER run from this run's own, when
#: both cover the same source range (the region/class/batch_size triple is
#: unchanged between runs -- the common case). RUNBOOKtp.sh exports
#: `SESNA_RUN_ID` once per invocation (date, time to the second, pid: the
#: same id names that invocation's own run record) and every part file
#: this process writes carries it, so a join only ever discovers parts of
#: its OWN run id -- a part from another run, or another concurrent one,
#: is a different filename entirely and is never a candidate. Invoked
#: outside the runbook (no variable set), a part keeps today's plain stem,
#: with one printed notice (not a silent fallback) that this invocation is
#: not run-scoped.
_RUN_ID_WARNED = False


def _run_id():
    global _RUN_ID_WARNED
    run_id = os.environ.get("SESNA_RUN_ID")
    if not run_id and not _RUN_ID_WARNED:
        print("fittp.sweep: SESNA_RUN_ID not set -- part files use the plain, "
              "un-run-scoped stem (today's behaviour outside RUNBOOKtp.sh); a part "
              "left by a different bare invocation sharing this path is indistinguishable "
              "from this one's own (A02 2026-10-22 revision)", flush=True)
        _RUN_ID_WARNED = True
    return run_id


def _part_path(path, bi):
    run_id = _run_id()
    if run_id:
        return "%s.%s.part%d" % (path, run_id, bi)
    return "%s.part%d" % (path, bi)


def _discover_parts(path):
    """Every `<path>[.run_id].partN` file already on disk for THIS process's
    own run id (`_run_id`), keyed by `N` (item 9): a restart's own re-run
    (`--batches`) writes only the batches it names, and `join_parts`
    re-discovers every part present -- from this run or an earlier call
    within it -- rather than trusting an in-memory list a crash would have
    lost. Returns a dict `{N: part_path}`, not a list: `join_parts` (A02)
    needs the index set itself, to tell a genuinely missing batch from an
    extra, stale one of the same count. Scoped to `_run_id()`'s own value
    (A02 revision): a part written under a different run id -- or no run
    id at all -- is a different filename and is never discovered here,
    however many of its rows' NAMEs happen to match this run's own.
    """
    run_id = _run_id()
    if run_id:
        glob_pattern = glob.escape(path) + "." + glob.escape(run_id) + ".part*"
        pattern = re.compile(re.escape(path) + r"\." + re.escape(run_id) + r"\.part(\d+)$")
    else:
        glob_pattern = glob.escape(path) + ".part*"
        pattern = re.compile(re.escape(path) + r"\.part(\d+)$")
    found = {}
    for p in glob.glob(glob_pattern):
        m = pattern.match(p)
        if m:
            found[int(m.group(1))] = p
    return found


def _new_batch_arrays(m, n_sub, topk):
    """P02 (round 6 review): one batch's own arrays, preallocated BEFORE
    the imap loop starts, so `_fill_batch_row` can write each source's own
    task result straight into its row as it arrives -- instead of the
    loop first gathering every one of the batch's `m` per-source dicts
    into a list (a second, duplicate parent-side payload) and only then
    copying each field out of that list into these same arrays. A batch-
    sized set of arrays, never a region-sized one (rule 10b)."""
    return dict(
        ln_evidence=np.empty((m, n_sub), dtype=np.float32),
        log10_flux_mean=np.empty((m, N_BANDS), dtype=np.float32),
        log10_flux_cov=np.empty((m, N_BANDS, N_BANDS), dtype=np.float32),
        topk_model=np.empty((m, topk), dtype=np.int32),
        topk_a_k=np.empty((m, topk), dtype=np.float32),
        topk_log10_b=np.empty((m, topk), dtype=np.float32),
        topk_chi2=np.empty((m, topk), dtype=np.float32),
        topk_ln_l=np.empty((m, topk), dtype=np.float32),
        topk_ln_prior_fit=np.empty((m, topk), dtype=np.float32),
        topk_ln_unseen=np.empty((m, topk), dtype=np.float32),
        topk_ln_gaia=np.empty((m, topk), dtype=np.float32),
        topk_ln_prior_ml=np.empty((m, topk), dtype=np.float32),
        topk_ln_gamma_ml=np.empty((m, topk), dtype=np.float32),
        topk_flux=np.empty((m, topk, N_BANDS), dtype=np.float32),
        topk_law=np.empty((m, topk), dtype=np.int8),
        occam_gap=np.empty(m, dtype=np.float32),
        ln_evidence_measured=np.empty(m, dtype=np.float32),
        ln_evidence_nondet=np.empty(m, dtype=np.float32),
        ln_prior_mass=np.empty(m, dtype=np.float32),
        a_k_post=np.empty(m, dtype=np.float32),
        a_k_post_sig=np.empty(m, dtype=np.float32),
        p_dense=np.empty(m, dtype=np.float32),
        frac_clamped=np.empty(m, dtype=np.float32),
        n_detected=np.empty(m, dtype=np.int8),
        failed=np.zeros(m, dtype=bool),
        zero_ext_count=0, n_templates_checked=0,
        id1_max_err=0.0, id1_sum_err=0.0, id1_n_checked=0, id1_n_undefined=0,
    )


def _fill_batch_row(arrays, r, k):
    """Writes one source's own `_source_task` result `r` into row `k` of
    `arrays` (`_new_batch_arrays`'s own preallocated dict), in source
    order, as `build_region_class`'s own `imap` loop consumes it -- the
    same fields `_assemble_batch` has always copied, one source at a time
    instead of all `m` after a list of every result was gathered first
    (P02, round 6 review). The four running totals (`zero_ext_count`,
    `n_templates_checked`, `id1_sum_err`, `id1_n_checked`, `id1_n_undefined`)
    accumulate in the same source order as before (`imap` yields in input
    order), so no reduction-order tolerance is needed; `id1_max_err` is a
    commutative max."""
    arrays["ln_evidence"][k] = r["ln_evidence"]
    arrays["log10_flux_mean"][k] = r["log10_flux_mean"]
    arrays["log10_flux_cov"][k] = r["log10_flux_cov"]
    arrays["topk_model"][k] = r["topk_model"]
    arrays["topk_a_k"][k] = r["topk_a_k"]
    arrays["topk_log10_b"][k] = r["topk_log10_b"]
    arrays["topk_chi2"][k] = r["topk_chi2"]
    arrays["topk_ln_l"][k] = r["topk_ln_l"]
    arrays["topk_ln_prior_fit"][k] = r["topk_ln_prior_fit"]
    arrays["topk_ln_unseen"][k] = r["topk_ln_unseen"]
    arrays["topk_ln_gaia"][k] = r["topk_ln_gaia"]
    arrays["topk_ln_prior_ml"][k] = r["topk_ln_prior_ml"]
    arrays["topk_ln_gamma_ml"][k] = r["topk_ln_gamma_ml"]
    arrays["topk_flux"][k] = r["topk_flux"]
    arrays["topk_law"][k] = r["topk_law"]
    arrays["occam_gap"][k] = r["occam_gap"]
    arrays["ln_evidence_measured"][k] = r["ln_evidence_measured"]
    arrays["ln_evidence_nondet"][k] = r["ln_evidence_nondet"]
    arrays["ln_prior_mass"][k] = r["ln_prior_mass"]
    arrays["a_k_post"][k] = r["a_k_post"]
    arrays["a_k_post_sig"][k] = r["a_k_post_sig"]
    arrays["p_dense"][k] = r["p_dense"]
    arrays["frac_clamped"][k] = r["frac_clamped"]
    arrays["n_detected"][k] = r["n_detected"]
    arrays["failed"][k] = r["failed"]
    arrays["zero_ext_count"] += r["zero_ext_count"]
    arrays["n_templates_checked"] += r["n_templates_checked"]
    arrays["id1_max_err"] = max(arrays["id1_max_err"], r["id1_max_err"])
    arrays["id1_sum_err"] += r["id1_sum_err"]
    arrays["id1_n_checked"] += r["id1_n_checked"]
    arrays["id1_n_undefined"] += r["id1_n_undefined"]


def _assemble_batch(arrays, name_slice):
    """Folds one batch's own arrays -- `_new_batch_arrays`'s preallocated
    dict, filled in place by `_fill_batch_row` as `build_region_class`'s
    own `imap` loop consumed it -- into the one dict `_write_part` writes,
    plus the batch's own `NAME` slice."""
    return dict(name=name_slice, **arrays)


def _write_part(part_path, batch, cls):
    """Written as `<part>.tmp` and renamed into place once closed, so a part
    file under its own name is always complete: several slurm part jobs of
    one {region, class} run at once, and the join (`_discover_parts`) must
    never pick up a part another job is still writing."""
    stem = _STEM(cls)
    tmp_path = part_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        for key in _PART_KEYS:
            build_module.write_dataset(f, key, batch[_FIELD_OF_KEY[key]], *REGISTRY[(stem, key)])
        # "FAILED" is a part-file-only column (module docstring), and a part
        # file is transient: `join_parts` folds it into FAILED_ROWS and then
        # removes the file, so no product a consumer opens ever carries this
        # column. It is the one dataset in the fitter with no registry entry
        # and no unit or reading, deliberately: rule 5 describes a product.
        f.create_dataset("FAILED", data=batch["failed"])
        f.attrs["ZERO_EXT_COUNT"] = batch["zero_ext_count"]
        f.attrs["N_TEMPLATES_CHECKED"] = batch["n_templates_checked"]
        # NONDET2 brief identity 1, report-only (not a product column: rule
        # 5 describes a product, this is the build's own check number).
        f.attrs["ID1_MAX_ERR"] = batch["id1_max_err"]
        f.attrs["ID1_SUM_ERR"] = batch["id1_sum_err"]
        f.attrs["ID1_N_CHECKED"] = batch["id1_n_checked"]
        f.attrs["ID1_N_UNDEFINED"] = batch["id1_n_undefined"]
    os.replace(tmp_path, part_path)


#: Section 3's own retry on `ctx.Pool`'s creation: the Aquila failure
#: (MAPARRAYS "Why") landed on HBM nodes still reclaiming a job just
#: vacated, where `fork` refuses outright with `ENOMEM` rather than
#: waiting a moment for the pages back. Four attempts total, 20 s apart
#: (three gaps) -- "three attempts" the three retries after the first.
POOL_RETRY_ATTEMPTS = 4
POOL_RETRY_SECONDS = 20.0


def _meminfo_free_available():
    """`(MemFree, MemAvailable)` in kB off `/proc/meminfo` -- the node's
    own two numbers the Stampede3 diagnostic read (MAPARRAYS "Why").
    `(None, None)` where the file does not exist (this laptop's own
    macOS has no `/proc`), so the retry's print line still runs
    everywhere, naming what it could not read."""
    free = avail = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemFree:"):
                    free = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1])
    except OSError:
        pass
    return free, avail


def _make_pool(ctx, n_workers):
    """`ctx.Pool(n_workers, initializer=_init_worker)`, retried on a
    transient refusal (section 3): up to `POOL_RETRY_ATTEMPTS` tries,
    `POOL_RETRY_SECONDS` apart, surviving only an `OSError` whose `errno`
    is `ENOMEM` -- any other exception raises straight through,
    unretried. The ordinary, never-fails case stays silent
    (CODING_RULES_BMSTP.md rule 17, "nothing else prints"): a line
    prints only once a failure has happened, one per failed attempt plus
    one confirming the eventual success, each with the node's own
    `MemFree`/`MemAvailable` (`_meminfo_free_available`); the last
    attempt's own `ENOMEM` raises instead of sleeping again, those same
    two numbers in the message."""
    for attempt in range(1, POOL_RETRY_ATTEMPTS + 1):
        try:
            pool = ctx.Pool(n_workers, initializer=_init_worker)
        except OSError as exc:
            if exc.errno != errno.ENOMEM:
                raise
            free, avail = _meminfo_free_available()
            print("fittp.sweep._make_pool: attempt %d/%d failed with ENOMEM -- "
                  "MemFree=%s kB MemAvailable=%s kB"
                  % (attempt, POOL_RETRY_ATTEMPTS, free, avail), flush=True)
            if attempt == POOL_RETRY_ATTEMPTS:
                raise OSError(
                    errno.ENOMEM,
                    "fittp.sweep._make_pool: %d attempts at ctx.Pool(%d) all failed "
                    "with ENOMEM -- MemFree=%s kB MemAvailable=%s kB"
                    % (POOL_RETRY_ATTEMPTS, n_workers, free, avail))
            time.sleep(POOL_RETRY_SECONDS)
            continue
        if attempt > 1:
            free, avail = _meminfo_free_available()
            print("fittp.sweep._make_pool: attempt %d/%d succeeded -- "
                  "MemFree=%s kB MemAvailable=%s kB"
                  % (attempt, POOL_RETRY_ATTEMPTS, free, avail), flush=True)
        return pool


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
    template_log, n_sub, subclass_prob = _register(config, cls)
    n_model = template_log.shape[0]
    # computed once, in the parent, never per source (rule 8): `ln
    # p_theta,k`, `-inf` where a template carries no mass in that
    # subclass -- `0.0` entries are exact zeros from `_register`'s own
    # construction (a one-hot row or a register's own stage fraction),
    # not a rounding artefact, so the divide-by-zero warning is expected
    # and silenced, not masked.
    with np.errstate(divide="ignore"):
        ln_subclass_prob = np.log(subclass_prob).astype(np.float64)
    gaia_term = GaiaTerm(config, region)
    # gaia.GaiaTerm.warm's own docstring (item 4): its per-class register
    # and field-star marginal are lazily cached on first `ln_gamma` call by
    # design; left lazy, each of this call's own forked workers would
    # independently open that product on its own first task. Warmed here,
    # in the parent, before the pool below forks.
    gaia_term.warm(cls.lower())
    topk = config.topk

    lib_path = config_module.product_path(config, "fittp", "check", "library-resolution", "survey")
    sigma_lib_l = likelihood.sigma_lib_by_class(lib_path)[cls]

    n_source = catalog["flux"].shape[0]
    # C7 rework: the atlas lookup covers the FULL region, row-aligned to
    # every other per-source array here, regardless of `--limit` (a
    # timing/acceptance device on the sweep loop below, never on what
    # this reads) -- captured before `n_source` is clipped.
    ln_prior_mass_by_source, prior_mass_matched = _prior_mass_by_source(config, region, cls, n_source)
    if limit is not None:
        n_source = min(n_source, limit)
    # the unmapped count `st.done` reports is over THIS run's own swept
    # rows only (0..n_source-1 post-`--limit`), not the whole region.
    n_prior_mass_unmapped = int(np.count_nonzero(~prior_mass_matched[:n_source]))

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

    batch_size = config.batch_size
    batch_bounds = [(s, min(s + batch_size, n_source)) for s in range(0, n_source, batch_size)]
    selected = set(batches) if batches is not None else None

    path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    _set_worker_state(config=config, cls=cls, reader=reader, gaia_term=gaia_term,
                       template_log=template_log, ln_subclass_prob=ln_subclass_prob, n_sub=n_sub,
                       ln_prior_mass=ln_prior_mass_by_source,
                       topk=topk, n_model=n_model,
                       sigma_lib_l=sigma_lib_l,
                       flux=catalog["flux"], sigma=catalog["sigma"], origin=catalog["origin"],
                       curve_tables=catalog["curve_tables"], source_row8=catalog["source_row8"],
                       band_a=catalog["band_a"], band_c=catalog["band_c"], band_n=catalog["band_n"],
                       max_n=catalog["max_n"])

    # numba JIT-compiles once per PROCESS, so a worker meeting a kernel cold
    # pays its compile memory privately -- measured at ~82 MB per kernel, which
    # 100 workers would carry 100 times over. Warming here, in the parent,
    # leaves the compiled code in pages every worker inherits copy-on-write:
    # measured private growth per worker falls from ~82 MB a kernel to ~1.6 MB.
    # The warm is one real source through the real task, so it compiles exactly
    # the signatures the workers go on to call (`_cell_sum` alone takes 23
    # arguments; an explicit signature list would be a second thing to keep
    # right). `_source_task` catches its own failures, so this cannot raise.
    numba.set_num_threads(1)
    _source_task(batch_bounds[0][0])

    # Forked below every load above, and below the warm -- the threading layer
    # pinned at import is fork-safe by numba's own contract.
    ctx = mp.get_context("fork")
    with _make_pool(ctx, n_workers) as pool:
        for bi, (bstart, bstop) in enumerate(batch_bounds):
            if selected is not None and bi not in selected:
                continue
            m = bstop - bstart
            arrays = _new_batch_arrays(m, n_sub, topk)
            for r in pool.imap(_source_task, range(bstart, bstop), chunksize=_IMAP_CHUNKSIZE):
                _fill_batch_row(arrays, r, r["i"] - bstart)
            batch = _assemble_batch(arrays, catalog["name"][bstart:bstop])
            _write_part(_part_path(path, bi), batch, cls)
            st.tick(bi + 1, len(batch_bounds), "batches")

    return _summary(config, region, cls, n_source, n_model,
                    n_prior_mass_unmapped=n_prior_mass_unmapped)


def _summary(config, region, cls, n_source, n_model, n_prior_mass_unmapped=None):
    """The numbers `join_parts` and the report need for one {region,
    class}, shared by a sweep and by a join-only call (`build`'s
    `join_only`), which sweeps nothing and so has no sweep to take them
    from. `n_prior_mass_unmapped` (C7 rework) is `None` for the
    join-only path -- it never computed `LN_PRIOR_MASS`'s own source-to-
    pixel mapping, so there is nothing to report."""
    path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
    density_file = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    lib, granule = prior_reader._LIB[cls]
    weights_file = config_module.product_path(
        config, "bmstp", "weights", lib, granule, region=(region if granule == "region" else None))
    batch_size = config.batch_size
    return dict(
        path=path, region=region, n_source=n_source, n_model=n_model,
        n_batches=(n_source + batch_size - 1) // batch_size,
        subclasses=definitions.SUBCLASSES_OF[cls], library=definitions.CLASS_REGISTER[cls],
        density_file=density_file, weights_file=weights_file,
        n_prior_mass_unmapped=n_prior_mass_unmapped,
    )


def join_parts(summary, topk, config):
    """Joins one {region, class}'s part files into the final P7 product,
    one part's rows at a time, dataset by dataset (rule 10b: never a
    region-sized array); removes the part files once the join has landed.
    Item 9: the part list comes from `_discover_parts`, not an in-memory
    record of what THIS call wrote, so a join after a restart (`--batches`
    naming just the missing indices, same `SESNA_RUN_ID`) picks up parts
    an earlier, crashed call of the SAME run already left on disk.

    A02 (review ledger, FINDINGS.md; coordinator's same-day revision): a
    correct join requires more than a part COUNT, and a NAME/index check
    alone cannot tell a stale part of an EARLIER run from this run's own
    when both cover the same source range. Four checks, each naming what
    is wrong rather than joining a silently incomplete or incompatible
    product:
      0. (`_discover_parts`/`_part_path`, above) only parts carrying THIS
         process's own `SESNA_RUN_ID` (or, outside the runbook, the plain
         stem) are even discovered -- a part from a different run id is a
         different filename, never a candidate, however many of its rows'
         NAMEs happen to match;
      1. the discovered indices are exactly `{0 .. n_batches-1}` -- any
         missing or extra index (e.g. a stale part left by a since-changed
         `[fit] batch_size`) is named and raises;
      2. each part's own `NAME` column is bit-identical to the catalogue's
         own slice for that batch's bounds (`_load_region_catalog`'s same
         source order, re-opened here by its own memory-mapped `NAME`
         dataset rather than the full catalogue) -- a part from the wrong
         region, the wrong batch position, or a stale run under a
         different source order is named and raises, rather than being
         silently spliced in because its row count alone happened to match;
      3. the joined row total equals `n_source` exactly.
    The join itself writes `<path>.tmp` and `os.replace`s it into place
    (the part writes already do this, `_write_part`), so a join that
    raises or is killed partway leaves the PRIOR joined file (if any)
    untouched and every part file still on disk for the next attempt --
    the part files are removed only after the replace has landed.

    Also folds every part's own `ZERO_EXT_COUNT`/`N_TEMPLATES_CHECKED`
    (for the report's `zero_ext_frac`) and `FAILED` column (item 7's
    `FAILED_ROWS`, the count reported as `n_failed`) across every part,
    present or reproduced.
    """
    path = summary["path"]
    region = summary["region"]
    n_source = summary["n_source"]
    n_batches = summary["n_batches"]
    found = _discover_parts(path)
    expected = set(range(n_batches))
    missing = sorted(expected - found.keys())
    extra = sorted(found.keys() - expected)
    if missing or extra:
        bits = []
        if missing:
            bits.append("missing index/indices %s" % missing)
        if extra:
            bits.append("extra index/indices %s" % extra)
        raise RuntimeError(
            "fittp.sweep.join_parts [%s]: %s of %d expected batch part files for %s -- "
            "rerun with --batches naming the missing indices before joining"
            % (summary.get("cls"), " and ".join(bits), n_batches, path))
    part_paths = [found[k] for k in range(n_batches)]

    batch_size = config.batch_size
    batch_bounds = [(s, min(s + batch_size, n_source)) for s in range(0, n_source, batch_size)]
    cat_path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(cat_path, "r") as cf:
        catalog_name = prior_reader.mmap_dataset(cf, "NAME")

    stem = _STEM(summary.get("cls"))
    zero_ext_count = 0
    n_templates_checked = 0
    id1_max_err, id1_sum_err, id1_n_checked, id1_n_undefined = 0.0, 0.0, 0, 0
    failed_chunks = []
    tmp_path = path + ".tmp"
    with h5py.File(tmp_path, "w") as out:
        for key in _PART_KEYS:
            with h5py.File(part_paths[0], "r") as pf0:
                shape = (n_source,) + pf0[key].shape[1:]
                dtype = pf0[key].dtype
            build_module.write_dataset(out, key, None, *REGISTRY[(stem, key)], shape=shape, dtype=dtype)
        offset = 0
        for bi, part_path in enumerate(part_paths):
            bstart, bstop = batch_bounds[bi]
            with h5py.File(part_path, "r") as pf:
                m = pf["NAME"].shape[0]
                if m != bstop - bstart:
                    raise RuntimeError(
                        "fittp.sweep.join_parts [%s]: part %d (%s) has %d rows, expected %d "
                        "for the catalogue slice [%d:%d) of %s -- rerun that batch before joining"
                        % (summary.get("cls"), bi, part_path, m, bstop - bstart, bstart, bstop, region))
                part_name = pf["NAME"][:]
                expected_name = np.asarray(catalog_name[bstart:bstop])
                if not np.array_equal(part_name, expected_name):
                    raise RuntimeError(
                        "fittp.sweep.join_parts [%s]: part %d (%s) NAME does not match %s's own "
                        "catalogue slice [%d:%d) -- stale or mismatched part file, rerun that "
                        "batch before joining"
                        % (summary.get("cls"), bi, part_path, region, bstart, bstop))
                for key in _PART_KEYS:
                    out[key][offset:offset + m] = pf[key][:]
                failed_here = np.nonzero(pf["FAILED"][:])[0]
                failed_chunks.append(failed_here + offset)
                zero_ext_count += int(pf.attrs["ZERO_EXT_COUNT"])
                n_templates_checked += int(pf.attrs["N_TEMPLATES_CHECKED"])
                id1_max_err = max(id1_max_err, float(pf.attrs["ID1_MAX_ERR"]))
                id1_sum_err += float(pf.attrs["ID1_SUM_ERR"])
                id1_n_checked += int(pf.attrs["ID1_N_CHECKED"])
                id1_n_undefined += int(pf.attrs["ID1_N_UNDEFINED"])
            offset += m
        if offset != n_source:
            raise RuntimeError(
                "fittp.sweep.join_parts [%s]: joined %d rows, expected n_source=%d for %s"
                % (summary.get("cls"), offset, n_source, path))
        failed_rows = (np.concatenate(failed_chunks) if failed_chunks
                        else np.array([], dtype=np.int64)).astype(np.int64)
        build_module.write_dataset(out, "FAILED_ROWS", failed_rows, *REGISTRY[(stem, "FAILED_ROWS")])
        out.attrs["GRANULE"] = "source"
        out.attrs["CLASS"] = summary.get("cls")
        out.attrs["SUBCLASSES"] = np.array(summary["subclasses"], dtype="S8")
        out.attrs["LIBRARY"] = summary["library"]
        out.attrs["N_MODEL"] = summary["n_model"]
        out.attrs["K"] = topk
        out.attrs["WEIGHTS_FILE"] = summary["weights_file"]
        out.attrs["DENSITY_FILE"] = summary["density_file"]
    os.replace(tmp_path, path)
    for part_path in part_paths:
        os.remove(part_path)
    zero_ext_frac = zero_ext_count / n_templates_checked if n_templates_checked else float("nan")
    id1_mean_err = id1_sum_err / id1_n_checked if id1_n_checked else float("nan")
    return dict(zero_ext_frac=zero_ext_frac, n_failed=int(failed_rows.size),
                id1_max_err=id1_max_err, id1_mean_err=id1_mean_err,
                id1_n_checked=id1_n_checked, id1_n_undefined=id1_n_undefined)


def build(config, regions=None, classes=None, limit=None, n_workers=1, batches=None,
          join=True, join_only=False):
    """Writes `fittp/fit/<CLS>_fit_source__R.hdf5` for every {region,
    class} pair (default all thirty regions, all seven classes; section
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

    A class split across several slurm jobs runs each part job with
    `join=False` (writes its batches' part files and stops) and then one
    `join_only=True` call once every part job has ended (sweeps nothing,
    joins every part file on disk, and fails naming the missing batches
    if any part is absent) -- so the part jobs can run at the same time
    without two of them joining at once.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    class_codes = classes if classes is not None else list(CLASSES)
    t0 = time.time()
    for region in region_names:
        catalog = None if join_only else _load_region_catalog(config, region)
        for cls in class_codes:
            with progress.Stage("fittp.sweep.%s" % cls, region) as st:
                if join_only:
                    n_source = _n_sources(config, region)
                    if limit is not None:
                        n_source = min(n_source, limit)
                    summary = _summary(config, region, cls, n_source, _register(config, cls)[0].shape[0])
                else:
                    reader = prior_reader.load(config, region, cls)
                    summary = build_region_class(config, region, cls, st, reader,
                                                  catalog, n_workers=n_workers, limit=limit,
                                                  batches=batches)
                summary["cls"] = cls
                if not join:
                    st.done(None, n=summary["n_source"], n_batches=summary["n_batches"],
                            parts_written=(len(batches) if batches is not None else summary["n_batches"]),
                            joined="no (--no-join)")
                    continue
                joined = join_parts(summary, config.topk, config)
                with h5py.File(summary["path"], "r") as f:
                    occam = np.asarray(f["OCCAM_GAP"][:])
                occam_finite = occam[np.isfinite(occam)]
                occam_median = float(np.median(occam_finite)) if occam_finite.size else float("nan")
                st.done(summary["path"], n=summary["n_source"], n_model=summary["n_model"],
                        zero_ext_frac=joined["zero_ext_frac"], occam_gap_median=occam_median,
                        n_batches=summary["n_batches"], n_failed=joined["n_failed"],
                        id1_max_err=joined["id1_max_err"], id1_mean_err=joined["id1_mean_err"],
                        id1_n_checked=joined["id1_n_checked"], id1_n_undefined=joined["id1_n_undefined"],
                        n_prior_mass_unmapped=summary["n_prior_mass_unmapped"])
    for cls in class_codes:
        progress.summary("fittp.sweep.%s" % cls, len(region_names), t0)


if __name__ == "__main__":
    import argparse
    import dataclasses

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="default: all seven -- every {region, class} pair is swept")
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
    parser.add_argument("--no-join", action="store_true",
                         help="write this call's part files and stop, without joining: one of "
                              "several part jobs of a split class running at the same time")
    parser.add_argument("--join-only", action="store_true",
                         help="sweep nothing; join every part file on disk into the class's "
                              "product (run once, after every part job of a split class has ended)")
    #: applied to config.n_jobs only; the sweep's own pool is --workers / fittp_workers, unchanged
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    if args.no_join and args.join_only:
        parser.error("--no-join and --join-only exclude each other")
    cfg = config_module.load(args.config)
    if args.n_jobs is not None:
        cfg = dataclasses.replace(cfg, n_jobs=args.n_jobs)
        print("%s: n_jobs=%d from --n-jobs (config says %d)"
              % ("sesnaimpute.fittp.sweep", args.n_jobs, config_module.load(args.config).n_jobs))
    n_workers = args.workers if args.workers is not None else cfg.fittp_workers
    build(cfg, regions=args.regions, classes=args.classes, limit=args.limit,
          n_workers=n_workers, batches=args.batches,
          join=not args.no_join, join_only=args.join_only)
