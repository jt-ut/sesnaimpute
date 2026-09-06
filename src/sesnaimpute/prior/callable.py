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

import os
import time

import h5py
import numba
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute.prior import gal as gal_module
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
    scales with the caller's own batch."""
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

        a_col = p.table["A_COL_K"][rows_f]
        sigma_col = p.table["A_COL_SIG_K"][rows_f]
        sl_rows = p.table["HPX256_ROW"][rows_f]
        map_class = p.table["A_COL_PROVENANCE"][rows_f]
        zp_sigma_k = p._zp_sigma_k[rows_f]
        p_a = _marginal_exact_chunked(p.yso_shape, a_f, sl_rows, a_col, sigma_col, map_class,
                                      zp_sigma_k=zp_sigma_k)
        p_a = np.where(a_f > 0.0, p_a, 0.0)

        mean_b = p.table["RIDGE_INTERCEPT"][rows_f] + p.table["RIDGE_SLOPE"][rows_f] * a_f
        width = p.table["RIDGE_WIDTH"][rows_f]
        z_score = (b_f - mean_b) / width
        p_b = _INV_SQRT_2PI / width * np.exp(-0.5 * z_score ** 2)
        return (p_a * p_b).reshape(shp)

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
        mi_f = model_index.ravel()
        valid = a_f > 0.0

        a_col = p.table["A_COL_K"][rows_f]
        sigma_col = p.table["A_COL_SIG_K"][rows_f]
        sl_rows = p.table["HPX256_ROW"][rows_f]
        map_class = p.table["A_COL_PROVENANCE"][rows_f]
        zp_sigma_k = p._zp_sigma_k[rows_f]
        p_a = _marginal_exact_chunked(p.yso_shape, a_f, sl_rows, a_col, sigma_col, map_class,
                                      zp_sigma_k=zp_sigma_k)
        p_a = np.where(valid, p_a, 0.0)

        log10_sigma = b_f + np.log10(p.h2s_fref[mi_f])
        z_score = (log10_sigma - p.h2s_logsig_mean) / p.h2s_logsig_std
        p_sigma = _INV_SQRT_2PI / p.h2s_logsig_std * np.exp(-0.5 * z_score ** 2)
        return (p_a * p_sigma).reshape(shp)

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
        eps_batch = p._prep_h2s_eps[local]
        eps_val = _interp_eps_2d(eps_batch, p.h2s_x_ladder, p.h2s_log10_sigma_grid,
                                 x_query, log10_sigma)
        return eps_val.reshape(shp)


#: `Z`'s own fixed quadrature grid (owner ruling, 2026-09-06): `Z` is
#: the normaliser of the density the fitter actually reads, so it is
#: computed here, at `prepare` time, by the SAME `shape`/`selection`
#: read `log_density` composes -- not a separately-stored upstream
#: integral. One rectangular grid per source: `_Z_QUAD_NA` log-spaced
#: points in `a` from a floor of `_Z_QUAD_A_FLOOR_FRAC . A_s` to
#: `_Z_QUAD_SIGMAS` kernel sigmas past the shifted mean (the same
#: generous bound `_extinction_grid` uses for the identity check, read
#: off the class's OWN kernel -- `ClassShape.kern` for STAR/AGB/PAHC,
#: the shared `YsoShape.kernel` for GAL/YSO/H2S), `_Z_QUAD_NB` points in
#: `log10 B` over the class's own tabulated/nominal range. Trapezoid in
#: both axes.
_Z_QUAD_NA = 48
_Z_QUAD_NB = 64
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


def _quad_grid_lin(lo, hi, num):
    """`(n, num)`: linearly-spaced grid per source between per-source
    `lo` and `hi` (both `(n,)`)."""
    t = np.linspace(0.0, 1.0, num)
    return lo[:, None] + t[None, :] * (hi - lo)[:, None]


def _z_by_quadrature(prior, cls, rows):
    """`(n,)`: `Z[s] = trapz_(log10 B) trapz_a shape(a, log10 B) .
    selection(a, log10 B)` on the `_Z_QUAD_NA` x `_Z_QUAD_NB` grid
    (module docstring), one rectangle per source. Reads `prior`'s own
    `shape`/`selection` objects -- the exact arithmetic `log_density`
    composes -- so a source whose selection is genuinely zero
    everywhere (no object of this class could be catalogued at its own
    limits) returns `Z = 0` here, the correct statement that the class
    is impossible on this sightline, not a defect (owner ruling,
    2026-09-06)."""
    obj = prior._classes[cls]
    n = rows.shape[0]
    a_col = prior.table["A_COL_K"][rows]
    sigma_col = prior.table["A_COL_SIG_K"][rows]
    map_class_code = prior.table["A_COL_PROVENANCE"][rows]
    a_floor = _Z_QUAD_A_FLOOR_FRAC * a_col

    if cls in FAMILY_CLASSES:
        shape_obj = prior.shapes[cls]
        map_class_str = np.where(
            map_class_code == star_shapes._PLANCK_PROVENANCE_CODE, "planck", "herschel")
        _, mu, sigma = shape_obj.kern.mixture(a_col, sigma_col, map_class_str)
        mu_tail = np.maximum(mu[:, 0] + _Z_QUAD_SIGMAS * sigma[:, 0],
                             mu[:, 1] + _Z_QUAD_SIGMAS * sigma[:, 1])
        a_hi = a_col * 10.0 ** mu_tail
        a_grid = _quad_grid_log(a_floor, a_hi, _Z_QUAD_NA)
        b_cell = float(np.mean(np.diff(shape_obj.b_edges)))
        b_row = np.linspace(shape_obj.b_edges[0] - 6 * b_cell,
                            shape_obj.b_edges[-1] + 6 * b_cell, _Z_QUAD_NB)
        b_grid = np.broadcast_to(b_row, (n, _Z_QUAD_NB))
        mi_grid = None
    elif cls in ("gal", "h2s"):
        mu, sigma = prior.yso_shape.kernel.params(a_col, sigma_col, map_class_code)
        a_hi = a_col * 10.0 ** (mu + _Z_QUAD_SIGMAS * sigma)
        a_grid = _quad_grid_log(a_floor, a_hi, _Z_QUAD_NA)
        if cls == "gal":
            s_grid, fref0 = prior.gal_log10_s_grid, prior.gal_fref[0]
        else:
            s_grid, fref0 = prior.h2s_log10_sigma_grid, prior.h2s_fref[0]
        b_row = np.linspace(s_grid[0], s_grid[-1], _Z_QUAD_NB) - np.log10(fref0)
        b_grid = np.broadcast_to(b_row, (n, _Z_QUAD_NB))
        mi_grid = np.zeros((n, _Z_QUAD_NA * _Z_QUAD_NB), dtype=np.intp)
    else:  # yso
        mu, sigma = prior.yso_shape.kernel.params(a_col, sigma_col, map_class_code)
        a_hi = a_col * 10.0 ** (mu + _Z_QUAD_SIGMAS * sigma)
        a_grid = _quad_grid_log(a_floor, a_hi, _Z_QUAD_NA)
        # the ridge's own mean MOVES with `a` (`RIDGE_SLOPE`); a `b_grid`
        # fixed at the nominal `a_col` misses essentially all the density
        # at the far end of a wide `a_grid` (the bug this sheared grid
        # fixes -- caught by a 5 source smoke test reading Z 5-10x too
        # small before this fix, `_integrate_yso`'s own sheared identity
        # grid is the model). `b_grid` is 3-D here, `(n, NA, NB)`, not
        # `(n, NB)`: one sheared window PER `a_grid` point.
        ridge_width = prior.table["RIDGE_WIDTH"][rows]
        mean_b_grid = (prior.table["RIDGE_INTERCEPT"][rows][:, None]
                       + prior.table["RIDGE_SLOPE"][rows][:, None] * a_grid)
        t = np.linspace(-1.0, 1.0, _Z_QUAD_NB) * _Z_QUAD_SIGMAS
        b_grid_3d = mean_b_grid[:, :, None] + t[None, None, :] * ridge_width[:, None, None]
        a_full = np.repeat(a_grid, _Z_QUAD_NB, axis=1)
        b_full = b_grid_3d.reshape(n, _Z_QUAD_NA * _Z_QUAD_NB)
        rows2d = np.broadcast_to(rows[:, None], a_full.shape)
        shape_val = obj.shape(rows2d, a_full, b_full, model_index=None)
        sel_val = obj.selection(rows2d, a_full, b_full, model_index=None)
        dens = (shape_val * sel_val).reshape(n, _Z_QUAD_NA, _Z_QUAD_NB)
        inner = np.trapz(dens, x=b_grid_3d, axis=2)
        return np.trapz(inner, x=a_grid, axis=1)

    a_full = np.repeat(a_grid, _Z_QUAD_NB, axis=1)
    b_full = np.tile(b_grid, (1, _Z_QUAD_NA))
    rows2d = np.broadcast_to(rows[:, None], a_full.shape)

    shape_val = obj.shape(rows2d, a_full, b_full, model_index=mi_grid)
    sel_val = obj.selection(rows2d, a_full, b_full, model_index=mi_grid)
    dens = (shape_val * sel_val).reshape(n, _Z_QUAD_NA, _Z_QUAD_NB)

    b_grid_3d = np.broadcast_to(b_grid[:, None, :], dens.shape)
    inner = np.trapz(dens, x=b_grid_3d, axis=2)
    return np.trapz(inner, x=a_grid, axis=1)


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

        # -- H2S: the region's Sigma lognormal and this region's
        # per-source selection curve (`prior.h2s`'s own `EPS[n, n_x,
        # n_sigma]`).
        self._h2s_selection_path = None
        self.h2s_x_ladder = self.h2s_log10_sigma_grid = None
        self.h2s_logsig_mean = self.h2s_logsig_std = self.h2s_fref = None
        if h2s_wanted:
            self._h2s_selection_path = config_module.product_path(
                config, "bms", "h2s", "selection", "source", region=region)
            with h5py.File(self._h2s_selection_path, "r") as f:
                self.h2s_x_ladder = f["X_LADDER"][:].astype(np.float64)
                self.h2s_log10_sigma_grid = f["LOG10_SIGMA_GRID"][:].astype(np.float64)
            h2s_region_path = config_module.product_path(
                config, "bms", "h2s", "prior", "region", region=region)
            with h5py.File(h2s_region_path, "r") as f:
                self.h2s_logsig_mean = float(f["LOGSIG_MEAN"][()])
                self.h2s_logsig_std = float(f["LOGSIG_STD"][()])
            self.h2s_fref = _library_reference_flux(config, "h2s")

        # -- per-batch tabulation (`prepare`, module docstring): unset
        # until a batch is prepared; `log_density` refuses every class
        # until then (rule 6: fail on the impossible).
        self._prep_rows = None
        self._prep_star_eps = None
        self._prep_h2s_eps = None

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
        `CODING_RULES.md` 10b) from whichever of the two remaining
        per-source selection products (STAR family, H2S) this instance
        was scoped to, float16 -> float32 -- the one quantity too large
        to hold for a whole survey in memory at once (module docstring).
        GAL needs no such gather: there is no per-source galaxy product
        (owner, 2026-09-06); its compiled kernel reads the source's own
        four IRAC limits straight off the region table, already
        resident."""
        rows = np.asarray(rows, dtype=np.intp)
        uniq_rows = np.unique(rows)

        star_eps = {}
        if self._star_selection_path is not None:
            with h5py.File(self._star_selection_path, "r") as f:
                for c in self.shapes:
                    star_eps[c] = f["EPS_%s" % c.upper()][uniq_rows, :, :].astype(np.float32)
        h2s_eps = None
        if self._h2s_selection_path is not None:
            with h5py.File(self._h2s_selection_path, "r") as f:
                h2s_eps = f["EPS"][uniq_rows, :, :].astype(np.float32)

        self._prep_rows = uniq_rows
        self._prep_star_eps = star_eps
        self._prep_h2s_eps = h2s_eps

        # `Z[s]` by quadrature (owner ruling, 2026-09-06): the normaliser
        # of the density the fitter actually reads, computed where the
        # read lives -- not `Z_<CLS>` off the table any more (module
        # docstring above `_z_by_quadrature`). Needs `_prep_star_eps`/
        # `_prep_h2s_eps` already set (just above): the family/H2S
        # `selection` read gathers from them.
        self._prep_z = {c: _z_by_quadrature(self, c, uniq_rows) for c in self._classes}

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
        `Z` is `prepare`'s own quadrature (`_z_by_quadrature`), the
        normaliser of THIS density, not `Z_<CLS>` off the table any
        more. `Z = 0` (every object of this class is genuinely
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
        local2d = self._prep_local_index(rows2d.ravel()).reshape(rows2d.shape)
        z = self._prep_z[cls][local2d]
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
    whose quadrature `Z` (`_z_by_quadrature`, `prepare`) is zero has no
    object of this class catalogueable at its own limits -- every
    template correctly reads `-inf` there, so its integral is correctly
    zero, and it is counted separately (`n_zero`) rather than folded
    into the identity's own typical/worst deviation (owner ruling,
    2026-09-06). Reports typical (median) and worst deviation per class
    over the sources where `Z > 0`, and the wall time of one source's
    `log_density` call for one class at 4,066 models times 9 query
    points, extrapolated linearly to the survey's 8.66e6 sources."""
    prior = SourcePrior(config, region)
    rng = np.random.default_rng(seed)
    n = min(int(n_sources), prior.n_source)
    rows = rng.choice(prior.n_source, size=n, replace=False)
    prior.prepare(rows)

    devs = {cls: [] for cls in CLASSES}
    n_zero = {cls: 0 for cls in CLASSES}
    for row in rows:
        a_col = float(prior.table["A_COL_K"][row])
        local = int(prior._prep_local_index(np.array([row], dtype=np.intp))[0])

        for cls in FAMILY_CLASSES:
            if prior._prep_z[cls][local] <= 0.0:
                n_zero[cls] += 1
                continue
            shape = prior.shapes[cls]
            a_grid, b_grid = _family_grid(shape, a_col)
            a_grid = a_grid[a_grid > 0.0]
            integral = _integrate(prior, cls, row, a_grid, b_grid)
            devs[cls].append(abs(integral - 1.0))

        a_grid_cloud = _extinction_grid(prior, row, a_col)

        if prior._prep_z["gal"][local] <= 0.0:
            n_zero["gal"] += 1
        else:
            b_grid_gal = prior.gal_log10_s_grid - np.log10(prior.gal_fref[0])
            integral = _integrate(prior, "gal", row, a_grid_cloud, b_grid_gal, model_index=0)
            devs["gal"].append(abs(integral - 1.0))

        if prior._prep_z["yso"][local] <= 0.0:
            n_zero["yso"] += 1
        else:
            integral = _integrate_yso(prior, row, a_grid_cloud)
            devs["yso"].append(abs(integral - 1.0))

        # The check's own Sigma domain must be the same grid `_interp_eps_2d`
        # interpolates on (module docstring's "end bins held" convention) --
        # a wider independent range double-counts the held edge value past
        # the real grid, the row-1222 callable/product mismatch this fixes.
        if prior._prep_z["h2s"][local] <= 0.0:
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

    return worst, typical, n_zero, read_cost


def report(region, worst, typical, n_zero, read_cost):
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
    return lines


if __name__ == "__main__":
    import sys

    from sesnaimpute import config as _config_module

    cfg = _config_module.load(sys.argv[1])
    for _region in sys.argv[2:]:
        _worst, _typical, _n_zero, _read_cost = check(cfg, _region)
        for _line in report(_region, _worst, _typical, _n_zero, _read_cost):
            print(_line, flush=True)
