"""The column kernel `p(T | A_measured)` (SPEC_PRIORS.md section 1.2).

The prior conditions on a beam-measured column. This module answers what
the true pencil-beam column `T` can be, given that measurement, by
composing three terms in order:

  1. the arm's per-source measurement uncertainty (`A_COL_SIG_K`), served
     as a scale mixture over nearby sources rather than one number per
     column node -- the per-source width distribution is bimodal
     (Herschel near-constant, Planck much wider), so a single node width
     would misrepresent both populations;
  2. the dispersion of true column within the beam, a property of cloud
     structure that grows with column, evaluated at the arm's STATED beam
     (36.3" Herschel, 301.52" Planck -- never a per-source roll-off beam,
     spec 1.2's "the only place the sightline's width enters");
  3. the Herschel field zero point, folded into the measurement sigma in
     quadrature (`sigma_total = sqrt(sigma**2 + sigma_zp**2)`, Planck
     carries no zero-point term).

Every class's `a`-axis is convolved with this kernel once, at tabulation.
"""

import glob
import json
import math
import os
import re

import h5py
import numpy as np
from scipy.special import erf

# ====================================================================
# Inputs
# ====================================================================

#: The old per-region adopted-column directory. `A_COL_K`/`A_COL_SIG_K` are
#: the currency SPEC_PRIORS.md section 1.1 defines; `A_COL_PROVENANCE` is
#: 0 = Herschel / 1 = Planck.
_OLD_DUST_COLUMN_DIR = (
    "/Users/jtaylor/Dropbox/Research/SESNA_Complete/sky/derived/source/"
    "herschel-planck_adopted-dust-columns")
_ARM_PRODUCT_RE = re.compile(r"^dust_column_(planck_arm_.+|herschel_arm)\.hdf5$")

_OLD_SUBBEAM_PATH = (
    "/Users/jtaylor/Dropbox/Research/SESNA_Complete/sky/derived/region/"
    "herschel-column-structure/herschel_column_structure.hdf5")
_OLD_BLUR_SCALING_PATH = (
    "/Users/jtaylor/Dropbox/Research/SESNA_Complete/bms/prior/staging/"
    "herschel-region_blur_scaling.hdf5")

#: The three beams the sub-beam conditional tables are tabulated at.
_TABULATED = (("L108", 108.0), ("L302", 301.8), ("L821", 821.0))
#: The `spread/table` column each map class's stated-beam width is pooled
#: from (for `Kernel.second_moment`).
_POOLED_WIDTH_COLUMN = {"herschel": "W_abs_36.3", "planck": "W_abs_302"}

#: The two beams the sub-beam term is ever evaluated at (spec 1.2).
STATED_BEAM_ARCSEC = {"herschel": 36.3, "planck": 301.52072}


def _load_source_columns(dust_column_dir=_OLD_DUST_COLUMN_DIR):
    """`(column, sigma, provenance)`: every source's adopted column, its
    per-source uncertainty, and its arm code (0 Herschel / 1 Planck),
    concatenated over every region's `dust_column_<region>.hdf5`.

    repoint when <catalog/source columns> product lands: reads the OLD
    per-region files directly; the new catalog product replaces this glob.
    """
    paths = sorted(
        p for p in glob.glob(os.path.join(dust_column_dir, "dust_column_*.hdf5"))
        if not _ARM_PRODUCT_RE.match(os.path.basename(p)))
    if not paths:
        raise ValueError(
            "the column kernel needs the adopted-column products and none "
            "matched %s" % dust_column_dir)
    col, sig, code = [], [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            col.append(f["A_COL_K"][:].astype(np.float64))
            sig.append(f["A_COL_SIG_K"][:].astype(np.float64))
            code.append(f["A_COL_PROVENANCE"][:])
    return np.concatenate(col), np.concatenate(sig), np.concatenate(code)


def _load_subbeam_and_scaling(subbeam_path=_OLD_SUBBEAM_PATH,
                              blur_scaling_path=_OLD_BLUR_SCALING_PATH):
    """`(conditional, spread, scalars, sigma_zp_ak)`: the sub-beam
    conditional quantile tables and the region-pooled spread table (both
    from the sub-beam product), and the per-region beam-scaling exponents
    plus the Herschel field zero point (from the blur-scaling product).

    repoint when <sky/derived/subbeam region> product lands: reads the OLD
    `herschel_column_structure.hdf5` (conditional tables, spread table)
    and the OLD `blur_scaling.hdf5` (scalars, field zero point) exactly as
    `blur.SubbeamKernel.__init__` and `acol_kernel._pooled_subbeam_widths`
    read them; the new sub-beam product folds both into one region file.
    """
    conditional = {}
    with h5py.File(subbeam_path, "r") as f:
        regions = list(f["per_region"].keys())
        for reg in regions:
            tables = {}
            for lab, _ in _TABULATED:
                d = f["per_region/%s/conditional/%s" % (reg, lab)]
                cols = json.loads(d.attrs["columns"])
                ci = {c: i for i, c in enumerate(cols)}
                tables[lab] = (d[:], ci)
            conditional[reg] = tables
        columns = json.loads(f["spread"].attrs["columns"])
        ci_spread = {c: i for i, c in enumerate(columns)}
        spread_regions = [r.decode() if isinstance(r, bytes) else r
                          for r in f["spread/region"][:]]
        spread_table = f["spread/table"][:]

    with h5py.File(blur_scaling_path, "r") as f:
        sigma_zp_ak = float(f["field_zeropoint"].attrs["sigma_zp_ak"])
        scalars = {}
        for reg in regions:
            a = f["per_region/" + reg].attrs
            scalars[reg] = dict(
                beta=float(a["beta"]), W_abs_36p3=float(a["W_abs_36p3"]),
                completion_L108=float(a["completion_L108"]),
                completion_L302=float(a["completion_L302"]),
                completion_L821=float(a["completion_L821"]),
                med_off_302=float(a["med_off_302"]),
                offset_powerlaw_p=float(a["offset_powerlaw_p"]))

    spread = {"columns": ci_spread, "regions": spread_regions,
              "table": spread_table}
    return regions, conditional, spread, scalars, sigma_zp_ak


# ====================================================================
# Part 1 -- the measurement blur (SPEC_PRIORS.md 1.2, term 1)
# ====================================================================

_H_TOL = 1.0e-10
_H_MAX_ITER = 200
#: The scipy-idiomatic truncation radius for the mixture, in units of the
#: widest component in the window.
_KERNEL_TRUNCATE_SIGMA = 6.0


def _freedman_diaconis_edges(x, min_bins=4, max_bins=20000):
    """Bin edges by the Freedman-Diaconis rule: width = 2*IQR*n**(-1/3),
    falling back to Scott's rule (using the standard deviation) when the
    sample's IQR is zero.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    q75, q25 = np.percentile(x, [75.0, 25.0])
    width = 2.0 * (q75 - q25) * x.size ** (-1.0 / 3.0)
    if not width > 0:
        width = 3.49 * float(np.std(x, ddof=1)) * x.size ** (-1.0 / 3.0)
    n_raw = int(np.ceil((hi - lo) / width)) if width > 0 else min_bins
    n_bins = int(np.clip(n_raw, min_bins, max_bins))
    return np.linspace(lo, hi, n_bins + 1)


class ColumnSigmaKernel(object):
    """The measurement blur: at any column `A`, the mixture of the
    per-source Gaussians `N(0, sigma_i**2)` over the sources within one
    LOCAL RESOLUTION of `A`. Zero-centred by construction: `A_COL_SIG_K`
    is an uncertainty on `A_COL_K`, so the two live in one frame.

    The window is self-consistent: `h(A) = sqrt(mean sigma_i**2)` over
    `|A_COL_K_i - A| <= h(A)`, the fixed point of that equation, seeded
    from the survey-wide rms -- `h(A)` IS `resolution(A)`, not a separate
    number. The mixture inside that window is evaluated on the
    Freedman-Diaconis-binned sigma distribution, not source by source.
    """

    def __init__(self, column, sigma, provenance):
        column = np.asarray(column, float)
        sigma = np.asarray(sigma, float)
        good = np.isfinite(column) & np.isfinite(sigma) & (sigma > 0.0)
        order = np.argsort(column[good], kind="stable")
        self.column = column[good][order]
        self.sigma = sigma[good][order]
        self.code = np.asarray(provenance)[good][order]
        if self.column.size < 2:
            raise ValueError("fewer than two usable (column, sigma) rows")
        self._csq = np.concatenate([[0.0], np.cumsum(self.sigma ** 2)])
        self._comp = {}

    def bandwidth(self, A):
        """`h(A)`, the self-consistent window half-width, which is also
        `resolution(A)`. Scalar or array in, same shape out.
        """
        a = np.asarray(A, dtype=float)
        flat = np.atleast_1d(a).ravel()
        n = self.column.size
        seed = float(np.sqrt(self._csq[-1] / n))
        h = np.full(flat.shape, seed)
        live = np.ones(flat.shape, dtype=bool)
        for _ in range(_H_MAX_ITER):
            if not live.any():
                break
            lo = np.searchsorted(self.column, flat - h)
            hi = np.searchsorted(self.column, flat + h)
            cnt = hi - lo
            empty = cnt < 1
            new = np.where(
                empty, h * 1.5,
                np.sqrt((self._csq[np.minimum(hi, n)]
                         - self._csq[np.minimum(lo, n)])
                        / np.maximum(cnt, 1)))
            done = live & (~empty) & (np.abs(new - h) <= _H_TOL)
            step = np.where(empty, new, 0.5 * (h + new))
            step = np.where(done, new, step)
            h = np.where(live, step, h)
            live &= ~done
        return h.reshape(np.shape(a)) if np.ndim(a) else float(h[0])

    def window(self, A):
        """`(sigma, (col_lo, col_hi), h)`: the sources standing for node
        `A`, within the window `bandwidth(A)` wide."""
        h = float(self.bandwidth(float(A)))
        lo, hi = np.searchsorted(self.column, [float(A) - h, float(A) + h])
        if hi - lo < 1:
            i = int(np.clip(np.searchsorted(self.column, float(A)), 0,
                            self.column.size - 1))
            lo, hi = i, i + 1
        return (self.sigma[lo:hi],
                (float(self.column[lo]), float(self.column[hi - 1])), h)

    def components(self, A):
        """`(sigma_bin, weight)`: the mixture's Freedman-Diaconis-binned
        components at node `A`, weight summing to 1."""
        if float(A) in self._comp:
            return self._comp[float(A)]
        s, _, _ = self.window(A)
        if s.size < 2 or float(s.max() - s.min()) <= 0.0:
            out = (np.array([float(np.mean(s))]), np.array([1.0]))
            self._comp[float(A)] = out
            return out
        edges = _freedman_diaconis_edges(s)
        idx = np.clip(np.searchsorted(edges, s, side="right") - 1,
                      0, edges.size - 2)
        cnt = np.bincount(idx, minlength=edges.size - 1).astype(float)
        tot = np.bincount(idx, weights=s, minlength=edges.size - 1)
        occ = cnt > 0
        out = (tot[occ] / cnt[occ], cnt[occ] / cnt.sum())
        self._comp[float(A)] = out
        return out

    def resolution(self, A):
        """The mixture's own standard deviation at column `A` -- the
        extinction axis's measurement resolution. Scalar or array in,
        same shape out."""
        return self.bandwidth(A)


# ====================================================================
# Part 2 -- the sub-beam kernel (SPEC_PRIORS.md 1.2, term 2)
# ====================================================================

_LN16 = float(np.log(16.0))
_SQRT2 = float(np.sqrt(2.0))


def _quantiles_at(table, ci, ln_column):
    """Linear-interpolate the raw quantile columns of a conditional table
    in ln(column), clamped at the ends -- no extrapolation past the
    measured column range for a fixed beam."""
    x = np.log(table[:, ci["A_HERSCHEL_SMOOTH_K"]])
    order = np.argsort(x)
    x = x[order]
    out = {}
    for name in ("median_d", "q16", "q84", "q99"):
        y = table[order, ci[name]]
        out[name] = float(np.interp(ln_column, x, y))
    return out


def _shape_from_quantiles(q):
    """`(mu, sigma, s_break, k)` of the Gaussian-core / exponential-tail
    composite, in closed form from the four raw quantiles."""
    mu = q["median_d"]
    sigma = max(0.5 * (q["q84"] - q["q16"]), 1.0e-6)
    s_break = q["q84"]
    denom = q["q99"] - q["q84"]
    k = _LN16 / denom if denom > 1.0e-6 else _LN16 / 1.0e-6
    return mu, sigma, s_break, k


def _composite_logpdf(s, mu, sigma, s_break, k):
    """log p(s) for one Gaussian-core / continuous-exponential-tail
    composite, normalised on the real line."""
    z_break = (s_break - mu) / sigma
    gt_log = -0.5 * z_break * z_break
    norm = (sigma * np.sqrt(np.pi / 2.0) * (erf(z_break / _SQRT2) + 1.0)
            + np.exp(gt_log) / k)
    log_norm = np.log(norm)
    s = np.asarray(s, dtype=float)
    lo = s <= s_break
    out = np.empty_like(s)
    z = (s - mu) / sigma
    out[lo] = -0.5 * z[lo] * z[lo] - log_norm
    out[~lo] = gt_log - k * (s[~lo] - s_break) - log_norm
    return out


def _composite_logpdf_batch(s, mu, sigma, s_break, k):
    """[perf] `_composite_logpdf` for a whole column of parameter sets at
    once: `s` is `(n_col, n_T)`; `mu`/`sigma`/`s_break`/`k` are
    `(n_col, 1)`. Row `j` reproduces the scalar function on row `j`."""
    z_break = (s_break - mu) / sigma
    gt_log = -0.5 * z_break * z_break
    norm = (sigma * np.sqrt(np.pi / 2.0) * (erf(z_break / _SQRT2) + 1.0)
            + np.exp(gt_log) / k)
    log_norm = np.log(norm)
    z = (s - mu) / sigma
    core = -0.5 * z * z - log_norm
    if not np.any(np.isfinite(s_break)):
        return core
    with np.errstate(invalid="ignore"):
        tail = gt_log - k * (s - s_break) - log_norm
    return np.where(s <= s_break, core, tail)


def _core_mass_fraction(mu, sigma, s_break, k):
    """The composite's own core mass fraction -- the split a sampler
    would use between the Gaussian core and the exponential tail."""
    z_break = (s_break - mu) / sigma
    core = sigma * np.sqrt(np.pi / 2.0) * (erf(z_break / _SQRT2) + 1.0)
    gt = np.exp(-0.5 * z_break * z_break)
    return core / (core + gt / k)


class _RegionKernel(object):
    """One region's sub-beam parameters: the raw conditional tables (three
    tabulated beams) plus the region's own beam-rescaling and
    offset-rescaling exponents."""

    _QUANTILE_COLUMNS = ("median_d", "q16", "q84", "q99")

    def __init__(self, name, tables, scalars):
        self.name = name
        self._sorted = {}
        for lab, (table, ci) in tables.items():
            x = np.log(table[:, ci["A_HERSCHEL_SMOOTH_K"]])
            order = np.argsort(x)
            self._sorted[lab] = (x[order],
                                 {n: table[order, ci[n]]
                                  for n in self._QUANTILE_COLUMNS})
        self.beta = scalars["beta"]
        self.W_abs_36p3 = scalars["W_abs_36p3"]
        self.completion = {"L108": scalars["completion_L108"],
                            "L302": scalars["completion_L302"],
                            "L821": scalars["completion_L821"]}
        self.med_off_302 = scalars["med_off_302"]
        self.offset_p = scalars["offset_powerlaw_p"]

    def w_abs(self, L):
        """`W_abs(L) = W_abs(36.3) * (L/36.3)**((beta-2)/2)`."""
        return self.W_abs_36p3 * (L / 36.3) ** ((self.beta - 2.0) / 2.0)

    def offset(self, L):
        """The offset power law through the two measured points."""
        return self.med_off_302 * (L / 301.8) ** self.offset_p

    def _nearest_ref(self, L):
        labs = [lab for lab, _ in _TABULATED]
        Ls = np.array([Lr for _, Lr in _TABULATED])
        i = int(np.argmin(np.abs(np.log(L) - np.log(Ls))))
        return labs[i], float(Ls[i])

    def params_batch(self, columns, beam_fwhm_arcsec):
        """`(mu, sigma, s_break, k)`, each `(n_col,)`: the composite
        kernel's shape for `d = ln T - ln column` at this region, this
        beam, over an array of conditioning columns.

        Below 36.3" (Herschel's own beam, where the table is measured
        FROM outward) there is no column-binned tail data, so this
        returns a plain, column-independent Gaussian (`s_break = +inf`).
        """
        columns = np.asarray(columns, dtype=float)
        L = float(beam_fwhm_arcsec)
        n = columns.size
        if L <= 36.3:
            return (np.full(n, self.offset(L)), np.full(n, self.w_abs(L)),
                    np.full(n, float("inf")), np.ones(n))

        lab, L_ref = self._nearest_ref(L)
        x, ys = self._sorted[lab]
        ln_column = np.log(columns)
        q = {name: np.interp(ln_column, x, ys[name])
             for name in self._QUANTILE_COLUMNS}

        mu0 = q["median_d"]
        sigma0 = np.maximum(0.5 * (q["q84"] - q["q16"]), 1.0e-6)
        s_break0 = q["q84"]
        denom = q["q99"] - q["q84"]
        with np.errstate(divide="ignore", invalid="ignore"):
            k0 = np.where(denom > 1.0e-6, _LN16 / denom, _LN16 / 1.0e-6)

        # completion[lab] = W_abs(L_ref)/sd_two_scale(L_ref), so the
        # ABSOLUTE (completed) width at L_ref is sigma0*completion[lab];
        # the smooth power law W_abs(L)/W_abs(L_ref) moves it to L.
        F = (self.w_abs(L) / self.w_abs(L_ref)) * self.completion[lab]
        R = (L / L_ref) ** self.offset_p

        mu = mu0 * R
        sigma = sigma0 * F
        s_break = mu + (s_break0 - mu0) * F
        k = k0 / F
        return mu, sigma, s_break, k


class _SubbeamKernel(object):
    """`p(T | column, beam)`: the sub-beam term, marginalised over the
    Herschel-region spread as an equal-weight mixture (region labels are
    never a conditioning variable). Below the smallest column any region's
    conditional table was measured at, the conditioning column is floored
    -- `s = ln(T) - ln(column)` is a log-ratio that diverges as the
    conditioning column falls toward zero.
    """

    def __init__(self, regions, conditional, scalars):
        self._kernels = []
        min_column = float("inf")
        for reg in regions:
            self._kernels.append(_RegionKernel(reg, conditional[reg],
                                                scalars[reg]))
            for lab, _ in _TABULATED:
                table, ci = conditional[reg][lab]
                col_here = float(table[:, ci["A_HERSCHEL_SMOOTH_K"]].min())
                min_column = min(min_column, col_here)
        self._log_n = float(np.log(len(self._kernels)))
        self.min_conditioning_column_ak = min_column

    def pdf_columns(self, T, columns, beam_fwhm_arcsec, chunk=256):
        """`(n_col, n_T)`: row `j` is `p(T | columns[j], beam)`, one pass
        over the region mixture for the whole conditioning-column axis."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        columns = np.atleast_1d(np.asarray(columns, dtype=float))
        eff = np.maximum(columns, self.min_conditioning_column_ak)
        ln_T = np.log(T)
        out = np.empty((eff.size, T.size), dtype=float)
        n_reg = len(self._kernels)
        for start in range(0, eff.size, int(chunk)):
            cols = eff[start:start + int(chunk)]
            s = ln_T[np.newaxis, :] - np.log(cols)[:, np.newaxis]
            terms = np.empty((n_reg, cols.size, T.size), dtype=float)
            for i, rk in enumerate(self._kernels):
                mu, sigma, s_break, k = rk.params_batch(cols, beam_fwhm_arcsec)
                terms[i] = _composite_logpdf_batch(
                    s, mu[:, np.newaxis], sigma[:, np.newaxis],
                    s_break[:, np.newaxis], k[:, np.newaxis])
            m = terms.max(axis=0)
            terms -= m
            np.exp(terms, out=terms)
            ln_p = m + np.log(terms.sum(axis=0)) - self._log_n
            out[start:start + int(chunk)] = np.exp(ln_p - ln_T[np.newaxis, :])
        return out

    def pdf(self, T, column_ak, beam_fwhm_arcsec):
        """`p(T | column, beam)` at a single conditioning column."""
        return self.pdf_columns(T, [column_ak], beam_fwhm_arcsec)[0]


# ====================================================================
# The composition (SPEC_PRIORS.md 1.2, all three terms)
# ====================================================================

_LN_SPAN = (-3.0, 5.0)
_DEFAULT_N_NODES = 257
#: Sanity floor on the T-quadrature's captured mass, checked before
#: `_renormalise_checked` restores exact unit mass -- guards a
#: badly-under-spanned grid; not the accuracy bar.
_MIN_CAPTURED_MASS = 0.99
#: Gauss-Hermite order for marginalising the (Gaussian) measurement blur
#: against the sub-beam kernel.
_GH_ORDER = 9

_GH_RULE = None


def _gauss_hermite(center, sigma):
    """`(a_prime, weight)`: Gauss-Hermite nodes/weights for
    `Integral f(a') N(a'; center, sigma) da' ~= sum weight * f(a_prime)`.
    """
    global _GH_RULE
    if _GH_RULE is None:
        nodes, weights = np.polynomial.hermite.hermgauss(_GH_ORDER)
        _GH_RULE = (nodes, weights / math.sqrt(math.pi))
    nodes, w = _GH_RULE
    a_prime = center + math.sqrt(2.0) * sigma * nodes
    return a_prime, w


def _t_grid(center, n, sigma_meas=0.0):
    """The T-quadrature grid, `_LN_SPAN` wide in `ln(T/center)`, widened
    additively when `sigma_meas` is a non-trivial fraction of `center`
    (the low-column regime, near the column floor) so the composed
    density's one-sided tail is not under-captured."""
    lo, hi = _LN_SPAN
    center = float(center)
    margin = (math.log1p(6.0 * float(sigma_meas) / max(center, 1.0e-6))
             if sigma_meas > 0.0 else 0.0)
    return center * np.exp(np.linspace(lo - margin, hi + margin, int(n)))


def _trapezoid_weights(t, dens):
    w = np.empty_like(t)
    w[0] = 0.5 * (t[1] - t[0])
    w[-1] = 0.5 * (t[-1] - t[-2])
    w[1:-1] = 0.5 * (t[2:] - t[:-2])
    return w * dens


def _renormalise_checked(t, w, context):
    total = float(np.sum(w))
    if total < _MIN_CAPTURED_MASS:
        raise ValueError(
            "the T quadrature captured only %.4f of p(T | %s); widen "
            "_LN_SPAN rather than truncating the sub-beam kernel's "
            "one-sided tail" % (total, context))
    return t, w / total


class Kernel(object):
    """`p(T | A_measured)`, composed from the measurement blur and the
    sub-beam kernel (SPEC_PRIORS.md 1.2). Constructed by `load(config)`.
    """

    def __init__(self, acol, subbeam, sigma_zp_herschel, w_herschel_36p3,
                w_planck_301p8):
        self._acol = acol
        self._subbeam = subbeam
        self.sigma_zp_herschel = sigma_zp_herschel
        self.w_herschel_36p3 = w_herschel_36p3
        self.w_planck_301p8 = w_planck_301p8
        self._frac_h_prefix = None

    # -- the batched composition core --------------------------------
    def _pdf_multi(self, t_matrix, cond_matrix, beam, chunk=64):
        """`(M, K, n)`: entry `[i, j]` is `p(t_matrix[i] | cond_matrix[i,
        j], beam)`, one pass over the sub-beam region mixture for the
        whole `(M, K)` conditioning grid instead of one call per row."""
        t_matrix = np.asarray(t_matrix, dtype=float)
        cond_matrix = np.asarray(cond_matrix, dtype=float)
        m, k = cond_matrix.shape
        n = t_matrix.shape[1]
        ln_T = np.log(t_matrix)
        floor = self._subbeam.min_conditioning_column_ak
        ln_cond = np.log(np.maximum(cond_matrix, floor))
        kernels = self._subbeam._kernels
        n_reg = len(kernels)
        log_n = self._subbeam._log_n
        out = np.empty((m, k, n), dtype=float)
        for start in range(0, m, int(chunk)):
            stop = min(start + int(chunk), m)
            b = stop - start
            flat_cond = np.exp(ln_cond[start:stop]).reshape(-1)
            ln_T_block = ln_T[start:stop]
            ln_cond_block = ln_cond[start:stop]
            s = (ln_T_block[:, np.newaxis, :]
                - ln_cond_block[:, :, np.newaxis])
            terms = np.empty((n_reg, b, k, n), dtype=float)
            for r, rk in enumerate(kernels):
                mu, sigma_p, s_break, kk = rk.params_batch(flat_cond, beam)
                terms[r] = _composite_logpdf_batch(
                    s, mu.reshape(b, k, 1), sigma_p.reshape(b, k, 1),
                    s_break.reshape(b, k, 1), kk.reshape(b, k, 1))
            mmax = terms.max(axis=0)
            terms -= mmax
            np.exp(terms, out=terms)
            ln_p = mmax + np.log(terms.sum(axis=0)) - log_n
            out[start:stop] = np.exp(ln_p - ln_T_block[:, np.newaxis, :])
        return out

    def _compose_gaussian_blur_batch(self, columns, sigma_meas, beam, n=None):
        """`(t, w)`, each `(M, n)`: row `i` is `p(T) = Integral K_subbeam(T
        | A', beam) N(A'; columns[i], sigma_meas[i]) dA'`, the measurement
        blur (already carrying the zero point in quadrature) composed
        with the sub-beam kernel by Gauss-Hermite quadrature. A row with
        `sigma_meas <= 0` is the exact special case: the sub-beam kernel's
        own density at `columns[i]`, no quadrature.
        """
        columns = np.asarray(columns, dtype=float)
        sigma_meas = np.asarray(sigma_meas, dtype=float)
        floor = self._subbeam.min_conditioning_column_ak
        n = _DEFAULT_N_NODES if n is None else int(n)
        m = columns.size
        if m == 0:
            return np.empty((0, n)), np.empty((0, n))

        t = np.empty((m, n), dtype=float)
        for i in range(m):
            t[i] = _t_grid(columns[i], n, sigma_meas=sigma_meas[i])

        exact = sigma_meas <= 0.0
        dens = np.empty((m, n), dtype=float)

        if np.any(exact):
            idx = np.nonzero(exact)[0]
            cond = np.maximum(columns[idx], floor).reshape(-1, 1)
            rows = self._pdf_multi(t[idx], cond, beam)
            dens[idx] = rows[:, 0, :]

        if np.any(~exact):
            idx = np.nonzero(~exact)[0]
            centers = columns[idx]
            sigmas = sigma_meas[idx]
            # populate the module-wide GH rule (if not already) and read
            # its bare nodes/weights directly, since the per-row centers
            # and sigmas below do the scaling `_gauss_hermite` would.
            _gauss_hermite(0.0, 0.0)
            gh_nodes, w_gh = _GH_RULE
            a_prime = (centers[:, np.newaxis]
                      + math.sqrt(2.0) * sigmas[:, np.newaxis]
                      * gh_nodes[np.newaxis, :])
            a_prime = np.maximum(a_prime, floor)
            rows = self._pdf_multi(t[idx], a_prime, beam)
            block = np.zeros((idx.size, n), dtype=float)
            for j in range(gh_nodes.size):
                block = block + w_gh[j] * rows[:, j, :]
            dens[idx] = block

        w = np.empty((m, n), dtype=float)
        for i in range(m):
            wi = _trapezoid_weights(t[i], dens[i])
            context = "column=%r, beam=%r" % (float(columns[i]), beam)
            _, w[i] = _renormalise_checked(t[i], wi, context)
        return t, w

    # -- public: the quadrature form, batched over sources -------------
    def per_sources(self, columns, sigma, map_class, n=None):
        """`(t, w)`, each `(M, n)`: the measurement blur (the source's own
        `sigma`, with the Herschel field zero point folded in in
        quadrature) composed with the sub-beam kernel at `map_class`'s
        stated beam. Weights sum to 1 per row.
        """
        if map_class not in STATED_BEAM_ARCSEC:
            raise ValueError("map_class must be one of %r, got %r"
                             % (sorted(STATED_BEAM_ARCSEC), map_class))
        beam = STATED_BEAM_ARCSEC[map_class]
        columns = np.atleast_1d(np.asarray(columns, dtype=float))
        sigma_zp = self.sigma_zp_herschel if map_class == "herschel" else 0.0
        sigma = np.broadcast_to(
            np.asarray(sigma, dtype=float), columns.shape).astype(float)
        sigma_total = np.sqrt(sigma ** 2 + sigma_zp ** 2)
        return self._compose_gaussian_blur_batch(columns, sigma_total, beam,
                                                  n=n)

    # -- public: the node-tabulated form, marginal route only ----------
    def nodes(self, column, map_class, n=None):
        """`(t, w)` on `T`: the node-tabulated composed kernel at column
        node `column` -- the measurement blur's own scale-mixture
        components at this node (`ColumnSigmaKernel.components`), each
        composed with the sub-beam kernel by Gauss-Hermite quadrature and
        combined by its own mixture weight. The Herschel field zero point
        is folded into every component in quadrature (the marginal
        route), matching `.per_sources`.
        """
        if map_class not in STATED_BEAM_ARCSEC:
            raise ValueError("map_class must be one of %r, got %r"
                             % (sorted(STATED_BEAM_ARCSEC), map_class))
        beam = STATED_BEAM_ARCSEC[map_class]
        A = float(column)
        sig, w_sig = self._acol.components(A)
        sig = np.array(sig, dtype=float, copy=True)
        if map_class == "herschel":
            sig = np.sqrt(sig ** 2 + self.sigma_zp_herschel ** 2)
        n = _DEFAULT_N_NODES if n is None else int(n)
        floor = self._subbeam.min_conditioning_column_ak
        center = max(A, floor)
        t = _t_grid(center, n, sigma_meas=float(np.max(sig)) if sig.size else 0.0)

        columns_gh = []
        plan = []
        for sig_i, wt_i in zip(sig, w_sig):
            if sig_i <= 0.0:
                plan.append((wt_i, None, len(columns_gh)))
                columns_gh.append(center)
                continue
            a_prime, w_gh = _gauss_hermite(center, float(sig_i))
            a_prime = np.maximum(a_prime, floor)
            plan.append((wt_i, w_gh, len(columns_gh)))
            columns_gh.extend(a_prime.tolist())
        rows = self._subbeam.pdf_columns(t, np.array(columns_gh, dtype=float),
                                         beam)
        dens = np.zeros_like(t)
        for wt_i, w_gh, at in plan:
            if w_gh is None:
                dens = dens + wt_i * rows[at]
                continue
            comp = np.zeros_like(t)
            for j in range(w_gh.size):
                comp = comp + w_gh[j] * rows[at + j]
            dens = dens + wt_i * comp
        return _renormalise_checked(
            t, _trapezoid_weights(t, dens),
            "node A=%r, beam=%r" % (A, beam))

    # -- public: the second moment -------------------------------------
    def _frac_herschel(self, a, h):
        """`f_H(A)`: the fraction of the node window's sources carrying
        Herschel provenance, batched over array `a` given its own window
        half-width `h`."""
        code = self._acol.code
        if self._frac_h_prefix is None:
            self._frac_h_prefix = np.concatenate(
                [[0.0], np.cumsum(code == 0)])
        column = self._acol.column
        n = column.size
        lo = np.searchsorted(column, a - h)
        hi = np.maximum(np.searchsorted(column, a + h), lo + 1)
        lo_c = np.minimum(lo, n)
        hi_c = np.minimum(hi, n)
        count = np.maximum(hi_c - lo_c, 0)
        prefix = self._frac_h_prefix
        with np.errstate(invalid="ignore"):
            frac = (prefix[hi_c] - prefix[lo_c]) / count
        return np.where(count > 0, frac, float("nan"))

    def second_moment(self, A):
        """`Var(T | A)`, the composed kernel's own second moment:

            Var(T|A) = sigma_A**2 * exp(W**2) + A**2 * (exp(W**2) - 1)
            sigma_A**2 = resolution(A)**2 + sigma_ZP**2 * f_H(A)

        `f_H(A)` is the node window's own Herschel fraction; `sigma_ZP` is
        Herschel-scoped so it enters weighted by `f_H`. `W` is the
        sub-beam width mixed over the two map classes by that same
        fraction. Scalar or array in, same shape out.
        """
        a = np.atleast_1d(np.asarray(A, dtype=float))
        resolution = self._acol.resolution(a)
        h = np.atleast_1d(np.asarray(resolution, dtype=float))
        f_h = self._frac_herschel(a, h).reshape(a.shape)
        sigma_a2 = resolution ** 2 + self.sigma_zp_herschel ** 2 * f_h
        w2 = (f_h * self.w_herschel_36p3 ** 2
              + (1.0 - f_h) * self.w_planck_301p8 ** 2)
        var = sigma_a2 * np.exp(w2) + a ** 2 * (np.exp(w2) - 1.0)
        return var.reshape(np.shape(A)) if np.ndim(A) else float(var[0])


def _pooled_subbeam_widths(spread):
    """`(w_herschel_36p3, w_planck_301p8)`: the unweighted arithmetic mean,
    over every Herschel-covered region, of that region's own `W_abs` at
    each map class's stated beam -- the widths `Kernel.second_moment`
    mixes over map class.
    """
    table, ci = spread["table"], spread["columns"]
    pooled = {}
    for map_class, col in _POOLED_WIDTH_COLUMN.items():
        pooled[map_class] = float(np.mean(table[:, ci[col]]))
    return pooled["herschel"], pooled["planck"]


def load(config):
    """Builds the composed column kernel from the survey's adopted-column
    products and sub-beam/blur-scaling products (SPEC_PRIORS.md 1.2)."""
    column, sigma, code = _load_source_columns()
    acol = ColumnSigmaKernel(column, sigma, code)

    regions, conditional, spread, scalars, sigma_zp_ak = \
        _load_subbeam_and_scaling()
    subbeam = _SubbeamKernel(regions, conditional, scalars)
    w_herschel_36p3, w_planck_301p8 = _pooled_subbeam_widths(spread)

    return Kernel(acol, subbeam, sigma_zp_ak, w_herschel_36p3, w_planck_301p8)
