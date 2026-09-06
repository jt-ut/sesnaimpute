"""Whether a source clears SESNA's own catalog cut, the hybrid
extinction law the test dims by, and the compiled kernels that turn a
class's external population into a source's exact pass fraction
(SPEC_PRIORS.md section 1.3, IMPLEMENTATION.md section 4).

SESNA catalogs a source when at least two of its eight bands, dimmed by
its own dust column, clear the source's detection limit
(`catalog.limits.limits`). The dimming coefficient per band is not one
fixed law: it blends smoothly, in log column, from a diffuse-ISM curve
below A_K = 0.5 to a dense-cloud curve above A_K = 1.0
(`law_dense_weight`, `kappa_hybrid`). A class's selection is evaluated
exactly, per source, at the source's own eight limits, and the
population's colour distribution is conditioned on brightness (each
member counted only within its own brightness bin): `pass_fractions_
binned` (one class) and `pass_fractions_binned_multi` (several classes
sharing one member subsample) walk a class's whole external population
once per source and per query extinction, reducing the two-of-eight test
to one critical brightness per population member and reading the pass
fraction off as a weight fraction within that member's own bin.
"""

import os

import numba
import numpy as np

from sesnaimpute import definitions

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: SESNA's catalog inclusion rule: >= 2 of 8 bands above the local limit.
MIN_BANDS = 2

#: SPEC_PRIORS.md 1.3 -- the two named laws the ramp blends between.
LAW_DIFFUSE = "draine_rv3.1"
LAW_DENSE = "whitney.r550"

#: The ramp's domain, A_K magnitudes: 0 (diffuse law only) at and below
#: LAW_RAMP_LO, 1 (dense law only) at and above LAW_RAMP_HI.
LAW_RAMP_LO = 0.5
LAW_RAMP_HI = 1.0

_V_BAND_UM = 0.55

_LAW_CACHE = {}
_K_CACHE = {}

#: The shared scaled-extinction ladder a source's exact selection is
#: tabulated on: `x = a / A_s`. The last four points cover the column
#: kernel's tail past the source's own column; `1.7` was added (owner,
#: 2026-09-06) because the tail past it held 5-10% of the kernel's mass
#: and interpolated at 6-8% error before the point was there.
X_LADDER = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.4, 1.7, 2.0, 2.8])

#: A class's own brightness grid: this many points.
N_B_GRID = 24

#: The fixed-seed population subsample cap. The colour distribution is
#: now conditioned on brightness (binned by each member's own log10 B on
#: the class's 24-point grid, `pass_fractions_binned`), so the relevant
#: draw size is per BIN, not the total: 20,000 members over 24 bins is
#: ~833/bin if the population were spread evenly (Monte Carlo error on a
#: pass fraction near 0.5, sqrt(0.25/833) ~ 1.7%), but the population
#: concentrates toward the middle of its own 0.1-99.9% brightness range,
#: so the populated central bins hold several thousand members each
#: (error well under 1%) and only the sparse tail bins run above it,
#: disclosed per source rather than hidden in an average.
#: Cut from 20,000 to 15,000 (owner, 2026-09-06): 20,000 measured 109 min
#: for Cygnus X, over the one-hour target; 15,000 costs ~0.8% Monte Carlo
#: error per brightness bin at the 1% bar, scaling as 1/sqrt(n).
SUBSAMPLE_CAP = 15_000

#: A brightness bin with fewer than this many subsample members is too
#: sparse for the conditioned (own-bin) estimate -- Monte Carlo error
#: above ~2% at eps=0.5 (owner, 2026-09-06). Below it, the bin's stored
#: selection is the MARGINAL estimate instead (every subsample member
#: rescaled to the bin's own brightness, the `pass_curves` form): a
#: statistician's pooled-stratum fallback, not a hidden zero.
MIN_BIN_MEMBERS = 500


def _parse_info(text):
    fields = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(None, 1)
        fields[key] = value.strip()
    return fields


def _load_law_curve(config, law):
    """`(wave_um, opacity_cm2_per_g)`, sorted ascending in wavelength,
    read from `<data_root>/sky/download/extinction_laws/<law>/<law>.par`
    at the columns named in the sibling `<law>.info` file -- the same
    file pair and column convention the SED fitter's own law loader
    reads. Fails with one sentence naming the RUNBOOK line that makes
    the input (`sky.download.extinction_laws.build`) when either file
    is missing.
    """
    cache_key = (config.data_root, law)
    if cache_key in _LAW_CACHE:
        return _LAW_CACHE[cache_key]
    law_dir = f"{config.data_root}/sky/download/extinction_laws/{law}"
    info_path = os.path.join(law_dir, f"{law}.info")
    par_path = os.path.join(law_dir, f"{law}.par")
    if not os.path.isfile(info_path) or not os.path.isfile(par_path):
        raise ValueError(
            f"prior.selection: no extinction law {law!r} at {law_dir!r} "
            f"-- run RUNBOOK.sh's sesnaimpute.sky.download.extinction_laws.build line")
    with open(info_path) as f:
        info = _parse_info(f.read())
    colidx_wav = int(info["colidx_wav"])
    colidx_extinction = int(info["colidx_extinction"])
    raw = np.loadtxt(par_path, usecols=(colidx_wav, colidx_extinction))
    order = np.argsort(raw[:, 0])
    wave_um, opacity = raw[order, 0], raw[order, 1]
    _LAW_CACHE[cache_key] = (wave_um, opacity)
    return _LAW_CACHE[cache_key]


def extinction_k(config, law):
    """`k_i = chi(lambda_i) / chi(0.55um)` for the 8 census bands, from
    the law's own tabulated curve, linear interpolation in wavelength.
    """
    cache_key = (config.data_root, law)
    if cache_key not in _K_CACHE:
        wave_um, opacity = _load_law_curve(config, law)
        wav = np.array([b.wvl_um for b in definitions.BANDS])
        _K_CACHE[cache_key] = (np.interp(wav, wave_um, opacity)
                               / np.interp(_V_BAND_UM, wave_um, opacity))
    return _K_CACHE[cache_key].copy()


def kappa_ak(config, law):
    """`kappa_i = k_i / k_Ks` -- the K-currency per-band dimming vector;
    `kappa_Ks == 1` exactly.
    """
    k = extinction_k(config, law)
    return k / k[BAND_KEYS.index("Ks")]


def _ak_per_av_curve(config, law):
    """`(A_K/A_V)` for `law`, read off the law's own curve at 0.55um and
    Ks -- the reciprocal of `extinction_k(config, law)` at Ks.
    """
    return float(extinction_k(config, law)[BAND_KEYS.index("Ks")])


def law_dense_weight(a):
    """`w(A_col)`: the smoothstep weight of the dense-cloud curve in the
    hybrid law, 0 at and below `LAW_RAMP_LO`, 1 at and above
    `LAW_RAMP_HI`, evaluated in log column (SPEC_PRIORS.md 1.3).
    """
    a = np.asarray(a, dtype=float)
    x = np.log(a / LAW_RAMP_LO) / np.log(LAW_RAMP_HI / LAW_RAMP_LO)
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def kappa_hybrid(config, w):
    """The per-band dimming vector at ramp weight `w`: the K-normalised
    convex blend `(1-w)*kappa_ak(config, LAW_DIFFUSE) + w*kappa_ak(config,
    LAW_DENSE)`.
    """
    w = np.asarray(w, dtype=float)
    kd = kappa_ak(config, LAW_DIFFUSE)
    kw = kappa_ak(config, LAW_DENSE)
    return (1.0 - w)[..., None] * kd + w[..., None] * kw


def ak_per_av(config, w):
    """`(A_K/A_V)` at ramp weight `w`: the harmonic blend of the two
    laws' own curve-internal ratios.
    """
    r_d = _ak_per_av_curve(config, LAW_DIFFUSE)
    r_w = _ak_per_av_curve(config, LAW_DENSE)
    w = np.asarray(w, dtype=float)
    return 1.0 / ((1.0 - w) / r_d + w / r_w)


def _kappa(config, law):
    """Resolve `law` to an `(8,)` K-currency dimming vector: a
    registered name (`kappa_ak`) or an already-built `(8,)` vector
    passed through unchanged.
    """
    if isinstance(law, str):
        return kappa_ak(config, law)
    kap = np.asarray(law, dtype=float)
    if kap.shape != (N_BANDS,):
        raise ValueError(
            f"law must be a registered name or an ({N_BANDS},) kappa "
            f"vector, got array of shape {kap.shape}")
    return kap


def epsilon(config, flux, f_lim, a, law, coord_flux=None, weights=None,
            min_bands=MIN_BANDS):
    """The two-of-eight detection test (SPEC_PRIORS.md 1.3), counted the
    direct way: the (weighted) fraction of `flux` rows whose dimmed
    8-band SED clears `>= min_bands` of `f_lim`.
    """
    flux = np.asarray(flux, dtype=float)
    if coord_flux is not None:
        coord_index = BAND_KEYS.index("I2")
        flux = flux * (float(coord_flux) / flux[:, coord_index:coord_index + 1])
    n_clear = np.sum(
        np.log10(flux) - 0.4 * float(a) * _kappa(config, law)
        >= np.log10(np.asarray(f_lim, dtype=float)), axis=1)
    passed = n_clear >= min_bands
    if weights is None:
        return float(np.mean(passed))
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights * passed) / np.sum(weights))


def column_threshold(config, flux, f_lim, law, min_bands=MIN_BANDS):
    """`A_max`: the largest K-band column at which an object still
    clears the catalog cut -- the exact algebraic reduction of `epsilon`
    (second-largest of `(log f - log F_lim) / (0.4 kappa)` per band).
    """
    flux = np.asarray(flux, dtype=float)
    kap = _kappa(config, law)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_band = (np.log10(flux) - np.log10(np.asarray(f_lim, dtype=float))) / (0.4 * kap)
    a_band = np.where(np.isfinite(a_band), a_band, -np.inf)
    return np.partition(a_band, -min_bands, axis=-1)[..., -min_bands]


@numba.njit(parallel=True)
def pass_curves(log10_lim, a_query, kappa, log10_flux, log10_b_pop, weight, b_grid):
    """`(n_src, n_x, n_b)` f4: the marginal exact selection curve for
    every source and query extinction, against the whole population,
    colour taken independent of brightness -- kept for AGB and AGB's
    photospheric bound, whose population (a handful of GRAMS-matched
    stars) is too small for the brightness-conditioned binning
    `pass_fractions_binned` uses for STAR and PAHC.

    `log10_lim` (n_src, 8): the source's own 8-band log10 detection
    limits (mJy), `catalog.limits` order. `a_query` (n_src, n_x): the
    extinctions to evaluate at. `kappa` (n_src, n_x, 8): the hybrid
    law's per-band dimming at each `a_query`. `log10_flux` (n_pop, 8),
    `log10_b_pop` (n_pop,), `weight` (n_pop,): the population members.
    `b_grid` (n_b,) ascending.

    For member `j` at `(s, k)`, `bcrit` is the second-smallest over
    bands `i` of `log10_lim[s,i] - (log10_flux[j,i] -
    0.4*a_query[s,k]*kappa[s,k,i] - log10_b_pop[j])` -- the two-of-eight
    test reduced to one critical brightness. A band whose `log10_flux`
    is not finite never clears (its threshold is +inf).
    `eps[s,k,m]` is the weight-fraction of members with `bcrit <=
    b_grid[m]`: each member's weight is dropped into a bin by
    `searchsorted` on `b_grid` and the bins are cumulatively summed.
    """
    n_src, n_x = a_query.shape
    n_b = b_grid.shape[0]
    n_pop, n_bands = log10_flux.shape
    eps = np.zeros((n_src, n_x, n_b), dtype=np.float32)

    w_total = 0.0
    for j in range(n_pop):
        w_total += weight[j]

    for s in numba.prange(n_src):
        thresh = np.empty((n_x, n_bands), dtype=np.float64)
        for k in range(n_x):
            for i in range(n_bands):
                thresh[k, i] = log10_lim[s, i] + 0.4 * a_query[s, k] * kappa[s, k, i]
        for k in range(n_x):
            bin_w = np.zeros(n_b + 1, dtype=np.float64)
            for j in range(n_pop):
                smallest = np.inf
                second = np.inf
                for i in range(n_bands):
                    lf = log10_flux[j, i]
                    if not np.isfinite(lf):
                        val = np.inf
                    else:
                        val = thresh[k, i] - lf + log10_b_pop[j]
                    if val < smallest:
                        second = smallest
                        smallest = val
                    elif val < second:
                        second = val
                idx = np.searchsorted(b_grid, second)
                if idx > n_b:
                    idx = n_b
                bin_w[idx] += weight[j]
            cum = 0.0
            for m in range(n_b):
                cum += bin_w[m]
                if w_total > 0.0:
                    eps[s, k, m] = cum / w_total
    return eps


@numba.njit(parallel=True)
def pass_fractions_binned_multi(log10_lim, a_query, kappa, log10_flux, log10_b_pop,
                                 weight, bin_of_pop, b_grid):
    """As `pass_fractions_binned`, but for `n_w` classes sharing ONE
    member subsample at once: `log10_flux` (n_pop, 8) and `log10_b_pop`
    (n_pop, n_w) -- each class's own brightness column for the SAME
    members -- `weight` and `bin_of_pop` (n_pop, n_w), `b_grid` (n_b,
    n_w). Returns `(n_src, n_x, n_b, n_w)` f4. Each population member's
    8-band flux is read once per `(s, k)` and its per-class critical
    brightness computed from that one read, rather than re-reading the
    same member's flux once per class.
    """
    n_src, n_x = a_query.shape
    n_pop, n_bands = log10_flux.shape
    n_b, n_w = b_grid.shape
    eps = np.zeros((n_src, n_x, n_b, n_w), dtype=np.float32)

    bin_total = np.zeros((n_b, n_w), dtype=np.float64)
    for j in range(n_pop):
        for w in range(n_w):
            bin_total[bin_of_pop[j, w], w] += weight[j, w]

    for s in numba.prange(n_src):
        thresh = np.empty((n_x, n_bands), dtype=np.float64)
        for k in range(n_x):
            for i in range(n_bands):
                thresh[k, i] = log10_lim[s, i] + 0.4 * a_query[s, k] * kappa[s, k, i]
        for k in range(n_x):
            num = np.zeros((n_b, n_w), dtype=np.float64)
            for j in range(n_pop):
                lf_row = log10_flux[j]
                for w in range(n_w):
                    m = bin_of_pop[j, w]
                    lb = log10_b_pop[j, w]
                    smallest = np.inf
                    second = np.inf
                    for i in range(n_bands):
                        lf = lf_row[i]
                        if not np.isfinite(lf):
                            val = np.inf
                        else:
                            val = thresh[k, i] - lf + lb
                        if val < smallest:
                            second = smallest
                            smallest = val
                        elif val < second:
                            second = val
                    if second <= b_grid[m, w]:
                        num[m, w] += weight[j, w]
            for m in range(n_b):
                for w in range(n_w):
                    if bin_total[m, w] > 0.0:
                        eps[s, k, m, w] = num[m, w] / bin_total[m, w]
    return eps


@numba.njit(parallel=True)
def pass_fractions_binned_star_pahc(log10_lim, a_query, kappa, log10_flux,
                                     log10_b_star, weight_star, bin_of_star, b_grid_star,
                                     conditioned_star,
                                     log10_b_pahc, weight_pahc, bin_of_pahc, b_grid_pahc,
                                     conditioned_pahc):
    """STAR and PAHC fused onto one shared member draw and one flux read
    (as `pass_fractions_binned_multi`), but PAHC's member weight is not
    one number per member -- it is `W_j * P(q_j(s))` (SPEC_PRIORS.md
    section 4, owner 2026-09-06): `q` uses the SOURCE's own 8 micron
    limit, so it is a different number for every source `s`, and the
    caller passes it in already folded to weight, as `weight_pahc`
    `(n_src, n_pop)`. STAR keeps one weight per member, `weight_star`
    `(n_pop,)`. Because PAHC's weight varies by source, so does its
    per-bin normalising total -- computed here, per source, from
    `weight_pahc[s]` -- while STAR's is computed once, outside the
    source loop.

    `conditioned_star`/`conditioned_pahc` (n_b,) bool (owner, 2026-09-06):
    a bin with too few subsample members for the conditioned (own-bin)
    estimate to be trustworthy is instead read off the MARGINAL curve --
    every member's critical brightness compared straight to the grid
    point, `pass_curves`' form, not restricted to members whose own
    brightness lands in that bin. Both the conditioned and the marginal
    accumulators are built from the SAME per-member critical-brightness
    pass (one loop over `j`), so carrying both costs one pass, not two.
    Returns `(eps_star, eps_pahc)`, each `(n_src, n_x, n_b)` f4.
    """
    n_src, n_x = a_query.shape
    n_pop, n_bands = log10_flux.shape
    n_b = b_grid_star.shape[0]
    eps_star = np.zeros((n_src, n_x, n_b), dtype=np.float32)
    eps_pahc = np.zeros((n_src, n_x, n_b), dtype=np.float32)

    bin_total_star = np.zeros(n_b, dtype=np.float64)
    for j in range(n_pop):
        bin_total_star[bin_of_star[j]] += weight_star[j]
    w_total_star = bin_total_star.sum()

    for s in numba.prange(n_src):
        bin_total_pahc = np.zeros(n_b, dtype=np.float64)
        for j in range(n_pop):
            bin_total_pahc[bin_of_pahc[j]] += weight_pahc[s, j]
        w_total_pahc = bin_total_pahc.sum()

        thresh = np.empty((n_x, n_bands), dtype=np.float64)
        for k in range(n_x):
            for i in range(n_bands):
                thresh[k, i] = log10_lim[s, i] + 0.4 * a_query[s, k] * kappa[s, k, i]

        for k in range(n_x):
            num_star = np.zeros(n_b, dtype=np.float64)
            num_pahc = np.zeros(n_b, dtype=np.float64)
            hist_star = np.zeros(n_b + 1, dtype=np.float64)
            hist_pahc = np.zeros(n_b + 1, dtype=np.float64)
            for j in range(n_pop):
                smallest_star, second_star = np.inf, np.inf
                smallest_pahc, second_pahc = np.inf, np.inf
                lb_star = log10_b_star[j]
                lb_pahc = log10_b_pahc[j]
                for i in range(n_bands):
                    lf = log10_flux[j, i]
                    if not np.isfinite(lf):
                        val_star = np.inf
                        val_pahc = np.inf
                    else:
                        base = thresh[k, i] - lf
                        val_star = base + lb_star
                        val_pahc = base + lb_pahc
                    if val_star < smallest_star:
                        second_star = smallest_star
                        smallest_star = val_star
                    elif val_star < second_star:
                        second_star = val_star
                    if val_pahc < smallest_pahc:
                        second_pahc = smallest_pahc
                        smallest_pahc = val_pahc
                    elif val_pahc < second_pahc:
                        second_pahc = val_pahc

                w_star_j = weight_star[j]
                m_star = bin_of_star[j]
                if second_star <= b_grid_star[m_star]:
                    num_star[m_star] += w_star_j
                idx_star = np.searchsorted(b_grid_star, second_star)
                if idx_star > n_b:
                    idx_star = n_b
                hist_star[idx_star] += w_star_j

                w_pahc_j = weight_pahc[s, j]
                m_pahc = bin_of_pahc[j]
                if second_pahc <= b_grid_pahc[m_pahc]:
                    num_pahc[m_pahc] += w_pahc_j
                idx_pahc = np.searchsorted(b_grid_pahc, second_pahc)
                if idx_pahc > n_b:
                    idx_pahc = n_b
                hist_pahc[idx_pahc] += w_pahc_j

            cum_star = 0.0
            cum_pahc = 0.0
            for m in range(n_b):
                cum_star += hist_star[m]
                cum_pahc += hist_pahc[m]
                if conditioned_star[m]:
                    if bin_total_star[m] > 0.0:
                        eps_star[s, k, m] = num_star[m] / bin_total_star[m]
                elif w_total_star > 0.0:
                    eps_star[s, k, m] = cum_star / w_total_star
                if conditioned_pahc[m]:
                    if bin_total_pahc[m] > 0.0:
                        eps_pahc[s, k, m] = num_pahc[m] / bin_total_pahc[m]
                elif w_total_pahc > 0.0:
                    eps_pahc[s, k, m] = cum_pahc / w_total_pahc
    return eps_star, eps_pahc


@numba.njit(parallel=True)
def pass_fractions_binned(log10_lim, a_query, kappa, log10_flux, log10_b_pop, weight,
                           b_grid, bin_of_pop):
    """As `pass_curves`, but each population member belongs to one
    brightness bin `bin_of_pop[j]` (int, `0..n_b-1`) and is counted only
    there: `eps[s,k,m]` is the weight-fraction, within bin `m` alone, of
    members whose own `bcrit <= b_grid[m]`.
    """
    n_src, n_x = a_query.shape
    n_b = b_grid.shape[0]
    n_pop, n_bands = log10_flux.shape
    eps = np.zeros((n_src, n_x, n_b), dtype=np.float32)

    bin_total = np.zeros(n_b, dtype=np.float64)
    for j in range(n_pop):
        bin_total[bin_of_pop[j]] += weight[j]

    for s in numba.prange(n_src):
        thresh = np.empty((n_x, n_bands), dtype=np.float64)
        for k in range(n_x):
            for i in range(n_bands):
                thresh[k, i] = log10_lim[s, i] + 0.4 * a_query[s, k] * kappa[s, k, i]
        for k in range(n_x):
            num = np.zeros(n_b, dtype=np.float64)
            for j in range(n_pop):
                m = bin_of_pop[j]
                smallest = np.inf
                second = np.inf
                for i in range(n_bands):
                    lf = log10_flux[j, i]
                    if not np.isfinite(lf):
                        val = np.inf
                    else:
                        val = thresh[k, i] - lf + log10_b_pop[j]
                    if val < smallest:
                        second = smallest
                        smallest = val
                    elif val < second:
                        second = val
                if second <= b_grid[m]:
                    num[m] += weight[j]
            for m in range(n_b):
                if bin_total[m] > 0.0:
                    eps[s, k, m] = num[m] / bin_total[m]
    return eps
