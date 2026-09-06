"""Whether a source clears SESNA's own catalog cut, the hybrid
extinction law the test dims by, and the two compiled kernels that turn
a class's external population into a source's exact pass fraction
(SPEC_PRIORS.md section 1.3, IMPLEMENTATION.md section 4).

SESNA catalogs a source when at least two of its eight bands, dimmed by
its own dust column, clear the source's detection limit
(`catalog.limits.limits`). The dimming coefficient per band is not one
fixed law: it blends smoothly, in log column, from a diffuse-ISM curve
below A_K = 0.5 to a dense-cloud curve above A_K = 1.0
(`law_dense_weight`, `kappa_hybrid`). A class's selection is evaluated
exactly, per source, at the source's own eight limits: `pass_curves` and
`pass_fractions_binned` walk a class's whole external population once
per source and per query extinction, reducing the two-of-eight test to
one critical brightness per population member and reading the pass
fraction off as a weighted cumulative count.
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
#: tabulated on: `x = a / A_s`. The last three points cover the column
#: kernel's tail past the source's own column.
X_LADDER = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.4, 2.0, 2.8])

#: A class's own brightness grid: this many points.
N_B_GRID = 24

#: The fixed-seed population subsample cap: the Monte Carlo error on a
#: pass fraction near 0.5 at this many draws is 0.5%.
SUBSAMPLE_CAP = 10_000


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
    every source and query extinction, against the whole population.

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
                        val = (log10_lim[s, i]
                               - (lf - 0.4 * a_query[s, k] * kappa[s, k, i] - log10_b_pop[j]))
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
                        val = (log10_lim[s, i]
                               - (lf - 0.4 * a_query[s, k] * kappa[s, k, i] - log10_b_pop[j]))
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
