"""
corrections.py
====================================================================
Bias and skew of the flux -> magnitude transform, for the 13 rows that
must be evaluated in magnitude space.

The other 32 rows are handled exactly in flux space (prob.py sections
4.1/4.2) and never touch this module.

A symmetric flux error maps to a SKEWED magnitude error, because
m = -2.5 log10(F/F0) compresses upward fluctuations and stretches
downward ones.  Writing F = F_true (1 + r eps) with eps ~ N(0,1) and
r = sigma_F / F:

    m_meas - m_true = -K ln(1 + r eps),        K = 2.5 / ln 10

which depends ONLY on r -- the absolute flux and the zero point both
cancel.  So the first three moments are functions of a single scalar
and are tabulated once, by quadrature, at import.

    mu(r)     E[m_meas - m_true]      bias, magnitudes
    sd(r)     sd[...]                 1 sigma, magnitudes
    gamma(r)  skewness[...]

Do NOT use the leading-order series (mu ~ 0.543 r^2, sd ~ 1.0857 r,
gamma ~ 3r).  They are good to r ~ 0.1 but drift to 6%, 6% and 22% by
r = 0.2 -- which is exactly Gutermuth's admission threshold, where the
near-boundary sources live.

The integrand is undefined for 1 + r eps <= 0, i.e. a flux draw that
went non-positive.  That is not a magnitude at all, so the moments are
computed CONDITIONAL on F > 0 and the truncated mass is reported by
`truncated_mass(r)`.  For r <= 0.2 it is below 3e-7.
====================================================================
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

K_MAG = 2.5 / np.log(10.0)

# Grid over the fractional flux error.  Routing admits sigma_m < 0.2,
# i.e. r <~ 0.184; the grid runs past that so the edge is never hit by
# interpolation, and stops at 0.30 where the truncated mass starts to
# matter and the moments lose meaning.
R_GRID = np.concatenate([
    np.linspace(0.0, 0.05, 26)[:-1],
    np.linspace(0.05, 0.30, 51),
])
R_MAX = R_GRID[-1]


def _raw_moments(r, n_max=3):
    """E[L^n] for L = -K ln(1 + r eps), conditional on 1 + r eps > 0."""
    if r <= 0:
        return np.zeros(n_max + 1)

    lo = -1.0 / r
    pdf = lambda e: np.exp(-0.5 * e * e) / np.sqrt(2.0 * np.pi)  # noqa: E731

    def integrand(e, n):
        return (-K_MAG * np.log1p(r * e)) ** n * pdf(e)

    out = np.empty(n_max + 1)
    # mass surviving the truncation, used to renormalise
    out[0] = quad(pdf, lo, 12.0, limit=200)[0]
    for n in range(1, n_max + 1):
        # split at the singular end so quad resolves the log
        a = quad(integrand, lo, lo + abs(lo) * 1e-3, args=(n,), limit=200)[0]
        b = quad(integrand, lo + abs(lo) * 1e-3, 12.0, args=(n,), limit=200)[0]
        out[n] = a + b
    return out


def _build_tables():
    mu = np.zeros_like(R_GRID)
    sd = np.zeros_like(R_GRID)
    gamma = np.zeros_like(R_GRID)
    mass = np.ones_like(R_GRID)

    for i, r in enumerate(R_GRID):
        if r <= 0:
            continue
        m = _raw_moments(r)
        mass[i] = m[0]
        m1, m2, m3 = m[1] / m[0], m[2] / m[0], m[3] / m[0]
        var = m2 - m1 ** 2
        mu[i] = m1
        sd[i] = np.sqrt(max(var, 0.0))
        # third central moment
        c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        gamma[i] = c3 / sd[i] ** 3 if sd[i] > 0 else 0.0
    return mu, sd, gamma, mass


_MU, _SD, _GAMMA, _MASS = _build_tables()


def moments(r):
    """(mu, sd, gamma) for a fractional flux error r, interpolated."""
    r = np.clip(np.abs(np.asarray(r, dtype=float)), 0.0, R_MAX)
    return (np.interp(r, R_GRID, _MU),
            np.interp(r, R_GRID, _SD),
            np.interp(r, R_GRID, _GAMMA))


def truncated_mass(r):
    """P(flux draw <= 0), the mass excluded from the moments above."""
    r = np.clip(np.abs(np.asarray(r, dtype=float)), 0.0, R_MAX)
    return 1.0 - np.interp(r, R_GRID, _MASS)


def cornish_fisher(z, gamma, clip=0.5):
    """P(margin > 0) from a standardised margin z and a skewness gamma.

    Third-order Cornish-Fisher.  Setting gamma = 0 recovers Phi(z)
    exactly, so this is a strict refinement of the uncorrected path.

    The expansion is asymptotic and misbehaves once |gamma (z^2-1)/6|
    grows, producing non-monotone probabilities.  The shift is therefore
    clipped: `clip` bounds how far the argument may move, in units of z.
    At the values that actually occur (gamma <~ 0.8, |z| <~ 4) the clip
    is inactive; it exists so a pathological input degrades to Phi(z)
    rather than to nonsense.
    """
    z = np.asarray(z, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    shift = (gamma / 6.0) * (z ** 2 - 1.0)
    shift = np.clip(shift, -clip * np.abs(z) - clip, clip * np.abs(z) + clip)
    return ndtr(z + shift)


def band_corrections(flux, sigma):
    """Per-band (r, mu, sd, gamma) for an (N, 8) source array.

    Computed once per source and reused by every magnitude-space row --
    the correction is a property of the band's S/N, not of the row.
    """
    flux = np.asarray(flux, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(flux > 0, np.abs(sigma / flux), 0.0)
    r = np.where(np.isfinite(r), r, 0.0)
    mu, sd, gamma = moments(r)
    return r, mu, sd, gamma
