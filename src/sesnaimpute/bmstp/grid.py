"""The common grid every class's shape lives on (SPEC_BMSTP_DRAFT.md sec. 2),
the weighted binning of a population sample onto it, and the per-source
column-kernel blur along the `log10 x` axis (sec. 2, 4.2).

`log10 x = log10(a / A_s)` runs -3.0 to +1.0 in 128 cells of 1/32 dex, one
axis shared by every class; the brightness axis is 120 cells of 0.1 dex,
its origin a per-shape attribute. A shape's mass outside its grid is the
`mass_outside` `bin` reports; its value in an empty or off-grid cell is
`FLOOR` of its peak cell (sec. 2, "the floor"). At bin time every shape is
smoothed by one cell along `log10 x` and, along the brightness axis, by
the larger of one cell and the region's distance uncertainty in
`-2 log10 d` (sec. 2, "minimum widths"): no shape is a delta narrower than
the fit's own uncertainty in `log10 B_hat`.
"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

#: `log10 x` cell edges, -3.0 to +1.0 in 128 cells of 1/32 dex (sec. 2).
LOG10_X_EDGES = np.linspace(-3.0, 1.0, 129)
_N_X = LOG10_X_EDGES.size - 1
_X_CELL_WIDTH = (LOG10_X_EDGES[-1] - LOG10_X_EDGES[0]) / _N_X
_X_CENTERS = LOG10_X_EDGES[:-1] + 0.5 * _X_CELL_WIDTH

#: The brightness axis: 120 cells of 0.1 dex, origin per shape (sec. 2).
N_B = 120
D_LOG10_B = 0.1

#: The template-unit brightness origin for the star family and YSO
#: (`log10 B` from -6.0 to +6.0 over 120 cells of 0.1 dex, sec. 2).
LOG10_B_ORIGIN_TEMPLATE = -6.0

#: A shape's value in an empty or off-grid cell, as a fraction of its
#: peak cell (sec. 2, "the floor").
FLOOR = 1e-6

_SQRT2 = float(np.sqrt(2.0))


def log10_b_edges(origin):
    """The `(N_B + 1,)` brightness-axis cell edges for a shape whose
    origin is `origin` (sec. 2): `N_B` cells of `D_LOG10_B` dex."""
    return origin + D_LOG10_B * np.arange(N_B + 1)


def bin(x, log10_b, w, origin, sigma_b_min):
    """The weighted 2-D histogram `(H, mass_outside)` of a population
    sample on the common grid (sec. 2): `x` the sample's scaled
    extinction, `log10_b` its brightness, `w` its weight, `origin` the
    shape's brightness-axis origin, `sigma_b_min` the region's distance
    uncertainty in `-2 log10 d` (dex) that floors the brightness-axis
    smoothing width. `H` is normalised to sum to `1 - mass_outside` over
    cells (the mass that fell outside the grid is not in `H`), then
    smoothed by one cell along `log10 x` and by
    `max(D_LOG10_B, sigma_b_min)` along `log10 B` (sec. 2, "minimum
    widths"), then floored at `FLOOR * H.max()`."""
    x = np.asarray(x, dtype=float)
    log10_b = np.asarray(log10_b, dtype=float)
    w = np.asarray(w, dtype=float)
    total_weight = w.sum()
    with np.errstate(divide="ignore"):
        log10_x = np.log10(x)
    b_edges = log10_b_edges(origin)
    H, _, _ = np.histogram2d(
        log10_x, log10_b, bins=[LOG10_X_EDGES, b_edges], weights=w
    )
    mass_outside = float((total_weight - H.sum()) / total_weight)
    H = H / total_weight
    # `mode="wrap"` is a circular convolution: no mass crosses the array
    # boundary undetected, so the normalisation above is exactly
    # preserved by the smoothing step (the sample is far from the grid's
    # own edges by construction of the class populations, sec. 2).
    H = gaussian_filter1d(H, sigma=1.0, axis=0, mode="wrap")
    sigma_b_cells = max(D_LOG10_B, float(sigma_b_min)) / D_LOG10_B
    H = gaussian_filter1d(H, sigma=sigma_b_cells, axis=1, mode="wrap")
    H = np.maximum(H, FLOOR * H.max())
    return H.astype(np.float64), mass_outside


def _shift_kernel(mu, sigma):
    """The `(128, 128)` linear operator `K` shifting the `log10 x` axis
    by `mu` and Gaussian-smoothing by `sigma` (floored at one cell, sec.
    2): `K[i, j]` is the mass landing in destination cell `i` of a point
    at source cell `j`'s center shifted by `mu`, from the shifted
    Gaussian's CDF differenced over cell `i`'s edges -- exact for any
    real-valued `mu`, not restricted to a whole number of cells. On the
    uniform grid `K[i, j]` depends only on the offset `i - j`, so it is
    Toeplitz: the CDF is evaluated once along the single offset axis
    (`2 * _N_X` points, not the full `(129, 128)` edge-by-center table)
    and the matrix is formed by `scipy.linalg.toeplitz`."""
    sigma_eff = max(float(sigma), _X_CELL_WIDTH)
    c = -0.5 * _X_CELL_WIDTH - mu
    d = np.arange(-(_N_X - 1), _N_X + 1)
    cdf = 0.5 * (1.0 + erf((c + d * _X_CELL_WIDTH) / (sigma_eff * _SQRT2)))
    offsets = cdf[1:] - cdf[:-1]  # offsets[m] = K[i, j] at i - j = m - (_N_X - 1)
    mid = _N_X - 1
    col = offsets[mid:mid + _N_X]  # i - j = 0 .. _N_X - 1
    row = offsets[mid::-1]  # i - j = 0, -1, .. -(_N_X - 1)
    return toeplitz(col, row)


def blur(H, w, mu1, sig1, mu2, sig2):
    """`(H_s, mass_lost)`: one source's column kernel applied along the
    `log10 x` axis (sec. 2, 4.2), `H_s = (w * K1 + (1 - w) * K2) @ H`,
    `K1`, `K2` the shift-and-smooth operators of `_shift_kernel` for the
    mixture's two components, combined before the one matrix product.
    `mass_lost` is the mass the shift carries past the `log10 x` edges
    (`H.sum() - H_s.sum()`)."""
    K1 = _shift_kernel(mu1, sig1)
    K2 = _shift_kernel(mu2, sig2)
    K = w * K1 + (1.0 - w) * K2
    H_s = K @ H
    mass_lost = float(H.sum() - H_s.sum())
    return H_s, mass_lost
