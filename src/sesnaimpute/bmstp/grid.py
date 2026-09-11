"""The common grid every class's shape lives on (SPEC_BMSTP_DRAFT.md sec. 2,
"the common grid"), the weighted binning of a population sample onto it, and
the per-source column-kernel blur along the `log10 x` axis (sec. 2, 4.2).

`log10 x = log10(a / A_s)` runs -3.0 to +1.0 in 128 cells of 1/32 dex; the
brightness axis, `log10 F_4.5` (the object's dereddened 4.5 micron flux in
mJy, "4.5B", the owner's ruling 2026-09-08), runs -4.0 to +7.0 in 110 cells
of 0.1 dex -- ONE origin, the same for every class (sec. 2's "brightness"
and "the common grid" rows, REWRITTEN after W24: the grid top moved to
+7.0, W24b, so that Perseus's evolved star above 1 kJy is on the grid). No
per-shape brightness origin, and no region-distance widening here -- YSO's
own distance-uncertainty widening is a Gaussian formed in
`bmstp.sample_cloud`, not a `bin`-time floor. The grid's LOWER edge is
every class's retention limit (sec. 2): mass below it is dropped from the
shape, and `1 - mass_outside` (the ON-GRID FRACTION) is what a consumer
stores beside the shape to scale the intrinsic density by (W26); the 0.1%
acceptance bar (sec. 9) applies to `mass_above_top` alone -- the
population's own raw weight above the grid's TOP edge, not to how much of
it sits below the retention limit. A shape's mass outside its grid is the
`mass_outside` `bin` reports; its value in an empty or off-grid cell is
`FLOOR` of its peak cell (sec. 2, "the floor"). At bin time every shape is
smoothed by exactly one cell along EACH axis (sec. 2, "minimum widths"): no
shape is a delta narrower than the fit's own uncertainty in `log10 B_hat`
(0.04-0.1 dex on a two-band source).
"""

import numpy as np
from scipy import ndimage
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

#: `log10 x` cell edges, -3.0 to +1.0 in 128 cells of 1/32 dex (sec. 2).
LOG10_X_EDGES = np.linspace(-3.0, 1.0, 129)
_N_X = LOG10_X_EDGES.size - 1
_X_CELL_WIDTH = (LOG10_X_EDGES[-1] - LOG10_X_EDGES[0]) / _N_X
_X_CENTERS = LOG10_X_EDGES[:-1] + 0.5 * _X_CELL_WIDTH

#: `log10 F_4.5` cell edges, -4.0 to +7.0 in 110 cells of 0.1 dex (sec. 2,
#: "the common grid", REWRITTEN after W24, W24b): one axis and one origin
#: for STAR, AGB, PAHC, GAL and YSO alike. The top moved from +6.0 to +7.0
#: so Perseus's own evolved star above 1 kJy (`sample_star.sample_agb`)
#: sits on the grid.
LOG10_F45_EDGES = np.linspace(-4.0, 7.0, 111)
_N_B = LOG10_F45_EDGES.size - 1
D_LOG10_F45 = (LOG10_F45_EDGES[-1] - LOG10_F45_EDGES[0]) / _N_B
_B_CENTERS = LOG10_F45_EDGES[:-1] + 0.5 * D_LOG10_F45

#: A shape's value in an empty or off-grid cell, as a fraction of its
#: peak cell (sec. 2, "the floor").
FLOOR = 1e-6

#: (sec. 2 "minimum widths", sec. 5.1 "Marks"): the number of
#: geometric depth-uncertainty width classes a field star's own `sigma_x`
#: (the map's propagated column sigma at the star's distance, in `log10
#: x`) is quantised to, between the one-cell floor and a tile's own
#: cloud-interval-span cap.
N_WIDTH_CLASSES = 8

_SQRT2 = float(np.sqrt(2.0))


def bin(x, log10_f45, w):
    """The weighted 2-D histogram `(H, mass_outside)` of a population
    sample on the common grid (sec. 2): `x` the sample's scaled
    extinction, `log10_f45` its dereddened 4.5 micron flux (log10 mJy),
    `w` its weight. `H` is normalised to sum to `1 - mass_outside` over
    cells (the mass that fell outside the grid is not in `H`), then
    smoothed by exactly one cell along EACH axis (sec. 2, "minimum
    widths") with `mode="constant"` (zero beyond the edges): any mass the
    smoothing pushes past an edge is mass outside the grid and is folded
    into `mass_outside`, so `H.sum() == 1 - mass_outside` stays an exact
    identity before the floor. `H` is then floored at `FLOOR * H.max()`."""
    x = np.asarray(x, dtype=float)
    log10_f45 = np.asarray(log10_f45, dtype=float)
    w = np.asarray(w, dtype=float)
    total_weight = w.sum()
    with np.errstate(divide="ignore"):
        log10_x = np.log10(x)
    # edge convention (sec. 2): a mark exactly on a `log10 x` cell edge
    # belongs to the cell below it (`x = 1`, the whole column, must sit
    # below `log10 x = 0`, never above). `np.histogram2d` bins are
    # left-inclusive/right-exclusive, so nudging every mark down by one
    # ULP (`np.nextafter`, negligible against the 1/32 dex cell width)
    # moves an exact-edge mark into the cell whose upper edge it sat on,
    # without moving any mark that is not on an edge.
    log10_x = np.nextafter(log10_x, -np.inf)
    H, _, _ = np.histogram2d(
        log10_x, log10_f45, bins=[LOG10_X_EDGES, LOG10_F45_EDGES], weights=w
    )
    mass_outside = float((total_weight - H.sum()) / total_weight)
    H = H / total_weight
    # `mode="constant"` (zero beyond the edges) on both axes: mass the
    # one-cell smoothing pushes past `log10 x = -3.0`/`+1.0`, or past
    # `log10 F_4.5 = -4.0`/`+7.0`, is mass outside the grid -- folded into
    # `mass_outside` below, not reappeared at the opposite edge.
    mass_before = float(H.sum())
    H = gaussian_filter1d(H, sigma=1.0, axis=0, mode="constant")
    H = gaussian_filter1d(H, sigma=1.0, axis=1, mode="constant")
    mass_outside += mass_before - float(H.sum())
    H = np.maximum(H, FLOOR * H.max())
    return H.astype(np.float64), mass_outside


def bin_star_widths(x, log10_f45, w, width_class, sigma_classes_cells):
    """STAR/AGB's own per-class depth-uncertainty smoothing (sec. 2
    "minimum widths", sec. 5.1 "Marks"): like `bin`, but the `log10 x`
    axis is smoothed by each star's OWN width class instead of the fixed
    one-cell floor -- the field-star depth mark carries the released
    posterior samples' own spread of the cumulative column at the star's
    distance (`sky.derived.edenhofer_samples`' `SIGMA_SAMPLES_K`), not a
    delta narrower than the map itself resolves. `sigma_classes_cells`
    (`N_WIDTH_CLASSES`,), in grid cells (>= 1.0, the floor, geometric to
    the sightline's own cloud-interval-span cap, `sample_star
    .tile_width_classes`); `width_class` (n_star,) each star's own class
    index, nearest its `sigma_x` in log space. One weighted histogram and
    smoothing pass per POPULATED class, summed before the floor -- up to
    `N_WIDTH_CLASSES` histograms and smoothings of the tile's grid in
    place of one, no per-star kernel, nothing at the read (AGB reuses the
    same stars' classes, sec. 5.2). The `log10 F_4.5` axis keeps the
    ordinary one-cell smoothing (sec. 2) in every pass. `H.sum() == 1 -
    mass_outside` stays exact before the floor, the same identity `bin`
    reports, since every star belongs to exactly one class."""
    x = np.asarray(x, dtype=float)
    log10_f45 = np.asarray(log10_f45, dtype=float)
    w = np.asarray(w, dtype=float)
    width_class = np.asarray(width_class, dtype=np.int64)
    total_weight = w.sum()
    if total_weight <= 0:
        return np.full((_N_X, _N_B), FLOOR), 1.0
    with np.errstate(divide="ignore"):
        log10_x = np.log10(x)
    # edge convention (sec. 2), the same nudge `bin` applies.
    log10_x = np.nextafter(log10_x, -np.inf)
    H = np.zeros((_N_X, _N_B), dtype=np.float64)
    for k in range(len(sigma_classes_cells)):
        sel = width_class == k
        if not np.any(sel):
            continue
        Hk, _, _ = np.histogram2d(
            log10_x[sel], log10_f45[sel], bins=[LOG10_X_EDGES, LOG10_F45_EDGES], weights=w[sel])
        Hk = Hk / total_weight
        Hk = gaussian_filter1d(Hk, sigma=float(sigma_classes_cells[k]), axis=0, mode="constant")
        Hk = gaussian_filter1d(Hk, sigma=1.0, axis=1, mode="constant")
        H += Hk
    mass_outside = float(1.0 - H.sum())
    H = np.maximum(H, FLOOR * H.max()) if H.max() > 0 else np.full_like(H, FLOOR)
    return H.astype(np.float64), mass_outside


def _explicit_gaussian_kernel1d(sigma, truncate=4.0):
    """The exact discrete kernel `scipy.ndimage.gaussian_filter1d` applies
    for `sigma` (its own default truncation, order 0), built explicitly so
    a caller can SQUARE it (repair list row 7, coordinator's addendum
    2026-09-10 19:45: `N_EFF`'s `S2` runs the squared-weight sum through
    every kernel the density's own chain uses, squared elementwise and NOT
    renormalised -- `gaussian_filter1d` itself always renormalises, so it
    cannot be reused for `S2`)."""
    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    phi = np.exp(-0.5 * (offsets / sigma) ** 2)
    return phi / phi.sum()


def n_eff_star_widths(x, log10_f45, w, width_class, sigma_classes_cells):
    """The per-cell EFFECTIVE COUNT of retained stars behind `bin_star_
    widths`'s own density at that cell (repair list row 7, coordinator's
    addendum 2026-09-10 19:45): a cell's density is `Sum_i w_i K_i(c)`,
    each star `i` placed through its own width class's Gaussian (`log10
    x` axis) and the grid's one-cell smoothing (`log10 F_4.5` axis); the
    evidence behind it is the SAME stars through the SAME two kernels, so
    `S1 = Sum_i w_i K_i(c)` is `bin_star_widths`'s own construction exactly
    (summed over classes, before the floor), and `S2 = Sum_i w_i**2
    K_i(c)**2` reruns it with each star's weight squared and BOTH kernels
    replaced by their own elementwise square, NOT renormalised (a squared
    Gaussian is narrower and smaller: it may not integrate to 1).  `N_EFF =
    S1**2 / S2`, 0 where no star's weight reaches the cell (`S2 == 0`).
    Invariant to any overall weight scale, so the `total_weight` division
    `bin_star_widths` applies to its own density does not matter here."""
    x = np.asarray(x, dtype=np.float64)
    log10_f45 = np.asarray(log10_f45, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    width_class = np.asarray(width_class, dtype=np.int64)
    with np.errstate(divide="ignore"):
        log10_x = np.nextafter(np.log10(x), -np.inf)
    f_kernel = _explicit_gaussian_kernel1d(1.0)
    f_kernel_sq = f_kernel ** 2
    s1 = np.zeros((_N_X, _N_B), dtype=np.float64)
    s2 = np.zeros((_N_X, _N_B), dtype=np.float64)
    for k in range(len(sigma_classes_cells)):
        sel = width_class == k
        if not np.any(sel):
            continue
        h1, _, _ = np.histogram2d(
            log10_x[sel], log10_f45[sel], bins=[LOG10_X_EDGES, LOG10_F45_EDGES], weights=w[sel])
        h2, _, _ = np.histogram2d(
            log10_x[sel], log10_f45[sel], bins=[LOG10_X_EDGES, LOG10_F45_EDGES], weights=w[sel] ** 2)
        x_kernel = _explicit_gaussian_kernel1d(float(sigma_classes_cells[k]))
        x_kernel_sq = x_kernel ** 2
        h1 = ndimage.convolve1d(h1, weights=x_kernel, axis=0, mode="constant")
        h1 = ndimage.convolve1d(h1, weights=f_kernel, axis=1, mode="constant")
        h2 = ndimage.convolve1d(h2, weights=x_kernel_sq, axis=0, mode="constant")
        h2 = ndimage.convolve1d(h2, weights=f_kernel_sq, axis=1, mode="constant")
        s1 += h1
        s2 += h2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s2 > 0, s1 ** 2 / s2, 0.0)


def n_eff_analytic(x, log10_f45, w):
    """GAL's own `N_EFF` (repair list row 7): GAL's shape is an analytic
    counts law, not a finite draw of members whose count could run short,
    so the sampling-uncertainty identity above does not apply -- every
    cell the law's own raw (pre-smoothing) mass touches carries INFINITE
    effective evidence (the law's disclosed systematic band, not a sample
    size, is its uncertainty, sec. 5.4), and every other cell carries
    none."""
    x = np.asarray(x, dtype=np.float64)
    log10_f45 = np.asarray(log10_f45, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log10_x = np.nextafter(np.log10(x), -np.inf)
    raw, _, _ = np.histogram2d(
        log10_x, log10_f45, bins=[LOG10_X_EDGES, LOG10_F45_EDGES], weights=w)
    return np.where(raw > 0, np.inf, 0.0)


def mass_above_top(log10_f45, w):
    """The fraction of a population sample's own weight whose brightness
    mark lies above the grid's TOP edge (sec. 2, "the common grid": the
    0.1% acceptance bar, sec. 9, is on this alone -- never on the mass
    the retention limit at the BOTTOM edge drops, which for YSO is most
    of the population by design, W24). Computed directly on the raw,
    unbinned sample, so it is unaffected by `bin`'s own one-cell edge
    smoothing."""
    log10_f45 = np.asarray(log10_f45, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    if total <= 0.0:
        return 0.0
    return float(w[log10_f45 > LOG10_F45_EDGES[-1]].sum() / total)


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
