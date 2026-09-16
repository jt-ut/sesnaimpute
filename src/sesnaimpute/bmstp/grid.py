"""The common grid every class's shape lives on (SPEC_BMSTP_DRAFT.md sec. 2,
"the common grid"), the weighted binning of a population sample onto it, and
the per-source column-kernel blur along the `log10 ξ` axis (sec. 2, 4.2).

`log10 ξ = log10(a / T)` (the distance coordinate: the object's foreground over the true column along its own pencil beam) runs -3.0 to +1.0 in 128 cells of 1/32 dex; the
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
it sits below the retention limit. At bin time every shape is smoothed by
exactly one cell along EACH axis (sec. 2, "minimum widths"): no shape is a
delta narrower than the fit's own uncertainty in `log10 B_hat` (0.04-0.1
dex on a two-band source).

Two more statements beside sec. 2's own: THE SUPPORT. `x = a / T <= 1`
by definition, so a cell with `log10 ξ > 0` is never part of the prior's
support even though the array keeps that extent (`N_XI_SUPPORT`, below)
for the kernels' own padding, and `ON_GRID_*`/`MASS_OUTSIDE_*` are
measured over the support alone. THE EDGE ξ = 1. `log10 ξ = 0` is a physical
wall, not a grid edge: no population member is ever above it, so no
placement or smoothing along `x` may carry mass past it and drop that
mass as outside the prior -- `bin`, `bin_star_widths`,
`sample_cloud._bin1d` and the YSO shift-kernel placement in
`bmstp.shapes` instead REFLECT (`blur`, the read's change of variables
into the measured coordinate, does not: see its own docstring): whatever mass a kernel
would place at `log10 ξ > 0` is folded back onto its mirror cell below
the edge ξ = 1 (`fold_wall`, below), conserving the kernel's own mass on the
support exactly, so a population sitting AT the edge ξ = 1 (every galaxy, most
background stars) reads `ON_GRID = 1` to float precision instead of
losing half its mass to a phantom edge. The array's own low and top
`log10 ξ` edges (`-3.0`, `+1.0`) keep the ordinary edge rule -- mass a
kernel pushes past THOSE is truly outside and is counted in
`mass_outside`, since `+1.0` is padding sized for the edge ξ = 1's own
reflection, one dex, never reached by the kernels this design uses. THE
FLOOR IS COMMON. `FLOOR` is one absolute value on the read prior density,
common to every class at a source, applied where the prior is read
(`fittp.prior_reader`), not a per-class, per-shape constant baked in
here: a stored shape carries its own true zeros, so an empty cell reads
equal across the six classes and the likelihood alone decides it.
`blur`'s own shift-and-smooth carries no floor step; its caller applies
the common one, at the read.
"""

import numpy as np
from scipy import ndimage
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

#: `log10 ξ` cell edges, -3.0 to +1.0 in 128 cells of 1/32 dex (sec. 2).
LOG10_XI_EDGES = np.linspace(-3.0, 1.0, 129)
_N_X = LOG10_XI_EDGES.size - 1
_X_CELL_WIDTH = (LOG10_XI_EDGES[-1] - LOG10_XI_EDGES[0]) / _N_X
_X_CENTERS = LOG10_XI_EDGES[:-1] + 0.5 * _X_CELL_WIDTH

#: the support rule (module docstring): `x <= 1` by definition, so only
#: the cells whose upper edge is at or below `log10 ξ = 0` are the
#: prior's support -- the one dex above it (cells `N_XI_SUPPORT` to
#: `_N_X - 1`) is kept in every array purely as the kernels' own padding
#: and is never part of a shape's stored mass or a reader's sum.
N_XI_SUPPORT = int(np.searchsorted(LOG10_XI_EDGES, 0.0))

#: `log10 F_4.5` cell edges, -4.0 to +7.0 in 110 cells of 0.1 dex (sec. 2,
#: "the common grid", REWRITTEN after W24, W24b): one axis and one origin
#: for STAR, AGB, PAHC, GAL and YSO alike. The top moved from +6.0 to +7.0
#: so Perseus's own evolved star above 1 kJy (`sample_star.sample_agb`)
#: sits on the grid.
LOG10_F45_EDGES = np.linspace(-4.0, 7.0, 111)
_N_B = LOG10_F45_EDGES.size - 1
D_LOG10_F45 = (LOG10_F45_EDGES[-1] - LOG10_F45_EDGES[0]) / _N_B
_B_CENTERS = LOG10_F45_EDGES[:-1] + 0.5 * D_LOG10_F45

#: The floor fraction (sec. 2, "the floor"; module docstring's common-floor
#: statement): not baked into a stored shape, which carries its own true
#: zeros -- read alone, as one value common to every class at a source,
#: `FLOOR` times the largest cell density any of the six classes reaches
#: there (`fittp.prior_reader.common_floor`).
FLOOR = 1e-6

#: (sec. 2 "minimum widths", sec. 5.1 "Marks"): the number of
#: geometric depth-uncertainty width classes a field star's own `sigma_x`
#: (the map's propagated column sigma at the star's distance, in `log10
#: x`) is quantised to, between the one-cell floor and a tile's own
#: cloud-interval-span cap.
N_WIDTH_CLASSES = 8

_SQRT2 = float(np.sqrt(2.0))

#: The `truncate` rule for every `gaussian_filter1d` call in this module
#: (TRUNCATE brief), replacing scipy's default of 4.0 sigma: every grid
#: this module fills is stored as `float32` (`bmstp/shapes.py`'s
#: `GRID_STAR`/`GRID_AGB`/`GRID_YSO`/`GRID_H2S`/`GRID` all
#: `.astype(np.float32)`), whose smallest normal magnitude is ~1.18e-38.
#: A unit Gaussian's peak-relative value first falls below that at the
#: radius `r` solving `exp(-r**2 / 2) = 1.18e-38`, i.e. `r =
#: sqrt(-2 * ln(1.18e-38)) ~= 13.2` sigma; a few more sigma past that
#: (15) reaches ~4e-49, already below `float32`'s smallest SUBNORMAL
#: (1.4e-45), so nothing beyond 15 sigma would survive the cast to
#: `float32` regardless of how much further the kernel ran. 15 sigma is
#: therefore the point past which a wider kernel buys nothing the stored
#: dtype can hold. The other stopping condition is the array's own
#: extent along the smoothed axis: running the kernel past the array
#: costs compute for values `mode="constant"` zero-padding already
#: supplies. `_truncate_for` returns, for a given `sigma` (in cells) and
#: axis length `n_cells`, the `truncate` (scipy's sigma-multiplier
#: convention) that reaches whichever of those two radii is smaller.
def _truncate_for(sigma, n_cells):
    """`truncate` for `gaussian_filter1d`/`gaussian_filter` reaching
    `float32`'s storable floor (15 sigma) or the smoothed axis's own
    extent (`n_cells`), whichever radius is smaller -- see the module
    constant block above this function for the dtype reasoning."""
    sigma = float(sigma)
    if sigma <= 0.0:
        return 15.0
    return min(15.0, n_cells / sigma)


def fold_wall(H):
    """THE EDGE ξ = 1 (module docstring): mirror-reflects whatever mass sits in
    the padding above the support (`H[N_XI_SUPPORT:]`, axis 0) back onto
    the support about `log10 ξ = 0`, and holds the padding at exact zero.
    Cell `N_XI_SUPPORT + k`'s mass lands in cell `N_XI_SUPPORT - 1 - k`, its
    mirror image across the edge ξ = 1 on the uniform grid -- the mapping every
    `x`-direction placement or smoothing step in this module uses so a
    kernel's mass is conserved on the support rather than dropped.
    Operates along axis 0 in place and returns `H`, whatever its trailing
    shape (a bare `(n_x,)` marginal or an `(n_x, n_b)` joint grid)."""
    n_pad = _N_X - N_XI_SUPPORT
    H[N_XI_SUPPORT - n_pad:N_XI_SUPPORT] += H[N_XI_SUPPORT:][::-1]
    H[N_XI_SUPPORT:] = 0.0
    return H


def bin(x, log10_f45, w):
    """The weighted 2-D histogram `(H, mass_outside)` of a population
    sample on the common grid (sec. 2): `x` the sample's scaled
    extinction, `log10_f45` its dereddened 4.5 micron flux (log10 mJy),
    `w` its weight. `H` is normalised to sum to `1 - mass_outside` over
    cells (the mass that fell outside the grid is not in `H`), then
    smoothed by exactly one cell along EACH axis (sec. 2, "minimum
    widths") with `mode="constant"` (zero beyond the array's own edges):
    any mass the smoothing pushes past the array's true `log10 ξ`/`log10
    F_4.5` edges is mass outside the grid and is folded into
    `mass_outside`. THE EDGE ξ = 1 (module docstring) then reflects whatever of
    that smoothed mass sits at `log10 ξ > 0` back onto the support
    (`fold_wall`) instead of dropping it, so `H.sum() == 1 - mass_outside`
    stays an exact identity over the support alone. `H` carries its own
    true zeros: the floor is read, not stored."""
    x = np.asarray(x, dtype=float)
    log10_f45 = np.asarray(log10_f45, dtype=float)
    w = np.asarray(w, dtype=float)
    total_weight = w.sum()
    with np.errstate(divide="ignore"):
        log10_xi = np.log10(x)
    # edge convention (sec. 2): a mark exactly on a `log10 ξ` cell edge
    # belongs to the cell below it (`x = 1`, the whole column, must sit
    # below `log10 ξ = 0`, never above). `np.histogram2d` bins are
    # left-inclusive/right-exclusive, so nudging every mark down by one
    # ULP (`np.nextafter`, negligible against the 1/32 dex cell width)
    # moves an exact-edge mark into the cell whose upper edge it sat on,
    # without moving any mark that is not on an edge.
    log10_xi = np.nextafter(log10_xi, -np.inf)
    H, _, _ = np.histogram2d(
        log10_xi, log10_f45, bins=[LOG10_XI_EDGES, LOG10_F45_EDGES], weights=w
    )
    mass_outside = float((total_weight - H.sum()) / total_weight)
    H = H / total_weight
    # `mode="constant"` (zero beyond the edges) on both axes: mass the
    # one-cell smoothing pushes past `log10 ξ = -3.0`/`+1.0`, or past
    # `log10 F_4.5 = -4.0`/`+7.0`, is mass outside the grid -- folded into
    # `mass_outside` below, not reappeared at the opposite edge.
    mass_before = float(H.sum())
    H = gaussian_filter1d(H, sigma=1.0, axis=0, mode="constant",
                           truncate=_truncate_for(1.0, _N_X))
    H = gaussian_filter1d(H, sigma=1.0, axis=1, mode="constant",
                           truncate=_truncate_for(1.0, _N_B))
    mass_outside += mass_before - float(H.sum())
    # THE EDGE ξ = 1: `log10 ξ = 0` reflects, it does not drop -- the one-cell
    # x smoothing's own mass above it is folded back onto the support,
    # never counted in `mass_outside`.
    H = fold_wall(H)
    return H.astype(np.float64), mass_outside


def bin_star_widths(x, log10_f45, w, width_class, sigma_classes_cells):
    """STAR/AGB's own per-class depth-uncertainty smoothing (sec. 2
    "minimum widths", sec. 5.1 "Marks"): like `bin`, but the `log10 ξ`
    axis is smoothed by each star's OWN width class instead of the fixed
    one-cell floor -- the field-star depth mark carries the released
    posterior samples' own spread of the cumulative column at the star's
    distance (`sky.derived.edenhofer_samples`' `SIGMA_SAMPLES_K`), not a
    delta narrower than the map itself resolves. `sigma_classes_cells`
    (`N_WIDTH_CLASSES`,), in grid cells (>= 1.0, the floor, geometric to
    the sightline's own cloud-interval-span cap, `sample_star
    .tile_width_classes`); `width_class` (n_star,) each star's own class
    index, nearest its `sigma_x` in log space. One weighted histogram and
    smoothing pass per POPULATED class, summed before THE EDGE ξ = 1 fold --
    up to `N_WIDTH_CLASSES` histograms and smoothings of the tile's grid
    in place of one, no per-star kernel, nothing at the read (AGB reuses
    the same stars' classes, sec. 5.2). The `log10 F_4.5` axis keeps the
    ordinary one-cell smoothing (sec. 2) in every pass. `H.sum() == 1 -
    mass_outside` stays exact over the support, the same identity `bin`
    reports, since every star belongs to exactly one class and the edge ξ = 1
    reflects (`fold_wall`) rather than drops what a wide width class
    carries past it; `H` carries its own true zeros, including the exact
    zero the edge ξ = 1 holds in the padding above it. `mass_outside` is
    tracked incrementally PER WIDTH CLASS, exactly as `bin` tracks it for
    its own one histogram (a mark's weight `histogram2d` drops silently
    outside the array's own extent, then the two smoothing passes' own
    edge loss, summed over classes before `fold_wall`) -- never read back
    off `1 - H.sum()`, which would make the identity above true by
    definition regardless of what `H` holds and unable to catch a
    bookkeeping bug (the same defect class as `bmstp.shapes`' old
    `MASS_OUTSIDE_YSO`)."""
    x = np.asarray(x, dtype=float)
    log10_f45 = np.asarray(log10_f45, dtype=float)
    w = np.asarray(w, dtype=float)
    width_class = np.asarray(width_class, dtype=np.int64)
    total_weight = w.sum()
    if total_weight <= 0:
        return np.zeros((_N_X, _N_B), dtype=np.float64), 1.0
    with np.errstate(divide="ignore"):
        log10_xi = np.log10(x)
    # edge convention (sec. 2), the same nudge `bin` applies.
    log10_xi = np.nextafter(log10_xi, -np.inf)
    H = np.zeros((_N_X, _N_B), dtype=np.float64)
    mass_outside = 0.0
    for k in range(len(sigma_classes_cells)):
        sel = width_class == k
        if not np.any(sel):
            continue
        w_k = w[sel]
        Hk, _, _ = np.histogram2d(
            log10_xi[sel], log10_f45[sel], bins=[LOG10_XI_EDGES, LOG10_F45_EDGES], weights=w_k)
        # a mark whose (xi, F) falls entirely outside the array's own
        # extent is dropped silently by `histogram2d`: counted against
        # the SAME `total_weight` every class's `mass_outside` is
        # measured against (the tile's whole retained sample), not
        # against this width class's own subtotal.
        mass_outside += (float(w_k.sum()) - float(Hk.sum())) / total_weight
        Hk = Hk / total_weight
        mass_before = float(Hk.sum())
        sigma_k = float(sigma_classes_cells[k])
        Hk = gaussian_filter1d(Hk, sigma=sigma_k, axis=0, mode="constant",
                                truncate=_truncate_for(sigma_k, _N_X))
        Hk = gaussian_filter1d(Hk, sigma=1.0, axis=1, mode="constant",
                                truncate=_truncate_for(1.0, _N_B))
        mass_outside += mass_before - float(Hk.sum())
        H += Hk
    # THE EDGE ξ = 1: fold whatever mass a width class's own smoothing carried
    # past `log10 ξ = 0` back onto the support -- reflected, never dropped,
    # and mass-conserving, so it does not change `mass_outside`.
    H = fold_wall(H)
    return H.astype(np.float64), mass_outside


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
    """The `(128, 128)` linear operator `K` shifting the `log10 ξ` axis
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
    """`(H_s, mass_lost)`: the source's column kernel applied along the
    `log10 ξ` axis (sec. 2, 4.2), carrying a stored shape from the
    DISTANCE coordinate it is built in, `x = a / T` (support `[0, 1]`,
    the edge ξ = 1 at `log10 ξ = 0` its boundary condition, where `H` still
    reflects), into the fitter's MEASURED coordinate, `a_hat / A_beam`
    (the distance coordinate times the pencil-over-beam ratio the column
    kernel distributes), whose support is not bounded at the edge ξ = 1: `H_s =
    (w * K1 + (1 - w) * K2) @ H`, `K1`, `K2` the shift-and-smooth
    operators of `_shift_kernel` for the mixture's two components,
    combined before the one matrix product. Mass the kernel carries past
    `log10 ξ = 0` is a real pencil column above the beam mean in the
    measured coordinate, kept where it lands, in the one dex of padding
    the array already carries above the edge ξ = 1 -- not folded back onto the
    support. `mass_lost` (`H.sum() - H_s.sum()`) is only the mass the
    shift carries past the array's own `log10 ξ` edges (`-3.0`, `+1.0`),
    which `K`'s finite extent already excludes from `H_s`."""
    K1 = _shift_kernel(mu1, sig1)
    K2 = _shift_kernel(mu2, sig2)
    K = w * K1 + (1.0 - w) * K2
    H_s = K @ H
    mass_lost = float(H.sum() - H_s.sum())
    return H_s, mass_lost
