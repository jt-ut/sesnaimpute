"""Whether a source clears SESNA's own catalog cut, and the exact
per-depth-group selection tables built from that test (SPEC_PRIORS.md
section 1.3).

SESNA catalogs a source when at least two of its eight bands, dimmed by
its own dust column, clear the source's detection limit
(`catalog.limits.limits`). The dimming coefficient per band is not one
fixed law: it blends smoothly, in log column, from a diffuse-ISM curve
below A_K = 0.5 to a dense-cloud curve above A_K = 1.0. Evaluating this
test exactly for a whole class population is affordable only a few
hundred times per region, so a class's exact selection is tabulated once
per depth group (`prior.depth_groups`) and a source is answered by its
own group's table, shifted by its own common-mode depth (`fit_by_
residual`, `PassFractionModel`).
"""

import json
import os

import h5py
import numpy as np

from sesnaimpute import definitions

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
N_BANDS = len(BAND_KEYS)

#: The five bands with a per-source detection-limit map -- the four-
#: degree-of-freedom residual `split_common_mode` leaves is measured
#: over these; the three 2MASS bands are a fixed region-wide limit and
#: carry no per-source Delta (SPEC_PRIORS.md section 1.3).
BANDS_DEPTH = ("I1", "I2", "I3", "I4", "M1")
_BAND_DEPTH_IDX = np.array([BAND_KEYS.index(b) for b in BANDS_DEPTH])

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


def _limit_offsets(config, f_lim, a, law):
    """`L_i = log10 F_lim,i + 0.4 * a * kappa_i` -- the dimmed limits,
    in the log-flux units a source's own `log10 F` is measured in.
    `f_lim` is `(..., 8)` mJy; `a` broadcasts against its leading axes.
    """
    f_lim = np.asarray(f_lim, dtype=float)
    if f_lim.shape[-1] != N_BANDS:
        raise ValueError(f"f_lim last axis must be {N_BANDS}, got {f_lim.shape}")
    if np.any(f_lim <= 0) or not np.all(np.isfinite(f_lim)):
        raise ValueError("detection limits must be positive and finite")
    return np.log10(f_lim) + 0.4 * np.asarray(a, dtype=float)[..., None] * _kappa(config, law)


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


def split_common_mode(log10_flim_8, ref_log10_flim_8):
    """`(s, Delta)`: the common-mode depth shift and the four-degree-of-
    freedom residual `epsilon` is invariant under, up to exactly
    compensating a `log10 B` shift by `s` (SPEC_PRIORS.md 1.3).

    `s = mean_i(l_i - lbar_i)` over the five `BANDS_DEPTH` bands;
    `Delta = (l - lbar) - s` over those same five bands.
    """
    log10_flim_8 = np.asarray(log10_flim_8, dtype=float)
    ref = np.asarray(ref_log10_flim_8, dtype=float)
    if log10_flim_8.shape[-1] != N_BANDS:
        raise ValueError(
            f"split_common_mode: log10_flim_8's trailing axis must be "
            f"{N_BANDS}, got {log10_flim_8.shape!r}")
    if ref.shape[-1] != N_BANDS:
        raise ValueError(
            f"split_common_mode: ref_log10_flim_8's trailing axis must "
            f"be {N_BANDS}, got {ref.shape!r}")
    diff5 = log10_flim_8[..., _BAND_DEPTH_IDX] - ref[..., _BAND_DEPTH_IDX]
    s = diff5.mean(axis=-1)
    delta5 = diff5 - s[..., None]
    return s, delta5


def _population_eps_batch(population_eps, points, n_a, n_b):
    """`(n_points, n_a, n_b)`: `population_eps` evaluated at every row
    of `points`, batched in one call when `population_eps` accepts a
    batch and returns the right shape, falling back to one call per row
    otherwise.
    """
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    try:
        batched = np.asarray(population_eps(points), dtype=float)
    except Exception:
        batched = None
    if batched is not None and batched.shape == (n_points, n_a, n_b):
        return batched
    out = np.empty((n_points, n_a, n_b), dtype=float)
    for i in range(n_points):
        row = np.asarray(population_eps(points[i]), dtype=float)
        if row.shape != (n_a, n_b):
            raise ValueError(
                f"prior.selection: population_eps must return "
                f"({n_a}, {n_b}), got {row.shape!r} at point {i}")
        out[i] = row
    return out


def _broadcast_inputs(node_lo, node_w, s, delta_5, log10_b=None):
    """Coerce a per-source call's arguments to one common batch length
    `n` by broadcasting -- shared by `evaluate`/`integrate_count`/
    `z_eps` so none of them loops over its own batch in Python.
    """
    node_lo = np.atleast_1d(np.asarray(node_lo))
    node_w = np.atleast_1d(np.asarray(node_w, dtype=float))
    s = np.atleast_1d(np.asarray(s, dtype=float))
    delta_5 = np.atleast_2d(np.asarray(delta_5, dtype=float))
    if delta_5.shape[-1] != len(BANDS_DEPTH):
        raise ValueError(
            f"delta_5's trailing axis must be {len(BANDS_DEPTH)}, "
            f"got {delta_5.shape!r}")
    arrs = [node_lo, node_w, s]
    if log10_b is not None:
        log10_b = np.atleast_1d(np.asarray(log10_b, dtype=float))
        arrs.append(log10_b)
    n = max([a.shape[0] for a in arrs] + [delta_5.shape[0]])

    def _b(a):
        if a.shape[0] == n:
            return a
        if a.shape[0] == 1:
            return np.broadcast_to(a, (n,))
        raise ValueError(f"batch length mismatch ({a.shape[0]} vs {n})")

    node_lo = _b(node_lo).astype(np.intp)
    node_w = _b(node_w)
    s = _b(s)
    if delta_5.shape[0] == 1 and n > 1:
        delta_5 = np.broadcast_to(delta_5, (n, delta_5.shape[1]))
    elif delta_5.shape[0] != n:
        raise ValueError(f"delta_5 batch length mismatch ({delta_5.shape[0]} vs {n})")
    if log10_b is not None:
        log10_b = _b(log10_b)
        return node_lo, node_w, s, delta_5, log10_b
    return node_lo, node_w, s, delta_5


class PassFractionModel:
    """`eps(a_node, b, Delta)`: a class's exact pass fraction, tabulated
    once per (region, class) at each of `knots.group_centres` (`fit`),
    evaluated per source by nearest-centre lookup with no per-row
    Python (`evaluate`). `knots` is a `depth_groups.DepthGroups`:
    anything exposing `.group_centres` `(K, 5)` and
    `.assign_group(delta_5) -> (n,)`.

    `a_nodes` is the shared column ladder; `b_grid` is the class's own
    brightness axis. `eps_table` has shape `(K, n_a, n_b)`.
    """

    def __init__(self, a_nodes, b_grid, knots, eps_table):
        self.a_nodes = np.asarray(a_nodes, dtype=float)
        self.b_grid = np.asarray(b_grid, dtype=float)
        self.knots = knots
        self.eps_table = np.asarray(eps_table, dtype=float)
        n_groups = np.asarray(knots.group_centres, dtype=float).shape[0]
        want = (n_groups, self.a_nodes.size, self.b_grid.size)
        if self.eps_table.shape != want:
            raise ValueError(
                f"PassFractionModel: eps_table must be {want!r} "
                f"(K, n_a, n_b), got {self.eps_table.shape!r}")

    @classmethod
    def fit(cls, population_eps, knots, a_nodes, b_grid):
        """`population_eps` evaluated once per group centre
        (`knots.group_centres`), batched in one call when
        `population_eps` accepts a `(K, 5)` batch.
        """
        a_nodes = np.asarray(a_nodes, dtype=float)
        b_grid = np.asarray(b_grid, dtype=float)
        n_a, n_b = a_nodes.size, b_grid.size
        centres = np.asarray(knots.group_centres, dtype=float)
        eps_table = _population_eps_batch(population_eps, centres, n_a, n_b)
        return cls(a_nodes, b_grid, knots, eps_table)

    def _eps_curve(self, node_lo, node_w, delta_5):
        """`(n, n_b)`: this batch's own eps-vs-b curve in the reference
        frame (`b` not yet shifted by `s`) -- the group lookup plus
        node blend `evaluate`/`integrate_count` both share.
        """
        n_a = self.a_nodes.size
        lo = np.clip(node_lo, 0, n_a - 1)
        hi = np.clip(node_lo + 1, 0, n_a - 1)
        group = self.knots.assign_group(delta_5)
        row = np.arange(group.shape[0])
        table = self.eps_table[group]
        c_lo = table[row, lo]
        c_hi = table[row, hi]
        w = node_w[:, None]
        curve = (1.0 - w) * c_lo + w * c_hi
        return np.clip(curve, 0.0, 1.0)

    def evaluate(self, node_lo, node_w, log10_b, s, delta_5):
        """`(n,)`: this batch's own pass fraction at its queried
        `log10_b`, shifted into the reference frame by `s` before the
        curve `_eps_curve` builds is read off.
        """
        node_lo, node_w, s, delta_5, log10_b = _broadcast_inputs(
            node_lo, node_w, s, delta_5, log10_b)
        curve = self._eps_curve(node_lo, node_w, delta_5)
        b_grid = self.b_grid
        shifted = np.clip(log10_b - s, b_grid[0], b_grid[-1])
        idx = np.clip(np.searchsorted(b_grid, shifted) - 1, 0, b_grid.size - 2)
        span = b_grid[idx + 1] - b_grid[idx]
        safe_span = np.where(span > 0, span, 1.0)
        t = np.where(span > 0, (shifted - b_grid[idx]) / safe_span, 0.0)
        row = np.arange(curve.shape[0])
        eps = curve[row, idx] * (1.0 - t) + curve[row, idx + 1] * t
        return np.clip(eps, 0.0, 1.0)

    def integrate_count(self, intrinsic_density, node_lo, node_w, s, delta_5):
        """`(n,)`: `integral p(b) eps(b; Delta, node blend) db` over
        `self.b_grid`, at each source's own limits.
        """
        node_lo, node_w, s, delta_5 = _broadcast_inputs(node_lo, node_w, s, delta_5)
        n = node_lo.shape[0]
        b_grid = self.b_grid
        intrinsic_density = np.atleast_2d(np.asarray(intrinsic_density, dtype=float))
        if intrinsic_density.shape[0] == 1 and n > 1:
            intrinsic_density = np.broadcast_to(intrinsic_density, (n, intrinsic_density.shape[1]))
        if intrinsic_density.shape != (n, b_grid.size):
            raise ValueError(
                f"integrate_count: intrinsic_density must be "
                f"({n}, {b_grid.size}), got {intrinsic_density.shape!r}")

        curve = self._eps_curve(node_lo, node_w, delta_5)
        shifted_grid = b_grid[None, :] - s[:, None]
        clamped = np.clip(shifted_grid, b_grid[0], b_grid[-1])
        idx = np.clip(np.searchsorted(b_grid, clamped) - 1, 0, b_grid.size - 2)
        lo_b, hi_b = b_grid[idx], b_grid[idx + 1]
        span = hi_b - lo_b
        safe_span = np.where(span > 0, span, 1.0)
        t = np.where(span > 0, (clamped - lo_b) / safe_span, 0.0)
        row = np.arange(n)[:, None]
        eps_at_b = np.clip(curve[row, idx] * (1.0 - t) + curve[row, idx + 1] * t, 0.0, 1.0)
        return np.trapz(intrinsic_density * eps_at_b, x=b_grid, axis=1)

    def z_eps(self, intrinsic_density, node_lo, node_w, s, delta_5):
        """`(n,)`: `integrate_count(...) / integral p(b) db` -- the
        per-source normaliser so the resulting prior integrates to 1.
        """
        node_lo, node_w, s, delta_5 = _broadcast_inputs(node_lo, node_w, s, delta_5)
        n = node_lo.shape[0]
        intrinsic_density = np.atleast_2d(np.asarray(intrinsic_density, dtype=float))
        if intrinsic_density.shape[0] == 1 and n > 1:
            intrinsic_density = np.broadcast_to(intrinsic_density, (n, intrinsic_density.shape[1]))
        numerator = self.integrate_count(intrinsic_density, node_lo, node_w, s, delta_5)
        denom = np.trapz(intrinsic_density, x=self.b_grid, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0.0, numerator / denom, 0.0)

    def write(self, h5group):
        """Writes `EPS_TABLE`/`B_GRID`/`GROUP_CENTRES` into `h5group`."""
        for name in ("EPS_TABLE", "B_GRID", "GROUP_CENTRES"):
            if name in h5group:
                del h5group[name]
        h5group.create_dataset("EPS_TABLE", data=self.eps_table.astype("f4"))
        h5group.create_dataset("B_GRID", data=self.b_grid.astype("f8"))
        h5group.create_dataset("GROUP_CENTRES", data=np.asarray(self.knots.group_centres, dtype="f8"))

    @classmethod
    def read(cls, h5group, knots, a_nodes):
        """The round trip of `write`: `B_GRID`/`EPS_TABLE` read back
        against `knots` and the caller's own `a_nodes`.
        """
        b_grid = np.asarray(h5group["B_GRID"][:], dtype=float)
        eps_table = np.asarray(h5group["EPS_TABLE"][:], dtype=float)
        return cls(np.asarray(a_nodes, dtype=float), b_grid, knots, eps_table)


#: `fit_by_residual`'s own search parameters: the worst-node relative-L1
#: bar the doubling/bisection search reports against (owner ruling
#: 2026-09-04), and the absolute ceiling on K regardless of source count.
RESIDUAL_BAR = 0.01
K_SEARCH_ABS_CAP = 128


def fit_by_residual(fit_depth_groups, region, population_eps, a_nodes, b_grid,
                     receipt_population_eps, n_sources, k_start,
                     seed=0, bar=RESIDUAL_BAR, n_exactness_sample=256):
    """`(knots, model, residual)`: the smallest `K` (doubling from
    `k_start`, then bisecting) whose fitted `PassFractionModel`'s own
    `table_residual` -- drawn from the region's real per-source `Delta`
    -- clears `bar` at the worst column node, capped at
    `min(K_SEARCH_ABS_CAP, sqrt(n_sources))`. Never refuses: the achieved
    `K` and residual are reported regardless of whether `bar` was met.

    `fit_depth_groups` is `depth_groups.fit_depth_groups`, passed in
    rather than imported, so this module never depends on that one.
    `population_eps` fits the table (called at the group centres),
    `receipt_population_eps` grades it (called at real per-source
    `Delta` draws) -- the ordinary case passes the same evaluator twice.
    """
    k_cap = max(1, min(K_SEARCH_ABS_CAP, int(np.sqrt(max(int(n_sources), 1)))))

    def _try(k):
        knots_k = fit_depth_groups(region, k_groups=k, seed=seed)
        model_k = PassFractionModel.fit(population_eps, knots_k, a_nodes, b_grid)
        rng_k = np.random.default_rng(np.random.SeedSequence([int(seed), int(k)]))
        residual_k = table_residual(model_k, receipt_population_eps, rng_k,
                                    n_sample=n_exactness_sample)
        return knots_k, model_k, residual_k

    k = max(1, min(int(k_start), k_cap))
    knots_k, model_k, residual_k = _try(k)
    if residual_k["l1_rel_worst_node"] <= bar or k >= k_cap:
        return knots_k, model_k, residual_k

    lo_k, hi_k, hi_state = k, None, None
    while True:
        k2 = min(k * 2, k_cap)
        knots_2, model_2, residual_2 = _try(k2)
        if residual_2["l1_rel_worst_node"] <= bar:
            hi_k, hi_state = k2, (knots_2, model_2, residual_2)
            break
        if k2 >= k_cap:
            return knots_2, model_2, residual_2
        lo_k = k2
        k = k2

    while hi_k - lo_k > 1:
        mid = (lo_k + hi_k) // 2
        knots_m, model_m, residual_m = _try(mid)
        if residual_m["l1_rel_worst_node"] <= bar:
            hi_k, hi_state = mid, (knots_m, model_m, residual_m)
        else:
            lo_k = mid
    return hi_state


def table_residual(model, population_eps_exact, rng, n_sample=256):
    """Draw `n_sample` `Delta` vectors from `model.knots.delta_sample`
    (the region's own real per-source `Delta`, with replacement), look
    each up in `model`'s fitted group table, and compare against
    `population_eps_exact`'s exact evaluation there. Falls back to
    sampling uniformly over the group centres' own bounding box when
    `delta_sample` is empty.
    """
    n_dims = len(BANDS_DEPTH)
    n_sample = int(n_sample)
    stored = np.asarray(model.knots.delta_sample, dtype=float)
    if stored.ndim == 2 and stored.shape[0] > 0:
        idx = rng.integers(0, stored.shape[0], size=n_sample)
        delta = stored[idx]
    else:
        centres = np.asarray(model.knots.group_centres, dtype=float)
        lo, hi = centres.min(axis=0), centres.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        delta = lo[None, :] + span[None, :] * rng.random((n_sample, n_dims))

    group = model.knots.assign_group(delta)
    approx = np.clip(model.eps_table[group], 0.0, 1.0)

    n_a, n_b = model.a_nodes.size, model.b_grid.size
    exact = _population_eps_batch(population_eps_exact, delta, n_a, n_b)

    abs_err = np.abs(approx - exact)
    node_l1_num = np.sum(abs_err, axis=(0, 2))
    node_l1_den = np.sum(np.abs(exact), axis=(0, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        node_rel = np.where(node_l1_den > 0.0, node_l1_num / node_l1_den, 0.0)
    return {
        "n_sample": int(delta.shape[0]),
        "median_abs_err": float(np.median(abs_err)),
        "max_abs_err": float(np.max(abs_err)),
        "l1_rel_worst_node": float(np.max(node_rel)),
    }
