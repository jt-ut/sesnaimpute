"""The per-source posterior-facing prior callable (`10_POSTERIOR.md`
section 3's three properties; `SPEC_PRIORS.md` section 0.2; the prior
table's own join, `IMPLEMENTATION.md` sections 3 and 5).

`SourcePrior(config, region)` loads every upstream product for one region
once: the prior table (`prior.table.read`), the three field-family tile
shapes (`prior.star_shapes`), the YSO sightline shape and its shared
column kernel (`prior.yso.YsoShape`, `prior.kernel.Kernel`), the galaxy
counts law (`bms/gal/counts/survey`), and the H2S region lognormal
(`bms/h2s/prior_h2s_region`). Every class's selection is now read directly
per source from the exact per-source selection products (`prior.
star_selection`, `prior.gal`, `prior.h2s`): `EPS_<cls>[s]`, an `(n_x, n_b)`
curve on the shared scaled-extinction ladder `X_LADDER` (`x = a / A_s`) by
the class's own second axis. `SourcePrior.prepare(rows)` gathers, once per
batch, only these three products' rows (float16 on disk, converted to
float32 for the batch) -- the one quantity too large to hold for a whole
survey in memory at once. `log_density(cls, rows, a, log10_b,
model_index=None)` returns `ln lambda~_cls(a, log10 B | I_s)`, vectorised
over the batch of sources and query points (rule 8): no Python loop over
either.

**The five formulas** (`SPEC_PRIORS.md` section 0.2's identity `N_C(s) =
Integral p_C(a,b|I_s) eps_s(a,b) da db`, `lambda~_C = p_C eps_s / N_C`):

STAR, AGB, PAHC
    `lambda~_C(a, log10 B) = p_C(a, log10 B | I_s) . eps_s(a, log10 B) / Z_C`

    `p_C` is the source's own tile shape (`star_shapes.ClassShape.
    density`'s own machinery, inlined here as the blend of the two shape
    nodes bracketing `A_s` in `log A`, bicubic in `(log10 x, log10 B)`,
    PAHC additionally blended over its own 8 micron limit grid) converted
    from the stored array's own MASS units to a density in `(a, log10 B)`
    by the grid's fixed cell area and the `d(log10 x)/da = 1/(a ln10)`
    Jacobian, exactly as before.

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
    constant) times this source's own `EPS[s]` (`prior.gal`'s per-source
    selection, an `(n_x, n_s)` curve on `X_LADDER` x `LOG10_S_GRID`),
    bilinearly interpolated at `x = a / A_s` and `log10 S = log10 B +
    log10 f_ref,h` (the query's library model's own 4.5 micron reference
    flux, `galz_register`'s `F_REF_I2`).

    `Z_GAL` is `Integral (normalised p(log10 S)) . eps(x=1, log10 S)
    dlog10 S`, computed once per source in `__init__`, evaluated at the
    source's own nominal column (`x = 1`, `a = A_s`) rather than
    integrated jointly over `a` as well: `p_a`'s own mass concentrates
    within a few tenths of a dex of `A_s` at the kernel's measured sigma,
    where the per-source `eps` curve's variation with `x` is small next
    to its variation with `log10 S` -- disclosed, not integrated exactly.

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
in memory for a whole survey at once is the three per-source selection
products' `EPS` arrays (float16 on disk, `(n_source, n_x, n_b-or-s-or-
sigma)`): `prepare(rows)` reads and float32-converts only the rows of the
current batch (about ten thousand sources, `CODING_RULES.md` 10b) from
each of the three files. Every other per-source quantity (`A_COL_K`,
`A_COL_SIG_K`, the ridge, the normalisers) is already resident in the
region table loaded once in `__init__`, and every extinction marginal
(GAL's kernel `pdf`, YSO/H2S's `marginal_exact`) is now closed-form, so
`log_density` needs no further per-batch tabulation for them.
"""

import os
import time

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute.prior import column_grid, star_shapes
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


def _marginal_exact_chunked(yso_shape, a, sl_rows, a_col, sigma_col):
    """`(n,)`: `YsoShape.marginal_exact`, called in `_MARGINAL_CHUNK`
    blocks so its own per-call `(chunk, n_embedding_cell)` gather never
    scales with the caller's own batch (module docstring)."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, _MARGINAL_CHUNK):
        stop = min(start + _MARGINAL_CHUNK, n)
        out[start:stop] = yso_shape.marginal_exact(
            a[start:stop], sl_rows[start:stop], a_col[start:stop], sigma_col[start:stop])
    return out


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


class SourcePrior(object):
    """`log_density(cls, rows, a, log10_b, model_index=None)`, one region's
    every upstream product loaded once (module docstring)."""

    def __init__(self, config, region):
        self.config = config
        self.region = region
        self.table = table_module.read(config, region)
        self.n_source = self.table["A_COL_K"].shape[0]
        self._idx_i4 = BAND_KEYS.index("I4")

        # -- STAR, AGB, PAHC: the tile shape (density only) and the
        # per-source selection product's own grids (its `EPS` arrays are
        # read per batch, `prepare`).
        self.shapes = {}
        self.family_grids = {}
        self._star_selection_path = config_module.product_path(
            config, "bms", "star", "selection", "source", region=region)
        with h5py.File(self._star_selection_path, "r") as f:
            x_ladder = f["X_LADDER"][:].astype(np.float64)
            for cls in FAMILY_CLASSES:
                shape = star_shapes.read(config, region, cls)
                dx = float(np.mean(np.diff(shape.x_edges)))
                db = float(np.mean(np.diff(shape.b_edges)))
                self.shapes[cls] = shape
                self.family_grids[cls] = dict(
                    dx=dx, db=db, x_ladder=x_ladder,
                    b_grid=f["LOG10_B_GRID_%s" % cls.upper()][:].astype(np.float64))

        # -- GAL: the survey-wide counts law and this region's per-source
        # selection curve (`prior.gal`'s own `EPS[n, n_x, n_s]`).
        self._gal_selection_path = config_module.product_path(
            config, "bms", "gal", "selection", "source", region=region)
        counts_path = config_module.product_path(config, "bms", "gal", "counts", "survey")
        with h5py.File(counts_path, "r") as f:
            law_log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
            gal_phi_s = f["PHI_S"][:].astype(np.float64)
        with h5py.File(self._gal_selection_path, "r") as f:
            self.gal_x_ladder = f["X_LADDER"][:].astype(np.float64)
            self.gal_log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
            if not np.allclose(law_log10_s_grid, self.gal_log10_s_grid):
                raise ValueError(
                    "prior.callable: GAL's per-source selection and the survey "
                    "counts law disagree on LOG10_S_GRID for region %r" % region)
            idx_x1 = int(np.argmin(np.abs(self.gal_x_ladder - 1.0)))
            eps_at_a_col = f["EPS"][:, idx_x1, :].astype(np.float32)
        self.gal_fref = _library_reference_flux(config, "gal")

        # `phi(S).S` is only PROPORTIONAL to a density (`SPEC_PRIORS.md`
        # section 5.2's own "prop"); `gal_phi_total` (a survey-wide
        # constant) makes it one. `gal_z[s]`, evaluated at this source's
        # own nominal column (module docstring's disclosed x=1
        # approximation), is the normaliser `Integral shape . eps = Z`
        # actually needs -- NOT `prior.gal`'s own stored `Z_GAL`
        # (`counts_star_family.gal_counts`'s eps-weighted average, an
        # unrelated quantity).
        self.gal_phi_total = float(np.trapz(
            gal_phi_s * (10.0 ** self.gal_log10_s_grid) * LN10, self.gal_log10_s_grid))
        self._gal_phi_density_grid = (
            gal_phi_s * (10.0 ** self.gal_log10_s_grid) * LN10) / self.gal_phi_total
        self.gal_z = np.trapz(
            self._gal_phi_density_grid[None, :] * eps_at_a_col,
            self.gal_log10_s_grid, axis=1)

        # -- YSO: the sightline shape (`marginal_exact`, shared with H2S).
        self.yso_shape = yso_module.YsoShape.read(config, region)

        # -- H2S: the region's Sigma lognormal and this region's
        # per-source selection curve (`prior.h2s`'s own `EPS[n, n_x,
        # n_sigma]`).
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
        self._prep_gal_eps = None
        self._prep_h2s_eps = None

    def prepare(self, rows):
        """Gathers this batch's own rows (about ten thousand sources,
        `CODING_RULES.md` 10b) from the three per-source selection
        products, float16 -> float32 -- the one quantity too large to
        hold for a whole survey in memory at once (module docstring)."""
        rows = np.asarray(rows, dtype=np.intp)
        uniq_rows = np.unique(rows)

        star_eps = {}
        with h5py.File(self._star_selection_path, "r") as f:
            for cls in FAMILY_CLASSES:
                star_eps[cls] = f["EPS_%s" % cls.upper()][uniq_rows, :, :].astype(np.float32)
        with h5py.File(self._gal_selection_path, "r") as f:
            gal_eps = f["EPS"][uniq_rows, :, :].astype(np.float32)
        with h5py.File(self._h2s_selection_path, "r") as f:
            h2s_eps = f["EPS"][uniq_rows, :, :].astype(np.float32)

        self._prep_rows = uniq_rows
        self._prep_star_eps = star_eps
        self._prep_gal_eps = gal_eps
        self._prep_h2s_eps = h2s_eps

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
        (module docstring)."""
        if cls not in CLASSES:
            raise ValueError("SourcePrior.log_density: unknown class %r, must be one of %r"
                             % (cls, CLASSES))
        if cls in ("gal", "h2s") and model_index is None:
            raise ValueError("SourcePrior.log_density: class %r needs model_index" % (cls,))
        rows2d, a2, b2, mi2, shp = self._prepare(rows, a, log10_b, model_index)
        if cls in FAMILY_CLASSES:
            out = self._log_density_family(cls, rows2d, a2, b2)
        elif cls == "gal":
            out = self._log_density_gal(rows2d, a2, b2, mi2)
        elif cls == "yso":
            out = self._log_density_yso(rows2d, a2, b2)
        else:
            out = self._log_density_h2s(rows2d, a2, b2, mi2)
        return out.reshape(shp)

    # -----------------------------------------------------------------
    # STAR, AGB, PAHC
    # -----------------------------------------------------------------

    def _log_density_family(self, cls, rows2d, a2, b2):
        shape = self.shapes[cls]
        fg = self.family_grids[cls]
        rows = rows2d.ravel()
        a = a2.ravel()
        b = b2.ravel()
        valid = a > 0.0
        a_safe = np.where(valid, a, 1.0)

        tile_id = self.table["TILE_ID"][rows]
        a_col = self.table["A_COL_K"][rows]
        x_query = np.where(valid, a_safe / a_col, 0.0)
        log_x = np.log10(np.where(x_query > 0.0, x_query, 1.0e-300))
        in_box = ((log_x >= shape.x_edges[0]) & (log_x <= shape.x_edges[-1])
                 & (b >= shape.b_edges[0]) & (b <= shape.b_edges[-1]))

        node_lo_shape, t_node = column_grid.bracket(np.log(a_col), np.log(shape.shape_nodes))
        node_hi_shape = np.minimum(node_lo_shape + 1, shape.shape_nodes.size - 1)

        def _density(node_idx, limit_idx):
            mass = shape._eval_node(tile_id, node_idx, log_x, b, limit_idx)
            mass = np.where(in_box, mass, 0.0)
            return mass / (a_safe * LN10 * fg["dx"] * fg["db"])

        if cls != "pahc":
            p_lo = _density(node_lo_shape, None)
            p_hi = _density(node_hi_shape, None)
        else:
            f_lim8 = self.table["F_LIM_50_MJY"][rows, self._idx_i4]
            limit_lo, t_limit = column_grid.bracket(np.log10(f_lim8), shape.limit_log)
            limit_hi = np.minimum(limit_lo + 1, shape.limit_log.size - 1)
            p_lolo, p_lohi = _density(node_lo_shape, limit_lo), _density(node_lo_shape, limit_hi)
            p_hilo, p_hihi = _density(node_hi_shape, limit_lo), _density(node_hi_shape, limit_hi)
            p_lo = (1.0 - t_limit) * p_lolo + t_limit * p_lohi
            p_hi = (1.0 - t_limit) * p_hilo + t_limit * p_hihi

        p_ab = (1.0 - t_node) * p_lo + t_node * p_hi

        local = self._prep_local_index(rows)
        eps_batch = self._prep_star_eps[cls][local]
        eps_s = _interp_eps_2d(eps_batch, fg["x_ladder"], fg["b_grid"], x_query, b)

        numerator = np.where(valid, p_ab * eps_s, 0.0)
        z = self.table["Z_%s" % cls.upper()][rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_val = np.log(numerator) - np.log(z)
        return np.where((numerator > 0.0) & (z > 0.0) & valid, ln_val, -np.inf)

    # -----------------------------------------------------------------
    # GAL
    # -----------------------------------------------------------------

    def _log_density_gal(self, rows2d, a2, b2, mi2):
        """`p_a` a plain elementwise log-normal density at each row's own
        `(A_s, sigma_col, map_class)` (module docstring: `Kernel.pdf`'s
        own formula, evaluated one `t` per source rather than `Kernel.
        pdf`'s own shared-`t`-across-sources broadcasting)."""
        rows = rows2d.ravel()
        a = a2.ravel()
        b = b2.ravel()
        mi = mi2.ravel()
        valid = a > 0.0
        a_safe = np.where(valid, a, 1.0)

        a_col = self.table["A_COL_K"][rows]
        sigma_col = self.table["A_COL_SIG_K"][rows]
        sl_rows = self.table["HPX256_ROW"][rows]
        map_class = self.yso_shape._map_class(sl_rows)
        mu, sigma = self.yso_shape.kernel.params(a_col, sigma_col, map_class)
        loc = np.log10(a_col) + mu
        z_score = (np.log10(a_safe) - loc) / sigma
        p_a = (np.exp(-0.5 * z_score * z_score) / (sigma * np.sqrt(2.0 * np.pi))
              / (a_safe * LN10))
        p_a = np.where(valid, p_a, 0.0)

        x_query = np.where(valid, a_safe / a_col, 0.0)
        local = self._prep_local_index(rows)
        eps_batch = self._prep_gal_eps[local]
        log10_s = b + np.log10(self.gal_fref[mi])
        eps_val = _interp_eps_2d(eps_batch, self.gal_x_ladder, self.gal_log10_s_grid,
                                 x_query, log10_s)
        phi_density = np.interp(log10_s, self.gal_log10_s_grid, self._gal_phi_density_grid)
        p_logs = phi_density * eps_val

        numerator = p_a * p_logs
        z = self.gal_z[rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_val = np.log(numerator) - np.log(z)
        return np.where((numerator > 0.0) & (z > 0.0) & valid, ln_val, -np.inf)

    # -----------------------------------------------------------------
    # YSO
    # -----------------------------------------------------------------

    def _log_density_yso(self, rows2d, a2, b2):
        """`p(a | A_s)` via `YsoShape.marginal_exact`, this source's own
        exact adopted column AND measurement uncertainty (module
        docstring: the exact form, not the node-bracket `marginal_at`)."""
        rows = rows2d.ravel()
        a = a2.ravel()
        b = b2.ravel()

        a_col = self.table["A_COL_K"][rows]
        sigma_col = self.table["A_COL_SIG_K"][rows]
        sl_rows = self.table["HPX256_ROW"][rows]
        p_a = _marginal_exact_chunked(self.yso_shape, a, sl_rows, a_col, sigma_col)
        p_a = np.where(a > 0.0, p_a, 0.0)

        mean_b = self.table["RIDGE_INTERCEPT"][rows] + self.table["RIDGE_SLOPE"][rows] * a
        width = self.table["RIDGE_WIDTH"][rows]
        z_score = (b - mean_b) / width
        p_b = _INV_SQRT_2PI / width * np.exp(-0.5 * z_score ** 2)

        numerator = p_a * p_b
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_val = np.log(numerator)
        return np.where((numerator > 0.0) & (a >= 0.0), ln_val, -np.inf)

    # -----------------------------------------------------------------
    # H2S
    # -----------------------------------------------------------------

    def _log_density_h2s(self, rows2d, a2, b2, mi2):
        """`p(a | A_s)` shared with YSO's `marginal_exact`; `eps` this
        source's own H2S `EPS[s]`, bilinearly interpolated (module
        docstring), no depth-group index."""
        rows = rows2d.ravel()
        a = a2.ravel()
        b = b2.ravel()
        mi = mi2.ravel()
        valid = a > 0.0
        a_safe = np.where(valid, a, 1.0)

        a_col = self.table["A_COL_K"][rows]
        sigma_col = self.table["A_COL_SIG_K"][rows]
        sl_rows = self.table["HPX256_ROW"][rows]
        p_a = _marginal_exact_chunked(self.yso_shape, a, sl_rows, a_col, sigma_col)
        p_a = np.where(valid, p_a, 0.0)

        log10_sigma = b + np.log10(self.h2s_fref[mi])
        z_score = (log10_sigma - self.h2s_logsig_mean) / self.h2s_logsig_std
        p_sigma = _INV_SQRT_2PI / self.h2s_logsig_std * np.exp(-0.5 * z_score ** 2)

        x_query = np.where(valid, a_safe / a_col, 0.0)
        local = self._prep_local_index(rows)
        eps_batch = self._prep_h2s_eps[local]
        eps_val = _interp_eps_2d(eps_batch, self.h2s_x_ladder, self.h2s_log10_sigma_grid,
                                 x_query, log10_sigma)

        z = self.table["Z_H2S"][rows]
        numerator = p_a * p_sigma * eps_val
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_val = np.log(numerator) - np.log(z)
        return np.where((numerator > 0.0) & (z > 0.0) & valid, ln_val, -np.inf)


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
    sigmas past `a_col`, at this source's own sightline arm: the exact
    extinction marginal every class but the family shapes reads is
    sharply peaked toward `a_col`, so log-spacing (the same fix
    `star_shapes` uses for its own `log10 x` axis) is needed to resolve
    the peak."""
    sl_row = int(prior.table["HPX256_ROW"][row])
    sigma_col = float(prior.table["A_COL_SIG_K"][row])
    map_class = prior.yso_shape._map_class(np.array([sl_row]))
    mu, sigma = prior.yso_shape.kernel.params(
        np.array([a_col]), np.array([sigma_col]), map_class)
    a_hi = a_col * 10.0 ** (mu[0] + _A_GRID_SIGMAS * sigma[0])
    return np.geomspace(_A_GRID_FLOOR, max(a_hi, 1.0e-5), _CHECK_GRID_N)


def check(config, region, n_sources=50, seed=0):
    """For `n_sources` random catalogue rows and each of the six classes,
    the numerical integral of `exp(log_density)` over a fine `(a, log10
    B)` grid against 1: within 0.02 for STAR/AGB/PAHC/GAL/H2S
    (`star_shapes.EPS_SHAPE`), within 1e-3 for YSO (analytic). Reports the
    worst deviation per class, and the wall time of one source's
    `log_density` call for one class at 4,066 models times 9 query
    points, extrapolated linearly to the survey's 8.66e6 sources."""
    prior = SourcePrior(config, region)
    rng = np.random.default_rng(seed)
    n = min(int(n_sources), prior.n_source)
    rows = rng.choice(prior.n_source, size=n, replace=False)
    prior.prepare(rows)

    worst = {cls: 0.0 for cls in CLASSES}
    for row in rows:
        a_col = float(prior.table["A_COL_K"][row])

        for cls in FAMILY_CLASSES:
            shape = prior.shapes[cls]
            a_grid, b_grid = _family_grid(shape, a_col)
            a_grid = a_grid[a_grid > 0.0]
            integral = _integrate(prior, cls, row, a_grid, b_grid)
            worst[cls] = max(worst[cls], abs(integral - 1.0))

        a_grid_cloud = _extinction_grid(prior, row, a_col)

        b_grid_gal = prior.gal_log10_s_grid - np.log10(prior.gal_fref[0])
        integral = _integrate(prior, "gal", row, a_grid_cloud, b_grid_gal, model_index=0)
        worst["gal"] = max(worst["gal"], abs(integral - 1.0))

        integral = _integrate_yso(prior, row, a_grid_cloud)
        worst["yso"] = max(worst["yso"], abs(integral - 1.0))

        log10_sigma_lo = prior.h2s_logsig_mean - 6 * prior.h2s_logsig_std
        log10_sigma_hi = prior.h2s_logsig_mean + 6 * prior.h2s_logsig_std
        b_grid_h2s = (np.linspace(log10_sigma_lo, log10_sigma_hi, _CHECK_GRID_N)
                     - np.log10(prior.h2s_fref[0]))
        integral = _integrate(prior, "h2s", row, a_grid_cloud, b_grid_h2s, model_index=0)
        worst["h2s"] = max(worst["h2s"], abs(integral - 1.0))

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

    return worst, read_cost


def report(region, worst, read_cost):
    lines = ["prior.callable: %s: normalisation check (bar 0.02 all but YSO 1e-3)" % region]
    for cls in CLASSES:
        bar = 1.0e-3 if cls == "yso" else 0.02
        lines.append("prior.callable: %s: %s: worst |integral - 1| = %.4g (bar %.4g)%s"
                     % (region, cls, worst[cls], bar,
                        "" if worst[cls] <= bar else "  ** EXCEEDS BAR **"))
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
        _worst, _read_cost = check(cfg, _region)
        for _line in report(_region, _worst, _read_cost):
            print(_line, flush=True)
