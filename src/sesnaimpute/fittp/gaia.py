"""The Gaia congruence `Gamma_{s,h}`
(10_POSTERIOR.md card T13, the `Gamma_{s,h}` row of its factor table;
reading/06_fitter_and_impute.md section A's `Gamma_{s,h}` row).

The cloud classes' (YSO, H2S) astrometric likelihood is centred on the dust
structure's own peak distance, not the region's literature distance, with
the structure's own line-of-sight depth in quadrature with the Gaia
parallax error (SPEC_BMSTP_DRAFT.md sec. 6.4): `mu = 1000/D_PEAK`, `sigma^2
= sigma_plx^2 + (1000*half_width_pc/D_PEAK^2)^2`, `half_width_pc` half the
region's own cloud interval (`bmstp.sample_cloud.cloud_interval_pc`, the
doubled 16-84% interval about `D_PEAK`). Centring on the literature
distance while sizing the width off the structure peaked elsewhere puts
the Normal's mass in the wrong tail wherever the two distances disagree
(R4 D2); using the same peak for both keeps the statement single-sourced.

    H_h      = sigmoid((G_LIM - Gmag_h) / TAU_G)     -- population.anchor_tiles.gaia_detection_weight
    Gamma_h  = G_s * H_h * A_X  +  (1 - G_s) * (1 - H_h)

`Gmag_h` is the model's own predicted Gaia G magnitude at its fitted
extinction and brightness (the register's `G0_FLUX`, dimmed by `KG_DRAINE`/
`KG_WHITNEY` blended at the source's own extinction ramp weight). `A_X`
depends on the source and class only, never on the model: a Normal in
parallax anchored at the cloud for YSO/H2S, at zero parallax for GAL, and
the region's simulated field-star population's own 1/D marginal for
STAR/AGB/PAHC (`SPEC_PRIORS.md` section 4: PAHC's parent population is the
whole simulated field-star population, so it shares STAR's shape). A
source with no Gaia counterpart at all carries no information (Gamma = 1,
ln Gamma = 0, every model); a matched source with no usable astrometric
solution degrades to A_X = 1, the "toward no information, never a cliff"
rule for the no-parallax branch.

The field-star classes' marginal is the same per-star reweighting every
other reader of the population applies (`population.star_population.
partition_weights`'s `W_STAR`/`W_AGB`, stored per tile in the tile
population product and keyed back to the pooled field-star row by
`STAR_INDEX`), not an unweighted count of retained rows: STAR's marginal
is `W_STAR`-weighted, AGB's is `W_AGB`-weighted, and PAHC shares STAR's
marginal (`SPEC_PRIORS.md` section 4: PAHC's parent population is the
whole field-star population, the same one STAR draws its own marginal
from) -- an unweighted count scores against a distribution the prior
never placed the class at (R4 D3). PAHC's own per-star contamination
weight (`P_PAHC`, at a grid of 8 micron limit values, keyed by
`STAR_INDEX` within each tile against a grid column a pooled read cannot
select) is a second, separate reweighting this term still does not carry
-- disclosed, not silently guessed (brief's own fallback).
"""

import math

import h5py
import numba
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute.bmstp import sample_cloud
from sesnaimpute.population import selection
from sesnaimpute.population.anchor_tiles import (
    GAIA_G_LIM_MAG, GAIA_G_ROLLOFF_MAG, gaia_detection_weight)

__all__ = ["GaiaTerm", "GAIA_G_LIM_MAG", "GAIA_G_ROLLOFF_MAG"]


# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: One SED-fit model register per class (`config.inputs["sed_models"]`/
#: `registers/`) -- the same files `population.callable._library_reference_flux`
#: (GAL, H2S) and `population.field_stars` (STAR's own atmosphere templates)
#: already read for their class's geometry; AGB's own dusty-shell library
#: is `agb_register.hdf5`, distinct from the sps atmosphere templates
#: `population.field_stars` matches a TRILEGAL star to. Built off
#: `definitions.CLASS_REGISTER` (shared with `fit.sweep`), lower-cased to
#: this module's own class-name convention, so every class the registers
#: cover -- STAR, AGB, PAHC, GAL, YSO, H2S -- resolves to a file here.
_REGISTER_FILE = {cls.lower(): "%s_register.hdf5" % key
                   for cls, key in definitions.CLASS_REGISTER.items()}

#: Classes whose A_X anchors on the cloud's own distance, with the depth
#: term (10_POSTERIOR.md card T4; "cloud classes").
CLOUD_ANCHORED_CLASSES = ("yso", "h2s")

#: Classes whose A_X is the region's simulated field-star population's own
#: parallax marginal. PAHC's parent population (`SPEC_PRIORS.md` section 4)
#: is the whole field-star population, the same one STAR draws its own
#: marginal from, so `_build_star_marginal`'s STAR branch (`cls != "agb"`)
#: already gives PAHC STAR's own marginal with no further branching.
STAR_MARGINAL_CLASSES = ("star", "agb", "pahc")

#: Per-tile weight column `_build_star_marginal` reads for a class
#: (`population.star_population.partition_weights`): AGB's own dusty share,
#: STAR's for STAR and PAHC alike (STAR_MARGINAL_CLASSES' comment).
_MARGINAL_WEIGHT_KEY = {"star": "W_STAR", "pahc": "W_STAR", "agb": "W_AGB"}

#: Bin width for the field-star parallax marginal, mas: narrow enough that
#: the midpoint rule's discretisation error stays small even at the
#: smallest disclosed sigma_eff a matched source can carry (0.05 mas),
#: w/sigma <= 0.25 there.
STAR_MARGINAL_BIN_WIDTH_MAS = 0.0125

_SQRT_2PI = np.sqrt(2.0 * np.pi)


@numba.njit(parallel=True, cache=True)
def _gaia_h_kernel(gmag_flat, out):
    """`H_h = sigmoid((G_LIM - Gmag_h) / TAU_G)` for every (source,
    template) pair, flattened (`population.anchor_tiles.
    gaia_detection_weight`'s own algebra, one `exp` per pair -- the
    floor), `@njit(parallel=True)` over the block's own pairs via
    `prange` rather than numpy's single-threaded `exp` (W9c: `ln_gamma`'s
    hot loop, 10_POSTERIOR.md T13).
    """
    n = gmag_flat.shape[0]
    for i in numba.prange(n):
        arg = -(GAIA_G_LIM_MAG - gmag_flat[i]) / GAIA_G_ROLLOFF_MAG
        if arg > 700.0:
            arg = 700.0
        out[i] = 1.0 / (1.0 + math.exp(arg))


def _normal_pdf(x, mean, sigma):
    z = (np.asarray(x, dtype=np.float64) - mean) / sigma
    return np.exp(-0.5 * z * z) / (sigma * _SQRT_2PI)


# ---------------------------------------------------------------------------
# per-region inputs the term reads once, never per source
# ---------------------------------------------------------------------------

def _cloud_normal_params_mas(config, region):
    """`(centre_mas, half_width_mas)`, the cloud-anchored Normal's own
    parameters (SPEC_BMSTP_DRAFT.md sec. 6.4): centred at `1000/D_PEAK`,
    with `half_width_mas` the linearised parallax half-width of the
    region's own cloud interval (`sample_cloud.cloud_interval_pc`, the
    doubled 16-84% interval about `D_PEAK`) -- both drawn from `D_PEAK`
    alone, so the centre and the width the depth term uses come from the
    one statement rather than the width being sized about a peak the
    centre does not use (R4 D2)."""
    depth_path = config_module.product_path(config, "sky/derived", "edenhofer", "depth", "region")
    with h5py.File(depth_path, "r") as f:
        names = [n.decode("utf-8") for n in f["REGION"][:]]
        row = names.index(region)
        d_peak_pc = float(f["D_PEAK_PC"][row])
    d_front_pc, d_back_pc = sample_cloud.cloud_interval_pc(config, region)
    half_width_pc = 0.5 * (d_back_pc - d_front_pc)
    return 1000.0 / d_peak_pc, 1000.0 * half_width_pc / d_peak_pc ** 2


def _load_gaia_match(config, region):
    """This region's own Gaia crossmatch (`match_gaia_source__<region>.
    hdf5`), row-aligned to the region's source order, read once for the
    whole region's sweep. `GAIA_SOURCE_ID == -1` is the product's own
    sentinel for "no counterpart at all" -- distinct from a matched
    source with no usable parallax (`NO_PM` or a non-finite `PLX_MAS`),
    which the A_X branch degrades to A_X = 1 rather than dropping the
    source's own `G_S`.
    """
    path = config_module.product_path(config, "sky/derived", "gaia", "match", "source", region=region)
    with h5py.File(path, "r") as f:
        return dict(
            matched=f["GAIA_SOURCE_ID"][:] != -1,
            g_s=f["G_S"][:].astype(np.float64),
            plx_mas=f["PLX_MAS"][:].astype(np.float64),
            e_plx_mas=f["E_PLX_MAS"][:].astype(np.float64),
            ruwe=f["RUWE"][:].astype(np.float64),
            no_pm=f["NO_PM"][:].astype(bool),
        )


def _build_star_marginal(config, region, cls, bin_width_mas=STAR_MARGINAL_BIN_WIDTH_MAS):
    """`(bin_centres_mas, p_bin)`: STAR/AGB/PAHC's binned parallax marginal
    off the region's simulated field-star population (10_POSTERIOR.md's
    STAR/AGB row; `SPEC_PRIORS.md` section 4's PAHC row, same parent
    population as STAR) -- `1000/DIST_PC` for every retained field star,
    weighted by the same per-tile reweighting every other reader of this
    population applies (`population.star_population.partition_weights`'s
    `W_STAR`/`W_AGB`, `_MARGINAL_WEIGHT_KEY`), not an unweighted count
    partitioned by `evolved_selector` (R4 D3): a star's weight already
    carries its class partition (`W_AGB` is zero off the evolved rows,
    `W_STAR` is the non-evolved complement plus the evolved rows' own
    non-dusty remainder), summed by `STAR_INDEX` over every tile a
    retained star's row falls in.
    """
    field_stars_path = config_module.product_path(
        config, "population", "trilegal", "field-stars", "region", region=region)
    with h5py.File(field_stars_path, "r") as f:
        n_pool = f["DIST_PC"].shape[0]
        dist_pc = f["DIST_PC"][:].astype(np.float64)

    weight_key = _MARGINAL_WEIGHT_KEY[cls]
    w_pooled = np.zeros(n_pool, dtype=np.float64)
    tile_path = config_module.product_path(config, "population", "star", "population", "tile", region=region)
    with h5py.File(tile_path, "r") as f:
        for name in f.keys():
            if not name.startswith("tile_"):
                continue
            grp = f[name]
            star_index = grp["STAR_INDEX"][()].astype(np.int64)
            np.add.at(w_pooled, star_index, grp[weight_key][()].astype(np.float64))

    keep = w_pooled > 0.0
    dist_pc = dist_pc[keep]
    w_pooled = w_pooled[keep]
    if dist_pc.size == 0:
        raise ValueError(f"GaiaTerm: zero {cls!r} weight in {region!r}'s field-star population")

    plx_mas = 1000.0 / dist_pc
    lo = np.floor(plx_mas.min() / bin_width_mas) * bin_width_mas
    hi = np.ceil(plx_mas.max() / bin_width_mas) * bin_width_mas
    n_bins = max(1, int(round((hi - lo) / bin_width_mas)))
    edges = lo + bin_width_mas * np.arange(n_bins + 1)
    counts, _ = np.histogram(plx_mas, bins=edges, weights=w_pooled)
    p_bin = counts / w_pooled.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = p_bin > 0
    return centres[keep], p_bin[keep]


# ---------------------------------------------------------------------------
# the term
# ---------------------------------------------------------------------------

class GaiaTerm:
    """One region's Gaia-congruence term: `ln_gamma` per source, vectorised
    over one class's models.

    Everything region-wide (the Gaia crossmatch, the cloud Normal's own
    centre and width) is read once at construction; a class's own model
    register (`G0_FLUX`, `KG_DRAINE`, `KG_WHITNEY`) and its field-star
    parallax marginal (STAR/AGB only) are read the first time that class
    is asked for and cached thereafter -- nothing here re-reads a register
    or a per-region product inside a per-source loop (CODING_RULES.md
    rule 9).
    """

    def __init__(self, config, region):
        self.config = config
        self.region = region
        self._cloud_centre_mas, self._cloud_depth_term_mas = _cloud_normal_params_mas(config, region)
        self._gaia = _load_gaia_match(config, region)
        self._registers = {}
        self._star_marginals = {}

    def _register(self, cls):
        cached = self._registers.get(cls)
        if cached is None:
            path = f"{self.config.inputs['sed_models']}/registers/{_REGISTER_FILE[cls]}"
            with h5py.File(path, "r") as f:
                m = f["models"]
                g0_flux = m["G0_FLUX"][:].astype(np.float64)
                # G0MAG: the template's own unextincted, unscaled Gaia G
                # magnitude, from G0_FLUX once per template at construction
                # (W9c) -- `ln_gamma`'s per-pair Gmag_h is then the additive
                # G0MAG - 2.5 log10_b + A_G, so no per-pair log10/power
                # round trip through flux space is needed. NaN (no G-band
                # coverage) and 0 (dark model, log10(0) = -inf) both pass
                # through quietly, exactly as G0_FLUX itself does.
                with np.errstate(divide="ignore", invalid="ignore"):
                    g0_mag = -2.5 * np.log10(g0_flux / constants.GAIA_G_VEGA_ZP_MJY)
                cached = dict(
                    g0_flux=g0_flux,
                    g0_mag=g0_mag,
                    kg_draine=m["KG_DRAINE"][:].astype(np.float64),
                    kg_whitney=m["KG_WHITNEY"][:].astype(np.float64),
                )
            self._registers[cls] = cached
        return cached

    def _star_marginal(self, cls):
        cached = self._star_marginals.get(cls)
        if cached is None:
            cached = _build_star_marginal(self.config, self.region, cls)
            self._star_marginals[cls] = cached
        return cached

    def warm(self, cls):
        """Populates this class's own lazy caches (`_register`, and, for
        the field-star classes, `_star_marginal`) now, in the CALLING
        process. `_register`/`_star_marginal` load on first use by design
        (a class `ln_gamma` never asks for is never read), which is exactly
        wrong for `fittp.sweep`'s forked worker pool (PARALLEL brief item
        4): left lazy, every worker would independently open the same
        register/population product on its own first task -- eight reads
        of one file instead of one. Call this in the PARENT, on the one
        class this call's pool will sweep, before the pool forks; the
        warmed cache then reaches every worker by fork, copy-on-write.
        """
        self._register(cls)
        if cls in STAR_MARGINAL_CLASSES:
            self._star_marginal(cls)

    def _g_s_and_a_x_block(self, rows, cls):
        """The source-and-class-level factors `ln_gamma` needs, vectorised
        over the block's own sources (`rows`, `(n_block,)`): `matched`
        (`(n_block,)` bool, `False` where the source has no Gaia
        counterpart at all -- the caller's signal to return the neutral
        ln Gamma = 0 for that row without touching a model), `g_s`
        (`(n_block,)`, only meaningful where `matched`) and `A_X`
        (`(n_block,)`, 10_POSTERIOR.md T13): `1` where the source is
        matched but carries no usable parallax (`NO_PM`, a non-finite
        `PLX_MAS`, or a degenerate `sigma_eff`); else the cloud-anchored
        Normal (depth added in quadrature), the zero-parallax Normal
        (GAL), or the field-star marginal (STAR/AGB/PAHC).
        """
        matched = self._gaia["matched"][rows]
        g_s = self._gaia["g_s"][rows]
        plx = self._gaia["plx_mas"][rows]
        no_pm = self._gaia["no_pm"][rows]
        sigma_eff = self._gaia["e_plx_mas"][rows] * np.maximum(1.0, self._gaia["ruwe"][rows])
        no_solution = no_pm | ~np.isfinite(plx) | ~np.isfinite(sigma_eff) | (sigma_eff <= 0)
        good = matched & ~no_solution

        a_x = np.ones(rows.shape[0], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            if cls in CLOUD_ANCHORED_CLASSES:
                sigma_hyp = np.hypot(sigma_eff, self._cloud_depth_term_mas)
                a_x_good = _normal_pdf(plx, self._cloud_centre_mas, sigma_hyp)
            elif cls == "gal":
                a_x_good = _normal_pdf(plx, 0.0, sigma_eff)
            elif cls in STAR_MARGINAL_CLASSES:
                bins_mas, p_bin = self._star_marginal(cls)
                z = (bins_mas[None, :] - plx[:, None]) / sigma_eff[:, None]
                pdf = np.exp(-0.5 * z * z) / (sigma_eff[:, None] * _SQRT_2PI)
                a_x_good = pdf @ p_bin
            else:
                raise ValueError(f"GaiaTerm: no A_X branch defined for class {cls!r}")
        a_x[good] = a_x_good[good]
        return matched, g_s, a_x

    def ln_gamma(self, rows, model_index, a, log10_b, cls):
        """`ln Gamma_{s,h}` for one block's own sources (`rows`, their row
        indices into this region's own row-aligned products, `(n_block,)`)
        over the models named by `model_index` (row positions into `cls`'s
        own register), vectorised over both sources and models: `a`/
        `log10_b`, shape `(n_block, n_model)`, are those models' own
        fitted A_K extinction and log10 brightness at each source.

        Returns an `(n_block, n_model)` float64 array, one natural-log
        multiplicative factor per source and model (the identity this
        replaces: the same numbers as the old per-source call, one row at
        a time).
        """
        rows = np.asarray(rows)
        model_index = np.asarray(model_index)
        n_model = model_index.shape[0]
        matched, g_s, a_x = self._g_s_and_a_x_block(rows, cls)

        reg = self._register(cls)
        g0 = reg["g0_flux"][model_index]
        g0_mag = reg["g0_mag"][model_index]
        kg_draine = reg["kg_draine"][model_index]
        kg_whitney = reg["kg_whitney"][model_index]

        a = np.asarray(a, dtype=np.float64)
        log10_b = np.asarray(log10_b, dtype=np.float64)

        # Gmag_h: the model's own predicted Gaia G at its fitted (a,
        # log10_b) -- G0_FLUX dimmed by A_G, `a` (A_K) carried to A_G
        # through the model's own KG_DRAINE/KG_WHITNEY, blended at the
        # source's own extinction-ramp weight (population.selection's
        # diffuse/dense law blend, evaluated directly in A_K units since
        # `a` is already A_K -- no A_V round trip needed).
        with np.errstate(divide="ignore", invalid="ignore"):
            w = selection.law_dense_weight(a)
        r_diffuse = float(selection.ak_per_av(self.config, 0.0))
        r_dense = float(selection.ak_per_av(self.config, 1.0))
        kappa_g = (1.0 - w) * (kg_draine[None, :] / r_diffuse) + w * (kg_whitney[None, :] / r_dense)
        a_g = a * kappa_g

        # Gmag_h = G0MAG - 2.5 log10_b + A_G (W9c: the same magnitude the
        # old flux-space round trip computed -- g_flux = G0_FLUX * B_hat *
        # 10^(-0.4 A_G), Gmag = -2.5 log10(g_flux/ZP) -- reached without a
        # per-pair log10 or power call over the (n_block, n_model) array;
        # only G0MAG (once per template, at register construction) ever
        # takes a log10).
        gmag = g0_mag[None, :] - 2.5 * log10_b + a_g
        h = np.empty(gmag.shape, dtype=np.float64)
        _gaia_h_kernel(np.ascontiguousarray(gmag.ravel()), h.ravel())
        # G0_FLUX = NaN (no G-band coverage, not darkness) -> H = 0
        # exactly; a dark model's G0_FLUX = 0 already drives Gmag to +inf
        # and H to 0 through the sigmoid's own limit, no special case.
        h = np.where(np.isnan(g0)[None, :], 0.0, h)

        gamma = g_s[:, None] * h * a_x[:, None] + (1.0 - g_s[:, None]) * (1.0 - h)
        # gamma = 0 is a real zero-probability model under a Gaia
        # counterpart's own detection/non-detection evidence (e.g. the
        # source is seen but this model's H_h says it should not be, or
        # vice versa) -- ln(0) = -inf is the ruled likelihood, not a
        # defect, so the divide-by-zero warning is silenced, not the
        # value. Rows with no Gaia counterpart at all carry no
        # information: ln Gamma = 0 for every model, exactly.
        with np.errstate(divide="ignore"):
            ln_g = np.log(gamma)
        return np.where(matched[:, None], ln_g, 0.0)
