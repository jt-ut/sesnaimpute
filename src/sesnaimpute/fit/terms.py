"""The Gaia congruence `Gamma_{s,h}` and the library-sampling weight `w_h`
(10_POSTERIOR.md card T13, the `Gamma_{s,h}` row of its factor table;
reading/06_fitter_and_impute.md section A's `Gamma_{s,h}` row).

Lifted from sesnacomplete.sed_fit.term_hooks.gaia_hook (quarry), with the
one addition the reading note names as a gap: the cloud classes' (YSO,
H2S) astrometric likelihood now carries the cloud's own line-of-sight
depth in quadrature with the Gaia parallax error, `sigma^2 = sigma_plx^2
+ (1000*depth_r/d_r^2)^2` (owner, 2026-09-05).

    H_h      = sigmoid((G_LIM - Gmag_h) / TAU_G)     -- prior.anchor_tiles.gaia_detection_weight
    Gamma_h  = G_s * H_h * A_X  +  (1 - G_s) * (1 - H_h)
    ln w_h  ~  -alpha_rho * ln(RHO_KDE1), normalised to sum to 1 within the register

`Gmag_h` is the model's own predicted Gaia G magnitude at its fitted
extinction and brightness (the register's `G0_FLUX`, dimmed by `KG_DRAINE`/
`KG_WHITNEY` blended at the source's own extinction ramp weight). `A_X`
depends on the source and class only, never on the model: a Normal in
parallax anchored at the cloud for YSO/H2S, at zero parallax for GAL, and
the region's simulated field-star population's own 1/D marginal for
STAR/AGB. A source with no Gaia counterpart at all carries no information
(Gamma = 1, ln Gamma = 0, every model); a matched source with no usable
astrometric solution degrades to A_X = 1, the same "toward no information,
never a cliff" rule the quarry states for its own no-parallax branch.

PAHC is out of scope: the quarry anchors it at the cloud distance like
YSO/H2S, but this port does not carry the depth-term fix through that
class (not named in the brief, not exercised by the timed run) -- `cls =
"pahc"` raises rather than guess.
"""

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import regions as regions_module
from sesnaimpute.prior import selection
from sesnaimpute.prior import star_population
from sesnaimpute.prior.anchor_tiles import (
    GAIA_G_LIM_MAG, GAIA_G_ROLLOFF_MAG, gaia_detection_weight)

__all__ = ["GaiaTerm", "library_weights", "GAIA_G_LIM_MAG", "GAIA_G_ROLLOFF_MAG"]


# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: One SED-fit model register per class (`config.inputs["sed_models"]`/
#: `registers/`) -- the same files `prior.callable._library_reference_flux`
#: (GAL, H2S) and `prior.field_stars` (STAR's own atmosphere templates)
#: already read for their class's geometry; AGB's own dusty-shell library
#: is `agb_register.hdf5`, distinct from the sps atmosphere templates
#: `prior.field_stars` matches a TRILEGAL star to.
_REGISTER_FILE = {
    "yso": "yso_register.hdf5",
    "h2s": "h2shock_register.hdf5",
    "gal": "galz_register.hdf5",
    "agb": "agb_register.hdf5",
    "star": "sps_register.hdf5",
}

#: alpha_rho: the exponent on the register's own sampling density
#: (`RHO_KDE1`) that turns it into a quadrature weight. The quarry's
#: production value (sesnacomplete/sed_fit/fit.py, `_W_H_ALPHA_RHO = 1.0`):
#: the full 1/rho correction, safe because the fixed-bandwidth RHO_KDE1
#: estimator bounds outlier weight.
LIBRARY_ALPHA_RHO = 1.0

#: Classes whose A_X anchors on the cloud's own distance, with the depth
#: term (10_POSTERIOR.md card T4; brief's "cloud classes"). The quarry
#: (term_hooks.CLOUD_ANCHORED_CLASSES) also anchors PAHC here; this port
#: does not (module docstring).
CLOUD_ANCHORED_CLASSES = ("yso", "h2s")

#: Classes whose A_X is the region's simulated field-star population's own
#: parallax marginal (quarry term_hooks.STAR_MARGINAL_CLASSES).
STAR_MARGINAL_CLASSES = ("star", "agb")

#: Bin width for the field-star parallax marginal, mas. The quarry's own
#: basis (term_hooks.STAR_MARGINAL_BIN_WIDTH_MAS): narrow enough that the
#: midpoint rule's discretisation error stays small even at the smallest
#: disclosed sigma_eff a matched source can carry (0.05 mas), w/sigma <=
#: 0.25 there.
STAR_MARGINAL_BIN_WIDTH_MAS = 0.0125

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _normal_pdf(x, mean, sigma):
    z = (np.asarray(x, dtype=np.float64) - mean) / sigma
    return np.exp(-0.5 * z * z) / (sigma * _SQRT_2PI)


# ---------------------------------------------------------------------------
# w_h: the library-sampling weight
# ---------------------------------------------------------------------------

def library_weights(config, cls):
    """`ln w_h` for every model in `cls`'s own register, in the register's
    own row order (10_POSTERIOR.md's `w_h` row: "a quadrature weight; sums
    to 1 within a library"): `w_h ~ RHO_KDE1^(-LIBRARY_ALPHA_RHO)`,
    normalised within this one register file.
    """
    path = f"{config.inputs['sed_models']}/registers/{_REGISTER_FILE[cls]}"
    with h5py.File(path, "r") as f:
        rho = f["models"]["RHO_KDE1"][:].astype(np.float64)
    w = rho ** (-LIBRARY_ALPHA_RHO)
    w = w / w.sum()
    return np.log(w)


# ---------------------------------------------------------------------------
# per-region inputs the term reads once, never per source
# ---------------------------------------------------------------------------

def _region_depth_pc(config, region):
    """`SIGMA_DEPTH_PC` for `region` -- the cloud's own line-of-sight
    depth (10_POSTERIOR.md card T4), read once from the survey-wide
    Edenhofer depth table."""
    path = config_module.product_path(config, "sky/derived", "edenhofer", "depth", "region")
    with h5py.File(path, "r") as f:
        names = [n.decode("utf-8") for n in f["REGION"][:]]
        row = names.index(region)
        return float(f["SIGMA_DEPTH_PC"][row])


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
    """`(bin_centres_mas, p_bin)`: STAR or AGB's binned parallax marginal
    off the region's simulated field-star population (10_POSTERIOR.md's
    STAR/AGB row) -- `1000/DIST_PC` for the class's own retained stars,
    uniform weight (every retained field star already carries
    `w_trilegal = 1`, `prior.field_stars`). AGB keeps the evolved subset
    (`star_population.evolved_selector`'s mask) uniformly, matching the
    quarry's own "evolved partition directly, uniform weight" branch;
    STAR keeps the non-evolved complement. This omits the small dusty
    remainder `partition_weights` would move from the evolved rows into
    STAR's own marginal (that split needs `f_dusty_by_chemistry`, out of
    scope for this term -- see the task report).
    """
    path = config_module.product_path(config, "bms", "trilegal", "field-stars", "region", region=region)
    with h5py.File(path, "r") as f:
        dist_pc = f["DIST_PC"][:].astype(np.float64)
        log_g = f["LOG_G"][:].astype(np.float64)
        log_teff = f["LOG_TEFF"][:].astype(np.float64)
        log_l = f["LOG_L"][:].astype(np.float64)
    evolved = star_population.evolved_selector(log_g, log_teff, log_l)
    dist_pc = dist_pc[evolved] if cls == "agb" else dist_pc[~evolved]
    if dist_pc.size == 0:
        raise ValueError(f"GaiaTerm: zero {cls!r} rows in {region!r}'s field-star population")

    plx_mas = 1000.0 / dist_pc
    lo = np.floor(plx_mas.min() / bin_width_mas) * bin_width_mas
    hi = np.ceil(plx_mas.max() / bin_width_mas) * bin_width_mas
    n_bins = max(1, int(round((hi - lo) / bin_width_mas)))
    edges = lo + bin_width_mas * np.arange(n_bins + 1)
    counts, _ = np.histogram(plx_mas, bins=edges)
    p_bin = counts / float(dist_pc.size)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = p_bin > 0
    return centres[keep], p_bin[keep]


# ---------------------------------------------------------------------------
# the term
# ---------------------------------------------------------------------------

class GaiaTerm:
    """One region's Gaia-congruence term: `ln_gamma` per source, vectorised
    over one class's models.

    Everything region-wide (the Gaia crossmatch, the cloud's canon
    distance and depth) is read once at construction; a class's own model
    register (`G0_FLUX`, `KG_DRAINE`, `KG_WHITNEY`) and its field-star
    parallax marginal (STAR/AGB only) are read the first time that class
    is asked for and cached thereafter -- nothing here re-reads a register
    or a per-region product inside a per-source loop (CODING_RULES.md
    rule 9).
    """

    def __init__(self, config, region):
        self.config = config
        self.region = region
        self._d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
        self._depth_pc = _region_depth_pc(config, region)
        self._gaia = _load_gaia_match(config, region)
        self._registers = {}
        self._star_marginals = {}

    def _register(self, cls):
        cached = self._registers.get(cls)
        if cached is None:
            path = f"{self.config.inputs['sed_models']}/registers/{_REGISTER_FILE[cls]}"
            with h5py.File(path, "r") as f:
                m = f["models"]
                cached = dict(
                    g0_flux=m["G0_FLUX"][:].astype(np.float64),
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

    def _g_s_and_a_x(self, row, cls):
        """The source-and-class-level factors `ln_gamma` needs: `None` for
        `g_s` if the source has no Gaia counterpart at all (the caller's
        signal to return the neutral ln Gamma = 0 without touching a
        model); else `(g_s, A_X)`, `A_X` by class (10_POSTERIOR.md T13):
        `1` if the source is matched but carries no usable parallax
        (`NO_PM`, a non-finite `PLX_MAS`, or a degenerate `sigma_eff`);
        else the cloud-anchored Normal (depth added in quadrature), the
        zero-parallax Normal (GAL), or the field-star marginal (STAR/AGB).
        """
        if not self._gaia["matched"][row]:
            return None, None
        g_s = float(self._gaia["g_s"][row])
        plx = float(self._gaia["plx_mas"][row])
        if self._gaia["no_pm"][row] or not np.isfinite(plx):
            return g_s, 1.0
        sigma_eff = float(self._gaia["e_plx_mas"][row]) * max(1.0, float(self._gaia["ruwe"][row]))
        if not np.isfinite(sigma_eff) or sigma_eff <= 0:
            return g_s, 1.0

        if cls in CLOUD_ANCHORED_CLASSES:
            depth_term_mas = 1000.0 * self._depth_pc / self._d_r_pc ** 2
            sigma_eff = float(np.hypot(sigma_eff, depth_term_mas))
            a_x = float(_normal_pdf(plx, 1000.0 / self._d_r_pc, sigma_eff))
        elif cls == "gal":
            a_x = float(_normal_pdf(plx, 0.0, sigma_eff))
        elif cls in STAR_MARGINAL_CLASSES:
            bins_mas, p_bin = self._star_marginal(cls)
            a_x = float(np.dot(p_bin, _normal_pdf(bins_mas, plx, sigma_eff)))
        else:
            raise ValueError(f"GaiaTerm: no A_X branch defined for class {cls!r}")
        return g_s, a_x

    def ln_gamma(self, rows, model_index, a, log10_b, cls):
        """`ln Gamma_{s,h}` for one source (`rows`, its row index into this
        region's own row-aligned products) over the models named by
        `model_index` (row positions into `cls`'s own register),
        vectorised over models: `a`/`log10_b`, shape `(n_model,)`, are
        those models' own fitted A_K extinction and log10 brightness at
        this source.

        Returns an `(n_model,)` float64 array, one natural-log
        multiplicative factor per model.
        """
        g_s, a_x = self._g_s_and_a_x(rows, cls)
        model_index = np.asarray(model_index)
        n_model = model_index.shape[0]
        if g_s is None:
            # No Gaia counterpart at all: no information, Gamma = 1 for
            # every model, exactly (10_POSTERIOR.md's conditioning
            # discipline -- absent data contributes nothing, not a guess).
            return np.zeros(n_model, dtype=np.float64)

        reg = self._register(cls)
        g0 = reg["g0_flux"][model_index]
        kg_draine = reg["kg_draine"][model_index]
        kg_whitney = reg["kg_whitney"][model_index]

        a = np.asarray(a, dtype=np.float64)
        log10_b = np.asarray(log10_b, dtype=np.float64)

        # Gmag_h: the model's own predicted Gaia G at its fitted (a,
        # log10_b) -- G0_FLUX dimmed by A_G, `a` (A_K) carried to A_G
        # through the model's own KG_DRAINE/KG_WHITNEY, blended at the
        # source's own extinction-ramp weight (prior.selection's diffuse/
        # dense law blend, evaluated directly in A_K units since `a` is
        # already A_K -- no A_V round trip needed).
        with np.errstate(divide="ignore", invalid="ignore"):
            w = selection.law_dense_weight(a)
        r_diffuse = float(selection.ak_per_av(self.config, 0.0))
        r_dense = float(selection.ak_per_av(self.config, 1.0))
        kappa_g = (1.0 - w) * (kg_draine / r_diffuse) + w * (kg_whitney / r_dense)
        a_g = a * kappa_g

        b = 10.0 ** log10_b
        with np.errstate(divide="ignore", invalid="ignore"):
            g_flux = g0 * b * 10.0 ** (-0.4 * a_g)
            gmag = -2.5 * np.log10(g_flux / constants.GAIA_G_VEGA_ZP_MJY)
        h = gaia_detection_weight(gmag)
        # G0_FLUX = NaN (no G-band coverage, not darkness) -> H = 0
        # exactly; a dark model's G0_FLUX = 0 already drives Gmag to +inf
        # and H to 0 through the sigmoid's own limit, no special case.
        h = np.where(np.isnan(g0), 0.0, h)

        gamma = g_s * h * a_x + (1.0 - g_s) * (1.0 - h)
        return np.log(gamma)
