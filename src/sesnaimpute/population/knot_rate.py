"""The H2S knot rate `eta_r`, knots per law-predicted young star, formed at
build time against the fitted young-star law (`_W83_design.md`, `population.
yso.law_count`'s `KAPPA_USED`) -- the owner's 2026-09-14 ruling that nothing
here may carry a coefficient-dependent number as a copied literal.

Three fields carry an external H2 knot survey (SPEC_PRIORS.md section 7,
S-D37b): `KNOTS_RAW`, `KNOT_COMPLETENESS` and `FOOTPRINT_AREA_DEG2` below
are their fixed knot-survey facts, independent of the law -- raw recovered
knot counts and completeness (Cygnus X, North America Nebula) or `eps_lim`
depth factor (Vela D) from Froebrich et al. 2015 (UWISH2) and Giannini et
al. 2013 (Vela D), and each field's own survey/SESNA imaging overlap.

`fitted` turns those into a depth-corrected knot rate per field, against
whichever `KAPPA_USED` the law product currently carries (`region_
predicted_yso`, `population.yso.law_count`'s per-source mean times the
footprint area -- the footprint's own HEALPix pixels are not tracked here,
so the mean-density form is used, not `law_area_integral`). `pooled` is the
rate reported to every region without its own knot survey: the geometric
mean of the fitted rates restricted to fields under 1 kpc (Cygnus X, at
1.4 kpc, is excluded -- at that distance the knot survey's completeness
correction is the largest of the three and the region's own predicted
young-star count is dominated by Planck-arm columns, so its fitted rate is
the least reliable of the three to carry to every other region). `eta_for_
region` is what every reader calls: a field's own fitted rate where it has
a survey, else the pooled rate.
"""

import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access
from sesnaimpute.population import yso as yso_module

#: Each field's own knot-survey positional facts (SPEC_PRIORS.md section 7,
#: S-D37b): the survey's own recovered knot count and the completeness
#: (Cygnus X, North America Nebula) or `eps_lim` depth factor (Vela D) that
#: count clears -- both purely geometric/survey facts, independent of the
#: YSO law.
KNOTS_RAW = {
    "Cygnus X": 1077.0,
    "North America Nebula": 255.0,
    "Vela D": 65.0,
}
KNOT_COMPLETENESS = {
    "Cygnus X": 0.885,
    "North America Nebula": 0.865,
    "Vela D": 0.675887,
}

#: Each field's own knot-survey overlap footprint, deg^2 (SPEC_PRIORS.md
#: section 7, S-D37b): UWISH2's own image union intersected with SESNA's
#: imaged area for Cygnus X and North America Nebula; SESNA's imaged area
#: alone for Vela D, where the Giannini search covers the same Spitzer
#: mosaic.
FOOTPRINT_AREA_DEG2 = {
    "Cygnus X": 23.73,
    "North America Nebula": 5.20,
    "Vela D": 1.51,
}

#: The pooled band's own floor, dex: the fitted fields' scatter about the
#: pooled log-rate is reported at least this wide even where the three
#: fields' own fitted rates happen to agree more closely than this.
ETA_BAND_DEX_FLOOR = 0.45

#: Distance below which a field's own fitted rate enters the pooled
#: average (`pooled`) -- Cygnus X, at 1.4 kpc, is excluded; see the module
#: docstring.
POOLED_D_R_PC_MAX = 1000.0


def knots_corrected(region):
    """The field's own depth-corrected knot count -- a knot-survey fact,
    not a law quantity: `KNOTS_RAW / KNOT_COMPLETENESS`."""
    return KNOTS_RAW[region] / KNOT_COMPLETENESS[region]


def _adopted_columns(config, region):
    """`(a_col, provenance)`, every source of `region`, in catalogue row
    order -- the same adopted-column product `population.yso.law_count`
    reads its own copy of."""
    path = config_module.product_path(config, "sky/derived", "adopted",
                                       "column", "source", region=region)
    cols = access.per_source(config, region, path,
                              ["A_COL_K", "A_COL_PROVENANCE"])
    return (np.asarray(cols["A_COL_K"], dtype=float),
            np.asarray(cols["A_COL_PROVENANCE"]))


def region_predicted_yso(config, region):
    """The law's own predicted young-star count on `region`'s knot-survey
    footprint: the region's own catalogued sources' mean law density
    (`population.yso.law_count`, whatever `KAPPA_USED` the law product
    currently carries) times the footprint's own fixed area."""
    a_col, provenance = _adopted_columns(config, region)
    n_law_source = yso_module.law_count(config, region, a_col, provenance)
    return float(np.mean(n_law_source)) * FOOTPRINT_AREA_DEG2[region]


def fitted(config):
    """`{region: eta_r}` for every field carrying a knot survey: each
    field's own depth-corrected knot count over the law's own predicted
    young-star count on that field's footprint -- formed here against
    whatever `KAPPA_USED` the law product currently carries, never a
    copied ratio."""
    return {region: knots_corrected(region) / region_predicted_yso(config, region)
            for region in KNOTS_RAW}


def pooled(config, fitted_rates=None):
    """`(eta_pooled, band_dex)`: `eta_pooled` is the geometric mean of the
    fitted rates restricted to fields under `POOLED_D_R_PC_MAX` (Cygnus X
    excluded; see the module docstring), reported to every region without
    its own knot survey; `band_dex` is the larger of `ETA_BAND_DEX_FLOOR`
    and the rms of those fields' own fitted rates' log10 about the pooled
    log."""
    fitted_rates = fitted(config) if fitted_rates is None else fitted_rates
    pooled_regions = [r for r in fitted_rates
                       if regions_module.REGIONS_BY_NAME[r].d_r_pc < POOLED_D_R_PC_MAX]
    log_rates = np.log10(np.array([fitted_rates[r] for r in pooled_regions]))
    eta_pooled = float(10.0 ** np.mean(log_rates))
    scatter = float(np.sqrt(np.mean((log_rates - np.mean(log_rates)) ** 2)))
    band_dex = max(ETA_BAND_DEX_FLOOR, scatter)
    return eta_pooled, band_dex


def eta_for_region(config, region):
    """`(eta_r, band_dex)`: the field's own fitted rate (band 0.0, exact
    for its own footprint) where `region` carries a knot survey, else the
    pooled rate and its band."""
    if region in KNOTS_RAW:
        return knots_corrected(region) / region_predicted_yso(config, region), 0.0
    return pooled(config)
