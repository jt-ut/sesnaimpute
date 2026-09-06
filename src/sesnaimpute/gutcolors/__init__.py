"""
gutcolors
====================================================================
The Gutermuth+2009 (ApJS 184, 18) colour-cut classification scheme,
applied to an arbitrary 8-band SED.

    (flux[8], sigma[8], valid[8], detected[8])  ->  label       crisp
                                    ->  P(label)         probabilistic

Flux and sigma in mJy; band order from `sesnaimpute.definitions.BANDS`
(J H Ks I1 I2 I3 I4 M1). A model SED is simply a row with sigma = 0 --
nothing special-cases it.

Two entry points:

    classify_crisp(flux, sigma, valid=None, detected=None)
    classify_prob (flux, sigma, valid=None, detected=None)

`classify_prob` is a thin wrapper.  A fitting loop that scores many
model SEDs against ONE source should split it, so that everything the
source's sigmas fix is computed once rather than per model:

    ctx = prepare_source(flux_s, sigma_s, valid=..., detected=...)
    out = classify_prob_batched(ctx, flux_models, sigma_models)

`valid` says which bands the flux vector holds a usable value for;
`detected` says which the survey actually saw.  They coincide for
observed photometry and separate only for an imputed SED.

A third, for callers who want ONE cut rather than the whole cascade --
"is this colour inside the shock wedge, and where do I draw it?":

    regions.get("SHOCK_BLOB")                  a label, gate or term
        .wedge().contains(x, y)                membership in its plane
        .wedge().polygon()                     vertices, for plotting
        .contains_sed(flux, sigma)             exact, from SEDs

Precedence is not applied there; regions.py answers only whether a
region's own conditions hold.  For the label, use classify_crisp.

The probabilistic cascade evaluates each boundary as the probability
that the photometry satisfies it, given the propagated per-band flux
errors, and composes these through Gutermuth's gating order as a chain
rule.  It has no free parameters -- the smoothing scale is the
photometry -- and collapses to the crisp label exactly at sigma = 0.
====================================================================
"""

from sesnaimpute.gutcolors.crisp import LABELS, LABEL_INDEX, Result, classify_crisp, table
from sesnaimpute.gutcolors.featurize import SchemeConfig
from sesnaimpute.gutcolors.prob import (
    ProbResult, classify_prob, classify_prob_batched, prepare_source)
from sesnaimpute.gutcolors import regions
from sesnaimpute.gutcolors.regions import Panel, Region, Wedge
from sesnaimpute.gutcolors.route import ROUTE_NAMES
from sesnaimpute.gutcolors.spec import BAND_ORDER, Table, load

__all__ = [
    "classify_crisp",
    "classify_prob",
    "classify_prob_batched",
    "prepare_source",
    "LABELS",
    "LABEL_INDEX",
    "SchemeConfig",
    "Result",
    "ProbResult",
    "BAND_ORDER",
    "ROUTE_NAMES",
    "Table",
    "load",
    "table",
    "regions",
    "Region",
    "Wedge",
    "Panel",
]
