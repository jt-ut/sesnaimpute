"""Per-library terms for the Bayesian model-selection posterior.

REGISTER: LIBRARY (spec 3.1). This package holds every quantity that is
a property of a MODEL LIBRARY alone -- derivable from flux.fits /
parameters.fits / classmap.fits plus the survey band set, with no
source, no cell, and no fit in the derivation. What belongs: the
survey-matched reference SED f_ref and its numerical floor, synthetic
Gaia G0/kG, band darkness, the PAHC `pahc_fstar` reference table, the
sampling-scale quotient-space projector (`density.build_quotient_space`
-- cheap, an SVD of an 8x3 matrix; shared with `sesnaimpute.sed_models.
sampling`'s own, differently-whitened one), and the build/read of the
container those become -- `<library>_register.hdf5`, one per library,
written beside that library. What does NOT belong: anything per-source
(A_s, sightline parameters), anything per-cell (headcounts N_C, shapes
lambda~), the colour-concordance matrix T_{g,c} (census- and
survey-dependent, and global rather than per-library), and every
analysis-register decision -- those live in `census` and `bmselect`.

[Curation density deletion] The grid-density quadrature estimator that
used to live here (RHO_KDE1 production, RHO_KDE2/RHO_K/RHO_2K
diagnostics, the BOUNDARY grid-edge flag, `density.derive`) is GONE:
the old design's library-density weighting, forbidden by the current
spec, read by nothing outside `studies/` and the retired `sed_fit`, and
23 of the YSO register line's 42 minutes / most of its 20 GB peak.

DEPENDENCY RULE: `sed_fit` imports interfaces FROM `census`, `gaia`,
`gut_colors`, and `sed_models_register`; the science packages NEVER import
`sed_fit`. The fitter is a tool this register hands terms to, not a
thing this register knows about.
"""
