"""
density.py
====================================================================
The quotient-space projector (curation instructions B3(a)-(c)) -- and
ONLY that. [Curation density deletion, this branch] `derive` (the
fixed-bandwidth-KDE/kNN grid-density estimator feeding RHO_K/RHO_2K/
RHO_KDE1/RHO_KDE2/R_1/BOUNDARY) and everything it alone needed
(`DensityResult`, `kde_density`, `ball_counts`, `boundary_flag`, and
the KDE/boundary constants) are DELETED: that was the old design's
library-density weighting, which the current spec forbids, nothing in
`sesnacomplete` outside `studies/` and the retired `sed_fit` reads any
of its outputs, and nothing in the fitter package `sesnaimpute` reads
them either (`RHO_KDE1` is explicitly named in `sesnaimpute`'s own
docstrings as never read, by ruling). It was also 23 of the YSO
register line's 42 minutes and most of its 20 GB peak.

WHAT SURVIVES, AND WHY THIS FILE ISN'T ALSO GONE. `QuotientSpace` /
`build_quotient_space` / `_av_law` / `DEFAULT_LAWS` are a DIFFERENT
thing from `derive`: a cheap (one SVD of an 8x3 matrix) projector
construction that defines a SAMPLING SCALE, not a measured density.
`sesnaimpute.sed_models.sampling.quotient_space` (the shared r-net
sampler every library driver now codes against) and `sed_models.
build_yso`'s duplicate-disclosure check both depend on it for their OWN
purposes, unrelated to grid-density weighting; so do the not-yet-
rewired per-driver duplicates this file predates (`sed_models_curate.
build.{pahc,yso,h2shock}`, `galaxy_curate.read_sampling_quotient_space`,
`sps_curate.build_sps_quotient_space`, `agb_curate.sample_agb_templates`'s
lazy import) -- deleting the whole file would break every one of those
at import time (the first five) or at call time (the last), none of
which this pass's brief named. Kept rather than guessed away; see the
report for the exact list measured.

THE SPACE, for the part that remains. Two things the fit can absorb
must be quotiented out first, or a sampling radius would measure
distance along directions the data cannot see:

  overall brightness  -- the fitter's free scale / distance term
  reddening            -- along an extinction law

So each model's 8-vector x_h = log10 f_ref(i) is projected onto the
orthogonal complement of span{1, kA, kB} (two laws, when both are
given -- `sesnaimpute.sed_models.sampling` instead removes gray and
ONE law, in a WHITENED space, for its own different reason; see that
module).

CONSTRUCTED BY SVD, NOT GRAM-SCHMIDT, deliberately. B3(c) specifies
Gram-Schmidt, but the two laws are nearly collinear in 8-band space --
cos(kA, kB) = 0.999337, i.e. 2.09 degrees apart, singular values
[2.833, 0.167, 0.00365], condition number 776. Gram-Schmidt on
near-collinear vectors loses orthogonality; SVD gives the same subspace
stably.
====================================================================
"""

from dataclasses import dataclass

import numpy as np

from sesnaimpute.sed_models.constants import LIBRARY_SAMPLING_SIGMA_LOG_VECTOR

from sesnaimpute.sed_models.constants import BANDS
from sesnaimpute.sed_models.data_loader import load_extinction_law

__all__ = ["QuotientSpace", "build_quotient_space", "DEFAULT_LAWS", "SIGMA_LOG_CITATION"]

_BAND_KEYS = list(BANDS)
_V_BAND_UM = 0.55
DEFAULT_LAWS = ("draine_rv3.1", "whitney.r550")

#: Provenance string for the default sigma_log, carried to every
#: register header as SIGLGSRC. A citation, not a path and mtime: the
#: vector is a literature constant, so its provenance cannot go stale.
SIGMA_LOG_CITATION = (
    "constants.LIBRARY_SAMPLING_SIGMA_LOG_DEX = 2 x the calibration "
    "floor -- "
    "systematics: 2MASS Skrutskie+2006 AJ 131,1163 (0.010 dex); IRAC "
    "Reach+2005 PASP 117,978 (0.013); MIPS24 Engelbracht+2007 PASP "
    "119,994 (0.017)")


@dataclass(frozen=True)
class QuotientSpace:
    """The projector, shared by every consumer so distances/densities
    measured in it stay comparable."""

    basis: np.ndarray            # (d, 8) orthonormal rows, the kept subspace
    removed: np.ndarray          # (3, 8) orthonormal rows: span{1, kA, kB}
    laws: tuple
    av_law: np.ndarray           # (n_law, 8) as the fitter computes it
    sigma_log: np.ndarray        # (8,)
    sigma_eff: float
    d: int
    sigma_is_placeholder: bool
    conditioning: dict
    sigma_source: str = ""       # provenance of sigma_log, for the header

    def project(self, x):
        """(n, 8) log-flux vectors -> (n, d) coordinates."""
        return np.asarray(x, dtype=float) @ self.basis.T


def _av_law(law_name):
    """-0.4 * chi(lambda)/chi(0.55um) at the BANDS wavelengths.

    Computed from constants.BANDS rather than from a loaded Fitter, so
    this has no model-grid dependency. Identical to the fitter's own
    av_law now that FILTWAV is standardised across libraries (A3); the
    fit-side regression guard verifies that rather than assuming it.
    """
    law = load_extinction_law(law_name)
    order = np.argsort(law.wave_um)
    w, chi = law.wave_um[order], law.opacity_cm2_per_g[order]
    wav = np.array([BANDS[b].wvl_effective_um for b in _BAND_KEYS])
    chi_v = np.interp(_V_BAND_UM, w, chi)
    return -0.4 * np.interp(wav, w, chi) / chi_v


def build_quotient_space(laws=DEFAULT_LAWS, sigma_log=None, sigma_source=""):
    """Build the projector once; share it across every consumer.

    `sigma_log` is the per-band photometric sigma in dex. It defaults to
    `constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR` -- twice the surveys'
    published absolute-calibration floor, which is the model libraries'
    sampling scale (see that constant for the derivation of the factor 2). They are
    literature constants with citations, not a quantity measured from our
    own catalogue, so there is no placeholder mode and no artifact to
    read: the default IS the shipping configuration.

    `sigma_source` is a free string identifying where the vector came
    from; it is carried to the container header and never interpreted
    here. It defaults to the citation.
    """
    if sigma_log is None:
        sigma_log = np.asarray(LIBRARY_SAMPLING_SIGMA_LOG_VECTOR, float)
        sigma_source = sigma_source or SIGMA_LOG_CITATION
    else:
        sigma_log = np.asarray(sigma_log, float)
    placeholder = False
    if sigma_log.shape != (len(_BAND_KEYS),):
        raise ValueError(f"sigma_log must be ({len(_BAND_KEYS)},), got {sigma_log.shape}")

    av = np.array([_av_law(name) for name in laws])
    M = np.column_stack([np.ones(len(_BAND_KEYS))] + [av[i] for i in range(len(laws))])

    # SVD rather than Gram-Schmidt: see module header.
    U, s, _ = np.linalg.svd(M, full_matrices=True)
    rank = int(np.sum(s > s[0] * 1e-12))
    removed = U[:, :rank].T
    basis = U[:, rank:].T
    d = basis.shape[0]

    P = basis.T @ basis
    sigma_eff = float(np.sqrt(np.sum(sigma_log ** 2 * np.diag(P))))

    cos_ab = float(np.dot(av[0], av[1]) /
                   (np.linalg.norm(av[0]) * np.linalg.norm(av[1]))) if len(laws) > 1 else None
    return QuotientSpace(
        basis=basis, removed=removed, laws=tuple(laws), av_law=av,
        sigma_log=sigma_log, sigma_eff=sigma_eff, d=d,
        sigma_is_placeholder=placeholder,
        sigma_source=str(sigma_source),
        conditioning={
            "singular_values": s.tolist(),
            "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
            "rank_removed": rank,
            "cos_angle_laws": cos_ab,
            "angle_laws_deg": (float(np.degrees(np.arccos(np.clip(cos_ab, -1, 1))))
                               if cos_ab is not None else None),
        },
    )
