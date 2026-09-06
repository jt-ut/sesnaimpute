"""Psi: Gutermuth's colour-cascade concordance, tempered by beta.

10_POSTERIOR.md Sec 1: the posterior carries a factor `Psi_{s,h}^beta`,
"a likelihood-like factor, beta a dial." This module supplies the two
things the fitter and the imputation stage read from it:

`PsiTerm.ln_psi` -- the per-model log contribution at fit time,
`beta * ln(Psi_{s,h})`, for one source, vectorised over the models of
one class's library. `Psi_{s,h}` is the cascade's own probability
(`gutcolors.prob`) that a model's predicted flux, dimmed by the
hypothesis's fitted extinction and scaled by its fitted brightness,
reads as this term's class under Gutermuth's cascade.

`verdicts` -- the cascade's verdict-probability vector read from a
region's own catalogue photometry, per source, for the imputation
stage to carry forward and aggregate (`sesnacomplete.bms_posterior.
impute.build_T` aggregated this survey-wide; that aggregation is not
this module's job).

`beta = 0` is the constant default (`sesnacomplete.bms_posterior.
impute.impute_source`'s own `beta: float = 0.0` -- the call that pins
`apply_psi`'s tempering off): at `beta = 0`, `ln_psi` returns exactly
zero for every model without evaluating the cascade at all, so a
zero, not-even-finite `Psi` can never produce a `0 * -inf` nan.
"""

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute.gutcolors import crisp
from sesnaimpute.gutcolors import prob as gc_prob
from sesnaimpute.prior import selection

#: beta = 0: Psi structurally disabled (module docstring).
BETA_DEFAULT = 0.0

#: Which of Gutermuth's eleven verdict categories (`constants.
#: GUTERMUTH_LABELS`, the cascade's own `id` column) count as
#: concordant with a model of each fitted class. AGB carries none --
#: Gutermuth's scheme has no category for a dusty evolved-star
#: photosphere -- so its concordant set is empty and Psi is 1
#: (ln_psi 0) for every AGB model, at any beta.
CONCORDANT_LABELS = {
    "STAR": ("DISKLESS_STAR",),
    "AGB": (),
    "PAHC": ("PAH_APERTURE",),
    "GAL": ("AGN", "PAH_GALAXY", "GENERIC_GALAXY"),
    "YSO": ("DEEPLY_EMBEDDED", "CLASS_I", "CLASS_II", "TRANSITION_DISK"),
    "H2S": ("SHOCK_BLOB",),
}

_N_BANDS = len(definitions.BANDS)


class PsiTerm:
    """`beta * ln(Psi_{s,h})` for one fitted class, one source, vectorised
    over models.

    The dimming vector is the single diffuse-cloud law
    (`prior.selection.LAW_DIFFUSE`), not the fitter's own per-source
    diffuse/dense ramp (`prior.selection.kappa_hybrid`): `ln_psi` carries
    no per-source ramp weight, and the cascade's verdict on a model's
    colours is not sensitive to which of the two similar near/mid-
    infrared curves supplies the K-normalised dimming vector.
    """

    def __init__(self, config, cls, beta=BETA_DEFAULT):
        if cls not in CONCORDANT_LABELS:
            raise ValueError(
                f"PsiTerm: unknown class {cls!r}, must be one of {tuple(CONCORDANT_LABELS)}")
        self.config = config
        self.cls = cls
        self.beta = float(beta)
        # K-normalised per-band dimming vector, BANDS order.
        self._kappa = selection.kappa_ak(config, selection.LAW_DIFFUSE)
        self._label_idx = np.array(
            [crisp.LABEL_INDEX[name] for name in CONCORDANT_LABELS[cls]], dtype=np.int64)
        # This class's own valid/detected mask never changes: every band
        # is a model prediction, so every band is valid, and there is no
        # survey to have missed one.
        placeholder_flux = np.ones((1, _N_BANDS))
        placeholder_sigma = np.zeros((1, _N_BANDS))
        all_true = np.ones((1, _N_BANDS), dtype=bool)
        self._ctx = gc_prob.prepare_source(
            placeholder_flux, placeholder_sigma, valid=all_true, detected=all_true)

    def ln_psi(self, rows, model_index, a, log10_b):
        """`beta * ln(Psi_{s,h})` for the models `rows[model_index]`,
        this source already fitted to `(a, log10_b)` at each model.

        Parameters
        ----------
        rows : ndarray, shape (n_library, 8)
            This class's library of native model fluxes, mJy,
            `definitions.BANDS` order, at zero extinction and the
            library's own brightness reference.
        model_index : ndarray, shape (n_models,)
            Which library rows this source's models are.
        a : ndarray, shape (n_models,)
            K-band extinction (mag) fitted for each model
            (10_POSTERIOR.md Sec 1).
        log10_b : ndarray, shape (n_models,)
            log10 of the fitted brightness scale (10_POSTERIOR.md
            Sec 1, `log10 B = -2 SC`).

        Returns
        -------
        ndarray, shape (n_models,)
        """
        model_index = np.asarray(model_index)
        n_models = model_index.shape[0]
        if self.beta == 0.0:
            return np.zeros(n_models, dtype=float)

        native_flux = np.asarray(rows, dtype=float)[model_index]
        a = np.asarray(a, dtype=float)
        log10_b = np.asarray(log10_b, dtype=float)

        # Dim by this model's fitted extinction (magnitudes, K currency)
        # and scale by its fitted brightness -- the model's own predicted
        # flux at its hypothesis, the same construction the fitter's
        # sweep already applies to score chi-squared.
        scale = 10.0 ** log10_b
        dimming = 10.0 ** (-0.4 * a[:, None] * self._kappa[None, :])
        dimmed_scaled_flux = native_flux * scale[:, None] * dimming
        zero_sigma = np.zeros_like(dimmed_scaled_flux)

        result = gc_prob.classify_prob_batched(self._ctx, dimmed_scaled_flux, zero_sigma)
        if self._label_idx.size == 0:
            psi = np.ones(n_models, dtype=float)
        else:
            psi = result.prob[:, self._label_idx].sum(axis=1)
        with np.errstate(divide="ignore"):
            ln_psi_val = np.log(psi)
        return self.beta * ln_psi_val


def verdicts(config, region):
    """`Q_s`, the cascade's verdict-probability vector (11 Gutermuth
    categories, `gutcolors.crisp.LABELS` order) for every source in
    `region`'s curated catalogue, read from that source's own real
    photometry: `ORIGIN_FNU == 1` bands are the survey's own
    detections, every other band is a valid but undetected
    completeness-limit substitute (`catalog.curated`'s own
    substitution scheme).

    This is what `sesnacomplete.bms_posterior.impute.build_T`
    aggregated survey-wide into the translation matrix `T`; here it is
    returned per source, plain, for the imputation stage to carry
    forward and aggregate.

    Returns
    -------
    ndarray, shape (n_sources, 11)
    """
    path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        flux = np.asarray(f["FNU_MJY"][:], dtype=float)
        sigma = np.asarray(f["SIGMA_FNU_MJY"][:], dtype=float)
        origin = np.asarray(f["ORIGIN_FNU"][:])
    detected = origin == 1
    valid = np.ones_like(detected, dtype=bool)
    result = gc_prob.classify_prob(flux, sigma, valid=valid, detected=detected)
    return result.prob
