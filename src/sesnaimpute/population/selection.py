"""Whether a source clears SESNA's own catalog cut, the hybrid
extinction law the test dims by, and the compiled kernels that turn a
class's external population into a source's exact pass fraction
(SPEC_PRIORS.md section 1.3, IMPLEMENTATION.md section 4).

SESNA catalogs a source when at least two of its eight bands, dimmed by
its own dust column, clear the source's detection limit
(`catalog.limits.limits`). The dimming coefficient per band is not one
fixed law: it blends smoothly, in log column, from a diffuse-ISM curve
below A_K = 0.5 to a dense-cloud curve above A_K = 1.0
(`law_dense_weight`, `kappa_hybrid`).
"""

import os

import numpy as np

from sesnaimpute import definitions

BAND_KEYS = tuple(b.key for b in definitions.BANDS)

#: SESNA's catalog inclusion rule: >= 2 of 8 bands above the local limit.
MIN_BANDS = 2

#: SPEC_PRIORS.md 1.3 -- the two named laws the ramp blends between.
#: `draine_rv3.1` is Weingartner & Draine (2001) / Draine (2003), R_V = 3.1.
#: `whitney.r550` is cited as Indebetouw et al. (2005, ApJ 619, 931)'s
#: IRAC-adjusted Kim, Martin & Hendry (1994) model via Robitaille; the
#: packaged file's 4.5 um A_lambda/A_K is 0.260, not the ~0.43 the
#: citation implies, and no ascii file matching the citation is reachable
#: from the sedfitter docs, the sedfitter or hyperion-rt/paper-2017-sed-models
#: GitHub repositories, or the Robitaille models_r06 FTP release (the
#: repositories carry only this same file and the un-adjusted KMH94
#: curve, 0.269 at 4.5 um; the FTP release bundles a law file only inside
#: a 5.3 GB model tarball). The mismatch stands uncorrected.
LAW_DIFFUSE = "draine_rv3.1"
LAW_DENSE = "whitney.r550"

#: The ramp's domain, A_K magnitudes: 0 (diffuse law only) at and below
#: LAW_RAMP_LO, 1 (dense law only) at and above LAW_RAMP_HI. No literature
#: source for these two numbers is found in SPEC_PRIORS.md, the bms_review
#: docs, or the predecessor packages; none is cited here because none
#: exists in the searched material.
LAW_RAMP_LO = 0.5
LAW_RAMP_HI = 1.0

_V_BAND_UM = 0.55

_LAW_CACHE = {}
_K_CACHE = {}

#: The shared scaled-extinction ladder a source's exact selection is
#: tabulated on: `x = a / A_s`. The last four points cover the column
#: kernel's tail past the source's own column; `1.7` was added (owner,
#: 2026-09-06) because the tail past it held 5-10% of the kernel's mass
#: and interpolated at 6-8% error before the point was there.
X_LADDER = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.4, 1.7, 2.0, 2.8])


def _parse_info(text):
    fields = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(None, 1)
        fields[key] = value.strip()
    return fields


#: The two laws ship as package data, not a downloaded product -- the
#: extinction curve is used for more than A_K (owner, 2026-09-08), so it
#: belongs with the code that reads it.
_LAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "extinction_laws")


def _load_law_curve(law):
    """`(wave_um, opacity_cm2_per_g)`, sorted ascending in wavelength,
    read from the package's own `data/extinction_laws/<law>/<law>.par`
    at the columns named in the sibling `<law>.info` file -- the same
    file pair and column convention the SED fitter's own law loader
    reads. Fails with one sentence when either file is missing.
    """
    if law in _LAW_CACHE:
        return _LAW_CACHE[law]
    law_dir = os.path.join(_LAW_DATA_DIR, law)
    info_path = os.path.join(law_dir, f"{law}.info")
    par_path = os.path.join(law_dir, f"{law}.par")
    if not os.path.isfile(info_path) or not os.path.isfile(par_path):
        raise ValueError(
            f"population.selection: no extinction law {law!r} packaged at {law_dir!r} "
            f"-- reinstall sesnaimpute")
    with open(info_path) as f:
        info = _parse_info(f.read())
    colidx_wav = int(info["colidx_wav"])
    colidx_extinction = int(info["colidx_extinction"])
    raw = np.loadtxt(par_path, usecols=(colidx_wav, colidx_extinction))
    order = np.argsort(raw[:, 0])
    wave_um, opacity = raw[order, 0], raw[order, 1]
    _LAW_CACHE[law] = (wave_um, opacity)
    return _LAW_CACHE[law]


def extinction_k(config, law):
    """`k_i = chi(lambda_i) / chi(0.55um)` for the 8 census bands, from
    the law's own tabulated curve, linear interpolation in wavelength.
    `config` is unused -- the law is package data, not a data-root product.
    """
    if law not in _K_CACHE:
        wave_um, opacity = _load_law_curve(law)
        wav = np.array([b.wvl_um for b in definitions.BANDS])
        _K_CACHE[law] = (np.interp(wav, wave_um, opacity)
                        / np.interp(_V_BAND_UM, wave_um, opacity))
    return _K_CACHE[law].copy()


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
    # a == 0 is the ruled Z = 0 case (callable.py): log(0) = -inf, clipped
    # to 0.0 two lines down, so let it through quietly instead of the
    # spurious "divide by zero" RuntimeWarning.
    with np.errstate(divide="ignore"):
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


