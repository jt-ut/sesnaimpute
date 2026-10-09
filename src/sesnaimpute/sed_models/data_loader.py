"""
data_loader.py
====================================================================
Single home for reading everything under the package's `data/` resource
directory. Lives as a sibling of `constants.py` rather than inside any one
subpackage (e.g. sed_models_curate), since `data/` holds package-wide resources,
not resources specific to one subpackage -- same reasoning `constants.py`
already gives for keeping shared, non-scheme-specific things at this level.

Uses `importlib.resources` (not a `__file__`-relative path) so lookups work
whether the package is running from an editable install (current state) or
a properly built/installed wheel -- see the filter-curve package-data entry
in pyproject.toml for the other half of that.
====================================================================
"""

import importlib.resources
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilterCurve:
    """Raw, unnormalized transmission curve for one photometric filter.

    wave_um / response are exactly as published by the source (see
    data/filter_curves/README.md) -- wavelength unit-converted to micron,
    but transmission left untouched. Normalization (e.g. so the integral
    over frequency is 1, which sedfitter's convolution requires) is a
    downstream processing choice specific to one consumer, not applied
    here.
    """
    name: str
    wave_um: np.ndarray
    response: np.ndarray


def load_filter_curve(name: str) -> FilterCurve:
    """
    Load the raw transmission curve for filter `name` (e.g. "J", "I1", "G")
    from data/filter_curves/{name}.dat. Call this once per filter needed --
    there is no bundled "load all filters" helper; callers wanting several
    just call this multiple times.
    """
    resource = (
        importlib.resources.files("sesnaimpute.sed_models")
        .joinpath("data", "filter_curves", f"{name}.dat")
    )
    if not resource.is_file():
        available = sorted(
            p.name[:-4] for p in
            importlib.resources.files("sesnaimpute.sed_models").joinpath("data", "filter_curves").iterdir()
            if p.name.endswith(".dat")
        )
        raise FileNotFoundError(
            f"No filter curve named {name!r} in data/filter_curves/. "
            f"Available: {available}"
        )

    with resource.open("r") as f:
        raw = np.loadtxt(f)

    return FilterCurve(name=name, wave_um=raw[:, 0], response=raw[:, 1])


@dataclass(frozen=True)
class ExtinctionLaw:
    """Extinction law: opacity vs wavelength, for use with sedfitter's
    `Extinction` object.

    wave_um / opacity_cm2_per_g come from the two columns of
    data/extinction/{name}/{name}.par identified by colidx_wav /
    colidx_extinction in the sibling {name}.info file (see
    data/extinction/README.md) -- used exactly as published, no
    renormalization applied here. label is the law's conventional
    short-hand name (what sedfitter setups traditionally store in a
    results table to record which law a fit used), distinct from `name`
    (the data/extinction/ directory key used to load it).

    av_over_ak / av_over_ak_curve are TWO DIFFERENT NUMBERS -- do not use
    them interchangeably:

    - `av_over_ak_curve` is THE PROJECT'S CANONICAL (A_V/A_K) for this
      law: chi(0.55um) / chi(Ks_wavelength_um), both read off THIS
      object's own `wave_um`/`opacity_cm2_per_g` curve by the same linear
      interpolation convention sedfitter's `Extinction.get_av` uses (see
      `sed_fit/docs/sedfitter_behavior.md` Q4). It is precomputed and
      committed in `data/extinction/{name}/derived_constants.json`
      (owner ruling 2026-08-18) so that the K-currency identity
      kappa_Ks == 1 holds exactly -- use THIS value in any computation
      that mixes A_V-currency and A_K-currency quantities for this law.
    - `av_over_ak` is the law file's PUBLISHED `AV_over_AK` SCALAR
      (`{name}.info`) -- PROVENANCE ONLY. It disagrees with
      `av_over_ak_curve` by a few percent (see `derived_constants.json`'s
      `frac_disagreement_curve_vs_info_scalar`) because it was not
      derived from this package's tabulated curve. Kept for comparison
      and history; must not be used in any dimming computation.
    """
    name: str
    label: str
    wave_um: np.ndarray
    opacity_cm2_per_g: np.ndarray
    av_over_ak: float
    av_over_ak_curve: float


def _parse_extinction_info(text):
    """Parse a `{name}.info` key/value file: whitespace-separated
    `key value...` lines, blank lines and lines starting with `#` ignored.
    """
    fields = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(None, 1)
        fields[key] = value.strip()
    return fields


def load_extinction_law(law: str) -> ExtinctionLaw:
    """
    Load extinction law `law` (e.g. "whitney.r550", "draine_rv3.1") from
    data/extinction/{law}/{law}.par, using the column indices and AV/AK
    ratio recorded in the sibling {law}.info file, plus the canonical
    curve-internal AV/AK recorded in the sibling `derived_constants.json`
    (see `ExtinctionLaw.av_over_ak_curve`). Call this once per law
    needed -- there is no bundled "load all laws" helper.
    """
    law_dir = importlib.resources.files("sesnaimpute.sed_models").joinpath("data", "extinction", law)
    info_resource = law_dir.joinpath(f"{law}.info")
    par_resource = law_dir.joinpath(f"{law}.par")
    derived_resource = law_dir.joinpath("derived_constants.json")

    if not info_resource.is_file() or not par_resource.is_file():
        extinction_dir = importlib.resources.files("sesnaimpute.sed_models").joinpath("data", "extinction")
        available = sorted(p.name for p in extinction_dir.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"No extinction law named {law!r} in data/extinction/. "
            f"Available: {available}"
        )
    if not derived_resource.is_file():
        raise FileNotFoundError(
            f"No derived_constants.json for extinction law {law!r} in "
            f"data/extinction/{law}/ -- av_over_ak_curve (the canonical "
            f"AV/AK; owner ruling 2026-08-18) has nowhere to come from."
        )

    with info_resource.open("r") as f:
        info = _parse_extinction_info(f.read())
    with derived_resource.open("r") as f:
        derived = json.load(f)

    colidx_wav = int(info["colidx_wav"])
    colidx_extinction = int(info["colidx_extinction"])

    with par_resource.open("r") as f:
        raw = np.loadtxt(f, usecols=(colidx_wav, colidx_extinction))

    return ExtinctionLaw(
        name=law,
        label=info["name"],
        wave_um=raw[:, 0],
        opacity_cm2_per_g=raw[:, 1],
        av_over_ak=float(info["AV_over_AK"]),
        av_over_ak_curve=float(derived["av_over_ak_curve"]),
    )
