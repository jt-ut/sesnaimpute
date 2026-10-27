"""
yso_dedup.py
====================================================================
Is a gated YSO model distinguishable from a bare stellar photosphere at
the fitter's resolution? ONE metric answers this exactly once in this
project -- the library build's own sampling metric
(`sesnaimpute.sed_models.sampling`): `quotient_space` removes the
gray/scale and one foreground-extinction direction, `whiten` scales
each band by its sampling sigma, and two models within radius
`R_NET_RADIUS` (1.0, the same radius the global r-net below uses) of
each other are not distinguished. A YSO model within that radius of
its nearest `sps/_raw/` model (the raw, pre-dedup stellar splice,
4,066 rows) is a photosphere duplicate.

**This module answers that question; it no longer decides what to do
with the answer (owner's ruling, 2026-10-20, superseding the
2026-10-08 removal design below).** `duplicate_mask` returns a flag and
a distance, unchanged; the caller (`build_yso.py`'s `run_dedup`/
`build`) used to drop every flagged row before the r-net ever saw it
and now does not drop anything. The YSO class is Stage 0, I, II,
transition disc AND Stage III, and a photosphere's class is decided by
distance, extinction and sky position (the prior's geometry and the
Gaia term), not by its photometry -- so a disk geometry's model that
happens to whiten onto a bare photosphere is the CORRECT statement that
STAR and YSO share that template at this resolution, not an error to
scrub. Dropping it enforced "no two classes share a template", which
was the wrong rule, and its practical cost was silent: a diskless cloud
member in the catalogue was then matched by the nearest edge-on disk
model instead of by the photosphere-like template its own SED actually
calls for, biasing its subclass read toward a disk it does not carry.
The flag and distance this module still computes now ride all the way
into the register (`PHOTOSPHERE_DUPLICATE`/`PHOTOSPHERE_DISTANCE` in
`/models`, `N_MEMBERS_PHOTOSPHERE_DUPLICATE` in the members table) and
override the hard label to CIII (`build_yso.py`'s labeling stage) --
visible to every downstream consumer, rather than invisible by removal.

**Metric provenance, 2026-10-08 (coordinator), superseding this
module's original design.** The original version fit every candidate
against the SPS library via a scale-marginalised sedfitter chi^2
search over A_V in [0, 40] (`sesnacomplete.sed_fit`'s in-memory path,
`fit_sources`/`load_fitterobj`). That is a SECOND, DIFFERENT metric for
the same question, and it failed outright against `sps/_raw/`:
`load_fitterobj`'s `load_ln_w_h` step looks up a model-register density
weight keyed by the model directory's own name, which has no meaning
for a raw, unregistered splice (`sps/_raw/` is dedup's INPUT, not a
curated library product -- `products.register_path` raises `KeyError`
on it, correctly, since nothing registers it). Rather than teach the
retired fitter machinery about an unregistered directory, this module
uses the SAME metric the global r-net already uses: one `cKDTree` over
the 4,066 whitened SPS points, one batched query over the gated YSO
pool. No `sed_fit` import remains here.

**Disclosure, not a correction.** The quotient-space distance is
two-sided: a YSO model whose SED is BLUER than every photosphere
(implying negative extinction relative to its nearest SPS match) can
land within radius exactly as a reddened one would, where a one-sided
A_V >= 0 fit would never have called it a duplicate. This module does
not special-case that direction -- the owner's ruling is to use the
one metric the library build itself uses, not to re-derive a one-sided
version of it for this one phase. The caller (`build_yso.py`) measures
and reports how many flagged models fall on that side.

**Floor and dark-band handling, exactly as the rest of this pipeline
applies the project's B0 floor** (`curve_of_growth.FLOOR_DEX_BELOW_PEAK`,
the same constant `yso_labeling.compute_row_features_from_cubes` and
`sed_models_register.density.derive` use): `sps/_raw/` is a point-source
library (`aperture_dependent = no`, confirmed on disk -- one trivial
aperture, `TOTAL_FLUX` carries no aperture axis), so there is no
curve-of-growth interpolation to do; each row's own floor is simply
its own peak flux across the 8 SESNA bands, ten dex down, and any band at or
below that floor is "dark" in the same sense `yso_labeling`/`sampling`
already use.
====================================================================
"""

import os

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

from sesnaimpute.sed_models import sampling
from sesnaimpute.sed_models.curate import curve_of_growth as cog
from sesnaimpute.sed_models.curate.model_io import align_by_name
from sesnaimpute.sed_models.curate.yso_labeling import ASSEMBLY_BANDS

MODEL_LIB_NAME = "SPS"


def read_sps_flux(sps_raw_dir):
    """Every `sps/_raw/` model's MODEL_NAME and (n, 11) raw per-band flux
    in mJy (ASSEMBLY_BANDS order), clamped at 0 for the same tiny-negative-
    float-noise reason `yso_labeling.read_geometry_flux_cubes` clamps
    raw cubes. One read per band's flat `convolved/{band}.fits`
    (confirmed on disk: no survey subdirectory, ASSEMBLY_BANDS's own short
    names -- the post-rewrite SPS layout, not the historical 2J/2H/2K
    convention), aligned to the first band's MODEL_NAME order."""
    names = None
    flux = None
    for i, band in enumerate(ASSEMBLY_BANDS):
        path = os.path.join(sps_raw_dir, "convolved", f"{band}.fits")
        with fits.open(path, memmap=True) as h:
            band_names = np.asarray(h[1].data["MODEL_NAME"]).astype(str)
            band_flux = np.asarray(h[1].data["TOTAL_FLUX"], dtype=np.float64).reshape(-1)
        if names is None:
            names = band_names
            flux = np.empty((names.size, len(ASSEMBLY_BANDS)), dtype=np.float64)
            flux[:, i] = np.maximum(band_flux, 0.0)
        else:
            order = align_by_name(names, band_names)
            flux[:, i] = np.maximum(band_flux[order], 0.0)
    return names, flux


def sps_quotient_coords(sps_raw_dir):
    """`(names, mu, dark_mask, coords)` for every `sps/_raw/` model:
    `mu`/`dark_mask` under the project's B0 floor (module docstring),
    `coords` their whitened quotient-space position
    (`sampling.whiten`) -- the SAME projector and sampling sigma the
    global r-net uses, so a YSO candidate's distance to the nearest of
    these is directly comparable to `R_NET_RADIUS`."""
    names, flux = read_sps_flux(sps_raw_dir)
    # peak over the eight SESNA bands only (peak_and_floor), as every library's floor
    _, _, floor_linear, _ = cog.peak_and_floor({b: flux[:, i:i + 1] for i, b in enumerate(ASSEMBLY_BANDS)})
    sed_linear = np.maximum(flux, floor_linear[:, None])
    mu = np.log10(sed_linear)
    dark_mask = np.isclose(sed_linear, floor_linear[:, None], rtol=1e-6)
    coords = sampling.whiten(mu, dark_mask)
    return names, mu, dark_mask, coords


def nearest_sps(coords, sps_raw_dir):
    """For each row of `coords` (already-whitened quotient-space
    coordinates, `sampling.whiten` applied by the caller to its own
    mu/dark_mask -- e.g. the gated YSO pool), the distance to and row
    index of its nearest `sps/_raw/` model. One `cKDTree` over the
    4,066 SPS points, one batched `query` over every candidate row --
    seconds, not a per-row fit.

    Returns `(dist, sps_index, sps_names, sps_mu)` -- the last two so a
    caller can run the bluer-than-every-photosphere disclosure check
    without re-reading `sps/_raw/` a second time.
    """
    sps_names, sps_mu, _sps_dark, sps_coords = sps_quotient_coords(sps_raw_dir)
    tree = cKDTree(sps_coords)
    dist, nearest = tree.query(np.ascontiguousarray(coords, dtype=float), k=1, workers=-1)
    return np.atleast_1d(dist), np.atleast_1d(nearest), sps_names, sps_mu


def duplicate_mask(coords, sps_raw_dir, radius):
    """`(dup, dist, nearest, sps_names, sps_mu)`: `dup` is True where a
    candidate's nearest `sps/_raw/` model lies within `radius` (the
    r-net's own `R_NET_RADIUS`) -- a photosphere duplicate. The caller
    carries `dup`/`dist` as a flag through labeling, the r-net and the
    register (owner's ruling, 2026-10-20); this function does not
    remove anything itself."""
    dist, nearest, sps_names, sps_mu = nearest_sps(coords, sps_raw_dir)
    return dist < radius, dist, nearest, sps_names, sps_mu
