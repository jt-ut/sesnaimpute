"""Survey depth: the 2MASS 50%-completeness flux and roll-off width, per
region and band (SPEC_PRIORS.md section 1.3).

2MASS (J, H, Ks) carries no per-source completeness map (COMPLETENESS_
2026-10-20.md), so this module's own free Poisson fit -- power-law counts
times an error-function completeness roll-off, `N(m) = A * 10**(alpha*m) *
C(m)`, `C(m) = 0.5 * erfc((m - m50) / (sqrt(2) * w))`, `A`, `alpha`, `m50`
and `w` all free, fit by Poisson likelihood on the detected-flux histogram
(`ORIGIN_FNU == 1`, 0.1-mag bins from the bright end to 2 mag beyond the
faintest detection) -- is still every consumer's one source of the 2MASS
bands' 50%-completeness flux and roll-off width: `F50_2MASS_MJY =
10**(-0.4 m50)` (mJy), `WIDTH_DEX = 0.4 * w` (the effective roll-off every
non-detection term reads, SPEC_BMSTP_DRAFT.md section 6.2).

The five Spitzer bands (I1-I4, M1) no longer fit anything here: Rob
Gutermuth's own measured completeness maps (`catalog.completeness`,
COMPLETENESS_2026-10-20.md) replace this module's old Spitzer-band fit
outright (`DELTA_DEX`, `W_MAG`, `SUBSTITUTED` and their Spitzer-only
`WIDTH_DEX`/`SIGMA_WIDTH_DEX` columns retired with it) -- a genuine
per-pixel, per-band measured curve needs no region-wide, map-relative
roll-off fit standing in for it.

Each fitted 2MASS width is reported (not enforced) against the 2MASS
Point Source Catalog's own intrinsic roll-off width -- `(m50 - m99) /
2.326` from its documented completeness curve (2MASS Explanatory
Supplement section VI.5a.i, Table 2 and Figures 3-5; Cutri et al. 2006) --
0.095 / 0.10 / 0.074 dex for J / H / Ks -- since an effective width cannot
be narrower than the survey's own roll-off; a region-band below that
floor is flagged in the build's printout.
"""

import os

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import erfc

from sesnaimpute import config as config_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run

_STEM = "depths_sesna_region"
_READINGS = {name: value for (stem, name), value in REGISTRY.items() if stem == _STEM}

TWOMASS_KEYS = ("J", "H", "Ks")

# Histogram bin width, in magnitude (SPEC_PRIORS.md section 1.3).
MAG_BIN = 0.1

# A histogram with fewer populated bins than this cannot constrain the
# model's three free shape parameters and is not fit.
MIN_POPULATED_BINS = 4

# 1/2.5: converts a magnitude offset to the dex offset of the flux it
# implies (the Pogson magnitude-flux relation).
DEX_PER_MAG = 0.4

_ALPHA_BOUNDS = (-2.0, 2.0)
_W_BOUNDS = (0.02, 3.0)

# The 2MASS Point Source Catalog's own intrinsic completeness-roll-off
# width, in dex, `(m50 - m99) / 2.326` from its documented completeness
# curve (2MASS Explanatory Supplement section VI.5a.i, Table 2 and
# Figures 3-5; Cutri et al. 2006), keyed by TWOMASS_KEYS order (J, H,
# Ks): the floor a region's fitted 2MASS width is checked against, since
# an effective roll-off cannot be narrower than the survey's own.
TWOMASS_INTRINSIC_WIDTH_DEX = (0.095, 0.10, 0.074)


def _completeness(m, m50, w):
    """The error-function completeness roll-off, `C(m50) = 0.5`."""
    return 0.5 * erfc((m - m50) / (np.sqrt(2.0) * w))


def _model_free(m, alpha, m50, w):
    """`N(m)` shape (up to `A`) with `m50` and `w` both free: no per-source
    map to peg to, so `m` is each detection's own absolute magnitude."""
    return 10.0 ** (alpha * m) * _completeness(m, m50, w)


def _poisson_nll_and_amplitude(f, n):
    """The Poisson negative log likelihood of histogram counts `n` against
    shape `f`, with the amplitude `A` profiled out analytically (the
    score equation for `A` sets the model's total count to the data's).
    """
    f = np.clip(f, 1e-300, None)
    total_n = float(n.sum())
    total_f = float(f.sum())
    if not total_f > 0:
        return np.inf, np.nan
    amplitude = total_n / total_f
    mu = np.clip(amplitude * f, 1e-300, None)
    nll = float(np.sum(mu - n * np.log(mu)))
    return nll, amplitude


def _bin_histogram(m):
    """Bins a magnitude-like array into 0.1-mag bins from the bright end
    to 2 mag beyond the faintest value.
    """
    lo = float(m.min())
    hi = float(m.max()) + 2.0
    n_bins = max(1, int(np.ceil((hi - lo) / MAG_BIN)))
    edges = lo + MAG_BIN * np.arange(n_bins + 1)
    counts, _ = np.histogram(m, bins=edges)
    centers = edges[:-1] + 0.5 * MAG_BIN
    return centers, counts.astype(float)


def _mag_histogram(detected_flux):
    """The detected-flux histogram in absolute magnitude."""
    return _bin_histogram(-2.5 * np.log10(detected_flux))


def _fit_free(centers, counts):
    """Poisson-likelihood fit of `(alpha, m50, w)`, all free."""
    m50_guess = centers[int(np.argmax(counts))] + 0.5

    def nll(theta):
        alpha, m50, w = theta
        f = _model_free(centers, alpha, m50, w)
        val, _ = _poisson_nll_and_amplitude(f, counts)
        return val

    x0 = np.array([0.3, m50_guess, 0.3])
    m50_bounds = (centers.min() - 2.0, centers.max() + 5.0)
    bounds = (_ALPHA_BOUNDS, m50_bounds, _W_BOUNDS)
    return minimize(nll, x0=x0, method="L-BFGS-B", bounds=bounds)


def _relative_l1_residual(centers, counts, model_values, reference_mag):
    """The relative L1 distance between the fitted model and the
    histogram, restricted to the roll-off region: bins fainter than
    `reference_mag - 1` mag.
    """
    mask = centers >= (reference_mag - 1.0)
    denom = float(counts[mask].sum())
    if not denom > 0:
        return float("nan")
    return float(np.sum(np.abs(model_values[mask] - counts[mask])) / denom)


def _fit_twomass_band(detected_flux):
    """One 2MASS band's free fit: `F50`, its fit residual, and the fitted
    roll-off width in magnitude (SPEC_BMSTP_DRAFT.md section 6.2)."""
    centers, counts = _mag_histogram(detected_flux)
    if np.count_nonzero(counts > 0) < MIN_POPULATED_BINS:
        return dict(f50=np.nan, residual=np.nan, w=np.nan)

    res = _fit_free(centers, counts)
    if not res.success:
        return dict(f50=np.nan, residual=np.nan, w=np.nan)

    alpha_hat, m50_hat, w_hat = res.x
    model_values = _model_free(centers, alpha_hat, m50_hat, w_hat)
    _, amplitude = _poisson_nll_and_amplitude(model_values, counts)
    residual = _relative_l1_residual(centers, counts, amplitude * model_values, m50_hat)
    f50 = 10.0 ** (-DEX_PER_MAG * m50_hat)
    return dict(f50=f50, residual=residual, w=w_hat)


def _region_detected_twomass(curated_path):
    """One region's own detected flux, each of the three 2MASS bands."""
    with h5py.File(curated_path, "r") as f:
        fnu = f["FNU_MJY"][:]
        origin = f["ORIGIN_FNU"][:]
        bands = [b.decode() if isinstance(b, bytes) else b for b in f.attrs["BANDS"]]
    return [fnu[origin[:, bands.index(key)] == 1, bands.index(key)] for key in TWOMASS_KEYS]


def build(config, regions=None):
    """Builds the region-granule 2MASS survey-depths product for the given
    regions (default: every region in `regions.REGIONS`). Reads the curated
    catalogues from `curated.build` (one short read per region, in this
    process), then fits every region's three 2MASS bands as one flat set
    of pieces on the shared pool (`pieces.run`, capped at `config.n_jobs`
    workers). Updates only the given regions' rows of the product
    (`tables.update_rows`).
    """
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]

    with progress_module.Stage("catalog.depths") as st:
        band_pieces = []  # one detected-flux array per region per 2MASS band, region x band order
        for region in regions:
            p = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"catalog.depths.build: curated catalogue missing for region {region!r} "
                    f"at {p!r} -- run the curated-catalogue RUNBOOK line for it"
                )
            band_pieces += _region_detected_twomass(p)

        n_regions = len(regions)
        n_twomass = len(TWOMASS_KEYS)
        fit_results = pieces_module.run(_fit_twomass_band, band_pieces, config.n_jobs, st, unit="band fits")

        results = []
        for r in range(n_regions):
            row = fit_results[r * n_twomass:(r + 1) * n_twomass]
            results.append((
                np.array([x["f50"] for x in row]),
                np.array([x["residual"] for x in row]),
                DEX_PER_MAG * np.array([x["w"] for x in row]),
            ))

        f50_2mass = np.array([r[0] for r in results])
        fit_residual_2mass = np.array([r[1] for r in results])
        width_dex_2mass = np.array([r[2] for r in results])

        # Report-only floor check (never repaired): a fitted 2MASS width
        # narrower than the Point Source Catalog's own intrinsic roll-off
        # (TWOMASS_INTRINSIC_WIDTH_DEX) flags that region-band, since an
        # effective width cannot be narrower than the survey's own.
        below_floor = width_dex_2mass < np.array(TWOMASS_INTRINSIC_WIDTH_DEX)[None, :]

        out_path = config_module.product_path(config, "catalog", "sesna", "depths", "region")
        tables_module.update_rows(
            out_path,
            regions,
            {
                "F50_2MASS_MJY": f50_2mass,
                "FIT_RESIDUAL_2MASS": fit_residual_2mass,
                "WIDTH_DEX": width_dex_2mass,
            },
            granule="region",
            readings=_READINGS,
        )
        for j, key in enumerate(TWOMASS_KEYS):
            n_below = int(below_floor[:, j].sum())
            if n_below:
                flagged = [regions[i] for i in np.flatnonzero(below_floor[:, j])]
                print("catalog.depths: %d region(s) fit a %s width below the 2MASS PSC's "
                      "intrinsic roll-off (%.3f dex): %s"
                      % (n_below, key, TWOMASS_INTRINSIC_WIDTH_DEX[j], flagged))
        st.done(out_path, regions=len(regions),
                median_f50_mjy=float(np.nanmedian(f50_2mass)),
                median_width_dex=float(np.nanmedian(width_dex_2mass)),
                below_intrinsic_floor=int(below_floor.sum()))


if __name__ == "__main__":
    run(build)
