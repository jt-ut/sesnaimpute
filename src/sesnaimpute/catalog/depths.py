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

Beside the region-wide fit, this module also fits every 2MASS band's OWN
free Poisson fit (the same `_fit_twomass_band`) per NESTED nside-128
HEALPix pixel of the region (briefs/round4_2026-10-20/C6_twomass_pixel_
depths.md "Why": a single region-wide 2MASS depth misses a pixel-to-pixel
spread of up to +-0.27 mag at the extremes measured for Perseus, and runs
away entirely inside Orion A's cluster core). A pixel's own fit replaces
the region's only where it clears four gates (`_pixel_fit_piece`): at
least `MIN_PIXEL_DETECTIONS` detections in that band in that pixel; the
fit converged (enough populated histogram bins, L-BFGS-B success); its
own 50%-completeness magnitude within `MAX_PIXEL_REGIONAL_DELTA_MAG` mag
of the region's; its own fitted width strictly inside `_W_BOUNDS` (pinned
to either edge of the optimizer's own box means the free optimum wanted
to leave the physically sane range, not a genuine interior fit). A pixel
failing any gate carries the region's own `F50_2MASS_MJY`/`WIDTH_DEX`/
`FIT_RESIDUAL_2MASS` copied in verbatim (`FROM_REGIONAL` true there), so
every pixel in the output always carries a usable value.

The per-source pixel id is `healpy.ang2pix(128, GAL_L_DEG, GAL_B_DEG,
nest=True, lonlat=True)` -- the catalogue's own galactic columns, the
SAME (lon, lat) pair `catalog.depth_grid.admitted_pixels_512` and
`granules/build.py` pixelise every other HEALPix axis in this package
with, not the catalogue's RA_DEG/DEC_DEG: `catalog.completeness`'s own
`PIXEL_CURVE_<J|H|Ks>` needs each admitted nside-512 pixel's PARENT
nside-128 pixel by a plain bit shift (`pix512 >> 4`), and that shift is
the correct parent only when both nsides are pixelised from the same
(lon, lat) pair (verified here: `hp.ang2pix(512, gl, gb, nest=True,
lonlat=True) >> 4 == hp.ang2pix(128, gl, gb, nest=True, lonlat=True)` for
every one of 2000 random positions; not true mixing galactic and
equatorial pairs across the two sides, since the two frames tile the
sphere differently).
"""

import functools
import os

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import erfc
# healpy's own import, AFTER scipy.special (measured 2026-10-20: importing
# healpy before `scipy.special.erfc` corrupts loky's pickling of that ufunc
# across worker processes -- BrokenProcessPool, "Can't get attribute 'erfc'
# on <module 'joblib.externals.loky.backend.popen_loky_posix'>" -- import
# order the only variable reproducing it; harmless once erfc is imported
# first).
import healpy as hp

from sesnaimpute import config as config_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run, write_dataset

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

# --- Per-nside-128-pixel 2MASS fit (C6_twomass_pixel_depths.md) ---

NSIDE_128 = 128

#: Gate 1: a pixel's own fit is not attempted below this many detections
#: in that band in that pixel (measured 2026-10-20 by the coordinator:
#: a histogram this thin cannot constrain the free Poisson fit's three
#: shape parameters at pixel scale the way it can at region scale).
MIN_PIXEL_DETECTIONS = 300

#: Gate 3: a pixel's own fitted 50%-completeness magnitude must sit
#: within this many magnitudes of the region's own fitted 50% point, or
#: the fit is treated as a fluke and the regional curve is kept instead.
MAX_PIXEL_REGIONAL_DELTA_MAG = 1.0

_STEM_HPX128 = "depths_sesna_hpx128"
_READINGS_HPX128 = {name: value for (stem, name), value in REGISTRY.items() if stem == _STEM_HPX128}


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


def _region_pixel128_arrays(curated_path):
    """This region's own per-source nside-128 pixel id (module docstring:
    `GAL_L_DEG`/`GAL_B_DEG`, the same (lon, lat) pair every other HEALPix
    axis in this package uses) and the three 2MASS bands' own flux and
    `ORIGIN_FNU` columns, `TWOMASS_KEYS` order."""
    with h5py.File(curated_path, "r") as f:
        gl = np.asarray(f["GAL_L_DEG"][:], dtype=np.float64)
        gb = np.asarray(f["GAL_B_DEG"][:], dtype=np.float64)
        fnu = np.asarray(f["FNU_MJY"][:], dtype=np.float64)
        origin = np.asarray(f["ORIGIN_FNU"][:], dtype=np.int64)
        bands = [b.decode() if isinstance(b, bytes) else b for b in f.attrs["BANDS"]]
    cols = [bands.index(key) for key in TWOMASS_KEYS]
    pix = hp.ang2pix(NSIDE_128, gl, gb, nest=True, lonlat=True).astype(np.int64)
    return pix, fnu[:, cols], origin[:, cols]


def _pixel_fit_piece(det_flux_sorted_by_band, f50_region, residual_region,
                      width_dex_region, m50_region, piece):
    """One (pixel, band) piece: `piece = (band_idx, start, stop)`, `start`/
    `stop` bounding this pixel's own slice of `det_flux_sorted_by_band[
    band_idx]` (that band's detected flux, sorted by pixel so every
    pixel's own detections sit in one contiguous run -- the caller's own
    `np.searchsorted` bounds). Returns `(f50, residual, width_dex,
    n_detected, from_regional)`: the region's own three values, copied in
    verbatim, unless all four gates hold (module docstring), in which
    case this pixel's own fit replaces them.
    """
    band_idx, start, stop = piece
    n_detected = stop - start
    f50 = float(f50_region[band_idx])
    residual = float(residual_region[band_idx])
    width_dex = float(width_dex_region[band_idx])
    from_regional = True
    if n_detected >= MIN_PIXEL_DETECTIONS:
        fit = _fit_twomass_band(det_flux_sorted_by_band[band_idx][start:stop])
        if np.isfinite(fit["f50"]):
            m50_pixel = -2.5 * np.log10(fit["f50"])  # invert f50 = 10**(-0.4 m50)
            w_hat = fit["w"]
            if (abs(m50_pixel - float(m50_region[band_idx])) <= MAX_PIXEL_REGIONAL_DELTA_MAG
                    and _W_BOUNDS[0] < w_hat < _W_BOUNDS[1]):
                from_regional = False
                f50, residual, width_dex = fit["f50"], fit["residual"], DEX_PER_MAG * w_hat
    return f50, residual, width_dex, int(n_detected), from_regional


def _build_pixel_product(config, region, f50_region, residual_region, width_dex_region, st):
    """This region's own per-nside-128-pixel 2MASS depths: every pixel
    that holds at least one catalogue source (any band), the three 2MASS
    bands' own fit or the region's (`_pixel_fit_piece`), pieces over
    `(pixel, band)` on the shared pool (module docstring)."""
    curated_path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    pix, fnu, origin = _region_pixel128_arrays(curated_path)
    pixel_axis = np.unique(pix)
    n_pix = pixel_axis.size
    m50_region = -2.5 * np.log10(np.asarray(f50_region, dtype=np.float64))

    det_flux_sorted_by_band = []
    starts_by_band = []
    stops_by_band = []
    for j in range(len(TWOMASS_KEYS)):
        detected = origin[:, j] == 1
        det_pix = pix[detected]
        det_flux = fnu[detected, j]
        order = np.argsort(det_pix, kind="stable")
        det_pix_sorted = det_pix[order]
        det_flux_sorted_by_band.append(det_flux[order])
        starts_by_band.append(np.searchsorted(det_pix_sorted, pixel_axis, side="left"))
        stops_by_band.append(np.searchsorted(det_pix_sorted, pixel_axis, side="right"))

    pieces = [(j, int(starts_by_band[j][k]), int(stops_by_band[j][k]))
              for j in range(len(TWOMASS_KEYS)) for k in range(n_pix)]
    fn = functools.partial(_pixel_fit_piece, det_flux_sorted_by_band,
                            f50_region, residual_region, width_dex_region, m50_region)
    results = pieces_module.run(fn, pieces, config.n_jobs, st, unit="pixel fits")

    f50_out = np.empty((n_pix, len(TWOMASS_KEYS)), dtype=np.float64)
    residual_out = np.empty((n_pix, len(TWOMASS_KEYS)), dtype=np.float64)
    width_out = np.empty((n_pix, len(TWOMASS_KEYS)), dtype=np.float64)
    n_detected_out = np.empty((n_pix, len(TWOMASS_KEYS)), dtype=np.int32)
    from_regional_out = np.empty((n_pix, len(TWOMASS_KEYS)), dtype=bool)
    idx = 0
    for j in range(len(TWOMASS_KEYS)):
        for k in range(n_pix):
            f50, residual, width_dex, n_det, from_regional = results[idx]
            f50_out[k, j] = f50
            residual_out[k, j] = residual
            width_out[k, j] = width_dex
            n_detected_out[k, j] = n_det
            from_regional_out[k, j] = from_regional
            idx += 1
    return pixel_axis, f50_out, residual_out, width_out, n_detected_out, from_regional_out


def build(config, regions=None):
    """Builds the region-granule 2MASS survey-depths product for the given
    regions (default: every region in `regions.REGIONS`). Reads the curated
    catalogues from `curated.build` (one short read per region, in this
    process), then fits every region's three 2MASS bands as one flat set
    of pieces on the shared pool (`pieces.run`, capped at `config.n_jobs`
    workers). Updates only the given regions' rows of the product
    (`tables.update_rows`). Beside that region-wide product, also builds
    each given region's own per-nside-128-pixel product (module docstring,
    `_build_pixel_product`), one file per region
    (`depths_sesna_hpx128__<Region>.hdf5`).
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

        for r, region in enumerate(regions):
            pix128, f50_pix, residual_pix, width_pix, n_det_pix, from_regional_pix = _build_pixel_product(
                config, region, f50_2mass[r], fit_residual_2mass[r], width_dex_2mass[r], st)
            out_path_128 = config_module.product_path(
                config, "catalog", "sesna", "depths", "hpx128", region=region)
            with h5py.File(out_path_128, "w") as f:
                f.attrs["GRANULE"] = "hpx128"
                f.attrs["REGION"] = region
                write_dataset(f, "HPX_PIX_128", pix128, *_READINGS_HPX128["HPX_PIX_128"])
                write_dataset(f, "F50_2MASS_MJY", f50_pix, *_READINGS_HPX128["F50_2MASS_MJY"])
                write_dataset(f, "WIDTH_DEX", width_pix, *_READINGS_HPX128["WIDTH_DEX"])
                write_dataset(f, "FIT_RESIDUAL_2MASS", residual_pix, *_READINGS_HPX128["FIT_RESIDUAL_2MASS"])
                write_dataset(f, "N_DETECTED", n_det_pix, *_READINGS_HPX128["N_DETECTED"])
                write_dataset(f, "FROM_REGIONAL", from_regional_pix, *_READINGS_HPX128["FROM_REGIONAL"])

            fitted = ~from_regional_pix
            for j, key in enumerate(TWOMASS_KEYS):
                n_fitted = int(fitted[:, j].sum())
                n_fallback = int(from_regional_pix[:, j].sum())
                if n_fitted:
                    delta = (-2.5 * np.log10(f50_pix[fitted[:, j], j])
                             - (-2.5 * np.log10(f50_2mass[r, j])))
                    lo, hi = np.percentile(delta, [16.0, 84.0])
                    spread = "m50_pixel - m50_regional 16-84%% = %.3f..%.3f mag" % (lo, hi)
                else:
                    spread = "no fitted pixels"
                print("catalog.depths: %s %s hpx128: %d/%d pixels fitted their own curve, "
                      "%d fell back to the regional curve, %s"
                      % (region, key, n_fitted, n_fitted + n_fallback, n_fallback, spread))


if __name__ == "__main__":
    run(build)
