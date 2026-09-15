"""The protostar check: HOPS (Orion A) and eHOPS (Aquila) against the
prior (SPEC_BMSTP_DRAFT.md sec. 5.5's "Check (report only)"; sec. 9's
protostar-fraction row). Report-only: nothing here feeds the prior or the
posterior. Per region, two panels against `sky.derived.protostars`'s
Herschel-confirmed protostar sample:

Panel 1, the verdict: assuming a protostar is a SESNA source with its own
4.5 micron datum, at its own point of the nuisance plane (its scaled
extinction x, its dereddened 4.5 micron flux) the prior's class posterior
`P(C | x, datum, s) = Sum_b Lambda_C[x, b] L(b) / Sum_C' Sum_b Lambda_C'[x,
b] L(b)`, `Lambda_C` the protostar's own SESNA counterpart's prior grid
(`atlas.shapes.lambda_grids`), `L(b)` the datum's one likelihood factor at
each brightness cell -- a Gaussian in log10 F for a measurement, the
survey's own non-detection probability below the pixel's 4.5 micron limit
otherwise (sec. 6.2). Panel 2, the map: the current intrinsic YSO density
map with the protostars overplotted (unchanged).

`CLASS` is read only for this report -- rule 7 ("never read a
classification label in anything that feeds a prior") does not apply,
since nothing computed here is a prior input.
"""

import argparse
import os

import astropy.units as u
import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib.colors import Normalize
from scipy.special import erf

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import plot_style
from sesnaimpute import progress
from sesnaimpute.atlas import captions
from sesnaimpute.atlas import shapes as shapes_module
from sesnaimpute.atlas.render import (
    LABEL_FONTSIZE, _add_panel, _footprint_geometry, _log_norm, _panel_colorbar, _reproject)
from sesnaimpute.bmstp import grid
from sesnaimpute.fittp import likelihood as likelihood_module
from sesnaimpute.population import kernel as kernel_module
from sesnaimpute.population import selection as population_selection

NSIDE_512 = 512
PAGE_W_IN = 16.0

#: The page title and its one-line subtitle (owner's ruling 2026-09-13):
#: bigger than the panel titles, since a reader meets the page here first
#: -- the two atlas pages' own convention (`atlas.shapes`).
_TITLE_FONTSIZE = 18
_SUBTITLE_FONTSIZE = 15

#: The three catalogue classes the check uses (SPEC_BMSTP_DRAFT.md sec.
#: 5.5's Class 0/I/flat protostellar population); the catalogue's own
#: Class II rows are excluded and only counted.
CLASS_USED = (b"0", b"I", b"flat")

#: The nearest-neighbour match radius to the curated catalogue (this
#: brief): inside it a protostar takes that source as its SESNA
#: counterpart; the atlas's own pixel (nside 512, 6.9 arcmin on a side)
#: is far coarser, so this is a position match, not a pixel one.
MATCH_RADIUS_ARCSEC = 2.0

#: Dunham et al. 2014, ApJ 783, 29, Table 1: Class 0+I+flat over all
#: young stellar objects, pooled over the c2d+Gould Belt clouds -- the
#: literature protostellar fraction SPEC_BMSTP_DRAFT.md's H2S section
#: (sec. 5.6) treats as a constant.
DUNHAM2014_PROTOSTELLAR_FRACTION = 0.27

BAND_KEYS = tuple(b.key for b in definitions.BANDS)
IDX_I2 = BAND_KEYS.index("I2")

_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))

#: Panel 1's marker SHAPE per class (colour is reserved for P(young star |
#: ...)); a not-measured protostar overrides this with a downward triangle
#: regardless of class (drawn at its own detection limit).
_CLASS_MARKER = {b"0": "o", b"I": "s", b"flat": "^"}
_NOT_MEASURED_MARKER = "v"
#: Panel 1's own open marker for a protostar whose fitted foreground
#: extinction sits far above its sightline's own column.
_BEYOND_REACH_MARKER = "o"
#: Panel 1's own open marker for a protostar with an unconstrained
#: foreground, `AV_FOREGROUND_MAG <= 0` (owner, 2026-09-12 evening ruling
#: 2) -- distinct from `_BEYOND_REACH_MARKER` since it is a different
#: reason to draw with no measured depth fraction of its own.
_NO_FOREGROUND_MARKER = "s"

#: The legend's own words per class (owner's 2026-09-13 ruling, item 4):
#: every panel's class legend reads "Class 0"/"Class I"/"flat" (never
#: "flat" as "Class flat"); the map panel's own legend also carries the
#: catalogue's Class II rows, counted there but never fed to the verdict.
_CLASS_LABEL = {b"0": "Class 0", b"I": "Class I", b"flat": "flat", b"II": "Class II"}

#: The fine grid this brief's item 2 reads the truncated-mixture median
#: and interval off (dex, in log10 ξ or log10 ξi_hat): 0.005 dex, this
#: brief's own choice.
_GRID_STEP_DEX = 0.005
#: `P(T >= a_p)` below this bar: too far past the edge ξ = 1 for the kernel to
#: place credible mass there -- BEYOND THE KERNEL'S REACH (this brief's
#: item 2), drawn at the edge ξ = 1 with an open marker and no interval.
_BEYOND_REACH_P = 0.01


def _pix512_galactic(ra_deg, dec_deg):
    """Each position's nside-512 galactic NESTED pixel -- the granule
    map's own pixelisation (`granules/build.py`), reached here by one
    ICRS -> galactic rotation (never a local approximation, sec. 8)."""
    gal = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    return hp.ang2pix(NSIDE_512, gal.l.deg, gal.b.deg, nest=True, lonlat=True)


def _read_protostars(config, region):
    """This region's rows of `sky.derived.protostars`'s pooled HOPS/eHOPS
    sample (`REGION` already the admitted-footprint match, sec. 3), the
    4.5 micron datum included."""
    path = f"{config.data_root}/sky/derived/protostars/protostars_survey.hdf5"
    if not os.path.exists(path):
        raise ValueError(
            f"atlas.protostars: no {path!r} -- run RUNBOOKtp.sh's "
            f"'PY sesnaimpute.sky.derived.protostars' line")
    with h5py.File(path, "r") as f:
        region_col = f["REGION"][:]
        mask = region_col == region.encode("utf-8")
        return dict(
            ra_deg=f["RA_DEG"][:][mask], dec_deg=f["DEC_DEG"][:][mask],
            cls=f["CLASS"][:][mask], av_mag=f["AV_FOREGROUND_MAG"][:][mask],
            survey=f["SURVEY"][:][mask],
            f45_mjy=f["F45_MJY"][:][mask].astype(np.float64),
            e_f45_mjy=f["E_F45_MJY"][:][mask].astype(np.float64),
            f45_measured=f["F45_MEASURED"][:][mask])


def _read_curated(config, region):
    """The curated catalogue's own `NAME`, position (`atlas.protostars`'s
    match target) -- `bmstp.atlas._observed_bright_counts`'s own read."""
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        name = f["NAME"][:]
        ra_deg = np.asarray(f["RA_DEG"][:], dtype=np.float64)
        dec_deg = np.asarray(f["DEC_DEG"][:], dtype=np.float64)
    return name, ra_deg, dec_deg


def _read_density_rows(config, region):
    """P1's per-source rows this check needs to place a protostar's
    matched counterpart in the prior: `NAME` (the alignment check below),
    `A_COL_K` (the row's own sightline BEAM column, sec. 3.2's own
    extinction column -- this check's own `A_beam`), `F_LIM_50_MJY` (the
    row's own per-band 50% limit, sec. 6.2's `F_lim,50`), and the column
    kernel's own per-row terms `A_COL_SIG_K`, `ARM`, `ZP_SIG_K`
    (`population.kernel.Kernel.mixture`) plus `SIGHTLINE_ROW` (this
    brief's item 5, the median source's own `XI_MARGINAL` row)."""
    path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    with h5py.File(path, "r") as f:
        name = f["NAME"][:]
        a_col = np.asarray(f["A_COL_K"][:], dtype=np.float64)
        f_lim_50_mjy = np.asarray(f["F_LIM_50_MJY"][:], dtype=np.float64)
        a_col_sig = np.asarray(f["A_COL_SIG_K"][:], dtype=np.float64)
        arm = f["ARM"][:]
        zp_sig = np.asarray(f["ZP_SIG_K"][:], dtype=np.float64)
        sightline = np.asarray(f["SIGHTLINE_ROW"][:], dtype=np.int64)
    return name, a_col, f_lim_50_mjy, a_col_sig, arm, zp_sig, sightline


def _read_xi_marginal(config, region, sightline_row):
    """The median source's own `XI_MARGINAL` row (P3, `bmstp.shapes`): the
    YSO `x` marginal at its sightline -- this brief's item 5, the kernel
    validation's own `x_member` distribution."""
    path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    with h5py.File(path, "r") as f:
        return np.asarray(f["XI_MARGINAL"][int(sightline_row)], dtype=np.float64)


def _norm_cdf(z):
    """`Phi(z)`, the standard normal CDF, off the exact `erf` (rule 18:
    `scipy.special` imported at module top, a compiled extension)."""
    return 0.5 * (1.0 + erf(z / _SQRT2))


#: Bisection iterations `_position_distribution` runs to invert the
#: analytic mixture CDF: each halves the bracket, so 60 of them narrow
#: any bracket to float64 precision (2^-60 of its own width) -- cheap,
#: since every iteration is one vectorised pass over all protostars, no
#: python loop over sources (rule 8).
_BISECT_ITERS = 60


def _mixture_cdf(z, w, mean0, mean1, sigma0, sigma1):
    """`P(log10 ξ <= z)`, the untruncated two-component mixture CDF, in
    closed form (`_norm_cdf`, exact `erf`) -- never a grid integral of the
    pdf (see `_position_distribution`'s own docstring for why that broke)."""
    return w * _norm_cdf((z - mean0) / sigma0) + (1.0 - w) * _norm_cdf((z - mean1) / sigma1)


def _position_distribution(log10_xi_hat, w, mu, sigma):
    """`(median, lo16, hi84, p_reach, beyond_reach)`: the read's own
    picture of a protostar's position in the DISTANCE coordinate (this
    brief's item 2). `log10 ξ = log10 ξi_hat - y`, `y` the two-component
    cloud-class column kernel mixture in dex (`w`, `mu` (n, 2), `sigma`
    (n, 2), `Kernel.mixture`'s own `exponent=2` reweighting, the star-gas
    law's own exponent) -- a mixture of Gaussians in `log10 ξ` itself,
    mean `log10 ξi_hat - mu_i`, the same `sigma_i`, weight `w_i` / `1 - w_i`.
    `p_reach = P(log10 ξ <= 0)` (`T >= a_p`) is the kernel's own
    UNTRUNCATED mass there; `median`/`lo16`/`hi84` are the TRUNCATED
    mixture's own 50%/16%/84% points, `CDF(z) = target * p_reach` INVERTED
    by bisection on the exact closed-form CDF (rule 8: every iteration is
    one array pass over all protostars, no python loop over sources) --
    not a fixed-grid trapezoidal pdf integral: a protostar whose floored
    `log10 ξi_hat` sits at the array's own low edge can have BOTH kernel
    components' means at or below a finite grid's own edge (a component's
    mean can itself be more negative than `log10 ξi_hat` when its own `mu` is
    positive), so a grid starting there silently integrates less than the
    component's own mass and its cumulative sum never reaches `p_reach` --
    the bisection here has no edge, so it cannot lose mass that way.
    `beyond_reach` marks `p_reach` below this brief's own 0.01 bar."""
    log10_xi_hat = np.asarray(log10_xi_hat, dtype=np.float64)
    mean0 = log10_xi_hat - mu[:, 0]
    mean1 = log10_xi_hat - mu[:, 1]
    sigma0, sigma1 = sigma[:, 0], sigma[:, 1]
    p_reach = w * _norm_cdf(-mean0 / sigma0) + (1.0 - w) * _norm_cdf(-mean1 / sigma1)

    # A bracket guaranteed to hold every root: comfortably below both
    # component means (`CDF` there is ~0) up to the edge ξ = 1 itself (`CDF(0)
    # = p_reach`, at or above every target below).
    span = 20.0 * np.maximum(sigma0, sigma1)
    lo_bracket = np.minimum(mean0, mean1) - span

    def _solve(target):
        lo = lo_bracket.copy()
        hi = np.zeros_like(lo_bracket)
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            go_right = _mixture_cdf(mid, w, mean0, mean1, sigma0, sigma1) < target
            lo = np.where(go_right, mid, lo)
            hi = np.where(go_right, hi, mid)
        return 0.5 * (lo + hi)

    median = _solve(0.5 * p_reach)
    lo16 = _solve(0.16 * p_reach)
    hi84 = _solve(0.84 * p_reach)
    beyond_reach = p_reach < _BEYOND_REACH_P
    return median, lo16, hi84, p_reach, beyond_reach


def _kernel_validation(log10_xi_hat, w, mu, sigma, xi_marginal, xi_centers, beyond_reach):
    """`(emp_median, emp_p84, pred_median, pred_p84, frac_beyond)`: this
    brief's item 5, the kernel's own prediction for a cloud member's beam
    ratio -- `log10 ξi_hat = log10 ξ_member + y`, `x_member` from the
    region's median-sightline YSO `x` marginal (`XI_MARGINAL`, shared
    across protostars), `y` from each protostar's OWN class kernel (`w`,
    `mu`, `sigma`) -- against the sample's own empirical `log10 ξi_hat`. The
    predicted CDF pools each protostar's own kernel CDF into ONE shared
    function `G` first (rule 8: an (n_proto, n_u) array, averaged over
    protostars), then convolves that one function with the shared
    `xi_marginal` (an (n_z, n_xi_centers) array) -- never the full
    (n_proto, n_xi_centers, n_grid) product."""
    u_grid = np.arange(-3.0, 3.0 + 1e-9, _GRID_STEP_DEX)
    z0 = (u_grid[None, :] - mu[:, 0][:, None]) / sigma[:, 0][:, None]
    z1 = (u_grid[None, :] - mu[:, 1][:, None]) / sigma[:, 1][:, None]
    cdf_y = w[:, None] * _norm_cdf(z0) + (1.0 - w[:, None]) * _norm_cdf(z1)
    g_pooled = cdf_y.mean(axis=0)  # (n_u,): the ensemble-pooled kernel CDF

    total = float(xi_marginal.sum())
    x_marg = xi_marginal / total if total > 0.0 else xi_marginal
    z_grid = np.arange(-4.0, 2.0 + 1e-9, _GRID_STEP_DEX)
    offsets = z_grid[:, None] - xi_centers[None, :]
    g_vals = np.interp(offsets.ravel(), u_grid, g_pooled, left=0.0, right=1.0).reshape(offsets.shape)
    cdf_pooled = (g_vals * x_marg[None, :]).sum(axis=1)

    pred_median = float(np.interp(0.5, cdf_pooled, z_grid))
    pred_p84 = float(np.interp(0.84, cdf_pooled, z_grid))
    n = log10_xi_hat.shape[0]
    emp_median = float(np.median(log10_xi_hat)) if n else float("nan")
    emp_p84 = float(np.percentile(log10_xi_hat, 84.0)) if n else float("nan")
    frac_beyond = float(np.mean(beyond_reach)) if beyond_reach.size else float("nan")
    return emp_median, emp_p84, pred_median, pred_p84, frac_beyond


def _read_w_dex_i2(config, region):
    """The region's own I2 (4.5 micron) roll-off width, `catalog.
    depth_grid`'s `W_DEX_PIX` (sec. 6.2) -- one value per band for the
    region, broadcast identically to every admitted pixel there
    (`bmstp.atlas._depth_grid`'s own read); this check takes any one row's
    entry, not a per-pixel join, since it is the region's own constant."""
    path = config_module.product_path(config, "catalog", "sesna", "depth-grid", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        return float(f["W_DEX_PIX"][0, IDX_I2])


def _read_prior_atlas(config, region):
    """P6's admitted nside-512 pixels, the cataloged YSO share (the
    position line, sec. 5.5's caption (ii)), and the young-star law's own
    AREA INTEGRAL per pixel `INTENSITY_YSO` (deg^-2) with each pixel's own
    survey `COVERAGE` -- Panel C's own map and its `N_YSO_prior` count
    (sec. 5.5), never the per-source law read at the sources' own columns
    (`DENSITY_YSO`, which is not the pixel's area integral). Sorted by
    pixel for the `searchsorted` joins below."""
    path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    if not os.path.exists(path):
        raise ValueError(
            f"atlas.protostars: no {path!r} -- run RUNBOOKtp.sh's "
            f"'PY sesnaimpute.bmstp.atlas' line")
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        share_yso = np.asarray(f["SHARE_YSO"][:], dtype=np.float64)
        intensity_yso = np.asarray(f["INTENSITY_YSO"][:], dtype=np.float64)
        coverage = np.asarray(f["COVERAGE"][:], dtype=np.float64)
    order = np.argsort(pix)
    return pix[order], share_yso[order], intensity_yso[order], coverage[order]


def _read_density_table_for_map(config, region):
    """P1's per-source `HPX_512`: each catalogued source's own pixel, the
    position line's own denominator (`_position_line`'s `density_hpx512`
    argument, item 6 of the 2026-09-13 owner ruling). Panel C's own map
    and count read the atlas product's own `INTENSITY_YSO`/`COVERAGE`
    directly (`_read_prior_atlas`) and no longer need this per-source
    table's `DENSITY_YSO`."""
    path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    with h5py.File(path, "r") as f:
        return np.asarray(f["HPX_512"][:], dtype=np.int64)


def _join(atlas_pix, atlas_values, query_pix):
    """`(values, found)`: `atlas_values` at each `query_pix`'s own atlas
    row (`atlas_pix` sorted), `found` False where `query_pix` is not one
    of the atlas's admitted pixels -- one `searchsorted`, no per-item
    loop."""
    loc = np.minimum(np.searchsorted(atlas_pix, query_pix), atlas_pix.size - 1)
    found = atlas_pix[loc] == query_pix
    values = np.full(query_pix.shape, np.nan, dtype=atlas_values.dtype)
    values[found] = atlas_values[loc[found]]
    return values, found


def _match_to_catalogue(config, region, ra_deg, dec_deg):
    """Each protostar's own density-table ROW (the section 3 "Match"):
    nearest curated-catalogue source within `MATCH_RADIUS_ARCSEC`; failing
    that, the nearest catalogued source sharing the protostar's own
    nside-512 pixel (its sightline stand-in); failing that, excluded.
    Returns `(density_row, direct, standin, excluded, name_mismatch)`,
    `density_row` -1 where excluded. The density table's row order is
    verified against the curated catalogue's own (both read at build from
    the same curated source list); a mismatch falls back to a match by
    `NAME` and is disclosed via `name_mismatch`."""
    name_cat, ra_cat, dec_cat = _read_curated(config, region)
    name_dens, a_col_dens, f_lim_dens, _, _, _, _ = _read_density_rows(config, region)

    aligned = (name_cat.size == name_dens.size) and np.array_equal(name_cat, name_dens)
    if aligned:
        cat_to_density = np.arange(name_cat.size, dtype=np.int64)
    else:
        idx_by_name = {n: i for i, n in enumerate(name_dens.tolist())}
        cat_to_density = np.array([idx_by_name.get(n, -1) for n in name_cat.tolist()], dtype=np.int64)

    coord_proto = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    coord_cat = SkyCoord(ra=ra_cat * u.deg, dec=dec_cat * u.deg, frame="icrs")
    idx_nn, sep2d, _ = coord_proto.match_to_catalog_sky(coord_cat)
    direct = sep2d.arcsec <= MATCH_RADIUS_ARCSEC

    density_row = np.full(ra_deg.size, -1, dtype=np.int64)
    density_row[direct] = cat_to_density[idx_nn[direct]]

    # The sightline stand-in: the nearest catalogued source sharing the
    # protostar's own nside-512 pixel, for every protostar the direct
    # match missed. The loop below is over the DISTINCT PIXELS among
    # those (typically a handful), not over sources; each iteration is
    # itself vectorised over the sources and candidates it holds.
    need_standin = ~direct
    standin = np.zeros(ra_deg.size, dtype=bool)
    if np.any(need_standin):
        pix_proto = _pix512_galactic(ra_deg, dec_deg)
        pix_cat = _pix512_galactic(ra_cat, dec_cat)
        order = np.argsort(pix_cat)
        pix_cat_sorted = pix_cat[order]
        for p in np.unique(pix_proto[need_standin]):
            lo = np.searchsorted(pix_cat_sorted, p, side="left")
            hi = np.searchsorted(pix_cat_sorted, p, side="right")
            if hi == lo:
                continue  # no catalogued source shares this pixel: stays excluded
            cand = order[lo:hi]
            rows_in_pix = np.where(need_standin & (pix_proto == p))[0]
            sep = coord_cat[cand][:, None].separation(coord_proto[rows_in_pix][None, :]).arcsec
            nearest = cand[np.argmin(sep, axis=0)]
            density_row[rows_in_pix] = cat_to_density[nearest]
            standin[rows_in_pix] = True

    excluded = density_row < 0
    return density_row, direct, standin, excluded, (not aligned), a_col_dens, f_lim_dens


def _likelihood_factor(f45_mjy, e_f45_mjy, measured, a_p, kappa_i2, f_lim_i2, w_i2, b_centers):
    """`L(b)` per protostar per brightness cell (SPEC_BMSTP_DRAFT.md sec.
    6.2, this brief's item 3): a Gaussian in log10 F for a measured datum,
    the survey's own non-detection probability below the pixel's I2 limit
    otherwise -- ONE array expression; `measured` only selects between the
    two already-computed arrays (rule 8, no branch on it). 47 of the 502
    pooled protostars (7 HOPS, 40 eHOPS) carry a measured `F45_MJY` but no
    `E_F45_MJY` (the VOTable's own uncertainty field masked though the
    flux is not) -- `sigma`'s own `E_F45/(F45 ln 10)` term drops to zero
    there rather than NaN, so the half-cell floor alone sets it."""
    log10_f_b = b_centers[None, :] - 0.4 * a_p[:, None] * kappa_i2[:, None]  # (n, n_b)

    safe_f45 = np.where(measured, f45_mjy, 1.0)
    log10_f45_dered = np.log10(safe_f45) + 0.4 * a_p * kappa_i2
    e_term = np.where(np.isfinite(e_f45_mjy), e_f45_mjy, 0.0) / (safe_f45 * np.log(10.0))
    sigma = np.maximum(e_term, 0.5 * grid.D_LOG10_F45)
    z_meas = (log10_f45_dered[:, None] - b_centers[None, :]) / sigma[:, None]
    l_measured = np.exp(-0.5 * z_meas * z_meas) / (sigma[:, None] * _SQRT2PI)

    z_nondet = (log10_f_b - np.log10(f_lim_i2)[:, None]) / (_SQRT2 * w_i2)
    l_not_measured = np.exp(likelihood_module._ln_one_minus_c(z_nondet))

    return np.where(measured[:, None], l_measured, l_not_measured)


def _verdict(config, region, protostars, used, density_row, idx_median):
    """Section 3's "The verdict": `P(C | xi_hat, datum, s)` for the six
    classes at every used protostar's own matched row, read at the
    MEASURED cell `log10 ξi_hat` from the blurred grids (`blur=True`). Also
    returns each protostar's own measured depth fraction `log10 ξi_hat`
    itself (`log10_xii_hat`, item 4 of the 2026-09-13 owner ruling), the
    page's own x coordinate; `p(log10 ξ | xi_hat, cloud kernel)`'s own
    reach test (`_position_distribution`'s `p_reach`) still marks which
    protostars sit beyond the kernel's reach, but its median/interval no
    longer feed the page (nothing there is drawn from the truncated
    mixture any more).

    A protostar with `AV_FOREGROUND_MAG <= 0` (owner, 2026-09-12 evening
    ruling 2) is an UNCONSTRAINED foreground, the fit's own lowest grid
    point, not a measurement: it has no depth fraction of its own at
    all. Its verdict is instead the DEPTH-MARGINALISED one, `P(C |
    datum, s)` with `Lambda_C` summed over the `x` cells inside the
    support (the same blurred read, marginalised over `x`) against the
    same 4.5 micron likelihood factor `L(b)`; the page places it at the
    median `log10_xii_hat` of the protostars that do have a fitted
    foreground (`_draw_figure`, not here), and it is excluded from the
    kernel check's own empirical/predicted comparison (it carries no
    fitted `log10 ξi_hat` to check)."""
    ra_u = protostars["ra_deg"][used]
    av_u = protostars["av_mag"][used]
    f45_u = protostars["f45_mjy"][used]
    e45_u = protostars["e_f45_mjy"][used]
    measured_u = protostars["f45_measured"][used]
    rows_u = density_row[used]
    no_fg = av_u <= 0.0

    (_, a_col_dens, f_lim_dens, a_col_sig_dens, arm_dens, zp_sig_dens,
     sightline_dens) = _read_density_rows(config, region)
    w_i2 = _read_w_dex_i2(config, region)

    # A_beam, sec. 3.2's own extinction column (A_COL_K): xi_hat = a_p / A_beam.
    a_col_row = a_col_dens[rows_u]
    w_ramp_col = population_selection.law_dense_weight(a_col_row)
    ak_per_av_col = population_selection.ak_per_av(config, w_ramp_col)
    a_p = av_u * ak_per_av_col
    xi_hat = a_p / a_col_row
    # Floored at the array's own low edge (`grid.LOG10_XI_EDGES[0]`) purely
    # as a numerical guard against `log10(0)` for a `no_fg` row -- that
    # row's own `log10_xii_hat` is never drawn at its own value (ruling 2,
    # the page places it at the median of the rows that do have one) and
    # its `log10_xi_hat` never enters the kernel check.
    xi_hat = np.maximum(xi_hat, 10.0 ** grid.LOG10_XI_EDGES[0])
    log10_xi_hat = np.log10(xi_hat)
    n_past_wall = int(np.count_nonzero(xi_hat[~no_fg] > 1.0))

    # The cloud-class column kernel at each protostar's own matched row
    # (`Kernel.mixture`'s own fitted within-beam tilt, `Kernel.
    # cloud_gamma_herschel` -- protostars are a cloud population, the
    # SAME 2-D joint fit `population.kernel._fit_cloud_gamma_sigma_
    # herschel` runs on this very sample, never a literal exponent):
    # `y`, `log10 T = log10 A_beam + y`.
    kernel = kernel_module.Kernel.read(config)
    w_mix, mu_mix, sigma_mix = kernel.mixture(
        a_col_row, a_col_sig_dens[rows_u], arm_dens[rows_u], zp_sigma_k=zp_sig_dens[rows_u],
        exponent=kernel.cloud_gamma_herschel)
    (_, _, _, p_reach, beyond_reach) = _position_distribution(log10_xi_hat, w_mix, mu_mix, sigma_mix)
    # `no_fg` rows carry no fitted position at all (ruling 2, not merely
    # a "beyond reach" one) and take their own category on the page, drawn
    # at a placement `_draw_figure` computes from `log10_xii_hat` directly.
    beyond_reach = beyond_reach & ~no_fg

    # The verdict's own cell: the MEASURED coordinate `log10 ξi_hat`, which
    # may sit above the edge ξ = 1 in the padding -- clipped only at the
    # array's own TOP edge (`grid.LOG10_XI_EDGES[-1]`, +1.0 dex), never at
    # the edge ξ = 1 (item 3): this is what the fitter reads. `no_fg` rows are
    # excluded below (their own P(C|datum,s) is depth-marginalised, not
    # read at this cell), so their cell index is unused, not counted.
    n_x_full = grid.LOG10_XI_EDGES.size - 1
    cell_x = np.clip(np.searchsorted(grid.LOG10_XI_EDGES, log10_xi_hat, side="right") - 1, 0, n_x_full - 1)
    n_top_clipped = int(np.count_nonzero(log10_xi_hat[~no_fg] > grid.LOG10_XI_EDGES[-1]))

    w_ramp_p = population_selection.law_dense_weight(a_p)
    kappa_i2 = population_selection.kappa_hybrid(config, w_ramp_p)[:, IDX_I2]
    f_lim_i2 = f_lim_dens[rows_u, IDX_I2]

    b_centers = 0.5 * (grid.LOG10_F45_EDGES[:-1] + grid.LOG10_F45_EDGES[1:])
    l_b = _likelihood_factor(f45_u, e45_u, measured_u, a_p, kappa_i2, f_lim_i2, w_i2, b_centers)

    lambda_verdict, _, class_order = shapes_module.lambda_grids(config, region, rows_u, blur=True)
    lambda_at_xp = lambda_verdict[np.arange(rows_u.size), :, cell_x, :]  # (n_used, 6, n_b)
    numerator_c = (lambda_at_xp * l_b[:, None, :]).sum(axis=2)  # (n_used, 6)
    denom = numerator_c.sum(axis=1)
    p_c = numerator_c / denom[:, None]

    if np.any(no_fg):
        # The DEPTH-MARGINALISED verdict (ruling 2): Lambda_C summed over
        # the x cells inside the support (the same blurred read), never
        # read at one measured cell, since `no_fg` carries no fitted `x`.
        lambda_support = lambda_verdict[no_fg][:, :, :grid.N_XI_SUPPORT, :].sum(axis=2)  # (n_no_fg, 6, n_b)
        numerator_nofg = (lambda_support * l_b[no_fg][:, None, :]).sum(axis=2)  # (n_no_fg, 6)
        denom_nofg = numerator_nofg.sum(axis=1)
        p_c = p_c.copy()
        p_c[no_fg] = numerator_nofg / denom_nofg[:, None]

    idx_yso = class_order.index("YSO")
    p_yso = p_c[:, idx_yso]
    leading = np.array(class_order)[np.argmax(p_c, axis=1)]
    leading_is_yso = leading == "YSO"

    y_dered_limit = np.log10(f_lim_i2) + 0.4 * a_p * kappa_i2
    log10_f45_dered = np.log10(np.where(measured_u, f45_u, 1.0)) + 0.4 * a_p * kappa_i2
    y_plot = np.where(measured_u, log10_f45_dered, y_dered_limit)

    # The kernel validation (item 5): the region's median-sightline YSO x
    # marginal against each protostar's own column kernel, pooled --
    # `no_fg` rows carry no fitted `log10 ξi_hat` (ruling 2) and are excluded
    # from both sides of this check.
    xi_marginal = _read_xi_marginal(config, region, sightline_dens[idx_median])
    xi_centers = 0.5 * (grid.LOG10_XI_EDGES[:-1] + grid.LOG10_XI_EDGES[1:])
    fg = ~no_fg
    emp_median, emp_p84, pred_median, pred_p84, frac_beyond = _kernel_validation(
        log10_xi_hat[fg], w_mix[fg], mu_mix[fg], sigma_mix[fg], xi_marginal, xi_centers, beyond_reach[fg])

    return dict(
        log10_xii_hat=log10_xi_hat,
        p_reach=p_reach, beyond_reach=beyond_reach, no_foreground=no_fg,
        y_plot=y_plot, p_yso=p_yso,
        leading_is_yso=leading_is_yso, measured=measured_u, n_past_wall=n_past_wall,
        n_top_clipped=n_top_clipped, class_order=class_order, rows_u=rows_u,
        n_beyond_reach=int(np.count_nonzero(beyond_reach)),
        n_no_foreground=int(np.count_nonzero(no_fg)),
        emp_median=emp_median, emp_p84=emp_p84, pred_median=pred_median, pred_p84=pred_p84,
        frac_beyond_reach=frac_beyond,
        frac_yso_leads=float(np.mean(leading_is_yso)) if leading_is_yso.size else float("nan"),
        median_p_yso=float(np.median(p_yso)) if p_yso.size else float("nan"),
        frac_yso_leads_measured=(float(np.mean(leading_is_yso[measured_u]))
                                  if np.any(measured_u) else float("nan")),
        median_p_yso_measured=(float(np.median(p_yso[measured_u]))
                                if np.any(measured_u) else float("nan")),
        frac_yso_leads_not_measured=(float(np.mean(leading_is_yso[~measured_u]))
                                      if np.any(~measured_u) else float("nan")),
        median_p_yso_not_measured=(float(np.median(p_yso[~measured_u]))
                                    if np.any(~measured_u) else float("nan")),
        frac_yso_leads_no_foreground=(float(np.mean(leading_is_yso[no_fg]))
                                       if np.any(no_fg) else float("nan")),
    )


def _position_line(config, region, protostars_pix, atlas_pix, share_yso, density_hpx512):
    """The stage's own printed position line: the prior's cataloged YSO
    share's median at the protostars' own pixels against every cataloged
    source's own pixel (off the page, item 6 of the 2026-09-13 owner
    ruling)."""
    share_proto, found = _join(atlas_pix, share_yso, protostars_pix)
    share_proto = share_proto[found]
    share_source, found_src = _join(atlas_pix, share_yso, density_hpx512)
    share_source = share_source[found_src]
    median_proto = float(np.median(share_proto)) if share_proto.size else float("nan")
    median_source = float(np.median(share_source)) if share_source.size else float("nan")
    return median_proto, median_source, int(np.count_nonzero(~found))


def _panel_c(atlas_pix, intensity_yso, coverage, n_proto_footprint):
    """The map of the prior's own per-pixel young-star INTENSITY (deg^-2,
    the law's area integral, `_read_prior_atlas`'s `INTENSITY_YSO` --
    already in `atlas_pix` order, one row per admitted pixel, so no
    per-source aggregation or alignment is needed here), the protostars
    overplotted by class, and the implied protostellar fraction against
    Dunham et al. 2014 (sec. 5.5 Panel C). `N_YSO_prior = Sum INTENSITY_YSO
    * pixel area * COVERAGE` over the admitted pixels -- the same
    coverage-weighted area integral the region page prints as the
    population before selection (`bmstp.atlas`'s own `TOTAL_PREDICTED`),
    never the per-source law read at the sources' own columns."""
    pixel_area_deg2 = float(hp.nside2pixarea(NSIDE_512, degrees=True))
    n_yso_prior = float(np.sum(intensity_yso * coverage) * pixel_area_deg2)
    ratio = n_proto_footprint / n_yso_prior if n_yso_prior > 0 else float("nan")

    geom = _footprint_geometry(atlas_pix)
    grid_ = _reproject(atlas_pix, intensity_yso, geom["grid_pix"], geom["shape"])
    return dict(geom=geom, grid=grid_, n_yso_prior=n_yso_prior, ratio=ratio)


def _catalogue_word(region, survey_used):
    """The subtitle's own catalogue word: the matched protostars' own
    `SURVEY` value (HOPS for Orion A and Orion B, eHOPS for Aquila);
    where a region's matched sample mixes surveys, "Herschel"."""
    uniq = np.unique(survey_used)
    return uniq[0].decode("utf-8") if uniq.size == 1 else "Herschel"


def build_region(config, region, formats=("png", "pdf")):
    """Writes `bmstp/atlas/figures/protostar-check_<region>.png/.pdf` and
    prints the check's numbers (SPEC_BMSTP_DRAFT.md sec. 5.5, sec. 9's
    protostellar-fraction row)."""
    with progress.Stage("atlas.protostars", region) as st:
        protostars = _read_protostars(config, region)
        n_class_ii = int(np.count_nonzero(protostars["cls"] == b"II"))
        used = np.isin(protostars["cls"], CLASS_USED)

        ra_u = protostars["ra_deg"][used]
        dec_u = protostars["dec_deg"][used]
        pix_u = _pix512_galactic(ra_u, dec_u)

        (density_row, direct, standin, excluded, name_mismatch,
         a_col_dens, f_lim_dens) = _match_to_catalogue(config, region, ra_u, dec_u)

        n_direct = int(np.count_nonzero(direct))
        n_standin = int(np.count_nonzero(standin))
        n_excluded = int(np.count_nonzero(excluded))
        n_used = int(used.sum()) - n_excluded

        # `used` narrows further to the matched (non-excluded) rows; the
        # excluded ones are counted, never fed to the verdict.
        used_matched = np.zeros(used.sum(), dtype=bool)
        used_matched[~excluded] = True

        idx_median = shapes_module._select_source(a_col_dens)
        verdict = _verdict(config, region, dict(
            ra_deg=ra_u, dec_deg=dec_u, av_mag=protostars["av_mag"][used],
            f45_mjy=protostars["f45_mjy"][used], e_f45_mjy=protostars["e_f45_mjy"][used],
            f45_measured=protostars["f45_measured"][used]), used_matched, density_row, idx_median)

        n_measured = int(np.count_nonzero(verdict["measured"]))
        n_not_measured = int(verdict["measured"].size - n_measured)

        atlas_pix, share_yso, intensity_yso, coverage = _read_prior_atlas(config, region)
        density_hpx512 = _read_density_table_for_map(config, region)
        median_proto, median_source, n_dropped_pos = _position_line(
            config, region, pix_u, atlas_pix, share_yso, density_hpx512)

        n_proto_footprint = pix_u.size - n_dropped_pos
        c = _panel_c(atlas_pix, intensity_yso, coverage, n_proto_footprint)
        catalogue_word = _catalogue_word(region, protostars["survey"][used][~excluded])

        print(f"atlas.protostars [{region}]: n_class_ii_excluded={n_class_ii} "
              f"n_used={n_used} name_mismatch={name_mismatch}")
        print(f"atlas.protostars [{region}] match: n_direct={n_direct} n_standin={n_standin} "
              f"n_excluded={n_excluded}")
        print(f"atlas.protostars [{region}] datum: n_measured={n_measured} "
              f"n_not_measured={n_not_measured}")
        print(f"atlas.protostars [{region}] depth: n_past_wall(xi_hat>1)={verdict['n_past_wall']} of "
              f"{n_used} n_beyond_reach={verdict['n_beyond_reach']} A_beam=A_COL_K (sec. 3.2) "
              f"n_top_clipped(log10 ξi_hat>+1.0)={verdict['n_top_clipped']}")
        print(f"atlas.protostars [{region}] no fitted foreground (AV_FOREGROUND_MAG<=0): "
              f"n_no_foreground={verdict['n_no_foreground']} of {n_used} "
              f"frac_yso_leads={verdict['frac_yso_leads_no_foreground']:.4g}")
        print(f"atlas.protostars [{region}] kernel check: empirical log10_xi_hat "
              f"median={verdict['emp_median']:.4g} p84={verdict['emp_p84']:.4g}; predicted "
              f"median={verdict['pred_median']:.4g} p84={verdict['pred_p84']:.4g}; "
              f"frac_beyond_reach={verdict['frac_beyond_reach']:.4g}")
        print(f"atlas.protostars [{region}] verdict (all): "
              f"frac_yso_leads={verdict['frac_yso_leads']:.4g} "
              f"median_P_YSO={verdict['median_p_yso']:.4g}")
        print(f"atlas.protostars [{region}] verdict (measured): "
              f"frac_yso_leads={verdict['frac_yso_leads_measured']:.4g} "
              f"median_P_YSO={verdict['median_p_yso_measured']:.4g}")
        print(f"atlas.protostars [{region}] verdict (not measured): "
              f"frac_yso_leads={verdict['frac_yso_leads_not_measured']:.4g} "
              f"median_P_YSO={verdict['median_p_yso_not_measured']:.4g}")
        print(f"atlas.protostars [{region}] position: median_share_protostars={median_proto:.4g} "
              f"median_share_sources={median_source:.4g}")
        print(f"atlas.protostars [{region}] count: N_proto={n_proto_footprint} "
              f"N_YSO_prior(young stars the prior expects in the covered footprint)="
              f"{c['n_yso_prior']:.4g} ratio={c['ratio']:.4g} "
              f"Dunham+2014 Class0+I+flat/all={DUNHAM2014_PROTOSTELLAR_FRACTION:g}")

        paths = _draw_figure(config, region, protostars, used, excluded, verdict,
                              n_used, n_not_measured, catalogue_word, c, formats)
        st.done(paths[0], n_used=n_used, n_measured=n_measured,
                frac_yso_leads=verdict["frac_yso_leads"], median_p_yso=verdict["median_p_yso"],
                n_yso_prior=c["n_yso_prior"], protostellar_fraction=c["ratio"])
    return paths


def _draw_figure(config, region, protostars, used, excluded, verdict, n_used, n_not_measured,
                  catalogue_word, c, formats):
    plot_style.apply_style()

    n_lead = int(np.count_nonzero(verdict["leading_is_yso"]))
    n_far = verdict["n_beyond_reach"]
    far_clause = f"; {n_far} of these far above it." if n_far > 0 else "."
    caption_text, caption_h = captions.caption_layout(
        "\n\n".join([
            captions.PROTOSTAR_PLANE,
            captions.PROTOSTAR_VERDICT.format(n_lead=n_lead, n=n_used, median=verdict["median_p_yso"]),
            captions.PROTOSTAR_CASES.format(
                n_past=verdict["n_past_wall"], n=n_used, far_clause=far_clause,
                n_nofg=verdict["n_no_foreground"], n_noflux=n_not_measured),
            captions.PROTOSTAR_RATIO.format(
                ratio=c["ratio"], dunham=DUNHAM2014_PROTOSTELLAR_FRACTION)]),
        160, 0.156, 0.20, 0.25)

    margin_l, margin_r, margin_t, gap = 0.85, 0.75, 1.35, 0.6
    # Room, inches, for each panel's own x-axis tick labels and label
    # below its axes, ahead of the caption strip (`atlas.shapes`'s own
    # `AXIS_LABEL_MARGIN_IN`: without it those labels draw past the axes'
    # bottom edge, into the caption text).
    axis_label_margin = 0.5
    panel_h = 6.0
    page_w = PAGE_W_IN
    panel_bottom = caption_h + axis_label_margin
    page_h = margin_t + panel_h + panel_bottom
    usable_w = page_w - margin_l - margin_r
    slot1_w = 0.60 * (usable_w - gap)
    slot2_w = usable_w - gap - slot1_w

    fig = plot_style.new_sized_figure(page_w, page_h)

    # Panel 1 -- every matched protostar at its own measured depth
    # fraction and its dereddened 4.5 micron flux, coloured by the
    # prior's own probability that a source there is a young star; no
    # background field, no interval -- one point per protostar (owner's
    # 2026-09-13 ruling).
    ax1 = fig.add_axes([margin_l / page_w, panel_bottom / page_h, slot1_w / page_w, panel_h / page_h])
    ax1.axvline(0.0, color="black", lw=0.8, linestyle="--", alpha=0.7)

    cmap_pt = plt.get_cmap("viridis")
    norm_pt = Normalize(0.0, 1.0)
    # `verdict`'s own arrays are already narrowed to the matched
    # (non-excluded) protostars, so `cls_matched` must be narrowed the
    # same way to stay aligned with them.
    cls_matched = protostars["cls"][used][~excluded]
    log10_xii = verdict["log10_xii_hat"]
    y_plot = verdict["y_plot"]
    p_yso = verdict["p_yso"]
    measured = verdict["measured"]
    beyond = verdict["beyond_reach"]
    no_fg_mask = verdict["no_foreground"]
    in_reach = ~beyond & ~no_fg_mask

    handles = []
    sc = None
    for cls_val, marker in _CLASS_MARKER.items():
        m_cls_meas = (cls_matched == cls_val) & in_reach & measured
        if np.any(m_cls_meas):
            sc = ax1.scatter(log10_xii[m_cls_meas], y_plot[m_cls_meas], c=p_yso[m_cls_meas],
                              cmap=cmap_pt, norm=norm_pt, marker=marker, s=22,
                              edgecolor="black", linewidths=0.3, zorder=5)
            handles.append(plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor="grey",
                                       markeredgecolor="black", markersize=7, label=_CLASS_LABEL[cls_val]))
    m_not_meas = in_reach & ~measured
    if np.any(m_not_meas):
        sc = ax1.scatter(log10_xii[m_not_meas], y_plot[m_not_meas], c=p_yso[m_not_meas],
                          cmap=cmap_pt, norm=norm_pt, marker=_NOT_MEASURED_MARKER, s=26,
                          edgecolor="black", linewidths=0.3, zorder=5)
        handles.append(plt.Line2D([0], [0], marker=_NOT_MEASURED_MARKER, color="w", markerfacecolor="grey",
                                   markeredgecolor="black", markersize=7, label="no 4.5 µm flux"))
    # Foreground extinction far above the sightline's column: drawn at its
    # OWN measured depth fraction, an open marker, still coloured by the
    # verdict (the edge, its face left empty).
    if np.any(beyond):
        edge_colors = cmap_pt(norm_pt(p_yso[beyond]))
        sc_open = ax1.scatter(
            log10_xii[beyond], y_plot[beyond], marker=_BEYOND_REACH_MARKER, s=32,
            facecolors="none", edgecolors=edge_colors, linewidths=1.3, zorder=6)
        sc = sc if sc is not None else sc_open
        handles.append(plt.Line2D(
            [0], [0], marker=_BEYOND_REACH_MARKER, color="w", markerfacecolor="none",
            markeredgecolor="black", markersize=7,
            label="foreground extinction far above the sightline's column"))
    # No fitted foreground: drawn at the median measured depth fraction of
    # the protostars that have one, a distinct hollow-square marker,
    # coloured by the depth-marginalised verdict.
    if np.any(no_fg_mask):
        fg = ~no_fg_mask
        median_xi_fg = float(np.median(log10_xii[fg])) if np.any(fg) else float(np.min(log10_xii))
        edge_colors_nofg = cmap_pt(norm_pt(p_yso[no_fg_mask]))
        x_nofg = np.full(int(np.count_nonzero(no_fg_mask)), median_xi_fg)
        sc_nofg = ax1.scatter(
            x_nofg, y_plot[no_fg_mask], marker=_NO_FOREGROUND_MARKER, s=32,
            facecolors="none", edgecolors=edge_colors_nofg, linewidths=1.3, zorder=6)
        sc = sc if sc is not None else sc_nofg
        handles.append(plt.Line2D([0], [0], marker=_NO_FOREGROUND_MARKER, color="w", markerfacecolor="none",
                                   markeredgecolor="black", markersize=7, label="no fitted foreground"))

    x_max = min(1.0, max(0.08, float(np.max(log10_xii)) if log10_xii.size else 0.08))
    ax1.set_xlim(-3.0, x_max)
    ax1.set_xlabel(plot_style.label(r"$\mathbf{log_{10}\,\hat{\xi}}$"), fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel(plot_style.label(r"$\mathbf{log_{10}\,F_{4.5}}$", "mJy"), fontsize=LABEL_FONTSIZE)
    ax1.legend(handles=handles, fontsize=7, loc="upper left")
    if sc is not None:
        cbar1 = _panel_colorbar(fig, ax1, sc, label="P(YSO)")
        cbar1.ax.yaxis.label.set_fontweight("bold")

    # Panel 2 -- the prior's own intrinsic young-star density map, the
    # protostars overplotted by class (unchanged data, `_panel_c`).
    wcs = c["geom"]["wcs"]
    rect2 = (margin_l + slot1_w + gap + 0.35, panel_bottom, slot2_w - 0.75, panel_h)
    ax2, im2 = _add_panel(fig, rect2, page_w, page_h, wcs, c["grid"], "viridis",
                           norm=_log_norm(c["grid"]), title="Prior YSO Density")
    class_colors = {b"0": "white", b"I": "gold", b"flat": "orange", b"II": "red"}
    for cls_val, color in class_colors.items():
        m = protostars["cls"] == cls_val
        if np.any(m):
            ax2.scatter(protostars["ra_deg"][m], protostars["dec_deg"][m],
                        transform=ax2.get_transform("world"), s=6, color=color,
                        edgecolor="black", linewidths=0.2, label=_CLASS_LABEL[cls_val], zorder=5)
    ax2.legend(fontsize=6, loc="upper right", markerscale=1.5)
    cbar2 = _panel_colorbar(fig, ax2, im2, label="deg$^{-2}$", log=True)
    cbar2.ax.yaxis.label.set_fontweight("bold")

    # The title names the page; the sample size and catalogue and the
    # prior's own verdict count are the subtitle below it (`figure.
    # titleweight` bolds the suptitle already; the subtitle, a plain
    # `fig.text`, is bolded explicitly) -- the two atlas pages' own
    # convention (`atlas.shapes`).
    fig.suptitle(f"{region} Prior Verdict on Herschel Protostars",
                 fontsize=_TITLE_FONTSIZE, y=1.0 - 0.15 / page_h)
    subtitle_text = f"Prior favors YSO for {n_lead} / {n_used} {catalogue_word} protostars"
    fig.text(0.5, 1.0 - 0.50 / page_h, subtitle_text, fontsize=_SUBTITLE_FONTSIZE,
              fontweight="bold", ha="center", va="top")

    fig.text(margin_l / page_w, (caption_h - 0.20) / page_h, caption_text,
              fontsize=8.0, va="top", ha="left", linespacing=1.3)

    out_dir = os.path.join(config.data_root, "bmstp", "atlas", "figures")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"protostar-check_{region}.{fmt}")
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths


def _regions_with_protostars(config):
    """Every region with at least one protostar in `REGION` (sec. 5.5's
    default), the union of `sky.derived.protostars`'s own admitted-
    footprint assignment."""
    path = f"{config.data_root}/sky/derived/protostars/protostars_survey.hdf5"
    with h5py.File(path, "r") as f:
        region_col = f["REGION"][:]
    names = np.unique(region_col)
    return [n.decode("utf-8") for n in names if n != b""]


def build(config, regions=None):
    """`build(config, regions=None)`: per region, `build_region`
    (default: every region with at least one protostar in `REGION`, sec.
    5.5 -- in practice Orion A and Aquila)."""
    region_names = regions if regions is not None else _regions_with_protostars(config)
    for region in region_names:
        build_region(config, region)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()
    config = config_module.load(args.config)
    build(config, regions=args.regions)


if __name__ == "__main__":
    _main()
