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
from sesnaimpute.atlas.render import _add_panel, _align, _colorbar, _footprint_geometry, _log_norm, _reproject
from sesnaimpute.bmstp import grid
from sesnaimpute.fittp import likelihood as likelihood_module
from sesnaimpute.fittp import prior_reader
from sesnaimpute.population import kernel as kernel_module
from sesnaimpute.population import selection as population_selection

NSIDE_512 = 512
PAGE_W_IN, PAGE_H_IN = 16.0, 9.0

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

#: Panel 1's marker SHAPE per class (colour is reserved for P(YSO | ...));
#: a not-measured protostar overrides this with a downward triangle
#: regardless of class (the brief's own "at its dereddened limit").
_CLASS_MARKER = {b"0": "o", b"I": "s", b"flat": "^"}
_NOT_MEASURED_MARKER = "v"
#: Panel 1's own open marker for a protostar beyond the kernel's reach
#: (this brief's item 2).
_BEYOND_REACH_MARKER = "o"

#: The fine grid this brief's item 2 reads the truncated-mixture median
#: and interval off (dex, in log10 x or log10 r_p): 0.005 dex, this
#: brief's own choice.
_GRID_STEP_DEX = 0.005
#: `P(T >= a_p)` below this bar: too far past the wall for the kernel to
#: place credible mass there -- BEYOND THE KERNEL'S REACH (this brief's
#: item 2), drawn at the wall with an open marker and no interval.
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
    brief's item 5, the median source's own `X_MARGINAL` row)."""
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


def _read_x_marginal(config, region, sightline_row):
    """The median source's own `X_MARGINAL` row (P3, `bmstp.shapes`): the
    YSO `x` marginal at its sightline -- this brief's item 5, the kernel
    validation's own `x_member` distribution."""
    path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    with h5py.File(path, "r") as f:
        return np.asarray(f["X_MARGINAL"][int(sightline_row)], dtype=np.float64)


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
    """`P(log10 x <= z)`, the untruncated two-component mixture CDF, in
    closed form (`_norm_cdf`, exact `erf`) -- never a grid integral of the
    pdf (see `_position_distribution`'s own docstring for why that broke)."""
    return w * _norm_cdf((z - mean0) / sigma0) + (1.0 - w) * _norm_cdf((z - mean1) / sigma1)


def _position_distribution(log10_r_p, w, mu, sigma):
    """`(median, lo16, hi84, p_reach, beyond_reach)`: the read's own
    picture of a protostar's position in the DISTANCE coordinate (this
    brief's item 2). `log10 x = log10 r_p - y`, `y` the two-component
    cloud-class column kernel mixture in dex (`w`, `mu` (n, 2), `sigma`
    (n, 2), `Kernel.mixture`'s own `exponent=2` reweighting, the star-gas
    law's own exponent) -- a mixture of Gaussians in `log10 x` itself,
    mean `log10 r_p - mu_i`, the same `sigma_i`, weight `w_i` / `1 - w_i`.
    `p_reach = P(log10 x <= 0)` (`T >= a_p`) is the kernel's own
    UNTRUNCATED mass there; `median`/`lo16`/`hi84` are the TRUNCATED
    mixture's own 50%/16%/84% points, `CDF(z) = target * p_reach` INVERTED
    by bisection on the exact closed-form CDF (rule 8: every iteration is
    one array pass over all protostars, no python loop over sources) --
    not a fixed-grid trapezoidal pdf integral: a protostar whose floored
    `log10 r_p` sits at the array's own low edge can have BOTH kernel
    components' means at or below a finite grid's own edge (a component's
    mean can itself be more negative than `log10 r_p` when its own `mu` is
    positive), so a grid starting there silently integrates less than the
    component's own mass and its cumulative sum never reaches `p_reach` --
    the bisection here has no edge, so it cannot lose mass that way.
    `beyond_reach` marks `p_reach` below this brief's own 0.01 bar."""
    log10_r_p = np.asarray(log10_r_p, dtype=np.float64)
    mean0 = log10_r_p - mu[:, 0]
    mean1 = log10_r_p - mu[:, 1]
    sigma0, sigma1 = sigma[:, 0], sigma[:, 1]
    p_reach = w * _norm_cdf(-mean0 / sigma0) + (1.0 - w) * _norm_cdf(-mean1 / sigma1)

    # A bracket guaranteed to hold every root: comfortably below both
    # component means (`CDF` there is ~0) up to the wall itself (`CDF(0)
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


def _kernel_validation(log10_r_p, w, mu, sigma, x_marginal, x_centers, beyond_reach):
    """`(emp_median, emp_p84, pred_median, pred_p84, frac_beyond)`: this
    brief's item 5, the kernel's own prediction for a cloud member's beam
    ratio -- `log10 r_p = log10 x_member + y`, `x_member` from the
    region's median-sightline YSO `x` marginal (`X_MARGINAL`, shared
    across protostars), `y` from each protostar's OWN class kernel (`w`,
    `mu`, `sigma`) -- against the sample's own empirical `log10 r_p`. The
    predicted CDF pools each protostar's own kernel CDF into ONE shared
    function `G` first (rule 8: an (n_proto, n_u) array, averaged over
    protostars), then convolves that one function with the shared
    `x_marginal` (an (n_z, n_x_centers) array) -- never the full
    (n_proto, n_x_centers, n_grid) product."""
    u_grid = np.arange(-3.0, 3.0 + 1e-9, _GRID_STEP_DEX)
    z0 = (u_grid[None, :] - mu[:, 0][:, None]) / sigma[:, 0][:, None]
    z1 = (u_grid[None, :] - mu[:, 1][:, None]) / sigma[:, 1][:, None]
    cdf_y = w[:, None] * _norm_cdf(z0) + (1.0 - w[:, None]) * _norm_cdf(z1)
    g_pooled = cdf_y.mean(axis=0)  # (n_u,): the ensemble-pooled kernel CDF

    total = float(x_marginal.sum())
    x_marg = x_marginal / total if total > 0.0 else x_marginal
    z_grid = np.arange(-4.0, 2.0 + 1e-9, _GRID_STEP_DEX)
    offsets = z_grid[:, None] - x_centers[None, :]
    g_vals = np.interp(offsets.ravel(), u_grid, g_pooled, left=0.0, right=1.0).reshape(offsets.shape)
    cdf_pooled = (g_vals * x_marg[None, :]).sum(axis=1)

    pred_median = float(np.interp(0.5, cdf_pooled, z_grid))
    pred_p84 = float(np.interp(0.84, cdf_pooled, z_grid))
    n = log10_r_p.shape[0]
    emp_median = float(np.median(log10_r_p)) if n else float("nan")
    emp_p84 = float(np.percentile(log10_r_p, 84.0)) if n else float("nan")
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
    """P6's admitted nside-512 pixels and cataloged YSO share, sorted by
    pixel for the `searchsorted` joins below (the position line, sec.
    5.5's caption (ii))."""
    path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    if not os.path.exists(path):
        raise ValueError(
            f"atlas.protostars: no {path!r} -- run RUNBOOKtp.sh's "
            f"'PY sesnaimpute.bmstp.atlas' line")
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        share_yso = np.asarray(f["SHARE_YSO"][:], dtype=np.float64)
    order = np.argsort(pix)
    return pix[order], share_yso[order]


def _read_density_table_for_map(config, region):
    """P1's per-source `HPX_512` and `DENSITY_YSO` (sec. 5.5's quadratic
    column law, evaluated at each catalogued source's own column) -- panel
    2's own map, unchanged from the prior design."""
    path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    with h5py.File(path, "r") as f:
        hpx512 = np.asarray(f["HPX_512"][:], dtype=np.int64)
        density_yso = np.asarray(f["DENSITY_YSO"][:], dtype=np.float64)
    return hpx512, density_yso


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


def _share_yso(lambda_row, class_order):
    """`P(YSO | cell, s)` over the whole grid for one row's own `Lambda`
    (n_class, n_x, n_b) -- row 2's own share (`atlas.shapes`), masked to
    NaN outside the support so it draws blank there."""
    idx_yso = class_order.index("YSO")
    support = np.zeros(lambda_row.shape[1], dtype=bool)
    support[:grid.N_X_SUPPORT] = True
    total = lambda_row.sum(axis=0)
    denom = np.where(support[:, None], total, 1.0)
    share = lambda_row[idx_yso] / denom
    return np.where(support[:, None], share, np.nan)


def _verdict(config, region, protostars, used, density_row, idx_median):
    """Section 3's "The verdict": `P(C | r_p, datum, s)` for the six
    classes at every used protostar's own matched row, read at the
    MEASURED cell `log10 r_p` from the blurred grids (`blur=True`, this
    brief's items 2-3) -- the position bar drawn instead in the DISTANCE
    coordinate, `p(log10 x | r_p, cloud kernel)` truncated at the wall
    (`_position_distribution`). The background (panel 1's map) is the
    region's median source's own UNBLURRED read (`blur=False`), so the
    bars and the field share one coordinate (item 3): a second
    `lambda_grids` call, since the two need different `blur` settings."""
    ra_u = protostars["ra_deg"][used]
    av_u = protostars["av_mag"][used]
    f45_u = protostars["f45_mjy"][used]
    e45_u = protostars["e_f45_mjy"][used]
    measured_u = protostars["f45_measured"][used]
    rows_u = density_row[used]

    (_, a_col_dens, f_lim_dens, a_col_sig_dens, arm_dens, zp_sig_dens,
     sightline_dens) = _read_density_rows(config, region)
    w_i2 = _read_w_dex_i2(config, region)

    # A_beam, sec. 3.2's own extinction column (A_COL_K): r_p = a_p / A_beam.
    a_col_row = a_col_dens[rows_u]
    w_ramp_col = population_selection.law_dense_weight(a_col_row)
    ak_per_av_col = population_selection.ak_per_av(config, w_ramp_col)
    a_p = av_u * ak_per_av_col
    r_p = a_p / a_col_row
    # Floored at the array's own low edge (`grid.LOG10_X_EDGES[0]`, this
    # brief's item 2's own grid floor): a zero-foreground protostar
    # (AV_FOREGROUND_MAG = 0) would otherwise take `log10 r_p = -inf`,
    # which does not merely warn -- `_grid_quantile`'s own cumulative
    # search sees a flat-zero CDF over the whole finite grid and returns
    # its TOP index, snapping the position bar to the wall instead of the
    # grid's low edge, backwards from what a vanishing ratio means.
    r_p = np.maximum(r_p, 10.0 ** grid.LOG10_X_EDGES[0])
    log10_r_p = np.log10(r_p)
    n_past_wall = int(np.count_nonzero(r_p > 1.0))

    # The cloud-class column kernel at each protostar's own matched row
    # (`Kernel.mixture`, `exponent=2`: protostars are a cloud population,
    # `prior_reader.KERNEL_EXPONENT["YSO"]`, the star-gas law's own
    # exponent, sec. 5.5): `y`, `log10 T = log10 A_beam + y`.
    kernel = kernel_module.Kernel.read(config)
    w_mix, mu_mix, sigma_mix = kernel.mixture(
        a_col_row, a_col_sig_dens[rows_u], arm_dens[rows_u], zp_sigma_k=zp_sig_dens[rows_u],
        exponent=prior_reader.KERNEL_EXPONENT["YSO"])
    (log10_x_med, log10_x_lo, log10_x_hi, p_reach, beyond_reach) = _position_distribution(
        log10_r_p, w_mix, mu_mix, sigma_mix)

    # The verdict's own cell: the MEASURED coordinate `log10 r_p`, which
    # may sit above the wall in the padding -- clipped only at the
    # array's own TOP edge (`grid.LOG10_X_EDGES[-1]`, +1.0 dex), never at
    # the wall (item 3): this is what the fitter reads.
    n_x_full = grid.LOG10_X_EDGES.size - 1
    cell_x = np.clip(np.searchsorted(grid.LOG10_X_EDGES, log10_r_p, side="right") - 1, 0, n_x_full - 1)
    n_top_clipped = int(np.count_nonzero(log10_r_p > grid.LOG10_X_EDGES[-1]))

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

    idx_yso = class_order.index("YSO")
    p_yso = p_c[:, idx_yso]
    leading = np.array(class_order)[np.argmax(p_c, axis=1)]
    leading_is_yso = leading == "YSO"

    y_dered_limit = np.log10(f_lim_i2) + 0.4 * a_p * kappa_i2
    log10_f45_dered = np.log10(np.where(measured_u, f45_u, 1.0)) + 0.4 * a_p * kappa_i2
    y_plot = np.where(measured_u, log10_f45_dered, y_dered_limit)

    # Panel 1's background: the median source's own YSO conditional in
    # the DISTANCE coordinate (`blur=False`, item 3), never the measured
    # one the verdict itself reads.
    lambda_background, _, background_class_order = shapes_module.lambda_grids(
        config, region, np.array([idx_median]), blur=False)
    share_yso_median = _share_yso(lambda_background[0], background_class_order)

    # The kernel validation (item 5): the region's median-sightline YSO x
    # marginal against each protostar's own column kernel, pooled.
    x_marginal = _read_x_marginal(config, region, sightline_dens[idx_median])
    x_centers = 0.5 * (grid.LOG10_X_EDGES[:-1] + grid.LOG10_X_EDGES[1:])
    emp_median, emp_p84, pred_median, pred_p84, frac_beyond = _kernel_validation(
        log10_r_p, w_mix, mu_mix, sigma_mix, x_marginal, x_centers, beyond_reach)

    return dict(
        log10_x_p=log10_x_med, log10_x_lo=log10_x_lo, log10_x_hi=log10_x_hi,
        p_reach=p_reach, beyond_reach=beyond_reach, y_plot=y_plot, p_yso=p_yso,
        leading_is_yso=leading_is_yso, measured=measured_u, n_past_wall=n_past_wall,
        n_top_clipped=n_top_clipped, class_order=class_order, rows_u=rows_u,
        share_yso_median=share_yso_median,
        n_beyond_reach=int(np.count_nonzero(beyond_reach)),
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
    )


def _position_line(config, region, protostars_pix, atlas_pix, share_yso, density_hpx512):
    """The caption's position line (ii): the prior's cataloged YSO share's
    median at the protostars' own pixels against every cataloged source's
    own pixel."""
    share_proto, found = _join(atlas_pix, share_yso, protostars_pix)
    share_proto = share_proto[found]
    share_source, found_src = _join(atlas_pix, share_yso, density_hpx512)
    share_source = share_source[found_src]
    median_proto = float(np.median(share_proto)) if share_proto.size else float("nan")
    median_source = float(np.median(share_source)) if share_source.size else float("nan")
    return median_proto, median_source, int(np.count_nonzero(~found))


def _panel_c(footprint_pix, density_hpx512, density_yso, n_proto_footprint):
    """The map of the prior's intrinsic YSO density per pixel, the
    protostars overplotted by class, and the implied protostellar
    fraction against Dunham et al. 2014 (sec. 5.5 Panel C, unchanged)."""
    uniq_pix, inverse = np.unique(density_hpx512, return_inverse=True)
    sum_density = np.bincount(inverse, weights=density_yso)
    count_per_pix = np.bincount(inverse)
    mean_density = sum_density / count_per_pix

    aligned = _align(footprint_pix, uniq_pix, mean_density, np.nan)
    pixel_area_deg2 = float(hp.nside2pixarea(NSIDE_512, degrees=True))
    n_yso_prior = float(np.nansum(aligned) * pixel_area_deg2)
    ratio = n_proto_footprint / n_yso_prior if n_yso_prior > 0 else float("nan")

    geom = _footprint_geometry(footprint_pix)
    grid_ = _reproject(footprint_pix, aligned, geom["grid_pix"], geom["shape"])
    return dict(geom=geom, grid=grid_, n_yso_prior=n_yso_prior, ratio=ratio)


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

        atlas_pix, share_yso = _read_prior_atlas(config, region)
        density_hpx512, density_yso = _read_density_table_for_map(config, region)
        median_proto, median_source, n_dropped_pos = _position_line(
            config, region, pix_u, atlas_pix, share_yso, density_hpx512)

        n_proto_footprint = pix_u.size - n_dropped_pos
        c = _panel_c(atlas_pix, density_hpx512, density_yso, n_proto_footprint)
        share_yso_median = verdict["share_yso_median"]

        print(f"atlas.protostars [{region}]: n_class_ii_excluded={n_class_ii} "
              f"n_used={n_used} name_mismatch={name_mismatch}")
        print(f"atlas.protostars [{region}] match: n_direct={n_direct} n_standin={n_standin} "
              f"n_excluded={n_excluded}")
        print(f"atlas.protostars [{region}] datum: n_measured={n_measured} "
              f"n_not_measured={n_not_measured}")
        print(f"atlas.protostars [{region}] depth: n_past_wall(r_p>1)={verdict['n_past_wall']} of "
              f"{n_used} n_beyond_reach={verdict['n_beyond_reach']} A_beam=A_COL_K (sec. 3.2) "
              f"n_top_clipped(log10 r_p>+1.0)={verdict['n_top_clipped']}")
        print(f"atlas.protostars [{region}] kernel check: empirical log10_r_p "
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
              f"N_YSO_prior={c['n_yso_prior']:.4g} ratio={c['ratio']:.4g} "
              f"Dunham+2014 Class0+I+flat/all={DUNHAM2014_PROTOSTELLAR_FRACTION:g}")

        paths = _draw_figure(config, region, protostars, used, density_row, excluded, verdict,
                              share_yso_median, n_proto_footprint, n_direct, n_standin, n_excluded,
                              median_proto, median_source, c, formats)
        st.done(paths[0], n_used=n_used, n_measured=n_measured,
                frac_yso_leads=verdict["frac_yso_leads"], median_p_yso=verdict["median_p_yso"],
                n_yso_prior=c["n_yso_prior"], protostellar_fraction=c["ratio"])
    return paths


def _draw_figure(config, region, protostars, used, density_row, excluded, verdict, share_yso_median,
                  n_proto_footprint, n_direct, n_standin, n_excluded, median_proto, median_source,
                  c, formats):
    plot_style.apply_style()
    fig = plot_style.new_sized_figure(PAGE_W_IN, PAGE_H_IN)

    caption_text, caption_h = captions.caption_layout(
        "\n\n".join([
            captions.PROTOSTAR_STATEMENT,
            captions.PROTOSTAR_POSITION.format(median_proto=median_proto, median_source=median_source),
            captions.PROTOSTAR_DEPTH.format(
                n_past_wall=verdict["n_past_wall"], n_beyond_reach=verdict["n_beyond_reach"],
                n_used=verdict["measured"].size),
            captions.PROTOSTAR_KERNEL_CHECK.format(
                emp_median=verdict["emp_median"], emp_p84=verdict["emp_p84"],
                pred_median=verdict["pred_median"], pred_p84=verdict["pred_p84"],
                frac_beyond=verdict["frac_beyond_reach"], n_used=verdict["measured"].size),
            captions.PROTOSTAR_TIERS.format(
                n_used=verdict["measured"].size,
                n_measured=int(np.count_nonzero(verdict["measured"])),
                n_not_measured=int(np.count_nonzero(~verdict["measured"])),
                n_direct=n_direct, n_standin=n_standin, n_excluded=n_excluded)]),
        160, 0.156, 0.20, 0.25)

    margin_l, margin_r, margin_t, gap = 0.85, 0.75, 0.75, 0.6
    # Room, inches, for each panel's own x-axis tick labels and label
    # below its axes, ahead of the caption strip (`atlas.shapes`'s own
    # `AXIS_LABEL_MARGIN_IN`: without it those labels draw past the axes'
    # bottom edge, into the caption text).
    axis_label_margin = 0.5
    usable_w = PAGE_W_IN - margin_l - margin_r
    slot1_w = 0.60 * (usable_w - gap)
    slot2_w = usable_w - gap - slot1_w
    panel_bottom = caption_h + axis_label_margin
    panel_h = PAGE_H_IN - margin_t - panel_bottom

    # Panel 1 -- the verdict's own picture in the DISTANCE coordinate: the
    # protostars at (log10 x, log10 F45_dered), the point at
    # `p(log10 x | r_p, cloud kernel)`'s own truncated median with a
    # horizontal bar over its 16%-84% interval, coloured by the MEASURED
    # verdict P(YSO | r_p, datum, s); the region's median source's own
    # UNBLURRED P(YSO | cell, s) as the background, so bars and field
    # share one coordinate (this brief's item 2-3) -- nothing to draw
    # above the wall here, so the axis stops at it.
    ax1 = fig.add_axes([margin_l / PAGE_W_IN, panel_bottom / PAGE_H_IN, slot1_w / PAGE_W_IN, panel_h / PAGE_H_IN])
    extent = [grid.LOG10_X_EDGES[0], grid.LOG10_X_EDGES[-1], grid.LOG10_F45_EDGES[0], grid.LOG10_F45_EDGES[-1]]
    cmap_bg = plt.get_cmap("Greys").copy()
    cmap_bg.set_bad("white")
    ax1.imshow(share_yso_median.T, origin="lower", aspect="auto", extent=extent,
               cmap=cmap_bg, norm=Normalize(0.0, 1.0))
    ax1.axvline(0.0, color="black", lw=0.8, linestyle="--", alpha=0.7)

    cmap_pt = plt.get_cmap("viridis")
    norm_pt = Normalize(0.0, 1.0)
    # `verdict`'s own arrays are already narrowed to the matched
    # (non-excluded) protostars, so `cls_matched` must be narrowed the
    # same way to stay aligned with them.
    cls_matched = protostars["cls"][used][~excluded]
    in_reach = ~verdict["beyond_reach"]
    sc = None
    for cls_val, marker in _CLASS_MARKER.items():
        m_cls = (cls_matched == cls_val) & in_reach
        m_meas = m_cls & verdict["measured"]
        m_not = m_cls & ~verdict["measured"]
        for mask, mk, size in ((m_meas, marker, 22), (m_not, _NOT_MEASURED_MARKER, 26)):
            if np.any(mask):
                ax1.errorbar(
                    verdict["log10_x_p"][mask], verdict["y_plot"][mask],
                    xerr=[verdict["log10_x_p"][mask] - verdict["log10_x_lo"][mask],
                          verdict["log10_x_hi"][mask] - verdict["log10_x_p"][mask]],
                    fmt="none", ecolor="0.4", elinewidth=0.6, capsize=1.5, zorder=4)
                sc = ax1.scatter(verdict["log10_x_p"][mask], verdict["y_plot"][mask],
                                  c=verdict["p_yso"][mask], cmap=cmap_pt, norm=norm_pt,
                                  marker=mk, s=size, edgecolor="black", linewidths=0.3, zorder=5)
    # BEYOND THE KERNEL'S REACH (item 2): drawn at the wall, an open
    # marker, no interval -- still coloured by the verdict (the edge, its
    # face left empty).
    beyond = verdict["beyond_reach"]
    if np.any(beyond):
        edge_colors = cmap_pt(norm_pt(verdict["p_yso"][beyond]))
        sc_open = ax1.scatter(
            np.zeros(int(np.count_nonzero(beyond))), verdict["y_plot"][beyond],
            marker=_BEYOND_REACH_MARKER, s=32, facecolors="none", edgecolors=edge_colors,
            linewidths=1.3, zorder=6)
        sc = sc if sc is not None else sc_open
    ax1.set_xlim(grid.LOG10_X_EDGES[0], 0.0)
    ax1.set_xlabel(plot_style.label(r"$\log_{10} x_p$"))
    ax1.set_ylabel(plot_style.label(r"$\log_{10} F_{4.5}$", "mJy"))
    ax1.set_title("verdict: P(YSO | r_p, datum, s); position: p(log10 x | r_p, kernel)", fontsize=10)
    handles = [plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor="grey",
                           markeredgecolor="black", markersize=7, label=cls_val.decode())
               for cls_val, marker in _CLASS_MARKER.items()]
    handles.append(plt.Line2D([0], [0], marker=_NOT_MEASURED_MARKER, color="w", markerfacecolor="grey",
                               markeredgecolor="black", markersize=7, label="not measured"))
    handles.append(plt.Line2D([0], [0], marker=_BEYOND_REACH_MARKER, color="w", markerfacecolor="none",
                               markeredgecolor="black", markersize=7, label="beyond the kernel's reach"))
    ax1.legend(handles=handles, fontsize=7, loc="upper left")
    if sc is not None:
        cax1 = fig.add_axes([(margin_l + slot1_w + 0.05) / PAGE_W_IN, panel_bottom / PAGE_H_IN,
                              0.15 / PAGE_W_IN, panel_h / PAGE_H_IN])
        cbar1 = fig.colorbar(sc, cax=cax1)
        cbar1.set_label(r"$P(\mathrm{YSO} \mid x, \mathrm{datum}, s)$", fontsize=10)

    # Panel 2 -- the map (unchanged).
    wcs = c["geom"]["wcs"]
    rect2 = (margin_l + slot1_w + gap + 0.35, panel_bottom, slot2_w - 0.75, panel_h)
    ax2, im2 = _add_panel(fig, rect2, PAGE_W_IN, PAGE_H_IN, wcs, c["grid"], "viridis",
                           norm=_log_norm(c["grid"]), title="count (deg$^{-2}$)")
    class_colors = {b"0": "white", b"I": "gold", b"flat": "orange", b"II": "red"}
    for cls_val, color in class_colors.items():
        m = protostars["cls"] == cls_val
        if np.any(m):
            ax2.scatter(protostars["ra_deg"][m], protostars["dec_deg"][m],
                        transform=ax2.get_transform("world"), s=6, color=color,
                        edgecolor="black", linewidths=0.2, label=cls_val.decode(), zorder=5)
    ax2.legend(fontsize=6, loc="upper right", markerscale=1.5)
    bar_rect = (rect2[0] + rect2[2] + 0.12, rect2[1] + 0.05 * rect2[3], 0.35, 0.9 * rect2[3])
    _colorbar(fig, im2, bar_rect, PAGE_W_IN, PAGE_H_IN)

    title = (f"{region} -- protostar check: N_proto={n_proto_footprint}, "
             f"YSO leads the prior for {int(np.count_nonzero(verdict['leading_is_yso']))} of "
             f"{verdict['leading_is_yso'].size}, median P(YSO)={verdict['median_p_yso']:.3f}; "
             f"N_YSO_prior={c['n_yso_prior']:.4g}, ratio={c['ratio']:.3f} "
             f"(Dunham+2014 {DUNHAM2014_PROTOSTELLAR_FRACTION:g}); "
             f"beyond the kernel's reach: {verdict['n_beyond_reach']} of {verdict['measured'].size}")
    # `fontsize` down from 11 (the page grew a "beyond the kernel's
    # reach" clause, this brief's item 4): at 11 pt the longer title ran
    # past the page's own 16-inch width and was cut off at both edges.
    fig.suptitle(title, fontsize=9, y=1.0 - 0.12 / PAGE_H_IN)

    fig.text(margin_l / PAGE_W_IN, (caption_h - 0.20) / PAGE_H_IN, caption_text,
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
