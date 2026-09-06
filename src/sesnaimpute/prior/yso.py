"""The YSO law count and the per-sightline shape (SPEC_PRIORS.md section 6.1,
6.3; IMPLEMENTATION.md section 3, the YSO row).

Two products, independent of each other and of the mass-based selection
(section 6.2, built elsewhere):

`law_count` (section 6.1): disk-bearing young stars form in proportion to
the square of the cloud's own column, `N_law = kappa * (d_r*pi/180)^2 *
A_s^2`, the source's whole adopted column, with `kappa` Pokhrel+2020's
pooled star-gas relation on Herschel columns: 14.5 young stars pc^-2 per
mag^2 of A_K at the Herschel arm's 36.3" beam, transferred to the Planck
arm's 5.03' by the measured beam ratio, 1.29, giving 18.7. `build` writes
one 30-row product with the region distance and the pc^2/deg^2 factor it
multiplies.

The shape (section 6.3): a young star sits at a depth drawn from the
cloud's own gas density, `p(u) ~ rho_gas(d(u))^(1/2)` where
`u = A(d)/A(inf)` (Parmentier & Pfalzner 2013's 3-D form of the observed
Sigma_YSO ~ Sigma_gas^2 surface law), on the profile's own piecewise-linear
cells; the far-field tail past the 3-D map's edge (section 1.4) is one
further cell carrying the residual column, so `u` reaches 1 on every
sightline. The brightness axis is the same placement read on distance:
`log10 B = -2 log10(d / 1 kpc)`, Gaussian in `log10 B` about a
rho_gas^(3/2)-weighted ridge fitted along the same ray, width the ridge's
own conditional residual.

The `a`-marginal is the exact expectation of the embedding density's CDF
`F_u` under the log-normal column kernel (section 1.2), `P(a<=a0|A_s) =
E_T[F_u(a0/T)]`, `T` a source's own true column, `log10 T ~ Normal(log10
A_s + mu, sigma)`. `F_u` is piecewise linear on `u_edges`, so this splits
over its cells into closed-form terms of the normal CDF `Phi` at the
cell's own `T`-range boundaries and the log-normal's partial inverse
moment `E[1/T; T in (lo, hi)]`, no quadrature and no `a`-grid: `YsoShape.
_cdf_rows`/`_marginal_rows` evaluate it exactly at any `a`. `build`
writes one product per region, sightline granule: `U_EDGES`/`P_U` (the
shape itself), `RIDGE_*` (the closed-form conditional brightness
density), and `IS_HERSCHEL` (each sightline's own map class). `YsoShape.
read` loads these plus the survey's column kernel and node grid, and
evaluates the marginal/cdf batched over sources, node-blended
(`marginal_at`, `cdf_at`, IMPLEMENTATION.md section 2) or at each
source's own exact column and measurement uncertainty (`marginal_exact`,
`cdf_exact`).

No selection (section 6.2) and no library enter either product (C3):
neither reads a template register or an IMF.
"""

import os

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from joblib import Parallel, delayed
from scipy.special import erf

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import column_grid
from sesnaimpute.prior import kernel as kernel_module
from sesnaimpute.sky.derived import herschel_column as sky_herschel_column
from sesnaimpute.sky.derived import planck_column as sky_planck_column

# ====================================================================
# 6.1 -- the law count
# ====================================================================

#: Pokhrel+2020's pooled star-gas relation on Herschel columns: 14.5 young
#: stars pc^-2 mag^-2 of A_K at the Herschel arm's 36.3" beam
#: (SPEC_PRIORS.md section 6.1).
KAPPA_HERSCHEL = 14.5

#: The Herschel level transferred to the Planck arm's 5.03' beam by the
#: same measured beam ratio (1.29) used before (SPEC_PRIORS.md section 6.1).
KAPPA_PLANCK = 18.7

#: Pokhrel+2020's cloud-to-cloud scatter of the normalisation: the
#: uncertainty on any one region's law level, reported and never
#: marginalised (SPEC_PRIORS.md section 6.1).
LAW_BAND_DEX = 0.36

#: Provenance codes of the adopted column (SPEC_PRIORS.md section 1.1):
#: 0 Herschel, 1 Planck.
PROVENANCE_HERSCHEL = 0
PROVENANCE_PLANCK = 1

_ADOPTED_COLUMN_CACHE = {}

_LN10 = float(np.log(10.0))
_SQRT2 = float(np.sqrt(2.0))


def pc2_per_deg2(d_r_pc):
    """The pc^2-per-deg^2 conversion at distance `d_r_pc`: one degree on
    the sky subtends `d_r_pc * pi/180` pc there, so its area factor is
    that length squared (SPEC_PRIORS.md section 6.1)."""
    return (np.asarray(d_r_pc, dtype=float) * np.pi / 180.0) ** 2


def _adopted_columns(config, region):
    """`(a_col, sigma_col, provenance)`, every source of `region`, in
    catalogue row order -- cached, since both `law_count` (per call) and
    the law product build read the same array."""
    key = (id(config), region)
    cached = _ADOPTED_COLUMN_CACHE.get(key)
    if cached is not None:
        return cached
    path = config_module.product_path(config, "sky/derived", "adopted",
                                       "column", "source", region=region)
    cols = access.per_source(config, region, path,
                              ["A_COL_K", "A_COL_SIG_K", "A_COL_PROVENANCE"])
    out = (np.asarray(cols["A_COL_K"], dtype=float),
           np.asarray(cols["A_COL_SIG_K"], dtype=float),
           np.asarray(cols["A_COL_PROVENANCE"]))
    _ADOPTED_COLUMN_CACHE[key] = out
    return out


def law_count(config, region, a_col, provenance):
    """`N_law`, young stars deg^-2, for arrays of adopted column and arm
    (SPEC_PRIORS.md section 6.1):

        N_law = kappa_arm * (d_r*pi/180)^2 * a_col^2

    the source's whole adopted column, no pedestal.
    """
    a_col = np.asarray(a_col, dtype=float)
    provenance = np.asarray(provenance)
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    kappa = np.where(provenance == PROVENANCE_HERSCHEL,
                      KAPPA_HERSCHEL, KAPPA_PLANCK)
    return kappa * pc2_per_deg2(d_r_pc) * a_col ** 2


def _law_row(config, region):
    """One region's law-product row: the region distance and the
    pc^2/deg^2 factor `law_count` multiplies."""
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    pc2 = float(pc2_per_deg2(d_r_pc))
    return dict(region=region, d_r_pc=float(d_r_pc), pc2_per_deg2=pc2)


def _write_law_product(config, regions, rows):
    """Writes the 30-row law product, in place for `regions`
    (CODING_RULES.md 5c), plus the three region-independent constants
    (`KAPPA_HERSCHEL`, `KAPPA_PLANCK`, `LAW_BAND_DEX`) as the same
    root-level scalar datasets every build has always written -- created
    once, since their value never depends on which regions ran."""
    path = config_module.product_path(config, "bms", "yso", "law", "region")
    tables_module.update_rows(
        path, regions,
        {
            "D_R_PC": np.array([r["d_r_pc"] for r in rows], dtype=np.float64),
            "PC2_PER_DEG2": np.array([r["pc2_per_deg2"] for r in rows], dtype=np.float64),
        },
        granule="region")
    with tables_module.open_product(path, granule="region") as f:
        for name, value in (("KAPPA_HERSCHEL", KAPPA_HERSCHEL),
                             ("KAPPA_PLANCK", KAPPA_PLANCK),
                             ("LAW_BAND_DEX", LAW_BAND_DEX)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=np.float64(value))
        for stale in ("PEDESTAL_K", "PEDESTAL_REMOVED_FRAC"):
            if stale in f:
                del f[stale]
    return path


# ====================================================================
# 6.1 (area form) -- the law count integrated over an anchor pixel's area
# ====================================================================
#
# SPEC_PRIORS.md section 2.1's "young stars in the anchors" row needs "the
# law count integrated over the tile", not a source-sampled mean: the
# anchor subtraction is an AREA count, and `N_law` is convex (squared) in
# column, so the mean of SOURCE-level law counts under-runs the true area
# integral -- SESNA's own sources avoid the densest gas, so they are a
# biased-low sample of the pixel's own column field. Section 6.4 item 1's
# per-cloud check needs the same integral, so it is served once here.

#: The anchor histograms' own HEALPix resolution (`prior.anchor_tiles`,
#: `prior.young_stars`): the nside this integral is evaluated at.
NSIDE_ANCHOR = 512

_KERNEL_CACHE = {}


def _read_kernel(config):
    """The survey's column kernel (`prior.kernel.Kernel.read`), cached
    per config like this module's other survey-wide reads
    (`_ADOPTED_COLUMN_CACHE`): `law_area_integral` is called once per
    region, so an uncached reload would cost O(regions)."""
    key = id(config)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = kernel_module.Kernel.read(config)
    return _KERNEL_CACHE[key]


def _kernel_second_moment_planck(config, a_col):
    """`E[T^2 | A]` for the Planck-only branch of `law_area_integral`
    (SPEC_PRIORS.md section 1.2), in closed form from the log-normal
    kernel's own moments: `E[T^p|A] = A^p * 10^(p*mu) *
    exp((p*sigma*ln10)^2/2)` at `p=2`, `mu`/`sigma` at zero source
    measurement uncertainty (the column map carries no per-source
    instrumental term, only the sub-beam kernel's own width)."""
    a_col = np.asarray(a_col, dtype=float)
    kern = _read_kernel(config)
    mu, sigma = kern.params(a_col, np.zeros_like(a_col),
                             np.full(a_col.shape, "planck"))
    s = sigma * _LN10
    return a_col ** 2 * 10.0 ** (2.0 * mu) * np.exp(2.0 * s * s)


#: Memory rules for the Herschel-map joblib pool in `_herschel_pixel_stats`:
#: the total concurrent budget, and the per-worker array multiplier (the
#: float32 cutout itself, its finite mask, and the block-reduce sum/count
#: arrays -- four arrays of about the cutout's own size at worst).
_HERSCHEL_POOL_BUDGET_BYTES = 6 * (1 << 30)
_HERSCHEL_ARRAYS_PER_WORKER = 4


def _map_window(path, rbox, pad_deg=0.1):
    """The FITS pixel window of one HGBS map covering `rbox` (padded by
    `pad_deg`, matching the candidate screen's own pad) -- header and WCS
    only, no pixel data touched, so every candidate map's cutout size is
    known before any of them is read: `dict(wcs, x0, x1, y0, y1, f,
    pixscale)`, an empty window (`x1 <= x0` or `y1 <= y0`) where the
    padded bbox misses the map's own pixel grid. `x0`/`y0` are rounded
    down, `x1`/`y1` up, to the block factor `f`
    (`sky.derived.planck_column.BLOCK_TARGET_ARCSEC`), so the cutout's
    own block grid lands on the same absolute pixel boundaries the
    un-cropped map's block-reduce would use -- the cutout changes what is
    read, never the block-reduced result, bit-identical to reading the
    whole mosaic (the generous padding means a block actually used by the
    region's anchor pixels is never split across the cutout edge)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(path, memmap=True) as hd:
        chosen = next(c for c in hd if c.header.get("NAXIS", 0) >= 2)
        hdr = chosen.header
        ny, nx = chosen.shape[-2:]
        wcs = WCS(hdr).celestial
        cd = hdr.get("CDELT1", hdr.get("CD1_1"))
        pixscale = abs(float(cd)) * 3600.0
    f = max(1, int(round(sky_planck_column.BLOCK_TARGET_ARCSEC / pixscale)))
    ra_lo, ra_hi, dec_lo, dec_hi = rbox
    ra_lo, ra_hi = ra_lo - pad_deg, ra_hi + pad_deg
    dec_lo, dec_hi = dec_lo - pad_deg, dec_hi + pad_deg
    xs, ys = wcs.wcs_world2pix([ra_lo, ra_lo, ra_hi, ra_hi],
                               [dec_lo, dec_hi, dec_lo, dec_hi], 0)
    x0 = int(np.clip((int(np.floor(np.nanmin(xs))) // f) * f, 0, nx))
    x1 = int(np.clip(int(np.ceil((np.nanmax(xs) + 1) / f)) * f, 0, nx))
    y0 = int(np.clip((int(np.floor(np.nanmin(ys))) // f) * f, 0, ny))
    y1 = int(np.clip(int(np.ceil((np.nanmax(ys) + 1) / f)) * f, 0, ny))
    return dict(wcs=wcs, x0=x0, x1=x1, y0=y0, y1=y1, f=f, pixscale=pixscale)


def _map_block_a_k(path, window):
    """One HGBS map's own valid cells inside its precomputed pixel
    `window` (`_map_window`) -- the FITS section is read off disk via
    memmap, never the whole mosaic (some HGBS mosaics run to several GB)
    -- block-reduced to `sky.derived.planck_column.BLOCK_TARGET_ARCSEC`
    (~12", already well-sampled relative to Herschel's own 36.3" beam):
    `(hpx512, a_k)`, the block's own nside-512 NESTED galactic pixel and
    its `A_K` (`sky.derived.herschel_column`'s currency), vectorised over
    the cutout at once (rule 8)."""
    x0, x1, y0, y1 = window["x0"], window["x1"], window["y0"], window["y1"]
    if x1 <= x0 or y1 <= y0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    from astropy.io import fits
    with fits.open(path, memmap=True) as hd:
        chosen = next(c for c in hd if c.header.get("NAXIS", 0) >= 2)
        raw = np.asarray(chosen.data[..., y0:y1, x0:x1])
    data = np.squeeze(raw).astype(np.float32, copy=False)
    while data.ndim > 2:
        data = data[0]
    cutout_wcs = window["wcs"].deepcopy()
    cutout_wcs.wcs.crpix = cutout_wcs.wcs.crpix - np.array([x0, y0])
    f = window["f"]
    finite = np.isfinite(data) & (data > 0)
    bs, bn, _ny2, _nx2 = sky_planck_column._block_reduce(data, finite, f)
    bmean = np.where(bn > 0, bs / np.maximum(bn, 1), np.nan)
    by, bx = np.mgrid[0:bmean.shape[0], 0:bmean.shape[1]]
    sel = bn > 0
    if not sel.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    xc = bx[sel] * f + (f - 1) / 2.0
    yc = by[sel] * f + (f - 1) / 2.0
    ra, dec = cutout_wcs.wcs_pix2world(xc.astype(np.float64), yc.astype(np.float64), 0)
    gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    hpx = hp.ang2pix(NSIDE_ANCHOR, gal.l.deg, gal.b.deg, nest=True, lonlat=True)
    a_k = bmean[sel].astype(np.float64) * sky_herschel_column.NH2_TO_AK
    return hpx.astype(np.int64), a_k


def _region_bbox_icrs(hpx_pix_512):
    """(ra_min, ra_max, dec_min, dec_max) of a set of nside-512 NESTED
    galactic pixel centres -- the same bbox-overlap screen
    `sky.derived.herschel_column._boxes_overlap` runs on, so only maps
    that can possibly touch this pixel set are opened."""
    l_deg, b_deg = hp.pix2ang(NSIDE_ANCHOR, hpx_pix_512, nest=True, lonlat=True)
    icrs = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic").icrs
    ra, dec = icrs.ra.deg, icrs.dec.deg
    return float(ra.min()), float(ra.max()), float(dec.min()), float(dec.max())


def _herschel_pixel_stats(config, pix_sorted):
    """`(sum_sq, count)`, each `(n_pix,)`, aligned to `pix_sorted`
    (ascending): the sum and count of `A_cell^2` over every
    HGBS map cell (`_map_block_a_k`) whose own nside-512 pixel is in the
    set, over every map overlapping the set's own footprint, each map
    read only as the FITS section covering that footprint (`_map_window`
    / `_map_block_a_k`), never the whole mosaic. The joblib pool is
    capped at `floor(_HERSCHEL_POOL_BUDGET_BYTES / (_HERSCHEL_ARRAYS_PER_
    WORKER * max_cutout_bytes))` workers -- 6 GB / (4 x the largest
    candidate cutout, known from `_map_window` before any data is read)
    -- so the worst case (every worker mid-reduction on its own cutout at
    once) stays under 6 GB regardless of `config.n_jobs` or how large the
    overlapping mosaics are. A pixel that two overlapping map reductions
    both touch is not deduplicated -- disclosed: HGBS reductions rarely
    overlap, and the effect on a pixel mean is second order against the
    gain this area integral makes over the source-sampled mean it
    replaces."""
    maps = sky_herschel_column._map_list(config)
    headers = {m["name"]: sky_herschel_column._map_header(m["path"]) for m in maps}
    rbox = _region_bbox_icrs(pix_sorted)
    candidates = [m for m in maps
                  if sky_herschel_column._boxes_overlap(headers[m["name"]]["bbox"], rbox, pad=0.1)]

    windows = [_map_window(m["path"], rbox) for m in candidates]
    cutout_bytes = [4 * max(0, w["x1"] - w["x0"]) * max(0, w["y1"] - w["y0"])
                    for w in windows]
    max_bytes = max(cutout_bytes) if cutout_bytes else 0
    n_jobs_bound = (_HERSCHEL_POOL_BUDGET_BYTES // (_HERSCHEL_ARRAYS_PER_WORKER * max_bytes)
                    if max_bytes else config.n_jobs)
    n_jobs_eff = max(1, int(min(config.n_jobs, n_jobs_bound)))

    results = Parallel(n_jobs=n_jobs_eff)(
        delayed(_map_block_a_k)(m["path"], w) for m, w in zip(candidates, windows))

    n_pix = pix_sorted.size
    sum_sq = np.zeros(n_pix)
    count = np.zeros(n_pix)
    for hpx, a_k in results:
        if hpx.size == 0:
            continue
        loc = np.searchsorted(pix_sorted, hpx)
        capped = np.minimum(loc, max(n_pix - 1, 0))
        matched = (n_pix > 0) & (pix_sorted[capped] == hpx)
        if not matched.any():
            continue
        idx = capped[matched]
        sum_sq += np.bincount(idx, weights=a_k[matched] ** 2, minlength=n_pix)
        count += np.bincount(idx, weights=np.ones(idx.size), minlength=n_pix)
    return sum_sq, count


def _planck_parent_column(config, parent256):
    """`A_K` of every requested nside-256 pixel (`parent256`), from the
    survey-wide Planck sightline product
    (`sky.derived.planck_column.build_column`) -- the fallback arm's own
    column, for pixels the Herschel maps do not reach."""
    path = config_module.product_path(config, "sky/derived", "planck",
                                       "column", "sightline")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.yso: Planck sightline column missing at %s -- run the "
            "'sky.derived.planck_column' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)
    order = np.argsort(sl_pix)
    sl_pix_sorted, a_k_sorted = sl_pix[order], a_k[order]
    loc = np.searchsorted(sl_pix_sorted, parent256)
    capped = np.minimum(loc, max(sl_pix_sorted.size - 1, 0))
    matched = (sl_pix_sorted.size > 0) & (sl_pix_sorted[capped] == parent256)
    if not np.all(matched):
        raise ValueError(
            "prior.yso: an anchor pixel's parent nside-256 sightline is "
            "absent from the Planck column product")
    return a_k_sorted[capped]


def law_area_integral(config, region, hpx_pix_512):
    """`N_law` integrated over each requested nside-512 pixel's own area
    (SPEC_PRIORS.md section 2.1's "young stars in the anchors" row, "the
    law count integrated over the tile"; section 6.4 item 1's per-cloud
    check reuses it). Returns young stars deg^-2, aligned to
    `hpx_pix_512`, in `law_count`'s own convention (a consumer multiplies
    by the pixel's own solid angle for a count) -- but genuinely
    area-averaged over the column MAP, not over the pixel's own catalogued
    sources, which avoid the densest gas and under-run a convex (squared)
    law.

    Herschel-covered pixels (any HGBS map cell falls inside,
    `_herschel_pixel_stats` at the map's own beam-scale resolution):

        kappa_H * mean_over_map_cells[ A_cell^2 ]

    No coverage-fraction weighting: a pixel only partly inside the HGBS
    mosaic still carries Gaia/2MASS detections across its WHOLE area,
    since the anchor histograms are all-sky (SPEC_PRIORS.md section 2.1)
    and never gated on the Herschel footprint; the covered cells' own mean
    is the best estimate available for the pixel's whole area and stands
    for it, rather than being scaled down by a coverage fraction that
    would zero out real young stars the anchors do see.

    Elsewhere, the Planck sightline column `A_P` of the parent nside-256
    pixel, with the log-normal kernel's own closed-form second raw
    moment (the true-column dispersion the Planck beam hides,
    SPEC_PRIORS.md section 1.2):

        kappa_P * E[T^2 | A_P]
    """
    hpx_pix_512 = np.asarray(hpx_pix_512, dtype=np.int64)
    order = np.argsort(hpx_pix_512)
    pix_sorted = hpx_pix_512[order]

    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    pc2 = pc2_per_deg2(d_r_pc)
    kappa_h_full = KAPPA_HERSCHEL * pc2
    kappa_p_full = KAPPA_PLANCK * pc2

    sum_sq, count = _herschel_pixel_stats(config, pix_sorted)
    herschel_covered = count > 0
    herschel_value = kappa_h_full * np.divide(
        sum_sq, count, out=np.zeros_like(sum_sq), where=herschel_covered)

    a_p = _planck_parent_column(config, pix_sorted >> 2)
    planck_value = kappa_p_full * _kernel_second_moment_planck(config, a_p)

    value_sorted = np.where(herschel_covered, herschel_value, planck_value)
    out = np.empty_like(value_sorted)
    out[order] = value_sorted
    return out


# ====================================================================
# 6.3 -- the shape: embedding density, ridge, kernel-convolved marginal
# ====================================================================

#: `rho_YSO ~ rho_gas^alpha` (Parmentier & Pfalzner 2013, A&A 549, A132):
#: the 3-D exponent behind the observed Sigma_YSO ~ Sigma_gas^2 surface
#: law. The embedding density per unit `u` is `rho_gas^(alpha-1)`; the
#: ridge weight is `rho_gas^alpha` (SPEC_PRIORS.md section 6.3).
ALPHA_YSO = 1.5

#: The brightness axis's reference distance (SPEC_PRIORS.md section 6.3):
#: `log10 B(d) = -2 log10(d / 1 kpc)`, the fitter's library reference.
D_REF_PC = 1000.0

#: A density floor guarding the alpha-1 power against an exact zero cell
#: (none occur in the measured profiles; kept as the same defensive floor
#: the profile's own `RHO_K_PER_PC` construction uses).
_RHO_FLOOR = np.finfo(np.float64).tiny


def _profile_path(config, region):
    return config_module.product_path(config, "sky/derived", "edenhofer",
                                       "profile", "sightline", region=region)


def _load_profile_arrays(config, region):
    """The raw sightline-profile arrays this shape is built from
    (`sky.derived.profile.build`'s own product, SPEC_PRIORS.md section
    1.4): `A_CUM_K` already normalised so its own edge never exceeds
    `A_INF_K`, and the far-field tail's residual column and e-folding
    scale past that edge."""
    path = _profile_path(config, region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.yso: sightline profile missing for region %r at %s -- "
            "run the 'sky.derived.profile' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(
            hpx_pix_256=np.asarray(f["HPX_PIX_256"][:], dtype=np.int64),
            dist_pc=np.asarray(f["DIST_PC"][:], dtype=np.float64),
            a_cum_k=np.asarray(f["A_CUM_K"][:], dtype=np.float64),
            a_inf_k=np.asarray(f["A_INF_K"][:], dtype=np.float64),
            rho_k_per_pc=np.asarray(f["RHO_K_PER_PC"][:], dtype=np.float64),
            tail_residual_k=np.asarray(f["TAIL_RESIDUAL_K"][:], dtype=np.float64),
            tail_efold_pc=np.asarray(f["TAIL_EFOLD_PC"][:], dtype=np.float64),
        )


def embedding_and_ridge(profile):
    """The whole region's `u` edges, embedding density, and a-B ridge in
    one vectorised pass -- no Python loop over sightlines, cells or nodes
    (SPEC_PRIORS.md section 6.3).

    The 3-D map's own cells (`RHO_K_PER_PC`, `n_d-1` of them) are joined
    by one more cell carrying the far-field tail's residual column, so
    `u` reaches 1 on every sightline (section 1.4). That cell's own
    characteristic density and depth stand in for its exponential decay
    (`TAIL_RESIDUAL_K`, `TAIL_EFOLD_PC`): the mean of an exponential
    population with that e-folding scale sits one scale length past the
    map's edge, and its density there is the residual column spread over
    that same scale length -- an analytic stand-in disclosed here, not a
    second placement law.

    Returns a dict of `(n_sl, ...)` arrays: `u_edges` (n_sl, n_cell+1),
    `p_u` (n_sl, n_cell) normalised over u in [0, 1], `u_median` (report,
    section 6.3's "spread of the per-sightline u medians"), and the ridge
    `intercept`/`slope`/`resid_sigma`/`corr` (n_sl,).
    """
    dist_pc = profile["dist_pc"]
    a_cum = profile["a_cum_k"]
    a_inf = profile["a_inf_k"]
    tail_residual = profile["tail_residual_k"]
    tail_efold = profile["tail_efold_pc"]
    n_sl, n_d = a_cum.shape

    d_widths = np.diff(dist_pc)                                   # (n_d-1,)
    rho_map = np.maximum(profile["rho_k_per_pc"], _RHO_FLOOR)      # (n_sl, n_d-1)
    rho_tail = np.maximum(tail_residual / tail_efold, _RHO_FLOOR)  # (n_sl,)
    rho_full = np.concatenate([rho_map, rho_tail[:, None]], axis=1)   # (n_sl, n_d)
    width_full = np.concatenate(
        [np.broadcast_to(d_widths, (n_sl, n_d - 1)), tail_efold[:, None]],
        axis=1)                                                    # (n_sl, n_d)

    # u = A(d)/A(inf) (section 1.4): the map's own cumulative edges,
    # followed by the sightline's total column at u = 1 exactly, so the
    # residual cell closes the support.
    a_edges = np.concatenate([a_cum, a_inf[:, None]], axis=1)      # (n_sl, n_d+1)
    u_edges = a_edges / a_inf[:, None]
    u_edges[:, -1] = 1.0
    u_widths = np.diff(u_edges, axis=1)                            # (n_sl, n_d)
    a_rep = 0.5 * (a_edges[:, :-1] + a_edges[:, 1:])                # (n_sl, n_d)

    d_mid_map = 0.5 * (dist_pc[:-1] + dist_pc[1:])                  # (n_d-1,)
    d_mid_tail = dist_pc[-1] + tail_efold                           # (n_sl,)
    d_mid = np.concatenate(
        [np.broadcast_to(d_mid_map, (n_sl, n_d - 1)), d_mid_tail[:, None]],
        axis=1)                                                    # (n_sl, n_d)
    log10_b = -2.0 * np.log10(d_mid / D_REF_PC)

    # the embedding density per unit u (section 6.3): rho_gas^(alpha-1).
    density_raw = rho_full ** (ALPHA_YSO - 1.0)
    mass = np.sum(density_raw * u_widths, axis=1, keepdims=True)
    p_u = density_raw / mass

    # the median u per sightline (report, section 6.3): interpolated off
    # the same piecewise-constant density's own cumulative mass.
    cdf = np.concatenate(
        [np.zeros((n_sl, 1)), np.cumsum(density_raw * u_widths, axis=1) / mass],
        axis=1)                                                    # (n_sl, n_d+1)
    idx = np.clip(np.sum(cdf <= 0.5, axis=1) - 1, 0, n_d - 1)
    cdf_lo = np.take_along_axis(cdf, idx[:, None], axis=1)[:, 0]
    cdf_hi = np.take_along_axis(cdf, (idx + 1)[:, None], axis=1)[:, 0]
    u_lo = np.take_along_axis(u_edges, idx[:, None], axis=1)[:, 0]
    u_hi = np.take_along_axis(u_edges, (idx + 1)[:, None], axis=1)[:, 0]
    frac = np.where(cdf_hi > cdf_lo,
                     (0.5 - cdf_lo) / np.maximum(cdf_hi - cdf_lo, 1e-300), 0.0)
    u_median = u_lo + frac * (u_hi - u_lo)

    # the ridge (section 6.3): the rho^alpha-weighted least-squares line
    # of log10 B on a -- the same population weight the embedding density
    # is built from, so the two are one placement read on two axes.
    w = rho_full ** ALPHA_YSO * width_full
    wn = w / np.sum(w, axis=1, keepdims=True)
    mean_a = np.sum(wn * a_rep, axis=1)
    mean_b = np.sum(wn * log10_b, axis=1)
    var_a = np.sum(wn * (a_rep - mean_a[:, None]) ** 2, axis=1)
    var_b = np.sum(wn * (log10_b - mean_b[:, None]) ** 2, axis=1)
    cov = np.sum(wn * (a_rep - mean_a[:, None]) * (log10_b - mean_b[:, None]), axis=1)
    slope = cov / var_a
    intercept = mean_b - slope * mean_a
    resid_var = np.maximum(var_b - slope * cov, 0.0)
    corr = np.where(var_b > 0, cov / np.sqrt(var_a * var_b), np.nan)

    return dict(u_edges=u_edges, p_u=p_u, u_median=u_median,
                ridge_intercept=intercept, ridge_slope=slope,
                ridge_resid_sigma=np.sqrt(resid_var), ridge_corr=corr)


def _majority_map_class(config, region, sl_pix):
    """Per sightline, the map class (`herschel`/`planck`) of the majority
    of its own sources' adopted-column arm (SPEC_PRIORS.md section 6.3's
    tabulation, "map class from the sightline's majority arm"). A
    sightline the granule map admits by Spitzer coverage alone, with no
    catalogued source of its own, has no arm to poll and falls back to
    Planck, the all-sky arm (section 1.1)."""
    src_pix = access.region_slice(config, region)["hpx_pix_256"]
    _, _, provenance = _adopted_columns(config, region)
    loc = np.searchsorted(sl_pix, src_pix)
    capped = np.minimum(loc, max(sl_pix.size - 1, 0))
    if src_pix.size and not np.all(sl_pix[capped] == src_pix):
        raise ValueError(
            "prior.yso: %r has a source whose own pixel is absent from its "
            "region's sightline profile" % region)
    is_herschel = (provenance == PROVENANCE_HERSCHEL).astype(float)
    n_herschel = np.bincount(loc, weights=is_herschel, minlength=sl_pix.size)
    n_total = np.bincount(loc, minlength=sl_pix.size)
    majority_herschel = (n_total > 0) & (n_herschel * 2 > n_total)
    return np.where(majority_herschel, "herschel", "planck")


def _normal_cdf(z):
    """Standard normal CDF `Phi(z)`, `z` any shape, `+/-inf` allowed."""
    return 0.5 * (1.0 + erf(z / _SQRT2))


def _lognormal_cdf(t, m, s):
    """`P(T <= t)`, `ln T ~ Normal(m, s)`; `t = 0` and `t = inf` both
    give the correct limits (0 and 1) through `Phi(+/-inf)`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.log(t) - m) / s
    return _normal_cdf(z)


def _lognormal_inv_moment_upto(t, m, s):
    """`E[1/T ; T <= t]` for `ln T ~ Normal(m, s)`: the log-normal's
    partial inverse moment, `exp(-m + s^2/2) * Phi((ln t - m)/s - s)`
    (SPEC_PRIORS.md section 6.3's derivation). `t = 0` gives 0, `t = inf`
    gives the total `E[1/T] = exp(-m + s^2/2)`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.log(t) - m) / s - s
    return np.exp(-m + 0.5 * s * s) * _normal_cdf(z)


class YsoShape(object):
    """One region's YSO shape, read once and evaluated exactly
    (SPEC_PRIORS.md section 6.3): the per-sightline embedding density
    (`U_EDGES`/`P_U`, a step function on `u`) and the survey's log-normal
    column kernel (`prior.kernel.Kernel`). The marginal

        P(a <= a0 | A_s) = E_T[F_u(a0 / T)]

    is the exact expectation of the embedding CDF under `T`'s log-normal
    distribution given `A_s`: `F_u` is piecewise linear, so this splits
    over its cells into closed-form terms of the normal CDF and the
    log-normal's partial inverse moment (module docstring), evaluated at
    the exact query `a`, never on a fixed grid; the density `p(a0|A_s)`
    is the matching closed form for `F_u`'s own derivative, `p_u`.

    Two ways to supply the per-source `(mu, sigma)` this needs from the
    kernel:

    - `marginal_exact`/`cdf_exact` take the source's own column and
      measurement uncertainty directly.
    - `marginal_at`/`cdf_at` keep the node-bracket signature
      (IMPLEMENTATION.md section 2) other consumers already call:
      `node_lo`/`node_w` are read back through the column grid to
      recover the source's own `A_s` exactly (the inverse of
      `column_grid.bracket`, which is linear in `A_s`), at zero
      source-measurement uncertainty -- the node bracket carries no
      per-source sigma, so this path drops that one (usually small)
      term rather than adding a fifth argument to a shared signature.
    """

    def __init__(self, u_edges, p_u, is_herschel, kernel, nodes_arr,
                 hpx_pix_256, sightline_id):
        self.u_edges = u_edges                      # (n_sl, n_cell+1)
        self.p_u = p_u                               # (n_sl, n_cell)
        self.is_herschel = is_herschel               # (n_sl,) bool
        self.kernel = kernel                         # prior.kernel.Kernel
        self.nodes_arr = nodes_arr                   # (n_node,) column grid
        self.n_node = nodes_arr.size
        self.hpx_pix_256 = hpx_pix_256
        self.sightline_id = sightline_id
        # the embedding density's own cumulative mass at every u edge --
        # the exact integral of a step function is piecewise linear
        # (SPEC_PRIORS.md section 6.3); CUM_U[:, -1] = 1 by p_u's own
        # normalisation (`embedding_and_ridge`).
        widths = np.diff(u_edges, axis=1)
        self.cum_u = np.concatenate(
            [np.zeros((u_edges.shape[0], 1)), np.cumsum(p_u * widths, axis=1)],
            axis=1)

    @classmethod
    def read(cls, config, region):
        """Reads one region's `bms/yso/prior_yso_sightline` product
        (`build_shape`'s own output) plus the survey's column kernel and
        node grid, shared across every region."""
        path = config_module.product_path(config, "bms", "yso", "prior",
                                           "sightline", region=region)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "prior.yso.YsoShape: no shape product for region %r at %s "
                "-- run the 'prior.yso' RUNBOOK line first" % (region, path))
        with h5py.File(path, "r") as f:
            u_edges = np.asarray(f["U_EDGES"][:], dtype=np.float64)
            p_u = np.asarray(f["P_U"][:], dtype=np.float64)
            is_herschel = np.asarray(f["IS_HERSCHEL"][:]).astype(bool)
            hpx_pix_256 = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
            sightline_id = np.asarray(f["SIGHTLINE_ID"][:], dtype=np.int64)
        kernel = kernel_module.Kernel.read(config)
        nodes_arr = column_grid.nodes(config)
        return cls(u_edges, p_u, is_herschel, kernel, nodes_arr,
                   hpx_pix_256, sightline_id)

    # -- the closed-form extinction marginal, given per-source (A_s, mu,
    # sigma) already matched to the query rows' own sightlines ---------
    def _cdf_rows(self, a, rows, a_col, mu, sigma):
        """`P(a' <= a | A_s) = E_T[F_u(a/T)]`, closed form (module
        docstring): the `T <= a` bulk (`F_u = 1` there) plus, cell by
        cell of the embedding step function, the normal-CDF mass and the
        log-normal partial inverse moment of `T` in that cell's own
        `a`-implied range. A Python loop over the (small, fixed) number
        of embedding cells, vectorised over every (row, a) pair."""
        a = np.asarray(a, dtype=float)
        m = np.log(a_col) + mu * _LN10
        s = sigma * _LN10
        a_safe = np.where(a > 0.0, a, 1.0)
        edges = self.u_edges[rows]
        cum_u = self.cum_u[rows]
        p_u = self.p_u[rows]
        total = _lognormal_cdf(a_safe, m, s)
        for k in range(p_u.shape[1]):
            e_lo, e_hi = edges[:, k], edges[:, k + 1]
            with np.errstate(divide="ignore"):
                hi_k = a_safe / e_lo
            lo_k = a_safe / e_hi
            d_cdf = _lognormal_cdf(hi_k, m, s) - _lognormal_cdf(lo_k, m, s)
            d_inv = (_lognormal_inv_moment_upto(hi_k, m, s)
                     - _lognormal_inv_moment_upto(lo_k, m, s))
            total = total + cum_u[:, k] * d_cdf + p_u[:, k] * (a_safe * d_inv - e_lo * d_cdf)
        return np.where(a > 0.0, np.clip(total, 0.0, 1.0), 0.0)

    def _marginal_rows(self, a, rows, a_col, mu, sigma):
        """`p(a | A_s) = sum_k p_k * E[1/T; a/T in cell k]`, closed form
        (module docstring), matching `_cdf_rows`'s own cells."""
        a = np.asarray(a, dtype=float)
        m = np.log(a_col) + mu * _LN10
        s = sigma * _LN10
        a_safe = np.where(a > 0.0, a, 1.0)
        edges = self.u_edges[rows]
        p_u = self.p_u[rows]
        total = np.zeros_like(a_safe)
        for k in range(p_u.shape[1]):
            e_lo, e_hi = edges[:, k], edges[:, k + 1]
            with np.errstate(divide="ignore"):
                hi_k = a_safe / e_lo
            lo_k = a_safe / e_hi
            d_inv = (_lognormal_inv_moment_upto(hi_k, m, s)
                     - _lognormal_inv_moment_upto(lo_k, m, s))
            total = total + p_u[:, k] * d_inv
        return np.where(a > 0.0, np.maximum(total, 0.0), 0.0)

    def _map_class(self, rows):
        return np.where(self.is_herschel[rows], "herschel", "planck")

    # -- public: each source's own exact column and measurement error --
    def marginal_exact(self, a, rows, a_col, sigma_col):
        """`p(a | A_s)` at a batch of sources, each at its own exact
        adopted column and measurement uncertainty (SPEC_PRIORS.md
        section 1.2/6.3): no node bracket, no approximation."""
        mu, sigma = self.kernel.params(a_col, sigma_col, self._map_class(rows))
        return self._marginal_rows(a, rows, a_col, mu, sigma)

    def cdf_exact(self, a, rows, a_col, sigma_col):
        """`cdf` at each source's own exact column and measurement
        uncertainty, matching `marginal_exact`."""
        mu, sigma = self.kernel.params(a_col, sigma_col, self._map_class(rows))
        return self._cdf_rows(a, rows, a_col, mu, sigma)

    # -- public: the node-bracket signature (IMPLEMENTATION.md section 2)
    def _node_bracket_column(self, node_lo, node_w):
        """`A_s`, recovered exactly from `(NODE_LO, NODE_W)` -- the
        inverse of `column_grid.bracket`, which is linear in `A_s`."""
        node_lo = np.asarray(node_lo, dtype=np.intp)
        node_hi = np.clip(node_lo + 1, 0, self.n_node - 1)
        lo, hi = self.nodes_arr[node_lo], self.nodes_arr[node_hi]
        return lo + np.asarray(node_w, dtype=float) * (hi - lo)

    def marginal_at(self, a, rows, node_lo, node_w):
        """`p(a | A_s)` at a batch of sources, each with its own
        sightline row and node bracket (class docstring: `A_s` exact,
        source measurement uncertainty dropped)."""
        rows = np.asarray(rows, dtype=np.intp)
        a_col = self._node_bracket_column(node_lo, node_w)
        mu, sigma = self.kernel.params(a_col, np.zeros_like(a_col), self._map_class(rows))
        return self._marginal_rows(a, rows, a_col, mu, sigma)

    def cdf_at(self, a, rows, node_lo, node_w):
        """`cdf` at a batch of sources via the node bracket, matching
        `marginal_at`."""
        rows = np.asarray(rows, dtype=np.intp)
        a_col = self._node_bracket_column(node_lo, node_w)
        mu, sigma = self.kernel.params(a_col, np.zeros_like(a_col), self._map_class(rows))
        return self._cdf_rows(a, rows, a_col, mu, sigma)


def _write_shape_product(path, hpx_pix_256, sightline_id, embed, is_herschel):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        f.create_dataset("HPX_PIX_256", data=hpx_pix_256)
        f.create_dataset("SIGHTLINE_ID", data=sightline_id)
        f.create_dataset("U_EDGES", data=embed["u_edges"])
        f.create_dataset("P_U", data=embed["p_u"])
        f.create_dataset("U_MEDIAN", data=embed["u_median"])
        f.create_dataset("RIDGE_INTERCEPT", data=embed["ridge_intercept"])
        f.create_dataset("RIDGE_SLOPE", data=embed["ridge_slope"])
        f.create_dataset("RIDGE_WIDTH", data=embed["ridge_resid_sigma"])
        f.create_dataset("RIDGE_CORR", data=embed["ridge_corr"])
        f.create_dataset("IS_HERSCHEL", data=is_herschel.astype(np.int8))


def _sightline_id_lookup(config, region, sl_pix):
    """`SIGHTLINE_ID` per profile row, matched against the granule map's
    own per-source column (IMPLEMENTATION.md section 1). A row the
    granule map admits by coverage alone, with no catalogued source
    sitting in it, carries no survey-wide sightline id at all and is
    marked -1 -- informational only; the prior table joins every product
    through `HPX_PIX_256` (`granules.access.per_source`), never through
    this id."""
    rs = access.region_slice(config, region)
    src_pix, src_sid = rs["hpx_pix_256"], rs["sightline_id"]
    order = np.argsort(src_pix)
    src_pix_sorted, src_sid_sorted = src_pix[order], src_sid[order]
    loc = np.searchsorted(src_pix_sorted, sl_pix)
    capped = np.minimum(loc, max(src_pix_sorted.size - 1, 0))
    matched = (src_pix_sorted.size > 0) & (src_pix_sorted[capped] == sl_pix)
    return np.where(matched, src_sid_sorted[capped], -1).astype(np.int64)


def build_shape(config, region):
    """Writes one region's `prior/yso/prior_yso_sightline` product
    (SPEC_PRIORS.md section 6.3): the embedding shape and a-B ridge for
    every occupied sightline and its own map class -- the exact
    ingredients `YsoShape` evaluates the closed-form a-marginal from
    (with the survey's column kernel, read once at `YsoShape.read`), at
    any `a`, with no tabulation.
    """
    profile = _load_profile_arrays(config, region)
    sl_pix = profile["hpx_pix_256"]
    embed = embedding_and_ridge(profile)
    map_class = _majority_map_class(config, region, sl_pix)
    is_herschel = map_class == "herschel"
    sightline_id = _sightline_id_lookup(config, region, sl_pix)

    path = config_module.product_path(config, "bms", "yso", "prior",
                                       "sightline", region=region)
    _write_shape_product(path, sl_pix, sightline_id, embed, is_herschel)
    return path, embed["u_median"], embed["ridge_resid_sigma"]


def build(config, regions=None):
    """Writes, per region, the YSO shape product (section 6.3) -- the
    embedding density and its map class -- and one 30-row (or subset)
    law product (section 6.1) over `regions` (default: all thirty)."""
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    law_rows = []
    for region in names:
        path, _, _ = build_shape(config, region)
        print("prior.yso: %s -> %s" % (region, path))
        law_rows.append(_law_row(config, region))
    _write_law_product(config, names, law_rows)


if __name__ == "__main__":
    run(build)
