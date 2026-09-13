"""The model's own expectation of young stars in each STAR anchor pixel
and magnitude bin, so the anchor-weight stage can subtract them from the
observed Gaia/2MASS histograms before fitting `W` (SPEC_PRIORS.md
section 2.1, the "young stars in the anchors" row; IMPLEMENTATION.md
section 6, build order stage 7).

Two pieces, per occupied nside-512 anchor pixel (the same pixels and
edges `prior.anchor_tiles.write_histograms` already wrote).

COUNT. The anchor subtraction is an AREA count (SPEC_PRIORS.md section
2.1's "young stars in the anchors" row: "the law count integrated over
the tile"), so the pixel's expectation is the law integrated over the
column MAP inside the pixel, `prior.yso.law_area_integral` (section 6.1's
area form: Herschel-covered pixels integrate the HGBS map's own cells at
their native beam scale; elsewhere the Planck sightline column carries
the kernel's sub-beam variance), times the pixel's own solid angle
(`OMEGA_PIX_DEG2`):

    N_YOUNG_TOTAL(pix) = law_area_integral(pix) * Omega_pix

`N_law` is convex (squared) in column, so the MEAN of the pixel's own
SOURCE-level law counts (`prior.yso.law_count` at each source's own
adopted column and arm) under-runs this integral: SESNA's own sources
avoid the densest gas, so they are a biased-low sample of the pixel's own
column field (Jensen's inequality on the unsampled structure). That
source-sampled quantity is carried alongside, unused past this module, so
the gap is visible:

    N_YOUNG_SOURCE_MEAN(pix) = mean_i[ N_law(A_col_i, arm_i) ] * Omega_pix

MAGNITUDES. Masses are drawn from the Chabrier system IMF over
0.1-1.4 Msun (`prior.yso_selection`'s closed-form IMF and its own mass
range, reused, not re-derived: `chabrier_integral`, `_imf_density_
unnormalized`, `_IMF_NORM`). The installed BHAC15 grid -- the Spitzer,
2MASS AND Gaia tables alike -- stops at 1.4 Msun, so the IMF's own
fraction above it, `yso_selection.IMF_FRAC_ABOVE_1P4` (about 0.1), is
carried as a separate bright count `N_BRIGHT`, disclosed rather than
binned: a >=1.4 Msun, 1 Myr photosphere at a region's own distance is
brighter than every anchor bin's own bright edge (`G_EDGES[0]`,
`KS_EDGES[0]`), so it can never land in a bin -- and no isochrone row
exists above 1.4 Msun to place it at anyway.

Each mass's photosphere is BHAC15's 1 Myr row: Ks from `BHAC15_iso.
2mass` (Vega system, `constants.VEGA_ZERO_POINT_MJY`, `prior.
yso_selection.abs_mag_grid`, reused), G from this module's own read of
`BHAC15_iso.GAIA` (the same block structure, parsed by `yso_selection.
_parse_bhac15_table`, reused; the file's own header states "Type of
calibration used: Vega", the same CALSPEC Alpha Lyrae standard as the
2MASS table, so its zero point is `constants.GAIA_G_VEGA_ZP_MJY`, Riello
et al. 2021 -- named here though never multiplied in: every quantity
below stays in magnitudes and the zero point only matters for a flux
conversion this module does not do). Both bands read the region's own
distance `d_r_pc` (section 2.1: "at the region distance"), never a
per-source distance.

Extinction is drawn from the pixel's own embedding density: `a = A_pix *
u`, `u` on the parent nside-256 sightline's own `U_EDGES`/`P_U`
(`prior.yso.build_shape`'s product, section 6.3, read unmodified -- a
young star in the anchor's own count is the same embedded population the
YSO prior places, so it sits on the same ray). `Ks_obs = Ks + a`;
`G_obs = G + a * kappa_G(a)`, `kappa_G` read off the SAME law curves
every other band's kappa comes from (`prior.selection._load_law_curve`,
`law_dense_weight`), just at Gaia G's own pivot wavelength (0.64 um)
instead of one of the eight survey bands -- blended by the section 1.3
ramp exactly like every other band, not a new Gaia-specific curve.
Gaia's own detection sigmoid (`prior.anchor_tiles.gaia_detection_
weight`, reused) weights `N_G_YOUNG`; `N_KS_YOUNG` carries none (2MASS
is treated complete to `KS_CUT_MAG`, the anchor histograms' own
convention).

DISCLOSED (section 2.1, once): the photosphere carries no disk excess.
Real Class I/II young stars are brighter than a bare photosphere by a
few tenths of a magnitude in Ks, so this subtraction is conservative
(undercounts) by about that much, worst in the youngest, most embedded
clusters.

Products, per region, `bms/anchors/young-stars_anchors_hpx512__<Region>
.hdf5`: `HPX_PIX_512`; `N_G_YOUNG` (n_pix, n_G_bins), Gaia-weighted;
`N_KS_YOUNG` (n_pix, n_Ks_bins), the 1 Myr isochrone; `N_KS_YOUNG_3MYR`
(n_pix, n_Ks_bins), the same construction at 3 Myr for the section 6.2
age band; `N_YOUNG_TOTAL` (n_pix), the area-integrated count above;
`N_YOUNG_SOURCE_MEAN` (n_pix), the source-sampled count beside it so the
area-integration gap is visible; `N_BRIGHT` (n_pix), the disclosed
>1.4 Msun share of `N_YOUNG_TOTAL`, never binned; `G_EDGES`, `KS_EDGES`
(copied from the histograms product so a consumer never has to
cross-open it); root attrs `GRANULE="hpx512"`, `N_YOUNG_TOTAL_REGION` and
`N_YOUNG_SOURCE_MEAN_REGION` (the region sums of the two counts above).

Algebraic acceptance (reported at build, per pixel, to 1e-9): the 1 Myr
Ks histogram's own bins, plus whatever of the binned (0.1-1.4 Msun)
population falls beyond `KS_EDGES`'s faint edge (`Ks_obs >= 14.3`) or
its bright edge (`Ks_obs < KS_EDGES[0]`, expected negligible -- "bin
edges cover the bright side" -- and reported), plus `N_BRIGHT`, sum to
exactly `N_YOUNG_TOTAL`: the mass quadrature's own weight is forced to
sum to the IMF's exact in-range fraction and the embedding density's own
`u`-cell weights already sum to exactly 1 (`prior.yso.embedding_and_
ridge`), so nothing but the three named pieces can carry the total.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.prior import anchor_tiles
from sesnaimpute.prior import selection
from sesnaimpute.prior import yso
from sesnaimpute.prior import yso_selection

# ---------------------------------------------------------------------
# constants -- every number cited
# ---------------------------------------------------------------------

#: BHAC15_iso.GAIA's own per-row columns (READ_INFO; the same block
#: structure `yso_selection._parse_bhac15_table` already reads for the
#: Spitzer and 2MASS tables), in the file's own header order.
_GAIA_ROW_COLUMNS = (
    "MASS_MSUN", "TEFF_K", "LOGL", "LOGG", "R_RSUN", "LI_LI0",
    "F33", "F33B", "F41", "F45B", "F47", "F51", "FHA", "F57", "F63B",
    "F67", "F75", "F78", "F82", "F82B", "F89",
    "G_RSV", "G", "G_BP", "G_RP",
)

#: Gaia G's own mean/pivot wavelength (Gaia DR3 documentation, an
#: unreddened source): where `kappa_g` reads the law curves, in place of
#: one of `definitions.BANDS`'s own wavelengths.
GAIA_G_PIVOT_UM = 0.64

#: This module's own mass-quadrature resolution (module docstring's IMF
#: x u vectorisation); `yso_selection.mass_quadrature` uses 500 points
#: for its own, finer selection-table need -- reused logic, this
#: module's own point count.
N_MASS_QUADRATURE = 200

#: The Chabrier IMF's own mass fraction inside the installed isochrone's
#: range (`yso_selection.IMF_FRAC_ABOVE_1P4` is its exact complement) --
#: partition of the whole IMF, exact by construction.
MASS_FRAC_IN_RANGE = 1.0 - yso_selection.IMF_FRAC_ABOVE_1P4

#: Anchor pixels per joblib block: bounds one block's `(n_pix, n_mass,
#: n_u)` array instead of holding a whole region's at once (rule 8/9),
#: the same blocking idiom `prior.yso._sightline_block` uses.
PIXEL_BLOCK = 32

_GAIA_TABLE_CACHE = {}


# ---------------------------------------------------------------------
# the Gaia photosphere -- this module's own read of BHAC15_iso.GAIA
# ---------------------------------------------------------------------

def _load_gaia_table(config):
    key = config.data_root
    if key not in _GAIA_TABLE_CACHE:
        path = f"{config.data_root}/sky/download/baraffe2015_bhac15/BHAC15_iso.GAIA"
        _GAIA_TABLE_CACHE[key] = yso_selection._parse_bhac15_table(path, _GAIA_ROW_COLUMNS)
    return _GAIA_TABLE_CACHE[key]


def gaia_abs_mag_grid(config, age_gyr, mass_grid):
    """Abs G at the isochrone's exact `age_gyr` row, linear
    interpolation in log10(mass) against the table's own tabulated
    masses -- the same convention `yso_selection.abs_mag_grid` uses for
    the other eight bands (module docstring: Vega system).
    """
    table = _load_gaia_table(config)
    mask = np.isclose(table[:, 0], age_gyr, atol=1e-6)
    if not np.any(mask):
        raise ValueError(
            f"prior.young_stars: no t={age_gyr:.4f} Gyr block in BHAC15_iso.GAIA")
    col_idx = 1 + _GAIA_ROW_COLUMNS.index("G")
    masses = table[mask, 1]
    g_mag = table[mask, col_idx]
    order = np.argsort(masses)
    return np.interp(np.log10(np.asarray(mass_grid, dtype=float)),
                      np.log10(masses[order]), g_mag[order])


def kappa_g(config, a):
    """`kappa_G(a) = chi(0.64um) / chi(Ks)`, blended by the section 1.3
    ramp exactly as `prior.selection.kappa_hybrid` blends the eight
    survey bands -- the SAME law curves (`prior.selection.
    _load_law_curve`), read at Gaia G's own pivot wavelength instead of
    one of `definitions.BANDS`'s own.
    """
    a = np.asarray(a, dtype=float)
    ks_wvl = definitions.BANDS_BY_KEY["Ks"].wvl_um
    w = selection.law_dense_weight(a)

    def _kappa_g_law(law):
        wave_um, opacity = selection._load_law_curve(config, law)
        chi_g = np.interp(GAIA_G_PIVOT_UM, wave_um, opacity)
        chi_ks = np.interp(ks_wvl, wave_um, opacity)
        return float(chi_g / chi_ks)

    kd = _kappa_g_law(selection.LAW_DIFFUSE)
    kw = _kappa_g_law(selection.LAW_DENSE)
    return (1.0 - w) * kd + w * kw


# ---------------------------------------------------------------------
# the population: Chabrier IMF, this module's own quadrature resolution
# ---------------------------------------------------------------------

def mass_grid_and_weight(n=N_MASS_QUADRATURE):
    """`(mass_grid, weight)`: `n` masses log-spaced over 0.1-1.4 Msun,
    and each point's own trapezoidal probability mass in log10(mass) --
    the same Chabrier density `yso_selection.mass_quadrature` integrates
    (`_imf_density_unnormalized`/`_IMF_NORM`, reused), at this module's
    own point count. `weight.sum()` is forced to exactly `MASS_FRAC_IN_
    RANGE` so the module docstring's algebraic acceptance is exact to
    float64, not left to quadrature error.
    """
    mass_grid = np.geomspace(yso_selection.IMF_MASS_MIN_MSUN,
                              yso_selection.ISOCHRONE_MASS_MAX_MSUN, n)
    density = yso_selection._imf_density_unnormalized(mass_grid) / yso_selection._IMF_NORM
    log10_mass = np.log10(mass_grid)
    dw = np.empty_like(log10_mass)
    dw[0] = 0.5 * (log10_mass[1] - log10_mass[0])
    dw[-1] = 0.5 * (log10_mass[-1] - log10_mass[-2])
    dw[1:-1] = 0.5 * (log10_mass[2:] - log10_mass[:-2])
    weight = density * dw
    weight *= MASS_FRAC_IN_RANGE / weight.sum()
    return mass_grid, weight


# ---------------------------------------------------------------------
# per-pixel inputs: the law count and the parent sightline's placement
# ---------------------------------------------------------------------

def law_count_per_pixel(config, region, pixels):
    """`N_law` averaged per pixel (deg^-2), aligned to `pixels` -- the
    SOURCE-sampled mean of each pixel's own sources' `prior.yso.law_count`
    (module docstring's `N_YOUNG_SOURCE_MEAN`), kept only as the reported
    comparison against the area-integrated `prior.yso.law_area_integral`
    this module now uses for `N_YOUNG_TOTAL`.
    """
    rs = access.region_slice(config, region)
    a_col, _, provenance = yso._adopted_columns(config, region)
    src_pix = rs["hpx_pix_512"]
    law_src = yso.law_count(config, region, a_col, provenance)

    uniq_pix, inverse = np.unique(src_pix, return_inverse=True)
    if not np.array_equal(uniq_pix, pixels):
        raise ValueError(
            "prior.young_stars: %r's granule-map pixel set disagrees with "
            "bms.anchors.histograms's own pixel set -- rerun the "
            "'prior.anchor_tiles' RUNBOOK line for this region" % region)
    n_src = np.bincount(inverse, minlength=pixels.size).astype(np.float64)
    sum_law = np.bincount(inverse, weights=law_src, minlength=pixels.size)
    return sum_law / n_src


def sightline_lookup(config, region, pixels):
    """`(xi_edges, p_u)` gathered per anchor pixel from its own parent
    nside-256 sightline's `prior.yso` shape product (section 6.3). The
    parent is the HEALPix NESTED ancestor, `pixels >> 2` -- no per-
    source lookup needed, since an anchor pixel's own sources are the
    same sources that put its parent sightline in that product.
    """
    path = config_module.product_path(config, "bms", "yso", "prior",
                                       "sightline", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.young_stars: YSO shape product missing for region %r at "
            "%s -- run the 'prior.yso' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        sl_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        xi_edges = np.asarray(f["U_EDGES"][:], dtype=np.float64)  # stored as U_EDGES; the depth fraction ξ's profile edges
        p_u = np.asarray(f["P_U"][:], dtype=np.float64)  # stored as P_U; the depth fraction ξ's profile density
    order = np.argsort(sl_pix)
    sl_pix_sorted = sl_pix[order]
    parent256 = pixels >> 2
    loc = np.searchsorted(sl_pix_sorted, parent256)
    capped = np.minimum(loc, max(sl_pix_sorted.size - 1, 0))
    matched = (sl_pix_sorted.size > 0) & (sl_pix_sorted[capped] == parent256)
    if not np.all(matched):
        raise ValueError(
            "prior.young_stars: %r has an anchor pixel whose parent "
            "nside-256 sightline is absent from its own prior.yso "
            "product" % region)
    idx = order[capped]
    return xi_edges[idx], p_u[idx]


# ---------------------------------------------------------------------
# the vectorised block: IMF x u-cells for one group of pixels
# ---------------------------------------------------------------------

def _weighted_hist(values, weights, edges):
    """`(counts, faint_overflow, bright_overflow)`: `counts` (n_blk,
    n_bins) is the weighted histogram of `values`/`weights` (n_blk,
    n_mass, n_u) into `edges`, one `bincount` over a flattened
    pixel*bin index (no Python loop below the block, rule 8);
    `faint_overflow`/`bright_overflow` (n_blk,) are the weight landing
    at or past `edges[-1]` / below `edges[0]`, so `counts.sum(axis=1) +
    faint_overflow + bright_overflow` recovers the block's own total
    weight exactly (the module docstring's acceptance identity).
    """
    n_blk = values.shape[0]
    n_bins = edges.size - 1
    pix_idx = np.broadcast_to(np.arange(n_blk)[:, None, None], values.shape)
    bin_idx = np.searchsorted(edges, values, side="right") - 1
    in_range = (values >= edges[0]) & (values < edges[-1])
    counts = np.bincount(pix_idx[in_range] * n_bins + bin_idx[in_range],
                          weights=weights[in_range],
                          minlength=n_blk * n_bins).reshape(n_blk, n_bins)
    faint_mask = values >= edges[-1]
    bright_mask = values < edges[0]
    faint_overflow = np.bincount(pix_idx[faint_mask], weights=weights[faint_mask],
                                  minlength=n_blk)
    bright_overflow = np.bincount(pix_idx[bright_mask], weights=weights[bright_mask],
                                   minlength=n_blk)
    return counts, faint_overflow, bright_overflow


def _pixel_block(config, a_pix_blk, xi_edges_blk, p_u_blk, n_young_blk,
                  mass_grid, mass_weight, ks_abs_1myr, ks_abs_3myr, g_abs_1myr,
                  mu, g_edges, ks_edges):
    """One block's `(N_G_YOUNG, N_KS_YOUNG, N_KS_YOUNG_3MYR)` and the
    1 Myr Ks acceptance pieces, vectorised over every mass and every
    `u`-cell of every pixel in the block at once.
    """
    u_mid = 0.5 * (xi_edges_blk[:, :-1] + xi_edges_blk[:, 1:])      # (n_blk, n_u)
    u_width = np.diff(xi_edges_blk, axis=1)                        # (n_blk, n_u)
    u_mass = p_u_blk * u_width                                    # sums to 1 per row

    a_rep = a_pix_blk[:, None] * u_mid                            # (n_blk, n_u)
    g_dimming = a_rep * kappa_g(config, a_rep)                    # (n_blk, n_u)

    weight = (n_young_blk[:, None, None] * mass_weight[None, :, None]
              * u_mass[:, None, :])                                # (n_blk, n_mass, n_u)

    ks_app_1myr = ks_abs_1myr + mu
    ks_app_3myr = ks_abs_3myr + mu
    g_app_1myr = g_abs_1myr + mu

    ks_obs_1myr = ks_app_1myr[None, :, None] + a_rep[:, None, :]
    ks_obs_3myr = ks_app_3myr[None, :, None] + a_rep[:, None, :]
    g_obs = g_app_1myr[None, :, None] + g_dimming[:, None, :]

    p_g = anchor_tiles.gaia_detection_weight(g_obs)
    n_g, _, _ = _weighted_hist(g_obs, weight * p_g, g_edges)
    n_ks, ks_faint, ks_bright = _weighted_hist(ks_obs_1myr, weight, ks_edges)
    n_ks_3myr, _, _ = _weighted_hist(ks_obs_3myr, weight, ks_edges)

    return n_g, n_ks, n_ks_3myr, ks_faint, ks_bright


# ---------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------

def build_region(config, region):
    """Computes and writes one region's young-star anchor-pixel product
    (module docstring)."""
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    mu = float(yso_selection.astro_utils.distance_modulus(d_r_pc))

    hist_path = config_module.product_path(config, "bms", "anchors",
                                            "histograms", "hpx512", region=region)
    if not os.path.exists(hist_path):
        raise FileNotFoundError(
            "prior.young_stars: anchor histograms missing for region %r at "
            "%s -- run the 'prior.anchor_tiles' RUNBOOK line first" % (region, hist_path))
    with h5py.File(hist_path, "r") as f:
        pixels = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        a_pix = np.asarray(f["A_PIX_K"][:], dtype=np.float64)
        omega_pix_deg2 = np.asarray(f["OMEGA_PIX_DEG2"][:], dtype=np.float64)
        g_edges = np.asarray(f["G_EDGES"][:], dtype=np.float64)
        ks_edges = np.asarray(f["KS_EDGES"][:], dtype=np.float64)

    n_young_source_mean = law_count_per_pixel(config, region, pixels) * omega_pix_deg2
    n_young_total = yso.law_area_integral(config, region, pixels) * omega_pix_deg2
    xi_edges, p_u = sightline_lookup(config, region, pixels)

    mass_grid, mass_weight = mass_grid_and_weight()
    ks_abs_1myr = yso_selection.abs_mag_grid(
        config, yso_selection.AGE_1MYR_GYR, mass_grid)[:, yso_selection.BAND_KEYS.index("Ks")]
    ks_abs_3myr = yso_selection.abs_mag_grid(
        config, yso_selection.AGE_3MYR_GYR, mass_grid)[:, yso_selection.BAND_KEYS.index("Ks")]
    g_abs_1myr = gaia_abs_mag_grid(config, yso_selection.AGE_1MYR_GYR, mass_grid)

    n_pix = pixels.size
    starts = list(range(0, n_pix, PIXEL_BLOCK))
    blocks = Parallel(n_jobs=config.n_jobs)(
        delayed(_pixel_block)(
            config, a_pix[s:s + PIXEL_BLOCK], xi_edges[s:s + PIXEL_BLOCK],
            p_u[s:s + PIXEL_BLOCK], n_young_total[s:s + PIXEL_BLOCK],
            mass_grid, mass_weight, ks_abs_1myr, ks_abs_3myr, g_abs_1myr,
            mu, g_edges, ks_edges)
        for s in starts)

    n_g_young = np.concatenate([b[0] for b in blocks], axis=0) if blocks else \
        np.empty((0, g_edges.size - 1))
    n_ks_young = np.concatenate([b[1] for b in blocks], axis=0) if blocks else \
        np.empty((0, ks_edges.size - 1))
    n_ks_young_3myr = np.concatenate([b[2] for b in blocks], axis=0) if blocks else \
        np.empty((0, ks_edges.size - 1))
    ks_faint = np.concatenate([b[3] for b in blocks]) if blocks else np.empty(0)
    ks_bright = np.concatenate([b[4] for b in blocks]) if blocks else np.empty(0)

    n_bright = n_young_total * yso_selection.IMF_FRAC_ABOVE_1P4

    # algebraic acceptance (module docstring): every named piece sums
    # back to N_YOUNG_TOTAL, per pixel, to float64 precision.
    closure = n_ks_young.sum(axis=1) + ks_faint + ks_bright + n_bright
    max_rel_dev = float(np.max(np.abs(closure - n_young_total)
                                / np.maximum(n_young_total, 1e-300)))
    if max_rel_dev >= 1e-9:
        raise ValueError(
            "prior.young_stars: %r's N_KS_YOUNG + overflow + N_BRIGHT closure "
            "against N_YOUNG_TOTAL misses 1e-9 (worst %.3e)" % (region, max_rel_dev))

    return dict(
        pixels=pixels, a_pix=a_pix, n_g_young=n_g_young, n_ks_young=n_ks_young,
        n_ks_young_3myr=n_ks_young_3myr, n_young_total=n_young_total,
        n_young_source_mean=n_young_source_mean,
        n_bright=n_bright, g_edges=g_edges, ks_edges=ks_edges,
        ks_faint_overflow=ks_faint, ks_bright_overflow=ks_bright,
        max_rel_dev=max_rel_dev,
    )


def _write_product(config, region, result):
    path = config_module.product_path(config, "bms", "anchors", "young-stars",
                                       "hpx512", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.attrs["N_YOUNG_TOTAL_REGION"] = float(result["n_young_total"].sum())
        f.attrs["N_YOUNG_SOURCE_MEAN_REGION"] = float(result["n_young_source_mean"].sum())
        f.create_dataset("HPX_PIX_512", data=result["pixels"].astype(np.int64))
        f.create_dataset("N_G_YOUNG", data=result["n_g_young"].astype(np.float32))
        f.create_dataset("N_KS_YOUNG", data=result["n_ks_young"].astype(np.float32))
        f.create_dataset("N_KS_YOUNG_3MYR", data=result["n_ks_young_3myr"].astype(np.float32))
        f.create_dataset("N_YOUNG_TOTAL", data=result["n_young_total"].astype(np.float64))
        f.create_dataset("N_YOUNG_SOURCE_MEAN", data=result["n_young_source_mean"].astype(np.float64))
        f.create_dataset("N_BRIGHT", data=result["n_bright"].astype(np.float64))
        f.create_dataset("G_EDGES", data=result["g_edges"])
        f.create_dataset("KS_EDGES", data=result["ks_edges"])
    return path


def build(config, regions=None):
    """Writes, per region (default: all thirty), `bms/anchors/young-
    stars_anchors_hpx512__<Region>.hdf5` (module docstring). Prints,
    per region, the total expected young-star count, the observed 2MASS
    count (`Ks < 14.3`) in the same pixels and their ratio, and the
    algebraic acceptance's worst-pixel relative deviation.
    """
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in names:
        with progress.Stage("prior.young_stars", region) as st:
            result = build_region(config, region)
            path = _write_product(config, region, result)

            hist_path = config_module.product_path(config, "bms", "anchors",
                                                    "histograms", "hpx512", region=region)
            with h5py.File(hist_path, "r") as f:
                n_ks_obs = np.asarray(f["N_KS_OBS"][:], dtype=np.float64).sum(axis=1)

            total_young = float(result["n_young_total"].sum())
            total_source_mean = float(result["n_young_source_mean"].sum())
            total_obs = float(n_ks_obs.sum())
            ratio = total_young / total_obs if total_obs > 0 else float("nan")
            area_over_source = (total_young / total_source_mean
                                 if total_source_mean > 0 else float("nan"))
            share = np.where(n_ks_obs > 0, result["n_young_total"] / n_ks_obs, 0.0)
            i_worst = int(np.argmax(share))
            st.done(path, n_young_total=total_young, ratio_to_2mass=ratio)
        print(
            "prior.young_stars: %s N_YOUNG_TOTAL=%.2f N_YOUNG_SOURCE_MEAN=%.2f "
            "(area/source-mean=%.3fx) 2MASS(Ks<14.3)=%.2f ratio=%.4f "
            "worst-pixel young/2MASS share=%.4f at A_PIX_K=%.3f "
            "max_rel_dev=%.3e -> %s"
            % (region, total_young, total_source_mean, area_over_source,
               total_obs, ratio, share[i_worst],
               result["a_pix"][i_worst], result["max_rel_dev"], path))


if __name__ == "__main__":
    run(build)
