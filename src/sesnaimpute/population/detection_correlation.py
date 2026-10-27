"""C20 brief (briefs/round4_2026-10-20/C20_detection_correlation.md): the
measurement behind LAMBDA, the one survey-wide mixture weight between
SESNA's independent band-detection model (`bmstp.atlas.catalogued_
probability`'s closed form, `fittp.likelihood._ln_nondet`, `fittp.
prior_reader._cell_sum`'s non-detection branch) and a fully comonotone
model (one latent uniform draw per source, shared by every band -- a star
detected in band b iff that one draw clears p_b). A library module with
its own stage-less `main` (no `progress` use, no RUNBOOK line, like
`fittp.likelihood`): the owner runs this by hand against a root.cfg, not
through RUNBOOKtp.sh, and reads LAMBDA's own value, with its provenance
and citation, off `constants.LAMBDA` once this module has measured it.

The method reproduces C18's test1b (`/Users/jtaylor/sesna_scratch/
c18_diag/test1b_empirical_completeness.py`) at the single-star level
instead of a per-Ks-bin aggregate mean, because a maximum-likelihood fit
over LAMBDA needs one Bernoulli catalogued/not-catalogued OUTCOME per
star, not a binned average of outcomes: every UKIDSS GPS clean-stellar
star (`sky.download.ukidss_gps`) inside a region's own deep-covered,
I1-and-I2-covered admitted pixels (the SAME quality cut test1b applies, so
coverage gaps on either side never masquerade as a detection-correlation
signal) is matched to the nearest SESNA source within 1 arcsec; a match
with I1 AND I2 both measured (`ORIGIN_FNU == 1`, the same criterion the
catalogue side applies) is "catalogued" (y=1), every other star is not
(y=0). The empirical chance-coincidence rate from an identical search
offset 10 arcsec away (C18's own check) is folded into the model as a
constant false-positive floor on the match probability, `P(match) =
P_cat + (1 - P_cat) * chance_rate`, rather than subtracted bin by bin from
a mean.

Each star's own model inputs -- `P_indep` (now `p_I1 * p_I2`, the
independent probability of the OBSERVED outcome `_match_outcomes` scores:
I1 AND I2 both detected) and `p_(2)` (now `min(p_I1, p_I2)`, that same
event's comonotone counterpart -- under one shared latent uniform, I1
AND I2 both clear iff the draw is below the SMALLER of the two) -- are
read off `_population_bin_table`: the SAME field-star population
quadrature C18's test1b runs (every STAR member of the region's own
`good`-pixel tiles, `W_STAR * (1 - P_PAHC)` weighted, dimmed through the
atlas's own column and hybrid law, evaluated against the SAME `PIXEL_
CURVE_<band>` table `bmstp.atlas.catalogued_probability` reads), binned
by each MEMBER's own dimmed Ks magnitude into the same half-magnitude
bins test1b uses, and averaged WITHIN EACH BIN before the nonlinear
`P_indep`/`p_(2)` statistics are formed. Every UKIDSS star in that Ks bin
then reads that one pair of numbers.

A first implementation instead inferred each star's own I1/I2/.../M1 flux
directly from its own observed Ks magnitude through the field
population's SINGLE weighted-median colour (one scalar ratio per band,
the reading the brief itself names: "the star's I1/I2 fluxes from its Ks
through the field population's median colours"), then evaluated `P_
indep`/`p_(2)` AT that one inferred flux per star. Checked against
Perseus (module docstring: completeness there is ~0.95 catalogue-wide,
where C18's own test1b shows the independent and comonotone models
cannot differ), that single-flux method manufactured a LAMBDA-saturating
gap test1b's own bin-averaged model does not show -- a Jensen's-gap
artifact of evaluating a nonlinear completeness curve at one population-
median flux instead of averaging the curve's own value over the
population's real flux SPREAD at fixed Ks. Binning by each member's own
dimmed Ks and averaging `P_indep`/`p_(2)` (already nonlinear functions of
flux) across the bin's own population, as below, is exactly test1b's own
construction and reproduces its numbers by build, removing that bias; it
is the method this module actually uses, disclosed here because it
departs from the brief's own literal phrasing.

LAMBDA is fit on ONE region (`--fit-region`, the brief's North America
Nebula: completeness there is low enough mid-catalogue for the
independent and comonotone models to differ) and checked for
insensitivity on a second (`--check-region`, Perseus). If the fit on its
own wants LAMBDA > 1 (the empirical fraction exceeding the fully
comonotone bound -- C18 saw this at 49-77 uJy in North America Nebula,
UKIDSS's clean-stellar cut itself favouring isolated stars over the
crowded population the model integrates over), this module reports the
uncapped maximum-likelihood value as a disclosure and returns LAMBDA
capped at 1.

S03 (round6 review, 2026-10-22): an earlier version of this module scored
the OBSERVED outcome as I1-and-I2-both-detected (`_match_outcomes`'s own
`y`, unchanged by this fix) but predicted the eight-band atlas event
`P(>=2 of 8)`/`p_(2)` (the second-largest of the eight `p_b`) -- a
mismatched event pair. At eight bands with moderate per-band
probabilities, `P(>=2 of 8)` and its comonotone bound both run far above
the two-band I1-and-I2 probabilities, which pushed the maximum-likelihood
fit out of `[0, 1]` on BOTH regions and left LAMBDA capped at 1 with no
interior value ever reachable. `_population_bin_table` now predicts the
SAME I1-and-I2 event the data score: independent `p_I1 * p_I2`, comonotone
`min(p_I1, p_I2)`. This is a correction to the measurement's own event
definition, not a change to the atlas/likelihood mixture model that
LAMBDA still feeds (that model's own event is, and remains, `P(>=2 of
8)`/`p_(2)` over all eight bands) -- a two-band proxy measurement of a
one-parameter mixture weight that the model then applies at eight-band
scale, on the ordinary assumption that one survey-wide LAMBDA governs the
same underlying band-correlation mechanism regardless of which bands are
used to measure it.
"""

import argparse

import h5py
import healpy as hp
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute.bmstp import atlas
from sesnaimpute.catalog import completeness as completeness_module
from sesnaimpute.population import selection as selection_module

BAND_KEYS = definitions.SELECTION.bands
_SEL = list(definitions.SELECTION_INDEX)
IDX_K = BAND_KEYS.index("Ks")
IDX_I1 = BAND_KEYS.index("I1")
IDX_I2 = atlas.IDX_I2

#: Cohen, Wheaton & Megeath 2003, AJ 126, 1090 (`definitions.BANDS`'s own
#: Ks row): the same zero point C18's test1b used as a bare literal
#: (666.7e3, mJy), read here off the one definition instead.
ZP_K_MJY = definitions.BANDS_BY_KEY["Ks"].vega_zero_point_jy * 1000.0

MATCH_RADIUS_ARCSEC = 1.0
CHANCE_OFFSET_ARCSEC = 10.0

#: test1b's own Ks binning (`test1b_empirical_completeness.py`'s `KE`).
KS_EDGES = np.arange(13.0, 17.5001, 0.5)

#: A member batch of this many rows per pixel batch (test1b's own
#: `1.5e6 // uc.size` rule, carried verbatim): keeps one batch's
#: `(n_pix_batch, n_mem_batch, 8)` working set bounded regardless of a
#: tile's own member count.
_MEMBER_ROWS_BUDGET = int(1.5e6)

#: NONDET brief's own floor-style guard: a profile-likelihood 68% interval
#: is read off where -2 ln L rises by 1 from its minimum (one parameter),
#: i.e. where this module's own negative log-likelihood rises by 0.5.
_DELTA_NLL_68 = 0.5


def _uvec(l_deg, b_deg):
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.column_stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def _region_pixels(config, region):
    """The region's admitted pixels (`pix`), the same quality mask C18's
    test1b applies (deep-covered, `COVERAGE >= 0.98`, I1 and I2 both
    on-map everywhere in the pixel -- `good`), `bmstp.atlas`'s own
    per-worker completeness band params (`bp`, the IDENTICAL `PIXEL_
    CURVE_<band>` table `catalogued_probability` reads), and `f_lim` (the
    bright-end 50% limit grid, `_pahc_weight`'s own input)."""
    pix, f_lim = atlas._depth_grid(config, region)
    coverage = atlas._coverage(config, region, pix)
    bp = atlas._completeness_band_params(config, region)
    comp_path = config_module.product_path(
        config, "catalog", "sesna", "completeness", "source", region=region)
    with h5py.File(comp_path, "r") as f:
        cov1 = np.asarray(f["PIXEL_COVERED_FRAC_I1"][:])
        cov2 = np.asarray(f["PIXEL_COVERED_FRAC_I2"][:])
    hist_path = config_module.product_path(
        config, "population", "anchors", "histograms", "hpx512", region=region)
    with h5py.File(hist_path, "r") as f:
        ap = np.asarray(f["HPX_PIX_512"][:])
        dc = np.asarray(f["DEEP_COVERED"][:])
    loc = np.minimum(np.searchsorted(ap, pix), ap.size - 1)
    good = (ap[loc] == pix) & dc[loc] & (coverage >= 0.98) & (cov1 == 1.0) & (cov2 == 1.0)
    return pix, good, bp, f_lim


def _sesna_outcomes(config, region, pix):
    """`(gl, gb, org)`: this region's own SESNA sources (position, `ORIGIN_
    FNU`), restricted to sources inside `pix` -- the same restriction
    C18's test1b applies, so an edge UKIDSS star can still match a SESNA
    source just inside the admitted-pixel boundary."""
    src_path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(src_path, "r") as f:
        gl = np.asarray(f["GAL_L_DEG"][:])
        gb = np.asarray(f["GAL_B_DEG"][:])
        org = np.asarray(f["ORIGIN_FNU"][:])[:, _SEL]
    s_pix = hp.ang2pix(512, gl, gb, nest=True, lonlat=True)
    keep = np.isin(s_pix, pix)
    return gl[keep], gb[keep], org[keep]


def _ukidss_stars(config, region, pix, good):
    """`(ul, ub, uK)`: every UKIDSS clean-stellar star inside a `good`
    admitted pixel of `region`."""
    path = f"{config.data_root}/sky/download/ukidss_gps/sources_ukidss_hpx512__{region}.csv"
    ud = pd.read_csv(path)
    ul = ud["l"].to_numpy(dtype=np.float64)
    ub = ud["b"].to_numpy(dtype=np.float64)
    uK = ud["k_1AperMag3"].to_numpy(dtype=np.float64)
    upx = hp.ang2pix(512, ul, ub, nest=True, lonlat=True)
    order = np.argsort(pix)
    loc = np.searchsorted(pix[order], upx)
    loc = np.minimum(loc, pix.size - 1)
    hit = order[loc]
    found = (pix[hit] == upx) & good[hit]
    return ul[found], ub[found], uK[found]


def _match_outcomes(gl, gb, org, ul, ub):
    """`(y, chance_rate)`: `y` (n_star,) 1.0 where the star matches a
    SESNA source within `MATCH_RADIUS_ARCSEC` with I1 and I2 both
    measured, else 0.0; `chance_rate` the SAME search's own match rate at
    a `CHANCE_OFFSET_ARCSEC` offset (C18's own check), a single number
    folded into the model as a constant false-positive floor rather than
    subtracted bin by bin."""
    r1 = np.radians(MATCH_RADIUS_ARCSEC / 3600.0)
    tree = cKDTree(_uvec(gl, gb))
    dist, j = tree.query(_uvec(ul, ub), distance_upper_bound=r1)
    hit = np.isfinite(dist)
    jj = np.minimum(j, gl.size - 1)
    y = (hit & (org[jj, IDX_I1] == 1) & (org[jj, IDX_I2] == 1)).astype(np.float64)
    dist_r, _ = tree.query(_uvec(ul, ub + CHANCE_OFFSET_ARCSEC / 3600.0), distance_upper_bound=r1)
    chance_rate = float(np.isfinite(dist_r).mean()) if dist_r.size else 0.0
    return y, chance_rate


def _population_bin_table(config, region, pix, good, bp, f_lim):
    """`(p_indep_bin, p2_bin, n_eff_bin)`, one triple per `KS_EDGES` bin:
    C18's test1b model computation (module docstring), reduced to the
    SAME I1-and-I2 event `_match_outcomes` scores (S03 fix) -- the SAME
    per-member `(n_pix_batch, n_mem_batch, 8)` `p` array test1b forms,
    reduced here to `p[..., IDX_I1] * p[..., IDX_I2]` (independent) AND
    `np.minimum(p[..., IDX_I1], p[..., IDX_I2])` (comonotone), both
    pixel-area- and population-weighted and binned by the member's own
    dimmed Ks magnitude before being averaged -- never the other way
    around (module docstring's disclosure)."""
    # C26 (owner ruling 2026-10-21): no longer reads a_col here -- each
    # member's own dimming comes from its own stored A_STAR_K now, not a
    # pixel's own A_COL_K times this member's placement fraction.
    tile_of_pix, _ = atlas._pixel_tile(config, region, pix)
    star_path = config_module.product_path(
        config, "population", "star", "population", "tile", region=region)
    field_path = config_module.product_path(
        config, "population", "trilegal", "field-stars", "region", region=region)
    with h5py.File(field_path, "r") as f:
        fnu_field = np.asarray(f["FNU_MJY"][:], dtype=np.float64)

    nb = KS_EDGES.size - 1
    acc_all = np.zeros(nb, dtype=np.float64)
    acc_cat = np.zeros(nb, dtype=np.float64)
    acc_p2 = np.zeros(nb, dtype=np.float64)

    with h5py.File(star_path, "r") as sf:
        limit8 = sf["LIMIT8_GRID_MJY"][()]
        for t in sorted(set(int(x) for x in tile_of_pix[good])):
            pp_all = np.flatnonzero(tile_of_pix == t)
            pp = pp_all[good[pp_all]]
            if pp.size == 0:
                continue
            g = sf[f"tile_{t}"]
            om = float(g.attrs["OMEGA_POINTING_DEG2"])
            si = g["STAR_INDEX"][()]
            u = g["XI"][()].astype(float)
            a_star_k = g["A_STAR_K"][()].astype(float)
            ws = g["W_STAR"][()].astype(float)
            ppahc = atlas._pahc_weight(
                limit8, g["P_PAHC"][()].astype(float),
                float(np.mean(f_lim[pp_all, atlas.IDX_I4])))
            f0all = fnu_field[si].astype(float)
            _uc, f0c, a_starc, wsc, _, _, _ = atlas._collapse_star_members(
                u, f0all, ws, ws * (1.0 - ppahc), ws * ppahc, a_star_k)
            keep = wsc > 0
            f0c, a_starc, wsc = f0c[keep], a_starc[keep], wsc[keep]
            if a_starc.size == 0:
                continue
            batch = max(1, _MEMBER_ROWS_BUDGET // a_starc.size)
            for s0 in range(0, pp.size, batch):
                p_ = pp[s0:s0 + batch]
                # C26 (owner ruling 2026-10-21): each member's own A_STAR_K
                # (its own local extinction at its own drawn pixel) is
                # ALREADY absolute -- broadcast to every pixel in this
                # batch rather than recombined with a DIFFERENT pixel's
                # own A_COL_K (the retired a_col * XI pattern; beyond the
                # Edenhofer cube's edge that reconstruction would silently
                # drop the far-field reddening XI's own clip discards).
                a = np.broadcast_to(a_starc[None, :], (p_.size, a_starc.size))
                kap = selection_module.kappa_selection(config, w=selection_module.law_dense_weight(a))
                flux = f0c[None] * 10.0 ** (-0.4 * a[:, :, None] * kap)
                p = np.empty_like(flux)
                rows2 = np.ascontiguousarray(np.broadcast_to(p_[:, None], flux.shape[:2])).ravel()
                for bi, band in enumerate(BAND_KEYS):
                    curve, g0, dm, n, aa, cc, scale = bp[band]
                    fb = flux[:, :, bi]
                    hs = fb > 0.0
                    lf = np.log10(np.where(hs, fb, 1.0))
                    v = completeness_module.completeness_many(
                        curve, rows2, g0, dm, n, np.ascontiguousarray(lf).ravel(), aa, cc, scale
                    ).reshape(fb.shape)
                    p[:, :, bi] = np.where(hs, np.minimum(v, completeness_module.C_CAP), 0.0)
                # S03 fix: predict the SAME I1-and-I2 event `_match_outcomes`
                # scores as "catalogued" (y), not the atlas's eight-band
                # P(>=2 of 8)/p_(2) -- those ran far above the two-band
                # I1-and-I2 probabilities and pushed the fit out of [0, 1].
                p_i1 = p[:, :, IDX_I1]
                p_i2 = p[:, :, IDX_I2]
                pc = p_i1 * p_i2
                p2v = np.minimum(p_i1, p_i2)
                wt = np.broadcast_to((wsc / om)[None, :] * atlas._HPX512_PIXEL_DEG2, pc.shape)
                km = -2.5 * np.log10(flux[..., IDX_K] / ZP_K_MJY)
                ib = np.digitize(km, KS_EDGES) - 1
                ok = (ib >= 0) & (ib < nb)
                acc_all += np.bincount(ib[ok], weights=wt[ok], minlength=nb)[:nb]
                acc_cat += np.bincount(ib[ok], weights=(wt * pc)[ok], minlength=nb)[:nb]
                acc_p2 += np.bincount(ib[ok], weights=(wt * p2v)[ok], minlength=nb)[:nb]

    safe = np.where(acc_all > 0.0, acc_all, 1.0)
    return acc_cat / safe, acc_p2 / safe, acc_all


def _p_cat_mix(p_indep, p2, lam):
    """The C20 mixture, evaluated at the OBSERVED I1-and-I2 event (S03
    fix): `(1 - LAMBDA) * p_I1 * p_I2 + LAMBDA * min(p_I1, p_I2)`. `p2`
    (named for the general atlas mixture's `p_(2)`) carries `min(p_I1,
    p_I2)` here: under one shared latent uniform, I1 AND I2 both clear
    iff the draw is below the smaller of the two."""
    return (1.0 - lam) * p_indep + lam * p2


def _neg_log_lik(lam, p_indep, p2, y, chance_rate):
    p_cat = np.clip(_p_cat_mix(p_indep, p2, lam), 0.0, 1.0)
    p_match = np.clip(p_cat + (1.0 - p_cat) * chance_rate, 1e-12, 1.0 - 1e-12)
    return -float(np.sum(y * np.log(p_match) + (1.0 - y) * np.log(1.0 - p_match)))


def _profile_68(p_indep, p2, y, chance_rate, lam_hat, nll_min, lo, hi, n=1201):
    """A `n`-point scan of `_neg_log_lik` over `[lo, hi]`, returning the
    widest interval around `lam_hat` where the NLL stays within
    `_DELTA_NLL_68` of `nll_min` -- a plain grid profile, not a curvature
    approximation (the mixture is not locally quadratic near LAMBDA=0 or
    1, where it is clipped)."""
    grid = np.linspace(lo, hi, n)
    nll = np.array([_neg_log_lik(l, p_indep, p2, y, chance_rate) for l in grid])
    inside = nll <= nll_min + _DELTA_NLL_68
    idx_hat = int(np.argmin(np.abs(grid - lam_hat)))
    if not inside[idx_hat]:
        inside[idx_hat] = True
    lo_idx = idx_hat
    while lo_idx > 0 and inside[lo_idx - 1]:
        lo_idx -= 1
    hi_idx = idx_hat
    while hi_idx < n - 1 and inside[hi_idx + 1]:
        hi_idx += 1
    return float(grid[lo_idx]), float(grid[hi_idx])


def fit_region(config, region, bin_table=None, lam_scan=(-0.5, 2.0), lam_eval=None):
    """Fits LAMBDA by maximum likelihood against `region`'s own matched/
    unmatched UKIDSS outcomes (module docstring). Returns a dict: `n_star`,
    `n_cat`, `chance_rate`, `lam_hat` (capped to `[0, 1]`), `lam_hat_
    uncapped`, `lo68`/`hi68` (on the CAPPED scale, i.e. clipped to `[0,
    1]` too), `nll_min`, and the per-Ks-bin printed-table rows (`table`:
    Ks bin, N, empirical fraction, model fraction at LAMBDA=0 and at
    `lam_eval` -- this region's OWN `lam_hat` by default, or a caller-
    supplied value, e.g. a second region's own fit, to show how THAT
    LAMBDA predicts here)."""
    pix, good, bp, f_lim = _region_pixels(config, region)
    if bin_table is None:
        bin_table = _population_bin_table(config, region, pix, good, bp, f_lim)
    p_indep_bin, p2_bin, n_eff_bin = bin_table

    gl, gb, org = _sesna_outcomes(config, region, pix)
    ul, ub, uK = _ukidss_stars(config, region, pix, good)
    y, chance_rate = _match_outcomes(gl, gb, org, ul, ub)

    ib_star = np.digitize(uK, KS_EDGES) - 1
    has_bin = (ib_star >= 0) & (ib_star < KS_EDGES.size - 1) & (n_eff_bin[np.clip(ib_star, 0, KS_EDGES.size - 2)] > 0)
    ib_star, y, uK = ib_star[has_bin], y[has_bin], uK[has_bin]
    p_indep = p_indep_bin[ib_star]
    p2 = p2_bin[ib_star]

    res = minimize_scalar(_neg_log_lik, bounds=lam_scan, method="bounded",
                           args=(p_indep, p2, y, chance_rate), options={"xatol": 1e-4})
    lam_uncapped = float(res.x)
    nll_min = float(res.fun)
    lam_hat = float(np.clip(lam_uncapped, 0.0, 1.0))
    lo68, hi68 = _profile_68(p_indep, p2, y, chance_rate, lam_uncapped, nll_min,
                              lo=lam_scan[0], hi=lam_scan[1])
    lo68, hi68 = float(np.clip(lo68, 0.0, 1.0)), float(np.clip(hi68, 0.0, 1.0))

    lam_eval = lam_hat if lam_eval is None else float(lam_eval)
    table = []
    for b in range(KS_EDGES.size - 1):
        m = ib_star == b
        n_b = int(m.sum())
        if n_b == 0:
            continue
        emp = float(np.mean(y[m]))
        pc0 = _p_cat_mix(p_indep[m], p2[m], 0.0)
        model0 = float(np.mean(pc0 + (1.0 - pc0) * chance_rate))
        pce = _p_cat_mix(p_indep[m], p2[m], lam_eval)
        model_eval = float(np.mean(pce + (1.0 - pce) * chance_rate))
        table.append((KS_EDGES[b], KS_EDGES[b + 1], n_b, emp, model0, model_eval))

    return dict(region=region, n_star=int(y.size), n_cat=int(y.sum()),
                chance_rate=chance_rate, lam_hat=lam_hat, lam_hat_uncapped=lam_uncapped,
                lo68=lo68, hi68=hi68, nll_min=nll_min, lam_eval=lam_eval, table=table,
                bin_table=bin_table)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root_cfg")
    ap.add_argument("--fit-region", default="North America Nebula")
    ap.add_argument("--check-region", default="Perseus")
    args = ap.parse_args(argv)
    config = config_module.load(args.root_cfg)

    fit = fit_region(config, args.fit_region)
    print("detection_correlation: %s: %d UKIDSS stars (good pixels, in-range Ks), %d catalogued, "
          "chance rate %.4f" % (fit["region"], fit["n_star"], fit["n_cat"], fit["chance_rate"]))
    print("detection_correlation: %s: LAMBDA_hat = %.4f (uncapped %.4f), 68%% interval "
          "[%.4f, %.4f], -lnL_min = %.3f"
          % (fit["region"], fit["lam_hat"], fit["lam_hat_uncapped"], fit["lo68"], fit["hi68"],
             fit["nll_min"]))
    if fit["lam_hat_uncapped"] > 1.0:
        print("detection_correlation: DISCLOSURE: the unconstrained fit on %s wants LAMBDA = "
              "%.4f > 1 -- the empirical catalogued fraction exceeds the fully comonotone "
              "bound there; capped at 1 (UKIDSS's clean-stellar cut favours isolated stars "
              "over the crowded population the model integrates over)."
              % (fit["region"], fit["lam_hat_uncapped"]))
    print(" Ks bin    |     N | empirical | model(LAMBDA=0) | model(LAMBDA=%.3f)" % fit["lam_hat"])
    for lo, hi, n_b, emp, m0, mf in fit["table"]:
        print(" %4.1f-%4.1f | %5d | %9.3f | %15.3f | %18.3f" % (lo, hi, n_b, emp, m0, mf))

    # The check region's own independent fit (is LAMBDA separately
    # constrained there at all?) plus its table EVALUATED AT THE FIT
    # REGION's own lam_hat (does the survey-wide value this module would
    # adopt predict the check region's own empirical fraction as well as
    # LAMBDA=0 does -- the insensitivity the brief asks for).
    check_own = fit_region(config, args.check_region)
    check = fit_region(config, args.check_region, bin_table=check_own["bin_table"], lam_eval=fit["lam_hat"])
    print("detection_correlation: %s (check): %d UKIDSS stars (good pixels, in-range Ks), %d "
          "catalogued, chance rate %.4f" % (check["region"], check["n_star"], check["n_cat"],
                                              check["chance_rate"]))
    print("detection_correlation: %s (check, own fit): LAMBDA_hat = %.4f (uncapped %.4f), 68%% "
          "interval [%.4f, %.4f] -- insensitive if this interval covers most of [0, 1]"
          % (check_own["region"], check_own["lam_hat"], check_own["lam_hat_uncapped"],
             check_own["lo68"], check_own["hi68"]))
    print(" Ks bin    |     N | empirical | model(LAMBDA=0) | model(LAMBDA=%.3f, from %s)"
          % (fit["lam_hat"], fit["region"]))
    for lo, hi, n_b, emp, m0, mf in check["table"]:
        print(" %4.1f-%4.1f | %5d | %9.3f | %15.3f | %18.3f" % (lo, hi, n_b, emp, m0, mf))

    return dict(fit=fit, check=check, check_own=check_own)


if __name__ == "__main__":
    main()
