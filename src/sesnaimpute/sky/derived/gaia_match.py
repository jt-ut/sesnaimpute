"""The SESNA-Gaia crossmatch weighted by match probability (SPEC_PRIORS.md
section 2.1, "joint and marginal bins" row) -- `G_s`, the Gaia congruence
term the fitter reads as `Gamma`'s `G_s` (10_POSTERIOR.md, card T13).

WHAT G_S IS. The ruled, even-odds likelihood ratio of the old
`fetch_external.gaia_source_crossmatch.match_probability` module, carried
forward unchanged in its statistical content:

    G_s = L / (1 + L),   L = f_A(match data) / f_B(match data)

`f_A` is the density of the observed match observation -- the nearest
Gaia candidate's separation, or "no neighbour" -- under a TRUE
counterpart; `f_B` under CHANCE ALIGNMENT. `G_s` is a likelihood ratio
renormalised at EVEN PRIOR ODDS, never a posterior under a counterpart-
fraction prior: no fraction is folded into it anywhere below. Where a
fraction is unavoidable inside the calibration (a mixture must be
normalised over its own population before its true-pair component can be
measured), it is one profiled nuisance per region, fit and discarded --
never applied to `G_s` itself.

Observation space: `{r : 0 < r <= R_MAX_ARCSEC}` union `{no neighbour}`,
`R_MAX_ARCSEC = 3.0` -- the download's own query-net radius (`sky.
download.gaia_crossmatch`), replacing the legacy 10". The nearest
candidate is chosen AFTER propagating its own proper motion from the
Gaia table epoch (2016.0) to the SESNA survey epoch (`region_pull`'s
rule); a candidate outliving no proper-motion solution is matched at
2016.0 directly and flagged `NO_PM`, never given a zero-motion
propagation.

    f_A(r) = f_R(r) S_C(r) + c_C(r) S_R(r),   P_A(no neighbour) = S_R(R_MAX) S_C(R_MAX)
    f_R(r) = two-component Rayleigh mixture (core s1, halo s2, mix eps)
    f_B(r) = c_C(r), the chance nearest-neighbour law;  P_B(no neighbour) = S_C(R_MAX)

WHAT IS DIFFERENT FROM THE OLD MODULE, AND WHY. The old module estimated
the chance density `rho` from the crossmatch's own candidate counts,
smoothed by k-NN and remapped through a fitted `(beta, gamma, delta)`
calibration -- machinery built entirely to correct a noisy PROXY density
for dilution and non-Poisson clustering. This build has no such proxy:
`rho`, the local Gaia surface density, is read directly from the Gaia
counts product (`sky/derived/gaia/counts_gaia_hpx512`) -- the true
all-sky Gaia source count in the source's own nside-512 pixel divided by
that pixel's solid angle, `G < 21` (that product's own faint edge). A
measured density needs no correction for the bias a proxy would carry,
so the chance law here is the plain Poisson nearest-neighbour law
(`beta = 1`; the old module's Weibull generalisation and its two extra
fitted numbers, `gamma`/`delta`, have no target to correct and are
dropped). What survives unchanged: the two-component Rayleigh true-pair
shape, fit per region by maximum likelihood with ONE profiled
counterpart-fraction nuisance (discarded, per the guard above), against
the chance law fixed by the measured density.

    S_C(r) = exp(-pi rho r^2),         c_C(r) = 2 pi rho r exp(-pi rho r^2)
    f_R(r) = (1-eps) (r/s1^2) exp(-r^2/2 s1^2) + eps (r/s2^2) exp(-r^2/2 s2^2)
    S_R(r) = (1-eps) exp(-r^2/2 s1^2)          + eps exp(-r^2/2 s2^2)

Datasets, catalogue row order: G_S, SEP_ARCSEC (NaN for no neighbour),
GAIA_SOURCE_ID (-1 for none), G_MAG, BP_MAG, RP_MAG, PLX_MAS, E_PLX_MAS,
RUWE, NO_PM, RHO_PER_ARCSEC2. Root attrs: GRANULE, S1_ARCSEC, S2_ARCSEC,
EPS_HALO, R_MAX_ARCSEC.
"""

import functools
import os

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.granules import access as granule_access
from sesnaimpute.sky.download.gaia_crossmatch import build as crossmatch_module

_STEM = "match_gaia_source"

# Gaia DR3 astrometric reference epoch (VizieR I/355/gaiadr3; region_pull.py's
# GAIA_EPOCH). Every candidate position in the download's CSV (RAdeg/DEdeg) is
# native to this epoch.
GAIA_EPOCH = 2016.0

# Owner-approved uniform SESNA survey-epoch approximation for the Gaia
# proper-motion propagation (root.cfg [run] survey_epoch_default). Documented
# approximation metadata, not a per-region measured epoch.
SURVEY_EPOCH = 2006.0

# The download's own query-net radius, arcsec (sky.download.gaia_crossmatch.
# build.QUERY_RADIUS_ARCSEC): the observation window's upper edge and the
# location of the no-neighbour atom. Replaces the legacy archive's 10".
R_MAX_ARCSEC = 3.0

# The Gaia counts product's own pixelisation and faint edge (sky.derived.
# gaia_counts.py: galactic nside-512 NESTED, MAG_EDGES 10-21 -> G < 21).
GAIA_COUNTS_NSIDE = 512
GAIA_COUNTS_G_LIMIT = 21.0

# Numerical floor/cap on the likelihood ratio L (match_probability.py's own
# choice, carried unchanged): L_MIN keeps a no-match row from being
# infinitely decisive (an unclamped S_R(R_MAX) underflows below any float);
# L_MAX bounds the evidence a sub-0.05" match can carry (the chance density
# vanishes faster than r as r -> 0, so the unclamped ratio diverges there).
L_MIN, L_MAX = 1e-6, 1e6

# True-pair shape fit: Nelder-Mead restart points for the halo scale s2
# (arcsec), spanning the plausible core/halo split; algorithmic seeds, not
# measured constants.
_S2_RESTARTS = (0.30, 0.60, 1.20)


# ------------------------------------------------------- proper motion & sep

def propagate_to_epoch(ra_deg, dec_deg, pmra_masyr, pmdec_masyr, epoch_from, epoch_to):
    """Linear tangent-plane proper-motion propagation (region_pull.py's
    `propagate_candidates`, unchanged): no curvature, parallax or
    perspective correction, adequate at these baselines (a few years to
    two decades) and this astrometric scale. `pmra_masyr` is Gaia's
    `mu_alpha* = (dRA/dt) cos(dec)`, already cos(dec)-corrected, so the RA
    step divides it back out."""
    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    dt_yr = epoch_to - epoch_from
    dec_new = dec_deg + np.asarray(pmdec_masyr, dtype=np.float64) * dt_yr / 3.6e6
    ra_new = ra_deg + (np.asarray(pmra_masyr, dtype=np.float64) * dt_yr / 3.6e6) / np.cos(np.radians(dec_deg))
    return ra_new, dec_new


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation, arcsec, via the haversine form (stable at
    sub-arcsec scales)."""
    ra1r, dec1r, ra2r, dec2r = (np.radians(np.asarray(x, dtype=np.float64))
                                 for x in (ra1, dec1, ra2, dec2))
    dra, ddec = ra2r - ra1r, dec2r - dec1r
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1r) * np.cos(dec2r) * np.sin(dra / 2.0) ** 2
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))) * 3600.0


# --------------------------------------------------------------- G_s model

def true_pair_pdf(r, s1, s2, eps):
    """f_R: two-component Rayleigh mixture density of the true-pair match
    distance, arcsec^-1."""
    r = np.asarray(r, dtype=np.float64)
    return ((1.0 - eps) * r / s1 ** 2 * np.exp(-0.5 * (r / s1) ** 2)
            + eps * r / s2 ** 2 * np.exp(-0.5 * (r / s2) ** 2))


def true_pair_sf(r, s1, s2, eps):
    """S_R: survival of the true-pair match distance."""
    r = np.asarray(r, dtype=np.float64)
    return ((1.0 - eps) * np.exp(-0.5 * (r / s1) ** 2)
            + eps * np.exp(-0.5 * (r / s2) ** 2))


def chance_sf(r, rho):
    """S_C: probability that no chance Gaia source lies within `r`, the
    Poisson nearest-neighbour law at the measured local density `rho`
    (arcsec^-2)."""
    r = np.asarray(r, dtype=np.float64)
    return np.exp(-np.pi * np.asarray(rho, dtype=np.float64) * r ** 2)


def chance_pdf(r, rho):
    """c_C: density of the nearest chance Gaia source's separation,
    arcsec^-1."""
    r = np.asarray(r, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    return 2.0 * np.pi * rho * r * np.exp(-np.pi * rho * r ** 2)


def lr_matched(sep, rho, s1, s2, eps):
    """L(r) = f_R(r)/c_C(r) + S_R(r) for a row with a neighbour inside
    R_MAX_ARCSEC. The chance factor does not cancel here (the competing-risks
    numerator still carries S_C)."""
    return true_pair_pdf(sep, s1, s2, eps) / chance_pdf(sep, rho) + true_pair_sf(sep, s1, s2, eps)


def lr_no_match(s1, s2, eps, r_max=R_MAX_ARCSEC):
    """L for the no-neighbour atom: S_R(R_MAX). The chance survival
    S_C(R_MAX) appears in both P_A and P_B and cancels exactly, so this does
    not depend on the local density."""
    return float(true_pair_sf(r_max, s1, s2, eps))


def g_from_lr(lr):
    """G = L/(1+L), clamped to [L_MIN, L_MAX] before the ratio so G is
    finite and strictly in (0, 1)."""
    lr = np.asarray(lr, dtype=np.float64)
    lr = np.where(np.isfinite(lr), lr, L_MAX)
    lr = np.clip(lr, L_MIN, L_MAX)
    return lr / (1.0 + lr), lr


# ---------------------------------------------------- true-pair shape fit

def _profile_counterpart_fraction(p_true, p_chance, atom_true, atom_chance):
    """The ONE profiled nuisance per region: the share `q` of sources whose
    separation was drawn from the true-pair branch rather than pure chance.
    Fit by 1-D maximum likelihood and returned only as a diagnostic -- never
    multiplied into `G_s` (the guard `evaluate`/`lr_matched`/`lr_no_match`
    above never takes `q` as an argument)."""
    def nll(q):
        e = np.maximum(q * p_true + (1.0 - q) * p_chance, 1e-300)
        ea = np.maximum(q * atom_true + (1.0 - q) * atom_chance, 1e-300)
        return -(float(np.log(e).sum()) + float(np.log(ea).sum()))
    res = minimize_scalar(nll, bounds=(1e-6, 1.0 - 1e-6), method="bounded",
                          options=dict(xatol=1e-8))
    return float(res.x), float(res.fun)


def _restart_fit(sep_matched, p_chance, chance_sf_matched, atom_chance, r_max, s2_0):
    """One Nelder-Mead restart of the (s1, s2, eps) search from its own
    starting point (`fit_true_pair_shape`'s `_S2_RESTARTS`), over the
    source arrays bound into this call once. The chance-law factors
    `p_chance = CHANCE_PDF(sep_matched, rho_matched)`, `chance_sf_matched =
    CHANCE_SF(sep_matched, rho_matched)` and `atom_chance = CHANCE_SF(r_max,
    rho_no_match)` do not depend on (s1, s2, eps), so every trial's
    likelihood (`nll`) recomputes only the two true-pair factors that do --
    TRUE_PAIR_PDF and TRUE_PAIR_SF at `sep_matched` -- the density of the
    observed separation under d=1 (a true counterpart is present,
    competing risks: the counterpart's own match distance or a closer
    chance star, whichever comes first) and the no-neighbour atom's
    true-pair survival."""
    def nll(params):
        s1, s2, eps = params
        if not (0.02 < s1 < 1.0 and s1 < s2 < 8.0 and 0.0 <= eps < 0.6):
            return 1e12
        p_true = (true_pair_pdf(sep_matched, s1, s2, eps) * chance_sf_matched
                  + p_chance * true_pair_sf(sep_matched, s1, s2, eps))
        atom_true = true_pair_sf(r_max, s1, s2, eps) * atom_chance
        _, value = _profile_counterpart_fraction(p_true, p_chance, atom_true, atom_chance)
        return value
    return minimize(nll, [0.12, s2_0, 0.06], method="Nelder-Mead",
                    options=dict(maxiter=4000, maxfev=4000, xatol=1e-6, fatol=1e-4))


def fit_true_pair_shape(sep_matched, rho_matched, rho_no_match, r_max=R_MAX_ARCSEC, n_jobs=1):
    """Maximum-likelihood (s1, s2, eps) of the true-pair Rayleigh mixture,
    the chance law fixed at each source's own measured density, with the
    counterpart fraction profiled out at every trial point. The chance-law
    factors that are the same at every trial (`p_chance`, `chance_sf_matched`,
    `atom_chance`) are computed once here, outside every trial, instead of
    inside one (`_restart_fit`'s own docstring names them). The three
    Nelder-Mead restarts (`_S2_RESTARTS`) are independent full optimisations
    over the same source arrays; they run on the shared stage pool
    (`pieces.run`, `n_jobs` workers, each restart one piece -- a restart's
    own trial count dwarfs the pool's per-piece dispatch cost, unlike a
    source-range piece of one trial's likelihood, measured net slower than
    the vectorised whole on this region's source counts) instead of one
    after another, reduced to the best by the parent in `_S2_RESTARTS`
    order -- the same order every time, so the restart that wins is
    unaffected by whether the others ran before, after, or alongside it."""
    p_chance = chance_pdf(sep_matched, rho_matched)
    chance_sf_matched = chance_sf(sep_matched, rho_matched)
    atom_chance = chance_sf(r_max, rho_no_match)

    fn = functools.partial(_restart_fit, sep_matched, p_chance, chance_sf_matched, atom_chance, r_max)
    results = pieces_module.run(fn, _S2_RESTARTS, min(int(n_jobs), len(_S2_RESTARTS)))
    best = min(results, key=lambda res: res.fun)
    s1, s2, eps = best.x
    if s2 < s1:  # keep the labelling stable (core = smaller scale)
        s1, s2, eps = s2, s1, 1.0 - eps
    return float(s1), float(s2), float(eps)


# --------------------------------------------------------------- local density

def local_gaia_density(config, region, hpx_pix_512):
    """RHO_PER_ARCSEC2: total Gaia DR3 sources (G < GAIA_COUNTS_G_LIMIT) in
    each source's own nside-512 pixel, over that pixel's solid angle -- the
    all-sky Gaia counts product, never the candidate crossmatch (SPEC: "the
    local density rho is NOT estimated from the candidates")."""
    import healpy as hp
    path = config_module.product_path(config, "sky/derived", "gaia", "counts", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"sky.derived.gaia_match.build: Gaia counts product missing for region {region!r} at "
            f"{path!r} -- run the 'sesnaimpute.sky.derived.gaia_counts' RUNBOOK line first")
    with h5py.File(path, "r") as f:
        pixels = np.asarray(f["HPX_PIX_512"][()], dtype=np.int64)
        counts_per_bin = np.asarray(f["N"][()], dtype=np.int64)
    n_total = counts_per_bin.sum(axis=1)
    loc = np.searchsorted(pixels, hpx_pix_512)
    capped = np.minimum(loc, pixels.size - 1) if pixels.size else loc
    valid = pixels.size and (pixels[capped] == hpx_pix_512)
    if not np.all(valid):
        raise ValueError(
            f"sky.derived.gaia_match.build: {region}: "
            f"{int(np.count_nonzero(~np.asarray(valid)))} source(s) carry an hpx512 pixel absent "
            "from the Gaia counts product -- the granule map and the counts product disagree")
    pixel_area_arcsec2 = hp.nside2pixarea(GAIA_COUNTS_NSIDE, degrees=True) * 3600.0 ** 2
    return n_total[capped].astype(np.float64) / pixel_area_arcsec2


# ------------------------------------------------------------------- inputs

def _read_curated_catalogue(config, region):
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"sky.derived.gaia_match.build: curated catalogue missing for region {region!r} at "
            f"{path!r} -- run the curated-catalogue RUNBOOK line for it")
    with h5py.File(path, "r") as f:
        name = np.array([s.decode() if isinstance(s, bytes) else s for s in f["NAME"][()]])
        ra_deg = np.asarray(f["RA_DEG"][()], dtype=np.float64)
        dec_deg = np.asarray(f["DEC_DEG"][()], dtype=np.float64)
    return name, ra_deg, dec_deg


def _read_hpx_pix_512(config, region, n_sources):
    granule_map_path = config_module.product_path(config, "granules", "sesna", "granule-map", "source")
    columns = granule_access.per_source(config, region, granule_map_path, ["HPX_PIX_512"])
    pix = np.asarray(columns["HPX_PIX_512"], dtype=np.int64)
    if pix.size != n_sources:
        raise ValueError(
            f"sky.derived.gaia_match.build: {region}: granule map has {pix.size} source rows, "
            f"curated catalogue has {n_sources} -- row counts cannot be reconciled")
    return pix


def _candidates_path(config, region):
    return f"{config.data_root}/sky/download/gaia_crossmatch/candidates_gaia_source__{region}.csv"


def _read_candidates(config, region, name_to_row):
    path = _candidates_path(config, region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"sky.derived.gaia_match.build: Gaia crossmatch candidates missing for region {region!r} at "
            f"{path!r} -- run the 'sesnaimpute.sky.download.gaia_crossmatch' RUNBOOK line first")
    df = pd.read_csv(path)
    # The candidates file's own definition is the source list it was
    # uploaded from (sky.download.gaia_crossmatch, 2026-10-22). A catalogue
    # source that was never uploaded has no candidates by construction and
    # would be scored as a Gaia non-detection: that is a stale crossmatch,
    # re-pulled by the download line, never papered over here. A candidate
    # whose source curation has since removed matches nothing and is dropped.
    # Owner ruling 2026-10-22: the crossmatch on disk is used as it is, never
    # refused. Where an upload record exists, the number of current sources it
    # never uploaded is PRINTED (those sources carry no Gaia candidate and are
    # scored as unmatched); without a record the fact is stated.
    upload_path = crossmatch_module.uploaded_names_path(config, region)
    if os.path.exists(upload_path):
        uploaded = crossmatch_module.read_uploaded_names(upload_path)
        n_missing = int(sum(1 for n in name_to_row.index if n not in uploaded))
        if n_missing:
            print(f"gaia_match derive: {region}: {n_missing} curated source(s) were not in the "
                  f"crossmatch upload; they carry no Gaia candidate")
    else:
        print(f"gaia_match derive: {region}: candidates file carries no upload record "
              f"(pulled before 2026-10-22); its coverage of the catalogue is not recorded")
    if df.empty:
        return None
    req_idx = df["NAME"].map(name_to_row)
    n_absent = int(req_idx.isna().sum())
    if n_absent:
        print(f"gaia_match derive: {region}: {n_absent} candidate row(s) belong to sources the "
              f"curated catalogue no longer holds; dropped")
        df = df[~req_idx.isna()]
        req_idx = req_idx[~req_idx.isna()]
    df = df.assign(REQ_IDX=req_idx.astype(np.int64))
    return df


# --------------------------------------------------------- nearest per source

def _propagated_separation(ra_src, dec_src, ra_2016, dec_2016, pmra, pmdec):
    """Every candidate's separation from its SESNA source AFTER propagating
    the candidate's own proper motion to SURVEY_EPOCH; candidates without a
    usable proper-motion solution stay at their 2016.0 position (NO_PM).
    Takes plain arrays (a region's candidates, or one source block's slice
    of them) rather than a dataframe, so a piece can call it on its own
    slice."""
    has_pm = np.isfinite(pmra) & np.isfinite(pmdec)
    ra_prop, dec_prop = ra_2016.copy(), dec_2016.copy()
    if has_pm.any():
        ra_prop[has_pm], dec_prop[has_pm] = propagate_to_epoch(
            ra_2016[has_pm], dec_2016[has_pm], pmra[has_pm], pmdec[has_pm], GAIA_EPOCH, SURVEY_EPOCH)
    sep = angular_sep_arcsec(ra_src, dec_src, ra_prop, dec_prop)
    return sep, ~has_pm


def _nearest_winner(req_idx, sep):
    """Per unique source in `req_idx`, the row index of its nearest
    (smallest `sep`) candidate. Stable mergesort on `sep`, then the first
    occurrence per source in that order -- the vectorised nearest-neighbour
    reduction, no Python loop over sources or candidates."""
    order = np.argsort(sep, kind="mergesort")
    _, first = np.unique(req_idx[order], return_index=True)
    return order[first]


def _empty_match(n_rows):
    """The no-neighbour-atom default match columns, `n_rows` long."""
    return dict(sep_arcsec=np.full(n_rows, np.nan),
               gaia_source_id=np.full(n_rows, -1, dtype=np.int64),
               g_mag=np.full(n_rows, np.nan), bp_mag=np.full(n_rows, np.nan),
               rp_mag=np.full(n_rows, np.nan), plx_mas=np.full(n_rows, np.nan),
               e_plx_mas=np.full(n_rows, np.nan), ruwe=np.full(n_rows, np.nan),
               no_pm=np.zeros(n_rows, dtype=bool), has_neighbor=np.zeros(n_rows, dtype=bool))


# candidate CSV columns the match needs, and the dtype each is read as; the
# region's whole candidate table is pulled into these plain arrays once
# (`_sorted_candidate_columns`), so a piece slices arrays, not a dataframe.
_CANDIDATE_COLUMNS = dict(REQ_IDX=np.int64, RA_DEG=np.float64, DEC_DEG=np.float64,
                          RAdeg=np.float64, DEdeg=np.float64, pmRA=np.float64, pmDE=np.float64,
                          Source=np.int64, Gmag=np.float64, BPmag=np.float64, RPmag=np.float64,
                          Plx=np.float64, e_Plx=np.float64, RUWE=np.float64)


def _sorted_candidate_columns(df):
    """The region's crossmatch candidates as plain arrays, sorted once by
    REQ_IDX -- the Gaia tree a source-block piece searches into with
    `np.searchsorted` for its own slice, built once in the parent (this
    function), never per piece. A stable sort, so two candidates of the same
    source keep their original row order; `_nearest_winner`'s own stable
    sort on separation then breaks a tie exactly as it would on the whole,
    unsorted table, so a piece's result is bit-identical to the undivided
    computation."""
    order = np.argsort(df["REQ_IDX"].to_numpy(dtype=np.int64), kind="mergesort")
    return {col: df[col].to_numpy(dtype=dtype)[order] for col, dtype in _CANDIDATE_COLUMNS.items()}


def _match_piece(cols, piece):
    """One SESNA source block's match columns (SEP_ARCSEC, GAIA_SOURCE_ID,
    photometry, astrometry, NO_PM) and its has-neighbour mask, in that
    block's catalogue row order. `cols` is the region's REQ_IDX-sorted
    candidate table (`_sorted_candidate_columns`); this piece touches only
    the contiguous slice its own source range owns. A source with no
    winning candidate inside R_MAX_ARCSEC -- including one with no candidate
    row at all -- carries the defaults (no-neighbour atom)."""
    start, stop = piece
    req_idx_sorted = cols["REQ_IDX"]
    c0 = int(np.searchsorted(req_idx_sorted, start))
    c1 = int(np.searchsorted(req_idx_sorted, stop))
    out = _empty_match(stop - start)
    if c1 <= c0:
        return out

    local_req = req_idx_sorted[c0:c1] - start
    sep, cand_no_pm = _propagated_separation(
        cols["RA_DEG"][c0:c1], cols["DEC_DEG"][c0:c1],
        cols["RAdeg"][c0:c1], cols["DEdeg"][c0:c1], cols["pmRA"][c0:c1], cols["pmDE"][c0:c1])
    winner_pos = _nearest_winner(local_req, sep)
    winner_req = local_req[winner_pos]
    winner_sep = sep[winner_pos]
    within = winner_sep <= R_MAX_ARCSEC
    sel_req, sel_pos = winner_req[within], winner_pos[within]

    out["sep_arcsec"][sel_req] = winner_sep[within]
    out["gaia_source_id"][sel_req] = cols["Source"][c0:c1][sel_pos]
    out["g_mag"][sel_req] = cols["Gmag"][c0:c1][sel_pos]
    out["bp_mag"][sel_req] = cols["BPmag"][c0:c1][sel_pos]
    out["rp_mag"][sel_req] = cols["RPmag"][c0:c1][sel_pos]
    out["plx_mas"][sel_req] = cols["Plx"][c0:c1][sel_pos]
    out["e_plx_mas"][sel_req] = cols["e_Plx"][c0:c1][sel_pos]
    out["ruwe"][sel_req] = cols["RUWE"][c0:c1][sel_pos]
    out["no_pm"][sel_req] = cand_no_pm[sel_pos]
    out["has_neighbor"][sel_req] = True
    return out


def _match_candidates(n_sources, df, n_jobs=1):
    """Per-source winner arrays (SEP_ARCSEC, GAIA_SOURCE_ID, photometry,
    astrometry, NO_PM) and the has-neighbour mask, catalogue row order,
    computed one SESNA source block at a time (`pieces.slices`) through the
    shared pool so a big region's crossmatch spreads over every worker
    instead of running on the one core the region used to occupy. The
    assembled columns are exactly `_match_piece`'s pieces concatenated in
    source-index order."""
    if df is None:
        return _empty_match(n_sources)
    cols = _sorted_candidate_columns(df)
    fn = functools.partial(_match_piece, cols)
    results = pieces_module.run(fn, pieces_module.slices(n_sources, n_jobs), n_jobs)
    return {key: np.concatenate([r[key] for r in results]) for key in results[0]}


# ------------------------------------------------------------------- write

def _write_region(config, region, matched, rho, g_s, s1, s2, eps):
    out_path = config_module.product_path(config, "sky/derived", "gaia", "match", "source", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.attrs["S1_ARCSEC"] = s1
        f.attrs["S2_ARCSEC"] = s2
        f.attrs["EPS_HALO"] = eps
        f.attrs["R_MAX_ARCSEC"] = R_MAX_ARCSEC
        for name, data in (
            ("G_S", g_s.astype(np.float32)),
            ("SEP_ARCSEC", matched["sep_arcsec"].astype(np.float32)),
            ("GAIA_SOURCE_ID", matched["gaia_source_id"].astype(np.int64)),
            ("G_MAG", matched["g_mag"].astype(np.float32)),
            ("BP_MAG", matched["bp_mag"].astype(np.float32)),
            ("RP_MAG", matched["rp_mag"].astype(np.float32)),
            ("PLX_MAS", matched["plx_mas"].astype(np.float32)),
            ("E_PLX_MAS", matched["e_plx_mas"].astype(np.float32)),
            ("RUWE", matched["ruwe"].astype(np.float32)),
            ("NO_PM", matched["no_pm"].astype(bool)),
            ("RHO_PER_ARCSEC2", rho.astype(np.float32)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])


# --------------------------------------------------------------- one region

def _build_one_region(config, region, n_jobs=1):
    name, _, _ = _read_curated_catalogue(config, region)
    n_sources = name.size
    hpx_pix_512 = _read_hpx_pix_512(config, region, n_sources)
    rho = local_gaia_density(config, region, hpx_pix_512)

    name_to_row = pd.Series(np.arange(n_sources, dtype=np.int64), index=name)
    df = _read_candidates(config, region, name_to_row)
    matched = _match_candidates(n_sources, df, n_jobs)
    has_neighbor = matched["has_neighbor"]

    s1, s2, eps = fit_true_pair_shape(
        matched["sep_arcsec"][has_neighbor], rho[has_neighbor], rho[~has_neighbor], R_MAX_ARCSEC,
        n_jobs=n_jobs)

    lr = np.empty(n_sources)
    lr[has_neighbor] = lr_matched(matched["sep_arcsec"][has_neighbor], rho[has_neighbor], s1, s2, eps)
    lr[~has_neighbor] = lr_no_match(s1, s2, eps, R_MAX_ARCSEC)
    g_s, _lr_clamped = g_from_lr(lr)

    _write_region(config, region, matched, rho, g_s, s1, s2, eps)

    frac_neighbor = float(has_neighbor.mean()) if n_sources else float("nan")
    median_g_neighbor = float(np.median(g_s[has_neighbor])) if has_neighbor.any() else float("nan")
    frac_g_over_half = float((g_s > 0.5).mean()) if n_sources else float("nan")
    print(f"gaia_match derive: {region}: {n_sources} sources, {frac_neighbor:.3f} with a neighbour "
          f"<= {R_MAX_ARCSEC}\", median G_S(neighbour)={median_g_neighbor:.3f}, "
          f"frac(G_S>0.5)={frac_g_over_half:.3f}, S1={s1:.4f}\" S2={s2:.4f}\" EPS={eps:.4f}")
    return region


def build(config, regions=None):
    """Writes `sky/derived/gaia/match_gaia_source__<Region>.hdf5` for each
    requested region (default: all thirty), one region after another
    (owner ruling 2026-10-09): a region's own crossmatch is spread over many
    SESNA source blocks through the shared pool (`_match_candidates`), so
    the region with the most sources -- Cygnus X -- uses every worker
    instead of leaving the rest idle while it alone runs on one core. The
    region-wide S1/S2/EPS fit that follows is unchanged, one call per
    region on the assembled columns."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    with progress_module.Stage("sky.derived.gaia_match") as st:
        n_regions = len(regions)
        for i, region in enumerate(regions):
            _build_one_region(config, region, config.n_jobs)
            st.tick(i + 1, n_regions, "regions")
        st.done(None, regions=n_regions)


if __name__ == "__main__":
    run(build)
