"""The SWIRE galaxy sample, survey-wide (SPEC_BMSTP_DRAFT.md sec 3.2, sec 5.4;
SPEC_PRIORS.md sec 5.1's star-galaxy split).

A view of SWIRE alone: the six blank-field catalogues (Lonsdale et al. 2003,
PASP 115, 897; Surace et al. 2005 DR2 release), star-galaxy separated, kept
as one row per surviving galaxy with its 4.5um flux, its three IRAC colours
and their errors, and its bin on the flux grid the counts law
(`bms/gal/counts_gal_survey.hdf5`) is tabulated on. No density and no
colour grid are formed here: SPEC_BMSTP sec 5.4's GAL template weights form
a kernel density from these rows, per flux node, at read time.

The star-galaxy split reproduces the one `prior.gal.select_star_galaxy_split`
adopted for `bms/gal/counts_gal_survey.hdf5`: the candidate whose own
removed-star count, summed across the six SWIRE fields, comes closest to
Fazio et al. 2004's own star columns transported to each field's own
latitude (studies/swire_vs_fazio.md sec 4, not the earlier residual-galaxy
criterion, which is unreachable once the IRAC point-source retention
p(S) < 1) -- IRAC 3.6um stellarity (SExtractor CLASS_STAR, Bertin &
Arnouts 1996, A&AS 117, 393) at or above `STELLARITY_STAR_MIN`, not the
per-band extended-flag rule (`prior.gal.classify_galaxy_extended_flag`,
which scored worse against the expected star count). Verified directly
against `bms/gal/counts_gal_survey.hdf5`'s own selection before adoption
here.

Each IRAC colour is stored as `prior.gal`'s own internal convention,
log10(flux_a) - log10(flux_b), dex -- not a magnitude scaled by -2.5 --
because this is the exact quantity `prior.gal.build_colour_cdf_tables`
tabulates `bms/gal/counts_gal_survey.hdf5`'s CDF axes on; only this
convention lets this module's acceptance identity (the two CDFs agreeing
exactly for I1, up to the axis sign for I3/I4 -- see below) hold. The sign
convention is uniform, `COLOUR_AB = log10(F_A) - log10(F_B)`, for all three
pairs; `prior.gal`'s own I3/I4 axes are built on the opposite sign
(`log10(F_I3) - log10(F_I2)`, `log10(F_I4) - log10(F_I2)`), so this
module's acceptance check evaluates those two up to that sign flip, not a
literal equality of the raw arrays. `UNIT`/`DEFINITION` attrs on every
COLOUR_*/SIGMA_COLOUR_* dataset say so for a reader of the file alone.

Each galaxy also carries its 70 um flux, uncertainty and detection flag
from the SWIRE MIPS 70 um catalogue (`_match_70um`): a galaxy with no 70 um
source is a non-detection below its field's `LIMIT_P70_MJY`.

Each galaxy also carries its 100 and 160 um flux from the Herschel PACS point
source catalogue (HPPSC, `sky/download/hppsc/ppsc_100.csv`, `ppsc_160.csv`),
where PACS looked (`_match_pacs`): the footprint is the union, over the
catalogue's observations, of the convex hull of each observation's sources
(an observation with fewer than `PACS_HULL_MIN_SOURCES` sources in the field
is not used). A galaxy outside the footprint has no 100/160 information
(`COVERED_P100`/`COVERED_P160` 0, flux NaN: it is neither a detection nor a
non-detection and is left out of every 100/160 statement); inside it, a galaxy
with an HPPSC source within `PACS_MATCH_ARCSEC` is a detection and one without
is a non-detection below that observation's faintest catalogued source. The
chance-match rate is carried by the same galaxies' positions shifted by
`PACS_CHANCE_OFFSET_ARCSEC` north and south and matched the same way
(`FLUX_P100_OFFSET_N_MJY` ...).

A band is missing when its flux or its uncertainty is absent or a sentinel
(flux <= 0 or uncertainty <= 0, the SWIRE -99 sentinel included; no
bandwidth is ever formed from a sentinel): both the colour and the sigma
built from it are `NaN`, not +-inf. `N_FINITE_I1I2`/`I2I3`/`I2I4`/`ALL`
(file attrs) and `N_NODE_I1I2`/`N_NODE_ALL` ((61,) datasets, per S node)
report how many rows actually carry a usable value -- what `population`'s
GAL template-weight kernel (SPEC_BMSTP_DRAFT sec 5.4) can build from.
"""

import os

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.population.gal import N_S_GRID, build_log10_s_grid

_STEM = "galaxies_swire_survey"

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: The six SWIRE fields' catalogue files, sky/download/swire/ (this
#: project's own `sesnaimpute.sky.download.swire.build`), in the same order
#: as `prior.gal.SWIRE_FIELD_FILES`.
SWIRE_FIELD_FILES = (
    "swire_elaisn1.csv", "swire_elaisn2.csv", "swire_elaiss1.csv",
    "swire_lockman.csv", "swire_xmmlss.csv", "swire_cdfs.csv",
)

#: SWIRE's own aperture-2 flux and uncertainty columns, 3.6/4.5/5.8/8.0um
#: (`sky.download.swire.build.COLUMNS`), uJy as distributed.
SWIRE_FLUX_COLUMNS = ("flux_ap2_36", "flux_ap2_45", "flux_ap2_58", "flux_ap2_80")
SWIRE_FLUX_ERR_COLUMNS = ("uncf_ap2_36", "uncf_ap2_45", "uncf_ap2_58", "uncf_ap2_80")

#: SWIRE's MIPS 70 um catalogue files (`sky.download.swire.build.FIELDS_70`),
#: one per field in `SWIRE_FIELD_FILES`' own order, with the columns this view
#: reads: the position and its 1-sigma uncertainty (arcsec), the
#: point-response-function flux and uncertainty (mJy) and the release's own
#: signal-to-noise.
SWIRE70_FIELD_FILES = (
    "swire70_elaisn1.csv", "swire70_elaisn2.csv", "swire70_elaiss1.csv",
    "swire70_lockman.csv", "swire70_xmmlss.csv", "swire70_cdfs.csv",
)

#: The signal-to-noise every source in the six SWIRE 70 um catalogue files
#: clears (the release's own detection rule: the smallest `snr` in all six
#: files is 7.0). A source's noise is its flux over this `snr` (the
#: catalogue's `uncf_prf` is the fit error alone, several times smaller than
#: the flux over `snr`). A galaxy with no 70 um source is a non-detection:
#: its flux lies below the field's own cut, the median flux of the catalogued
#: sources within `SWIRE70_SNR_WINDOW` of `SWIRE70_SNR_MIN`.
SWIRE70_SNR_MIN = 7.0
SWIRE70_SNR_WINDOW = 1.0

#: A 70 um source belongs to the nearest SWIRE row within this many of its own
#: 1-sigma position uncertainties (3 sigma, the usual association radius of a
#: Gaussian position error).
SWIRE70_MATCH_SIGMAS = 3.0

#: Herschel PACS point source catalogue files (`sky/download/hppsc`, a sibling
#: of the SWIRE directory), one per band: (dataset suffix, file).
HPPSC_DIR = "hppsc"
PACS_BANDS = (("P100", "ppsc_100.csv"), ("P160", "ppsc_160.csv"))

#: A PACS source is a galaxy's counterpart if it lies within this many arcsec
#: of it (the 100 um census of studies/galaxy_100um_census_2026-11-06, 5").
PACS_MATCH_ARCSEC = 5.0

#: Two catalogued PACS sources closer than this are one source (the
#: higher-signal one is kept): overlapping observations list a source twice.
PACS_DEDUPE_ARCSEC = 4.0

#: The chance-match test: the galaxies' positions moved this far north and
#: south (arcsec) and matched to the same sources the same way.
PACS_CHANCE_OFFSET_ARCSEC = 60.0

#: An observation's footprint is the convex hull of its catalogued sources in
#: the field; fewer sources than this and no hull is formed. Sources are
#: taken within this many degrees of the field's bounding box.
PACS_HULL_MIN_SOURCES = 10

#: An observation's censoring limit is the flux below which it does not detect
#: sources: the `PACS_LIMIT_PERCENTILE`-th percentile of its own detected
#: fluxes in the band when it has at least `PACS_LIMIT_MIN_DETECTIONS`
#: detections, else that percentile of the pooled detections of the
#: observations whose faintest detection is within a factor `PACS_LIMIT_POOL_FACTOR`
#: of its own (the factor doubled until the pool holds that many). The
#: faintest catalogued source, the earlier limit, lies far below where the
#: catalogue begins to detect (an observation's faintest source is a few
#: signal-to-noise; studies/galaxy_100um_census 2026-11-06 and U1 report).
PACS_LIMIT_PERCENTILE = 10.0
PACS_LIMIT_MIN_DETECTIONS = 100
PACS_LIMIT_POOL_FACTOR = 2.0
PACS_BBOX_PAD_DEG = 0.3

#: The adopted star-galaxy split (module docstring): IRAC 3.6um stellarity
#: (`prior.gal`'s `stell_36` column) at or above this threshold is a star.
#: An unmeasured stellarity is kept as a galaxy, `prior.gal.
#: classify_galaxy_stellarity`'s own conservative default. Raised from
#: 0.95: at S < 0.16mJy the 0.95 rule removed 1.1-1.9x the expected star
#: count (studies/swire_vs_fazio.md sec 4/6), and 0.98 restores it while
#: changing nothing at 0.1-0.5mJy, where 0.95 was already right to 1%.
STELLARITY_STAR_MIN = 0.98

#: The 61-node flux grid every GAL product is tabulated on (SPEC_BMSTP_DRAFT
#: sec 3.2; `population.gal.build_log10_s_grid`, the owner of this grid --
#: SWIRE's own I2 5-sigma depth to Fazio et al. 2004 Table 1's brightest
#: tabulated 4.5um row): read from there rather than kept as a second copy
#: here, so this module's node grid matches `population/gal`'s
#: `counts_gal_survey` product's LOG10_S_GRID by construction, not by two
#: literals happening to agree (review ledger C16).
FAZIO_PATH_SUFFIX = "sky/download/fazio2004/fazio2004_table1_irac_counts.csv"

#: The floor every bright node's own galaxy count is pooled up to before
#: its colour KDE is trusted (review ledger C16): node 60 (the grid's
#: bright edge, 18 mJy) holds 13 SWIRE galaxies on its own, and every
#: `build_galz` brightness cell above that edge clamps to that one node's
#: KDE (`bmstp.template_weights.build_galz`'s own `s_query` clip) --
#: `LIBRARY_DENSITY_MIN_COUNT`'s own floor (20) is a template-density
#: floor, not a galaxy-count one, so this is its own number, chosen to
#: match the planner's ruling (`briefs/SWEEP_2026-10-09.md` sec 5, C16:
#: "pool the bright nodes so none holds fewer than 100 galaxies").
MIN_NODE_GALAXIES = 100


def _pool_bright_nodes(node, n_node, min_count):
    """Remaps `node` (per-galaxy indices into the 61-point LOG10_S_GRID) so
    that every node a galaxy is ever assigned to holds at least `min_count`
    galaxies, by merging thin bright nodes DOWNWARD into their nearest
    better-populated fainter neighbour (review ledger C16): the counts law
    falls steeply toward bright flux, so thinness is a bright-end problem,
    never a faint-end one over this grid. Walking from the brightest node
    (60) to the faintest (0), galaxies accumulate into the open group's
    representative -- the group's own brightest member -- until the running
    count clears `min_count`; the group then closes and a fresh one opens at
    the next (fainter) node. A merged-away node (zero galaxies of its own
    after the remap) is read by `build_galz` through the SAME nearest-node
    borrowing `bmstp.atlas._gal_members` already applies to an empty node,
    not left uniform. Returns `(node_pooled, counts_pooled)`, the second the
    61 canonical positions' own post-merge counts (zero at a merged-away
    position) for the build's report."""
    counts = np.bincount(node[node >= 0], minlength=n_node)
    remap = np.arange(n_node)
    acc, rep = 0, n_node - 1
    for k in range(n_node - 1, -1, -1):
        remap[k] = rep
        acc += int(counts[k])
        if acc >= min_count:
            acc, rep = 0, k - 1
    node_pooled = np.where(node >= 0, remap[np.clip(node, 0, n_node - 1)], node).astype(node.dtype)
    counts_pooled = np.bincount(node_pooled[node_pooled >= 0], minlength=n_node)
    return node_pooled, counts_pooled


def _match_70um(path70, ra, dec, keep):
    """The 70 um flux of each SWIRE row of one field, by position (the 70 um
    table has its own source key and none of the IRAC table's, so the two
    are joined on the sky): each 70 um source goes to the nearest SWIRE row
    within `SWIRE70_MATCH_SIGMAS` of its own position uncertainty, and a row
    claimed twice keeps the higher-signal source. Returns `(flux, sigma,
    limit, n_catalogued, n_matched_galaxy)` per row, flux and sigma NaN where
    the row has no 70 um source (sigma the flux over the catalogue's own
    signal-to-noise), `limit` the field's 70 um flux below which nothing is
    catalogued (`SWIRE70_SNR_WINDOW`, above), on every row. A 70 um source whose nearest SWIRE row is a star or has no
    4.5 um flux (`keep` False) is not any galaxy's."""
    d70 = pd.read_csv(path70)
    n_row = ra.size
    flux = np.full(n_row, np.nan)
    sigma = np.full(n_row, np.nan)

    def unit_vector(r, d):
        r, d = np.radians(r), np.radians(d)
        return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])

    tree = cKDTree(unit_vector(ra, dec))
    sep, row = tree.query(unit_vector(d70["ra"].to_numpy(float), d70["dec"].to_numpy(float)))
    sep_arcsec = np.degrees(2.0 * np.arcsin(np.minimum(sep, 2.0) / 2.0)) * 3600.0
    pos_sigma = np.maximum(d70["unc_ra"].to_numpy(float), d70["unc_dec"].to_numpy(float))
    ok = (sep_arcsec <= SWIRE70_MATCH_SIGMAS * pos_sigma) & keep[row]
    # visit sources faintest-signal first so the highest-signal source claims a shared row last
    order = np.argsort(d70["snr"].to_numpy(float))
    for j in order[ok[order]]:
        i = row[j]
        flux[i] = d70["flux_prf"].iat[j]
        sigma[i] = d70["flux_prf"].iat[j] / d70["snr"].iat[j]
    at_cut = d70["snr"].to_numpy(float) < SWIRE70_SNR_MIN + SWIRE70_SNR_WINDOW
    limit = float(np.median(d70["flux_prf"].to_numpy(float)[at_cut]))
    return flux, sigma, limit, int(len(d70)), int(np.isfinite(flux[keep]).sum())


def _unit_vector(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _chord(arcsec):
    return 2.0 * np.sin(np.radians(arcsec / 3600.0) / 2.0)


class _ObsLimits(dict):
    """Observation id -> censoring limit (mJy), computed on first use from the
    band's detections by observation (see `PACS_LIMIT_PERCENTILE`). `record`
    holds `(n_det, faintest, limit)` for every observation asked about."""

    def __init__(self, flux_by_obs):
        super().__init__()
        self.flux_by_obs = flux_by_obs
        self.faintest = {o: float(f.min()) for o, f in flux_by_obs.items()}
        order = sorted(self.faintest, key=self.faintest.get)
        self.sorted_obs = order
        self.sorted_faintest = np.array([self.faintest[o] for o in order])
        self.record = {}
        self.printed = set()

    def __missing__(self, obsid):
        flux = self.flux_by_obs[obsid]
        own = self.faintest[obsid]
        if flux.size >= PACS_LIMIT_MIN_DETECTIONS:
            pool = flux
        else:
            factor = PACS_LIMIT_POOL_FACTOR
            while True:
                lo = np.searchsorted(self.sorted_faintest, own / factor, side="left")
                hi = np.searchsorted(self.sorted_faintest, own * factor, side="right")
                pool = np.concatenate([self.flux_by_obs[o] for o in self.sorted_obs[lo:hi]])
                if pool.size >= PACS_LIMIT_MIN_DETECTIONS or (lo == 0 and hi == len(self.sorted_obs)):
                    break
                factor *= PACS_LIMIT_POOL_FACTOR
        value = float(np.percentile(pool, PACS_LIMIT_PERCENTILE))
        self.record[obsid] = (int(flux.size), own, value)
        self[obsid] = value
        return value


def _read_pacs(path):
    """One HPPSC band file -> `(sources, limit_of_obs)`: the sources' `ra`,
    `dec`, `flux` (mJy), `snr`, `obsid` (finite positive flux and signal-to-
    noise only), and, per observation, its censoring limit (an `_ObsLimits`:
    obsid -> mJy, the flux below which that observation does not detect)."""
    src = pd.read_csv(path, usecols=["ra", "dec", "flux", "snr", "obsid"])
    src = src[np.isfinite(src["flux"]) & (src["flux"] > 0) & np.isfinite(src["snr"]) & (src["snr"] > 0)]
    flux_by_obs = {o: g.to_numpy(float) for o, g in src.groupby("obsid")["flux"]}
    return src, _ObsLimits(flux_by_obs)


def _match_pacs(src, limit_of_obs, ra_field, dec_field, ra, dec):
    """The PACS flux of each galaxy of one field (`ra`, `dec`) from one HPPSC
    band's sources. Footprint: the convex hull of each observation's
    sources inside the field's bounding box (padded), in the tangent plane at
    the field's mean position; an observation with fewer than
    `PACS_HULL_MIN_SOURCES` such sources has none. A galaxy is covered if it
    lies in any hull; its limit is the faintest source of the deepest
    observation covering it (`limit_of_obs`). Sources within
    `PACS_DEDUPE_ARCSEC` of a higher-signal source are dropped. Returns
    `(covered, flux, sigma, limit, offset_n, offset_s, counts)` per galaxy:
    `flux`/`sigma` NaN where covered and unmatched or not covered, `sigma` the
    flux over the catalogue's signal-to-noise; `offset_n`/`offset_s` the flux
    matched with the galaxy's position moved north/south by
    `PACS_CHANCE_OFFSET_ARCSEC` (NaN if unmatched or if the moved position is
    not covered); `counts` the field's sources used and dropped."""
    ra0, dec0 = float(np.mean(ra_field)), float(np.mean(dec_field))
    cosd = np.cos(np.radians(dec0))
    box = (ra_field.min() - PACS_BBOX_PAD_DEG, ra_field.max() + PACS_BBOX_PAD_DEG,
           dec_field.min() - PACS_BBOX_PAD_DEG, dec_field.max() + PACS_BBOX_PAD_DEG)
    q = src[(src["ra"] >= box[0]) & (src["ra"] <= box[1]) & (src["dec"] >= box[2]) & (src["dec"] <= box[3])]
    hulls, used_obs = [], []
    for obsid, s in q.groupby("obsid"):
        if len(s) < PACS_HULL_MIN_SOURCES:
            continue
        pts = np.column_stack([(s["ra"].to_numpy(float) - ra0) * cosd, s["dec"].to_numpy(float) - dec0])
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        hulls.append((hull.equations, pts.min(axis=0), pts.max(axis=0), limit_of_obs[obsid]))
        used_obs.append(obsid)

    def cover(r, d):
        """(covered, limit) of positions."""
        x, y = (r - ra0) * cosd, d - dec0
        cov = np.zeros(r.size, dtype=bool)
        lim = np.full(r.size, np.inf)
        for eq, lo, hi, limit_obs in hulls:
            near = np.flatnonzero((x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1]))
            if near.size == 0:
                continue
            inside = np.all(eq[:, 0:1] * x[near][None, :] + eq[:, 1:2] * y[near][None, :] + eq[:, 2:3] <= 1e-12, axis=0)
            hit = near[inside]
            cov[hit] = True
            lim[hit] = np.minimum(lim[hit], limit_obs)
        return cov, np.where(cov, lim, np.nan)

    covered, limit = cover(ra, dec)
    d_n, d_s = dec + PACS_CHANCE_OFFSET_ARCSEC / 3600.0, dec - PACS_CHANCE_OFFSET_ARCSEC / 3600.0
    cov_n, _ = cover(ra, d_n)
    cov_s, _ = cover(ra, d_s)

    q = q[q["obsid"].isin(used_obs)].sort_values("snr", ascending=False).reset_index(drop=True)
    sra, sdec = q["ra"].to_numpy(float), q["dec"].to_numpy(float)
    drop = np.zeros(len(q), dtype=bool)
    if len(q):
        tree_s = cKDTree(_unit_vector(sra, sdec))
        for i, nb in enumerate(tree_s.query_ball_point(_unit_vector(sra, sdec), _chord(PACS_DEDUPE_ARCSEC))):
            if drop[i]:
                continue
            for j in nb:
                if j > i:
                    drop[j] = True
    q = q[~drop].reset_index(drop=True)
    sflux, ssnr = q["flux"].to_numpy(float), q["snr"].to_numpy(float)

    def match(r, d, ok):
        flux = np.full(r.size, np.nan)
        sig = np.full(r.size, np.nan)
        if len(q) == 0 or not ok.any():
            return flux, sig
        tree = cKDTree(_unit_vector(q["ra"].to_numpy(float), q["dec"].to_numpy(float)))
        dist, j = tree.query(_unit_vector(r[ok], d[ok]))
        hit = dist <= _chord(PACS_MATCH_ARCSEC)
        where = np.flatnonzero(ok)[hit]
        flux[where] = sflux[j[hit]]
        sig[where] = sflux[j[hit]] / ssnr[j[hit]]
        return flux, sig

    flux, sigma = match(ra, dec, covered)
    off_n, _ = match(ra, d_n, cov_n)
    off_s, _ = match(ra, d_s, cov_s)
    return covered, flux, sigma, limit, off_n, off_s, (len(q), int(drop.sum()), len(used_obs))


def _read_field(path, field_index, log10_s_grid, pacs):
    """One SWIRE field CSV -> the surviving galaxies' rows (rule 10b: the
    catalogue is read one field at a time, never all six loaded together).
    A row survives if it is classed a galaxy by the adopted split and its
    4.5um (I2) flux is measured and positive -- the same population
    `prior.gal.build_colour_cdf_tables` tabulates the CDF axes on.
    """
    cols = list(SWIRE_FLUX_COLUMNS) + list(SWIRE_FLUX_ERR_COLUMNS) + ["stell_36", "ra", "dec"]
    df = pd.read_csv(path, usecols=cols)
    flux_mjy = df[list(SWIRE_FLUX_COLUMNS)].to_numpy(dtype=float) / 1000.0
    err_mjy = df[list(SWIRE_FLUX_ERR_COLUMNS)].to_numpy(dtype=float) / 1000.0
    stell36 = df["stell_36"].to_numpy(dtype=float)

    is_star = np.isfinite(stell36) & (stell36 >= STELLARITY_STAR_MIN)
    f1, f2, f3, f4 = flux_mjy[:, 0], flux_mjy[:, 1], flux_mjy[:, 2], flux_mjy[:, 3]
    valid_i2 = np.isfinite(f2) & (f2 > 0)
    keep = (~is_star) & valid_i2
    idx = np.flatnonzero(keep)

    # the 70 um flux of each surviving galaxy, by position (`_match_70um`)
    f70, s70, limit70, n_cat70, n_match70 = _match_70um(
        os.path.join(os.path.dirname(path), SWIRE70_FIELD_FILES[field_index]),
        df["ra"].to_numpy(float), df["dec"].to_numpy(float), keep)
    print(f"swire_galaxies: field {field_index}: 70 um catalogue {n_cat70} sources, "
          f"{n_match70} matched to galaxies; limit {limit70:.2f} mJy")

    # the PACS 100 and 160 um flux of each surviving galaxy (`_match_pacs`)
    all_ra, all_dec = df["ra"].to_numpy(float), df["dec"].to_numpy(float)
    pacs_cols = {}
    for tag, (src, limit_of_obs) in pacs.items():
        cov, fl, sg, lim, on, os_, (n_src, n_drop, n_obs) = _match_pacs(
            src, limit_of_obs, all_ra, all_dec, all_ra[idx], all_dec[idx])
        n_cov, n_det = int(cov.sum()), int(np.isfinite(fl).sum())
        for obs, (n_d, old_lim, new_lim) in sorted(limit_of_obs.record.items()):
            if obs not in limit_of_obs.printed:
                limit_of_obs.printed.add(obs)
                print(f"swire_galaxies: PACS limit {tag} obs {obs}: n_det={n_d} faintest={old_lim:.2f} "
                      f"limit={new_lim:.2f} mJy", flush=True)
        print(f"swire_galaxies: field {field_index}: {tag}: {n_obs} observations, {n_src} sources used "
              f"({n_drop} duplicates dropped); {n_cov} of {idx.size} galaxies covered, {n_det} detected, "
              f"{n_cov - n_det} censored; offset matches north {int(np.isfinite(on).sum())}, "
              f"south {int(np.isfinite(os_).sum())}", flush=True)
        pacs_cols.update({
            f"COVERED_{tag}": cov.astype(np.int8),
            f"FLUX_{tag}_MJY": fl.astype(np.float32),
            f"SIGMA_FLUX_{tag}_MJY": sg.astype(np.float32),
            f"LIMIT_{tag}_MJY": lim.astype(np.float32),
            f"DETECTED_{tag}": np.isfinite(fl).astype(np.int8),
            f"FLUX_{tag}_OFFSET_N_MJY": on.astype(np.float32),
            f"FLUX_{tag}_OFFSET_S_MJY": os_.astype(np.float32),
        })

    f1k, f2k, f3k, f4k = f1[idx], f2[idx], f3[idx], f4[idx]
    e1, e2, e3, e4 = err_mjy[idx, 0], err_mjy[idx, 1], err_mjy[idx, 2], err_mjy[idx, 3]

    # A band is missing -- flux or its own uncertainty absent or a
    # sentinel (SWIRE's -99), never usable for a colour or a bandwidth --
    # so both go NaN for that band, not +-inf (owner ruling): a reader
    # forming a kernel bandwidth downstream must never see a sentinel.
    def band_valid(f, e):
        return np.isfinite(f) & (f > 0) & np.isfinite(e) & (e > 0)

    v1, v2, v3, v4 = band_valid(f1k, e1), band_valid(f2k, e2), band_valid(f3k, e3), band_valid(f4k, e4)

    def log10_or_nan(f, v):
        return np.where(v, np.log10(np.where(v, f, 1.0)), np.nan)

    log10_f1, log10_f2 = log10_or_nan(f1k, v1), log10_or_nan(f2k, v2)
    log10_f3, log10_f4 = log10_or_nan(f3k, v3), log10_or_nan(f4k, v4)

    ln10 = np.log(10.0)
    sigma1 = np.where(v1, e1 / (f1k * ln10), np.nan)
    sigma2 = np.where(v2, e2 / (f2k * ln10), np.nan)
    sigma3 = np.where(v3, e3 / (f3k * ln10), np.nan)
    sigma4 = np.where(v4, e4 / (f4k * ln10), np.nan)

    # NODE reads I2's flux alone (the split's own gate, `valid_i2`, already
    # guarantees it is finite and positive), independent of whether I2's
    # own uncertainty is a sentinel -- a bad error never removes a galaxy
    # from the flux grid, only from a colour or sigma that uses that band.
    # `log10_s_grid` is the OWNER's grid (`population.gal.build_log10_s_grid`,
    # passed in by `build`), never a second copy (review ledger C16); the
    # bright-end pooling that keeps each node's own galaxy count usable
    # (`_pool_bright_nodes`) runs once in `build`, after every field's rows
    # are concatenated, not per field here.
    lo, hi = float(log10_s_grid[0]), float(log10_s_grid[-1])
    edges = 0.5 * (log10_s_grid[1:] + log10_s_grid[:-1])
    log10_f2_flux = np.log10(f2k)
    in_range = (log10_f2_flux >= lo) & (log10_f2_flux <= hi)
    node = np.clip(np.searchsorted(edges, log10_f2_flux), 0, log10_s_grid.size - 1)
    node = np.where(in_range, node, -1).astype(np.int16)

    colour_i1i2 = (log10_f1 - log10_f2).astype(np.float32)
    colour_i2i3 = (log10_f2 - log10_f3).astype(np.float32)
    colour_i2i4 = (log10_f2 - log10_f4).astype(np.float32)

    return dict(
        LOG10_S=log10_f2_flux.astype(np.float32),
        NODE=node,
        COLOUR_I1I2=colour_i1i2,
        COLOUR_I2I3=colour_i2i3,
        COLOUR_I2I4=colour_i2i4,
        SIGMA_COLOUR_I1I2=np.sqrt(sigma1 ** 2 + sigma2 ** 2).astype(np.float32),
        SIGMA_COLOUR_I2I3=np.sqrt(sigma2 ** 2 + sigma3 ** 2).astype(np.float32),
        SIGMA_COLOUR_I2I4=np.sqrt(sigma2 ** 2 + sigma4 ** 2).astype(np.float32),
        FIELD=np.full(idx.size, field_index, dtype=np.int8),
        FLUX_P70_MJY=f70[idx].astype(np.float32),
        SIGMA_FLUX_P70_MJY=s70[idx].astype(np.float32),
        LIMIT_P70_MJY=np.full(idx.size, limit70, dtype=np.float32),
        DETECTED_P70=np.isfinite(f70[idx]).astype(np.int8),
        **pacs_cols,
    ), int(is_star.sum()), int(df.shape[0])


def build(config, regions=None):
    """Writes `sky/derived/swire/galaxies_swire_survey.hdf5`, one row per
    surviving galaxy across all six SWIRE fields. Survey-wide: `regions` is
    accepted for the standard `build` signature and ignored (rule 5c)."""
    if regions is not None:
        print("swire_galaxies: survey-wide product, --regions ignored")

    dest_dir = f"{config.data_root}/sky/download/swire"
    fazio_path = f"{config.data_root}/{FAZIO_PATH_SUFFIX}"
    log10_s_grid = build_log10_s_grid(fazio_path)
    with progress_module.Stage("sky.derived.swire_galaxies") as st:
        field_blocks, n_stars_removed, n_fields = [], 0, len(SWIRE_FIELD_FILES)
        pacs = {}
        for tag, name in PACS_BANDS:
            pacs_path = f"{config.data_root}/sky/download/{HPPSC_DIR}/{name}"
            if not os.path.exists(pacs_path):
                raise FileNotFoundError(f"swire_galaxies: no HPPSC catalogue at {pacs_path!r}")
            pacs[tag] = _read_pacs(pacs_path)
        for i, name in enumerate(SWIRE_FIELD_FILES):
            path = f"{dest_dir}/{name}"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"swire_galaxies: no SWIRE field catalogue at {path!r} -- run the "
                    f"'sesnaimpute.sky.download.swire.build' RUNBOOK line")
            block, n_star, n_row = _read_field(path, i, log10_s_grid, pacs)
            field_blocks.append(block)
            n_stars_removed += n_star
            print(f"swire_galaxies: {name}: {n_row} rows, {n_star} stars removed, "
                  f"{block['LOG10_S'].size} galaxies kept")
            st.tick(i + 1, n_fields, "fields")

        columns = {k: np.concatenate([b[k] for b in field_blocks]) for k in field_blocks[0]}
        n_galaxies = int(columns["LOG10_S"].size)

        # the bright-end pool (review ledger C16): every node a galaxy is
        # read from a column survey-wide, so pooling runs once here, on the
        # six fields combined, not per field.
        counts_raw = np.bincount(columns["NODE"][columns["NODE"] >= 0], minlength=N_S_GRID)
        columns["NODE"], counts_pooled = _pool_bright_nodes(columns["NODE"], N_S_GRID, MIN_NODE_GALAXIES)
        n_merged = int(np.sum((counts_raw > 0) & (counts_pooled == 0)))
        print(f"swire_galaxies: bright-node pool: floor={MIN_NODE_GALAXIES}, "
              f"{n_merged} of {N_S_GRID} nodes merged away; node counts before/after "
              f"at the bright edge (56-60)="
              f"{list(zip(counts_raw[56:].tolist(), counts_pooled[56:].tolist()))}")

        finite_i1i2 = np.isfinite(columns["COLOUR_I1I2"])
        finite_i2i3 = np.isfinite(columns["COLOUR_I2I3"])
        finite_i2i4 = np.isfinite(columns["COLOUR_I2I4"])
        finite_all = finite_i1i2 & finite_i2i3 & finite_i2i4
        node = columns["NODE"]
        n_node_i1i2 = np.zeros(N_S_GRID, dtype=np.int64)
        n_node_all = np.zeros(N_S_GRID, dtype=np.int64)
        in_grid = node >= 0
        np.add.at(n_node_i1i2, node[in_grid & finite_i1i2], 1)
        np.add.at(n_node_all, node[in_grid & finite_all], 1)

        out_path = config_module.product_path(config, "sky/derived", "swire", "galaxies", "survey")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs["GRANULE"] = "survey"
            f.attrs["STELLARITY_STAR_MIN"] = STELLARITY_STAR_MIN
            f.attrs["FIELDS"] = ";".join(SWIRE_FIELD_FILES)
            f.attrs["N_STARS_REMOVED"] = n_stars_removed
            f.attrs["N_GALAXIES"] = n_galaxies
            f.attrs["N_FINITE_I1I2"] = int(finite_i1i2.sum())
            f.attrs["N_FINITE_I2I3"] = int(finite_i2i3.sum())
            f.attrs["N_FINITE_I2I4"] = int(finite_i2i4.sum())
            f.attrs["N_FINITE_ALL"] = int(finite_all.sum())
            f.attrs["MIN_NODE_GALAXIES"] = MIN_NODE_GALAXIES
            f.attrs["N_NODES_MERGED"] = n_merged
            build_module.write_dataset(
                f, "LOG10_S_GRID", log10_s_grid.astype(np.float64),
                *REGISTRY[(_STEM, "LOG10_S_GRID")])
            build_module.write_dataset(f, "N_NODE_I1I2", n_node_i1i2, *REGISTRY[(_STEM, "N_NODE_I1I2")])
            build_module.write_dataset(f, "N_NODE_ALL", n_node_all, *REGISTRY[(_STEM, "N_NODE_ALL")])
            for name, arr in columns.items():
                build_module.write_dataset(f, name, arr, *REGISTRY[(_STEM, name)])

        st.done(out_path, n_galaxies=n_galaxies, n_stars_removed=n_stars_removed,
                 n_nodes_merged=n_merged)
    return dict(n_galaxies=n_galaxies, n_stars_removed=n_stars_removed, path=out_path,
                n_nodes_merged=n_merged)


if __name__ == "__main__":
    build_module.run(build)
