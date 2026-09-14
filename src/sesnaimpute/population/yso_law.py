"""The young-star law's coefficient, fitted per SESNA region on the
Dunham et al. 2015 census (c2d + Gould Belt, 2,966 YSOs, classified
independently of SESNA) -- the owner's 2026-09-14 ruling (`_W83_design.md`):
the young-star level is no longer Pokhrel et al. 2020's coefficient, it is
fitted here, on our own column maps, nothing about SESNA read except
positions and the footprint.

Method, per region: the region's ADMITTED nside-512 pixels (`catalog.
depth_grid`'s own product) restricted to a FIT FOOTPRINT -- those pixels
whose gas column (`sky/derived/adopted/column_adopted_sightline.hdf5`'s
`A_K` at nside 256, the pixel's own parent) is at least `FIT_A_K_MIN`: 0.22
mag (A_V >= 2, the c2d coverage rule) where the census cloud(s) overlapping
the region are c2d clouds, 0.33 mag (A_V >= 3) where they are Gould Belt
clouds -- decided from `table1.dat`'s own `Survey` field for whichever
Dunham cloud(s) the region's admitted (unrestricted) footprint contains
census objects from, majority by count, ties to Gould Belt's own stricter
floor. `N_CENSUS_REGION` is the count of census objects (every slope) whose
position falls in the fit footprint. The model's own predicted count over
that same footprint, with the coefficient normalised OUT (so what remains
is a pure geometric/column integral, coefficient 1): `population.yso.
law_area_integral(config, region, pix)` (already `KAPPA_HERSCHEL` times the
geometric integral) times the pixel's solid angle and its `cloud_frac**2`
-- `population.young_stars.build_region` forms exactly this per pixel, so
its own `sightline_lookup` is reused here for `cloud_frac`, not re-derived
-- divided by `KAPPA_HERSCHEL` to strip the old coefficient back out,
summed over the footprint: `I`. `KAPPA_REGION = N / I`, the Poisson
maximum-likelihood coefficient, fitted only where `N >= 50`; its 68%
interval is the standard Poisson (Garwood) interval on `N`, converted by
the same `/ I`. `KAPPA_POOLED` is the geometric mean of the fitted regions'
kappa (the centre of a lognormal cloud-to-cloud scatter, each cloud weighing
the same); `LAW_BAND_DEX` is the rms of the fitted regions' own log10 kappa
about log10 `KAPPA_POOLED`. `KAPPA_USED` is `KAPPA_REGION` where fitted, else
`KAPPA_POOLED`. The class shares (`CLASS_SHARE_PROTO/DISK/WEAK`) are the
whole census's own `ALPHA0` slope fractions, independent of region.

Writes `population/yso/law_yso_region.hdf5` (unchanged path/name); no
`KAPPA_HERSCHEL`/`KAPPA_PLANCK` dataset is written any more -- a reader
that still expects one fails loudly, which is the point.
"""

import os

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.stats import chi2

from sesnaimpute import config as config_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute import tables as tables_module
from sesnaimpute.build import run
from sesnaimpute.population import yso as yso_module
from sesnaimpute.population import young_stars as young_stars_module

#: The fit footprint's own column floor (SPEC ruling, `_W83_design.md`):
#: the c2d coverage rule (A_V >= 2) and the Gould Belt rule (A_V >= 3),
#: at A_K = A_V / 9.something folded into these two mag values directly
#: as the ruling states them.
FIT_A_K_MIN_C2D = 0.22
FIT_A_K_MIN_GB = 0.33

#: A region is fitted only with at least this many census objects in its
#: own fit footprint; fewer leaves `KAPPA_REGION` (and its interval) NaN.
MIN_N_FOR_FIT = 50

#: Report-only column bins (mag of A_K) so the exponent 2 can be eyed
#: per region (the design's acceptance bar).
_REPORT_BIN_EDGES = (0.22, 0.5, 1.0, np.inf)

#: nside-512's own fixed pixel solid angle (deg^2) -- every HEALPix pixel
#: at a given nside has the identical area, so this is one constant, not
#: a per-pixel read.
OMEGA_PIX_512_DEG2 = float(hp.nside2pixarea(512, degrees=True))


# ====================================================================
# the census: positions, clouds, slopes
# ====================================================================

def _census_survey_path(config):
    return config_module.product_path(config, "sky/derived", "dunham2015", "yso", "survey")


def _load_census(config):
    """The whole 2,966-row census: `ra_deg`, `dec_deg`, `cloud` (survey
    cloud name), `alpha0`, and `pix512` (the source's own nside-512
    galactic NESTED pixel, the survey's standard pixelisation -- the
    SAME frame the region admission and depth-grid pixels are in)."""
    path = _census_survey_path(config)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "population.yso_law: no Dunham census at %s -- run the "
            "'sesnaimpute.sky.derived.dunham_yso' RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        ra_deg = np.asarray(f["RA_DEG"][:], dtype=np.float64)
        dec_deg = np.asarray(f["DEC_DEG"][:], dtype=np.float64)
        cloud = np.array([c.decode("utf-8") if isinstance(c, bytes) else c
                           for c in f["CLOUD"][:]])
        alpha0 = np.asarray(f["ALPHA0"][:], dtype=np.float64)
    gal = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    pix512 = hp.ang2pix(512, gal.l.deg, gal.b.deg, nest=True, lonlat=True).astype(np.int64)
    return dict(ra_deg=ra_deg, dec_deg=dec_deg, cloud=cloud, alpha0=alpha0, pix512=pix512)


def _cloud_survey_map(config):
    """`{cloud_name: "c2d" or "GB"}`, table1.dat's own `Survey` field
    (the census cloud table), read directly rather than duplicated as a
    stored column: the pipe-delimited `Cloud|Survey|Dist...` layout
    `sky.derived.dunham_yso._read_table1` also reads."""
    path = f"{config.data_root}/sky/download/dunham2015/table1.dat"
    survey = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("|")
            survey[fields[0].strip()] = fields[1].strip()
    return survey


# ====================================================================
# per-region fit footprint
# ====================================================================

def _admitted_pix512(config, region):
    path = config_module.product_path(config, "catalog", "sesna", "depth-grid",
                                       "hpx512", region=region)
    with h5py.File(path, "r") as f:
        return np.sort(np.asarray(f["HPX_PIX_512"][:], dtype=np.int64))


_COLUMN_SIGHTLINE_CACHE = {}


def _column_sightline(config):
    """`(pix256_sorted, a_k_sorted)`, the adopted sightline column's own
    nside-256 pixel and `A_K`, cached (every region's fit reads it)."""
    key = id(config)
    if key not in _COLUMN_SIGHTLINE_CACHE:
        path = config_module.product_path(config, "sky/derived", "adopted",
                                           "column", "sightline")
        with h5py.File(path, "r") as f:
            pix256 = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
            a_k = np.asarray(f["A_K"][:], dtype=np.float64)
        order = np.argsort(pix256)
        _COLUMN_SIGHTLINE_CACHE[key] = (pix256[order], a_k[order])
    return _COLUMN_SIGHTLINE_CACHE[key]


def _a_k_of_pix512(config, pix512):
    """Each `pix512` pixel's own gas column, `A_K` of its nside-256
    PARENT sightline (`pix512 >> 2`) -- `NaN` where that parent carries
    no sightline column row."""
    pix256_sorted, a_k_sorted = _column_sightline(config)
    parent = pix512 >> 2
    loc = np.searchsorted(pix256_sorted, parent)
    capped = np.minimum(loc, max(pix256_sorted.size - 1, 0))
    matched = (pix256_sorted.size > 0) & (pix256_sorted[capped] == parent)
    out = np.full(pix512.shape, np.nan, dtype=np.float64)
    out[matched] = a_k_sorted[capped[matched]]
    return out


def _fit_a_k_min(census, survey_map, admitted_pix512):
    """The region's own `FIT_A_K_MIN`: which Dunham cloud(s) the
    region's (unrestricted) admitted footprint holds census objects
    from, majority vote by count, ties to Gould Belt's own stricter
    floor -- `NaN`/`None` where the region holds no census object at
    all (no cloud identifiable, so no fit is possible)."""
    admitted_sorted = admitted_pix512  # already sorted by caller
    loc = np.searchsorted(admitted_sorted, census["pix512"])
    capped = np.minimum(loc, max(admitted_sorted.size - 1, 0))
    matched = (admitted_sorted.size > 0) & (admitted_sorted[capped] == census["pix512"])
    if not matched.any():
        return None
    clouds_here = census["cloud"][matched]
    uniq, counts = np.unique(clouds_here, return_counts=True)
    surveys = np.array([survey_map.get(c, "GB") for c in uniq])
    n_c2d = int(counts[surveys == "c2d"].sum())
    n_gb = int(counts[surveys == "GB"].sum())
    return FIT_A_K_MIN_C2D if n_c2d > n_gb else FIT_A_K_MIN_GB


def _poisson_interval_68(n):
    """The standard (Garwood) 68% Poisson confidence interval on an
    observed count `n` -- the exact Gamma-function interval (Gehrels
    1986's own construction), not a Gaussian approximation, since
    several regions fit near the `MIN_N_FOR_FIT = 50` floor."""
    alpha = 1.0 - 0.6826894921370859
    lo = 0.0 if n == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * n)
    hi = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (n + 1))
    return lo, hi


def _fit_region(config, region, census, survey_map):
    """One region's fit row: `dict(d_r_pc, pc2_per_deg2, fit_a_k_min,
    n_census, kappa, kappa_lo, kappa_hi, i_integral, bin_report)`.
    `kappa`/`kappa_lo`/`kappa_hi` are `NaN` where `n_census <
    MIN_N_FOR_FIT` or no cloud overlap exists at all."""
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    pc2 = float(yso_module.pc2_per_deg2(d_r_pc))

    admitted = _admitted_pix512(config, region)
    fit_a_k_min = _fit_a_k_min(census, survey_map, admitted)

    row = dict(d_r_pc=float(d_r_pc), pc2_per_deg2=pc2,
               fit_a_k_min=np.nan, n_census=0,
               kappa=np.nan, kappa_lo=np.nan, kappa_hi=np.nan,
               i_integral=np.nan, bin_report=None)
    if fit_a_k_min is None or admitted.size == 0:
        return row
    row["fit_a_k_min"] = fit_a_k_min

    a_k_admitted = _a_k_of_pix512(config, admitted)
    in_footprint = np.isfinite(a_k_admitted) & (a_k_admitted >= fit_a_k_min)
    footprint = admitted[in_footprint]
    a_k_footprint = a_k_admitted[in_footprint]
    if footprint.size == 0:
        return row

    loc = np.searchsorted(footprint, census["pix512"])
    capped = np.minimum(loc, max(footprint.size - 1, 0))
    matched = (footprint.size > 0) & (footprint[capped] == census["pix512"])
    n_per_pix = np.bincount(capped[matched], minlength=footprint.size).astype(np.float64)
    n_census = int(n_per_pix.sum())
    row["n_census"] = n_census

    d_front, d_back = yso_module.cloud_interval_pc(config, region)
    _xi, _mass, cloud_frac, _removed = young_stars_module.sightline_lookup(
        config, region, footprint, d_front, d_back)

    # `population.young_stars.build_region` forms exactly this per
    # pixel (`law_area_integral * omega_pix * cloud_frac**2`), reused
    # rather than re-derived, at coefficient 1: the pure geometric/column
    # integral this fit solves for.
    i_per_pix = (yso_module.law_area_integral(config, region, footprint, kappa=1.0)
                 * OMEGA_PIX_512_DEG2 * cloud_frac ** 2)
    i_integral = float(i_per_pix.sum())
    row["i_integral"] = i_integral

    bins = []
    for lo, hi in zip(_REPORT_BIN_EDGES[:-1], _REPORT_BIN_EDGES[1:]):
        sel = (a_k_footprint >= lo) & (a_k_footprint < hi)
        bins.append((lo, hi, float(n_per_pix[sel].sum()), float(i_per_pix[sel].sum())))
    row["bin_report"] = bins

    if n_census >= MIN_N_FOR_FIT and i_integral > 0:
        kappa = n_census / i_integral
        lo_n, hi_n = _poisson_interval_68(n_census)
        row["kappa"] = kappa
        row["kappa_lo"] = lo_n / i_integral
        row["kappa_hi"] = hi_n / i_integral
    return row


# ====================================================================
# class shares (whole census, region-independent)
# ====================================================================

def _class_shares(census):
    alpha0 = census["alpha0"]
    finite = np.isfinite(alpha0)
    a = alpha0[finite]
    n = a.size
    proto = float(np.count_nonzero(a >= -0.3)) / n
    disk = float(np.count_nonzero((a >= -1.6) & (a < -0.3))) / n
    weak = float(np.count_nonzero(a < -1.6)) / n
    return proto, disk, weak


# ====================================================================
# build
# ====================================================================

def build(config, regions=None):
    """Writes `population/yso/law_yso_region.hdf5` (module docstring).
    `regions` (default: all thirty) selects which regions' own rows are
    fitted and written this call (CODING_RULES.md 5c, other rows left
    untouched); `KAPPA_POOLED`/`LAW_BAND_DEX`/the class shares are
    computed over THIS call's own fitted regions -- a full run (the
    default) is what the schema's own pooled figures describe."""
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]

    census = _load_census(config)
    survey_map = _cloud_survey_map(config)

    rows = {}
    with progress.Stage("population.yso_law") as st:
        for i, region in enumerate(names):
            rows[region] = _fit_region(config, region, census, survey_map)
            st.tick(i + 1, len(names), region)

        fitted = [r for r in names if np.isfinite(rows[r]["kappa"])]
        # the pooled coefficient is the geometric mean over the fitted
        # clouds (owner's ruling 2026-09-14): the coefficient scatters
        # lognormally from cloud to cloud, so the centre of that
        # distribution is the mean of the logarithm, and the band is
        # the rms of the logarithm about it; each cloud weighs the same,
        # since the cloud-to-cloud spread is ten times any cloud's own
        # interval
        if fitted:
            log_k = np.array([np.log10(rows[r]["kappa"]) for r in fitted])
            kappa_pooled = float(10.0 ** np.mean(log_k))
            law_band_dex = float(np.sqrt(np.mean((log_k - np.log10(kappa_pooled)) ** 2)))
        else:
            kappa_pooled, law_band_dex = np.nan, np.nan

        proto, disk, weak = _class_shares(census)

        path = config_module.product_path(config, "population", "yso", "law", "region")
        tables_module.update_rows(
            path, names,
            {
                "D_R_PC": np.array([rows[r]["d_r_pc"] for r in names], dtype=np.float64),
                "PC2_PER_DEG2": np.array([rows[r]["pc2_per_deg2"] for r in names], dtype=np.float64),
                "KAPPA_REGION": np.array([rows[r]["kappa"] for r in names], dtype=np.float64),
                "KAPPA_REGION_LO": np.array([rows[r]["kappa_lo"] for r in names], dtype=np.float64),
                "KAPPA_REGION_HI": np.array([rows[r]["kappa_hi"] for r in names], dtype=np.float64),
                "N_CENSUS_REGION": np.array([rows[r]["n_census"] for r in names], dtype=np.int64),
                "FIT_A_K_MIN": np.array([rows[r]["fit_a_k_min"] for r in names], dtype=np.float64),
            },
            granule="region")

        with tables_module.open_product(path, granule="region") as f:
            for stale in ("KAPPA_HERSCHEL", "KAPPA_PLANCK"):
                if stale in f:
                    del f[stale]
            kappa_region = np.asarray(f["KAPPA_REGION"][:], dtype=np.float64)
            kappa_used = np.where(np.isfinite(kappa_region), kappa_region, kappa_pooled)
            for name, value in (
                ("KAPPA_USED", kappa_used),
                ("KAPPA_POOLED", np.float64(kappa_pooled)),
                ("LAW_BAND_DEX", np.float64(law_band_dex)),
                ("CLASS_SHARE_PROTO", np.float64(proto)),
                ("CLASS_SHARE_DISK", np.float64(disk)),
                ("CLASS_SHARE_WEAK", np.float64(weak)),
            ):
                if name in f:
                    del f[name]
                f.create_dataset(name, data=value)
            f.attrs["KAPPA_EXPONENT"] = 2.0
            f.attrs["CENSUS"] = "Dunham et al. 2015"

        st.done(path, n_fitted=len(fitted), kappa_pooled=float(kappa_pooled),
                 law_band_dex=float(law_band_dex))

    print("region                    N        I           kappa            68%% interval  A_K_min")
    for region in names:
        r = rows[region]
        k = r["kappa"]
        k_str = ("%8.3f" % k) if np.isfinite(k) else "     NaN"
        lo_str = ("%.3f" % r["kappa_lo"]) if np.isfinite(r["kappa_lo"]) else "NaN"
        hi_str = ("%.3f" % r["kappa_hi"]) if np.isfinite(r["kappa_hi"]) else "NaN"
        amin = ("%.2f" % r["fit_a_k_min"]) if np.isfinite(r["fit_a_k_min"]) else " -- "
        print("%-25s %5d %11.4f  %s  [%s, %s]   %s"
              % (region, r["n_census"], r["i_integral"] if np.isfinite(r["i_integral"]) else float("nan"),
                 k_str, lo_str, hi_str, amin))
        if r["bin_report"]:
            for lo, hi, n_bin, i_bin in r["bin_report"]:
                hi_disp = "inf" if not np.isfinite(hi) else ("%.2f" % hi)
                print("    A_K [%.2f, %s): N=%.0f I=%.4f" % (lo, hi_disp, n_bin, i_bin))
    print("KAPPA_POOLED=%.3f  LAW_BAND_DEX=%.3f  n_fitted=%d"
          % (kappa_pooled, law_band_dex, len(fitted)))
    print("CLASS_SHARE_PROTO=%.4f CLASS_SHARE_DISK=%.4f CLASS_SHARE_WEAK=%.4f"
          % (proto, disk, weak))
    return path


if __name__ == "__main__":
    run(build)
