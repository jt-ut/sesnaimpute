"""The Planck arm of the source column, `A_K` from thermal-dust emission
(SPEC_PRIORS.md section 1.1).

Per source: bilinear TAU353 at the source's Galactic position (the
sub-pixel information a nearest-pixel read would throw away) and
nearest-pixel ERR_TAU (error propagation is not meaningful under
interpolation), from the Planck R1.20 all-sky thermal-dust model (nside
2048, NESTED). `A_K = A_TAU * TAU353`. The uncertainty composes three
terms in quadrature: the per-source statistical term from ERR_TAU, the
within-region dispersion of the TAU353->A_K relation (a fixed offset plus
a term growing with `A_K`), and the region-to-region scatter of the same
relation, evaluated per source (not divided by the number of independent
beams a coarser pixel would average over -- a sightline sees exactly one
Planck beam). This is one of the two arms merged in `column.py`; never
blended with the Herschel arm here.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run

# Planck R1.20 thermal-dust model: Planck Collaboration XI 2014, A&A 571, A11.
PLANCK_NSIDE = 2048
PLANCK_BAD = -1.6375e30  # the map's own bad-pixel sentinel

# Process-local cache so a joblib worker that draws more than one region
# reads the 1.6 GB FITS map once, not once per region.
_MAP_CACHE = {}


def _planck_map_path(config):
    import glob
    found = sorted(glob.glob(os.path.join(config.data_root, "sky/download/planck_r120", "*.fits")))
    if not found:
        raise FileNotFoundError(
            "sky.derived.planck_source_column.build: no Planck R1.20 FITS map under "
            f"{config.data_root}/sky/download/planck_r120 -- run the Planck R1.20 download RUNBOOK line"
        )
    return found[0]


def _load_planck_map(fits_path):
    """TAU353 and ERR_TAU as full HEALPix NESTED arrays at `PLANCK_NSIDE`,
    bad pixels set to NaN; cached per process."""
    cached = _MAP_CACHE.get(fits_path)
    if cached is not None:
        return cached
    from astropy.io import fits
    with fits.open(fits_path, memmap=True) as hd:
        hdr = hd[1].header
        if int(hdr["NSIDE"]) != PLANCK_NSIDE or hdr["ORDERING"] != "NESTED":
            raise ValueError(f"{fits_path}: expected nside {PLANCK_NSIDE} NESTED, got "
                             f"nside {hdr['NSIDE']} {hdr['ORDERING']}")
        tau353 = np.asarray(hd[1].data["TAU353"], dtype=np.float64)
        err_tau = np.asarray(hd[1].data["ERR_TAU"], dtype=np.float64)
    tau353[tau353 <= PLANCK_BAD / 2.0] = np.nan
    err_tau[err_tau <= PLANCK_BAD / 2.0] = np.nan
    _MAP_CACHE[fits_path] = (tau353, err_tau)
    return tau353, err_tau


def _load_planck_calibration(config):
    """The survey-level Planck calibration scalars: A_TAU (the TAU353 ->
    A_K coefficient), SIGMA_WITHIN_S0/SIGMA_WITHIN_F (the within-region
    dispersion's offset and slope), SIGMA_REGION_FRAC (the region-to-
    region scatter, as a fraction of A_K), PLANCK_FWHM_ARCMIN (the
    measured effective beam)."""
    path = config_module.product_path(config, "sky/derived", "planck", "calibration", "survey")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"sky.derived.planck_source_column.build: Planck calibration product missing at {path!r} "
            "-- run the sesnaimpute.sky.derived.planck_column RUNBOOK line for it"
        )
    with h5py.File(path, "r") as f:
        return dict(
            a_tau=float(f.attrs["A_TAU"]),
            s0_mag=float(f.attrs["SIGMA_WITHIN_S0"]),
            f_rel=float(f.attrs["SIGMA_WITHIN_F"]),
            sig_region_frac=float(f.attrs["SIGMA_REGION_FRAC"]),
            fwhm_arcmin=float(f.attrs["PLANCK_FWHM_ARCMIN"]),
        )


def sample_planck_column(l_deg, b_deg, tau353, err_tau, nside=PLANCK_NSIDE):
    """Bilinear TAU353 and nearest-pixel ERR_TAU at `(l_deg, b_deg)`.
    `tau353`/`err_tau` are full HEALPix NESTED maps at `nside`. Returns
    (pix, tau_nearest, err_nearest, tau_bilinear)."""
    import healpy as hp
    l_deg = np.asarray(l_deg, dtype=np.float64)
    b_deg = np.asarray(b_deg, dtype=np.float64)
    pix = hp.ang2pix(nside, l_deg, b_deg, nest=True, lonlat=True)
    tau_nn = np.asarray(tau353)[pix]
    err_nn = np.asarray(err_tau)[pix]
    tau_interp = hp.get_interp_val(tau353, l_deg, b_deg, nest=True, lonlat=True)
    return pix, tau_nn, err_nn, tau_interp


def compose_planck_sigma(a_col, err_tau_nn, a_tau, s0_mag, f_rel, sig_region_frac):
    """SIGMA_A_K^2 = SIGMA_STAT_K^2 + SIGMA_WITHIN_K^2 + SIGMA_REGION_K^2.
    SIGMA_STAT_K is not divided by the number of independent beams a
    coarser pixel would average over: a per-source sightline sees exactly
    one Planck beam."""
    sig_stat = a_tau * err_tau_nn
    sig_within = np.sqrt(s0_mag ** 2 + (f_rel * a_col) ** 2)
    sig_region = sig_region_frac * a_col
    sig = np.sqrt(sig_stat ** 2 + sig_within ** 2 + sig_region ** 2)
    return sig, sig_stat, sig_within, sig_region


def _read_curated_positions(config, region):
    path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"sky.derived.planck_source_column.build: curated catalogue missing for region {region!r} "
            f"at {path!r} -- run the curated-catalogue RUNBOOK line for it"
        )
    with h5py.File(path, "r") as f:
        return np.asarray(f["GAL_L_DEG"][:], dtype=np.float64), np.asarray(f["GAL_B_DEG"][:], dtype=np.float64)


def _build_one_region(config, region, fits_path, cal):
    l, b = _read_curated_positions(config, region)
    tau353, err_tau = _load_planck_map(fits_path)
    _pix, _tau_nn, err_nn, tau_interp = sample_planck_column(l, b, tau353, err_tau)

    a_col = cal["a_tau"] * tau_interp
    sig, sig_stat, sig_within, sig_region = compose_planck_sigma(
        a_col, err_nn, cal["a_tau"], cal["s0_mag"], cal["f_rel"], cal["sig_region_frac"])

    out_path = config_module.product_path(config, "sky/derived", "planck", "column", "source", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.create_dataset("A_K", data=a_col.astype(np.float32))
        f.create_dataset("SIGMA_A_K", data=sig.astype(np.float32))
        f.create_dataset("SIGMA_STAT_K", data=sig_stat.astype(np.float32))
        f.create_dataset("SIGMA_WITHIN_K", data=sig_within.astype(np.float32))
        f.create_dataset("SIGMA_REGION_K", data=sig_region.astype(np.float32))
    return region, int(l.size), float(np.median(a_col))


def build(config, regions=None):
    """Builds the Planck source-column product for each region in
    `regions` (default: every region in `regions.REGIONS`), parallelised
    over regions with joblib; each worker reads the Planck map once per
    process and samples every region it draws from its own cached copy.
    """
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    fits_path = _planck_map_path(config)
    cal = _load_planck_calibration(config)
    # Threads, not processes: `_MAP_CACHE` is process-local, so a process
    # pool would hold one 1.6 GB Planck map per worker (config.n_jobs
    # copies at once); every per-region computation here is vectorised
    # numpy/healpy array arithmetic, which releases the GIL, so threads
    # cost nothing and keep the map to the one cached copy (CODING_RULES.md 10a).
    Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(_build_one_region)(config, region, fits_path, cal) for region in regions)


if __name__ == "__main__":
    run(build)
