"""The adopted source column: the finer arm wherever it reaches
(SPEC_PRIORS.md section 1.1).

`A_COL_K = A_HERSCHEL_K - ZP_FIELD[field]` (36.3 arcsec beam) where the
region has a Herschel product and the source is covered, finite and
positive; `A_COL_K = A_K` from the Planck arm (`planck_source_column.py`,
PLANCK_FWHM_ARCMIN beam) elsewhere. A measured systematic left unapplied
is an error of its own size (owner, 2026-09-06): `field` is the source's
own region, one of the 13 `_measure_field_zeropoints` finds enough
low-column Herschel-covered admitted sightline pixels for. That
measurement -- median `A_HERSCHEL - A_PLANCK` at each field's own
Herschel-covered admitted sightline pixels, `A_PLANCK` below
`FIELD_ZP_LOW_COLUMN_CUT_AK` -- runs here, in the same pass that merges
the two arms (`build`), and is written into `sky.derived.herschel_column`'s
`sigma`/`survey` product (`FIELD_NAME`/`ZP_FIELD`/`ZP_SIGMA_FIELD`),
not read from a second product that needs a prior pass of this same
stage to exist. `merge_region(config, region, cal)` called standalone,
with no `field_zp` of its own, instead reads that already-written table
(`_load_field_zeropoints`); a Herschel-covered region absent from it (no
qualifying pixels) gets no offset and `ZP_SIGMA_K = 0`, unchanged from
before. Sigma, the beam and the
Herschel map id all follow the same arm as the value -- one arm per
source, never blended; sigma is rebuilt from the Herschel arm's own
random term (`SIGMA_RAND_K`) plus the field's own `ZP_SIGMA_K`, not
copied from `column`/`source`'s stale, pre-refit `SIGMA_A_K`.
`ZP_SIGMA_K` (mag), zero for a Planck-arm source, is a new per-source
column of the adopted product: the field offset's own uncertainty, not
the offset itself. A merged value that is anywhere non-finite or
non-positive raises rather than ships, since both arms guarantee 100%
finite, positive coverage on their own before this correction.

`build_sightline` writes the same Herschel-where-covered, Planck-elsewhere
rule at sightline granule (SPEC_PRIORS.md 1.1, 1.4): the adopted column
`profile.py` normalises the 3-D map to, instead of the Planck arm alone.
For every admitted nside-256 pixel of every region, the Herschel value is
the HGBS N(H2) maps block-averaged over the pixel -- `planck_column.py`'s
own block-average, reused unchanged, at its `MIN_FILL`-free coverage rule
(a pixel counts covered when at least half its native samples are
finite); its sigma is `herschel_column.py`'s per-map SIG_ZP/SIG_RAND
model. Elsewhere the pixel takes the Planck sightline column and its
composed sigma from `planck_column.py`'s own sightline product.

`build_column_check` is the disagreement check SPEC_PRIORS.md 1.1/1.4
motivates: on every Herschel-covered sightline pixel, the Herschel value,
the Planck value, and the 3-D map's own cumulative extinction at its
edge (`profile.py`'s inner/outer/splice/measure chain, called before any
rescaling), binned by quartile of the Herschel column and by region.

`_build_one_extinction_region`/`build_extinction_sightline` write the
second column, the one a star's light passes through: the emission-based
adopted column calibrated onto the scale of that region's Juvela &
Montillaud 2016 NICEST star-colour map (a whole-sightline 2MASS reddening
measurement, `sky.derived.juvela_extinction`), by the method of Lombardi,
Bouy, Alves & Lada (2014, A&A 566, A45), who calibrated Planck tau353
against NICEST the same way. One region-wide linear fit, `A_K(NICEST) =
CAL_OFFSET_K + CAL_SLOPE * A_K(adopted)`, is made over the nside-1024
cells (the reference map's own 3' beam) where both the cell's mean
NICEST and mean adopted column read below `UNSATURATED_A_K_MAG` --
neither tracer has begun to saturate there -- and then applied to every
source's and every sightline's own adopted-column value, in every cell,
whether or not that cell entered the fit. `CAL_OFFSET_K` and
`CAL_SLOPE` are stored once per region, in the source product, and read
from there to rescale the sightline product.
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.sky.derived import planck_column
from sesnaimpute.sky.derived import profile as profile_module
from sesnaimpute.sky.derived.planck_source_column import _load_planck_calibration

_COLUMN_SOURCE_STEM = "column_adopted_source"
_EXTINCTION_SOURCE_STEM = "extinction_adopted_source"
_COLUMN_SIGHTLINE_STEM = "column_adopted_sightline"
_EXTINCTION_SIGHTLINE_STEM = "extinction_adopted_sightline"
_CHECK_STEM = "column-check_adopted_survey"
_HERSCHEL_SIGMA_STEM = "sigma_herschel_survey"

PROV_HERSCHEL = np.uint8(0)
PROV_PLANCK = np.uint8(1)
HERSCHEL_STATED_FWHM_ARCSEC = 36.3

#: Below this Planck-arm column, the two arms' disagreement is dominated
#: by Herschel's additive zero point rather than the tau353 calibration's
#: own multiplicative residual: the column-check table (page 5's read of
#: `column-check_adopted_survey.hdf5`) shows the Planck/Herschel ratio
#: departing from unity mostly below A_K ~ 0.3 and sitting within 1-5% of
#: unity from 0.19 to 0.50, so a cut here isolates the offset the way the
#: measurement is meant to.
FIELD_ZP_LOW_COLUMN_CUT_AK = 0.3

#: SPEC_PRIORS.md 1.1's sightline product: a pixel is Herschel-covered
#: when at least half its native HGBS block samples are finite -- looser
#: than the calibration's own MIN_FILL=0.98, which selects pixels clean
#: enough to fit the tau353-to-A_K coefficient, not just covered ones.
SIGHTLINE_MIN_FILL = 0.5

#: Lombardi, Bouy, Alves & Lada (2014, A&A 566, A45)'s unsaturated range:
#: below this K-band extinction, neither the NICEST star-colour map nor
#: the dust-emission map has begun to saturate, so a cell's two means are
#: compared on equal footing there.
UNSATURATED_A_K_MAG = 1.0


def _dec(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def _load_herschel_arm(config, region):
    """`{covered, a_k, sigma_rand_k, map_id, map_names}` for `region`, in
    catalogue row order. `sigma_rand_k` is the per-map random term alone
    (`herschel_column.py`'s `SIGMA_RAND_K`, no zero point in it) -- the
    field zero point is composed in fresh here, per field, not read from
    this file's own (pre-refit) `SIGMA_A_K`."""
    path = config_module.product_path(config, "sky/derived", "herschel", "column", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"sky.derived.column.build: Herschel arm missing for region {region!r} at "
            f"{path!r} -- run the sesnaimpute.sky.derived.herschel_column RUNBOOK line for it"
        )
    with h5py.File(path, "r") as f:
        return dict(
            covered=np.asarray(f["COVERED"][:], dtype=bool),
            a_k=np.asarray(f["A_K"][:], dtype=np.float64),
            sigma_rand_k=np.asarray(f["SIGMA_RAND_K"][:], dtype=np.float64),
            map_id=np.asarray(f["MAP_ID"][:], dtype=np.int32),
            map_names=np.asarray(f["MAP_NAME"][:]),
        )


def _load_field_zeropoints(config):
    """`{field: (zp_field, zp_sigma_field)}` from `herschel_column.py`'s
    `sigma`/`survey` product (owner, 2026-09-06) -- empty if the product
    predates the per-field measurement, so every region falls back to no
    offset and `ZP_SIGMA_K = 0`, exactly the old behaviour."""
    path = config_module.product_path(config, "sky/derived", "herschel", "sigma", "survey")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "sky.derived.column._load_field_zeropoints: %r missing -- run the "
            "sesnaimpute.sky.derived.herschel_column RUNBOOK line for it" % path)
    with h5py.File(path, "r") as f:
        if "FIELD_NAME" not in f:
            return {}
        names = [_dec(x) for x in f["FIELD_NAME"][:]]
        zp = np.asarray(f["ZP_FIELD"][:], dtype=np.float64)
        zp_sigma = np.asarray(f["ZP_SIGMA_FIELD"][:], dtype=np.float64)
    return {n: (float(z), float(s)) for n, z, s in zip(names, zp, zp_sigma)}


def _robust_sigma(x):
    """1.4826 * MAD: the same robust scatter estimator
    `herschel_column.py`'s own map-pair replicate uses for its random
    term."""
    med = np.median(x)
    return float(1.482602218505602 * np.median(np.abs(x - med)))


def _measure_field_zeropoints(config):
    """The Herschel zero point, one number per field (owner, 2026-09-06):
    for every HGBS-covered SESNA region, the median of
    `A_HERSCHEL - A_PLANCK` over that region's Herschel-covered admitted
    sightline pixels with `A_PLANCK < FIELD_ZP_LOW_COLUMN_CUT_AK`, and its
    uncertainty as that offset's own scatter (`_robust_sigma` of the same
    per-pixel differences) divided by sqrt(n). This is the arm-against-arm
    comparison at sightline granule (`planck_column.load_hgbs`, the
    granule map's admitted sightlines, `_load_planck_sightline`) -- none
    of it is this stage's own output, so it runs in the same pass that
    merges the two arms, before the per-source merge (`merge_region`)
    consumes the result. Returns `(names, zp_ak, zp_sigma_ak, n)`, one
    entry per field with at least 2 qualifying pixels, region-name
    order."""
    herschel = planck_column.load_hgbs(config, min_fill=SIGHTLINE_MIN_FILL)
    covered_regions = sorted(r for r, h in herschel.items() if h["pix"].size)
    planck_pix, planck_ak, _ = _load_planck_sightline(config)

    names, zp, zp_sigma, n_out = [], [], [], []
    for region in covered_regions:
        admitted_pix, _ = profile_module._admitted_sightlines(config, region)
        _, pix, a_h = _herschel_on_admitted(admitted_pix, herschel[region])
        if pix.size == 0:
            continue
        loc = np.searchsorted(planck_pix, pix)
        a_p = planck_ak[loc]
        sel = a_p < FIELD_ZP_LOW_COLUMN_CUT_AK
        n = int(np.count_nonzero(sel))
        if n < 2:
            continue
        d = a_h[sel] - a_p[sel]
        names.append(region)
        zp.append(float(np.median(d)))
        zp_sigma.append(_robust_sigma(d) / np.sqrt(n))
        n_out.append(n)
    return (names, np.asarray(zp, dtype=np.float64),
            np.asarray(zp_sigma, dtype=np.float64), np.asarray(n_out, dtype=np.int64))


def _write_field_zeropoints(config, names, zp, zp_sigma, n_field):
    """Writes this pass's field zero point (`_measure_field_zeropoints`)
    into `herschel`/`sigma`/`survey`'s `FIELD_NAME`/`ZP_FIELD`/
    `ZP_SIGMA_FIELD`/`ZP_N_FIELD`, and `SIGMA_ZP_K` as the RMS of the
    per-field values, for a reader that still wants one number
    (`population.kernel._load_sigma_zp_herschel`'s fallback).
    `BEAM_FWHM_ARCSEC`/`C0`/`C1`/`N_PAIRS`/`MAP_NAME` --
    `herschel_column.build`'s own -- are left untouched. The file must
    already exist (that RUNBOOK line)."""
    out_path = config_module.product_path(config, "sky/derived", "herschel", "sigma", "survey")
    if not os.path.exists(out_path):
        raise FileNotFoundError(
            "sky.derived.column._write_field_zeropoints: %r missing -- run the "
            "sesnaimpute.sky.derived.herschel_column RUNBOOK line for it" % out_path)
    sigma_zp_survey = float(np.sqrt(np.mean(zp ** 2))) if zp.size else 0.0
    with h5py.File(out_path, "a") as f:
        for ds in ("FIELD_NAME", "ZP_FIELD", "ZP_SIGMA_FIELD", "ZP_N_FIELD", "SIGMA_ZP_K"):
            if ds in f:
                del f[ds]
        for name, data in (
            ("FIELD_NAME", np.array([n.encode("utf-8") for n in names])),
            ("ZP_FIELD", zp),
            ("ZP_SIGMA_FIELD", zp_sigma),
            ("ZP_N_FIELD", n_field),
            ("SIGMA_ZP_K", np.float64(sigma_zp_survey)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_HERSCHEL_SIGMA_STEM, name)])
    print("column.build FIELD_ZP n_fields=%d cut=A_PLANCK<%.2f sigma_zp_survey_rms=%.4f"
          % (len(names), FIELD_ZP_LOW_COLUMN_CUT_AK, sigma_zp_survey), flush=True)
    for name, z, zs, n in zip(names, zp, zp_sigma, n_field):
        print("  field %-22s n=%5d  ZP_FIELD=%+.4f  ZP_SIGMA_FIELD=%.4f" % (name, n, z, zs), flush=True)
    return sigma_zp_survey


def merge_region(config, region, cal, field_zp=None):
    """Reads this region's two arms and merges them: Herschel where
    covered, finite and positive (its own field's zero point subtracted,
    the offset's uncertainty carried as `ZP_SIGMA_K`); else Planck.
    Returns the merged arrays. `field_zp` is `_load_field_zeropoints`'s
    dict, loaded once by `build()`; `merge_region(config, region, cal)`
    still works standalone, loading it itself."""
    if field_zp is None:
        field_zp = _load_field_zeropoints(config)
    planck_path = config_module.product_path(config, "sky/derived", "planck", "column", "source", region=region)
    if not os.path.exists(planck_path):
        raise FileNotFoundError(
            f"sky.derived.column.build: Planck arm missing for region {region!r} at "
            f"{planck_path!r} -- run the Planck source-column RUNBOOK line for it"
        )
    with h5py.File(planck_path, "r") as f:
        a_col = np.asarray(f["A_K"][:], dtype=np.float64)
        sig_col = np.asarray(f["SIGMA_A_K"][:], dtype=np.float64)

    n = a_col.size
    fwhm = np.full(n, cal["fwhm_arcmin"] * 60.0, dtype=np.float32)
    prov = np.full(n, PROV_PLANCK, dtype=np.uint8)
    map_id = np.full(n, -1, dtype=np.int32)
    zp_sigma_k = np.zeros(n, dtype=np.float64)

    herschel = _load_herschel_arm(config, region)
    covered, a_h_raw, sig_rand = herschel["covered"], herschel["a_k"], herschel["sigma_rand_k"]
    zp_offset, zp_sigma = field_zp.get(region, (0.0, 0.0))
    a_h = a_h_raw - zp_offset
    use_h = covered & np.isfinite(a_h) & (a_h > 0)
    a_col[use_h] = a_h[use_h]
    sig_col[use_h] = np.sqrt(sig_rand[use_h] ** 2 + zp_sigma ** 2)
    zp_sigma_k[use_h] = zp_sigma
    fwhm[use_h] = HERSCHEL_STATED_FWHM_ARCSEC
    prov[use_h] = PROV_HERSCHEL
    map_id[use_h] = herschel["map_id"][use_h]
    map_names = herschel["map_names"]

    bad = ~np.isfinite(a_col) | (a_col <= 0)
    if np.any(bad):
        raise ValueError(
            f"sky.derived.column.build: {region!r} has {int(np.count_nonzero(bad))} source(s) "
            "with a non-finite or non-positive merged column"
        )

    return dict(a_col=a_col, sig_col=sig_col, prov=prov, fwhm=fwhm, map_id=map_id, map_names=map_names,
               zp_sigma_k=zp_sigma_k, n_herschel=int(np.count_nonzero(prov == PROV_HERSCHEL)), n=n)


def _build_one_region(config, region, cal, field_zp=None):
    d = merge_region(config, region, cal, field_zp=field_zp)
    out_path = config_module.product_path(config, "sky/derived", "adopted", "column", "source", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    name_bytes = np.array([(x if isinstance(x, bytes) else str(x).encode("utf-8")) for x in d["map_names"]])
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        for name, data in (
            ("A_COL_K", d["a_col"].astype(np.float32)),
            ("A_COL_SIG_K", d["sig_col"].astype(np.float32)),
            ("A_COL_PROVENANCE", d["prov"]),
            ("A_COL_FWHM_ARCSEC", d["fwhm"]),
            ("HERSCHEL_MAP_ID", d["map_id"]),
            ("MAP_NAME", name_bytes),
            ("ZP_SIGMA_K", d["zp_sigma_k"].astype(np.float32)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_COLUMN_SOURCE_STEM, name)])
    return region, d["n"], d["n_herschel"]


def _load_juvela_source_view(config, region):
    """The per-source NICEST view: `A_K` (2MASS star-colour extinction,
    the adopted law's own `A_J/A_K`) and each source's nside-1024 cell,
    in the adopted source product's own row order."""
    path = config_module.product_path(config, "sky/derived", "juvela", "extinction", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"sky.derived.column.build: NICEST source view missing for region {region!r} at "
            f"{path!r} -- run the sesnaimpute.sky.derived.juvela_extinction RUNBOOK line for it"
        )
    with h5py.File(path, "r") as f:
        return np.asarray(f["A_K"][:], dtype=np.float64), np.asarray(f["HPX_PIX_1024"][:], dtype=np.int64)


def _region_calibration(a_nicest, a_adopted, cell_pix):
    """This region's one Lombardi et al. (2014) calibration: bin both
    `a_nicest` (NICEST) and `a_adopted` (the gas column) into their
    nside-1024 cell means, keep the cells where both means read below
    `UNSATURATED_A_K_MAG`, and fit `mean_nicest = offset + slope *
    mean_adopted` by ordinary least squares over those cells alone.
    Returns `(offset, slope, rms, n_fit)`; `rms` is the fit's own
    residual scatter on the cells it was fit to, the scale the two
    regimes (region-wide, protostar positions) are checked to agree
    within."""
    uniq, inv, counts = np.unique(cell_pix, return_inverse=True, return_counts=True)
    mean_nicest = np.bincount(inv, weights=a_nicest, minlength=uniq.size) / counts
    mean_adopted = np.bincount(inv, weights=a_adopted, minlength=uniq.size) / counts
    unsaturated = (mean_nicest < UNSATURATED_A_K_MAG) & (mean_adopted < UNSATURATED_A_K_MAG)
    n_fit = int(np.count_nonzero(unsaturated))
    if n_fit < 2:
        raise ValueError("sky.derived.column._region_calibration: only %d unsaturated nside-1024 "
                          "cell(s) (below %.2f mag on both tracers), too few to fit a line"
                          % (n_fit, UNSATURATED_A_K_MAG))
    slope, offset = np.polyfit(mean_adopted[unsaturated], mean_nicest[unsaturated], 1)
    resid = mean_nicest[unsaturated] - (offset + slope * mean_adopted[unsaturated])
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return float(offset), float(slope), rms, n_fit


def _build_one_extinction_region(config, region, cal, field_zp=None):
    """Writes this region's extinction source product: the adopted column
    (re-merged from the same two arms `merge_region` reads), converted
    onto the NICEST star-colour map's scale by this region's one
    Lombardi et al. (2014) calibration (`_region_calibration`), plus the
    fitted `CAL_OFFSET_K`/`CAL_SLOPE` pair itself."""
    d = merge_region(config, region, cal, field_zp=field_zp)
    a_nicest, cell_pix = _load_juvela_source_view(config, region)
    if a_nicest.size != d["n"]:
        raise ValueError(f"sky.derived.column.build: {region!r} NICEST view has {a_nicest.size} "
                         f"source(s), the adopted column has {d['n']}")
    offset, slope, rms, n_fit = _region_calibration(a_nicest, d["a_col"], cell_pix)
    a_col_cal = offset + slope * d["a_col"]
    bad = ~np.isfinite(a_col_cal) | (a_col_cal <= 0)
    if np.any(bad):
        raise ValueError(f"sky.derived.column.build: {region!r} has {int(np.count_nonzero(bad))} "
                         "source(s) with a non-finite or non-positive calibrated extinction column "
                         f"(offset={offset:.4f}, slope={slope:.4f})")

    out_path = config_module.product_path(config, "sky/derived", "adopted", "extinction", "source", region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    name_bytes = np.array([(x if isinstance(x, bytes) else str(x).encode("utf-8")) for x in d["map_names"]])
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        for name, data in (
            ("A_COL_K", a_col_cal.astype(np.float32)),
            ("A_COL_SIG_K", (d["sig_col"] * abs(slope)).astype(np.float32)),
            ("A_COL_PROVENANCE", d["prov"]),
            ("A_COL_FWHM_ARCSEC", d["fwhm"]),
            ("HERSCHEL_MAP_ID", d["map_id"]),
            ("MAP_NAME", name_bytes),
            ("ZP_SIGMA_K", d["zp_sigma_k"].astype(np.float32)),
            ("CAL_OFFSET_K", np.float64(offset)),
            ("CAL_SLOPE", np.float64(slope)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_EXTINCTION_SOURCE_STEM, name)])
    return region, d["n"], offset, slope, rms, n_fit


def _extinction_row_one_region(config, region, region_code, adopted_pix, adopted_a_k, adopted_sig, adopted_prov):
    """One region's extinction sightline row: the adopted sightline column
    (`build_sightline`'s own Herschel/Planck value, unchanged) converted
    onto the NICEST star-colour map's scale by this region's one
    `CAL_OFFSET_K`/`CAL_SLOPE` pair (read from the region's own extinction
    source product, where `_build_one_extinction_region` fit and stored
    it) -- the same affine transform for every admitted sightline of the
    region, whether or not it carries a catalogued source.
    `PROVENANCE`/`REGION_CODE` are carried from the adopted sightline
    column product."""
    ext_path = config_module.product_path(config, "sky/derived", "adopted", "extinction", "source", region=region)
    if not os.path.exists(ext_path):
        raise FileNotFoundError(
            f"sky.derived.column.build_extinction_sightline: extinction source product missing for region "
            f"{region!r} at {ext_path!r} -- run the sesnaimpute.sky.derived.column RUNBOOK line for it"
        )
    with h5py.File(ext_path, "r") as f:
        offset = float(f["CAL_OFFSET_K"][()])
        slope = float(f["CAL_SLOPE"][()])

    admitted_pix, _ = profile_module._admitted_sightlines(config, region)
    loc = np.searchsorted(adopted_pix, admitted_pix)
    capped = np.minimum(loc, adopted_pix.size - 1) if adopted_pix.size else loc
    ok = adopted_pix.size and np.all(adopted_pix[capped] == admitted_pix)
    if not ok:
        raise ValueError(f"sky.derived.column.build_extinction_sightline: {region!r} missing from the "
                         "adopted sightline column product")

    return dict(pix=admitted_pix, region_code=np.full(admitted_pix.size, region_code, dtype=np.int16),
               a_k=offset + slope * adopted_a_k[capped], sig_a_k=abs(slope) * adopted_sig[capped],
               prov=adopted_prov[capped])


def build_extinction_sightline(config, stage=None):
    """Writes the extinction sightline column: one
    row per admitted nside-256 pixel of every region, the adopted
    sightline column converted onto the NICEST star-colour map's scale by
    that region's own `CAL_OFFSET_K`/`CAL_SLOPE` pair -- the quantity a
    star's light passes through, distinct from the gas column
    `build_sightline` writes. Survey-wide regardless of any `regions`
    list a caller passed to `build`, mirroring `build_sightline`."""
    regions = [r.name for r in regions_module.REGIONS]
    codes = _region_codes(config, regions)
    adopted_path = config_module.product_path(config, "sky/derived", "adopted", "column", "sightline")
    if not os.path.exists(adopted_path):
        raise FileNotFoundError(
            "sky.derived.column.build_extinction_sightline: adopted sightline column missing at "
            "%r -- run sky.derived.column.build_sightline first (this module's own RUNBOOK line)" % adopted_path)
    with h5py.File(adopted_path, "r") as f:
        adopted_pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        adopted_prov = np.asarray(f["PROVENANCE"][:])
        adopted_a_k = np.asarray(f["A_K"][:], dtype=np.float64)
        adopted_sig = np.asarray(f["SIGMA_A_K"][:], dtype=np.float64)
    order = np.argsort(adopted_pix)
    adopted_pix, adopted_prov = adopted_pix[order], adopted_prov[order]
    adopted_a_k, adopted_sig = adopted_a_k[order], adopted_sig[order]

    n_regions = len(regions)
    chunk = max(1, config.n_jobs)
    rows = []
    for start in range(0, n_regions, chunk):
        part = regions[start:start + chunk]
        rows.extend(Parallel(n_jobs=config.n_jobs)(
            delayed(_extinction_row_one_region)(config, region, codes[region],
                                                adopted_pix, adopted_a_k, adopted_sig, adopted_prov)
            for region in part))
        if stage is not None:
            stage.tick(min(start + chunk, n_regions), n_regions, "regions")

    pix = np.concatenate([r["pix"] for r in rows])
    region_code = np.concatenate([r["region_code"] for r in rows])
    a_k = np.concatenate([r["a_k"] for r in rows])
    sig = np.concatenate([r["sig_a_k"] for r in rows])
    prov_out = np.concatenate([r["prov"] for r in rows])

    out_path = config_module.product_path(config, "sky/derived", "adopted", "extinction", "sightline")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as fh:
        fh.attrs["GRANULE"] = "sightline"
        for name, data in (
            ("HPX_PIX_256", pix),
            ("REGION_CODE", region_code),
            ("A_K", a_k.astype(np.float32)),
            ("SIGMA_A_K", sig.astype(np.float32)),
            ("PROVENANCE", prov_out),
        ):
            build_module.write_dataset(fh, name, data, *REGISTRY[(_EXTINCTION_SIGHTLINE_STEM, name)])

    n = int(pix.size)
    print("column.build_extinction_sightline: %d sightline rows, %d regions"
          % (n, len(regions)), flush=True)
    return out_path


def _region_codes(config, regions):
    """{region: REGION_CODE} read from the granule map, the same codes
    every other product keys a region by."""
    path = config_module.product_path(config, "granules", "sesna", "granule-map", "source")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "sky.derived.column._region_codes: granule map missing at %r -- run the "
            "sesnaimpute.granules.build RUNBOOK line for it" % path)
    with h5py.File(path, "r") as f:
        names = [_dec(v) for v in f["region/REGION"][:]]
        codes = np.asarray(f["region/REGION_CODE"][:], dtype=np.int16)
    by_name = dict(zip(names, codes.tolist()))
    missing = [r for r in regions if r not in by_name]
    if missing:
        raise ValueError(f"sky.derived.column: region(s) {missing} absent from granule map {path!r}")
    return by_name


def _herschel_sigma_model(config):
    """The per-map sigma model `herschel_column.py` fits over overlapping-
    map replicates: `sig(A_K) = sqrt(SIGMA_ZP_K**2 + C0**2 + (C1*A_K)**2)`."""
    path = config_module.product_path(config, "sky/derived", "herschel", "sigma", "survey")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "sky.derived.column.build_sightline: Herschel sigma model missing at "
            f"{path!r} -- run the sesnaimpute.sky.derived.herschel_column RUNBOOK line for it"
        )
    with h5py.File(path, "r") as f:
        return float(f["SIGMA_ZP_K"][()]), float(f["C0"][()]), float(f["C1"][()])


def _load_planck_sightline(config):
    """`(pix, a_k, sigma_a_k)`, sorted by pixel, from `planck_column.py`'s
    survey-wide sightline product."""
    path = config_module.product_path(config, "sky/derived", "planck", "column", "sightline")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "sky.derived.column.build_sightline: Planck sightline column missing at "
            f"{path!r} -- run the sesnaimpute.sky.derived.planck_column RUNBOOK line for it"
        )
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        a_k = np.asarray(f["A_K"][:], dtype=np.float64)
        sig = np.asarray(f["SIGMA_A_K"][:], dtype=np.float64)
    order = np.argsort(pix)
    return pix[order], a_k[order], sig[order]


def _herschel_on_admitted(admitted_pix, herschel_region):
    """The region's Herschel-covered pixels that are also admitted
    sightlines: `(position_in_admitted_pix, pix, a_k)`."""
    empty_i = np.empty(0, dtype=np.int64)
    if herschel_region is None or herschel_region["pix"].size == 0 or admitted_pix.size == 0:
        return empty_i, empty_i, np.empty(0, dtype=np.float64)
    h_pix = herschel_region["pix"]
    loc = np.searchsorted(admitted_pix, h_pix)
    capped = np.minimum(loc, admitted_pix.size - 1)
    ok = admitted_pix[capped] == h_pix
    return capped[ok], h_pix[ok], herschel_region["ak"][ok]


def _sightline_row_one_region(config, region, region_code, herschel, planck_pix, planck_ak, planck_sig,
                              sig_zp, c0, c1):
    """One region's sightline rows: Planck everywhere admitted, Herschel
    substituted where the pixel is covered (SPEC_PRIORS.md 1.1)."""
    admitted_pix, _ = profile_module._admitted_sightlines(config, region)
    n = admitted_pix.size
    loc = np.searchsorted(planck_pix, admitted_pix)
    capped = np.minimum(loc, planck_pix.size - 1) if planck_pix.size else loc
    ok = planck_pix.size and np.all(planck_pix[capped] == admitted_pix)
    if not ok:
        missing = admitted_pix[planck_pix[capped] != admitted_pix][:5] if planck_pix.size else admitted_pix[:5]
        raise ValueError("sky.derived.column.build_sightline: region %r has no Planck sightline "
                         "column for pixel(s) %s" % (region, missing.tolist()))
    a_col = planck_ak[capped].copy()
    sig_col = planck_sig[capped].copy()
    prov = np.full(n, PROV_PLANCK, dtype=np.uint8)

    idx, _, a_h = _herschel_on_admitted(admitted_pix, herschel.get(region))
    if idx.size:
        a_col[idx] = a_h
        sig_col[idx] = np.sqrt(sig_zp ** 2 + c0 ** 2 + (c1 * a_h) ** 2)
        prov[idx] = PROV_HERSCHEL

    return dict(pix=admitted_pix, region_code=np.full(n, region_code, dtype=np.int16),
               a_col=a_col, sig_col=sig_col, prov=prov)


def build_sightline(config, stage=None):
    """Writes the adopted sightline column (SPEC_PRIORS.md 1.1, 1.4): one
    row per admitted nside-256 pixel of every region, Herschel where
    covered, else Planck. Survey-wide regardless of any `regions` list a
    caller passed to `build` -- `profile.py` looks up whatever pixel it is
    currently building, of whatever region, so the product cannot be
    partial.

    `stage`, if given, is the caller's own `progress.Stage`: regions are
    dispatched in chunks (still `config.n_jobs`-wide within a chunk) so
    one `stage.tick` fires after each chunk -- this stage otherwise runs
    to completion with no progress line at all (owner ruling
    2026-09-06)."""
    regions = [r.name for r in regions_module.REGIONS]
    herschel = planck_column.load_hgbs(config, min_fill=SIGHTLINE_MIN_FILL)
    sig_zp, c0, c1 = _herschel_sigma_model(config)
    planck_pix, planck_ak, planck_sig = _load_planck_sightline(config)
    codes = _region_codes(config, regions)

    n_regions = len(regions)
    chunk = max(1, config.n_jobs)
    rows = []
    for start in range(0, n_regions, chunk):
        part = regions[start:start + chunk]
        rows.extend(Parallel(n_jobs=config.n_jobs)(
            delayed(_sightline_row_one_region)(
                config, region, codes[region], herschel, planck_pix, planck_ak, planck_sig, sig_zp, c0, c1)
            for region in part))
        if stage is not None:
            stage.tick(min(start + chunk, n_regions), n_regions, "regions")

    pix = np.concatenate([r["pix"] for r in rows])
    region_code = np.concatenate([r["region_code"] for r in rows])
    a_col = np.concatenate([r["a_col"] for r in rows])
    sig_col = np.concatenate([r["sig_col"] for r in rows])
    prov = np.concatenate([r["prov"] for r in rows])

    out_path = config_module.product_path(config, "sky/derived", "adopted", "column", "sightline")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "sightline"
        for name, data in (
            ("HPX_PIX_256", pix),
            ("REGION_CODE", region_code),
            ("A_K", a_col.astype(np.float32)),
            ("SIGMA_A_K", sig_col.astype(np.float32)),
            ("PROVENANCE", prov),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_COLUMN_SIGHTLINE_STEM, name)])

    n = int(pix.size)
    n_h = int(np.count_nonzero(prov == PROV_HERSCHEL))
    print("column.build_sightline: %d sightline rows, %d regions, %d Herschel (%.1f%%)"
          % (n, len(regions), n_h, 100.0 * n_h / n if n else 0.0), flush=True)
    return out_path


def _map_edge_for_pixels(config, region, pixels, input_dir):
    """`A_MAP_EDGE`: profile.py's own inner/outer/splice/measure chain for
    `pixels` of `region`, called exactly as `profile._build_one_region`
    does, before the sightline-column rescaling step. `a_out`'s last
    column does not depend on the per-pixel weight `measure_region` also
    computes (it only shapes the unused structure-depth diagnostic), so a
    uniform weight is passed rather than the region's true source counts."""
    canon = regions_module.REGIONS_BY_NAME[region]
    inner = profile_module.read_inner_map(f"{input_dir}/{profile_module.MAP_INNER_NAME}",
                                          pixels, profile_module.ZGR23_R_KS)
    splice = profile_module.read_outer_increment(f"{input_dir}/{profile_module.MAP_OUTER_NAME}",
                                                  pixels, profile_module.ZGR23_R_KS)
    dist_new, rho_dist_new = profile_module.splice_axes(
        inner["dist1"], inner["radii1"], splice["bnd2"], splice["cen2"], splice["k"])
    weight = profile_module.region_weight(np.ones(pixels.size))
    measured = profile_module.measure_region(canon.d_r_pc, inner, splice, dist_new, rho_dist_new, weight)
    return measured["a_out"][:, -1]


def _binned_stats(x, group, n_groups):
    """Per-group median, 16th and 84th percentile of `x`; NaN for an
    empty group."""
    med = np.full(n_groups, np.nan)
    lo = np.full(n_groups, np.nan)
    hi = np.full(n_groups, np.nan)
    for g in range(n_groups):
        sel = group == g
        if sel.any():
            lo[g], med[g], hi[g] = np.percentile(x[sel], [16, 50, 84])
    return med, lo, hi


def build_column_check(config, stage=None):
    """Writes and prints SPEC_PRIORS.md 1.1/1.4's tracer-disagreement
    check: on every Herschel-covered sightline pixel, A_HERSCHEL,
    A_PLANCK, and A_MAP_EDGE (the 3-D map's own cumulative extinction at
    its edge, unrescaled), and the two ratios A_MAP_EDGE/A_HERSCHEL and
    A_PLANCK/A_HERSCHEL, binned by quartile of A_HERSCHEL and by region --
    whether the Planck calibration runs low on diffuse sightlines, the 3-D
    map runs high, or both.

    `stage`, if given, is the caller's own `progress.Stage`: one
    `stage.tick` per region of this function's own sequential loop
    (owner ruling 2026-09-06 -- this stage otherwise runs to completion
    with no progress line at all)."""
    herschel = planck_column.load_hgbs(config, min_fill=SIGHTLINE_MIN_FILL)
    covered_regions = sorted(r for r, h in herschel.items() if h["pix"].size)
    planck_pix, planck_ak, _ = _load_planck_sightline(config)
    codes = _region_codes(config, covered_regions)
    input_dir = profile_module._input_dir(config)

    per_region = []
    n_covered = len(covered_regions)
    for i, region in enumerate(covered_regions):
        admitted_pix, _ = profile_module._admitted_sightlines(config, region)
        _, pix, a_h = _herschel_on_admitted(admitted_pix, herschel[region])
        if pix.size == 0:
            if stage is not None:
                stage.tick(i + 1, n_covered, "regions")
            continue
        loc = np.searchsorted(planck_pix, pix)
        a_p = planck_ak[loc]
        a_edge = _map_edge_for_pixels(config, region, pix, input_dir)
        per_region.append(dict(region=region, code=codes[region], pix=pix,
                               a_herschel=a_h, a_planck=a_p, a_edge=a_edge))
        if stage is not None:
            stage.tick(i + 1, n_covered, "regions")

    if not per_region:
        raise ValueError("sky.derived.column.build_column_check: no Herschel-covered "
                         "admitted sightline pixel found in any region")

    pix = np.concatenate([r["pix"] for r in per_region])
    region_code = np.concatenate([np.full(r["pix"].size, r["code"], dtype=np.int16) for r in per_region])
    a_herschel = np.concatenate([r["a_herschel"] for r in per_region])
    a_planck = np.concatenate([r["a_planck"] for r in per_region])
    a_edge = np.concatenate([r["a_edge"] for r in per_region])
    ratio_map = a_edge / a_herschel
    ratio_planck = a_planck / a_herschel

    q_edges = np.percentile(a_herschel, [0, 25, 50, 75, 100])
    bin_idx = np.clip(np.searchsorted(q_edges, a_herschel, side="right") - 1, 0, 3)
    bin_map_med, bin_map_lo, bin_map_hi = _binned_stats(ratio_map, bin_idx, 4)
    bin_planck_med, bin_planck_lo, bin_planck_hi = _binned_stats(ratio_planck, bin_idx, 4)

    # np.unique's own inverse index, not a searchsorted against per_region's
    # (alphabetical, not code-sorted) order, so a region's rows land in the
    # right group regardless of how covered_regions was ordered above.
    region_code_axis, region_pos = np.unique(region_code, return_inverse=True)
    code_to_name = {v: k for k, v in codes.items()}
    region_names = np.array([code_to_name[int(c)] for c in region_code_axis], dtype="S64")
    reg_map_med, reg_map_lo, reg_map_hi = _binned_stats(ratio_map, region_pos, region_code_axis.size)
    reg_planck_med, reg_planck_lo, reg_planck_hi = _binned_stats(ratio_planck, region_pos, region_code_axis.size)

    print("column.build_column_check: %d Herschel-covered sightline pixels over %d regions"
          % (pix.size, region_code_axis.size), flush=True)
    for g in range(4):
        print("  A_HERSCHEL bin %d [%.3f, %.3f): map_edge/A_HERSCHEL med=%.3f [%.3f, %.3f]  "
              "planck/A_HERSCHEL med=%.3f [%.3f, %.3f]"
              % (g, q_edges[g], q_edges[g + 1], bin_map_med[g], bin_map_lo[g], bin_map_hi[g],
                 bin_planck_med[g], bin_planck_lo[g], bin_planck_hi[g]), flush=True)
    for i, region in enumerate(region_names):
        print("  region %-20s map_edge/A_HERSCHEL med=%.3f [%.3f, %.3f]  "
              "planck/A_HERSCHEL med=%.3f [%.3f, %.3f]"
              % (_dec(region), reg_map_med[i], reg_map_lo[i], reg_map_hi[i],
                 reg_planck_med[i], reg_planck_lo[i], reg_planck_hi[i]), flush=True)

    out_path = config_module.product_path(config, "sky/derived", "adopted", "column-check", "survey")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "survey"
        for name, data in (
            ("HPX_PIX_256", pix),
            ("REGION_CODE", region_code),
            ("A_HERSCHEL", a_herschel.astype(np.float32)),
            ("A_PLANCK", a_planck.astype(np.float32)),
            ("A_MAP_EDGE", a_edge.astype(np.float32)),
            ("A_HERSCHEL_QUARTILE_EDGES", q_edges),
            ("BIN_MAP_EDGE_RATIO_MEDIAN", bin_map_med),
            ("BIN_MAP_EDGE_RATIO_P16", bin_map_lo),
            ("BIN_MAP_EDGE_RATIO_P84", bin_map_hi),
            ("BIN_PLANCK_RATIO_MEDIAN", bin_planck_med),
            ("BIN_PLANCK_RATIO_P16", bin_planck_lo),
            ("BIN_PLANCK_RATIO_P84", bin_planck_hi),
            ("REGION", region_names),
            ("REGION_CODE_AXIS", region_code_axis),
            ("REGION_MAP_EDGE_RATIO_MEDIAN", reg_map_med),
            ("REGION_MAP_EDGE_RATIO_P16", reg_map_lo),
            ("REGION_MAP_EDGE_RATIO_P84", reg_map_hi),
            ("REGION_PLANCK_RATIO_MEDIAN", reg_planck_med),
            ("REGION_PLANCK_RATIO_P16", reg_planck_lo),
            ("REGION_PLANCK_RATIO_P84", reg_planck_hi),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_CHECK_STEM, name)])
    return out_path


def build(config, regions=None):
    """Builds the adopted source-column product for each region in
    `regions` (default: every region in `regions.REGIONS`), merging the
    Planck and Herschel arms one region at a time, parallelised with
    joblib; then the adopted sightline column and its Herschel/Planck/
    3-D-map check, both survey-wide. The field zero point
    (`_measure_field_zeropoints`) is measured and written
    (`_write_field_zeropoints`) first, in this same pass, from the two
    arms' sightline views alone -- not read back from a second product --
    so a cold build (no prior run of this stage) still applies it."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    cal = _load_planck_calibration(config)
    zp_names, zp_ak, zp_sigma_ak, zp_n = _measure_field_zeropoints(config)
    _write_field_zeropoints(config, zp_names, zp_ak, zp_sigma_ak, zp_n)
    field_zp = {n: (float(z), float(s)) for n, z, s in zip(zp_names, zp_ak, zp_sigma_ak)}
    with progress_module.Stage("sky.derived.column") as st:
        n_regions = len(regions)
        n_done = [0]

        def _one(region):
            r = _build_one_region(config, region, cal, field_zp=field_zp)
            n_done[0] += 1
            st.tick(n_done[0], n_regions, "regions")
            return r

        results = Parallel(n_jobs=config.n_jobs)(delayed(_one)(region) for region in regions)
        total_n = sum(r[1] for r in results)
        total_herschel = sum(r[2] for r in results)
        st.done(None, regions=n_regions, sources=total_n, herschel_sources=total_herschel)

    with progress_module.Stage("sky.derived.column.extinction") as st:
        n_done_ext = [0]

        def _one_ext(region):
            r = _build_one_extinction_region(config, region, cal, field_zp=field_zp)
            n_done_ext[0] += 1
            st.tick(n_done_ext[0], n_regions, "regions")
            return r

        ext_results = Parallel(n_jobs=config.n_jobs)(delayed(_one_ext)(region) for region in regions)
        st.done(None, regions=n_regions, sources=sum(r[1] for r in ext_results),
                median_offset_k=float(np.median([r[2] for r in ext_results])) if ext_results else float("nan"),
                median_slope=float(np.median([r[3] for r in ext_results])) if ext_results else float("nan"),
                median_fit_rms=float(np.median([r[4] for r in ext_results])) if ext_results else float("nan"))

    with progress_module.Stage("sky.derived.column.sightline") as st:
        out_path = build_sightline(config, stage=st)
        st.done(out_path)

    with progress_module.Stage("sky.derived.column.extinction_sightline") as st:
        out_path = build_extinction_sightline(config, stage=st)
        st.done(out_path)


if __name__ == "__main__":
    run(build)
