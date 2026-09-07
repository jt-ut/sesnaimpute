"""The adopted source column: the finer arm wherever it reaches
(SPEC_PRIORS.md section 1.1).

`A_COL_K = A_HERSCHEL_K - ZP_FIELD[field]` (36.3 arcsec beam) where the
region has a Herschel product and the source is covered, finite and
positive; `A_COL_K = A_K` from the Planck arm (`planck_source_column.py`,
PLANCK_FWHM_ARCMIN beam) elsewhere. A measured systematic left unapplied
is an error of its own size (owner, 2026-09-06): `field` is the source's
own region, one of the 13 named in `herschel_column.py`'s `sigma`/
`survey` `FIELD_NAME`/`ZP_FIELD`/`ZP_SIGMA_FIELD`; a Herschel-covered
region absent from that table (no column-check pixels) gets no offset
and `ZP_SIGMA_K = 0`, unchanged from before. Sigma, the beam and the
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
"""

import os

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.sky.derived import planck_column
from sesnaimpute.sky.derived import profile as profile_module
from sesnaimpute.sky.derived.planck_source_column import _load_planck_calibration

PROV_HERSCHEL = np.uint8(0)
PROV_PLANCK = np.uint8(1)
HERSCHEL_STATED_FWHM_ARCSEC = 36.3

#: SPEC_PRIORS.md 1.1's sightline product: a pixel is Herschel-covered
#: when at least half its native HGBS block samples are finite -- looser
#: than the calibration's own MIN_FILL=0.98, which selects pixels clean
#: enough to fit the tau353-to-A_K coefficient, not just covered ones.
SIGHTLINE_MIN_FILL = 0.5


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
    with h5py.File(path, "r") as f:
        if "FIELD_NAME" not in f:
            return {}
        names = [_dec(x) for x in f["FIELD_NAME"][:]]
        zp = np.asarray(f["ZP_FIELD"][:], dtype=np.float64)
        zp_sigma = np.asarray(f["ZP_SIGMA_FIELD"][:], dtype=np.float64)
    return {n: (float(z), float(s)) for n, z, s in zip(names, zp, zp_sigma)}


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
        f.create_dataset("A_COL_K", data=d["a_col"].astype(np.float32))
        f.create_dataset("A_COL_SIG_K", data=d["sig_col"].astype(np.float32))
        f.create_dataset("A_COL_PROVENANCE", data=d["prov"])
        f.create_dataset("A_COL_FWHM_ARCSEC", data=d["fwhm"])
        f.create_dataset("HERSCHEL_MAP_ID", data=d["map_id"])
        f.create_dataset("MAP_NAME", data=name_bytes)
        f.create_dataset("ZP_SIGMA_K", data=d["zp_sigma_k"].astype(np.float32))
    return region, d["n"], d["n_herschel"]


def _region_codes(config, regions):
    """{region: REGION_CODE} read from the granule map, the same codes
    every other product keys a region by."""
    path = config_module.product_path(config, "granules", "sesna", "granule-map", "source")
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
        f.create_dataset("HPX_PIX_256", data=pix)
        f.create_dataset("REGION_CODE", data=region_code)
        f.create_dataset("A_K", data=a_col.astype(np.float32))
        f.create_dataset("SIGMA_A_K", data=sig_col.astype(np.float32))
        f.create_dataset("PROVENANCE", data=prov)

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
        f.create_dataset("HPX_PIX_256", data=pix)
        f.create_dataset("REGION_CODE", data=region_code)
        f.create_dataset("A_HERSCHEL", data=a_herschel.astype(np.float32))
        f.create_dataset("A_PLANCK", data=a_planck.astype(np.float32))
        f.create_dataset("A_MAP_EDGE", data=a_edge.astype(np.float32))
        f.create_dataset("A_HERSCHEL_QUARTILE_EDGES", data=q_edges)
        f.create_dataset("BIN_MAP_EDGE_RATIO_MEDIAN", data=bin_map_med)
        f.create_dataset("BIN_MAP_EDGE_RATIO_P16", data=bin_map_lo)
        f.create_dataset("BIN_MAP_EDGE_RATIO_P84", data=bin_map_hi)
        f.create_dataset("BIN_PLANCK_RATIO_MEDIAN", data=bin_planck_med)
        f.create_dataset("BIN_PLANCK_RATIO_P16", data=bin_planck_lo)
        f.create_dataset("BIN_PLANCK_RATIO_P84", data=bin_planck_hi)
        f.create_dataset("REGION", data=region_names)
        f.create_dataset("REGION_CODE_AXIS", data=region_code_axis)
        f.create_dataset("REGION_MAP_EDGE_RATIO_MEDIAN", data=reg_map_med)
        f.create_dataset("REGION_MAP_EDGE_RATIO_P16", data=reg_map_lo)
        f.create_dataset("REGION_MAP_EDGE_RATIO_P84", data=reg_map_hi)
        f.create_dataset("REGION_PLANCK_RATIO_MEDIAN", data=reg_planck_med)
        f.create_dataset("REGION_PLANCK_RATIO_P16", data=reg_planck_lo)
        f.create_dataset("REGION_PLANCK_RATIO_P84", data=reg_planck_hi)
    return out_path


def build(config, regions=None):
    """Builds the adopted source-column product for each region in
    `regions` (default: every region in `regions.REGIONS`), merging the
    Planck and Herschel arms one region at a time, parallelised with
    joblib; then the adopted sightline column and its Herschel/Planck/
    3-D-map check, both survey-wide."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    cal = _load_planck_calibration(config)
    field_zp = _load_field_zeropoints(config)
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

    with progress_module.Stage("sky.derived.column.sightline") as st:
        out_path = build_sightline(config, stage=st)
        st.done(out_path)

    with progress_module.Stage("sky.derived.column.column_check") as st:
        out_path = build_column_check(config, stage=st)
        st.done(out_path)


if __name__ == "__main__":
    run(build)
