"""P1, the per-source density table (SPEC_BMSTP_DRAFT.md section 4.1; the
six sky densities of section 5; IMPLEMENTATION_BMSTP_DRAFT.md section 1.2
P1). One row per curated-catalogue source, in catalogue order: the column,
the source's tile and sightline (as row indices into P2/P3's grain axes),
its detection limits, and the six sky densities `A_STAR ... A_H2S`,
objects per square degree.

STAR/AGB (section 5.1, 5.2) are per-tile aggregates of the population's
retained field-star sample, gathered to sources by tile; PAHC (section
5.3) is STAR's density verbatim. GAL (section 5.4) is one survey number,
the counts law integrated over its tabulated grid. YSO (section 5.5) is
the quadratic law applied to the CLOUD's own share of the column,
`A_cloud` (only the sightline's FOREGROUND deducted from the source's
whole adopted column, at the front edge of `bmstp.sample_cloud`'s own
cloud interval: the 3-D map cannot partition the column reliably behind
a cloud at a kiloparsec, so nothing behind the front edge is deducted),
`kappa` selected by which arm reached the source (`ARM`, from the
adopted column's own provenance flag). H2S (section 5.6) rides on that
same INTRINSIC young-star law density, scaled by the region's `eta` and
the universal `eps_ext`, but on the Herschel arm the law itself is the
knot-driver kernel's convolution of the region's HGBS map
(`bmstp.knot_field.convolved_law`), sampled at the source's own
position -- a map operation, once per region, never a per-source
convolution; a Herschel-arm source whose position falls outside the
convolved map, and every Planck-arm source (the kernel is sub-beam at
Planck's 5.03' beam), takes the law at its own column instead. Every
density is a RETAINED density (section 4.1): the intrinsic count times
the population's on-grid fraction at the source's own grain -- STAR/AGB
by tile, YSO and H2S by sightline (both read from P3, the cloud shape
product), GAL the one survey-wide attr, PAHC reading STAR's own
fraction (section 5.5, 5.6): `ON_GRID_H2S` is a real, stored per-sightline
retained fraction on the same common depth axis `ON_GRID_YSO` uses, not
an assumed 1, since the knots' brightness axis moved onto that common
grid, so it is applied exactly like every other class's on-grid
fraction rather than treated as formed at read time.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.batches import batches
from sesnaimpute.build import run
from sesnaimpute.granules import access
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.population import field_stars
from sesnaimpute.population.yso import PROVENANCE_HERSCHEL, law_count, pc2_per_deg2
from sesnaimpute.bmstp import grid
from sesnaimpute.bmstp import knot_field
from sesnaimpute.bmstp import sample_cloud
from sesnaimpute.bmstp import sample_gal

#: Not yet in constants.py -- added here per CODING_RULES_BMSTP.md rule 3,
#: SPEC_BMSTP_DRAFT.md section 10.
#: Young-star law amplitude at the Herschel beam, pc^-2 mag^-2 -- Pokhrel
#: et al. 2020's pooled star-gas relation.
KAPPA_HERSCHEL = 14.5
#: The same law at Planck's beam -- Pokhrel+2020 times Lada et al. 2013's
#: 1.29 beam-ratio correction.
KAPPA_PLANCK = 18.7
#: Knots per law-predicted young star, depth-corrected, by region --
#: Froebrich et al. 2015 (UWISH2), Giannini et al. 2013.
ETA = {
    "Cygnus X": 0.0065,
    "North America Nebula": 0.0394,
    "Vela D": 0.0408,
}
ETA_ELSEWHERE = 0.0401
#: The fraction of knots clearing the limits that the source finder
#: catalogues -- cross-matches against five knot surveys (section 5.6).
EPS_EXT = 0.25

_I4_INDEX = [b.key for b in definitions.BANDS].index("I4")

#: Per-source working-set estimate for the batch loop (rule 10b): the
#: column, limit and index columns this stage reads and writes, well
#: under the batch budget even at 100,000 sources.
ROW_BYTES = 256

#: HEALPix nside=512 pixel area, deg^2 (4*pi steradian = 41252.96 deg^2
#: over the whole sky, 12*nside^2 equal-area pixels) -- used only for the
#: report-only area estimate behind the level check (section 8).
_HPX512_PIXEL_DEG2 = 41252.96 / (12 * 512 ** 2)


def _tile_membership(config, region):
    """`(pix, tile)`, sorted by pixel: the STAR anchor's own nside-512
    pixel-to-tile map (`population/anchors/tiles_anchors_hpx512__R.hdf5`,
    `population.star_population`'s tile definition), the pixel-granule
    counterpart of `granules.access`'s bms-area `_tile_membership` (this
    design keeps its own copy under `population/`, never reading `bms/`)."""
    path = config_module.product_path(config, "population", "anchors", "tiles", "hpx512", region=region)
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        tile = np.asarray(f["TILE_ID"][:], dtype=np.int64)
    order = np.argsort(pix)
    return pix[order], tile[order]


def _resolve_tile_row(config, region, hpx_pix_512, tile_id_axis):
    """Each source's row index into P2's `TILE_ID` axis: the STAR anchor
    tile-membership table gives the source's tile id; that id is then
    located in P2's own tile axis."""
    tile_pix, tile_of_pix = _tile_membership(config, region)
    loc = np.searchsorted(tile_pix, hpx_pix_512)
    capped = np.minimum(loc, tile_pix.size - 1)
    if not np.all(tile_pix[capped] == hpx_pix_512):
        raise ValueError("bmstp.density: source(s) with no tile in the STAR anchor "
                         "tile-membership table")
    source_tile = tile_of_pix[capped]
    row = np.searchsorted(tile_id_axis, source_tile)
    row_capped = np.minimum(row, tile_id_axis.size - 1)
    if not np.all(tile_id_axis[row_capped] == source_tile):
        raise ValueError("bmstp.density: source tile absent from P2's TILE_ID axis")
    return row_capped.astype(np.int32)


def _resolve_sightline_row(hpx_pix_256, sightline_axis):
    """Each source's row index into P3's `HPX_PIX_256` axis."""
    row = np.searchsorted(sightline_axis, hpx_pix_256)
    capped = np.minimum(row, sightline_axis.size - 1)
    if not np.all(sightline_axis[capped] == hpx_pix_256):
        raise ValueError("bmstp.density: source sightline absent from P3's HPX_PIX_256 axis")
    return capped.astype(np.int32)


def _star_family_density_by_tile(config, region, tile_id_axis):
    """`A_STAR(s)`, `A_AGB(s)` per tile, section 5.1/5.2: the retained
    field-star sample's `W_STAR`/`W_AGB` summed over the tile's simulated
    stars, divided by the ONE TRILEGAL pointing that tile's sample is
    drawn from, `OMEGA_POINTING_DEG2` (recorded per tile group on
    `population/star/population_star_tile__R.hdf5`) -- not the region's
    whole-simulation `OMEGA_SIM_DEG2`, which a multi-pointing region's
    per-tile sample does not cover."""
    path = config_module.product_path(
        config, "population", "star", "population", "tile", region=region)
    star = np.empty(tile_id_axis.size, dtype=np.float64)
    agb = np.empty(tile_id_axis.size, dtype=np.float64)
    with h5py.File(path, "r") as f:
        omega_sim = float(f.attrs["OMEGA_SIM_DEG2"])
        f_dusty_o = float(f.attrs["F_DUSTY_O"])
        f_dusty_c = float(f.attrs["F_DUSTY_C"])
        f_c = float(f.attrs["F_C"])
        for i, tile_id in enumerate(tile_id_axis):
            grp = f[f"tile_{int(tile_id)}"]
            omega_t = float(grp.attrs["OMEGA_POINTING_DEG2"])
            star[i] = float(np.sum(grp["W_STAR"][()], dtype=np.float64)) / omega_t
            agb[i] = float(np.sum(grp["W_AGB"][()], dtype=np.float64)) / omega_t
    return star, agb, omega_sim, f_dusty_o, f_dusty_c, f_c


def _gal_density(config):
    """`A_GAL`, section 5.4 "Sky density": `integral phi(S) dS = Sigma
    phi(S) S ln10 Delta log10 S` over the counts law's tabulated grid,
    one survey number -- `bmstp.sample_gal.density`, NOT the shape weight
    `sample`'s `w = phi(S) . S . d(log10 S)` sums to (that sum is short by
    `ln 10`, sec. 5.4)."""
    return sample_gal.density(config)


def _cloud_column_fraction(config, region):
    """`A_cloud(sightline) / A_s`, section 5.5 "Sky density": the CLOUD's
    own share of the sightline's column, `1 - u(d_front)`, `u(d) =
    A_CUM_K / A_INF_K` (the profile product,
    `sky/derived/edenhofer/profile_edenhofer_sightline__R.hdf5`)
    interpolated linearly in distance on its own `DIST_PC` axis, at the
    front edge of the region's cloud interval, `d_front`
    (`bmstp.sample_cloud`'s own interval rule, W24b -- imported, not
    re-derived). Only the FOREGROUND is deducted: the 3-D dust map
    partitions a sightline's column reliably in front of a cloud and not
    behind it at a kiloparsec (toward NGC 7129 it spreads the cloud over
    500-2000 pc), so nothing behind `d_front` is deducted -- the in-map
    background and the disc tail past the map's own reach both stay with
    the cloud. Returns the per-sightline fraction, one value per row of
    the profile product's own `HPX_PIX_256` axis -- the SAME axis, in the
    SAME order, P3's `HPX_PIX_256` is built from (`bmstp.shapes.build_cloud`
    writes it straight from `sample_cloud._region_profile`'s own read of
    this product), so a source's `sightline_row` into P3 indexes it
    directly."""
    path = config_module.product_path(
        config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    with h5py.File(path, "r") as f:
        dist_pc = np.asarray(f["DIST_PC"][:], dtype=np.float64)
        a_cum_k = np.asarray(f["A_CUM_K"][:], dtype=np.float64)
        a_inf_k = np.asarray(f["A_INF_K"][:], dtype=np.float64)
    u = a_cum_k / a_inf_k[:, None]  # (n_sl, n_d): u(DIST_PC[j]) per sightline

    def _u_at(d):
        j = int(np.clip(np.searchsorted(dist_pc, d), 1, dist_pc.size - 1))
        d0, d1 = dist_pc[j - 1], dist_pc[j]
        frac = (d - d0) / (d1 - d0) if d1 > d0 else 0.0
        return u[:, j - 1] + frac * (u[:, j] - u[:, j - 1])  # (n_sl,)

    d_front, _d_back = sample_cloud.cloud_interval_pc(config, region)
    u_front = _u_at(d_front)
    return 1.0 - u_front, d_front


def build_region(config, region, st):
    rs = access.region_slice(config, region)
    n = rs["n_sources"]
    hpx512 = rs["hpx_pix_512"]
    hpx256 = rs["hpx_pix_256"]

    cat_path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(cat_path, "r") as f:
        name = f["NAME"][:]
        ra_deg = np.asarray(f["RA_DEG"][:], dtype=np.float64)
        dec_deg = np.asarray(f["DEC_DEG"][:], dtype=np.float64)
    if name.shape[0] != n:
        raise ValueError("bmstp.density: %s has %d rows, region has %d catalogued sources"
                         % (cat_path, name.shape[0], n))

    # A_s, the prior's depth axis x = a/A_s (SPEC_BMSTP_DRAFT.md sec.
    # 4.1): what a source's own light passes through, so this is the
    # extinction column, not the gas column the young-star law is
    # measured on
    col_path = config_module.product_path(
        config, "sky/derived", "adopted", "extinction", "source", region=region)
    if not os.path.exists(col_path):
        raise FileNotFoundError(
            "bmstp.density: no extinction column for %s at %s -- run the "
            "'sesnaimpute.sky.derived.column' RUNBOOKtp.sh line first" % (region, col_path))
    with h5py.File(col_path, "r") as f:
        a_col = np.asarray(f["A_COL_K"][:], dtype=np.float64)
        a_col_sig = np.asarray(f["A_COL_SIG_K"][:], dtype=np.float32)
        arm = np.asarray(f["A_COL_PROVENANCE"][:], dtype=np.uint8)
        zp_sig = np.asarray(f["ZP_SIGMA_K"][:], dtype=np.float32)
    if a_col.shape[0] != n:
        raise ValueError("bmstp.density: %s row count disagrees with the catalogue" % col_path)

    # A_cloud, the young-star law's own input (sec. 5.5): the gas
    # column exactly as adopted, not extinction -- Pokhrel's law was
    # measured on this quantity
    gas_path = config_module.product_path(
        config, "sky/derived", "adopted", "column", "source", region=region)
    with h5py.File(gas_path, "r") as f:
        a_col_gas = np.asarray(f["A_COL_K"][:], dtype=np.float64)
    if a_col_gas.shape[0] != n:
        raise ValueError("bmstp.density: %s row count disagrees with the catalogue" % gas_path)

    f_lim = limits_module.limits(config, region).astype(np.float32)
    d_pahc = (-np.log10(f_lim[:, _I4_INDEX])).astype(np.float32)

    p2_path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
    with h5py.File(p2_path, "r") as f:
        tile_id_axis = np.asarray(f["TILE_ID"][:], dtype=np.int64)
        on_grid_star_by_tile = np.asarray(f["ON_GRID_STAR"][:], dtype=np.float64)
        on_grid_agb_by_tile = np.asarray(f["ON_GRID_AGB"][:], dtype=np.float64)
    p3_path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    with h5py.File(p3_path, "r") as f:
        sightline_axis = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        on_grid_yso_by_sightline = np.asarray(f["ON_GRID_YSO"][:], dtype=np.float64)
        on_grid_h2s_by_sightline = np.asarray(f["ON_GRID_H2S"][:], dtype=np.float64)
    p4_path = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
    with h5py.File(p4_path, "r") as f:
        on_grid_gal = float(f.attrs["ON_GRID_GAL"])

    tile_row = _resolve_tile_row(config, region, hpx512, tile_id_axis)
    sightline_row = _resolve_sightline_row(hpx256, sightline_axis)
    on_grid_star = on_grid_star_by_tile[tile_row]
    on_grid_agb = on_grid_agb_by_tile[tile_row]
    on_grid_yso = on_grid_yso_by_sightline[sightline_row]
    on_grid_h2s = on_grid_h2s_by_sightline[sightline_row]
    st.tick(1, 4, "batches")

    star_by_tile, agb_by_tile, omega_sim, f_dusty_o, f_dusty_c, f_c = \
        _star_family_density_by_tile(config, region, tile_id_axis)
    density_star_raw = star_by_tile[tile_row]
    density_agb_raw = agb_by_tile[tile_row]
    density_pahc_raw = density_star_raw.copy()
    # section 4.1: every density is a RETAINED density, the intrinsic
    # count times the population's on-grid fraction at the source's own
    # grain -- STAR/AGB by tile (PAHC reads STAR's own grid, W26).
    density_star = density_star_raw * on_grid_star
    density_agb = density_agb_raw * on_grid_agb
    density_pahc = density_pahc_raw * on_grid_star
    st.tick(2, 4, "batches")

    density_gal_value = _gal_density(config)
    density_gal_raw = np.full(n, density_gal_value, dtype=np.float64)
    density_gal = density_gal_raw * on_grid_gal

    reg = regions_module.REGIONS_BY_NAME[region]
    d_r_pc = float(reg.d_r_pc)
    pc2 = float(pc2_per_deg2(d_r_pc))
    # sec. 5.5 "Sky density": the law is applied to the CLOUD's own share
    # of the column, `A_cloud = A_s . [1 - u(d_front)]`, not the source's
    # whole adopted column -- only the FOREGROUND is deducted; the 3-D
    # map cannot partition the column behind a cloud at a kiloparsec, so
    # nothing behind the front edge is deducted (W26b).
    cloud_frac_by_sightline, d_front = _cloud_column_fraction(config, region)
    cloud_frac = cloud_frac_by_sightline[sightline_row]
    a_cloud = a_col_gas * cloud_frac
    density_yso_intrinsic = law_count(config, region, a_cloud, arm)
    density_yso = density_yso_intrinsic * on_grid_yso
    st.tick(3, 4, "batches")

    law_path = config_module.product_path(config, "population", "yso", "law", "region")
    with h5py.File(law_path, "r") as f:
        law_region_names = [v.decode("utf-8") for v in f["REGION"][:]]
        i_law = law_region_names.index(region)
        file_kappa_h = float(f["KAPPA_HERSCHEL"][()])
        file_kappa_p = float(f["KAPPA_PLANCK"][()])
        file_pc2 = float(f["PC2_PER_DEG2"][i_law])
    yso_law_err = max(abs(KAPPA_HERSCHEL - file_kappa_h), abs(KAPPA_PLANCK - file_kappa_p),
                      abs(pc2 - file_pc2) / file_pc2)

    # H2S, sec. 5.6 "Sky density": `A_H2S(s) = L(s) . eta_r . eps_ext .
    # ON_GRID_H2S(s)`. `L(s)` is the INTRINSIC young-star law at the
    # source's own column/arm/region distance -- `density_yso_intrinsic`
    # above -- EXCEPT for a Herschel-arm source whose position the
    # region's convolved law map
    # (`bmstp.knot_field.convolved_law`) reaches, where `L(s)` is that
    # convolution sampled at the source instead (a map operation, once
    # per region). An edge Herschel-arm source (outside the convolved
    # map) falls back to `density_yso_intrinsic`, counted below.
    eta_r = ETA.get(region, ETA_ELSEWHERE)
    law_map, law_wcs, knot_meta = knot_field.convolved_law(config, region)
    herschel_mask = arm == PROVENANCE_HERSCHEL
    l_of_s = density_yso_intrinsic.copy()
    n_herschel = int(np.count_nonzero(herschel_mask))
    n_edge = 0
    knot_ratio_median = knot_ratio_p90 = float("nan")
    if law_map is not None and n_herschel:
        l_convolved = knot_field.sample_at(law_map, law_wcs, ra_deg[herschel_mask], dec_deg[herschel_mask])
        finite = np.isfinite(l_convolved)
        n_edge = int(np.count_nonzero(~finite))
        idx = np.flatnonzero(herschel_mask)
        l_of_s[idx[finite]] = l_convolved[finite]
        # sec. 5.6's report: the kernel's own effect on Herschel-arm
        # sources, `L(s) / (kappa_Herschel A_s^2 . pc2/deg2)` -- the
        # ratio of the convolved to the unconvolved law at the same
        # source, `density_yso_intrinsic` being exactly that unconvolved
        # value.
        ratio = l_convolved[finite] / density_yso_intrinsic[idx[finite]]
        if ratio.size:
            knot_ratio_median = float(np.median(ratio))
            knot_ratio_p90 = float(np.percentile(ratio, 90))
    # every class's density is the retained density (sec. 4.1): H2S
    # multiplies in `ON_GRID_H2S`, its own per-sightline retained
    # fraction on the common depth axis, read from P3 exactly as
    # `ON_GRID_YSO` is -- it is stored there, not "formed at read", since
    # the knots' brightness axis moved onto that common grid.
    density_h2s = l_of_s * eta_r * EPS_EXT * on_grid_h2s
    st.tick(4, 4, "batches")

    retention_limits = field_stars.deepest_limits(config, region).astype(np.float64)

    return dict(
        name=name, a_col=a_col.astype(np.float32), a_col_sig=a_col_sig, arm=arm, zp_sig=zp_sig,
        tile=tile_row, sightline_row=sightline_row, hpx512=hpx512.astype(np.int64),
        f_lim=f_lim, d_pahc=d_pahc, a_cloud=a_cloud.astype(np.float32), cloud_frac=cloud_frac,
        density_star=density_star, density_agb=density_agb, density_pahc=density_pahc,
        density_gal=density_gal, density_yso=density_yso, density_h2s=density_h2s,
        density_star_raw=density_star_raw, density_agb_raw=density_agb_raw,
        density_gal_raw=density_gal_raw, density_yso_intrinsic=density_yso_intrinsic,
        on_grid_star=on_grid_star, on_grid_agb=on_grid_agb, on_grid_yso=on_grid_yso,
        on_grid_h2s=on_grid_h2s, on_grid_gal=on_grid_gal, d_front=d_front,
        omega_sim=omega_sim, f_dusty_o=f_dusty_o, f_dusty_c=f_dusty_c, f_c=f_c,
        eta_r=eta_r, retention_limits=retention_limits, yso_law_err=yso_law_err,
        d_r_pc=d_r_pc, n=n, knot_meta=knot_meta, n_herschel=n_herschel, n_edge=n_edge,
        knot_ratio_median=knot_ratio_median, knot_ratio_p90=knot_ratio_p90)


def write_region(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.create_dataset("NAME", data=result["name"])
        f.create_dataset("A_COL_K", data=result["a_col"])
        f.create_dataset("A_COL_SIG_K", data=result["a_col_sig"])
        f.create_dataset("ARM", data=result["arm"])
        f.create_dataset("ZP_SIG_K", data=result["zp_sig"])
        f.create_dataset("TILE", data=result["tile"])
        f.create_dataset("SIGHTLINE_ROW", data=result["sightline_row"])
        f.create_dataset("HPX_512", data=result["hpx512"])
        f.create_dataset("F_LIM_50_MJY", data=result["f_lim"])
        f.create_dataset("D_PAHC", data=result["d_pahc"])
        f.create_dataset("A_CLOUD_K", data=result["a_cloud"])
        f.create_dataset("DENSITY_STAR", data=result["density_star"].astype(np.float64))
        f.create_dataset("DENSITY_AGB", data=result["density_agb"].astype(np.float64))
        f.create_dataset("DENSITY_PAHC", data=result["density_pahc"].astype(np.float64))
        f.create_dataset("DENSITY_GAL", data=result["density_gal"].astype(np.float64))
        f.create_dataset("DENSITY_YSO", data=result["density_yso"].astype(np.float64))
        f.create_dataset("DENSITY_H2S", data=result["density_h2s"].astype(np.float64))
        f.attrs["KAPPA_HERSCHEL"] = KAPPA_HERSCHEL
        f.attrs["KAPPA_PLANCK"] = KAPPA_PLANCK
        f.attrs["ETA"] = result["eta_r"]
        f.attrs["EPS_EXT"] = EPS_EXT
        f.attrs["F_DUSTY_O"] = result["f_dusty_o"]
        f.attrs["F_DUSTY_C"] = result["f_dusty_c"]
        f.attrs["F_C"] = result["f_c"]
        f.attrs["OMEGA_SIM_DEG2"] = result["omega_sim"]
        f.attrs["RETENTION_LIMITS_MJY"] = result["retention_limits"]
        # sec. 4.1/4.2: the grid's own lower edge -- every class's
        # retention limit, `F_4.5 >= 0.1 uJy` dereddened -- recorded
        # beside the six retained densities (W26).
        f.attrs["RETENTION_LOG10_F45"] = float(grid.LOG10_F45_EDGES[0])


def build(config, regions=None):
    """Writes `bmstp/density/table_density_source[__R].hdf5` for `regions`
    (default all thirty), one file per region."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        with progress.Stage("bmstp.density", region) as st:
            path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
            # rule: read the CURRENT product's own DENSITY_* before this
            # build overwrites it -- the sec. 9 identity (STAR/GAL before
            # retention bit-identical to the current, pre-W26 product)
            # and the YSO before/after report both need it.
            old = None
            if os.path.exists(path):
                with h5py.File(path, "r") as f:
                    old = dict(
                        density_star=np.asarray(f["DENSITY_STAR"][:], dtype=np.float64),
                        density_gal=np.asarray(f["DENSITY_GAL"][:], dtype=np.float64),
                        density_yso=np.asarray(f["DENSITY_YSO"][:], dtype=np.float64),
                        density_h2s=np.asarray(f["DENSITY_H2S"][:], dtype=np.float64),
                    )

            result = build_region(config, region, st)
            write_region(path, result)

            if old is not None and old["density_star"].shape == result["density_star_raw"].shape:
                star_dev = float(np.max(np.abs(old["density_star"] - result["density_star_raw"])))
                gal_dev = float(np.max(np.abs(old["density_gal"] - result["density_gal_raw"])))
                yso_sum_before = float(np.sum(old["density_yso"]))
            else:
                star_dev = gal_dev = float("nan")
                yso_sum_before = float("nan")
            yso_sum_after = float(np.sum(result["density_yso"]))
            print(f"bmstp.density {region}: identity DENSITY_STAR before-retention max abs dev "
                  f"vs current product={star_dev:.3e}, DENSITY_GAL before-retention max abs dev "
                  f"vs current product={gal_dev:.3e} (sec. 9: bit-identical, W26 adds only the "
                  f"multiplicative retention step)")
            print(f"bmstp.density {region}: DENSITY_YSO sum over sources before={yso_sum_before:.6g} "
                  f"after={yso_sum_after:.6g} deg^-2 (before: law_count(A_s), no retention; after: "
                  f"law_count(A_cloud) x ON_GRID_YSO, W26)")

            cloud_frac = result["cloud_frac"]
            cf_med, cf_16, cf_84 = np.percentile(cloud_frac, [50, 16, 84])
            print(f"bmstp.density {region}: A_cloud/A_s median={cf_med:.6g} "
                  f"16-84%=[{cf_16:.6g}, {cf_84:.6g}] (cloud interval front edge "
                  f"d_front={result['d_front']:.6g} pc)")
            for cls, og in (("STAR", result["on_grid_star"]), ("AGB", result["on_grid_agb"]),
                            ("YSO", result["on_grid_yso"]), ("H2S", result["on_grid_h2s"])):
                print(f"bmstp.density {region}: on-grid fraction {cls} median={np.median(og):.6g}")
            print(f"bmstp.density {region}: on-grid fraction GAL={result['on_grid_gal']:.6g} "
                  f"(survey-wide attr; PAHC uses STAR's own on-grid fraction, sec. 4.1)")

            # every dataset but DENSITY_H2S is unaffected by this change:
            # a bit-identical check against the current product for each,
            # and DENSITY_H2S's own before/after median ratio, which is
            # exactly ON_GRID_H2S since nothing else in its computation
            # moved.
            if old is not None and old["density_h2s"].shape == result["density_h2s"].shape:
                for cls, key in (("STAR", "density_star"), ("GAL", "density_gal"),
                                  ("YSO", "density_yso")):
                    dev = float(np.max(np.abs(old[key] - result[key])))
                    print(f"bmstp.density {region}: identity {key.upper()} vs current "
                          f"product max abs dev={dev:.3e}")
                h2s_ratio = result["density_h2s"] / old["density_h2s"]
                print(f"bmstp.density {region}: DENSITY_H2S before/after median ratio="
                      f"{np.median(h2s_ratio):.6g} (should equal ON_GRID_H2S's own median)")

            # sec. 9-style acceptance: DENSITY_YSO on the first ten sources,
            # recomputed BY HAND straight off the adopted-column and
            # profile products (full float64, independent of `a_col`'s
            # float32 storage cast and of `_cloud_column_fraction`'s own
            # code path) -- `law_count(A_cloud) x ON_GRID_YSO`, bar 1e-9.
            n_check = min(10, result["n"])
            if n_check:
                col_path_h = config_module.product_path(
                    config, "sky/derived", "adopted", "column", "source", region=region)
                with h5py.File(col_path_h, "r") as f:
                    a_col_h = np.asarray(f["A_COL_K"][:n_check], dtype=np.float64)
                profile_path_h = config_module.product_path(
                    config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
                rows_h = result["sightline_row"][:n_check]
                with h5py.File(profile_path_h, "r") as f:
                    dist_pc_h = np.asarray(f["DIST_PC"][:], dtype=np.float64)
                    # h5py fancy indexing needs increasing order; the
                    # ten sources' own sightline rows are not sorted, so
                    # the full arrays are read once (report-only, ten
                    # sources) and indexed in numpy instead.
                    a_cum_h = np.asarray(f["A_CUM_K"][:], dtype=np.float64)[rows_h, :]
                    a_inf_h = np.asarray(f["A_INF_K"][:], dtype=np.float64)[rows_h]
                d_front_h = result["d_front"]
                kappa_h = np.where(result["arm"][:n_check] == PROVENANCE_HERSCHEL, KAPPA_HERSCHEL, KAPPA_PLANCK)
                pc2_h = float(pc2_per_deg2(result["d_r_pc"]))
                hand_yso = np.empty(n_check, dtype=np.float64)
                for k in range(n_check):
                    u_row = a_cum_h[k] / a_inf_h[k]
                    u_front_h = np.interp(d_front_h, dist_pc_h, u_row)
                    a_cloud_h = a_col_h[k] * (1.0 - u_front_h)
                    hand_yso[k] = kappa_h[k] * pc2_h * a_cloud_h ** 2 * result["on_grid_yso"][k]
                hand_dev = float(np.max(np.abs(hand_yso - result["density_yso"][:n_check])))
            else:
                hand_dev = float("nan")
            print(f"bmstp.density {region}: DENSITY_YSO hand check on {n_check} sources, "
                  f"max abs dev={hand_dev:.3e} (bar 1e-9)")

            n = result["n"]
            n_pix = int(np.unique(result["hpx512"]).size)
            area_deg2 = n_pix * _HPX512_PIXEL_DEG2
            classes = ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S")
            keys = ("density_star", "density_agb", "density_pahc",
                    "density_gal", "density_yso", "density_h2s")
            total_sum = 0.0
            print(f"bmstp.density {region}: n={n}, occupied nside-512 pixels={n_pix}, "
                  f"report-only area~{area_deg2:.4g} deg^2 (occupied-pixel estimate, not the "
                  f"admitted grid -- section 8's level check is report-only)")
            for cls, key in zip(classes, keys):
                mean_density = float(np.mean(result[key]))
                total = mean_density * area_deg2
                total_sum += total
                print(f"bmstp.density {region}: {cls} mean={mean_density:.6g} deg^-2, "
                      f"total over area={total:.6g}")
            ratio = total_sum / n if n else float("nan")
            n_unresolved = int(np.count_nonzero(result["tile"] < 0) +
                              np.count_nonzero(result["sightline_row"] < 0))

            # sec. 5.6's brief report: the knot-driver convolution's own
            # numbers, once per region -- the map operation `knot_field.
            # convolved_law` ran (or "no HGBS map" if the region is all
            # Planck arm, e.g. NGC 7129).
            km = result["knot_meta"]
            if km is None:
                print(f"bmstp.density {region}: H2S convolution skipped, no HGBS map serves "
                      f"this region (n_herschel_arm={result['n_herschel']})")
            else:
                print(f"bmstp.density {region}: H2S convolution map={km['map_name']} "
                      f"native={km['shape_native']} downsample_factor={km['downsample_factor']} "
                      f"downsampled={km['shape_ds']} kernel_radius_px={km['kernel_radius_px']} "
                      f"kernel_width_px={km['kernel_width_px']:.2f} wall_s={km['wall_s']:.4g}")
                print(f"bmstp.density {region}: H2S conservation total_before={km['total_before']:.6g} "
                      f"total_after={km['total_after']:.6g} rel_err={km['conservation_rel_err']:.3e} "
                      f"(sec. 5.6: kernel conserves the law's total)")
                print(f"bmstp.density {region}: H2S n_herschel_arm={result['n_herschel']} "
                      f"n_edge_own_column={result['n_edge']} "
                      f"L(s)/(kappa A_s^2) median={result['knot_ratio_median']:.4g} "
                      f"p90={result['knot_ratio_p90']:.4g}")
            density_h2s_before = float(np.mean(result["density_yso_intrinsic"] * result["eta_r"] * EPS_EXT))
            density_h2s_after = float(np.mean(result["density_h2s"]))
            print(f"bmstp.density {region}: RATIO_H2S (mean density, deg^-2) "
                  f"before={density_h2s_before:.6g} after={density_h2s_after:.6g} "
                  f"ratio_after/before={density_h2s_after / density_h2s_before if density_h2s_before else float('nan'):.4g}")

            st.done(path, n=n, tile_sightline_unresolved=n_unresolved,
                     yso_law_identity_err=result["yso_law_err"],
                     six_class_total_over_n_catalogued=ratio)


if __name__ == "__main__":
    run(build)
