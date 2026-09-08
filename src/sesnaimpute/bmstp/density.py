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
the quadratic column law, `kappa` selected by which arm reached the
source (`ARM`, from the adopted column's own provenance flag). H2S
(section 5.6) rides on that same young-star law density, scaled by the
region's `eta` and the universal `eps_ext`, but on the Herschel arm the
law itself is the knot-driver kernel's convolution of the region's HGBS
map (`bmstp.knot_field.convolved_law`), sampled at the source's own
position -- a map operation, once per region, never a per-source
convolution; a Herschel-arm source whose position falls outside the
convolved map, and every Planck-arm source (the kernel is sub-beam at
Planck's 5.03' beam), takes the law at its own column instead.
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
from sesnaimpute.population.yso import PROVENANCE_HERSCHEL, pc2_per_deg2
from sesnaimpute.bmstp import knot_field
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

    col_path = config_module.product_path(
        config, "sky/derived", "adopted", "column", "source", region=region)
    with h5py.File(col_path, "r") as f:
        a_col = np.asarray(f["A_COL_K"][:], dtype=np.float64)
        a_col_sig = np.asarray(f["A_COL_SIG_K"][:], dtype=np.float32)
        arm = np.asarray(f["A_COL_PROVENANCE"][:], dtype=np.uint8)
        zp_sig = np.asarray(f["ZP_SIGMA_K"][:], dtype=np.float32)
    if a_col.shape[0] != n:
        raise ValueError("bmstp.density: %s row count disagrees with the catalogue" % col_path)

    f_lim = limits_module.limits(config, region).astype(np.float32)
    d_pahc = (-np.log10(f_lim[:, _I4_INDEX])).astype(np.float32)

    p2_path = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
    with h5py.File(p2_path, "r") as f:
        tile_id_axis = np.asarray(f["TILE_ID"][:], dtype=np.int64)
    p3_path = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    with h5py.File(p3_path, "r") as f:
        sightline_axis = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)

    tile_row = _resolve_tile_row(config, region, hpx512, tile_id_axis)
    sightline_row = _resolve_sightline_row(hpx256, sightline_axis)
    st.tick(1, 4, "batches")

    star_by_tile, agb_by_tile, omega_sim, f_dusty_o, f_dusty_c, f_c = \
        _star_family_density_by_tile(config, region, tile_id_axis)
    density_star = star_by_tile[tile_row]
    density_agb = agb_by_tile[tile_row]
    density_pahc = density_star.copy()
    st.tick(2, 4, "batches")

    density_gal_value = _gal_density(config)
    density_gal = np.full(n, density_gal_value, dtype=np.float64)

    reg = regions_module.REGIONS_BY_NAME[region]
    d_r_pc = float(reg.d_r_pc)
    pc2 = float(pc2_per_deg2(d_r_pc))
    kappa = np.where(arm == 0, KAPPA_HERSCHEL, KAPPA_PLANCK)
    density_yso = kappa * pc2 * (a_col ** 2)
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

    # H2S, sec. 5.6 "Sky density": `A_H2S(s) = L(s) . eta_r . eps_ext`.
    # `L(s)` is the young-star law at the source's own column/arm/region
    # distance -- `density_yso` above, already that quantity -- EXCEPT for
    # a Herschel-arm source whose position the region's convolved law map
    # (`bmstp.knot_field.convolved_law`) reaches, where `L(s)` is that
    # convolution sampled at the source instead (a map operation, once
    # per region). An edge Herschel-arm source (outside the convolved
    # map) falls back to `density_yso`, counted below.
    eta_r = ETA.get(region, ETA_ELSEWHERE)
    law_map, law_wcs, knot_meta = knot_field.convolved_law(config, region)
    herschel_mask = arm == PROVENANCE_HERSCHEL
    l_of_s = density_yso.copy()
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
        # source, `density_yso` being exactly that unconvolved value.
        ratio = l_convolved[finite] / density_yso[idx[finite]]
        if ratio.size:
            knot_ratio_median = float(np.median(ratio))
            knot_ratio_p90 = float(np.percentile(ratio, 90))
    density_h2s = l_of_s * eta_r * EPS_EXT
    st.tick(4, 4, "batches")

    retention_limits = field_stars.deepest_limits(config, region).astype(np.float64)

    return dict(
        name=name, a_col=a_col.astype(np.float32), a_col_sig=a_col_sig, arm=arm, zp_sig=zp_sig,
        tile=tile_row, sightline_row=sightline_row, hpx512=hpx512.astype(np.int64),
        f_lim=f_lim, d_pahc=d_pahc,
        density_star=density_star, density_agb=density_agb, density_pahc=density_pahc,
        density_gal=density_gal, density_yso=density_yso, density_h2s=density_h2s,
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


def build(config, regions=None):
    """Writes `bmstp/density/table_density_source[__R].hdf5` for `regions`
    (default all thirty), one file per region."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in region_names:
        with progress.Stage("bmstp.density", region) as st:
            result = build_region(config, region, st)
            path = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
            write_region(path, result)

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
            density_h2s_before = float(np.mean(result["density_yso"] * result["eta_r"] * EPS_EXT))
            density_h2s_after = float(np.mean(result["density_h2s"]))
            print(f"bmstp.density {region}: RATIO_H2S (mean density, deg^-2) "
                  f"before={density_h2s_before:.6g} after={density_h2s_after:.6g} "
                  f"ratio_after/before={density_h2s_after / density_h2s_before if density_h2s_before else float('nan'):.4g}")

            st.done(path, n=n, tile_sightline_unresolved=n_unresolved,
                     yso_law_identity_err=result["yso_law_err"],
                     six_class_total_over_n_catalogued=ratio)


if __name__ == "__main__":
    run(build)
