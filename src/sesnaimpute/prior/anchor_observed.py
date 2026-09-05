"""The STAR anchors' observed joint histogram and the young-star
subtraction (SPEC_PRIORS.md section 2.1, the "joint and marginal bins"
and "young stars in the anchors" rows), per nside-512 anchor pixel --
the same pixels and edges `prior.anchor_tiles.write_histograms` and
`prior.young_stars` already wrote.

N_GK_OBS. SESNA sources that carry both a measured `Ks` (`ORIGIN_FNU`
== 1 in the `Ks` band -- a real detection, not a substituted bound) and
a Gaia match, 2-D digitised on `(G_MAG` of the match, `Ks` Vega
magnitude of the source's own flux`)` onto the histograms product's own
`G_EDGES`/`KS_EDGES`, each source weighted by its own match probability
`G_S` rather than counted once. This joint histogram is conditioned on
membership in SESNA, which is complete for stars brighter than both
cuts, so the conditioning is benign (section 2.1, "joint and marginal
bins").

THE SUBTRACTION (section 2.1, "young stars in the anchors"). Before a
weight is formed elsewhere, each observed histogram has the model's own
expectation of young stars subtracted, floored at zero:

    N_G_SUB  = max(N_G_OBS  - N_G_YOUNG,  0)
    N_KS_SUB = max(N_KS_OBS - N_KS_YOUNG, 0)
    N_GK_SUB = max(N_GK_OBS - N_GK_YOUNG, 0)

`prior.young_stars` carries only the two young-star marginals, not their
joint, so `N_GK_YOUNG` is formed here as the outer product of
`N_G_YOUNG` and `N_KS_YOUNG`, normalised to the pixel's own young-star
total (`N_YOUNG_TOTAL`, zero where the total is zero):

    N_GK_YOUNG[pix] = N_G_YOUNG[pix] (x) N_KS_YOUNG[pix] / N_YOUNG_TOTAL[pix]

This assumes a young star's `G` and `Ks` placements are independent
given the pixel, which they are not exactly -- both come from the same
drawn mass and the same drawn extinction. The exact joint is a later
refinement of `young_stars.py`, which would have to carry it directly
rather than leave it to be reconstructed from two marginals.

`FLOORED_G`/`FLOORED_KS`/`FLOORED_GK` (n_pix): the number of bins per
pixel where the floor actually triggered (`N_*_OBS < N_*_YOUNG`).
`SUBTRACTED_FRAC_G`/`SUBTRACTED_FRAC_KS` (n_pix): the pixel-total share
of the observed marginal removed by the subtraction, `(N_OBS.sum() -
N_SUB.sum()) / N_OBS.sum()` (zero where the pixel has no observed
counts at all).

Product, per region, `bms/anchors/observed_anchors_hpx512__<Region>
.hdf5`: `HPX_PIX_512`, `G_EDGES`, `KS_EDGES`, `N_GK_OBS`, `N_G_SUB`,
`N_KS_SUB`, `N_GK_SUB` (n_pix, n_G_bin, n_Ks_bin` for the joint arrays,
`(n_pix, n_bin)` for the marginal), `FLOORED_G`, `FLOORED_KS`,
`FLOORED_GK`, `SUBTRACTED_FRAC_G`, `SUBTRACTED_FRAC_KS`; root attr
`GRANULE="hpx512"`.
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.granules import access

#: `definitions.BANDS`'s own column position of `Ks` inside `FNU_MJY`/
#: `ORIGIN_FNU` (n_source, 8), catalog.curated's own band order.
_KS_BAND_INDEX = [b.key for b in definitions.BANDS].index("Ks")

#: `catalog.curated`'s own code for "measured detection" in `ORIGIN_FNU`
#: (catalog/curated.py module docstring).
_ORIGIN_MEASURED = 1


# ---------------------------------------------------------------------
# the observed joint histogram
# ---------------------------------------------------------------------

def ks_vega_mag(fnu_ks_mjy):
    """`Ks` Vega magnitude from the curated catalogue's own `FNU_MJY`,
    through `constants.VEGA_ZERO_POINT_MJY["Ks"]` (the same zero point
    `catalog.curated`'s own flux assembly cites)."""
    fnu_ks_mjy = np.asarray(fnu_ks_mjy, dtype=np.float64)
    return -2.5 * np.log10(fnu_ks_mjy / constants.VEGA_ZERO_POINT_MJY["Ks"])


def _joint_histogram(pix_idx, g_vals, ks_vals, weights, n_pix, g_edges, ks_edges):
    """`N_GK_OBS` (n_pix, n_G_bin, n_Ks_bin): every entering source's own
    `(G, Ks)` weighted by `weights`, 2-D digitised via one flattened
    `bincount` (no Python loop over pixels or sources -- CODING_RULES.md
    rule 8). Returns the histogram and the entering mask (in both edge
    ranges), the module's own acceptance identity's "entered" set.
    """
    n_g = g_edges.size - 1
    n_ks = ks_edges.size - 1
    g_idx = np.searchsorted(g_edges, g_vals, side="right") - 1
    ks_idx = np.searchsorted(ks_edges, ks_vals, side="right") - 1
    in_range = ((g_vals >= g_edges[0]) & (g_vals < g_edges[-1])
                & (ks_vals >= ks_edges[0]) & (ks_vals < ks_edges[-1]))
    flat_idx = (pix_idx[in_range] * n_g + g_idx[in_range]) * n_ks + ks_idx[in_range]
    counts = np.bincount(flat_idx, weights=weights[in_range], minlength=n_pix * n_g * n_ks)
    return counts.reshape(n_pix, n_g, n_ks), in_range


# ---------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------

def _read_histograms(config, region):
    path = config_module.product_path(config, "bms", "anchors", "histograms", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.anchor_observed: anchor histograms missing for region %r at "
            "%s -- run the 'prior.anchor_tiles' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        return dict(
            pixels=np.asarray(f["HPX_PIX_512"][:], dtype=np.int64),
            g_edges=np.asarray(f["G_EDGES"][:], dtype=np.float64),
            ks_edges=np.asarray(f["KS_EDGES"][:], dtype=np.float64),
            n_g_obs=np.asarray(f["N_G_OBS"][:], dtype=np.float64),
            n_ks_obs=np.asarray(f["N_KS_OBS"][:], dtype=np.float64),
        )


def _read_young_stars(config, region, pixels):
    path = config_module.product_path(config, "bms", "anchors", "young-stars", "hpx512", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.anchor_observed: young-star anchor expectation missing for region %r at "
            "%s -- run the 'prior.young_stars' RUNBOOK line first" % (region, path))
    with h5py.File(path, "r") as f:
        yp = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        n_g_young = np.asarray(f["N_G_YOUNG"][:], dtype=np.float64)
        n_ks_young = np.asarray(f["N_KS_YOUNG"][:], dtype=np.float64)
        n_young_total = np.asarray(f["N_YOUNG_TOTAL"][:], dtype=np.float64)
    if not np.array_equal(pixels, yp):
        raise ValueError(
            "prior.anchor_observed: %r's young-star product pixel set disagrees with "
            "the anchor histograms' own pixel set -- rerun 'prior.anchor_tiles' or "
            "'prior.young_stars' for this region" % region)
    return n_g_young, n_ks_young, n_young_total


def _read_sesna_ks_and_pixels(config, region):
    cat_path = config_module.product_path(config, "catalog", "sesna", "sources", "source", region=region)
    if not os.path.exists(cat_path):
        raise FileNotFoundError(
            "prior.anchor_observed: curated catalogue missing for region %r at "
            "%s -- run the curated-catalogue RUNBOOK line for it" % (region, cat_path))
    with h5py.File(cat_path, "r") as f:
        fnu_ks = np.asarray(f["FNU_MJY"][:, _KS_BAND_INDEX], dtype=np.float64)
        ks_measured = np.asarray(f["ORIGIN_FNU"][:, _KS_BAND_INDEX]) == _ORIGIN_MEASURED
        n_sources = fnu_ks.size

    granule_map_path = config_module.product_path(config, "granules", "sesna", "granule-map", "source")
    hpx_pix_512 = access.per_source(config, region, granule_map_path, ["HPX_PIX_512"])["HPX_PIX_512"]
    hpx_pix_512 = np.asarray(hpx_pix_512, dtype=np.int64)
    if hpx_pix_512.size != n_sources:
        raise ValueError(
            "prior.anchor_observed: %r's granule map has %d source rows, curated catalogue "
            "has %d -- row counts cannot be reconciled" % (region, hpx_pix_512.size, n_sources))
    return fnu_ks, ks_measured, hpx_pix_512


def _read_gaia_match(config, region):
    path = config_module.product_path(config, "sky/derived", "gaia", "match", "source", region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "prior.anchor_observed: Gaia match product missing for region %r at "
            "%s -- run the 'sesnaimpute.sky.derived.gaia_match' RUNBOOK line first" % (region, path))
    cols = access.per_source(config, region, path, ["G_S", "G_MAG"])
    return np.asarray(cols["G_S"], dtype=np.float64), np.asarray(cols["G_MAG"], dtype=np.float64)


def build_region(config, region):
    hist = _read_histograms(config, region)
    pixels, g_edges, ks_edges = hist["pixels"], hist["g_edges"], hist["ks_edges"]
    n_pix = pixels.size

    n_g_young, n_ks_young, n_young_total = _read_young_stars(config, region, pixels)

    fnu_ks, ks_measured, src_pix = _read_sesna_ks_and_pixels(config, region)
    g_s, g_mag = _read_gaia_match(config, region)
    has_match = np.isfinite(g_mag)

    member = ks_measured & has_match
    pix_idx_all = np.searchsorted(pixels, src_pix)
    capped = np.minimum(pix_idx_all, max(n_pix - 1, 0))
    pix_valid = (n_pix > 0) & (pixels[capped] == src_pix)
    if not np.all(pix_valid[member]):
        raise ValueError(
            "prior.anchor_observed: %r has a SESNA source entering the joint observed "
            "histogram whose hpx512 pixel is absent from the anchor histograms' own pixel "
            "set" % region)

    ks_vals_all = ks_vega_mag(fnu_ks)
    n_gk_obs, in_range = _joint_histogram(
        capped[member], g_mag[member], ks_vals_all[member], g_s[member],
        n_pix, g_edges, ks_edges)
    n_entering = int(np.count_nonzero(in_range))
    g_s_entering_sum = float(g_s[member][in_range].sum())

    n_young_total_safe = np.where(n_young_total > 0, n_young_total, 1.0)
    n_gk_young = (n_g_young[:, :, None] * n_ks_young[:, None, :]
                  / n_young_total_safe[:, None, None])
    n_gk_young = np.where(n_young_total[:, None, None] > 0, n_gk_young, 0.0)

    n_g_obs, n_ks_obs = hist["n_g_obs"], hist["n_ks_obs"]
    n_g_sub = np.maximum(n_g_obs - n_g_young, 0.0)
    n_ks_sub = np.maximum(n_ks_obs - n_ks_young, 0.0)
    n_gk_sub = np.maximum(n_gk_obs - n_gk_young, 0.0)

    floored_g = np.count_nonzero(n_g_obs < n_g_young, axis=1).astype(np.int64)
    floored_ks = np.count_nonzero(n_ks_obs < n_ks_young, axis=1).astype(np.int64)
    floored_gk = np.count_nonzero(n_gk_obs < n_gk_young, axis=(1, 2)).astype(np.int64)

    total_g_obs = n_g_obs.sum(axis=1)
    total_ks_obs = n_ks_obs.sum(axis=1)
    subtracted_frac_g = np.where(
        total_g_obs > 0, (total_g_obs - n_g_sub.sum(axis=1)) / np.where(total_g_obs > 0, total_g_obs, 1.0), 0.0)
    subtracted_frac_ks = np.where(
        total_ks_obs > 0, (total_ks_obs - n_ks_sub.sum(axis=1)) / np.where(total_ks_obs > 0, total_ks_obs, 1.0), 0.0)

    return dict(
        pixels=pixels, g_edges=g_edges, ks_edges=ks_edges,
        n_gk_obs=n_gk_obs, n_g_sub=n_g_sub, n_ks_sub=n_ks_sub, n_gk_sub=n_gk_sub,
        floored_g=floored_g, floored_ks=floored_ks, floored_gk=floored_gk,
        subtracted_frac_g=subtracted_frac_g, subtracted_frac_ks=subtracted_frac_ks,
        n_entering=n_entering, g_s_entering_sum=g_s_entering_sum,
        total_g_obs=float(total_g_obs.sum()), total_ks_obs=float(total_ks_obs.sum()),
    )


def _write_product(config, region, result):
    path = config_module.product_path(config, "bms", "anchors", "observed", "hpx512", region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = "hpx512"
        f.create_dataset("HPX_PIX_512", data=result["pixels"].astype(np.int64))
        f.create_dataset("G_EDGES", data=result["g_edges"])
        f.create_dataset("KS_EDGES", data=result["ks_edges"])
        f.create_dataset("N_GK_OBS", data=result["n_gk_obs"].astype(np.float32))
        f.create_dataset("N_G_SUB", data=result["n_g_sub"].astype(np.float32))
        f.create_dataset("N_KS_SUB", data=result["n_ks_sub"].astype(np.float32))
        f.create_dataset("N_GK_SUB", data=result["n_gk_sub"].astype(np.float32))
        f.create_dataset("FLOORED_G", data=result["floored_g"])
        f.create_dataset("FLOORED_KS", data=result["floored_ks"])
        f.create_dataset("FLOORED_GK", data=result["floored_gk"])
        f.create_dataset("SUBTRACTED_FRAC_G", data=result["subtracted_frac_g"].astype(np.float32))
        f.create_dataset("SUBTRACTED_FRAC_KS", data=result["subtracted_frac_ks"].astype(np.float32))
    return path


def build(config, regions=None):
    """Writes, per region (default: all thirty), `bms/anchors/observed_
    anchors_hpx512__<Region>.hdf5` (module docstring). Prints, per
    region, the number of sources entering `N_GK_OBS` and their summed
    `G_S` weight, the fraction of pixels carrying any floored bin, and
    the region-total subtracted fraction in `G` and `Ks`.
    """
    names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    for region in names:
        result = build_region(config, region)
        path = _write_product(config, region, result)

        any_floored = ((result["floored_g"] > 0) | (result["floored_ks"] > 0)
                        | (result["floored_gk"] > 0))
        frac_floored_pix = float(any_floored.mean()) if any_floored.size else float("nan")

        total_g_obs, total_ks_obs = result["total_g_obs"], result["total_ks_obs"]
        region_frac_g = ((total_g_obs - float(result["n_g_sub"].sum())) / total_g_obs
                          if total_g_obs > 0 else float("nan"))
        region_frac_ks = ((total_ks_obs - float(result["n_ks_sub"].sum())) / total_ks_obs
                           if total_ks_obs > 0 else float("nan"))
        print(
            "prior.anchor_observed: %s N_GK_OBS entering=%d G_S_sum=%.2f "
            "frac_pixels_floored=%.4f region_subtracted_frac G=%.4f Ks=%.4f -> %s"
            % (region, result["n_entering"], result["g_s_entering_sum"], frac_floored_pix,
               region_frac_g, region_frac_ks, path))


if __name__ == "__main__":
    run(build)
