"""The young-star law convolved with the knot-driver displacement kernel,
as a map operation (SPEC_BMSTP_DRAFT.md sec. 5.6 "Sky density"): a knot
rides on `kappa_arm . A^2` (sec. 5.5's law) smeared by `K(r) ~
e^{-r/lambda}/r`, the driver-to-knot separation kernel (Davis+2009,
Walawender+2005). The convolution runs once per region on the region's
own HGBS column map, at whatever coarsening keeps the kernel resolved and
the map under a few million pixels; `bmstp.density` and `bmstp.atlas`
sample the result at each Herschel-arm source/pixel (sec. 5.6: "the old
per-source evaluation of this convolution is gone"). A Planck-arm
source/pixel never reaches this module: the kernel is sub-beam at
Planck's 5.03' resolution, so it takes the law at its own column instead
(sec. 5.6, `bmstp.density`/`bmstp.atlas`).
"""

import time
import warnings

import h5py
import numpy as np
from scipy.signal import fftconvolve

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.population import yso as yso_module
from sesnaimpute.sky.derived import herschel_column as herschel_column_module

#: The knot-driver displacement kernel's scale length: a maximum-
#: likelihood fit to knot-to-protostar separations in Orion A and
#: Perseus (SPEC_BMSTP_DRAFT.md sec. 5.6).
LAMBDA_PC = 0.32
#: The kernel is truncated here, in units of LAMBDA_PC (sec. 5.6's brief).
TRUNCATE_N_LAMBDA = 5.0
#: The map is downsampled until the kernel scale spans at least this many
#: pixels (sec. 5.6's brief).
MIN_KERNEL_PIXELS = 4.0
#: ...and until the map holds at most this many pixels (sec. 5.6's brief).
MAX_MAP_PIXELS = 4_000_000


def _serving_map_name(config, region):
    """The HGBS map name reaching the most Herschel-arm sources of
    `region`, from the adopted column product's own MAP_NAME/
    HERSCHEL_MAP_ID (`sky.derived.column`'s merge of
    `sky.derived.herschel_column`'s per-source lookup -- reused, not
    re-derived); `None` if no map serves the region at all (an all-
    Planck-arm region, e.g. NGC 7129). A region whose sources split
    across more than one HGBS reduction (Aquila, Cepheus Flare,
    Chameleon, Lupus) convolves only the dominant one; a source served
    by another map there is an edge case (its position falls outside
    this map's own footprint) and is handled as such by the caller."""
    path = config_module.product_path(config, "sky/derived", "adopted", "column", "source", region=region)
    with h5py.File(path, "r") as f:
        names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in f["MAP_NAME"][:]]
        if not names:
            return None
        map_id = np.asarray(f["HERSCHEL_MAP_ID"][:], dtype=np.int64)
    counts = np.bincount(map_id[map_id >= 0], minlength=len(names))
    return names[int(np.argmax(counts))]


def _downsample_factor(ny, nx, lambda_px_native):
    """The integer coarsening factor: the smallest `f` bringing the map
    under `MAX_MAP_PIXELS` pixels, checked against the floor that the
    kernel scale still spans `MIN_KERNEL_PIXELS` pixels at that
    coarsening (sec. 5.6's brief) -- the size cap decides in practice,
    since the kernel (0.32 pc) is far wider than one HGBS beam."""
    f_size = max(1, int(np.ceil(np.sqrt((ny * nx) / float(MAX_MAP_PIXELS)))))
    f_kernel_max = max(1, int(np.floor(lambda_px_native / MIN_KERNEL_PIXELS)))
    if f_size > f_kernel_max:
        raise ValueError(
            "bmstp.knot_field: no integer downsampling factor keeps the map under "
            "%d pixels while leaving the %.1f pc kernel %.1f pixels wide (native "
            "kernel width %.1f px)" % (MAX_MAP_PIXELS, LAMBDA_PC, MIN_KERNEL_PIXELS, lambda_px_native))
    return f_size


def _block_mean(a, f):
    """`a` downsampled by integer factor `f` in both axes, plain block
    mean (the map is already zero-filled at invalid pixels, sec. 5.6:
    unmeasured column is not a real number to average); trims any
    remainder row/column so the array divides evenly by `f`."""
    if f == 1:
        return a
    ny, nx = a.shape
    ny2, nx2 = (ny // f) * f, (nx // f) * f
    return a[:ny2, :nx2].reshape(ny2 // f, f, nx2 // f, f).mean(axis=(1, 3))


def _kernel(pixscale_pc, truncate_pc):
    """`K(r) ~ e^{-r/lambda}/r`, sampled at pixel centres on a square
    grid out to `truncate_pc` (sec. 5.6's brief: 5 lambda), the `1/r`
    singularity regularised by flooring `r` at half a pixel width, then
    normalised so the discrete grid sums to one (unit integral)."""
    rad_px = int(np.ceil(truncate_pc / pixscale_pc))
    yy, xx = np.mgrid[-rad_px:rad_px + 1, -rad_px:rad_px + 1]
    r_px = np.hypot(yy, xx)
    r_pc = np.maximum(r_px * pixscale_pc, 0.5 * pixscale_pc)
    k = np.exp(-r_pc / LAMBDA_PC) / r_pc
    k[r_px * pixscale_pc > truncate_pc] = 0.0
    k /= k.sum()
    return k, rad_px


def convolved_law(config, region):
    """`(law_map, wcs, meta)`: the region's dominant HGBS column map, as
    `kappa_Herschel . (pc^2/deg^2)_r . A_K^2` (sec. 5.5's law, already in
    the sky density's own deg^-2 units so a consumer can use it exactly
    where it uses `DENSITY_YSO`), downsampled and convolved once with
    the knot-driver kernel above. `(None, None, None)` if no HGBS map
    serves the region (an all-Planck-arm region). `meta` carries the
    map name, the downsampling factor, the kernel's radius and width in
    downsampled pixels, the convolution's wall time, and the pre-/post-
    convolution totals (deg^-2 x deg^2 = a count) for the report's
    conservation identity (sec. 5.6's brief: "the kernel conserves the
    law's total")."""
    reg = regions_module.REGIONS_BY_NAME[region]
    d_r_pc = float(reg.d_r_pc)
    name = _serving_map_name(config, region)
    if name is None:
        return None, None, None

    maps_by_name = {m["name"]: m["path"] for m in herschel_column_module._map_list(config)}
    if name not in maps_by_name:
        raise FileNotFoundError(
            "bmstp.knot_field: HGBS map %r (region %r's own adopted-column provenance) "
            "is not among the fetched HGBS files" % (name, region))
    data, wcs, pixscale_arcsec_native = herschel_column_module._open_hgbs_map(maps_by_name[name])

    # The law on the map's own native grid, sec. 5.5: `kappa_Herschel *
    # A_K^2`, carried straight to deg^-2 units (the same region-constant
    # pc^2/deg^2 factor `bmstp.density`'s DENSITY_YSO uses) so the
    # convolved map is directly comparable to it; unmeasured pixels
    # (`herschel_column`'s own "on" test: finite and positive) contribute
    # zero, not a fabricated column.
    valid = np.isfinite(data) & (data > 0)
    a_k = np.where(valid, data.astype(np.float64) * herschel_column_module.NH2_TO_AK, 0.0)
    pc2 = float(yso_module.pc2_per_deg2(d_r_pc))
    law_native = yso_module.KAPPA_HERSCHEL * pc2 * a_k ** 2

    pc_per_arcsec = (np.pi / 180.0 / 3600.0) * d_r_pc
    lambda_arcsec = LAMBDA_PC / pc_per_arcsec
    lambda_px_native = lambda_arcsec / pixscale_arcsec_native
    f = _downsample_factor(data.shape[0], data.shape[1], lambda_px_native)

    law_ds = _block_mean(law_native, f)
    pixscale_arcsec_ds = pixscale_arcsec_native * f
    pixscale_pc_ds = pixscale_arcsec_ds * pc_per_arcsec
    kernel, kernel_rad_px = _kernel(pixscale_pc_ds, TRUNCATE_N_LAMBDA * LAMBDA_PC)

    t0 = time.time()
    convolved = fftconvolve(law_ds, kernel, mode="same")
    wall_s = time.time() - t0

    wcs_ds = wcs.deepcopy()
    # Standard block-downsampling WCS transform (e.g. SWarp/reproject's
    # own convention): CRPIX shifts so pixel q's centre lands on the
    # centre of the f-pixel block it averages; CDELT scales by f.
    wcs_ds.wcs.crpix = (wcs.wcs.crpix - 0.5) / f + 0.5
    wcs_ds.wcs.cdelt = wcs.wcs.cdelt * f

    pix_area_ds_deg2 = (pixscale_arcsec_ds / 3600.0) ** 2
    total_before = float(np.sum(law_ds)) * pix_area_ds_deg2
    total_after = float(np.sum(convolved)) * pix_area_ds_deg2
    conservation_rel_err = (abs(total_after - total_before) / total_before
                             if total_before > 0 else float("nan"))

    meta = dict(
        map_name=name, downsample_factor=f,
        shape_native=tuple(int(v) for v in data.shape),
        shape_ds=tuple(int(v) for v in law_ds.shape),
        kernel_radius_px=int(kernel_rad_px),
        kernel_width_px=float(LAMBDA_PC / pixscale_pc_ds),
        pixscale_arcsec_ds=float(pixscale_arcsec_ds),
        wall_s=float(wall_s),
        total_before=total_before, total_after=total_after,
        conservation_rel_err=conservation_rel_err,
    )
    return convolved, wcs_ds, meta


def sample_at(law_map, wcs, lon_deg, lat_deg, frame="icrs"):
    """Nearest-pixel sample of a `convolved_law` map at each `(lon,
    lat)`, `frame` any astropy-recognised frame (`world_to_pixel`
    transforms it into the map's own WCS frame); NaN wherever a position
    falls outside the map's own pixel grid -- an edge case, sec. 5.6's
    brief ("a Herschel-arm source falling outside the convolved map")."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    out = np.full(lon_deg.shape, np.nan, dtype=np.float64)
    if lon_deg.size == 0:
        return out
    x, y = wcs.world_to_pixel(SkyCoord(lon_deg * u.deg, lat_deg * u.deg, frame=frame))
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    ny, nx = law_map.shape
    inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    out[inside] = law_map[yi[inside], xi[inside]]
    return out


def mean_over_area(law_map, wcs, lon_deg, lat_deg, width_deg, frame="icrs", n_sub=3):
    """The mean of a `convolved_law` map over an approximately
    `width_deg` x `width_deg` patch centred at each `(lon, lat)` (sec.
    5.6 item 3: "the mean of L over the pixel's area"): an `n_sub x
    n_sub` grid of tangent-plane sub-positions per patch (the longitude
    offset scaled by `1/cos(lat)`), sampled in one vectorised
    `sample_at` call over every position at once (rule 8 -- no loop over
    pixels). NaN only where every sub-position of a patch falls outside
    the map (a fully edge pixel)."""
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    out = np.full(lon_deg.shape, np.nan, dtype=np.float64)
    if lon_deg.size == 0:
        return out
    offsets = (np.arange(1, n_sub + 1) / (n_sub + 1) - 0.5) * width_deg
    doff_x, doff_y = np.meshgrid(offsets, offsets)
    doff_x = doff_x.ravel()[None, :]
    doff_y = doff_y.ravel()[None, :]
    cos_lat = np.maximum(np.cos(np.deg2rad(lat_deg))[:, None], 1.0e-6)
    lon_sub = (lon_deg[:, None] + doff_x / cos_lat).ravel()
    lat_sub = (lat_deg[:, None] + doff_y).ravel()
    sampled = sample_at(law_map, wcs, lon_sub, lat_sub, frame=frame).reshape(lon_deg.size, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(sampled, axis=1)
