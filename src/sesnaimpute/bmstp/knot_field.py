"""The young-star law convolved with the knot-driver displacement kernel,
as a map operation (SPEC_BMSTP_DRAFT.md sec. 5.6 "Sky density"): a knot
rides on `kappa_pooled . A_cloud^2` -- sec. 5.5's law, on the CLOUD's own
share of whichever column the arm carries (sec. 5.5: "the cloud's share
of the column"; the same `A_cloud = A . cloud_frac` every reader of the
law applies, never the raw column alone, since `eta_r`'s own calibration
(sec. 5.6's constant table) is formed against that same cloud-share-
squared denominator), at the SURVEY-POOLED coefficient `kappa_pooled`
rather than the region's own fitted `KAPPA_USED` (`population.knot_rate`'s
own calibration is formed at that pooled level; WP-POP-5's finding --
H2S's own law, never YSO's, whose density stays at the region's own
coefficient everywhere else) -- smeared by `K(r) ~ e^{-r/lambda}/r`, the
driver-to-knot separation kernel (Davis+2009, Walawender+2005). The
convolution runs once per region on the region's own HGBS column map, at
whatever coarsening keeps the kernel resolved and the map under a few
million pixels; `bmstp.density` and `bmstp.atlas` sample the result at
each Herschel-arm source/pixel (sec. 5.6: "the old per-source evaluation
of this convolution is gone"). Every downsampled cell the native map
touched at all is "covered": a cell nothing measured is filled with its
own nearest covered cell's convolved value (never left at the
unconvolved law), and `sample_at` extends that rule to a query position
outside the array itself, by projecting onto the array's own nearest
in-grid pixel -- so every Herschel-arm source, in the map's own footprint
or not, reads a genuine convolved value (sec. 5.6's brief). A Planck-arm
source/pixel never reaches this module: the kernel is sub-beam at
Planck's 5.03' resolution, so it takes the SAME `kappa_pooled`-evaluated
law at its own column instead (`law_count_at_kappa`, below;
`bmstp.density`/`bmstp.atlas`), UNCONVOLVED -- `bmstp.density`'s own
docstring states what that means for the arm-to-arm step."""

import time
import warnings

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.bmstp import sample_cloud as sample_cloud_module
from sesnaimpute.population import yso as yso_module
from sesnaimpute.sky.derived import herschel_column as herschel_column_module

#: The sightline grain's own HEALPix resolution, galactic NESTED
#: (`granules.build.NSIDE_256`, `sky/derived/profile.py`'s own `NSIDE`):
#: the pixelisation `cloud_column_fraction`'s per-map-pixel lookup shares
#: with every source's own `A_CLOUD_K`.
_NSIDE_SIGHTLINE = 256

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


def kappa_pooled(config):
    """`KAPPA_POOLED`, the survey-wide pooled young-star-law coefficient
    (`population/yso/law_yso_region.hdf5`'s own scalar attribute, the
    SAME value for every region -- `population.yso` sec. 6.1's "geometric
    mean over the fitted regions"). H2S's own law evaluation reads this,
    never a region's fitted `KAPPA_USED` (`yso_module._kappa_used`):
    the knot count scales with the column-squared integral calibrated at
    the knot survey's own anchors (`population.knot_rate.eta_for_region`'s
    own denominator, `region_predicted_yso`, is formed at the SAME pooled
    level by construction -- it reads `yso_module.law_area_integral`
    with `kappa=None`, which falls back to `KAPPA_USED`, but every
    region but the pooled one's own anchor is calibrated relative to it;
    WP-POP-5's finding), independent of which level the YSO class itself
    happens to be fitted at in that one region. YSO's own density is
    UNCHANGED by this: it still reads `_kappa_used`, via `population.yso.
    law_count`, exactly as before."""
    path = config_module.product_path(config, "population", "yso", "law", "region")
    with h5py.File(path, "r") as f:
        return float(f["KAPPA_POOLED"][()])


def law_count_at_kappa(config, region, a_col, provenance, kappa):
    """`N_law`, exactly `population.yso.law_count`'s own formula (the
    region's `(pc^2/deg^2)_r` factor, the Planck arm's kernel second
    moment) -- but at an EXPLICIT coefficient `kappa` rather than the
    region's own fitted `KAPPA_USED`. H2S's own law evaluation calls this
    at `kappa_pooled(config)`, on both arms, so the ratio `eta_r` is
    itself calibrated against (`population.knot_rate`) never picks up a
    second, uncancelled region-specific factor; YSO's own `law_count`
    call elsewhere is untouched."""
    a_col = np.asarray(a_col, dtype=float)
    provenance = np.asarray(provenance)
    d_r_pc = regions_module.REGIONS_BY_NAME[region].d_r_pc
    pc2 = yso_module.pc2_per_deg2(d_r_pc)
    herschel_value = kappa * pc2 * a_col ** 2
    planck_value = kappa * pc2 * yso_module._kernel_second_moment_planck(config, a_col)
    return np.where(provenance == yso_module.PROVENANCE_HERSCHEL, herschel_value, planck_value)


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
    this map's own footprint) and is handled as such by the caller. Reads
    the gas column product: the knot field convolves the young-star law,
    which was measured on the gas column, not extinction."""
    path = config_module.product_path(config, "sky/derived", "adopted", "column", "source", region=region)
    with h5py.File(path, "r") as f:
        names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in f["MAP_NAME"][:]]
        if not names:
            return None
        map_id = np.asarray(f["HERSCHEL_MAP_ID"][:], dtype=np.int64)
    counts = np.bincount(map_id[map_id >= 0], minlength=len(names))
    return names[int(np.argmax(counts))]


def cloud_column_fraction(config, region):
    """`A_cloud(sightline) / A_s`, the cloud's own share of the sightline's
    column (SPEC_BMSTP_DRAFT.md sec. 5.5 "Sky density"): `1 - u(d_front)`,
    `u(d) = A_CUM_K / A_INF_K` (`sky/derived/edenhofer/profile/sightline`'s
    own 3-D-map profile) interpolated linearly in distance at the region's
    cloud-interval front edge, `d_front` (`bmstp.sample_cloud.
    cloud_interval_pc`). The ONE place this fraction is measured: every
    source's own `A_CLOUD_K` (`bmstp.density`) and the knot field this
    module convolves (below) both read it from here, so the cloud's share
    of the column is one number, read twice, never two measurements that
    could drift apart.

    Returns `(cloud_frac, sightline_axis, d_front)`: `cloud_frac` and
    `sightline_axis` share one row order, the profile product's own
    `HPX_PIX_256` axis (sorted ascending -- `sky/derived/profile.py`'s own
    `_admitted_sightlines` sorts it, and `bmstp.shapes.build_cloud` writes
    P3's own sightline grain straight from this same order)."""
    path = config_module.product_path(
        config, "sky/derived", "edenhofer", "profile", "sightline", region=region)
    with h5py.File(path, "r") as f:
        sightline_axis = np.asarray(f["HPX_PIX_256"][:], dtype=np.int64)
        dist_pc = np.asarray(f["DIST_PC"][:], dtype=np.float64)
        a_cum_k = np.asarray(f["A_CUM_K"][:], dtype=np.float64)
        a_inf_k = np.asarray(f["A_INF_K"][:], dtype=np.float64)
    u_of_d = a_cum_k / a_inf_k[:, None]  # (n_sl, n_d): u(DIST_PC[j]) per sightline
    d_front, _d_back = sample_cloud_module.cloud_interval_pc(config, region)
    j = int(np.clip(np.searchsorted(dist_pc, d_front), 1, dist_pc.size - 1))
    d0, d1 = dist_pc[j - 1], dist_pc[j]
    frac_j = (d_front - d0) / (d1 - d0) if d1 > d0 else 0.0
    u_front = u_of_d[:, j - 1] + frac_j * (u_of_d[:, j] - u_of_d[:, j - 1])
    return 1.0 - u_front, sightline_axis, d_front


def _cloud_frac_on_grid(wcs, shape, sightline_axis, cloud_frac_by_sightline):
    """Each grid cell's own cloud fraction (`cloud_column_fraction`,
    above), looked up at the cell centre's galactic nside-256 NESTED
    HEALPix pixel -- the SAME pixelisation `granules.build` assigns every
    catalogued source (`_NSIDE_SIGHTLINE`), so a map cell and a source
    sitting in the same admitted sightline carry the identical fraction.
    A cell whose sightline is not admitted (outside the survey's own
    footprint, e.g. the HGBS mosaic's own edge beyond the region's
    admitted pixels) carries no cloud at all here: zero, the same
    "unmeasured is not a real number" rule `_block_mean` already applies
    to the raw column."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = wcs.pixel_to_world(xx.ravel(), yy.ravel())
    gal = world.galactic
    pix = hp.ang2pix(_NSIDE_SIGHTLINE, gal.l.deg, gal.b.deg, nest=True, lonlat=True).astype(np.int64)
    row = np.searchsorted(sightline_axis, pix)
    capped = np.minimum(row, sightline_axis.size - 1)
    found = sightline_axis.size > 0 and (sightline_axis[capped] == pix)
    frac = np.where(found, cloud_frac_by_sightline[capped], 0.0)
    return frac.reshape(ny, nx)


def _fill_uncovered(values, covered):
    """`values`, with every cell `covered` is False at replaced by its
    own nearest COVERED cell's value (`scipy.ndimage.distance_transform_
    edt`'s standard nearest-feature-transform idiom, vectorised, rule 8):
    sec. 5.6's brief, "a source outside the convolved map's footprint
    takes the convolved value of its nearest covered cell, never the
    unconvolved law silently." A no-op where every cell is covered, or
    where none is (nothing to borrow from)."""
    if covered.all() or not covered.any():
        return values
    _dist, nearest_idx = distance_transform_edt(~covered, return_distances=True, return_indices=True)
    return values[tuple(nearest_idx)]


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
    `kappa_pooled . (pc^2/deg^2)_r . A_cloud^2` -- sec. 5.5's law, on
    the CLOUD's own share of the column (`cloud_column_fraction`, the
    SAME fraction every source's own `A_CLOUD_K` carries), already in
    the sky density's own deg^-2 units so a consumer can use it exactly
    where it uses `DENSITY_YSO` -- downsampled, cloud-shared and
    convolved once with the knot-driver kernel above, then filled so
    every downsampled cell the map's own footprint reaches holds a
    genuinely convolved value (`_fill_uncovered`, sec. 5.6's brief: never
    the unconvolved law silently). This map is H2S's OWN, never read for
    YSO's own density: it is evaluated at the SURVEY-POOLED coefficient
    `kappa_pooled`, not the region's own fitted `KAPPA_USED`, because the
    knot count scales with the column-squared integral `eta_r` is
    calibrated against at the pooled level (WP-POP-5's finding;
    `kappa_pooled`, above). `(None, None, None)` if no HGBS map serves
    the region (an all-Planck-arm region). `meta` carries the map name,
    the downsampling factor, the kernel's radius and width in
    downsampled pixels, the convolution's wall time, the covered
    fraction before the nearest-cell fill, and the pre-/post-convolution
    totals (deg^-2 x deg^2 = a count) for the report's conservation
    identity (sec. 5.6's brief: "the kernel conserves the law's total")."""
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

    # The law on the map's own native grid, sec. 5.5: `kappa_pooled *
    # A_K^2`, carried straight to deg^-2 units (the same region-constant
    # pc^2/deg^2 factor `bmstp.density`'s DENSITY_YSO uses, so the
    # convolved map is directly comparable to it) but at the SURVEY-
    # POOLED coefficient, never the region's own `KAPPA_USED` -- H2S's
    # own law, not YSO's (see this function's own docstring,
    # `kappa_pooled`, above). Unmeasured pixels (`herschel_column`'s own
    # "on" test: finite and positive) contribute zero, not a fabricated
    # column.
    valid = np.isfinite(data) & (data > 0)
    a_k = np.where(valid, data.astype(np.float64) * herschel_column_module.NH2_TO_AK, 0.0)
    pc2 = float(yso_module.pc2_per_deg2(d_r_pc))
    law_native = kappa_pooled(config) * pc2 * a_k ** 2

    pc_per_arcsec = (np.pi / 180.0 / 3600.0) * d_r_pc
    lambda_arcsec = LAMBDA_PC / pc_per_arcsec
    lambda_px_native = lambda_arcsec / pixscale_arcsec_native
    f = _downsample_factor(data.shape[0], data.shape[1], lambda_px_native)

    law_ds = _block_mean(law_native, f)
    # covered: did ANY native pixel of this downsampled block carry a
    # measured column at all -- the footprint `_fill_uncovered` resolves
    # against, below.
    covered_ds = _block_mean(valid.astype(np.float64), f) > 0.0
    pixscale_arcsec_ds = pixscale_arcsec_native * f
    pixscale_pc_ds = pixscale_arcsec_ds * pc_per_arcsec
    kernel, kernel_rad_px = _kernel(pixscale_pc_ds, TRUNCATE_N_LAMBDA * LAMBDA_PC)

    wcs_ds = wcs.deepcopy()
    # Standard block-downsampling WCS transform (e.g. SWarp/reproject's
    # own convention): CRPIX shifts so pixel q's centre lands on the
    # centre of the f-pixel block it averages; CDELT scales by f.
    wcs_ds.wcs.crpix = (wcs.wcs.crpix - 0.5) / f + 0.5
    wcs_ds.wcs.cdelt = wcs.wcs.cdelt * f

    # sec. 5.5's "Sky density": the law is applied to the CLOUD'S own
    # share of the column, never the raw column alone -- `eta_r`'s own
    # calibration (sec. 5.6's constant table, `population.knot_rate`) is
    # formed against that same cloud-share-squared denominator, so the
    # map this module convolves must carry it too, or the two would
    # disagree on what "the law" means. Looked up at the DOWNSAMPLED
    # grid (the fraction is a nside-256, ~0.05 deg^2 quantity, far
    # coarser than either the native or the downsampled pixel, so
    # squaring-then-block-averaging and block-averaging-then-squaring
    # agree to the fraction's own smoothness).
    cloud_frac_by_sl, sightline_axis, _d_front = cloud_column_fraction(config, region)
    cloud_frac_ds = _cloud_frac_on_grid(wcs_ds, law_ds.shape, sightline_axis, cloud_frac_by_sl)
    law_ds = law_ds * cloud_frac_ds ** 2

    t0 = time.time()
    convolved = fftconvolve(law_ds, kernel, mode="same")
    # `law_ds` is non-negative (kappa_pooled . A_cloud^2, with unmeasured
    # pixels contributing zero, not a fabricated column) and `K(r) ~
    # e^{-r/lambda}/r` is non-negative, so the convolution of the two is
    # non-negative everywhere in exact arithmetic. Any negative value
    # here is FFT round-off -- measured at -1.9e-13 against a map scale
    # of order 1 -- and it reaches `bmstp.density` as `DENSITY_H2S =
    # L(s) . eta_r . eps_ext . ON_GRID_H2S`, where `fittp.prior_reader.
    # ln_prior`'s `log(density)` turns it into a NaN that takes the
    # source's whole classification with it (3,997 sources in Orion A,
    # 1,644 in Aquila). Clamping removes the round-off, not signal; a
    # genuinely zero knot density is physical (owner's ruling
    # 2026-09-16) and reads as -inf, which contributes nothing to the
    # evidence instead of poisoning it.
    np.maximum(convolved, 0.0, out=convolved)
    covered_frac_before_fill = float(np.mean(covered_ds))
    # sec. 5.6's brief: "a source outside the convolved map's footprint
    # takes the convolved value of its nearest covered cell, never the
    # unconvolved law silently" -- resolved HERE, once per region, so
    # every reader (`sample_at`, `mean_over_area`) sees one complete
    # map, with no uncovered cell left in it.
    convolved = _fill_uncovered(convolved, covered_ds)
    wall_s = time.time() - t0

    pix_area_ds_deg2 = (pixscale_arcsec_ds / 3600.0) ** 2
    total_before = float(np.sum(law_ds)) * pix_area_ds_deg2
    total_after = float(np.sum(np.where(covered_ds, convolved, 0.0))) * pix_area_ds_deg2
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
        covered_frac=covered_frac_before_fill,
    )
    return convolved, wcs_ds, meta


def sample_at(law_map, wcs, lon_deg, lat_deg, frame="icrs"):
    """Nearest-pixel sample of a `convolved_law` map at each `(lon,
    lat)`, `frame` any astropy-recognised frame (`world_to_pixel`
    transforms it into the map's own WCS frame). Every in-grid cell of
    `law_map` already carries a genuinely convolved value
    (`convolved_law`'s own nearest-covered-cell fill), so a position
    OUTSIDE the grid is projected onto its nearest in-grid pixel --
    clamping each axis independently, the Euclidean-nearest point a
    rectangular grid can represent -- rather than returned as NaN (sec.
    5.6's brief: "a source outside the convolved map's footprint takes
    the convolved value of its nearest covered cell, never the
    unconvolved law silently")."""
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    if lon_deg.size == 0:
        return np.full(lon_deg.shape, np.nan, dtype=np.float64)
    x, y = wcs.world_to_pixel(SkyCoord(lon_deg * u.deg, lat_deg * u.deg, frame=frame))
    ny, nx = law_map.shape
    xi = np.clip(np.rint(x).astype(np.int64), 0, nx - 1)
    yi = np.clip(np.rint(y).astype(np.int64), 0, ny - 1)
    return law_map[yi, xi]


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
