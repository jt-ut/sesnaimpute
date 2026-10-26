"""Same-band overlap ownership and a shared detection/calibration fixed point.

Calibrated map inputs are explicit: this module does not infer a PSF, beam,
or extraction stencil from a band name. Maps remain open in the caller.
"""
from dataclasses import dataclass, replace
import time
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.spatial import cKDTree

from .constants import DETECTION_SNR, N_POS
from .geometry import MapGrid, DetectedPacsNeighbours, sample_cells, select_deeper
from .stage import SingletonStage, usable_xy, measure_xy
from .completeness import calibrate_cell_scatter


@dataclass(frozen=True)
class MapInput:
    opened: object
    psf: object
    exact_psf: object
    half_size: int
    annulus_arcsec: tuple
    min_coverage: float
    beam_arcsec: float


@dataclass(frozen=True)
class RegionalStage:
    maps: tuple
    stages: tuple
    source_map: np.ndarray
    flux_mjy: np.ndarray
    sigma_mjy: np.ndarray
    detected: np.ndarray
    iterations: int
    seconds: float


def select_sources(stages):
    """Select total-sigma minima; input stages must follow sorted map IDs."""
    if not stages:
        raise ValueError('At least one map stage is required')
    n = stages[0].covered.size
    owner = np.full(n, -1, np.int32)
    sigma = np.full(n, np.inf)
    flux = np.full(n, np.nan)
    indices = np.arange(n)
    for i, stage in enumerate(stages):
        if stage.covered.shape != (n,):
            raise ValueError('All maps require the same candidate row order')
        invalid = select_deeper(i, indices, stage.sigma_mjy, stage.covered, owner, sigma)
        if invalid:
            raise RuntimeError('Covered source has invalid calibrated uncertainty')
        owned = owner == i
        flux[owned] = stage.flux_mjy[owned]
    detected = (owner >= 0) & (flux >= DETECTION_SNR * sigma)
    return owner, flux, sigma, detected


def run_regional_fixed_point(map_inputs, candidates, *, seed, area_draws,
                             max_draws, max_iterations, max_seconds,
                             config=None, n_jobs=1, batch_size=200,
                             n_positions=N_POS):
    """Calibrate every map against one mask; freeze mask AND map ownership.

Only changed detected-neighbour lists invalidate a cell's calibration. Source
flux and propagated pixel errors are measured once per covering map. Switching
ownership cannot create an independently frozen per-map detection catalogue.
"""
    maps = tuple(sorted(map_inputs, key=lambda m: m.opened.record.map_id))
    if not maps or len({m.opened.record.map_id for m in maps}) != len(maps):
        raise ValueError('Distinct maps are required')
    if len({m.opened.record.band for m in maps}) != 1:
        raise ValueError('Overlap ownership is defined separately within each band')
    if max_iterations < 1 or max_seconds <= 0 or n_positions < 2:
        raise ValueError('Positive budgets and at least two samples required')
    if any(not np.isfinite(m.beam_arcsec) or m.beam_arcsec <= 0 for m in maps):
        raise ValueError('Every map requires an explicit calibrated positive beam')
    if n_jobs > 1 and config is None:
        raise ValueError('Parallel calibration requires config')
    started = time.monotonic()
    deadline = started + max_seconds
    run_key = str(time.time_ns())
    sky = SkyCoord(candidates.ra_deg*u.deg, candidates.dec_deg*u.deg, frame='icrs')
    vectors = sky.cartesian.xyz.value.T
    states = []
    for m in maps:
        grid = MapGrid.from_wcs(m.opened.wcs, m.opened.image.shape)
        x, y = m.opened.wcs.world_to_pixel(sky)
        covered = usable_xy(m.opened, x, y, half_size=m.half_size, min_coverage=m.min_coverage)
        ids = grid.cell_ids(x, y)
        cells = np.unique(ids[covered])
        rows = np.flatnonzero(covered)
        f, p, condition = (np.full(x.size, np.nan) for _ in range(3))
        f[rows], p[rows], condition[rows] = measure_xy(
            m.opened, x[rows], y[rows], m.psf, half_size=m.half_size,
            annulus_arcsec=m.annulus_arcsec, min_coverage=m.min_coverage, batch_size=batch_size)
        x0, x1, y0, y1 = grid.bounds(cells)
        centre = grid.wcs.pixel_to_world((x0+x1)/2, (y0+y1)/2).icrs
        radius = np.zeros(cells.size)
        for xx, yy in ((x0,y0),(x0,y1),(x1,y0),(x1,y1)):
            radius = np.maximum(radius, centre.separation(grid.wcs.pixel_to_world(xx,yy).icrs).arcsec)
        chord = 2*np.sin(np.deg2rad((radius+m.beam_arcsec+1e-6)/3600)/2)
        states.append(dict(grid=grid, covered=covered, ids=ids, cells=cells, rows=rows,
                           flux=f, pixel=p, condition=condition, centres=centre.cartesian.xyz.value.T,
                           chord=chord, previous=[None]*cells.size,
                           mean=np.empty(cells.size), scatter=np.empty(cells.size),
                           rms=np.empty(cells.size), count=np.empty(cells.size,np.int64)))
        print('PACS regional measured', m.opened.record.map_id, rows.size, 'sources', flush=True)
    detected = np.zeros(sky.size, bool)
    previous_owner = np.full(sky.size, -1, np.int32)
    history = set()
    for iteration in range(1, max_iterations+1):
        if time.monotonic() > deadline:
            raise RuntimeError('Regional overlap calibration time budget exhausted')
        neighbours = DetectedPacsNeighbours(candidates, detected)
        detected_rows = np.flatnonzero(detected)
        tree = cKDTree(vectors[detected_rows]) if detected_rows.size else None
        stages = []
        for m, s in zip(maps, states):
            cells = s['cells']
            lists = tree.query_ball_point(s['centres'], s['chord']) if tree is not None else [[] for _ in cells]
            local = [np.sort(detected_rows[np.asarray(v,np.int64)]) for v in lists]
            update = np.asarray([i for i in range(cells.size) if s['previous'][i] is None or
                                 not np.array_equal(s['previous'][i],local[i])],np.int64)
            if n_jobs > 1 and update.size:
                from .regional_workers import iterate_cell_calibrations
                stream = iterate_cell_calibrations(
                    config,m.opened.record,m.psf,cells[update], candidates.ra_deg[detected],
                    candidates.dec_deg[detected],n_jobs=n_jobs,run_key=run_key,pass_key=iteration,
                    seed=seed,area_draws=area_draws,max_draws=max_draws,half_size=m.half_size,
                    annulus_arcsec=m.annulus_arcsec,min_coverage=m.min_coverage,
                    beam_arcsec=m.beam_arcsec,n_positions=n_positions,batch_size=batch_size,deadline=deadline)
                for piece in stream:
                    at = np.searchsorted(cells,piece.cell_ids)
                    for key, value in (('mean',piece.mean_mjy),('scatter',piece.scatter_mjy),
                                       ('rms',piece.pixel_sigma_rms_mjy),('count',piece.n_positions)):
                        s[key][at] = value
                    for i in at:
                        s['previous'][i] = local[i].copy()
            else:
                def usable(xx, yy):
                    return usable_xy(m.opened,xx,yy,half_size=m.half_size,min_coverage=m.min_coverage)
                for i in update:
                    if time.monotonic() > deadline:
                        raise RuntimeError('Regional overlap calibration time budget exhausted')
                    sample = sample_cells(s['grid'],cells[i:i+1],neighbours,m.beam_arcsec,usable,
                                          seed=seed,area_draws=area_draws,max_draws=max_draws,n_positions=n_positions)
                    f,p,_ = measure_xy(m.opened,sample.x.ravel(),sample.y.ravel(),m.psf,
                                       half_size=m.half_size,annulus_arcsec=m.annulus_arcsec,
                                       min_coverage=m.min_coverage,batch_size=batch_size)
                    stat = calibrate_cell_scatter(f[None],p[None],np.ones((1,f.size),bool))
                    s['mean'][i],s['scatter'][i],s['rms'][i],s['count'][i] = (
                        stat.mean_mjy[0],stat.scatter_mjy[0],stat.pixel_sigma_rms_mjy[0],stat.n_positions[0])
                    s['previous'][i] = local[i].copy()
            sigma = np.full(sky.size,np.nan)
            rows = s['rows']
            sigma[rows] = np.hypot(s['pixel'][rows],s['scatter'][np.searchsorted(cells,s['ids'][rows])])
            stages.append(SingletonStage(candidates,s['covered'],s['ids'],s['flux'],s['pixel'],sigma,
                s['condition'],detected,cells,s['mean'],s['scatter'],s['rms'],s['count'],
                iteration,time.monotonic()-started,int(seed)))
        owner, flux, sigma, new = select_sources(stages)
        print('PACS regional pass',iteration,'detected',new.sum(),'mask changes',np.count_nonzero(new!=detected),
              'ownership changes',np.count_nonzero(owner!=previous_owner),flush=True)
        if np.array_equal(new,detected) and np.array_equal(owner,previous_owner):
            for stage in stages:
                for value in vars(stage).values():
                    if isinstance(value,np.ndarray):
                        value.flags.writeable = False
            for value in (owner,flux,sigma,new):
                value.flags.writeable = False
            return RegionalStage(maps,tuple(stages),owner,flux,sigma,new,iteration,time.monotonic()-started)
        signature = (np.packbits(new).tobytes(),owner.tobytes())
        if signature in history:
            raise RuntimeError('Regional ownership/detection cycle; no forced convergence')
        history.add(signature)
        detected, previous_owner = new, owner
    raise RuntimeError('Regional overlap calibration iteration budget exhausted')


def select_sample_maps(regional, ra_deg, dec_deg, calibration_cells, cell_scatter,
                       *, curve_lookups=None, batch_size=200):
    """Own sky samples by total sigma, then link to that exact map's curve.

Calibration arrays cover every usable queried cell, including cells without
catalogue sources. Curves are conditional on the winning map's exclusion
radius. Excluded samples may lack a row; accepted samples must have one.
Returns owner, sigma, native cell ID, excluded mask and optional curve row.
The same routine applies to flattened 256-child HEALPix samples.
"""
    from .geometry import rows_for_cells
    ra, dec = np.asarray(ra_deg,float), np.asarray(dec_deg,float)
    if ra.ndim != 1 or dec.shape != ra.shape:
        raise ValueError('Matching one-dimensional sky arrays required')
    if len(calibration_cells) != len(regional.maps) or len(cell_scatter) != len(regional.maps):
        raise ValueError('Provide calibration arrays in regional map order')
    if curve_lookups is not None and len(curve_lookups) != len(regional.maps):
        raise ValueError('Provide curve lookups in regional map order')
    sky = SkyCoord(ra*u.deg,dec*u.deg,frame='icrs')
    owner = np.full(ra.size,-1,np.int32)
    sigma = np.full(ra.size,np.inf)
    cells = np.full(ra.size,-1,np.int64)
    for i,m in enumerate(regional.maps):
        x,y = m.opened.wcs.world_to_pixel(sky)
        covered = usable_xy(m.opened,x,y,half_size=m.half_size,min_coverage=m.min_coverage)
        rows = np.flatnonzero(covered)
        grid = MapGrid.from_wcs(m.opened.wcs,m.opened.image.shape)
        ids = grid.cell_ids(x[rows],y[rows])
        cc, ss = np.asarray(calibration_cells[i]),np.asarray(cell_scatter[i],float)
        if cc.ndim != 1 or ss.shape != cc.shape or np.any(np.diff(cc)<=0):
            raise ValueError('Map calibration cells must be sorted and unique')
        if np.any(~np.isfinite(ss)|(ss<0)):
            raise ValueError('Map cell scatter must be measured finite nonnegative values')
        at = np.searchsorted(cc,ids)
        if np.any(at>=cc.size) or not np.array_equal(cc[at],ids):
            raise RuntimeError('Usable overlap sample lacks calibrated native cell')
        _,pixel,_ = measure_xy(m.opened,x[rows],y[rows],m.psf,half_size=m.half_size,
                               annulus_arcsec=m.annulus_arcsec,min_coverage=m.min_coverage,
                               batch_size=batch_size)
        invalid = select_deeper(i,rows,np.hypot(pixel,ss[at]),np.ones(rows.size,bool),owner,sigma)
        if invalid:
            raise RuntimeError('Usable overlap sample has invalid total uncertainty')
        take = owner[rows]==i
        cells[rows[take]] = ids[take]
    excluded = np.zeros(ra.size,bool)
    curve_rows = np.full(ra.size,-1,np.int32) if curve_lookups is not None else None
    neighbours = DetectedPacsNeighbours(regional.stages[0].candidates,regional.detected)
    for i,m in enumerate(regional.maps):
        rows = np.flatnonzero(owner==i)
        excluded[rows] = neighbours.excluded(ra[rows],dec[rows],m.beam_arcsec)
        if curve_rows is not None:
            accepted = rows[~excluded[rows]]
            curve_rows[accepted] = rows_for_cells(cells[accepted],curve_lookups[i],np.ones(accepted.size,bool))
    return owner,sigma,cells,excluded,curve_rows


def regional_source_arrays(regional, catalogue_filename, n_source):
    """Assemble only owned source rows; retain the global first-pass mask.

Returned mappings are ownership-filtered for applying joint group results:
foreign-map nuisance members never overwrite their owning map's photometry.
"""
    from .photometry import source_product_arrays
    result = None
    mappings = []
    for i,(m,stage) in enumerate(zip(regional.maps,regional.stages)):
        owned = regional.source_map==i
        subset = replace(stage,covered=stage.covered&owned)
        arrays,mapping = source_product_arrays(subset,catalogue_filename,n_source,m.opened.record)
        mapping = mapping.copy()
        mapping[~owned] = -1
        rows = mapping[mapping>=0]
        mappings.append(mapping)
        if result is None:
            result = arrays
        else:
            for key,value in arrays.items():
                if value.dtype.kind=='S' and value.dtype.itemsize>result[key].dtype.itemsize:
                    result[key] = result[key].astype(value.dtype)
                result[key][rows] = value[rows]
    return result,tuple(mappings)


def iter_owned_group_targets(regional, *, calibration_seed, curve_seed,
                             area_draws, max_draws, n_positions=N_POS):
    """Fit owned targets with all nuisance members retained on their map.

For an inseparable fit the canonical anchor is evaluated even if it is owned
elsewhere, solely to define this map's shared group bound. Ownership-filtered
source mappings must be used when applying that result.
"""
    from .groups import detected_groups
    from .stage import run_group_target
    c = regional.stages[0].candidates
    identities=np.asarray([str(cat)+':'+str(int(row)).zfill(12)
                           for cat,row in zip(c.catalogue,c.source_row)])
    group_offset=0
    for i,(m,stage) in enumerate(zip(regional.maps,regional.stages)):
        groups=detected_groups(c.ra_deg,c.dec_deg,regional.detected,identities,beam_arcsec=m.beam_arcsec)
        for group_id,member_rows in enumerate(groups.members):
            members=np.asarray(member_rows)[np.argsort(identities[member_rows],kind='stable')]
            targets=np.flatnonzero(regional.source_map[members]==i)
            for target in targets:
                def run(t):
                    return run_group_target(m.opened,stage,group_id,members,int(t),m.exact_psf,
                        half_size=m.half_size,annulus_arcsec=m.annulus_arcsec,min_coverage=m.min_coverage,
                        calibration_seed=calibration_seed,curve_seed=curve_seed,area_draws=area_draws,
                        max_draws=max_draws,n_positions=n_positions,beam_arcsec=m.beam_arcsec)
                value=run(target)
                if not np.all(value.source_measurement.identifiable[0]):
                    if target!=0:
                        value=run(0)
                    yield i,replace(value,group_id=group_offset+group_id)
                    break
                yield i,replace(value,group_id=group_offset+group_id)
        group_offset+=len(groups.members)
