"""Bounded regional PACS measurement orchestration; no fabricated final rows.

Singleton fluxes/pixel variances are measured once. A fixed random stream is
reselected with the current detected-PACS mask until the entire detection set
is self-consistent. Cycles, invalid accepted fits and exhausted budgets raise.
The converged mask is then immutable for group fits and completeness.
"""
from dataclasses import dataclass
import time
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.spatial import cKDTree
from .constants import DETECTION_SNR,N_POS,P70_PARALLEL60_GROUP_RADIUS_ARCSEC
from .geometry import MapGrid,DetectedPacsNeighbours,sample_cells
from .photometry import prepare_xy
from .completeness import calibrate_cell_scatter


@dataclass(frozen=True)
class SingletonStage:
    candidates: object
    covered: np.ndarray
    cell_ids: np.ndarray
    flux_mjy: np.ndarray
    pixel_sigma_mjy: np.ndarray
    sigma_mjy: np.ndarray
    condition: np.ndarray
    detected: np.ndarray
    calibration_cell_ids: np.ndarray
    calibration_mean_mjy: np.ndarray
    calibration_scatter_mjy: np.ndarray
    calibration_pixel_sigma_rms_mjy: np.ndarray
    calibration_count: np.ndarray
    iterations: int
    seconds: float
    calibration_seed: int = None


def usable_xy(open_map,x,y,*,half_size,min_coverage):
    """Usable centre and complete stencil, independent of extracted flux."""
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    finite=np.isfinite(x)&np.isfinite(y)
    ix=np.floor(np.where(finite,x,0)+.5).astype(np.int64)
    iy=np.floor(np.where(finite,y,0)+.5).astype(np.int64)
    ny,nx=open_map.image.shape
    good=finite&(ix>=half_size)&(ix<nx-half_size)&(iy>=half_size)&(iy<ny-half_size)
    ix,iy=np.clip(ix,0,nx-1),np.clip(iy,0,ny-1)
    im,er,co=(v[iy,ix] for v in (open_map.image,open_map.error,open_map.coverage))
    return good&np.isfinite(im)&np.isfinite(er)&(er>0)&np.isfinite(co)&(co>=min_coverage)


def measure_xy(open_map,x,y,psf,*,half_size,annulus_arcsec,min_coverage,batch_size=200,return_gain=False):
    """Return physical paired flux, pixel sigma and conditioning in mJy."""
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=1 or x.shape!=y.shape or batch_size<1:
        raise ValueError('Matching 1D coordinates and positive batch size required')
    flux=np.empty(x.size);sigma=np.empty(x.size);condition=np.empty(x.size)
    gain=np.empty(x.size) if return_gain else None
    for start in range(0,x.size,batch_size):
        stop=min(start+batch_size,x.size)
        b=prepare_xy(open_map,x[start:stop],y[start:stop],psf,half_size=half_size,
                     annulus_arcsec=annulus_arcsec,min_coverage=min_coverage)
        if return_gain:
            from .extractor import prepare_fit,measure_sources
            ctx=prepare_fit(b.image,b.error,b.coverage,b.templates,b.annulus_mask,
                            min_coverage=min_coverage)
            z=measure_sources(b.image,ctx,b.templates,np.array([0,0]),cell_scatter_mjy=0.)
            functional=ctx.operator[:,:2].sum(axis=1)
            injection=b.templates.mean(axis=1).reshape(stop-start,-1)
            gain[start:stop]=np.einsum('bi,bi->b',functional,injection)
        else:
            z=b.measure(cell_scatter_mjy=0.)
        good=(b.covered & z.valid & z.identifiable[:,0] &
              np.isfinite(z.flux[:,0]) & np.isfinite(z.pixel_sigma[:,0]) &
              (z.pixel_sigma[:,0]>0))
        if not np.all(good):
            from .extractor import prepare_fit
            k=int(np.flatnonzero(~good)[0])
            diagnostic=prepare_fit(b.image[k:k+1],b.error[k:k+1],b.coverage[k:k+1],
                                   b.templates[k:k+1],b.annulus_mask[k:k+1],
                                   min_coverage=min_coverage)
            raise RuntimeError('Invalid accepted paired extraction; no silent trial rejection; '
                'x=%r y=%r covered=%r valid=%r identifiable=%r condition=%r pixel_sigma=%r '
                'rank=%r fit_pixels=%r group_identifiable=%r raw_pixel_variance=%r' %
                (x[start+k],y[start+k],b.covered[k],z.valid[k],z.identifiable[k,0],
                 z.condition[k],z.pixel_sigma[k,0],diagnostic.rank[0],
                 int(diagnostic.fit_mask[0].sum()),diagnostic.group_identifiable[0],
                 z.pixel_covariance[k,0,0]))
        flux[start:stop]=z.flux[:,0];sigma[start:stop]=z.pixel_sigma[:,0]
        condition[start:stop]=z.condition
    return (flux,sigma,condition,gain) if return_gain else (flux,sigma,condition)


def run_singleton_fixed_point(open_map,candidates,psf,*,half_size,annulus_arcsec,
                              min_coverage,seed,area_draws,max_draws,max_iterations,
                              max_seconds,batch_size=200,n_positions=N_POS,config=None,n_jobs=1,
                              beam_arcsec=P70_PARALLEL60_GROUP_RADIUS_ARCSEC):
    """Measure every supplied source and calibrate its cell to mask stability.

    Explicit budgets control termination, never science selection. Same cell ID
    and seed generate the same predeclared candidate stream at every iteration.
    Empty bootstrap mask is not frozen prematurely. Only changed detected-source
    neighbourhoods invalidate cached calibrations, an exact spatial dependency.
    No FITS data or noise model changes between iterations.
    """
    if max_iterations<1 or max_seconds<=0 or n_positions<2:
        raise ValueError('Positive iteration/time budgets and >=2 calibration samples required')
    started=time.monotonic()
    if n_jobs>1 and config is None:
        raise ValueError("Parallel calibration requires config for worker map initialization")
    run_key=str(time.time_ns())
    grid=MapGrid.from_wcs(open_map.wcs,open_map.image.shape)
    sky=SkyCoord(candidates.ra_deg*u.deg,candidates.dec_deg*u.deg,frame='icrs')
    x,y=open_map.wcs.world_to_pixel(sky)
    covered=usable_xy(open_map,x,y,half_size=half_size,min_coverage=min_coverage)
    ids=grid.cell_ids(x,y)
    cells=np.unique(ids[covered])
    rows=np.flatnonzero(covered)
    flux=np.full(x.size,np.nan);pixel=np.full(x.size,np.nan);condition=np.full(x.size,np.nan)
    flux[rows],pixel[rows],condition[rows]=measure_xy(
        open_map,x[rows],y[rows],psf,half_size=half_size,annulus_arcsec=annulus_arcsec,
        min_coverage=min_coverage,batch_size=batch_size)
    print('PACS first pass measured',rows.size,'of',x.size,'sources;',cells.size,'calibration cells',flush=True)
    detected=np.zeros(x.size,dtype=bool);history=[]
    mean=np.empty(cells.size);scatter=np.empty(cells.size);rms=np.empty(cells.size)
    count=np.empty(cells.size,dtype=np.int64);previous_neighbour_ids=[None]*cells.size
    cx0,cx1,cy0,cy1=grid.bounds(cells)
    centre=grid.wcs.pixel_to_world((cx0+cx1)/2,(cy0+cy1)/2).icrs
    # Conservative cell cone encloses corners plus exclusion radius, ensuring
    # all changed exclusions invalidate that cell's cached random selection.
    radius=np.zeros(cells.size)
    for xx,yy in ((cx0,cy0),(cx0,cy1),(cx1,cy0),(cx1,cy1)):
        radius=np.maximum(radius,centre.separation(grid.wcs.pixel_to_world(xx,yy).icrs).arcsec)
    radius+=beam_arcsec+1e-6
    centre_vectors=centre.cartesian.xyz.value.T
    sky_vectors=sky.cartesian.xyz.value.T
    chord=2*np.sin(np.deg2rad(radius/3600)/2)
    callback=lambda xx,yy:usable_xy(open_map,xx,yy,half_size=half_size,min_coverage=min_coverage)
    for iteration in range(1,max_iterations+1):
        if time.monotonic()-started>max_seconds:
            raise RuntimeError('Regional calibration time budget exhausted before convergence')
        neighbours=DetectedPacsNeighbours(candidates,detected)
        changed=0
        detected_rows=np.flatnonzero(detected)
        if detected_rows.size:
            local_lists=cKDTree(sky_vectors[detected_rows]).query_ball_point(centre_vectors,chord)
        else:
            local_lists=[[] for _ in cells]
        local_ids=[np.sort(detected_rows[np.asarray(v,dtype=np.int64)]) for v in local_lists]
        update=np.array([i for i in range(cells.size)
                         if previous_neighbour_ids[i] is None or
                         not np.array_equal(local_ids[i],previous_neighbour_ids[i])],dtype=np.int64)
        if n_jobs>1 and update.size:
            from .regional_workers import iterate_cell_calibrations
            stream=iterate_cell_calibrations(config,open_map.record,psf,cells[update],
                       candidates.ra_deg[detected],candidates.dec_deg[detected],
                       n_jobs=n_jobs,run_key=run_key,pass_key=iteration,seed=seed,
                       area_draws=area_draws,max_draws=max_draws,half_size=half_size,
                       annulus_arcsec=annulus_arcsec,min_coverage=min_coverage,
                       beam_arcsec=beam_arcsec,n_positions=n_positions,batch_size=batch_size,
                       deadline=started+max_seconds)
            for result in stream:
                where=np.searchsorted(cells,result.cell_ids)
                mean[where]=result.mean_mjy;scatter[where]=result.scatter_mjy
                rms[where]=result.pixel_sigma_rms_mjy;count[where]=result.n_positions
                for i in where:
                    previous_neighbour_ids[i]=local_ids[i].copy()
                changed+=where.size
                if time.monotonic()-started>max_seconds:
                    raise RuntimeError('Regional calibration time budget exhausted within parallel pass')
                if changed//100!=(changed-where.size)//100:
                    print('PACS calibration pass',iteration,'updated cells',changed,'of',update.size,
                          'seconds',time.monotonic()-started,flush=True)
        else:
            for i in update:
                cell=cells[i]
                if time.monotonic()-started>max_seconds:
                    raise RuntimeError('Regional calibration time budget exhausted within pass')
                selected=sample_cells(grid,np.asarray([cell]),neighbours,beam_arcsec,callback,
                                      seed=seed,area_draws=area_draws,max_draws=max_draws,
                                      n_positions=n_positions)
                f,p,_=measure_xy(open_map,selected.x.ravel(),selected.y.ravel(),psf,
                                half_size=half_size,annulus_arcsec=annulus_arcsec,
                                min_coverage=min_coverage,batch_size=batch_size)
                calibration=calibrate_cell_scatter(f[None,:],p[None,:],np.ones((1,f.size),bool))
                mean[i]=calibration.mean_mjy[0];scatter[i]=calibration.scatter_mjy[0]
                rms[i]=calibration.pixel_sigma_rms_mjy[0];count[i]=calibration.n_positions[0]
                previous_neighbour_ids[i]=local_ids[i].copy();changed+=1
                if changed%100==0:
                    print('PACS calibration pass',iteration,'updated cells',changed,'seconds',time.monotonic()-started,flush=True)
        sigma=np.full(x.size,np.nan)
        sigma[rows]=np.hypot(pixel[rows],scatter[np.searchsorted(cells,ids[rows])])
        new=covered&(flux>=DETECTION_SNR*sigma)
        print('PACS mask pass',iteration,'detected',np.count_nonzero(new),
              'changed',np.count_nonzero(new!=detected),'seconds',time.monotonic()-started,flush=True)
        if np.array_equal(new,detected):
            for value in (covered,ids,flux,pixel,sigma,condition,new,cells,mean,scatter,rms,count):
                value.flags.writeable=False
            return SingletonStage(candidates,covered,ids,flux,pixel,sigma,condition,new,cells,
                                  mean,scatter,rms,count,iteration,time.monotonic()-started,int(seed))
        packed=np.packbits(new).tobytes()
        if packed in history:
            raise RuntimeError('Regional detection/calibration cycle; no forced convergence')
        history.append(packed);detected=new
    raise RuntimeError('Regional calibration iteration budget exhausted before convergence')


@dataclass(frozen=True)
class GroupTargetStage:
    group_id: int
    members: np.ndarray
    target_source: int
    target_cell_id: int
    offsets_xy: np.ndarray
    calibration_mean_mjy: np.ndarray
    calibration_covariance_mjy2: np.ndarray
    source_measurement: object
    curve_x: np.ndarray
    curve_y: np.ndarray
    curve_measurement: object
    gain: np.ndarray
    area: object


def sample_group_translations(open_map,grid,cell_id,offsets_xy,neighbours,*,
                              seed,area_draws,max_draws,n_positions,
                              half_size,min_coverage,beam_arcsec):
    """Anchor uniformly in the target cell; all translated members eligible."""
    from .geometry import exclusion_area,SamplingBudgetError
    offsets=np.asarray(offsets_xy,float)
    if offsets.ndim!=2 or offsets.shape[1]!=2 or not np.all(np.isfinite(offsets)):
        raise ValueError('Finite member offsets (nsource,2) required')
    if not 1<=area_draws<=max_draws or not 2<=n_positions<=max_draws:
        raise ValueError('Invalid group sampling budgets')
    accepted_x=[];accepted_y=[];n_accepted=0;draw_start=0;area=None
    block=max(area_draws,2*n_positions)
    while draw_start<max_draws and n_accepted<n_positions:
        size=min(block,max_draws-draw_start)
        ax,ay=grid.sample_pixels(np.array([cell_id],dtype=np.int64),size,seed,
                                 draw_start=draw_start)
        x=ax.ravel()[:,None]+offsets[None,:,0]
        y=ay.ravel()[:,None]+offsets[None,:,1]
        covered=np.all(usable_xy(open_map,x,y,half_size=half_size,min_coverage=min_coverage),axis=1)
        sky=grid.wcs.pixel_to_world(x,y).icrs
        excluded=np.any(neighbours.excluded(sky.ra.deg,sky.dec.deg,beam_arcsec),axis=1)
        if draw_start==0:
            area=exclusion_area(covered[None,:area_draws],excluded[None,:area_draws])
        selected=np.flatnonzero(covered&~excluded)[:n_positions-n_accepted]
        accepted_x.append(x[selected]);accepted_y.append(y[selected]);n_accepted+=selected.size
        draw_start+=size
        # Bound allocation independently of the eventual explicit refill budget.
        block=min(block*2,8192)
    if n_accepted<n_positions:
        raise SamplingBudgetError(np.array([cell_id]),np.array([n_accepted]),area,None)
    return np.concatenate(accepted_x),np.concatenate(accepted_y),area


def run_group_target(open_map,singletons,group_id,members,target_source,psf,*,
                     half_size,annulus_arcsec,min_coverage,calibration_seed,
                     curve_seed,area_draws,max_draws,n_positions=N_POS,
                     beam_arcsec=P70_PARALLEL60_GROUP_RADIUS_ARCSEC):
    """Matched joint calibration, actual measurement and held-out target CDF.

    Every target is anchored in its own native cell. Full vector sample
    covariance is measured from independent rigid group translations. This
    keeps source correlations and the exact paired joint linear functional.
    The frozen singleton detection mask is never changed by the joint result.
    """
    from .photometry import measure_group_xy
    if calibration_seed==curve_seed:
        raise ValueError('Group calibration and held-out curve streams must differ')
    members=np.asarray(members,dtype=np.int64)
    if members.ndim!=1 or members.size<2 or np.unique(members).size!=members.size:
        raise ValueError('A complete unique nonsingleton group is required')
    if not 0<=target_source<members.size or not np.all(singletons.covered[members]):
        raise ValueError('Target index and all group members must be covered')
    c=singletons.candidates
    sky=SkyCoord(c.ra_deg[members]*u.deg,c.dec_deg[members]*u.deg,frame='icrs')
    sx,sy=open_map.wcs.world_to_pixel(sky)
    offsets=np.column_stack((sx-sx[target_source],sy-sy[target_source]))
    cell=int(singletons.cell_ids[members[target_source]])
    grid=MapGrid.from_wcs(open_map.wcs,open_map.image.shape)
    neighbours=DetectedPacsNeighbours(c,singletons.detected)
    kwargs=dict(area_draws=area_draws,max_draws=max_draws,n_positions=n_positions,
                half_size=half_size,min_coverage=min_coverage,beam_arcsec=beam_arcsec)
    x,y,_=sample_group_translations(open_map,grid,cell,offsets,neighbours,
                                    seed=calibration_seed,**kwargs)
    zero=np.zeros((members.size,members.size))
    calibration,_,_=measure_group_xy(open_map,x,y,psf,half_size=half_size,
                        annulus_arcsec=annulus_arcsec,min_coverage=min_coverage,
                        cell_covariance_mjy2=zero)
    if not np.all(np.isfinite(calibration.flux)):
        raise RuntimeError('Invalid accepted group calibration flux')
    background=np.cov(calibration.flux,rowvar=False,ddof=1)
    mean=calibration.flux.mean(axis=0)
    actual,_,_=measure_group_xy(open_map,sx[None,:],sy[None,:],psf,half_size=half_size,
                        annulus_arcsec=annulus_arcsec,min_coverage=min_coverage,
                        cell_covariance_mjy2=background)
    if actual.identifiable[0,target_source] and not np.all(calibration.identifiable[:,target_source]):
        raise RuntimeError('Accepted calibration translation has an unidentifiable target')
    x,y,area=sample_group_translations(open_map,grid,cell,offsets,neighbours,
                                       seed=curve_seed,**kwargs)
    curve,batch,context=measure_group_xy(open_map,x,y,psf,half_size=half_size,
                        annulus_arcsec=annulus_arcsec,min_coverage=min_coverage,
                        cell_covariance_mjy2=background)
    # Equal mixture merely probes the response of the summed source functional;
    # both scan columns are in the fit, so any unit-sum mixture has the same gain.
    functional=context.operator[:,2*target_source:2*target_source+2].sum(axis=1)
    injection=batch.templates[:,2*target_source:2*target_source+2].mean(axis=1).reshape(n_positions,-1)
    gain=np.einsum('bi,bi->b',functional,injection)
    if actual.identifiable[0,target_source] and (not np.all(curve.identifiable[:,target_source]) or
            not np.all(np.isfinite(gain)&(gain>0))):
        raise RuntimeError('Accepted translated group has invalid target response')
    return GroupTargetStage(int(group_id),members.copy(),int(target_source),cell,offsets,
                            mean,background,actual,x,y,curve,gain,area)


def iter_group_targets(open_map,singletons,psf,*,half_size,annulus_arcsec,
                       min_coverage,calibration_seed,curve_seed,area_draws,
                       max_draws,n_positions=N_POS,
                       beam_arcsec=P70_PARALLEL60_GROUP_RADIUS_ARCSEC):
    """Yield each canonical group's matched target measurements/curves.

    Canonical identity is catalogue filename plus original row, not NAME.
    Member zero (the smallest identity) supplies the shared inseparable bound.
    No group result feeds back into the frozen singleton exclusion mask.
    """
    from .groups import detected_groups
    c=singletons.candidates
    ids=np.asarray([str(cat)+':'+str(int(row)).zfill(12)
                    for cat,row in zip(c.catalogue,c.source_row)])
    groups=detected_groups(c.ra_deg,c.dec_deg,singletons.detected,ids,
                           beam_arcsec=beam_arcsec)
    for group_id,member_rows in enumerate(groups.members):
        members=np.asarray(member_rows)[np.argsort(ids[member_rows],kind='stable')]
        for target in range(members.size):
            result=run_group_target(open_map,singletons,group_id,members,target,psf,
                         half_size=half_size,annulus_arcsec=annulus_arcsec,
                         min_coverage=min_coverage,calibration_seed=calibration_seed,
                         curve_seed=curve_seed,area_draws=area_draws,max_draws=max_draws,
                         n_positions=n_positions,beam_arcsec=beam_arcsec)
            print('PACS group',group_id,'members',members.size,'target',target,
                  'actual sigma',result.source_measurement.sigma[0,target],
                  'random sigma range',np.nanmin(result.curve_measurement.sigma[:,target]),
                  np.nanmax(result.curve_measurement.sigma[:,target]),flush=True)
            yield result
            if not np.all(result.source_measurement.identifiable[0]):
                # One full-covariance, canonical-anchor total bound is shared
                # by all members; there are no likelihood CDFs for this band.
                break
