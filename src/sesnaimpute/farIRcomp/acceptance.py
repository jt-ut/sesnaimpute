"""Bounded independent real-map injection checks; no product writes."""
import numpy as np


def verify_cells(open_map, cached_psf, exact_psf, neighbours, cell_ids,
                 cell_scatter_mjy, *, seed, production_seeds, flux_grid_mjy,
                 beam_arcsec, min_coverage, half_size=12, annulus_arcsec=(25.,35.),
                 area_draws=1024, max_draws=8192,
                 allow_curve_seed=False, stored_floor_counts=None):
    """Use 200 heldout positions per cell and exact GL8 injected profiles.

    The caller supplies the frozen product's detection tree and cell scatter.
    Extraction uses the production response cache; injected profiles are evaluated
    independently with the published paired GL8 model. No scatter is re-estimated,
    no failed fits are discarded, and the three mixtures are injection hypotheses
    only. This diagnostic neither changes nor certifies the scientific flux grid.
    """
    from .geometry import MapGrid, sample_cells
    from .stage import usable_xy
    from .photometry import prepare_xy
    from .extractor import prepare_fit
    from .completeness import verify_injection_batch
    calibration_seed,curve_seed=map(int,production_seeds)
    if int(seed)==calibration_seed or (int(seed)==curve_seed and not allow_curve_seed):
        raise ValueError('Reject calibration seed; curve seed needs explicit stored-floor replay')
    if stored_floor_counts is not None and int(seed)!=curve_seed:
        raise ValueError('Stored floor comparison requires identical production curve positions')
    ids=np.asarray(cell_ids,dtype=np.int64)
    scatter=np.asarray(cell_scatter_mjy,float)
    if ids.ndim!=1 or np.unique(ids).size!=ids.size or scatter.shape!=ids.shape:
        raise ValueError('Unique cells and one frozen scatter per cell required')
    if np.any(~np.isfinite(scatter)|(scatter<0)):
        raise ValueError('Invalid frozen cell scatter')
    floors=None if stored_floor_counts is None else np.asarray(stored_floor_counts)
    if floors is not None and floors.shape!=ids.shape:
        raise ValueError("One stored floor count per selected cell required")
    grid=MapGrid.from_wcs(open_map.wcs,open_map.image.shape)
    def usable(x,y):
        return usable_xy(open_map,x,y,half_size=half_size,min_coverage=min_coverage)
    summaries=[]
    checked_grid=np.asarray(flux_grid_mjy,float)
    for cell_index,(cell,noise) in enumerate(zip(ids,scatter)):
        selected=sample_cells(grid,np.array([cell]),neighbours,beam_arcsec,usable,
            seed=seed,area_draws=area_draws,max_draws=max_draws,n_positions=200)
        x,y=selected.x.ravel(),selected.y.ravel()
        cut=prepare_xy(open_map,x,y,cached_psf,half_size=half_size,
            annulus_arcsec=annulus_arcsec,min_coverage=min_coverage)
        d=np.arange(-half_size,half_size+1)
        yy,xx=np.meshgrid(d,d,indexing='ij')
        px=np.floor(x+.5).astype(np.int64)[:,None,None]+xx
        py=np.floor(y+.5).astype(np.int64)[:,None,None]+yy
        sky=open_map.wcs.pixel_to_world(x,y)[:,None,None]
        exact=exact_psf(open_map.wcs,sky,px,py,8)
        context=prepare_fit(cut.image,cut.error,cut.coverage,cut.templates,
            cut.annulus_mask,min_coverage=min_coverage)
        operator=context.operator[:,:2].sum(axis=1)
        for weight in (0.,.5,1.):
            profile=weight*exact[:,0]+(1-weight)*exact[:,1]
            gain=np.einsum('bp,bp->b',operator,profile.reshape(200,-1))
            result=verify_injection_batch(cut.image,cut.error,cut.coverage,
                cut.templates,cut.annulus_mask,min_coverage=min_coverage,
                cell_shape=(1,200),flux_grid_mjy=flux_grid_mjy,
                component_source=np.array([0,0]),cell_scatter_mjy=noise,
                injection_template=profile)
            rebuilt_floor=int(round(float(result['samples'].floor_fraction[0])*200))
            if floors is not None and rebuilt_floor!=int(floors[cell_index]):
                raise AssertionError('Rebuilt zero-injection floor differs from stored count')
            summary=dict(cell_id=int(cell),injection_weight=weight,
                rebuilt_floor_count=rebuilt_floor,
                tested_flux_planes=checked_grid.copy(),
                n_positions=200,floor=float(result['samples'].floor_fraction[0]),
                max_gain_error=float(np.max(np.abs(gain-1))),
                max_sigma_relative_error=result['max_sigma_relative_error'],
                max_flux_absolute_error_mjy=result['max_flux_absolute_error_mjy'],
                analytic_counts=result['analytic_counts'],
                rebuilt_counts=result['injected_counts'])
            summaries.append(summary)
            print('PACS independent injection '+str(summary),flush=True)
    return summaries
